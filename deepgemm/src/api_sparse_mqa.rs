//! SM100 sparse MQA logits — the DSA top-k indexer scoring path
//! (port of upstream `sm100_sparse_mqa_logits` + its metadata scheduler).
//!
//! Two kernels, launched back-to-back on the same stream (PDL):
//!
//! 1. [`sparse_mqa_metadata`] — merges each Q-token pair's sorted top-k
//!    selected-block lists, cuts them into KV splits, resolves physical
//!    block locations, and balances an SM schedule of `(q block, split
//!    range)` entries.
//! 2. [`sparse_mqa_logits_contiguous`] / [`sparse_mqa_logits_paged`] — the
//!    persistent scoring kernel: gathers exactly the selected blocks,
//!    runs block-scaled `tcgen05.mma` over them, and scatter-stores the
//!    weighted-ReLU logits into each token's *compressed* row (column =
//!    selection slot * `sparse_block_kv` + token-in-block).
//!
//! Buffer sizing helpers ([`sparse_mqa_metadata_bytes`],
//! [`sparse_mqa_workspace_bytes`], [`SparseMqaConfig::logits_smem_bytes`])
//! mirror the device structs byte-for-byte; every JIT wrapper embeds a
//! `static_assert(sizeof(...))` so any layout drift fails the compile.

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, Operand};

// ---------------------------------------------------------------------------
// Fixed layout constants (layout/sparse_mqa_logits.cuh)
// ---------------------------------------------------------------------------
pub const SPARSE_MQA_HEAD_DIM: u32 = 128;
pub const SPARSE_MQA_MAX_HEADS: u32 = 32;
pub const SPARSE_MQA_BLOCK_Q: u32 = 2;
pub const SPARSE_MQA_MAX_SPARSE_BLOCKS: u32 = 4096;
pub const SPARSE_MQA_METADATA_THREADS: u32 = 256;
pub const SPARSE_MQA_SPLITS_PER_ENTRY: u32 = 8;
/// SM100A max dynamic shared memory (B200), upstream `SM100ArchSpec`.
const SMEM_CAPACITY: u32 = 232448;
/// `sizeof(SparseWorkspaceState)`: three counters on separate 128B lines.
const WORKSPACE_STATE_BYTES: u64 = 384;

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}
fn align_to(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}

// ---------------------------------------------------------------------------
// Tile choice — port of csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp
// ---------------------------------------------------------------------------

/// Launch configuration of the sparse-MQA scoring kernel (upstream constants:
/// `kNumQStages = 2`, `kNumTmemStages = 5`, `split_kv = 640/512` for
/// MXFP4/MXFP8, `num_kv_stages = 5/3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMqaConfig {
    /// Q heads (the indexer's MQA head count); multiple of 8, <= 32.
    /// (The kernel needs %4; the weights TMA needs 16B rows => %8.)
    pub num_heads: u32,
    /// Tokens per selected KV block: 8 or 16.
    pub sparse_block_kv: u32,
    /// KV tokens per split == `num_math_warpgroups * 128` (512 FP8 / 640 FP4).
    pub split_kv: u32,
    pub num_math_warpgroups: u32,
    pub q_stages: u32,
    pub kv_stages: u32,
    pub tmem_stages: u32,
    pub is_fp4: bool,
    /// Contiguous mode only: per-token KV spans may start mid-block.
    pub use_unaligned_ks: bool,
    /// 0 for the contiguous (flat-cache) variant.
    pub page_kv: u32,
    pub num_sms: u32,
}

impl SparseMqaConfig {
    /// Upstream tile heuristic: the only free knob is the dtype (FP4 needs a
    /// wider split to feed the same 128-row MMA tiles at half the bytes).
    pub fn new(
        num_heads: u32,
        dtype: Dtype,
        sparse_block_kv: u32,
        use_unaligned_ks: bool,
        page_kv: Option<u32>,
        num_sms: u32,
    ) -> DgResult<Self> {
        if !matches!(dtype, Dtype::Fp4 | Dtype::Fp8) {
            return Err(DgError::InvalidArg(
                "sparse MQA needs MXFP8 or MXFP4".into(),
            ));
        }
        if num_heads % 8 != 0 || num_heads > SPARSE_MQA_MAX_HEADS || num_heads == 0 {
            return Err(DgError::InvalidArg(
                "num_heads must be a multiple of 8 (<= 32): the weights TMA needs 16B rows".into(),
            ));
        }
        if sparse_block_kv != 8 && sparse_block_kv != 16 {
            return Err(DgError::InvalidArg(
                "sparse_block_kv must be 8 or 16".into(),
            ));
        }
        let is_fp4 = dtype == Dtype::Fp4;
        let split_kv = if is_fp4 { 640 } else { 512 };
        let cfg = SparseMqaConfig {
            num_heads,
            sparse_block_kv,
            split_kv,
            num_math_warpgroups: split_kv / 128,
            q_stages: 2,
            kv_stages: if is_fp4 { 5 } else { 3 },
            tmem_stages: 5,
            is_fp4,
            use_unaligned_ks,
            page_kv: page_kv.unwrap_or(0),
            num_sms,
        };
        if let Some(pk) = page_kv {
            if pk % sparse_block_kv != 0 {
                return Err(DgError::InvalidArg(
                    "page_kv must be a multiple of sparse_block_kv (blocks must not cross pages)"
                        .into(),
                ));
            }
        }
        let smem = cfg.logits_smem_bytes();
        if smem > SMEM_CAPACITY {
            return Err(DgError::InvalidArg(format!(
                "sparse MQA smem {smem} exceeds SM100 capacity {SMEM_CAPACITY}"
            )));
        }
        Ok(cfg)
    }

    /// CTA size of the scoring kernel:
    /// `(math_wgs + 1 control wg + math_wgs/4 copy wgs) * 128`.
    pub fn num_threads(&self) -> u32 {
        (self.num_math_warpgroups + 1 + self.num_math_warpgroups / 4) * 128
    }

    /// Dynamic shared memory of the scoring kernel — a byte-exact mirror of
    /// `dg::SparseLogitsSmem` (see the layout table in the kernel file).
    /// The JIT wrapper `static_assert`s this number against `sizeof`.
    pub fn logits_smem_bytes(&self) -> u32 {
        let pack = if self.is_fp4 { 2 } else { 1 };
        let bytes_per_token = SPARSE_MQA_HEAD_DIM / pack;
        // Largest member alignment: the swizzle atom of the Q/KV tiles.
        let swizzle_align = 8 * SPARSE_MQA_HEAD_DIM / pack;
        let q_rows = SPARSE_MQA_BLOCK_Q * SPARSE_MQA_MAX_HEADS;

        let q = self.q_stages * q_rows * bytes_per_token;
        let kv_off = align_to(q, swizzle_align);
        let kv = self.kv_stages * self.split_kv * bytes_per_token;
        let sfq_off = align_to(kv_off + kv, 128);
        let sfq = self.q_stages * 128 * 4; // kNumSFQ = align(2*32, 128) words
        let sfkv_off = align_to(sfq_off + sfq, 128);
        let sfkv = self.kv_stages * self.split_kv * 4;
        let w_off = align_to(sfkv_off + sfkv, 128);
        let w = self.q_stages * q_rows * 2; // bf16
        let kvi_off = align_to(w_off + w, 16);
        let kvi = self.kv_stages * (self.split_kv / self.sparse_block_kv) * 8;
        let qb_off = align_to(kvi_off + kvi, 16);
        let qb = self.q_stages * 16; // alignas(16) QBlock (3 x u32 + pad)
        let hdr_off = align_to(qb_off + qb, 16);
        let hdr = self.kv_stages * 16;
        let bar_off = align_to(hdr_off + hdr, 8);
        let bar = (3 * self.q_stages + 4 * self.kv_stages + 2 * self.tmem_stages) * 8;
        align_to(bar_off + bar + 4, swizzle_align)
    }
}

/// Dynamic shared memory of the metadata kernel (upstream
/// `launch_sm100_sparse_mqa_logits_metadata` formula; the wrapper
/// `static_assert`s the 128-rounded `sizeof` equals this).
pub fn sparse_mqa_metadata_smem_bytes(
    num_max_sparse_blocks: u32,
    split_kv: u32,
    sparse_block_kv: u32,
) -> u32 {
    let warps = SPARSE_MQA_METADATA_THREADS / 32;
    let blocks_per_split = split_kv / sparse_block_kv;
    let max_merged = SPARSE_MQA_BLOCK_Q * num_max_sparse_blocks;
    let max_splits = ceil_div(max_merged, blocks_per_split);
    let mut b = (SPARSE_MQA_BLOCK_Q * num_max_sparse_blocks + max_merged + max_splits) * 4;
    b += warps * 4; // warp_sums
    b += (1 + warps * (SPARSE_MQA_SPLITS_PER_ENTRY + 1)) * 4; // num_waves + histograms
    b = align_to(b, 16);
    b += align_to((3 + SPARSE_MQA_BLOCK_Q) * 4, 16); // alignas(16) q_block
    b += 4; // is_last_cta
    align_to(b, 128)
}

/// Bytes of the metadata output buffer:
/// `MetadataHeader | KVSplit[]... | ScheduleEntry[]` (the schedule starts
/// right after the last split, rounded to whole entries per SM).
#[allow(clippy::too_many_arguments)]
pub fn sparse_mqa_metadata_bytes(
    num_q_tokens: u32,
    num_max_sparse_blocks: u32,
    sparse_block_kv: u32,
    split_kv: u32,
    is_paged: bool,
    num_sms: u32,
) -> u64 {
    let blocks_per_split = split_kv / sparse_block_kv;
    let split_bytes = 16 + blocks_per_split as u64 * 8;
    let max_merged = SPARSE_MQA_BLOCK_Q * num_max_sparse_blocks;
    let max_kv_splits = if is_paged {
        num_q_tokens as u64 * ceil_div(num_max_sparse_blocks, blocks_per_split) as u64
    } else {
        ceil_div(num_q_tokens, SPARSE_MQA_BLOCK_Q) as u64
            * ceil_div(max_merged, blocks_per_split) as u64
    };
    let max_schedule_entries = align_to(max_kv_splits as u32, num_sms) as u64;
    16 + max_kv_splits * split_bytes + max_schedule_entries * 16
}

/// Bytes of the metadata kernel's workspace:
/// `SparseWorkspaceState (384) | QBlockInfo[num_q_tokens]`.
/// Must be ZERO before the first launch; the kernel self-resets the counters.
pub fn sparse_mqa_workspace_bytes(num_q_tokens: u32) -> u64 {
    WORKSPACE_STATE_BYTES + num_q_tokens as u64 * 8
}

/// Allocate a correctly-sized, zeroed workspace for
/// [`sparse_mqa_metadata`]. (After a completed launch the kernel leaves the
/// counters zeroed, so the buffer can be reused directly.)
pub fn sparse_mqa_alloc_workspace(dev: &Device, num_q_tokens: u32) -> DgResult<DevBuffer> {
    DevBuffer::alloc_zeros(dev, sparse_mqa_workspace_bytes(num_q_tokens) as usize)
}

// ---------------------------------------------------------------------------
// TMA maps specific to the sparse variant (Q/KV reuse crate::tma helpers)
// ---------------------------------------------------------------------------

/// SF-Q map: int32 `[num_heads (inner), num_q_tokens]`, box `[heads, 2]`
/// (both tokens of a Q pair in one copy, head-contiguous).
fn make_tma_sparse_sf_q(
    dev: &Device,
    buf: &DevBuffer,
    num_heads: u32,
    num_tokens: u32,
) -> DgResult<sys::TensorMap> {
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        sys::tm_dtype_int32(),
        2,
        buf.ptr as *mut _,
        &[num_heads as u64, num_tokens as u64],
        &[num_heads as u64 * 4],
        &[num_heads, SPARSE_MQA_BLOCK_Q],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(0),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// SF-KV map (contiguous mode): int32 `[num_kv_tokens (inner), 1]`,
/// box `[128, 1]` — one packed UE8M0 word per KV token, 128 per TMA.
/// (Upstream passes outer stride 0; we pass the equivalent contiguous
/// stride since the outer dim has extent 1 either way.)
fn make_tma_sparse_sf_kv(
    dev: &Device,
    buf: &DevBuffer,
    num_kv_tokens: u32,
) -> DgResult<sys::TensorMap> {
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        sys::tm_dtype_int32(),
        2,
        buf.ptr as *mut _,
        &[num_kv_tokens as u64, 1],
        &[num_kv_tokens as u64 * 4],
        &[128, 1],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(0),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// Weights map: bf16 `[num_heads (inner), num_q_tokens]`,
/// box `[align(heads, 8), 2]` — the aligned inner box lets small head
/// counts still issue one copy; OOB columns zero-fill.
fn make_tma_sparse_weights(
    dev: &Device,
    buf: &DevBuffer,
    num_heads: u32,
    num_tokens: u32,
) -> DgResult<sys::TensorMap> {
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        sys::tm_dtype_bfloat16(),
        2,
        buf.ptr as *mut _,
        &[num_heads as u64, num_tokens as u64],
        &[num_heads as u64 * 2],
        &[align_to(num_heads, 8), SPARSE_MQA_BLOCK_Q],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(0),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

// ---------------------------------------------------------------------------
// JIT wrapper bodies (shared by the launchers and the offline compile checks)
// ---------------------------------------------------------------------------

/// Wrapper body of the metadata kernel (also used by the CPU compile checks).
#[allow(clippy::too_many_arguments)]
pub fn sparse_mqa_metadata_body(
    is_paged: bool,
    use_unaligned_ks: bool,
    split_kv: u32,
    sparse_block_kv: u32,
    num_max_sparse_blocks: u32,
    page_kv: u32,
    num_sms: u32,
) -> String {
    let smem = sparse_mqa_metadata_smem_bytes(num_max_sparse_blocks, split_kv, sparse_block_kv);
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned num_q_tokens, unsigned num_kv_tokens,
    const unsigned* cu_seq_len_k_start, const unsigned* cu_seq_len_k_end,
    const unsigned* context_lens, const unsigned* block_table, unsigned block_table_stride,
    const unsigned* indices,
    const unsigned* sparse_kv_block_indices, unsigned char* metadata, unsigned char* workspace) {{
    static_assert((sizeof(dg::SparseMetaSmem<{bq}, {nmax}, {bps}, {spe}, {threads}>) + 127) / 128 * 128 == {smem},
                  "sparse-MQA metadata smem layout drift");
    dg::sparse_mqa_metadata_impl<{paged}, {uks}, {bq}, {skv}, {sbkv}, {nmax}, {page_kv}, {spe}, {sms}, {threads}>(
        num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, context_lens,
        block_table, block_table_stride, indices, sparse_kv_block_indices, metadata, workspace);
}}"#,
        bq = SPARSE_MQA_BLOCK_Q,
        nmax = num_max_sparse_blocks,
        bps = split_kv / sparse_block_kv,
        spe = SPARSE_MQA_SPLITS_PER_ENTRY,
        threads = SPARSE_MQA_METADATA_THREADS,
        smem = smem,
        paged = is_paged as u32,
        uks = use_unaligned_ks as u32,
        skv = split_kv,
        sbkv = sparse_block_kv,
        page_kv = page_kv,
        sms = num_sms,
    )
}

/// Wrapper body of the contiguous scoring kernel.
pub fn sparse_mqa_logits_body(cfg: &SparseMqaConfig) -> String {
    let smem = cfg.logits_smem_bytes();
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned logits_stride, unsigned num_kv_tokens, unsigned short* logits,
    const unsigned char* kv, const unsigned int* sf_kv, const unsigned char* metadata,
    const __grid_constant__ dg::TmaMap tma_q,
    const __grid_constant__ dg::TmaMap tma_sf_q,
    const __grid_constant__ dg::TmaMap tma_w,
    const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_sf_kv) {{
    static_assert(sizeof(dg::SparseLogitsSmem<{bq}, {sbkv}, {skv}, {qs}, {kvs}, {ts}, {fp4}>) == {smem},
                  "sparse-MQA logits smem layout drift");
    dg::sparse_mqa_logits_impl<{heads}, {sbkv}, {qs}, {kvs}, {ts}, {mwg}, {sms}, {bq}, {uks}, {fp4}>(
        logits_stride, num_kv_tokens, logits, kv, sf_kv, metadata,
        tma_q, tma_sf_q, tma_w, tma_kv, tma_sf_kv);
}}"#,
        bq = SPARSE_MQA_BLOCK_Q,
        sbkv = cfg.sparse_block_kv,
        skv = cfg.split_kv,
        qs = cfg.q_stages,
        kvs = cfg.kv_stages,
        ts = cfg.tmem_stages,
        fp4 = cfg.is_fp4 as u32,
        smem = smem,
        heads = cfg.num_heads,
        mwg = cfg.num_math_warpgroups,
        sms = cfg.num_sms,
        uks = cfg.use_unaligned_ks as u32,
    )
}

/// Wrapper body of the paged scoring kernel.
pub fn sparse_mqa_logits_paged_body(cfg: &SparseMqaConfig) -> String {
    let smem = cfg.logits_smem_bytes();
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned logits_stride, unsigned kv_page_stride_bytes, unsigned short* logits,
    const unsigned char* fused_kv_cache, const unsigned char* metadata,
    const __grid_constant__ dg::TmaMap tma_q,
    const __grid_constant__ dg::TmaMap tma_sf_q,
    const __grid_constant__ dg::TmaMap tma_w) {{
    static_assert(sizeof(dg::SparseLogitsSmem<{bq}, {sbkv}, {skv}, {qs}, {kvs}, {ts}, {fp4}>) == {smem},
                  "sparse-MQA logits smem layout drift");
    dg::sparse_mqa_logits_paged_impl<{heads}, {page_kv}, {sbkv}, {qs}, {kvs}, {ts}, {mwg}, {sms}, {bq}, {fp4}>(
        logits_stride, kv_page_stride_bytes, logits, fused_kv_cache, metadata,
        tma_q, tma_sf_q, tma_w);
}}"#,
        bq = SPARSE_MQA_BLOCK_Q,
        sbkv = cfg.sparse_block_kv,
        skv = cfg.split_kv,
        qs = cfg.q_stages,
        kvs = cfg.kv_stages,
        ts = cfg.tmem_stages,
        fp4 = cfg.is_fp4 as u32,
        smem = smem,
        heads = cfg.num_heads,
        page_kv = cfg.page_kv,
        mwg = cfg.num_math_warpgroups,
        sms = cfg.num_sms,
    )
}

fn opt_ptr(b: Option<&DevBuffer>) -> sys::DevicePtr {
    b.map(|b| b.ptr).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Launcher: metadata scheduler kernel
// ---------------------------------------------------------------------------

/// Run the sparse-MQA metadata/scheduler kernel.
///
/// Buffers (upstream `sm100_sparse_mqa_logits_metadata`):
/// * `sparse_kv_block_indices`: int32 `[num_q_tokens, num_max_sparse_blocks]`
///   — each row the *sorted* logical ids of that token's top-k selected KV
///   blocks (block ids for aligned contiguous mode; token ids for
///   `use_unaligned_ks`). Only the first `ceil(kv_span / sparse_block_kv)`
///   entries per row are read.
/// * `cu_seq_len_k_start` / `cu_seq_len_k_end`: per-token KV span (contiguous
///   mode). `context_lens` / `indices` / `block_table` (paged mode).
/// * `metadata`: [`sparse_mqa_metadata_bytes`] bytes (output).
/// * `workspace`: [`sparse_mqa_workspace_bytes`] bytes, ZERO on first use
///   (see [`sparse_mqa_alloc_workspace`]; the kernel self-resets it).
#[allow(clippy::too_many_arguments)]
pub fn sparse_mqa_metadata(
    dev: &Device,
    stream: &DevStream,
    dtype: Dtype,
    is_paged: bool,
    use_unaligned_ks: bool,
    num_q_tokens: u32,
    num_kv_tokens: u32,
    sparse_block_kv: u32,
    num_max_sparse_blocks: u32,
    page_kv: u32,
    cu_seq_len_k_start: Option<&DevBuffer>,
    cu_seq_len_k_end: Option<&DevBuffer>,
    context_lens: Option<&DevBuffer>,
    block_table: Option<&DevBuffer>,
    block_table_stride: u32,
    indices: Option<&DevBuffer>,
    sparse_kv_block_indices: &DevBuffer,
    metadata: &mut DevBuffer,
    workspace: &mut DevBuffer,
) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm100) {
        return Err(DgError::Unsupported(
            "the tcgen05 kernels require SM100 (Blackwell)".into(),
        ));
    }
    if is_paged && use_unaligned_ks {
        return Err(DgError::InvalidArg(
            "paged sparse MQA does not use ks".into(),
        ));
    }
    if !matches!(dtype, Dtype::Fp4 | Dtype::Fp8) {
        return Err(DgError::InvalidArg(
            "sparse MQA needs MXFP8 or MXFP4".into(),
        ));
    }
    if sparse_block_kv != 8 && sparse_block_kv != 16 {
        return Err(DgError::InvalidArg(
            "sparse_block_kv must be 8 or 16".into(),
        ));
    }
    if num_max_sparse_blocks == 0
        || num_max_sparse_blocks % 4 != 0
        || num_max_sparse_blocks > SPARSE_MQA_MAX_SPARSE_BLOCKS
    {
        return Err(DgError::InvalidArg(
            "num_max_sparse_blocks must be a positive multiple of 4 (<= 4096)".into(),
        ));
    }
    let split_kv = if dtype == Dtype::Fp4 { 640 } else { 512 };

    let body = sparse_mqa_metadata_body(
        is_paged,
        use_unaligned_ks,
        split_kv,
        sparse_block_kv,
        num_max_sparse_blocks,
        if is_paged { page_kv } else { 0 },
        dev.num_sms,
    );
    let sig = format!(
        "sparse_meta_{paged}_{uks}_{skv}_{sbkv}_{nmax}_{page}_{sms}",
        paged = is_paged as u32,
        uks = use_unaligned_ks as u32,
        skv = split_kv,
        sbkv = sparse_block_kv,
        nmax = num_max_sparse_blocks,
        page = if is_paged { page_kv } else { 0 },
        sms = dev.num_sms,
    );
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::SPARSE_MQA,
        "sparse_mqa_metadata",
        &sig,
        &body,
    )?;

    // Upstream grid: at most 4 waves of CTAs, one Q block per CTA per step.
    let num_q_blocks = if is_paged {
        num_q_tokens
    } else {
        ceil_div(num_q_tokens, SPARSE_MQA_BLOCK_Q)
    };
    let num_ctas = num_q_blocks.min(dev.num_sms * 4).max(1);
    let smem = sparse_mqa_metadata_smem_bytes(num_max_sparse_blocks, split_kv, sparse_block_kv);

    let args = Args::new()
        .u32(num_q_tokens)
        .u32(num_kv_tokens)
        .devptr(opt_ptr(cu_seq_len_k_start))
        .devptr(opt_ptr(cu_seq_len_k_end))
        .devptr(opt_ptr(context_lens))
        .devptr(opt_ptr(block_table))
        .u32(block_table_stride)
        .devptr(opt_ptr(indices))
        .devptr(sparse_kv_block_indices.ptr)
        .devptr(metadata.ptr)
        .devptr(workspace.ptr);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (num_ctas, 1, 1),
            block: (SPARSE_MQA_METADATA_THREADS, 1, 1),
            smem,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

// ---------------------------------------------------------------------------
// Launcher: scoring kernel (contiguous / paged)
// ---------------------------------------------------------------------------

/// Validate the shared operand shapes and build the tile config.
fn check_logits_common(
    dev: &Device,
    q: &Operand,
    num_heads: u32,
    sparse_block_kv: u32,
) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm100) {
        return Err(DgError::Unsupported(
            "the tcgen05 kernels require SM100 (Blackwell)".into(),
        ));
    }
    if !matches!(q.dtype, Dtype::Fp4 | Dtype::Fp8) {
        return Err(DgError::InvalidArg(
            "sparse MQA needs MXFP8 or MXFP4 Q/KV".into(),
        ));
    }
    if q.k != SPARSE_MQA_HEAD_DIM {
        return Err(DgError::InvalidArg(
            "sparse MQA requires head_dim == 128 (kSparseHeadDim)".into(),
        ));
    }
    if q.outer_stride != SPARSE_MQA_HEAD_DIM {
        return Err(DgError::InvalidArg(
            "sparse MQA requires a contiguous Q (outer stride == head_dim)".into(),
        ));
    }
    if sparse_block_kv != 8 && sparse_block_kv != 16 {
        return Err(DgError::InvalidArg(
            "sparse_block_kv must be 8 or 16".into(),
        ));
    }
    if num_heads == 0 || num_heads % 8 != 0 || num_heads > SPARSE_MQA_MAX_HEADS {
        return Err(DgError::InvalidArg(
            "num_heads must be a multiple of 8 (<= 32)".into(),
        ));
    }
    Ok(())
}

fn check_q_rows(q: &Operand, num_tokens: u32, num_heads: u32) -> DgResult<()> {
    if q.rows != num_tokens * num_heads {
        return Err(DgError::InvalidArg(
            "sparse MQA expects q.rows == num_tokens * num_heads".into(),
        ));
    }
    Ok(())
}

/// Score the selected (top-k) KV blocks against Q with a flat KV cache.
///
/// * `q`: `[num_tokens * num_heads, 128]` (K-major, FP8 or packed FP4);
/// * `q_sf`: int32 `[num_tokens, num_heads]` — one packed UE8M0 word per
///   (token, head) covering the 4 x 32-K blocks of head_dim;
/// * `kv`: `[num_kv_tokens, 128]`; `kv_sf`: int32 `[num_kv_tokens]`;
/// * `weights`: bf16 `[num_tokens, num_heads]` (row stride = num_heads);
/// * `metadata`: filled by [`sparse_mqa_metadata`];
/// * `logits`: bf16 `[num_tokens, logits_stride]` — token i's selected
///   blocks land at columns `slot * sparse_block_kv + t`.
#[allow(clippy::too_many_arguments)]
pub fn sparse_mqa_logits_contiguous(
    dev: &Device,
    stream: &DevStream,
    q: &Operand,
    q_sf: &DevBuffer,
    kv: &Operand,
    kv_sf: &DevBuffer,
    weights: &DevBuffer,
    metadata: &DevBuffer,
    num_tokens: u32,
    num_heads: u32,
    sparse_block_kv: u32,
    use_unaligned_ks: bool,
    logits: &mut DevBuffer,
    logits_stride: u32,
) -> DgResult<()> {
    check_logits_common(dev, q, num_heads, sparse_block_kv)?;
    check_q_rows(q, num_tokens, num_heads)?;
    if kv.k != SPARSE_MQA_HEAD_DIM || kv.dtype != q.dtype {
        return Err(DgError::InvalidArg(
            "sparse MQA requires KV [num_kv_tokens, 128] of Q's dtype".into(),
        ));
    }
    let num_kv_tokens = kv.rows;
    let cfg = SparseMqaConfig::new(
        num_heads,
        q.dtype,
        sparse_block_kv,
        use_unaligned_ks,
        None,
        dev.num_sms,
    )?;

    // Q/KV maps are the standard MQA QK maps (swizzle = head_dim / pack,
    // box outer = the per-TMA row count: 2*heads for Q, 128 tokens for KV).
    let tm_q = tma::make_tma_mqa_qk(
        dev,
        q.dtype,
        &q.data,
        q.rows,
        SPARSE_MQA_HEAD_DIM,
        SPARSE_MQA_BLOCK_Q * num_heads,
        cfg.is_fp4,
    )?;
    let tm_kv = tma::make_tma_mqa_qk(
        dev,
        kv.dtype,
        &kv.data,
        kv.rows,
        SPARSE_MQA_HEAD_DIM,
        128, // kSparseKVTokensPerTMA
        cfg.is_fp4,
    )?;
    let tm_sf_q = make_tma_sparse_sf_q(dev, q_sf, num_heads, num_tokens)?;
    let tm_sf_kv = make_tma_sparse_sf_kv(dev, kv_sf, num_kv_tokens)?;
    let tm_w = make_tma_sparse_weights(dev, weights, num_heads, num_tokens)?;

    let body = sparse_mqa_logits_body(&cfg);
    let sig = format!(
        "sparse_logits_{heads}_{sbkv}_{qs}_{kvs}_{ts}_{mwg}_{sms}_{bq}_{uks}_{fp4}",
        heads = cfg.num_heads,
        sbkv = cfg.sparse_block_kv,
        qs = cfg.q_stages,
        kvs = cfg.kv_stages,
        ts = cfg.tmem_stages,
        mwg = cfg.num_math_warpgroups,
        sms = cfg.num_sms,
        bq = SPARSE_MQA_BLOCK_Q,
        uks = cfg.use_unaligned_ks as u32,
        fp4 = cfg.is_fp4 as u32,
    );
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::SPARSE_MQA,
        "sparse_mqa_logits",
        &sig,
        &body,
    )?;

    let args = Args::new()
        .u32(logits_stride)
        .u32(num_kv_tokens)
        .devptr(logits.ptr)
        .devptr(kv.data.ptr)
        .devptr(kv_sf.ptr)
        .devptr(metadata.ptr)
        .tensormap(&tm_q)
        .tensormap(&tm_sf_q)
        .tensormap(&tm_w)
        .tensormap(&tm_kv)
        .tensormap(&tm_sf_kv);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (cfg.num_threads(), 1, 1),
            smem: cfg.logits_smem_bytes(),
            cluster: None,
            pdl: true,
        },
        args,
    )
}

/// Score the selected (top-k) KV blocks against Q with a paged fused cache.
///
/// * `fused_kv_cache`: `[num_pages, kv_page_stride_bytes]`; each page holds
///   `[page_kv tokens x head_dim/pack bytes of KV][page_kv int32 SF words]`
///   (the stride must be a multiple of 512);
/// * `page_kv`: tokens per page (multiple of `sparse_block_kv`);
/// * `metadata`: filled by [`sparse_mqa_metadata`] in paged mode.
#[allow(clippy::too_many_arguments)]
pub fn sparse_mqa_logits_paged(
    dev: &Device,
    stream: &DevStream,
    q: &Operand,
    q_sf: &DevBuffer,
    fused_kv_cache: &DevBuffer,
    page_kv: u32,
    kv_page_stride_bytes: u32,
    weights: &DevBuffer,
    metadata: &DevBuffer,
    num_tokens: u32,
    num_heads: u32,
    sparse_block_kv: u32,
    logits: &mut DevBuffer,
    logits_stride: u32,
) -> DgResult<()> {
    check_logits_common(dev, q, num_heads, sparse_block_kv)?;
    check_q_rows(q, num_tokens, num_heads)?;
    if page_kv == 0 || page_kv % sparse_block_kv != 0 {
        return Err(DgError::InvalidArg(
            "page_kv must be a positive multiple of sparse_block_kv".into(),
        ));
    }
    if kv_page_stride_bytes % 512 != 0 {
        return Err(DgError::InvalidArg(
            "kv_page_stride_bytes must be a multiple of 512".into(),
        ));
    }
    let cfg = SparseMqaConfig::new(
        num_heads,
        q.dtype,
        sparse_block_kv,
        false,
        Some(page_kv),
        dev.num_sms,
    )?;

    let tm_q = tma::make_tma_mqa_qk(
        dev,
        q.dtype,
        &q.data,
        q.rows,
        SPARSE_MQA_HEAD_DIM,
        SPARSE_MQA_BLOCK_Q * num_heads,
        cfg.is_fp4,
    )?;
    let tm_sf_q = make_tma_sparse_sf_q(dev, q_sf, num_heads, num_tokens)?;
    let tm_w = make_tma_sparse_weights(dev, weights, num_heads, num_tokens)?;

    let body = sparse_mqa_logits_paged_body(&cfg);
    let sig = format!(
        "sparse_logits_paged_{heads}_{page}_{sbkv}_{qs}_{kvs}_{ts}_{mwg}_{sms}_{bq}_{fp4}",
        heads = cfg.num_heads,
        page = cfg.page_kv,
        sbkv = cfg.sparse_block_kv,
        qs = cfg.q_stages,
        kvs = cfg.kv_stages,
        ts = cfg.tmem_stages,
        mwg = cfg.num_math_warpgroups,
        sms = cfg.num_sms,
        bq = SPARSE_MQA_BLOCK_Q,
        fp4 = cfg.is_fp4 as u32,
    );
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::SPARSE_MQA,
        "sparse_mqa_logits_paged",
        &sig,
        &body,
    )?;

    let args = Args::new()
        .u32(logits_stride)
        .u32(kv_page_stride_bytes)
        .devptr(logits.ptr)
        .devptr(fused_kv_cache.ptr)
        .devptr(metadata.ptr)
        .tensormap(&tm_q)
        .tensormap(&tm_sf_q)
        .tensormap(&tm_w);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (cfg.num_threads(), 1, 1),
            smem: cfg.logits_smem_bytes(),
            cluster: None,
            pdl: true,
        },
        args,
    )
}
