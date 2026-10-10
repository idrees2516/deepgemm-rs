//! SM90 (Hopper) MQA-logits launchers — contiguous-KV and paged-KV variants
//! (port of upstream `csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp`).
//!
//! # What these kernels compute
//!
//! The MLA *lightning indexer* score (see `kernels/mqa_logits_sm90.cu` for
//! the full concept guide):
//!
//! ```text
//! logits[i, j] = s_kv[j] * sum_h w[i, h] * relu(<q[i, h, :], kv[j, :]>)
//! ```
//!
//! Q is raw E4M3 (no scales on SM90), KV is E4M3 with ONE fp32 scale per KV
//! token, weights are fp32 per (token, head). Output is fp32.
//!
//! # Contiguous variant ([`mqa_logits_sm90`])
//!
//! * `q`: fp8 `[seq_len, num_heads, head_dim]`, contiguous (row stride
//!   `head_dim`); passed as an [`Operand`] with `rows = seq_len * heads`.
//! * `kv`: fp8 `[seq_len_kv, head_dim]`, contiguous.
//! * `kv_scales`: fp32 `[seq_len_kv]`.
//! * `weights`: fp32 `[seq_len, num_heads]` with row stride `weights_stride`
//!   (must be a multiple of 4 floats — 16B TMA alignment).
//! * `cu_k_start`/`cu_k_end`: u32 `[seq_len]` per-token KV windows; must
//!   satisfy `cu_k_start[i] <= cu_k_end[i] <= seq_len_kv` (values are clamped
//!   to `seq_len_kv`, mis-ordered windows break the block loop).
//! * `logits`: fp32 `[align_up(seq_len, block_q), logits_stride]` — rows are
//!   padded to whole `block_q` tiles (the kernel's last partial tile writes
//!   with the last token's window); each token's valid span is stored
//!   COMPRESSED at columns `[0, k_end_i - k_start_i)`. Upstream sizes the
//!   stride as `align(align(max_seqlen_k, 256), 256)`.
//!
//! Tiles (upstream host constants, mirrored by [`sm90_mqa_contig_config`]):
//! `block_q = 128 / num_heads` (so the WGMMA N is always 128),
//! `block_kv = 256 = math_threads / 2`, 3 Q stages, 3 KV stages,
//! 128 TMA threads + 512 math threads.
//!
//! # Paged variant ([`mqa_paged_logits_sm90_metadata`] + [`mqa_paged_logits_sm90`])
//!
//! Decode shape with a paged KV cache of `PAGE_KV = 64`-token pages:
//!
//! * `q`: fp8 `[batch, next_n, num_heads, head_dim]` (`next_n` in {1, 2});
//!   the map's row stride is the stride between `(b, n)` token planes along
//!   the head axis (`q.stride(2)` upstream) — `head_dim` when contiguous.
//! * `kv_pages`: fp8 `[num_pages, 64, head_dim]`; `kv_sf_pages`: fp32
//!   `[num_pages, 64]` (one scale per KV token).
//! * `context_lens`: u32 `[batch, next_n]` (2D only; the last token of a
//!   request carries its full context length).
//! * `block_table`: u32 `[batch, block_table_stride]` — page id per request.
//! * `schedule_meta`: u32 `[(num_sms + 1) * 2]`, produced on-device by
//!   [`mqa_paged_logits_sm90_metadata`] (launched with PDL so the main
//!   kernel's prologue overlaps it).
//! * `logits`: fp32 `[batch * next_n, logits_stride]`, DENSE (unconditional
//!   writes — the last split of a request may write columns beyond its
//!   context length; size `logits_stride` accordingly, upstream uses
//!   `align(max_context_len, 256)` with `logits_stride % 256 == 0`).
//!
//! Split geometry: `SPLIT_KV = 256 = 64 (BLOCK_KV) x 4 math warpgroups`; one
//! producer warp feeds each math warpgroup's private KV pipe.

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, Operand};

/// Upstream `SPLIT_KV` constant for the SM90 paged kernel.
const SPLIT_KV: u32 = 256;
/// Paged KV page size (statically asserted by the kernel: BLOCK_KV == 64).
const PAGE_KV: u32 = 64;

fn require_sm90(dev: &Device) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm90) {
        return Err(DgError::Unsupported(format!(
            "the SM90 MQA logits kernels require SM90 (Hopper); this device is {:?}",
            dev.arch
        )));
    }
    Ok(())
}

fn validate_mqa_shape(num_heads: u32, head_dim: u32) -> DgResult<()> {
    if num_heads != 32 && num_heads != 64 {
        return Err(DgError::InvalidArg(
            "SM90 MQA logits require num_heads in {32, 64} (upstream dispatch assert)".into(),
        ));
    }
    if !matches!(head_dim, 32 | 64 | 128) {
        return Err(DgError::InvalidArg(
            "SM90 MQA logits require head_dim in {32, 64, 128}".into(),
        ));
    }
    Ok(())
}

fn align_up(x: u32, a: u32) -> u32 {
    x.div_ceil(a) * a
}

/// SM90 MQA translation unit: prelude (auto-prepended by the JIT) + wgmma.h
/// + this crate's SM90 MQA kernels.
fn mqa_sm90_unit() -> &'static str {
    static U: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    U.get_or_init(|| {
        format!(
            "{}\n{}",
            jit::kernel_src::WGMMA_H,
            jit::kernel_src::MQA_SM90
        )
    })
    .as_str()
}

// ---------------------------------------------------------------------------
// Tile choice (private; mirrors upstream host constants)
// ---------------------------------------------------------------------------

/// Contiguous tiles: `BLOCK_Q * heads == 128` (one WGMMA N),
/// `BLOCK_KV == math_threads / 2 == 256`, 3/3 stages, 128+512 threads.
fn sm90_mqa_contig_config(num_heads: u32) -> DgResult<(u32, u32)> {
    validate_mqa_shape(num_heads, 128)?;
    Ok((128 / num_heads, 256))
}

// ---------------------------------------------------------------------------
// TMA descriptors (private; the two flavors tma.rs has no public builder for)
// ---------------------------------------------------------------------------

/// Weights map: fp32 `[num_heads (inner), num_tokens (outer)]`, box
/// `[num_heads, box_rows]`, explicit row stride (f32 elements), no swizzle.
/// Port of `make_tma_2d_desc(weights, num_heads, seq, num_heads, block_q,
/// weights.stride(0), 0)`.
#[allow(clippy::too_many_arguments)]
fn make_tma_mqa_weights_f32(
    dev: &Device,
    buf: &DevBuffer,
    num_heads: u32,
    num_tokens: u32,
    box_rows: u32,
    row_stride: u32,
) -> DgResult<sys::TensorMap> {
    if row_stride % 4 != 0 {
        return Err(DgError::InvalidArg(
            "weights row stride must be a multiple of 4 floats (16B TMA alignment)".into(),
        ));
    }
    if row_stride < num_heads {
        return Err(DgError::InvalidArg("weights row stride < num_heads".into()));
    }
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        sys::tm_dtype_float32(),
        2,
        buf.ptr as *mut _,
        &[num_heads as u64, num_tokens as u64],
        &[row_stride as u64 * 4],
        &[num_heads, box_rows],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(0),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// Q map: fp8 `[head_dim (inner), rows (outer)]`, box `[head_dim, box_rows]`,
/// `head_dim`-byte swizzle (the 8-row WGMMA atom), explicit row stride in fp8
/// elements (upstream `q.stride(2)` for the paged tensor).
#[allow(clippy::too_many_arguments)]
fn make_tma_mqa_q_fp8(
    dev: &Device,
    buf: &DevBuffer,
    rows: u32,
    head_dim: u32,
    box_rows: u32,
    row_stride: u32,
) -> DgResult<sys::TensorMap> {
    if row_stride % 16 != 0 {
        return Err(DgError::InvalidArg(
            "Q row stride must be 16B aligned (fp8 elements)".into(),
        ));
    }
    if row_stride < head_dim {
        return Err(DgError::InvalidArg("Q row stride < head_dim".into()));
    }
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        sys::tm_dtype_uint8(),
        2,
        buf.ptr as *mut _,
        &[head_dim as u64, rows as u64],
        &[row_stride as u64],
        // swizzle != 0: the SMEM box inner is the swizzle atom (head_dim).
        &[head_dim, box_rows],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(head_dim.min(128)),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

// ---------------------------------------------------------------------------
// Kernel wrapper bodies (public: shared with the offline compile-check test
// `tests/sm90_mqa_cpu.rs`, so the test compiles exactly what the launcher
// launches).
// ---------------------------------------------------------------------------

/// Wrapper body for the contiguous SM90 MQA kernel.
pub fn mqa_logits_sm90_body(num_heads: u32, head_dim: u32, num_sms: u32) -> DgResult<String> {
    let (block_q, block_kv) = sm90_mqa_contig_config(num_heads)?;
    validate_mqa_shape(num_heads, head_dim)?;
    Ok(format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned seq_len, unsigned seq_len_kv, unsigned stride_logits,
    const unsigned* cu_k_start, const unsigned* cu_k_end, float* logits,
    const __grid_constant__ dg::TmaMap tma_q,
    const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_kv_scales,
    const __grid_constant__ dg::TmaMap tma_weights) {{
    dg::mqa_logits_sm90_impl<{heads}, {head_dim}, {block_q}, {block_kv},
        3, 3, {num_sms}, 128, 512>
        (seq_len, seq_len_kv, stride_logits, cu_k_start, cu_k_end, logits,
         tma_q, tma_kv, tma_kv_scales, tma_weights);
}}"#,
        heads = num_heads,
        head_dim = head_dim,
        block_q = block_q,
        block_kv = block_kv,
        num_sms = num_sms,
    ))
}

/// Wrapper body for the paged SM90 MQA kernel (`is_context_lens_2d` is fixed
/// to 1: only 2D context lens are supported, as upstream; the grid size does
/// not enter the template — the per-SM schedule arrives via `schedule_meta`).
pub fn mqa_paged_logits_sm90_body(next_n: u32, num_heads: u32, head_dim: u32) -> DgResult<String> {
    if next_n != 1 && next_n != 2 {
        return Err(DgError::InvalidArg(
            "SM90 paged MQA requires next_n in {1, 2}".into(),
        ));
    }
    validate_mqa_shape(num_heads, head_dim)?;
    Ok(format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned batch_size, unsigned logits_stride, unsigned block_table_stride,
    const unsigned* context_lens, float* logits,
    const unsigned* block_table, const unsigned* indices, const unsigned* schedule_meta,
    const __grid_constant__ dg::TmaMap tma_q,
    const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_kv_scales,
    const __grid_constant__ dg::TmaMap tma_weights) {{
    dg::mqa_paged_logits_sm90_impl<{next_n}, {heads}, {head_dim}, 64, 1, 0,
        3, 3, {split_kv}, 128, 512>
        (batch_size, logits_stride, block_table_stride, context_lens, logits,
         block_table, indices, schedule_meta, tma_q, tma_kv, tma_kv_scales, tma_weights);
}}"#,
        next_n = next_n,
        heads = num_heads,
        head_dim = head_dim,
        split_kv = SPLIT_KV,
    ))
}

/// Wrapper body for the paged SM90 MQA metadata kernel.
pub fn mqa_paged_logits_sm90_metadata_body(aligned_batch_size: u32, num_sms: u32) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned batch_size, unsigned next_n, unsigned is_context_lens_2d,
    const unsigned* context_lens, const unsigned* indices, unsigned* schedule_metadata) {{
    dg::sm90_paged_mqa_logits_metadata_impl<{aligned_batch}, {split_kv}, {num_sms}, 0>
        (batch_size, next_n, is_context_lens_2d, context_lens, indices, schedule_metadata);
}}"#,
        aligned_batch = aligned_batch_size,
        split_kv = SPLIT_KV,
    )
}

// ---------------------------------------------------------------------------
// Contiguous-KV launcher
// ---------------------------------------------------------------------------

/// SM90 MQA logits, contiguous KV (see the module docs for tensor layouts).
/// One persistent 640-thread launch per SM; PDL-enabled.
// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn mqa_logits_sm90(
    dev: &Device,
    stream: &DevStream,
    q: &Operand,
    kv: &Operand,
    kv_scales: &DevBuffer,
    weights: &DevBuffer,
    weights_stride: u32,
    cu_k_start: &DevBuffer,
    cu_k_end: &DevBuffer,
    num_tokens: u32,
    num_kv_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    logits: &mut DevBuffer,
    logits_stride: u32,
) -> DgResult<()> {
    require_sm90(dev)?;
    validate_mqa_shape(num_heads, head_dim)?;
    let (block_q, block_kv) = sm90_mqa_contig_config(num_heads)?;
    const Q_STAGES: u32 = 3;
    const KV_STAGES: u32 = 3;
    const MATH_THREADS: u32 = 512;

    if q.dtype != Dtype::Fp8 || kv.dtype != Dtype::Fp8 {
        return Err(DgError::InvalidArg(
            "SM90 MQA logits require FP8 (E4M3) Q and KV".into(),
        ));
    }
    if num_tokens == 0 || num_kv_tokens == 0 {
        return Err(DgError::InvalidArg(
            "empty sequences are not supported".into(),
        ));
    }
    if q.rows != num_tokens * num_heads || q.k != head_dim || q.outer_stride != head_dim {
        return Err(DgError::InvalidArg(
            "q must be contiguous [seq_len, num_heads, head_dim] (rows = seq*heads)".into(),
        ));
    }
    if kv.rows != num_kv_tokens || kv.k != head_dim || kv.outer_stride != head_dim {
        return Err(DgError::InvalidArg(
            "kv must be contiguous [seq_len_kv, head_dim]".into(),
        ));
    }
    let need = |cond: bool, what: &str| -> DgResult<()> {
        if cond {
            Ok(())
        } else {
            Err(DgError::InvalidArg(format!("{what} buffer too small")))
        }
    };
    need(
        (q.data.len as u64) >= num_tokens as u64 * num_heads as u64 * head_dim as u64,
        "q",
    )?;
    need(
        (kv.data.len as u64) >= num_kv_tokens as u64 * head_dim as u64,
        "kv",
    )?;
    need(
        (kv_scales.len as u64) >= num_kv_tokens as u64 * 4,
        "kv_scales",
    )?;
    need(
        (weights.len as u64) >= num_tokens as u64 * weights_stride as u64 * 4,
        "weights",
    )?;
    need(
        (cu_k_start.len as u64) >= num_tokens as u64 * 4,
        "cu_k_start",
    )?;
    need((cu_k_end.len as u64) >= num_tokens as u64 * 4, "cu_k_end")?;
    let aligned_rows = align_up(num_tokens, block_q);
    need(
        (logits.len as u64) >= aligned_rows as u64 * logits_stride as u64 * 4,
        "logits",
    )?;

    // Tensor maps (port of the upstream host descriptors).
    let tm_q = make_tma_mqa_q_fp8(
        dev,
        &q.data,
        q.rows,
        head_dim,
        block_q * num_heads,
        head_dim,
    )?;
    let tm_kv = tma::make_tma_mqa_qk(
        dev,
        Dtype::Fp8,
        &kv.data,
        kv.rows,
        head_dim,
        block_kv,
        false,
    )?;
    let tm_kv_scales = tma::make_tma_mqa_sf(dev, kv_scales, num_kv_tokens, block_kv)?;
    let tm_weights =
        make_tma_mqa_weights_f32(dev, weights, num_heads, num_tokens, block_q, weights_stride)?;

    // Shared memory budget (exact mirror of the upstream host formula; the
    // barrier term over-allocates by 8 barrier slots + 4 bytes of padding).
    let smem = Q_STAGES * (block_q * num_heads * head_dim)
        + KV_STAGES * (block_kv * head_dim)
        + Q_STAGES * (block_q * num_heads * 4)
        + KV_STAGES * (block_kv * 4)
        + (Q_STAGES * 2 + KV_STAGES * 2 + (MATH_THREADS / 128) * 2) * 8
        + 4;
    if smem > dev.smem_capacity {
        return Err(DgError::Unsupported(format!(
            "SM90 MQA logits need {smem} B of shared memory (capacity {})",
            dev.smem_capacity
        )));
    }

    let sig = format!(
        "mqa90c_{num_heads}_{head_dim}_{block_q}_{block_kv}_{}",
        dev.num_sms
    );
    let body = mqa_logits_sm90_body(num_heads, head_dim, dev.num_sms)?;
    let func = jit::get_kernel(dev, mqa_sm90_unit(), "sm90_mqa_logits", &sig, &body)?;

    let args = Args::new()
        .u32(num_tokens)
        .u32(num_kv_tokens)
        .u32(logits_stride)
        .ptr(cu_k_start.ptr as *const u8 as *const u32)
        .ptr(cu_k_end.ptr as *const u8 as *const u32)
        .ptr(logits.ptr as *const u8 as *const f32)
        .tensormap(&tm_q)
        .tensormap(&tm_kv)
        .tensormap(&tm_kv_scales)
        .tensormap(&tm_weights);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + MATH_THREADS, 1, 1),
            smem,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

// ---------------------------------------------------------------------------
// Paged-KV launchers
// ---------------------------------------------------------------------------

/// SM90 paged MQA metadata: turns `context_lens` into per-SM (q_atom,
/// kv_split) start ranges in `schedule_meta` (`u32 [(num_sms + 1) * 2]`).
/// Launch it (with PDL) immediately before [`mqa_paged_logits_sm90`].
pub fn mqa_paged_logits_sm90_metadata(
    dev: &Device,
    stream: &DevStream,
    context_lens: &DevBuffer,      // u32 [batch * next_n] (2D)
    schedule_meta: &mut DevBuffer, // u32 [(num_sms + 1) * 2]
    batch_size: u32,
    next_n: u32,
) -> DgResult<()> {
    require_sm90(dev)?;
    if batch_size == 0 {
        return Err(DgError::InvalidArg("batch_size must be >= 1".into()));
    }
    if (context_lens.len as u64) < batch_size as u64 * next_n as u64 * 4 {
        return Err(DgError::InvalidArg("context_lens buffer too small".into()));
    }
    if (schedule_meta.len as u64) < (dev.num_sms as u64 + 1) * 2 * 4 {
        return Err(DgError::InvalidArg(
            "schedule_meta must hold (num_sms + 1) * 2 u32 words".into(),
        ));
    }
    let aligned_batch_size = align_up(batch_size, 32);
    let smem = aligned_batch_size * 4;
    if smem > dev.smem_capacity {
        return Err(DgError::Unsupported(format!(
            "SM90 paged MQA metadata needs {smem} B of shared memory"
        )));
    }

    let sig = format!("mqa90meta_{aligned_batch_size}_{}", dev.num_sms);
    let body = mqa_paged_logits_sm90_metadata_body(aligned_batch_size, dev.num_sms);
    let func = jit::get_kernel(dev, mqa_sm90_unit(), "sm90_mqa_meta", &sig, &body)?;

    let args = Args::new()
        .u32(batch_size)
        .u32(next_n)
        .u32(1) // is_context_lens_2d (only 2D lens are supported)
        .ptr(context_lens.ptr as *const u8 as *const u32)
        .devptr(0) // indices (varlen-only; not supported on SM90)
        .ptr(schedule_meta.ptr as *const u8 as *const u32);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            smem,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

/// SM90 paged MQA logits (decode path; see the module docs for layouts).
/// `schedule_meta` must have been produced by
/// [`mqa_paged_logits_sm90_metadata`] on the same stream.
// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn mqa_paged_logits_sm90(
    dev: &Device,
    stream: &DevStream,
    q: &Operand,             // fp8 [batch, next_n, heads, head_dim]
    kv_pages: &DevBuffer,    // fp8 [num_pages, 64, head_dim]
    kv_sf_pages: &DevBuffer, // fp32 [num_pages, 64]
    weights: &DevBuffer,     // fp32 [batch * next_n, heads]
    weights_stride: u32,
    context_lens: &DevBuffer, // u32 [batch, next_n]
    block_table: &DevBuffer,  // u32 [batch, block_table_stride]
    block_table_stride: u32,
    schedule_meta: &DevBuffer, // u32 [(num_sms + 1) * 2]
    batch_size: u32,
    next_n: u32,
    num_heads: u32,
    head_dim: u32,
    num_pages: u32,
    logits: &mut DevBuffer, // fp32 [batch * next_n, logits_stride]
    logits_stride: u32,
) -> DgResult<()> {
    require_sm90(dev)?;
    validate_mqa_shape(num_heads, head_dim)?;
    if next_n != 1 && next_n != 2 {
        return Err(DgError::InvalidArg(
            "SM90 paged MQA requires next_n in {1, 2}".into(),
        ));
    }
    const Q_STAGES: u32 = 3;
    const KV_STAGES: u32 = 3;
    const MATH_THREADS: u32 = 512;
    let num_math_wgs = SPLIT_KV / 64;

    if q.dtype != Dtype::Fp8 {
        return Err(DgError::InvalidArg("SM90 paged MQA requires FP8 Q".into()));
    }
    if q.rows != batch_size * next_n * num_heads || q.k != head_dim {
        return Err(DgError::InvalidArg(
            "q must be [batch, next_n, num_heads, head_dim] (rows = batch*next_n*heads)".into(),
        ));
    }
    if logits_stride % SPLIT_KV != 0 {
        return Err(DgError::InvalidArg(format!(
            "logits_stride must be a multiple of {SPLIT_KV} (upstream assert)"
        )));
    }
    if num_pages == 0 {
        return Err(DgError::InvalidArg("num_pages must be >= 1".into()));
    }
    let need = |cond: bool, what: &str| -> DgResult<()> {
        if cond {
            Ok(())
        } else {
            Err(DgError::InvalidArg(format!("{what} buffer too small")))
        }
    };
    let q_rows = batch_size as u64 * next_n as u64 * num_heads as u64;
    need(
        (q.data.len as u64) >= (q_rows - 1) * q.outer_stride as u64 + head_dim as u64,
        "q",
    )?;
    need(
        (kv_pages.len as u64) >= num_pages as u64 * PAGE_KV as u64 * head_dim as u64,
        "kv_pages",
    )?;
    need(
        (kv_sf_pages.len as u64) >= num_pages as u64 * PAGE_KV as u64 * 4,
        "kv_sf_pages",
    )?;
    need(
        (weights.len as u64) >= batch_size as u64 * next_n as u64 * weights_stride as u64 * 4,
        "weights",
    )?;
    need(
        (context_lens.len as u64) >= batch_size as u64 * next_n as u64 * 4,
        "context_lens",
    )?;
    need(
        (block_table.len as u64) >= batch_size as u64 * block_table_stride as u64 * 4,
        "block_table",
    )?;
    need(
        (schedule_meta.len as u64) >= (dev.num_sms as u64 + 1) * 2 * 4,
        "schedule_meta",
    )?;
    need(
        (logits.len as u64) >= batch_size as u64 * next_n as u64 * logits_stride as u64 * 4,
        "logits",
    )?;

    // Tensor maps (port of the upstream host descriptors).
    let tm_q = make_tma_mqa_q_fp8(
        dev,
        &q.data,
        q.rows,
        head_dim,
        next_n * num_heads,
        q.outer_stride,
    )?;
    let tm_kv = tma::make_tma_mqa_qk_paged(
        dev,
        Dtype::Fp8,
        kv_pages,
        head_dim,
        PAGE_KV,
        num_pages,
        false,
    )?;
    let tm_kv_scales = tma::make_tma_mqa_sf_paged(dev, kv_sf_pages, PAGE_KV, num_pages)?;
    let tm_weights = make_tma_mqa_weights_f32(
        dev,
        weights,
        num_heads,
        batch_size * next_n,
        next_n,
        weights_stride,
    )?;

    // Shared memory budget (exact mirror of the upstream host formula):
    // one Q pipe + one KV pipe per math warpgroup, barrier areas padded to
    // the swizzle alignment, plus the (unused upstream) UMMA-barrier tail.
    let swz = head_dim * 8;
    let smem_q_pipe = Q_STAGES
        * (next_n * num_heads * head_dim + align_up(next_n * num_heads * 4, swz))
        + align_up(Q_STAGES * 8 * 2, swz);
    let smem_kv_pipe = KV_STAGES * (PAGE_KV * head_dim + align_up(PAGE_KV * 4, swz))
        + align_up(KV_STAGES * 8 * 2, swz);
    let smem = smem_q_pipe + num_math_wgs * smem_kv_pipe + num_math_wgs * 2 * 8 + 4;
    if smem > dev.smem_capacity {
        return Err(DgError::Unsupported(format!(
            "SM90 paged MQA logits need {smem} B of shared memory (capacity {})",
            dev.smem_capacity
        )));
    }

    let sig = format!(
        "mqa90p_{next_n}_{num_heads}_{head_dim}_{split_kv}_{sms}",
        split_kv = SPLIT_KV,
        sms = dev.num_sms
    );
    let body = mqa_paged_logits_sm90_body(next_n, num_heads, head_dim)?;
    let func = jit::get_kernel(dev, mqa_sm90_unit(), "sm90_mqa_paged", &sig, &body)?;

    let args = Args::new()
        .u32(batch_size)
        .u32(logits_stride)
        .u32(block_table_stride)
        .ptr(context_lens.ptr as *const u8 as *const u32)
        .ptr(logits.ptr as *const u8 as *const f32)
        .ptr(block_table.ptr as *const u8 as *const u32)
        .devptr(0) // indices (varlen-only; not supported on SM90)
        .ptr(schedule_meta.ptr as *const u8 as *const u32)
        .tensormap(&tm_q)
        .tensormap(&tm_kv)
        .tensormap(&tm_kv_scales)
        .tensormap(&tm_weights);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + MATH_THREADS, 1, 1),
            smem,
            cluster: None,
            pdl: true,
        },
        args,
    )
}
