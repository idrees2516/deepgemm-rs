//! SM100 fp8xfp4 MegaMoE megakernel — Rust launcher + host-side layout math
//! (port of upstream `csrc/jit_kernels/heuristics/mega_moe.hpp`,
//! `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`,
//! `csrc/apis/mega_moe.hpp` and `deep_gemm/mega/__init__.py`).
//!
//! # What the megakernel does
//!
//! One persistent launch per rank fuses the whole MoE layer — expert dispatch
//! (NVLink symmetric-memory all-to-all), the L1 (up-projection + SwiGLU +
//! act-quant) grouped GEMM, the L2 (down-projection) grouped GEMM, and the
//! top-k combine — around the repo's tcgen05 2-CTA pipeline (see
//! `kernels/mega_moe_sm100.cu` for the full concept banner). The host side
//! this module implements:
//!
//! * **Symmetric-buffer sizing/slicing** ([`mega_moe_symm_buffer_bytes`],
//!   [`mega_moe_buffer_layout`]): byte-exact mirrors of the device
//!   `MegaMoEBuffer` / `Workspace` construction, cross-checked against the
//!   frozen [`crate::moe_layout`] formulas (a `static_assert` inside the JIT
//!   wrapper pins `sizeof(dg::MegaMoESignals<kNumRanks>)` to the Rust
//!   number for every compiled variant).
//! * **Launch-config selection** ([`MegaMoeConfig::new`]): the port of
//!   upstream `get_mega_moe_config` — BLOCK_M from the expected-tokens
//!   heuristic over `kCandidateBlockM`, store blocks, UTCCP-aligned SF
//!   blocks, pull width, pipeline stages from the SMEM budget (evaluated
//!   with the device-side `MegaMoeSmemLayout` formula so the launch always
//!   covers `sizeof(SharedStorage)`).
//! * **The 18 TMA descriptors** the kernel takes by value: per-level
//!   acts / acts-SF / weights(4D) / weights-SF / L1-output maps, routed and
//!   shared-expert. Weights are described by 4D maps `(K, N/2, 2 dies, E)`
//!   (raw `cuTensorMapEncodeTiled` — the non-localized "virtual die split"
//!   form of upstream `make_tma_weights_3d_desc`).
//! * **Weight preparation helpers**: [`interleave_weights`] (gate/up gran-8
//!   interleave, port of `_interleave_weights`) and
//!   [`transpose_sf_for_utccp`] (the 32x4<->4x32 SF-group transposition the
//!   UTCCP TMEM layout needs).
//!
//! # Buffer contract (who writes what)
//!
//! The symmetric buffer is allocated by the caller (torch symmetric memory
//! upstream; any device allocation for `num_ranks == 1`), must be **zeroed
//! once before the first launch** ([`mega_moe_zero_sym_buffer`]), and the
//! per-rank input regions (x fp8, x SF, topk idx, topk weights — offsets
//! from [`mega_moe_buffer_layout`]) must be (re-)written before every call.
//! The kernel self-cleans the control block between launches.

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, Args};
use crate::moe_layout;
use crate::sys;
use crate::types::Dtype;

// ---------------------------------------------------------------------------
// Constants (layout/mega_moe.cuh + heuristics/mega_moe.hpp)
// ---------------------------------------------------------------------------

/// Upstream `layout::kCandidateBlockM` (frozen in [`moe_layout`]).
pub const CANDIDATE_BLOCK_MS: [u32; 8] = moe_layout::CANDIDATE_BLOCK_MS;
/// LCM of all candidate BLOCK_M values (`kLCMCandidateBlockM`).
pub const LCM_CANDIDATE_BLOCK_M: u32 = moe_layout::LCM_CANDIDATE_BLOCK_M;
/// SM100A (B200) max dynamic shared memory, upstream `SM100ArchSpec`.
pub const SMEM_CAPACITY: u32 = 232_448;
/// Die count the weight 4D TMA maps split N across (upstream
/// `LocalityDomainAllocator::get_num_locality_domains()`). NOTE: distinct
/// from the *task-scheduling* locality domains
/// ([`moe_layout::NUM_DEVICE_LOCALITY_DOMAINS`], 8) — weights are physically
/// split across the two dies only.
pub const NUM_WEIGHT_LOCALITY_DOMAINS: u32 = 2;
/// `kNumTMAStoreStages` / `kNumScheduleStages` / `kNumEpilogueStages` (fixed
/// by the kernel).
pub const NUM_TMA_STORE_STAGES: u32 = 2;
pub const NUM_SCHEDULE_STAGES: u32 = 2;
/// `kMaxPullBytes` for the MXFP8FP4 dispatch pull (`is_mma_with_sf`).
const NUM_MAX_PULL_BYTES: u32 = 8192;
/// `kActivationClampBits` value meaning "no clamp" (+inf bits).
pub const ACTIVATION_CLAMP_NONE: u32 = 0x7f80_0000;

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}
fn align_to(v: u32, a: u32) -> u32 {
    v.div_ceil(a) * a
}
fn align_to_u64(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

// ---------------------------------------------------------------------------
// Device-layout mirrors (every formula the kernel's device structs compute)
// ---------------------------------------------------------------------------

/// `sizeof(dg::MegaMoESignals<num_ranks>)` — mirrors the constexpr
/// `moe_signals_*` model in the kernel. Byte-identical to the frozen
/// [`moe_layout::MegaMoESignalsLayout::new`] region offsets.
pub fn mega_moe_signals_bytes(num_ranks: u32) -> u64 {
    // Head: grid_sync_count[4] + nvl counter + signals[2] + 4 task-count
    // arrays of NUM_DEVICE_LOCALITY_DOMAINS entries; combine_ready is
    // 128-aligned after it.
    let off_combine_ready = align_to(
        16 + 4 + 8 + 4 * moe_layout::NUM_DEVICE_LOCALITY_DOMAINS * 4,
        128,
    );
    let off_peer = align_to(off_combine_ready + num_ranks * 8, 128);
    let off_expert_send = off_peer + align_to(num_ranks * 8, 128);
    let off_ring = off_expert_send as u64 + 3 * moe_layout::NUM_MAX_EXPERTS as u64 * 8;
    let off_shared_l2 = off_ring + moe_layout::NUM_MAX_RING_BLOCKS as u64 * 20;
    off_shared_l2 + moe_layout::NUM_MAX_SHARED_L2_BLOCKS as u64 * 4
}

/// `Workspace::get_num_bytes()`: the signals block + dispatch metadata
/// regions at the head of the symmetric buffer.
pub fn mega_moe_workspace_bytes(
    num_ranks: u32,
    num_experts: u32,
    num_max_tokens_per_rank: u32,
    num_topk: u32,
) -> u64 {
    let ws =
        moe_layout::MoEWorkspace::new(num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    let mut num_bytes = ws.signals_bytes as u64;
    // Source token-topk: [local expert][source rank][token]
    num_bytes += num_experts as u64 * num_max_tokens_per_rank as u64 * 4;
    // Combine push source indices (full pool span), 12 bytes each
    num_bytes += ws.num_max_pool_tokens as u64 * 12;
    // Align to TMA descriptor requirements
    align_to_u64(num_bytes, 16)
}

/// Shared-expert SF capacity `layout::get_num_max_shared_sf_tokens`
/// (frozen in [`moe_layout::num_max_shared_sf_tokens`]).
pub fn mega_moe_shared_sf_tokens(num_max_tokens_per_rank: u32) -> u32 {
    moe_layout::num_max_shared_sf_tokens(num_max_tokens_per_rank)
}

/// `layout::get_num_sf_ring_tokens`.
pub fn mega_moe_sf_ring_tokens(num_ring_tokens: u32, block_m: u32) -> u32 {
    moe_layout::num_sf_ring_tokens(num_ring_tokens, block_m)
}

/// Byte-exact mirror of the device `MegaMoESmemLayout` (static_assert'd
/// against `sizeof(SharedStorage)` inside the kernel, and against the Rust
/// literal inside the JIT wrapper). This is the dynamic SMEM the launch must
/// allocate.
#[allow(clippy::too_many_arguments)]
pub fn mega_moe_smem_bytes(
    num_experts: u32,
    num_dispatch_warps: u32,
    num_bytes_per_pull: u32,
    num_epilogue_warpgroups: u32,
    num_epilogue_warps: u32,
    store_block_m_l1: u32,
    l1_out_block_n: u32,
    store_block_m_l2: u32,
    block_n: u32,
    load_block_n: u32,
    num_stages: u32,
    load_block_m: u32,
    block_k: u32,
    sf_block_m: u32,
    sf_block_n: u32,
) -> u32 {
    let off_dispatch = align_to(num_experts * 4, 1024);
    let cd_l1 = num_epilogue_warpgroups * NUM_TMA_STORE_STAGES * store_block_m_l1 * l1_out_block_n;
    let cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * 2;
    let off_smem_d = align_to(off_dispatch + num_dispatch_warps * num_bytes_per_pull, 1024);
    let off_smem_a = align_to(off_smem_d + cd_l1.max(cd_l2), 1024);
    let off_smem_b = off_smem_a + num_stages * load_block_m * block_k;
    let off_smem_sfa = off_smem_b + num_stages * load_block_n * block_k;
    let off_smem_sfb = off_smem_sfa + num_stages * sf_block_m * (block_k / 128) * 4;
    let off_amax = align_to(
        off_smem_sfb + num_stages * sf_block_n * (block_k / 128) * 4,
        8,
    );
    let off_tasks = align_to(
        off_amax + num_epilogue_warps * (store_block_m_l1 / 2) * 8,
        16,
    );
    let off_barriers = off_tasks + NUM_SCHEDULE_STAGES * 32;
    let b = off_barriers
        + (num_dispatch_warps
            + num_stages * 2
            + 2 * 2
            + num_epilogue_warps * 2
            + NUM_SCHEDULE_STAGES * 2)
            * 8
        + 4; // tmem_ptr_in_smem
    align_to(b, 1024)
}

/// The whole symmetric buffer, region by region — the host-side mirror of
/// the device `MegaMoEBuffer` construction (identical member-init order, so
/// every `base` matches what the kernel computes from
/// `sym_buffer.get_base_ptr()`).
///
/// Region order (byte offsets from the symmetric-buffer base):
/// `workspace (signals + dispatch metadata) | x | x_sf | topk_idx |
/// topk_weights | shared_l1_sf | shared_l2_acts | shared_l2_sf | l1_acts |
/// l1_acts_sf | l1_topk_weights | l2_acts | l2_acts_sf | combine`.
/// (`shared_l1_acts` *is* `x` — shared experts read the input directly.)
#[derive(Clone, Copy, Debug)]
pub struct MegaMoeBufferLayout {
    /// `Workspace::get_num_bytes()`.
    pub workspace_bytes: u64,
    /// x: fp8 activations, `[num_max_tokens_per_rank, hidden]` row-major.
    pub off_input_token: u64,
    /// x SF: packed UE8M0 int32 `[num_max_tokens_per_rank, hidden / 128]`
    /// (K-major, one row per token — NOT transposed).
    pub off_input_sf: u64,
    /// topk expert indices: int64 `[num_max_tokens_per_rank, num_topk]`.
    pub off_input_topk_idx: u64,
    /// topk weights: f32 `[num_max_tokens_per_rank, num_topk]`.
    pub off_input_topk_weights: u64,
    /// Shared-expert L1 SF: MN-major `[hidden / 128][shared_sf_tokens]`
    /// int32 (the transposed copy of x_sf the shared-L1 GEMM TMA-loads).
    /// 0 bytes when `num_shared_experts == 0`.
    pub off_shared_l1_sf: u64,
    /// Shared L2 acts: fp8 `[num_max_tokens_per_rank, shared_intermediate]`.
    pub off_shared_l2_token: u64,
    /// Shared L2 SF: MN-major int32.
    pub off_shared_l2_sf: u64,
    /// L1 acts ring: fp8 `[num_ring_tokens, hidden]`.
    pub off_l1_token: u64,
    /// L1 acts SF ring: MN-major `[hidden / 128][num_sf_ring_tokens]`.
    pub off_l1_sf: u64,
    /// Per-slot topk weights of the L1 ring: f32 `[num_ring_tokens]`.
    pub off_l1_topk_weights: u64,
    /// L2 acts ring: fp8 `[num_ring_tokens, intermediate_hidden]`.
    pub off_l2_token: u64,
    /// L2 acts SF ring: MN-major `[intermediate_hidden / 128][num_sf_ring_tokens]`.
    pub off_l2_sf: u64,
    /// Combine slots: bf16 `[(num_topk + shared) ranks][T][hidden]`.
    pub off_combine_token: u64,
    /// End of the buffer (`MegaMoEBuffer::get_num_bytes`).
    pub total_bytes: u64,
}

/// Compute the symmetric-buffer region layout (see [`MegaMoeBufferLayout`]).
#[allow(clippy::too_many_arguments)]
pub fn mega_moe_buffer_layout(
    num_ranks: u32,
    num_experts: u32,
    num_max_tokens_per_rank: u32,
    num_topk: u32,
    hidden: u32,
    intermediate_hidden: u32,
    num_ring_tokens: u32,
    num_sf_ring_tokens: u32,
    num_shared_experts: u32,
) -> MegaMoeBufferLayout {
    assert!(num_ranks > 0 && num_experts.is_multiple_of(num_ranks));
    let t = num_max_tokens_per_rank;
    let shared_ih = intermediate_hidden * num_shared_experts;
    let has_shared = num_shared_experts > 0;
    let shared_sf_rows = if has_shared {
        mega_moe_shared_sf_tokens(t)
    } else {
        0
    };

    let workspace_bytes = mega_moe_workspace_bytes(num_ranks, num_experts, t, num_topk);
    let mut off = workspace_bytes;

    let off_input_token = off;
    off += t as u64 * hidden as u64;
    let off_input_sf = off;
    off += t as u64 * (hidden / 32) as u64;
    let off_input_topk_idx = off;
    off += t as u64 * num_topk as u64 * 8;
    let off_input_topk_weights = off;
    off += t as u64 * num_topk as u64 * 4;

    // Shared experts (with_sf is always true for the fp8xfp4 kernel; the
    // shared-L1 tokens reuse `input_token_buffer`).
    let off_shared_l1_sf = off;
    off += shared_sf_rows as u64 * (hidden / 32) as u64;
    let off_shared_l2_token = off;
    off += if has_shared {
        t as u64 * shared_ih as u64
    } else {
        0
    };
    let off_shared_l2_sf = off;
    off += shared_sf_rows as u64 * (shared_ih / 32) as u64;

    // Routed-expert rings
    let off_l1_token = off;
    off += num_ring_tokens as u64 * hidden as u64;
    let off_l1_sf = off;
    off += num_sf_ring_tokens as u64 * (hidden / 32) as u64;
    let off_l1_topk_weights = off;
    off += num_ring_tokens as u64 * 4;
    let off_l2_token = off;
    off += num_ring_tokens as u64 * intermediate_hidden as u64;
    let off_l2_sf = off;
    off += num_sf_ring_tokens as u64 * (intermediate_hidden / 32) as u64;
    let off_combine_token = off;
    let num_combine_ranks = num_topk + u32::from(has_shared);
    off += num_combine_ranks as u64 * t as u64 * (hidden * 2) as u64;

    MegaMoeBufferLayout {
        workspace_bytes,
        off_input_token,
        off_input_sf,
        off_input_topk_idx,
        off_input_topk_weights,
        off_shared_l1_sf,
        off_shared_l2_token,
        off_shared_l2_sf,
        off_l1_token,
        off_l1_sf,
        off_l1_topk_weights,
        off_l2_token,
        off_l2_sf,
        off_combine_token,
        total_bytes: off,
    }
}

// ---------------------------------------------------------------------------
// Scheduler math (scheduler/mega_moe.cuh) — needed for the ring capacity
// ---------------------------------------------------------------------------

/// `sched::get_num_l1_warmup_waves`.
pub fn get_num_l1_warmup_waves(
    num_total_m_blocks: u32,
    num_clusters: u32,
    num_l1_n_clusters: u32,
    num_l2_n_clusters: u32,
) -> u32 {
    let num_first_l2_wave_m_blocks = ceil_div(num_clusters, num_l2_n_clusters);
    let num_l1_warmup_clusters_for_first_l2_wave =
        ceil_div(num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);
    let num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters.saturating_sub(num_l2_n_clusters);
    let num_warmup_waves_for_interleave_schedule = ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters,
    ) + 1;
    num_l1_warmup_clusters_for_first_l2_wave.max(num_warmup_waves_for_interleave_schedule)
}

/// `sched::get_num_max_live_pool_blocks`.
pub fn get_num_max_live_pool_blocks(
    num_total_m_blocks: u32,
    num_sms: u32,
    hidden: u32,
    intermediate_hidden: u32,
) -> DgResult<u32> {
    const BLOCK_N: u32 = 128;
    const CTAS_PER_CLUSTER: u32 = 2;
    if !(intermediate_hidden * 2).is_multiple_of(CTAS_PER_CLUSTER * BLOCK_N)
        || !hidden.is_multiple_of(CTAS_PER_CLUSTER * BLOCK_N)
    {
        return Err(DgError::InvalidArg(format!(
            "MegaMoE shapes must be multiples of {}: hidden={hidden}, intermediate={intermediate_hidden}",
            CTAS_PER_CLUSTER * BLOCK_N
        )));
    }
    let num_clusters = num_sms / CTAS_PER_CLUSTER;
    let num_l1_n_clusters = intermediate_hidden * 2 / (CTAS_PER_CLUSTER * BLOCK_N);
    let num_l2_n_clusters = hidden / (CTAS_PER_CLUSTER * BLOCK_N);
    let num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    let num_l1_waves = ceil_div(num_l1_clusters, num_clusters);
    let num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks,
        num_clusters,
        num_l1_n_clusters,
        num_l2_n_clusters,
    );
    let num_l1_warmup_waves = num_min_l1_warmup_waves.min(num_l1_waves);
    let num_l1_warmup_clusters = (num_l1_warmup_waves * num_clusters).min(num_l1_clusters);
    let num_live_blocks_after_warmup = ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);
    let frontier_growth = if num_l2_n_clusters > num_l1_n_clusters {
        ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters),
            num_l2_n_clusters,
        )
    } else {
        0
    };
    let wave_margin = ceil_div(num_clusters, num_l1_n_clusters.min(num_l2_n_clusters));
    Ok(num_total_m_blocks.min(num_live_blocks_after_warmup + frontier_growth + wave_margin))
}

/// Ring capacities for the routed-expert token pool: the worst-case live
/// pool blocks over all candidate BLOCK_M, aligned to
/// [`LCM_CANDIDATE_BLOCK_M`]. Returns `(num_ring_tokens, num_sf_ring_tokens)`.
#[allow(clippy::too_many_arguments)]
pub fn mega_moe_ring_tokens(
    num_ranks: u32,
    num_experts: u32,
    num_max_tokens_per_rank: u32,
    num_topk: u32,
    num_sms: u32,
    hidden: u32,
    intermediate_hidden: u32,
) -> DgResult<(u32, u32)> {
    let num_experts_per_rank = num_experts / num_ranks;
    let num_active_topk = num_topk.min(num_experts_per_rank);
    let num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;
    let mut num_ring_tokens = 0u32;
    for &block_m in CANDIDATE_BLOCK_MS.iter() {
        let num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        let num_live =
            get_num_max_live_pool_blocks(num_pool_blocks, num_sms, hidden, intermediate_hidden)?;
        num_ring_tokens = num_ring_tokens.max(num_live * block_m);
    }
    num_ring_tokens = align_to(num_ring_tokens, LCM_CANDIDATE_BLOCK_M);
    let num_sf_ring_tokens = CANDIDATE_BLOCK_MS
        .iter()
        .map(|&bm| mega_moe_sf_ring_tokens(num_ring_tokens, bm))
        .max()
        .unwrap_or(0);
    Ok((num_ring_tokens, num_sf_ring_tokens))
}

/// Port of upstream `get_symm_buffer_size_for_mega_moe` (mma_type
/// "fp8xfp8"/"fp8xfp4", activation "swiglu"): ring capacity over all
/// candidate BLOCK_M + the full [`MegaMoeBufferLayout`] span. This is the
/// number of bytes each rank must allocate (symmetric memory when
/// `num_ranks > 1`).
#[allow(clippy::too_many_arguments)]
pub fn mega_moe_symm_buffer_bytes(
    num_ranks: u32,
    num_experts: u32,
    num_max_tokens_per_rank: u32,
    num_topk: u32,
    hidden: u32,
    intermediate_hidden: u32,
    num_shared_experts: u32,
    num_sms: u32,
) -> DgResult<u64> {
    if !num_experts.is_multiple_of(num_ranks) {
        return Err(DgError::InvalidArg(
            "num_experts must split evenly across ranks".into(),
        ));
    }
    if !hidden.is_multiple_of(128) || !intermediate_hidden.is_multiple_of(128) {
        return Err(DgError::InvalidArg(
            "hidden/intermediate_hidden must be multiples of 128 (SF packing)".into(),
        ));
    }
    let shared_ih = intermediate_hidden * num_shared_experts;
    if !shared_ih.is_multiple_of(128) {
        return Err(DgError::InvalidArg(
            "shared intermediate hidden must be a multiple of 128".into(),
        ));
    }
    let (num_ring_tokens, num_sf_ring_tokens) = mega_moe_ring_tokens(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        num_sms,
        hidden,
        intermediate_hidden,
    )?;
    if num_sf_ring_tokens % 4 != 0 {
        return Err(DgError::InvalidArg(
            "SF ring token count must be a multiple of 4".into(),
        ));
    }
    Ok(mega_moe_buffer_layout(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_ring_tokens,
        num_sf_ring_tokens,
        num_shared_experts,
    )
    .total_bytes)
}

// ---------------------------------------------------------------------------
// Launch configuration (heuristics/mega_moe.hpp)
// ---------------------------------------------------------------------------

/// Full instantiation parameters of `dg::mega_moe_fp8_fp4_impl` — the fields
/// the upstream `get_mega_moe_config` computes, plus the problem shape they
/// are selected for. [`MegaMoeConfig::new`] is the port of the upstream
/// heuristic; the fields are public so power users can override them
/// (mirroring upstream's env overrides) before building the kernel body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MegaMoeConfig {
    // Problem shape (also template parameters)
    pub num_max_tokens_per_rank: u32,
    pub hidden: u32,
    pub intermediate_hidden: u32,
    pub num_experts: u32,
    pub num_shared_experts: u32,
    pub num_topk: u32,
    pub num_ring_tokens: u32,
    pub num_sf_ring_tokens: u32,
    pub num_sms: u32,
    pub num_ranks: u32,

    // Block tiling
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub load_block_m: u32,
    pub load_block_n: u32,
    pub store_block_m_l1: u32,
    pub store_block_m_l2: u32,
    /// UTCCP 128-aligned SF tile heights/widths.
    pub sf_block_m: u32,
    pub sf_block_n: u32,

    // Pipeline
    pub num_stages: u32,
    /// Dynamic SMEM of the launch (the device `MegaMoeSmemLayout` formula).
    pub smem_bytes: u32,
    /// Dispatch pull chunk width in bytes (`kNumBytesPerPull`).
    pub num_bytes_per_pull: u32,

    // Threads (upstream: 128 dispatch + 128 non-epilogue + 256 epilogue)
    pub num_dispatch_threads: u32,
    pub num_non_epilogue_threads: u32,
    pub num_epilogue_threads: u32,

    // Kernel flags
    /// f32 bits; [`ACTIVATION_CLAMP_NONE`] disables clamping.
    pub activation_clamp_bits: u32,
    pub fast_math: bool,
    /// false => FP4 weights (unpacked to 1 byte/elem in SMEM).
    pub is_weight_fp8: bool,
}

impl MegaMoeConfig {
    /// Port of upstream `get_mega_moe_config` (MXFP8FP4 flavor): BLOCK_M from
    /// the expected-tokens-per-expert heuristic, asymmetric store blocks,
    /// `block_k = 128`, SF blocks UTCCP-aligned, pull width halved until
    /// `<= 8192` bytes, stages from the SMEM budget.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_ranks: u32,
        num_experts: u32,
        num_max_tokens_per_rank: u32,
        num_tokens: u32,
        num_topk: u32,
        hidden: u32,
        intermediate_hidden: u32,
        num_ring_tokens: u32,
        num_sf_ring_tokens: u32,
        num_shared_experts: u32,
        num_sms: u32,
        weight_dtype: Dtype,
        activation_clamp: Option<f32>,
        fast_math: bool,
    ) -> DgResult<MegaMoeConfig> {
        if !matches!(weight_dtype, Dtype::Fp8 | Dtype::Fp4) {
            return Err(DgError::InvalidArg(
                "MegaMoE weights must be FP8 or FP4".into(),
            ));
        }
        if num_ranks == 0 || num_ranks > moe_layout::NUM_MAX_RANKS as u32 {
            return Err(DgError::InvalidArg("invalid rank count".into()));
        }
        if !num_experts.is_multiple_of(num_ranks)
            || num_experts > moe_layout::NUM_MAX_EXPERTS as u32
        {
            return Err(DgError::InvalidArg("invalid expert count".into()));
        }
        let num_experts_per_rank = num_experts / num_ranks;
        if num_experts_per_rank > moe_layout::NUM_MAX_EXPERTS_PER_RANK as u32 {
            return Err(DgError::InvalidArg("too many experts per rank".into()));
        }
        if num_tokens > num_max_tokens_per_rank {
            return Err(DgError::InvalidArg(
                "num_tokens exceeds the per-rank capacity".into(),
            ));
        }
        if num_topk == 0 || num_topk + u32::from(num_shared_experts > 0) > 32 {
            return Err(DgError::InvalidArg(
                "invalid top-k (<= 32 slots in a warp)".into(),
            ));
        }
        // MMA shape constraints (kernel static_asserts): L2_SHAPE_N = hidden
        // needs % (BLOCK_N * 2) == 0; L1_SHAPE_N = 2 * intermediate needs
        // intermediate % 128 == 0.
        if !hidden.is_multiple_of(256) || !intermediate_hidden.is_multiple_of(128) {
            return Err(DgError::InvalidArg(
                "hidden must be a multiple of 256 and intermediate_hidden of 128 (2-CTA N tiling)"
                    .into(),
            ));
        }
        let shared_ih = intermediate_hidden * num_shared_experts;
        if num_shared_experts > 0 && !shared_ih.is_multiple_of(128) {
            return Err(DgError::InvalidArg(
                "shared intermediate hidden must be a multiple of 128".into(),
            ));
        }
        if num_ring_tokens == 0 || !num_ring_tokens.is_multiple_of(LCM_CANDIDATE_BLOCK_M) {
            return Err(DgError::InvalidArg(format!(
                "num_ring_tokens must be a positive multiple of {LCM_CANDIDATE_BLOCK_M}"
            )));
        }
        if !num_sf_ring_tokens.is_multiple_of(4) {
            return Err(DgError::InvalidArg(
                "num_sf_ring_tokens must be a multiple of 4".into(),
            ));
        }

        // ---- get_block_config_for_mega_moe ----
        let num_expected_tokens =
            num_tokens as f64 * num_ranks as f64 * num_topk as f64 / num_experts as f64;
        let num_covered_tokens = num_expected_tokens + num_expected_tokens.sqrt();
        let mut block_m = if num_expected_tokens <= 10.0 { 16 } else { 32 };
        if num_expected_tokens > 24.0 {
            let mut num_blocks = (num_covered_tokens / 240.0).ceil() as u32;
            if num_blocks == 1 && num_covered_tokens > 192.0 && num_experts / num_ranks < 14 {
                num_blocks = 2;
            }
            for &candidate in &[64u32, 128, 192, 240] {
                block_m = candidate;
                if num_blocks as f64 * candidate as f64 >= num_covered_tokens {
                    break;
                }
            }
        }
        if !CANDIDATE_BLOCK_MS.contains(&block_m) {
            return Err(DgError::InvalidArg("block_m heuristic failed".into()));
        }
        // Asymmetric store blocks at FP8xFP4 prefill (save smem for depth)
        let store_block_m_l1 = if block_m <= 16 {
            8
        } else if block_m <= 64 {
            16
        } else if block_m <= 192 {
            32
        } else {
            24
        };
        let is_weight_fp8 = weight_dtype == Dtype::Fp8;
        let store_block_m_l2 = if block_m == 240 && !is_weight_fp8 {
            8
        } else {
            store_block_m_l1
        };

        let block_n = 128u32;
        let block_k = 128u32; // 128 / num_mma_elem_bytes, elem = 1 byte
        let load_block_m = block_m / 2; // Always multicast on A
        let load_block_n = block_n;
        // SM100ArchSpec::get_sf_uttcp_aligned_block_sizes (SF path always on)
        let sf_block_m = align_to(block_m, 128).max(128);
        let sf_block_n = block_n;

        // Threads
        let num_dispatch_threads = 128u32;
        let num_non_epilogue_threads = 128u32;
        let num_epilogue_threads = 2 * 128u32;

        // Pull width: divide token bytes by 2 until <= max
        let mut num_bytes_per_pull = hidden; // num_mma_elem_bytes == 1
        while num_bytes_per_pull > NUM_MAX_PULL_BYTES {
            if !num_bytes_per_pull.is_multiple_of(2) {
                return Err(DgError::InvalidArg(
                    "hidden must stay even while halving the pull width".into(),
                ));
            }
            num_bytes_per_pull /= 2;
        }

        // ---- get_pipeline_config_for_mega_moe (smem via the device model) ----
        let num_dispatch_warps = num_dispatch_threads / 32;
        let num_epilogue_warps = num_epilogue_threads / 32;
        let num_epilogue_warpgroups = num_epilogue_warps / 4;
        let smem_at = |stages: u32| {
            mega_moe_smem_bytes(
                num_experts,
                num_dispatch_warps,
                num_bytes_per_pull,
                num_epilogue_warpgroups,
                num_epilogue_warps,
                store_block_m_l1,
                block_n / 2,
                store_block_m_l2,
                block_n,
                load_block_n,
                stages,
                load_block_m,
                block_k,
                sf_block_m,
                sf_block_n,
            )
        };
        let smem0 = smem_at(0);
        if smem0 > SMEM_CAPACITY {
            return Err(DgError::InvalidArg(format!(
                "MegaMoE fixed smem {smem0} exceeds capacity {SMEM_CAPACITY}"
            )));
        }
        // Max stages such that the *device-layout* total fits the budget.
        // (Upstream divides the capacity by a closed-form per-stage size;
        // evaluating the actual layout keeps the launch >= sizeof(SharedStorage)
        // even when the 1024-byte final alignment rounds up.)
        let per_stage =
            load_block_m * block_k + load_block_n * block_k + sf_block_m * 4 + sf_block_n * 4 + 16;
        let mut num_stages = (SMEM_CAPACITY - smem0) / per_stage;
        let mut smem_bytes = smem_at(num_stages);
        while smem_bytes > SMEM_CAPACITY && num_stages > 2 {
            num_stages -= 1;
            smem_bytes = smem_at(num_stages);
        }
        if num_stages < 2 {
            return Err(DgError::InvalidArg(
                "MegaMoE pipeline needs at least 2 stages".into(),
            ));
        }
        if num_stages > 32 {
            num_stages = 32;
            smem_bytes = smem_at(num_stages);
        }

        let clamp = activation_clamp.unwrap_or(f32::INFINITY);
        if !clamp.is_finite() && !clamp.is_infinite() || clamp < 0.0 {
            return Err(DgError::InvalidArg(
                "activation clamp must be a non-negative float".into(),
            ));
        }
        let activation_clamp_bits = if clamp.is_infinite() {
            ACTIVATION_CLAMP_NONE
        } else {
            clamp.to_bits()
        };

        Ok(MegaMoeConfig {
            num_max_tokens_per_rank,
            hidden,
            intermediate_hidden,
            num_experts,
            num_shared_experts,
            num_topk,
            num_ring_tokens,
            num_sf_ring_tokens,
            num_sms,
            num_ranks,
            block_m,
            block_n,
            block_k,
            load_block_m,
            load_block_n,
            store_block_m_l1,
            store_block_m_l2,
            sf_block_m,
            sf_block_n,
            num_stages,
            smem_bytes,
            num_bytes_per_pull,
            num_dispatch_threads,
            num_non_epilogue_threads,
            num_epilogue_threads,
            activation_clamp_bits,
            fast_math,
            is_weight_fp8,
        })
    }

    /// Total CTA size: dispatch + non-epilogue + epilogue threads.
    pub fn num_threads(&self) -> u32 {
        self.num_dispatch_threads + self.num_non_epilogue_threads + self.num_epilogue_threads
    }

    /// Register split the kernel's `setmaxnreg` will apply (upstream:
    /// lean experts grant the epilogue extra registers).
    pub fn register_split(&self) -> (u32, u32, u32) {
        let num_experts_per_rank = self.num_experts / self.num_ranks;
        if num_experts_per_rank <= 64 {
            (48, 40, 208)
        } else {
            (96, 72, 168)
        }
    }
}

// ---------------------------------------------------------------------------
// JIT wrapper body
// ---------------------------------------------------------------------------

const KERNEL: &str = include_str!("../kernels/mega_moe_sm100.cu");

/// The MegaMoE translation unit (prelude is auto-prepended by
/// [`jit::get_kernel`] / [`jit::compile_check_kernel`]).
pub fn mega_moe_unit() -> &'static str {
    KERNEL
}

/// Generate the `extern "C" __global__ void __dg_kernel(...)` wrapper that
/// instantiates `dg::mega_moe_fp8_fp4_impl<...>` for `cfg`. The embedded
/// `static_assert`s cross-check the *device* struct layouts against the
/// Rust-side numbers (the frozen [`crate::moe_layout`] signals contract and
/// the [`mega_moe_smem_bytes`] pipeline model), so any drift fails the
/// compile instead of corrupting buffers at runtime.
pub fn mega_moe_body(cfg: &MegaMoeConfig) -> String {
    let signals_bytes = mega_moe_signals_bytes(cfg.num_ranks);
    let smem = cfg.smem_bytes;
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    void* y, int* cumulative_local_expert_recv_stats, const unsigned num_tokens,
    const __grid_constant__ dg::SymBuffer<{num_ranks}> sym_buffer,
    const __grid_constant__ dg::TmaMap tensor_map_l1_acts,
    const __grid_constant__ dg::TmaMap tensor_map_l1_acts_sf,
    const __grid_constant__ dg::TmaMap tensor_map_l1_weights,
    const __grid_constant__ dg::TmaMap tensor_map_l1_weights_sf,
    const __grid_constant__ dg::TmaMap tensor_map_l1_output,
    const __grid_constant__ dg::TmaMap tensor_map_l2_acts,
    const __grid_constant__ dg::TmaMap tensor_map_l2_acts_sf,
    const __grid_constant__ dg::TmaMap tensor_map_l2_weights,
    const __grid_constant__ dg::TmaMap tensor_map_l2_weights_sf,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l1_acts,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l1_acts_sf,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l1_weights,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l1_weights_sf,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l1_output,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l2_acts,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l2_acts_sf,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l2_weights,
    const __grid_constant__ dg::TmaMap tensor_map_shared_l2_weights_sf,
    const unsigned char* sm_locality_domains) {{
    static_assert(sizeof(dg::MegaMoESignals<{num_ranks}>) == {signals_bytes},
                  "MegaMoE signals layout drift (frozen moe_layout.rs contract)");
    static_assert(dg::MegaMoeSmemLayout<
        {num_experts}, {num_dispatch_warps}, {num_bytes_per_pull},
        {num_epilogue_warpgroups}, {num_epilogue_warps},
        {tma_store_stages}, {store_block_m_l1}, {l1_out_block_n},
        {store_block_m_l2}, {block_n}, {load_block_n},
        {num_stages}, {load_block_m}, {block_k},
        {sf_block_m}, {sf_block_n}, {schedule_stages}>::num_bytes() == {smem},
                  "MegaMoE smem layout drift (api_mega_moe::mega_moe_smem_bytes)");
    dg::mega_moe_fp8_fp4_impl<
        {num_max_tokens_per_rank},
        {hidden}, {intermediate_hidden},
        {num_experts}, {num_shared_experts},
        {num_topk},
        {block_m}, {block_n}, {block_k},
        {store_block_m_l1}, {store_block_m_l2},
        {sf_block_m}, {sf_block_n},
        {num_ring_tokens},
        {num_sf_ring_tokens},
        {num_stages},
        {num_bytes_per_pull},
        {num_dispatch_threads}, {num_non_epilogue_threads}, {num_epilogue_threads},
        {num_sms}, {num_ranks},
        {activation_clamp_bits:#x}u, {fast_math}, {is_weight_fp8}
    >(y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer,
      tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, tensor_map_l1_weights_sf,
      tensor_map_l1_output, tensor_map_l2_acts, tensor_map_l2_acts_sf, tensor_map_l2_weights,
      tensor_map_l2_weights_sf, tensor_map_shared_l1_acts, tensor_map_shared_l1_acts_sf,
      tensor_map_shared_l1_weights, tensor_map_shared_l1_weights_sf, tensor_map_shared_l1_output,
      tensor_map_shared_l2_acts, tensor_map_shared_l2_acts_sf, tensor_map_shared_l2_weights,
      tensor_map_shared_l2_weights_sf, sm_locality_domains);
}}"#,
        num_ranks = cfg.num_ranks,
        num_experts = cfg.num_experts,
        num_dispatch_warps = cfg.num_dispatch_threads / 32,
        num_bytes_per_pull = cfg.num_bytes_per_pull,
        num_epilogue_warpgroups = cfg.num_epilogue_threads / 32 / 4,
        num_epilogue_warps = cfg.num_epilogue_threads / 32,
        tma_store_stages = NUM_TMA_STORE_STAGES,
        store_block_m_l1 = cfg.store_block_m_l1,
        l1_out_block_n = cfg.block_n / 2,
        store_block_m_l2 = cfg.store_block_m_l2,
        block_n = cfg.block_n,
        load_block_n = cfg.load_block_n,
        num_stages = cfg.num_stages,
        load_block_m = cfg.load_block_m,
        block_k = cfg.block_k,
        sf_block_m = cfg.sf_block_m,
        sf_block_n = cfg.sf_block_n,
        schedule_stages = NUM_SCHEDULE_STAGES,
        num_max_tokens_per_rank = cfg.num_max_tokens_per_rank,
        hidden = cfg.hidden,
        intermediate_hidden = cfg.intermediate_hidden,
        num_shared_experts = cfg.num_shared_experts,
        num_topk = cfg.num_topk,
        block_m = cfg.block_m,
        num_ring_tokens = cfg.num_ring_tokens,
        num_sf_ring_tokens = cfg.num_sf_ring_tokens,
        num_dispatch_threads = cfg.num_dispatch_threads,
        num_non_epilogue_threads = cfg.num_non_epilogue_threads,
        num_epilogue_threads = cfg.num_epilogue_threads,
        num_sms = cfg.num_sms,
        activation_clamp_bits = cfg.activation_clamp_bits,
        fast_math = cfg.fast_math as u32,
        is_weight_fp8 = cfg.is_weight_fp8 as u32,
        signals_bytes = signals_bytes,
        smem = smem,
    )
}

// ---------------------------------------------------------------------------
// TMA descriptor builders (runtime_utils.hpp ports; raw cuTensorMapEncode)
// ---------------------------------------------------------------------------

/// Wire dtype for one TMA element: UINT8 for FP8 and packed FP4 (raw bytes
/// on the global side), 16U4_ALIGN16B for FP4 unpacked to 1 byte/elem in SMEM.
fn tm_weight_dtype(dtype: Dtype) -> DgResult<sys::TmDtype> {
    match dtype {
        Dtype::Fp8 => Ok(sys::tm_dtype_uint8()),
        Dtype::Fp4 => Ok(sys::tm_dtype_16u4_align16b()),
        _ => Err(DgError::InvalidArg("weights must be fp8/fp4".into())),
    }
}

/// 2D act/output map — upstream `make_tma_2d_desc`: gmem `[inner, outer]`
/// with an outer byte stride, SMEM box `[block_inner, block_outer]`, the
/// inner box reduced to one swizzle atom when swizzling. `inner`/`stride`
/// are in *elements* (bytes for fp8).
#[allow(clippy::too_many_arguments)]
fn make_tma_map_2d(
    dev: &Device,
    addr: sys::DevicePtr,
    inner: u32,
    outer: u32,
    block_inner: u32,
    block_outer: u32,
    outer_stride_bytes: u64,
    swizzle_mode: u32,
    dtype: sys::TmDtype,
    elem_bytes: u32,
) -> DgResult<sys::TensorMap> {
    let mut smem_inner = block_inner;
    if swizzle_mode != 0 {
        smem_inner = swizzle_mode / elem_bytes;
    }
    if !outer_stride_bytes.is_multiple_of(16) {
        return Err(DgError::InvalidArg(format!(
            "TMA outer stride {outer_stride_bytes}B is not 16B-aligned"
        )));
    }
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype,
        2,
        addr as *mut _,
        &[inner as u64, outer as u64],
        &[outer_stride_bytes],
        &[smem_inner, block_outer],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// MN-major scale-factor map — upstream `make_tma_sf_desc`: int32
/// `[tma_aligned(mn) inner, ceil_div(k, 128) * groups outer]`, no swizzle,
/// box `[block_mn, sf_block_k]`. `sf_k_stride == 0` selects the compact
/// TMA-aligned layout.
#[allow(clippy::too_many_arguments)]
fn make_tma_map_sf(
    dev: &Device,
    addr: sys::DevicePtr,
    shape_mn: u32,
    shape_k: u32,
    block_mn: u32,
    sf_block_k: u32,
    num_groups: u32,
    sf_k_stride: u32,
) -> DgResult<sys::TensorMap> {
    let tma_aligned = ceil_div(shape_mn, 4) * 4; // 16B / 4B int32
    let packed_rows = ceil_div(shape_k, 32 * 4);
    let outer_stride = if sf_k_stride == 0 {
        tma_aligned
    } else {
        sf_k_stride
    };
    if outer_stride < tma_aligned {
        return Err(DgError::InvalidArg(
            "SF row stride smaller than the TMA-aligned MN".into(),
        ));
    }
    make_tma_map_2d(
        dev,
        addr,
        tma_aligned,
        packed_rows * num_groups,
        block_mn,
        sf_block_k.max(1),
        outer_stride as u64 * 4,
        0,
        sys::tm_dtype_int32(),
        4,
    )
}

/// 4D weights map — upstream `make_tma_weights_3d_desc` / `_2d_desc`
/// (non-localized "virtual die split"): `(K, N / 2, 2 dies, experts)` with
/// byte strides `(row, n_half, expert)`; box `(block_k, load_block_n, 1, 1)`
/// and the inner box reduced to the 128B swizzle atom. FP4 weights use
/// `16U4_ALIGN16B` (elements are nibbles in GMEM, 1 byte each in SMEM).
#[allow(clippy::too_many_arguments)]
fn make_tma_weights_4d(
    dev: &Device,
    dtype: Dtype,
    addr: sys::DevicePtr,
    n: u32,
    k: u32,
    num_experts: u32,
    row_stride_bytes: u64,
    block_k: u32,
    load_block_n: u32,
    swizzle_mode: u32,
) -> DgResult<sys::TensorMap> {
    let num_domains = NUM_WEIGHT_LOCALITY_DOMAINS;
    let n_per_domain = n / num_domains;
    let tm_dtype = tm_weight_dtype(dtype)?;
    // Global inner dim is in *elements*: nibbles for FP4, bytes for FP8. In
    // SMEM both unpack to 1 byte/element, so the 128B swizzle atom is 128
    // elements either way.
    if dtype == Dtype::Fp4 && !k.is_multiple_of(128) {
        return Err(DgError::InvalidArg(
            "FP4 global inner dim must be a multiple of 128 elements".into(),
        ));
    }
    let smem_inner = if swizzle_mode != 0 {
        swizzle_mode
    } else {
        block_k
    };
    let slice_stride = n_per_domain as u64 * row_stride_bytes;
    let expert_stride = n as u64 * row_stride_bytes;
    for s in [row_stride_bytes, slice_stride, expert_stride] {
        if s % 16 != 0 {
            return Err(DgError::InvalidArg(format!(
                "weights TMA stride {s}B is not 16B-aligned"
            )));
        }
    }
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        tm_dtype,
        4,
        addr as *mut _,
        &[
            k as u64,
            n_per_domain as u64,
            num_domains as u64,
            num_experts as u64,
        ],
        &[row_stride_bytes, slice_stride, expert_stride],
        &[smem_inner, load_block_n, 1, 1],
        &[1, 1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

// ---------------------------------------------------------------------------
// Host-side weight preparation (deep_gemm/mega/__init__.py ports)
// ---------------------------------------------------------------------------

/// Gate/up interleave at granularity `gran` (default 8) — port of upstream
/// `_interleave_weights`: rows `[gate: 0..7, up: 0..7, gate: 8..15, ...]`
/// instead of `[gate | up]`. `src` is one (or many, stacked along dim 0)
/// `[n, row_bytes]` weight tensors with `n = 2 * half`; the interleaved
/// bytes are returned. L1 weights (and L1 weight SFs, bytewise) must be
/// prepared with this before the launch.
pub fn interleave_weights(src: &[u8], num_groups: usize, n: usize, row_bytes: usize) -> Vec<u8> {
    assert!(n.is_multiple_of(2) && src.len() >= num_groups * n * row_bytes);
    let half = n / 2;
    let gran = 8usize;
    let mut dst = vec![0u8; src.len()];
    for g in 0..num_groups {
        let src_base = g * n * row_bytes;
        let dst_base = g * n * row_bytes;
        let mut out_row = 0usize;
        let mut base_row = 0usize;
        while base_row < half {
            let take = gran.min(half - base_row);
            // gate block then up block
            for src_off in [0, half] {
                let start = src_base + (base_row + src_off) * row_bytes;
                dst[dst_base + out_row * row_bytes..dst_base + (out_row + take) * row_bytes]
                    .copy_from_slice(&src[start..start + take * row_bytes]);
                out_row += take;
            }
            base_row += take;
        }
    }
    dst
}

/// UTCCP SF-group transposition — port of upstream `_transpose_sf_for_utccp`
/// for packed UE8M0 words: within each 128-row group of the MN dimension,
/// row `r` moves to `(r & !127) + (r & 31) * 4 + ((r >> 5) & 3)` (a 32x4 <->
/// 4x32 transpose matching the `tcgen05.cp` TMEM layout). `words` is
/// `[num_groups, mn, packed_sf_k]` int32; `mn` must be a multiple of 128.
/// Weight SFs (L1 after interleave, and L2) must be prepared with this.
pub fn transpose_sf_for_utccp(
    words: &[u32],
    num_groups: usize,
    mn: usize,
    packed_sf_k: usize,
) -> Vec<u32> {
    assert!(mn.is_multiple_of(128) && words.len() >= num_groups * mn * packed_sf_k);
    let mut dst = vec![0u32; words.len()];
    for g in 0..num_groups {
        for r in 0..mn {
            let t = (r & !127) + (r & 31) * 4 + ((r >> 5) & 3);
            let src = (g * mn + r) * packed_sf_k;
            let dst_off = (g * mn + t) * packed_sf_k;
            dst[dst_off..dst_off + packed_sf_k].copy_from_slice(&words[src..src + packed_sf_k]);
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// The launcher
// ---------------------------------------------------------------------------

/// One GEMM level's routed (or shared) weights, K-major
/// `[num_experts_per_rank, n, k]` (shared: no expert dim, `n`/`k` only).
#[derive(Clone, Copy)]
pub struct MegaMoeWeights<'a> {
    /// Weight bytes: fp8 1 byte/element, fp4 packed 2/byte.
    pub data: &'a DevBuffer,
    /// Packed UE8M0 SFs, int32 `[num_experts][ceil_div(k, 128)][tma_aligned(n)]`
    /// MN-major (compact TMA-aligned layout; L1 additionally gate/up
    /// interleaved + UTCCP-transposed — see [`interleave_weights`] /
    /// [`transpose_sf_for_utccp`]). Required for the fp8xfp4 kernel.
    pub sf: &'a DevBuffer,
    /// Logical N (elements) — `2 * intermediate_hidden` for L1,
    /// `hidden` for L2 (interleaved gate/up order for L1).
    pub n: u32,
    /// Logical K (elements).
    pub k: u32,
    /// Byte stride between N rows (== k bytes for fp8, k/2 for fp4).
    pub row_stride_bytes: u32,
    /// Fp8 or Fp4.
    pub dtype: Dtype,
}

/// Zero the whole symmetric buffer (upstream `SymmBuffer.__init__` /
/// `DG_COMM_KERNEL_DEBUG` re-zero). Must be done once before the first
/// launch; inputs must be re-copied afterwards.
pub fn mega_moe_zero_sym_buffer(
    dev: &Device,
    stream: &DevStream,
    sym_buffer: &DevBuffer,
) -> DgResult<()> {
    dev.bind()?;
    sys::memset_d8(sym_buffer.ptr, 0, sym_buffer.len, stream.raw())
}

/// The `dg::SymBuffer<kNumRanks>` by-value parameter bytes:
/// `[rank_idx u32 | pad u32 | bases[num_ranks] u64]`.
fn sym_buffer_bytes(num_ranks: u32, rank_idx: u32, bases: &[u64]) -> DgResult<Vec<u8>> {
    if bases.len() != num_ranks as usize {
        return Err(DgError::InvalidArg(
            "one symmetric-buffer base pointer per rank is required".into(),
        ));
    }
    let mut bytes = vec![0u8; 8 + 8 * num_ranks as usize];
    bytes[0..4].copy_from_slice(&rank_idx.to_ne_bytes());
    for (i, base) in bases.iter().enumerate() {
        bytes[8 + 8 * i..8 + 8 * i + 8].copy_from_slice(&base.to_ne_bytes());
    }
    Ok(bytes)
}

/// Launch the fp8xfp4 MegaMoE megakernel (port of upstream
/// `sm100_fp8_fp4_mega_moe`).
///
/// # Arguments
/// * `y` — bf16 output `[num_tokens, hidden]` (plain device buffer).
/// * `sym_buffer` — the per-rank symmetric buffer,
///   [`mega_moe_symm_buffer_bytes`] bytes, zeroed before first use; per-rank
///   input regions re-written by the caller before every call (offsets from
///   [`mega_moe_buffer_layout`]).
/// * `sym_buffer_bases` — base device pointer of the *same* symmetric
///   buffer on every rank (length == `num_ranks`; one entry for single GPU).
/// * `l1_weights` / `l2_weights` — routed expert weights (L1 gate/up
///   interleaved; SFs UTCCP-transposed), `[num_experts_per_rank, ...]`.
/// * `shared_l1_weights` / `shared_l2_weights` — optional shared experts
///   (both or neither), no expert dimension; fp8 only.
/// * `cumulative_local_expert_recv_stats` — optional int32
///   `[num_experts_per_rank]` accumulated across launches.
/// * `sm_locality_domains` — optional `[num_sms]` u8 table; `None` builds
///   the even TPC-round-robin mapping ([`crate::locality::even_sm_locality_domains`]).
#[allow(clippy::too_many_arguments)]
pub fn mega_moe_fp8_fp4(
    dev: &Device,
    stream: &DevStream,
    y: &DevBuffer,
    sym_buffer: &DevBuffer,
    sym_buffer_bases: &[u64],
    rank_idx: u32,
    num_tokens: u32,
    l1_weights: &MegaMoeWeights,
    l2_weights: &MegaMoeWeights,
    shared_l1_weights: Option<&MegaMoeWeights>,
    shared_l2_weights: Option<&MegaMoeWeights>,
    cumulative_local_expert_recv_stats: Option<&DevBuffer>,
    num_max_tokens_per_rank: u32,
    num_experts: u32,
    num_topk: u32,
    activation_clamp: Option<f32>,
    fast_math: bool,
    sm_locality_domains: Option<&DevBuffer>,
) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm100) {
        return Err(DgError::Unsupported(
            "the tcgen05 kernels require SM100 (Blackwell)".into(),
        ));
    }
    let num_ranks = sym_buffer_bases.len() as u32;
    let num_experts_per_rank = num_experts / num_ranks;
    if !num_experts.is_multiple_of(num_ranks) {
        return Err(DgError::InvalidArg(
            "num_experts must split evenly across ranks".into(),
        ));
    }

    // ---- Shape consistency (upstream check_weights_layout_*) ----
    let hidden = l2_weights.n;
    let intermediate_hidden = l1_weights.n / 2;
    let num_shared_experts = match (shared_l1_weights, shared_l2_weights) {
        (Some(l1), Some(l2)) => {
            let shared_ih = l2.k;
            if shared_ih % intermediate_hidden != 0
                || l1.n != shared_ih * 2
                || l1.k != hidden
                || l2.n != hidden
                || l1.dtype != Dtype::Fp8
                || l2.dtype != Dtype::Fp8
            {
                return Err(DgError::InvalidArg(
                    "inconsistent shared-expert weight shapes".into(),
                ));
            }
            shared_ih / intermediate_hidden
        }
        (None, None) => 0,
        _ => {
            return Err(DgError::InvalidArg(
                "shared L1 and L2 weights must be provided together".into(),
            ))
        }
    };
    if l1_weights.k != hidden
        || l2_weights.k != intermediate_hidden
        || l1_weights.n != intermediate_hidden * 2
        || l2_weights.n != hidden
        || l1_weights.dtype != l2_weights.dtype
    {
        return Err(DgError::InvalidArg(
            "inconsistent L1/L2 weight shapes (L1: [E, 2*I, H], L2: [E, H, I])".into(),
        ));
    }

    // ---- Workspace + ring capacities (moe_layout contract) ----
    let (num_ring_tokens, num_sf_ring_tokens) = mega_moe_ring_tokens(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        dev.num_sms,
        hidden,
        intermediate_hidden,
    )?;
    let layout = mega_moe_buffer_layout(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_ring_tokens,
        num_sf_ring_tokens,
        num_shared_experts,
    );
    if sym_buffer.len < layout.total_bytes as usize {
        return Err(DgError::InvalidArg(format!(
            "symmetric buffer too small: {} < {} bytes",
            sym_buffer.len, layout.total_bytes
        )));
    }
    if num_tokens > num_max_tokens_per_rank {
        return Err(DgError::InvalidArg(
            "num_tokens exceeds num_max_tokens_per_rank".into(),
        ));
    }

    // ---- Config + kernel ----
    let cfg = MegaMoeConfig::new(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_tokens,
        num_topk,
        hidden,
        intermediate_hidden,
        num_ring_tokens,
        num_sf_ring_tokens,
        num_shared_experts,
        dev.num_sms,
        l1_weights.dtype,
        activation_clamp,
        fast_math,
    )?;
    let body = mega_moe_body(&cfg);
    let sig = format!(
        "mega_moe_r{num_ranks}_e{num_experts}_t{num_max_tokens_per_rank}_k{num_topk}_\
         h{hidden}_i{intermediate_hidden}_s{num_shared_experts}_m{block_m}_\
         ring{num_ring_tokens}_st{num_stages}_fp8_{is_fp8}_sms{num_sms}",
        block_m = cfg.block_m,
        num_stages = cfg.num_stages,
        is_fp8 = cfg.is_weight_fp8 as u32,
        num_sms = dev.num_sms,
    );
    let func = jit::get_kernel(dev, mega_moe_unit(), "mega_moe", &sig, &body)?;

    // ---- The 18 TMA descriptors ----
    // Region bases inside the symmetric buffer.
    let region = |off: u64| -> sys::DevicePtr { sym_buffer.ptr + off };
    let swizzle_acts = 128u32;
    let swizzle_weights = 128u32;
    let sf_smem_outer_dim = cfg.block_k / (32 * 4); // always 1 at BLOCK_K=128

    // Routed acts: [num_ring_tokens rows, hidden/intermediate inner] fp8.
    let mk_acts = |addr, rows, k| {
        make_tma_map_2d(
            dev,
            addr,
            k,
            rows,
            cfg.block_k,
            cfg.load_block_m,
            k as u64,
            swizzle_acts,
            sys::tm_dtype_uint8(),
            1,
        )
    };
    let tm_l1_acts = mk_acts(region(layout.off_l1_token), num_ring_tokens, hidden)?;
    let tm_l2_acts = mk_acts(
        region(layout.off_l2_token),
        num_ring_tokens,
        intermediate_hidden,
    )?;
    // Routed acts SF: MN-major rings.
    let mk_acts_sf = |addr, k| {
        make_tma_map_sf(
            dev,
            addr,
            num_sf_ring_tokens,
            k,
            cfg.sf_block_m,
            sf_smem_outer_dim,
            1,
            num_sf_ring_tokens,
        )
    };
    let tm_l1_acts_sf = mk_acts_sf(region(layout.off_l1_sf), hidden)?;
    let tm_l2_acts_sf = mk_acts_sf(region(layout.off_l2_sf), intermediate_hidden)?;
    // Routed weights (4D) + weight SFs (groups stack along packed K).
    let tm_l1_weights = make_tma_weights_4d(
        dev,
        l1_weights.dtype,
        l1_weights.data.ptr,
        l1_weights.n,
        l1_weights.k,
        num_experts_per_rank,
        l1_weights.row_stride_bytes as u64,
        cfg.block_k,
        cfg.load_block_n,
        swizzle_weights,
    )?;
    let tm_l2_weights = make_tma_weights_4d(
        dev,
        l2_weights.dtype,
        l2_weights.data.ptr,
        l2_weights.n,
        l2_weights.k,
        num_experts_per_rank,
        l2_weights.row_stride_bytes as u64,
        cfg.block_k,
        cfg.load_block_n,
        swizzle_weights,
    )?;
    let mk_weights_sf = |w: &MegaMoeWeights, groups| {
        make_tma_map_sf(
            dev,
            w.sf.ptr,
            w.n,
            w.k,
            cfg.block_n, // SF_BLOCK_N == BLOCK_N (no padding)
            sf_smem_outer_dim,
            groups,
            0,
        )
    };
    let tm_l1_weights_sf = mk_weights_sf(l1_weights, num_experts_per_rank)?;
    let tm_l2_weights_sf = mk_weights_sf(l2_weights, num_experts_per_rank)?;

    // L1 output: post-SwiGLU N is halved (BLOCK_N/2 per input tile), so the
    // swizzle halves too (128 -> 64). L1 output and L2 acts are the same
    // tensor region.
    let tm_l1_output = make_tma_map_2d(
        dev,
        region(layout.off_l2_token),
        intermediate_hidden,
        num_ring_tokens,
        cfg.block_n / 2,
        cfg.store_block_m_l1,
        intermediate_hidden as u64,
        swizzle_acts / 2,
        sys::tm_dtype_uint8(),
        1,
    )?;

    // Shared experts (fall back to the routed maps when disabled, exactly
    // like upstream's `: tensor_map_l1_acts` defaults).
    let (
        tm_shared_l1_acts,
        tm_shared_l1_acts_sf,
        tm_shared_l1_weights,
        tm_shared_l1_weights_sf,
        tm_shared_l1_output,
        tm_shared_l2_acts,
        tm_shared_l2_acts_sf,
        tm_shared_l2_weights,
        tm_shared_l2_weights_sf,
    ) = if let (Some(sl1), Some(sl2)) = (shared_l1_weights, shared_l2_weights) {
        let shared_ih = intermediate_hidden * num_shared_experts;
        let shared_sf_rows = mega_moe_shared_sf_tokens(num_max_tokens_per_rank);
        (
            make_tma_map_2d(
                dev,
                region(layout.off_input_token),
                hidden,
                num_max_tokens_per_rank,
                cfg.block_k,
                cfg.load_block_m,
                hidden as u64,
                swizzle_acts,
                sys::tm_dtype_uint8(),
                1,
            )?,
            make_tma_map_sf(
                dev,
                region(layout.off_shared_l1_sf),
                shared_sf_rows,
                hidden,
                cfg.sf_block_m,
                sf_smem_outer_dim,
                1,
                shared_sf_rows,
            )?,
            make_tma_weights_4d(
                dev,
                sl1.dtype,
                sl1.data.ptr,
                sl1.n,
                sl1.k,
                1,
                sl1.row_stride_bytes as u64,
                cfg.block_k,
                cfg.load_block_n,
                swizzle_weights,
            )?,
            mk_weights_sf(sl1, 1)?,
            make_tma_map_2d(
                dev,
                region(layout.off_shared_l2_token),
                shared_ih,
                num_max_tokens_per_rank,
                cfg.block_n / 2,
                cfg.store_block_m_l1,
                shared_ih as u64,
                swizzle_acts / 2,
                sys::tm_dtype_uint8(),
                1,
            )?,
            make_tma_map_2d(
                dev,
                region(layout.off_shared_l2_token),
                shared_ih,
                num_max_tokens_per_rank,
                cfg.block_k,
                cfg.load_block_m,
                shared_ih as u64,
                swizzle_acts,
                sys::tm_dtype_uint8(),
                1,
            )?,
            make_tma_map_sf(
                dev,
                region(layout.off_shared_l2_sf),
                shared_sf_rows,
                shared_ih,
                cfg.sf_block_m,
                sf_smem_outer_dim,
                1,
                shared_sf_rows,
            )?,
            make_tma_weights_4d(
                dev,
                sl2.dtype,
                sl2.data.ptr,
                sl2.n,
                sl2.k,
                1,
                sl2.row_stride_bytes as u64,
                cfg.block_k,
                cfg.load_block_n,
                swizzle_weights,
            )?,
            mk_weights_sf(sl2, 1)?,
        )
    } else {
        (
            tm_l1_acts,
            tm_l1_acts_sf,
            tm_l1_weights,
            tm_l1_weights_sf,
            tm_l1_output,
            tm_l2_acts,
            tm_l2_acts_sf,
            tm_l2_weights,
            tm_l2_weights_sf,
        )
    };

    // ---- SM locality domains (even TPC-round-robin fallback) ----
    let even_domains = crate::device::alloc_and_upload(
        dev,
        &crate::locality::even_sm_locality_domains(dev.num_sms as usize),
        stream.raw(),
    )?;
    let domains_buf = sm_locality_domains.unwrap_or(&even_domains);

    // ---- Launch (grid = num_sms, 2-CTA clusters, PDL) ----
    let sym = sym_buffer_bytes(num_ranks, rank_idx, sym_buffer_bases)?;
    let args = Args::new()
        .devptr(y.ptr)
        .devptr(
            cumulative_local_expert_recv_stats
                .map(|b| b.ptr)
                .unwrap_or(0),
        )
        .u32(num_tokens)
        .raw_bytes(&sym)
        .tensormap(&tm_l1_acts)
        .tensormap(&tm_l1_acts_sf)
        .tensormap(&tm_l1_weights)
        .tensormap(&tm_l1_weights_sf)
        .tensormap(&tm_l1_output)
        .tensormap(&tm_l2_acts)
        .tensormap(&tm_l2_acts_sf)
        .tensormap(&tm_l2_weights)
        .tensormap(&tm_l2_weights_sf)
        .tensormap(&tm_shared_l1_acts)
        .tensormap(&tm_shared_l1_acts_sf)
        .tensormap(&tm_shared_l1_weights)
        .tensormap(&tm_shared_l1_weights_sf)
        .tensormap(&tm_shared_l1_output)
        .tensormap(&tm_shared_l2_acts)
        .tensormap(&tm_shared_l2_acts_sf)
        .tensormap(&tm_shared_l2_weights)
        .tensormap(&tm_shared_l2_weights_sf)
        .devptr(domains_buf.ptr);

    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (cfg.num_sms, 1, 1),
            block: (cfg.num_threads(), 1, 1),
            smem: cfg.smem_bytes,
            cluster: Some((2, 1, 1)),
            pdl: true,
        },
        args,
    )
}
