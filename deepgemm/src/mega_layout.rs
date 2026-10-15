//! Host-side layout & sizing math for the megakernels.
//!
//! Mirrors upstream `layout/mega_moe.cuh` + `csrc/apis/mega_moe.hpp`
//! (symmetric-buffer sizing / slicing) and `layout/mega_mhc.cuh`
//! (split-K scratch geometry). The device-side structs in
//! `kernels/mega_moe_sm100.cu` reconstruct the same view from the base
//! pointer — the two implementations must stay in lockstep.

use crate::device::Device;
use crate::error::{DgError, DgResult};

pub const NUM_DEVICE_LOCALITY_DOMAINS: u32 = 2;
pub const MAX_RANKS: usize = 72;

pub const CANDIDATE_BLOCK_MS: [u32; 8] = [8, 16, 32, 64, 96, 128, 192, 240];
pub const MAX_CANDIDATE_BLOCK_M: u32 = 240;
pub const MIN_CANDIDATE_BLOCK_M: u32 = 8;
pub const LCM_CANDIDATE_BLOCK_M: u32 = 1920;

pub fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}
pub fn align_up(a: u32, b: u32) -> u32 {
    a.div_ceil(b) * b
}
fn align_up_u64(a: u64, b: u64) -> u64 {
    a.div_ceil(b) * b
}

// ---------------------------------------------------------------------------
// Pool / ring sizing (scheduler/mega_moe.cuh host mirror)
// ---------------------------------------------------------------------------
fn num_l1_warmup_waves(
    num_total_m_blocks: u32,
    num_clusters: u32,
    num_l1_n_clusters: u32,
    num_l2_n_clusters: u32,
) -> u32 {
    let first_l2_m = ceil_div(num_clusters, num_l2_n_clusters);
    let warm_first = ceil_div(first_l2_m * num_l1_n_clusters, num_clusters);
    let diff = num_l1_n_clusters.saturating_sub(num_l2_n_clusters);
    let interleave = ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * diff,
        num_clusters,
    ) + 1;
    warm_first.max(interleave)
}

/// Closed-form bound on concurrently-live pool blocks; the ring parity
/// protocol relies on this cap (mirrors `get_num_max_live_pool_blocks`).
pub fn num_max_live_pool_blocks(
    num_total_m_blocks: u32,
    num_sms: u32,
    hidden: u32,
    intermediate_hidden: u32,
) -> u32 {
    const MOE_BLOCK_N: u32 = 128;
    const CTAS_PER_CLUSTER: u32 = 2;
    let num_clusters = num_sms / CTAS_PER_CLUSTER;
    let l1_n = intermediate_hidden * 2 / (CTAS_PER_CLUSTER * MOE_BLOCK_N);
    let l2_n = hidden / (CTAS_PER_CLUSTER * MOE_BLOCK_N);
    let l1_clusters = num_total_m_blocks * l1_n;
    let l1_waves = ceil_div(l1_clusters, num_clusters);
    let warmup = num_l1_warmup_waves(num_total_m_blocks, num_clusters, l1_n, l2_n).min(l1_waves);
    let warmup_clusters = (warmup * num_clusters).min(l1_clusters);
    let live_after = ceil_div(warmup_clusters, l1_n);
    let growth = if l2_n > l1_n {
        ceil_div(num_total_m_blocks * (l2_n - l1_n), l2_n)
    } else {
        0
    };
    let margin = ceil_div(num_clusters, l1_n.min(l2_n));
    num_total_m_blocks.min(live_after + growth + margin)
}

fn num_max_pool_tokens(
    num_ranks: u32,
    max_tokens_per_rank: u32,
    topk: u32,
    experts_per_rank: u32,
) -> u32 {
    let recv = num_ranks * max_tokens_per_rank;
    let per_token = topk.min(experts_per_rank);
    align_up(
        recv * per_token + experts_per_rank * (MAX_CANDIDATE_BLOCK_M - 1),
        LCM_CANDIDATE_BLOCK_M,
    )
}

fn num_sf_ring_tokens(num_ring_tokens: u32, block_m: u32) -> u32 {
    (num_ring_tokens / block_m) * align_up(block_m, 128)
}

fn num_max_shared_sf_tokens(max_tokens_per_rank: u32) -> u32 {
    ceil_div(max_tokens_per_rank, MIN_CANDIDATE_BLOCK_M) * 128
}

// ---------------------------------------------------------------------------
// Buffer slicing: byte offsets of every region inside the symm allocation.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, Default)]
pub struct MegaMoeBufferOffsets {
    pub workspace_end: u64, // workspace (signals + metadata) ends here
    pub input_token: u64,
    pub input_sf: u64,
    pub input_topk_idx: u64,
    pub input_topk_weights: u64,
    pub shared_l1_sf: u64, // shared L1 tokens reuse input_token
    pub shared_l2_token: u64,
    pub shared_l2_sf: u64,
    pub l1_token: u64,
    pub l1_sf: u64,
    pub l1_topk_weights: u64,
    pub l2_token: u64,
    pub l2_sf: u64,
    pub combine: u64,
    pub total: u64,
}

/// One region's layout: `num_bytes` per token slot, `num_tokens` slots,
/// TMA alignment requirement (16B when TMA touches it).
#[derive(Clone, Copy)]
struct Region(u32, u32, bool);

struct Slicer {
    cursor: u64,
}

impl Slicer {
    fn place(&mut self, Region(nb, nt, _tma): Region) -> u64 {
        let end = align_up_u64(self.cursor + nb as u64 * nt as u64, 16);
        let base = self.cursor;
        self.cursor = end;
        base
    }
    fn end(&self) -> u64 {
        self.cursor
    }
}

/// Port of `get_symm_buffer_size_for_mega_moe`: iterate all candidate
/// BLOCK_M for the worst-case ring, then slice the full buffer.
pub fn mega_moe_buffer_layout(
    num_ranks: u32,
    num_experts: u32,
    num_max_tokens_per_rank: u32,
    num_topk: u32,
    hidden: u32,
    intermediate_hidden: u32,
    num_shared_experts: u32,
    num_sms: u32,
) -> DgResult<(u64, MegaMoeBufferOffsets)> {
    if num_experts % num_ranks != 0 {
        return Err(DgError::InvalidArg("num_experts % num_ranks != 0".into()));
    }
    let experts_per_rank = num_experts / num_ranks;
    let max_routed = num_max_tokens_per_rank * num_ranks * num_topk.min(experts_per_rank);

    // Ring capacity: worst case over all candidate BLOCK_M.
    let mut num_ring_tokens = 0u32;
    for &bm in CANDIDATE_BLOCK_MS.iter() {
        let pool_blocks = ceil_div(max_routed, bm) + experts_per_rank;
        let live = num_max_live_pool_blocks(pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = num_ring_tokens.max(live * bm);
    }
    num_ring_tokens = align_up(num_ring_tokens, LCM_CANDIDATE_BLOCK_M);

    let num_sf_ring = CANDIDATE_BLOCK_MS
        .iter()
        .map(|&bm| num_sf_ring_tokens(num_ring_tokens, bm))
        .max()
        .unwrap_or(0);

    let shared_ih = intermediate_hidden * num_shared_experts;
    if hidden % 128 != 0 || intermediate_hidden % 128 != 0 || shared_ih % 128 != 0 {
        return Err(DgError::InvalidArg(
            "hidden/intermediate must be multiples of 128 (SF packing)".into(),
        ));
    }
    if num_sf_ring % 4 != 0 {
        return Err(DgError::InvalidArg("SF ring tokens % 4 != 0".into()));
    }

    // Workspace: signals + src-token-topk + combine metadata (16B aligned).
    // sizeof(MegaMoESignals) — computed with the exact C++ alignment rules
    // (verified against the kernel struct; see scripts/ for the derivation):
    //   fixed head 28B -> combine_ready[72]@128 .. peer_grid_idx[72]@768
    //   .. expert_send[2048]@1408 (alignas 128) .. expert_recv[2048]@17792
    //   .. recv_sum[512]@34176 .. ring arrays (1<<20) x {4,4,8,4}
    //   .. shared_l2_full[1<<15] -> total rounded to alignas(128).
    const SIGNALS: u64 = 21_140_864;
    let pool_span = num_max_pool_tokens(
        num_ranks,
        num_max_tokens_per_rank,
        num_topk,
        experts_per_rank,
    );
    let workspace_bytes = align_up_u64(
        SIGNALS + num_experts as u64 * num_max_tokens_per_rank as u64 * 4 + pool_span as u64 * 12, // TokenSrcMetadata = 3 x u32
        16,
    );

    let shared_sf_tokens = num_max_shared_sf_tokens(num_max_tokens_per_rank);
    let mut s = Slicer {
        cursor: workspace_bytes,
    };
    let mut off = MegaMoeBufferOffsets::default();
    off.workspace_end = workspace_bytes;

    let r_input_token = Region(hidden, num_max_tokens_per_rank, true); // fp8 1B
    let r_input_sf = Region(hidden / 32, num_max_tokens_per_rank, false);
    let r_topk_idx = Region(num_topk * 8, num_max_tokens_per_rank, false);
    let r_topk_w = Region(num_topk * 4, num_max_tokens_per_rank, false);
    let r_l1_topk_w = Region(4, num_ring_tokens, false);
    let r_inter = Region(intermediate_hidden, num_ring_tokens, true);
    let r_inter_sf = Region(intermediate_hidden / 32, num_sf_ring, false);

    off.input_token = s.place(r_input_token);
    off.input_sf = s.place(r_input_sf);
    off.input_topk_idx = s.place(r_topk_idx);
    off.input_topk_weights = s.place(r_topk_w);
    off.shared_l1_sf = if num_shared_experts > 0 {
        s.place(Region(hidden / 32, shared_sf_tokens, false))
    } else {
        0
    };
    off.shared_l2_token = if num_shared_experts > 0 {
        s.place(Region(shared_ih, num_max_tokens_per_rank, true))
    } else {
        0
    };
    off.shared_l2_sf = if num_shared_experts > 0 {
        s.place(Region(shared_ih / 32, shared_sf_tokens, false))
    } else {
        0
    };
    // Routed ring (l1 tokens sized by the RING capacity, not the input slice).
    off.l1_token = s.place(Region(hidden, num_ring_tokens, true));
    off.l1_sf = s.place(Region(hidden / 32, num_sf_ring, false));
    off.l1_topk_weights = s.place(r_l1_topk_w);
    off.l2_token = s.place(r_inter);
    off.l2_sf = s.place(r_inter_sf);
    // Combine buffer: [topk+shared slots] x [num_max_tokens_per_rank] rows of
    // bf16 hidden vectors; the per-slot views are carved by the caller.
    {
        let slots = num_topk + u32::from(num_shared_experts > 0);
        off.combine = s.place(Region(hidden * 2, slots * num_max_tokens_per_rank, true));
    }
    off.total = s.end();
    Ok((off.total, off))
}

/// SF ring capacity to expose to callers (matches the C++ slicer).
pub fn mega_moe_num_sf_ring_tokens(num_ring_tokens: u32, block_m: u32) -> u32 {
    num_sf_ring_tokens(num_ring_tokens, block_m)
}

// ---------------------------------------------------------------------------
// mHC split-K scratch geometry (layout/mega_mhc.cuh mirror)
// ---------------------------------------------------------------------------
pub const MHC_BLOCK_M: u32 = 64;
pub const MHC_NUM_HC_OUTPUTS: u32 = 24;
pub const MHC_DEFAULT_SPLITS: u32 = 16;
pub const MHC_MAX_SPLITS: u32 = 64;
pub const MHC_BLOCK_K: u32 = 64;
pub const MHC_SWIZZLE: u32 = 128;

pub fn mhc_num_m_blocks(num_tokens: u32) -> u32 {
    ceil_div(num_tokens, MHC_BLOCK_M)
}

/// Fewest splits that reach the shortest possible longest task.
pub fn mhc_num_splits(num_m_blocks: u32, hidden: u32, num_sms: u32, deterministic: bool) -> u32 {
    if deterministic {
        return MHC_DEFAULT_SPLITS;
    }
    let k_blocks = hidden / MHC_BLOCK_K;
    let max_splits = MHC_MAX_SPLITS.min((num_sms / num_m_blocks).max(MHC_DEFAULT_SPLITS));
    ceil_div(k_blocks, ceil_div(k_blocks, max_splits))
}

pub fn mhc_scratch_bytes(num_m_blocks: u32, splits: u32) -> u64 {
    let tasks = num_m_blocks as u64 * splits as u64;
    tasks * (MHC_BLOCK_M as u64 * MHC_NUM_HC_OUTPUTS as u64 + 2 * MHC_BLOCK_M as u64) * 4
}

/// Split-barrier storage: 2 barriers x kNumMaxMBlocks lines x 128B.
pub fn mhc_split_barrier_bytes() -> u64 {
    2 * 16384 * 128 // kNumMaxMBlocks = ceil(2^20 / 64) = 16384
}

// ---------------------------------------------------------------------------
// Gate score scratch (layout/mega_gate.cuh mirror)
// ---------------------------------------------------------------------------
pub const GATE_BLOCK_K: u32 = 64;
pub const GATE_SWIZZLE: u32 = 128;
pub const GATE_MAX_BLOCK_TOKENS: u32 = 240; // fp8 cap (bf16 upstream: 256)
pub const GATE_MIN_BLOCK_TOKENS: u32 = 16;
pub const GATE_EXPERT_ALIGNMENT: u32 = 128;
pub const GATE_MAX_LOGICAL_CTAS: u32 = 64;
pub const GATE_MAX_STAGES: u32 = 32;
pub const GATE_MAX_METADATA_BYTES: u32 = 4 * 1024;

pub fn gate_scratch_bytes(
    num_token_blocks: u32,
    num_splits: u32,
    block_tokens: u32,
    num_experts: u32,
) -> u64 {
    num_token_blocks as u64 * num_splits as u64 * block_tokens as u64 * num_experts as u64 * 4
}

// ---------------------------------------------------------------------------
// SM locality domains: even TPC-alternating fallback (probe offline)
// ---------------------------------------------------------------------------
pub fn even_sm_locality_domains(num_sms: u32) -> Vec<u8> {
    let num_tpcs = num_sms / 2;
    let mut out = Vec::with_capacity(num_sms as usize);
    for tpc in 0..num_tpcs {
        let d = (tpc % NUM_DEVICE_LOCALITY_DOMAINS) as u8;
        out.push(d);
        out.push(d);
    }
    out
}

/// Validation helper shared by the API entry points.
pub fn require_blackwell(dev: &Device) -> DgResult<()> {
    match dev.arch {
        crate::device::Arch::Sm100 | crate::device::Arch::Sm103 => Ok(()),
        other => Err(DgError::Unsupported(format!(
            "the tcgen05 megakernels require SM100 (B200) or SM103 (B300); this device is {other:?}"
        ))),
    }
}
