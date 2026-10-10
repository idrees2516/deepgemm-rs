//! MegaMoE host-side layout math — port of `layout/mega_moe.cuh` and the
//! `Workspace` contract of `impls/sm100_fp8_fp4_mega_moe.cuh`.
//!
//! The megakernel itself (a single persistent launch that fuses act-quant,
//! DeepEP dispatch, per-expert grouped GEMM and combine, synchronizing
//! through NVLink release/acquire signals) is **not yet ported**; this
//! module pins the *host-side contract* it depends on, so the kernel port
//! lands against a frozen, tested surface:
//!   * pool capacities (token pool, SF ring, shared-L2 pools),
//!   * the `MegaMoESignals` control-block layout (exact upstream offsets),
//!   * per-rank workspace sizing.
//!
//! All formulas are exact ports (see tests); `num_ranks = 1` gives the
//! single-GPU degenerate sizes used by local testing.

/// Candidate BLOCK_M set (upstream `kCandidateBlockM`).
pub const CANDIDATE_BLOCK_MS: [u32; 8] = [8, 16, 32, 64, 96, 128, 192, 240];
pub const MAX_CANDIDATE_BLOCK_M: u32 = 240;
pub const MIN_CANDIDATE_BLOCK_M: u32 = 8;
pub const LCM_CANDIDATE_BLOCK_M: u32 = 1920;

/// Shared-expert token pool capacity: worst-case received tokens + per-
/// expert BLOCK_M alignment padding, over all possible BLOCK_M choices.
/// `align` is upstream `math::constexpr_align` (round up to `to`).
pub fn num_max_pool_tokens(
    num_ranks: u32,
    num_max_tokens_per_rank: u32,
    num_topk: u32,
    num_experts_per_rank: u32,
) -> u32 {
    let num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    let num_max_experts_per_token = num_topk.min(num_experts_per_rank);
    align(
        num_max_recv_tokens * num_max_experts_per_token
            + num_experts_per_rank * (MAX_CANDIDATE_BLOCK_M - 1),
        LCM_CANDIDATE_BLOCK_M,
    )
}

/// SF pool capacity: pool blocks x aligned BLOCK_M rows per block.
pub fn num_sf_ring_tokens(num_ring_tokens: u32, block_m: u32) -> u32 {
    (num_ring_tokens / block_m) * align(block_m, 128)
}

/// Shared-L2 input SF capacity: worst-case aligned SF pages.
pub fn num_max_shared_sf_tokens(num_max_tokens_per_rank: u32) -> u32 {
    ceil_div(num_max_tokens_per_rank, MIN_CANDIDATE_BLOCK_M) * 128
}

fn align(x: u32, to: u32) -> u32 {
    ceil_div(x, to) * to
}
fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Locality domains (upstream `kNumDeviceLocalityDomains`).
pub const NUM_DEVICE_LOCALITY_DOMAINS: u32 = 8;
/// Upper bounds from `MegaMoESignals` (exact upstream constants).
pub const NUM_MAX_GRID_SYNC_COUNTERS: usize = 4;
pub const NUM_MAX_EXPERTS: usize = 2048;
pub const NUM_MAX_EXPERTS_PER_RANK: usize = 512;
/// 20 MiB of routed-expert ring blocks.
pub const NUM_MAX_RING_BLOCKS: usize = 1 << 20;
/// 128 KiB of shared-expert L2 blocks.
pub const NUM_MAX_SHARED_L2_BLOCKS: usize = 1 << 15;
pub const NUM_MAX_RANKS: usize = 64;

/// The control block the megakernel synchronizes through (single allocation
/// at the head of the MoE workspace; offsets are load-bearing — the kernel
/// asserts `offsetof(combine_ready_grid_idx) == 128`).
///
/// Layout (bytes, upstream order):
/// ```text
///   0    grid_sync_count[4]                    16B
///   16   nvl_barrier_counter                    8B
///   24   nvl_barrier_signals[2]                 8B   (align 32 below)
///   32   l1/l2/shared task counts [4 x 8 dom]  128B
///   160  combine_ready_grid_idx[ranks]          (align128)
///   ..   peer_grid_idx[ranks]                   (align128)
///   ..   expert_send/recv/recv_sum counts       3 x 2048 u64
///   ..   ring signals (full/empty/mask)         1<<20 blocks
///   ..   shared_l2 signals                      1<<15 blocks
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MegaMoESignalsLayout {
    pub num_ranks: usize,
    pub total_bytes: usize,
    pub offset_combine_ready: usize,
    pub offset_peer_grid_idx: usize,
    pub offset_expert_send: usize,
    pub offset_ring_signals: usize,
    pub offset_shared_l2: usize,
}

impl MegaMoESignalsLayout {
    pub fn new(num_ranks: usize) -> MegaMoESignalsLayout {
        assert!(num_ranks > 0 && num_ranks <= NUM_MAX_RANKS);
        let pad128 = |x: usize| (x + 127) & !127usize;
        let offset_combine_ready = pad128(160 /* fixed head: sync + barrier + task counts */);
        let offset_peer_grid_idx = pad128(offset_combine_ready + num_ranks * 8);
        // expert_send + expert_recv + expert_recv_sum (u64 x 2048 each)
        let offset_expert_send = offset_peer_grid_idx + pad128(num_ranks * 8);
        let offset_ring_signals = offset_expert_send + 3 * NUM_MAX_EXPERTS * 8;
        // ring: l1_full u32 + l1_empty u32 + l2_mask u64 + l2_empty u32
        // per block; shared_l2: u32 per block.
        let ring_bytes = NUM_MAX_RING_BLOCKS * (4 + 4 + 8 + 4);
        let offset_shared_l2 = offset_ring_signals + ring_bytes;
        let total = offset_shared_l2 + NUM_MAX_SHARED_L2_BLOCKS * 4;
        MegaMoESignalsLayout {
            num_ranks,
            total_bytes: total,
            offset_combine_ready,
            offset_peer_grid_idx,
            offset_expert_send,
            offset_ring_signals,
            offset_shared_l2,
        }
    }
}

/// Host-side workspace description for one rank.
#[derive(Clone, Copy, Debug)]
pub struct MoEWorkspace {
    pub num_ranks: u32,
    pub num_experts: u32,
    pub num_experts_per_rank: u32,
    pub num_max_tokens_per_rank: u32,
    pub num_max_pool_tokens: u32,
    pub num_shared_l2_pool_blocks: u32,
    pub signals_bytes: usize,
}

impl MoEWorkspace {
    pub fn new(
        num_ranks: u32,
        num_experts: u32,
        num_max_tokens_per_rank: u32,
        num_topk: u32,
    ) -> MoEWorkspace {
        assert!(num_ranks > 0 && num_ranks <= NUM_MAX_RANKS as u32);
        assert!(
            num_experts % num_ranks == 0,
            "experts split evenly across ranks"
        );
        let num_experts_per_rank = num_experts / num_ranks;
        let num_max_pool_tokens = num_max_pool_tokens(
            num_ranks,
            num_max_tokens_per_rank,
            num_topk,
            num_experts_per_rank,
        );
        let num_shared_l2_pool_blocks = ceil_div(num_max_tokens_per_rank, MIN_CANDIDATE_BLOCK_M);
        MoEWorkspace {
            num_ranks,
            num_experts,
            num_experts_per_rank,
            num_max_tokens_per_rank,
            num_max_pool_tokens,
            num_shared_l2_pool_blocks,
            signals_bytes: MegaMoESignalsLayout::new(num_ranks as usize).total_bytes,
        }
    }
}
