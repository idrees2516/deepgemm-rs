// DeepGEMM-RS: SM100 fp8xfp4 MegaMoE megakernel.
// Port of upstream `impls/sm100_fp8_fp4_mega_moe.cuh` (1528 lines) +
// `layout/mega_moe.cuh` + `layout/sym_buffer.cuh` + `comm/barrier.cuh` +
// `scheduler/mega_moe.cuh`, re-expressed for the zero-include NVRTC dialect
// (prelude.h is auto-prepended by the JIT engine).
//
// ===========================================================================
// CONCEPTS — the megakernel idea
// ===========================================================================
// A MoE layer ("act-quant -> all-to-all dispatch -> grouped GEMM up ->
// SwiGLU act-quant -> grouped GEMM down -> all-to-all combine") is
// traditionally a chain of kernel launches whose data round-trips HBM in
// between. The megakernel is ONE persistent launch per rank that does all:
//
//   x (fp8, per-rank)         weights (fp8/fp4, per-expert, interleaved)
//        │                          │
//   ┌────▼──────────────────────────▼──────────────────────────────────┐
//   │ dispatch warps: count tokens per expert, atomically claim slots   │
//   │  in every destination rank's pool (NVLink symmetric memory),      │
//   │  TMA-pull token bytes + SFs into a local ring buffer              │
//   ├───────────────────────────────────────────────────────────────────┤
//   │ scheduler warp: builds task descriptors (L1/L2/shared per N-tile) │
//   │  and publishes them into a 2-slot smem ring via st.async.cluster  │
//   ├───────────────────────────────────────────────────────────────────┤
//   │ A-TMA / B-TMA / MMA warps: the tcgen05 2-CTA pipeline from        │
//   │  gemm_sm100.cu, fed by the ring + weights                         │
//   ├───────────────────────────────────────────────────────────────────┤
//   │ epilogue warpgroups:                                              │
//   │   L1 phase  = SwiGLU + FP8 act-quant + SF (UE8M0) out via TMA     │
//   │   L2 phase  = BF16 accumulate, scatter into *remote* combine      │
//   │               buffers over NVLink (release stores, peer grid tags)│
//   │   then a final combine phase: top-k weighted reduce over the      │
//   │   per-slot combine buffers -> y                                   │
//   └───────────────────────────────────────────────────────────────────┘
//
// Synchronization is a device-side producer-consumer graph over a control
// block (`MegaMoESignals`) at the head of the symmetric buffer:
//   * ring counters (red.add.release / ld.acquire) for ring slot lifetimes,
//   * an XOR "l2_full_mask" so L2 K-blocks can start as soon as their L1
//     N-blocks finish (fine-grained L1 -> L2 dependency),
//   * grid-tag barriers (grid sense-reversal counters + st.release.sys /
//     ld.acquire.sys "combine_ready_grid_idx" peer tags) for cross-rank
//     combine readiness — grid indices are unique per launch, so no reset,
//   * an NVLink barrier (counter + signed signals) bracketing the dispatch
//     pull and the workspace cleanup.
//
// The routed-expert token pool is a RING (kNumRingTokens = num_ring_blocks *
// BLOCK_M) sized by `get_num_max_live_pool_blocks`: a physical slot is reused
// only after its L1 *and* L2 consumers finished (empty counters), so live
// pool blocks stay bounded while logical pool offsets keep growing.
//
// The SF path transposes twice: the dispatch warps write SFs through
// `transform_sf_token_idx` (the UTCCP 4x32 transpose index map) so the GEMM's
// TMA loads them MN-major directly; the L1 epilogue writes its output SFs
// with the same transform for the L2 GEMM.
//
// Register economics: dispatch (48) / non-epilogue (40) warps donate registers
// to the epilogue warpgroups (208) via setmaxnreg — the epilogue holds both
// the TMEM drain and the combine accumulator set.
//
// Shared L1/L2 phases ("shared experts" = always-active dense experts):
// they bypass dispatch (input x is read directly), write their L1 output into
// a *non-ring* per-token buffer, and their L2 result lands in combine slot
// `kNumTopk` (a virtual extra top-k slot).
// ===========================================================================
//
// Deviations from upstream (documented, forced by this repo's contract):
//  * `kNumDeviceLocalityDomains = 8` and `kNumMaxRanks = 64` (frozen host
//    contract moe_layout.rs; upstream uses 2 / 72). `expert_recv_count_sum`
//    is sized [kNumMaxExperts] to match the frozen region formulas.
//  * `weight_dtype_t` -> `bool kIsWeightFP8` template parameter; the float
//    `kActivationClamp` template parameter -> `kActivationClampBits` (float
//    non-type template parameters are C++20; the JIT compiles with C++17).
//  * The epilogue's C++20 template-lambdas are namespace-scope template
//    functions (`mega_load_epi_block*`).

namespace dg {

// ===========================================================================
// 0. Local PTX helpers (not in prelude.h; NVRTC-safe)
// ===========================================================================

constexpr uint64_t kEvictFirstHint = 0x12f0000000000000ull;

// `barrier.sync` (unaligned form) vs prelude's `bar.sync`: the dispatch and
// epilogue regions meet at 384 threads (3 warpgroups) — arrival is
// warp-granular but the count is not a multiple of the CTA, so upstream uses
// the unaligned barrier opcode for those rendezvous.
DG_DEVICE void sync_unaligned(uint32_t num_threads, uint32_t barrier_idx) {
    asm volatile("barrier.sync %0, %1;" :: "r"(barrier_idx), "r"(num_threads) : "memory");
}
DG_DEVICE void sync_aligned(uint32_t num_threads, uint32_t barrier_idx) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_idx), "r"(num_threads) : "memory");
}

// Full cluster sync (aligned arrive) — prelude only has the relaxed flavor.
DG_DEVICE void cluster_arrive() { asm volatile("barrier.cluster.arrive;" ::: "memory"); }
DG_DEVICE void cluster_sync_full() { cluster_arrive(); cluster_wait(); }

// redux.sync wrappers (upstream uses the __reduce_*_sync intrinsics).
DG_DEVICE uint32_t reduce_add_sync_u32(uint32_t v) {
    uint32_t r;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}
DG_DEVICE uint32_t reduce_min_sync_u32(uint32_t v) {
    uint32_t r;
    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
    return r;
}
// Find the n-th set bit of `mask` (1-based), like __fns.
DG_DEVICE uint32_t fns_u32(uint32_t mask, uint32_t base, uint32_t offset) {
    uint32_t r;
    asm volatile("fns.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(mask), "r"(base), "r"(offset));
    return r;
}

// Wait for an mbarrier phase and flip the local phase bit (pull pipeline).
DG_DEVICE void mbarrier_wait_and_flip_phase(Barrier* bar, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "DG_MW_LOOP:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra DG_MW_DONE;\n\t"
        "bra DG_MW_LOOP;\n\t"
        "DG_MW_DONE:\n\t}"
        :: "r"(cvta_shared_to_u32(&bar->barrier_)), "r"(phase), "r"(0x989680u)
        : "memory");
    phase ^= 1;
}

// Arrive-with-expect-tx at a CLUSTER PEER's barrier (mapa-translated), used
// by the scheduler's task publish (upstream ClusterTransactionBarrier::
// arrive_and_expect_tx(tx, cta_id)).
DG_DEVICE void arrive_and_expect_tx_cluster(Barrier* bar, uint32_t tx_bytes, uint32_t cta_id) {
    const uint32_t remote = mapa_shared_cluster(&bar->barrier_, cta_id);
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote), "r"(tx_bytes) : "memory");
}

// 16-byte asynchronous store into a *cluster peer's* shared memory with
// mbarrier complete_tx signaling (the task-info publish path). Addresses are
// mapa-translated cluster addresses (u32 shared::cluster window).
DG_DEVICE void st_async_cluster_16b(uint32_t dst_smem_cluster_addr,
                                    uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                    uint32_t barrier_cluster_addr) {
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%0], {%1, %2, %3, %4}, [%5];"
        :: "r"(dst_smem_cluster_addr), "r"(a), "r"(b), "r"(c), "r"(d),
           "r"(barrier_cluster_addr)
        : "memory");
}

// 4D TMA load for a 2-CTA cluster (weights are 4D: K x N/2 x 2-die x expert).
DG_DEVICE void tma_load_4d_2sm(const TmaMap* map, Barrier* bar, void* smem,
                               uint64_t cache_hint,
                               uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :: "r"(cvta_shared_to_u32(smem)), "l"(map),
           "r"(cvta_shared_to_u32(&bar->barrier_)),
           "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(cache_hint)
        : "memory");
}

// `tma::copy` (2D, cta_group::2): split the box's inner extent into swizzle
// atoms; consecutive atoms land at `BLOCK_OUTER * atom_bytes` smem offsets.
DG_DEVICE void tma_copy_2d_2sm(const TmaMap* map, Barrier* bar, void* smem,
                               uint32_t block_inner, uint32_t block_outer,
                               uint32_t swizzle_mode, uint32_t wire_elem_bytes,
                               uint32_t inner_idx, uint32_t outer_idx) {
    const uint32_t inner_bytes = block_inner * wire_elem_bytes;
    const uint32_t atom_bytes = swizzle_mode == 0 ? inner_bytes : swizzle_mode;
    const uint32_t num_atoms = inner_bytes / atom_bytes;  // exact by construction
    const uint32_t atom_elems = atom_bytes / wire_elem_bytes;
    #pragma unroll 4
    for (uint32_t i = 0; i < num_atoms; ++i) {
        tma_load_2d_2sm(map, bar, (uint8_t*)smem + i * block_outer * atom_bytes,
                        kEvictNormalHint, inner_idx + i * atom_elems, outer_idx);
    }
}

// `tma::copy_nd` (4D, cta_group::2) for the weight tiles.
DG_DEVICE void tma_copy_4d_2sm(const TmaMap* map, Barrier* bar, void* smem,
                               uint32_t block_inner, uint32_t block_outer,
                               uint32_t swizzle_mode, uint32_t wire_elem_bytes,
                               uint32_t k_idx, uint32_t n_idx, uint32_t dom_idx, uint32_t expert_idx) {
    const uint32_t inner_bytes = block_inner * wire_elem_bytes;
    const uint32_t atom_bytes = swizzle_mode == 0 ? inner_bytes : swizzle_mode;
    const uint32_t num_atoms = inner_bytes / atom_bytes;
    const uint32_t atom_elems = atom_bytes / wire_elem_bytes;
    #pragma unroll 4
    for (uint32_t i = 0; i < num_atoms; ++i) {
        tma_load_4d_2sm(map, bar, (uint8_t*)smem + i * block_outer * atom_bytes,
                        kEvictNormalHint, k_idx + i * atom_elems, n_idx, dom_idx, expert_idx);
    }
}

// Lane exchange (upstream ptx::exchange) — works for any 4-byte type.
DG_DEVICE uint32_t exchange_u32(uint32_t v, uint32_t src_lane_idx) {
    return __shfl_sync(0xffffffff, v, (int)src_lane_idx);
}
DG_DEVICE float exchange_f32(float v, uint32_t src_lane_idx) {
    return __shfl_sync(0xffffffff, v, (int)src_lane_idx);
}

// Warp inclusive prefix sum (upstream math::warp_inclusive_sum).
DG_DEVICE uint32_t warp_inclusive_sum_u32(uint32_t value, uint32_t lane_idx) {
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t synced = __shfl_up_sync(0xffffffff, value, (int)offset);
        if (lane_idx >= offset) value += synced;
    }
    return value;
}

// Full-warp f32 max reduce (upstream math::warp_reduce<4, true, ReduceMax>:
// xor 1/2/4/8/16 — the intergroup-reduce flavor over 4-lane groups).
DG_DEVICE float warp_reduce_max_f32(float value) {
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, (int)offset));
    return value;
}

DG_DEVICE float fast_rcp_f32(float x) {
    float r;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
DG_DEVICE uint32_t hmin2_bf16x2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("min.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
DG_DEVICE uint32_t hmax2_bf16x2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
// Four f32 -> one packed e4m3x4 word (byte order a,b,c,d).
DG_DEVICE uint32_t fp8x4_e4m3(float a, float b, float c, float d) {
    const uint32_t lo = cvt_e4m3x2_f32(a, b);  // bytes (a, b)
    const uint32_t hi = cvt_e4m3x2_f32(c, d);  // bytes (c, d)
    return (lo & 0xffu) | ((lo & 0xff00u) << 8) | ((hi & 0xffu) << 16) | ((hi & 0xff00u) << 24);
}
// Fused BF16 -> FP32 accumulate (SM100 `add.rn.f32.bf16`), per half.
DG_DEVICE void accumulate_f32_bf16(float2& a, uint32_t b_bf16x2) {
    const uint16_t bx = (uint16_t)(b_bf16x2 & 0xffffu), by = (uint16_t)(b_bf16x2 >> 16);
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(bx));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(by));
}
// UE8M0 SF exponent for E4M3 quantization of an FP32 amax (upstream
// math::get_ue8m0_sf_exp<float, e4m3>): the integer carry of the mantissa
// addition performs the exponent ceiling onto a power of two that keeps
// |x * sf| <= 448, clamped to the denormal floor 2^-22.
DG_DEVICE uint32_t get_ue8m0_sf_exp_f32(float amax) {
    const uint32_t amax_bits = __float_as_uint(amax);
    const uint32_t rounded_exp = (amax_bits + 0x7fffffu - 0x600000u) >> 23;
    return dg_max(rounded_exp, 105u + 8u) - 8u;
}
DG_DEVICE float get_ue8m0_sf_inv_f32(uint32_t sf_exp) {
    return __uint_as_float((254u - sf_exp) << 23);
}
DG_DEVICE void red_add_i32(int32_t* p, int32_t v) {
    asm volatile("red.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
DG_DEVICE int32_t ld_acq_sys_i32(const int32_t* p) {
    int32_t v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// Device-side assert that never touches the host runtime (trap only).
#define DG_TRAP_ASSERT(cond) do { if (__builtin_expect(!(cond), 0)) asm volatile("trap;"); } while (0)

// ===========================================================================
// 1. Layout constants and the control block (`MegaMoESignals`)
// ===========================================================================
// The byte offsets below are the *frozen host-side contract* implemented by
// `deepgemm/src/moe_layout.rs` (`MegaMoESignalsLayout`); the constexpr
// functions mirror its formulas so compile wrappers can cross-check them.
// Frozen-contract deviations vs upstream (see file banner): 8 locality-domain
// task-count slots, 64 rank slots, expert_recv_count_sum sized [2048].

constexpr uint32_t kNumMaxGridSyncCounters = 4;
constexpr uint32_t kNumMaxExperts = 2048;
constexpr uint32_t kNumMaxExpertsPerRank = 512;  // runtime capacity assert (upstream array bound)
constexpr uint32_t kNumMaxRingBlocks = 1u << 20;  // 20 MiB of ring signals
constexpr uint32_t kNumMaxSharedL2Blocks = 1u << 15;  // 128 KiB
constexpr uint32_t kMoeNumMaxRanks = 64;
constexpr uint32_t kNumDeviceLocalityDomains = 8;

constexpr DG_DEVICE uint32_t moe_align_u32(uint32_t a, uint32_t b) { return (a + b - 1) / b * b; }

constexpr uint32_t kSignalsOffCombineReady = moe_align_u32(
    16u /*grid_sync_count[4]*/ + 4u /*nvl_barrier_counter*/ + 8u /*nvl_barrier_signals[2]*/
    + 4u * kNumDeviceLocalityDomains * 4u /*l1/l2/shared_l1/shared_l2 task counts*/,
    128u);
constexpr uint32_t moe_signals_off_peer_grid_idx(uint32_t num_ranks) {
    return moe_align_u32(kSignalsOffCombineReady + num_ranks * 8u, 128u);
}
constexpr uint32_t moe_signals_off_expert_send(uint32_t num_ranks) {
    return moe_signals_off_peer_grid_idx(num_ranks) + moe_align_u32(num_ranks * 8u, 128u);
}
constexpr uint64_t moe_signals_off_ring(uint32_t num_ranks) {
    return moe_signals_off_expert_send(num_ranks) + 3ull * kNumMaxExperts * 8ull;
}
constexpr uint64_t moe_signals_off_shared_l2(uint32_t num_ranks) {
    return moe_signals_off_ring(num_ranks) + (uint64_t)kNumMaxRingBlocks * 20ull;
}
constexpr uint64_t moe_signals_num_bytes(uint32_t num_ranks) {
    return moe_signals_off_shared_l2(num_ranks) + (uint64_t)kNumMaxSharedL2Blocks * 4ull;
}
// Shared-L2 input SF capacity (layout::get_num_max_shared_sf_tokens).
constexpr DG_DEVICE uint32_t moe_shared_sf_tokens(uint32_t num_max_tokens_per_rank) {
    return (num_max_tokens_per_rank + 7u) / 8u * 128u;
}

struct alignas(128) MegaMoESignals {
    // Grid and NVLink synchronization
    uint32_t grid_sync_count[kNumMaxGridSyncCounters];
    uint32_t nvl_barrier_counter;
    int nvl_barrier_signals[2];

    // Task scheduling (one counter per locality domain)
    uint32_t l1_task_count[kNumDeviceLocalityDomains];
    uint32_t l2_task_count[kNumDeviceLocalityDomains];
    uint32_t shared_l1_task_count[kNumDeviceLocalityDomains];
    uint32_t shared_l2_task_count[kNumDeviceLocalityDomains];

    // Combine readiness: `combine_ready_grid_idx[peer] == own grid index`
    // means the peer's L2 writes into this rank are done. Grid indices are
    // unique per launch, so no reset is needed; peers push theirs during
    // dispatch.
    alignas(128) uint64_t combine_ready_grid_idx[kMoeNumMaxRanks];
    alignas(128) uint64_t peer_grid_idx[kMoeNumMaxRanks];

    // Expert token counts (send: per-source; recv: per-destination;
    // recv_sum: (token_count | num_arrivals << 32) accumulator)
    alignas(128) uint64_t expert_send_count[kNumMaxExperts];
    uint64_t expert_recv_count[kNumMaxExperts];
    uint64_t expert_recv_count_sum[kNumMaxExperts];  // frozen-contract size (upstream: [kNumMaxExpertsPerRank])

    // Routed-expert ring signals; `l2_full_mask` has one bit per L1 N block
    uint32_t l1_full_count[kNumMaxRingBlocks];
    uint32_t l1_empty_count[kNumMaxRingBlocks];
    uint64_t l2_full_mask[kNumMaxRingBlocks];
    uint32_t l2_empty_count[kNumMaxRingBlocks];

    // Shared-expert signals
    uint32_t shared_l2_full_count[kNumMaxSharedL2Blocks];
};
// The struct must land exactly on the frozen moe_layout.rs formulas.
static_assert(kSignalsOffCombineReady == 256, "moe_layout.rs offset_combine_ready");
static_assert(sizeof(MegaMoESignals) == moe_signals_num_bytes(kMoeNumMaxRanks),
              "MegaMoESignals layout drift (frozen moe_layout.rs contract)");

// ===========================================================================
// 2. SymBuffer — NVLink symmetric-memory rank mapping (passed BY VALUE)
// ===========================================================================
// Byte layout, built Rust-side (`Args::raw_bytes`) and passed by value as a
// `__grid_constant__` kernel parameter. Mirrors upstream's SymBuffer, which
// maps an address on this rank to the same symmetric offset on `dst_rank`:
//
//   [0:4)     uint32_t rank_idx       (this rank)
//   [4:8)     padding                 (zero)
//   [8:8+8*N) uint64_t bases[N]       (per-rank symmetric buffer base)
template <uint32_t kNumRanks>
struct SymBuffer {
    uint32_t rank_idx;
    uint32_t pad_;
    uint64_t bases[kNumRanks];

    DG_DEVICE void* get_base_ptr() const { return reinterpret_cast<void*>(bases[rank_idx]); }

    // Map a pointer on this rank to the corresponding pointer on `dst_rank_idx`.
    template <typename ptr_t>
    DG_DEVICE ptr_t map(const ptr_t ptr, const uint32_t dst_rank_idx) const {
        if (kNumRanks == 1) return ptr;  // single rank: identity (degenerate path)
        const uint64_t mapped = bases[dst_rank_idx] - bases[rank_idx]
                              + reinterpret_cast<uint64_t>(ptr);
        return reinterpret_cast<ptr_t>(mapped);
    }
};

// ===========================================================================
// 3. Workspace / Buffer / MegaMoEBuffer (exact port of layout/mega_moe.cuh)
// ===========================================================================

struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

// Shared-expert token pool capacity (layout::get_num_max_pool_tokens).
constexpr DG_DEVICE uint32_t moe_pool_tokens(uint32_t num_ranks,
                                             uint32_t num_max_tokens_per_rank,
                                             uint32_t num_topk,
                                             uint32_t num_experts_per_rank) {
    const uint32_t num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const uint32_t num_max_experts_per_token = num_topk < num_experts_per_rank ? num_topk
                                                                                    : num_experts_per_rank;
    return moe_align_u32(num_max_recv_tokens * num_max_experts_per_token
                             + num_experts_per_rank * (240u - 1u),
                         1920u /*kLCMCandidateBlockM*/);
}

struct Workspace {
    MegaMoESignals* signals;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_shared_l2_pool_blocks;
    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    DG_DEVICE Workspace(void* base, uint32_t num_ranks_, uint32_t num_experts_,
                        uint32_t num_max_tokens_per_rank_, uint32_t num_topk,
                        uint32_t num_ring_tokens)
        : signals(reinterpret_cast<MegaMoESignals*>(base)),
          num_ranks(num_ranks_), num_experts(num_experts_),
          num_max_tokens_per_rank(num_max_tokens_per_rank_) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = moe_pool_tokens(num_ranks, num_max_tokens_per_rank,
                                              num_topk, num_experts_per_rank);
        num_shared_l2_pool_blocks = ceil_div_u32(num_max_tokens_per_rank, 8 /*kMinCandidateBlockM*/);
        DG_TRAP_ASSERT(num_ranks > 0);
        DG_TRAP_ASSERT(num_ranks <= kMoeNumMaxRanks);
        DG_TRAP_ASSERT(num_experts % num_ranks == 0);
        DG_TRAP_ASSERT(num_experts <= kNumMaxExperts);
        DG_TRAP_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
        DG_TRAP_ASSERT(num_ring_tokens <= kNumMaxRingBlocks * 8);
        DG_TRAP_ASSERT(num_shared_l2_pool_blocks <= kNumMaxSharedL2Blocks);
    }

    DG_DEVICE uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += sizeof(MegaMoESignals);
        // Source token-topk: [local expert][source rank][token]
        num_bytes += (uint64_t)num_experts * num_max_tokens_per_rank * 4ull;
        // Combine push source indices (full pool span)
        num_bytes += (uint64_t)num_max_pool_tokens * sizeof(TokenSrcMetadata);
        // Align to TMA descriptor requirements
        num_bytes = (num_bytes + 15ull) & ~15ull;
        return num_bytes;
    }

    DG_DEVICE void* get_end_ptr() const {
        return reinterpret_cast<uint8_t*>(signals) + get_num_bytes();
    }

    template <uint32_t kIndex>
    DG_DEVICE uint32_t* get_grid_sync_count_ptr() const {
        static_assert(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return signals->grid_sync_count + kIndex;
    }
    DG_DEVICE uint32_t* get_nvl_barrier_counter_ptr() const {
        return &signals->nvl_barrier_counter;
    }
    DG_DEVICE int* get_nvl_barrier_signal_ptr(uint32_t phase) const {
        // NOTES: the signal is signed, as we may minus
        return signals->nvl_barrier_signals + phase;
    }
    DG_DEVICE uint64_t* get_combine_ready_grid_idx_ptr(uint32_t peer_rank_idx) const {
        return signals->combine_ready_grid_idx + peer_rank_idx;
    }
    DG_DEVICE uint64_t* get_peer_grid_idx_ptr(uint32_t peer_rank_idx) const {
        return signals->peer_grid_idx + peer_rank_idx;
    }
    DG_DEVICE uint32_t* get_l1_task_count_ptr() const { return signals->l1_task_count; }
    DG_DEVICE uint32_t* get_l2_task_count_ptr() const { return signals->l2_task_count; }
    DG_DEVICE uint32_t* get_shared_l1_task_count_ptr() const { return signals->shared_l1_task_count; }
    DG_DEVICE uint32_t* get_shared_l2_task_count_ptr() const { return signals->shared_l2_task_count; }
    DG_DEVICE uint64_t* get_expert_send_count_ptr(uint32_t expert_idx) const {
        return signals->expert_send_count + expert_idx;
    }
    DG_DEVICE uint64_t* get_expert_recv_count_ptr(uint32_t rank_idx, uint32_t expert_idx) const {
        return signals->expert_recv_count + rank_idx * num_experts_per_rank + expert_idx;
    }
    DG_DEVICE uint64_t* get_expert_recv_count_sum_ptr(uint32_t expert_idx) const {
        return signals->expert_recv_count_sum + expert_idx;
    }
    DG_DEVICE uint32_t* get_l1_full_count_ptr(uint32_t ring_block_idx) const {
        return signals->l1_full_count + ring_block_idx;
    }
    DG_DEVICE uint32_t* get_l1_empty_count_ptr(uint32_t ring_block_idx) const {
        return signals->l1_empty_count + ring_block_idx;
    }
    DG_DEVICE uint64_t* get_l2_full_mask_ptr(uint32_t ring_block_idx) const {
        return signals->l2_full_mask + ring_block_idx;
    }
    DG_DEVICE uint32_t* get_l2_empty_count_ptr(uint32_t ring_block_idx) const {
        return signals->l2_empty_count + ring_block_idx;
    }
    DG_DEVICE uint32_t* get_shared_l2_full_count_ptr(uint32_t block_idx) const {
        return signals->shared_l2_full_count + block_idx;
    }
    // For dispatch pulling: [expert][source rank][token]
    DG_DEVICE uint32_t* get_src_token_topk_idx_ptr(uint32_t expert_idx, uint32_t rank_idx,
                                                   uint32_t token_idx) const {
        const uint64_t offset = ((uint64_t)expert_idx * num_ranks + rank_idx)
                                  * num_max_tokens_per_rank + token_idx;
        return reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(signals) + sizeof(MegaMoESignals)) + offset;
    }
    // For combine usages (full pool span). The metadata region begins where
    // the src-token-topk region of the first `num_experts_per_rank` experts
    // (= all of it) ends.
    DG_DEVICE TokenSrcMetadata* get_token_src_metadata_ptr(uint32_t pool_token_idx) const {
        auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank, 0, 0));
        return base + pool_token_idx;
    }
};

// layout::Data — one token's slice of a buffer region.
struct MoEData {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    DG_DEVICE MoEData(uint32_t num_bytes_, bool require_tma_alignment_ = true, void* base_ = nullptr)
        : num_bytes(num_bytes_), require_tma_alignment(require_tma_alignment_), base(base_) {
        DG_TRAP_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }
    DG_DEVICE void* get_base_ptr() const { return base; }
};

// layout::Buffer — `rows` ranks of `per_rank_rows` tokens of `data` bytes.
struct MoEBuffer {
    MoEData data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    void* base;

    DG_DEVICE MoEBuffer(const MoEData& data_layout_, uint32_t num_ranks_,
                        uint32_t num_max_tokens_per_rank_, void* base_ = nullptr)
        : data_layout(data_layout_), num_ranks(num_ranks_),
          num_max_tokens_per_rank(num_max_tokens_per_rank_), base(base_) {}

    DG_DEVICE uint64_t get_num_bytes_per_rank() const {
        return (uint64_t)num_max_tokens_per_rank * data_layout.num_bytes;
    }
    DG_DEVICE uint64_t get_num_bytes() const { return get_num_bytes_per_rank() * num_ranks; }
    DG_DEVICE void* get_base_ptr() const { return base; }
    DG_DEVICE void* get_end_ptr() const {
        return reinterpret_cast<uint8_t*>(base) + get_num_bytes();
    }
    DG_DEVICE MoEBuffer get_rank_buffer(uint32_t rank_idx) const {
        return MoEBuffer(data_layout, 1, num_max_tokens_per_rank,
                         reinterpret_cast<uint8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }
    DG_DEVICE MoEData get_data_buffer(uint32_t token_idx) const {
        DG_TRAP_ASSERT(num_ranks == 1);
        return MoEData(data_layout.num_bytes, data_layout.require_tma_alignment,
                       reinterpret_cast<uint8_t*>(base) + (uint64_t)data_layout.num_bytes * token_idx);
    }
};

// layout::MegaMoEBuffer — the whole symmetric buffer, region by region.
// (Construction order equals declaration order; every region begins where
// the previous one ends. `with_sf` is always true for the fp8xfp4 kernel.)
struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    MoEBuffer input_token_buffer, input_sf_buffer,
              input_topk_idx_buffer, input_topk_weights_buffer;

    // Shared expert buffers (shared L1 tokens reuse `input_token_buffer`)
    MoEBuffer shared_l1_token_buffer, shared_l1_sf_buffer,
              shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    MoEBuffer l1_token_buffer, l1_sf_buffer, l1_topk_weights_buffer,
              l2_token_buffer, l2_sf_buffer, combine_token_buffer;

    DG_DEVICE MegaMoEBuffer(void* base, uint32_t hidden, uint32_t intermediate_hidden,
                            uint32_t num_ranks, uint32_t num_experts,
                            uint32_t num_max_tokens_per_rank, uint32_t num_topk,
                            uint32_t num_ring_tokens, uint32_t num_sf_ring_tokens,
                            bool with_sf, uint32_t num_shared_experts) :
        workspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_ring_tokens),
        input_token_buffer(MoEData(hidden * (with_sf ? 1u : 2u)), 1, num_max_tokens_per_rank,
                           workspace.get_end_ptr()),
        input_sf_buffer(MoEData(with_sf ? hidden / 32 : 0, false), 1, num_max_tokens_per_rank,
                        input_token_buffer.get_end_ptr()),
        input_topk_idx_buffer(MoEData(num_topk * sizeof(int64_t), false), 1, num_max_tokens_per_rank,
                              with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr()),
        input_topk_weights_buffer(MoEData(num_topk * 4, false), 1, num_max_tokens_per_rank,
                                  input_topk_idx_buffer.get_end_ptr()),
        shared_l1_token_buffer(input_token_buffer),
        shared_l1_sf_buffer(MoEData(with_sf ? hidden / 32 : 0, false), 1,
                            num_shared_experts > 0 ? moe_shared_sf_tokens(num_max_tokens_per_rank) : 0,
                            input_topk_weights_buffer.get_end_ptr()),
        shared_l2_token_buffer(MoEData(intermediate_hidden * num_shared_experts * (with_sf ? 1u : 2u)), 1,
                               num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
                               with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr()),
        shared_l2_sf_buffer(MoEData(with_sf ? intermediate_hidden * num_shared_experts / 32 : 0, false), 1,
                            num_shared_experts > 0 ? moe_shared_sf_tokens(num_max_tokens_per_rank) : 0,
                            shared_l2_token_buffer.get_end_ptr()),
        l1_token_buffer(MoEData(hidden * (with_sf ? 1u : 2u)), 1, num_ring_tokens,
                        num_shared_experts > 0 ?
                            (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                            input_topk_weights_buffer.get_end_ptr()),
        l1_sf_buffer(MoEData(with_sf ? hidden / 32 : 0, false), 1, num_sf_ring_tokens,
                     l1_token_buffer.get_end_ptr()),
        l1_topk_weights_buffer(MoEData(4, false), 1, num_ring_tokens,
                               with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr()),
        l2_token_buffer(MoEData(intermediate_hidden * (with_sf ? 1u : 2u)), 1, num_ring_tokens,
                        l1_topk_weights_buffer.get_end_ptr()),
        l2_sf_buffer(MoEData(with_sf ? intermediate_hidden / 32 : 0, false), 1, num_sf_ring_tokens,
                     l2_token_buffer.get_end_ptr()),
        combine_token_buffer(MoEData(hidden * 2), num_topk + (num_shared_experts > 0 ? 1u : 0u),
                             num_max_tokens_per_rank,
                             with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr()) {}

    DG_DEVICE uint64_t get_num_bytes() const {
        return reinterpret_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
             - reinterpret_cast<uint8_t*>(workspace.signals);
    }
};

// ===========================================================================
// 4. Communication barriers (comm/barrier.cuh port)
// ===========================================================================

// 60s timeout at 2 GHz.
constexpr int64_t kMoeNumTimeoutCycles = 60ll * 2000000000ll;

// Spin until `pred()` holds; on timeout print and trap.
template <typename pred_t, typename print_timeout_t>
DG_DEVICE void moe_wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    const int64_t start_clock = clock64();
    while (!pred()) {
        if (clock64() - start_clock >= kMoeNumTimeoutCycles) {
            print_timeout();
            asm volatile("trap;");
        }
    }
}

// Slightly faster cluster sync with a relaxed arrive (weaker ordering).
DG_DEVICE void cluster_sync_with_relaxed_arrive() {
    cluster_arrive_relaxed();
    cluster_wait();
}

// Grid-wide barrier among `sync_scope`-sized thread scopes per SM (the
// cooperative-groups grid.sync protocol over a sense-tagged counter: SM 0
// contributes 0x80000000 - (num_sms - 1), everyone else +1; the finisher's
// carry flips the tag bit, which waiters observe via acquire loads).
template <uint32_t kNumSMs, uint32_t kGridSyncIndex, typename sync_scope_t>
DG_DEVICE void grid_sync(const Workspace& workspace, uint32_t sm_idx, uint32_t thread_idx,
                         const sync_scope_t& sync_scope) {
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const uint32_t old_value = atom_add_rel_u32(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1u);
        uint32_t new_value = 0;
        moe_wait_until([&]() { return (((new_value = ld_acq_u32(count_ptr)) ^ old_value) & kFinishSumTag) != 0; },
                       [&]() {
                           printf("DeepGEMM-RS grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u\n",
                                  sm_idx, thread_idx, kGridSyncIndex, old_value, new_value);
                       });
    }
    sync_scope();
}

// Cross-rank NVLink barrier (only SM 0 signals peers), bracketed by optional
// grid syncs. Degenerates correctly for kNumRanks == 1 (map is identity).
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
DG_DEVICE void nvlink_barrier(const Workspace& workspace, const SymBuffer<kNumRanks>& sym_buffer,
                              uint32_t sm_idx, uint32_t thread_idx, const sync_scope_t& sync_scope,
                              bool sync_prologue = true, bool sync_epilogue = true) {
    static_assert(kNumRanks <= kNumThreads, "Insufficient threads");

    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const uint32_t status = (*counter_ptr) & 3;
        const uint32_t signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks (signed: the count may decrease)
        if (thread_idx < kNumRanks)
            red_add_rel_sys_i32(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update the status and wait for all peers' signals
        if (thread_idx == 0) {
            red_add_u32(counter_ptr, 1u);
            const int target = signal_sign ? 0 : (int)kNumRanks;
            moe_wait_until([&]() { return ld_acq_sys_i32(signal_ptr) == target; }, [&]() {
                printf("DeepGEMM-RS NVLink barrier timeout: rank=%u, counter=%d, signal=%d, target=%d, phase=%u, sign=%u, tag=%u\n",
                       sym_buffer.rank_idx, *counter_ptr, ld_acq_sys_i32(signal_ptr),
                       target, signal_phase, signal_sign, kTag);
            });
        }
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

// ===========================================================================
// 5. Scheduler (scheduler/mega_moe.cuh port)
// ===========================================================================

// Minimal L1 warmup waves to ensure no L1 -> L2 deadlock (constexpr model —
// the Rust `api_mega_moe::get_num_l1_warmup_waves` mirrors this exactly).
constexpr DG_DEVICE int get_num_l1_warmup_waves(int num_total_m_blocks, int num_clusters,
                                                int num_l1_n_clusters, int num_l2_n_clusters) {
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = (num_clusters + num_l2_n_clusters - 1) / num_l2_n_clusters;
    const int num_l1_warmup_clusters_for_first_l2_wave =
        (num_first_l2_wave_m_blocks * num_l1_n_clusters + num_clusters - 1) / num_clusters;

    // Interleaved-schedule warmup: no L2 task of an M block may be scheduled
    // before that block's L1 tasks are issued; the last M block is the
    // bottleneck (its own L1 tasks plus the pending surplus of all preceding
    // blocks), plus one extra CTA-pair wave for partial-wave rounding.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule =
        (num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block
         + num_clusters - 1) / num_clusters + 1;

    return num_l1_warmup_clusters_for_first_l2_wave > num_warmup_waves_for_interleave_schedule
               ? num_l1_warmup_clusters_for_first_l2_wave
               : num_warmup_waves_for_interleave_schedule;
}

// Computation phase for the current block.
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    DG_DEVICE TaskInfo() : TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}
    DG_DEVICE TaskInfo(BlockPhase block_phase_, uint32_t local_expert_idx_, uint32_t m_block_idx_,
                       uint32_t pair_n_block_idx_, uint32_t pool_block_idx_, uint32_t valid_m_,
                       uint32_t shape_n_, uint32_t shape_k_)
        : block_phase(block_phase_), local_expert_idx(local_expert_idx_), m_block_idx(m_block_idx_),
          n_cluster_idx(pair_n_block_idx_), pool_block_idx(pool_block_idx_), valid_m(valid_m_),
          shape_n(shape_n_), shape_k(shape_k_) {}

    DG_DEVICE bool is_valid() const { return block_phase != BlockPhase::None; }
    DG_DEVICE uint32_t get_umma_aligned_valid_m() const { return (valid_m + 15u) & ~15u; }
    DG_DEVICE bool is_shared() const {
        return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
    }
};
static_assert(sizeof(TaskInfo<true>) == sizeof(TaskInfo<false>), "Invalid TaskInfo layout");

// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks
// can start as soon as they are fed. Mask parity alternates per ring
// generation (relied upon by the ring capacity bound).
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : ((1ull << kNumL1BlockNs) - 1);
    static_assert(kNumL1BlockNs <= 64, "Too many L1 N blocks for the L2 readiness mask");
    static_assert(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask;

    DG_DEVICE L2KBlockDependency(const uint64_t* mask_ptr_, uint32_t generation_idx)
        : mask_ptr(mask_ptr_), expected_mask(generation_idx & 1 ? 0ull : kFullMask),
          pending_mask(kFullMask) {}

    DG_DEVICE static void arrive(const uint64_t* mask_ptr, uint32_t n_block_idx) {
        red_xor_rel_u64(const_cast<uint64_t*>(mask_ptr), 1ull << n_block_idx);
    }

    // Re-read the mask only while the wanted K block is not fed yet.
    DG_DEVICE void wait(uint32_t k_block_idx) {
        const uint64_t k_block_mask =
            ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ld_acq_u64(mask_ptr) ^ expected_mask;
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks, uint32_t kNumLocalityDomains,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumExpertsPerLane = (kNumExpertsPerRank + 31) / 32,
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    static_assert(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    static_assert(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    static_assert(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    static_assert(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    static_assert(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    static_assert(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    static_assert(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    static_assert(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    // N block counts must be even so that 2 adjacent CTAs in a cluster always
    // land on the same m_block_idx with n_block_idx differing by 1.
    static_assert(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    static_assert(kNumRingBlocks > 0, "Invalid ring buffer config");
    static_assert(kNumLocalityDomains == 1 or kNumLocalityDomains == kNumDeviceLocalityDomains,
                  "Invalid locality domain count");
    static_assert(kNumL1Clusters % kNumLocalityDomains == 0 and
                  kNumL2Clusters % kNumLocalityDomains == 0,
                  "Each domain must take whole clusters");

    const Workspace& workspace;

    // Task-info smem double-buffering (full/empty mbarriers per slot)
    static constexpr uint32_t kNumScheduleStages = 2;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts: `stored_num_tokens_per_expert[i]`
    // holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane];
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all CTA-pair schedulers form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    // Locality-domain affinity with bounded work stealing (disabled when few
    // tasks, to avoid cross-die latency).
    static constexpr uint32_t kNumMinStealWaves = 2;
    uint32_t sm_locality_domain_idx = 0;

    DG_DEVICE MegaMoEScheduler(const Workspace& workspace_,
                               Barrier* task_info_full_barriers_,
                               Barrier* task_info_empty_barriers_,
                               task_info_t* task_infos_)
        : workspace(workspace_),
          task_info_full_barriers(task_info_full_barriers_),
          task_info_empty_barriers(task_info_empty_barriers_),
          task_infos(task_infos_) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) stored_num_tokens_per_expert[i] = 0;
    }

    DG_DEVICE void advance_sched_pipeline() {
        static_assert(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    DG_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    DG_DEVICE void release_task_info() const {
        // Complete metadata reads before the scheduler can overwrite this slot.
        fence_acq_rel_cta();
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive_cluster(0);
    }

    DG_DEVICE uint32_t get_num_tokens(uint32_t expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i)
            valid_value = (expert_idx == i * 32 + get_lane_idx()) ? stored_num_tokens_per_expert[i]
                                                                  : valid_value;
        return exchange_u32(valid_value, expert_idx % 32);
    }

    // Pool block offset of `expert_idx` from the per-lane token counts.
    DG_DEVICE uint32_t get_pool_block_offset(uint32_t expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i)
            if (i * 32 + get_lane_idx() < expert_idx)
                num_blocks += ceil_div_u32(stored_num_tokens_per_expert[i], BLOCK_M);
        return reduce_add_sync_u32(num_blocks);
    }

    DG_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    // Spin until every (SM x rank) dispatch pushed its expert counts, then
    // cache per-lane counts and compute the L1 warmup wave count.
    DG_DEVICE void fetch_expert_recv_count() {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            const uint32_t expert_idx = i * 32 + get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ld_vol_u64(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while ((uint32_t)(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = (uint32_t)value;
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = ceil_div_u32(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = (uint32_t)get_num_l1_warmup_waves(
            (int)num_total_m_blocks, (int)(kNumSMs / 2), (int)kNumL1Clusters, (int)kNumL2Clusters);
        num_sched_l1_waves = num_sched_l1_waves < min_l1_warmup_waves ? num_sched_l1_waves : min_l1_warmup_waves;
        num_sched_l1_waves = dg_min(num_sched_l1_waves, num_total_l1_waves);
    }

    // Resolve (m_block_idx -> owning expert, m offset, valid_m) from the
    // per-lane expert block counts via warp ballots and shuffles.
    DG_DEVICE task_info_t create_task(BlockPhase block_phase, uint32_t m_block_idx,
                                      uint32_t n_cluster_idx, uint32_t shape_n,
                                      uint32_t shape_k) const {
        const uint32_t lane_idx = get_lane_idx();
        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = ceil_div_u32(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = warp_inclusive_sum_u32(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);
            if (owner_mask) {
                const uint32_t owner_lane_idx = (uint32_t)(__ffs((int)owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = dg_min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = exchange_u32(expert_idx, owner_lane_idx);
                result.m_block_idx = exchange_u32(owner_m_block_idx, owner_lane_idx);
                result.valid_m = exchange_u32(owner_valid_m, owner_lane_idx);
            }
            block_offset += exchange_u32(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    DG_DEVICE uint32_t get_next_local_task_idx(uint32_t* task_count_ptr,
                                               uint32_t locality_domain_idx) const {
        uint32_t result = 0;
        if (elect_one_sync())
            result = atom_add_u32(task_count_ptr + locality_domain_idx, 1u);
        return exchange_u32(result, 0);
    }

    template <uint32_t kNumClusters>
    DG_DEVICE bool get_next_task_idx(uint32_t* task_count_ptr, uint32_t locality_domain_idx,
                                     uint32_t num_m_blocks, uint32_t& m_block_idx,
                                     uint32_t& n_cluster_idx) const {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
        return true;
    }

    template <uint32_t kNumClusters>
    DG_DEVICE bool get_next_task_idx(uint32_t* task_count_ptr, uint32_t num_m_blocks,
                                     uint32_t& m_block_idx, uint32_t& n_cluster_idx) const {
        if (get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx, num_m_blocks,
                                            m_block_idx, n_cluster_idx))
            return true;
        return kNumLocalityDomains > 1 and
               num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
               get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1,
                                               num_m_blocks, m_block_idx, n_cluster_idx);
    }

    // Interleaved L1/L2 task generation with L1 warmup waves (no L2 task for
    // an M block may run ahead of that block's L1 tasks).
    DG_DEVICE task_info_t get_next_task() {
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally one CTA-pair wave.
                --num_sched_l1_waves;
                if (!get_next_task_idx<kNumL1Clusters>(workspace.get_l1_task_count_ptr(),
                                                       num_total_m_blocks, m_block_idx, n_cluster_idx)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }
                return create_task(BlockPhase::Linear1, m_block_idx, n_cluster_idx,
                                   L1_SHAPE_N, L1_SHAPE_K);
            } else {
                if (!get_next_task_idx<kNumL2Clusters>(workspace.get_l2_task_count_ptr(),
                                                       num_total_m_blocks, m_block_idx, n_cluster_idx))
                    break;
                // The next task should be L1 again.
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;
                auto task_info = create_task(BlockPhase::Linear2, m_block_idx, n_cluster_idx,
                                              L2_SHAPE_N, L2_SHAPE_K);
                // Wait until all required L1 tasks are fetched from all queues.
                const uint32_t num_required_l1_tasks =
                    (task_info.pool_block_idx + 1) * (kNumL1Clusters / kNumLocalityDomains);
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++i)
                    while (ld_vol_u32(workspace.get_l1_task_count_ptr() + i) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    // Publish into the 2-slot smem ring: lanes 0/1 st.async the 32-byte task
    // into CTA 0/1 of the cluster, each with its own arrive+expect_tx.
    DG_DEVICE void publish_task(const task_info_t& task_info, uint32_t lane_idx) {
        if (lane_idx < 2) {
            arrive_and_expect_tx_cluster(&task_info_full_barriers[sched_stage_idx],
                                         sizeof(task_info_t), lane_idx);
            const uint32_t* src_words = reinterpret_cast<const uint32_t*>(&task_info);
            const uint32_t dst_addr = mapa_shared_cluster(task_infos + sched_stage_idx, lane_idx);
            const uint32_t bar_addr = mapa_shared_cluster(&task_info_full_barriers[sched_stage_idx], lane_idx);
            st_async_cluster_16b(dst_addr, src_words[0], src_words[1], src_words[2], src_words[3], bar_addr);
            st_async_cluster_16b(dst_addr + 16, src_words[4], src_words[5], src_words[6], src_words[7], bar_addr);
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    DG_DEVICE void shared_mainloop(uint32_t num_tokens, uint32_t lane_idx, uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        static_assert(kNumNClusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
        const uint32_t num_m_blocks = ceil_div_u32(num_tokens, BLOCK_M);
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            // Dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            if (!get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m_block_idx, n_cluster_idx))
                break;
            const uint32_t valid_m = dg_min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx,
                                     valid_m, kShapeN, kShapeK),
                         lane_idx);
        }
    }

    DG_DEVICE void mainloop(uint32_t num_tokens, const uint8_t* sm_locality_domains) {
        const uint32_t lane_idx = get_lane_idx();
        sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[get_sm_idx()];

        // Shared L2 tasks go before the routed tasks if one wave fits the
        // dispatch gap, else at the tail.
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2;
        const bool is_shared_l2_early =
            ceil_div_u32(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2;
        if (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait for dispatch results
        fetch_expert_recv_count();

        // Routed tasks: wait -> claim -> publish (claiming advances global
        // counters and must not run before the slot is released).
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if (kHasShared) {
            if (!is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

// ===========================================================================
// 6. Shared-memory layout model (constexpr mirror of the smem struct inside
//    the kernel; the Rust `api_mega_moe::smem_layout` implements the identical
//    math so compile wrappers can static_assert the two against each other).
// ===========================================================================
template <uint32_t kNumExperts, uint32_t kNumDispatchWarps, uint32_t kNumBytesPerPull,
          uint32_t kNumEpilogueWarpgroups, uint32_t kNumEpilogueWarps,
          uint32_t kNumTMAStoreStages, uint32_t STORE_BLOCK_M_L1, uint32_t L1_OUT_BLOCK_N,
          uint32_t STORE_BLOCK_M_L2, uint32_t BLOCK_N, uint32_t LOAD_BLOCK_N,
          uint32_t kNumStages, uint32_t LOAD_BLOCK_M, uint32_t BLOCK_K,
          uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N, uint32_t kNumScheduleStages>
struct MegaMoeSmemLayout {
    static constexpr uint32_t off_expert_token_count() { return 0; }
    static constexpr uint32_t off_dispatch_send_buffer() {
        return moe_align_u32(kNumExperts * 4, 1024);
    }
    static constexpr uint32_t cd_l1_bytes() {
        return kNumEpilogueWarpgroups * kNumTMAStoreStages * STORE_BLOCK_M_L1 * L1_OUT_BLOCK_N;
    }
    static constexpr uint32_t cd_l2_bytes() {
        return kNumEpilogueWarpgroups * STORE_BLOCK_M_L2 * BLOCK_N * 2;
    }
    static constexpr uint32_t off_smem_d() {
        return moe_align_u32(off_dispatch_send_buffer() + kNumDispatchWarps * kNumBytesPerPull, 1024);
    }
    static constexpr uint32_t off_smem_a() {
        return moe_align_u32(off_smem_d() + (cd_l1_bytes() > cd_l2_bytes() ? cd_l1_bytes() : cd_l2_bytes()), 1024);
    }
    static constexpr uint32_t off_smem_b() {
        return off_smem_a() + kNumStages * LOAD_BLOCK_M * BLOCK_K;
    }
    static constexpr uint32_t off_smem_sfa() {
        return off_smem_b() + kNumStages * LOAD_BLOCK_N * BLOCK_K;
    }
    static constexpr uint32_t off_smem_sfb() {
        return off_smem_sfa() + kNumStages * SF_BLOCK_M * (BLOCK_K / 128) * 4;
    }
    static constexpr uint32_t off_amax_reduction() {
        return moe_align_u32(off_smem_sfb() + kNumStages * SF_BLOCK_N * (BLOCK_K / 128) * 4, 8);
    }
    static constexpr uint32_t off_task_infos() {
        return moe_align_u32(off_amax_reduction() + kNumEpilogueWarps * (STORE_BLOCK_M_L1 / 2) * 8, 16);
    }
    static constexpr uint32_t off_dispatch_barriers() {
        return off_task_infos() + kNumScheduleStages * 32;
    }
    // Everything before the barrier arrays is reusable by the combine phase.
    static constexpr uint32_t reusable_bytes() { return off_dispatch_barriers(); }
    static constexpr uint32_t num_bytes() {
        uint32_t b = off_dispatch_barriers();
        b += (kNumDispatchWarps + kNumStages * 2 + 2 * 2 + kNumEpilogueWarps * 2 + kNumScheduleStages * 2) * 8;
        b += 4;  // tmem_ptr_in_smem
        return moe_align_u32(b, 1024);
    }
};

// ===========================================================================
// 7. Epilogue TMEM-drain helpers (upstream's C++20 template lambdas as
//    namespace-scope template functions).
// ===========================================================================
// NOTE: tensor memory addresses are simplified — the hardware ignores the
// warp index bits — and the `| 0x00100000` form addresses datapaths +16.
template <uint32_t kNumBuffers, uint32_t kNumAtomsPerStore, uint32_t ATOM_M,
          uint32_t UMMA_N, uint32_t WG_BLOCK_M>
DG_DEVICE void mega_load_epi_block(uint32_t s, uint32_t accum_stage_idx, uint32_t epilogue_wg_idx,
                                   uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M]) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumAtomsPerStore; ++i) {
        uint32_t (&values)[ATOM_M] = raw_values[s % kNumBuffers][i];
        const uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M
                                 + (s * kNumAtomsPerStore + i) * ATOM_M;
        tmem_load_16dp256b_x1(tmem_addr, values[0], values[1], values[2], values[3]);
        tmem_load_16dp256b_x1(tmem_addr | 0x00100000u, values[4], values[5], values[6], values[7]);
    }
}

// Release the TMEM accumulator stage back to the MMA warp (leader CTA).
DG_DEVICE void mega_release_tmem(Barrier* tmem_empty_barrier, uint32_t accum_stage_idx) {
    tcgen05_before_thread_sync();
    tmem_empty_barrier[accum_stage_idx].arrive_cluster(0);
}

template <uint32_t kNumBuffers, uint32_t kNumAtomsPerStore, uint32_t ATOM_M,
          uint32_t UMMA_N, uint32_t WG_BLOCK_M>
DG_DEVICE void mega_load_epi_block_if_valid(uint32_t s, uint32_t num_valid_store_blocks,
                                             uint32_t accum_stage_idx, uint32_t epilogue_wg_idx,
                                             uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M],
                                             Barrier* tmem_empty_barrier) {
    if (s < num_valid_store_blocks) {
        mega_load_epi_block<kNumBuffers, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
            s, accum_stage_idx, epilogue_wg_idx, raw_values);
        // The last valid block is waited for right away and the TMEM stage
        // released, as nothing else will be read from it.
        if (s + 1 == num_valid_store_blocks) {
            fence_view_async_tmem_load();
            mega_release_tmem(tmem_empty_barrier, accum_stage_idx);
        }
    }
}

// Prefetch the first `kNumBuffers - 1` store blocks (no wait); a warpgroup
// with no valid tokens releases the stage right away.
template <uint32_t kNumBuffers, uint32_t kNumAtomsPerStore, uint32_t ATOM_M,
          uint32_t UMMA_N, uint32_t WG_BLOCK_M>
DG_DEVICE void mega_prefetch_prologue(uint32_t num_valid_store_blocks, uint32_t accum_stage_idx,
                                      uint32_t epilogue_wg_idx,
                                      uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M],
                                      Barrier* tmem_empty_barrier) {
    #pragma unroll
    for (uint32_t s = 0; s + 1 < kNumBuffers; ++s)
        mega_load_epi_block_if_valid<kNumBuffers, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
            s, num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx, raw_values, tmem_empty_barrier);
    if (num_valid_store_blocks == 0)
        mega_release_tmem(tmem_empty_barrier, accum_stage_idx);
}

// Wait for block `s`'s loads, then issue block `s + kNumPrefetchStages` so it
// lands while block `s` is processed.
template <uint32_t kNumBuffers, uint32_t kNumAtomsPerStore, uint32_t ATOM_M,
          uint32_t UMMA_N, uint32_t WG_BLOCK_M>
DG_DEVICE void mega_wait_and_prefetch_next(uint32_t s, uint32_t num_valid_store_blocks,
                                           uint32_t accum_stage_idx, uint32_t epilogue_wg_idx,
                                           uint32_t (&raw_values)[kNumBuffers][kNumAtomsPerStore][ATOM_M],
                                           Barrier* tmem_empty_barrier) {
    constexpr uint32_t kNumPrefetchStages = kNumBuffers - 1;
    if (kNumPrefetchStages == 0)
        mega_load_epi_block_if_valid<kNumBuffers, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
            s, num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx, raw_values, tmem_empty_barrier);
    fence_view_async_tmem_load();
    if (kNumPrefetchStages > 0)
        mega_load_epi_block_if_valid<kNumBuffers, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
            s + kNumPrefetchStages, num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx,
            raw_values, tmem_empty_barrier);
}

// ===========================================================================
// 8. The megakernel
// ===========================================================================
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M_L1, uint32_t STORE_BLOCK_M_L2,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    uint32_t kActivationClampBits,   // f32 bits; 0x7f800000 (+inf) = no clamp
    bool kFastMath,
    bool kIsWeightFP8,               // false => FP4 weights (unpacked in SMEM)
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = moe_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = TaskInfo<kHasShared>
>
DG_GLOBAL void __launch_bounds__(kNumThreads, 1)
mega_moe_fp8_fp4_impl(void* y,
                      int32_t* cumulative_local_expert_recv_stats,
                      const uint32_t num_tokens,
                      const __grid_constant__ SymBuffer<kNumRanks> sym_buffer,
                      const __grid_constant__ TmaMap tensor_map_l1_acts,
                      const __grid_constant__ TmaMap tensor_map_l1_acts_sf,
                      const __grid_constant__ TmaMap tensor_map_l1_weights,
                      const __grid_constant__ TmaMap tensor_map_l1_weights_sf,
                      const __grid_constant__ TmaMap tensor_map_l1_output,
                      const __grid_constant__ TmaMap tensor_map_l2_acts,
                      const __grid_constant__ TmaMap tensor_map_l2_acts_sf,
                      const __grid_constant__ TmaMap tensor_map_l2_weights,
                      const __grid_constant__ TmaMap tensor_map_l2_weights_sf,
                      const __grid_constant__ TmaMap tensor_map_shared_l1_acts,
                      const __grid_constant__ TmaMap tensor_map_shared_l1_acts_sf,
                      const __grid_constant__ TmaMap tensor_map_shared_l1_weights,
                      const __grid_constant__ TmaMap tensor_map_shared_l1_weights_sf,
                      const __grid_constant__ TmaMap tensor_map_shared_l1_output,
                      const __grid_constant__ TmaMap tensor_map_shared_l2_acts,
                      const __grid_constant__ TmaMap tensor_map_shared_l2_acts_sf,
                      const __grid_constant__ TmaMap tensor_map_shared_l2_weights,
                      const __grid_constant__ TmaMap tensor_map_shared_l2_weights_sf,
                      const uint8_t* sm_locality_domains) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    // Use one domain when clusters cannot split evenly across device locality
    // domains (frozen-contract note: 8 slots; shapes must divide 8*2*BLOCK_N).
    constexpr uint32_t kNumLocalityDomains =
        L1_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 and
        L2_SHAPE_N % (BLOCK_N * 2 * kNumDeviceLocalityDomains) == 0 ? kNumDeviceLocalityDomains : 1;

    // Activation clamp (absent == +inf bits)
    constexpr bool kHasActivationClamp = kActivationClampBits != 0x7f800000u;

    // Template checks
    static_assert(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    static_assert(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    static_assert(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    static_assert(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = get_block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        prefetch_tma_map(&tensor_map_l1_acts);
        prefetch_tma_map(&tensor_map_l1_acts_sf);
        prefetch_tma_map(&tensor_map_l1_weights);
        prefetch_tma_map(&tensor_map_l1_weights_sf);
        prefetch_tma_map(&tensor_map_l1_output);
        prefetch_tma_map(&tensor_map_l2_acts);
        prefetch_tma_map(&tensor_map_l2_acts_sf);
        prefetch_tma_map(&tensor_map_l2_weights);
        prefetch_tma_map(&tensor_map_l2_weights_sf);
        prefetch_tma_map(&tensor_map_shared_l1_acts);
        prefetch_tma_map(&tensor_map_shared_l1_acts_sf);
        prefetch_tma_map(&tensor_map_shared_l1_weights);
        prefetch_tma_map(&tensor_map_shared_l1_weights_sf);
        prefetch_tma_map(&tensor_map_shared_l1_output);
        prefetch_tma_map(&tensor_map_shared_l2_acts);
        prefetch_tma_map(&tensor_map_shared_l2_acts_sf);
        prefetch_tma_map(&tensor_map_shared_l2_weights);
        prefetch_tma_map(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and buffer
    const auto buffer = MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependencyT = L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF and its buffer configs (UTCCP 128-element alignment)
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static_assert(SF_BLOCK_M == moe_align_u32(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    static_assert(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group:
    // index idx -> (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3), i.e. a
    // 32x4 <-> 4x32 transposition of the SF groups so the dispatch writes
    // and the GEMM's MN-major TMA loads agree with the UTCCP TMEM layout.
    const auto transform_sf_token_idx = [](uint32_t token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M
             + (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // MMA configs: always swap A/B (weights are the MMA's A), 2-CTA MMA,
    // matrices are K-major. FP4 weights unpack to 1 byte/elem in SMEM, so
    // A/B SMEM descriptors share addressing either way.
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_BLOCK_K = 128;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    static_assert(BLOCK_M % 16 == 0, "Invalid block M");
    static_assert(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    static_assert(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // Prefetch so the TMEM stage is released earlier (BLOCK_M == 240 only).
    constexpr uint32_t kNumEpiPrefetchStages = BLOCK_M == 240 ? 1 : 0;
    constexpr uint32_t ATOM_M = 8;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads;

    // FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU columns),
    // BF16 output for L2 (no TMA, a single stage).
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t AMAX_REDUCTION_WARP_BUFFER_SIZE = STORE_BLOCK_M_L1 / 2;  // float2

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) uint8_t l1[kNumEpilogueWarpgroups][kNumTMAStoreStages][STORE_BLOCK_M_L1 * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) uint8_t l2[kNumEpilogueWarpgroups][STORE_BLOCK_M_L2 * BLOCK_N * 2];
        } smem_d;
        alignas(kSharedMemoryAlignment) uint8_t smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) uint8_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
        uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
        float2 amax_reduction[kNumEpilogueWarps][AMAX_REDUCTION_WARP_BUFFER_SIZE];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier combine_barriers[kNumEpilogueWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
        uint32_t tmem_ptr_in_smem;
    };
    using SmemModel = MegaMoeSmemLayout<kNumExperts, kNumDispatchWarps, kNumBytesPerPull,
                                        kNumEpilogueWarpgroups, kNumEpilogueWarps,
                                        kNumTMAStoreStages, STORE_BLOCK_M_L1, L1_OUT_BLOCK_N,
                                        STORE_BLOCK_M_L2, BLOCK_N, LOAD_BLOCK_N, kNumStages,
                                        LOAD_BLOCK_M, BLOCK_K, SF_BLOCK_M, SF_BLOCK_N,
                                        kNumScheduleStages>;
    static_assert(sizeof(SharedStorage) == SmemModel::num_bytes(), "SharedStorage layout model drift");
    constexpr uint32_t kNumReusableSmemBytes = SmemModel::reusable_bytes();
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    const auto smem_send_buffers = MoEBuffer(
        MoEData(kNumBytesPerPull), kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    static_assert(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // A cluster sync is essential for 2CTA tensor memory allocation
    cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory (zero-fill the expert token counts)
        if (elect_one_sync())
            st_shared_bulk_zero(shared_storage.expert_token_count,
                                moe_align_u32(kNumExperts * 4u, kSharedMemoryAlignment));
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                // 4 arrivals: the A and B TMA warps of both CTAs each arrive
                // once at the leader's barrier (arrive_and_expect_tx /
                // arrive-cluster), while their cta_group::2 loads drop the
                // transaction bytes there.
                shared_storage.full_barriers[i].init(2 * 2);
                shared_storage.empty_barriers[i].init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                shared_storage.tmem_full_barriers[i].init(1);
                // Arrivals from both CTAs' epilogue threads at the leader.
                shared_storage.tmem_empty_barriers[i].init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        tmem_alloc_2sm(kNumTmemCols, &shared_storage.tmem_ptr_in_smem);
    }
    cluster_sync_full();

    // Wait for the primary kernel's completion (PDL)
    griddepcontrol_wait();

    // Task scheduler
    auto scheduler = MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks, kNumLocalityDomains,
        kNumSharedExperts>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        // Flip phases only when reaching the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Register reconfig economics: more experts per rank cost scheduler
    // registers, so only lean experts grant the epilogue extra registers.
    constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
    constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 72;
    constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 168;
    static_assert(kNumDispatchRegisters * kNumDispatchThreads +
                  kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                  kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                  "Too many registers");

    // Grid sync indices (dispatch and epilogue use separate counters)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        setmaxnreg_dec<kNumDispatchRegisters>();

        // ===================== Dispatch warps =====================
        static_assert(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // Each warp claims a strided range of tokens; within a warp, lane
            // l covers topk slot (l % kNumTopk) of token (base + l / kNumTopk),
            // whose flattened index is base * kNumTopk + l exactly.
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        reinterpret_cast<const int64_t*>(
                            buffer.input_topk_idx_buffer.get_base_ptr())[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](uint32_t /*token_topk_idx*/, int expert_idx) {
            atom_add_u32_block(shared_storage.expert_token_count + expert_idx, 1u);
        });
        sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offsets: pack (arrival tag << 32 | count) and swap the local
        // count for the global reservation (~6.5 us).
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | (uint64_t)shared_storage.expert_token_count[i];
            shared_storage.expert_token_count[i] =
                (uint32_t)atom_add_u64(workspace.get_expert_send_count_ptr(i), send_value);
        }
        sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens): each token-topk claims
        // its slot on the destination rank via a local atomic, then writes its
        // flattened token*topk+slot index into the destination's workspace.
        read_topk_idx([&](uint32_t token_topk_idx, int expert_idx) {
            const uint32_t dst_rank_idx = (uint32_t)expert_idx / kNumExpertsPerRank;
            const uint32_t dst_slot_idx = atom_add_u32_block(shared_storage.expert_token_count + expert_idx, 1u);
            auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                (uint32_t)expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); });

        // Write expert counts + push this launch's grid index tag (+1 so it
        // differs from a zeroed workspace) to every peer.
        if (sm_idx == 0) {
            static_assert(kNumRanks <= kNumDispatchThreads, "Insufficient threads for the grid index push");
            if (thread_idx < kNumRanks)
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx)
                    = get_grid_id() + 1;
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t dst_rank_idx = i / kNumExpertsPerRank;
                const uint32_t dst_local_expert_idx = i % kNumExpertsPerRank;
                const uint64_t expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                atom_add_u64_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx),
                                   dst_rank_idx),
                    expert_status);
            }
        }
        sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // NVLink barrier before pulling (all ranks' dispatch results visible)
        nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                       kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /*No grid sync prologue (one just ran above)*/ false,
            /*Grid sync epilogue after the NVLink barrier*/ true);

        // Ensure the epilogue barrier cannot run with the pull barrier
        sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // ====== Pull tokens + SFs from remote ranks into the ring ======
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for the current expert (reloaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = (kNumRanks + 31) / 32;
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane];
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++i) stored_rank_count[i] = 0;
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        // Wait for token counts to arrive
        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance the expert until the token index is within range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= (int)kNumExpertsPerRank)
                    break;
                // Update the pool block offset for the new expert
                expert_pool_block_offset += ceil_div_u32(expert_end_idx - expert_start_idx, BLOCK_M);
                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens((uint32_t)current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= (int)kNumExpertsPerRank)
                break;

            // Load per-rank counts when the expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        (uint32_t)*workspace.get_expert_recv_count_ptr(j, (uint32_t)current_expert_idx) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling: repeatedly
            // take min(count) tokens from every active rank so tokens from
            // all ranks interleave (balanced NVLink + pool locality).
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Active count and min across all ranks (per-lane first, then
                // one warp reduce)
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = dg_min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = reduce_add_sync_u32(num_actives_in_lane);
                const uint32_t length = reduce_min_sync_u32(min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + fns_u32(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                    remaining[i] -= dg_min(remaining[i], length);
            }

            // Read the source token-topk index (written by remote dispatch)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                (uint32_t)current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into TMA pull chunks
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
            static_assert(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");

            // TMA load the token from the remote rank and store into the local ring
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;

            // Wait for the ring slot to be available (previous consumers must
            // have finished all N blocks of the previous generation)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const uint32_t l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                auto empty_ptr = workspace.get_l1_empty_count_ptr(pool_block_idx % kNumRingBlocks);
                while (ld_acq_u32(empty_ptr) < l1_empty_count_target) {}
            }

            const auto src_base_ptr = sym_buffer.map(
                (const uint8_t*)buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                current_rank_in_expert_idx);
            const auto dst_base_ptr = (uint8_t*)buffer.l1_token_buffer
                .get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](uint32_t i) {
                mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                tma_store_1d(dst_base_ptr + i * kNumBytesPerPull,
                             pull_buffer.get_base_ptr(), kNumBytesPerPull, kEvictNormalHint);
                tma_store_arrive();
                tma_store_wait<0>();
            };
            if (elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++i) {
                    tma_load_1d(pull_buffer.get_base_ptr(),
                                src_base_ptr + i * kNumBytesPerPull,
                                pull_mbarrier, kNumBytesPerPull, kEvictFirstHint);
                    pull_mbarrier->arrive_and_expect_tx(kNumBytesPerPull);
                    if (i != kNumChunks - 1) issue_and_wait_pull_store(i);
                }
            }
            __syncwarp();

            // Load the topk weight first so its remote latency overlaps the
            // SF copy below
            const float weight = *sym_buffer.map(
                reinterpret_cast<const float*>(buffer.input_topk_weights_buffer.get_base_ptr()) + src_token_topk_idx,
                current_rank_in_expert_idx);

            // Load and store the SF (overlaps with the last chunk's TMA
            // load). The SF ring is written through `transform_sf_token_idx`
            // so the GEMM's MN-major SF TMA loads match the UTCCP TMEM layout.
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            static_assert(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                reinterpret_cast<const uint32_t*>(
                    buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr()),
                current_rank_in_expert_idx);
            auto local_sf_ptr = reinterpret_cast<uint32_t*>(buffer.l1_sf_buffer.get_base_ptr());
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
            const uint32_t sf_ring_token_idx = ring_block_idx * SF_BLOCK_M
                + transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < (kNumSFUint32 + 31) / 32; ++i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store the weight and metadata
            if (elect_one_sync()) {
                *reinterpret_cast<float*>(
                    buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr()) = weight;
                // Source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    TokenSrcMetadata{current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                // Complete the last chunk's store, then mark ring progress:
                // the last token of a block credits the padding remainder.
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                red_add_rel_u32(
                    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
                    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u);
            }
            __syncwarp();
        }

        // Clean the workspace for the next usage; also accumulate stats.
        // Overlaps with the combine reduction epilogue.
        sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        static_assert(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send counts and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumDeviceLocalityDomains; ++i) {
                    workspace.get_l1_task_count_ptr()[i] = workspace.get_l2_task_count_ptr()[i] = 0;
                    workspace.get_shared_l1_task_count_ptr()[i] = workspace.get_shared_l2_task_count_ptr()[i] = 0;
                }
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean ring blocks expert by expert
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read the expert token count before clearing
                const uint32_t num_recv_tokens = (uint32_t)
                    *workspace.get_expert_recv_count_sum_ptr(i);
                const uint32_t num_recv_m_blocks = ceil_div_u32(num_recv_tokens, BLOCK_M);

                // Compute the expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait for the count read to be ready
                sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean the expert token count and add cumulative results
                static_assert(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        red_add_i32(cumulative_local_expert_recv_stats + i, (int)num_recv_tokens);
                    __syncwarp();
                }

                // Clean per-rank token counts
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1/L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                       kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /*Grid sync prologue before the NVLink barrier*/ true,
            /*At the end of the kernel, no sync needed*/ false);
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        setmaxnreg_dec<kNumNonEpilogueRegisters>();

        // ===== GEMM TMA load warp for tokens (acts) with SFA =====
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const TmaMap* tensor_map_a_ptr = task_info.block_phase == BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                             task_info.block_phase == BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                             task_info.block_phase == BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                             /*SharedLinear2*/ &tensor_map_shared_l2_acts;
            const TmaMap* tensor_map_sfa_ptr = task_info.block_phase == BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                               task_info.block_phase == BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                               task_info.block_phase == BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                               /*SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const uint32_t num_k_blocks = ceil_div_u32(task_info.shape_k, BLOCK_K);

            // Pool block index for this expert (ring offset for routed tasks)
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait for the entire token arrival (L1) or shared-L2 readiness
            if (task_info.block_phase == BlockPhase::Linear1) {
                auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const uint32_t num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ld_acq_u32(ptr) != num_expected_tokens) {}
            } else if (task_info.block_phase == BlockPhase::SharedLinear2) {
                auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const uint32_t num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ld_acq_u32(ptr) != num_expected_blocks) {}
            }

            L2KBlockDependencyT l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx),
                                                       pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == BlockPhase::Linear2)
                    l2_k_block_dependency.wait(k_block_idx);

                // Wait for the consumer to release the stage
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Token offsets from the block index
                uint32_t m_idx = block_idx * BLOCK_M;
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfa_m_idx = block_idx * SF_BLOCK_M;
                const uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);

                // Add the 2-CTA offset for the non-leader CTA
                if (!is_leader_cta)
                    m_idx += task_info.get_umma_aligned_valid_m() / 2;

                // TMA copy tokens and SFA (cta_group::2, signals the leader's
                // barrier), then arrive at the full barrier
                if (elect_one_sync()) {
                    tma_copy_2d_2sm(tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx],
                                    shared_storage.smem_a[stage_idx],
                                    BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, 1 /*fp8: 1 byte/elem*/,
                                    k_idx, m_idx);
                    tma_copy_2d_2sm(tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx],
                                    shared_storage.smem_sfa[stage_idx],
                                    SF_BLOCK_M, 1, 0 /*no swizzle*/, 4 /*uint32 SF*/,
                                    sfa_m_idx, sfa_k_idx);
                    if (is_leader_cta) {
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(SharedStorage::smem_a[0]) * 2 + sizeof(SharedStorage::smem_sfa[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive_cluster(0);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        setmaxnreg_dec<kNumNonEpilogueRegisters>();

        // ===== GEMM TMA load warp for weights with SF =====
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const TmaMap* tensor_map_b_ptr = task_info.block_phase == BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                             task_info.block_phase == BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                             task_info.block_phase == BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                             /*SharedLinear2*/ &tensor_map_shared_l2_weights;
            const TmaMap* tensor_map_sfb_ptr = task_info.block_phase == BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                               task_info.block_phase == BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                               task_info.block_phase == BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                               /*SharedLinear2*/ &tensor_map_shared_l2_weights_sf;

            const uint32_t shape_k = task_info.shape_k;
            const uint32_t shape_n = task_info.shape_n;
            const uint32_t shape_sfb_k = ceil_div_u32(shape_k, kGranK * 4);
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const uint32_t num_k_blocks = ceil_div_u32(shape_k, BLOCK_K);

            // Interleaved gate/up weights: the gate half occupies N in
            // [0, half), the up half [half, 2*half); the 4D weights map puts
            // the two (die-localized) halves on coordinate 2.
            const uint32_t half_n = shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
            const uint32_t half_n_idx = n_idx - half_idx * half_n;
            const uint32_t expert_idx = task_info.is_shared() ? 0u : task_info.local_expert_idx;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait for the consumer to release the stage
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // Weight offsets
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                const uint32_t sfb_k_idx = task_info.is_shared()
                    ? k_block_idx * (BLOCK_K / 128)
                    : task_info.local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                // TMA copy weights with SF
                if (elect_one_sync()) {
                    tma_copy_4d_2sm(tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx],
                                    shared_storage.smem_b[stage_idx],
                                    BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, 1 /*1 byte/elem (fp8 or unpacked fp4)*/,
                                    k_idx, half_n_idx, half_idx, expert_idx);
                    tma_copy_2d_2sm(tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx],
                                    shared_storage.smem_sfb[stage_idx],
                                    BLOCK_N, 1, 0, 4,
                                    sfb_n_idx, sfb_k_idx);
                    if (is_leader_cta) {
                        constexpr uint32_t kNumWeightBytes =
                            sizeof(SharedStorage::smem_b[0]) * 2 / (kIsWeightFP8 ? 1 : 2);
                        shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            kNumWeightBytes + sizeof(SharedStorage::smem_sfb[0]) * 2);
                    } else {
                        shared_storage.full_barriers[stage_idx].arrive_cluster(0);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        setmaxnreg_dec<kNumNonEpilogueRegisters>();

        // ===== GEMM MMA issue warp (leader CTA only) =====
        if (is_leader_cta) {
            // Block-scaled instruction descriptors (always swap A/B: the
            // weight is the MMA's A operand, the activation its B).
            InstrDescriptorBlockScaled routed_instr_desc = make_instr_desc_bs(
                kIsWeightFP8 ? 0u /*E4M3*/ : 5u /*E2M1*/, 0u /*E4M3 acts*/,
                UMMA_M, UMMA_N, MAJOR_K, MAJOR_K);
            InstrDescriptorBlockScaled shared_instr_desc = make_instr_desc_bs(
                0u /*E4M3*/, 0u, UMMA_M, UMMA_N, MAJOR_K, MAJOR_K);
            SmemDescriptor sf_desc = make_sf_desc(nullptr);

            static_assert(kNumStages <= 32, "Too many stages");
            auto a_desc = make_umma_desc<MAJOR_K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode, 1, 1>(
                shared_storage.smem_a[0], 0, 0);
            auto b_desc = make_umma_desc<MAJOR_K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode, 1, 1>(
                shared_storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;

            static_assert((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                          (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                          (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                          "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            task_info_t task_info;
            while (scheduler.get_next_task(task_info)) {
                const uint32_t num_k_blocks = task_info.shape_k / BLOCK_K;

                // Dynamic UMMA N update from the effective M
                auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
                instr_desc.n_dim_ = task_info.get_umma_aligned_valid_m() >> 3;

                // Wait for the tensor memory empty barrier
                const uint32_t accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const uint32_t accum_phase = (current_iter_idx++ / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
                tcgen05_after_thread_sync();

                auto empty_barrier_arrive = [&](bool do_tmem_full_arrive) {
                    constexpr uint16_t kCTAMask = (1 << 2) - 1;
                    umma_arrive_2sm_multicast(&shared_storage.empty_barriers[stage_idx], kCTAMask);
                    // The TMEM accumulator pipeline has nothing to do with
                    // multicasting, but the commit must reach both CTAs.
                    if (do_tmem_full_arrive)
                        umma_arrive_2sm_multicast(&shared_storage.tmem_full_barriers[accum_stage_idx], kCTAMask);
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait for the TMA loads
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    tcgen05_after_thread_sync();

                    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(SharedStorage::smem_a[0]) / 16;
                    const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * sizeof(SharedStorage::smem_b[0]) / 16;
                    if (elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++umma_k_block_idx) {
                            // UTCCP copy SFA and SFB into TMEM (4x32dp128b warpx4)
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i) {
                                auto smem_ptr = shared_storage.smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                replace_smem_desc_addr(sf_desc, smem_ptr);
                                utccp_4x32dp128bit_2cta(sf_desc.desc_, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {
                                auto smem_ptr = shared_storage.smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                replace_smem_desc_addr(sf_desc, smem_ptr);
                                utccp_4x32dp128bit_2cta(sf_desc.desc_, kTmemStartColOfSFB + i * 4);
                            }

                            // Issue UMMA (swap AB: the B-desc is the MMA's A
                            // operand, and the SF columns swap with it)
                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++k) {
                                const uint64_t runtime_instr_desc =
                                    make_runtime_instr_desc_bs(instr_desc, k, k);
                                a_desc.lo = advance_umma_desc_lo<MAJOR_K, LOAD_BLOCK_M, kSwizzleAMode, 1, 1>(
                                    a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M, k * UMMA_K);
                                b_desc.lo = advance_umma_desc_lo<MAJOR_K, LOAD_BLOCK_N, kSwizzleBMode, 1, 1>(
                                    b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N, k * UMMA_K);
                                mma_mxf8f6f4_2sm(b_desc.desc_, a_desc.desc_,
                                                 accum_stage_idx * UMMA_N,
                                                 (k_block_idx > 0 or umma_k_block_idx > 0 or k > 0) ? 1u : 0u,
                                                 runtime_instr_desc,
                                                 kTmemStartColOfSFB, kTmemStartColOfSFA);
                            }
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier (implicit tcgen05 fence via commit)
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            }

            // To safely deconstruct the barriers, one more round of waits
            if (current_iter_idx > 0) {
                const uint32_t accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                shared_storage.tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages].wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        setmaxnreg_dec<kNumNonEpilogueRegisters>();

        // ===== Scheduler mainloop (leader CTA only) =====
        if (is_leader_cta)
            scheduler.mainloop(num_tokens, sm_locality_domains);
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        setmaxnreg_inc<kNumEpilogueRegisters>();

        // TMEM addresses are simplified — the hardware ignores the warp index
        // bits — and two CTAs never share an SM's tensor memory.
        DG_TRAP_ASSERT(ld_shared_u32(&shared_storage.tmem_ptr_in_smem) == 0);

        // ================ GEMM epilogue + combine warps ================
        const uint32_t epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg = epilogue_warp_idx % 4;
        static_assert((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                      kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // 2 warpgroups divide BM into BM/2; 4 warps divide BN into BN/4;
        // store blocks divide BM/2; atoms divide the store blocks.
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        static_assert(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        static_assert(WG_BLOCK_M % STORE_BLOCK_M_L1 == 0 and WG_BLOCK_M % STORE_BLOCK_M_L2 == 0,
                      "Invalid warpgroup block M");
        static_assert(STORE_BLOCK_M_L1 % ATOM_M == 0 and STORE_BLOCK_M_L2 % ATOM_M == 0,
                      "Invalid store block M");
        static_assert(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            // Wait for the UMMA arrival
            const uint32_t accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const uint32_t accum_phase = (current_iter_idx++ / kNumEpilogueStages) & 1;
            shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
            tcgen05_after_thread_sync();

            // Now we can release the task info slot
            scheduler.release_task_info();

            // Offsets (the shuffle tells the compiler warp divergence won't
            // happen)
            const uint32_t valid_m = exchange_u32(task_info.valid_m, 0);
            const uint32_t wg_start_m = epilogue_wg_idx * WG_BLOCK_M;
            const uint32_t num_valid_wg_rows = valid_m <= wg_start_m ? 0u : dg_min(valid_m - wg_start_m, WG_BLOCK_M);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring offset for reusable buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;  // Full-pool offset for metadata
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + (is_leader_cta ? 0u : 1u);
            const uint32_t n_idx = n_block_idx * BLOCK_N;

            Barrier* tmem_empty_barrier = &shared_storage.tmem_empty_barriers[accum_stage_idx];

            if (task_info.block_phase == BlockPhase::Linear1 or task_info.block_phase == BlockPhase::SharedLinear1) {
                if (!task_info.is_shared()) {
                    // Wait for the L2 blocks of this ring slot to be empty
                    auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const uint32_t num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ld_acq_u32(l2_empty_ptr) != num_expected_blocks) {}
                }

                // ---- Unified L1 epilogue: SwiGLU on granularity-8
                // interleaved gate/up weights, FP8 act-quant + SF out. With
                // SM100_TMEM_LOAD_16dp256b1x, gate/up pairs land in adjacent
                // registers. ----
                float stored_cached_weight = 1.0f;
                const float clamp = kHasActivationClamp ? __uint_as_float(kActivationClampBits) : 0.0f;

                constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M_L1 / ATOM_M;
                constexpr uint32_t kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M_L1;
                constexpr uint32_t kNumPrefetchStages = dg_min(kNumEpiPrefetchStages, kNumStoreBlocks - 1);
                uint32_t raw_values[kNumPrefetchStages + 1][kNumAtomsPerStore][ATOM_M];

                // Store blocks with at least one valid row (the last one may
                // be partial); <= kNumStoreBlocks by construction.
                const uint32_t num_valid_store_blocks = ceil_div_u32(num_valid_wg_rows, STORE_BLOCK_M_L1);
                mega_prefetch_prologue<kNumPrefetchStages + 1, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
                    num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx, raw_values, tmem_empty_barrier);

                #pragma unroll
                for (uint32_t s = 0; s < kNumStoreBlocks; ++s) {
                    if (s >= num_valid_store_blocks)
                        break;
                    mega_wait_and_prefetch_next<kNumPrefetchStages + 1, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
                        s, num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx, raw_values, tmem_empty_barrier);
                    uint32_t (&store_block_raw_values)[kNumAtomsPerStore][ATOM_M] =
                        raw_values[s % (kNumPrefetchStages + 1)];

                    // Iterate all atoms in the store block
                    float2 activation_values[kNumAtomsPerStore][2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load topk weights into the register cache per 32 tokens
                        static_assert(32 % ATOM_M == 0, "Invalid block size");
                        if (!task_info.is_shared() and (j * ATOM_M) % 32 == 0 and
                            (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *reinterpret_cast<const float*>(
                                buffer.l1_topk_weights_buffer
                                    .get_data_buffer(ring_m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                    .get_base_ptr());
                        }

                        // Fetch this lane's gate/up weight pair from the cache
                        const float weights_x = exchange_f32(
                            stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0);
                        const float weights_y = exchange_f32(
                            stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1);

                        // Apply SwiGLU: silu(gate) * up * w
                        auto fp32_values = reinterpret_cast<const float2*>(store_block_raw_values[i]);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++k) {
                            uint32_t gate_bf16 = cvt_bf16x2_f32(fp32_values[k * 2 + 0].x, fp32_values[k * 2 + 0].y);
                            uint32_t up_bf16 = cvt_bf16x2_f32(fp32_values[k * 2 + 1].x, fp32_values[k * 2 + 1].y);

                            // Clamp (bf16 domain, upstream __hmin2/__hmax2)
                            if (kHasActivationClamp) {
                                const uint32_t clamp2 = cvt_bf16x2_f32(clamp, clamp);
                                const uint32_t nclamp2 = cvt_bf16x2_f32(-clamp, -clamp);
                                gate_bf16 = hmin2_bf16x2(gate_bf16, clamp2);
                                up_bf16 = hmax2_bf16x2(up_bf16, nclamp2);
                                up_bf16 = hmin2_bf16x2(up_bf16, clamp2);
                            }
                            float gate_x = f32_from_bf16(gate_bf16 & 0xffffu);
                            float gate_y = f32_from_bf16(gate_bf16 >> 16);
                            const float up_x = f32_from_bf16(up_bf16 & 0xffffu);
                            const float up_y = f32_from_bf16(up_bf16 >> 16);
                            if (kFastMath) {
                                gate_x = gate_x * fast_rcp_f32(1.0f + __expf(-gate_x));
                                gate_y = gate_y * fast_rcp_f32(1.0f + __expf(-gate_y));
                            } else {
                                gate_x = gate_x / (1.0f + expf(-gate_x));
                                gate_y = gate_y / (1.0f + expf(-gate_y));
                            }
                            activation_values[i][k] = make_float2(gate_x * up_x * weights_x,
                                                                   gate_y * up_y * weights_y);
                        }

                        // Amax reduction (thread level)
                        float2 thread_local_amax = make_float2(0.f, 0.f);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++k) {
                            thread_local_amax.x = fmaxf(thread_local_amax.x, fabsf(activation_values[i][k].x));
                            thread_local_amax.y = fmaxf(thread_local_amax.y, fabsf(activation_values[i][k].y));
                        }

                        // Amax reduction (warp level)
                        amax_values[i].x = warp_reduce_max_f32(thread_local_amax.x);
                        amax_values[i].y = warp_reduce_max_f32(thread_local_amax.y);

                        // Exchange amaxes through smem (warp-pair level)
                        if (lane_idx < 4)
                            shared_storage.amax_reduction[epilogue_warp_idx][i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait for the previous TMA store to release the smem
                    // stage (also fences `amax_reduction`)
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    tma_store_wait<kNumTMAStoreStages - 1>();
                    sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++i) {
                        // Reduce amax (warp-pair level)
                        const float2 wp_amax =
                            shared_storage.amax_reduction[epilogue_warp_idx ^ 1][i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = fmaxf(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = fmaxf(amax_values[i].y, wp_amax.y);

                        // Calculate the SF pair
                        const uint32_t sf_exp_x = get_ue8m0_sf_exp_f32(amax_values[i].x);
                        const uint32_t sf_exp_y = get_ue8m0_sf_exp_f32(amax_values[i].y);
                        const float sf_inv_x = get_ue8m0_sf_inv_f32(sf_exp_x);
                        const float sf_inv_y = get_ue8m0_sf_inv_f32(sf_exp_y);

                        // Cast
                        const float2 upper = activation_values[i][0];
                        const float2 lower = activation_values[i][1];
                        const uint32_t fp8x4_values = fp8x4_e4m3(
                            upper.x * sf_inv_x, upper.y * sf_inv_y,
                            lower.x * sf_inv_x, lower.y * sf_inv_y);

                        // STSM (transposed b8; 64B-swizzled staging)
                        const uint32_t row = lane_idx;
                        const uint32_t col = warp_idx_in_wg;
                        auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx])
                            + i * ATOM_M * L1_OUT_BLOCK_N
                            + row * L1_OUT_BLOCK_N
                            // 64B swizzle for SwiGLU (divided by 2)
                            + (col ^ (row / 2)) * kNumBankGroupBytes;
                        stsm_x1_b8_trans(cvta_shared_to_u32(smem_ptr), fp8x4_values);

                        // Store the SF as UE8M0 (MN-major) into the L2 SF
                        // buffer. Only one warp per pair writes (both hold the
                        // same SF after the cross-warp reduce); each lane < 4
                        // holds the SF for a row pair.
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_sf_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_sf_idx / 4, byte_idx = k_sf_idx % 4;
                            const uint32_t mn_stride = (task_info.is_shared() ? kNumSharedSFTokens : kNumSFRingTokens) * 4u;
                            auto sf_base_ptr = reinterpret_cast<uint8_t*>(
                                task_info.is_shared() ? buffer.shared_l2_sf_buffer.get_base_ptr()
                                                      : buffer.l2_sf_buffer.get_base_ptr());
                            // Consecutive tokens (t, t+1) sit in the same
                            // 32-group, so the transformed indices differ by
                            // 4; `token_base_idx` is provably < BLOCK_M, so
                            // the transform decomposes (see upstream note).
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1 + i * ATOM_M;
                            const uint32_t sf_token_idx = block_idx * SF_BLOCK_M
                                + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const uint32_t sf_addr = k_uint_idx * mn_stride
                                + sf_token_idx * 4u + byte_idx;
                            sf_base_ptr[sf_addr] = (uint8_t)sf_exp_x;
                            sf_base_ptr[sf_addr + 4 * 4u] = (uint8_t)sf_exp_y;
                        }
                        __syncwarp();
                    }
                    sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue the TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        const TmaMap* tensor_map_l1_output_ptr =
                            task_info.is_shared() ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                        tma_store_fence();
                        tma_store_2d(tensor_map_l1_output_ptr,
                                     shared_storage.smem_d.l1[epilogue_wg_idx][tma_stage_idx],
                                     out_n_idx,
                                     m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L1);
                        tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2 and increment the L1 empty count
                tma_store_wait<0>();
                sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and elect_one_sync()) {
                    if (task_info.is_shared()) {
                        red_add_rel_u32(workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        // Toggle this N block's bit in the L2 readiness mask
                        L2KBlockDependencyT::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);
                        // Increment the L1 empty count for this physical slot
                        red_add_u32(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
            } else {
                // Increment the L2 empty count for this physical slot
                if (!task_info.is_shared()) {
                    if (epilogue_warp_idx == 0 and elect_one_sync())
                        red_add_u32(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    __syncwarp();
                }

                constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M_L2 / ATOM_M;
                constexpr uint32_t kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M_L2;
                constexpr uint32_t kNumPrefetchStages = dg_min(kNumEpiPrefetchStages, kNumStoreBlocks - 1);
                uint32_t raw_values[kNumPrefetchStages + 1][kNumAtomsPerStore][ATOM_M];

                const uint32_t num_valid_store_blocks = ceil_div_u32(num_valid_wg_rows, STORE_BLOCK_M_L2);
                mega_prefetch_prologue<kNumPrefetchStages + 1, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
                    num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx, raw_values, tmem_empty_barrier);

                // ---- L2 BF16 epilogue: GEMM output into remote combine
                // buffers via NVLink (the weighted top-k reduce happens in
                // the combine phase) ----
                #pragma unroll
                for (uint32_t s = 0; s < kNumStoreBlocks; ++s) {
                    if (s >= num_valid_store_blocks)
                        break;
                    mega_wait_and_prefetch_next<kNumPrefetchStages + 1, kNumAtomsPerStore, ATOM_M, UMMA_N, WG_BLOCK_M>(
                        s, num_valid_store_blocks, accum_stage_idx, epilogue_wg_idx, raw_values, tmem_empty_barrier);
                    uint32_t (&store_block_raw_values)[kNumAtomsPerStore][ATOM_M] =
                        raw_values[s % (kNumPrefetchStages + 1)];

                    // Read the source metadata of this warp's rows (2 per
                    // atom) before the smem stores, to overlap the latency
                    TokenSrcMetadata cached_src_metadata[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumAtomsPerStore; ++j) {
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2
                                                      + j * ATOM_M + warp_idx_in_wg * 2 + lane_idx / 16;
                        if (m_idx_in_block < valid_m)
                            cached_src_metadata[j] = task_info.is_shared()
                                ? TokenSrcMetadata{sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk}
                                : *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                    }
                    __syncwarp();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++i) {
                        // Wait for the previous NVLink store to release the
                        // smem (skip the first block: the full barrier above
                        // already ensured completion)
                        if (i == 0 and s > 0)
                            sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Store into shared memory (2 warps share a BF16
                        // swizzle atom; each lane provides its own address)
                        uint32_t (&values)[ATOM_M] = store_block_raw_values[i];
                        const uint32_t row = lane_idx % 8;
                        const uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        stsm_x4_trans(cvta_shared_to_u32(smem_ptr),
                                      cast_bf16_and_pack(values[0], values[1]),
                                      cast_bf16_and_pack(values[2], values[3]),
                                      cast_bf16_and_pack(values[4], values[5]),
                                      cast_bf16_and_pack(values[6], values[7]));
                    }

                    // Wait for the shared memory stores
                    sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers: each warp writes 2 rows
                    // (lane/16 splits the warp into two halves, one per row)
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumAtomsPerStore; ++j) {
                        const uint32_t row_in_store = j * ATOM_M + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M_L2 + row_in_store;

                        // Skip padding rows beyond the actual token count
                        if (m_idx_in_block >= valid_m)
                            break;

                        const TokenSrcMetadata& meta = cached_src_metadata[j];

                        // Read from shared memory (128-bit)
                        auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d.l2[epilogue_wg_idx]) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M_L2 * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const uint4 packed = ld_shared_u128(reinterpret_cast<const uint32_t*>(smem_ptr));

                        // Write into the remote combine buffer (each lane one
                        // float4 = 16B slice of the token's hidden row)
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(meta.topk_idx)
                                                   .get_data_buffer(meta.token_idx);
                        auto dst_ptr = reinterpret_cast<float4*>(reinterpret_cast<uint8_t*>(dst_token.get_base_ptr())
                            + n_idx * 2u /*sizeof(bf16)*/ + (lane_idx % 16) * 16u);
                        *sym_buffer.map(dst_ptr, meta.rank_idx) = packed;
                    }
                }

                // Ensure the next epilogue can safely use the shared memory
                sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }

        // Deallocate tensor memory (same logical warp ID on both CTAs)
        if (epilogue_warp_idx == 0)
            tmem_dealloc_2sm(0, kNumTmemCols);

        // ============ Combine: reduce top-k results and write back ============
        // Reuses the shared memory from the start up to the barriers.
        constexpr uint32_t kNumHiddenBytes = kHidden * 2;
        constexpr uint32_t kNumElemsPerUint4 = 16 / 4;  // sizeof(uint4)/sizeof(bf16x2)

        // 3 chunk slots: 2 load stages + 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // Either 1 or 2 chunks for simplicity (smem- and register-bound)
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= kNumReusableSmemBytes and
            kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / 16;
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        static_assert(kHidden % kNumChunks == 0, "Hidden must be divisible by the number of chunks");
        static_assert(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes,
                      "Hidden is too large");
        static_assert(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        static_assert(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        static_assert(kNumTopk + (kNumSharedExperts > 0 ? 1u : 0u) <= 32u,
                      "Top-k + shared must fit in a single warp");

        DG_TRAP_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= kNumReusableSmemBytes);

        // Per-warp buffers: 2 load stages + 1 store
        const auto combine_load_buffer = [&](uint32_t i) {
            return reinterpret_cast<uint4*>(smem_buffer + (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        };
        const auto combine_store_buffer = reinterpret_cast<uint4*>(
            smem_buffer + (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp load barriers
        const auto combine_load_barrier = [&](uint32_t i) -> Barrier* {
            return &shared_storage.combine_barriers[i + epilogue_warp_idx * 2];
        };

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Peers' grid indices for tagging their combine readiness; the load
        // overlaps with the grid sync.
        static_assert(kNumRanks <= kNumEpilogueThreads, "Insufficient threads for combine readiness");
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(epilogue_thread_idx);

        // All local L2 writes are done after this grid sync
        grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });

        // Notify remote ranks (ordered before the cleanup barrier by the
        // dispatch/epilogue sync below)
        if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
            st_rel_sys_u64(
                sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), epilogue_thread_idx),
                peer_grid_idx);

        // Barrier with the dispatch warps so they can clean the workspace
        sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Iterate over all token chunks (1 token 1 topk latency: ~3 us)
        const uint64_t grid_idx = get_grid_id() + 1;
        for (uint32_t token_chunk_idx = epilogue_warp_idx * kNumSMs + sm_idx;
             token_chunk_idx < num_tokens * kNumChunks;
             token_chunk_idx += kNumSMs * kNumEpilogueWarps) {
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;

            // Read top-k slot indices: each lane reads one slot; the shared
            // expert occupies the virtual slot right after top-k
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(reinterpret_cast<const int64_t*>(
                    buffer.input_topk_idx_buffer.get_base_ptr())[token_idx * kNumTopk + lane_idx]) :
                (kNumSharedExperts > 0 and lane_idx == kNumTopk ? (int)kNumTopk : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Wait for the ranks of the selected experts to finish their L2
            // writes (peer grid-idx readiness tags)
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
            auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? (uint32_t)stored_topk_slot_idx / kNumExpertsPerRank : 0);
            moe_wait_until([&]() {
                return __all_sync(0xffffffff, !is_routed or ld_acq_sys_u64(peer_ready_ptr) == grid_idx);
            }, [&]() {
                printf("DeepGEMM-RS combine peers timeout: rank=%u, token=%u\n",
                       sym_buffer.rank_idx, token_idx);
            });

            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;

            // Move the mask and load: the shared slot goes first, then routed
            // slots in top-k order.
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](uint32_t i) -> bool {
                if (mask) {
                    const uint32_t slot_idx = (mask >> kNumTopk) ? kNumTopk : (uint32_t)(__ffs((int)mask) - 1);
                    mask ^= 1u << slot_idx;

                    if (elect_one_sync()) {
                        auto src_ptr = reinterpret_cast<const uint8_t*>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                .get_data_buffer(token_idx).get_base_ptr()) + chunk_byte_offset;
                        tma_load_1d(combine_load_buffer(i), src_ptr, combine_load_barrier(i),
                                    kNumChunkBytes, kEvictFirstHint);
                        combine_load_barrier(i)->arrive_and_expect_tx(kNumChunkBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            // Load the first selection
            bool do_reduce = move_mask_and_load(load_stage_idx);

            // Accumulate all top-k contributions in float registers
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4];
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane * kNumElemsPerUint4; ++j) reduced[j] = make_float2(0.f, 0.f);
            while (do_reduce) {
                // Prefetch the next top-k into the other buffer while the
                // current one accumulates
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                // Accumulate
                combine_load_barrier(load_stage_idx)->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                    const uint4 uint4_values = combine_load_buffer(load_stage_idx)[j * 32 + lane_idx];
                    const uint32_t* bf16_values = reinterpret_cast<const uint32_t*>(&uint4_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)
                        accumulate_f32_bf16(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                }
                fence_view_async_shared();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

            // Cast to bf16 and write out
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                uint4 casted;
                uint32_t* casted_bf16 = reinterpret_cast<uint32_t*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)
                    casted_bf16[l] = cvt_bf16x2_f32(reduced[j * kNumElemsPerUint4 + l].x,
                                                    reduced[j * kNumElemsPerUint4 + l].y);

                // Wait for the shared memory release and write
                if (j == 0) {
                    tma_store_wait<0>();
                    __syncwarp();
                }
                st_shared_u128(reinterpret_cast<uint32_t*>(combine_store_buffer + j * 32 + lane_idx), casted);
            }
            __syncwarp();

            // TMA store the token chunk
            if (elect_one_sync()) {
                tma_store_fence();
                tma_store_1d(reinterpret_cast<uint8_t*>(y)
                                 + (uint64_t)token_idx * kNumHiddenBytes + chunk_byte_offset,
                             combine_store_buffer, kNumChunkBytes, kEvictNormalHint);
                tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        asm volatile("trap;");  // sm_100a only
#endif
}

} // namespace dg
