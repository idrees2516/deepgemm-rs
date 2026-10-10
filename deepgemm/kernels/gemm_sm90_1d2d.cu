// ===========================================================================
// gemm_sm90_1d2d.cu — Hopper (SM90a) FP8 GEMM with 1D2D fine-grained scaling.
// Ported 1:1 from upstream DeepGEMM `impls/sm90_fp8_gemm_1d2d.cuh`
// (launch configs: csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp).
//
// ---------------------------------------------------------------------------
// CONCEPT 1 — WHAT "1D2D" SCALING MEANS (and why it exists)
// ---------------------------------------------------------------------------
// Two different tensors get quantized to FP8 with *different* scale
// granularities, because their statistics differ:
//
//   A (activations): one FP32 scale PER TOKEN per 128-K slice  — "1D"
//                    sfa layout: [m, k/128]     (a 1D row of scales per token)
//   B (weights):     one FP32 scale PER 128x128 BLOCK           — "2D"
//                    sfb layout: [n/128, k/128] (a 2D grid of scale blocks)
//
//   D[m][n] = sum_kb  sfa[m][kb] * sfb[n/128][kb] * ( A[m, kb*128:(kb+1)*128]
//                                                   . B[n, kb*128:(kb+1)*128] )
//
// Activations have wildly varying per-token magnitudes, so they need the
// finer per-token (1D) granularity; weights are statistically stationary, so
// a coarse per-128x128-block (2D) grid suffices and keeps the weight scale
// tensor tiny. This is the DeepSeek-V3 training-kernel recipe (the "1d2d"
// GEMM in DeepGEMM); the inference recipe on B200 instead uses 1d1d with
// UE8M0 scales folded into tcgen05 hardware (see gemm_sm100.cu).
//
// ---------------------------------------------------------------------------
// CONCEPT 2 — WHY *FP32* SCALE FACTORS ON SM90 (vs UE8M0 on SM100)
// ---------------------------------------------------------------------------
// Hopper's tensor core (wgmma) has NO block-scaled MMA: it cannot apply
// per-128-K scales itself. So the kernel must "promote" in software:
//
//     wgmma (fp8 e4m3 dot-product over one K=128 slice) -> FP32 accumulators
//     final[m][n] += sfa[m][kb] * sfb[n/128][kb] * accum[m][n]   (registers)
//
// The scales are therefore kept as plain FP32 (full precision, software
// applied). SM100's tcgen05.mma kind::mxf8f6f4 instead multiplies UE8M0
// scales in hardware through a TMEM scale-factor path — two generations,
// one idea, different machinery. Because the promotion runs between
// `wgmma.wait_group` and the next stage, it sits on the critical path; the
// register budget (248/232 regs for math warps) leaves room for exactly one
// accumulator tile + one promoted tile.
//
// ---------------------------------------------------------------------------
// CONCEPT 3 — TWO COMPLETELY DIFFERENT SF TRANSPORT PATHS (the 1d2d trick)
// ---------------------------------------------------------------------------
//   * SFA (1D) rides the TMA pipeline: one [BLOCK_M, 1] FP32 box per K
//     stage, multicast together with the A tile it scales (same m-block!).
//     It changes every stage -> must be pipelined like A/B.
//
//   * SFB (2D) does NOT use TMA at all. The MATH warps preload it straight
//     from GLOBAL memory into a tiny SMEM staging array ONCE per output
//     block (it is constant across the whole K sweep of that block):
//
//         smem_sfb[0 .. k_blocks)            = sfb[n0/128][0 .. k_blocks)
//         smem_sfb[k_blocks .. 2*k_blocks)   = sfb[n0/128 + 1][0 .. k_blocks)
//
//     (two "rows" because a tile may straddle a 128-column scale boundary —
//     see CONCEPT 4). Volume: k_blocks * <=2 floats (a few hundred bytes),
//     read once per persistent block — orders of magnitude less traffic than
//     SFA. Keeping it off the TMA path frees producer bandwidth and needs no
//     descriptor gymnastics for the 2D layout. The preload is spread over
//     all math warps EXCEPT warp 0 ("except the first warp, we want to
//     overlap loading B scales with TMA stores between tasks" — warp 0 is
//     the one committing epilogue TMA stores) and synced with named
//     barrier 0 before the WGMMA loop starts.
//
// ---------------------------------------------------------------------------
// CONCEPT 4 — THE B-SCALE STRADDLE (num_former_iters predicate)
// ---------------------------------------------------------------------------
// A [BLOCK_N]-wide B tile spans 1 or 2 of the 128-wide sfb column blocks:
//
//        n:  0        128       256
//            | sfb[0] | sfb[1] | ...        BLOCK_N=192 tile starting at
//            [  former  |  remainder ]       n=64: cols 64..127 use sfb[0],
//                                             cols 128..255 use sfb[1]
//
// WGMMA accumulator column group i (8 columns) uses
//     sfb row 0   iff  i <  num_former_iters
//     sfb row 1   iff  i >= num_former_iters
// where num_former_iters = min(BLOCK_N, 128 - (n0 % 128)) / 8 is the number
// of 8-column groups inside the FIRST scale block. `kMustUseUniformedScaleB`
// (BLOCK_K % BLOCK_N == 0, i.e. BLOCK_N divides 128) short-circuits this to
// a single row. Because num_former_iters is runtime, the kernel dispatches
// on it with a compile-time ladder (dispatch_num_former_iters) so the
// unrolled promotion loop sees a CONSTANT predicate — the reason the
// promotion stays branch-free (upstream: "making it as predicates is very
// important for performance").
//
// ---------------------------------------------------------------------------
// CONCEPT 5 — PIPELINE + WARP SPECIALIZATION (shared with the 1d1d kernel)
// ---------------------------------------------------------------------------
//   threads [0, kNumMathThreads)         math warpgroup(s): SFB preload,
//                                        wgmma + register promotion, STSM
//                                        epilogue + TMA stores
//   threads [kNumMathThreads, +128)      TMA warpgroup: warp +2 (the third)
//                                        issues A/SFA/B loads
//
//   producer (1 elected lane)                math warpgroup(s)
//   ─────────────────────────────            ─────────────────────────────
//   wait empty[stage] (phase^1)              read sfb[kb] (SMEM, stage-free)
//   TMA A  [BLOCK_K, BLOCK_M] @ (k, m)       wait full[stage] (phase)
//   TMA SFA[BLOCK_M, 1] @ (m, kb)            read sfa rows r0/r1
//   TMA B  [BLOCK_K, BLOCK_N] @ (k, n)       wgmma x4 (K=128 = 4 x k32)
//   expect_tx(A+B+SFA bytes)                 wait_group<0>; release empty[stage]
//                                             promote: final += sfa*sfb*accum
//
//   * kNumStages-deep ring of (full, empty) mbarrier pairs; phases flip on
//     wrap. Empty barriers expect kNumTMAMulticast * math_warps arrivals
//     (each cluster CTA's math warps answer BOTH CTAs' barriers — lane 0
//     -> CTA0, lane 1 -> CTA1; if the peer CTA already exited, both lanes
//     arrive locally to keep the count).
//   * Multicast (cluster of 2): ONE `cp.async.bulk.tensor...multicast`
//     issued by cluster rank 0 credits BOTH CTAs' full barriers; each CTA
//     still expect_tx on its own barrier. A and SFA are multicast when
//     kIsTMAMulticastOnA (cluster_n = 2), otherwise B is.
//   * setmaxnreg: TMA warps donate down to 40 registers, math warps take
//     248 (one math WG) / 232 (two math WGs).
//   * PDL: griddepcontrol.wait before the persistent loop.
//
// ---------------------------------------------------------------------------
// CONCEPT 6 — EPILOGUE (BF16-only, swizzled STSM staging)
// ---------------------------------------------------------------------------
// Accumulators (BLOCK_N/2 floats/lane) are packed bf16x2 and committed with
// `stmatrix.x2` (STSM) into a TMA-swizzled staging tile, then split into
// BLOCK_N / TMA_D_BLOCK_N bulk `cp.async.bulk.tensor` stores (one per
// thread; TMA clips out-of-bounds columns/rows at shape_n/shape_m). With
// kSwizzleDMode == 0 the staging is plain row-major and a single store
// covers the whole tile. No C accumulation on this kernel (BF16 out only).
//
// ---------------------------------------------------------------------------
// Supported GemmTypes: Normal (this repo's launcher), MGroupedContiguous,
// MGroupedMasked and Batched (BMM; 3D A/B/D descriptors, multicast
// statically disabled — upstream's heuristics disable it too). K-grouped
// does not exist for 1d2d upstream. NOTE: masked+multicast additionally
// depends on prelude Scheduler peer-liveness semantics; the Rust launcher
// only exposes the Normal flavor, which is fully covered.
// ---------------------------------------------------------------------------
// This file is a single NVRTC translation unit: the JIT engine concatenates
// prelude.h + wgmma.h + this file (see src/jit.rs `kernel_src` / api_1d2d).
// ===========================================================================

namespace dg {

// ---------------------------------------------------------------------------
// Small constexpr helpers (upstream common/math.cuh bits not in prelude.h)
// ---------------------------------------------------------------------------
constexpr DG_DEVICE uint32_t cexpr_gcd(uint32_t a, uint32_t b) {
    return b == 0 ? a : cexpr_gcd(b, a % b);
}

// Ladder dispatch over `num_former_iters` (0, kGap, 2*kGap, ... < kEnd).
// Upstream: `dispatch_num_former_iters` — recursion in kGap steps; the
// value only materializes as `cute::Int<N>` at the call site, and the
// guarded `num_former_iters == kNumFormerIters` comparison lets the
// compiler fold the predicate inside `func` after inlining.
// NOTE: the functor takes a plain `uint32_t` VALUE here, not upstream's
// `cute::Int<N>` tag — NVRTC's JIT front end only infers __device__ for
// NON-generic lambdas (a generic lambda's templated operator() is rejected
// as a "host function"). The literal argument of this __forceinline__
// dispatch serves the same constant-folding purpose after inlining.
template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
DG_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(kNumFormerIters);
        return;
    }
    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

// ===========================================================================
// sm90_fp8_gemm_1d2d_impl — FP8, 1D2D scaling, BF16 output, persistent.
//
// Template contract (mirrors upstream + the Rust-side tile chooser):
//   * BLOCK_K == 128 (one scale granule per stage), A/B 128B-swizzled K-major
//     FP8, swizzles A/B == BLOCK_K bytes (no TMA splits);
//   * BLOCK_N <= 192 and either fits one 128-col scale block or straddles
//     exactly two (static assert below == upstream "Too much B scales");
//   * BLOCK_M in {16, 32, 64, 128, 256}; WGMMA is m64n{BLOCK_N}k32, two
//     math warpgroups for BLOCK_M > 64, WAVE_BLOCK_M=128 waves above that;
//   * cd_dtype == 1 (BF16) — upstream asserts `bfloat16_t`;
//   * kGemmType in {Normal, MGroupedContiguous, MGroupedMasked, Batched};
//   * Batched must not use multicast (no 3D multicast primitive here;
//     upstream heuristics disable cluster for Batched anyway).
// ===========================================================================
template <uint32_t kSFBIsMajorMN,       // 0: sfb K-major [n/128, k/128];
                                      // 1: sfb MN-major [k/128, n/128]
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, uint32_t kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType,
          uint32_t cd_dtype>            // 1 = BF16 (only supported flavor)
__launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) __global__
void sm90_fp8_gemm_1d2d_impl(const float* sfb, int* grouped_layout,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const TmaMap tensor_map_a, const TmaMap tensor_map_b,
                             const TmaMap tensor_map_d, const TmaMap tensor_map_sfa) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // ---- scaling / type checks (upstream, verbatim semantics) ------------
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div_u32(BLOCK_N, BLOCK_K) == 1 ||
                     cexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K,
                     "Too much B scales in a single block");
    DG_STATIC_ASSERT(cd_dtype == 1, "Invalid C/D data dtype (1d2d outputs BF16)");
    DG_STATIC_ASSERT(kNumTMAThreads == 128 && kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal || kGemmType == GemmType::MGroupedContiguous ||
                     kGemmType == GemmType::MGroupedMasked || kGemmType == GemmType::Batched,
                     "Unsupported GEMM type for the 1d2d kernel");
    DG_STATIC_ASSERT(kGemmType != GemmType::Batched || kNumTMAMulticast == 1,
                     "Batched 1d2d has no 3D multicast load (upstream disables multicast for BMM)");

    // WGMMA selector: m64 n{BLOCK_N} k32, BLOCK_N/2 FP32 accumulators per lane.
    static constexpr uint32_t WGMMA_M = 64, WGMMA_K = 32, kNumAccum = BLOCK_N / 2;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA_M == 0 || BLOCK_M < WGMMA_M, "Invalid block size");

    // Overwrite shape constants if the compiler gives them.
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // ---- scale-factor geometry (runtime) ---------------------------------
    // kMustUseUniformedScaleB: the whole B tile sits inside ONE 128-wide
    // sfb column block (BLOCK_N divides BLOCK_K) -> single smem_sfb row,
    // no straddle predicate.
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = (BLOCK_M * BLOCK_N * 2 + 1023u) & ~1023u;  // bf16
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K;               // fp8
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K;               // fp8
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * 4;                   // fp32 (TMA bytes)
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = (BLOCK_M * 4 + 127u) & ~127u;
    const uint32_t shape_k_scales = ceil_div_u32(shape_k, BLOCK_K);   // sfb k-blocks
    const uint32_t shape_n_sfb = ceil_div_u32(shape_n, BLOCK_K);      // sfb n-blocks
    // SFB staging: up to 2 rows of k_blocks floats, padded to the barrier
    // alignment (barriers follow it in SMEM — this offset is RUNTIME).
    const uint32_t smem_sfb_size = align_u32(
        shape_k_scales * (kMustUseUniformedScaleB ? 1u : 2u) * 4u, (uint32_t)sizeof(Barrier));

    // WGMMA reads at 8-row granularity from the (possibly smaller) A stage;
    // the read may run past the A stage into the B stages (harmless: rows
    // beyond BLOCK_M are never stored). Upstream guard, kept verbatim.
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA_M * BLOCK_K;
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages,
                     "Memory Out of bound for WGMMA");
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // ---- configs ----------------------------------------------------------
    const uint32_t num_total_k_blocks = ceil_div_u32(shape_k, BLOCK_K);
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch the four descriptors into the TMA unit's cache.
    if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        prefetch_tma_map(&tensor_map_sfa);
        prefetch_tma_map(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // ---- shared-memory layout ---------------------------------------------
    //   [0, SMEM_D_SIZE)                      D staging (bf16, swizzled)
    //   [.., +S*SMEM_A)                       A stages (fp8, 128B swizzle)
    //   [.., +S*SMEM_B)                       B stages (fp8, 128B swizzle)
    //   [.., +S*ALIGNED_SMEM_SFA)             SFA stages (fp32 rows)
    //   [.., +smem_sfb_size)                  SFB staging (2 x k_blocks)
    //   [.., +2*kNumStages*sizeof(Barrier))   full/empty mbarriers
    uint8_t* smem_d = smem_buffer;
    auto smem_a_of = [&](uint32_t i) {
        return smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE;
    };
    auto smem_b_of = [&](uint32_t i) {
        return smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE;
    };
    static constexpr uint32_t SMEM_SF_OFFSET =
        SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa_of = [&](uint32_t i) {
        return (float*)(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    };
    float* smem_sfb = (float*)(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers (SFB staging sits between the SFA stages and them).
    Barrier* barrier_start = (Barrier*)((uint8_t*)smem_sfb + smem_sfb_size);
    auto full_barrier_of = [&](uint32_t i) { return barrier_start + i; };
    auto empty_barrier_of = [&](uint32_t i) { return barrier_start + kNumStages + i; };

    if (warp_idx == kNumMathThreads / 32 + 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barrier_of(i)->init(1);
            // Every math warp of every cluster CTA releases a stage.
            empty_barrier_of(i)->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        // Make initialized barriers visible in the async proxy.
        fence_barrier_init();
    }
    // Synchronize all threads to make barriers visible in the normal model.
    kNumTMAMulticast > 1 ? cluster_sync_relaxed() : (void)__syncthreads();

    // Register rebalance: TMA warps shrink to 40, math warps grow to
    // 248 (single math WG) / 232 (two math WGs — the second WG's accumulators
    // spill past 232 otherwise).
    static constexpr uint32_t kNumTMARegisters = 40;
    static constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // PDL: wait for the primary kernel's completion signal.
    griddepcontrol_wait();

    // ---- persistent block scheduler ----------------------------------------
    Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumTMAMulticast, kIsTMAMulticastOnA != 0, kNumSMs>
        scheduler(shape_m, shape_n, shape_k, grouped_layout);
    scheduler.kNumGroupsRuntime = kNumGroups;
    using Sched = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumTMAMulticast, kIsTMAMulticastOnA != 0, kNumSMs>;

    // Pipeline cursor: (stage, phase), advanced post-body (upstream form).
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        // Flip the phase only when the ring wraps to the first stage.
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    uint32_t m_block_idx, n_block_idx;

    if (warp_idx >= kNumMathThreads / 32) {
        // ================= TMA producer warpgroup =================
        setmaxnreg_dec<kNumTMARegisters>();
        // Use the THIRD warp: warp 0/1 of this group may still be draining
        // WGMMA-era register shuffles on tiny BLOCK_M (upstream note).
        if (warp_idx == kNumMathThreads / 32 + 2 && elect_one_sync()) {
            const uint16_t cta_mask = (uint16_t)((1u << kNumTMAMulticast) - 1u);
            // Persistently schedule over blocks.
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Multicast legality (odd tails / group boundaries). There may
                // be additional odd rows/columns where multicast is invalid.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a =
                    (kIsTMAMulticastOnA != 0 && is_tma_multicast_valid) ? kNumTMAMulticast : 1u;
                const uint32_t num_tma_multicast_b =
                    (kIsTMAMulticastOnA == 0 && is_tma_multicast_valid) ? kNumTMAMulticast : 1u;

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait for the consumer to release this stage.
                    empty_barrier_of(stage_idx)->wait(phase ^ 1);
                    Barrier* fb = full_barrier_of(stage_idx);

                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0u);
                    // Masked layout: A/SFA/D coordinates are group-relative.
                    constexpr bool kWithGroupOffsetA = (kGemmType == GemmType::MGroupedMasked);
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t m_idx = scheduler.template get_global_idx<kWithGroupOffsetA>(
                        shape_m, BLOCK_M, m_block_idx);
                    // SFA coordinate: outer SF row = (batch offset for BMM)
                    // + k_block — the SF_K index type folds the group in.
                    const uint32_t sf_k_idx =
                        scheduler.template get_global_idx<kWithGroupOffsetA, Sched::IndexType::SF_K>(
                            shape_k_scales, 1, k_block_idx);

                    // ---- A: box [BLOCK_K, BLOCK_M] @ (k, m), 128B swizzle.
                    // Multicast is issued by cluster rank 0 ONLY: one issue
                    // credits the full barrier of BOTH CTAs (upstream guards
                    // this inside tma::copy's multicast branch).
                    if (kIsBatchedMM) {
                        tma_load_3d(&tensor_map_a, fb, smem_a_of(stage_idx), kEvictNormalHint,
                                    k_idx, m_idx, batch_idx);
                    } else if (num_tma_multicast_a > 1) {
                        if (get_block_rank_in_cluster() == 0)
                            tma_load_2d_multicast(&tensor_map_a, fb, smem_a_of(stage_idx),
                                                  cta_mask, k_idx, m_idx);
                    } else {
                        tma_load_2d(&tensor_map_a, fb, smem_a_of(stage_idx), kEvictNormalHint,
                                    k_idx, m_idx);
                    }
                    // ---- SFA: box [BLOCK_M, 1] FP32 @ (m, kb), no swizzle.
                    // Rides together with A (same m-block, multicast with it).
                    if (num_tma_multicast_a > 1 && !kIsBatchedMM) {
                        if (get_block_rank_in_cluster() == 0)
                            tma_load_2d_multicast(&tensor_map_sfa, fb, smem_sfa_of(stage_idx),
                                                  cta_mask, m_idx, sf_k_idx);
                    } else {
                        tma_load_2d(&tensor_map_sfa, fb, smem_sfa_of(stage_idx), kEvictNormalHint,
                                    m_idx, sf_k_idx);
                    }
                    // ---- B: box [BLOCK_K, BLOCK_N] @ (k, n), 128B swizzle.
                    // Contiguous layout: B is per-expert (group-relative n).
                    const uint32_t n_idx = scheduler.template get_global_idx<true>(
                        shape_n, BLOCK_N, n_block_idx, m_block_idx);
                    if (kIsBatchedMM) {
                        tma_load_3d(&tensor_map_b, fb, smem_b_of(stage_idx), kEvictNormalHint,
                                    k_idx, n_idx, batch_idx);
                    } else if (num_tma_multicast_b > 1) {
                        if (get_block_rank_in_cluster() == 0)
                            tma_load_2d_multicast(&tensor_map_b, fb, smem_b_of(stage_idx),
                                                  cta_mask, k_idx, n_idx);
                    } else {
                        tma_load_2d(&tensor_map_b, fb, smem_b_of(stage_idx), kEvictNormalHint,
                                    k_idx, n_idx);
                    }
                    // SFB does NOT travel on the TMA pipeline (math warps
                    // preload it — see the consumer side).
                    fb->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                                             SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, one more
            // round of empty waits drains every stage release.
            if (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barrier_of(stage_idx)->wait(phase ^ 1);
            }
        }
    } else {
        // ================= Math warpgroup(s) =================
        setmaxnreg_inc<kNumMathRegisters>();

        const uint32_t math_wg_idx = threadIdx.x / 128;
        // Lane -> output row map (WGMMA m64 accumulator layout, see wgmma.h):
        // warp w owns rows [16w, 16w+16); r_1 is the second 8-row half.
        const uint32_t r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Stage-0 GMMA descriptors, built once and shfl-broadcast as uniform
        // registers; the loop then walks them with a cheap 16B-unit add.
        // LBO=0 / SBO=1024 + layout B128 == upstream `make_smem_desc(ptr, 1)`.
        GmmaDescriptor a_desc0 = make_gmma_desc(
            smem_a_of(0) + math_wg_idx * WGMMA_M * BLOCK_K, GmmaLayoutType::B128, 0, 1024);
        GmmaDescriptor b_desc0 = make_gmma_desc(smem_b_of(0), GmmaLayoutType::B128, 0, 1024);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc0.lo, 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc0.lo, 0);

        // Warpgroup row offset for computation-validity checks (grouped
        // layouts): warpgroup wg covers output rows [wg*64, wg*64+64).
        auto is_computation_valid = [&](uint32_t m_offset) {
            if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched) {
                (void)m_offset;
                return true;
            } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                // Padding rows carry a negative expert index.
                return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
            } else {  // MGroupedMasked: rows beyond the group's m are invalid.
                return m_offset + m_block_idx * BLOCK_M <
                       (uint32_t)grouped_layout[scheduler.current_group_idx];
            }
        };

        // Persistently schedule over blocks.
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // ---- B-scale straddle geometry (CONCEPT 4) ----
            // num_former_iters: 8-column groups inside the FIRST sfb column
            // block; num_full_iters: groups actually in-bounds (shape_n tail).
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (!kMustUseUniformedScaleB) {
                num_former_iters = dg_min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = dg_min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            // 1 row of SFB suffices when every valid group is in row 0.
            const uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1u : 2u);

            // ---- SFB preload: global -> SMEM with the MATH warps ----
            // All math warps except warp 0 (busy with the previous block's
            // TMA store tail) strided-copy the 1-2 scale rows. Group offset:
            // m-grouped stacks per-expert SFB blocks; SF_K folds the batch.
            if (threadIdx.x >= 32) {
                const uint32_t previous_group_offset =
                    scheduler.template get_global_idx<true, Sched::IndexType::SF_K>(
                        shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                // Strides for the two possible sfb majors:
                //   K-major  [n/128, k/128]: (n, k) -> n*ks + k
                //   MN-major [k/128, n/128]: (n, k) -> n + k*ns
                const uint32_t stride_n_sfb = kSFBIsMajorMN != 0 ? 1u : shape_k_scales;
                const uint32_t stride_k_sfb = kSFBIsMajorMN != 0 ? shape_n_sfb : 1u;
                // Base of the FIRST sfb column block this tile touches.
                const float* local_sfb = sfb + previous_group_offset +
                                         ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;
                #pragma unroll 2
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                    smem_sfb[i] = (i < shape_k_scales)
                                      ? local_sfb[i * stride_k_sfb]                       // row 0
                                      : local_sfb[(i - shape_k_scales) * stride_k_sfb +
                                                   stride_n_sfb];                          // row 1
            }
            // All math warps (and warp 0's epilogue tail) meet here before
            // the WGMMA loop reads smem_sfb.
            named_barrier_sync(kNumMathThreads, 0);

            // Per-warpgroup WGMMA accumulator + promoted (scaled) tile.
            // BLOCK_M > 128: each warpgroup sweeps WAVE_BLOCK_M=128-row waves.
            static constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA_M ? BLOCK_M : WGMMA_M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[kNumAccum];
            float final_accum[kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // BLOCK_M < 64: only the first WAVE_BLOCK_M*2 threads store.
            DG_STATIC_ASSERT(BLOCK_M >= 64 || kNumMathThreads == 128,
                             "Only one math warp group for `BLOCK_M < 64`");
            static constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA_M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA_M || warp_idx < kNumWGMMAStoreThreads / 32;

            // Stage release: lane 0 (multicast: lanes 0/1 -> CTA 0/1). If the
            // peer CTA already exited, both lanes arrive locally (keeps the
            // empty barrier's expected arrival count consistent).
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    if (lane_idx == 0) empty_barrier_of(stage_idx)->arrive();
                } else {
                    const uint32_t target_cta =
                        scheduler.is_peer_cta_alive ? lane_idx : get_block_rank_in_cluster();
                    if (lane_idx < kNumTMAMulticast) empty_barrier_of(stage_idx)->arrive_cluster(target_cta);
                }
            };

            // Skip useless computations (grouped layouts: this warpgroup's
            // rows may be entirely padding).
            if (is_computation_valid(math_wg_idx * WGMMA_M)) {
                // Compile-time specialization ladder for the straddle
                // predicate: BLOCK_K/gcd(BLOCK_K, BLOCK_N) <= 4 means only a
                // handful of num_former_iters values are possible.
                static constexpr bool kShouldOptimize =
                    BLOCK_K / cexpr_gcd(BLOCK_K, BLOCK_N) <= 4 && !kMustUseUniformedScaleB;
                static constexpr uint32_t kGap = cexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                static constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                dispatch_num_former_iters<0, kGap, kEnd>(
                    kShouldOptimize ? num_former_iters : 0, [&](uint32_t constant_former_iters) {
                    // `#pragma unroll 8` (upstream): partial unroll keeps the
                    // pipeline body small while overlapping wgmma waits.
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // Read the B scales BEFORE the TMA wait: smem_sfb is
                        // filled once per output block, outside the pipeline.
                        // ("even some blocks do not need the second row, we
                        // still load one to align with other blocks"; the
                        // uniformed-BLOCK_N instantiation never references
                        // row 1 — and must not, its staging is 1 row.)
                        const float scale_b_0 = smem_sfb[k_block_idx];
                        [[maybe_unused]] float scale_b_1 = 0.0f;
                        if constexpr (!kMustUseUniformedScaleB)
                            scale_b_1 = smem_sfb[k_block_idx + shape_k_scales];

                        // Wait for the TMA arrivals of this stage.
                        full_barrier_of(stage_idx)->wait(phase);

                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                            const uint32_t m_offset = local_idx * WAVE_BLOCK_M;

                            // Read A scales for our two half-rows. ALL shared
                            // memory reads must happen BEFORE `warpgroup_arrive`:
                            // the next scheduled block's TMA may overwrite the
                            // stage as soon as we release it below. Non-store
                            // warps (BLOCK_M < 64) predicate the read off —
                            // their r_0/r_1 may run past the SFA stage.
                            const float scale_a_0 = do_wgmma_store ? smem_sfa_of(stage_idx)[r_0 + m_offset] : 0.0f;
                            const float scale_a_1 = do_wgmma_store ? smem_sfa_of(stage_idx)[r_1 + m_offset] : 0.0f;

                            // One wgmma batch per K=128 stage: 4 issues of k32
                            // (first overwrites the accumulator, rest add).
                            #pragma unroll
                            for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                            warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA_K; ++k) {
                                a_desc0.lo = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA_K) / 16;
                                b_desc0.lo = b_desc_base_lo + k * WGMMA_K / 16;
                                wgmma_f8<BLOCK_N>(a_desc0.desc_, b_desc0.desc_, accum, k == 0 ? 0u : 1u);
                            }
                            warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                            warpgroup_wait_group<0>();

                            // Release the stage at the last warpgroup wave.
                            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts.
                            if (!do_wgmma_store)
                                continue;

                            // ---- the fine-grained-scaling promotion ----
                            const float scale_0_0 = scale_a_0 * scale_b_0;
                            const float scale_1_0 = scale_a_1 * scale_b_0;
                            float scale_0_1 = 0.0f, scale_1_1 = 0.0f;
                            if constexpr (!kMustUseUniformedScaleB) {
                                scale_0_1 = scale_a_0 * scale_b_1;
                                scale_1_1 = scale_a_1 * scale_b_1;
                            }
                            // The dispatched rung makes the `i <` predicate a
                            // compile-time constant inside each instantiation
                            // (upstream passes a `cute::Int<N>` tag for the
                            // same purpose; the inlined literal argument folds
                            // identically, and falls back to the runtime value
                            // exactly when upstream does: BLOCK_N = 144-style
                            // tiles skip the ladder).
                            const uint32_t num_former_ct =
                                kShouldOptimize ? constant_former_iters : num_former_iters;
                            float* shifted_accum = final_accum + kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
                                const bool predicate = kMustUseUniformedScaleB || i < num_former_ct;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }
                });
            } else {
                // Invalid warpgroup (pure padding): still drain the pipeline —
                // the producer is waiting on this stage's empty barrier.
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
                     advance_pipeline(k_block_idx)) {
                    full_barrier_of(stage_idx)->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // ---- epilogue: registers -> STSM -> TMA bulk stores ----
            static constexpr uint32_t kNumElemBytes = 2;  // bf16
            static constexpr uint32_t TMA_D_BLOCK_N =
                kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            static constexpr uint32_t WGMMA_M_PER_WARP = WGMMA_M / 4;  // 16 rows/warp
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 && BLOCK_N / TMA_D_BLOCK_N <= 32,
                             "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");
            DG_STATIC_ASSERT(kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");

            // Skip the store for the unfilled parts.
            if (!do_wgmma_store)
                continue;

            // Wait for the previous block's bulk stores before overwriting D.
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                tma_store_wait<0>();
            named_barrier_sync(kNumWGMMAStoreThreads, 1);

            // Pack bf16x2 and commit with STSM x2 (16 lanes' addresses used).
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                float* shifted_accum = final_accum + kNumAccum * local_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
                    uint8_t* smem_ptr;
                    if (kSwizzleDMode > 0) {
                        // Swizzled staging: XOR the 16B bank group by the row
                        // so the TMA store atom reads conflict-free.
                        static constexpr uint32_t kNumBankGroupBytes = 16;
                        const uint32_t atom_offset = i / (TMA_D_BLOCK_N / 8);
                        const uint32_t in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                        const uint32_t bank_group_index =
                            in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                        // Reshape the atom as (BLOCK_M*sw/16/8, 8); the 128B
                        // swizzle has an address-calc shortcut.
                        static constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        const uint32_t row = kHasShortcut ? (in_atom_offset / 8 + lane_idx)
                                                          : (bank_group_index / 8);
                        uint32_t col = kHasShortcut ? in_atom_offset : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);
                        smem_ptr = smem_d +
                                   warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +  // warp rows
                                   m_offset * kSwizzleDMode +                       // wave rows
                                   atom_offset * BLOCK_M * kSwizzleDMode +          // n-atom
                                   row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    } else {
                        // No swizzle: row-major [BLOCK_M, BLOCK_N] staging.
                        smem_ptr = smem_d + ((m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) *
                                                 BLOCK_N + i * 8) * kNumElemBytes;
                    }
                    // STSM x2: two 8x8 bf16 matrices per issue.
                    stsm_x2_b16_n(cvta_shared_to_u32(smem_ptr),
                                  cvt_bf16x2_f32(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]),
                                  cvt_bf16x2_f32(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
                }
            }
            tma_store_fence();
            named_barrier_sync(kNumWGMMAStoreThreads, 1);

            // Bulk stores: one TMA store atom per thread (TMA clips the
            // shape_n / shape_m tails). Swizzled atoms are BLOCK_M *
            // TMA_D_BLOCK_N * 2 bytes apart.
            static constexpr bool kWithGroupOffsetD = (kGemmType == GemmType::MGroupedMasked);
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                const uint32_t in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                uint8_t* src = smem_d + in_block_n_offset * BLOCK_M * kNumElemBytes;
                const uint32_t n_idx = n_block_idx * BLOCK_N + in_block_n_offset;
                const uint32_t m_idx = scheduler.template get_global_idx<kWithGroupOffsetD>(
                    shape_m, BLOCK_M, m_block_idx);
                if (kGemmType == GemmType::Batched) {
                    tma_store_3d(&tensor_map_d, src, n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    tma_store_2d(&tensor_map_d, src, n_idx, m_idx);
                }
                tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    // SM90a-only kernel (upstream traps here): the JIT only compiles this TU
    // for compute_90a; keep the guard structural.
    if (blockIdx.x == 0 && threadIdx.x == 0) __trap();
#endif
}

} // namespace dg
