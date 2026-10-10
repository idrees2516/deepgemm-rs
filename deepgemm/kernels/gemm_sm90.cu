// ===========================================================================
// gemm_sm90.cu — Hopper (SM90a) kernels, ported 1:1 from upstream DeepGEMM:
//   * `sm90_fp8_gemm_1d1d_impl`  (impls/sm90_fp8_gemm_1d1d.cuh) — FP8 GEMM
//     with 1D1D fine-grained scaling (FP32 SF per 128 channels per row/col),
//     Normal + KGroupedContiguous (weight-grad) via runtime tensormap patching.
//   * `sm90_bf16_gemm_impl`      (impls/sm90_bf16_gemm.cuh) — BF16 GEMM
//     (K/MN-major operands, m-grouped contiguous/masked, C accumulation,
//     swizzled STSM epilogue, stage-merge optimization).
//
// ---------------------------------------------------------------------------
// WGMMA CONCEPTS (Hopper tensor cores) — read prelude.h for the SM100
// (tcgen05) counterpart; this file is the SM90 generation:
//
// 1) WARP-GROUP MMA.  `wgmma.mma_async` computes m64 x nN x k32 (FP8) per
//    issue.  Unlike Blackwell's tcgen05 (single-thread issue, accumulators in
//    TMEM), WGMMA is issued by a WARPGROUP (4 warps, 128 lanes) and the
//    accumulators live in REGISTERS: N/2 floats per lane laid out as
//
//        lane = (warp_row 16*w + r8*? ) ... concretely, per lane:
//          row_idx = lane/4, col_idx = lane%4
//          acc[i*4+0], acc[i*4+1] -> row (16*w + row_idx),     cols 8i + col*2 + {0,1}
//          acc[i*4+2], acc[i*4+3] -> row (16*w + row_idx + 8), cols 8i + col*2 + {0,1}
//
//    (w = warp index inside the warpgroup; row +8 is the "second 8-row half"
//    each lane also owns — see the diagram at the top of wgmma.h.)
//
// 2) FINE-GRAINED SCALING (1D1D) — THE DeepSeek trick.  FP8 has too little
//    range for LLM tensors, so DeepGEMM quantizes per 128-channel block:
//      A[m, kb] (fp8) * sfa[m, kb] == original A value,  sfa: FP32
//      B[n, kb] (fp8) * sfb[n, kb] == original B value,  sfb: FP32
//    The MMA must then compute  sum_kb sfa[m,kb]*sfb[n,kb] * (A_kb . B_kb).
//    WGMMA cannot scale per 128-K slice by itself, so the kernel splits K
//    into 128-wide stages and *promotes* per-stage:
//
//        wgmma(accum[k_block])            // raw fp8 dot-product over K=128
//        final += sfa * sfb * accum       // per-row/col FP32 promotion
//
//    in registers, between `wgmma.wait_group` and the next stage.  This is
//    the exact arithmetic the Blackwell kernel instead folds into tcgen05's
//    hardware block-scales (UE8M0 *tmem* path) — two generations, one idea.
//
// 3) WARP SPECIALIZATION (2 roles, no epilogue warps on SM90):
//        threads [0, kNumMathThreads)       math warpgroup(s): wgmma + promote
//                                          + epilogue TMA store
//        threads [kNumMathThreads, +128)    TMA warpgroup: 1 elected thread
//                                          issues all TMA loads (A/B/SFA/SFB),
//                                          patches tensormaps (K-grouped only)
//    The two groups handshake through full/empty mbarriers (kNumStages deep,
//    phase-flipped on wrap).  Register budget is rebalanced with setmaxnreg:
//    math gets 232-248 regs, TMA gets 24-40 — the TMA warp is mostly idle.
//
// 4) TMA MULTICAST (cluster of 2).  When cluster_n = 2, both CTAs need the
//    same A tiles (same m-block, different n-blocks).  CTA rank 0 issues ONE
//    `cp.async.bulk.tensor...multicast::cluster` load; the TMA unit delivers
//    the bytes into BOTH CTAs' smem and credits BOTH full barriers.  Each
//    CTA still loads its own B/SFB.  The math warps answer on BOTH CTAs'
//    empty barriers (lane 0 -> CTA0, lane 1 -> CTA1), so a stage is reused
//    only after both consumers finished — a distributed 2-producer pipeline.
//
// 5) K-GROUPED WEIGHT-GRAD (KGroupedContiguous).  dC = sum_g (dA_g @ dB_g)
//    with A/B stacked along K.  Instead of one launch per group, the TMA
//    descriptors themselves are PATCHED at every group transition:
//        tensormap.replace.tile.global_address  <- base + k_start*stride
//        tensormap.replace.tile.global_dim[0]   <- shape_k(g)
//        tensormap.replace.tile.global_stride[0]<- shape_k(g)
//        cp.async.bulk.commit_group / wait_group  // drain in-flight reads
//        *gmem_map = *smem_map;                  // publish per-CTA desc
//        fence.proxy.tensormap::generic.release/acquire.gpu
//    All num_sms CTAs run the whole loop in ONE persistent launch; the C D
//    buffer accumulates via `cp.reduce.async.bulk` (TMA reduce-add) — this
//    is why the epilogue STORES WITH ADD: multiple (m,n) blocks may add into
//    the same C tile across groups (and across the persistent schedule).
//
// 6) PDL (programmatic dependent launch).  `griddepcontrol.wait` at kernel
//    start lets the NEXT kernel overlap its prologue with this one's
//    epilogue — the Rust launcher enables CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC
//    STREAM SERIALIZATION when the caller asks for a PDL chain.
// ---------------------------------------------------------------------------
// This file is a single NVRTC translation unit: the JIT engine concatenates
// prelude.h + wgmma.h + this file (see src/jit.rs `kernel_src`).
// ---------------------------------------------------------------------------

namespace dg {

// ---------------------------------------------------------------------------
// SM90 shared helper: high-level GMMA descriptor builders (mma/sm90.cuh port)
// ---------------------------------------------------------------------------
template <uint32_t kSwizzleMode>
DG_DEVICE GmmaLayoutType to_gmma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 || kSwizzleMode == 16 || kSwizzleMode == 32 ||
                     kSwizzleMode == 64 || kSwizzleMode == 128, "Invalid swizzling mode");
    if (kSwizzleMode == 32) return GmmaLayoutType::B32;
    if (kSwizzleMode == 64) return GmmaLayoutType::B64;
    if (kSwizzleMode == 128) return GmmaLayoutType::B128;
    return GmmaLayoutType::INTERLEAVE;  // 0 / 16
}

// K-stride (in elements) of the descriptor base within a stage:
//   K-major -> 1 (contiguous),  MN-major -> one swizzle atom.
template <uint32_t kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, uint32_t kElemSize>
DG_DEVICE uint32_t get_gmma_desc_stride_k() {
    return kMajorMode == MAJOR_K ? 1u
                                 : (kSwizzleMode == 0 ? BLOCK_MN : kSwizzleMode) / kElemSize;
}

// Advance only the low 32 bits of a descriptor by a (mn_idx, k_idx) element
// offset — the bf16 kernel shfl-broadcasts the stage base ONCE and walks
// this cheap add per wgmma instead of recomputing the full descriptor.
template <uint32_t kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          uint32_t kElemSize>
DG_DEVICE uint32_t advance_gmma_desc_lo(uint32_t base, uint32_t mn_idx, uint32_t k_idx,
                                         uint32_t offset = 0) {
    const uint32_t stride_k = get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, kElemSize>();
    return base + (((offset + mn_idx * BLOCK_K + k_idx * stride_k) * kElemSize) >> 4);
}

// Full descriptor for one (mn_idx, k_idx) corner of a stage tile.
template <uint32_t kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          uint32_t kElemSize>
DG_DEVICE GmmaDescriptor make_gmma_desc_t(const void* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, kElemSize>();
    const GmmaLayoutType layout_type = to_gmma_layout_type<kSwizzleMode>();
    constexpr uint32_t num_non_contiguous = 128 / 16;  // 8 x 16B atom rows
    if (kMajorMode == MAJOR_K) {
        // One swizzle atom spans all of K (swizzle == BLOCK_K bytes, asserted
        // upstream "Unexpected value" otherwise); SBO strides 8-row groups.
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * kElemSize;
        const uint32_t leading_byte_offset = 0;
        return make_gmma_desc((const uint8_t*)base_smem_ptr +
                                   (mn_idx * BLOCK_K + k_idx * stride_k) * kElemSize,
                               layout_type, leading_byte_offset, stride_byte_offset);
    } else {
        // MN-major: atom = swizzle bytes on MN; LBO = K extent * atom,
        // SBO = 8 * atom (swapped for the non-swizzled 16B interleave).
        const uint32_t BLOCK_MN_ATOM = (kSwizzleMode == 0 ? BLOCK_MN : kSwizzleMode) / kElemSize;
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * kElemSize;
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * kElemSize;
        if (kSwizzleMode == 16) { uint32_t t = stride_byte_offset; stride_byte_offset = leading_byte_offset; leading_byte_offset = t; }
        return make_gmma_desc((const uint8_t*)base_smem_ptr +
                                   (mn_idx * BLOCK_K + k_idx * stride_k) * kElemSize,
                               layout_type, leading_byte_offset, stride_byte_offset);
    }
}

// ===========================================================================
// sm90_fp8_gemm_1d1d_impl — FP8, 1D1D per-128 FP32 scaling, persistent.
//
// Template contract (mirrors upstream, checked by heuristics::sm90_*):
//   BLOCK_K == 128 (one scale granule per stage), swizzle A/B == 128B,
//   BLOCK_M in {64, 128} (WGMMA::M=64; 128 = two math warpgroups),
//   kGemmType in {Normal, KGroupedContiguous}, cd_dtype = FP32.
// ===========================================================================
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType>
__launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) __global__
void sm90_fp8_gemm_1d1d_impl(const uint8_t* gmem_a_ptr, const uint8_t* gmem_b_ptr,
                             int* grouped_layout, TmaMap* tensor_map_buffer,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const TmaMap tensor_map_a_base, const TmaMap tensor_map_b_base,
                             const TmaMap tensor_map_sfa, const TmaMap tensor_map_sfb,
                             const TmaMap tensor_map_cd) {
    DG_STATIC_ASSERT(kNumTMAThreads == 128 && kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal || kGemmType == GemmType::KGroupedContiguous,
                     "Invalid GEMM type");
    // C/D: FP32 with accumulation across blocks (reduce-add epilogue).
    static_assert(sizeof(float) == 4, "");

    // WGMMA selector: m64 n{BLOCK_N} k32, accum = BLOCK_N/2 floats per lane.
    constexpr uint32_t WGMMA_M = 64, WGMMA_K = 32, kNumAccum = BLOCK_N / 2;
    DG_STATIC_ASSERT(BLOCK_M == WGMMA_M * (BLOCK_M <= 64 ? 1u : 2u), "Invalid block sizes");

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // ---- shared memory layout -------------------------------------------
    //   [0, 256)   2 TmaMaps in SMEM (K-grouped only; the patch scratch)
    //   [..)       D staging (fp32), then A stages, B stages, SFA, SFB,
    //              then 2*kNumStages mbarriers.
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(TmaMap) * 2 : 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K;
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K;
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    // SFB staging is BLOCK_N*4 bytes padded to 128B rows in the smem layout
    // (the TMA box itself transfers exactly BLOCK_N*4 bytes — see expect_tx).
    static constexpr uint32_t ALIGNED_SMEM_SFB = (BLOCK_N * sizeof(float) + 127u) & ~127u;
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");

    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch all five descriptors into the TMA unit's cache.
    if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a_base);
        prefetch_tma_map(&tensor_map_b_base);
        prefetch_tma_map(&tensor_map_sfa);
        prefetch_tma_map(&tensor_map_sfb);
        prefetch_tma_map(&tensor_map_cd);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "D smem must align to 1024B");

    TmaMap* smem_tensor_map_a = (TmaMap*)smem_buffer;
    TmaMap* smem_tensor_map_b = smem_tensor_map_a + 1;
    TmaMap* gmem_tensor_map_a = tensor_map_buffer + blockIdx.x * 2;
    TmaMap* gmem_tensor_map_b = gmem_tensor_map_a + 1;

    // Stage pointers (upstream uses PatternVisitor; plain lambdas here).
    float* smem_d = (float*)(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    auto smem_a_of = [&](uint32_t i) { return smem_buffer + SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE; };
    auto smem_b_of = [&](uint32_t i) { return smem_buffer + SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE; };
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa_of = [&](uint32_t i) { return (float*)(smem_buffer + SMEM_SF_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE); };
    auto smem_sfb_of = [&](uint32_t i) { return (float*)(smem_buffer + SMEM_SF_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * ALIGNED_SMEM_SFB); };

    constexpr uint32_t SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB);
    auto full_barrier_of = [&](uint32_t i) { return (Barrier*)(smem_buffer + SMEM_BARRIER_OFFSET + i * sizeof(Barrier)); };
    auto empty_barrier_of = [&](uint32_t i) { return (Barrier*)(smem_buffer + SMEM_BARRIER_OFFSET + (kNumStages + i) * sizeof(Barrier)); };

    if (warp_idx == kNumMathThreads / 32 + 1 && elect_one_sync()) {
        if (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tensor_map_a = tensor_map_a_base;
            *smem_tensor_map_b = tensor_map_b_base;
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barrier_of(i)->init(1);
            // Every math warp of every cluster CTA releases a stage.
            empty_barrier_of(i)->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        fence_barrier_init();
    }
    kNumTMAMulticast > 1 ? cluster_sync_relaxed() : (void)__syncthreads();

    // Pipeline unroll control: full unroll only when the K trip count is
    // compile-time shaped; K-grouped runs a dynamic loop (unroll 0).
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    griddepcontrol_wait();

    Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>
        scheduler(shape_m, shape_n, shape_k, grouped_layout);
    scheduler.kNumGroupsRuntime = kNumGroups;
    using Sched = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>;

    // Pipeline: (stage, phase) from a monotonically increasing iter index.
    const auto get_pipeline = [](uint32_t iter, uint32_t& stage, uint32_t& phase) {
        stage = iter % kNumStages;
        phase = (iter / kNumStages) & 1;
    };
    uint32_t iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // ================= TMA producer warpgroup =================
        setmaxnreg_dec<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
            uint32_t last_group_idx = kNumGroups;
            uint32_t m_block_idx, n_block_idx;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                if (kGemmType == GemmType::KGroupedContiguous && scheduler.current_shape_k == 0)
                    continue;

                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA && is_tma_multicast_valid) ? kNumTMAMulticast : 1u;
                const uint32_t num_tma_multicast_b = (!kIsTMAMulticastOnA && is_tma_multicast_valid) ? kNumTMAMulticast : 1u;

                const uint32_t num_k_blocks = ceil_div_u32(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                // ---- K-group transition: patch + publish descriptors ----
                if (kGemmType == GemmType::KGroupedContiguous && last_group_idx != scheduler.current_group_idx) {
                    last_group_idx = scheduler.current_group_idx;
                    const uint64_t current_k_offset = scheduler.current_k_start;
                    // A is [k, m] (K-major, row stride shape_m), B likewise with n.
                    tensormap_replace_global_addr(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m);
                    tensormap_replace_global_addr(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n);
                    tensormap_replace_global_inner_dim(smem_tensor_map_a, scheduler.current_shape_k);
                    tensormap_replace_global_inner_dim(smem_tensor_map_b, scheduler.current_shape_k);
                    tensormap_replace_global_inner_stride(smem_tensor_map_a, scheduler.current_shape_k);
                    tensormap_replace_global_inner_stride(smem_tensor_map_b, scheduler.current_shape_k);
                    // Drain any in-flight TMA reads of the OLD descriptor.
                    tma_desc_commit_group();
                    tma_desc_wait_group();
                    __syncwarp(1u << lane_idx);
                    // Publish to the per-CTA GMEM slot and re-acquire.
                    *gmem_tensor_map_a = *smem_tensor_map_a;
                    *gmem_tensor_map_b = *smem_tensor_map_b;
                    tensormap_fence_release_gpu();
                    tensormap_fence_acquire_gpu(gmem_tensor_map_a);
                    tensormap_fence_acquire_gpu(gmem_tensor_map_b);
                }

                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                    uint32_t stage, phase;
                    get_pipeline(iter_idx++, stage, phase);
                    empty_barrier_of(stage)->wait(phase ^ 1);

                    Barrier* fb = full_barrier_of(stage);
                    // Cluster multicast target mask: CTAs 0..(multicast-1).
                    const uint16_t cta_mask = (uint16_t)((1u << kNumTMAMulticast) - 1u);
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    // Scales live at the *concatenated* K position for
                    // K-grouped, hence current_k_start/BLOCK_K + kb.
                    const uint32_t sf_k_idx = (kGemmType == GemmType::KGroupedContiguous
                                                   ? scheduler.current_k_start / BLOCK_K + k_block_idx
                                                   : k_block_idx);
                    const TmaMap* map_a = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_a : &tensor_map_a_base);
                    const TmaMap* map_b = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_b : &tensor_map_b_base);

                    // SFA/SFB: box [BLOCK_M, 1] / [BLOCK_N, 1], unsizzled.
                    if (num_tma_multicast_a > 1)
                        tma_load_2d_multicast(&tensor_map_sfa, fb, smem_sfa_of(stage), cta_mask, m_idx, sf_k_idx);
                    else
                        tma_load_2d(&tensor_map_sfa, fb, smem_sfa_of(stage), kEvictNormalHint, m_idx, sf_k_idx);
                    if (num_tma_multicast_b > 1)
                        tma_load_2d_multicast(&tensor_map_sfb, fb, smem_sfb_of(stage), cta_mask, n_idx, sf_k_idx);
                    else
                        tma_load_2d(&tensor_map_sfb, fb, smem_sfb_of(stage), kEvictNormalHint, n_idx, sf_k_idx);
                    // A: box [BLOCK_K, BLOCK_M] @ (k_idx, m_idx), 128B swizzle.
                    if (num_tma_multicast_a > 1)
                        tma_load_2d_multicast(map_a, fb, smem_a_of(stage), cta_mask, k_idx, m_idx);
                    else
                        tma_load_2d(map_a, fb, smem_a_of(stage), kEvictNormalHint, k_idx, m_idx);
                    if (num_tma_multicast_b > 1)
                        tma_load_2d_multicast(map_b, fb, smem_b_of(stage), cta_mask, k_idx, n_idx);
                    else
                        tma_load_2d(map_b, fb, smem_b_of(stage), kEvictNormalHint, k_idx, n_idx);
                    fb->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                                             SMEM_SFA_SIZE_PER_STAGE + BLOCK_N * sizeof(float));
                }
            }
            // Tear-down: every CTA's distributed barriers must observe all
            // stages released before the cluster disbands.
            if (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++s) {
                    uint32_t stage, phase;
                    get_pipeline(iter_idx++, stage, phase);
                    empty_barrier_of(stage)->wait(phase ^ 1);
                }
            }
        }
    } else {
        // ================= Math warpgroup(s) =================
        setmaxnreg_inc<kNumMathRegisters>();
        const uint32_t math_wg_idx = threadIdx.x / 128;
        const uint32_t row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            if (kGemmType == GemmType::KGroupedContiguous && scheduler.current_shape_k == 0)
                continue;

            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0u);
            const uint32_t num_k_blocks = ceil_div_u32(current_shape_k, BLOCK_K);
            float accum[kNumAccum], final_accum[kNumAccum] = {0};
            float2 scales_b[kNumAccum / 4];

            // Stage release: lane 0 (multicast: lanes 0/1 -> CTA 0/1).
            auto empty_barrier_arrive = [&](uint32_t s) {
                if (kNumTMAMulticast == 1) {
                    if (lane_idx == 0) empty_barrier_of(s)->arrive();
                } else {
                    const uint32_t target_cta = scheduler.is_peer_cta_alive ? lane_idx : get_block_rank_in_cluster();
                    if (lane_idx < kNumTMAMulticast) empty_barrier_of(s)->arrive_cluster(target_cta);
                }
            };

            #pragma unroll
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                uint32_t stage, phase;
                get_pipeline(iter_idx++, stage, phase);
                full_barrier_of(stage)->wait(phase);

                // Read both SF rows for our two half-rows BEFORE any wgmma
                // fences (upstream: "all shared memory read must be prior to
                // warpgroup_arrive" — next block could overwrite the stage).
                const float scale_a_0 = smem_sfa_of(stage)[r_0];
                const float scale_a_1 = smem_sfa_of(stage)[r_1];
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum / 4; ++i)
                    scales_b[i] = *(float2*)(smem_sfb_of(stage) + i * 8 + col_idx * 2);

                // One wgmma batch per K=128 stage: 4 issues of k32.
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA_K; ++k) {
                    GmmaDescriptor da = make_gmma_desc((const uint8_t*)smem_a_of(stage) +
                                math_wg_idx * WGMMA_M * BLOCK_K + k * WGMMA_K,
                                GmmaLayoutType::B128, 0, 1024);
                    GmmaDescriptor db = make_gmma_desc((const uint8_t*)smem_b_of(stage) + k * WGMMA_K,
                                GmmaLayoutType::B128, 0, 1024);
                    wgmma_f8<BLOCK_N>(da.desc_, db.desc_, accum, k == 0 ? 0u : 1u);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                warpgroup_wait_group<0>();
                empty_barrier_arrive(stage);

                // ---- the fine-grained-scaling promotion ----
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
                    const float sb0 = scales_b[i].x, sb1 = scales_b[i].y;
                    final_accum[i * 4 + 0] += scale_a_0 * sb0 * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a_0 * sb1 * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a_1 * sb0 * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a_1 * sb1 * accum[i * 4 + 3];
                }
            }

            // ---- epilogue: registers -> smem_d -> TMA reduce-add ----
            if (warp_idx % 4 == 0 && elect_one_sync()) tma_store_wait<0>();
            named_barrier_sync(128, math_wg_idx + 2);

            float2* sd_0 = (float2*)(smem_d + r_0 * BLOCK_N + col_idx * 2);
            float2* sd_1 = (float2*)(smem_d + r_1 * BLOCK_N + col_idx * 2);
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
                sd_0[i * 4] = make_float2(final_accum[i * 4 + 0], final_accum[i * 4 + 1]);
                sd_1[i * 4] = make_float2(final_accum[i * 4 + 2], final_accum[i * 4 + 3]);
            }
            tma_store_fence();
            named_barrier_sync(128, math_wg_idx + 2);

            // Store with ADD: K-grouped weight-grad accumulates into C across
            // groups; for Normal GEMM the caller zero-initializes C once.
            if (warp_idx % 4 == 0 && elect_one_sync()) {
                tma_reduce_add_2d(&tensor_map_cd, smem_d + r_0 * BLOCK_N,
                                  n_block_idx * BLOCK_N,
                                  current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0);
                tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

// ===========================================================================
// sm90_bf16_gemm_impl — BF16 GEMM on Hopper.
//
// Differences vs the 1D1D FP8 kernel above (all faithful to upstream):
//   * No scaling: wgmma bf16 accumulates straight into fp32 registers, so
//     the math loop is a plain K sweep (no per-stage promotion, no SF TMA).
//   * Operand major-ness is templated (K or MN): MN-major swaps the TMA box
//     order AND sets the wgmma transpose bits (tA/tB) so the MMA reads the
//     other layout from the same swizzled smem.
//   * Two epilogue flavors:
//       - cd=BF16: rows are packed f32->bf16x2 and committed with STSM
//         (stmatrix) into a TMA-swizzled D staging, then split into
//         BLOCK_N/TMA_D_BLOCK_N bulk stores. STSM x2 writes 2 8x8 b16
//         matrices per issue using only 16 lanes' addresses.
//       - cd=FP32: plain st.shared vectorized by float2.
//   * `kWithAccumulation`: C += AB — the store becomes TMA REDUCE-ADD.
//   * `kDoMergeStages` (>= 10 stages, NT normal, 1 math WG): BLOCK_K is
//     enlarged by merging TMA stages (BLOCK_K = BLOCK_K_ * stages/5) so one
//     wgmma batch covers several stages — cuts `wgmma.wait_group` stalls.
//     The descriptor then advances across TMA-stage atoms via
//     `atom_k_idx * BLOCK_M * BLOCK_ATOM_K`.
// ===========================================================================
template <uint32_t kMajorA, uint32_t kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          uint32_t cd_dtype>  // 0: FP32, 1: BF16 (enum-like, keeps NVRTC simple)
__launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) __global__
void sm90_bf16_gemm_impl(int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const TmaMap tensor_map_a, const TmaMap tensor_map_b,
                         const TmaMap tensor_map_cd) {
    // Merge TMA stages when deep pipelining only buys wait overhead
    // (normal NT, K/K operands, one math warpgroup).
    constexpr bool kDoMergeStages =
        kNumStages_ >= 10 && kGemmType == GemmType::Normal &&
        kMajorA == MAJOR_K && kMajorB == MAJOR_K && kNumMathThreads == 128;
    constexpr uint32_t kNumMinStages = 5;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    // wgmma bf16: m64 n{BLOCK_N} k16, fp32 accum. Transpose bits mirror the
    // operand major-ness: MN-major operand -> transposed read in the MMA.
    constexpr uint32_t kTransA = (kMajorA == MAJOR_MN) ? 1u : 0u;
    constexpr uint32_t kTransB = (kMajorB == MAJOR_MN) ? 1u : 0u;
    constexpr uint32_t WGMMA_M = 64, WGMMA_K = 16, kNumAccum = BLOCK_N / 2;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA_M == 0 || BLOCK_M < WGMMA_M, "Invalid block size");
    DG_STATIC_ASSERT(cd_dtype == 0 || cd_dtype == 1, "Invalid C/D dtype");

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    static constexpr uint32_t CD_ELEM = (cd_dtype == 1 ? 2u : 4u);  // bytes
    static constexpr uint32_t SMEM_D_SIZE = (BLOCK_M * BLOCK_N * CD_ELEM + 1023u) & ~1023u;
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * 2;
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * 2;
    // WGMMA reads at 8-row granularity; ensure the *merged* A tile fits in the
    // A+B smem across stages (upstream guard, kept verbatim).
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA_M * BLOCK_K;
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages,
                     "Memory Out of bound for WGMMA");
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0 && SMEM_A_SIZE_PER_STAGE % 1024 == 0 &&
                     SMEM_B_SIZE_PER_STAGE % 1024 == 0, "A/B/D smem must align to 1024B");

    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t lane_idx = threadIdx.x % 32;

    if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        prefetch_tma_map(&tensor_map_cd);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    uint8_t* smem_d = smem_buffer;
    auto smem_a_of = [&](uint32_t i) { return smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE; };
    auto smem_b_of = [&](uint32_t i) { return smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE; };
    auto full_barrier_of = [&](uint32_t i) { return (Barrier*)(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * sizeof(Barrier)); };
    auto empty_barrier_of = [&](uint32_t i) { return (Barrier*)(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + (kNumStages + i) * sizeof(Barrier)); };

    if (warp_idx == kNumMathThreads / 32 + 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barrier_of(i)->init(1);
            empty_barrier_of(i)->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        fence_barrier_init();
    }
    kNumTMAMulticast > 1 ? cluster_sync_relaxed() : (void)__syncthreads();

    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    griddepcontrol_wait();

    Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>
        scheduler(shape_m, shape_n, shape_k, grouped_layout);
    scheduler.kNumGroupsRuntime = kNumGroups;
    using Sched = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>;

    // Pipeline cursor advanced *inside* the loop post-body (upstream form).
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k) {
        ++k;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= (stage_idx == 0);
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // ================= TMA producer =================
        setmaxnreg_dec<kNumTMARegisters>();
        // Third TMA warp issues (upstream comment: warp 0/1 of the TMA group
        // may still be finishing WGMMA-era register shuffles on tiny BLOCK_M).
        if (warp_idx == kNumMathThreads / 32 + 2 && elect_one_sync()) {
            uint32_t m_block_idx, n_block_idx;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                if (gemm_type_is_k_grouped(kGemmType) && scheduler.current_shape_k == 0)
                    continue;

                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA && is_tma_multicast_valid) ? kNumTMAMulticast : 1u;
                const uint32_t num_tma_multicast_b = (!kIsTMAMulticastOnA && is_tma_multicast_valid) ? kNumTMAMulticast : 1u;
                const uint16_t cta_mask = (uint16_t)((1u << kNumTMAMulticast) - 1u);

                const uint32_t num_total_k_blocks = ceil_div_u32(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barrier_of(stage_idx)->wait(phase ^ 1);
                    Barrier* fb = full_barrier_of(stage_idx);

                    // Group-aware (contiguous: per-m-row group base) indices.
                    constexpr bool kWithGroupOffsetA = (kGemmType == GemmType::MGroupedMasked);
                    const uint32_t m_idx = scheduler.template get_global_idx<kWithGroupOffsetA, Sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    const uint32_t n_idx = scheduler.template get_global_idx<kMajorB == MAJOR_MN, Sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                    uint32_t k_a_idx = scheduler.template get_global_idx<kMajorA == MAJOR_MN, Sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<kMajorB == MAJOR_MN, Sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // A: K-major -> box [BLOCK_K, BLOCK_M] @ (k, m);
                    //    MN-major -> box [BLOCK_M, BLOCK_K] @ (m, k).
                    if (kMajorA == MAJOR_K) {
                        if (num_tma_multicast_a > 1) tma_load_2d_multicast(&tensor_map_a, fb, smem_a_of(stage_idx), cta_mask, k_a_idx, m_idx);
                        else tma_load_2d(&tensor_map_a, fb, smem_a_of(stage_idx), kEvictNormalHint, k_a_idx, m_idx);
                    } else {
                        if (num_tma_multicast_a > 1) tma_load_2d_multicast(&tensor_map_a, fb, smem_a_of(stage_idx), cta_mask, m_idx, k_a_idx);
                        else tma_load_2d(&tensor_map_a, fb, smem_a_of(stage_idx), kEvictNormalHint, m_idx, k_a_idx);
                    }
                    if (kMajorB == MAJOR_K) {
                        if (num_tma_multicast_b > 1) tma_load_2d_multicast(&tensor_map_b, fb, smem_b_of(stage_idx), cta_mask, k_b_idx, n_idx);
                        else tma_load_2d(&tensor_map_b, fb, smem_b_of(stage_idx), kEvictNormalHint, k_b_idx, n_idx);
                    } else {
                        if (num_tma_multicast_b > 1) tma_load_2d_multicast(&tensor_map_b, fb, smem_b_of(stage_idx), cta_mask, n_idx, k_b_idx);
                        else tma_load_2d(&tensor_map_b, fb, smem_b_of(stage_idx), kEvictNormalHint, n_idx, k_b_idx);
                    }
                    fb->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }
            if (kNumTMAMulticast > 1) {
                for (uint32_t s = 0; s < kNumStages; advance_pipeline(s))
                    empty_barrier_of(stage_idx)->wait(phase ^ 1);
            }
        }
    } else {
        // ================= Math warpgroup(s) =================
        setmaxnreg_inc<kNumMathRegisters>();
        const uint32_t math_wg_idx = threadIdx.x / 128;

        // Stage-0 descriptors, shfl-broadcast once, then walk with the cheap
        // `_lo` add per wgmma issue (upstream optimization).
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        GmmaDescriptor a_desc0 = make_gmma_desc_t<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, 2>(
            smem_a_of(0), math_wg_idx * WGMMA_M, 0);
        GmmaDescriptor b_desc0 = make_gmma_desc_t<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, 2>(
            smem_b_of(0), 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc0.lo, 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc0.lo, 0);

        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            if (gemm_type_is_k_grouped(kGemmType) && scheduler.current_shape_k == 0)
                continue;

            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA_M ? BLOCK_M : WGMMA_M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Which threads commit the epilogue (BLOCK_M < 64 -> one WG only).
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA_M);
            const bool do_wgmma_store = BLOCK_M >= 64 || warp_idx < kNumWGMMAStoreThreads / 32;

            auto empty_barrier_arrive = [&](uint32_t s) {
                if (kNumTMAMulticast == 1) {
                    if (lane_idx == 0) empty_barrier_of(s)->arrive();
                } else {
                    const uint32_t target_cta = scheduler.is_peer_cta_alive ? lane_idx : get_block_rank_in_cluster();
                    if (lane_idx < kNumTMAMulticast) empty_barrier_of(s)->arrive_cluster(target_cta);
                }
            };

            const uint32_t num_total_k_blocks = ceil_div_u32(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Precomputed stage bases (16B units) — see the `_lo` trick.
                const uint32_t a_base = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const uint32_t b_base = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                full_barrier_of(stage_idx)->wait(phase);

                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                    float* shifted = accum + kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA_K; ++k) {
                        const uint32_t atom_k_idx = k * WGMMA_K / BLOCK_ATOM_K;
                        GmmaDescriptor a_desc, b_desc;
                        a_desc.lo = advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, 2>(
                            a_base, local_idx * WAVE_BLOCK_M, (k * WGMMA_K) % BLOCK_ATOM_K,
                            atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.lo = advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, 2>(
                            b_base, 0, (k * WGMMA_K) % BLOCK_ATOM_K,
                            atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        wgmma_bf16<BLOCK_N, kTransA, kTransB>(a_desc.desc_, b_desc.desc_, shifted, 1);
                    }
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait_group<0>();
                empty_barrier_arrive(stage_idx);
            }

            // ---- epilogue ----
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : kSwizzleDMode / 2;
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA_M / 4;  // 16 rows/warp
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 && BLOCK_N / TMA_D_BLOCK_N <= 32,
                             "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            if (!do_wgmma_store)
                continue;

            // Wait for the previous block's bulk stores before overwriting D.
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                tma_store_wait<0>();
            named_barrier_sync(kNumWGMMAStoreThreads, 4);

            if constexpr (cd_dtype == 1) {
                // ===== BF16 out: STSM into the TMA-swizzled D staging =====
                DG_STATIC_ASSERT(kSwizzleDMode > 0, "BF16 output requires swizzled D");
                DG_STATIC_ASSERT(kNumAccum % 4 == 0, "STSM x2 vectorization");
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                    const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                    float* shifted = accum + kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
                        // Split the BLOCK_N row into TMA store atoms; each STSM
                        // writes one 16B bank group at the swizzled address.
                        uint8_t* smem_ptr;
                        {
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            const uint32_t atom_offset = i / (TMA_D_BLOCK_N / 8);
                            const uint32_t in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                            const uint32_t bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                            // Reshape (BLOCK_M, sw/16B) as (BLOCK_M*sw/16B/8, 8)
                            // and XOR-swizzle the bank group by the row.
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            const uint32_t row = kHasShortcut ? (in_atom_offset / 8 + lane_idx)
                                                             : (bank_group_index / 8);
                            uint32_t col = kHasShortcut ? in_atom_offset : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);
                            smem_ptr = smem_d +
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +   // warp rows
                                m_offset * kSwizzleDMode +                        // wave rows
                                atom_offset * BLOCK_M * kSwizzleDMode +           // n-atom
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                        }
                        // 16 lanes' addresses are consumed by STSM x2.
                        stsm_x2_b16_n(cvta_shared_to_u32(smem_ptr),
                                     cvt_bf16x2_f32(shifted[i * 4 + 0], shifted[i * 4 + 1]),
                                     cvt_bf16x2_f32(shifted[i * 4 + 2], shifted[i * 4 + 3]));
                    }
                }
            } else {
                // ===== FP32 out: plain st.shared, float2 vectorized =====
                // (staging is row-major [BLOCK_M, BLOCK_N] floats — no swizzle)
                float* fstage = (float*)smem_d;
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++local_idx) {
                    const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                    float* shifted = accum + kNumAccum * local_idx;
                    float2* sd_0 = (float2*)(fstage + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                    float2* sd_1 = (float2*)(fstage + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
                        sd_0[i * 4] = make_float2(shifted[i * 4 + 0], shifted[i * 4 + 1]);
                        sd_1[i * 4] = make_float2(shifted[i * 4 + 2], shifted[i * 4 + 3]);
                    }
                }
            }
            tma_store_fence();
            named_barrier_sync(kNumWGMMAStoreThreads, 4);

            // ---- bulk store: split into BLOCK_N/TMA_D_BLOCK_N atom stores,
            //      one per thread; REDUCE-ADD when accumulating into C ----
            const uint32_t m_idx = scheduler.template get_global_idx<!gemm_type_is_m_grouped_contiguous(kGemmType), Sched::IndexType::MN>(
                shape_m, BLOCK_M, m_block_idx);
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                const uint32_t in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                // Swizzled D staging atoms are BLOCK_M * TMA_D_BLOCK_N * CD_ELEM
                // bytes apart: in elements that's in_block_n_offset * BLOCK_M.
                uint8_t* src = smem_d + in_block_n_offset * BLOCK_M;
                if (kWithAccumulation)
                    tma_reduce_add_2d(&tensor_map_cd, src, n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                else
                    tma_store_2d(&tensor_map_cd, src, n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

} // namespace dg

