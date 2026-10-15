// DeepGEMM-RS: mHC megakernel — the multi-head connector layer, fused.
//
// Faithful port of upstream DeepGEMM `sm100_mega_mhc.cuh` (704 lines) +
// `epilogue/sm100_mega_mhc.cuh` (428) + `layout/mega_mhc.cuh` + the
// SplitBarrier grid-tag scheduler.
//
// FORMAT CONTRACT ("all parts in fp8"): the HC projection GEMM runs in TF32
// (tensor-core FP32; the `fn` matrix is FP32 router state — quantizing router
// weights to FP8 would change model numerics, and TF32 IS the tensor-core
// path for FP32 inputs). The residual stream is BF16 — upstream's shared
// numerical boundary: "this BF16 round is the shared numerical boundary for
// all consumers". The OUTPUT path is fully FP8 with UE8M0 scale factors
// packed 4x/int32 in two layouts (TMA-aligned GEMM SF / routed+shared
// Mega-MoE SF), exactly as upstream.
//
// ===========================================================================
// WHAT THE KERNEL COMPUTES (per token t, hidden H, routes R=4, HC=24)
// ===========================================================================
//   x[t, H]  residual[t, r, H]   (r: HC route)
//   post_mix[t, r]              comb_res_mix[t, r_in, r_out]
//   shifted_prev_mix[t, r]      fn[24, R*H]
//   ─────────────────────────────────────────────────────────────────────
//   (1) Post workers (WG1-2, 2 WGs): for every (t, k-block, r_in, r_out):
//         new_residual[t, r_out] = post_mix[t, r_out] * x[t]
//                                 + sum_ri comb[r_in, r_out] * residual[t, r_in]
//       rounded to BF16 (the shared numerical boundary), stored back into the
//       residual SMEM tile (in place) + squared-norm accumulation; the
//       tile is tcgen05.st'd into TMEM as the MMA's A operand.
//       Shifted additionally forms X1[t, r] = sum_r prev[r] * new_residual.
//   (2) MMA (WG0 warp 2): A from TMEM (post output), B = fn (SMEM):
//         hc[24, tokens] = fn @ new_residual^T   — TF32, split-K over K
//         (K = R*H split into kNumSplits slices; task = (m-block, split)).
//   (3) WG3 workspace epilogue: drains one accumulator block per task into
//       the FP32 split scratch; rows past num_tokens are zeroed.
//   (4) Mix (WG4, shifted): per token, reduce the split partials, RMS-scale,
//       affine (scales/bases), sigmoid for pre/post, sinkhorn-normalize the
//       4x4 comb matrix; writes new_prev_mix/new_post_mix/new_comb_res_mix.
//   (5) Norm: fused (normal mHC): each WG runs Mix then RMSNorm + optional
//       FP8 store for one token; shifted: X1 materialized in y_bf16, then a
//       shifted Norm (2 tokens/task) consumes X1 + split x1 sums.
//
//   Split barriers (gmem, grid-tag protocol): Mix waits for all splits of an
//   m-block; shifted Norm waits for X1 stores + statistics. Tags embed
//   (grid_idx+1)*kNumSplitBarriers*kNumMaxSplits so stale launches never alias.
//
// ===========================================================================
// WARP MAP (768 threads = 6 WGs; NO cluster — 1-SM TF32 MMA)
// ===========================================================================
//   WG0 (warps 0-3): producers
//     warp 0: residual+x TMA loads (IO ring, 4 stages)
//     warp 1: fn TMA loads (Fn ring, 2 stages)
//     warp 2: HC MMA (TF32 TS form; A from TMEM per-UMMA-K slice)
//     warp 3: output-store (new_residual TMA 3D store, EVICT_LAST in normal),
//             coeff prefetch (post/comb[/prev] 2D loads), Norm arrival
//   WG1-2 (warps 4-11): Post workers — one per WG; round-robin UMMA atoms;
//     write A into TMEM (tcgen05.st), gate MMA on store_ready barriers
//   WG3 (warps 12-15): workspace epilogue — drains accumulators to gmem
//   WG4 (warps 16-19): shifted ? Mix workers (1/token) : repurposed Normal
//   WG5 (warps 20-23): shifted ? shifted Norm : Normal
//   Normal mode repurposes every producer WG as soon as it retires (all
//   static roles join the shared Norm task queue after a named barrier).
// ===========================================================================

namespace dg {
namespace mega_mhc {

constexpr uint32_t kNumRoutes = 4;
constexpr uint32_t kNumHCOutputs = kNumRoutes * (kNumRoutes + 2);

// FP8 SF layout constants (32 elems per SF, 128 per packed word).
constexpr uint32_t kHiddenPerSF = 32;
constexpr uint32_t kHiddenPerSFWord = 128;

// Kernel tile and pipeline geometry.
constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = kNumHCOutputs;
constexpr uint32_t BLOCK_K = 64;
constexpr uint32_t kSwizzleMode = 128;
constexpr uint32_t kFnAtomK = kSwizzleMode / 4;  // 32 floats
constexpr uint32_t kSwizzleAlignment = 8 * kSwizzleMode;
constexpr uint32_t kNumIOStages = 4;
constexpr uint32_t kNumFnStages = 2;
constexpr uint32_t kNumAStages = 4;
constexpr uint32_t kNumAccumStages = 2;
constexpr uint32_t kNumCoeffStages = 2;
constexpr uint32_t kNumWarpsPerWG = 4;
constexpr uint32_t kNumThreads = 6 * kNumWarpsPerWG * 32;

// Split-K workspace geometry.
constexpr uint32_t kNumGemmPartialElementsPerTask = BLOCK_M * kNumHCOutputs;
constexpr uint32_t kNumSqrSumPartialElementsPerTask = BLOCK_M;
constexpr uint32_t kDefaultNumSplits = 16;
constexpr uint32_t kNumMaxSplits = 64;
constexpr uint32_t kNumSplitBarriers = 2;
constexpr uint64_t kSplitBarrierLineBytes = 128;
constexpr uint32_t kNumMaxTokens = 1u << 20;
constexpr uint32_t kNumMaxMBlocks = (kNumMaxTokens + BLOCK_M - 1) / BLOCK_M;

// Host/device ABI payloads (tensor maps stay standalone params).
struct MixArgs {
    const float* scales;
    const float* bases;
    float* new_prev_mix;
    float* new_post_mix;
    float* new_comb_res_mix;
    float hc_norm_eps;
    float hc_pre_eps;
    float hc_post_scale;
    float sinkhorn_eps;
    uint32_t num_sinkhorn_iters;
};

struct NormArgs {
    uint32_t num_tokens;
    const uint16_t* weight;        // bf16 [H]
    const uint16_t* new_residual;  // bf16 [T, R, H]
    float eps;
    float scale;
    uint16_t* y_bf16;              // bf16 [T, H] (or scratch)
    uint8_t* y_fp8;                // e4m3 [T, H] (optional)
    uint32_t* y_primary_sf;        // int32 [T, H/128] (gemm or routed layout)
    int64_t y_primary_sf_stride_token;
    int64_t y_primary_sf_stride_word;
    uint32_t* y_shared_sf;         // int32, shared-expert layout (optional)
    int64_t y_shared_sf_stride_word;
};

// View over the global split scratch:
//   gemm_partials            [m_blocks, splits, BLOCK_M, 24]
//   hc_norm_sqr_sum_partials [m_blocks, splits, BLOCK_M]
//   x1_sqr_sum_partials      [m_blocks, splits, BLOCK_M]
template <uint32_t kNumSplits = kDefaultNumSplits>
struct Workspace {
    void* gmem_scratch;
    uint32_t num_m_blocks;

    Workspace(void* scratch, uint32_t m_blocks)
        : gmem_scratch(scratch), num_m_blocks(m_blocks) {}

    static uint32_t get_num_m_blocks_fn(uint32_t num_tokens) {
        return (num_tokens + BLOCK_M - 1) / BLOCK_M;
    }
    static uint64_t get_num_scratch_bytes_fn(uint32_t m_blocks, uint32_t splits) {
        const uint64_t tasks = (uint64_t)m_blocks * splits;
        return tasks * (kNumGemmPartialElementsPerTask + 2 * kNumSqrSumPartialElementsPerTask) * 4;
    }
    DG_DEVICE float* get_gemm_partial_ptr(uint32_t task) const {
        return (float*)gmem_scratch + (uint64_t)task * kNumGemmPartialElementsPerTask;
    }
    DG_DEVICE float* get_hc_norm_sqr_sum_partial_ptr(uint32_t task) const {
        float* base = get_gemm_partial_ptr(num_m_blocks * kNumSplits);
        return base + (uint64_t)task * kNumSqrSumPartialElementsPerTask;
    }
    DG_DEVICE float* get_x1_sqr_sum_partial_ptr(uint32_t task) const {
        float* base = get_hc_norm_sqr_sum_partial_ptr(num_m_blocks * kNumSplits);
        return base + (uint64_t)task * kNumSqrSumPartialElementsPerTask;
    }
};

// ---------------------------------------------------------------------------
// Shared storage. Route-major TMA layouts keep every 64x64 tile contiguous.
// ---------------------------------------------------------------------------
struct SharedStorage {
    // IO ring: residual [routes][M*K/2 bf16x2] + x [M*K/2 bf16x2], per stage.
    alignas(kSwizzleAlignment) uint16_t residual[kNumIOStages][kNumRoutes][BLOCK_M * BLOCK_K / 2];
    alignas(kSwizzleAlignment) uint16_t x[kNumIOStages][BLOCK_M * BLOCK_K / 2];

    // Fn ring: [BLOCK_K/kFnAtomK][routes][24 * kFnAtomK] floats per stage.
    alignas(kSwizzleAlignment) float fn[kNumFnStages][BLOCK_K / kFnAtomK][kNumRoutes][kNumHCOutputs * kFnAtomK];

    // Coefficients (TMA-native per-token layouts).
    alignas(kSwizzleMode) float pre_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes];
    alignas(kSwizzleMode) float post_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes];
    alignas(kSwizzleMode) float comb_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes * kNumRoutes];

    // Post-owned partials + fused-Normal scratch.
    alignas(kSwizzleMode) float hc_norm_sqr_sums[2][BLOCK_M];
    alignas(kSwizzleMode) float x1_sqr_sums[2][BLOCK_M];
    alignas(kSwizzleMode) float normal_scratch[BLOCK_M];

    // Full/empty barrier pairs (static roles; no CTA-wide block-boundary sync).
    alignas(8) Barrier full_io_barriers[kNumIOStages];
    alignas(8) Barrier empty_io_barriers[kNumIOStages];
    alignas(8) Barrier store_ready_io_barriers[kNumIOStages];

    alignas(8) Barrier full_fn_barriers[kNumFnStages];
    alignas(8) Barrier empty_fn_barriers[kNumFnStages];

    alignas(8) Barrier full_a_barriers[kNumAStages];
    alignas(8) Barrier empty_a_barriers[kNumAStages];

    alignas(8) Barrier full_accum_barriers[kNumAccumStages];
    alignas(8) Barrier empty_accum_barriers[kNumAccumStages];

    alignas(8) Barrier full_coeff_barriers[kNumCoeffStages];
    alignas(8) Barrier empty_coeff_barriers[kNumCoeffStages];

    alignas(8) Barrier full_stats_barrier;
    alignas(8) Barrier empty_stats_barrier;

    // Mix arrival gating + the shared Norm task queue.
    alignas(8) Barrier mix_arrival_barrier;
    uint32_t tmem_base;
    uint32_t next_norm_task_ticket;
};

// ---------------------------------------------------------------------------
// SplitBarrier: grid-tag protocol (see mega_gate::ScoreBarrier for rationale).
// ---------------------------------------------------------------------------
template <uint32_t kBarrierIndex>
struct SplitBarrier {
    static DG_DEVICE uint64_t* get_ptr(uint64_t* bars, uint32_t m_block) {
        const uint64_t line = (uint64_t)kBarrierIndex * kNumMaxMBlocks + m_block;
        return (uint64_t*)((uint8_t*)bars + line * kSplitBarrierLineBytes);
    }
    static DG_DEVICE uint64_t encode(uint64_t grid_idx, uint32_t arrived = 0) {
        return (grid_idx + 1) * kNumSplitBarriers * kNumMaxSplits + arrived;
    }
    static DG_DEVICE void init(uint64_t* bars, uint32_t m_block) {
        st_rel_gpu_u64(get_ptr(bars, m_block), encode(get_grid_idx()));
    }
    template <uint32_t kNumSplits>
    static DG_DEVICE void wait_init(uint64_t* bars, uint32_t m_block) {
        const uint64_t base = encode(get_grid_idx());
        while (ld_acq_gpu_u64(get_ptr(bars, m_block)) - base >= kNumSplits) {}
    }
    template <uint32_t kNumSplits>
    static DG_DEVICE void arrive(uint64_t* bars, uint32_t m_block) {
        // red.async: SM100 retires the release while the CTA moves on.
        red_async_inc_rel_u64(get_ptr(bars, m_block));
    }
    template <uint32_t kNumSplits>
    static DG_DEVICE void wait(uint64_t* bars, uint32_t m_block) {
        const uint64_t all = encode(get_grid_idx(), kNumSplits);
        while (ld_acq_gpu_u64(get_ptr(bars, m_block)) != all) {}
    }
};
using NormBarrier = SplitBarrier<0>;
using MixBarrier = SplitBarrier<1>;

// ---------------------------------------------------------------------------
// BF16x2 swizzled SMEM addressing (128B swizzle atoms).
// ---------------------------------------------------------------------------
DG_DEVICE uint32_t get_bf16x2_offset(uint32_t row, uint32_t col) {
    // kNumColsPerRow = 128B / 4B = 32 bf16x2; 4 per 16B bank group.
    constexpr uint32_t kNumColsPerRow = kSwizzleMode / 4;
    constexpr uint32_t kNumColsPerBank = 4;
    const uint32_t bg = (col / kNumColsPerBank) ^ (row & (kSwizzleMode / 16 - 1));
    return row * kNumColsPerRow + bg * kNumColsPerBank + col % kNumColsPerBank;
}
// Comb rows use a separate 64B swizzle (4-float groups across banks).
DG_DEVICE float load_comb_coeff(const float* row, uint32_t row_idx, uint32_t coeff) {
    const uint32_t bg = (coeff / 4) ^ ((row_idx / 2) & 3);
    return row[bg * 4 + coeff % 4];
}

// ---------------------------------------------------------------------------
// Epilogue: Mix (per token), shared + fused Normal, shifted Norm.
// ---------------------------------------------------------------------------
constexpr uint32_t kHiddenPerNormPack = 256;
constexpr uint32_t kHiddenPerNormLane = kHiddenPerNormPack / 32;  // 8
constexpr uint32_t kNumBF16PairsPerLane = 16 / 4;                 // 4 pairs per uint4
constexpr uint32_t kNumLanesPerSF = kHiddenPerSF / kHiddenPerNormLane;      // 4
constexpr uint32_t kNumSFWordsPerNormPack = kHiddenPerNormPack / kHiddenPerSFWord;  // 2
constexpr uint32_t kNumLanesPerSFWord = kHiddenPerSFWord / kHiddenPerNormLane;      // 16

DG_DEVICE float2 normalize_bf16_pair(uint32_t input2, uint32_t weight2, float scale) {
    float i0, i1, w0, w1;
    bf16x2_to_f32x2(input2, i0, i1);
    bf16x2_to_f32x2(weight2, w0, w1);
    return make_float2(i0 * scale * w0, i1 * scale * w1);
}

template <uint32_t SF_BLOCK_M>
DG_DEVICE uint32_t get_shared_sf_row_idx(uint32_t token_idx) {
    // Mega MoE SF pages interleave four 32-row groups per 128-row page.
    const uint32_t idx = token_idx % SF_BLOCK_M;
    return token_idx / SF_BLOCK_M * ((SF_BLOCK_M + 127u) & ~127u) +
           (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
}

// FP8 store + UE8M0 packing: 4 bf16x2 -> 8 fp8; SF packed from 4 lanes.
DG_DEVICE uint32_t store_fp8_and_pack_sf(const NormArgs& na, uint64_t out_off,
                                          uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
                                          uint32_t sf_exp) {
    // Splat the pow-2 sf_inv into both packed halves (bf16 mul is exact).
    const uint32_t inv2 = ue8m0_sf_inv_bf16(sf_exp);
    const uint32_t p0 = scale_bf16x2_into_fp8x4(v0, v1, inv2, inv2);
    const uint32_t p1 = scale_bf16x2_into_fp8x4(v2, v3, inv2, inv2);
    *(uint2*)(na.y_fp8 + out_off) = make_uint2(p0, p1);

    // Pack four 32-element SF groups from each 16-lane group into one int32.
    const uint32_t paired = __byte_perm(sf_exp, __shfl_xor_sync(0xffffffff, sf_exp, kNumLanesPerSF), 0x1140);
    return __byte_perm(paired, __shfl_xor_sync(0xffffffff, paired, 2 * kNumLanesPerSF), 0x5410);
}

// One token's Mix: reduce split partials -> RMS scale -> affine -> sigmoid
// (pre/post) / comb logits -> sinkhorn; write the next mixes.
template <uint32_t kHidden, bool kIsShifted, uint32_t kNumSplits>
DG_DEVICE float run_mix_task(const Workspace<kNumSplits>& ws, uint64_t* split_bars,
                              const MixArgs& ma, uint32_t token_idx, uint32_t lane_idx) {
    const uint32_t m_block = token_idx / BLOCK_M;
    const uint32_t row_idx = token_idx % BLOCK_M;
    MixBarrier::wait<kNumSplits>(split_bars, m_block);

    // Reduce HC norm statistics across splits (strided by lane).
    const bool is_hc_lane = lane_idx < kNumHCOutputs;
    const uint32_t first_task = m_block * kNumSplits;
    const float* norm_partials = ws.get_hc_norm_sqr_sum_partial_ptr(first_task);
    float hc_norm_sqr_sum = 0.0f;
    for (uint32_t sp = lane_idx; sp < kNumSplits; sp += 32)
        hc_norm_sqr_sum += norm_partials[sp * kNumSqrSumPartialElementsPerTask + row_idx];
    hc_norm_sqr_sum = warp_reduce_sum_f32(hc_norm_sqr_sum, 32);

    const uint32_t safe_out = dg_min(lane_idx, kNumHCOutputs - 1);
    const float* gemm_partials = ws.get_gemm_partial_ptr(first_task);
    float hc_output = 0.0f;
    #pragma unroll
    for (uint32_t sp = 0; sp < kNumSplits; ++sp)
        hc_output += gemm_partials[sp * kNumGemmPartialElementsPerTask +
                                   row_idx * kNumHCOutputs + safe_out];
    hc_output = is_hc_lane ? hc_output : 0.0f;
    const float hc_rms_scale = rsqrtf(
        hc_norm_sqr_sum * (1.0f / (float)(kNumRoutes * kHidden)) + ma.hc_norm_eps);

    // Lanes map to [Pre R | Post R | Comb R x R | inactive].
    constexpr uint32_t kFirstCombLane = 2 * kNumRoutes;
    const bool is_pre_lane = lane_idx < kNumRoutes;
    const bool is_pre_or_post = lane_idx < kFirstCombLane;
    const bool is_comb_lane = is_hc_lane && !is_pre_or_post;
    float mix_value = 0.0f;
    if (is_hc_lane) {
        const uint32_t scale_idx = is_pre_lane ? 0u : (is_pre_or_post ? 1u : 2u);
        const float affine = hc_output * hc_rms_scale * ma.scales[scale_idx] + ma.bases[lane_idx];
        if (is_pre_or_post) {
            const float sig = fast_rcp_f32(1.0f + __expf(-affine));
            mix_value = is_pre_lane ? sig + ma.hc_pre_eps : sig * ma.hc_post_scale;
        } else {
            mix_value = affine;  // comb logits
        }
    }
    __syncwarp();

    const auto sinkhorn_normalize = [&](float value) {
        value = is_comb_lane ? value : 0.0f;
        const float row_max = warp_reduce_max_f32(value, kNumRoutes);
        value = __expf(value - row_max);
        float row_sum = warp_reduce_sum_f32(value, kNumRoutes);
        value = value * fast_rcp_f32(row_sum) + ma.sinkhorn_eps;

        const auto col_sum = [](float e) {
            // For 4 routes: XORs combine adjacent rows, then row pairs.
            float s = e + __shfl_xor_sync(0xffffffff, e, kNumRoutes);
            s += __shfl_xor_sync(0xffffffff, s, 6 * kNumRoutes);
            return s;
        };
        value *= fast_rcp_f32(col_sum(value) + ma.sinkhorn_eps);
        #pragma unroll 1
        for (uint32_t it = 1; it < ma.num_sinkhorn_iters; ++it) {
            row_sum = warp_reduce_sum_f32(value, kNumRoutes);
            value *= fast_rcp_f32(row_sum + ma.sinkhorn_eps);
            value *= fast_rcp_f32(col_sum(value) + ma.sinkhorn_eps);
        }
        return value;
    };
    const float normalized_comb = sinkhorn_normalize(mix_value);

    // Store the next mixes (pre only in shifted mode).
    if (is_hc_lane && (kIsShifted || !is_pre_lane)) {
        float* out = is_pre_lane ? ma.new_prev_mix
                   : is_comb_lane ? ma.new_comb_res_mix : ma.new_post_mix;
        const uint32_t stride = is_comb_lane ? kNumRoutes * kNumRoutes : kNumRoutes;
        const uint32_t idx = is_comb_lane ? lane_idx - kFirstCombLane : (lane_idx & (kNumRoutes - 1));
        out[(uint64_t)token_idx * stride + idx] = is_comb_lane ? normalized_comb : mix_value;
    }
    __syncwarp();
    return mix_value;  // pre coefficient broadcast to the Norm worker
}

// ===========================================================================
// The megakernel
// ===========================================================================
// task = (m-block, k-split); kNumKBlocksPerSplit blocks per split; long splits
// take one extra block (kNumLongSplits).
// ---------------------------------------------------------------------------
// Fused Normal worker (1 token/task): Mix -> accumulate routes -> RMSNorm ->
// optional FP8 store with primary (+shared) UE8M0 SF packing.
// ---------------------------------------------------------------------------
template <uint32_t kHidden, uint32_t kNumSMs, bool kStoreBF16, bool kStoreFP8,
          uint32_t SF_BLOCK_M, uint32_t kNumSplits>
DG_DEVICE void run_normal_worker(SharedStorage& smem, const Workspace<kNumSplits>& ws,
                                  uint32_t wg, uint32_t warp_in_wg, uint32_t lane_idx,
                                  uint64_t* split_bars, const MixArgs& ma, const NormArgs& na) {
    constexpr uint32_t kNumWGs = kNumThreads / (kNumWarpsPerWG * 32);
    constexpr uint32_t kNumPacksPerWarp = kHidden / (kHiddenPerNormPack * kNumWarpsPerWG);
    static_assert(kHidden % (kHiddenPerNormPack * kNumWarpsPerWG) == 0, "hidden/warps");
    constexpr uint32_t kNumWarpSqrSums = kNumWGs * kNumWarpsPerWG;
    static_assert(kNumWarpSqrSums * 4 + kNumWGs * (4 + kNumRoutes * 4) <= sizeof(smem.normal_scratch),
                  "scratch");

    // Scratch: [warp sums | token indices | pre coefficients per WG].
    const uint32_t pack_begin = warp_in_wg * kNumPacksPerWarp;
    const uint32_t barrier_idx = 1 + wg;
    float* warp_sqr_sums = smem.normal_scratch;
    uint32_t* token_indices = (uint32_t*)(warp_sqr_sums + kNumWarpSqrSums);
    float* pre_coeffs = (float*)(token_indices + kNumWGs) + wg * kNumRoutes;

    while (true) {
        // Draw a task from the shared queue (warp 0 of each WG).
        if (warp_in_wg == 0) {
            uint32_t ticket = 0;
            if (elect_one_sync())
                ticket = atomic_add_shared_u32(&smem.next_norm_task_ticket, 1u);
            ticket = warp_exchange(ticket, 0);
            const uint32_t token = blockIdx.x + ticket * kNumSMs;
            token_indices[wg] = token;
            if (token < na.num_tokens) {
                const float pre = run_mix_task<kHidden, false>(ws, split_bars, ma, token, lane_idx);
                if (lane_idx < kNumRoutes)
                    pre_coeffs[lane_idx] = pre;
            }
        }
        named_barrier_sync(kNumWarpsPerWG * 32, barrier_idx);
        const uint32_t token = token_indices[wg];
        if (token >= na.num_tokens) return;
        const float pre_coeff = lane_idx < kNumRoutes ? pre_coeffs[lane_idx] : 0.0f;

        // Accumulate routes in FP32; RMS derives from the BF16-rounded values.
        const uint16_t* token_residual = na.new_residual + (uint64_t)token * kNumRoutes * kHidden;
        uint4 retained[kNumPacksPerWarp];
        float2 sqr_sum = make_float2(0.f, 0.f);
        #pragma unroll
        for (uint32_t po = 0; po < kNumPacksPerWarp; ++po) {
            const uint32_t pack_idx = pack_begin + po;
            const uint32_t h = pack_idx * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
            uint32_t* out = (uint32_t*)&retained[po];
            float2 values[kNumBF16PairsPerLane] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f),
                                                   make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
            #pragma unroll
            for (uint32_t r = 0; r < kNumRoutes; ++r) {
                const float coeff = warp_exchange(pre_coeff, r);
                const uint4 in = ld_evict_first_u128((const uint4*)(token_residual + r * kHidden + h));
                const uint32_t* iw = (const uint32_t*)&in;
                #pragma unroll
                for (uint32_t p = 0; p < kNumBF16PairsPerLane; ++p) {
                    float a0, a1;
                    bf16x2_to_f32x2(iw[p], a0, a1);
                    values[p].x += a0 * coeff;
                    values[p].y += a1 * coeff;
                }
            }
            #pragma unroll
            for (uint32_t p = 0; p < kNumBF16PairsPerLane; ++p) {
                out[p] = f32x2_to_bf16x2(values[p].x, values[p].y);
                float v0, v1;
                bf16x2_to_f32x2(out[p], v0, v1);
                sqr_sum.x += v0 * v0;
                sqr_sum.y += v1 * v1;
            }
        }

        const float warp_sum = warp_reduce_sum_f32(sqr_sum.x + sqr_sum.y, 32);
        warp_sqr_sums[wg * kNumWarpsPerWG + warp_in_wg] = warp_sum;
        named_barrier_sync(kNumWarpsPerWG * 32, barrier_idx);
        float rms = lane_idx < kNumWarpsPerWG ? warp_sqr_sums[wg * kNumWarpsPerWG + lane_idx] : 0.f;
        rms = warp_reduce_sum_f32(rms, kNumWarpsPerWG);
        rms = warp_exchange(rms, 0);
        constexpr float kInvHidden = 1.0f / (float)kHidden;
        rms = rsqrtf(rms * kInvHidden + na.eps) * na.scale;
        const uint64_t out_off = (uint64_t)token * kHidden;
        uint32_t shared_row = 0;
        if (SF_BLOCK_M > 0)
            shared_row = get_shared_sf_row_idx<SF_BLOCK_M>(token);

        const uint16_t* w_base = na.weight + pack_begin * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
        uint4 weight = ld_evict_first_u128((const uint4*)w_base);
        #pragma unroll
        for (uint32_t po = 0; po < kNumPacksPerWarp; ++po) {
            const uint32_t pack_idx = pack_begin + po;
            const bool has_next = po + 1 < kNumPacksPerWarp;
            uint4 next_weight;
            if (has_next)
                next_weight = ld_evict_first_u128((const uint4*)(w_base + (po + 1) * (kHiddenPerNormPack / kHiddenPerNormLane) * 4));

            const uint32_t h = pack_idx * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
            uint32_t* vals = (uint32_t*)&retained[po];          // written back in place
            const uint32_t* w32 = (const uint32_t*)&weight;
            #pragma unroll
            for (uint32_t p = 0; p < kNumBF16PairsPerLane; ++p) {
                const float2 n2 = normalize_bf16_pair(vals[p], w32[p], rms);
                vals[p] = f32x2_to_bf16x2(n2.x, n2.y);
            }
            if (kStoreBF16)
                *(uint4*)(na.y_bf16 + out_off + h) = retained[po];
            if (kStoreFP8) {
                // amax over the pack (bf16 domain), 4-lane group reduce.
                const uint32_t amax4[4] = {vals[0], vals[1], vals[2], vals[3]};
                const uint32_t amax_bf = get_packed_bf16_amax(amax4);
                float amax_f;
                bf16x2_to_f32x2(amax_bf, amax_f, amax_f);  // both halves are the scalar max
                (void)amax_f;
                // bf16 -> f32 amax: take the larger half explicitly.
                float a0, a1;
                bf16x2_to_f32x2(amax_bf, a0, a1);
                const float amax = fmaxf(a0, a1);
                const float amax_red = warp_reduce_max_f32(amax, kNumLanesPerSF);
                const uint32_t sf_word = pack_idx * kNumSFWordsPerNormPack + lane_idx / kNumLanesPerSFWord;
                const uint32_t packed = store_fp8_and_pack_sf(
                    na, out_off + h, vals[0], vals[1], vals[2], vals[3],
                    get_ue8m0_sf_exp_f32<true>(amax_red));
                if (SF_BLOCK_M > 0) {
                    if (lane_idx % kNumLanesPerSFWord == 0) {
                        na.y_primary_sf[(uint64_t)token * na.y_primary_sf_stride_token +
                                        sf_word * na.y_primary_sf_stride_word] = packed;
                        na.y_shared_sf[(int64_t)sf_word * na.y_shared_sf_stride_word + shared_row] = packed;
                    }
                } else {
                    retained[po].x = packed;  // parked in dead registers
                }
            }
            if (has_next) weight = next_weight;
        }

        // Column-major SF stays in dead registers until the stores finish.
        if (kStoreFP8 && SF_BLOCK_M == 0) {
            if (lane_idx % kNumLanesPerSFWord == 0) {
                const int64_t stride = na.y_primary_sf_stride_word;
                uint32_t* sfp = na.y_primary_sf + (uint64_t)token * na.y_primary_sf_stride_token +
                    (pack_begin * kNumSFWordsPerNormPack + lane_idx / kNumLanesPerSFWord) * stride;
                #pragma unroll
                for (uint32_t po = 0; po < kNumPacksPerWarp; ++po) {
                    *sfp = retained[po].x;
                    sfp += kNumSFWordsPerNormPack * stride;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shifted Norm: X1 materialized in y_bf16; 2 tokens/task; partitions subdivide
// the hidden packs when the grid would be underfilled.
// ---------------------------------------------------------------------------
constexpr uint32_t kNumTokensPerShiftedTask = 2;

struct ShiftedPack {
    uint4 x1[2];     // token 0 / token 1 packs (16B each per lane)
    uint4 weight;    // rmsnorm weight pack
};

template <uint32_t kHidden, bool kStoreBF16, bool kStoreFP8, uint32_t SF_BLOCK_M,
          uint32_t kNumSplits>
DG_DEVICE void run_shifted_norm_task(const Workspace<kNumSplits>& ws, uint64_t* split_bars,
                                      const NormArgs& na, uint32_t task, uint32_t parts,
                                      uint32_t lane_idx) {
    constexpr uint32_t kNumPacks = kHidden / kHiddenPerNormPack;
    const uint32_t part = task % parts;
    const uint32_t begin = part * kNumPacks / parts;
    const uint32_t end = (part + 1) * kNumPacks / parts;
    const uint32_t m_idx = task / parts * kNumTokensPerShiftedTask;
    const uint32_t m_block = m_idx / BLOCK_M;
    const bool has_second = m_idx + 1 < na.num_tokens;
    uint2 shared_rows = {0u, 0u};

    NormBarrier::wait<kNumSplits>(split_bars, m_block);
    const auto load_pack = [&](uint64_t off, uint32_t h) {
        ShiftedPack p;
        p.x1[0] = ld_evict_first_u128((const uint4*)(na.y_bf16 + off));
        p.x1[1] = ld_evict_first_u128((const uint4*)(na.y_bf16 + off + (has_second ? kHidden : 0)));
        p.weight = ld_evict_first_u128((const uint4*)(na.weight + h));
        return p;
    };

    // Current/next in registers; prefetch 2 ahead (hides the BF16 reread).
    ShiftedPack pack, next;
    const uint32_t first_h = begin * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
    const uint64_t first_off = (uint64_t)m_idx * kHidden + first_h;
    pack = load_pack(first_off, first_h);
    if (begin + 1 < end)
        next = load_pack(first_off + kHiddenPerNormPack, first_h + kHiddenPerNormPack);

    float x1_sqr = 0.0f;
    if (lane_idx < 2 && m_idx + lane_idx < na.num_tokens) {
        const float* p = ws.get_x1_sqr_sum_partial_ptr(m_block * kNumSplits);
        #pragma unroll
        for (uint32_t sp = 0; sp < kNumSplits; ++sp)
            x1_sqr += p[sp * kNumSqrSumPartialElementsPerTask + m_idx % BLOCK_M + lane_idx];
    }
    constexpr float kInvHidden = 1.0f / (float)kHidden;
    float rms_scale = rsqrtf(x1_sqr * kInvHidden + na.eps) * na.scale;
    const float2 rms2 = {warp_exchange(rms_scale, 0), warp_exchange(rms_scale, 1)};
    if (SF_BLOCK_M > 0) {
        shared_rows.x = get_shared_sf_row_idx<SF_BLOCK_M>(m_idx);
        shared_rows.y = get_shared_sf_row_idx<SF_BLOCK_M>(m_idx + 1);
    }
    for (uint32_t pk = begin; pk < end; ++pk) {
        const uint32_t h = pk * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
        const uint64_t off = (uint64_t)m_idx * kHidden + h;
        const bool prefetch = pk + 2 < end;
        ShiftedPack pre_p;
        if (prefetch)
            pre_p = load_pack(off + 2 * kHiddenPerNormPack, h + 2 * kHiddenPerNormPack);
        uint32_t values[8] = {};  // 2 tokens x 4 bf16x2 words
        const uint32_t* xw = (const uint32_t*)&pack.x1;
        const uint32_t* ww = (const uint32_t*)&pack.weight;
        #pragma unroll
        for (uint32_t p = 0; p < kNumBF16PairsPerLane; ++p) {
            const float2 n0 = normalize_bf16_pair(xw[p], ww[p], rms2.x);
            values[p] = f32x2_to_bf16x2(n0.x, n0.y);
            const float2 n1 = normalize_bf16_pair(xw[kNumBF16PairsPerLane + p],
                                                  ww[kNumBF16PairsPerLane + p], rms2.y);
            values[kNumBF16PairsPerLane + p] = f32x2_to_bf16x2(n1.x, n1.y);
        }
        if (kStoreBF16) {
            *(uint4*)(na.y_bf16 + off) = *(const uint4*)&values[0];
            if (has_second)
                *(uint4*)(na.y_bf16 + off + kHidden) = *(const uint4*)&values[4];
        }
        if (kStoreFP8) {
            // Per-token amax (scalar bf16 tree over the pack's 4 words).
            float amax0_f, amax1_f;
            {
                const uint32_t a0 = get_packed_bf16_amax((const uint32_t*)&values[0]);
                const uint32_t a1 = get_packed_bf16_amax((const uint32_t*)&values[4]);
                float x0, x1, y0, y1;
                bf16x2_to_f32x2(a0, x0, x1);
                bf16x2_to_f32x2(a1, y0, y1);
                amax0_f = fmaxf(x0, x1);
                amax1_f = has_second ? fmaxf(y0, y1) : 0.0f;
            }
            const uint32_t sf0 = get_ue8m0_sf_exp_f32<true>(
                warp_reduce_max_f32(amax0_f, kNumLanesPerSF));
            const uint32_t sf1 = get_ue8m0_sf_exp_f32<true>(
                warp_reduce_max_f32(amax1_f, kNumLanesPerSF));
            const uint32_t sf_word = pk * kNumSFWordsPerNormPack + lane_idx / kNumLanesPerSFWord;
            const uint32_t packed0 = store_fp8_and_pack_sf(na, off, values[0], values[1],
                                                             values[2], values[3], sf0);
            if (lane_idx % kNumLanesPerSFWord == 0) {
                na.y_primary_sf[(uint64_t)m_idx * na.y_primary_sf_stride_token +
                                sf_word * na.y_primary_sf_stride_word] = packed0;
                if (SF_BLOCK_M > 0)
                    na.y_shared_sf[(int64_t)sf_word * na.y_shared_sf_stride_word + shared_rows.x]
                        = packed0;
            }
            if (has_second) {
                const uint32_t packed1 = store_fp8_and_pack_sf(na, off + kHidden,
                                                                 values[4], values[5],
                                                                 values[6], values[7], sf1);
                if (lane_idx % kNumLanesPerSFWord == 0) {
                    na.y_primary_sf[(uint64_t)(m_idx + 1) * na.y_primary_sf_stride_token +
                                    sf_word * na.y_primary_sf_stride_word] = packed1;
                    if (SF_BLOCK_M > 0)
                        na.y_shared_sf[(int64_t)sf_word * na.y_shared_sf_stride_word + shared_rows.y]
                            = packed1;
                }
            }
        }
        if (pk + 1 < end) pack = next;
        if (prefetch) next = pre_p;
    }
}


template <uint32_t kHidden, uint32_t kNumSMs, uint32_t kNumRegisters, bool kStoreBF16,
          bool kStoreFP8, uint32_t SF_BLOCK_M, uint32_t kNumSplits>
DG_DEVICE void run_shifted_norm_worker(SharedStorage& smem, const Workspace<kNumSplits>& ws,
                                        uint64_t* split_bars, const NormArgs& na,
                                        uint32_t lane_idx) {
    constexpr uint32_t kWave = kNumSMs * kNumWarpsPerWG;
    constexpr uint32_t kNumPacks = kHidden / kHiddenPerNormPack;
    const uint32_t num_pairs = (na.num_tokens + kNumTokensPerShiftedTask - 1) / kNumTokensPerShiftedTask;
    // Subdivide underfilled FP8 grids when a pair can span multiple CTAs.
    constexpr uint32_t kNumFP8Parts =
        kNumSplits > kDefaultNumSplits && kNumPacks > 2 * kNumWarpsPerWG ? kNumPacks / 2 : 2;
    const uint32_t parts = kStoreFP8 ? kNumFP8Parts
        : dg_max((kNumPacks + dg_max(1u, num_pairs * kNumPacks / kWave) - 1) / dg_max(1u, num_pairs * kNumPacks / kWave), 2u);
    const uint32_t num_tasks = num_pairs * parts;
    const uint32_t base = blockIdx.x * kNumWarpsPerWG;
    if (base >= num_tasks) return;

    setmaxnreg_inc<kNumRegisters>();
    while (true) {
        uint32_t ticket = 0;
        if (elect_one_sync())
            ticket = atomic_add_shared_u32(&smem.next_norm_task_ticket, 1u);
        ticket = warp_exchange(ticket, 0);
        // Preserve the 4 CTA-local warp residue classes, then +1 grid wave.
        const uint32_t task = base + ticket % kNumWarpsPerWG +
                               ticket / kNumWarpsPerWG * kWave;
        if (task >= num_tasks) return;
        run_shifted_norm_task<kHidden, kStoreBF16, kStoreFP8, SF_BLOCK_M, kNumSplits>(
            ws, split_bars, na, task, parts, lane_idx);
    }
}

// Compile-time int constant for the post-worker dispatch.
template <uint32_t V>
struct IntConstant {
    static constexpr uint32_t value = V;
};


template <uint32_t kHidden, uint32_t kNumSplits, uint32_t kNumSMs,
          bool kIsShifted, bool kStoreBF16, bool kStoreFP8, uint32_t SF_BLOCK_M>
__global__ void __launch_bounds__(kNumThreads, 1)
mega_mhc_impl(const __grid_constant__ dg::TmaMap tensor_map_residual,
              const __grid_constant__ dg::TmaMap tensor_map_x,
              const __grid_constant__ dg::TmaMap tensor_map_fn,
              const __grid_constant__ dg::TmaMap tensor_map_post_mix,
              const __grid_constant__ dg::TmaMap tensor_map_comb_res_mix,
              const __grid_constant__ dg::TmaMap tensor_map_shifted_prev_mix,
              const __grid_constant__ dg::TmaMap tensor_map_new_residual,
              const __grid_constant__ dg::TmaMap tensor_map_y_bf16,
              const __grid_constant__ MixArgs mix_args,
              const __grid_constant__ NormArgs norm_args,
              void* gmem_scratch,
              uint64_t* gmem_split_barriers) {
    constexpr uint32_t UMMA_K = 32 / 4;  // 8 (tf32)
    constexpr uint32_t kNumTokenRowsPerLane = 2;

    static_assert(kHidden > 0 && kHidden % BLOCK_K == 0 && kNumSplits <= kHidden / BLOCK_K,
                  "splits");
    static_assert(SF_BLOCK_M == 0 || kStoreFP8, "shared SF requires FP8");

    // TMEM: A stages (per-route slices) then accumulators.
    constexpr uint32_t kNumTmemColumnsPerAStage = kNumRoutes * UMMA_K;
    constexpr uint32_t kAccumTmemStart = kNumAStages * kNumTmemColumnsPerAStage;
    constexpr uint32_t kNumTmemCols =
        get_num_aligned_tmem_cols<kAccumTmemStart + kNumAccumStages * BLOCK_N>();
    constexpr uint32_t kNumKBlocksPerSplit = kHidden / BLOCK_K / kNumSplits;
    constexpr uint32_t kNumLongSplits = kHidden / BLOCK_K % kNumSplits;
    const auto task_k_begin = [&](uint32_t split) -> uint32_t {
        return split * (kNumKBlocksPerSplit * BLOCK_K) + dg_min(split, kNumLongSplits) * BLOCK_K;
    };
    const auto task_k_blocks = [&](uint32_t split) -> uint32_t {
        return kNumKBlocksPerSplit + (split < kNumLongSplits ? 1u : 0u);
    };

    // CTA roles.
    constexpr uint32_t kFirstPostWG = 1;
    constexpr uint32_t kWorkspaceEpilogueWG = kFirstPostWG + 2;  // 3
    constexpr uint32_t kMixWG = kWorkspaceEpilogueWG + 1;        // 4
    constexpr uint32_t kNormWG = kMixWG + 1;                     // 5
    constexpr uint32_t kNumWGsPerCTA = kNormWG + 1;              // 6
    constexpr uint32_t kNumPostWarps = 2 * kNumWarpsPerWG;
    static_assert(kNumWGsPerCTA == 6, "topology");

    // Register budgets (see the banner in the prelude for setmaxnreg).
    constexpr uint32_t kNumControlRegisters = 32;
    constexpr uint32_t kNumPostRegisters = kIsShifted ? 136 : 112;
    constexpr uint32_t kNumLightweightRegisters = 40;
    constexpr uint32_t kNumShiftedNormRegisters = 96;
    constexpr uint32_t kNumNormalRegisters = 80;

    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    // Post/workspace-epilogue lane mapping: 2 rows per lane, 1 column pair.
    constexpr uint32_t kNumRowsPerWarp = BLOCK_M / kNumWarpsPerWG;         // 16
    constexpr uint32_t kNumRowGroupsPerWarp = kNumRowsPerWarp / 2;         // 8
    constexpr uint32_t kNumLanesPerRowGroup = 32 / kNumRowGroupsPerWarp;   // 4
    const auto get_first_row_idx = [&]() -> uint32_t {
        return warp_idx % kNumWarpsPerWG * kNumRowsPerWarp + lane_idx / kNumLanesPerRowGroup;
    };

    const uint32_t num_m_blocks = Workspace<kNumSplits>::get_num_m_blocks_fn(norm_args.num_tokens);
    const uint32_t num_tasks = num_m_blocks * kNumSplits;
    Workspace<kNumSplits> workspace(gmem_scratch, num_m_blocks);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    SharedStorage& smem = *(SharedStorage*)smem_buffer;

    if (warp_idx == 0 && elect_one_sync()) {
        smem.next_norm_task_ticket = 0;
        prefetch_tma_map(&tensor_map_residual);
        prefetch_tma_map(&tensor_map_x);
        prefetch_tma_map(&tensor_map_fn);
        prefetch_tma_map(&tensor_map_post_mix);
        prefetch_tma_map(&tensor_map_comb_res_mix);
        prefetch_tma_map(&tensor_map_shifted_prev_mix);
        prefetch_tma_map(&tensor_map_new_residual);
        prefetch_tma_map(&tensor_map_y_bf16);

        #pragma unroll
        for (uint32_t i = 0; i < kNumIOStages; ++i) { smem.full_io_barriers[i].init(1); }
        #pragma unroll
        for (uint32_t i = 0; i < kNumIOStages; ++i) { smem.empty_io_barriers[i].init(1); }
        #pragma unroll
        for (uint32_t i = 0; i < kNumIOStages; ++i) { smem.store_ready_io_barriers[i].init(kNumPostWarps); }
        #pragma unroll
        for (uint32_t i = 0; i < kNumCoeffStages; ++i) { smem.full_coeff_barriers[i].init(1); }
        #pragma unroll
        for (uint32_t i = 0; i < kNumCoeffStages; ++i) { smem.empty_coeff_barriers[i].init(kNumPostWarps); }
        fence_barrier_init();
    } else if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumFnStages; ++i) smem.full_fn_barriers[i].init(1);
        #pragma unroll
        for (uint32_t i = 0; i < kNumFnStages; ++i) smem.empty_fn_barriers[i].init(1);
        fence_barrier_init();
    } else if (warp_idx == 2 && elect_one_sync()) {
        // MMA + Post + workspace-epilogue pipelines.
        #pragma unroll
        for (uint32_t i = 0; i < kNumAStages; ++i) smem.full_a_barriers[i].init(128);
        #pragma unroll
        for (uint32_t i = 0; i < kNumAStages; ++i) smem.empty_a_barriers[i].init(1);
        #pragma unroll
        for (uint32_t i = 0; i < kNumAccumStages; ++i) smem.full_accum_barriers[i].init(1);
        #pragma unroll
        for (uint32_t i = 0; i < kNumAccumStages; ++i) smem.empty_accum_barriers[i].init(kNumWarpsPerWG);
        smem.full_stats_barrier.init(kNumPostWarps);
        smem.empty_stats_barrier.init(kNumWarpsPerWG + (kIsShifted ? 1 : 0));
        smem.mix_arrival_barrier.init(kNumWarpsPerWG + (kIsShifted ? 0 : 1));
        fence_barrier_init();
    }

    if (warp_idx == 3)
        tmem_alloc_1sm(kNumTmemCols, &smem.tmem_base);

    // Local init may overlap the prior grid; split barriers may not.
    griddepcontrol_wait();
    if (warp_idx == kFirstPostWG * kNumWarpsPerWG + 1 && elect_one_sync()) {
        for (uint32_t m = blockIdx.x; m < num_m_blocks; m += kNumSMs)
            MixBarrier::init(gmem_split_barriers, m);
    }
    if (kIsShifted && warp_idx == kFirstPostWG * kNumWarpsPerWG && elect_one_sync()) {
        for (uint32_t m = blockIdx.x; m < num_m_blocks; m += kNumSMs)
            NormBarrier::init(gmem_split_barriers, m);
    }

    __syncthreads();
    const uint32_t tmem_base = ld_shared_u32(&smem.tmem_base);

    // ===================== WG0: PRODUCERS =====================
    if (warp_idx < kFirstPostWG * kNumWarpsPerWG) {
        setmaxnreg_dec<kNumControlRegisters>();

        // TMA load workers: same task traversal, only hidden-block copies differ.
        const auto run_tma_load_worker = [&](Barrier* empty_bars, auto load_hidden_block) {
            dg::RingPipelineT<kNumIOStages> pipeline;
            if (elect_one_sync()) {
                for (uint32_t task = blockIdx.x; task < num_tasks; task += kNumSMs) {
                    const uint32_t split = task % kNumSplits;
                    const uint32_t n_kb = task_k_blocks(split);
                    const uint32_t m_idx = task / kNumSplits * BLOCK_M;
                    #pragma unroll 1
                    for (uint32_t kb = 0; kb < n_kb; ++kb) {
                        uint32_t s, p;
                        pipeline.advance(1, &s, &p);
                        empty_bars[s].wait(p ^ 1u);
                        load_hidden_block(s, task_k_begin(split) + kb * BLOCK_K, m_idx);
                    }
                }
            }
            __syncwarp();
        };

        if (warp_idx == 0) {
            // residual + x IO loads (3D residual, 2D x).
            run_tma_load_worker(smem.empty_io_barriers,
                [&](uint32_t s, uint32_t k_idx, uint32_t m_idx) {
                    tma_load_3d(&tensor_map_residual, &smem.full_io_barriers[s],
                                smem.residual[s][0], k_idx, m_idx, 0);
                    tma_load_2d(&tensor_map_x, &smem.full_io_barriers[s],
                                smem.x[s], kEvictNormalHint, k_idx, m_idx);
                    smem.full_io_barriers[s].arrive_and_expect_tx(sizeof(smem.residual[0]) + sizeof(smem.x[0]));
                });
        } else if (warp_idx == 1) {
            // Fn: 3D box (k, 24, routes) per atom — one 3D TMA fills all routes.
            run_tma_load_worker(smem.empty_fn_barriers,
                [&](uint32_t s, uint32_t k_idx, uint32_t) {
                    tma_load_3d(&tensor_map_fn, &smem.full_fn_barriers[s],
                                smem.fn[s][0][0], k_idx, 0u, 0u);
                    smem.full_fn_barriers[s].arrive_and_expect_tx(sizeof(smem.fn[0]));
                });
        } else if (warp_idx == 2) {
            // HC MMA: TF32 TS form; A from TMEM (Post output), B = fn SMEM.
            const uint64_t runtime_desc = make_runtime_instr_desc(
                make_instr_desc_f16(2 /*TF32*/, 2 /*TF32*/, 1 /*F32*/, BLOCK_M, BLOCK_N,
                                     MAJOR_K, MAJOR_K));
            dg::RingPipelineT<kNumFnStages> fn_pipeline;
            dg::RingPipelineT<kNumAStages> a_pipeline;
            dg::RingPipelineT<kNumAccumStages> accum_pipeline;

            for (uint32_t task = blockIdx.x; task < num_tasks; task += kNumSMs) {
                const uint32_t split = task % kNumSplits;
                const uint32_t n_kb = task_k_blocks(split);
                uint32_t as, ap, fs, fp, ac, ap2;
                accum_pipeline.advance(1, &ac, &ap2);
                smem.empty_accum_barriers[ac].wait(ap2 ^ 1u);
                tcgen05_after_thread_sync();

                #pragma unroll 1
                for (uint32_t kb = 0; kb < n_kb; ++kb) {
                    fn_pipeline.advance(1, &fs, &fp);
                    smem.full_fn_barriers[fs].wait(fp);
                    tcgen05_after_thread_sync();

                    #pragma unroll
                    for (uint32_t uk = 0; uk < BLOCK_K / UMMA_K; ++uk) {
                        a_pipeline.advance(1, &as, &ap);
                        smem.full_a_barriers[as].wait(ap);
                        tcgen05_after_thread_sync();
                        // Routes accumulate into one 24-wide output block.
                        #pragma unroll
                        for (uint32_t r = 0; r < kNumRoutes; ++r) {
                            SmemDescriptor b_desc = make_umma_desc<MAJOR_K, BLOCK_N, kFnAtomK,
                                                                   kSwizzleMode, 1, 4>(
                                smem.fn[fs][uk * UMMA_K / kFnAtomK][r], 0, uk * UMMA_K % kFnAtomK);
                            mma_tf32_ts_1sm(tmem_base + as * kNumTmemColumnsPerAStage + r * UMMA_K,
                                            b_desc.desc_,
                                            tmem_base + kAccumTmemStart + ac * BLOCK_N,
                                            (kb > 0 || uk > 0 || r > 0) ? 1u : 0u, runtime_desc);
                        }
                        umma_arrive_1sm(&smem.empty_a_barriers[as]);
                    }
                    umma_arrive_1sm(&smem.empty_fn_barriers[fs]);
                }
                umma_arrive_1sm(&smem.full_accum_barriers[ac]);
            }
        } else {
            // warp 3: output-store + coefficient prefetch + Norm arrival.
            dg::RingPipelineT<kNumIOStages> io_pipeline;
            dg::RingPipelineT<kNumCoeffStages> coeff_pipeline;

            const auto issue_coeff_tma = [&](uint32_t m_idx, uint32_t s) {
                tma_load_2d(&tensor_map_post_mix, &smem.full_coeff_barriers[s],
                            smem.post_coeff[s][0], 0u, kEvictNormalHint, m_idx);
                tma_load_2d_hint(&tensor_map_comb_res_mix, &smem.full_coeff_barriers[s],
                                 smem.comb_coeff[s][0], 0u, m_idx, kEvictNormalHint);
                if (kIsShifted)
                    tma_load_2d(&tensor_map_shifted_prev_mix, &smem.full_coeff_barriers[s],
                                smem.pre_coeff[s][0], 0u, kEvictNormalHint, m_idx);
                smem.full_coeff_barriers[s].arrive_and_expect_tx(
                    sizeof(smem.post_coeff[0]) + sizeof(smem.comb_coeff[0]) +
                    (kIsShifted ? sizeof(smem.pre_coeff[0]) : 0));
            };
            const auto advance_coeff = [&](uint32_t& s, uint32_t& p) {
                coeff_pipeline.advance(1, &s, &p);
                smem.empty_coeff_barriers[s].wait(p ^ 1u);
            };

            // Seed the first task's coefficients (no preceding store window).
            if (blockIdx.x < num_tasks) {
                uint32_t cs, cp;
                advance_coeff(cs, cp);
                if (elect_one_sync())
                    issue_coeff_tma(blockIdx.x / kNumSplits * BLOCK_M, cs);
                __syncwarp();
            }

            for (uint32_t task = blockIdx.x, phase = 0; task < num_tasks;
                 task += kNumSMs, phase ^= 1u) {
                const uint32_t split = task % kNumSplits;
                const uint32_t n_kb = task_k_blocks(split);
                const uint32_t m_block = task / kNumSplits;
                if (kIsShifted)
                    NormBarrier::wait_init<kNumSplits>(gmem_split_barriers, m_block);
                uint32_t next_coeff = 0;

                #pragma unroll 1
                for (uint32_t kb = 0; kb < n_kb; ++kb) {
                    uint32_t is, ip;
                    io_pipeline.advance(1, &is, &ip);
                    smem.store_ready_io_barriers[is].wait(ip);

                    const uint32_t k_idx = task_k_begin(split) + kb * BLOCK_K;
                    const uint32_t m_idx = m_block * BLOCK_M;
                    const bool prefetch_next = kb == n_kb - 1 && task + kNumSMs < num_tasks;
                    if (prefetch_next) {
                        uint32_t cs, cp;
                        advance_coeff(cs, cp);
                        next_coeff = cs;
                    }

                    if (elect_one_sync()) {
                        // new_residual: re-read by fused Normal (EVICT_LAST);
                        // Shifted consumes it next layer (EVICT_NORMAL).
                        tma_store_3d_hint(&tensor_map_new_residual, smem.residual[is][0],
                                          k_idx, m_idx, 0u,
                                          kIsShifted ? kEvictNormalHint : kEvictLastHint);
                        if (kIsShifted)
                            tma_store_2d_hint(&tensor_map_y_bf16, smem.x[is], k_idx, m_idx,
                                              kEvictLastHint);
                        tma_store_arrive();
                        if (prefetch_next)
                            issue_coeff_tma((task + kNumSMs) / kNumSplits * BLOCK_M, next_coeff);
                        tma_store_wait<0>();
                        smem.empty_io_barriers[is].arrive();
                    }
                }
                __syncwarp();

                if (kIsShifted) {
                    // Arrive at Norm once both Post workers committed statistics.
                    smem.full_stats_barrier.wait(phase);
                    float* x1_partial = workspace.get_x1_sqr_sum_partial_ptr(task);
                    float2 row_sqr = make_float2(0.f, 0.f);
                    #pragma unroll
                    for (uint32_t w = 0; w < 2; ++w) {
                        row_sqr.x += smem.x1_sqr_sums[w][lane_idx];
                        row_sqr.y += smem.x1_sqr_sums[w][lane_idx + 32];
                    }
                    x1_partial[lane_idx] = row_sqr.x;
                    x1_partial[lane_idx + 32] = row_sqr.y;
                    __syncwarp();
                    if (elect_one_sync()) {
                        smem.empty_stats_barrier.arrive();
                        NormBarrier::arrive<kNumSplits>(gmem_split_barriers, m_block);
                    }
                } else if (elect_one_sync()) {
                    smem.mix_arrival_barrier.arrive();
                }
            }
        }
    }
    // ===================== WG1-2: POST WORKERS =====================
    else if (warp_idx < kWorkspaceEpilogueWG * kNumWarpsPerWG) {
        const auto get_bf16_off2 = [&](uint32_t row, uint32_t col) {
            return get_bf16x2_offset(row, col);
        };
        static_assert(UMMA_K % kNumLanesPerRowGroup == 0, "post geometry");

        const auto run_post_worker = [&](auto post_worker) {
            constexpr uint32_t kPW = decltype(post_worker)::value;
            setmaxnreg_inc<kNumPostRegisters>();
            const uint32_t col_pair = lane_idx & (kNumLanesPerRowGroup - 1);
            const uint32_t first_row = get_first_row_idx();
            const uint32_t rows[2] = {first_row, first_row + kNumRowGroupsPerWarp};

            dg::RingPipelineT<kNumIOStages> io_pipeline;
            dg::RingPipelineT<kNumAStages> a_pipeline;
            dg::RingPipelineT<kNumCoeffStages> coeff_pipeline;
            a_pipeline.advance(kPW);  // offset the two workers' A stages

            for (uint32_t task = blockIdx.x, phase = 0; task < num_tasks;
                 task += kNumSMs, phase ^= 1u) {
                const uint32_t split = task % kNumSplits;
                const uint32_t n_kb = task_k_blocks(split);
                uint32_t cs, cp;
                coeff_pipeline.advance(1, &cs, &cp);
                float2 hc_norm_sqr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
                float2 x1_sqr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};

                // Coefficients live in registers for the whole task.
                smem.full_coeff_barriers[cs].wait(cp);
                float2 row_pre[kNumRoutes];
                float2 row_post[kNumRoutes];
                #pragma unroll
                for (uint32_t r = 0; r < kNumRoutes; ++r) {
                    if (kIsShifted)
                        row_pre[r] = ld_shared_f32x2((const float2*)&smem.pre_coeff[cs][rows[0]][r]);
                    row_post[r] = ld_shared_f32x2((const float2*)&smem.post_coeff[cs][rows[0]][r]);
                    // (y = row 1's value; extracted below)
                    float pre_y = 0.f, post_y = 0.f;
                    if (kIsShifted) pre_y = smem.pre_coeff[cs][rows[1]][r];
                    post_y = smem.post_coeff[cs][rows[1]][r];
                    if (kIsShifted) row_pre[r].y = pre_y;
                    row_post[r].y = post_y;
                }
                float2 row_comb[kNumRoutes * kNumRoutes];
                #pragma unroll
                for (uint32_t c = 0; c < kNumRoutes * kNumRoutes; ++c) {
                    row_comb[c] = make_float2(
                        load_comb_coeff(smem.comb_coeff[cs][rows[0]], rows[0], c),
                        load_comb_coeff(smem.comb_coeff[cs][rows[1]], rows[1], c));
                }

                __syncwarp();
                if (elect_one_sync()) {
                    fence_view_async_shared();
                    smem.empty_coeff_barriers[cs].arrive();
                }

                // Round-robin UMMA atoms; statistics retained until split end.
                #pragma unroll 1
                for (uint32_t kb = 0; kb < n_kb; ++kb) {
                    uint32_t is, ip;
                    io_pipeline.advance(1, &is, &ip);
                    smem.full_io_barriers[is].wait(ip);

                    #pragma unroll
                    for (uint32_t uk = kPW; uk < BLOCK_K / UMMA_K; uk += 2) {
                        const uint32_t hp = uk * (UMMA_K / 2) + col_pair * (UMMA_K / kNumLanesPerRowGroup / 2);
                        float2 xv[2];
                        float2 res[2][kNumRoutes];
                        #pragma unroll
                        for (uint32_t ri = 0; ri < 2; ++ri) {
                            const uint32_t off = get_bf16_off2(rows[ri], hp);
                            bf16x2_to_f32x2((uint32_t)smem.x[is][off] | ((uint32_t)smem.x[is][off + 1] << 16),
                                            xv[ri].x, xv[ri].y);
                            #pragma unroll
                            for (uint32_t rin = 0; rin < kNumRoutes; ++rin) {
                                const uint32_t roff = get_bf16_off2(rows[ri], hp);
                                const uint32_t packed = (uint32_t)smem.residual[is][rin][roff] |
                                                        ((uint32_t)smem.residual[is][rin][roff + 1] << 16);
                                bf16x2_to_f32x2(packed, res[ri][rin].x, res[ri][rin].y);
                            }
                        }

                        // Wait MMA release of this worker's next A stage.
                        uint32_t as, ap;
                        a_pipeline.advance(2, &as, &ap);
                        smem.empty_a_barriers[as].wait(ap ^ 1u);
                        tcgen05_after_thread_sync();

                        float2 x1[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
                        float2 post_values[kNumRoutes][2];
                        #pragma unroll
                        for (uint32_t ri = 0; ri < 2; ++ri) {
                            #pragma unroll
                            for (uint32_t rout = 0; rout < kNumRoutes; ++rout) {
                                const float2 post = row_post[rout];
                                const float pc = ri == 0 ? post.x : post.y;
                                float2& pv = post_values[rout][ri];
                                pv = make_float2(xv[ri].x * pc, xv[ri].y * pc);
                                #pragma unroll
                                for (uint32_t rin = 0; rin < kNumRoutes; ++rin) {
                                    const float2 comb = row_comb[rin * kNumRoutes + rout];
                                    const float cc = ri == 0 ? comb.x : comb.y;
                                    pv.x += res[ri][rin].x * cc;
                                    pv.y += res[ri][rin].y * cc;
                                }
                                // The BF16 round is the shared numerical boundary.
                                const uint32_t pv_bf = f32x2_to_bf16x2(pv.x, pv.y);
                                bf16x2_to_f32x2(pv_bf, pv.x, pv.y);
                                const uint32_t off = get_bf16_off2(rows[ri], hp);
                                smem.residual[is][rout][off] = (uint16_t)(pv_bf & 0xffff);
                                smem.residual[is][rout][off + 1] = (uint16_t)(pv_bf >> 16);
                                hc_norm_sqr[ri].x += pv.x * pv.x;
                                hc_norm_sqr[ri].y += pv.y * pv.y;
                                if (kIsShifted) {
                                    const float2 pre = row_pre[rout];
                                    const float rc = ri == 0 ? pre.x : pre.y;
                                    x1[ri].x += pv.x * rc;
                                    x1[ri].y += pv.y * rc;
                                }
                            }
                        }

                        // tcgen05.st the A tile (4 routes x 8 rows of this pair).
                        const uint32_t* pv_words = (const uint32_t*)post_values;
                        const uint32_t tmem_addr = tmem_base + as * kNumTmemColumnsPerAStage;
                        tmem_store_16dp256b_x4(tmem_addr,
                                               pv_words[0], pv_words[1], pv_words[2], pv_words[3],
                                               pv_words[4], pv_words[5], pv_words[6], pv_words[7],
                                               pv_words[8], pv_words[9], pv_words[10], pv_words[11],
                                               pv_words[12], pv_words[13], pv_words[14], pv_words[15]);

                        if (kIsShifted) {
                            #pragma unroll
                            for (uint32_t ri = 0; ri < 2; ++ri) {
                                const uint32_t x1_bf = f32x2_to_bf16x2(x1[ri].x, x1[ri].y);
                                // Squares accumulate from the BF16-rounded X1.
                                accumulate_square_f32_bf16(x1_sqr[ri].x, x1_bf & 0xffff);
                                accumulate_square_f32_bf16(x1_sqr[ri].y, x1_bf >> 16);
                                const uint32_t off = get_bf16_off2(rows[ri], hp);
                                smem.x[is][off] = (uint16_t)(x1_bf & 0xffff);
                                smem.x[is][off + 1] = (uint16_t)(x1_bf >> 16);
                            }
                        }

                        // All 4 warps stored before A becomes visible to the MMA.
                        fence_view_async_tmem_store();
                        tcgen05_before_thread_sync();
                        smem.full_a_barriers[as].arrive();
                    }

                    __syncwarp();
                    if (elect_one_sync()) {
                        tma_store_fence();
                        smem.store_ready_io_barriers[is].arrive();
                    }
                }

                // Commit statistics once register-held values are safe to reuse.
                smem.empty_stats_barrier.wait(phase ^ 1u);
                const float hn0 = warp_reduce_sum_f32(hc_norm_sqr[0].x + hc_norm_sqr[0].y, kNumLanesPerRowGroup);
                const float hn1 = warp_reduce_sum_f32(hc_norm_sqr[1].x + hc_norm_sqr[1].y, kNumLanesPerRowGroup);
                smem.hc_norm_sqr_sums[kPW][rows[0]] = hn0;
                smem.hc_norm_sqr_sums[kPW][rows[1]] = hn1;
                if (kIsShifted) {
                    const float x10 = warp_reduce_sum_f32(x1_sqr[0].x + x1_sqr[0].y, kNumLanesPerRowGroup);
                    const float x11 = warp_reduce_sum_f32(x1_sqr[1].x + x1_sqr[1].y, kNumLanesPerRowGroup);
                    smem.x1_sqr_sums[kPW][rows[0]] = x10;
                    smem.x1_sqr_sums[kPW][rows[1]] = x11;
                }
                __syncwarp();
                if (elect_one_sync())
                    smem.full_stats_barrier.arrive();
                __syncwarp();
            }
            setmaxnreg_dec<kNumLightweightRegisters>();
        };
        warp_idx < (kFirstPostWG + 1) * kNumWarpsPerWG
            ? run_post_worker(IntConstant<0>()) : run_post_worker(IntConstant<1>());
    }
    // ===================== WG3: WORKSPACE EPILOGUE =====================
    else if (warp_idx < kMixWG * kNumWarpsPerWG) {
        setmaxnreg_dec<kNumLightweightRegisters>();
        const uint32_t warp_in_wg = warp_idx % kNumWarpsPerWG;
        const uint32_t col_pair = lane_idx & (kNumLanesPerRowGroup - 1);
        const uint32_t first_row = get_first_row_idx();
        const uint32_t rows[2] = {first_row, first_row + kNumRowGroupsPerWarp};

        dg::RingPipelineT<kNumAccumStages> accum_pipeline;
        for (uint32_t task = blockIdx.x, phase = 0; task < num_tasks;
             task += kNumSMs, phase ^= 1u) {
            const uint32_t m_block = task / kNumSplits;
            if (warp_in_wg == 0)
                MixBarrier::wait_init<kNumSplits>(gmem_split_barriers, m_block);

            const uint32_t tokens[2] = {m_block * BLOCK_M + rows[0], m_block * BLOCK_M + rows[1]};
            float* gemm_partial = workspace.get_gemm_partial_ptr(task);

            uint32_t ac, ap;
            accum_pipeline.advance(1, &ac, &ap);
            smem.full_accum_barriers[ac].wait(ap);
            tcgen05_after_thread_sync();

            // 3 groups x 8 outputs per row; the 2x and 1x loads fetch all 24.
            constexpr uint32_t kPerLoad = 32 / 4;               // 8
            constexpr uint32_t kPerLane = kPerLoad / kNumLanesPerRowGroup;  // 2
            uint32_t values[3 * kNumLanesPerRowGroup];
            const uint32_t tmem_addr = tmem_base + kAccumTmemStart + ac * BLOCK_N;
            tmem_load_16dp256b_x2(tmem_addr,
                                  values[0], values[1], values[2], values[3],
                                  values[4], values[5], values[6], values[7]);
            tmem_load_16dp256b_x1(tmem_addr + 2 * kPerLoad,
                                  values[8], values[9], values[10], values[11]);
            fence_view_async_tmem_load();

            #pragma unroll
            for (uint32_t o = 0; o < BLOCK_N; o += kPerLoad) {
                const uint32_t v = o / kPerLane;
                float2 rv[2] = {
                    make_float2(__uint_as_float(values[v]), __uint_as_float(values[v + 1])),
                    make_float2(__uint_as_float(values[v + 2]), __uint_as_float(values[v + 3]))};
                #pragma unroll
                for (uint32_t ri = 0; ri < 2; ++ri)
                    if (tokens[ri] >= norm_args.num_tokens)
                        rv[ri] = make_float2(0.f, 0.f);
                ((float2*)(gemm_partial + rows[0] * BLOCK_N + o + col_pair * kPerLane))[0] = rv[0];
                ((float2*)(gemm_partial + rows[1] * BLOCK_N + o + col_pair * kPerLane))[0] = rv[1];
            }

            tcgen05_before_thread_sync();
            if (elect_one_sync())
                smem.empty_accum_barriers[ac].arrive();

            // Statistics for this split; gate the Mix arrival.
            smem.full_stats_barrier.wait(phase);
            if (lane_idx < kNumRowsPerWarp) {
                const uint32_t row = warp_in_wg * kNumRowsPerWarp + lane_idx;
                float* p = workspace.get_hc_norm_sqr_sum_partial_ptr(task);
                float sum = 0.0f;
                #pragma unroll
                for (uint32_t w = 0; w < 2; ++w)
                    sum += smem.hc_norm_sqr_sums[w][row];
                p[row] = sum;
            }
            __syncwarp();
            if (elect_one_sync()) {
                smem.empty_stats_barrier.arrive();
                smem.mix_arrival_barrier.arrive();
            }
            if (warp_in_wg == 0) {
                smem.mix_arrival_barrier.wait(phase);
                if (elect_one_sync())
                    MixBarrier::arrive<kNumSplits>(gmem_split_barriers, m_block);
            }
        }
    }
    // ===================== WG4: MIX (shifted only) =====================
    else if (warp_idx < kNormWG * kNumWarpsPerWG) {
        if (kIsShifted) {
            setmaxnreg_dec<kNumLightweightRegisters>();
            const uint32_t mix_warp = warp_idx - kMixWG * kNumWarpsPerWG;
            for (uint32_t t = blockIdx.x * kNumWarpsPerWG + mix_warp;
                 t < norm_args.num_tokens; t += kNumSMs * kNumWarpsPerWG)
                run_mix_task<kHidden, true>(workspace, gmem_split_barriers, mix_args, t, lane_idx);
        }
    }

    // ===================== NORMAL vs SHIFTED NORM =====================
    if (!kIsShifted) {
        // Normal repurposes WG4-5 immediately and each producer on retirement.
        if (warp_idx >= kMixWG * kNumWarpsPerWG)
            setmaxnreg_dec<kNumLightweightRegisters>();
        const uint32_t wg = warp_idx / kNumWarpsPerWG;
        named_barrier_sync(128, 1 + wg);  // per-WG dedicated barrier ids 1..6
        setmaxnreg_inc<kNumNormalRegisters>();
        run_normal_worker<kHidden, kNumSMs, kStoreBF16, kStoreFP8, SF_BLOCK_M, kNumSplits>(
            smem, workspace, wg, warp_idx % kNumWarpsPerWG, lane_idx,
            gmem_split_barriers, mix_args, norm_args);
        setmaxnreg_dec<kNumControlRegisters>();
    } else {
        // WG5 starts immediately; retired Post workers join the Norm queue.
        const uint32_t wg = warp_idx / kNumWarpsPerWG;
        if ((wg >= kFirstPostWG && wg < kWorkspaceEpilogueWG) || wg == kNormWG) {
            run_shifted_norm_worker<kHidden, kNumSMs, kNumShiftedNormRegisters,
                                     kStoreBF16, kStoreFP8, SF_BLOCK_M, kNumSplits>(
                smem, workspace, gmem_split_barriers, norm_args, lane_idx);
        }
    }

    __syncthreads();
    if (warp_idx == 3)
        tmem_dealloc_1sm(0, kNumTmemCols);
}


} // namespace mega_mhc
} // namespace dg
