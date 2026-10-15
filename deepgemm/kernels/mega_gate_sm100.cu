// DeepGEMM-RS: fp8 MegaGate — the fused router/gate megakernel, FP8 rewrite.
//
// Upstream `sm100_bf16_mega_gate_impl` runs the router GEMM in BF16. Per the
// project directive ("all parts written in fp4 and fp8"), this port upgrades
// the GEMM data plane to block-scaled FP8 (e4m3 x e4m3, UE8M0 scale factors,
// kind::mxf8f6f4): weights are the A operand (experts = UMMA_M rows),
// activations are the B operand (tokens = UMMA_N columns). The score path
// (TMEM FP32 accumulator -> gmem FP32 scratch) is bit-identical to upstream;
// only the GEMM input format changes, plus the new SFA/SFB TMA + UTCCP
// pipeline feeding the block-scaled MMA.
//
// Everything else is a faithful port of the upstream gate machinery:
//   * split-K over K (score partials in FP32 scratch, reduced in the topk loop)
//   * bias / image-bias / mask / fix-routing / force-random tokens
//   * physical-map duplicate spreading ((rank + token*23333) % num_duplicates)
//   * sqrt-softplus activation, applied at store (single split) or in the
//     topk loop (split-K, after summing partials)
//   * warp-level top-k: per-lane 4-element sort network + 3-bit cursor rounds
//     across expert waves, redux.sync.max.f32 for the warp best
//   * top-k weight normalization (unbiased score / sum * routed_scaling)
//   * ScoreBarrier: per-token-block gmem grid-tag barrier so every logical CTA
//     of a block ranks only after all splits' scores are visible
//   * counter-based Philox4x32-10 RNG for force-random tokens (curand-free)
//
// ===========================================================================
// WARP MAP (kNumNonEpilogueThreads=128 | kNumGateThreads = W*128)
// ===========================================================================
//   warp 0   TMA producer: x (EVICT_FIRST stream) + weight + SFA + SFB tiles
//   warp 1   MMA issuer (leader CTA only): UTCCP SFA/SFB + mxf8f6f4
//            (1SM when kNumMmaCtas==1, 2x1SM on a CTA pair otherwise)
//   warps 4..   gate warps:
//            - optional metadata cache fill (bias/image bias/logical count)
//            - per token block: re-arm the ScoreBarrier (CTA 0), wait the
//              accumulator stage, transpose scores TMEM->gmem (32dp32b),
//              release TMEM, arrive at the ScoreBarrier, wait all splits,
//              then rank the block's tokens (top-k + outputs)
//
// Task geometry: a token block is computed by kNumLogicalCtas =
// num_expert_groups * num_mma_ctas * num_split_k co-resident CTAs; worker
// groups stride over token blocks persistently (grid = num_launch_sms).
//
// TMEM budget (fp8): accum 2 x UMMA_N (<= 240) + SFA (<=8) + SFB (<=8)
// => <= 496 columns; the host caps block_tokens at 240 (16-aligned).
// ===========================================================================

namespace dg {
namespace mega_gate {

constexpr uint32_t BLOCK_K = 64;
constexpr uint32_t kSwizzleMode = 128;
constexpr uint32_t kNumNonEpilogueThreads = 128;
constexpr uint32_t kNumEpilogueStages = 2;
constexpr uint32_t kNumMaxBlockTokens = 240;   // fp8 TMEM cap (upstream bf16: 256)
constexpr uint32_t kNumMaxBlockTokensCap = 240;  // smem/tmem capacity alias
constexpr uint32_t kExpertAlignment = 128;
constexpr uint32_t kSharedMemoryAlignment = 1024;

constexpr uint32_t kNumMinBlockTokens = 16;
constexpr uint32_t kNumMaxTokens = 1u << 20;
constexpr uint32_t kNumMaxTokenBlocks = (kNumMaxTokens + kNumMinBlockTokens - 1) / kNumMinBlockTokens;

constexpr uint32_t kScoreBarrierLineBytes = 128;
constexpr uint32_t kNumMaxLogicalCtas = 64;
constexpr uint32_t kNumMaxMetadataCacheBytes = 4 * 1024;

// Top-k geometry: each lane holds 4 consecutive experts of every 128-wave.
constexpr uint32_t kNumExpertsPerLaneVector = 4;
constexpr uint32_t kNumExpertsPerWave = 32 * kNumExpertsPerLaneVector;
static_assert(kNumExpertsPerWave == kExpertAlignment, "waves");

constexpr uint32_t kNumMaxStages = 32;

// Host/device ABI payload (mirrors upstream RoutingArgs).
struct RoutingArgs {
    const int* to_physical_map;
    const int* logical_count;
    int64_t* topk_idx;
    int64_t* unmapped_topk_idx;
    float* topk_weights;
    int64_t unmapped_topk_idx_stride;
    uint32_t num_routed_experts;
    uint32_t num_shared_experts;
    uint32_t num_duplicate_experts;
    uint32_t rank_idx;
    float routed_scaling_factor;
};

constexpr bool caches_logical_count(uint32_t num_routed_experts, bool has_bias,
                                     bool has_image_token_mask, bool has_physical_map) {
    const uint32_t bias_bytes =
        num_routed_experts * ((has_bias ? 1u : 0u) + (has_image_token_mask ? 1u : 0u)) * 4u;
    return has_physical_map &&
           bias_bytes + num_routed_experts * 4u <= kNumMaxMetadataCacheBytes;
}

struct SharedStorage {
    Barrier full_barriers[kNumMaxStages];
    Barrier empty_barriers[kNumMaxStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
    float metadata_cache[kNumMaxMetadataCacheBytes / 4];
};
constexpr uint32_t kStageDataOffset = kSharedMemoryAlignment;  // fixed header

DG_DEVICE float* x_stage_ptr(uint8_t* smem, uint32_t stage, uint32_t x_bytes) {
    return (float*)(smem + kStageDataOffset + stage * x_bytes);
}
DG_DEVICE float* weight_stage_ptr(uint8_t* smem, uint32_t stage, uint32_t num_stages,
                                    uint32_t x_bytes, uint32_t w_bytes) {
    return (float*)(smem + kStageDataOffset + num_stages * x_bytes + stage * w_bytes);
}
DG_DEVICE float* sfa_stage_ptr(uint8_t* smem, uint32_t num_stages,
                                 uint32_t x_bytes, uint32_t w_bytes) {
    return (float*)(smem + kStageDataOffset + num_stages * (x_bytes + w_bytes));
}
DG_DEVICE float* sfb_stage_ptr(uint8_t* smem, uint32_t num_stages,
                                 uint32_t x_bytes, uint32_t w_bytes, uint32_t sfa_bytes) {
    return (float*)(smem + kStageDataOffset + num_stages * (x_bytes + w_bytes) + sfa_bytes);
}

// View over the FP32 score scratch: [tokens, num_splits, E_aligned].
template <uint32_t kNumSplits = 0, uint32_t kNumExperts = 0>
struct Workspace {
    static constexpr uint32_t kTokenStride = kNumSplits * kNumExperts;
    void* gmem_scratch;
    void* gmem_score_barriers;

    Workspace(void* scratch, void* bars)
        : gmem_scratch(scratch), gmem_score_barriers(bars) {}

    static uint64_t get_num_scratch_bytes(uint32_t num_token_blocks, uint32_t num_splits,
                                            uint32_t block_tokens, uint32_t num_experts) {
        return (uint64_t)num_token_blocks * num_splits * block_tokens * num_experts * 4;
    }
    DG_DEVICE float* get_score_ptr(uint32_t token_idx, uint32_t split_idx = 0) const {
        return (float*)gmem_scratch + (uint64_t)token_idx * kTokenStride
               + (uint64_t)split_idx * kNumExperts;
    }
    DG_DEVICE uint64_t* get_score_barrier_ptr(uint32_t token_block_idx) const {
        return (uint64_t*)((uint8_t*)gmem_score_barriers +
                           (uint64_t)token_block_idx * kScoreBarrierLineBytes);
    }
};

// Grid-tag score barrier: state = (grid_idx + 1) * kNumMaxLogicalCtas +
// arrived; wait_init's unsigned distance rejects stale grid tags.
template <uint32_t kNumLogicalCtas>
struct ScoreBarrier {
    static_assert(kNumLogicalCtas < kNumMaxLogicalCtas, "ctas");

    static DG_DEVICE uint64_t encode(uint64_t grid_idx, uint32_t arrived = 0) {
        return (grid_idx + 1) * kNumMaxLogicalCtas + arrived;
    }
    static DG_DEVICE void init(uint64_t* p) { st_rel_gpu_u64(p, encode(get_grid_idx())); }
    static DG_DEVICE void wait_init(uint64_t* p) {
        const uint64_t base = encode(get_grid_idx());
        // Unsigned distance: rejects stale tags, accepts any current count.
        while (ld_acq_gpu_u64(p) - base >= kNumLogicalCtas) {}
    }
    static DG_DEVICE void arrive(uint64_t* p) { red_async_inc_rel_u64(p); }
    static DG_DEVICE void wait(uint64_t* p) {
        const uint64_t all_arrived = encode(get_grid_idx(), kNumLogicalCtas);
        while (ld_acq_gpu_u64(p) != all_arrived) {}
    }
};

// ---------------------------------------------------------------------------
// Numerics / ranking helpers (epilogue/sm100_mega_gate.cuh port)
// ---------------------------------------------------------------------------
DG_DEVICE float sqrt_softplus(float v) {
    const float precise = log1pf(expf(v));
    return sqrtf(v > 20.0f ? v : precise);
}
DG_DEVICE void sqrt_softplus_x4(float (&v)[4]) {
    float e[4], sp[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) e[i] = expf(v[i]);
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) sp[i] = log1pf(e[i]);
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) v[i] = sqrtf(v[i] > 20.0f ? v[i] : sp[i]);
}
DG_DEVICE void warp_reduce_best(float score, int& expert_idx) {
    const float best = redux_max_f32(score);
    const uint32_t tied = score == best && expert_idx >= 0 ? (uint32_t)expert_idx : 0xffffffffu;
    expert_idx = (int)redux_min_u32(tied);  // ties resolve to the smallest index
}
DG_DEVICE void compare_swap_best_first(float& ls, int& li, float& rs, int& ri) {
    if (rs > ls || (rs == ls && ri < li)) {
        const float s = ls; const int i = li;
        ls = rs; li = ri;
        rs = s; ri = i;
    }
}
DG_DEVICE uint32_t sort4_best_first(float& s0, float& s1, float& s2, float& s3) {
    int i0 = 0, i1 = 1, i2 = 2, i3 = 3;
    compare_swap_best_first(s0, i0, s1, i1);
    compare_swap_best_first(s2, i2, s3, i3);
    compare_swap_best_first(s0, i0, s2, i2);
    compare_swap_best_first(s1, i1, s3, i3);
    compare_swap_best_first(s1, i1, s2, i2);
    return (uint32_t)(i0 | i1 << 2 | i2 << 4 | i3 << 6);
}
template <uint32_t kNumExpertWaves, uint32_t kNumTopk>
DG_DEVICE void select_warp_topk(float (&scores)[kNumExpertWaves * kNumExpertsPerLaneVector],
                                 int& selected_expert_idx) {
    static_assert(kNumExpertsPerLaneVector == 4, "4-element sort network");
    uint32_t perms[kNumExpertWaves];
    #pragma unroll
    for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
        const uint32_t base = w * kNumExpertsPerLaneVector;
        perms[w] = sort4_best_first(scores[base + 0], scores[base + 1],
                                     scores[base + 2], scores[base + 3]);
    }
    // Cursor rounds: each lane offers each wave's best unconsumed candidate;
    // the winning lane alone advances its cursor for that wave.
    const uint32_t lane_idx = get_lane_idx();
    uint32_t packed_cursors = 0;
    #pragma unroll
    for (uint32_t out = 0; out < kNumTopk; ++out) {
        float best_score = -3.402823466e38f /* -FLT_MAX */;
        int best_idx = -1;
        uint32_t best_wave = 0;
        #pragma unroll
        for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
            const uint32_t cursor = packed_cursors >> (w * 3) & 7;
            const uint32_t base = w * kNumExpertsPerLaneVector;
            const float cand = cursor == 4 ? -3.402823466e38f :
                cursor & 2 ? (cursor & 1 ? scores[base + 3] : scores[base + 2])
                           : (cursor & 1 ? scores[base + 1] : scores[base + 0]);
            const uint32_t off = perms[w] >> (cursor * 2) & 3;
            const int cand_idx = (int)(w * kNumExpertsPerWave +
                                       lane_idx * kNumExpertsPerLaneVector + off);
            if (cand > best_score) {
                best_score = cand;
                best_idx = cand_idx;
                best_wave = w;
            }
        }
        const int local_best = best_idx;
        warp_reduce_best(best_score, best_idx);
        if (local_best == best_idx)
            packed_cursors += 1u << (best_wave * 3);
        if (lane_idx == out)
            selected_expert_idx = best_idx;
    }
}
template <uint32_t kNumExpertWaves>
DG_DEVICE float select_warp_expert_value(
        const float (&values)[kNumExpertWaves * kNumExpertsPerLaneVector], int expert_idx) {
    const int value_idx = expert_idx >= 0
        ? (int)(expert_idx / (int)kNumExpertsPerWave * (int)kNumExpertsPerLaneVector +
                expert_idx % (int)kNumExpertsPerLaneVector)
        : -1;
    const uint32_t src_lane = expert_idx >= 0
        ? (uint32_t)(expert_idx % (int)kNumExpertsPerWave / (int)kNumExpertsPerLaneVector) : 0u;
    float selected = 0.0f;
    #pragma unroll
    for (uint32_t v = 0; v < kNumExpertWaves * kNumExpertsPerLaneVector; ++v) {
        const float ex = warp_exchange(values[v], src_lane);
        if (value_idx == (int)v) selected = ex;
    }
    return selected;
}
template <uint32_t kNumTopkV>
DG_DEVICE void store_token_outputs(const RoutingArgs& args, uint32_t token_idx,
                                    uint32_t lane_idx, int64_t physical_expert_idx,
                                    float weight, int64_t logical_expert_idx) {
    const uint32_t num_physical_topk = kNumTopkV + args.num_shared_experts;
    if (lane_idx < num_physical_topk) {
        const uint64_t out = (uint64_t)token_idx * num_physical_topk + lane_idx;
        args.topk_idx[out] = physical_expert_idx;
        args.topk_weights[out] = weight;
    }
    if (kNumTopkV > 0) {}  // (unmapped handled by the caller flag below)
}
template <uint32_t kNumTopkV>
DG_DEVICE void store_token_outputs_unmapped(const RoutingArgs& args, uint32_t token_idx,
                                             uint32_t lane_idx, int64_t logical_expert_idx) {
    if (lane_idx < kNumTopkV)
        args.unmapped_topk_idx[(uint64_t)token_idx * args.unmapped_topk_idx_stride + lane_idx] =
            logical_expert_idx;
}
template <uint32_t kNumTopkV>
DG_DEVICE void store_topk_token(const RoutingArgs& args, const int* routed_logical_count,
                                  uint32_t token_idx, uint32_t lane_idx,
                                  int selected_expert_idx, float unbiased_score,
                                  bool has_physical_map, bool has_unmapped) {
    const uint32_t num_physical_topk = kNumTopkV + args.num_shared_experts;
    // Shared experts follow the routed ones in the slot order.
    if (lane_idx >= kNumTopkV && lane_idx < num_physical_topk)
        selected_expert_idx = (int)(lane_idx + args.num_routed_experts - kNumTopkV);

    int64_t physical_idx = selected_expert_idx;
    if (has_physical_map && lane_idx < num_physical_topk) {
        const uint32_t logical = (uint32_t)selected_expert_idx;
        const int* counts = lane_idx < kNumTopkV ? routed_logical_count : args.logical_count;
        const uint32_t dups = (uint32_t)counts[logical];
        // Spread a logical expert's tokens over its duplicates, per rank.
        const uint32_t dup = (args.rank_idx + token_idx * 23333u) % dups;
        physical_idx = args.to_physical_map[(uint64_t)logical * args.num_duplicate_experts + dup];
    }

    // Weight normalization: unbiased / sum * scaling (>= 8-lane reduce).
    const uint32_t width = kNumTopkV <= 8 ? 8u : (kNumTopkV <= 16 ? 16u : 32u);
    const float topk_sum = warp_reduce_sum_f32(unbiased_score, width) + 1e-20f;
    const float weight = lane_idx < kNumTopkV
        ? unbiased_score / topk_sum * args.routed_scaling_factor : 1.0f;
    store_token_outputs<kNumTopkV>(args, token_idx, lane_idx, physical_idx, weight,
                                    selected_expert_idx);
    if (has_unmapped)
        store_token_outputs_unmapped<kNumTopkV>(args, token_idx, lane_idx,
                                                 (int64_t)selected_expert_idx);
}
template <uint32_t kNumTopkV>
DG_DEVICE void store_random_topk_token(const RoutingArgs& args, uint32_t token_idx,
                                         uint32_t lane_idx, bool has_physical_map,
                                         bool has_unmapped) {
    const uint32_t num_physical_topk = kNumTopkV + args.num_shared_experts;
    uint32_t num_physical_experts = args.num_routed_experts + args.num_shared_experts;
    if (has_physical_map) {
        uint32_t local = 0;
        for (uint32_t e = lane_idx; e < num_physical_experts; e += 32)
            local += (uint32_t)args.logical_count[e];
        num_physical_experts = redux_add_u32(local);
    }

    // Sample distinct experts: lane i draws from n - i, ranks below every
    // earlier draw shift up by one.
    PhiloxState rng;
    philox_init(&rng, args.rank_idx, (uint64_t)token_idx * 32 + lane_idx);
    const float random_weight = philox_uniform(&rng);
    constexpr uint32_t kInvalid = 0xffffffffu;
    if (num_physical_experts < num_physical_topk) asm volatile("trap;");
    uint32_t candidate = lane_idx < num_physical_topk
        ? philox_next(&rng) % (num_physical_experts - lane_idx) : kInvalid;
    uint32_t selected = kInvalid;
    for (uint32_t out = 0; out < num_physical_topk; ++out) {
        const uint32_t min_cand = redux_min_u32(candidate);
        const uint32_t min_lane = redux_min_u32(candidate == min_cand ? lane_idx : kInvalid);
        if (candidate != kInvalid) {
            if (min_lane == lane_idx) {
                selected = candidate;
                candidate = kInvalid;
            } else if (candidate >= min_cand && min_lane < lane_idx) {
                ++candidate;
            }
        }
    }
    store_token_outputs<kNumTopkV>(args, token_idx, lane_idx, (int64_t)selected,
                                    random_weight, (int64_t)-1);
    if (has_unmapped)
        store_token_outputs_unmapped<kNumTopkV>(args, token_idx, lane_idx, (int64_t)-1);
}

// ---------------------------------------------------------------------------
// Score transpose: TMEM -> gmem scratch (fp32), leader partitioned token
// paths when UMMA_M == 128 on a CTA pair (the 64 experts of a CTA spread
// over the two lane halves; warps 0/1 and 2/3 see the same experts).
// ---------------------------------------------------------------------------
template <bool kScoreOnStore, uint32_t kScoreTokenStride,
          uint32_t UMMA_M, uint32_t kNumMmaCtas, uint32_t kNumGateWarps>
DG_DEVICE bool store_tmem_scores_to_global(float* scores, uint32_t tmem_base,
                                             uint32_t effective_umma_n,
                                             uint32_t expert_base_idx,
                                             uint32_t mma_cta_rank,
                                             uint32_t gate_warp_idx) {
    constexpr uint32_t kNumRows = 8;
    constexpr bool kPartitioned = UMMA_M == 128 && kNumMmaCtas == 2;
    constexpr uint32_t kNumExpertsPerCta = UMMA_M / kNumMmaCtas;
    const uint32_t warp_in_wg = gate_warp_idx % 4;
    const uint32_t wg_idx = gate_warp_idx / 4;
    constexpr uint32_t kTokenStep = kNumGateWarps / 4 * kNumRows;
    const uint32_t num_token_cols = kPartitioned ? effective_umma_n / 2 : effective_umma_n;
    const uint32_t token_base = kPartitioned ? warp_in_wg / 2 * num_token_cols : 0u;
    const uint32_t expert_atom = kPartitioned ? warp_in_wg % 2 : warp_in_wg;
    const uint32_t global_expert = expert_base_idx + mma_cta_rank * kNumExpertsPerCta +
                                    expert_atom * 32 + get_lane_idx();

    bool finite = true;
    for (uint32_t t = wg_idx * kNumRows; t < num_token_cols; t += kTokenStep) {
        uint32_t values[kNumRows];
        tmem_load_32dp32b_rows8(tmem_base + t, values);
        fence_view_async_tmem_load();
        float* row_base = scores + (uint64_t)(token_base + t) * kScoreTokenStride + global_expert;
        #pragma unroll
        for (uint32_t r = 0; r < kNumRows; ++r) {
            float score = __uint_as_float(values[r]);
            if (kScoreOnStore)
                score = sqrt_softplus(score);
            finite = finite && isfinite(score);
            row_base[(uint64_t)r * kScoreTokenStride] = score;
        }
    }
    return finite;
}

// ===========================================================================
// The megakernel
// ===========================================================================
template <uint32_t SHAPE_K, uint32_t kNumRoutedExperts,
          uint32_t kNumGateThreads,
          uint32_t kNumMmaCtas, uint32_t kNumSplitK, uint32_t kNumExpertGroups,
          uint32_t kNumTopk,
          bool kHasPaddedExperts, bool kHasBias, bool kHasImageTokenMask,
          bool kHasMask, bool kHasFixRoutingMask, bool kHasForceRandom,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumGateThreads, 1)
fp8_mega_gate_impl(const __grid_constant__ dg::TmaMap tensor_map_x,
                   const __grid_constant__ dg::TmaMap tensor_map_weight,
                   const __grid_constant__ dg::TmaMap tensor_map_x_sf,
                   const __grid_constant__ dg::TmaMap tensor_map_weight_sf,
                   const float* bias, const float* image_bias, const bool* image_token_mask,
                   const bool* mask, const bool* fix_routing_mask, const bool* force_random,
                   const __grid_constant__ RoutingArgs routing_args,
                   void* gmem_scratch, void* gmem_score_barriers,
                   uint32_t num_tokens, uint32_t block_tokens, uint32_t num_stages,
                   uint32_t num_worker_groups, uint32_t num_token_blocks) {
    constexpr uint32_t kNumNonEpilogueWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumGateWarps = kNumGateThreads / 32;
    static_assert(kNumGateThreads % 128 == 0, "gate warps");
    constexpr uint32_t UMMA_K = 16;  // tf32-era atom; fp8 uses K=32 per instr
    constexpr uint32_t kNumExpertsPerGroup = kNumRoutedExperts / kNumExpertGroups;
    constexpr uint32_t kNumLogicalCtas = kNumMmaCtas * kNumExpertGroups * kNumSplitK;
    using ScoreBarrierT = ScoreBarrier<kNumLogicalCtas>;
    constexpr uint32_t LOAD_BLOCK_M = kNumMaxBlockTokensCap / kNumMmaCtas;  // tokens/CTA
    constexpr uint32_t LOAD_BLOCK_N = kNumExpertsPerGroup / kNumMmaCtas;
    static_assert((kNumMmaCtas == 1 && kNumExpertsPerGroup == 128) ||
                  (kNumMmaCtas == 2 && (kNumExpertsPerGroup == 128 || kNumExpertsPerGroup == 256)),
                  "MMA shape");

    constexpr uint32_t UMMA_M = kNumExpertsPerGroup;
    constexpr uint32_t UMMA_N = kNumMaxBlockTokensCap;  // smem/tmem capacity
    constexpr bool kCacheLogicalCount =
        (bool)caches_logical_count(kNumRoutedExperts, kHasBias, kHasImageTokenMask, kHasPhysicalMap);
    constexpr bool kHasMetadataCache = kHasBias || kHasImageTokenMask || kCacheLogicalCount;

    // SF granularity (fp8): 32 elements per SF byte, packed 4x/int32.
    constexpr uint32_t SF_BLOCK_M = (UMMA_M + 127u) & ~127u;     // experts
    constexpr uint32_t SF_BLOCK_N = (UMMA_N + 127u) & ~127u;     // tokens
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols =
        get_num_aligned_tmem_cols<kNumEpilogueStages * UMMA_N + kNumSFATmemCols + kNumSFBTmemCols>();
    static_assert(kNumTmemCols <= 512, "tmem budget (host caps block_tokens <= 240)");
    constexpr uint32_t kTmemStartColOfSFA = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kTmemStartColOfSFB = kTmemStartColOfSFA + kNumSFATmemCols;

    constexpr uint32_t kNumWeightStageBytes = LOAD_BLOCK_N * BLOCK_K;  // fp8: 1B/elem
    constexpr uint32_t kNumXStageBytes = (kNumMaxBlockTokensCap / kNumMmaCtas) * BLOCK_K;

    const uint32_t mma_cta_rank = get_block_rank_in_cluster();
    const uint32_t mma_cluster_idx = blockIdx.x / kNumMmaCtas;
    const uint32_t slice_idx = mma_cluster_idx % (kNumExpertGroups * kNumSplitK);
    const uint32_t expert_group_idx = slice_idx % kNumExpertGroups;
    const uint32_t split_k_idx = slice_idx / kNumExpertGroups;
    const uint32_t logical_cta_rank = slice_idx * kNumMmaCtas + mma_cta_rank;
    constexpr uint32_t kNumKBlocksPerSlice = (SHAPE_K / BLOCK_K) / kNumSplitK;
    const uint32_t split_k_offset = split_k_idx * kNumKBlocksPerSlice * BLOCK_K;
    const bool is_leader_cta = mma_cta_rank == 0;
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    if (warp_idx == 0) {
        prefetch_tma_map(&tensor_map_x);
        prefetch_tma_map(&tensor_map_weight);
        prefetch_tma_map(&tensor_map_x_sf);
        prefetch_tma_map(&tensor_map_weight_sf);
    }

    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];
    SharedStorage& smem = *(SharedStorage*)smem_buffer;
    float* smem_bias = smem.metadata_cache;
    float* smem_image_bias = smem_bias + (kHasBias ? kNumRoutedExperts : 0);
    int* smem_logical_count = (int*)(smem_image_bias + (kHasImageTokenMask ? kNumRoutedExperts : 0));
    const int* routed_logical_count = kCacheLogicalCount ? smem_logical_count
                                                          : routing_args.logical_count;

    if (kNumMmaCtas > 1)
        cluster_sync_relaxed();

    // Barrier init (warp 0-3 strided over stages).
    if (warp_idx < 4 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = warp_idx; i < kNumMaxStages; i += 4) {
            smem.full_barriers[i].init(kNumMmaCtas);
            smem.empty_barriers[i].init(1);
        }
        if (warp_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                smem.tmem_full_barriers[i].init(1);
                smem.tmem_empty_barriers[i].init(kNumMmaCtas);
            }
        }
        fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == 2)
        tmem_alloc_1sm(kNumTmemCols, &smem.tmem_ptr);
    kNumMmaCtas > 1 ? cluster_sync() : __syncthreads();

    griddepcontrol_wait();

    using WorkspaceT = Workspace<kNumSplitK, kNumRoutedExperts>;
    const WorkspaceT workspace(gmem_scratch, gmem_score_barriers);
    const auto effective_umma_n = [&](uint32_t token_block_idx) -> uint32_t {
        return align_u32(dg_min(block_tokens, num_tokens - token_block_idx * block_tokens), 16u);
    };
    const uint32_t worker_group_idx = blockIdx.x / kNumLogicalCtas;

    if (ld_shared_u32(&smem.tmem_ptr) != 0) asm volatile("trap;");
    if (num_stages > kNumMaxStages) asm volatile("trap;");

    // ===================== TMA PRODUCER WARP =====================
    if (warp_idx == 0 && elect_one_sync()) {
        // Per-CTA x window: block_tokens/ctas tokens at the split's offset.
        const uint32_t num_x_stage_bytes = (block_tokens / kNumMmaCtas) * BLOCK_K;
        const uint32_t expert_base = expert_group_idx * kNumExpertsPerGroup + mma_cta_rank * LOAD_BLOCK_N;
        uint32_t stage_idx = 0, phase = 0;
        for (uint32_t token_block = worker_group_idx; token_block < num_token_blocks;
             token_block += num_worker_groups) {
            const uint32_t eff_n = effective_umma_n(token_block);
            const uint32_t x_token = token_block * block_tokens + mma_cta_rank * (eff_n / kNumMmaCtas);
            #pragma unroll 4
            for (uint32_t kb = 0; kb < kNumKBlocksPerSlice; ++kb) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);
                const uint32_t k_idx = split_k_offset + kb * BLOCK_K;

                // x (tokens, B operand): EVICT_FIRST (streamed once).
                if (kNumMmaCtas == 2)
                    tma_load_2d_2sm_hint(&tensor_map_x, &smem.full_barriers[stage_idx],
                                         x_stage_ptr(smem_buffer, stage_idx, num_x_stage_bytes),
                                         k_idx, x_token, kEvictFirstHint);
                else
                    tma_load_2d_hint(&tensor_map_x, &smem.full_barriers[stage_idx],
                                     x_stage_ptr(smem_buffer, stage_idx, num_x_stage_bytes),
                                     k_idx, x_token, kEvictFirstHint);
                // weights (experts, A operand).
                if (kNumMmaCtas == 2)
                    tma_load_2d_2sm_hint(&tensor_map_weight, &smem.full_barriers[stage_idx],
                                         weight_stage_ptr(smem_buffer, stage_idx, num_stages,
                                                          num_x_stage_bytes, kNumWeightStageBytes),
                                         k_idx, expert_base, kEvictNormalHint);
                else
                    tma_load_2d_hint(&tensor_map_weight, &smem.full_barriers[stage_idx],
                                     weight_stage_ptr(smem_buffer, stage_idx, num_stages,
                                                      num_x_stage_bytes, kNumWeightStageBytes),
                                     k_idx, expert_base, kEvictNormalHint);
                // SFs: full tiles on every CTA (the MMA reads both halves).
                if (kNumMmaCtas == 2) {
                    tma_load_2d_2sm(&tensor_map_weight_sf, &smem.full_barriers[stage_idx],
                                    sfa_stage_ptr(smem_buffer, num_stages, num_x_stage_bytes,
                                                  kNumWeightStageBytes),
                                    kEvictNormalHint, expert_base, 0u);
                    tma_load_2d_2sm(&tensor_map_x_sf, &smem.full_barriers[stage_idx],
                                    sfb_stage_ptr(smem_buffer, num_stages, num_x_stage_bytes,
                                                  kNumWeightStageBytes,
                                                  SF_BLOCK_M * (SHAPE_K / 128) * 4),
                                    kEvictNormalHint, 0u, 0u);
                } else {
                    tma_load_2d(&tensor_map_weight_sf, &smem.full_barriers[stage_idx],
                                sfa_stage_ptr(smem_buffer, num_stages, num_x_stage_bytes,
                                              kNumWeightStageBytes),
                                kEvictNormalHint, expert_base, 0u);
                    tma_load_2d(&tensor_map_x_sf, &smem.full_barriers[stage_idx],
                                sfb_stage_ptr(smem_buffer, num_stages, num_x_stage_bytes,
                                              kNumWeightStageBytes,
                                              SF_BLOCK_M * (SHAPE_K / 128) * 4),
                                kEvictNormalHint, 0u, 0u);
                }

                if (is_leader_cta)
                    smem.full_barriers[stage_idx].arrive_and_expect_tx(
                        (num_x_stage_bytes + kNumWeightStageBytes) * kNumMmaCtas
                        + (SF_BLOCK_M + SF_BLOCK_N) * (SHAPE_K / 128) * 4 * kNumMmaCtas);
                else
                    smem.full_barriers[stage_idx].arrive();
                if (++stage_idx == num_stages) { stage_idx = 0; phase ^= 1u; }
            }
        }
    }
    // ===================== MMA WARP (leader CTA only) =====================
    else if (warp_idx == 1 && is_leader_cta) {
        // Block-scaled descriptors: A = weights (e4m3), B = x (e4m3).
        InstrDescriptorBlockScaled instr_desc =
            make_instr_desc_bs(0 /*e4m3*/, 0 /*e4m3*/, UMMA_M, UMMA_N, MAJOR_K, MAJOR_K);
        const uint32_t num_x_stage_bytes = (block_tokens / kNumMmaCtas) * BLOCK_K;

        SmemDescriptor a_desc = make_umma_desc<MAJOR_K, LOAD_BLOCK_N, BLOCK_K, kSwizzleMode, 1, 1>(
            weight_stage_ptr(smem_buffer, 0, num_stages, num_x_stage_bytes, kNumWeightStageBytes),
            0, 0);
        SmemDescriptor b_desc = make_umma_desc<MAJOR_K, kNumMaxBlockTokensCap / kNumMmaCtas, BLOCK_K,
                                               kSwizzleMode, 1, 1>(
            x_stage_ptr(smem_buffer, 0, num_x_stage_bytes), 0, 0);
        SmemDescriptor sf_desc = make_sf_desc(nullptr);
        const uint32_t a_desc_lo = a_desc.lo, b_desc_lo = b_desc.lo;
        uint32_t a_base_lo = a_desc_lo, b_base_lo = b_desc_lo;

        uint32_t stage_idx = 0, phase = 0, iter_idx = 0;
        for (uint32_t token_block = worker_group_idx; token_block < num_token_blocks;
             token_block += num_worker_groups, ++iter_idx) {
            // Dynamic N from the effective token count.
            instr_desc.n_dim_ = effective_umma_n(token_block) >> 3;

            const uint32_t accum_stage = iter_idx % kNumEpilogueStages;
            const uint32_t accum_phase = iter_idx / kNumEpilogueStages & 1;
            smem.tmem_empty_barriers[accum_stage].wait(accum_phase ^ 1);
            tcgen05_after_thread_sync();

            const auto empty_arrive = [&](bool do_tmem_full) {
                if (kNumMmaCtas == 1)
                    umma_arrive_1sm(&smem.empty_barriers[stage_idx]);
                else
                    umma_arrive_2sm_multicast(&smem.empty_barriers[stage_idx], 0x3);
                if (do_tmem_full) {
                    if (kNumMmaCtas == 1)
                        umma_arrive_1sm(&smem.tmem_full_barriers[accum_stage]);
                    else
                        umma_arrive_2sm_multicast(&smem.tmem_full_barriers[accum_stage], 0x3);
                }
                __syncwarp();
            };

            for (uint32_t kb = 0; kb < kNumKBlocksPerSlice; ++kb) {
                smem.full_barriers[stage_idx].wait(phase);
                tcgen05_after_thread_sync();
                const uint64_t runtime_desc = make_runtime_instr_desc_bs(instr_desc, 0, 0);

                if (elect_one_sync()) {
                    // UTCCP: SFA (experts) + SFB (tokens), 128-row pages.
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / 128; ++i) {
                        replace_smem_desc_addr(sf_desc,
                            (uint32_t*)sfa_stage_ptr(smem_buffer, num_stages, 0,
                                                     kNumWeightStageBytes) + i * 128);
                        if (kNumMmaCtas == 1)
                            utccp_4x32dp128bit_1cta(sf_desc.desc_, kTmemStartColOfSFA + i * 4);
                        else
                            utccp_4x32dp128bit_2cta(sf_desc.desc_, kTmemStartColOfSFA + i * 4);
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / 128; ++i) {
                        replace_smem_desc_addr(sf_desc,
                            (uint32_t*)sfb_stage_ptr(smem_buffer, num_stages, 0,
                                                     kNumWeightStageBytes,
                                                     SF_BLOCK_M * (SHAPE_K / 128) * 4) + i * 128);
                        if (kNumMmaCtas == 1)
                            utccp_4x32dp128bit_1cta(sf_desc.desc_, kTmemStartColOfSFB + i * 4);
                        else
                            utccp_4x32dp128bit_2cta(sf_desc.desc_, kTmemStartColOfSFB + i * 4);
                    }
                    // BLOCK_K=64 fp8: 2 MMA atoms (K=32), sf_id = atom index.
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / 32; ++k) {
                        a_desc.lo = advance_umma_desc_lo<MAJOR_K, LOAD_BLOCK_N, kSwizzleMode, 1, 1>(
                            a_base_lo, 0, k * 32);
                        b_desc.lo = advance_umma_desc_lo<MAJOR_K, kNumMaxBlockTokensCap / kNumMmaCtas,
                                                         kSwizzleMode, 1, 1>(
                            b_base_lo, 0, k * 32);
                        const uint64_t rd = make_runtime_instr_desc_bs(instr_desc, k, k);
                        if (kNumMmaCtas == 1)
                            mma_mxf8f6f4_1sm(a_desc.desc_, b_desc.desc_,
                                             accum_stage * UMMA_N,
                                             (kb > 0 || k > 0) ? 1u : 0u, rd,
                                             kTmemStartColOfSFA, kTmemStartColOfSFB);
                        else
                            mma_mxf8f6f4_2sm(a_desc.desc_, b_desc.desc_,
                                             accum_stage * UMMA_N,
                                             (kb > 0 || k > 0) ? 1u : 0u, rd,
                                             kTmemStartColOfSFA, kTmemStartColOfSFB);
                    }
                }
                __syncwarp();

                empty_arrive(kb == kNumKBlocksPerSlice - 1);
                if (++stage_idx == num_stages) {
                    stage_idx = 0; phase ^= 1;
                    a_base_lo = a_desc_lo; b_base_lo = b_desc_lo;
                } else {
                    a_base_lo += kNumWeightStageBytes / 16;
                    b_base_lo += (block_tokens / kNumMmaCtas) * BLOCK_K / 16;
                }
            }
        }
        // Barrier deconstruction safety for remote (peer CTA) arrivals.
        if (kNumMmaCtas > 1 && iter_idx > 0) {
            const uint32_t last = iter_idx - 1;
            smem.tmem_empty_barriers[last % kNumEpilogueStages].wait(last / kNumEpilogueStages & 1);
        }
    }
    // ===================== GATE WARPS =====================
    else if (warp_idx >= kNumNonEpilogueWarps) {
        const uint32_t gate_warp_idx = warp_idx - kNumNonEpilogueWarps;
        const uint32_t gate_thread_idx = threadIdx.x - kNumNonEpilogueThreads;

        // Metadata cache fill (bias / image bias / logical count).
        if (kHasMetadataCache) {
            #pragma unroll
            for (uint32_t c = gate_thread_idx; c < kNumRoutedExperts; c += kNumGateThreads) {
                const bool valid = !kHasPaddedExperts || c < routing_args.num_routed_experts;
                if (kHasBias) smem_bias[c] = valid ? (*&bias[c]) : 0.0f;
                if (kHasImageTokenMask) smem_image_bias[c] = valid ? (*&image_bias[c]) : 0.0f;
                if (kCacheLogicalCount)
                    smem_logical_count[c] = valid ? (*&routing_args.logical_count[c]) : 0;
            }
        }
        for (uint32_t token_block = worker_group_idx, iter = 0;
             token_block < num_token_blocks;
             token_block += num_worker_groups, ++iter) {
            const uint32_t token_base = token_block * block_tokens;
            const uint32_t token_end = dg_min(token_base + block_tokens, num_tokens);
            const uint32_t eff_n = effective_umma_n(token_block);
            const uint32_t accum_stage = iter % kNumEpilogueStages;
            const uint32_t accum_phase = (iter / kNumEpilogueStages) & 1;
            uint64_t* score_bar = workspace.get_score_barrier_ptr(token_block);

            // The first logical CTA re-arms the barrier for this grid.
            if (gate_warp_idx == kNumGateWarps - 1 && kNumLogicalCtas > 1 && elect_one_sync()) {
                if (logical_cta_rank == 0)
                    ScoreBarrierT::init(score_bar);
                else
                    ScoreBarrierT::wait_init(score_bar);
            }
            __syncwarp();

            smem.tmem_full_barriers[accum_stage].wait(accum_phase);
            tcgen05_after_thread_sync();

            const uint32_t expert_group_base = expert_group_idx * kNumExpertsPerGroup;
            constexpr bool kHasScoringBias = kHasBias || kHasImageTokenMask;
            // sqrt-softplus is non-linear: split-K sums partials in the topk
            // loop; a single split applies it at store.
            constexpr bool kScoreOnStore = kHasScoringBias && kNumSplitK == 1;
            const bool gate_ok = store_tmem_scores_to_global<
                kScoreOnStore, WorkspaceT::kTokenStride, UMMA_M, kNumMmaCtas, kNumGateWarps>(
                    workspace.get_score_ptr(token_base, split_k_idx),
                    accum_stage * UMMA_N, eff_n, expert_group_base,
                    mma_cta_rank, gate_warp_idx);
            named_barrier_sync(kNumGateThreads, 0);
            if (gate_warp_idx == kNumGateWarps - 1 && elect_one_sync()) {
                tcgen05_before_thread_sync();
                smem.tmem_empty_barriers[accum_stage].arrive();
                if (kNumLogicalCtas > 1)
                    ScoreBarrierT::arrive(score_bar);
            }
            if (gate_warp_idx == 0 && kNumLogicalCtas > 1 && elect_one_sync())
                ScoreBarrierT::wait(score_bar);
            named_barrier_sync(kNumGateThreads, 0);

            // Load + reduce split partials (float or float4 vector by width).
            const auto load_scores = [&](uint32_t tok, uint32_t exp, auto& out) {
                constexpr uint32_t kN = sizeof(out) / 4;
                const float* p = workspace.get_score_ptr(tok) + exp;
                if constexpr (kN == 1) {
                    out = *p;
                    #pragma unroll
                    for (uint32_t sp = 1; sp < kNumSplitK; ++sp)
                        out += *(p + (uint64_t)sp * kNumRoutedExperts);
                } else {
                    float4 v4 = *(const float4*)p;
                    #pragma unroll
                    for (uint32_t sp = 1; sp < kNumSplitK; ++sp) {
                        const float4 q4 = *(const float4*)(p + (uint64_t)sp * kNumRoutedExperts);
                        v4.x += q4.x; v4.y += q4.y; v4.z += q4.z; v4.w += q4.w;
                    }
                    out = v4;
                }
            };
            const auto load_reduced = [&](uint32_t tok, uint32_t exp) {
                float s = 0.0f;
                load_scores(tok, exp, s);
                return s;
            };

            // Each logical CTA ranks its share of the block's tokens.
            for (uint32_t tok = token_base + gate_warp_idx * kNumLogicalCtas + logical_cta_rank;
                 tok < token_end;
                 tok += kNumGateWarps * kNumLogicalCtas) {
                const bool is_masked = kHasMask && !(*&mask[tok]);
                const bool is_force_random = kHasForceRandom && (*&force_random[tok]);
                const bool has_fixed = kHasFixRoutingMask && (*&fix_routing_mask[tok]);
                const bool is_image = kHasImageTokenMask && (*&image_token_mask[tok]);

                if (is_masked) {
                    store_token_outputs<kNumTopk>(routing_args, tok, lane_idx, -1, 0.0f, -1);
                    if (kHasUnmappedTopkIdx)
                        store_token_outputs_unmapped<kNumTopk>(routing_args, tok, lane_idx, -1);
                    continue;
                }
                if (is_force_random) {
                    store_random_topk_token<kNumTopk>(routing_args, tok, lane_idx, kHasPhysicalMap,
                                                      kHasUnmappedTopkIdx);
                    continue;
                }
                if (kHasFixRoutingMask && has_fixed) {
                    int selected = -1;
                    float unbiased = 0.0f;
                    if (lane_idx < kNumTopk) {
                        const int64_t fixed = (*&routing_args.unmapped_topk_idx[
                            (uint64_t)tok * routing_args.unmapped_topk_idx_stride + lane_idx]);
                        selected = (int)fixed;
                        unbiased = load_reduced(tok, (uint32_t)selected);
                        if (!kScoreOnStore)
                            unbiased = sqrt_softplus(unbiased);
                    }
                    store_topk_token<kNumTopk>(routing_args, routed_logical_count, tok, lane_idx,
                                               selected, unbiased, kHasPhysicalMap,
                                               kHasUnmappedTopkIdx);
                    continue;
                }

                // Full ranking path: all experts in 4-wide lane vectors.
                constexpr uint32_t kNumExpertWaves = kNumRoutedExperts / kNumExpertsPerWave;
                float scores_local[kNumExpertWaves * kNumExpertsPerLaneVector];
                #pragma unroll
                for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
                    const uint32_t gbase = w * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
                    const uint32_t lbase = w * kNumExpertsPerLaneVector;
                    float4 sv;
                    load_scores(tok, gbase, sv);
                    (void)lbase;
                    scores_local[lbase + 0] = sv.x;
                    scores_local[lbase + 1] = sv.y;
                    scores_local[lbase + 2] = sv.z;
                    scores_local[lbase + 3] = sv.w;
                }
                #pragma unroll
                for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
                    const uint32_t gbase = w * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
                    const uint32_t lbase = w * kNumExpertsPerLaneVector;
                    if (kHasScoringBias && kNumSplitK > 1)
                        sqrt_softplus_x4((float (&)[4])scores_local[lbase]);
                    // Bias (image tokens swap in the image bias).
                    const float* bias_src = nullptr;
                    if (kHasBias) bias_src = smem_bias;
                    if (kHasImageTokenMask) bias_src = is_image ? smem_image_bias : bias_src;
                    float4 bv = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (bias_src) {
                        const float4 b4 = *(const float4*)(bias_src + gbase);
                        bv = b4;
                    }
                    // Only the last wave can hold padded experts.
                    const bool valid_expert = !kHasPaddedExperts ||
                                              w + 1 < kNumExpertWaves ||
                                              gbase < routing_args.num_routed_experts;
                    #pragma unroll
                    for (uint32_t v = 0; v < kNumExpertsPerLaneVector; ++v) {
                        const float s = scores_local[lbase + v];
                        scores_local[lbase + v] = valid_expert
                            ? s + ((const float*)&bv)[v]
                            : -3.402823466e38f;
                    }
                }

                int selected_expert = -1;
                float selected_unbiased = 0.0f;
                select_warp_topk<kNumExpertWaves, kNumTopk>(scores_local, selected_expert);
                const bool gate_valid = gate_ok && (lane_idx >= kNumTopk ||
                    (selected_expert >= 0 && selected_expert < (int)routing_args.num_routed_experts));
                // Keep re-reads in bounds; the trap below reports invalid picks.
                selected_expert = lane_idx < kNumTopk
                    ? (int)dg_min((uint32_t)dg_max(selected_expert, 0),
                                  routing_args.num_routed_experts - 1)
                    : selected_expert;
                if (kNumSplitK == 1) {
                    if (lane_idx < kNumTopk)
                        selected_unbiased = load_reduced(tok, (uint32_t)selected_expert);
                    __syncwarp();
                } else {
                    // Split-K: fetch the unbiased value from the wave arrays.
                    float unbiased_local[kNumExpertWaves * kNumExpertsPerLaneVector];
                    #pragma unroll
                    for (uint32_t w = 0; w < kNumExpertWaves; ++w) {
                        const uint32_t gbase = w * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
                        const uint32_t lbase = w * kNumExpertsPerLaneVector;
                        float4 uv;
                        load_scores(tok, gbase, uv);
                        (void)lbase;
                        unbiased_local[lbase + 0] = uv.x;
                        unbiased_local[lbase + 1] = uv.y;
                        unbiased_local[lbase + 2] = uv.z;
                        unbiased_local[lbase + 3] = uv.w;
                    }
                    selected_unbiased = select_warp_expert_value<kNumExpertWaves>(
                        unbiased_local, selected_expert);
                }
                // Without a bias the ranking is on raw scores (monotonic);
                // only the selected ones get activated.
                if (!kHasScoringBias && lane_idx < kNumTopk)
                    selected_unbiased = sqrt_softplus(selected_unbiased);

                store_topk_token<kNumTopk>(routing_args, routed_logical_count, tok, lane_idx,
                                           selected_expert, selected_unbiased,
                                           kHasPhysicalMap, kHasUnmappedTopkIdx);
                if (!gate_valid) asm volatile("trap;");
            }
        }
    }

    kNumMmaCtas > 1 ? cluster_sync_relaxed() : __syncthreads();
    if (warp_idx == 0)
        tmem_dealloc_1sm(0, kNumTmemCols);
}

} // namespace mega_gate
} // namespace dg
