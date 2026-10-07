// DeepGEMM-RS: SM100 MQA logits (weighted-ReLU MQA scoring for the MLA
// lightning indexer / absorbed decode), contiguous-KV variant.
// Port of upstream `impls/sm100_mqa_logits.cuh` +
// `scheduler/sm100_mqa_logits.cuh` (grid-stride mode, no schedule metadata).
//
// Computes, for each query token i and KV position j in [k_start_i, k_end_i):
//   logits[i, j - k_start_i] = sum_h w[i, h] * relu(<q[i, h, :], kv[j, :]>)
// with MXFP8 or MXFP4 (packed E2M1) Q/KV and UE8M0 block scales.
//
// ===========================================================================
// STRUCTURE — one kernel, four cooperating warp classes
// ===========================================================================
// Thread layout (kNumSpecializedThreads=128 + kNumMathThreads=256):
//
//   lane axis (threads) ──────────────────────────────────────────────────>
//   0                128                256               384
//   ├─────────────────┬─────────────────────┬───────────────────────────────┤
//   │ math WG 0       │ math WG 1           │  warp 8  9 10 11 (specialized)
//   │ (drain TMEM,    │ (drain TMEM,        │   │   │  │  └ MMA issue (w11)
//   │  weighted ReLU, │  weighted ReLU,     │   │   │  └ SF transpose+UTCCP (w10)
//   │  scatter store) │  scatter store)     │   │   └ SF/Q/W TMA producer (w9)
//   └─────────────────┴─────────────────────┴───┴────┴─ KV TMA producer (w8)
//
// Register economics: specialized warps run setmaxnreg.dec to 56 registers
// (they only push descriptors); the freed registers are granted to the math
// warpgroups via setmaxnreg.inc — more live accumulators per math thread,
// fewer TMEM round trips.
//
// RingPipeline staging (Q, KV, SF, TMEM each have their own stage rings):
//   producer: stage = ring.advance()  (returns {stage, phase})
//             ... fill smem[stage] ...
//             full_barrier[stage].arrive()
//   consumer: {stage, phase} = ring.advance()  (upstream pre-advance
//             semantics: a fresh ring's first advance returns stage 0
//             phase 0 without consuming, then waits on parity)
//             full_barrier[stage].wait(phase)
//   KV stages are reused across Q blocks (the same KV span serves many
//   tokens), so the KV ring is the deepest; Q weights arrive piggy-backed
//   on the SF producer to keep one TMA issue stream.
//
// The MMA: Q is [BLOCK_Q * heads, head_dim] (UMMA_N aligned to 8), KV is
// [SPLIT_KV, head_dim]; the MMA computes all (token, kv) logits for one
// stage into TMEM; math warps then read them with tcgen05.ld.32x32b
// (lane = token slot), apply w * relu(.) per head pair in bf16x2 FMA, and
// scatter-store each token's row into the global logits buffer.
// ===========================================================================
// ===========================================================================

namespace dg {

// ---------------------------------------------------------------------------
// Scheduler (grid-stride over Q blocks; port of SM100MQALogitsScheduler)
// ---------------------------------------------------------------------------
struct MQALogitsTask {
    uint32_t q_token_base;
    uint32_t num_q_tokens;
    uint32_t kv_token_base;
    uint32_t num_kv_splits;
    bool kv_shared_with_prev;
};

template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs>
struct MQALogitsScheduler {
    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    const uint32_t* cu_seq_len_k_start;
    const uint32_t* cu_seq_len_k_end;

    uint32_t current_q_block_idx;

    DG_DEVICE MQALogitsScheduler(uint32_t sm_idx, uint32_t num_q_tokens_, uint32_t num_kv_tokens_,
                                 const uint32_t* k_start_, const uint32_t* k_end_)
        : num_q_blocks(ceil_div_u32(num_q_tokens_, BLOCK_Q)),
          num_q_tokens(num_q_tokens_), num_kv_tokens(num_kv_tokens_),
          cu_seq_len_k_start(k_start_), cu_seq_len_k_end(k_end_),
          current_q_block_idx(sm_idx) {}

    template <bool kLoadSeqBounds>
    DG_DEVICE bool advance(MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        if (current_q_block_idx >= num_q_blocks) return false;
        const uint32_t q_block_idx = current_q_block_idx;
        current_q_block_idx += kNumSMs;
        task.q_token_base = q_block_idx * BLOCK_Q;
        task.num_q_tokens = BLOCK_Q;
        task.kv_shared_with_prev = false;

        uint32_t start = 0xffffffffu, end = 0u;
        #pragma unroll
        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++token_idx) {
            const uint32_t row_idx = dg_min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
            const uint32_t k_start = dg_min(cu_seq_len_k_start[row_idx], num_kv_tokens);
            const uint32_t k_end = dg_min(cu_seq_len_k_end[row_idx], num_kv_tokens);
            if (kLoadSeqBounds) {
                seq_k_start[token_idx] = k_start;
                seq_k_end[token_idx] = k_end;
            }
            start = dg_min(start, k_start);
            end = dg_max(end, k_end);
        }
        task.kv_token_base = start / 4 * 4;
        task.num_kv_splits = ceil_div_u32(end - task.kv_token_base, SPLIT_KV);
        return true;
    }

    DG_DEVICE bool next_q_block(MQALogitsTask& task) { return advance<false>(task, nullptr, nullptr); }
    DG_DEVICE bool next_q_block(MQALogitsTask& task, uint32_t* s, uint32_t* e) { return advance<true>(task, s, e); }
};

// Ring pipeline cursor (port of common/ring_pipeline.cuh): `advance(step)`
// returns the (stage, phase) BEFORE stepping, as a packed u32 pair.
struct StagePhase {
    uint32_t stage, phase;
};

template <uint32_t kNumStages>
struct RingPipeline {
    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    DG_DEVICE StagePhase advance(const uint32_t step = 1) {
        const StagePhase cur = {stage_idx, phase};
        uint32_t next = stage_idx + step;
        if ((kNumStages & (kNumStages - 1)) == 0) {
            stage_idx = next % kNumStages;
            phase ^= next / kNumStages;
        } else {
            stage_idx = next;
            if (stage_idx >= kNumStages) {
                stage_idx -= kNumStages;
                phase ^= 1u;
            }
        }
        return cur;
    }
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t UMMA_N,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs, bool kIsFP4>
DG_GLOBAL void __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
mqa_logits_sm100_impl(uint32_t num_q_tokens, uint32_t num_kv_tokens, uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
                      bf16_raw* logits,
                      const __grid_constant__ TmaMap tensor_map_q,
                      const __grid_constant__ TmaMap tensor_map_sf_q,
                      const __grid_constant__ TmaMap tensor_map_kv,
                      const __grid_constant__ TmaMap tensor_map_sf_kv,
                      const __grid_constant__ TmaMap tensor_map_weights) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128, "4 specialized warps");
    DG_STATIC_ASSERT(kNumMathThreads % 128 == 0, "math threads multiple of 128");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * 128, "SPLIT_KV == math threads");
    DG_STATIC_ASSERT(kNumTmemStages == kNumMathWarpGroups, "TMEM stage per math WG");

    constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;          // packed E2M1
    constexpr uint32_t kWireElemSize = 1;                      // fp8 / packed fp4
    constexpr uint32_t kNumQKBytesPerToken = kHeadDim * kWireElemSize / kPackFactor;
    constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
    constexpr uint32_t kNumSFQ = BLOCK_QH;                     // one packed SF word per (token, head)
    constexpr uint32_t kNumSFKV = SPLIT_KV;                    // one packed SF word per KV token
    constexpr uint32_t kNumWeightElemsPerRow = kNumHeads;      // bf16 weights
    constexpr uint32_t kNumWeightBytesPerRow = kNumHeads * 2;
    constexpr uint32_t kNumKVBytesPerStage = SPLIT_KV * kNumQKBytesPerToken;
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor * kWireElemSize;
    DG_STATIC_ASSERT((!kIsFP4 && (kHeadDim == 32 || kHeadDim == 64 || kHeadDim == 128)) ||
                     (kIsFP4 && (kHeadDim == 64 || kHeadDim == 128)),
                     "Invalid head dim");
    DG_STATIC_ASSERT(8 <= UMMA_N && UMMA_N <= 256, "Invalid UMMA_N");

    // TMEM: accumulators (UMMA_N per stage) + SF Q/KV columns.
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
    constexpr uint32_t kNumSFQColsPerStage = kNumSFQ / 32;
    constexpr uint32_t kNumSFKVColsPerStage = kNumSFKV / 32;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<
        kNumAccumTmemCols + kNumQStages * kNumSFQColsPerStage + kNumKVStages * kNumSFKVColsPerStage>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumQStages * kNumSFQColsPerStage;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many TMEM cols");

    struct SharedStorage {
        alignas(1024) uint8_t smem_q[kNumQStages][BLOCK_QH * kNumQKBytesPerToken];
        alignas(1024) uint8_t smem_kv[kNumKVStages][kNumKVBytesPerStage];
        alignas(1024) uint32_t smem_sf_q[kNumQStages][kNumSFQ];
        alignas(1024) uint32_t smem_sf_kv[kNumKVStages][kNumSFKV];
        alignas(1024) uint16_t smem_weights[kNumQStages][BLOCK_Q * kNumWeightElemsPerRow];
        Barrier full_q_barriers[kNumQStages];
        Barrier full_sf_q_barriers[kNumQStages];
        Barrier empty_q_barriers[kNumQStages];
        Barrier full_kv_barriers[kNumKVStages];
        Barrier full_sf_kv_barriers[kNumKVStages];
        Barrier empty_kv_barriers[kNumKVStages];
        Barrier full_tmem_barriers[kNumTmemStages];
        Barrier empty_tmem_barriers[kNumTmemStages];
        uint32_t tmem_ptr_in_smem;
    };
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SharedStorage* smem = (SharedStorage*)smem_buffer;

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    if (warp_idx == kSpecWarpStart) {
        prefetch_tma_map(&tensor_map_q);
        prefetch_tma_map(&tensor_map_sf_q);
        prefetch_tma_map(&tensor_map_weights);
        prefetch_tma_map(&tensor_map_kv);
        prefetch_tma_map(&tensor_map_sf_kv);
    }

    if (warp_idx == kSpecWarpStart + 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++i) {
            smem->full_q_barriers[i].init(2);
            smem->full_sf_q_barriers[i].init(1);
            smem->empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++i) {
            smem->full_kv_barriers[i].init(2);
            smem->full_sf_kv_barriers[i].init(1);
            smem->empty_kv_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem->full_tmem_barriers[i].init(1);
            smem->empty_tmem_barriers[i].init(128);
        }
        fence_barrier_init();
    }
    __syncthreads();
    if (warp_idx == kSpecWarpStart + 2)
        tmem_alloc_1sm(kNumTmemCols, &smem->tmem_ptr_in_smem);
    __syncthreads();

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumWarpGroups = kNumMathWarpGroups + 1;
    constexpr uint32_t kNumEntryRegisters = (512 / kNumWarpGroups / 8) * 8;
    constexpr uint32_t kNumMathRegisters =
        ((kNumEntryRegisters * kNumWarpGroups - kNumSpecializedRegisters) / kNumMathWarpGroups / 8) * 8;

    griddepcontrol_wait();

    MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> make_sched(sm_idx, num_q_tokens, num_kv_tokens,
                                                              cu_seq_len_k_start, cu_seq_len_k_end);

    // Shared producers ------------------------------------------------------
    const auto issue_contiguous_kv = [&](uint32_t kv_stage_idx, uint32_t kv_token_offset,
                                          bool is_sf_producer) {
        constexpr uint32_t kNumKVTokensPerTMA = SPLIT_KV % 256 == 0 ? 256 : 128;
        DG_STATIC_ASSERT(SPLIT_KV % kNumKVTokensPerTMA == 0, "whole TMA tiles");
        if (is_sf_producer) {
            #pragma unroll
            for (uint32_t t = 0; t < SPLIT_KV; t += kNumKVTokensPerTMA) {
                tma_load_2d(&tensor_map_sf_kv, &smem->full_sf_kv_barriers[kv_stage_idx],
                            smem->smem_sf_kv[kv_stage_idx] + t, kEvictNormalHint,
                            kv_token_offset + t, 0);
            }
            smem->full_sf_kv_barriers[kv_stage_idx].arrive_and_expect_tx(kNumSFKV * sizeof(uint32_t));
        } else {
            #pragma unroll
            for (uint32_t t = 0; t < SPLIT_KV; t += kNumKVTokensPerTMA) {
                tma_load_2d(&tensor_map_kv, &smem->full_kv_barriers[kv_stage_idx],
                            smem->smem_kv[kv_stage_idx] + t * kNumQKBytesPerToken,
                            kEvictNormalHint, 0, kv_token_offset + t);
            }
            smem->full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(kNumKVBytesPerStage);
        }
    };

    if (warp_idx == kSpecWarpStart) {
        // KV data producer
        setmaxnreg_dec<kNumSpecializedRegisters>();
        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> sched = make_sched;
        MQALogitsTask task;
        while (sched.next_q_block(task)) {
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                const StagePhase kv = kv_pipeline.advance();
                if (elect_one_sync()) {
                    smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                    issue_contiguous_kv(kv.stage, task.kv_token_base + kv_split_idx * SPLIT_KV, false);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        // Q + weights + KV SF producer
        setmaxnreg_dec<kNumSpecializedRegisters>();
        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> sched = make_sched;
        MQALogitsTask task;
        while (sched.next_q_block(task)) {
            const StagePhase q = q_pipeline.advance();
            if (elect_one_sync())
                smem->empty_q_barriers[q.stage].wait(q.phase ^ 1);
            __syncwarp();

            if (elect_one_sync()) {
                // Q SF: box (BLOCK_QH, 1) at (0, q_token_base).
                tma_load_2d(&tensor_map_sf_q, &smem->full_sf_q_barriers[q.stage],
                            smem->smem_sf_q[q.stage], kEvictNormalHint,
                            0, task.q_token_base);
                smem->full_sf_q_barriers[q.stage].arrive_and_expect_tx(BLOCK_QH * sizeof(uint32_t));
                // Q data: box (kHeadDim, BLOCK_QH) at (0, q_token_base * kNumHeads).
                tma_load_2d(&tensor_map_q, &smem->full_q_barriers[q.stage],
                            smem->smem_q[q.stage], kEvictNormalHint,
                            0, task.q_token_base * kNumHeads);
                // Weights: box (kNumHeads, BLOCK_Q) at (0, q_token_base).
                tma_load_2d(&tensor_map_weights, &smem->full_q_barriers[q.stage],
                            smem->smem_weights[q.stage], kEvictNormalHint,
                            0, task.q_token_base);
                smem->full_q_barriers[q.stage].arrive_and_expect_tx(
                    BLOCK_QH * kNumQKBytesPerToken + BLOCK_Q * kNumWeightBytesPerRow);
            }
            __syncwarp();

            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                const StagePhase kv = kv_pipeline.advance();
                if (elect_one_sync()) {
                    smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                    issue_contiguous_kv(kv.stage, task.kv_token_base + kv_split_idx * SPLIT_KV, true);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        // SF transpose + UTCCP
        setmaxnreg_dec<kNumSpecializedRegisters>();

        auto utccp_required_smem_warp_transpose = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                values[i] = ld_shared_u32(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            st_shared_u32x4(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        SmemDescriptor sf_desc = make_sf_desc(nullptr);

        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> sched = make_sched;
        MQALogitsTask task;
        while (sched.next_q_block(task)) {
            const StagePhase q = q_pipeline.advance();
            smem->full_sf_q_barriers[q.stage].wait(q.phase);
            tcgen05_after_thread_sync();

            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / 128; ++i) {
                uint32_t* p = smem->smem_sf_q[q.stage] + i * 128;
                utccp_required_smem_warp_transpose(p);
            }
            fence_view_async_shared();
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / 128; ++i) {
                replace_smem_desc_addr(sf_desc, smem->smem_sf_q[q.stage] + i * 128);
                if (elect_one_sync())
                    utccp_4x32dp128bit_1cta(sf_desc.desc_,
                        kTmemStartColOfSFQ + q.stage * kNumSFQColsPerStage + i * 4);
                __syncwarp();
            }
            if (elect_one_sync()) {
                tcgen05_before_thread_sync();
                smem->full_q_barriers[q.stage].arrive();
            }
            __syncwarp();

            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                const StagePhase kv = kv_pipeline.advance();
                smem->full_sf_kv_barriers[kv.stage].wait(kv.phase);
                tcgen05_after_thread_sync();

                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / 128; ++i) {
                    uint32_t* p = smem->smem_sf_kv[kv.stage] + i * 128;
                    utccp_required_smem_warp_transpose(p);
                }
                fence_view_async_shared();
                __syncwarp();
                if (elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / 128; ++i) {
                        replace_smem_desc_addr(sf_desc, smem->smem_sf_kv[kv.stage] + i * 128);
                        utccp_4x32dp128bit_1cta(sf_desc.desc_,
                            kTmemStartColOfSFKV + kv.stage * kNumSFKVColsPerStage + i * 4);
                    }
                    tcgen05_before_thread_sync();
                    smem->full_kv_barriers[kv.stage].arrive();
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        // MMA issue: A = KV (128 rows per math WG), B = Q.
        setmaxnreg_dec<kNumSpecializedRegisters>();
        const uint32_t tmem_base = ld_shared_vol_u32(&smem->tmem_ptr_in_smem);
        if (elect_one_sync()) {
            constexpr uint32_t kNumUMMAK = kHeadDim / UMMA_K;
            InstrDescriptorBlockScaled instr_desc = make_instr_desc_bs(
                kIsFP4 ? 5u : 0u, kIsFP4 ? 5u : 0u, UMMA_M, UMMA_N, MAJOR_K, MAJOR_K);
            SmemDescriptor a_desc = make_umma_desc<MAJOR_K, 0, kHeadDim, kQKSwizzleMode, kPackFactor, 1>(
                smem->smem_kv[0], 0, 0);
            SmemDescriptor b_desc = make_umma_desc<MAJOR_K, 0, kHeadDim, kQKSwizzleMode, kPackFactor, 1>(
                smem->smem_q[0], 0, 0);
            const uint32_t a_desc_lo = a_desc.lo;
            const uint32_t b_desc_lo = b_desc.lo;
            constexpr uint32_t kNumKVDescPerStage = sizeof(smem->smem_kv[0]) / 16;
            constexpr uint32_t kNumQDescPerStage = sizeof(smem->smem_q[0]) / 16;
            uint64_t runtime_instr_descs[kNumUMMAK];
            #pragma unroll
            for (uint32_t k = 0; k < kNumUMMAK; ++k)
                runtime_instr_descs[k] = make_runtime_instr_desc_bs(instr_desc, k * kPackFactor, k * kPackFactor);

            uint32_t tmem_phase = 0;
            MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> sched = make_sched;
            MQALogitsTask task;
            while (sched.next_q_block(task)) {
                const StagePhase q = q_pipeline.advance();
                smem->full_q_barriers[q.stage].wait(q.phase);
                tcgen05_after_thread_sync();
                const uint32_t b_desc_stage_lo = b_desc_lo + q.stage * kNumQDescPerStage;
                const uint32_t tmem_sfb = tmem_base + kTmemStartColOfSFQ + q.stage * kNumSFQColsPerStage;

                for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                    const StagePhase kv = kv_pipeline.advance();
                    smem->full_kv_barriers[kv.stage].wait(kv.phase);
                    const uint32_t a_desc_stage_lo = a_desc_lo + kv.stage * kNumKVDescPerStage;
                    const uint32_t tmem_sfa = tmem_base + kTmemStartColOfSFKV
                                            + kv.stage * kNumSFKVColsPerStage;
                    #pragma unroll
                    for (uint32_t tmem_stage_idx = 0; tmem_stage_idx < kNumMathWarpGroups; ++tmem_stage_idx) {
                        smem->empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        tcgen05_after_thread_sync();
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumUMMAK; ++k) {
                            a_desc.lo = advance_umma_desc_lo<MAJOR_K, 0, kQKSwizzleMode, kPackFactor, 1>(
                                a_desc_stage_lo, tmem_stage_idx * UMMA_M * kHeadDim, k * UMMA_K);
                            b_desc.lo = advance_umma_desc_lo<MAJOR_K, 0, kQKSwizzleMode, kPackFactor, 1>(
                                b_desc_stage_lo, 0, k * UMMA_K);
                            const uint32_t accumulate = (k > 0) ? 1u : 0u;
                            if (kIsFP4) {
                                mma_mxf4_1sm(a_desc.desc_, b_desc.desc_,
                                             tmem_base + tmem_stage_idx * UMMA_N,
                                             accumulate, runtime_instr_descs[k],
                                             tmem_sfa + tmem_stage_idx * 4, tmem_sfb);
                            } else {
                                mma_mxf8f6f4_1sm(a_desc.desc_, b_desc.desc_,
                                                 tmem_base + tmem_stage_idx * UMMA_N,
                                                 accumulate, runtime_instr_descs[k],
                                                 tmem_sfa + tmem_stage_idx * 4, tmem_sfb);
                            }
                        }
                        umma_arrive_1sm(&smem->full_tmem_barriers[tmem_stage_idx]);
                    }
                    tmem_phase ^= 1;
                    umma_arrive_1sm(&smem->empty_kv_barriers[kv.stage]);
                }
                // The MMA warp counts toward the Q release barrier.
                smem->empty_q_barriers[q.stage].arrive_count(32);
            }
        }
        __syncwarp();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups: reduce weighted ReLU logits and scatter-store.
        setmaxnreg_inc<kNumMathRegisters>();
        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> sched = make_sched;
        uint32_t seq_k_start[BLOCK_Q];
        uint32_t seq_k_end[BLOCK_Q];
        const uint32_t math_warpgroup_idx = warp_idx / 4;
        const uint32_t math_thread_idx = warp_idx * 32 + lane_idx;
        tmem_pipeline.advance(math_warpgroup_idx);  // offset each WG's cursor

        DG_STATIC_ASSERT(kNumHeads % 4 == 0, "heads multiple of 4");
        // Every token's weights stay in registers.
        uint32_t weights[BLOCK_Q][kNumHeads / 2];
        float accum[kNumHeads < 16 ? kNumHeads : 16];

        MQALogitsTask task;
        while (sched.next_q_block(task, seq_k_start, seq_k_end)) {
            const StagePhase q = q_pipeline.advance();
            smem->full_q_barriers[q.stage].wait(q.phase);

            // Load per-token, per-head weights (bf16 pairs as packed u32).
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                const uint32_t* row = (const uint32_t*)smem->smem_weights[q.stage]
                                    + i * (kNumHeads / 2);
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 2; ++j)
                    weights[i][j] = ld_shared_u32(row + j);
            }

            // Output bases offset by -k_start so shared KV indexing works.
            bf16_raw* output_bases[BLOCK_Q];
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i)
                output_bases[i] = logits + (uint64_t)(task.q_token_base + i) * logits_stride - seq_k_start[i];

            const auto store_token = [&](uint32_t i, uint32_t kv_offset, uint32_t sum_0, uint32_t sum_1) {
                const uint32_t sum = add_bf16x2(sum_0, sum_1);
                const uint32_t result = add_bf16x2(low2_bf16x2(sum), high2_bf16x2(sum));
                if (seq_k_start[i] <= kv_offset && kv_offset < seq_k_end[i])
                    st_global_u16(output_bases[i] + kv_offset, result);
            };
            const auto reduce_pairs = [&](const float* chunk, const uint32_t* chunk_weights,
                                          uint32_t num_heads, uint32_t& sum_0, uint32_t& sum_1) {
                #pragma unroll
                for (uint32_t h = 0; h < num_heads; h += 4) {
                    const uint32_t c0 = cvt_relu_bf16x2_f32(chunk[h + 0], chunk[h + 1]);
                    const uint32_t c1 = cvt_relu_bf16x2_f32(chunk[h + 2], chunk[h + 3]);
                    sum_0 = fma_bf16x2(c0, chunk_weights[h / 2], sum_0);
                    sum_1 = fma_bf16x2(c1, chunk_weights[h / 2 + 1], sum_1);
                }
            };

            #pragma unroll 2
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                const uint32_t kv_offset = task.kv_token_base + kv_split_idx * SPLIT_KV + math_thread_idx;
                const StagePhase tp = tmem_pipeline.advance(kNumMathWarpGroups);
                smem->full_tmem_barriers[tp.stage].wait(tp.phase);
                tcgen05_after_thread_sync();

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                    uint32_t sum_0 = cvt_bf16x2_f32(0.0f, 0.0f);
                    uint32_t sum_1 = cvt_bf16x2_f32(0.0f, 0.0f);

                    // Visit this token's heads in 16/8/4-wide TMEM chunks.
                    const uint32_t tmem_col = tp.stage * UMMA_N + i * kNumHeads;
                    uint32_t head_base = 0;
                    #pragma unroll
                    while (head_base < kNumHeads) {
                        const uint32_t chunk = kNumHeads - head_base >= 16 ? 16 :
                                               (kNumHeads - head_base >= 8 ? 8 : 4);
                        if (chunk == 16) tmem_load_32dp32b_x16(tmem_col + head_base, accum);
                        else if (chunk == 8) {
                            tmem_load_32dp32b_x8(tmem_col + head_base, accum[0], accum[1], accum[2], accum[3],
                                                 accum[4], accum[5], accum[6], accum[7]);
                        } else {
                            tmem_load_32dp32b_x4(tmem_col + head_base, accum[0], accum[1], accum[2], accum[3]);
                        }
                        fence_view_async_tmem_load();
                        // Release TMEM after the last token's last chunk.
                        if (head_base + chunk == kNumHeads && i == BLOCK_Q - 1) {
                            tcgen05_before_thread_sync();
                            smem->empty_tmem_barriers[tp.stage].arrive();
                        }
                        reduce_pairs(accum, weights[i] + head_base / 2, chunk, sum_0, sum_1);
                        head_base += chunk;
                    }
                    store_token(i, kv_offset, sum_0, sum_1);
                }
            }

            fence_view_async_shared();
            smem->empty_q_barriers[q.stage].arrive();
        }

        named_barrier_sync(kNumMathThreads, 8);
        if (warp_idx == 0)
            tmem_dealloc_1sm(0, kNumTmemCols);
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("trap;");
#endif
}

} // namespace dg
