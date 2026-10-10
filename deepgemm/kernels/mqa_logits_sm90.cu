// ===========================================================================
// mqa_logits_sm90.cu — Hopper (SM90a) MQA-logits kernels, ported 1:1 from
// upstream DeepGEMM:
//   * `mqa_logits_sm90_impl`               (impls/sm90_fp8_mqa_logits.cuh)
//       contiguous-KV variant: a ragged Q sequence scored against one packed
//       KV buffer ([seq_len_kv, head_dim] fp8 + per-token fp32 scales).
//   * `mqa_paged_logits_sm90_impl`         (impls/sm90_fp8_paged_mqa_logits.cuh)
//       paged-KV decode variant: 64-token pages mapped per request by a block
//       table; a metadata kernel balances the work across SMs and the main
//       kernel walks per-SM (q_atom, kv_split) ranges.
//   * `sm90_paged_mqa_logits_metadata_impl` (scheduler/sm90_paged_mqa_logits.cuh)
//       the 32-thread metadata kernel; turns per-request context lengths into
//       per-SM schedule entries [num_sms + 1, 2].
//
// ---------------------------------------------------------------------------
// CONCEPT 1 — WHAT "MQA LOGITS" COMPUTES (the MLA lightning indexer)
// ---------------------------------------------------------------------------
// DeepSeek's MLA attention uses a *lightning indexer*: before the full
// attention pass, a cheap scorer ranks each (query token, KV position) pair
// so the expensive path only attends to the top positions.  The scorer is a
// single MQA-style projection with a twist — the per-head dot products are
// passed through a ReLU and a *learned per-(token, head) weight* before being
// folded over heads (the "weighted ReLU", DeepSeek-V3.2 DSA indexer):
//
//     logits[i, j] = s_kv[j] * SUM_h  w[i, h] * relu( <q[i, h, :>, kv[j, :]> )
//
//     q   : [seq, heads, head_dim]  fp8 (E4M3), NO scale on SM90 (raw fp8)
//     kv  : [seq_kv, head_dim]      fp8, dequantized by a per-token fp32
//                                    scale s_kv (this is where SM90 differs
//                                    from SM100: block scales are plain fp32
//                                    per KV token, not MX UE8M0)
//     w   : [seq, heads]            fp32 per-(token, head) weights
//
// The KV scale is folded in at the very END (after the head sum) because it
// is per-KV-row:  each lane owns 2 KV rows (v_0/v_1 below) and reads the two
// matching scales from SMEM before the WGMMA is even issued.
//
// ---------------------------------------------------------------------------
// CONCEPT 2 — THE WGMMA ORIENTATION AND WHY THE REDUCE WORKS
// ---------------------------------------------------------------------------
// One WGMMA instruction is m64 x nN x k32 (fp8, both operands from SMEM
// through 64-bit GMMA descriptors).  The MQA trick is to fold the *head*
// dimension into the N dimension of the MMA:
//
//     A = kv tile [BLOCK_KV, head_dim]   (M side; rows = KV positions)
//     B = q  tile [BLOCK_Q*heads, head_dim]  (N side; rows = token*heads,
//                                            i.e. row n = i*heads + h)
//     D[m, n] = <kv[m], q[n]>  ->  the (token i, head h, kv m) partial score.
//
// The accumulator lives in REGISTERS (SM90 WGMMA, no TMEM): N/2 floats per
// lane, laid out (see wgmma.h) as
//
//     acc[j]:  col(j) = 8*(j/4) + (lane%4)*2 + (j&1)      j in [0, N/2)
//              row(j) = 16*warp + lane/4 (+8 for j%4 >= 2)
//
// Because col(j) = i*heads + h, the BLOCK_Q consecutive accumulator slices of
// token i are exactly j in [i*heads/2, (i+1)*heads/2)  (heads/2 == kNumAccum
// per token) — that is the `shifted_accum = accum + i*(heads/2)` base below.
// Within token i, head h(j) = 8*(j/4) + 2*(lane%4) + (j&1), and the weight
// read `smem_w[i*heads + (w/2)*8 + (w&1) + (lane%4)*2]` with w=(j/4)*2+(j&1)
// evaluates to exactly `w[i, h]` (algebra in the comment at the read).
// The four lanes of a quad (lane%4) each hold 1/4 of the head sum; the final
// `shfl_xor 1, 2` butterfly over the quad completes it — that is the
// "inter-thread reduction".
//
// ---------------------------------------------------------------------------
// CONCEPT 3 — WARP SPECIALIZATION AND REGISTER ECONOMICS (setmaxnreg)
// ---------------------------------------------------------------------------
// 640 threads = 512 math (4 warpgroups) + 128 TMA (1 warpgroup).  The math
// warpgroups hold BLOCK_Q*heads/2 fp32 accumulators + BLOCK_Q*heads/4 weight
// registers per thread, which does not fit 84 (the equal-split budget of
// 640-thread blocks).  `setmaxnreg` re-splits the physical register file
// while the warps run: the TMA warpgroup shrinks to 32 (paged: 64) registers
// — it only pushes descriptors — and every free register is granted to the
// math warpgroups (112 contiguous / 104 paged).  This is a hardware feature
// of Hopper: registers are allocated per warpgroup at launch and the
// `setmaxnreg.{dec,inc}.sync.aligned` pair moves them across *converged*
// warpgroups only.
//
// ---------------------------------------------------------------------------
// CONCEPT 4 — THE TWO RINGS (Q pipeline, global KV pipeline)
// ---------------------------------------------------------------------------
// Full/empty mbarrier pairs per stage; `full.init(1)` (the single TMA
// arrival also posts the expected byte count), `empty.init(#math threads)`.
// Producer waits `empty.wait(phase^1)` — parity ^1 is the standard trick
// that makes a *fresh* ring pass immediately (the "last completed phase" of
// an untouched barrier is 1 by convention), then flips to real waits.
//
//   * Q ring (kNumQStages deep): the producer runs AHEAD by one task — it
//     issues Q(task i+1) *before* the KV blocks of task i (upstream's
//     "pre-advance" schedule: the consumer never waits on an empty pipe).
//   * KV ring: CONTIGUOUS uses a ring that is never reset — the stage index
//     is `(num_total_kv_blocks + j) % kNumKVStages`, with the running block
//     counter shared by producer and consumer, so the KV stages of task i
//     are already in flight when the consumer starts task i.
//   * PAGED: each math warpgroup g is paired with its own TMA producer warp
//     (kv_group) and its own KV pipe in SMEM; a block-table row is fetched
//     once per 8 blocks by the producer warp and rotated through the lanes
//     with `shfl_sync` (kv_block_idx_ptr walk) — one 4-byte load per block.
//
// ---------------------------------------------------------------------------
// CONCEPT 5 — OUTPUT CONTRACTS
// ---------------------------------------------------------------------------
//   * CONTIGUOUS: rows are RAGGED.  Token i only owns the window
//     [k_start_i, k_end_i); it is stored COMPRESSED at column
//     (j - k_start_i), guarded per store (`k_start_i <= j < k_end_i`).
//     The union window of a BLOCK_Q tile is [start, end) aligned down to a
//     multiple of 4 (`start/4*4`) — the KV-scale TMA box must start 16B
//     aligned in gmem (fp32, 4 bytes/element).
//   * PAGED: rows are DENSE [batch*next_n, stride_logits].  Writes are
//     unconditional (upstream: "we have redundant writes here") — the last
//     split of a request and the lanes of a quad may write the same or
//     out-of-range columns; the buffer is sized for it (stride rounded up to
//     a whole split) and the caller slices the valid span afterwards.
// ---------------------------------------------------------------------------
// NVRTC notes: zero-include TU (prelude.h + wgmma.h are auto-prepended);
// `fmaxf`, `__shfl*_sync`, `uint2` are compiler builtins; all template
// parameters are compile-time constants (constexpr use below is safe).
// ===========================================================================

namespace dg {

// ---------------------------------------------------------------------------
// Local helpers (head_dim -> GMMA swizzle layout; mirrors
// mma::sm90::to_swizzle_cute_type<kHeadDim>()).
// The SMEM Q/KV tiles are [rows, head_dim] fp8 with an 8-row x head_dim-byte
// swizzle atom (32/64/128B for head_dim 32/64/128), so:
//   LBO = 0 (one atom along K), SBO = head_dim*8 (8-row atom groups).
// ---------------------------------------------------------------------------
template <uint32_t kHeadDim>
DG_DEVICE GmmaLayoutType mqa_gmma_layout() {
    DG_STATIC_ASSERT(kHeadDim == 32 || kHeadDim == 64 || kHeadDim == 128, "Invalid swizzling");
    return kHeadDim == 32 ? GmmaLayoutType::B32
         : kHeadDim == 64 ? GmmaLayoutType::B64
                          : GmmaLayoutType::B128;
}

// ===========================================================================
// Paged scheduler (port of sched::SM90PagedMQALogitsScheduler).
//
// The metadata kernel gives every SM a half-open range of the GLOBAL task
// sequence as a pair of (q_atom, kv_split) cursors; this struct walks it.
// A "task" = one SPLIT_KV=256 chunk of one "q atom" = next_n tokens of one
// request served by one math warpgroup set (kNumBlocksPerSplit = 4 WGMMA
// blocks of 64).  `fetch_next_task` returns the CURRENT task and advances;
// it reports false once the cursor equals the SM's end entry.
//
// Not ported (dead in the SM90 kernel, which statically rejects varlen and
// asserts next_n in {1,2}): atom_to_token_idx / atom_to_block_table_row /
// get_last_advance — the kernel indexes `q_idx * kNextN` (token) and
// `q_idx` (block-table row) directly, which is exact for next_n in {1,2}
// (kNextNAtom == kNextN and one atom per request).
// ===========================================================================
template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit, uint32_t kNumNextNAtoms>
struct SM90PagedMQALogitsScheduler {
    // 2-token atoms for next_n >= 2 (and varlen); 1-token atoms otherwise.
    static constexpr uint32_t kNextNAtom = (kIsVarlen || kNextN >= 2) ? 2 : 1;

    const uint32_t* context_lens;   // [batch * next_n] (2D: last of each pair)
    const uint32_t* indices;        // varlen only (request id per token)
    uint32_t batch_size;

    uint32_t current_q_atom_idx, current_kv_idx;    // cursor (kv in BLOCK_KV units)
    uint32_t end_q_atom_idx, end_kv_idx;            // next SM's start entry
    uint32_t current_num_kv;       // ceil(ctx_len / BLOCK_KV) of the current atom
    uint32_t current_advance;      // atoms to skip at the next boundary (varlen)

    // Recompute `current_num_kv` (and varlen pairing advance) for `q_atom_idx`.
    DG_DEVICE void refresh_num_kv_and_advance(uint32_t q_atom_idx) {
        if constexpr (kIsVarlen) {
            // varlen: consecutive tokens with equal indices form one atom;
            // a paired atom uses the LONGER context (second token).
            const bool is_paired = (q_atom_idx + 1 < batch_size &&
                                    indices[q_atom_idx] == indices[q_atom_idx + 1]);
            current_advance = is_paired ? 2 : 1;
            const uint32_t ctx_len = is_paired ? context_lens[q_atom_idx + 1]
                                              : context_lens[q_atom_idx];
            current_num_kv = ceil_div_u32(ctx_len, BLOCK_KV);
        } else {
            current_advance = 1;
            const uint32_t q_idx = q_atom_idx / kNumNextNAtoms;
            // 2D context_lens: [batch, next_n] — the last token of the request
            // carries the full context length.
            const uint32_t lens_idx = kIsContextLens2D ? q_idx * kNextN + kNextN - 1 : q_idx;
            current_num_kv = ceil_div_u32(context_lens[lens_idx], BLOCK_KV);
        }
    }

    DG_DEVICE SM90PagedMQALogitsScheduler(uint32_t sm_idx, uint32_t batch_size_,
                                          const uint32_t* context_lens_,
                                          const uint32_t* schedule_meta,
                                          const uint32_t* indices_) {
        context_lens = context_lens_;
        batch_size = batch_size_;
        indices = indices_;
        // One uint2 per SM plus a trailing sentinel entry written by the
        // metadata kernel (sm_idx + 1 reads it for the last SM).
        const uint2 current_pack = ((const uint2*)schedule_meta)[sm_idx];
        const uint2 end_pack = ((const uint2*)schedule_meta)[sm_idx + 1];
        current_q_atom_idx = current_pack.x;
        current_kv_idx = current_pack.y * kNumBlocksPerSplit;
        end_q_atom_idx = end_pack.x;
        end_kv_idx = end_pack.y * kNumBlocksPerSplit;
        // Unconditional: the reversed metadata allocation keeps the start
        // cursor in-bounds even for empty SMs (atom 0 / last request).
        refresh_num_kv_and_advance(current_q_atom_idx);
    }

    // Whether num_kv must be refreshed after advancing to q_atom_idx.
    DG_DEVICE bool should_refresh_num_kv(uint32_t q_atom_idx) const {
        if constexpr (kIsVarlen) {
            return true;   // every atom may have a different context_len
        } else {
            return q_atom_idx % kNumNextNAtoms == 0;   // atom-group boundary
        }
    }

    DG_DEVICE bool exist_q_atom_idx(uint32_t q_atom_idx) const {
        return q_atom_idx < end_q_atom_idx ||
               (q_atom_idx == end_q_atom_idx && 0 < end_kv_idx);
    }

    // Emit the current task; advance the cursor to the next one.
    DG_DEVICE bool fetch_next_task(uint32_t& q_atom_idx, uint32_t& kv_idx, uint32_t& num_kv) {
        q_atom_idx = current_q_atom_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;
        if (current_q_atom_idx == end_q_atom_idx && current_kv_idx == end_kv_idx)
            return false;
        current_kv_idx += kNumBlocksPerSplit;
        if (current_kv_idx >= current_num_kv) {
            current_kv_idx = 0;
            current_q_atom_idx += current_advance;
            if (should_refresh_num_kv(current_q_atom_idx) && exist_q_atom_idx(current_q_atom_idx))
                refresh_num_kv_and_advance(current_q_atom_idx);
        }
        return true;
    }
};

// ===========================================================================
// Metadata kernel (port of sched::sm90_paged_mqa_logits_metadata).
//
// One warp, kAlignedBatchSize (= align(batch, 32)) slots of SMEM.  It runs
// an inclusive prefix scan (ceil_div(ctx_len, SPLIT_KV) per request, warp
// shfl-up + running carry) and then hands each SM its slice of the task
// sequence with a binary search on the prefix array.
//
// The work distribution is "reversed": with total = sum*q + r over kNumSMs
// SMs, the FIRST (kNumSMs - r) SMs get q segments and the last r get q+1 —
// reversed because seg_starts adds (sm_idx - pivot) only ABOVE the pivot.
// Empty SMs (total < kNumSMs) land on atom 0, which keeps the scheduler's
// unconditional `refresh_num_kv_and_advance(start)` in-bounds.
//
// Output: schedule_metadata[sm][0] = q_atom start, [sm][1] = kv_split start
// (kv splits in SPLIT_KV units; the scheduler multiplies by 4 blocks), plus
// a trailing [batch * num_next_n_atoms, 0] sentinel for sm_idx + 1.
// ===========================================================================
template <uint32_t kAlignedBatchSize, uint32_t SPLIT_KV, uint32_t kNumSMs, bool kIsVarlen>
__launch_bounds__(32, 1) __global__
void sm90_paged_mqa_logits_metadata_impl(const uint32_t batch_size, const uint32_t next_n,
                                         const uint32_t is_context_lens_2d,
                                         const uint32_t* context_lens, const uint32_t* indices,
                                         uint32_t* schedule_metadata) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(DG_HOST_EDIT)
    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, "Invalid aligned batch size");
    const uint32_t lane_idx = get_lane_idx();

    // Wait for the previous kernel in the stream (PDL); we read context_lens
    // which its producer may still be writing.
    griddepcontrol_wait();

    extern __shared__ __align__(16) uint32_t smem_buffer[];
    uint32_t* prefix_sum = smem_buffer;                                 // [kAlignedBatchSize]
    uint32_t* atom_token_start = smem_buffer + kAlignedBatchSize;       // [items] (varlen)
    uint32_t* atom_context_len = smem_buffer + kAlignedBatchSize * 2;   // [items] (varlen)
    uint32_t* num_atoms_shared = smem_buffer + kAlignedBatchSize * 3;   // [1] (varlen)
    uint32_t num_items;

    if constexpr (kIsVarlen) {
        // Build varlen atoms: group consecutive tokens with equal indices;
        // a paired atom takes the second token's (longer) context.
        if (lane_idx == 0) {
            uint32_t t = 0, atom_count = 0;
            while (t < batch_size) {
                atom_token_start[atom_count] = t;
                const bool is_paired = (t + 1 < batch_size && indices[t] == indices[t + 1]);
                atom_context_len[atom_count] = is_paired ? context_lens[t + 1] : context_lens[t];
                t += is_paired ? 2 : 1;
                ++atom_count;
            }
            *num_atoms_shared = atom_count;
        }
        __syncwarp();
        num_items = *num_atoms_shared;
    } else {
        num_items = batch_size;
    }

    // Inclusive prefix scan of ceil_div(context_len, SPLIT_KV) over the
    // (aligned) request slots: 32-wide shfl-up chains, carry in `sum`.
    uint32_t sum = 0;
    #pragma unroll 16
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++k) {
        const uint32_t q_idx = k * 32 + lane_idx;
        uint32_t context_len;
        if constexpr (kIsVarlen) {
            context_len = (q_idx < num_items ? atom_context_len[q_idx] : 0);
        } else {
            const uint32_t lens_idx = (is_context_lens_2d ? q_idx * next_n + next_n - 1 : q_idx);
            context_len = (q_idx < batch_size ? context_lens[lens_idx] : 0);
        }
        uint32_t x = ceil_div_u32(context_len, SPLIT_KV);
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffff, x, offset);
            x += (lane_idx >= offset ? y : 0);
        }
        x += sum;
        prefix_sum[k * 32 + lane_idx] = x;
        sum = __shfl_sync(0xffffffff, x, 31);
    }

    // SM work distribution (reversed allocation).
    if constexpr (kIsVarlen) {
        const uint32_t total = sum;
        const uint32_t q = total / kNumSMs, r = total % kNumSMs;
        const uint32_t pivot = kNumSMs - r;
        // NOTE: writes num_sms + 1 entries (the loop covers the sentinel).
        for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 32) {
            const uint32_t seg_starts = sm_idx * q + (sm_idx > pivot ? sm_idx - pivot : 0);
            uint32_t lo = 0, hi = num_items;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (prefix_sum[mid] <= seg_starts) lo = mid + 1; else hi = mid;
            }
            const uint32_t atom_idx = lo;
            const uint32_t kv_split_idx = (atom_idx == 0 ? seg_starts : seg_starts - prefix_sum[atom_idx - 1]);
            const uint32_t q_atom_idx = (atom_idx < num_items ? atom_token_start[atom_idx] : batch_size);
            __syncwarp();
            schedule_metadata[sm_idx * 2] = q_atom_idx;
            schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
        }
    } else {
        // Atoms of 2 tokens for next_n >= 2 (1 otherwise): the segment count
        // total = sum * num_next_n_atoms is distributed over the SMs; each SM
        // start is bisected back onto the (request, atom, split) grid.
        const uint32_t next_n_atom = (next_n >= 2) ? 2 : 1;
        const uint32_t num_next_n_atoms = ceil_div_u32(next_n, next_n_atom);
        const uint32_t total = sum * num_next_n_atoms;
        const uint32_t q = total / kNumSMs, r = total % kNumSMs;
        const uint32_t pivot = kNumSMs - r;
        for (uint32_t sm_idx = lane_idx; sm_idx < kNumSMs; sm_idx += 32) {
            const uint32_t seg_starts = sm_idx * q + (sm_idx > pivot ? sm_idx - pivot : 0);
            uint32_t lo = 0, hi = batch_size;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (prefix_sum[mid] * num_next_n_atoms <= seg_starts) lo = mid + 1; else hi = mid;
            }
            const uint32_t q_idx = lo;
            const uint32_t offset_in_q = (q_idx == 0 ? seg_starts
                                                    : seg_starts - prefix_sum[q_idx - 1] * num_next_n_atoms);
            const uint32_t num_segs_q = (q_idx == 0 ? prefix_sum[0]
                                                    : prefix_sum[q_idx] - prefix_sum[q_idx - 1]);
            const uint32_t atom_idx = num_segs_q > 0 ? offset_in_q / num_segs_q : 0;
            const uint32_t kv_split_idx = num_segs_q > 0 ? offset_in_q % num_segs_q : 0;
            const uint32_t q_atom_idx = q_idx * num_next_n_atoms + atom_idx;
            __syncwarp();
            schedule_metadata[sm_idx * 2] = q_atom_idx;
            schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
        }
        if (lane_idx == 0) {
            // End sentinel: every SM reads entry sm_idx + 1; the last SM
            // lands on (batch * num_next_n_atoms, 0) — one past the final atom.
            schedule_metadata[kNumSMs * 2] = batch_size * num_next_n_atoms;
            schedule_metadata[kNumSMs * 2 + 1] = 0;
        }
    }
#endif
}

// ===========================================================================
// Contiguous-KV kernel (port of sm90_fp8_mqa_logits).
//
// Persistent grid (one CTA per SM, grid-stride over q blocks).  For one
// q block of BLOCK_Q tokens:
//   * the consumer-side union KV window [start, end) comes from the per-token
//     k_start/k_end arrays clamped to seq_len_kv; `start` is rounded DOWN to
//     a multiple of 4 (16B TMA alignment of the fp32 KV-scale box);
//   * num_kv_blocks = ceil(end - start, BLOCK_KV) 256-token blocks; every
//     block j is served by ALL 4 math warpgroups (each owns 64 KV rows) with
//     one WGMMA m64n128k32 chain per 32 lanes of head_dim;
//   * the producer issues Q(task i+1) before the KV of task i — one task of
//     lookahead — so the consumer's full_q wait is always already posted.
// ===========================================================================
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) __global__
void mqa_logits_sm90_impl(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t stride_logits,
                          const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
                          float* logits,
                          const TmaMap tensor_map_q, const TmaMap tensor_map_kv,
                          const TmaMap tensor_map_kv_scales, const TmaMap tensor_map_weights) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(DG_HOST_EDIT)
    // WGMMA m64 n{BLOCK_Q*heads} k32 — the head dim is folded into N.
    constexpr uint32_t WGMMA_M = 64, WGMMA_K = 32;
    constexpr uint32_t kNumAccum = BLOCK_Q * kNumHeads / 2;
    constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;   // regs per token
    DG_STATIC_ASSERT(kNumTMAThreads == 128 && kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
    DG_STATIC_ASSERT(kHeadDim % WGMMA_K == 0, "Invalid head dim");
    DG_STATIC_ASSERT(kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
    DG_STATIC_ASSERT(kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
    DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

    const uint32_t num_q_blocks = ceil_div_u32(seq_len, BLOCK_Q);

    // ---- shared memory layout (upstream PatternVisitor -> plain lambdas) --
    //   [ Q stages | KV stages | weight stages | scale stages | barriers ]
    // Q/KV stages are head_dim*8-byte aligned (8-row swizzle atoms); the
    // un-swizzled weight/scale stages only need 16B (TMA no-swizzle boxes).
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim;          // fp8
    constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim;                    // fp8
    constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    constexpr uint32_t KV_BASE = kNumQStages * SMEM_Q_SIZE_PER_STAGE;
    constexpr uint32_t W_BASE = KV_BASE + kNumKVStages * SMEM_KV_SIZE_PER_STAGE;
    constexpr uint32_t S_BASE = W_BASE + kNumQStages * SMEM_WEIGHT_SIZE_PER_STAGE;
    constexpr uint32_t BAR_BASE = S_BASE + kNumKVStages * SMEM_KV_SCALE_SIZE_PER_STAGE;
    auto smem_q_of = [&](uint32_t i) { return smem_buffer + i * SMEM_Q_SIZE_PER_STAGE; };
    auto smem_kv_of = [&](uint32_t i) { return smem_buffer + KV_BASE + i * SMEM_KV_SIZE_PER_STAGE; };
    auto smem_w_of = [&](uint32_t i) {
        return (float*)(smem_buffer + W_BASE + i * SMEM_WEIGHT_SIZE_PER_STAGE);
    };
    auto smem_s_of = [&](uint32_t i) {
        return (float*)(smem_buffer + S_BASE + i * SMEM_KV_SCALE_SIZE_PER_STAGE);
    };
    Barrier* barrier_ptr = (Barrier*)(smem_buffer + BAR_BASE);
    auto full_q_of = [&](uint32_t i) { return barrier_ptr + i; };
    auto empty_q_of = [&](uint32_t i) { return barrier_ptr + kNumQStages + i; };
    auto full_kv_of = [&](uint32_t i) { return barrier_ptr + kNumQStages * 2 + i; };
    auto empty_kv_of = [&](uint32_t i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; };

    // Prefetch the four descriptors into the TMA unit (one warp, one lane).
    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x && threadIdx.x < kNumMathThreads + 32;
    if (threadIdx.x / 32 == kNumMathThreads / 32 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_q);
        prefetch_tma_map(&tensor_map_kv);
        prefetch_tma_map(&tensor_map_kv_scales);
        prefetch_tma_map(&tensor_map_weights);
    }
    __syncwarp();

    // Barrier init: full from the single TMA producer; empty from all math
    // threads (each math thread arrives once per consumed stage).
    if (is_tma_load_warp && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++i) {
            full_q_of(i)->init(1);
            empty_q_of(i)->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++i) {
            full_kv_of(i)->init(1);
            empty_kv_of(i)->init(kNumMathThreads);
        }
        fence_barrier_init();
    }
    __syncthreads();

    // Register rebalance (Concept 3): TMA warpgroup 32 regs, math 112.
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler: grid-stride over q blocks.
    const uint32_t sm_idx = blockIdx.x;
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    // Per-task schedule.  `q_iter_offset` = 1 gives the NEXT task's stage and
    // phase while keeping the CURRENT task's KV window (the producer's
    // one-task lookahead).  Also fills the per-token k windows (consumer use).
    const auto load_schedule = [&](uint32_t q_iter_offset, uint32_t& q_stage_idx, uint32_t& q_phase,
                                   uint32_t& kv_start, uint32_t& num_kv_blocks) {
        uint32_t start = 0xffffffffu, end = 0;
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const uint32_t q_idx = dg_min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = dg_min(start, dg_min(seq_k_start[i], seq_len_kv));
            end = dg_max(end, dg_min(seq_k_end[i], seq_len_kv));
        }
        kv_start = start / 4 * 4;   // KV-scale TMA box must start 16B-aligned
        num_kv_blocks = ceil_div_u32(end - kv_start, BLOCK_KV);
        q_stage_idx = (q_iter_idx + q_iter_offset) % kNumQStages;
        q_phase = ((q_iter_idx + q_iter_offset) / kNumQStages) & 1;
    };

    // Global KV ring: the stage of block j of the current task is indexed by
    // the RUNNING block counter (never reset across tasks), so task i's KV
    // stages are in flight before the consumer reaches task i.
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](uint32_t kv_block_idx, uint32_t& kv_stage_idx, uint32_t& kv_phase) {
        kv_stage_idx = (num_total_kv_blocks + kv_block_idx) % kNumKVStages;
        kv_phase = ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1;
    };

    // Wait for the preceding kernel in the stream (PDL).
    griddepcontrol_wait();

    if (threadIdx.x >= kNumMathThreads) {
        // ======================= TMA producer ================================
        setmaxnreg_dec<kNumTMARegisters>();
        if (!is_tma_load_warp) return;   // only warp 0 of the TMA warpgroup stays

        // Q tile + weight rows in one barrier: Q box [head_dim, BLOCK_Q*heads]
        // (swizzle head_dim), weights box [heads, BLOCK_Q] (no swizzle).
        const auto issue_tma_q = [&](uint32_t stage_idx, uint32_t block_idx) {
            tma_load_2d(&tensor_map_q, full_q_of(stage_idx), smem_q_of(stage_idx),
                        kEvictNormalHint, 0, block_idx * BLOCK_Q * kNumHeads);
            tma_load_2d(&tensor_map_weights, full_q_of(stage_idx), smem_w_of(stage_idx),
                        kEvictNormalHint, 0, block_idx * BLOCK_Q);
            full_q_of(stage_idx)->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (elect_one_sync() && block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Persistent schedule on one lane.
        if (elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                // NEXT task's Q stage/phase, CURRENT task's KV window.
                uint32_t q_stage_idx, q_phase, kv_start, num_kv_blocks;
                load_schedule(1, q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait consumer release, then issue Q for the NEXT block.
                empty_q_of(q_stage_idx)->wait(q_phase ^ 1);
                const uint32_t next_block_q_idx = block_q_idx + kNumSMs;
                if (next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue the CURRENT task's KV blocks (global ring order).
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
                    uint32_t kv_stage_idx, kv_phase;
                    get_kv_pipeline(kv_block_idx, kv_stage_idx, kv_phase);
                    empty_kv_of(kv_stage_idx)->wait(kv_phase ^ 1);
                    tma_load_2d(&tensor_map_kv, full_kv_of(kv_stage_idx), smem_kv_of(kv_stage_idx),
                                kEvictNormalHint, 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma_load_2d(&tensor_map_kv_scales, full_kv_of(kv_stage_idx), smem_s_of(kv_stage_idx),
                                kEvictNormalHint, kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_of(kv_stage_idx)->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Advance to the next block: {block_q_idx + kNumSMs, iter + 1}.
                block_q_idx += kNumSMs;
                ++q_iter_idx;
            }
        }
    } else {
        // ======================= math warpgroups =============================
        setmaxnreg_inc<kNumMathRegisters>();

        // shfl-broadcast keeps warp_idx uniform (register-friendly).
        const uint32_t thread_idx = threadIdx.x % kNumMathThreads;
        const uint32_t warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const uint32_t warpgroup_idx = warp_idx / 4;
        const uint32_t lane_idx = get_lane_idx();
        float accum[kNumAccum];
        float weights[BLOCK_Q][kNumHeads / 4];

        // WGMMA row mapping: warp w owns KV rows [16w, 16w+16) of the block;
        // each lane covers row (lane/4) and row (lane/4 + 8) — v_0 / v_1.
        const uint32_t warp_offset = warp_idx * 16;
        const uint32_t v_0_offset = lane_idx / 4 + 0;
        const uint32_t v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            uint32_t q_stage_idx, q_phase, kv_start, num_kv_blocks;
            load_schedule(0, q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival.
            full_q_of(q_stage_idx)->wait(q_phase);

            // Read the per-(token, head-pair) weights into registers.
            // j in [0, heads/4); the smem offset (w/2)*8 + (w&1) + (lane%4)*2
            // with w = (j/4)*2 + (j&1) evaluates to head h(j) of token i —
            // exactly the column of accumulator element j (Concept 2).
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++j)
                    weights[i][j] = smem_w_of(q_stage_idx)[i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2];
            }

            // Compute over this task's KV blocks.
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
                uint32_t kv_stage_idx, kv_phase;
                get_kv_pipeline(kv_block_idx, kv_stage_idx, kv_phase);
                full_kv_of(kv_stage_idx)->wait(kv_phase);

                // Per-KV-row scales: fold at the end of the reduce (Concept 1).
                const float scale_kv_0 = smem_s_of(kv_stage_idx)[warp_offset + v_0_offset];
                const float scale_kv_1 = smem_s_of(kv_stage_idx)[warp_offset + v_1_offset];

                // WGMMA chain over head_dim/32 K-slices; slice 0 OVERWRITES
                // the accumulator (scale_d = 0), the rest accumulate.
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA_K; ++k) {
                    // A = KV rows [wg*64, wg*64+64) of this stage; B = the
                    // whole Q tile; both 8-row atoms, SBO = head_dim*8, LBO 0.
                    const GmmaDescriptor desc_a = make_gmma_desc(
                        smem_kv_of(kv_stage_idx) + (warpgroup_idx * WGMMA_M) * kHeadDim + k * WGMMA_K,
                        mqa_gmma_layout<kHeadDim>(), 0, kHeadDim * 8);
                    const GmmaDescriptor desc_b = make_gmma_desc(
                        smem_q_of(q_stage_idx) + k * WGMMA_K,
                        mqa_gmma_layout<kHeadDim>(), 0, kHeadDim * 8);
                    wgmma_f8<BLOCK_Q * kNumHeads>(desc_a.desc_, desc_b.desc_, accum, k != 0 ? 1u : 0u);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                warpgroup_wait_group<0>();

                // Release the KV stage (all 512 math threads).
                empty_kv_of(kv_stage_idx)->arrive();

                // Weighted-ReLU reduce + ragged scatter store.
                const uint32_t kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                    const float* shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](uint32_t j) {
                        return fmaxf(shifted_accum[j], 0.0f) * weights[i][(j / 4) * 2 + (j & 1)];
                    };
                    // Intra-thread: 4 partial head sums (quad lanes share rows).
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; ++k)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;   // row warp_offset + lane/4
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;   // row warp_offset + lane/4 + 8

                    // Inter-thread: butterfly over the 4 lanes of the quad.
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++j) {
                        const int offset = (int)(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // Compressed ragged store (guarded by the token's window).
                    const uint64_t q_offset = (uint64_t)(block_q_idx * BLOCK_Q + i) * stride_logits;
                    if (seq_k_start[i] <= kv_offset + v_0_offset && kv_offset + v_0_offset < seq_k_end[i])
                        logits[q_offset + kv_offset + v_0_offset - seq_k_start[i]] = v_0;
                    if (seq_k_start[i] <= kv_offset + v_1_offset && kv_offset + v_1_offset < seq_k_end[i])
                        logits[q_offset + kv_offset + v_1_offset - seq_k_start[i]] = v_1;
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release the Q stage (all 512 math threads).
            empty_q_of(q_stage_idx)->arrive();

            block_q_idx += kNumSMs;
            ++q_iter_idx;
        }
    }
#endif
}

// ===========================================================================
// Paged-KV kernel (port of sm90_fp8_paged_mqa_logits).
//
// Decode shape: Q is [batch, next_n (1..2), heads, head_dim]; the KV cache
// lives in 64-token pages, one row per request in `block_table`.  SPLIT_KV
// (256) = BLOCK_KV (64) x 4 math warpgroups: a task covers 256 KV columns
// split as 4x64, one slice per (math warpgroup, TMA producer warp) pair —
// "kv_group".  Each group has its OWN KV pipe and barriers in SMEM, so the
// 4 producers run fully in parallel; the Q pipe is owned by kv_group 0.
//
// The producer warp reads its request's block-table row cooperatively (lane
// L gathers block indices kv_idx+g+4L, guarded by num_kv) and rotates the
// row through the warp with shfl as the task advances (kv_block_idx_ptr).
// ===========================================================================
template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) __global__
void mqa_paged_logits_sm90_impl(const uint32_t batch_size,
                                const uint32_t logits_stride, const uint32_t block_table_stride,
                                const uint32_t* context_lens, float* logits,
                                const uint32_t* block_table, const uint32_t* indices,
                                const uint32_t* schedule_meta,
                                const TmaMap tensor_map_q, const TmaMap tensor_map_kv,
                                const TmaMap tensor_map_kv_scales, const TmaMap tensor_map_weights) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(DG_HOST_EDIT)
    DG_STATIC_ASSERT(!kIsVarlen, "Varlen is not supported for SM90 paged MQA logits");
    // The kernel indexes tokens as q_atom*kNextN and block-table rows as
    // q_atom; both are exact only for next_n in {1,2} (one atom per request,
    // kNextNAtom == kNextN) — asserted here to keep the direct math honest.
    DG_STATIC_ASSERT(kNextN == 1 || kNextN == 2, "SM90 paged MQA supports next_n in {1, 2}");

    constexpr uint32_t WGMMA_M = 64, WGMMA_K = 32;
    constexpr uint32_t kNumAccum = kNextN * kNumHeads / 2;
    constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
    constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 && kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups, "Invalid `SPLIT_KV`");
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");
    DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
    DG_STATIC_ASSERT(kHeadDim % WGMMA_K == 0, "Invalid head dim");
    DG_STATIC_ASSERT(kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
    DG_STATIC_ASSERT(kNumAccum / kNumAccumPerReduce == kNextN, "Invalid accumulation");
    DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

    // shfl-broadcast warp indices (uniform across each warp).
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t warpgroup_idx = warp_idx / 4;
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors (first TMA warp, one lane).
    if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_q);
        prefetch_tma_map(&tensor_map_kv);
        prefetch_tma_map(&tensor_map_kv_scales);
        prefetch_tma_map(&tensor_map_weights);
    }
    __syncwarp();

    // ---- shared memory layout: Q pipe, then 4 per-group KV pipes ----------
    // Weight/scale stages are padded up to the 8-row swizzle alignment so
    // every pipe starts kSwizzleAlignment-aligned; the per-pipe barrier area
    // is padded the same way (align(kNumStages*8*2, kSwizzleAlignment)).
    constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim;               // fp8
    constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE =
        align_u32(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    constexpr uint32_t SMEM_Q_PIPE_SIZE =
        kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
        align_u32(kNumQStages * 8 * 2, kSwizzleAlignment);
    constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim;                        // fp8
    constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE =
        align_u32(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    constexpr uint32_t SMEM_KV_PIPE_SIZE =
        kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
        align_u32(kNumKVStages * 8 * 2, kSwizzleAlignment);
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_q_of = [&](uint32_t i) { return smem_buffer + i * SMEM_Q_SIZE_PER_STAGE; };
    auto smem_w_of = [&](uint32_t i) {
        return (float*)(smem_buffer + kNumQStages * SMEM_Q_SIZE_PER_STAGE +
                        i * ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE);
    };
    Barrier* q_barrier_ptr = (Barrier*)(smem_buffer + kNumQStages * SMEM_Q_SIZE_PER_STAGE +
                                        kNumQStages * ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE);
    auto full_q_of = [&](uint32_t i) { return q_barrier_ptr + i; };
    auto empty_q_of = [&](uint32_t i) { return q_barrier_ptr + kNumQStages + i; };

    // Split math warpgroups and TMA warps into kv groups (one producer warp
    // per math warpgroup).  TMA warps 4..7 (kv_group >= 4) exit below.
    const uint32_t kv_group_idx = __shfl_sync(0xffffffff,
        threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);
    const uint32_t smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv_of = [&](uint32_t i) {
        return smem_buffer + smem_offset + i * SMEM_KV_SIZE_PER_STAGE;
    };
    auto smem_s_of = [&](uint32_t i) {
        return (float*)(smem_buffer + smem_offset + kNumKVStages * SMEM_KV_SIZE_PER_STAGE +
                        i * ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE);
    };
    Barrier* kv_barrier_ptr = (Barrier*)(smem_buffer + smem_offset +
                                         kNumKVStages * SMEM_KV_SIZE_PER_STAGE +
                                         kNumKVStages * ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE);
    auto full_kv_of = [&](uint32_t i) { return kv_barrier_ptr + i; };
    auto empty_kv_of = [&](uint32_t i) { return kv_barrier_ptr + kNumKVStages + i; };

    // Barrier init (kv_group 0 owns the Q pipe; every group owns its KV pipe).
    if (warp_idx >= kNumMathThreads / 32 && elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++i) {
                full_q_of(i)->init(1);
                empty_q_of(i)->init(kNumMathThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++i) {
                full_kv_of(i)->init(1);
                empty_kv_of(i)->init(128);   // one math warpgroup consumes it
            }
        }
        fence_barrier_init();
    }
    __syncthreads();

    // Register rebalance: TMA 64, math 104.
    constexpr uint32_t kNumTMARegisters = 64;
    constexpr uint32_t kNumMathRegisters = 104;

    // PDL: the schedule metadata (written by the metadata kernel) must be
    // complete before the scheduler below dereferences it.
    griddepcontrol_wait();

    // Per-SM task ranges from the metadata kernel (kv splits * 4 blocks).
    SM90PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumMathWarpGroups, 1>
        scheduler(blockIdx.x, batch_size, context_lens, schedule_meta, indices);

    const auto get_q_pipeline = [&](uint32_t iter, uint32_t& stage, uint32_t& phase) {
        stage = iter % kNumQStages;
        phase = (iter / kNumQStages) & 1;
    };
    const auto get_kv_pipeline = [&](uint32_t iter, uint32_t& stage, uint32_t& phase) {
        stage = iter % kNumKVStages;
        phase = (iter / kNumKVStages) & 1;
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // ======================= TMA producer warps ==========================
        setmaxnreg_dec<kNumTMARegisters>();
        if (kv_group_idx >= kNumMathWarpGroups) return;

        // Q box [head_dim, next_n*heads] at row q*next_n*heads; weight box
        // [heads, next_n] at row q*next_n.  Issued only by kv_group 0.
        const auto issue_tma_q = [&](uint32_t stage_idx, uint32_t q_atom) {
            if (kv_group_idx == 0 && elect_one_sync()) {
                tma_load_2d(&tensor_map_q, full_q_of(stage_idx), smem_q_of(stage_idx),
                            kEvictNormalHint, 0, q_atom * kNextN * kNumHeads);
                tma_load_2d(&tensor_map_weights, full_q_of(stage_idx), smem_w_of(stage_idx),
                            kEvictNormalHint, 0, q_atom * kNextN);
                full_q_of(stage_idx)->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // `q_idx = batch_size` (one past the last atom) marks "no atom yet".
        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first atom's Q into stage 0.
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv))) {
            issue_tma_q(0, next_q_idx);
            q_iter_idx = 1;
        }

        // Sliding block-table window: each lane gathers block indices for
        // kv positions (kv_idx + g + 4L); the broadcast walks the warp.
        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch Q for atom (a+1) while emitting the first task of
            // atom a — the math can then advance atoms without stalling.
            const bool prefetch_q = (q_idx != next_q_idx && scheduler.exist_q_atom_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            if (prefetch_q) {
                uint32_t q_stage_idx, q_phase;
                get_q_pipeline(q_iter_idx++, q_stage_idx, q_phase);
                empty_q_of(q_stage_idx)->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            // Gather the block-table row for this task (reload at wrap or at
            // the start of a request's split walk).
            if (kv_idx == 0 || kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage =
                    (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv)
                        ? block_table[(uint64_t)q_idx * block_table_stride +
                                      (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)]
                        : 0;
            }
            const uint32_t kv_block_idx =
                __shfl_sync(0xffffffff, kv_block_idx_storage, (uint32_t)(kv_block_idx_ptr++));

            // Wait this group's consumer release, then issue the page.
            uint32_t kv_stage_idx, kv_phase;
            get_kv_pipeline(kv_iter_idx++, kv_stage_idx, kv_phase);
            empty_kv_of(kv_stage_idx)->wait(kv_phase ^ 1);
            if (elect_one_sync()) {
                // 3D map [head_dim, BLOCK_KV, num_pages]: page = kv_block_idx.
                tma_load_3d(&tensor_map_kv, full_kv_of(kv_stage_idx), smem_kv_of(kv_stage_idx),
                            kEvictNormalHint, 0, 0, kv_block_idx);
                tma_load_2d(&tensor_map_kv_scales, full_kv_of(kv_stage_idx), smem_s_of(kv_stage_idx),
                            kEvictNormalHint, 0, kv_block_idx);
                full_kv_of(kv_stage_idx)->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else {
        // ======================= math warpgroups =============================
        setmaxnreg_inc<kNumMathRegisters>();

        float accum[kNumAccum];
        float weights[kNextN][kNumHeads / 4];
        // Warp w in the group owns KV columns [16w, 16w+16) of the 64-wide
        // slice; lanes cover (lane/4) and (lane/4 + 8) — v_0 / v_1.
        const uint32_t sub_warp_offset = (warp_idx % 4) * 16;
        const uint32_t v_0_offset = lane_idx / 4 + 0;
        const uint32_t v_1_offset = lane_idx / 4 + 8;

        // `q_idx = batch_size` marks "no atom yet"; q_stage/phase of the
        // atom currently in the registers.
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx = 0, q_phase = 0;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            // Current atom changes: release the PREVIOUS Q stage (index
            // iter-1), wait the new one, read the weights.
            if (q_idx != next_q_idx) {
                if (q_iter_idx > 0)
                    empty_q_of((q_iter_idx - 1) % kNumQStages)->arrive();
                get_q_pipeline(q_iter_idx++, q_stage_idx, q_phase);
                full_q_of(q_stage_idx)->wait(q_phase);

                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++j)
                        weights[i][j] = smem_w_of(q_stage_idx)[i * kNumHeads +
                            (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2];
                }
            }
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            // Logits base for this task: token rows q_idx*next_n, KV columns
            // (kv_idx + kv_group)*BLOCK_KV + sub_warp_offset.
            const uint64_t kv_offset = (uint64_t)q_idx * kNextN * logits_stride +
                                       ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            uint32_t kv_stage_idx, kv_phase;
            get_kv_pipeline(kv_iter_idx++, kv_stage_idx, kv_phase);
            full_kv_of(kv_stage_idx)->wait(kv_phase);

            // WGMMA over this group's 64-row slice (own pipe -> no wg offset).
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < kHeadDim / WGMMA_K; ++k) {
                const GmmaDescriptor desc_a = make_gmma_desc(
                    smem_kv_of(kv_stage_idx) + k * WGMMA_K,
                    mqa_gmma_layout<kHeadDim>(), 0, kHeadDim * 8);
                const GmmaDescriptor desc_b = make_gmma_desc(
                    smem_q_of(q_stage_idx) + k * WGMMA_K,
                    mqa_gmma_layout<kHeadDim>(), 0, kHeadDim * 8);
                wgmma_f8<kNextN * kNumHeads>(desc_a.desc_, desc_b.desc_, accum, k != 0 ? 1u : 0u);
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);

            // Scales before the wgmma wait: overlap smem loads with MMA.
            const float scale_kv_0 = smem_s_of(kv_stage_idx)[sub_warp_offset + v_0_offset];
            const float scale_kv_1 = smem_s_of(kv_stage_idx)[sub_warp_offset + v_1_offset];

            warpgroup_wait_group<0>();
            empty_kv_of(kv_stage_idx)->arrive();

            // Weighted-ReLU reduce (identical to the contiguous variant,
            // over the kNextN tokens of this atom) + dense store.
            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++i) {
                const float* shifted_accum = accum + i * kNumAccumPerReduce;
                const auto transform = [&](uint32_t j) {
                    return fmaxf(shifted_accum[j], 0.0f) * weights[i][(j / 4) * 2 + (j & 1)];
                };
                float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                #pragma unroll
                for (uint32_t j = 1; j < kNumHeads / 8; ++j) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++k)
                        sum[k] += transform(j * 4 + k);
                }
                float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                float v_1 = (sum[2] + sum[3]) * scale_kv_1;
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++j) {
                    const int offset = (int)(1u << j);
                    v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                    v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                }
                // Dense unconditional store (upstream: redundant by design).
                logits[kv_offset + (uint64_t)i * logits_stride + v_0_offset] = v_0;
                logits[kv_offset + (uint64_t)i * logits_stride + v_1_offset] = v_1;
            }
        }
    }
#endif
}

} // namespace dg
