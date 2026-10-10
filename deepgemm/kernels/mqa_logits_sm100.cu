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

// ---------------------------------------------------------------------------
// Paged-KV scheduler + metadata (port of scheduler/sm100_paged_mqa_logits.cuh)
//
// Paged mode serves *decode* workloads: Q tokens carry `indices` (request id
// per token) and `context_lens`; the KV cache is PAGES of PAGE_KV tokens,
// mapped per request by `block_table[request, page]`. A metadata kernel
// (launched with PDL) balances estimated split cost across SMs; the main
// kernel then walks per-SM (q_token, kv_split) ranges, chunk by chunk, so a
// chunk's KV is loaded once and REUSED by every Q block of the request (the
// `kv_shared_with_prev` full-ring reuse below).
// ---------------------------------------------------------------------------
// CTA-wide exclusive sum of `val` over all threads (warp reduce + smem scan).
// Returns this thread's exclusive prefix; `warp_sums` is [kNumThreads/32]
// scratch. Used by the metadata kernel and the prefix scan below.
DG_DEVICE uint32_t cta_exclusive_sum_32(uint32_t val, uint32_t* warp_sums,
                                        uint32_t num_threads) {
    const uint32_t saved = val;
    #pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, o);   // val == warp total on every lane
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t num_warps = num_threads / 32;
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();
    if (warp == 0) {
        uint32_t run = 0;
        for (uint32_t w = 0; w < num_warps; ++w) {
            const uint32_t t = warp_sums[w];
            warp_sums[w] = run;
            run += t;
        }
    }
    __syncthreads();
    // Exclusive intra-warp prefix of the ORIGINAL per-lane value.
    uint32_t excl_in_warp = 0, v = saved;
    #pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t got = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) { excl_in_warp += got; v += got; }
    }
    return warp_sums[warp] + excl_in_warp;
}

template <uint32_t kNumThreads>
DG_DEVICE void mqa_meta_prefix_scan(uint32_t thread_idx, uint32_t num_items,
                                    uint32_t* values, uint32_t* warp_sums) {
    const uint32_t num_items_per_thread = ceil_div_u32(num_items, kNumThreads) | 1u;
    const uint32_t item_begin_idx = dg_min(thread_idx * num_items_per_thread, num_items);
    const uint32_t item_end_idx = dg_min(item_begin_idx + num_items_per_thread, num_items);
    // Per-thread strided even/odd scan (upstream: two independent chains).
    uint32_t even_sum = 0, odd_sum = 0;
    uint32_t item_idx = item_begin_idx;
    for (; item_idx + 2 <= item_end_idx; item_idx += 2) {
        even_sum += values[item_idx];
        values[item_idx] = even_sum + odd_sum;
        odd_sum += values[item_idx + 1];
        values[item_idx + 1] = odd_sum + even_sum;
    }
    if (item_idx < item_end_idx) {
        even_sum += values[item_idx];
        values[item_idx] = even_sum + odd_sum;
    }
    const uint32_t thread_offset =
        cta_exclusive_sum_32(even_sum + odd_sum, warp_sums, kNumThreads);
    for (item_idx = item_begin_idx; item_idx < item_end_idx; ++item_idx)
        values[item_idx] += thread_offset;
    __syncthreads();
}

// Index of the first prefix > target (or count if none).
DG_DEVICE uint32_t mqa_meta_upper_bound(const uint32_t* prefix, uint32_t count,
                                        uint32_t target) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (prefix[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Geometry of one request.
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV>
struct MQARequestInfo {
    uint32_t q_token_start;
    uint32_t num_q_tokens;
    uint32_t num_q_blocks;
    uint32_t num_kv_splits;
    uint32_t num_kv_pages;

    MQARequestInfo() = default;
    DG_DEVICE MQARequestInfo(uint32_t q_token_start_, uint32_t num_q_tokens_, uint32_t context_len)
        : q_token_start(q_token_start_), num_q_tokens(num_q_tokens_),
          num_q_blocks(ceil_div_u32(num_q_tokens_, BLOCK_Q)),
          num_kv_splits(ceil_div_u32(context_len, SPLIT_KV)),
          num_kv_pages(ceil_div_u32(context_len, PAGE_KV)) {}

    // Group consecutive equal indices; the last token has the longest context.
    DG_DEVICE static MQARequestInfo from_q_token(uint32_t q_token_idx, uint32_t num_q_tokens_total,
                                                 const uint32_t* context_lens, const uint32_t* indices) {
        const uint32_t request_id = indices[q_token_idx];
        uint32_t q_token_end_idx = q_token_idx + 1;
        while (q_token_end_idx < num_q_tokens_total && indices[q_token_end_idx] == request_id)
            ++q_token_end_idx;
        return MQARequestInfo(q_token_idx, q_token_end_idx - q_token_idx,
                               context_lens[q_token_end_idx - 1]);
    }

    // Distribute request tokens evenly across Q blocks.
    DG_DEVICE void get_q_block_span(uint32_t q_block_idx, uint32_t& token_offset,
                                    uint32_t& num_tokens) const {
        const uint32_t base = num_q_tokens / num_q_blocks, remainder = num_q_tokens % num_q_blocks;
        token_offset = q_block_idx * base + dg_min(q_block_idx, remainder);
        num_tokens = base + (q_block_idx < remainder ? 1u : 0u);
    }
};

// Paged scheduler: per-SM (q_token, kv_split) ranges from `schedule_meta`.
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kNumSMs,
          uint32_t kSplitsPerChunk = 8>
struct MQAPagedScheduler {
    static constexpr bool kIsPaged = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t kNumCachedPages = 32;
    DG_STATIC_ASSERT(SPLIT_KV % PAGE_KV == 0 && kNumPagesPerSplit <= kNumCachedPages,
                     "Invalid split shape");
    DG_STATIC_ASSERT(BLOCK_Q * 4 /*kNumHeads bound via impl*/ <= 128 || true, "");

    using Info = MQARequestInfo<BLOCK_Q, SPLIT_KV, PAGE_KV>;

    uint32_t num_q_tokens_total;
    const uint32_t* context_lens;
    const uint32_t* indices;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t end_q_token_idx, end_kv_split_idx;   // next SM's start

    Info current;
    uint32_t current_kv_split_base;
    uint32_t current_q_block_in_request;

    // Emitted-task page lookup state.
    const uint32_t* task_block_table_row;
    uint32_t task_num_kv_pages;
    uint32_t cached_page_base;
    uint32_t cached_page_coord;

    DG_DEVICE MQAPagedScheduler(uint32_t sm_idx, uint32_t num_q_tokens_total_,
                                const uint32_t* context_lens_, const uint32_t* indices_,
                                const uint32_t* block_table_, uint32_t block_table_stride_,
                                const uint32_t* schedule_meta)
        : num_q_tokens_total(num_q_tokens_total_), context_lens(context_lens_),
          indices(indices_), block_table(block_table_),
          block_table_stride(block_table_stride_) {
        // Null meta (contiguous mode): {0,0} starts make has_work() false.
        uint2 start = make_uint2(0u, 0u), end = make_uint2(0u, 0u);
        if (schedule_meta != nullptr) {
            start = ((const uint2*)schedule_meta)[sm_idx];
            end = ((const uint2*)schedule_meta)[sm_idx + 1];
        }
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;
        current.q_token_start = start.x;
        current_kv_split_base = start.y;
        current_q_block_in_request = 0;
        if (has_work())
            current = Info::from_q_token(start.x, num_q_tokens_total, context_lens, indices);
    }

    DG_DEVICE bool has_work() const {
        return current.q_token_start < num_q_tokens_total &&
               (current.q_token_start != end_q_token_idx || current_kv_split_base < end_kv_split_idx);
    }

    DG_DEVICE bool next_q_block(MQALogitsTask& task) { return next_q_block_impl(task, nullptr, nullptr); }
    DG_DEVICE bool next_q_block(MQALogitsTask& task, uint32_t* s, uint32_t* e) {
        return next_q_block_impl(task, s, e);
    }

    DG_DEVICE bool next_q_block_impl(MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        if (!has_work()) return false;
        const uint32_t upper = (current.q_token_start == end_q_token_idx)
            ? end_kv_split_idx : current.num_kv_splits;
        const uint32_t remaining = upper - current_kv_split_base;
        uint32_t q_block_token_offset;
        current.get_q_block_span(current_q_block_in_request, q_block_token_offset, task.num_q_tokens);
        task.q_token_base = current.q_token_start + q_block_token_offset;
        task.kv_token_base = current_kv_split_base * SPLIT_KV;
        task.num_kv_splits = (current.num_q_blocks == 1) ? remaining : dg_min(remaining, kSplitsPerChunk);
        task.kv_shared_with_prev = current_q_block_in_request > 0;
        task_block_table_row = block_table + (uint64_t)current.q_token_start * block_table_stride;
        task_num_kv_pages = current.num_kv_pages;
        cached_page_base = 0xffffffffu;

        if (++current_q_block_in_request == current.num_q_blocks) {
            current_q_block_in_request = 0;
            current_kv_split_base += task.num_kv_splits;
            if (current_kv_split_base >= upper && current.q_token_start != end_q_token_idx) {
                current.q_token_start += current.num_q_tokens;
                current_kv_split_base = 0;
                if (has_work())
                    current = Info::from_q_token(current.q_token_start, num_q_tokens_total,
                                                 context_lens, indices);
            }
        }
        if (seq_k_start != nullptr) {
            // Per-token KV bounds: [0, context_len) of the token's request.
            // Lane L holds the bound for token (q_token_base + L) (L < BLOCK_Q).
            const uint32_t lane_idx = threadIdx.x % 32;
            const uint32_t row_idx = dg_min(task.q_token_base + lane_idx, num_q_tokens_total - 1);
            uint32_t lane_k_end = lane_idx < BLOCK_Q ? context_lens[row_idx] : 0;
            #pragma unroll
            for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++token_idx) {
                seq_k_start[token_idx] = 0;
                seq_k_end[token_idx] = lane_k_end;
                // ptx::exchange: rotate the lane values by one each step.
                uint32_t nxt = __shfl_down_sync(0xffffffff, lane_k_end, 1);
                if (lane_idx == 31) nxt = __shfl_sync(0xffffffff, lane_k_end, 0);
                lane_k_end = nxt;
            }
        }
        return true;
    }

    // Warp-wide page lookup with a sliding cache; OOR pages map to 0.
    DG_DEVICE void get_kv_page_coords(const MQALogitsTask& task, uint32_t kv_split_idx,
                                       int (&page_coords)[kNumPagesPerSplit]) {
        const uint32_t page_base = task.kv_token_base / PAGE_KV + kv_split_idx * kNumPagesPerSplit;
        if (page_base < cached_page_base ||
            page_base + kNumPagesPerSplit > cached_page_base + kNumCachedPages) {
            const uint32_t page_offset = page_base + (threadIdx.x % 32);
            cached_page_base = page_base;
            cached_page_coord = page_offset < task_num_kv_pages
                ? task_block_table_row[page_offset] : 0;
        }
        #pragma unroll
        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++page_idx) {
            // ptx::exchange: swap the lane's cached value with the next lane's,
            // rotating one page coordinate per lane per split.
            uint32_t v = cached_page_coord;
            uint32_t nxt = __shfl_xor_sync(0xffffffff, cached_page_coord, 1);
            if ((threadIdx.x % 2) == 0) cached_page_coord = nxt;
            page_coords[page_idx] = (int)v;
        }
    }
};

// Metadata kernel: balance split cost across SMs at request boundaries.
template <uint32_t SPLIT_KV, uint32_t kNumSMs, uint32_t BLOCK_Q, uint32_t kNumThreads>
DG_GLOBAL __launch_bounds__(kNumThreads, 1)
void mqa_paged_metadata_impl(const uint32_t* context_lens, const uint32_t* indices,
                             uint32_t num_q_tokens_total, uint32_t* schedule_meta) {
    extern __shared__ uint32_t smem_meta[];
    const uint32_t thread_idx = threadIdx.x;
    griddepcontrol_wait();
    if (threadIdx.x / 32 == 0 && elect_one_sync())
        griddepcontrol_launch_dependent();

    uint32_t* request_q_token_start_idx = smem_meta;                       // [num_q_tokens_total]
    uint32_t* request_work_prefix = request_q_token_start_idx + num_q_tokens_total;
    uint32_t* warp_sums = request_work_prefix + num_q_tokens_total;        // [kNumThreads/32]
    uint32_t* num_requests_shared = warp_sums + kNumThreads / 32;

    // Scan changes in indices twice: count request starts, then write them.
    uint32_t num_request_starts = 0;
    const uint32_t num_tokens_per_thread = ceil_div_u32(num_q_tokens_total, kNumThreads * 2) * 2;
    const uint32_t token_begin_idx = dg_min(thread_idx * num_tokens_per_thread, num_q_tokens_total);
    const uint32_t token_end_idx = dg_min(token_begin_idx + num_tokens_per_thread, num_q_tokens_total);
    // Scan changes in `indices` (request starts). Written twice (count pass,
    // then write pass) — no generic lambda: NVRTC JIT mode treats a generic
    // lambda's operator() as a host function.
    #define DG_SCAN_REQUEST_STARTS(ON_START)                                          \
    {                                                                                 \
        uint32_t prev_id = (token_begin_idx > 0 && token_begin_idx < token_end_idx)   \
            ? indices[token_begin_idx - 1] : 0u;                                      \
        uint32_t token_idx_ = token_begin_idx;                                         \
        for (; token_idx_ + 2 <= token_end_idx; token_idx_ += 2) {                    \
            const uint2 ids = *(const uint2*)(indices + token_idx_);                  \
            if (token_idx_ == 0 || ids.x != prev_id) { ON_START(token_idx_); }         \
            if (ids.y != ids.x) { ON_START(token_idx_ + 1); }                         \
            prev_id = ids.y;                                                          \
        }                                                                             \
        if (token_idx_ < token_end_idx &&                                             \
            (token_idx_ == 0 || indices[token_idx_] != prev_id)) {                    \
            ON_START(token_idx_);                                                     \
        }                                                                             \
    }
    DG_SCAN_REQUEST_STARTS(++num_request_starts;)
    uint32_t request_idx = num_request_starts;
    // CTA-wide exclusive sum of request counts.
    {
        uint32_t val = num_request_starts;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, o);
        const uint32_t warp = thread_idx / 32, lane = thread_idx % 32;
        const uint32_t num_warps = kNumThreads / 32;
        if (lane == 0) warp_sums[warp] = val;
        __syncthreads();
        uint32_t warp_off = 0;
        if (warp == 0) {
            uint32_t run = 0;
            for (uint32_t w = 0; w < num_warps; ++w) { uint32_t t = warp_sums[w]; warp_sums[w] = run; run += t; }
        }
        __syncthreads();
        warp_off = warp_sums[warp];
        // Intra-warp exclusive prefix of the original value.
        uint32_t excl = 0, v2 = num_request_starts;
        #pragma unroll
        for (uint32_t o = 1; o < 32; o <<= 1) {
            uint32_t got = __shfl_up_sync(0xffffffff, v2, o);
            if (lane >= o) { excl += got; v2 += got; }
        }
        request_idx = warp_off + excl;
    }
    DG_SCAN_REQUEST_STARTS({
        request_q_token_start_idx[request_idx++] = token_idx_;
    });
    #undef DG_SCAN_REQUEST_STARTS
    if (thread_idx == kNumThreads - 1)
        *num_requests_shared = request_idx;
    __syncthreads();
    const uint32_t num_requests = *num_requests_shared;

    auto get_request_info = [&](uint32_t r, uint32_t& q_start, uint32_t& num_q, uint32_t& ctx) {
        q_start = request_q_token_start_idx[r];
        const uint32_t q_end = r + 1 < num_requests
            ? request_q_token_start_idx[r + 1] : num_q_tokens_total;
        num_q = q_end - q_start;
        ctx = context_lens[q_end - 1];
    };
    // Cost: KV/MMA scales with Q blocks; epilogue with valid tokens.
    auto get_split_cost = [&](uint32_t num_q_tokens) {
        return 2 * BLOCK_Q * ceil_div_u32(num_q_tokens, BLOCK_Q) + num_q_tokens;
    };

    constexpr uint32_t kCostOverflowFlag = 0x80000000u;
    bool cost_overflow = false;
    for (uint32_t r = thread_idx; r < num_requests; r += kNumThreads) {
        uint32_t q_start, num_q, ctx;
        get_request_info(r, q_start, num_q, ctx);
        const uint64_t cost = (uint64_t)ceil_div_u32(ctx, SPLIT_KV) * get_split_cost(num_q);
        request_work_prefix[r] = (uint32_t)cost;
        cost_overflow |= cost > 0xffffffffu;
    }
    __syncthreads();
    if (num_requests > 0)
        mqa_meta_prefix_scan<kNumThreads>(thread_idx, num_requests, request_work_prefix, warp_sums);
    for (uint32_t r = thread_idx + 1; r < num_requests; r += kNumThreads)
        cost_overflow |= request_work_prefix[r] < request_work_prefix[r - 1];
    if (cost_overflow)
        atomicOr(num_requests_shared, kCostOverflowFlag);
    __syncthreads();
    const bool use_token_cost = (*num_requests_shared & kCostOverflowFlag) != 0;
    if (use_token_cost) {
        for (uint32_t r = thread_idx; r < num_requests; r += kNumThreads) {
            uint32_t q_start, num_q, ctx;
            get_request_info(r, q_start, num_q, ctx);
            request_work_prefix[r] = ceil_div_u32(ctx, SPLIT_KV) * num_q;
        }
        __syncthreads();
        mqa_meta_prefix_scan<kNumThreads>(thread_idx, num_requests, request_work_prefix, warp_sums);
    }
    const uint32_t total_cost = num_requests > 0 ? request_work_prefix[num_requests - 1] : 0u;

    // Balance cost across SMs at request split boundaries.
    const uint32_t cost_per_sm = total_cost / kNumSMs;
    const uint32_t cost_remainder = total_cost % kNumSMs;
    for (uint32_t sm_idx = thread_idx; sm_idx <= kNumSMs; sm_idx += kNumThreads) {
        const uint32_t target = sm_idx * cost_per_sm + dg_min(sm_idx, cost_remainder);
        const uint32_t r = mqa_meta_upper_bound(request_work_prefix, num_requests, target);
        uint32_t q_token_idx = num_q_tokens_total, kv_split_idx = 0;
        if (r < num_requests) {
            const uint32_t cost_before = r == 0 ? 0u : request_work_prefix[r - 1];
            uint32_t num_q, ctx;
            get_request_info(r, q_token_idx, num_q, ctx);
            kv_split_idx = (target - cost_before) /
                (use_token_cost ? num_q : get_split_cost(num_q));
        }
        ((uint2*)schedule_meta)[sm_idx] = make_uint2(q_token_idx, kv_split_idx);
    }
}

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
          uint32_t kNumSMs, bool kIsFP4,
          bool kIsPaged = false, uint32_t PAGE_KV = 0>
DG_GLOBAL void __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
mqa_logits_sm100_impl(uint32_t num_q_tokens, uint32_t num_kv_tokens, uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
                      bf16_raw* logits,
                      const __grid_constant__ TmaMap tensor_map_q,
                      const __grid_constant__ TmaMap tensor_map_sf_q,
                      const __grid_constant__ TmaMap tensor_map_kv,
                      const __grid_constant__ TmaMap tensor_map_sf_kv,
                      const __grid_constant__ TmaMap tensor_map_weights,
                      // Paged-only (ignored when !kIsPaged).
                      const uint32_t* context_lens, const uint32_t* indices,
                      const uint32_t* block_table, uint32_t block_table_stride,
                      const uint32_t* schedule_meta) {
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
    // Paged scheduler (unused when !kIsPaged; params may be null then).
    [[maybe_unused]] MQAPagedScheduler<BLOCK_Q, SPLIT_KV, PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>
        make_paged_sched(sm_idx, num_q_tokens, context_lens, indices, block_table,
                         block_table_stride, schedule_meta);
    // Full-ring KV reuse: a task that shares KV with the previous task and
    // whose splits exactly fill the ring re-arms barriers with no TMA.
    const auto reuses_kv_stages = [&](const MQALogitsTask& task) {
        return kIsPaged && task.kv_shared_with_prev && task.num_kv_splits == kNumKVStages;
    };

    // Shared producers ------------------------------------------------------
    // Paged split: gather SF pages (int4 gather4 where possible) and split KV
    // pages between the two producer warps (upstream's `issue_paged_split`).
    const auto issue_paged_kv = [&](uint32_t kv_stage_idx, const int* page_coords,
                                    bool is_sf_producer) {
        constexpr uint32_t kPageKV = (PAGE_KV == 0 ? 64 : PAGE_KV);
        constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / kPageKV;
        // SF producer takes the first half of pages, KV producer the rest.
        constexpr uint32_t kNumGatherPages = kNumPagesPerSplit / 4 * 4;
        constexpr uint32_t kNumSFTMAs = kNumGatherPages / 4 + kNumPagesPerSplit - kNumGatherPages;
        constexpr uint32_t kNumKVPagesFromSFProducer =
            kNumPagesPerSplit > 2 * kNumSFTMAs ? (kNumPagesPerSplit - kNumSFTMAs) / 2 : 0;
        const uint32_t kv_page_begin = is_sf_producer ? 0 : kNumKVPagesFromSFProducer;
        const uint32_t kv_page_end = is_sf_producer ? kNumKVPagesFromSFProducer : kNumPagesPerSplit;

        if (is_sf_producer) {
            #pragma unroll
            for (uint32_t page_idx = 0; page_idx < kNumGatherPages; page_idx += 4) {
                const int4 pc = make_int4(page_coords[page_idx], page_coords[page_idx + 1],
                                          page_coords[page_idx + 2], page_coords[page_idx + 3]);
                tma_gather4_2d(&tensor_map_sf_kv, smem->full_sf_kv_barriers[kv_stage_idx],
                               smem->smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                               0, pc, kEvictNormalHint);
            }
            #pragma unroll
            for (uint32_t page_idx = kNumGatherPages; page_idx < kNumPagesPerSplit; ++page_idx) {
                tma_load_2d(&tensor_map_sf_kv, &smem->full_sf_kv_barriers[kv_stage_idx],
                            smem->smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                            kEvictNormalHint, 0, (uint32_t)page_coords[page_idx]);
            }
            smem->full_sf_kv_barriers[kv_stage_idx].arrive_and_expect_tx(kNumSFKV * sizeof(uint32_t));
        }
        #pragma unroll
        for (uint32_t page_idx = kv_page_begin; page_idx < kv_page_end; ++page_idx) {
            // 3D copy: [head_dim, PAGE_KV, page] box, page row = page_coords.
            tma_load_3d(&tensor_map_kv, &smem->full_kv_barriers[kv_stage_idx],
                        smem->smem_kv[kv_stage_idx] + (uint32_t)page_idx * kPageKV * kNumQKBytesPerToken,
                        kEvictNormalHint, 0, 0, (uint32_t)page_coords[page_idx]);
        }
        if (!is_sf_producer)
            smem->full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(kNumKVBytesPerStage);
    };
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
        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> contig_sched = make_sched;
        [[maybe_unused]] MQAPagedScheduler<BLOCK_Q, SPLIT_KV, PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>
            paged_sched = make_paged_sched;
        auto next_task = [&](MQALogitsTask& t, uint32_t* s_ = nullptr,
                              uint32_t* e_ = nullptr) -> bool {
            if (kIsPaged)
                return s_ ? paged_sched.next_q_block(t, s_, e_) : paged_sched.next_q_block(t);
            return s_ ? contig_sched.next_q_block(t, s_, e_) : contig_sched.next_q_block(t);
        };
        MQALogitsTask task;
        while (next_task(task)) {
            const bool reuse_kv = reuses_kv_stages(task);
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                const StagePhase kv = kv_pipeline.advance();
                if (reuse_kv) {
                    if (elect_one_sync()) {
                        smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                        smem->full_kv_barriers[kv.stage].arrive();   // re-arm, no TMA
                    }
                    __syncwarp();
                    continue;
                }
                if constexpr (kIsPaged) {
                    int page_coords[MQAPagedScheduler<BLOCK_Q, SPLIT_KV,
                        PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>::kNumPagesPerSplit];
                    paged_sched.get_kv_page_coords(task, kv_split_idx, page_coords);
                    if (elect_one_sync()) {
                        smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                        issue_paged_kv(kv.stage, page_coords, /*is_sf_producer=*/false);
                    }
                } else if (elect_one_sync()) {
                    smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                    issue_contiguous_kv(kv.stage, task.kv_token_base + kv_split_idx * SPLIT_KV, false);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        // Q + weights + KV SF producer
        setmaxnreg_dec<kNumSpecializedRegisters>();
        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> contig_sched = make_sched;
        [[maybe_unused]] MQAPagedScheduler<BLOCK_Q, SPLIT_KV, PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>
            paged_sched = make_paged_sched;
        auto next_task = [&](MQALogitsTask& t, uint32_t* s_ = nullptr,
                              uint32_t* e_ = nullptr) -> bool {
            if (kIsPaged)
                return s_ ? paged_sched.next_q_block(t, s_, e_) : paged_sched.next_q_block(t);
            return s_ ? contig_sched.next_q_block(t, s_, e_) : contig_sched.next_q_block(t);
        };
        MQALogitsTask task;
        while (next_task(task)) {
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
                if (reuses_kv_stages(task)) {
                    if (elect_one_sync()) {
                        smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                        smem->full_sf_kv_barriers[kv.stage].arrive();  // re-arm, no TMA
                    }
                    __syncwarp();
                    continue;
                }
                if constexpr (kIsPaged) {
                    constexpr uint32_t kPageKV = (PAGE_KV == 0 ? 64 : PAGE_KV);
                    int page_coords[MQAPagedScheduler<BLOCK_Q, SPLIT_KV, kPageKV, kNumSMs>::kNumPagesPerSplit];
                    paged_sched.get_kv_page_coords(task, kv_split_idx, page_coords);
                    if (elect_one_sync()) {
                        smem->empty_kv_barriers[kv.stage].wait(kv.phase ^ 1);
                        issue_paged_kv(kv.stage, page_coords, true);
                    }
                } else if (elect_one_sync()) {
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

        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> contig_sched = make_sched;
        [[maybe_unused]] MQAPagedScheduler<BLOCK_Q, SPLIT_KV, PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>
            paged_sched = make_paged_sched;
        auto next_task = [&](MQALogitsTask& t, uint32_t* s_ = nullptr,
                              uint32_t* e_ = nullptr) -> bool {
            if (kIsPaged)
                return s_ ? paged_sched.next_q_block(t, s_, e_) : paged_sched.next_q_block(t);
            return s_ ? contig_sched.next_q_block(t, s_, e_) : contig_sched.next_q_block(t);
        };
        MQALogitsTask task;
        while (next_task(task)) {
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
            MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> contig_sched = make_sched;
        [[maybe_unused]] MQAPagedScheduler<BLOCK_Q, SPLIT_KV, PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>
            paged_sched = make_paged_sched;
        auto next_task = [&](MQALogitsTask& t, uint32_t* s_ = nullptr,
                              uint32_t* e_ = nullptr) -> bool {
            if (kIsPaged)
                return s_ ? paged_sched.next_q_block(t, s_, e_) : paged_sched.next_q_block(t);
            return s_ ? contig_sched.next_q_block(t, s_, e_) : contig_sched.next_q_block(t);
        };
            MQALogitsTask task;
            while (next_task(task)) {
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
        MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs> contig_sched = make_sched;
        [[maybe_unused]] MQAPagedScheduler<BLOCK_Q, SPLIT_KV, PAGE_KV == 0 ? 64 : PAGE_KV, kNumSMs>
            paged_sched = make_paged_sched;
        auto next_task = [&](MQALogitsTask& t, uint32_t* s_ = nullptr,
                              uint32_t* e_ = nullptr) -> bool {
            if (kIsPaged)
                return s_ ? paged_sched.next_q_block(t, s_, e_) : paged_sched.next_q_block(t);
            return s_ ? contig_sched.next_q_block(t, s_, e_) : contig_sched.next_q_block(t);
        };
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
        while (next_task(task, seq_k_start, seq_k_end)) {
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

            #pragma unroll(kIsPaged ? 1 : 2)
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++kv_split_idx) {
                const uint32_t kv_offset = task.kv_token_base + kv_split_idx * SPLIT_KV + math_thread_idx;
                const StagePhase tp = tmem_pipeline.advance(kNumMathWarpGroups);
                smem->full_tmem_barriers[tp.stage].wait(tp.phase);
                tcgen05_after_thread_sync();

                const uint32_t num_valid = kIsPaged ? task.num_q_tokens : BLOCK_Q;
                #pragma unroll
                for (uint32_t i = 0; i < num_valid; ++i) {
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
                        if (head_base + chunk == kNumHeads && i == num_valid - 1) {
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
