// DeepGEMM-RS: SM100 sparse MQA logits — DSA top-k indexer scoring.
// Port of upstream `impls/sm100_sparse_mqa_logits.cuh` (scoring kernel) +
// `scheduler/sm100_sparse_mqa_logits_metadata.cuh` (metadata scheduler kernel)
// + `layout/sparse_mqa_logits.cuh` (device-side layout structs).
//
// ===========================================================================
// WHAT IS DSA TOP-K INDEXER SCORING? (the algorithm this file serves)
// ===========================================================================
// DeepSeek Sparse Attention (DSA, DeepSeek-V3.2/V4 class models) replaces the
// "attend over the whole context" of standard MLA with a two-stage scheme:
//
//   Stage 1 (coarse): a tiny MQA "indexer" head scores every KV chunk of the
//     context (this is the *dense* `sm100_mqa_logits` kernel, already ported
//     in mqa_logits_sm100.cu). Its per-chunk scores feed a top-k selection.
//
//   Stage 2 (fine, THIS FILE): only the top-k *selected* 8/16-token blocks
//     per query token are re-scored, at token granularity, producing the
//     logits that rank the selected slots:
//
//       for token i, selected block slot s (0..k_i), token t in block s:
//         logits[i, s * SPARSE_BLOCK_KV + t]
//            = sum_h w[i, h] * relu( <q[i, h, :], kv[block_s, t, :]> )
//
//     The output row is therefore *compressed*: token i's row holds exactly
//     its own k_i selected blocks back-to-back (column = slot * BLOCK + t),
//     not a dense context column. That is what `q_slot_base + q_slot_offset`
//     remaps below: the scheduler hands each KV split the *global* selection
//     slot index of its first block per Q token (`q?_slot_base`), and each
//     block carries a split-local offset, so the math warps can reconstruct
//     the compressed column without a second indirection.
//
// ===========================================================================
// HOW SPARSE DIFFERS FROM (PAGED) MQA — the task model
// ===========================================================================
// * PAGED MQA (mqa_logits_sm100.cu, paged variant) is *dense in coverage*:
//   every token of the request's KV cache is scored; only the *location* is
//   scattered (pages). Its scheduler balances (request, kv_split) cost.
// * SPARSE MQA is *sparse in coverage*: a per-token top-k list
//   (`sparse_kv_block_indices[i, 0..k_i]`, sorted logical block ids) names
//   the only blocks to score. Two extra problems appear:
//   (a) PAIRS of query tokens (BLOCK_Q = 2) are processed together so one
//       tensor-core pass serves both; their selections overlap, so the
//       metadata kernel MERGES + DE-DUPLICATES the two sorted lists
//       (merge-path partition, one pass over the concatenation). A merged
//       block present in both lists is loaded once and used for both.
//   (b) the merged block ids are only *logical*; the physical location is
//       resolved per access mode:
//         - contiguous KV: block b lives at token b * SPARSE_BLOCK_KV in the
//           flat KV tensor. When a whole split of merged blocks happens to be
//           a contiguous run, the copy warps are replaced by ONE bulk TMA
//           ("contiguous split" fast path, flagged in KVSplitHeader).
//         - paged KV: pages of PAGE_KV tokens fuse KV bytes and per-token SF
//           words in one buffer; block b lives at
//           page(block_table[req, b / blocks_per_page]) + in-page offset.
//   Both modes share the rest of the pipeline. The contiguous mode also
//   supports "unaligned ks" (a token's KV span starts mid-block: logical
//   indices are token indices) and tail masking (a selected block may run
//   past the end of the KV tensor -> zfill copies).
//
// ===========================================================================
// TWO-KERNEL STRUCTURE — metadata scheduler, then persistent scorer
// ===========================================================================
// The scoring kernel is PERSISTENT (one CTA per SM, forever) and reads a
// precomputed schedule, because the per-token split counts are data-dependent
// (top-k overlap varies) and computing them inline would serialize the CTA.
// A small metadata kernel (launched with PDL right before) runs first:
//
//   1. per Q-block (2 tokens): load both sorted top-k lists, merge-dedup
//      (merge-path partition over 256 threads), pack each merged block's
//      "present in q0 / present in q1 + slot index" into one 32-bit word
//      (16 bits per token: bit 15 = present, bits 0..14 = slot index,
//      kSparseInvalidSlot = 0xffff = absent),
//   2. cut the merged stream into KV splits of kNumKVBlocksPerSplit blocks,
//      reserve a global split-id range via one atomic (workspace counter),
//      and emit per-split metadata:
//        KVSplitHeader: q_token_base, #blocks | contiguous | partial-tail
//                       flags, q0/q1 slot bases of the split's first block,
//        KVBlockInfo per block: physical block id + packed q0/q1 slot
//                       offsets (absent = kSparseInvalidSlot),
//   3. the LAST CTA (release/acquire counter in the workspace) builds the
//      SM schedule of ScheduleEntry waves:
//        - contiguous mode: split the split-id range evenly per SM, then cut
//          each SM's range at Q-block boundaries (an SM never straddles two
//          Q blocks, so its Q stage stays coherent),
//        - paged mode: every Q block becomes 1..n entries of at most 8
//          splits, then each wave is sorted by split count and rotated so
//          heavy entries spread across SMs (two-wave tail is reversed so the
//          active-SM sets are complementary).
//   The workspace counters self-reset at the end (the last CTA zeroes them),
//   so a workspace buffer is reusable without re-zeroing between launches.
//
// ===========================================================================
// SCORING KERNEL — warp roles (kNumThreads = (math_wgs + 1 + math_wgs/4)*128)
// ===========================================================================
// FP8: SPLIT_KV = 512 = 4 math WGs -> 768 threads. FP4: 640 = 5 WGs -> 896.
//
//   lanes ─────────────────────────────────────────────────────────────────>
//   0            128*WG        128*WG+32   +33      +34   128*WG+128       kNumThreads
//   ├──────────────┬────────────┬──────────┬────────┬─────┬────────────────┤
//   │ math WGs 0..N-1           │ Q+meta   │ SF     │ UMMA│ KV copy warps  │
//   │ drain TMEM, weighted      │ producer │ trans- │     │ (kNumMathWarp-  │
//   │ ReLU, scatter to          │ (TMA Q/  │ pose + │     │  Groups warps, │
//   │ compressed logits         │ W, cp.as-│ UTCCP  │     │  cp.async KV + │
//   │                           │ ync meta)│        │     │  SF gather)    │
//   └──────────────┴────────────┴──────────┴────────┴─────┴────────────────┘
//   math thread t handles KV row t: each math WG owns UMMA_M = 128 rows of
//   the split = SPLIT_KV / kNumMathWarpGroups, i.e. its own TMEM tile.
//
// Register economics: the 3 control warps + KV-copy warps shed registers
// (setmaxnreg.dec to 64); the freed pool is redistributed to the math WGs
// (setmaxnreg.inc, 88 regs FP8 / 72 regs FP4) for accumulator headroom.
//
// ===========================================================================
// PIPELINE / BARRIER TOPOLOGY (who waits for whom, with arrival counts)
// ===========================================================================
// Q ring (kNumQStages = 2): smem q / sf_q / weights / q_blocks.
//   full_q       (1): producer's arrive_and_expect_tx (TMA Q + SF + weights).
//   empty_q (math + 64): math threads (kNumMathThreads) + producer warp (32,
//     after its last metadata issue) + UMMA warp (arrive_count 32, after the
//     last MMA). The SF-transpose warp is covered transitively: it finishes
//     reading sf_q before arriving full_sf_q, which the UMMA warp waits on
//     before its first MMA — so empty_q's flip implies its reads are done.
//   full_sf_q    (1): SF-transpose warp, after UTCCP of sf_q into TMEM.
//
// KV ring (kNumKVStages = 3 FP8 / 5 FP4): smem kv / sf_kv / split headers /
//   kv_block_infos.
//   full_metadata (kNumKVCopyThreads = producer-warp lanes): cp.async of the
//     split header + block infos; cp.async.mbarrier.arrive.noinc fires one
//     arrival per lane when its copies land (count never bumped -> init
//     equals the participating thread count).
//   full_sf_copy  (kNumKVCopyThreads): copy warps, when the SF words of the
//     split are in smem (lets the SF transpose overlap the bulk KV copy).
//   full_kv       (kNumKVCopyThreads + 1): copy warps (data landed) + the
//     SF-transpose warp's +1 after its UTCCPs — so the UMMA warp sees both
//     the KV tile in smem AND its SFs in TMEM when full_kv flips.
//   empty_kv  (kNumMathThreads + 1): every math thread (after reading the
//     split headers / its block info) + the UMMA warp's tcgen05.commit (all
//     MMAs reading smem.kv issued). The copy warps' reads of
//     kv_block_infos are covered transitively via full_kv -> UMMA.
//
// TMEM ring (kNumTmemStages = 5 >= math WGs): accumulators only,
//   UMMA_N = BLOCK_Q * kNumHeads columns per stage.
//   full_tmem (1): UMMA warp's tcgen05.commit.  empty_tmem (128): the one
//   math warpgroup draining that stage, each thread after its last
//   tcgen05.ld.
//
// TMEM column map (kNumTmemCols, power-of-two sized):
//   [ accum: UMMA_N * tmem_stages | SF-Q: 4 * q_stages | SF-KV: 16/20 * kv_stages ]
// The tcgen05.alloc base is asserted-0 upstream; we keep raw column offsets
// (the single allocation of the kernel always lands at column 0).
//
// Termination: the producer publishes a sentinel Q block (num_q_tokens = 0)
// through full_q and a sentinel split (packed_num_kv_blocks = 0) through
// full_metadata; every consumer loop breaks on its sentinel, and the last
// `empty_*` waits of the producer are satisfied by the final real stage.
// ===========================================================================
// ===========================================================================

namespace dg {

// ---------------------------------------------------------------------------
// Layout constants + metadata record types
// (port of layout/sparse_mqa_logits.cuh; byte-identical to upstream — the
//  metadata kernel writes and the scoring kernel reads the same buffer)
// ---------------------------------------------------------------------------
inline constexpr uint32_t kSparseSlotBits = 16;
inline constexpr uint32_t kSparseInvalidSlot = (1u << kSparseSlotBits) - 1;
inline constexpr uint32_t kSparseMaxHeads = 32;
inline constexpr uint32_t kSparseHeadDim = 128;
inline constexpr uint32_t kSparseUTCCPElems = 128;
inline constexpr uint32_t kSparseKVTokensPerTMA = 128;
inline constexpr uint32_t kSparseBlockQ = 2;

// Thread count of the scoring kernel: math warpgroups + 1 control warpgroup
// (Q/metadata, SF-transpose, UMMA warps live in it) + 1/4 extra warpgroup of
// KV copy warps per 4 math warpgroups (i.e. math_wgs copy warps total).
constexpr DG_DEVICE uint32_t sparse_num_threads(const uint32_t num_math_warpgroups) {
    return (num_math_warpgroups + 1 + num_math_warpgroups / 4) * 128;
}

constexpr DG_DEVICE uint32_t sparse_gcd(const uint32_t a, const uint32_t b) {
    return b == 0 ? a : sparse_gcd(b, a % b);
}
constexpr DG_DEVICE uint64_t ceil_div_u64(const uint64_t a, const uint64_t b) { return (a + b - 1) / b; }

struct alignas(16) SparseMetadataHeader {
    uint32_t num_kv_splits;
    uint32_t num_waves;
    uint32_t use_unaligned_ks;
    // Schedule entries are indexed by the SM count used at generation time
    uint32_t num_sms;
};

struct alignas(16) SparseKVSplitHeader {
    static constexpr uint32_t kContiguousFlag = 0x80000000u;
    // A generic-copy split whose last block extends past the end of the KV tensor (non-paged only)
    static constexpr uint32_t kPartialTailFlag = 0x40000000u;
    static constexpr uint32_t kFlagMask = kContiguousFlag | kPartialTailFlag;

    uint32_t q_token_base;
    uint32_t packed_num_kv_blocks;
    uint32_t q0_slot_base;
    uint32_t q1_slot_base;

    DG_DEVICE SparseKVSplitHeader() = default;
    DG_DEVICE SparseKVSplitHeader(const uint32_t q_token_base_, const uint32_t num_kv_blocks,
                                  const bool is_contiguous, const bool has_partial_tail,
                                  const uint32_t q0_slot_base_, const uint32_t q1_slot_base_)
        : q_token_base(q_token_base_),
          packed_num_kv_blocks(num_kv_blocks | (is_contiguous ? kContiguousFlag : 0u) |
                               (has_partial_tail ? kPartialTailFlag : 0u)),
          q0_slot_base(q0_slot_base_), q1_slot_base(q1_slot_base_) {}
};

struct SparseKVBlockInfo {
    uint32_t physical_kv_block_idx;
    // q0 slot offset | (q1 slot offset << 16); kSparseInvalidSlot = absent
    uint32_t packed_slot_offsets;

    DG_DEVICE SparseKVBlockInfo() = default;
    DG_DEVICE SparseKVBlockInfo(const uint32_t physical_kv_block_idx_,
                                const uint32_t q0_slot_offset, const uint32_t q1_slot_offset)
        : physical_kv_block_idx(physical_kv_block_idx_),
          packed_slot_offsets(q0_slot_offset | (q1_slot_offset << kSparseSlotBits)) {}
};

struct alignas(16) SparseScheduleEntry {
    uint32_t kv_split_begin;
    uint32_t kv_split_end;
    uint32_t q_token_base;
    uint32_t num_q_tokens;

    DG_DEVICE SparseScheduleEntry() = default;
    DG_DEVICE SparseScheduleEntry(const uint32_t kv_split_begin_, const uint32_t kv_split_end_,
                                  const uint32_t q_token_base_, const uint32_t num_q_tokens_)
        : kv_split_begin(kv_split_begin_), kv_split_end(kv_split_end_),
          q_token_base(q_token_base_), num_q_tokens(num_q_tokens_) {}
};

template <uint32_t kNumKVBlocksPerSplit>
struct SparseKVSplit {
    SparseKVSplitHeader header;
    SparseKVBlockInfo kv_block_infos[kNumKVBlocksPerSplit];
};

// Q-block descriptor published in the scoring kernel's smem by the producer.
struct alignas(16) SparseQBlock {
    uint32_t q_token_base;
    uint32_t num_q_tokens;
    uint32_t num_kv_splits;

    DG_DEVICE SparseQBlock() = default;
    DG_DEVICE SparseQBlock(const uint32_t q_token_base_, const uint32_t num_q_tokens_,
                           const uint32_t num_kv_splits_)
        : q_token_base(q_token_base_), num_q_tokens(num_q_tokens_), num_kv_splits(num_kv_splits_) {}
};

// ---------------------------------------------------------------------------
// Shared storage of the scoring kernel. `kIsFP4` selects the packed-E2M1
// smem layout (2 elements per byte) vs FP8 (1 per byte); everything else is
// dtype-independent. The Rust launcher mirrors this layout bit-for-bit
// (api_sparse_mqa::SparseMqaConfig::logits_smem_bytes) and the generated
// wrapper static_asserts the two agree.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_Q, uint32_t SPARSE_BLOCK_KV, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages, bool kIsFP4>
struct SparseLogitsSmem {
    static constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
    static constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    // SF-Q padded to a whole UTCCP atom (128 words -> 4 TMEM columns)
    static constexpr uint32_t kNumSFQ = align_u32(BLOCK_Q * kSparseMaxHeads, kSparseUTCCPElems);
    static constexpr uint32_t kSwizzleAlignment = 8 * kSparseHeadDim / kPackFactor;

    DG_STATIC_ASSERT(BLOCK_Q == 2, "Sparse metadata packs exactly two Q slots");
    DG_STATIC_ASSERT(SPARSE_BLOCK_KV == 8 || SPARSE_BLOCK_KV == 16, "Invalid sparse KV block size");
    DG_STATIC_ASSERT(SPLIT_KV % SPARSE_BLOCK_KV == 0 && SPLIT_KV % 128 == 0, "Invalid sparse KV split size");
    DG_STATIC_ASSERT(kNumKVBlocksPerSplit <= kSparseInvalidSlot,
                     "Sparse split-local slots must not use the invalid-slot value");

    alignas(kSwizzleAlignment) uint8_t q[kNumQStages][BLOCK_Q * kSparseMaxHeads * (kSparseHeadDim / kPackFactor)];
    alignas(kSwizzleAlignment) uint8_t kv[kNumKVStages][SPLIT_KV * (kSparseHeadDim / kPackFactor)];
    alignas(128) uint32_t sf_q[kNumQStages][kNumSFQ];
    alignas(128) uint32_t sf_kv[kNumKVStages][SPLIT_KV];
    alignas(128) uint16_t weights[kNumQStages][BLOCK_Q * kSparseMaxHeads];
    alignas(16) SparseKVBlockInfo kv_block_infos[kNumKVStages][kNumKVBlocksPerSplit];
    alignas(16) SparseQBlock q_blocks[kNumQStages];
    alignas(16) SparseKVSplitHeader kv_split_headers[kNumKVStages];

    Barrier full_q_barriers[kNumQStages];
    Barrier full_sf_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_metadata_barriers[kNumKVStages];
    Barrier full_sf_copy_barriers[kNumKVStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

// Metadata-kernel workspace: three monotonically bumped counters, each on
// its own L2 line so the per-CTA atomics never contend.
struct alignas(128) SparseWorkspaceState {
    uint32_t num_kv_splits;
    alignas(128) uint32_t next_q_offset;
    alignas(128) uint32_t num_finished_ctas;
};

// Per-Q-block info left in the workspace for the last CTA's schedule pass.
struct alignas(8) SparseQBlockInfo {
    uint32_t kv_split_base;
    uint32_t num_kv_splits;

    DG_DEVICE SparseQBlockInfo() = default;
    DG_DEVICE SparseQBlockInfo(const uint32_t kv_split_base_, const uint32_t num_kv_splits_)
        : kv_split_base(kv_split_base_), num_kv_splits(num_kv_splits_) {}
};

// ---------------------------------------------------------------------------
// Small local helpers (prelude covers the rest; these are sparse-specific)
// ---------------------------------------------------------------------------

// Ring pipeline cursor (identical to mqa_logits_sm100.cu's local port of
// common/ring_pipeline.cuh): advance() returns the (stage, phase) BEFORE
// stepping; a fresh ring's first advance yields stage 0 / phase 0.
struct SparseStagePhase {
    uint32_t stage, phase;
};

template <uint32_t kNumStages>
struct SparseRingPipeline {
    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    DG_DEVICE SparseStagePhase advance(const uint32_t step = 1) {
        const SparseStagePhase cur = {stage_idx, phase};
        const uint32_t next = stage_idx + step;
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

// ptx::exchange: warp-wide value fetch from lane `src_lane`.
DG_DEVICE uint32_t shfl_u32(const uint32_t v, const uint32_t src_lane) {
    return __shfl_sync(0xffffffff, v, (int)src_lane);
}
DG_DEVICE const uint8_t* shfl_ptr(const uint8_t* p, const uint32_t src_lane) {
    return reinterpret_cast<const uint8_t*>(
        __shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(p), (int)src_lane));
}
DG_DEVICE const uint32_t* shfl_ptr(const uint32_t* p, const uint32_t src_lane) {
    return reinterpret_cast<const uint32_t*>(
        __shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(p), (int)src_lane));
}

// cp.async.cg 16B with the L2::64B prefetch hint (SF words; the prelude's
// cp_async_cg16 carries the 256B hint used for the bulk KV rows).
DG_DEVICE void cp_async_cg16_l2_64b(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared::cta.global.L2::64B [%0], [%1], 16;"
                 :: "r"(cvta_shared_to_u32(smem)), "l"(gmem));
}
// shared-memory unsigned max (schedule wave counter).
DG_DEVICE void atom_max_u32_block(uint32_t* p, const uint32_t v) {
    asm volatile("atom.shared.max.u32 _, [%0], %1;"
                 :: "r"(cvta_shared_to_u32(p)), "r"(v) : "memory");
}

// CTA-wide exclusive sum (port of math::cta_exclusive_sum): returns this
// thread's exclusive prefix; `total` receives the CTA-wide sum.
template <uint32_t kNumThreads>
DG_DEVICE uint32_t sparse_cta_exclusive_sum(const uint32_t value, uint32_t* warp_sums,
                                            uint32_t& total) {
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    const uint32_t lane_idx = get_lane_idx();
    const uint32_t warp_idx = get_warp_idx();

    // Inclusive warp scan of the per-lane value.
    uint32_t lane_sum = value;
    #pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t got = __shfl_up_sync(0xffffffff, lane_sum, (int)o);
        if (lane_idx >= o) lane_sum += got;
    }
    if (lane_idx == 31)
        warp_sums[warp_idx] = lane_sum;
    __syncthreads();

    // Lane `w` now scans warp w's total; broadcast pieces back.
    const uint32_t warp_total = lane_idx < kNumWarps ? warp_sums[lane_idx] : 0u;
    uint32_t warp_sum = warp_total;
    #pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t got = __shfl_up_sync(0xffffffff, warp_sum, (int)o);
        if (lane_idx >= o) warp_sum += got;
    }
    total = __shfl_sync(0xffffffff, warp_sum, (int)(kNumWarps - 1));
    return lane_sum - value + __shfl_sync(0xffffffff, warp_sum - warp_total, (int)warp_idx);
}

// 16B-chunk swizzle for the generic (cp.async) KV copy path, matching the
// TMA swizzle atom the contiguous fast path writes: within one swizzle atom
// (kSwizzleMode bytes), the 16B chunk column is XOR-ed with the row group.
template <uint32_t kSwizzleMode>
DG_DEVICE uint32_t get_swizzled_kv_chunk_idx(const uint32_t logical_chunk_idx) {
    constexpr uint32_t kMask = kSwizzleMode / 16 - 1;
    return (logical_chunk_idx & ~kMask) | ((logical_chunk_idx & kMask) ^ ((logical_chunk_idx >> 3u) & kMask));
}

// ---------------------------------------------------------------------------
// KV accessors — how a *logical* selected block becomes a physical address.
// Both expose the same 3-method protocol used by the copy warps:
//   resolve_kv_block(physical_idx) -> per-lane ref (shuffled between lanes)
//   get_kv_block(ref, src_lane)    -> row-0 pointer of that block
//   get_sf_kv_block(ref, src_lane) -> SF words of that block
// ---------------------------------------------------------------------------
template <uint32_t SPARSE_BLOCK_KV, bool kIsFP4>
struct ContiguousSparseKVAccessor {
    static constexpr bool kSupportsContiguousTMA = true;  // bulk-TMA fast path
    static constexpr bool kNeedsTailMask = true;          // last block may run past the end
    static constexpr uint32_t kNumQKBytesPerToken = kSparseHeadDim / (kIsFP4 ? 2u : 1u);
    using KVBlockRef = uint32_t;  // token index of the block's first token

    const uint8_t* kv;
    const uint32_t* sf_kv;
    const uint32_t num_kv_tokens;
    const TmaMap* tensor_map_kv;
    const TmaMap* tensor_map_sf_kv;

    DG_DEVICE ContiguousSparseKVAccessor(const uint8_t* kv_, const uint32_t* sf_kv_,
                                         const uint32_t num_kv_tokens_,
                                         const TmaMap* tensor_map_kv_, const TmaMap* tensor_map_sf_kv_)
        : kv(kv_), sf_kv(sf_kv_), num_kv_tokens(num_kv_tokens_),
          tensor_map_kv(tensor_map_kv_), tensor_map_sf_kv(tensor_map_sf_kv_) {}

    DG_DEVICE void prefetch_tma_descriptors() const {
        prefetch_tma_map(tensor_map_kv);
        prefetch_tma_map(tensor_map_sf_kv);
    }

    // Fast path: a fully contiguous split is one bulk TMA pair instead of
    // kNumKVBlocksPerSplit gathered copies. SF goes first so its transpose
    // overlaps the bulk KV transfer.
    template <uint32_t SPLIT_KV>
    DG_DEVICE void copy_contiguous_kv_split(Barrier& kv_barrier, Barrier& sf_barrier,
                                            void* smem_kv, void* smem_sf_kv,
                                            const uint32_t kv_token_start) const {
        DG_STATIC_ASSERT(SPLIT_KV % kSparseKVTokensPerTMA == 0, "KV split must contain whole TMA tiles");
        #pragma unroll
        for (uint32_t token_offset = 0; token_offset < SPLIT_KV; token_offset += kSparseKVTokensPerTMA) {
            tma_load_2d(tensor_map_sf_kv, &sf_barrier,
                        static_cast<uint32_t*>(smem_sf_kv) + token_offset,
                        kEvictNormalHint, kv_token_start + token_offset, 0);
        }
        sf_barrier.arrive_and_expect_tx(SPLIT_KV * sizeof(uint32_t));
        #pragma unroll
        for (uint32_t token_offset = 0; token_offset < SPLIT_KV; token_offset += kSparseKVTokensPerTMA) {
            tma_load_2d(tensor_map_kv, &kv_barrier,
                        static_cast<uint8_t*>(smem_kv) + token_offset * kNumQKBytesPerToken,
                        kEvictNormalHint, 0, kv_token_start + token_offset);
        }
        kv_barrier.arrive_and_expect_tx(SPLIT_KV * kNumQKBytesPerToken);
    }

    DG_DEVICE KVBlockRef resolve_kv_block(const uint32_t physical_kv_block_idx) const {
        return physical_kv_block_idx;
    }
    DG_DEVICE const uint8_t* get_kv_block(const KVBlockRef kv_block_ref,
                                          const uint32_t src_lane_idx) const {
        const uint32_t kv_token_start = shfl_u32(kv_block_ref, src_lane_idx);
        return kv + static_cast<uint64_t>(kv_token_start) * kNumQKBytesPerToken;
    }
    DG_DEVICE const uint32_t* get_sf_kv_block(const KVBlockRef kv_block_ref,
                                              const uint32_t src_lane_idx) const {
        return sf_kv + shfl_u32(kv_block_ref, src_lane_idx);
    }
    // Number of tokens of the block that lie inside the KV tensor.
    DG_DEVICE uint32_t get_num_valid_tokens(const KVBlockRef kv_block_ref) const {
        return dg_min(SPARSE_BLOCK_KV, num_kv_tokens - dg_min(kv_block_ref, num_kv_tokens));
    }
};

// Paged mode: each page fuses [PAGE_KV tokens x bytes/token KV][PAGE_KV SF
// words]; pages are mapped per request by `block_table`. No bulk TMA (the
// blocks of a split land anywhere), no tail masking (pages are dense).
template <uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV, bool kIsFP4>
struct PagedSparseKVAccessor {
    static constexpr bool kSupportsContiguousTMA = false;
    static constexpr bool kNeedsTailMask = false;
    static constexpr uint32_t kNumQKBytesPerToken = kSparseHeadDim / (kIsFP4 ? 2u : 1u);

    struct KVBlockRef {
        const uint8_t* kv;
        const uint32_t* sf_kv;
        DG_DEVICE KVBlockRef(const uint8_t* kv_, const uint32_t* sf_kv_) : kv(kv_), sf_kv(sf_kv_) {}
    };

    const uint8_t* fused_kv_cache;
    const uint32_t kv_page_stride_bytes;

    DG_DEVICE PagedSparseKVAccessor(const uint8_t* fused_kv_cache_, const uint32_t kv_page_stride_bytes_)
        : fused_kv_cache(fused_kv_cache_), kv_page_stride_bytes(kv_page_stride_bytes_) {}

    DG_DEVICE KVBlockRef resolve_kv_block(const uint32_t physical_kv_block_idx) const {
        constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
        const uint32_t physical_page_idx = physical_kv_block_idx / kNumKVBlocksPerPage;
        const uint32_t kv_block_idx_in_page = physical_kv_block_idx % kNumKVBlocksPerPage;
        const uint8_t* page = fused_kv_cache + static_cast<uint64_t>(physical_page_idx) * kv_page_stride_bytes;
        const uint32_t token_idx_in_page = kv_block_idx_in_page * SPARSE_BLOCK_KV;
        return KVBlockRef(
            page + token_idx_in_page * kNumQKBytesPerToken,
            reinterpret_cast<const uint32_t*>(page + PAGE_KV * kNumQKBytesPerToken) + token_idx_in_page);
    }
    DG_DEVICE const uint8_t* get_kv_block(const KVBlockRef& kv_block_ref,
                                          const uint32_t src_lane_idx) const {
        return shfl_ptr(kv_block_ref.kv, src_lane_idx);
    }
    DG_DEVICE const uint32_t* get_sf_kv_block(const KVBlockRef& kv_block_ref,
                                              const uint32_t src_lane_idx) const {
        return shfl_ptr(kv_block_ref.sf_kv, src_lane_idx);
    }
};

// ===========================================================================
// METADATA SCHEDULER KERNEL
// (port of scheduler/sm100_sparse_mqa_logits_metadata.cuh)
// ===========================================================================

// Shared storage of the metadata kernel (mirror of upstream's SharedStorage;
// the Rust launcher computes the same rounded size).
template <uint32_t BLOCK_Q, uint32_t NUM_MAX_SPARSE_BLOCKS,
          uint32_t kNumKVBlocksPerSplit, uint32_t kNumKVSplitsPerEntry, uint32_t kNumThreads>
struct SparseMetaSmem {
    static constexpr uint32_t kNumMaxMergedKVBlocks = BLOCK_Q * NUM_MAX_SPARSE_BLOCKS;
    static constexpr uint32_t kNumMaxKVSplits = (kNumMaxMergedKVBlocks + kNumKVBlocksPerSplit - 1) / kNumKVBlocksPerSplit;
    static constexpr uint32_t kNumWarps = kNumThreads / 32;

    uint32_t logical_kv_block_indices[BLOCK_Q][NUM_MAX_SPARSE_BLOCKS];
    uint32_t packed_slots_by_merged_kv_block[kNumMaxMergedKVBlocks];
    uint32_t packed_slot_bases_by_kv_split[kNumMaxKVSplits];
    uint32_t warp_sums[kNumWarps];
    uint32_t num_waves;
    uint32_t kv_split_histograms[kNumWarps][kNumKVSplitsPerEntry + 1];
    struct alignas(16) {
        uint32_t q_token_base;
        uint32_t num_q_tokens;
        uint32_t num_kv_blocks[BLOCK_Q];
        uint32_t kv_split_base;
    } q_block;
    uint32_t is_last_cta;
};

// Contiguous mode: divide the split-id range evenly per SM, then cut each
// SM's range at Q-block boundaries (an entry never straddles Q blocks).
template <uint32_t BLOCK_Q, uint32_t kNumKVBlocksPerSplit, uint32_t kNumSMs, typename smem_t>
DG_DEVICE uint32_t build_contiguous_schedule(
    SparseScheduleEntry* schedule_entries, const SparseKVSplit<kNumKVBlocksPerSplit>* kv_splits,
    const SparseQBlockInfo* q_block_infos, const uint32_t num_q_tokens,
    const uint32_t total_kv_splits, smem_t& smem) {
    if (threadIdx.x == 0)
        smem.num_waves = 1;
    __syncthreads();

    uint32_t num_sm_entries = 0;
    if (threadIdx.x < kNumSMs) {
        uint32_t kv_split_idx = (uint32_t)ceil_div_u64(
            static_cast<uint64_t>(total_kv_splits) * threadIdx.x, kNumSMs);
        const uint32_t kv_split_end = (uint32_t)ceil_div_u64(
            static_cast<uint64_t>(total_kv_splits) * (threadIdx.x + 1), kNumSMs);
        while (kv_split_idx < kv_split_end) {
            const uint32_t q_token_base = kv_splits[kv_split_idx].header.q_token_base;
            const SparseQBlockInfo q_block_info = q_block_infos[q_token_base];
            const uint32_t entry_kv_split_end =
                dg_min(kv_split_end, q_block_info.kv_split_base + q_block_info.num_kv_splits);
            schedule_entries[num_sm_entries * kNumSMs + threadIdx.x] = SparseScheduleEntry(
                kv_split_idx, entry_kv_split_end, q_token_base, dg_min(BLOCK_Q, num_q_tokens - q_token_base));
            kv_split_idx = entry_kv_split_end;
            ++num_sm_entries;
        }
        atom_max_u32_block(&smem.num_waves, num_sm_entries);
    }
    __syncthreads();

    const uint32_t num_waves = smem.num_waves;
    if (threadIdx.x < kNumSMs) {
        for (uint32_t wave_idx = num_sm_entries; wave_idx < num_waves; ++wave_idx)
            schedule_entries[wave_idx * kNumSMs + threadIdx.x] = SparseScheduleEntry(0, 0, 0, 0);
    }
    return num_waves;
}

// Sort each wave by split count (counting rank over a shared histogram) and
// rotate heavy entries across SMs; reverse the 2-wave tail so the active SM
// sets of the two waves are complementary.
template <uint32_t kNumSMs, uint32_t kNumKVSplitsPerEntry, typename smem_t>
DG_DEVICE void balance_wave_entries(SparseScheduleEntry* schedule_entries, const uint32_t num_waves,
                                    smem_t& smem) {
    __syncthreads();
    if (num_waves == 1)
        return;
    const uint32_t lane_idx = get_lane_idx();
    const uint32_t warp_idx = get_warp_idx();

    for (uint32_t wave_idx = warp_idx; wave_idx < num_waves; wave_idx += smem_t::kNumWarps) {
        constexpr uint32_t kNumEntriesPerLane = (kNumSMs + 31) / 32;
        uint32_t* histogram = smem.kv_split_histograms[warp_idx];
        SparseScheduleEntry* wave_entries = schedule_entries + wave_idx * kNumSMs;
        SparseScheduleEntry entries[kNumEntriesPerLane];
        uint32_t ranks[kNumEntriesPerLane];

        #pragma unroll
        for (uint32_t sm_idx = lane_idx, entry_idx = 0; sm_idx < kNumSMs; sm_idx += 32, ++entry_idx)
            entries[entry_idx] = wave_entries[sm_idx];
        for (uint32_t num_kv_splits = lane_idx; num_kv_splits <= kNumKVSplitsPerEntry; num_kv_splits += 32)
            histogram[num_kv_splits] = 0;
        __syncwarp();

        // Rank each entry among those with the same split count.
        #pragma unroll
        for (uint32_t sm_idx = lane_idx, entry_idx = 0; sm_idx < kNumSMs; sm_idx += 32, ++entry_idx)
            ranks[entry_idx] = atom_add_u32_block(
                histogram + (entries[entry_idx].kv_split_end - entries[entry_idx].kv_split_begin), 1u);
        __syncwarp();

        // Exclusive-prefix the histogram into rank bases.
        if (elect_one_sync()) {
            uint32_t rank_begin = 0;
            for (uint32_t num_kv_splits = 0; num_kv_splits <= kNumKVSplitsPerEntry; ++num_kv_splits) {
                const uint32_t frequency = histogram[num_kv_splits];
                histogram[num_kv_splits] = rank_begin;
                rank_begin += frequency;
            }
        }
        __syncwarp();

        #pragma unroll
        for (uint32_t sm_idx = lane_idx, entry_idx = 0; sm_idx < kNumSMs; sm_idx += 32, ++entry_idx) {
            const uint32_t rank = histogram[entries[entry_idx].kv_split_end - entries[entry_idx].kv_split_begin] +
                                  ranks[entry_idx];
            // Reverse the two-wave tail to make active SM sets complementary.
            const uint32_t dst_sm_idx = (num_waves == 2 && wave_idx == 1)
                ? kNumSMs - 1 - rank
                : (rank + kNumSMs - wave_idx * kNumSMs / num_waves) % kNumSMs;
            wave_entries[dst_sm_idx] = entries[entry_idx];
        }
    }
}

// Paged mode: each Q block becomes 1..n bounded entries (at most
// kNumKVSplitsPerEntry splits each, remainder spread over the first ones),
// then each wave is balanced.
template <uint32_t BLOCK_Q, uint32_t kNumKVSplitsPerEntry,
          uint32_t kNumSMs, uint32_t kNumThreads, typename smem_t>
DG_DEVICE uint32_t build_paged_schedule(SparseScheduleEntry* schedule_entries,
                                        const SparseQBlockInfo* q_block_infos,
                                        const uint32_t num_q_tokens, const uint32_t* indices,
                                        smem_t& smem) {
    uint32_t num_entries = 0;
    for (uint32_t q_token_begin = 0; q_token_begin < num_q_tokens; q_token_begin += kNumThreads) {
        const uint32_t q_token_idx = q_token_begin + threadIdx.x;
        const SparseQBlockInfo q_block_info =
            q_token_idx < num_q_tokens ? q_block_infos[q_token_idx] : SparseQBlockInfo(0, 0);
        const uint32_t num_kv_splits = q_block_info.num_kv_splits;
        const uint32_t num_q_entries = ceil_div_u32(num_kv_splits, kNumKVSplitsPerEntry);
        uint32_t num_batch_entries;
        const uint32_t entry_begin = num_entries + sparse_cta_exclusive_sum<kNumThreads>(
            num_q_entries, smem.warp_sums, num_batch_entries);

        const uint32_t kv_splits_per_entry = num_q_entries == 0 ? 0 : num_kv_splits / num_q_entries;
        const uint32_t num_larger_entries = num_q_entries == 0 ? 0 : num_kv_splits % num_q_entries;
        const uint32_t num_q_block_tokens =
            q_token_idx + 1 < num_q_tokens && indices[q_token_idx + 1] == indices[q_token_idx] ? BLOCK_Q : 1;
        uint32_t kv_split_begin = q_block_info.kv_split_base;
        for (uint32_t q_entry_idx = 0; q_entry_idx < num_q_entries; ++q_entry_idx) {
            const uint32_t kv_split_end =
                kv_split_begin + kv_splits_per_entry + (q_entry_idx < num_larger_entries ? 1u : 0u);
            schedule_entries[entry_begin + q_entry_idx] =
                SparseScheduleEntry(kv_split_begin, kv_split_end, q_token_idx, num_q_block_tokens);
            kv_split_begin = kv_split_end;
        }
        num_entries += num_batch_entries;
        __syncthreads();
    }

    const uint32_t num_waves = dg_max(1u, ceil_div_u32(num_entries, kNumSMs));
    for (uint32_t entry_idx = num_entries + threadIdx.x; entry_idx < num_waves * kNumSMs;
         entry_idx += kNumThreads)
        schedule_entries[entry_idx] = SparseScheduleEntry(0, 0, 0, 0);

    balance_wave_entries<kNumSMs, kNumKVSplitsPerEntry>(schedule_entries, num_waves, smem);
    return num_waves;
}

// The metadata kernel itself. Grid = min(#q_blocks, 4 * num_sms) CTAs of 256
// threads; every CTA loops over Q blocks (contiguous: grid-stride pairs;
// paged: atomic work stealing of single tokens so blocks stay within one
// request), and the LAST CTA builds the SM schedule.
template <bool kIsPaged, bool kUseUnalignedKs, uint32_t BLOCK_Q,
          uint32_t SPLIT_KV, uint32_t SPARSE_BLOCK_KV,
          uint32_t NUM_MAX_SPARSE_BLOCKS, uint32_t PAGE_KV,
          uint32_t kNumKVSplitsPerEntry, uint32_t kNumSMs, uint32_t kNumThreads>
DG_GLOBAL void __launch_bounds__(kNumThreads, 4)
sparse_mqa_metadata_impl(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                         const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end,
                         const uint32_t* context_lens, const uint32_t* block_table,
                         const uint32_t block_table_stride,
                         const uint32_t* indices,
                         const uint32_t* sparse_kv_block_indices, uint8_t* metadata,
                         uint8_t* workspace) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    using smem_t = SparseMetaSmem<BLOCK_Q, NUM_MAX_SPARSE_BLOCKS, kNumKVBlocksPerSplit,
                                  kNumKVSplitsPerEntry, kNumThreads>;

    // Template checks (upstream parity)
    DG_STATIC_ASSERT(BLOCK_Q == 2 && kNumThreads == 256 && kNumSMs <= kNumThreads, "Unsupported metadata shape");
    DG_STATIC_ASSERT(SPLIT_KV % SPARSE_BLOCK_KV == 0 && SPLIT_KV % 128 == 0, "Invalid sparse split shape");
    DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS % 4 == 0, "Sparse index rows must be 16-byte aligned");
    DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS <= 4096, "Too many sparse KV blocks");
    DG_STATIC_ASSERT(kNumKVBlocksPerSplit <= kSparseInvalidSlot, "Sparse slot offset overflow");
    DG_STATIC_ASSERT(kNumKVSplitsPerEntry > 0, "Invalid number of KV splits per entry");
    DG_STATIC_ASSERT(!kIsPaged || PAGE_KV % SPARSE_BLOCK_KV == 0, "Invalid page shape");
    DG_STATIC_ASSERT(!kIsPaged || !kUseUnalignedKs, "Paged sparse MQA does not use ks");

    const uint32_t thread_idx = threadIdx.x;
    griddepcontrol_wait();

    auto workspace_state = reinterpret_cast<SparseWorkspaceState*>(workspace);
    auto q_block_infos = reinterpret_cast<SparseQBlockInfo*>(workspace + sizeof(SparseWorkspaceState));
    auto kv_splits = reinterpret_cast<SparseKVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(SparseMetadataHeader));

    // Shared memory
    extern __shared__ __align__(16) uint8_t storage[];
    auto& smem = *reinterpret_cast<smem_t*>(storage);

    // ---- Pass 1: build per-Q-block merged KV-block metadata -----------------
    uint32_t q_token_idx = blockIdx.x * (kIsPaged ? 1u : BLOCK_Q);
    while (true) {
        if (thread_idx == 0) {
            smem.q_block.num_q_tokens = 0;
            while (q_token_idx < num_q_tokens) {
                uint32_t num_q_block_tokens = dg_min(BLOCK_Q, num_q_tokens - q_token_idx);
                if constexpr (kIsPaged) {
                    // Keep Q blocks within one request
                    const uint32_t request_idx = indices[q_token_idx];
                    // Requests contain few Q tokens, so use a linear scan
                    uint32_t request_q_token_base = q_token_idx;
                    while (request_q_token_base > 0 && indices[request_q_token_base - 1] == request_idx)
                        --request_q_token_base;
                    if ((q_token_idx - request_q_token_base) % BLOCK_Q != 0) {
                        // Misaligned inside the request: steal a fresh token
                        q_token_idx = gridDim.x + atom_add_u32(&workspace_state->next_q_offset, 1u);
                        continue;
                    }
                    num_q_block_tokens = q_token_idx + 1 < num_q_tokens &&
                        indices[q_token_idx + 1] == request_idx ? BLOCK_Q : 1;
                }
                const auto get_num_kv_blocks = [&](const uint32_t q_idx) {
                    const uint32_t kv_begin = kIsPaged ? 0u : dg_min(cu_seq_len_k_start[q_idx], num_kv_tokens);
                    const uint32_t kv_end = kIsPaged ? context_lens[q_idx]
                        : dg_max(kv_begin, dg_min(cu_seq_len_k_end[q_idx], num_kv_tokens));
                    return dg_min(NUM_MAX_SPARSE_BLOCKS, ceil_div_u32(kv_end - kv_begin, SPARSE_BLOCK_KV));
                };
                smem.q_block.q_token_base = q_token_idx;
                smem.q_block.num_q_tokens = num_q_block_tokens;
                smem.q_block.num_kv_blocks[0] = get_num_kv_blocks(q_token_idx);
                smem.q_block.num_kv_blocks[1] = num_q_block_tokens == BLOCK_Q ? get_num_kv_blocks(q_token_idx + 1) : 0;
                break;
            }
        }
        __syncthreads();

        const uint32_t num_q_block_tokens = smem.q_block.num_q_tokens;
        if (num_q_block_tokens == 0)
            break;
        const uint32_t q_token_base = smem.q_block.q_token_base;
        const uint32_t num_kv_blocks_in_q0 = smem.q_block.num_kv_blocks[0];
        const uint32_t num_kv_blocks_in_q1 = smem.q_block.num_kv_blocks[1];

        // Load the two sorted top-k index lists into smem. Unaligned-ks
        // stores logical *token* indices (block * 8/16 + span offset); the
        // aligned path copies raw block indices with cp.async.
        #pragma unroll
        for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++q_token_offset) {
            const uint32_t* src = sparse_kv_block_indices +
                static_cast<uint64_t>(q_token_base + q_token_offset) * NUM_MAX_SPARSE_BLOCKS;
            uint32_t* dst = smem.logical_kv_block_indices[q_token_offset];
            const uint32_t num_kv_blocks = q_token_offset == 0 ? num_kv_blocks_in_q0 : num_kv_blocks_in_q1;
            if constexpr (kUseUnalignedKs) {
                if (num_kv_blocks != 0) {
                    const uint32_t kv_offset = cu_seq_len_k_start[q_token_base + q_token_offset] % SPARSE_BLOCK_KV;
                    for (uint32_t q_slot_idx = thread_idx * 4; q_slot_idx < num_kv_blocks;
                         q_slot_idx += kNumThreads * 4) {
                        const uint4 blocks = ld_global_evict_first_u128(reinterpret_cast<const uint4*>(src + q_slot_idx));
                        st_shared_u32x4(dst + q_slot_idx,
                                        blocks.x * SPARSE_BLOCK_KV + kv_offset,
                                        blocks.y * SPARSE_BLOCK_KV + kv_offset,
                                        blocks.z * SPARSE_BLOCK_KV + kv_offset,
                                        blocks.w * SPARSE_BLOCK_KV + kv_offset);
                    }
                }
            } else {
                for (uint32_t q_slot_idx = thread_idx * 4; q_slot_idx < num_kv_blocks;
                     q_slot_idx += kNumThreads * 4)
                    cp_async_cg16(reinterpret_cast<uint4*>(dst + q_slot_idx),
                                  reinterpret_cast<const uint4*>(src + q_slot_idx));
            }
        }
        if constexpr (!kUseUnalignedKs) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // ---- Merge-path partition of the two sorted lists -------------------
        // Each thread owns the merged range [merge_begin, merge_end) of the
        // concatenated input; a binary search finds its starting (q0, q1)
        // cursor pair, and each thread merges forward, packing both tokens'
        // slot presence into one word per merged block.
        const uint32_t num_input_kv_blocks = num_kv_blocks_in_q0 + num_kv_blocks_in_q1;
        const uint32_t merge_begin = thread_idx * num_input_kv_blocks / kNumThreads;
        const uint32_t merge_end = (thread_idx + 1) * num_input_kv_blocks / kNumThreads;
        uint32_t lo = dg_max(merge_begin, num_kv_blocks_in_q1) - num_kv_blocks_in_q1;
        uint32_t hi = dg_min(merge_begin, num_kv_blocks_in_q0);
        while (lo < hi) {
            const uint32_t q0_slot_idx = (lo + hi) / 2;
            const uint32_t q1_slot_idx = merge_begin - q0_slot_idx;
            if (q1_slot_idx > 0 && q0_slot_idx < num_kv_blocks_in_q0 &&
                    ld_shared_u32(smem.logical_kv_block_indices[1] + q1_slot_idx - 1) >=
                    ld_shared_u32(smem.logical_kv_block_indices[0] + q0_slot_idx))
                lo = q0_slot_idx + 1;
            else
                hi = q0_slot_idx;
        }
        uint32_t q0_slot_idx = lo;
        uint32_t q1_slot_idx = merge_begin - lo;
        uint32_t num_remaining_inputs = merge_end - merge_begin;
        uint32_t num_merged_kv_blocks_in_thread = 0;

        // Drop a duplicate carried across merge partitions
        if (num_remaining_inputs > 0 && q0_slot_idx > 0 && q1_slot_idx < num_kv_blocks_in_q1 &&
            ld_shared_u32(smem.logical_kv_block_indices[0] + q0_slot_idx - 1) ==
                ld_shared_u32(smem.logical_kv_block_indices[1] + q1_slot_idx)) {
            ++q1_slot_idx;
            --num_remaining_inputs;
        }

        // Merge and pack each Q's sparse slot and presence bit
        constexpr uint32_t kPresentBit = 1u << (kSparseSlotBits - 1);
        constexpr uint32_t kSlotIndexMask = kPresentBit - 1;
        const auto pack_slot = [](const uint32_t slot_idx, const bool is_present) {
            return slot_idx | (is_present ? kPresentBit : 0u);
        };
        constexpr uint32_t kNumKVBlocksPerThread =
            (smem_t::kNumMaxMergedKVBlocks + kNumThreads - 1) / kNumThreads;
        uint32_t packed_slots_in_thread[kNumKVBlocksPerThread];
        #pragma unroll
        for (uint32_t merged_kv_block_offset_in_thread = 0;
             merged_kv_block_offset_in_thread < kNumKVBlocksPerThread; ++merged_kv_block_offset_in_thread) {
            if (num_remaining_inputs == 0)
                continue;
            const uint32_t q0_logical_kv_block_idx =
                q0_slot_idx < num_kv_blocks_in_q0 ? ld_shared_u32(smem.logical_kv_block_indices[0] + q0_slot_idx) : ~0u;
            const uint32_t q1_logical_kv_block_idx =
                q1_slot_idx < num_kv_blocks_in_q1 ? ld_shared_u32(smem.logical_kv_block_indices[1] + q1_slot_idx) : ~0u;
            const bool in_q0 = q0_logical_kv_block_idx <= q1_logical_kv_block_idx;
            const bool in_q1 = q1_logical_kv_block_idx <= q0_logical_kv_block_idx;
            // Carry a final duplicate into the next partition
            const bool consume_q1 = in_q1 && num_remaining_inputs > (in_q0 ? 1u : 0u);
            packed_slots_in_thread[merged_kv_block_offset_in_thread] =
                pack_slot(q0_slot_idx, in_q0) | (pack_slot(q1_slot_idx, in_q1) << kSparseSlotBits);
            ++num_merged_kv_blocks_in_thread;
            q0_slot_idx += in_q0 ? 1u : 0u;
            q1_slot_idx += consume_q1 ? 1u : 0u;
            num_remaining_inputs -= (in_q0 ? 1u : 0u) + (consume_q1 ? 1u : 0u);
        }

        // ---- Compact merged blocks and reserve their KV-split range --------
        uint32_t num_merged_kv_blocks;
        const uint32_t merged_kv_block_base = sparse_cta_exclusive_sum<kNumThreads>(
            num_merged_kv_blocks_in_thread, smem.warp_sums, num_merged_kv_blocks);
        const uint32_t num_kv_splits_in_q_block = ceil_div_u32(num_merged_kv_blocks, kNumKVBlocksPerSplit);
        if (thread_idx == 0) {
            smem.q_block.kv_split_base = num_kv_splits_in_q_block == 0 ? 0u
                : atom_add_u32(&workspace_state->num_kv_splits, num_kv_splits_in_q_block);
            q_block_infos[q_token_base] = SparseQBlockInfo(smem.q_block.kv_split_base, num_kv_splits_in_q_block);
            if constexpr (kIsPaged) {
                if (num_q_block_tokens == BLOCK_Q)
                    q_block_infos[q_token_base + 1] = SparseQBlockInfo(0, 0);
            }
        }
        #pragma unroll
        for (uint32_t merged_kv_block_offset_in_thread = 0;
             merged_kv_block_offset_in_thread < kNumKVBlocksPerThread; ++merged_kv_block_offset_in_thread) {
            if (merged_kv_block_offset_in_thread >= num_merged_kv_blocks_in_thread)
                continue;
            const uint32_t merged_kv_block_idx = merged_kv_block_base + merged_kv_block_offset_in_thread;
            if constexpr (kIsPaged) {
                // Paged keeps one packed Q-slot base per split (its first block's slots)
                if (merged_kv_block_idx % kNumKVBlocksPerSplit == 0) {
                    smem.packed_slot_bases_by_kv_split[merged_kv_block_idx / kNumKVBlocksPerSplit] =
                        packed_slots_in_thread[merged_kv_block_offset_in_thread] &
                        (kSlotIndexMask | (kSlotIndexMask << kSparseSlotBits));
                }
            } else {
                // Non-paged retains each merged KV block's Q slots for its second pass
                smem.packed_slots_by_merged_kv_block[merged_kv_block_idx] =
                    packed_slots_in_thread[merged_kv_block_offset_in_thread];
            }
        }
        __syncthreads();

        const uint32_t kv_split_base = smem.q_block.kv_split_base;
        // Publish one merged block into its KV split (header on the first
        // block of the split, block info for every block).
        const auto write_merged_kv_block = [&](const uint32_t merged_kv_block_idx, const uint32_t packed_slots) {
            const uint32_t kv_split_offset = merged_kv_block_idx / kNumKVBlocksPerSplit;
            const uint32_t kv_split_idx = kv_split_base + kv_split_offset;
            const uint32_t kv_block_idx_in_split = merged_kv_block_idx % kNumKVBlocksPerSplit;
            // Paged stores one packed Q-slot base per split; non-paged reads the first
            // block of the split from the per-block array
            uint32_t packed_slot_bases;
            if constexpr (kIsPaged)
                packed_slot_bases = ld_shared_u32(smem.packed_slot_bases_by_kv_split + kv_split_offset);
            else
                packed_slot_bases = ld_shared_u32(smem.packed_slots_by_merged_kv_block +
                                                  merged_kv_block_idx - kv_block_idx_in_split);
            const uint32_t q0_slot_idx2 = packed_slots & kSlotIndexMask;
            const uint32_t q1_slot_idx2 = (packed_slots >> kSparseSlotBits) & kSlotIndexMask;
            const uint32_t q0_slot_base = packed_slot_bases & kSlotIndexMask;
            const uint32_t q1_slot_base = (packed_slot_bases >> kSparseSlotBits) & kSlotIndexMask;
            const bool in_q0 = (packed_slots & kPresentBit) != 0;
            const bool in_q1 = ((packed_slots >> kSparseSlotBits) & kPresentBit) != 0;
            const uint32_t logical_kv_block_idx = in_q0
                ? ld_shared_u32(smem.logical_kv_block_indices[0] + q0_slot_idx2)
                : ld_shared_u32(smem.logical_kv_block_indices[1] + q1_slot_idx2);
            uint32_t physical_kv_block_idx;
            if constexpr (kIsPaged) {
                constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
                const uint32_t logical_page_idx = logical_kv_block_idx / kNumKVBlocksPerPage;
                const uint64_t block_table_idx = static_cast<uint64_t>(q_token_base) * block_table_stride + logical_page_idx;
                physical_kv_block_idx = block_table[block_table_idx] * kNumKVBlocksPerPage +
                                        logical_kv_block_idx % kNumKVBlocksPerPage;
            } else {
                physical_kv_block_idx = logical_kv_block_idx * (kUseUnalignedKs ? 1u : SPARSE_BLOCK_KV);
            }
            const uint32_t q0_slot_offset = in_q0 ? q0_slot_idx2 - q0_slot_base : kSparseInvalidSlot;
            const uint32_t q1_slot_offset = in_q1 ? q1_slot_idx2 - q1_slot_base : kSparseInvalidSlot;
            if (kv_block_idx_in_split == 0) {
                const uint32_t num_kv_blocks_in_split =
                    dg_min(kNumKVBlocksPerSplit, num_merged_kv_blocks - merged_kv_block_idx);
                bool is_contiguous = false, has_partial_tail = false;
                if constexpr (!kIsPaged) {
                    // Merged blocks are sorted, so only the last one can run past the KV tensor
                    const uint32_t last_packed_slots = ld_shared_u32(
                        smem.packed_slots_by_merged_kv_block + merged_kv_block_idx + num_kv_blocks_in_split - 1);
                    const bool last_in_q0 = (last_packed_slots & kPresentBit) != 0;
                    const uint32_t last_q_slot_idx = last_in_q0 ? last_packed_slots & kSlotIndexMask
                                                                : (last_packed_slots >> kSparseSlotBits) & kSlotIndexMask;
                    const uint32_t* last_logical_kv_block_indices = smem.logical_kv_block_indices[last_in_q0 ? 0 : 1];
                    const uint32_t last_logical_kv_block_idx = ld_shared_u32(last_logical_kv_block_indices + last_q_slot_idx);
                    const uint32_t last_kv_token_start = last_logical_kv_block_idx * (kUseUnalignedKs ? 1u : SPARSE_BLOCK_KV);
                    // TMA copies full splits; partial splits stay on the generic path
                    if constexpr (!kUseUnalignedKs) {
                        is_contiguous = num_kv_blocks_in_split == kNumKVBlocksPerSplit &&
                                        last_logical_kv_block_idx == logical_kv_block_idx + num_kv_blocks_in_split - 1;
                    }
                    // TMA zero-fills past the tensor itself, so only generic copies need the clipped path
                    has_partial_tail = !is_contiguous && last_kv_token_start + SPARSE_BLOCK_KV > num_kv_tokens;
                }
                kv_splits[kv_split_idx].header = SparseKVSplitHeader(
                    q_token_base, num_kv_blocks_in_split, is_contiguous, has_partial_tail,
                    q0_slot_base, num_q_block_tokens == BLOCK_Q ? q1_slot_base : ~0u);
            }
            kv_splits[kv_split_idx].kv_block_infos[kv_block_idx_in_split] =
                SparseKVBlockInfo(physical_kv_block_idx, q0_slot_offset, q1_slot_offset);
        };

        // Write KV split metadata (paged: from this thread's registers;
        // non-paged: second pass over the compacted smem array).
        if constexpr (kIsPaged) {
            #pragma unroll
            for (uint32_t merged_kv_block_offset_in_thread = 0;
                 merged_kv_block_offset_in_thread < kNumKVBlocksPerThread; ++merged_kv_block_offset_in_thread) {
                if (merged_kv_block_offset_in_thread >= num_merged_kv_blocks_in_thread)
                    continue;
                const uint32_t merged_kv_block_idx = merged_kv_block_base + merged_kv_block_offset_in_thread;
                write_merged_kv_block(merged_kv_block_idx, packed_slots_in_thread[merged_kv_block_offset_in_thread]);
            }
        } else {
            for (uint32_t merged_kv_block_idx = thread_idx; merged_kv_block_idx < num_merged_kv_blocks;
                 merged_kv_block_idx += kNumThreads)
                write_merged_kv_block(merged_kv_block_idx,
                                      ld_shared_u32(smem.packed_slots_by_merged_kv_block + merged_kv_block_idx));
        }

        // Pad the last split with absent blocks to keep the main-kernel copy
        // loop branch-free (every warp copies its share unconditionally).
        for (uint32_t padded_kv_block_slot_idx = num_merged_kv_blocks + thread_idx;
             padded_kv_block_slot_idx < num_kv_splits_in_q_block * kNumKVBlocksPerSplit;
             padded_kv_block_slot_idx += kNumThreads) {
            kv_splits[kv_split_base + padded_kv_block_slot_idx / kNumKVBlocksPerSplit]
                .kv_block_infos[padded_kv_block_slot_idx % kNumKVBlocksPerSplit] =
                SparseKVBlockInfo(0, kSparseInvalidSlot, kSparseInvalidSlot);
        }
        if (thread_idx == 0) {
            q_token_idx = kIsPaged ? gridDim.x + atom_add_u32(&workspace_state->next_q_offset, 1u)
                                   : q_token_idx + gridDim.x * BLOCK_Q;
        }
    }

    // ---- Pass 2: the last CTA acquires all producer writes and builds the
    // schedule. Release (atom.add.release) by every CTA's thread 0 publishes
    // its kv_splits / q_block_infos writes; the last CTA's ld.acquire + the
    // __syncthreads above make them visible to all of its threads.
    if (thread_idx == 0) {
        smem.is_last_cta = atom_add_rel_u32(&workspace_state->num_finished_ctas, 1u) + 1 == gridDim.x ? 1u : 0u;
        if (smem.is_last_cta != 0)
            (void)ld_acq_u32(&workspace_state->num_finished_ctas);
    }
    __syncthreads();
    if (smem.is_last_cta == 0)
        return;

    const uint32_t total_kv_splits = workspace_state->num_kv_splits;
    auto schedule_entries = reinterpret_cast<SparseScheduleEntry*>(kv_splits + total_kv_splits);
    uint32_t num_waves;
    if constexpr (!kIsPaged) {
        num_waves = build_contiguous_schedule<BLOCK_Q, kNumKVBlocksPerSplit, kNumSMs>(
            schedule_entries, kv_splits, q_block_infos, num_q_tokens, total_kv_splits, smem);
    } else {
        num_waves = build_paged_schedule<BLOCK_Q, kNumKVSplitsPerEntry, kNumSMs, kNumThreads>(
            schedule_entries, q_block_infos, num_q_tokens, indices, smem);
    }
    if (thread_idx == 0) {
        auto header = reinterpret_cast<SparseMetadataHeader*>(metadata);
        header->num_kv_splits = total_kv_splits;
        header->num_waves = num_waves;
        header->use_unaligned_ks = kUseUnalignedKs ? 1u : 0u;
        header->num_sms = kNumSMs;
        // Self-reset the workspace counters for the next launch.
        workspace_state->num_kv_splits = 0;
        workspace_state->next_q_offset = 0;
        workspace_state->num_finished_ctas = 0;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("trap;");
#endif
}

// ===========================================================================
// SCORING KERNEL
// (port of impls/sm100_sparse_mqa_logits.cuh)
// ===========================================================================
template <uint32_t kNumHeads, uint32_t SPARSE_BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages, uint32_t kNumMathWarpGroups, uint32_t kNumSMs, uint32_t BLOCK_Q,
          bool kUseUnalignedKs, bool kIsFP4, typename KVAccessor>
DG_DEVICE void sparse_mqa_logits_core_impl(const uint32_t logits_stride, bf16_raw* logits,
        const uint8_t* metadata, const TmaMap& tensor_map_q,
        const TmaMap& tensor_map_sf_q, const TmaMap& tensor_map_weights,
        uint8_t* smem_buffer, const KVAccessor& kv_accessor) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    // MMA configs. A = KV (128 rows per math WG), B = Q (BLOCK_Q * heads
    // rows); one TMEM accumulator tile of UMMA_N columns per (split, WG).
    constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    constexpr uint32_t kNumWeightElementsPerRow = align_u32(kNumHeads, 8u);
    // Whole 4/8/16-head loads avoid reading beyond the last token's TMEM columns.
    constexpr uint32_t kNumHeadsPerLoad = sparse_gcd(kNumHeads, 16u);
    constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    constexpr uint32_t SPLIT_KV = kNumMathWarpGroups * UMMA_M;
    constexpr uint32_t kQKSwizzleMode = kSparseHeadDim / kPackFactor;

    // Thread and register configs
    constexpr uint32_t kNumThreads = sparse_num_threads(kNumMathWarpGroups);
    constexpr uint32_t kNumWarpGroups = kNumThreads / 128;
    constexpr uint32_t kNumMathThreads = kNumMathWarpGroups * 128;
    constexpr uint32_t kNumKVCopyThreads = kNumMathWarpGroups * 32;
    constexpr uint32_t kNumEntryRegisters = (512 / kNumWarpGroups / 8) * 8;
    constexpr uint32_t kNumControlRegisters = 64;
    constexpr uint32_t kNumKVCopyRegisters = 64;
    constexpr uint32_t kNumMathRegisters = dg_min(120u, ((kNumEntryRegisters * kNumWarpGroups - kNumControlRegisters -
        kNumKVCopyRegisters * (kNumKVCopyThreads / 128)) / kNumMathWarpGroups / 8) * 8);

    // Memory configs
    using smem_t = SparseLogitsSmem<BLOCK_Q, SPARSE_BLOCK_KV, SPLIT_KV,
                                    kNumQStages, kNumKVStages, kNumTmemStages, kIsFP4>;
    constexpr uint32_t kNumKVBlocksPerSplit = smem_t::kNumKVBlocksPerSplit;
    constexpr uint32_t kNumSFQ = smem_t::kNumSFQ;
    constexpr uint32_t kNumSFQCols = kNumSFQ / 32;
    constexpr uint32_t kNumSFKVColsPerStage = SPLIT_KV / 32;
    constexpr uint32_t kTmemStartColOfSFQ = UMMA_N * kNumTmemStages;
    constexpr uint32_t kTmemStartColOfSFKV = kTmemStartColOfSFQ + kNumQStages * kNumSFQCols;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kTmemStartColOfSFKV + kNumKVStages * kNumSFKVColsPerStage>();

    // Template checks
    DG_STATIC_ASSERT(BLOCK_Q == 2, "Sparse MQA requires BLOCK_Q=2");
    DG_STATIC_ASSERT(kNumHeads > 0 && kNumHeads <= kSparseMaxHeads && kNumHeads % 4 == 0, "Invalid head count");
    DG_STATIC_ASSERT(kNumMathWarpGroups == 4 || kNumMathWarpGroups == 5,
                     "Invalid number of math warpgroups");
    DG_STATIC_ASSERT(kNumTmemStages >= kNumMathWarpGroups, "Invalid TMEM stage count");
    DG_STATIC_ASSERT(kNumTmemCols <= 512 && kNumThreads <= 1024, "Sparse MQA resource overflow");
    DG_STATIC_ASSERT(kNumMathRegisters * kNumMathWarpGroups + kNumControlRegisters +
                     kNumKVCopyRegisters * (kNumKVCopyThreads / 128) <= kNumEntryRegisters * kNumWarpGroups,
                     "Register reconfiguration exceeds the CTA entry pool");

    // Thread indices
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();
    const uint32_t warpgroup_idx = warp_idx / 4;
    constexpr uint32_t kQAndMetadataWarpIdx = kNumMathWarpGroups * 4;
    constexpr uint32_t kSFTransposeWarpIdx = kQAndMetadataWarpIdx + 1;
    constexpr uint32_t kUMMAWarpIdx = kQAndMetadataWarpIdx + 2;
    constexpr uint32_t kFirstKVCopyWarpIdx = kNumThreads / 32 - kNumMathWarpGroups;

    // Shared memory (base passed in by the __global__ wrapper)
    smem_t& smem = *reinterpret_cast<smem_t*>(smem_buffer);
    const SparseKVSplit<kNumKVBlocksPerSplit>* kv_splits =
        reinterpret_cast<const SparseKVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(SparseMetadataHeader));

    // Initialization
    if (warp_idx == kQAndMetadataWarpIdx && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_q);
        prefetch_tma_map(&tensor_map_sf_q);
        prefetch_tma_map(&tensor_map_weights);
        if constexpr (KVAccessor::kSupportsContiguousTMA)
            kv_accessor.prefetch_tma_descriptors();
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumQStages; ++stage_idx) {
            smem.full_q_barriers[stage_idx].init(1);
            smem.full_sf_q_barriers[stage_idx].init(1);
            // Released by the producer warp, UMMA warp, and math warpgroups
            // (see the barrier-topology section of the file banner).
            smem.empty_q_barriers[stage_idx].init(kNumMathThreads + 64);
        }
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx) {
            smem.full_metadata_barriers[stage_idx].init(32);
            smem.full_sf_copy_barriers[stage_idx].init(kNumKVCopyThreads);
            // The SF transpose warp contributes the final arrival after its UTCCP completes.
            smem.full_kv_barriers[stage_idx].init(kNumKVCopyThreads + 1);
            smem.empty_kv_barriers[stage_idx].init(kNumMathThreads + 1);
        }
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumTmemStages; ++stage_idx) {
            smem.full_tmem_barriers[stage_idx].init(1);
            smem.empty_tmem_barriers[stage_idx].init(128);
        }
        fence_barrier_init();
    }
    if (warp_idx == kSFTransposeWarpIdx)
        tmem_alloc_1sm(kNumTmemCols, &smem.tmem_ptr_in_smem);

    // Zero padded Q scales for UTCCP: the TMA only refreshes the first
    // BLOCK_Q * kNumHeads words of each stage; the tail up to the 128-word
    // UTCCP atom must read as UE8M0 exponent 0 forever after.
    constexpr uint32_t kNumSFQValues = BLOCK_Q * kNumHeads;
    constexpr uint32_t kNumSFQPaddingValues = kNumSFQ - kNumSFQValues;
    for (uint32_t padding_idx = threadIdx.x; padding_idx < kNumQStages * kNumSFQPaddingValues;
         padding_idx += kNumThreads) {
        const uint32_t q_stage_idx = padding_idx / kNumSFQPaddingValues;
        const uint32_t stage_padding_idx = padding_idx % kNumSFQPaddingValues;
        smem.sf_q[q_stage_idx][kNumSFQValues + stage_padding_idx] = 0;
    }
    __syncthreads();

    griddepcontrol_wait();

    if (warp_idx == kQAndMetadataWarpIdx) {
        // ---- Q / weights / metadata producer --------------------------------
        setmaxnreg_dec<kNumControlRegisters>();
        SparseRingPipeline<kNumQStages> q_pipeline;
        SparseRingPipeline<kNumKVStages> kv_pipeline;

        const SparseMetadataHeader* header = reinterpret_cast<const SparseMetadataHeader*>(metadata);
        // The schedule is laid out as `[wave][num_sms]`, so a changed SM count needs new metadata
        const uint32_t num_waves = header->num_waves;
        const SparseScheduleEntry* schedule_entries =
            reinterpret_cast<const SparseScheduleEntry*>(kv_splits + header->num_kv_splits);
        for (uint32_t wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
            const SparseScheduleEntry entry = schedule_entries[wave_idx * kNumSMs + blockIdx.x];
            if (entry.kv_split_begin == entry.kv_split_end)
                continue;

            const SparseStagePhase qsp = q_pipeline.advance();
            smem.empty_q_barriers[qsp.stage].wait(qsp.phase ^ 1u);
            if (elect_one_sync()) {
                smem.q_blocks[qsp.stage] = SparseQBlock(
                    entry.q_token_base, entry.num_q_tokens, entry.kv_split_end - entry.kv_split_begin);
                // Q rows: [head_dim, 2 * heads] box at (0, q_token_base * heads).
                tma_load_2d(&tensor_map_q, &smem.full_q_barriers[qsp.stage], smem.q[qsp.stage],
                            kEvictNormalHint, 0, entry.q_token_base * kNumHeads);
                // Q SF: [heads, 2] box of packed UE8M0 words at (0, q_token_base).
                tma_load_2d(&tensor_map_sf_q, &smem.full_q_barriers[qsp.stage], smem.sf_q[qsp.stage],
                            kEvictNormalHint, 0, entry.q_token_base);
                // Per-token head weights: [align(heads, 8), 2] bf16 box.
                tma_load_2d(&tensor_map_weights, &smem.full_q_barriers[qsp.stage], smem.weights[qsp.stage],
                            kEvictNormalHint, 0, entry.q_token_base);
                smem.full_q_barriers[qsp.stage].arrive_and_expect_tx(
                    BLOCK_Q * (kNumHeads * (kSparseHeadDim / kPackFactor + sizeof(uint32_t)) +
                               kNumWeightElementsPerRow * sizeof(uint16_t)));
            }

            // Stage this entry's KV-split metadata (header + per-block infos)
            // into smem with cp.async; the copy warps take over from there.
            for (uint32_t kv_split_idx = entry.kv_split_begin; kv_split_idx < entry.kv_split_end; ++kv_split_idx) {
                const SparseStagePhase kvp = kv_pipeline.advance();
                smem.empty_kv_barriers[kvp.stage].wait(kvp.phase ^ 1u);
                if (elect_one_sync())
                    cp_async_cg16(&smem.kv_split_headers[kvp.stage],
                                  &kv_splits[kv_split_idx].header);
                for (uint32_t chunk_idx = lane_idx; chunk_idx < kNumKVBlocksPerSplit / 2; chunk_idx += 32) {
                    cp_async_cg16(reinterpret_cast<uint4*>(smem.kv_block_infos[kvp.stage]) + chunk_idx,
                                  reinterpret_cast<const uint4*>(kv_splits[kv_split_idx].kv_block_infos) + chunk_idx);
                }
                cpasync_barrier_arrive_noinc(&smem.full_metadata_barriers[kvp.stage]);
            }
            smem.empty_q_barriers[qsp.stage].arrive();
        }

        // Sentinels: an empty Q block for the full_q consumers and an empty
        // split header for the copy warps.
        const SparseStagePhase qsp = q_pipeline.advance();
        smem.empty_q_barriers[qsp.stage].wait(qsp.phase ^ 1u);
        const SparseStagePhase kvp = kv_pipeline.advance();
        smem.empty_kv_barriers[kvp.stage].wait(kvp.phase ^ 1u);
        if (elect_one_sync()) {
            smem.q_blocks[qsp.stage].num_q_tokens = 0;
            smem.kv_split_headers[kvp.stage].packed_num_kv_blocks = 0;
            // Publish the generic sentinel with a regular release arrival
            smem.full_metadata_barriers[kvp.stage].arrive_count(32);
            smem.full_q_barriers[qsp.stage].arrive();
        }
    } else if (warp_idx == kSFTransposeWarpIdx) {
        // ---- SF transpose + UTCCP --------------------------------------------
        setmaxnreg_dec<kNumControlRegisters>();
        // 32x128b UTCCP wants each lane's 4 consecutive SF words strided by
        // 32 (one word per datapath row); the TMA delivered them token-major,
        // so a warp 4x32 transpose fixes the layout in place.
        const auto transpose_sf = [&](uint32_t* smem_ptr, const uint32_t num_sf) {
            for (uint32_t sf_base = 0; sf_base < num_sf; sf_base += kSparseUTCCPElems) {
                uint32_t values[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i)
                    values[i] = ld_shared_u32(smem_ptr + sf_base + i * 32 + lane_idx);
                __syncwarp();
                st_shared_u32x4(smem_ptr + sf_base + lane_idx * 4, values[0], values[1], values[2], values[3]);
            }
        };
        SmemDescriptor sf_desc = make_sf_desc(nullptr);
        SparseRingPipeline<kNumQStages> q_pipeline;
        SparseRingPipeline<kNumKVStages> kv_pipeline;
        while (true) {
            const SparseStagePhase qsp = q_pipeline.advance();
            smem.full_q_barriers[qsp.stage].wait(qsp.phase);
            if (ld_shared_u32(&smem.q_blocks[qsp.stage].num_q_tokens) == 0)
                break;
            const uint32_t num_kv_splits = ld_shared_u32(&smem.q_blocks[qsp.stage].num_kv_splits);
            transpose_sf(smem.sf_q[qsp.stage], kNumSFQ);
            fence_view_async_shared();
            if (elect_one_sync()) {
                replace_smem_desc_addr(sf_desc, smem.sf_q[qsp.stage]);
                utccp_4x32dp128bit_1cta(sf_desc.desc_, kTmemStartColOfSFQ + qsp.stage * kNumSFQCols);
                tcgen05_before_thread_sync();
                smem.full_sf_q_barriers[qsp.stage].arrive();
            }
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                const SparseStagePhase kvp = kv_pipeline.advance();
                // The copy warps only signal this after consuming the corresponding metadata.
                smem.full_sf_copy_barriers[kvp.stage].wait(kvp.phase);
                transpose_sf(smem.sf_kv[kvp.stage], SPLIT_KV);
                fence_view_async_shared();
                if (elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t sf_idx = 0; sf_idx < SPLIT_KV; sf_idx += kSparseUTCCPElems) {
                        replace_smem_desc_addr(sf_desc, smem.sf_kv[kvp.stage] + sf_idx);
                        utccp_4x32dp128bit_1cta(sf_desc.desc_,
                            kTmemStartColOfSFKV + kvp.stage * kNumSFKVColsPerStage + sf_idx / 32);
                    }
                    tcgen05_before_thread_sync();
                    smem.full_kv_barriers[kvp.stage].arrive();
                }
            }
        }
    } else if (warp_idx == kUMMAWarpIdx) {
        // ---- MMA issue --------------------------------------------------------
        setmaxnreg_dec<kNumControlRegisters>();
        if (elect_one_sync()) {
            SparseRingPipeline<kNumQStages> q_pipeline;
            SparseRingPipeline<kNumKVStages> kv_pipeline;
            SparseRingPipeline<kNumTmemStages> tmem_pipeline;
            while (true) {
                const SparseStagePhase qsp = q_pipeline.advance();
                smem.full_q_barriers[qsp.stage].wait(qsp.phase);
                if (ld_shared_u32(&smem.q_blocks[qsp.stage].num_q_tokens) == 0)
                    break;
                const uint32_t num_kv_splits = ld_shared_u32(&smem.q_blocks[qsp.stage].num_kv_splits);
                smem.full_sf_q_barriers[qsp.stage].wait(qsp.phase);
                tcgen05_after_thread_sync();
                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                    const SparseStagePhase kvp = kv_pipeline.advance();
                    smem.full_kv_barriers[kvp.stage].wait(kvp.phase);
                    fence_view_async_shared();
                    #pragma unroll
                    for (uint32_t m_idx = 0; m_idx < kNumMathWarpGroups; ++m_idx) {
                        const SparseStagePhase tp = tmem_pipeline.advance();
                        const uint32_t tmem_addr = tp.stage * UMMA_N;
                        smem.empty_tmem_barriers[tp.stage].wait(tp.phase ^ 1u);
                        tcgen05_after_thread_sync();
                        #pragma unroll
                        for (uint32_t k_idx = 0; k_idx < kSparseHeadDim / UMMA_K; ++k_idx) {
                            // One packed UE8M0 SF word covers 4 x 32 K elements
                            // (= head_dim); the byte selector advances with k.
                            const uint32_t sf_id = k_idx * kPackFactor;
                            const uint64_t runtime_instr_desc = make_runtime_instr_desc_bs(
                                make_instr_desc_bs(kIsFP4 ? 5u : 0u, kIsFP4 ? 5u : 0u,
                                                   UMMA_M, UMMA_N, MAJOR_K, MAJOR_K),
                                sf_id, sf_id);
                            const SmemDescriptor a_desc =
                                make_umma_desc<MAJOR_K, 0, kSparseHeadDim, kQKSwizzleMode, kPackFactor, 1>(
                                    smem.kv[kvp.stage], m_idx * UMMA_M, k_idx * UMMA_K);
                            const SmemDescriptor b_desc =
                                make_umma_desc<MAJOR_K, 0, kSparseHeadDim, kQKSwizzleMode, kPackFactor, 1>(
                                    smem.q[qsp.stage], 0, k_idx * UMMA_K);
                            if (kIsFP4) {
                                mma_mxf4_1sm(a_desc.desc_, b_desc.desc_, tmem_addr, k_idx, runtime_instr_desc,
                                             kTmemStartColOfSFKV + kvp.stage * kNumSFKVColsPerStage + m_idx * 4,
                                             kTmemStartColOfSFQ + qsp.stage * kNumSFQCols);
                            } else {
                                mma_mxf8f6f4_1sm(a_desc.desc_, b_desc.desc_, tmem_addr, k_idx, runtime_instr_desc,
                                                 kTmemStartColOfSFKV + kvp.stage * kNumSFKVColsPerStage + m_idx * 4,
                                                 kTmemStartColOfSFQ + qsp.stage * kNumSFQCols);
                            }
                        }
                        umma_arrive_1sm(&smem.full_tmem_barriers[tp.stage]);
                    }
                    umma_arrive_1sm(&smem.empty_kv_barriers[kvp.stage]);
                }
                smem.empty_q_barriers[qsp.stage].arrive_count(32);
            }
        }
    } else if (warp_idx >= kFirstKVCopyWarpIdx) {
        // ---- KV copy warps (gather the selected blocks) -----------------------
        if (warpgroup_idx == kNumMathWarpGroups)
            setmaxnreg_dec<kNumControlRegisters>();
        else
            setmaxnreg_dec<kNumKVCopyRegisters>();
        constexpr uint32_t kNumKVBlocksPerWarp = UMMA_M / SPARSE_BLOCK_KV;
        constexpr uint32_t kNumChunksPerKVBlock = SPARSE_BLOCK_KV * (kSparseHeadDim / kPackFactor) / 16;
        constexpr uint32_t kNumSFChunksPerKVBlock = SPARSE_BLOCK_KV * sizeof(uint32_t) / 16;
        DG_STATIC_ASSERT(kNumKVBlocksPerWarp % 2 == 0, "Each KV copy warp must process pairs of sparse KV blocks");
        DG_STATIC_ASSERT(kNumKVBlocksPerWarp * kNumSFChunksPerKVBlock == 32,
                         "Each KV copy warp must issue exactly one full-warp SF copy");
        const uint32_t copy_warp_idx = warp_idx - kFirstKVCopyWarpIdx;
        SparseRingPipeline<kNumKVStages> kv_pipeline;
        while (true) {
            const SparseStagePhase kvp = kv_pipeline.advance();
            smem.full_metadata_barriers[kvp.stage].wait(kvp.phase);
            const uint32_t packed_num_kv_blocks =
                ld_shared_u32(&smem.kv_split_headers[kvp.stage].packed_num_kv_blocks);
            const uint32_t num_kv_blocks = packed_num_kv_blocks & ~SparseKVSplitHeader::kFlagMask;
            if (num_kv_blocks == 0)
                break;

            if constexpr (KVAccessor::kSupportsContiguousTMA) {
                if (packed_num_kv_blocks & SparseKVSplitHeader::kContiguousFlag) {
                    // One TMA lane replaces all generic copy warps for a contiguous split
                    if (copy_warp_idx == 0 && elect_one_sync()) {
                        const uint32_t kv_token_start = ld_shared_u32(
                            &smem.kv_block_infos[kvp.stage][0].physical_kv_block_idx);
                        kv_accessor.template copy_contiguous_kv_split<SPLIT_KV>(
                            smem.full_kv_barriers[kvp.stage], smem.full_sf_copy_barriers[kvp.stage],
                            smem.kv[kvp.stage], smem.sf_kv[kvp.stage], kv_token_start);
                    }
                    __syncwarp();
                    // Each copy warp releases its share only after consuming the header.
                    // Warp 0 arrives 31: its elected lane's arrive_and_expect_tx
                    // covers the 32nd arrival on both barriers.
                    const uint32_t num_arrivals = copy_warp_idx == 0 ? 31 : 32;
                    const bool is_elected = elect_one_sync();
                    mbarrier_arrive_count_pred(&smem.full_kv_barriers[kvp.stage], num_arrivals, is_elected);
                    mbarrier_arrive_count_pred(&smem.full_sf_copy_barriers[kvp.stage], num_arrivals, is_elected);
                    continue;
                }
            }

            const uint32_t kv_block_base_in_split = copy_warp_idx * kNumKVBlocksPerWarp;
            if constexpr (KVAccessor::kNeedsTailMask) {
                // Keep clipping out of the regular copy loop
                if (packed_num_kv_blocks & SparseKVSplitHeader::kPartialTailFlag) {
                    if (kv_block_base_in_split < num_kv_blocks) {
                        // Lane L resolves block (base + L) so the whole warp's
                        // block refs are available for shuffles; validity is
                        // per-block (a block may only be partially inside KV).
                        const auto kv_block_ref = kv_accessor.resolve_kv_block(
                            lane_idx < kNumKVBlocksPerWarp
                                ? ld_shared_u32(&smem.kv_block_infos[kvp.stage]
                                                [kv_block_base_in_split + lane_idx].physical_kv_block_idx)
                                : 0);
                        const uint32_t num_valid_tokens =
                            kv_block_base_in_split + lane_idx < num_kv_blocks
                                ? kv_accessor.get_num_valid_tokens(kv_block_ref) : 0;
                        // Scalar SF copies also handle unaligned KS and partially valid 16-byte vectors
                        constexpr uint32_t kNumSFBlocksPerIteration = 32 / SPARSE_BLOCK_KV;
                        #pragma unroll
                        for (uint32_t sf_kv_block_base = 0; sf_kv_block_base < kNumKVBlocksPerWarp;
                             sf_kv_block_base += kNumSFBlocksPerIteration) {
                            const uint32_t sf_kv_block_offset = sf_kv_block_base + lane_idx / SPARSE_BLOCK_KV;
                            const uint32_t token_in_kv_block = lane_idx % SPARSE_BLOCK_KV;
                            const uint32_t* sf_kv_block = kv_accessor.get_sf_kv_block(kv_block_ref, sf_kv_block_offset);
                            const bool is_valid = token_in_kv_block < shfl_u32(num_valid_tokens, sf_kv_block_offset);
                            cp_async_ca4_zfill(
                                &smem.sf_kv[kvp.stage][(kv_block_base_in_split + sf_kv_block_offset) * SPARSE_BLOCK_KV + token_in_kv_block],
                                sf_kv_block + (is_valid ? token_in_kv_block : 0),
                                is_valid ? sizeof(uint32_t) : 0);
                        }
                        cpasync_barrier_arrive_noinc(&smem.full_sf_copy_barriers[kvp.stage]);

                        // Half a warp (16 lanes) per block: kNumChunksPerKVBlock/16
                        // iterations of 16B chunks, swizzled to match the TMA layout.
                        #pragma unroll
                        for (uint32_t kv_block_pair_offset = 0; kv_block_pair_offset < kNumKVBlocksPerWarp;
                             kv_block_pair_offset += 2) {
                            const uint32_t kv_block_offset = kv_block_pair_offset + lane_idx / 16;
                            const uint4* kv_block = reinterpret_cast<const uint4*>(
                                kv_accessor.get_kv_block(kv_block_ref, kv_block_offset));
                            const uint32_t num_valid_chunks =
                                shfl_u32(num_valid_tokens, kv_block_offset) * (kSparseHeadDim / kPackFactor / 16);
                            #pragma unroll
                            for (uint32_t chunk_base = 0; chunk_base < kNumChunksPerKVBlock; chunk_base += 16) {
                                const uint32_t chunk_idx = chunk_base + lane_idx % 16;
                                const bool is_valid = chunk_idx < num_valid_chunks;
                                // Invalid tokens and dummy blocks use a valid base pointer with a zero source size
                                cp_async_cg16_zfill(
                                    reinterpret_cast<uint4*>(smem.kv[kvp.stage]) +
                                        (kv_block_base_in_split + kv_block_offset) * kNumChunksPerKVBlock +
                                        get_swizzled_kv_chunk_idx<kQKSwizzleMode>(chunk_idx),
                                    kv_block + (is_valid ? chunk_idx : 0), is_valid ? 16 : 0);
                            }
                        }
                    } else {
                        cpasync_barrier_arrive_noinc(&smem.full_sf_copy_barriers[kvp.stage]);
                    }
                    cpasync_barrier_arrive_noinc(&smem.full_kv_barriers[kvp.stage]);
                    continue;
                }
            }
            if (kv_block_base_in_split < num_kv_blocks) {
                const auto kv_block_ref = kv_accessor.resolve_kv_block(
                    lane_idx < kNumKVBlocksPerWarp
                        ? ld_shared_u32(&smem.kv_block_infos[kvp.stage]
                                        [kv_block_base_in_split + lane_idx].physical_kv_block_idx)
                        : 0);

                // Copy SF first (so the SF-transpose warp can start early)
                if constexpr (kUseUnalignedKs) {
                    // NOTES: unaligned ks keeps KV rows aligned, but may only 4-byte align SF rows
                    constexpr uint32_t kNumSFBlocksPerIteration = 32 / SPARSE_BLOCK_KV;
                    #pragma unroll
                    for (uint32_t sf_kv_block_base = 0; sf_kv_block_base < kNumKVBlocksPerWarp;
                         sf_kv_block_base += kNumSFBlocksPerIteration) {
                        const uint32_t sf_kv_block_offset = sf_kv_block_base + lane_idx / SPARSE_BLOCK_KV;
                        const uint32_t token_in_kv_block = lane_idx % SPARSE_BLOCK_KV;
                        const uint32_t* sf_kv_block = kv_accessor.get_sf_kv_block(kv_block_ref, sf_kv_block_offset);
                        cp_async_ca4(&smem.sf_kv[kvp.stage]
                                     [(kv_block_base_in_split + sf_kv_block_offset) * SPARSE_BLOCK_KV + token_in_kv_block],
                                     sf_kv_block + token_in_kv_block);
                    }
                } else {
                    // One full-warp SF copy: lane -> (block, 16B SF chunk)
                    const uint32_t sf_kv_block_offset = lane_idx / kNumSFChunksPerKVBlock;
                    const uint32_t chunk_idx = lane_idx % kNumSFChunksPerKVBlock;
                    const uint32_t* sf_kv_block = kv_accessor.get_sf_kv_block(kv_block_ref, sf_kv_block_offset);
                    cp_async_cg16_l2_64b(reinterpret_cast<uint4*>(smem.sf_kv[kvp.stage] +
                                        (kv_block_base_in_split + sf_kv_block_offset) * SPARSE_BLOCK_KV) + chunk_idx,
                                         reinterpret_cast<const uint4*>(sf_kv_block) + chunk_idx);
                }
                cpasync_barrier_arrive_noinc(&smem.full_sf_copy_barriers[kvp.stage]);

                // Copy sparse KV blocks (16B chunks, half a warp per block)
                #pragma unroll
                for (uint32_t kv_block_pair_offset = 0; kv_block_pair_offset < kNumKVBlocksPerWarp;
                     kv_block_pair_offset += 2) {
                    const uint32_t kv_block_offset = kv_block_pair_offset + lane_idx / 16;
                    const uint8_t* kv_block = kv_accessor.get_kv_block(kv_block_ref, kv_block_offset);
                    #pragma unroll
                    for (uint32_t chunk_base = 0; chunk_base < kNumChunksPerKVBlock; chunk_base += 16) {
                        const uint32_t chunk_idx = chunk_base + lane_idx % 16;
                        cp_async_cg16(reinterpret_cast<uint4*>(smem.kv[kvp.stage]) +
                                          (kv_block_base_in_split + kv_block_offset) * kNumChunksPerKVBlock +
                                          get_swizzled_kv_chunk_idx<kQKSwizzleMode>(chunk_idx),
                                      reinterpret_cast<const uint4*>(kv_block) + chunk_idx);
                    }
                }
            } else {
                cpasync_barrier_arrive_noinc(&smem.full_sf_copy_barriers[kvp.stage]);
            }
            cpasync_barrier_arrive_noinc(&smem.full_kv_barriers[kvp.stage]);
        }
    } else if (warpgroup_idx < kNumMathWarpGroups) {
        // ---- Math warpgroups: weighted-ReLU reduce + compressed scatter -------
        setmaxnreg_inc<kNumMathRegisters>();
        const uint32_t math_thread_idx = (warp_idx % 4) * 32 + lane_idx;
        SparseRingPipeline<kNumQStages> q_pipeline;
        SparseRingPipeline<kNumKVStages> kv_pipeline;
        SparseRingPipeline<kNumTmemStages> tmem_pipeline;
        tmem_pipeline.advance(warpgroup_idx);  // each WG starts on its own stage

        while (true) {
            const SparseStagePhase qsp = q_pipeline.advance();
            smem.full_q_barriers[qsp.stage].wait(qsp.phase);
            const uint32_t q_token_base = ld_shared_u32(&smem.q_blocks[qsp.stage].q_token_base);
            const uint32_t num_q_tokens = ld_shared_u32(&smem.q_blocks[qsp.stage].num_q_tokens);
            const uint32_t num_kv_splits = ld_shared_u32(&smem.q_blocks[qsp.stage].num_kv_splits);
            if (num_q_tokens == 0)
                break;

            bf16_raw* output_rows[BLOCK_Q];
            // Packed bf16x2 weight pairs per token (raw u32 words).
            uint32_t weights[BLOCK_Q][kNumHeads / 2];
            float accum[kNumHeadsPerLoad];

            #pragma unroll
            for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++q_token_offset) {
                output_rows[q_token_offset] =
                    logits + static_cast<uint64_t>(q_token_base + q_token_offset) * logits_stride;
                if (q_token_offset >= num_q_tokens)
                    continue;
                const uint32_t* packed_weights = reinterpret_cast<const uint32_t*>(
                    smem.weights[qsp.stage] + q_token_offset * kNumWeightElementsPerRow);
                #pragma unroll
                for (uint32_t i = 0; i < kNumHeads / 2; ++i)
                    weights[q_token_offset][i] = packed_weights[i];
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                const SparseStagePhase kvp = kv_pipeline.advance();
                const SparseStagePhase tp = tmem_pipeline.advance(kNumMathWarpGroups);

                // One 128-token tile per math warpgroup: math thread t reads
                // TMEM row t (its KV token) of this WG's accumulator stage.
                const uint32_t token_in_kv_split = warpgroup_idx * UMMA_M + math_thread_idx;
                const uint32_t kv_block_idx_in_split = token_in_kv_split / SPARSE_BLOCK_KV;
                const uint32_t token_in_kv_block = token_in_kv_split - kv_block_idx_in_split * SPARSE_BLOCK_KV;

                smem.full_tmem_barriers[tp.stage].wait(tp.phase);
                tcgen05_after_thread_sync();
                // Metadata stays in the generic proxy, so no proxy fence is needed before releasing the stage
                const uint32_t q0_slot_base = ld_shared_u32(&smem.kv_split_headers[kvp.stage].q0_slot_base);
                const uint32_t q1_slot_base = ld_shared_u32(&smem.kv_split_headers[kvp.stage].q1_slot_base);
                const uint32_t packed_slot_offsets =
                    ld_shared_u32(&smem.kv_block_infos[kvp.stage][kv_block_idx_in_split].packed_slot_offsets);
                smem.empty_kv_barriers[kvp.stage].arrive();

                #pragma unroll
                for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++q_token_offset) {
                    if (q_token_offset >= num_q_tokens)
                        continue;
                    const uint32_t tmem_addr = tp.stage * UMMA_N + q_token_offset * kNumHeads;
                    uint32_t sum_0 = cvt_bf16x2_f32(0.0f, 0.0f);
                    uint32_t sum_1 = cvt_bf16x2_f32(0.0f, 0.0f);
                    #pragma unroll
                    for (uint32_t head_base = 0; head_base < kNumHeads; head_base += kNumHeadsPerLoad) {
                        // 32dp32b TMEM load: lane = KV token, columns = heads.
                        if (kNumHeadsPerLoad == 16)
                            tmem_load_32dp32b_x16(tmem_addr + head_base, accum);
                        else if (kNumHeadsPerLoad == 8)
                            tmem_load_32dp32b_x8(tmem_addr + head_base,
                                                 accum[0], accum[1], accum[2], accum[3],
                                                 accum[4], accum[5], accum[6], accum[7]);
                        else
                            tmem_load_32dp32b_x4(tmem_addr + head_base,
                                                 accum[0], accum[1], accum[2], accum[3]);
                        fence_view_async_tmem_load();
                        // Release the TMEM stage after the last valid token's last head chunk.
                        if (q_token_offset + 1 == num_q_tokens && head_base + kNumHeadsPerLoad == kNumHeads) {
                            tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tp.stage].arrive();
                        }
                        #pragma unroll
                        for (uint32_t head_offset = 0; head_offset < kNumHeadsPerLoad; head_offset += 4) {
                            const uint32_t relu_0 = cvt_relu_bf16x2_f32(accum[head_offset], accum[head_offset + 1]);
                            const uint32_t relu_1 = cvt_relu_bf16x2_f32(accum[head_offset + 2], accum[head_offset + 3]);
                            sum_0 = fma_bf16x2(relu_0, weights[q_token_offset][(head_base + head_offset) / 2], sum_0);
                            sum_1 = fma_bf16x2(relu_1, weights[q_token_offset][(head_base + head_offset + 2) / 2], sum_1);
                        }
                    }
                    // Horizontal reduce: (h0+h1+..) pairs then both halves.
                    const uint32_t sum = add_bf16x2(sum_0, sum_1);
                    const uint32_t reduced = add_bf16x2(low2_bf16x2(sum), high2_bf16x2(sum));

                    // Map split-local slots back to compressed-logits columns
                    const uint32_t q_slot_offset =
                        (packed_slot_offsets >> (q_token_offset * kSparseSlotBits)) & kSparseInvalidSlot;
                    if (q_slot_offset != kSparseInvalidSlot) {
                        const uint32_t q_slot_base = q_token_offset == 0 ? q0_slot_base : q1_slot_base;
                        const uint32_t output_col_idx =
                            (q_slot_base + q_slot_offset) * SPARSE_BLOCK_KV + token_in_kv_block;
                        st_global_u16(&output_rows[q_token_offset][output_col_idx], reduced);
                    }
                }
            }
            fence_view_async_shared();
            smem.empty_q_barriers[qsp.stage].arrive();
        }
        named_barrier_sync(kNumMathThreads, 0);
        if (warp_idx == 0)
            tmem_dealloc_1sm(0, kNumTmemCols);
    } else {
        // Idle control warps still donate their registers to the math WGs.
        setmaxnreg_dec<kNumControlRegisters>();
    }
#else
    // Pre-SM100 targets are not supported by the sparse MQA pipeline.
#endif
}

// __global__ entry points (instantiated by the Rust-side wrapper bodies).

// Contiguous (flat) KV cache: kv = [num_kv_tokens][head_dim] + sf_kv =
// [num_kv_tokens] packed UE8M0 words.
template <uint32_t kNumHeads, uint32_t SPARSE_BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages, uint32_t kNumMathWarpGroups, uint32_t kNumSMs, uint32_t BLOCK_Q,
          bool kUseUnalignedKs, bool kIsMXFP4>
DG_GLOBAL void __launch_bounds__((kNumMathWarpGroups + 1 + kNumMathWarpGroups / 4) * 128, 1)
sparse_mqa_logits_impl(const uint32_t logits_stride, const uint32_t num_kv_tokens, bf16_raw* logits,
                       const uint8_t* kv, const uint32_t* sf_kv, const uint8_t* metadata,
                       const __grid_constant__ TmaMap tensor_map_q,
                       const __grid_constant__ TmaMap tensor_map_sf_q,
                       const __grid_constant__ TmaMap tensor_map_weights,
                       const __grid_constant__ TmaMap tensor_map_kv,
                       const __grid_constant__ TmaMap tensor_map_sf_kv) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    // Keep the dtype flag out of the KV accessor's type where possible; the
    // accessor is built here and passed by const reference to the core.
    const ContiguousSparseKVAccessor<SPARSE_BLOCK_KV, kIsMXFP4> kv_accessor(
        kv, sf_kv, num_kv_tokens, &tensor_map_kv, &tensor_map_sf_kv);
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    sparse_mqa_logits_core_impl<kNumHeads, SPARSE_BLOCK_KV, kNumQStages, kNumKVStages,
                                kNumTmemStages, kNumMathWarpGroups, kNumSMs, BLOCK_Q,
                                kUseUnalignedKs, kIsMXFP4>(
        logits_stride, logits, metadata, tensor_map_q, tensor_map_sf_q, tensor_map_weights,
        smem_buffer, kv_accessor);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("trap;");
#endif
}

// Paged fused KV cache: each page is [PAGE_KV tokens of KV bytes][PAGE_KV SF
// words]; `kv_page_stride_bytes` (multiple of 512) separates pages.
template <uint32_t kNumHeads, uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV, uint32_t kNumQStages,
          uint32_t kNumKVStages, uint32_t kNumTmemStages, uint32_t kNumMathWarpGroups,
          uint32_t kNumSMs, uint32_t BLOCK_Q, bool kIsMXFP4>
DG_GLOBAL void __launch_bounds__((kNumMathWarpGroups + 1 + kNumMathWarpGroups / 4) * 128, 1)
sparse_mqa_logits_paged_impl(const uint32_t logits_stride, const uint32_t kv_page_stride_bytes,
                             bf16_raw* logits, const uint8_t* fused_kv_cache, const uint8_t* metadata,
                             const __grid_constant__ TmaMap tensor_map_q,
                             const __grid_constant__ TmaMap tensor_map_sf_q,
                             const __grid_constant__ TmaMap tensor_map_weights) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    DG_STATIC_ASSERT(PAGE_KV % SPARSE_BLOCK_KV == 0, "Sparse KV blocks must not cross pages");
    const PagedSparseKVAccessor<PAGE_KV, SPARSE_BLOCK_KV, kIsMXFP4> kv_accessor(
        fused_kv_cache, kv_page_stride_bytes);
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    sparse_mqa_logits_core_impl<kNumHeads, SPARSE_BLOCK_KV, kNumQStages, kNumKVStages,
                                kNumTmemStages, kNumMathWarpGroups, kNumSMs, BLOCK_Q,
                                false, kIsMXFP4>(
        logits_stride, logits, metadata, tensor_map_q, tensor_map_sf_q, tensor_map_weights,
        smem_buffer, kv_accessor);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("trap;");
#endif
}

} // namespace dg
