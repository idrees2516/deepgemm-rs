// ===========================================================================
// bmk_bnk.cu — batched "bmk, bnk -> mn" BF16 GEMM (MLA projections) and the
//              PsumLayout grouped-GEMM scheduler + kernel (SM90a + SM100a).
//
// Port of upstream DeepGEMM:
//   * impls/sm100_bmk_bnk_mn.cuh -> dg::bmk_bnk_mn_sm100_impl  (tcgen05)
//   * impls/sm90_bmk_bnk_mn.cuh  -> dg::bmk_bnk_mn_sm90_impl   (wgmma)
//   * scheduler/gemm.cuh         -> dg::psum::Scheduler (full port, all
//     GemmType branches incl. the two *WithPsumLayout variants)
//   * csrc/jit_kernels/impls/sm{90,100}_bmk_bnk_mn.hpp -> the launch configs
//     mirrored by the tile choosers in src/api_bmk.rs
// plus dg::gemm_psum_impl: a persistent SM100 BF16 grouped GEMM that consumes
// the psum scheduler for both
//   GemmType::MGroupedContiguousWithPsumLayout and
//   GemmType::KGroupedContiguousWithPsumLayout.
// ===========================================================================
//
// ---------------------------------------------------------------------------
// 1. WHAT THE "bmk, bnk -> mn" EINSUM IS (MLA attention projections)
// ---------------------------------------------------------------------------
// Multi-head Latent Attention projects a shared latent cache through two
// per-head/head-group matrices.  When those projections are gathered over a
// batch of sequences the algebra looks like:
//
//     A: [S, M, K]   the "q-side" projection  (S = batch/sequences,
//     B: [S, N, K]   the "kv-side" projection   M/N = head-dim spans,
//     D: [M, N]                                        K = latent head dim)
//
//         D[m, n]  =  sum_s  sum_k  A[s, m, k] * B[s, n, k]
//
// i.e. a batch-reduced cross-projection gram: the ENTIRE (S x K) product space
// is the reduction axis of one [M, N] output.  Exactly the upstream Python
// `deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=d)` (tests/test_einsum.py).
//
// Because M and N are small (128..384 in the upstream tests — a couple of
// head-group tiles) while S*K is huge (up to 8192 * 384), the ONLY way to feed
// the GPU is **split-K over the (S, K/BLOCK_K) slice space**:
//
//     grid = (num_mn_blocks) * (num_sk_blocks / kSplitFactor)
//
//   * `mn_block_idx = blockIdx.x % num_mn_blocks` picks the output tile
//     (m_block, n_block);  `sk_block_idx = blockIdx.x / num_mn_blocks` picks
//     which chunk of the reduction this CTA owns.
//   * The linear slice counter `s` enumerates S*(K/BLOCK_K) slices:
//     slice index `sk = (sk_block_idx * kSplitFactor + s)`; the physical
//     coordinates are  k_idx = sk % SHAPE_K,  s_idx = sk / SHAPE_K
//     — batch and K are folded into ONE axis, so the tail of one batch flows
//     into the head of the next with zero special-casing.
//   * Every CTA computes a FP32 PARTIAL SUM of its slices and accumulates it
//     into the single global D:
//       SM100: TMA `cp.reduce.async.bulk.tensor.2d...add` — the tensor
//              memory epilogue stages the tile in swizzled SMEM and the TMA
//              unit itself performs the read-modify-write (deterministic,
//              128B-atom granularity).
//       SM90:  `red.global.add.v2.f32` (atomicAdd(float2*)) straight out of
//              the wgmma accumulator registers.
//     => D is read-modify-write: the caller owns its initialization (upstream
//     passes c == d; a BF16 output is produced through a zeroed FP32
//     workspace that is cast at the end).
//
// ---------------------------------------------------------------------------
// 2. THE TWO KERNEL BODIES (warp roles)
// ---------------------------------------------------------------------------
// SM100 (128 threads, one (mn, sk) block per CTA — upstream layout):
//   warp 0 : TMA producer — for each of its `num_total_stages` slices: wait
//            empty[stage] (phase ^ 1), issue the A and B 128B-swizzled boxes
//            at (k_idx, m_idx + s_idx*SHAPE_M) / (k_idx, n_idx + s_idx*SHAPE_N),
//            arrive_and_expect_tx(A+B bytes).
//   warp 1 : MMA issue (elect_one) + barrier init. `tcgen05.mma.kind::f16`
//            (UMMA_M=128, UMMA_N=128, UMMA_K=16): per-lane stage-descriptor
//            table (lane i precomputes desc.lo + i*stage_bytes/16, shfl by
//            stage), 4 UMMA steps per BLOCK_K=64, scale_c = (s>0 || k>0).
//            tcgen05.commit -> empty[stage]; after the K loop one final commit
//            -> tmem_full.
//   warp 2 : tcgen05.alloc of the TMEM accumulator columns.
//   ALL 4 warps then run the epilogue (barrier-synchronized, bmk-style):
//   wait tmem_full -> for each of the BLOCK_N/STORE_BLOCK_N = 4 swizzle atoms:
//   tma_store_wait<1> + bar.sync(128) -> 8x (tcgen05.ld.32dp32b.x4 ->
//   swizzled st.shared.v4) -> tma_store_fence + bar.sync(128) -> TMA
//   reduce-add of the [32, 128] atom -> commit_group.  warp 1 deallocs TMEM.
//
// SM90 (384 threads = 128 TMA + 256 math):
//   warp 8 : TMA producer (elect_one; warps 8-11 reg-dealloc to 40).
//   warps 0-7: two math warpgroups (reg-alloc 232).  Per slice: wait
//            full[stage], 4x wgmma.mma_async.m64n128k16.f32.bf16.bf16 (SS
//            descriptors, B128 layout, SBO=1024), commit, wait<0>, arrive
//            empty[stage] (init count 256).  Epilogue: each lane owns rows
//   (warp*16 + lane/4) and (+8), columns (lane%4)*2 + 8i — atomicAdd(float2)
//   into D.
//
// ---------------------------------------------------------------------------
// 3. THE PSUM ("partial-sum") LAYOUT CONTRACT
// ---------------------------------------------------------------------------
// Grouped GEMMs in DeepGEMM describe their ragged group boundaries with an
// int32 `grouped_layout` device buffer.  Two encodings exist per axis:
//
//   * plain:   layout is per-ROW (m-grouped) or per-GROUP K SIZE (k-grouped),
//              group starts are exactly aligned to the layout alignment.
//   * PSUM:    layout[g] is a *prefix sum with unaligned ends*:
//                end_g = align(end_{g-1}, ALIGNMENT) + real_size_g
//              i.e. the buffer is physically padded per group (the padding
//              must be zero for "ensure zero padding" callers) and the value
//              stored is the *end offset* (the partial sum), not the size.
//
//   MGroupedContiguousWithPsumLayout (activation-grad / MoE, masked-to-psum):
//       A: [M_psum, K] K-major (M_psum = physical padded span; rows in
//          [psum_m_g, align(psum_m_g, BLOCK_M)) are padding/gaps),
//       B: [num_groups, N, K] K-major,
//       D: [M_psum, N],
//       grouped_layout[g] = cumulative END of group g's rows (unaligned).
//     The scheduler walks groups, deriving each group's m-block range as
//     [last_psum_m/BLOCK_M, ceil(end_g/BLOCK_M)) and offsetting m_block_idx
//     into ABSOLUTE block space (m_block_idx += last_psum_m / BLOCK_M).
//
//   KGroupedContiguousWithPsumLayout (weight-grad over ragged experts):
//       A: [SUM_K, M] MN-major, B: [SUM_K, N] MN-major (SUM_K = physical
//          padded span; per-group K tails are zero),
//       D: [num_groups, M, N] (+ optional accumulation),
//       grouped_layout[g] = cumulative END of group g's K (unaligned).
//     Per group: physical K start = align(end_{g-1}, kKAlignment),
//     logical K = end_g - align(end_{g-1}, kKAlignment).  The k-block loop
//     runs over that logical span; the zero tail makes partial BLOCK_K tiles
//     exact.
//
// The scheduler port below (dg::psum::Scheduler) is a faithful copy of
// upstream scheduler/gemm.cuh — every GemmType branch — so the CPU models in
// tests/bmk_psum_cpu.rs can be checked against the same formulas.
//
// ---------------------------------------------------------------------------
// 4. gemm_psum_impl — the psum scheduler's consumer (SM100)
// ---------------------------------------------------------------------------
// A focused persistent BF16 grouped GEMM (tcgen05) that exercises both psum
// variants end-to-end:
//   256 threads: warp 0 TMA producer, warp 1 MMA issue, warp 2 TMEM alloc,
//   warps 4-7 epilogue warpgroup (STSM + TMA store/reduce-add).
//   Tiling is fixed at BLOCK_M = BLOCK_N = 128, BLOCK_K = 64 (the BF16
//   canonical tile); no multicast, no swap-AB:
//     * m-grouped-psum upstream always uses swap-AB (tiny per-group M);
//       this port keeps plain-AB with BLOCK_M = 128 — a documented deviation
//       (see the notes at gemm_psum_impl) — because the swap-AB epilogue
//       machinery lives in gemm_sm100.cu, which this port may not touch.
//       Correctness is identical: padding rows of A must be zeroed by the
//       caller (the upstream contiguous-psum contract; the masked-psum flavor
//       leaves them uninitialized and simply never checks the gap rows).
//   TMEM: single accumulator (kNumEpilogueStages = 1) of UMMA_N columns;
//   MMA of block i+1 waits the epilogue's tmem_empty arrival of block i.
// ---------------------------------------------------------------------------

namespace dg {

// ===========================================================================
// Local PTX helpers (this file only — prelude.h is frozen)
// ===========================================================================

// Unconditional trap: the zero-include stand-in for DG_TRAP_ONLY_DEVICE_ASSERT
// (device assert() needs CUDA headers; `trap` is the underlying PTX).
DG_DEVICE void dg_trap() { asm volatile("trap;"); }

// 8-byte atomic FP32 add pair (SM90+): the NVRTC-free form of
// `atomicAdd(float2*, float2)` — `red` (no returned value), vector .v2.f32,
// which ptxas accepts for sm_90a and sm_100a (probed in scripts/verify_bmk.py
// era; CUDA's own atomicAdd(float2*) lowers to the same vector atom).
DG_DEVICE void red_global_add_v2_f32(void* p, float a, float b) {
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};"
                 :: "l"(p), "f"(a), "f"(b) : "memory");
}

// TMA operand-tile load with swizzle-atom splitting (port of the relevant
// slice of deep_gemm/common/tma_copy.cuh `copy_nd`, 2-D, no multicast).
//   K-major  tile: inner = BLOCK_K elements  (one 128B atom for bf16),
//   MN-major tile: inner = BLOCK_MN elements (two 128B atoms for 128 bf16).
// Consecutive atoms land BLOCK_OUTER * atom_bytes apart in SMEM — the layout
// the UMMA/GMMA descriptors' LBO/SBO encoding expects.
template <bool kIsKMajor>
DG_DEVICE void bmk_tma_load_tile(const TmaMap* map, Barrier* bar, uint8_t* smem,
                                 uint32_t inner_idx, uint32_t outer_idx,
                                 uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
                                 uint32_t kSwizzleMode, uint32_t kElemSize) {
    const uint32_t inner_bytes = BLOCK_INNER * kElemSize;
    const uint32_t atom_bytes = kSwizzleMode == 0 ? inner_bytes : kSwizzleMode;
    const uint32_t num_atoms = inner_bytes / atom_bytes;
    const uint32_t atom_elems = atom_bytes / kElemSize;
    #pragma unroll 4
    for (uint32_t i = 0; i < num_atoms; ++i) {
        tma_load_2d(map, bar, smem + i * BLOCK_OUTER * atom_bytes,
                    kEvictNormalHint, inner_idx + i * atom_elems, outer_idx);
    }
}

// ===========================================================================
// dg::psum — the PsumLayout scheduler (full port of scheduler/gemm.cuh)
// ===========================================================================
// NOTE ON ENUM NUMBERING: upstream `deep_gemm::GemmType` numbers the psum
// variants 5/6; the repo's prelude.h deliberately re-used 5 for its SM90
// K-grouped flavor.  This local copy keeps the UPSTREAM numbering so the CPU
// models in tests/bmk_psum_cpu.rs can be diffed line-by-line against
// scheduler/gemm.cuh.  It lives in its own namespace to avoid clashing with
// prelude's `dg::GemmType` / `dg::Scheduler`.
// ===========================================================================
namespace psum {

enum class GemmType : uint32_t {
    Normal                           = 0,
    MGroupedContiguous               = 1,
    MGroupedMasked                   = 2,
    KGroupedContiguous               = 3,
    Batched                          = 4,
    MGroupedContiguousWithPsumLayout = 5,
    KGroupedContiguousWithPsumLayout = 6,
};

constexpr DG_DEVICE bool is_m_grouped_contiguous(GemmType t) {
    return t == GemmType::MGroupedContiguous || t == GemmType::MGroupedContiguousWithPsumLayout;
}
constexpr DG_DEVICE bool is_k_grouped_contiguous(GemmType t) {
    return t == GemmType::KGroupedContiguous || t == GemmType::KGroupedContiguousWithPsumLayout;
}

enum class IndexType : uint32_t { MN = 0, K = 1, SF_K = 2 };

// L2-swizzle group size: minimize (group footprint) — pick 8 or 16.
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs,
          bool kIsMulticastOnA>
constexpr DG_DEVICE uint32_t get_num_1d_blocks_per_group() {
    uint32_t num_best_blocks = 0, min_usage = 0xffffffffu;
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t candidate = i == 0 ? 8u : 16u;
        const uint32_t usage = kIsMulticastOnA
            ? candidate * BLOCK_N + ceil_div_u32(kNumSMs, candidate) * BLOCK_M  // grouping on N
            : candidate * BLOCK_M + ceil_div_u32(kNumSMs, candidate) * BLOCK_N; // grouping on M
        if (usage < min_usage) {
            min_usage = usage;
            num_best_blocks = candidate;
        }
    }
    return num_best_blocks;
}

template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kEnsureZeroPadding = true,
          uint32_t kKAlignment = 128u,    // psum k-group start alignment
          uint32_t kSFKSpan = 128u,       // K covered by one k-grouped SF row
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
    // A/B group starts must be aligned to whole K blocks. SF rows are packed
    // independently per group and tracked by `current_sf_k_cumsum`.
    DG_STATIC_ASSERT(!is_k_grouped_contiguous(kGemmType) || kKAlignment % 128 == 0,
                     "K alignment must be a multiple of BLOCK_K (128)");

    int current_iter = -1;

    // Block configs (upstream leaves `num_blocks` unset for the masked /
    // psum-m flavors, which derive it on the fly; default-zeroed here — the
    // psum branches never read it before writing).
    uint32_t num_blocks = 0;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group = 0;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m = 0, current_m_block_cumsum = 0;
    // Only used for k-grouped layout.  `current_k_start` is the current
    // group's physical K start offset (always a multiple of `kKAlignment`),
    // maintained by both psum and non-psum paths.
    uint32_t current_shape_k, current_k_start = 0, current_sf_k_cumsum = 0;
    // Only used for `KGroupedContiguousWithPsumLayout`
    uint32_t current_k_end = 0;

    // Load the K-group selected by `current_group_idx`.
    DG_DEVICE void get_next_k_group() {
        if (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            // `grouped_layout[i]` is the psum end offset in K elements.
            const uint32_t next_k_end = (uint32_t)grouped_layout[current_group_idx];
            current_k_start = align_u32(current_k_end, kKAlignment);
            current_shape_k = next_k_end - current_k_start;
            current_k_end = next_k_end;
        } else {
            current_k_start += current_shape_k;
            current_shape_k = (uint32_t)grouped_layout[current_group_idx];
        }
    }

    DG_DEVICE Scheduler(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        int* grouped_layout_) : grouped_layout(grouped_layout_) {
        num_m_blocks = ceil_div_u32(shape_m, BLOCK_M);
        num_n_blocks = ceil_div_u32(shape_n, BLOCK_N);
        current_shape_k = is_k_grouped_contiguous(kGemmType) ? 0 : shape_k;
        if (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::MGroupedMasked) {
            // num_blocks derived on the fly in get_next_block
        } else if (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            current_psum_m = (uint32_t)grouped_layout[0];
            num_m_blocks = ceil_div_u32(current_psum_m, BLOCK_M);
        } else {  // k-grouped (plain + psum)
            num_blocks = num_m_blocks * num_n_blocks;
            get_next_k_group();
        }
    }

    DG_DEVICE void get_swizzled_block_idx(uint32_t block_idx, uint32_t& m_block_idx,
                                          uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const uint32_t primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const uint32_t secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const uint32_t num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const uint32_t group_idx = block_idx / num_blocks_per_group;
        uint32_t first_block_idx = group_idx * kNum1DBlocksPerGroup;
        uint32_t in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = dg_min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast (SM90 only: SM90 can dynamically
        // disable TMA multicast while SM100 uses 2-CTA and cannot).
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)) || !defined(__CUDA_ARCH__)
        if (kNumMulticast > 1 && (num_blocks_in_group & 1u) != 0u) {
            if (in_group_idx < (num_blocks_in_group ^ 1u) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1u;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1u) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1u;
                num_blocks_in_group = 1;
            }
        }
#endif

        // `kIsMulticastOnA == true` leads to groups on N
        if (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    DG_DEVICE uint32_t get_global_idx(uint32_t shape_dim, uint32_t block_size,
                                      uint32_t block_idx, uint32_t m_block_idx = 0) {
        if (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::MGroupedContiguous) {
            const int offset = kWithGroupOffset ? dg_max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return (uint32_t)offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::MGroupedMasked ||
                   kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const uint32_t offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if (is_k_grouped_contiguous(kGemmType)) {
            uint32_t offset = 0;
            if (kWithGroupOffset) {
                if (kIndexType == IndexType::MN) {
                    offset = current_group_idx * shape_dim;
                } else if (kIndexType == IndexType::K) {
                    offset = current_k_start;
                } else {  // SF_K
                    offset = current_sf_k_cumsum;
                }
            }
            return offset + block_idx * block_size;
        } else {  // Batched: ignore kWithGroupOffset, apply it for SF_K only
            const uint32_t offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only: the effective (16-aligned) M rows of
    // the current block — the LAST m-block of an m-grouped-psum group is
    // partial (psum ends are unaligned).
    DG_DEVICE uint32_t get_aligned_effective_m_in_block(uint32_t m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if (kGemmType == GemmType::MGroupedContiguousWithPsumLayout && !kEnsureZeroPadding)
            return align_u32(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1
                                 ? current_psum_m - m_block_idx * BLOCK_M
                                 : BLOCK_M,
                             UMMA_STEP_N);
        return BLOCK_M;
    }

    DG_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t next_block_idx = (uint32_t)(++current_iter) * kNumSMs + blockIdx.x;

        if (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;
                // Within current group
                num_m_blocks = ceil_div_u32((uint32_t)grouped_layout[current_group_idx], BLOCK_M);
                const uint32_t cur_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < cur_m_block_cumsum * num_n_blocks)
                    break;
                // Move to check the next group
                current_group_idx++;
                current_m_cumsum = cur_m_block_cumsum;
            }
            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks,
                                   m_block_idx, n_block_idx);
        } else if (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;
                // Move to check the next group
                if (++current_group_idx == kNumGroups)
                    return false;
                // `num_m_blocks` varies with the increase of the group index
                last_psum_m = align_u32(current_psum_m, BLOCK_M);
                current_psum_m = (uint32_t)grouped_layout[current_group_idx];
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = ceil_div_u32(current_psum_m - last_psum_m, BLOCK_M);
            }
            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks,
                                   m_block_idx, n_block_idx);
            // `last_psum_m` is aligned with block M
            m_block_idx += last_psum_m / BLOCK_M;
        } else if (is_k_grouped_contiguous(kGemmType)) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;
                // Within current group
                if (next_block_idx < (current_group_idx + 1) * num_blocks)
                    break;
                // Move to check the next group
                current_group_idx++;
                if (current_group_idx >= kNumGroups)
                    return false;
                const uint32_t aligned_shape_k = align_u32(current_shape_k, kKAlignment);
                current_sf_k_cumsum += ceil_div_u32(aligned_shape_k, kSFKSpan);
                get_next_k_group();
            }
            get_swizzled_block_idx(next_block_idx - current_group_idx * num_blocks,
                                   m_block_idx, n_block_idx);
        } else if (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;
            current_group_idx = next_block_idx / num_blocks;
            const uint32_t block_idx = next_block_idx - current_group_idx * num_blocks;
            if (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {  // Normal / MGroupedContiguous
            if (next_block_idx >= num_blocks)
                return false;
            // SM90 only: peer CTA shares this m-block when its swizzled
            // block is in bound (multicast validity).
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 ||
                                num_m_blocks % kNumMulticast == 0 ||
                                (next_block_idx ^ 1u) < num_blocks;
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // For SM90 only
    DG_DEVICE bool is_tma_multicast_valid(uint32_t m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if (kGemmType == GemmType::Normal || kGemmType == GemmType::MGroupedMasked ||
            is_k_grouped_contiguous(kGemmType) || kGemmType == GemmType::Batched ||
            kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if (kIsMulticastOnA) {
                return true;
            } else {
                const int group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const int peer_group_idx = grouped_layout[(m_block_idx ^ 1u) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    DG_DEVICE bool is_computation_valid(uint32_t m_block_idx, uint32_t m_offset) const {
        if (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched) {
            return true;
        } else if (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < (uint32_t)grouped_layout[current_group_idx];
        } else if (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            return true;  // k-grouped (unreachable upstream; kept total here)
        }
    }
};

}  // namespace psum

// ===========================================================================
// bmk_bnk_mn_sm100_impl — SM100 (Blackwell) "bmk, bnk -> mn" split-K GEMM.
//
// Port of upstream `sm100_bmn_bnk_mn_gemm_impl` (impls/sm100_bmk_bnk_mn.cuh).
// Template contract (mirrors the upstream static asserts):
//   BLOCK_M == 128 (LAYOUT_AD_M), BLOCK_N == 128, BLOCK_K == 64,
//   kSwizzleABMode == kSwizzleCDMode == 128, kNumThreads == 128.
// SHAPE_M/N/K are compile-time (upstream compiles the shapes into the JIT
// variant); `shape_s` (the batch) stays a runtime argument.
// ===========================================================================
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumThreads>
DG_GLOBAL void __launch_bounds__(kNumThreads, 1)
bmk_bnk_mn_sm100_impl(uint32_t shape_s,
                      const __grid_constant__ TmaMap tensor_map_a,
                      const __grid_constant__ TmaMap tensor_map_b,
                      const __grid_constant__ TmaMap tensor_map_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumTMAStoreStages = 2;

    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();
    DG_STATIC_ASSERT(BLOCK_M == LAYOUT_AD_M && BLOCK_N == 128 && BLOCK_K == 64,
                     "Invalid block size");
    DG_STATIC_ASSERT(kSwizzleABMode == 128 && kSwizzleCDMode == 128, "Invalid swizzle mode");
    DG_STATIC_ASSERT(kNumThreads == 128, "Invalid thread count");

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * 2;  // bf16
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * 2;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        prefetch_tma_map(&tensor_map_d);
    }

    // Real tensor memory size and offsets
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<BLOCK_N>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols && kNumTmemCols <= 512, "Invalid tensor memory columns");

    // SMEM pointers (upstream uses PatternVisitor lambdas)
    auto smem_cd = [&](uint32_t i) { return (uint8_t*)(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); };
    auto smem_a = [&](uint32_t i) {
        return (uint8_t*)(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    };
    auto smem_b = [&](uint32_t i) {
        return (uint8_t*)(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
                          i * SMEM_B_SIZE_PER_STAGE);
    };

    // Barriers: full[kNumStages], empty[kNumStages], tmem_full, then the
    // 4-byte TMEM base-pointer slot.
    Barrier* barrier_start = (Barrier*)(smem_buffer + SMEM_CD_SIZE +
                                        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers = [&](uint32_t i) { return barrier_start + i; };
    auto empty_barriers = [&](uint32_t i) { return barrier_start + (kNumStages + i); };
    Barrier* tmem_full_barrier = barrier_start + kNumStages * 2;
    uint32_t* tmem_ptr_in_smem = (uint32_t*)(barrier_start + kNumStages * 2 + 1);

    // Initialize barriers
    if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers(i)->init(1);
            empty_barriers(i)->init(1);
        }
        tmem_full_barrier->init(1);
        // Make initialized barrier visible in async proxy
        fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        tmem_alloc_1sm(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Block indices: the (S x K/BLOCK_K) slice space is one giant split-K.
    const uint32_t num_n_blocks = ceil_div_u32(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * ceil_div_u32(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages =
        dg_min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    // Wait for primary kernel completion (PDL)
    griddepcontrol_wait();

    if (warp_idx == 0) {
        // TMA load warp
        for (uint32_t s = 0; s < num_total_stages; ++s) {
            const uint32_t stage_idx = s % kNumStages;
            empty_barriers(stage_idx)->wait(((s / kNumStages) & 1) ^ 1);

            const uint32_t m_idx = BLOCK_M * m_block_idx;
            const uint32_t n_idx = BLOCK_N * n_block_idx;
            const uint32_t sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
            const uint32_t k_idx = sk_idx % SHAPE_K;
            const uint32_t s_idx = sk_idx / SHAPE_K;

            // Issue TMAs (single 128B atom: BLOCK_K=64 bf16 == the swizzle)
            if (elect_one_sync()) {
                tma_load_2d(&tensor_map_a, full_barriers(stage_idx), smem_a(stage_idx),
                            kEvictNormalHint, k_idx, m_idx + s_idx * SHAPE_M);
                tma_load_2d(&tensor_map_b, full_barriers(stage_idx), smem_b(stage_idx),
                            kEvictNormalHint, k_idx, n_idx + s_idx * SHAPE_N);
            }
            __syncwarp();

            // Arrive at full barriers
            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            if (elect_one_sync())
                full_barriers(stage_idx)->arrive_and_expect_tx(kNumArrivalBytes);
        }
    } else if (warp_idx == 1) {
        // MMA issue warp (single CTA -> every CTA is the leader)
        // Make instruction descriptor: BF16 x BF16 -> F32, UMMA_M=128 (the
        // SMEM "layout A/D" M), UMMA_N=BLOCK_N, K-major A/B operands.
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 32 / 2;  // 32 bytes / bf16
        InstrDescriptor instr_desc = make_instr_desc_f16(
            1 /* a_format: BF16 */, 1 /* b_format: BF16 */, 1 /* c_format: F32 */,
            UMMA_M, UMMA_N, MAJOR_K, MAJOR_K);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Per-lane table of stage base descriptors (lane i holds stage i).
        SmemDescriptor a_desc = make_umma_desc<MAJOR_K, BLOCK_M, BLOCK_K, kSwizzleABMode, 1, 2>(
            smem_a(0), 0, 0);
        SmemDescriptor b_desc = make_umma_desc<MAJOR_K, BLOCK_N, BLOCK_K, kSwizzleABMode, 1, 2>(
            smem_b(0), 0, 0);
        const uint32_t a_desc_lo =
            lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        const uint32_t b_desc_lo =
            lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        DG_STATIC_ASSERT((UMMA_M == 128 && UMMA_N % 16 == 0 && 16 <= UMMA_N && UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Wait tensor memory empty barrier arrival
        tcgen05_after_thread_sync();

        // Launch MMAs
        for (uint32_t s = 0; s < num_total_stages; ++s) {
            // Wait TMA arrival
            const uint32_t stage_idx = s % kNumStages;
            full_barriers(stage_idx)->wait((s / kNumStages) & 1);
            tcgen05_after_thread_sync();

            // Issue UMMAs
            const uint64_t runtime_instr_desc = make_runtime_instr_desc(instr_desc);
            const uint32_t a_desc_base_lo = __shfl_sync(0xffffffffu, a_desc_lo, (int)stage_idx);
            const uint32_t b_desc_base_lo = __shfl_sync(0xffffffffu, b_desc_lo, (int)stage_idx);
            if (elect_one_sync()) {
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                    a_desc.lo = advance_umma_desc_lo<MAJOR_K, BLOCK_M, kSwizzleABMode, 1, 2>(
                        a_desc_base_lo, 0, k * UMMA_K);
                    b_desc.lo = advance_umma_desc_lo<MAJOR_K, BLOCK_N, kSwizzleABMode, 1, 2>(
                        b_desc_base_lo, 0, k * UMMA_K);
                    mma_f16_1sm(a_desc.desc_, b_desc.desc_, 0,
                                (s > 0 || k > 0) ? 1u : 0u, runtime_instr_desc);
                }
            }
            __syncwarp();

            // Commit to the mbarrier object (tcgen05.commit implicitly
            // performs tcgen05.fence::before_thread_sync).
            umma_arrive_1sm(empty_barriers(stage_idx));
        }
        umma_arrive_1sm(tmem_full_barrier);
    }

    // TMEM allocation result check (hardware ignores the warp index bits, so
    // the allocated base must be 0; two CTAs may not share an SM's TMEM).
    if (warp_idx == 2 && ld_shared_u32(tmem_ptr_in_smem) != 0)
        dg_trap();

    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / 4;  // fp32
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / 4;
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    // Wait UMMA arrival (single use: phase 0)
    tmem_full_barrier->wait(0);
    tcgen05_after_thread_sync();

    // Load from tensor memory into registers, write shared memory (STSM),
    // one TMA reduce-add per swizzle atom — pipelined over 2 CD stages.
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
    #pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++s) {
        // Wait shared memory to be released (at most 1 store group in flight)
        if (warp_idx == 0 && elect_one_sync())
            tma_store_wait<kNumTMAStoreStages - 1>();
        __syncthreads();

        // The pipeline stage
        const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
        const uint32_t m_idx = m_block_idx * BLOCK_M;
        const uint32_t n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

        // Store into shared memory: each lane owns staging row `lane_idx`,
        // 16B bank-group `i` of that row, XOR-swizzled within the atom.
        #pragma unroll
        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
            // Index of the bank group to be written in the atom
            const uint32_t bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

            // Reshape the atom: (128, 8) -> (128 * 8 / 8, 8); "8" bank
            // groups per staging row, "16" bytes per group.  With
            // kSwizzleCDMode == 128 the shortcut holds: row = lane, col = i.
            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
            const uint32_t row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
            uint32_t col = kHasShortcut ? i : (bank_group_index % 8);
            col ^= row % (kSwizzleCDMode / 16);

            // Source and destination memory address
            const uint32_t tmem_addr = s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
            uint8_t* smem_ptr = smem_cd(tma_stage_idx)
                              + warp_idx * 32 * kSwizzleCDMode       // warp offset
                              + row * (kNumBankGroupBytes * 8)       // in-atom row
                              + col * kNumBankGroupBytes;            // in-atom bank group

            // Load from tensor memory, store into shared memory
            uint32_t values[kNumElemsPerBankGroup];
            tmem_load_32dp32b_x4(tmem_addr, values[0], values[1], values[2], values[3]);
            fence_view_async_tmem_load();
            st_shared_u32x4((uint32_t*)smem_ptr, values[0], values[1], values[2], values[3]);
        }

        // Synchronize all threads and issue TMA (D += partial)
        tma_store_fence();
        __syncthreads();
        if (warp_idx == 0 && elect_one_sync()) {
            tma_reduce_add_2d(&tensor_map_d, smem_cd(tma_stage_idx), n_idx, m_idx);
            tma_store_arrive();
        }
    }

    // Deallocate tensor memory by warp 1 (warp 0 issues the TMA stores)
    if (warp_idx == 1)
        tmem_dealloc_1sm(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        dg_trap();  // "This kernel only supports sm_100a"
#endif
}

// ===========================================================================
// bmk_bnk_mn_sm90_impl — SM90 (Hopper) "bmk, bnk -> mn" split-K GEMM.
//
// Port of upstream `sm90_bmn_bnk_mn_gemm_impl` (impls/sm90_bmk_bnk_mn.cuh):
// wgmma.mma_async.m64n128k16.f32.bf16.bf16, TMA producer warpgroup + two math
// warpgroups, register epilogue with float2 global atomics into D.
// Template contract: BLOCK_M == 128, kNumTMAThreads == 128,
// kNumMathThreads == 256, swizzle == BLOCK_K * 2 == 128B.
// ===========================================================================
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
DG_GLOBAL void __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
bmk_bnk_mn_sm90_impl(uint32_t shape_s,
                     const __grid_constant__ TmaMap tensor_map_a,
                     const __grid_constant__ TmaMap tensor_map_b,
                     float* d) {
// Upstream guards with `__CUDA_ARCH__ >= 900`; this repo narrows it to
// `&& < 1000` because wgmma.mma_async has NO SASS encoding on sm_100a and
// the two SM generations live in the SAME translation unit here (same
// convention as kernels/hc_prenorm.cu).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
    // WGMMA selector: m64 n{BLOCK_N} k16, accumulator = BLOCK_N/2 per lane.
    constexpr uint32_t WGMMA_M = 64, WGMMA_K = 16, kNumAccum = BLOCK_N / 2;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA_M == 0, "Invalid block size");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0, "Invalid WGMMA N");

    // Shared memory
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * 2;  // bf16
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * 2;

    const uint32_t warp_idx = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();
    DG_STATIC_ASSERT(BLOCK_M == 128, "Invalid block M");
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Invalid number of TMA threads");
    DG_STATIC_ASSERT(kNumMathThreads == 256, "Invalid number of math threads");

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_a = [&](uint32_t i) { return smem_buffer + i * SMEM_A_SIZE_PER_STAGE; };
    auto smem_b = [&](uint32_t i) {
        return smem_buffer + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE;
    };

    // Barriers
    Barrier* barrier_start = (Barrier*)(smem_buffer +
                                        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers = [&](uint32_t i) { return barrier_start + i; };
    auto empty_barriers = [&](uint32_t i) { return barrier_start + (kNumStages + i); };

    // Initialize barriers: every math thread releases a stage.
    if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers(i)->init(1);
            empty_barriers(i)->init(kNumMathThreads);
        }
        fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Block indices (same split-K decomposition as SM100)
    const uint32_t num_n_blocks = ceil_div_u32(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * ceil_div_u32(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages =
        dg_min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    // Wait for primary kernel completion (PDL)
    griddepcontrol_wait();

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        setmaxnreg_dec<kNumTMARegisters>();

        // NOTES: only one warp (one thread of it) is used
        if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
            // Persistently schedule over this block's split-K slices
            for (uint32_t s = 0; s < num_total_stages; ++s) {
                // Wait consumer release
                const uint32_t stage_idx = s % kNumStages;
                empty_barriers(stage_idx)->wait(((s / kNumStages) + 1) & 1);

                Barrier* fb = full_barriers(stage_idx);
                const uint32_t sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
                const uint32_t k_idx = sk_idx % SHAPE_K;
                const uint32_t s_idx = sk_idx / SHAPE_K;

                tma_load_2d(&tensor_map_a, fb, smem_a(stage_idx), kEvictNormalHint,
                            k_idx, m_block_idx * BLOCK_M + s_idx * SHAPE_M);
                tma_load_2d(&tensor_map_b, fb, smem_b(stage_idx), kEvictNormalHint,
                            k_idx, n_block_idx * BLOCK_N + s_idx * SHAPE_N);
                fb->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        setmaxnreg_inc<kNumMathRegisters>();

        // NOTES: `__shfl_sync` encourages NVCC to use unified registers
        const uint32_t math_wg_idx = __shfl_sync(0xffffffffu, threadIdx.x / 128, 0);
        float accum[kNumAccum] = {0};

        // Launch MMAs
        for (uint32_t s = 0; s < num_total_stages; ++s) {
            // Wait TMA arrivals
            const uint32_t stage_idx = s % kNumStages;
            full_barriers(stage_idx)->wait((s / kNumStages) & 1);

            // Commit WGMMA instructions
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < BLOCK_K / WGMMA_K; ++k) {
                // K-major B128 descriptors: SBO = 8 * 128B atom rows.
                GmmaDescriptor desc_a = make_gmma_desc(
                    smem_a(stage_idx) + (math_wg_idx * WGMMA_M) * BLOCK_K * 2 + k * WGMMA_K * 2,
                    GmmaLayoutType::B128, 0, 1024);
                GmmaDescriptor desc_b = make_gmma_desc(
                    smem_b(stage_idx) + k * WGMMA_K * 2, GmmaLayoutType::B128, 0, 1024);
                wgmma_bf16<BLOCK_N, 0, 0>(desc_a.desc_, desc_b.desc_, accum, 1);
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_wait_group<0>();

            // Notify barrier arrival (all 256 math threads)
            empty_barriers(stage_idx)->arrive();
        }

        // Register epilogue: each lane owns rows (warp*16 + lane/4) and
        // (row + 8), columns (lane%4)*2 + 8i of the [BLOCK_M, BLOCK_N] tile
        // (the m64nN accumulator register map; see wgmma.h's header).
        const uint32_t row = m_block_idx * BLOCK_M + warp_idx * 16 + lane_idx / 4;
        const uint32_t col = n_block_idx * BLOCK_N + (lane_idx % 4) * 2;
        #pragma unroll
        for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
            if (col + i * 8 >= SHAPE_N)
                break;
            if (row < SHAPE_M) {
                red_global_add_v2_f32(d + (row + 0) * SHAPE_N + col + i * 8,
                                      accum[i * 4 + 0], accum[i * 4 + 1]);
            }
            if (row + 8 < SHAPE_M) {
                red_global_add_v2_f32(d + (row + 8) * SHAPE_N + col + i * 8,
                                      accum[i * 4 + 2], accum[i * 4 + 3]);
            }
        }
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        dg_trap();  // "This kernel only supports sm_90a"
#endif
}

// ===========================================================================
// gemm_psum_impl — persistent SM100 BF16 grouped GEMM over the psum layouts.
//
// Consumes dg::psum::Scheduler for BOTH psum GemmTypes:
//
//   kGemmType == MGroupedContiguousWithPsumLayout:
//     A [M_psum, K] K-major, B [G, N, K] K-major (2D map, group offset in the
//     outer dim via get_global_idx<true>), D [M_psum, N] (BF16 or FP32, no
//     accumulation).  B tile of block (m,n) of group g loads at
//     (k, g * shape_n + n * BLOCK_N).
//
//   kGemmType == KGroupedContiguousWithPsumLayout:
//     A [SUM_K, M] MN-major, B [SUM_K, N] MN-major, D [G, M, N] with optional
//     accumulation (FP32 reduce-add onto D == C; direct store otherwise).
//     Both operands load at (mn, current_k_start + k * BLOCK_K); D stores at
//     3D coordinate (n, m, current_group_idx).
//
// DEVIATION (documented): upstream's m-grouped flavors always run swap-AB
// (BLOCK_M is tiny per expert; the scheduler's get_aligned_effective_m_in_block
// feeds a dynamic UMMA_N).  This port keeps plain-AB with BLOCK_M = 128 — the
// swap-AB epilogue lives in gemm_sm100.cu (frozen for this task).  Numerics
// are unchanged: A rows in [psum_m_g, align(psum_m_g, BLOCK_M)) are padding
// that the CALLER must zero for the contiguous-psum contract (they then
// produce exact zeros in D); the masked-psum flavor leaves them
// uninitialized and its D gap rows are never checked (valid_mask).
//
// Warp roles (256 threads): 0 = TMA producer, 1 = MMA issue (elect_one),
// 2 = TMEM alloc, 4-7 = epilogue warpgroup (STSM + TMA store).  Accumulator
// TMEM is single-buffered: the epilogue arrives tmem_empty (128 arrivals)
// after its last tcgen05.ld of a block; the MMA warp waits it before the
// first UMMA of every subsequent block (phase = block counter & 1).
// ===========================================================================
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumThreads,
          psum::GemmType kGemmType,
          bool kWithAccumulation, bool kCdIsFloat,
          uint32_t kKAlignment, uint32_t kNumSMs, bool kEnsureZeroPadding>
DG_GLOBAL void __launch_bounds__(kNumThreads, 1)
gemm_psum_impl(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
               int* grouped_layout,
               const __grid_constant__ TmaMap tensor_map_a,
               const __grid_constant__ TmaMap tensor_map_b,
               const __grid_constant__ TmaMap tensor_map_cd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    static_assert(sizeof(float) == 4, "");
    constexpr bool kIsMGroupedPsum = kGemmType == psum::GemmType::MGroupedContiguousWithPsumLayout;
    constexpr bool kIsKGroupedPsum = kGemmType == psum::GemmType::KGroupedContiguousWithPsumLayout;
    DG_STATIC_ASSERT(kIsMGroupedPsum || kIsKGroupedPsum, "gemm_psum_impl: psum GemmTypes only");
    DG_STATIC_ASSERT(kCdIsFloat || !kWithAccumulation,
                     "BF16 output supports no TMA reduce-add accumulation");
    DG_STATIC_ASSERT(BLOCK_M == 128 && BLOCK_N == 128 && BLOCK_K == 64, "Invalid block size");
    DG_STATIC_ASSERT(kSwizzleABMode == 128 && kSwizzleCDMode == 128, "Invalid swizzle mode");
    DG_STATIC_ASSERT(kNumThreads == 256, "Invalid thread count");
    DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
    DG_STATIC_ASSERT(kKAlignment % 128 == 0, "k-group starts must be 128-aligned");

    // ---- MMA geometry ----
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 32 / 2;  // 32B of bf16 per MMA step
    // m-grouped: A/B both K-major; k-grouped: both MN-major (upstream twins).
    constexpr uint32_t kMajorA = kIsKGroupedPsum ? MAJOR_MN : MAJOR_K;
    constexpr uint32_t kMajorB = kIsKGroupedPsum ? MAJOR_MN : MAJOR_K;
    constexpr uint32_t kCdElemSize = kCdIsFloat ? 4u : 2u;

    // ---- Epilogue geometry ----
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M;                 // one M wave
    constexpr uint32_t kNumUMMAStoreThreads = 128;              // warps 4-7
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / kCdElemSize;
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;

    // ---- TMEM ----
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<UMMA_N>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols && kNumTmemCols <= 512, "Invalid tensor memory columns");

    // ---- Shared storage ----
    constexpr uint32_t kAStageBytes = BLOCK_M * BLOCK_K * 2;
    constexpr uint32_t kBStageBytes = BLOCK_N * BLOCK_K * 2;
    constexpr uint32_t kCdStageBytes = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t kNumTMABytesPerStage = kAStageBytes + kBStageBytes;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_cd = [&](uint32_t i) { return smem_buffer + i * kCdStageBytes; };
    auto smem_a = [&](uint32_t i) {
        return smem_buffer + kNumTMAStoreStages * kCdStageBytes + i * kAStageBytes;
    };
    auto smem_b = [&](uint32_t i) {
        return smem_buffer + kNumTMAStoreStages * kCdStageBytes + kNumStages * kAStageBytes +
               i * kBStageBytes;
    };
    Barrier* barrier_start = (Barrier*)(smem_buffer + kNumTMAStoreStages * kCdStageBytes +
                                        kNumStages * (kAStageBytes + kBStageBytes));
    auto full_barriers = [&](uint32_t i) { return barrier_start + i; };
    auto empty_barriers = [&](uint32_t i) { return barrier_start + (kNumStages + i); };
    Barrier* tmem_full_barrier = barrier_start + kNumStages * 2;
    Barrier* tmem_empty_barrier = barrier_start + kNumStages * 2 + 1;
    uint32_t* tmem_ptr_in_smem = (uint32_t*)(barrier_start + kNumStages * 2 + 2);

    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        prefetch_tma_map(&tensor_map_cd);
    }

    // Initialize barriers + allocate TMEM
    if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers(i)->init(1);
            empty_barriers(i)->init(1);
        }
        tmem_full_barrier->init(1);
        tmem_empty_barrier->init(kNumUMMAStoreThreads);  // every epilogue thread
        fence_barrier_init();
    } else if (warp_idx == 2) {
        tmem_alloc_1sm(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Wait for primary kernel completion (PDL)
    griddepcontrol_wait();

    // Block scheduler: every thread keeps its own (deterministic) copy.
    using Sched = psum::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups,
                                  /*kNumMulticast=*/1, /*kIsMulticastOnA=*/false, kNumSMs,
                                  kEnsureZeroPadding, kKAlignment>;
    Sched scheduler(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline phases run across blocks (never reset), like upstream bf16.
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == 0 && elect_one_sync()) {
        // ================= TMA load warp =================
        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // For k-grouped layout, the number of block K is variable; empty
            // groups still run ONE (zero-filled) block so every warp's ring
            // stays in lockstep (upstream `max(1, ...)`).
            const uint32_t num_total_k_blocks = dg_max(1u, ceil_div_u32(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
                 advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers(stage_idx)->wait(phase ^ 1);

                // Offsets: the group is always concatenated with the outer dim.
                constexpr bool kAWithGroupMNOffset = false;  // A's MN is the flat axis
                constexpr bool kBWithGroupMNOffset = kIsMGroupedPsum;  // B is per-group
                const uint32_t m_idx = scheduler.template get_global_idx<
                    kAWithGroupMNOffset, psum::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                const uint32_t n_idx = scheduler.template get_global_idx<
                    kBWithGroupMNOffset, psum::IndexType::MN>(shape_n, BLOCK_N, n_block_idx,
                                                            m_block_idx);
                constexpr bool kAWithGroupKOffset = (kMajorA == MAJOR_MN);
                constexpr bool kBWithGroupKOffset = (kMajorB == MAJOR_MN);
                const uint32_t k_a_idx = scheduler.template get_global_idx<
                    kAWithGroupKOffset, psum::IndexType::K>(shape_k, BLOCK_K, k_block_idx,
                                                           m_block_idx);
                const uint32_t k_b_idx = scheduler.template get_global_idx<
                    kBWithGroupKOffset, psum::IndexType::K>(shape_k, BLOCK_K, k_block_idx,
                                                           m_block_idx);

                // Issue TMAs (atom-split: K-major = 1 atom, MN-major = 2)
                if (kMajorA == MAJOR_K) {
                    bmk_tma_load_tile<true>(&tensor_map_a, full_barriers(stage_idx),
                                            smem_a(stage_idx), k_a_idx, m_idx,
                                            BLOCK_K, BLOCK_M, kSwizzleABMode, 2);
                } else {
                    bmk_tma_load_tile<false>(&tensor_map_a, full_barriers(stage_idx),
                                             smem_a(stage_idx), m_idx, k_a_idx,
                                             BLOCK_M, BLOCK_K, kSwizzleABMode, 2);
                }
                if (kMajorB == MAJOR_K) {
                    bmk_tma_load_tile<true>(&tensor_map_b, full_barriers(stage_idx),
                                            smem_b(stage_idx), k_b_idx, n_idx,
                                            BLOCK_K, BLOCK_N, kSwizzleABMode, 2);
                } else {
                    bmk_tma_load_tile<false>(&tensor_map_b, full_barriers(stage_idx),
                                             smem_b(stage_idx), n_idx, k_b_idx,
                                             BLOCK_N, BLOCK_K, kSwizzleABMode, 2);
                }

                // Arrive at full barriers
                full_barriers(stage_idx)->arrive_and_expect_tx(kNumTMABytesPerStage);
            }
        }
    } else if (warp_idx == 1) {
        // ================= MMA issue warp =================
        InstrDescriptor instr_desc = make_instr_desc_f16(1 /*BF16*/, 1 /*BF16*/, 1 /*F32*/,
                                                          UMMA_M, UMMA_N, kMajorA, kMajorB);
        SmemDescriptor a_desc = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleABMode, 1, 2>(
            smem_a(0), 0, 0);
        SmemDescriptor b_desc = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleABMode, 1, 2>(
            smem_b(0), 0, 0);
        const uint32_t a_desc_lo =
            lane_idx < kNumStages ? a_desc.lo + lane_idx * kAStageBytes / 16 : 0u;
        const uint32_t b_desc_lo =
            lane_idx < kNumStages ? b_desc.lo + lane_idx * kBStageBytes / 16 : 0u;
        tcgen05_after_thread_sync();

        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t block_counter = (uint32_t)scheduler.current_iter;
            const uint32_t accum_phase = block_counter & 1;
            // The accumulator TMEM region is single-buffered: wait for the
            // epilogue of the PREVIOUS block to have drained it.
            if (block_counter > 0)
                tmem_empty_barrier->wait(accum_phase ^ 1);
            tcgen05_after_thread_sync();

            const uint32_t num_total_k_blocks = dg_max(1u, ceil_div_u32(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
                 advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers(stage_idx)->wait(phase);
                tcgen05_after_thread_sync();

                // Issue UMMAs
                const uint64_t runtime_instr_desc = make_runtime_instr_desc(instr_desc);
                const uint32_t a_base = __shfl_sync(0xffffffffu, a_desc_lo, (int)stage_idx);
                const uint32_t b_base = __shfl_sync(0xffffffffu, b_desc_lo, (int)stage_idx);
                if (elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++umma_k_idx) {
                        a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M, kSwizzleABMode, 1, 2>(
                            a_base, 0, umma_k_idx * UMMA_K);
                        b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleABMode, 1, 2>(
                            b_base, 0, umma_k_idx * UMMA_K);
                        mma_f16_1sm(a_desc.desc_, b_desc.desc_, 0,
                                    (umma_k_idx > 0 || k_block_idx > 0) ? 1u : 0u,
                                    runtime_instr_desc);
                    }
                }
                __syncwarp();

                // Commit (also fences before_thread_sync)
                umma_arrive_1sm(empty_barriers(stage_idx));
            }
            umma_arrive_1sm(tmem_full_barrier);
        }
    } else if (warp_idx >= 4) {
        // ================= Epilogue warp group (warps 4-7) =================
        const uint32_t epilogue_warp_idx = warp_idx - 4;

        // TMEM allocation check (base must be 0; warps 4-7 map onto TMEM
        // rows [0, 128) because the hardware ignores the warp index bits).
        if (ld_shared_u32(tmem_ptr_in_smem) != 0)
            dg_trap();

        // Share the TMA store pipeline between blocks.
        uint32_t tma_stage_idx = 0;
        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t block_counter = (uint32_t)scheduler.current_iter;
            const uint32_t accum_phase = block_counter & 1;

            // Wait UMMA arrival
            tmem_full_barrier->wait(accum_phase);
            tcgen05_after_thread_sync();

            // Which (m, n) tile to store.  kCDWithGroupOffset is false for
            // both psum variants (m is the flat axis / kk uses the 3D group
            // coordinate instead).
            const uint32_t base_m_idx = m_block_idx * BLOCK_M;
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;
            // Empty k-groups must not touch D (a direct store would clobber
            // C with zeros); invalid m-grouped blocks are pure padding.
            const bool is_empty_group = kIsKGroupedPsum && scheduler.current_shape_k == 0;
            const bool is_computation_valid =
                !kIsMGroupedPsum || scheduler.is_computation_valid(m_block_idx, 0);

            if (!is_empty_group && is_computation_valid) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores;
                     ++s, tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages) {
                    // Wait shared memory to be released
                    if (epilogue_warp_idx == 0)
                        tma_store_wait<kNumTMAStoreStages - 1>();
                    named_barrier_sync(kNumUMMAStoreThreads, 8);

                    uint8_t* smem_base = smem_cd(tma_stage_idx);
                    const uint32_t m_idx = base_m_idx;
                    const uint32_t n_idx = base_n_idx + s * STORE_BLOCK_N;

                    constexpr uint32_t kNumBankGroupBytes = 16;
                    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / kCdElemSize;
                    constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerBankGroup;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumLoads; ++i) {
                        // Swizzled staging address (kSwizzleCDMode == 128:
                        // shortcut row = lane, col = i)
                        const uint32_t bank_group_index =
                            i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                        const uint32_t row = kHasShortcut ? (i / 8 + lane_idx)
                                                          : (bank_group_index / 8);
                        uint32_t col = kHasShortcut ? i : (bank_group_index % 8);
                        col ^= row % (kSwizzleCDMode / 16);
                        uint8_t* smem_ptr = smem_base
                            + epilogue_warp_idx * 32 * kSwizzleCDMode
                            + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                        const uint32_t tmem_addr = s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                        uint32_t values[kNumElemsPerBankGroup];
                        if (kCdIsFloat) {
                            // 4 fp32 = one 16B bank group
                            tmem_load_32dp32b_x4(tmem_addr, values[0], values[1], values[2],
                                                 values[3]);
                            fence_view_async_tmem_load();
                            st_shared_u32x4((uint32_t*)smem_ptr, values[0], values[1], values[2],
                                            values[3]);
                        } else {
                            // 8 bf16 = one 16B bank group: two x4 loads,
                            // packed pairwise into 4 words
                            tmem_load_32dp32b_x8(tmem_addr, values[0], values[1], values[2],
                                                 values[3], values[4], values[5], values[6],
                                                 values[7]);
                            fence_view_async_tmem_load();
                            st_shared_u32x4((uint32_t*)smem_ptr,
                                            cast_bf16_and_pack(values[0], values[1]),
                                            cast_bf16_and_pack(values[2], values[3]),
                                            cast_bf16_and_pack(values[4], values[5]),
                                            cast_bf16_and_pack(values[6], values[7]));
                        }
                    }

                    tma_store_fence();
                    named_barrier_sync(kNumUMMAStoreThreads, 8);
                    if (epilogue_warp_idx == 0 && elect_one_sync()) {
                        if (kIsKGroupedPsum) {
                            // D is [G, M, N]: group rides the 3rd coordinate.
                            if (kWithAccumulation)
                                tma_reduce_add_3d(&tensor_map_cd, smem_base, n_idx, m_idx,
                                                  scheduler.current_group_idx);
                            else
                                tma_store_3d(&tensor_map_cd, smem_base, n_idx, m_idx,
                                             scheduler.current_group_idx);
                        } else {
                            tma_store_2d(&tensor_map_cd, smem_base, n_idx, m_idx);
                        }
                        tma_store_arrive();
                    }
                    __syncwarp();
                }
            }

            // Release the accumulator TMEM region (all 128 epilogue threads).
            tcgen05_before_thread_sync();
            tmem_empty_barrier->arrive();
            __syncwarp();
        }
    }

    // Tear-down: all TMEM reads are ordered before this barrier.
    __syncthreads();
    if (warp_idx == 1)
        tmem_dealloc_1sm(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        dg_trap();  // "This kernel only supports sm_100a"
#endif
}

}  // namespace dg
