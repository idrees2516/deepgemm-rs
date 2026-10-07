// DeepGEMM-RS kernel prelude: self-contained device helpers for NVRTC.
// No #includes; every exotic instruction is inline PTX. CUDA 12.8+, SM90a/SM100a.
//
// This file mirrors the union of upstream DeepGEMM's
//   ptx/{ld_st,tcgen05,tma,utils}.cuh, mma/sm100.cuh, comm/barrier.cuh,
//   common/{math,packing,utils}.cuh and the CUTLASS bits they rely on
// (SmemDescriptor / InstrDescriptorBlockScaled / ClusterTransactionBarrier).

#pragma once

// ===========================================================================
// CONCEPTS — the Blackwell (SM100) execution model in one page
// ===========================================================================
// This prelude is the shared "runtime" of every kernel. The sections below
// explain each hardware concept the inline PTX below exercises; the same
// material is rendered as diagrams in docs/concepts.md.
//
// ---------------------------------------------------------------------------
// 1. THE ASYNCHRONOUS DATAFLOW MODEL
// ---------------------------------------------------------------------------
// A GEMM on SM100 is a *system of engines*, not a sequence of loads and math:
//
//   HBM (global) ──TMA──> SMEM ──tcgen05.mma──> TMEM ──tcgen05.ld──> registers
//        │                 (staging)      (tensor core,          (epilogue
//        │                                 async engine)           warps)
//        └──────────────────── TMA store <────────────────────────────┘
//
// Every arrow is asynchronous and overlapped: while the tensor core computes
// stage s, TMA fetches stage s+1, and the epilogue warps drain stage s-1.
// The kernel's job is to *schedule* these engines, not to do arithmetic —
// almost no FMA runs on the CUDA cores in the main loop.
//
// ---------------------------------------------------------------------------
// 2. TMEM — the tensor core's private memory (tcgen05.alloc / .ld / .st)
// ---------------------------------------------------------------------------
// Each SM has 256 columns x 128 lanes x 32-bit of "tensor memory" that ONLY
// the tcgen05 unit writes. The MMA writes D (accumulators) there; a scale-
// factor path additionally writes the per-block scales into SF columns.
//
//    TMEM (one SM, cta_group::1 view):
//      cols 0..127              cols 128..255
//    ┌──────────────────────┬──────────────────────┐
//    │ D (accumulators)     │ optional SF cols      │  row r = lane r of
//    │ UMMA_M x UMMA_N      │ (SF_BLOCK_K * 32)     │  the 128 "dp" lanes
//    └──────────────────────┴──────────────────────┘
//     tcgen05.ld.32x32b: lane i reads row i — a warp sees a 32x32 tile.
//
//   * tcgen05.alloc   — claim columns (must be >= 32, power of two, issued by
//                        ONE warp); .relinquish_alloc_permit lets other CTAs
//                        in the cluster take over the allocation.
//   * tcgen05.ld      — move TMEM -> registers (epilogue / math warps read
//                        D). 32x32b shape: one lane per TMEM row.
//   * tcgen05.commit  — make the *completion of prior MMAs* visible to an
//                        mbarrier (the async engine's only sync primitive).
//
// The "TMEM overlap trick" (used when SF columns exceed the 512-col budget):
// beyond 512 total columns, SF columns at (j) alias accumulator columns at
// (j - 512) — the kernel pipelines around it by never reading a TMEM region
// while the overlapping region's consumer still owns it.
//
// ---------------------------------------------------------------------------
// 3. TMA — descriptor-driven bulk async copy (cp.async.bulk.tensor.*)
// ---------------------------------------------------------------------------
// TMA copies WHOLE TILES between global and shared memory with one
// instruction, honoring a swizzle pattern baked into the descriptor:
//
//   global tensor [outer, inner]          SMEM tile [BLOCK_OUTER, atom*...]
//   ┌────────────────────┐   TMA box   ┌─────┬─────┬─────┐  XOR swizzle:
//   │ ████ tile          │ ──────────> │ A0  │ A1  │ A2  │  atom = 128B
//   │                    │             └─────┴─────┴─────┘  (row r, byte c)
//   └────────────────────┘                                lands at
//   The descriptor (cuTensorMapEncodeTiled, built host-side in tma.rs and      (r ^ (c/16)) — banks never
//   passed BY VALUE as __grid_constant__) encodes: dtype, box shape,           collide for 16B accesses.
//   strides, and the swizzle atom (16B/32B/64B/128B).
//
//   * `.mbarrier::complete_tx::bytes` — TMA arrival feeds the consumer
//     barrier with the *byte count* (expect_tx); the barrier flips when both
//     the expected bytes AND the arrival count are satisfied.
//   * `.multicast::cluster` — one TMA can deliver the same tile to BOTH CTAs
//     of a cluster (each counts its own expect_tx).
//   * `.L2::cache_hint` — a 64-bit access-policy descriptor (register "l"
//     operand); kEvictNormalHint keeps streaming tiles from thrashing L2.
//
// ---------------------------------------------------------------------------
// 4. MBARRIER — the pipeline primitive (init / arrive / expect_tx / wait)
// ---------------------------------------------------------------------------
// An mbarrier is a 64-bit SMEM word carrying a phase bit. Producers arrive
// (optionally with a transaction-byte expectation), consumers spin on
// `try_wait.parity` until the phase flips:
//
//   producer (warp 0)                    consumer (epilogue warps)
//   ───────────────────                  ─────────────────────────
//   mbarrier.arrive.expect_tx(_, bar,    wait(bar, parity=0)
//          TX_BYTES)                     ... flips when bytes arrive
//   tma_load(...) -> SMEM               tcgen05.ld / tma_store
//   (TMA hardware: complete_tx)          wait(bar, parity=1)  // next epoch
//
//   * Parity alternates 0/1 per reuse of the same stage slot, so a
//     multi-stage ring needs only ONE phase bit per barrier — the classic
//     double (N-stage) buffer:
//        time ──>  [P: fill s0][P: fill s1][P: fill s2]...
//                  [C: drain s0]        [C: drain s1] ...
//   * `arrive.cluster` (mapa) signals a barrier living in ANOTHER CTA — used
//     by the leader-CTA MMA scheme and 2-SM TMA.
//
// ---------------------------------------------------------------------------
// 5. WARP SPECIALIZATION — who does what (GEMM kernel, 256 threads)
// ---------------------------------------------------------------------------
//   CTA (256 threads) ─ cluster pair with the buddy CTA (cta_group::2 MMA)
//   ┌────────────────────────────────────────────────────────────────────┐
//   │ warp 0   TMA producer: for each k-block, issue A/B/SF tile loads    │
//   │ warp 1   MMA issuer: elect_one; builds per-stage descriptors,       │
//   │          fires tcgen05.mma (leader CTA only in 2-SM mode) + UTCCP   │
//   │ warps 2,3 SF transposers: rearrange SF bytes in SMEM so UTCCP's     │
//   │          32x128b warp quads hit the right TMEM lanes                 │
//   │ warps 4..7 epilogue: tcgen05.ld accumulators -> SMEM (swizzle-      │
//   │          staged) -> TMA store to global; bf16 pack on the way       │
//   └────────────────────────────────────────────────────────────────────┘
//   setmaxnreg.inc/dec rebalances the physical register file between
//   specialized (few) and math (many) warps — the epilogue warps donate
//   registers to the producer path at zero cost.
//
// ---------------------------------------------------------------------------
// 6. NUMERIC FORMATS & BLOCK SCALING (OCP MX, DeepSeek recipe)
// ---------------------------------------------------------------------------
//   E4M3 (FP8): sign|4-bit exp (bias 7)|3-bit mant  — max 448, min 2^-9
//   E2M1 (FP4): sign|2-bit exp (bias 1)|1-bit mant  — grid {0,.5,1,1.5,2,3,4,6}
//   UE8M0 (SF) : pure 8-bit exponent, value = 2^(code-127), code 0 reserved
//
//   Block scaling: for element (i, k) with granularity g (32 for MX, 128 for
//   the DeepSeek FP8 recipe):
//       contribution = code(i,k) * 2^(sfA(i, k/g) - 127) * code(j,k) * 2^(sfB(j, k/g) - 127)
//   The MMA hardware (kind::mxf8f6f4 / kind::mxf4) applies the *linear*
//   code product; the power-of-two scales ride a SEPARATE SF path:
//   SF words are packed 4-per-int32 in K order (byte j = k*4+j of the
//   group's row), laid out MN-contiguous so a 16B TMA row = 16 rows.
//   UTCCP (tcgen05.cp.32x128b.warpx4) transposes them into TMEM SF columns.
//
// ---------------------------------------------------------------------------
// 7. PERSISTENT KERNEL + L2-BLOCK SWIZZLE SCHEDULING
// ---------------------------------------------------------------------------
// The grid is fixed at num_SMs blocks (one wave forever); each CTA loops
// over output tiles. Tile visit order is "swizzled" in groups of
// kNum1DBlocksPerGroup so that concurrent CTAs work on the same rows-band —
// their A-tile TMA loads coalesce in L2 (shared across the group), while B
// tiles multicast. Programmatic dependent launch (griddepcontrol.wait /
// launch_dependents) lets back-to-back kernels overlap prologue/epilogue.
// ===========================================================================

#define DG_DEVICE __device__ __forceinline__
#define DG_GLOBAL __global__
#define DG_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

// Fixed-width integer types: NVRTC compiles with *no* implicit headers
// (no stdint.h), so the standard names must be provided here. Safe against
// redefinition because these translation units are zero-include by design.
// (This was a real bug caught by the sandbox `compile-check`: without these,
// every kernel fails to JIT — NVRTC does not know `uint32_t` etc.)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
// Use the compiler's built-in vector types (predefined by NVRTC).
// uint4/int4/float2/float4 are available without headers in NVRTC.

namespace dg {

constexpr DG_DEVICE uint32_t ceil_div_u32(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
constexpr DG_DEVICE uint32_t align_u32(uint32_t a, uint32_t b) { return ceil_div_u32(a, b) * b; }
constexpr DG_DEVICE uint32_t dg_min(uint32_t a, uint32_t b) { return a < b ? a : b; }
constexpr DG_DEVICE uint32_t dg_max(uint32_t a, uint32_t b) { return a > b ? a : b; }
template <typename T> DG_DEVICE void dg_swap(T& a, T& b) { T t = a; a = b; b = t; }

DG_DEVICE uint32_t get_lane_idx() { return threadIdx.x & 31; }
DG_DEVICE uint32_t get_warp_idx() { return threadIdx.x >> 5; }
DG_DEVICE uint32_t get_block_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}
DG_DEVICE void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
DG_DEVICE void cluster_wait() { asm volatile("barrier.cluster.wait;" ::: "memory"); }
DG_DEVICE void cluster_sync_relaxed() { cluster_arrive_relaxed(); cluster_wait(); }

DG_DEVICE void griddepcontrol_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

DG_DEVICE bool elect_one_sync() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\tselp.b32 %0, 1, 0, p;\n\t}"
                 : "=r"(pred));
    return pred != 0;
}

DG_DEVICE uint32_t cvta_shared_to_u32(const void* ptr) {
    // Convert generic -> shared. The PTX form must be .u64 on both operands
    // (a generic pointer is 64-bit; mixing a .b64 source with a .u32 opcode
    // is rejected by ptxas with "Arguments mismatch"). The shared window
    // occupies the low 32 bits, so truncating afterwards is exact.
    // (Upstream uses the __cvta_generic_to_shared builtin; this is the
    // equivalent zero-include inline-asm form.)
    uint64_t addr64;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr64) : "l"(ptr));
    return static_cast<uint32_t>(addr64);
}

DG_DEVICE void named_barrier_sync(uint32_t num_threads, uint32_t id) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(num_threads) : "memory");
}

// ---------------------------------------------------------------------------
// mbarrier (ClusterTransactionBarrier)
// ---------------------------------------------------------------------------
struct Barrier {
    uint64_t barrier_;

    DG_DEVICE void init(uint32_t arrive_count) const {
        asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::
                     "r"(arrive_count), "r"(cvta_shared_to_u32(&barrier_)));
    }

    // Arrive at the barrier in this CTA.
    DG_DEVICE void arrive() const {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::
                     "r"(cvta_shared_to_u32(&barrier_)));
    }

    // Arrive at the barrier in CTA `cta_id` of the cluster (mapa).
    DG_DEVICE void arrive_cluster(uint32_t cta_id) const {
        asm volatile(
            "{\n\t.reg .b32 remAddr32;\n\t"
            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
            "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];\n\t}"
            :: "r"(cvta_shared_to_u32(&barrier_)), "r"(cta_id));
    }

    DG_DEVICE void arrive_count(uint32_t count) const {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::
                     "r"(cvta_shared_to_u32(&barrier_)), "r"(count));
    }

    DG_DEVICE void arrive_and_expect_tx(uint32_t tx_bytes) const {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" ::
                     "r"(tx_bytes), "r"(cvta_shared_to_u32(&barrier_)));
    }

    DG_DEVICE void wait(uint32_t phase) const {
        asm volatile(
            "{\n\t.reg .pred P1;\n\t"
            "DG_WAIT_LOOP:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
            "@P1 bra DG_WAIT_DONE;\n\t"
            "bra DG_WAIT_LOOP;\n\t"
            "DG_WAIT_DONE:\n\t}"
            :: "r"(cvta_shared_to_u32(&barrier_)), "r"(phase), "r"(0x989680u)
            : "memory");
    }
};

DG_DEVICE void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
DG_DEVICE void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
DG_DEVICE void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
DG_DEVICE void fence_view_async_tmem_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
DG_DEVICE void fence_view_async_tmem_store() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
DG_DEVICE void tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int kNumRemainingWaits = 0>
DG_DEVICE void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(kNumRemainingWaits) : "memory");
}

// ---------------------------------------------------------------------------
// TMA (cp.async.bulk.tensor / cp.async.bulk)
// ---------------------------------------------------------------------------
// 128-byte aligned tensor map passed by value as a kernel parameter.
struct alignas(64) TmaMap {
    uint64_t data_[16];
};

DG_DEVICE void prefetch_tma_map(const TmaMap* map) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

// 2D tile load: shared::cluster (multicast mask 0 -> plain), with L2 cache hint.
DG_DEVICE void tma_load_2d(const TmaMap* map, Barrier* bar, void* smem,
                           uint64_t cache_hint, uint32_t c_inner, uint32_t c_outer) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(cvta_shared_to_u32(smem)), "l"(map),
           "r"(cvta_shared_to_u32(&bar->barrier_)), "r"(c_inner), "r"(c_outer), "l"(cache_hint)
        : "memory");
}

// 2D multicast load (SM90 path).
DG_DEVICE void tma_load_2d_multicast(const TmaMap* map, Barrier* bar, void* smem,
                                     uint16_t cta_mask, uint32_t c_inner, uint32_t c_outer) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3;"
        :: "r"(cvta_shared_to_u32(smem)), "l"(map),
           "r"(cvta_shared_to_u32(&bar->barrier_)), "h"(cta_mask), "r"(c_inner), "r"(c_outer)
        : "memory");
}

// 2D load for a 2-CTA cluster (SM100): signals the peer CTA's barrier.
DG_DEVICE void tma_load_2d_2sm(const TmaMap* map, Barrier* bar, void* smem,
                               uint64_t cache_hint, uint32_t c_inner, uint32_t c_outer) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(cvta_shared_to_u32(smem)), "l"(map),
           "r"(cvta_shared_to_u32(&bar->barrier_)), "r"(c_inner), "r"(c_outer), "l"(cache_hint)
        : "memory");
}

DG_DEVICE void tma_store_2d(const TmaMap* map, const void* smem,
                            uint32_t c_inner, uint32_t c_outer) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(map), "r"(cvta_shared_to_u32(smem)), "r"(c_inner), "r"(c_outer)
        : "memory");
}

DG_DEVICE void tma_reduce_add_2d(const TmaMap* map, const void* smem,
                                 uint32_t c_inner, uint32_t c_outer) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(map), "r"(cvta_shared_to_u32(smem)), "r"(c_inner), "r"(c_outer)
        : "memory");
}

DG_DEVICE void tma_store_3d(const TmaMap* map, const void* smem,
                            uint32_t c0, uint32_t c1, uint32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :: "l"(map), "r"(cvta_shared_to_u32(smem)), "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

DG_DEVICE void tma_reduce_add_3d(const TmaMap* map, const void* smem,
                                 uint32_t c0, uint32_t c1, uint32_t c2) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :: "l"(map), "r"(cvta_shared_to_u32(smem)), "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}

// tile::gather4 for paged MQA logits (4 rows gathered in one op).
DG_DEVICE void tma_gather4_2d(const TmaMap* map, Barrier& bar, void* smem,
                              uint32_t col_idx, int4 row_idxs, uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(cvta_shared_to_u32(smem)), "l"(map), "r"(col_idx),
           "r"((uint32_t)row_idxs.x), "r"((uint32_t)row_idxs.y),
           "r"((uint32_t)row_idxs.z), "r"((uint32_t)row_idxs.w),
           "r"(cvta_shared_to_u32(&bar.barrier_)), "l"(cache_hint)
        : "memory");
}

DG_DEVICE void tma_load_1d(void* smem, const void* gmem, Barrier* bar,
                           uint32_t num_bytes, uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1], %2, [%3], %4;"
        :: "r"(cvta_shared_to_u32(smem)), "l"(gmem), "r"(num_bytes),
           "r"(cvta_shared_to_u32(&bar->barrier_)), "l"(cache_hint)
        : "memory");
}

// ---------------------------------------------------------------------------
// Shared memory ld/st (vectorized)
// ---------------------------------------------------------------------------
DG_DEVICE uint32_t ld_shared_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(cvta_shared_to_u32(p)));
    return v;
}
DG_DEVICE void st_shared_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(cvta_shared_to_u32(p)), "r"(v));
}
DG_DEVICE uint32_t ld_shared_vol_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(v) : "r"(cvta_shared_to_u32(p)));
    return v;
}
DG_DEVICE void st_shared_u32x4(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                 "r"(cvta_shared_to_u32(p)), "r"(a), "r"(b), "r"(c), "r"(d));
}
DG_DEVICE uint4 ld_shared_u128(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(cvta_shared_to_u32(p)));
    return v;
}
DG_DEVICE void st_shared_u128(uint32_t* p, const uint4& v) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                 "r"(cvta_shared_to_u32(p)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}
DG_DEVICE void st_global_u16(void* p, uint32_t v) {
    asm volatile(
        "{\n\t.reg .b16 lo;\n\tmov.b32 {lo, _}, %1;\n\tst.global.b16 [%0], lo;\n\t}"
        :: "l"(p), "r"(v));
}
DG_DEVICE uint32_t ld_global_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
DG_DEVICE void st_global_u32(void* p, uint32_t v) {
    asm volatile("st.global.b32 [%0], %1;" :: "l"(p), "r"(v));
}

// ---------------------------------------------------------------------------
// BF16 / FP8 / FP4 conversions (all via PTX; no cuda_bf16.h)
// ---------------------------------------------------------------------------
typedef uint16_t bf16_raw;  // raw bits

DG_DEVICE uint32_t cvt_bf16x2_f32(float lo, float hi) {
    // Packs (lo, hi) into bf16x2 (RN). Matches __floats2bfloat162_rn(lo, hi).
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DG_DEVICE uint32_t cast_bf16_and_pack(uint32_t a_f32bits, uint32_t b_f32bits) {
    float a = __int_as_float((int)a_f32bits), b = __int_as_float((int)b_f32bits);
    return cvt_bf16x2_f32(a, b);
}
DG_DEVICE uint32_t fma_bf16x2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
DG_DEVICE uint32_t add_bf16x2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
DG_DEVICE uint32_t cvt_relu_bf16x2_f32(float lo, float hi) {
    uint32_t r;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DG_DEVICE uint32_t low2_bf16x2(uint32_t v) {
    uint32_t lo = v & 0xffffu;
    return lo | (lo << 16);
}
DG_DEVICE uint32_t high2_bf16x2(uint32_t v) {
    uint32_t hi = v >> 16;
    return hi | (hi << 16);
}
DG_DEVICE uint32_t cvt_e4m3x2_f32(float lo, float hi) {
    // Two f32 -> packed e4m3x2 (byte0 = lo, byte1 = hi), returned in a u16.
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}
// RN (ties-to-even) + satfinite-to-6 onto the E2M1 grid {0,.5,1,1.5,2,3,4,6}.
// Matches cvt.rn.satfinite.e2m1x2.f32 bit-exactly (cross-checked against the
// CPU golden model). Implemented in integer/float compares instead of the
// hardware cvt: the .b8 destination of that instruction cannot be packed into
// a .b32 with a single-element `mov {t}` (ptxas rejects the form), and this
// path sits in the memory-bound quant kernel, so ALU cost is immaterial.
DG_DEVICE uint32_t e2m1_code(float v) {
    if (v <= 0.25f) return 0u;   // tie at 0.25  -> 0     (even code)
    if (v <  0.75f) return 1u;   // (.25, .75)   -> 0.5
    if (v <= 1.25f) return 2u;   // tie at 0.75 / 1.25 -> 1.0
    if (v <  1.75f) return 3u;   // (1.25, 1.75) -> 1.5
    if (v <= 2.5f)  return 4u;   // ties at 1.75 / 2.5  -> 2.0
    if (v <  3.5f)  return 5u;   // (2.5, 3.5)   -> 3.0
    if (v <= 5.0f)  return 6u;   // ties at 3.5 / 5.0   -> 4.0
    return 7u;                   // (5, inf)     -> 6   (satfinite)
}
DG_DEVICE uint32_t cvt_e2m1x2_f32(float lo, float hi) {
    // Two f32 -> packed e2m1x2 nibble pair (low nibble = lo), as u32 byte.
    const uint32_t c0 = e2m1_code(lo < 0.0f ? -lo : lo) | (lo < 0.0f ? 8u : 0u);
    const uint32_t c1 = e2m1_code(hi < 0.0f ? -hi : hi) | (hi < 0.0f ? 8u : 0u);
    return c0 | (c1 << 4);
}
// f16 (raw bits) -> f32
DG_DEVICE float f32_from_f16(uint32_t h) {
    float f;
    asm volatile("{\n\t.reg .f16 t;\n\tmov.b16 t, %1;\n\tcvt.f32.f16 %0, t;\n\t}"
                 : "=f"(f) : "h"((uint16_t)h));
    return f;
}
// Packed e4m3x2 (u16) -> two f32 (bits returned via u32 pair).
DG_DEVICE void cvt_f32x2_e4m3x2(uint32_t packed, uint32_t& lo, uint32_t& hi) {
    uint32_t f16x2;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"((uint16_t)packed));
    float f0 = f32_from_f16(f16x2 & 0xffffu);
    float f1 = f32_from_f16(f16x2 >> 16);
    lo = __float_as_uint(f0);
    hi = __float_as_uint(f1);
}
// One e2m1 nibble -> f32 (values: 0, .5, 1, 1.5, 2, 3, 4, 6 and negatives).
DG_DEVICE float f32_from_e2m1(uint32_t v) {
    const uint32_t sign = (v >> 3) & 1u;
    const uint32_t exp = (v >> 1) & 3u;
    const uint32_t man = v & 1u;
    float out;
    if (exp == 0) {
        out = man ? 0.5f : 0.0f;
    } else {
        const uint32_t bits = ((exp - 1u + 127u) << 23) | (man << 22);
        out = __uint_as_float(bits);
    }
    return sign ? -out : out;
}
// Packed e2m1x2 byte -> two f32.
DG_DEVICE void cvt_f32x2_e2m1x2(uint32_t byte, uint32_t& lo, uint32_t& hi) {
    float f0 = f32_from_e2m1(byte & 0xfu);
    float f1 = f32_from_e2m1((byte >> 4) & 0xfu);
    lo = __float_as_uint(f0);
    hi = __float_as_uint(f1);
}

// ---------------------------------------------------------------------------
// UMMA shared-memory descriptor (SM100)
// ---------------------------------------------------------------------------
enum class UmmaLayoutType : uint8_t {
    SWIZZLE_NONE = 0,
    SWIZZLE_128B_BASE32B = 1,
    SWIZZLE_128B = 2,
    SWIZZLE_64B = 4,
    SWIZZLE_32B = 6,
};

union SmemDescriptor {
    uint64_t desc_;
    struct {
        uint16_t start_address_ : 14, : 2;
        uint16_t leading_byte_offset_ : 14, : 2;
        uint16_t stride_byte_offset_ : 14, version_ : 2;
        uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;
        uint8_t : 5, layout_type_ : 3;
    };
    struct { uint32_t lo, hi; };
};
static_assert(sizeof(SmemDescriptor) == 8, "bad smem desc size");

DG_DEVICE SmemDescriptor make_smem_desc(UmmaLayoutType layout, const void* smem_ptr,
                                        uint32_t stride_byte_offset, uint32_t leading_byte_offset) {
    SmemDescriptor d;
    d.desc_ = 0;
    d.version_ = 1;
    d.lbo_mode_ = 0;
    d.layout_type_ = (uint8_t)layout;
    d.start_address_ = (uint16_t)(cvta_shared_to_u32(smem_ptr) >> 4);
    d.base_offset_ = 0;
    d.stride_byte_offset_ = stride_byte_offset >> 4;
    d.leading_byte_offset_ = leading_byte_offset >> 4;
    return d;
}

// ---------------------------------------------------------------------------
// UMMA instruction descriptors
// ---------------------------------------------------------------------------
enum class UmmaFormat : uint8_t {
    E4M3 = 0, E5M2 = 1, BF16 = 1 /*F32F16 fmt*/, E2M3 = 3, E3M2 = 4, E2M1 = 5,
};

union InstrDescriptor {
    uint32_t desc_;
    struct {
        uint16_t sparse_id2_ : 2,
                 sparse_flag_ : 1,
                 saturate_ : 1,
                 c_format_ : 2,
                 : 1,
                 a_format_ : 3,
                 b_format_ : 3,
                 a_negate_ : 1,
                 b_negate_ : 1,
                 a_major_ : 1;
        uint16_t b_major_ : 1,
                 n_dim_ : 6,
                 : 1,
                 m_dim_ : 5,
                 k_size_ : 1,
                 max_shift_ : 2;
    };
};
static_assert(sizeof(InstrDescriptor) == 4, "bad instr desc size");

union InstrDescriptorBlockScaled {
    uint32_t desc_;
    struct {
        uint16_t sparse_id2_ : 2,
                 sparse_flag_ : 1,
                 : 1,
                 b_sf_id_ : 2,
                 : 1,
                 a_format_ : 3,
                 b_format_ : 3,
                 a_negate_ : 1,
                 b_negate_ : 1,
                 a_major_ : 1;
        uint16_t b_major_ : 1,
                 n_dim_ : 6,
                 scale_format_ : 1,
                 m_dim_ : 5,
                 a_sf_id_ : 2,
                 k_size_ : 1;
    };
};
static_assert(sizeof(InstrDescriptorBlockScaled) == 4, "bad bs instr desc size");

DG_DEVICE InstrDescriptorBlockScaled make_instr_desc_bs(
        uint32_t a_format, uint32_t b_format, uint32_t m_dim, uint32_t n_dim,
        uint32_t a_major, uint32_t b_major) {
    InstrDescriptorBlockScaled d;
    d.desc_ = 0;
    d.a_format_ = a_format;
    d.b_format_ = b_format;
    d.scale_format_ = 1;  // UE8M0
    d.m_dim_ = m_dim >> 4;
    d.n_dim_ = n_dim >> 3;
    d.a_major_ = a_major;
    d.b_major_ = b_major;
    d.a_sf_id_ = 0;
    d.b_sf_id_ = 0;
    return d;
}

DG_DEVICE InstrDescriptor make_instr_desc_f16(
        uint32_t a_format, uint32_t b_format, uint32_t c_format,
        uint32_t m_dim, uint32_t n_dim, uint32_t a_major, uint32_t b_major) {
    InstrDescriptor d;
    d.desc_ = 0;
    d.a_format_ = a_format;
    d.b_format_ = b_format;
    d.c_format_ = c_format;
    d.m_dim_ = m_dim >> 4;
    d.n_dim_ = n_dim >> 3;
    d.a_major_ = a_major;
    d.b_major_ = b_major;
    d.k_size_ = 0;
    return d;
}

DG_DEVICE uint64_t make_runtime_instr_desc_bs(InstrDescriptorBlockScaled d,
                                              uint32_t sfa_id, uint32_t sfb_id) {
    d.a_sf_id_ = sfa_id;
    d.b_sf_id_ = sfb_id;
    return (uint64_t)d.desc_ << 32;
}
DG_DEVICE uint64_t make_runtime_instr_desc(InstrDescriptor d) {
    return (uint64_t)d.desc_ << 32;
}

// ---------------------------------------------------------------------------
// tcgen05: MMA, UTCCP, TMEM load/store, alloc/dealloc, commit
// ---------------------------------------------------------------------------
// kind::mxf8f6f4 with UE8M0 scale factors, 1 CTA.
DG_DEVICE void mma_mxf8f6f4_1sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                uint32_t scale_c, uint64_t idesc,
                                uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"((uint32_t)(idesc >> 32)), "r"(scale_c),
           "r"(tmem_sfa), "r"(tmem_sfb));
}
// kind::mxf8f6f4, 2 CTA cluster.
DG_DEVICE void mma_mxf8f6f4_2sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                                uint32_t scale_c, uint64_t idesc,
                                uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"((uint32_t)(idesc >> 32)), "r"(scale_c),
           "r"(tmem_sfa), "r"(tmem_sfb));
}
// kind::mxf4 (packed E2M1, block32 scales), 1 CTA. CUDA >= 12.9 uses .block32.
DG_DEVICE void mma_mxf4_1sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                            uint32_t scale_c, uint64_t idesc,
                            uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"((uint32_t)(idesc >> 32)), "r"(scale_c),
           "r"(tmem_sfa), "r"(tmem_sfb));
}
DG_DEVICE void mma_mxf4_2sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                            uint32_t scale_c, uint64_t idesc,
                            uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"((uint32_t)(idesc >> 32)), "r"(scale_c),
           "r"(tmem_sfa), "r"(tmem_sfb));
}
// kind::f16 (BF16 operands), 1/2 CTA.
DG_DEVICE void mma_f16_1sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                           uint32_t scale_c, uint64_t idesc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"((uint32_t)(idesc >> 32)), "r"(scale_c));
}
DG_DEVICE void mma_f16_2sm(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c,
                           uint32_t scale_c, uint64_t idesc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"((uint32_t)(idesc >> 32)), "r"(scale_c));
}

// UTCCP: copy 128 packed SF elements from SMEM to TMEM (32x128b warpx4).
DG_DEVICE void utccp_4x32dp128bit_1cta(uint64_t src_desc, uint32_t dst_tmem) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(dst_tmem), "l"(src_desc));
}
DG_DEVICE void utccp_4x32dp128bit_2cta(uint64_t src_desc, uint32_t dst_tmem) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(dst_tmem), "l"(src_desc));
}

// TMEM loads.
DG_DEVICE void tmem_load_32dp32b_x4(uint32_t addr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(addr));
}
DG_DEVICE void tmem_load_32dp32b_x8(uint32_t addr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
                                    uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3), "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
                 : "r"(addr));
}
DG_DEVICE void tmem_load_32dp32b_x32(uint32_t addr, uint32_t* v) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
                 " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
                   "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
                   "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
                 : "r"(addr));
}
// Float overloads (upstream passes `float accum[...]` arrays through
// `reinterpret_cast<uint32_t*>`; these overloads keep kernel code natural).
DG_DEVICE void tmem_load_32dp32b_x4(uint32_t addr, float& v0, float& v1, float& v2, float& v3) {
    tmem_load_32dp32b_x4(addr, reinterpret_cast<uint32_t&>(v0), reinterpret_cast<uint32_t&>(v1),
                         reinterpret_cast<uint32_t&>(v2), reinterpret_cast<uint32_t&>(v3));
}
DG_DEVICE void tmem_load_32dp32b_x8(uint32_t addr, float& v0, float& v1, float& v2, float& v3,
                                    float& v4, float& v5, float& v6, float& v7) {
    tmem_load_32dp32b_x8(addr, reinterpret_cast<uint32_t&>(v0), reinterpret_cast<uint32_t&>(v1),
                         reinterpret_cast<uint32_t&>(v2), reinterpret_cast<uint32_t&>(v3),
                         reinterpret_cast<uint32_t&>(v4), reinterpret_cast<uint32_t&>(v5),
                         reinterpret_cast<uint32_t&>(v6), reinterpret_cast<uint32_t&>(v7));
}
DG_DEVICE void tmem_load_32dp32b_x16(uint32_t addr, uint32_t* v) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                   "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
                 : "r"(addr));
}
DG_DEVICE void tmem_load_32dp32b_x16(uint32_t addr, float* v) {
    tmem_load_32dp32b_x16(addr, reinterpret_cast<uint32_t*>(v));
}
// 16x256b: two rows per lane; satisfies the STSM layout for the swap-AB epilogue.
DG_DEVICE void tmem_load_16dp256b_x1(uint32_t addr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(addr));
}

// STSM (transposed) for the swap-AB epilogue.
DG_DEVICE void stsm_x4_trans(uint32_t smem_addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(smem_addr), "r"(a), "r"(b), "r"(c), "r"(d));
}

// tcgen05 commit -> mbarrier arrival.
DG_DEVICE void umma_arrive_1sm(Barrier* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(cvta_shared_to_u32(&bar->barrier_)));
}
DG_DEVICE void umma_arrive_2sm(Barrier* bar) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(cvta_shared_to_u32(&bar->barrier_)));
}
DG_DEVICE void umma_arrive_2sm_multicast(Barrier* bar, uint16_t cta_mask) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :: "r"(cvta_shared_to_u32(&bar->barrier_)), "h"(cta_mask));
}
DG_DEVICE void tcgen05_before_thread_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DG_DEVICE void tcgen05_after_thread_sync() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// TMEM allocation.
DG_DEVICE void tmem_alloc_1sm(uint32_t num_cols, uint32_t* dst_smem_addr) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(cvta_shared_to_u32(dst_smem_addr)), "r"(num_cols));
}
DG_DEVICE void tmem_alloc_2sm(uint32_t num_cols, uint32_t* dst_smem_addr) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(cvta_shared_to_u32(dst_smem_addr)), "r"(num_cols));
}
DG_DEVICE void tmem_dealloc_1sm(uint32_t tmem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr), "r"(num_cols));
}
DG_DEVICE void tmem_dealloc_2sm(uint32_t tmem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr), "r"(num_cols));
}
DG_DEVICE void tmem_relinquish_1sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
DG_DEVICE void tmem_relinquish_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}

// Register reallocation for warp-specialized kernels.
// PTX `"n"` operands must be compile-time immediates, so the register count
// is a template parameter (call sites pass constexpr values).
template <uint32_t N>
DG_DEVICE void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
}
template <uint32_t N>
DG_DEVICE void setmaxnreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
}

// ---------------------------------------------------------------------------
// GEMM scheduler (port of deep_gemm/scheduler/gemm.cuh)
// ---------------------------------------------------------------------------
enum class GemmType : uint32_t {
    Normal = 0,
    MGroupedContiguous = 1,
    MGroupedMasked = 2,
    Batched = 4,
};

DG_DEVICE bool gemm_type_is_m_grouped_contiguous(GemmType t) { return t == GemmType::MGroupedContiguous; }
DG_DEVICE bool gemm_type_is_k_grouped(GemmType t) { return false; }  // not ported

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs>
constexpr DG_DEVICE uint32_t get_num_1d_blocks_per_group() {
    uint32_t num_best = 0, min_usage = 0xffffffffu;
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t candidate = i == 0 ? 8u : 16u;
        const uint32_t usage = kIsMulticastOnA
            ? candidate * BLOCK_N + dg::ceil_div_u32(kNumSMs, candidate) * BLOCK_M
            : candidate * BLOCK_M + dg::ceil_div_u32(kNumSMs, candidate) * BLOCK_N;
        if (usage < min_usage) min_usage = usage, num_best = candidate;
    }
    return num_best;
}

template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA, kNumSMs>()>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_blocks_in_group = 0;

    int* grouped_layout;
    uint32_t current_group_idx = 0;
    uint32_t current_m_cumsum = 0;
    uint32_t current_shape_k;

    enum class IndexType { MN, K, SF_K };

    DG_DEVICE Scheduler(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int* grouped_layout_)
        : grouped_layout(grouped_layout_) {
        num_m_blocks = dg::ceil_div_u32(shape_m, BLOCK_M);
        num_n_blocks = dg::ceil_div_u32(shape_n, BLOCK_N);
        current_shape_k = shape_k;
        if (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched ||
            kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else {  // MGroupedMasked
            num_blocks = 0;
        }
    }

    DG_DEVICE void get_swizzled_block_idx(uint32_t block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const uint32_t secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const uint32_t num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const uint32_t group_idx = block_idx / num_blocks_per_group;
        uint32_t first_block_idx = group_idx * kNum1DBlocksPerGroup;
        uint32_t in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = dg_min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

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
        } else if (kGemmType == GemmType::MGroupedMasked) {
            const uint32_t offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else {  // Batched
            const uint32_t offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    DG_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t next_block_idx = (uint32_t)(++current_iter) * kNumSMs + blockIdx.x;

        if (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                if (current_group_idx >= kNumGroupsRuntime) return false;
                num_m_blocks = dg::ceil_div_u32((uint32_t)grouped_layout[current_group_idx], BLOCK_M);
                const uint32_t current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks) {
                    get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks,
                                           m_block_idx, n_block_idx);
                    return true;
                }
                current_group_idx++;
                current_m_cumsum = current_m_block_cumsum;
            }
        } else if (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroupsRuntime) return false;
            current_group_idx = next_block_idx / num_blocks;
            const uint32_t block_idx = next_block_idx - current_group_idx * num_blocks;
            if (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
            return true;
        } else {
            if (next_block_idx >= num_blocks) return false;
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
            return true;
        }
    }

    // Number of groups (runtime), passed via the kernel's grouped-layout buffer.
    uint32_t kNumGroupsRuntime = 0;
};

// Aligned TMEM column count (port of utils::get_num_aligned_tmem_cols).
template <uint32_t kNumCols>
constexpr DG_DEVICE uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if (kNumCols <= 32) return 32;
    if (kNumCols <= 64) return 64;
    if (kNumCols <= 128) return 128;
    if (kNumCols <= 256) return 256;
    return 512;
}

// SMEM pack factor: packed FP4 (mxf4 MMA) stores 2 elements per byte.
template <uint32_t kIsPackedFp4>
DG_DEVICE uint32_t smem_pack_factor() { return kIsPackedFp4 ? 2 : 1; }

// TMA atom size helper (port of tma::get_inner_block_atom_size).
template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, uint32_t kPackFactor, uint32_t kWireElemSize>
DG_DEVICE uint32_t inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor : kSwizzleMode / kWireElemSize;
}


// ---------------------------------------------------------------------------
// UMMA majors, cache hints, and descriptor builders (mma/sm100.cuh port)
// ---------------------------------------------------------------------------
enum : uint32_t { MAJOR_K = 0, MAJOR_MN = 1 };
// 64-bit cache-hint descriptor for `.L2::cache_hint` ("l" register operand).
// 0x10...0 = access_property::normal (upstream DeepGEMM / CUDA access-policy
// encoding). Must stay uint64_t — a uint32_t silently truncates to 0 (no hint).
constexpr uint64_t kEvictNormalHint = 0x1000000000000000ull;

constexpr DG_DEVICE uint32_t get_atom_base(UmmaLayoutType layout_type) {
    return layout_type == UmmaLayoutType::SWIZZLE_128B_BASE32B ? 32u : 16u;
}

template <uint32_t kSwizzleMode>
DG_DEVICE UmmaLayoutType to_umma_layout_type() {
    if (kSwizzleMode == 0 || kSwizzleMode == 16) return UmmaLayoutType::SWIZZLE_NONE;
    if (kSwizzleMode == 32) return UmmaLayoutType::SWIZZLE_32B;
    if (kSwizzleMode == 64) return UmmaLayoutType::SWIZZLE_64B;
    return UmmaLayoutType::SWIZZLE_128B;
}

// Stride (in storage elements) between consecutive K atoms, for MN-major.
template <uint32_t BLOCK_MN, uint32_t kSwizzleMode, uint32_t kPackFactor, uint32_t kStorageElemSize>
DG_DEVICE uint32_t inner_block_atom_size_mn() {
    return kSwizzleMode == 0 ? BLOCK_MN / kPackFactor : kSwizzleMode / kStorageElemSize;
}

template <uint32_t kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode,
          uint32_t kPackFactor, uint32_t kStorageElemSize>
DG_DEVICE uint32_t advance_umma_desc_lo(uint32_t base, uint32_t offset, uint32_t k_idx) {
    const uint32_t stride_k = kMajorMode == MAJOR_K
        ? 1u
        : inner_block_atom_size_mn<BLOCK_MN, kSwizzleMode, kPackFactor, kStorageElemSize>();
    const uint32_t byte_offset = (offset + k_idx * stride_k) * kStorageElemSize / kPackFactor;
    return base + (byte_offset >> 4);
}

template <uint32_t kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          uint32_t kPackFactor, uint32_t kStorageElemSize>
DG_DEVICE SmemDescriptor make_umma_desc(const void* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const UmmaLayoutType layout_type = to_umma_layout_type<kSwizzleMode>();
    const uint32_t num_non_contiguous = 128 / get_atom_base(layout_type);
    if (kMajorMode == MAJOR_K) {
        // One swizzle atom per stage on K; SBO strides 8-row atom groups.
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * kStorageElemSize / kPackFactor;
        const uint32_t byte_ptr_offset = (mn_idx * BLOCK_K + k_idx) * kStorageElemSize / kPackFactor;
        const uint8_t* p = (const uint8_t*)base_smem_ptr + byte_ptr_offset;
        return make_smem_desc(layout_type, p, stride_byte_offset, 0);
    } else {
        const uint32_t BLOCK_MN_ATOM = inner_block_atom_size_mn<BLOCK_MN, kSwizzleMode, kPackFactor, kStorageElemSize>();
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * kStorageElemSize;
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * kStorageElemSize;
        if (kSwizzleMode == 16) dg_swap(stride_byte_offset, leading_byte_offset);
        const uint32_t byte_ptr_offset = (mn_idx * BLOCK_K + k_idx) * kStorageElemSize;
        const uint8_t* p = (const uint8_t*)base_smem_ptr + byte_ptr_offset;
        return make_smem_desc(layout_type, p, stride_byte_offset, leading_byte_offset);
    }
}

DG_DEVICE SmemDescriptor make_sf_desc(const void* smem_ptr) {
    // UTCCP atom: 8 x 128 bits; SBO = 8*16 bytes, LBO = 0 (1 atom on K).
    return make_smem_desc(UmmaLayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}
DG_DEVICE void replace_smem_desc_addr(SmemDescriptor& desc, const void* smem_ptr) {
    desc.start_address_ = (uint16_t)(cvta_shared_to_u32(smem_ptr) >> 4);
}


} // namespace dg
