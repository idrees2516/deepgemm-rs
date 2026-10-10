// ===========================================================================
// hc_prenorm.cu — TF32 "hyperconnection pre-norm" GEMM (SM90a + SM100a).
//
// Port of upstream DeepGEMM:
//   * impls/sm90_tf32_hc_prenorm_gemm.cuh  -> dg::hc_prenorm_sm90_impl
//   * impls/sm100_tf32_hc_prenorm_gemm.cuh -> dg::hc_prenorm_sm100_impl
// launched by the Rust wrappers in src/api_hc_prenorm.rs.
// ===========================================================================
//
// ---------------------------------------------------------------------------
// 1. WHAT "HYPERCONNECTION PRE-NORM" IS
// ---------------------------------------------------------------------------
// Hyperconnection (the DSv4-style multi-stream residual: each layer mixes n
// residual streams h_1..h_n with learned connection weights) normalizes the
// *mixed* hidden state per row before applying the layer's projections
// (the "pre-norm" of the hyperconnection, an RMS-style normalization):
//
//        x  = sum_i a_i * h_i                    (connection mixing)
//        y  = (x / rms(x)) @ W^T                 (pre-normalized projection)
//        rms(x)_m = sqrt( (1/K) * sum_k x[m,k]^2 )
//
// By linearity of the GEMM, the normalization can ride *outside* the matmul:
//
//        y[m,n] = ( sum_k x[m,k] * W[n,k] ) / (K * rms(x)_m)
//               = ( sum_k x[m,k] * W[n,k] ) / sqrt( sum_k x[m,k]^2 )
//
// so ONE fused kernel can produce, per row m of A:
//   * D[m, n]  = the *unnormalized* projection  sum_k A[m,k] * B[n,k]
//   * sqr[m]   = the pre-norm statistic         sum_k A[m,k]^2
// and the caller divides column n of D by sqrt(sqr[m]) (per row).  Both
// quantities must come from the *same* values of A — hence the fusion; a
// separate norm kernel would re-read A and could disagree with what the MMA
// actually consumed.
//
// The shapes this kernel family targets: M = tokens (large), N = the
// hyperconnection width (tiny — 24 in the upstream tests, i.e. a handful of
// streams x heads), K = hidden size (7k-30k, huge).  The output is minuscule
// and K dominates, so both variants implement **split-K**: block
// `block_idx` = (m_block, k_split) computes the K-range `[k_offset,
// k_offset + num_total_stages*BLOCK_K)` of the same math and stores PARTIALS:
//
//        D[s, m, n] = sum_{k in split s} tf32(A[m,k]) * tf32(B[n,k])   (fp32)
//        sqr_sum[s, m] = sum_{k in split s} A[m,k]^2                   (fp32)
//
// (sqr_sum is laid out [num_splits, m] — indexed `shape_m * s + m`; D is
// [num_splits, m, n] via a 3D TMA store when num_splits > 1, plain 2D else).
// The caller reduces over s: D_total = sum_s D[s], sqr_total = sum_s
// sqr_sum[s].  This split-K partial contract is exactly the upstream
// `tf32_hc_prenorm_gemm(a, b, d, s, num_splits)` API.
//
// ---------------------------------------------------------------------------
// 2. TF32 NUMERICS (why A is bf16 and B is fp32)
// ---------------------------------------------------------------------------
// TF32 = 19-bit float: f32's exponent range with a 10-bit mantissa.
//   * A (the hidden state) arrives as **bf16** (8-bit mantissa).  Every bf16
//     value is exactly representable in TF32 (8 <= 10 mantissa bits), so the
//     bf16 -> f32 -> tf32 chain is LOSSLESS: the MMA consumes A exactly.
//   * B (the projection weights) arrives as **fp32** and is fed to the MMA
//     *unconverted*: the tensor core reads the top 19 bits of each fp32 word,
//     silently truncating the low 13 mantissa bits (no cvt.rna rounding —
//     upstream's deliberate choice; the weight tensor is expected to tolerate
//     it, matching torch's allow_tf32 matmul to ~1e-8 relative).
//   * Accumulation is full FP32 in both wgmma (f32 accumulator registers) and
//     tcgen05 (kind::tf32 with f32 C/D in TMEM).
//   * sqr_sum is computed on the CUDA cores in fp32 from the *exact* bf16
//     values (never TF32-rounded), so the pre-norm statistic is exact; only
//     its summation ORDER (per-lane partials + 4-lane butterfly) differs from
//     a sequential sum.
//
// ---------------------------------------------------------------------------
// 3. HOW THIS DIFFERS FROM THE PLAIN GEMMS OF THIS REPO
// ---------------------------------------------------------------------------
//   a) DUAL OUTPUT: every A element is consumed twice — as an MMA operand AND
//      as a squared addend.  That forces A through registers:
//        SM90:  A is lifted from SMEM into registers (bf16 -> f32), the
//               squares are accumulated on the fly, and the very registers
//               feed the *RS* form of wgmma (`wgmma.mma_async.m64nNk8.f32.
//               tf32.tf32` with A-in-registers) — B stays in SMEM via the SS
//               descriptor.
//        SM100: tcgen05.mma cannot take A from registers, so four "cast
//               warps" LDSM A out of SMEM, square-reduce while converting
//               bf16x2 -> fp32, and `tcgen05.st.16x256b` it into a TMEM A
//               region; the MMA warp then issues the *TS* form
//               (`tcgen05.mma.kind::tf32 [d-tmem], [a-tmem], b-desc, ...`).
//   b) NO PERSISTENT SCHEDULER: one CTA per (m-block, k-split); the grid is
//      `num_splits * ceil(m / BLOCK_M)` — no L2 swizzle, no clusters, no
//      multicast (N is tiny; B is loaded identically by every CTA — the L2
//      handles the broadcast).
//   c) The epilogue stores fp32 D through a swizzled SMEM staging tile with
//      ONE TMA store per block (box = [BLOCK_N, BLOCK_M] == one swizzle atom
//      of the D map), instead of the multi-atom store_cd machinery.
//
// ---------------------------------------------------------------------------
// 4. SM90 PIPELINE (256 threads = 128 math + 128 TMA-group)
// ---------------------------------------------------------------------------
//   warp 0-3 (math warpgroup, 256 regs):
//       per stage s: wait full[s] -> LDS A fragment of stage s from SMEM
//       (128B-swizzle-aware addressing), accumulating sqr_sum partials ->
//       wgmma.wait_group<0> (previous stage's wgmma drained) -> arrive
//       empty[s-1] (SMEM of s-1 fully consumed: A by the LDS above, B by the
//       wgmma that just completed) -> issue 8 RS-wgmma (BLOCK_K=64 / K=8)
//       against B SMEM descriptors -> commit.
//       after the K loop: butterfly-reduce sqr_sum, store it to GMEM, arrive
//       empty[last], then the register epilogue (D: wgmma accumulator
//       registers -> swizzled SMEM -> TMA store 2D/3D).
//   warp 4 (TMA, 40 regs, elect_one): per stage: wait empty[s] (phase^1) ->
//       TMA A box [BLOCK_K bf16 = 128B, BLOCK_M] (one swizzle atom) + TMA B
//       box [32 fp32 = 128B, BLOCK_N] x 2 atoms -> arrive_and_expect_tx.
//       (warps 5-7 only participate in the register rebalance.)
//   Barriers: full[i] init(1) (TMA tx-count), empty[i] init(128) (every math
//   thread arrives once per stage).
//
//   A-fragment addressing (the heart of the SM90 math path): the A tile is
//   [BLOCK_M, BLOCK_K] bf16 under TMA 128B swizzle — each 128B row is one
//   swizzle span, so logical 16B bank-group g of row r sits at physical
//   group (g ^ (r % 8)).  Lane l of warp w owns rows r = 16w + l/4 (+8) and,
//   for the k8 wgmma slice at bank-group i, the two k elements
//   `i*8 + (l%4)` and `i*8 + (l%4) + 4`; the RS fragment register order is
//   (row, row+8, row @ k+4, row+8 @ k+4) per cute's MMA_64xNx8_F32TF32TF32_RS.
//
// ---------------------------------------------------------------------------
// 5. SM100 PIPELINE (256 threads = 128 MMA-group + 128 cast group)
// ---------------------------------------------------------------------------
//   warp 0 (TMA, elect_one)  : identical producer loop to SM90 (A atom + 2 B
//                              atoms -> full[s], expect_tx).
//   warp 1 (MMA)             : waits full_cast[c] (A casted into TMEM cast
//                              stage c = s % 2), then issues BLOCK_K/UMMA_K
//                              TS-MMAs walking a per-lane table of stage
//                              B-descriptor bases (shfl by stage, advance by
//                              atom/in-atom offsets); each stage commits to
//                              empty_cast[c] (TMEM A region reusable) and
//                              empty[s] (B SMEM reusable).  After the K loop
//                              one final commit -> tmem_full_barrier.
//   warp 2                   : tcgen05.alloc of the TMEM columns.
//   warp 4-7 (cast+reduce)   : per stage s: wait full[s] -> LDSM x4 the
//                              warp's 16 rows x 32 bf16 (2 bank groups) of A
//                              -> wait empty_cast[c] -> convert bf16x2->f32,
//                              square-accumulate into per-lane float2 sums,
//                              tcgen05.st.16x256b into TMEM columns
//                              `c*BLOCK_K + i*8` -> fence -> arrive
//                              full_cast[c].  After the loop: butterfly
//                              reduce + sqr_sum GMEM store.
//   epilogue (warps 0-3)     : wait tmem_full_barrier (phase 0, single use) ->
//                              32dp32b4x TMEM loads of the [BLOCK_M, BLOCK_N]
//                              accumulator (column base BLOCK_K*2, after the
//                              two A cast regions) -> swizzled v4 st.shared ->
//                              tma_store_fence -> bar.sync(128, id 0) -> one
//                              TMA store (warp 0); warp 1 deallocs TMEM.
//
//   TMEM budget: cols = align(BLOCK_K*2 + BLOCK_N) — [0, BLOCK_K*2) is the
//   double-buffered A operand (cast stages 0/1 at column c*BLOCK_K, each
//   stage holding BLOCK_K columns of f32 A), then the f32 accumulator.
//
//   B-descriptor walk (mirror of the main SM100 GEMM): B SMEM tile = 2
//   swizzle-128B atoms of [BLOCK_N, 32] fp32; make_umma_desc<..., BLOCK_K=
//   32 (=swizzle/elem), ...> builds the atom descriptor once, each lane
//   precomputes `lo + lane * stage_bytes/16`, and the MMA warp shfl-broadcasts
//   the current stage's base, advancing it by (atom, in-atom-k) offsets.
// ---------------------------------------------------------------------------

namespace dg {

// ===========================================================================
// Local PTX helpers (kept OUT of prelude.h per this repo's porting rules).
// ===========================================================================

// st.shared.v2.b32 — the SM90 D-epilogue writes accumulator pairs (8 bytes,
// one half of a 16B bank group per thread of a lane pair).
DG_DEVICE void st_shared_u32x2(uint32_t* p, uint32_t a, uint32_t b) {
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::
                 "r"(cvta_shared_to_u32(p)), "r"(a), "r"(b));
}

// wgmma.mma_async RS form, TF32 x TF32 -> FP32, m64nNk8: the A operand lives
// in 4 registers per lane (uint32 bit patterns of the f32 values — cute's
// exact convention: the kernel passes `__float_as_uint(a[i])`).  The
// prelude's wgmma.h has a float-typed variant of this instruction, but an
// "r" constraint on a *float* expression is rejected by NVRTC ("operand type
// size does not match constraint"), so the hc-prenorm port carries its own
// uint32_t wrapper here.
template <uint32_t N>
DG_DEVICE void hc_wgmma_tf32_rs(const float* a, uint64_t desc_b, float* d, uint32_t scale_d) {
    static_assert(N == 8 || N == 16 || N == 32 || N == 64 || N == 128 || N == 256,
                  "unsupported hc_wgmma_tf32_rs N");
    const uint32_t a0 = __float_as_uint(a[0]), a1 = __float_as_uint(a[1]);
    const uint32_t a2 = __float_as_uint(a[2]), a3 = __float_as_uint(a[3]);
    if constexpr (N == 8) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
            "{%0,%1,%2,%3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));
    } else if constexpr (N == 16) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8, %9, %10, %11}, %12, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));
    } else if constexpr (N == 32) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
            "{%16, %17, %18, %19}, %20, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
              "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
              "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));
    } else if constexpr (N == 64) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
            "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
              "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
              "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
              "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
              "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
              "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
              "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));
    } else if constexpr (N == 128) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
            "{%64, %65, %66, %67}, %68, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
              "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
              "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
              "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
              "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
              "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
              "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
              "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
              "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
              "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
              "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
              "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
              "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
              "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
              "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));
    } else if constexpr (N == 256) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
            "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
            "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
            "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,"
            "%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,"
            "%122,%123,%124,%125,%126,%127}, "
            "{%128, %129, %130, %131}, %132, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
              "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
              "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
              "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
              "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
              "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
              "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
              "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
              "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
              "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
              "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
              "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
              "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
              "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
              "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
              "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),
              "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
              "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
              "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
              "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
              "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
              "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),
              "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
              "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]),
              "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
              "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),
              "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
              "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
              "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
              "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
              "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));
    }
}

// Intra-warp sum over 4-lane groups (lanes l, l^1, l^2, l^3 — the lanes that
// own the same output row).  Port of math::warp_reduce_sum<4>: XOR-shuffles
// with 2 then 1, no inter-group reduce.
DG_DEVICE float warp_reduce_sum_4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// Packed bf16x2 (u32) -> two f32 (the SM100 cast path's __bfloat1622float2).
DG_DEVICE float2 bf16x2_to_float2(uint32_t packed) {
    float2 r;
    r.x = f32_from_bf16(packed & 0xffffu);
    r.y = f32_from_bf16(packed >> 16);
    return r;
}

// float2 FMA (the SM100 cast path's __ffma2_rn): per-component rn fma.
DG_DEVICE float2 ffma2_rn(float2 a, float2 b, float2 c) {
    return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
}

// SM90 D-epilogue swizzle: given the logical 16B bank-group `offset` within a
// row and the lane's row position, return the PHYSICAL bank-group index
// (0 .. kSwizzleCDMode/16 - 1) the TMA-swizzled D staging expects.
// Port of upstream sm90 `get_swizzled_bank_group_idx` (which returns an
// index, scaled by 16B at the call site).
template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
DG_DEVICE uint32_t get_swizzled_bank_group_idx(uint32_t offset, uint32_t lane_idx) {
    constexpr uint32_t kGroupsInSwizzleRange = kSwizzleMode / kSwizzleBase;
    const uint32_t bank_group_idx = offset + lane_idx * kGroupsInSwizzleRange;
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = kGroupsInSwizzleRange == kNumBankGroups;
    uint32_t row = kHasShortcut ? (offset / kNumBankGroups + lane_idx)
                                : (bank_group_idx / kNumBankGroups);
    uint32_t col = kHasShortcut ? offset : (bank_group_idx % kNumBankGroups);
    col ^= row % kGroupsInSwizzleRange;
    return (row * kNumBankGroups + col) % kGroupsInSwizzleRange;
}

// SM100 swizzle helper (A-cast LDSM addresses + D-epilogue stores): returns
// the PHYSICAL BYTE OFFSET inside the (128B-wide) swizzle-atom view.
// Port of upstream sm100 `get_swizzled_smem_offset`.
template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
DG_DEVICE uint32_t get_swizzled_smem_offset(uint32_t offset, uint32_t lane_idx) {
    // Index of the 16B bank group to be written in the atom.
    const uint32_t bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);
    // Reshape the atom view: (rows, kSwizzleMode/16) -> (x, 8 groups) and
    // XOR-swizzle the group by the (atom-)row.
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
    uint32_t row = kHasShortcut ? (offset / kNumBankGroups + lane_idx)
                                : (bank_group_idx / kNumBankGroups);
    uint32_t col = kHasShortcut ? offset : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);
    return row * 128 + col * kSwizzleBase;
}

// K-major 2D TMA load with swizzle-atom splitting (port of upstream
// tma::copy<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode> for the 2D, 1-CTA case
// this kernel needs).  The inner extent is split into `kSwizzleMode`-byte
// atoms; atom i lands at `BLOCK_OUTER * atom_bytes` in SMEM and starts at
// inner coordinate `inner_idx + i * atom_elems`.  For this kernel:
//   A: BLOCK_INNER=64 bf16 = 128B = one atom;  B: 64 fp32 = 2 atoms of 32.
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, uint32_t kElemSize>
DG_DEVICE void hc_tma_copy(const TmaMap* map, Barrier* bar, uint8_t* smem_ptr,
                           uint32_t inner_idx, uint32_t outer_idx) {
    constexpr uint32_t kAtomElems = kSwizzleMode / kElemSize;   // inner elements per atom
    constexpr uint32_t kAtomBytes = kSwizzleMode;               // == kAtomElems * kElemSize
    constexpr uint32_t kNumAtoms = BLOCK_INNER / kAtomElems;    // exact split (asserted below)
    static_assert(BLOCK_INNER % kAtomElems == 0, "TMA inner must split into whole atoms");
    #pragma unroll
    for (uint32_t i = 0; i < kNumAtoms; ++i)
        tma_load_2d(map, bar, smem_ptr + i * BLOCK_OUTER * kAtomBytes,
                    kEvictNormalHint, inner_idx + i * kAtomElems, outer_idx);
}

// ===========================================================================
// hc_prenorm_sm90_impl — Hopper: wgmma RS-form TF32, A via registers.
// ===========================================================================
// Template contract (mirrors the upstream launcher):
//   SHAPE_N / SHAPE_K : compile-time N/K (K must be % BLOCK_K; the kernel
//                       derives its split-K constants from SHAPE_K),
//   BLOCK_M = 64, BLOCK_K = 64, BLOCK_N in {16, 32} (kSwizzleCDMode/4),
//   kNumSplits >= 1 (grid = num_splits * ceil(m / BLOCK_M)),
//   kNumMathThreads = 128 (one warpgroup), kNumTMAThreads = 128.
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
__launch_bounds__(kNumMathThreads + kNumTMAThreads, 1) __global__
void hc_prenorm_sm90_impl(const uint32_t shape_m,
                          const TmaMap tensor_map_a, const TmaMap tensor_map_b,
                          const TmaMap tensor_map_d, float* sqr_sum) {
    // Both impls live in this one TU; NVRTC compiles a single arch at a time,
    // so guard each body to its own family (wgmma f32 exists only < sm_100,
    // tcgen05 only >= sm_100).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
    // A: bf16, B: fp32; both K-major, both 128B-swizzled (BLOCK_K spans one
    // atom for A, two for B).
    constexpr uint32_t kSwizzleAMode = (BLOCK_K * 2 < 128) ? BLOCK_K * 2 : 128;
    constexpr uint32_t kSwizzleBMode = (BLOCK_K * 4 < 128) ? BLOCK_K * 4 : 128;
    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K");
    DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");
    DG_STATIC_ASSERT(kSwizzleBMode == 128, "Invalid swizzle B mode");
    DG_STATIC_ASSERT(kSwizzleCDMode / 4 == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMathThreads == 128, "Invalid MMA threads");

    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    // Align to 1024 bytes for the swizzle-128B tiles.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // SMEM layout: [D staging][A stages][B stages][full|empty barriers].
    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * 2;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * 4;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "D staging must align to 1024B");

    if (warp_idx == 0 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        prefetch_tma_map(&tensor_map_d);
    }

    float* smem_cd = (float*)smem_buffer;
    auto smem_a_of = [&](uint32_t i) {
        return smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE;
    };
    auto smem_b_of = [&](uint32_t i) {
        return smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE
             + i * SMEM_B_SIZE_PER_STAGE;
    };
    Barrier* barrier_start = (Barrier*)(smem_buffer + SMEM_CD_SIZE
                                        + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barrier_of = [&](uint32_t i) { return barrier_start + i; };
    auto empty_barrier_of = [&](uint32_t i) { return barrier_start + kNumStages + i; };

    // Barrier init: full expects 1 arrival + the TMA tx byte count; empty is
    // released by all 128 math threads (each arrives once per consumed stage).
    if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barrier_of(i)->init(1);
            empty_barrier_of(i)->init(kNumMathThreads);
        }
        fence_barrier_init();
    }
    __syncthreads();

    // Split-K decomposition (compile-time from SHAPE_K): the first
    // `kRemainKBlocks` splits get one extra K block.
    constexpr uint32_t kNumKBlocks = ceil_div_u32(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit
                               + dg_min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;  // sqr_sum slice base
    const uint32_t num_total_stages = kNumKBlocksPerSplit
                                    + (k_split_idx < kRemainKBlocks ? 1u : 0u);
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 256;

    // Wait for the primary kernel (PDL); no-op when launched without PDL.
    griddepcontrol_wait();

    if (warp_idx >= kNumMathThreads / 32) {
        // ================= TMA producer warpgroup =================
        // (Register rebalance for the whole group — setmaxnreg is a
        // warp-aligned instruction; only warp 4/lane 0 issues TMA.)
        setmaxnreg_dec<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32 && elect_one_sync()) {
            for (uint32_t s = 0; s < num_total_stages; ++s) {
                const uint32_t stage_idx = s % kNumStages;
                empty_barrier_of(stage_idx)->wait(((s / kNumStages) & 1) ^ 1);

                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t k_idx = k_offset + s * BLOCK_K;
                hc_tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, 2>(
                    &tensor_map_a, full_barrier_of(stage_idx), smem_a_of(stage_idx), k_idx, m_idx);
                hc_tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, 4>(
                    &tensor_map_b, full_barrier_of(stage_idx), smem_b_of(stage_idx), k_idx, 0);
                full_barrier_of(stage_idx)->arrive_and_expect_tx(
                    SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
            // Tear-down: watch every stage's final release before exiting
            // (keeps the distributed barrier state consistent).
            for (uint32_t s = num_total_stages; s < num_total_stages + kNumStages; ++s) {
                const uint32_t stage_idx = s % kNumStages;
                empty_barrier_of(stage_idx)->wait(((s / kNumStages) & 1) ^ 1);
            }
        }
    } else {
        // ================= Math warpgroup =================
        setmaxnreg_inc<kNumMathRegisters>();

        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(BLOCK_K * 2 == kSwizzleAMode, "Invalid block K");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;   // 16 rows per warp
        constexpr uint32_t WGMMA_K = 8;
        constexpr uint32_t kNumAccum = BLOCK_N / 2;          // m64nN f32 regs/lane

        float accum[kNumAccum] = {0};

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / 2;  // 8 bf16
        constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;     // 8 k-slices
        float sqr_sum_acc_0 = 0;   // row (warp*16 + lane/4)     partial
        float sqr_sum_acc_1 = 0;   // row (warp*16 + lane/4 + 8) partial

        // Upstream unroll hint: full unroll for shallow pipelines, halve for
        // deep ones (code-size vs. ILP).
        #pragma unroll (kNumStages < 8 ? kNumStages : kNumStages / 2)
        for (uint32_t s = 0; s < num_total_stages; ++s) {
            const uint32_t stage_idx = s % kNumStages;
            full_barrier_of(stage_idx)->wait((s / kNumStages) & 1);

            constexpr uint32_t kNumRegPerWgmma = 64 * WGMMA_K / 128;        // 4
            constexpr uint32_t kNumWgmmaPerBlockK = BLOCK_K / WGMMA_K;      // 8
            float a[kNumRegPerWgmma * kNumWgmmaPerBlockK];                  // 32 f32

            // ---- A: SMEM (128B-swizzled bf16) -> registers (f32) ----
            // Lane l of warp w owns rows r = 16w + l/4 and r + 8; the k8
            // wgmma slice at logical bank group i holds k = i*8 + (l%4) and
            // +4, which the TMA swizzle placed at physical group
            // (i ^ (r % 8)).  Fragment order: (r, r+8, r@k+4, r+8@k+4).
            const uint32_t row = warp_idx * 16 + lane_idx / 4;
            const bf16_raw* sa = (const bf16_raw*)smem_a_of(stage_idx);
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++i) {
                const uint32_t bank_group_idx = (row ^ i) % 8;
                const bf16_raw* ptr_upper = sa + row * BLOCK_K
                                          + bank_group_idx * kNumElemsPerBankGroup;
                const bf16_raw* ptr_lower = sa + (row + 8) * BLOCK_K
                                          + bank_group_idx * kNumElemsPerBankGroup;
                const uint32_t elem_offset = lane_idx % 4;
                const float v0 = f32_from_bf16(ptr_upper[elem_offset]);
                const float v2 = f32_from_bf16(ptr_upper[elem_offset + 4]);
                const float v1 = f32_from_bf16(ptr_lower[elem_offset]);
                const float v3 = f32_from_bf16(ptr_lower[elem_offset + 4]);
                a[i * 4 + 0] = v0;
                a[i * 4 + 1] = v1;
                a[i * 4 + 2] = v2;
                a[i * 4 + 3] = v3;
                // Pre-norm statistic: exact squares of the exact bf16 values.
                sqr_sum_acc_0 += v0 * v0 + v2 * v2;
                sqr_sum_acc_1 += v1 * v1 + v3 * v3;
            }

            // Previous stage's wgmma must be drained before (a) reusing the
            // accumulator registers and (b) releasing its SMEM (B is read by
            // that wgmma; A of s-1 was consumed by its LDS above).
            warpgroup_wait_group<0>();
            if (s > 0)
                empty_barrier_of((s - 1) % kNumStages)->arrive();

            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();

            // ---- B: SMEM descriptor walk over the 2 swizzle atoms ----
            constexpr uint32_t kNumElemsInSwizzleRange = 128 / 4;                    // 32 f32
            constexpr uint32_t kNumWgmmaInSwizzleRange = kNumElemsInSwizzleRange / WGMMA_K;  // 4
            DG_STATIC_ASSERT(BLOCK_K % kNumElemsInSwizzleRange == 0, "Invalid block K");
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_K / kNumElemsInSwizzleRange; ++i) {
                #pragma unroll
                for (uint32_t k = 0; k < kNumWgmmaInSwizzleRange; ++k) {
                    GmmaDescriptor b_desc = make_gmma_desc(
                        smem_b_of(stage_idx)
                            + (i * BLOCK_N * kNumElemsInSwizzleRange + k * WGMMA_K) * 4,
                        GmmaLayoutType::B128, 0, 1024);
                    hc_wgmma_tf32_rs<BLOCK_N>(
                        a + (i * kNumWgmmaInSwizzleRange + k) * kNumRegPerWgmma,
                        b_desc.desc_, accum, 1u);
                }
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        }

        // ---- sqr_sum epilogue: 4-lane butterfly + scatter to GMEM ----
        const float reduced_sum_0 = warp_reduce_sum_4(sqr_sum_acc_0);
        const float reduced_sum_1 = warp_reduce_sum_4(sqr_sum_acc_1);
        const uint32_t m_idx = m_block_idx * BLOCK_M
                             + warp_idx * BLOCK_M_PER_WARP + lane_idx / 4;
        if (lane_idx % 4 == 0) {
            if (m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum_0;
            if (m_idx + 8 < shape_m)
                sqr_sum[m_offset + m_idx + 8] = reduced_sum_1;
        }

        // Release the final stage (its wgmma is drained).
        warpgroup_wait_group<0>();
        empty_barrier_of((num_total_stages - 1) % kNumStages)->arrive();

        // ---- D epilogue: accumulator registers -> swizzled SMEM -> TMA ----
        // Each lane pair (same lane/2) writes one 16B bank group: the even
        // lane covers its low 8 bytes, the odd lane the high 8.  WGMMA's
        // register layout puts acc[j*4+{0,1}] at row r0, cols 8j+col*2+{0,1}
        // and acc[j*4+{2,3}] at row r0+8 — matching the two stores below
        // (8*kSwizzleCDMode bytes = 8 rows apart).
        const uint32_t is_odd_pair = lane_idx / 2 % 2;
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t reordered_pair_idx = is_odd_pair * 8 + row_idx;
        uint8_t* shifted_smem_ptr = smem_buffer
            + (warp_idx * BLOCK_M_PER_WARP + row_idx) * kSwizzleCDMode  // row offset
            + (lane_idx % 2) * 8;                                       // half bank group

        #pragma unroll
        for (uint32_t i = 0; i < (kSwizzleCDMode / 4) / 4; i += 2) {
            const uint32_t bank_group_idx =
                get_swizzled_bank_group_idx<kSwizzleCDMode>(i + is_odd_pair, reordered_pair_idx);
            uint8_t* smem_ptr = shifted_smem_ptr + bank_group_idx * kNumBankGroupBytes;
            const uint32_t* values = (const uint32_t*)(accum + i * 2);
            st_shared_u32x2((uint32_t*)smem_ptr, values[0], values[1]);
            st_shared_u32x2((uint32_t*)(smem_ptr + 8 * kSwizzleCDMode), values[2], values[3]);
        }
        tma_store_fence();
        named_barrier_sync(kNumMathThreads, 1);

        if (warp_idx == 0 && elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                tma_store_2d(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                tma_store_3d(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            tma_store_arrive();
        }
    }
#endif  // __CUDA_ARCH__ >= 900 && < 1000
}

// ===========================================================================
// hc_prenorm_sm100_impl — Blackwell: tcgen05 TS-form TF32, A via TMEM.
// ===========================================================================
// Template contract (mirrors the upstream launcher):
//   BLOCK_M = 64 (the cast path asserts it; 128 is wired in the epilogue but
//               the upstream launcher only ever instantiates 64),
//   BLOCK_N in {16, 32}, BLOCK_K = 64, kNumCastStages = 2 (hardcoded).
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMMAThreads, uint32_t kNumCastAndReduceThreads>
__launch_bounds__(kNumMMAThreads + kNumCastAndReduceThreads, 1) __global__
void hc_prenorm_sm100_impl(const uint32_t shape_m,
                           const TmaMap tensor_map_a, const TmaMap tensor_map_b,
                           const TmaMap tensor_map_d, float* sqr_sum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr uint32_t kNumCastStages = 2;  // TMEM A double buffer
    constexpr uint32_t kSwizzleAMode = (BLOCK_K * 2 < 128) ? BLOCK_K * 2 : 128;
    constexpr uint32_t kSwizzleBMode = (BLOCK_K * 4 < 128) ? BLOCK_K * 4 : 128;
    DG_STATIC_ASSERT(kNumCastStages <= kNumStages, "Invalid cast stages");
    DG_STATIC_ASSERT(kSwizzleCDMode / 4 == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMMAThreads == 128, "Invalid MMA threads");

    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * 2;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * 4;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "D staging must align to 1024B");

    // TMEM: [0, BLOCK_K*kNumCastStages) = A operand (2 cast buffers of
    // BLOCK_K f32 columns), then the [BLOCK_M, BLOCK_N] f32 accumulator.
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols && kNumTmemCols <= 512, "Invalid tensor memory columns");

    if (warp_idx == 0 && elect_one_sync()) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        prefetch_tma_map(&tensor_map_d);
    }

    float* smem_cd = (float*)smem_buffer;
    auto smem_a_of = [&](uint32_t i) {
        return smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE;
    };
    auto smem_b_of = [&](uint32_t i) {
        return smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE
             + i * SMEM_B_SIZE_PER_STAGE;
    };
    // SMEM layout: [D][A stages][B stages][full | full_cast | empty | empty_cast
    //               (x kNumStages each)][tmem_full][tmem ptr].
    Barrier* barrier_start = (Barrier*)(smem_buffer + SMEM_CD_SIZE
                                        + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barrier_of = [&](uint32_t i) { return barrier_start + i; };
    auto full_cast_barrier_of = [&](uint32_t i) { return barrier_start + kNumStages + i; };
    auto empty_barrier_of = [&](uint32_t i) { return barrier_start + 2 * kNumStages + i; };
    auto empty_cast_barrier_of = [&](uint32_t i) { return barrier_start + 3 * kNumStages + i; };
    Barrier* tmem_full_barrier = barrier_start + kNumStages * 4;
    uint32_t* tmem_ptr_in_smem = (uint32_t*)(barrier_start + kNumStages * 4 + 1);

    if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barrier_of(i)->init(1);                              // TMA tx
            full_cast_barrier_of(i)->init(kNumCastAndReduceThreads);  // 128 cast threads
            empty_barrier_of(i)->init(1);                             // 1 umma commit
            empty_cast_barrier_of(i)->init(1);                        // 1 umma commit
        }
        tmem_full_barrier->init(1);
        fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate TMEM (single fully-active warp; base lands in SMEM — the
        // kernel's 0-based column addressing assumes a fresh allocation).
        tmem_alloc_1sm(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    constexpr uint32_t kNumKBlocks = ceil_div_u32(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit
                               + dg_min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit
                                    + (k_split_idx < kRemainKBlocks ? 1u : 0u);

    griddepcontrol_wait();

    if (warp_idx < kNumMMAThreads / 32) {
        // ================= TMA load warp =================
        if (warp_idx == 0 && elect_one_sync()) {
            for (uint32_t s = 0; s < num_total_stages; ++s) {
                const uint32_t stage_idx = s % kNumStages;
                empty_barrier_of(stage_idx)->wait(((s / kNumStages) & 1) ^ 1);

                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t k_idx = k_offset + s * BLOCK_K;
                hc_tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, 2>(
                    &tensor_map_a, full_barrier_of(stage_idx), smem_a_of(stage_idx), k_idx, m_idx);
                hc_tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, 4>(
                    &tensor_map_b, full_barrier_of(stage_idx), smem_b_of(stage_idx), k_idx, 0);
                full_barrier_of(stage_idx)->arrive_and_expect_tx(
                    SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }

        // ================= MMA issue warp =================
        if (warp_idx == 1) {
            constexpr uint32_t UMMA_M = BLOCK_M;
            constexpr uint32_t UMMA_N = BLOCK_N;
            constexpr uint32_t UMMA_K = 32 / 4;                    // 8 tf32 elems
            constexpr uint32_t BLOCK_SWIZZLED_BK = kSwizzleBMode / 4;  // 32 f32/atom
            // Instruction descriptor: a/b format = TF32 (F32F16Format code 2),
            // C/D = F32 (code 1), M/N dims packed >>4/>>3, K-major operands.
            InstrDescriptor instr_desc = make_instr_desc_f16(
                2 /*TF32*/, 2 /*TF32*/, 1 /*F32*/, UMMA_M, UMMA_N, MAJOR_K, MAJOR_K);
            const uint64_t runtime_instr_desc = make_runtime_instr_desc(instr_desc);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            // B stage-0 descriptor + per-lane table of stage bases (lane i
            // holds stage i's `lo`; broadcast by shfl at issue time).
            SmemDescriptor b_desc = make_umma_desc<MAJOR_K, BLOCK_N, BLOCK_SWIZZLED_BK,
                                                   kSwizzleBMode, 1, 4>(smem_b_of(0), 0, 0);
            const uint32_t b_desc_lo = lane_idx < kNumStages
                ? b_desc.lo + lane_idx * (SMEM_B_SIZE_PER_STAGE / 16) : 0u;

            // MMA shape checks (mirror upstream's CUTLASS-trait asserts).
            DG_STATIC_ASSERT((UMMA_M == 64  && UMMA_N %  8 == 0 &&  8 <= UMMA_N && UMMA_N <= 256) ||
                             (UMMA_M == 128 && UMMA_N %  8 == 0 &&  8 <= UMMA_N && UMMA_N <= 256) ||
                             (UMMA_M == 256 && UMMA_N % 16 == 0 && 16 <= UMMA_N && UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // The stage loop must NOT be unrolled (dynamic trip count; the
            // inner k loop is the unrolled one).
            for (uint32_t s = 0; s < num_total_stages; ++s) {
                const uint32_t stage_idx = s % kNumStages;
                const uint32_t cast_stage_idx = s % kNumCastStages;
                full_cast_barrier_of(cast_stage_idx)->wait((s / kNumCastStages) & 1);
                tcgen05_after_thread_sync();

                const uint32_t b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, stage_idx);
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                    // B tile = 2 swizzle atoms on K; walk (atom, in-atom k).
                    const uint32_t atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
                    const uint32_t in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
                    const uint32_t offset = atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK;
                    b_desc.lo = advance_umma_desc_lo<MAJOR_K, BLOCK_N, kSwizzleBMode, 1, 4>(
                        b_desc_base_lo, offset, in_atom_idx);
                    // TS MMA: D at column BLOCK_K*kNumCastStages (after the A
                    // region), A at cast buffer `cast_stage_idx`, column k*8.
                    // Accumulate except on the very first k of the first stage.
                    if (elect_one_sync())
                        mma_tf32_ts_1sm(BLOCK_K * kNumCastStages,
                                        BLOCK_K * cast_stage_idx + k * UMMA_K,
                                        b_desc.desc_, (s > 0 || k > 0) ? 1u : 0u,
                                        runtime_instr_desc);
                }
                __syncwarp();
                // Commit: MMA drained the TMEM A cast buffer and the B stage.
                if (elect_one_sync()) {
                    umma_arrive_1sm(empty_cast_barrier_of(cast_stage_idx));
                    umma_arrive_1sm(empty_barrier_of(stage_idx));
                }
                __syncwarp();
            }

            // Final commit -> epilogue (accumulator TMEM is ready).
            if (elect_one_sync())
                umma_arrive_1sm(tmem_full_barrier);
            __syncwarp();
        }

        // ================= Epilogue (all 128 MMA-group threads) =================
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / 4;  // 4 fp32
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
        // TMEM accumulator layouts: F (M=64, one lane row pair) / D (M=128).
        DG_STATIC_ASSERT(BLOCK_M == 64 || BLOCK_M == 128, "Invalid block M");

        // Single-use barrier: phase 0.
        tmem_full_barrier->wait(0);
        tcgen05_after_thread_sync();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_N / kNumElemsPerBankGroup; ++i) {
            const uint32_t tmem_addr = BLOCK_K * kNumCastStages + i * kNumElemsPerBankGroup;
            uint8_t* smem_ptr = smem_buffer
                + warp_idx * (BLOCK_M / 4) * kSwizzleCDMode          // warp rows
                + get_swizzled_smem_offset<kSwizzleCDMode>(i, lane_idx);  // in-atom

            uint32_t values[kNumElemsPerBankGroup];
            tmem_load_32dp32b_x4(tmem_addr, values[0], values[1], values[2], values[3]);
            fence_view_async_tmem_load();
            // Layout F (M=64): only the low half of each 32-lane load holds
            // valid accumulator rows for this warp's slice.
            if (BLOCK_M == 128 || (BLOCK_M == 64 && lane_idx < 16))
                st_shared_u32x4((uint32_t*)smem_ptr, values[0], values[1], values[2], values[3]);
            if (BLOCK_M == 64)
                __syncwarp();
        }

        tma_store_fence();
        named_barrier_sync(kNumMMAThreads, 0);
        if (warp_idx == 0 && elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                tma_store_2d(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                tma_store_3d(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            tma_store_arrive();
        }

        // Deallocate TMEM from warp 1 (warp 0 is waiting on the TMA store).
        if (warp_idx == 1)
            tmem_dealloc_1sm(0, kNumTmemCols);
    } else {
        // ================= Cast + square-reduce warpgroup (warps 4-7) ========
        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(kNumCastAndReduceThreads == 128, "Invalid cast-and-reduce threads");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;  // 16 rows per warp
        const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;

        DG_STATIC_ASSERT(BLOCK_K * 2 == kSwizzleAMode, "Invalid block K");

        float2 sum[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
        #pragma unroll (kNumStages)
        for (uint32_t s = 0; s < num_total_stages; ++s) {
            const uint32_t stage_idx = s % kNumStages;
            full_barrier_of(stage_idx)->wait((s / kNumStages) & 1);

            // ---- LDSM the warp's 16x32 bf16 slice of A (2 bank groups) ----
            constexpr uint32_t kNumBankGroupBytes = 16;
            constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / 2;  // 8 bf16
            constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;     // 8
            uint8_t* smem_base_ptr = smem_a_of(stage_idx)
                                   + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;
            DG_STATIC_ASSERT(kNumLoads % 2 == 0, "Invalid number of loads");

            uint32_t uint32_values[2][kNumLoads];
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; i += 2) {
                // Lanes 0-15 point at bank group (i, rows 0-15); lanes 16-31
                // at group (i+1, same rows) — LDSM x4 turns that into
                // [upper/lower 8 rows] x [group i / group i+1] fragments.
                uint8_t* smem_ptr = smem_base_ptr
                    + get_swizzled_smem_offset<kSwizzleAMode>(i + lane_idx / 16, lane_idx % 16);
                ldsm_x4_b16_n(cvta_shared_to_u32(smem_ptr),
                              uint32_values[0][i], uint32_values[1][i],
                              uint32_values[0][i + 1], uint32_values[1][i + 1]);
            }

            // TMEM A region double buffer: wait for the MMA to drain it.
            const uint32_t cast_stage_idx = s % kNumCastStages;
            empty_cast_barrier_of(cast_stage_idx)->wait(((s / kNumCastStages) & 1) ^ 1);

            // ---- Cast bf16->f32, square-reduce, store into TMEM ----
            float2 fp32x2_values[2][kNumLoads];
            const uint32_t* upper_view = (const uint32_t*)&fp32x2_values[0][0];
            const uint32_t* lower_view = (const uint32_t*)&fp32x2_values[1][0];
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++i) {
                #pragma unroll
                for (uint32_t u = 0; u < 2; ++u) {
                    fp32x2_values[u][i] = bf16x2_to_float2(uint32_values[u][i]);
                    sum[u] = ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u]);
                }
                // One 16x256b store covers 8 columns: upper/lower 8-row
                // halves x 2 k-elements per lane.
                const uint32_t idx_0 = i * 2, idx_1 = i * 2 + 1;
                tmem_store_16dp256b_x1(cast_stage_idx * BLOCK_K + i * 8,
                                       upper_view[idx_0], upper_view[idx_1],
                                       lower_view[idx_0], lower_view[idx_1]);
            }
            // Make the TMEM stores visible before signalling the MMA warp.
            fence_view_async_tmem_store();
            fence_view_async_shared();
            tcgen05_before_thread_sync();
            full_cast_barrier_of(cast_stage_idx)->arrive();
        }

        // Intra-warp reduction and sqr_sum write-back (u selects the upper /
        // lower 8-row half of the warp's block, matching the LDSM halves).
        #pragma unroll
        for (uint32_t u = 0; u < 2; ++u) {
            const float reduced_sum = warp_reduce_sum_4(sum[u].x + sum[u].y);
            const uint32_t m_idx = m_block_idx * BLOCK_M
                                 + sub_warp_idx * BLOCK_M_PER_WARP + lane_idx / 4 + u * 8;
            if (lane_idx % 4 == 0 && m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum;
        }
    }
#endif  // __CUDA_ARCH__ >= 1000
}

} // namespace dg
