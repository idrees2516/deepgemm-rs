// DeepGEMM-RS: unified SM100 FP8/FP4/BF16 GEMM kernel.
//
// Faithful port of upstream DeepGEMM `sm100_fp8_fp4_gemm_1d1d.cuh` +
// `epilogue/sm100_store_cd{,_swap_ab}.cuh`, adapted to a single self-contained
// NVRTC translation unit (compiled with prelude.h prepended).
//
// ===========================================================================
// PIPELINE — how one output tile flows through the engines
// ===========================================================================
// Per k-block iteration (stage s = k_idx % kNumStages), three engines run
// concurrently on DIFFERENT stages:
//
//   k:      0        1        2        3        4        5     ...
//          ┌────────┬────────┬────────┬────────┬────────┬────────┐
//   TMA  : │load s0 │load s1 │load s2 │load s3 │  ... (SF tiles first:
//   (w0)  │ A+B+SF │        │        │        │        │  SF is expected at
//          └────────┴────────┴────────┴────────┴────────┴────────┘ the MMA issue
//   MMA  :          │mma(s0) │mma(s1) │mma(s2) │   ... (w1 waits full[s],
//   (w1)  :          │+UTCCP  │        │        │        │  leader CTA only;
//          └────────┴────────┴────────┴────────┴────────┴────────┘ tcgen05.commit
//   EPI  :                   │drain s0│drain s1│  ... (w4..7 wait empty[s],
//   (w4-7):                   │TMEM->SM│->TMA   │        │  release TMEM col)
//          └────────────────────────────────────────────────────────┘
//
// Barrier topology per stage s (phase = (k_idx / kNumStages) & 1):
//   full[s]    : TMA completion (byte-count) -> MMA issuer + SF transposers
//   empty[s]   : epilogue drained the PREVIOUS use of this SMEM slot ->
//                TMA producer may overwrite it
//   tmem_empty : epilogue finished the TMEM region -> next MMA may write it
//   (2-CTA cluster: buddy CTA's barriers are arrived via mapa when needed;
//    the leader CTA issues the joint MMA for both, with umma_arrive
//    committing to each CTA's tmem barrier.)
//
// Swapped-AB mode (m-grouped): the MMA computes (B^T A)^T instead — operand
// roles exchange (BLOCK_M becomes the UMMA_N axis), which lets tiny per-expert
// M hit full UMMA_M=128 tiles through the N side. The epilogue then reads
// TMEM with the 16x256b layout and STSMs into SMEM transposed.
//
// Numeric path (see prelude.h §6): tcgen05.mma kind::mxf8f6f4 (FP8) or
// kind::mxf4 (packed FP4, BLOCK_K=256 = 64 UMMA_K steps of 4 nibbles) with
// SFs flowing SMEM -> (warp transpose) -> UTCCP -> TMEM SF columns.
// ===========================================================================
//
// Warp specialization (256 threads = 128 non-epilogue + 128 epilogue):
//   warp 0          : TMA load producer (elect_one)
//   warp 1          : MMA issue (leader CTA only, elect_one) + SF UTCCP
//   warp 2 (and 3)  : SF SMEM transpose for UTCCP (when SF_BLOCK_K == 2)
//   warps 4..7      : epilogue (TMEM -> SMEM -> TMA store)
//
// 2-CTA cluster MMA (UMMA_M = 256): the pair's A rows/B columns are split
// across the two CTAs' SMEMs; the MMA descriptor (issued by the leader) reads
// both CTAs at the same SMEM offsets and writes each CTA's TMEM with its half
// of the 256 output rows.

namespace dg {

// ---------------------------------------------------------------------------
// TMA copies with atom splitting (port of deep_gemm/common/tma_copy.cuh, 2D/3D)
// ---------------------------------------------------------------------------
// K-major: inner = K (BLOCK_K logical elements), outer = MN rows.
// MN-major: inner = MN (LOAD_BLOCK_MN elements), outer = K.
// The box inner extent is split into swizzle atoms; consecutive atoms land at
// `BLOCK_OUTER * atom_bytes` offsets in SMEM.
template <bool kIsKMajor, bool kIs3D>
DG_DEVICE void tma_copy_ab(const TmaMap* map, Barrier* bar, uint8_t* smem_ptr,
                           uint32_t num_multicast,
                           uint32_t inner_idx, uint32_t outer_idx, uint32_t batch_idx,
                           uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
                           uint32_t kSwizzleMode, uint32_t kWireElemSize, uint32_t kPackFactor) {
    // Inner span in bytes.
    const uint32_t inner_bytes_full = BLOCK_INNER * kWireElemSize / kPackFactor;
    const uint32_t atom_bytes = kSwizzleMode == 0 ? inner_bytes_full : kSwizzleMode;
    const uint32_t num_atoms = inner_bytes_full / atom_bytes;  // exact by construction
    const uint32_t atom_logical_elems = atom_bytes * kPackFactor / kWireElemSize;

    #pragma unroll 4
    for (uint32_t i = 0; i < num_atoms; ++i) {
        uint8_t* dst = smem_ptr + i * BLOCK_OUTER * atom_bytes;
        const uint32_t c_inner = inner_idx + i * atom_logical_elems;
        if (kIs3D) {
            if (num_multicast == 1) {
                // 3D plain load
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%3, %4, %5}], [%2], %6;"
                    :: "r"(cvta_shared_to_u32(dst)), "l"(map), "r"(cvta_shared_to_u32(&bar->barrier_)),
                       "r"(c_inner), "r"(outer_idx), "r"(batch_idx), "l"(kEvictNormalHint)
                    : "memory");
            } else {
                asm volatile(
                    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%3, %4, %5}], [%2], %6;"
                    :: "r"(cvta_shared_to_u32(dst)), "l"(map), "r"(cvta_shared_to_u32(&bar->barrier_)),
                       "r"(c_inner), "r"(outer_idx), "r"(batch_idx), "l"(kEvictNormalHint)
                    : "memory");
            }
        } else {
            if (num_multicast == 1) tma_load_2d(map, bar, dst, kEvictNormalHint, c_inner, outer_idx);
            else tma_load_2d_2sm(map, bar, dst, kEvictNormalHint, c_inner, outer_idx);
        }
    }
}

// SF tile load: box = (inner: SF_BLOCK_MN along MN, outer: SF_BLOCK_K rows), no swizzle.
DG_DEVICE void tma_load_2d_sf(const TmaMap* map, Barrier* bar, uint32_t* smem,
                              uint32_t mn_idx, uint32_t k_idx) {
    tma_load_2d(map, bar, smem, kEvictNormalHint, mn_idx, k_idx);
}

// ---------------------------------------------------------------------------
// Epilogues
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode, uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads,
          uint32_t kNumOverlappedTmemCols, GemmType kGemmType, bool kWithAccumulation,
          bool kCdIsFloat, uint32_t kCdElemSize, bool kWithOutputSF, typename Smem>
DG_DEVICE void store_cd(Smem& smem, uint32_t& tma_stage_idx, uint32_t tmem_base_addr,
                        uint32_t base_m_idx, uint32_t base_n_idx, uint32_t batch_idx,
                        uint32_t epilogue_warp_idx, uint32_t lane_idx,
                        bool reverse_store_order,
                        const Barrier* tmem_overlap_barrier, const Barrier* tmem_empty_barrier,
                        const TmaMap& tensor_map_cd,
                        uint32_t* sfd, uint32_t sfd_stride, uint32_t shape_m, uint32_t shape_n) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / kCdElemSize;
    constexpr uint32_t kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
    constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerBankGroup;
    constexpr uint32_t kNumOverlapLoads = (kNumOverlappedTmemCols + kNumElemsPerBankGroup - 1) / kNumElemsPerBankGroup;

    // ---- Fused QuantizeToFP8 (upstream `epilogue::operators::QuantizeToFP8`) ----
    // Casts the fp32 accumulator to E4M3 with dynamic per-row, per-32-column
    // UE8M0 scale factors, written packed into `sfd` in the same TMA-aligned
    // MN-major layout the GEMM accepts for SFA. The accumulator is rounded
    // into BF16 *before* amax/scale/cast, so the output bitwise matches a
    // BF16 D followed by the standalone per-token cast kernel.
    constexpr uint32_t kSFGranN = 32;
    DG_STATIC_ASSERT(!kWithOutputSF || (kCdElemSize == 1 && !kCdIsFloat && !kWithAccumulation),
                     "QuantizeToFP8 requires a direct E4M3 D");
    DG_STATIC_ASSERT(!kWithOutputSF || STORE_BLOCK_N % kSFGranN == 0,
                     "A store must cover complete SF groups");

    for (uint32_t w = 0; w < kNumMWaves; ++w) {
        for (uint32_t s = 0; s < kNumStores; ++s, tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages) {
            const uint32_t store_idx = reverse_store_order ? kNumStores - 1 - s : s;
            uint8_t* smem_base_ptr = (uint8_t*)smem->cd[tma_stage_idx];

            if (epilogue_warp_idx == 0) tma_store_wait<kNumTMAStoreStages - 1>();
            named_barrier_sync(kNumUMMAStoreThreads, 8);

            const uint32_t m_idx = base_m_idx + w * STORE_BLOCK_M;
            const uint32_t n_idx = base_n_idx + store_idx * STORE_BLOCK_N;

            const auto get_swizzled_smem_ptr = [&](uint32_t bank_group_idx) -> uint8_t* {
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                const uint32_t shifted_idx = bank_group_idx + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                uint32_t row = kHasShortcut ? (bank_group_idx / 8 + lane_idx) : (shifted_idx / 8);
                uint32_t col = kHasShortcut ? bank_group_idx : (shifted_idx % 8);
                col ^= row % (kSwizzleCDMode / 16);
                return smem_base_ptr
                     + epilogue_warp_idx * 32 * kSwizzleCDMode
                     + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
            };

            if (kWithOutputSF) {
                // One SF group = 32 consecutive N values = two 16-value bank
                // groups, both belonging to this lane's row (`lane = row`).
                // `fg` is the flat SF-group index inside the store; under
                // `reverse_store_order` groups (and their halves) are loaded
                // in descending column order so the overlap release points
                // keep their original issue-order meaning, but values are
                // assembled and stored by COLUMN index.
                constexpr uint32_t kNumSFGroupsPerStore = STORE_BLOCK_N / kSFGranN;
                uint32_t issue_cnt = 0;  // issue-order counter (releases)
                for (uint32_t g = 0; g < kNumSFGroupsPerStore; ++g) {
                    const uint32_t fg = reverse_store_order ? (kNumSFGroupsPerStore - 1 - g) : g;
                    uint32_t vals[kSFGranN];
                    #pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t i = reverse_store_order
                            ? (kNumLoads - 1 - (g * 2 + half)) : (fg * 2 + half);
                        const uint32_t tmem_addr = tmem_base_addr
                                                 + w * BLOCK_N
                                                 + store_idx * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                        uint32_t raw[16];
                        tmem_load_32dp32b_x16(tmem_addr, raw);
                        fence_view_async_tmem_load();
                        // Assemble by column: odd bank group = upper 16 cols.
                        const uint32_t dst_base = (i & 1u) ? 16u : 0u;
                        #pragma unroll
                        for (uint32_t j = 0; j < 16; ++j) vals[dst_base + j] = raw[j];

                        if (kNumOverlapLoads > 0) {
                            if (w == 0 && s == 0 && issue_cnt + 1 == kNumOverlapLoads) {
                                tcgen05_before_thread_sync();
                                tmem_overlap_barrier->arrive_cluster(0);
                            }
                        }
                        if (w == kNumMWaves - 1 && s == kNumStores - 1 && issue_cnt == kNumLoads - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barrier->arrive_cluster(0);
                        }
                        ++issue_cnt;
                    }

                    // Round to BF16 first (the bitwise contract), then amax.
                    uint32_t packed[16];
                    #pragma unroll
                    for (uint32_t j = 0; j < 16; ++j)
                        packed[j] = cast_bf16_and_pack(vals[2 * j], vals[2 * j + 1]);
                    uint32_t amax = get_packed_bf16_amax(packed[0]);
                    #pragma unroll
                    for (uint32_t j = 1; j < 16; ++j)
                        amax = hmax2_bf16x2(amax, get_packed_bf16_amax(packed[j]));
                    const uint32_t sf_exp = get_ue8m0_sf_exp_e4m3(amax);
                    const uint32_t sf_inv = get_ue8m0_sf_inv_bf16(sf_exp);

                    // Scale + cast: 32 bf16 -> 32 E4M3 bytes (8 packed words).
                    uint32_t q[8];
                    #pragma unroll
                    for (uint32_t j = 0; j < 8; ++j)
                        q[j] = scale_bf16x2_into_fp8x4(packed[2 * j], packed[2 * j + 1], sf_inv);
                    // Lower 16 values (q[0..4)) -> bank group fg*2, upper -> fg*2+1.
                    st_shared_u32x4((uint32_t*)get_swizzled_smem_ptr(fg * 2), q[0], q[1], q[2], q[3]);
                    st_shared_u32x4((uint32_t*)get_swizzled_smem_ptr(fg * 2 + 1), q[4], q[5], q[6], q[7]);

                    // Store the SF byte (upstream `store_sf`): word index
                    // `sf_idx/4` in the MN-major SFD, byte lane `sf_idx%4`;
                    // batches flatten their SF columns along (batch, n).
                    // Row: this lane's row within the warp's 32-row slab of
                    // the store block (same mapping as the swizzle atom).
                    const uint32_t row_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
                    const uint32_t group_n_idx = n_idx + fg * kSFGranN;
                    if (row_idx < shape_m && group_n_idx < shape_n) {
                        const uint32_t sf_idx = (batch_idx * shape_n + group_n_idx) / kSFGranN;
                        uint32_t* sf_word_ptr = sfd + (sf_idx / 4) * sfd_stride + row_idx;
                        ((uint8_t*)sf_word_ptr)[sf_idx % 4] = (uint8_t)sf_exp;
                    }
                }
            } else {
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++i) {
                const uint32_t load_idx = reverse_store_order ? kNumLoads - 1 - i : i;
                const uint32_t tmem_addr = tmem_base_addr
                                         + w * BLOCK_N
                                         + store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerBankGroup;

                uint32_t values[kNumElemsPerBankGroup];
                if (kCdIsFloat || kNumElemsPerBankGroup == 4) {
                    tmem_load_32dp32b_x4(tmem_addr, values[0], values[1], values[2], values[3]);
                } else {
                    tmem_load_32dp32b_x8(tmem_addr, values[0], values[1], values[2], values[3],
                                         values[4], values[5], values[6], values[7]);
                }
                fence_view_async_tmem_load();

                if (kNumOverlapLoads > 0) {
                    if (w == 0 && s == 0 && i + 1 == kNumOverlapLoads) {
                        tcgen05_before_thread_sync();
                        tmem_overlap_barrier->arrive_cluster(0);
                    }
                }
                if (w == kNumMWaves - 1 && s == kNumStores - 1 && i == kNumLoads - 1) {
                    tcgen05_before_thread_sync();
                    tmem_empty_barrier->arrive_cluster(0);
                }

                if (kCdIsFloat) {
                    st_shared_u32x4((uint32_t*)get_swizzled_smem_ptr(load_idx),
                                    values[0], values[1], values[2], values[3]);
                } else {
                    st_shared_u32x4((uint32_t*)get_swizzled_smem_ptr(load_idx),
                                    cast_bf16_and_pack(values[0], values[1]),
                                    cast_bf16_and_pack(values[2], values[3]),
                                    cast_bf16_and_pack(values[4], values[5]),
                                    cast_bf16_and_pack(values[6], values[7]));
                }
            }
            }

            tma_store_fence();
            named_barrier_sync(kNumUMMAStoreThreads, 8);
            if (epilogue_warp_idx == 0 && elect_one_sync()) {
                if (kGemmType == GemmType::Batched) {
                    if (kWithAccumulation) tma_reduce_add_3d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                    else tma_store_3d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    if (kWithAccumulation) tma_reduce_add_2d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                    else tma_store_2d(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode, uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads,
          uint32_t kNumOverlappedTmemCols, GemmType kGemmType, bool kWithAccumulation,
          bool kCdIsFloat, uint32_t kCdElemSize, typename Smem>
DG_DEVICE void store_cd_swap_ab(Smem& smem, uint32_t& tma_stage_idx, uint32_t tmem_base_addr,
                                uint32_t base_m_idx, uint32_t base_n_idx, uint32_t batch_idx,
                                uint32_t epilogue_warp_idx, uint32_t lane_idx,
                                bool reverse_store_order,
                                const Barrier* tmem_overlap_barrier, const Barrier* tmem_empty_barrier,
                                const TmaMap& tensor_map_cd) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / kCdElemSize;
    constexpr uint32_t kNumTmemLoads = STORE_BLOCK_M / kNumSwizzleAtomRows;
    constexpr uint32_t kNumOverlapLoads = (kNumOverlappedTmemCols + kNumSwizzleAtomRows - 1) / kNumSwizzleAtomRows;

    const uint32_t num_stores = BLOCK_M / STORE_BLOCK_M;  // effective M is BLOCK_M in this port
    for (uint32_t s = 0; s < num_stores; ++s, tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages) {
        const uint32_t store_idx = reverse_store_order ? num_stores - 1 - s : s;
        if (epilogue_warp_idx == 0) tma_store_wait<kNumTMAStoreStages - 1>();
        named_barrier_sync(kNumUMMAStoreThreads, 8);

        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemLoads; ++i) {
            const uint32_t load_idx = reverse_store_order ? kNumTmemLoads - 1 - i : i;
            const uint32_t tmem_addr = tmem_base_addr + store_idx * STORE_BLOCK_M + load_idx * kNumSwizzleAtomRows;
            uint32_t values[kNumSwizzleAtomRows];

            if (kCdIsFloat) {
                tmem_load_32dp32b_x8(tmem_addr, values[0], values[1], values[2], values[3],
                                     values[4], values[5], values[6], values[7]);
                fence_view_async_tmem_load();
            } else {
                tmem_load_16dp256b_x1(tmem_addr, values[0], values[1], values[2], values[3]);
                tmem_load_16dp256b_x1(tmem_addr | 0x00100000u, values[4], values[5], values[6], values[7]);
                fence_view_async_tmem_load();
            }

            if (kNumOverlapLoads > 0) {
                const uint32_t num_loads_before_arrive = dg_min(kNumOverlapLoads, num_stores * kNumTmemLoads);
                if (s * kNumTmemLoads + i + 1 == num_loads_before_arrive) {
                    tcgen05_before_thread_sync();
                    tmem_overlap_barrier->arrive_cluster(0);
                }
            }
            if (s == num_stores - 1 && i == kNumTmemLoads - 1) {
                tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive_cluster(0);
            }

            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            const uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
            const uint32_t inner_atom_offset = load_idx * kNumSwizzleAtomRows * kSwizzleCDMode;
            uint8_t* smem_base_ptr = (uint8_t*)smem->cd[tma_stage_idx] + outer_atom_offset + inner_atom_offset;

            if (kCdIsFloat) {
                const uint32_t col = lane_idx / 4;
                #pragma unroll
                for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++row) {
                    uint8_t* p = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                               + ((col ^ row) * kNumBankGroupBytes) + (lane_idx % 4) * sizeof(float);
                    st_shared_u32((uint32_t*)p, values[row]);
                }
            } else {
                const uint32_t row = lane_idx % 8;
                const uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                uint8_t* smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                  + ((col ^ row) * kNumBankGroupBytes);
                stsm_x4_trans(cvta_shared_to_u32(smem_ptr),
                              cast_bf16_and_pack(values[0], values[1]),
                              cast_bf16_and_pack(values[2], values[3]),
                              cast_bf16_and_pack(values[4], values[5]),
                              cast_bf16_and_pack(values[6], values[7]));
            }
        }

        tma_store_fence();
        named_barrier_sync(kNumUMMAStoreThreads, 8);
        if (epilogue_warp_idx == 0 && elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++i) {
                const uint8_t* smem_ptr = (uint8_t*)smem->cd[tma_stage_idx] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                const uint32_t m_idx = base_m_idx + store_idx * STORE_BLOCK_M;
                const uint32_t n_idx = base_n_idx + i * STORE_BLOCK_N_ATOM;
                if (kGemmType == GemmType::Batched) {
                    if (kWithAccumulation) tma_reduce_add_3d(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                    else tma_store_3d(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                } else {
                    if (kWithAccumulation) tma_reduce_add_2d(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                    else tma_store_2d(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                }
            }
            tma_store_arrive();
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// The unified GEMM kernel
// ---------------------------------------------------------------------------
template <uint32_t kMajorA, uint32_t kMajorB,
          uint32_t kGranKA, uint32_t kGranKB, bool kHasSF, bool kIsMXF4,
          uint32_t kAFormat, uint32_t kBFormat,
          uint32_t kStorageElemA, uint32_t kStorageElemB,
          uint32_t kPackA, uint32_t kPackB,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumTMAStoreStages,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, GemmType kGemmType, bool kWithAccumulation,
          bool kCdIsFloat, uint32_t kCdElemSize,
          uint32_t kNumSMs, bool kWithOutputSF = false>
DG_GLOBAL void __launch_bounds__(256, 1)
gemm_sm100_impl(int* grouped_layout, uint32_t num_groups,
                uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                const __grid_constant__ TmaMap tensor_map_a,
                const __grid_constant__ TmaMap tensor_map_b,
                const __grid_constant__ TmaMap tensor_map_sfa,
                const __grid_constant__ TmaMap tensor_map_sfb,
                const __grid_constant__ TmaMap tensor_map_cd,
                uint32_t* sfd = nullptr, uint32_t sfd_stride = 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(DG_HOST_EDIT)
    constexpr bool kIsMGroupedContig = kGemmType == GemmType::MGroupedContiguous;
    constexpr bool kIsBatched = kGemmType == GemmType::Batched;

    // ---- MMA geometry ----
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = kIsMXF4 ? 64 : (kHasSF ? 32 : 16);
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT((kIsMXF4 && BLOCK_K == 256) || (!kIsMXF4 && (BLOCK_K == 128 || BLOCK_K == 64)),
                     "Invalid BLOCK_K");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "BLOCK_K % UMMA_K");
    DG_STATIC_ASSERT(kNumMulticast == 1 || kNumMulticast == 2, "multicast 1/2");
    DG_STATIC_ASSERT((kSwapAB && BLOCK_N == LAYOUT_AD_M) ||
                     (!kSwapAB && (BLOCK_M == 32 || BLOCK_M == 64 || BLOCK_M == LAYOUT_AD_M)),
                     "Invalid block sizes");

    // ---- SF geometry ----
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = kHasSF ? ((BLOCK_M + 127) / 128) * 128 : 1;
    constexpr uint32_t SF_BLOCK_N = kHasSF ? ((BLOCK_N + 127) / 128) * 128 : 1;
    constexpr uint32_t SF_BLOCK_K_RAW = BLOCK_K / 128;  // 0 for BF16 (BLOCK_K=64)
    constexpr uint32_t SF_BLOCK_K = kHasSF ? SF_BLOCK_K_RAW : 1;  // array sizing only
    constexpr uint32_t kNumSFArrivals = kHasSF ? 32 * SF_BLOCK_K_RAW : 0;
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;

    // ---- Epilogue geometry ----
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t STORE_BLOCK_M = kSwapAB ? 16 : dg_min(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwapAB ? BLOCK_N : kSwizzleCDMode / kCdElemSize;
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? 128u : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "store threads multiple of 32");

    // ---- TMEM budget ----
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M * SF_BLOCK_K / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N * SF_BLOCK_K / 32;
    constexpr uint32_t kNumSFTmemCols = kNumSFATmemCols + kNumSFBTmemCols;
    constexpr uint32_t kNumOverlappedTmemCols = dg_max(kNumAccumTmemCols + kNumSFTmemCols, 512u) - 512u;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFTmemCols - kNumOverlappedTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols - kNumOverlappedTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kTmemStartColOfSFA + kNumSFATmemCols;
    DG_STATIC_ASSERT(kNumOverlappedTmemCols <= UMMA_N, "overlap <= UMMA_N");
    DG_STATIC_ASSERT(!kSwapAB || kNumOverlappedTmemCols == 0 || kNumOverlappedTmemCols <= STORE_BLOCK_N,
                     "overlap fits first store");
    DG_STATIC_ASSERT(32 <= kNumTmemCols && kNumTmemCols <= 512, "tmem cols");

    // ---- Shared storage ----
    constexpr uint32_t kAStageBytes = LOAD_BLOCK_M * BLOCK_K * kStorageElemA / kPackA;
    constexpr uint32_t kBStageBytes = LOAD_BLOCK_N * BLOCK_K * kStorageElemB / kPackB;
    constexpr uint32_t kCdStageBytes = STORE_BLOCK_M * STORE_BLOCK_N * kCdElemSize;
    constexpr uint32_t kNumTMABytesPerStage = kAStageBytes + kBStageBytes;

    struct SharedStorage {
        alignas(1024) uint8_t cd[kNumTMAStoreStages][kCdStageBytes];
        alignas(1024) uint8_t a[kNumStages][kAStageBytes];
        alignas(1024) uint8_t b[kNumStages][kBStageBytes];
        alignas(1024) uint32_t sfa[kNumStages][kHasSF ? SF_BLOCK_M * SF_BLOCK_K : 1];
        alignas(1024) uint32_t sfb[kNumStages][kHasSF ? SF_BLOCK_N * SF_BLOCK_K : 1];
        Barrier full_barriers[kNumStages];
        Barrier sf_full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier tmem_overlap_barriers[kNumEpilogueStages];
        uint32_t tmem_ptr;
    };
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SharedStorage* smem = (SharedStorage*)smem_buffer;

    // ---- Setup ----
    if (kNumMulticast > 1) cluster_sync_relaxed();
    const bool is_leader_cta = get_block_rank_in_cluster() == 0;
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();

    if (warp_idx == 0) {
        prefetch_tma_map(&tensor_map_a);
        prefetch_tma_map(&tensor_map_b);
        if (kHasSF) {
            prefetch_tma_map(&tensor_map_sfa);
            prefetch_tma_map(&tensor_map_sfb);
        }
        prefetch_tma_map(&tensor_map_cd);
    }

    const uint32_t shape_sfa_k = ceil_div_u32(shape_k, kGranKA * 4);
    const uint32_t shape_sfb_k = ceil_div_u32(shape_k, kGranKB * 4);

    if (warp_idx == 1 && elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            smem->sf_full_barriers[i].init(1);
            smem->empty_barriers[i].init(1);
            smem->full_barriers[i].init(kNumMulticast * (1 + kNumSFArrivals));
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            smem->tmem_full_barriers[i].init(1);
            smem->tmem_empty_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
            smem->tmem_overlap_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
        }
        fence_barrier_init();
    } else if (warp_idx == 2) {
        if (kNumMulticast == 1) tmem_alloc_1sm(kNumTmemCols, &smem->tmem_ptr);
        else tmem_alloc_2sm(kNumTmemCols, &smem->tmem_ptr);
    }
    if (kNumMulticast > 1) cluster_sync_relaxed();
    else __syncthreads();

    griddepcontrol_wait();

    using Sched = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA, kNumSMs>;
    Sched scheduler(shape_m, shape_n, shape_k, grouped_layout);
    scheduler.kNumGroupsRuntime = num_groups;

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
            const uint32_t num_total_k_blocks = dg_max(1u, ceil_div_u32(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                smem->empty_barriers[stage_idx].wait(phase ^ 1);

                constexpr bool kAWithGroupMNOffset = kGemmType == GemmType::MGroupedMasked;
                uint32_t m_idx = scheduler.template get_global_idx<kAWithGroupMNOffset, Sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool kBWithGroupMNOffset = (kMajorB == MAJOR_K) &&
                    (kIsMGroupedContig || kGemmType == GemmType::MGroupedMasked);
                uint32_t n_idx = scheduler.template get_global_idx<kBWithGroupMNOffset, Sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                constexpr bool kAWithGroupKOffset = (kMajorA == MAJOR_MN);
                constexpr bool kBWithGroupKOffset = (kMajorB == MAJOR_MN);
                uint32_t k_a_idx = scheduler.template get_global_idx<kAWithGroupKOffset, Sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<kBWithGroupKOffset, Sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                if (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (get_block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (get_block_rank_in_cluster() * LOAD_BLOCK_N);
                }
                const uint32_t batch_idx = kIsBatched ? scheduler.current_group_idx : 0;

                // SF loads first, so the transpose overlaps the A/B transfer.
                uint32_t sf_arrival_bytes = 0;
                if (kHasSF) {
                    if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                        const uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                        const uint32_t sfa_k_idx = scheduler.template get_global_idx<!kIsMGroupedContig, Sched::IndexType::SF_K>(
                            shape_sfa_k, SF_BLOCK_K, k_block_idx / kNumSFAStagesPerLoad);
                        tma_load_2d_sf(&tensor_map_sfa, &smem->sf_full_barriers[stage_idx],
                                       smem->sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                        sf_arrival_bytes += sizeof(smem->sfa[0]);
                    }
                    if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                        const uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                        const uint32_t sfb_k_idx = scheduler.template get_global_idx<true, Sched::IndexType::SF_K>(
                            shape_sfb_k, SF_BLOCK_K, k_block_idx / kNumSFBStagesPerLoad, m_block_idx);
                        tma_load_2d_sf(&tensor_map_sfb, &smem->sf_full_barriers[stage_idx],
                                       smem->sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                        sf_arrival_bytes += sizeof(smem->sfb[0]);
                    }
                    smem->sf_full_barriers[stage_idx].arrive_and_expect_tx(sf_arrival_bytes);
                }

                if (kMajorA == MAJOR_K) {
                    tma_copy_ab<true, kIsBatched>(&tensor_map_a, &smem->full_barriers[stage_idx],
                                                  smem->a[stage_idx], kNumMulticast,
                                                  k_a_idx, m_idx, batch_idx,
                                                  BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, kStorageElemA, kPackA);
                } else {
                    tma_copy_ab<false, kIsBatched>(&tensor_map_a, &smem->full_barriers[stage_idx],
                                                   smem->a[stage_idx], kNumMulticast,
                                                   m_idx, k_a_idx, batch_idx,
                                                   LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, kStorageElemA, kPackA);
                }
                if (kMajorB == MAJOR_K) {
                    tma_copy_ab<true, kIsBatched>(&tensor_map_b, &smem->full_barriers[stage_idx],
                                                  smem->b[stage_idx], kNumMulticast,
                                                  k_b_idx, n_idx, batch_idx,
                                                  BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, kStorageElemB, kPackB);
                } else {
                    tma_copy_ab<false, kIsBatched>(&tensor_map_b, &smem->full_barriers[stage_idx],
                                                   smem->b[stage_idx], kNumMulticast,
                                                   n_idx, k_b_idx, batch_idx,
                                                   LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, kStorageElemB, kPackB);
                }

                if (is_leader_cta) {
                    smem->full_barriers[stage_idx].arrive_and_expect_tx(kNumTMABytesPerStage * kNumMulticast);
                } else {
                    smem->full_barriers[stage_idx].arrive_cluster(0);
                }
            }
        }
    } else if (warp_idx == 1 && is_leader_cta) {
        // ================= MMA issue warp =================
        InstrDescriptorBlockScaled instr_desc_bs;
        InstrDescriptor instr_desc_f16v;
        if (kHasSF) {
            instr_desc_bs = make_instr_desc_bs(kAFormat, kBFormat, UMMA_M, UMMA_N, kMajorA, kMajorB);
        } else {
            instr_desc_f16v = make_instr_desc_f16(kAFormat, kBFormat, 1 /* F32 accum */,
                                                  UMMA_M, UMMA_N, kMajorA, kMajorB);
        }
        SmemDescriptor sf_desc = make_sf_desc(nullptr);
        SmemDescriptor a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, kPackA, kStorageElemA>(smem->a[0], 0, 0);
        SmemDescriptor b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, kPackB, kStorageElemB>(smem->b[0], 0, 0);
        // Per-lane table of stage base descriptors (lane i holds stage i's lo).
        const uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * (kAStageBytes / 16) : 0u;
        const uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * (kBStageBytes / 16) : 0u;

        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t accum_stage_idx = (uint32_t)scheduler.current_iter % kNumEpilogueStages;
            const uint32_t accum_phase_idx = ((uint32_t)scheduler.current_iter / kNumEpilogueStages) & 1;
            smem->tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
            tcgen05_after_thread_sync();

            auto empty_barrier_arrive = [&](bool do_tmem_full_arrive) {
                if (kNumMulticast == 1) {
                    umma_arrive_1sm(&smem->empty_barriers[stage_idx]);
                    if (do_tmem_full_arrive) umma_arrive_1sm(&smem->tmem_full_barriers[accum_stage_idx]);
                } else {
                    constexpr uint16_t kCTAMask = (1u << kNumMulticast) - 1;
                    umma_arrive_2sm_multicast(&smem->empty_barriers[stage_idx], kCTAMask);
                    if (do_tmem_full_arrive)
                        umma_arrive_2sm_multicast(&smem->tmem_full_barriers[accum_stage_idx], kCTAMask);
                }
                __syncwarp();
            };

            const uint32_t num_total_k_blocks = dg_max(1u, ceil_div_u32(scheduler.current_shape_k, BLOCK_K));
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Broadcast the current stage's descriptor bases from the per-lane table.
                const uint32_t a_stage_lo = __shfl_sync(0xffffffffu, a_desc_lo, stage_idx);
                const uint32_t b_stage_lo = __shfl_sync(0xffffffffu, b_desc_lo, stage_idx);

                smem->full_barriers[stage_idx].wait(phase);
                tcgen05_after_thread_sync();

                const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                if (kHasSF && elect_one_sync()) {
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_K * SF_BLOCK_M / kNumUTCCPAlignedElems; ++i) {
                            replace_smem_desc_addr(sf_desc, smem->sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                            if (kNumMulticast == 1) utccp_4x32dp128bit_1cta(sf_desc.desc_, kTmemStartColOfSFA + i * 4);
                            else utccp_4x32dp128bit_2cta(sf_desc.desc_, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_K * SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {
                            replace_smem_desc_addr(sf_desc, smem->sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                            if (kNumMulticast == 1) utccp_4x32dp128bit_1cta(sf_desc.desc_, kTmemStartColOfSFB + i * 4);
                            else utccp_4x32dp128bit_2cta(sf_desc.desc_, kTmemStartColOfSFB + i * 4);
                        }
                    }
                }
                __syncwarp();

                if (kNumOverlappedTmemCols > 0) {
                    if (k_block_idx == 0 && scheduler.current_iter > 0) {
                        tcgen05_before_thread_sync();
                        const uint32_t preceding_iter_idx = (uint32_t)scheduler.current_iter - 1;
                        const uint32_t preceding_stage_idx = preceding_iter_idx % kNumEpilogueStages;
                        const uint32_t preceding_phase_idx = (preceding_iter_idx / kNumEpilogueStages) & 1;
                        smem->tmem_overlap_barriers[preceding_stage_idx].wait(preceding_phase_idx);
                        tcgen05_after_thread_sync();
                    }
                }

                if (elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++umma_k_idx) {
                        const uint32_t offset = umma_k_idx * UMMA_K;
                        const uint32_t subblock_idx = offset / kNumUTCCPAlignedElems;
                        const uint32_t sf_id_in_subblock = (offset % kNumUTCCPAlignedElems) / 32;
                        const uint32_t tmem_col_sfa = kTmemStartColOfSFA + subblock_idx * SF_BLOCK_M / 32;
                        const uint32_t tmem_col_sfb = kTmemStartColOfSFB + subblock_idx * SF_BLOCK_N / 32;
                        const uint32_t sfa_id = (kGranKA == 32) ? sf_id_in_subblock : sfa_stage_in_group_idx;
                        const uint32_t sfb_id = (kGranKB == 32) ? sf_id_in_subblock : sfb_stage_in_group_idx;

                        uint64_t runtime_instr_desc;
                        if (kHasSF) {
                            runtime_instr_desc = kSwapAB
                                ? make_runtime_instr_desc_bs(instr_desc_bs, sfb_id, sfa_id)
                                : make_runtime_instr_desc_bs(instr_desc_bs, sfa_id, sfb_id);
                        } else {
                            runtime_instr_desc = make_runtime_instr_desc(instr_desc_f16v);
                        }

                        a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, kPackA, kStorageElemA>(a_stage_lo, 0, offset);
                        b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, kPackB, kStorageElemB>(b_stage_lo, 0, offset);
                        const uint32_t accumulate = (umma_k_idx > 0 || k_block_idx > 0) ? 1u : 0u;
                        const uint32_t accum_col = accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols);
                        if (kHasSF) {
                            if (kSwapAB) {
                                if (kIsMXF4) {
                                    if (kNumMulticast == 1) mma_mxf4_1sm(b_desc.desc_, a_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfb, tmem_col_sfa);
                                    else mma_mxf4_2sm(b_desc.desc_, a_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfb, tmem_col_sfa);
                                } else {
                                    if (kNumMulticast == 1) mma_mxf8f6f4_1sm(b_desc.desc_, a_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfb, tmem_col_sfa);
                                    else mma_mxf8f6f4_2sm(b_desc.desc_, a_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfb, tmem_col_sfa);
                                }
                            } else {
                                if (kIsMXF4) {
                                    if (kNumMulticast == 1) mma_mxf4_1sm(a_desc.desc_, b_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfa, tmem_col_sfb);
                                    else mma_mxf4_2sm(a_desc.desc_, b_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfa, tmem_col_sfb);
                                } else {
                                    if (kNumMulticast == 1) mma_mxf8f6f4_1sm(a_desc.desc_, b_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfa, tmem_col_sfb);
                                    else mma_mxf8f6f4_2sm(a_desc.desc_, b_desc.desc_, accum_col, accumulate, runtime_instr_desc, tmem_col_sfa, tmem_col_sfb);
                                }
                            }
                        } else {
                            if (kNumMulticast == 1) mma_f16_1sm(a_desc.desc_, b_desc.desc_, accum_col, accumulate, runtime_instr_desc);
                            else mma_f16_2sm(a_desc.desc_, b_desc.desc_, accum_col, accumulate, runtime_instr_desc);
                        }
                    }
                }
                __syncwarp();
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        if (kNumMulticast > 1 && scheduler.current_iter > 0) {
            const uint32_t iter_idx = (uint32_t)scheduler.current_iter - 1;
            const uint32_t accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            smem->tmem_empty_barriers[iter_idx % kNumEpilogueStages].wait(accum_phase_idx);
        }
    } else if (kHasSF && (warp_idx == 2 || (SF_BLOCK_K == 2 && warp_idx == 3))) {
        // ================= SF UTCCP transposer =================
        const uint32_t sf_k_subblock_idx = warp_idx - 2;
        auto utccp_required_smem_warp_transpose = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                values[i] = ld_shared_u32(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            st_shared_u32x4(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t num_total_k_blocks = dg_max(1u, ceil_div_u32(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                smem->sf_full_barriers[stage_idx].wait(phase);
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i)
                        utccp_required_smem_warp_transpose(smem->sfa[stage_idx] + sf_k_subblock_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i)
                        utccp_required_smem_warp_transpose(smem->sfb[stage_idx] + sf_k_subblock_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                }
                fence_view_async_shared();
                smem->full_barriers[stage_idx].arrive_cluster(0);
            }
        }
    } else if (warp_idx >= 4 && warp_idx < 4 + kNumUMMAStoreThreads / 32) {
        // ================= Epilogue =================
        const uint32_t epilogue_warp_idx = warp_idx - 4;
        uint32_t tma_stage_idx = 0;

        uint32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t accum_stage_idx = (uint32_t)scheduler.current_iter % kNumEpilogueStages;
            const uint32_t accum_phase_idx = ((uint32_t)scheduler.current_iter / kNumEpilogueStages) & 1;

            smem->tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
            tcgen05_after_thread_sync();

            const uint32_t tmem_base_addr = accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols);
            const bool reverse_store_order = kNumOverlappedTmemCols > 0 && accum_stage_idx == 0;
            constexpr bool kCDWithGroupOffset = !kIsMGroupedContig;
            const uint32_t base_m_idx = scheduler.template get_global_idx<kCDWithGroupOffset, Sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;

            if (kSwapAB) {
                DG_STATIC_ASSERT(!kWithOutputSF || !kSwapAB,
                                 "QuantizeToFP8 is only wired for the non-swapped epilogue");
                store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                                 kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                                 kNumOverlappedTmemCols, kGemmType, kWithAccumulation,
                                 kCdIsFloat, kCdElemSize>(
                    smem, tma_stage_idx, tmem_base_addr, base_m_idx, base_n_idx,
                    scheduler.current_group_idx, epilogue_warp_idx, lane_idx,
                    reverse_store_order,
                    &smem->tmem_overlap_barriers[accum_stage_idx],
                    &smem->tmem_empty_barriers[accum_stage_idx],
                    tensor_map_cd);
            } else {
                store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                         kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                         kNumOverlappedTmemCols, kGemmType, kWithAccumulation,
                         kCdIsFloat, kCdElemSize, kWithOutputSF>(
                    smem, tma_stage_idx, tmem_base_addr, base_m_idx, base_n_idx,
                    scheduler.current_group_idx, epilogue_warp_idx, lane_idx,
                    reverse_store_order,
                    &smem->tmem_overlap_barriers[accum_stage_idx],
                    &smem->tmem_empty_barriers[accum_stage_idx],
                    tensor_map_cd,
                    sfd, sfd_stride, shape_m, shape_n);
            }
        }
    }

    if (kNumMulticast > 1) cluster_sync_relaxed();
    else __syncthreads();

    if (warp_idx == 0) {
        if (kNumMulticast == 1) { tmem_relinquish_1sm(); tmem_dealloc_1sm(0, kNumTmemCols); }
        else { tmem_relinquish_2sm(); tmem_dealloc_2sm(0, kNumTmemCols); }
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("trap;");
    }
#endif
}

} // namespace dg
