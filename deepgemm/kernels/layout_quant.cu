// DeepGEMM-RS: scale-factor transforms, activation quantization, and dequant.
// Port of upstream `impls/smxx_layout.cuh` (transpose_and_pack_fp32_into_ue8m0)
// plus the per-token-group MX quantization needed to prepare GEMM inputs.
//
// ===========================================================================
// SF PACKING — the layout every consumer (TMA, UTCCP, golden model) agrees on
// ===========================================================================
// transform_sf turns row-major f32 scales [mn, sf_k] into the "MN-major"
// packed form the tensor core SF path consumes:
//
//   input  (logical):  sf[m][k]        m = 0..mn-1 (rows), k = 0..sf_k-1
//   output (packed) :  word[k/4][m] as u32[ceil(sf_k/4)][align(mn, 4)]:
//
//        word(k, m) = UE8M0(k+0) | UE8M0(k+1) << 8 | UE8M0(k+2) << 16
//                    | UE8M0(k+3) << 24          with UE8M0(x) = bits 23..30
//                    of the f32 scale (an exact power of two), so the byte is
//                    the biased exponent.
//
//   Why MN-contiguous? A TMA row is 16B = 4 int32 = 4 consecutive m rows of
//   the SAME k group — exactly the granularity the mma SF descriptor wants,
//   and it makes `align(mn, 4)` the row padding. The quant kernel merges its
//   bytes into the same words with one atomicOr per group, so both producers
//   (transform_sf and quant_mx) emit bit-identical buffers — asserted by the
//   e2e test `transform_sf_bit_exact_vs_golden`.
//
// quant_mx (activations): per group of `gran` elements (32 or 128),
//   amax -> UE8M0 exponent via the upstream bit-trick:
//     rounded_exp = (bits(amax) + 0x7FFFFF - kQuantMaxMantissa) >> 23
//   where the +0x7FFFFF rounds amax UP to the next power of two when its
//   fraction exceeds 0.5 (so amax=7 -> 8 -> scale 2; amax=500 -> 512).
//   Then each element is scaled by 2^-exp and RNE'd onto the E4M3 (or E2M1)
//   grid. Floors: fp8 105 (amax 1e-4), fp4 1 (max(amax, 6*2^-126)).
// ===========================================================================

namespace dg {

// ---------------------------------------------------------------------------
// transform_sf: fp32 SFs [mn, sf_k] row-major -> packed UE8M0 int32
//               [packed_sf_k, tma_aligned_mn] (MN-contiguous).
// Port of `transpose_and_pack_fp32_into_ue8m0` (single group, no psum layout).
// Input SFs must be exact powers of two (only the exponent is packed).
// ---------------------------------------------------------------------------
template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K, uint32_t BLOCK_SF_K>
DG_GLOBAL __launch_bounds__(kNumThreads, 4)
void transform_sf_impl(const float* sf, uint32_t* out, const uint32_t mn) {
    extern __shared__ uint32_t smem_buffer[];
    constexpr uint32_t kNumPackedSFK = (SF_K + 3) / 4;
    constexpr uint32_t kNumTMAAlignedElems = 16 / 4;  // int32
    constexpr uint32_t kSmemSFK = SF_K < BLOCK_SF_K ? SF_K : BLOCK_SF_K + 1;

    const uint32_t num_mn_blocks = ceil_div_u32(mn, BLOCK_MN);
    const uint32_t block_mn_idx = SF_K < BLOCK_SF_K ? blockIdx.x : blockIdx.x % num_mn_blocks;
    const uint32_t block_sf_k_idx = SF_K < BLOCK_SF_K ? 0u : blockIdx.x / num_mn_blocks;
    const uint32_t block_sf_k_start = block_sf_k_idx * BLOCK_SF_K;
    const uint32_t in_block_mn = dg_min(BLOCK_MN, mn - block_mn_idx * BLOCK_MN);
    const uint32_t in_block_sf_k = SF_K < BLOCK_SF_K ? SF_K : dg_min(BLOCK_SF_K, SF_K - block_sf_k_start);
    const uint32_t in_block_packed_sf_k = ceil_div_u32(in_block_sf_k, 4u);
    const uint32_t tma_aligned_mn = align_u32(mn, kNumTMAAlignedElems);

    out += (uint64_t)blockIdx.y * tma_aligned_mn * kNumPackedSFK;
    const float* local_sf = sf + (uint64_t)block_mn_idx * BLOCK_MN * SF_K + block_sf_k_start;

    griddepcontrol_wait();

    // Stage FP32 SFs through shared memory (row-major: [mn, sf_k]).
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = get_lane_idx();
    if (SF_K < BLOCK_SF_K) {
        const uint32_t num_values = in_block_mn * SF_K;
        for (uint32_t i = threadIdx.x; i < num_values / 4; i += kNumThreads) {
            const float4 v = ((const float4*)local_sf)[i];
            const uint4 u = *(const uint4*)&v;
            st_shared_u128(smem_buffer + i * 4, u);
        }
        for (uint32_t i = num_values / 4 * 4 + threadIdx.x; i < num_values; i += kNumThreads) {
            uint32_t b = __float_as_uint(local_sf[i]);
            st_shared_u32(smem_buffer + i, b);
        }
    } else {
        for (uint32_t row = warp_idx; row < in_block_mn; row += kNumThreads / 32) {
            for (uint32_t col = lane_idx * 4; col < in_block_sf_k; col += 32 * 4) {
                const float4 v = *(const float4*)(local_sf + (uint64_t)row * SF_K + col);
                const uint4 u = *(const uint4*)&v;
                st_shared_u32(smem_buffer + row * kSmemSFK + col + 0, u.x);
                st_shared_u32(smem_buffer + row * kSmemSFK + col + 1, u.y);
                st_shared_u32(smem_buffer + row * kSmemSFK + col + 2, u.z);
                st_shared_u32(smem_buffer + row * kSmemSFK + col + 3, u.w);
            }
        }
    }
    __syncthreads();

    // Pack 4 UE8M0 bytes per int32 and scatter into the MN-major output.
    for (uint32_t i = threadIdx.x; i < in_block_packed_sf_k * BLOCK_MN; i += kNumThreads) {
        const uint32_t sf_k_pack_idx = i / BLOCK_MN;
        const uint32_t mn_idx = i % BLOCK_MN;
        const uint32_t global_mn_idx = block_mn_idx * BLOCK_MN + mn_idx;
        const uint32_t global_sf_k_pack_idx = block_sf_k_start / 4 + sf_k_pack_idx;

        uint32_t values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t sf_k_idx = sf_k_pack_idx * 4 + j;
            const uint32_t global_sf_k_idx = block_sf_k_start + sf_k_idx;
            values[j] = (global_mn_idx < mn && global_sf_k_idx < SF_K)
                ? ld_shared_u32(smem_buffer + mn_idx * kSmemSFK + sf_k_idx) : 0;
        }

        uint32_t packed = 0;
        packed |= (values[0] >> 23u);
        packed |= (values[1] >> 15u);
        packed |= (values[2] >> 7u);
        packed |= (values[3] << 1u);
        if (global_mn_idx < mn)
            out[(uint64_t)global_sf_k_pack_idx * tma_aligned_mn + global_mn_idx] = packed;
    }
}

// ---------------------------------------------------------------------------
// UE8M0 scale-factor exponent (port of math::get_ue8m0_sf_exp).
//   fp8 (e4m3): quant max = 448 = 1.75 * 2^8, min amax floor 1e-4.
//   fp4 (e2m1): quant max = 6 = 1.5 * 2^2.
// ---------------------------------------------------------------------------
DG_DEVICE uint32_t get_ue8m0_sf_exp_fp8(float amax) {
    constexpr uint32_t kMantissaBits = 23;
    constexpr uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint32_t kQuantMaxMantissa = 0x60u << (kMantissaBits - 7);
    constexpr uint32_t kQuantMaxExponent = 8;
    constexpr uint32_t kMinSFExponent = 105;
    const uint32_t amax_bits = __float_as_uint(amax);
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    return dg_max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}
DG_DEVICE uint32_t get_ue8m0_sf_exp_fp4(float amax) {
    constexpr uint32_t kMantissaBits = 23;
    constexpr uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint32_t kQuantMaxMantissa = 0x40u << (kMantissaBits - 7);
    constexpr uint32_t kQuantMaxExponent = 2;
    constexpr uint32_t kMinSFExponent = 1;
    const uint32_t amax_bits = __float_as_uint(amax);
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    return dg_max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}
// Reciprocal scale as f32 (2^-exp): biased exponents sum to 254.
DG_DEVICE float ue8m0_sf_inv(uint32_t sf_exp) {
    return __uint_as_float((254u - sf_exp) << 23);
}

// ---------------------------------------------------------------------------
// Activation quantization: f32 [m, k] row-major ->
//   e4m3 data [m, k] + packed UE8M0 SF [ceil(k/gran/4), tma_aligned_m] int32
//   e2m1 data [m, k/2 bytes] + SF (MXFP4, gran 32).
// One block per row; warps stride over the row's groups. `out_sf` must be
// zero-initialized (bytes are merged with atomics).
// Packed SF word (row r, m): byte b = UE8M0 exponent of group r*4+b.
// ---------------------------------------------------------------------------
template <uint32_t kNumThreads, uint32_t kGran, bool kIsFp4>
DG_GLOBAL __launch_bounds__(kNumThreads, 2)
void quant_mx_impl(const float* x, uint32_t m, uint32_t k,
                   uint8_t* out_data, uint32_t* out_sf) {
    const uint32_t row = blockIdx.x;
    if (row >= m) return;
    griddepcontrol_wait();

    const uint32_t lane_idx = get_lane_idx();
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t num_warps = kNumThreads / 32;
    const uint32_t num_groups = k / kGran;
    const uint32_t tma_aligned_m = align_u32(m, 4);

    for (uint32_t g = warp_idx; g < num_groups; g += num_warps) {
        const uint32_t group_base = g * kGran;
        // Warp max-reduce over the group.
        float amax = 0.0f;
        for (uint32_t i = lane_idx; i < kGran; i += 32) {
            const float v = x[(uint64_t)row * k + group_base + i];
            amax = fmaxf(amax, fabsf(v));
        }
        #pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        amax = fmaxf(amax, 1e-30f);

        const uint32_t sf_exp = kIsFp4 ? get_ue8m0_sf_exp_fp4(amax) : get_ue8m0_sf_exp_fp8(amax);
        const float inv_sf = ue8m0_sf_inv(sf_exp);

        // Quantize pairs of elements (even lanes handle 2 elements).
        if (kIsFp4) {
            for (uint32_t i = lane_idx * 2; i < kGran; i += 64) {
                const float v0 = x[(uint64_t)row * k + group_base + i] * inv_sf;
                const float v1 = (i + 1 < kGran)
                    ? x[(uint64_t)row * k + group_base + i + 1] * inv_sf : 0.0f;
                const uint32_t byte = cvt_e2m1x2_f32(v0, v1);
                if (out_data != nullptr)
                    out_data[(uint64_t)row * (k / 2) + (group_base + i) / 2] = (uint8_t)byte;
            }
        } else {
            for (uint32_t i = lane_idx * 2; i < kGran; i += 64) {
                const float v0 = x[(uint64_t)row * k + group_base + i] * inv_sf;
                const float v1 = (i + 1 < kGran)
                    ? x[(uint64_t)row * k + group_base + i + 1] * inv_sf : 0.0f;
                const uint32_t pair = cvt_e4m3x2_f32(v0, v1);
                if (out_data != nullptr) {
                    out_data[(uint64_t)row * k + group_base + i] = (uint8_t)(pair & 0xffu);
                    out_data[(uint64_t)row * k + group_base + i + 1] = (uint8_t)(pair >> 8);
                }
            }
        }

        // Merge the SF byte into the packed word (uniform value; one lane fires).
        if (out_sf != nullptr && lane_idx == (g % 4)) {
            const uint32_t sf_row = g / 4;
            const uint32_t byte_off = g % 4;
            atomicOr(&out_sf[(uint64_t)sf_row * tma_aligned_m + row],
                     (sf_exp & 0xffu) << (8 * byte_off));
        }
    }
}

// ---------------------------------------------------------------------------
// Reference dequant: e4m3/e2m1 data + packed UE8M0 SF -> f32.
// Used by e2e tests (K-major data, SF [packed_k, tma_aligned_mn]).
// ---------------------------------------------------------------------------
template <bool kIsFp4, uint32_t kGran>
DG_GLOBAL __launch_bounds__(256)
void dequant_mx_impl(const uint8_t* data, const uint32_t* sf,
                     uint32_t m, uint32_t k, uint32_t tma_aligned_m, float* out) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)m * k;
    if (idx >= total) return;
    griddepcontrol_wait();

    const uint32_t row = (uint32_t)(idx / k);
    const uint32_t col = (uint32_t)(idx % k);
    const uint32_t sf_slot = col / kGran;
    const uint32_t packed_row = sf_slot / 4;
    const uint32_t byte_in_word = sf_slot % 4;
    const uint32_t word = ld_global_u32(sf + (uint64_t)packed_row * tma_aligned_m + row);
    const uint32_t sf_exp = (word >> (8 * byte_in_word)) & 0xffu;
    const float scale = exp2f((float)sf_exp - 127.0f);

    if (kIsFp4) {
        const uint32_t byte = data[(uint64_t)row * (k / 2) + col / 2];
        const float v = f32_from_e2m1((col & 1) ? ((byte >> 4) & 0xfu) : (byte & 0xfu));
        out[idx] = v * scale;
    } else {
        const uint32_t byte = data[(uint64_t)row * k + col];
        uint32_t lo, hi;
        cvt_f32x2_e4m3x2(byte, lo, hi);
        out[idx] = __uint_as_float(lo) * scale;
    }
}

} // namespace dg
