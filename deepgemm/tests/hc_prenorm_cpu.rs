//! Integration tests for the TF32 hyperconnection pre-norm GEMM
//! (`hc_prenorm.cu` + `api_hc_prenorm.rs`) — all CPU-only, no GPU needed.
//!
//! Two planes of validation (the repo's "sandbox" model):
//!  1. A pure-Rust reference of the pre-norm scaling math the kernels
//!     implement: the TF32 operand semantics (bf16 A exact, fp32 B
//!     mantissa-truncated), the split-K partial decomposition, the squared
//!     row-sum statistic, and the post-GEMM normalization identity that
//!     motivates the fusion — plus structural checks of the ported
//!     SMEM-swizzle formulas.
//!  2. Offline NVRTC compile checks (PTX + SASS) of both kernel variants
//!     (`90a` and `100a`), the same compilation the runtime JIT performs.

use deepgemm::jit::{self, kernel_src};

// ---------------------------------------------------------------------------
// Small numeric helpers (bf16 / tf32)
// ---------------------------------------------------------------------------

/// bf16 bits -> f32 (exact).
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> bf16 value (round-to-nearest-even, via the golden encoder).
fn to_bf16(v: f32) -> f32 {
    bf16_bits_to_f32(deepgemm::golden::f32_to_bf16_bits(v))
}

/// The hardware's TF32 interpretation of an *unconverted* fp32 operand:
/// keep the sign, the 8-bit exponent and the top 10 mantissa bits — the low
/// 13 bits are what `wgmma...tf32.tf32` / `tcgen05.mma.kind::tf32` silently
/// drop when fed raw fp32 (B in this kernel family).
fn tf32_trunc(v: f32) -> f32 {
    f32::from_bits(f32::to_bits(v) & 0xffff_e000)
}

/// Deterministic LCG (same generator family as the golden tests).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next_u32() as f32 / u32::MAX as f32) * (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// Reference model of the kernel's math (formula extraction)
// ---------------------------------------------------------------------------

/// Exact port of the kernel's split-K decomposition:
/// `k_offset(split) = (split * per + min(split, remain)) * BLOCK_K`,
/// `stages(split) = per + (split < remain)`.
fn split_k_ranges(k: usize, block_k: usize, num_splits: usize) -> Vec<(usize, usize)> {
    let num_k_blocks = k.div_ceil(block_k);
    let per = num_k_blocks / num_splits;
    let remain = num_k_blocks % num_splits;
    (0..num_splits)
        .map(|s| {
            let k_offset = (s * per + s.min(remain)) * block_k;
            let stages = per + usize::from(s < remain);
            (k_offset, k_offset + stages * block_k)
        })
        .collect()
}

/// CPU reference of the kernel outputs (per split): TF32-truncated B,
/// exact bf16 A, fp32 accumulation (sequential order — the GPU's wgmma /
/// tcgen05 tree order differs by O(eps) only).
fn hc_prenorm_reference(
    a: &[f32], // exact bf16 values, [m, k] row-major
    b: &[f32], // fp32 weights, [n, k] row-major
    m: usize,
    n: usize,
    k: usize,
    num_splits: usize,
) -> (Vec<f32>, Vec<f32>) {
    let block_k = 64;
    let ranges = split_k_ranges(k, block_k, num_splits);
    let mut d = vec![0f32; num_splits * m * n];
    let mut s = vec![0f32; num_splits * m];
    for (split, &(k_lo, k_hi)) in ranges.iter().enumerate() {
        for mi in 0..m {
            // sqr_sum: exact squares of the exact (bf16) A values.
            let mut ss = 0f32;
            for kk in k_lo..k_hi {
                let v = a[mi * k + kk];
                ss += v * v;
            }
            s[split * m + mi] = ss;
            // D: TF32 x TF32 products, fp32 accumulation.
            for ni in 0..n {
                let mut acc = 0f32;
                for kk in k_lo..k_hi {
                    acc += a[mi * k + kk] * tf32_trunc(b[ni * k + kk]);
                }
                d[(split * m + mi) * n + ni] = acc;
            }
        }
    }
    (d, s)
}

fn rel_err(got: f32, want: f64) -> f64 {
    (got as f64 - want).abs() / want.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// 1. The pre-norm math
// ---------------------------------------------------------------------------

#[test]
fn hc_prenorm_prenorm_math_reference() {
    // Upstream test shape family: small M, tiny N (hyperconnection width),
    // large K.  m = 13 also exercises the partial last M block (TMA OOB
    // zero-fill of rows 13..64 — rows >= m never contribute).  k = 7680
    // makes the split-K uneven (120 blocks / 16 splits: first 8 splits get
    // one extra block).
    let (m, n) = (13usize, 24);
    for &(k, num_splits) in &[(7168usize, 1usize), (7168, 16), (7680, 16)] {
        let mut rng = Lcg(0x5eed_1234);
        let a: Vec<f32> = (0..m * k)
            .map(|_| to_bf16(rng.next_f32(-1.5, 1.5)))
            .collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.next_f32(-0.05, 0.05)).collect();
        let (d, s) = hc_prenorm_reference(&a, &b, m, n, k, num_splits);

        // (a) split-K partials sum to the full-K computation (fp32 order
        // differs, hence a small relative tolerance).
        let (d_full, s_full) = hc_prenorm_reference(&a, &b, m, n, k, 1);
        if num_splits == 1 {
            assert_eq!(d, d_full);
            assert_eq!(s, s_full);
        } else {
            for mi in 0..m {
                for ni in 0..n {
                    let total: f32 = (0..num_splits).map(|sp| d[(sp * m + mi) * n + ni]).sum();
                    assert!(
                        rel_err(total, d_full[mi * n + ni] as f64) < 1e-3,
                        "D split sum mismatch at ({mi},{ni}) splits={num_splits}"
                    );
                }
            }
            for mi in 0..m {
                let total: f32 = (0..num_splits).map(|sp| s[sp * m + mi]).sum();
                assert!(
                    rel_err(total, s_full[mi] as f64) < 1e-3,
                    "sqr_sum split sum mismatch at {mi} splits={num_splits}"
                );
            }
        }

        // (b) the pre-norm statistic is exact: sqr_sum == sum_k a^2 (f64
        // ground truth; fp32 accumulation over K ~ 7k terms costs up to
        // ~K*eps ~ 1e-3 relative — a wrong formula is off by O(1)).
        for mi in 0..m {
            let exact: f64 = (0..k).map(|kk| (a[mi * k + kk] as f64).powi(2)).sum();
            let total: f32 = (0..num_splits).map(|sp| s[sp * m + mi]).sum();
            assert!(
                rel_err(total, exact) < 1e-3,
                "sqr_sum vs exact mismatch at {mi}"
            );
        }

        // (c) THE pre-norm identity the fusion relies on: normalizing after
        // the GEMM equals normalizing before it:
        //     (sum_k a*tf32(b)) / sqrt(sum_k a^2)
        //   = sum_k (a / sqrt(sum a^2)) * tf32(b)
        let (d_ref, s_ref) = hc_prenorm_reference(&a, &b, m, n, k, 1);
        for mi in 0..m {
            let rms = (s_ref[mi] as f64).sqrt();
            for ni in 0..n {
                // normalized-then-projected reference (f64 accumulation).
                let mut y = 0f64;
                for kk in 0..k {
                    y += (a[mi * k + kk] as f64 / rms) * tf32_trunc(b[ni * k + kk]) as f64;
                }
                let got = d_ref[mi * n + ni] as f64 / rms;
                assert!(
                    (got - y).abs() / y.abs().max(1.0) < 1e-3,
                    "pre-norm identity broken at ({mi},{ni})"
                );
            }
        }

        // (d) A's bf16 values are exactly representable in TF32 — the A
        // operand suffers NO rounding on the tensor core.
        for &v in &a {
            assert_eq!(tf32_trunc(v), v, "bf16 value {v} not exact in tf32");
        }
    }
}

#[test]
fn hc_prenorm_split_k_partition_covers_k_exactly() {
    // The kernel's split decomposition must tile [0, K) disjointly for every
    // (K, splits) pair — including uneven splits and splits > K blocks.
    let block_k = 64;
    for &(k, splits) in &[
        (7168usize, 1usize),
        (7168, 16),
        (7680, 16),
        (256, 4),
        (128, 4), // per = 0, remain = 2: only splits 0/1 work, rest empty
        (64, 8),  // one block, split 0 only
    ] {
        let ranges = split_k_ranges(k, block_k, splits);
        let mut covered = vec![false; k];
        for &(lo, hi) in &ranges {
            assert!(lo <= hi && hi <= k, "range out of bounds");
            for c in &mut covered[lo..hi] {
                assert!(!*c, "overlapping split at k={k} splits={splits}");
                *c = true;
            }
        }
        assert_eq!(
            covered.iter().filter(|c| **c).count(),
            k.div_ceil(block_k) * block_k,
            "split coverage mismatch at k={k} splits={splits}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Ported swizzle formulas are permutations (no bank-group collisions)
// ---------------------------------------------------------------------------

/// Rust mirror of the SM90 epilogue's `get_swizzled_bank_group_idx`.
fn sm90_bank_group(swizzle: u32, offset: u32, lane: u32) -> u32 {
    let groups_in_range = swizzle / 16;
    let bank_group_idx = offset + lane * groups_in_range;
    let num_bank_groups = 128 / 16;
    let has_shortcut = groups_in_range == num_bank_groups;
    let row = if has_shortcut {
        offset / num_bank_groups + lane
    } else {
        bank_group_idx / num_bank_groups
    };
    let mut col = if has_shortcut {
        offset
    } else {
        bank_group_idx % num_bank_groups
    };
    col ^= row % groups_in_range;
    (row * num_bank_groups + col) % groups_in_range
}

/// Rust mirror of the SM100 `get_swizzled_smem_offset` (byte offset).
fn sm100_smem_offset(swizzle: u32, offset: u32, lane: u32) -> u32 {
    let bank_group_idx = offset + lane * (swizzle / 16);
    let num_bank_groups = 128 / 16;
    let has_shortcut = (swizzle / 16) == num_bank_groups;
    let row = if has_shortcut {
        offset / num_bank_groups + lane
    } else {
        bank_group_idx / num_bank_groups
    };
    let mut col = if has_shortcut {
        offset
    } else {
        bank_group_idx % num_bank_groups
    };
    col ^= row % (swizzle / 16);
    row * 128 + col * 16
}

#[test]
fn sm90_epilogue_swizzle_is_bijective_per_row() {
    // For a fixed lane (fixed staging row), the logical bank groups of the
    // row must map to distinct physical groups: XOR with a constant key.
    for &swizzle in &[64u32, 128] {
        let groups = swizzle / 16;
        for lane in 0..16u32 {
            let mut seen = std::collections::HashSet::new();
            for offset in 0..groups {
                assert!(
                    seen.insert(sm90_bank_group(swizzle, offset, lane)),
                    "collision: swizzle={swizzle} lane={lane}"
                );
            }
        }
    }
}

#[test]
fn sm100_smem_swizzle_is_bijective_per_warp_tile() {
    // Over one warp's tile, (offset, lane) must map to distinct 16B slots:
    // swizzle 128 -> 16 rows x 8 groups; swizzle 64 -> 16 atom rows x 8
    // groups (= 32 staging rows of 4 groups).
    for &swizzle in &[64u32, 128] {
        let offsets = swizzle / 16; // 16B groups per logical row
        let mut seen = std::collections::HashSet::new();
        for lane in 0..16u32 {
            for offset in 0..offsets {
                assert!(
                    seen.insert(sm100_smem_offset(swizzle, offset, lane)),
                    "collision: swizzle={swizzle} lane={lane} offset={offset}"
                );
            }
        }
        assert_eq!(seen.len(), (offsets * 16) as usize);
    }
}

// ---------------------------------------------------------------------------
// 3. Offline NVRTC compile checks (PTX + SASS) — both variants
// ---------------------------------------------------------------------------

#[test]
fn compile_check_hc_prenorm_sm90() {
    let tu = format!("{}\n{}", kernel_src::WGMMA_H, kernel_src::HC_PRENORM);
    let body = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, float* sqr_sum) {
    dg::hc_prenorm_sm90_impl<24, 7168, 64, 32, 64, 16, 128, 12, 128, 128>
        (shape_m, tma_a, tma_b, tma_d, sqr_sum);
}"#;
    let r = jit::compile_check_kernel(&tu, body, "90a", "hc-prenorm")
        .expect("SM90 hc-prenorm must compile (PTX)");
    assert!(
        r.cubin_len.is_some(),
        "SM90 hc-prenorm SASS (CUBIN) generation failed"
    );
}

#[test]
fn compile_check_hc_prenorm_sm100() {
    let tu = kernel_src::HC_PRENORM.to_string();
    let body = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, float* sqr_sum) {
    dg::hc_prenorm_sm100_impl<24, 7168, 64, 32, 64, 16, 128, 12, 128, 128>
        (shape_m, tma_a, tma_b, tma_d, sqr_sum);
}"#;
    let r = jit::compile_check_kernel(&tu, body, "100a", "hc-prenorm")
        .expect("SM100 hc-prenorm must compile (PTX)");
    assert!(
        r.cubin_len.is_some(),
        "SM100 hc-prenorm SASS (CUBIN) generation failed"
    );
}
