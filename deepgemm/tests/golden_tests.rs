//! Pure-CPU golden-model tests — these run **without any GPU** (sandbox, CI,
//! laptop) and exercise the full numeric path of the library in software:
//! quantization, scale-factor transforms, block-scaled GEMM and MQA logits.
//!
//! On a B200, the same golden model is additionally cross-checked against the
//! real `tcgen05` kernels in `tests/e2e_gpu.rs` (feature `e2e`).

use deepgemm::golden::*;
use deepgemm::types::{Dtype, SfGran};

/// Deterministic LCG — reproducible everywhere, no external RNG.
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
// Format encoders/decoders
// ---------------------------------------------------------------------------

#[test]
fn e4m3_roundtrip_covers_full_grid() {
    // Every non-NaN code decodes to a finite value, and quant(decode(c)) == c
    // for all codes (the grid is closed under RNE).
    for code in 0u8..=0xFF {
        if code & 0x7f == 0x7f {
            continue; // NaN encodings
        }
        let v = e4m3_decode(code);
        assert!(v.is_finite(), "code {code:#04x} not finite");
        let back = quant_e4m3(v);
        assert_eq!(back, code, "code {code:#04x} not closed under RNE (v={v})");
    }
}

#[test]
fn e2m1_roundtrip_covers_full_grid() {
    for code in 0u8..=0xF {
        let v = e2m1_decode(code);
        assert_eq!(quant_e2m1(v), code, "nibble {code:#04x} not closed (v={v})");
    }
    // Exact grid values:
    assert_eq!(e2m1_decode(0x6), 4.0);
    assert_eq!(e2m1_decode(0x7), 6.0);
    assert_eq!(e2m1_decode(0x8), -0.0);
    assert_eq!(e2m1_decode(0x9), -0.5);
    // RNE ties: 3.5 is exactly between 3 (code 5, odd) and 4 (code 6, even).
    assert_eq!(quant_e2m1(3.5), 0x6, "tie 3.5 must go to even code 4");
    assert_eq!(quant_e2m1(2.5), 0x4, "tie 2.5 must go to even code 2");
    assert_eq!(quant_e2m1(0.25), 0x0, "tie 0.25 must go to even code 0");
    // Saturation:
    assert_eq!(quant_e2m1(100.0), 0x7, "satfinite clamps to +6");
    assert_eq!(quant_e2m1(-100.0), 0xF, "satfinite clamps to -6");
}

#[test]
fn ue8m0_matches_transform_packing_bits() {
    // The transform packs `bits >> 23` — i.e. the IEEE exponent byte of a
    // power of two. Cross-check a few decades.
    for e in -40i32..=40 {
        let v = (2.0f32).powi(e);
        assert_eq!(f32_to_ue8m0(v), ((e + 127) as u8), "2^{e}");
        // And the decode inverts it:
        let scale = ue8m0_decode(f32_to_ue8m0(v));
        assert!(
            (scale - v).abs() <= v.abs() * 1e-6,
            "decode(encode(2^{e})) != 2^{e}"
        );
    }
}

// ---------------------------------------------------------------------------
// transform_sf host packing
// ---------------------------------------------------------------------------

#[test]
fn transform_sf_host_layout_matches_kernel_semantics() {
    // mn = 33, sf_k = 7 → tma_aligned(mn) = 36, rows = ceil(7/4) = 2.
    let mn = 33u32;
    let sf_k = 7u32;
    let mut rng = Lcg(0xDEADBEEF);
    let mut sf = Vec::with_capacity((mn * sf_k) as usize);
    for _ in 0..(mn * sf_k) {
        // powers of two only, as required by transform_sf
        sf.push((2.0f32).powi(rng.next_i32_bounded(14) - 7));
    }
    let packed = transform_sf_host(&sf, mn, SfGran::G32);
    let cols = tma_aligned(mn, 4);
    let rows = ceil_div(sf_k, 4);
    assert_eq!(packed.len(), (rows * cols) as usize);
    assert_eq!(cols, 36, "tma alignment to 16B rows (4 int32)");
    // Spot-check every element against the definition:
    // word(k/4, m) byte (k%4) = exponent byte of sf[m, k]; padding = 0.
    for m in 0..mn {
        for k in 0..sf_k {
            let w = packed[(k / 4 * cols + m) as usize];
            let byte = ((w >> (8 * (k % 4))) & 0xff) as u8;
            assert_eq!(
                byte,
                f32_to_ue8m0(sf[(m * sf_k + k) as usize]),
                "at (m={m}, k={k})"
            );
        }
    }
    // Padding columns (m >= 33) must be zero:
    for r in 0..rows {
        for m in mn..cols {
            assert_eq!(packed[(r * cols + m) as usize], 0, "padding col m={m}");
        }
    }
}

// helper on Lcg
impl Lcg {
    fn next_i32_bounded(&mut self, n: i32) -> i32 {
        if n <= 0 {
            0
        } else {
            (self.next_u32() % n as u32) as i32
        }
    }
}

// ---------------------------------------------------------------------------
// quant / dequant round trips (the full activation pipeline)
// ---------------------------------------------------------------------------

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let denom = x.abs().max(y.abs()).max(1e-6);
        let e = ((x - y) / denom).abs();
        if e > worst {
            worst = e;
        }
    }
    worst
}

#[test]
fn quant_dequant_fp8_round_trip_error() {
    // E4M3 with per-32 UE8M0 scales: worst-case relative error of a random
    // uniform signal is bounded by half the E4M3 grid step at the top of the
    // block (1/16 relative at most between grid points 8 → 2^-4 ... in
    // practice ~2-3% for uniform input).
    let (m, k) = (16u32, 256u32);
    let mut rng = Lcg(12345);
    let x: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-4.0, 4.0)).collect();
    let q = quant_mx_host(&x, m, k, Dtype::Fp8, SfGran::G32);
    assert_eq!(q.data.len(), (m * k) as usize);
    let back = dequant_mx_host(&q.data, &q.sf, q.sf_cols, m, k, Dtype::Fp8, SfGran::G32);
    let err = max_rel_err(&x, &back);
    assert!(err < 0.08, "FP8 round-trip relative error too large: {err}");
    // Scales must be powers of two and every byte present:
    for w in &q.sf {
        for b in 0..4 {
            let code = ((w >> (8 * b)) & 0xff) as u8;
            if code != 0 {
                let v = ue8m0_decode(code);
                assert_eq!(v, (2.0f32).powi(code as i32 - 127));
            }
        }
    }
}

#[test]
fn quant_dequant_fp4_round_trip_error() {
    // E2M1 has coarse precision: per-element bound, checked against each
    // element's own block scale s (32-element UE8M0):
    //   |x - deq(x)| <= 0.35 * max(|x|, s)
    // plus an aggregate check with the robust metric |x-y| / (|x| + s).
    let (m, k) = (16u32, 256u32);
    let mut rng = Lcg(54321);
    let x: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-6.0, 6.0)).collect();
    let q = quant_mx_host(&x, m, k, Dtype::Fp4, SfGran::G32);
    assert_eq!(
        q.data.len(),
        (m * k / 2) as usize,
        "FP4 packs 2 nibbles/byte"
    );
    let back = dequant_mx_host(&q.data, &q.sf, q.sf_cols, m, k, Dtype::Fp4, SfGran::G32);
    let mut worst_bound = 0f32;
    let mut agg_sum = 0f32;
    for idx in 0..(m * k) as usize {
        let row = (idx as u32 / k) as usize;
        let col = (idx as u32 % k) as usize;
        let slot = col / 32;
        let word = q.sf[(slot / 4) * q.sf_cols as usize + row];
        let exp = ((word >> (8 * (slot % 4))) & 0xff) as u8;
        let s = ue8m0_decode(exp);
        let xi = x[idx];
        let yi = back[idx];
        let bound = 0.35 * xi.abs().max(s);
        let slack = (bound - (xi - yi).abs()) / bound;
        if slack < worst_bound {
            worst_bound = slack;
        }
        assert!(
            (xi - yi).abs() <= bound,
            "FP4 element (row={row}, col={col}): |{xi} - {yi}| > 0.35*max(|x|, s={s})"
        );
        agg_sum += (xi - yi).abs() / (xi.abs() + s);
    }
    let agg = agg_sum / (m * k) as f32;
    assert!(agg < 0.25, "FP4 aggregate error too large: {agg}");
}

#[test]
fn quant_fp4_scale_selection_is_upstream_formula() {
    // Bit-exact checks of the amax→UE8M0 exponent selection against the
    // upstream bit-trick formula (verified against DeepGEMM common/math.cuh:
    // kQuantMaxMantissa 0x60/0x40 << 16, kQuantMaxExponent 8/2, floor 105/1).
    //
    // NOTE the `+ 0x7FFFFF` mantissa mask in the formula implements an
    // RNE round-up of amax to the next power of two when the fraction
    // exceeds 0.5 — so amax=7.0 rounds to 8 → scale 2 (7/2 = 3.5 ≤ 6),
    // and amax=500 rounds to 512 → scale 2 (512/2 = 256 ≤ 448). At the
    // exact quant max (6 / 448, fraction 0.5 = tie) it stays in the lower
    // binade → scale 1.
    assert_eq!(ue8m0_exp_fp4(6.0), 127); // 6 = E2M1 quant max, scale 1
    assert_eq!(ue8m0_exp_fp4(7.0), 128); // 7 → RNE up to 8 → scale 2
    assert_eq!(ue8m0_exp_fp4(8.0), 128); // → scale 2
    assert_eq!(ue8m0_exp_fp8(448.0), 127); // 448 = E4M3 quant max, scale 1
    assert_eq!(ue8m0_exp_fp8(500.0), 128); // 500 → RNE up to 512 → scale 2
    assert_eq!(ue8m0_exp_fp8(512.0), 128); // → scale 2
                                           // Floors: fp8 105 (amax 1e-4), fp4 1 (max(amax, 6*2^-126)).
    assert_eq!(ue8m0_exp_fp8(1e-30), 105);
    assert_eq!(ue8m0_exp_fp8(0.0), 105);
    assert_eq!(ue8m0_exp_fp4(0.0), 1);
    // The fp4 floor is a *floor*, not the output: tiny amax still gets a
    // meaningful scale (here ≈ 2^(result-127)); the result is ≥ 1 by the
    // max(.., 1 + 2) − 2 clamp.
    assert!(ue8m0_exp_fp4(1e-30) >= 1 && ue8m0_exp_fp4(1e-30) <= 30);
}

// ---------------------------------------------------------------------------
// Golden block-scaled GEMM
// ---------------------------------------------------------------------------

#[test]
fn golden_gemm_matches_direct_fp32_matmul() {
    // Quantize random operands, run the golden block-scaled GEMM, and compare
    // against a plain f32 matmul of the *dequantized* operands. They must
    // agree to ~1e-5 (both are exact arithmetic on the same codes; the golden
    // GEMM multiplies decoded values including scales, the direct matmul does
    // the same after dequant — so agreement is near-exact).
    let (m, n, k) = (32u32, 48u32, 256u32);
    let mut rng = Lcg(999);
    let xa: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-4.0, 4.0)).collect();
    let xb: Vec<f32> = (0..n * k).map(|_| rng.next_f32(-4.0, 4.0)).collect();
    let qa = quant_mx_host(&xa, m, k, Dtype::Fp8, SfGran::G32);
    let qb = quant_mx_host(&xb, n, k, Dtype::Fp8, SfGran::G32);
    let da = dequant_mx_host(&qa.data, &qa.sf, qa.sf_cols, m, k, Dtype::Fp8, SfGran::G32);
    let db = dequant_mx_host(&qb.data, &qb.sf, qb.sf_cols, n, k, Dtype::Fp8, SfGran::G32);

    let a = GoldenOperand {
        dtype: Dtype::Fp8,
        rows: m,
        k,
        data: &qa.data,
        sf: &qa.sf,
        sf_cols: qa.sf_cols,
        gran: 32,
    };
    let b = GoldenOperand {
        dtype: Dtype::Fp8,
        rows: n,
        k,
        data: &qb.data,
        sf: &qb.sf,
        sf_cols: qb.sf_cols,
        gran: 32,
    };
    let got = gemm_ref(&a, &b, m, n, k);

    // Direct reference on dequantized values.
    let mut want = vec![0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += da[(i * k + kk) as usize] * db[(j * k + kk) as usize];
            }
            want[(i * n + j) as usize] = acc;
        }
    }
    let err = max_rel_err(&got, &want);
    assert!(err < 1e-4, "golden GEMM vs direct matmul: {err}");
}

#[test]
fn golden_gemm_fp4_and_bf16_output_pack() {
    // FP4 operands end-to-end (E2M1 + UE8M0/32) and a bf16 packing sanity check.
    let (m, n, k) = (24u32, 32u32, 128u32);
    let mut rng = Lcg(0xC0FFEE);
    let xa: Vec<f32> = (0..m * k).map(|_| rng.next_f32(-6.0, 6.0)).collect();
    let xb: Vec<f32> = (0..n * k).map(|_| rng.next_f32(-6.0, 6.0)).collect();
    let qa = quant_mx_host(&xa, m, k, Dtype::Fp4, SfGran::G32);
    let qb = quant_mx_host(&xb, n, k, Dtype::Fp4, SfGran::G32);
    let a = GoldenOperand {
        dtype: Dtype::Fp4,
        rows: m,
        k,
        data: &qa.data,
        sf: &qa.sf,
        sf_cols: qa.sf_cols,
        gran: 32,
    };
    let b = GoldenOperand {
        dtype: Dtype::Fp4,
        rows: n,
        k,
        data: &qb.data,
        sf: &qb.sf,
        sf_cols: qb.sf_cols,
        gran: 32,
    };
    let got = gemm_ref(&a, &b, m, n, k);
    assert_eq!(got.len(), (m * n) as usize);
    // Non-trivial: at least some outputs must be non-zero.
    assert!(
        got.iter().any(|v| v.abs() > 1e-3),
        "all-zero FP4 GEMM output"
    );

    // bf16 RNE packing: ties to even, saturation not applied.
    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
    assert_eq!(f32_to_bf16_bits(0.5), 0x3f00);
    // 1 + 2^-8 is exactly between bf16 1.0 (m=0, even) and 1+2^-7 (m=1, odd)
    // → ties to even → stays 1.0.
    assert_eq!(f32_to_bf16_bits(1.0 + 2f32.powi(-8)), 0x3f80);
    // 1 + 3*2^-8 is exactly between 1+2^-7 (m=1, odd) and 1+2*2^-7 (m=2,
    // even) → ties to even → 1.015625.
    assert_eq!(f32_to_bf16_bits(1.0 + 3f32 * 2f32.powi(-8)), 0x3f82);
    // 1 + 6*2^-8 = 1 + 3*2^-7 is exactly on-grid → code 3.
    assert_eq!(f32_to_bf16_bits(1.0 + 6f32 * 2f32.powi(-8)), 0x3f83);
}

// ---------------------------------------------------------------------------
// Golden MQA logits
// ---------------------------------------------------------------------------

#[test]
fn mqa_logits_ref_matches_naive_computation() {
    let (num_tokens, num_heads, head_dim, num_kv) = (8u32, 4u32, 64u32, 32u32);
    let mut rng = Lcg(31337);
    let q: Vec<f32> = (0..num_tokens * num_heads * head_dim)
        .map(|_| rng.next_f32(-1.0, 1.0))
        .collect();
    let kv: Vec<f32> = (0..num_kv * head_dim)
        .map(|_| rng.next_f32(-1.0, 1.0))
        .collect();
    let weights: Vec<f32> = (0..num_tokens * num_heads)
        .map(|_| rng.next_f32(0.1, 0.9))
        .collect();
    // Variable spans per token.
    let k_start: Vec<u32> = (0..num_tokens).map(|t| t % 4).collect();
    let k_end: Vec<u32> = (0..num_tokens).map(|t| 32 - t % 3).collect();

    let got = mqa_logits_ref(
        &q, &kv, &weights, num_tokens, num_heads, head_dim, num_kv, &k_start, &k_end,
    );

    // Naive re-implementation with an independent loop structure.
    for t in 0..num_tokens {
        for kvv in 0..num_kv {
            let in_span = kvv >= k_start[t as usize] && kvv < k_end[t as usize];
            let idx = (t * num_kv + kvv) as usize;
            if !in_span {
                assert_eq!(got[idx], 0.0, "outside span must stay zero");
                continue;
            }
            let mut acc = 0f64;
            for h in 0..num_heads {
                let mut dot = 0f64;
                for d in 0..head_dim {
                    dot += q[((t * num_heads + h) * head_dim + d) as usize] as f64
                        * kv[(kvv * head_dim + d) as usize] as f64;
                }
                if dot > 0.0 {
                    acc += weights[(t * num_heads + h) as usize] as f64 * dot;
                }
            }
            assert!(
                (got[idx] - acc as f32).abs() <= 1e-4 * (1.0 + acc.abs() as f32),
                "logits mismatch t={t} kv={kvv}: got {} want {}",
                got[idx],
                acc
            );
        }
    }
}
