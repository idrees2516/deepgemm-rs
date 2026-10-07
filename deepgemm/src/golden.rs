//! CPU golden model: bit-exact, portable re-implementation of the numeric
//! semantics of the GPU kernels.
//!
//! # Why a golden model?
//!
//! The GPU kernels in this crate (`tcgen05.mma`, TMA, UTCCP ...) cannot run
//! without an SM100 GPU. To make the *numeric and layout* semantics testable
//! anywhere — a CI sandbox, a laptop, an H100 dev box — this module
//! re-implements, in plain Rust, exactly what the hardware path computes:
//!
//! ```text
//!   GPU path (needs B200)                 CPU golden path (this module)
//!   ---------------------                 -----------------------------
//!   TMA copy of A/B        ──┐            same global-memory layouts,
//!   tcgen05.mma (FP4/FP8)   │            software decode of E2M1/E4M3,
//!   UTCCP scale factors     ├─ equal ──>  UE8M0 scale lookup, f64 dot
//!   tcgen05.ld + epilogue   │            products, bf16/f32 output.
//!                           ──┘
//! ```
//!
//! Every function here mirrors a specific kernel and is cross-checked
//! against it in the e2e tests (`tests/e2e_gpu.rs`, feature `e2e`) and in
//! the pure-CPU tests (`tests/golden_tests.rs`), so a mismatch is caught
//! either in the sandbox or on the B200 — whichever runs first.
//!
//! # Floating-point formats used by DeepGEMM (quick reference)
//!
//! ```text
//!   E4M3 (FP8)   1-bit sign | 4-bit exp (bias 7) | 3-bit mantissa
//!        normal  : (1 + m/8) * 2^(e-7)          e in [1,15]
//!        subnorm : m * 2^-9       (exp field 0) max finite = 448
//!        grid    : ..., 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, ...
//!   E2M1 (FP4)   1-bit sign | 2-bit exp (bias 1) | 1-bit mantissa
//!        grid    : 0, 0.5, 1, 1.5, 2, 3, 4, 6   (positive half;
//!                  negative half mirrored) — only 16 codes total.
//!   UE8M0 (SF)   8-bit *pure exponent* (bias 127), no mantissa, no sign:
//!        value   = 2^(code - 127)  in [2^-127, 2^127]; the all-zero byte
//!                  is reserved (never produced by our kernels).
//! ```
//!
//! Block-scaled GEMM semantics (OCP MX, as used by DeepGEMM / DeepSeek-V4 /
//! MiMo on Blackwell):
//! ```text
//!   D[i,j] = sum_k  A[i,k] * SA[i, k/g] * B[j,k] * SB[j, k/g]
//!            \_ 4/8-bit code _/   \_ per-block power-of-two scale _/
//!            g = 32 (MX) or 128 (DeepSeek FP8 recipe)
//! ```

use crate::types::{Dtype, SfGran};

// ---------------------------------------------------------------------------
// Small helpers shared with the layout code
// ---------------------------------------------------------------------------

/// `ceil_div(a, b)` for u32.
pub fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Align `x` up to a multiple of `to`.
pub fn align_up(x: u32, to: u32) -> u32 {
    debug_assert!(to > 0);
    x.div_ceil(to) * to
}

/// TMA row alignment for scale-factor tensors, in *elements* of the given
/// element size: a TMA transaction accesses rows of 16 bytes, so an int32
/// SF row must contain a multiple of `16 / elem_size` elements.
/// (Port of `heuristics::tma_aligned_size` and the kernel's
/// `align_u32(mn, 16 / elem_size)`.)
pub fn tma_aligned(mn: u32, elem_size: u32) -> u32 {
    align_up(mn, (16 / elem_size).max(1))
}

// ---------------------------------------------------------------------------
// Format decode / encode
// ---------------------------------------------------------------------------

/// Decode one E4M3 (FP8) byte to f32. `0x7f`/`0xff` are NaN on this format.
pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let mant = (b & 0x7) as f32;
    if exp == 0 {
        // Subnormal: m * 2^-9 (bias 7, hidden bit absent).
        sign * mant * (2.0f32).powi(-9)
    } else if b & 0x7f == 0x7f {
        f32::NAN // 0x7f / 0xff encodings are NaN in CUDA E4M3
    } else {
        sign * (1.0 + mant / 8.0) * (2.0f32).powi(exp - 7)
    }
}

/// Decode one E2M1 (FP4) nibble to f32. Grid: {0, .5, 1, 1.5, 2, 3, 4, 6}.
pub fn e2m1_decode(n: u8) -> f32 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = GRID[(n & 0x7) as usize];
    if n & 0x8 != 0 {
        -v
    } else {
        v
    }
}

/// Decode one UE8M0 scale byte: value = 2^(code-127). Code 0 is reserved
/// (treated as 2^-127; our kernels never emit it).
pub fn ue8m0_decode(code: u8) -> f32 {
    (2.0f32).powi(code as i32 - 127)
}

/// Encode a power-of-two f32 into a UE8M0 byte by taking the IEEE-754
/// exponent field (bits 23..31). This is bit-identical to the GPU packing
/// in `transform_sf_impl` (`packed |= value >> 23` etc.) — inputs must be
/// finite (the upstream kernel asserts exponent-only values).
pub fn f32_to_ue8m0(v: f32) -> u8 {
    ((v.to_bits() >> 23) & 0xff) as u8
}

/// The positive finite grids, used for round-to-nearest-even quantization.
fn e4m3_positive_grid() -> Vec<f64> {
    let mut g = Vec::with_capacity(72);
    for m in 0..8u32 {
        g.push((m as f64) * 2f64.powi(-9)); // subnormals: 0 .. 7*2^-9
    }
    for e in 1..=15i32 {
        for m in 0..8u32 {
            g.push((1.0 + m as f64 / 8.0) * 2f64.powi(e - 7));
        }
    }
    g
}

fn e2m1_positive_grid() -> Vec<f64> {
    vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
}

/// Round-to-nearest-even onto an ascending grid. Ties (exact midpoints)
/// resolve to the neighbor whose *mantissa parity* is even — for both E4M3
/// and E2M1 the grid is `1.m`-structured inside each binade, so "even code"
/// is the correct tie-break, matching PTX `cvt.rn.satfinite.*`.
/// Returns the chosen grid *value* and its *index*.
fn rne_on_grid(a: f64, grid: &[f64]) -> (f64, usize) {
    if a <= grid[0] {
        return (grid[0], 0);
    }
    if a >= *grid.last().unwrap() {
        let i = grid.len() - 1;
        return (grid[i], i); // satfinite: clamp, never inf
    }
    // Binary search for the bracketing pair.
    let mut lo = 0usize;
    let mut hi = grid.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if grid[mid] <= a {
            lo = mid
        } else {
            hi = mid
        }
    }
    let (gl, gh) = (grid[lo], grid[hi]);
    let mid = (gl + gh) / 2.0;
    if a < mid {
        (gl, lo)
    } else if a > mid {
        (gh, hi)
    } else {
        // Exact tie: choose the even code (grid index parity == mantissa parity
        // because each binade starts at an even index in these grids).
        if lo % 2 == 0 {
            (gl, lo)
        } else {
            (gh, hi)
        }
    }
}

/// Quantize `v` to the nearest E4M3 code (RNE, saturate to ±448).
/// Mirrors PTX `cvt.rn.satfinite.e4m3x2.f32`.
pub fn quant_e4m3(v: f32) -> u8 {
    let grid = e4m3_positive_grid();
    let (mag, code) = rne_on_grid(v.abs() as f64, &grid);
    let _ = mag;
    if v.is_sign_negative() {
        0x80 | code as u8
    } else {
        code as u8
    }
}

/// Quantize `v` to the nearest E2M1 nibble (RNE, saturate to ±6).
/// Mirrors PTX `cvt.rn.satfinite.e2m1x2.f32`.
pub fn quant_e2m1(v: f32) -> u8 {
    let grid = e2m1_positive_grid();
    let (mag, code) = rne_on_grid(v.abs() as f64, &grid);
    let _ = mag;
    if v.is_sign_negative() {
        0x8 | code as u8
    } else {
        code as u8
    }
}

// ---------------------------------------------------------------------------
// UE8M0 scale-factor exponent selection (bit-exact port of the kernels'
// `get_ue8m0_sf_exp_fp8` / `_fp4` — see kernels/layout_quant.cu)
// ---------------------------------------------------------------------------

/// FP8 (E4M3) recipe: quant max = 448 = 1.75 * 2^8, amax floor 1e-4
/// (kMinSFExponent 105). Returns the biased UE8M0 exponent.
pub fn ue8m0_exp_fp8(amax: f32) -> u8 {
    const MANTISSA_MASK: u32 = 0x7f_ffff;
    const QUANT_MAX_MANTISSA: u32 = 0x60 << (23 - 7); // 448 = 1.75 * 2^8
    const QUANT_MAX_EXPONENT: u32 = 8;
    const MIN_SF_EXPONENT: u32 = 105;
    let bits = amax.to_bits();
    let rounded = (bits
        .wrapping_add(MANTISSA_MASK)
        .wrapping_sub(QUANT_MAX_MANTISSA))
        >> 23;
    rounded
        .max(MIN_SF_EXPONENT + QUANT_MAX_EXPONENT)
        .wrapping_sub(QUANT_MAX_EXPONENT) as u8
}

/// FP4 (E2M1) recipe: quant max = 6 = 1.5 * 2^2, amax floor 2 (so that
/// amax/scale lands in [3, 6] — see upstream `get_ue8m0_sf_exp` for FP4).
pub fn ue8m0_exp_fp4(amax: f32) -> u8 {
    const MANTISSA_MASK: u32 = 0x7f_ffff;
    const QUANT_MAX_MANTISSA: u32 = 0x40 << (23 - 7); // 6 = 1.5 * 2^2
    const QUANT_MAX_EXPONENT: u32 = 2;
    const MIN_SF_EXPONENT: u32 = 1;
    let bits = amax.to_bits();
    let rounded = (bits
        .wrapping_add(MANTISSA_MASK)
        .wrapping_sub(QUANT_MAX_MANTISSA))
        >> 23;
    rounded
        .max(MIN_SF_EXPONENT + QUANT_MAX_EXPONENT)
        .wrapping_sub(QUANT_MAX_EXPONENT) as u8
}

// ---------------------------------------------------------------------------
// Host-side `transform_sf` (bit-exact port of the GPU kernel)
// ---------------------------------------------------------------------------
//
// GPU layout (see kernels/layout_quant.cu and docs/concepts.md):
//
//   input : f32 [mn, sf_k] row-major, values are powers of two
//   output: u32 [ceil(sf_k/4), align_up(mn, 4)] — "MN-major":
//
//        word(k, m) = SF[m, k]   | SF[m, k+1] << 8 | SF[m, k+2] << 16
//                    | SF[m, k+3] << 24        (SF[x] = UE8M0 byte)
//
//   i.e. *consecutive K scales* are packed into the bytes of one int32 and
//   the fast dimension of the array is MN — exactly the transposed layout
//   the tensor core's scale-factor path consumes.

/// Host equivalent of the `transform_sf` API/kernel: FP32 scales
/// `[mn, sf_k]` row-major → packed UE8M0 words `[ceil(sf_k/4), tma_aligned(mn)]`.
pub fn transform_sf_host(sf: &[f32], mn: u32, _gran: SfGran) -> Vec<u32> {
    assert!(mn > 0);
    let sf_k = (sf.len() / mn as usize) as u32;
    assert_eq!(
        (sf.len() / mn as usize),
        sf_k as usize,
        "sf len must be mn * sf_k"
    );
    let cols = tma_aligned(mn, 4);
    let rows = ceil_div(sf_k, 4);
    let mut out = vec![0u32; (rows * cols) as usize];
    for m in 0..mn {
        for k in 0..sf_k {
            let v = sf[(m as usize) * sf_k as usize + k as usize];
            let byte = f32_to_ue8m0(v);
            let word = &mut out[(k as usize / 4) * cols as usize + m as usize];
            *word |= (byte as u32) << (8 * (k % 4));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Host-side activation quantization / dequantization (MXFP8 / MXFP4)
// ---------------------------------------------------------------------------

/// Result of host quantization: data bytes + packed SF words, same layouts
/// as the GPU kernels (data: row-major `[m, k]` (fp8) or `[m, k/2]` packed
/// nibbles (fp4); SF: `[ceil(k/g/4), tma_aligned(m)]`).
pub struct HostQuant {
    pub data: Vec<u8>,
    pub sf: Vec<u32>,
    pub sf_rows: u32,
    pub sf_cols: u32,
}

/// Host equivalent of the `quant_mx` kernel (bit-exact):
/// E4M3 or packed-E2M1 data plus UE8M0 per-`gran` scales.
pub fn quant_mx_host(x: &[f32], m: u32, k: u32, dtype: Dtype, gran: SfGran) -> HostQuant {
    assert_eq!(x.len(), (m * k) as usize, "x must be [m, k] row-major");
    let g = gran.k();
    assert!(k % g == 0, "K must be a multiple of the SF granularity");
    let num_groups = k / g;
    let sf_cols = tma_aligned(m, 4);
    let sf_rows = ceil_div(num_groups, 4);
    let is_fp4 = dtype == Dtype::Fp4;
    let data_len = if is_fp4 {
        (m * k / 2) as usize
    } else {
        (m * k) as usize
    };
    let mut data = vec![0u8; data_len];
    let mut sf = vec![0u32; (sf_rows * sf_cols) as usize];

    for row in 0..m {
        for gi in 0..num_groups {
            let base = gi * g;
            // amax over the group (same order as the kernel's warp reduce —
            // max is order-independent, so this is bit-exact).
            let mut amax: f32 = 0.0;
            for i in 0..g {
                let v = x[(row * k + base + i) as usize].abs();
                if v > amax {
                    amax = v;
                }
            }
            if amax < 1e-30 {
                amax = 1e-30;
            }
            let exp = if is_fp4 {
                ue8m0_exp_fp4(amax)
            } else {
                ue8m0_exp_fp8(amax)
            };
            let inv_sf = f32::from_bits(((254u32).wrapping_sub(exp as u32)) << 23); // 2^-exp
                                                                                    // Quantize each element: code = RNE(x * inv_sf).
            for i in 0..g {
                let v = x[(row * k + base + i) as usize] * inv_sf;
                if is_fp4 {
                    let kk = base + i;
                    let byte_idx = (row * k / 2 + kk / 2) as usize;
                    let nib = quant_e2m1(v);
                    if kk % 2 == 0 {
                        data[byte_idx] = (data[byte_idx] & 0xf0) | (nib & 0x0f);
                    } else {
                        data[byte_idx] = (data[byte_idx] & 0x0f) | (nib << 4);
                    }
                } else {
                    data[(row * k + base + i) as usize] = quant_e4m3(v);
                }
            }
            // Pack the scale byte at byte (gi % 4) of word [gi/4, row].
            let word = &mut sf[(gi as usize / 4) * sf_cols as usize + row as usize];
            *word |= (exp as u32) << (8 * (gi % 4));
        }
    }
    HostQuant {
        data,
        sf,
        sf_rows,
        sf_cols,
    }
}

/// Host equivalent of the `dequant_mx` kernel: data + packed SF → f32 `[m, k]`.
pub fn dequant_mx_host(
    data: &[u8],
    sf: &[u32],
    sf_cols: u32,
    m: u32,
    k: u32,
    dtype: Dtype,
    gran: SfGran,
) -> Vec<f32> {
    let g = gran.k();
    let is_fp4 = dtype == Dtype::Fp4;
    let mut out = vec![0f32; (m * k) as usize];
    for row in 0..m {
        for col in 0..k {
            let slot = col / g; // scale slot along K
            let packed_row = slot / 4;
            let byte_in_word = slot % 4;
            let word = sf[(packed_row * sf_cols + row) as usize];
            let exp = ((word >> (8 * byte_in_word)) & 0xff) as u8;
            let scale = ue8m0_decode(exp);
            let v = if is_fp4 {
                let b = data[(row * k / 2 + col / 2) as usize];
                e2m1_decode(if col & 1 == 0 { b & 0xf } else { b >> 4 }) * scale
            } else {
                e4m3_decode(data[(row * k + col) as usize]) * scale
            };
            out[(row * k + col) as usize] = v;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Golden block-scaled GEMM (software model of the tcgen05.mma pipeline)
// ---------------------------------------------------------------------------

/// Operand view for the golden GEMM: K-major `[rows, k]` data plus the
/// packed-UE8M0 SF tensor exactly as `transform_sf`/`quant_mx` produce it.
pub struct GoldenOperand<'a> {
    pub dtype: Dtype,
    pub rows: u32,
    pub k: u32,
    /// FP8: `[rows*k]`; FP4: `[rows*k/2]` nibble-packed (even k = low nibble).
    pub data: &'a [u8],
    /// Packed SF words `[ceil(k/g/4), tma_aligned(rows)]`.
    pub sf: &'a [u32],
    pub sf_cols: u32,
    pub gran: u32,
}

impl<'a> GoldenOperand<'a> {
    /// Decode element (r, kk) including its per-block scale.
    fn scaled(&self, r: u32, kk: u32) -> f64 {
        let code = if self.dtype == Dtype::Fp4 {
            let b = self.data[((r * self.k / 2) + kk / 2) as usize];
            if kk & 1 == 0 {
                b & 0xf
            } else {
                b >> 4
            }
        } else {
            self.data[(r * self.k + kk) as usize]
        };
        let raw = if self.dtype == Dtype::Fp4 {
            e2m1_decode(code) as f64
        } else {
            e4m3_decode(code) as f64
        };
        let g = self.gran;
        let slot = kk / g;
        let word = self.sf[((slot / 4) * self.sf_cols + r) as usize];
        let exp = ((word >> (8 * (slot % 4))) & 0xff) as u8;
        raw * ue8m0_decode(exp) as f64
    }
}

/// Golden block-scaled GEMM: `D = A @ B^T` in f64 (accumulation order differs
/// from the tensor core's, so treat as reference with tolerance, not exact).
pub fn gemm_ref(a: &GoldenOperand, b: &GoldenOperand, m: u32, n: u32, k: u32) -> Vec<f32> {
    assert_eq!(a.k, k);
    assert_eq!(b.k, k);
    let mut out = vec![0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += a.scaled(i, kk) * b.scaled(j, kk);
            }
            out[(i * n + j) as usize] = acc as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Golden MQA logits (weighted-ReLU scoring, MLA lightning indexer)
// ---------------------------------------------------------------------------

/// `logits[t, kv] = sum_h w[t,h] * relu(q[t,h] . kv[kv])`, computed in f64.
/// `q` is `[num_tokens, num_heads, head_dim]` (fp8 grid + per-32 scales are
/// already applied by the caller — pass decoded f32), `kv` is
/// `[num_kv, head_dim]` f32, `weights` is `[num_tokens, num_heads]` f32.
#[allow(clippy::too_many_arguments)] // mirrors the kernel signature
pub fn mqa_logits_ref(
    q: &[f32],
    kv: &[f32],
    weights: &[f32],
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    num_kv: u32,
    k_start: &[u32],
    k_end: &[u32],
) -> Vec<f32> {
    let mut out = vec![0f32; (num_tokens * num_kv) as usize];
    for t in 0..num_tokens {
        let lo = k_start[t as usize];
        let hi = k_end[t as usize].min(num_kv);
        for kvv in lo..hi {
            let mut acc = 0f64;
            for h in 0..num_heads {
                let mut dot = 0f64;
                let qbase = ((t * num_heads + h) * head_dim) as usize;
                let kvbase = (kvv * head_dim) as usize;
                for d in 0..head_dim as usize {
                    dot += q[qbase + d] as f64 * kv[kvbase + d] as f64;
                }
                if dot > 0.0 {
                    acc += weights[(t * num_heads + h) as usize] as f64 * dot;
                }
            }
            out[(t * num_kv + kvv) as usize] = acc as f32;
        }
    }
    out
}

/// f32 → bf16 bits, round-to-nearest-even (matches the GPU epilogue's
/// `cvt.rn.bf16.f32` packing).
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fff + lsb;
    (((bits + rounding) >> 16) as u16) & 0x7fff | ((bits >> 31) as u16) << 15
}
