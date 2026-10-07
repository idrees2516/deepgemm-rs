//! GPU end-to-end tests (run on a B200/SM100 GPU):
//! `LD_LIBRARY_PATH=... cargo test -p deepgemm --features e2e --release -- --nocapture`

#![cfg(feature = "e2e")]

use deepgemm::prelude::*;

fn test_dev() -> (std::sync::Arc<Device>, DevStream) {
    let dev = Device::new(0).expect("CUDA device");
    let stream = DevStream::new(&dev).expect("stream");
    (dev, stream)
}

fn quantize_host(x: &[f32], k: u32, gran: u32, fp4: bool) -> (Vec<u8>, Vec<f32>) {
    // Host MX quantization reference (matches the GPU quant kernel).
    let n = x.len() as u32 / k;
    let mut data = vec![0u8; if fp4 { (x.len() / 2) as usize } else { x.len() }];
    let mut sfs = Vec::new();
    let sf_exp_of = |amax: f32| -> u32 {
        // Port of math::get_ue8m0_sf_exp.
        let (mant_mask, qmax_mant, qmax_exp, min_exp) = if fp4 {
            ((1u32 << 23) - 1, 0x40u32 << 16, 2u32, 1u32)
        } else {
            ((1u32 << 23) - 1, 0x60u32 << 16, 8u32, 105u32)
        };
        let bits = amax.to_bits();
        let rounded = (bits.wrapping_add(mant_mask - qmax_mant)) >> 23;
        rounded.max(min_exp + qmax_exp) - qmax_exp
    };
    for row in 0..n {
        for g in 0..k / gran {
            let base = (row * k + g * gran) as usize;
            let mut amax = 0f32;
            for i in 0..gran as usize {
                amax = amax.max(x[base + i].abs());
            }
            amax = amax.max(1e-30);
            let exp = sf_exp_of(amax);
            let inv = f32::from_bits((254 - exp) << 23);
            sfs.push(f32::from_bits((127 + exp) << 23)); // 2^exp
            for i in 0..gran as usize {
                let v = x[base + i] * inv;
                let idx = base + i;
                if fp4 {
                    let byte = idx / 2;
                    let nib = e2m1_quantize(v);
                    if idx % 2 == 0 {
                        data[byte] = nib;
                    } else {
                        data[byte] |= nib << 4;
                    }
                } else {
                    data[idx] = e4m3_quantize(v);
                }
            }
        }
    }
    (data, sfs)
}

fn e4m3_quantize(v: f32) -> u8 {
    // Round-to-nearest-even saturating E4M3 via f32 bits.
    let clamped = v.clamp(-448.0, 448.0);
    // bf16-style rounding on the 4-bit mantissa boundary.
    let bits = clamped.to_bits();
    let mant_mask = (1u32 << 23) - 1;
    let _ = mant_mask;
    // Simple approach: use the exponent/mantissa split.
    let sign = (bits >> 31) as u8;
    let abs = clamped.abs();
    let e = if abs == 0.0 { 0 } else { ((abs.log2().floor() as i32) + 1).clamp(0, 15) as u32 };
    let m = if e == 0 {
        (abs / 2f32.powi(-6) ).round() as u32 // subnormal: mantissa * 2^-6
    } else {
        let frac = abs / 2f32.powi(e as i32 - 7);
        ((frac - 1.0) * 8.0).round().clamp(0.0, 7.0) as u32
    };
    (sign << 7) | (e << 3) as u8 | m as u8
}

fn e2m1_quantize(v: f32) -> u8 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = v.abs();
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &c) in LUT.iter().enumerate() {
        let err = (a - c).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    if v < 0.0 { 0x8 | best } else { best }
}

fn e4m3_dequant(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * v
}

fn e2m1_dequant(n: u8) -> f32 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = LUT[(n & 7) as usize];
    if n & 8 != 0 { -v } else { v }
}

fn rand_data(n: usize, seed: u64) -> Vec<f32> {
    let mut x = 0x12345678u64.wrapping_add(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = ((x >> 40) as i32 as f32) / (1u32 << 22) as f32 - 1.0;
        out.push(f);
    }
    out
}

#[test]
fn e2e_quant_dequant_roundtrip() {
    let (dev, stream) = test_dev();
    let (m, k) = (64u32, 512u32);
    let x = rand_data((m * k) as usize, 7);

    for (dtype, gran) in [(Dtype::Fp8, SfGran::G32), (Dtype::Fp8, SfGran::G128), (Dtype::Fp4, SfGran::G32)] {
        let fp4 = dtype == Dtype::Fp4;
        let (data_dev, sf_dev) = quant_mx(&dev, &stream, &x, m, k, dtype, gran).expect("quant");
        // Verify SF exponents against the host reference.
        let (host_data, host_sf) = quantize_host(&x, k, gran.k(), fp4);
        let sf_host_packed = transform_sf(&dev, &stream, &host_sf, m, gran).expect("transform");
        let got_sf: Vec<u32> = download(&dev, &sf_dev.buf, stream.raw()).unwrap();
        let want_sf: Vec<u32> = download(&dev, &sf_host_packed.buf, stream.raw()).unwrap();
        let sf_len = sf_dev.buf.len / 4;
        let mut mismatches = 0;
        for i in 0..sf_len {
            if got_sf[i] != want_sf[i] {
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "SF mismatch for {dtype:?}/{gran:?}");

        // Data check: dequantize GPU data with host SF and compare to a
        // tolerance band around the input.
        let got_data: Vec<u8> = download(&dev, &data_dev, stream.raw()).unwrap();
        for i in 0..host_data.len() {
            assert_eq!(got_data[i], host_data[i], "quant data mismatch at {i} for {dtype:?}");
        }
    }
}

#[test]
fn e2e_fp8_gemm_nt() {
    let (dev, stream) = test_dev();
    let (m, n, k) = (256u32, 256u32, 512u32);
    let xa = rand_data((m * k) as usize, 1);
    let xb = rand_data((n * k) as usize, 2);

    for gran in [SfGran::G32, SfGran::G128] {
        let (a_data, a_sf_h) = quantize_host(&xa, k, gran.k(), false);
        let (b_data, b_sf_h) = quantize_host(&xb, k, gran.k(), false);
        let a_dev = alloc_and_upload(&dev, &a_data, stream.raw()).unwrap();
        let b_dev = alloc_and_upload(&dev, &b_data, stream.raw()).unwrap();
        let sfa = transform_sf(&dev, &stream, &a_sf_h, m, gran).unwrap();
        let sfb = transform_sf(&dev, &stream, &b_sf_h, n, gran).unwrap();

        let a = Operand { dtype: Dtype::Fp8, major: Major::K, rows: m, k, outer_stride: k,
                          sf: Some(sfa), data: a_dev };
        let b = Operand { dtype: Dtype::Fp8, major: Major::K, rows: n, k, outer_stride: k,
                          sf: Some(sfb), data: b_dev };
        let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
        let mut out = Output { dtype: Dtype::Bf16, rows: m, cols: n, stride: n, data: out_data };

        fp8_gemm_nt(&dev, &stream, &a, &b, &mut out, false).expect("gemm");

        // Host reference: dequant + matmul.
        let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
        let mut max_rel_err = 0f32;
        for r in 0..m as usize {
            for c in 0..n as usize {
                let mut acc = 0f64;
                for i in 0..k as usize {
                    let sa = a_sf_h[r * (k as usize / gran.k() as usize) + i / gran.k() as usize] as f64;
                    let sb = b_sf_h[c * (k as usize / gran.k() as usize) + i / gran.k() as usize] as f64;
                    acc += e4m3_dequant(a_data[r * k as usize + i]) as f64 * sa
                         * e4m3_dequant(b_data[c * k as usize + i]) as f64 * sb;
                }
                let ref_v = acc as f32;
                let bits = got[r * n as usize + c];
                let v = f32::from_bits(((bits as u32) << 16) & 0xffff0000);
                if ref_v.abs() > 1e-3 {
                    let rel = ((v - ref_v) / ref_v).abs();
                    max_rel_err = max_rel_err.max(rel);
                }
            }
        }
        assert!(max_rel_err < 0.06, "fp8 nt gran {gran:?} rel err {max_rel_err}");
    }
}

#[test]
fn e2e_fp4_gemm_nt_native() {
    let (dev, stream) = test_dev();
    let (m, n, k) = (256u32, 256u32, 512u32); // K must be a multiple of 256 for MXF4
    let k = 512;
    let xa = rand_data((m * k) as usize, 3);
    let xb = rand_data((n * k) as usize, 4);

    let (a_data, a_sf_h) = quantize_host(&xa, k, 32, true);
    let (b_data, b_sf_h) = quantize_host(&xb, k, 32, true);
    let a_dev = alloc_and_upload(&dev, &a_data, stream.raw()).unwrap();
    let b_dev = alloc_and_upload(&dev, &b_data, stream.raw()).unwrap();
    let sfa = transform_sf(&dev, &stream, &a_sf_h, m, SfGran::G32).unwrap();
    let sfb = transform_sf(&dev, &stream, &b_sf_h, n, SfGran::G32).unwrap();

    let a = Operand { dtype: Dtype::Fp4, major: Major::K, rows: m, k, outer_stride: k,
                      sf: Some(sfa), data: a_dev };
    let b = Operand { dtype: Dtype::Fp4, major: Major::K, rows: n, k, outer_stride: k,
                      sf: Some(sfb), data: b_dev };
    let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
    let mut out = Output { dtype: Dtype::Bf16, rows: m, cols: n, stride: n, data: out_data };

    fp4_gemm_nt(&dev, &stream, &a, &b, &mut out, false).expect("fp4 gemm");

    let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
    let mut max_rel_err = 0f32;
    for r in 0..m as usize {
        for c in 0..n as usize {
            let mut acc = 0f64;
            for i in 0..k as usize {
                let sa = a_sf_h[r * (k as usize / 32) + i / 32] as f64;
                let sb = b_sf_h[c * (k as usize / 32) + i / 32] as f64;
                let av = e2m1_dequant(a_data[(r * k as usize + i) / 2] >> (4 * ((r * k as usize + i) % 2)) & 0xf);
                let bv = e2m1_dequant(b_data[(c * k as usize + i) / 2] >> (4 * ((c * k as usize + i) % 2)) & 0xf);
                acc += av as f64 * sa * bv as f64 * sb;
            }
            let ref_v = acc as f32;
            let bits = got[r * n as usize + c];
            let v = f32::from_bits(((bits as u32) << 16) & 0xffff0000);
            if ref_v.abs() > 0.05 {
                let rel = ((v - ref_v) / ref_v).abs();
                max_rel_err = max_rel_err.max(rel);
            }
        }
    }
    // E2M1 has ~1-2 bits of mantissa; 4% average error is expected.
    assert!(max_rel_err < 0.12, "fp4 nt rel err {max_rel_err}");
}

#[test]
fn e2e_bf16_gemm_nt() {
    let (dev, stream) = test_dev();
    let (m, n, k) = (128u32, 128u32, 128u32);
    let xa = rand_data((m * k) as usize, 5);
    let xb = rand_data((n * k) as usize, 6);

    let host_bf16 = |v: f32| -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounding = 0x7fff + lsb;
        (((bits + rounding) >> 16) as u16) & 0x7fff | ((bits >> 31) as u16) << 15
    };
    let bf16_f = |h: u16| f32::from_bits(((h as u32) << 16) & 0xffff0000);

    let a_h: Vec<u16> = xa.iter().map(|&v| host_bf16(v)).collect();
    let b_h: Vec<u16> = xb.iter().map(|&v| host_bf16(v)).collect();
    let a_dev = alloc_and_upload(&dev, &a_h, stream.raw()).unwrap();
    let b_dev = alloc_and_upload(&dev, &b_h, stream.raw()).unwrap();

    let a = Operand { dtype: Dtype::Bf16, major: Major::K, rows: m, k, outer_stride: k,
                      sf: None, data: a_dev };
    let b = Operand { dtype: Dtype::Bf16, major: Major::K, rows: n, k, outer_stride: k,
                      sf: None, data: b_dev };
    let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
    let mut out = Output { dtype: Dtype::Bf16, rows: m, cols: n, stride: n, data: out_data };

    bf16_gemm_nt(&dev, &stream, &a, &b, &mut out, false).expect("bf16 gemm");

    let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
    let mut max_err = 0f32;
    for r in 0..m as usize {
        for c in 0..n as usize {
            let mut acc = 0f64;
            for i in 0..k as usize {
                acc += bf16_f(a_h[r * k as usize + i]) as f64 * bf16_f(b_h[c * k as usize + i]) as f64;
            }
            let v = bf16_f(got[r * n as usize + c]);
            max_err = max_err.max((v - acc as f32).abs());
        }
    }
    assert!(max_err < 1.0, "bf16 nt abs err {max_err}");
}

#[test]
fn e2e_m_grouped_masked() {
    let (dev, stream) = test_dev();
    let groups = 4u32;
    let m = 64u32; // per-group M (padding inside)
    let n = 128u32;
    let k = 512u32;
    let xa = rand_data((groups * m * k) as usize, 7);
    let xb = rand_data((groups * n * k) as usize, 8);

    // Build per-group stacked operands.
    let mut max_rel = 0f32;
    let (a_data, a_sf_h) = quantize_host(&xa, k, 32, false);
    let (b_data, b_sf_h) = quantize_host(&xb, k, 32, false);
    let a_dev = alloc_and_upload(&dev, &a_data, stream.raw()).unwrap();
    let b_dev = alloc_and_upload(&dev, &b_data, stream.raw()).unwrap();
    let sfa = transform_sf(&dev, &stream, &a_sf_h, groups * m, SfGran::G32).unwrap();
    let sfb = transform_sf(&dev, &stream, &b_sf_h, groups * n, SfGran::G32).unwrap();

    let a = Operand { dtype: Dtype::Fp8, major: Major::K, rows: m, k, outer_stride: k,
                      sf: Some(sfa), data: a_dev };
    let b = Operand { dtype: Dtype::Fp8, major: Major::K, rows: n, k, outer_stride: k,
                      sf: Some(sfb), data: b_dev };
    let out_data = DevBuffer::alloc_zeros(&dev, (groups * m * n * 2) as usize).unwrap();
    let mut out = Output { dtype: Dtype::Bf16, rows: m, cols: n, stride: n, data: out_data };

    let masked = vec![m; groups as usize];
    let masked_buf = alloc_and_upload(&dev, &masked, stream.raw()).unwrap();
    m_grouped_gemm_nt_masked(&dev, &stream, &a, &b, &mut out, &masked_buf, groups, m)
        .expect("masked gemm");

    let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
    for g in 0..groups as usize {
        for r in 0..m as usize {
            for c in 0..n as usize {
                let mut acc = 0f64;
                for i in 0..k as usize {
                    let sa = a_sf_h[(g * m as usize + r) * (k as usize / 32) + i / 32] as f64;
                    let sb = b_sf_h[(g * n as usize + c) * (k as usize / 32) + i / 32] as f64;
                    acc += e4m3_dequant(a_data[(g * m as usize + r) * k as usize + i]) as f64 * sa
                         * e4m3_dequant(b_data[(g * n as usize + c) * k as usize + i]) as f64 * sb;
                }
                let ref_v = acc as f32;
                let bits = got[(g * m as usize + r) * n as usize + c];
                let v = f32::from_bits(((bits as u32) << 16) & 0xffff0000);
                if ref_v.abs() > 1e-2 {
                    max_rel = max_rel.max(((v - ref_v) / ref_v).abs());
                }
            }
        }
    }
    assert!(max_rel < 0.06, "masked rel err {max_rel}");
}

#[test]
fn e2e_m_grouped_contiguous() {
    let (dev, stream) = test_dev();
    let groups = 4u32;
    let per_group = 64u32;
    let m = groups * per_group; // already 128-aligned
    let n = 128u32;
    let k = 512u32;
    let xa = rand_data((m * k) as usize, 9);
    let xb = rand_data((groups * n * k) as usize, 10);

    let (a_data, a_sf_h) = quantize_host(&xa, k, 32, false);
    let (b_data, b_sf_h) = quantize_host(&xb, k, 32, false);
    let a_dev = alloc_and_upload(&dev, &a_data, stream.raw()).unwrap();
    let b_dev = alloc_and_upload(&dev, &b_data, stream.raw()).unwrap();
    let sfa = transform_sf(&dev, &stream, &a_sf_h, m, SfGran::G32).unwrap();
    let sfb = transform_sf(&dev, &stream, &b_sf_h, groups * n, SfGran::G32).unwrap();

    let a = Operand { dtype: Dtype::Fp8, major: Major::K, rows: m, k, outer_stride: k,
                      sf: Some(sfa), data: a_dev };
    let b = Operand { dtype: Dtype::Fp8, major: Major::K, rows: n, k, outer_stride: k,
                      sf: Some(sfb), data: b_dev };
    let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
    let mut out = Output { dtype: Dtype::Bf16, rows: m, cols: n, stride: n, data: out_data };

    let mut m_indices = Vec::new();
    for g in 0..groups {
        for _ in 0..per_group / 128 * 128 / 128 {
            m_indices.push(g as i32);
        }
    }
    let m_idx = alloc_and_upload(&dev, &m_indices, stream.raw()).unwrap();
    m_grouped_gemm_nt_contiguous(&dev, &stream, &a, &b, &mut out, &m_idx, groups)
        .expect("contiguous gemm");

    // Reference.
    let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
    let mut max_rel = 0f32;
    for r in 0..m as usize {
        let g = m_indices[r / 128] as usize;
        for c in 0..n as usize {
            let mut acc = 0f64;
            for i in 0..k as usize {
                let sa = a_sf_h[r * (k as usize / 32) + i / 32] as f64;
                let sb = b_sf_h[(g * n as usize + c) * (k as usize / 32) + i / 32] as f64;
                acc += e4m3_dequant(a_data[r * k as usize + i]) as f64 * sa
                     * e4m3_dequant(b_data[(g * n as usize + c) * k as usize + i]) as f64 * sb;
            }
            let ref_v = acc as f32;
            let bits = got[r * n as usize + c];
            let v = f32::from_bits(((bits as u32) << 16) & 0xffff0000);
            if ref_v.abs() > 1e-2 {
                max_rel = max_rel.max(((v - ref_v) / ref_v).abs());
            }
        }
    }
    assert!(max_rel < 0.06, "contiguous rel err {max_rel}");
}

#[test]
fn e2e_mqa_logits() {
    let (dev, stream) = test_dev();
    let num_tokens = 32u32;
    let num_kv = 512u32;
    let heads = 64u32;
    let head_dim = 128u32;
    let q_rows = num_tokens * heads;

    let xq = rand_data((q_rows * head_dim) as usize, 11);
    let xkv = rand_data((num_kv * head_dim) as usize, 12);
    let (q_data, q_sf_h) = quantize_host(&xq, head_dim, 32, false);
    let (kv_data, kv_sf_h) = quantize_host(&xkv, head_dim, 32, false);
    let q_dev = alloc_and_upload(&dev, &q_data, stream.raw()).unwrap();
    let kv_dev = alloc_and_upload(&dev, &kv_data, stream.raw()).unwrap();
    let q_sf = transform_sf(&dev, &stream, &q_sf_h, q_rows, SfGran::G32).unwrap();
    let kv_sf = transform_sf(&dev, &stream, &kv_sf_h, num_kv, SfGran::G32).unwrap();

    let host_bf16 = |v: f32| -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounding = 0x7fff + lsb;
        (((bits + rounding) >> 16) as u16) & 0x7fff | ((bits >> 31) as u16) << 15
    };
    let weights: Vec<u16> = (0..num_tokens * heads)
        .map(|i| host_bf16(0.25 + (i % 5) as f32 * 0.1))
        .collect();
    let w_dev = alloc_and_upload(&dev, &weights, stream.raw()).unwrap();
    let ks = vec![0u32; num_tokens as usize];
    let ke = vec![num_kv; num_tokens as usize];
    let ks_dev = alloc_and_upload(&dev, &ks, stream.raw()).unwrap();
    let ke_dev = alloc_and_upload(&dev, &ke, stream.raw()).unwrap();

    let q = Operand { dtype: Dtype::Fp8, major: Major::K, rows: q_rows, k: head_dim,
                      outer_stride: head_dim, sf: None, data: q_dev };
    let kv = Operand { dtype: Dtype::Fp8, major: Major::K, rows: num_kv, k: head_dim,
                       outer_stride: head_dim, sf: None, data: kv_dev };
    let mut logits = DevBuffer::alloc_zeros(&dev, (num_tokens * num_kv * 2) as usize).unwrap();

    mqa_logits(&dev, &stream, &q, &q_sf, &kv, &kv_sf, &w_dev, &ks_dev, &ke_dev,
               num_tokens, num_kv, heads, head_dim, &mut logits, num_kv).expect("mqa");

    let got: Vec<u16> = download(&dev, &logits, stream.raw()).unwrap();
    let bf = |h: u16| f32::from_bits(((h as u32) << 16) & 0xffff0000);
    let mut max_err = 0f32;
    for t in 0..num_tokens as usize {
        for j in 0..num_kv as usize {
            let mut acc = 0f64;
            for h in 0..heads as usize {
                let mut dot = 0f64;
                for d in 0..head_dim as usize {
                    let sq = q_sf_h[(t * heads as usize + h) * (head_dim as usize / 32) + d / 32] as f64;
                    let skv = kv_sf_h[j * (head_dim as usize / 32) + d / 32] as f64;
                    dot += e4m3_dequant(q_data[(t * heads as usize + h) * head_dim as usize + d]) as f64 * sq
                         * e4m3_dequant(kv_data[j * head_dim as usize + d]) as f64 * skv;
                }
                acc += bf(weights[t * heads as usize + h]) as f64 * dot.max(0.0);
            }
            let v = bf(got[t * num_kv as usize + j]);
            max_err = max_err.max((v - acc as f32).abs());
        }
    }
    // BF16 accumulation tolerance.
    assert!(max_err < 1.5, "mqa abs err {max_err}");
}
