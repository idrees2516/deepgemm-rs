//! GPU end-to-end tests (run on a B200/SM100 GPU):
//! `LD_LIBRARY_PATH=... cargo test -p deepgemm --features e2e --release -- --nocapture`
//!
//! All CPU-side references come from `deepgemm::golden` — the same bit-exact
//! model that the GPU-less tests (`golden_tests.rs`) exercise — so a semantic
//! disagreement between the hardware path and the model is caught on whichever
//! plane runs first.

#![cfg(feature = "e2e")]

use deepgemm::golden;
use deepgemm::prelude::*;

fn test_dev() -> (std::sync::Arc<Device>, DevStream) {
    let dev = Device::new(0).expect("CUDA device");
    let stream = DevStream::new(&dev).expect("stream");
    (dev, stream)
}

/// Host quantization via the golden model; returns (data bytes, f32 scales
/// per [row, k/gran] slot) — scales as powers of two, ready for `transform_sf`.
fn quantize_host(x: &[f32], k: u32, gran: u32, fp4: bool) -> (Vec<u8>, Vec<f32>) {
    let dtype = if fp4 { Dtype::Fp4 } else { Dtype::Fp8 };
    let g = if gran >= 128 {
        SfGran::G128
    } else {
        SfGran::G32
    };
    let m = (x.len() as u32) / k;
    let q = golden::quant_mx_host(x, m, k, dtype, g);
    // Reconstruct the row-major f32 scale list from the packed golden words
    // (identical to what the golden model itself consumes).
    let num_slots = (k / g.k()) as usize;
    let mut sfs = Vec::with_capacity(m as usize * num_slots);
    for r in 0..m as usize {
        for gi in 0..num_slots {
            let w = q.sf[(gi / 4) * q.sf_cols as usize + r];
            let exp = ((w >> (8 * (gi % 4))) & 0xff) as u8;
            sfs.push(golden::ue8m0_decode(exp));
        }
    }
    (q.data, sfs)
}

/// bf16 raw-bits -> f32 (exact; bf16 ⊂ f32).
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(((h as u32) << 16) & 0xffff0000)
}

fn rand_data(n: usize, seed: u64) -> Vec<f32> {
    let mut x = 0x12345678u64.wrapping_add(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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

    for (dtype, gran) in [
        (Dtype::Fp8, SfGran::G32),
        (Dtype::Fp8, SfGran::G128),
        (Dtype::Fp4, SfGran::G32),
    ] {
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
            assert_eq!(
                got_data[i], host_data[i],
                "quant data mismatch at {i} for {dtype:?}"
            );
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

        let a = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: m,
            k,
            outer_stride: k,
            sf: Some(sfa),
            data: a_dev,
        };
        let b = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: n,
            k,
            outer_stride: k,
            sf: Some(sfb),
            data: b_dev,
        };
        let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
        let mut out = Output {
            dtype: Dtype::Bf16,
            rows: m,
            cols: n,
            stride: n,
            data: out_data,
        };

        fp8_gemm_nt(&dev, &stream, &a, &b, &mut out, false).expect("gemm");

        // Host reference: dequant + matmul.
        let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
        let mut max_rel_err = 0f32;
        for r in 0..m as usize {
            for c in 0..n as usize {
                let mut acc = 0f64;
                for i in 0..k as usize {
                    let sa =
                        a_sf_h[r * (k as usize / gran.k() as usize) + i / gran.k() as usize] as f64;
                    let sb =
                        b_sf_h[c * (k as usize / gran.k() as usize) + i / gran.k() as usize] as f64;
                    acc += golden::e4m3_decode(a_data[r * k as usize + i]) as f64
                        * sa
                        * golden::e4m3_decode(b_data[c * k as usize + i]) as f64
                        * sb;
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
        assert!(
            max_rel_err < 0.06,
            "fp8 nt gran {gran:?} rel err {max_rel_err}"
        );
    }
}

#[test]
fn e2e_fp4_gemm_nt_native() {
    let (dev, stream) = test_dev();
    let (m, n, _k) = (256u32, 256u32, 512u32); // K must be a multiple of 256 for MXF4
    let k = 512;
    let xa = rand_data((m * k) as usize, 3);
    let xb = rand_data((n * k) as usize, 4);

    let (a_data, a_sf_h) = quantize_host(&xa, k, 32, true);
    let (b_data, b_sf_h) = quantize_host(&xb, k, 32, true);
    let a_dev = alloc_and_upload(&dev, &a_data, stream.raw()).unwrap();
    let b_dev = alloc_and_upload(&dev, &b_data, stream.raw()).unwrap();
    let sfa = transform_sf(&dev, &stream, &a_sf_h, m, SfGran::G32).unwrap();
    let sfb = transform_sf(&dev, &stream, &b_sf_h, n, SfGran::G32).unwrap();

    let a = Operand {
        dtype: Dtype::Fp4,
        major: Major::K,
        rows: m,
        k,
        outer_stride: k,
        sf: Some(sfa),
        data: a_dev,
    };
    let b = Operand {
        dtype: Dtype::Fp4,
        major: Major::K,
        rows: n,
        k,
        outer_stride: k,
        sf: Some(sfb),
        data: b_dev,
    };
    let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
    let mut out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: out_data,
    };

    fp4_gemm_nt(&dev, &stream, &a, &b, &mut out, false).expect("fp4 gemm");

    let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
    let mut max_rel_err = 0f32;
    for r in 0..m as usize {
        for c in 0..n as usize {
            let mut acc = 0f64;
            for i in 0..k as usize {
                let sa = a_sf_h[r * (k as usize / 32) + i / 32] as f64;
                let sb = b_sf_h[c * (k as usize / 32) + i / 32] as f64;
                let av = golden::e2m1_decode(
                    a_data[(r * k as usize + i) / 2] >> (4 * ((r * k as usize + i) % 2)) & 0xf,
                );
                let bv = golden::e2m1_decode(
                    b_data[(c * k as usize + i) / 2] >> (4 * ((c * k as usize + i) % 2)) & 0xf,
                );
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

    let host_bf16 = |v: f32| golden::f32_to_bf16_bits(v);
    let bf16_f = |h: u16| bf16_to_f32(h);

    let a_h: Vec<u16> = xa.iter().map(|&v| host_bf16(v)).collect();
    let b_h: Vec<u16> = xb.iter().map(|&v| host_bf16(v)).collect();
    let a_dev = alloc_and_upload(&dev, &a_h, stream.raw()).unwrap();
    let b_dev = alloc_and_upload(&dev, &b_h, stream.raw()).unwrap();

    let a = Operand {
        dtype: Dtype::Bf16,
        major: Major::K,
        rows: m,
        k,
        outer_stride: k,
        sf: None,
        data: a_dev,
    };
    let b = Operand {
        dtype: Dtype::Bf16,
        major: Major::K,
        rows: n,
        k,
        outer_stride: k,
        sf: None,
        data: b_dev,
    };
    let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
    let mut out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: out_data,
    };

    bf16_gemm_nt(&dev, &stream, &a, &b, &mut out, false).expect("bf16 gemm");

    let got: Vec<u16> = download(&dev, &out.data, stream.raw()).unwrap();
    let mut max_err = 0f32;
    for r in 0..m as usize {
        for c in 0..n as usize {
            let mut acc = 0f64;
            for i in 0..k as usize {
                acc +=
                    bf16_f(a_h[r * k as usize + i]) as f64 * bf16_f(b_h[c * k as usize + i]) as f64;
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

    let a = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: m,
        k,
        outer_stride: k,
        sf: Some(sfa),
        data: a_dev,
    };
    let b = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: n,
        k,
        outer_stride: k,
        sf: Some(sfb),
        data: b_dev,
    };
    let out_data = DevBuffer::alloc_zeros(&dev, (groups * m * n * 2) as usize).unwrap();
    let mut out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: out_data,
    };

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
                    acc += golden::e4m3_decode(a_data[(g * m as usize + r) * k as usize + i])
                        as f64
                        * sa
                        * golden::e4m3_decode(b_data[(g * n as usize + c) * k as usize + i]) as f64
                        * sb;
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

    let a = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: m,
        k,
        outer_stride: k,
        sf: Some(sfa),
        data: a_dev,
    };
    let b = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: n,
        k,
        outer_stride: k,
        sf: Some(sfb),
        data: b_dev,
    };
    let out_data = DevBuffer::alloc_zeros(&dev, (m * n * 2) as usize).unwrap();
    let mut out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: out_data,
    };

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
                acc += golden::e4m3_decode(a_data[r * k as usize + i]) as f64
                    * sa
                    * golden::e4m3_decode(b_data[(g * n as usize + c) * k as usize + i]) as f64
                    * sb;
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

    let host_bf16 = |v: f32| golden::f32_to_bf16_bits(v);
    let weights: Vec<u16> = (0..num_tokens * heads)
        .map(|i| host_bf16(0.25 + (i % 5) as f32 * 0.1))
        .collect();
    let w_dev = alloc_and_upload(&dev, &weights, stream.raw()).unwrap();
    let ks = vec![0u32; num_tokens as usize];
    let ke = vec![num_kv; num_tokens as usize];
    let ks_dev = alloc_and_upload(&dev, &ks, stream.raw()).unwrap();
    let ke_dev = alloc_and_upload(&dev, &ke, stream.raw()).unwrap();

    let q = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: q_rows,
        k: head_dim,
        outer_stride: head_dim,
        sf: None,
        data: q_dev,
    };
    let kv = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: num_kv,
        k: head_dim,
        outer_stride: head_dim,
        sf: None,
        data: kv_dev,
    };
    let mut logits = DevBuffer::alloc_zeros(&dev, (num_tokens * num_kv * 2) as usize).unwrap();

    mqa_logits(
        &dev,
        &stream,
        &q,
        &q_sf,
        &kv,
        &kv_sf,
        &w_dev,
        &ks_dev,
        &ke_dev,
        num_tokens,
        num_kv,
        heads,
        head_dim,
        &mut logits,
        num_kv,
    )
    .expect("mqa");

    let got: Vec<u16> = download(&dev, &logits, stream.raw()).unwrap();
    let bf = |h: u16| bf16_to_f32(h);
    let mut max_err = 0f32;
    for t in 0..num_tokens as usize {
        for j in 0..num_kv as usize {
            let mut acc = 0f64;
            for h in 0..heads as usize {
                let mut dot = 0f64;
                for d in 0..head_dim as usize {
                    let sq =
                        q_sf_h[(t * heads as usize + h) * (head_dim as usize / 32) + d / 32] as f64;
                    let skv = kv_sf_h[j * (head_dim as usize / 32) + d / 32] as f64;
                    dot += golden::e4m3_decode(
                        q_data[(t * heads as usize + h) * head_dim as usize + d],
                    ) as f64
                        * sq
                        * golden::e4m3_decode(kv_data[j * head_dim as usize + d]) as f64
                        * skv;
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

#[test]
fn e2e_transform_sf_bit_exact_vs_golden() {
    // The GPU transform_sf kernel must produce byte-identical packed words to
    // the CPU golden model — the strongest possible cross-check of the layout
    // path that the tcgen05 SF/UTCCP machinery consumes.
    let (dev, stream) = test_dev();
    for (mn, sf_k) in [
        (33u32, 7u32),
        (64u32, 16u32),
        (128u32, 4u32),
        (257u32, 32u32),
    ] {
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as i32 % 21) - 10
        };
        let sf: Vec<f32> = (0..mn * sf_k).map(|_| (2.0f32).powi(next())).collect();
        let got = transform_sf(&dev, &stream, &sf, mn, SfGran::G32).expect("transform");
        let got_words: Vec<u32> = download(&dev, &got.buf, stream.raw()).unwrap();
        let want = golden::transform_sf_host(&sf, mn, SfGran::G32);
        assert_eq!(
            got_words.len(),
            want.len(),
            "length mismatch (mn={mn}, sf_k={sf_k})"
        );
        let mut bad = 0usize;
        for (i, (g, w)) in got_words.iter().zip(want.iter()).enumerate() {
            if g != w {
                if bad < 5 {
                    eprintln!("  word {i}: gpu {g:#010x} golden {w:#010x} (mn={mn} sf_k={sf_k})");
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad, 0,
            "transform_sf: {bad} words differ from golden (mn={mn}, sf_k={sf_k})"
        );
    }
}

// ===========================================================================
// SM90 (Hopper) e2e — requires --features e2e on an SM90 device.
// ===========================================================================
#[cfg(test)]
mod sm90_e2e {
    use deepgemm::device::{alloc_and_upload, DevBuffer};
    use deepgemm::prelude::*;
    use deepgemm::sm90;
    use deepgemm::types::{Dtype, Major, Operand, Output};

    fn dev_stream() -> (std::sync::Arc<Device>, DevStream) {
        let dev = Device::new(0).expect("CUDA device");
        let stream = DevStream::new(&dev).expect("stream");
        (dev, stream)
    }

    fn dl_f32(dev: &Device, buf: &DevBuffer, stream: &DevStream) -> Vec<f32> {
        deepgemm::device::download(dev, buf, stream.raw()).unwrap()
    }

    /// Decode e4m3 byte (round-trip of the bench data: we upload raw bytes, so
    /// reference uses the standard e4m3 decode).
    fn e4m3(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1f32 } else { 1f32 };
        let exp = ((b >> 3) & 0xf) as i32;
        let man = (b & 7) as f32;
        match (exp, b & 7) {
            (0, 0) => 0.0,
            (0, _) => sign * 2f32.powi(-6) * (1.0 + man / 8.0),
            (15, 0) => sign * 448.0,
            (15, _) => 448.0 * sign, // saturating payload; bench data avoids it
            _ => sign * 2f32.powi(exp - 7) * (1.0 + man / 8.0),
        }
    }

    #[test]
    fn sm90_fp8_gemm_nt_matches_reference() {
        let (dev, stream) = dev_stream();
        if !matches!(dev.arch, deepgemm::device::Arch::Sm90) {
            eprintln!("skip: not SM90");
            return;
        }
        let (m, n, k) = (256u32, 128u32, 512u32);
        // Random e4m3 codes (avoid NaN/inf patterns 0x7f/0xff).
        let a_host: Vec<u8> = (0..(m * k))
            .map(|i| ((i * 2654435761 + 7) % 0x70) as u8)
            .collect();
        let b_host: Vec<u8> = (0..(n * k))
            .map(|i| ((i * 40503 + 3) % 0x70) as u8)
            .collect();
        let sfa_host: Vec<f32> = (0..(m * k / 128))
            .map(|i| 2f32.powi(-3 + (i % 7) as i32))
            .collect();
        let sfb_host: Vec<f32> = (0..(n * k / 128))
            .map(|i| 2f32.powi(-4 + (i % 5) as i32))
            .collect();

        let a = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: m,
            k,
            outer_stride: k,
            sf: None,
            data: alloc_and_upload(&dev, &a_host, stream.raw()).unwrap(),
        };
        let b = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: n,
            k,
            outer_stride: k,
            sf: None,
            data: alloc_and_upload(&dev, &b_host, stream.raw()).unwrap(),
        };
        let sfa = sm90::sf_fp32_from_host(&dev, &stream, &sfa_host, m, k / 128).unwrap();
        let sfb = sm90::sf_fp32_from_host(&dev, &stream, &sfb_host, n, k / 128).unwrap();
        let out_data = DevBuffer::alloc(&dev, (m * n * 4) as usize).unwrap();
        let mut out = Output {
            dtype: Dtype::F32,
            rows: m,
            cols: n,
            stride: n,
            data: out_data,
        };
        sm90::fp8_gemm_nt(&dev, &stream, &a, &sfa, &b, &sfb, &mut out).unwrap();
        stream.sync().unwrap();

        let got: Vec<f32> = dl_f32(&dev, &out.data, &stream);
        let mut max_err = 0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += e4m3(a_host[(i * k + kk) as usize]) as f64
                        * e4m3(b_host[(j * k + kk) as usize]) as f64
                        * sfa_host[(i * k / 128 + kk / 128) as usize] as f64
                        * sfb_host[(j * k / 128 + kk / 128) as usize] as f64;
                }
                let want = acc as f32;
                let err = (got[(i * n + j) as usize] - want).abs();
                assert!(
                    err < 1e-3,
                    "({i},{j}) got {} want {want}",
                    got[(i * n + j) as usize]
                );
                max_err = max_err.max(err as f64);
            }
        }
        eprintln!("sm90_fp8_nt: max_err {max_err}");
    }

    #[test]
    fn sm90_fp8_gemm_kk_matches_reference() {
        let (dev, stream) = dev_stream();
        if !matches!(dev.arch, deepgemm::device::Arch::Sm90) {
            eprintln!("skip: not SM90");
            return;
        }
        let (m, n) = (128u32, 64u32);
        let ks = [256u32, 128u32, 384u32];
        let k: u32 = ks.iter().sum();
        // Stacked tiles: group g is [mn, ks_g] K-major at offset k_start*mn.
        let a_host: Vec<u8> = (0..(m as usize * k as usize))
            .map(|i| ((i * 2654435761 + 11) % 0x70) as u8)
            .collect();
        let b_host: Vec<u8> = (0..(n as usize * k as usize))
            .map(|i| ((i * 40503 + 5) % 0x70) as u8)
            .collect();
        // NOTE: the flat buffer IS the stacked layout when built group-by-group
        // with row stride ks_g — a per-group [mn, ks_g] concat equals a plain
        // [mn, sum] buffer only when every ks_g equals... it does NOT. Build it
        // properly: for each group, m rows of ks_g.
        let mut a_stack: Vec<u8> = Vec::with_capacity(a_host.len());
        let mut b_stack: Vec<u8> = Vec::with_capacity(b_host.len());
        let mut ref_a: Vec<Vec<f64>> = vec![vec![0.0; k as usize]; m as usize];
        let mut ref_b: Vec<Vec<f64>> = vec![vec![0.0; k as usize]; n as usize];
        let mut k_start = 0u32;
        for &g in &ks {
            for r in 0..m {
                for kk in 0..g {
                    let v = ((r * 31 + kk * 17 + k_start) % 0x70) as u8;
                    a_stack.push(v);
                    ref_a[r as usize][(k_start + kk) as usize] = e4m3(v) as f64;
                }
            }
            for r in 0..n {
                for kk in 0..g {
                    let v = ((r * 13 + kk * 7 + k_start) % 0x70) as u8;
                    b_stack.push(v);
                    ref_b[r as usize][(k_start + kk) as usize] = e4m3(v) as f64;
                }
            }
            k_start += g;
        }
        let sfa_host: Vec<f32> = (0..(m * k / 128))
            .map(|i| 2f32.powi(-3 + (i % 7) as i32))
            .collect();
        let sfb_host: Vec<f32> = (0..(n * k / 128))
            .map(|i| 2f32.powi(-4 + (i % 5) as i32))
            .collect();

        let a = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: m,
            k,
            outer_stride: k,
            sf: None,
            data: alloc_and_upload(&dev, &a_stack, stream.raw()).unwrap(),
        };
        let b = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: n,
            k,
            outer_stride: k,
            sf: None,
            data: alloc_and_upload(&dev, &b_stack, stream.raw()).unwrap(),
        };
        let sfa = sm90::sf_fp32_from_host(&dev, &stream, &sfa_host, m, k / 128).unwrap();
        let sfb = sm90::sf_fp32_from_host(&dev, &stream, &sfb_host, n, k / 128).unwrap();
        let out_data = DevBuffer::alloc(&dev, (ks.len() as u32 * m * n * 4) as usize).unwrap();
        let mut out = Output {
            dtype: Dtype::F32,
            rows: m,
            cols: n,
            stride: n,
            data: out_data,
        };
        sm90::fp8_gemm_kk(&dev, &stream, &a, &sfa, &b, &sfb, &ks, &mut out).unwrap();
        stream.sync().unwrap();
        let got: Vec<f32> = dl_f32(&dev, &out.data, &stream);

        let mut max_err = 0f64;
        for (g, _ks) in ks.iter().enumerate() {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += ref_a[i as usize][kk as usize]
                            * ref_b[j as usize][kk as usize]
                            * sfa_host[(i * k / 128 + kk / 128) as usize] as f64
                            * sfb_host[(j * k / 128 + kk / 128) as usize] as f64;
                    }
                    let want = acc as f32;
                    let idx = (g as u32 * m * n + i * n + j) as usize;
                    let err = (got[idx] - want).abs();
                    assert!(err < 1e-3, "g{g} ({i},{j}) got {} want {want}", got[idx]);
                    max_err = max_err.max(err as f64);
                }
            }
        }
        eprintln!("sm90_fp8_kk: max_err {max_err}");
    }

    #[test]
    fn sm90_bf16_gemm_nt_matches_reference() {
        let (dev, stream) = dev_stream();
        if !matches!(dev.arch, deepgemm::device::Arch::Sm90) {
            eprintln!("skip: not SM90");
            return;
        }
        let (m, n, k) = (192u32, 96u32, 384u32);
        let a_host: Vec<u16> = (0..(m * k))
            .map(|i| deepgemm::golden::f32_to_bf16_bits(((i % 97) as f32) * 0.25 - 8.0))
            .collect();
        let b_host: Vec<u16> = (0..(n * k))
            .map(|i| deepgemm::golden::f32_to_bf16_bits(((i % 89) as f32) * 0.125 - 4.0))
            .collect();
        let a = Operand {
            dtype: Dtype::Bf16,
            major: Major::K,
            rows: m,
            k,
            outer_stride: k,
            sf: None,
            data: alloc_and_upload(&dev, &a_host, stream.raw()).unwrap(),
        };
        let b = Operand {
            dtype: Dtype::Bf16,
            major: Major::K,
            rows: n,
            k,
            outer_stride: k,
            sf: None,
            data: alloc_and_upload(&dev, &b_host, stream.raw()).unwrap(),
        };
        let out_data = DevBuffer::alloc(&dev, (m * n * 4) as usize).unwrap();
        let mut out = Output {
            dtype: Dtype::F32,
            rows: m,
            cols: n,
            stride: n,
            data: out_data,
        };
        sm90::bf16_gemm_nt(&dev, &stream, &a, &b, &mut out, false).unwrap();
        stream.sync().unwrap();
        let got: Vec<f32> = dl_f32(&dev, &out.data, &stream);

        let bf16 = |h: u16| f32::from_bits(((h as u32) << 16));
        let mut max_err = 0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += bf16(a_host[(i * k + kk) as usize]) as f64
                        * bf16(b_host[(j * k + kk) as usize]) as f64;
                }
                let want = acc as f32;
                let err = (got[(i * n + j) as usize] - want).abs();
                assert!(
                    err < 1e-2,
                    "({i},{j}) got {} want {want}",
                    got[(i * n + j) as usize]
                );
                max_err = max_err.max(err as f64);
            }
        }
        eprintln!("sm90_bf16_nt(fp32 out): max_err {max_err}");

        // BF16 output flavor + accumulation.
        let out2 = DevBuffer::alloc(&dev, (m * n * 2) as usize).unwrap();
        let mut out_bf16 = Output {
            dtype: Dtype::Bf16,
            rows: m,
            cols: n,
            stride: n,
            data: out2,
        };
        sm90::bf16_gemm_nt(&dev, &stream, &a, &b, &mut out_bf16, true).unwrap();
        stream.sync().unwrap();
        let got2: Vec<u16> =
            deepgemm::device::download(&dev, &out_bf16.data, stream.raw()).unwrap();
        // accumulate=true doubles the reference.
        let (i, j) = (3u32, 5u32);
        let mut acc = 0f64;
        for kk in 0..k {
            acc += bf16(a_host[(i * k + kk) as usize]) as f64
                * bf16(b_host[(j * k + kk) as usize]) as f64;
        }
        let want = (2.0 * acc) as f32;
        let gotv = f32::from_bits(((got2[(i * n + j) as usize] as u32) << 16));
        assert!(
            (gotv - want).abs() < 0.1,
            "bf16-out ({i},{j}) got {gotv} want {want}"
        );
    }

    #[test]
    fn sm90_transpose_sf_roundtrip() {
        let (dev, stream) = dev_stream();
        if !matches!(dev.arch, deepgemm::device::Arch::Sm90) {
            eprintln!("skip: not SM90");
            return;
        }
        let (mn, k_blocks) = (130u32, 17u32);
        let src: Vec<f32> = (0..(mn * k_blocks)).map(|i| i as f32 * 0.5 - 3.0).collect();
        let buf = alloc_and_upload(&dev, &src, stream.raw()).unwrap();
        let sf = sm90::transpose_sf_fp32(&dev, &stream, &buf, mn, k_blocks).unwrap();
        stream.sync().unwrap();
        let got: Vec<f32> = dl_f32(&dev, &sf.buf, &stream);
        let ta = deepgemm::heuristics::tma_aligned_size(mn, 4);
        for kb in 0..k_blocks {
            for idx in 0..ta {
                let want = if idx < mn {
                    src[(idx * k_blocks + kb) as usize]
                } else {
                    0.0
                };
                assert_eq!(got[(kb * ta + idx) as usize], want, "({kb},{idx})");
            }
        }
    }
}

// ===========================================================================
// Megakernel e2e (B200/B300; --features e2e)
// ===========================================================================
mod mega_e2e {
    use super::*;
    use deepgemm::api::{self, MegaMoEWeights, RoutingSpec};
    use deepgemm::golden;
    use deepgemm::types::{Dtype, Major, SfGran};

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    /// Transpose the golden SF words into the per-row-major layout the mega
    /// kernels consume: [rows][k/128] int32 words (4 UE8M0 bytes each).
    fn sf_row_major(x: &[f32], rows: u32, k: u32, gran: u32) -> Vec<u32> {
        let q = golden::quant_mx_host(
            x,
            rows,
            k,
            Dtype::Fp8,
            if gran >= 128 {
                SfGran::G128
            } else {
                SfGran::G32
            },
        );
        // golden layout: word-major [word][row]; we need row-major [row][word].
        let words = (k / 128) as usize;
        let mut out = vec![0u32; rows as usize * words];
        for r in 0..rows as usize {
            for w in 0..words {
                out[r * words + w] = q.sf[w * q.sf_cols as usize + r];
            }
        }
        out
    }

    /// Quantize + interleave + SF-transpose the MoE weights for the kernel.
    fn moe_weights(
        dev: &Device,
        stream: &DevStream,
        w: &[f32],
        e: u32,
        n: u32,
        k: u32,
    ) -> anyhow_of(MegaMoEWeights) {
        let data = DevBuffer::from_bytes(dev, &quantize_host(w, k, 32, false).0, stream.raw())?;
        // SF: row-major words, then UTCCP-transposed on device.
        let sf_words = sf_row_major(w, e * n, k, 32);
        let sf = DevBuffer::from_bytes(
            dev,
            unsafe {
                std::slice::from_raw_parts(sf_words.as_ptr() as *const u8, sf_words.len() * 4)
            },
            stream.raw(),
        )?;
        // interleave gate/up along N + transpose SF pages.
        let data = api::interleave_weights(dev, stream, &data, e, n, k)?;
        let sf = api::transpose_sf_utccp(dev, stream, &sf, e, n, k)?;
        Ok(MegaMoEWeights {
            data,
            sf,
            n,
            k,
            is_fp4: false,
        })
    }

    type TmResult<T> = Result<T, deepgemm::error::DgError>;
    fn anyhow_of<T>(t: T) -> T {
        t
    }

    #[test]
    fn mega_moe_e2e_fp8() {
        let (dev, stream) = test_dev();
        let (e, i, h, t, topk) = (8u32, 256u32, 512u32, 256u32, 4u32);
        let mut seed = 42u64;
        let x_f: Vec<f32> = (0..t * h)
            .map(|_| (lcg(&mut seed) % 200) as f32 / 100.0 - 1.0)
            .collect();
        let w13_f: Vec<f32> = (0..e * 2 * i * h)
            .map(|_| (lcg(&mut seed) % 200) as f32 / 100.0 - 1.0)
            .collect();
        let w2_f: Vec<f32> = (0..e * h * i)
            .map(|_| (lcg(&mut seed) % 200) as f32 / 100.0 - 1.0)
            .collect();
        let topk_idx: Vec<i64> = (0..t * topk).map(|_| (lcg(&mut seed) % e) as i64).collect();
        let weights: Vec<f32> = vec![0.7; (t * topk) as usize];

        let w13 = moe_weights(&dev, &stream, &w13_f, e, 2 * i, h).unwrap();
        let w2 = moe_weights(&dev, &stream, &w2_f, e, h, i).unwrap();

        // Sym buffer + input fills.
        let (total, off) =
            deepgemm::mega_layout::mega_moe_buffer_layout(1, e, t, topk, h, i, 0, dev.num_sms)
                .unwrap();
        let sym = DevBuffer::alloc(&dev, total as usize).unwrap();
        let base = sym.ptr as u64;
        let mut put = |o: u64, bytes: &[u8]| -> TmResult<()> {
            let b = unsafe { DevBuffer::from_bytes(&dev, bytes, stream.raw())? };
            // copy into the sym view via device-side memcpy? simplest: fill
            // through a staging buffer + d2d copy.
            let _ = b;
            Ok(())
        };
        let _ = &mut put;
        // Fill the input regions with a single host->device upload each.
        let xq = quantize_host(&x_f, h, 32, false).0;
        let x_sf = sf_row_major(&x_f, t, h, 32);
        let x_sf_b =
            unsafe { std::slice::from_raw_parts(x_sf.as_ptr() as *const u8, x_sf.len() * 4) };
        let topk_b: Vec<u8> = topk_idx.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let w_b: Vec<u8> = weights.iter().flat_map(|v| v.to_ne_bytes()).collect();
        // Direct placement: build one host image of the whole input span.
        let span = off.l1_token - off.input_token;
        let mut img = vec![0u8; span as usize];
        img[0..xq.len()].copy_from_slice(&xq);
        let sf_off = (off.input_sf - off.input_token) as usize;
        img[sf_off..sf_off + x_sf_b.len()].copy_from_slice(x_sf_b);
        let tk_off = (off.input_topk_idx - off.input_token) as usize;
        img[tk_off..tk_off + topk_b.len()].copy_from_slice(&topk_b);
        let w_off = (off.input_topk_weights - off.input_token) as usize;
        img[w_off..w_off + w_b.len()].copy_from_slice(&w_b);
        let _ = put;
        // Upload: allocate a temporary then memcpy_d2d? Use direct from_host
        // into the view (DevBuffer::from_bytes allocates; instead reuse
        // sym + offset with copy_from_host on a view).
        let view = |o: u64, len: usize| DevBuffer {
            ptr: (base + o) as _,
            len,
        };
        view(off.input_token, xq.len())
            .copy_from_host(&dev, &img[0..xq.len()], stream.raw())
            .unwrap();
        view(off.input_sf, x_sf_b.len())
            .copy_from_host(&dev, x_sf_b, stream.raw())
            .unwrap();
        view(off.input_topk_idx, topk_b.len())
            .copy_from_host(&dev, &topk_b, stream.raw())
            .unwrap();
        view(off.input_topk_weights, w_b.len())
            .copy_from_host(&dev, &w_b, stream.raw())
            .unwrap();

        let mut y = DevBuffer::alloc(&dev, (t * h * 2) as usize).unwrap();
        api::mega_moe(
            &dev,
            &stream,
            &mut y,
            &w13,
            &w2,
            None,
            None,
            &sym,
            &[sym.ptr as u64],
            0,
            t,
            0,
            t,
            topk,
            h,
            i,
            Some(88.0),
            true,
            None,
        )
        .unwrap();
        stream.sync().unwrap();

        let mut out = vec![0u8; (t * h * 2) as usize];
        y.copy_to_host(&dev, &mut out, stream.raw()).unwrap();
        let bits: Vec<u16> = out
            .chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert!(bits.iter().any(|&b| b != 0), "all-zero mega_moe output");
        // Fraction of NaNs must be zero.
        let nans = bits.iter().filter(|&&b| (b & 0x7f80) == 0x7f80).count();
        assert_eq!(nans, 0, "NaNs in mega_moe output");
    }

    #[test]
    fn mega_gate_e2e_fp8() {
        let (dev, stream) = test_dev();
        let (t, h, e, topk) = (512u32, 512u32, 128u32, 4u32);
        let mut seed = 7u64;
        let x_f: Vec<f32> = (0..t * h)
            .map(|_| (lcg(&mut seed) % 200) as f32 / 100.0 - 1.0)
            .collect();
        let w_f: Vec<f32> = (0..e * h)
            .map(|_| (lcg(&mut seed) % 200) as f32 / 100.0 - 1.0)
            .collect();
        let bias: Vec<f32> = (0..e).map(|i| i as f32 * 0.01 - 0.5).collect();

        let xq = quantize_host(&x_f, h, 32, false);
        let wq = quantize_host(&w_f, h, 32, false);
        let x_sf = sf_row_major(&x_f, t, h, 32);
        let w_sf = sf_row_major(&w_f, e, h, 32);
        let dev_x = DevBuffer::from_bytes(&dev, &xq.0, stream.raw()).unwrap();
        let dev_xsf = DevBuffer::from_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(x_sf.as_ptr() as *const u8, x_sf.len() * 4) },
            stream.raw(),
        )
        .unwrap();
        let dev_w = DevBuffer::from_bytes(&dev, &wq.0, stream.raw()).unwrap();
        let dev_wsf = DevBuffer::from_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(w_sf.as_ptr() as *const u8, w_sf.len() * 4) },
            stream.raw(),
        )
        .unwrap();
        let dev_bias = DevBuffer::from_bytes(
            &dev,
            unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const u8, bias.len() * 4) },
            stream.raw(),
        )
        .unwrap();

        let topk_idx = DevBuffer::alloc(&dev, (t * (topk + 1) * 8) as usize).unwrap();
        let topk_w = DevBuffer::alloc(&dev, (t * (topk + 1) * 4) as usize).unwrap();
        let routing = RoutingSpec {
            to_physical_map: None,
            logical_count: None,
            topk_idx,
            unmapped_topk_idx: None,
            topk_weights: topk_w,
            num_routed_experts: e,
            num_shared_experts: 1,
            num_duplicate_experts: 1,
            rank_idx: 0,
            routed_scaling_factor: 2.5,
        };
        let (_scratch, _bars) = api::mega_gate(
            &dev,
            &stream,
            &dev_x,
            &dev_xsf,
            &dev_w,
            &dev_wsf,
            Some(&dev_bias),
            None,
            None,
            None,
            None,
            None,
            &routing,
            t,
            h,
            topk,
            false,
        )
        .unwrap();
        stream.sync().unwrap();

        let mut idx_b = vec![0u8; (t * (topk + 1) * 8) as usize];
        routing
            .topk_idx
            .copy_to_host(&dev, &mut idx_b, stream.raw())
            .unwrap();
        let mut w_b = vec![0u8; (t * (topk + 1) * 4) as usize];
        routing
            .topk_weights
            .copy_to_host(&dev, &mut w_b, stream.raw())
            .unwrap();
        let idx: Vec<i64> = idx_b
            .chunks(8)
            .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let wt: Vec<f32> = w_b
            .chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        for ti in 0..t as usize {
            for j in 0..topk as usize {
                let pick = idx[ti * (topk + 1) as usize + j];
                assert!(pick >= 0 && pick < e as i64, "invalid pick {pick}");
            }
            let sum: f32 = (0..topk as usize)
                .map(|j| wt[ti * (topk + 1) as usize + j])
                .sum();
            assert!((sum - 2.5).abs() < 0.15, "weights sum {sum}");
        }
    }

    #[test]
    fn mega_mhc_e2e_shifted_fp8() {
        let (dev, stream) = test_dev();
        let (t, h) = (512u32, 512u32);
        let mut seed = 9u64;
        let r = |s: &mut u64, n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (lcg(s) % 200) as f32 / 100.0 - 1.0)
                .collect()
        };
        let x_b: Vec<u16> = r(&mut seed, (t * h) as usize)
            .iter()
            .map(|&v| golden::f32_to_bf16_bits(v))
            .collect();
        let res_b: Vec<u16> = r(&mut seed, (t * 4 * h) as usize)
            .iter()
            .map(|&v| golden::f32_to_bf16_bits(v))
            .collect();
        let post = r(&mut seed, (t * 4) as usize);
        let comb = r(&mut seed, (t * 16) as usize);
        let prev = r(&mut seed, (t * 4) as usize);
        let f_n = r(&mut seed, (24 * 4 * h) as usize);
        let scales = vec![0.5f32, 1.0, 1.5];
        let bases: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 24.0).collect();
        let nw: Vec<u16> = (0..h).map(|_| golden::f32_to_bf16_bits(0.9)).collect();

        let up = |b: &[u8]| DevBuffer::from_bytes(&dev, b, stream.raw()).unwrap();
        let dev_x =
            up(unsafe { std::slice::from_raw_parts(x_b.as_ptr() as *const u8, x_b.len() * 2) });
        let dev_res =
            up(unsafe { std::slice::from_raw_parts(res_b.as_ptr() as *const u8, res_b.len() * 2) });
        let dev_post =
            up(unsafe { std::slice::from_raw_parts(post.as_ptr() as *const u8, post.len() * 4) });
        let dev_comb =
            up(unsafe { std::slice::from_raw_parts(comb.as_ptr() as *const u8, comb.len() * 4) });
        let dev_prev =
            up(unsafe { std::slice::from_raw_parts(prev.as_ptr() as *const u8, prev.len() * 4) });
        let dev_fn =
            up(unsafe { std::slice::from_raw_parts(f_n.as_ptr() as *const u8, f_n.len() * 4) });
        let dev_scales =
            up(unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, 12) });
        let dev_bases =
            up(unsafe { std::slice::from_raw_parts(bases.as_ptr() as *const u8, bases.len() * 4) });
        let dev_nw =
            up(unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 2) });

        let dev_new_res = DevBuffer::alloc(&dev, (t * 4 * h * 2) as usize).unwrap();
        let dev_npm = DevBuffer::alloc(&dev, (t * 4 * 4) as usize).unwrap();
        let dev_npost = DevBuffer::alloc(&dev, (t * 4 * 4) as usize).unwrap();
        let dev_ncomb = DevBuffer::alloc(&dev, (t * 16 * 4) as usize).unwrap();
        let mut y = DevBuffer::alloc(&dev, (t * h * 2) as usize).unwrap();
        let y_fp8 = DevBuffer::alloc(&dev, (t * h) as usize).unwrap();
        let y_psf = DevBuffer::alloc(&dev, (t * (h / 128) * 4) as usize).unwrap();
        let y_ssf = DevBuffer::alloc(&dev, (t * (h / 128) * 4) as usize).unwrap();

        api::mega_mhc(
            &dev,
            &stream,
            &dev_x,
            &dev_res,
            &dev_post,
            &dev_comb,
            Some(&dev_prev),
            &dev_fn,
            &dev_scales,
            &dev_bases,
            1e-6,
            0.1,
            1.0,
            1e-4,
            3,
            &dev_nw,
            1e-6,
            &dev_new_res,
            Some(&mut dev_npm),
            &dev_npost,
            &dev_ncomb,
            Some(&mut y),
            Some(&y_fp8),
            Some(&y_psf),
            Some(&y_ssf),
            (h / 128) as u32 * t,
            t,
            128,
            1.0,
            t,
            h,
        )
        .unwrap();
        stream.sync().unwrap();

        let mut yb = vec![0u8; (t * h * 2) as usize];
        y.copy_to_host(&dev, &mut yb, stream.raw()).unwrap();
        let bits: Vec<u16> = yb
            .chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert!(bits.iter().any(|&b| b != 0), "all-zero mhc output");
        let nans = bits.iter().filter(|&&b| (b & 0x7f80) == 0x7f80).count();
        assert_eq!(nans, 0, "NaNs in mhc output");
    }
}
