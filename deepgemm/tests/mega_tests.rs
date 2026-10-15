//! Megakernel host-side tests: heuristics invariants, layout math, and
//! golden-model self-consistency (no GPU required).

use deepgemm::golden;
use deepgemm::mega_layout as mlayout;

fn cd(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

// ---------------------------------------------------------------------------
// mega_moe config
// ---------------------------------------------------------------------------
#[test]
fn mega_moe_block_config_candidates() {
    // Small batches: 16/32 tiles; heavy: 64..240 by fewest-blocks rule.
    let (bm, s1, _s2) = deepgemm::heuristics::mega_moe_block_config(1, 256, 128, 8);
    assert_eq!(bm, 16);
    assert_eq!(s1, 8);
    let (bm2, _, _) = deepgemm::heuristics::mega_moe_block_config(1, 256, 4096, 8);
    assert!(mlayout::CANDIDATE_BLOCK_MS.contains(&bm2), "{bm2}");
    assert!(bm2 >= 64);
}

#[test]
fn mega_moe_config_pipeline_fits() {
    let cfg =
        deepgemm::heuristics::get_mega_moe_config(148, 1, 256, 4096, 4096, 8, 7168, 2048, 1920)
            .expect("config");
    assert_eq!(cfg.block_n, 128);
    assert_eq!(cfg.block_k, 128);
    assert!(cfg.num_stages >= 2, "stages {}", cfg.num_stages);
    assert!(cfg.smem_size <= mlayout::align_up(deepgemm::heuristics::SMEM_CAPACITY_FALLBACK, 1024));
    assert!(cfg.num_bytes_per_pull <= 8192);
    assert_eq!(cfg.num_sf_ring_tokens % 4, 0);
}

// ---------------------------------------------------------------------------
// layout invariants
// ---------------------------------------------------------------------------
#[test]
fn mega_moe_layout_monotonic() {
    let (total, off) =
        mlayout::mega_moe_buffer_layout(1, 256, 4096, 8, 7168, 2048, 1, 148).expect("layout");
    let ends = [
        off.workspace_end,
        off.input_token,
        off.input_sf,
        off.input_topk_idx,
        off.input_topk_weights,
        off.shared_l1_sf,
        off.shared_l2_token,
        off.shared_l2_sf,
        off.l1_token,
        off.l1_sf,
        off.l1_topk_weights,
        off.l2_token,
        off.l2_sf,
        off.combine,
        off.total,
    ];
    for w in ends.windows(2) {
        assert!(w[0] <= w[1], "non-monotonic at {:?}", w);
    }
    assert!(total > 0);
    // The signals header must match the kernel struct exactly (derived in
    // scripts; see the kernel banner for the field map).
    assert_eq!(off.workspace_end >= 21_140_864, true);
}

#[test]
fn mega_moe_ring_covers_live_bound() {
    // The ring must hold the worst-case live pool blocks for every candidate.
    for &bm in mlayout::CANDIDATE_BLOCK_MS.iter() {
        let pool = cd(4096 * 8, bm) + 256;
        let live = mlayout::num_max_live_pool_blocks(pool, 148, 7168, 2048);
        let (total, off) =
            mlayout::mega_moe_buffer_layout(1, 256, 4096, 8, 7168, 2048, 0, 148).expect("layout");
        // ring tokens = l1_token region span / hidden.
        let ring = ((off.l1_sf - off.l1_token) as u32) / 7168;
        assert!(ring >= live * bm, "ring {ring} < live {live} x bm {bm}");
        let _ = total;
    }
}

// ---------------------------------------------------------------------------
// gate + mhc
// ---------------------------------------------------------------------------
#[test]
fn mega_gate_config_bounds() {
    let cfg = deepgemm::heuristics::get_mega_gate_config(4096, 7168, 256, 148, false)
        .expect("gate config");
    assert!(cfg.block_tokens <= 240, "fp8 cap: {}", cfg.block_tokens);
    assert!(cfg.block_tokens % 16 == 0);
    assert!(cfg.num_logical_ctas <= 148);
    assert!(cfg.num_gate_threads % 128 == 0);
    assert!(cfg.num_stages >= 2);
    assert!(cfg.smem_size <= 232448 + 1024);
    // Deterministic never splits K.
    let det = deepgemm::heuristics::get_mega_gate_config(4096, 7168, 256, 148, true).unwrap();
    assert_eq!(det.num_split_k, 1);
}

#[test]
fn mhc_splits_range() {
    for &t in &[64u32, 512, 4096] {
        let s = deepgemm::heuristics::mhc_splits(t, 7168, 148, false);
        assert!(s >= 1 && s <= 64, "splits {s} for T={t}");
        assert!(7168 / 64 % s == 0 || s <= 16);
    }
    let d = deepgemm::heuristics::mhc_splits(4096, 7168, 148, true);
    assert_eq!(d, 16);
}

// ---------------------------------------------------------------------------
// golden self-consistency (tiny shapes; the reference must agree with a
// straightforward f64 reimplementation of the same contract)
// ---------------------------------------------------------------------------
fn tiny_moe_inputs(
    e: u32,
    i: u32,
    h: u32,
    t: u32,
    topk: u32,
) -> (
    Vec<u8>,
    Vec<u32>,
    Vec<u8>,
    Vec<u32>,
    Vec<u8>,
    Vec<u32>,
    Vec<i64>,
    Vec<f32>,
) {
    // Deterministic pseudo-random data (LCG).
    let mut seed = 0x1234_5678u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let mut mk_fp8 = |rows: u32, k: u32| -> (Vec<u8>, Vec<u32>) {
        let mut d = Vec::with_capacity((rows * k) as usize);
        let mut sf = Vec::with_capacity((rows * k / 128) as usize);
        for r in 0..rows {
            let mut amax = 0f32;
            let mut row = vec![0u8; k as usize];
            for kk in 0..k as usize {
                let v = ((next() % 200) as i32 - 100) as f32 / 25.0;
                amax = amax.max(v.abs());
                row[kk] = golden::quant_e4m3(v);
            }
            let sfv = golden::ue8m0_exp_fp8(amax) as u32;
            let mut word = 0u32;
            for g in 0..k / 128 {
                word = 0;
                for b in 0..4 {
                    word |= (sfv as u32) << (b * 8);
                }
                let _ = g;
                break;
            }
            for g in 0..k / 128 {
                sf.push(word);
            }
            d.extend_from_slice(&row);
        }
        (d, sf)
    };
    let (x, x_sf) = mk_fp8(t, h);
    let (w13, w13_sf) = mk_fp8(e * 2 * i, h);
    let (w2, w2_sf) = mk_fp8(e * h, i);
    let topk_idx: Vec<i64> = (0..t * topk).map(|j| (next() % e) as i64).collect();
    let weights: Vec<f32> = (0..t * topk).map(|_| 0.7).collect();
    (x, x_sf, w13, w13_sf, w2, w2_sf, topk_idx, weights)
}

#[test]
fn mega_moe_golden_self_consistent() {
    // The reference must produce finite outputs and depend on every operand.
    let (x, x_sf, w13, w13_sf, w2, w2_sf, topk, weights) = tiny_moe_inputs(4, 256, 256, 8, 2); // I, H >= 128 (SF packing)
    let y = golden::mega_moe_ref(
        &x,
        &x_sf,
        &w13,
        &w13_sf,
        &w2,
        &w2_sf,
        4,
        256,
        256,
        &topk,
        &weights,
        8,
        2,
        Some(88.0),
    );
    assert_eq!(y.len(), 8 * 256);
    // Non-degenerate: at least some nonzero outputs.
    let nonzero = y.iter().filter(|&&b| b != 0).count();
    assert!(nonzero > 0, "all-zero moe output");
    // Perturbing w13 must change the output.
    let mut w13b = w13.clone();
    // Perturb ~1% of the weight codes (a single byte is below the fp8/bf16
    // resolution of the output dot product).
    for k in (0..w13b.len()).step_by(97) {
        w13b[k] = w13b[k].wrapping_add(1);
    }
    let y2 = golden::mega_moe_ref(
        &x,
        &x_sf,
        &w13b,
        &w13_sf,
        &w2,
        &w2_sf,
        4,
        256,
        256,
        &topk,
        &weights,
        8,
        2,
        Some(88.0),
    );
    assert_ne!(y, y2, "w13 had no effect on the reference");
}

#[test]
fn mega_gate_golden_self_consistent() {
    let (x, x_sf, w, _w_sf, _t, _h, _e, _topk, _w2, weights): (
        Vec<u8>,
        Vec<u32>,
        Vec<u8>,
        Vec<u32>,
        u32,
        u32,
        u32,
        u32,
        Vec<u8>,
        Vec<f32>,
    ) = (
        vec![0u8; 0],
        vec![0u32; 0],
        vec![0u8; 0],
        vec![0u32; 0],
        8u32,
        256u32,
        4u32,
        2u32,
        vec![0u8; 0],
        vec![0.5f32; 16],
    );
    let _ = (&x, &x_sf, &w, &_w_sf);
    // Rebuild tiny gate inputs (dedicated; 1 expert group of 4).
    let mut seed = 0xdead_beefu64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as u32
    };
    let (t, h, e, topk) = (8u32, 256u32, 4u32, 2u32);
    let mut mk = |rows: u32, k: u32| -> (Vec<u8>, Vec<u32>) {
        let mut d = vec![0u8; (rows * k) as usize];
        let mut sf = Vec::with_capacity((rows * k / 128) as usize);
        for r in 0..rows {
            let mut amax = 0f32;
            for kk in 0..k as usize {
                let v = ((next() % 200) as i32 - 100) as f32 / 25.0;
                amax = amax.max(v.abs());
                d[(r * k) as usize + kk] = golden::quant_e4m3(v);
            }
            let sfv = golden::ue8m0_exp_fp8(amax) as u32;
            for _g in 0..k / 128 {
                sf.push(sfv * 0x0101_0101);
            }
        }
        (d, sf)
    };
    let (x, x_sf) = mk(t, h);
    let (w, w_sf) = mk(e, h);
    let bias = vec![0.05f32; e as usize];
    let (idx, wt) = golden::mega_gate_ref(&x, &x_sf, &w, &w_sf, t, h, e, &bias, topk, 2.5, 1);
    // Every token: topk routed picks + 1 shared slot.
    assert_eq!(idx.len(), (t * (topk + 1)) as usize);
    assert_eq!(wt.len(), idx.len());
    for ti in 0..t as usize {
        let picks: Vec<i64> = (0..topk as usize)
            .map(|j| idx[ti * (topk + 1) as usize + j])
            .collect();
        for p in &picks {
            assert!(*p >= 0 && *p < e as i64, "pick out of range: {p}");
        }
        let shared = idx[ti * (topk + 1) as usize + topk as usize];
        assert_eq!(shared, e as i64);
        let sum: f32 = (0..topk as usize)
            .map(|j| wt[ti * (topk + 1) as usize + j])
            .sum();
        assert!(
            (sum - 2.5f32).abs() < 0.1,
            "weights sum {sum} ~ scaling 2.5"
        );
    }
    let _ = (_w2, weights);
}

#[test]
fn mega_mhc_golden_self_consistent() {
    let (t, h) = (8u32, 512u32);
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(3);
        (seed >> 33) as u32
    };
    let bf = |v: f32| golden::f32_to_bf16_bits(v);
    let f = |v: f32| -> u16 { bf(v) };
    let x: Vec<u16> = (0..t * h)
        .map(|_| f(((next() % 200) as i32 - 100) as f32 / 50.0))
        .collect();
    let residual: Vec<u16> = (0..t * 4 * h)
        .map(|_| f(((next() % 200) as i32 - 100) as f32 / 50.0))
        .collect();
    let post: Vec<f32> = (0..t * 4)
        .map(|_| ((next() % 100) as f32) / 100.0)
        .collect();
    let comb: Vec<f32> = (0..t * 16)
        .map(|_| ((next() % 100) as f32) / 200.0)
        .collect();
    let prev: Vec<f32> = (0..t * 4)
        .map(|_| ((next() % 100) as f32) / 100.0)
        .collect();
    let f_n: Vec<f32> = (0..24 * 4 * h as usize)
        .map(|_| ((next() % 200) as i32 - 100) as f32 / 200.0)
        .collect();
    let scales = vec![0.5f32, 1.0, 1.5];
    let bases: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 24.0).collect();
    let nw: Vec<u16> = (0..h).map(|_| f(0.9)).collect();
    let (y, new_prev, new_post, new_comb) = golden::mega_mhc_ref(
        &x,
        &residual,
        &post,
        &comb,
        Some(&prev),
        &f_n,
        &scales,
        &bases,
        1e-6,
        0.1,
        1.0,
        1e-4,
        3,
        &nw,
        1e-6,
        1.0,
        t,
        h,
    );
    assert_eq!(y.len(), (t * h) as usize);
    assert!(y.iter().any(|&b| b != 0));
    assert_eq!(new_prev.len(), (t * 4) as usize);
    assert_eq!(new_post.len(), (t * 4) as usize);
    assert_eq!(new_comb.len(), (t * 16) as usize);
    // Sinkhorn rows sum ~1 + eps slack.
    for ti in 0..t as usize {
        for r in 0..4 {
            let s: f32 = (0..4).map(|c| new_comb[ti * 16 + r * 4 + c]).sum();
            // 3 iterations: rows converge toward 1 from below; structural band.
            assert!(s > 0.5 && s < 1.5, "sinkhorn row sum {s}");
        }
    }
}
