//! CPU-only unit tests (no GPU required): heuristics, SF layout math, packing.

use deepgemm::heuristics;
use deepgemm::types::{Dtype, GemmType, Major};

fn desc(m: u32, n: u32, k: u32, a: Dtype, b: Dtype, num_sms: u32) -> heuristics::GemmDesc {
    heuristics::GemmDesc {
        gemm_type: GemmType::Normal,
        m,
        n,
        k,
        num_groups: 1,
        a_dtype: a,
        b_dtype: b,
        cd_dtype: Dtype::Bf16,
        major_a: Major::K,
        major_b: Major::K,
        num_sms,
        smem_capacity: 232448,
        expected_m: m,
        expected_num_groups: 1,
    }
}

#[test]
fn heuristics_picks_multicast_for_large_shapes() {
    let d = desc(8192, 8192, 7168, Dtype::Fp8, Dtype::Fp8, 148);
    let cfg = heuristics::get_best_config(&d).unwrap();
    assert!(cfg.layout.cluster_size() >= 1);
    assert!(cfg.layout.block_m >= 32 && cfg.layout.block_m <= 128);
    assert!(cfg.pipeline.num_stages >= 2);
    assert!(cfg.pipeline.smem_size <= 232448);
    // For a big square FP8 problem, 2-CTA clusters should win.
    assert_eq!(
        cfg.layout.cluster_size(),
        2,
        "expected cluster 2 for large square FP8"
    );
}

#[test]
fn heuristics_fp4_uses_block_k_256_and_128b_swizzle() {
    let d = desc(4096, 4096, 7168, Dtype::Fp4, Dtype::Fp4, 148);
    let cfg = heuristics::get_best_config(&d).unwrap();
    assert_eq!(cfg.layout.block_k, 256);
    assert_eq!(cfg.storage.swizzle_a_mode, 128);
    assert_eq!(cfg.storage.swizzle_b_mode, 128);
}

#[test]
fn heuristics_fp8_block_k_128() {
    let d = desc(4096, 4096, 7168, Dtype::Fp8, Dtype::Fp8, 148);
    let cfg = heuristics::get_best_config(&d).unwrap();
    assert_eq!(cfg.layout.block_k, 128);
    assert_eq!(cfg.storage.swizzle_a_mode, 128);
}

#[test]
fn heuristics_bf16_block_k_64() {
    let d = desc(2048, 2048, 2048, Dtype::Bf16, Dtype::Bf16, 148);
    let cfg = heuristics::get_best_config(&d).unwrap();
    assert_eq!(cfg.layout.block_k, 64);
}

#[test]
fn heuristics_m_grouped_forces_swap_ab() {
    let d = heuristics::GemmDesc {
        gemm_type: GemmType::MGroupedContiguous,
        m: 4096,
        n: 2048,
        k: 7168,
        num_groups: 128,
        a_dtype: Dtype::Fp8,
        b_dtype: Dtype::Fp8,
        cd_dtype: Dtype::Bf16,
        major_a: Major::K,
        major_b: Major::K,
        num_sms: 148,
        smem_capacity: 232448,
        expected_m: 4096,
        expected_num_groups: 128,
    };
    let cfg = heuristics::get_best_config(&d).unwrap();
    assert!(cfg.layout.swap_ab);
    assert_eq!(cfg.layout.block_n, 128);
}

#[test]
fn heuristics_stage_budget() {
    let d = desc(8192, 8192, 7168, Dtype::Fp4, Dtype::Fp4, 148);
    let cfg = heuristics::get_best_config(&d).unwrap();
    // FP4 packed: 128B per 256-elem row; A/B stages each LOAD x 128B.
    let l = &cfg.layout;
    let s = &cfg.storage;
    let per_stage = s.load_block_m * l.block_k / 2
        + s.load_block_n * l.block_k / 2
        + heuristics::get_sf_block_sizes(l.block_m, l.block_n, true).0 * l.block_k / 32
        + heuristics::get_sf_block_sizes(l.block_m, l.block_n, true).1 * l.block_k / 32;
    assert!(
        cfg.pipeline.smem_size + per_stage > 232448,
        "stages must saturate the SMEM budget"
    );
}

#[test]
fn sf_packing_layout() {
    // SF tensor dims: [ceil(k/gran/4), TMA-aligned(mn)] int32.
    assert_eq!(heuristics::sf_rows(7168, deepgemm::types::SfGran::G32), 56); // 7168/32/4
    assert_eq!(heuristics::sf_rows(7168, deepgemm::types::SfGran::G128), 14); // 7168/128/4
    assert_eq!(heuristics::tma_aligned_size(100, 4), 100);
    assert_eq!(heuristics::tma_aligned_size(101, 4), 104);
    // mk alignment for contiguous layouts (upstream: 128 on SM100).
    assert_eq!(heuristics::mk_alignment_for_contiguous_layout(), 128);
}

#[test]
fn ue8m0_packing_matches_reference() {
    // The transform packs 4 UE8M0 bytes per int32:
    //   packed = e0>>23 | e1>>15 | e2>>7 | e3<<1  (as fp32 bit tricks)
    // Equivalently: byte j of the word = exponent (biased) of SF j.
    let exponents: [u32; 4] = [130, 131, 129, 128];
    let floats: Vec<f32> = exponents.iter().map(|&e| f32::from_bits(e << 23)).collect();
    let packed = exponents[0] | (exponents[1] << 8) | (exponents[2] << 16) | (exponents[3] << 24);
    // Same as the device code's shifts:
    let words: Vec<u32> = floats.iter().map(|f| f.to_bits()).collect();
    let device_packed = (words[0] >> 23) | (words[1] >> 15) | (words[2] >> 7) | (words[3] << 1);
    assert_eq!(packed, device_packed);
}
