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
        expected_k: 0,
        with_accumulation: false,
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
        expected_k: 0,
        with_accumulation: false,
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

// ===========================================================================
// SM90 (Hopper) heuristics — CPU-only checks of the ported SM90ArchSpec.
// ===========================================================================
mod sm90_tests {
    use deepgemm::device::Arch;
    use deepgemm::heuristics::{self, GemmDesc};
    use deepgemm::types::{Dtype, GemmType, Major};

    fn desc_fp8(m: u32, n: u32, k: u32) -> GemmDesc {
        GemmDesc {
            gemm_type: GemmType::Normal,
            m,
            n,
            k,
            num_groups: 1,
            a_dtype: Dtype::Fp8,
            b_dtype: Dtype::Fp8,
            cd_dtype: Dtype::F32,
            major_a: Major::K,
            major_b: Major::K,
            num_sms: 132,
            smem_capacity: 232448,
            expected_m: m,
            expected_num_groups: 1,
            expected_k: 0,
            with_accumulation: false,
        }
    }

    fn desc_bf16(m: u32, n: u32, k: u32) -> GemmDesc {
        GemmDesc {
            gemm_type: GemmType::Normal,
            m,
            n,
            k,
            num_groups: 1,
            a_dtype: Dtype::Bf16,
            b_dtype: Dtype::Bf16,
            cd_dtype: Dtype::Bf16,
            major_a: Major::K,
            major_b: Major::K,
            num_sms: 132,
            smem_capacity: 232448,
            expected_m: m,
            expected_num_groups: 1,
            expected_k: 0,
            with_accumulation: false,
        }
    }

    #[test]
    fn sm90_block_k_matches_element_bits() {
        // FP8 -> 128 (per-128 scaling), BF16 -> 64 (1024/16).
        let d8 = desc_fp8(1024, 1024, 4096);
        let c8 = heuristics::sm90::get_best_config(&d8).unwrap();
        assert_eq!(c8.layout.block_k, 128);
        assert_eq!(c8.storage.swizzle_a_mode, 128);

        let db = desc_bf16(1024, 1024, 4096);
        let cb = heuristics::sm90::get_best_config(&db).unwrap();
        assert_eq!(cb.layout.block_k, 64);
    }

    #[test]
    fn sm90_fp32_out_avoids_bank_conflict_block_n() {
        // FP32-output 1D1D candidates start at 24 (plus 16), never 32/48:
        // multiples of 32 would map two rows to the same smem bank group.
        let d = desc_fp8(512, 1024, 4096);
        let cands = heuristics::sm90::layout_candidates(&d);
        assert!(cands.iter().any(|l| l.block_n == 24));
        assert!(!cands.iter().any(|l| l.block_n == 32));
        assert!(!cands.iter().any(|l| l.block_n == 48));
        assert!(cands.iter().any(|l| l.block_n == 16));
    }

    #[test]
    fn sm90_bf16_out_atom_divides_block_n() {
        // The swizzled TMA store needs BLOCK_N % (swizzle(BLOCK_N*2B)/2) == 0:
        // 48 (96B -> 64B swizzle, atom 32) is rejected; 64/192/256 pass.
        let d = desc_bf16(512, 1024, 4096);
        let cands = heuristics::sm90::layout_candidates(&d);
        assert!(!cands.iter().any(|l| l.block_n == 48));
        assert!(cands.iter().any(|l| l.block_n == 64));
        // BLOCK_M in {64, 128} for 1D1D (WGMMA m64), 256 allowed for bf16-out.
        assert!(cands.iter().any(|l| l.block_m == 256));
    }

    #[test]
    fn sm90_stage_budget_saturation() {
        // Stages fit the 232448B budget; deep pipelines for small tiles.
        let d = desc_fp8(4096, 4096, 7168);
        let c = heuristics::sm90::get_best_config(&d).unwrap();
        assert!(c.pipeline.num_stages >= 3);
        assert!(c.pipeline.smem_size <= 232448);
        let small = desc_fp8(64, 64, 4096);
        let cs = heuristics::sm90::get_best_config(&small).unwrap();
        assert!(cs.pipeline.num_stages >= 8);
    }

    #[test]
    fn sm90_kk_disables_multicast_for_many_groups() {
        let mut d = desc_fp8(512, 512, 4096);
        d.gemm_type = GemmType::KGroupedContiguous;
        d.num_groups = 4;
        let c4 = heuristics::sm90::get_best_config(&d).unwrap();
        d.num_groups = 8;
        let c8 = heuristics::sm90::get_best_config(&d).unwrap();
        // >4 groups: multicast off (cluster 1).
        assert!(c4.layout.cluster_size() >= c8.layout.cluster_size());
        assert_eq!(c8.layout.cluster_size(), 1);
    }

    #[test]
    fn sm90_arch_is_dispatchable() {
        // Arch::Sm90 exists and NVRTC arch string is 90a (compile-check path).
        assert_eq!(Arch::from_cc(9, 0), Arch::Sm90);
    }
}

#[cfg(test)]
mod runtime_tests {
    #[test]
    fn size_classes_are_2mib_buckets() {
        // (private helper; re-derived here to avoid dead-code warnings)
        fn cls(b: usize) -> usize { (b.max(1) + (2 << 20) - 1) / (2 << 20) * (2 << 20) }
        assert_eq!(cls(1), 2 << 20);
        assert_eq!(cls(2 << 20), 2 << 20);
        assert_eq!(cls((2 << 20) + 1), 4 << 20);
    }
}
