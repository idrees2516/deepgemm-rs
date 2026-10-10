//! CPU tests for the SM90 FP8 1D2D GEMM port (`kernels/gemm_sm90_1d2d.cu` +
//! `src/api_1d2d.rs`).
//!
//! 1. `sf_indexing_math_*` — a pure-Rust simulation of the kernel's
//!    scale-factor indexing, checked bit-exactly against the mathematical
//!    1d2d reference `D[m][n] = sum_kb sfa[m][kb] * sfb[n/128][kb] *
//!    dot(A[m,kb], B[n,kb])`. This is where the 1d2d kernel is subtle:
//!    * which SFB entry (of the 2D `[n/128, k/128]` grid, K- or MN-major)
//!      the math warps preload into which `smem_sfb` slot,
//!    * the straddle predicate (`i < num_former_iters`) selecting row 0 vs
//!      row 1 for each 8-column accumulator group,
//!    * the SFA TMA box read out of the transposed `[k/128, tma_aligned(m)]`
//!      FP32 layout,
//!    * the m-grouped-contiguous SFB group offset.
//!
//! All values are dyadic rationals with tiny mantissas, so f32
//! arithmetic is exact and equality is bit-exact.
//! 2. `sm90_1d2d_offline_compile_check` — NVRTC-compiles the exact
//!    translation unit + wrapper the launcher builds (several tile shapes,
//!    both SFB majors) to PTX (compute_90a) and SASS (sm_90a).
//! 3. `tile_chooser_*` — the private 1d2d tile chooser only returns tiles
//!    satisfying the kernel's static contract.

use deepgemm::api_1d2d::{
    choose_tiling, multicast_tail_safe, num_1d_blocks_per_group, wrapper_body, Tile1d2d,
};
use deepgemm::jit;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn idx(_rows: u32, cols: u32, r: u32, c: u32) -> usize {
    (r * cols + c) as usize
}

/// Dot product of one 128-wide K slice, zero-padded past `k` (TMA loads
/// zero-fill out-of-bounds elements, and zero contributes nothing).
fn slice_dot(a_row: &[f32], b_row: &[f32], k: u32, kb: u32) -> f32 {
    let base = kb * 128;
    let mut acc = 0.0f32;
    let end = 128.min(k.saturating_sub(base));
    for kk in 0..end {
        acc += a_row[(base + kk) as usize] * b_row[(base + kk) as usize];
    }
    acc
}

/// Logical SFB element `(n_block, kb)` for the given major:
/// K-major `[n/128, k/128]`: n contiguous along k; MN-major: transposed.
fn sfb_at(sfb: &[f32], n_blocks: u32, kb_total: u32, mn_major: bool, nb: u32, kb: u32) -> f32 {
    let (stride_n, stride_k) = if mn_major {
        (1, n_blocks)
    } else {
        (kb_total, 1)
    };
    sfb[(nb * stride_n + kb * stride_k) as usize]
}

// ---------------------------------------------------------------------------
// Kernel-side simulation (mirrors gemm_sm90_1d2d.cu line by line)
// ---------------------------------------------------------------------------

/// Simulate the Normal-GEMM 1d2d kernel for one tile shape.
/// `a`: `[m][k]`, `b`: `[n][k]`, `sfa`: `[m][kb]`, `sfb`: `[n/128][kb]`
/// (logical layout per `mn_major`). Returns `D[m][n]`.
// Simulation harness mirrors the full kernel parameter set.
#[allow(clippy::too_many_arguments)]
fn kernel_sim_normal(
    block_m: u32,
    block_n: u32,
    mn_major: bool,
    m: u32,
    n: u32,
    k: u32,
    a: &[f32],
    b: &[f32],
    sfa: &[f32],
    sfb: &[f32],
) -> Vec<f32> {
    let kb_total = ceil_div(k, 128);
    let n_sfb = ceil_div(n, 128);
    let uniform = 128 % block_n == 0;
    let mut d = vec![0.0f32; (m * n) as usize];

    for m_block in 0..ceil_div(m, block_m) {
        for n_block in 0..ceil_div(n, block_n) {
            let n0 = n_block * block_n;
            // --- straddle geometry (CUDA: num_former_iters / num_full_iters)
            let (num_former, num_full) = if uniform {
                (block_n / 8, block_n / 8)
            } else {
                (
                    block_n.min(128 - n0 % 128) / 8,
                    n.saturating_sub(n0).min(block_n) / 8,
                )
            };
            let num_sfb = kb_total * if num_former >= num_full { 1 } else { 2 };
            // --- SFB preload into smem_sfb (group offset = 0 for Normal)
            let previous_group_offset = 0u32;
            let (stride_n, stride_k) = if mn_major { (1, n_sfb) } else { (kb_total, 1) };
            let local_base = previous_group_offset + (n0 / 128) * stride_n;
            let mut smem_sfb = vec![0.0f32; (kb_total * 2) as usize];
            for i in 0..num_sfb {
                smem_sfb[i as usize] = if i < kb_total {
                    sfb[(local_base + i * stride_k) as usize]
                } else {
                    sfb[(local_base + (i - kb_total) * stride_k + stride_n) as usize]
                };
            }
            // --- K sweep: wgmma + register promotion
            let mut final_accum = vec![0.0f32; (block_m * block_n) as usize];
            for kb in 0..kb_total {
                let scale_b_0 = smem_sfb[kb as usize];
                let scale_b_1 = if uniform {
                    0.0
                } else {
                    smem_sfb[(kb + kb_total) as usize]
                };
                for r in 0..block_m {
                    let gr = m_block * block_m + r;
                    if gr >= m {
                        continue;
                    }
                    let a_row = &a[idx(m, k, gr, 0)..idx(m, k, gr + 1, 0)];
                    let scale_a = sfa[idx(m, kb_total, gr, kb)];
                    for j in 0..block_n {
                        let gn = n0 + j;
                        if gn >= n {
                            continue;
                        }
                        let b_row = &b[idx(n, k, gn, 0)..idx(n, k, gn + 1, 0)];
                        let accum = slice_dot(a_row, b_row, k, kb);
                        let i = j / 8;
                        let predicate = uniform || i < num_former;
                        let scale = if predicate { scale_b_0 } else { scale_b_1 };
                        final_accum[(r * block_n + j) as usize] += scale_a * scale * accum;
                    }
                }
            }
            // --- epilogue: each (m, n) written exactly once
            for r in 0..block_m {
                let gr = m_block * block_m + r;
                if gr >= m {
                    continue;
                }
                for j in 0..block_n {
                    let gn = n0 + j;
                    if gn < n {
                        d[idx(m, n, gr, gn)] = final_accum[(r * block_n + j) as usize];
                    }
                }
            }
        }
    }
    d
}

/// Mathematical reference: `D[m][n] = sum_kb sfa[m][kb] * sfb[n/128][kb] *
/// dot(A[m], B[n])` over K slice `kb`.
// Reference GEMM mirrors the kernel parameter set.
#[allow(clippy::too_many_arguments)]
fn reference_normal(
    m: u32,
    n: u32,
    k: u32,
    a: &[f32],
    b: &[f32],
    sfa: &[f32],
    sfb: &[f32],
    mn_major: bool,
) -> Vec<f32> {
    let kb_total = ceil_div(k, 128);
    let n_sfb = ceil_div(n, 128);
    let mut d = vec![0.0f32; (m * n) as usize];
    for gr in 0..m {
        let a_row = &a[idx(m, k, gr, 0)..idx(m, k, gr + 1, 0)];
        for gn in 0..n {
            let b_row = &b[idx(n, k, gn, 0)..idx(n, k, gn + 1, 0)];
            let mut acc = 0.0f32;
            for kb in 0..kb_total {
                let scale = sfa[idx(m, kb_total, gr, kb)]
                    * sfb_at(sfb, n_sfb, kb_total, mn_major, gn / 128, kb);
                acc += scale * slice_dot(a_row, b_row, k, kb);
            }
            d[idx(m, n, gr, gn)] = acc;
        }
    }
    d
}

/// Deterministic pseudo-data with exact f32 representations (small integers
/// and dyadic-rational scales), so simulations compare bit-exactly.
fn gen_values(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // [-4, 4] integers / 2 => half-integers (exact in f32).
            (((x % 17) as i32) - 8) as f32 * 0.5
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. SF indexing math
// ---------------------------------------------------------------------------

/// The kernel's SFA transport: TMA box `[BLOCK_M, 1]` at `(m_idx, kb)` reads
/// the transposed layout `sfa_t[kb][tma_aligned(m)]` (built by
/// `sm90::transpose_sf_fp32`) and must deliver `sfa[m][kb]` for every row
/// of the block.
#[test]
fn sf_indexing_math_sfa_transposed_layout() {
    let (m, k) = (100u32, 640u32); // kb = 5, m not 4-aligned
    let kb_total = ceil_div(k, 128);
    let tma_aligned = ceil_div(m, 4) * 4; // heuristics::tma_aligned_size(m, 4)
    let sfa: Vec<f32> = gen_values((m * kb_total) as usize, 11);
    // transpose (mn, kb) -> [kb][tma_aligned(m)]
    let mut sfa_t = vec![0.0f32; (tma_aligned * kb_total) as usize];
    for kb in 0..kb_total {
        for r in 0..m {
            sfa_t[(kb * tma_aligned + r) as usize] = sfa[idx(m, kb_total, r, kb)];
        }
    }
    // TMA box reads for every (m_block, kb): row r of the block gets
    // sfa_t[kb][m_idx + r].
    for block_m in [16u32, 32, 64, 128, 256] {
        for m_block in 0..ceil_div(m, block_m) {
            let m_idx = m_block * block_m;
            for kb in 0..kb_total {
                for r in 0..block_m {
                    let gr = m_idx + r;
                    if gr < m {
                        let staged = sfa_t[(kb * tma_aligned + gr) as usize];
                        assert_eq!(
                            staged,
                            sfa[idx(m, kb_total, gr, kb)],
                            "SFA transposed-layout mismatch (block_m {block_m}, row {gr}, kb {kb})"
                        );
                    }
                }
            }
        }
    }
}

/// Core 1d2d check: the kernel's SFB preload + straddle predicate must
/// multiply each accumulator by exactly `sfa[m][kb] * sfb[n/128][kb]`.
/// Sweeps tile shapes (uniform and straddling), K tails, both SFB majors.
#[test]
fn sf_indexing_math_normal_gemm() {
    let cases: &[(u32, u32, u32, u32)] = &[
        // (block_m, block_n, m, n) — k varies below
        (64, 128, 100, 256),  // uniform: BLOCK_N | 128
        (128, 192, 300, 576), // straddle: n0 % 128 in {0, 64}
        (128, 96, 200, 300),  // straddle: n0 % 128 in {0, 96, 64, 32}
        (64, 160, 130, 480),  // straddle: n0 % 128 in {0, 32, 64, 96}
        (128, 64, 64, 96),    // uniform, n tail (96 % 64 != 0)
        (64, 16, 40, 40),     // uniform tiny
        (256, 128, 300, 128), // two WGMMA waves per warpgroup
        (16, 48, 16, 100),    // BLOCK_M < 64 (single store warp)
    ];
    for &(bm, bn, m, n) in cases {
        for &k in &[512u32, 576, 384] {
            // 512 = 4 blocks; 576 = 4.5 (tail); 384 = 3 blocks
            for &mn_major in &[false, true] {
                let kb_total = ceil_div(k, 128);
                let n_sfb = ceil_div(n, 128);
                let a = gen_values((m * k) as usize, 1);
                let b = gen_values((n * k) as usize, 2);
                let sfa = gen_values((m * kb_total) as usize, 3);
                let sfb = gen_values((n_sfb * kb_total) as usize, 4);
                let want = reference_normal(m, n, k, &a, &b, &sfa, &sfb, mn_major);
                let got = kernel_sim_normal(bm, bn, mn_major, m, n, k, &a, &b, &sfa, &sfb);
                for gr in 0..m {
                    for gn in 0..n {
                        assert_eq!(
                            want[idx(m, n, gr, gn)],
                            got[idx(m, n, gr, gn)],
                            "bm={bm} bn={bn} m={m} n={n} k={k} mn_major={mn_major} at ({gr},{gn})"
                        );
                    }
                }
            }
        }
    }
}

/// M-grouped-contiguous flavor: the SFB preload adds the group base offset
/// `group * (n_sfb * kb_total)` and the B tile is group-relative, while SFA
/// and D stay absolute. Padding m-blocks (negative expert id) produce zeros.
#[test]
fn sf_indexing_math_m_grouped_contiguous() {
    let (bm, bn) = (64u32, 128u32);
    let (m, n, k) = (256u32, 192u32, 256u32);
    let kb_total = ceil_div(k, 128);
    let n_sfb = ceil_div(n, 128);
    // 4 m-blocks: experts [1, -1 (padding), 0, 1]
    let grouped_layout: [i32; 4] = [1, -1, 0, 1];
    let num_groups = 2u32;

    let a = gen_values((m * k) as usize, 5);
    let sfa = gen_values((m * kb_total) as usize, 6);
    let b_per_group: Vec<Vec<f32>> = (0..num_groups)
        .map(|g| gen_values((n * k) as usize, 20 + g as u64))
        .collect();
    let sfb_per_group: Vec<Vec<f32>> = (0..num_groups)
        .map(|g| gen_values((n_sfb * kb_total) as usize, 30 + g as u64))
        .collect();
    // Stacked SFB (what the kernel's `sfb` pointer sees).
    let sfb_all: Vec<f32> = sfb_per_group
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    let mn_major = false;

    // --- kernel simulation (contiguous) ---
    let mut d = vec![0.0f32; (m * n) as usize];
    let uniform = 128 % bn == 0;
    for m_block in 0..ceil_div(m, bm) {
        let group = grouped_layout[m_block as usize].max(0) as u32;
        let block_is_padding = grouped_layout[m_block as usize] < 0;
        for n_block in 0..ceil_div(n, bn) {
            let n0 = n_block * bn;
            let (num_former, num_full) = if uniform {
                (bn / 8, bn / 8)
            } else {
                (bn.min(128 - n0 % 128) / 8, n.saturating_sub(n0).min(bn) / 8)
            };
            let num_sfb = kb_total * if num_former >= num_full { 1 } else { 2 };
            // CUDA: previous_group_offset = get_global_idx<true, SF_K>(
            //       n_sfb * kb_total, 0, 0, m_block) = group * n_sfb * kb_total
            let previous_group_offset = group * n_sfb * kb_total;
            let (stride_n, stride_k) = if mn_major { (1, n_sfb) } else { (kb_total, 1) };
            let local_base = previous_group_offset + (n0 / 128) * stride_n;
            let mut smem_sfb = vec![0.0f32; (kb_total * 2) as usize];
            for i in 0..num_sfb {
                smem_sfb[i as usize] = if i < kb_total {
                    sfb_all[(local_base + i * stride_k) as usize]
                } else {
                    sfb_all[(local_base + (i - kb_total) * stride_k + stride_n) as usize]
                };
            }
            // is_computation_valid: padding blocks skip the MMA but still
            // write their (zero) accumulators through the epilogue.
            for kb in 0..kb_total {
                let scale_b_0 = if block_is_padding {
                    0.0
                } else {
                    smem_sfb[kb as usize]
                };
                for r in 0..bm {
                    let gr = m_block * bm + r;
                    if gr >= m || block_is_padding {
                        continue;
                    }
                    let a_row = &a[idx(m, k, gr, 0)..idx(m, k, gr + 1, 0)];
                    let scale_a = sfa[idx(m, kb_total, gr, kb)];
                    for j in 0..bn {
                        let gn = n0 + j;
                        if gn >= n {
                            continue;
                        }
                        // B is group-relative: column gn of THIS group's B.
                        let b_row =
                            &b_per_group[group as usize][idx(n, k, gn, 0)..idx(n, k, gn + 1, 0)];
                        let accum = slice_dot(a_row, b_row, k, kb);
                        let i = j / 8;
                        let predicate = uniform || i < num_former;
                        let scale_b_1 = if uniform {
                            0.0
                        } else {
                            smem_sfb[(kb + kb_total) as usize]
                        };
                        let scale = if predicate { scale_b_0 } else { scale_b_1 };
                        d[idx(m, n, gr, gn)] += scale_a * scale * accum;
                    }
                }
            }
        }
    }

    // --- reference ---
    for gr in 0..m {
        let m_block = gr / bm;
        let group = grouped_layout[m_block as usize];
        for gn in 0..n {
            let want = if group < 0 {
                0.0
            } else {
                let g = group as u32;
                let a_row = &a[idx(m, k, gr, 0)..idx(m, k, gr + 1, 0)];
                let b_row = &b_per_group[g as usize][idx(n, k, gn, 0)..idx(n, k, gn + 1, 0)];
                let mut acc = 0.0f32;
                for kb in 0..kb_total {
                    let scale = sfa[idx(m, kb_total, gr, kb)]
                        * sfb_at(
                            &sfb_per_group[g as usize],
                            n_sfb,
                            kb_total,
                            mn_major,
                            gn / 128,
                            kb,
                        );
                    acc += scale * slice_dot(a_row, b_row, k, kb);
                }
                acc
            };
            assert_eq!(
                want,
                d[idx(m, n, gr, gn)],
                "contiguous mismatch at ({gr},{gn}) (group {group})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Tile chooser contract
// ---------------------------------------------------------------------------

#[test]
fn tile_chooser_returns_legal_tiles() {
    let (sms, cap) = (132u32, 232448u32);
    for &(m, n, k) in &[
        (4096u32, 7168, 7168),
        (1, 16, 128),
        (17, 33, 129),
        (128, 128, 512),
        (300, 576, 576),
        (64, 4096, 16384),
        (8192, 8192, 2048),
    ] {
        if n % 8 != 0 {
            continue;
        }
        let t = choose_tiling(m, n, k, sms, cap).expect("tiling must exist");
        assert!(t.block_m.is_multiple_of(8), "BLOCK_M % 8: {t:?}");
        assert!(
            t.block_n.is_multiple_of(8) && t.block_n <= 192,
            "BLOCK_N bounds: {t:?}"
        );
        assert!(
            t.block_m <= 128 || t.block_n <= 128,
            "register budget: {t:?}"
        );
        // 1d2d straddle legality (kernel static assert).
        if t.block_n > 128 {
            let diff = t.block_n - 128;
            assert!(
                t.block_n.is_multiple_of(diff) || 128 % diff == 0,
                "straddle ladder: {t:?}"
            );
        }
        // BF16 store atom legality (kernel static assert).
        let atom = if t.swizzle_cd_mode == 0 {
            t.block_n
        } else {
            t.swizzle_cd_mode / 2
        };
        assert!(
            t.block_n.is_multiple_of(atom) && t.block_n / atom <= 32 && atom % 8 == 0,
            "{t:?}"
        );
        assert!(t.num_stages >= 3, "{t:?}");
        assert!(t.smem_size <= cap, "smem budget: {t:?}");
        assert!(matches!(t.num_math_threads, 128 | 256));
        assert_eq!(t.num_math_threads, if t.block_m <= 64 { 128 } else { 256 });
        assert!(t.cluster_size == 1 || t.cluster_size == 2);
    }
}

#[test]
fn tile_chooser_avoids_multicast_odd_tails() {
    // m=300 with BLOCK_M=64 => 5 m-blocks; a 1-D group size of 8 leaves an
    // odd tail of 5 -> multicast-on-B schedules must be rejected.
    let t = choose_tiling(300, 4096, 4096, 132, 232448).unwrap();
    if t.cluster_size == 2 {
        let primary = if t.multicast_on_a {
            ceil_div(4096, t.block_n)
        } else {
            ceil_div(300, t.block_m)
        };
        let group = num_1d_blocks_per_group(t.block_m, t.block_n, t.multicast_on_a, 132);
        assert!(
            multicast_tail_safe(primary, group),
            "odd-tail multicast scheduled: {t:?} (primary {primary}, group {group})"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Offline NVRTC compile check (PTX + SASS)
// ---------------------------------------------------------------------------

#[test]
fn sm90_1d2d_offline_compile_check() {
    let tu = format!(
        "{}\n{}",
        jit::kernel_src::WGMMA_H,
        jit::kernel_src::GEMM_SM90_1D2D
    );

    // (a) chooser-driven tiles for realistic shapes, both SFB majors.
    let mut variants: Vec<(String, Tile1d2d)> = Vec::new();
    for &(m, n, k) in &[(4096u32, 7168, 7168), (64, 4096, 7168), (128, 128, 512)] {
        let tile = choose_tiling(m, n, k, 132, 232448).expect("tiling");
        variants.push((format!("chooser m{m}n{n}k{k}"), tile));
    }
    // (b) hand-picked tiles covering every kernel code path:
    //  - BLOCK_N=192: 128B-swizzled D + straddle ladder (2 rungs).
    //  - BLOCK_N=96: 64B-swizzled D + straddle ladder (4 rungs).
    //  - BLOCK_N=16: 32B-swizzled D, uniform scales.
    //  - BLOCK_M=16: single store warp; BLOCK_M=256: two WGMMA waves.
    //  - swizzle_cd=0: unswizzled row-major D staging (never chosen by the
    //    heuristic, still a legal kernel instantiation).
    let sms = 132u32;
    variants.push((
        "straddle-192".into(),
        Tile1d2d {
            block_m: 128,
            block_n: 192,
            swizzle_cd_mode: 128,
            num_stages: 4,
            smem_size: 227328,
            num_math_threads: 256,
            cluster_size: 1,
            multicast_on_a: false,
            num_sms: sms,
        },
    ));
    variants.push((
        "straddle-96-sw64".into(),
        Tile1d2d {
            block_m: 64,
            block_n: 96,
            swizzle_cd_mode: 64,
            num_stages: 8,
            smem_size: 212992,
            num_math_threads: 128,
            cluster_size: 1,
            multicast_on_a: false,
            num_sms: sms,
        },
    ));
    variants.push((
        "uniform-16-sw32".into(),
        Tile1d2d {
            block_m: 16,
            block_n: 16,
            swizzle_cd_mode: 32,
            num_stages: 16,
            smem_size: 69120,
            num_math_threads: 128,
            cluster_size: 1,
            multicast_on_a: false,
            num_sms: sms,
        },
    ));
    variants.push((
        "two-waves-256x64-multicast".into(),
        Tile1d2d {
            block_m: 256,
            block_n: 64,
            swizzle_cd_mode: 128,
            num_stages: 4,
            smem_size: 200704,
            num_math_threads: 256,
            cluster_size: 2,
            multicast_on_a: true,
            num_sms: sms,
        },
    ));
    variants.push((
        "unswizzled-d".into(),
        Tile1d2d {
            block_m: 64,
            block_n: 128,
            swizzle_cd_mode: 0,
            num_stages: 8,
            smem_size: 216064,
            num_math_threads: 128,
            cluster_size: 1,
            multicast_on_a: false,
            num_sms: sms,
        },
    ));

    for (name, tile) in &variants {
        for &sfb_mn in &[false, true] {
            let body = wrapper_body(tile, sfb_mn);
            let tag = format!("sm90-1d2d/{name}/mn={sfb_mn}");
            let r = jit::compile_check_kernel(&tu, &body, "90a", &tag)
                .unwrap_or_else(|e| panic!("{tag}: {e}"));
            let sass = r.cubin_len.unwrap_or(0);
            println!(
                "{tag}: PTX {} B, SASS {} B (bm={} bn={} swd={} stages={} math={} cluster={})",
                r.ptx_len,
                sass,
                tile.block_m,
                tile.block_n,
                tile.swizzle_cd_mode,
                tile.num_stages,
                tile.num_math_threads,
                tile.cluster_size
            );
            assert!(r.ptx_len > 10_000, "{tag}: suspiciously small PTX");
            assert!(sass > 0, "{tag}: SASS generation failed");
        }
    }
}
