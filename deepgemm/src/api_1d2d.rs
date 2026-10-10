//! SM90 (Hopper) FP8 1D2D GEMM — port of upstream DeepGEMM's
//! `sm90_fp8_gemm_1d2d` (`impls/sm90_fp8_gemm_1d2d.cuh` +
//! `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`).
//!
//! # 1D2D scaling, in one page
//! * `A` (activations): **1D** FP32 scales — one per token per 128-K slice
//!   (`SfFp32`, `[k/128, tma_aligned(m)]`, built with
//!   [`crate::sm90::transpose_sf_fp32`] / [`crate::sm90::sf_fp32_from_host`]).
//!   Loaded per stage by TMA together with the A tile.
//! * `B` (weights): **2D** FP32 scales — one per 128x128 block
//!   ([`SfbFp32`], `[ceil(n/128), ceil(k/128)]`, K- or MN-major). NOT loaded
//!   by TMA: the math warps preload the 1–2 relevant rows into SMEM once per
//!   output block and read them per K block (see `kernels/gemm_sm90_1d2d.cu`).
//! * `D` is **BF16** (upstream supports no other C/D dtype for 1d2d) and is
//!   written exactly once per tile — no zero-init, no accumulation.
//!
//! # Layout contract
//! `A`: `[m, k]` FP8 K-major (row stride `a.outer_stride >= k`); `B`: `[n, k]`
//! FP8 K-major; `D`: `[m, n]` BF16 with row stride `d.stride >= n`.
//!
//! # Tile choice
//! The kernel variant is picked by a private, upstream-faithful port of the
//! SM90 `Kernel1D2D` heuristics (`csrc/jit_kernels/heuristics/sm90.hpp`):
//! BLOCK_M {64, 128} (+16/32 for tiny M, +256 since CD is BF16), BLOCK_N
//! multiples of 16 up to **192** (1d2d register ceiling), the 1d2d
//! B-scale-straddle legality filter, the BF16 STSM store-atom filter, a SMEM
//! budget that includes the runtime SFB staging, and the upstream
//! bandwidth-cycle comparator. Multicast (cluster 2) is additionally
//! rejected when the swizzled schedule's 1-D block groups would leave an odd
//! tail >= 3 (the prelude's scheduler lacks upstream's odd-tail repair, so
//! the launcher avoids ever scheduling one).

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, Major, Operand, Output};

// Re-exported for call-site ergonomics: the 1D activation-scale tensor and
// its constructors are shared with the SM90 1D1D path.
pub use crate::sm90::{sf_fp32_from_host, transpose_sf_fp32, SfFp32};

// ---------------------------------------------------------------------------
// 2D weight-scale tensor (SFB)
// ---------------------------------------------------------------------------

/// FP32 2D scale factors for `B`: one scale per 128x128 block, i.e.
/// `[ceil(n/128), ceil(k/128)]` with the given major (K-major: k contiguous;
/// MN-major: n contiguous). Consumed directly from global memory by the
/// kernel's math warps — no transform, no TMA descriptor.
pub struct SfbFp32 {
    pub buf: DevBuffer,
    /// `ceil(n / 128)`.
    pub n_blocks: u32,
    /// `ceil(k / 128)`.
    pub k_blocks: u32,
    pub major: Major,
}

/// Build an [`SfbFp32`] from host `(n_blocks * k_blocks)` row-major data.
pub fn sfb_fp32_from_host(
    dev: &Device,
    stream: &DevStream,
    data: &[f32],
    n_blocks: u32,
    k_blocks: u32,
    major: Major,
) -> DgResult<SfbFp32> {
    if data.len() != (n_blocks as usize) * (k_blocks as usize) {
        return Err(DgError::InvalidArg(
            "sfb_fp32 host data size mismatch".into(),
        ));
    }
    let buf = crate::device::alloc_and_upload(dev, data, stream.raw())?;
    Ok(SfbFp32 {
        buf,
        n_blocks,
        k_blocks,
        major,
    })
}

fn require_sm90(dev: &Device) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm90) {
        return Err(DgError::Unsupported(format!(
            "the 1d2d kernel requires SM90 (Hopper); this device is {:?}",
            dev.arch
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Private tile chooser — upstream SM90 heuristics for Kernel1D2D
// ---------------------------------------------------------------------------

const BLOCK_K: u32 = 128;
const NUM_MAX_STAGES: u32 = 16;

/// Chosen kernel tiling (all values feed the kernel's template parameters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile1d2d {
    pub block_m: u32,
    pub block_n: u32,
    /// TMA swizzle atom of the D staging (bytes; 0 = unswizzled row-major).
    pub swizzle_cd_mode: u32,
    pub num_stages: u32,
    /// Dynamic SMEM bytes to reserve at launch (>= the kernel's usage).
    pub smem_size: u32,
    pub num_math_threads: u32,
    /// Cluster size (1 or 2; 2 => TMA multicast).
    pub cluster_size: u32,
    /// Multicast the A (+SFA) tiles (cluster_n = 2) or B (cluster_m = 2).
    pub multicast_on_a: bool,
    pub num_sms: u32,
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn align_up(a: u32, b: u32) -> u32 {
    ceil_div(a, b) * b
}

/// Upstream `get_swizzle_mode`: the largest TMA swizzle atom in
/// {128, 64, 32, 16} dividing `inner_bytes`. This crate's TMA layer uses
/// mode 0 (no swizzle) where upstream picks the 16B interleave.
fn swizzle_mode_upstream(inner_bytes: u32) -> u32 {
    for &mode in &[128u32, 64, 32] {
        if inner_bytes % mode == 0 {
            return mode;
        }
    }
    0
}

/// `kNum1DBlocksPerGroup` (prelude `get_num_1d_blocks_per_group`), needed
/// host-side to avoid scheduling multicast over an odd group tail.
pub fn num_1d_blocks_per_group(
    block_m: u32,
    block_n: u32,
    multicast_on_a: bool,
    num_sms: u32,
) -> u32 {
    let mut best = 0u32;
    let mut min_usage = u32::MAX;
    for &candidate in &[8u32, 16] {
        let usage = if multicast_on_a {
            candidate * block_n + ceil_div(num_sms, candidate) * block_m
        } else {
            candidate * block_m + ceil_div(num_sms, candidate) * block_n
        };
        if usage < min_usage {
            min_usage = usage;
            best = candidate;
        }
    }
    best
}

/// An odd (>1) tail of the last 1-D block group would pair a cluster's two
/// CTAs across different m-blocks (upstream repairs this inside the
/// scheduler; the shared prelude does not, so the chooser avoids it).
pub fn multicast_tail_safe(primary_blocks: u32, group_size: u32) -> bool {
    let tail = primary_blocks % group_size;
    tail == 0 || tail == 1 || tail % 2 == 0
}

/// Pick the 1d2d kernel tiling for a Normal `(m, n, k)` GEMM.
pub fn choose_tiling(
    m: u32,
    n: u32,
    k: u32,
    num_sms: u32,
    smem_capacity: u32,
) -> DgResult<Tile1d2d> {
    let capacity = smem_capacity.max(crate::heuristics::SMEM_CAPACITY_FALLBACK);

    // ---- BLOCK_M candidates (upstream SM90: Normal + BF16 CD) ----
    let mut block_m_candidates: Vec<u32> = vec![64, 128];
    if m <= 16 {
        block_m_candidates.push(16);
    }
    if m <= 32 {
        block_m_candidates.push(32);
    }
    block_m_candidates.push(256); // BF16 output supports 256

    // ---- enumerate (cluster, BLOCK_M, BLOCK_N) ----
    struct Cand {
        tile: Tile1d2d,
        num_cycles: u64,
    }
    let mut cands: Vec<Cand> = Vec::new();

    for (cluster_m, cluster_n) in [(1u32, 1u32), (1, 2), (2, 1)] {
        let cluster_size = cluster_m * cluster_n;
        if cluster_size > 2 {
            continue;
        }
        // SM count must be divisible by the cluster; the (blockIdx.x,
        // blockIdx.x ^ 1) peer pairing additionally needs an even SM count.
        if cluster_size > 1 && (num_sms % cluster_size != 0 || num_sms % 2 != 0) {
            continue;
        }
        let multicast_on_a = cluster_n > 1;
        for &block_m in &block_m_candidates {
            for block_n in (16..=192).step_by(16) {
                // Register budget: at least one dim below 128.
                if block_m > 128 && block_n > 128 {
                    continue;
                }
                // 1D2D unroll requirement (upstream): a BLOCK_N > BLOCK_K
                // tile must straddle the 128-column scale grid on a lattice
                // the compile-time ladder can enumerate.
                if block_n > BLOCK_K {
                    let diff = block_n - BLOCK_K;
                    if block_n % diff != 0 && BLOCK_K % diff != 0 {
                        continue;
                    }
                }
                // BF16 store atom: BLOCK_N must tile into swizzle/2-column
                // TMA store boxes (kernel static asserts mirror this).
                let swizzle_cd = swizzle_mode_upstream(block_n * 2);
                let atom_n = if swizzle_cd == 0 {
                    block_n
                } else {
                    swizzle_cd / 2
                };
                if block_n % atom_n != 0 || block_n / atom_n > 32 || atom_n % 8 != 0 {
                    continue;
                }
                // Odd-tail multicast guard (see `multicast_tail_safe`).
                if cluster_size > 1 {
                    let primary = if multicast_on_a {
                        ceil_div(n, block_n)
                    } else {
                        ceil_div(m, block_m)
                    };
                    let group = num_1d_blocks_per_group(block_m, block_n, multicast_on_a, num_sms);
                    if !multicast_tail_safe(primary, group) {
                        continue;
                    }
                }

                // ---- SMEM budget (upstream 1d2d pipeline config) ----
                let smem_cd = align_up(block_m * block_n * 2, 1024);
                let smem_barriers = NUM_MAX_STAGES * 8 * 2;
                let uniform_sfb = BLOCK_K % block_n == 0;
                let smem_extra_sfb = align_up(
                    ceil_div(k, BLOCK_K) * 4 * if uniform_sfb { 1 } else { 2 },
                    8,
                );
                let smem_extra = smem_cd + smem_barriers + smem_extra_sfb;
                let per_stage = block_m * BLOCK_K + block_n * BLOCK_K + align_up(block_m * 4, 128);
                if per_stage == 0 || smem_extra + per_stage * 3 > capacity {
                    continue;
                }
                let num_stages = ((capacity - smem_extra) / per_stage).min(NUM_MAX_STAGES);
                // Hide TMA latency: >= 3 stages (>= 4 for small tiles).
                if num_stages < 3 || (block_m * block_n < 128 * 192 && num_stages < 4) {
                    continue;
                }
                let smem_size = smem_extra + num_stages * per_stage;
                if smem_size > capacity {
                    continue;
                }

                // ---- upstream bandwidth-cycle comparator ----
                let num_blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
                let num_waves = ceil_div(num_blocks, num_sms);
                let l2_bw = (64u64 * num_sms as u64).min(8_000_000u64 / 1300);
                let l1_bw = 128u64 * num_sms as u64;
                let bytes_l2_ab = k as u64 * (block_m / cluster_n + block_n / cluster_m) as u64;
                let bytes_l1_ab = k as u64 * (block_m + block_n) as u64;
                let bytes_l1_tc = k as u64 * (64u32.max(block_m) + block_n) as u64
                    + (block_m * block_n) as u64 * 2;
                let bytes_cd = (block_m * block_n) as u64 * 2;
                let l2_cycles = (bytes_l2_ab + bytes_cd) * num_blocks as u64 / l2_bw;
                let l1_cycles = (bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks as u64 / l1_bw;
                let wave_eff = num_blocks as f64 / (num_waves as f64 * num_sms as f64);
                let mut num_cycles = (l1_cycles.max(l2_cycles) as f64 / wave_eff.max(1e-9)) as u64;
                // Multicast that cannot save a wave is a net loss.
                if cluster_size > 1 && num_waves <= 1 {
                    num_cycles = u64::MAX;
                }

                cands.push(Cand {
                    tile: Tile1d2d {
                        block_m,
                        block_n,
                        swizzle_cd_mode: swizzle_cd,
                        num_stages,
                        smem_size,
                        num_math_threads: if block_m <= 64 { 128 } else { 256 },
                        cluster_size,
                        multicast_on_a,
                        num_sms,
                    },
                    num_cycles,
                });
            }
        }
    }

    let best = cands
        .into_iter()
        .min_by_key(|c| c.num_cycles)
        .map(|c| c.tile)
        .ok_or_else(|| {
            DgError::Unsupported(format!(
                "no viable SM90 1d2d tiling for m={m} n={n} k={k} (smem {capacity}B)"
            ))
        })?;
    if crate::heuristics::print_configs_enabled() {
        eprintln!("[deepgemm-rs] sm90 1d2d config: {best:?}");
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// Kernel wrapper (the NVRTC-compiled `__dg_kernel` body)
// ---------------------------------------------------------------------------

/// Build the `extern "C" __dg_kernel` instantiation body for one tile —
/// exactly what [`fp8_gemm_nt_1d2d`] launches (also used by the offline
/// compile-check tests). Normal GEMM, BF16 output, runtime shapes.
pub fn wrapper_body(tile: &Tile1d2d, sfb_is_mn_major: bool) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    const float* sfb, int* grouped_layout,
    unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, const __grid_constant__ dg::TmaMap tma_sfa) {{
    dg::sm90_fp8_gemm_1d2d_impl<{sfb_mn}, 0, 0, 0, 1,
        {bm}, {bn}, 128, 128, 128, {swd},
        {stages}, 128, {math}, {mcast}, {mcoa}, {sms},
        (dg::GemmType)0, 1>
        (sfb, grouped_layout, m, n, k, tma_a, tma_b, tma_d, tma_sfa);
}}"#,
        sfb_mn = sfb_is_mn_major as u32,
        bm = tile.block_m,
        bn = tile.block_n,
        swd = tile.swizzle_cd_mode,
        stages = tile.num_stages,
        math = tile.num_math_threads,
        mcast = tile.cluster_size,
        mcoa = tile.multicast_on_a as u32,
        sms = tile.num_sms,
    )
}

/// SM90 1d2d translation unit: prelude (auto-prepended by the JIT) +
/// wgmma.h + gemm_sm90_1d2d.cu.
fn unit() -> &'static str {
    static U: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    U.get_or_init(|| {
        format!(
            "{}\n{}",
            jit::kernel_src::WGMMA_H,
            jit::kernel_src::GEMM_SM90_1D2D
        )
    })
    .as_str()
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

/// FP8 GEMM, NT layout, 1D2D fine-grained scaling (Hopper):
/// `D = (A * sfa) @ (B * sfb)^T` with per-token 1D scales on `A` and
/// per-128x128-block 2D scales on `B` (see the module docs). `D` is BF16.
// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_nt_1d2d(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    sfa: &SfFp32,
    b: &Operand,
    sfb: &SfbFp32,
    d: &mut Output,
) -> DgResult<()> {
    require_sm90(dev)?;
    if a.dtype != Dtype::Fp8 || b.dtype != Dtype::Fp8 {
        return Err(DgError::InvalidArg(
            "fp8_gemm_nt_1d2d requires FP8 operands".into(),
        ));
    }
    if a.major != Major::K || b.major != Major::K {
        return Err(DgError::InvalidArg(
            "fp8_gemm_nt_1d2d requires K-major operands".into(),
        ));
    }
    if d.dtype != Dtype::Bf16 {
        return Err(DgError::InvalidArg(
            "fp8_gemm_nt_1d2d outputs BF16 (upstream supports no other dtype)".into(),
        ));
    }
    let (m, n, k) = (a.rows, b.rows, a.k);
    if b.k != k {
        return Err(DgError::InvalidArg("A/B K mismatch".into()));
    }
    if d.rows != m || d.cols != n {
        return Err(DgError::InvalidArg("output shape mismatch".into()));
    }
    if n % 8 != 0 {
        // Upstream traps on this in-kernel; surface it as a clean error.
        return Err(DgError::InvalidArg("n must be a multiple of 8".into()));
    }
    if a.outer_stride < k || b.outer_stride < k {
        return Err(DgError::InvalidArg(
            "operand row stride smaller than K".into(),
        ));
    }
    if d.stride < d.cols {
        return Err(DgError::InvalidArg(
            "output row stride smaller than N".into(),
        ));
    }
    let want_kb = k.div_ceil(128);
    if sfa.mn != m || sfa.k_blocks != want_kb {
        return Err(DgError::InvalidArg(
            "sfa must be [m, k/128] (SfFp32 with mn == m, k_blocks == k/128)".into(),
        ));
    }
    if sfb.n_blocks != n.div_ceil(128) || sfb.k_blocks != want_kb {
        return Err(DgError::InvalidArg(
            "sfb must be [n/128, k/128] (SfbFp32 blocks must cover n and k)".into(),
        ));
    }

    let tile = choose_tiling(m, n, k, dev.num_sms, dev.smem_capacity)?;

    // TMA maps. A/B: FP8 K-major, box [BLOCK_K, BLOCK_M/N], 128B swizzle
    // (swizzle == BLOCK_K bytes: no TMA splits — upstream asserts this).
    // SFA: FP32 1D, box [BLOCK_M, 1] over the transposed [k/128,
    // tma_aligned(m)] layout. SFB needs no map (plain global loads).
    let tm_a = tma::make_tma_ab(
        dev,
        Dtype::Fp8,
        Major::K,
        &a.data,
        m,
        k,
        tile.block_m,
        BLOCK_K,
        a.outer_stride,
        1,
        128,
        false,
    )?;
    let tm_b = tma::make_tma_ab(
        dev,
        Dtype::Fp8,
        Major::K,
        &b.data,
        n,
        k,
        tile.block_n,
        BLOCK_K,
        b.outer_stride,
        1,
        128,
        false,
    )?;
    let tm_d = tma::make_tma_cd(
        dev,
        Dtype::Bf16,
        &d.data,
        m,
        n,
        tile.block_m,
        tile.block_n,
        d.stride,
        1,
        tile.swizzle_cd_mode,
    )?;
    let tm_sfa = tma::make_tma_sf_fp32(dev, &sfa.buf, sfa.mn, k, tile.block_m, 1)?;

    let sfb_mn = sfb.major == Major::Mn;
    let sig = format!("{tile:?}/sfb_mn={sfb_mn}");
    let body = wrapper_body(&tile, sfb_mn);
    let func = jit::get_kernel(dev, unit(), "gemm_sm90_1d2d", &sig, &body)?;

    let args = Args::new()
        .devptr(sfb.buf.ptr)
        .devptr(0) // grouped_layout: unused for GemmType::Normal
        .u32(m)
        .u32(n)
        .u32(k)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_d)
        .tensormap(&tm_sfa);

    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + tile.num_math_threads, 1, 1),
            smem: tile.smem_size,
            cluster: Some((tile.cluster_size, 1, 1)),
            pdl: true,
        },
        args,
    )
}
