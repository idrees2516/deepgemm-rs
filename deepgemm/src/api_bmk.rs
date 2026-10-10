//! `bmk, bnk -> mn` einsum kernels (MLA projections) + PsumLayout grouped
//! GEMM launchers — see `kernels/bmk_bnk.cu` for the full concept notes.
//!
//! * [`bf16_bmk_bnk_mn_sm100`] / [`bf16_bmk_bnk_mn_sm90`]: the batched
//!   cross-projection `D[m, n] += sum_s sum_k A[s, m, k] * B[s, n, k]`
//!   (upstream `deep_gemm.einsum('bmk,bnk->mn', ...)`), split-K over the
//!   `(S, K/BLOCK_K)` slice space.  **D is read-modify-write**: initialize it
//!   before the call (the upstream contract passes `c == d`; a BF16 result is
//!   produced through a zeroed FP32 workspace, cast by the caller).
//! * [`bf16_m_grouped_gemm_nt_psum`] / [`bf16_k_grouped_gemm_tn_psum`]: the
//!   two *WithPsumLayout grouped-GEMM scheduler variants
//!   (`GemmType::MGroupedContiguousWithPsumLayout` /
//!   `GemmType::KGroupedContiguousWithPsumLayout`), driven by the local
//!   `dg::psum::Scheduler` port inside `bmk_bnk.cu`.
//!
//! # Psum layout contract (mirrors upstream `tests/generators.py`)
//!
//! `grouped_layout[g]` is a **prefix sum with unaligned ends**:
//!
//! ```text
//! end_g = align(end_{g-1}, ALIGNMENT) + real_size_g      (end_{-1} = 0)
//! ```
//!
//! * m-grouped (`ALIGNMENT` = the mk alignment, 128 here): `A` is
//!   `[M_psum, K]` K-major where `M_psum = align(end_last, 128)`; the rows in
//!   `[end_g, align(end_g, 128))` are per-group padding that MUST be zero for
//!   the "ensure zero padding" guarantee (they then produce exact zeros in
//!   `D`); `B` is `[G, N, K]` K-major; `D` is `[M_psum, N]` (BF16 or FP32).
//! * k-grouped (`ALIGNMENT` = `k_alignment`, a multiple of 128): `A` is
//!   `[SUM_K, M]` MN-major and `B` is `[SUM_K, N]` MN-major where
//!   `SUM_K = sum_g align(real_k_g, k_alignment)`; the K rows in
//!   `[end_g, align(end_g, k_alignment))` must be zero (upstream zero-fills
//!   the whole buffer first); `D` is `[G, M, N]`, optionally accumulated
//!   (`D += A_g^T B_g`, FP32 reduce-add — the weight-grad contract).
//!
//! # TU composition
//!
//! SM100 bodies compile with `BMK_BNK` alone (prelude auto-prepended by the
//! JIT); the SM90 body references the wgmma layer, so its unit is
//! `WGMMA_H + BMK_BNK` (same rule as `sm90.rs` / `api_hc_prenorm.rs`).

use crate::device::{DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, Major, Operand, Output};

/// The BMK/BNK + psum kernel translation unit.
pub const BMK_BNK: &str = include_str!("../kernels/bmk_bnk.cu");

/// SM90 unit: wgmma.h first (the SM90 body references the wgmma layer), then
/// this file's kernels.  The SM90 arch guard is `900 <= arch < 1000`, so the
/// same unit also compiles under 100a with the SM90 body dead.
pub fn sm90_unit() -> String {
    format!("{}\n{}", jit::kernel_src::WGMMA_H, BMK_BNK)
}

// ---------------------------------------------------------------------------
// Fixed tiling (both upstream bmk launchers): BLOCK 128x128x64, 128B swizzle.
// ---------------------------------------------------------------------------
const BLOCK_M: u32 = 128;
const BLOCK_N: u32 = 128;
const BLOCK_K: u32 = 64;
const SWIZZLE_AB: u32 = 128;
const SWIZZLE_CD: u32 = 128;
const NUM_THREADS_SM100: u32 = 128;
const NUM_TMA_THREADS_SM90: u32 = 128;
const NUM_MATH_THREADS_SM90: u32 = 256;

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn align_up(a: u32, b: u32) -> u32 {
    ceil_div(a, b) * b
}

/// Upstream `get_swizzle_mode(inner_dim, elem_size)` (heuristics): the first
/// mode of {128, 64, 32} that covers `inner * elem` bytes, else 0.
fn swizzle_mode(inner_dim: u32, elem_size: u32) -> u32 {
    let bytes = (inner_dim * elem_size).min(128);
    match bytes {
        0..=16 => 0,
        17..=32 => 32,
        33..=64 => 64,
        _ => 128,
    }
}

fn require_arch(dev: &Device, sm100: bool) -> DgResult<()> {
    let ok = if sm100 {
        matches!(dev.arch, crate::device::Arch::Sm100)
    } else {
        matches!(dev.arch, crate::device::Arch::Sm90)
    };
    if !ok {
        return Err(DgError::Unsupported(format!(
            "bmk_bnk_mn_sm{} requires SM{}; this device is {:?}",
            if sm100 { 100 } else { 90 },
            if sm100 { 100 } else { 90 },
            dev.arch
        )));
    }
    Ok(())
}

fn smem_capacity_of(dev: &Device) -> u32 {
    if dev.smem_capacity > 0 {
        dev.smem_capacity
    } else {
        crate::heuristics::SMEM_CAPACITY_FALLBACK
    }
}

// ---------------------------------------------------------------------------
// Tile chooser — private helper mirroring the upstream launch configs
// (csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp + sm90_bmk_bnk_mn.hpp):
// block = (128, 128, 64) fixed; split factor from the SM count; stage count
// from 4 downward while the SMEM budget fits (upstream: "we select 4 as
// start, as it is tested to be faster than values > 4").
// ---------------------------------------------------------------------------
struct BmkTile {
    split_factor: u32,
    num_stages: u32,
    smem_size: u32,
    num_mn_blocks: u32,
    grid_x: u32,
}

fn bmk_tile(s: u32, m: u32, n: u32, k: u32, num_sms: u32, smem_capacity: u32, sm100: bool) -> DgResult<BmkTile> {
    if k % BLOCK_K != 0 {
        return Err(DgError::InvalidArg(format!(
            "bmk_bnk: k = {k} must be a multiple of {BLOCK_K}"
        )));
    }
    if m % 64 != 0 || n % 64 != 0 {
        return Err(DgError::InvalidArg(format!(
            "bmk_bnk: m = {m}, n = {n} must be multiples of 64"
        )));
    }
    if u64::from(s) * u64::from(m.max(n)) > i32::MAX as u64 {
        return Err(DgError::InvalidArg(
            "bmk_bnk: s * max(m, n) exceeds int32 (kernel index space)".into(),
        ));
    }
    // Swizzles are structurally fixed (the kernels assert them).
    debug_assert_eq!(swizzle_mode(BLOCK_K, 2), SWIZZLE_AB);
    debug_assert_eq!(swizzle_mode(BLOCK_N, 4), SWIZZLE_CD);

    let num_mn_blocks = ceil_div(m, BLOCK_M) * ceil_div(n, BLOCK_N);
    let num_sk_blocks = s * (k / BLOCK_K);
    let split_factor = ceil_div(num_sk_blocks, (num_sms / num_mn_blocks).max(1));

    let smem_a_per_stage = BLOCK_M * BLOCK_K * 2;
    let smem_b_per_stage = BLOCK_N * BLOCK_K * 2;
    let mut num_stages = 4u32;
    let mut smem_size;
    loop {
        // SM100 additionally stages the TMA-reduce-add CD tiles (2 stages of
        // BLOCK_M * swizzle_cd bytes) plus the 4-byte TMEM base slot; the
        // barrier block is the upstream `num_stages * 8 * 3 + 2 * 8 * 2 + 8`
        // (3 barrier words per stage budgeted, tmem_full + 2 epilogue words,
        // plus one spare word).
        let barriers = if sm100 {
            num_stages * 8 * 3 + 2 * 8 * 2 + 8
        } else {
            num_stages * 8 * 2
        };
        let smem_cd = if sm100 { BLOCK_M * SWIZZLE_CD * 2 } else { 0 };
        let smem_tmem_ptr = if sm100 { 4 } else { 0 };
        smem_size = smem_cd + (smem_a_per_stage + smem_b_per_stage) * num_stages + barriers + smem_tmem_ptr;
        if smem_size <= smem_capacity || num_stages == 1 {
            break;
        }
        num_stages -= 1;
    }
    if smem_size > smem_capacity {
        return Err(DgError::Unsupported(
            "bmk_bnk: shared memory too small for one stage".into(),
        ));
    }
    let grid_x = num_mn_blocks * ceil_div(num_sk_blocks, split_factor.max(1));
    Ok(BmkTile {
        split_factor,
        num_stages,
        smem_size,
        num_mn_blocks,
        grid_x,
    })
}

// ---------------------------------------------------------------------------
// Body generators (public for compile-check tests / bench wiring)
// ---------------------------------------------------------------------------

/// `__dg_kernel` wrapper for the SM100 bmk variant (mirrors the JIT body of
/// upstream `sm100_bmn_bnk_mn_gemm`).
pub fn bmk_sm100_body(m: u32, n: u32, k: u32, split_factor: u32, num_stages: u32) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_s,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d) {{
    dg::bmk_bnk_mn_sm100_impl<{m}, {n}, {k}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K},
                              {split_factor}, {SWIZZLE_AB}, {SWIZZLE_CD}, {num_stages}, {NUM_THREADS_SM100}>
        (shape_s, tma_a, tma_b, tma_d);
}}"#
    )
}

/// `__dg_kernel` wrapper for the SM90 bmk variant.
pub fn bmk_sm90_body(m: u32, n: u32, k: u32, split_factor: u32, num_stages: u32) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_s,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    float* d) {{
    dg::bmk_bnk_mn_sm90_impl<{m}, {n}, {k}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K},
                             {split_factor}, {num_stages}, {NUM_TMA_THREADS_SM90}, {NUM_MATH_THREADS_SM90}>
        (shape_s, tma_a, tma_b, d);
}}"#
    )
}

/// `__dg_kernel` wrapper for the psum grouped-GEMM variant.
/// `gemm_type` is the upstream `GemmType` code (5 = m-grouped psum,
/// 6 = k-grouped psum).
#[allow(clippy::too_many_arguments)]
pub fn psum_body(
    gemm_type: u32,
    num_groups: u32,
    num_stages: u32,
    accumulate: bool,
    cd_is_float: bool,
    k_alignment: u32,
    num_sms: u32,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_cd) {{
    dg::gemm_psum_impl<{BLOCK_M}, {BLOCK_N}, {BLOCK_K}, {num_groups},
                       {SWIZZLE_AB}, {SWIZZLE_CD}, {num_stages}, 256,
                       (dg::psum::GemmType){gemm_type}, {accumulate}, {cd_float},
                       {k_alignment}, {num_sms}, false>
        (shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd);
}}"#,
        accumulate = accumulate as u32,
        cd_float = cd_is_float as u32,
    )
}

// ---------------------------------------------------------------------------
// bmk, bnk -> mn (both architectures)
// ---------------------------------------------------------------------------

/// Validate the shared bmk operand contract (upstream `einsum::bmk_bnk_mn`).
fn validate_bmk(a: &Operand, b: &Operand, d: &Output, s: u32, m: u32, n: u32, k: u32) -> DgResult<()> {
    if a.dtype != Dtype::Bf16 || b.dtype != Dtype::Bf16 {
        return Err(DgError::InvalidArg("bmk_bnk: A/B must be BF16".into()));
    }
    if a.major != Major::K || b.major != Major::K {
        return Err(DgError::InvalidArg("bmk_bnk: A/B must be K-major".into()));
    }
    if d.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg(
            "bmk_bnk: D must be FP32 (the split-K accumulation target)".into(),
        ));
    }
    if d.rows != m || d.cols != n {
        return Err(DgError::InvalidArg("bmk_bnk: D must be [m, n]".into()));
    }
    let need_a = s as usize * m as usize * k as usize * 2;
    let need_b = s as usize * n as usize * k as usize * 2;
    let need_d = m as usize * n as usize * 4;
    if a.data.len < need_a {
        return Err(DgError::InvalidArg(format!(
            "bmk_bnk: A buffer {} B < required {} B for s={s}",
            a.data.len, need_a
        )));
    }
    if b.data.len < need_b {
        return Err(DgError::InvalidArg(format!(
            "bmk_bnk: B buffer {} B < required {} B for s={s}",
            b.data.len, need_b
        )));
    }
    if d.data.len < need_d {
        return Err(DgError::InvalidArg(format!(
            "bmk_bnk: D buffer {} B < required {} B",
            d.data.len, need_d
        )));
    }
    Ok(())
}

/// Run one bmk variant (shared validation + maps + launch).
fn run_bmk(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &Output,
    s: u32,
    sm100: bool,
) -> DgResult<()> {
    require_arch(dev, sm100)?;
    let (m, n, k) = (a.rows, b.rows, a.k);
    if b.k != k {
        return Err(DgError::InvalidArg("bmk_bnk: A/B K mismatch".into()));
    }
    validate_bmk(a, b, d, s, m, n, k)?;
    if s == 0 || m == 0 || n == 0 || k == 0 {
        return Ok(()); // nothing to accumulate
    }

    let tile = bmk_tile(s, m, n, k, dev.num_sms, smem_capacity_of(dev), sm100)?;
    // Every output tile is owned by at least one sk-block (split-K grid).
    debug_assert!(tile.grid_x >= tile.num_mn_blocks);
    if crate::heuristics::print_configs_enabled() {
        println!(
            "bmk_bnk_mn_sm{arch}: s={s}, m={m}, n={n}, k={k} -> block=(128, 128, 64), \
             split_k={sp}, stages={st}, smem={sm}, swizzle_ab=128, swizzle_cd=128, grid={g}",
            arch = if sm100 { 100 } else { 90 },
            sp = tile.split_factor,
            st = tile.num_stages,
            sm = tile.smem_size,
            g = tile.grid_x,
        );
    }

    // A: [k, s*m] K-major box [BLOCK_K, BLOCK_M]; B: [k, s*n]; D: [n, m].
    let stride_a = if a.outer_stride != 0 { a.outer_stride } else { k };
    let stride_b = if b.outer_stride != 0 { b.outer_stride } else { k };
    let tm_a = tma::make_tma_ab(
        dev, Dtype::Bf16, Major::K, &a.data, s * m, k, BLOCK_M, BLOCK_K, stride_a, 1,
        SWIZZLE_AB, false,
    )?;
    let tm_b = tma::make_tma_ab(
        dev, Dtype::Bf16, Major::K, &b.data, s * n, k, BLOCK_N, BLOCK_K, stride_b, 1,
        SWIZZLE_AB, false,
    )?;
    let row_stride = if d.stride != 0 { d.stride } else { d.cols };
    let tm_d = tma::make_tma_cd(
        dev, Dtype::F32, &d.data, m, n, BLOCK_M, SWIZZLE_CD / 4, row_stride, 1, SWIZZLE_CD,
    )?;

    let (body, tag, threads) = if sm100 {
        (
            bmk_sm100_body(m, n, k, tile.split_factor, tile.num_stages),
            "bmk_bnk_sm100",
            NUM_THREADS_SM100,
        )
    } else {
        (
            bmk_sm90_body(m, n, k, tile.split_factor, tile.num_stages),
            "bmk_bnk_sm90",
            NUM_TMA_THREADS_SM90 + NUM_MATH_THREADS_SM90,
        )
    };
    let sig = format!("{:?}", (m, n, k, tile.split_factor, tile.num_stages));
    let tu = if sm100 {
        BMK_BNK.to_string()
    } else {
        sm90_unit()
    };
    let func = jit::get_kernel(dev, &tu, tag, &sig, &body)?;

    let mut args = Args::new()
        .u32(s)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_d);
    if !sm100 {
        args = args.devptr(d.data.ptr);
    }
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (tile.grid_x, 1, 1),
            block: (threads, 1, 1),
            smem: tile.smem_size,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

/// Batched cross-projection on SM100 (Blackwell):
/// `D[m, n] += sum_s sum_k A[s, m, k] * B[s, n, k]` — the MLA
/// `bmk, bnk -> mn` einsum, tcgen05 split-K with TMA reduce-add epilogue.
///
/// `A` is `[s, m, k]` and `B` is `[s, n, k]` (both contiguous K-major BF16;
/// set `a.rows = m`, `b.rows = n`, `a.k = b.k = k`, `outer_stride = k`).
/// **`d` is read-modify-write** (FP32 `[m, n]`): initialize it (e.g. zero, or
/// the C to accumulate) before calling.
#[allow(clippy::too_many_arguments)]
pub fn bf16_bmk_bnk_mn_sm100(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    shape_s: u32,
) -> DgResult<()> {
    run_bmk(dev, stream, a, b, d, shape_s, true)
}

/// Batched cross-projection on SM90 (Hopper): same contract as
/// [`bf16_bmk_bnk_mn_sm100`], implemented with wgmma + `red.global.add.v2.f32`
/// (float2 global atomics) instead of the TMA reduce-add epilogue.
#[allow(clippy::too_many_arguments)]
pub fn bf16_bmk_bnk_mn_sm90(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    shape_s: u32,
) -> DgResult<()> {
    run_bmk(dev, stream, a, b, d, shape_s, false)
}

// ---------------------------------------------------------------------------
// PsumLayout grouped GEMM
// ---------------------------------------------------------------------------

/// Validate a psum layout (prefix-sum encoding): non-decreasing unaligned
/// ends, every aligned end within the physical span.
fn validate_psum_layout(layout: &[i32], span: u32, alignment: u32, what: &str) -> DgResult<()> {
    if layout.is_empty() {
        return Err(DgError::InvalidArg(format!("{what}: empty layout")));
    }
    let mut prev_end = 0u32;
    for (g, &e) in layout.iter().enumerate() {
        if e < 0 {
            return Err(DgError::InvalidArg(format!(
                "{what}: negative end {e} at group {g}"
            )));
        }
        let end = e as u32;
        if end < prev_end {
            return Err(DgError::InvalidArg(format!(
                "{what}: layout must be non-decreasing (group {g}: {end} < {prev_end})"
            )));
        }
        if align_up(end, alignment) > span {
            return Err(DgError::InvalidArg(format!(
                "{what}: aligned end {} of group {g} exceeds the physical span {span}",
                align_up(end, alignment)
            )));
        }
        prev_end = end;
    }
    Ok(())
}

/// Psum tile chooser: fixed (128, 128, 64) BF16 tiling; stage count from 8
/// downward while the SMEM budget fits (CD staging 2 x 128 x 128 B + A/B
/// stages + (2 * stages + 2) mbarriers + the TMEM pointer slot).
fn psum_tile(smem_capacity: u32) -> DgResult<(u32, u32)> {
    let smem_cd = 2 * BLOCK_M * SWIZZLE_CD;
    let per_stage = (BLOCK_M + BLOCK_N) * BLOCK_K * 2;
    let mut num_stages = 8u32;
    let mut smem_size;
    loop {
        // Barriers: full/empty per stage + tmem_full + tmem_empty, plus the
        // 4-byte TMEM base pointer slot.
        let barriers = (2 * num_stages + 2) * 8 + 4;
        smem_size = smem_cd + per_stage * num_stages + barriers;
        if smem_size <= smem_capacity || num_stages == 1 {
            break;
        }
        num_stages -= 1;
    }
    if smem_size > smem_capacity {
        return Err(DgError::Unsupported(
            "gemm_psum: shared memory too small for one stage".into(),
        ));
    }
    Ok((num_stages, smem_size))
}

/// 3D D map for the k-grouped psum output `[G, M, N]` (n contiguous).
/// Local builder: the shared `tma::make_tma_cd_3d` derives the batch stride
/// from `cols` alone (only right for contiguous square layouts is not even
/// the issue — it ignores `rows * row_stride`), so mirror upstream
/// `make_tma_3d_desc(d, n, m, num_groups, ..., d.stride(1), d.stride(0), swz)`
/// exactly: dims [n, m, G], strides [row_stride, rows * row_stride].
#[allow(clippy::too_many_arguments)]
fn make_tma_cd_3d_groups(
    dev: &Device,
    dtype: Dtype,
    buf: &crate::device::DevBuffer,
    rows: u32,
    cols: u32,
    block_m: u32,
    block_n: u32,
    row_stride: u32,
    num_groups: u32,
    swizzle: u32,
) -> DgResult<sys::TensorMap> {
    let elem = dtype.elem_size() as u64;
    let stride0 = row_stride as u64 * elem;
    let stride1 = rows as u64 * row_stride as u64 * elem;
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        match dtype {
            Dtype::F32 => sys::tm_dtype_float32(),
            _ => sys::tm_dtype_bfloat16(),
        },
        3,
        buf.ptr as *mut _,
        &[cols as u64, rows as u64, num_groups as u64],
        &[stride0, stride1],
        &[block_n, block_m, 1],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// m-grouped contiguous GEMM with the psum layout (SM100):
/// for every group `g`, `D[rows of g, :] = A[rows of g, :] @ B[g]^T`.
///
/// `a`: K-major BF16 `[M_psum, K]` (`M_psum` = the padded span; per-group
/// padding rows in `[end_g, align(end_g, 128))` must be zero for the
/// zero-padding guarantee).  `b`: K-major BF16 `[G, N, K]` (`b.rows = N`).
/// `grouped_layout`: the host-side psum ends (uploaded by this call).
/// `d`: `[M_psum, N]` BF16 (or FP32), `d.rows = a.rows`.
#[allow(clippy::too_many_arguments)]
pub fn bf16_m_grouped_gemm_nt_psum(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    grouped_layout: &[i32],
    d: &mut Output,
) -> DgResult<()> {
    require_arch(dev, true)?;
    if a.dtype != Dtype::Bf16 || b.dtype != Dtype::Bf16 {
        return Err(DgError::InvalidArg("gemm_psum: A/B must be BF16".into()));
    }
    if a.major != Major::K || b.major != Major::K {
        return Err(DgError::InvalidArg(
            "m-grouped psum requires K-major A/B".into(),
        ));
    }
    if d.dtype != Dtype::Bf16 && d.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg(
            "m-grouped psum: D must be BF16 or FP32".into(),
        ));
    }
    let (m_psum, n, k) = (a.rows, b.rows, a.k);
    if b.k != k {
        return Err(DgError::InvalidArg("gemm_psum: A/B K mismatch".into()));
    }
    if d.cols != n || d.rows != m_psum {
        return Err(DgError::InvalidArg(
            "m-grouped psum: D must be [M_psum, N] with M_psum = a.rows".into(),
        ));
    }
    if m_psum % BLOCK_M != 0 {
        return Err(DgError::InvalidArg(format!(
            "m-grouped psum: M_psum = {m_psum} must be a multiple of {BLOCK_M} (the psum gap alignment)"
        )));
    }
    if n % 64 != 0 || k % BLOCK_K != 0 {
        return Err(DgError::InvalidArg(format!(
            "m-grouped psum: n = {n} must be % 64, k = {k} must be % {BLOCK_K}"
        )));
    }
    let num_groups = grouped_layout.len() as u32;
    validate_psum_layout(grouped_layout, m_psum, BLOCK_M, "m-grouped psum")?;
    let need_b = num_groups as usize * n as usize * k as usize * 2;
    if b.data.len < need_b {
        return Err(DgError::InvalidArg(format!(
            "m-grouped psum: B buffer {} B < required {need_b} B",
            b.data.len
        )));
    }
    if m_psum == 0 || n == 0 || k == 0 || num_groups == 0 {
        return Ok(());
    }

    let (num_stages, smem_size) = psum_tile(smem_capacity_of(dev))?;
    if crate::heuristics::print_configs_enabled() {
        println!(
            "gemm_psum[m-grouped]: groups={num_groups}, m={m_psum}, n={n}, k={k} -> \
             block=(128, 128, 64), stages={num_stages}, smem={smem_size}"
        );
    }

    let stride_a = if a.outer_stride != 0 { a.outer_stride } else { k };
    let stride_b = if b.outer_stride != 0 { b.outer_stride } else { k };
    let tm_a = tma::make_tma_ab(
        dev, Dtype::Bf16, Major::K, &a.data, m_psum, k, BLOCK_M, BLOCK_K, stride_a, 1,
        SWIZZLE_AB, false,
    )?;
    // B is per-group: the group offset rides the outer dim (g * n + n_idx).
    let tm_b = tma::make_tma_ab(
        dev, Dtype::Bf16, Major::K, &b.data, n, k, BLOCK_N, BLOCK_K, stride_b, num_groups,
        SWIZZLE_AB, false,
    )?;
    let cd_elem = d.dtype.elem_size() as u32;
    let row_stride = if d.stride != 0 { d.stride } else { d.cols };
    let tm_cd = tma::make_tma_cd(
        dev, d.dtype, &d.data, m_psum, n, BLOCK_M, SWIZZLE_CD / cd_elem, row_stride, 1,
        SWIZZLE_CD,
    )?;

    let body = psum_body(5, num_groups, num_stages, false, d.dtype == Dtype::F32, BLOCK_M, dev.num_sms);
    let sig = format!(
        "{:?}",
        (5, num_groups, num_stages, d.dtype == Dtype::F32, k, n)
    );
    let func = jit::get_kernel(dev, BMK_BNK, "gemm_psum_mg", &sig, &body)?;

    let gl = crate::device::alloc_and_upload(dev, grouped_layout, stream.raw())?;
    let args = Args::new()
        .u32(m_psum)
        .u32(n)
        .u32(k)
        .devptr(gl.ptr)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_cd);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (256, 1, 1),
            smem: smem_size,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

/// k-grouped contiguous (weight-grad) GEMM with the psum layout (SM100):
/// `D[g] (+)= A_g^T @ B_g` where `A` is `[SUM_K, M]` MN-major, `B` is
/// `[SUM_K, N]` MN-major and group `g` physically occupies the K rows
/// `[align(end_{g-1}, k_alignment), end_g)` (zero tails).
///
/// `grouped_layout`: host-side psum ends. `d`: `[G, M, N]` — FP32 when
/// `accumulate` (TMA reduce-add onto the existing D/C), else FP32 or BF16
/// direct store. `k_alignment`: the layout's alignment (multiple of 128).
#[allow(clippy::too_many_arguments)]
pub fn bf16_k_grouped_gemm_tn_psum(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    grouped_layout: &[i32],
    k_alignment: u32,
    d: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    require_arch(dev, true)?;
    if a.dtype != Dtype::Bf16 || b.dtype != Dtype::Bf16 {
        return Err(DgError::InvalidArg("gemm_psum: A/B must be BF16".into()));
    }
    if a.major != Major::Mn || b.major != Major::Mn {
        return Err(DgError::InvalidArg(
            "k-grouped psum requires MN-major A/B ([SUM_K, M] / [SUM_K, N])".into(),
        ));
    }
    if k_alignment % 128 != 0 {
        return Err(DgError::InvalidArg(format!(
            "k-grouped psum: k_alignment = {k_alignment} must be a multiple of 128"
        )));
    }
    if accumulate && d.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg(
            "k-grouped psum: accumulation requires FP32 D (TMA reduce-add)".into(),
        ));
    }
    let (m, n, sum_k) = (a.rows, b.rows, a.k);
    if b.k != sum_k {
        return Err(DgError::InvalidArg("gemm_psum: A/B K mismatch".into()));
    }
    if d.rows != m || d.cols != n {
        return Err(DgError::InvalidArg(
            "k-grouped psum: D must describe one [M, N] group slice".into(),
        ));
    }
    if m % 64 != 0 || n % 64 != 0 {
        return Err(DgError::InvalidArg(format!(
            "k-grouped psum: m = {m}, n = {n} must be multiples of 64"
        )));
    }
    let num_groups = grouped_layout.len() as u32;
    validate_psum_layout(grouped_layout, sum_k, k_alignment, "k-grouped psum")?;
    let need_d = num_groups as usize * m as usize * n as usize * d.dtype.elem_size();
    if d.data.len < need_d {
        return Err(DgError::InvalidArg(format!(
            "k-grouped psum: D buffer {} B < required {need_d} B for {num_groups} groups",
            d.data.len
        )));
    }
    if m == 0 || n == 0 || sum_k == 0 || num_groups == 0 {
        return Ok(());
    }

    let (num_stages, smem_size) = psum_tile(smem_capacity_of(dev))?;
    if crate::heuristics::print_configs_enabled() {
        println!(
            "gemm_psum[k-grouped]: groups={num_groups}, m={m}, n={n}, sum_k={sum_k}, \
             align={k_alignment}, acc={accumulate} -> block=(128, 128, 64), \
             stages={num_stages}, smem={smem_size}"
        );
    }

    // MN-major operands: inner = MN, outer = K; the group offset rides the
    // outer (K) coordinate via the scheduler's `current_k_start`.
    let stride_a = if a.outer_stride != 0 { a.outer_stride } else { m };
    let stride_b = if b.outer_stride != 0 { b.outer_stride } else { n };
    let tm_a = tma::make_tma_ab(
        dev, Dtype::Bf16, Major::Mn, &a.data, m, sum_k, BLOCK_M, BLOCK_K, stride_a, 1,
        SWIZZLE_AB, false,
    )?;
    let tm_b = tma::make_tma_ab(
        dev, Dtype::Bf16, Major::Mn, &b.data, n, sum_k, BLOCK_N, BLOCK_K, stride_b, 1,
        SWIZZLE_AB, false,
    )?;
    let cd_elem = d.dtype.elem_size() as u32;
    let row_stride = if d.stride != 0 { d.stride } else { d.cols };
    let tm_cd = make_tma_cd_3d_groups(
        dev, d.dtype, &d.data, m, n, BLOCK_M, SWIZZLE_CD / cd_elem, row_stride, num_groups,
        SWIZZLE_CD,
    )?;

    let body = psum_body(
        6,
        num_groups,
        num_stages,
        accumulate,
        d.dtype == Dtype::F32,
        k_alignment,
        dev.num_sms,
    );
    let sig = format!(
        "{:?}",
        (
            6,
            num_groups,
            num_stages,
            accumulate,
            d.dtype == Dtype::F32,
            k_alignment,
            sum_k
        )
    );
    let func = jit::get_kernel(dev, BMK_BNK, "gemm_psum_kk", &sig, &body)?;

    let gl = crate::device::alloc_and_upload(dev, grouped_layout, stream.raw())?;
    let args = Args::new()
        .u32(m)
        .u32(n)
        .u32(sum_k)
        .devptr(gl.ptr)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_cd);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (256, 1, 1),
            smem: smem_size,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Upstream reference shape family: s in {129, 4096, 8192},
    // (m, n, k) in {(128, 384, 128), (256, 256, 256), (384, 128, 384)}.
    #[test]
    fn bmk_tile_matches_upstream_formulas() {
        // SM100, (s=4096, m, n, k=256): B200 (148 SMs, 232448 B smem).
        // num_mn_blocks = ceil(m/128) * ceil(n/128); num_sk = 4096*4.
        // Stages stay at 4 (smem = 32768 + 32768*4 + 136 = 163980 <= 232448).
        for &(m, n, num_mn) in &[(128u32, 384u32, 3u32), (256, 256, 4), (384, 128, 3)] {
            let t = bmk_tile(4096, m, n, 256, 148, 232448, true).unwrap();
            assert_eq!(t.num_mn_blocks, num_mn);
            assert_eq!(t.num_stages, 4);
            assert_eq!(t.split_factor, ceil_div(4096 * 4, (148 / num_mn).max(1)));
            assert_eq!(t.grid_x, num_mn * ceil_div(4096 * 4, t.split_factor));
            // Upstream smem formula, verbatim.
            assert_eq!(
                t.smem_size,
                128 * 128 * 2 + (128 * 64 * 2 + 128 * 64 * 2) * 4 + (4 * 8 * 3 + 2 * 8 * 2 + 8) + 4
            );
        }
        // SM90: no CD staging, barrier block = stages*8*2.
        let t = bmk_tile(4096, 256, 256, 256, 132, 232448, false).unwrap();
        assert_eq!(t.num_stages, 4);
        assert_eq!(t.smem_size, (128 * 64 * 2 + 128 * 64 * 2) * 4 + 4 * 8 * 2);
        // Tiny SMEM forces fewer stages: 2 stages need 98396 B <= 100000,
        // 3 would need 131188 B.
        let t = bmk_tile(129, 128, 128, 128, 148, 100_000, true).unwrap();
        assert_eq!(t.num_stages, 2);
        // Degenerate shapes are rejected like upstream's host asserts.
        assert!(bmk_tile(129, 100, 128, 128, 148, 232448, true).is_err()); // m % 64
        assert!(bmk_tile(129, 128, 128, 100, 148, 232448, true).is_err()); // k % 64
    }

    #[test]
    fn bmk_tile_split_k_covers_all_slices() {
        // Every (mn_block, slice) pair must be covered exactly once: the
        // sk-blocks tile the num_sk_blocks slice space disjointly.
        let (s, m, n, k) = (8192u32, 384u32, 128u32, 384u32);
        let t = bmk_tile(s, m, n, k, 148, 232448, true).unwrap();
        let num_sk = s * (k / BLOCK_K);
        let mut covered = vec![false; num_sk as usize];
        for sk_block in 0..ceil_div(num_sk, t.split_factor) {
            let base = sk_block * t.split_factor;
            // Upstream: num_total_stages = min(kSplitFactor, remaining).
            let stages = t.split_factor.min(num_sk - base);
            for i in 0..stages {
                assert!(!covered[(base + i) as usize], "slice covered twice");
                covered[(base + i) as usize] = true;
            }
        }
        assert!(covered.iter().all(|c| *c), "every slice covered");
        assert_eq!(t.grid_x, t.num_mn_blocks * ceil_div(num_sk, t.split_factor));
    }

    #[test]
    fn psum_tile_budget() {
        // 6 stages fit B200 (229476 B); 7 would need 262260 B.
        let (stages, smem) = psum_tile(232448).unwrap();
        assert_eq!(stages, 6);
        assert_eq!(
            smem,
            2 * 128 * 128 + (128 + 128) * 64 * 2 * 6 + (2 * 6 + 2) * 8 + 4
        );
        // A tiny capacity falls back to 1 stage (65580 B), not an error.
        let (stages, smem) = psum_tile(70_000).unwrap();
        assert_eq!(stages, 1);
        assert!(smem <= 70_000);
    }

    #[test]
    fn psum_layout_validation() {
        let ok = [128, 300, 500];
        assert!(validate_psum_layout(&ok, 512, 128, "t").is_ok());
        // Decreasing end.
        assert!(validate_psum_layout(&[300, 128], 512, 128, "t").is_err());
        // Aligned end beyond the span: 300 -> align 384 ok; 600 -> 640 > 512.
        assert!(validate_psum_layout(&[600], 512, 128, "t").is_err());
        assert!(validate_psum_layout(&[], 512, 128, "t").is_err());
    }
}
