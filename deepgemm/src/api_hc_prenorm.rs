//! TF32 hyperconnection pre-norm GEMM launchers (SM90 + SM100).
//!
//! Port of upstream `sm90_tf32_hc_prenorm_gemm` / `sm100_tf32_hc_prenorm_gemm`
//! (see `kernels/hc_prenorm.cu` for the full concept notes).
//!
//! # Math contract
//! `A` is BF16 `[m, k]` (K-major, the hidden state), `B` is FP32 `[n, k]`
//! (K-major, the projection weights), with `n` tiny (the hyperconnection
//! width) and `k` huge — so the kernel is **split-K**:
//!
//! ```text
//! D[s, m, n]    = sum_{k in split s} tf32(A[m,k]) * tf32(B[n,k])   (FP32)
//! sqr_sum[s, m] = sum_{k in split s} A[m,k]^2                      (FP32, exact)
//! ```
//!
//! The caller reduces over `s` and applies the pre-norm that hyperconnection
//! needs: `y[m, :] = (sum_s D[s, m, :]) / sqrt(sum_s sqr_sum[s, m])`.
//!
//! # Buffer shapes
//! * `num_splits == 1`: `d` is `[m, n]` FP32, `sqr_sum` is `[m]` FP32.
//! * `num_splits > 1`: `d.data` holds `[num_splits, m, n]` FP32
//!   (`d.rows = m`, `d.cols = n` still describe ONE split), and `sqr_sum`
//!   holds `[num_splits, m]` FP32.
//!
//! # Constraints (upstream-faithful)
//! `k % 64 == 0`, `0 < n <= 32`, `n % 8 == 0` (the kernel's D-swizzle
//! static-asserts `swizzle_cd/4 == block_n`, which pins `block_n` to
//! {16, 32}; the upstream SM100 launcher advertises `n <= 128` but its own
//! kernel cannot instantiate beyond 32 — we surface that as a host error).

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, Major, Operand, Output};

/// Fixed tiling constants of both upstream hc-prenorm launchers.
const BLOCK_M: u32 = 64;
const BLOCK_K: u32 = 64;
const NUM_MMA_THREADS: u32 = 128;
const NUM_AUX_THREADS: u32 = 128; // TMA group (SM90) / cast+reduce group (SM100)
const MAX_N: u32 = 32;

// ---------------------------------------------------------------------------
// Tile choice — private helper mirroring the upstream launchers
// (`sm90_tf32_hc_prenorm_gemm.hpp` / `sm100_tf32_hc_prenorm_gemm.hpp`):
// block_m = 64, block_k = 64, block_n = align(n, 16), num_stages from 12
// downward while the SMEM budget exceeds the architecture capacity.
// ---------------------------------------------------------------------------

/// Upstream `heuristics::get_swizzle_mode`: first mode of {128, 64, 32, 16}
/// whose byte width divides `block_size * elem_size`.
fn swizzle_mode(block_size: u32, elem_size: u32) -> u32 {
    for mode in [128u32, 64, 32, 16] {
        if (block_size * elem_size) % mode == 0 {
            return mode;
        }
    }
    unreachable!("block_size * elem_size is divisible by 16")
}

struct HcTile {
    block_n: u32,
    swizzle_cd: u32,
    num_stages: u32,
    /// Bytes the kernel actually touches (stage budget accounting).
    smem_size: u32,
}

fn pick_tile(n: u32, sm100: bool, smem_capacity: u32) -> DgResult<HcTile> {
    let block_n = n.div_ceil(16) * 16; // upstream `align(n, 16)`
    let swizzle_cd = swizzle_mode(block_n, 4);
    // The kernel's D staging asserts kSwizzleCDMode/4 == BLOCK_N, which only
    // holds when BLOCK_N spans exactly one swizzle atom ({16, 32}).
    if swizzle_cd != block_n * 4 {
        return Err(DgError::InvalidArg(format!(
            "hc_prenorm: n = {n} unsupported (block_n = {block_n} does not map to a single TMA swizzle atom)"
        )));
    }
    let smem_a = BLOCK_M * BLOCK_K * 2; // bf16 A stage
    let smem_b = block_n * BLOCK_K * 4; // fp32 B stage
    let smem_cd = BLOCK_M * swizzle_cd;
    let mut num_stages = 12u32;
    let mut smem_size = 0u32;
    while num_stages > 0 {
        // SM90: full|empty barriers.  SM100: 4 barrier arrays + tmem_full
        // + the 4-byte TMEM base pointer slot.
        let barriers = if sm100 {
            (num_stages * 4 + 1) * 8 + 4
        } else {
            num_stages * 2 * 8
        };
        smem_size = (smem_a + smem_b) * num_stages + smem_cd + barriers;
        if smem_size <= smem_capacity {
            break;
        }
        num_stages -= 1;
    }
    if num_stages == 0 {
        return Err(DgError::InvalidArg(
            "hc_prenorm: shared memory too small for one stage".into(),
        ));
    }
    Ok(HcTile {
        block_n,
        swizzle_cd,
        num_stages,
        smem_size,
    })
}

// ---------------------------------------------------------------------------
// TMA maps
// ---------------------------------------------------------------------------

/// 3D D map for split-K partials: `[num_splits, m, n]` FP32 with the n dim
/// contiguous.  Mirrors upstream `make_tma_3d_desc(d, n, m, num_splits,
/// block_n, block_m, 1, d.stride(-2), d.stride(-3), swizzle_cd_mode)`.
/// (Local builder: the shared `tma::make_tma_cd_3d` computes the batch
/// stride from `cols` alone, which is only right for square outputs.)
#[allow(clippy::too_many_arguments)]
fn make_tma_d_3d(
    dev: &Device,
    buf: &DevBuffer,
    rows: u32,
    cols: u32,
    block_m: u32,
    block_n: u32,
    row_stride: u32,
    num_splits: u32,
    swizzle: u32,
) -> DgResult<sys::TensorMap> {
    let elem = 4u64; // FP32
    let stride0 = row_stride as u64 * elem;
    let stride1 = rows as u64 * row_stride as u64 * elem;
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        sys::tm_dtype_float32(),
        3,
        buf.ptr as *mut _,
        &[cols as u64, rows as u64, num_splits as u64],
        &[stride0, stride1],
        &[block_n, block_m, 1],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

// ---------------------------------------------------------------------------
// Shared validation + launch
// ---------------------------------------------------------------------------

fn require_arch(dev: &Device, sm100: bool) -> DgResult<()> {
    let ok = if sm100 {
        matches!(dev.arch, crate::device::Arch::Sm100)
    } else {
        matches!(dev.arch, crate::device::Arch::Sm90)
    };
    if !ok {
        return Err(DgError::Unsupported(format!(
            "hc_prenorm_sm{} requires SM{}; this device is {:?}",
            if sm100 { 100 } else { 90 },
            if sm100 { 100 } else { 90 },
            dev.arch
        )));
    }
    Ok(())
}

fn outer_stride_of(op: &Operand) -> u32 {
    if op.outer_stride != 0 {
        op.outer_stride
    } else {
        op.k
    }
}

// Upstream-mirroring signature: one argument per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
fn run_hc_prenorm(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    sqr_sum: &DevBuffer,
    num_splits: u32,
    sm100: bool,
) -> DgResult<()> {
    require_arch(dev, sm100)?;
    // Types (upstream: A bf16, B/D/sqr_sum fp32, all K-major AB, N-major D).
    if a.dtype != Dtype::Bf16 || a.major != Major::K {
        return Err(DgError::InvalidArg(
            "hc_prenorm: A must be K-major BF16".into(),
        ));
    }
    if b.dtype != Dtype::F32 || b.major != Major::K {
        return Err(DgError::InvalidArg(
            "hc_prenorm: B must be K-major FP32".into(),
        ));
    }
    if d.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg("hc_prenorm: D must be FP32".into()));
    }
    let (m, n, k) = (a.rows, b.rows, a.k);
    if b.k != k {
        return Err(DgError::InvalidArg("hc_prenorm: A/B K mismatch".into()));
    }
    if d.rows != m || d.cols != n {
        return Err(DgError::InvalidArg(
            "hc_prenorm: D shape must be [m, n]".into(),
        ));
    }
    if n == 0 || k == 0 {
        return Err(DgError::InvalidArg("hc_prenorm: empty problem".into()));
    }
    if n > MAX_N || n % 8 != 0 {
        return Err(DgError::InvalidArg(format!(
            "hc_prenorm: n = {n} must be in (0, {MAX_N}] and % 8 == 0"
        )));
    }
    if k % BLOCK_K != 0 {
        return Err(DgError::InvalidArg(format!(
            "hc_prenorm: k = {k} must be a multiple of {BLOCK_K}"
        )));
    }
    if num_splits == 0 {
        return Err(DgError::InvalidArg(
            "hc_prenorm: num_splits must be >= 1".into(),
        ));
    }
    let need_d = num_splits as usize * m as usize * n as usize * 4;
    if d.data.len < need_d {
        return Err(DgError::InvalidArg(format!(
            "hc_prenorm: D buffer {} B < required {} B for num_splits = {num_splits}",
            d.data.len, need_d
        )));
    }
    let need_s = num_splits as usize * m as usize * 4;
    if sqr_sum.len < need_s {
        return Err(DgError::InvalidArg(format!(
            "hc_prenorm: sqr_sum buffer {} B < required {} B",
            sqr_sum.len, need_s
        )));
    }
    if m == 0 {
        return Ok(()); // nothing to do (upstream early-out)
    }

    let capacity = if dev.smem_capacity > 0 {
        dev.smem_capacity
    } else {
        crate::heuristics::SMEM_CAPACITY_FALLBACK
    };
    let tile = pick_tile(n, sm100, capacity)?;
    let row_stride = if d.stride != 0 { d.stride } else { d.cols };

    // Config echo (mirrors the upstream DG_PRINT_CONFIGS printf).
    if std::env::var_os("DG_PRINT_CONFIGS").is_some() {
        println!(
            "hc_prenorm_sm{arch}: m={m}, n={n}, k={k} -> block=({BLOCK_M}, {bn}, {BLOCK_K}), \
             splits={num_splits}, stages={stages}, smem={smem}, swizzle_cd={swcd}",
            arch = if sm100 { 100 } else { 90 },
            bn = tile.block_n,
            stages = tile.num_stages,
            smem = tile.smem_size,
            swcd = tile.swizzle_cd,
        );
    }

    // A/B maps: 128B-swizzled K-major operand maps (A = one atom per stage,
    // B = two 32-fp32 atoms per stage — the kernel's hc_tma_copy splits).
    let tm_a = tma::make_tma_ab(
        dev,
        Dtype::Bf16,
        Major::K,
        &a.data,
        m,
        k,
        BLOCK_M,
        BLOCK_K,
        outer_stride_of(a),
        1,
        128,
        false,
    )?;
    let tm_b = tma::make_tma_ab(
        dev,
        Dtype::F32,
        Major::K,
        &b.data,
        n,
        k,
        tile.block_n,
        BLOCK_K,
        outer_stride_of(b),
        1,
        128,
        false,
    )?;
    // D map: one swizzle atom per store box (block_n == swizzle_cd / 4).
    let tm_d = if num_splits == 1 {
        tma::make_tma_cd(
            dev,
            Dtype::F32,
            &d.data,
            m,
            n,
            BLOCK_M,
            tile.block_n,
            row_stride,
            1,
            tile.swizzle_cd,
        )?
    } else {
        make_tma_d_3d(
            dev,
            &d.data,
            m,
            n,
            BLOCK_M,
            tile.block_n,
            row_stride,
            num_splits,
            tile.swizzle_cd,
        )?
    };

    // Kernel variant (compile-time N/K/splits/stages like upstream's JIT).
    let sig = format!(
        "{:?}",
        (
            n,
            k,
            tile.block_n,
            tile.swizzle_cd,
            num_splits,
            tile.num_stages
        )
    );
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, float* sqr_sum) {{
    dg::hc_prenorm_sm{arch}_impl<{n}, {k}, {bm}, {bn}, {bk}, {splits}, {swcd}, {stages}, {t1}, {t2}>
        (shape_m, tma_a, tma_b, tma_d, sqr_sum);
}}"#,
        arch = if sm100 { 100 } else { 90 },
        n = n,
        k = k,
        bm = BLOCK_M,
        bn = tile.block_n,
        bk = BLOCK_K,
        splits = num_splits,
        swcd = tile.swizzle_cd,
        stages = tile.num_stages,
        t1 = NUM_MMA_THREADS,
        t2 = NUM_AUX_THREADS,
    );
    // TU composition: the SM90 body needs the wgmma layer first; SM100 uses
    // the prelude (auto-prepended) + hc_prenorm.cu alone.
    let (tu, tag) = if sm100 {
        (jit::kernel_src::HC_PRENORM.to_string(), "hc_prenorm_sm100")
    } else {
        (
            format!(
                "{}\n{}",
                jit::kernel_src::WGMMA_H,
                jit::kernel_src::HC_PRENORM
            ),
            "hc_prenorm_sm90",
        )
    };
    let func = jit::get_kernel(dev, &tu, tag, &sig, &body)?;

    let args = Args::new()
        .u32(m)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_d)
        .devptr(sqr_sum.ptr);

    // Upstream launches with the full SMEM capacity as the dynamic size
    // (the kernel touches only `tile.smem_size` of it).
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (num_splits * m.div_ceil(BLOCK_M), 1, 1),
            block: (NUM_MMA_THREADS + NUM_AUX_THREADS, 1, 1),
            smem: capacity,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// TF32 hyperconnection pre-norm GEMM (Hopper, SM90a): WGMMA RS-form TF32.
///
/// Computes the split-K partials `D[s] = A @ B^T` (TF32 MMA, FP32 accum) and
/// `sqr_sum[s, m] = sum_k A[m, k]^2` in one pass; see the module docs for the
/// pre-norm reduction the caller applies. `sqr_sum` must hold
/// `num_splits * m` FP32 values; `d.data` must hold `num_splits * m * n`.
// Upstream-mirroring signature: one argument per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn hc_prenorm_sm90(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    sqr_sum: &DevBuffer,
    num_splits: u32,
) -> DgResult<()> {
    run_hc_prenorm(dev, stream, a, b, d, sqr_sum, num_splits, false)
}

/// TF32 hyperconnection pre-norm GEMM (Blackwell, SM100a): `tcgen05.mma
/// kind::tf32` TS-form with the A operand cast into TMEM by four
/// cast-and-square-reduce warps.
///
/// Same contract as [`hc_prenorm_sm90`].
// Upstream-mirroring signature: one argument per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn hc_prenorm_sm100(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    sqr_sum: &DevBuffer,
    num_splits: u32,
) -> DgResult<()> {
    run_hc_prenorm(dev, stream, a, b, d, sqr_sum, num_splits, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_choice_matches_upstream_budgets() {
        // n = 24 -> block_n = 32, swizzle 128; 12 stages fit both arches
        // (SM90: 204992 B, SM100: 205596 B <= 232448).
        for sm100 in [false, true] {
            let t = pick_tile(24, sm100, 232448).unwrap();
            assert_eq!(t.block_n, 32);
            assert_eq!(t.swizzle_cd, 128);
            assert_eq!(t.num_stages, 12);
            assert!(t.smem_size <= 232448);
        }
        // n = 16 -> swizzle 64 staging.
        let t = pick_tile(16, false, 232448).unwrap();
        assert_eq!(t.block_n, 16);
        assert_eq!(t.swizzle_cd, 64);
        assert_eq!(t.num_stages, 12);
        // n = 8 aligns to block_n = 16 (padding via TMA OOB).
        assert_eq!(pick_tile(8, true, 232448).unwrap().block_n, 16);
        // n = 40 -> block_n = 48: no single-atom swizzle -> rejected.
        assert!(pick_tile(40, false, 232448).is_err());
        // Tiny SMEM forces fewer stages: 5 stages need
        // 16384*5 + 8192 + 5*16 = 90192 B <= 100000, 6 would need 106592 B.
        let t = pick_tile(24, false, 100_000).unwrap();
        assert_eq!(t.num_stages, 5);
    }

    #[test]
    fn smem_budget_arithmetic() {
        // Exact mirrors of the upstream formulas, for regression-locking.
        let smem = |bn: u32, stages: u32, sm100: bool| {
            let sw = if bn == 32 { 128 } else { 64 };
            let barriers = if sm100 {
                (stages * 4 + 1) * 8 + 4
            } else {
                stages * 2 * 8
            };
            (8192 + bn * 256) * stages + 64 * sw + barriers
        };
        // SM100 barrier block: (kNumStages*4 + 1) mbarriers (full, full_cast,
        // empty, empty_cast, tmem_full) + the 4-byte tmem pointer slot.
        assert_eq!(smem(32, 12, false), 204_992);
        assert_eq!(smem(32, 12, true), 205_196);
        assert_eq!(smem(16, 12, false), 151_744);
        assert_eq!(smem(16, 12, true), 151_948);
    }
}
