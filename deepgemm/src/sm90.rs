//! SM90 (Hopper) kernels — the wgmma generation of this crate.
//!
//! * FP8 1D1D GEMM: fine-grained per-128-channel FP32 scaling, WGMMA, TMA
//!   multicast (cluster <= 2), persistent scheduling, and the
//!   K-grouped weight-grad flavor (runtime tensormap patching).
//! * BF16 GEMM: K/MN-major operands, m-grouped contiguous/masked, optional
//!   C accumulation, swizzled STSM epilogue.
//!
//! # Scale layout (1D1D)
//! `SfFp32` holds FP32 scales as `[kb, tma_aligned(mn)]` (k-block-major,
//! MN-contiguous rows, 16B-aligned) — the exact layout the SM90 TMA SF maps
//! describe. Build one with [`transpose_sf_fp32`] from an `(mn, k/128)`
//! row-major scale tensor.
//!
//! # K-grouped (weight-grad) data layout
//! "A/B stacked along K" means: for each group g (in order) a contiguous
//! `[mn, ks_g]` K-major tile, groups concatenated — group g's tile starts at
//! byte offset `k_start_g * mn`. The kernel patches its TMA descriptors per
//! group, so ONE persistent launch covers every group; D is
//! `[num_groups, m, n]` and each group accumulates into its own D slice via
//! TMA reduce-add (call repeatedly to accumulate further).

use crate::device::{alloc_and_upload, DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::heuristics::{self, GemmDesc};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, GemmType, Major, Operand, Output};

// ---------------------------------------------------------------------------
// FP32 scale tensor (SM90 1D1D layout)
// ---------------------------------------------------------------------------
pub struct SfFp32 {
    pub buf: DevBuffer,
    /// Logical MN (m for A-scales, n for B-scales).
    pub mn: u32,
    /// Number of 128-wide K blocks (concatenated across K groups if kk).
    pub k_blocks: u32,
}

/// Transpose `(mn, k/128)` row-major FP32 scales into the SM90 1D1D layout
/// `[kb, tma_aligned(mn)]` (upstream Python `get_col_major_tma_aligned_tensor`).
pub fn transpose_sf_fp32(
    dev: &Device,
    stream: &DevStream,
    src: &DevBuffer,
    mn: u32,
    k_blocks: u32,
) -> DgResult<SfFp32> {
    let tma_aligned = heuristics::tma_aligned_size(mn, 4);
    let out = DevBuffer::alloc(dev, (tma_aligned * k_blocks * 4) as usize)?;
    let body = r#"extern "C" __global__ void __dg_kernel(
    const float* in, float* out, unsigned mn, unsigned kb, unsigned ta) {
    dg::transpose_sf_fp32_impl<256>(in, out, mn, kb, ta);
}"#;
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::LAYOUT_QUANT,
        "transpose_sf_fp32",
        "v1",
        body,
    )?;
    let args = Args::new()
        .devptr(src.ptr)
        .devptr(out.ptr)
        .u32(mn)
        .u32(k_blocks)
        .u32(tma_aligned);
    let n = (tma_aligned * k_blocks).max(1);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (n.div_ceil(256), 1, 1),
            block: (256, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;
    Ok(SfFp32 {
        buf: out,
        mn,
        k_blocks,
    })
}

/// Build an `SfFp32` directly from host `(mn, k/128)` row-major data
/// (host transpose; fine for small tensors, `transpose_sf_fp32` for large).
pub fn sf_fp32_from_host(
    dev: &Device,
    stream: &DevStream,
    data: &[f32],
    mn: u32,
    k_blocks: u32,
) -> DgResult<SfFp32> {
    if data.len() != (mn as usize) * (k_blocks as usize) {
        return Err(DgError::InvalidArg(
            "sf_fp32 host data size mismatch".into(),
        ));
    }
    let src = alloc_and_upload(dev, data, stream.raw())?;
    transpose_sf_fp32(dev, stream, &src, mn, k_blocks)
}

fn require_sm90(dev: &Device) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm90) {
        return Err(DgError::Unsupported(format!(
            "the wgmma kernels require SM90 (Hopper); this device is {:?}",
            dev.arch
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FP8 1D1D (shared by Normal and K-grouped)
// ---------------------------------------------------------------------------
struct Fp8Inputs<'a> {
    a: &'a Operand,
    b: &'a Operand,
    sfa: &'a SfFp32,
    sfb: &'a SfFp32,
}

#[allow(clippy::too_many_arguments)]
fn run_fp8_1d1d(
    dev: &Device,
    stream: &DevStream,
    x: &Fp8Inputs,
    d: &mut Output,
    ks: Option<&[u32]>, // Some => K-grouped weight-grad
) -> DgResult<()> {
    require_sm90(dev)?;
    if x.a.dtype != Dtype::Fp8 || x.b.dtype != Dtype::Fp8 {
        return Err(DgError::InvalidArg("fp8_1d1d requires FP8 operands".into()));
    }
    if x.a.major != Major::K || x.b.major != Major::K {
        return Err(DgError::InvalidArg(
            "fp8_1d1d requires K-major operands".into(),
        ));
    }
    if d.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg("fp8_1d1d outputs FP32".into()));
    }
    if d.cols != x.b.rows || d.rows != x.a.rows {
        return Err(DgError::InvalidArg("output shape mismatch".into()));
    }
    let k = x.a.k;
    if x.b.k != k {
        return Err(DgError::InvalidArg("A/B K mismatch".into()));
    }
    let sum_kb = x.sfa.k_blocks;
    let want_kb = k.div_ceil(128);
    if sum_kb != x.sfb.k_blocks || sum_kb != want_kb {
        return Err(DgError::InvalidArg(
            "scale K blocks must equal K/128".into(),
        ));
    }
    if x.sfa.mn != x.a.rows || x.sfb.mn != x.b.rows {
        return Err(DgError::InvalidArg("scale MN mismatch".into()));
    }

    let is_kk = ks.is_some();
    let num_groups = ks.map(|v| v.len() as u32).unwrap_or(1);
    let (first_k, _total_k) = if is_kk {
        let ks = ks.unwrap();
        let mut first_k = 0u32;
        let mut sum = 0u32;
        for &g in ks {
            if g % 128 != 0 {
                return Err(DgError::InvalidArg("kk group K must be % 128".into()));
            }
            if first_k == 0 && g != 0 {
                first_k = g;
            }
            sum += g;
        }
        if sum != k {
            return Err(DgError::InvalidArg("sum(ks) must equal A.k".into()));
        }
        if first_k == 0 {
            return Err(DgError::InvalidArg(
                "kk needs at least one non-empty group".into(),
            ));
        }
        (first_k, k)
    } else {
        (k, k)
    };

    let desc = GemmDesc {
        gemm_type: if is_kk {
            GemmType::KGroupedContiguous
        } else {
            GemmType::Normal
        },
        m: d.rows,
        n: d.cols,
        k,
        num_groups,
        a_dtype: Dtype::Fp8,
        b_dtype: Dtype::Fp8,
        cd_dtype: Dtype::F32,
        major_a: Major::K,
        major_b: Major::K,
        num_sms: dev.num_sms,
        smem_capacity: dev.smem_capacity,
        expected_m: d.rows,
        expected_num_groups: if is_kk { num_groups } else { 1 },
        expected_k: if is_kk { first_k } else { 0 },
        with_accumulation: is_kk,
    };
    let cfg = heuristics::sm90::get_best_config(&desc)?;
    let l = &cfg.layout;
    let s = &cfg.storage;
    let math = if l.block_m <= 64 { 128 } else { 256 };
    let mc_on_a = l.cluster_n > 1;
    let mcast = l.cluster_size();

    // Normal path: the epilogue is reduce-add, so D starts at zero.
    if !is_kk {
        crate::device::memset_f32(dev, stream, &d.data, d.rows * d.stride)?;
    }

    // Base A/B maps. For kk the dims/strides are placeholders for the FIRST
    // non-empty group (first_k); the kernel patches per group at runtime.
    let tm_a = tma::make_tma_ab(
        dev,
        Dtype::Fp8,
        Major::K,
        &x.a.data,
        x.a.rows,
        first_k,
        s.load_block_m,
        l.block_k,
        first_k,
        1,
        s.swizzle_a_mode,
        false,
    )?;
    let tm_b = tma::make_tma_ab(
        dev,
        Dtype::Fp8,
        Major::K,
        &x.b.data,
        x.b.rows,
        first_k,
        s.load_block_n,
        l.block_k,
        first_k,
        1,
        s.swizzle_b_mode,
        false,
    )?;
    let tm_sfa = tma::make_tma_sf_fp32(dev, &x.sfa.buf, x.sfa.mn, k, l.block_m, 1)?;
    let tm_sfb = tma::make_tma_sf_fp32(dev, &x.sfb.buf, x.sfb.mn, k, l.block_n, 1)?;
    let tm_cd = tma::make_tma_cd(
        dev,
        Dtype::F32,
        &d.data,
        d.rows,
        d.cols,
        s.store_block_m,
        s.store_block_n,
        d.stride,
        if is_kk { num_groups } else { 1 },
        0,
    )?;

    // Grouped-layout + tensormap scratch for kk.
    let gl_buf;
    let map_buf;

    let (gl_ptr, map_ptr) = if is_kk {
        let ks = ks.unwrap();
        gl_buf = alloc_and_upload(
            dev,
            &ks.iter().map(|&v| v as i32).collect::<Vec<i32>>(),
            stream.raw(),
        )?;
        // 2 descriptors (256B) per CTA, 128B-aligned base.
        map_buf = DevBuffer::alloc(dev, (dev.num_sms as usize) * 2 * 128)?;
        (gl_buf.ptr, map_buf.ptr)
    } else {
        (0, 0)
    };

    let gt = if is_kk { 5 } else { 0 };
    let sig = format!("{:?}", (cfg, gt, is_kk, x.a.k));
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    const unsigned char* a, const unsigned char* b,
    int* grouped_layout, dg::TmaMap* map_buf,
    unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_sfa, const __grid_constant__ dg::TmaMap tma_sfb,
    const __grid_constant__ dg::TmaMap tma_cd) {{
    dg::sm90_fp8_gemm_1d1d_impl<0, 0, 0, {num_groups},
        {bm}, {bn}, 128, {swa}, {swb},
        {stages}, 128, {math}, {mcast}, {mcoa}, {sms},
        (dg::GemmType){gt}>
        (a, b, grouped_layout, map_buf, m, n, k, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd);
}}"#,
        num_groups = num_groups,
        bm = l.block_m,
        bn = l.block_n,
        swa = s.swizzle_a_mode,
        swb = s.swizzle_b_mode,
        stages = cfg.pipeline.num_stages,
        math = math,
        mcast = mcast,
        mcoa = mc_on_a as u32,
        sms = dev.num_sms,
        gt = gt,
    );
    let func = jit::get_kernel(dev, jit::kernel_src::sm90_unit(), "gemm_sm90", &sig, &body)?;
    let args = Args::new()
        .devptr(x.a.data.ptr)
        .devptr(x.b.data.ptr)
        .devptr(gl_ptr)
        .devptr(map_ptr)
        .u32(d.rows)
        .u32(d.cols)
        .u32(k)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_sfa)
        .tensormap(&tm_sfb)
        .tensormap(&tm_cd);

    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + math, 1, 1),
            smem: cfg.pipeline.smem_size,
            cluster: Some((mcast, 1, 1)),
            pdl: true,
        },
        args,
    )
}

/// FP8 GEMM, NT layout, 1D1D fine-grained scaling (Hopper).
/// `sfa`/`sfb`: per-128-channel FP32 scales (see [`SfFp32`]).
/// D is zero-initialized then written (FP32).
pub fn fp8_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    sfa: &SfFp32,
    b: &Operand,
    sfb: &SfFp32,
    d: &mut Output,
) -> DgResult<()> {
    run_fp8_1d1d(dev, stream, &Fp8Inputs { a, b, sfa, sfb }, d, None)
}

/// K-grouped weight-grad FP8 GEMM (Hopper):
/// `D[g] += A_g @ B_g^T` for every group in ONE persistent launch.
/// `ks[g]` is the K size of group g (each % 128 == 0, sum == a.k).
/// A/B are "stacked along K": per-group `[mn, ks_g]` K-major tiles,
/// concatenated (group g at byte offset `prefix_sum(ks)*mn`).
// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_kk(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    sfa: &SfFp32,
    b: &Operand,
    sfb: &SfFp32,
    ks: &[u32],
    d: &mut Output,
) -> DgResult<()> {
    run_fp8_1d1d(dev, stream, &Fp8Inputs { a, b, sfa, sfb }, d, Some(ks))
}

// ---------------------------------------------------------------------------
// BF16 GEMM
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn run_bf16(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    grouped_layout: Option<&DevBuffer>,
    gemm_type: GemmType,
    expected_m: u32,
    num_groups: u32,
    accumulate: bool,
) -> DgResult<()> {
    require_sm90(dev)?;
    if a.dtype != Dtype::Bf16 || b.dtype != Dtype::Bf16 {
        return Err(DgError::InvalidArg(
            "bf16_gemm requires BF16 operands".into(),
        ));
    }
    if a.k != b.k {
        return Err(DgError::InvalidArg("A/B K mismatch".into()));
    }
    if d.cols != b.rows || d.rows != a.rows {
        return Err(DgError::InvalidArg("output shape mismatch".into()));
    }
    if d.dtype != Dtype::Bf16 && d.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg("output must be BF16 or FP32".into()));
    }
    if matches!(
        gemm_type,
        GemmType::MGroupedContiguous | GemmType::MGroupedMasked
    ) && grouped_layout.is_none()
    {
        return Err(DgError::InvalidArg(
            "grouped GEMM needs a grouped layout".into(),
        ));
    }

    let desc = GemmDesc {
        gemm_type,
        m: d.rows,
        n: d.cols,
        k: a.k,
        num_groups,
        a_dtype: Dtype::Bf16,
        b_dtype: Dtype::Bf16,
        cd_dtype: d.dtype,
        major_a: a.major,
        major_b: b.major,
        num_sms: dev.num_sms,
        smem_capacity: dev.smem_capacity,
        expected_m,
        expected_num_groups: if gemm_type == GemmType::MGroupedMasked {
            num_groups
        } else {
            1
        },
        expected_k: 0,
        with_accumulation: accumulate,
    };
    let cfg = heuristics::sm90::get_best_config(&desc)?;
    let l = &cfg.layout;
    let s = &cfg.storage;
    let math = if l.block_m <= 64 { 128 } else { 256 };
    let mc_on_a = l.cluster_n > 1;
    let mcast = l.cluster_size();

    let ma = match a.major {
        Major::K => 0,
        Major::Mn => 1,
    };
    let mb = match b.major {
        Major::K => 0,
        Major::Mn => 1,
    };
    let cd_code = if d.dtype == Dtype::Bf16 { 1 } else { 0 };

    // A/B maps. Group stacking follows upstream exactly:
    //   normal: A/B/CD groups=1; contiguous: B per-expert (groups=G), A/CD=1
    //   (A blocks are absolute in the padded span); masked: A/B/CD groups=G.
    let ga = if gemm_type == GemmType::MGroupedMasked {
        num_groups
    } else {
        1
    };
    let gb = if gemm_type == GemmType::Normal {
        1
    } else {
        num_groups
    };
    let tm_a = tma::make_tma_ab(
        dev,
        Dtype::Bf16,
        a.major,
        &a.data,
        a.rows,
        a.k,
        s.load_block_m,
        l.block_k,
        a.outer_stride,
        ga.max(1),
        s.swizzle_a_mode,
        false,
    )?;
    let tm_b = tma::make_tma_ab(
        dev,
        Dtype::Bf16,
        b.major,
        &b.data,
        b.rows,
        b.k,
        s.load_block_n,
        l.block_k,
        b.outer_stride,
        gb,
        s.swizzle_b_mode,
        false,
    )?;
    let cd_atom_n = if s.swizzle_cd_mode == 0 {
        l.block_n
    } else {
        s.swizzle_cd_mode / d.dtype.elem_size() as u32
    };
    let tm_cd = tma::make_tma_cd(
        dev,
        d.dtype,
        &d.data,
        d.rows,
        d.cols,
        s.store_block_m,
        cd_atom_n,
        d.stride,
        if gemm_type == GemmType::MGroupedMasked {
            num_groups
        } else {
            1
        },
        s.swizzle_cd_mode,
    )?;

    let gt = match gemm_type {
        GemmType::Normal => 0,
        GemmType::MGroupedContiguous => 1,
        GemmType::MGroupedMasked => 2,
        _ => unreachable!(),
    };
    let sig = format!("{:?}", (cfg, gt, ma, mb, cd_code, accumulate, a.k));
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    int* grouped_layout, unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_cd) {{
    dg::sm90_bf16_gemm_impl<{ma}, {mb}, 0, 0, 0, {ng},
        {bm}, {bn}, {bk}, {swa}, {swb}, {swd},
        {stages}, 128, {math}, {mcast}, {mcoa}, {sms},
        (dg::GemmType){gt}, {acc}, {cd}>
        (grouped_layout, m, n, k, tma_a, tma_b, tma_cd);
}}"#,
        ma = ma,
        mb = mb,
        ng = num_groups,
        bm = l.block_m,
        bn = l.block_n,
        bk = l.block_k,
        swa = s.swizzle_a_mode,
        swb = s.swizzle_b_mode,
        swd = s.swizzle_cd_mode,
        stages = cfg.pipeline.num_stages,
        math = math,
        mcast = mcast,
        mcoa = mc_on_a as u32,
        sms = dev.num_sms,
        gt = gt,
        acc = accumulate as u32,
        cd = cd_code,
    );
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::sm90_unit(),
        "gemm_sm90_bf16",
        &sig,
        &body,
    )?;
    let gl_ptr = grouped_layout.map(|b| b.ptr).unwrap_or(0);
    let args = Args::new()
        .devptr(gl_ptr)
        .u32(d.rows)
        .u32(d.cols)
        .u32(a.k)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_cd);

    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + math, 1, 1),
            smem: cfg.pipeline.smem_size,
            cluster: Some((mcast, 1, 1)),
            pdl: true,
        },
        args,
    )
}

/// BF16 GEMM, NT layout (Hopper): `D (+)= A @ B^T`.
pub fn bf16_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    d: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    run_bf16(
        dev,
        stream,
        a,
        b,
        d,
        None,
        GemmType::Normal,
        a.rows,
        1,
        accumulate,
    )
}

/// BF16 m-grouped contiguous (Hopper). `expected_m` is the *padded* M span
/// (sum of per-group alignments).
// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_nt_m_grouped_contiguous(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    grouped_layout: &DevBuffer,
    expected_m: u32,
    d: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    run_bf16(
        dev,
        stream,
        a,
        b,
        d,
        Some(grouped_layout),
        GemmType::MGroupedContiguous,
        expected_m,
        1,
        accumulate,
    )
}

/// BF16 m-grouped masked (Hopper). `expected_m` is the per-group M cap;
/// D is `[num_groups, expected_m, n]`.
// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_nt_m_grouped_masked(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    grouped_layout: &DevBuffer,
    expected_m: u32,
    num_groups: u32,
    d: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    run_bf16(
        dev,
        stream,
        a,
        b,
        d,
        Some(grouped_layout),
        GemmType::MGroupedMasked,
        expected_m,
        num_groups,
        accumulate,
    )
}
