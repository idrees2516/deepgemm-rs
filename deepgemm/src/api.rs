//! Public GEMM / MQA / transform / quant APIs (mirrors upstream `deep_gemm`).

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::heuristics::{self, GemmConfig, GemmDesc};
use crate::jit::{self, Args};
use crate::sys;
use crate::tma;
use crate::types::{Dtype, GemmType, Major, Operand, Output, SfGran, SfTensor};

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn umma_format(dt: Dtype) -> u32 {
    match dt {
        Dtype::Fp8 => 0,  // E4M3
        Dtype::Fp4 => 5,  // E2M1
        Dtype::Bf16 => 1, // BF16 (kind::f16 descriptor format)
        Dtype::F32 => 1,
    }
}

/// Generate the kernel body for a config (template instantiation wrapper).
/// (Mirrors the upstream kernel template parameter list — 8+ args by nature.)
#[allow(clippy::too_many_arguments)]
fn gemm_body(
    a: &Operand,
    b: &Operand,
    out: &Output,
    cfg: &GemmConfig,
    gemm_type: GemmType,
    gran_a: u32,
    gran_b: u32,
    accumulate: bool,
    num_sms: u32,
    with_output_sf: bool,
    with_alpha: bool,
    k_grouped: Option<(u32, u32)>, // (k_alignment, sfk_span)
) -> String {
    let has_sf = !(a.dtype == Dtype::Bf16 && b.dtype == Dtype::Bf16);
    let is_mxf4 = a.dtype == Dtype::Fp4 && b.dtype == Dtype::Fp4;
    let pack_a = if is_mxf4 { 2 } else { 1 };
    let pack_b = if is_mxf4 { 2 } else { 1 };
    let storage_a = match a.dtype {
        Dtype::Bf16 => 2,
        _ => 1,
    };
    let storage_b = match b.dtype {
        Dtype::Bf16 => 2,
        _ => 1,
    };
    let cd_is_float = out.dtype == Dtype::F32;
    let cd_elem = out.dtype.elem_size() as u32;

    let l = &cfg.layout;
    let s = &cfg.storage;
    let gt = match gemm_type {
        GemmType::Normal => 0,
        GemmType::MGroupedContiguous => 1,
        GemmType::MGroupedMasked => 2,
        GemmType::Batched => 4,
        GemmType::KGroupedContiguous => 5,
    };

    format!(
        r#"extern "C" __global__ void __dg_kernel(
    int* grouped_layout, unsigned num_groups,
    unsigned shape_m, unsigned shape_n, unsigned shape_k,
    const __grid_constant__ dg::TmaMap tma_a,
    const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_sfa,
    const __grid_constant__ dg::TmaMap tma_sfb,
    const __grid_constant__ dg::TmaMap tma_cd{sf_params}{alpha_params}) {{
    dg::gemm_sm100_impl<
        {major_a}, {major_b},
        {gran_a}, {gran_b}, {has_sf}, {is_mxf4},
        {fmt_a}, {fmt_b},
        {storage_a}, {storage_b},
        {pack_a}, {pack_b},
        {block_m}, {block_n}, {block_k},
        {swz_a}, {swz_b}, {swz_cd},
        {stages}, 2,
        {cluster}, {mc_on_a},
        {swap_ab}, (dg::GemmType){gt}, {accum},
        {cd_float}, {cd_elem},
        {num_sms}, {with_alpha}, {with_sf}{kg_params}
    >(grouped_layout, num_groups, shape_m, shape_n, shape_k,
      tma_a, tma_b, tma_sfa, tma_sfb, tma_cd{sf_args}{alpha_args});
}}"#,
        sf_params = if with_output_sf {
            String::from(",\n    unsigned* sfd, unsigned sfd_stride")
        } else {
            String::new()
        },
        sf_args = if with_output_sf {
            String::from(", sfd, sfd_stride")
        } else {
            String::new()
        },
        alpha_params = if with_alpha { ",\n    float alpha" } else { "" },
        // Kernel ABI: `(.., tma_cd, sfd, sfd_stride, alpha)` — when only
        // alpha is present, the two defaulted sfd slots must be occupied
        // explicitly (defaults resolve positionally).
        alpha_args = if !with_alpha {
            String::new()
        } else if with_output_sf {
            String::from(", sfd, sfd_stride, alpha")
        } else {
            String::from(", (unsigned*)0, 0, alpha")
        },
        kg_params = k_grouped
            .map(|(align, span)| format!(", {align}, {span}"))
            .unwrap_or_default(),
        major_a = a.major as u32,
        major_b = b.major as u32,
        gran_a = if has_sf { gran_a } else { 32 },
        gran_b = if has_sf { gran_b } else { 32 },
        has_sf = has_sf as u32,
        is_mxf4 = is_mxf4 as u32,
        fmt_a = umma_format(a.dtype),
        fmt_b = umma_format(b.dtype),
        storage_a = storage_a,
        storage_b = storage_b,
        pack_a = pack_a,
        pack_b = pack_b,
        block_m = l.block_m,
        block_n = l.block_n,
        block_k = l.block_k,
        swz_a = s.swizzle_a_mode,
        swz_b = s.swizzle_b_mode,
        swz_cd = s.swizzle_cd_mode,
        stages = cfg.pipeline.num_stages,
        cluster = l.cluster_size(),
        mc_on_a = (l.cluster_n > 1) as u32,
        swap_ab = l.swap_ab as u32,
        gt = gt,
        accum = accumulate as u32,
        cd_float = cd_is_float as u32,
        cd_elem = cd_elem,
        num_sms = num_sms,
        with_sf = with_output_sf as u32,
        with_alpha = with_alpha as u32,
    )
}

pub(crate) struct LaunchCtx<'a> {
    pub(crate) dev: &'a Device,
    pub(crate) stream: sys::Stream,
}

#[allow(clippy::too_many_arguments)]
fn run_gemm(
    ctx: &LaunchCtx,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    gemm_type: GemmType,
    grouped_layout: Option<&DevBuffer>,
    num_groups: u32,
    expected_m: u32,
    accumulate: bool,
    output_sf: Option<(&DevBuffer, u32)>,
    alpha: Option<f32>,
    k_alignment: u32,
) -> DgResult<()> {
    let dev = ctx.dev;
    if !matches!(dev.arch, crate::device::Arch::Sm100) {
        return Err(DgError::Unsupported(format!(
            "the tcgen05 kernels require SM100 (Blackwell); this device is {:?}",
            dev.arch
        )));
    }
    let has_sf = !(a.dtype == Dtype::Bf16 && b.dtype == Dtype::Bf16);
    let is_mxf4 = a.dtype == Dtype::Fp4 && b.dtype == Dtype::Fp4;
    if is_mxf4 && (a.major != Major::K || b.major != Major::K) {
        return Err(DgError::InvalidArg(
            "MXF4 MMA requires K-major operands".into(),
        ));
    }
    if has_sf && (a.sf.is_none() || b.sf.is_none()) {
        return Err(DgError::InvalidArg(
            "FP8/FP4 operands require scale factors".into(),
        ));
    }

    if let Some((_, _)) = output_sf {
        // Fused QuantizeToFP8 epilogue (upstream `operators::QuantizeToFP8`).
        if gemm_type != GemmType::Normal {
            return Err(DgError::InvalidArg(
                "fused output-SF quantization supports the Normal GEMM path".into(),
            ));
        }
        if out.dtype != Dtype::Fp8 {
            return Err(DgError::InvalidArg(
                "fused output-SF quantization requires an E4M3 D".into(),
            ));
        }
        if accumulate {
            return Err(DgError::InvalidArg(
                "fused output-SF quantization requires a direct (non-accumulating) D".into(),
            ));
        }
        if out.cols < 128 || out.cols % 32 != 0 {
            return Err(DgError::InvalidArg(
                "fused output-SF quantization requires N >= 128 and N % 32 == 0 \
                 (a store must cover complete 32-wide SF groups)"
                    .into(),
            ));
        }
        if alpha.is_some() {
            return Err(DgError::InvalidArg(
                "alpha and the fused FP8 output are exclusive (upstream contract)".into(),
            ));
        }
    }
    let is_k_grouped = gemm_type == GemmType::KGroupedContiguous;
    if is_k_grouped {
        if let Some(layout) = grouped_layout {
            if (layout.len / 4) as u32 != num_groups {
                return Err(DgError::InvalidArg(
                    "k-grouped layout must hold exactly num_groups entries".into(),
                ));
            }
        } else {
            return Err(DgError::InvalidArg(
                "k-grouped GEMM requires a grouped_layout buffer".into(),
            ));
        }
    }

    // Heuristics
    let desc = GemmDesc {
        gemm_type,
        m: out.rows,
        n: out.cols,
        k: a.k,
        num_groups,
        a_dtype: a.dtype,
        b_dtype: b.dtype,
        cd_dtype: out.dtype,
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
    let cfg = heuristics::get_best_config(&desc)?;
    let l = &cfg.layout;
    let s = &cfg.storage;

    // Granularities
    let gran_a = a.sf.as_ref().map(|sf| sf.gran.k()).unwrap_or(32);
    let gran_b = b.sf.as_ref().map(|sf| sf.gran.k()).unwrap_or(32);

    // Tensor maps
    let fp4_unpacked = a.dtype == Dtype::Fp4 && !is_mxf4;
    let fp4_unpacked_b = b.dtype == Dtype::Fp4 && !is_mxf4;
    // Data maps: batched uses 3D maps; masked stacks groups along the outer dim.
    let data_groups_a = match gemm_type {
        GemmType::MGroupedMasked => num_groups,
        _ => 1,
    };
    let data_groups_b = match gemm_type {
        // K-grouped B is a single shared `[sum_k, N]` arena (the scheduler
        // offsets K by `current_k_start`), NOT per-group stacked.
        GemmType::MGroupedMasked | GemmType::MGroupedContiguous => num_groups,
        GemmType::Normal | GemmType::KGroupedContiguous => 1,
        GemmType::Batched => num_groups,
    };
    // SF maps: groups (or batches) stack along the packed-K outer dim.
    // K-grouped SFs share one packed arena (per-group padded rows).
    let sf_groups_a = match gemm_type {
        GemmType::Normal | GemmType::MGroupedContiguous | GemmType::KGroupedContiguous => 1,
        _ => num_groups,
    };
    let sf_groups_b = match gemm_type {
        GemmType::Normal | GemmType::KGroupedContiguous => 1,
        _ => num_groups,
    };
    let tm_a = if gemm_type == GemmType::Batched {
        tma::make_tma_ab_3d(
            dev,
            a.dtype,
            a.major,
            &a.data,
            a.rows,
            a.k,
            s.load_block_m,
            l.block_k,
            a.outer_stride,
            data_groups_a,
            s.swizzle_a_mode,
            fp4_unpacked,
        )?
    } else {
        tma::make_tma_ab(
            dev,
            a.dtype,
            a.major,
            &a.data,
            a.rows,
            a.k,
            s.load_block_m,
            l.block_k,
            a.outer_stride,
            data_groups_a,
            s.swizzle_a_mode,
            fp4_unpacked,
        )?
    };
    let tm_b = if gemm_type == GemmType::Batched {
        tma::make_tma_ab_3d(
            dev,
            b.dtype,
            b.major,
            &b.data,
            b.rows,
            b.k,
            s.load_block_n,
            l.block_k,
            b.outer_stride,
            num_groups,
            s.swizzle_b_mode,
            fp4_unpacked_b,
        )?
    } else {
        tma::make_tma_ab(
            dev,
            b.dtype,
            b.major,
            &b.data,
            b.rows,
            b.k,
            s.load_block_n,
            l.block_k,
            b.outer_stride,
            data_groups_b,
            s.swizzle_b_mode,
            fp4_unpacked_b,
        )?
    };

    let (sf_block_m, sf_block_n) = heuristics::get_sf_block_sizes(l.block_m, l.block_n, has_sf);
    let sf_block_k = l.block_k / 128;
    // K-grouped: the layout's K padding unit must be a whole number of
    // BLOCK_K tiles (upstream `k_alignment % block_k == 0`), and the packed
    // SF arena's rows are the per-group padded counts (SfTensor.rows), not
    // `ceil_div(sum_k, gran*4)`.
    if is_k_grouped && k_alignment % l.block_k != 0 {
        return Err(DgError::InvalidArg(format!(
            "k-grouped k_alignment ({k_alignment}) must be a multiple of BLOCK_K ({})",
            l.block_k
        )));
    }
    let tm_sfa = if has_sf {
        let sf = a.sf.as_ref().unwrap();
        if is_k_grouped {
            tma::make_tma_sf_packed_rows(dev, &sf.buf, sf.cols, sf.rows, sf_block_m, sf_block_k)?
        } else {
            tma::make_tma_sf(
                dev,
                &sf.buf,
                sf.cols,
                a.k,
                sf.gran.k(),
                sf_block_m,
                sf_block_k,
                sf_groups_a,
                0,
            )?
        }
    } else {
        tma::make_tma_sf(dev, &DevBuffer::alloc(dev, 128)?, 32, 128, 32, 32, 1, 1, 0)?
    };
    let tm_sfb = if has_sf {
        let sf = b.sf.as_ref().unwrap();
        if is_k_grouped {
            tma::make_tma_sf_packed_rows(dev, &sf.buf, sf.cols, sf.rows, sf_block_n, sf_block_k)?
        } else {
            tma::make_tma_sf(
                dev,
                &sf.buf,
                sf.cols,
                b.k,
                sf.gran.k(),
                sf_block_n,
                sf_block_k,
                sf_groups_b,
                0,
            )?
        }
    } else {
        tma::make_tma_sf(dev, &DevBuffer::alloc(dev, 128)?, 32, 128, 32, 32, 1, 1, 0)?
    };
    let cd_atom_n = s.swizzle_cd_mode / out.dtype.elem_size() as u32;
    // K-grouped D is a 3D `[groups, m, n]` stack like Batched.
    let tm_cd = if gemm_type == GemmType::Batched || is_k_grouped {
        tma::make_tma_cd_3d(
            dev,
            out.dtype,
            &out.data,
            out.rows,
            out.cols,
            s.store_block_m,
            cd_atom_n,
            out.stride,
            num_groups,
            s.swizzle_cd_mode,
        )?
    } else {
        let cd_groups = match gemm_type {
            GemmType::Normal | GemmType::MGroupedContiguous => 1,
            _ => num_groups,
        };
        tma::make_tma_cd(
            dev,
            out.dtype,
            &out.data,
            out.rows,
            out.cols,
            s.store_block_m,
            cd_atom_n,
            out.stride,
            cd_groups,
            s.swizzle_cd_mode,
        )?
    };

    // Kernel + launch
    let body = gemm_body(
        a,
        b,
        out,
        &cfg,
        gemm_type,
        gran_a,
        gran_b,
        accumulate,
        dev.num_sms,
        output_sf.is_some(),
        alpha.is_some(),
        if is_k_grouped {
            Some((k_alignment, gran_a.max(gran_b) * 4))
        } else {
            None
        },
    );
    let sig = format!(
        "{:?}",
        (
            cfg,
            gemm_type,
            a.dtype,
            b.dtype,
            out.dtype,
            gran_a,
            gran_b,
            accumulate,
            output_sf.is_some(),
            alpha.is_some(),
            k_alignment
        )
    );
    let func = jit::get_kernel(dev, jit::kernel_src::GEMM_SM100, "gemm_sm100", &sig, &body)?;

    let gl_ptr = grouped_layout.map(|b| b.ptr).unwrap_or(0);
    let mut args = Args::new()
        .devptr(gl_ptr)
        .u32(num_groups)
        .u32(out.rows)
        .u32(out.cols)
        .u32(a.k)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_sfa)
        .tensormap(&tm_sfb)
        .tensormap(&tm_cd);
    if let Some((sfd, sfd_stride)) = output_sf {
        args = args.devptr(sfd.ptr).u32(sfd_stride);
    }
    if let Some(alpha) = alpha {
        args = args.f32(alpha);
    }

    let launch_cfg = sys::LaunchEx {
        grid: (dev.num_sms, 1, 1),
        block: (256, 1, 1),
        smem: cfg.pipeline.smem_size,
        cluster: Some((l.cluster_size(), 1, 1)),
        pdl: true,
    };
    jit::launch(dev, func, ctx.stream, &launch_cfg, args)
}

/// The post-map launch core of [`run_gemm`], exposed for callers that build
/// their own TMA descriptors (the einsum equations permute operand dims and
/// therefore need custom 3D strides). All semantic parameters mirror
/// `run_gemm`; `maps` = (a, b, sfa, sfb, cd).
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_gemm_with_maps(
    ctx: &LaunchCtx,
    a: &Operand,
    b: &Operand,
    out: &Output,
    gemm_type: GemmType,
    grouped_layout: Option<&DevBuffer>,
    num_groups: u32,
    accumulate: bool,
    output_sf: Option<(&DevBuffer, u32)>,
    alpha: Option<f32>,
    k_alignment: u32,
    maps: (
        sys::TensorMap,
        sys::TensorMap,
        sys::TensorMap,
        sys::TensorMap,
        sys::TensorMap,
    ),
) -> DgResult<()> {
    let dev = ctx.dev;
    let desc = GemmDesc {
        gemm_type,
        m: out.rows,
        n: out.cols,
        k: a.k,
        num_groups,
        a_dtype: a.dtype,
        b_dtype: b.dtype,
        cd_dtype: out.dtype,
        major_a: a.major,
        major_b: b.major,
        num_sms: dev.num_sms,
        smem_capacity: dev.smem_capacity,
        expected_m: out.rows,
        expected_num_groups: if gemm_type == GemmType::MGroupedMasked {
            num_groups
        } else {
            1
        },
        expected_k: 0,
        with_accumulation: accumulate,
    };
    let cfg = heuristics::get_best_config(&desc)?;
    let l = &cfg.layout;
    let gran_a = a.sf.as_ref().map(|sf| sf.gran.k()).unwrap_or(32);
    let gran_b = b.sf.as_ref().map(|sf| sf.gran.k()).unwrap_or(32);
    let is_k_grouped = gemm_type == GemmType::KGroupedContiguous;
    let (tm_a, tm_b, tm_sfa, tm_sfb, tm_cd) = maps;

    let body = gemm_body(
        a,
        b,
        out,
        &cfg,
        gemm_type,
        gran_a,
        gran_b,
        accumulate,
        dev.num_sms,
        output_sf.is_some(),
        alpha.is_some(),
        if is_k_grouped {
            Some((k_alignment, gran_a.max(gran_b) * 4))
        } else {
            None
        },
    );
    let sig = format!(
        "{:?}",
        (
            cfg,
            gemm_type,
            a.dtype,
            b.dtype,
            out.dtype,
            gran_a,
            gran_b,
            accumulate,
            output_sf.is_some(),
            alpha.is_some(),
            k_alignment
        )
    );
    let func = jit::get_kernel(dev, jit::kernel_src::GEMM_SM100, "gemm_sm100", &sig, &body)?;

    let gl_ptr = grouped_layout.map(|b| b.ptr).unwrap_or(0);
    let mut args = Args::new()
        .devptr(gl_ptr)
        .u32(num_groups)
        .u32(out.rows)
        .u32(out.cols)
        .u32(a.k)
        .tensormap(&tm_a)
        .tensormap(&tm_b)
        .tensormap(&tm_sfa)
        .tensormap(&tm_sfb)
        .tensormap(&tm_cd);
    if let Some((sfd, sfd_stride)) = output_sf {
        args = args.devptr(sfd.ptr).u32(sfd_stride);
    }
    if let Some(alpha) = alpha {
        args = args.f32(alpha);
    }

    let launch_cfg = sys::LaunchEx {
        grid: (dev.num_sms, 1, 1),
        block: (256, 1, 1),
        smem: cfg.pipeline.smem_size,
        cluster: Some((l.cluster_size(), 1, 1)),
        pdl: true,
    };
    jit::launch(dev, func, ctx.stream, &launch_cfg, args)
}

// ---------------------------------------------------------------------------
// Public GEMM entry points
// ---------------------------------------------------------------------------
//
// Layout naming follows BLAS/upstream (`gemm_xy` = (A layout, B layout) of the
// two *logical* `[M, K]` / `[N, K]` matrices):
//
//   nt: A K-major `[M, K]`,  B K-major `[N, K]`   (the canonical path)
//   nn: A K-major `[M, K]`,  B MN-major `[K, N]`  (B column-major)
//   tn: A MN-major `[K, M]`, B K-major `[N, K]`   (A column-major)
//   tt: A MN-major `[K, M]`, B MN-major `[K, N]`
//
// Upstream implements nn/tn/tt as host-side transpose *views* routed into the
// NT kernel (`fp8_fp4_gemm_nn(a, b) == fp8_fp4_gemm_nt(a, b^T)`); the kernel
// itself takes the majorness as a template parameter, so the same is true
// here: [`Operand`] carries `major` + `outer_stride`, and the wrappers below
// merely construct the view and call [`gemm_nt_layout`]. The SF layout is
// independent of data majorness (always the packed MN-major form produced by
// [`transform_sf`]), so no SF transformation differs between layouts.
//
// Constraints (upstream-faithful): FP4 x FP4 requires K-major operands
// (`fp8_fp4_requires_k_major`); K-major SF *views* are not accepted — the
// packed UE8M0 tensor must already be MN-major TMA-aligned.

/// Dense GEMM: `D (+)= A @ B^T` with NT layout (K-major A/B).
pub fn gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    gemm_nt_layout(dev, stream, a, b, out, accumulate, None)
}

/// [`gemm_nt`] with an optional BLAS-style `alpha` on the product term:
/// `D = alpha * (A @ B^T) (+ C)`. The scale multiplies the FP32 accumulator
/// before the output cast (upstream `epilogue::operators::ScaleByAlpha`);
/// the accumulated C is never scaled. Exclusive with the fused FP8 output.
pub fn gemm_nt_alpha(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
    alpha: Option<f32>,
) -> DgResult<()> {
    gemm_nt_layout(dev, stream, a, b, out, accumulate, alpha)
}

/// The general dense entry: operands may individually be K- or MN-major
/// (covering the nn/tn/tt BLAS layouts); `alpha` as in [`gemm_nt_alpha`].
pub fn gemm_nt_layout(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
    alpha: Option<f32>,
) -> DgResult<()> {
    validate_operand(a, "A")?;
    validate_operand(b, "B")?;
    if a.k != b.k {
        return Err(DgError::InvalidArg("A/B K mismatch".into()));
    }
    if out.cols != b.rows || out.rows != a.rows {
        return Err(DgError::InvalidArg("output shape mismatch".into()));
    }
    if out.dtype != Dtype::Bf16 && out.dtype != Dtype::F32 {
        return Err(DgError::InvalidArg("output must be BF16 or FP32".into()));
    }
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::Normal,
        None,
        1,
        a.rows,
        accumulate,
        None,
        alpha,
        128,
    )
}

/// `D (+)= A @ B` with A K-major `[M, K]` and B MN-major `[K, N]`
/// (upstream `fp8_fp4_gemm_nn` / `bf16_gemm_nn`). Zero-copy: constructs the
/// `[N, K]`-viewed MN-major operand over the same device memory.
pub fn gemm_nn(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    let b = b.as_viewed_mn_major("B")?;
    gemm_nt_layout(dev, stream, a, b, out, accumulate, None)
}

/// `D (+)= A^T @ B^T` with A MN-major `[K, M]` and B K-major `[N, K]`
/// (upstream `*_gemm_tn`).
pub fn gemm_tn(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    let a = a.as_viewed_mn_major("A")?;
    gemm_nt_layout(dev, stream, a, b, out, accumulate, None)
}

/// `D (+)= A^T @ B` with A and B MN-major (`[K, M]`, `[K, N]`)
/// (upstream `*_gemm_tt`).
pub fn gemm_tt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    let a = a.as_viewed_mn_major("A")?;
    let b = b.as_viewed_mn_major("B")?;
    gemm_nt_layout(dev, stream, a, b, out, accumulate, None)
}

/// Alias of [`gemm_nt`] for FP8 operands (MXFP8, UE8M0 block scales).
pub fn fp8_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    gemm_nt(dev, stream, a, b, out, accumulate)
}

/// SFD layout for [`fp8_gemm_nt_quant_out`]: `(word_rows, row_stride)` of
/// the packed UE8M0 scale-factor output, i.e. `[ceil(n/32/4), align(m, 4)]`
/// `uint32` words (MN-major, TMA-aligned — the same layout the GEMM accepts
/// as SFA, so the output feeds straight into the next FP8 GEMM).
pub fn output_sf_layout(n: u32, m: u32) -> (u32, u32) {
    (n.div_ceil(128), m.div_ceil(4) * 4)
}

/// FP8 GEMM whose epilogue *fuses* the dynamic-output quantization (upstream
/// `epilogue::operators::QuantizeToFP8`): the fp32 accumulator is rounded to
/// BF16, then cast to E4M3 with per-row, per-32-column UE8M0 scale factors
/// packed into `sfd` — bitwise identical to a BF16 `D` followed by the
/// standalone cast kernel, without the extra round trip through HBM.
///
/// * `out.dtype` must be [`Dtype::Fp8`]; no accumulation; Normal GEMM only.
/// * `sfd` must hold `output_sf_layout(n, m)` words and be **zeroed by the
///   caller** (rows/blocks past `shape_m`/`shape_n` stay zero).
/// * Requires `n >= 128` and `n % 32 == 0`.
pub fn fp8_gemm_nt_quant_out(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    sfd: &DevBuffer,
) -> DgResult<()> {
    let (words_n, stride) = output_sf_layout(out.cols, out.rows);
    let need = words_n as usize * stride as usize * 4;
    if sfd.len < need {
        return Err(DgError::InvalidArg(format!(
            "sfd buffer too small: {need} bytes required, {} given",
            sfd.len
        )));
    }
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::Normal,
        None,
        1,
        a.rows,
        false,
        Some((sfd, stride)),
        None,
        128,
    )
}

/// Alias of [`gemm_nt`] for packed FP4 operands — the native Blackwell
/// `tcgen05.mma.kind::mxf4` path (`bench fp4_nt_native`).
pub fn fp4_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    gemm_nt(dev, stream, a, b, out, accumulate)
}

/// Alias of [`gemm_nt`] for BF16 operands.
pub fn bf16_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
) -> DgResult<()> {
    gemm_nt(dev, stream, a, b, out, accumulate)
}

/// MoE m-grouped contiguous GEMM (prefill): `m_indices[block]` holds the
/// expert id of each BLOCK_M-aligned token block (negative = padding).
///
/// B may be K-major (`[G, N, K]`, the common case) or MN-major
/// (`[G, K, N]`, upstream's `m_grouped_*_nn_contiguous`); for MN-major B the
/// expert groups stack along the K dim of the shared `[K, G*N]`-shaped view
/// (scheduler `kBWithGroupKOffset`).
pub fn m_grouped_gemm_nt_contiguous(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    m_indices: &DevBuffer,
    num_groups: u32,
) -> DgResult<()> {
    let alignment = heuristics::mk_alignment_for_contiguous_layout();
    if a.rows % alignment != 0 {
        return Err(DgError::InvalidArg(format!(
            "contiguous layout requires M ({}) to be a multiple of {alignment}",
            a.rows
        )));
    }
    if a.major != Major::K {
        return Err(DgError::InvalidArg(
            "m-grouped contiguous requires K-major A".into(),
        ));
    }
    if is_mxf4_pair(a, b) && b.major != Major::K {
        return Err(DgError::InvalidArg(
            "MXF4 MMA requires K-major operands".into(),
        ));
    }
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::MGroupedContiguous,
        Some(m_indices),
        num_groups,
        a.rows,
        false,
        None,
        None,
        128,
    )
}

/// MoE m-grouped contiguous GEMM with an MN-major B view: the weights come
/// as a `[G, K, N]` N-contiguous stack (upstream
/// `m_grouped_fp8_fp4_gemm_nn_contiguous` / `m_grouped_bf16_gemm_nn_*`).
/// Pure view routing into [`m_grouped_gemm_nt_contiguous`].
pub fn m_grouped_gemm_nn_contiguous(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    m_indices: &DevBuffer,
    num_groups: u32,
) -> DgResult<()> {
    b.as_viewed_mn_major("B")?;
    m_grouped_gemm_nt_contiguous(dev, stream, a, b, out, m_indices, num_groups)
}

/// True when both operands are FP4 (the `mxf4` block-scaled MMA path).
fn is_mxf4_pair(a: &Operand, b: &Operand) -> bool {
    a.dtype == Dtype::Fp4 && b.dtype == Dtype::Fp4
}

/// MoE m-grouped masked GEMM (decode): `masked_m[g]` is the valid M of group g;
/// A/B/D are per-group stacked tensors.
#[allow(clippy::too_many_arguments)]
pub fn m_grouped_gemm_nt_masked(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    masked_m: &DevBuffer,
    num_groups: u32,
    expected_m: u32,
) -> DgResult<()> {
    if a.major != Major::K || b.major != Major::K {
        return Err(DgError::InvalidArg(
            "m-grouped masked requires K-major A/B".into(),
        ));
    }
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::MGroupedMasked,
        Some(masked_m),
        num_groups,
        expected_m,
        false,
        None,
        None,
        128,
    )
}

/// Batched GEMM (BMM): A/B/D stacked along the batch dimension.
pub fn fp8_bmm(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    batch: u32,
    accumulate: bool,
) -> DgResult<()> {
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::Batched,
        None,
        batch,
        a.rows,
        accumulate,
        None,
        None,
        128,
    )
}

fn validate_operand(op: &Operand, name: &str) -> DgResult<()> {
    let k = op.k;
    match op.dtype {
        Dtype::Fp4 => {
            if k % 256 != 0 {
                return Err(DgError::InvalidArg(format!(
                    "{name}: FP4 K ({k}) must be a multiple of 256"
                )));
            }
        }
        Dtype::Fp8 => {
            if k % 128 != 0 {
                return Err(DgError::InvalidArg(format!(
                    "{name}: FP8 K ({k}) must be a multiple of 128"
                )));
            }
        }
        Dtype::Bf16 => {
            if k % 64 != 0 {
                return Err(DgError::InvalidArg(format!(
                    "{name}: BF16 K ({k}) must be a multiple of 64"
                )));
            }
        }
        Dtype::F32 => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// transform_sf
// ---------------------------------------------------------------------------

/// Transform FP32 scale factors `[mn, sf_k]` (row-major, powers of two) into
/// the packed UE8M0 layout `[ceil(sf_k/4), TMA-aligned(mn)]` int32 required by
/// the SM100 kernels.
pub fn transform_sf(
    dev: &Device,
    stream: &DevStream,
    sf: &[f32],
    mn: u32,
    gran: SfGran,
) -> DgResult<SfTensor> {
    let sf_k = (sf.len() as u32) / mn;
    if sf.len() as u32 != mn * sf_k {
        return Err(DgError::InvalidArg("sf length must be mn * sf_k".into()));
    }
    let tma_aligned = heuristics::tma_aligned_size(mn, 4);
    let packed_rows = ceil_div(sf_k, 4);
    let out = DevBuffer::alloc_zeros(dev, (packed_rows * tma_aligned * 4) as usize)?;
    let in_buf = crate::device::alloc_and_upload(dev, sf, stream.raw())?;
    transform_sf_launch(
        dev,
        stream,
        &in_buf,
        &out,
        0,
        mn,
        sf_k,
        packed_rows,
        tma_aligned,
    )?;

    Ok(SfTensor {
        rows: packed_rows,
        cols: tma_aligned,
        gran,
        buf: out,
    })
}

/// Launch the single-batch `transform_sf` pack kernel writing into `out` at
/// `packed-row offset` (the batched arena builder reuses this per batch).
#[allow(clippy::too_many_arguments)]
fn transform_sf_launch(
    dev: &Device,
    stream: &DevStream,
    in_buf: &DevBuffer,
    out: &DevBuffer,
    row_offset: u32,
    mn: u32,
    sf_k: u32,
    packed_rows: u32,
    tma_aligned: u32,
) -> DgResult<()> {
    let _ = packed_rows;
    // Kernel: SF_K < 16 uses the small single-pass path; else 16-wide blocks.
    const BLOCK_SF_K: u32 = 16;
    let num_threads = 128u32;
    let block_mn = 64u32;
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(const float* sf, unsigned* out, unsigned mn) {{
    dg::transform_sf_impl<{threads}, {block_mn}, {sf_k}, {block_sf_k}>(sf, out, mn);
}}"#,
        threads = num_threads,
        block_mn = block_mn,
        sf_k = sf_k,
        block_sf_k = BLOCK_SF_K,
    );
    let sig = format!("transform_sf_{sf_k}_{block_mn}");
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::LAYOUT_QUANT,
        "transform_sf",
        &sig,
        &body,
    )?;

    let num_mn_blocks = ceil_div(mn, block_mn);
    let num_k_blocks = if sf_k < BLOCK_SF_K {
        1
    } else {
        ceil_div(sf_k, BLOCK_SF_K)
    };
    let smem = block_mn
        * (if sf_k < BLOCK_SF_K {
            sf_k
        } else {
            BLOCK_SF_K + 1
        })
        * 4;
    let grid = (num_mn_blocks * num_k_blocks, 1, 1);
    let out_base = out.ptr + (row_offset * tma_aligned * 4) as u64;
    let args = Args::new()
        .ptr(in_buf.ptr as *const u8 as *const u32)
        .ptr(out_base as *const u8 as *const u32)
        .u32(mn);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid,
            block: (num_threads, 1, 1),
            smem,
            cluster: None,
            pdl: false,
        },
        args,
    )
}

/// K-grouped SF packing (upstream `transform_k_grouped_sf_into_required_layout`
/// -> `pack_fp32_into_ue8m0`): per-group FP32 scales into the packed UE8M0
/// arena where EVERY group starts at a fresh packed row.
///
/// * `sf` — the concatenated transposed arena `[sum_sf_rows, mn]`, K-major
///   along `mn` (each row = one K-group of scales for all MN lanes; this is
///   the `sf.T` of the per-token `[mn, sf_k]` form, i.e. `sf[row * mn + m]`).
/// * `grouped_layout` — non-psum flavor: `grouped_layout[g]` = padded K size
///   of group g (each a multiple of `k_alignment` and of the granularity).
/// * `gran` — SF granularity along K (G32 or G128).
///
/// The returned [`SfTensor::rows`] is the *padded* packed-row count
/// (`sum_g ceil_div(k_g / gran, 4)`), which can exceed
/// `ceil_div(sum_k / gran, 4)` — exactly what the k-grouped GEMM scheduler's
/// `current_sf_k_cumsum` addressing expects.
#[allow(clippy::too_many_arguments)]
pub fn transform_sf_k_grouped(
    dev: &Device,
    stream: &DevStream,
    sf: &[f32],
    mn: u32,
    gran: SfGran,
    grouped_layout: &[i32],
    k_alignment: u32,
) -> DgResult<SfTensor> {
    let num_groups = grouped_layout.len() as u32;
    if num_groups == 0 || num_groups > 128 {
        return Err(DgError::InvalidArg(
            "k-grouped layout must hold 1..=128 group entries".into(),
        ));
    }
    if mn % 4 != 0 {
        return Err(DgError::InvalidArg(
            "k-grouped SF requires mn % 4 == 0".into(),
        ));
    }
    let gran_k = gran.k();
    let mut sum_sf_rows = 0u32;
    let mut packed_rows = 0u32;
    for &k in grouped_layout {
        let k = if k < 0 { 0 } else { k as u32 };
        if k % k_alignment != 0 || k % gran_k != 0 {
            return Err(DgError::InvalidArg(format!(
                "k-grouped K sizes must be multiples of k_alignment ({k_alignment}) and the SF granularity ({gran_k})"
            )));
        }
        let rows = k / gran_k;
        sum_sf_rows += rows;
        packed_rows += ceil_div(rows, 4);
    }
    if sf.len() as u32 != sum_sf_rows * mn {
        return Err(DgError::InvalidArg(format!(
            "k-grouped SF arena must hold sum_k/gran * mn = {} f32 values (got {})",
            sum_sf_rows * mn,
            sf.len()
        )));
    }
    let tma_aligned = heuristics::tma_aligned_size(mn, 4);
    let out = DevBuffer::alloc_zeros(dev, (packed_rows * tma_aligned * 4) as usize)?;
    let in_buf = crate::device::alloc_and_upload(dev, sf, stream.raw())?;
    let layout_buf = crate::device::alloc_and_upload(dev, grouped_layout, stream.raw())?;

    const NUM_THREADS: u32 = 512;
    const BLOCK_MN: u32 = 128;
    const BLOCK_PACKED_SF_K: u32 = NUM_THREADS / 32; // 16 = one warp per row
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(const float* sf, unsigned* out, const int* grouped_layout,
    unsigned mn, unsigned sf_k, unsigned packed_sf_k, unsigned gran_k, unsigned k_alignment) {{
    dg::pack_k_grouped_sf_impl<{num_groups}, {threads}, {block_mn}, {block_packed_sf_k}, false>(
        sf, out, grouped_layout, mn, sf_k, packed_sf_k, gran_k, k_alignment);
}}"#,
        num_groups = num_groups,
        threads = NUM_THREADS,
        block_mn = BLOCK_MN,
        block_packed_sf_k = BLOCK_PACKED_SF_K,
    );
    let sig = format!("pack_k_grouped_sf_{num_groups}_{gran_k}");
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::LAYOUT_QUANT,
        "pack_k_grouped_sf",
        &sig,
        &body,
    )?;
    let grid = (
        ceil_div(mn, BLOCK_MN) * ceil_div(packed_rows, BLOCK_PACKED_SF_K),
        1,
        1,
    );
    let args = Args::new()
        .ptr(in_buf.ptr as *const u8 as *const u32)
        .ptr(out.ptr as *const u8 as *const u32)
        .ptr(layout_buf.ptr as *const u8 as *const i32)
        .u32(mn)
        .u32(sum_sf_rows)
        .u32(packed_rows)
        .u32(gran_k)
        .u32(k_alignment);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid,
            block: (NUM_THREADS, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;

    Ok(SfTensor {
        rows: packed_rows,
        cols: tma_aligned,
        gran,
        buf: out,
    })
}

/// Batched (BMM) SF packing: `sf` is the per-batch per-token scales
/// `[batch, mn, sf_k]` (batch-major), packed into the concatenated arena
/// `[batch * packed_sf_k, tma_aligned_mn]` the Batched GEMM scheduler
/// addresses (`sfa_k_idx = batch_idx * packed_rows + block`). Each batch's
/// rows are packed independently, exactly like `batch * transform_sf` calls
/// but in one allocation.
pub fn transform_sf_batched(
    dev: &Device,
    stream: &DevStream,
    sf: &[f32],
    batch: u32,
    mn: u32,
    gran: SfGran,
) -> DgResult<SfTensor> {
    if batch == 0 {
        return Err(DgError::InvalidArg("batch must be nonzero".into()));
    }
    let per_batch = (sf.len() / batch as usize) as u32;
    if per_batch == 0 || sf.len() as u32 != per_batch * batch || per_batch % mn != 0 {
        return Err(DgError::InvalidArg(
            "batched SF must hold batch * mn * sf_k values".into(),
        ));
    }
    let sf_k = per_batch / mn;
    let tma_aligned = heuristics::tma_aligned_size(mn, 4);
    let packed_rows = ceil_div(sf_k, 4);
    // ONE contiguous arena; each batch's rows land at batch_idx *
    // packed_rows (the layout the Batched scheduler's SF_K offset and the
    // `packed_rows * batch` TMA outer dim expect).
    let arena = DevBuffer::alloc_zeros(dev, (packed_rows * batch * tma_aligned * 4) as usize)?;
    for g in 0..batch as usize {
        let in_buf = crate::device::alloc_and_upload(
            dev,
            &sf[g * per_batch as usize..(g + 1) * per_batch as usize],
            stream.raw(),
        )?;
        transform_sf_launch(
            dev,
            stream,
            &in_buf,
            &arena,
            g as u32 * packed_rows,
            mn,
            sf_k,
            packed_rows,
            tma_aligned,
        )?;
    }
    Ok(SfTensor {
        rows: packed_rows * batch,
        cols: tma_aligned,
        gran,
        buf: arena,
    })
}

/// Elementwise FP32 -> BF16 (RNE) cast kernel — the tail of the
/// `bmk,bnk->mn` BF16 einsum path (upstream's implicit `d.copy_(workspace)`).
pub fn cast_to_bf16(
    dev: &Device,
    stream: &DevStream,
    src: &DevBuffer,
    dst: &DevBuffer,
    n: u32,
) -> DgResult<()> {
    if src.len < n as usize * 4 || dst.len < n as usize * 2 {
        return Err(DgError::InvalidArg(
            "cast_to_bf16: buffer smaller than n elements".into(),
        ));
    }
    let body = r#"extern "C" __global__ void __dg_kernel(const float* in, unsigned short* out, unsigned n) {
    dg::cast_f32_to_bf16_impl<256>(in, out, n);
}"#;
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::LAYOUT_QUANT,
        "cast_f32_bf16",
        "c",
        body,
    )?;
    let grid = (ceil_div(n, 2048).max(1), 1, 1);
    let args = Args::new()
        .ptr(src.ptr as *const u8 as *const f32)
        .ptr(dst.ptr as *const u8 as *const u16)
        .u32(n);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid,
            block: (256, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )
}

// ---------------------------------------------------------------------------
// K-grouped (weight-grad) GEMMs
// ---------------------------------------------------------------------------
//
// D[g] (+)= A_g @ B_g^T where the per-group K-slices of A/B live in one
// shared arena: FP8-TN stacks them MN-major (`[sum_k, M]` / `[sum_k, N]`,
// upstream `k_grouped_fp8_gemm_tn_contiguous`, the gradient-gather layout),
// FP4-NT keeps the canonical K-major packing (`[M, sum_k/2]`, upstream
// `k_grouped_fp4_gemm_nt_contiguous`). `grouped_layout[g]` is group g's
// padded K size (multiple of `k_alignment`); SFs come from
// [`transform_sf_k_grouped`]. D is `[groups, m, n]`.
//
// The scheduler walks every (m, n) block once per group (`current_k_start`
// hops the arena); each group's partial products accumulate into that
// group's own D slice via the 3D CD map.

/// FP8 K-grouped weight-grad GEMM, TN layout (A/B MN-major).
/// `gran` selects the SF granularity (G32 = MX recipe, G128 = DeepSeek
/// recipe); both operands must use the same one (upstream asserts this too).
#[allow(clippy::too_many_arguments)]
pub fn k_grouped_fp8_gemm_tn_contiguous(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    grouped_layout: &DevBuffer,
    num_groups: u32,
    accumulate: bool,
    gran: SfGran,
) -> DgResult<()> {
    let _ = gran; // both operands must share the granularity (checked below)
    a.as_viewed_mn_major("A")?;
    b.as_viewed_mn_major("B")?;
    if a.dtype != Dtype::Fp8 || b.dtype != Dtype::Fp8 {
        return Err(DgError::InvalidArg(
            "the k-grouped TN path is the FP8 weight-grad GEMM".into(),
        ));
    }
    if let (Some(sfa), Some(sfb)) = (a.sf.as_ref(), b.sf.as_ref()) {
        if sfa.gran != sfb.gran {
            return Err(DgError::InvalidArg(
                "k-grouped operands must share the SF granularity".into(),
            ));
        }
    }
    if a.k != b.k {
        return Err(DgError::InvalidArg("A/B K mismatch".into()));
    }
    let k_alignment = 128u32;
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::KGroupedContiguous,
        Some(grouped_layout),
        num_groups,
        a.rows,
        accumulate,
        None,
        None,
        k_alignment,
    )
}

/// FP4 K-grouped weight-grad GEMM, NT layout (A/B K-major, packed bytes).
#[allow(clippy::too_many_arguments)]
pub fn k_grouped_fp4_gemm_nt_contiguous(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    grouped_layout: &DevBuffer,
    num_groups: u32,
    accumulate: bool,
) -> DgResult<()> {
    if a.major != Major::K || b.major != Major::K {
        return Err(DgError::InvalidArg(
            "k-grouped FP4 requires K-major operands".into(),
        ));
    }
    if a.dtype != Dtype::Fp4 || b.dtype != Dtype::Fp4 {
        return Err(DgError::InvalidArg(
            "the k-grouped NT FP4 path expects MXF4 operands".into(),
        ));
    }
    if a.k != b.k {
        return Err(DgError::InvalidArg("A/B K mismatch".into()));
    }
    // MXF4 runs BLOCK_K = 256, so the layout's K padding unit must match.
    let k_alignment = 256u32;
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    run_gemm(
        &ctx,
        a,
        b,
        out,
        GemmType::KGroupedContiguous,
        Some(grouped_layout),
        num_groups,
        a.rows,
        accumulate,
        None,
        None,
        k_alignment,
    )
}

// ---------------------------------------------------------------------------
// Activation quantization (MXFP8 / MXFP4)
// ---------------------------------------------------------------------------

/// Quantize FP32 activations `[m, k]` row-major into MXFP8 (E4M3 + UE8M0 per
/// `gran` elements) or packed MXFP4 (E2M1 + UE8M0 per 32 elements).
pub fn quant_mx(
    dev: &Device,
    stream: &DevStream,
    x: &[f32],
    m: u32,
    k: u32,
    dtype: Dtype,
    gran: SfGran,
) -> DgResult<(DevBuffer, SfTensor)> {
    if dtype == Dtype::Fp4 && (k % 32 != 0 || gran != SfGran::G32) {
        return Err(DgError::InvalidArg(
            "MXFP4 requires K % 32 == 0 and gran 32".into(),
        ));
    }
    if dtype == Dtype::Fp8 && k % gran.k() != 0 {
        return Err(DgError::InvalidArg(
            "K must be a multiple of the SF granularity".into(),
        ));
    }
    let x_buf = crate::device::alloc_and_upload(dev, x, stream.raw())?;
    let data_bytes = if dtype == Dtype::Fp4 {
        (m * k / 2) as usize
    } else {
        (m * k) as usize
    };
    let data = DevBuffer::alloc_zeros(dev, data_bytes.max(1))?;

    let tma_aligned = heuristics::tma_aligned_size(m, 4);
    let packed_rows = ceil_div(ceil_div(k, gran.k()), 4);
    let sf = DevBuffer::alloc_zeros(dev, (packed_rows * tma_aligned * 4) as usize)?;

    let threads = 256u32;
    let is_fp4 = dtype == Dtype::Fp4;
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(const float* x, unsigned m, unsigned k,
                                     unsigned char* out_data, unsigned* out_sf) {{
    dg::quant_mx_impl<{threads}, {gran}, {is_fp4}>(x, m, k, out_data, out_sf);
}}"#,
        threads = threads,
        gran = gran.k(),
        is_fp4 = is_fp4 as u32,
    );
    let sig = format!("quant_{:?}_{is_fp4}", gran);
    let func = jit::get_kernel(dev, jit::kernel_src::LAYOUT_QUANT, "quant", &sig, &body)?;
    let args = Args::new()
        .ptr(x_buf.ptr as *const u8 as *const f32)
        .u32(m)
        .u32(k)
        .ptr(data.ptr as *const u8)
        .ptr(sf.ptr as *const u8 as *const u32);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (m, 1, 1),
            block: (threads, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;

    Ok((
        data,
        SfTensor {
            rows: packed_rows,
            cols: tma_aligned,
            gran,
            buf: sf,
        },
    ))
}

/// Dequantize MXFP8/MXFP4 back to FP32 (reference path for tests/tools).
pub fn dequant_mx(
    dev: &Device,
    stream: &DevStream,
    data: &DevBuffer,
    sf: &SfTensor,
    m: u32,
    k: u32,
    dtype: Dtype,
) -> DgResult<DevBuffer> {
    let out = DevBuffer::alloc(dev, (m * k * 4) as usize)?;
    let threads = 256u32;
    let is_fp4 = dtype == Dtype::Fp4;
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(const unsigned char* data, const unsigned* sf,
                                     unsigned m, unsigned k, unsigned tma_aligned_m, float* out) {{
    dg::dequant_mx_impl<{is_fp4}, {gran}>(data, sf, m, k, tma_aligned_m, out);
}}"#,
        is_fp4 = is_fp4 as u32,
        gran = sf.gran.k(),
    );
    let sig = format!("dequant_{:?}_{is_fp4}", sf.gran);
    let func = jit::get_kernel(dev, jit::kernel_src::LAYOUT_QUANT, "dequant", &sig, &body)?;
    let total = m * k;
    let grid = (ceil_div(total, threads), 1, 1);
    let args = Args::new()
        .ptr(data.ptr as *const u8)
        .ptr(sf.buf.ptr as *const u8 as *const u32)
        .u32(m)
        .u32(k)
        .u32(sf.cols)
        .ptr(out.ptr as *const u8 as *const f32);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid,
            block: (threads, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// MQA logits (MLA lightning indexer, contiguous KV)
// ---------------------------------------------------------------------------

/// Weighted-ReLU MQA scoring for the lightning indexer.
/// - `q`: (data, sf) with `q_rows = num_tokens * num_heads` rows of `head_dim`
/// - `kv`: (data, sf) with `num_kv_tokens` rows of `head_dim`
/// - `weights`: bf16 `[num_tokens, num_heads]`
/// - `cu_k_start` / `cu_k_end`: per-token KV span `[num_tokens]`
/// - `out`: bf16 `[num_tokens, logits_stride]`; row i stores its span at col 0.
#[allow(clippy::too_many_arguments)]
pub fn mqa_logits(
    dev: &Device,
    stream: &DevStream,
    q: &Operand,
    q_sf: &SfTensor,
    kv: &Operand,
    kv_sf: &SfTensor,
    weights: &DevBuffer,
    cu_k_start: &DevBuffer,
    cu_k_end: &DevBuffer,
    num_tokens: u32,
    num_kv_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    logits: &mut DevBuffer,
    logits_stride: u32,
) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm100) {
        return Err(DgError::Unsupported(
            "the tcgen05 kernels require SM100 (Blackwell)".into(),
        ));
    }
    if q.dtype != kv.dtype {
        return Err(DgError::InvalidArg("Q/KV dtype mismatch".into()));
    }
    if head_dim % 32 != 0 || head_dim > 128 {
        return Err(DgError::InvalidArg("head_dim must be 32/64/128".into()));
    }
    if num_heads % 4 != 0 || num_heads > 256 {
        return Err(DgError::InvalidArg(
            "num_heads must be a multiple of 4 (<= 256)".into(),
        ));
    }
    let is_fp4 = q.dtype == Dtype::Fp4;
    if is_fp4 && head_dim != 64 && head_dim != 128 {
        return Err(DgError::InvalidArg("MXFP4 requires head_dim 64/128".into()));
    }

    // Config: BLOCK_Q * num_heads <= 256 (UMMA_N) and >= 128 (full UTCCP
    // granularity); math = 2 WGs => SPLIT_KV = 256.
    let block_q = (128 / num_heads).clamp(1, 32).max(256 / num_heads).min(32);
    let umma_n = ceil_div(block_q * num_heads, 8) * 8;
    let num_math_threads = 256u32;
    let split_kv = 256u32;
    let (q_stages, kv_stages, tmem_stages) = (2u32, 2u32, 2u32);

    // Tensor maps.
    let tm_q = tma::make_tma_mqa_qk(
        dev,
        q.dtype,
        &q.data,
        q.rows,
        head_dim,
        block_q * num_heads,
        is_fp4,
    )?;
    // Box sizes must equal the producer's per-TMA step (kNumKVTokensPerTMA =
    // 256 for SPLIT_KV=256), otherwise the full barrier under-credits and the
    // stage never completes (latent-hang fix).
    let tm_kv = tma::make_tma_mqa_qk(dev, kv.dtype, &kv.data, kv.rows, head_dim, split_kv, is_fp4)?;
    let tm_sf_q = tma::make_tma_mqa_sf(dev, &q_sf.buf, q.rows, block_q * num_heads)?;
    let tm_sf_kv = tma::make_tma_mqa_sf(dev, &kv_sf.buf, kv.rows, split_kv)?;
    let tm_w = tma::make_tma_mqa_weights(dev, weights, num_heads, num_tokens)?;

    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned num_q_tokens, unsigned num_kv_tokens, unsigned logits_stride,
    const unsigned* cu_k_start, const unsigned* cu_k_end, unsigned short* logits,
    const __grid_constant__ dg::TmaMap tma_q,
    const __grid_constant__ dg::TmaMap tma_sf_q,
    const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_sf_kv,
    const __grid_constant__ dg::TmaMap tma_w) {{
    dg::mqa_logits_sm100_impl<
        {heads}, {head_dim},
        {block_q}, {split_kv}, {umma_n},
        {q_stages}, {kv_stages}, {tmem_stages},
        128, {math_threads},
        {num_sms}, {is_fp4}, false, 0
    >(num_q_tokens, num_kv_tokens, logits_stride, cu_k_start, cu_k_end, logits,
      tma_q, tma_sf_q, tma_kv, tma_sf_kv, tma_w, 0, 0, 0, 0, 0);
}}"#,
        heads = num_heads,
        head_dim = head_dim,
        block_q = block_q,
        split_kv = split_kv,
        umma_n = umma_n,
        q_stages = q_stages,
        kv_stages = kv_stages,
        tmem_stages = tmem_stages,
        math_threads = num_math_threads,
        num_sms = dev.num_sms,
        is_fp4 = is_fp4 as u32,
    );
    let sig = format!("mqa_{num_heads}_{head_dim}_{block_q}_{is_fp4}");
    let func = jit::get_kernel(dev, jit::kernel_src::MQA_LOGITS, "mqa_logits", &sig, &body)?;

    // Shared memory budget.
    let qk_bytes_per_token = if is_fp4 { head_dim / 2 } else { head_dim };
    let smem = block_q * num_heads * qk_bytes_per_token * q_stages
        + split_kv * qk_bytes_per_token * kv_stages
        + block_q * num_heads * 4 * q_stages
        + split_kv * 4 * kv_stages
        + block_q * num_heads * 2 * q_stages
        + 1024u32;

    let args = Args::new()
        .u32(num_tokens)
        .u32(num_kv_tokens)
        .u32(logits_stride)
        .ptr(cu_k_start.ptr as *const u8 as *const u32)
        .ptr(cu_k_end.ptr as *const u8 as *const u32)
        .ptr(logits.ptr as *const u8 as *const u16)
        .tensormap(&tm_q)
        .tensormap(&tm_sf_q)
        .tensormap(&tm_kv)
        .tensormap(&tm_sf_kv)
        .tensormap(&tm_w);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + num_math_threads, 1, 1),
            smem,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

// ---------------------------------------------------------------------------
// MQA logits, PAGED KV variant (decode path): port of
// `sm100_paged_mqa_logits` (scheduler + metadata kernel + main kernel).
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn mqa_logits_paged(
    dev: &Device,
    stream: &DevStream,
    q: &Operand,
    q_sf: &SfTensor,
    kv_pages: &DevBuffer,    // [num_pages, PAGE_KV, head_dim]
    kv_sf_pages: &DevBuffer, // [num_pages, PAGE_KV] int32 (word/token)
    weights: &DevBuffer,     // [num_q_tokens, heads] bf16
    page_kv: u32,
    num_pages: u32,
    context_lens: &DevBuffer, // [num_q_tokens]
    indices: &DevBuffer,      // [num_q_tokens] request ids (sorted)
    block_table: &DevBuffer,  // [num_q_tokens, stride] (page ids per request)
    block_table_stride: u32,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    logits: &mut DevBuffer,
    logits_stride: u32,
) -> DgResult<()> {
    if !matches!(dev.arch, crate::device::Arch::Sm100) {
        return Err(DgError::Unsupported(
            "the tcgen05 kernels require SM100 (Blackwell)".into(),
        ));
    }
    if head_dim % 32 != 0 || head_dim > 128 {
        return Err(DgError::InvalidArg("head_dim must be 32/64/128".into()));
    }
    if num_heads % 4 != 0 || num_heads > 256 {
        return Err(DgError::InvalidArg(
            "num_heads must be a multiple of 4 (<= 256)".into(),
        ));
    }
    if page_kv == 0 || page_kv % 4 != 0 || 256 % page_kv != 0 {
        return Err(DgError::InvalidArg(
            "page_kv must divide 256 (multiple of 4)".into(),
        ));
    }
    let is_fp4 = q.dtype == Dtype::Fp4;
    if q.dtype != Dtype::Fp8 && !is_fp4 {
        return Err(DgError::InvalidArg("paged MQA needs FP8 or FP4 Q".into()));
    }

    let block_q = (128 / num_heads).clamp(1, 32).max(256 / num_heads).min(32);
    let umma_n = ceil_div(block_q * num_heads, 8) * 8;
    let num_math_threads = 256u32;
    let split_kv = 256u32;
    // Deeper KV ring so full-ring reuse (num_kv_splits == kNumKVStages) fires.
    let (q_stages, kv_stages, tmem_stages) = (2u32, 4u32, 2u32);

    let tm_q = tma::make_tma_mqa_qk(
        dev,
        q.dtype,
        &q.data,
        q.rows,
        head_dim,
        block_q * num_heads,
        is_fp4,
    )?;
    let tm_kv =
        tma::make_tma_mqa_qk_paged(dev, q.dtype, kv_pages, head_dim, page_kv, num_pages, is_fp4)?;
    let tm_sf_q = tma::make_tma_mqa_sf(dev, &q_sf.buf, q.rows, block_q * num_heads)?;
    let tm_sf_kv = tma::make_tma_mqa_sf_paged(dev, kv_sf_pages, page_kv, num_pages)?;
    let tm_w = tma::make_tma_mqa_weights(dev, weights, num_heads, num_tokens)?;

    // schedule_meta: [num_sms + 1] uint2 (per-SM starts + sentinel).
    let meta = DevBuffer::alloc(dev, (dev.num_sms as usize + 1) * 8)?;
    let meta_body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    const unsigned* context_lens, const unsigned* indices,
    unsigned num_q_tokens, unsigned* schedule_meta) {{
    dg::mqa_paged_metadata_impl<{split_kv}, {sms}, {block_q}, 128>(
        context_lens, indices, num_q_tokens, schedule_meta);
}}"#,
        split_kv = split_kv,
        sms = dev.num_sms,
        block_q = block_q,
    );
    let meta_sig = format!("meta_{block_q}_{split_kv}");
    let meta_func = jit::get_kernel(
        dev,
        jit::kernel_src::MQA_LOGITS,
        "mqa_meta",
        &meta_sig,
        &meta_body,
    )?;
    let meta_args = Args::new()
        .ptr(context_lens.ptr as *const u8 as *const u32)
        .ptr(indices.ptr as *const u8 as *const u32)
        .u32(num_tokens)
        .devptr(meta.ptr);
    let meta_smem = (2 * num_tokens as usize + 32 / 4 + 1) * 4 + 64;
    jit::launch(
        dev,
        meta_func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (1, 1, 1),
            block: (128, 1, 1),
            smem: meta_smem as u32,
            cluster: None,
            pdl: false,
        },
        meta_args,
    )?;

    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned num_q_tokens, unsigned num_kv_tokens, unsigned logits_stride,
    const unsigned* cu_k_start, const unsigned* cu_k_end, unsigned short* logits,
    const __grid_constant__ dg::TmaMap tma_q,
    const __grid_constant__ dg::TmaMap tma_sf_q,
    const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_sf_kv,
    const __grid_constant__ dg::TmaMap tma_w,
    const unsigned* context_lens, const unsigned* indices,
    const unsigned* block_table, unsigned block_table_stride,
    const unsigned* schedule_meta) {{
    dg::mqa_logits_sm100_impl<
        {heads}, {head_dim},
        {block_q}, {split_kv}, {umma_n},
        {q_stages}, {kv_stages}, {tmem_stages},
        128, {math_threads},
        {num_sms}, {is_fp4},
        true, {page_kv}
    >(num_q_tokens, num_kv_tokens, logits_stride, cu_k_start, cu_k_end, logits,
      tma_q, tma_sf_q, tma_kv, tma_sf_kv, tma_w,
      context_lens, indices, block_table, block_table_stride, schedule_meta);
}}"#,
        heads = num_heads,
        head_dim = head_dim,
        block_q = block_q,
        split_kv = split_kv,
        umma_n = umma_n,
        q_stages = q_stages,
        kv_stages = kv_stages,
        tmem_stages = tmem_stages,
        math_threads = num_math_threads,
        num_sms = dev.num_sms,
        is_fp4 = is_fp4 as u32,
        page_kv = page_kv,
    );
    let sig = format!("mqa_paged_{num_heads}_{head_dim}_{block_q}_{is_fp4}_{page_kv}");
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::MQA_LOGITS,
        "mqa_logits_paged",
        &sig,
        &body,
    )?;

    let qk_bytes_per_token = if is_fp4 { head_dim / 2 } else { head_dim };
    let smem = block_q * num_heads * qk_bytes_per_token * q_stages
        + split_kv * qk_bytes_per_token * kv_stages
        + block_q * num_heads * 4 * q_stages
        + split_kv * 4 * kv_stages
        + block_q * num_heads * 2 * q_stages
        + 1024u32;

    let args = Args::new()
        .u32(num_tokens)
        .u32(num_pages * page_kv)
        .u32(logits_stride)
        .devptr(0)
        .devptr(0)
        .ptr(logits.ptr as *const u8 as *const u16)
        .tensormap(&tm_q)
        .tensormap(&tm_sf_q)
        .tensormap(&tm_kv)
        .tensormap(&tm_sf_kv)
        .tensormap(&tm_w)
        .ptr(context_lens.ptr as *const u8 as *const u32)
        .ptr(indices.ptr as *const u8 as *const u32)
        .ptr(block_table.ptr as *const u8 as *const u32)
        .u32(block_table_stride)
        .devptr(meta.ptr);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (128 + num_math_threads, 1, 1),
            smem,
            cluster: None,
            pdl: true,
        },
        args,
    )
}

// ---------------------------------------------------------------------------
// Dynamic-output FP8 quantization (upstream QuantizeToFP8 contract, standalone
// kernel form: fused epilogues are specified to bitwise match this cast).
// ---------------------------------------------------------------------------
/// Quantize an FP32 buffer `[m, n]` (row stride `in_stride`) into E4M3 with
/// per-(row, 32-col-group) UE8M0 dynamic scales; also emits the packed SF
/// words `[ceil(n/32)/4, tma_aligned(m)]` (int32).
#[allow(clippy::too_many_arguments)]
pub fn quantize_output_fp8(
    dev: &Device,
    stream: &DevStream,
    src: &DevBuffer,
    m: u32,
    n: u32,
    in_stride: u32,
    stochastic: bool,
) -> DgResult<(DevBuffer, DevBuffer)> {
    if n % 32 != 0 {
        return Err(DgError::InvalidArg(
            "quantize_output_fp8 needs n % 32 == 0".into(),
        ));
    }
    let out = DevBuffer::alloc(dev, (m * n) as usize)?;
    let tma_aligned = heuristics::tma_aligned_size(m, 4);
    let sf_words = (n / 32).div_ceil(4) as usize * tma_aligned as usize;
    let sfd = DevBuffer::alloc(dev, sf_words * 4)?;
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    const float* in, unsigned char* out, int* sfd,
    unsigned m, unsigned n, unsigned is_, unsigned os_, unsigned ss_, unsigned ta) {{
    dg::quantize_output_fp8_impl<256, {stoch}>
        (in, out, sfd, m, n, is_, os_, ss_, ta);
}}"#,
        stoch = stochastic as u32,
    );
    let sig = format!("qo8_{stochastic}");
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::LAYOUT_QUANT,
        "quant_out_fp8",
        &sig,
        &body,
    )?;
    let args = Args::new()
        .devptr(src.ptr)
        .devptr(out.ptr)
        .devptr(sfd.ptr)
        .u32(m)
        .u32(n)
        .u32(in_stride)
        .u32(n)
        .u32(tma_aligned)
        .u32(tma_aligned);
    let total = m * (n / 32);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (total.div_ceil(256), 1, 1),
            block: (256, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;
    Ok((out, sfd))
}

// ===========================================================================
// Megakernel entry points (mega_moe / mega_gate / mega_mhc)
// ===========================================================================
use crate::mega_layout as mlayout;

/// Build the 592-byte by-value SymBuffer param:
/// base (local slice) + per-rank offsets + rank_idx + num_ranks.
#[allow(dead_code)] // used by the multi-rank mega_moe path (SymBuffer ABI)
fn encode_sym_buffer(base: u64, rank_ptrs: &[u64], rank_idx: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(592);
    b.extend_from_slice(&base.to_ne_bytes());
    for r in 0..mlayout::MAX_RANKS {
        let off = if r < rank_ptrs.len() {
            (rank_ptrs[r] as i64) - (base as i64)
        } else {
            0i64
        };
        b.extend_from_slice(&off.to_ne_bytes());
    }
    b.extend_from_slice(&rank_idx.to_ne_bytes());
    b.extend_from_slice(&(rank_ptrs.len() as u32).to_ne_bytes());
    b
}

/// Host-side weight preprocessing (mirror of the Python layer):
/// gate/up interleave (gran 8) along N for the data plane.
pub fn interleave_weights(
    dev: &Device,
    stream: &DevStream,
    w: &DevBuffer, // fp8 [G, N, K] K-major
    g: u32,
    n: u32,
    k: u32,
) -> DgResult<DevBuffer> {
    let out = DevBuffer::alloc(dev, (g as usize) * (n as usize) * (k as usize))?;
    let body = r#"extern "C" __global__ void __dg_kernel(
    const unsigned char* w, unsigned char* out, unsigned G, unsigned N, unsigned K) {
    dg::mega_moe::interleave_weights_impl(w, out, G, N, K, 8);
}"#
    .to_string();
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::MEGA_MOE,
        "interleave",
        &format!("iwc_{g}_{n}_{k}"),
        &body,
    )?;
    let args = Args::new()
        .ptr(w.ptr as *const u8 as *const u32 as *const u8)
        .ptr(out.ptr as *const u8 as *const u32 as *const u8)
        .u32(g)
        .u32(n)
        .u32(k);
    let total = (g as u64) * (n as u64) * (k as u64);
    let block = 256u32;
    let grid = total.div_ceil(block as u64).max(1) as u32;
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;
    Ok(out)
}

/// SF 4x32 transposition within each 128-row page (UTCCP consumption).
pub fn transpose_sf_utccp(
    dev: &Device,
    stream: &DevStream,
    sf: &DevBuffer, // int32 [G, N, K/128]
    g: u32,
    n: u32,
    k: u32,
) -> DgResult<DevBuffer> {
    let packed = k / 128;
    let out = DevBuffer::alloc(dev, (g as usize) * (n as usize) * (packed as usize) * 4)?;
    let body = r#"extern "C" __global__ void __dg_kernel(
    const int* sf, int* out, unsigned G, unsigned N, unsigned P) {
    dg::mega_moe::transpose_sf_impl(sf, out, G, N, P);
}"#
    .to_string();
    let func = jit::get_kernel(
        dev,
        jit::kernel_src::MEGA_MOE,
        "tsf",
        &format!("tsf_{g}_{n}_{k}"),
        &body,
    )?;
    let args = Args::new()
        .ptr(sf.ptr as *const u8 as *const u32 as *const u8)
        .ptr(out.ptr as *const u8 as *const u32 as *const u8)
        .u32(g)
        .u32(n)
        .u32(packed);
    let total = (g as u64) * (n as u64) * (packed as u64);
    let block = 256u32;
    let grid = total.div_ceil(block as u64).max(1) as u32;
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            smem: 0,
            cluster: None,
            pdl: false,
        },
        args,
    )?;
    Ok(out)
}

/// Routing arguments for mega_gate (mirrors upstream `RoutingArgs`).
#[allow(clippy::too_many_arguments)]
pub struct RoutingSpec {
    pub to_physical_map: Option<DevBuffer>, // int32 [logical, duplicates]
    pub logical_count: Option<DevBuffer>,   // int32 [physical]
    pub topk_idx: DevBuffer,                // i64 [T, topk+shared]
    pub unmapped_topk_idx: Option<DevBuffer>, // i64 [T, topk] (stride = topk)
    pub topk_weights: DevBuffer,            // f32 [T, topk+shared]
    pub num_routed_experts: u32,
    pub num_shared_experts: u32,
    pub num_duplicate_experts: u32,
    pub rank_idx: u32,
    pub routed_scaling_factor: f32,
}

fn encode_routing_args(r: &RoutingSpec, num_topk: u32) -> Vec<u8> {
    let p = |b: &Option<DevBuffer>| b.as_ref().map(|b| b.ptr).unwrap_or(0);
    let mut v = Vec::with_capacity(64);
    v.extend_from_slice(&p(&r.to_physical_map).to_ne_bytes());
    v.extend_from_slice(&p(&r.logical_count).to_ne_bytes());
    v.extend_from_slice(&r.topk_idx.ptr.to_ne_bytes());
    v.extend_from_slice(&p(&r.unmapped_topk_idx).to_ne_bytes());
    v.extend_from_slice(&r.topk_weights.ptr.to_ne_bytes());
    v.extend_from_slice(&(num_topk as u64).to_ne_bytes()); // stride
    v.extend_from_slice(&r.num_routed_experts.to_ne_bytes());
    v.extend_from_slice(&r.num_shared_experts.to_ne_bytes());
    v.extend_from_slice(&r.num_duplicate_experts.to_ne_bytes());
    v.extend_from_slice(&r.rank_idx.to_ne_bytes());
    let sf = r.routed_scaling_factor.to_bits();
    v.extend_from_slice(&sf.to_ne_bytes());
    v
}

/// fp8 MegaGate: x [T, H] e4m3 + x_sf [T, H/128] int32; weight
/// [E_aligned, H] e4m3 + weight_sf [E_aligned, H/128] int32 (both K-major,
/// quantized with our quant_mx/transform path).
#[allow(clippy::too_many_arguments)]
pub fn mega_gate(
    dev: &Device,
    stream: &DevStream,
    x: &DevBuffer,
    x_sf: &DevBuffer,
    weight: &DevBuffer,
    weight_sf: &DevBuffer,
    bias: Option<&DevBuffer>,
    image_bias: Option<&DevBuffer>,
    image_token_mask: Option<&DevBuffer>,
    mask: Option<&DevBuffer>,
    fix_routing_mask: Option<&DevBuffer>,
    force_random: Option<&DevBuffer>,
    routing: &RoutingSpec,
    num_tokens: u32,
    hidden: u32,
    num_topk: u32,
    deterministic: bool,
) -> DgResult<(DevBuffer, DevBuffer)> {
    mlayout::require_blackwell(dev)?;
    let cfg = heuristics::get_mega_gate_config(
        num_tokens,
        hidden,
        routing.num_routed_experts,
        dev.num_sms,
        deterministic,
    )?;
    let aligned = mlayout::align_up(routing.num_routed_experts, mlayout::GATE_EXPERT_ALIGNMENT);
    let token_blocks = mlayout::ceil_div(num_tokens, cfg.block_tokens);
    let scratch = DevBuffer::alloc(
        dev,
        mlayout::gate_scratch_bytes(token_blocks, cfg.num_split_k, cfg.block_tokens, aligned)
            as usize,
    )?;
    // Score barriers: 128B lines, zeroed once (grid tags make resets unneeded).
    let bars_bytes = (token_blocks as u64).div_ceil(128).max(1) as usize * 128;
    let bars = mhc::alloc_zeroed(dev, bars_bytes, stream.raw())?;

    let tm_x = tma::make_tma_ab(
        dev,
        Dtype::Fp8,
        Major::K,
        x,
        num_tokens,
        hidden,
        cfg.load_block_m,
        mlayout::GATE_BLOCK_K,
        hidden,
        1,
        mlayout::GATE_SWIZZLE,
        false,
    )?;
    let tm_w = tma::make_tma_ab(
        dev,
        Dtype::Fp8,
        Major::K,
        weight,
        aligned,
        hidden,
        cfg.load_block_n,
        mlayout::GATE_BLOCK_K,
        hidden,
        1,
        mlayout::GATE_SWIZZLE,
        false,
    )?;
    // SF maps: full-tile SMEM boxes (SF covers the whole block for the MMA).
    let tm_x_sf = tma::make_tma_sf(
        dev,
        x_sf,
        num_tokens,
        hidden,
        32,
        mlayout::align_up(mlayout::GATE_MAX_BLOCK_TOKENS, 128),
        hidden / 128,
        1,
        0,
    )?;
    let tm_w_sf = tma::make_tma_sf(
        dev,
        weight_sf,
        aligned,
        hidden,
        32,
        mlayout::align_up(aligned / cfg.num_expert_groups, 128),
        hidden / 128,
        1,
        0,
    )?;

    let padded = (routing.num_routed_experts != aligned) as u32;
    let has_bias = bias.is_some() as u32;
    let has_img = image_token_mask.is_some() as u32;
    let has_mask = mask.is_some() as u32;
    let has_fix = fix_routing_mask.is_some() as u32;
    let has_rand = force_random.is_some() as u32;
    let has_map = routing.to_physical_map.is_some() as u32;
    let has_unmapped = routing.unmapped_topk_idx.is_some() as u32;
    let sig = format!(
        "gate_{}_{}_{}_{}_{}_{}_{}{}_{}{}{}{}{}{}{}",
        hidden,
        aligned,
        cfg.num_gate_threads,
        cfg.num_mma_ctas,
        cfg.num_split_k,
        cfg.num_expert_groups,
        num_topk,
        padded,
        has_bias,
        has_img,
        has_mask,
        has_fix,
        has_rand,
        has_map,
        has_unmapped
    );
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    const __grid_constant__ dg::TmaMap tmx,
    const __grid_constant__ dg::TmaMap tmw,
    const __grid_constant__ dg::TmaMap tmxs,
    const __grid_constant__ dg::TmaMap tmws,
    const float* bias, const float* image_bias, const bool* image_token_mask,
    const bool* mask, const bool* fix_routing_mask, const bool* force_random,
    const __grid_constant__ dg::mega_gate::RoutingArgs ra,
    void* scratch, void* bars,
    unsigned num_tokens, unsigned block_tokens, unsigned num_stages,
    unsigned num_worker_groups, unsigned num_token_blocks) {{
    dg::mega_gate::fp8_mega_gate_impl<
        {k}, {e},
        {gt},
        {mma}, {sk}, {eg},
        {topk},
        {padded}, {hb}, {hi},
        {hm}, {hf}, {hr},
        {hmap}, {hu}
    >(tmx, tmw, tmxs, tmws, bias, image_bias, image_token_mask, mask,
      fix_routing_mask, force_random, ra, scratch, bars,
      num_tokens, block_tokens, num_stages, num_worker_groups, num_token_blocks);
}}"#,
        k = hidden,
        e = aligned,
        gt = cfg.num_gate_threads,
        mma = cfg.num_mma_ctas,
        sk = cfg.num_split_k,
        eg = cfg.num_expert_groups,
        topk = num_topk,
        padded = padded,
        hb = has_bias,
        hi = has_img,
        hm = has_mask,
        hf = has_fix,
        hr = has_rand,
        hmap = has_map,
        hu = has_unmapped
    );
    let func = jit::get_kernel(dev, jit::kernel_src::MEGA_GATE, "mega_gate", &sig, &body)?;

    let opt = |b: &Option<&DevBuffer>| b.map(|b| b.ptr).unwrap_or(0);
    let args = Args::new()
        .tensormap(&tm_x)
        .tensormap(&tm_w)
        .tensormap(&tm_x_sf)
        .tensormap(&tm_w_sf)
        .u64(opt(&bias))
        .u64(opt(&image_bias))
        .u64(opt(&image_token_mask))
        .u64(opt(&mask))
        .u64(opt(&fix_routing_mask))
        .u64(opt(&force_random))
        .bytes(encode_routing_args(routing, num_topk))
        .u64(scratch.ptr as u64)
        .u64(bars.ptr as u64)
        .u32(num_tokens)
        .u32(cfg.block_tokens)
        .u32(cfg.num_stages)
        .u32(cfg.num_worker_groups)
        .u32(token_blocks);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (cfg.num_launch_sms, 1, 1),
            block: (128 + cfg.num_gate_threads, 1, 1),
            smem: cfg.smem_size,
            cluster: Some((cfg.num_mma_ctas, 1, 1)),
            pdl: true,
        },
        args,
    )?;
    Ok((scratch, bars))
}

/// mHC: full fused connector layer. `shifted_prev_mix == None` selects the
/// normal (non-shifted) variant. FP8 outputs: `y_fp8` + `y_primary_sf`
/// (either the TMA-aligned GEMM layout, or routed+shared with SF_BLOCK_M>0).
#[allow(clippy::too_many_arguments)]
pub fn mega_mhc(
    dev: &Device,
    stream: &DevStream,
    x: &DevBuffer,                        // bf16 [T, H]
    residual: &DevBuffer,                 // bf16 [T, R, H]
    post_mix: &DevBuffer,                 // f32 [T, R]
    comb_res_mix: &DevBuffer,             // f32 [T, R, R]
    shifted_prev_mix: Option<&DevBuffer>, // f32 [T, R]
    fn_weights: &DevBuffer,               // f32 [24, R*H]
    mix_scales: &DevBuffer,               // f32 [3]
    mix_bases: &DevBuffer,                // f32 [24]
    hc_norm_eps: f32,
    hc_pre_eps: f32,
    hc_post_scale: f32,
    sinkhorn_eps: f32,
    num_sinkhorn_iters: u32,
    rmsnorm_weight: &DevBuffer, // bf16 [H]
    rmsnorm_eps: f32,
    new_residual: &DevBuffer, // bf16 [T, R, H] out
    new_prev_mix: Option<&DevBuffer>,
    new_post_mix: &DevBuffer,
    new_comb_res_mix: &DevBuffer,
    y_bf16: Option<&mut DevBuffer>,
    y_fp8: Option<&DevBuffer>,
    y_primary_sf: Option<&DevBuffer>,
    y_shared_sf: Option<&DevBuffer>,
    primary_sf_stride_word: u32,
    shared_sf_stride_word: u32,
    shared_sf_block_m: u32,
    rmsnorm_scale: f32,
    num_tokens: u32,
    hidden: u32,
) -> DgResult<DevBuffer> {
    mlayout::require_blackwell(dev)?;
    if num_tokens == 0 {
        return DevBuffer::alloc(dev, 0);
    }
    let is_shifted = shifted_prev_mix.is_some();
    if y_fp8.is_some() != y_primary_sf.is_some() {
        return Err(DgError::InvalidArg("y_fp8 requires y_primary_sf".into()));
    }
    if y_shared_sf.is_some() != (shared_sf_block_m > 0) {
        return Err(DgError::InvalidArg(
            "y_shared_sf requires shared_sf_block_m > 0".into(),
        ));
    }

    let num_splits = heuristics::mhc_splits(num_tokens, hidden, dev.num_sms, false);
    let m_blocks = mlayout::mhc_num_m_blocks(num_tokens);
    let scratch = DevBuffer::alloc(
        dev,
        mlayout::mhc_scratch_bytes(m_blocks, num_splits) as usize,
    )?;
    let split_bars = mhc::shared_split_bars(dev, stream.raw())?;
    // Snapshot the optional outputs (the &mut is consumed by encode below).
    let y_bf16_buf: Option<&DevBuffer> = y_bf16.as_deref();
    let _has_y_bf16 = y_bf16.is_some();

    let tm_res = tma::make_tma_mhc_residual(dev, residual, hidden, num_tokens)?;
    let tm_x = tma::make_tma_mhc_hidden(dev, x, hidden, num_tokens)?;
    let tm_fn = tma::make_tma_mhc_fn(dev, fn_weights, hidden)?;
    let tm_post = tma::make_tma_mhc_coeff(
        dev,
        post_mix,
        mlayout::NUM_DEVICE_LOCALITY_DOMAINS.min(4),
        num_tokens,
        0,
    )?;
    let tm_comb = tma::make_tma_mhc_coeff(dev, comb_res_mix, 16, num_tokens, 64)?;
    let tm_prev =
        tma::make_tma_mhc_coeff(dev, shifted_prev_mix.unwrap_or(post_mix), 4, num_tokens, 0)?;
    let tm_new_res = tma::make_tma_mhc_residual(dev, new_residual, hidden, num_tokens)?;
    let tm_y = tma::make_tma_mhc_hidden(dev, y_bf16_buf.unwrap_or(x), hidden, num_tokens)?;

    // Snapshot the optional outputs (the &mut is consumed by encode below).
    let _y_bf16_buf: Option<&DevBuffer> = y_bf16.as_deref();
    let has_y_bf16 = y_bf16.is_some();
    // MixArgs + NormArgs by-value payloads.
    let ma = mhc::encode_mix_args(
        dev,
        mix_scales,
        mix_bases,
        new_prev_mix,
        new_post_mix,
        new_comb_res_mix,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        num_sinkhorn_iters,
    );
    let na = mhc::encode_norm_args(
        dev,
        num_tokens,
        rmsnorm_weight,
        new_residual,
        rmsnorm_eps,
        rmsnorm_scale,
        y_bf16,
        y_fp8,
        y_primary_sf,
        primary_sf_stride_word,
        y_shared_sf,
        shared_sf_stride_word,
    );

    let sb = is_shifted as u32;
    let bf16o = has_y_bf16 as u32;
    let fp8o = y_fp8.is_some() as u32;
    let sig = format!(
        "mhc_{}_{}_{}_{}{}{}_{shared_sf_block_m}",
        hidden, num_splits, dev.num_sms, sb, bf16o, fp8o
    );
    let body = format!(
        r#"extern "C" __global__ void __dg_kernel(
    const __grid_constant__ dg::TmaMap tmr,
    const __grid_constant__ dg::TmaMap tmx,
    const __grid_constant__ dg::TmaMap tmf,
    const __grid_constant__ dg::TmaMap tmp,
    const __grid_constant__ dg::TmaMap tmc,
    const __grid_constant__ dg::TmaMap tppv,
    const __grid_constant__ dg::TmaMap tmnr,
    const __grid_constant__ dg::TmaMap tmy,
    const __grid_constant__ dg::mega_mhc::MixArgs ma,
    const __grid_constant__ dg::mega_mhc::NormArgs na,
    void* scratch, unsigned long long* bars) {{
    dg::mega_mhc::mega_mhc_impl<
        {h}, {s}, {sms},
        {sh}, {b16}, {f8}, {sfm}
    >(tmr, tmx, tmf, tmp, tmc, tppv, tmnr, tmy, ma, na, scratch, bars);
}}"#,
        h = hidden,
        s = num_splits,
        sms = dev.num_sms,
        sh = sb,
        b16 = bf16o,
        f8 = fp8o,
        sfm = shared_sf_block_m
    );
    let func = jit::get_kernel(dev, jit::kernel_src::MEGA_MHC, "mega_mhc", &sig, &body)?;

    let args = Args::new()
        .tensormap(&tm_res)
        .tensormap(&tm_x)
        .tensormap(&tm_fn)
        .tensormap(&tm_post)
        .tensormap(&tm_comb)
        .tensormap(&tm_prev)
        .tensormap(&tm_new_res)
        .tensormap(&tm_y)
        .bytes(ma)
        .bytes(na)
        .u64(scratch.ptr)
        .u64(split_bars.ptr as u64);
    jit::launch(
        dev,
        func,
        stream.raw(),
        &sys::LaunchEx {
            grid: (dev.num_sms, 1, 1),
            block: (768, 1, 1),
            smem: mhc::shared_storage_bytes(),
            cluster: None,
            pdl: true,
        },
        args,
    )?;
    Ok(scratch)
}

/// mHC host-side helpers (smem sizing, split-barrier storage, ABI payloads).
pub mod mhc {
    use super::*;

    /// Zeroed device buffer (stream-ordered cuMemsetD8Async).
    pub fn alloc_zeroed(dev: &Device, bytes: usize, stream: sys::Stream) -> DgResult<DevBuffer> {
        let b = DevBuffer::alloc(dev, bytes)?;
        if bytes > 0 {
            dev.bind()?;
            sys::memset_d8(b.ptr, 0, bytes, stream)?;
        }
        Ok(b)
    }

    /// Split-barrier storage (grid tags need no reset after first zeroing).
    pub fn shared_split_bars(dev: &Device, stream: sys::Stream) -> DgResult<DevBuffer> {
        alloc_zeroed(
            dev,
            crate::mega_layout::mhc_split_barrier_bytes() as usize,
            stream,
        )
    }

    /// sizeof(SharedStorage) — placement mirrors the kernel struct exactly.
    pub fn shared_storage_bytes() -> u32 {
        const ALIGN: u32 = 1024;
        const SWZ: u32 = 128;
        let mut off = 0u32;
        let mut place = |n: u32, a: u32| -> u32 {
            off = off.div_ceil(a) * a;
            let base = off;
            off += n;
            base
        };
        // residual[4][64*32] u16 + x[64*32] u16 (1024-aligned each)
        place(4 * 64 * 32 * 2, ALIGN);
        place(64 * 32 * 2, ALIGN);
        // fn[2][2][4][24*32] f32
        place(2 * 2 * 4 * 24 * 32 * 4, ALIGN);
        // coeffs (128-aligned)
        place(2 * 64 * 4 * 4, SWZ); // pre
        place(2 * 64 * 4 * 4, SWZ); // post
        place(2 * 64 * 16 * 4, SWZ); // comb
                                     // stats + scratch
        place(2 * 64 * 4, SWZ); // hc_norm_sqr_sums
        place(2 * 64 * 4, SWZ); // x1_sqr_sums
        place(64 * 4, SWZ); // normal_scratch
                            // barriers: 12 io + 4 fn + 8 a + 4 accum + 4 coeff + 2 stats + 1 mix
        place(35 * 8, 8);
        place(8, 4); // tmem_base + ticket
        off
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_mix_args(
        dev: &Device,
        scales: &DevBuffer,
        bases: &DevBuffer,
        new_prev: Option<&DevBuffer>,
        new_post: &DevBuffer,
        new_comb: &DevBuffer,
        hc_norm_eps: f32,
        hc_pre_eps: f32,
        hc_post_scale: f32,
        sinkhorn_eps: f32,
        iters: u32,
    ) -> Vec<u8> {
        let _ = dev;
        let mut v = Vec::with_capacity(48);
        v.extend_from_slice(&scales.ptr.to_ne_bytes());
        v.extend_from_slice(&bases.ptr.to_ne_bytes());
        v.extend_from_slice(&(new_prev.map(|b| b.ptr).unwrap_or(0)).to_ne_bytes());
        v.extend_from_slice(&new_post.ptr.to_ne_bytes());
        v.extend_from_slice(&new_comb.ptr.to_ne_bytes());
        for f in [hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps] {
            v.extend_from_slice(&f.to_bits().to_ne_bytes());
        }
        v.extend_from_slice(&iters.to_ne_bytes());
        v
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_norm_args(
        dev: &Device,
        num_tokens: u32,
        weight: &DevBuffer,
        new_residual: &DevBuffer,
        eps: f32,
        scale: f32,
        y_bf16: Option<&mut DevBuffer>,
        y_fp8: Option<&DevBuffer>,
        y_primary_sf: Option<&DevBuffer>,
        primary_stride_word: u32,
        y_shared_sf: Option<&DevBuffer>,
        shared_stride_word: u32,
    ) -> Vec<u8> {
        let _ = dev;
        let _p = |o: &Option<&DevBuffer>| o.map(|b| b.ptr).unwrap_or(0);
        let mut v = Vec::with_capacity(96);
        v.extend_from_slice(&num_tokens.to_ne_bytes());
        v.extend_from_slice(&weight.ptr.to_ne_bytes());
        v.extend_from_slice(&new_residual.ptr.to_ne_bytes());
        v.extend_from_slice(&eps.to_bits().to_ne_bytes());
        v.extend_from_slice(&scale.to_bits().to_ne_bytes());
        v.extend_from_slice(&(y_bf16.as_ref().map(|b| b.ptr).unwrap_or(0)).to_ne_bytes());
        let pf8 = y_fp8.map(|b| b.ptr).unwrap_or(0);
        let psf = y_primary_sf.map(|b| b.ptr).unwrap_or(0);
        v.extend_from_slice(&pf8.to_ne_bytes());
        v.extend_from_slice(&psf.to_ne_bytes());
        v.extend_from_slice(&(primary_stride_word as u64).to_ne_bytes());
        let pshared = y_shared_sf.map(|b| b.ptr).unwrap_or(0);
        v.extend_from_slice(&pshared.to_ne_bytes());
        v.extend_from_slice(&(shared_stride_word as u64).to_ne_bytes());
        v
    }
}
