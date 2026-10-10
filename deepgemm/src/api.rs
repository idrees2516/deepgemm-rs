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
    const __grid_constant__ dg::TmaMap tma_cd) {{
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
        {num_sms}
    >(grouped_layout, num_groups, shape_m, shape_n, shape_k,
      tma_a, tma_b, tma_sfa, tma_sfb, tma_cd);
}}"#,
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
    )
}

struct LaunchCtx<'a> {
    dev: &'a Device,
    stream: sys::Stream,
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
        GemmType::Normal => 1,
        _ => num_groups,
    };
    // SF maps: groups (or batches) stack along the packed-K outer dim.
    let sf_groups_a = match gemm_type {
        GemmType::Normal | GemmType::MGroupedContiguous => 1,
        _ => num_groups,
    };
    let sf_groups_b = match gemm_type {
        GemmType::Normal => 1,
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
    let tm_sfa = if has_sf {
        let sf = a.sf.as_ref().unwrap();
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
    } else {
        tma::make_tma_sf(dev, &DevBuffer::alloc(dev, 128)?, 32, 128, 32, 32, 1, 1, 0)?
    };
    let tm_sfb = if has_sf {
        let sf = b.sf.as_ref().unwrap();
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
    } else {
        tma::make_tma_sf(dev, &DevBuffer::alloc(dev, 128)?, 32, 128, 32, 32, 1, 1, 0)?
    };
    let cd_atom_n = s.swizzle_cd_mode / out.dtype.elem_size() as u32;
    let tm_cd = if gemm_type == GemmType::Batched {
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
    );
    let sig = format!(
        "{:?}",
        (cfg, gemm_type, a.dtype, b.dtype, out.dtype, gran_a, gran_b, accumulate)
    );
    let func = jit::get_kernel(dev, jit::kernel_src::GEMM_SM100, "gemm_sm100", &sig, &body)?;

    let gl_ptr = grouped_layout.map(|b| b.ptr).unwrap_or(0);
    let args = Args::new()
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

/// Dense GEMM: `D (+)= A @ B^T` with NT layout (K-major A/B).
pub fn gemm_nt(
    dev: &Device,
    stream: &DevStream,
    a: &Operand,
    b: &Operand,
    out: &mut Output,
    accumulate: bool,
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
    )
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
    if a.major != Major::K || b.major != Major::K {
        return Err(DgError::InvalidArg(
            "m-grouped contiguous requires K-major A/B".into(),
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
    )
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
    let args = Args::new()
        .ptr(in_buf.ptr as *const u8 as *const u32)
        .ptr(out.ptr as *const u8 as *const u32)
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
    )?;

    Ok(SfTensor {
        rows: packed_rows,
        cols: tma_aligned,
        gran,
        buf: out,
    })
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
