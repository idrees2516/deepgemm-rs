//! Einsum kernels (port of upstream `csrc/apis/einsum.hpp`).
//!
//! Four hardcoded BF16 equations and three FP8 equations cover the GLM /
//! hyperconnection attention projections:
//!
//! ```text
//! BF16 (einsum):
//!   bmk,bnk->mn   MLA score-space projection: sum_b A[b,m,k] B[b,n,k]
//!                 -> the dedicated split-K BMK/BNK kernel (api_bmk), FP32
//!                 accumulation; a BF16 D runs through an FP32 workspace and
//!                 an RNE cast (upstream's implicit `d.copy_(workspace)`).
//!   bhr,hdr->bhd  attention out-proj: per (b,h) [r] x [d,r]^T
//!                 -> Batched GEMM (batch=h, m=b, n=d, k=r)
//!   bhd,hdr->bhr  attention v-proj:   (batch=h, m=b, n=r, k=d), B MN-major
//!   bhd,bhr->hdr  o-proj weight-side reduce over b: K dim is the batch
//!                 -> cuBLASLt upstream; here too (`crate::cublaslt`)
//!
//! FP8 (fp8_einsum): the same first three equations as permuted-view FP8
//!   BMMs with these (batch, m, n, k) tuples (upstream einsum.hpp):
//!     bhr,hdr->bhd : (h, b, d, r)
//!     bhd,hdr->bhr : (h, b, r, d)
//!     bhd,bhr->hdr : (h, d, r, b)
//!   Scale-factor contracts (exactly the upstream generators'):
//!     eq1: A per-token [b*h, r/gran] ((b,h) order); B per-128x128-block
//!          [h][d/128][r/gran] (legacy recipe, broadcast along mn)
//!     eq2: A per-token [b*h, d/gran]; B per-block [h][d/128][r/gran]
//!          serving the transposed [r, d] view — per-token along the
//!          GEMM's n (r), per-128 along its k (d)
//!     eq3: A per-channel [ceil(b/gran), h*d] (one scale per (h,d) column
//!          over the batch); B per-channel [ceil(b/gran), h*r]
//! ```
//!
//! The GEMM-backed equations never move data: they build 3D TMA maps with
//! EXPLICIT strides over the user's contiguous tensors (upstream
//! `make_tma_3d_desc` + tensor `permute` views) and launch the unified SM100
//! kernel through [`crate::api::launch_gemm_with_maps`]. The batch dim (h)
//! is an *interior* memory dim for the b/hr-side tensors — that is why the
//! maps cannot reuse the plain BMM builders, which assume the batch is the
//! outermost dim.

use crate::api::{cast_to_bf16, transform_sf_batched, LaunchCtx};
use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::heuristics;
use crate::tma;
use crate::types::{Dtype, GemmType, Major, Operand, Output, SfGran};

/// A contiguous 3D device tensor `[d0, d1, d2]` (row-major, d2 inner).
pub struct Tensor3 {
    pub data: DevBuffer,
    pub dims: [u32; 3],
}

impl Tensor3 {
    pub fn new(data: DevBuffer, dims: [u32; 3]) -> Tensor3 {
        Tensor3 { data, dims }
    }

    /// Row-major element strides.
    fn strides(&self) -> [u64; 3] {
        [(self.dims[1] * self.dims[2]) as u64, self.dims[2] as u64, 1]
    }

    fn check(&self, dtype: Dtype, what: &str) -> DgResult<()> {
        let need = self.dims.iter().map(|&d| d as u64).product::<u64>() * dtype.elem_size() as u64;
        if (self.data.len as u64) < need {
            return Err(DgError::InvalidArg(format!(
                "{what}: buffer holds {} bytes, shape {:?} of {dtype:?} needs {need}",
                self.data.len, self.dims
            )));
        }
        Ok(())
    }
}

/// SMEM box inner extent: the swizzle atom when swizzled, else the full
/// gmem inner dim (unswizzled boxes span the whole inner extent).
fn box_inner(swz: u32, elem: u32, inner: u32) -> u32 {
    if swz != 0 {
        swz / elem
    } else {
        inner
    }
}

// ---------------------------------------------------------------------------
// BF16 equations
// ---------------------------------------------------------------------------

/// Strided Batched GEMM over the unified SM100 kernel (the shared engine of
/// the GEMM-backed einsum equations).
#[allow(clippy::too_many_arguments)]
fn batched_gemm_strided(
    dev: &Device,
    stream: &DevStream,
    // (buf, inner, outer, batch, stride_outer, stride_batch, mn_major)
    a: (&DevBuffer, u32, u32, u32, u64, u64, bool),
    b: (&DevBuffer, u32, u32, u32, u64, u64, bool),
    cd: (&DevBuffer, u32, u32, u32, u64, u64),
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    accumulate: bool,
) -> DgResult<()> {
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    let a_op = Operand {
        dtype: Dtype::Bf16,
        major: if a.6 { Major::Mn } else { Major::K },
        rows: m,
        k,
        outer_stride: a.4 as u32,
        sf: None,
        data: a.0.clone_view(),
    };
    let b_op = Operand {
        dtype: Dtype::Bf16,
        major: if b.6 { Major::Mn } else { Major::K },
        rows: n,
        k,
        outer_stride: b.4 as u32,
        sf: None,
        data: b.0.clone_view(),
    };
    let out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: cd.0.clone_view(),
    };
    let desc = heuristics::GemmDesc {
        gemm_type: GemmType::Batched,
        m,
        n,
        k,
        num_groups: batch,
        a_dtype: Dtype::Bf16,
        b_dtype: Dtype::Bf16,
        cd_dtype: Dtype::Bf16,
        major_a: a_op.major,
        major_b: b_op.major,
        num_sms: dev.num_sms,
        smem_capacity: dev.smem_capacity,
        expected_m: m,
        expected_num_groups: 1,
        expected_k: 0,
        with_accumulation: accumulate,
    };
    let cfg = heuristics::get_best_config(&desc)?;
    let s = &cfg.storage;
    let elem = Dtype::Bf16.elem_size() as u32;

    let tm_a = tma::make_tma_3d_strided(
        dev,
        Dtype::Bf16,
        a.0,
        a.1,
        a.2,
        a.3,
        box_inner(s.swizzle_a_mode, elem, a.1),
        s.load_block_m,
        a.4,
        a.5,
        s.swizzle_a_mode,
    )?;
    let tm_b = tma::make_tma_3d_strided(
        dev,
        Dtype::Bf16,
        b.0,
        b.1,
        b.2,
        b.3,
        box_inner(s.swizzle_b_mode, elem, b.1),
        s.load_block_n,
        b.4,
        b.5,
        s.swizzle_b_mode,
    )?;
    let cd_atom_n = s.swizzle_cd_mode / elem;
    let tm_cd = tma::make_tma_3d_strided(
        dev,
        Dtype::Bf16,
        cd.0,
        cd.1,
        cd.2,
        cd.3,
        if s.swizzle_cd_mode != 0 {
            cd_atom_n
        } else {
            cd.1
        },
        s.store_block_m,
        cd.4,
        cd.5,
        s.swizzle_cd_mode,
    )?;
    // BF16 pair: no SFs — dummy maps (the kernel never loads them).
    let dummy = DevBuffer::alloc_zeros(dev, 128)?;
    let tm_sf = tma::make_tma_sf(dev, &dummy, 32, 128, 32, 32, 1, 1, 0)?;

    crate::api::launch_gemm_with_maps(
        &ctx,
        &a_op,
        &b_op,
        &out,
        GemmType::Batched,
        None,
        batch,
        accumulate,
        None,
        None,
        128,
        (tm_a, tm_b, tm_sf, tm_sf, tm_cd),
    )
}

/// `einsum("bmk,bnk->mn", a, b, d, c=None)` — the MLA score projection.
///
/// * `a`: `[s, m, k]` BF16; `b`: `[s, n, k]` BF16; `d`: `[m, n]`.
/// * FP32 `d`: `c` MUST alias `d` (the kernel accumulates into it).
/// * BF16 `d`: an FP32 workspace is zeroed, accumulated, then RNE-cast
///   (upstream: `d.copy_(workspace)`).
pub fn einsum_bmk_bnk_mn(
    dev: &Device,
    stream: &DevStream,
    a: &Tensor3,
    b: &Tensor3,
    d: &mut Output,
    c: Option<&DevBuffer>,
) -> DgResult<()> {
    let [s, m, k] = a.dims;
    let [s2, n, k2] = b.dims;
    if s != s2 || k != k2 {
        return Err(DgError::InvalidArg("bmk/bnk shape mismatch".into()));
    }
    if d.rows != m || d.cols != n {
        return Err(DgError::InvalidArg("output shape mismatch".into()));
    }
    let a_op = Operand {
        dtype: Dtype::Bf16,
        major: Major::K,
        rows: s * m,
        k,
        outer_stride: k,
        sf: None,
        data: a.data.clone_view(),
    };
    let b_op = Operand {
        dtype: Dtype::Bf16,
        major: Major::K,
        rows: s * n,
        k,
        outer_stride: k,
        sf: None,
        data: b.data.clone_view(),
    };
    match d.dtype {
        Dtype::F32 => {
            let c = c.ok_or_else(|| {
                DgError::InvalidArg("FP32 D requires C == D (accumulated expression)".into())
            })?;
            if c.ptr != d.data.ptr {
                return Err(DgError::InvalidArg(
                    "FP32 D requires C == D (accumulated expression)".into(),
                ));
            }
            crate::api_bmk::bf16_bmk_bnk_mn_sm100(dev, stream, &a_op, &b_op, d, s)
        }
        Dtype::Bf16 => {
            if c.is_some() {
                return Err(DgError::InvalidArg(
                    "BF16 D takes no C (upstream contract)".into(),
                ));
            }
            let ws = DevBuffer::alloc_zeros(dev, (m as usize) * (n as usize) * 4)?;
            let mut d32 = Output {
                dtype: Dtype::F32,
                rows: m,
                cols: n,
                stride: n,
                data: ws.clone_view(),
            };
            crate::api_bmk::bf16_bmk_bnk_mn_sm100(dev, stream, &a_op, &b_op, &mut d32, s)?;
            cast_to_bf16(dev, stream, &ws, &d.data, m * n)
        }
        _ => Err(DgError::InvalidArg("D must be BF16 or FP32".into())),
    }
}

/// `einsum("bhr,hdr->bhd", a, b, d, c)` — attention out-projection.
/// `a: [b, h, r]`, `b: [h, d, r]`, `d/c: [b, h, d]`, all BF16, r inner.
pub fn einsum_bhr_hdr_bhd(
    dev: &Device,
    stream: &DevStream,
    a: &Tensor3,
    b: &Tensor3,
    d: &Tensor3,
    c: Option<&Tensor3>,
) -> DgResult<()> {
    let [bt, h, r] = a.dims;
    let [h2, dt, r2] = b.dims;
    let [bt2, h3, dt2] = d.dims;
    if h != h2 || h != h3 || bt != bt2 || r != r2 || dt != dt2 {
        return Err(DgError::InvalidArg("bhr,hdr->bhd shape mismatch".into()));
    }
    a.check(Dtype::Bf16, "A")?;
    b.check(Dtype::Bf16, "B")?;
    d.check(Dtype::Bf16, "D")?;
    let a_st = a.strides();
    let b_st = b.strides();
    let d_st = d.strides();
    let cd = match c {
        Some(c) => {
            if c.dims != [bt, h, dt] {
                return Err(DgError::InvalidArg("C/D shape mismatch".into()));
            }
            c
        }
        None => d,
    };
    // (batch=h, m=b, n=d, k=r): A dims (r, b, h) strides (h*r between b, r
    // between h); B dims (r, d, h) strides (r between d, d*r between h);
    // CD dims (d, b, h) strides (h*d between b, d between h).
    batched_gemm_strided(
        dev,
        stream,
        (&a.data, r, bt, h, a_st[0], a_st[1], false),
        (&b.data, r, dt, h, b_st[1], b_st[0], false),
        (&cd.data, dt, bt, h, d_st[0], d_st[1]),
        bt,
        dt,
        r,
        h,
        c.is_some(),
    )
}

/// `einsum("bhd,hdr->bhr", a, b, d, c)` — attention value projection.
/// `a: [b, h, d]`, `b: [h, d, r]`, `d/c: [b, h, r]`, all BF16.
pub fn einsum_bhd_hdr_bhr(
    dev: &Device,
    stream: &DevStream,
    a: &Tensor3,
    b: &Tensor3,
    d: &Tensor3,
    c: Option<&Tensor3>,
) -> DgResult<()> {
    let [bt, h, dt] = a.dims;
    let [h2, dt2, r] = b.dims;
    let [bt2, h3, r2] = d.dims;
    if h != h2 || h != h3 || bt != bt2 || dt != dt2 || r != r2 {
        return Err(DgError::InvalidArg("bhd,hdr->bhr shape mismatch".into()));
    }
    a.check(Dtype::Bf16, "A")?;
    b.check(Dtype::Bf16, "B")?;
    d.check(Dtype::Bf16, "D")?;
    let a_st = a.strides();
    let b_st = b.strides();
    let d_st = d.strides();
    let cd = match c {
        Some(c) => {
            if c.dims != [bt, h, r] {
                return Err(DgError::InvalidArg("C/D shape mismatch".into()));
            }
            c
        }
        None => d,
    };
    // (batch=h, m=b, n=r, k=d): A dims (d, b, h) strides (h*d, d); B is
    // MN-major (r inner): dims (r, d, h) strides (r between d, d*r between
    // h); CD dims (r, b, h) strides (h*r, r).
    batched_gemm_strided(
        dev,
        stream,
        (&a.data, dt, bt, h, a_st[0], a_st[1], false),
        (&b.data, r, dt, h, b_st[1], b_st[0], true),
        (&cd.data, r, bt, h, d_st[0], d_st[1]),
        bt,
        r,
        dt,
        h,
        c.is_some(),
    )
}

/// `einsum("bhd,bhr->hdr", a, b, d, c)` — the output projection's
/// weight-side reduction over the batch. Upstream routes this to cuBLASLt
/// (`cublaslt_bhd_bhr_hdr`): the GEMM K dim is the (small) batch `b`, which
/// the DeepGEMM kernels' K-multiple constraints (BF16: 64) don't
/// accommodate. `d/c: [h, d, r]` FP32, C optional (accumulates).
pub fn einsum_bhd_bhr_hdr(
    dev: &Device,
    stream: &DevStream,
    a: &Tensor3,
    b: &Tensor3,
    d: &Tensor3,
    c: Option<&Tensor3>,
) -> DgResult<()> {
    let [bt, h, dt] = a.dims;
    let [bt2, h2, r] = b.dims;
    let [h3, dt2, r2] = d.dims;
    if bt != bt2 || h != h2 || h != h3 || dt != dt2 || r != r2 {
        return Err(DgError::InvalidArg("bhd,bhr->hdr shape mismatch".into()));
    }
    a.check(Dtype::Bf16, "A")?;
    b.check(Dtype::Bf16, "B")?;
    crate::cublaslt::bhd_bhr_hdr(dev, stream, a, b, d, c, bt, h, r, dt)
}

// ---------------------------------------------------------------------------
// FP8 einsum: permuted-view FP8 BMMs
// ---------------------------------------------------------------------------

/// An FP8 einsum operand: FP8 data (contiguous, equation-specific shape)
/// plus FP32 scales in the tensor's own memory order. Each equation entry
/// documents which of the two scale forms it takes.
pub struct Fp8EinsumOperand {
    pub data: DevBuffer,
    /// Per-token scales `[tokens, k/gran]`, row-major over the tensor's
    /// leading token dims (e.g. `[b*h, r/gran]` for a `[b, h, r]` operand).
    /// (eq1/eq2 A-side; eq3 uses this field for the per-channel scales —
    /// see [`fp8_einsum_bhd_bhr_hdr`].)
    pub scales: Vec<f32>,
    /// Optional per-128x128-block scales `[h][mn/128][k/gran]` (the legacy
    /// recipe; the weight-side operands of eq1/eq2). When set, `scales`
    /// must be empty.
    pub block_scales: Vec<f32>,
    pub gran: SfGran,
}

impl Fp8EinsumOperand {
    pub fn per_token(data: DevBuffer, scales: Vec<f32>, gran: SfGran) -> Self {
        Fp8EinsumOperand {
            data,
            scales,
            block_scales: Vec::new(),
            gran,
        }
    }

    pub fn per_block(data: DevBuffer, blocks: Vec<f32>, gran: SfGran) -> Self {
        Fp8EinsumOperand {
            data,
            scales: Vec::new(),
            block_scales: blocks,
            gran,
        }
    }
}

/// Broadcast per-128x128-block scales `[mn/128, sf_k]` into per-token
/// `[mn, sf_k]` (upstream: `sf.index_select(-2, arange(mn) // gran_mn)`).
fn broadcast_block_scales(blocks: &[f32], mn: u32, sf_k: u32) -> DgResult<Vec<f32>> {
    if mn % 128 != 0 || blocks.len() != ((mn / 128) * sf_k) as usize {
        return Err(DgError::InvalidArg(
            "block scales must hold (mn/128) * sf_k values, mn % 128 == 0".into(),
        ));
    }
    let mut out = vec![0f32; (mn * sf_k) as usize];
    for row in 0..mn {
        let b = (row / 128 * sf_k) as usize;
        out[(row * sf_k) as usize..((row + 1) * sf_k) as usize]
            .copy_from_slice(&blocks[b..b + sf_k as usize]);
    }
    Ok(out)
}

/// Reorder per-token scales from `[d0, d1, sf_k]` memory order into
/// `[d1, d0, sf_k]` (the leading-dim permutation `permute(1, 0, 2)` the
/// fp8 equations apply before the batched SF packing).
fn swap_leading_dims(src: &[f32], d0: u32, d1: u32, sf_k: u32) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for i in 0..d0 as usize {
        for j in 0..d1 as usize {
            let s = (i * d1 as usize + j) * sf_k as usize;
            let t = (j * d0 as usize + i) * sf_k as usize;
            out[t..t + sf_k as usize].copy_from_slice(&src[s..s + sf_k as usize]);
        }
    }
    out
}

/// The FP8 BMM engine: strided 3D maps + batch-packed SF arenas. `cd_dtype`
/// selects a BF16 or FP32 output.
#[allow(clippy::too_many_arguments)]
fn fp8_bmm_strided(
    dev: &Device,
    stream: &DevStream,
    // (buf, inner, outer, batch, stride_outer, stride_batch, mn_major)
    a_map: (&DevBuffer, u32, u32, u32, u64, u64, bool),
    b_map: (&DevBuffer, u32, u32, u32, u64, u64, bool),
    cd: (&DevBuffer, u32, u32, u32, u64, u64),
    sfa: &DevBuffer,
    sfb: &DevBuffer,
    sfa_mn: u32,
    sfb_mn: u32,
    sfa_rows_per_batch: u32,
    sfb_rows_per_batch: u32,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    accumulate: bool,
    cd_dtype: Dtype,
) -> DgResult<()> {
    let ctx = LaunchCtx {
        dev,
        stream: stream.raw(),
    };
    let a_op = Operand {
        dtype: Dtype::Fp8,
        major: if a_map.6 { Major::Mn } else { Major::K },
        rows: m,
        k,
        outer_stride: a_map.4 as u32,
        sf: None,
        data: a_map.0.clone_view(),
    };
    let b_op = Operand {
        dtype: Dtype::Fp8,
        major: if b_map.6 { Major::Mn } else { Major::K },
        rows: n,
        k,
        outer_stride: b_map.4 as u32,
        sf: None,
        data: b_map.0.clone_view(),
    };
    let out = Output {
        dtype: cd_dtype,
        rows: m,
        cols: n,
        stride: n,
        data: cd.0.clone_view(),
    };
    let desc = heuristics::GemmDesc {
        gemm_type: GemmType::Batched,
        m,
        n,
        k,
        num_groups: batch,
        a_dtype: Dtype::Fp8,
        b_dtype: Dtype::Fp8,
        cd_dtype,
        major_a: a_op.major,
        major_b: b_op.major,
        num_sms: dev.num_sms,
        smem_capacity: dev.smem_capacity,
        expected_m: m,
        expected_num_groups: 1,
        expected_k: 0,
        with_accumulation: accumulate,
    };
    let cfg = heuristics::get_best_config(&desc)?;
    let s = &cfg.storage;
    let (sf_block_m, sf_block_n) =
        heuristics::get_sf_block_sizes(cfg.layout.block_m, cfg.layout.block_n, true);
    let sf_block_k = cfg.layout.block_k / 128;

    let tm_a = tma::make_tma_3d_strided(
        dev,
        Dtype::Fp8,
        a_map.0,
        a_map.1,
        a_map.2,
        a_map.3,
        box_inner(s.swizzle_a_mode, 1, a_map.1),
        s.load_block_m,
        a_map.4,
        a_map.5,
        s.swizzle_a_mode,
    )?;
    let tm_b = tma::make_tma_3d_strided(
        dev,
        Dtype::Fp8,
        b_map.0,
        b_map.1,
        b_map.2,
        b_map.3,
        box_inner(s.swizzle_b_mode, 1, b_map.1),
        s.load_block_n,
        b_map.4,
        b_map.5,
        s.swizzle_b_mode,
    )?;
    let cd_elem = cd_dtype.elem_size() as u32;
    let cd_atom_n = s.swizzle_cd_mode / cd_elem;
    let tm_cd = tma::make_tma_3d_strided(
        dev,
        cd_dtype,
        cd.0,
        cd.1,
        cd.2,
        cd.3,
        if s.swizzle_cd_mode != 0 {
            cd_atom_n
        } else {
            cd.1
        },
        s.store_block_m,
        cd.4,
        cd.5,
        s.swizzle_cd_mode,
    )?;
    let sfa_aligned = heuristics::tma_aligned_size(sfa_mn, 4);
    let sfb_aligned = heuristics::tma_aligned_size(sfb_mn, 4);
    let tm_sfa = tma::make_tma_sf_packed_rows(
        dev,
        sfa,
        sfa_aligned,
        sfa_rows_per_batch * batch,
        sf_block_m,
        sf_block_k,
    )?;
    let tm_sfb = tma::make_tma_sf_packed_rows(
        dev,
        sfb,
        sfb_aligned,
        sfb_rows_per_batch * batch,
        sf_block_n,
        sf_block_k,
    )?;

    crate::api::launch_gemm_with_maps(
        &ctx,
        &a_op,
        &b_op,
        &out,
        GemmType::Batched,
        None,
        batch,
        accumulate,
        None,
        None,
        128,
        (tm_a, tm_b, tm_sfa, tm_sfb, tm_cd),
    )
}

/// `fp8_einsum("bhr,hdr->bhd", a, b, d, c)` — FP8 attention out-proj.
///
/// * `a`: data `[b, h, r]` E4M3, **per-token** scales `[b*h, r/gran]`
///   (memory order (b, h)).
/// * `b`: data `[h, d, r]` E4M3, **per-block** scales
///   `[h][d/128][r/gran]` (legacy 128x128 recipe; upstream's
///   `per_block_cast_to_fp8(y[i])` per h slice).
/// * `d/c`: `[b, h, d]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn fp8_einsum_bhr_hdr_bhd(
    dev: &Device,
    stream: &DevStream,
    a: &Fp8EinsumOperand,
    b: &Fp8EinsumOperand,
    d: &Tensor3,
    c: Option<&Tensor3>,
) -> DgResult<()> {
    let [bt, h, dt] = d.dims;
    if a.scales.is_empty() || b.block_scales.is_empty() {
        return Err(DgError::InvalidArg(
            "eq1 takes per-token A scales and per-block B scales".into(),
        ));
    }
    let r = (a.scales.len() / (bt * h) as usize) as u32 * a.gran.k();
    if r == 0 || r % a.gran.k() != 0 || r % b.gran.k() != 0 {
        return Err(DgError::InvalidArg("A scales: [b*h, r/gran]".into()));
    }
    let sf_k = r / a.gran.k();
    let sfk_b = r / b.gran.k();
    // A: per-token [b, h, sf_k] -> batch-major [h, b, sf_k].
    let sfa_host = swap_leading_dims(&a.scales, bt, h, sf_k);
    // B: per-block [h][d/128][sfk_b] -> broadcast per-(h,d)-token rows.
    let mut sfb_host = Vec::with_capacity((h * dt * sfk_b) as usize);
    for hb in 0..h {
        let base = (hb * (dt / 128) * sfk_b) as usize;
        sfb_host.extend(broadcast_block_scales(
            &b.block_scales[base..base + (dt / 128 * sfk_b) as usize],
            dt,
            sfk_b,
        )?);
    }
    let sfa = transform_sf_batched(dev, stream, &sfa_host, h, bt, a.gran)?;
    let sfb = transform_sf_batched(dev, stream, &sfb_host, h, dt, b.gran)?;
    let d_st = d.strides();
    let cd = match c {
        Some(c) => {
            if c.dims != d.dims {
                return Err(DgError::InvalidArg("C/D shape mismatch".into()));
            }
            c
        }
        None => d,
    };
    // (batch=h, m=b, n=d, k=r): A dims (r, b, h) strides (h*r, r); B dims
    // (r, d, h) strides (r, d*r); CD dims (d, b, h) strides (h*d, d).
    fp8_bmm_strided(
        dev,
        stream,
        (&a.data, r, bt, h, (h * r) as u64, r as u64, false),
        (&b.data, r, dt, h, r as u64, (dt * r) as u64, false),
        (&cd.data, dt, bt, h, d_st[0], d_st[1]),
        &sfa.buf,
        &sfb.buf,
        bt,
        dt,
        sfa.rows / h,
        sfb.rows / h,
        bt,
        dt,
        r,
        h,
        c.is_some(),
        Dtype::Bf16,
    )
}

/// `fp8_einsum("bhd,hdr->bhr", a, b, d, c)` — FP8 value projection.
///
/// * `a`: data `[b, h, d]`, per-token scales `[b*h, d/gran]`.
/// * `b`: data `[h, d, r]`, per-block scales `[h][d/128][r/gran]` — blocks
///   of the `[d, r]` matrix; they serve the transposed `[r, d]` GEMM view:
///   per-token along the GEMM's n (r, granularity `b.gran`), per-128 along
///   its k (d) — exactly upstream's `sfb.permute(0, 2, 1)`.
/// * `d/c`: `[b, h, r]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn fp8_einsum_bhd_hdr_bhr(
    dev: &Device,
    stream: &DevStream,
    a: &Fp8EinsumOperand,
    b: &Fp8EinsumOperand,
    d: &Tensor3,
    c: Option<&Tensor3>,
) -> DgResult<()> {
    let [bt, h, r] = d.dims;
    if a.scales.is_empty() || b.block_scales.is_empty() {
        return Err(DgError::InvalidArg(
            "eq2 takes per-token A scales and per-block B scales".into(),
        ));
    }
    let dt = (a.scales.len() / (bt * h) as usize) as u32 * a.gran.k();
    if dt == 0 || dt % 128 != 0 || r % b.gran.k() != 0 {
        return Err(DgError::InvalidArg(
            "eq2 needs d % 128 (block-K) and r % gran".into(),
        ));
    }
    let sf_k = dt / a.gran.k();
    // A: per-token [b, h, sf_k] -> batch-major [h, b, sf_k].
    let sfa_host = swap_leading_dims(&a.scales, bt, h, sf_k);
    // B: [h][r rows of n][d/128 k-words] (batch-major already).
    let k_blocks = dt / 128;
    let r_groups = r / b.gran.k();
    if b.block_scales.len() != (h * k_blocks * r_groups) as usize {
        return Err(DgError::InvalidArg(
            "B must carry per-block scales [h][d/128][r/gran]".into(),
        ));
    }
    let mut sfb_host = vec![0f32; (h * r * k_blocks) as usize];
    for hb in 0..h as usize {
        for n_row in 0..r as usize {
            let r_grp = n_row / b.gran.k() as usize;
            for kb in 0..k_blocks as usize {
                sfb_host[(hb * r as usize + n_row) * k_blocks as usize + kb] = b.block_scales
                    [hb * k_blocks as usize * r_groups as usize + kb * r_groups as usize + r_grp];
            }
        }
    }
    let sfa = transform_sf_batched(dev, stream, &sfa_host, h, bt, a.gran)?;
    // B's GEMM-k granularity is 128 (block scales along d).
    let sfb = transform_sf_batched(dev, stream, &sfb_host, h, r, SfGran::G128)?;
    let d_st = d.strides();
    let cd = match c {
        Some(c) => {
            if c.dims != d.dims {
                return Err(DgError::InvalidArg("C/D shape mismatch".into()));
            }
            c
        }
        None => d,
    };
    // (batch=h, m=b, n=r, k=d): A dims (d, b, h) strides (h*d, d); B
    // MN-major dims (r, d, h) strides (r, d*r); CD dims (r, b, h)
    // strides (h*r, r).
    fp8_bmm_strided(
        dev,
        stream,
        (&a.data, dt, bt, h, (h * dt) as u64, dt as u64, false),
        (&b.data, r, dt, h, r as u64, (dt * r) as u64, true),
        (&cd.data, r, bt, h, d_st[0], d_st[1]),
        &sfa.buf,
        &sfb.buf,
        bt,
        r,
        sfa.rows / h,
        sfb.rows / h,
        bt,
        r,
        dt,
        h,
        c.is_some(),
        Dtype::Bf16,
    )
}

/// `fp8_einsum("bhd,bhr->hdr", a, b, d, c)` — FP8 o-proj weight-side reduce.
///
/// * `a`: data `[b, h, d]`, **per-channel** scales `[ceil(b/gran), h*d]`
///   (one scale per (h, d) column over the batch; upstream
///   `per_channel_cast_to_fp8(x.view(b, -1))` viewed `[b/gran, h, d]`).
/// * `b`: data `[b, h, r]`, per-channel scales `[ceil(b/gran), h*r]`.
/// * `d/c`: `[h, d, r]` FP32, C optional (accumulates).
///
/// The GEMM K dim is the batch `b`, which must be a multiple of 128 (FP8
/// K-multiple constraint; upstream tests use b in {4096, 8192}).
#[allow(clippy::too_many_arguments)]
pub fn fp8_einsum_bhd_bhr_hdr(
    dev: &Device,
    stream: &DevStream,
    a: &Fp8EinsumOperand,
    b: &Fp8EinsumOperand,
    d: &Tensor3,
    c: Option<&Tensor3>,
) -> DgResult<()> {
    let [h, dt, r] = d.dims;
    if a.scales.is_empty() || b.scales.is_empty() {
        return Err(DgError::InvalidArg(
            "eq3 takes per-channel scales in the `scales` field".into(),
        ));
    }
    let bt = (a.scales.len() / ((h * dt) as usize).max(1)) as u32 * a.gran.k();
    if bt == 0 || bt % 128 != 0 {
        return Err(DgError::InvalidArg(format!(
            "eq3 needs the batch b={bt} to be a multiple of 128 (K constraint)"
        )));
    }
    // Per-channel [b/gran, h*mn] -> GEMM-side per-token [batch=h, mn, k/gran].
    let build = |src: &[f32], mn: u32, gran: SfGran| -> Vec<f32> {
        let words = bt / gran.k();
        let mut out = vec![0f32; (h * mn * words) as usize];
        for hb in 0..h as usize {
            for row in 0..mn as usize {
                let col = hb * mn as usize + row;
                for w in 0..words as usize {
                    out[(hb * mn as usize + row) * words as usize + w] =
                        src[w * (h * mn) as usize + col];
                }
            }
        }
        out
    };
    let sfa_host = build(&a.scales, dt, a.gran);
    let sfb_host = build(&b.scales, r, b.gran);
    let sfa = transform_sf_batched(dev, stream, &sfa_host, h, dt, a.gran)?;
    let sfb = transform_sf_batched(dev, stream, &sfb_host, h, r, b.gran)?;
    let d_st = d.strides();
    let cd = c.unwrap_or(d);
    // (batch=h, m=d, n=r, k=b): A' MN-major dims (d, b, h) strides (h*d
    // between b, d between h); B' MN-major dims (r, b, h) strides (h*r, r);
    // CD dims (r, d, h) strides (r between d, d*r between h). FP32 output.
    fp8_bmm_strided(
        dev,
        stream,
        (&a.data, dt, bt, h, (h * dt) as u64, dt as u64, true),
        (&b.data, r, bt, h, (h * r) as u64, r as u64, true),
        (&cd.data, r, dt, h, d_st[1], d_st[0]),
        &sfa.buf,
        &sfb.buf,
        dt,
        r,
        sfa.rows / h,
        sfb.rows / h,
        dt,
        r,
        bt,
        h,
        c.is_some(),
        Dtype::F32,
    )
}
