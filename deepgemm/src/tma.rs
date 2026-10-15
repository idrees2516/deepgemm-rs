//! TMA descriptor construction (port of upstream `runtime_utils.hpp`).
//!
//! # What a TMA descriptor is
//!
//! `cuTensorMapEncodeTiled` bakes a *copy program* into a 128-byte struct
//! that the kernel passes by value (`__grid_constant__ dg::TmaMap`). One
//! `cp.async.bulk.tensor.*` instruction then moves a whole tile, enforcing:
//!
//! * **Box shape** — `[smem_inner, smem_outer]` elements per copy; we split
//!   the inner extent into *swizzle atoms* so the SMEM tile matches the MMA
//!   descriptor's layout expectations (see `get_atom_base` in the kernels).
//! * **Global strides** — the outer (non-contiguous) dimension stride in
//!   *bytes*, which is how grouped/batched tensors stack experts without
//!   copying: the "3D" maps simply add a batch index coordinate.
//! * **Swizzle atom** — 16B/32B/64B/128B; the SMEM tile is stored so that
//!   16B accesses never bank-conflict (row XOR with atom/16). FP4 MXF4
//!   requires 128B; unpacked FP4 uses `16U4_ALIGN16B` (gmem inner multiple
//!   of 128 nibbles = 64B) vs packed `16U4_ALIGN8B` (SMEM inner = swizzle*2).
//! * **Dtype code** — UINT8 for FP8 (raw bytes), 16U4 pair for FP4,
//!   BF16/FP32 for the f16 path. The *data* is always moved as raw bits;
//!   the dtype only shapes the box arithmetic.
//!
//! # Map flavors used by the GEMM
//!
//! | map   | tensor        | dims (inner→outer)          | notes                |
//! |-------|---------------|-----------------------------|----------------------|
//! | ab    | A or B        | K (or MN) x rows            | 2D; 3D for batched   |
//! | sf    | scale factors | packed-K rows x TMA-align MN| int32 words, no swz  |
//! | cd    | output D      | swizzle atom x store rows   | TMA store path       |
//! | mqa_* | Q/KV/SF/W     | special headDim packings    | see mqa_logits       |
//!
//! All "SF" maps share one invariant with [`crate::golden`]: the packed word
//! at `(k/4, m)` carries bytes for k, k+1, k+2, k+3 in little-endian order.

use crate::device::{DevBuffer, Device};
use crate::error::{DgError, DgResult};
use crate::sys;
use crate::types::{Dtype, Major};

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn tma_aligned_size(size: u32, elem_size: u32) -> u32 {
    ceil_div(size, (16 / elem_size).max(1))
}

fn dtype_of(dtype: Dtype, fp4_unpacked_smem: bool) -> DgResult<sys::TmDtype> {
    match dtype {
        Dtype::Fp8 => Ok(sys::tm_dtype_uint8()),
        Dtype::Fp4 => Ok(if fp4_unpacked_smem {
            sys::tm_dtype_16u4_align16b()
        } else {
            sys::tm_dtype_16u4_align8b()
        }),
        Dtype::Bf16 => Ok(sys::tm_dtype_bfloat16()),
        Dtype::F32 => Ok(sys::tm_dtype_float32()),
    }
}

/// Encode a 2D tiled tensor map.
#[allow(clippy::too_many_arguments)]
/// Wire dtype for the paged 3D maps (same encoding as make_tma_2d uses for
/// mqa Q/KV: FP8/FP4-packed/BF16 as INT8/16U4_ALIGN8B/INT16).
fn dtype_of2(dtype: Dtype) -> DgResult<sys::TmDtype> {
    dtype_of(dtype, false)
}

// Upstream-mirroring signature: one arg per DeepGEMM parameter.
#[allow(clippy::too_many_arguments)]
fn make_tma_2d(
    dev: &Device,
    dtype: Dtype,
    fp4_unpacked_smem: bool,
    addr: sys::DevicePtr,
    gmem_inner: u32,
    gmem_outer: u32,
    smem_inner: u32,
    smem_outer: u32,
    outer_stride_elems: u32,
    swizzle_mode: u32,
) -> DgResult<sys::TensorMap> {
    let elem_size = dtype.elem_size() as u32;
    let wire = if dtype == Dtype::Fp4 { 1 } else { elem_size };
    let outer_stride_bytes = outer_stride_elems as u64 * wire as u64;
    if outer_stride_bytes % 16 != 0 {
        return Err(DgError::InvalidArg(format!(
            "TMA outer stride {outer_stride_bytes}B not 16B-aligned"
        )));
    }
    let mut smem_inner = smem_inner;
    if swizzle_mode != 0 {
        smem_inner = swizzle_mode / wire;
    }
    if dtype == Dtype::Fp4 {
        if fp4_unpacked_smem && gmem_inner % 128 != 0 {
            return Err(DgError::InvalidArg(
                "FP4 (unpacked smem) global inner dim must be a multiple of 128 elements".into(),
            ));
        }
        if !fp4_unpacked_smem && swizzle_mode != 0 {
            smem_inner = swizzle_mode * 2;
        }
    }

    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype_of(dtype, fp4_unpacked_smem)?,
        2,
        addr as *mut _,
        &[gmem_inner as u64, gmem_outer as u64],
        &[outer_stride_bytes],
        &[smem_inner, smem_outer],
        &[1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// A/B operand map. `rows` = M (for A) or N (for B) including group stacking;
/// `outer_stride` = stride of the non-contiguous dim in elements.
#[allow(clippy::too_many_arguments)]
pub fn make_tma_ab(
    dev: &Device,
    dtype: Dtype,
    major: Major,
    buf: &DevBuffer,
    rows: u32,
    k: u32,
    load_block_mn: u32,
    block_k: u32,
    outer_stride: u32,
    num_groups: u32,
    swizzle_mode: u32,
    fp4_unpacked_smem: bool,
) -> DgResult<sys::TensorMap> {
    // Wire dims: FP4 logical elements pack 2/byte in GMEM.
    let (gmem_inner, gmem_outer, smem_inner, smem_outer) = match major {
        Major::K => (k, rows * num_groups, block_k, load_block_mn),
        Major::Mn => (rows, k * num_groups.max(1), load_block_mn, block_k),
    };
    // Note: for MN-major, groups stack along K in the outer dim (stride covers rows).
    let outer_stride = match major {
        Major::K => outer_stride,
        Major::Mn => outer_stride,
    };
    make_tma_2d(
        dev,
        dtype,
        fp4_unpacked_smem,
        buf.ptr,
        gmem_inner,
        gmem_outer,
        smem_inner,
        smem_outer,
        outer_stride,
        swizzle_mode,
    )
}

/// Batched A/B map: 3D (inner, outer, batch).
#[allow(clippy::too_many_arguments)]
pub fn make_tma_ab_3d(
    dev: &Device,
    dtype: Dtype,
    major: Major,
    buf: &DevBuffer,
    rows: u32,
    k: u32,
    load_block_mn: u32,
    block_k: u32,
    outer_stride: u32,
    batch: u32,
    swizzle_mode: u32,
    fp4_unpacked_smem: bool,
) -> DgResult<sys::TensorMap> {
    let elem_wire = if dtype == Dtype::Fp4 {
        1
    } else {
        dtype.elem_size() as u32
    };
    let (inner, outer, _smem_inner, smem_outer) = match major {
        Major::K => (k, rows, block_k, load_block_mn),
        Major::Mn => (rows, k, load_block_mn, block_k),
    };
    let s0 = if swizzle_mode != 0 {
        swizzle_mode / elem_wire
    } else {
        inner
    };
    let mut s0 = s0;
    if dtype == Dtype::Fp4 && !fp4_unpacked_smem && swizzle_mode != 0 {
        s0 = swizzle_mode * 2;
    }
    let stride0 = outer_stride as u64 * elem_wire as u64;
    let stride1 = match major {
        Major::K => rows as u64 * stride0,
        Major::Mn => k as u64 * stride0,
    };
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype_of(dtype, fp4_unpacked_smem)?,
        3,
        buf.ptr as *mut _,
        &[inner as u64, outer as u64, batch as u64],
        &[stride0, stride1],
        &[s0, smem_outer, 1],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// C/D map: [n, m*groups] inner=N (contiguous), swizzled store boxes.
/// `store_block_n` must be the TMA store atom width (swizzle bytes / element).
#[allow(clippy::too_many_arguments)]
pub fn make_tma_cd(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    rows: u32,
    cols: u32,
    store_block_m: u32,
    store_block_n: u32,
    row_stride: u32,
    num_groups: u32,
    swizzle_mode: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(
        dev,
        dtype,
        false,
        buf.ptr,
        cols,
        rows * num_groups,
        store_block_n,
        store_block_m,
        row_stride,
        swizzle_mode,
    )
}

/// Batched C/D map: 3D [n, m, batch].
#[allow(clippy::too_many_arguments)]
pub fn make_tma_cd_3d(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    rows: u32,
    cols: u32,
    store_block_m: u32,
    store_block_n: u32,
    row_stride: u32,
    batch: u32,
    swizzle_mode: u32,
) -> DgResult<sys::TensorMap> {
    let elem = dtype.elem_size() as u64;
    let stride0 = row_stride as u64 * elem;
    // Batch stride: `[batch, m, n]` slices are `m` rows of `row_stride`
    // apart — `rows * row_stride` elements (upstream passes `d.stride(0)`).
    // (Passing `cols * elem` here interleaved batches — every batched CD
    // store would land at the wrong slice for m > 1.)
    let stride1 = rows as u64 * row_stride as u64 * elem;
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype_of(dtype, false)?,
        3,
        buf.ptr as *mut _,
        &[cols as u64, rows as u64, batch as u64],
        &[stride0, stride1],
        &[store_block_n, store_block_m, 1],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// Scale-factor map: int32 [mn, packed_k * groups], MN-contiguous, no swizzle.
#[allow(clippy::too_many_arguments)]
pub fn make_tma_sf(
    dev: &Device,
    buf: &DevBuffer,
    rows: u32,       // logical MN (or per-group MN)
    k: u32,          // logical K
    gran_k: u32,     // SF granularity (32 or 128)
    block_mn: u32,   // SMEM box along MN (SF_BLOCK_M/N)
    sf_block_k: u32, // SMEM box along packed K rows
    num_groups: u32,
    packed_row_stride: u32, // 0 => compact TMA-aligned layout
) -> DgResult<sys::TensorMap> {
    let tma_aligned = tma_aligned_size(rows, 4);
    let packed_rows = ceil_div(k, gran_k * 4);
    let outer_stride = if packed_row_stride == 0 {
        tma_aligned
    } else {
        packed_row_stride
    };
    if outer_stride < tma_aligned {
        return Err(DgError::InvalidArg(
            "SF row stride smaller than TMA-aligned MN".into(),
        ));
    }
    make_tma_2d(
        dev,
        Dtype::F32,
        false,
        buf.ptr,
        tma_aligned,
        packed_rows * num_groups,
        block_mn,
        sf_block_k.max(1),
        outer_stride,
        0,
    )
}

/// K-grouped SF arena map: the packed row count comes from the per-group
/// padded counts (each group's SF rows round up to whole packed words), so it
/// can exceed `ceil_div(sum_k, gran_k * 4)`. Single shared arena, one MN dim.
#[allow(clippy::too_many_arguments)]
pub fn make_tma_sf_packed_rows(
    dev: &Device,
    buf: &DevBuffer,
    aligned_mn: u32,
    packed_rows: u32,
    block_mn: u32,
    sf_block_k: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(
        dev,
        Dtype::F32,
        false,
        buf.ptr,
        aligned_mn,
        packed_rows,
        block_mn,
        sf_block_k.max(1),
        aligned_mn,
        0,
    )
}

/// Weights map for MQA logits: bf16 [heads, tokens] (inner = heads).
pub fn make_tma_mqa_weights(
    dev: &Device,
    buf: &DevBuffer,
    num_heads: u32,
    num_tokens: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(
        dev,
        Dtype::Bf16,
        false,
        buf.ptr,
        num_heads,
        num_tokens,
        num_heads,
        1,
        num_heads,
        0,
    )
}

/// Q / KV data map for MQA logits: [head_dim, rows] (inner = K).
/// The SMEM tile is swizzled to match the UMMA descriptor
/// (`kQKSwizzleMode = head_dim / pack` bytes).
pub fn make_tma_mqa_qk(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    rows: u32,
    head_dim: u32,
    load_rows: u32,
    is_packed_fp4: bool,
) -> DgResult<sys::TensorMap> {
    let pack = if is_packed_fp4 { 2 } else { 1 };
    let swizzle = (head_dim / pack).min(128);
    make_tma_2d(
        dev, dtype, false, buf.ptr, head_dim, rows, head_dim, load_rows, head_dim, swizzle,
    )
}

/// Q / KV SF map for MQA logits: int32 [rows, 1] packed.
pub fn make_tma_mqa_sf(
    dev: &Device,
    buf: &DevBuffer,
    rows: u32,
    load_rows: u32,
) -> DgResult<sys::TensorMap> {
    let tma_aligned = tma_aligned_size(rows, 4);
    make_tma_2d(
        dev,
        Dtype::F32,
        false,
        buf.ptr,
        tma_aligned,
        1,
        load_rows,
        1,
        tma_aligned,
        0,
    )
}

/// SM90 1D1D scale-factor map: FP32 scales, one row per 128-K block,
/// MN-contiguous (col-major TMA-aligned), no swizzle, box [block_mn, 1].
/// The GMEM layout is `scales[kb][mn]` with row stride `tma_aligned(mn, 4)`
/// — produced by `api::transpose_sf_fp32` (port of the upstream Python
/// `get_col_major_tma_aligned_tensor` step).
#[allow(clippy::too_many_arguments)]
pub fn make_tma_sf_fp32(
    dev: &Device,
    buf: &DevBuffer,
    rows: u32,     // logical MN
    k: u32,        // logical K
    block_mn: u32, // SMEM box along MN (BLOCK_M or BLOCK_N)
    num_groups: u32,
) -> DgResult<sys::TensorMap> {
    let tma_aligned = tma_aligned_size(rows, 4);
    let sf_rows = ceil_div(k, 128);
    make_tma_2d(
        dev,
        Dtype::F32,
        false,
        buf.ptr,
        tma_aligned,
        sf_rows * num_groups,
        block_mn,
        1,
        tma_aligned,
        0,
    )
}

/// Paged MQA KV map: 3D [head_dim (inner, same units as the 2D QK map),
/// PAGE_KV (tokens/page), num_pages], box [head_dim, PAGE_KV, 1] — the gather
/// loads exactly one page per call (tma_load_3d with page row coord).
#[allow(clippy::too_many_arguments)]
pub fn make_tma_mqa_qk_paged(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    head_dim: u32,
    page_kv: u32,
    num_pages: u32,
    is_packed_fp4: bool,
) -> DgResult<sys::TensorMap> {
    let _ = is_packed_fp4;
    let pack = if dtype == Dtype::Fp4 { 2 } else { 1 };
    let swizzle = (head_dim / pack).min(128);
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype_of2(dtype)?,
        3,
        buf.ptr as *mut _,
        &[(head_dim / pack) as u64, page_kv as u64, num_pages as u64],
        &[
            (head_dim / pack) as u64 * dtype.elem_size() as u64,
            (head_dim / pack) as u64 * dtype.elem_size() as u64 * page_kv as u64,
        ],
        &[(head_dim / pack), page_kv, 1],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// Paged MQA SF map: 2D [PAGE_KV slots (inner), num_pages], box [PAGE_KV, 1].
/// One packed SF word per KV token (same granularity as the contiguous map).
pub fn make_tma_mqa_sf_paged(
    dev: &Device,
    buf: &DevBuffer,
    page_kv: u32,
    num_pages: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(
        dev,
        Dtype::F32,
        false,
        buf.ptr,
        page_kv,
        num_pages,
        page_kv,
        1,
        page_kv,
        0,
    )
}

// ===========================================================================
// Megakernel tensor maps (mega_moe / mega_gate / mega_mhc)
// ===========================================================================
use crate::mega_layout as mlayout;

/// Routed expert weights [E, N, K] (K-major fp8/fp4) -> 4D map
/// (K, N/domain, domains=2, experts); box (block_k, load_block_n, 1, 1).
/// Matches upstream `make_tma_weights_3d_desc` with num_domains = 2.
#[allow(clippy::too_many_arguments)]
pub fn make_tma_moe_weights_3d(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    num_experts: u32,
    n: u32,
    k: u32,
    block_k: u32,
    load_block_n: u32,
    swizzle_mode: u32,
    fp4_unpacked: bool,
) -> DgResult<sys::TensorMap> {
    const DOMAINS: u32 = mlayout::NUM_DEVICE_LOCALITY_DOMAINS;
    let elem = if fp4_unpacked {
        1
    } else {
        dtype.elem_size() as u32
    };
    let n_per_domain = n / DOMAINS;
    let gmem0 = k; // logical elements (fp4 gmem rows are packed)
    let gmem1 = n_per_domain;
    let gmem2 = DOMAINS;
    let gmem3 = num_experts;
    let stride0 = k as u64 * elem as u64; // dim1 stride: one N row
    let stride1 = n_per_domain as u64 * k as u64 * elem as u64; // dim2: domain slice
    let stride2 = n as u64 * k as u64 * elem as u64; // dim3: expert batch
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype_of(dtype, fp4_unpacked)?,
        4,
        buf.ptr as *mut _,
        &[gmem0 as u64, gmem1 as u64, gmem2 as u64, gmem3 as u64],
        &[stride0, stride1, stride2],
        &[block_k, load_block_n, 1, 1],
        &[1, 1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// Shared-expert weights [N, K] -> 4D map (K, N/domain, domains=2, 1).
#[allow(clippy::too_many_arguments)]
pub fn make_tma_moe_weights_2d(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    n: u32,
    k: u32,
    block_k: u32,
    load_block_n: u32,
    swizzle_mode: u32,
    fp4_unpacked: bool,
) -> DgResult<sys::TensorMap> {
    make_tma_moe_weights_3d(
        dev,
        dtype,
        buf,
        1,
        n,
        k,
        block_k,
        load_block_n,
        swizzle_mode,
        fp4_unpacked,
    )
}

/// mHC hidden 2D map: x/y [T, H] bf16, box (BLOCK_K, BLOCK_M), 128B swizzle.
pub fn make_tma_mhc_hidden(
    dev: &Device,
    buf: &DevBuffer,
    hidden: u32,
    num_tokens: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(
        dev,
        Dtype::Bf16,
        false,
        buf.ptr,
        hidden,
        num_tokens,
        mlayout::MHC_BLOCK_K,
        mlayout::MHC_BLOCK_M,
        hidden,
        mlayout::MHC_SWIZZLE,
    )
}

/// mHC residual 3D map: [T, R, H] -> dims (H, T, R); box (BLOCK_K, BLOCK_M, R).
pub fn make_tma_mhc_residual(
    dev: &Device,
    buf: &DevBuffer,
    hidden: u32,
    num_tokens: u32,
) -> DgResult<sys::TensorMap> {
    dev.bind()?;
    let elem = 2u64; // bf16
    sys::tensor_map_encode_tiled(
        crate::tma::dtype_of(Dtype::Bf16, false)?,
        3,
        buf.ptr as *mut _,
        &[
            hidden as u64,
            num_tokens as u64,
            mlayout::NUM_DEVICE_LOCALITY_DOMAINS.min(4) as u64,
        ],
        &[
            hidden as u64 * elem,
            hidden as u64 * mlayout::NUM_DEVICE_LOCALITY_DOMAINS.min(4) as u64 * elem,
        ],
        &[mlayout::MHC_BLOCK_K, mlayout::MHC_BLOCK_M, 4],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(mlayout::MHC_SWIZZLE),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// mHC fn 3D map: fn [24, R*H] fp32 -> dims (H, 24, R); strides (4H, H).
pub fn make_tma_mhc_fn(dev: &Device, buf: &DevBuffer, hidden: u32) -> DgResult<sys::TensorMap> {
    dev.bind()?;
    let routes = 4u64;
    sys::tensor_map_encode_tiled(
        crate::tma::dtype_of(Dtype::F32, false)?,
        3,
        buf.ptr as *mut _,
        &[hidden as u64, mlayout::MHC_NUM_HC_OUTPUTS as u64, routes],
        &[(routes * hidden as u64) * 4, (hidden as u64) * 4],
        &[mlayout::MHC_BLOCK_K, mlayout::MHC_NUM_HC_OUTPUTS, 4],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(mlayout::MHC_SWIZZLE),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}

/// mHC coefficient 2D map: [T, channels] fp32; box (channels, BLOCK_M).
pub fn make_tma_mhc_coeff(
    dev: &Device,
    buf: &DevBuffer,
    channels: u32,
    num_tokens: u32,
    swizzle_mode: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(
        dev,
        Dtype::F32,
        false,
        buf.ptr,
        channels,
        num_tokens,
        channels,
        mlayout::MHC_BLOCK_M,
        channels,
        swizzle_mode,
    )
}

/// General strided 3D operand map for the einsum permutations: dims
/// `(inner, outer, batch)` with EXPLICIT element strides for the outer and
/// batch dims (e.g. upstream `bhr,hdr->bhd` maps A `[b, h, r]` to
/// `(r, b, h)` with strides `(h*r, r)` — the batch dim is NOT the outermost
/// memory dim). `smem_inner`/`smem_outer` are the SMEM box dims.
#[allow(clippy::too_many_arguments)]
pub fn make_tma_3d_strided(
    dev: &Device,
    dtype: Dtype,
    buf: &DevBuffer,
    inner: u32,
    outer: u32,
    batch: u32,
    _smem_inner: u32,
    smem_outer: u32,
    stride_outer: u64, // elements
    stride_batch: u64, // elements
    swizzle_mode: u32,
) -> DgResult<sys::TensorMap> {
    let elem = dtype.elem_size() as u64;
    // Swizzle constrains the SMEM box inner extent to the swizzle atom.
    let s0 = if swizzle_mode != 0 {
        swizzle_mode as u64 / elem
    } else {
        inner as u64
    };
    dev.bind()?;
    sys::tensor_map_encode_tiled(
        dtype_of(dtype, false)?,
        3,
        buf.ptr as *mut _,
        &[inner as u64, outer as u64, batch as u64],
        &[stride_outer * elem, stride_batch * elem],
        &[s0 as u32, smem_outer, 1],
        &[1, 1, 1],
        sys::tm_interleave_none(),
        sys::tm_swizzle(swizzle_mode),
        sys::tm_l2_256b(),
        sys::tm_oob_fill_none(),
    )
}
