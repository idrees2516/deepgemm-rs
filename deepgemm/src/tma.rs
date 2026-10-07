//! TMA descriptor construction (port of upstream `runtime_utils.hpp`).

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
    make_tma_2d(dev, dtype, fp4_unpacked_smem, buf.ptr, gmem_inner, gmem_outer,
                smem_inner, smem_outer, outer_stride, swizzle_mode)
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
    let elem_wire = if dtype == Dtype::Fp4 { 1 } else { dtype.elem_size() as u32 };
    let (inner, outer, _smem_inner, smem_outer) = match major {
        Major::K => (k, rows, block_k, load_block_mn),
        Major::Mn => (rows, k, load_block_mn, block_k),
    };
    let s0 = if swizzle_mode != 0 { swizzle_mode / elem_wire } else { inner };
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
    make_tma_2d(dev, dtype, false, buf.ptr, cols, rows * num_groups,
                store_block_n, store_block_m, row_stride, swizzle_mode)
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
    let stride1 = cols as u64 * elem;
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
    rows: u32,          // logical MN (or per-group MN)
    k: u32,             // logical K
    gran_k: u32,        // SF granularity (32 or 128)
    block_mn: u32,      // SMEM box along MN (SF_BLOCK_M/N)
    sf_block_k: u32,    // SMEM box along packed K rows
    num_groups: u32,
    packed_row_stride: u32, // 0 => compact TMA-aligned layout
) -> DgResult<sys::TensorMap> {
    let tma_aligned = tma_aligned_size(rows, 4);
    let packed_rows = ceil_div(k, gran_k * 4);
    let outer_stride = if packed_row_stride == 0 { tma_aligned } else { packed_row_stride };
    if outer_stride < tma_aligned {
        return Err(DgError::InvalidArg("SF row stride smaller than TMA-aligned MN".into()));
    }
    make_tma_2d(dev, Dtype::F32, false, buf.ptr, tma_aligned, packed_rows * num_groups,
                block_mn, sf_block_k.max(1), outer_stride, 0)
}

/// Weights map for MQA logits: bf16 [heads, tokens] (inner = heads).
pub fn make_tma_mqa_weights(
    dev: &Device,
    buf: &DevBuffer,
    num_heads: u32,
    num_tokens: u32,
) -> DgResult<sys::TensorMap> {
    make_tma_2d(dev, Dtype::Bf16, false, buf.ptr, num_heads, num_tokens,
                num_heads, 1, num_heads, 0)
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
    make_tma_2d(dev, dtype, false, buf.ptr, head_dim, rows,
                head_dim, load_rows, head_dim, swizzle)
}

/// Q / KV SF map for MQA logits: int32 [rows, 1] packed.
pub fn make_tma_mqa_sf(
    dev: &Device,
    buf: &DevBuffer,
    rows: u32,
    load_rows: u32,
) -> DgResult<sys::TensorMap> {
    let tma_aligned = tma_aligned_size(rows, 4);
    make_tma_2d(dev, Dtype::F32, false, buf.ptr, tma_aligned, 1,
                load_rows, 1, tma_aligned, 0)
}
