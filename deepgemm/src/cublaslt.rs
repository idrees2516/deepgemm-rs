//! cuBLASLt integration (port of upstream `csrc/jit_kernels/impls/smxx_cublaslt.hpp`).
//!
//! Upstream DeepGEMM keeps cuBLASLt around for the cases its own kernels
//! intentionally do not cover, and uses it as the reference in tests and
//! benchmarks:
//!
//! * `bhd,bhr->hdr` einsum — the GEMM K dim is the (small) batch, which the
//!   DeepGEMM kernels' K-multiple constraints don't accommodate.
//! * BF16 reference / fallback GEMMs (`cublaslt_gemm_nt` and friends).
//! * NVFP4 block-scaled reference (`cublaslt_nvfp4_gemm_nt`, UE4M3 scales
//!   with 16-element blocks — cuBLASLt's only FP4 recipe).
//!
//! The port is a faithful translation of `call_cublaslt_api`: cuBLASLt is
//! column-major, so the operands are swapped (`rhs` plays "A") and the
//! transpose flags come from the *data* majors. A persistent handle +
//! 32 MiB workspace is kept per process (upstream's runtime workspace
//! sizing); the heuristic picks the fastest algorithm each call.

use crate::api_einsum::Tensor3;
use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::types::{Dtype, Major};
use std::sync::{Mutex, OnceLock};

// cuBLAS ABI constants (cublasOperation_t): N = 0, T = 1.
const OP_N: i32 = 0;
const OP_T: i32 = 1;

const WS_BYTES: usize = 32 << 20;

struct LtState {
    handle: SendHandle,
    workspace: DevBuffer,
}

/// cuBLASLt handles are plain context pointers; usage here is serialized
/// behind the global mutex, so the wrapper is `Send` by construction.
struct SendHandle(cudarc::cublaslt::sys::cublasLtHandle_t);
unsafe impl Send for SendHandle {}

static STATE: OnceLock<Mutex<Option<LtState>>> = OnceLock::new();

fn ensure_handle_and_ws(
    dev: &Device,
) -> DgResult<(cudarc::cublaslt::sys::cublasLtHandle_t, DevBuffer)> {
    let mut guard = STATE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let handle = cudarc::cublaslt::result::create_handle()
            .map_err(|e| DgError::Unsupported(format!("cublasLt handle: {e:?}")))?;
        let ws = DevBuffer::alloc_zeros(dev, WS_BYTES)?;
        *guard = Some(LtState {
            handle: SendHandle(handle),
            workspace: ws,
        });
    }
    let s = guard.as_ref().expect("just initialized");
    Ok((s.handle.0, s.workspace.clone_view()))
}

fn cublas_type(dt: Dtype) -> DgResult<cudarc::cublaslt::sys::cudaDataType> {
    use cudarc::cublaslt::sys::cudaDataType as T;
    Ok(match dt {
        Dtype::Bf16 => T::CUDA_R_16BF,
        Dtype::F32 => T::CUDA_R_32F,
        Dtype::Fp8 => T::CUDA_R_8F_E4M3,
        Dtype::Fp4 => T::CUDA_R_4F_E2M1,
    })
}

/// One operand's column-major layout: (ptr, dtype, rows, cols, ld,
/// optional (batch_count, batch_stride_bytes... in ELEMENTS — the API takes
/// bytes for STRIDED_BATCH_OFFSET? No: elements — upstream passes
/// `t.stride(0)` which is in elements).
type LayoutSpec = (
    u64,
    cudarc::cublaslt::sys::cudaDataType,
    i64,
    i64,
    i64,
    Option<(i32, i64)>,
);

/// The core matmul (port of `call_cublaslt_api`).
#[allow(clippy::too_many_arguments)]
fn call(
    dev: &Device,
    stream: &DevStream,
    trans_a: i32,
    trans_b: i32,
    a: LayoutSpec,
    b: LayoutSpec,
    d: LayoutSpec,
    accumulate: bool,
    alpha: f32,
    block_sf: Option<(u64, u64)>,
) -> DgResult<()> {
    use cudarc::cublaslt::{result as lt, sys};
    unsafe {
        let (handle, ws) = ensure_handle_and_ws(dev)?;
        let with_sf = block_sf.is_some();
        let compute = if with_sf {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        };
        let scale_t = sys::cudaDataType::CUDA_R_32F;

        let mk_layout = |(ptr_dt, rows, cols, ld, batch): (
            sys::cudaDataType,
            i64,
            i64,
            i64,
            Option<(i32, i64)>,
        )|
         -> DgResult<sys::cublasLtMatrixLayout_t> {
            let _ = ptr_dt;
            let l = lt::create_matrix_layout(ptr_dt, rows as u64, cols as u64, ld)
                .map_err(|e| DgError::Runtime(format!("cublasLt layout: {e:?}")))?;
            if let Some((count, stride)) = batch {
                lt::set_matrix_layout_attribute(
                    l,
                    sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                    (&count) as *const i32 as *const std::ffi::c_void,
                    std::mem::size_of::<i32>(),
                )
                .map_err(|e| DgError::Runtime(format!("cublasLt batch: {e:?}")))?;
                lt::set_matrix_layout_attribute(
                    l,
                    sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    (&stride) as *const i64 as *const std::ffi::c_void,
                    std::mem::size_of::<i64>(),
                )
                .map_err(|e| DgError::Runtime(format!("cublasLt batch: {e:?}")))?;
            }
            Ok(l)
        };
        let layout_a = mk_layout((a.1, a.2, a.3, a.4, a.5))?;
        let layout_b = mk_layout((b.1, b.2, b.3, b.4, b.5))?;
        let layout_d = mk_layout((d.1, d.2, d.3, d.4, d.5))?;

        let desc = lt::create_matmul_desc(compute, scale_t)
            .map_err(|e| DgError::Runtime(format!("cublasLt desc: {e:?}")))?;
        let set =
            |attr: sys::cublasLtMatmulDescAttributes_t, v: *const std::ffi::c_void, sz: usize| {
                lt::set_matmul_desc_attribute(desc, attr, v, sz)
                    .map_err(|e| DgError::Runtime(format!("cublasLt desc attr: {e:?}")))
            };
        set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&trans_a) as *const i32 as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )?;
        set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            (&trans_b) as *const i32 as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )?;
        set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SCALE_TYPE,
            (&scale_t) as *const sys::cudaDataType as *const std::ffi::c_void,
            std::mem::size_of::<sys::cudaDataType>(),
        )?;
        if let Some((sf_a, sf_b)) = block_sf {
            let mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                (&mode) as *const sys::cublasLtMatmulMatrixScale_t as *const std::ffi::c_void,
                std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
            )?;
            set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                (&mode) as *const sys::cublasLtMatmulMatrixScale_t as *const std::ffi::c_void,
                std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
            )?;
            set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                (&sf_a) as *const u64 as *const std::ffi::c_void,
                std::mem::size_of::<u64>(),
            )?;
            set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                (&sf_b) as *const u64 as *const std::ffi::c_void,
                std::mem::size_of::<u64>(),
            )?;
        }
        let num_sms = dev.num_sms as i32;
        set(
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
            (&num_sms) as *const i32 as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )?;
        if a.1 == sys::cudaDataType::CUDA_R_8F_E4M3 {
            let fast_accum: i32 = 0;
            set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                (&fast_accum) as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
            )?;
        }

        let pref = lt::create_matmul_pref()
            .map_err(|e| DgError::Runtime(format!("cublasLt pref: {e:?}")))?;
        let ws_len = ws.len;
        lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&ws_len) as *const usize as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .map_err(|e| DgError::Runtime(format!("cublasLt pref attr: {e:?}")))?;
        let reduction: i32 = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as i32
            | sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as i32;
        lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            (&reduction) as *const i32 as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )
        .map_err(|e| DgError::Runtime(format!("cublasLt pref attr: {e:?}")))?;

        let heuristic = lt::get_matmul_algo_heuristic(
            handle, desc, layout_a, layout_b, layout_d, layout_d, pref,
        )
        .map_err(|e| DgError::Runtime(format!("cublasLt heuristic: {e:?}")))?;

        let beta: f32 = if accumulate { 1.0 } else { 0.0 };
        lt::matmul(
            handle,
            desc,
            (&alpha) as *const f32 as *const std::ffi::c_void,
            (&beta) as *const f32 as *const std::ffi::c_void,
            a.0 as *const std::ffi::c_void,
            layout_a,
            b.0 as *const std::ffi::c_void,
            layout_b,
            d.0 as *const std::ffi::c_void,
            layout_d,
            d.0 as *mut std::ffi::c_void,
            layout_d,
            &heuristic.algo,
            ws.ptr as *mut std::ffi::c_void,
            ws_len,
            stream.raw() as cudarc::cublaslt::sys::cudaStream_t,
        )
        .map_err(|e| DgError::Runtime(format!("cublasLt matmul: {e:?}")))?;

        lt::destroy_matmul_pref(pref).ok();
        lt::destroy_matmul_desc(desc).ok();
        lt::destroy_matrix_layout(layout_a).ok();
        lt::destroy_matrix_layout(layout_b).ok();
        lt::destroy_matrix_layout(layout_d).ok();
        Ok(())
    }
}

/// Row-major BLAS GEMM `D = alpha * (lhs @ rhs^T) + beta * D` with either
/// major per operand (upstream `cublaslt_gemm`). `lhs`: `[M, K]`,
/// `rhs`: `[N, K]`, `D`: `[M, N]` row-major.
#[allow(clippy::too_many_arguments)]
pub fn cublaslt_gemm(
    dev: &Device,
    stream: &DevStream,
    lhs: &DevBuffer,
    rhs: &DevBuffer,
    out: &DevBuffer,
    m: u32,
    n: u32,
    k: u32,
    lhs_major: Major,
    rhs_major: Major,
    dtype: Dtype,
    accumulate: bool,
    alpha: f32,
) -> DgResult<()> {
    let dt = cublas_type(dtype)?;
    // Column-major swap (upstream): A = rhs viewed `[K, N]`, B = lhs
    // viewed `[K, M]`, D = `[N, M]`. The transpose flags invert the
    // row-major majors; ld = the pitch of the contiguous run.
    let trans_a = if rhs_major == Major::K { OP_T } else { OP_N };
    let trans_b = if lhs_major == Major::K { OP_N } else { OP_T };
    let (a_rows, a_cols, ld_a) = if rhs_major == Major::K {
        (k as i64, n as i64, k as i64)
    } else {
        (n as i64, k as i64, n as i64)
    };
    let (b_rows, b_cols, ld_b) = if lhs_major == Major::K {
        (k as i64, m as i64, k as i64)
    } else {
        (m as i64, k as i64, m as i64)
    };
    call(
        dev,
        stream,
        trans_a,
        trans_b,
        (rhs.ptr, dt, a_rows, a_cols, ld_a, None),
        (lhs.ptr, dt, b_rows, b_cols, ld_b, None),
        (out.ptr, dt, n as i64, m as i64, n as i64, None),
        accumulate,
        alpha,
        None,
    )
}

/// [`cublaslt_gemm`] with NT layout (both operands K-major) — the reference
/// GEMM used by tests and the bench baseline.
#[allow(clippy::too_many_arguments)]
pub fn cublaslt_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    lhs: &DevBuffer,
    rhs: &DevBuffer,
    out: &DevBuffer,
    m: u32,
    n: u32,
    k: u32,
    dtype: Dtype,
    accumulate: bool,
) -> DgResult<()> {
    cublaslt_gemm(
        dev,
        stream,
        lhs,
        rhs,
        out,
        m,
        n,
        k,
        Major::K,
        Major::K,
        dtype,
        accumulate,
        1.0,
    )
}

/// NVFP4 block-scaled GEMM (upstream `cublaslt_nvfp4_gemm`): both operands
/// K-major packed FP4 with UE4M3 scale factors in cuBLASLt's 16-element
/// VEC16 layout (`sf[ceil(K/16), MN]` E4M3 bytes, MN-contiguous). NT only.
#[allow(clippy::too_many_arguments)]
pub fn cublaslt_nvfp4_gemm_nt(
    dev: &Device,
    stream: &DevStream,
    lhs: &DevBuffer,    // [M, K/2] packed e2m1
    lhs_sf: &DevBuffer, // [ceil(K/16), M] UE4M3 bytes (MN-contiguous)
    rhs: &DevBuffer,    // [N, K/2]
    rhs_sf: &DevBuffer, // [ceil(K/16), N]
    out: &DevBuffer,    // [M, N] BF16
    m: u32,
    n: u32,
    k: u32,
    accumulate: bool,
) -> DgResult<()> {
    // Dimensions/lds in *logical* FP4 elements (2 per byte): ld = K.
    let dt4 = cudarc::cublaslt::sys::cudaDataType::CUDA_R_4F_E2M1;
    let dtb = cudarc::cublaslt::sys::cudaDataType::CUDA_R_16BF;
    call(
        dev,
        stream,
        OP_T,
        OP_N,
        (rhs.ptr, dt4, k as i64, n as i64, k as i64, None),
        (lhs.ptr, dt4, k as i64, m as i64, k as i64, None),
        (out.ptr, dtb, n as i64, m as i64, n as i64, None),
        accumulate,
        1.0,
        Some((rhs_sf.ptr, lhs_sf.ptr)),
    )
}

/// `bhr,hdr->bhd` via cuBLASLt (upstream `cublaslt_bhr_hdr_bhd`): batched
/// over h with `(m=d, n=b, k=r)`; the batch strides are the tensors' own
/// h-dim strides. BF16, optional accumulation, optional alpha.
#[allow(clippy::too_many_arguments)]
pub fn bhr_hdr_bhd(
    dev: &Device,
    stream: &DevStream,
    a: &Tensor3, // [b, h, r]
    b: &Tensor3, // [h, d, r]
    d: &Tensor3, // [b, h, d]
    c: Option<&Tensor3>,
    bt: u32,
    h: u32,
    r: u32,
    dt: u32,
    accumulate: bool,
    alpha: f32,
) -> DgResult<()> {
    let cd = c.unwrap_or(d);
    let dtb = cudarc::cublaslt::sys::cudaDataType::CUDA_R_16BF;
    // A = b `[h][d, r]` col-major (r, d) ld=r, batch h stride d*r;
    // B = a `[b, h, r]` col-major (r, b) ld=h*r, batch h stride r;
    // D = `[b, h, d]` col-major (d, b) ld=h*d, batch h stride d.
    call(
        dev,
        stream,
        OP_T,
        OP_N,
        (
            b.data.ptr,
            dtb,
            r as i64,
            dt as i64,
            r as i64,
            Some((h as i32, (dt * r) as i64)),
        ),
        (
            a.data.ptr,
            dtb,
            r as i64,
            bt as i64,
            (h * r) as i64,
            Some((h as i32, r as i64)),
        ),
        (
            cd.data.ptr,
            dtb,
            dt as i64,
            bt as i64,
            (h * dt) as i64,
            Some((h as i32, dt as i64)),
        ),
        accumulate,
        alpha,
        None,
    )
}

/// `bhd,bhr->hdr` via cuBLASLt (upstream `cublaslt_bhd_bhr_hdr`): batched
/// over h with `(m=r, n=d, k=b)`; the einsum entry point (D FP32, C
/// optional, accumulates).
#[allow(clippy::too_many_arguments)]
pub fn bhd_bhr_hdr(
    dev: &Device,
    stream: &DevStream,
    a: &Tensor3, // [b, h, d] BF16
    b: &Tensor3, // [b, h, r] BF16
    d: &Tensor3, // [h, d, r] FP32
    c: Option<&Tensor3>,
    bt: u32,
    h: u32,
    r: u32,
    dt: u32,
) -> DgResult<()> {
    let cd = c.unwrap_or(d);
    let dtb = cudarc::cublaslt::sys::cudaDataType::CUDA_R_16BF;
    let dtf = cudarc::cublaslt::sys::cudaDataType::CUDA_R_32F;
    if (d.data.len as u64) < (h as u64) * (dt as u64) * (r as u64) * 4 {
        return Err(DgError::InvalidArg("D must be FP32 [h, d, r]".into()));
    }
    // A = b `[b,h,r]` col (r, b) ld=h*r, batch h stride r (OP_N: m=r rows);
    // B = a `[b,h,d]` col (d, b) ld=h*d, batch h stride d (OP_T);
    // D `[h,d,r]` col (r, d) ld=r, batch h stride d*r.
    call(
        dev,
        stream,
        OP_N,
        OP_T,
        (
            b.data.ptr,
            dtb,
            r as i64,
            bt as i64,
            (h * r) as i64,
            Some((h as i32, r as i64)),
        ),
        (
            a.data.ptr,
            dtb,
            dt as i64,
            bt as i64,
            (h * dt) as i64,
            Some((h as i32, dt as i64)),
        ),
        (
            cd.data.ptr,
            dtf,
            r as i64,
            dt as i64,
            r as i64,
            Some((h as i32, (dt * r) as i64)),
        ),
        c.is_some(),
        1.0,
        None,
    )
}
