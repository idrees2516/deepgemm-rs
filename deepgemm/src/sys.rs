//! Thin, checked wrappers over the CUDA driver API (via `cudarc::driver::sys`).
//!
//! DeepGEMM-RS needs a few driver entry points that the safe cudarc API does not
//! expose (cluster launches via `cuLaunchKernelEx`, `cuTensorMapEncodeTiled`,
//! `cuFuncSetAttribute` for large dynamic shared memory). Everything goes through
//! the same dlopen'd `libcuda` that cudarc loads, so no extra linking is needed.

use cudarc::driver::sys::{self, CUlaunchAttribute, CUlaunchConfig};

#[inline]
fn dg_lib() -> &'static sys::Lib {
    unsafe { sys::lib() }
}
use std::ffi::c_void;
use std::sync::OnceLock;

use crate::error::{DgError, DgResult};

pub type Ctx = sys::CUcontext;
pub type Stream = sys::CUstream;
pub type Func = sys::CUfunction;
pub type Module = sys::CUmodule;
pub type DevicePtr = sys::CUdeviceptr;
pub type TensorMap = sys::CUtensorMap;

pub const CU_STREAM_NON_BLOCKING: u32 = 0x01;

#[inline]
fn cu(res: sys::CUresult) -> DgResult<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DgError::Driver(format!("CUDA driver error {res:?}")))
    }
}

fn opt_fn<T, E>(f: &Result<T, E>, name: &str) -> DgResult<T>
where
    T: Copy,
    E: std::fmt::Debug,
{
    match f {
        Ok(v) => Ok(*v),
        Err(e) => Err(DgError::Driver(format!("symbol {name} unavailable: {e:?}"))),
    }
}

/// Ensure `cuInit` has run (idempotent).
pub fn init() -> DgResult<()> {
    static INIT: OnceLock<()> = OnceLock::new();
    if INIT.get().is_none() {
        let f = opt_fn(&dg_lib().cuInit, "cuInit")?;
        cu(unsafe { f(0) })?;
        let _ = INIT.set(());
    }
    Ok(())
}

pub fn device_count() -> DgResult<i32> {
    init()?;
    let mut n = 0i32;
    let f = opt_fn(&dg_lib().cuDeviceGetCount, "cuDeviceGetCount")?;
    cu(unsafe { f(&mut n) })?;
    Ok(n)
}

pub fn device_get(ordinal: i32) -> DgResult<sys::CUdevice> {
    let mut dev: sys::CUdevice = 0;
    let f = opt_fn(&dg_lib().cuDeviceGet, "cuDeviceGet")?;
    cu(unsafe { f(&mut dev, ordinal) })?;
    Ok(dev)
}

pub type DeviceAttr = sys::CUdevice_attribute_enum;
pub const ATTR_MULTIPROCESSOR_COUNT: DeviceAttr =
    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
pub const ATTR_SHARED_MEMORY_PER_BLOCK_OPTIN: DeviceAttr =
    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN;
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: DeviceAttr =
    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
pub const ATTR_COMPUTE_CAPABILITY_MINOR: DeviceAttr =
    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;

pub fn device_get_attribute(dev: sys::CUdevice, attr: DeviceAttr) -> DgResult<i32> {
    let mut v = 0i32;
    let f = opt_fn(&dg_lib().cuDeviceGetAttribute, "cuDeviceGetAttribute")?;
    cu(unsafe { f(&mut v, attr, dev) })?;
    Ok(v)
}

/// Compute capability as (major, minor).
pub fn device_compute_capability(dev: sys::CUdevice) -> DgResult<(i32, i32)> {
    Ok((
        device_get_attribute(dev, ATTR_COMPUTE_CAPABILITY_MAJOR)?,
        device_get_attribute(dev, ATTR_COMPUTE_CAPABILITY_MINOR)?,
    ))
}

pub fn device_name(dev: sys::CUdevice) -> DgResult<String> {
    let mut buf = [0u8; 128];
    let f = opt_fn(&dg_lib().cuDeviceGetName, "cuDeviceGetName")?;
    cu(unsafe { f(buf.as_mut_ptr() as *mut i8, 128, dev) })?;
    let end = buf.iter().position(|&c| c == 0).unwrap_or(128);
    Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
}

pub fn primary_ctx_retain(dev: sys::CUdevice) -> DgResult<Ctx> {
    init()?;
    let mut ctx = std::ptr::null_mut();
    let f = opt_fn(&dg_lib().cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")?;
    cu(unsafe { f(&mut ctx, dev) })?;
    Ok(ctx)
}

pub fn ctx_set_current(ctx: Ctx) -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuCtxSetCurrent, "cuCtxSetCurrent")?;
    cu(unsafe { f(ctx) })
}

pub fn stream_create() -> DgResult<Stream> {
    let mut s = std::ptr::null_mut();
    let f = opt_fn(&dg_lib().cuStreamCreate, "cuStreamCreate")?;
    cu(unsafe { f(&mut s, CU_STREAM_NON_BLOCKING) })?;
    Ok(s)
}

pub fn stream_destroy(s: Stream) {
    if let Ok(f) = opt_fn(&dg_lib().cuStreamDestroy_v2, "cuStreamDestroy_v2") {
        let _ = unsafe { f(s) };
    }
}

pub fn stream_synchronize(s: Stream) -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuStreamSynchronize, "cuStreamSynchronize")?;
    cu(unsafe { f(s) })
}

pub fn ctx_synchronize() -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuCtxSynchronize, "cuCtxSynchronize")?;
    cu(unsafe { f() })
}

pub fn mem_alloc(bytes: usize) -> DgResult<DevicePtr> {
    let mut p = 0u64;
    let f = opt_fn(&dg_lib().cuMemAlloc_v2, "cuMemAlloc_v2")?;
    cu(unsafe { f(&mut p, bytes) })?;
    Ok(p)
}

pub fn mem_free(p: DevicePtr) {
    if let Ok(f) = opt_fn(&dg_lib().cuMemFree_v2, "cuMemFree_v2") {
        let _ = unsafe { f(p) };
    }
}

pub fn memcpy_h2d(dst: DevicePtr, src: *const c_void, bytes: usize, s: Stream) -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuMemcpyHtoDAsync_v2, "cuMemcpyHtoDAsync_v2")?;
    cu(unsafe { f(dst, src, bytes, s) })?;
    stream_synchronize(s)
}

pub fn memcpy_d2h(dst: *mut c_void, src: DevicePtr, bytes: usize, s: Stream) -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuMemcpyDtoHAsync_v2, "cuMemcpyDtoHAsync_v2")?;
    cu(unsafe { f(dst, src, bytes, s) })?;
    stream_synchronize(s)
}

pub fn memcpy_d2d(dst: DevicePtr, src: DevicePtr, bytes: usize, s: Stream) -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuMemcpyDtoDAsync_v2, "cuMemcpyDtoDAsync_v2")?;
    cu(unsafe { f(dst, src, bytes, s) })?;
    stream_synchronize(s)
}

pub fn memset_d8(dst: DevicePtr, value: u8, bytes: usize, s: Stream) -> DgResult<()> {
    let f = opt_fn(&dg_lib().cuMemsetD8Async, "cuMemsetD8Async")?;
    cu(unsafe { f(dst, value, bytes, s) })?;
    stream_synchronize(s)
}

/// Load a PTX/CUBIN image (NUL-terminated string) as a module.
pub fn module_load_data(image: *const c_void) -> DgResult<Module> {
    let f = opt_fn(&dg_lib().cuModuleLoadData, "cuModuleLoadData")?;
    let mut m = std::ptr::null_mut();
    cu(unsafe { f(&mut m, image) })?;
    Ok(m)
}

pub fn module_unload(m: Module) {
    if let Ok(f) = opt_fn(&dg_lib().cuModuleUnload, "cuModuleUnload") {
        let _ = unsafe { f(m) };
    }
}

pub fn module_get_function(m: Module, name: &str) -> DgResult<Func> {
    let f = opt_fn(&dg_lib().cuModuleGetFunction, "cuModuleGetFunction")?;
    let mut func = std::ptr::null_mut();
    let c = std::ffi::CString::new(name).map_err(|e| DgError::Driver(e.to_string()))?;
    cu(unsafe { f(&mut func, m, c.as_ptr()) })?;
    Ok(func)
}

pub type FuncAttr = sys::CUfunction_attribute_enum;
pub const FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: FuncAttr =
    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;

pub fn func_set_attribute(f: Func, attr: FuncAttr, value: i32) -> DgResult<()> {
    let g = opt_fn(&dg_lib().cuFuncSetAttribute, "cuFuncSetAttribute")?;
    cu(unsafe { g(f, attr, value) })
}

pub fn func_set_cache_config(f: Func, pref: sys::CUfunc_cache_enum) -> DgResult<()> {
    let g = opt_fn(&dg_lib().cuFuncSetCacheConfig, "cuFuncSetCacheConfig")?;
    cu(unsafe { g(f, pref) })
}

/// Launch configuration accepted by [`launch_kernel_ex`].
#[derive(Clone, Copy, Debug)]
pub struct LaunchEx {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
    pub cluster: Option<(u32, u32, u32)>,
    /// Programmatic dependent launch (upstream DeepGEMM enables PDL).
    pub pdl: bool,
}

/// Launch a kernel with optional cluster dims and PDL via `cuLaunchKernelEx`.
///
/// # Safety
/// `params` must point to an array of `void*` of length `num_params`, each
/// pointing to storage matching the kernel's parameter types.
pub unsafe fn launch_kernel_ex(func: Func, cfg: &LaunchEx, stream: Stream, params: &[*const c_void]) -> DgResult<()> {
    let mut attrs: [CUlaunchAttribute; 2] = std::mem::zeroed();
    let mut num_attrs = 0u32;
    if let Some((cx, cy, cz)) = cfg.cluster {
        attrs[num_attrs as usize].id = sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attrs[num_attrs as usize].value.clusterDim.x = cx;
        attrs[num_attrs as usize].value.clusterDim.y = cy;
        attrs[num_attrs as usize].value.clusterDim.z = cz;
        num_attrs += 1;
    }
    if cfg.pdl {
        attrs[num_attrs as usize].id =
            sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attrs[num_attrs as usize].value.programmaticStreamSerializationAllowed = 1;
        num_attrs += 1;
    }

    let config = CUlaunchConfig {
        gridDimX: cfg.grid.0,
        gridDimY: cfg.grid.1,
        gridDimZ: cfg.grid.2,
        blockDimX: cfg.block.0,
        blockDimY: cfg.block.1,
        blockDimZ: cfg.block.2,
        sharedMemBytes: cfg.smem,
        hStream: stream,
        attrs: if num_attrs == 0 { std::ptr::null_mut() } else { attrs.as_mut_ptr() },
        numAttrs: num_attrs,
    };
    let f = opt_fn(&dg_lib().cuLaunchKernelEx, "cuLaunchKernelEx")?;
    cu(f(&config, func, params.as_ptr() as *mut *mut c_void, std::ptr::null_mut()))
}

/// Encode a tiled tensor map for TMA. Mirrors `cuTensorMapEncodeTiled`.
#[allow(clippy::too_many_arguments)]
pub fn tensor_map_encode_tiled(
    dtype: sys::CUtensorMapDataType,
    rank: u32,
    addr: *mut c_void,
    gmem_dims: &[u64],
    gmem_strides_bytes: &[u64],
    smem_dims: &[u32],
    elem_strides: &[u32],
    interleave: sys::CUtensorMapInterleave,
    swizzle: sys::CUtensorMapSwizzle,
    l2_promotion: sys::CUtensorMapL2promotion,
    oob_fill: sys::CUtensorMapFloatOOBfill,
) -> DgResult<TensorMap> {
    let mut map: TensorMap = unsafe { std::mem::zeroed() };
    let f = opt_fn(&dg_lib().cuTensorMapEncodeTiled, "cuTensorMapEncodeTiled")?;
    cu(unsafe {
        f(
            &mut map,
            dtype,
            rank,
            addr,
            gmem_dims.as_ptr(),
            gmem_strides_bytes.as_ptr(),
            smem_dims.as_ptr(),
            elem_strides.as_ptr(),
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )
    })?;
    Ok(map)
}

// Re-exported enum constants (values are ABI-stable).
pub use sys::CUtensorMapDataType as TmDtype;
pub use sys::CUtensorMapInterleave as TmInterleave;
pub use sys::CUtensorMapL2promotion as TmL2Promo;
pub use sys::CUtensorMapFloatOOBfill as TmOobFill;
pub use sys::CUtensorMapSwizzle as TmSwizzle;

pub fn tm_dtype_uint8() -> TmDtype {
    sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8
}
pub fn tm_dtype_int32() -> TmDtype {
    sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32
}
pub fn tm_dtype_bfloat16() -> TmDtype {
    sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
}
pub fn tm_dtype_float32() -> TmDtype {
    sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32
}
pub fn tm_dtype_16u4_align16b() -> TmDtype {
    sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
}
pub fn tm_dtype_16u4_align8b() -> TmDtype {
    sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
}
pub fn tm_swizzle(mode: u32) -> TmSwizzle {
    match mode {
        0 | 16 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        32 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
        64 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        128 => sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        _ => panic!("invalid swizzle mode {mode}"),
    }
}

pub fn tm_interleave_none() -> TmInterleave {
    sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE
}
pub fn tm_l2_256b() -> TmL2Promo {
    sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
}
pub fn tm_oob_fill_none() -> TmOobFill {
    sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
}
