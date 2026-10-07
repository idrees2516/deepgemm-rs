//! Device context: primary context + stream + device properties + buffers.

use crate::error::DgResult;
use crate::sys;
use std::sync::Arc;

/// Architecture family, derived from the compute capability.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    /// Hopper (SM90): WGMMA + TMA.
    Sm90,
    /// Blackwell datacenter (SM100): tcgen05 (B200/GB200).
    Sm100,
    /// Blackwell consumer/other — unsupported for tcgen05 kernels.
    Other(i32, i32),
}

impl Arch {
    pub fn from_cc(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (9, _) => Arch::Sm90,
            (10, 0) => Arch::Sm100,
            _ => Arch::Other(major, minor),
        }
    }

    /// NVRTC arch string (`-arch`).
    pub fn nvrtc_arch(&self) -> &'static str {
        match self {
            Arch::Sm90 => "90a",
            Arch::Sm100 => "100a",
            Arch::Other(m, n) => match (*m, *n) {
                (10, 3) => "103a",
                (12, 0) => "120a",
                _ => "100a",
            },
        }
    }
}

/// GPU device handle: retains the primary context, owns a stream, and caches
/// device properties needed by the heuristics.
pub struct Device {
    pub ordinal: i32,
    pub name: String,
    pub arch: Arch,
    pub cc: (i32, i32),
    pub num_sms: u32,
    /// Max dynamic shared memory per block (opt-in), bytes.
    pub smem_capacity: u32,
    ctx: sys::Ctx,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn new(ordinal: i32) -> DgResult<Arc<Device>> {
        sys::init()?;
        let dev = sys::device_get(ordinal)?;
        let name = sys::device_name(dev)?;
        let cc = sys::device_compute_capability(dev)?;
        let num_sms = sys::device_get_attribute(dev, sys::ATTR_MULTIPROCESSOR_COUNT)? as u32;
        let smem_capacity =
            sys::device_get_attribute(dev, sys::ATTR_SHARED_MEMORY_PER_BLOCK_OPTIN)? as u32;
        let ctx = sys::primary_ctx_retain(dev)?;
        sys::ctx_set_current(ctx)?;
        Ok(Arc::new(Device {
            ordinal,
            name,
            arch: Arch::from_cc(cc.0, cc.1),
            cc,
            num_sms,
            smem_capacity,
            ctx,
        }))
    }

    pub fn ctx(&self) -> sys::Ctx {
        self.ctx
    }

    /// Bind this device's context to the calling thread (required before raw
    /// driver calls in a fresh thread).
    pub fn bind(&self) -> DgResult<()> {
        sys::ctx_set_current(self.ctx)
    }

    /// The theoretical dense peak for the tensor core path, in TFLOPS
    /// (multiply-accumulate pairs per second). Values for the well-known
    /// datacenter parts at boost clocks; used only for bench reporting.
    pub fn peak_tflops(&self, dtype: crate::types::Dtype) -> f64 {
        match (&self.name[..], self.cc) {
            (n, _) if n.contains("B200") || n.contains("GB200") => match dtype {
                crate::types::Dtype::Fp4 => 9000.0,
                crate::types::Dtype::Fp8 => 4500.0,
                crate::types::Dtype::Bf16 => 2250.0,
                _ => 2250.0,
            },
            (n, _) if n.contains("B100") => match dtype {
                crate::types::Dtype::Fp4 => 3500.0,
                crate::types::Dtype::Fp8 => 1750.0,
                crate::types::Dtype::Bf16 => 875.0,
                _ => 875.0,
            },
            (n, _) if n.contains("H100") || n.contains("H800") || n.contains("H200") => match dtype
            {
                crate::types::Dtype::Fp8 => 1979.0,
                crate::types::Dtype::Bf16 => 989.0,
                _ => 989.0,
            },
            _ => {
                // Generic estimate: 2048 FMA pipes/SM * 2 ops * clock.
                match dtype {
                    crate::types::Dtype::Fp4 => 2.0 * 4096.0 * 128.0 * 1.75e9 / 1e12,
                    crate::types::Dtype::Fp8 => 4096.0 * 128.0 * 1.75e9 / 1e12,
                    crate::types::Dtype::Bf16 => 2048.0 * 128.0 * 1.75e9 / 1e12,
                    _ => 2048.0 * 128.0 * 1.75e9 / 1e12,
                }
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Primary context is ref-counted by the driver; nothing to do.
    }
}

/// RAII device stream.
pub struct DevStream {
    raw: sys::Stream,
}

unsafe impl Send for DevStream {}
unsafe impl Sync for DevStream {}

impl DevStream {
    pub fn new(dev: &Device) -> DgResult<DevStream> {
        dev.bind()?;
        Ok(DevStream {
            raw: sys::stream_create()?,
        })
    }

    pub fn raw(&self) -> sys::Stream {
        self.raw
    }

    pub fn sync(&self) -> DgResult<()> {
        sys::stream_synchronize(self.raw)
    }
}

impl Drop for DevStream {
    fn drop(&mut self) {
        sys::stream_destroy(self.raw);
    }
}

/// RAII device buffer of bytes.
pub struct DevBuffer {
    pub ptr: sys::DevicePtr,
    pub len: usize,
}

unsafe impl Send for DevBuffer {}
unsafe impl Sync for DevBuffer {}

impl DevBuffer {
    pub fn alloc(dev: &Device, bytes: usize) -> DgResult<DevBuffer> {
        dev.bind()?;
        if bytes == 0 {
            return Ok(DevBuffer { ptr: 0, len: 0 });
        }
        Ok(DevBuffer {
            ptr: sys::mem_alloc(bytes)?,
            len: bytes,
        })
    }

    pub fn alloc_zeros(dev: &Device, bytes: usize) -> DgResult<DevBuffer> {
        let buf = Self::alloc(dev, bytes)?;
        if bytes > 0 {
            let s = sys::stream_create()?;
            sys::memset_d8(buf.ptr, 0, bytes, s)?;
            sys::stream_destroy(s);
        }
        Ok(buf)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_host(&self, dev: &Device, src: &[u8], stream: sys::Stream) -> DgResult<()> {
        assert!(
            src.len() <= self.len,
            "host slice larger than device buffer"
        );
        dev.bind()?;
        if !src.is_empty() {
            sys::memcpy_h2d(self.ptr, src.as_ptr() as *const _, src.len(), stream)?;
        }
        Ok(())
    }

    pub fn copy_to_host(&self, dev: &Device, dst: &mut [u8], stream: sys::Stream) -> DgResult<()> {
        assert!(
            dst.len() <= self.len,
            "host slice larger than device buffer"
        );
        dev.bind()?;
        if !dst.is_empty() {
            sys::memcpy_d2h(dst.as_mut_ptr() as *mut _, self.ptr, dst.len(), stream)?;
        }
        Ok(())
    }
}

impl Drop for DevBuffer {
    fn drop(&mut self) {
        if self.len != 0 {
            sys::mem_free(self.ptr);
        }
    }
}

/// Typed view helpers for DevBuffer.
pub fn alloc_and_upload<T: bytemuck::Pod>(
    dev: &Device,
    data: &[T],
    stream: sys::Stream,
) -> DgResult<DevBuffer> {
    let bytes = std::mem::size_of_val(data);
    let buf = DevBuffer::alloc(dev, bytes.max(1))?;
    if bytes > 0 {
        buf.copy_from_host(dev, bytemuck::cast_slice(data), stream)?;
    }
    Ok(buf)
}

pub fn download<T: bytemuck::Pod + Default + Clone>(
    dev: &Device,
    buf: &DevBuffer,
    stream: sys::Stream,
) -> DgResult<Vec<T>> {
    let n = buf.len / std::mem::size_of::<T>();
    let mut out = vec![T::default(); n];
    if n > 0 {
        buf.copy_to_host(dev, bytemuck::cast_slice_mut(&mut out), stream)?;
    }
    Ok(out)
}
