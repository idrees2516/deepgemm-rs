//! JIT engine: assembles kernel sources, compiles them with NVRTC (cached in
//! memory and on disk), loads modules, and launches via `cuLaunchKernelEx`
//! with cluster dims / dynamic smem / optional PDL.

use crate::device::Device;
use crate::error::{DgError, DgResult};
use crate::sys;
use cudarc::nvrtc::result as nvrtc_result;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const PRELUDE: &str = include_str!("../kernels/prelude.h");

/// Kernel translation units (compiled with the prelude prepended).
pub mod kernel_src {
    pub const GEMM_SM100: &str = include_str!("../kernels/gemm_sm100.cu");
    pub const LAYOUT_QUANT: &str = include_str!("../kernels/layout_quant.cu");
    pub const MQA_LOGITS: &str = include_str!("../kernels/mqa_logits_sm100.cu");
}

struct SendPtr<T>(T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

struct JitState {
    modules: HashMap<String, SendPtr<sys::Module>>,
    kernels: HashMap<String, SendPtr<sys::Func>>,
}

fn state() -> &'static Mutex<JitState> {
    static S: OnceLock<Mutex<JitState>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(JitState { modules: HashMap::new(), kernels: HashMap::new() }))
}

fn cache_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("DG_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
            PathBuf::from(home).join(".cache").join("deepgemm-rs")
        });
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compute a stable hash of a string (FNV-1a, 64-bit).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compile a CUDA source for the given arch and return the PTX text.
/// Uses the on-disk cache when available.
fn compile_ptx_cached(source: &str, arch: &str, tag: &str) -> DgResult<String> {
    let key = format!("{:016x}", fnv1a(&format!("{source}\x00{arch}\x00{tag}")));
    if let Some(dir) = cache_dir() {
        let path = dir.join(format!("{key}.{arch}.ptx"));
        if let Ok(ptx) = std::fs::read_to_string(&path) {
            return Ok(ptx);
        }
        let ptx = compile_ptx_nocache(source, arch)?;
        let _ = std::fs::write(&path, &ptx);
        Ok(ptx)
    } else {
        compile_ptx_nocache(source, arch)
    }
}

fn compile_ptx_nocache(source: &str, arch: &str) -> DgResult<String> {
    let prog = nvrtc_result::create_program(source, None)
        .map_err(|e| DgError::Nvrtc(format!("NVRTC create: {e:?}")))?;
    let options: Vec<Vec<u8>> = vec![
        format!("--gpu-architecture=compute_{arch}").into_bytes(),
        b"--std=c++17".to_vec(),
    ];
    unsafe {
        nvrtc_result::compile_program(prog, &options)
            .map_err(|e| {
                let log = nvrtc_result::get_program_log(prog)
                    .map(|l| l.iter().map(|&c| c as u8).collect::<Vec<u8>>())
                    .map(|l| String::from_utf8_lossy(&l).into_owned())
                    .unwrap_or_default();
                DgError::Nvrtc(format!("NVRTC compile ({arch}): {e:?}\n{log}"))
            })?;
        let ptx = nvrtc_result::get_ptx(prog)
            .map_err(|e| DgError::Nvrtc(format!("NVRTC get_ptx: {e:?}")))?;
        let _ = nvrtc_result::destroy_program(prog);
        // Strip trailing NUL.
        let mut ptx: Vec<u8> = ptx.iter().map(|&c| c as u8).collect();
        while ptx.last() == Some(&0) {
            ptx.pop();
        }
        Ok(String::from_utf8_lossy(&ptx).into_owned())
    }
}

/// Get (or compile+load) a kernel. `kernel_src` is the kernel translation
/// unit (see [`kernel_src`]); `body` is the instantiation wrapper appended
/// after it. `signature` uniquely names the compiled variant.
pub fn get_kernel(dev: &Device, kernel_src: &str, tag: &str, signature: &str, body: &str) -> DgResult<sys::Func> {
    let cache_key = format!("{tag}/{signature}");
    {
        let st = state().lock().unwrap();
        if let Some(k) = st.kernels.get(&cache_key) {
            return Ok(k.0);
        }
    }

    let source = format!("{PRELUDE}\n{kernel_src}\n{body}");
    let arch = dev.arch.nvrtc_arch();
    let ptx = compile_ptx_cached(&source, arch, tag)?;
    let c_ptx = CString::new(ptx).map_err(|e| DgError::Nvrtc(e.to_string()))?;

    dev.bind()?;
    let module = sys::module_load_data(c_ptx.as_ptr() as *const _)?;
    let func = sys::module_get_function(module, "__dg_kernel")?;

    let mut st = state().lock().unwrap();
    st.modules.insert(cache_key.clone(), SendPtr(module));
    st.kernels.insert(cache_key.clone(), SendPtr(func));
    Ok(func)
}

/// Test-compile a source (used by the `smoke` bench subcommand / e2e tests).
pub fn smoke_compile(dev: &Device, kernel_src: &str, tag: &str, body: &str) -> DgResult<()> {
    let source = format!("{PRELUDE}\n{kernel_src}\n{body}");
    compile_ptx_cached(&source, dev.arch.nvrtc_arch(), tag)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Launch plumbing
// ---------------------------------------------------------------------------

/// A pending kernel argument list; pointers are passed by value.
pub struct Args {
    raw: Vec<Vec<u8>>,
    ptrs: Vec<*const std::ffi::c_void>,
}

impl Args {
    pub fn new() -> Args {
        Args { raw: Vec::new(), ptrs: Vec::new() }
    }

    pub fn u32(mut self, v: u32) -> Self {
        self.raw.push(v.to_ne_bytes().to_vec());
        self
    }
    pub fn i32(mut self, v: i32) -> Self {
        self.raw.push(v.to_ne_bytes().to_vec());
        self
    }
    pub fn f32(mut self, v: f32) -> Self {
        self.raw.push(v.to_ne_bytes().to_vec());
        self
    }
    pub fn u64(mut self, v: u64) -> Self {
        self.raw.push(v.to_ne_bytes().to_vec());
        self
    }
    /// Device pointer (passed by value as CUdeviceptr = u64).
    pub fn devptr(mut self, p: sys::DevicePtr) -> Self {
        self.raw.push(p.to_ne_bytes().to_vec());
        self
    }
    /// Raw pointer (e.g. a global-memory address to be dereferenced in-kernel).
    pub fn ptr<T>(mut self, p: *const T) -> Self {
        self.raw.push((p as u64).to_ne_bytes().to_vec());
        self
    }
    /// TMA descriptor passed by value (128 bytes, 64-byte aligned).
    pub fn tensormap(mut self, m: &sys::TensorMap) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(m as *const sys::TensorMap as *const u8, 128)
        };
        self.raw.push(bytes.to_vec());
        self
    }

    fn finish(mut self) -> Vec<*const std::ffi::c_void> {
        for r in &self.raw {
            self.ptrs.push(r.as_ptr() as *const _);
        }
        self.ptrs
    }
}

/// Launch a kernel with optional cluster dims and large dynamic shared memory.
pub fn launch(
    dev: &Device,
    func: sys::Func,
    stream: sys::Stream,
    cfg: &sys::LaunchEx,
    args: Args,
) -> DgResult<()> {
    dev.bind()?;
    if cfg.smem > 48 * 1024 {
        sys::func_set_attribute(func, sys::FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES, cfg.smem as i32)?;
    }
    let params = args.finish();
    unsafe { sys::launch_kernel_ex(func, cfg, stream, &params) }
}
