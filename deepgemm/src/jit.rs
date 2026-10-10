//! JIT engine: assembles kernel sources, compiles them with NVRTC (cached in
//! memory and on disk), loads modules, and launches via `cuLaunchKernelEx`
//! with cluster dims / dynamic smem / optional PDL.

// Launch plumbing hands raw driver handles (CUfunction/CUstream) into the
// FFI boundary; and `% b == 0` is kept for MSRV 1.75 compatibility.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::manual_is_multiple_of)]

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
    pub const MQA_SM90: &str = include_str!("../kernels/mqa_logits_sm90.cu");
    pub const SPARSE_MQA: &str = include_str!("../kernels/sparse_mqa_sm100.cu");
    pub const HC_PRENORM: &str = include_str!("../kernels/hc_prenorm.cu");
    pub const GEMM_SM90_1D2D: &str = include_str!("../kernels/gemm_sm90_1d2d.cu");
    pub const WGMMA_H: &str = include_str!("../kernels/wgmma.h");
    pub const GEMM_SM90_CU: &str = include_str!("../kernels/gemm_sm90.cu");

    /// SM100 unit: prelude (auto-prepended by get_kernel) + gemm_sm100.cu.
    pub fn sm100_unit() -> &'static str {
        GEMM_SM100
    }

    /// SM90 unit: prelude (auto-prepended) + wgmma.h + gemm_sm90.cu.
    /// wgmma.h must come first: gemm_sm90.cu's descriptor builders and both
    /// kernels reference the wgmma layer.
    pub fn sm90_unit() -> &'static str {
        static U: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        U.get_or_init(|| format!("{WGMMA_H}\n{GEMM_SM90_CU}"))
            .as_str()
    }
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
    S.get_or_init(|| {
        Mutex::new(JitState {
            modules: HashMap::new(),
            kernels: HashMap::new(),
        })
    })
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
        nvrtc_result::compile_program(prog, &options).map_err(|e| {
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
pub fn get_kernel(
    dev: &Device,
    kernel_src: &str,
    tag: &str,
    signature: &str,
    body: &str,
) -> DgResult<sys::Func> {
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
// Offline (GPU-less) compile checks
//
// NVRTC is a pure compiler: `nvrtcCompileProgram` + `nvrtcGetPTX` (and even
// `nvrtcGetCUBIN`, which runs the built-in ptxas backend) never touch the
// driver or a device. That means the *entire* kernel suite can be validated
// on a machine without a GPU — a CI sandbox, a laptop, an H100 box — as long
// as libnvrtc can be loaded. This is the "run it in this sandbox through
// anyway" path:
//
//   PTX  pass: -arch=compute_100a  — exactly what the runtime JIT does
//              (identical option string), so a green check here means the
//              on-device `smoke` will also compile.
//   CUBIN pass: -arch=sm_100a      — additionally runs the SASS backend,
//              catching instruction-level issues (register pressure is
//              still a runtime property, but encoding/selector bugs are
//              caught here).
// ---------------------------------------------------------------------------

/// Result of one offline compile check.
#[derive(Debug, Clone)]
pub struct CompileCheckResult {
    /// Size of the generated PTX text, bytes.
    pub ptx_len: usize,
    /// Size of the generated CUBIN (SASS), bytes, when the NVRTC build
    /// supports `nvrtcGetCUBIN` (CUDA >= 11.8) — `None` otherwise.
    pub cubin_len: Option<usize>,
}

/// Make `libnvrtc.so.12` loadable *without* a CUDA installation:
/// 1. If the plain dlopen already works (LD_LIBRARY_PATH, system CUDA),
///    nothing to do — cudarc's `sys::lib()` will find it.
/// 2. Otherwise probe common locations and `dlopen(..., RTLD_GLOBAL)` the
///    absolute path; glibc then resolves later by-soname lookups to the
///    already-loaded object.
/// 3. `DG_NVRTC_PATH` may point at a directory (or exact .so) to use.
pub fn ensure_nvrtc() -> DgResult<()> {
    // Fast path: does by-name dlopen already succeed?
    if probe_dlopen(None) {
        return Ok(());
    }
    let mut candidates: Vec<String> = Vec::new();
    if let Some(p) = std::env::var_os("DG_NVRTC_PATH") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            candidates.push(p.join("libnvrtc.so.12").to_string_lossy().into_owned());
            candidates.push(p.join("libnvrtc.so").to_string_lossy().into_owned());
        } else {
            candidates.push(p.to_string_lossy().into_owned());
        }
    }
    // venv roots (VIRTUAL_ENV, ~/.venv): walk lib/python*/site-packages.
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let venvs: Vec<PathBuf> = std::env::var_os("VIRTUAL_ENV")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    for v in venvs {
        push_pip_libnvrtc(&v, &mut candidates);
    }
    push_pip_libnvrtc(&home.join(".venv"), &mut candidates);
    push_pip_libnvrtc(&home.join("miniconda3"), &mut candidates);
    // System CUDA installs.
    candidates.push("/usr/local/cuda/lib64/libnvrtc.so.12".into());
    candidates.push("/usr/lib/x86_64-linux-gnu/libnvrtc.so.12".into());
    for c in &candidates {
        if probe_dlopen(Some(c)) {
            return Ok(());
        }
    }
    // Also try without any hint — maybe it appeared meanwhile.
    if probe_dlopen(None) {
        return Ok(());
    }
    Err(DgError::Nvrtc(
        "libnvrtc.so.12 not loadable. Install it (e.g. `pip install nvidia-cuda-nvrtc-cu12`), \
         then either set LD_LIBRARY_PATH to its lib dir or DG_NVRTC_PATH to the .so path"
            .into(),
    ))
}

/// Append `lib/pythonX/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12`
/// candidates for a venv/conda root, if present.
fn push_pip_libnvrtc(root: &std::path::Path, out: &mut Vec<String>) {
    let lib = root.join("lib");
    let Ok(entries) = std::fs::read_dir(&lib) else {
        return;
    };
    for e in entries.flatten() {
        let sp = e
            .path()
            .join("site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12");
        if sp.is_file() {
            out.push(sp.to_string_lossy().into_owned());
        }
    }
}

/// dlopen probe. `None` = by soname only.
fn probe_dlopen(path: Option<&str>) -> bool {
    unsafe extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut std::ffi::c_void;
    }
    const RTLD_NOW: i32 = 2;
    const RTLD_GLOBAL: i32 = 0x100;
    let c = match path {
        Some(p) => match std::ffi::CString::new(p) {
            Ok(c) => c,
            Err(_) => return false,
        },
        None => std::ffi::CString::new("libnvrtc.so.12").unwrap(),
    };
    unsafe { !dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL).is_null() }
}

/// Offline-compile one kernel variant for `arch` (e.g. "100a"):
/// PTX (compute_ arch, the runtime path) + CUBIN (sm_ arch, SASS backend).
pub fn compile_check_kernel(
    source_tu: &str,
    body: &str,
    arch: &str,
    tag: &str,
) -> DgResult<CompileCheckResult> {
    ensure_nvrtc()?;
    let source = format!("{PRELUDE}\n{source_tu}\n{body}");
    let ptx = compile_ptx_cached(&source, arch, tag)?;
    let cubin_len = compile_cubin(&source, arch, tag)
        .map_err(|e| {
            // PTX passed but ptxas (SASS) failed: surface the reason —
            // a kernel that cannot reach SASS will not run on hardware.
            eprintln!("warning: SASS generation failed for {tag} (arch {arch}): {e}");
            e
        })
        .ok()
        .map(|v| v.len());
    Ok(CompileCheckResult {
        ptx_len: ptx.len(),
        cubin_len,
    })
}

/// Compile to CUBIN (SASS) via `--gpu-architecture=sm_<arch>`.
/// Uses `nvrtcGetCUBIN`/`nvrtcGetCUBINSize` from the raw binding layer.
fn compile_cubin(source: &str, arch: &str, tag: &str) -> DgResult<Vec<u8>> {
    let key = format!("{:016x}", fnv1a(&format!("{source}\x00sm{arch}\x00{tag}")));
    if let Some(dir) = cache_dir() {
        let path = dir.join(format!("{key}.{arch}.cubin"));
        if let Ok(c) = std::fs::read(&path) {
            return Ok(c);
        }
        match compile_cubin_nocache(source, arch) {
            Ok(c) => {
                let _ = std::fs::write(&path, &c);
                Ok(c)
            }
            Err(e) => Err(e),
        }
    } else {
        compile_cubin_nocache(source, arch)
    }
}

fn compile_cubin_nocache(source: &str, arch: &str) -> DgResult<Vec<u8>> {
    let prog = nvrtc_result::create_program(source, None)
        .map_err(|e| DgError::Nvrtc(format!("NVRTC create: {e:?}")))?;
    let options: Vec<Vec<u8>> = vec![
        format!("--gpu-architecture=sm_{arch}").into_bytes(),
        b"--std=c++17".to_vec(),
    ];
    unsafe {
        nvrtc_result::compile_program(prog, &options).map_err(|e| {
            let log = nvrtc_result::get_program_log(prog)
                .map(|l| l.iter().map(|&c| c as u8).collect::<Vec<u8>>())
                .map(|l| String::from_utf8_lossy(&l).into_owned())
                .unwrap_or_default();
            DgError::Nvrtc(format!("NVRTC cubin ({arch}): {e:?}\n{log}"))
        })?;
        let lib = cudarc::nvrtc::sys::lib();
        use cudarc::nvrtc::sys::nvrtcResult;
        let mut size = 0usize;
        let r = lib.nvrtcGetCUBINSize(prog, &mut size);
        if r != nvrtcResult::NVRTC_SUCCESS {
            let _ = nvrtc_result::destroy_program(prog);
            return Err(DgError::Nvrtc(format!(
                "nvrtcGetCUBINSize({arch}) -> {r:?}"
            )));
        }
        let mut buf = vec![0u8; size.max(1)];
        let r2 = lib.nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut std::ffi::c_char);
        let _ = nvrtc_result::destroy_program(prog);
        if r2 != nvrtcResult::NVRTC_SUCCESS {
            return Err(DgError::Nvrtc(format!("nvrtcGetCUBIN({arch}) -> {r2:?}")));
        }
        buf.truncate(size);
        Ok(buf)
    }
}

/// NVRTC runtime version as "(major, minor)", for diagnostics.
pub fn nvrtc_version() -> DgResult<(i32, i32)> {
    ensure_nvrtc()?;
    unsafe {
        let lib = cudarc::nvrtc::sys::lib();
        use cudarc::nvrtc::sys::nvrtcResult;
        let (mut maj, mut min) = (0i32, 0i32);
        let r = lib.nvrtcVersion(&mut maj, &mut min);
        if r != nvrtcResult::NVRTC_SUCCESS {
            return Err(DgError::Nvrtc(format!("nvrtcVersion -> {r:?}")));
        }
        Ok((maj, min))
    }
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
        Args {
            raw: Vec::new(),
            ptrs: Vec::new(),
        }
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
        let bytes =
            unsafe { std::slice::from_raw_parts(m as *const sys::TensorMap as *const u8, 128) };
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

impl Default for Args {
    fn default() -> Self {
        Self::new()
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
        sys::func_set_attribute(
            func,
            sys::FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            cfg.smem as i32,
        )?;
    }
    let params = args.finish();
    unsafe { sys::launch_kernel_ex(func, cfg, stream, &params) }
}
