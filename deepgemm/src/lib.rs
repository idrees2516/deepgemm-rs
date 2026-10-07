//! # DeepGEMM-RS
//!
//! A Rust port of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) —
//! DeepSeek's high-performance tensor-core kernel library — targeting the
//! inference stack of DeepSeek-V4-class and MiMo-class MoE models on NVIDIA
//! Blackwell (SM100: B200/GB200) GPUs.
//!
//! * Unified FP8/FP4/BF16 GEMM (`tcgen05.mma` `kind::mxf8f6f4` /
//!   `kind::mxf4` / `kind::f16`) with UE8M0 block scales, TMA pipelines,
//!   2-CTA cluster MMA, swap-AB, persistent L2-swizzled scheduling.
//! * MoE m-grouped GEMMs (contiguous + masked), batched GEMM.
//! * MQA logits (weighted-ReLU scoring for the MLA lightning indexer).
//! * `transform_sf` and MX quantization utilities.
//!
//! All kernels are JIT-compiled at runtime with NVRTC (DeepGEMM's DeepJIT
//! model) — no CUDA toolkit is needed at build time; the CUDA driver and
//! NVRTC libraries are loaded at runtime. Results are cached on disk.
//!
//! Requires CUDA 12.8+ (SM100a `tcgen05` + `16U4` TMA types).
//!
//! ```no_run
//! use deepgemm::prelude::*;
//! let dev = Device::new(0).unwrap();
//! let stream = DevStream::new(&dev).unwrap();
//! // ... allocate operands, call fp8_gemm_nt / fp4_gemm_nt / ...
//! ```

// FFI-adjacent lints: raw driver handles cross the boundary by design, and
// `% b == 0` is kept instead of `is_multiple_of` (MSRV 1.75).
#![allow(clippy::manual_is_multiple_of)]

pub mod api;
pub mod device;
pub mod error;
pub mod golden;
pub mod heuristics;
pub mod jit;
pub mod sys;
pub mod tma;
pub mod types;

pub mod prelude {
    pub use crate::api::*;
    pub use crate::device::{alloc_and_upload, download, Arch, DevBuffer, DevStream, Device};
    pub use crate::error::{DgError, DgResult};
    pub use crate::golden;
    pub use crate::types::{Dtype, GemmType, Major, Operand, Output, SfGran, SfTensor};
}
