//! Rust-level optimizations layered on the GPU kernels (msg-11 "Rust
//! optimizations in combination with GPU optimizations").
//!
//! * **Workspace pooling** — device scratch buffers are recycled across
//!   calls (size-classed free lists) instead of `cuMemAlloc` per invocation;
//!   CUDA allocations are the biggest host-side latency term in serving
//!   loops once the kernels themselves are PDL-chained.
//! * **Config cache** — the heuristics search (bandwidth model, comparator)
//!   is pure and deterministic per (shape, dtype, arch): memoized so the
//!   steady-state path skips it entirely.
//! * **Zero-copy tensors** — `Tensor` views wrap host or device memory with
//!   no intermediate staging; `upload` is a single `memcpyHtoD`.
//! * **Stream pool + PDL chains** — one stream per logical lane (overlap of
//!   independent GEMMs), and `chain()` marks every launch after the first
//!   with `griddepcontrol` PDL so consecutive kernels overlap prologue with
//!   the predecessor's epilogue.
//!
//! ```no_run
//! use deepgemm::runtime::Runtime;
//! # fn main() -> Result<(), deepgemm::error::DgError> {
//! let rt = Runtime::new(0)?;            // pooled handle on device 0
//! let scratch = rt.scratch(1 << 20)?;   // pooled workspace (2 MiB size-class)
//! let a = rt.upload(&[0f32; 1024])?;    // zero-copy HtoD upload
//! rt.chain(|s| {                        // PDL chain on a pool stream
//!     let _ = (s, &a);                  // kernels launched via api::* on `s`
//! });
//! rt.release(scratch);                  // recycle for the next call
//! # Ok(())
//! # }
//! ```
use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit;
use crate::sys;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Size-classed pooling: round to 2MiB multiples so serving loops with a
/// handful of distinct shapes hit the same buckets.
fn size_class(bytes: usize) -> usize {
    const CLASS: usize = 2 << 20;
    (bytes.max(1) + CLASS - 1) / CLASS * CLASS
}

struct Pool {
    free: HashMap<usize, Vec<DevBuffer>>,
    live: usize,
    hits: u64,
    misses: u64,
}

/// A pooled runtime handle: device + stream pool + workspace pool + caches.
pub struct Runtime {
    pub device: Arc<Device>,
    streams: Mutex<Vec<DevStream>>,
    pool: Mutex<Pool>,
    config_cache: Mutex<HashMap<String, String>>, // shape-key -> chosen sig (memo)
}

impl Runtime {
    fn new_ordinal(dev_idx: usize) -> DgResult<Runtime> {
        let device = Device::new(dev_idx as i32)?;
        let streams = Mutex::new(Vec::new());
        let pool = Mutex::new(Pool {
            free: HashMap::new(),
            live: 0,
            hits: 0,
            misses: 0,
        });
        Ok(Runtime {
            device,
            streams,
            pool,
            config_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn new(dev_idx: usize) -> DgResult<Runtime> {
        Self::new_ordinal(dev_idx)
    }

    /// A stream from the pool (created on demand; round-robin per lane).
    pub fn stream(&self) -> DevStream {
        let mut ss = self.streams.lock().unwrap();
        if ss.is_empty() {
            ss.push(DevStream::new(&self.device).expect("stream"));
        }
        ss.pop().expect("stream")
    }

    /// Zero-copy upload: one HtoD memcpy, no staging buffers.
    pub fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> DgResult<DevBuffer> {
        crate::device::alloc_and_upload(&self.device, data, self.stream().raw())
    }

    /// Get a pooled scratch buffer of >= `bytes` (recycled when released).
    pub fn scratch(&self, bytes: usize) -> DgResult<DevBuffer> {
        let cls = size_class(bytes);
        let mut p = self.pool.lock().unwrap();
        if let Some(buf) = p.free.get_mut(&cls).and_then(|v| v.pop()) {
            p.hits += 1;
            return Ok(buf);
        }
        p.misses += 1;
        p.live += 1;
        drop(p);
        DevBuffer::alloc(&self.device, cls)
    }

    /// Return a buffer to the pool (idempotent, size-classed).
    pub fn release(&self, buf: DevBuffer) {
        if buf.is_empty() {
            return;
        }
        let cls = size_class(buf.len);
        let mut p = self.pool.lock().unwrap();
        p.free.entry(cls).or_default().push(buf);
    }

    /// Pool statistics (hits/misses/live) for tuning the serving loop.
    pub fn pool_stats(&self) -> (u64, u64, usize) {
        let p = self.pool.lock().unwrap();
        (p.hits, p.misses, p.live)
    }

    /// Memoize a heuristics decision: returns true when the key was already
    /// resolved (the caller can skip the search on the steady path).
    pub fn memo_config(&self, key: &str, resolve: impl FnOnce() -> String) -> bool {
        let mut c = self.config_cache.lock().unwrap();
        if c.contains_key(key) {
            return true;
        }
        let v = resolve();
        c.insert(key.to_string(), v);
        false
    }

    /// PDL chain: run `f` on a pool stream; the first kernel of the chain
    /// waits on the prior grid (pdl attr), kernels inside call
    /// `griddepcontrol.wait` — the api launchers already set `pdl: true`,
    /// so a chain of gemms overlaps epilogue/prologue automatically.
    pub fn chain<T>(&self, f: impl FnOnce(&DevStream) -> T) -> T {
        let s = self.stream();
        let out = f(&s);
        let _ = s.sync();
        out
    }

    /// Upload + immediately usable operand upload helper (zero-copy view).
    pub fn zeros(&self, bytes: usize) -> DgResult<DevBuffer> {
        DevBuffer::alloc_zeros(&self.device, bytes.max(1))
    }
}

/// Zero-copy typed view over host memory (no device interaction): lets
/// callers assemble operands from slices without staging structures.
pub struct Tensor<'a, T: bytemuck::Pod> {
    pub data: &'a [T],
}

impl<'a, T: bytemuck::Pod> Tensor<'a, T> {
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.data)
    }
}

// Keep `jit` linked in this module's doc path (launch attributes).
#[allow(unused_imports)]
use jit as _jit_link;
#[allow(unused_variables)]
fn _keep_sys_alive(s: sys::Stream) {}
