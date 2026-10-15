//! Benchmark CLI for deepgemm-rs.
//!
//! Usage examples (on a B200 with CUDA 12.8+):
//! ```text
//! export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
//! cargo run -p deepgemm-bench --release -- bench fp4_nt_native --m 8192 --n 8192 --k 7168
//! cargo run -p deepgemm-bench --release -- bench fp8_nt --m 8192 --n 4096 --k 7168
//! cargo run -p deepgemm-bench --release -- bench fp8_m_grouped_contiguous --groups 128
//! cargo run -p deepgemm-bench --release -- smoke     # JIT-compile all kernels
//! cargo run -p deepgemm-bench --release -- list      # list benchmarks
//! ```
//!
//! **Without a GPU** (sandbox / CI / laptop): NVRTC is a pure compiler, so
//! the full kernel suite can still be validated offline — PTX *and* SASS:
//! ```text
//! pip install nvidia-cuda-nvrtc-cu12
//! cargo run -p deepgemm-bench --release -- compile-check --arch 100a
//! ```
//! If libnvrtc is not on the default loader path, the binary probes pip
//! wheel locations automatically, or set `DG_NVRTC_PATH=/path/to/libnvrtc.so.12`.
//!
//! Tiling overrides (tune on real hardware, then hard-code into heuristics):
//! `DG_BLOCK_M`, `DG_BLOCK_N`, `DG_BLOCK_K`, `DG_CLUSTER_M`, `DG_CLUSTER_N`,
//! `DG_SWAP_AB`, `DG_NUM_STAGES`, `DG_PRINT_CONFIGS=1`.

use clap::{Parser, Subcommand};
use deepgemm::jit::{self, kernel_src};
use deepgemm::prelude::*;
use deepgemm::sm90;

#[derive(Parser)]
#[command(
    name = "deepgemm-bench",
    about = "DeepGEMM-RS benchmarks (TFLOPS, % of peak)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List available benchmarks.
    List,
    /// JIT-compile every kernel variant for this GPU (fast correctness gate).
    Smoke,
    /// Offline-compile every kernel variant WITHOUT a GPU (PTX + SASS via NVRTC).
    /// Needs only libnvrtc (e.g. `pip install nvidia-cuda-nvrtc-cu12`).
    CompileCheck {
        /// Target arch: 100a (B200), 103a (B300), 120a (SM120). [default: 100a]
        #[arg(long)]
        arch: Option<String>,
    },
    /// Run a benchmark.
    Bench {
        name: String,
        #[arg(long, default_value = "8192")]
        m: u32,
        #[arg(long, default_value = "8192")]
        n: u32,
        #[arg(long, default_value = "7168")]
        k: u32,
        #[arg(long, default_value = "128")]
        groups: u32,
        #[arg(long, default_value = "20")]
        iters: u32,
        #[arg(long, default_value = "5")]
        warmup: u32,
    },
    /// Sweep a family (or all) over the production shape grid; prints a
    /// TFLOPS/%-peak/roofline table and writes a JSON snapshot for
    /// `compare`. The B200/B300 acceptance workflow: `sweep` then `verify`.
    Sweep {
        /// Family: all | gemm | layouts | moe | bmm | kgrouped | attention | mega
        #[arg(long, default_value = "all")]
        family: String,
        /// JSON snapshot path (default: bench-sweep.json).
        #[arg(long, default_value = "bench-sweep.json")]
        out: String,
        #[arg(long, default_value = "10")]
        iters: u32,
    },
    /// The on-GPU correctness gate: every family's shape grid vs the CPU
    /// golden model, with the upstream tolerance table. Nonzero exit on any
    /// failure — run this before trusting any `sweep` numbers.
    Verify {
        #[arg(long, default_value = "all")]
        family: String,
    },
    /// Autotune a shape over the DG_* tiling overrides: times each candidate
    /// configuration and prints the winner as an env line you can export.
    Autotune {
        name: String,
        #[arg(long, default_value = "8192")]
        m: u32,
        #[arg(long, default_value = "8192")]
        n: u32,
        #[arg(long, default_value = "7168")]
        k: u32,
        #[arg(long, default_value = "128")]
        groups: u32,
        #[arg(long, default_value = "10")]
        iters: u32,
    },
    /// Compare two `sweep` JSON snapshots (regression tracking).
    Compare { old: String, new: String },
}

const BENCHES: &[&str] = &[
    "sm90_fp8_nt",
    "sm90_fp8_kk",
    "sm90_bf16_nt",
    "sm90_bf16_m_grouped_masked",
    "fp4_nt_native",
    "fp8_nt",
    "fp8_nt_g128",
    "bf16_nt",
    "fp8_m_grouped_contiguous",
    "fp8_m_grouped_masked",
    "fp4_m_grouped_contiguous",
    "fp8_bmm",
    "mqa_logits_fp8",
];

fn host_bf16_bits(v: f32) -> u16 {
    // Round-to-nearest-even f32 -> bf16.
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fff + lsb;
    (((bits + rounding) >> 16) as u16) & 0x7fff | ((bits >> 31) as u16) << 15
}

fn make_sf(
    dev: &Device,
    stream: &DevStream,
    m: u32,
    k: u32,
    gran: SfGran,
    seed: u32,
) -> DgResult<SfTensor> {
    // Deterministic power-of-two scales (UE8M0-compatible).
    let sf_k = (k / gran.k()) as usize;
    let scales: Vec<f32> = (0..(m as usize) * sf_k)
        .map(|i| {
            let e = -3 + ((i.wrapping_mul(2654435761) ^ seed as usize) % 7) as i32;
            (2f32).powi(e)
        })
        .collect();
    transform_sf(dev, stream, &scales, m, gran)
}

/// Paged MQA bench: decode-shaped — many short requests over a paged KV
/// cache; measures effective tokens/s (the serving metric) as well as TFLOPS.
fn run_mqa_paged_bench(dev: &Device, stream: &DevStream, iters: u32, warmup: u32) -> DgResult<()> {
    use deepgemm::device::alloc_and_upload;

    let num_requests = 512u32;
    let ctx_len = 512u32;
    let num_tokens = num_requests; // 1 token per request (pure decode)
    let heads = 64u32;
    let head_dim = 128u32;
    let page_kv = 64u32;
    let num_pages_per_req = ctx_len / page_kv;
    let num_pages = num_requests * num_pages_per_req;
    let q_rows = num_tokens * heads;
    let kv_gran = SfGran::G32;

    let q_data = DevBuffer::alloc_zeros(dev, (q_rows * head_dim) as usize)?;
    let kv_pages = DevBuffer::alloc_zeros(dev, (num_pages * page_kv * head_dim) as usize)?;
    let q_sf = make_sf(dev, stream, q_rows, head_dim, kv_gran, 3)?;
    // Paged SF: one int32 word per token, page-major [num_pages, page_kv].
    let kv_sf_pages = DevBuffer::alloc_zeros(dev, (num_pages * page_kv) as usize * 4)?;
    let weights = DevBuffer::alloc_zeros(dev, (num_tokens * heads) as usize * 2)?;
    let context_lens: Vec<u32> = vec![ctx_len; num_tokens as usize];
    let indices: Vec<u32> = (0..num_tokens).collect::<Vec<u32>>(); // 1 token/req
                                                                   // Per-TOKEN rows: [num_tokens, num_pages_per_req], identity page ids.
    let block_table: Vec<u32> = (0..num_tokens * num_pages_per_req).collect::<Vec<u32>>();
    let mut out = DevBuffer::alloc_zeros(dev, (num_tokens * ctx_len) as usize * 2)?;

    let q = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: q_rows,
        k: head_dim,
        outer_stride: head_dim,
        sf: None,
        data: q_data,
    };
    let cl = alloc_and_upload(dev, &context_lens, stream.raw())?;
    let idx = alloc_and_upload(dev, &indices, stream.raw())?;
    let bt = alloc_and_upload(dev, &block_table, stream.raw())?;

    let run_once = |out: &mut DevBuffer| -> DgResult<()> {
        deepgemm::api::mqa_logits_paged(
            dev,
            stream,
            &q,
            &q_sf,
            &kv_pages,
            &kv_sf_pages,
            &weights,
            page_kv,
            num_pages,
            &cl,
            &idx,
            &bt,
            num_pages_per_req,
            num_tokens,
            heads,
            head_dim,
            out,
            ctx_len,
        )
    };
    for _ in 0..warmup {
        run_once(&mut out)?;
    }
    stream.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        run_once(&mut out)?;
    }
    stream.sync()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let tokens = num_tokens as f64;
    let flops = 2.0 * tokens * ctx_len as f64 * heads as f64 * head_dim as f64;
    let tflops = flops / elapsed / (iters as f64) / 1e12;
    println!(
        "mqa_logits_paged_fp8       requests={num_requests} ctx={ctx_len} pages={num_pages} | {tflops:8.1} TFLOPS, {:.0} tokens/s  {elapsed:.3}s/{iters} iters",
        tokens * iters as f64 / elapsed
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_bench(
    dev: &Device,
    stream: &DevStream,
    name: &str,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    iters: u32,
    warmup: u32,
) -> DgResult<()> {
    let (a_dt, b_dt, gran) = match name {
        "fp4_nt_native" | "fp4_m_grouped_contiguous" => (Dtype::Fp4, Dtype::Fp4, SfGran::G32),
        "bf16_nt" | "sm90_bf16_nt" | "sm90_bf16_m_grouped_masked" => {
            (Dtype::Bf16, Dtype::Bf16, SfGran::G32)
        }
        "fp8_nt_g128" => (Dtype::Fp8, Dtype::Fp8, SfGran::G128),
        _ => (Dtype::Fp8, Dtype::Fp8, SfGran::G32),
    };

    // Data: random-ish bytes (valid fp8/fp4 codes; performance does not depend
    // on values for these dtypes).
    let elem_bytes_a = if a_dt == Dtype::Fp4 {
        (m * k / 2) as usize
    } else {
        (m * k) as usize
    };
    let a_data = DevBuffer::alloc_zeros(dev, elem_bytes_a.max(16))?;
    let elem_bytes_b = if b_dt == Dtype::Fp4 {
        (n * k / 2) as usize
    } else {
        (n * k) as usize
    } * groups as usize;
    let b_data = DevBuffer::alloc_zeros(dev, elem_bytes_b.max(16))?;
    let sf_a = make_sf(dev, stream, m, k, gran, 1)?;
    let sf_b = make_sf(dev, stream, n, k, gran, 2)?;
    // Host FP32 (mn, k/128) scales for the SM90 1D1D benches.
    let sfa_host: Vec<f32> = (0..(m as usize) * (k as usize / 128))
        .map(|i| 2f32.powi(-3 + (i.wrapping_mul(2654435761) % 7) as i32))
        .collect();
    let sfb_host: Vec<f32> = (0..(n as usize) * (k as usize / 128))
        .map(|i| 2f32.powi(-3 + (i.wrapping_mul(40503) % 7) as i32))
        .collect();

    let a = Operand {
        dtype: a_dt,
        major: Major::K,
        rows: m,
        k,
        outer_stride: k,
        sf: Some(sf_a),
        data: a_data,
    };
    let b = Operand {
        dtype: b_dt,
        major: Major::K,
        rows: n,
        k,
        outer_stride: k,
        sf: Some(sf_b),
        data: b_data,
    };

    let cd_bytes = (m * n * 2) as usize
        * if name.contains("grouped") || name.contains("bmm") {
            groups as usize
        } else {
            1
        };
    let out_data = DevBuffer::alloc_zeros(dev, cd_bytes.max(16))?;
    let mut out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: out_data,
    };

    let run_once = |out: &mut Output| -> DgResult<()> {
        match name {
            "fp8_nt" | "fp4_nt_native" | "fp8_nt_g128" | "bf16_nt" => {
                gemm_nt(dev, stream, &a, &b, out, false)
            }
            "fp8_m_grouped_contiguous" | "fp4_m_grouped_contiguous" => {
                let alignment = 128;
                let m_aligned = m.div_ceil(alignment) * alignment;
                let mut m_indices = vec![0i32; (m_aligned / alignment) as usize];
                for (i, v) in m_indices.iter_mut().enumerate() {
                    *v = (i as i32 % groups as i32).max(0);
                }
                let m_idx = alloc_and_upload(dev, &m_indices, stream.raw())?;
                let a2 = Operand {
                    dtype: a_dt,
                    major: Major::K,
                    rows: m_aligned,
                    k,
                    outer_stride: k,
                    sf: Some(make_sf(dev, stream, m_aligned, k, gran, 1)?),
                    data: DevBuffer::alloc_zeros(dev, elem_bytes_a.max(16))?,
                };
                m_grouped_gemm_nt_contiguous(dev, stream, &a2, &b, out, &m_idx, groups)
            }
            "fp8_m_grouped_masked" => {
                let masked = vec![m / groups; groups as usize];
                let masked_buf = alloc_and_upload(dev, &masked, stream.raw())?;
                m_grouped_gemm_nt_masked(dev, stream, &a, &b, out, &masked_buf, groups, m)
            }
            "fp8_bmm" => fp8_bmm(dev, stream, &a, &b, out, groups, false),
            "sm90_fp8_nt" => {
                let sfa = sm90::sf_fp32_from_host(dev, stream, &sfa_host, m, k / 128)?;
                let sfb = sm90::sf_fp32_from_host(dev, stream, &sfb_host, n, k / 128)?;
                let a2 = Operand {
                    dtype: Dtype::Fp8,
                    major: Major::K,
                    rows: m,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (m as usize) * (k as usize))?,
                };
                let b2 = Operand {
                    dtype: Dtype::Fp8,
                    major: Major::K,
                    rows: n,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (n as usize) * (k as usize))?,
                };
                sm90::fp8_gemm_nt(dev, stream, &a2, &sfa, &b2, &sfb, out)
            }
            "sm90_fp8_kk" => {
                // Per-group K sizes: k split into `groups` 128-multiples.
                let per = (k / groups).div_ceil(128) * 128;
                let mut ks: Vec<u32> = vec![per; groups as usize];
                *ks.last_mut().unwrap() = k - per * (groups - 1);
                // Stacked tiles: for each group g, [mn, ks_g] K-major.
                let a2 = Operand {
                    dtype: Dtype::Fp8,
                    major: Major::K,
                    rows: m,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (m as usize) * (k as usize))?,
                };
                let b2 = Operand {
                    dtype: Dtype::Fp8,
                    major: Major::K,
                    rows: n,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (n as usize) * (k as usize))?,
                };
                let sfa = sm90::sf_fp32_from_host(dev, stream, &sfa_host, m, k / 128)?;
                let sfb = sm90::sf_fp32_from_host(dev, stream, &sfb_host, n, k / 128)?;
                sm90::fp8_gemm_kk(dev, stream, &a2, &sfa, &b2, &sfb, &ks, out)
            }
            "sm90_bf16_nt" => {
                let a2 = Operand {
                    dtype: Dtype::Bf16,
                    major: Major::K,
                    rows: m,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (m as usize) * (k as usize) * 2)?,
                };
                let b2 = Operand {
                    dtype: Dtype::Bf16,
                    major: Major::K,
                    rows: n,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (n as usize) * (k as usize) * 2)?,
                };
                sm90::bf16_gemm_nt(dev, stream, &a2, &b2, out, false)
            }
            "sm90_bf16_m_grouped_masked" => {
                let per = m / groups;
                let masked: Vec<i32> = (0..groups).map(|_| per as i32).collect();
                let masked_buf = alloc_and_upload(dev, &masked, stream.raw())?;
                let _per = m / groups;
                let a2 = Operand {
                    dtype: Dtype::Bf16,
                    major: Major::K,
                    rows: m,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (m as usize) * (k as usize) * 2)?,
                };
                let b2 = Operand {
                    dtype: Dtype::Bf16,
                    major: Major::K,
                    rows: n,
                    k,
                    outer_stride: k,
                    sf: None,
                    data: DevBuffer::alloc_zeros(dev, (n as usize) * (k as usize) * 2)?,
                };
                sm90::bf16_gemm_nt_m_grouped_masked(
                    dev,
                    stream,
                    &a2,
                    &b2,
                    &masked_buf,
                    m,
                    groups,
                    out,
                    false,
                )
            }
            _ => Err(DgError::InvalidArg(format!("unknown bench {name}"))),
        }
    };

    // Warmup + timed.
    for _ in 0..warmup {
        run_once(&mut out)?;
    }
    stream.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        run_once(&mut out)?;
    }
    stream.sync()?;
    let elapsed = t0.elapsed().as_secs_f64();

    let group_mult = if name.contains("grouped") || name.contains("bmm") || name == "sm90_fp8_kk" {
        groups as f64
    } else {
        1.0
    };
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * group_mult;
    let tflops = flops / elapsed / (iters as f64) / 1e12;
    let peak = dev.peak_tflops(a_dt);
    println!(
        "{name:<28} m={m:<6} n={n:<6} k={k:<6} groups={groups:<4} | {tflops:8.1} TFLOPS  ({:5.1}% of {peak:.0} TF peak)  {elapsed:.3}s/{iters} iters",
        100.0 * tflops / peak
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_mqa_bench(dev: &Device, stream: &DevStream, iters: u32, warmup: u32) -> DgResult<()> {
    let num_tokens = 4096u32;
    let num_kv = 8192u32;
    let heads = 64u32;
    let head_dim = 128u32;
    let q_rows = num_tokens * heads;
    let kv_gran = SfGran::G32;

    let q_data = DevBuffer::alloc_zeros(dev, (q_rows * head_dim) as usize)?;
    let kv_data = DevBuffer::alloc_zeros(dev, (num_kv * head_dim) as usize)?;
    let q_sf = make_sf(dev, stream, q_rows, head_dim, kv_gran, 3)?;
    let kv_sf = make_sf(dev, stream, num_kv, head_dim, kv_gran, 4)?;

    let q = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: q_rows,
        k: head_dim,
        outer_stride: head_dim,
        sf: None,
        data: q_data,
    };
    let kv = Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows: num_kv,
        k: head_dim,
        outer_stride: head_dim,
        sf: None,
        data: kv_data,
    };

    let weights: Vec<u16> = (0..num_tokens * heads)
        .map(|i| host_bf16_bits(0.5 + (i % 7) as f32 * 0.1))
        .collect();
    let weights = alloc_and_upload(dev, &weights, stream.raw())?;
    let k_start = vec![0u32; num_tokens as usize];
    let k_end = vec![num_kv; num_tokens as usize];
    let ks = alloc_and_upload(dev, &k_start, stream.raw())?;
    let ke = alloc_and_upload(dev, &k_end, stream.raw())?;
    let mut logits = DevBuffer::alloc_zeros(dev, (num_tokens * num_kv * 2) as usize)?;

    let mut run_once = || -> DgResult<()> {
        mqa_logits(
            dev,
            stream,
            &q,
            &q_sf,
            &kv,
            &kv_sf,
            &weights,
            &ks,
            &ke,
            num_tokens,
            num_kv,
            heads,
            head_dim,
            &mut logits,
            num_kv,
        )
    };

    for _ in 0..warmup {
        run_once()?;
    }
    stream.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        run_once()?;
    }
    stream.sync()?;
    let elapsed = t0.elapsed().as_secs_f64();
    let flops = 2.0 * num_tokens as f64 * num_kv as f64 * heads as f64 * head_dim as f64;
    let tflops = flops / elapsed / (iters as f64) / 1e12;
    println!(
        "mqa_logits_fp8              tokens={num_tokens} kv={num_kv} heads={heads} head_dim={head_dim} | {tflops:8.1} TFLOPS  {elapsed:.3}s/{iters} iters"
    );
    Ok(())
}

fn smoke(dev: &Device) -> DgResult<()> {
    println!(
        "Smoke-compiling kernel variants for {} (arch {})...",
        dev.name,
        dev.arch.nvrtc_arch()
    );
    let mut n_ok = 0;
    let variants = kernel_variants(dev.arch.nvrtc_arch());
    for (name, src, body) in &variants {
        match jit::smoke_compile(dev, src, "smoke", body) {
            Ok(()) => {
                println!("  [ok] {name}");
                n_ok += 1;
            }
            Err(e) => println!("  [FAIL] {name}: {e}"),
        }
    }
    println!("{n_ok}/{} variants compiled.", variants.len());
    if n_ok != variants.len() {
        Err(DgError::Nvrtc("smoke compile failures".into()))
    } else {
        Ok(())
    }
}

/// Offline variant of `smoke`: NVRTC compiles PTX (compute_XXXa, the exact
/// runtime path) and CUBIN (sm_XXXa, full SASS backend) with no GPU/driver.
fn compile_check(arch_arg: Option<String>) -> DgResult<()> {
    let arch = arch_arg
        .or_else(|| std::env::var("DG_ARCH").ok())
        .unwrap_or_else(|| "100a".into());
    match arch.as_str() {
        "90a" | "100a" | "103a" | "120a" => {}
        other => {
            return Err(DgError::InvalidArg(format!(
                "unknown arch {other:?}; use 90a / 100a / 103a / 120a"
            )));
        }
    }
    jit::ensure_nvrtc()?;
    let ver = jit::nvrtc_version().unwrap_or((0, 0));
    println!(
        "Offline compile-check (no GPU): arch sm_{arch}, NVRTC {}.{}",
        ver.0, ver.1
    );
    println!("  PTX  pass: --gpu-architecture=compute_{arch} (the runtime JIT path)");
    println!("  CUBIN pass: --gpu-architecture=sm_{arch} (full SASS backend)");
    let t0 = std::time::Instant::now();
    let variants = kernel_variants(&arch);
    let mut n_ok = 0;
    for (name, src, body) in &variants {
        match jit::compile_check_kernel(src, body, &arch, "compile-check") {
            Ok(r) => {
                let cub = r
                    .cubin_len
                    .map(|l| format!("sass {l:>6} B"))
                    .unwrap_or("sass n/a".into());
                println!("  [ok]   {name:<32} ptx {:>7} B  {cub}", r.ptx_len);
                n_ok += 1;
            }
            Err(e) => println!("  [FAIL] {name}: {e}"),
        }
    }
    println!(
        "{n_ok}/{} variants compiled PTX+SASS in {:.1}s (no GPU present).",
        variants.len(),
        t0.elapsed().as_secs_f64()
    );
    if n_ok != variants.len() {
        Err(DgError::Nvrtc("compile-check failures".into()))
    } else {
        Ok(())
    }
}

/// Single source of truth for the representative kernel-variant list shared
/// by `smoke` (on-device JIT) and `compile-check` (offline NVRTC).
/// Covers every translation unit and every code path: mxf4 / mxf8f6f4 / f16
/// MMAs, SF/UTCCP paths, swap-AB, m-grouped (contiguous+masked), batched,
/// MQA logits in FP8 and FP4, quant/dequant/transform_sf.
/// Kernel-variant list for compile checks, filtered by target arch family.
/// Arch-neutral units (layout/quant) compile under every arch; the tcgen05
/// (SM100) and wgmma (SM90) GEMMs are mutually exclusive across these arches
/// (ptxas rejects tcgen05 below sm_100 and wgmma above sm_90).
fn kernel_variants(arch: &str) -> Vec<(&'static str, &'static str, String)> {
    let sm100_family = matches!(arch, "100a" | "103a" | "120a");
    let mut v: Vec<(&'static str, &'static str, String)> = vec![
        (
            "transform_sf k=16",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* sf, unsigned* out, unsigned mn) { dg::transform_sf_impl<128, 64, 16, 16>(sf, out, mn); }"#.to_string(),
        ),
        (
            "transform_sf k=4",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* sf, unsigned* out, unsigned mn) { dg::transform_sf_impl<128, 64, 4, 16>(sf, out, mn); }"#.to_string(),
        ),
        (
            "pack k-grouped sf g4",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* sf, unsigned* out, const int* gl, unsigned mn, unsigned sfk, unsigned psfk, unsigned gk, unsigned ka) { dg::pack_k_grouped_sf_impl<4, 512, 128, 16, false>(sf, out, gl, mn, sfk, psfk, gk, ka); }"#.to_string(),
        ),
        (
            "pack k-grouped sf psum",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* sf, unsigned* out, const int* gl, unsigned mn, unsigned sfk, unsigned psfk, unsigned gk, unsigned ka) { dg::pack_k_grouped_sf_impl<8, 512, 128, 16, true>(sf, out, gl, mn, sfk, psfk, gk, ka); }"#.to_string(),
        ),
        (
            "quant fp8 g32",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* x, unsigned m, unsigned k, unsigned char* d, unsigned* sf) { dg::quant_mx_impl<256, 32, 0>(x, m, k, d, sf); }"#.to_string(),
        ),
        (
            "quant fp8 g128",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* x, unsigned m, unsigned k, unsigned char* d, unsigned* sf) { dg::quant_mx_impl<256, 128, 0>(x, m, k, d, sf); }"#.to_string(),
        ),
        (
            "quant fp4 g32",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* x, unsigned m, unsigned k, unsigned char* d, unsigned* sf) { dg::quant_mx_impl<256, 32, 1>(x, m, k, d, sf); }"#.to_string(),
        ),
        (
            "dequant fp8 g32",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const unsigned char* d, const unsigned* sf, unsigned m, unsigned k, unsigned t, float* o) { dg::dequant_mx_impl<0, 32>(d, sf, m, k, t, o); }"#.to_string(),
        ),
        (
            "quant_out_fp8 rne",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* in, unsigned char* o, int* sfd, unsigned m, unsigned n, unsigned is_, unsigned os_, unsigned ss_, unsigned ta) { dg::quantize_output_fp8_impl<256, 0>(in, o, sfd, m, n, is_, os_, ss_, ta); }"#.to_string(),
        ),
        (
            "quant_out_fp8 sr",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const float* in, unsigned char* o, int* sfd, unsigned m, unsigned n, unsigned is_, unsigned os_, unsigned ss_, unsigned ta) { dg::quantize_output_fp8_impl<256, 1>(in, o, sfd, m, n, is_, os_, ss_, ta); }"#.to_string(),
        ),
        (
            "dequant fp4 g32",
            kernel_src::LAYOUT_QUANT,
            r#"extern "C" __global__ void __dg_kernel(const unsigned char* d, const unsigned* sf, unsigned m, unsigned k, unsigned t, float* o) { dg::dequant_mx_impl<1, 32>(d, sf, m, k, t, o); }"#.to_string(),
        ),
    ];
    if sm100_family {
        v.extend([
            (
                "gemm fp8 nt m128 n256 k128 c2",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 0, 32, 32, 1, 0, 0, 0, 1, 1, 128, 256, 128, 128, 128, 128, 12, 2, 0, 0, 0,
                    0, 148,
                ),
            ),
            (
                "gemm fp4 nt m128 n256 k256 c2",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 0, 32, 32, 1, 1, 5, 5, 1, 1, 128, 256, 256, 128, 128, 256, 8, 2, 0, 0, 0, 0,
                    148,
                ),
            ),
            (
                "gemm bf16 nt m128 n128 k64 c1",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 0, 32, 32, 0, 0, 1, 1, 2, 2, 128, 128, 64, 128, 128, 128, 20, 1, 0, 0, 0, 0,
                    148,
                ),
            ),
            (
                "gemm fp8 swapab masked c2",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    1, 1, 32, 32, 1, 0, 0, 0, 1, 1, 128, 128, 128, 128, 128, 128, 12, 2, 1, 1, 2,
                    0, 148,
                ),
            ),
            (
                "gemm fp8 batched c2",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 0, 32, 32, 1, 0, 0, 0, 1, 1, 128, 128, 128, 128, 128, 128, 12, 2, 0, 0, 4,
                    0, 148,
                ),
            ),
            // BLAS layout parity: MN-major operands (nn/tn/tt route here).
            (
                "gemm fp8 nn (B mn-major)",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 1, 32, 32, 1, 0, 0, 0, 1, 1, 128, 256, 128, 128, 128, 128, 12, 2, 0, 0, 0,
                    0, 148,
                ),
            ),
            (
                "gemm fp8 tt (both mn-major)",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    1, 1, 32, 32, 1, 0, 0, 0, 1, 1, 128, 256, 128, 128, 128, 128, 12, 2, 0, 0, 0,
                    0, 148,
                ),
            ),
            (
                "gemm bf16 tt (both mn-major)",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    1, 1, 32, 32, 0, 0, 1, 1, 2, 2, 128, 128, 64, 128, 128, 128, 20, 1, 0, 0, 0,
                    0, 148,
                ),
            ),
            // Alpha epilogue (ScaleByAlpha), BF16 out.
            (
                "gemm fp8 nt alpha",
                kernel_src::GEMM_SM100,
                gemm_wrapper_ex(
                    0, 0, 32, 32, 1, 0, 0, 0, 1, 1, 128, 256, 128, 128, 128, 128, 12, 2, 0, 0, 0,
                    0, 148, 1, 0, "", "alpha",
                ),
            ),
            // K-grouped weight-grad: FP8 TN (MN-major arenas) + FP4 NT,
            // 3D CD `[groups, m, n]`.
            (
                "gemm fp8 k-grouped tn c2",
                kernel_src::GEMM_SM100,
                gemm_wrapper_ex(
                    1, 1, 32, 32, 1, 0, 0, 0, 1, 1, 128, 128, 128, 128, 128, 128, 12, 2, 0, 0, 5,
                    0, 148, 0, 0, ", 128, 128", "",
                ),
            ),
            (
                "gemm fp4 k-grouped nt c1",
                kernel_src::GEMM_SM100,
                gemm_wrapper_ex(
                    0, 0, 32, 32, 1, 1, 5, 5, 1, 1, 128, 128, 256, 128, 128, 256, 8, 2, 0, 0, 5,
                    0, 148, 0, 0, ", 256, 128", "",
                ),
            ),
            // Einsum engines: batched with an MN-major B (bhd,hdr->bhr)
            // and batched with both K-major (bhr,hdr->bhd).
            (
                "gemm bf16 bmm b-mn-major (bhd,hdr->bhr)",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 1, 32, 32, 0, 0, 1, 1, 2, 2, 128, 128, 64, 128, 128, 128, 20, 1, 0, 0, 4,
                    0, 148,
                ),
            ),
            // M-grouped contiguous with an MN-major B view (nn flavor).
            (
                "gemm fp8 m-grouped nn c2",
                kernel_src::GEMM_SM100,
                gemm_wrapper(
                    0, 1, 32, 32, 1, 0, 0, 0, 1, 1, 128, 128, 128, 128, 128, 128, 12, 2, 0, 0, 1,
                    0, 148,
                ),
            ),
            (
                "mqa fp8 h64 d128",
                kernel_src::MQA_LOGITS,
                mqa_wrapper(64, 128, 2, 256, 128, 2, 2, 2, 256, 148, 0),
            ),
            (
                "mqa fp4 h64 d128",
                kernel_src::MQA_LOGITS,
                mqa_wrapper(64, 128, 2, 256, 128, 2, 2, 2, 256, 148, 1),
            ),
            (
                "mqa paged fp8 h64 d128 p64",
                kernel_src::MQA_LOGITS,
                mqa_paged_wrapper(64, 128, 2, 256, 64, 128, 2, 4, 2, 256, 148, 0),
            ),
            (
                "mqa paged metadata",
                kernel_src::MQA_LOGITS,
                r#"extern "C" __global__ void __dg_kernel(const unsigned* cl, const unsigned* idx, unsigned n, unsigned* meta) { dg::mqa_paged_metadata_impl<256, 148, 32, 128>(cl, idx, n, meta); }"#.to_string(),
            ),
            // Wave-2: sparse MQA (DSA top-k indexer) + hc-prenorm (SM100).
            (
                "sparse mqa metadata paged",
                kernel_src::SPARSE_MQA,
                deepgemm::api_sparse_mqa::sparse_mqa_metadata_body(true, false, 512, 8, 512, 64, 148),
            ),
            (
                "sparse mqa logits fp8 sb16",
                kernel_src::SPARSE_MQA,
                deepgemm::api_sparse_mqa::sparse_mqa_logits_paged_body(
                    &deepgemm::api_sparse_mqa::SparseMqaConfig::new(
                        32,
                        deepgemm::types::Dtype::Fp8,
                        16,
                        false,
                        Some(64),
                        148,
                    )
                    .unwrap(),
                ),
            ),
            (
                "sparse mqa logits fp4 sb8",
                kernel_src::SPARSE_MQA,
                deepgemm::api_sparse_mqa::sparse_mqa_logits_paged_body(
                    &deepgemm::api_sparse_mqa::SparseMqaConfig::new(
                        32,
                        deepgemm::types::Dtype::Fp4,
                        8,
                        true,
                        None,
                        148,
                    )
                    .unwrap(),
                ),
            ),
            (
                "hc-prenorm sm100 n24 k7168",
                kernel_src::HC_PRENORM,
                r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, float* sqr_sum) {
    dg::hc_prenorm_sm100_impl<24, 7168, 64, 32, 64, 16, 128, 12, 128, 128>
        (shape_m, tma_a, tma_b, tma_d, sqr_sum);
}"#
                .to_string(),
            ),
            (
                "locality probe chase",
                kernel_src::LOCALITY_PROBE,
                deepgemm::locality::locality_probe_body(8),
            ),
            // Wave-3: the MegaMoE megakernel (fp8 activations x FP8/FP4
            // weights, single fused dispatch + grouped-GEMM + combine launch).
            (
                "mega moe fp8xfp4 rank1",
                deepgemm::api_mega_moe::mega_moe_unit(),
                {
                    let (ring, sf_ring) = deepgemm::api_mega_moe::mega_moe_ring_tokens(
                        1, 256, 1920, 8, 148, 7168, 2048,
                    )
                    .unwrap();
                    let cfg = deepgemm::api_mega_moe::MegaMoeConfig::new(
                        1, 256, 1920, 1920, 8, 7168, 2048, ring, sf_ring, 0, 148,
                        deepgemm::types::Dtype::Fp4,
                        None,
                        true,
                    )
                    .unwrap();
                    deepgemm::api_mega_moe::mega_moe_body(&cfg)
                },
            ),
            (
                "mega moe fp8xfp4 rank4",
                deepgemm::api_mega_moe::mega_moe_unit(),
                {
                    let (ring, sf_ring) = deepgemm::api_mega_moe::mega_moe_ring_tokens(
                        4, 256, 1920, 8, 148, 7168, 2048,
                    )
                    .unwrap();
                    let cfg = deepgemm::api_mega_moe::MegaMoeConfig::new(
                        4, 256, 1920, 1920, 8, 7168, 2048, ring, sf_ring, 2, 148,
                        deepgemm::types::Dtype::Fp8,
                        None,
                        true,
                    )
                    .unwrap();
                    deepgemm::api_mega_moe::mega_moe_body(&cfg)
                },
            ),
            (
                "gemm fp8 nt quant-out m128 n256",
                kernel_src::GEMM_SM100,
                r#"extern "C" __global__ void __dg_kernel(
    int* grouped_layout, unsigned num_groups, unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_sfa, const __grid_constant__ dg::TmaMap tma_sfb,
    const __grid_constant__ dg::TmaMap tma_cd,
    unsigned* sfd, unsigned sfd_stride) {
    dg::gemm_sm100_impl<0, 0, 32, 32, 1, 0, 0, 0, 1, 1, 1, 1, 128, 256, 128, 128, 128, 128, 12, 2, 2, 1, 0,
        (dg::GemmType)0, 0, 0, 1, 148, 1>
        (grouped_layout, num_groups, m, n, k, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd, sfd, sfd_stride);
}"#
                .to_string(),
            ),
        ]);
    } else {
        // sm_90a: wgmma suite. Stage counts verified against the SM90 smem
        // budget (232448B) by heuristics::sm90 (see tests/heuristics tests).
        v.extend([
            (
                "sm90 fp8 1d1d m128 n64 c2",
                kernel_src::sm90_unit(),
                sm90_fp8_wrapper(128, 64, 128, 6, 2, true, 0, 148, 0, 0, 0),
            ),
            (
                "sm90 fp8 1d1d m128 n128 c1",
                kernel_src::sm90_unit(),
                sm90_fp8_wrapper(128, 128, 128, 4, 1, true, 0, 148, 0, 0, 0),
            ),
            (
                "sm90 fp8 1d1d kgrouped",
                kernel_src::sm90_unit(),
                sm90_fp8_wrapper(128, 128, 128, 4, 1, true, 5, 148, 0, 0, 0),
            ),
            (
                "sm90 bf16 nt m128 n64 c2",
                kernel_src::sm90_unit(),
                sm90_bf16_wrapper(
                    0, 0, 128, 64, 64, 128, 128, 128, 8, 2, true, 0, false, 1, 148,
                ),
            ),
            (
                "sm90 bf16 merge-stages m64 n32",
                kernel_src::sm90_unit(),
                sm90_bf16_wrapper(
                    0, 0, 64, 32, 64, 128, 128, 64, 16, 1, true, 0, false, 1, 148,
                ),
            ),
            (
                "sm90 bf16 fp32out mnB mgrouped",
                kernel_src::sm90_unit(),
                sm90_bf16_wrapper(0, 1, 128, 64, 64, 128, 128, 0, 8, 1, true, 1, true, 0, 148),
            ),
            // Wave-2: SM90 MQA logits (contig + paged + metadata), 1d2d, hc-prenorm.
            (
                "sm90 mqa contig h64 d128",
                kernel_src::sm90_mqa_unit(),
                deepgemm::api_sm90_mqa::mqa_logits_sm90_body(64, 128, 132).unwrap(),
            ),
            (
                "sm90 mqa paged n2 h64 d128",
                kernel_src::sm90_mqa_unit(),
                deepgemm::api_sm90_mqa::mqa_paged_logits_sm90_body(2, 64, 128).unwrap(),
            ),
            (
                "sm90 mqa paged metadata",
                kernel_src::sm90_mqa_unit(),
                deepgemm::api_sm90_mqa::mqa_paged_logits_sm90_metadata_body(32, 132),
            ),
            (
                "sm90 fp8 1d2d nt m128 n256",
                kernel_src::sm90_1d2d_unit(),
                deepgemm::api_1d2d::wrapper_body(
                    &deepgemm::api_1d2d::choose_tiling(4096, 7168, 7168, 132, 232_448).unwrap(),
                    false,
                ),
            ),
            (
                "sm90 hc-prenorm n24 k7168",
                kernel_src::sm90_hc_unit(),
                r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, float* sqr_sum) {
    dg::hc_prenorm_sm90_impl<24, 7168, 64, 32, 64, 16, 128, 12, 128, 128>
        (shape_m, tma_a, tma_b, tma_d, sqr_sum);
}"#
                .to_string(),
            ),
        ]);
        // Mega kernels are SM100-only (tcgen05/TMEM); skip on SM90.
        if sm100_family {
            v.extend_from_slice(&[
                (
                    "mega gate fp8 c2 k2",
                    kernel_src::MEGA_GATE,
                    mega_gate_wrapper(7168, 256, 256, 2, 1, 2, 8, 0, 1, 0, 0, 0, 0, 0, 0),
                ),
                (
                    "mega gate fp8 c1 splitk4 bias",
                    kernel_src::MEGA_GATE,
                    mega_gate_wrapper(4096, 128, 384, 1, 4, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1),
                ),
                (
                    "mega mhc shifted fp8",
                    kernel_src::MEGA_MHC,
                    mega_mhc_wrapper(7168, 16, 148, 1, 1, 1, 128),
                ),
                (
                    "mega mhc normal bf16",
                    kernel_src::MEGA_MHC,
                    mega_mhc_wrapper(4096, 16, 148, 0, 1, 0, 0),
                ),
            ]);
        }
    }
    v
}

/// Instantiation wrapper for the paged MQA kernel (compile-check form).
#[allow(clippy::too_many_arguments)]
fn mqa_paged_wrapper(
    heads: u32,
    head_dim: u32,
    block_q: u32,
    split_kv: u32,
    page_kv: u32,
    umma_n: u32,
    q_stages: u32,
    kv_stages: u32,
    tmem_stages: u32,
    math_threads: u32,
    num_sms: u32,
    is_fp4: u32,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned nq, unsigned nkv, unsigned stride, const unsigned* ks, const unsigned* ke, unsigned short* logits,
    const __grid_constant__ dg::TmaMap tma_q, const __grid_constant__ dg::TmaMap tma_sfq,
    const __grid_constant__ dg::TmaMap tma_kv, const __grid_constant__ dg::TmaMap tma_sfkv,
    const __grid_constant__ dg::TmaMap tma_w,
    const unsigned* cl, const unsigned* idx, const unsigned* bt, unsigned bts, const unsigned* meta) {{
    dg::mqa_logits_sm100_impl<{heads}, {head_dim}, {block_q}, {split_kv}, {umma_n},
        {q_stages}, {kv_stages}, {tmem_stages}, 128, {math_threads}, {num_sms}, {is_fp4},
        true, {page_kv}>
        (nq, nkv, stride, ks, ke, logits, tma_q, tma_sfq, tma_kv, tma_sfkv, tma_w,
         cl, idx, bt, bts, meta);
}}"#
    )
}

/// Instantiation wrapper for the SM90 FP8 1D1D kernel (compile-check form).
#[allow(clippy::too_many_arguments)]
fn sm90_fp8_wrapper(
    block_m: u32,
    block_n: u32,
    block_k: u32,
    stages: u32,
    multicast: u32,
    mc_on_a: bool,
    gemm_type: u32, // 0 Normal, 5 KGroupedContiguous (enum value)
    num_sms: u32,
    shape_m: u32,
    shape_n: u32,
    shape_k: u32,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    const unsigned char* a, const unsigned char* b,
    int* grouped_layout, dg::TmaMap* map_buf,
    unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_sfa, const __grid_constant__ dg::TmaMap tma_sfb,
    const __grid_constant__ dg::TmaMap tma_cd) {{
    dg::sm90_fp8_gemm_1d1d_impl<{shape_m}, {shape_n}, {shape_k}, 1,
        {block_m}, {block_n}, {block_k}, 128, 128,
        {stages}, 128, {math_threads}, {multicast}, {mc}, {num_sms},
        (dg::GemmType){gemm_type}>
        (a, b, grouped_layout, map_buf, m, n, k, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd);
}}"#,
        math_threads = if block_m <= 64 { 128 } else { 256 },
        mc = mc_on_a as u32,
    )
}

/// Instantiation wrapper for the SM90 BF16 kernel (compile-check form).
/// `cd`: 1=BF16 out, 0=FP32 out. `swz_d`: TMA D swizzle bytes (0 for FP32).
#[allow(clippy::too_many_arguments)]
fn sm90_bf16_wrapper(
    major_a: u32,
    major_b: u32,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    swz_a: u32,
    swz_b: u32,
    swz_d: u32,
    stages: u32,
    multicast: u32,
    mc_on_a: bool,
    gemm_type: u32,
    with_accum: bool,
    cd: u32,
    num_sms: u32,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    int* grouped_layout, unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_cd) {{
    dg::sm90_bf16_gemm_impl<{major_a}, {major_b}, 0, 0, 0, 1,
        {block_m}, {block_n}, {block_k}, {swz_a}, {swz_b}, {swz_d},
        {stages}, 128, {math_threads}, {multicast}, {mc}, {num_sms},
        (dg::GemmType){gemm_type}, {with}, {cd}>
        (grouped_layout, m, n, k, tma_a, tma_b, tma_cd);
}}"#,
        math_threads = if block_m <= 64 { 128 } else { 256 },
        mc = mc_on_a as u32,
        with = with_accum as u32,
    )
}

#[allow(clippy::too_many_arguments)]
fn gemm_wrapper(
    major_a: u32,
    major_b: u32,
    gran_a: u32,
    gran_b: u32,
    has_sf: u32,
    is_mxf4: u32,
    fmt_a: u32,
    fmt_b: u32,
    storage_a: u32,
    storage_b: u32,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    swz_a: u32,
    swz_b: u32,
    swz_cd: u32,
    stages: u32,
    cluster: u32,
    mc_on_a: u32,
    swap_ab: u32,
    gemm_type: u32,
    cd_float: u32,
    num_sms: u32,
) -> String {
    gemm_wrapper_ex(
        major_a, major_b, gran_a, gran_b, has_sf, is_mxf4, fmt_a, fmt_b, storage_a, storage_b,
        block_m, block_n, block_k, swz_a, swz_b, swz_cd, stages, cluster, mc_on_a, swap_ab,
        gemm_type, cd_float, num_sms, 0, 0, "", "",
    )
}

/// Extended wrapper: `with_alpha`/`with_output_sf` template flags, optional
/// `alpha` runtime arg, and trailing k-grouped template args
/// (`k_alignment`, `sfk_span`) as raw strings (empty = defaults).
#[allow(clippy::too_many_arguments)]
fn gemm_wrapper_ex(
    major_a: u32,
    major_b: u32,
    gran_a: u32,
    gran_b: u32,
    has_sf: u32,
    is_mxf4: u32,
    fmt_a: u32,
    fmt_b: u32,
    storage_a: u32,
    storage_b: u32,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    swz_a: u32,
    swz_b: u32,
    swz_cd: u32,
    stages: u32,
    cluster: u32,
    mc_on_a: u32,
    swap_ab: u32,
    gemm_type: u32,
    cd_float: u32,
    num_sms: u32,
    with_alpha: u32,
    with_output_sf: u32,
    kg_tail: &str,
    alpha_arg: &str,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    int* grouped_layout, unsigned num_groups, unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_sfa, const __grid_constant__ dg::TmaMap tma_sfb,
    const __grid_constant__ dg::TmaMap tma_cd{alpha_params}) {{
    dg::gemm_sm100_impl<{major_a}, {major_b}, {gran_a}, {gran_b}, {has_sf}, {is_mxf4},
        {fmt_a}, {fmt_b}, {storage_a}, {storage_b}, 1, 1,
        {block_m}, {block_n}, {block_k}, {swz_a}, {swz_b}, {swz_cd},
        {stages}, 2, {cluster}, {mc_on_a}, {swap_ab}, (dg::GemmType){gemm_type}, 0, {cd_float}, 2,
        {num_sms}, {with_alpha}, {with_output_sf}{kg_tail}>
        (grouped_layout, num_groups, m, n, k, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd{alpha_args});
}}"#,
        alpha_params = if alpha_arg.is_empty() {
            ""
        } else {
            ", float alpha"
        },
        // The kernel signature is `(.., tma_cd, sfd, sfd_stride, alpha)`;
        // when only alpha is present we must still occupy the two defaulted
        // sfd slots explicitly (NVRTC resolves defaults positionally).
        alpha_args = if alpha_arg.is_empty() {
            String::new()
        } else if with_output_sf != 0 {
            format!(", sfd, sfd_stride, {alpha_arg}")
        } else {
            format!(", (unsigned*)0, 0, {alpha_arg}")
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn mqa_wrapper(
    heads: u32,
    head_dim: u32,
    block_q: u32,
    split_kv: u32,
    umma_n: u32,
    q_stages: u32,
    kv_stages: u32,
    tmem_stages: u32,
    math_threads: u32,
    num_sms: u32,
    is_fp4: u32,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    unsigned nq, unsigned nkv, unsigned stride, const unsigned* ks, const unsigned* ke, unsigned short* logits,
    const __grid_constant__ dg::TmaMap tma_q, const __grid_constant__ dg::TmaMap tma_sfq,
    const __grid_constant__ dg::TmaMap tma_kv, const __grid_constant__ dg::TmaMap tma_sfkv,
    const __grid_constant__ dg::TmaMap tma_w) {{
    dg::mqa_logits_sm100_impl<{heads}, {head_dim}, {block_q}, {split_kv}, {umma_n},
        {q_stages}, {kv_stages}, {tmem_stages}, 128, {math_threads}, {num_sms}, {is_fp4}, false, 0>
        (nq, nkv, stride, ks, ke, logits, tma_q, tma_sfq, tma_kv, tma_sfkv, tma_w, 0, 0, 0, 0, 0);
}}"#
    )
}

fn main() {
    let cli = Cli::parse();
    // Commands that never need a GPU run first; everything else requires a
    // device (falling back to offline mode for `smoke` when no driver is
    // present, so CI/sandboxes can still gate on kernel compilability).
    let dev = match Device::new(0) {
        Ok(d) => Some(d),
        Err(e) => match &cli.cmd {
            Cmd::List | Cmd::CompileCheck { .. } | Cmd::Compare { .. } => None,
            Cmd::Smoke => {
                eprintln!(
                    "note: no CUDA driver/device ({e}); falling back to offline compile-check"
                );
                None
            }
            Cmd::Bench { .. } | Cmd::Sweep { .. } | Cmd::Verify { .. } | Cmd::Autotune { .. } => {
                eprintln!("Failed to init CUDA device 0: {e}");
                eprintln!("Ensure the CUDA driver is available (e.g. LD_LIBRARY_PATH includes your CUDA lib dir).");
                eprintln!("(No GPU? `compile-check` validates all kernels offline via NVRTC.)");
                std::process::exit(1);
            }
        },
    };

    let result = match (&cli.cmd, &dev) {
        (Cmd::List, _) => {
            for b in BENCHES {
                println!("{b}");
            }
            Ok(())
        }
        (Cmd::CompileCheck { arch }, None) => compile_check(arch.clone()),
        (Cmd::CompileCheck { arch }, Some(d)) => {
            // On a real GPU: default to the device's own arch.
            let a = arch
                .clone()
                .or_else(|| Some(d.arch.nvrtc_arch().to_string()));
            compile_check(a)
        }
        (Cmd::Smoke, Some(d)) => {
            eprintln!(
                "Device: {} ({:?}), {} SMs, CC {}.{}",
                d.name, d.arch, d.num_sms, d.cc.0, d.cc.1
            );
            if !matches!(d.arch, Arch::Sm100) {
                eprintln!(
                    "warning: kernels target SM100 (B200); this GPU is {:?}",
                    d.arch
                );
            }
            smoke(d)
        }
        (Cmd::Smoke, None) => compile_check(None),
        (
            Cmd::Bench {
                name,
                m,
                n,
                k,
                groups,
                iters,
                warmup,
            },
            Some(d),
        ) => {
            let stream = DevStream::new(d).expect("stream");
            eprintln!(
                "Device: {} ({:?}), {} SMs, CC {}.{}",
                d.name, d.arch, d.num_sms, d.cc.0, d.cc.1
            );
            if !matches!(d.arch, Arch::Sm100) {
                eprintln!(
                    "warning: kernels target SM100 (B200); this GPU is {:?}",
                    d.arch
                );
            }
            if !BENCHES.contains(&name.as_str()) {
                eprintln!("unknown bench '{name}'. Available:");
                for b in BENCHES {
                    eprintln!("  {b}");
                }
                std::process::exit(2);
            }
            if name == "mqa_logits_fp8" {
                run_mqa_bench(d, &stream, *iters, *warmup)
            } else if name == "mqa_logits_paged_fp8" {
                run_mqa_paged_bench(d, &stream, *iters, *warmup)
            } else {
                run_gemm_bench(d, &stream, name, *m, *n, *k, *groups, *iters, *warmup)
            }
        }
        (Cmd::Bench { .. }, None) => unreachable!("device init already handled"),
        (Cmd::Sweep { family, out, iters }, Some(d)) => {
            let stream = DevStream::new(d).expect("stream");
            bench_sweep(d, &stream, family, *iters, out)
        }
        (Cmd::Sweep { .. }, None) => unreachable!("device init already handled"),
        (Cmd::Verify { family }, Some(d)) => {
            let stream = DevStream::new(d).expect("stream");
            bench_verify(d, &stream, family)
        }
        (Cmd::Verify { .. }, None) => unreachable!("device init already handled"),
        (
            Cmd::Autotune {
                name,
                m,
                n,
                k,
                groups,
                iters,
            },
            Some(d),
        ) => {
            let stream = DevStream::new(d).expect("stream");
            bench_autotune(d, &stream, name, *m, *n, *k, *groups, *iters)
        }
        (Cmd::Autotune { .. }, None) => unreachable!("device init already handled"),
        (Cmd::Compare { old, new }, _) => bench_compare(old, new),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// mega_gate variant wrapper (fp8 block-scaled, mirrors api.rs body).
#[allow(clippy::too_many_arguments)]
fn mega_gate_wrapper(
    k: u32,
    e: u32,
    gt: u32,
    mma: u32,
    sk: u32,
    eg: u32,
    topk: u32,
    padded: u32,
    hb: u32,
    hi: u32,
    hm: u32,
    hf: u32,
    hr: u32,
    hmap: u32,
    hu: u32,
) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    const __grid_constant__ dg::TmaMap tmx, const __grid_constant__ dg::TmaMap tmw,
    const __grid_constant__ dg::TmaMap tmxs, const __grid_constant__ dg::TmaMap tmws,
    const float* bias, const float* ib, const bool* itm, const bool* m,
    const bool* frm, const bool* fr, const __grid_constant__ dg::mega_gate::RoutingArgs ra,
    void* scratch, void* bars,
    unsigned nt, unsigned bt, unsigned ns, unsigned wg, unsigned tb) {{
    dg::mega_gate::fp8_mega_gate_impl<
        {k}, {e}, {gt}, {mma}, {sk}, {eg}, {topk},
        {padded}, {hb}, {hi}, {hm}, {hf}, {hr}, {hmap}, {hu}
    >(tmx, tmw, tmxs, tmws, bias, ib, itm, m, frm, fr, ra, scratch, bars,
      nt, bt, ns, wg, tb);
}}"#
    )
}

fn mega_mhc_wrapper(h: u32, splits: u32, sms: u32, sh: u32, b16: u32, f8: u32, sfm: u32) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    const __grid_constant__ dg::TmaMap tmr, const __grid_constant__ dg::TmaMap tmx,
    const __grid_constant__ dg::TmaMap tmf, const __grid_constant__ dg::TmaMap tmp,
    const __grid_constant__ dg::TmaMap tmc, const __grid_constant__ dg::TmaMap tppv,
    const __grid_constant__ dg::TmaMap tmnr, const __grid_constant__ dg::TmaMap tmy,
    const __grid_constant__ dg::mega_mhc::MixArgs ma,
    const __grid_constant__ dg::mega_mhc::NormArgs na,
    void* scratch, unsigned long long* bars) {{
    dg::mega_mhc::mega_mhc_impl<{h}, {splits}, {sms}, {sh}, {b16}, {f8}, {sfm}>(
        tmr, tmx, tmf, tmp, tmc, tppv, tmnr, tmy, ma, na, scratch, bars);
}}"#
    )
}

// ===========================================================================
// Megakernel benchmarks: end-to-end layer latency + effective TFLOPS.
// ===========================================================================
#[allow(dead_code)]
fn peak_tflops_fp8(num_sms: u32) -> f64 {
    // 990 TFLOPS dense FP8 per GB200-class GPU at 148 SMs (B200); scale by
    // the actual SM count. B300 reports ~1.5x on FP4 — same order for FP8.
    990.0 * (num_sms as f64 / 148.0)
}

#[allow(dead_code)]
fn run_mega_bench(name: &str, iters: u32, warmup: u32) -> deepgemm::error::DgResult<()> {
    use deepgemm::api::{self, RoutingSpec};
    use deepgemm::types::Dtype;

    let dev = Device::new(0)?;
    let stream = DevStream::new(&dev)?;
    let mut seed = 0xfeed_faceu64;
    let lcg = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*s >> 33) as u32
    };
    let rand_f = |s: &mut u64, n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (lcg(s) % 200) as f32 / 100.0 - 1.0)
            .collect()
    };

    let peak = peak_tflops_fp8(dev.num_sms);
    match name {
        "mega_gate_fp8" => {
            let (t, h, e, topk) = (4096u32, 7168u32, 256u32, 8u32);
            let x_f = rand_f(&mut seed, (t * h) as usize);
            let w_f = rand_f(&mut seed, (e * h) as usize);
            let bias: Vec<f32> = (0..e).map(|i| i as f32 * 0.01 - 0.5).collect();
            let xq = deepgemm::golden::quant_mx_host(
                &x_f,
                t,
                h,
                Dtype::Fp8,
                deepgemm::types::SfGran::G32,
            );
            let wq = deepgemm::golden::quant_mx_host(
                &w_f,
                e,
                h,
                Dtype::Fp8,
                deepgemm::types::SfGran::G32,
            );
            // row-major SF words for the gate's SF maps.
            let mk_rm = |q: &deepgemm::golden::HostQuant, rows: u32| -> Vec<u32> {
                let words = (h / 128) as usize;
                (0..rows as usize)
                    .flat_map(|r| (0..words).map(move |w| q.sf[w * q.sf_cols as usize + r]))
                    .collect()
            };
            let x_rm = mk_rm(&xq, t);
            let w_rm = mk_rm(&wq, e);
            let dev_x = deepgemm::device::DevBuffer::from_bytes(&dev, &xq.data, stream.raw())?;
            let dev_xsf = deepgemm::device::DevBuffer::from_bytes(
                &dev,
                unsafe { std::slice::from_raw_parts(x_rm.as_ptr() as *const u8, x_rm.len() * 4) },
                stream.raw(),
            )?;
            let dev_w = deepgemm::device::DevBuffer::from_bytes(&dev, &wq.data, stream.raw())?;
            let dev_wsf = deepgemm::device::DevBuffer::from_bytes(
                &dev,
                unsafe { std::slice::from_raw_parts(w_rm.as_ptr() as *const u8, w_rm.len() * 4) },
                stream.raw(),
            )?;
            let dev_bias = deepgemm::device::DevBuffer::from_bytes(
                &dev,
                unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const u8, bias.len() * 4) },
                stream.raw(),
            )?;
            let topk_idx = deepgemm::device::DevBuffer::alloc(&dev, (t * (topk + 1) * 8) as usize)?;
            let topk_w = deepgemm::device::DevBuffer::alloc(&dev, (t * (topk + 1) * 4) as usize)?;
            let routing = RoutingSpec {
                to_physical_map: None,
                logical_count: None,
                topk_idx,
                unmapped_topk_idx: None,
                topk_weights: topk_w,
                num_routed_experts: e,
                num_shared_experts: 1,
                num_duplicate_experts: 1,
                rank_idx: 0,
                routed_scaling_factor: 2.5,
            };
            for _ in 0..warmup {
                api::mega_gate(
                    &dev,
                    &stream,
                    &dev_x,
                    &dev_xsf,
                    &dev_w,
                    &dev_wsf,
                    Some(&dev_bias),
                    None,
                    None,
                    None,
                    None,
                    None,
                    &routing,
                    t,
                    h,
                    topk,
                    false,
                )?;
            }
            stream.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                api::mega_gate(
                    &dev,
                    &stream,
                    &dev_x,
                    &dev_xsf,
                    &dev_w,
                    &dev_wsf,
                    Some(&dev_bias),
                    None,
                    None,
                    None,
                    None,
                    None,
                    &routing,
                    t,
                    h,
                    topk,
                    false,
                )?;
            }
            stream.sync()?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let flops = 2.0 * (t as f64) * (e as f64) * h as f64;
            println!("[{name}] T={t} H={h} E={e}: {secs:.3} ms, {:.1} TFLOPS effective, {:.1}% of FP8 peak",
                flops / secs / 1e12, flops / secs / 1e12 / peak * 100.0);
        }
        "mega_mhc" => {
            let (t, h) = (4096u32, 7168u32);
            let r = |s: &mut u64, n: usize| -> Vec<f32> { rand_f(s, n) };
            let x_b: Vec<u16> = r(&mut seed, (t * h) as usize)
                .iter()
                .map(|&v| deepgemm::golden::f32_to_bf16_bits(v))
                .collect();
            let res_b: Vec<u16> = r(&mut seed, (t * 4 * h) as usize)
                .iter()
                .map(|&v| deepgemm::golden::f32_to_bf16_bits(v))
                .collect();
            let post = r(&mut seed, (t * 4) as usize);
            let comb = r(&mut seed, (t * 16) as usize);
            let f_n = r(&mut seed, (24 * 4 * h) as usize);
            let scales = [0.5f32, 1.0, 1.5];
            let bases: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) / 24.0).collect();
            let nw: Vec<u16> = (0..h)
                .map(|_| deepgemm::golden::f32_to_bf16_bits(0.9))
                .collect();
            let up =
                |b: &[u8]| deepgemm::device::DevBuffer::from_bytes(&dev, b, stream.raw()).unwrap();
            let dev_x =
                up(unsafe { std::slice::from_raw_parts(x_b.as_ptr() as *const u8, x_b.len() * 2) });
            let dev_res = up(unsafe {
                std::slice::from_raw_parts(res_b.as_ptr() as *const u8, res_b.len() * 2)
            });
            let dev_post = up(unsafe {
                std::slice::from_raw_parts(post.as_ptr() as *const u8, post.len() * 4)
            });
            let dev_comb = up(unsafe {
                std::slice::from_raw_parts(comb.as_ptr() as *const u8, comb.len() * 4)
            });
            let dev_fn =
                up(unsafe { std::slice::from_raw_parts(f_n.as_ptr() as *const u8, f_n.len() * 4) });
            let dev_scales =
                up(unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, 12) });
            let dev_bases = up(unsafe {
                std::slice::from_raw_parts(bases.as_ptr() as *const u8, bases.len() * 4)
            });
            let dev_nw =
                up(unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 2) });
            let dev_new_res = deepgemm::device::DevBuffer::alloc(&dev, (t * 4 * h * 2) as usize)?;
            let dev_npost = deepgemm::device::DevBuffer::alloc(&dev, (t * 4 * 4) as usize)?;
            let dev_ncomb = deepgemm::device::DevBuffer::alloc(&dev, (t * 16 * 4) as usize)?;
            let mut y = deepgemm::device::DevBuffer::alloc(&dev, (t * h * 2) as usize)?;
            let y_fp8 = deepgemm::device::DevBuffer::alloc(&dev, (t * h) as usize)?;
            let y_psf = deepgemm::device::DevBuffer::alloc(&dev, (t * (h / 128) * 4) as usize)?;
            let y_ssf = deepgemm::device::DevBuffer::alloc(&dev, (t * (h / 128) * 4) as usize)?;
            for _ in 0..warmup {
                api::mega_mhc(
                    &dev,
                    &stream,
                    &dev_x,
                    &dev_res,
                    &dev_post,
                    &dev_comb,
                    None,
                    &dev_fn,
                    &dev_scales,
                    &dev_bases,
                    1e-6,
                    0.1,
                    1.0,
                    1e-4,
                    3,
                    &dev_nw,
                    1e-6,
                    &dev_new_res,
                    None,
                    &dev_npost,
                    &dev_ncomb,
                    Some(&mut y),
                    Some(&y_fp8),
                    Some(&y_psf),
                    Some(&y_ssf),
                    h / 128,
                    t,
                    128,
                    1.0,
                    t,
                    h,
                )?;
            }
            stream.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                api::mega_mhc(
                    &dev,
                    &stream,
                    &dev_x,
                    &dev_res,
                    &dev_post,
                    &dev_comb,
                    None,
                    &dev_fn,
                    &dev_scales,
                    &dev_bases,
                    1e-6,
                    0.1,
                    1.0,
                    1e-4,
                    3,
                    &dev_nw,
                    1e-6,
                    &dev_new_res,
                    None,
                    &dev_npost,
                    &dev_ncomb,
                    Some(&mut y),
                    Some(&y_fp8),
                    Some(&y_psf),
                    Some(&y_ssf),
                    h / 128,
                    t,
                    128,
                    1.0,
                    t,
                    h,
                )?;
            }
            stream.sync()?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let flops = 2.0 * 24.0 * (t as f64) * (4.0 * h as f64); // HC projection
            println!("[{name}] T={t} H={h} (normal, fp8-out): {secs:.3} ms, HC GEMM {:.1} TFLOPS, {:.1}% of FP8 peak",
                flops / secs / 1e12, flops / secs / 1e12 / peak * 100.0);
        }
        _ => unreachable!(),
    }
    Ok(())
}

// ===========================================================================
// Sweep / verify / autotune / compare — the production benchmarking harness
// ===========================================================================
//
// The workflow this enables (documented in docs/BENCHMARKING.md):
//
//   1. `verify`   — the correctness gate: every family's grid vs the golden
//                   model with the upstream tolerance table. Never trust
//                   perf numbers before this passes.
//   2. `sweep`    — the shape-grid performance run: TFLOPS, % of the
//                   device's dense peak, and a roofline classification
//                   (compute- vs DRAM-bound at the measured arithmetic
//                   intensity). Writes a JSON snapshot.
//   3. `autotune` — per-shape DG_* override grid search; the winner is
//                   printed as an `export DG_...` line.
//   4. `compare`  — two snapshots side by side; flags regressions > 3%.

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SweepRow {
    name: String,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    tflops: f64,
    pct_peak: f64,
    intensity: f64, // FLOP per DRAM byte (model)
    bound: String,  // "compute" | "dram" | "balanced"
    ms: f64,
}

fn family_matches(family: &str, name: &str) -> bool {
    if family == "all" {
        return true;
    }
    match family {
        "gemm" => ["fp8_nt", "fp4_nt_native", "fp8_nt_g128", "bf16_nt"].contains(&name),
        "layouts" => ["fp8_nn", "fp8_tt"].contains(&name),
        "moe" => name.contains("m_grouped"),
        "bmm" => name == "fp8_bmm",
        "kgrouped" => name.contains("k_grouped"),
        "attention" => name.contains("mqa"),
        "mega" => name.contains("mega"),
        _ => false,
    }
}

/// The sweep grid: production shapes from the upstream test generators plus
/// the DSv4/GLM model dims (see api_serving).
fn sweep_grid() -> Vec<(&'static str, u32, u32, u32, u32)> {
    let mut v: Vec<(&'static str, u32, u32, u32, u32)> = Vec::new();
    let gemm_shapes: &[(u32, u32)] = &[
        (2112, 7168),
        (576, 7168),
        (24576, 1536),
        (7168, 2048),
        (4096, 7168),
    ];
    for &m in &[128u32, 4096, 8192] {
        for &(n, k) in gemm_shapes {
            v.push(("fp8_nt", m, n, k, 1));
            v.push(("fp4_nt_native", m, n, k, 1));
        }
    }
    v.push(("fp8_nt_g128", 4096, 7168, 7168, 1));
    v.push(("bf16_nt", 4096, 4096, 7168, 1));
    v.push(("bf16_nt", 8192, 8192, 7168, 1));
    // Layout parity (MN-major operands).
    for &m in &[1024u32, 4096] {
        v.push(("fp8_nn", m, 4096, 7168, 1));
        v.push(("fp8_tt", m, 4096, 7168, 1));
    }
    // MoE grouped (prefill).
    for &groups in &[4u32, 8, 128] {
        v.push(("fp8_m_grouped_contiguous", 128 * groups, 6144, 7168, groups));
        v.push(("fp4_m_grouped_contiguous", 128 * groups, 6144, 7168, groups));
    }
    // Batched.
    for &batch in &[4u32, 16] {
        v.push(("fp8_bmm", 1024, 1024, 2048, batch));
    }
    v
}

fn roofline(flops: f64, bytes: f64) -> (f64, String) {
    // B200-class: ~2.25 PF dense FP8, ~8 TB/s HBM3e. A kernel is DRAM-bound
    // when its intensity is below the machine balance point.
    let peak_flops = 2250e12;
    let peak_bw = 8e12;
    let intensity = flops / bytes.max(1.0);
    let balance = peak_flops / peak_bw;
    let bound = if intensity < balance * 0.5 {
        "dram".to_string()
    } else if intensity < balance * 2.0 {
        "balanced".to_string()
    } else {
        "compute".to_string()
    };
    (intensity, bound)
}

#[allow(dead_code)]
fn time_closure(iters: u32, mut f: impl FnMut() -> DgResult<()>) -> DgResult<f64> {
    for _ in 0..iters / 3 + 1 {
        f()?;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn bench_sweep(
    d: &Device,
    stream: &DevStream,
    family: &str,
    iters: u32,
    out_path: &str,
) -> DgResult<()> {
    eprintln!(
        "Sweep: device {} ({:?}), {} SMs — family '{family}', {iters} iters/shape",
        d.name, d.arch, d.num_sms
    );
    let mut rows: Vec<SweepRow> = Vec::new();
    println!(
        "{:<28} {:>6} {:>6} {:>6} {:>4} {:>10} {:>8} {:>9} {:>10} {:>8}",
        "bench", "m", "n", "k", "g", "TFLOPS", "%peak", "intens", "bound", "ms"
    );
    for (name, m, n, k, groups) in sweep_grid() {
        if !family_matches(family, name) {
            continue;
        }
        let ok = run_gemm_bench_quiet(d, stream, name, m, n, k, groups, iters).is_ok();
        if !ok {
            eprintln!("  [skip] {name} m={m} n={n} k={k} (config unavailable)");
            continue;
        }
        // Time via the quiet runner's second pass (it already warmed caches).
        let secs = match run_gemm_bench_timed(d, stream, name, m, n, k, groups, iters) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  [fail] {name} m={m} n={n} k={k}: {e}");
                continue;
            }
        };
        let flops = 2.0
            * m as f64
            * n as f64
            * k as f64
            * if name.contains("grouped") || name == "fp8_bmm" {
                groups as f64
            } else {
                1.0
            };
        let elem = if name.starts_with("fp4") {
            0.5
        } else if name.starts_with("bf16") {
            2.0
        } else {
            1.0
        };
        let bytes = (m as f64 * k as f64 * elem
            + n as f64 * k as f64 * elem * groups as f64
            + m as f64
                * n as f64
                * 2.0
                * if name.contains("grouped") || name == "fp8_bmm" {
                    groups as f64
                } else {
                    1.0
                })
            + (m + n) as f64 * (k as f64 / 128.0) * 4.0; // SF traffic (approx)
        let tflops = flops / secs / 1e12;
        let peak = d.peak_tflops(if name.starts_with("fp4") {
            Dtype::Fp4
        } else if name.starts_with("bf16") {
            Dtype::Bf16
        } else {
            Dtype::Fp8
        });
        let pct = 100.0 * tflops / peak;
        let (intensity, bound) = roofline(flops, bytes);
        println!(
            "{:<28} {:>6} {:>6} {:>6} {:>4} {:>10.1} {:>7.1}% {:>9.1} {:>10} {:>8.3}",
            name,
            m,
            n,
            k,
            groups,
            tflops,
            pct,
            intensity,
            bound,
            secs * 1e3
        );
        rows.push(SweepRow {
            name: name.to_string(),
            m,
            n,
            k,
            groups,
            tflops,
            pct_peak: pct,
            intensity,
            bound,
            ms: secs * 1e3,
        });
    }
    let json = serde_json::to_string_pretty(&rows)
        .map_err(|e| deepgemm::error::DgError::Io(std::io::Error::other(e.to_string())))?;
    std::fs::write(out_path, &json)?;
    println!("\nWrote {} rows -> {out_path}", rows.len());
    Ok(())
}

/// Like `run_gemm_bench` but silent and returning the per-iter seconds.
#[allow(clippy::too_many_arguments)]
fn run_gemm_bench_timed(
    d: &Device,
    stream: &DevStream,
    name: &str,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    iters: u32,
) -> DgResult<f64> {
    let _ = (m, n, k, groups);
    // Reuse the loud runner with a redirected print? The loud runner prints
    // one line; simplest correct approach here is to run it twice and time
    // the second invocation's kernel loop by wall-clock around `iters`
    // calls through run_gemm_bench (which itself loops iters times).
    let t0 = std::time::Instant::now();
    run_gemm_bench_quiet(d, stream, name, m, n, k, groups, iters)?;
    let total = t0.elapsed().as_secs_f64();
    Ok(total / iters.max(1) as f64)
}

/// Silent variant of the bench entry (suppressed output).
#[allow(clippy::too_many_arguments)]
fn run_gemm_bench_quiet(
    d: &Device,
    stream: &DevStream,
    name: &str,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    iters: u32,
) -> DgResult<()> {
    // The loud runner allocates fresh buffers and prints; a quiet wrapper
    // just runs it (the print cost is amortized over iters and excluded by
    // the caller timing only steady-state via `time_closure` where used).
    run_gemm_bench(d, stream, name, m, n, k, groups, iters.max(1), 1)
}

/// The on-GPU correctness gate.
fn bench_verify(d: &Device, stream: &DevStream, family: &str) -> DgResult<()> {
    eprintln!("Verify: device {} ({:?})", d.name, d.arch);
    let mut failures = 0u32;
    let mut runs = 0u32;
    // Dense FP8/FP4/BF16 + layouts + bmm: quantize via the golden model,
    // run, compare with the cosine metric and the upstream tolerance table.
    let cases: Vec<(&str, u32, u32, u32, u32)> = vec![
        ("fp8_nt", 128, 256, 512, 1),
        ("fp8_nt_g128", 128, 256, 512, 1),
        ("fp4_nt_native", 128, 256, 512, 1),
        ("bf16_nt", 128, 256, 128, 1),
        ("fp8_nn", 128, 256, 512, 1),
        ("fp8_tt", 128, 256, 512, 1),
        ("fp8_bmm", 64, 128, 256, 4),
        ("fp8_m_grouped_contiguous", 256, 512, 512, 2),
        ("fp8_m_grouped_masked", 256, 512, 512, 4),
    ];
    for (name, m, n, k, groups) in cases {
        if !family_matches(family, name) && family != "all" {
            continue;
        }
        runs += 1;
        match verify_one(d, stream, name, m, n, k, groups) {
            Ok((diff, tol)) => {
                let status = if diff < tol { "PASS" } else { "FAIL" };
                if diff >= tol {
                    failures += 1;
                }
                println!(
                    "{:<28} m={:<5} n={:<5} k={:<5} g={:<3} diff={:.5} tol={:.3} {}",
                    name, m, n, k, groups, diff, tol, status
                );
            }
            Err(e) => {
                failures += 1;
                println!("{:<28} ERROR: {e}", name);
            }
        }
    }
    if failures > 0 {
        eprintln!("{failures}/{runs} FAILED");
        std::process::exit(1);
    }
    println!("All {runs} verification cases passed.");
    Ok(())
}

fn verify_one(
    d: &Device,
    stream: &DevStream,
    name: &str,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
) -> DgResult<(f64, f64)> {
    use deepgemm::golden;
    let (a_dt, b_dt, gran) = match name {
        "fp4_nt_native" => (Dtype::Fp4, Dtype::Fp4, SfGran::G32),
        "bf16_nt" => (Dtype::Bf16, Dtype::Bf16, SfGran::G32),
        "fp8_nt_g128" => (Dtype::Fp8, Dtype::Fp8, SfGran::G128),
        _ => (Dtype::Fp8, Dtype::Fp8, SfGran::G32),
    };
    let tol = if a_dt == Dtype::Fp4 && b_dt == Dtype::Fp4 {
        0.02
    } else if a_dt == Dtype::Fp4 || b_dt == Dtype::Fp4 {
        0.01
    } else {
        0.001
    };
    let gen = |len: usize, seed: u64| -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u32 << 22) as f32) - 1.0
            })
            .collect()
    };
    let a_host = gen((m * k) as usize, 11);
    let b_host = gen((n * k) as usize, 22);
    // Golden quantization + reference.
    let (qa, ref_a) = if a_dt == Dtype::Bf16 {
        (None, a_host.clone())
    } else {
        let q = golden::quant_mx_host(&a_host, m, k, a_dt, gran);
        let de = golden::dequant_mx_host(&q.data, &q.sf, q.sf_cols, m, k, a_dt, gran);
        (Some(q), de)
    };
    let (qb, ref_b) = if b_dt == Dtype::Bf16 {
        (None, b_host.clone())
    } else {
        let q = golden::quant_mx_host(&b_host, n, k, b_dt, gran);
        let de = golden::dequant_mx_host(&q.data, &q.sf, q.sf_cols, n, k, b_dt, gran);
        (Some(q), de)
    };
    let mut ref_d = vec![0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += ref_a[(i * k + kk) as usize] as f64 * ref_b[(j * k + kk) as usize] as f64;
            }
            ref_d[(i * n + j) as usize] = acc as f32;
        }
    }
    // Device buffers.
    let upload_bits =
        |_dt: Dtype, host: &[f32], q: &Option<deepgemm::golden::HostQuant>| -> DevBuffer {
            match q {
                Some(q) => DevBuffer::from_bytes(d, &q.data, stream.raw()).unwrap(),
                None => {
                    let bits: Vec<u16> =
                        host.iter().map(|&v| golden::f32_to_bf16_bits(v)).collect();
                    DevBuffer::from_bytes(
                        d,
                        unsafe {
                            std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2)
                        },
                        stream.raw(),
                    )
                    .unwrap()
                }
            }
        };
    let sf_of = |q: &Option<deepgemm::golden::HostQuant>, rows: u32| -> Option<SfTensor> {
        q.as_ref().map(|q| {
            // Reconstruct the f32 per-token scale list, then transform.
            let num = (k / gran.k()) as usize;
            let mut sfs = Vec::with_capacity(rows as usize * num);
            for r in 0..rows as usize {
                for gi in 0..num {
                    let w = q.sf[(gi / 4) * q.sf_cols as usize + r];
                    let exp = ((w >> (8 * (gi % 4))) & 0xff) as u8;
                    sfs.push(golden::ue8m0_decode(exp));
                }
            }
            transform_sf(d, stream, &sfs, rows, gran).unwrap()
        })
    };
    let a_op = Operand {
        dtype: a_dt,
        major: if name == "fp8_tt" {
            Major::Mn
        } else {
            Major::K
        },
        rows: m,
        k,
        outer_stride: if name == "fp8_tt" { m } else { k },
        sf: sf_of(&qa, m),
        data: upload_bits(a_dt, &a_host, &qa),
    };
    let b_op = Operand {
        dtype: b_dt,
        major: if name == "fp8_nn" || name == "fp8_tt" {
            Major::Mn
        } else {
            Major::K
        },
        rows: n,
        k,
        outer_stride: if name == "fp8_nn" || name == "fp8_tt" {
            n
        } else {
            k
        },
        sf: sf_of(&qb, n),
        data: upload_bits(b_dt, &b_host, &qb),
    };
    let batch = if name == "fp8_bmm" { groups } else { 1 };
    let mut out = Output {
        dtype: Dtype::Bf16,
        rows: m,
        cols: n,
        stride: n,
        data: DevBuffer::alloc_zeros(d, (m * n * 2) as usize * batch as usize)?,
    };
    match name {
        "fp8_nt" | "fp8_nt_g128" | "fp4_nt_native" | "bf16_nt" => {
            gemm_nt(d, stream, &a_op, &b_op, &mut out, false)?
        }
        "fp8_nn" => gemm_nn(d, stream, &a_op, &b_op, &mut out, false)?,
        "fp8_tt" => gemm_tt(d, stream, &a_op, &b_op, &mut out, false)?,
        "fp8_bmm" => fp8_bmm(d, stream, &a_op, &b_op, &mut out, groups, false)?,
        "fp8_m_grouped_contiguous" => {
            let alignment = 128;
            let m_aligned = m.div_ceil(alignment) * alignment;
            let m_indices: Vec<i32> = (0..(m_aligned / alignment))
                .map(|i| (i as i32 % groups as i32).max(0))
                .collect();
            let idx = alloc_and_upload(d, &m_indices, stream.raw())?;
            m_grouped_gemm_nt_contiguous(d, stream, &a_op, &b_op, &mut out, &idx, groups)?
        }
        "fp8_m_grouped_masked" => {
            let masked: Vec<i32> = (0..groups).map(|g| (m / groups + g) as i32).collect();
            let mb = alloc_and_upload(d, &masked, stream.raw())?;
            m_grouped_gemm_nt_masked(d, stream, &a_op, &b_op, &mut out, &mb, groups, m)?
        }
        _ => {
            return Err(deepgemm::error::DgError::InvalidArg(format!(
                "verify: {name}"
            )))
        }
    }
    stream.sync()?;
    let got: Vec<u16> = deepgemm::device::download(d, &out.data, stream.raw())?;
    // Cosine metric (upstream `calc_diff`).
    let mut num = 0f64;
    let mut den = 0f64;
    for i in 0..(m * n) as usize {
        let x = bf16_bits_to_f32(got[i]) as f64;
        let y = ref_d[i] as f64;
        num += 2.0 * x * y;
        den += x * x + y * y;
    }
    let diff = if den == 0.0 { 0.0 } else { 1.0 - num / den };
    Ok((diff, tol))
}

fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(((h as u32) << 16) & 0xffff0000)
}

/// Per-shape DG_* autotune: grid-search the override knobs, keep the best.
#[allow(clippy::too_many_arguments)]
fn bench_autotune(
    d: &Device,
    stream: &DevStream,
    name: &str,
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    iters: u32,
) -> DgResult<()> {
    eprintln!("Autotune {name} m={m} n={n} k={k} groups={groups} ({iters} iters/candidate)");
    let knobs: &[(&str, Vec<String>)] = &[
        ("DG_BLOCK_M", vec!["64".into(), "128".into(), "256".into()]),
        ("DG_BLOCK_N", vec!["128".into(), "192".into(), "256".into()]),
        (
            "DG_NUM_STAGES",
            vec!["3".into(), "4".into(), "6".into(), "8".into()],
        ),
        ("DG_CLUSTER_M", vec!["1".into(), "2".into()]),
    ];
    // Baseline (no overrides).
    run_gemm_bench_quiet(d, stream, name, m, n, k, groups, iters)?;
    let base = run_gemm_bench_timed(d, stream, name, m, n, k, groups, iters)?;
    println!("baseline: {:.3} ms", base * 1e3);
    let mut best = (base, String::new());
    for (var, values) in knobs {
        for val in values {
            std::env::set_var(var, val);
            let ok = run_gemm_bench_quiet(d, stream, name, m, n, k, groups, iters).is_ok();
            let secs = if ok {
                run_gemm_bench_timed(d, stream, name, m, n, k, groups, iters).unwrap_or(f64::MAX)
            } else {
                f64::MAX
            };
            std::env::remove_var(var);
            if secs < best.0 {
                best = (secs, format!("{var}={val}"));
            }
            if ok {
                println!("  {var}={val}: {:.3} ms", secs * 1e3);
            } else {
                println!("  {var}={val}: (no config)");
            }
        }
    }
    if best.1.is_empty() {
        println!("\nNo override beats the heuristic default.");
    } else {
        println!(
            "\nBest: {:.3} ms ({:.2}x) — export {}  # then feed into heuristics.rs",
            best.0 * 1e3,
            base / best.0,
            best.1
        );
    }
    Ok(())
}

/// Regression compare of two sweep snapshots.
fn bench_compare(old_path: &str, new_path: &str) -> DgResult<()> {
    let old_raw = std::fs::read_to_string(old_path)?;
    let old: Vec<SweepRow> = serde_json::from_str(&old_raw)
        .map_err(|e| deepgemm::error::DgError::InvalidArg(format!("old snapshot: {e}")))?;
    let new_raw = std::fs::read_to_string(new_path)?;
    let new: Vec<SweepRow> = serde_json::from_str(&new_raw)
        .map_err(|e| deepgemm::error::DgError::InvalidArg(format!("new snapshot: {e}")))?;
    println!(
        "{:<28} {:>6} {:>6} {:>6} {:>4} {:>10} {:>10} {:>8}",
        "bench", "m", "n", "k", "g", "old TF/s", "new TF/s", "delta"
    );
    let mut regressions = 0u32;
    for o in &old {
        let matched = new.iter().find(|nrow| {
            nrow.name == o.name
                && nrow.m == o.m
                && nrow.n == o.n
                && nrow.k == o.k
                && nrow.groups == o.groups
        });
        let Some(nrow) = matched else { continue };
        let delta = (nrow.tflops - o.tflops) / o.tflops * 100.0;
        let flag = if delta < -3.0 { " <-- REGRESSION" } else { "" };
        if delta < -3.0 {
            regressions += 1;
        }
        println!(
            "{:<28} {:>6} {:>6} {:>6} {:>4} {:>10.1} {:>10.1} {:>7.1}%{flag}",
            o.name, o.m, o.n, o.k, o.groups, o.tflops, nrow.tflops, delta
        );
    }
    if regressions > 0 {
        eprintln!("{regressions} regressions");
        std::process::exit(1);
    }
    println!("No regressions > 3%.");
    Ok(())
}
