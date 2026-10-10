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
    use deepgemm::types::SfTensor;

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
                let per = m / groups;
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
    let variants = kernel_variants(&dev.arch.nvrtc_arch().to_string());
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
        ]);
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
    format!(
        r#"extern "C" __global__ void __dg_kernel(
    int* grouped_layout, unsigned num_groups, unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_sfa, const __grid_constant__ dg::TmaMap tma_sfb,
    const __grid_constant__ dg::TmaMap tma_cd) {{
    dg::gemm_sm100_impl<{major_a}, {major_b}, {gran_a}, {gran_b}, {has_sf}, {is_mxf4},
        {fmt_a}, {fmt_b}, {storage_a}, {storage_b}, 1, 1,
        {block_m}, {block_n}, {block_k}, {swz_a}, {swz_b}, {swz_cd},
        {stages}, 2, {cluster}, {mc_on_a}, {swap_ab}, (dg::GemmType){gemm_type}, 0, {cd_float}, 2, {num_sms}>
        (grouped_layout, num_groups, m, n, k, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd);
}}"#
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
            Cmd::List | Cmd::CompileCheck { .. } => None,
            Cmd::Smoke => {
                eprintln!(
                    "note: no CUDA driver/device ({e}); falling back to offline compile-check"
                );
                None
            }
            Cmd::Bench { .. } => {
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
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
