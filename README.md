# DeepGEMM-RS

**A Rust port of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — DeepSeek's tensor-core kernel library — for the inference stack of DeepSeek-V4-class and MiMo-class MoE models on NVIDIA Blackwell (B200 / GB200).**

Everything is JIT-compiled at runtime with NVRTC (the same model as upstream DeepGEMM's DeepJIT): there is no CUDA build step, no `nvcc`, and no toolkit needed to *build* — kernels are compiled on first use, per shape-config, and cached on disk (`~/.cache/deepgemm-rs`).

Because NVRTC is a pure compiler and the numeric semantics are mirrored by a CPU golden model, **the whole project runs — and is validated — in two planes**: a GPU-less sandbox (compile + math tests) and the B200 itself (correctness + TFLOPS). See [docs/concepts.md](docs/concepts.md) for the full concept guide with diagrams.

```text
   GPU-less plane (CI / laptop / this sandbox)          B200 plane
  ┌─────────────────────────────┐            ┌─────────────────────────────┐
  │ compile-check  (PTX+SASS)   │            │ smoke (on-device JIT)       │
  │ cargo test    (golden math) │── same ──▶│ cargo test --features e2e   │
  └─────────────────────────────┘  kernels  │ bench ... (TFLOPS vs peak) │
                              └─ and model ┘┴────────────────────────────┘
```

## What's implemented

| Operator | Kernel path | Notes |
|---|---|---|
| `fp8_gemm_nt` | `tcgen05.mma.kind::mxf8f6f4.block_scale` | MXFP8 (E4M3) with UE8M0 block scales, granularity 32 (MX) or 128 (DeepSeek recipe) |
| `fp4_gemm_nt` (**native**) | `tcgen05.mma.kind::mxf4.block_scale.block32` | packed E2M1 operands (2/byte), BLOCK_K=256, UMMA_K=64 — the flagship Blackwell path (`bench fp4_nt_native`) |
| mixed FP8 x FP4 | `kind::mxf8f6f4` with unpacked E2M1 | |
| `bf16_gemm_nt` | `tcgen05.mma.kind::f16` | no scale factors |
| `m_grouped_*_gemm_nt_contiguous` | same unified kernel, swap-AB | MoE prefill (m_indices layout) |
| `m_grouped_*_gemm_nt_masked` | same unified kernel, swap-AB | MoE decode (CUDA-graph friendly) |
| `fp8_bmm` | 3D TMA batched variant | |
| `mqa_logits` | MXF8F6F4/MXF4 MMA + TMEM reduce | weighted-ReLU MQA scoring for the MLA lightning indexer (contiguous KV) |
| `transform_sf` | transpose & pack FP32 -> UE8M0 | required SF layout for SM100 |
| `quant_mx` / `dequant_mx` | fused activation quant | MXFP8 / MXFP4 with power-of-two UE8M0 scales |

All GEMMs share the upstream-faithful warp-specialized pipeline:

- **warp 0** — TMA producer (A/B tiles + SF tiles, 2-CTA `cta_group::2` loads, multicast)
- **warp 1** — MMA issue (leader CTA only): SF `tcgen05.cp` (UTCCP) into TMEM, then `tcgen05.mma` with scale-factor TMEM operands; `tcgen05.commit` mbarrier signaling
- **warps 2/3** — SF 4x32 SMEM transpose feeding the UTCCP layout (upstream's CUDA-core transpose trick)
- **warps 4-7** — epilogue: `tcgen05.ld` accumulators -> swizzled SMEM -> TMA store (or `cp.reduce.async` for accumulation), TMEM overlap-barrier pipeline
- 2-CTA cluster MMA (`UMMA_M = 256`), persistent scheduler with L2-friendly block swizzle, PDL (`griddepcontrol`), TMEM column-overlap optimization, 2-stage accumulator / 2-stage TMA-store pipelines

The heuristics are a line-by-line port of upstream `SM100ArchSpec` (block-size enumeration, cluster selection, stage count from the 227KB SMEM budget, wave/multicast/utilization comparator).

## Requirements

- **For running on a GPU**: NVIDIA Blackwell datacenter GPU (B200 / GB200; SM100a), CUDA **driver** 12.8+ (for `tcgen05` + `16U4` TMA data types) with `libnvrtc`, Rust 1.75+.
- **For the GPU-less plane** (compile-check + tests): no GPU, no driver — just `libnvrtc.so.12`, e.g. `pip install nvidia-cuda-nvrtc-cu12` (CUDA 12.8+).

The CUDA libraries are loaded at runtime (dlopen); make sure the loader can find them:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:$LD_LIBRARY_PATH
# or, without any CUDA install:
#   pip install nvidia-cuda-nvrtc-cu12   # then either set LD_LIBRARY_PATH to its
#   lib dir, or DG_NVRTC_PATH=/path/to/libnvrtc.so.12 — the binaries probe
#   pip wheel locations automatically.
```

## Running without a GPU (sandbox / CI / laptop)

Two mechanisms, both used by this project's own development sandbox:

**1. Offline kernel validation — `compile-check`.** NVRTC never touches the
GPU, so every kernel variant is compiled to **PTX** (`compute_100a` — the
exact runtime JIT path) *and* to **SASS** (`sm_100a`, full ptxas backend via
`nvrtcGetCUBIN`). If it compiles here, it compiles on the B200:

```bash
pip install nvidia-cuda-nvrtc-cu12
cargo run -p deepgemm-bench --release -- compile-check            # sm_100a
#   --arch 103a   # B300
#   --arch 120a   # SM120 (RTX 50-class)
cargo run -p deepgemm-bench --release -- smoke   # on a GPU-less machine this
                                                 # auto-falls back to compile-check
```

Current status: **14/14 variants, PTX + SASS, in ~4s** (covers every TU and
code path: mxf4 / mxf8f6f4 / f16 MMAs, UTCCP SF paths, swap-AB, m-grouped
contiguous+masked, batched, MQA in FP8 and FP4, quant/transform_sf).

**2. CPU golden model — `cargo test`.** `deepgemm/src/golden.rs` reimplements
the numeric semantics bit-exactly (E4M3/E2M1/UE8M0, RNE quantization, SF
packing, block-scaled GEMM, MQA logits). 18 tests run anywhere; on a B200 the
e2e feature additionally cross-checks the *hardware* against the same model
(including a byte-exact `transform_sf` comparison).

```bash
cargo test --workspace        # golden + heuristics tests, no GPU needed
```

## Build & smoke test (no GPU code compiled at build time)

```bash
cargo build --release
cargo run -p deepgemm-bench --release -- list
cargo run -p deepgemm-bench --release -- smoke   # JIT-compiles every kernel variant for this GPU
                                               # (falls back to offline compile-check without a driver)
```

## Correctness tests

```bash
cargo test --workspace                      # CPU plane: golden model + heuristics (no GPU)
cargo test -p deepgemm --features e2e --release -- --nocapture   # B200 plane: GPU vs golden
```

Covers: quant/transform_sf round-trips, `fp8_gemm_nt` (gran 32 & 128), **native `fp4_gemm_nt`**, `bf16_gemm_nt`, m-grouped contiguous & masked, `mqa_logits` — each against the bit-exact CPU golden model (`src/golden.rs`), plus a byte-exact `transform_sf` GPU-vs-model comparison.

## Benchmarks (TFLOPS + % of peak)

```bash
cargo run -p deepgemm-bench --release -- bench fp4_nt_native --m 8192 --n 8192 --k 7168
cargo run -p deepgemm-bench --release -- bench fp8_nt        --m 8192 --n 8192 --k 7168
cargo run -p deepgemm-bench --release -- bench fp8_nt_g128   --m 4096 --n 4096 --k 7168   # DeepSeek 1x128/128x128 recipe
cargo run -p deepgemm-bench --release -- bench fp8_m_grouped_contiguous --m 8192 --n 2048 --k 7168 --groups 128
cargo run -p deepgemm-bench --release -- bench fp8_m_grouped_masked      --m 128  --n 2048 --k 7168 --groups 128
cargo run -p deepgemm-bench --release -- bench bf16_nt      --m 8192 --n 8192 --k 7168
cargo run -p deepgemm-bench --release -- bench fp8_bmm      --m 1024 --n 1024 --k 1024 --groups 8
cargo run -p deepgemm-bench --release -- bench mqa_logits_fp8
```

### Tuning the tiling on real hardware

Every launch consults the heuristics, which can be overridden by environment variables — the intended workflow for closing the last % toward upstream DeepGEMM:

| Variable | Meaning |
|---|---|
| `DG_BLOCK_M`, `DG_BLOCK_N`, `DG_BLOCK_K` | pin the tile shape |
| `DG_CLUSTER_M`, `DG_CLUSTER_N` | pin cluster dims (1 or 2) |
| `DG_SWAP_AB` | force swap-AB on/off |
| `DG_NUM_STAGES` | pin pipeline depth |
| `DG_PRINT_CONFIGS=1` | print the chosen config to stderr |

Example sweep:

```bash
for bm in 128 256; do for bn in 128 192 256; do
  DG_BLOCK_M=$bm DG_BLOCK_N=$bn DG_PRINT_CONFIGS=1 \
    cargo run -q -p deepgemm-bench --release -- bench fp4_nt_native --m 8192 --n 8192 --k 7168
done; done
```

Once you find the best tiling for your exact shapes, hard-code it in `deepgemm/src/heuristics.rs` (`layout_candidates`).

## Layout & API conventions (matching upstream)

- Inputs are NT-logical: `D = A @ B^T`; A is `[M, K]` K-major, B is `[N, K]` K-major (MN-major operands are supported by the kernel; the safe API currently exposes K-major).
- FP8/FP4 operands require **packed UE8M0** scale factors: `[ceil(K / gran / 4), TMA-aligned(M or N)]` int32, 4 exponents per word. Build them from FP32 scales with `transform_sf`, or quantize activations end-to-end with `quant_mx`.
- `m_grouped_*_contiguous`: `m_indices[i]` = expert id of the `i`-th 128-row block (negative = padding).
- `m_grouped_*_masked`: per-group stacked A/B/D; `masked_m[g]` = valid rows of group g.
- C/D: BF16 or FP32, N-major.

```rust
use deepgemm::prelude::*;

let dev = Device::new(0)?;
let stream = DevStream::new(&dev)?;

let x: Vec<f32> = ...;                       // activations [M, K]
let (a_data, a_sf) = quant_mx(&dev, &stream, &x, m, k, Dtype::Fp4, SfGran::G32)?;
let a = Operand { dtype: Dtype::Fp4, major: Major::K, rows: m, k,
                  outer_stride: k, sf: Some(a_sf), data: a_data };
// ... likewise B (weights, pre-quantized MXFP4 on disk) ...
let mut out = Output { dtype: Dtype::Bf16, rows: m, cols: n, stride: n, data: out_buf };
fp4_gemm_nt(&dev, &stream, &a, &b, &mut out, false)?;
```

## Project layout

```
docs/
  concepts.md      concept guide (this repo's "textbook") with diagrams
  diagrams-src/    Mermaid sources   ── regen: python3 scripts/gen_diagrams.py
  img/             rendered SVG diagrams
deepgemm/            the library crate
  src/
    sys.rs           thin checked wrappers over the CUDA driver (via cudarc's dlopen'd libcuda)
    device.rs        device context, streams, buffers
    golden.rs        bit-exact CPU golden model (formats, SF packing, GEMM, MQA)
    jit.rs           NVRTC compile + disk cache + cuLaunchKernelEx (cluster/PDL)
                   + offline compile-check (PTX & SASS, no GPU)
    tma.rs           cuTensorMapEncodeTiled builders (A/B/SF/CD, 2D/3D, swizzles, 16U4)
    heuristics.rs    port of upstream SM100 config search + DG_* overrides
    api.rs           public GEMM / MQA / transform / quant entry points
    types.rs         dtypes, majors, operand/output descriptors
  kernels/           NVRTC CUDA C++ (zero #includes; all exotic ops are inline PTX)
                   each file opens with a concept tutorial (pipelines, warp maps,
                   barrier topology, SF packing diagrams)
    prelude.h        barriers, TMA, UMMA/tcgen05 descriptors & ops, scheduler, conversions
    gemm_sm100.cu    the unified FP8/FP4/BF16 GEMM + epilogues (swap-AB & normal)
    layout_quant.cu  transform_sf, MX quantization, dequant
    mqa_logits_sm100.cu  MLA lightning-indexer scoring
  tests/
    golden_tests.rs  CPU-plane tests (run anywhere)
    e2e_gpu.rs       B200-plane tests (feature e2e; GPU vs golden, bit-exact SF check)
deepgemm-bench/      benchmark CLI (bench / smoke / compile-check / list)
```

## Relation to upstream / not-yet-ported

Ported with full fidelity: the unified SM100 GEMM (all block-scaled paths), MQA logits (contiguous KV), transform_sf, quant, heuristics.

Not yet wired (PRs welcome): SM90 (Hopper WGMMA) kernels, paged-MQA scheduling metadata kernel, MegaMoE/MegaGate (DeepEP-coupled megakernels), k-grouped/psum layouts (weight-grad), FP8 dynamic-output epilogue (QuantizeToFP8), stochastic-rounding epilogue. The kernel templates are parameterized so these slot in without restructuring.

## License

MIT, following upstream DeepGEMM.
