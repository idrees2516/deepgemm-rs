# Production Deployment Guide

This guide covers running DeepGEMM-RS in a production inference stack
(DeepSeek-V4.1-class and GLM-5.3-class models) on B200 (SM100) and B300
Blackwell Ultra (SM103).

## Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| GPU | B200 / GB200 (SM100a) or B300 (SM103a) | `tcgen05` tensor cores + `16U4` TMA |
| CUDA driver | 12.8+ | loaded at runtime via dlopen; no toolkit needed at build time |
| NVRTC | 12.x | `pip install nvidia-cuda-nvrtc-cu12` or the toolkit's — probed at runtime |
| cuBLASLt | 12.x | optional; the einsum `bhd,bhr->hdr` path + reference GEMMs |

No CUDA toolkit is needed at **build** time — kernels are JIT-compiled from
embedded CUDA C++ source via NVRTC on first use, exactly like upstream
DeepGEMM's DeepJIT model.

## Startup sequence (the order matters)

```text
1. Device::new(0)                          — bind the driver, read SM count/arch
2. Serving::new(device)                    — stream pool + workspace cache
3. serving.warmup()                        — NVRTC-compiles the model's kernel
                                             variants (also disk-cached in
                                             ~/.cache/deepgemm-rs across restarts)
4. serving.<layer calls>                   — steady state: zero compile latency,
                                             zero allocations per token
```

The JIT disk cache is keyed by (kernel source hash, template signature,
arch). A rolling deploy on identical hardware gets cold-start compile hits
from the cache; the first deploy pays ~10-30 s of one-time compilation.

## Precision recipes

Every GEMM data plane runs FP8 (E4M3 + UE8M0) or FP4 (E2M1 + UE8M0):

- **G128** (DeepSeek recipe): one UE8M0 scale per 128 K elements — the
  `transform_sf` packing covers `[ceil(k/128/4), align(mn,4)]`.
- **G32** (OCP MX): one scale per 32 elements — the native
  `tcgen05.mma.kind::mxf4`/`mxf8f6f4` block-scaled path.
- MoE expert weights are quantized per-expert-row; the megakernel path adds
  the gate/up interleave + UTCCP SF transpose (`interleave_weights`,
  `transpose_sf_utccp`).

Tolerances (the upstream `QuantConfig::max_diff()` contract, asserted by
`deepgemm-bench verify`): FP4xFP4 ≤ 0.02, mixed ≤ 0.01, FP8 ≤ 0.001 on the
cosine metric (`1 - 2<x,y>/(|x|²+|y|²)`).

## DSv4.1-class serving map

| Layer | Entry | Notes |
|-------|-------|-------|
| MLA indexer scores (decode) | `Serving::mla_scores` | weighted-ReLU MQA logits, the lightning-indexer kernel |
| MLA indexer scores (prefill) | `api::mqa_logits` directly | contiguous-KV flavor |
| Paged KV | `api::mqa_logits_paged` | cost-balanced metadata kernel first |
| Sparse KV (DSA) | `api_sparse_mqa::*` | top-k indexer + sparse scoring |
| Routed FFN (fused) | `api_mega_moe::mega_moe_fp8_fp4` | dispatch + 2 GEMMs + SwiGLU + combine on-SM |
| Routed FFN (grouped) | `Serving::moe_grouped_layer` | m-grouped contiguous GEMMs; activation by the caller |
| Router | `api::mega_gate` | FP8 megakernel gate (top-k + bias + normalization) |

## GLM-5.3-class (hyperconnection) serving map

| Layer | Entry | Notes |
|-------|-------|-------|
| Attention out-proj | `Serving::glm_out_proj` / `fp8_einsum_bhr_hdr_bhd` | `bhr,hdr->bhd` |
| Value projection | `Serving::glm_v_proj` / `fp8_einsum_bhd_hdr_bhr` | `bhd,hdr->bhr` |
| Weight-side batch reduce | `Serving::glm_wgrad_reduce` | `bhd,bhr->hdr` (cuBLASLt — small-K contract) |
| Hyperconnection pre-norm | `api_hc_prenorm::hc_prenorm_sm100` | TF32 + `sqr_sum` output |
| Hyperconnection megakernel | `api::mega_mhc` | fused Mix+Norm workers |
| Score projection | `api_einsum::einsum_bmk_bnk_mn` | split-K BMK/BNK kernel |

## Stream & memory discipline

- `Serving` owns a lane pool (`Serving::lane(i)`); independent projections
  should draw different lanes so the driver overlaps them.
- All kernels launch with PDL (programmatic dependent launch) enabled where
  the upstream kernels do — consecutive launches on one stream overlap
  prologue with the predecessor's epilogue automatically.
- Workspaces (MoE intermediate, logits) are sized once and reused; the
  `Runtime` layer adds size-classed scratch pooling for ad-hoc buffers.
- Buffers are raw device allocations — keep them alive as long as any
  in-flight kernel references them (the API launches asynchronously and
  returns immediately; `stream.sync()` is the caller's barrier).

## Environment knobs

| Variable | Effect |
|----------|--------|
| `DG_NVRTC_PATH` | exact `libnvrtc.so.12` path or its directory |
| `DG_BLOCK_M/N/K`, `DG_NUM_STAGES`, `DG_CLUSTER_M/N` | tiling overrides for autotuning |
| `DG_PRINT_CONFIGS=1` | print every chosen kernel config |
| `DG_TC_UTIL` | cap the persistent-kernel SM count |

## Determinism & safety

- Same inputs + same config ⇒ **bitwise-identical** outputs (asserted by
  `e2e_determinism_bitwise`); the only nondeterministic paths are the
  explicitly stochastic ones (stochastic BF16 rounding) and multi-rank
  NVLink reductions in the megakernels.
- The library never panics on user input: every public entry validates
  shapes/strides and returns `DgError` (invalid args surface as
  `DgError::InvalidArg` with the exact constraint in the message).
- Kernels are compiled per (shape-class, dtype, layout) signature; a bad
  shape combination fails at the API boundary, not inside the kernel.

## B300 (Blackwell Ultra) notes

SM103 is a first-class target: `compile-check --arch 103a` validates SASS
for every variant, and `Arch::Sm103` gates the same tcgen05 paths as B200.
The B300's larger HBM and SM count are picked up automatically from the
device query (heuristics re-enumerate tiling candidates per `num_sms`).
