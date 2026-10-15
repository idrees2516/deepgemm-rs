# Changelog

## 0.2.0 — production readiness wave

### Added
- **BLAS layout parity**: `gemm_nn` / `gemm_tn` / `gemm_tt` (+ FP8/BF16
  aliases) and `m_grouped_gemm_nn_contiguous` — zero-copy MN-major operand
  views routed into the unified kernel, mirroring upstream's host-side
  transpose wrappers. `Operand::k_major`/`mn_major` constructors document
  the view contract.
- **Alpha epilogue** (`gemm_nt_alpha`): BLAS-style `D = alpha·(A@B) + C`
  product-term scaling in the FP32 accumulator (upstream
  `epilogue::operators::ScaleByAlpha`), both store paths, exclusive with
  the fused FP8 output.
- **SM100 K-grouped weight-grad GEMMs**: `k_grouped_fp8_gemm_tn_contiguous`
  (MN-major shared arenas) and `k_grouped_fp4_gemm_nt_contiguous` (packed
  K-major), with the per-group-padded SF packing kernel
  (`transform_sf_k_grouped`) and the scheduler's `current_sf_k_cumsum`
  addressing ported from upstream.
- **Einsum API** (`api_einsum`): `bmk,bnk->mn` (split-K kernel + RNE BF16
  cast), `bhr,hdr->bhd` and `bhd,hdr->bhr` (strided 3D TMA maps over the
  user's tensors — zero copies), and the FP8 flavors of all three
  equations with the exact upstream scale-factor contracts.
- **cuBLASLt layer** (`cublaslt`): `cublaslt_gemm_nt` (+ nn/tn/tt via
  `cublaslt_gemm`), NVFP4 block-scaled reference, and the einsum
  `bhd,bhr->hdr` batch-reduce (the small-K case upstream also delegates).
- **Serving composites** (`api_serving`): `Serving` handle (lane pool +
  workspace cache) with DSv4.1 MLA scoring, grouped MoE layer, GLM-5.3
  attention projections, and startup `warmup()` (JIT pre-compile).
- **Benchmark harness**: `sweep` (shape grids, TFLOPS / %peak / roofline,
  JSON snapshots), `verify` (golden-tolerance correctness gate),
  `autotune` (DG_* grid search), `compare` (regression tracking).
- **Upstream-parity test suite** (`parity_tests`): the upstream
  `generators.py` enumerations (normal / m-grouped contiguous + masked /
  k-grouped) asserted against the heuristics (no-gaps coverage), the
  quantization tolerance model on the exact `calc_diff` cosine metric, and
  seeded fuzz invariants.
- **CI** (`.github/workflows/ci.yml`): fmt + clippy `-D warnings`, host
  tests, and offline PTX+SASS compile-check matrices for 90a/100a/103a on
  GPU-less runners.
- **Docs**: `docs/PRODUCTION.md` (deployment guide), `docs/BENCHMARKING.md`
  (acceptance + tuning workflow), this changelog.

### Fixed
- **Batched CD TMA map**: the 3D map's batch stride was `cols·elem` instead
  of `rows·row_stride·elem` — every Batched GEMM CD store landed in the
  wrong slice for m > 1. (`make_tma_cd_3d`.)
- **90a compile-check**: the mega gate/mhc variants (SM100-only kernels)
  were wrongly enumerated for SM90 (2 pre-existing failures).
- Scheduler: K-grouped `get_global_idx` branch (MN/K/SF_K offsets),
  first-group constructor init, and the SF-row cumsum at group transitions.

### Changed
- `run_gemm` internally threads `alpha` + `k_alignment`; all public
  signatures keep their shapes (new capabilities are additive entries).
- 36/36 compile-check variants on 100a and 103a; 22/22 on 90a.

## 0.1.0 — initial port

Unified SM100 FP8/FP4/BF16 GEMM, m-grouped GEMMs, batched GEMM, MQA logits
(contiguous + paged), transform_sf + MX quantization, heuristics, NVRTC JIT
with disk cache, CPU golden model, offline compile-check, SM90 suite,
sparse MQA, hc-prenorm, locality domain, BMK/BNK + psum kernels, and the
mega family (fp8_fp4_mega_moe, fp8 mega_gate, mega_mhc).
