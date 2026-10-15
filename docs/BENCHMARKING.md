# Benchmarking & Tuning Guide

The bench CLI (`deepgemm-bench`) drives the full workflow. Everything runs
on-device; the correctness gate and performance sweep share the same shape
grids (drawn from the upstream test generators plus the DSv4/GLM model dims).

## The acceptance workflow (B200 / B300)

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 0. (GPU-less) SASS gate — optional, also what CI runs
cargo run -p deepgemm-bench --release -- compile-check --arch 100a   # or 103a

# 1. JIT smoke: compile every variant for THIS device
cargo run -p deepgemm-bench --release -- smoke

# 2. Correctness gate — golden-model tolerances, nonzero exit on failure
cargo run -p deepgemm-bench --release -- verify

# 3. Performance sweep — TFLOPS / %peak / roofline, JSON snapshot
cargo run -p deepgemm-bench --release -- sweep --family all --out run1.json

# 4. Per-shape autotune — DG_* grid search, prints the winning env line
cargo run -p deepgemm-bench --release -- autotune fp8_nt --m 8192 --n 7168 --k 7168

# 5. Regression tracking — compare two snapshots (flags > 3% drops)
cargo run -p deepgemm-bench --release -- compare run1.json run2.json
```

## What `verify` checks

Every family's grid runs against the CPU golden model (bit-exact
quantization + f64 accumulation) with the **upstream tolerance table**
(cosine metric, `1 - 2<x,y>/(|x|²+|y|²)`): FP4xFP4 ≤ 0.02, FP4-mixed ≤ 0.01,
FP8 ≤ 0.001. Coverage: dense FP8 (G32/G128), native FP4, BF16, the nn/tt
MN-major layouts, batched BMM (per-batch slice correctness), m-grouped
contiguous + masked.

## Reading the sweep table

```
bench                          m      n      k   g    TFLOPS   %peak   intens      bound       ms
fp8_nt                       8192   7168   7168   1     1587.2    70.5%    220.5   compute   3.12
fp8_m_grouped_contiguous     1024   6144   7168   8      940.1    41.8%     95.2   balanced  1.87
```

- **TFLOPS** — effective rate over the sweep iteration average.
- **%peak** — fraction of the device's dense-peak FLOPs for that dtype
  (`Device::peak_tflops`, B200: FP8 ≈ 2.25 PF dense).
- **intens / bound** — model arithmetic intensity (FLOP per DRAM byte,
  including SF traffic) and the roofline classification at the B200 balance
  point (~281 FLOP/byte). Decode shapes (m=1..128) classify `dram` — judge
  them on GB/s, not TFLOPS.

## Autotuning into the heuristics

`autotune` grid-searches `DG_BLOCK_M/N`, `DG_NUM_STAGES`, `DG_CLUSTER_M`
for one shape and prints the winner as an export line. The intended loop:

1. `autotune` the shapes that matter for your traffic mix.
2. If an override wins consistently (> 5%), fold it into
   `deepgemm/src/heuristics.rs` — the comparator is a line-by-line port of
   upstream's, so upstream's tuned defaults are the baseline.
3. Re-run `verify` (the override must not break correctness — the kernel
   asserts catch illegal tilings at compile time) and `compare` for the
   before/after.

## Mega kernels

`bench mega_moe_fp8` / `mega_gate_fp8` / `mega_mhc` time the megakernels
with representative DSv4 shapes (effective TFLOPS include the fused
dispatch/combine work). These are persistent kernels — they occupy all SMs
for the whole layer; benchmark them with realistic token counts
(`--groups` = experts, m = tokens) rather than tiny sizes.
