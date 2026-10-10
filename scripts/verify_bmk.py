#!/usr/bin/env python3
"""Standalone NVRTC check for the bmk/bnk + PsumLayout kernels (no cargo).

Concatenates prelude.h (+ wgmma.h for the SM90 variant) + bmk_bnk.cu + an
`extern "C" __dg_kernel` wrapper (identical shape to the Rust launchers in
src/api_bmk.rs), then compiles each variant for BOTH the PTX pass
(compute_90a / compute_100a — what the runtime JIT does) and the SASS pass
(sm_90a / sm_100a — the ptxas backend via nvrtcGetCUBIN).  Prints sizes.

Variants (template parameters mirror the launcher tile choices):
  * sm100 bmk        — tcgen05 split-K + TMA reduce-add (100a)
  * sm90  bmk        — wgmma split-K + red.global.add.v2.f32 (90a)
  * psum m-grouped   — MGroupedContiguousWithPsumLayout, BF16 out (100a)
  * psum k-grouped   — KGroupedContiguousWithPsumLayout, FP32 accumulate (100a)
  * psum k-grouped   — ... direct BF16 out, k_alignment 384 (100a)

Usage:  python3 scripts/verify_bmk.py
"""
import ctypes
import sys

NVRTC = "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
KERNELS = "/home/z/my-project/deepgemm-rs/deepgemm/kernels/"

# `bmk_bnk_mn_sm{90,100}_impl` keep the problem shapes as template params
# (upstream compiles them into the JIT variant); `gemm_psum_impl` is the
# persistent psum scheduler consumer.  kNumStages 6 = the B200 budget choice
# of api_bmk::psum_tile (232448 B capacity).
WRAPPER = r'''
extern "C" __global__ void __dg_kernel(
    {params}) {{
    dg::{impl}<{tpl}>
        ({args});
}}
'''

VARIANTS = [
    # (name, arches, extra TU header, impl, params, tpl, args)
    (
        "sm100 bmk   s=4096 m/n/k=256",
        ["100a"],
        None,
        "bmk_bnk_mn_sm100_impl",
        "unsigned shape_s,\n    const __grid_constant__ dg::TmaMap tma_a, "
        "const __grid_constant__ dg::TmaMap tma_b,\n    const __grid_constant__ dg::TmaMap tma_d",
        "256, 256, 256, 128, 128, 64, 443, 128, 128, 4, 128",
        "shape_s, tma_a, tma_b, tma_d",
    ),
    (
        "sm90  bmk   s=4096 m/n/k=256",
        ["90a"],
        "wgmma.h",
        "bmk_bnk_mn_sm90_impl",
        "unsigned shape_s,\n    const __grid_constant__ dg::TmaMap tma_a, "
        "const __grid_constant__ dg::TmaMap tma_b,\n    float* d",
        "256, 256, 256, 128, 128, 64, 443, 4, 128, 256",
        "shape_s, tma_a, tma_b, d",
    ),
    (
        "psum m-grouped           g=8 ",
        ["100a"],
        None,
        "gemm_psum_impl",
        "unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,\n"
        "    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,\n"
        "    const __grid_constant__ dg::TmaMap tma_cd",
        "128, 128, 64, 8, 128, 128, 6, 256,\n"
        "                       (dg::psum::GemmType)5, false, false, 128, 148, false",
        "shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd",
    ),
    (
        "psum k-grouped acc  align=256",
        ["100a"],
        None,
        "gemm_psum_impl",
        "unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,\n"
        "    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,\n"
        "    const __grid_constant__ dg::TmaMap tma_cd",
        "128, 128, 64, 16, 128, 128, 6, 256,\n"
        "                       (dg::psum::GemmType)6, true, true, 256, 148, false",
        "shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd",
    ),
    (
        "psum k-grouped bf16 align=384",
        ["100a"],
        None,
        "gemm_psum_impl",
        "unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,\n"
        "    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,\n"
        "    const __grid_constant__ dg::TmaMap tma_cd",
        "128, 128, 64, 16, 128, 128, 6, 256,\n"
        "                       (dg::psum::GemmType)6, false, false, 384, 148, false",
        "shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd",
    ),
]


def main() -> int:
    l = ctypes.CDLL(NVRTC)
    prelude = open(KERNELS + "prelude.h").read()
    wgmma = open(KERNELS + "wgmma.h").read()
    bmk = open(KERNELS + "bmk_bnk.cu").read()

    failures = 0
    for name, arches, extra, impl, params, tpl, args in VARIANTS:
        tu = bmk if extra is None else wgmma + "\n" + bmk
        body = WRAPPER.format(params=params, impl=impl, tpl=tpl, args=args)
        src = (prelude + "\n" + tu + "\n" + body).encode()
        for arch in arches:
            for mode, opt_arch, getter in (
                ("PTX ", ("compute_" + arch).encode(), "ptx"),
                ("SASS", ("sm_" + arch).encode(), "cubin"),
            ):
                prog = ctypes.c_void_p()
                rc = l.nvrtcCreateProgram(ctypes.byref(prog), src, b"verify_bmk.cu", 0, None, None)
                assert rc == 0, rc
                opts = [b"--gpu-architecture=" + opt_arch, b"--std=c++17"]
                arr = (ctypes.c_char_p * len(opts))(*[ctypes.c_char_p(o) for o in opts])
                rc = l.nvrtcCompileProgram(prog, len(opts), arr)
                if rc != 0:
                    n = ctypes.c_size_t()
                    l.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
                    buf = ctypes.create_string_buffer(n.value + 1)
                    l.nvrtcGetProgramLog(prog, buf)
                    print(f"FAIL  {name} [{arch}] {mode}:")
                    printed = 0
                    for ln in buf.value.decode(errors="replace").splitlines():
                        if ("error" in ln or "^" in ln) and printed < 20:
                            print("   ", ln)
                            printed += 1
                    failures += 1
                else:
                    n = ctypes.c_size_t()
                    if getter == "ptx":
                        l.nvrtcGetPTXSize(prog, ctypes.byref(n))
                    else:
                        l.nvrtcGetCUBINSize(prog, ctypes.byref(n))
                    print(f"OK    {name} [{arch}] {mode}: {n.value:>8d} B")
                l.nvrtcDestroyProgram(ctypes.byref(prog))

    if failures:
        print(f"\n{failures} compilation(s) FAILED")
        return 1
    print("\nall bmk/psum variants: PTX + SASS green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
