#!/usr/bin/env python3
"""Standalone NVRTC check for the TF32 hc-prenorm kernels (no cargo needed).

Concatenates prelude.h (+ wgmma.h for the SM90 variant) + hc_prenorm.cu + an
`extern "C" __dg_kernel` wrapper (identical shape to the Rust launchers in
src/api_hc_prenorm.rs), then compiles each variant for BOTH the PTX pass
(compute_90a / compute_100a — what the runtime JIT does) and the SASS pass
(sm_90a / sm_100a — the ptxas backend via nvrtcGetCUBIN).  Prints sizes.

Usage:  python3 scripts/verify_hc_prenorm.py
"""
import ctypes
import sys

NVRTC = "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
KERNELS = "/home/z/my-project/deepgemm-rs/deepgemm/kernels/"

# Template parameters mirror the launcher's tile choice for the upstream
# reference shape (m=4096, n=24, k=7168, splits in {1, 16}):
#   block = (64, 32, 64), swizzle_cd = 128, stages = 12, threads = 128+128.
WRAPPER = r'''
extern "C" __global__ void __dg_kernel(
    unsigned shape_m,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, float* sqr_sum) {{
    dg::hc_prenorm_sm{arch}_impl<{n}, {k}, 64, {bn}, 64, {splits}, {swcd}, {stages}, 128, 128>
        (shape_m, tma_a, tma_b, tma_d, sqr_sum);
}}
'''

VARIANTS = [
    # (name, arches, extra TU header, template params)
    ("sm90  n=24 k=7168 splits=1 ", ["90a"], "wgmma.h", dict(arch=90, n=24, k=7168, bn=32, splits=1, swcd=128, stages=12)),
    ("sm90  n=24 k=7168 splits=16", ["90a"], "wgmma.h", dict(arch=90, n=24, k=7168, bn=32, splits=16, swcd=128, stages=12)),
    ("sm90  n=16 k=7680 splits=16", ["90a"], "wgmma.h", dict(arch=90, n=16, k=7680, bn=16, splits=16, swcd=64, stages=12)),
    ("sm100 n=24 k=7168 splits=1 ", ["100a"], None, dict(arch=100, n=24, k=7168, bn=32, splits=1, swcd=128, stages=12)),
    ("sm100 n=24 k=7168 splits=16", ["100a"], None, dict(arch=100, n=24, k=7168, bn=32, splits=16, swcd=128, stages=12)),
    ("sm100 n=16 k=7680 splits=16", ["100a"], None, dict(arch=100, n=16, k=7680, bn=16, splits=16, swcd=64, stages=12)),
]


def main() -> int:
    l = ctypes.CDLL(NVRTC)
    prelude = open(KERNELS + "prelude.h").read()
    wgmma = open(KERNELS + "wgmma.h").read()
    hc = open(KERNELS + "hc_prenorm.cu").read()

    failures = 0
    for name, arches, extra, params in VARIANTS:
        tu = hc if extra is None else wgmma + "\n" + hc
        body = WRAPPER.format(**params)
        src = (prelude + "\n" + tu + "\n" + body).encode()
        for arch in arches:
            for mode, opt_arch, getter in (
                ("PTX ", ("compute_" + arch).encode(), "ptx"),
                ("SASS", ("sm_" + arch).encode(), "cubin"),
            ):
                prog = ctypes.c_void_p()
                rc = l.nvrtcCreateProgram(ctypes.byref(prog), src, b"verify_hc.cu", 0, None, None)
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
    print("\nall hc-prenorm variants: PTX + SASS green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
