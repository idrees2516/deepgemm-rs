#!/usr/bin/env python3
"""Standalone offline NVRTC verification for the SM90 FP8 1D2D GEMM port.

Concatenates prelude.h + wgmma.h + kernels/gemm_sm90_1d2d.cu with the exact
`__dg_kernel` wrapper that `deepgemm/src/api_1d2d.rs::wrapper_body` builds,
compiles each variant to PTX (compute_90a — the runtime JIT path) and SASS
(sm_90a via nvrtcGetCUBIN, the ptxas backend), and prints sizes.

Run: python3 scripts/verify_1d2d.py
"""
import ctypes
import sys

NVRTC = "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
ROOT = "/home/z/my-project/deepgemm-rs/deepgemm/kernels/"

prelude = open(ROOT + "prelude.h").read()
wgmma = open(ROOT + "wgmma.h").read()
tu = open(ROOT + "gemm_sm90_1d2d.cu").read()

# (name, sfb_mn, bm, bn, swd, stages, math, mcast, mcoa, sms)
# smem_size is a launch-time reservation (not compile-visible), listed for
# reference only. Mirrors the hand-picked variants in tests/one2d_cpu.rs.
CONFIGS = [
    # chooser-realistic tiles (k=7168 stage budgets)
    ("chooser-ish 64x128 sw128",      0, 64, 128, 128, 8, 128, 1, 0, 132),
    ("chooser-ish 128x192 sw128",     0, 128, 192, 128, 4, 256, 1, 0, 132),
    # straddle ladder + both SFB majors
    ("straddle-96 sw64",              1, 64, 96, 64, 8, 128, 1, 0, 132),
    ("straddle-160 sw64",             1, 128, 160, 64, 6, 256, 1, 0, 132),
    # tiny tile (single store warp) + uniform scales
    ("uniform-16 sw32",               0, 16, 16, 32, 16, 128, 1, 0, 132),
    # two WGMMA waves per warpgroup + cluster-2 multicast on A
    ("two-waves 256x64 mcast",        0, 256, 64, 128, 4, 256, 2, 1, 132),
    # unswizzled row-major D staging (padding epilogue path)
    ("unswizzled-d 64x128",           0, 64, 128, 0, 8, 128, 1, 0, 132),
]

WRAPPER = r'''
extern "C" __global__ void __dg_kernel(
    const float* sfb, int* grouped_layout,
    unsigned m, unsigned n, unsigned k,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d, const __grid_constant__ dg::TmaMap tma_sfa) {
    dg::sm90_fp8_gemm_1d2d_impl<SFB_MN, 0, 0, 0, 1,
        BM, BN, 128, 128, 128, SWD,
        STAGES, 128, MATH, MCAST, MCOA, SMS,
        (dg::GemmType)0, 1>
        (sfb, grouped_layout, m, n, k, tma_a, tma_b, tma_d, tma_sfa);
}
'''


def build_wrapper(cfg):
    (sfb_mn, _bm, _bn, _swd, stages, math, mcast, mcoa, sms) = cfg[1:]
    return (WRAPPER
            .replace("SFB_MN", str(sfb_mn))
            .replace("BM", str(cfg[2]))
            .replace("BN", str(cfg[3]))
            .replace("SWD", str(cfg[4]))
            .replace("STAGES", str(stages))
            .replace("MATH", str(math))
            .replace("MCAST", str(mcast))
            .replace("MCOA", str(mcoa))
            .replace("SMS", str(sms)))


def compile_variant(name, cfg):
    src = (prelude + "\n" + wgmma + "\n" + tu + "\n" + build_wrapper(cfg)).encode()
    results = []
    for arch, getter in ((b"compute_90a", "ptx"), (b"sm_90a", "cubin")):
        prog = ctypes.c_void_p()
        rc = l.nvrtcCreateProgram(ctypes.byref(prog), src, b"verify_1d2d.cu", 0, None, None)
        assert rc == 0, rc
        opts = (ctypes.c_char_p * 2)(ctypes.c_char_p(b"--gpu-architecture=" + arch),
                                      ctypes.c_char_p(b"--std=c++17"))
        rc = l.nvrtcCompileProgram(prog, 2, opts)
        if rc != 0:
            n = ctypes.c_size_t()
            l.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value + 1)
            l.nvrtcGetProgramLog(prog, buf)
            log = buf.value.decode(errors="replace")
            print(f"FAIL [{name}] {arch.decode()}:\n" + "\n".join(
                ln for ln in log.splitlines() if "error" in ln or "note" in ln)[:4000])
            sys.exit(1)
        if getter == "ptx":
            n = ctypes.c_size_t()
            l.nvrtcGetPTXSize(prog, ctypes.byref(n))
            results.append(n.value)
        else:
            n = ctypes.c_size_t()
            l.nvrtcGetCUBINSize(prog, ctypes.byref(n))
            results.append(n.value)
        l.nvrtcDestroyProgram(ctypes.byref(prog))
    return results


if __name__ == "__main__":
    l = ctypes.CDLL(NVRTC)
    major, minor = ctypes.c_int(), ctypes.c_int()
    l.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    print(f"NVRTC {major.value}.{minor.value}; TU = prelude.h + wgmma.h + gemm_sm90_1d2d.cu")
    total_ptx = total_sass = 0
    for cfg in CONFIGS:
        ptx, sass = compile_variant(cfg[0], cfg)
        total_ptx += ptx
        total_sass += sass
        print(f"OK   [{cfg[0]:<28}] sfb_mn={cfg[1]} bm={cfg[2]:>3} bn={cfg[3]:>3} "
              f"swd={cfg[4]:>3} stages={cfg[5]:>2} math={cfg[6]} cluster={cfg[7]} "
              f"-> PTX {ptx:>8} B, SASS {sass:>7} B")
    print(f"\n{len(CONFIGS)} variants: PTX total {total_ptx} B, SASS total {total_sass} B — all green")
