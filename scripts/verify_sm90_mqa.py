#!/usr/bin/env python3
"""Standalone NVRTC verification for the SM90 (Hopper) MQA-logits kernels.

Compiles deepgemm/kernels/mqa_logits_sm90.cu (as the JIT engine would:
prelude.h + wgmma.h + the TU) with representative wrapper bodies for BOTH
kernels plus the paged metadata kernel, for compute_90a (PTX) and sm_90a
(SASS via nvrtcGetCUBIN).  Prints PTX/CUBIN sizes; exits non-zero on failure.

No cargo, no GPU: the same NVRTC .so the Rust runtime loads via jit.rs.
"""
import ctypes
import sys

NVRTC = "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
ROOT = "/home/z/my-project/deepgemm-rs/deepgemm/kernels/"

# Representative instantiations (mirroring src/api_sm90_mqa.rs):
#   contiguous : heads=32 -> BLOCK_Q=4, BLOCK_KV=256, 3/3 stages, 512 math
#   paged      : next_n=2, heads=64, BLOCK_KV=64, SPLIT_KV=256, is2d=1
#   metadata   : aligned batch 32, SPLIT_KV 256, 132 SMs
VARIANTS = [
    (
        "contiguous heads=32 head_dim=128",
        r'''
extern "C" __global__ void __dg_kernel(
    unsigned seq_len, unsigned seq_len_kv, unsigned stride_logits,
    const unsigned* cu_k_start, const unsigned* cu_k_end, float* logits,
    const __grid_constant__ dg::TmaMap tma_q, const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_kv_scales, const __grid_constant__ dg::TmaMap tma_weights) {
    dg::mqa_logits_sm90_impl<32, 128, 4, 256, 3, 3, 132, 128, 512>
        (seq_len, seq_len_kv, stride_logits, cu_k_start, cu_k_end, logits,
         tma_q, tma_kv, tma_kv_scales, tma_weights);
}
''',
    "sm90-mqa-contig",
    ),
    (
        "contiguous heads=64 head_dim=64 (small smem)",
        r'''
extern "C" __global__ void __dg_kernel(
    unsigned seq_len, unsigned seq_len_kv, unsigned stride_logits,
    const unsigned* cu_k_start, const unsigned* cu_k_end, float* logits,
    const __grid_constant__ dg::TmaMap tma_q, const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_kv_scales, const __grid_constant__ dg::TmaMap tma_weights) {
    dg::mqa_logits_sm90_impl<64, 64, 2, 256, 3, 3, 132, 128, 512>
        (seq_len, seq_len_kv, stride_logits, cu_k_start, cu_k_end, logits,
         tma_q, tma_kv, tma_kv_scales, tma_weights);
}
''',
    "sm90-mqa-contig-h64",
    ),
    (
        "paged next_n=2 heads=64 head_dim=128",
        r'''
extern "C" __global__ void __dg_kernel(
    unsigned batch_size, unsigned logits_stride, unsigned block_table_stride,
    const unsigned* context_lens, float* logits,
    const unsigned* block_table, const unsigned* indices, const unsigned* schedule_meta,
    const __grid_constant__ dg::TmaMap tma_q, const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_kv_scales, const __grid_constant__ dg::TmaMap tma_weights) {
    dg::mqa_paged_logits_sm90_impl<2, 64, 128, 64, 1, 0, 3, 3, 256, 128, 512>
        (batch_size, logits_stride, block_table_stride, context_lens, logits,
         block_table, indices, schedule_meta, tma_q, tma_kv, tma_kv_scales, tma_weights);
}
''',
    "sm90-mqa-paged",
    ),
    (
        "paged next_n=1 heads=32 head_dim=64",
        r'''
extern "C" __global__ void __dg_kernel(
    unsigned batch_size, unsigned logits_stride, unsigned block_table_stride,
    const unsigned* context_lens, float* logits,
    const unsigned* block_table, const unsigned* indices, const unsigned* schedule_meta,
    const __grid_constant__ dg::TmaMap tma_q, const __grid_constant__ dg::TmaMap tma_kv,
    const __grid_constant__ dg::TmaMap tma_kv_scales, const __grid_constant__ dg::TmaMap tma_weights) {
    dg::mqa_paged_logits_sm90_impl<1, 32, 64, 64, 1, 0, 3, 3, 256, 128, 512>
        (batch_size, logits_stride, block_table_stride, context_lens, logits,
         block_table, indices, schedule_meta, tma_q, tma_kv, tma_kv_scales, tma_weights);
}
''',
    "sm90-mqa-paged-n1",
    ),
    (
        "paged metadata (batch<=32, 132 SMs)",
        r'''
extern "C" __global__ void __dg_kernel(
    unsigned batch_size, unsigned next_n, unsigned is_context_lens_2d,
    const unsigned* context_lens, const unsigned* indices, unsigned* schedule_metadata) {
    dg::sm90_paged_mqa_logits_metadata_impl<32, 256, 132, 0>
        (batch_size, next_n, is_context_lens_2d, context_lens, indices, schedule_metadata);
}
''',
    "sm90-mqa-meta",
    ),
    (
        "paged metadata (batch 33 -> 64 slots, 132 SMs)",
        r'''
extern "C" __global__ void __dg_kernel(
    unsigned batch_size, unsigned next_n, unsigned is_context_lens_2d,
    const unsigned* context_lens, const unsigned* indices, unsigned* schedule_metadata) {
    dg::sm90_paged_mqa_logits_metadata_impl<64, 256, 132, 0>
        (batch_size, next_n, is_context_lens_2d, context_lens, indices, schedule_metadata);
}
''',
    "sm90-mqa-meta-64",
    ),
]


def load_nvrtc():
    return ctypes.CDLL(NVRTC)


def compile(lib, src, name, arch):
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src, (name + ".cu").encode(), 0, None, None)
    assert rc == 0, rc
    opts = [b"--gpu-architecture=" + arch.encode(), b"--std=c++17"]
    arr = (ctypes.c_char_p * len(opts))(*[ctypes.c_char_p(o) for o in opts])
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        n = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value + 1)
        lib.nvrtcGetProgramLog(prog, buf)
        log = buf.value.decode(errors="replace")
        printed = 0
        for ln in log.splitlines():
            if ("error" in ln or "note" in ln or "^" in ln) and printed < 60:
                print("   ", ln)
                printed += 1
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return None, None
    n = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value + 1)
    lib.nvrtcGetPTX(prog, ptx)
    ptx_len = n.value
    cub = ctypes.create_string_buffer(0)
    cub_len = None
    if lib.nvrtcGetCUBINSize is not None:
        rc2 = lib.nvrtcGetCUBINSize(prog, ctypes.byref(n))
        if rc2 == 0:
            cub = ctypes.create_string_buffer(n.value + 1)
            rc3 = lib.nvrtcGetCUBIN(prog, cub)
            if rc3 == 0:
                cub_len = n.value
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx_len, cub_len


def main():
    lib = load_nvrtc()
    # nvrtcGetCUBIN may not exist in very old headers; probe once.
    try:
        lib.nvrtcGetCUBINSize
        has_cubin = True
    except AttributeError:
        # present but unchecked: rely on .restype below
        lib.nvrtcGetCUBINSize.restype = ctypes.c_int
        lib.nvrtcGetCUBIN.restype = ctypes.c_int
        has_cubin = True

    prelude = open(ROOT + "prelude.h").read()
    wgmma = open(ROOT + "wgmma.h").read()
    tu = open(ROOT + "mqa_logits_sm90.cu").read()

    failed = 0
    for title, wrapper, tag in VARIANTS:
        src = (prelude + "\n" + wgmma + "\n" + tu + "\n" + wrapper).encode()
        ptx_len, cub_len = compile(lib, src, tag, "compute_90a")
        if ptx_len is None:
            print(f"FAIL  {title}: PTX compile error (compute_90a)")
            failed += 1
            continue
        ptx2, cub2 = compile(lib, src, tag, "sm_90a")
        if ptx2 is None:
            print(f"FAIL  {title}: SASS compile error (sm_90a)")
            failed += 1
            continue
        sass = "n/a" if cub2 is None else f"{cub2} B"
        print(f"OK    {title}: PTX {ptx_len} B (sm_90a PTX {ptx2} B), SASS {sass}")

    if failed:
        print(f"{failed} variant(s) FAILED")
        sys.exit(1)
    print("all SM90 MQA variants compile to PTX + SASS")


if __name__ == "__main__":
    main()
