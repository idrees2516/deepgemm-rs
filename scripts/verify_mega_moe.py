#!/usr/bin/env python3
"""Standalone NVRTC compile check for the MegaMoE fp8xfp4 megakernel
(`kernels/mega_moe_sm100.cu`), no cargo / GPU driver needed.

Mirrors what the Rust launcher (`src/api_mega_moe.rs`) JITs at runtime:
prelude.h + kernel TU + `extern "C" __dg_kernel` wrapper that instantiates
`dg::mega_moe_fp8_fp4_impl<...>` with a realistic launch configuration
(the same selection logic as `heuristics/mega_moe.hpp`, mirrored in
`MegaMoeConfig::new`). Two variants are checked — single rank (the local
degenerate case) and 4 ranks (the NVLink multi-rank path) — each compiled
to PTX (compute_100a, the runtime path) and SASS (sm_100a, the ptxas
backend). Any NVRTC diagnostic is dumped in full.
"""
import ctypes
import sys

NVRTC = "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
ROOT = "/home/z/my-project/deepgemm-rs/deepgemm/kernels/"


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


# ---------------------------------------------------------------------------
# Layout math (exact mirrors of layout/mega_moe.cuh; see src/moe_layout.rs)
# ---------------------------------------------------------------------------

LCM_CANDIDATE_BLOCK_M = 1920
MIN_CANDIDATE_BLOCK_M = 8


def get_num_l1_warmup_waves(num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters):
    num_first_l2_wave_m_blocks = ceil_div(num_clusters, num_l2_n_clusters)
    num_l1_warmup_clusters_for_first_l2_wave = ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters)
    num_interleave_cluster_diff_per_m_block = (
        num_l1_n_clusters - num_l2_n_clusters) if num_l1_n_clusters > num_l2_n_clusters else 0
    num_warmup_waves_for_interleave_schedule = ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1
    return max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule)


def get_num_max_live_pool_blocks(num_total_m_blocks, num_sms, hidden, intermediate_hidden):
    block_n, ctas_per_cluster = 128, 2
    num_clusters = num_sms // ctas_per_cluster
    num_l1_n_clusters = intermediate_hidden * 2 // (ctas_per_cluster * block_n)
    num_l2_n_clusters = hidden // (ctas_per_cluster * block_n)
    num_l1_clusters = num_total_m_blocks * num_l1_n_clusters
    num_l1_waves = ceil_div(num_l1_clusters, num_clusters)
    num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters)
    num_l1_warmup_waves = min(num_min_l1_warmup_waves, num_l1_waves)
    num_l1_warmup_clusters = min(num_l1_warmup_waves * num_clusters, num_l1_clusters)
    num_live_blocks_after_warmup = ceil_div(num_l1_warmup_clusters, num_l1_n_clusters)
    frontier_growth = (ceil_div(num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters),
                                num_l2_n_clusters)
                       if num_l2_n_clusters > num_l1_n_clusters else 0)
    wave_margin = ceil_div(num_clusters, min(num_l1_n_clusters, num_l2_n_clusters))
    return min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin)


def ring_tokens(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                num_sms, hidden, intermediate_hidden, block_m):
    num_experts_per_rank = num_experts // num_ranks
    num_active_topk = min(num_topk, num_experts_per_rank)
    num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk
    num_ring_tokens = 0
    for bm in (8, 16, 32, 64, 96, 128, 192, 240):
        num_pool_blocks = ceil_div(num_max_routed_tokens, bm) + num_experts_per_rank
        num_live = get_num_max_live_pool_blocks(num_pool_blocks, num_sms, hidden, intermediate_hidden)
        num_ring_tokens = max(num_ring_tokens, num_live * bm)
    return align(num_ring_tokens, LCM_CANDIDATE_BLOCK_M)


def sf_ring_tokens(num_ring_tokens, block_m):
    return (num_ring_tokens // block_m) * align(block_m, 128)


# ---------------------------------------------------------------------------
# Pipeline config (mirror of get_pipeline_config_for_mega_moe; smem evaluated
# with the device-side MegaMoeSmemLayout formula, which is what the launch
# must allocate — the wrapper static_asserts it against sizeof(SharedStorage))
# ---------------------------------------------------------------------------

SMEM_CAPACITY = 232448


def smem_layout_bytes(num_experts, num_dispatch_warps, num_bytes_per_pull,
                      num_epilogue_warpgroups, num_epilogue_warps,
                      store_block_m_l1, l1_out_block_n, store_block_m_l2,
                      block_n, load_block_n, num_stages, load_block_m, block_k,
                      sf_block_m, sf_block_n):
    off_dispatch = align(num_experts * 4, 1024)
    cd_l1 = num_epilogue_warpgroups * 2 * store_block_m_l1 * l1_out_block_n
    cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * 2
    off_smem_d = align(off_dispatch + num_dispatch_warps * num_bytes_per_pull, 1024)
    off_smem_a = align(off_smem_d + max(cd_l1, cd_l2), 1024)
    off_smem_b = off_smem_a + num_stages * load_block_m * block_k
    off_smem_sfa = off_smem_b + num_stages * load_block_n * block_k
    off_smem_sfb = off_smem_sfa + num_stages * sf_block_m * (block_k // 128) * 4
    off_amax = align(off_smem_sfb + num_stages * sf_block_n * (block_k // 128) * 4, 8)
    off_tasks = align(off_amax + num_epilogue_warps * (store_block_m_l1 // 2) * 8, 16)
    off_barriers = off_tasks + 2 * 32
    b = off_barriers + (num_dispatch_warps + num_stages * 2 + 2 * 2
                        + num_epilogue_warps * 2 + 2 * 2) * 8 + 4
    return align(b, 1024)


def pipeline(num_experts, num_bytes_per_pull, hidden, block_m, block_n, block_k,
             store_block_m_l1, store_block_m_l2, sf_block_m, sf_block_n):
    num_dispatch_warps, num_epilogue_warps = 4, 8
    num_epilogue_warpgroups = num_epilogue_warps // 4
    load_block_m = block_m // 2
    per_stage = (load_block_m * block_k + block_n * block_k + sf_block_m * 4 + sf_block_n * 4 + 16)
    fixed = smem_layout_bytes(num_experts, num_dispatch_warps, num_bytes_per_pull,
                              num_epilogue_warpgroups, num_epilogue_warps, store_block_m_l1,
                              block_n // 2, store_block_m_l2, block_n, block_n,
                              0, load_block_m, block_k, sf_block_m, sf_block_n)
    # fixed includes 0 stages: solve max stages with total <= capacity
    num_stages = (SMEM_CAPACITY - fixed) // per_stage
    smem = smem_layout_bytes(num_experts, num_dispatch_warps, num_bytes_per_pull,
                             num_epilogue_warpgroups, num_epilogue_warps, store_block_m_l1,
                             block_n // 2, store_block_m_l2, block_n, block_n,
                             num_stages, load_block_m, block_k, sf_block_m, sf_block_n)
    while smem > SMEM_CAPACITY and num_stages > 2:
        num_stages -= 1
        smem = smem_layout_bytes(num_experts, num_dispatch_warps, num_bytes_per_pull,
                                 num_epilogue_warpgroups, num_epilogue_warps, store_block_m_l1,
                                 block_n // 2, store_block_m_l2, block_n, block_n,
                                 num_stages, load_block_m, block_k, sf_block_m, sf_block_n)
    return num_stages, smem


def config(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk,
           hidden, intermediate_hidden, num_sms, is_weight_fp8):
    # get_block_config_for_mega_moe (MXFP8FP4)
    num_expected = num_tokens * num_ranks * num_topk / num_experts
    num_covered = num_expected + num_expected ** 0.5
    block_m = 16 if num_expected <= 10 else 32
    if num_expected > 24:
        import math
        num_blocks = math.ceil(num_covered / 240)
        if num_blocks == 1 and num_covered > 192 and num_experts // num_ranks < 14:
            num_blocks = 2
        for cand in (64, 128, 192, 240):
            block_m = cand
            if num_blocks * cand >= num_covered:
                break
    store_block_m_l1 = (8 if block_m <= 16 else 16 if block_m <= 64 else
                        32 if block_m <= 192 else (24 if not is_weight_fp8 else 24))
    store_block_m_l2 = 8 if (block_m == 240 and not is_weight_fp8) else store_block_m_l1
    block_n, block_k = 128, 128
    sf_block_m, sf_block_n = align(block_m, 128), block_n
    num_ring = ring_tokens(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                           num_sms, hidden, intermediate_hidden, block_m)
    num_sf_ring = sf_ring_tokens(num_ring, block_m)
    num_bytes_per_pull = hidden
    while num_bytes_per_pull > 8192:
        num_bytes_per_pull //= 2
    num_stages, smem = pipeline(num_experts, num_bytes_per_pull, hidden, block_m, block_n,
                                block_k, store_block_m_l1, store_block_m_l2,
                                sf_block_m, sf_block_n)
    return dict(block_m=block_m, block_n=block_n, block_k=block_k,
                store_block_m_l1=store_block_m_l1, store_block_m_l2=store_block_m_l2,
                sf_block_m=sf_block_m, sf_block_n=sf_block_n,
                num_ring_tokens=num_ring, num_sf_ring_tokens=num_sf_ring,
                num_stages=num_stages, smem=smem, num_bytes_per_pull=num_bytes_per_pull)


def wrapper(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk,
            hidden, intermediate_hidden, num_sms, is_weight_fp8):
    c = config(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk,
               hidden, intermediate_hidden, num_sms, is_weight_fp8)
    maps = ",\n".join(f"    const __grid_constant__ dg::TmaMap tensor_map_{n}"
                      for n in ("l1_acts", "l1_acts_sf", "l1_weights", "l1_weights_sf", "l1_output",
                                "l2_acts", "l2_acts_sf", "l2_weights", "l2_weights_sf",
                                "shared_l1_acts", "shared_l1_acts_sf", "shared_l1_weights",
                                "shared_l1_weights_sf", "shared_l1_output",
                                "shared_l2_acts", "shared_l2_acts_sf", "shared_l2_weights",
                                "shared_l2_weights_sf"))
    tpl = f"""{num_max_tokens_per_rank}, {hidden}, {intermediate_hidden},
        {num_experts}, 0, {num_topk},
        {c['block_m']}, {c['block_n']}, {c['block_k']},
        {c['store_block_m_l1']}, {c['store_block_m_l2']},
        {c['sf_block_m']}, {c['sf_block_n']},
        {c['num_ring_tokens']}, {c['num_sf_ring_tokens']},
        {c['num_stages']},
        {c['num_bytes_per_pull']},
        128, 128, 256,
        {num_sms}, {num_ranks},
        0x7f800000u /* no clamp */, true, {1 if is_weight_fp8 else 0}"""
    return f'''extern "C" __global__ void __dg_kernel(
    void* y, int* cumulative_local_expert_recv_stats, const unsigned num_tokens,
    const __grid_constant__ dg::SymBuffer<{num_ranks}> sym_buffer,
{maps},
    const unsigned char* sm_locality_domains) {{
    dg::mega_moe_fp8_fp4_impl<
        {tpl}
    >(y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer,
      tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, tensor_map_l1_weights_sf,
      tensor_map_l1_output, tensor_map_l2_acts, tensor_map_l2_acts_sf, tensor_map_l2_weights,
      tensor_map_l2_weights_sf, tensor_map_shared_l1_acts, tensor_map_shared_l1_acts_sf,
      tensor_map_shared_l1_weights, tensor_map_shared_l1_weights_sf, tensor_map_shared_l1_output,
      tensor_map_shared_l2_acts, tensor_map_shared_l2_acts_sf, tensor_map_shared_l2_weights,
      tensor_map_shared_l2_weights_sf, sm_locality_domains);
}}
''', c


def compile_one(label, arch, source):
    l = ctypes.CDLL(NVRTC)
    prog = ctypes.c_void_p()
    rc = l.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"mega_moe.cu", 0, None, None)
    assert rc == 0, rc
    opts = [b"--gpu-architecture=" + arch.encode(), b"--std=c++17"]
    arr = (ctypes.c_char_p * len(opts))(*[ctypes.c_char_p(o) for o in opts])
    rc = l.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        n = ctypes.c_size_t()
        l.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value + 1)
        l.nvrtcGetProgramLog(prog, buf)
        log = buf.value.decode(errors="replace")
        printed = 0
        for ln in log.splitlines():
            if ("error" in ln or "note" in ln or "^" in ln
                    or "warning" in ln.lower() and "ptxas" in ln.lower()) and printed < 60:
                print(ln)
                printed += 1
        return None
    size = ctypes.c_size_t()
    l.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = size.value
    cubin = 0
    if arch.startswith("sm_"):
        l.nvrtcGetCUBINSize(prog, ctypes.byref(size))
        l.nvrtcGetCUBIN(prog, ctypes.create_string_buffer(size.value + 1))
        cubin = size.value
    return ptx, cubin


def main():
    prelude = open(ROOT + "prelude.h").read()
    tu = open(ROOT + "mega_moe_sm100.cu").read()

    variants = [
        ("rank1-fp4", dict(num_ranks=1, num_experts=256, num_max_tokens_per_rank=1920,
                           num_tokens=1920, num_topk=8, hidden=7168, intermediate_hidden=2048,
                           num_sms=148, is_weight_fp8=False)),
        ("rank4-fp8", dict(num_ranks=4, num_experts=256, num_max_tokens_per_rank=1920,
                           num_tokens=1920, num_topk=8, hidden=7168, intermediate_hidden=2048,
                           num_sms=148, is_weight_fp8=True)),
    ]
    ok = True
    for label, v in variants:
        body, c = wrapper(**v)
        src = prelude + "\n" + tu + "\n" + body
        print(f"== {label}: block_m={c['block_m']} stages={c['num_stages']} "
              f"ring={c['num_ring_tokens']} sf_ring={c['num_sf_ring_tokens']} "
              f"smem={c['smem']} pull={c['num_bytes_per_pull']}")
        for arch in ("compute_100a", "sm_100a"):
            r = compile_one(label, arch, src)
            if r is None:
                print(f"   {arch}: FAILED (see log above)")
                ok = False
            else:
                print(f"   {arch}: PTX {r[0]} B, SASS/CUBIN {r[1]} B")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
