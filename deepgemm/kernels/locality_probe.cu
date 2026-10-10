// Locality-domain probe kernel — port of `sm100_locality_domain.cuh`.
//
// CONCEPT — memory locality domains
// ================================
// A B200 is physically two compute dies, each with its own HBM stacks. A
// memory "locality domain" is a set of HBM partitions that are cheapest for
// a given set of SMs to reach. If the MegaMoE weight buffers are *homed* in
// the domain closest to the SMs that consume them, every TMA load of the
// weights crosses fewer inter-die links: more bandwidth, lower latency.
//
// How do we discover which SM belongs to which domain? By *measuring* it:
// allocate one probe buffer per domain (homed in that domain), then have
// every SM pointer-chase through its slice of each buffer and time the
// hops. The domain an SM reads fastest is its domain.
//
// The chase (this kernel)
// ----------------------
// One chunk per (SM, chunk_idx) is 4096 bytes = 32 lines of 128 bytes.
// The host builds a chain inside the chunk: word 0 of line `l` stores the
// word offset of line `(l + 7) mod 32` — 7 is coprime with 32, so the
// chain visits every line exactly once before returning to the start.
// Each hop therefore touches a different 128B line: every timed load is a
// cold miss serviced from HBM (the L2 is flushed before the probe), which
// is exactly the signal we want — HBM proximity, not cache luck.
//
// Timing: `clock64()` around the 32-hop loop, divided by the hop count;
// the per-hop latency in cycles (uint16) is written to `out`. A sentinel
// result of 0xffffffff means "chain broken" and is stored as 0 (invalid),
// which the host-side assertion (`latency > 0` everywhere) rejects.
//
// Upstream: deep_gemm/include/deep_gemm/impls/sm100_locality_domain.cuh
// (26 lines); launch config in csrc/jit_kernels/impls/sm100_locality_domain.hpp.

namespace dg {

// `buf`: [num_sms, kNumChunksPerSM, kNumChunkBytes/4] word chain buffer.
// `out`: [num_sms, kNumChunksPerSM] uint16 cycles-per-hop.
template <uint32_t kNumHops, uint32_t kNumChunksPerSM, uint32_t kNumChunkBytes>
__global__ __launch_bounds__(32, 1) void locality_probe_chase_impl(const uint32_t* buf,
                                                                   uint16_t* out) {
    // One probing thread per SM; the other 31 lanes sleep.
    if (threadIdx.x != 0)
        return;
    const uint32_t sm_idx = get_sm_idx();
    #pragma unroll
    for (uint32_t chunk_idx = 0; chunk_idx < kNumChunksPerSM; ++ chunk_idx) {
        // This SM's chunk within the per-domain buffer.
        const uint32_t* chunk = buf +
            static_cast<uint64_t>(sm_idx * kNumChunksPerSM + chunk_idx) * (kNumChunkBytes / 4);
        // Prime the chain, then time `kNumHops` dependent loads. The
        // dependency chain defeats both the compiler and the memory system:
        // no prefetching, no overlap — each hop pays full HBM latency.
        uint32_t idx = ld_global_cg_u32(chunk);
        const int64_t start = clock64();
        #pragma unroll
        for (uint32_t i = 0; i < kNumHops; ++ i)
            idx = ld_global_cg_u32(chunk + idx);
        const int64_t cycles = clock64() - start;
        out[sm_idx * kNumChunksPerSM + chunk_idx] =
            idx == 0xffffffffu ? 0u : static_cast<uint16_t>(cycles / kNumHops);
    }
}

} // namespace dg
