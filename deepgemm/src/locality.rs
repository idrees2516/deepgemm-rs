//! Locality domains — host-side port of DeepGEMM's `locality_domain` runtime
//! (csrc/apis/locality_domain.hpp + csrc/runtime/locality_domain.hpp).
//!
//! CONCEPT
//! =======
//! B200 is a dual-die GPU: each die has its own HBM stacks and its SMs reach
//! the near die's memory faster. A **locality domain** is one such memory
//! partition. DeepGEMM's MegaMoE megakernel takes an `sm_locality_domains`
//! table (one domain id per SM) and schedules weight tiles so an SM mostly
//! reads weights homed in its own domain.
//!
//! What is ported here:
//! * the **probe machinery** — the pointer-chase kernel
//!   (`kernels/locality_probe.cu`) plus the host math that turns raw
//!   per-(SM, domain, chunk) cycle counts into a validated SM→domain table:
//!   median over chunks, per-SM argmin, a *clear separation* check (2nd-best
//!   domain at least `MIN_FAR_NEAR_RATIO`× slower), a *TPC consistency*
//!   check (both SMs of a TPC must agree), and up to `MAX_ATTEMPTS` retries
//!   before falling back to an even table.
//! * the **balance** step: when a two-domain mapping puts more TPCs in one
//!   domain, surplus TPCs are flipped so each domain drives half the chip.
//!
//! What is intentionally NOT ported: the MLOPart IPC allocator (upstream
//! `mlopart.hpp` creates domain-homed memory through a Python helper and a
//! POSIX-fd `cuMemImportFromShareableHandle` round-trip, a stopgap until
//! CUDA 13.4 exposes `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`).
//! Without domain-homed allocations the probe cannot discriminate domains,
//! so [`sm_locality_domains`] returns the **even mapping** — which upstream
//! itself notes is "functionally correct without localization". When
//! domain-homed allocation lands (CUDA 13.4 driver API), plug a per-domain
//! buffer into [`locality_probe_chase`] and the validation path here turns
//! into the full probed mapping with no other changes.

use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::jit::{self, kernel_src};
use std::sync::Mutex;

/// Probe geometry (upstream `kNumLocalityDomainProbe*` constants).
pub const PROBE_CHUNK_BYTES: usize = 4096;
pub const PROBE_LINE_BYTES: usize = 128;
pub const PROBE_HOPS: usize = PROBE_CHUNK_BYTES / PROBE_LINE_BYTES;
pub const PROBE_CHUNKS_PER_SM: usize = 8;
/// Coprime with the line count, so the chain visits every line.
pub const CHAIN_LINE_STRIDE: usize = 7;
/// Second-best domain must be >= 1.25x the best domain's latency.
pub const MIN_FAR_NEAR_RATIO: f32 = 1.25;
pub const MAX_PROBE_ATTEMPTS: usize = 5;
/// SMs per TPC (both must agree on a domain for the probe to be valid).
pub const SMS_PER_TPC: usize = 2;
/// Device locality-domain count (upstream `kNumDeviceLocalityDomains`).
pub const NUM_LOCALITY_DOMAINS: u32 = crate::moe_layout::NUM_DEVICE_LOCALITY_DOMAINS;

// ---------------------------------------------------------------------------
// Pure host math (all unit-testable without a GPU)
// ---------------------------------------------------------------------------

/// The probe chain for one chunk: word 0 of line `l` stores the word offset
/// of line `(l + CHAIN_LINE_STRIDE) % num_lines`. All other words are 0.
/// Visiting order is a permutation of the lines (7 coprime 32).
pub fn probe_chain() -> Vec<u32> {
    let words_per_line = PROBE_LINE_BYTES / 4;
    let mut chain = vec![0u32; PROBE_CHUNK_BYTES / 4];
    for l in 0..PROBE_HOPS {
        let next = (l + CHAIN_LINE_STRIDE) % PROBE_HOPS;
        chain[l * words_per_line] = (next * words_per_line) as u32;
    }
    chain
}

/// Even (fallback) mapping: TPC `t` gets domain `t % NUM_DOMAINS` — without
/// localization an arbitrary SM mapping is functionally correct, but keep it
/// TPC-aligned and round-robin so cluster pairs stay domain-consistent.
pub fn even_sm_locality_domains(num_sms: usize) -> Vec<u8> {
    let num_domains = NUM_LOCALITY_DOMAINS as usize;
    (0..num_sms)
        .map(|sm| ((sm / SMS_PER_TPC) % num_domains) as u8)
        .collect()
}

/// Balance a **two-domain** mapping: if one domain drives more than half the
/// TPCs, surplus TPCs (in index order) flip to the other domain. Upstream
/// asserts an even TPC count; so do we.
pub fn balance_sm_locality_domains(sm_domain: &[u8]) -> Vec<u8> {
    assert!(
        sm_domain.len() % SMS_PER_TPC == 0,
        "TPC-aligned table required"
    );
    let num_tpcs = sm_domain.len() / SMS_PER_TPC;
    let tpc_domain: Vec<u8> = (0..num_tpcs).map(|t| sm_domain[t * SMS_PER_TPC]).collect();
    let num_domain_1: usize = tpc_domain.iter().map(|&d| (d == 1) as usize).sum();
    // The domain holding more TPCs (ties count as domain 0 being "larger").
    let larger: u8 = ((num_domain_1 * 2) > num_tpcs) as u8;
    // First `num_tpcs / 2` TPCs of the larger domain flip to the other.
    let mut surplus_seen = 0usize;
    let mut balanced = sm_domain.to_vec();
    for (t, &d) in tpc_domain.iter().enumerate() {
        if d == larger {
            if surplus_seen < num_tpcs / 2 {
                surplus_seen += 1;
            } else {
                balanced[t * SMS_PER_TPC] = larger ^ 1;
                balanced[t * SMS_PER_TPC + 1] = larger ^ 1;
            }
        }
    }
    balanced
}

/// One probe attempt over per-domain per-SM **median** latencies
/// `latency[domain][sm]` (cycles/hop). Returns the SM→domain table when the
/// measurement is trustworthy, `None` to retry:
/// * clear separation: the second-best domain must be at least
///   `MIN_FAR_NEAR_RATIO`× the best one (a 25% margin over the winner);
/// * TPC consistency: both SMs of a TPC must pick the same domain (they
///   share a frontend; a disagreement means the measurement was noisy).
pub fn try_probe_from_medians(latency: &[Vec<f32>]) -> Option<Vec<u8>> {
    let num_domains = latency.len();
    let num_sms = latency.first().map(|d| d.len()).unwrap_or(0);
    if num_domains == 0 || num_sms == 0 {
        return None;
    }
    let mut sm_domain = vec![0u8; num_sms];
    for sm in 0..num_sms {
        // (domain, latency) pairs sorted by latency.
        let mut order: Vec<(u8, f32)> = (0..num_domains)
            .map(|d| (d as u8, latency[d][sm]))
            .collect();
        order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sm_domain[sm] = order[0].0;
        let (best, second) = (order[0].1, order[1].1);
        if second.partial_cmp(&(best * MIN_FAR_NEAR_RATIO)) != Some(std::cmp::Ordering::Greater) {
            return None; // ambiguous winner
        }
    }
    // TPC consistency.
    for t in 0..num_sms / SMS_PER_TPC {
        let base = t * SMS_PER_TPC;
        for s in 1..SMS_PER_TPC {
            if sm_domain[base + s] != sm_domain[base] {
                return None;
            }
        }
    }
    Some(sm_domain)
}

// ---------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------

/// TU for the probe kernel: prelude (auto-prepended) + this file.
pub fn locality_unit() -> &'static str {
    kernel_src::LOCALITY_PROBE
}

/// Compile-check wrapper body for the probe kernel.
pub fn locality_probe_body(num_chunks_per_sm: u32) -> String {
    format!(
        r#"extern "C" __global__ void __dg_kernel(const unsigned* buf, unsigned short* out) {{
    dg::locality_probe_chase_impl<{hops}, {chunks}, {chunk_bytes}>(buf, out);
}}"#,
        hops = PROBE_HOPS,
        chunks = num_chunks_per_sm,
        chunk_bytes = PROBE_CHUNK_BYTES,
    )
}

/// Launch the probe for one per-domain buffer:
/// `buf` is `[num_sms, PROBE_CHUNKS_PER_SM, PROBE_CHUNK_BYTES]` bytes
/// (chain-filled), `out` receives `[num_sms, PROBE_CHUNKS_PER_SM]` uint16
/// cycles-per-hop. Grid = num_sms single-warp blocks (one probing lane).
pub fn locality_probe_chase(
    dev: &Device,
    stream: &DevStream,
    buf: &DevBuffer,
    out: &mut DevBuffer,
    num_sms: u32,
) -> DgResult<()> {
    if buf.len < num_sms as usize * PROBE_CHUNKS_PER_SM * PROBE_CHUNK_BYTES {
        return Err(DgError::InvalidArg("probe buffer too small".into()));
    }
    let body = locality_probe_body(PROBE_CHUNKS_PER_SM as u32);
    let sig = format!("locality_probe/chunks{}", PROBE_CHUNKS_PER_SM);
    let func = jit::get_kernel(dev, locality_unit(), "locality_probe", &sig, &body)?;
    let cfg = crate::sys::LaunchEx {
        grid: (num_sms, 1, 1),
        block: (32, 1, 1),
        smem: 0,
        cluster: None,
        pdl: false,
    };
    let args = jit::Args::new().devptr(buf.ptr).devptr(out.ptr);
    jit::launch(dev, func, stream.raw(), &cfg, args)
}

// ---------------------------------------------------------------------------
// Cached table API (what the megakernel consumes)
// ---------------------------------------------------------------------------

static TABLE: Mutex<Option<Vec<u8>>> = Mutex::new(None);

/// MLOPart-style domain-homed allocation is not ported (see module docs):
/// the probe has no domain-homed buffers to chase, so localization is
/// unavailable in this port and the even table is used.
pub fn is_localization_available() -> bool {
    false
}

/// The SM→domain table the MegaMoE megakernel's scheduler consumes
/// (`sm_locality_domains` kernel parameter). Cached after the first call.
/// With localization unavailable this is the even mapping; a future
/// domain-homed allocator plugs into [`locality_probe_chase`] +
/// [`try_probe_from_medians`] to upgrade it to the probed mapping.
pub fn sm_locality_domains(dev: &Device, num_sms: u32) -> Vec<u8> {
    let mut guard = TABLE.lock().unwrap();
    if let Some(t) = guard.as_ref() {
        return t.clone();
    }
    let table = if is_localization_available() {
        // Probe path: per-domain buffers + kernel launches + validation.
        // Kept as the ready-made upgrade once CUDA 13.4 domain-homed
        // allocation exists; today `is_localization_available()` is false.
        probe_sm_locality_domains(dev, num_sms)
    } else {
        even_sm_locality_domains(num_sms as usize)
    };
    *guard = Some(table.clone());
    table
}

/// Full probe flow (per-domain buffers, median, validate, retry, fallback).
/// Used only when [`is_localization_available`] — see module docs.
fn probe_sm_locality_domains(dev: &Device, num_sms: u32) -> Vec<u8> {
    let stream = DevStream::new(dev).expect("stream");
    let chain = probe_chain();
    let num_domains = NUM_LOCALITY_DOMAINS as usize;
    let buf_words = num_sms as usize * PROBE_CHUNKS_PER_SM * PROBE_CHUNK_BYTES / 4;
    for _attempt in 0..MAX_PROBE_ATTEMPTS {
        // Per-domain probe buffers, chain-filled; SM d>0 slices chase the
        // same chain (domain 0's buffer is the layout template).
        let mut latencies: Vec<Vec<f32>> = Vec::with_capacity(num_domains);
        let mut ok = true;
        for _d in 0..num_domains {
            let mut words = chain.repeat(num_sms as usize * PROBE_CHUNKS_PER_SM);
            words.resize(buf_words, 0);
            let buf = crate::device::alloc_and_upload(dev, &words, stream.raw()).ok();
            let out = DevBuffer::alloc_zeros(dev, num_sms as usize * PROBE_CHUNKS_PER_SM * 2).ok();
            let (Some(buf), Some(ref mut o)) = (buf, out) else {
                ok = false;
                break;
            };
            if locality_probe_chase(dev, &stream, &buf, o, num_sms).is_err() {
                ok = false;
                break;
            }
            let raw = match crate::device::download::<u8>(dev, o, stream.raw()) {
                Ok(r) => r,
                Err(_) => {
                    ok = false;
                    break;
                }
            };
            // Median over the 8 chunks per SM (u16 little-endian).
            let mut per_sm = Vec::with_capacity(num_sms as usize);
            for sm in 0..num_sms as usize {
                let base = sm * PROBE_CHUNKS_PER_SM;
                let mut v: Vec<f32> = (0..PROBE_CHUNKS_PER_SM)
                    .map(|c| {
                        let b = (base + c) * 2;
                        u16::from_le_bytes([raw[b], raw[b + 1]]) as f32
                    })
                    .collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                per_sm.push(v[PROBE_CHUNKS_PER_SM / 2]);
            }
            latencies.push(per_sm);
        }
        if ok {
            if let Some(table) = try_probe_from_medians(&latencies) {
                return table;
            }
        }
    }
    // Unstable measurement: an arbitrary mapping is functionally correct.
    even_sm_locality_domains(num_sms as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_visits_every_line_once() {
        let chain = probe_chain();
        let words = PROBE_LINE_BYTES / 4;
        let mut visited = [false; PROBE_HOPS];
        let mut word = 0usize; // start at line 0, word 0
        for _ in 0..PROBE_HOPS {
            let line = word / words;
            assert!(!visited[line], "line {line} visited twice");
            visited[line] = true;
            word = chain[word] as usize;
        }
        assert!(visited.iter().all(|&v| v));
    }

    #[test]
    fn even_table_is_tpc_aligned_round_robin() {
        let t = even_sm_locality_domains(148);
        assert_eq!(t.len(), 148);
        for (sm, &d) in t.iter().enumerate() {
            assert_eq!(d as usize, (sm / 2) % NUM_LOCALITY_DOMAINS as usize);
        }
    }

    #[test]
    fn balance_flips_surplus_tpcs() {
        // 8 TPCs: 6 in domain 1, 2 in domain 0 -> flip 2 TPCs to domain 0.
        let mut sm = vec![1u8; 16];
        for t in [0usize, 1] {
            sm[t * 2] = 0;
            sm[t * 2 + 1] = 0;
        }
        let balanced = balance_sm_locality_domains(&sm);
        let ones = balanced.iter().filter(|&&d| d == 1).count();
        assert_eq!(ones, 8, "each domain drives half the SMs");
        // TPC-aligned: both SMs of a TPC agree.
        for t in 0..8 {
            assert_eq!(balanced[t * 2], balanced[t * 2 + 1]);
        }
    }

    #[test]
    fn probe_validation_accepts_clear_mapping() {
        // Domain 0 clearly fastest for all SMs (2.0x separation).
        let near = 600.0f32;
        let far = 1200.0f32;
        let lat = (0..NUM_LOCALITY_DOMAINS as usize)
            .map(|d| {
                (0..148)
                    .map(|_| if d == 0 { near } else { far })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let table = try_probe_from_medians(&lat).expect("clear mapping accepted");
        assert!(table.iter().all(|&d| d == 0));
    }

    #[test]
    fn probe_validation_rejects_ambiguous() {
        // Second-best within the 1.25x margin -> retry.
        let lat = (0..NUM_LOCALITY_DOMAINS as usize)
            .map(|d| vec![600.0 + d as f32 * 10.0; 148])
            .collect::<Vec<_>>();
        assert!(try_probe_from_medians(&lat).is_none());
    }

    #[test]
    fn probe_validation_rejects_tpc_disagreement() {
        // Clear per-SM winners, but SMs within TPC 0 disagree.
        let mut lat: Vec<Vec<f32>> = (0..NUM_LOCALITY_DOMAINS as usize)
            .map(|_| vec![1200.0; 148])
            .collect();
        lat[0][0] = 600.0; // SM 0 -> domain 0
        lat[1][1] = 600.0; // SM 1 -> domain 1 (same TPC!)
        assert!(try_probe_from_medians(&lat).is_none());
    }

    #[test]
    fn compile_check_locality_probe() {
        // Offline NVRTC plane: PTX (compute_100a) + SASS (sm_100a).
        let r = jit::compile_check_kernel(
            locality_unit(),
            &locality_probe_body(PROBE_CHUNKS_PER_SM as u32),
            "100a",
            "locality-probe",
        )
        .expect("probe kernel must compile");
        assert!(r.cubin_len.is_some(), "probe kernel SASS generation failed");
    }
}
