//! CPU-only tests for the SM90 (Hopper) MQA-logits port (no GPU required):
//!
//! * pure-Rust mirrors of the ported scheduler/layout math (the contiguous
//!   `load_schedule` window/alignment, the WGMMA accumulator -> weight
//!   register index algebra, the paged metadata work distribution and the
//!   `SM90PagedMQALogitsScheduler` task walk);
//! * offline NVRTC compile checks (PTX + SASS) of both kernels and the
//!   metadata kernel, using the SAME wrapper bodies the launchers build.

use deepgemm::api_sm90_mqa;
use deepgemm::jit;

// ---------------------------------------------------------------------------
// Mirror of the contiguous kernel's `load_schedule` (mqa_logits_sm90_impl)
// ---------------------------------------------------------------------------

/// Returns (q_stage, q_phase, kv_start, num_kv_blocks) for one q block,
/// mirroring the device lambda (kNumQStages = 3, BLOCK_KV = 256).
fn contig_schedule(
    block_q_idx: u32,
    block_q: u32,
    seq_len: u32,
    seq_len_kv: u32,
    ks: &[u32],
    ke: &[u32],
    q_iter_idx: u32,
) -> (u32, u32, u32, u32) {
    let mut start = u32::MAX;
    let mut end = 0u32;
    for i in 0..block_q {
        let q_idx = (block_q_idx * block_q + i).min(seq_len - 1) as usize;
        start = start.min(ks[q_idx].min(seq_len_kv));
        end = end.max(ke[q_idx].min(seq_len_kv));
    }
    let kv_start = start / 4 * 4; // 16B TMA alignment of the fp32 KV-scale box
    let num_kv_blocks = (end - kv_start).div_ceil(256);
    (
        q_iter_idx % 3,
        (q_iter_idx / 3) & 1,
        kv_start,
        num_kv_blocks,
    )
}

#[test]
fn contig_schedule_window_and_alignment() {
    // heads=32 -> BLOCK_Q=4 tokens per block; heads=64 -> 2.
    let ks = [7u32, 100, 3, 9];
    let ke = [301u32, 400, 8, 512];
    let (stage, phase, kv_start, num_kv) = contig_schedule(0, 4, 4, 512, &ks, &ke, 0);
    // start = min(7, 100, 3, 9) = 3 -> aligned DOWN to 0 (multiple of 4)
    assert_eq!(kv_start, 0, "kv_start must be floored to a multiple of 4");
    // end = max(301, 400, 8, 512) = 512 -> ceil(512/256) = 2 blocks
    assert_eq!(num_kv, 2);
    assert_eq!((stage, phase), (0, 0));

    // Second iteration: stage 1 phase 0; the phase flips at iter 3 (3 stages).
    assert_eq!((&contig_schedule(0, 4, 4, 512, &ks, &ke, 1)), &(1, 0, 0, 2));
    assert_eq!((&contig_schedule(0, 4, 4, 512, &ks, &ke, 2)), &(2, 0, 0, 2));
    assert_eq!((&contig_schedule(0, 4, 4, 512, &ks, &ke, 3)), &(0, 1, 0, 2));
    assert_eq!((&contig_schedule(0, 4, 4, 512, &ks, &ke, 4)), &(1, 1, 0, 2));
    // Iter 6 wraps the ring again: phase back to 0.
    assert_eq!((&contig_schedule(0, 4, 4, 512, &ks, &ke, 7)), &(1, 0, 0, 2));

    // Partial span crossing a block boundary: [260, 513) clamped to 512.
    let (.., kv_start, num_kv) = contig_schedule(0, 4, 4, 512, &[260; 4], &[513; 4], 0);
    assert_eq!(kv_start, 260, "260 is already 4-aligned");
    assert_eq!(num_kv, 1, "ceil((512-260)/256) = 1");

    // Empty windows: end clamps below start-aligned start -> 0 blocks.
    let (.., num_kv) = contig_schedule(0, 4, 4, 512, &[8; 4], &[8; 4], 0);
    assert_eq!(num_kv, 0, "empty union window produces no KV blocks");
}

// ---------------------------------------------------------------------------
// Mirror of the WGMMA accumulator -> weight-register algebra (Concept 2 in
// kernels/mqa_logits_sm90.cu): accumulator element (j', lane) of token i
// covers logits column col = 8*(j'/4) + 2*(lane%4) + (j'&1), which must land
// inside token i's head slice, and the weight read
// `i*heads + (w/2)*8 + (w&1) + 2*(lane%4)` with w = (j/4)*2 + (j&1)
// (j = j' - i*heads/2 local) must resolve to exactly weights[i][col - i*heads].
// ---------------------------------------------------------------------------

#[test]
fn wgmma_accum_weight_index_algebra() {
    for &(heads, block_q) in &[(32u32, 4u32), (64, 2)] {
        let n = block_q * heads; // WGMMA_N, always 128
        assert_eq!(n, 128, "BLOCK_Q * heads is the fixed WGMMA N");
        let num_accum = n / 2; // per-lane accumulators
        let per_token = heads / 2; // kNumAccumPerReduce
        assert_eq!(num_accum / per_token, block_q);

        // (col -> times covered) across the 4 lanes of a quad, per row half:
        // accumulator elements j%4 in {0,1} own row (lane/4), j%4 in {2,3}
        // own row (lane/4 + 8) — the v_0 / v_1 KV rows of the reduce.
        let mut covered_v0 = vec![0u32; heads as usize];
        let mut covered_v1 = vec![0u32; heads as usize];
        for i in 0..block_q {
            for lane_q in 0..4u32 {
                for j in 0..per_token {
                    let j_global = i * per_token + j;
                    let col = 8 * (j_global / 4) + 2 * lane_q + (j_global & 1);
                    // Column must belong to token i's slice of the N dim.
                    assert!(
                        col >= i * heads && col < (i + 1) * heads,
                        "heads={heads}: accum j_global={j_global} lane_q={lane_q} -> col {col} \
                         outside token {i} slice",
                    );
                    let h = col - i * heads;
                    // Weight register index and SMEM offset (device formula).
                    let w = (j / 4) * 2 + (j & 1);
                    let smem_off = i * heads + (w / 2) * 8 + (w & 1) + 2 * lane_q;
                    assert_eq!(
                        smem_off,
                        i * heads + h,
                        "heads={heads}: weight read must resolve to w[token][head]"
                    );
                    if i == 0 {
                        if j_global % 4 < 2 {
                            covered_v0[h as usize] += 1;
                        } else {
                            covered_v1[h as usize] += 1;
                        }
                    }
                }
            }
        }
        // Each head is folded exactly once per row half (v_0 and v_1).
        assert!(
            covered_v0.iter().all(|&c| c == 1) && covered_v1.iter().all(|&c| c == 1),
            "heads={heads}: every head must appear exactly once per KV row half"
        );
    }
}

// ---------------------------------------------------------------------------
// Mirrors of the paged metadata kernel + scheduler (SM90PagedMQALogitsScheduler)
// ---------------------------------------------------------------------------

const SPLIT_KV: u32 = 256;
const BLOCK_KV: u32 = 64;
const BLOCKS_PER_SPLIT: u32 = 4;

/// Host mirror of `sm90_paged_mqa_logits_metadata_impl` (non-varlen, 2D lens):
/// inclusive prefix of ceil(ctx/SPLIT_KV), reversed work distribution, binary
/// search back onto the (request, atom, split) grid.  `context_lens` is
/// `[batch, next_n]`; returns `num_sms + 1` (q_atom, kv_split) entries.
fn metadata_host(context_lens: &[u32], next_n: u32, num_sms: u32) -> Vec<(u32, u32)> {
    let batch = (context_lens.len() / next_n as usize) as u32;
    let aligned = batch.div_ceil(32) * 32;
    let ctx_of = |req: u32| -> u32 {
        // 2D lens: the LAST token of the request carries the full context.
        context_lens[req as usize * next_n as usize + next_n as usize - 1]
    };

    let mut prefix = vec![0u32; aligned as usize];
    let mut run = 0u32;
    for (i, slot) in prefix.iter_mut().enumerate() {
        let ctx = if (i as u32) < batch {
            ctx_of(i as u32)
        } else {
            0
        };
        run += ctx.div_ceil(SPLIT_KV);
        *slot = run;
    }

    let next_n_atom = if next_n >= 2 { 2 } else { 1 };
    let num_atoms = next_n.div_ceil(next_n_atom); // 1 for next_n in {1, 2}
    let total = run * num_atoms;
    let q = total / num_sms;
    let r = total % num_sms;
    let pivot = num_sms - r;

    let mut meta = Vec::with_capacity(num_sms as usize + 1);
    for sm in 0..num_sms {
        // Device form `sm > pivot ? sm - pivot : 0` == saturating_sub (clippy).
        let seg_starts = sm * q + sm.saturating_sub(pivot);
        // First request whose segment count exceeds seg_starts.
        let (mut lo, mut hi) = (0usize, batch as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if prefix[mid] * num_atoms <= seg_starts {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let q_idx = lo as u32;
        let offset_in_q = if q_idx == 0 {
            seg_starts
        } else {
            seg_starts - prefix[(q_idx - 1) as usize] * num_atoms
        };
        let num_segs_q = if q_idx == 0 {
            prefix[0]
        } else {
            prefix[q_idx as usize] - prefix[(q_idx - 1) as usize]
        };
        // Device form `num_segs_q > 0 ? a / n : 0` == checked_div().unwrap_or(0)
        // (clippy); the modulo below is guarded by the same condition.
        let atom_idx = offset_in_q.checked_div(num_segs_q).unwrap_or(0);
        let kv_split_idx = if num_segs_q > 0 {
            offset_in_q % num_segs_q
        } else {
            0
        };
        meta.push((q_idx * num_atoms + atom_idx, kv_split_idx));
    }
    // End sentinel (one past the final atom).
    meta.push((batch * num_atoms, 0));
    meta
}

/// Host mirror of `SM90PagedMQALogitsScheduler::fetch_next_task` walking one
/// SM's [start, end) range.  Emits (q_atom, kv_block_idx) tasks
/// (kv in BLOCK_KV units, advancing by BLOCKS_PER_SPLIT).
fn walk_sm(meta: &[(u32, u32)], sm: u32, num_kv_blocks: impl Fn(u32) -> u32) -> Vec<(u32, u32)> {
    let start = meta[sm as usize];
    let end = meta[sm as usize + 1];
    let mut current_q = start.0;
    let mut current_kv = start.1 * BLOCKS_PER_SPLIT;
    let end_q = end.0;
    let end_kv = end.1 * BLOCKS_PER_SPLIT;

    let exist = |atom: u32| atom < end_q || (atom == end_q && 0 < end_kv);
    // Unconditional initial refresh (in-bounds by the reversed allocation).
    let mut num_kv = num_kv_blocks(current_q);

    let mut out = Vec::new();
    loop {
        if current_q == end_q && current_kv == end_kv {
            break;
        }
        out.push((current_q, current_kv));
        current_kv += BLOCKS_PER_SPLIT;
        if current_kv >= num_kv {
            current_kv = 0;
            current_q += 1; // current_advance == 1 (non-varlen)
            if exist(current_q) {
                num_kv = num_kv_blocks(current_q);
            }
        }
    }
    out
}

/// The exhaustive ground truth: every (atom, split) of every request, in
/// global order (atoms of one token pair share the request's context).
fn global_tasks(context_lens: &[u32], next_n: u32) -> Vec<(u32, u32)> {
    let batch = (context_lens.len() / next_n as usize) as u32;
    let ctx_of = |req: u32| context_lens[req as usize * next_n as usize + next_n as usize - 1];
    let mut out = Vec::new();
    for atom in 0..batch {
        for split in 0..ctx_of(atom).div_ceil(SPLIT_KV) {
            out.push((atom, split));
        }
    }
    out
}

#[test]
fn paged_metadata_covers_every_task_exactly_once() {
    let cases: &[(&[u32], u32, u32)] = &[
        // (context_lens [batch, next_n], next_n, num_sms)
        (&[100, 100, 900, 900, 600, 600, 256, 256], 2, 4),
        (&[100, 100, 900, 900, 600, 600, 256, 256], 2, 132),
        (&[64, 512, 300, 1, 1000, 128], 1, 3),
        (&[64, 512, 300, 1, 1000, 128], 1, 132),
        (&[0, 0, 0], 1, 4),            // all-empty: no tasks anywhere
        (&[1], 1, 8),                  // single tiny request, more SMs than work
        (&[4096; 6], 2, 4),            // heavy requests, few SMs
        (&[257, 257, 255, 255], 2, 2), // splits straddle requests
    ];
    for &(lens, next_n, num_sms) in cases {
        let meta = metadata_host(lens, next_n, num_sms);
        assert_eq!(meta.len(), num_sms as usize + 1, "sentinel present");

        let batch = (lens.len() / next_n as usize) as u32;
        let num_kv_blocks = |atom: u32| -> u32 {
            // In-bounds even for the sentinel atom (last slot of the lens).
            let idx = (atom as usize).min(batch as usize - 1);
            lens[idx * next_n as usize + next_n as usize - 1].div_ceil(BLOCK_KV)
        };

        let mut walked = Vec::new();
        for sm in 0..num_sms {
            walked.extend(walk_sm(&meta, sm, num_kv_blocks));
        }
        let truth = global_tasks(lens, next_n);
        // Tasks are (atom, split); the walk emits block units (split*4).
        let walked: Vec<(u32, u32)> = walked
            .into_iter()
            .map(|(a, kv)| (a, kv / BLOCKS_PER_SPLIT))
            .collect();
        assert_eq!(
            walked, truth,
            "lens={lens:?} next_n={next_n} num_sms={num_sms}: SM walks must cover the \
             global task sequence exactly once, in order"
        );
    }
}

#[test]
fn paged_metadata_balance() {
    // Per-SM task counts may differ by at most 1 and must sum to the total.
    let lens: Vec<u32> = (0..13u32).map(|i| 300 + i * 111).collect();
    let num_sms = 7u32;
    let meta = metadata_host(&lens, 1, num_sms);
    let num_kv_blocks = |atom: u32| lens[atom as usize].div_ceil(BLOCK_KV);
    let counts: Vec<usize> = (0..num_sms)
        .map(|sm| walk_sm(&meta, sm, num_kv_blocks).len())
        .collect();
    let total: usize = counts.iter().sum();
    let truth = global_tasks(&lens, 1).len();
    assert_eq!(total, truth);
    let lo = counts.iter().min().unwrap();
    let hi = counts.iter().max().unwrap();
    assert!(hi - lo <= 1, "counts {counts:?} differ by more than 1");
}

// ---------------------------------------------------------------------------
// Shared-memory budget mirrors (kernel layout must fit the launcher's
// allocation, which must fit the 227 KB SM90 opt-in capacity)
// ---------------------------------------------------------------------------

fn align_up(x: u32, a: u32) -> u32 {
    x.div_ceil(a) * a
}

#[test]
fn sm90_mqa_smem_budgets() {
    const Q_STAGES: u32 = 3;
    const KV_STAGES: u32 = 3;
    const MATH_THREADS: u32 = 512;
    const CAP: u32 = 232_448; // SM90 opt-in smem per block

    for &heads in &[32u32, 64] {
        let block_q = 128 / heads;
        for &head_dim in &[32u32, 64, 128] {
            // ---- contiguous ----
            let swz = head_dim * 8;
            let q_stage = block_q * heads * head_dim;
            let kv_stage = 256 * head_dim;
            let w_stage = block_q * heads * 4;
            let s_stage = 256 * 4;
            // Kernel-side layout: 4 regions + 12 barriers.
            let kernel_total = Q_STAGES * q_stage
                + KV_STAGES * kv_stage
                + Q_STAGES * w_stage
                + KV_STAGES * s_stage
                + (Q_STAGES * 2 + KV_STAGES * 2) * 8;
            // Launcher allocation (upstream host formula; over-allocates the
            // barrier area by (math/128)*2 slots + 4 bytes).
            let host = Q_STAGES * q_stage
                + KV_STAGES * kv_stage
                + Q_STAGES * w_stage
                + KV_STAGES * s_stage
                + (Q_STAGES * 2 + KV_STAGES * 2 + (MATH_THREADS / 128) * 2) * 8
                + 4;
            assert!(
                kernel_total <= host,
                "kernel layout must fit the allocation"
            );
            assert!(host <= CAP, "contiguous smem {host} exceeds capacity");
            assert_eq!(q_stage % swz, 0, "Q stage must be swizzle-aligned");
            assert_eq!(kv_stage % swz, 0, "KV stage must be swizzle-aligned");
        }
    }

    for &next_n in &[1u32, 2] {
        for &heads in &[32u32, 64] {
            for &head_dim in &[32u32, 64, 128] {
                // ---- paged ----
                let swz = head_dim * 8;
                let q_stage = next_n * heads * head_dim;
                let w_aligned = align_up(next_n * heads * 4, swz);
                let q_pipe = Q_STAGES * (q_stage + w_aligned) + align_up(Q_STAGES * 8 * 2, swz);
                let kv_stage = 64 * head_dim;
                let s_aligned = align_up(64 * 4, swz);
                let kv_pipe = KV_STAGES * (kv_stage + s_aligned) + align_up(KV_STAGES * 8 * 2, swz);
                // Kernel: Q pipe + 4 KV pipes (barriers inside the padded
                // areas).  Launcher: same + the (unused) UMMA-barrier tail.
                let kernel_total = q_pipe + 4 * kv_pipe;
                let host = q_pipe + 4 * kv_pipe + 4 * 2 * 8 + 4;
                assert!(kernel_total <= host);
                assert!(host <= CAP, "paged smem {host} exceeds capacity");
                assert_eq!(q_stage % swz, 0);
                assert_eq!(kv_stage % swz, 0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Offline NVRTC compile checks (PTX compute_90a + SASS sm_90a) — the
// GPU-less verification plane.  The bodies are the exact strings the
// launchers pass to jit::get_kernel.
// ---------------------------------------------------------------------------

fn mqa_sm90_unit() -> String {
    format!(
        "{}\n{}",
        jit::kernel_src::WGMMA_H,
        jit::kernel_src::MQA_SM90
    )
}

#[test]
fn compile_check_sm90_mqa_contiguous() {
    let tu = mqa_sm90_unit();
    for (heads, head_dim) in [(32u32, 128u32), (32, 64), (64, 128), (64, 32)] {
        let body = api_sm90_mqa::mqa_logits_sm90_body(heads, head_dim, 132).unwrap();
        let r = jit::compile_check_kernel(&tu, &body, "90a", "sm90-mqa").unwrap();
        assert!(
            r.ptx_len > 10_000,
            "suspiciously small PTX for heads={heads}"
        );
        assert!(
            r.cubin_len.unwrap_or(0) > 0,
            "SASS generation failed for contiguous heads={heads} head_dim={head_dim}"
        );
    }
}

#[test]
fn compile_check_sm90_mqa_paged() {
    let tu = mqa_sm90_unit();
    for (next_n, heads, head_dim) in [
        (2u32, 64u32, 128u32),
        (2, 32, 128),
        (1, 64, 64),
        (1, 32, 64),
    ] {
        let body = api_sm90_mqa::mqa_paged_logits_sm90_body(next_n, heads, head_dim).unwrap();
        let r = jit::compile_check_kernel(&tu, &body, "90a", "sm90-mqa-paged").unwrap();
        assert!(
            r.ptx_len > 10_000,
            "suspiciously small PTX for n={next_n} h={heads}"
        );
        assert!(
            r.cubin_len.unwrap_or(0) > 0,
            "SASS generation failed for paged next_n={next_n} heads={heads} head_dim={head_dim}"
        );
    }
}

#[test]
fn compile_check_sm90_mqa_metadata() {
    let tu = mqa_sm90_unit();
    for aligned_batch in [32u32, 64, 128] {
        let body = api_sm90_mqa::mqa_paged_logits_sm90_metadata_body(aligned_batch, 132);
        let r = jit::compile_check_kernel(&tu, &body, "90a", "sm90-mqa-meta").unwrap();
        assert!(r.ptx_len > 1_000);
        assert!(
            r.cubin_len.unwrap_or(0) > 0,
            "SASS generation failed for metadata aligned_batch={aligned_batch}"
        );
    }
}
