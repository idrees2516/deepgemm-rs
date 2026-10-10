//! CPU-only tests for the SM100 sparse MQA (DSA top-k indexer) port.
//!
//! (a) Pure-Rust models of the layout/scheduler math that the CUDA metadata
//!     kernel implements on-device: the merge-path dedup of the two Q
//!     tokens' selected-block lists, split-local slot compression, the
//!     contiguous-mode per-SM schedule split, the paged-mode bounded-entry
//!     split, the 16B-chunk swizzle, and the smem/metadata buffer sizing.
//! (b) Offline NVRTC compile checks (PTX for compute_100a + SASS for
//!     sm_100a) of both kernels — the GPU-less verification plane; each
//!     wrapper embeds a `static_assert(sizeof(...))` that cross-checks the
//!     Rust smem math against the device struct layout.

use deepgemm::api_sparse_mqa::{
    sparse_mqa_metadata_body, sparse_mqa_metadata_bytes, sparse_mqa_metadata_smem_bytes,
    sparse_mqa_workspace_bytes, SparseMqaConfig, SPARSE_MQA_BLOCK_Q, SPARSE_MQA_SPLITS_PER_ENTRY,
};
use deepgemm::jit::{self, kernel_src};
use deepgemm::types::Dtype;

const NUM_SMS: u32 = 148;

fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

// ---------------------------------------------------------------------------
// (a) Scheduler math models
// ---------------------------------------------------------------------------

/// One merged KV block: which logical block it is, whether each of the two Q
/// tokens selected it, and at which slot of each token's top-k list.
#[derive(Clone, Copy, Debug)]
struct Merged {
    block: u32,
    in_q0: bool,
    in_q1: bool,
    slot0: u32,
    slot1: u32,
}

/// Sequential reference of the metadata kernel's warp-parallel merge: two
/// sorted selection lists walk forward together; equal blocks merge into one
/// output entry present in both. (`~0u` plays the role of the device's
/// "cursor past the end" sentinel.)
fn merge_selected(q0: &[u32], q1: &[u32]) -> Vec<Merged> {
    assert!(q0.windows(2).all(|w| w[0] <= w[1]), "q0 must be sorted");
    assert!(q1.windows(2).all(|w| w[0] <= w[1]), "q1 must be sorted");
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::new();
    while i < q0.len() || j < q1.len() {
        let b0 = if i < q0.len() { q0[i] } else { u32::MAX };
        let b1 = if j < q1.len() { q1[j] } else { u32::MAX };
        let in_q0 = b0 <= b1;
        let in_q1 = b1 <= b0;
        out.push(Merged {
            block: b0.min(b1),
            in_q0,
            in_q1,
            slot0: if in_q0 { i as u32 } else { 0 },
            slot1: if in_q1 { j as u32 } else { 0 },
        });
        i += in_q0 as usize;
        j += in_q1 as usize;
    }
    out
}

#[test]
fn merge_selected_dedups_and_keeps_slots() {
    let q0 = [0u32, 1, 3, 7, 8];
    let q1 = [1u32, 2, 7, 9];
    let m = merge_selected(&q0, &q1);
    // Sorted, unique, superset of both inputs.
    let blocks: Vec<u32> = m.iter().map(|b| b.block).collect();
    let mut want: Vec<u32> = q0.iter().chain(q1.iter()).copied().collect();
    want.sort_unstable();
    want.dedup();
    assert_eq!(blocks, want);
    // Shared blocks are present in both lists with both slot indices.
    for e in &m {
        if e.in_q0 {
            assert_eq!(q0[e.slot0 as usize], e.block);
        }
        if e.in_q1 {
            assert_eq!(q1[e.slot1 as usize], e.block);
        }
        // A block shared by both tokens must carry equal block ids.
        if e.in_q0 && e.in_q1 {
            assert_eq!(q0[e.slot0 as usize], q1[e.slot1 as usize]);
        }
    }
    // Every input consumed exactly once.
    assert_eq!(m.iter().filter(|e| e.in_q0).count(), q0.len());
    assert_eq!(m.iter().filter(|e| e.in_q1).count(), q1.len());
    // Edge cases.
    assert!(merge_selected(&[], &[]).is_empty());
    assert_eq!(merge_selected(&[5], &[]).len(), 1);
    assert_eq!(merge_selected(&[5], &[5]).len(), 1);
}

/// Split-local slot compression (KVSplitHeader::q?_slot_base +
/// KVBlockInfo::packed_slot_offsets): the first merged block of each split
/// anchors both tokens' slot indices; every later block stores the
/// difference. Round-trips to the original slot and stays below the invalid
/// sentinel.
#[test]
fn slot_compression_round_trips() {
    let q0: Vec<u32> = (0..37u32).map(|i| i * 3).collect(); // 0,3,6,...108
    let q1: Vec<u32> = (0..23u32).map(|i| i * 5 + 1).collect(); // 1,6,11,...
    let merged = merge_selected(&q0, &q1);
    const INVALID: u32 = 0xffff;
    const PRESENT: u32 = 1 << 15;
    const MASK: u32 = PRESENT - 1;

    for blocks_per_split in [8u32, 16, 40, 64] {
        for (split_idx, chunk) in merged.chunks(blocks_per_split as usize).enumerate() {
            let first = &chunk[0];
            let q0_base = first.slot0 & MASK;
            let q1_base = first.slot1 & MASK;
            for e in chunk {
                // packed_slot_offsets for this block (absent = INVALID).
                let off0 = if e.in_q0 { e.slot0 - q0_base } else { INVALID };
                let off1 = if e.in_q1 { e.slot1 - q1_base } else { INVALID };
                if off0 != INVALID {
                    assert!(off0 < INVALID, "q0 slot offset overflows 15 bits");
                    assert_eq!(q0_base + off0, e.slot0, "split {split_idx} q0 round-trip");
                }
                if off1 != INVALID {
                    assert!(off1 < INVALID);
                    assert_eq!(q1_base + off1, e.slot1, "split {split_idx} q1 round-trip");
                }
                // Presence bits: the packed word carries slot | present<<15
                // in each 16-bit half (kernel `pack_slot`, kPresentBit = 1<<15).
                let packed = (e.slot0 & MASK)
                    | (e.in_q0 as u32 * PRESENT)
                    | ((e.slot1 & MASK) | (e.in_q1 as u32 * PRESENT)) << 16;
                assert_eq!((packed >> 16) & PRESENT, (e.in_q1 as u32) * PRESENT);
                assert_eq!(packed & PRESENT, (e.in_q0 as u32) * PRESENT);
                assert!(
                    e.slot0 < PRESENT && e.slot1 < PRESENT,
                    "slot indices are 15-bit"
                );
            }
        }
    }
}

/// Contiguous-mode schedule: the split-id range [0, total) is divided evenly
/// per SM with ceil-div boundaries, then each SM's range is cut at Q-block
/// boundaries. Model of `build_contiguous_schedule`.
#[test]
fn contiguous_schedule_partitions_splits_at_q_blocks() {
    // Three Q blocks with different split counts. Tuples are
    // (split base, num splits, token base): each Q block covers exactly
    // BLOCK_Q tokens (a pair), so its token base is a BLOCK_Q multiple.
    let q_blocks: Vec<(u32, u32, u32)> = vec![(0, 5, 0), (5, 1, 2), (6, 10, 4)];
    let total: u32 = q_blocks.iter().map(|b| b.1).sum();
    let sms = 4usize;
    let split_owner = |s: u32| -> (u32, u32, u32) {
        // Linear scan (like the kernel's forward walk over kv_splits).
        let mut base = 0u32;
        for (sb, n, tb) in &q_blocks {
            if s < base + n {
                return (*sb, *n, *tb);
            }
            base += n;
        }
        panic!("split {s} out of range");
    };

    let mut covered: Vec<(u32, u32)> = Vec::new();
    for sm in 0..sms {
        let mut idx = (ceil_div(total as u64 * sm as u64, sms as u64)) as u32;
        let end = (ceil_div(total as u64 * (sm as u64 + 1), sms as u64)) as u32;
        while idx < end {
            let (qb, _, token_base) = split_owner(idx);
            let (qb_base, qb_num, _) = q_blocks.iter().find(|b| b.0 == qb).copied().unwrap();
            let entry_end = end.min(qb_base + qb_num);
            assert!(entry_end > idx, "no progress");
            assert_eq!(
                token_base % SPARSE_MQA_BLOCK_Q,
                0,
                "q blocks are token pairs"
            );
            // num_q_tokens of the entry: min(BLOCK_Q, tokens - q_token_base).
            covered.push((idx, entry_end));
            idx = entry_end;
        }
    }
    // The entries exactly partition [0, total) in order.
    let mut flat = covered.clone();
    flat.sort_unstable();
    let mut cursor = 0u32;
    for (b, e) in flat {
        assert_eq!(b, cursor, "gap/overlap in schedule coverage");
        cursor = e;
    }
    assert_eq!(cursor, total);
    // Hand-checked breakdown: SM0:(0,4); SM1:(4,5),(5,6),(6,8); SM2:(8,12); SM3:(12,16).
    assert_eq!(covered.len(), 6);
}

/// Paged-mode entry split: a Q block with `s` splits becomes
/// `ceil(s / 8)` entries; the remainder spreads one extra split over the
/// first entries. Model of `build_paged_schedule`'s inner loop.
#[test]
// Plain `entries > 0` guard reads better than a checked-div chain here.
#[allow(clippy::manual_checked_ops)]
fn paged_schedule_splits_are_bounded_and_exact() {
    for splits in 0u32..=40 {
        let entries = ceil_div(splits as u64, SPARSE_MQA_SPLITS_PER_ENTRY as u64) as u32;
        let mut ranges = Vec::new();
        let mut begin = 0u32;
        if entries > 0 {
            let base = splits / entries;
            let larger = splits % entries;
            for e in 0..entries {
                let size = base + if e < larger { 1 } else { 0 };
                ranges.push((begin, begin + size));
                begin += size;
            }
        }
        assert_eq!(begin, splits, "entries must cover all splits exactly");
        for (b, e) in &ranges {
            assert!(
                e - b <= SPARSE_MQA_SPLITS_PER_ENTRY,
                "entry exceeds the bound"
            );
            assert!(e > b);
        }
        // No entry larger than the previous one (larger entries come first).
        for w in ranges.windows(2) {
            assert!(w[1].1 - w[1].0 <= w[0].1 - w[0].0);
        }
    }
}

/// The 16B-chunk swizzle of the generic copy path (`get_swizzled_kv_chunk_idx`)
/// must reproduce the TMA swizzle atoms: 128B => chunk ^ row, 64B =>
/// chunk ^ (row / 2), both bijective so cp.async never collides in smem.
#[test]
fn kv_chunk_swizzle_matches_tma_atoms() {
    let swizzle = |mode: u32, i: u32| -> u32 {
        let mask = mode / 16 - 1;
        (i & !mask) | ((i & mask) ^ ((i >> 3) & mask))
    };
    // FP8 (mode 128): 8 chunks per 128B row; row r XORs with r.
    // Row 0 identity, row 1 swaps chunk pairs, row 2 rotates by 2.
    assert_eq!(swizzle(128, 0), 0);
    assert_eq!(swizzle(128, 8), 9);
    assert_eq!(swizzle(128, 9), 8);
    assert_eq!(swizzle(128, 10), 11);
    assert_eq!(swizzle(128, 16), 18);
    assert_eq!(swizzle(128, 17), 19);
    // Bijective over a full 8-row atom (and one SPARSE_BLOCK_KV=8 block).
    let out: std::collections::HashSet<u32> = (0..64u32).map(|i| swizzle(128, i)).collect();
    assert_eq!(out.len(), 64);
    // Row-major reference: chunk c of row r -> c ^ (r % 8).
    for r in 0..8u32 {
        for c in 0..8u32 {
            assert_eq!(swizzle(128, r * 8 + c), r * 8 + (c ^ r));
        }
    }
    // FP4 (mode 64): 4 chunks per 64B row; XOR advances every 2 rows.
    for r in 0..8u32 {
        for c in 0..4u32 {
            assert_eq!(swizzle(64, r * 4 + c), r * 4 + (c ^ (r / 2)));
        }
    }
    let out: std::collections::HashSet<u32> = (0..32u32).map(|i| swizzle(64, i)).collect();
    assert_eq!(out.len(), 32);
}

/// Upstream launch table: split/thread/stage choices and the exact smem
/// sizes of the four shipped configs (validated against the device structs
/// by the compile checks' static_asserts below).
#[test]
fn tile_config_matches_upstream_launch_table() {
    let fp8 = SparseMqaConfig::new(16, Dtype::Fp8, 8, false, None, NUM_SMS).unwrap();
    assert_eq!(fp8.split_kv, 512);
    assert_eq!(fp8.num_math_warpgroups, 4);
    assert_eq!(fp8.kv_stages, 3);
    assert_eq!(fp8.num_threads(), 768);
    assert_eq!(fp8.logits_smem_bytes(), 223_232);

    let fp8_b16 = SparseMqaConfig::new(16, Dtype::Fp8, 16, false, None, NUM_SMS).unwrap();
    assert_eq!(fp8_b16.logits_smem_bytes(), 222_208);

    let fp4 = SparseMqaConfig::new(16, Dtype::Fp4, 8, true, None, NUM_SMS).unwrap();
    assert_eq!(fp4.split_kv, 640);
    assert_eq!(fp4.num_math_warpgroups, 5);
    assert_eq!(fp4.kv_stages, 5);
    assert_eq!(fp4.num_threads(), 896);
    assert!(fp4.use_unaligned_ks);
    assert_eq!(fp4.logits_smem_bytes(), 230_912);

    let fp4_b16 = SparseMqaConfig::new(32, Dtype::Fp4, 16, false, Some(64), NUM_SMS).unwrap();
    assert_eq!(fp4_b16.page_kv, 64);
    assert_eq!(fp4_b16.logits_smem_bytes(), 229_376);
    assert!(fp4_b16.logits_smem_bytes() <= 232_448);

    // The weights TMA requires 16B rows -> head count must be a multiple of 8.
    assert!(SparseMqaConfig::new(4, Dtype::Fp8, 8, false, None, NUM_SMS).is_err());
    assert!(SparseMqaConfig::new(16, Dtype::Bf16, 8, false, None, NUM_SMS).is_err());
}

#[test]
fn metadata_buffer_sizing() {
    // Metadata smem mirrors the device SharedStorage rounded to 128B.
    assert_eq!(sparse_mqa_metadata_smem_bytes(512, 512, 8), 8_704);
    assert_eq!(sparse_mqa_metadata_smem_bytes(256, 640, 16), 4_608);
    // Output buffer: header + KVSplit[] + ScheduleEntry[].
    // (1000 tokens, 512 max blocks, 64 blocks/split => 8000 max splits,
    //  16 + 64*8 bytes per split, schedule padded to a multiple of 148.)
    assert_eq!(
        sparse_mqa_metadata_bytes(1000, 512, 8, 512, false, NUM_SMS),
        16 + 8_000 * 528 + 8_140 * 16
    );
    // Paged: one split range per (token, ceil(max_blocks / blocks_per_split)).
    assert_eq!(
        sparse_mqa_metadata_bytes(1000, 256, 8, 512, true, NUM_SMS),
        16 + 4_000 * 528 + 4_144 * 16
    );
    // Workspace: 3 counters on separate 128B lines + per-token QBlockInfo.
    assert_eq!(sparse_mqa_workspace_bytes(1000), 384 + 8_000);
    assert_eq!(sparse_mqa_workspace_bytes(0), 384);
}

// ---------------------------------------------------------------------------
// (b) Offline NVRTC compile checks (PTX compute_100a + SASS sm_100a)
// ---------------------------------------------------------------------------

fn compile_check(body: &str, tag: &str) {
    let r = jit::compile_check_kernel(kernel_src::SPARSE_MQA, body, "100a", tag)
        .unwrap_or_else(|e| panic!("{tag} failed to compile: {e}"));
    assert!(r.ptx_len > 0);
    assert!(
        r.cubin_len.is_some(),
        "{tag}: SASS (CUBIN) generation failed"
    );
    println!(
        "{tag}: PTX {} B, SASS {} B",
        r.ptx_len,
        r.cubin_len.unwrap()
    );
}

#[test]
fn compile_check_metadata_contiguous() {
    let body = sparse_mqa_metadata_body(false, false, 512, 8, 512, 0, NUM_SMS);
    compile_check(&body, "sparse-meta-contig");
}

#[test]
fn compile_check_metadata_paged() {
    let body = sparse_mqa_metadata_body(true, false, 512, 8, 512, 64, NUM_SMS);
    compile_check(&body, "sparse-meta-paged");
}

#[test]
fn compile_check_logits_contiguous_fp8() {
    let cfg = SparseMqaConfig::new(16, Dtype::Fp8, 8, false, None, NUM_SMS).unwrap();
    compile_check(
        &deepgemm::api_sparse_mqa::sparse_mqa_logits_body(&cfg),
        "sparse-logits-fp8",
    );
}

#[test]
fn compile_check_logits_contiguous_fp4_unaligned() {
    let cfg = SparseMqaConfig::new(16, Dtype::Fp4, 16, true, None, NUM_SMS).unwrap();
    compile_check(
        &deepgemm::api_sparse_mqa::sparse_mqa_logits_body(&cfg),
        "sparse-logits-fp4",
    );
}

#[test]
fn compile_check_logits_paged_fp8() {
    let cfg = SparseMqaConfig::new(32, Dtype::Fp8, 16, false, Some(64), NUM_SMS).unwrap();
    compile_check(
        &deepgemm::api_sparse_mqa::sparse_mqa_logits_paged_body(&cfg),
        "sparse-logits-paged",
    );
}
