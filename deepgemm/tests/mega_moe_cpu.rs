//! CPU-only tests for the SM100 fp8xfp4 MegaMoE megakernel port
//! (`kernels/mega_moe_sm100.cu` + `src/api_mega_moe.rs`).
//!
//! (a) Pure-Rust mirrors of every device-side layout formula the kernel
//!     computes — the `MegaMoESignals` control-block offsets (cross-checked
//!     against the *frozen* `moe_layout` contract), the `Workspace` dispatch
//!     metadata offsets, the `MegaMoEBuffer` region chain, the ring
//!     capacity (`get_num_max_live_pool_blocks`), the launch-config table
//!     (BLOCK_M candidates, thread/register budget), the gate/up interleave
//!     (gran 8, with a worked example) and the SF ring transposition
//!     (`transform_sf_token_idx` / `_transpose_sf_for_utccp`).
//! (b) Offline NVRTC compile checks (PTX for compute_100a + SASS for
//!     sm_100a) of the pre-existing megakernel for `kNumRanks = 1` and
//!     `kNumRanks = 4` — the wrapper embeds `static_assert`s that pin the
//!     device struct sizes to the Rust mirrors, so a green compile *is* the
//!     layout cross-check.
//!
//! The launcher module is not yet re-exported through `lib.rs` (frozen in
//! this task); it is compiled into this test crate via `#[path]`, with the
//! library re-exported at the crate root so `crate::*` paths resolve.

pub use deepgemm::{device, error, jit, locality, moe_layout, sys, tma, types};
pub use deepgemm::{error::DgError, types::Dtype};

// The GPU launcher path (weights struct, TMA builders, launch) is exercised
// only at runtime on a device; suppress dead-code noise for the test-crate
// copy of the module (it is live API once re-exported through `lib.rs`).
#[allow(dead_code)]
use deepgemm::api_mega_moe;

use api_mega_moe::{
    get_num_l1_warmup_waves, get_num_max_live_pool_blocks, interleave_weights, mega_moe_body,
    mega_moe_buffer_layout, mega_moe_ring_tokens, mega_moe_signals_bytes,
    mega_moe_symm_buffer_bytes, mega_moe_unit, transpose_sf_for_utccp, MegaMoeConfig,
    CANDIDATE_BLOCK_MS, LCM_CANDIDATE_BLOCK_M, SMEM_CAPACITY,
};
use moe_layout::{
    num_max_pool_tokens, MegaMoESignalsLayout, MoEWorkspace, NUM_DEVICE_LOCALITY_DOMAINS,
};

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}
fn align_up(a: u32, b: u32) -> u32 {
    a.div_ceil(b) * b
}

// A B200-like problem shape used throughout (and by the compile checks):
// 148 SMs, DeepSeek-V3-class MoE geometry.
const NUM_SMS: u32 = 148;
const HIDDEN: u32 = 7168;
const INTERMEDIATE: u32 = 2048;
const NUM_EXPERTS: u32 = 256;
const TOPK: u32 = 8;
const T: u32 = 1920; // align(1920, kLCMCandidateBlockM)

// ---------------------------------------------------------------------------
// (a) Signals control block: device offsets vs the frozen moe_layout contract
// ---------------------------------------------------------------------------

/// The device-side `MegaMoESignals<num_ranks>` region mirrors — the exact
/// constexpr model from the kernel (section 1), which must equal the frozen
/// `moe_layout::MegaMoESignalsLayout` for every rank count.
#[test]
fn signals_region_matches_frozen_moe_layout() {
    for &nr in &[1u32, 2, 4, 8, 32, 64] {
        let frozen = MegaMoESignalsLayout::new(nr as usize);
        assert_eq!(
            mega_moe_signals_bytes(nr) as usize,
            frozen.total_bytes,
            "signals total for {nr} ranks"
        );
        // Head / per-rank region offsets, in order.
        let off_combine_ready = align_up(16 + 4 + 8 + 4 * NUM_DEVICE_LOCALITY_DOMAINS * 4, 128);
        assert_eq!(off_combine_ready, 256, "frozen offset_combine_ready");
        assert_eq!(off_combine_ready as usize, frozen.offset_combine_ready);
        let off_peer = align_up(256 + nr * 8, 128);
        assert_eq!(
            off_peer as usize, frozen.offset_peer_grid_idx,
            "peer for {nr}"
        );
        let off_send = off_peer + align_up(nr * 8, 128);
        assert_eq!(
            off_send as usize, frozen.offset_expert_send,
            "send for {nr}"
        );
        let off_ring = off_send as u64 + 3 * 2048 * 8;
        assert_eq!(off_ring as usize, frozen.offset_ring_signals);
        let off_shared_l2 = off_ring + (1u64 << 20) * 20;
        assert_eq!(off_shared_l2 as usize, frozen.offset_shared_l2);
        assert_eq!(
            off_shared_l2 as usize + (1usize << 15) * 4,
            frozen.total_bytes
        );
    }
}

/// `Workspace::get_*_ptr` byte offsets (the device-side pointer arithmetic
/// the kernel performs against the signals block).
#[test]
fn workspace_dispatch_metadata_offsets() {
    let (nr, e, t, topk) = (4u32, 256u32, T, 8u32);
    let ws = MoEWorkspace::new(nr, e, t, topk);
    let signals = mega_moe_signals_bytes(nr);
    // get_src_token_topk_idx_ptr(l, r, tok) = signals + (l*nr + r)*t + tok
    let src = |l: u32, r: u32, tok: u32| {
        signals + ((l as u64 * nr as u64 + r as u64) * t as u64 + tok as u64) * 4
    };
    assert_eq!(src(0, 0, 0), signals);
    // The metadata region begins where the [local expert][rank][token] table
    // of all `num_experts_per_rank` local experts ends:
    // get_token_src_metadata_ptr(0) == get_src_token_topk_idx_ptr(epr, 0, 0).
    let epr = e / nr;
    assert_eq!(src(epr, 0, 0), signals + (e as u64 * t as u64 * 4));
    // And `Workspace::get_num_bytes` covers both, 16B-aligned:
    let expect = signals + e as u64 * t as u64 * 4 + ws.num_max_pool_tokens as u64 * 12;
    let expect = (expect + 15) & !15u64;
    assert_eq!(
        api_mega_moe::mega_moe_workspace_bytes(nr, e, t, topk),
        expect
    );
    // Pool capacity itself is the frozen formula.
    assert_eq!(
        ws.num_max_pool_tokens,
        num_max_pool_tokens(nr, t, topk, e / nr)
    );
    assert_eq!(
        ws.num_shared_l2_pool_blocks,
        ceil_div(t, 8),
        "shared-L2 pool blocks = ceil(T / kMinCandidateBlockM)"
    );
}

// ---------------------------------------------------------------------------
// (a) Symmetric-buffer region chain (device MegaMoEBuffer mirror)
// ---------------------------------------------------------------------------

#[test]
fn buffer_layout_matches_device_construction() {
    // No shared experts (the compile-check config).
    let (nr, e, t, topk) = (1u32, NUM_EXPERTS, T, TOPK);
    let (ring, sf_ring) =
        mega_moe_ring_tokens(nr, e, t, topk, NUM_SMS, HIDDEN, INTERMEDIATE).expect("ring capacity");
    let lay = mega_moe_buffer_layout(nr, e, t, topk, HIDDEN, INTERMEDIATE, ring, sf_ring, 0);

    // Hand-rolled device construction, region by region.
    let ws_bytes = api_mega_moe::mega_moe_workspace_bytes(nr, e, t, topk);
    let mut off = ws_bytes;
    assert_eq!(lay.off_input_token, off);
    off += t as u64 * HIDDEN as u64; // x fp8
    assert_eq!(lay.off_input_sf, off);
    off += t as u64 * (HIDDEN / 32) as u64; // x SF packed UE8M0
    assert_eq!(lay.off_input_topk_idx, off);
    off += t as u64 * topk as u64 * 8; // int64 indices
    assert_eq!(lay.off_input_topk_weights, off);
    off += t as u64 * topk as u64 * 4; // f32 weights
                                       // Shared experts disabled: their regions are empty and contiguous.
    assert_eq!(lay.off_shared_l1_sf, off);
    assert_eq!(lay.off_shared_l2_token, off);
    assert_eq!(lay.off_shared_l2_sf, off);
    // Routed rings.
    assert_eq!(lay.off_l1_token, off);
    off += ring as u64 * HIDDEN as u64;
    assert_eq!(lay.off_l1_sf, off);
    off += sf_ring as u64 * (HIDDEN / 32) as u64;
    assert_eq!(lay.off_l1_topk_weights, off);
    off += ring as u64 * 4;
    assert_eq!(lay.off_l2_token, off);
    off += ring as u64 * INTERMEDIATE as u64;
    assert_eq!(lay.off_l2_sf, off);
    off += sf_ring as u64 * (INTERMEDIATE / 32) as u64;
    // Combine: one extra virtual slot only with shared experts.
    assert_eq!(lay.off_combine_token, off);
    off += topk as u64 * t as u64 * (HIDDEN * 2) as u64;
    assert_eq!(lay.total_bytes, off, "MegaMoEBuffer::get_num_bytes");
}

#[test]
fn buffer_layout_with_shared_experts() {
    let (nr, e, t, topk, s) = (1u32, NUM_EXPERTS, T, TOPK, 2u32);
    let shared_ih = INTERMEDIATE * s;
    let (ring, sf_ring) =
        mega_moe_ring_tokens(nr, e, t, topk, NUM_SMS, HIDDEN, INTERMEDIATE).unwrap();
    let lay = mega_moe_buffer_layout(nr, e, t, topk, HIDDEN, INTERMEDIATE, ring, sf_ring, s);
    let no_shared = mega_moe_buffer_layout(nr, e, t, topk, HIDDEN, INTERMEDIATE, ring, sf_ring, 0);

    // Shared regions sized by the shared-SF capacity; combine gains a slot.
    let shared_sf_rows = moe_layout::num_max_shared_sf_tokens(t);
    assert!(lay.off_shared_l2_token > lay.off_shared_l1_sf);
    assert_eq!(
        lay.off_shared_l2_token - lay.off_shared_l1_sf,
        shared_sf_rows as u64 * (HIDDEN / 32) as u64
    );
    assert_eq!(
        lay.off_shared_l2_sf - lay.off_shared_l2_token,
        t as u64 * shared_ih as u64
    );
    assert_eq!(
        lay.off_l1_token - lay.off_shared_l2_sf,
        shared_sf_rows as u64 * (shared_ih / 32) as u64
    );
    assert_eq!(
        lay.total_bytes - lay.off_combine_token,
        (topk as u64 + 1) * t as u64 * (HIDDEN * 2) as u64
    );
    assert_eq!(
        no_shared.total_bytes - no_shared.off_combine_token,
        topk as u64 * t as u64 * (HIDDEN * 2) as u64
    );
}

/// `mega_moe_symm_buffer_bytes` is exactly the span the device
/// `MegaMoEBuffer(nullptr, ...)` computes (upstream
/// `get_symm_buffer_size_for_mega_moe`).
#[test]
fn symm_buffer_size_formula() {
    for &(nr, s) in &[(1u32, 0u32), (1, 2), (4, 1), (8, 0)] {
        let total =
            mega_moe_symm_buffer_bytes(nr, NUM_EXPERTS, T, TOPK, HIDDEN, INTERMEDIATE, s, NUM_SMS)
                .expect("size");
        let (ring, sf_ring) =
            mega_moe_ring_tokens(nr, NUM_EXPERTS, T, TOPK, NUM_SMS, HIDDEN, INTERMEDIATE).unwrap();
        assert_eq!(
            total,
            mega_moe_buffer_layout(
                nr,
                NUM_EXPERTS,
                T,
                TOPK,
                HIDDEN,
                INTERMEDIATE,
                ring,
                sf_ring,
                s
            )
            .total_bytes
        );
        // Monotone in shared experts and rank count.
        let smaller =
            mega_moe_symm_buffer_bytes(nr, NUM_EXPERTS, T, TOPK, HIDDEN, INTERMEDIATE, 0, NUM_SMS)
                .unwrap();
        assert!(total >= smaller, "shared experts only add regions");
    }
    // Capacity invariants of the ring sweep (the kernel's live-pool bound).
    let (ring, sf_ring) =
        mega_moe_ring_tokens(1, NUM_EXPERTS, T, TOPK, NUM_SMS, HIDDEN, INTERMEDIATE).unwrap();
    assert_eq!(ring % LCM_CANDIDATE_BLOCK_M, 0, "ring is LCM-aligned");
    let num_max_routed = T * TOPK.min(NUM_EXPERTS);
    for &bm in CANDIDATE_BLOCK_MS.iter() {
        let pool_blocks = ceil_div(num_max_routed, bm) + NUM_EXPERTS;
        let live =
            get_num_max_live_pool_blocks(pool_blocks, NUM_SMS, HIDDEN, INTERMEDIATE).unwrap();
        assert!(
            ring as u64 >= live as u64 * bm as u64,
            "ring covers candidate BLOCK_M={bm}"
        );
        // SF ring capacity for this candidate.
        let sf = (ring / bm) * align_up(bm, 128);
        assert!(sf_ring >= sf, "SF ring covers BLOCK_M={bm}");
    }
    // Degenerate single-rank numbers stay stable (regression pin).
    // SF ring: max over candidates of (ring/bm)*align(bm,128) — the BLOCK_M=8
    // candidate dominates: (38400/8)*128 = 614400 SF slots.
    assert_eq!(ring, 38400);
    assert_eq!(sf_ring, 614400);
}

// ---------------------------------------------------------------------------
// (a) Scheduler math (warmup waves / live pool blocks)
// ---------------------------------------------------------------------------

#[test]
fn l1_warmup_waves_model() {
    // Worked example: 4 M blocks, 74 clusters (148 SMs), L1 8 N-clusters,
    // L2 3.5 -> hidden=7168: l1_n = 2*2048/256 = 16, l2_n = 7168/256 = 28.
    let l1_n = 16u32;
    let l2_n = 28u32;
    let clusters = NUM_SMS / 2;
    // First-L2-wave term: ceil(74/28)=3 M blocks * 16 / 74 = ceil(48/74)=1.
    let first_wave = ceil_div(ceil_div(clusters, l2_n) * l1_n, clusters);
    // Interleave term: diff = 0 (l1_n > l2_n is false here? 16 > 28 no) => 0;
    // so waves = ceil(16 + 0)/74 + 1 = 1 + 1 = 2... careful: ceil(16/74)=1.
    let interleave = ceil_div(l1_n, clusters) + 1;
    assert_eq!(
        get_num_l1_warmup_waves(4, clusters, l1_n, l2_n),
        first_wave.max(interleave)
    );
    // Growth in M blocks is monotone.
    let mut last = 0;
    for m in 1..=32u32 {
        let w = get_num_l1_warmup_waves(m, clusters, l1_n, l2_n);
        assert!(w >= last, "warmup waves monotone in total M blocks");
        last = w;
    }
    // L1-heavier shapes grow with the M-block count.
    let a = get_num_l1_warmup_waves(8, 8, 16, 28);
    let b = get_num_l1_warmup_waves(64, 8, 16, 28);
    assert!(b >= a);
}

#[test]
fn live_pool_blocks_worked_example() {
    // Small hand-checkable shape: ih=128 => l1_n = 1, hidden=256 => l2_n = 1.
    // clusters = 74. num_total_m_blocks = 5:
    //   l1_clusters = 5, l1_waves = 1, min_warmup = min(warmup(5,74,1,1),1)
    //   warmup: first-wave = ceil(74/1)*1/74 = 1; interleave = ceil(1+0)/74+1 = 2;
    //   -> 2 -> min(2, 1) = 1 wave. warmup_clusters = min(74, 5) = 5.
    //   live_after_warmup = ceil(5/1) = 5; frontier 0; margin = ceil(74/1)=74;
    //   -> min(5, 5 + 0 + 74) = 5.
    let live = get_num_max_live_pool_blocks(5, 148, 256, 128).unwrap();
    assert_eq!(live, 5);
    // Clamped at the total M-block count.
    let live2 = get_num_max_live_pool_blocks(3, 148, 256, 128).unwrap();
    assert_eq!(live2, 3);
    // Bad shapes are rejected.
    assert!(get_num_max_live_pool_blocks(5, 148, 256, 192).is_err());
}

// ---------------------------------------------------------------------------
// (a) Launch configuration table (heuristics port)
// ---------------------------------------------------------------------------

fn config_for(num_tokens: u32, num_experts: u32, weight_dtype: Dtype) -> MegaMoeConfig {
    // Capacity covers the requested batch (upstream asserts tokens <= cap).
    let cap = num_tokens.max(T);
    let (ring, sf_ring) =
        mega_moe_ring_tokens(1, num_experts, cap, TOPK, NUM_SMS, HIDDEN, INTERMEDIATE).unwrap();
    MegaMoeConfig::new(
        1,
        num_experts,
        cap,
        num_tokens,
        TOPK,
        HIDDEN,
        INTERMEDIATE,
        ring,
        sf_ring,
        0,
        NUM_SMS,
        weight_dtype,
        None,
        true,
    )
    .expect("config")
}

#[test]
fn block_m_heuristic_table() {
    // Expected tokens/expert = tokens * num_ranks * topk / num_experts.
    // 1920 tokens, 256 experts, topk 8: 60 (+sqrt) -> covered 67.7:
    // one 128-row block (64 < 67.7 <= 128).
    let c = config_for(1920, 256, Dtype::Fp4);
    assert_eq!(c.block_m, 128);
    assert_eq!(c.store_block_m_l1, 32);
    assert_eq!(c.store_block_m_l2, 32); // not (240 && fp4)
                                        // 8192 tokens, 32 experts: 2048 (+45.3) -> 9 blocks of 240.
    let c2 = config_for(8192, 32, Dtype::Fp4);
    assert_eq!(c2.block_m, 240);
    assert_eq!(c2.store_block_m_l1, 24);
    assert_eq!(
        c2.store_block_m_l2, 8,
        "fp4 at BLOCK_M=240 uses the 8-row L2 store"
    );
    // Few tokens: expected 2 tokens/expert (<= 10) -> the 16-row tile.
    let c3 = config_for(64, 256, Dtype::Fp8);
    assert_eq!(c3.block_m, 16, "expected 2 tokens/expert");
    let c4 = config_for(16, 512, Dtype::Fp8);
    assert_eq!(c4.block_m, 16);
    assert_eq!(c4.store_block_m_l1, 8);
    // Every selected BLOCK_M is a candidate; SF blocks UTCCP-aligned.
    for &tokens in &[64u32, 512, 1920, 4096, 8192] {
        for &e in &[32u32, 64, 128, 256, 512] {
            for &dt in &[Dtype::Fp8, Dtype::Fp4] {
                let c = config_for(tokens, e, dt);
                assert!(CANDIDATE_BLOCK_MS.contains(&c.block_m), "{} {}", tokens, e);
                assert_eq!(c.sf_block_m, align_up(c.block_m, 128).max(128));
                assert_eq!(c.sf_block_n, c.block_n);
                assert_eq!(c.block_n, 128);
                assert_eq!(c.block_k, 128);
                assert_eq!(c.load_block_m, c.block_m / 2);
            }
        }
    }
}

#[test]
fn pipeline_budget_saturates_but_never_overflows() {
    for &tokens in &[64u32, 512, 1920, 8192] {
        for &e in &[64u32, 256, 512] {
            let c = config_for(tokens, e, Dtype::Fp4);
            assert!(c.num_stages >= 2, "pipeline depth");
            assert!(c.smem_bytes <= SMEM_CAPACITY, "smem within B200 budget");
            assert_eq!(c.smem_bytes % 1024, 0, "smem is 1024-aligned");
            // Monotone in stages (the layout model is additive).
            let more = api_mega_moe::mega_moe_smem_bytes(
                c.num_experts,
                c.num_dispatch_threads / 32,
                c.num_bytes_per_pull,
                c.num_epilogue_threads / 32 / 4,
                c.num_epilogue_threads / 32,
                c.store_block_m_l1,
                c.block_n / 2,
                c.store_block_m_l2,
                c.block_n,
                c.load_block_n,
                c.num_stages + 1,
                c.load_block_m,
                c.block_k,
                c.sf_block_m,
                c.sf_block_n,
            );
            assert!(more > c.smem_bytes);
        }
    }
}

#[test]
fn thread_and_register_budget() {
    for &tokens in &[64u32, 1920, 8192] {
        for &e in &[64u32, 256, 512] {
            let c = config_for(tokens, e, Dtype::Fp8);
            assert_eq!(
                c.num_threads(),
                512,
                "128 dispatch + 128 non-epilogue + 256 epilogue"
            );
            let (dispatch, non_epi, epi) = c.register_split();
            let total = dispatch * c.num_dispatch_threads
                + non_epi * c.num_non_epilogue_threads
                + epi * c.num_epilogue_threads;
            assert!(total <= 64512, "setmaxnreg budget {total}");
            // Lean experts grant the epilogue extra registers.
            let (d0, _, e0) = config_for(tokens, 64, Dtype::Fp8).register_split();
            assert_eq!((d0, e0), (48, 208));
        }
    }
}

#[test]
fn body_embeds_layout_cross_checks() {
    let c = config_for(1920, 256, Dtype::Fp4);
    let body = mega_moe_body(&c);
    assert!(body.contains("dg::mega_moe_fp8_fp4_impl<"));
    assert!(body.contains("static_assert(sizeof(dg::MegaMoESignals<1>) == 21152256"));
    assert!(body.contains(&format!("num_bytes() == {}", c.smem_bytes)));
    // The wrapper takes all 18 maps by value + SymBuffer + locality table.
    assert_eq!(body.matches("__grid_constant__ dg::TmaMap").count(), 18);
    assert!(body.contains("dg::SymBuffer<1> sym_buffer"));
    assert!(body.contains("sm_locality_domains"));
    // Clamp bits default.
    assert!(body.contains("0x7f800000u, 1, 0"));
}

// ---------------------------------------------------------------------------
// (a) Weight preparation helpers
// ---------------------------------------------------------------------------

/// Gate/up interleave, gran 8 — worked byte-level example.
#[test]
fn interleave_weights_worked_example() {
    // n = 16 rows, row_bytes = 2; row i is tagged [i, i] for recognition.
    let row = |i: u8| vec![i, 0xffu8];
    let mut src = Vec::new();
    for i in 0u8..16 {
        src.extend_from_slice(&row(i));
    }
    let dst = interleave_weights(&src, 1, 16, 2);
    // Expected: rows [0..7 | 8..15] -> [0..7, 8..15] pairs of 8:
    // gate 0-7 then up 0-7 — for n=16, half=8, one gran block: identity.
    let expect: Vec<u8> = (0u8..16).flat_map(&row).collect();
    assert_eq!(dst, expect, "n=16 is a single gran block (identity)");

    // n = 32: gate rows 0..15, up rows 16..31 ->
    // [g0..g7 | u0..u7 | g8..g15 | u8..u15].
    let mut src32 = Vec::new();
    for i in 0u8..32 {
        src32.extend_from_slice(&row(i));
    }
    let dst32 = interleave_weights(&src32, 1, 32, 2);
    let order: Vec<u8> = [
        0u8, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 24,
        25, 26, 27, 28, 29, 30, 31,
    ]
    .to_vec();
    let expect32: Vec<u8> = order.iter().flat_map(|&i| row(i)).collect();
    assert_eq!(dst32, expect32, "two gran blocks interleave separately");

    // Multi-group (stacked experts): each group interleaves independently.
    let mut src2 = Vec::new();
    for i in 0u8..32 {
        src2.extend_from_slice(&row(i));
    }
    for i in 0u8..32 {
        src2.extend_from_slice(&row(i + 100)); // group 1 marker
    }
    let dst2 = interleave_weights(&src2, 2, 32, 2);
    assert_eq!(&dst2[..64], &expect32[..]);
    let g1: Vec<u8> = order
        .iter()
        .flat_map(|&i| row(i + 100))
        .collect::<Vec<u8>>();
    assert_eq!(&dst2[64..], &g1[..]);
}

/// `transform_sf_token_idx` (the kernel's dispatch-side SF ring write) and
/// `_transpose_sf_for_utccp` (the host-side weight-SF transform) implement
/// the same 32x4 <-> 4x32 transposition.
#[test]
fn sf_ring_transposition_math() {
    // Mirror of the kernel lambda:
    //   idx -> idx / BLOCK_M * SF_BLOCK_M + (i & ~127) + (i & 31) * 4 + ((i >> 5) & 3)
    let transform = |token_idx: u32, block_m: u32| {
        let sf_block_m = align_up(block_m, 128);
        let i = token_idx % block_m;
        token_idx / block_m * sf_block_m + (i & !127) + (i & 31) * 4 + ((i >> 5) & 3)
    };
    for &block_m in &[64u32, 128, 240] {
        let sf_block_m = align_up(block_m, 128);
        // Within each 128 group, the map is a permutation (bijective).
        for group in 0..(block_m / 128 + 1) {
            let mut seen = std::collections::HashSet::new();
            for i in 0..128u32 {
                let g = group * 128;
                if g + i >= block_m {
                    break;
                }
                let t = (i & !127) + (i & 31) * 4 + ((i >> 5) & 3);
                assert!(t < 128, "stays in the 128 group");
                assert!(seen.insert(t), "injective at BLOCK_M={block_m}");
            }
        }
        // Cross-block invariance: the block offset is SF_BLOCK_M-aligned.
        for tok in 0..(3 * block_m) {
            let t = transform(tok, block_m);
            assert_eq!(t % sf_block_m, transform(tok % block_m, block_m));
        }
        // Worked values (BLOCK_M=128): row 0->0, 1->4, 4->16, 31->124,
        // 32->1, 33->5, 8->32, 9->36: consecutive pairs (t, t+1) sit 4 apart
        // inside a 32-group — exactly the UTCCP 32x4 pattern.
        if block_m == 128 {
            assert_eq!(transform(0, 128), 0);
            assert_eq!(transform(1, 128), 4);
            assert_eq!(transform(4, 128), 16);
            assert_eq!(transform(31, 128), 124);
            assert_eq!(transform(32, 128), 1);
            assert_eq!(transform(33, 128), 5);
            assert_eq!(transform(40, 128), 33); // third 32-group: (40&31)*4 + 1
                                                // The same map on packed words == transpose_sf_for_utccp.
            let mn = 256usize;
            let packed_sf_k = 3usize;
            let words: Vec<u32> = (0..mn * packed_sf_k).map(|i| i as u32).collect();
            let t = transpose_sf_for_utccp(&words, 1, mn, packed_sf_k);
            for r in 0..mn {
                let mapped = (r & !127) + (r & 31) * 4 + ((r >> 5) & 3);
                for k in 0..packed_sf_k {
                    assert_eq!(
                        t[mapped * packed_sf_k + k],
                        words[r * packed_sf_k + k],
                        "row {r} -> {mapped}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (a) SymBuffer by-value parameter bytes
// ---------------------------------------------------------------------------

// Re-implement the byte packing mirror (private in the module).
#[test]
fn sym_buffer_param_bytes() {
    // Layout: [0:4) rank_idx, [4:8) pad, [8:8+8N) bases.
    // Mirror of dg::SymBuffer<kNumRanks> (u32, u32, u64[kNumRanks]) —
    // little-endian, no implicit padding (8+8N bytes).
    let bases = [0x10u64 << 40, 0x20 << 40, 0x30 << 40, 0x40 << 40];
    let mut bytes = vec![0u8; 8 + 8 * bases.len()];
    bytes[0..4].copy_from_slice(&3u32.to_ne_bytes());
    for (i, b) in bases.iter().enumerate() {
        bytes[8 + 8 * i..16 + 8 * i].copy_from_slice(&b.to_ne_bytes());
    }
    assert_eq!(bytes[0], 3);
    assert_eq!(&bytes[4..8], &[0; 4], "pad is zero");
    for (i, b) in bases.iter().enumerate() {
        assert_eq!(&bytes[8 + 8 * i..16 + 8 * i], &b.to_ne_bytes()[..]);
    }
}

// ---------------------------------------------------------------------------
// (b) Offline compile checks — the critical kernel verification
// ---------------------------------------------------------------------------

fn compile_check(num_ranks: u32, weight_fp8: bool) {
    jit::ensure_nvrtc().expect("libnvrtc");
    let (ring, sf_ring) = mega_moe_ring_tokens(
        num_ranks,
        NUM_EXPERTS,
        T,
        TOPK,
        NUM_SMS,
        HIDDEN,
        INTERMEDIATE,
    )
    .expect("ring capacity");
    let cfg = MegaMoeConfig::new(
        num_ranks,
        NUM_EXPERTS,
        T,
        T,
        TOPK,
        HIDDEN,
        INTERMEDIATE,
        ring,
        sf_ring,
        0,
        NUM_SMS,
        if weight_fp8 { Dtype::Fp8 } else { Dtype::Fp4 },
        None,
        true,
    )
    .expect("config");
    let body = mega_moe_body(&cfg);
    let res = jit::compile_check_kernel(
        mega_moe_unit(),
        &body,
        "100a",
        &format!("mega-moe-r{num_ranks}-fp{}", if weight_fp8 { 8 } else { 4 }),
    )
    .expect("NVRTC compile (PTX + SASS) of the megakernel");
    assert!(res.cubin_len.is_some(), "SASS generation must succeed");
    println!(
        "mega-moe rank{num_ranks} fp{}: block_m={} stages={} smem={} ring={} sf_ring={}: \
         PTX {} B, SASS {} B",
        if weight_fp8 { 8 } else { 4 },
        cfg.block_m,
        cfg.num_stages,
        cfg.smem_bytes,
        ring,
        sf_ring,
        res.ptx_len,
        res.cubin_len.unwrap()
    );
}

#[test]
fn compile_check_rank1() {
    compile_check(1, false);
}

#[test]
fn compile_check_rank4() {
    compile_check(4, true);
}
