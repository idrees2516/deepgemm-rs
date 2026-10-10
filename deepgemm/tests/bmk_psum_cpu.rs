//! Integration tests for the BMK/BNK einsum + PsumLayout kernels
//! (`kernels/bmk_bnk.cu` + `src/api_bmk.rs`) — all CPU-only, no GPU needed.
//!
//! Two planes of validation (the repo's "sandbox" model):
//!
//!  1. **Pure-Rust models of the psum scheduler** — a line-by-line mirror of
//!     `dg::psum::Scheduler` (the port of upstream `scheduler/gemm.cuh`) whose
//!     task enumeration and index mapping are checked against
//!     (a) the upstream *formulas* extracted from `tests/generators.py` /
//!     `csrc/apis/gemm.hpp` (the psum prefix-sum encoding, group spans,
//!     physical starts), including a fully worked example, and
//!     (b) brute-force reference enumerations (each output tile owned by
//!     exactly one (group, block) task; every k slice covered once),
//!     (c) full GEMM-semantics simulations (model-driven tile GEMM ==
//!     direct per-group reference, with zero tails and empty groups).
//!  2. **Offline NVRTC compile checks** (PTX + SASS) of every kernel variant
//!     (`90a` for the wgmma bmk body, `100a` for the tcgen05 bodies) — the
//!     same compilation the runtime JIT performs.

use deepgemm::jit::{self, kernel_src};

/// The BMK/BNK + psum translation unit (not yet re-exported through
/// `jit::kernel_src`; include it directly, exactly as `api_bmk.rs` does).
const BMK_BNK: &str = include_str!("../kernels/bmk_bnk.cu");

// ---------------------------------------------------------------------------
// Upstream psum layout formula (tests/generators.py `build_psum_layout_from_ks`)
// ---------------------------------------------------------------------------

fn align_up(a: u32, b: u32) -> u32 {
    a.div_ceil(b) * b
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// `end_g = align(end_{g-1}, alignment) + real_g` (the psum prefix sum).
fn build_psum_layout(real_sizes: &[u32], alignment: u32) -> Vec<u32> {
    let mut psum = Vec::with_capacity(real_sizes.len());
    let mut prev_end = 0u32;
    for &k in real_sizes {
        let end = align_up(prev_end, alignment) + k;
        psum.push(end);
        prev_end = end;
    }
    psum
}

/// Physical span of a psum buffer: sum of the per-group ALIGNED sizes.
fn psum_physical_span(real_sizes: &[u32], alignment: u32) -> u32 {
    real_sizes.iter().map(|&k| align_up(k, alignment)).sum()
}

// ---------------------------------------------------------------------------
// Pure-Rust model of dg::psum::Scheduler (the extracted CUDA state machine)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PsumType {
    MGroupedContiguousWithPsumLayout,
    KGroupedContiguousWithPsumLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Task {
    group: u32,
    m_block: u32,
    n_block: u32,
    /// k-grouped only: physical K start / logical K of the current group.
    k_start: u32,
    shape_k: u32,
}

/// Field-for-field mirror of the CUDA `dg::psum::Scheduler` state (some
/// fields exist only for specific GemmTypes — kept for the 1:1 mapping).
struct PsumSchedModel {
    gemm_type: PsumType,
    block_m: u32,
    block_n: u32,
    num_groups: u32,
    num_sms: u32,
    num_1d_blocks_per_group: u32,
    k_alignment: u32,

    current_iter: i64,
    num_blocks: u32,
    num_m_blocks: u32,
    num_n_blocks: u32,
    num_blocks_in_group: u32,
    grouped_layout: Vec<u32>,
    current_group_idx: u32,
    last_psum_m: u32,
    current_psum_m: u32,
    current_m_block_cumsum: u32,
    current_shape_k: u32,
    current_k_start: u32,
    current_k_end: u32,
}

impl PsumSchedModel {
    /// Mirror of `psum::Scheduler::Scheduler` (kNumMulticast = 1,
    /// kIsMulticastOnA = false — the `gemm_psum_impl` instantiation).
    // Upstream-mirroring signature: one arg per Scheduler template param.
    #[allow(clippy::too_many_arguments)]
    fn new(
        gemm_type: PsumType,
        shape_m: u32,
        shape_n: u32,
        shape_k: u32,
        layout: &[u32],
        block_m: u32,
        block_n: u32,
        num_sms: u32,
        k_alignment: u32,
    ) -> Self {
        // get_num_1d_blocks_per_group (grouping on M, multicast off).
        let pick = |candidate: u32| candidate * block_m + ceil_div(num_sms, candidate) * block_n;
        let num_1d_blocks_per_group = if pick(8) <= pick(16) { 8 } else { 16 };

        let is_k_grouped = matches!(gemm_type, PsumType::KGroupedContiguousWithPsumLayout);
        let mut m = PsumSchedModel {
            gemm_type,
            block_m,
            block_n,
            num_groups: layout.len() as u32,
            num_sms,
            num_1d_blocks_per_group,
            k_alignment,
            current_iter: -1,
            num_blocks: 0,
            num_m_blocks: ceil_div(shape_m, block_m),
            num_n_blocks: ceil_div(shape_n, block_n),
            num_blocks_in_group: 0,
            grouped_layout: layout.to_vec(),
            current_group_idx: 0,
            last_psum_m: 0,
            current_psum_m: 0,
            current_m_block_cumsum: 0,
            current_shape_k: if is_k_grouped { 0 } else { shape_k },
            current_k_start: 0,
            current_k_end: 0,
        };
        match gemm_type {
            PsumType::MGroupedContiguousWithPsumLayout => {
                m.current_psum_m = m.grouped_layout[0];
                m.num_m_blocks = ceil_div(m.current_psum_m, block_m);
            }
            PsumType::KGroupedContiguousWithPsumLayout => {
                m.num_blocks = m.num_m_blocks * m.num_n_blocks;
                m.get_next_k_group();
            }
        }
        m
    }

    fn is_k_grouped(&self) -> bool {
        matches!(self.gemm_type, PsumType::KGroupedContiguousWithPsumLayout)
    }

    /// Mirror of `get_next_k_group`.
    fn get_next_k_group(&mut self) {
        if self.is_k_grouped() {
            let next_k_end = self.grouped_layout[self.current_group_idx as usize];
            self.current_k_start = align_up(self.current_k_end, self.k_alignment);
            self.current_shape_k = next_k_end - self.current_k_start;
            self.current_k_end = next_k_end;
        } else {
            self.current_k_start += self.current_shape_k;
            self.current_shape_k = self.grouped_layout[self.current_group_idx as usize];
        }
    }

    /// Mirror of `get_swizzled_block_idx`.
    fn get_swizzled_block_idx(&mut self, block_idx: u32) -> (u32, u32) {
        let primary = self.num_m_blocks; // kIsMulticastOnA = false
        let secondary = self.num_n_blocks;
        let per_group = secondary * self.num_1d_blocks_per_group;
        let group = block_idx / per_group;
        let first = group * self.num_1d_blocks_per_group;
        let in_group = block_idx % per_group;
        self.num_blocks_in_group = self.num_1d_blocks_per_group.min(primary - first);
        let m_block = first + in_group % self.num_blocks_in_group;
        let n_block = in_group / self.num_blocks_in_group;
        (m_block, n_block)
    }

    /// Mirror of `get_global_idx` for the index kinds the kernels use.
    fn global_idx(
        &self,
        with_group_offset: bool,
        is_k_index: bool,
        shape_dim: u32,
        block_size: u32,
        block_idx: u32,
    ) -> u32 {
        if self.is_k_grouped() {
            let offset = if with_group_offset && is_k_index {
                self.current_k_start
            } else if with_group_offset {
                self.current_group_idx * shape_dim
            } else {
                0
            };
            offset + block_idx * block_size
        } else {
            // m-grouped psum: MN index offset by the group, K index flat.
            let offset = if with_group_offset && !is_k_index {
                self.current_group_idx * shape_dim
            } else {
                0
            };
            offset + block_idx * block_size
        }
    }

    /// Mirror of `get_aligned_effective_m_in_block` (ensure_zero_padding =
    /// false): the last m-block of an m-grouped-psum group is partial.
    fn aligned_effective_m_in_block(&self, m_block_idx: u32) -> u32 {
        const UMMA_STEP_N: u32 = 16;
        if matches!(self.gemm_type, PsumType::MGroupedContiguousWithPsumLayout) {
            let last_group_block = self.last_psum_m / self.block_m + self.num_m_blocks - 1;
            let rows = if m_block_idx == last_group_block {
                self.current_psum_m - m_block_idx * self.block_m
            } else {
                self.block_m
            };
            align_up(rows, UMMA_STEP_N)
        } else {
            self.block_m
        }
    }

    /// Mirror of `is_computation_valid` (m-grouped psum branch).
    fn is_computation_valid(&self, m_block_idx: u32, m_offset: u32) -> bool {
        if matches!(self.gemm_type, PsumType::MGroupedContiguousWithPsumLayout) {
            m_offset + m_block_idx * self.block_m < self.current_psum_m
        } else {
            true
        }
    }

    /// Mirror of `get_next_block` for CTA `block_idx_x`.
    fn get_next_block(&mut self, block_idx_x: u32) -> Option<Task> {
        let next_block_idx = ((self.current_iter + 1) as u32) * self.num_sms + block_idx_x;
        self.current_iter += 1;

        let (m_block, n_block);
        match self.gemm_type {
            PsumType::MGroupedContiguousWithPsumLayout => loop {
                if next_block_idx
                    < (self.current_m_block_cumsum + self.num_m_blocks) * self.num_n_blocks
                {
                    let (mb, nb) = self.get_swizzled_block_idx(
                        next_block_idx - self.current_m_block_cumsum * self.num_n_blocks,
                    );
                    // `last_psum_m` is aligned with block M
                    m_block = mb + self.last_psum_m / self.block_m;
                    n_block = nb;
                    break;
                }
                self.current_group_idx += 1;
                if self.current_group_idx == self.num_groups {
                    return None;
                }
                self.last_psum_m = align_up(self.current_psum_m, self.block_m);
                self.current_psum_m = self.grouped_layout[self.current_group_idx as usize];
                self.current_m_block_cumsum += self.num_m_blocks;
                self.num_m_blocks = ceil_div(self.current_psum_m - self.last_psum_m, self.block_m);
            },
            PsumType::KGroupedContiguousWithPsumLayout => loop {
                if self.current_group_idx == self.num_groups {
                    return None;
                }
                if next_block_idx < (self.current_group_idx + 1) * self.num_blocks {
                    let (mb, nb) = self.get_swizzled_block_idx(
                        next_block_idx - self.current_group_idx * self.num_blocks,
                    );
                    m_block = mb;
                    n_block = nb;
                    break;
                }
                self.current_group_idx += 1;
                if self.current_group_idx >= self.num_groups {
                    return None;
                }
                self.get_next_k_group();
            },
        }
        Some(Task {
            group: self.current_group_idx,
            m_block,
            n_block,
            k_start: self.current_k_start,
            shape_k: self.current_shape_k,
        })
    }

    /// Enumerate the whole persistent schedule (all CTAs, all iterations).
    /// `self` must be a freshly constructed model (no `get_next_block` yet).
    fn all_tasks(&self) -> Vec<(u32, u32, Task)> {
        assert_eq!(self.current_iter, -1, "all_tasks needs a fresh model");
        let mut out = Vec::new();
        for cta in 0..self.num_sms {
            let mut m = self.copy_state();
            while let Some(t) = m.get_next_block(cta) {
                out.push((cta, m.current_iter as u32, t));
            }
        }
        out
    }

    /// Plain copy of the state (the model owns no resources).
    fn copy_state(&self) -> PsumSchedModel {
        PsumSchedModel {
            gemm_type: self.gemm_type,
            block_m: self.block_m,
            block_n: self.block_n,
            num_groups: self.num_groups,
            num_sms: self.num_sms,
            num_1d_blocks_per_group: self.num_1d_blocks_per_group,
            k_alignment: self.k_alignment,
            current_iter: self.current_iter,
            num_blocks: self.num_blocks,
            num_m_blocks: self.num_m_blocks,
            num_n_blocks: self.num_n_blocks,
            num_blocks_in_group: self.num_blocks_in_group,
            grouped_layout: self.grouped_layout.clone(),
            current_group_idx: self.current_group_idx,
            last_psum_m: self.last_psum_m,
            current_psum_m: self.current_psum_m,
            current_m_block_cumsum: self.current_m_block_cumsum,
            current_shape_k: self.current_shape_k,
            current_k_start: self.current_k_start,
            current_k_end: self.current_k_end,
        }
    }
}

// ---------------------------------------------------------------------------
// 1. The extracted upstream formulas, as a worked example
// ---------------------------------------------------------------------------

#[test]
fn psum_layout_formulas_worked_example() {
    // real_ks = [128, 100, 0, 64], alignment 128 (the k-grouped flavor):
    //   end_0 = align(0, 128) + 128 = 128
    //   end_1 = align(128, 128) + 100 = 228
    //   end_2 = align(228, 128) + 0   = 256
    //   end_3 = align(256, 128) + 64  = 320
    let ends = build_psum_layout(&[128, 100, 0, 64], 128);
    assert_eq!(ends, vec![128, 228, 256, 320]);
    // Physical span = sum of ALIGNED sizes = 128 + 128 + 0 + 128 = 384: the
    // last group's data ends at 320 and its zero tail runs to 384.
    assert_eq!(psum_physical_span(&[128, 100, 0, 64], 128), 384);
    assert!(align_up(*ends.last().unwrap(), 128) <= 384);
    // Per-group physical K start / logical K (the scheduler's fields):
    let starts = [0u32, 128, 256, 256];
    let shapes = [128u32, 100, 0, 64];
    let mut model = PsumSchedModel::new(
        PsumType::KGroupedContiguousWithPsumLayout,
        256,
        128,
        320,
        &ends,
        128,
        128,
        148,
        128,
    );
    for g in 0..4 {
        assert_eq!(model.current_k_start, starts[g], "group {g} k_start");
        assert_eq!(model.current_shape_k, shapes[g], "group {g} shape_k");
        assert_eq!(model.current_k_end, ends[g], "group {g} k_end");
        // Advance like get_next_block's group scan does.
        model.current_group_idx += 1;
        if model.current_group_idx < 4 {
            model.get_next_k_group();
        }
    }

    // m-grouped flavor: masked-to-psum ends (generate_m_grouped_masked:
    // psum_m[j] = align(psum_m[j-1]) + masked_m[j]) with masked = [64, 100, 20]:
    //   end_0 = align(0, 128) + 64  =  64
    //   end_1 = align(64, 128) + 100 = 228
    //   end_2 = align(228, 128) + 20 = 276
    let m_ends = build_psum_layout(&[64, 100, 20], 128);
    assert_eq!(m_ends, vec![64, 228, 276]);
    assert_eq!(align_up(*m_ends.last().unwrap(), 128), 384);
    // ... and the contiguous flavor (generate_m_grouped_contiguous): groups
    // are packed at their ALIGNED sizes with the actual (unaligned) end
    // stored — which is the SAME prefix-sum formula; the only difference vs
    // the masked flavor is that the padding rows [end_g, align(end_g, 128))
    // are explicitly ZEROED (so D's padding rows come out exactly zero).
    let c_ends = {
        let mut e = Vec::new();
        let mut start = 0u32;
        for &a in &[64u32, 100, 20] {
            e.push(start + a);
            start += align_up(a, 128);
        }
        e
    };
    assert_eq!(c_ends, vec![64, 228, 276]);
    assert_eq!(c_ends, m_ends, "both psum flavors share the encoding");
}

// ---------------------------------------------------------------------------
// 2. Task enumeration vs brute force — m-grouped psum
// ---------------------------------------------------------------------------

/// Brute force from the psum semantics: group g owns the absolute m-blocks
/// [align(end_{g-1})/BM, ceil(end_g/BM)) x all n-blocks.
fn mg_psum_reference_tasks(
    ends: &[u32],
    m_psum: u32,
    n: u32,
    bm: u32,
    bn: u32,
) -> Vec<(u32, u32, u32)> {
    let num_n_blocks = ceil_div(n, bn);
    let mut out = Vec::new();
    let mut last = 0u32;
    for (g, &end) in ends.iter().enumerate() {
        let last_aligned = align_up(last, bm);
        let m_blocks = ceil_div(end - last_aligned, bm);
        for mb in 0..m_blocks {
            for nb in 0..num_n_blocks {
                out.push((g as u32, last_aligned / bm + mb, nb));
            }
        }
        last = end;
    }
    debug_assert!(align_up(*ends.last().unwrap(), bm) <= m_psum);
    out
}

#[test]
fn mg_psum_task_enumeration_matches_brute_force() {
    let bm = 128u32;
    // Masked-to-psum style: unaligned ends, gaps between groups.
    let masked = [64u32, 100, 20, 128, 5];
    let ends = build_psum_layout(&masked, bm);
    let m_psum = align_up(*ends.last().unwrap(), bm);
    let (n, num_sms) = (256u32, 148u32);

    let model = PsumSchedModel::new(
        PsumType::MGroupedContiguousWithPsumLayout,
        m_psum,
        n,
        512,
        &ends,
        bm,
        128,
        num_sms,
        bm,
    );
    let tasks = model.all_tasks();
    let reference = mg_psum_reference_tasks(&ends, m_psum, n, bm, 128);

    // Same task multiset (order differs: persistent + L2-swizzled).
    let mut got: Vec<_> = tasks
        .iter()
        .map(|(_, _, t)| (t.group, t.m_block, t.n_block))
        .collect();
    let mut want = reference.clone();
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want, "m-grouped psum task sets must match brute force");

    // Each (cta, iter) slot appears once and every task exactly once.
    assert_eq!(tasks.len(), reference.len());
    let mut slots = std::collections::HashSet::new();
    for (cta, iter, _) in &tasks {
        assert!(slots.insert((*cta, *iter)), "duplicate (cta, iter) slot");
    }

    // is_computation_valid: exactly the blocks with at least one valid row
    // (single-CTA schedule => every task visited sequentially).
    let mut model = PsumSchedModel::new(
        PsumType::MGroupedContiguousWithPsumLayout,
        m_psum,
        n,
        512,
        &ends,
        bm,
        128,
        1,
        bm,
    );
    let mut checked = 0;
    while let Some(t) = model.get_next_block(0) {
        let first_row = t.m_block * bm;
        let valid = first_row < model.current_psum_m;
        assert_eq!(
            model.is_computation_valid(t.m_block, 0),
            valid,
            "validity of block {} of group {}",
            t.m_block,
            t.group
        );
        // The block's rows must lie inside the group's padded span:
        // [last_psum_m, align(current_psum_m, BM)).
        assert!(first_row >= model.last_psum_m);
        assert!(first_row + bm <= align_up(model.current_psum_m, bm));
        // get_aligned_effective_m_in_block: full blocks stay 128; the last
        // block of a group is the 16-aligned remainder.
        let is_last = t.m_block == model.last_psum_m / bm + model.num_m_blocks - 1;
        let expect = if is_last {
            align_up(model.current_psum_m - t.m_block * bm, 16)
        } else {
            bm
        };
        assert_eq!(model.aligned_effective_m_in_block(t.m_block), expect);
        checked += 1;
    }
    assert_eq!(checked, reference.len());
}

// ---------------------------------------------------------------------------
// 3. Task enumeration vs brute force — k-grouped psum
// ---------------------------------------------------------------------------

#[test]
fn kk_psum_task_enumeration_matches_brute_force() {
    let (bm, bn) = (128u32, 128u32);
    let real_ks = [128u32, 100, 0, 64, 300];
    let k_align = 128u32;
    let ends = build_psum_layout(&real_ks, k_align);
    let sum_k = psum_physical_span(&real_ks, k_align);
    let (m, n, num_sms) = (256u32, 128u32, 148u32);

    let model = PsumSchedModel::new(
        PsumType::KGroupedContiguousWithPsumLayout,
        m,
        n,
        sum_k,
        &ends,
        bm,
        bn,
        num_sms,
        k_align,
    );
    let tasks = model.all_tasks();

    // Brute force: every group owns the full (m, n) tile grid (even the
    // EMPTY group 2 — the kernel runs one zero-filled k block and skips the
    // D store, but the tile still flows through the pipeline).
    let num_m_blocks = ceil_div(m, bm);
    let num_n_blocks = ceil_div(n, bn);
    let mut want = Vec::new();
    let mut k_start = 0u32;
    for (g, &end) in ends.iter().enumerate() {
        let start = align_up(k_start, k_align);
        for mb in 0..num_m_blocks {
            for nb in 0..num_n_blocks {
                want.push((g as u32, mb, nb, start, end - start));
            }
        }
        k_start = end;
    }
    let mut got: Vec<_> = tasks
        .iter()
        .map(|(_, _, t)| (t.group, t.m_block, t.n_block, t.k_start, t.shape_k))
        .collect();
    got.sort_unstable();
    let mut w = want.clone();
    w.sort_unstable();
    assert_eq!(got, w, "k-grouped psum task sets must match brute force");
    assert_eq!(tasks.len(), want.len());

    // The K-index mapping: get_global_idx<K> == current_k_start + kb*BLOCK_K.
    let mut model = PsumSchedModel::new(
        PsumType::KGroupedContiguousWithPsumLayout,
        m,
        n,
        sum_k,
        &ends,
        bm,
        bn,
        1,
        k_align,
    );
    while let Some(t) = model.get_next_block(0) {
        for kb in 0..4 {
            assert_eq!(
                model.global_idx(true, true, sum_k, 64, kb),
                t.k_start + kb * 64
            );
        }
        // MN indices are flat for both operands (A: m, B: n).
        assert_eq!(
            model.global_idx(false, false, m, bm, t.m_block),
            t.m_block * bm
        );
        assert_eq!(
            model.global_idx(false, false, n, bn, t.n_block),
            t.n_block * bn
        );
    }
}

// ---------------------------------------------------------------------------
// 4. GEMM semantics: the model-driven tile GEMM equals the direct reference
// ---------------------------------------------------------------------------

/// Deterministic LCG (same generator family as the golden tests).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn mg_psum_gemm_semantics_simulation() {
    // A [M_psum, K] with zeroed padding rows, B [G, N, K], D [M_psum, N].
    // The kernel computes, per task, the FULL BLOCK_M tile of A against the
    // group's B — padding rows are zeros, hence exact zeros in D.
    let (bm, bn) = (128u32, 128u32);
    let masked = [128u32, 100, 20];
    let ends = build_psum_layout(&masked, bm);
    let m_psum = align_up(*ends.last().unwrap(), bm);
    let (n, k, g_count) = (128u32, 128u32, masked.len() as u32);

    let mut rng = Lcg(0x1234_5678);
    let mut a = vec![0f32; (m_psum * k) as usize];
    let mut b = vec![0f32; (g_count * n * k) as usize];
    // Fill only the VALID rows of each group.
    let mut last = 0u32;
    for (g, &end) in ends.iter().enumerate() {
        let lo = align_up(last, bm);
        for r in lo..end {
            for c in 0..k {
                a[(r * k + c) as usize] = rng.next_f32();
            }
        }
        for idx in 0..n * k {
            b[(g as u32 * n * k + idx) as usize] = rng.next_f32();
        }
        last = end;
    }

    // Model-driven tile GEMM (exactly what gemm_psum_impl computes).
    let mut d = vec![0f32; (m_psum * n) as usize];
    // num_sms = 1: the single CTA's schedule enumerates EVERY task, which is
    // what the semantics simulation needs (the multi-CTA split only changes
    // which CTA owns which task, never the task set).
    let mut model = PsumSchedModel::new(
        PsumType::MGroupedContiguousWithPsumLayout,
        m_psum,
        n,
        k,
        &ends,
        bm,
        bn,
        1,
        bm,
    );
    while let Some(t) = model.get_next_block(0) {
        let valid = model.is_computation_valid(t.m_block, 0);
        // The B tile's global N index: group offset + n block (upstream's
        // get_global_idx<(kMajorB == K)> for B).
        // The B tile's global N index carries the group offset (upstream's
        // get_global_idx<(kMajorB == K)> for B, whose TMA map stacks groups
        // along the outer dim); D's column is LOCAL (kCDWithGroupOffset =
        // false for both psum variants).
        let b_col = model.global_idx(true, false, n, bn, t.n_block);
        let d_col = t.n_block * bn;
        for r in 0..bm {
            for c in 0..bn {
                let row = t.m_block * bm + r;
                let mut acc = 0f32;
                for kk in 0..k {
                    let av = if valid {
                        a[(row * k + kk) as usize]
                    } else {
                        0.0 // skipped blocks never reach D
                    };
                    // B is the flattened [G * N, K] buffer.
                    acc += av * b[((b_col + c) * k + kk) as usize];
                }
                if valid {
                    d[(row * n + d_col + c) as usize] = acc;
                }
            }
        }
    }

    // Direct reference: per group, D[lo:end] = A[lo:end] @ B_g^T, padding 0.
    let mut ref_d = vec![0f32; (m_psum * n) as usize];
    let mut last = 0u32;
    for (g, &end) in ends.iter().enumerate() {
        let lo = align_up(last, bm);
        for r in lo..end {
            for c in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a[(r * k + kk) as usize] * b[(g as u32 * n * k + c * k + kk) as usize];
                }
                ref_d[(r * n + c) as usize] = acc;
            }
        }
        last = end;
    }
    for (i, (&x, &y)) in d.iter().zip(ref_d.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-4,
            "mg-psum D mismatch at flat {i}: {x} vs {y}"
        );
    }
}

#[test]
fn kk_psum_gemm_semantics_simulation() {
    // A/B [SUM_K, M/N] MN-major (modeled row-major here), D [G, M, N] with
    // accumulation; zero tails between psum ends; one EMPTY group.
    let (bm, bn, bk) = (128u32, 128u32, 64u32);
    let k_align = 128u32;
    let real_ks = [128u32, 100, 0, 64];
    let ends = build_psum_layout(&real_ks, k_align);
    let sum_k = psum_physical_span(&real_ks, k_align);
    let (m, n) = (128u32, 128u32);
    let g_count = real_ks.len() as u32;

    let mut rng = Lcg(0xdead_beef);
    // Physical buffers, zero tails (upstream zero-fills the whole buffer).
    let mut a = vec![0f32; (sum_k * m) as usize];
    let mut b = vec![0f32; (sum_k * n) as usize];
    let mut k_start = 0u32;
    for &real in &real_ks {
        let lo = align_up(k_start, k_align);
        for r in 0..real {
            for c in 0..m {
                a[((lo + r) * m + c) as usize] = rng.next_f32();
            }
            for c in 0..n {
                b[((lo + r) * n + c) as usize] = rng.next_f32();
            }
        }
        k_start += real;
    }

    // Model-driven tile GEMM with accumulation (the kernel's D += partials).
    let mut d = vec![0.5f32; (g_count * m * n) as usize]; // C pre-load
    let mut model = PsumSchedModel::new(
        PsumType::KGroupedContiguousWithPsumLayout,
        m,
        n,
        sum_k,
        &ends,
        bm,
        bn,
        1,
        k_align,
    );
    while let Some(t) = model.get_next_block(0) {
        let empty = t.shape_k == 0;
        // The kernel runs max(1, ceil(shape_k / BLOCK_K)) k blocks (zero
        // tails + TMA OOB zero-fill keep partials exact).
        let k_blocks = ceil_div(t.shape_k, bk).max(1);
        for r in 0..bm {
            for c in 0..bn {
                let mut acc = 0f32;
                if !empty {
                    for kb in 0..k_blocks {
                        for kk in 0..bk {
                            let row = t.k_start + kb * bk + kk;
                            // TMA OOB rows beyond sum_k read as zeros.
                            let av = if row < sum_k {
                                a[(row * m + t.m_block * bm + r) as usize]
                            } else {
                                0.0
                            };
                            let bv = if row < sum_k {
                                b[(row * n + t.n_block * bn + c) as usize]
                            } else {
                                0.0
                            };
                            acc += av * bv;
                        }
                    }
                }
                // Empty groups skip the store entirely (upstream
                // `is_empty_group`), so D keeps the C value.
                if !empty {
                    d[((t.group * m + t.m_block * bm + r) * n + t.n_block * bn + c) as usize] +=
                        acc;
                }
            }
        }
    }

    // Direct reference: D[g] = C + A_g^T @ B_g over the LOGICAL k range.
    let mut ref_d = vec![0.5f32; (g_count * m * n) as usize];
    let mut k_start = 0u32;
    for (g, (&real, &end)) in real_ks.iter().zip(ends.iter()).enumerate() {
        let lo = align_up(k_start, k_align);
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for kk in 0..real {
                    acc += a[((lo + kk) * m + r) as usize] * b[((lo + kk) * n + c) as usize];
                }
                ref_d[((g as u32 * m + r) * n + c) as usize] += acc;
            }
        }
        k_start += real;
        let _ = end;
    }
    for (i, (&x, &y)) in d.iter().zip(ref_d.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-4,
            "kk-psum D mismatch at flat {i}: {x} vs {y}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. The bmk split-K decomposition (kernel block indexing, both archs)
// ---------------------------------------------------------------------------

/// Mirror of the kernels' block-index decomposition:
/// `slice sk -> (k_idx, s_idx) = div_rem(sk * BLOCK_K, SHAPE_K)`.
#[test]
fn bmk_split_k_slices_fold_batch_into_k() {
    let (block_k, block_m, block_n) = (64u32, 128u32, 128u32);
    // Upstream test family: s in {129, 4096, 8192},
    // (m, n, k) in {(128, 384, 128), (256, 256, 256), (384, 128, 384)}.
    for &(s, m, n, k) in &[
        (129u32, 128u32, 384u32, 128u32),
        (4096, 256, 256, 256),
        (8192, 384, 128, 384),
        (3, 128, 128, 64),
    ] {
        let num_n_blocks = ceil_div(n, block_n);
        let num_mn_blocks = num_n_blocks * ceil_div(m, block_m);
        let num_sk = s * (k / block_k);
        // The linear slice counter enumerates every (s_idx, k_block) exactly
        // once — the "split-K over heads" fold.
        let mut seen = std::collections::HashSet::new();
        for sk in 0..num_sk {
            let sk_idx = sk * block_k;
            let k_idx = sk_idx % k;
            let s_idx = sk_idx / k;
            assert!(s_idx < s && k_idx < k && k_idx % block_k == 0);
            assert!(
                seen.insert((s_idx, k_idx)),
                "duplicate (s, k-block) at sk={sk}"
            );
        }
        assert_eq!(seen.len() as u32, num_sk);
        // Grid: mn blocks x sk blocks; every slice covered once per mn block.
        for mn in 0..num_mn_blocks {
            let n_block = mn % num_n_blocks;
            let m_block = mn / num_n_blocks;
            assert!(m_block < ceil_div(m, block_m) && n_block < num_n_blocks);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Offline NVRTC compile checks (PTX + SASS) — all kernel variants
// ---------------------------------------------------------------------------

/// Wrapper bodies mirror `api_bmk::{bmk_sm100_body, bmk_sm90_body, psum_body}`
/// (the module is wired into the crate by this wave's integration step, so
/// the test carries its own copies of the same template instantiations).
const BMK_SM100_BODY: &str = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_s,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_d) {
    dg::bmk_bnk_mn_sm100_impl<256, 256, 256, 128, 128, 64, 64, 128, 128, 4, 128>
        (shape_s, tma_a, tma_b, tma_d);
}"#;

const BMK_SM90_BODY: &str = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_s,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    float* d) {
    dg::bmk_bnk_mn_sm90_impl<384, 128, 384, 128, 128, 64, 64, 4, 128, 256>
        (shape_s, tma_a, tma_b, d);
}"#;

const PSUM_MG_BODY: &str = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_cd) {
    dg::gemm_psum_impl<128, 128, 64, 8, 128, 128, 6, 256,
        (dg::psum::GemmType)5, false, false, 128, 148, false>
        (shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd);
}"#;

const PSUM_KK_ACC_BODY: &str = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_cd) {
    dg::gemm_psum_impl<128, 128, 64, 16, 128, 128, 6, 256,
        (dg::psum::GemmType)6, true, true, 256, 148, false>
        (shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd);
}"#;

const PSUM_KK_DIRECT_BODY: &str = r#"extern "C" __global__ void __dg_kernel(
    unsigned shape_m, unsigned shape_n, unsigned shape_k, int* grouped_layout,
    const __grid_constant__ dg::TmaMap tma_a, const __grid_constant__ dg::TmaMap tma_b,
    const __grid_constant__ dg::TmaMap tma_cd) {
    dg::gemm_psum_impl<128, 128, 64, 16, 128, 128, 6, 256,
        (dg::psum::GemmType)6, false, false, 384, 148, false>
        (shape_m, shape_n, shape_k, grouped_layout, tma_a, tma_b, tma_cd);
}"#;

fn assert_compiles(tu: &str, body: &str, arch: &str, name: &str) {
    let r = jit::compile_check_kernel(tu, body, arch, "bmk-psum")
        .unwrap_or_else(|e| panic!("{name} must compile ({arch}): {e}"));
    assert!(r.ptx_len > 0, "{name}: empty PTX");
    assert!(
        r.cubin_len.is_some_and(|len| len > 0),
        "{name}: SASS (CUBIN) generation failed for {arch}"
    );
}

#[test]
fn compile_check_bmk_sm100() {
    assert_compiles(BMK_BNK, BMK_SM100_BODY, "100a", "bmk_bnk_mn_sm100_impl");
}

#[test]
fn compile_check_bmk_sm90() {
    let tu = format!("{}\n{}", kernel_src::WGMMA_H, BMK_BNK);
    assert_compiles(&tu, BMK_SM90_BODY, "90a", "bmk_bnk_mn_sm90_impl");
}

#[test]
fn compile_check_psum_mg() {
    assert_compiles(BMK_BNK, PSUM_MG_BODY, "100a", "gemm_psum_impl[m-grouped]");
}

#[test]
fn compile_check_psum_kk_acc() {
    assert_compiles(
        BMK_BNK,
        PSUM_KK_ACC_BODY,
        "100a",
        "gemm_psum_impl[k-grouped, acc]",
    );
}

#[test]
fn compile_check_psum_kk_direct() {
    assert_compiles(
        BMK_BNK,
        PSUM_KK_DIRECT_BODY,
        "100a",
        "gemm_psum_impl[k-grouped, direct]",
    );
}
