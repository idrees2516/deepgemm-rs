//! Tiling heuristics: a faithful port of upstream DeepGEMM's
//! `csrc/jit_kernels/heuristics/sm100.hpp` config search, plus environment
//! overrides so tile shapes can be tuned on real hardware:
//!
//! - `DG_BLOCK_M`, `DG_BLOCK_N`, `DG_BLOCK_K`: pin the tile shape.
//! - `DG_CLUSTER_M`, `DG_CLUSTER_N`: pin the cluster dims (1 or 2).
//! - `DG_SWAP_AB`: force swap-AB on/off (`1`/`0`).
//! - `DG_NUM_STAGES`: pin the number of pipeline stages.
//! - `DG_MULTICAST`: `1` or `2` (max cluster size).
//! - `DG_PRINT_CONFIG=1`: print the chosen config to stderr.

use crate::error::{DgError, DgResult};
use crate::types::{Dtype, GemmType, Major, SfGran};
use std::env;

pub const SMEM_CAPACITY_FALLBACK: u32 = 232448;
pub const NUM_MAX_STAGES: u32 = 32;

fn env_u32(name: &str) -> Option<u32> {
    env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

pub fn print_configs_enabled() -> bool {
    env::var("DG_PRINT_CONFIGS").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Pinned overrides from the environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct Overrides {
    pub block_m: Option<u32>,
    pub block_n: Option<u32>,
    pub block_k: Option<u32>,
    pub cluster_m: Option<u32>,
    pub cluster_n: Option<u32>,
    pub swap_ab: Option<bool>,
    pub num_stages: Option<u32>,
}

impl Overrides {
    pub fn from_env() -> Overrides {
        let swap_ab = env::var("DG_SWAP_AB").ok().and_then(|v| match v.as_str() {
            "1" | "true" | "on" => Some(true),
            "0" | "false" | "off" => Some(false),
            _ => None,
        });
        Overrides {
            block_m: env_u32("DG_BLOCK_M"),
            block_n: env_u32("DG_BLOCK_N"),
            block_k: env_u32("DG_BLOCK_K"),
            cluster_m: env_u32("DG_CLUSTER_M"),
            cluster_n: env_u32("DG_CLUSTER_N"),
            swap_ab,
            num_stages: env_u32("DG_NUM_STAGES"),
        }
    }
}

/// Tile layout choice (mirrors upstream `Layout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub swap_ab: bool,
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub cluster_m: u32,
    pub cluster_n: u32,
}

impl Layout {
    pub fn cluster_size(&self) -> u32 {
        self.cluster_m * self.cluster_n
    }
}

/// Shared-memory / swizzle configuration (mirrors upstream `StorageConfig`).
#[derive(Clone, Copy, Debug)]
pub struct StorageConfig {
    pub load_block_m: u32,
    pub load_block_n: u32,
    pub store_block_m: u32,
    pub store_block_n: u32,
    pub swizzle_a_mode: u32,
    pub swizzle_b_mode: u32,
    pub swizzle_cd_mode: u32,
}

/// Pipeline configuration (mirrors upstream `PipelineConfig`).
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub smem_size: u32,
    pub num_stages: u32,
    pub num_tma_store_stages: u32,
}

/// Launch configuration (mirrors upstream `LaunchConfig`).
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub num_sms: u32,
    pub num_non_epilogue_threads: u32, // 128: TMA + MMA + SF transposers
    pub num_epilogue_threads: u32,     // 128: epilogue warpgroup
}

#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub layout: Layout,
    pub storage: StorageConfig,
    pub pipeline: PipelineConfig,
    pub launch: LaunchConfig,
}

/// Problem description used by the heuristics.
pub struct GemmDesc {
    pub gemm_type: GemmType,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub num_groups: u32,
    pub a_dtype: Dtype,
    pub b_dtype: Dtype,
    pub cd_dtype: Dtype,
    pub major_a: Major,
    pub major_b: Major,
    pub num_sms: u32,
    pub smem_capacity: u32,
    pub expected_m: u32,
    pub expected_num_groups: u32,
}

impl GemmDesc {
    pub fn is_mxf4_mma(&self) -> bool {
        self.a_dtype == Dtype::Fp4 && self.b_dtype == Dtype::Fp4
    }

    pub fn has_sf(&self) -> bool {
        !(self.a_dtype == Dtype::Bf16 && self.b_dtype == Dtype::Bf16)
    }

    pub fn smem_pack_factor(&self) -> u32 {
        if self.is_mxf4_mma() { 2 } else { 1 }
    }
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn align_up(a: u32, b: u32) -> u32 {
    ceil_div(a, b) * b
}

fn get_swizzle_mode(inner_dim_elems: u32, elem_size: u32) -> u32 {
    let bytes = inner_dim_elems.saturating_mul(elem_size).min(128);
    match bytes {
        0 | 1..=16 => 0,
        17..=32 => 32,
        33..=64 => 64,
        _ => 128,
    }
}

/// SF block sizes aligned for UTCCP (per 128 K elements).
pub fn get_sf_block_sizes(block_m: u32, block_n: u32, has_sf: bool) -> (u32, u32) {
    if !has_sf {
        (0, 0)
    } else {
        (align_up(block_m, 128).max(128), align_up(block_n, 128).max(128))
    }
}

pub fn block_k_for(dtype: Dtype, is_mxf4: bool) -> u32 {
    // 128 * 8 / element bits: FP8 -> 128, packed FP4 -> 256, BF16 -> 64.
    if is_mxf4 {
        256
    } else {
        128 * 8 / dtype.elem_bits()
    }
}

pub fn storage_config(desc: &GemmDesc, layout: &Layout) -> StorageConfig {
    const LAYOUT_AD_M: u32 = 128;
    const UMMA_STEP_N: u32 = 16;

    let load_block_m = layout.block_m / layout.cluster_n.max(1);
    let load_block_n = layout.block_n / layout.cluster_m.max(1);
    let store_block_m = if layout.swap_ab { UMMA_STEP_N } else { LAYOUT_AD_M.min(layout.block_m) };
    let store_block_n = layout.block_n;

    // Wire element sizes: FP4 is stored packed (2 logical elements per byte).
    let a_wire = if desc.a_dtype == Dtype::Fp4 { 1 } else { desc.a_dtype.elem_size() as u32 };
    let b_wire = if desc.b_dtype == Dtype::Fp4 { 1 } else { desc.b_dtype.elem_size() as u32 };
    let cd_wire = if desc.cd_dtype == Dtype::Fp4 { 1 } else { desc.cd_dtype.elem_size() as u32 };
    let pack = desc.smem_pack_factor();

    let swizzle_a_mode = if desc.major_a == Major::K {
        get_swizzle_mode(layout.block_k / pack, a_wire)
    } else {
        get_swizzle_mode(load_block_m / pack, a_wire)
    };
    let swizzle_b_mode = if desc.major_b == Major::K {
        get_swizzle_mode(layout.block_k / pack, b_wire)
    } else {
        get_swizzle_mode(load_block_n / pack, b_wire)
    };
    let swizzle_cd_mode = get_swizzle_mode(store_block_n, cd_wire);

    StorageConfig {
        load_block_m,
        load_block_n,
        store_block_m,
        store_block_n,
        swizzle_a_mode,
        swizzle_b_mode,
        swizzle_cd_mode,
    }
}

pub fn pipeline_config(desc: &GemmDesc, layout: &Layout, storage: &StorageConfig, has_sf: bool) -> PipelineConfig {
    let cd_elem = desc.cd_dtype.elem_size() as u32;
    // Swap-AB stores (STORE_BLOCK_M x BLOCK_N) per stage; normal stores
    // (min(128, BLOCK_M) x swizzle_cd bytes).
    let smem_cd = if layout.swap_ab {
        storage.store_block_m * storage.store_block_n * cd_elem
    } else {
        storage.store_block_m * storage.swizzle_cd_mode
    } * 2 /* num TMA store stages */;

    // Barriers: worst-case 32 stages * (full, sf_full, empty) * 8B
    // + 2 epilogue stages * (tmem full/empty/overlap) * 8B + 8B.
    let smem_barriers = NUM_MAX_STAGES * 8 * 3 + 2 * 8 * 3 + 8;
    let smem_tmem_ptr = 4;

    let a_wire = if desc.a_dtype == Dtype::Fp4 { 1 } else { desc.a_dtype.elem_size() as u32 };
    let b_wire = if desc.b_dtype == Dtype::Fp4 { 1 } else { desc.b_dtype.elem_size() as u32 };
    let pack = desc.smem_pack_factor();
    let smem_a_per_stage = storage.load_block_m * layout.block_k * a_wire / pack;
    let smem_b_per_stage = storage.load_block_n * layout.block_k * b_wire / pack;

    let mut smem_sfa_per_stage = 0u32;
    let mut smem_sfb_per_stage = 0u32;
    if has_sf {
        let (sf_block_m, sf_block_n) = get_sf_block_sizes(layout.block_m, layout.block_n, true);
        smem_sfa_per_stage = sf_block_m * layout.block_k / 32;
        smem_sfb_per_stage = sf_block_n * layout.block_k / 32;
    }

    let smem_extra = smem_cd + smem_barriers + smem_tmem_ptr + 1024; // 1024B-align padding slack
    let smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
    let budget = desc.smem_capacity.max(SMEM_CAPACITY_FALLBACK);
    let mut num_stages = ((budget.saturating_sub(smem_extra)) / smem_per_stage.max(1)).min(NUM_MAX_STAGES);
    if let Some(pinned) = Overrides::from_env().num_stages {
        num_stages = pinned;
    }
    num_stages = num_stages.max(2);
    PipelineConfig {
        smem_size: smem_extra + num_stages * smem_per_stage,
        num_stages,
        num_tma_store_stages: 2,
    }
}

/// Enumerate layout candidates (port of `SM100ArchSpec::get_layout_candidates`).
pub fn layout_candidates(desc: &GemmDesc) -> Vec<Layout> {
    let ov = Overrides::from_env();
    let block_k = ov.block_k.unwrap_or_else(|| block_k_for(desc.a_dtype, desc.is_mxf4_mma()));
    let has_sf = desc.has_sf();
    let mk_alignment = 128u32; // mk alignment for contiguous layout on SM100 (256 optional upstream)

    let mut out = Vec::new();

    // M-grouped GEMMs: always swap A/B, block_n = LAYOUT_AD_M, block_m = alignment.
    if desc.gemm_type == GemmType::MGroupedContiguous || desc.gemm_type == GemmType::MGroupedMasked {
        let block_m = ov.block_m.unwrap_or(mk_alignment).max(128);
        let block_n = ov.block_n.unwrap_or(128);
        let cluster_n = if ceil_div(desc.n, block_n) % 2 == 0 && desc.num_sms % 2 == 0 { 2 } else { 1 };
        let cluster_n = ov.cluster_n.unwrap_or(cluster_n).min(2);
        let cluster_m = ov.cluster_m.unwrap_or(1);
        out.push(Layout {
            swap_ab: true,
            block_m,
            block_n,
            block_k,
            cluster_m,
            cluster_n,
        });
        return out;
    }

    let swap_range: &[bool] = match ov.swap_ab {
        Some(true) => &[true],
        Some(false) => &[false],
        None => &[false, true],
    };

    for &swap_ab in swap_range {
        let (block_ms, block_ns): (Vec<u32>, Vec<u32>) = if swap_ab {
            // After swap, BLOCK_M becomes the UMMA N dimension (up to 256).
            let mut ms: Vec<u32> = Vec::new();
            let mut b = 16u32;
            while b <= 256 {
                ms.push(b);
                b += 16;
            }
            if let Some(p) = ov.block_m {
                ms = vec![p];
            }
            let ns = match ov.block_n {
                Some(p) => vec![p],
                None => vec![128],
            };
            (ms, ns)
        } else {
            let ms = match ov.block_m {
                Some(p) => vec![p],
                None if desc.m <= 32 => vec![32],
                None if desc.m <= 64 => vec![64],
                None => vec![128],
            };
            let mut ns: Vec<u32> = Vec::new();
            if desc.k <= 256 {
                // smaller stores help epilogue overlap for small K
                let mut b = 32u32;
                while b <= 128 {
                    ns.push(b);
                    b += 32;
                }
            } else {
                let mut b = 32u32;
                while b <= 256 {
                    ns.push(b);
                    b += 32;
                }
            }
            if let Some(p) = ov.block_n {
                ns = vec![p];
            }
            (ms, ns)
        };

        let cluster_m_choices: Vec<u32> = match ov.cluster_m {
            Some(v) => vec![v],
            None => vec![1, 2],
        };
        let cluster_n_choices: Vec<u32> = match ov.cluster_n {
            Some(v) => vec![v],
            None => vec![1, 2],
        };
        for &cluster_m in &cluster_m_choices {
            for &cluster_n in &cluster_n_choices {
                if cluster_m * cluster_n > 2 {
                    continue;
                }
                if swap_ab && cluster_m > 1 {
                    continue; // after swapping, A/D only multicast on N
                }
                if !swap_ab && cluster_n > 1 {
                    continue; // upstream only supports layout A/D clusters
                }
                if desc.num_sms % (cluster_m * cluster_n) != 0 {
                    continue;
                }
                for &block_m in &block_ms {
                    for &block_n in &block_ns {
                        // swap-AB requires BLOCK_N to be the UMMA M (128).
                        if swap_ab && block_n != 128 {
                            continue;
                        }
                        // Multicast divisibility.
                        if ceil_div(desc.m, block_m) % cluster_m.max(1) != 0 {
                            continue;
                        }
                        if ceil_div(desc.n, block_n) % cluster_n.max(1) != 0 {
                            continue;
                        }
                        // A-desc reads may extend into the B stage (upstream's
                        // UMMA padding): align(block_m, 128) rows must fit in
                        // (block_m + block_n) rows of SMEM.
                        if !swap_ab {
                            let aligned_m = (block_m + 127) / 128 * 128;
                            if aligned_m > block_m + block_n {
                                continue;
                            }
                        }
                        // TMEM capacity: accum + SF columns must fit 512.
                        let (sf_m, sf_n) = get_sf_block_sizes(block_m, block_n, has_sf);
                        let sf_block_k = block_k / 128;
                        let tmem_sf_cols = if has_sf {
                            sf_m * sf_block_k / 32 + sf_n * sf_block_k / 32
                        } else {
                            0
                        };
                        let umma_n = if swap_ab { block_m } else { block_n };
                        if umma_n + tmem_sf_cols > 512 {
                            continue;
                        }
                        // MN-major operands require swizzle-aligned load blocks.
                        let a_req = if desc.major_a == Major::Mn { if desc.a_dtype == Dtype::Fp4 { 128 } else { 64 } } else { 8 };
                        if (block_m / cluster_n.max(1)) % a_req != 0 {
                            continue;
                        }
                        let b_req = if desc.major_b == Major::Mn { if desc.b_dtype == Dtype::Fp4 { 128 } else { 64 } } else { 8 };
                        if (block_n / cluster_m.max(1)) % b_req != 0 {
                            continue;
                        }
                        out.push(Layout {
                            swap_ab,
                            block_m,
                            block_n,
                            block_k,
                            cluster_m,
                            cluster_n,
                        });
                    }
                }
            }
        }
    }

    // Prefer large swizzles: when either operand is K-major, require 128B swizzle.
    if desc.major_a == Major::K || desc.major_b == Major::K {
        out.retain(|l| {
            let s = storage_config(desc, l);
            let fp4_128 = if desc.is_mxf4_mma() { 128 } else { 64 };
            let a_ok = desc.major_a != Major::K || s.swizzle_a_mode >= fp4_128;
            let b_ok = desc.major_b != Major::K || s.swizzle_b_mode >= fp4_128;
            a_ok && b_ok
        });
    }

    if out.is_empty() {
        // Fall back to a safe single-CTA layout.
        let block_m = ov.block_m.unwrap_or(128);
        let block_n = ov.block_n.unwrap_or(128);
        out.push(Layout {
            swap_ab: false,
            block_m,
            block_n,
            block_k,
            cluster_m: 1,
            cluster_n: 1,
        });
    }
    out
}

#[derive(Clone, Copy)]
struct LayoutInfo {
    num_waves: u32,
    last_wave_util: u32,
    cluster_size: u32,
    layout: Layout,
}

fn better(a: &LayoutInfo, b: &LayoutInfo) -> bool {
    // Returns true if a is better than b.
    if (a.num_waves == 1 || b.num_waves == 1) && a.num_waves != b.num_waves {
        return a.num_waves < b.num_waves;
    }
    if a.cluster_size != b.cluster_size {
        return a.cluster_size > b.cluster_size;
    }
    if a.num_waves != b.num_waves {
        return a.num_waves < b.num_waves;
    }
    if a.last_wave_util != b.last_wave_util {
        return a.last_wave_util > b.last_wave_util;
    }
    if a.layout.block_m + a.layout.block_n != b.layout.block_m + b.layout.block_n {
        return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;
    }
    a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n
}

/// Port of `get_best_config<SM100ArchSpec>`: enumerate, score, pick the winner.
pub fn get_best_config(desc: &GemmDesc) -> DgResult<GemmConfig> {
    let candidates = layout_candidates(desc);
    let has_sf = desc.has_sf();

    let mut best: Option<LayoutInfo> = None;
    for layout in &candidates {
        let storage = storage_config(desc, layout);
        let pipeline = pipeline_config(desc, layout, &storage, has_sf);
        if pipeline.smem_size > desc.smem_capacity.max(SMEM_CAPACITY_FALLBACK) {
            continue;
        }
        let num_blocks = ceil_div(desc.expected_m.max(1), layout.block_m)
            * ceil_div(desc.n, layout.block_n)
            * desc.expected_num_groups.max(1);
        let num_waves = ceil_div(num_blocks, desc.num_sms);
        let last = num_blocks % desc.num_sms;
        let util = if last == 0 { desc.num_sms } else { last };
        let info = LayoutInfo {
            num_waves,
            last_wave_util: util,
            cluster_size: layout.cluster_size(),
            layout: *layout,
        };
        best = match best {
            None => Some(info),
            Some(b) if better(&info, &b) => Some(info),
            Some(b) => Some(b),
        };
    }

    let layout = best
        .map(|i| i.layout)
        .ok_or_else(|| DgError::Unsupported("no viable SM100 layout for shape".into()))?;
    let storage = storage_config(desc, &layout);
    let pipeline = pipeline_config(desc, &layout, &storage, has_sf);
    let cfg = GemmConfig {
        layout,
        storage,
        pipeline,
        launch: LaunchConfig {
            num_sms: desc.num_sms,
            num_non_epilogue_threads: 128,
            num_epilogue_threads: 128,
        },
    };
    if print_configs_enabled() {
        eprintln!("[deepgemm-rs] config: {:?}", cfg);
    }
    Ok(cfg)
}

/// K alignment required for the m-grouped contiguous layout
/// (upstream: `get_mk_alignment_for_contiguous_layout`, 128 on SM100).
pub fn mk_alignment_for_contiguous_layout() -> u32 {
    128
}

/// TMA-aligned size for SF rows (16B granularity for int32 = 4 elements).
pub fn tma_aligned_size(size: u32, elem_size: u32) -> u32 {
    align_up(size, (16 / elem_size).max(1))
}

/// Number of packed SF rows for a given K and granularity.
pub fn sf_rows(k: u32, gran: SfGran) -> u32 {
    ceil_div(k, gran.k() * 4)
}
