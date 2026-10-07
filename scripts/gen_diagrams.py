#!/usr/bin/env python3
"""Generate docs/img/*.svg for deepgemm-rs from Mermaid sources.

Sources live in the repo at docs/diagrams-src/*.mmd (editable), rendered
with mmdc into docs/img/*.svg. Re-run after editing any .mmd:
    python3 scripts/gen_diagrams.py   # from the repo, or pass the repo root
"""
import subprocess, sys, pathlib

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else __file__).resolve().parent.parent
SRC = REPO / "docs" / "diagrams-src"
OUT = REPO / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

DIAGRAMS = {
# ---------------------------------------------------------------------------
"architecture": """
flowchart TB
    subgraph HOST["Rust host (deepgemm crate)"]
        API["api.rs<br/>gemm_nt · m_grouped · mqa_logits<br/>transform_sf · quant_mx"]
        HEUR["heuristics.rs<br/>SM100 tiling search<br/>(block/cluster/stages)"]
        TMA["tma.rs<br/>cuTensorMapEncodeTiled<br/>A/B/SF/CD descriptors"]
        JIT["jit.rs<br/>NVRTC runtime compile<br/>PTX ⇄ SASS cache"]
        SYS["sys.rs<br/>cuLaunchKernelEx<br/>cluster + PDL + 48KB+ smem"]
        API --> HEUR --> TMA
        API --> JIT
        API --> SYS
        TMA --> SYS
    end
    subgraph DEV["Device (B200, SM100a)"]
        subgraph CTAX["one CTA of a persistent, L2-swizzled grid"]
            TMAE["TMA engine<br/>cp.async.bulk.tensor"]
            SMEM[("SMEM stages<br/>swizzled tiles + SF")]
            MMA["tcgen05.mma<br/>kind::mxf4 / mxf8f6f4 / f16"]
            TMEM[("TMEM<br/>accum + SF cols")]
            TC05LD["tcgen05.ld / UTCCP"]
            EPI["epilogue warps<br/>bf16 pack + TMA store"]
            TMAE --> SMEM --> MMA --> TMEM --> TC05LD --> EPI
        end
        HBM[("HBM2e")]
        HBM <--> TMAE
        EPI --> HBM
    end
    SYS -->|launch + descriptors| CTAX
    JIT -->|module load| CTAX
    style HOST fill:#eef,stroke:#88f
    style DEV fill:#efe,stroke:#8c8
""",
# ---------------------------------------------------------------------------
"gemm_pipeline": """
sequenceDiagram
    participant W0 as warp 0 (TMA)
    participant SM as SMEM stage s
    participant W1 as warp 1 (MMA)
    participant TC as tensor core
    participant TM as TMEM
    participant W4 as warps 4-7 (epilogue)
    Note over W0,W4: k-block loop, stage s = k %% numStages, phase = (k/stages) & 1
    W0->>W0: wait empty[s] (phase)
    W0->>SM: mbarrier.arrive.expect_tx(full[s], TX)
    W0->>W0: tma_load A/B/SF (multicast, L2 hint)
    SM-->>W1: full[s] flips (bytes arrived)
    W1->>TC: tcgen05.mma(s) (+ UTCCP SF)
    W1->>W1: tcgen05.commit → tmem barrier
    TC-->>TM: D accumulators written
    TM-->>W4: tmem_empty[s] (phase)
    W4->>TM: tcgen05.ld.32x32b
    W4->>SM: swizzled staging (XOR banks)
    W4->>W4: bf16 pack · fence
    W4->>W0: empty[s].arrive (slot reusable)
    W4->>W4: tma_store → HBM (+ PDL launch_dependents)
""",
# ---------------------------------------------------------------------------
"warp_specialization": """
flowchart LR
    subgraph CTA["CTA (256 threads) — cluster-paired, cta_group::2 MMA"]
        direction TB
        W0["warp 0 — TMA producer<br/>elect_one · descriptors by value"]
        W1["warp 1 — MMA issuer (leader CTA)<br/>per-stage desc table + shfl<br/>dynamic UMMA-N (swapAB)"]
        W23["warps 2,3 — SF transposers<br/>4x32 SMEM shuffle for UTCCP"]
        W47["warps 4..7 — epilogue<br/>tcgen05.ld → SMEM swizzle → TMA store"]
        REG["setmaxnreg<br/>specialized 56 ↔ math 224"]
        W0 -- "full[s]" --> W1
        W0 -- "full[s]" --> W23
        W23 -- "UTCCP TMEM SF" --> W1
        W1 -- "tmem_empty (commit)" --> W47
        W47 -- "empty[s]" --> W0
        REG -.- W0 & W1 & W47
    end
    style CTA fill:#fff8ee,stroke:#e9a13b
    style REG fill:#fdd,stroke:#d77
""",
# ---------------------------------------------------------------------------
"numeric_block_scaling": """
flowchart TB
    subgraph FORMATS["code formats"]
        FP8["E4M3 · FP8<br/>s eeee mmm (bias 7)<br/>max 448 · min 2^-9"]
        FP4["E2M1 · FP4<br/>s ee m (bias 1)<br/>grid 0, .5, 1, 1.5, 2, 3, 4, 6"]
        SF["UE8M0 scale<br/>8-bit pure exponent<br/>value = 2^(code-127)"]
    end
    subgraph PACK["SF packing — one int32 word"]
        direction LR
        B0["byte0 = sf(k)"]
        B1["byte1 = sf(k+1)"]
        B2["byte2 = sf(k+2)"]
        B3["byte3 = sf(k+3)"]
        B0 --- B1 --- B2 --- B3
    end
    subgraph EQ["block-scaled dot product (granularity g = 32 or 128)"]
        TERM["D_i,j = Σ_k codeA_i,k · codeB_j,k<br/>· 2^(sfA_i,k/g − 127) · 2^(sfB_j,k/g − 127)"]
    end
    FP8 & FP4 --> TERM
    SF --> PACK --> TERM
    PACK -. "MN-major: word(k/4, m),<br/>TMA row = 16B = 4 rows" .-> TERM
    style FORMATS fill:#eef,stroke:#88f
    style PACK fill:#efe,stroke:#8c8
    style EQ fill:#fee,stroke:#e99
""",
# ---------------------------------------------------------------------------
"tmem_map": """
flowchart TB
    subgraph TM["TMEM — 512 columns x 128 lanes x 32-bit (per SM)"]
        direction LR
        subgraph D["accumulator columns (tcgen05.mma writes)"]
            DA["0 .. UMMA_N-1<br/>(or swapped for swapAB)"]
        end
        subgraph S["SF columns (UTCCP writes)"]
            SA["SF_BLOCK_K · 32 cols"]
        end
        OV["overlap zone: cols ≥ 512 alias<br/>accum − 512 (guarded by tmem barriers)"]
    end
    MMA["tcgen05.mma → D region"]
    UTC["tcgen05.cp UTCCP → SF region"]
    LD32["tcgen05.ld.32x32b<br/>lane i reads row i"]
    MMA --> D
    UTC --> S
    LD32 --> D
    LD32 --> S
    style TM fill:#f7f7ff,stroke:#99a
    style OV fill:#fdd,stroke:#d77
""",
# ---------------------------------------------------------------------------
"sandbox_planes": """
flowchart TB
    subgraph NOGPU["No-GPU plane (sandbox / CI / laptop)"]
        CC["deepgemm-bench compile-check<br/>--arch 100a | 103a | 120a"]
        PTX["NVRTC PTX pass<br/>compute_100a (runtime path)"]
        SASS["NVRTC CUBIN pass<br/>sm_100a (ptxas SASS)"]
        GOLD["cargo test (CPU)<br/>golden model: formats,<br/>SF packing, GEMM, MQA"]
        CC --> PTX & SASS
        PTX & SASS --> OK["14/14 kernels validated"]
        GOLD --> OK2["18/18 tests green"]
    end
    subgraph GPU["B200 plane (real hardware)"]
        SMK["smoke (JIT on device)"]
        E2E["cargo test --features e2e<br/>GPU vs golden bit-exact"]
        BENCH["bench fp4_nt_native ...<br/>TFLOPS vs peak"]
        SMK --> E2E --> BENCH
    end
    OK -. same kernels .-> SMK
    OK2 -. same model .-> E2E
    style NOGPU fill:#efe,stroke:#8c8
    style GPU fill:#eef,stroke:#88f
""",
}

cfg = """{"theme":"default","themeVariables":{"fontFamily":"DejaVu Sans"}}"""
(SRC / "config.json").write_text(cfg)

fail = 0
for name, src in DIAGRAMS.items():
    mmd = SRC / f"{name}.mmd"
    mmd.write_text(src.strip() + "\n")
    svg = OUT / f"{name}.svg"
    r = subprocess.run(
        ["mmdc", "-i", str(mmd), "-o", str(svg), "-c", str(SRC / "config.json"), "-b", "transparent"],
        capture_output=True, text=True)
    if r.returncode != 0 or not svg.exists():
        print(f"[FAIL] {name}: {r.stderr[-400:]}")
        fail += 1
    else:
        print(f"[ok] {name}: {svg.stat().st_size} B")

sys.exit(1 if fail else 0)
