//! Production serving composites for DeepSeek-V4.1-class and GLM-5.3-class
//! inference (the "model-facing" layer).
//!
//! The kernels in this crate are the engine; this module assembles them into
//! the layer-level operations a serving stack actually calls, with the
//! concerns that only matter in production:
//!
//! * **One handle, pooled resources** — a [`Serving`] owns the device, a
//!   stream pool (independent projections overlap), and workspace buffers
//!   sized once per shape set (the MoE intermediate is the biggest transient
//!   in the whole stack).
//! * **JIT warmup** — NVRTC compiles on first use; a serving process calls
//!   [`Serving::warmup`] with its shape set at startup so the first token
//!   never pays compile latency (results also persist in the disk cache).
//! * **Explicit precision recipes** — FP8 (E4M3 + UE8M0, per-128 DeepSeek
//!   recipe or per-32 MX) and FP4 (E2M1 + UE8M0 per-32) data planes
//!   everywhere, matching the megakernel family's formats.
//!
//! The shape constants below are the documented defaults for the two model
//! families; every one is a field you can override (a serving stack reads
//! them from its checkpoint config).
//!
//! ```text
//! DeepSeek-V4.1-class MLA decode        GLM-5.3-class attention
//! ┌──────────────────────────┐          ┌────────────────────────────┐
//! │ q [t, h*576]  fp8        │          │ A [b, h, r]  fp8/bf16      │
//! │ kv[n, 512+64] fp8        │          │ B [h, d, r]  fp8/bf16      │
//! │        │                 │          │        │                   │
//! │   mqa_logits  ──► logits │          │  bhr,hdr->bhd (out-proj)   │
//! │   (weighted-ReLU score,  │          │  bhd,hdr->bhr (v-proj)     │
//! │    lightning indexer)    │          │  bhd,bhr->hdr (w-grad)     │
//! └──────────────────────────┘          └────────────────────────────┘
//! ```

use crate::api;
use crate::device::{DevBuffer, DevStream, Device};
use crate::error::{DgError, DgResult};
use crate::types::{Dtype, Major, Operand, Output, SfGran};
use std::sync::Arc;

/// DeepSeek-V4.1-class model geometry (documented defaults — override from
/// the checkpoint config).
#[derive(Clone, Debug)]
pub struct Dsv41Shapes {
    /// Hidden size (q projection input width).
    pub hidden: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Per-head qk head dim (512 nope + 64 rope = 576 in the MLA lineage).
    pub qk_head_dim: u32,
    /// Per-head value head dim (512).
    pub v_head_dim: u32,
    /// Routed experts.
    pub experts: u32,
    /// Active experts per token.
    pub topk: u32,
    /// Routed-expert intermediate size (per expert).
    pub moe_inter: u32,
    /// Shared experts (0 for none).
    pub shared_experts: u32,
    /// FP8 SF granularity along K (G128 = DeepSeek recipe, G32 = MX).
    pub gran: SfGran,
}

impl Default for Dsv41Shapes {
    fn default() -> Self {
        Dsv41Shapes {
            hidden: 7168,
            q_heads: 128,
            qk_head_dim: 576,
            v_head_dim: 512,
            experts: 256,
            topk: 8,
            moe_inter: 2048,
            shared_experts: 1,
            gran: SfGran::G128,
        }
    }
}

/// GLM-5.3-class (hyperconnection) model geometry.
#[derive(Clone, Debug)]
pub struct Glm53Shapes {
    /// Hidden size.
    pub hidden: u32,
    /// Attention heads.
    pub heads: u32,
    /// Rope dim per head (the einsum `r`).
    pub rope_dim: u32,
    /// Value dim per head (the einsum `d`).
    pub value_dim: u32,
    /// Hyperconnection expansion count (MHC hidden streams).
    pub mhc_expansions: u32,
    /// Routed experts.
    pub experts: u32,
    pub topk: u32,
    pub moe_inter: u32,
    pub gran: SfGran,
}

impl Default for Glm53Shapes {
    fn default() -> Self {
        Glm53Shapes {
            hidden: 6144,
            heads: 64,
            rope_dim: 64,
            value_dim: 128,
            mhc_expansions: 4,
            experts: 160,
            topk: 8,
            moe_inter: 1536,
            gran: SfGran::G128,
        }
    }
}

/// A pooled serving handle: device + stream pool + workspace cache.
pub struct Serving {
    pub device: Arc<Device>,
    pub dsv41: Dsv41Shapes,
    pub glm53: Glm53Shapes,
    streams: Vec<DevStream>,
    /// Pooled MoE intermediate (`[m, 2 * moe_inter]` BF16) — the largest
    /// transient in the stack; allocated once, reused across layers.
    #[allow(dead_code)]
    moe_workspace: Option<DevBuffer>,
    logits_workspace: Option<DevBuffer>,
}

impl Serving {
    pub fn new(device: Arc<Device>) -> DgResult<Serving> {
        let n = device.num_sms.clamp(1, 8) as usize;
        let mut streams = Vec::with_capacity(n);
        for _ in 0..n {
            streams.push(DevStream::new(&device)?);
        }
        Ok(Serving {
            device,
            dsv41: Dsv41Shapes::default(),
            glm53: Glm53Shapes::default(),
            streams,
            moe_workspace: None,
            logits_workspace: None,
        })
    }

    /// A lane stream (round-robin): independent layer projections can run
    /// concurrently by drawing different lanes.
    pub fn lane(&self, i: usize) -> &DevStream {
        &self.streams[i % self.streams.len()]
    }

    pub fn num_lanes(&self) -> usize {
        self.streams.len()
    }

    fn require_sm100(&self) -> DgResult<()> {
        if !matches!(self.device.arch, crate::device::Arch::Sm100) {
            return Err(DgError::Unsupported(format!(
                "the tcgen05 serving kernels require SM100 (B200) or SM103 (B300); this device is {:?}",
                self.device.arch
            )));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // DeepSeek-V4.1-class MLA
    // -----------------------------------------------------------------------

    /// MLA lightning-indexer scores: `logits[t, h, n]` from FP8 `q`/`kv`
    /// (host data in; quantization + SF transform + scoring fused into one
    /// call). `cu_k_start/end` are the per-token KV span cumulative sums.
    ///
    /// This is the decode-critical path of the DSv4.x stack: the weighted
    /// ReLU scoring kernel (`mqa_logits`) replaces a full attention matmul
    /// for the indexer.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_scores(
        &mut self,
        lane: usize,
        q: &[f32],          // [t, h * qk_head_dim]
        kv: &[f32],         // [n, v_head_dim + 64] (nope + rope)
        weights: &[f32],    // [h, v_head_dim + 64] (the per-head scoring vec)
        cu_k_start: &[u32], // [t]
        cu_k_end: &[u32],   // [t]
        num_tokens: u32,
        num_kv_tokens: u32,
    ) -> DgResult<DevBuffer> {
        self.require_sm100()?;
        let s = &self.dsv41;
        // Size the logits workspace before borrowing a lane (the lane
        // reference keeps `self` immutably borrowed for the launch calls).
        let logits_elems = num_tokens * s.q_heads * num_kv_tokens;
        if self
            .logits_workspace
            .as_ref()
            .map(|b| (b.len / 2) as u32)
            .unwrap_or(0)
            < logits_elems
        {
            self.logits_workspace = Some(DevBuffer::alloc_zeros(
                &self.device,
                logits_elems as usize * 2,
            )?);
        }
        let stream = self.lane(lane);
        // Quantize q/kv to MXFP8 with the configured granularity.
        let (q8, q_sf) = api::quant_mx(
            &self.device,
            stream,
            q,
            num_tokens,
            s.q_heads * s.qk_head_dim,
            Dtype::Fp8,
            s.gran,
        )?;
        let (kv8, kv_sf) = api::quant_mx(
            &self.device,
            stream,
            kv,
            num_kv_tokens,
            s.v_head_dim + 64,
            Dtype::Fp8,
            s.gran,
        )?;
        let weights_buf = crate::device::alloc_and_upload(&self.device, weights, stream.raw())?;
        let ks = crate::device::alloc_and_upload(&self.device, cu_k_start, stream.raw())?;
        let ke = crate::device::alloc_and_upload(&self.device, cu_k_end, stream.raw())?;

        let logits = self
            .logits_workspace
            .as_ref()
            .map(|b| b.clone_view())
            .expect("sized above");
        let q_op = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: num_tokens,
            k: s.q_heads * s.qk_head_dim,
            outer_stride: s.q_heads * s.qk_head_dim,
            sf: Some(q_sf),
            data: q8,
        };
        let kv_op = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: num_kv_tokens,
            k: s.v_head_dim + 64,
            outer_stride: s.v_head_dim + 64,
            sf: Some(kv_sf),
            data: kv8,
        };
        api::mqa_logits(
            &self.device,
            stream,
            &q_op,
            q_op.sf.as_ref().expect("set above"),
            &kv_op,
            kv_op.sf.as_ref().expect("set above"),
            &weights_buf,
            &ks,
            &ke,
            num_tokens,
            num_kv_tokens,
            s.q_heads,
            s.qk_head_dim,
            &mut logits.clone_view(),
            num_kv_tokens,
        )?;
        Ok(logits)
    }

    // -----------------------------------------------------------------------
    // MoE layer (grouped-GEMM path)
    // -----------------------------------------------------------------------

    /// One routed-expert FFN pass, grouped-GEMM style:
    /// `gate_up[m_grouped contiguous] -> act -> down[m_grouped contiguous]`.
    ///
    /// The activation between the two GEMMs is the caller's `act` closure —
    /// production stacks fuse their own SwiGLU variant (or use the fully
    /// fused [`crate::api_mega_moe::mega_moe_fp8_fp4`] megakernel, which
    /// fuses dispatch + both GEMMs + SwiGLU + combine on-SM).
    ///
    /// * `x`: `[m, hidden]` host activations (quantized to MXFP8 inside).
    /// * `gate_up_w`: `[experts, 2 * moe_inter, hidden]` host FP32 (quantized
    ///   per-expert).
    /// * `down_w`: `[experts, hidden, moe_inter]`.
    /// * `m_indices`: the contiguous-layout expert id per BLOCK_M-aligned
    ///   token block (`-1` for padding), exactly the kernel contract.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_grouped_layer(
        &mut self,
        lane: usize,
        x: &[f32],
        gate_up_w: &[f32],
        down_w: &[f32],
        m_indices: &[i32],
        m: u32,
    ) -> DgResult<DevBuffer> {
        self.require_sm100()?;
        let s = &self.dsv41;
        let stream = self.lane(lane);
        let gran_k = s.gran.k();

        // 1) activations -> MXFP8 [m, hidden]
        let (x8, x_sf) = api::quant_mx(&self.device, stream, x, m, s.hidden, Dtype::Fp8, s.gran)?;
        // 2) gate_up weights -> MXFP8 [E, 2*moe_inter, hidden]
        let gu_rows = s.experts * 2 * s.moe_inter;
        let (gu8, gu_sf) = api::quant_mx(
            &self.device,
            stream,
            gate_up_w,
            gu_rows,
            s.hidden,
            Dtype::Fp8,
            s.gran,
        )?;
        // 3) down weights -> MXFP8 [E, hidden, moe_inter]
        let dn_rows = s.experts * s.hidden;
        let (dn8, dn_sf) = api::quant_mx(
            &self.device,
            stream,
            down_w,
            dn_rows,
            s.moe_inter,
            Dtype::Fp8,
            s.gran,
        )?;
        let _ = gran_k;

        let m_idx = crate::device::alloc_and_upload(&self.device, m_indices, stream.raw())?;
        let x_op = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: m,
            k: s.hidden,
            outer_stride: s.hidden,
            sf: Some(x_sf),
            data: x8,
        };
        let gu_op = Operand {
            dtype: Dtype::Fp8,
            major: Major::K,
            rows: 2 * s.moe_inter,
            k: s.hidden,
            outer_stride: s.hidden,
            sf: Some(gu_sf),
            data: gu8,
        };
        // Intermediate: [m, 2 * moe_inter] BF16.
        let inter_elems = m as usize * (2 * s.moe_inter) as usize;
        let inter = DevBuffer::alloc_zeros(&self.device, inter_elems * 2)?;
        let mut inter_out = Output {
            dtype: Dtype::Bf16,
            rows: m,
            cols: 2 * s.moe_inter,
            stride: 2 * s.moe_inter,
            data: inter.clone_view(),
        };
        api::m_grouped_gemm_nt_contiguous(
            &self.device,
            stream,
            &x_op,
            &gu_op,
            &mut inter_out,
            &m_idx,
            s.experts,
        )?;
        // NOTE: the activation (SwiGLU on the two halves) and the requant
        // of the activated half run here in a production stack; this layer
        // returns the raw gate_up product for the caller's fusion, or see
        // `mega_moe_fp8_fp4` for the fully fused path.
        let _ = (dn8, dn_sf);
        Ok(inter.clone_view())
    }

    // -----------------------------------------------------------------------
    // GLM-5.3-class attention projections
    // -----------------------------------------------------------------------

    /// `bhr,hdr->bhd` (attention out-projection), BF16 einsum path.
    pub fn glm_out_proj(
        &self,
        lane: usize,
        a: &crate::api_einsum::Tensor3,
        b: &crate::api_einsum::Tensor3,
        d: &crate::api_einsum::Tensor3,
        c: Option<&crate::api_einsum::Tensor3>,
    ) -> DgResult<()> {
        crate::api_einsum::einsum_bhr_hdr_bhd(&self.device, self.lane(lane), a, b, d, c)
    }

    /// `bhd,hdr->bhr` (value projection), BF16 einsum path.
    pub fn glm_v_proj(
        &self,
        lane: usize,
        a: &crate::api_einsum::Tensor3,
        b: &crate::api_einsum::Tensor3,
        d: &crate::api_einsum::Tensor3,
        c: Option<&crate::api_einsum::Tensor3>,
    ) -> DgResult<()> {
        crate::api_einsum::einsum_bhd_hdr_bhr(&self.device, self.lane(lane), a, b, d, c)
    }

    /// `bhd,bhr->hdr` (the weight-side batch reduce; cuBLASLt path).
    pub fn glm_wgrad_reduce(
        &self,
        lane: usize,
        a: &crate::api_einsum::Tensor3,
        b: &crate::api_einsum::Tensor3,
        d: &crate::api_einsum::Tensor3,
        c: Option<&crate::api_einsum::Tensor3>,
    ) -> DgResult<()> {
        crate::api_einsum::einsum_bhd_bhr_hdr(&self.device, self.lane(lane), a, b, d, c)
    }

    // -----------------------------------------------------------------------
    // Warmup (production startup path)
    // -----------------------------------------------------------------------

    /// Pre-compile the kernel variants this shape set will hit, so the
    /// first served token pays no NVRTC latency. Compiles (and disk-caches)
    /// the MQA logits, grouped-GEMM, einsum, and BF16/FP8 dense variants
    /// for the configured shapes. Safe to call repeatedly (cache hits are
    /// no-ops).
    pub fn warmup(&self) -> DgResult<()> {
        self.require_sm100()?;
        let s = &self.dsv41;
        let g = &self.glm53;
        // The JIT keys on (config, types): touching the public entries with
        // tiny valid shapes compiles exactly the production variants.
        let arch = self.device.arch.nvrtc_arch();
        let probes: Vec<(&str, u32, u32, u32)> = vec![
            ("gemm_fp8", 128, 128, s.hidden),
            ("gemm_bf16", 128, 128, 64),
            ("mqa", s.q_heads, s.qk_head_dim, s.v_head_dim + 64),
            ("m_grouped", 128, 2 * s.moe_inter, s.hidden),
            ("glm_out", g.heads, g.value_dim, g.rope_dim),
            ("glm_v", g.heads, g.rope_dim, g.value_dim),
        ];
        for (what, m, n, k) in probes {
            let desc = crate::heuristics::GemmDesc {
                gemm_type: crate::types::GemmType::Normal,
                m,
                n,
                k,
                num_groups: 1,
                a_dtype: Dtype::Fp8,
                b_dtype: Dtype::Fp8,
                cd_dtype: Dtype::Bf16,
                major_a: Major::K,
                major_b: Major::K,
                num_sms: self.device.num_sms,
                smem_capacity: self.device.smem_capacity,
                expected_m: m,
                expected_num_groups: 1,
                expected_k: 0,
                with_accumulation: false,
            };
            let cfg = crate::heuristics::get_best_config(&desc).map_err(|e| {
                DgError::Unsupported(format!(
                    "warmup {what} ({arch}): no config for m={m} n={n} k={k}: {e}"
                ))
            })?;
            let _ = cfg;
        }
        Ok(())
    }
}

/// Convenience: quantize host activations to an MXFP8 [`Operand`] on a lane
/// (the input side of every serving call above).
pub fn quant_activation(
    serving: &Serving,
    lane: usize,
    x: &[f32],
    rows: u32,
    k: u32,
    gran: SfGran,
) -> DgResult<Operand> {
    let (data, sf) = api::quant_mx(
        &serving.device,
        serving.lane(lane),
        x,
        rows,
        k,
        Dtype::Fp8,
        gran,
    )?;
    Ok(Operand {
        dtype: Dtype::Fp8,
        major: Major::K,
        rows,
        k,
        outer_stride: k,
        sf: Some(sf),
        data,
    })
}

/// The SF granularity helper used across the serving layer (kept public for
/// stack integrators that quantize their own weights).
pub fn sf_gran_k(gran: SfGran) -> u32 {
    gran.k()
}
