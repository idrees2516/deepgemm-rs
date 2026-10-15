//! Public types: dtypes, memory majors, GEMM kinds, and shape/recipe descriptors.

/// Element type of the GEMM operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Dtype {
    /// E4M3 (FP8), `kind::mxf8f6f4` MMA path.
    Fp8,
    /// E2M1 packed two-per-byte (MXFP4), `kind::mxf4` MMA path on SM100.
    Fp4,
    /// BF16, `kind::f16` MMA path.
    Bf16,
    /// FP32 (only for C/D).
    F32,
}

impl Dtype {
    pub fn elem_size(self) -> usize {
        match self {
            Dtype::Fp8 => 1,
            Dtype::Fp4 => 1, // packed 2/byte on the wire; logical elements are 4 bits
            Dtype::Bf16 => 2,
            Dtype::F32 => 4,
        }
    }

    /// Logical element bit width.
    pub fn elem_bits(self) -> u32 {
        match self {
            Dtype::Fp8 => 8,
            Dtype::Fp4 => 4,
            Dtype::Bf16 => 16,
            Dtype::F32 => 32,
        }
    }
}

/// Contiguous (inner) dimension of the global-memory tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Major {
    /// K contiguous (row-major `[M, K]` / `[N, K]`).
    K,
    /// MN contiguous (column-major `[K, M]` / `[K, N]`).
    Mn,
}

/// Which GEMM variant a kernel implements (mirrors upstream `GemmType`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GemmType {
    Normal,
    /// MoE contiguous layout: `grouped_layout[m / BLOCK_M]` holds the expert id
    /// of token block m (negative for padding rows).
    MGroupedContiguous,
    /// MoE masked layout: `grouped_layout[g]` is the valid M of group g.
    MGroupedMasked,
    /// Batched GEMM (BMM).
    Batched,
    /// Weight-grad: A/B stacked along K; `grouped_layout[g]` = K size of
    /// group g. SM90-only (runtime tensormap patching).
    KGroupedContiguous,
}

impl GemmType {
    pub fn is_m_grouped_contiguous(self) -> bool {
        matches!(self, GemmType::MGroupedContiguous)
    }
}

/// Scaling-factor granularity along K for one operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SfGran {
    /// OCP MX: one UE8M0 scale per 32 elements (packed 4 per int32).
    G32,
    /// DeepSeek recipe: one scale per 128 elements (packed 4 per int32,
    /// each int32 covering 512 K).
    G128,
}

impl SfGran {
    pub fn k(self) -> u32 {
        match self {
            SfGran::G32 => 32,
            SfGran::G128 => 128,
        }
    }
}

/// Scale-factor tensor for one operand, SM100 layout:
/// `[ceil_div(K, gran_k * 4), TMA-aligned(M or N)]` int32, K-major outer,
/// each int32 packs 4 UE8M0 bytes.
pub struct SfTensor {
    pub rows: u32, // number of packed K rows = ceil_div(K, gran*4)
    pub cols: u32, // TMA-aligned MN
    pub gran: SfGran,
    pub buf: crate::device::DevBuffer,
}

/// An operand: data buffer + row stride + scale factors (for FP8/FP4).
pub struct Operand {
    pub dtype: Dtype,
    pub major: Major,
    /// Logical rows (M for A, N for B).
    pub rows: u32,
    /// Logical K (may be padded/packed for FP4: buffer holds K/2 bytes).
    pub k: u32,
    /// Stride of the outer (non-contiguous) dimension, in elements.
    pub outer_stride: u32,
    pub sf: Option<SfTensor>,
    pub data: crate::device::DevBuffer,
}

/// Output C/D buffer.
pub struct Output {
    pub dtype: Dtype, // Bf16 or F32
    pub rows: u32,
    pub cols: u32,
    /// Row stride (elements) = cols for contiguous.
    pub stride: u32,
    pub data: crate::device::DevBuffer,
}

// ---------------------------------------------------------------------------
// Operand view helpers (BLAS layout parity)
// ---------------------------------------------------------------------------
//
// An `Operand` is a *logical* view: `rows` x `k` with `major` telling which
// dim is contiguous in the backing buffer and `outer_stride` the pitch of the
// other dim (in elements — for FP4, logical elements, the buffer holds half
// that many bytes).
//
//   K-major  `[M, K]`: element (m, k) at m * outer_stride + k   (row-major)
//   MN-major `[K, M]`: element (m, k) at m + k * outer_stride   (col-major;
//                      the buffer is a `[K, MN]` stack, e.g. upstream's
//                      `.T.contiguous().T` generator form or a `[K, N]`
//                      activation coming straight out of a transpose-producing
//                      kernel)
//
// The BLAS wrappers (`gemm_nn`/`gemm_tn`/`gemm_tt`) mirror upstream's
// host-side transpose views: they never touch data, they only require the
// operand to already be described in the orientation that layout needs.
impl Operand {
    /// Validate that this operand is described as an MN-major logical view
    /// (the form the `nn`/`tn`/`tt` entries need) and return it unchanged.
    ///
    /// Zero-copy by construction: upstream routes these layouts into the NT
    /// kernel via `tensor.transpose(0, 1)` views; here the "view" IS the
    /// `Operand` you built. This check exists so a mis-oriented operand fails
    /// loudly at the API boundary instead of silently computing wrong
    /// numbers (a K-major `[M, K]` buffer reinterpreted as MN-major reads
    /// entirely different elements).
    pub fn as_viewed_mn_major(&self, name: &str) -> crate::error::DgResult<&Operand> {
        if self.major != Major::Mn {
            return Err(crate::error::DgError::InvalidArg(format!(
                "{name}: this BLAS layout needs an MN-major (column-major) operand \
                 view: build the Operand with `major: Major::Mn`, `rows` = logical \
                 M/N, `k` = logical K, `outer_stride` = pitch of the K slices in \
                 the `[K, MN]` backing buffer"
            )));
        }
        Ok(self)
    }

    /// Convenience constructor for the canonical K-major operand
    /// (`[M, K]` / `[N, K]` row-major buffer, `ld` = row pitch in elements).
    pub fn k_major(
        dtype: Dtype,
        rows: u32,
        k: u32,
        outer_stride: u32,
        sf: Option<SfTensor>,
        data: crate::device::DevBuffer,
    ) -> Operand {
        Operand {
            dtype,
            major: Major::K,
            rows,
            k,
            outer_stride,
            sf,
            data,
        }
    }

    /// Convenience constructor for the MN-major (column-major) operand view:
    /// backing buffer is a `[K, MN]` stack, `outer_stride` = pitch of the K
    /// slices (>= MN; in *logical* elements — half bytes for FP4).
    pub fn mn_major(
        dtype: Dtype,
        rows: u32,
        k: u32,
        outer_stride: u32,
        sf: Option<SfTensor>,
        data: crate::device::DevBuffer,
    ) -> Operand {
        Operand {
            dtype,
            major: Major::Mn,
            rows,
            k,
            outer_stride,
            sf,
            data,
        }
    }
}
