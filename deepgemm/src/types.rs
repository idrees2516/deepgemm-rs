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
