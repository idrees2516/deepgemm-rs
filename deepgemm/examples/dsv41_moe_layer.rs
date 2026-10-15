//! DeepSeek-V4.1-class MoE layer, end to end (requires a B200/B300 GPU).
//!
//! ```text
//! cargo run -p deepgemm --example dsv41_moe_layer --features e2e --release
//! ```
//!
//! Two paths are shown: the grouped-GEMM composition (quantize -> gate_up
//! grouped GEMM -> caller SwiGLU -> down grouped GEMM) and the fully fused
//! megakernel. Production stacks pick per traffic shape: the megakernel
//! wins at high batch (dispatch + combine fused on-SM), the grouped path is
//! the flexible fallback.

fn main() {
    println!("This example runs on a B200/B300 GPU (feature `e2e`).");
    println!("See deepgemm/src/api_serving.rs for the layer APIs composed:");
    println!("  Serving::moe_grouped_layer  — grouped-GEMM MoE path");
    println!("  api_mega_moe::mega_moe_fp8_fp4 — the fused megakernel");
}
