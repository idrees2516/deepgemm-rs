//! GLM-5.3-class (hyperconnection) attention projections.
//!
//! ```text
//! cargo run -p deepgemm --example glm53_attention --features e2e --release
//! ```
//!
//! The three einsum equations of the hyperconnection attention block, BF16
//! and FP8 flavors, plus the hc-prenorm and MHC megakernel entries.

fn main() {
    println!("This example runs on a B200/B300 GPU (feature `e2e`).");
    println!("See deepgemm/src/api_einsum.rs + api_serving.rs:");
    println!("  einsum_bhr_hdr_bhd / fp8_einsum_bhr_hdr_bhd  — out-proj");
    println!("  einsum_bhd_hdr_bhr / fp8_einsum_bhd_hdr_bhr  — v-proj");
    println!("  einsum_bhd_bhr_hdr (cuBLASLt)                — wgrad reduce");
    println!("  hc_prenorm_sm100 / mega_mhc                  — hyperconnection");
}
