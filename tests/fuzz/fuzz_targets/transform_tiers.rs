#![no_main]
use libfuzzer_sys::fuzz_target;

// Dequantisation, inverse transform and residual add: every SIMD tier equals the scalar one.
fuzz_target!(|data: &[u8]| {
    assert!(hevc::__fuzz_transform::tiers_agree(data));
});
