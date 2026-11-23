#![no_main]
use libfuzzer_sys::fuzz_target;

// Deblocking kernels: every SIMD tier of the host against the scalar one, on luma and chroma
// line batches built from the input bytes (all bit depths, any tc, any flag mix).
fuzz_target!(|data: &[u8]| {
    assert!(hevc::__fuzz_deblock::tiers(data));
});
