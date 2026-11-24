#![no_main]
use libfuzzer_sys::fuzz_target;

// The deblocking kernels (luma and chroma, vertical and horizontal edges, 4 segments of 4 lines)
// on every SIMD tier of this CPU against the scalar tier, on samples and parameters built from
// arbitrary bytes: any difference is a crash.
fuzz_target!(|data: &[u8]| {
    assert!(hevc::__fuzz_deblock::tiers(data));
});
