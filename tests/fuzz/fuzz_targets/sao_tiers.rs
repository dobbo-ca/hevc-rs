#![no_main]
use libfuzzer_sys::fuzz_target;

// The SAO row kernels (band and edge offset) on every SIMD tier of this CPU against the scalar
// tier, on rows built from arbitrary bytes: any difference is a crash.
fuzz_target!(|data: &[u8]| {
    assert!(hevc::__fuzz_sao::tiers(data));
});
