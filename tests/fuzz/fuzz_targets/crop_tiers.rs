#![no_main]
use libfuzzer_sys::fuzz_target;

// Every SIMD tier of the output crop gives the bytes the scalar tier gives.
fuzz_target!(|data: &[u8]| {
    assert!(hevc::__fuzz_crop::tiers_match(data));
});
