#![no_main]
use libfuzzer_sys::fuzz_target;

// Intra prediction on every SIMD tier of this CPU against the scalar tier: a block of reference
// samples, size, mode and bit depth built from the input must give the same samples.
fuzz_target!(|data: &[u8]| {
    assert!(hevc::__fuzz_intra::run(data));
});
