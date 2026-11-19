#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::limits;

// probe must keep the limits it was given: an Ok Info never exceeds them.
fuzz_target!(|data: &[u8]| {
    let l = limits();
    if let Ok(info) = hevc::probe(data, &l) {
        assert!(info.coded_width <= l.max_width && info.coded_height <= l.max_height);
        assert!(u64::from(info.coded_width) * u64::from(info.coded_height) <= l.max_pixels);
        assert!(info.width <= info.coded_width && info.height <= info.coded_height);
    }
});
