#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::limits;

// The Annex B splitter, the parameter set store and the slice header parser (`intra_prefix`) on
// arbitrary bytes: errors are fine, panics are not. A prefix that comes back is a stream of its
// own: it must not grow past the input by more than one byte per NAL unit (see `cap` there).
fuzz_target!(|data: &[u8]| {
    if let Ok(prefix) = hevc::intra_prefix(data, &limits()) {
        assert!(prefix.len() <= data.len() + data.len() / 4 + 8);
    }
});
