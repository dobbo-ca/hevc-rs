#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::limits;

// The Annex B splitter and the SPS parser on arbitrary bytes: errors are fine, panics are not.
fuzz_target!(|data: &[u8]| {
    let _ = hevc::probe(data, &limits());
});
