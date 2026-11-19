#![no_main]
use hevc::Options;
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

// The first byte is the split point between the hvcC configuration record and the samples.
fuzz_target!(|src: &[u8]| {
    let Some((&split, rest)) = src.split_first() else {
        return;
    };
    let (config, samples) = rest.split_at(usize::from(split).min(rest.len()));
    let l = limits();
    let _ = hevc::probe_hvcc(config, &l);
    let _ = hevc::decode_hvcc(config, samples, &Options::default(), &l, &NO);
});
