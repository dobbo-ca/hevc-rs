#![no_main]
use hevc::Options;
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

// decode_heic on arbitrary bytes: no panic, and an Ok image has a luma plane of its Info size.
fuzz_target!(|data: &[u8]| {
    if let Ok(h) = hevc::decode_heic(data, &Options::default(), &limits(), &NO) {
        let i = &h.image.info;
        let luma = &h.image.planes[0];
        assert_eq!((luma.width, luma.height), (i.width, i.height));
    }
});
