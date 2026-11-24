#![no_main]
use hevc::{Limits, Options};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

// decode_heic on arbitrary bytes: no panic, an Ok image has a luma plane of its Info size, and
// the result (image or error) is the same with 1 and with 4 threads (grid tiles, overlays). The
// same with `apply_transforms` (crop, rotation and mirror through isobmff-rs).
fuzz_target!(|data: &[u8]| {
    let four = Limits {
        max_threads: 4,
        ..limits()
    };
    let apply = Options {
        apply_transforms: true,
        ..Options::default()
    };
    for opts in [Options::default(), apply] {
        let one = hevc::decode_heic(data, &opts, &limits(), &NO);
        let par = hevc::decode_heic(data, &opts, &four, &NO);
        assert_eq!(one, par);
        if let Ok(h) = one {
            let i = &h.image.info;
            let luma = &h.image.planes[0];
            assert_eq!((luma.width, luma.height), (i.width, i.height));
        }
    }
});
