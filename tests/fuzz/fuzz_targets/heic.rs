#![no_main]
use hevc::{Limits, Options};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

// decode_heic on arbitrary bytes: no panic, an Ok image has a luma plane of its Info size, and
// the result (image or error) is the same with 1 and with 4 threads (grid tiles, overlays). The
// same with `apply_transforms` (crop, rotation and mirror through isobmff-rs). Then
// decode_heic_item, with the same checks, on the first 8 items of the parse whatever their kind
// (tiles, alpha, `iden`, `iovl`, non-image items), so the transforms of a non-primary item and
// the budget of that entry point are fuzzed too.
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
    let iso = isobmff::Limits {
        max_alloc: 32 << 20,
        max_threads: 1,
        ..isobmff::Limits::default()
    };
    let Ok(heif) = isobmff::parse(data, &iso, &NO) else {
        return;
    };
    for id in heif.items.iter().take(8).map(|it| it.id) {
        for opts in [Options::default(), apply] {
            let one = hevc::decode_heic_item(&heif, data, id, &opts, &limits(), &NO);
            let par = hevc::decode_heic_item(&heif, data, id, &opts, &four, &NO);
            assert_eq!(one, par);
            if let Ok(p) = one {
                let luma = &p.planes[0];
                assert_eq!((luma.width, luma.height), (p.info.width, p.info.height));
            }
        }
    }
});
