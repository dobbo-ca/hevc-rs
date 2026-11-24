#![no_main]
use hevc::{ChromaFormat, Options, Samples};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

// Whatever the stream is, every picture that comes back has the planes its Info says. The low
// 4 bits of the last byte pick deblocking, SAO, metadata and the hash check; the stream is the whole
// input. The result and the pictures must not depend on max_threads (3 against 1).
fuzz_target!(|data: &[u8]| {
    let bits = data.last().copied().unwrap_or(0);
    let opts = Options {
        deblocking: bits & 1 == 0,
        sao: bits & 2 == 0,
        metadata: bits & 4 == 0,
        check_hash: bits & 8 == 0,
        apply_transforms: false,
    };
    let mut pictures = Vec::new();
    let one = hevc::decode_into(data, &opts, &limits(), &NO, &mut pictures);
    let three = hevc::Limits {
        max_threads: 3,
        ..limits()
    };
    let mut other = Vec::new();
    assert_eq!(one, hevc::decode_into(data, &opts, &three, &NO, &mut other));
    assert_eq!(pictures, other);
    if one.is_err() {
        return;
    }
    for pic in &pictures {
        let i = &pic.info;
        let (w, h) = (i.width as usize, i.height as usize);
        let chroma = match i.chroma_format {
            ChromaFormat::Mono => 0,
            ChromaFormat::Yuv420 => 2,
        };
        assert_eq!(pic.planes.len(), if chroma == 0 { 1 } else { 3 });
        for (c, plane) in pic.planes.iter().enumerate() {
            let (pw, ph) = if c == 0 {
                (w, h)
            } else {
                (w / chroma, h / chroma)
            };
            let depth = if c == 0 {
                i.bit_depth_luma
            } else {
                i.bit_depth_chroma
            };
            assert_eq!((plane.width as usize, plane.height as usize), (pw, ph));
            let len = match &plane.samples {
                Samples::U8(v) => {
                    assert!(depth == 8);
                    v.len()
                }
                Samples::U16(v) => {
                    assert!(depth > 8);
                    v.len()
                }
            };
            assert_eq!(len, pw * ph);
        }
    }
});
