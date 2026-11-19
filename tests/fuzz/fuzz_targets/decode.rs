#![no_main]
use hevc::{ChromaFormat, Options, Samples};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

// Whatever the stream is, every picture that comes back has the planes its Info says.
fuzz_target!(|data: &[u8]| {
    let Ok(pictures) = hevc::decode(data, &Options::default(), &limits(), &NO) else {
        return;
    };
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
