#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::limits;

// The data become the raw bytes of one parameter set NAL unit (a start code and the two-byte NAL
// header in front), so the SPS and PPS parsers see them without a stream around them.
// 0x42 0x01: SPS (type 33), 0x44 0x01: PPS (type 34).
fuzz_target!(|data: &[u8]| {
    let l = limits();
    for header in [[0x42u8, 0x01], [0x44, 0x01]] {
        let mut stream = vec![0, 0, 0, 1, header[0], header[1]];
        stream.extend_from_slice(data);
        let _ = hevc::probe(&stream, &l);
    }
});
