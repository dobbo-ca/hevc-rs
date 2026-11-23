//! Shared by the fuzz targets: small limits and a flag that is never set.
#![allow(dead_code)]

use hevc::Limits;
use std::sync::atomic::AtomicBool;

pub static NO: AtomicBool = AtomicBool::new(false);

/// Small, so a hostile stream cannot reach the fuzzer's malloc limit.
pub fn limits() -> Limits {
    Limits {
        max_alloc: 256 << 20,
        max_threads: 1,
        max_width: 4096,
        max_height: 4096,
        max_pixels: 1 << 22,
        max_pictures: 8,
        max_metadata: 1 << 16,
    }
}
