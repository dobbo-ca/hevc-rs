//! `bench [--threads N] <stream>`: decodes one Annex B file once (`bench --tier` prints the SIMD
//! tier and exits). Exit status 0 when it yields a
//! picture (`InterSlice` after a picture is not a failure), 1 otherwise, 2 on bad usage. The
//! timing is done by `tests/parity/bench.sh` around the process.

use hevc::{Error, Limits, Options};
use std::sync::atomic::AtomicBool;

fn main() {
    let mut limits = Limits::default();
    let mut file = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--tier" {
            println!("{}", hevc::simd_tier());
            return;
        } else if a == "--threads" {
            match args.next().and_then(|v| v.parse().ok()) {
                Some(n) => limits.max_threads = n,
                None => usage(),
            }
        } else {
            file = Some(a);
        }
    }
    let Some(file) = file else { usage() };
    let data = std::fs::read(&file).unwrap();
    let mut out = Vec::new();
    let r = hevc::decode_into(
        &data,
        &Options::default(),
        &limits,
        &AtomicBool::new(false),
        &mut out,
    );
    let ok = matches!(r, Ok(()) | Err(Error::InterSlice)) && !out.is_empty();
    std::process::exit(i32::from(!ok));
}

fn usage() -> ! {
    eprintln!("usage: bench [--threads N] <stream>");
    std::process::exit(2);
}
