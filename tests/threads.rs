//! Pictures of one call decoded in parallel: the result and the output pictures do not depend on
//! `max_threads`. The slice data of the test streams is pseudo-random, not valid CABAC, so every
//! picture fails inside the CTU loop (a job error: the call is replayed with one thread); a P
//! slice at the end is a front-end error. The streams reach the workers, the replay and the
//! ticket order; the success path of the workers is covered by the parity gate G5 and the fuzz
//! target.

use hevc::{Error, Limits, Options, Picture};
use std::sync::atomic::AtomicBool;

/// Bit writer for the hand-built parameter sets (u(n), ue(v), se(v)), as in `ps::tests`.
#[derive(Default)]
struct W(Vec<bool>);

impl W {
    fn u(&mut self, n: u32, v: u64) {
        for i in (0..n).rev() {
            self.0.push((v >> i) & 1 == 1);
        }
    }
    fn flag(&mut self, v: bool) {
        self.0.push(v);
    }
    fn ue(&mut self, v: u32) {
        let x = u64::from(v) + 1;
        let len = 64 - x.leading_zeros();
        self.u(len - 1, 0);
        self.u(len, x);
    }
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }
    /// The bits, then a stop bit (so the last byte is never zero), zero padded to a byte.
    fn rbsp(mut self) -> Vec<u8> {
        self.flag(true);
        self.0
            .chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u8, |b, (i, &x)| if x { b | (0x80 >> i) } else { b })
            })
            .collect()
    }
}

/// A Main profile SPS (CTB 32, 4:2:0 8 bit, SAO on, no VUI) of `w` x `h` luma samples.
fn sps(w: u32, h: u32) -> Vec<u8> {
    let mut b = W::default();
    b.u(4, 0);
    b.u(3, 0);
    b.flag(true);
    b.u(2, 0);
    b.flag(false);
    b.u(5, 1);
    b.u(32, 0x6000_0000);
    b.u(32, 0);
    b.u(16, 0);
    b.u(8, 93);
    for v in [0, 1, w, h] {
        b.ue(v); // sps id, 4:2:0, size
    }
    b.flag(true); // conformance window
    for v in [0, 1, 0, 1, 0, 0, 4] {
        b.ue(v);
    }
    b.flag(true);
    for v in [1, 0, 0, 0, 2, 0, 3, 1, 1] {
        b.ue(v);
    }
    for f in [false, true, true, false] {
        b.flag(f); // scaling lists, amp, sao, pcm
    }
    b.ue(0); // no st rps
    for f in [false, false, true, false, false] {
        b.flag(f); // long term, tmvp, strong intra smoothing, vui, extension
    }
    b.rbsp()
}

/// A PPS with no tiles, no sync, default everything (`output_flag_present_flag` as given).
fn pps(output_flag_present: bool) -> Vec<u8> {
    let mut b = W::default();
    b.ue(0);
    b.ue(0);
    b.flag(false);
    b.flag(output_flag_present);
    b.u(3, 0);
    b.flag(true); // sign data hiding
    b.flag(false);
    b.ue(0);
    b.ue(0);
    b.se(-2); // init_qp 24
    for _ in 0..3 {
        b.flag(false);
    }
    b.se(1);
    b.se(-1);
    for _ in 0..6 {
        b.flag(false);
    }
    for _ in 0..3 {
        b.flag(false); // loop filter across slices, deblocking control, scaling list
    }
    b.flag(false);
    b.ue(0);
    b.flag(false);
    b.flag(false); // extension
    b.rbsp()
}

/// Escapes 00 00 0x sequences (7.4.2) so the payload holds no start code.
fn escape(p: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(p.len() + 8);
    let mut zeros = 0;
    for &x in p {
        if zeros >= 2 && x <= 3 {
            o.push(3);
            zeros = 0;
        }
        o.push(x);
        zeros = if x == 0 { zeros + 1 } else { 0 };
    }
    o
}

/// Annex B stream: VPS, SPS, PPS, then `n` IDR I slices of `size` x `size` samples (each with
/// `data` pseudo-random slice bytes of its own), then, with `inter`, a TRAIL_R slice that starts
/// a picture and is a P slice.
fn stream_n(size: u32, data: usize, n: usize, inter: bool) -> Vec<u8> {
    stream_with(size, data, n, inter, false)
}

/// As `stream_n`; with `hidden` the PPS has output_flag_present_flag and every picture has
/// pic_output_flag 0 (never output, so in no output queue).
fn stream_with(size: u32, data: usize, n: usize, inter: bool, hidden: bool) -> Vec<u8> {
    let mut nals = vec![
        (32u8, vec![0x0C, 0xFF]),
        (33, sps(size, size)),
        (34, pps(hidden)),
    ];
    for i in 0..n {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ ((i as u64 + 1) << 32);
        // first_slice, pps 0, I, [pic_output_flag 0,] no SAO, qp delta 0, alignment
        let mut slice = if hidden {
            vec![0xAC, 0x60]
        } else {
            vec![0xAC, 0xC0]
        };
        for _ in 0..data {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            slice.push((x >> 24) as u8 | 1);
        }
        nals.push((19, slice));
    }
    if inter {
        nals.push((1, vec![0xD0, 0x40, 0x01]));
    }
    let mut s = Vec::new();
    for (kind, p) in nals {
        s.extend_from_slice(&[0, 0, 0, 1, kind << 1, 1]);
        s.extend_from_slice(&escape(&p));
    }
    s
}

type Outcome = (Result<(), Error>, Vec<Picture>);

fn run(s: &[u8], opts: &Options, limits: &Limits, cancel: &AtomicBool) -> Outcome {
    let mut out = Vec::new();
    let r = hevc::decode_into(s, opts, limits, cancel, &mut out);
    (r, out)
}

fn with_threads(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        ..Limits::default()
    }
}

const STREAMS: [(u32, usize, usize, bool); 6] = [
    (32, 64, 1, false),
    (32, 200, 2, false),
    (64, 400, 5, false),
    (64, 1500, 8, true),
    (96, 3000, 3, true),
    (32, 5, 4, true),
];

#[test]
fn same_result_for_every_thread_count() {
    let cancel = AtomicBool::new(false);
    let all = Options::default();
    let checked = Options {
        check_hash: true,
        ..all
    };
    let bare = Options {
        deblocking: false,
        sao: false,
        metadata: false,
        ..all
    };
    for (size, data, n, inter) in STREAMS {
        let s = stream_n(size, data, n, inter);
        for opts in [all, checked, bare] {
            let want = run(&s, &opts, &with_threads(1), &cancel);
            for t in [0, 2, 3, 8] {
                let got = run(&s, &opts, &with_threads(t), &cancel);
                assert_eq!(got, want, "threads {t}, stream {size}/{data}/{n}/{inter}");
            }
        }
    }
}

#[test]
fn same_result_with_a_picture_limit() {
    let cancel = AtomicBool::new(false);
    let opts = Options::default();
    let s = stream_n(64, 400, 5, true);
    for max_pictures in 0..=6 {
        let one = Limits {
            max_pictures,
            ..with_threads(1)
        };
        let want = run(&s, &opts, &one, &cancel);
        for t in [2, 3, 8] {
            let limits = Limits {
                max_threads: t,
                ..one
            };
            assert_eq!(
                run(&s, &opts, &limits, &cancel),
                want,
                "threads {t}, max_pictures {max_pictures}"
            );
        }
    }
}

#[test]
fn same_result_across_the_memory_budget() {
    let cancel = AtomicBool::new(false);
    let opts = Options::default();
    let s = stream_n(128, 600, 4, true);
    // 64 KiB is booked up front; one 128 x 128 frame and its tables are a few tens of KiB.
    let mut max_alloc = 60_000;
    while max_alloc < 700_000 {
        let one = Limits {
            max_alloc,
            ..with_threads(1)
        };
        let want = run(&s, &opts, &one, &cancel);
        for t in [2, 3, 8] {
            let limits = Limits {
                max_threads: t,
                ..one
            };
            assert_eq!(
                run(&s, &opts, &limits, &cancel),
                want,
                "threads {t}, max_alloc {max_alloc}"
            );
        }
        max_alloc += 6_113;
    }
}

#[test]
fn cancel_gives_cancelled_and_no_pictures() {
    let cancel = AtomicBool::new(true);
    let opts = Options::default();
    let s = stream_n(64, 400, 5, false);
    for t in [1, 2, 3, 8] {
        let (r, out) = run(&s, &opts, &with_threads(t), &cancel);
        assert_eq!(r, Err(Error::Cancelled), "threads {t}");
        assert!(out.is_empty());
    }
}

#[test]
fn errors_of_pictures_that_are_not_output_count_for_every_thread_count() {
    let cancel = AtomicBool::new(false);
    let opts = Options::default();
    for (data, n) in [(64, 1), (400, 3)] {
        let s = stream_with(64, data, n, false, true);
        let want = run(&s, &opts, &with_threads(1), &cancel);
        assert!(want.0.is_err(), "hidden pictures {n}: {:?}", want.0);
        for t in [2, 3, 8] {
            assert_eq!(
                run(&s, &opts, &with_threads(t), &cancel),
                want,
                "threads {t}, pictures {n}"
            );
        }
    }
}
