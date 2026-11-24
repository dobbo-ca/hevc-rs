//! The two resource peaks of a real call: heap bytes against `max_alloc` (a counting global
//! allocator) and threads against `max_threads` (Linux: `/proc/self/task`). One test in this file
//! on purpose, so no other test thread allocates or counts while it runs.

mod common;

use hevc::{Limits, Options};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// The wrapper only forwards to `System` and counts bytes; the test crate is not the library
// (which is `forbid(unsafe_code)`).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller's contract is passed through unchanged.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        // SAFETY: the caller's contract is passed through unchanged.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // SAFETY: the caller's contract is passed through unchanged.
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            let now = LIVE.fetch_add(new, Ordering::Relaxed) + new;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

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

/// A PPS with no tiles, no sync, default everything.
fn pps() -> Vec<u8> {
    let mut b = W::default();
    b.ue(0);
    b.ue(0);
    b.flag(false);
    b.flag(false);
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

/// Annex B stream: VPS, SPS, PPS and one IDR I slice of `size` x `size` samples whose slice data
/// is `data` pseudo-random bytes (deterministic xorshift, never zero). The data is not a valid
/// CABAC stream, but the call reaches `activate`, `Frame::new` and the CTU loop before it fails.
fn stream(size: u32, data: usize) -> Vec<u8> {
    stream_n(size, data, 1)
}

/// Like [`stream`] with `n` copies of the IDR picture.
fn stream_n(size: u32, data: usize, n: usize) -> Vec<u8> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut slice = vec![0xAC, 0xC0]; // first_slice, pps 0, I, no SAO, qp delta 0, alignment
    for _ in 0..data {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        slice.push((x >> 24) as u8 | 1);
    }
    let mut s = Vec::new();
    let mut nals = vec![(32u8, vec![0x0C, 0xFF]), (33, sps(size, size)), (34, pps())];
    nals.extend((0..n).map(|_| (19, slice.clone())));
    for (kind, p) in nals {
        s.extend_from_slice(&[0, 0, 0, 1, kind << 1, 1]);
        s.extend_from_slice(&escape(&p));
    }
    s
}

#[cfg(target_os = "linux")]
fn threads_now() -> usize {
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    // A thread that has exited (and was joined) can stay listed for a moment in state Z or X,
    // and pthread_join can return while the thread is still in the kernel's exit path (state R
    // or D, with PF_EXITING = 0x4 in the flags, the 6th field after the state); only live
    // threads count.
    dir.filter_map(Result::ok)
        .filter(|e| {
            std::fs::read_to_string(e.path().join("stat")).is_ok_and(|s| {
                let mut f = s.rsplit_once(')').map_or("", |(_, r)| r).split_whitespace();
                let state = f.next().unwrap_or("Z");
                let flags: u32 = f.nth(5).and_then(|v| v.parse().ok()).unwrap_or(0);
                state != "Z" && state != "X" && flags & 0x4 == 0
            })
        })
        .count()
}

/// Runs `f` and returns whether it failed with `AllocLimit` and the heap peak above the live
/// bytes at the start.
fn peaked(f: impl FnOnce() -> Result<(), hevc::Error>) -> (bool, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let r = f();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    (r == Err(hevc::Error::AllocLimit), peak)
}

#[test]
fn heap_peak_stays_under_max_alloc_and_no_thread_is_spawned() {
    let stream = stream(512, 4096);
    let cancel = AtomicBool::new(false);
    let opts = Options::default();
    let run = |limits: &Limits| {
        let a = peaked(|| hevc::decode(&stream, &opts, limits, &cancel).map(drop));
        let b = peaked(|| hevc::probe(&stream, limits).map(drop));
        let c = peaked(|| hevc::intra_prefix(&stream, limits).map(drop));
        [a, b, c]
    };

    // What a decode needs: the frame (at least the luma plane) was really allocated.
    let (limit, need) = run(&Limits::default())[0];
    assert!(!limit);
    assert!(
        need > 256 << 10,
        "the decode never reached the frame: peak {need}"
    );

    // Budgets from far under to far over: whenever a call gets past the limit, its real heap
    // peak is within the budget; the small ones must fail with AllocLimit.
    let small = 70_000;
    for max_alloc in [
        small,
        need / 2,
        need - 1,
        need,
        need + (64 << 10),
        4 * need,
        1 << 30,
    ] {
        for max_threads in [0, 1, 2, 8] {
            let limits = Limits {
                max_alloc,
                max_threads,
                ..Limits::default()
            };
            for (_, peak) in run(&limits) {
                assert!(
                    peak <= max_alloc,
                    "heap peak {peak} over max_alloc {max_alloc} (threads {max_threads})"
                );
            }
        }
    }
    let tight = Limits {
        max_alloc: small,
        ..Limits::default()
    };
    assert!(
        run(&tight)[0].0,
        "a 70 kB budget must not hold a 512x512 frame"
    );
    let none = Limits {
        max_alloc: 0,
        ..Limits::default()
    };
    assert!(run(&none).iter().all(|r| r.0));

    // The same over a stream whose pictures reach the workers (six pictures; each fails in the
    // CTU loop, so the call is replayed with one thread): the real heap of all threads together
    // stays within the budget.
    let many = stream_n(64, 2048, 6);
    let (_, many_need) =
        peaked(|| hevc::decode(&many, &opts, &Limits::default(), &cancel).map(drop));
    for max_alloc in [
        small,
        many_need / 2,
        many_need,
        many_need + (64 << 10),
        4 * many_need,
    ] {
        for max_threads in [2, 8] {
            let limits = Limits {
                max_alloc,
                max_threads,
                ..Limits::default()
            };
            let (_, peak) = peaked(|| hevc::decode(&many, &opts, &limits, &cancel).map(drop));
            assert!(
                peak <= max_alloc,
                "heap peak {peak} over max_alloc {max_alloc} (threads {max_threads}, six pictures)"
            );
        }
    }

    // Parameter sets replaced while pictures are in flight (`Decoder::stream`, the 32..=34 arm):
    // twelve real 256x256 pictures, each after its own VPS, SPS and PPS (the SPS with 64
    // short-term RPS and 32 long-term entries, about 25 KB with the PPS), decoded while the
    // budget steps by 8 KiB (32 KiB with one thread) from half of what the call needs. The real
    // heap of all threads stays within max_alloc. This cannot tell on its own whether the
    // replaced sets stay booked: only the oldest picture in flight can hold a released pair
    // (every later release waits in `Decoder::late`), and one pair fits in the 64 KiB small
    // reserve. `decoder::tests::a_replaced_set_stays_booked_while_the_picture_being_built_holds_it`
    // checks the booking itself.
    let repeated = common::stream(&common::Spec {
        width: 256,
        height: 256,
        pictures: 12,
        big_sps: true,
        repeat: true,
        ..Default::default()
    });
    let eight = Limits {
        max_threads: 8,
        ..Limits::default()
    };
    let decoded = hevc::decode(&repeated, &opts, &eight, &cancel).map(|p| p.len());
    assert_eq!(decoded, Ok(12), "the coded stream must decode");
    let (_, rep_need) = peaked(|| hevc::decode(&repeated, &opts, &eight, &cancel).map(drop));
    for (max_threads, step) in [(8, 8 << 10), (1, 32 << 10)] {
        let mut max_alloc = rep_need / 2;
        while max_alloc <= rep_need + (64 << 10) {
            let limits = Limits {
                max_alloc,
                max_threads,
                ..Limits::default()
            };
            let (_, peak) = peaked(|| hevc::decode(&repeated, &opts, &limits, &cancel).map(drop));
            assert!(
                peak <= max_alloc,
                "heap peak {peak} over max_alloc {max_alloc} (threads {max_threads}, replaced parameter sets)"
            );
            max_alloc += step;
        }
    }

    // decode_heic with `Options::apply_transforms`: a real 256x256 picture turned by 90 degrees
    // (new planes next to the decoded ones). The real heap of the call (the parse, the decode and
    // the buffers of the transform) stays within max_alloc while the budget steps by 16 KiB.
    let (hvcc, data) = common::hvc1(&common::stream(&common::Spec {
        width: 256,
        height: 256,
        pictures: 1,
        ..Default::default()
    }));
    let item = common::Item {
        id: 1,
        kind: *b"hvc1",
        data,
        props: vec![common::ispe(256, 256), hvcc, common::bx(b"irot", &[1])],
    };
    let heic = common::heif(&[item], &[]);
    let apply = Options {
        apply_transforms: true,
        ..opts
    };
    let ok = hevc::decode_heic(&heic, &apply, &Limits::default(), &cancel);
    assert!(
        ok.is_ok_and(|h| h.transforms_applied),
        "the HEIC file must decode"
    );
    let (_, heic_need) =
        peaked(|| hevc::decode_heic(&heic, &apply, &Limits::default(), &cancel).map(drop));
    let mut max_alloc = heic_need / 2;
    while max_alloc <= 2 * heic_need {
        let limits = Limits {
            max_alloc,
            ..Limits::default()
        };
        let (_, peak) = peaked(|| hevc::decode_heic(&heic, &apply, &limits, &cancel).map(drop));
        assert!(
            peak <= max_alloc,
            "heap peak {peak} over max_alloc {max_alloc} (HEIC with transforms)"
        );
        max_alloc += 16 << 10;
    }

    // Band-parallel filters on a real picture: one 512x512 picture (four bands of 64 Ki luma
    // samples; one picture, so no picture worker) gives the same output for every max_threads,
    // and with two or more a band thread runs next to the calling thread. The checks come after
    // the scope, so a failure cannot leave the sampler running.
    #[cfg(target_os = "linux")]
    {
        let big = common::stream(&common::Spec {
            width: 512,
            height: 512,
            pictures: 1,
            ..Default::default()
        });
        let one = Limits {
            max_threads: 1,
            ..Limits::default()
        };
        let want = hevc::decode(&big, &opts, &one, &cancel);
        assert!(
            want.as_ref().is_ok_and(|p| p.len() == 1),
            "the coded picture must decode: {:?}",
            want.as_ref().map(Vec::len)
        );
        for max_threads in [1usize, 2, 4, 8] {
            // The fewest threads over 20 ms: a thread joined just before (the last sampler) can
            // still be listed for a moment, and would raise the base.
            let base = (0..20)
                .map(|_| {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    threads_now()
                })
                .min()
                .unwrap_or(0);
            let done = AtomicBool::new(false);
            let seen = AtomicUsize::new(base);
            let mut same = true;
            std::thread::scope(|sc| {
                sc.spawn(|| {
                    while !done.load(Ordering::Relaxed) {
                        seen.fetch_max(threads_now(), Ordering::Relaxed);
                    }
                });
                let limits = Limits {
                    max_threads,
                    ..Limits::default()
                };
                for _ in 0..10 {
                    same &= hevc::decode(&big, &opts, &limits, &cancel) == want;
                }
                done.store(true, Ordering::Relaxed);
            });
            assert!(same, "output differs with max_threads {max_threads}");
            // The sampler is one thread more; any other is a band thread. No upper bound here:
            // a band thread joined at the end of one filter pass can still be listed while the
            // next pass starts its own (the bound of the picture workers is checked below).
            let seen = seen.load(Ordering::Relaxed);
            if max_threads == 1 {
                assert!(seen <= base + 1, "a thread ran with max_threads 1");
            } else {
                assert!(
                    seen >= base + 2,
                    "no band thread ran with max_threads {max_threads} (base {base}, seen {seen})"
                );
            }
        }
    }

    // Threads: the peak during decodes is the test's own, plus the sampler, plus at most
    // max_threads - 1 workers (none at max_threads 1). Six pictures, so the workers start.
    #[cfg(target_os = "linux")]
    {
        for max_threads in [1usize, 2, 4] {
            let many = stream_n(64, 2048, 6);
            let base = threads_now();
            let done = AtomicBool::new(false);
            let seen = AtomicUsize::new(base);
            std::thread::scope(|sc| {
                sc.spawn(|| {
                    while !done.load(Ordering::Relaxed) {
                        seen.fetch_max(threads_now(), Ordering::Relaxed);
                    }
                });
                let limits = Limits {
                    max_threads,
                    ..Limits::default()
                };
                for _ in 0..50 {
                    // A worker joined at the end of the previous call can still be listed for a
                    // moment while it exits; let it go before the next call starts its own.
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    let _ = hevc::decode(&many, &opts, &limits, &cancel);
                }
                done.store(true, Ordering::Relaxed);
            });
            assert!(
                seen.load(Ordering::Relaxed) <= base + max_threads,
                "threads over max_threads {max_threads}"
            );
        }
    }
}
