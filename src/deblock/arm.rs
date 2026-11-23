//! NEON deblocking filters: 4 lines (one per 32-bit lane) at a time, so each 8-line batch runs as
//! two halves. The same expressions as `luma_line` / `chroma_line` in the parent module, with the
//! branches turned into masks (-1 / 0 lanes) and bit selects: bit-exact with the scalar code.

use super::{CLines, Lines};
use crate::simd::arm::{ld_i32x4, st_i32x4};
use core::arch::aarch64::{
    int32x4_t, vabsq_s32 as abs, vaddq_s32 as add, vandq_s32 as and, vbicq_s32 as bic,
    vbslq_s32 as bsl, vcgtq_s32, vdupq_n_s32 as set1, vmaxq_s32 as max, vminq_s32 as min,
    vreinterpretq_s32_u32 as to_s, vreinterpretq_u32_s32 as to_u, vshlq_n_s32 as shl,
    vshrq_n_s32 as sra, vsubq_s32 as sub,
};

/// Lower (`hi` false) or upper half of the 8 lanes.
fn half(a: &[i32; 8], hi: bool) -> [i32; 4] {
    let h = if hi {
        a.last_chunk::<4>()
    } else {
        a.first_chunk::<4>()
    };
    h.copied().unwrap_or([0; 4])
}

fn put_half(a: &mut [i32; 8], hi: bool, v: [i32; 4]) {
    let h = if hi {
        a.last_chunk_mut::<4>()
    } else {
        a.first_chunk_mut::<4>()
    };
    if let Some(d) = h {
        *d = v;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn clamp(x: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    min(max(x, lo), hi)
}

/// `a` in the lanes where `m` is set, `b` elsewhere.
#[inline]
#[target_feature(enable = "neon")]
fn sel(m: int32x4_t, a: int32x4_t, b: int32x4_t) -> int32x4_t {
    bsl(to_u(m), a, b)
}

/// `!a & b`.
#[inline]
#[target_feature(enable = "neon")]
fn andn(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    bic(b, a)
}

/// Mask of the lanes where `a > b`.
#[inline]
#[target_feature(enable = "neon")]
fn gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    to_s(vcgtq_s32(a, b))
}

/// One half of a luma batch: samples, tc and the flags.
struct Half {
    s: [[i32; 4]; 8],
    tc: [i32; 4],
    strong: [i32; 4],
    dep: [i32; 4],
    deq: [i32; 4],
    no_p: [i32; 4],
    no_q: [i32; 4],
}

/// 8.7.2.5.7 on 8 lines; see `luma_line`.
#[target_feature(enable = "neon")]
pub(super) fn luma(l: &mut Lines) {
    for hi in [false, true] {
        let h = Half {
            s: l.s.each_ref().map(|a| half(a, hi)),
            tc: half(&l.tc, hi),
            strong: half(&l.strong, hi),
            dep: half(&l.dep, hi),
            deq: half(&l.deq, hi),
            no_p: half(&l.no_p, hi),
            no_q: half(&l.no_q, hi),
        };
        let o = luma4(&h, l.max);
        for (row, v) in l.s.iter_mut().zip(o) {
            put_half(row, hi, v);
        }
    }
}

#[target_feature(enable = "neon")]
fn luma4(h: &Half, maxs: i32) -> [[i32; 4]; 8] {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = &h.s;
    let (p3, p2, p1, p0) = (ld_i32x4(r0), ld_i32x4(r1), ld_i32x4(r2), ld_i32x4(r3));
    let (q0, q1, q2, q3) = (ld_i32x4(r4), ld_i32x4(r5), ld_i32x4(r6), ld_i32x4(r7));
    let tc = ld_i32x4(&h.tc);
    let (strong, dep, deq) = (ld_i32x4(&h.strong), ld_i32x4(&h.dep), ld_i32x4(&h.deq));
    let (no_p, no_q) = (ld_i32x4(&h.no_p), ld_i32x4(&h.no_q));
    let zero = set1(0);
    let maxv = set1(maxs);
    let (c1, c2, c4) = (set1(1), set1(2), set1(4));

    // Strong filter (dE == 2), clipped to +-2 tc.
    let t = add(tc, tc);
    let sp = add(add(p1, p0), q0);
    let sp0 = add(add(add(p2, add(sp, sp)), q1), c4);
    let sp1 = add(add(p2, sp), c2);
    let sp2 = add(add(add(p3, p3), add(add(p2, p2), p2)), add(sp, c4));
    let sq = add(add(p0, q0), q1);
    let sq0 = add(add(add(p1, add(sq, sq)), q2), c4);
    let sq1 = add(add(sq, q2), c2);
    let sq2 = add(add(sq, add(add(q2, q2), q2)), add(add(q3, q3), c4));
    let sp0 = clamp(sra::<3>(sp0), sub(p0, t), add(p0, t));
    let sp1 = clamp(sra::<2>(sp1), sub(p1, t), add(p1, t));
    let sp2 = clamp(sra::<3>(sp2), sub(p2, t), add(p2, t));
    let sq0 = clamp(sra::<3>(sq0), sub(q0, t), add(q0, t));
    let sq1 = clamp(sra::<2>(sq1), sub(q1, t), add(q1, t));
    let sq2 = clamp(sra::<3>(sq2), sub(q2, t), add(q2, t));

    // Normal filter. delta = (9 (q0 - p0) - 3 (q1 - p1) + 8) >> 4.
    let a = sub(q0, p0);
    let b = sub(q1, p1);
    let nine_a = add(shl::<3>(a), a);
    let three_b = add(shl::<1>(b), b);
    let delta = sra::<4>(add(sub(nine_a, three_b), set1(8)));
    let tc10 = add(shl::<3>(tc), shl::<1>(tc));
    let apply = gt(tc10, abs(delta)); // |delta| < 10 tc
    let delta = clamp(delta, sub(zero, tc), tc);
    let hh = sra::<1>(tc);
    let nh = sub(zero, hh);
    let np0 = clamp(add(p0, delta), zero, maxv);
    let avg_p = sra::<1>(add(add(p2, p0), c1));
    let dp = clamp(sra::<1>(add(sub(avg_p, p1), delta)), nh, hh);
    let np1 = clamp(add(p1, dp), zero, maxv);
    let nq0 = clamp(sub(q0, delta), zero, maxv);
    let avg_q = sra::<1>(add(add(q2, q0), c1));
    let dq = clamp(sra::<1>(sub(sub(avg_q, q1), delta)), nh, hh);
    let nq1 = clamp(add(q1, dq), zero, maxv);

    // Which lanes change which samples.
    let weak = andn(strong, apply);
    let strong_p = andn(no_p, strong);
    let weak_p = andn(no_p, weak);
    let weak_p1 = and(weak_p, dep);
    let strong_q = andn(no_q, strong);
    let weak_q = andn(no_q, weak);
    let weak_q1 = and(weak_q, deq);
    let o1 = sel(strong_p, sp2, p2);
    let o2 = sel(weak_p1, np1, sel(strong_p, sp1, p1));
    let o3 = sel(weak_p, np0, sel(strong_p, sp0, p0));
    let o4 = sel(weak_q, nq0, sel(strong_q, sq0, q0));
    let o5 = sel(weak_q1, nq1, sel(strong_q, sq1, q1));
    let o6 = sel(strong_q, sq2, q2);
    let mut o = h.s;
    let [_, w1, w2, w3, w4, w5, w6, _] = &mut o;
    st_i32x4(w1, o1);
    st_i32x4(w2, o2);
    st_i32x4(w3, o3);
    st_i32x4(w4, o4);
    st_i32x4(w5, o5);
    st_i32x4(w6, o6);
    o
}

/// 8.7.2.5.5 and 8.7.2.5.8 on 8 lines; see `chroma_line`.
#[target_feature(enable = "neon")]
pub(super) fn chroma(l: &mut CLines) {
    for hi in [false, true] {
        let s = l.s.each_ref().map(|a| half(a, hi));
        let o = chroma4(
            &s,
            [half(&l.tc, hi), half(&l.no_p, hi), half(&l.no_q, hi)],
            l.max,
        );
        for (row, v) in l.s.iter_mut().zip(o) {
            put_half(row, hi, v);
        }
    }
}

/// `flags` is `[tc, no_p, no_q]`.
#[target_feature(enable = "neon")]
fn chroma4(s: &[[i32; 4]; 4], flags: [[i32; 4]; 3], maxs: i32) -> [[i32; 4]; 4] {
    let [r0, r1, r2, r3] = s;
    let (p1, p0, q0, q1) = (ld_i32x4(r0), ld_i32x4(r1), ld_i32x4(r2), ld_i32x4(r3));
    let [f0, f1, f2] = &flags;
    let (tc, no_p, no_q) = (ld_i32x4(f0), ld_i32x4(f1), ld_i32x4(f2));
    let zero = set1(0);
    let maxv = set1(maxs);
    let d = sra::<3>(add(add(shl::<2>(sub(q0, p0)), sub(p1, q1)), set1(4)));
    let delta = clamp(d, sub(zero, tc), tc);
    let p0n = sel(no_p, p0, clamp(add(p0, delta), zero, maxv));
    let q0n = sel(no_q, q0, clamp(sub(q0, delta), zero, maxv));
    let mut o = *s;
    let [_, w1, w2, _] = &mut o;
    st_i32x4(w1, p0n);
    st_i32x4(w2, q0n);
    o
}
