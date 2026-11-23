//! AVX2 deblocking filters: 8 lines (one per 32-bit lane) at a time. Plain 32-bit arithmetic, the
//! same expressions as `luma_line` / `chroma_line` in the parent module, with the branches turned
//! into masks (-1 / 0 lanes) and blends, so the result is bit-exact with the scalar code. The
//! Avx512 tier uses these kernels as well: a segment is 4 lines, so 16 lanes would only pay off
//! when four segments are always queued together.

use super::{CLines, Lines};
use crate::simd::x86::{ld_i32x8, st_i32x8};
use core::arch::x86_64::{
    __m256i, _mm256_abs_epi32 as abs, _mm256_add_epi32 as add, _mm256_and_si256 as and,
    _mm256_andnot_si256 as andn, _mm256_blendv_epi8 as blendv, _mm256_cmpgt_epi32 as gt,
    _mm256_max_epi32 as max, _mm256_min_epi32 as min, _mm256_set1_epi32 as set1,
    _mm256_slli_epi32 as shl, _mm256_srai_epi32 as sra, _mm256_sub_epi32 as sub,
};

/// `x` limited to `lo..=hi`.
#[inline]
#[target_feature(enable = "avx2")]
fn clamp(x: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    min(max(x, lo), hi)
}

/// `a` in the lanes where `m` is set, `b` elsewhere.
#[inline]
#[target_feature(enable = "avx2")]
fn sel(m: __m256i, a: __m256i, b: __m256i) -> __m256i {
    blendv(b, a, m)
}

/// 8.7.2.5.7 on 8 lines; see `luma_line`.
#[target_feature(enable = "avx2")]
pub(super) fn luma(l: &mut Lines) {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = &l.s;
    let (p3, p2, p1, p0) = (ld_i32x8(r0), ld_i32x8(r1), ld_i32x8(r2), ld_i32x8(r3));
    let (q0, q1, q2, q3) = (ld_i32x8(r4), ld_i32x8(r5), ld_i32x8(r6), ld_i32x8(r7));
    let tc = ld_i32x8(&l.tc);
    let (strong, dep, deq) = (ld_i32x8(&l.strong), ld_i32x8(&l.dep), ld_i32x8(&l.deq));
    let (no_p, no_q) = (ld_i32x8(&l.no_p), ld_i32x8(&l.no_q));
    let zero = set1(0);
    let maxv = set1(l.max);
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
    let h = sra::<1>(tc);
    let nh = sub(zero, h);
    let np0 = clamp(add(p0, delta), zero, maxv);
    let avg_p = sra::<1>(add(add(p2, p0), c1));
    let dp = clamp(sra::<1>(add(sub(avg_p, p1), delta)), nh, h);
    let np1 = clamp(add(p1, dp), zero, maxv);
    let nq0 = clamp(sub(q0, delta), zero, maxv);
    let avg_q = sra::<1>(add(add(q2, q0), c1));
    let dq = clamp(sra::<1>(sub(sub(avg_q, q1), delta)), nh, h);
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
    let [_, w1, w2, w3, w4, w5, w6, _] = &mut l.s;
    st_i32x8(w1, o1);
    st_i32x8(w2, o2);
    st_i32x8(w3, o3);
    st_i32x8(w4, o4);
    st_i32x8(w5, o5);
    st_i32x8(w6, o6);
}

/// 8.7.2.5.5 and 8.7.2.5.8 on 8 lines; see `chroma_line`.
#[target_feature(enable = "avx2")]
pub(super) fn chroma(l: &mut CLines) {
    let [r0, r1, r2, r3] = &l.s;
    let (p1, p0, q0, q1) = (ld_i32x8(r0), ld_i32x8(r1), ld_i32x8(r2), ld_i32x8(r3));
    let tc = ld_i32x8(&l.tc);
    let (no_p, no_q) = (ld_i32x8(&l.no_p), ld_i32x8(&l.no_q));
    let zero = set1(0);
    let maxv = set1(l.max);
    let d = sra::<3>(add(add(shl::<2>(sub(q0, p0)), sub(p1, q1)), set1(4)));
    let delta = clamp(d, sub(zero, tc), tc);
    let p0n = sel(no_p, p0, clamp(add(p0, delta), zero, maxv));
    let q0n = sel(no_q, q0, clamp(sub(q0, delta), zero, maxv));
    let [_, w1, w2, _] = &mut l.s;
    st_i32x8(w1, p0n);
    st_i32x8(w2, q0n);
}
