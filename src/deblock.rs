//! Deblocking filter (8.7.2) for a whole picture, 4:0:0 or 4:2:0.
//!
//! The edge strengths (`Frame::edge_v` / `edge_h`, 0 or 2) already carry the filterEdgeFlag and
//! slice_deblocking_filter_disabled_flag decisions made while the CUs were parsed. Here: luma
//! edges on the 8x8 grid, chroma edges where the chroma sample position is a multiple of 8 and
//! bS is 2; all vertical edges of the picture first, then the horizontal edges on that result.
//! Samples of PCM CUs with pcm_loop_filter_disabled_flag and of cu_transquant_bypass CUs are kept
//! (`Frame::no_filter`).
//!
//! The edges are filtered in batches of 4 segments of 4 lines (16 lines along one edge): first
//! the parameters of the 4 segments ([`Seg`]: beta, tC, which side is kept), then the samples.
//! The scalar code ([`luma_segment`], [`chroma_segment`]) is the oracle. The SIMD kernels
//! (bit depth up to 10, so every intermediate fits a 16-bit lane) filter the 16 lines at once:
//! the 8 rows across a horizontal edge are loaded as they are, the 16 rows of 8 samples across a
//! vertical edge are transposed into 8 vectors of 16 lines. The decisions of a segment (dE, dEp,
//! dEq, strong or normal) are made in the vectors from its lines 0 and 3 and applied with masks.

use crate::error::{Error, Result};
use crate::frame::{Frame, band_rows, par_bands, plane_cuts};
use crate::ps::Active;
use crate::simd::{self, Tier};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

/// beta' of Table 8-12, indexed by Q = 0..=51.
const BETA_TAB: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// tC' of Table 8-12, indexed by Q = 0..=53.
const TC_TAB: [u8; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// QpC of Table 8-10 for qPi = 30..=43.
const QPC_TAB: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];

/// Table 8-10 (ChromaArrayType 1).
fn qpc(qpi: i32) -> i32 {
    if qpi < 30 {
        qpi
    } else if qpi > 43 {
        qpi - 6
    } else {
        usize::try_from(qpi - 30)
            .ok()
            .and_then(|i| QPC_TAB.get(i))
            .copied()
            .unwrap_or(qpi)
    }
}

/// Table 8-12 lookup with the Clip3 of (8-348) / (8-350) / (8-383) folded in.
fn lookup(t: &[u8], q: i32) -> i32 {
    let last = i32::try_from(t.len()).unwrap_or(1) - 1;
    usize::try_from(q.clamp(0, last))
        .ok()
        .and_then(|i| t.get(i))
        .map_or(0, |&v| i32::from(v))
}

/// Filter parameters of one edge segment of 4 lines, scaled to the bit depth. `tc` 0 means
/// nothing to filter (with tC 0 neither filter changes a sample), so the default is "no edge".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Seg {
    /// beta (luma only).
    beta: i32,
    /// tC.
    tc: i32,
    /// The p side is kept (pcm_loop_filter_disabled or cu_transquant_bypass).
    no_p: bool,
    /// The q side is kept.
    no_q: bool,
}

/// [`Seg`] of the luma (`chroma` None) or chroma (`Some(cQpPicOffset)`) edge segment whose
/// first q sample covers luma (x, y), x and y multiples of 8; `scale` is `1 << (BitDepth - 8)`.
/// The default when nothing is filtered: bS 0 (or not 2 for chroma), CTB of q not decoded, tC
/// or beta 0, or both sides kept.
fn seg(
    meta: &Frame,
    act: &Active,
    (x, y): (usize, usize),
    vertical: bool,
    chroma: Option<i32>,
    scale: i32,
) -> Seg {
    seg_opt(meta, act, (x, y), vertical, chroma, scale).unwrap_or_default()
}

fn seg_opt(
    meta: &Frame,
    act: &Active,
    (x, y): (usize, usize),
    vertical: bool,
    chroma: Option<i32>,
    scale: i32,
) -> Option<Seg> {
    let gw = meta.grid_w;
    let q = (y >> 2).checked_mul(gw)?.checked_add(x >> 2)?;
    let (grid, p) = if vertical {
        (&meta.edge_v, q.checked_sub(1)?)
    } else {
        (&meta.edge_h, q.checked_sub(gw)?)
    };
    let bs = i32::from(*grid.get(q)?);
    if bs == 0 || (chroma.is_some() && bs != 2) {
        return None;
    }
    let ctb = Frame::ctb_addr(act, i32::try_from(x).ok()?, i32::try_from(y).ok()?)?;
    let s = meta.ctb_slice(ctb)?;
    // qPL (8-347) before the chroma offset.
    let qp = (i32::from(*meta.qp_y.get(q)?) + i32::from(*meta.qp_y.get(p)?) + 1) >> 1;
    let no_p = *meta.no_filter.get(p)? != 0;
    let no_q = *meta.no_filter.get(q)? != 0;
    let (beta, tc) = match chroma {
        None => (
            lookup(&BETA_TAB, qp + (s.beta_offset_div2 << 1)) * scale,
            lookup(&TC_TAB, qp + 2 * (bs - 1) + (s.tc_offset_div2 << 1)) * scale,
        ),
        Some(off) => (
            0,
            lookup(&TC_TAB, qpc(qp + off) + 2 + (s.tc_offset_div2 << 1)) * scale,
        ),
    };
    if tc == 0 || (chroma.is_none() && beta == 0) || (no_p && no_q) {
        return None;
    }
    Some(Seg {
        beta,
        tc,
        no_p,
        no_q,
    })
}

/// The `N` samples of one line across an edge: sample `i` is `plane[at + (i - N / 2) * is]`
/// (`at` is q0). None when the line leaves `plane`.
fn read<const N: usize>(plane: &[u16], at: usize, is: usize) -> Option<[i32; N]> {
    let start = at.checked_sub(N / 2 * is)?;
    let s = plane.get(start..=start.checked_add((N - 1) * is)?)?;
    let mut a = [0i32; N];
    for (d, &v) in a.iter_mut().zip(s.iter().step_by(is.max(1))) {
        *d = i32::from(v);
    }
    Some(a)
}

/// Writes back what [`read`] read (values are already clipped to the sample range).
fn write<const N: usize>(plane: &mut [u16], at: usize, is: usize, a: &[i32; N]) {
    let Some(start) = at.checked_sub(N / 2 * is) else {
        return;
    };
    let Some(end) = start.checked_add((N - 1) * is) else {
        return;
    };
    if let Some(s) = plane.get_mut(start..=end) {
        for (d, &v) in s.iter_mut().step_by(is.max(1)).zip(a) {
            *d = u16::try_from(v).unwrap_or(0);
        }
    }
}

/// 8.7.2.5.6: dSam.
fn d_sam(dpq: i32, p0: i32, p3: i32, q0: i32, q3: i32, beta: i32, tc: i32) -> bool {
    dpq < (beta >> 2)
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
        && (p0 - q0).abs() < ((5 * tc + 1) >> 1)
}

/// 8.7.2.5.7 for one line `[p3, p2, p1, p0, q0, q1, q2, q3]`; `strong` is dE == 2.
#[allow(clippy::too_many_arguments)]
fn luma_line(
    a: [i32; 8],
    strong: bool,
    dep: bool,
    deq: bool,
    tc: i32,
    max: i32,
    no_p: bool,
    no_q: bool,
) -> [i32; 8] {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = a;
    let mut o = a;
    if strong {
        let t = 2 * tc;
        if !no_p {
            o[3] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - t, p0 + t);
            o[2] = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - t, p1 + t);
            o[1] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - t, p2 + t);
        }
        if !no_q {
            o[4] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - t, q0 + t);
            o[5] = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - t, q1 + t);
            o[6] = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - t, q2 + t);
        }
        return o;
    }
    let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
    if delta.abs() >= tc * 10 {
        return o;
    }
    let delta = delta.clamp(-tc, tc);
    let h = tc >> 1;
    if !no_p {
        o[3] = (p0 + delta).clamp(0, max);
        if dep {
            let d = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-h, h);
            o[2] = (p1 + d).clamp(0, max);
        }
    }
    if !no_q {
        o[4] = (q0 - delta).clamp(0, max);
        if deq {
            let d = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-h, h);
            o[5] = (q1 + d).clamp(0, max);
        }
    }
    o
}

/// 8.7.2.5.3, 8.7.2.5.4 and 8.7.2.5.7 for the 4 lines of one luma edge segment. `at` is q0 of
/// line 0, `is` the index step across the edge, `ks` the step along it.
fn luma_segment(plane: &mut [u16], at: usize, is: usize, ks: usize, s: &Seg, max: i32) {
    let mut l = [[0i32; 8]; 4];
    for (k, line) in l.iter_mut().enumerate() {
        match read::<8>(plane, at + k * ks, is) {
            Some(a) => *line = a,
            None => return,
        }
    }
    let [l0, _, _, l3] = l;
    let [p3_0, p2_0, p1_0, p0_0, q0_0, q1_0, q2_0, q3_0] = l0;
    let [p3_3, p2_3, p1_3, p0_3, q0_3, q1_3, q2_3, q3_3] = l3;
    let dp0 = (p2_0 - 2 * p1_0 + p0_0).abs();
    let dp3 = (p2_3 - 2 * p1_3 + p0_3).abs();
    let dq0 = (q2_0 - 2 * q1_0 + q0_0).abs();
    let dq3 = (q2_3 - 2 * q1_3 + q0_3).abs();
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    let (beta, tc) = (s.beta, s.tc);
    if dpq0 + dpq3 >= beta {
        return; // dE = 0
    }
    let strong = d_sam(2 * dpq0, p0_0, p3_0, q0_0, q3_0, beta, tc)
        && d_sam(2 * dpq3, p0_3, p3_3, q0_3, q3_3, beta, tc);
    let side = (beta + (beta >> 1)) >> 3;
    let (dep, deq) = (dp0 + dp3 < side, dq0 + dq3 < side);
    for (k, &line) in l.iter().enumerate() {
        let o = luma_line(line, strong, dep, deq, tc, max, s.no_p, s.no_q);
        write(plane, at + k * ks, is, &o);
    }
}

/// 8.7.2.5.5 and 8.7.2.5.8 for the 4 lines of one chroma edge segment.
fn chroma_segment(plane: &mut [u16], at: usize, is: usize, ks: usize, s: &Seg, max: i32) {
    for k in 0..4 {
        let base = at + k * ks;
        let Some([p1, p0, q0, q1]) = read::<4>(plane, base, is) else {
            return;
        };
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-s.tc, s.tc);
        let p0n = if s.no_p {
            p0
        } else {
            (p0 + delta).clamp(0, max)
        };
        let q0n = if s.no_q {
            q0
        } else {
            (q0 - delta).clamp(0, max)
        };
        write(plane, base, is, &[p1, p0n, q0n, q1]);
    }
}

/// `N` samples of `s` from `at`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn row<const N: usize>(s: &[u16], at: usize) -> Option<&[u16; N]> {
    s.get(at..)?.first_chunk()
}

/// `N` mutable samples of `s` from `at`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn row_mut<const N: usize>(s: &mut [u16], at: usize) -> Option<&mut [u16; N]> {
    s.get_mut(at..)?.first_chunk_mut()
}

/// `v` in the 4 16-bit lanes of one segment, as the 64 bits of those lanes.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn lanes4(v: i32) -> u64 {
    u64::from(v as u16) * 0x0001_0001_0001_0001
}

#[cfg(target_arch = "x86_64")]
mod x86k {
    use super::{Seg, lanes4, row, row_mut};
    use crate::simd::x86::{ld_u16x8, ld_u16x16, st_u16x8, st_u16x16};
    use core::arch::x86_64::*;

    /// `f(segment)` in the 4 lanes of each of the 4 segments.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn per_seg(segs: &[Seg; 4], f: fn(&Seg) -> i32) -> __m256i {
        let [a, b, c, d] = segs;
        let l = |s: &Seg| lanes4(f(s)) as i64;
        _mm256_set_epi64x(l(d), l(c), l(b), l(a))
    }

    /// Lane 0 of each group of 4 lanes in all 4.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn bcast0(v: __m256i) -> __m256i {
        _mm256_shufflehi_epi16::<0>(_mm256_shufflelo_epi16::<0>(v))
    }

    /// Lanes 0 and 3 of each segment added, in all 4 lanes of the segment.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum03(v: __m256i) -> __m256i {
        bcast0(_mm256_add_epi16(v, _mm256_srli_epi64::<48>(v)))
    }

    /// Mask lanes 0 and 3 of each segment and-ed, in all 4 lanes of the segment.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn all03(m: __m256i) -> __m256i {
        bcast0(_mm256_and_si256(m, _mm256_srli_epi64::<48>(m)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn clip(v: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
        _mm256_min_epi16(_mm256_max_epi16(v, lo), hi)
    }

    /// `v` clipped to `x - t..=x + t`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn around(v: __m256i, x: __m256i, t: __m256i) -> __m256i {
        clip(v, _mm256_sub_epi16(x, t), _mm256_add_epi16(x, t))
    }

    /// 8x8 transpose of 16-bit lanes in each 128-bit half (rows 0..8 low, rows 8..16 high).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn transpose(a: [__m256i; 8]) -> [__m256i; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
        let t0 = _mm256_unpacklo_epi16(a0, a1);
        let t1 = _mm256_unpackhi_epi16(a0, a1);
        let t2 = _mm256_unpacklo_epi16(a2, a3);
        let t3 = _mm256_unpackhi_epi16(a2, a3);
        let t4 = _mm256_unpacklo_epi16(a4, a5);
        let t5 = _mm256_unpackhi_epi16(a4, a5);
        let t6 = _mm256_unpacklo_epi16(a6, a7);
        let t7 = _mm256_unpackhi_epi16(a6, a7);
        let u0 = _mm256_unpacklo_epi32(t0, t2);
        let u1 = _mm256_unpackhi_epi32(t0, t2);
        let u2 = _mm256_unpacklo_epi32(t1, t3);
        let u3 = _mm256_unpackhi_epi32(t1, t3);
        let u4 = _mm256_unpacklo_epi32(t4, t6);
        let u5 = _mm256_unpackhi_epi32(t4, t6);
        let u6 = _mm256_unpacklo_epi32(t5, t7);
        let u7 = _mm256_unpackhi_epi32(t5, t7);
        [
            _mm256_unpacklo_epi64(u0, u4),
            _mm256_unpackhi_epi64(u0, u4),
            _mm256_unpacklo_epi64(u1, u5),
            _mm256_unpackhi_epi64(u1, u5),
            _mm256_unpacklo_epi64(u2, u6),
            _mm256_unpackhi_epi64(u2, u6),
            _mm256_unpacklo_epi64(u3, u7),
            _mm256_unpackhi_epi64(u3, u7),
        ]
    }

    /// [`super::luma_segment`] on 16 lines: `l` is p3..q3, one line per lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn luma(l: &mut [__m256i; 8], segs: &[Seg; 4], max: i32) {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *l;
        let beta = per_seg(segs, |s| s.beta);
        let tc = per_seg(segs, |s| s.tc);
        let keep_p = per_seg(segs, |s| -i32::from(s.no_p));
        let keep_q = per_seg(segs, |s| -i32::from(s.no_q));
        let dp = _mm256_abs_epi16(_mm256_add_epi16(
            _mm256_sub_epi16(p2, _mm256_add_epi16(p1, p1)),
            p0,
        ));
        let dq = _mm256_abs_epi16(_mm256_add_epi16(
            _mm256_sub_epi16(q2, _mm256_add_epi16(q1, q1)),
            q0,
        ));
        let dpq = _mm256_add_epi16(dp, dq);
        // dE != 0: dpq0 + dpq3 < beta.
        let on = _mm256_cmpgt_epi16(beta, sum03(dpq));
        if _mm256_testz_si256(on, on) != 0 {
            return;
        }
        let side = _mm256_srai_epi16::<3>(_mm256_add_epi16(beta, _mm256_srai_epi16::<1>(beta)));
        let dep = _mm256_cmpgt_epi16(side, sum03(dp));
        let deq = _mm256_cmpgt_epi16(side, sum03(dq));
        // dE == 2: dSam of lines 0 and 3.
        let one = _mm256_set1_epi16(1);
        let tc5 = _mm256_srai_epi16::<1>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_slli_epi16::<2>(tc), tc),
            one,
        ));
        let s1 = _mm256_cmpgt_epi16(_mm256_srai_epi16::<2>(beta), _mm256_add_epi16(dpq, dpq));
        let s2 = _mm256_cmpgt_epi16(
            _mm256_srai_epi16::<3>(beta),
            _mm256_add_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(p3, p0)),
                _mm256_abs_epi16(_mm256_sub_epi16(q0, q3)),
            ),
        );
        let s3 = _mm256_cmpgt_epi16(tc5, _mm256_abs_epi16(_mm256_sub_epi16(p0, q0)));
        let strong = _mm256_and_si256(on, all03(_mm256_and_si256(s1, _mm256_and_si256(s2, s3))));
        // Strong filter, clipped to +-2 tC.
        let t2 = _mm256_add_epi16(tc, tc);
        let (c2, c4) = (_mm256_set1_epi16(2), _mm256_set1_epi16(4));
        let a = _mm256_add_epi16(_mm256_add_epi16(p1, p0), q0);
        let b = _mm256_add_epi16(_mm256_add_epi16(p0, q0), q1);
        let p0s = _mm256_srai_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p2, _mm256_add_epi16(a, a)),
            _mm256_add_epi16(q1, c4),
        ));
        let p1s = _mm256_srai_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(p2, a), c2));
        let p2s = _mm256_srai_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_slli_epi16::<1>(_mm256_add_epi16(p3, p2)), p2),
            _mm256_add_epi16(a, c4),
        ));
        let q0s = _mm256_srai_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p1, _mm256_add_epi16(b, b)),
            _mm256_add_epi16(q2, c4),
        ));
        let q1s = _mm256_srai_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(q2, b), c2));
        let q2s = _mm256_srai_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_slli_epi16::<1>(_mm256_add_epi16(q3, q2)), q2),
            _mm256_add_epi16(b, c4),
        ));
        let (p0s, p1s, p2s) = (
            around(p0s, p0, t2),
            around(p1s, p1, t2),
            around(p2s, p2, t2),
        );
        let (q0s, q1s, q2s) = (
            around(q0s, q0, t2),
            around(q1s, q1, t2),
            around(q2s, q2, t2),
        );
        // Normal filter where |delta| < 10 tC.
        let (d0, d1) = (_mm256_sub_epi16(q0, p0), _mm256_sub_epi16(q1, p1));
        let delta = _mm256_srai_epi16::<4>(_mm256_add_epi16(
            _mm256_sub_epi16(
                _mm256_add_epi16(_mm256_slli_epi16::<3>(d0), d0),
                _mm256_add_epi16(_mm256_add_epi16(d1, d1), d1),
            ),
            _mm256_set1_epi16(8),
        ));
        let tc10 = _mm256_add_epi16(_mm256_slli_epi16::<3>(tc), _mm256_add_epi16(tc, tc));
        let weak = _mm256_andnot_si256(
            strong,
            _mm256_and_si256(on, _mm256_cmpgt_epi16(tc10, _mm256_abs_epi16(delta))),
        );
        let z = _mm256_setzero_si256();
        let maxv = _mm256_set1_epi16(max as i16);
        let dc = clip(delta, _mm256_sub_epi16(z, tc), tc);
        let p0n = clip(_mm256_add_epi16(p0, dc), z, maxv);
        let q0n = clip(_mm256_sub_epi16(q0, dc), z, maxv);
        let h = _mm256_srai_epi16::<1>(tc);
        let nh = _mm256_sub_epi16(z, h);
        let p1d = _mm256_srai_epi16::<1>(_mm256_sub_epi16(
            _mm256_add_epi16(_mm256_avg_epu16(p2, p0), dc),
            p1,
        ));
        let p1n = clip(_mm256_add_epi16(p1, clip(p1d, nh, h)), z, maxv);
        let q1d = _mm256_srai_epi16::<1>(_mm256_sub_epi16(
            _mm256_sub_epi16(_mm256_avg_epu16(q2, q0), q1),
            dc,
        ));
        let q1n = clip(_mm256_add_epi16(q1, clip(q1d, nh, h)), z, maxv);
        // Choose per lane; a kept side takes neither filter.
        let (sp, sq) = (
            _mm256_andnot_si256(keep_p, strong),
            _mm256_andnot_si256(keep_q, strong),
        );
        let (wp, wq) = (
            _mm256_andnot_si256(keep_p, weak),
            _mm256_andnot_si256(keep_q, weak),
        );

        l[1] = _mm256_blendv_epi8(p2, p2s, sp);
        l[2] = _mm256_blendv_epi8(
            _mm256_blendv_epi8(p1, p1n, _mm256_and_si256(wp, dep)),
            p1s,
            sp,
        );
        l[3] = _mm256_blendv_epi8(_mm256_blendv_epi8(p0, p0n, wp), p0s, sp);
        l[4] = _mm256_blendv_epi8(_mm256_blendv_epi8(q0, q0n, wq), q0s, sq);
        l[5] = _mm256_blendv_epi8(
            _mm256_blendv_epi8(q1, q1n, _mm256_and_si256(wq, deq)),
            q1s,
            sq,
        );
        l[6] = _mm256_blendv_epi8(q2, q2s, sq);
    }

    /// [`super::chroma_segment`] on 16 lines: `l[2..6]` are p1, p0, q0, q1.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn chroma(l: &mut [__m256i; 8], segs: &[Seg; 4], max: i32) {
        let [_, _, p1, p0, q0, q1, _, _] = *l;
        let tc = per_seg(segs, |s| s.tc);
        let keep_p = per_seg(segs, |s| -i32::from(s.no_p));
        let keep_q = per_seg(segs, |s| -i32::from(s.no_q));
        let delta = _mm256_srai_epi16::<3>(_mm256_add_epi16(
            _mm256_sub_epi16(
                _mm256_add_epi16(_mm256_slli_epi16::<2>(_mm256_sub_epi16(q0, p0)), p1),
                q1,
            ),
            _mm256_set1_epi16(4),
        ));
        let z = _mm256_setzero_si256();
        let maxv = _mm256_set1_epi16(max as i16);
        let dc = clip(delta, _mm256_sub_epi16(z, tc), tc);
        let p0n = clip(_mm256_add_epi16(p0, dc), z, maxv);
        let q0n = clip(_mm256_sub_epi16(q0, dc), z, maxv);
        l[3] = _mm256_blendv_epi8(p0n, p0, keep_p);
        l[4] = _mm256_blendv_epi8(q0n, q0, keep_q);
    }

    /// 4 segments of one edge, 16 lines. `win` starts at p3 of line 0 and its rows are `stride`
    /// apart: a vertical edge is 16 rows of 8 samples, a horizontal edge 8 rows (p3..q3) of 16.
    #[target_feature(enable = "avx2")]
    pub(super) fn edges(
        win: &mut [u16],
        stride: usize,
        vertical: bool,
        luma_edge: bool,
        segs: &[Seg; 4],
        max: i32,
    ) {
        let mut l = [_mm256_setzero_si256(); 8];
        for (r, v) in l.iter_mut().enumerate() {
            if vertical {
                let (Some(lo), Some(hi)) =
                    (row::<8>(win, r * stride), row::<8>(win, (r + 8) * stride))
                else {
                    return;
                };
                *v = _mm256_set_m128i(ld_u16x8(hi), ld_u16x8(lo));
            } else {
                let Some(a) = row::<16>(win, r * stride) else {
                    return;
                };
                *v = ld_u16x16(a);
            }
        }
        if vertical {
            l = transpose(l);
        }
        if luma_edge {
            luma(&mut l, segs, max);
        } else {
            chroma(&mut l, segs, max);
        }
        if vertical {
            l = transpose(l);
        }
        for (r, &v) in l.iter().enumerate() {
            if vertical {
                if let Some(a) = row_mut::<8>(win, r * stride) {
                    st_u16x8(a, _mm256_castsi256_si128(v));
                }
                if let Some(a) = row_mut::<8>(win, (r + 8) * stride) {
                    st_u16x8(a, _mm256_extracti128_si256::<1>(v));
                }
            } else if let Some(a) = row_mut::<16>(win, r * stride) {
                st_u16x16(a, v);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod armk {
    use super::{Seg, lanes4, row, row_mut};
    use crate::simd::arm::{ld_u16x8, st_u16x8};
    use core::arch::aarch64::*;

    /// `f(segment)` in the 4 lanes of each of 2 segments.
    #[inline]
    #[target_feature(enable = "neon")]
    fn per_seg(a: &Seg, b: &Seg, f: fn(&Seg) -> i32) -> int16x8_t {
        vcombine_s16(vcreate_s16(lanes4(f(a))), vcreate_s16(lanes4(f(b))))
    }

    /// Lanes 0 and 3 of each segment added, in all 4 lanes of the segment.
    #[inline]
    #[target_feature(enable = "neon")]
    fn sum03(v: int16x8_t) -> int16x8_t {
        let s = vaddq_s16(
            v,
            vreinterpretq_s16_u64(vshrq_n_u64::<48>(vreinterpretq_u64_s16(v))),
        );
        vcombine_s16(
            vdup_lane_s16::<0>(vget_low_s16(s)),
            vdup_lane_s16::<0>(vget_high_s16(s)),
        )
    }

    /// Mask lanes 0 and 3 of each segment and-ed, in all 4 lanes of the segment.
    #[inline]
    #[target_feature(enable = "neon")]
    fn all03(m: uint16x8_t) -> uint16x8_t {
        let s = vandq_u16(
            m,
            vreinterpretq_u16_u64(vshrq_n_u64::<48>(vreinterpretq_u64_u16(m))),
        );
        vcombine_u16(
            vdup_lane_u16::<0>(vget_low_u16(s)),
            vdup_lane_u16::<0>(vget_high_u16(s)),
        )
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn clip(v: int16x8_t, lo: int16x8_t, hi: int16x8_t) -> int16x8_t {
        vminq_s16(vmaxq_s16(v, lo), hi)
    }

    /// `v` clipped to `x - t..=x + t`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn around(v: int16x8_t, x: int16x8_t, t: int16x8_t) -> int16x8_t {
        clip(v, vsubq_s16(x, t), vaddq_s16(x, t))
    }

    /// 8x8 transpose of 16-bit lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    fn transpose(a: [int16x8_t; 8]) -> [int16x8_t; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = a;
        let t0 = vreinterpretq_s32_s16(vzip1q_s16(a0, a1));
        let t1 = vreinterpretq_s32_s16(vzip2q_s16(a0, a1));
        let t2 = vreinterpretq_s32_s16(vzip1q_s16(a2, a3));
        let t3 = vreinterpretq_s32_s16(vzip2q_s16(a2, a3));
        let t4 = vreinterpretq_s32_s16(vzip1q_s16(a4, a5));
        let t5 = vreinterpretq_s32_s16(vzip2q_s16(a4, a5));
        let t6 = vreinterpretq_s32_s16(vzip1q_s16(a6, a7));
        let t7 = vreinterpretq_s32_s16(vzip2q_s16(a6, a7));
        let u0 = vreinterpretq_s64_s32(vzip1q_s32(t0, t2));
        let u1 = vreinterpretq_s64_s32(vzip2q_s32(t0, t2));
        let u2 = vreinterpretq_s64_s32(vzip1q_s32(t1, t3));
        let u3 = vreinterpretq_s64_s32(vzip2q_s32(t1, t3));
        let u4 = vreinterpretq_s64_s32(vzip1q_s32(t4, t6));
        let u5 = vreinterpretq_s64_s32(vzip2q_s32(t4, t6));
        let u6 = vreinterpretq_s64_s32(vzip1q_s32(t5, t7));
        let u7 = vreinterpretq_s64_s32(vzip2q_s32(t5, t7));
        [
            vreinterpretq_s16_s64(vzip1q_s64(u0, u4)),
            vreinterpretq_s16_s64(vzip2q_s64(u0, u4)),
            vreinterpretq_s16_s64(vzip1q_s64(u1, u5)),
            vreinterpretq_s16_s64(vzip2q_s64(u1, u5)),
            vreinterpretq_s16_s64(vzip1q_s64(u2, u6)),
            vreinterpretq_s16_s64(vzip2q_s64(u2, u6)),
            vreinterpretq_s16_s64(vzip1q_s64(u3, u7)),
            vreinterpretq_s16_s64(vzip2q_s64(u3, u7)),
        ]
    }

    /// [`super::luma_segment`] on 8 lines (segments `a` and `b`): `l` is p3..q3.
    #[inline]
    #[target_feature(enable = "neon")]
    fn luma(l: &mut [int16x8_t; 8], a: &Seg, b: &Seg, max: i32) {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = *l;
        let beta = per_seg(a, b, |s| s.beta);
        let tc = per_seg(a, b, |s| s.tc);
        let keep_p = vreinterpretq_u16_s16(per_seg(a, b, |s| -i32::from(s.no_p)));
        let keep_q = vreinterpretq_u16_s16(per_seg(a, b, |s| -i32::from(s.no_q)));
        let dp = vabsq_s16(vaddq_s16(vsubq_s16(p2, vaddq_s16(p1, p1)), p0));
        let dq = vabsq_s16(vaddq_s16(vsubq_s16(q2, vaddq_s16(q1, q1)), q0));
        let dpq = vaddq_s16(dp, dq);
        let on = vcltq_s16(sum03(dpq), beta);
        if vmaxvq_u16(on) == 0 {
            return;
        }
        let side = vshrq_n_s16::<3>(vaddq_s16(beta, vshrq_n_s16::<1>(beta)));
        let dep = vcltq_s16(sum03(dp), side);
        let deq = vcltq_s16(sum03(dq), side);
        let tc5 = vshrq_n_s16::<1>(vaddq_s16(
            vaddq_s16(vshlq_n_s16::<2>(tc), tc),
            vdupq_n_s16(1),
        ));
        let s1 = vcltq_s16(vaddq_s16(dpq, dpq), vshrq_n_s16::<2>(beta));
        let s2 = vcltq_s16(
            vaddq_s16(vabsq_s16(vsubq_s16(p3, p0)), vabsq_s16(vsubq_s16(q0, q3))),
            vshrq_n_s16::<3>(beta),
        );
        let s3 = vcltq_s16(vabsq_s16(vsubq_s16(p0, q0)), tc5);
        let strong = vandq_u16(on, all03(vandq_u16(s1, vandq_u16(s2, s3))));
        let t2 = vaddq_s16(tc, tc);
        let (c2, c4) = (vdupq_n_s16(2), vdupq_n_s16(4));
        let sa = vaddq_s16(vaddq_s16(p1, p0), q0);
        let sb = vaddq_s16(vaddq_s16(p0, q0), q1);
        let p0s = vshrq_n_s16::<3>(vaddq_s16(
            vaddq_s16(p2, vaddq_s16(sa, sa)),
            vaddq_s16(q1, c4),
        ));
        let p1s = vshrq_n_s16::<2>(vaddq_s16(vaddq_s16(p2, sa), c2));
        let p2s = vshrq_n_s16::<3>(vaddq_s16(
            vaddq_s16(vshlq_n_s16::<1>(vaddq_s16(p3, p2)), p2),
            vaddq_s16(sa, c4),
        ));
        let q0s = vshrq_n_s16::<3>(vaddq_s16(
            vaddq_s16(p1, vaddq_s16(sb, sb)),
            vaddq_s16(q2, c4),
        ));
        let q1s = vshrq_n_s16::<2>(vaddq_s16(vaddq_s16(q2, sb), c2));
        let q2s = vshrq_n_s16::<3>(vaddq_s16(
            vaddq_s16(vshlq_n_s16::<1>(vaddq_s16(q3, q2)), q2),
            vaddq_s16(sb, c4),
        ));
        let (p0s, p1s, p2s) = (
            around(p0s, p0, t2),
            around(p1s, p1, t2),
            around(p2s, p2, t2),
        );
        let (q0s, q1s, q2s) = (
            around(q0s, q0, t2),
            around(q1s, q1, t2),
            around(q2s, q2, t2),
        );
        let (d0, d1) = (vsubq_s16(q0, p0), vsubq_s16(q1, p1));
        let delta = vshrq_n_s16::<4>(vaddq_s16(
            vsubq_s16(
                vaddq_s16(vshlq_n_s16::<3>(d0), d0),
                vaddq_s16(vaddq_s16(d1, d1), d1),
            ),
            vdupq_n_s16(8),
        ));
        let tc10 = vaddq_s16(vshlq_n_s16::<3>(tc), vaddq_s16(tc, tc));
        let weak = vbicq_u16(vandq_u16(on, vcltq_s16(vabsq_s16(delta), tc10)), strong);
        let z = vdupq_n_s16(0);
        let maxv = vdupq_n_s16(max as i16);
        let dc = clip(delta, vnegq_s16(tc), tc);
        let p0n = clip(vaddq_s16(p0, dc), z, maxv);
        let q0n = clip(vsubq_s16(q0, dc), z, maxv);
        let h = vshrq_n_s16::<1>(tc);
        let nh = vnegq_s16(h);
        let p1d = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(vrhaddq_s16(p2, p0), dc), p1));
        let p1n = clip(vaddq_s16(p1, clip(p1d, nh, h)), z, maxv);
        let q1d = vshrq_n_s16::<1>(vsubq_s16(vsubq_s16(vrhaddq_s16(q2, q0), q1), dc));
        let q1n = clip(vaddq_s16(q1, clip(q1d, nh, h)), z, maxv);
        let (sp, sq) = (vbicq_u16(strong, keep_p), vbicq_u16(strong, keep_q));
        let (wp, wq) = (vbicq_u16(weak, keep_p), vbicq_u16(weak, keep_q));
        // vbslq: lanes of the second argument where the mask is set.
        l[1] = vbslq_s16(sp, p2s, p2);
        l[2] = vbslq_s16(sp, p1s, vbslq_s16(vandq_u16(wp, dep), p1n, p1));
        l[3] = vbslq_s16(sp, p0s, vbslq_s16(wp, p0n, p0));
        l[4] = vbslq_s16(sq, q0s, vbslq_s16(wq, q0n, q0));
        l[5] = vbslq_s16(sq, q1s, vbslq_s16(vandq_u16(wq, deq), q1n, q1));
        l[6] = vbslq_s16(sq, q2s, q2);
    }

    /// [`super::chroma_segment`] on 8 lines: `l[2..6]` are p1, p0, q0, q1.
    #[inline]
    #[target_feature(enable = "neon")]
    fn chroma(l: &mut [int16x8_t; 8], a: &Seg, b: &Seg, max: i32) {
        let [_, _, p1, p0, q0, q1, _, _] = *l;
        let tc = per_seg(a, b, |s| s.tc);
        let keep_p = vreinterpretq_u16_s16(per_seg(a, b, |s| -i32::from(s.no_p)));
        let keep_q = vreinterpretq_u16_s16(per_seg(a, b, |s| -i32::from(s.no_q)));
        let delta = vshrq_n_s16::<3>(vaddq_s16(
            vsubq_s16(vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0, p0)), p1), q1),
            vdupq_n_s16(4),
        ));
        let z = vdupq_n_s16(0);
        let maxv = vdupq_n_s16(max as i16);
        let dc = clip(delta, vnegq_s16(tc), tc);
        l[3] = vbslq_s16(keep_p, p0, clip(vaddq_s16(p0, dc), z, maxv));
        l[4] = vbslq_s16(keep_q, q0, clip(vsubq_s16(q0, dc), z, maxv));
    }

    /// 4 segments of one edge, 16 lines, as [`super::x86k`]'s `edges`: two halves of 8 lines.
    #[target_feature(enable = "neon")]
    pub(super) fn edges(
        win: &mut [u16],
        stride: usize,
        vertical: bool,
        luma_edge: bool,
        segs: &[Seg; 4],
        max: i32,
    ) {
        let [s0, s1, s2, s3] = segs;
        for (half, (a, b)) in [(s0, s1), (s2, s3)].into_iter().enumerate() {
            if a.tc == 0 && b.tc == 0 {
                continue;
            }
            // Offset of the half and the step between the 8 loaded vectors.
            let (base, step) = if vertical {
                (half * 8 * stride, stride)
            } else {
                (half * 8, stride)
            };
            let mut l = [vdupq_n_s16(0); 8];
            for (r, v) in l.iter_mut().enumerate() {
                let Some(x) = row::<8>(win, base + r * step) else {
                    return;
                };
                *v = vreinterpretq_s16_u16(ld_u16x8(x));
            }
            if vertical {
                l = transpose(l);
            }
            if luma_edge {
                luma(&mut l, a, b, max);
            } else {
                chroma(&mut l, a, b, max);
            }
            if vertical {
                l = transpose(l);
            }
            for (r, &v) in l.iter().enumerate() {
                if let Some(x) = row_mut::<8>(win, base + r * step) {
                    st_u16x8(x, vreinterpretq_u16_s16(v));
                }
            }
        }
    }
}

/// The SIMD kernel on the 16 lines of 4 segments that start at `at` (q0 of line 0). False when
/// `tier` has no kernel or the lines are not all in `plane`; the caller then runs the scalar
/// code. `tier` must come from `simd::tier()` or `simd::available()`.
// NEON is in the aarch64 baseline: the call may need no unsafe, so unused_unsafe is allowed.
#[allow(unsafe_code, unused_unsafe, clippy::too_many_arguments)]
fn kernel(
    tier: Tier,
    plane: &mut [u16],
    at: usize,
    stride: usize,
    vertical: bool,
    luma: bool,
    segs: &[Seg; 4],
    max: i32,
) -> bool {
    // The window: 16 rows of 8 samples from p3 (vertical edge) or 8 rows of 16 (horizontal).
    let (start, len) = if vertical {
        (
            at.checked_sub(4),
            stride.checked_mul(15).and_then(|v| v.checked_add(8)),
        )
    } else {
        (
            stride.checked_mul(4).and_then(|o| at.checked_sub(o)),
            stride.checked_mul(7).and_then(|v| v.checked_add(16)),
        )
    };
    let Some(win) = start
        .zip(len)
        .and_then(|(s, n)| plane.get_mut(s..s.checked_add(n)?))
    else {
        return false;
    };
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: the match guard has just detected avx2, the only feature the kernel enables;
            // the kernel reads and writes `win` through bounds-checked safe Rust.
            unsafe { x86k::edges(win, stride, vertical, luma, segs, max) };
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: simd::tier() / simd::available() return Neon only when
            // is_aarch64_feature_detected! found neon (src/simd/mod.rs).
            unsafe { armk::edges(win, stride, vertical, luma, segs, max) };
            true
        }
        _ => false,
    }
}

/// Filters the first `n` of the 4 segments `segs` of one edge; segment `k` starts at
/// `at + 4 * k` lines along the edge (`at` is q0 of its first line). A full batch of 4 runs
/// on the SIMD kernel of `tier` (Scalar for bit depths above 10), the rest on the scalar code.
#[allow(clippy::too_many_arguments)]
fn filter4(
    tier: Tier,
    plane: &mut [u16],
    at: usize,
    stride: usize,
    vertical: bool,
    luma: bool,
    segs: &[Seg; 4],
    n: usize,
    max: i32,
) {
    if n == 4 && tier != Tier::Scalar && kernel(tier, plane, at, stride, vertical, luma, segs, max)
    {
        return;
    }
    let (is, ks) = if vertical { (1, stride) } else { (stride, 1) };
    for (k, s) in segs.iter().enumerate().take(n) {
        if s.tc == 0 {
            continue;
        }
        let a = at + 4 * k * ks;
        if luma {
            luma_segment(plane, a, is, ks, s, max);
        } else {
            chroma_segment(plane, a, is, ks, s, max);
        }
    }
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

/// The vertical (`vertical`) or horizontal edges of plane `c` whose first line is in `rows`
/// (rows of this plane). `plane` is the band of the plane that starts at row `row0`: `rows`
/// start at `row0` for the vertical edges and 4 rows below it for the horizontal ones (the filter
/// reads up to 4 rows above the edge).
#[allow(clippy::too_many_arguments)]
fn band(
    tier: Tier,
    meta: &Frame,
    act: &Active,
    c: usize,
    plane: &mut [u16],
    (row0, rows): (usize, Range<usize>),
    vertical: bool,
    cancel: &AtomicBool,
) -> Result<()> {
    let (Some(&w), Some(&stride)) = (meta.width.get(c), meta.stride.get(c)) else {
        return Err(Error::InvalidData("plane"));
    };
    let luma = c == 0;
    let depth = if luma {
        act.sps.bit_depth_luma
    } else {
        act.sps.bit_depth_chroma
    };
    let scale = 1i32 << depth.saturating_sub(8).min(8);
    let max = (1i32 << depth.min(16)) - 1;
    // The kernels work on 16-bit lanes, which hold every intermediate up to depth 10.
    let tier = if depth <= 10 { tier } else { Tier::Scalar };
    // cQpPicOffset: the PPS offset alone.
    let chroma = match c {
        0 => None,
        1 => Some(act.pps.cb_qp_offset),
        _ => Some(act.pps.cr_qp_offset),
    };
    // Luma position of a sample of this plane (4:2:0 chroma is half size both ways).
    let sh = usize::from(!luma);
    let mut segs = [Seg::default(); 4];
    if vertical {
        // Batches of 16 rows on each edge x = 8, 16, ...
        for y in rows.clone().step_by(16) {
            check(cancel)?;
            let n = ((rows.end - y) / 4).min(4);
            for x in (8..w).step_by(8) {
                for (k, s) in segs.iter_mut().enumerate() {
                    *s = if k < n {
                        seg(meta, act, (x << sh, (y + 4 * k) << sh), true, chroma, scale)
                    } else {
                        Seg::default()
                    };
                }
                if segs.iter().any(|s| s.tc != 0) {
                    let at = (y - row0) * stride + x;
                    filter4(tier, plane, at, stride, true, luma, &segs, n, max);
                }
            }
        }
    } else {
        // Edges y = 8, 16, ..., batches of 16 columns.
        for y in rows.filter(|&y| y >= 8 && y.is_multiple_of(8)) {
            check(cancel)?;
            for x in (0..w).step_by(16) {
                let n = ((w - x) / 4).min(4);
                for (k, s) in segs.iter_mut().enumerate() {
                    *s = if k < n {
                        seg(
                            meta,
                            act,
                            ((x + 4 * k) << sh, y << sh),
                            false,
                            chroma,
                            scale,
                        )
                    } else {
                        Seg::default()
                    };
                }
                if segs.iter().any(|s| s.tc != 0) {
                    let at = (y - row0) * stride + x;
                    filter4(tier, plane, at, stride, false, luma, &segs, n, max);
                }
            }
        }
    }
    Ok(())
}

/// Deblocks the whole picture in place: vertical edges first, then horizontal edges. `workers`
/// bands of CTB rows run in parallel (1: on the calling thread, nothing is spawned); the picture
/// is the same for every `workers`.
pub(crate) fn deblock(
    frame: &mut Frame,
    act: &Active,
    cancel: &AtomicBool,
    workers: usize,
) -> Result<()> {
    deblock_with(simd::tier(), frame, act, cancel, workers)
}

/// [`deblock`] on an explicit tier (tests run every available one).
fn deblock_with(
    tier: Tier,
    frame: &mut Frame,
    act: &Active,
    cancel: &AtomicBool,
    workers: usize,
) -> Result<()> {
    // The planes go out of the frame so that the bands can borrow them mutably while every band
    // reads the metadata grids; they are put back on every path.
    let mut planes = std::mem::take(&mut frame.planes);
    let r = deblock_planes(tier, frame, act, &mut planes, cancel, workers);
    frame.planes = planes;
    r
}

/// Both passes over every plane, each pass split into bands of whole CTB rows. The vertical
/// edges of a band only touch samples of its own rows. A horizontal edge at row `y` reads and
/// writes rows `y - 4..y + 4` (luma) or `y - 2..y + 2` (chroma), so the band that owns the edges
/// of CTB rows `[B, B')` is the slice `[B - 4, B' - 4)` of the plane: the footprints stay inside
/// it, and horizontal edges are independent of each other, so any split gives the serial result.
/// A pass ends when all its bands are done (the scope of `par_bands`), before the next pass.
fn deblock_planes(
    tier: Tier,
    meta: &Frame,
    act: &Active,
    planes: &mut [Vec<u16>],
    cancel: &AtomicBool,
    workers: usize,
) -> Result<()> {
    let ctb = act.sps.ctb_size_y as usize;
    let ctb_rows = band_rows(act.sps.pic_height_in_ctbs_y as usize, workers);
    for vertical in [true, false] {
        for (c, plane) in planes.iter_mut().enumerate() {
            let (Some(&h), Some(&stride)) = (meta.height.get(c), meta.stride.get(c)) else {
                return Err(Error::InvalidData("plane"));
            };
            let unit = ctb
                / if c == 0 {
                    1
                } else {
                    (act.sps.sub_height_c as usize).max(1)
                };
            let own = plane_cuts(&ctb_rows, unit, 0, h);
            let cuts = if vertical {
                own.clone()
            } else {
                plane_cuts(&ctb_rows, unit, 4, h)
            };
            par_bands(plane, stride, &cuts, |k, row0, band_plane| {
                let (Some(&y0), Some(&y1)) = (own.get(k), own.get(k + 1)) else {
                    return Ok(());
                };
                band(
                    tier,
                    meta,
                    act,
                    c,
                    band_plane,
                    (row0, y0..y1),
                    vertical,
                    cancel,
                )
            })?;
        }
    }
    Ok(())
}

/// Fuzz entry (`__fuzz_deblock`): one batch of 4 luma or chroma edge segments, vertical or
/// horizontal, built from `data`, on every tier of this CPU, must equal the scalar result.
pub(crate) fn fuzz_tiers(data: &[u8]) -> bool {
    let Some((head, rest)) = data.split_first_chunk::<10>() else {
        return true;
    };
    let [mode, b0, b1, b2, b3, t0, t1, t2, t3, keep] = *head;
    let (vertical, luma) = (mode & 1 != 0, mode & 2 != 0);
    let depth = 8 + u32::from((mode >> 2) % 3);
    let (scale, max) = (1i32 << (depth - 8), (1i32 << depth) - 1);
    let mut segs = [Seg::default(); 4];
    for (k, (s, (b, t))) in segs
        .iter_mut()
        .zip([b0, b1, b2, b3].into_iter().zip([t0, t1, t2, t3]))
        .enumerate()
    {
        *s = Seg {
            beta: if luma { i32::from(b % 65) * scale } else { 0 },
            tc: i32::from(t % 25) * scale,
            no_p: (keep >> (2 * k)) & 1 != 0,
            no_q: (keep >> (2 * k + 1)) & 1 != 0,
        };
    }
    // A 24 x 20 plane; the batch starts at row 4, column 8 (vertical: 16 rows across x = 8;
    // horizontal: 16 columns across y = 4). Samples are the bytes scaled to the depth.
    let stride = 24;
    let at = 4 * stride + 8;
    let mut want = [0u16; 24 * 20];
    for (d, &v) in want.iter_mut().zip(rest.iter().cycle()) {
        *d = u16::from(v) << (depth - 8);
    }
    let mut scalar = want;
    filter4(
        Tier::Scalar,
        &mut scalar,
        at,
        stride,
        vertical,
        luma,
        &segs,
        4,
        max,
    );
    simd::available().into_iter().all(|tier| {
        let mut got = want;
        filter4(tier, &mut got, at, stride, vertical, luma, &segs, 4, max);
        got == scalar
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::SliceParams;
    use crate::ledger::Ledger;
    use crate::limits::Limits;
    use crate::ps::tests::{ledger, pps_bits, sps_bits, sps_bits_ctb16};
    use crate::ps::{activate, parse_pps, parse_sps};
    use std::sync::Arc;

    /// 64x64 4:2:0 picture (one slice, four 32x32 CTBs), every QpY set to `qp`.
    fn setup(l: &Ledger, qp: i8) -> (Active, Frame) {
        let sps = Arc::new(parse_sps(&sps_bits(64, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let act = activate(sps, pps, &Limits::default(), l).unwrap();
        let mut f = Frame::new(&act, l).unwrap();
        let s = SliceParams {
            slice_addr_rs: 0,
            qp_y: 26,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            loop_filter_across_slices: true,
            sao_luma: true,
            sao_chroma: true,
        };
        let i = f.push_slice(s, l).unwrap();
        for rs in 0..4 {
            f.start_ctb(rs, i).unwrap();
        }
        f.fill_qp_y(0, 0, 64, 64, qp).unwrap();
        (act, f)
    }

    fn row(f: &Frame, c: usize, y: usize, x0: usize, n: usize) -> Vec<u16> {
        (x0..x0 + n).map(|x| f.sample(c, x, y).unwrap()).collect()
    }

    #[test]
    fn strong_vertical_edge() {
        let l = ledger();
        let (act, mut f) = setup(&l, 40);
        for y in 0..64 {
            for x in 8..64 {
                f.set_sample(0, x, y, 104).unwrap();
            }
            for x in 0..8 {
                f.set_sample(0, x, y, 100).unwrap();
            }
        }
        f.fill_edge_v(8, 0, 4, 4, 2).unwrap();
        deblock(&mut f, &act, &AtomicBool::new(false), 1).unwrap();
        // QpL 40: beta 42, tC 7 (Q 42); flat sides give dE 2, strong filter by hand.
        assert_eq!(
            row(&f, 0, 0, 4, 8),
            [100, 101, 101, 102, 103, 103, 104, 104]
        );
        assert_eq!(
            row(&f, 0, 3, 4, 8),
            [100, 101, 101, 102, 103, 103, 104, 104]
        );
        // The next segment has no edge mark.
        assert_eq!(
            row(&f, 0, 4, 4, 8),
            [100, 100, 100, 100, 104, 104, 104, 104]
        );
    }

    #[test]
    fn weak_horizontal_edge_and_unfiltered_side() {
        let l = ledger();
        let (act, mut f) = setup(&l, 30);
        for y in 0..64 {
            for x in 0..64 {
                f.set_sample(0, x, y, if y < 8 { 100 } else { 108 })
                    .unwrap();
            }
        }
        f.fill_edge_h(0, 8, 4, 4, 2).unwrap();
        let mut g = Frame::new(&act, &l).unwrap();
        g.planes = f.planes.clone();
        g.qp_y = f.qp_y.clone();
        g.slice_idx = f.slice_idx.clone();
        g.slices = f.slices.clone();
        g.edge_h = f.edge_h.clone();
        // pcm / bypass above the edge: p side stays.
        g.fill_no_filter(0, 0, 64, 8, 1).unwrap();
        let col = |f: &Frame| -> Vec<u16> { (4..12).map(|y| f.sample(0, 0, y).unwrap()).collect() };
        deblock(&mut f, &act, &AtomicBool::new(false), 1).unwrap();
        deblock(&mut g, &act, &AtomicBool::new(false), 1).unwrap();
        // QpL 30: beta 22, tC 3 (Q 32); |p0-q0| = 8 is not < 8, so weak.
        // delta = 3, p0 103, q0 105; p1 += 1, q1 -= 1.
        assert_eq!(col(&f), [100, 100, 101, 103, 105, 107, 108, 108]);
        assert_eq!(col(&g), [100, 100, 100, 100, 105, 107, 108, 108]);
        l.release(g.charged);
    }

    #[test]
    fn chroma_edge_uses_pps_offset() {
        let l = ledger();
        let (act, mut f) = setup(&l, 40);
        for c in 1..3 {
            for y in 0..32 {
                for x in 0..32 {
                    f.set_sample(c, x, y, if x < 8 { 100 } else { 104 })
                        .unwrap();
                }
            }
        }
        // Chroma x 8 is luma x 16; one 8-row luma segment.
        f.fill_edge_v(16, 0, 4, 8, 2).unwrap();
        deblock(&mut f, &act, &AtomicBool::new(false), 1).unwrap();
        // Cb: qPi 41 -> QpC 36 -> Q 38 -> tC 5; delta = 2 (not clipped). Cr (offset -1): tC 4.
        for c in 1..3 {
            assert_eq!(row(&f, c, 3, 6, 4), [100, 102, 102, 104]);
            assert_eq!(row(&f, c, 4, 6, 4), [100, 100, 104, 104]);
        }
    }

    #[test]
    fn cancel_and_tables() {
        let l = ledger();
        let (act, mut f) = setup(&l, 40);
        assert_eq!(
            deblock(&mut f, &act, &AtomicBool::new(true), 1),
            Err(Error::Cancelled)
        );
        assert_eq!(
            (
                lookup(&BETA_TAB, 51),
                lookup(&BETA_TAB, 99),
                lookup(&BETA_TAB, -5)
            ),
            (64, 64, 0)
        );
        assert_eq!(
            (
                lookup(&TC_TAB, 53),
                lookup(&TC_TAB, 18),
                lookup(&TC_TAB, 17)
            ),
            (24, 1, 0)
        );
        assert_eq!(
            (qpc(29), qpc(30), qpc(43), qpc(44), qpc(-3)),
            (29, 29, 37, 38, -3)
        );
    }

    /// A random `w` x `h` picture (CTB 32, or 16 if `ctb16`; the last CTB row is partial): two
    /// slices with different offsets, random QpY, edges, no_filter cells and blocky samples of
    /// `depth` bits (a few blocks at the ends of the range).
    fn random_sized(
        l: &Ledger,
        seed: u64,
        (w, h): (usize, usize),
        ctb16: bool,
        depth: u32,
    ) -> (Active, Frame) {
        let (w32, h32) = (u32::try_from(w).unwrap(), u32::try_from(h).unwrap());
        let bits = if ctb16 {
            sps_bits_ctb16(w32, h32)
        } else {
            sps_bits(w32, h32)
        };
        let mut sps = parse_sps(&bits).unwrap();
        (sps.bit_depth_luma, sps.bit_depth_chroma) = (depth, depth);
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let act = activate(Arc::new(sps), pps, &Limits::default(), l).unwrap();
        let mut f = Frame::new(&act, l).unwrap();
        let (sc, maxv) = (1i32 << (depth - 8), (1i32 << depth) - 1);
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for k in 0..2 {
            let s = SliceParams {
                slice_addr_rs: k,
                qp_y: 26,
                cb_qp_offset: 0,
                cr_qp_offset: 0,
                deblocking_disabled: false,
                beta_offset_div2: i32::try_from(next() % 13).unwrap() - 6,
                tc_offset_div2: i32::try_from(next() % 13).unwrap() - 6,
                loop_filter_across_slices: true,
                sao_luma: true,
                sao_chroma: true,
            };
            f.push_slice(s, l).unwrap();
        }
        for rs in 0..f.slice_idx.len() {
            f.start_ctb(rs, u32::try_from(next() % 2).unwrap()).unwrap();
        }
        for c in 0..3 {
            let (pw, ph) = (f.width[c], f.height[c]);
            let bw = pw / 8 + 1;
            let bases: Vec<i32> = (0..bw * (ph / 8 + 1))
                .map(|_| match next() % 16 {
                    0 => i32::try_from(next() % 3).unwrap(),
                    1 => maxv - i32::try_from(next() % 3).unwrap(),
                    _ => (90 + i32::try_from(next() % 60).unwrap()) * sc,
                })
                .collect();
            for y in 0..ph {
                for x in 0..pw {
                    let noise = (i32::try_from(next() % 5).unwrap() - 2) * sc;
                    let v = (bases[(y / 8) * bw + x / 8] + noise).clamp(0, maxv);
                    f.planes[c][y * pw + x] = u16::try_from(v).unwrap();
                }
            }
        }
        for gy in 0..f.grid_h {
            for gx in 0..f.grid_w {
                let (x, y) = (gx * 4, gy * 4);
                let qp = i8::try_from(next() % 70).unwrap() - 10;
                f.fill_qp_y(x, y, 4, 4, qp).unwrap();
                f.fill_edge_v(x, y, 4, 4, [0, 1, 2, 2][(next() % 4) as usize])
                    .unwrap();
                f.fill_edge_h(x, y, 4, 4, [0, 1, 2, 2][(next() % 4) as usize])
                    .unwrap();
                if next().is_multiple_of(8) {
                    f.fill_no_filter(x, y, 4, 4, 1).unwrap();
                }
            }
        }
        (act, f)
    }

    /// Any number of bands gives the planes of one band; cancel puts the planes back.
    #[test]
    fn workers_match_serial() {
        let l = ledger();
        let cancel = AtomicBool::new(false);
        for (w, h, ctb16) in [
            (64, 72, false),
            (200, 136, false),
            (64, 72, true),
            (200, 136, true),
        ] {
            for seed in 1..=4u64 {
                let (act, mut want) = random_sized(&l, seed * 0x1234_5679, (w, h), ctb16, 8);
                let orig = want.planes.clone();
                deblock(&mut want, &act, &cancel, 1).unwrap();
                assert!(
                    orig != want.planes,
                    "not filtered: {w}x{h} ctb16 {ctb16} seed {seed}"
                );
                for workers in [2, 3, 5, 64] {
                    let (act, mut got) = random_sized(&l, seed * 0x1234_5679, (w, h), ctb16, 8);
                    deblock(&mut got, &act, &cancel, workers).unwrap();
                    assert!(
                        got.planes == want.planes,
                        "{w}x{h} ctb16 {ctb16} seed {seed} workers {workers}"
                    );
                    l.release(got.charged);
                }
                l.release(want.charged);
            }
        }
        let (act, mut f) = random_sized(&l, 7, (200, 136), false, 8);
        let orig = f.planes.clone();
        assert_eq!(
            deblock(&mut f, &act, &AtomicBool::new(true), 3),
            Err(Error::Cancelled)
        );
        assert!(f.planes == orig);
    }

    /// Every SIMD tier gives the scalar planes: whole pictures with random bS, QpY, slice
    /// offsets, PCM/bypass cells and samples, bit depths 8 to 12 (above 10 the tiers run scalar),
    /// sizes that leave partial batches of segments at the right and bottom.
    #[test]
    fn tiers_match_scalar() {
        let l = ledger();
        let cancel = AtomicBool::new(false);
        let tiers = simd::available();
        for depth in [8, 9, 10, 12] {
            for (w, h, ctb16) in [
                (64, 72, false),
                (200, 136, false),
                (16, 8, true),
                (24, 40, true),
                (40, 24, true),
                (200, 136, true),
            ] {
                for seed in 1..=3u64 {
                    let s = seed * 0x9e37_79b9 + u64::from(depth);
                    let (act, mut want) = random_sized(&l, s, (w, h), ctb16, depth);
                    let orig = want.planes.clone();
                    deblock_with(Tier::Scalar, &mut want, &act, &cancel, 1).unwrap();
                    assert!(w < 64 || orig != want.planes, "not filtered");
                    for &tier in &tiers {
                        let (act, mut got) = random_sized(&l, s, (w, h), ctb16, depth);
                        deblock_with(tier, &mut got, &act, &cancel, 1).unwrap();
                        assert!(
                            got.planes == want.planes,
                            "{tier:?} depth {depth} {w}x{h} ctb16 {ctb16} seed {seed}"
                        );
                        l.release(got.charged);
                    }
                    l.release(want.charged);
                }
            }
        }
    }

    /// The fuzz entry on xorshift bytes: random samples, smooth samples and smooth samples
    /// with a step at the edge (so every filter decision is taken).
    #[test]
    fn fuzz_tiers_random_bytes() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        assert!(fuzz_tiers(&[]));
        for i in 0..4000 {
            let mut d: Vec<u8> = (0..10 + i % 700).map(|_| next() as u8).collect();
            if i % 3 != 0 {
                // A 24 x 20 plane: a level, noise of 0..4, and on every third case a step
                // between the p side and the q side of the batch (x 8 or row 4).
                let (lvl, step) = (next() as u8 % 200, next() as u8 % 48);
                let vertical = d[0] & 1 != 0;
                d.truncate(10);
                for k in 0..480usize {
                    let (r, c) = (k / 24, k % 24);
                    let q = if vertical { c >= 8 } else { r >= 4 };
                    let st = if i % 3 == 2 && q { step } else { 0 };
                    d.push(lvl + (next() % 4) as u8 + st);
                }
            }
            assert!(fuzz_tiers(&d), "case {i}");
        }
    }
}
