//! AVX-512 kernels of intra prediction (docs/safety.md). Safe `#[target_feature]` fns on 16
//! lanes of i32 (AVX-512 F; the dispatch also requires BW and VL, which `Tier::Avx512` means).
//! A 4x4 block fills one vector with 4 rows and an 8x8 block with 2 rows, so the small blocks
//! (the most frequent ones) cost one set of vector ops per 16 samples, not per row. The
//! reference samples of those rows come from `vpermd` of one load, the weights are computed per
//! lane. Planar and angular of 16x16 and 32x32 blocks gain nothing from 16 lanes (the per-row
//! work is the same), so they run the AVX2 kernels. Output stays the AVX2 kernels' bit for bit.
//! Called only from `super::run`.

use super::{Block, MAX_REF, Op, Pred, RF_LEN, Refs, arr};
use crate::simd::x86::{
    ld_i32x4, ld_i32x8, ld_i32x16, st_i32x4, st_i32x8, st_i32x16, st_u16x4, st_u16x8, st_u16x16,
};
use core::arch::x86_64::*;

/// Runs one kernel call.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) fn run(op: Op) {
    match op {
        Op::Planar { log2, .. } if log2 >= 4 => super::x86::run(op),
        Op::Angular { n, .. } if n >= 16 => super::x86::run(op),
        Op::Smooth { src, dst, len } => smooth(src, dst, len),
        Op::Planar { p, out, log2 } => planar(p, out, log2),
        Op::Angular { rf, angle, n, out } => angular(rf, angle, n, out),
        Op::Store { block, out, n, max } => store(block, out, n, max),
    }
}

/// Lanes 0..16.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn lanes() -> __m512i {
    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

/// Rows of a block in one vector of 16 lanes: 4 for n = 4, 2 for n = 8, else 1.
fn rows_per_vec(n: usize) -> usize {
    16usize.checked_div(n).unwrap_or(1).max(1)
}

/// The first 4 samples of row y of `block`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn ld4(block: &Block, y: usize) -> Option<__m128i> {
    Some(ld_i32x4(block.get(y)?.as_chunks::<4>().0.first()?))
}

/// The first 8 samples of row y of `block`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn ld8(block: &Block, y: usize) -> Option<__m256i> {
    Some(ld_i32x8(block.get(y)?.as_chunks::<8>().0.first()?))
}

/// Stores `v` (rows y.. of an n-wide block, n = 4 or 8) into `out`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn put(out: &mut Block, y: usize, n: usize, v: __m512i) {
    match n {
        4 => {
            let parts = [
                _mm512_castsi512_si128(v),
                _mm512_extracti32x4_epi32::<1>(v),
                _mm512_extracti32x4_epi32::<2>(v),
                _mm512_extracti32x4_epi32::<3>(v),
            ];
            for (k, q) in parts.into_iter().enumerate() {
                if let Some(d) = out
                    .get_mut(y + k)
                    .and_then(|r| r.as_chunks_mut::<4>().0.first_mut())
                {
                    st_i32x4(d, q);
                }
            }
        }
        _ => {
            let parts = [_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64::<1>(v)];
            for (k, q) in parts.into_iter().enumerate() {
                if let Some(d) = out
                    .get_mut(y + k)
                    .and_then(|r| r.as_chunks_mut::<8>().0.first_mut())
                {
                    st_i32x8(d, q);
                }
            }
        }
    }
}

/// (8-41)..(8-45): dst[i] = (src[i-1] + 2 * src[i] + src[i+1] + 2) >> 2 for 0 < i < len - 1,
/// 16 at a time. The vectors run on past len - 1 (into the padding of the arrays); the tail copy
/// puts the unchanged end point and the zero padding back. len <= 129, so i <= 113 and the loads
/// reach index 129 of the 136 slots.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn smooth(src: &[i32; MAX_REF], dst: &mut [i32; MAX_REF], len: usize) {
    let two = _mm512_set1_epi32(2);
    for i in (1..len.saturating_sub(1)).step_by(16) {
        let (Some(a), Some(b), Some(c)) = (
            arr::<16>(src, i - 1),
            arr::<16>(src, i),
            arr::<16>(src, i + 1),
        ) else {
            continue;
        };
        let Some(d) = dst
            .get_mut(i..i + 16)
            .and_then(|s| <&mut [i32; 16]>::try_from(s).ok())
        else {
            continue;
        };
        let vb = ld_i32x16(b);
        let s = _mm512_add_epi32(
            _mm512_add_epi32(ld_i32x16(a), ld_i32x16(c)),
            _mm512_add_epi32(_mm512_add_epi32(vb, vb), two),
        );
        st_i32x16(d, _mm512_srai_epi32::<2>(s));
    }
    let end = len.saturating_sub(1);
    if let (Some(d), Some(s)) = (dst.get_mut(end..), src.get(end..)) {
        d.copy_from_slice(s);
    }
}

/// (8-46) for log2 = 2..=3 (`run` sends larger blocks to AVX2). A vector holds 16 / n rows, with
/// the left and top samples of its lanes picked from one load by `vpermd`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn planar(p: &Refs, out: &mut Block, log2: u32) {
    let n = 1usize << log2;
    let ni = n as i32;
    let (tr, bl) = (p.top(n + 1), p.left(n + 1));
    let shift = _mm_cvtsi32_si128(log2 as i32 + 1);
    let rsh = _mm_cvtsi32_si128(log2 as i32);
    let lane = lanes();
    let xmask = _mm512_set1_epi32(ni - 1);
    let (vtr, vbl, one, vnm1, vn) = (
        _mm512_set1_epi32(tr),
        _mm512_set1_epi32(bl),
        _mm512_set1_epi32(1),
        _mm512_set1_epi32(ni - 1),
        _mm512_set1_epi32(ni),
    );
    // Small blocks: left(1..=n) is r[2n - 1 - y] and top(1..=n) is r[2n + 1 + x], both in one load.
    let (Some(lp), Some(tp)) = (arr::<16>(&p.r, 0), arr::<16>(&p.r, 2 * n + 1)) else {
        return;
    };
    let (leftv, topv) = (ld_i32x16(lp), ld_i32x16(tp));
    for g in (0..n).step_by(rows_per_vec(n)) {
        let vy = _mm512_add_epi32(_mm512_set1_epi32(g as i32), _mm512_srl_epi32(lane, rsh));
        let vl =
            _mm512_permutexvar_epi32(_mm512_sub_epi32(_mm512_set1_epi32(2 * ni - 1), vy), leftv);
        let wy = _mm512_sub_epi32(vnm1, vy);
        let k = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_add_epi32(vy, one), vbl), vn);
        let vx = _mm512_and_si512(lane, xmask);
        let top = _mm512_permutexvar_epi32(vx, topv);
        let l = _mm512_mullo_epi32(_mm512_sub_epi32(vnm1, vx), vl);
        let r = _mm512_mullo_epi32(_mm512_add_epi32(vx, one), vtr);
        let t = _mm512_mullo_epi32(wy, top);
        let s = _mm512_add_epi32(_mm512_add_epi32(l, r), _mm512_add_epi32(t, k));
        put(out, g, n, _mm512_sra_epi32(s, shift));
    }
}

/// out[a][b] = ((32 - f) * ref[b + i + 1] + f * ref[b + i + 2] + 16) >> 5 (ref[x] at x + 32,
/// i = (a + 1) * angle >> 5, f = (a + 1) * angle & 31). For f = 0 this is ref[b + i + 1], the
/// scalar's special case. For n = 4 or 8 (`run` sends larger blocks to AVX2) a vector holds 16 / n
/// rows; their reference samples are picked from one load at the smallest i of the rows by
/// `vpermd` (the rows differ by at most 3 in i, so 16 lanes reach every sample). The load starts
/// at index 32 + imin + 1, between 25 and 40, and ends before 56, inside the 112 slots of `rf`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn angular(rf: &[i32; RF_LEN], angle: i32, n: usize, out: &mut Block) {
    let rsh = _mm_cvtsi32_si128(n.trailing_zeros() as i32);
    let lane = lanes();
    let bmask = _mm512_set1_epi32(n as i32 - 1);
    let (rnd, c32, c31, one, vangle) = (
        _mm512_set1_epi32(16),
        _mm512_set1_epi32(32),
        _mm512_set1_epi32(31),
        _mm512_set1_epi32(1),
        _mm512_set1_epi32(angle),
    );
    let rp = rows_per_vec(n);
    for g in (0..n).step_by(rp) {
        let a = _mm512_add_epi32(_mm512_set1_epi32(g as i32 + 1), _mm512_srl_epi32(lane, rsh));
        let pos = _mm512_mullo_epi32(a, vangle);
        let vidx = _mm512_srai_epi32::<5>(pos);
        let vf = _mm512_and_si512(pos, c31);
        let wa = _mm512_sub_epi32(c32, vf);
        let i0 = ((g as i32 + 1) * angle) >> 5;
        let imin = i0.min(((g as i32 + rp as i32) * angle) >> 5);
        let Ok(base) = usize::try_from(32 + imin + 1) else {
            continue;
        };
        let Some(r) = arr::<16>(rf, base) else {
            continue;
        };
        let rel = _mm512_add_epi32(
            _mm512_sub_epi32(vidx, _mm512_set1_epi32(imin)),
            _mm512_and_si512(lane, bmask),
        );
        let rv = ld_i32x16(r);
        let x = _mm512_permutexvar_epi32(rel, rv);
        let y = _mm512_permutexvar_epi32(_mm512_add_epi32(rel, one), rv);
        let s = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_mullo_epi32(wa, x), _mm512_mullo_epi32(vf, y)),
            rnd,
        );
        put(out, g, n, _mm512_srai_epi32::<5>(s));
    }
}

/// Clamp 16 lanes to 0..=hi and narrow them to u16 (`hi` <= 65535, so the truncation is exact).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn clamp16(v: __m512i, hi: __m512i) -> __m256i {
    _mm512_cvtepi32_epi16(_mm512_min_epi32(
        _mm512_max_epi32(v, _mm512_setzero_si512()),
        hi,
    ))
}

/// Clamp to 0..=max and narrow to u16. n = 4 does 4 rows and n = 8 does 2 rows per vector.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn store(block: &Block, out: &mut Pred, n: usize, max: i32) {
    let hi = _mm512_set1_epi32(max);
    match n {
        4 => {
            let (Some(a), Some(b), Some(c), Some(d)) =
                (ld4(block, 0), ld4(block, 1), ld4(block, 2), ld4(block, 3))
            else {
                return;
            };
            let v = _mm512_inserti32x4::<3>(
                _mm512_inserti32x4::<2>(_mm512_inserti32x4::<1>(_mm512_castsi128_si512(a), b), c),
                d,
            );
            let w = clamp16(v, hi);
            let (l, h) = (_mm256_castsi256_si128(w), _mm256_extracti128_si256::<1>(w));
            let parts = [l, _mm_unpackhi_epi64(l, l), h, _mm_unpackhi_epi64(h, h)];
            for (y, q) in parts.into_iter().enumerate() {
                if let Some(d) = out
                    .get_mut(y)
                    .and_then(|r| r.as_chunks_mut::<4>().0.first_mut())
                {
                    st_u16x4(d, q);
                }
            }
        }
        8 => {
            for y in (0..n).step_by(2) {
                let (Some(a), Some(b)) = (ld8(block, y), ld8(block, y + 1)) else {
                    continue;
                };
                let w = clamp16(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b), hi);
                let parts = [_mm256_castsi256_si128(w), _mm256_extracti128_si256::<1>(w)];
                for (k, q) in parts.into_iter().enumerate() {
                    if let Some(d) = out
                        .get_mut(y + k)
                        .and_then(|r| r.as_chunks_mut::<8>().0.first_mut())
                    {
                        st_u16x8(d, q);
                    }
                }
            }
        }
        _ => {
            for (src, dst) in block.iter().zip(out.iter_mut()).take(n) {
                for (s, d) in src
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<16>().0.iter_mut())
                    .take(n.div_ceil(16))
                {
                    st_u16x16(d, clamp16(ld_i32x16(s), hi));
                }
            }
        }
    }
}
