//! AVX2 kernels of intra prediction (docs/safety.md). Safe `#[target_feature]` fns on 8 lanes of
//! i32; the arrays come from slices in safe code (`arr`, `chunks_exact`), the loads and stores
//! are the helpers of `simd::x86`. Called only from `super::run`.

use super::{Block, MAX_REF, Op, Pred, RF_LEN, Refs, arr};
use crate::simd::x86::{ld_i32x8, st_i32x8, st_u16x8};
use core::arch::x86_64::*;

/// Runs one kernel call.
#[target_feature(enable = "avx2")]
pub(super) fn run(op: Op) {
    match op {
        Op::Smooth { src, dst, len } => smooth(src, dst, len),
        Op::Planar { p, out, log2 } => planar(p, out, log2),
        Op::Angular { rf, angle, n, out } => angular(rf, angle, n, out),
        Op::Store { block, out, n, max } => store(block, out, n, max),
    }
}

/// (8-41)..(8-45): dst[i] = (src[i-1] + 2 * src[i] + src[i+1] + 2) >> 2 for 0 < i < len - 1,
/// 8 at a time. The vectors run on past len - 1 (into the padding of the arrays); the tail copy
/// puts the unchanged end point and the zero padding back.
#[target_feature(enable = "avx2")]
fn smooth(src: &[i32; MAX_REF], dst: &mut [i32; MAX_REF], len: usize) {
    let two = _mm256_set1_epi32(2);
    for i in (1..len.saturating_sub(1)).step_by(8) {
        let (Some(a), Some(b), Some(c)) =
            (arr::<8>(src, i - 1), arr::<8>(src, i), arr::<8>(src, i + 1))
        else {
            continue;
        };
        let Some(d) = dst
            .get_mut(i..i + 8)
            .and_then(|s| <&mut [i32; 8]>::try_from(s).ok())
        else {
            continue;
        };
        let vb = ld_i32x8(b);
        let s = _mm256_add_epi32(
            _mm256_add_epi32(ld_i32x8(a), ld_i32x8(c)),
            _mm256_add_epi32(_mm256_add_epi32(vb, vb), two),
        );
        st_i32x8(d, _mm256_srai_epi32::<2>(s));
    }
    let end = len.saturating_sub(1);
    if let (Some(d), Some(s)) = (dst.get_mut(end..), src.get(end..)) {
        d.copy_from_slice(s);
    }
}

/// (8-46) for log2 = 2..=5; the lanes past the block size are scratch.
#[target_feature(enable = "avx2")]
fn planar(p: &Refs, out: &mut Block, log2: u32) {
    let n = 1usize << log2;
    let ni = n as i32;
    let (tr, bl) = (p.top(n + 1), p.left(n + 1));
    let shift = _mm_cvtsi32_si128(log2 as i32 + 1);
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (vtr, one, vnm1) = (
        _mm256_set1_epi32(tr),
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(ni - 1),
    );
    for (y, row) in out.iter_mut().enumerate().take(n) {
        let yi = y as i32;
        let vl = _mm256_set1_epi32(p.left(y + 1));
        let wy = _mm256_set1_epi32(ni - 1 - yi);
        let k = _mm256_set1_epi32((yi + 1) * bl + ni);
        for (ci, c) in row.chunks_exact_mut(8).enumerate().take(n.div_ceil(8)) {
            let Ok(dst) = <&mut [i32; 8]>::try_from(c) else {
                continue;
            };
            let Some(t) = arr::<8>(&p.r, 2 * n + 1 + 8 * ci) else {
                continue;
            };
            let x = _mm256_add_epi32(lanes, _mm256_set1_epi32(8 * ci as i32));
            let l = _mm256_mullo_epi32(_mm256_sub_epi32(vnm1, x), vl);
            let r = _mm256_mullo_epi32(_mm256_add_epi32(x, one), vtr);
            let top = _mm256_mullo_epi32(wy, ld_i32x8(t));
            let s = _mm256_add_epi32(_mm256_add_epi32(l, r), _mm256_add_epi32(top, k));
            st_i32x8(dst, _mm256_sra_epi32(s, shift));
        }
    }
}

/// out[a][b] = ((32 - f) * ref[b + i + 1] + f * ref[b + i + 2] + 16) >> 5 (ref[x] at x + 32,
/// i = (a + 1) * angle >> 5, f = (a + 1) * angle & 31). For f = 0 this is ref[b + i + 1], the
/// scalar's special case.
#[target_feature(enable = "avx2")]
fn angular(rf: &[i32; RF_LEN], angle: i32, n: usize, out: &mut Block) {
    let rnd = _mm256_set1_epi32(16);
    for (a, row) in out.iter_mut().enumerate().take(n) {
        let pos = (a as i32 + 1) * angle;
        let (idx, f) = (pos >> 5, pos & 31);
        let (wa, wb) = (_mm256_set1_epi32(32 - f), _mm256_set1_epi32(f));
        for (ci, c) in row.chunks_exact_mut(8).enumerate().take(n.div_ceil(8)) {
            let Ok(dst) = <&mut [i32; 8]>::try_from(c) else {
                continue;
            };
            let Ok(base) = usize::try_from(32 + 8 * ci as i32 + idx + 1) else {
                continue;
            };
            let (Some(x), Some(y)) = (arr::<8>(rf, base), arr::<8>(rf, base + 1)) else {
                continue;
            };
            let s = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(wa, ld_i32x8(x)),
                    _mm256_mullo_epi32(wb, ld_i32x8(y)),
                ),
                rnd,
            );
            st_i32x8(dst, _mm256_srai_epi32::<5>(s));
        }
    }
}

/// Clamp to 0..=max and narrow to u16, 8 samples at a time.
#[target_feature(enable = "avx2")]
fn store(block: &Block, out: &mut Pred, n: usize, max: i32) {
    let (lo, hi) = (_mm256_setzero_si256(), _mm256_set1_epi32(max));
    for (src, dst) in block.iter().zip(out.iter_mut()).take(n) {
        for (s, d) in src
            .chunks_exact(8)
            .zip(dst.chunks_exact_mut(8))
            .take(n.div_ceil(8))
        {
            let (Ok(s), Ok(d)) = (<&[i32; 8]>::try_from(s), <&mut [u16; 8]>::try_from(d)) else {
                continue;
            };
            let v = _mm256_min_epi32(_mm256_max_epi32(ld_i32x8(s), lo), hi);
            let w = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
            st_u16x8(d, w);
        }
    }
}
