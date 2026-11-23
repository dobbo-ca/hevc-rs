//! NEON kernels of intra prediction (docs/safety.md). Safe `#[target_feature]` fns on 4 lanes of
//! i32 (a block is a whole number of 4-lane chunks); the arrays come from slices in safe code
//! (`arr`, `chunks_exact`), the loads and stores are the helpers of `simd::arm`. Called only
//! from `super::run`.

use super::{Block, MAX_REF, Op, Pred, RF_LEN, Refs, arr};
use crate::simd::arm::{ld_i32x4, st_i32x4, st_u16x4};
use core::arch::aarch64::*;

/// Runs one kernel call.
#[target_feature(enable = "neon")]
pub(super) fn run(op: Op) {
    match op {
        Op::Smooth { src, dst, len } => smooth(src, dst, len),
        Op::Planar { p, out, log2 } => planar(p, out, log2),
        Op::Angular { rf, angle, n, out } => angular(rf, angle, n, out),
        Op::Store { block, out, n, max } => store(block, out, n, max),
    }
}

/// (8-41)..(8-45): dst[i] = (src[i-1] + 2 * src[i] + src[i+1] + 2) >> 2 for 0 < i < len - 1,
/// 4 at a time. The vectors run on past len - 1 (into the padding of the arrays); the tail copy
/// puts the unchanged end point and the zero padding back.
#[target_feature(enable = "neon")]
fn smooth(src: &[i32; MAX_REF], dst: &mut [i32; MAX_REF], len: usize) {
    let two = vdupq_n_s32(2);
    for i in (1..len.saturating_sub(1)).step_by(4) {
        let (Some(a), Some(b), Some(c)) =
            (arr::<4>(src, i - 1), arr::<4>(src, i), arr::<4>(src, i + 1))
        else {
            continue;
        };
        let Some(d) = dst
            .get_mut(i..i + 4)
            .and_then(|s| <&mut [i32; 4]>::try_from(s).ok())
        else {
            continue;
        };
        let vb = ld_i32x4(b);
        let s = vaddq_s32(
            vaddq_s32(ld_i32x4(a), ld_i32x4(c)),
            vaddq_s32(vaddq_s32(vb, vb), two),
        );
        st_i32x4(d, vshrq_n_s32::<2>(s));
    }
    let end = len.saturating_sub(1);
    if let (Some(d), Some(s)) = (dst.get_mut(end..), src.get(end..)) {
        d.copy_from_slice(s);
    }
}

/// (8-46) for log2 = 2..=5.
#[target_feature(enable = "neon")]
fn planar(p: &Refs, out: &mut Block, log2: u32) {
    let n = 1usize << log2;
    let ni = n as i32;
    let (tr, bl) = (p.top(n + 1), p.left(n + 1));
    // vshlq_s32 with a negative count is an arithmetic shift right.
    let shift = vdupq_n_s32(-(log2 as i32 + 1));
    let lanes = ld_i32x4(&[0, 1, 2, 3]);
    let (vtr, one, vnm1) = (vdupq_n_s32(tr), vdupq_n_s32(1), vdupq_n_s32(ni - 1));
    for (y, row) in out.iter_mut().enumerate().take(n) {
        let yi = y as i32;
        let vl = vdupq_n_s32(p.left(y + 1));
        let wy = vdupq_n_s32(ni - 1 - yi);
        let k = vdupq_n_s32((yi + 1) * bl + ni);
        for (ci, c) in row.chunks_exact_mut(4).enumerate().take(n / 4) {
            let Ok(dst) = <&mut [i32; 4]>::try_from(c) else {
                continue;
            };
            let Some(t) = arr::<4>(&p.r, 2 * n + 1 + 4 * ci) else {
                continue;
            };
            let x = vaddq_s32(lanes, vdupq_n_s32(4 * ci as i32));
            let l = vmulq_s32(vsubq_s32(vnm1, x), vl);
            let r = vmulq_s32(vaddq_s32(x, one), vtr);
            let top = vmulq_s32(wy, ld_i32x4(t));
            let s = vaddq_s32(vaddq_s32(l, r), vaddq_s32(top, k));
            st_i32x4(dst, vshlq_s32(s, shift));
        }
    }
}

/// out[a][b] = ((32 - f) * ref[b + i + 1] + f * ref[b + i + 2] + 16) >> 5 (ref[x] at x + 32,
/// i = (a + 1) * angle >> 5, f = (a + 1) * angle & 31). For f = 0 this is ref[b + i + 1], the
/// scalar's special case.
#[target_feature(enable = "neon")]
fn angular(rf: &[i32; RF_LEN], angle: i32, n: usize, out: &mut Block) {
    let rnd = vdupq_n_s32(16);
    for (a, row) in out.iter_mut().enumerate().take(n) {
        let pos = (a as i32 + 1) * angle;
        let (idx, f) = (pos >> 5, pos & 31);
        let (wa, wb) = (vdupq_n_s32(32 - f), vdupq_n_s32(f));
        for (ci, c) in row.chunks_exact_mut(4).enumerate().take(n / 4) {
            let Ok(dst) = <&mut [i32; 4]>::try_from(c) else {
                continue;
            };
            let Ok(base) = usize::try_from(32 + 4 * ci as i32 + idx + 1) else {
                continue;
            };
            let (Some(x), Some(y)) = (arr::<4>(rf, base), arr::<4>(rf, base + 1)) else {
                continue;
            };
            let s = vaddq_s32(
                vaddq_s32(vmulq_s32(wa, ld_i32x4(x)), vmulq_s32(wb, ld_i32x4(y))),
                rnd,
            );
            st_i32x4(dst, vshrq_n_s32::<5>(s));
        }
    }
}

/// Clamp to 0..=max and narrow to u16, 4 samples at a time.
#[target_feature(enable = "neon")]
fn store(block: &Block, out: &mut Pred, n: usize, max: i32) {
    let (lo, hi) = (vdupq_n_s32(0), vdupq_n_s32(max));
    for (src, dst) in block.iter().zip(out.iter_mut()).take(n) {
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)).take(n / 4) {
            let (Ok(s), Ok(d)) = (<&[i32; 4]>::try_from(s), <&mut [u16; 4]>::try_from(d)) else {
                continue;
            };
            let v = vminq_s32(vmaxq_s32(ld_i32x4(s), lo), hi);
            st_u16x4(d, vqmovun_s32(v));
        }
    }
}
