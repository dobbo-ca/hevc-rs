//! NEON kernels of the transform family (docs/safety.md). Safe `target_feature` fns: the arrays
//! come from slices in safe code and the loads and stores go through `simd::arm`.
//!
//! Both transform passes are one routine: output row `a` is `sum_j src[j][a] * M[j][..]`, i.e.
//! the input column `a` broadcast against the matrix rows (`vmlal_n_s16`, i16 inputs, i32
//! accumulators). Pass 1 reads the coefficients and writes the transposed intermediate, pass 2
//! reads that and writes the rows of the residual. Rows and columns past the non-zero extent
//! are skipped.

use super::{DCT, DST, Dq, TuParams, pack};
use crate::simd::arm::*;
use core::arch::aarch64::*;

/// Row `j`, column `x` of the matrix `M[j][x]` of an n x n transform; `W = max(n, 8)` (the
/// columns past `n` stay 0).
#[allow(clippy::indexing_slicing)]
const fn mat<const N: usize, const W: usize>(dst: bool) -> [[i16; W]; N] {
    let stride = 32 / N;
    let mut t = [[0i16; W]; N];
    let mut j = 0;
    while j < N {
        let mut x = 0;
        while x < N {
            t[j][x] = (if dst { DST[j][x] } else { DCT[j * stride][x] }) as i16;
            x += 1;
        }
        j += 1;
    }
    t
}

static T_DST: [[i16; 8]; 4] = mat::<4, 8>(true);
static T4: [[i16; 8]; 4] = mat::<4, 8>(false);
static T8: [[i16; 8]; 8] = mat::<8, 8>(false);
static T16: [[i16; 16]; 16] = mat::<16, 16>(false);
static T32: [[i16; 32]; 32] = mat::<32, 32>(false);

/// Scaling (in place, of the chunks that cover the extent `ext` of the non-zero coefficients),
/// transform (or transform skip) and rounding of the block into `res`. `dq` is None when the
/// coefficients are already scaled. `coeffs` and `res` have n * n entries, `ext` is at most
/// (n, n); `p` is validated.
#[target_feature(enable = "neon")]
pub(super) fn residual(
    coeffs: &mut [i32],
    ext: (usize, usize),
    dq: Option<Dq<'_>>,
    p: &TuParams,
    res: &mut [i32],
) {
    let n = 1usize << p.log2_size;
    if let Some(dq) = dq {
        dequant(coeffs, n, ext, &dq);
    }
    let bd2 = 20 - p.bit_depth;
    if p.transform_skip {
        ts_round(coeffs, res, 5 + p.log2_size, bd2);
    } else {
        transform(coeffs, ext, p, bd2, res);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dq4(v: int32x4_t, ms: int32x4_t, rnd: int32x4_t, sh: int32x4_t) -> int32x4_t {
    let d = vshlq_s32(vaddq_s32(vmulq_s32(v, ms), rnd), sh);
    vmaxq_s32(vminq_s32(d, vdupq_n_s32(32767)), vdupq_n_s32(-32768))
}

/// 8.6.3 in i32 on the chunks of 8 entries that cover the first `rows` rows and `cols` columns
/// (a chunk is two rows of a 4x4 block); the caller has checked that no product overflows
/// (`dq_for`). A zero coefficient scales to zero, so covering more than the extent is harmless.
#[inline]
#[target_feature(enable = "neon")]
fn dequant(coeffs: &mut [i32], n: usize, (rows, cols): (usize, usize), dq: &Dq<'_>) {
    let rnd = vdupq_n_s32(1 << (dq.bd_shift - 1));
    let sh = vdupq_n_s32(-(dq.bd_shift as i32));
    let flat = vdupq_n_s32(dq.scale * 16);
    let scale = vdupq_n_s32(dq.scale);
    let mat = dq.mat.map(|m| m.as_chunks::<8>().0);
    // Chunks per row, and the rows and chunks of a row that cover the extent.
    let (per_row, nrows, ncols) = if n == 4 {
        (1, rows.div_ceil(2), 1)
    } else {
        (n / 8, rows, cols.div_ceil(8))
    };
    let chunks = coeffs.as_chunks_mut::<8>().0;
    for r in 0..nrows {
        for (k, ch) in chunks.iter_mut().enumerate().skip(r * per_row).take(ncols) {
            let ms = match mat {
                None => [flat, flat],
                Some(m) => match m.get(k) {
                    Some(f) => {
                        let f = vmovl_u8(ld_u8x8(f));
                        [
                            vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(f))), scale),
                            vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(f))), scale),
                        ]
                    }
                    None => continue,
                },
            };
            for (c4, ms) in ch.as_chunks_mut::<4>().0.iter_mut().zip(ms) {
                st_i32x4(c4, dq4(ld_i32x4(c4), ms, rnd, sh));
            }
        }
    }
}

/// Transform skip (8-298) and the bdShift rounding: `res = (c << ts + round) >> bd2`.
#[inline]
#[target_feature(enable = "neon")]
fn ts_round(coeffs: &[i32], res: &mut [i32], ts: u32, bd2: u32) {
    let l = vdupq_n_s32(ts as i32);
    let r = vdupq_n_s32(-(bd2 as i32));
    let rnd = vdupq_n_s32(1 << (bd2 - 1));
    for (ch, out) in coeffs
        .as_chunks::<4>()
        .0
        .iter()
        .zip(res.as_chunks_mut::<4>().0)
    {
        let v = vshlq_s32(ld_i32x4(ch), l);
        st_i32x4(out, vshlq_s32(vaddq_s32(v, rnd), r));
    }
}

/// `sum_j src[j][a] * M[j][..]` for the first `nj` source rows (`src` is n x n, row-major), as
/// up to eight vectors of four columns; unused vectors stay 0.
#[inline]
#[target_feature(enable = "neon")]
fn row_acc<const W: usize>(
    src: &[i16],
    n: usize,
    a: usize,
    tab: &[[i16; W]],
    nj: usize,
) -> [int32x4_t; 8] {
    let mut accs = [vdupq_n_s32(0); 8];
    for (&s, t) in src.iter().skip(a).step_by(n).zip(tab.iter().take(nj)) {
        if s == 0 {
            continue;
        }
        for (ac, c) in accs
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(t.as_chunks::<8>().0)
        {
            let v = ld_i16x8(c);
            let [lo, hi] = ac;
            *lo = vmlal_n_s16(*lo, vget_low_s16(v), s);
            *hi = vmlal_n_s16(*hi, vget_high_s16(v), s);
        }
    }
    accs
}

/// Stores the lanes of `v` ((x + 64) >> 7 saturated to i16) as one row of n entries.
#[inline]
#[target_feature(enable = "neon")]
fn store_i16(out: &mut [i16], v: &[int32x4_t; 8]) {
    if out.len() == 4 {
        if let Some(o) = out.first_chunk_mut::<4>() {
            st_i16x4(o, vqrshrn_n_s32::<7>(v[0]));
        }
        return;
    }
    for (o, q) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(v.as_chunks::<2>().0)
    {
        let [a, b] = q;
        st_i16x8(
            o,
            vcombine_s16(vqrshrn_n_s32::<7>(*a), vqrshrn_n_s32::<7>(*b)),
        );
    }
}

/// Stores the lanes of `v` as one row of n i32 entries.
#[inline]
#[target_feature(enable = "neon")]
fn store_i32(out: &mut [i32], v: &[int32x4_t; 8]) {
    for (o, x) in out.as_chunks_mut::<4>().0.iter_mut().zip(v.iter()) {
        st_i32x4(o, *x);
    }
}

/// Both 1-D passes (8.6.4.1) and the final bdShift rounding, into `res`.
#[inline]
#[target_feature(enable = "neon")]
fn transform(coeffs: &[i32], ext: (usize, usize), p: &TuParams, bd2: u32, res: &mut [i32]) {
    match p.log2_size {
        2 => sized::<16>(coeffs, ext, p, bd2, res),
        3 => sized::<64>(coeffs, ext, p, bd2, res),
        4 => sized::<256>(coeffs, ext, p, bd2, res),
        _ => sized::<1024>(coeffs, ext, p, bd2, res),
    }
}

/// [`transform`] with scratch arrays of `NN` (= n * n) entries, so a 4x4 block zeroes 64
/// bytes and not 4 KiB.
#[inline]
#[target_feature(enable = "neon")]
fn sized<const NN: usize>(
    coeffs: &[i32],
    ext: (usize, usize),
    p: &TuParams,
    bd2: u32,
    res: &mut [i32],
) {
    let n = 1usize << p.log2_size;
    let mut c16 = [0i16; NN];
    let mut ht = [0i16; NN];
    let (Some(c16), Some(ht)) = (c16.get_mut(..n * n), ht.get_mut(..n * n)) else {
        return;
    };
    let ext = pack(coeffs, n, ext, c16);
    match (n, p.dst) {
        (4, true) => run::<8>(&T_DST, c16, ht, res, n, ext, bd2),
        (4, false) => run::<8>(&T4, c16, ht, res, n, ext, bd2),
        (8, _) => run::<8>(&T8, c16, ht, res, n, ext, bd2),
        (16, _) => run::<16>(&T16, c16, ht, res, n, ext, bd2),
        _ => run::<32>(&T32, c16, ht, res, n, ext, bd2),
    }
}

/// `ext` is the (rows, cols) extent of the non-zero coefficients in `c16`; `ht` is all zero.
/// Pass 1 transforms the first `cols` columns from their first `rows` inputs, pass 2 every row
/// from its first `cols` inputs (the other columns of the intermediate are zero).
#[inline]
#[target_feature(enable = "neon")]
fn run<const W: usize>(
    tab: &[[i16; W]],
    c16: &[i16],
    ht: &mut [i16],
    res: &mut [i32],
    n: usize,
    ext: (usize, usize),
    bd2: u32,
) {
    let (rows, cols) = ext;
    // Pass 1 (columns, 8.6.4.1 steps 1 and 2): row x of `ht` is column x of the intermediate.
    // Columns past `cols` are 0.
    for (a, out) in ht.chunks_exact_mut(n).enumerate().take(cols) {
        let v = row_acc(c16, n, a, tab, rows);
        store_i16(out, &v);
    }
    // Pass 2 (rows, step 3) and the rounding of 8-299.
    let rnd = vdupq_n_s32(1 << (bd2 - 1));
    let sh = vdupq_n_s32(-(bd2 as i32));
    for (a, out) in res.chunks_exact_mut(n).enumerate() {
        let mut v = row_acc(ht, n, a, tab, cols);
        for x in v.iter_mut() {
            *x = vshlq_s32(vaddq_s32(*x, rnd), sh);
        }
        store_i32(out, &v);
    }
}

/// 8.6.7 for one row. The residual is clamped to +-65536 first (the result is the same, the
/// samples are at most 16 bits) so that the i32 sum cannot overflow.
#[target_feature(enable = "neon")]
pub(super) fn add_row(row: &mut [u16], res: &[i32], max: i32) {
    let rl = vdupq_n_s32(-65536);
    let rh = vdupq_n_s32(65536);
    let zero = vdupq_n_s32(0);
    let mx = vdupq_n_s32(max);
    for (p, r) in row
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(res.as_chunks::<4>().0)
    {
        let r = vminq_s32(vmaxq_s32(ld_i32x4(r), rl), rh);
        let px = vreinterpretq_s32_u32(vmovl_u16(ld_u16x4(p)));
        let v = vminq_s32(vmaxq_s32(vaddq_s32(px, r), zero), mx);
        st_u16x4(p, vqmovun_s32(v));
    }
}
