//! AVX2 kernels of the transform family (docs/safety.md). Safe `target_feature` fns: the arrays
//! come from slices in safe code and the loads and stores go through `simd::x86`.
//!
//! Both transform passes are one routine: output row `a` is `sum_j src[j][a] * M[j][..]`, i.e.
//! the input column `a` broadcast against the matrix rows. Pass 1 reads the coefficients and
//! writes the transposed intermediate, pass 2 reads that and writes the rows of the residual.
//! Pairs of rows are multiplied with `madd_epi16` (inputs and matrix entries fit i16, sums fit
//! i32), and rows and columns past the non-zero extent are skipped.

use super::{DCT, DST, Dq, TuParams, pack};
use crate::simd::x86::*;
use core::arch::x86_64::*;

/// `[j]` pair `p`, column `x`: `M[2p][x]` in the low and `M[2p + 1][x]` in the high 16 bits.
/// `n` is the transform size, `W = max(n, 8)` (the columns past `n` stay 0).
#[allow(clippy::indexing_slicing)]
const fn pk<const P: usize, const W: usize>(n: usize, dst: bool) -> [[i32; W]; P] {
    let stride = 32 / n;
    let mut t = [[0i32; W]; P];
    let mut p = 0;
    while p < P {
        let mut x = 0;
        while x < n {
            let (a, b) = if dst {
                (DST[2 * p][x], DST[2 * p + 1][x])
            } else {
                (DCT[2 * p * stride][x], DCT[(2 * p + 1) * stride][x])
            };
            t[p][x] = (a & 0xffff) | (b << 16);
            x += 1;
        }
        p += 1;
    }
    t
}

static PK_DST: [[i32; 8]; 2] = pk::<2, 8>(4, true);
static PK4: [[i32; 8]; 2] = pk::<2, 8>(4, false);
static PK8: [[i32; 8]; 4] = pk::<4, 8>(8, false);
static PK16: [[i32; 16]; 8] = pk::<8, 16>(16, false);
static PK32: [[i32; 32]; 16] = pk::<16, 32>(32, false);

/// Scaling, transform (or transform skip) and rounding of the block, in place. `dq` is None when
/// the coefficients are already scaled. `coeffs` has n * n entries; `p` is validated.
#[target_feature(enable = "avx2")]
pub(super) fn residual(coeffs: &mut [i32], dq: Option<Dq<'_>>, p: &TuParams) {
    if let Some(dq) = dq {
        dequant(coeffs, &dq);
    }
    let bd2 = 20 - p.bit_depth;
    if p.transform_skip {
        ts_round(coeffs, 5 + p.log2_size, bd2);
    } else {
        transform(coeffs, p, bd2);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dq8(v: __m256i, ms: __m256i, rnd: __m256i, sh: __m128i) -> __m256i {
    let d = _mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(v, ms), rnd), sh);
    _mm256_min_epi32(
        _mm256_max_epi32(d, _mm256_set1_epi32(-32768)),
        _mm256_set1_epi32(32767),
    )
}

/// 8.6.3 in i32; the caller has checked that no product overflows (`dq_for`).
#[inline]
#[target_feature(enable = "avx2")]
fn dequant(coeffs: &mut [i32], dq: &Dq<'_>) {
    let rnd = _mm256_set1_epi32(1 << (dq.bd_shift - 1));
    let sh = _mm_cvtsi32_si128(dq.bd_shift as i32);
    match dq.mat {
        None => {
            let ms = _mm256_set1_epi32(dq.scale * 16);
            for ch in coeffs.as_chunks_mut::<8>().0 {
                st_i32x8(ch, dq8(ld_i32x8(ch), ms, rnd, sh));
            }
        }
        Some(m) => {
            let scale = _mm256_set1_epi32(dq.scale);
            for (cc, mm) in coeffs
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(m.as_chunks::<16>().0)
            {
                let f = ld_u8x16(mm);
                let fs = [
                    _mm256_cvtepu8_epi32(f),
                    _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(f)),
                ];
                for (ch, f) in cc.as_chunks_mut::<8>().0.iter_mut().zip(fs) {
                    let ms = _mm256_mullo_epi32(f, scale);
                    st_i32x8(ch, dq8(ld_i32x8(ch), ms, rnd, sh));
                }
            }
        }
    }
}

/// Transform skip (8-298) and the bdShift rounding: `(c << ts + round) >> bd2`.
#[inline]
#[target_feature(enable = "avx2")]
fn ts_round(coeffs: &mut [i32], ts: u32, bd2: u32) {
    let l = _mm_cvtsi32_si128(ts as i32);
    let r = _mm_cvtsi32_si128(bd2 as i32);
    let rnd = _mm256_set1_epi32(1 << (bd2 - 1));
    for ch in coeffs.as_chunks_mut::<8>().0 {
        let v = _mm256_sll_epi32(ld_i32x8(ch), l);
        st_i32x8(ch, _mm256_sra_epi32(_mm256_add_epi32(v, rnd), r));
    }
}

/// Eight i32 lanes of `a` then of `b` as 16 i16 in order, saturated.
#[inline]
#[target_feature(enable = "avx2")]
fn pack16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a, b))
}

/// `sum_j src[j][a] * M[j][..]` for the first `pairs` pairs of source rows (`src` is n x n,
/// row-major), as up to four vectors of eight columns; unused vectors stay 0.
#[inline]
#[target_feature(enable = "avx2")]
fn row_acc<const W: usize>(
    src: &[i16],
    n: usize,
    a: usize,
    tab: &[[i32; W]],
    pairs: usize,
) -> [__m256i; 4] {
    let mut accs = [_mm256_setzero_si256(); 4];
    let mut col = src.iter().skip(a).step_by(n).copied();
    for t in tab.iter().take(pairs) {
        let lo = col.next().unwrap_or(0);
        let hi = col.next().unwrap_or(0);
        if lo == 0 && hi == 0 {
            continue;
        }
        let v = _mm256_set1_epi32(i32::from(lo.cast_unsigned()) | (i32::from(hi) << 16));
        for (acc, c) in accs.iter_mut().zip(t.as_chunks::<8>().0) {
            *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(v, ld_i32x8(c)));
        }
    }
    accs
}

/// Stores the lanes of `v` (saturated to i16) as one row of n entries.
#[inline]
#[target_feature(enable = "avx2")]
fn store_i16(out: &mut [i16], v: &[__m256i; 4]) {
    let p0 = pack16(v[0], v[1]);
    match out.len() {
        4 => {
            if let Some(o) = out.first_chunk_mut::<4>() {
                st_i16x4(o, _mm256_castsi256_si128(p0));
            }
        }
        8 => {
            if let Some(o) = out.first_chunk_mut::<8>() {
                st_i16x8(o, _mm256_castsi256_si128(p0));
            }
        }
        _ => {
            let pairs = [p0, pack16(v[2], v[3])];
            for (o, q) in out.as_chunks_mut::<16>().0.iter_mut().zip(pairs) {
                st_i16x16(o, q);
            }
        }
    }
}

/// Stores the lanes of `v` as one row of n i32 entries.
#[inline]
#[target_feature(enable = "avx2")]
fn store_i32(out: &mut [i32], v: &[__m256i; 4]) {
    if out.len() == 4 {
        if let Some(o) = out.first_chunk_mut::<4>() {
            st_i32x4(o, _mm256_castsi256_si128(v[0]));
        }
        return;
    }
    for (o, x) in out.as_chunks_mut::<8>().0.iter_mut().zip(v.iter()) {
        st_i32x8(o, *x);
    }
}

/// Both 1-D passes (8.6.4.1) and the final bdShift rounding.
#[inline]
#[target_feature(enable = "avx2")]
fn transform(coeffs: &mut [i32], p: &TuParams, bd2: u32) {
    match p.log2_size {
        2 => sized::<16>(coeffs, p, bd2),
        3 => sized::<64>(coeffs, p, bd2),
        4 => sized::<256>(coeffs, p, bd2),
        _ => sized::<1024>(coeffs, p, bd2),
    }
}

/// [`transform`] with scratch arrays of `NN` (= n * n) entries, so a 4x4 block zeroes 64
/// bytes and not 4 KiB.
#[inline]
#[target_feature(enable = "avx2")]
fn sized<const NN: usize>(coeffs: &mut [i32], p: &TuParams, bd2: u32) {
    let n = 1usize << p.log2_size;
    let mut c16 = [0i16; NN];
    let mut ht = [0i16; NN];
    let (Some(c16), Some(ht)) = (c16.get_mut(..n * n), ht.get_mut(..n * n)) else {
        return;
    };
    let ext = pack(coeffs, n, c16);
    match (n, p.dst) {
        (4, true) => run::<8>(&PK_DST, c16, ht, coeffs, n, ext, bd2),
        (4, false) => run::<8>(&PK4, c16, ht, coeffs, n, ext, bd2),
        (8, _) => run::<8>(&PK8, c16, ht, coeffs, n, ext, bd2),
        (16, _) => run::<16>(&PK16, c16, ht, coeffs, n, ext, bd2),
        _ => run::<32>(&PK32, c16, ht, coeffs, n, ext, bd2),
    }
}

/// `ext` is the (rows, cols) extent of the non-zero coefficients in `c16`; `ht` is all zero.
#[inline]
#[target_feature(enable = "avx2")]
fn run<const W: usize>(
    tab: &[[i32; W]],
    c16: &[i16],
    ht: &mut [i16],
    coeffs: &mut [i32],
    n: usize,
    ext: (usize, usize),
    bd2: u32,
) {
    let (rows, cols) = ext;
    // Pass 1 (columns, 8.6.4.1 steps 1 and 2): row x of `ht` is column x of the intermediate,
    // clipped to i16 by the saturating pack. Columns past `cols` are 0.
    let r64 = _mm256_set1_epi32(64);
    for (a, out) in ht.chunks_exact_mut(n).enumerate().take(cols) {
        let mut v = row_acc(c16, n, a, tab, rows.div_ceil(2));
        for x in v.iter_mut() {
            *x = _mm256_srai_epi32::<7>(_mm256_add_epi32(*x, r64));
        }
        store_i16(out, &v);
    }
    // Pass 2 (rows, step 3) and the rounding of 8-299.
    let rnd = _mm256_set1_epi32(1 << (bd2 - 1));
    let sh = _mm_cvtsi32_si128(bd2 as i32);
    for (a, out) in coeffs.chunks_exact_mut(n).enumerate() {
        let mut v = row_acc(ht, n, a, tab, cols.div_ceil(2));
        for x in v.iter_mut() {
            *x = _mm256_sra_epi32(_mm256_add_epi32(*x, rnd), sh);
        }
        store_i32(out, &v);
    }
}

/// 8.6.7 for one row. The residual is clamped to +-65536 first (the result is the same, the
/// samples are at most 16 bits) so that the i32 sum cannot overflow.
#[target_feature(enable = "avx2")]
pub(super) fn add_row(row: &mut [u16], res: &[i32], max: i32) {
    if row.len() == 4 {
        if let (Some(p), Some(r)) = (row.first_chunk_mut::<4>(), res.first_chunk::<4>()) {
            let r = _mm_min_epi32(
                _mm_max_epi32(ld_i32x4(r), _mm_set1_epi32(-65536)),
                _mm_set1_epi32(65536),
            );
            let v = _mm_add_epi32(_mm_cvtepu16_epi32(ld_u16x4(p)), r);
            let v = _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), _mm_set1_epi32(max));
            st_u16x4(p, _mm_packus_epi32(v, v));
        }
        return;
    }
    let rl = _mm256_set1_epi32(-65536);
    let rh = _mm256_set1_epi32(65536);
    let zero = _mm256_setzero_si256();
    let mx = _mm256_set1_epi32(max);
    for (p, r) in row
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(res.as_chunks::<8>().0)
    {
        let r = _mm256_min_epi32(_mm256_max_epi32(ld_i32x8(r), rl), rh);
        let v = _mm256_add_epi32(_mm256_cvtepu16_epi32(ld_u16x8(p)), r);
        let v = _mm256_min_epi32(_mm256_max_epi32(v, zero), mx);
        let q = _mm256_permute4x64_epi64::<0x08>(_mm256_packus_epi32(v, v));
        st_u16x8(p, _mm256_castsi256_si128(q));
    }
}
