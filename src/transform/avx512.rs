//! AVX-512 kernels of the transform family (docs/safety.md), for the blocks where 512 bits pay:
//! 16x16 and 32x32 (a row of 16 or 32 i16 / 16 i32 accumulators in one zmm), and the residual
//! add of a 32-sample row. Smaller blocks and rows run the AVX2 kernels of `avx2.rs`, whose
//! algorithm and matrix tables these share. Safe `target_feature` fns; the loads and stores go
//! through `simd::x86`.

use super::avx2::{self, PK16, PK32};
use super::{Dq, TuParams, pack};
use crate::simd::x86::*;
use core::arch::x86_64::*;

/// Scaling, transform (or transform skip) and rounding of the block, in place; same contract as
/// `avx2::residual`.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(super) fn residual(coeffs: &mut [i32], dq: Option<Dq<'_>>, p: &TuParams) {
    if p.log2_size < 4 {
        avx2::residual(coeffs, dq, p);
        return;
    }
    if let Some(dq) = dq {
        dequant(coeffs, &dq);
    }
    let bd2 = 20 - p.bit_depth;
    if p.transform_skip {
        ts_round(coeffs, 5 + p.log2_size, bd2);
    } else if p.log2_size == 4 {
        sized::<256>(coeffs, p, bd2);
    } else {
        sized::<1024>(coeffs, p, bd2);
    }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn dq16(v: __m512i, ms: __m512i, rnd: __m512i, sh: __m128i) -> __m512i {
    let d = _mm512_sra_epi32(_mm512_add_epi32(_mm512_mullo_epi32(v, ms), rnd), sh);
    _mm512_min_epi32(
        _mm512_max_epi32(d, _mm512_set1_epi32(-32768)),
        _mm512_set1_epi32(32767),
    )
}

/// 8.6.3 in i32; the caller has checked that no product overflows (`dq_for`). n * n is a
/// multiple of 16 here.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn dequant(coeffs: &mut [i32], dq: &Dq<'_>) {
    let rnd = _mm512_set1_epi32(1 << (dq.bd_shift - 1));
    let sh = _mm_cvtsi32_si128(dq.bd_shift as i32);
    match dq.mat {
        None => {
            let ms = _mm512_set1_epi32(dq.scale * 16);
            for ch in coeffs.as_chunks_mut::<16>().0 {
                st_i32x16(ch, dq16(ld_i32x16(ch), ms, rnd, sh));
            }
        }
        Some(m) => {
            let scale = _mm512_set1_epi32(dq.scale);
            for (ch, mm) in coeffs
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(m.as_chunks::<16>().0)
            {
                let ms = _mm512_mullo_epi32(_mm512_cvtepu8_epi32(ld_u8x16(mm)), scale);
                st_i32x16(ch, dq16(ld_i32x16(ch), ms, rnd, sh));
            }
        }
    }
}

/// Transform skip (8-298) and the bdShift rounding: `(c << ts + round) >> bd2`.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn ts_round(coeffs: &mut [i32], ts: u32, bd2: u32) {
    let l = _mm_cvtsi32_si128(ts as i32);
    let r = _mm_cvtsi32_si128(bd2 as i32);
    let rnd = _mm512_set1_epi32(1 << (bd2 - 1));
    for ch in coeffs.as_chunks_mut::<16>().0 {
        let v = _mm512_sll_epi32(ld_i32x16(ch), l);
        st_i32x16(ch, _mm512_sra_epi32(_mm512_add_epi32(v, rnd), r));
    }
}

/// Sixteen i32 lanes of `a` then of `b` as 32 i16 in order, saturated (`packs` works per 128-bit
/// lane, the permute puts the qwords back in order).
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn pack32(a: __m512i, b: __m512i) -> __m512i {
    let idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(idx, _mm512_packs_epi32(a, b))
}

/// `sum_j src[j][a] * M[j][..]` for the first `pairs` pairs of source rows (`src` is n x n,
/// row-major), as up to two vectors of 16 columns; unused vectors stay 0.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn row_acc<const W: usize>(
    src: &[i16],
    n: usize,
    a: usize,
    tab: &[[i32; W]],
    pairs: usize,
) -> [__m512i; 2] {
    let mut accs = [_mm512_setzero_si512(); 2];
    let mut col = src.iter().skip(a).step_by(n).copied();
    for t in tab.iter().take(pairs) {
        let lo = col.next().unwrap_or(0);
        let hi = col.next().unwrap_or(0);
        if lo == 0 && hi == 0 {
            continue;
        }
        let v = _mm512_set1_epi32(i32::from(lo.cast_unsigned()) | (i32::from(hi) << 16));
        for (acc, c) in accs.iter_mut().zip(t.as_chunks::<16>().0) {
            *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(v, ld_i32x16(c)));
        }
    }
    accs
}

/// Stores the lanes of `v` (saturated to i16) as one row of n (16 or 32) entries.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn store_i16(out: &mut [i16], v: &[__m512i; 2]) {
    if out.len() == 16 {
        if let Some(o) = out.first_chunk_mut::<16>() {
            st_i16x16(o, _mm512_cvtsepi32_epi16(v[0]));
        }
    } else if let Some(o) = out.first_chunk_mut::<32>() {
        st_i16x32(o, pack32(v[0], v[1]));
    }
}

/// Stores the lanes of `v` as one row of n i32 entries.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn store_i32(out: &mut [i32], v: &[__m512i; 2]) {
    for (o, x) in out.as_chunks_mut::<16>().0.iter_mut().zip(v.iter()) {
        st_i32x16(o, *x);
    }
}

/// Both 1-D passes with scratch arrays of `NN` (= n * n) entries; n is 16 or 32.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
fn sized<const NN: usize>(coeffs: &mut [i32], p: &TuParams, bd2: u32) {
    let n = 1usize << p.log2_size;
    let mut c16 = [0i16; NN];
    let mut ht = [0i16; NN];
    let (Some(c16), Some(ht)) = (c16.get_mut(..n * n), ht.get_mut(..n * n)) else {
        return;
    };
    let ext = pack(coeffs, n, c16);
    if n == 16 {
        run::<16>(&PK16, c16, ht, coeffs, n, ext, bd2);
    } else {
        run::<32>(&PK32, c16, ht, coeffs, n, ext, bd2);
    }
}

/// `ext` is the (rows, cols) extent of the non-zero coefficients in `c16`; `ht` is all zero.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
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
    let r64 = _mm512_set1_epi32(64);
    for (a, out) in ht.chunks_exact_mut(n).enumerate().take(cols) {
        let mut v = row_acc(c16, n, a, tab, rows.div_ceil(2));
        for x in v.iter_mut() {
            *x = _mm512_srai_epi32::<7>(_mm512_add_epi32(*x, r64));
        }
        store_i16(out, &v);
    }
    // Pass 2 (rows, step 3) and the rounding of 8-299.
    let rnd = _mm512_set1_epi32(1 << (bd2 - 1));
    let sh = _mm_cvtsi32_si128(bd2 as i32);
    for (a, out) in coeffs.chunks_exact_mut(n).enumerate() {
        let mut v = row_acc(ht, n, a, tab, cols.div_ceil(2));
        for x in v.iter_mut() {
            *x = _mm512_sra_epi32(_mm512_add_epi32(*x, rnd), sh);
        }
        store_i32(out, &v);
    }
}

/// 8.6.7 for one row of 32 samples (other lengths run `avx2::add_row`). The residual is clamped
/// to +-65536 first, as in the AVX2 kernel.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
pub(super) fn add_row(row: &mut [u16], res: &[i32], max: i32) {
    if row.len() != 32 || res.len() != 32 {
        avx2::add_row(row, res, max);
        return;
    }
    let (Some(p), ([r0, r1], _)) = (row.first_chunk_mut::<32>(), res.as_chunks::<16>()) else {
        return;
    };
    let rl = _mm512_set1_epi32(-65536);
    let rh = _mm512_set1_epi32(65536);
    let zero = _mm512_setzero_si512();
    let mx = _mm512_set1_epi32(max);
    let px = ld_u16x32(p);
    let lo = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(px));
    let hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(px));
    let r0 = _mm512_min_epi32(_mm512_max_epi32(ld_i32x16(r0), rl), rh);
    let r1 = _mm512_min_epi32(_mm512_max_epi32(ld_i32x16(r1), rl), rh);
    let lo = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(lo, r0), zero), mx);
    let hi = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(hi, r1), zero), mx);
    // The lanes are in 0..=65535, so the unsigned saturating pack is exact.
    let idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    st_u16x32(
        p,
        _mm512_permutexvar_epi64(idx, _mm512_packus_epi32(lo, hi)),
    );
}
