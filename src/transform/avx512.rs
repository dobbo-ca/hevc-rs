//! AVX-512 kernels of the transform family (docs/safety.md). Safe `target_feature` fns: the
//! arrays come from slices in safe code and the loads and stores go through `simd::x86`.
//!
//! Same two passes as `avx2.rs` (output row `a` is `sum_j src[j][a] * M[j][..]`, rows and
//! columns past the non-zero extent are skipped), with the work cut instead of just widened:
//!
//! - `vpdpwssd` (VNNI) does the 16-bit multiply, the pair add and the accumulate in one op.
//! - 4x4 and 8x8: one zmm holds 16 outputs (a whole 4x4 block, or two rows of an 8x8 block).
//!   The operand with the input pair of each output row is made by one `vpermw` from the
//!   source rows, so there is no scalar gather and no broadcast. The 4x4 intermediate stays in
//!   a register between the passes.
//! - 16x16 and 32x32: the input pairs of all rows are interleaved once per pass (`vpermw` or
//!   `vpunpck*`) into a small i32 scratch, then every pair is one broadcast load.
//! - Narrowing is `vpmovsdw` (saturating, the i16 clip between the passes) and `vpmovusdw`.

use super::{DCT, DST, Dq, TuParams, pack};
use crate::simd::x86::*;
use core::arch::x86_64::*;

/// Pair `p` of the matrix of the n-point transform over `W` lanes: lane `l` has `M[2p][l % n]`
/// in the low and `M[2p + 1][l % n]` in the high 16 bits (n divides W, so a vector holds
/// `W / n` copies of the row: the 4x4 and 8x8 passes use them for 4 and 2 output rows).
#[allow(clippy::indexing_slicing)]
const fn tab<const P: usize, const W: usize>(n: usize, dst: bool) -> [[i32; W]; P] {
    let stride = 32 / n;
    let mut t = [[0i32; W]; P];
    let mut p = 0;
    while p < P {
        let mut l = 0;
        while l < W {
            let x = l % n;
            let (a, b) = if dst {
                (DST[2 * p][x], DST[2 * p + 1][x])
            } else {
                (DCT[2 * p * stride][x], DCT[(2 * p + 1) * stride][x])
            };
            t[p][l] = (a & 0xffff) | (b << 16);
            l += 1;
        }
        p += 1;
    }
    t
}

static T4: [[i32; 16]; 2] = tab::<2, 16>(4, false);
static TD4: [[i32; 16]; 2] = tab::<2, 16>(4, true);
static T8: [[i32; 16]; 4] = tab::<4, 16>(8, false);
static T16: [[i32; 16]; 8] = tab::<8, 16>(16, false);
static T32: [[i32; 32]; 16] = tab::<16, 32>(32, false);

/// `vpermw` indices of the 4x4 pass: the source is the 16 entries of the block (row-major);
/// i32 lane `a * 4 + x` of pair `p` gets entries `(2p) * 4 + a` and `(2p + 1) * 4 + a`, i.e.
/// the input column `a` of rows `2p` and `2p + 1`, for every output column `x`.
#[allow(clippy::indexing_slicing)]
const fn idx4() -> [[i16; 32]; 2] {
    let mut t = [[0i16; 32]; 2];
    let mut p = 0;
    while p < 2 {
        let mut j = 0;
        while j < 16 {
            t[p][2 * j] = (8 * p + j / 4) as i16;
            t[p][2 * j + 1] = (8 * p + 4 + j / 4) as i16;
            j += 1;
        }
        p += 1;
    }
    t
}

/// `vpermw` indices of the 8x8 pass: the source is rows `2p` and `2p + 1` (16 entries); the
/// i32 lanes `0..8` of group `h` get the input column `2h` of the two rows, the lanes `8..16`
/// the column `2h + 1`.
#[allow(clippy::indexing_slicing)]
const fn idx8() -> [[i16; 32]; 4] {
    let mut t = [[0i16; 32]; 4];
    let mut h = 0;
    while h < 4 {
        let mut j = 0;
        while j < 16 {
            let a = 2 * h + j / 8;
            t[h][2 * j] = a as i16;
            t[h][2 * j + 1] = (8 + a) as i16;
            j += 1;
        }
        h += 1;
    }
    t
}

/// `vpermw` indices that interleave two rows of 16 entries (one 32-entry source): i32 lane `a`
/// gets entry `a` and entry `16 + a`.
#[allow(clippy::indexing_slicing)]
const fn idxw() -> [i16; 32] {
    let mut t = [0i16; 32];
    let mut a = 0;
    while a < 16 {
        t[2 * a] = a as i16;
        t[2 * a + 1] = (16 + a) as i16;
        a += 1;
    }
    t
}

static IDX4: [[i16; 32]; 2] = idx4();
static IDX8: [[i16; 32]; 4] = idx8();
static IDXW: [i16; 32] = idxw();

/// Scaling (in place, of the chunks that cover the extent `ext` of the non-zero coefficients),
/// transform (or transform skip) and rounding of the block into `res`. `dq` is None when the
/// coefficients are already scaled. `coeffs` and `res` have n * n entries, `ext` is at most
/// (n, n); `p` is validated.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn dq16(v: __m512i, ms: __m512i, rnd: __m512i, sh: __m128i) -> __m512i {
    let d = _mm512_sra_epi32(_mm512_add_epi32(_mm512_mullo_epi32(v, ms), rnd), sh);
    _mm512_min_epi32(
        _mm512_max_epi32(d, _mm512_set1_epi32(-32768)),
        _mm512_set1_epi32(32767),
    )
}

/// 8.6.3 in i32 on the chunks of 16 entries that cover the first `rows` rows and `cols`
/// columns (a chunk is a whole 4x4 block, two rows of an 8x8 block, or half or all of a row
/// of a larger one); the caller has checked that no product overflows (`dq_for`). A zero
/// coefficient scales to zero, so covering more than the extent is harmless.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn dequant(coeffs: &mut [i32], n: usize, (rows, cols): (usize, usize), dq: &Dq<'_>) {
    let rnd = _mm512_set1_epi32(1 << (dq.bd_shift - 1));
    let sh = _mm_cvtsi32_si128(dq.bd_shift as i32);
    let flat = _mm512_set1_epi32(dq.scale * 16);
    let scale = _mm512_set1_epi32(dq.scale);
    let mat = dq.mat.map(|m| m.as_chunks::<16>().0);
    // Chunks per row, and the rows and chunks of a row that cover the extent. Rows of a block
    // narrower than 16 share chunks: one "row" of the first chunks that hold `rows` rows.
    let (per_row, nrows, ncols) = if n < 16 {
        (0, 1, (rows * n).div_ceil(16))
    } else {
        (n / 16, rows, cols.div_ceil(16))
    };
    let chunks = coeffs.as_chunks_mut::<16>().0;
    for r in 0..nrows {
        for (k, ch) in chunks.iter_mut().enumerate().skip(r * per_row).take(ncols) {
            let ms = match mat {
                None => flat,
                Some(m) => match m.get(k) {
                    Some(f) => _mm512_mullo_epi32(_mm512_cvtepu8_epi32(ld_u8x16(f)), scale),
                    None => continue,
                },
            };
            st_i32x16(ch, dq16(ld_i32x16(ch), ms, rnd, sh));
        }
    }
}

/// Transform skip (8-298) and the bdShift rounding: `res = (c << ts + round) >> bd2`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn ts_round(coeffs: &[i32], res: &mut [i32], ts: u32, bd2: u32) {
    let l = _mm_cvtsi32_si128(ts as i32);
    let r = _mm_cvtsi32_si128(bd2 as i32);
    let rnd = _mm512_set1_epi32(1 << (bd2 - 1));
    for (ch, out) in coeffs
        .as_chunks::<16>()
        .0
        .iter()
        .zip(res.as_chunks_mut::<16>().0)
    {
        let v = _mm512_sll_epi32(ld_i32x16(ch), l);
        st_i32x16(out, _mm512_sra_epi32(_mm512_add_epi32(v, rnd), r));
    }
}

/// Both 1-D passes (8.6.4.1) and the final bdShift rounding, into `res`. The i16 scratch
/// arrays hold n * n entries, so a 4x4 block zeroes 32 bytes and not 2 KiB.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn transform(coeffs: &[i32], ext: (usize, usize), p: &TuParams, bd2: u32, res: &mut [i32]) {
    let n = 1usize << p.log2_size;
    match p.log2_size {
        2 => {
            let mut c16 = [0i16; 16];
            let ext = pack(coeffs, n, ext, &mut c16);
            run4(if p.dst { &TD4 } else { &T4 }, &c16, res, ext, bd2);
        }
        3 => {
            let mut c16 = [0i16; 64];
            let mut ht = [0i16; 64];
            let ext = pack(coeffs, n, ext, &mut c16);
            run8(&T8, &c16, &mut ht, res, ext, bd2);
        }
        4 => {
            let mut c16 = [0i16; 256];
            let mut ht = [0i16; 256];
            let mut scr = [0i32; 128];
            let ext = pack(coeffs, n, ext, &mut c16);
            run_wide(&T16, &c16, &mut ht, &mut scr, res, n, ext, bd2);
        }
        _ => {
            let mut c16 = [0i16; 1024];
            let mut ht = [0i16; 1024];
            let mut scr = [0i32; 512];
            let ext = pack(coeffs, n, ext, &mut c16);
            run_wide(&T32, &c16, &mut ht, &mut scr, res, n, ext, bd2);
        }
    }
}

/// One 4x4 pass over the first `pairs` pairs of source rows: i32 lane `a * 4 + x` is
/// `sum_j src[j][a] * M[j][x]`, so the whole output block is one vector.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn pass4(src: __m256i, tab: &[[i32; 16]; 2], pairs: usize) -> __m512i {
    let x = _mm512_castsi256_si512(src);
    let mut acc = _mm512_setzero_si512();
    for (ix, t) in IDX4.iter().zip(tab).take(pairs) {
        let v = _mm512_permutexvar_epi16(ld_i16x32(ix), x);
        acc = _mm512_dpwssd_epi32(acc, v, ld_i32x16(t));
    }
    acc
}

/// 4x4: `ext` is the (rows, cols) extent of the non-zero coefficients in `c16`. The
/// intermediate (row `x` is column `x` of the pass 1 output, clipped to i16 by the saturating
/// narrowing) is already in the layout pass 2 reads, and stays in a register.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn run4(
    tab: &[[i32; 16]; 2],
    c16: &[i16; 16],
    res: &mut [i32],
    (rows, cols): (usize, usize),
    bd2: u32,
) {
    let v = _mm512_add_epi32(
        pass4(ld_i16x16(c16), tab, rows.div_ceil(2)),
        _mm512_set1_epi32(64),
    );
    let ht = _mm512_cvtsepi32_epi16(_mm512_srai_epi32::<7>(v));
    let rnd = _mm512_set1_epi32(1 << (bd2 - 1));
    let sh = _mm_cvtsi32_si128(bd2 as i32);
    let v = _mm512_add_epi32(pass4(ht, tab, cols.div_ceil(2)), rnd);
    if let Some(o) = res.first_chunk_mut::<16>() {
        st_i32x16(o, _mm512_sra_epi32(v, sh));
    }
}

/// One 8x8 pass over the first `pairs` pairs of source rows (`src` is 8 x 8, row-major):
/// vector `h` of the result has the output rows `2h` (lanes 0..8) and `2h + 1` (lanes 8..16);
/// only the first `groups` vectors are computed, the others stay 0.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn pass8(src: &[i16], tab: &[[i32; 16]; 4], pairs: usize, groups: usize) -> [__m512i; 4] {
    let mut accs = [_mm512_setzero_si512(); 4];
    for (s, t) in src.as_chunks::<16>().0.iter().zip(tab).take(pairs) {
        let x = _mm512_castsi256_si512(ld_i16x16(s));
        let t = ld_i32x16(t);
        for (acc, ix) in accs.iter_mut().zip(IDX8.iter()).take(groups) {
            let v = _mm512_permutexvar_epi16(ld_i16x32(ix), x);
            *acc = _mm512_dpwssd_epi32(*acc, v, t);
        }
    }
    accs
}

/// 8x8: pass 1 transforms the first `cols` columns from their first `rows` inputs (row `a` of
/// `ht`, which is all zero on entry, is column `a` of the intermediate), pass 2 every row from
/// its first `cols` inputs. A odd `cols` also stores the next column of the intermediate,
/// which is computed from zero inputs and is 0.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn run8(
    tab: &[[i32; 16]; 4],
    c16: &[i16],
    ht: &mut [i16],
    res: &mut [i32],
    (rows, cols): (usize, usize),
    bd2: u32,
) {
    let groups = cols.div_ceil(2);
    let r64 = _mm512_set1_epi32(64);
    let accs = pass8(c16, tab, rows.div_ceil(2), groups);
    for (o, v) in ht.as_chunks_mut::<16>().0.iter_mut().zip(accs).take(groups) {
        let v = _mm512_srai_epi32::<7>(_mm512_add_epi32(v, r64));
        st_i16x16(o, _mm512_cvtsepi32_epi16(v));
    }
    let rnd = _mm512_set1_epi32(1 << (bd2 - 1));
    let sh = _mm_cvtsi32_si128(bd2 as i32);
    let accs = pass8(ht, tab, groups, 4);
    for (o, v) in res.as_chunks_mut::<16>().0.iter_mut().zip(accs) {
        st_i32x16(o, _mm512_sra_epi32(_mm512_add_epi32(v, rnd), sh));
    }
}

/// Interleaves the first `pairs` pairs of rows of a 16 x 16 `src` into `scr`: `scr[p * 16 + a]`
/// is `src[2p][a]` in the low and `src[2p + 1][a]` in the high 16 bits.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn weave16(src: &[i16], pairs: usize, scr: &mut [i32]) {
    let ix = ld_i16x32(&IDXW);
    for (s, o) in src
        .as_chunks::<32>()
        .0
        .iter()
        .zip(scr.as_chunks_mut::<16>().0)
        .take(pairs)
    {
        st_i32x16(o, _mm512_permutexvar_epi16(ix, ld_i16x32(s)));
    }
}

/// [`weave16`] for 32 x 32: `scr[p * 32 + pos(a)]`, with the order of `vpunpck*` (see `pos32`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn weave32(src: &[i16], pairs: usize, scr: &mut [i32]) {
    for (r, o) in src
        .as_chunks::<32>()
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .zip(scr.as_chunks_mut::<32>().0)
        .take(pairs)
    {
        let [r0, r1] = r;
        let halves = o.as_chunks_mut::<16>().0;
        let [lo, hi] = halves else {
            continue;
        };
        let (a, b) = (ld_i16x32(r0), ld_i16x32(r1));
        st_i32x16(lo, _mm512_unpacklo_epi16(a, b));
        st_i32x16(hi, _mm512_unpackhi_epi16(a, b));
    }
}

/// [`weave16`] or [`weave32`] by the size `n`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn weave(src: &[i16], n: usize, pairs: usize, scr: &mut [i32]) {
    if n == 16 {
        weave16(src, pairs, scr);
    } else {
        weave32(src, pairs, scr);
    }
}

/// Where [`weave32`] puts column `a`: `unpacklo` of each 128-bit lane holds columns 0..4 of the
/// lane's 8 (at i32 lane `4 * lane`), `unpackhi` columns 4..8 (in the second half of the pair).
fn pos32(a: usize) -> usize {
    ((a >> 2) & 1) * 16 + (a >> 3) * 4 + (a & 3)
}

/// `sum_j src[j][a] * M[j][..]` for the first `pairs` pairs of source rows, from their
/// interleaved form in `scr`, as two vectors of 16 columns; the second stays 0 for n = 16.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn row_acc<const W: usize>(
    scr: &[i32],
    n: usize,
    a: usize,
    tab: &[[i32; W]],
    pairs: usize,
) -> [__m512i; 2] {
    let pa = if n == 32 { pos32(a) } else { a };
    let mut accs = [_mm512_setzero_si512(); 2];
    for (p, t) in tab.iter().enumerate().take(pairs) {
        let w = scr.get(p * n + pa).copied().unwrap_or(0);
        if w == 0 {
            continue;
        }
        let v = _mm512_set1_epi32(w);
        for (acc, c) in accs.iter_mut().zip(t.as_chunks::<16>().0) {
            *acc = _mm512_dpwssd_epi32(*acc, v, ld_i32x16(c));
        }
    }
    accs
}

/// 16x16 and 32x32 (`W` = n): as `run` of `avx2.rs`, with `scr` (n / 2 * n entries) holding the
/// interleaved source pairs of the pass.
#[inline]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn run_wide<const W: usize>(
    tab: &[[i32; W]],
    c16: &[i16],
    ht: &mut [i16],
    scr: &mut [i32],
    res: &mut [i32],
    n: usize,
    (rows, cols): (usize, usize),
    bd2: u32,
) {
    // Pass 1 (columns): row `a` of `ht` is column `a` of the intermediate, clipped to i16 by
    // the saturating narrowing. Columns past `cols` are 0.
    weave(c16, n, rows.div_ceil(2), scr);
    let r64 = _mm512_set1_epi32(64);
    for (a, out) in ht.chunks_exact_mut(n).enumerate().take(cols) {
        let v = row_acc(scr, n, a, tab, rows.div_ceil(2));
        for (o, x) in out.as_chunks_mut::<16>().0.iter_mut().zip(v) {
            let x = _mm512_srai_epi32::<7>(_mm512_add_epi32(x, r64));
            st_i16x16(o, _mm512_cvtsepi32_epi16(x));
        }
    }
    // Pass 2 (rows) and the rounding of 8-299.
    weave(ht, n, cols.div_ceil(2), scr);
    let rnd = _mm512_set1_epi32(1 << (bd2 - 1));
    let sh = _mm_cvtsi32_si128(bd2 as i32);
    for (a, out) in res.chunks_exact_mut(n).enumerate() {
        let v = row_acc(scr, n, a, tab, cols.div_ceil(2));
        for (o, x) in out.as_chunks_mut::<16>().0.iter_mut().zip(v) {
            st_i32x16(o, _mm512_sra_epi32(_mm512_add_epi32(x, rnd), sh));
        }
    }
}

/// 8.6.7 for one row, 16 lanes per op and the clip to [0, max] done by the unsigned saturating
/// narrowing (`vpmovusdw`, to 65535) and one `min_epu16`. The residual is clamped to at most
/// 65536 first so that the i32 sum cannot overflow; a very negative residual cannot overflow
/// (the prediction is not negative) and the sum is clipped to 0 whatever its value.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) fn add_row(row: &mut [u16], res: &[i32], max: i32) {
    let mx = _mm_set1_epi16(max as i16);
    match row.len() {
        4 => {
            if let (Some(p), Some(r)) = (row.first_chunk_mut::<4>(), res.first_chunk::<4>()) {
                let r = _mm_min_epi32(ld_i32x4(r), _mm_set1_epi32(65536));
                let v = _mm_add_epi32(_mm_cvtepu16_epi32(ld_u16x4(p)), r);
                let v = _mm_max_epi32(v, _mm_setzero_si128());
                st_u16x4(p, _mm_min_epu16(_mm_cvtusepi32_epi16(v), mx));
            }
        }
        8 => {
            if let (Some(p), Some(r)) = (row.first_chunk_mut::<8>(), res.first_chunk::<8>()) {
                let r = _mm256_min_epi32(ld_i32x8(r), _mm256_set1_epi32(65536));
                let v = _mm256_add_epi32(_mm256_cvtepu16_epi32(ld_u16x8(p)), r);
                let v = _mm256_max_epi32(v, _mm256_setzero_si256());
                st_u16x8(p, _mm_min_epu16(_mm256_cvtusepi32_epi16(v), mx));
            }
        }
        _ => {
            let mx = _mm256_set1_epi16(max as i16);
            for (p, r) in row
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(res.as_chunks::<16>().0)
            {
                let r = _mm512_min_epi32(ld_i32x16(r), _mm512_set1_epi32(65536));
                let v = _mm512_add_epi32(_mm512_cvtepu16_epi32(ld_u16x16(p)), r);
                let v = _mm512_max_epi32(v, _mm512_setzero_si512());
                st_u16x16(p, _mm256_min_epu16(_mm512_cvtusepi32_epi16(v), mx));
            }
        }
    }
}
