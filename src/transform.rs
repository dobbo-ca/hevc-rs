//! Scaling and transformation, clauses 8.6.2 to 8.6.4 (v1: no extended precision, no
//! transform-skip rotation, no RDPCM), plus the picture construction of 8.6.7.

use crate::bits::BitReader;
use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::ps::Active;
use crate::scaling::ScalingList;
use crate::simd::{self, Tier};

/// Inputs of one transform unit component that are not in the frame or the parameter sets.
pub(crate) struct TuParams {
    /// qP of 8.6.2: Qp'Y or Qp'Cb / Qp'Cr, i.e. including QpBdOffset.
    pub qp: i32,
    pub bit_depth: u32,
    pub transform_skip: bool,
    /// cu_transquant_bypass_flag.
    pub bypass: bool,
    /// trType 1: luma intra 4x4 (8.6.4.1).
    pub dst: bool,
    /// matrixId of Table 7-4 (3 * inter + cIdx).
    pub matrix_id: usize,
    pub log2_size: u32,
}

/// Equation 8-316: `DST[j][i]` multiplies x[j] into y[i].
const DST: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// Equations 8-318 to 8-321, one printed row per entry: `DCT[k][n]` is the weight of frequency k
/// at sample n (the spec's transMatrix[n][k]); an nTbS transform uses rows k * (32 / nTbS).
const DCT: [[i32; 32]; 32] = [
    [
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64,
    ],
    [
        90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46,
        -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
    ],
    [
        90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70,
        -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90,
    ],
    [
        90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78,
        88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90,
    ],
    [
        89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18,
        -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
    ],
    [
        88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90,
        -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88,
    ],
    [
        87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80,
        90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87,
    ],
    [
        85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4,
        -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85,
    ],
    [
        83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83,
        -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
    ],
    [
        82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67,
        90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82,
    ],
    [
        80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25,
        -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80,
    ],
    [
        78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54,
        -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78,
    ],
    [
        75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50,
        50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
    ],
    [
        73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88,
        46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73,
    ],
    [
        70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90,
        -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70,
    ],
    [
        67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82,
        31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67,
    ],
    [
        64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    ],
    [
        61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38,
        -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61,
    ],
    [
        57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9,
        87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57,
    ],
    [
        54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22,
        78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54,
    ],
    [
        50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75,
        -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
    ],
    [
        46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13,
        85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46,
    ],
    [
        43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25,
        87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43,
    ],
    [
        38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90,
        -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38,
    ],
    [
        36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36,
        -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
    ],
    [
        31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67,
        90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31,
    ],
    [
        25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80,
        -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25,
    ],
    [
        22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13,
        -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22,
    ],
    [
        18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89,
        89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
    ],
    [
        13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46,
        -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13,
    ],
    [
        9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70,
        80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9,
    ],
    [
        4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85,
        82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4,
    ],
];

const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// 8.6.4.2: y[i] = sum_j M[j][i] * x[j] for the n outputs, over the first `nin` inputs (the
/// others are zero; `nin` = n is the full transform). `stride` picks DCT rows.
fn inverse_1d(x: &[i32], nin: usize, n: usize, stride: usize, dst: bool, y: &mut [i32; 32]) {
    *y = [0; 32];
    for (j, &xj) in x.iter().enumerate().take(nin) {
        let row: &[i32] = if dst {
            DST.get(j).map(|r| r.as_slice()).unwrap_or_default()
        } else {
            DCT.get(j * stride)
                .map(|r| r.as_slice())
                .unwrap_or_default()
        };
        for (acc, &m) in y.iter_mut().zip(row).take(n) {
            *acc += m * xj;
        }
    }
}

/// The scale and bdShift of 8.6.3: `(levelScale[qP % 6] << (qP / 6), bdShift)`.
fn dequant_params(p: &TuParams) -> (i64, u32) {
    let bd_shift = p.bit_depth + p.log2_size - 5;
    let scale = LEVEL_SCALE.get((p.qp % 6) as usize).copied().unwrap_or(40) << (p.qp / 6);
    (scale, bd_shift)
}

/// One scaled coefficient of 8.6.3, in i64: `(c * m * scale + round) >> bdShift`, clipped to
/// 16 bits.
fn dq1(c: i32, m: i64, scale: i64, bd_shift: u32) -> i32 {
    let d = ((i64::from(c) * m * scale) + (1 << (bd_shift - 1))) >> bd_shift;
    d.clamp(-32768, 32767) as i32
}

/// 8.6.3 scaling in i64, in place, of the first `rows` rows and `cols` columns (`(n, n)` is the
/// whole block: the oracle). The scalar tier uses it on the extent of the non-zero
/// coefficients, the SIMD tiers when their i32 arithmetic could overflow.
fn dequant_scalar(
    coeffs: &mut [i32],
    (rows, cols): (usize, usize),
    list: Option<&ScalingList>,
    p: &TuParams,
) {
    let n = 1usize << p.log2_size;
    let (scale, bd_shift) = dequant_params(p);
    for (y, row) in coeffs.chunks_exact_mut(n).enumerate().take(rows) {
        for (x, c) in row.iter_mut().enumerate().take(cols) {
            let m = list.map_or(16, |l| i64::from(l.factor(p.log2_size, p.matrix_id, x, y)));
            *c = dq1(*c, m, scale, bd_shift);
        }
    }
}

/// Everything after the scaling, in place: the transform (or the transform-skip shift) and the
/// bdShift rounding of 8.6.2 equation 8-299 (the oracle).
fn transform_scalar(coeffs: &mut [i32], p: &TuParams) {
    let n = 1usize << p.log2_size;
    let bd_shift2 = 20 - p.bit_depth;
    if p.transform_skip {
        // 8-298, tsShift = 5 + Log2(nTbS).
        for c in coeffs.iter_mut() {
            *c <<= 5 + p.log2_size;
        }
    } else {
        let stride = 1usize << (5 - p.log2_size);
        let mut col = [0i32; 32];
        let mut out = [0i32; 32];
        // 8.6.4.1 step 1 and 2: columns, then the intermediate clip.
        for x in 0..n {
            for (v, row) in col.iter_mut().zip(coeffs.chunks_exact(n)) {
                *v = row.get(x).copied().unwrap_or(0);
            }
            inverse_1d(&col, n, n, stride, p.dst, &mut out);
            for (v, row) in out.iter().zip(coeffs.chunks_exact_mut(n)) {
                if let Some(c) = row.get_mut(x) {
                    *c = ((*v + 64) >> 7).clamp(-32768, 32767);
                }
            }
        }
        // Step 3: rows.
        for row in coeffs.chunks_exact_mut(n) {
            col = [0; 32];
            for (v, c) in col.iter_mut().zip(row.iter()) {
                *v = *c;
            }
            inverse_1d(&col, n, n, stride, p.dst, &mut out);
            for (c, v) in row.iter_mut().zip(out.iter()) {
                *c = *v;
            }
        }
    }
    for c in coeffs.iter_mut() {
        *c = (*c + (1 << (bd_shift2 - 1))) >> bd_shift2;
    }
}

/// The scalar tier: [`transform_scalar`] of scaled coefficients that are zero outside the first
/// `rows` rows and `cols` columns, from `coeffs` into `res` (n * n entries, any content on
/// entry). Pass 1 transforms only the first `cols` columns, from their first `rows` inputs;
/// the other columns of the intermediate are zero, so pass 2 reads only the first `cols`
/// inputs of each row. The terms left out are products with zero: the result is the same bit
/// for bit, the intermediate clip included.
fn transform_sparse(coeffs: &[i32], (rows, cols): (usize, usize), p: &TuParams, res: &mut [i32]) {
    let n = 1usize << p.log2_size;
    let bd_shift2 = 20 - p.bit_depth;
    if p.transform_skip {
        for (r, c) in res.iter_mut().zip(coeffs) {
            *r = *c << (5 + p.log2_size);
        }
    } else {
        let stride = 1usize << (5 - p.log2_size);
        let mut col = [0i32; 32];
        let mut out = [0i32; 32];
        for x in 0..cols {
            for (v, row) in col.iter_mut().zip(coeffs.chunks_exact(n)).take(rows) {
                *v = row.get(x).copied().unwrap_or(0);
            }
            inverse_1d(&col, rows, n, stride, p.dst, &mut out);
            for (v, row) in out.iter().zip(res.chunks_exact_mut(n)) {
                if let Some(c) = row.get_mut(x) {
                    *c = ((*v + 64) >> 7).clamp(-32768, 32767);
                }
            }
        }
        for row in res.chunks_exact_mut(n) {
            for (v, c) in col.iter_mut().zip(row.iter()).take(cols) {
                *v = *c;
            }
            inverse_1d(&col, cols, n, stride, p.dst, &mut out);
            for (c, v) in row.iter_mut().zip(out.iter()) {
                *c = *v;
            }
        }
    }
    for c in res.iter_mut() {
        *c = (*c + (1 << (bd_shift2 - 1))) >> bd_shift2;
    }
}

/// The residual of a DCT block whose only non-zero coefficient is the DC one (no transform
/// skip, no DST): every sample gets the same value. With d the scaled coefficient, pass 1 gives
/// `DCT[0][y] * d = 64 * d` in every row of column 0 and zero elsewhere, clipped to
/// `g = (64 * d + 64) >> 7`; pass 2 gives `64 * g` in every sample, then the bdShift rounding of
/// 8-299. The same operations as [`transform_scalar`] in the same order.
fn dc_residual(c: i32, list: Option<&ScalingList>, p: &TuParams) -> i32 {
    let (scale, bd_shift) = dequant_params(p);
    let m = list.map_or(16, |l| i64::from(l.factor(p.log2_size, p.matrix_id, 0, 0)));
    let d = dq1(c, m, scale, bd_shift);
    let g = ((64 * d + 64) >> 7).clamp(-32768, 32767);
    let bd_shift2 = 20 - p.bit_depth;
    (64 * g + (1 << (bd_shift2 - 1))) >> bd_shift2
}

/// The scaling of 8.6.3 as the SIMD tiers do it, in i32.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct Dq<'a> {
    /// The scaling factors m, row-major; None is the flat 16.
    mat: Option<&'a [u8]>,
    /// levelScale[qP % 6] << (qP / 6).
    scale: i32,
    bd_shift: u32,
}

/// The largest `f(c, m)` over the first `rows` rows and `cols` columns of the n x n block
/// (`m` is the scaling factor at that position, 16 without a matrix).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn max_in(
    coeffs: &[i32],
    mat: Option<&[u8]>,
    n: usize,
    (rows, cols): (usize, usize),
    f: impl Fn(i32, u8) -> u64,
) -> u64 {
    let mut max = 0u64;
    for (y, row) in coeffs.chunks_exact(n).enumerate().take(rows) {
        let mrow = mat.and_then(|m| m.get(y * n..)).unwrap_or_default();
        for (x, &c) in row.iter().enumerate().take(cols) {
            max = max.max(f(c, mrow.get(x).copied().unwrap_or(16)));
        }
    }
    max
}

/// The i32 scaling of the coefficients in the first `rows` rows and `cols` columns when
/// `|c| * m * scale + round` fits i32 for all of them (then every product is exact and equals
/// the i64 one); otherwise scales them here in i64 and gives None.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn dq_for<'a>(
    coeffs: &mut [i32],
    ext: (usize, usize),
    list: Option<&'a ScalingList>,
    p: &TuParams,
) -> Option<Dq<'a>> {
    let (scale, bd_shift) = dequant_params(p);
    let n = 1usize << p.log2_size;
    let mat = list
        .and_then(|l| l.matrix(p.log2_size, p.matrix_id))
        .filter(|m| m.len() == n * n);
    // At most 2^31 * 255: no u64 overflow, nor below with scale <= 72 * 2^16.
    let max_cm = max_in(coeffs, mat, n, ext, |c, m| {
        u64::from(c.unsigned_abs()) * u64::from(m)
    });
    let bound = max_cm * scale.unsigned_abs() + (1u64 << (bd_shift - 1));
    match i32::try_from(scale) {
        Ok(scale) if bound <= u64::from(i32::MAX.unsigned_abs()) => Some(Dq {
            mat,
            scale,
            bd_shift,
        }),
        _ => {
            dequant_scalar(coeffs, ext, list, p);
            None
        }
    }
}

/// The scaled coefficients in the first `rows` rows and `cols` columns as i16 (they are clamped
/// to that range) in `c16`, which is zero elsewhere; returns (rows, cols) again, tightened to
/// the non-zero ones, so that rows and columns past it can be skipped.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn pack(coeffs: &[i32], n: usize, (rows, cols): (usize, usize), c16: &mut [i16]) -> (usize, usize) {
    let mut cols_or = [0i16; 32];
    let mut last = 0;
    for (i, (src, dst)) in coeffs
        .chunks_exact(n)
        .zip(c16.chunks_exact_mut(n))
        .enumerate()
        .take(rows)
    {
        let mut any = 0i16;
        for ((d, &s), o) in dst.iter_mut().zip(src).zip(cols_or.iter_mut()).take(cols) {
            let v = s as i16;
            *d = v;
            *o |= v;
            any |= v;
        }
        if any != 0 {
            last = i + 1;
        }
    }
    let cols = cols_or.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    (last, cols)
}

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Scaling (in place, of the first `rows` rows and `cols` columns) and transform of the block
/// into `res` by the SIMD tier; false when `tier` has no kernel (scalar, or not detected here).
/// The coefficients are zero outside the extent `ext`.
#[allow(unsafe_code)]
fn residual_simd(
    tier: Tier,
    coeffs: &mut [i32],
    ext: (usize, usize),
    list: Option<&ScalingList>,
    p: &TuParams,
    res: &mut [i32],
) -> bool {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            let dq = dq_for(coeffs, ext, list, p);
            // SAFETY: the guard above found AVX2 on this CPU (Avx512 implies it, see
            // src/simd/mod.rs); `coeffs` and `res` have n * n entries, `ext` is at most (n, n)
            // and `p` is validated (residual_block).
            unsafe { avx2::residual(coeffs, ext, dq, p, res) };
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            let dq = dq_for(coeffs, ext, list, p);
            // SAFETY: the guard above found NEON on this CPU (src/simd/mod.rs); `coeffs` and
            // `res` have n * n entries, `ext` is at most (n, n) and `p` is validated
            // (residual_block).
            unsafe { neon::residual(coeffs, ext, dq, p, res) };
            true
        }
        _ => false,
    }
}

/// 8.6.7 for one row: adds `res` to the prediction in `row` and clips to [0, max].
#[allow(unsafe_code)]
fn add_row(tier: Tier, row: &mut [u16], res: &[i32], max: i32) {
    // The kernels handle equal-length rows of 4, 8, 16 or 32 only; anything else runs scalar.
    let simd_ok = row.len() == res.len() && matches!(row.len(), 4 | 8 | 16 | 32);
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if simd_ok && std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: the guard above found AVX2 on this CPU (Avx512 implies it, see
            // src/simd/mod.rs); the kernel only reads and writes within the two slices.
            unsafe { avx2::add_row(row, res, max) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if simd_ok && std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: the guard above found NEON on this CPU (src/simd/mod.rs); the kernel only
            // reads and writes within the two slices.
            unsafe { neon::add_row(row, res, max) }
        }
        _ => {
            for (s, &r) in row.iter_mut().zip(res) {
                *s = i32::from(*s).saturating_add(r).clamp(0, max) as u16;
            }
        }
    }
}

/// The residual of one transform block: one value for every sample (a DC-only block), or n * n
/// values, row-major.
enum Res<'a> {
    Dc([i32; 32]),
    Block(&'a [i32]),
}

impl Res<'_> {
    /// Row `y` of the n x n residual.
    fn row(&self, y: usize, n: usize) -> &[i32] {
        match self {
            Res::Dc(v) => v.get(..n).unwrap_or_default(),
            Res::Block(r) => r.get(y * n..).and_then(|r| r.get(..n)).unwrap_or_default(),
        }
    }
}

/// The checks of [`residual_block`] and of the oracle [`residual_full`]: n, or the error.
fn check(p: &TuParams, len: usize) -> Result<usize> {
    if !(2..=5).contains(&p.log2_size)
        || !(8..=16).contains(&p.bit_depth)
        || (p.dst && p.log2_size != 2)
    {
        return Err(Error::InvalidData("transform block size or bit depth"));
    }
    let n = 1usize << p.log2_size;
    if len < n * n {
        return Err(Error::InvalidData("coefficient buffer too small"));
    }
    // 8.6.3 (CoeffMin/Max are -32768 / 32767 without extended precision).
    if !p.bypass && !(0..=99).contains(&p.qp) {
        return Err(Error::InvalidData("qP out of range"));
    }
    Ok(n)
}

/// The scaling list of 8.6.3: none (m = 16) for transform skip of a block larger than 4x4.
fn list_for<'a>(list: Option<&'a ScalingList>, p: &TuParams) -> Option<&'a ScalingList> {
    if p.transform_skip && p.log2_size > 2 {
        None
    } else {
        list
    }
}

/// The oracle: residual of one transform block (8.6.2 to 8.6.4) by the full scalar transform of
/// the whole block, in place; returns the first n * n entries of `coeffs`.
fn residual_full<'a>(
    coeffs: &'a mut [i32],
    list: Option<&ScalingList>,
    p: &TuParams,
) -> Result<&'a [i32]> {
    let n = check(p, coeffs.len())?;
    let coeffs = coeffs.get_mut(..n * n).unwrap_or_default();
    if !p.bypass {
        dequant_scalar(coeffs, (n, n), list_for(list, p), p);
        transform_scalar(coeffs, p);
    }
    Ok(coeffs)
}

/// Residual of one transform block (8.6.2 to 8.6.4) on tier `tier`: validates `p`, scales the
/// coefficients in place and transforms them into `res` (n * n entries, any content on entry).
/// `coeffs` is row-major (`coeffs[y * n + x]` is TransCoeffLevel[x][y]) and zero outside the
/// first `rows` rows and `cols` columns (`ext`), so the work scales with that extent; only that
/// part of `coeffs` is written. Bypass copies the coefficients.
fn residual_block<'a>(
    tier: Tier,
    coeffs: &mut [i32],
    ext: (usize, usize),
    list: Option<&ScalingList>,
    p: &TuParams,
    res: &'a mut [i32],
) -> Result<Res<'a>> {
    let n = check(p, coeffs.len())?;
    let coeffs = coeffs.get_mut(..n * n).unwrap_or_default();
    let res = res
        .get_mut(..n * n)
        .ok_or(Error::InvalidData("residual buffer too small"))?;
    let ext = (ext.0.min(n), ext.1.min(n));
    if p.bypass {
        res.copy_from_slice(coeffs);
        return Ok(Res::Block(res));
    }
    let list = list_for(list, p);
    if ext.0 <= 1 && ext.1 <= 1 && !p.transform_skip && !p.dst {
        let c = coeffs.first().copied().unwrap_or(0);
        return Ok(Res::Dc([dc_residual(c, list, p); 32]));
    }
    if !residual_simd(tier, coeffs, ext, list, p, res) {
        dequant_scalar(coeffs, ext, list, p);
        transform_sparse(coeffs, ext, p, res);
    }
    Ok(Res::Block(res))
}

/// The extent (rows, cols) of the non-zero entries of an n x n block: all of them are in the
/// first `rows` rows and the first `cols` columns.
pub(crate) fn extent(coeffs: &[i32], n: usize) -> (usize, usize) {
    let (mut rows, mut cols) = (0, 0);
    for (y, row) in coeffs.chunks_exact(n).enumerate() {
        if let Some(x) = row.iter().rposition(|&c| c != 0) {
            rows = y + 1;
            cols = cols.max(x + 1);
        }
    }
    (rows, cols)
}

/// Computes the residual of one transform block from `coeffs` (row-major, `coeffs[y * n + x]` is
/// TransCoeffLevel[x][y], zero outside the first `rows` rows and `cols` columns, `ext`) and adds
/// it to the prediction already in the frame at component sample position (`x0`, `y0`), clipping
/// to the bit depth (8.6.7). `res` is scratch of n * n entries. `coeffs` is all zero again on
/// return, also on error: only the extent is cleared, so the caller can keep one buffer that
/// residual_coding fills without clearing it first.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct(
    frame: &mut Frame,
    act: &Active,
    c_idx: usize,
    x0: usize,
    y0: usize,
    coeffs: &mut [i32],
    ext: (usize, usize),
    res: &mut [i32],
    p: &TuParams,
) -> Result<()> {
    let tier = simd::tier();
    let out = residual_block(tier, coeffs, ext, act.scaling(), p, res);
    clear(coeffs, p.log2_size, ext);
    let out = out?;
    // 8.6.7.
    let n = 1usize << p.log2_size;
    let max = (1i32 << p.bit_depth) - 1;
    for y in 0..n {
        let row = frame.row_mut(c_idx, x0, y0.saturating_add(y), n)?;
        add_row(tier, row, out.row(y, n), max);
    }
    Ok(())
}

/// Zeroes the first `rows` rows and `cols` columns of the n x n block in `coeffs`: the only
/// entries that residual_coding and the scaling write. The whole buffer when `log2_size` is not
/// a transform size.
fn clear(coeffs: &mut [i32], log2_size: u32, (rows, cols): (usize, usize)) {
    if !(2..=5).contains(&log2_size) {
        coeffs.fill(0);
        return;
    }
    for row in coeffs.chunks_exact_mut(1 << log2_size).take(rows) {
        row.iter_mut().take(cols).for_each(|c| *c = 0);
    }
}

/// True when every tier of this CPU (scalar included), given the tight extent of the non-zero
/// coefficients and given the whole block, gives the same residual and the same picture rows as
/// the full scalar transform of the whole block (`pix`: n * n prediction samples, row-major).
/// The cross-tier tests and the fuzz target use it.
pub(crate) fn tiers_agree(
    coeffs: &[i32],
    list: Option<&ScalingList>,
    p: &TuParams,
    pix: &[u16],
) -> bool {
    let n = 1usize << p.log2_size.min(5);
    let max = (1i32 << p.bit_depth.min(16)) - 1;
    let run = |r: Result<Res<'_>>, tier: Tier| -> (Result<Vec<i32>>, Vec<u16>) {
        let mut out = pix.to_vec();
        let r = r.map(|r| {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                add_row(tier, row, r.row(y, n), max);
            }
            (0..n).flat_map(|y| r.row(y, n)).copied().collect()
        });
        (r, out)
    };
    let mut c = coeffs.to_vec();
    let want = run(residual_full(&mut c, list, p).map(Res::Block), Tier::Scalar);
    let exts = [extent(coeffs, n), (n, n)];
    simd::available().into_iter().all(|t| {
        exts.iter().all(|&ext| {
            let mut c = coeffs.to_vec();
            // Not zero: the paths must write every entry they return.
            let mut res = [0x5A5A_5A5A; 1024];
            run(residual_block(t, &mut c, ext, list, p, &mut res), t) == want
        })
    })
}

/// Builds one block from fuzz bytes (a small header, then coefficients and prediction samples)
/// and checks that every tier agrees with the full scalar transform. Sizes are bounded (at most
/// 32 x 32). The coefficients outside a bounding box taken from the header are zero, so the
/// sparse paths (DC only, partial passes) are reached.
pub(crate) fn fuzz_tiers(data: &[u8]) -> bool {
    let h = |i: usize| data.get(i).copied().unwrap_or(0);
    let flags = h(1);
    let log2_size = 2 + u32::from(h(0) % 4);
    let p = TuParams {
        qp: i32::from(h(2) % 100),
        bit_depth: 8 + u32::from(h(3) % 9),
        transform_skip: flags & 1 != 0,
        bypass: flags & 2 != 0,
        dst: flags & 4 != 0 && log2_size == 2,
        matrix_id: usize::from(h(4) % 6),
        log2_size,
    };
    let rest = data.get(6..).unwrap_or_default();
    let owned = match (flags >> 4) & 3 {
        1 => Some(ScalingList::default_lists()),
        2 => ScalingList::parse(&mut BitReader::new(rest)).ok(),
        _ => None,
    };
    let n = 1usize << log2_size;
    let n2 = n * n;
    let mut coeffs = vec![0i32; n2];
    let width = if flags & 8 != 0 { 4 } else { 2 };
    for (c, bytes) in coeffs.iter_mut().zip(rest.chunks(width)) {
        let mut v = 0u32;
        for (k, &b) in bytes.iter().enumerate() {
            v |= u32::from(b) << (8 * k);
        }
        *c = if width == 4 {
            v as i32
        } else {
            i32::from(v as u16 as i16)
        };
    }
    // Box of 1..=n rows and columns: 5 bits each from byte 5 and flag bits 6 and 7.
    let rows = 1 + (usize::from(h(5) & 15) * 2 + usize::from((flags >> 6) & 1)) % n;
    let cols = 1 + (usize::from(h(5) >> 4) * 2 + usize::from(flags >> 7)) % n;
    for (y, row) in coeffs.chunks_exact_mut(n).enumerate() {
        for (x, c) in row.iter_mut().enumerate() {
            if y >= rows || x >= cols {
                *c = 0;
            }
        }
    }
    let pix: Vec<u16> = (0..n2)
        .map(|i| (u32::from(h(6 + i)) * 257 % (1 << p.bit_depth)) as u16)
        .collect();
    tiers_agree(&coeffs, owned.as_ref(), &p, &pix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Ledger;
    use crate::limits::Limits;
    use crate::ps::tests::{ledger, pps_bits, sps_bits};
    use crate::ps::{activate, parse_pps, parse_sps};
    use std::sync::Arc;

    fn setup(l: &Ledger) -> (Active, Frame) {
        let sps = Arc::new(parse_sps(&sps_bits(64, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let act = activate(sps, pps, &Limits::default(), l).unwrap();
        let f = Frame::new(&act, l).unwrap();
        (act, f)
    }

    fn tu(log2_size: u32) -> TuParams {
        // qp 4: levelScale 64, so with m = 16 and 8 bits d = c * 2^(7 - log2_size).
        TuParams {
            qp: 4,
            bit_depth: 8,
            transform_skip: false,
            bypass: false,
            dst: false,
            matrix_id: 0,
            log2_size,
        }
    }

    #[test]
    fn matrix_shape() {
        assert!(DCT[0].iter().all(|&v| v == 64));
        for (k, row) in DCT.iter().enumerate() {
            for n in 0..16 {
                let mirror = if k.is_multiple_of(2) { row[n] } else { -row[n] };
                assert_eq!(row[31 - n], mirror, "row {k} col {n}");
            }
        }
        for (a, ra) in DCT.iter().enumerate() {
            for (b, rb) in DCT.iter().enumerate() {
                let dot: i32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
                if a == b {
                    assert!((dot - 131072).abs() < 2000, "norm {a}: {dot}");
                } else {
                    assert!(dot.abs() < 2000, "dot {a} {b}: {dot}");
                }
            }
        }
    }

    #[test]
    fn dc_only_dct() {
        for log2 in 2..=5u32 {
            let l = ledger();
            let (act, mut f) = setup(&l);
            let n = 1usize << log2;
            let mut c = vec![0i32; n * n];
            c[0] = 4 << log2; // d = 512, e = 32768, g = 256, r = 16384 -> (16384 + 2048) >> 12 = 4
            rec(&mut f, &act, 0, 8, 16, &mut c, &tu(log2)).unwrap();
            for y in 0..n {
                for x in 0..n {
                    assert_eq!(f.sample(0, 8 + x, 16 + y), Some(132), "size {n} at {x},{y}");
                }
            }
            assert_eq!(f.sample(0, 8 + n, 16), Some(128));
        }
    }

    #[test]
    fn dst_first_basis() {
        let l = ledger();
        let (act, mut f) = setup(&l);
        let mut c = vec![0i32; 16];
        c[0] = 16; // d = 512; g = {116, 220, 296, 336}; r = {29, 55, 74, 84}[x] * g[y]
        let p = TuParams { dst: true, ..tu(2) };
        rec(&mut f, &act, 0, 0, 0, &mut c, &p).unwrap();
        assert_eq!(f.sample(0, 0, 0), Some(129)); // 3364 -> 1
        assert_eq!(f.sample(0, 3, 3), Some(135)); // 28224 -> 7
        assert_eq!(f.sample(0, 0, 3), Some(130)); // 9744 -> 2
    }

    #[test]
    fn transform_skip() {
        let l = ledger();
        let (act, mut f) = setup(&l);
        let mut c = vec![0i32; 16];
        c[1] = 2; // d = 64, << 7 = 8192, (8192 + 2048) >> 12 = 2
        let p = TuParams {
            transform_skip: true,
            ..tu(2)
        };
        rec(&mut f, &act, 1, 4, 4, &mut c, &p).unwrap();
        assert_eq!(f.sample(1, 5, 4), Some(130));
        assert_eq!(f.sample(1, 4, 4), Some(128));
        assert_eq!(f.sample(1, 5, 5), Some(128));
    }

    #[test]
    fn bypass_and_clip() {
        let l = ledger();
        let (act, mut f) = setup(&l);
        let mut c = vec![0i32; 16];
        c[0] = 200;
        c[5] = -200;
        c[10] = 7;
        let p = TuParams {
            bypass: true,
            ..tu(2)
        };
        rec(&mut f, &act, 0, 0, 0, &mut c, &p).unwrap();
        assert_eq!(f.sample(0, 0, 0), Some(255));
        assert_eq!(f.sample(0, 1, 1), Some(0));
        assert_eq!(f.sample(0, 2, 2), Some(135));
    }

    #[test]
    fn rejects_bad_input() {
        let l = ledger();
        let (act, mut f) = setup(&l);
        let mut c = vec![0i32; 16];
        // A coefficient in each call: `rec` checks that the buffer is cleared on error too.
        c[5] = 3;
        assert!(rec(&mut f, &act, 0, 62, 0, &mut c, &tu(2)).is_err());
        c[5] = 3;
        assert!(rec(&mut f, &act, 0, 0, 0, &mut c[..8], &tu(2)).is_err());
        c[5] = 3;
        assert!(rec(&mut f, &act, 0, 0, 0, &mut c, &tu(6)).is_err());
        let p = TuParams { qp: -1, ..tu(2) };
        c[5] = 3;
        assert!(rec(&mut f, &act, 0, 0, 0, &mut c, &p).is_err());
        let p = TuParams {
            bit_depth: 7,
            ..tu(2)
        };
        c[5] = 3;
        assert!(rec(&mut f, &act, 0, 0, 0, &mut c, &p).is_err());
    }

    /// The residual equals the full transform for a DC-only block on every tier, for every size,
    /// bit depth, qP, scaling list and the extreme coefficients (the fast path of
    /// `residual_block`), and it is one value for the whole block.
    #[test]
    fn tiers_dc_only_match_full() {
        let lists = [None, Some(ScalingList::default_lists())];
        let dcs = [
            i32::MIN,
            -65536,
            -32768,
            -3001,
            -1,
            0,
            1,
            2,
            77,
            32767,
            65535,
            i32::MAX,
        ];
        for log2_size in 2..=5u32 {
            let n2 = 1usize << (2 * log2_size);
            for bit_depth in [8u32, 10, 12, 16] {
                for qp in [0i32, 4, 17, 33, 51, 75, 99] {
                    for &dc in &dcs {
                        for list in &lists {
                            let mut c = vec![0i32; n2];
                            c[0] = dc;
                            let p = TuParams {
                                qp,
                                bit_depth,
                                matrix_id: (qp % 6) as usize,
                                ..tu(log2_size)
                            };
                            let pix = vec![(1u16 << (bit_depth - 1)) - 1; n2];
                            assert!(
                                tiers_agree(&c, list.as_ref(), &p, &pix),
                                "size {log2_size} depth {bit_depth} qp {qp} dc {dc}"
                            );
                            let mut res = [0; 1024];
                            let r = residual_block(
                                Tier::Scalar,
                                &mut c,
                                (1, 1),
                                list.as_ref(),
                                &p,
                                &mut res,
                            )
                            .unwrap();
                            assert!(matches!(r, Res::Dc(_)));
                        }
                    }
                }
            }
        }
    }

    /// Every bounding box of every size: random sparse coefficients inside it (its last row and
    /// last column non-zero), DCT, DST, transform skip and bypass, flat and default scaling
    /// lists; every tier, given the tight extent and the whole block, against the full scalar
    /// transform. 32x32 boxes take one of the cases each (time under qemu).
    #[test]
    fn tiers_sparse_match_full() {
        let lists = [None, Some(ScalingList::default_lists())];
        let cases = [
            (false, false, false),
            (false, false, true),
            (true, false, false),
            (false, true, false),
        ];
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for log2_size in 2..=5u32 {
            let n = 1usize << log2_size;
            for rows in 1..=n {
                for cols in 1..=n {
                    let picks = if n == 32 {
                        let k = (rows * 7 + cols) % 8;
                        k..k + 1
                    } else {
                        0..8
                    };
                    for pick in picks {
                        let (transform_skip, bypass, dst) = cases[pick % 4];
                        if dst && n != 4 {
                            continue;
                        }
                        let list = &lists[pick / 4];
                        let bit_depth = [8u32, 10, 12, 16][(rng.next() % 4) as usize];
                        let max = (1i32 << bit_depth) - 1;
                        let mut c = vec![0i32; n * n];
                        for y in 0..rows {
                            for x in 0..cols {
                                if rng.next().is_multiple_of(3) {
                                    c[y * n + x] = rng.range(-3000, 3000);
                                }
                            }
                        }
                        let x = (rng.next() % cols as u64) as usize;
                        c[(rows - 1) * n + x] = rng.range(1, 3000);
                        let y = (rng.next() % rows as u64) as usize;
                        c[y * n + cols - 1] = -rng.range(1, 3000);
                        assert_eq!(extent(&c, n), (rows, cols));
                        let pix: Vec<u16> = (0..n * n).map(|_| rng.range(0, max) as u16).collect();
                        let p = TuParams {
                            qp: rng.range(0, 60),
                            bit_depth,
                            transform_skip,
                            bypass,
                            dst,
                            matrix_id: (rng.next() % 6) as usize,
                            log2_size,
                        };
                        assert!(
                            tiers_agree(&c, list.as_ref(), &p, &pix),
                            "size {log2_size} box {rows}x{cols} ts {transform_skip} bypass {bypass} dst {dst} list {}",
                            list.is_some()
                        );
                    }
                }
            }
        }
    }

    /// [`reconstruct`] with the extent of `c` and a scratch residual that is not zero; checks
    /// that `c` is all zero afterwards, also on error.
    fn rec(
        f: &mut Frame,
        act: &Active,
        c_idx: usize,
        x: usize,
        y: usize,
        c: &mut [i32],
        p: &TuParams,
    ) -> Result<()> {
        let ext = extent(c, 1 << p.log2_size.clamp(2, 5));
        let mut res = [0x5A5A_5A5A; 1024];
        let r = reconstruct(f, act, c_idx, x, y, c, ext, &mut res, p);
        assert!(
            c.iter().all(|&v| v == 0),
            "coefficients left after the block"
        );
        r
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (i64::from(hi) - i64::from(lo) + 1) as u64) as i32
        }
    }

    fn gen_coeffs(rng: &mut Rng, n2: usize, kind: u32) -> Vec<i32> {
        let n = n2.isqrt();
        let mut c = vec![0i32; n2];
        let extremes = [i32::MIN, i32::MAX, -32768, 32767, 0, 1, -1, 65535, -65536];
        match kind {
            0 => {}
            1 => c[0] = rng.range(-3000, 3000),
            2 => {
                for v in c.iter_mut() {
                    if rng.next().is_multiple_of(10) {
                        *v = rng.range(-50, 50);
                    }
                }
            }
            3 => c.iter_mut().for_each(|v| *v = rng.range(-32768, 32767)),
            4 => c
                .iter_mut()
                .for_each(|v| *v = extremes[(rng.next() % 9) as usize]),
            5 => c[n2 - 1] = rng.range(-32768, 32767),
            6 => c.iter_mut().for_each(|v| *v = rng.range(-1000, 1000)),
            7 => c[..n].iter_mut().for_each(|v| *v = rng.range(-5000, 5000)),
            _ => {
                for v in c.iter_mut() {
                    if rng.next().is_multiple_of(16) {
                        *v = rng.range(-2_000_000, 2_000_000);
                    }
                }
            }
        }
        c
    }

    #[test]
    fn tiers_match_scalar() {
        let avail = simd::available();
        #[cfg(target_arch = "x86_64")]
        assert!(avail.contains(&Tier::Avx2), "AVX2 not detected");
        #[cfg(target_arch = "aarch64")]
        assert!(avail.contains(&Tier::Neon), "NEON not detected");
        assert!(avail.len() > 1, "no SIMD tier to test");
        let lists = [None, Some(ScalingList::default_lists())];
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for log2_size in 2..=5u32 {
            let n2 = 1usize << (2 * log2_size);
            for bit_depth in [8u32, 10, 12, 16] {
                let max = (1i32 << bit_depth) - 1;
                for qp in [0i32, 4, 17, 33, 51, 75, 99] {
                    for (transform_skip, bypass, dst) in [
                        (false, false, false),
                        (false, false, true),
                        (true, false, false),
                        (false, true, false),
                    ] {
                        if dst && log2_size != 2 {
                            continue;
                        }
                        for kind in 0..9 {
                            let coeffs = gen_coeffs(&mut rng, n2, kind);
                            let pix: Vec<u16> = (0..n2).map(|_| rng.range(0, max) as u16).collect();
                            let p = TuParams {
                                qp,
                                bit_depth,
                                transform_skip,
                                bypass,
                                dst,
                                matrix_id: (rng.next() % 6) as usize,
                                log2_size,
                            };
                            for list in &lists {
                                assert!(
                                    tiers_agree(&coeffs, list.as_ref(), &p, &pix),
                                    "size {log2_size} depth {bit_depth} qp {qp} ts {transform_skip} bypass {bypass} dst {dst} kind {kind} list {}",
                                    list.is_some()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tiers_reject_alike() {
        let c = vec![1i32; 1024];
        let pix = vec![0u16; 1024];
        for (log2_size, bit_depth, qp) in [
            (1u32, 8u32, 4i32),
            (6, 8, 4),
            (3, 7, 4),
            (3, 17, 4),
            (3, 8, -1),
            (3, 8, 100),
        ] {
            let p = TuParams {
                qp,
                bit_depth,
                transform_skip: false,
                bypass: false,
                dst: false,
                matrix_id: 0,
                log2_size,
            };
            assert!(tiers_agree(&c, None, &p, &pix));
            assert!(
                residual_block(
                    Tier::Scalar,
                    &mut c.clone(),
                    (32, 32),
                    None,
                    &p,
                    &mut [0; 1024]
                )
                .is_err()
            );
        }
    }

    #[test]
    fn fuzz_tiers_random_bytes() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for len in [0usize, 1, 5, 6, 7, 40, 200, 2100, 4200] {
            for _ in 0..40 {
                let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
                assert!(fuzz_tiers(&data), "len {len}");
                // Mostly zero bytes: sparse blocks.
                let sparse: Vec<u8> = data.iter().map(|&b| if b < 200 { 0 } else { b }).collect();
                assert!(fuzz_tiers(&sparse), "sparse len {len}");
            }
        }
    }
}
