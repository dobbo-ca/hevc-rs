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

/// 8.6.4.2: y[i] = sum_j M[j][i] * x[j] over the first `n` entries (`stride` picks DCT rows).
fn inverse_1d(x: &[i32], n: usize, stride: usize, dst: bool, y: &mut [i32; 32]) {
    *y = [0; 32];
    for (j, &xj) in x.iter().enumerate().take(n) {
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

/// 8.6.3 scaling in i64 (the oracle; the SIMD tiers use it when their i32 arithmetic could
/// overflow).
fn dequant_scalar(coeffs: &mut [i32], list: Option<&ScalingList>, p: &TuParams) {
    let n = 1usize << p.log2_size;
    let (scale, bd_shift) = dequant_params(p);
    for (i, c) in coeffs.iter_mut().enumerate() {
        let m = list.map_or(16, |l| {
            i64::from(l.factor(p.log2_size, p.matrix_id, i % n, i / n))
        });
        let d = ((i64::from(*c) * m * scale) + (1 << (bd_shift - 1))) >> bd_shift;
        *c = d.clamp(-32768, 32767) as i32;
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
            inverse_1d(&col, n, stride, p.dst, &mut out);
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
            inverse_1d(&col, n, stride, p.dst, &mut out);
            for (c, v) in row.iter_mut().zip(out.iter()) {
                *c = *v;
            }
        }
    }
    for c in coeffs.iter_mut() {
        *c = (*c + (1 << (bd_shift2 - 1))) >> bd_shift2;
    }
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

/// The i32 scaling of `coeffs` when `|c| * m * scale + round` fits i32 for all of them (then
/// every product is exact and equals the i64 one); otherwise scales them here in i64 and gives
/// None.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn dq_for<'a>(coeffs: &mut [i32], list: Option<&'a ScalingList>, p: &TuParams) -> Option<Dq<'a>> {
    let (scale, bd_shift) = dequant_params(p);
    let mat = list
        .and_then(|l| l.matrix(p.log2_size, p.matrix_id))
        .filter(|m| m.len() == 1usize << (2 * p.log2_size));
    let mmax = mat.map_or(16u64, |m| m.iter().copied().max().map_or(16, u64::from));
    let max_abs = coeffs
        .iter()
        .map(|c| u64::from(c.unsigned_abs()))
        .max()
        .unwrap_or(0);
    // At most 2^31 * 255 * 72 * 2^16: no u64 overflow.
    let bound = max_abs * mmax * scale.unsigned_abs() + (1u64 << (bd_shift - 1));
    match i32::try_from(scale) {
        Ok(scale) if bound <= u64::from(i32::MAX.unsigned_abs()) => Some(Dq {
            mat,
            scale,
            bd_shift,
        }),
        _ => {
            dequant_scalar(coeffs, list, p);
            None
        }
    }
}

/// Dequantised coefficients of a non-zero-skipping transform as i16 (they are clamped to that
/// range) in `c16`; returns (rows, cols): the extent of the non-zero ones, so that rows and
/// columns past it can be skipped.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn pack(coeffs: &[i32], n: usize, c16: &mut [i16]) -> (usize, usize) {
    let mut cols_or = [0i16; 32];
    let mut rows = 0;
    for (i, (src, dst)) in coeffs
        .chunks_exact(n)
        .zip(c16.chunks_exact_mut(n))
        .enumerate()
    {
        let mut any = 0i16;
        for ((d, &s), o) in dst.iter_mut().zip(src).zip(cols_or.iter_mut()) {
            let v = s as i16;
            *d = v;
            *o |= v;
            any |= v;
        }
        if any != 0 {
            rows = i + 1;
        }
    }
    let cols = cols_or.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    (rows, cols)
}

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Scaling and transform of the whole block in place by the SIMD tier; false when `tier` has
/// no kernel (scalar, or not detected here).
#[allow(unsafe_code)]
fn residual_simd(tier: Tier, coeffs: &mut [i32], list: Option<&ScalingList>, p: &TuParams) -> bool {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            let dq = dq_for(coeffs, list, p);
            // SAFETY: the guard above found AVX2 on this CPU (Avx512 implies it, see
            // src/simd/mod.rs); `coeffs` has n * n entries and `p` is validated (residual_block).
            unsafe { avx2::residual(coeffs, dq, p) };
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            let dq = dq_for(coeffs, list, p);
            // SAFETY: the guard above found NEON on this CPU (src/simd/mod.rs); `coeffs` has
            // n * n entries and `p` is validated (residual_block).
            unsafe { neon::residual(coeffs, dq, p) };
            true
        }
        _ => false,
    }
}

/// 8.6.7 for one row: adds `res` to the prediction in `row` and clips to [0, max].
#[allow(unsafe_code)]
fn add_row(tier: Tier, row: &mut [u16], res: &[i32], max: i32) {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: the guard above found AVX2 on this CPU (Avx512 implies it, see
            // src/simd/mod.rs); the kernel only reads and writes within the two slices.
            unsafe { avx2::add_row(row, res, max) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
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

/// Residual of one transform block (8.6.2 to 8.6.4) on tier `tier`: validates `p`, scales and
/// transforms `coeffs` (row-major, `coeffs[y * n + x]` is TransCoeffLevel[x][y]) in place and
/// returns its first n * n entries, the residual after bdShift. Bypass leaves them unchanged.
fn residual_block<'a>(
    tier: Tier,
    coeffs: &'a mut [i32],
    list: Option<&ScalingList>,
    p: &TuParams,
) -> Result<&'a [i32]> {
    if !(2..=5).contains(&p.log2_size)
        || !(8..=16).contains(&p.bit_depth)
        || (p.dst && p.log2_size != 2)
    {
        return Err(Error::InvalidData("transform block size or bit depth"));
    }
    let n = 1usize << p.log2_size;
    let coeffs = coeffs
        .get_mut(..n * n)
        .ok_or(Error::InvalidData("coefficient buffer too small"))?;
    if !p.bypass {
        // 8.6.3 (CoeffMin/Max are -32768 / 32767 without extended precision).
        if !(0..=99).contains(&p.qp) {
            return Err(Error::InvalidData("qP out of range"));
        }
        let list = if p.transform_skip && n > 4 {
            None
        } else {
            list
        };
        if !residual_simd(tier, coeffs, list, p) {
            dequant_scalar(coeffs, list, p);
            transform_scalar(coeffs, p);
        }
    }
    Ok(coeffs)
}

/// Computes the residual of one transform block from `coeffs` (row-major, `coeffs[y * n + x]` is
/// TransCoeffLevel[x][y]; overwritten) and adds it to the prediction already in the frame at
/// component sample position (`x0`, `y0`), clipping to the bit depth (8.6.7).
pub(crate) fn reconstruct(
    frame: &mut Frame,
    act: &Active,
    c_idx: usize,
    x0: usize,
    y0: usize,
    coeffs: &mut [i32],
    p: &TuParams,
) -> Result<()> {
    let tier = simd::tier();
    let res = residual_block(tier, coeffs, act.scaling(), p)?;
    // 8.6.7.
    let n = 1usize << p.log2_size;
    let max = (1i32 << p.bit_depth) - 1;
    for (y, res) in res.chunks_exact(n).enumerate() {
        let row = frame.row_mut(c_idx, x0, y0.saturating_add(y), n)?;
        add_row(tier, row, res, max);
    }
    Ok(())
}

/// True when every tier of this CPU gives the same residual and the same picture row sums as the
/// scalar one for these inputs (`pix`: n * n prediction samples, row-major). The cross-tier
/// tests and the fuzz target use it.
pub(crate) fn tiers_agree(
    coeffs: &[i32],
    list: Option<&ScalingList>,
    p: &TuParams,
    pix: &[u16],
) -> bool {
    let n = 1usize << p.log2_size.min(5);
    let max = (1i32 << p.bit_depth.min(16)) - 1;
    let run = |tier: Tier| -> (Result<Vec<i32>>, Vec<u16>) {
        let mut c = coeffs.to_vec();
        let res = residual_block(tier, &mut c, list, p).map(<[i32]>::to_vec);
        let mut out = pix.to_vec();
        if let Ok(r) = &res {
            for (row, r) in out.chunks_exact_mut(n).zip(r.chunks_exact(n)) {
                add_row(tier, row, r, max);
            }
        }
        (res, out)
    };
    let want = run(Tier::Scalar);
    simd::available().into_iter().all(|t| run(t) == want)
}

/// Builds one block from fuzz bytes (a small header, then coefficients and prediction samples)
/// and checks that every tier agrees with scalar. Sizes are bounded (at most 32 x 32).
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
    let n2 = 1usize << (2 * log2_size);
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
            reconstruct(&mut f, &act, 0, 8, 16, &mut c, &tu(log2)).unwrap();
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
        reconstruct(&mut f, &act, 0, 0, 0, &mut c, &p).unwrap();
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
        reconstruct(&mut f, &act, 1, 4, 4, &mut c, &p).unwrap();
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
        reconstruct(&mut f, &act, 0, 0, 0, &mut c, &p).unwrap();
        assert_eq!(f.sample(0, 0, 0), Some(255));
        assert_eq!(f.sample(0, 1, 1), Some(0));
        assert_eq!(f.sample(0, 2, 2), Some(135));
    }

    #[test]
    fn rejects_bad_input() {
        let l = ledger();
        let (act, mut f) = setup(&l);
        let mut c = vec![0i32; 16];
        assert!(reconstruct(&mut f, &act, 0, 62, 0, &mut c, &tu(2)).is_err());
        assert!(reconstruct(&mut f, &act, 0, 0, 0, &mut c[..8], &tu(2)).is_err());
        assert!(reconstruct(&mut f, &act, 0, 0, 0, &mut c, &tu(6)).is_err());
        let p = TuParams { qp: -1, ..tu(2) };
        assert!(reconstruct(&mut f, &act, 0, 0, 0, &mut c, &p).is_err());
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
            assert!(residual_block(Tier::Scalar, &mut c.clone(), None, &p).is_err());
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
