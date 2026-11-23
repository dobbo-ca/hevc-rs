//! Scaling and transformation, clauses 8.6.2 to 8.6.4 (v1: no extended precision, no
//! transform-skip rotation, no RDPCM), plus the picture construction of 8.6.7.

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::ps::Active;

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
        let bd_shift = p.bit_depth + p.log2_size - 5;
        let scale = LEVEL_SCALE.get((p.qp % 6) as usize).copied().unwrap_or(40) << (p.qp / 6);
        let list = if p.transform_skip && n > 4 {
            None
        } else {
            act.scaling()
        };
        for (i, c) in coeffs.iter_mut().enumerate() {
            let m = list.map_or(16, |l| {
                i64::from(l.factor(p.log2_size, p.matrix_id, i % n, i / n))
            });
            let d = ((i64::from(*c) * m * scale) + (1 << (bd_shift - 1))) >> bd_shift;
            *c = d.clamp(-32768, 32767) as i32;
        }
        // 8.6.2 equation 8-299.
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
    // 8.6.7.
    let max = (1i32 << p.bit_depth) - 1;
    for (y, res) in coeffs.chunks_exact(n).enumerate() {
        let row = frame.row_mut(c_idx, x0, y0.saturating_add(y), n)?;
        for (s, &r) in row.iter_mut().zip(res) {
            *s = i32::from(*s).saturating_add(r).clamp(0, max) as u16;
        }
    }
    Ok(())
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
}
