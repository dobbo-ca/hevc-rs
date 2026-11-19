//! Scaling lists: `scaling_list_data` (7.3.4), the default lists (Tables 7-5 and 7-6) and the
//! `ScalingFactor` derivation of 7.4.5. The factors are built once at parse time, so a lookup
//! during dequantisation is one array read.

use crate::bits::BitReader;
use crate::error::{Error, Result};

/// Table 7-6, intra default for 8x8 (and up), as the 8x8 matrix in raster order (symmetric).
/// The spec lists it in up-right diagonal order; `default_list` converts.
const INTRA8: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, //
    16, 16, 16, 16, 17, 19, 22, 25, //
    16, 16, 17, 18, 20, 22, 25, 29, //
    16, 16, 18, 21, 24, 27, 31, 36, //
    17, 17, 20, 24, 30, 35, 41, 47, //
    18, 19, 22, 27, 35, 44, 54, 65, //
    21, 22, 25, 31, 41, 54, 70, 88, //
    24, 25, 29, 36, 47, 65, 88, 115,
];

/// Table 7-6, inter default, same layout.
const INTER8: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, //
    16, 16, 16, 17, 18, 20, 24, 25, //
    16, 16, 17, 18, 20, 24, 25, 28, //
    16, 17, 18, 20, 24, 25, 28, 33, //
    17, 18, 20, 24, 25, 28, 33, 41, //
    18, 20, 24, 25, 28, 33, 41, 54, //
    20, 24, 25, 28, 33, 41, 54, 71, //
    24, 25, 28, 33, 41, 54, 71, 91,
];

/// 6.5.3 up-right diagonal scan of a `blk` x `blk` block (`blk` is 4 or 8): entry i is (x, y).
fn diag_scan(blk: usize) -> [(usize, usize); 64] {
    let mut out = [(0, 0); 64];
    let mut i = 0;
    for d in 0..2 * blk {
        for x in 0..=d {
            let y = d - x;
            if x < blk
                && y < blk
                && let Some(o) = out.get_mut(i)
            {
                *o = (x, y);
                i += 1;
            }
        }
    }
    out
}

/// The default `ScalingList[sizeId][matrixId]` in coefficient (scan) order: Table 7-5 for
/// sizeId 0 (flat 16), Table 7-6 otherwise (matrixId 0..=2 intra, 3..=5 inter).
fn default_list(size_id: usize, matrix_id: usize) -> [u8; 64] {
    let mut out = [16u8; 64];
    if size_id > 0 {
        let src = if matrix_id < 3 { &INTRA8 } else { &INTER8 };
        for (o, &(x, y)) in out.iter_mut().zip(diag_scan(8).iter()) {
            *o = src.get(y * 8 + x).copied().unwrap_or(16);
        }
    }
    out
}

/// All 24 default lists, index `sizeId * 6 + matrixId`.
fn default_set() -> [[u8; 64]; 24] {
    let mut lists = [[16u8; 64]; 24];
    for (i, l) in lists.iter_mut().enumerate() {
        *l = default_list(i / 6, i % 6);
    }
    lists
}

/// Writes one `ScalingFactor` matrix (row-major, `size` wide) from a coefficient list: the
/// list is placed along the diagonal scan of a 4x4 (`size` 4) or 8x8 block and replicated
/// `size / blk` times in each direction (7.4.5); `dc` replaces entry [0][0] for 16x16 and 32x32.
fn fill(dst: &mut [u8], size: usize, list: &[u8; 64], dc: Option<u8>) {
    let blk = if size == 4 { 4 } else { 8 };
    let rep = size / blk;
    let scan = diag_scan(blk);
    for (&v, &(x, y)) in list.iter().zip(scan.iter().take(blk * blk)) {
        for j in 0..rep {
            for k in 0..rep {
                if let Some(d) = dst.get_mut((y * rep + j) * size + x * rep + k) {
                    *d = v;
                }
            }
        }
    }
    if let (Some(v), Some(d)) = (dc, dst.first_mut()) {
        *d = v;
    }
}

/// `ScalingFactor` for sizeId 0..=3 (4x4 .. 32x32), matrixId 0..=5 (0..=2 intra Y/Cb/Cr,
/// 3..=5 inter). Matrices are row-major: index `y * size + x`. For 32x32 only matrixId 0 and 3
/// are coded; the others stay at the defaults and are not used (no 4:4:4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScalingList {
    f4: [[u8; 16]; 6],
    f8: [[u8; 64]; 6],
    f16: [[u8; 256]; 6],
    f32: [[u8; 1024]; 6],
}

impl ScalingList {
    /// Builds the factors from the 24 coefficient lists (index `sizeId * 6 + matrixId`) and
    /// the 24 DC values.
    fn build(lists: &[[u8; 64]; 24], dc: &[u8; 24]) -> Self {
        let mut s = Self {
            f4: [[16; 16]; 6],
            f8: [[16; 64]; 6],
            f16: [[16; 256]; 6],
            f32: [[16; 1024]; 6],
        };
        for m in 0..6 {
            let list = |size_id: usize| lists.get(size_id * 6 + m);
            let dcv = |size_id: usize| dc.get(size_id * 6 + m).copied();
            if let (Some(l), Some(f)) = (list(0), s.f4.get_mut(m)) {
                fill(f, 4, l, None);
            }
            if let (Some(l), Some(f)) = (list(1), s.f8.get_mut(m)) {
                fill(f, 8, l, None);
            }
            if let (Some(l), Some(f)) = (list(2), s.f16.get_mut(m)) {
                fill(f, 16, l, dcv(2));
            }
            if let (Some(l), Some(f)) = (list(3), s.f32.get_mut(m)) {
                fill(f, 32, l, dcv(3));
            }
        }
        s
    }

    /// The default lists (Tables 7-5 and 7-6, DC 16): used when `scaling_list_enabled_flag` is
    /// 1 and no list data is sent.
    pub fn default_lists() -> Self {
        Self::build(&default_set(), &[16; 24])
    }

    /// `scaling_list_data()` of 7.3.4 with the semantics of 7.4.5.
    pub fn parse(r: &mut BitReader) -> Result<Self> {
        let mut lists = default_set();
        let mut dc = [16u8; 24];
        for size_id in 0..4usize {
            let step = if size_id == 3 { 3 } else { 1 };
            for m in (0..6usize).step_by(step) {
                let id = size_id * 6 + m;
                let (list, dc_val) = if !r.flag()? {
                    // Copy of a reference list, or the default list when the delta is 0.
                    let delta = r.ue()? as usize;
                    if delta > m / step {
                        return Err(Error::InvalidData("scaling_list_pred_matrix_id_delta"));
                    }
                    if delta == 0 {
                        (default_list(size_id, m), 16)
                    } else {
                        let rid = id - delta * step;
                        match (lists.get(rid), dc.get(rid)) {
                            (Some(&l), Some(&d)) => (l, d),
                            _ => return Err(Error::InvalidData("scaling list reference")),
                        }
                    }
                } else {
                    let coef_num = if size_id == 0 { 16 } else { 64 };
                    let mut next = 8i32;
                    let mut dc_val = 16u8;
                    if size_id > 1 {
                        let v = r.se()?;
                        if !(-7..=247).contains(&v) {
                            return Err(Error::InvalidData("scaling_list_dc_coef_minus8"));
                        }
                        next = v + 8;
                        dc_val = next as u8;
                    }
                    let mut l = [16u8; 64];
                    for c in l.iter_mut().take(coef_num) {
                        let d = r.se()?;
                        if !(-128..=127).contains(&d) {
                            return Err(Error::InvalidData("scaling_list_delta_coef"));
                        }
                        next = (next + d + 256).rem_euclid(256);
                        *c = next as u8;
                    }
                    (l, dc_val)
                };
                if let (Some(l), Some(d)) = (lists.get_mut(id), dc.get_mut(id)) {
                    *l = list;
                    *d = dc_val;
                }
            }
        }
        Ok(Self::build(&lists, &dc))
    }

    /// m[x][y] of 8.6.4.2 for a TB of side `1 << log2_size` (2..=5): `matrix_id` is 3 * (inter)
    /// + cIdx (7.4.5; 0 or 3 for 32x32), (x, y) the position in the TB. Out of range gives 16.
    pub fn factor(&self, log2_size: u32, matrix_id: usize, x: usize, y: usize) -> u8 {
        let i = (y << log2_size.min(5)).saturating_add(x);
        let v = match log2_size {
            2 => self.f4.get(matrix_id).and_then(|f| f.get(i)),
            3 => self.f8.get(matrix_id).and_then(|f| f.get(i)),
            4 => self.f16.get(matrix_id).and_then(|f| f.get(i)),
            5 => self.f32.get(matrix_id).and_then(|f| f.get(i)),
            _ => None,
        };
        v.copied().unwrap_or(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_factors() {
        let s = ScalingList::default_lists();
        assert_eq!(s.factor(2, 0, 3, 3), 16);
        assert_eq!(s.factor(3, 0, 0, 0), 16);
        assert_eq!(s.factor(3, 0, 7, 7), 115);
        assert_eq!(s.factor(3, 3, 7, 7), 91);
        assert_eq!(s.factor(3, 0, 4, 0), 17);
        // 16x16 and 32x32: upsampled from the 8x8 list, DC 16.
        assert_eq!(s.factor(4, 0, 15, 15), 115);
        assert_eq!(s.factor(4, 0, 14, 14), 115);
        assert_eq!(s.factor(4, 0, 0, 0), 16);
        assert_eq!(s.factor(5, 3, 31, 31), 91);
        assert_eq!(s.factor(5, 0, 16, 0), 17);
        assert_eq!(s.factor(9, 0, 0, 0), 16);
    }

    #[test]
    fn diag_scan_4x4() {
        let s = diag_scan(4);
        assert_eq!(&s[..6], &[(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]);
        assert_eq!(s[15], (3, 3));
    }

    /// Bits of `scaling_list_data` where every list is "copy the default" (pred_mode 0, delta 0).
    fn all_default_bits() -> Vec<u8> {
        // 20 lists, each flag 0 + ue(0) = "01": 40 bits.
        let mut bits = String::new();
        for _ in 0..20 {
            bits.push_str("01");
        }
        bits.as_bytes()
            .chunks(8)
            .map(|c| {
                let mut b = 0u8;
                for (i, &ch) in c.iter().enumerate() {
                    if ch == b'1' {
                        b |= 0x80 >> i;
                    }
                }
                b
            })
            .collect()
    }

    #[test]
    fn parse_default_copy() {
        let data = all_default_bits();
        let mut r = BitReader::new(&data);
        assert_eq!(
            ScalingList::parse(&mut r).unwrap(),
            ScalingList::default_lists()
        );
    }
}
