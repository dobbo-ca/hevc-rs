//! Output crop: copies the conformance window out of a `u16` plane and, for 8-bit components,
//! narrows it to `u8`. The cancel flag is read once per row. A SIMD pack was tried and removed:
//! it gave no measurable gain (docs/safety.md, "Kernels removed").

use crate::Samples;
use crate::error::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Copies the window `(x0, y0, w, h)` of `src` (row stride `stride`) into a new `Samples`: `U8`
/// when `eight_bit`, else `U16`. Every row is checked against the plane; rows missing from the
/// plane are not produced. `cancel` is read before each row: `Error::Cancelled` when set.
pub(crate) fn crop_plane(
    src: &[u16],
    stride: usize,
    (x0, y0, w, h): (usize, usize, usize, usize),
    eight_bit: bool,
    cancel: &AtomicBool,
) -> Result<Samples> {
    let bad = || Error::InvalidData("plane");
    if stride == 0 {
        return Err(bad());
    }
    let n = w.checked_mul(h).ok_or(Error::AllocLimit)?;
    let x1 = x0.checked_add(w).ok_or_else(bad)?;
    let rows = src.chunks_exact(stride).skip(y0).take(h);
    if eight_bit {
        let mut v: Vec<u8> = Vec::new();
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        for row in rows {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            v.extend(row.get(x0..x1).ok_or_else(bad)?.iter().map(|&s| s as u8));
        }
        Ok(Samples::U8(v))
    } else {
        let mut v: Vec<u16> = Vec::new();
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        for row in rows {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            v.extend_from_slice(row.get(x0..x1).ok_or_else(bad)?);
        }
        Ok(Samples::U16(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// The plain definition of the crop, independent of `crop_plane`.
    fn naive(
        src: &[u16],
        stride: usize,
        (x0, y0, w, h): (usize, usize, usize, usize),
        eight: bool,
    ) -> Samples {
        let mut p = Vec::new();
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                p.push(src[y * stride + x]);
            }
        }
        if eight {
            Samples::U8(p.iter().map(|&s| s as u8).collect())
        } else {
            Samples::U16(p)
        }
    }

    #[test]
    fn crop_matches_naive() {
        let no = AtomicBool::new(false);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..400 {
            let w = 1 + xorshift(&mut s) as usize % 100;
            let h = 1 + xorshift(&mut s) as usize % 5;
            let (x0, y0) = (xorshift(&mut s) as usize % 9, xorshift(&mut s) as usize % 4);
            let stride = x0 + w + xorshift(&mut s) as usize % 9;
            let rows = y0 + h + xorshift(&mut s) as usize % 3;
            let src: Vec<u16> = (0..stride * rows)
                .map(|_| xorshift(&mut s) as u16)
                .collect();
            for eight in [true, false] {
                let win = (x0, y0, w, h);
                let got = crop_plane(&src, stride, win, eight, &no).unwrap();
                assert_eq!(
                    got,
                    naive(&src, stride, win, eight),
                    "{win:?} stride {stride}"
                );
            }
        }
    }

    #[test]
    fn hostile_windows_are_errors_not_panics() {
        let no = AtomicBool::new(false);
        let src = [0u16; 16];
        let bad = |r: Result<Samples>| assert_eq!(r.unwrap_err(), Error::InvalidData("plane"));
        bad(crop_plane(&src, 0, (0, 0, 4, 4), true, &no));
        // Window wider than the row.
        bad(crop_plane(&src, 4, (2, 0, 4, 1), true, &no));
        bad(crop_plane(&src, 4, (usize::MAX, 0, 4, 1), false, &no));
        // Taller than the plane: the rows that exist are produced, none is read past the end.
        let r = crop_plane(&src, 4, (0, 2, 4, 9), false, &no).unwrap();
        assert_eq!(r, Samples::U16(vec![0; 8]));
    }

    #[test]
    fn crop_stops_when_cancelled() {
        let src = [1u16; 64];
        let set = AtomicBool::new(true);
        for eight in [true, false] {
            let r = crop_plane(&src, 8, (0, 0, 8, 8), eight, &set);
            assert_eq!(r.unwrap_err(), Error::Cancelled);
        }
    }
}
