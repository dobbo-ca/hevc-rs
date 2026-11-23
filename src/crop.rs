//! Output crop: copies the conformance window out of a `u16` plane and, for 8-bit components,
//! packs it to `u8`. The pack is the only SIMD kernel (docs/safety.md, "Crop"); `u16` rows are a
//! plain `extend_from_slice` (memcpy). The cancel flag is read once per row.
//!
//! Tiers: Scalar is the oracle (`s as u8`, truncating). Avx2 masks to the low byte and uses
//! `packus` plus a lane permute. Avx512 uses `vpmovwb` (truncating). Neon uses `vmovn`
//! (truncating). All four give the same bytes for every `u16` input.

use crate::Samples;
use crate::error::{Error, Result};
use crate::simd::{self, Tier};
use std::sync::atomic::{AtomicBool, Ordering};

fn pack_scalar(dst: &mut [u8], src: &[u16]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as u8;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86k {
    use crate::simd::x86::{ld_u16x16, ld_u16x32, st_u8x32};
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16,
        _mm512_cvtepi16_epi8,
    };

    /// 32 samples per step: mask to the low byte (so `packus` never saturates), pack the two
    /// halves (which interleaves the 128-bit lanes), then restore the order.
    #[target_feature(enable = "avx2")]
    pub(super) fn pack_avx2(dst: &mut [u8], src: &[u16]) {
        let mask = _mm256_set1_epi16(0xFF);
        let (sc, st) = src.as_chunks::<32>();
        let (dc, dt) = dst.as_chunks_mut::<32>();
        for (d, s) in dc.iter_mut().zip(sc) {
            if let [a, b] = s.as_chunks::<16>().0 {
                let a = _mm256_and_si256(ld_u16x16(a), mask);
                let b = _mm256_and_si256(ld_u16x16(b), mask);
                let p = _mm256_packus_epi16(a, b);
                st_u8x32(d, _mm256_permute4x64_epi64::<0xD8>(p));
            }
        }
        super::pack_scalar(dt, st);
    }

    /// 32 samples per step with the truncating `vpmovwb`.
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
    pub(super) fn pack_avx512(dst: &mut [u8], src: &[u16]) {
        let (sc, st) = src.as_chunks::<32>();
        let (dc, dt) = dst.as_chunks_mut::<32>();
        for (d, s) in dc.iter_mut().zip(sc) {
            st_u8x32(d, _mm512_cvtepi16_epi8(ld_u16x32(s)));
        }
        super::pack_scalar(dt, st);
    }
}

#[cfg(target_arch = "aarch64")]
mod neonk {
    use crate::simd::arm::{ld_u16x8, st_u8x16};
    use core::arch::aarch64::{vmovn_high_u16, vmovn_u16};

    /// 16 samples per step: `vmovn` the low half, `vmovn_high` the upper half.
    #[target_feature(enable = "neon")]
    pub(super) fn pack_neon(dst: &mut [u8], src: &[u16]) {
        let (sc, st) = src.as_chunks::<16>();
        let (dc, dt) = dst.as_chunks_mut::<16>();
        for (d, s) in dc.iter_mut().zip(sc) {
            if let [a, b] = s.as_chunks::<8>().0 {
                st_u8x16(d, vmovn_high_u16(vmovn_u16(ld_u16x8(a)), ld_u16x8(b)));
            }
        }
        super::pack_scalar(dt, st);
    }
}

/// `dst[i] = src[i] as u8` for the common length of the two slices, on `tier`. `tier` must be
/// one of `simd::available()` (the kernels need the CPU features that list proves).
#[allow(unsafe_code)]
fn pack_u8_with(tier: Tier, dst: &mut [u8], src: &[u16]) {
    let n = dst.len().min(src.len());
    let (Some(dst), Some(src)) = (dst.get_mut(..n), src.get(..n)) else {
        return;
    };
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => {
            // SAFETY: simd::tier()/available() list Avx2 only when is_x86_feature_detected!
            // found "avx2" (src/simd/mod.rs); both slices are trimmed to one length above.
            unsafe { x86k::pack_avx2(dst, src) }
        }
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512 => {
            // SAFETY: simd::tier()/available() list Avx512 only when is_x86_feature_detected!
            // found avx2, avx512f, avx512bw and avx512vl (src/simd/mod.rs).
            unsafe { x86k::pack_avx512(dst, src) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: simd::tier()/available() list Neon only when
            // is_aarch64_feature_detected! found "neon" (src/simd/mod.rs).
            unsafe { neonk::pack_neon(dst, src) }
        }
        _ => pack_scalar(dst, src),
    }
}

/// Copies the window `(x0, y0, w, h)` of `src` (row stride `stride`) into a new `Samples`: `U8`
/// when `eight_bit`, else `U16`. Every row is checked against the plane; rows missing from the
/// plane are not produced. `cancel` is read before each row: `Error::Cancelled` when set.
pub(crate) fn crop_plane(
    src: &[u16],
    stride: usize,
    window: (usize, usize, usize, usize),
    eight_bit: bool,
    cancel: &AtomicBool,
) -> Result<Samples> {
    crop_plane_with(simd::tier(), src, stride, window, eight_bit, cancel)
}

/// [`crop_plane`] on an explicit tier (the cross-tier tests and the fuzz target).
pub(crate) fn crop_plane_with(
    tier: Tier,
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
            let r = row.get(x0..x1).ok_or_else(bad)?;
            let at = v.len();
            v.resize(at + r.len(), 0);
            pack_u8_with(tier, v.get_mut(at..).unwrap_or_default(), r);
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

/// Cross-tier check for the fuzz target: the input picks a window and fills a plane; true when
/// every available tier returns what Scalar returns. Sizes are bounded (at most 2970 samples).
pub(crate) fn fuzz_tiers(data: &[u8]) -> bool {
    let Some((hdr, rest)) = data.split_at_checked(5) else {
        return true;
    };
    let &[a, b, c, d, e] = hdr else {
        return true;
    };
    let w = 1 + (usize::from(a) | (usize::from(b) << 8)) % 300;
    let h = 1 + usize::from(c) % 6;
    let (x0, y0) = (usize::from(d) % 16, usize::from(e) % 4);
    let stride = x0 + w + usize::from(d >> 4);
    let eight_bit = e & 0x80 != 0;
    let mut words = rest
        .chunks(2)
        .map(|p| p.iter().fold(0u16, |acc, &x| (acc << 8) | u16::from(x)))
        .cycle();
    let src: Vec<u16> = (0..stride * (y0 + h))
        .map(|_| words.next().unwrap_or(0))
        .collect();
    let no = AtomicBool::new(false);
    let want = crop_plane_with(Tier::Scalar, &src, stride, (x0, y0, w, h), eight_bit, &no);
    simd::available()
        .into_iter()
        .all(|t| crop_plane_with(t, &src, stride, (x0, y0, w, h), eight_bit, &no) == want)
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

    /// The plain definition of the crop, independent of `crop_plane_with`.
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
    fn tiers_available_on_ci() {
        let avail = simd::available();
        #[cfg(target_arch = "x86_64")]
        assert!(avail.contains(&Tier::Avx2), "AVX2 not detected");
        #[cfg(target_arch = "aarch64")]
        assert!(avail.contains(&Tier::Neon), "NEON not detected");
        assert!(avail.contains(&Tier::Scalar));
    }

    #[test]
    fn pack_matches_scalar_every_length_and_offset() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        // Samples of 8, 10, 12 and 16 bits, plus the saturation edges 0x00FF, 0x0100, 0xFFFF.
        for mask in [0x00FFu16, 0x03FF, 0x0FFF, 0xFFFF] {
            let src: Vec<u16> = (0..200)
                .map(|i| match i % 7 {
                    0 => 0x00FF,
                    1 => 0x0100,
                    2 => mask,
                    _ => xorshift(&mut s) as u16 & mask,
                })
                .collect();
            for off in 0..5 {
                for len in 0..=130 {
                    let r = &src[off..off + len];
                    let want: Vec<u8> = r.iter().map(|&v| v as u8).collect();
                    for t in simd::available() {
                        let mut got = vec![0xAAu8; len + 3];
                        pack_u8_with(t, &mut got[..len], r);
                        assert_eq!(got[..len], want[..], "{} len {len} off {off}", t.name());
                        assert_eq!(got[len..], [0xAA; 3], "{} wrote past the end", t.name());
                    }
                }
            }
        }
    }

    #[test]
    fn crop_matches_scalar_and_naive() {
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
                let want = naive(&src, stride, win, eight);
                for t in simd::available() {
                    let got = crop_plane_with(t, &src, stride, win, eight, &no).unwrap();
                    assert_eq!(got, want, "{} {win:?} stride {stride}", t.name());
                }
            }
        }
    }

    #[test]
    fn hostile_windows_are_errors_not_panics() {
        let no = AtomicBool::new(false);
        let src = [0u16; 16];
        for t in simd::available() {
            let bad = |r: Result<Samples>| assert_eq!(r.unwrap_err(), Error::InvalidData("plane"));
            bad(crop_plane_with(t, &src, 0, (0, 0, 4, 4), true, &no));
            // Window wider than the row.
            bad(crop_plane_with(t, &src, 4, (2, 0, 4, 1), true, &no));
            bad(crop_plane_with(
                t,
                &src,
                4,
                (usize::MAX, 0, 4, 1),
                false,
                &no,
            ));
            // Taller than the plane: the rows that exist are produced, none is read past the end.
            let r = crop_plane_with(t, &src, 4, (0, 2, 4, 9), false, &no).unwrap();
            assert_eq!(r, Samples::U16(vec![0; 8]));
        }
    }

    #[test]
    fn crop_stops_when_cancelled() {
        let src = [1u16; 64];
        let set = AtomicBool::new(true);
        for t in simd::available() {
            for eight in [true, false] {
                let r = crop_plane_with(t, &src, 8, (0, 0, 8, 8), eight, &set);
                assert_eq!(r.unwrap_err(), Error::Cancelled);
            }
        }
    }

    #[test]
    fn fuzz_entry_agrees() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        for len in 0..40usize {
            let data: Vec<u8> = (0..len * 13).map(|_| xorshift(&mut s) as u8).collect();
            assert!(fuzz_tiers(&data));
        }
    }
}
