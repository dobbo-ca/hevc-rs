//! SAO row kernels (8.7.3.2): band offset and edge offset of one run of samples that the caller
//! has already cleared of every exclusion (picture border, slice/tile boundary, PCM/bypass).
//! `Tier::Scalar` is the plain per-sample code; AVX2 (also used for AVX-512, 16 u16 lanes are
//! enough for a row kernel that is bound by loads) and NEON work on 16-bit lanes.
//!
//! The add step is `clamp(rec + off, 0, max)` as `min(sat_sub(sat_add(rec, max(off, 0)),
//! max(-off, 0)), max)`, which is exact for `off` in `-32767..=32767` (the caller excludes
//! `i16::MIN`). The offset of a lane is looked up from a 5-entry table by byte shuffle.

use crate::simd::Tier;

/// Edge offset: `tab[edgeIdx]` with edgeIdx = 2 + sign(rec - a) + sign(rec - b) (8-411), already
/// remapped, i.e. `[SaoOffsetVal[1], [2], [0], [3], [4]]`.
fn eo_scalar(cur: &[u16], a: &[u16], b: &[u16], out: &mut [u16], tab: &[i16; 5], max: u16) {
    for (((o, &c), &a), &b) in out.iter_mut().zip(cur).zip(a).zip(b) {
        let (r, a, b) = (i32::from(c), i32::from(a), i32::from(b));
        let e = 2 + (r - a).signum() + (r - b).signum();
        let add = usize::try_from(e)
            .ok()
            .and_then(|e| tab.get(e))
            .map_or(0, |&v| i32::from(v));
        *o = u16::try_from((r + add).clamp(0, i32::from(max))).unwrap_or(0);
    }
}

/// Band offset: `tab` is `[SaoOffsetVal[1], [2], [3], [4], 0]`, indexed by
/// `min(((rec >> shift) - band_pos) & 31, 4)`.
fn bo_scalar(cur: &[u16], out: &mut [u16], tab: &[i16; 5], band_pos: u16, shift: u32, max: u16) {
    for (o, &c) in out.iter_mut().zip(cur) {
        let r = i32::from(c);
        let k = ((r >> shift.min(16)) - i32::from(band_pos)) & 31;
        let add = usize::try_from(k.min(4))
            .ok()
            .and_then(|k| tab.get(k))
            .map_or(0, |&v| i32::from(v));
        *o = u16::try_from((r + add).clamp(0, i32::from(max))).unwrap_or(0);
    }
}

/// The 5-entry table in the first 5 lanes of 8 (the rest 0).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn tab8(tab: &[i16; 5]) -> [i16; 8] {
    let [a, b, c, d, e] = *tab;
    [a, b, c, d, e, 0, 0, 0]
}

#[cfg(target_arch = "x86_64")]
mod x86k {
    use super::{bo_scalar, eo_scalar, tab8};
    use crate::simd::x86::{ld_i16x16, ld_u16x16, st_u16x16};
    use core::arch::x86_64::*;

    /// Table in both 128-bit lanes (`pshufb` works per lane).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn table(tab: &[i16; 5]) -> __m256i {
        let t = tab8(tab);
        let mut w = [0i16; 16];
        for (d, s) in w.iter_mut().zip(t.iter().chain(t.iter())) {
            *d = *s;
        }
        ld_i16x16(&w)
    }

    /// `clamp(r + tab[idx], 0, max)` for 16 lanes; `idx` holds 0..=4 per lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn apply(r: __m256i, idx: __m256i, t: __m256i, max: __m256i) -> __m256i {
        let z = _mm256_setzero_si256();
        // Byte index of lane value `i`: low byte 2i, high byte 2i + 1.
        let bi = _mm256_add_epi16(
            _mm256_mullo_epi16(idx, _mm256_set1_epi16(0x0202)),
            _mm256_set1_epi16(0x0100),
        );
        let add = _mm256_shuffle_epi8(t, bi);
        let pos = _mm256_max_epi16(add, z);
        let neg = _mm256_max_epi16(_mm256_sub_epi16(z, add), z);
        let s = _mm256_subs_epu16(_mm256_adds_epu16(r, pos), neg);
        _mm256_min_epu16(s, max)
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn eo(cur: &[u16], a: &[u16], b: &[u16], out: &mut [u16], tab: &[i16; 5], max: u16) {
        let t = table(tab);
        let mx = _mm256_set1_epi16(max as i16);
        let flip = _mm256_set1_epi16(i16::MIN);
        let two = _mm256_set1_epi16(2);
        let (cc, ct) = cur.as_chunks::<16>();
        let (ac, at) = a.as_chunks::<16>();
        let (bc, bt) = b.as_chunks::<16>();
        let (oc, ot) = out.as_chunks_mut::<16>();
        for (((c, a), b), o) in cc.iter().zip(ac).zip(bc).zip(oc.iter_mut()) {
            let r = ld_u16x16(c);
            // Unsigned compare through the sign flip; a mask is -1 where true.
            let rf = _mm256_xor_si256(r, flip);
            let af = _mm256_xor_si256(ld_u16x16(a), flip);
            let bf = _mm256_xor_si256(ld_u16x16(b), flip);
            let gt = _mm256_add_epi16(_mm256_cmpgt_epi16(rf, af), _mm256_cmpgt_epi16(rf, bf));
            let lt = _mm256_add_epi16(_mm256_cmpgt_epi16(af, rf), _mm256_cmpgt_epi16(bf, rf));
            // e = 2 + (sign(r - a) + sign(r - b)), sign = lt_mask - gt_mask.
            let e = _mm256_add_epi16(_mm256_sub_epi16(lt, gt), two);
            st_u16x16(o, apply(r, e, t, mx));
        }
        eo_scalar(ct, at, bt, ot, tab, max);
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn bo(
        cur: &[u16],
        out: &mut [u16],
        tab: &[i16; 5],
        band_pos: u16,
        shift: u32,
        max: u16,
    ) {
        let t = table(tab);
        let mx = _mm256_set1_epi16(max as i16);
        let bp = _mm256_set1_epi16(band_pos as i16);
        let m31 = _mm256_set1_epi16(31);
        let four = _mm256_set1_epi16(4);
        let cnt = _mm_cvtsi32_si128(shift.min(16) as i32);
        let (cc, ct) = cur.as_chunks::<16>();
        let (oc, ot) = out.as_chunks_mut::<16>();
        for (c, o) in cc.iter().zip(oc.iter_mut()) {
            let r = ld_u16x16(c);
            let band = _mm256_srl_epi16(r, cnt);
            let k = _mm256_and_si256(_mm256_sub_epi16(band, bp), m31);
            st_u16x16(o, apply(r, _mm256_min_epu16(k, four), t, mx));
        }
        bo_scalar(ct, ot, tab, band_pos, shift, max);
    }
}

#[cfg(target_arch = "aarch64")]
mod armk {
    use super::{bo_scalar, eo_scalar, tab8};
    use crate::simd::arm::{ld_i16x8, ld_u16x8, st_u16x8};
    use core::arch::aarch64::*;

    /// `clamp(r + tab[idx], 0, max)` for 8 lanes; `idx` holds 0..=4 per lane.
    #[inline]
    #[target_feature(enable = "neon")]
    fn apply(r: uint16x8_t, idx: uint16x8_t, t: uint8x16_t, max: uint16x8_t) -> uint16x8_t {
        let z = vdupq_n_s16(0);
        // Byte index of lane value `i`: low byte 2i, high byte 2i + 1.
        let bi = vaddq_u16(vmulq_n_u16(idx, 0x0202), vdupq_n_u16(0x0100));
        let add = vreinterpretq_s16_u8(vqtbl1q_u8(t, vreinterpretq_u8_u16(bi)));
        let pos = vreinterpretq_u16_s16(vmaxq_s16(add, z));
        let neg = vreinterpretq_u16_s16(vmaxq_s16(vnegq_s16(add), z));
        vminq_u16(vqsubq_u16(vqaddq_u16(r, pos), neg), max)
    }

    #[target_feature(enable = "neon")]
    pub(super) fn eo(cur: &[u16], a: &[u16], b: &[u16], out: &mut [u16], tab: &[i16; 5], max: u16) {
        let t = vreinterpretq_u8_s16(ld_i16x8(&tab8(tab)));
        let mx = vdupq_n_u16(max);
        let two = vdupq_n_s16(2);
        let (cc, ct) = cur.as_chunks::<8>();
        let (ac, at) = a.as_chunks::<8>();
        let (bc, bt) = b.as_chunks::<8>();
        let (oc, ot) = out.as_chunks_mut::<8>();
        for (((c, a), b), o) in cc.iter().zip(ac).zip(bc).zip(oc.iter_mut()) {
            let r = ld_u16x8(c);
            let (av, bv) = (ld_u16x8(a), ld_u16x8(b));
            // Masks are all ones (-1) where true.
            let gt = vaddq_s16(
                vreinterpretq_s16_u16(vcgtq_u16(r, av)),
                vreinterpretq_s16_u16(vcgtq_u16(r, bv)),
            );
            let lt = vaddq_s16(
                vreinterpretq_s16_u16(vcltq_u16(r, av)),
                vreinterpretq_s16_u16(vcltq_u16(r, bv)),
            );
            // e = 2 + (sign(r - a) + sign(r - b)), sign = lt_mask - gt_mask.
            let e = vreinterpretq_u16_s16(vaddq_s16(vsubq_s16(lt, gt), two));
            st_u16x8(o, apply(r, e, t, mx));
        }
        eo_scalar(ct, at, bt, ot, tab, max);
    }

    #[target_feature(enable = "neon")]
    pub(super) fn bo(
        cur: &[u16],
        out: &mut [u16],
        tab: &[i16; 5],
        band_pos: u16,
        shift: u32,
        max: u16,
    ) {
        let t = vreinterpretq_u8_s16(ld_i16x8(&tab8(tab)));
        let mx = vdupq_n_u16(max);
        let bp = vdupq_n_u16(band_pos);
        let m31 = vdupq_n_u16(31);
        let four = vdupq_n_u16(4);
        // A negative count shifts right.
        let cnt = vdupq_n_s16(-(shift.min(16) as i16));
        let (cc, ct) = cur.as_chunks::<8>();
        let (oc, ot) = out.as_chunks_mut::<8>();
        for (c, o) in cc.iter().zip(oc.iter_mut()) {
            let r = ld_u16x8(c);
            let band = vshlq_u16(r, cnt);
            let k = vandq_u16(vsubq_u16(band, bp), m31);
            st_u16x8(o, apply(r, vminq_u16(k, four), t, mx));
        }
        bo_scalar(ct, ot, tab, band_pos, shift, max);
    }
}

/// Edge offset on `out.len()` samples: `cur` are the samples, `a` and `b` their two neighbours
/// along the class direction (all the same length). `tier` must come from `simd::tier()` or
/// `simd::available()`.
#[allow(unsafe_code)]
pub(crate) fn eo_row(
    tier: Tier,
    cur: &[u16],
    a: &[u16],
    b: &[u16],
    out: &mut [u16],
    tab: &[i16; 5],
    max: u16,
) {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => {
            // SAFETY: simd::tier() / simd::available() return Avx2 or Avx512 only when
            // is_x86_feature_detected! found avx2 (src/simd/mod.rs); the kernel's slices are
            // bounds-checked safe Rust.
            unsafe { x86k::eo(cur, a, b, out, tab, max) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: simd::tier() / simd::available() return Neon only when
            // is_aarch64_feature_detected! found neon (src/simd/mod.rs).
            unsafe { armk::eo(cur, a, b, out, tab, max) }
        }
        _ => eo_scalar(cur, a, b, out, tab, max),
    }
}

/// Band offset on `out.len()` samples of `cur` (same length). `tier` as for [`eo_row`].
#[allow(unsafe_code)]
pub(crate) fn bo_row(
    tier: Tier,
    cur: &[u16],
    out: &mut [u16],
    tab: &[i16; 5],
    band_pos: u16,
    shift: u32,
    max: u16,
) {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 => {
            // SAFETY: simd::tier() / simd::available() return Avx2 or Avx512 only when
            // is_x86_feature_detected! found avx2 (src/simd/mod.rs).
            unsafe { x86k::bo(cur, out, tab, band_pos, shift, max) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => {
            // SAFETY: simd::tier() / simd::available() return Neon only when
            // is_aarch64_feature_detected! found neon (src/simd/mod.rs).
            unsafe { armk::bo(cur, out, tab, band_pos, shift, max) }
        }
        _ => bo_scalar(cur, out, tab, band_pos, shift, max),
    }
}

/// Fuzz entry (see `__fuzz_sao`): rows built from `data` run on every available tier must equal
/// the scalar result.
pub(crate) fn fuzz_tiers(data: &[u8]) -> bool {
    let Some((head, rest)) = data.split_first_chunk::<10>() else {
        return true;
    };
    let [m, depth, bp, o0, o1, o2, o3, o4, n, x] = *head;
    let tab = [o0, o1, o2, o3, o4].map(|v| i16::from(v as i8) << (x & 7));
    let max = if m & 1 == 0 {
        (1u32 << u32::from(depth % 17)).wrapping_sub(1).min(65535) as u16
    } else {
        u16::MAX
    };
    let n = usize::from(n);
    let mut samples = rest.chunks(2).map(|c| match *c {
        [lo, hi] => u16::from_le_bytes([lo, hi]),
        [lo] => u16::from(lo),
        _ => 0,
    });
    let mut take = |k: usize| -> Vec<u16> { samples.by_ref().take(k).collect() };
    let (mut cur, mut a, mut b) = (take(n), take(n), take(n));
    let n = cur.len().min(a.len()).min(b.len());
    cur.truncate(n);
    a.truncate(n);
    b.truncate(n);
    let (cur, a, b) = (&cur, &a, &b);
    let mut want_e = vec![0u16; n];
    let mut want_b = vec![0u16; n];
    eo_scalar(cur, a, b, &mut want_e, &tab, max);
    let shift = u32::from(depth % 17).saturating_sub(5);
    let band_pos = u16::from(bp & 31);
    bo_scalar(cur, &mut want_b, &tab, band_pos, shift, max);
    crate::simd::available().into_iter().all(|t| {
        let mut e = vec![0u16; n];
        let mut bo = vec![0u16; n];
        eo_row(t, cur, a, b, &mut e, &tab, max);
        bo_row(t, cur, &mut bo, &tab, band_pos, shift, max);
        e == want_e && bo == want_b
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn tiers_equal_scalar() {
        #[cfg(target_arch = "x86_64")]
        assert!(simd::available().contains(&Tier::Avx2), "AVX2 not detected");
        #[cfg(target_arch = "aarch64")]
        assert!(simd::available().contains(&Tier::Neon), "NEON not detected");
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for iter in 0..3000 {
            let n = [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100][iter % 12];
            let depth = [8u32, 10, 12, 16][(iter / 12) % 4];
            let max = u16::try_from((1u32 << depth) - 1).unwrap();
            // Extreme samples, small-range samples (many ties) and full-range ones.
            let mode = (iter / 48) % 3;
            let gen_s = |s: &mut u64| -> u16 {
                let r = xorshift(s);
                match mode {
                    0 => (r % 4) as u16 + max / 2,
                    1 => [0, max, 1, max - 1][(r % 4) as usize],
                    _ => (r % (u64::from(max) + 1)) as u16,
                }
            };
            let cur: Vec<u16> = (0..n).map(|_| gen_s(&mut s)).collect();
            let a: Vec<u16> = (0..n).map(|_| gen_s(&mut s)).collect();
            let b: Vec<u16> = (0..n).map(|_| gen_s(&mut s)).collect();
            let mut tab = [0i16; 5];
            for v in &mut tab {
                *v = match xorshift(&mut s) % 4 {
                    0 => i16::MAX,
                    1 => -i16::MAX,
                    _ => ((xorshift(&mut s) % 129) as i16 - 64) << (depth - 8),
                };
            }
            let band_pos = (xorshift(&mut s) % 32) as u16;
            let shift = depth - 5;
            let mut we = vec![0u16; n];
            let mut wb = vec![0u16; n];
            eo_scalar(&cur, &a, &b, &mut we, &tab, max);
            bo_scalar(&cur, &mut wb, &tab, band_pos, shift, max);
            for t in simd::available() {
                let mut e = vec![0u16; n];
                let mut bo = vec![0u16; n];
                eo_row(t, &cur, &a, &b, &mut e, &tab, max);
                bo_row(t, &cur, &mut bo, &tab, band_pos, shift, max);
                assert_eq!(e, we, "eo {} iter {iter}", t.name());
                assert_eq!(bo, wb, "bo {} iter {iter}", t.name());
            }
        }
    }

    #[test]
    fn scalar_row_matches_spec() {
        // 100 is a local min (edgeIdx 0 -> SaoOffsetVal[1]), 140 a local max (edgeIdx 4).
        let tab = [4, 2, 0, -2, -4];
        let mut o = [0u16; 3];
        eo_scalar(
            &[100, 140, 128],
            &[128, 128, 128],
            &[128, 128, 128],
            &mut o,
            &tab,
            255,
        );
        assert_eq!(o, [104, 136, 128]);
    }

    #[test]
    fn fuzz_entry_runs() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 37 % 251) as u8).collect();
        assert!(fuzz_tiers(&data));
        assert!(fuzz_tiers(&[]));
    }
}
