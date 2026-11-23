//! Sample adaptive offset (8.7.3) for a whole picture, 4:0:0 or 4:2:0.
//!
//! Each component is processed from a copy of the deblocked plane (the "recPicture" input), the
//! result is written into the frame plane. Slices and tiles are made of whole CTBs, so the
//! neighbour rules of 8.7.3.2 (different slice, MinTbAddrZs order, different tile) are decided
//! once per neighbouring CTB; for CTBs MinTbAddrZs order is CtbAddrInTs order.

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::ledger::Ledger;
use crate::ps::Active;
use crate::sao_simd::{bo_row, eo_row};
use crate::simd::{self, Tier};
use std::sync::atomic::{AtomicBool, Ordering};

/// Table 8-13: (hPos[0], vPos[0], hPos[1], vPos[1]) by SaoEoClass.
const EO_POS: [(isize, isize, isize, isize); 4] =
    [(-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (1, -1, -1, 1)];

/// Whether a sample of CTB `addr` may use samples of the CTB at (`nrx`, `nry`) for edge offset.
fn neighbour_usable(frame: &Frame, act: &Active, addr: usize, nrx: i64, nry: i64) -> bool {
    let sps = &act.sps;
    let (w, h) = (
        i64::from(sps.pic_width_in_ctbs_y),
        i64::from(sps.pic_height_in_ctbs_y),
    );
    if nrx < 0 || nry < 0 || nrx >= w || nry >= h {
        return false;
    }
    let Ok(naddr) = usize::try_from(nry * w + nrx) else {
        return false;
    };
    if naddr == addr {
        return true;
    }
    let (Some(cur), Some(nb)) = (frame.ctb_slice(addr), frame.ctb_slice(naddr)) else {
        return false;
    };
    let (Some(&cts), Some(&nts)) = (act.rs_to_ts.get(addr), act.rs_to_ts.get(naddr)) else {
        return false;
    };
    if cur.slice_addr_rs != nb.slice_addr_rs {
        // The later of the two slices in decoding order decides.
        let later = if nts < cts { cur } else { nb };
        if !later.loop_filter_across_slices {
            return false;
        }
    }
    if !act.pps.loop_filter_across_tiles_enabled {
        let tile = |ts: u32| act.tile_id.get(ts as usize);
        if tile(cts) != tile(nts) {
            return false;
        }
    }
    true
}

/// One CTB of component `c`: reads `src` (deblocked), writes `dst`.
#[allow(clippy::too_many_arguments)]
fn sao_ctb(
    tier: Tier,
    frame: &Frame,
    act: &Active,
    c: usize,
    (rx, ry): (usize, usize),
    src: &[u16],
    dst: &mut [u16],
    (w, h, stride): (usize, usize, usize),
) {
    let sps = &act.sps;
    let addr = ry * sps.pic_width_in_ctbs_y as usize + rx;
    let (Some(p), Some(sl)) = (frame.sao_at(addr), frame.ctb_slice(addr)) else {
        return;
    };
    let on = if c == 0 { sl.sao_luma } else { sl.sao_chroma };
    let (Some(&ty), Some(offs)) = (p.type_idx.get(c), p.offset_val.get(c)) else {
        return;
    };
    if !on || ty == 0 {
        return;
    }
    let (sw, sh) = if c == 0 {
        (1, 1)
    } else {
        (sps.sub_width_c as usize, sps.sub_height_c as usize)
    };
    let (cw, ch) = (sps.ctb_size_y as usize / sw, sps.ctb_size_y as usize / sh);
    let (x0, y0) = (rx * cw, ry * ch);
    let (x1, y1) = ((x0 + cw).min(w), (y0 + ch).min(h));
    let depth = if c == 0 {
        sps.bit_depth_luma
    } else {
        sps.bit_depth_chroma
    };
    let max = (1i32 << depth.min(16)) - 1;
    let offset = |i: i32| {
        usize::try_from(i)
            .ok()
            .and_then(|i| offs.get(i))
            .map_or(0, |&v| i32::from(v))
    };

    // Edge offset: which of the 3x3 neighbouring CTBs may be used.
    let mut usable = [false; 9];
    let (rxi, ryi) = (rx as i64, ry as i64);
    for (i, u) in usable.iter_mut().enumerate() {
        *u = neighbour_usable(
            frame,
            act,
            addr,
            rxi + (i % 3) as i64 - 1,
            ryi + (i / 3) as i64 - 1,
        );
    }
    let (eo_h0, eo_v0, eo_h1, eo_v1) = p
        .eo_class
        .get(c)
        .and_then(|&k| EO_POS.get(usize::from(k)))
        .copied()
        .unwrap_or((0, 0, 0, 0));
    let band_pos = p.band_position.get(c).map_or(0, |&b| i32::from(b));

    // Neighbour sample at (nx, ny), None when it is outside the picture or may not be used.
    let nb = |nx: isize, ny: isize| -> Option<i32> {
        if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
            return None;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        let dx = if nx < x0 {
            0
        } else if nx >= x0 + cw {
            2
        } else {
            1
        };
        let dy = if ny < y0 {
            0
        } else if ny >= y0 + ch {
            2
        } else {
            1
        };
        if !usable.get(dy * 3 + dx).copied().unwrap_or(false) {
            return None;
        }
        src.get(ny * stride + nx).map(|&v| i32::from(v))
    };

    // SIMD tiers: the offsets as a lookup table (`i16::MIN` cannot be negated, scalar handles it).
    let tab = (tier != Tier::Scalar && offs.iter().all(|&v| v != i16::MIN)).then(|| {
        let [o0, o1, o2, o3, o4] = *offs;
        if ty == 2 {
            [o1, o2, o0, o3, o4]
        } else {
            [o1, o2, o3, o4, 0]
        }
    });
    let shift = depth.saturating_sub(5).min(16);
    let max16 = u16::try_from(max).unwrap_or(u16::MAX);
    let step = (4 / sw).max(1);

    for y in y0..y1 {
        // The run of this row that a row kernel handles: no PCM/bypass cell, and for edge offset
        // the columns whose neighbours are inside this CTB column and picture, and a usable
        // CTB row. Everything else (and Tier::Scalar: all of it) takes the per-sample path.
        let span = tab
            .and_then(|tab| {
                let (fs, fe) = if ty == 2 {
                    (x0 + 1, x1.saturating_sub(1))
                } else {
                    (x0, x1)
                };
                if fs >= fe
                    || !(fs / step..=(fe - 1) / step)
                        .all(|cx| frame.no_filter_at(cx * step * sw, y * sh) == Some(0))
                {
                    return None;
                }
                let row = y * stride;
                let cur = src.get(row + fs..row + fe)?;
                let out = dst.get_mut(row + fs..row + fe)?;
                if ty == 2 {
                    let nbrow = |dh: isize, dv: isize| -> Option<&[u16]> {
                        let ny = usize::try_from(y as isize + dv).ok().filter(|&n| n < h)?;
                        let dy = if ny < y0 {
                            0
                        } else if ny >= y0 + ch {
                            2
                        } else {
                            1
                        };
                        if !usable.get(dy * 3 + 1).copied().unwrap_or(false) {
                            return None;
                        }
                        let start = ny * stride + usize::try_from(fs as isize + dh).ok()?;
                        src.get(start..start + (fe - fs))
                    };
                    let (a, b) = (nbrow(eo_h0, eo_v0)?, nbrow(eo_h1, eo_v1)?);
                    eo_row(tier, cur, a, b, out, &tab, max16);
                } else {
                    let bp = u16::try_from(band_pos).unwrap_or(0);
                    bo_row(tier, cur, out, &tab, bp, shift, max16);
                }
                Some(fs..fe)
            })
            .unwrap_or(x0..x0);
        for x in (x0..span.start).chain(span.end..x1) {
            // pcm with pcm_loop_filter_disabled_flag, or cu_transquant_bypass: not modified.
            if frame.no_filter_at(x * sw, y * sh).is_none_or(|f| f != 0) {
                continue;
            }
            let Some(&rec) = src.get(y * stride + x) else {
                continue;
            };
            let rec = i32::from(rec);
            let add = if ty == 2 {
                let (xi, yi) = (x as isize, y as isize);
                let (Some(a), Some(b)) = (nb(xi + eo_h0, yi + eo_v0), nb(xi + eo_h1, yi + eo_v1))
                else {
                    continue;
                };
                // (8-411), (8-412)
                let e = 2 + (rec - a).signum() + (rec - b).signum();
                offset(match e {
                    0 => 1,
                    1 => 2,
                    2 => 0,
                    e => e,
                })
            } else {
                // (8-414), (8-415): bands saoLeftClass..+3 (mod 32) map to offsets 1..=4.
                let band = rec >> depth.saturating_sub(5).min(16);
                let k = (band - band_pos) & 31;
                if k < 4 { offset(k + 1) } else { 0 }
            };
            if let Some(d) = dst.get_mut(y * stride + x) {
                *d = u16::try_from((rec + add).clamp(0, max)).unwrap_or(0);
            }
        }
    }
}

fn sao_plane(
    tier: Tier,
    frame: &mut Frame,
    act: &Active,
    c: usize,
    cancel: &AtomicBool,
) -> Result<()> {
    let (Some(&w), Some(&h), Some(&stride)) =
        (frame.width.get(c), frame.height.get(c), frame.stride.get(c))
    else {
        return Err(Error::InvalidData("plane"));
    };
    let Some(plane) = frame.planes.get_mut(c) else {
        return Err(Error::InvalidData("plane"));
    };
    let mut dst = std::mem::take(plane);
    let mut src = Vec::new();
    let r = src
        .try_reserve_exact(dst.len())
        .map_err(|_| Error::OutOfMemory)
        .and_then(|()| {
            src.extend_from_slice(&dst);
            for ry in 0..act.sps.pic_height_in_ctbs_y as usize {
                if cancel.load(Ordering::Relaxed) {
                    return Err(Error::Cancelled);
                }
                for rx in 0..act.sps.pic_width_in_ctbs_y as usize {
                    sao_ctb(
                        tier,
                        frame,
                        act,
                        c,
                        (rx, ry),
                        &src,
                        &mut dst,
                        (w, h, stride),
                    );
                }
            }
            Ok(())
        });
    if let Some(plane) = frame.planes.get_mut(c) {
        *plane = dst;
    }
    r
}

/// Applies SAO to the deblocked picture in place.
pub(crate) fn sao(
    frame: &mut Frame,
    act: &Active,
    ledger: &Ledger,
    cancel: &AtomicBool,
) -> Result<()> {
    sao_with(simd::tier(), frame, act, ledger, cancel)
}

/// [`sao`] on an explicit tier (tests run every available one).
fn sao_with(
    tier: Tier,
    frame: &mut Frame,
    act: &Active,
    ledger: &Ledger,
    cancel: &AtomicBool,
) -> Result<()> {
    for c in 0..frame.num_planes() {
        if !frame
            .sao
            .iter()
            .any(|s| s.type_idx.get(c).is_some_and(|&t| t != 0))
        {
            continue;
        }
        let bytes = frame
            .planes
            .get(c)
            .map_or(0, Vec::len)
            .checked_mul(2)
            .ok_or(Error::AllocLimit)?;
        ledger.reserve(bytes)?;
        let r = sao_plane(tier, frame, act, c, cancel);
        ledger.release(bytes);
        r?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{SaoParams, SliceParams};
    use crate::limits::Limits;
    use crate::ps::tests::{ledger, pps_bits, sps_bits};
    use crate::ps::{activate, parse_pps, parse_sps};
    use std::sync::Arc;

    fn slice(addr: u32, across: bool) -> SliceParams {
        SliceParams {
            slice_addr_rs: addr,
            qp_y: 26,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            loop_filter_across_slices: across,
            sao_luma: true,
            sao_chroma: true,
        }
    }

    /// 64x64 picture, four 32x32 CTBs: CTB 0 in slice A, CTBs 1..3 in slice B (`across_b`).
    fn setup(l: &Ledger, across_b: bool) -> (Active, Frame) {
        let sps = Arc::new(parse_sps(&sps_bits(64, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let act = activate(sps, pps, &Limits::default(), l).unwrap();
        let mut f = Frame::new(&act, l).unwrap();
        let a = f.push_slice(slice(0, true), l).unwrap();
        let b = f.push_slice(slice(1, across_b), l).unwrap();
        f.start_ctb(0, a).unwrap();
        for rs in 1..4 {
            f.start_ctb(rs, b).unwrap();
        }
        (act, f)
    }

    fn row(f: &Frame, y: usize, x0: usize, n: usize) -> Vec<u16> {
        (x0..x0 + n).map(|x| f.sample(0, x, y).unwrap()).collect()
    }

    #[test]
    fn edge_offset_horizontal() {
        let l = ledger();
        let (act, mut f) = setup(&l, true);
        f.set_sample(0, 4, 5, 120).unwrap();
        f.set_sample(0, 8, 5, 136).unwrap();
        *f.sao_at_mut(0).unwrap() = SaoParams {
            type_idx: [2, 0, 0],
            eo_class: [0, 0, 0],
            offset_val: [[0, 4, 2, -2, -4], [0; 5], [0; 5]],
            ..Default::default()
        };
        sao(&mut f, &act, &l, &AtomicBool::new(false)).unwrap();
        // Local min (edgeIdx 1: +4), local max (edgeIdx 4: -4), slopes (+2 / -2 from edgeIdx 2 / 3
        // after the remap), flat 128 untouched.
        assert_eq!(
            row(&f, 5, 2, 9),
            [128, 126, 124, 126, 128, 130, 132, 130, 128]
        );
        assert_eq!(row(&f, 4, 2, 9), [128; 9]);
        // The picture border is never a neighbour; CTB 1 has SAO off.
        assert_eq!(f.sample(0, 0, 5), Some(128));
    }

    #[test]
    fn edge_offset_slice_boundary() {
        for across in [true, false] {
            let l = ledger();
            let (act, mut f) = setup(&l, across);
            f.set_sample(0, 31, 5, 136).unwrap();
            *f.sao_at_mut(0).unwrap() = SaoParams {
                type_idx: [2, 0, 0],
                offset_val: [[0, 4, 2, -2, -4], [0; 5], [0; 5]],
                ..Default::default()
            };
            sao(&mut f, &act, &l, &AtomicBool::new(false)).unwrap();
            // The right neighbour is in slice B, which comes later and decides.
            assert_eq!(f.sample(0, 31, 5), Some(if across { 132 } else { 136 }));
        }
    }

    #[test]
    fn band_offset_and_no_filter() {
        let l = ledger();
        let (act, mut f) = setup(&l, true);
        for (x, v) in [128, 136, 144, 152, 160, 100].into_iter().enumerate() {
            f.set_sample(0, x, 0, v).unwrap();
        }
        f.set_sample(0, 40, 0, 128).unwrap();
        *f.sao_at_mut(0).unwrap() = SaoParams {
            type_idx: [1, 0, 0],
            band_position: [16, 0, 0],
            offset_val: [[0, 3, -3, 5, -5], [0; 5], [0; 5]],
            ..Default::default()
        };
        f.fill_no_filter(0, 8, 4, 4, 1).unwrap();
        sao(&mut f, &act, &l, &AtomicBool::new(false)).unwrap();
        // bandShift 3: bands 16..19 get offsets 1..=4; band 20 and band 12 are untouched.
        assert_eq!(row(&f, 0, 0, 6), [131, 133, 149, 147, 160, 100]);
        assert_eq!(f.sample(0, 40, 0), Some(128)); // CTB 1: SaoTypeIdx 0
        assert_eq!(f.sample(0, 0, 8), Some(128)); // no_filter cell
        assert_eq!(f.sample(0, 4, 8), Some(131));
    }

    /// Random samples, SAO parameters and no-filter cells, the same for the same `seed`.
    fn random_frame(across: bool, seed: u64) -> (Active, Frame) {
        let l = ledger();
        let (act, mut f) = setup(&l, across);
        let mut s = seed;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let extreme = seed % 3 == 0;
        for c in 0..f.num_planes() {
            for y in 0..f.height[c] {
                for x in 0..f.width[c] {
                    let v = if extreme {
                        [0, 255, 1, 254][(rnd() % 4) as usize]
                    } else {
                        128 + (rnd() % 5) as u16
                    };
                    f.set_sample(c, x, y, v).unwrap();
                }
            }
        }
        for addr in 0..4 {
            let mut p = SaoParams::default();
            for c in 0..3 {
                p.type_idx[c] = (rnd() % 3) as u8;
                p.band_position[c] = (rnd() % 32) as u8;
                p.eo_class[c] = (rnd() % 4) as u8;
                for o in &mut p.offset_val[c] {
                    *o = match rnd() % 5 {
                        0 => i16::MAX,
                        1 => -i16::MAX,
                        _ => (rnd() % 17) as i16 - 8,
                    };
                }
            }
            *f.sao_at_mut(addr).unwrap() = p;
        }
        for _ in 0..6 {
            let (x, y) = ((rnd() % 48) as usize, (rnd() % 56) as usize);
            f.fill_no_filter(x, y, 1 + (rnd() % 8) as usize, 4, 1)
                .unwrap();
        }
        (act, f)
    }

    #[test]
    fn tiers_equal_scalar() {
        let l = ledger();
        for seed in 1..40u64 {
            let (act, mut want) = random_frame(seed % 2 == 0, seed * 0x9e37_79b9);
            sao_with(Tier::Scalar, &mut want, &act, &l, &AtomicBool::new(false)).unwrap();
            for t in simd::available() {
                let (act, mut got) = random_frame(seed % 2 == 0, seed * 0x9e37_79b9);
                sao_with(t, &mut got, &act, &l, &AtomicBool::new(false)).unwrap();
                assert_eq!(got.planes, want.planes, "{} seed {seed}", t.name());
            }
        }
    }

    #[test]
    fn ledger_and_cancel() {
        let l = ledger();
        let (act, mut f) = setup(&l, true);
        f.sao_at_mut(0).unwrap().type_idx = [1, 0, 0];
        assert_eq!(
            sao(&mut f, &act, &l, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        );
        assert_eq!(f.sample(0, 0, 0), Some(128)); // plane put back
        let tiny = Ledger::default();
        tiny.reset(crate::ledger::SMALL + 100, 0).unwrap();
        assert_eq!(
            sao(&mut f, &act, &tiny, &AtomicBool::new(false)),
            Err(Error::AllocLimit)
        );
        assert_eq!(tiny.left(), 100);
    }
}
