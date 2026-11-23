//! Deblocking filter (8.7.2) for a whole picture, 4:0:0 or 4:2:0.
//!
//! The edge strengths (`Frame::edge_v` / `edge_h`, 0 or 2) already carry the filterEdgeFlag and
//! slice_deblocking_filter_disabled_flag decisions made while the CUs were parsed. Here: luma
//! edges on the 8x8 grid, chroma edges where the chroma sample position is a multiple of 8 and
//! bS is 2; all vertical edges of the picture first, then the horizontal edges on that result.
//! Samples of PCM CUs with pcm_loop_filter_disabled_flag and of cu_transquant_bypass CUs are kept
//! (`Frame::no_filter`).

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::ps::Active;
use std::sync::atomic::{AtomicBool, Ordering};

/// beta' of Table 8-12, indexed by Q = 0..=51.
const BETA_TAB: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// tC' of Table 8-12, indexed by Q = 0..=53.
const TC_TAB: [u8; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// QpC of Table 8-10 for qPi = 30..=43.
const QPC_TAB: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];

/// Table 8-10 (ChromaArrayType 1).
fn qpc(qpi: i32) -> i32 {
    if qpi < 30 {
        qpi
    } else if qpi > 43 {
        qpi - 6
    } else {
        usize::try_from(qpi - 30)
            .ok()
            .and_then(|i| QPC_TAB.get(i))
            .copied()
            .unwrap_or(qpi)
    }
}

/// Table 8-12 lookup with the Clip3 of (8-348) / (8-350) / (8-383) folded in.
fn lookup(t: &[u8], q: i32) -> i32 {
    let last = i32::try_from(t.len()).unwrap_or(1) - 1;
    usize::try_from(q.clamp(0, last))
        .ok()
        .and_then(|i| t.get(i))
        .map_or(0, |&v| i32::from(v))
}

/// What one 4-sample edge segment needs from the metadata.
struct Edge {
    /// qPL (8-347) before the chroma offset: (QpQ + QpP + 1) >> 1.
    qp: i32,
    beta_off: i32,
    tc_off: i32,
    no_p: bool,
    no_q: bool,
}

/// Metadata of the edge segment whose first q sample is luma (xq, yq) and whose first p sample is
/// luma (xp, yp). None when the CTB of q was never decoded.
fn edge(
    frame: &Frame,
    act: &Active,
    (xq, yq): (usize, usize),
    (xp, yp): (usize, usize),
) -> Option<Edge> {
    let ctb = Frame::ctb_addr(act, i32::try_from(xq).ok()?, i32::try_from(yq).ok()?)?;
    let s = frame.ctb_slice(ctb)?;
    let qp_q = i32::from(frame.qp_y_at(xq, yq)?);
    let qp_p = i32::from(frame.qp_y_at(xp, yp)?);
    Some(Edge {
        qp: (qp_q + qp_p + 1) >> 1,
        beta_off: s.beta_offset_div2,
        tc_off: s.tc_offset_div2,
        no_p: frame.no_filter_at(xp, yp)? != 0,
        no_q: frame.no_filter_at(xq, yq)? != 0,
    })
}

/// `N` samples across an edge: sample `i` is `plane[base + (first + i) * step]`.
fn load<const N: usize>(plane: &[u16], base: usize, step: usize, first: isize) -> Option<[i32; N]> {
    let step = isize::try_from(step).ok()?;
    let mut a = [0i32; N];
    for (slot, o) in a.iter_mut().zip(first..) {
        let idx = base.checked_add_signed(o.checked_mul(step)?)?;
        *slot = i32::from(*plane.get(idx)?);
    }
    Some(a)
}

/// Writes back what [`load`] read (values are already clipped to the sample range).
fn store<const N: usize>(plane: &mut [u16], base: usize, step: usize, first: isize, a: &[i32; N]) {
    let Ok(step) = isize::try_from(step) else {
        return;
    };
    for (&v, o) in a.iter().zip(first..) {
        let idx = o.checked_mul(step).and_then(|d| base.checked_add_signed(d));
        if let Some(s) = idx.and_then(|i| plane.get_mut(i)) {
            *s = u16::try_from(v).unwrap_or(0);
        }
    }
}

/// 8.7.2.5.6: dSam.
fn d_sam(dpq: i32, p0: i32, p3: i32, q0: i32, q3: i32, beta: i32, tc: i32) -> bool {
    dpq < (beta >> 2)
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
        && (p0 - q0).abs() < ((5 * tc + 1) >> 1)
}

/// 8.7.2.5.7 for one line `[p3, p2, p1, p0, q0, q1, q2, q3]`; `strong` is dE == 2.
#[allow(clippy::too_many_arguments)]
fn luma_line(
    a: [i32; 8],
    strong: bool,
    dep: bool,
    deq: bool,
    tc: i32,
    max: i32,
    no_p: bool,
    no_q: bool,
) -> [i32; 8] {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = a;
    let mut o = a;
    if strong {
        let t = 2 * tc;
        if !no_p {
            o[3] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - t, p0 + t);
            o[2] = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - t, p1 + t);
            o[1] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - t, p2 + t);
        }
        if !no_q {
            o[4] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - t, q0 + t);
            o[5] = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - t, q1 + t);
            o[6] = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - t, q2 + t);
        }
        return o;
    }
    let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
    if delta.abs() >= tc * 10 {
        return o;
    }
    let delta = delta.clamp(-tc, tc);
    let h = tc >> 1;
    if !no_p {
        o[3] = (p0 + delta).clamp(0, max);
        if dep {
            let d = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-h, h);
            o[2] = (p1 + d).clamp(0, max);
        }
    }
    if !no_q {
        o[4] = (q0 - delta).clamp(0, max);
        if deq {
            let d = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-h, h);
            o[5] = (q1 + d).clamp(0, max);
        }
    }
    o
}

/// 8.7.2.5.3 and 8.7.2.5.4 for the 4 lines of one luma edge segment. `q0` is the index of q0 of
/// line 0, `is` the index step across the edge, `ks` the step along it.
fn luma_segment(plane: &mut [u16], q0: usize, is: usize, ks: usize, bs: i32, e: &Edge, depth: u32) {
    let scale = 1i32 << depth.saturating_sub(8).min(8);
    let max = (1i32 << depth.min(16)) - 1;
    let beta = lookup(&BETA_TAB, e.qp + (e.beta_off << 1)) * scale;
    let tc = lookup(&TC_TAB, e.qp + 2 * (bs - 1) + (e.tc_off << 1)) * scale;
    let (Some(l0), Some(l3)) = (
        load::<8>(plane, q0, is, -4),
        load::<8>(plane, q0 + 3 * ks, is, -4),
    ) else {
        return;
    };
    let [p3_0, p2_0, p1_0, p0_0, q0_0, q1_0, q2_0, q3_0] = l0;
    let [p3_3, p2_3, p1_3, p0_3, q0_3, q1_3, q2_3, q3_3] = l3;
    let dp0 = (p2_0 - 2 * p1_0 + p0_0).abs();
    let dp3 = (p2_3 - 2 * p1_3 + p0_3).abs();
    let dq0 = (q2_0 - 2 * q1_0 + q0_0).abs();
    let dq3 = (q2_3 - 2 * q1_3 + q0_3).abs();
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    if dpq0 + dpq3 >= beta {
        return; // dE = 0
    }
    let strong = d_sam(2 * dpq0, p0_0, p3_0, q0_0, q3_0, beta, tc)
        && d_sam(2 * dpq3, p0_3, p3_3, q0_3, q3_3, beta, tc);
    let side = (beta + (beta >> 1)) >> 3;
    let (dep, deq) = (dp0 + dp3 < side, dq0 + dq3 < side);
    for k in 0..4 {
        let base = q0 + k * ks;
        if let Some(a) = load::<8>(plane, base, is, -4) {
            let o = luma_line(a, strong, dep, deq, tc, max, e.no_p, e.no_q);
            store(plane, base, is, -4, &o);
        }
    }
}

/// 8.7.2.5.5 and 8.7.2.5.8 for the 4 lines of one chroma edge segment.
fn chroma_segment(plane: &mut [u16], at: usize, is: usize, ks: usize, tc: i32, max: i32, e: &Edge) {
    for k in 0..4 {
        let base = at + k * ks;
        let Some([p1, p0, q0, q1]) = load::<4>(plane, base, is, -2) else {
            return;
        };
        let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
        let p0n = if e.no_p {
            p0
        } else {
            (p0 + delta).clamp(0, max)
        };
        let q0n = if e.no_q {
            q0
        } else {
            (q0 - delta).clamp(0, max)
        };
        store(plane, base, is, -2, &[p1, p0n, q0n, q1]);
    }
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

fn luma_pass(frame: &mut Frame, act: &Active, vertical: bool, cancel: &AtomicBool) -> Result<()> {
    let depth = act.sps.bit_depth_luma;
    let stride = frame.stride[0];
    for gy in 0..frame.grid_h {
        check(cancel)?;
        for gx in 0..frame.grid_w {
            let (x, y) = (gx * 4, gy * 4);
            let (bs, along) = if vertical {
                (frame.edge_v_at(x, y), x)
            } else {
                (frame.edge_h_at(x, y), y)
            };
            // 8x8 grid only; the picture border is not an edge.
            if !along.is_multiple_of(8) || along < 8 {
                continue;
            }
            let bs = i32::from(bs.unwrap_or(0));
            if bs == 0 {
                continue;
            }
            let (p, is, ks) = if vertical {
                ((x - 1, y), 1, stride)
            } else {
                ((x, y - 1), stride, 1)
            };
            let Some(e) = edge(frame, act, (x, y), p) else {
                continue;
            };
            if let Some(plane) = frame.planes.first_mut() {
                luma_segment(plane, y * stride + x, is, ks, bs, &e, depth);
            }
        }
    }
    Ok(())
}

fn chroma_pass(
    frame: &mut Frame,
    act: &Active,
    c: usize,
    vertical: bool,
    cancel: &AtomicBool,
) -> Result<()> {
    let (Some(&w), Some(&h), Some(&stride)) =
        (frame.width.get(c), frame.height.get(c), frame.stride.get(c))
    else {
        return Err(Error::InvalidData("chroma plane"));
    };
    // cQpPicOffset: the PPS offset alone.
    let pic_offset = if c == 1 {
        act.pps.cb_qp_offset
    } else {
        act.pps.cr_qp_offset
    };
    let depth = act.sps.bit_depth_chroma;
    let scale = 1i32 << depth.saturating_sub(8).min(8);
    let max = (1i32 << depth.min(16)) - 1;
    // Segments of 4 chroma samples along the edge (8 luma samples in 4:2:0).
    for cy in (0..h).step_by(4) {
        check(cancel)?;
        for cx in (0..w).step_by(4) {
            let across = if vertical { cx } else { cy };
            if !across.is_multiple_of(8) || across < 8 {
                continue;
            }
            let (x, y) = (cx * 2, cy * 2);
            let bs = if vertical {
                frame.edge_v_at(x, y)
            } else {
                frame.edge_h_at(x, y)
            };
            if bs != Some(2) {
                continue;
            }
            let (p, is, ks) = if vertical {
                ((x - 1, y), 1, stride)
            } else {
                ((x, y - 1), stride, 1)
            };
            let Some(e) = edge(frame, act, (x, y), p) else {
                continue;
            };
            let tc = lookup(&TC_TAB, qpc(e.qp + pic_offset) + 2 + (e.tc_off << 1)) * scale;
            if let Some(plane) = frame.planes.get_mut(c) {
                chroma_segment(plane, cy * stride + cx, is, ks, tc, max, &e);
            }
        }
    }
    Ok(())
}

/// Deblocks the whole picture in place: vertical edges first, then horizontal edges.
pub(crate) fn deblock(frame: &mut Frame, act: &Active, cancel: &AtomicBool) -> Result<()> {
    for vertical in [true, false] {
        luma_pass(frame, act, vertical, cancel)?;
        for c in 1..frame.num_planes() {
            chroma_pass(frame, act, c, vertical, cancel)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::SliceParams;
    use crate::ledger::Ledger;
    use crate::limits::Limits;
    use crate::ps::tests::{ledger, pps_bits, sps_bits};
    use crate::ps::{activate, parse_pps, parse_sps};
    use std::sync::Arc;

    /// 64x64 4:2:0 picture (one slice, four 32x32 CTBs), every QpY set to `qp`.
    fn setup(l: &Ledger, qp: i8) -> (Active, Frame) {
        let sps = Arc::new(parse_sps(&sps_bits(64, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let act = activate(sps, pps, &Limits::default(), l).unwrap();
        let mut f = Frame::new(&act, l).unwrap();
        let s = SliceParams {
            slice_addr_rs: 0,
            qp_y: 26,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            loop_filter_across_slices: true,
            sao_luma: true,
            sao_chroma: true,
        };
        let i = f.push_slice(s, l).unwrap();
        for rs in 0..4 {
            f.start_ctb(rs, i).unwrap();
        }
        f.fill_qp_y(0, 0, 64, 64, qp).unwrap();
        (act, f)
    }

    fn row(f: &Frame, c: usize, y: usize, x0: usize, n: usize) -> Vec<u16> {
        (x0..x0 + n).map(|x| f.sample(c, x, y).unwrap()).collect()
    }

    #[test]
    fn strong_vertical_edge() {
        let l = ledger();
        let (act, mut f) = setup(&l, 40);
        for y in 0..64 {
            for x in 8..64 {
                f.set_sample(0, x, y, 104).unwrap();
            }
            for x in 0..8 {
                f.set_sample(0, x, y, 100).unwrap();
            }
        }
        f.fill_edge_v(8, 0, 4, 4, 2).unwrap();
        deblock(&mut f, &act, &AtomicBool::new(false)).unwrap();
        // QpL 40: beta 42, tC 7 (Q 42); flat sides give dE 2, strong filter by hand.
        assert_eq!(
            row(&f, 0, 0, 4, 8),
            [100, 101, 101, 102, 103, 103, 104, 104]
        );
        assert_eq!(
            row(&f, 0, 3, 4, 8),
            [100, 101, 101, 102, 103, 103, 104, 104]
        );
        // The next segment has no edge mark.
        assert_eq!(
            row(&f, 0, 4, 4, 8),
            [100, 100, 100, 100, 104, 104, 104, 104]
        );
    }

    #[test]
    fn weak_horizontal_edge_and_unfiltered_side() {
        let l = ledger();
        let (act, mut f) = setup(&l, 30);
        for y in 0..64 {
            for x in 0..64 {
                f.set_sample(0, x, y, if y < 8 { 100 } else { 108 })
                    .unwrap();
            }
        }
        f.fill_edge_h(0, 8, 4, 4, 2).unwrap();
        let mut g = Frame::new(&act, &l).unwrap();
        g.planes = f.planes.clone();
        g.qp_y = f.qp_y.clone();
        g.slice_idx = f.slice_idx.clone();
        g.slices = f.slices.clone();
        g.edge_h = f.edge_h.clone();
        // pcm / bypass above the edge: p side stays.
        g.fill_no_filter(0, 0, 64, 8, 1).unwrap();
        let col = |f: &Frame| -> Vec<u16> { (4..12).map(|y| f.sample(0, 0, y).unwrap()).collect() };
        deblock(&mut f, &act, &AtomicBool::new(false)).unwrap();
        deblock(&mut g, &act, &AtomicBool::new(false)).unwrap();
        // QpL 30: beta 22, tC 3 (Q 32); |p0-q0| = 8 is not < 8, so weak.
        // delta = 3, p0 103, q0 105; p1 += 1, q1 -= 1.
        assert_eq!(col(&f), [100, 100, 101, 103, 105, 107, 108, 108]);
        assert_eq!(col(&g), [100, 100, 100, 100, 105, 107, 108, 108]);
        l.release(g.charged);
    }

    #[test]
    fn chroma_edge_uses_pps_offset() {
        let l = ledger();
        let (act, mut f) = setup(&l, 40);
        for c in 1..3 {
            for y in 0..32 {
                for x in 0..32 {
                    f.set_sample(c, x, y, if x < 8 { 100 } else { 104 })
                        .unwrap();
                }
            }
        }
        // Chroma x 8 is luma x 16; one 8-row luma segment.
        f.fill_edge_v(16, 0, 4, 8, 2).unwrap();
        deblock(&mut f, &act, &AtomicBool::new(false)).unwrap();
        // Cb: qPi 41 -> QpC 36 -> Q 38 -> tC 5; delta = 2 (not clipped). Cr (offset -1): tC 4.
        for c in 1..3 {
            assert_eq!(row(&f, c, 3, 6, 4), [100, 102, 102, 104]);
            assert_eq!(row(&f, c, 4, 6, 4), [100, 100, 104, 104]);
        }
    }

    #[test]
    fn cancel_and_tables() {
        let l = ledger();
        let (act, mut f) = setup(&l, 40);
        assert_eq!(
            deblock(&mut f, &act, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        );
        assert_eq!(
            (
                lookup(&BETA_TAB, 51),
                lookup(&BETA_TAB, 99),
                lookup(&BETA_TAB, -5)
            ),
            (64, 64, 0)
        );
        assert_eq!(
            (
                lookup(&TC_TAB, 53),
                lookup(&TC_TAB, 18),
                lookup(&TC_TAB, 17)
            ),
            (24, 1, 0)
        );
        assert_eq!(
            (qpc(29), qpc(30), qpc(43), qpc(44), qpc(-3)),
            (29, 29, 37, 38, -3)
        );
    }
}
