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
use crate::simd::{self, Tier};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x86;

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

/// Eight luma edge lines in structure-of-arrays form, one SIMD lane per line: `s[r][n]` is sample
/// `r` of line `n`, `[p3, p2, p1, p0, q0, q1, q2, q3]`. Flags are -1 (set) or 0. Segments of one
/// pass touch disjoint samples (edges 8 apart, 4 samples read on each side), so lines of
/// different segments can be filtered together.
#[derive(Clone, Default)]
pub(crate) struct Lines {
    s: [[i32; 8]; 8],
    tc: [i32; 8],
    strong: [i32; 8],
    dep: [i32; 8],
    deq: [i32; 8],
    no_p: [i32; 8],
    no_q: [i32; 8],
    max: i32,
}

/// Eight chroma edge lines, `[p1, p0, q0, q1]`, same layout as [`Lines`].
#[derive(Clone, Default)]
pub(crate) struct CLines {
    s: [[i32; 8]; 4],
    tc: [i32; 8],
    no_p: [i32; 8],
    no_q: [i32; 8],
    max: i32,
}

fn mask(b: bool) -> i32 {
    -i32::from(b)
}

fn flag(a: &[i32; 8], n: usize) -> bool {
    a.get(n).is_some_and(|&v| v != 0)
}

fn put(a: &mut [i32; 8], n: usize, v: i32) {
    if let Some(x) = a.get_mut(n) {
        *x = v;
    }
}

fn set_lane<const R: usize>(rows: &mut [[i32; 8]; R], n: usize, a: &[i32; R]) {
    for (row, &v) in rows.iter_mut().zip(a) {
        put(row, n, v);
    }
}

fn get_lane<const R: usize>(rows: &[[i32; 8]; R], n: usize) -> [i32; R] {
    let mut a = [0i32; R];
    for (v, row) in a.iter_mut().zip(rows) {
        *v = row.get(n).copied().unwrap_or(0);
    }
    a
}

/// The Scalar tier: [`luma_line`] on every lane.
fn luma_scalar(l: &mut Lines) {
    for n in 0..8 {
        let o = luma_line(
            get_lane(&l.s, n),
            flag(&l.strong, n),
            flag(&l.dep, n),
            flag(&l.deq, n),
            l.tc.get(n).copied().unwrap_or(0),
            l.max,
            flag(&l.no_p, n),
            flag(&l.no_q, n),
        );
        set_lane(&mut l.s, n, &o);
    }
}

/// The Scalar tier: [`chroma_line`] on every lane.
fn chroma_scalar(l: &mut CLines) {
    for n in 0..8 {
        let o = chroma_line(
            get_lane(&l.s, n),
            l.tc.get(n).copied().unwrap_or(0),
            l.max,
            flag(&l.no_p, n),
            flag(&l.no_q, n),
        );
        set_lane(&mut l.s, n, &o);
    }
}

/// Runs the luma filter of `tier` on the 8 lines. A tier the CPU lacks runs the Scalar code.
#[allow(unsafe_code)]
fn filter_luma(tier: Tier, l: &mut Lines) {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: is_x86_feature_detected! found avx2 in the guard (Avx512 is only selected
            // together with avx2, src/simd/mod.rs); the kernel has no other precondition.
            unsafe { x86::luma(l) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: is_aarch64_feature_detected! found neon in the guard (src/simd/mod.rs).
            unsafe { arm::luma(l) }
        }
        _ => luma_scalar(l),
    }
}

/// Runs the chroma filter of `tier` on the 8 lines (see [`filter_luma`]).
#[allow(unsafe_code)]
fn filter_chroma(tier: Tier, l: &mut CLines) {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: is_x86_feature_detected! found avx2 in the guard (Avx512 is only selected
            // together with avx2, src/simd/mod.rs); the kernel has no other precondition.
            unsafe { x86::chroma(l) }
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: is_aarch64_feature_detected! found neon in the guard (src/simd/mod.rs).
            unsafe { arm::chroma(l) }
        }
        _ => chroma_scalar(l),
    }
}

/// What the lines of one luma segment share.
struct Seg {
    strong: bool,
    dep: bool,
    deq: bool,
    tc: i32,
    no_p: bool,
    no_q: bool,
}

/// Lines of luma segments waiting for the filter; `bases[n]` is where line `n` came from.
struct LumaBatch {
    l: Lines,
    bases: [usize; 8],
    n: usize,
    /// Index step across the edge.
    is: usize,
    tier: Tier,
}

impl LumaBatch {
    fn new(tier: Tier, is: usize, depth: u32) -> Self {
        let l = Lines {
            max: (1i32 << depth.min(16)) - 1,
            ..Lines::default()
        };
        Self {
            l,
            bases: [0; 8],
            n: 0,
            is,
            tier,
        }
    }

    fn push(&mut self, plane: &mut [u16], base: usize, a: [i32; 8], g: &Seg) {
        let n = self.n;
        set_lane(&mut self.l.s, n, &a);
        put(&mut self.l.tc, n, g.tc);
        put(&mut self.l.strong, n, mask(g.strong));
        put(&mut self.l.dep, n, mask(g.dep));
        put(&mut self.l.deq, n, mask(g.deq));
        put(&mut self.l.no_p, n, mask(g.no_p));
        put(&mut self.l.no_q, n, mask(g.no_q));
        if let Some(b) = self.bases.get_mut(n) {
            *b = base;
        }
        self.n += 1;
        if self.n >= 8 {
            self.flush(plane);
        }
    }

    /// Filters the queued lines and writes them back.
    fn flush(&mut self, plane: &mut [u16]) {
        if self.n == 0 {
            return;
        }
        filter_luma(self.tier, &mut self.l);
        for (n, &base) in self.bases.iter().enumerate().take(self.n) {
            store(plane, base, self.is, -4, &get_lane(&self.l.s, n));
        }
        self.n = 0;
    }
}

/// 8.7.2.5.3 and 8.7.2.5.4 for the 4 lines of one luma edge segment: the decision here, the
/// lines are queued in `b`. `q0` is the index of q0 of line 0, `ks` the step along the edge.
fn luma_segment(
    b: &mut LumaBatch,
    plane: &mut [u16],
    q0: usize,
    ks: usize,
    bs: i32,
    e: &Edge,
    depth: u32,
) {
    let scale = 1i32 << depth.saturating_sub(8).min(8);
    let beta = lookup(&BETA_TAB, e.qp + (e.beta_off << 1)) * scale;
    let tc = lookup(&TC_TAB, e.qp + 2 * (bs - 1) + (e.tc_off << 1)) * scale;
    let (Some(l0), Some(l3)) = (
        load::<8>(plane, q0, b.is, -4),
        load::<8>(plane, q0 + 3 * ks, b.is, -4),
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
    let g = Seg {
        strong,
        dep: dp0 + dp3 < side,
        deq: dq0 + dq3 < side,
        tc,
        no_p: e.no_p,
        no_q: e.no_q,
    };
    for k in 0..4 {
        let base = q0 + k * ks;
        if let Some(a) = load::<8>(plane, base, b.is, -4) {
            b.push(plane, base, a, &g);
        }
    }
}

/// 8.7.2.5.5 and 8.7.2.5.8 for one chroma line `[p1, p0, q0, q1]`.
fn chroma_line(a: [i32; 4], tc: i32, max: i32, no_p: bool, no_q: bool) -> [i32; 4] {
    let [p1, p0, q0, q1] = a;
    let delta = ((((q0 - p0) << 2) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
    let p0n = if no_p { p0 } else { (p0 + delta).clamp(0, max) };
    let q0n = if no_q { q0 } else { (q0 - delta).clamp(0, max) };
    [p1, p0n, q0n, q1]
}

/// Lines of chroma segments waiting for the filter (see [`LumaBatch`]).
struct ChromaBatch {
    l: CLines,
    bases: [usize; 8],
    n: usize,
    is: usize,
    tier: Tier,
}

impl ChromaBatch {
    fn new(tier: Tier, is: usize, depth: u32) -> Self {
        let l = CLines {
            max: (1i32 << depth.min(16)) - 1,
            ..CLines::default()
        };
        Self {
            l,
            bases: [0; 8],
            n: 0,
            is,
            tier,
        }
    }

    fn push(&mut self, plane: &mut [u16], base: usize, a: [i32; 4], tc: i32, e: &Edge) {
        let n = self.n;
        set_lane(&mut self.l.s, n, &a);
        put(&mut self.l.tc, n, tc);
        put(&mut self.l.no_p, n, mask(e.no_p));
        put(&mut self.l.no_q, n, mask(e.no_q));
        if let Some(b) = self.bases.get_mut(n) {
            *b = base;
        }
        self.n += 1;
        if self.n >= 8 {
            self.flush(plane);
        }
    }

    fn flush(&mut self, plane: &mut [u16]) {
        if self.n == 0 {
            return;
        }
        filter_chroma(self.tier, &mut self.l);
        for (n, &base) in self.bases.iter().enumerate().take(self.n) {
            store(plane, base, self.is, -2, &get_lane(&self.l.s, n));
        }
        self.n = 0;
    }
}

/// The 4 lines of one chroma edge segment, queued in `b`.
fn chroma_segment(b: &mut ChromaBatch, plane: &mut [u16], at: usize, ks: usize, tc: i32, e: &Edge) {
    for k in 0..4 {
        let base = at + k * ks;
        let Some(a) = load::<4>(plane, base, b.is, -2) else {
            return;
        };
        b.push(plane, base, a, tc, e);
    }
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

fn luma_pass(
    tier: Tier,
    frame: &mut Frame,
    act: &Active,
    vertical: bool,
    cancel: &AtomicBool,
) -> Result<()> {
    let depth = act.sps.bit_depth_luma;
    let stride = frame.stride[0];
    let mut batch = LumaBatch::new(tier, if vertical { 1 } else { stride }, depth);
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
            let (p, ks) = if vertical {
                ((x - 1, y), stride)
            } else {
                ((x, y - 1), 1)
            };
            let Some(e) = edge(frame, act, (x, y), p) else {
                continue;
            };
            if let Some(plane) = frame.planes.first_mut() {
                luma_segment(&mut batch, plane, y * stride + x, ks, bs, &e, depth);
            }
        }
    }
    if let Some(plane) = frame.planes.first_mut() {
        batch.flush(plane);
    }
    Ok(())
}

fn chroma_pass(
    tier: Tier,
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
    let mut batch = ChromaBatch::new(tier, if vertical { 1 } else { stride }, depth);
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
            let (p, ks) = if vertical {
                ((x - 1, y), stride)
            } else {
                ((x, y - 1), 1)
            };
            let Some(e) = edge(frame, act, (x, y), p) else {
                continue;
            };
            let tc = lookup(&TC_TAB, qpc(e.qp + pic_offset) + 2 + (e.tc_off << 1)) * scale;
            if let Some(plane) = frame.planes.get_mut(c) {
                chroma_segment(&mut batch, plane, cy * stride + cx, ks, tc, &e);
            }
        }
    }
    if let Some(plane) = frame.planes.get_mut(c) {
        batch.flush(plane);
    }
    Ok(())
}

/// Deblocks the whole picture in place: vertical edges first, then horizontal edges.
pub(crate) fn deblock(frame: &mut Frame, act: &Active, cancel: &AtomicBool) -> Result<()> {
    deblock_with(simd::tier(), frame, act, cancel)
}

/// [`deblock`] on an explicit tier (the tests compare every tier with Scalar).
fn deblock_with(tier: Tier, frame: &mut Frame, act: &Active, cancel: &AtomicBool) -> Result<()> {
    for vertical in [true, false] {
        luma_pass(tier, frame, act, vertical, cancel)?;
        for c in 1..frame.num_planes() {
            chroma_pass(tier, frame, act, c, vertical, cancel)?;
        }
    }
    Ok(())
}

/// Reads bytes as a never-ending stream (zeros past the end).
struct Bytes<'a>(&'a [u8], usize);

impl Bytes<'_> {
    fn byte(&mut self) -> u32 {
        let v = self.0.get(self.1).copied().unwrap_or(0);
        self.1 = self.1.saturating_add(1);
        u32::from(v)
    }
}

/// Sample depths the generators draw from.
const DEPTHS: [u32; 4] = [8, 10, 12, 16];

/// Draws `N` samples of a random shape: noise, smooth, 0/max extremes or a step edge.
fn gen_samples<const N: usize>(
    rng: &mut impl FnMut() -> u32,
    mode: u32,
    max: i32,
    scale: i32,
) -> [i32; N] {
    let m = u32::try_from(max + 1).unwrap_or(1);
    let base = i32::try_from(rng() % m).unwrap_or(0);
    let off = i32::try_from(rng() % 64).unwrap_or(0) - 32;
    let mut a = [0i32; N];
    for (i, v) in a.iter_mut().enumerate() {
        let noise = i32::try_from(rng() % 9).unwrap_or(0) - 4;
        let step = if i < N / 2 { 0 } else { off * scale };
        *v = match mode {
            0 => i32::try_from(rng() % m).unwrap_or(0),
            1 => base + noise * scale,
            2 => {
                if rng() & 1 == 0 {
                    0
                } else {
                    max
                }
            }
            _ => base + step + noise.signum() * scale,
        }
        .clamp(0, max);
    }
    a
}

/// A random luma batch; `rng` is a source of 32-bit values.
fn gen_lines(rng: &mut impl FnMut() -> u32) -> Lines {
    let depth = DEPTHS.get(rng() as usize % 4).copied().unwrap_or(8);
    let (max, scale) = ((1i32 << depth) - 1, 1i32 << (depth - 8));
    let mut l = Lines {
        max,
        ..Lines::default()
    };
    let mode = rng() % 4;
    for n in 0..8 {
        set_lane(&mut l.s, n, &gen_samples::<8>(rng, mode, max, scale));
        let r = rng();
        put(&mut l.tc, n, i32::try_from(rng() % 25).unwrap_or(0) * scale);
        put(&mut l.strong, n, mask(r & 1 != 0));
        put(&mut l.dep, n, mask(r & 2 != 0));
        put(&mut l.deq, n, mask(r & 4 != 0));
        put(&mut l.no_p, n, mask(r & 24 == 24));
        put(&mut l.no_q, n, mask(r & 96 == 96));
    }
    l
}

/// A random chroma batch.
fn gen_clines(rng: &mut impl FnMut() -> u32) -> CLines {
    let depth = DEPTHS.get(rng() as usize % 4).copied().unwrap_or(8);
    let (max, scale) = ((1i32 << depth) - 1, 1i32 << (depth - 8));
    let mut l = CLines {
        max,
        ..CLines::default()
    };
    let mode = rng() % 4;
    for n in 0..8 {
        set_lane(&mut l.s, n, &gen_samples::<4>(rng, mode, max, scale));
        let r = rng();
        put(&mut l.tc, n, i32::try_from(rng() % 25).unwrap_or(0) * scale);
        put(&mut l.no_p, n, mask(r & 3 == 3));
        put(&mut l.no_q, n, mask(r & 12 == 12));
    }
    l
}

/// Every tier of this CPU against Scalar on kernel inputs drawn from `rng`; true when all match.
fn tiers_match(rng: &mut impl FnMut() -> u32) -> bool {
    let (l, c) = (gen_lines(rng), gen_clines(rng));
    let (mut lw, mut cw) = (l.clone(), c.clone());
    filter_luma(Tier::Scalar, &mut lw);
    filter_chroma(Tier::Scalar, &mut cw);
    simd::available().into_iter().all(|t| {
        let (mut a, mut b) = (l.clone(), c.clone());
        filter_luma(t, &mut a);
        filter_chroma(t, &mut b);
        a.s == lw.s && b.s == cw.s
    })
}

/// Cross-tier fuzz entry (see `__fuzz_deblock` in lib.rs): kernel inputs built from `data`.
pub(crate) fn fuzz_tiers(data: &[u8]) -> bool {
    let mut bytes = Bytes(data, 0);
    // 4 bytes make one 32-bit value.
    let mut rng = move || {
        let [a, b, c, d] = [bytes.byte(), bytes.byte(), bytes.byte(), bytes.byte()];
        a | (b << 8) | (c << 16) | (d << 24)
    };
    (0..16).all(|_| tiers_match(&mut rng))
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
    /// xorshift64*, no dependency.
    struct Xs(u64);

    impl Xs {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
    }

    #[test]
    fn tiers_are_there() {
        let avail = simd::available();
        #[cfg(target_arch = "x86_64")]
        assert!(avail.contains(&Tier::Avx2), "AVX2 not detected");
        #[cfg(target_arch = "aarch64")]
        assert!(avail.contains(&Tier::Neon), "NEON not detected");
        assert!(avail.contains(&Tier::Scalar));
    }

    /// Kernels: random and extreme lines (depth 8/10/12/16, tc 0..=24 * scale, all flag mixes).
    #[test]
    fn kernels_match_scalar() {
        let mut x = Xs(0x9E37_79B9_7F4A_7C15);
        for i in 0..20_000 {
            assert!(tiers_match(&mut || x.next()), "batch {i}");
        }
    }

    /// Fixed worst cases: all samples 0 or max, every tc, every flag combination, per tier.
    #[test]
    fn kernels_extremes() {
        for depth in DEPTHS {
            let (max, scale) = ((1i32 << depth) - 1, 1i32 << (depth - 8));
            for tc in 0..=24 {
                for pat in 0..256u32 {
                    let mut l = Lines {
                        max,
                        ..Lines::default()
                    };
                    for n in 0..8 {
                        let a: [i32; 8] =
                            std::array::from_fn(|i| if (pat >> i) & 1 == 1 { max } else { 0 });
                        set_lane(&mut l.s, n, &a);
                        put(&mut l.tc, n, tc * scale);
                        put(&mut l.strong, n, mask(n & 1 != 0));
                        put(&mut l.dep, n, mask(n & 2 != 0));
                        put(&mut l.deq, n, mask(n & 4 != 0));
                        put(&mut l.no_p, n, mask(n == 5));
                        put(&mut l.no_q, n, mask(n == 6));
                    }
                    let mut want = l.clone();
                    filter_luma(Tier::Scalar, &mut want);
                    for t in simd::available() {
                        let mut got = l.clone();
                        filter_luma(t, &mut got);
                        assert_eq!(
                            got.s,
                            want.s,
                            "{} depth {depth} tc {tc} pat {pat}",
                            t.name()
                        );
                    }
                    let mut c = CLines {
                        max,
                        ..CLines::default()
                    };
                    for n in 0..8 {
                        let a: [i32; 4] =
                            std::array::from_fn(
                                |i| if (pat >> (i + n)) & 1 == 1 { max } else { 0 },
                            );
                        set_lane(&mut c.s, n, &a);
                        put(&mut c.tc, n, tc * scale);
                        put(&mut c.no_p, n, mask(n == 2));
                        put(&mut c.no_q, n, mask(n == 3));
                    }
                    let mut want = c.clone();
                    filter_chroma(Tier::Scalar, &mut want);
                    for t in simd::available() {
                        let mut got = c.clone();
                        filter_chroma(t, &mut got);
                        assert_eq!(
                            got.s,
                            want.s,
                            "{} chroma depth {depth} tc {tc} pat {pat}",
                            t.name()
                        );
                    }
                }
            }
        }
    }

    /// 64x64 picture with random QpY, offsets, edges, no_filter cells and blocky samples.
    fn random_picture(l: &Ledger, seed: u64) -> (Active, Frame) {
        let mut r = Xs(seed);
        let (act, mut f) = setup(l, 30);
        for qy in (0..64).step_by(4) {
            for qx in (0..64).step_by(4) {
                let qp = i8::try_from(r.next() % 70).unwrap() - 10;
                f.fill_qp_y(qx, qy, 4, 4, qp).unwrap();
            }
        }
        let beta = i32::try_from(r.next() % 13).unwrap() - 6;
        let tc = i32::try_from(r.next() % 13).unwrap() - 6;
        f.slices[0].beta_offset_div2 = beta;
        f.slices[0].tc_offset_div2 = tc;
        for c in 0..3 {
            let w = f.width[c];
            let bases: Vec<i32> = (0..81)
                .map(|_| 90 + i32::try_from(r.next() % 60).unwrap())
                .collect();
            for y in 0..f.height[c] {
                for x in 0..w {
                    let noise = i32::try_from(r.next() % 5).unwrap() - 2;
                    let v = (bases[(y / 8) * 9 + x / 8] + noise).clamp(0, 255);
                    f.planes[c][y * w + x] = u16::try_from(v).unwrap();
                }
            }
        }
        for gy in 0..16 {
            for gx in 0..16 {
                let (x, y) = (gx * 4, gy * 4);
                f.fill_edge_v(x, y, 4, 4, [0, 1, 2, 2][(r.next() % 4) as usize])
                    .unwrap();
                f.fill_edge_h(x, y, 4, 4, [0, 1, 2, 2][(r.next() % 4) as usize])
                    .unwrap();
                if r.next() % 8 == 0 {
                    f.fill_no_filter(x, y, 4, 4, 1).unwrap();
                }
            }
        }
        (act, f)
    }

    /// Whole pictures: every tier gives the planes of Scalar.
    #[test]
    fn pictures_match_scalar() {
        let l = ledger();
        let cancel = AtomicBool::new(false);
        let mut changed = 0;
        for seed in 1..=60u64 {
            let (act, mut want) = random_picture(&l, seed * 0x1234_5679);
            let orig = want.planes.clone();
            deblock_with(Tier::Scalar, &mut want, &act, &cancel).unwrap();
            changed += usize::from(orig != want.planes);
            for t in simd::available() {
                let (act, mut got) = random_picture(&l, seed * 0x1234_5679);
                deblock_with(t, &mut got, &act, &cancel).unwrap();
                assert!(got.planes == want.planes, "{} seed {seed}", t.name());
                l.release(got.charged);
            }
            l.release(want.charged);
        }
        assert!(changed > 30, "the pictures are not filtered: {changed}");
    }

    #[test]
    fn fuzz_entry_runs() {
        assert!(fuzz_tiers(&[]));
        assert!(fuzz_tiers(&[0xff; 700]));
        let data: Vec<u8> = (0..4000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        assert!(fuzz_tiers(&data));
    }
}
