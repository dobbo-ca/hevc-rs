//! Intra sample prediction (8.4.4.2) for ChromaArrayType 0 and 1.

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::ps::Active;
use crate::simd::{self, Tier};

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x86;

/// Largest transform block (32) gives 4 * 32 + 1 = 129 reference samples; 7 more slots of
/// zero padding keep the 8-lane loads of the smoothing kernel inside the array.
const MAX_REF: usize = 136;

/// Length of the `ref[]` array of the angular modes: x = -32..=64 at index x + 32, plus padding
/// for the 8-lane loads of the angular kernels.
const RF_LEN: usize = 112;

/// A predicted block before the clamp, [y][x].
type Block = [[i32; 32]; 32];
/// A predicted block, [y][x]; the columns at and past the block size are scratch.
type Pred = [[u16; 32]; 32];

/// intraPredAngle (Table 8-5) for predModeIntra 2..=34, index = mode - 2.
const INTRA_PRED_ANGLE: [i32; 33] = [
    32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9,
    -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

/// invAngle (Table 8-6) for predModeIntra 11..=25, index = mode - 11.
const INV_ANGLE: [i32; 15] = [
    -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096,
];

fn bad() -> Error {
    Error::InvalidData("intra prediction block outside the picture")
}

fn at(a: &[i32], i: usize) -> i32 {
    a.get(i).copied().unwrap_or(0)
}

/// The neighbouring samples p[][] as one line of `4 * n + 1` values, in the order of the
/// substitution search of 8.4.4.2.2: index 0 is p[-1][2n-1], index 2n - 1 - y is p[-1][y],
/// index 2n is p[-1][-1] and index 2n + 1 + x is p[x][-1].
#[derive(Clone)]
struct Refs {
    r: [i32; MAX_REF],
    n: usize,
}

impl Refs {
    /// p[-1][-1 + k] (k = 0 is the corner).
    fn left(&self, k: usize) -> i32 {
        at(&self.r, (2 * self.n).wrapping_sub(k))
    }
    /// p[-1 + k][-1] (k = 0 is the corner).
    fn top(&self, k: usize) -> i32 {
        at(&self.r, 2 * self.n + k)
    }
}

/// 8.4.4.2: predicts the `1 << log2_size` square block at (x0, y0) of component `c_idx` and
/// writes it into the frame. `mode` is IntraPredModeY or IntraPredModeC.
pub(crate) fn predict(
    frame: &mut Frame,
    act: &Active,
    c_idx: usize,
    x0: usize,
    y0: usize,
    log2_size: u32,
    mode: u32,
) -> Result<()> {
    predict_with(simd::tier(), frame, act, c_idx, x0, y0, log2_size, mode)
}

/// `predict` on an explicit tier (tests run every available one).
#[allow(clippy::too_many_arguments)]
fn predict_with(
    tier: Tier,
    frame: &mut Frame,
    act: &Active,
    c_idx: usize,
    x0: usize,
    y0: usize,
    log2_size: u32,
    mode: u32,
) -> Result<()> {
    if !(2..=5).contains(&log2_size) || mode > 34 || c_idx >= frame.num_planes() {
        return Err(Error::InvalidData("bad intra prediction parameters"));
    }
    let n = 1usize << log2_size;
    // The whole block must lie inside the plane (checked via its last row).
    frame.row_mut(c_idx, x0, y0.checked_add(n - 1).ok_or_else(bad)?, n)?;

    let (bit_depth, sub) = if c_idx == 0 {
        (act.sps.bit_depth_luma, 1)
    } else {
        (act.sps.bit_depth_chroma, 2) // SubWidthC = SubHeightC = 2 for 4:2:0
    };
    let mut p = Refs { r: [0; MAX_REF], n };
    gather(frame, act, c_idx, x0, y0, n, sub, bit_depth, &mut p)?;

    let strong = act.sps.strong_intra_smoothing_enabled;
    let pred = compute(tier, &mut p, mode, log2_size, c_idx == 0, strong, bit_depth)?;
    for (y, row) in pred.iter().take(n).enumerate() {
        let dst = frame.row_mut(c_idx, x0, y0 + y, n)?;
        for (d, &v) in dst.iter_mut().zip(row) {
            *d = v;
        }
    }
    Ok(())
}

/// Everything after the reference gather: filtering (luma only, 8.4.4.2.3), the prediction
/// proper and the clamp to the bit depth. `p.n` is `1 << log2`. The samples of the returned
/// block are valid for x, y < `p.n`.
fn compute(
    tier: Tier,
    p: &mut Refs,
    mode: u32,
    log2: u32,
    luma: bool,
    strong: bool,
    bit_depth: u32,
) -> Result<Pred> {
    if luma {
        filter(tier, p, mode, strong, bit_depth);
    }
    let max = (1i32 << bit_depth) - 1;
    let mut pred: Block = [[0; 32]; 32]; // [y][x]
    match mode {
        0 => planar_with(tier, p, &mut pred, log2),
        1 => dc(p, &mut pred, log2, luma),
        _ => angular(tier, p, &mut pred, mode, luma, max)?,
    }
    let mut out: Pred = [[0; 32]; 32];
    store(tier, &pred, &mut out, p.n, max);
    Ok(out)
}

/// One vector kernel call, for `run`.
enum Op<'a> {
    /// (8-41)..(8-45) on `len` samples of `src`, into `dst`.
    Smooth {
        src: &'a [i32; MAX_REF],
        dst: &'a mut [i32; MAX_REF],
        len: usize,
    },
    /// 8.4.4.2.4.
    Planar {
        p: &'a Refs,
        out: &'a mut Block,
        log2: u32,
    },
    /// The interpolation of 8.4.4.2.6: `out[a][b]` for a, b < n, from `rf` (see `angular`).
    Angular {
        rf: &'a [i32; RF_LEN],
        angle: i32,
        n: usize,
        out: &'a mut Block,
    },
    /// Clamps the n x n block to 0..=max and narrows it to u16.
    Store {
        block: &'a Block,
        out: &'a mut Pred,
        n: usize,
        max: i32,
    },
}

/// Runs `op` on the vector kernels of `tier`; false (nothing done) for `Scalar` or a tier this
/// CPU lacks, and the caller then runs the scalar code. Avx512 uses the AVX2 kernels: the blocks
/// are at most 32 samples wide, so 512-bit lanes (16 of i32) would not fill.
#[allow(unsafe_code, unused_unsafe)]
fn run(tier: Tier, op: Op) -> bool {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 | Tier::Avx512 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: the guard has just detected AVX2 (the only feature the kernel enables);
            // `simd::available()` lists Avx512 only on top of AVX2 (src/simd/mod.rs).
            unsafe { x86::run(op) };
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: the guard has just detected NEON, the only feature the kernel enables.
            unsafe { arm::run(op) };
            true
        }
        _ => false,
    }
}

/// `N` values of `a` from index `i`, for the vector loads; None when they are not all there.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn arr<const N: usize>(a: &[i32], i: usize) -> Option<&[i32; N]> {
    a.get(i..i.checked_add(N)?)?.try_into().ok()
}

/// Clamps the n x n block to the bit depth and narrows it.
fn store(tier: Tier, pred: &Block, out: &mut Pred, n: usize, max: i32) {
    // The vector store saturates to u16::MAX where the scalar tail gives 0, so only run it in range.
    if max <= i32::from(u16::MAX)
        && run(
            tier,
            Op::Store {
                block: pred,
                out: &mut *out,
                n,
                max,
            },
        )
    {
        return;
    }
    for (src, dst) in pred.iter().zip(out.iter_mut()).take(n) {
        for (d, &v) in dst.iter_mut().zip(src).take(n) {
            *d = u16::try_from(v.clamp(0, max)).unwrap_or(0);
        }
    }
}

/// Reference sample construction (8.4.4.2.1) and substitution (8.4.4.2.2).
#[allow(clippy::too_many_arguments)]
fn gather(
    frame: &Frame,
    act: &Active,
    c: usize,
    x0: usize,
    y0: usize,
    n: usize,
    sub: usize,
    bit_depth: u32,
    p: &mut Refs,
) -> Result<()> {
    let len = 4 * n + 1;
    let to_i = |v: usize| i32::try_from(v).map_err(|_| bad());
    let sub_i = to_i(sub)?;
    let (xt, yt) = (to_i(x0)?, to_i(y0)?);
    let (xt_y, yt_y) = (xt * sub_i, yt * sub_i);
    let mut avail = [false; MAX_REF];
    // 6.4.1 looks at the luma location through its min-TB cell (z-scan order) and its CTB, so
    // neighbours in one cell share the answer: ask once per run of equal cells.
    let shift = act.sps.log2_min_trafo_size;
    let mut cached: Option<((i32, i32), bool)> = None;
    // Component location of line index i.
    for i in 0..len {
        let (dx, dy) = if i < 2 * n {
            (-1, to_i(2 * n - 1 - i)?)
        } else if i == 2 * n {
            (-1, -1)
        } else {
            (to_i(i - 2 * n - 1)?, -1)
        };
        let (xn, yn) = (xt + dx, yt + dy);
        let sample = usize::try_from(xn)
            .ok()
            .zip(usize::try_from(yn).ok())
            .and_then(|(x, y)| frame.sample(c, x, y));
        let usable = sample.filter(|_| {
            let (xl, yl) = (xn * sub_i, yn * sub_i);
            let cell = (xl >> shift, yl >> shift);
            match cached {
                Some((c, a)) if c == cell => a,
                _ => {
                    let a = frame.available(act, xt_y, yt_y, xl, yl);
                    cached = Some((cell, a));
                    a
                }
            }
        });
        if let (Some(v), Some(a), Some(d)) = (usable, avail.get_mut(i), p.r.get_mut(i)) {
            *a = true;
            *d = i32::from(v);
        }
    }
    let line = p.r.get_mut(..len).ok_or_else(bad)?;
    let avail = avail.get(..len).ok_or_else(bad)?;
    match avail.iter().position(|&a| a) {
        // No sample available: 1 << (bitDepth - 1).
        None => line.fill(1 << (bit_depth - 1)),
        Some(first) => {
            // Step 1: p[-1][2n-1] takes the first available value in search order.
            let v = at(line, first);
            if let Some(d) = line.first_mut() {
                *d = v;
            }
            // Steps 2 and 3: an unavailable sample copies its predecessor in the line.
            let mut prev = v;
            for (d, &a) in line.iter_mut().zip(avail).skip(1) {
                if a {
                    prev = *d;
                } else {
                    *d = prev;
                }
            }
        }
    }
    Ok(())
}

/// 8.4.4.2.3.
fn filter(tier: Tier, p: &mut Refs, mode: u32, strong_enabled: bool, bit_depth: u32) {
    let n = p.n;
    let thres = match n {
        8 => 7,
        16 => 1,
        32 => 0,
        _ => return, // nTbS 4: no filtering
    };
    if mode == 1 {
        return;
    }
    let m = i32::try_from(mode).unwrap_or(0);
    if (m - 26).abs().min((m - 10).abs()) <= thres {
        return;
    }
    let len = 4 * n + 1;
    let c = p.top(0);
    let bi = strong_enabled
        && n == 32
        && (c + p.top(2 * n) - 2 * p.top(n)).abs() < (1 << (bit_depth - 5))
        && (c + p.left(2 * n) - 2 * p.left(n)).abs() < (1 << (bit_depth - 5));
    let src = p.r;
    if bi {
        // (8-36)..(8-40): bilinear between the corner and the far ends.
        let (l63, t63) = (p.left(64), p.top(64));
        for k in 0..63 {
            let k_i = k as i32;
            let l = ((63 - k_i) * c + (k_i + 1) * l63 + 32) >> 6;
            let t = ((63 - k_i) * c + (k_i + 1) * t63 + 32) >> 6;
            if let Some(d) = p.r.get_mut(63 - k) {
                *d = l;
            }
            if let Some(d) = p.r.get_mut(65 + k) {
                *d = t;
            }
        }
    } else if !run(
        tier,
        Op::Smooth {
            src: &src,
            dst: &mut p.r,
            len,
        },
    ) {
        // (8-41)..(8-45): [1 2 1] along the line, end points unchanged.
        for (i, w) in src.get(..len).unwrap_or(&[]).windows(3).enumerate() {
            if let (&[a, b, d], Some(o)) = (w, p.r.get_mut(i + 1)) {
                *o = (a + 2 * b + d + 2) >> 2;
            }
        }
    }
}

fn planar_with(tier: Tier, p: &Refs, pred: &mut Block, log2: u32) {
    if !run(
        tier,
        Op::Planar {
            p,
            out: &mut *pred,
            log2,
        },
    ) {
        planar(p, pred, log2);
    }
}

/// 8.4.4.2.4 (8-46).
fn planar(p: &Refs, pred: &mut Block, log2: u32) {
    let n = 1usize << log2;
    let (tr, bl) = (p.top(n + 1), p.left(n + 1));
    for (y, row) in pred.iter_mut().enumerate().take(n) {
        for (x, v) in row.iter_mut().enumerate().take(n) {
            *v = ((n - 1 - x) as i32 * p.left(y + 1)
                + (x as i32 + 1) * tr
                + (n - 1 - y) as i32 * p.top(x + 1)
                + (y as i32 + 1) * bl
                + n as i32)
                >> (log2 + 1);
        }
    }
}

/// 8.4.4.2.5 (8-47)..(8-52).
fn dc(p: &Refs, pred: &mut Block, log2: u32, luma: bool) {
    let n = 1usize << log2;
    let sum: i32 = (1..=n).map(|k| p.top(k) + p.left(k)).sum();
    let v = (sum + n as i32) >> (log2 + 1);
    for row in pred.iter_mut().take(n) {
        row.iter_mut().take(n).for_each(|d| *d = v);
    }
    if luma && n < 32 {
        if let Some(d) = pred.first_mut().and_then(|r| r.first_mut()) {
            *d = (p.left(1) + 2 * v + p.top(1) + 2) >> 2;
        }
        for k in 1..n {
            if let Some(d) = pred.first_mut().and_then(|r| r.get_mut(k)) {
                *d = (p.top(k + 1) + 3 * v + 2) >> 2;
            }
            if let Some(d) = pred.get_mut(k).and_then(|r| r.first_mut()) {
                *d = (p.left(k + 1) + 3 * v + 2) >> 2;
            }
        }
    }
}

/// 8.4.4.2.6.
fn angular(tier: Tier, p: &Refs, pred: &mut Block, mode: u32, luma: bool, max: i32) -> Result<()> {
    let n = p.n;
    let ni = n as i32;
    let idx = usize::try_from(mode - 2).map_err(|_| bad())?;
    let angle = INTRA_PRED_ANGLE.get(idx).copied().ok_or_else(bad)?;
    let vertical = mode >= 18;
    // main(k) / side(k) are p[-1+k][-1] / p[-1][-1+k] for vertical modes, swapped otherwise.
    let main = |k: usize| if vertical { p.top(k) } else { p.left(k) };
    let side = |k: usize| if vertical { p.left(k) } else { p.top(k) };
    // ref[x] for x = -n..=2n lives at index x + 32.
    let mut rf = [0i32; RF_LEN];
    for k in 0..=n {
        if let Some(d) = rf.get_mut(32 + k) {
            *d = main(k);
        }
    }
    let last = (ni * angle) >> 5;
    if angle < 0 {
        if last < -1 {
            let inv = mode
                .checked_sub(11)
                .and_then(|i| INV_ANGLE.get(i as usize))
                .copied()
                .ok_or_else(bad)?;
            for x in last..=-1 {
                let k = usize::try_from((x * inv + 128) >> 8).map_err(|_| bad())?;
                if let Some(d) = rf.get_mut((32 + x) as usize) {
                    *d = side(k);
                }
            }
        }
    } else {
        for k in n + 1..=2 * n {
            if let Some(d) = rf.get_mut(32 + k) {
                *d = main(k);
            }
        }
    }
    // The vector kernels give out[a][b]; horizontal modes are transposed into place.
    let done = if vertical {
        run(
            tier,
            Op::Angular {
                rf: &rf,
                angle,
                n,
                out: &mut *pred,
            },
        )
    } else {
        let mut t: Block = [[0; 32]; 32];
        let ok = run(
            tier,
            Op::Angular {
                rf: &rf,
                angle,
                n,
                out: &mut t,
            },
        );
        if ok {
            for (a, row) in t.iter().enumerate().take(n) {
                for (b, &v) in row.iter().enumerate().take(n) {
                    if let Some(d) = pred.get_mut(b).and_then(|r| r.get_mut(a)) {
                        *d = v;
                    }
                }
            }
        }
        ok
    };
    if !done {
        for a in 0..n {
            let i_idx = ((a as i32 + 1) * angle) >> 5;
            let i_fact = ((a as i32 + 1) * angle) & 31;
            for b in 0..n {
                let base = (32 + b as i32 + i_idx + 1) as usize;
                let v = if i_fact != 0 {
                    ((32 - i_fact) * at(&rf, base) + i_fact * at(&rf, base + 1) + 16) >> 5
                } else {
                    at(&rf, base)
                };
                // Vertical modes: a = y, b = x; horizontal modes: a = x, b = y.
                let (x, y) = if vertical { (b, a) } else { (a, b) };
                if let Some(d) = pred.get_mut(y).and_then(|r| r.get_mut(x)) {
                    *d = v;
                }
            }
        }
    }
    // Boundary filters (8-60), (8-68); disableIntraBoundaryFilter is 0.
    if luma && n < 32 && (mode == 26 || mode == 10) {
        let corner = p.top(0);
        for k in 0..n {
            let v = (main(1) + ((side(k + 1) - corner) >> 1)).clamp(0, max);
            let (x, y) = if mode == 26 { (0, k) } else { (k, 0) };
            if let Some(d) = pred.get_mut(y).and_then(|r| r.get_mut(x)) {
                *d = v;
            }
        }
    }
    Ok(())
}

/// True when the n x n samples of both blocks are equal.
fn same(a: &Pred, b: &Pred, n: usize) -> bool {
    a.iter()
        .zip(b)
        .take(n)
        .all(|(x, y)| x.iter().zip(y).take(n).all(|(p, q)| p == q))
}

/// Differential fuzz entry (`__fuzz_intra` in lib.rs): builds one block of reference samples, a
/// size, a mode, a bit depth (8..=16) and the luma/strong switches from `data` (zeros past its
/// end) and returns true when every available tier gives scalar's samples. The reference
/// samples are masked to the bit depth unless a flag asks for raw 16-bit values.
pub(crate) fn fuzz_tiers(data: &[u8]) -> bool {
    let mut it = data.iter().copied();
    let mut next = || it.next().unwrap_or(0);
    let log2 = 2 + u32::from(next() & 3);
    let mode = u32::from(next()) % 35;
    let flags = next();
    let bit_depth = 8 + u32::from(next()) % 9;
    let (luma, strong, raw) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
    let n = 1usize << log2;
    let max = (1u32 << bit_depth) - 1;
    let mut base = Refs { r: [0; MAX_REF], n };
    for d in base.r.iter_mut().take(4 * n + 1) {
        let v = u32::from(next()) | (u32::from(next()) << 8);
        *d = i32::try_from(if raw { v } else { v & max }).unwrap_or(0);
    }
    let run_tier = |t: Tier| {
        let mut p = base.clone();
        compute(t, &mut p, mode, log2, luma, strong, bit_depth)
    };
    let Ok(want) = run_tier(Tier::Scalar) else {
        return false;
    };
    simd::available()
        .into_iter()
        .all(|t| run_tier(t).is_ok_and(|got| same(&got, &want, n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::SliceParams;
    use crate::limits::Limits;
    use crate::ps::tests::{ledger, pps_bits, sps_bits};
    use crate::ps::{activate, parse_pps, parse_sps};
    use std::sync::Arc;

    /// 64x64 picture, every CTB started in slice 0.
    fn setup() -> (Active, Frame) {
        let l = ledger();
        let sps = Arc::new(parse_sps(&sps_bits(64, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let act = activate(sps, pps, &Limits::default(), &l).unwrap();
        let mut f = Frame::new(&act, &l).unwrap();
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
        let idx = f.push_slice(s, &l).unwrap();
        let ctbs = (act.sps.pic_width_in_ctbs_y * act.sps.pic_height_in_ctbs_y) as usize;
        for a in 0..ctbs {
            f.start_ctb(a, idx).unwrap();
        }
        (act, f)
    }

    /// Sets the 4x4 block at (16, 16) up with the given neighbours and clears the block.
    fn load(f: &mut Frame, corner: u16, top: [u16; 8], left: [u16; 8]) {
        f.set_sample(0, 15, 15, corner).unwrap();
        for i in 0..8 {
            f.set_sample(0, 16 + i, 15, top[i]).unwrap();
            f.set_sample(0, 15, 16 + i, left[i]).unwrap();
        }
        for y in 16..20 {
            f.row_mut(0, 16, y, 4).unwrap().fill(0);
        }
    }

    fn out(f: &Frame, x: usize, y: usize) -> u16 {
        f.sample(0, 16 + x, 16 + y).unwrap()
    }

    #[test]
    fn dc() {
        let (act, mut f) = setup();
        load(
            &mut f,
            0,
            [8, 8, 8, 8, 0, 0, 0, 0],
            [16, 16, 16, 16, 0, 0, 0, 0],
        );
        predict(&mut f, &act, 0, 16, 16, 2, 1).unwrap();
        assert_eq!(out(&f, 0, 0), 12);
        assert_eq!(out(&f, 1, 0), 11);
        assert_eq!(out(&f, 0, 1), 13);
        assert_eq!(out(&f, 3, 3), 12);
    }

    #[test]
    fn planar() {
        let (act, mut f) = setup();
        load(
            &mut f,
            0,
            [8, 8, 8, 8, 40, 40, 40, 40],
            [16, 16, 16, 16, 48, 48, 48, 48],
        );
        predict(&mut f, &act, 0, 16, 16, 2, 0).unwrap();
        assert_eq!(out(&f, 0, 0), 20);
        assert_eq!(out(&f, 3, 3), 44);
    }

    #[test]
    fn vertical_horizontal() {
        let (act, mut f) = setup();
        let top = [8, 16, 24, 32, 0, 0, 0, 0];
        load(&mut f, 4, top, [16, 16, 16, 16, 0, 0, 0, 0]);
        predict(&mut f, &act, 0, 16, 16, 2, 26).unwrap();
        assert_eq!((out(&f, 0, 2), out(&f, 1, 2), out(&f, 3, 0)), (14, 16, 32));
        load(&mut f, 4, top, [16, 16, 16, 16, 0, 0, 0, 0]);
        predict(&mut f, &act, 0, 16, 16, 2, 10).unwrap();
        let row0: Vec<u16> = (0..4).map(|x| out(&f, x, 0)).collect();
        assert_eq!(row0, [18, 22, 26, 30]);
        assert_eq!(out(&f, 2, 3), 16);
    }

    #[test]
    fn diagonals() {
        let (act, mut f) = setup();
        let left = [10, 20, 30, 40, 50, 60, 70, 80];
        load(&mut f, 5, [8, 16, 24, 32, 0, 0, 0, 0], left);
        predict(&mut f, &act, 0, 16, 16, 2, 2).unwrap();
        assert_eq!((out(&f, 0, 0), out(&f, 3, 3)), (20, 80));
        // Mode 18: negative angle, reference extended with invAngle.
        predict(&mut f, &act, 0, 16, 16, 2, 18).unwrap();
        assert_eq!((out(&f, 1, 0), out(&f, 0, 1), out(&f, 0, 3)), (8, 10, 30));
        assert_eq!(out(&f, 2, 2), 5);
    }

    #[test]
    fn substitution() {
        let (act, mut f) = setup();
        // Nothing available at the picture origin: every reference is 128.
        f.row_mut(0, 0, 0, 4).unwrap().fill(0);
        predict(&mut f, &act, 0, 0, 0, 2, 1).unwrap();
        assert_eq!(f.sample(0, 0, 0), Some(128));
        // Only the left column (and below) available at (16, 0): top and corner copy left(0).
        for y in 0..8 {
            f.set_sample(0, 15, y, 10 * (y as u16 + 1)).unwrap();
        }
        for y in 0..4 {
            f.row_mut(0, 16, y, 4).unwrap().fill(0);
        }
        predict(&mut f, &act, 0, 16, 0, 2, 1).unwrap();
        assert_eq!(f.sample(0, 17, 1), Some(18));
        assert_eq!(f.sample(0, 16, 0), Some(14));
    }

    #[test]
    fn rejects_outside() {
        let (act, mut f) = setup();
        assert!(predict(&mut f, &act, 0, 62, 0, 2, 1).is_err());
        assert!(predict(&mut f, &act, 0, 0, 0, 2, 35).is_err());
    }

    /// xorshift64.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Reference samples of 2^log2 blocks: `kind` picks random in range, random 16-bit (past the
    /// bit depth), extremes only, constant, or a ramp.
    fn refs(rng: &mut Rng, log2: u32, bit_depth: u32, kind: u32) -> Refs {
        let n = 1usize << log2;
        let max = (1u64 << bit_depth) - 1;
        let mut p = Refs { r: [0; MAX_REF], n };
        let c = rng.next_u64() & max;
        for (i, d) in p.r.iter_mut().take(4 * n + 1).enumerate() {
            let v = match kind {
                0 => rng.next_u64() & max,
                1 => rng.next_u64() & 0xffff,
                2 => {
                    if rng.next_u64() & 1 == 0 {
                        0
                    } else {
                        max
                    }
                }
                3 => c,
                _ => (i as u64 * max / (4 * n as u64)).min(max),
            };
            *d = v as i32;
        }
        p
    }

    #[test]
    fn tiers_match_scalar() {
        let avail = simd::available();
        #[cfg(target_arch = "x86_64")]
        assert!(avail.contains(&Tier::Avx2), "AVX2 not detected");
        #[cfg(target_arch = "aarch64")]
        assert!(avail.contains(&Tier::Neon), "NEON not detected");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for bd in [8, 10, 12, 16] {
            for log2 in 2..=5 {
                for mode in 0..=34 {
                    for (luma, strong) in [(true, false), (true, true), (false, false)] {
                        for kind in 0..5 {
                            for _ in 0..2 {
                                let base = refs(&mut rng, log2, bd, kind);
                                let mut p = base.clone();
                                let want =
                                    compute(Tier::Scalar, &mut p, mode, log2, luma, strong, bd)
                                        .unwrap();
                                for &t in &avail {
                                    let mut p = base.clone();
                                    let got =
                                        compute(t, &mut p, mode, log2, luma, strong, bd).unwrap();
                                    assert!(
                                        same(&got, &want, 1 << log2),
                                        "{t:?} bd {bd} n {} mode {mode} luma {luma} strong {strong} kind {kind}",
                                        1 << log2
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simd_tiers_run_kernels() {
        let (block, mut out) = ([[0i32; 32]; 32], [[0u16; 32]; 32]);
        for t in simd::available() {
            let op = Op::Store {
                block: &block,
                out: &mut out,
                n: 4,
                max: 255,
            };
            assert_eq!(run(t, op), t != Tier::Scalar, "{t:?}");
        }
    }

    #[test]
    fn fuzz_entry() {
        let mut rng = Rng(12345);
        for len in [0usize, 1, 4, 7, 40, 300] {
            for _ in 0..50 {
                let data: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
                assert!(fuzz_tiers(&data));
            }
        }
    }

    /// Every tier writes the same frame: random planes (8-bit range and raw 16-bit), blocks of
    /// every size at every aligned position, luma and chroma, so substitution and all the
    /// availability cases are covered.
    #[test]
    fn frames_match_scalar() {
        let modes = [0, 1, 2, 5, 9, 10, 11, 17, 18, 19, 25, 26, 27, 34];
        for mask in [0xffu64, 0xffff] {
            let mut want: Option<Vec<u16>> = None;
            for t in simd::available() {
                let (act, mut f) = setup();
                let mut rng = Rng(777);
                let mut snap = Vec::new();
                for c in 0..3 {
                    let w = if c == 0 { 64 } else { 32 };
                    for y in 0..w {
                        for x in 0..w {
                            f.set_sample(c, x, y, (rng.next_u64() & mask) as u16)
                                .unwrap();
                        }
                    }
                }
                for c in 0..3 {
                    let w = if c == 0 { 64 } else { 32 };
                    for log2 in 2..=5u32 {
                        let n = 1usize << log2;
                        for &mode in &modes {
                            for y0 in (0..w).step_by(n) {
                                for x0 in (0..w).step_by(n) {
                                    let r = predict_with(t, &mut f, &act, c, x0, y0, log2, mode);
                                    snap.push(u16::from(r.is_ok()));
                                }
                            }
                        }
                    }
                    for y in 0..w {
                        for x in 0..w {
                            snap.push(f.sample(c, x, y).unwrap());
                        }
                    }
                }
                // The first tier is scalar.
                let first = want.get_or_insert_with(|| snap.clone());
                assert!(*first == snap, "{t:?} mask {mask:x}");
            }
        }
    }

    /// The gather before the per-cell availability cache: one `available` call per sample.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn gather_per_sample(
        frame: &Frame,
        act: &Active,
        c: usize,
        x0: usize,
        y0: usize,
        n: usize,
        sub: usize,
        bit_depth: u32,
        p: &mut Refs,
    ) {
        let len = 4 * n + 1;
        let (xt, yt) = (x0 as i32, y0 as i32);
        let s = sub as i32;
        let mut avail = [false; MAX_REF];
        for i in 0..len {
            let (dx, dy) = if i < 2 * n {
                (-1, (2 * n - 1 - i) as i32)
            } else if i == 2 * n {
                (-1, -1)
            } else {
                ((i - 2 * n - 1) as i32, -1)
            };
            let (xn, yn) = (xt + dx, yt + dy);
            let sample = usize::try_from(xn)
                .ok()
                .zip(usize::try_from(yn).ok())
                .and_then(|(x, y)| frame.sample(c, x, y));
            if let Some(v) = sample.filter(|_| frame.available(act, xt * s, yt * s, xn * s, yn * s))
            {
                avail[i] = true;
                p.r[i] = i32::from(v);
            }
        }
        let line = &mut p.r[..len];
        match avail[..len].iter().position(|&a| a) {
            None => line.fill(1 << (bit_depth - 1)),
            Some(first) => {
                line[0] = line[first];
                let mut prev = line[0];
                for i in 1..len {
                    if avail[i] {
                        prev = line[i];
                    } else {
                        line[i] = prev;
                    }
                }
            }
        }
    }

    #[test]
    fn gather_matches_per_sample() {
        let (act, mut f) = setup();
        let mut rng = Rng(99);
        for c in 0..3 {
            let w = if c == 0 { 64 } else { 32 };
            for y in 0..w {
                for x in 0..w {
                    f.set_sample(c, x, y, (rng.next_u64() & 0xff) as u16)
                        .unwrap();
                }
            }
            let sub = if c == 0 { 1 } else { 2 };
            for log2 in 2..=5u32 {
                let n = 1usize << log2;
                for y0 in (0..w).step_by(n) {
                    for x0 in (0..w).step_by(n) {
                        let mut a = Refs { r: [0; MAX_REF], n };
                        let mut b = a.clone();
                        gather(&f, &act, c, x0, y0, n, sub, 8, &mut a).unwrap();
                        gather_per_sample(&f, &act, c, x0, y0, n, sub, 8, &mut b);
                        assert!(a.r == b.r, "c {c} n {n} at {x0},{y0}");
                    }
                }
            }
        }
    }
}
