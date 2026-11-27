//! Per-picture state shared by slice decoding and the in-loop filters: the sample planes, the
//! metadata grids, the per-CTB records and the z-scan availability of 6.4.1.
//!
//! Memory: [`Frame::new`] books every buffer in the ledger up front and keeps the total in
//! [`Frame::charged`]; [`Frame::push_slice`] books each slice record as it is added. Whoever owns
//! the `Frame` releases `frame.charged` with `Ledger::release` when it drops the frame (the same
//! rule as `Active::charged` in `ps.rs`; the decoder does both at the end of the picture).
//!
//! Coordinates: sample accessors take component sample coordinates `(x, y)` of plane `c`
//! (0 = Y, 1 = Cb, 2 = Cr). The metadata grids are on the 4x4 luma grid (`grid_w` x `grid_h`
//! cells, row-major) and their accessors take luma sample coordinates; a block fill covers every
//! 4x4 cell the block touches. Per-CTB records are indexed by CtbAddrInRs.

use crate::error::{Error, Result};
use crate::ledger::{Ledger, filled, grow_one};
use crate::ps::Active;
use std::mem::size_of;
use std::sync::{Mutex, PoisonError};
use std::thread;

/// `slice_idx` of a CTB whose decoding has not started.
pub(crate) const NO_SLICE: u32 = u32::MAX;

/// `no_filter` bit 0: a PCM CU with pcm_loop_filter_disabled_flag (deblocking and SAO leave the
/// samples alone, 8.7.2.5.7 and 8.7.3).
pub(crate) const NO_FILTER_PCM: u8 = 1;
/// `no_filter` bit 1: cu_transquant_bypass_flag.
pub(crate) const NO_FILTER_BYPASS: u8 = 2;

/// SAO parameters of one CTB (7.3.8.3, 7.4.9.3), per component index (0 = Y, 1 = Cb, 2 = Cr).
/// For Cb and Cr the syntax sends one `sao_type_idx_chroma` and one `sao_eo_class_chroma`; the
/// parser stores them in both entries. All zero (the Default) means SAO off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SaoParams {
    /// 0 = not applied, 1 = band offset, 2 = edge offset.
    pub type_idx: [u8; 3],
    /// `sao_band_position`, 0..=31 (band offset only).
    pub band_position: [u8; 3],
    /// `SaoEoClass`, 0..=3 (edge offset only): 0 horizontal, 1 vertical, 2 135 degrees,
    /// 3 45 degrees.
    pub eo_class: [u8; 3],
    /// SaoOffsetVal[0..=4] of 7.4.9.3.2: index 0 is 0, indexes 1..=4 are the signed offsets
    /// already scaled by `<< log2OffsetScale` (and with the edge offset signs applied).
    pub offset_val: [[i16; 5]; 3],
}

/// Parameters of one slice segment that the filters and the QP derivation need. Dependent slice
/// segments get their own record (with the values copied from their slice) so that a CTB always
/// finds its parameters through its own `slice_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SliceParams {
    /// SliceAddrRs: CtbAddrInRs of the first CTB of the independent slice this segment belongs
    /// to. Two blocks are in the same slice for 6.4.1 when this is equal.
    pub slice_addr_rs: u32,
    /// SliceQpY (7-54).
    pub qp_y: i32,
    /// pps_cb_qp_offset + slice_cb_qp_offset, the offset of the chroma QP derivation for
    /// dequantisation (8.6.1). The deblocking filter uses `pps_cb_qp_offset` alone
    /// (cQpPicOffset) from the PPS.
    pub cb_qp_offset: i32,
    /// pps_cr_qp_offset + slice_cr_qp_offset.
    pub cr_qp_offset: i32,
    /// slice_deblocking_filter_disabled_flag.
    pub deblocking_disabled: bool,
    /// slice_beta_offset_div2.
    pub beta_offset_div2: i32,
    /// slice_tc_offset_div2.
    pub tc_offset_div2: i32,
    /// slice_loop_filter_across_slices_enabled_flag.
    pub loop_filter_across_slices: bool,
    /// slice_sao_luma_flag.
    pub sao_luma: bool,
    /// slice_sao_chroma_flag.
    pub sao_chroma: bool,
}

/// The decoded picture under construction (coded size, before cropping).
#[derive(Debug)]
pub(crate) struct Frame {
    /// 1 (4:0:0) or 3 (4:2:0) planes of samples, row-major with `stride[c]` samples per row.
    pub planes: Vec<Vec<u16>>,
    /// Plane widths in samples (entries of absent planes are 0).
    pub width: [usize; 3],
    /// Plane heights in samples.
    pub height: [usize; 3],
    /// Row stride of each plane in samples (equal to `width`).
    pub stride: [usize; 3],
    /// Columns of the 4x4 grid: ceil(luma width / 4).
    pub grid_w: usize,
    /// Rows of the 4x4 grid.
    pub grid_h: usize,
    /// QpY per 4x4 cell, written for every CU (initialised to 0).
    pub qp_y: Vec<i8>,
    /// IntraPredModeY per 4x4 cell. Initialised to 1 (INTRA_DC); a PCM CU must store 1 as well
    /// (candIntraPredModeX of 8.4.2).
    pub intra_mode: Vec<u8>,
    /// [`NO_FILTER_PCM`] | [`NO_FILTER_BYPASS`] per 4x4 cell.
    pub no_filter: Vec<u8>,
    /// Deblocking boundary strength (0 or 2) of the vertical edge on the left side of each 4x4
    /// cell.
    pub edge_v: Vec<u8>,
    /// Deblocking boundary strength (0 or 2) of the horizontal edge on the top side of each 4x4
    /// cell.
    pub edge_h: Vec<u8>,
    /// CtDepth per 4x4 cell (ctxInc of split_cu_flag).
    pub ct_depth: Vec<u8>,
    /// Per CTB: index into `slices`, or [`NO_SLICE`] while the CTB has not been started. This is
    /// also the "decoded" marker for the availability process: the decoder calls
    /// [`Frame::start_ctb`] before it parses a CTB, so a block of the current CTB at an earlier
    /// z-scan position is available and a CTB that was never started is not.
    pub slice_idx: Vec<u32>,
    /// Per CTB: SAO parameters (all zero for a CTB without SAO).
    pub sao: Vec<SaoParams>,
    /// The slice segments of the picture, in decoding order.
    pub slices: Vec<SliceParams>,
    /// Bytes booked in the ledger for this frame so far; release them when the frame is dropped.
    pub charged: usize,
}

fn out_of_picture() -> Error {
    Error::InvalidData("block outside the picture")
}

/// Value at luma sample (x, y) of a 4x4 grid `g` of `gw` x `gh` cells.
fn grid_get<T: Copy>(g: &[T], gw: usize, gh: usize, x: usize, y: usize) -> Option<T> {
    let (gx, gy) = (x >> 2, y >> 2);
    if gx >= gw || gy >= gh {
        return None;
    }
    g.get(gy * gw + gx).copied()
}

/// Sets every cell touched by the luma block (x, y, w, h) to `v`.
#[allow(clippy::too_many_arguments)]
fn grid_fill<T: Copy>(
    g: &mut [T],
    gw: usize,
    gh: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    v: T,
) -> Result<()> {
    let x1 = x.checked_add(w).ok_or_else(out_of_picture)?;
    let y1 = y.checked_add(h).ok_or_else(out_of_picture)?;
    let (gx0, gx1) = (x >> 2, x1.div_ceil(4));
    let (gy0, gy1) = (y >> 2, y1.div_ceil(4));
    if gx1 > gw || gy1 > gh {
        return Err(out_of_picture());
    }
    for gy in gy0..gy1 {
        g.get_mut(gy * gw + gx0..gy * gw + gx1)
            .ok_or_else(out_of_picture)?
            .fill(v);
    }
    Ok(())
}

macro_rules! grid_accessors {
    ($(($field:ident, $ty:ty, $get:ident, $fill:ident)),* $(,)?) => {
        impl Frame {
            $(
                #[doc = concat!("`", stringify!($field), "` of the 4x4 cell covering luma sample (x, y); None outside the picture.")]
                pub(crate) fn $get(&self, x: usize, y: usize) -> Option<$ty> {
                    grid_get(&self.$field, self.grid_w, self.grid_h, x, y)
                }

                #[doc = concat!("Sets `", stringify!($field), "` of every 4x4 cell touched by the luma block (x, y, w, h); Err when it leaves the picture.")]
                pub(crate) fn $fill(&mut self, x: usize, y: usize, w: usize, h: usize, v: $ty) -> Result<()> {
                    grid_fill(&mut self.$field, self.grid_w, self.grid_h, x, y, w, h, v)
                }
            )*
        }
    };
}

grid_accessors!(
    (qp_y, i8, qp_y_at, fill_qp_y),
    (intra_mode, u8, intra_mode_at, fill_intra_mode),
    (no_filter, u8, no_filter_at, fill_no_filter),
    (edge_v, u8, edge_v_at, fill_edge_v),
    (edge_h, u8, edge_h_at, fill_edge_h),
    (ct_depth, u8, ct_depth_at, fill_ct_depth),
);

impl Frame {
    /// Allocates the frame for the active SPS: planes filled with `1 << (bitDepth - 1)`, QpY 0,
    /// IntraPredModeY 1 (DC), no filter bits, no edges, CtDepth 0, no CTB started, no SAO, no
    /// slices. Everything is booked in `ledger` first (`charged`); on error nothing stays booked.
    pub(crate) fn new(act: &Active, ledger: &Ledger) -> Result<Self> {
        let sps = &act.sps;
        let to_usize = |v: u32| usize::try_from(v).map_err(|_| Error::AllocLimit);
        let (w, h) = (to_usize(sps.width)?, to_usize(sps.height)?);
        let n_planes = if sps.chroma_format_idc == 0 { 1 } else { 3 };
        let (cw, ch) = if n_planes == 1 {
            (0, 0)
        } else {
            (w / 2, h / 2)
        };
        let width = [w, cw, cw];
        let height = [h, ch, ch];
        let (grid_w, grid_h) = (w.div_ceil(4), h.div_ceil(4));
        let cells = grid_w.checked_mul(grid_h).ok_or(Error::AllocLimit)?;
        let ctbs = to_usize(sps.pic_width_in_ctbs_y)?
            .checked_mul(to_usize(sps.pic_height_in_ctbs_y)?)
            .ok_or(Error::AllocLimit)?;
        let samples = width
            .iter()
            .zip(&height)
            .try_fold(0usize, |a, (&pw, &ph)| a.checked_add(pw.checked_mul(ph)?))
            .ok_or(Error::AllocLimit)?;
        // Six one-byte grids, then per CTB a slice index and the SAO record.
        let total = samples
            .checked_mul(size_of::<u16>())
            .and_then(|b| b.checked_add(cells.checked_mul(6)?))
            .and_then(|b| {
                b.checked_add(ctbs.checked_mul(size_of::<u32>() + size_of::<SaoParams>())?)
            })
            .ok_or(Error::AllocLimit)?;
        ledger.reserve(total)?;
        let mut f = Self::build(act, width, height, (grid_w, grid_h), cells, ctbs)
            .inspect_err(|_| ledger.release(total))?;
        f.charged = total;
        Ok(f)
    }

    fn build(
        act: &Active,
        width: [usize; 3],
        height: [usize; 3],
        (grid_w, grid_h): (usize, usize),
        cells: usize,
        ctbs: usize,
    ) -> Result<Self> {
        let sps = &act.sps;
        let mid = |bit_depth: u32| 1u16 << bit_depth.saturating_sub(1).min(15);
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(3)
            .map_err(|_| Error::OutOfMemory)?;
        for (c, (&pw, &ph)) in width.iter().zip(&height).enumerate() {
            if pw == 0 || ph == 0 {
                continue;
            }
            let depth = if c == 0 {
                sps.bit_depth_luma
            } else {
                sps.bit_depth_chroma
            };
            planes.push(filled(
                pw.checked_mul(ph).ok_or(Error::AllocLimit)?,
                mid(depth),
            )?);
        }
        Ok(Self {
            planes,
            width,
            height,
            stride: width,
            grid_w,
            grid_h,
            qp_y: filled(cells, 0)?,
            intra_mode: filled(cells, 1)?,
            no_filter: filled(cells, 0)?,
            edge_v: filled(cells, 0)?,
            edge_h: filled(cells, 0)?,
            ct_depth: filled(cells, 0)?,
            slice_idx: filled(ctbs, NO_SLICE)?,
            sao: filled(ctbs, SaoParams::default())?,
            slices: Vec::new(),
            charged: 0,
        })
    }

    /// Number of planes (1 or 3).
    pub(crate) fn num_planes(&self) -> usize {
        self.planes.len()
    }

    /// The sample at (x, y) of plane `c`; None outside the plane.
    #[allow(dead_code, reason = "used by the tests only")]
    pub(crate) fn sample(&self, c: usize, x: usize, y: usize) -> Option<u16> {
        if x >= *self.width.get(c)? || y >= *self.height.get(c)? {
            return None;
        }
        self.planes
            .get(c)?
            .get(y * self.stride.get(c)? + x)
            .copied()
    }

    /// Sets the sample at (x, y) of plane `c`; Err outside the plane.
    #[allow(dead_code, reason = "used by the tests only")]
    pub(crate) fn set_sample(&mut self, c: usize, x: usize, y: usize, v: u16) -> Result<()> {
        let s = self
            .row_mut(c, x, y, 1)?
            .first_mut()
            .ok_or_else(out_of_picture)?;
        *s = v;
        Ok(())
    }

    /// `n` consecutive samples of row `y` of plane `c`, starting at column `x`; Err when they
    /// are not all inside the plane.
    pub(crate) fn row_mut(&mut self, c: usize, x: usize, y: usize, n: usize) -> Result<&mut [u16]> {
        let (w, h) = match (self.width.get(c), self.height.get(c)) {
            (Some(&w), Some(&h)) => (w, h),
            _ => return Err(out_of_picture()),
        };
        if y >= h || x.checked_add(n).is_none_or(|e| e > w) {
            return Err(out_of_picture());
        }
        let start = y * w + x;
        self.planes
            .get_mut(c)
            .and_then(|p| p.get_mut(start..start + n))
            .ok_or_else(out_of_picture)
    }

    /// Adds a slice segment record and returns its index for [`Frame::start_ctb`]. Books the
    /// record in the ledger (added to `charged`).
    pub(crate) fn push_slice(&mut self, p: SliceParams, ledger: &Ledger) -> Result<u32> {
        let idx = u32::try_from(self.slices.len())
            .ok()
            .filter(|&i| i != NO_SLICE)
            .ok_or(Error::InvalidData("too many slice segments"))?;
        self.charged += grow_one(&mut self.slices, ledger)?;
        self.slices.push(p);
        Ok(idx)
    }

    /// Marks the CTB at CtbAddrInRs `addr_rs` as started and in slice segment `slice_idx`
    /// (returned by [`Frame::push_slice`]). Call it before parsing the CTB.
    pub(crate) fn start_ctb(&mut self, addr_rs: usize, slice_idx: u32) -> Result<()> {
        if self.slices.get(slice_idx as usize).is_none() {
            return Err(Error::InvalidData("slice index"));
        }
        *self
            .slice_idx
            .get_mut(addr_rs)
            .ok_or(Error::InvalidData("CTB address"))? = slice_idx;
        Ok(())
    }

    /// True once [`Frame::start_ctb`] was called for the CTB.
    pub(crate) fn ctb_started(&self, addr_rs: usize) -> bool {
        self.slice_idx.get(addr_rs).is_some_and(|&i| i != NO_SLICE)
    }

    /// The slice segment record of a started CTB.
    pub(crate) fn ctb_slice(&self, addr_rs: usize) -> Option<&SliceParams> {
        self.slices.get(*self.slice_idx.get(addr_rs)? as usize)
    }

    /// SAO parameters of the CTB.
    pub(crate) fn sao_at(&self, addr_rs: usize) -> Option<&SaoParams> {
        self.sao.get(addr_rs)
    }

    /// Mutable SAO parameters of the CTB.
    pub(crate) fn sao_at_mut(&mut self, addr_rs: usize) -> Option<&mut SaoParams> {
        self.sao.get_mut(addr_rs)
    }

    /// CtbAddrInRs of the CTB covering luma sample (x, y), None outside the picture.
    pub(crate) fn ctb_addr(act: &Active, x: i32, y: i32) -> Option<usize> {
        let sps = &act.sps;
        if x < 0 || y < 0 || x as u32 >= sps.width || y as u32 >= sps.height {
            return None;
        }
        let (cx, cy) = (
            (x as u32) >> sps.ctb_log2_size_y,
            (y as u32) >> sps.ctb_log2_size_y,
        );
        usize::try_from(cy * sps.pic_width_in_ctbs_y + cx).ok()
    }

    /// 6.4.1: availability of the block covering luma location (x_nb, y_nb) for the current
    /// block at (x_curr, y_curr), in z-scan order. False when the neighbour is outside the
    /// picture, later in z-scan order (MinTbAddrZs), in a CTB not started yet, in another slice
    /// (SliceAddrRs differs) or in another tile.
    pub(crate) fn available(
        &self,
        act: &Active,
        x_curr: i32,
        y_curr: i32,
        x_nb: i32,
        y_nb: i32,
    ) -> bool {
        self.available_opt(act, x_curr, y_curr, x_nb, y_nb)
            .unwrap_or(false)
    }

    fn available_opt(&self, act: &Active, xc: i32, yc: i32, xn: i32, yn: i32) -> Option<bool> {
        let (cc, cn) = (Self::ctb_addr(act, xc, yc)?, Self::ctb_addr(act, xn, yn)?);
        let shift = act.sps.log2_min_trafo_size;
        let zs = |x: i32, y: i32| act.min_tb_addr_zs((x as usize) >> shift, (y as usize) >> shift);
        if zs(xn, yn)? > zs(xc, yc)? {
            return Some(false);
        }
        let (sc, sn) = (self.ctb_slice(cc)?, self.ctb_slice(cn)?);
        if sc.slice_addr_rs != sn.slice_addr_rs {
            return Some(false);
        }
        let tile = |ctb: usize| act.tile_id.get(*act.rs_to_ts.get(ctb)? as usize);
        Some(tile(cc)? == tile(cn)?)
    }
}

/// [`Frame::available`] for many neighbours of one current block: the CTB, z-scan order, slice
/// and tile of the current block are looked up once, and the CTB part of the answer (started,
/// same slice, same tile) is kept for the last CTB asked. Same answers as `Frame::available`.
///
/// MinTbAddrZs (6-10) is CtbAddrRsToTs of the CTB, shifted left by 2 * (CtbLog2SizeY -
/// Log2MinTrafoSize), plus the z-scan order of the min-TB cell inside the CTB. So the
/// comparison of two addresses is that of (CtbAddrInTs, order inside the CTB), and the order
/// inside the CTB comes from the cell coordinates (`Avail::morton`), with no load from the
/// picture-sized MinTbAddrZs table.
pub(crate) struct Avail<'a> {
    frame: &'a Frame,
    act: &'a Active,
    /// CtbAddrInRs and CtbAddrInTs of the current block, and its z-scan order in the CTB.
    rs: usize,
    ts: u32,
    m: u32,
    /// Luma position of the current block, for the debug check against `Frame::available`.
    #[cfg(debug_assertions)]
    cur: (usize, usize),
    slice_addr_rs: u32,
    tile: u32,
    last: Option<(usize, bool)>,
}

impl<'a> Avail<'a> {
    /// The current block at luma (x, y). None when it is outside the picture or its CTB was not
    /// started: then no neighbour is available.
    pub(crate) fn new(frame: &'a Frame, act: &'a Active, x: usize, y: usize) -> Option<Self> {
        let rs = Self::ctb(act, x, y)?;
        let ts = *act.rs_to_ts.get(rs)?;
        Some(Self {
            frame,
            act,
            rs,
            ts,
            m: Self::morton(act, x, y),
            #[cfg(debug_assertions)]
            cur: (x, y),
            slice_addr_rs: frame.ctb_slice(rs)?.slice_addr_rs,
            tile: *act.tile_id.get(ts as usize)?,
            last: None,
        })
    }

    /// 6.4.1 for the neighbour at luma (x, y).
    pub(crate) fn at(&mut self, x: usize, y: usize) -> bool {
        let a = self.at_zs(x, y);
        #[cfg(debug_assertions)]
        {
            let c = |v: usize| i32::try_from(v).unwrap_or(i32::MAX);
            let (cx, cy) = self.cur;
            debug_assert_eq!(
                a,
                self.frame.available(self.act, c(cx), c(cy), c(x), c(y)),
                "Avail::at against Frame::available at ({x}, {y}) for ({cx}, {cy})"
            );
        }
        a
    }

    fn at_zs(&mut self, x: usize, y: usize) -> bool {
        let Some(rs) = Self::ctb(self.act, x, y) else {
            return false;
        };
        if rs == self.rs {
            // The current CTB: started, same slice and tile; only the z-scan order counts.
            return Self::morton(self.act, x, y) <= self.m;
        }
        if self.act.rs_to_ts.get(rs).is_none_or(|&t| t > self.ts) {
            return false;
        }
        match self.last {
            Some((r, a)) if r == rs => a,
            _ => {
                let a = self
                    .frame
                    .ctb_slice(rs)
                    .is_some_and(|s| s.slice_addr_rs == self.slice_addr_rs)
                    && Self::tile(self.act, rs) == Some(self.tile);
                self.last = Some((rs, a));
                a
            }
        }
    }

    /// CtbAddrInRs of luma (x, y), None outside the picture.
    fn ctb(act: &Active, x: usize, y: usize) -> Option<usize> {
        let sps = &act.sps;
        if x >= sps.width as usize || y >= sps.height as usize {
            return None;
        }
        let l = sps.ctb_log2_size_y;
        Some((y >> l) * sps.pic_width_in_ctbs_y as usize + (x >> l))
    }

    fn tile(act: &Active, rs: usize) -> Option<u32> {
        act.tile_id.get(*act.rs_to_ts.get(rs)? as usize).copied()
    }

    /// The z-scan order of the min-TB cell of luma (x, y) inside its CTB: the low
    /// 2 * (CtbLog2SizeY - Log2MinTrafoSize) bits of MinTbAddrZs (6-10), x bits at the even
    /// positions and y bits at the odd ones. The SPS bounds the difference to 1..=4.
    fn morton(act: &Active, x: usize, y: usize) -> u32 {
        let sps = &act.sps;
        let s = sps.log2_min_trafo_size;
        let mask = (1usize << sps.ctb_log2_size_y.saturating_sub(s).min(8)) - 1;
        // Bits 0..8 of v to the even bit positions 0..16.
        let spread = |v: usize| {
            let mut v = (v & mask) as u32;
            v = (v | v << 4) & 0x0f0f;
            v = (v | v << 2) & 0x3333;
            (v | v << 1) & 0x5555
        };
        spread(x >> s) | spread(y >> s) << 1
    }
}

/// CTB-row cuts of `workers` balanced bands over `ctb_rows` CTB rows: `[0, .., ctb_rows]`, one
/// more entry than bands. At most one band per CTB row, at least one band.
pub(crate) fn band_rows(ctb_rows: usize, workers: usize) -> Vec<usize> {
    let n = workers.clamp(1, ctb_rows.max(1));
    (0..=n).map(|i| i * ctb_rows / n).collect()
}

/// Sample-row cuts of a plane from CTB-row cuts: cut `c` becomes `c * unit - shift` (`unit` is the
/// CTB height in this plane), capped at `rows`; the last cut is `rows`. `shift` 4 gives the
/// bands of the horizontal deblocking pass, which start 4 rows above the rows they own.
pub(crate) fn plane_cuts(ctb_cuts: &[usize], unit: usize, shift: usize, rows: usize) -> Vec<usize> {
    let last = ctb_cuts.len().saturating_sub(1);
    ctb_cuts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if i == last {
                rows
            } else {
                c.saturating_mul(unit).saturating_sub(shift).min(rows)
            }
        })
        .collect()
}

/// Stack of a band worker: the filters do not recurse, so far less than the 2 MiB default.
const BAND_STACK: usize = 512 * 1024;

/// A band waiting for a worker: its first plane row and its samples.
type Band<'a> = Mutex<Option<(usize, &'a mut [u16])>>;

fn run_band<F>(f: &F, k: usize, band: &Band<'_>) -> Result<()>
where
    F: Fn(usize, usize, &mut [u16]) -> Result<()>,
{
    let taken = band.lock().unwrap_or_else(PoisonError::into_inner).take();
    taken.map_or(Ok(()), |(row0, s)| f(k, row0, s))
}

/// Splits `plane` (rows of `stride` samples) at the sample-row `cuts` (`[0, .., end]`, band `k`
/// is rows `cuts[k]..cuts[k + 1]`) and calls `f(k, first_row, band)` on every band, band 0 on the
/// calling thread and the others on scoped threads (a refused spawn runs that band here). All
/// bands run; the result is the error of the lowest failing band, whatever the thread timing.
/// One band spawns nothing. A cut list that does not start at 0, goes back, or leaves the plane
/// is `InvalidData`.
pub(crate) fn par_bands<F>(plane: &mut [u16], stride: usize, cuts: &[usize], f: F) -> Result<()>
where
    F: Fn(usize, usize, &mut [u16]) -> Result<()> + Sync,
{
    let bad = || Error::InvalidData("band cuts");
    let mut bands: Vec<Band<'_>> = Vec::new();
    bands
        .try_reserve_exact(cuts.len().saturating_sub(1))
        .map_err(|_| Error::OutOfMemory)?;
    let mut rest = plane;
    let mut at = 0;
    for w in cuts.windows(2) {
        let &[a, b] = w else { continue };
        let len = b
            .checked_sub(a)
            .and_then(|r| r.checked_mul(stride))
            .filter(|_| a == at)
            .ok_or_else(bad)?;
        let (head, tail) = std::mem::take(&mut rest)
            .split_at_mut_checked(len)
            .ok_or_else(bad)?;
        bands.push(Mutex::new(Some((a, head))));
        rest = tail;
        at = b;
    }
    let f = &f;
    let Some((first, others)) = bands.split_first() else {
        return Ok(());
    };
    if others.is_empty() {
        return run_band(f, 0, first);
    }
    thread::scope(|s| {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(others.len())
            .map_err(|_| Error::OutOfMemory)?;
        for (i, band) in others.iter().enumerate() {
            let h = thread::Builder::new()
                .stack_size(BAND_STACK)
                .spawn_scoped(s, move || run_band(f, i + 1, band))
                .ok();
            handles.push(h);
        }
        let mut res = run_band(f, 0, first);
        for (i, (band, h)) in others.iter().zip(handles).enumerate() {
            let r = match h {
                Some(h) => h
                    .join()
                    .unwrap_or(Err(Error::InvalidData("band worker panicked"))),
                None => run_band(f, i + 1, band),
            };
            if res.is_ok() {
                res = r;
            }
        }
        res
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::Limits;
    use crate::ps::tests::{ledger, pps_bits, sps_bits};
    use crate::ps::{activate, parse_pps, parse_sps};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn active(w: u32, h: u32, tiles: u32, l: &Ledger) -> Active {
        let sps = Arc::new(parse_sps(&sps_bits(w, h)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(tiles)).unwrap());
        activate(sps, pps, &Limits::default(), l).unwrap()
    }

    fn slice(addr: u32) -> SliceParams {
        SliceParams {
            slice_addr_rs: addr,
            qp_y: 26,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            loop_filter_across_slices: true,
            sao_luma: true,
            sao_chroma: true,
        }
    }

    #[test]
    fn new_fills_and_charges() {
        let l = ledger();
        let start = l.left();
        let act = active(128, 64, 1, &l);
        let before = l.left();
        let mut f = Frame::new(&act, &l).unwrap();
        assert_eq!(before - l.left(), f.charged);
        assert_eq!(f.num_planes(), 3);
        assert_eq!((f.width, f.height), ([128, 64, 64], [64, 32, 32]));
        assert_eq!(f.sample(0, 127, 63), Some(128));
        assert_eq!(f.sample(2, 63, 31), Some(128));
        assert_eq!(f.sample(2, 64, 0), None);
        assert_eq!((f.grid_w, f.grid_h), (32, 16));
        assert_eq!(f.intra_mode_at(5, 5), Some(1));
        assert_eq!(f.qp_y_at(128, 0), None);
        f.set_sample(1, 3, 2, 77).unwrap();
        assert_eq!(f.sample(1, 3, 2), Some(77));
        assert!(f.set_sample(1, 64, 2, 1).is_err());
        assert!(f.row_mut(0, 120, 0, 9).is_err());
        assert_eq!(f.row_mut(0, 120, 0, 8).unwrap().len(), 8);
        // Block fills cover every 4x4 cell they touch.
        f.fill_qp_y(4, 4, 8, 8, -3).unwrap();
        assert_eq!(f.qp_y_at(4, 4), Some(-3));
        assert_eq!(f.qp_y_at(11, 11), Some(-3));
        assert_eq!(f.qp_y_at(12, 4), Some(0));
        assert_eq!(f.qp_y_at(3, 4), Some(0));
        assert!(f.fill_ct_depth(124, 0, 8, 4, 1).is_err());
        f.fill_edge_v(8, 0, 4, 16, 2).unwrap();
        assert_eq!(f.edge_v_at(8, 12), Some(2));
        assert_eq!(f.edge_v_at(8, 16), Some(0));
        let held = f.charged;
        let i = f.push_slice(slice(0), &l).unwrap();
        assert_eq!(i, 0);
        // The first push grows the Vec to 4 records, and all 4 are booked.
        assert_eq!(f.charged, held + 4 * size_of::<SliceParams>());
        for k in 1..5 {
            assert_eq!(f.push_slice(slice(0), &l).unwrap(), k);
        }
        assert_eq!(
            f.charged,
            held + f.slices.capacity() * size_of::<SliceParams>()
        );
        l.release(f.charged);
        l.release(act.charged);
        assert_eq!(l.left(), start);
    }

    #[test]
    fn alloc_limit_leaves_nothing_booked() {
        let l = ledger();
        let act = active(64, 64, 1, &l);
        let tiny = Ledger::default();
        tiny.reset(crate::ledger::SMALL + 100, 0).unwrap();
        assert_eq!(Frame::new(&act, &tiny).unwrap_err(), Error::AllocLimit);
        assert_eq!(tiny.left(), 100);
    }

    #[test]
    fn availability_6_4_1() {
        // 128x128 = 4x4 CTBs of 32, 2x2 uniform tiles; min TB 4.
        let l = ledger();
        let act = active(128, 128, 2, &l);
        let mut f = Frame::new(&act, &l).unwrap();
        let a = f.push_slice(slice(0), &l).unwrap();
        let b = f.push_slice(slice(4), &l).unwrap();
        // CTB rs 0, 1 (tile 0) and rs 2 (tile 1) in slice A; rs 4 (tile 0) in slice B.
        for (rs, s) in [(0, a), (1, a), (2, a), (4, b)] {
            f.start_ctb(rs, s).unwrap();
        }
        assert!(f.ctb_started(4) && !f.ctb_started(5));
        assert_eq!(f.ctb_slice(4).unwrap().slice_addr_rs, 4);
        // Left CTB, same slice and tile.
        assert!(f.available(&act, 32, 0, 31, 0));
        // Later in z-scan order.
        assert!(!f.available(&act, 0, 0, 32, 0));
        // Outside the picture.
        assert!(!f.available(&act, 0, 0, -1, 0));
        assert!(!f.available(&act, 0, 0, 0, -1));
        assert!(!f.available(&act, 124, 0, 128, 0));
        // Other tile.
        assert!(!f.available(&act, 64, 0, 63, 0));
        // Other slice (CTB rs 4 is slice B, the CTB above it is slice A).
        assert!(!f.available(&act, 0, 32, 0, 31));
        // Current CTB not started.
        assert!(!f.available(&act, 32, 32, 31, 32));
        // Inside one CTB: earlier z-scan position yes, later no.
        assert!(f.available(&act, 8, 8, 4, 4));
        assert!(!f.available(&act, 4, 4, 8, 8));
        // CTB rs 2 is tile-scan address 4, after rs 1 (address 1).
        assert!(!f.available(&act, 32, 0, 64, 0));
        l.release(f.charged);
        l.release(act.charged);
    }

    #[test]
    fn band_cuts() {
        assert_eq!(band_rows(5, 2), [0, 2, 5]);
        assert_eq!(band_rows(5, 64), [0, 1, 2, 3, 4, 5]);
        assert_eq!(band_rows(5, 1), [0, 5]);
        assert_eq!(band_rows(5, 0), [0, 5]);
        assert_eq!(plane_cuts(&[0, 1, 2, 3], 32, 0, 72), [0, 32, 64, 72]);
        assert_eq!(plane_cuts(&[0, 1, 2, 3], 32, 4, 72), [0, 28, 60, 72]);
        assert_eq!(plane_cuts(&[0, 1, 2, 3], 16, 4, 36), [0, 12, 28, 36]);
    }

    #[test]
    fn par_bands_first_error_in_band_order() {
        let mut plane = vec![0u16; 4 * 8];
        let calls = AtomicUsize::new(0);
        let r = par_bands(&mut plane, 4, &[0, 2, 4, 8], |k, row0, band| {
            calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(row0, [0, 2, 4][k]);
            band.fill(u16::try_from(k + 1).unwrap());
            match k {
                1 => Err(Error::InvalidData("one")),
                2 => Err(Error::InvalidData("two")),
                _ => Ok(()),
            }
        });
        assert_eq!(r, Err(Error::InvalidData("one")));
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        let want: Vec<u16> = [1, 1, 2, 2, 3, 3, 3, 3]
            .iter()
            .flat_map(|&v| [v; 4])
            .collect();
        assert_eq!(plane, want);
    }

    #[test]
    fn par_bands_threads_and_bad_cuts() {
        let me = thread::current().id();
        let mut plane = [0u16; 16];
        let ids = Mutex::new(Vec::new());
        let note = |k: usize, _: usize, _: &mut [u16]| {
            ids.lock().unwrap().push((k, thread::current().id()));
            Ok(())
        };
        // One band: no scope, no spawn.
        par_bands(&mut plane, 4, &[0, 4], note).unwrap();
        assert_eq!(*ids.lock().unwrap(), [(0, me)]);
        ids.lock().unwrap().clear();
        // Several bands: band 0 stays on the calling thread, the others do not run on it.
        par_bands(&mut plane, 4, &[0, 1, 2, 4], note).unwrap();
        let got = ids.lock().unwrap().clone();
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|&(k, id)| (id == me) == (k == 0)));
        let bad = |cuts: &[usize]| par_bands(&mut [0u16; 16], 4, cuts, |_, _, _| Ok(()));
        let e = Err(Error::InvalidData("band cuts"));
        assert_eq!(bad(&[0, 5]), e);
        assert_eq!(bad(&[1, 2]), e);
        assert_eq!(bad(&[0, 3, 2]), e);
        assert_eq!(bad(&[0, 4]), Ok(()));
        assert_eq!(bad(&[]), Ok(()));
    }

    /// Threads used never exceed the bands (a process-wide /proc sampler would race with the other
    /// unit tests, so the threads are counted by id).
    #[test]
    fn par_bands_uses_at_most_one_thread_per_band() {
        let ids = Mutex::new(std::collections::HashSet::new());
        let mut plane = [0u16; 64];
        let cuts: Vec<usize> = (0..=8).collect();
        par_bands(&mut plane, 8, &cuts, |_, _, _| {
            ids.lock().unwrap().insert(thread::current().id());
            Ok(())
        })
        .unwrap();
        assert!(ids.lock().unwrap().len() <= 8);
    }
}
