//! slice_segment_data (H.265 7.3.8.1 .. 7.3.8.10) for I slices: parsing and reconstruction in
//! decoding order. For every transform block the intra prediction runs first, then the residual
//! is parsed (`residual.rs`) and added (`transform.rs`), so later blocks see reconstructed
//! neighbours.
//!
//! Also here: the CABAC context initialisation / WPP / dependent-slice rules (9.3.1, 9.3.2),
//! SAO syntax (7.3.8.3), the luma / chroma intra mode derivation (8.4.2, 8.4.3), the QP derivation
//! (8.6.1) and the deblocking edge marking (8.7.2.2, 8.7.2.4: every edge is intra, so bS is 2).

use crate::Options;
use crate::bits::BitReader;
use crate::cabac::{Cabac, Contexts, Ctx};
use crate::error::{Error, Result};
use crate::frame::{Frame, NO_FILTER_BYPASS, NO_FILTER_PCM, SaoParams, SliceParams};
use crate::intra::{Scratch, predict};
use crate::ps::Active;
use crate::residual::{ResidualParams, residual_coding, scan_idx};
use crate::slice::SliceHeader;
use crate::transform::{TuParams, reconstruct};
use std::sync::atomic::{AtomicBool, Ordering};

/// State that outlives one slice segment of a picture.
pub(crate) struct SliceState {
    /// TableStateIdxDs / TableMpsValDs: the contexts at the end of the previous slice segment
    /// (stored only when dependent_slice_segments_enabled_flag is 1).
    ctx_ds: Option<Contexts>,
    /// TableStateIdxWpp / TableMpsValWpp (9.3.1 storage process).
    ctx_wpp: Option<Contexts>,
    /// qPY_PREV carry: QpY of the last CU decoded (8.6.1); a dependent slice segment continues
    /// the QG sequence of its slice.
    prev_qp: i32,
    /// Coefficient scratch for one transform block (32x32 at most).
    coeffs: [i32; 1024],
    /// Block buffers of intra prediction.
    intra: Scratch,
}

impl Default for SliceState {
    fn default() -> Self {
        Self {
            ctx_ds: None,
            ctx_wpp: None,
            prev_qp: 0,
            coeffs: [0; 1024],
            intra: Scratch::default(),
        }
    }
}

/// A transform tree node: location, parent location (base), size, depth, blkIdx.
#[derive(Clone, Copy)]
struct Tt {
    x0: usize,
    y0: usize,
    xb: usize,
    yb: usize,
    log2: u32,
    depth: u32,
    blk: usize,
}

/// Per coding unit values.
#[derive(Default)]
struct Cu {
    x0: usize,
    y0: usize,
    bypass: bool,
    /// IntraSplitFlag (PART_NxN).
    nxn: bool,
    max_depth: u32,
    /// IntraPredModeC (ChromaArrayType 1).
    chroma_mode: u32,
    /// filterLeftCbEdgeFlag, filterTopCbEdgeFlag (8.7.2.1).
    filter_left: bool,
    filter_top: bool,
    /// QpY with the CuQpDeltaVal known so far.
    qp_y: i32,
}

struct Dec<'a, 'b> {
    frame: &'a mut Frame,
    act: &'a Active,
    st: &'a mut SliceState,
    cabac: Cabac<'b>,
    ctx: Contexts,
    sp: SliceParams,
    slice_idx: u32,
    chroma: bool,
    deblock: bool,
    rs: usize,
    ts: usize,
    /// IsCuQpDeltaCoded, CuQpDeltaVal, qPY_PRED of the current quantization group.
    qg_coded: bool,
    cu_qp_delta: i32,
    qp_pred: i32,
    cu: Cu,
}

fn ctx_at(a: &mut [Ctx], i: usize) -> Result<&mut Ctx> {
    a.get_mut(i).ok_or(Error::InvalidData("context index"))
}

/// Zero bits up to the next byte boundary (pcm_alignment_zero_bit, byte_alignment() after the
/// 1 bit, rbsp_trailing_bits() after the stop bit).
fn align_zero(r: &mut BitReader) -> Result<()> {
    while !r.byte_aligned() {
        if r.flag()? {
            return Err(Error::InvalidData("non-zero alignment bit"));
        }
    }
    Ok(())
}

/// Table 8-10 for ChromaArrayType 1.
fn qp_c(qpi: i32) -> i32 {
    match qpi {
        ..=29 => qpi,
        30..=43 => {
            const T: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
            usize::try_from(qpi - 30)
                .ok()
                .and_then(|i| T.get(i))
                .copied()
                .unwrap_or(qpi)
        }
        _ => qpi - 6,
    }
}

/// Table 8-2 for ChromaArrayType 1.
fn chroma_pred_mode(intra_chroma_pred_mode: u32, luma: u32) -> u32 {
    let m = match intra_chroma_pred_mode {
        0 => 0,
        1 => 26,
        2 => 10,
        3 => 1,
        _ => return luma,
    };
    if m == luma { 34 } else { m }
}

/// 7.3.8.1 slice_segment_data for the slice segment whose record is `frame.slices[slice_idx]`.
/// `rbsp` is the unescaped slice segment NAL payload (the header is `hdr`, data starts at
/// `hdr.data_offset`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_slice_segment(
    frame: &mut Frame,
    act: &Active,
    hdr: &SliceHeader,
    slice_idx: u16,
    rbsp: &[u8],
    st: &mut SliceState,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<()> {
    let pps = &act.pps;
    let sp = *frame
        .slices
        .get(usize::from(slice_idx))
        .ok_or(Error::InvalidData("slice index"))?;
    let data = rbsp
        .get(hdr.data_offset..)
        .ok_or(Error::InvalidData("slice data offset"))?;
    let cabac = Cabac::new(BitReader::new(data))?;
    let mut ts = *act
        .rs_to_ts
        .get(hdr.segment_address as usize)
        .ok_or(Error::InvalidData("slice_segment_address"))? as usize;
    let mut d = Dec {
        frame,
        act,
        st,
        cabac,
        ctx: Contexts::init(sp.qp_y),
        sp,
        slice_idx: u32::from(slice_idx),
        chroma: act.sps.chroma_format_idc != 0,
        deblock: opts.deblocking && !sp.deblocking_disabled,
        rs: 0,
        ts,
        qg_coded: false,
        cu_qp_delta: 0,
        qp_pred: sp.qp_y,
        cu: Cu::default(),
    };
    let w = act.sps.pic_width_in_ctbs_y as usize;
    let mut first = true;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let rs = *act
            .ts_to_rs
            .get(ts)
            .ok_or(Error::InvalidData("CTB address"))? as usize;
        if d.frame.ctb_started(rs) {
            return Err(Error::InvalidData("CTB decoded twice"));
        }
        d.frame.start_ctb(rs, d.slice_idx)?;
        (d.rs, d.ts) = (rs, ts);
        d.init_ctb(first, hdr.dependent)?;
        first = false;
        d.ctu()?;
        // 9.3.1 storage process for WPP.
        let tile = |r: usize| {
            act.rs_to_ts
                .get(r)
                .and_then(|&t| act.tile_id.get(t as usize))
        };
        if pps.entropy_coding_sync_enabled
            && (rs % w == 1 || (rs > 1 && act.tile_id.get(ts) != tile(rs - 2)))
        {
            d.st.ctx_wpp = Some(d.ctx.clone());
        }
        let end = d.cabac.terminate()?;
        ts += 1;
        if end {
            break;
        }
        let next = *act
            .ts_to_rs
            .get(ts)
            .ok_or(Error::InvalidData("slice data runs past the picture"))?
            as usize;
        let new_tile = pps.tiles_enabled && act.tile_id.get(ts) != act.tile_id.get(ts - 1);
        let new_row = pps.entropy_coding_sync_enabled
            && (next.is_multiple_of(w) || act.tile_id.get(ts) != tile(next - 1));
        if new_tile || new_row {
            if !d.cabac.terminate()? {
                return Err(Error::InvalidData("end_of_subset_one_bit is 0"));
            }
            align_zero(d.cabac.reader())?;
            d.cabac.reinit()?;
        }
    }
    if pps.dependent_slice_segments_enabled {
        d.st.ctx_ds = Some(d.ctx.clone());
    }
    // rbsp_slice_segment_trailing_bits: the stop bit was the last bit read by the terminate bin;
    // then zero bits to the byte boundary and cabac_zero_words. Bytes after that are ignored, as
    // libde265 and libheif do: some HEIF writers leave junk after the slice data of an item
    // (MPEG FileFormatConformance iff_hevc_single_item); the decoded picture is not affected.
    align_zero(d.cabac.reader())?;
    Ok(())
}

impl Dec<'_, '_> {
    fn sps(&self) -> &crate::ps::Sps {
        &self.act.sps
    }

    /// Context initialisation / synchronisation at the start of a CTU (9.3.2.1), and the
    /// qPY_PREV resets of 8.6.1 (first QG in a slice, tile, or WPP CTB row).
    fn init_ctb(&mut self, first_in_seg: bool, dependent: bool) -> Result<()> {
        let (act, rs, ts) = (self.act, self.rs, self.ts);
        let w = act.sps.pic_width_in_ctbs_y as usize;
        let first_in_tile = ts == 0 || act.tile_id.get(ts) != act.tile_id.get(ts - 1);
        let row_start = act.pps.entropy_coding_sync_enabled
            && (rs.is_multiple_of(w)
                || act.tile_id.get(ts)
                    != act
                        .rs_to_ts
                        .get(rs - 1)
                        .and_then(|&t| act.tile_id.get(t as usize)));
        if first_in_tile {
            self.ctx = Contexts::init(self.sp.qp_y);
            self.st.prev_qp = self.sp.qp_y;
        } else if row_start {
            let cs = self.sps().ctb_size_y as i32;
            let (x0, y0) = self.ctb_xy();
            let avail =
                self.frame
                    .available(act, x0 as i32, y0 as i32, x0 as i32 + cs, y0 as i32 - cs);
            self.ctx = if avail {
                self.st
                    .ctx_wpp
                    .clone()
                    .ok_or(Error::InvalidData("missing WPP contexts"))?
            } else {
                Contexts::init(self.sp.qp_y)
            };
            self.st.prev_qp = self.sp.qp_y;
        } else if first_in_seg {
            if dependent {
                self.ctx = self
                    .st
                    .ctx_ds
                    .clone()
                    .ok_or(Error::InvalidData("missing dependent slice contexts"))?;
            } else {
                self.ctx = Contexts::init(self.sp.qp_y);
                self.st.prev_qp = self.sp.qp_y;
            }
        }
        Ok(())
    }

    fn ctb_xy(&self) -> (usize, usize) {
        let s = self.sps();
        let w = s.pic_width_in_ctbs_y as usize;
        (
            (self.rs % w) << s.ctb_log2_size_y,
            (self.rs / w) << s.ctb_log2_size_y,
        )
    }

    /// 7.3.8.2 coding_tree_unit.
    fn ctu(&mut self) -> Result<()> {
        let (x0, y0) = self.ctb_xy();
        if self.sp.sao_luma || self.sp.sao_chroma {
            self.sao()?;
        }
        let log2 = self.sps().ctb_log2_size_y;
        self.quadtree(x0, y0, log2, 0)
    }

    fn tile_of(&self, rs: usize) -> Option<u32> {
        let ts = *self.act.rs_to_ts.get(rs)? as usize;
        self.act.tile_id.get(ts).copied()
    }

    /// 7.3.8.3 sao (and 7.4.9.3 semantics).
    fn sao(&mut self) -> Result<()> {
        let rs = self.rs;
        let w = self.sps().pic_width_in_ctbs_y as usize;
        let slice_addr = self.sp.slice_addr_rs as usize;
        let (mut left, mut up) = (false, false);
        if !rs.is_multiple_of(w) && rs > slice_addr && self.tile_of(rs) == self.tile_of(rs - 1) {
            left = self.cabac.decode(&mut self.ctx.sao_merge)?;
        }
        if rs >= w && !left && rs - w >= slice_addr && self.tile_of(rs) == self.tile_of(rs - w) {
            up = self.cabac.decode(&mut self.ctx.sao_merge)?;
        }
        let p = if left || up {
            let src = if left { rs - 1 } else { rs - w };
            *self
                .frame
                .sao_at(src)
                .ok_or(Error::InvalidData("SAO neighbour"))?
        } else {
            self.sao_params()?
        };
        *self
            .frame
            .sao_at_mut(rs)
            .ok_or(Error::InvalidData("SAO address"))? = p;
        Ok(())
    }

    fn sao_params(&mut self) -> Result<SaoParams> {
        let mut p = SaoParams::default();
        let n = if self.chroma { 3 } else { 1 };
        for c in 0..n {
            if (c == 0 && !self.sp.sao_luma) || (c > 0 && !self.sp.sao_chroma) {
                continue;
            }
            if c < 2 {
                // sao_type_idx: TR cMax 2, first bin context coded, second bypass.
                let t = if !self.cabac.decode(&mut self.ctx.sao_type_idx)? {
                    0
                } else if self.cabac.bypass()? {
                    2
                } else {
                    1
                };
                *p.type_idx.get_mut(c).ok_or(Error::InvalidData("SAO"))? = t;
            } else {
                p.type_idx[2] = p.type_idx[1];
            }
            let t = *p.type_idx.get(c).ok_or(Error::InvalidData("SAO"))?;
            if t == 0 {
                continue;
            }
            let depth = if c == 0 {
                self.sps().bit_depth_luma
            } else {
                self.sps().bit_depth_chroma
            };
            let c_max = (1u32 << (depth.min(10) - 5)) - 1;
            let mut abs = [0i32; 4];
            for a in &mut abs {
                let mut v = 0u32;
                while v < c_max && self.cabac.bypass()? {
                    v += 1;
                }
                *a = v as i32;
            }
            let mut neg = [false, false, true, true];
            if t == 1 {
                for (a, ng) in abs.iter().zip(&mut neg) {
                    *ng = *a != 0 && self.cabac.bypass()?;
                }
                *p.band_position
                    .get_mut(c)
                    .ok_or(Error::InvalidData("SAO"))? = self.cabac.bypass_bits(5)? as u8;
            } else if c < 2 {
                *p.eo_class.get_mut(c).ok_or(Error::InvalidData("SAO"))? =
                    self.cabac.bypass_bits(2)? as u8;
            } else {
                p.eo_class[2] = p.eo_class[1];
            }
            let scale = if c == 0 {
                self.act.pps.log2_sao_offset_scale_luma
            } else {
                self.act.pps.log2_sao_offset_scale_chroma
            };
            // 7-72: SaoOffsetVal[i + 1] = sign * (abs << log2OffsetScale).
            let row = p.offset_val.get_mut(c).ok_or(Error::InvalidData("SAO"))?;
            for (i, (a, ng)) in abs.iter().zip(&neg).enumerate() {
                let v = (*a << scale) * if *ng { -1 } else { 1 };
                *row.get_mut(i + 1).ok_or(Error::InvalidData("SAO"))? =
                    i16::try_from(v).map_err(|_| Error::InvalidData("SAO offset"))?;
            }
        }
        Ok(p)
    }

    /// 7.3.8.4 coding_quadtree.
    fn quadtree(&mut self, x0: usize, y0: usize, log2: u32, depth: u32) -> Result<()> {
        let (pw, ph) = (self.sps().width as usize, self.sps().height as usize);
        let min_cb = self.sps().min_cb_log2_size_y;
        let size = 1usize << log2;
        let split = if x0 + size <= pw && y0 + size <= ph && log2 > min_cb {
            // 9.3.4.2.2: ctxInc from the CtDepth of the left and above blocks.
            let (xi, yi) = (x0 as i32, y0 as i32);
            let cond = |s: &Self, xn: i32, yn: i32| {
                s.frame.available(s.act, xi, yi, xn, yn)
                    && s.frame
                        .ct_depth_at(xn as usize, yn as usize)
                        .is_some_and(|d| u32::from(d) > depth)
            };
            let inc = usize::from(cond(self, xi - 1, yi)) + usize::from(cond(self, xi, yi - 1));
            let c = ctx_at(&mut self.ctx.split_cu_flag, inc)?;
            self.cabac.decode(c)?
        } else {
            log2 > min_cb
        };
        let pps = &self.act.pps;
        if pps.cu_qp_delta_enabled
            && log2 + pps.diff_cu_qp_delta_depth >= self.sps().ctb_log2_size_y
        {
            self.qg_coded = false;
            self.cu_qp_delta = 0;
            self.start_qg(x0, y0);
        }
        if split {
            let half = size >> 1;
            let (x1, y1) = (x0 + half, y0 + half);
            self.quadtree(x0, y0, log2 - 1, depth + 1)?;
            if x1 < pw {
                self.quadtree(x1, y0, log2 - 1, depth + 1)?;
            }
            if y1 < ph {
                self.quadtree(x0, y1, log2 - 1, depth + 1)?;
            }
            if x1 < pw && y1 < ph {
                self.quadtree(x1, y1, log2 - 1, depth + 1)?;
            }
            Ok(())
        } else {
            self.coding_unit(x0, y0, log2, depth)
        }
    }

    /// 8.6.1 steps 1 to 4: qPY_PRED of the quantization group at (xq, yq). A neighbour is used
    /// only inside the same CTB, where it is always available (earlier in z-scan, same slice).
    fn start_qg(&mut self, xq: usize, yq: usize) {
        let mask = self.sps().ctb_size_y as usize - 1;
        let prev = self.st.prev_qp;
        let a = if xq & mask != 0 {
            self.frame.qp_y_at(xq - 1, yq).map_or(prev, i32::from)
        } else {
            prev
        };
        let b = if yq & mask != 0 {
            self.frame.qp_y_at(xq, yq - 1).map_or(prev, i32::from)
        } else {
            prev
        };
        self.qp_pred = (a + b + 1) >> 1;
    }

    /// QpY (8-283) from the current qPY_PRED and CuQpDeltaVal.
    fn qp_y_now(&self) -> i32 {
        let off = self.sps().qp_bd_offset_y;
        (self.qp_pred + self.cu_qp_delta + 52 + 2 * off).rem_euclid(52 + off) - off
    }

    /// 8.7.2.1 filterLeftCbEdgeFlag / filterTopCbEdgeFlag for a CU at (x0, y0).
    fn cu_edge_flags(&self, x0: usize, y0: usize) -> (bool, bool) {
        let mask = self.sps().ctb_size_y as usize - 1;
        let left = x0 != 0 && (x0 & mask != 0 || self.edge_across(x0 - 1, y0));
        let top = y0 != 0 && (y0 & mask != 0 || self.edge_across(x0, y0 - 1));
        (left, top)
    }

    /// Whether the CTB edge towards the neighbouring luma sample (xn, yn) may be filtered:
    /// tile boundary with loop_filter_across_tiles_enabled_flag 0 and slice boundary with the
    /// current slice's slice_loop_filter_across_slices_enabled_flag 0 stop it.
    fn edge_across(&self, xn: usize, yn: usize) -> bool {
        let Some(nb) = Frame::ctb_addr(self.act, xn as i32, yn as i32) else {
            return false;
        };
        if !self.act.pps.loop_filter_across_tiles_enabled
            && self.tile_of(nb) != self.tile_of(self.rs)
        {
            return false;
        }
        if !self.sp.loop_filter_across_slices {
            return self
                .frame
                .ctb_slice(nb)
                .is_some_and(|s| s.slice_addr_rs == self.sp.slice_addr_rs);
        }
        true
    }

    /// 8.7.2.2 for a leaf transform block (or a PCM CU): marks its left and top edges on the 8x8
    /// grid with bS 2 (all blocks are intra, 8.7.2.4).
    fn mark_tb(&mut self, x: usize, y: usize, log2: u32) -> Result<()> {
        if !self.deblock {
            return Ok(());
        }
        let n = 1usize << log2;
        if x & 7 == 0 && (x != self.cu.x0 || self.cu.filter_left) {
            self.frame.fill_edge_v(x, y, 4, n, 2)?;
        }
        if y & 7 == 0 && (y != self.cu.y0 || self.cu.filter_top) {
            self.frame.fill_edge_h(x, y, n, 4, 2)?;
        }
        Ok(())
    }

    /// 7.3.8.5 coding_unit (intra only).
    fn coding_unit(&mut self, x0: usize, y0: usize, log2: u32, depth: u32) -> Result<()> {
        let n = 1usize << log2;
        let bypass = self.act.pps.transquant_bypass_enabled
            && self.cabac.decode(&mut self.ctx.cu_transquant_bypass_flag)?;
        // part_mode: bin 1 is PART_2Nx2N, 0 is PART_NxN (only at the minimum CB size).
        let nxn =
            log2 == self.sps().min_cb_log2_size_y && !self.cabac.decode(&mut self.ctx.part_mode)?;
        let pcm_cfg = self.sps().pcm;
        let pcm = match pcm_cfg {
            Some(p) if !nxn && (p.log2_min_size..=p.log2_max_size).contains(&log2) => {
                self.cabac.terminate()?
            }
            _ => false,
        };
        let (filter_left, filter_top) = self.cu_edge_flags(x0, y0);
        self.cu = Cu {
            x0,
            y0,
            bypass,
            nxn,
            max_depth: self.sps().max_transform_hierarchy_depth_intra + u32::from(nxn),
            chroma_mode: 0,
            filter_left,
            filter_top,
            qp_y: self.qp_y_now(),
        };
        self.frame.fill_ct_depth(x0, y0, n, n, depth as u8)?;
        let mut nf = if bypass { NO_FILTER_BYPASS } else { 0 };
        if pcm {
            // 7.3.8.7 pcm_sample after pcm_alignment_zero_bit, then 9.3.2.6 re-initialisation.
            align_zero(self.cabac.reader())?;
            self.pcm_samples(x0, y0, log2)?;
            self.cabac.reinit()?;
            // candIntraPredModeX of a PCM block is INTRA_DC (8.4.2).
            self.frame.fill_intra_mode(x0, y0, n, n, 1)?;
            if pcm_cfg.is_some_and(|p| p.loop_filter_disabled) {
                nf |= NO_FILTER_PCM;
            }
            self.mark_tb(x0, y0, log2)?;
        } else {
            self.intra_modes(x0, y0, n)?;
            self.transform_tree(
                Tt {
                    x0,
                    y0,
                    xb: x0,
                    yb: y0,
                    log2,
                    depth: 0,
                    blk: 0,
                },
                (false, false),
            )?;
        }
        if nf != 0 {
            self.frame.fill_no_filter(x0, y0, n, n, nf)?;
        }
        // The QpY of a CU covers the whole CU, also without a coded residual (8.6.1).
        let qp = i8::try_from(self.cu.qp_y).map_err(|_| Error::InvalidData("QpY"))?;
        self.frame.fill_qp_y(x0, y0, n, n, qp)?;
        self.st.prev_qp = self.cu.qp_y;
        Ok(())
    }

    /// 7.3.8.7 pcm_sample.
    fn pcm_samples(&mut self, x0: usize, y0: usize, log2: u32) -> Result<()> {
        let pcm = self.sps().pcm.ok_or(Error::InvalidData("pcm"))?;
        let n = 1usize << log2;
        let planes = if self.chroma { 3 } else { 1 };
        for c in 0..planes {
            let (bits, depth, size, x, y) = if c == 0 {
                (pcm.bit_depth_luma, self.sps().bit_depth_luma, n, x0, y0)
            } else {
                (
                    pcm.bit_depth_chroma,
                    self.sps().bit_depth_chroma,
                    n >> 1,
                    x0 >> 1,
                    y0 >> 1,
                )
            };
            let bits = u32::from(bits);
            let shift = depth.saturating_sub(bits);
            for j in 0..size {
                let row = self.frame.row_mut(c, x, y + j, size)?;
                let r = self.cabac.reader();
                for s in row {
                    *s = (r.u(bits)? as u16) << shift;
                }
            }
        }
        Ok(())
    }

    /// 8.4.2 candidate mode of the neighbour (xn, yn) of the PB at (x, y).
    fn cand_mode(&self, x: usize, y: usize, xn: i32, yn: i32, is_b: bool) -> u32 {
        let ctb_log2 = self.sps().ctb_log2_size_y;
        if !self.frame.available(self.act, x as i32, y as i32, xn, yn)
            || (is_b && (yn as i64) < (((y >> ctb_log2) << ctb_log2) as i64))
        {
            return 1;
        }
        self.frame
            .intra_mode_at(xn as usize, yn as usize)
            .map_or(1, u32::from)
    }

    /// prev_intra_luma_pred_flag / mpm_idx / rem_intra_luma_pred_mode (7.3.8.5 order),
    /// intra_chroma_pred_mode, and the mode derivations 8.4.2 and 8.4.3.
    fn intra_modes(&mut self, x0: usize, y0: usize, n: usize) -> Result<()> {
        let (pb, parts) = if self.cu.nxn { (n / 2, 4) } else { (n, 1) };
        let mut prev = [false; 4];
        for p in prev.iter_mut().take(parts) {
            *p = self.cabac.decode(&mut self.ctx.prev_intra_luma_pred_flag)?;
        }
        let mut first_mode = 1;
        for (i, &flag) in prev.iter().enumerate().take(parts) {
            let (x, y) = (x0 + (i & 1) * pb, y0 + (i >> 1) * pb);
            let a = self.cand_mode(x, y, x as i32 - 1, y as i32, false);
            let b = self.cand_mode(x, y, x as i32, y as i32 - 1, true);
            let mut l = if a == b {
                if a < 2 {
                    [0, 1, 26]
                } else {
                    [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
                }
            } else {
                let c = if a != 0 && b != 0 {
                    0
                } else if a != 1 && b != 1 {
                    1
                } else {
                    26
                };
                [a, b, c]
            };
            let mode = if flag {
                // mpm_idx: TR cMax 2, bypass.
                let idx = if !self.cabac.bypass()? {
                    0
                } else {
                    1 + usize::from(self.cabac.bypass()?)
                };
                *l.get(idx).ok_or(Error::InvalidData("mpm_idx"))?
            } else {
                let mut m = self.cabac.bypass_bits(5)?;
                l.sort_unstable();
                for c in l {
                    if m >= c {
                        m += 1;
                    }
                }
                m
            };
            self.frame.fill_intra_mode(x, y, pb, pb, mode as u8)?;
            if i == 0 {
                first_mode = mode;
            }
        }
        if self.chroma {
            // intra_chroma_pred_mode: bin 0 context coded (0 means 4), then 2 bypass bits.
            let icpm = if self.cabac.decode(&mut self.ctx.intra_chroma_pred_mode)? {
                self.cabac.bypass_bits(2)?
            } else {
                4
            };
            self.cu.chroma_mode = chroma_pred_mode(icpm, first_mode);
        }
        Ok(())
    }

    /// 7.3.8.8 transform_tree.
    fn transform_tree(&mut self, t: Tt, parent: (bool, bool)) -> Result<()> {
        let sps = self.sps();
        let (max_tb, min_tb) = (sps.log2_max_trafo_size, sps.log2_min_trafo_size);
        let split = if t.log2 <= max_tb
            && t.log2 > min_tb
            && t.depth < self.cu.max_depth
            && !(self.cu.nxn && t.depth == 0)
        {
            let c = ctx_at(
                &mut self.ctx.split_transform_flag,
                5usize.saturating_sub(t.log2 as usize),
            )?;
            self.cabac.decode(c)?
        } else {
            // Inferred: interSplitFlag is 0 for intra.
            t.log2 > max_tb || (self.cu.nxn && t.depth == 0)
        };
        let (mut cb, mut cr) = (false, false);
        if self.chroma {
            if t.log2 > 2 {
                let d = t.depth as usize;
                if t.depth == 0 || parent.0 {
                    cb = self.cabac.decode(ctx_at(&mut self.ctx.cbf_cb_cr, d)?)?;
                }
                if t.depth == 0 || parent.1 {
                    cr = self.cabac.decode(ctx_at(&mut self.ctx.cbf_cb_cr, d)?)?;
                }
            } else {
                // Not present at log2TrafoSize 2: the parent's flags (7.4.9.8).
                (cb, cr) = parent;
            }
        }
        if split {
            if t.log2 <= 2 {
                return Err(Error::InvalidData("transform tree splits below 4x4"));
            }
            let h = 1usize << (t.log2 - 1);
            for blk in 0..4 {
                self.transform_tree(
                    Tt {
                        x0: t.x0 + (blk & 1) * h,
                        y0: t.y0 + (blk >> 1) * h,
                        xb: t.x0,
                        yb: t.y0,
                        log2: t.log2 - 1,
                        depth: t.depth + 1,
                        blk,
                    },
                    (cb, cr),
                )?;
            }
            return Ok(());
        }
        // cbf_luma is always coded in an intra CU.
        let inc = usize::from(t.depth == 0);
        let cbf_l = self.cabac.decode(ctx_at(&mut self.ctx.cbf_luma, inc)?)?;
        self.transform_unit(t, [cbf_l, cb, cr])?;
        self.mark_tb(t.x0, t.y0, t.log2)
    }

    /// 7.3.8.10 transform_unit with the reconstruction of its blocks.
    fn transform_unit(&mut self, t: Tt, cbf: [bool; 3]) -> Result<()> {
        let [cbf_l, cb, cr] = cbf;
        if cbf_l || (self.chroma && (cb || cr)) {
            self.delta_qp()?;
        }
        self.block(0, t.x0, t.y0, t.log2, cbf_l)?;
        if self.chroma {
            if t.log2 > 2 {
                self.block(1, t.x0 >> 1, t.y0 >> 1, t.log2 - 1, cb)?;
                self.block(2, t.x0 >> 1, t.y0 >> 1, t.log2 - 1, cr)?;
            } else if t.blk == 3 {
                // 4x4 chroma blocks of the four luma blocks, at the parent's location.
                self.block(1, t.xb >> 1, t.yb >> 1, 2, cb)?;
                self.block(2, t.xb >> 1, t.yb >> 1, 2, cr)?;
            }
        }
        Ok(())
    }

    /// delta_qp() of 7.3.8.14 with the binarization of cu_qp_delta_abs (9.3.3.10).
    fn delta_qp(&mut self) -> Result<()> {
        if !self.act.pps.cu_qp_delta_enabled || self.qg_coded {
            return Ok(());
        }
        self.qg_coded = true;
        // Prefix: TR cMax 5, bin 0 uses context 0, bins 1..=4 context 1.
        let mut abs = 0i64;
        if self
            .cabac
            .decode(ctx_at(&mut self.ctx.cu_qp_delta_abs, 0)?)?
        {
            abs = 1;
            while abs < 5
                && self
                    .cabac
                    .decode(ctx_at(&mut self.ctx.cu_qp_delta_abs, 1)?)?
            {
                abs += 1;
            }
        }
        if abs == 5 {
            // Suffix: EG0 bypass.
            let mut k = 0u32;
            while self.cabac.bypass()? {
                abs += 1i64 << k;
                k += 1;
                if k >= 31 {
                    return Err(Error::InvalidData("cu_qp_delta_abs too long"));
                }
            }
            abs += i64::from(self.cabac.bypass_bits(k)?);
        }
        if abs > 0 && self.cabac.bypass()? {
            abs = -abs;
        }
        let half = i64::from(self.sps().qp_bd_offset_y / 2);
        if !(-(26 + half)..=25 + half).contains(&abs) {
            return Err(Error::InvalidData("CuQpDeltaVal out of range"));
        }
        self.cu_qp_delta = abs as i32;
        self.cu.qp_y = self.qp_y_now();
        Ok(())
    }

    /// Qp'Y (c = 0) or Qp'Cb / Qp'Cr (8.6.1) of the current CU.
    fn qp_prime(&self, c: usize) -> i32 {
        let s = self.sps();
        if c == 0 {
            return self.cu.qp_y + s.qp_bd_offset_y;
        }
        let off = if c == 1 {
            self.sp.cb_qp_offset
        } else {
            self.sp.cr_qp_offset
        };
        let qpi = (self.cu.qp_y + off).clamp(-s.qp_bd_offset_c, 57);
        qp_c(qpi) + s.qp_bd_offset_c
    }

    /// Predicts one transform block of component `c` at component sample (x, y) and, when its
    /// cbf is set, parses and adds the residual (8.4.4.1).
    fn block(&mut self, c: usize, x: usize, y: usize, log2: u32, cbf: bool) -> Result<()> {
        let mode = if c == 0 {
            self.frame.intra_mode_at(x, y).map_or(1, u32::from)
        } else {
            self.cu.chroma_mode
        };
        predict(
            &mut self.st.intra,
            self.frame,
            self.act,
            c,
            x,
            y,
            log2,
            mode,
        )?;
        if !cbf {
            return Ok(());
        }
        let qp = self.qp_prime(c);
        let n = 1usize << log2;
        let coeffs = self
            .st
            .coeffs
            .get_mut(..n * n)
            .ok_or(Error::InvalidData("transform size"))?;
        coeffs.fill(0);
        let pps = &self.act.pps;
        let bypass = self.cu.bypass;
        let p = ResidualParams {
            log2_size: log2,
            c_idx: c,
            scan_idx: scan_idx(log2, c, mode),
            transform_skip_allowed: pps.transform_skip_enabled && !bypass && log2 <= 2,
            bypass,
            sign_hiding: pps.sign_data_hiding_enabled && !bypass,
        };
        let out = residual_coding(&mut self.cabac, &mut self.ctx, &p, coeffs)?;
        let s = &self.act.sps;
        let tu = TuParams {
            qp,
            bit_depth: if c == 0 {
                s.bit_depth_luma
            } else {
                s.bit_depth_chroma
            },
            transform_skip: out.transform_skip,
            bypass,
            dst: c == 0 && log2 == 2,
            matrix_id: c,
            log2_size: log2,
        };
        reconstruct(self.frame, self.act, c, x, y, coeffs, &tu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_qp_table_8_10() {
        let want = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(qp_c(30 + i as i32), *w);
        }
        assert_eq!(qp_c(29), 29);
        assert_eq!(qp_c(-5), -5);
        assert_eq!(qp_c(44), 38);
        assert_eq!(qp_c(57), 51);
    }

    #[test]
    fn chroma_mode_table_8_2() {
        // intra_chroma_pred_mode 0..=3 give 0, 26, 10, 1 unless equal to the luma mode (then 34).
        assert_eq!(chroma_pred_mode(0, 5), 0);
        assert_eq!(chroma_pred_mode(0, 0), 34);
        assert_eq!(chroma_pred_mode(1, 26), 34);
        assert_eq!(chroma_pred_mode(2, 10), 34);
        assert_eq!(chroma_pred_mode(3, 1), 34);
        assert_eq!(chroma_pred_mode(3, 7), 1);
        assert_eq!(chroma_pred_mode(4, 33), 33);
    }
}
