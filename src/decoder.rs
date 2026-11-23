//! The NAL loop: parameter sets, picture start and finish, POC (8.3.1), slice dispatch, the
//! in-loop filters, the decoded picture hash check and the DPB output process (C.5.2).

use crate::ctu::{self, SliceState};
use crate::error::{Error, Result};
use crate::frame::{Frame, NO_SLICE, SliceParams};
use crate::ledger::{Held, Ledger};
use crate::limits::Limits;
use crate::nal::{self, Nal};
use crate::ps::{self, Active, Pps, Sps, SubLayerOrdering};
use crate::sei::{self, PictureHash};
use crate::slice::{self, SliceHeader};
use crate::{ChromaFormat, Info, Options, Picture, Plane, Samples, SeiMessage, deblock, sao};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest parameter set NAL payload accepted (design.md, ps.rs).
const MAX_PS_BYTES: usize = 1 << 20;

/// VCL NAL unit types that carry a coded slice segment (7.4.2.2): TRAIL..RASL and IRAP. The
/// reserved VCL types are ignored.
fn is_vcl(kind: u8) -> bool {
    matches!(kind, 0..=9 | 16..=21)
}

/// Unescapes a NAL payload into a fresh buffer that is charged to the ledger until `Held` drops.
fn rbsp<'a>(payload: &[u8], ledger: &'a Ledger) -> Result<(Vec<u8>, Held<'a>)> {
    let held = ledger.charge(payload.len())?;
    let mut v = Vec::new();
    v.try_reserve_exact(payload.len())
        .map_err(|_| Error::OutOfMemory)?;
    nal::unescape(payload, &mut v);
    Ok((v, held))
}

/// Like [`rbsp`] for a parameter set: the payload size is bounded first.
fn ps_rbsp<'a>(payload: &[u8], ledger: &'a Ledger) -> Result<(Vec<u8>, Held<'a>)> {
    if payload.len() > MAX_PS_BYTES {
        return Err(Error::InvalidData("parameter set too large"));
    }
    rbsp(payload, ledger)
}

/// Stream facts of an SPS: the size after the conformance window (offsets in luma samples).
fn info_of(sps: &Sps) -> Result<Info> {
    let chroma_format = match sps.chroma_format_idc {
        0 => ChromaFormat::Mono,
        1 => ChromaFormat::Yuv420,
        _ => return Err(Error::Unsupported("chroma format")),
    };
    let crop = |a: u32, b: u32, total: u32| {
        a.checked_add(b)
            .and_then(|c| total.checked_sub(c))
            .filter(|&n| n > 0)
            .ok_or(Error::InvalidData("conformance window"))
    };
    Ok(Info {
        width: crop(sps.crop_left, sps.crop_right, sps.width)?,
        height: crop(sps.crop_top, sps.crop_bottom, sps.height)?,
        coded_width: sps.width,
        coded_height: sps.height,
        chroma_format,
        bit_depth_luma: sps.bit_depth_luma as u8,
        bit_depth_chroma: sps.bit_depth_chroma as u8,
        profile_idc: sps.profile_idc,
        level_idc: sps.level_idc,
        vui: sps.vui,
    })
}

/// The SEI messages collected for one access unit, with the bytes booked for them in the ledger
/// (`charged`; handed on with the pictures, so only released when the set is dropped unused).
#[derive(Default)]
struct SeiSet {
    msgs: Vec<SeiMessage>,
    /// Bytes counted against `Limits::max_metadata`; equals `charged`.
    charged: usize,
}

impl SeiSet {
    /// Keeps one message. A message that would take the set past `max` bytes (payload plus
    /// `size_of::<SeiMessage>()`) is dropped without error.
    fn add(
        &mut self,
        suffix: bool,
        payload_type: u32,
        payload: &[u8],
        max: usize,
        ledger: &Ledger,
    ) -> Result<()> {
        let cost = payload.len().saturating_add(size_of::<SeiMessage>());
        if self.charged.saturating_add(cost) > max {
            return Ok(());
        }
        ledger.reserve(cost)?;
        let mut v = Vec::new();
        let r = v
            .try_reserve_exact(payload.len())
            .and_then(|()| self.msgs.try_reserve_exact(1));
        if r.is_err() {
            ledger.release(cost);
            return Err(Error::OutOfMemory);
        }
        v.extend_from_slice(payload);
        self.msgs.push(SeiMessage {
            suffix,
            payload_type,
            payload: v,
        });
        self.charged += cost;
        Ok(())
    }

    /// Moves the messages of `other` to the end of this set, in order. A message that would pass
    /// `max` (or cannot get memory) is dropped and its charge released.
    fn append(&mut self, other: &mut SeiSet, max: usize, ledger: &Ledger) {
        for m in std::mem::take(&mut other.msgs) {
            let cost = m.payload.len().saturating_add(size_of::<SeiMessage>());
            // `other` already holds the charge for `m`: it moves with the message or is released.
            if self.charged.saturating_add(cost) > max || self.msgs.try_reserve_exact(1).is_err() {
                ledger.release(cost);
                continue;
            }
            self.msgs.push(m);
            self.charged += cost;
        }
        other.charged = 0;
    }
}

/// Parameter sets seen so far, and the last independent slice segment header of the picture.
struct Store {
    sps: [Option<Arc<Sps>>; 16],
    pps: [Option<Arc<Pps>>; 64],
    prev: Option<SliceHeader>,
}

impl Store {
    fn new() -> Self {
        Self {
            sps: std::array::from_fn(|_| None),
            pps: std::array::from_fn(|_| None),
            prev: None,
        }
    }

    /// The SPS and PPS that a PPS id leads to.
    fn lookup(&self, pps_id: u32) -> Option<(Arc<Sps>, Arc<Pps>)> {
        let pps = self.pps.get(usize::try_from(pps_id).ok()?)?.as_ref()?;
        let sps = self.sps.get(usize::try_from(pps.sps_id).ok()?)?.as_ref()?;
        Some((Arc::clone(sps), Arc::clone(pps)))
    }

    /// VPS, SPS or PPS NAL unit (7.3.2.1 .. 7.3.2.3). A set replaces the one with its id.
    fn param_set(&mut self, nal: &Nal<'_>, ledger: &Ledger) -> Result<()> {
        let (r, _held) = ps_rbsp(nal.payload, ledger)?;
        match nal.kind {
            32 => {
                ps::parse_vps(&r)?;
            }
            33 => {
                let sps = ps::parse_sps(&r)?;
                let id = usize::try_from(sps.sps_id).map_err(|_| Error::InvalidData("sps id"))?;
                let slot = self.sps.get_mut(id).ok_or(Error::InvalidData("sps id"))?;
                *slot = Some(Arc::new(sps));
            }
            _ => {
                let pps = ps::parse_pps(&r)?;
                let id = usize::try_from(pps.pps_id).map_err(|_| Error::InvalidData("pps id"))?;
                let slot = self.pps.get_mut(id).ok_or(Error::InvalidData("pps id"))?;
                *slot = Some(Arc::new(pps));
            }
        }
        Ok(())
    }

    /// Slice segment header (7.3.6.1) of a VCL NAL unit whose unescaped payload is `r`. A P or B
    /// slice is `Error::InterSlice`.
    fn slice_header(&mut self, r: &[u8], kind: u8) -> Result<SliceHeader> {
        let sh = slice::parse(r, kind, &self.sps, &self.pps, self.prev.as_ref())?;
        if !sh.dependent {
            self.prev = Some(sh.clone());
        }
        Ok(sh)
    }

    /// Activates the SPS and PPS at the first slice segment of a picture (7.4.2.4.2) and checks
    /// the limits.
    fn activate(
        &self,
        pps_id: u32,
        limits: &Limits,
        pictures: &mut usize,
        ledger: &Ledger,
    ) -> Result<ps::Active> {
        let (sps, pps) = self.lookup(pps_id).ok_or(Error::InvalidData(
            "slice refers to a missing parameter set",
        ))?;
        if *pictures >= limits.max_pictures {
            return Err(Error::PictureLimit);
        }
        *pictures += 1;
        ps::activate(sps, pps, limits, ledger)
    }
}

/// The picture being decoded. Releases its ledger charges (frame buffers, slice records and
/// scan tables) when it drops.
struct Cur<'a> {
    frame: Frame,
    act: Active,
    ledger: &'a Ledger,
    st: SliceState,
    poc: i32,
    /// PicOutputFlag.
    output: bool,
    pps_id: u32,
    /// SliceAddrRs of the current slice (the address of its independent segment).
    slice_addr_rs: u32,
    /// The decoded picture hash of the suffix SEI that followed the VCL NAL units.
    hash: Option<PictureHash>,
    /// The SEI messages of the access unit so far (moved to the output picture).
    sei: SeiSet,
}

impl Drop for Cur<'_> {
    fn drop(&mut self) {
        self.ledger.release(self.sei.charged);
        self.ledger.release(self.frame.charged);
        self.ledger.release(self.act.charged);
    }
}

/// A decoded picture waiting for output in the DPB ("needed for output"); the conversion to the
/// cropped output picture has already happened and `charged` bytes are booked for it.
struct Waiting {
    pic: Picture,
    charged: usize,
    /// PicLatencyCount.
    latency: u32,
}

/// The "bumping" conditions of C.5.2.2 / C.5.2.3. `check_full` adds the DPB fullness condition
/// of C.5.2.2 (all pictures are intra and none is kept for reference, so the DPB holds exactly
/// the pictures waiting for output).
fn must_bump(dpb: &[Waiting], o: &SubLayerOrdering, check_full: bool) -> bool {
    if dpb.is_empty() {
        return false;
    }
    if dpb.len() > o.max_num_reorder_pics as usize {
        return true;
    }
    if o.max_latency_increase_plus1 != 0 {
        // SpsMaxLatencyPictures (7-9).
        let max_latency = o
            .max_num_reorder_pics
            .saturating_add(o.max_latency_increase_plus1)
            - 1;
        if dpb.iter().any(|w| w.latency >= max_latency) {
            return true;
        }
    }
    check_full && dpb.len() > o.max_dec_pic_buffering_minus1 as usize
}

/// The conformance window of component `c` as `(x0, y0, width, height)` in samples of that
/// component (the SPS offsets are in luma samples).
fn crop_rect(frame: &Frame, sps: &Sps, c: usize) -> Result<(usize, usize, usize, usize)> {
    let bad = || Error::InvalidData("conformance window");
    let (sw, sh) = if c == 0 {
        (1, 1)
    } else {
        (sps.sub_width_c as usize, sps.sub_height_c as usize)
    };
    let (pw, ph) = (
        *frame.width.get(c).ok_or_else(bad)?,
        *frame.height.get(c).ok_or_else(bad)?,
    );
    let x0 = sps.crop_left as usize / sw;
    let y0 = sps.crop_top as usize / sh;
    let w = pw
        .checked_sub(x0 + sps.crop_right as usize / sw)
        .filter(|&v| v > 0)
        .ok_or_else(bad)?;
    let h = ph
        .checked_sub(y0 + sps.crop_bottom as usize / sh)
        .filter(|&v| v > 0)
        .ok_or_else(bad)?;
    Ok((x0, y0, w, h))
}

/// Crops the decoded picture to the conformance window and converts it to the output samples
/// (`U8` when the component bit depth is 8). Returns the picture and the bytes booked in
/// `ledger` for it (kept booked: the picture belongs to the caller).
fn to_picture(
    frame: &Frame,
    sps: &Sps,
    poc: i32,
    hash_ok: Option<bool>,
    sei: Vec<SeiMessage>,
    ledger: &Ledger,
) -> Result<(Picture, usize)> {
    let info = info_of(sps)?;
    let depth_of = |c: usize| {
        if c == 0 {
            sps.bit_depth_luma
        } else {
            sps.bit_depth_chroma
        }
    };
    let mut total = 0usize;
    for c in 0..frame.num_planes() {
        let (_, _, w, h) = crop_rect(frame, sps, c)?;
        let bytes = if depth_of(c) == 8 { 1 } else { 2 };
        total = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(bytes))
            .and_then(|n| n.checked_add(total))
            .ok_or(Error::AllocLimit)?;
    }
    ledger.reserve(total)?;
    let planes = (0..frame.num_planes())
        .map(|c| {
            let (x0, y0, w, h) = crop_rect(frame, sps, c)?;
            let src = frame.planes.get(c).ok_or(Error::InvalidData("plane"))?;
            let stride = *frame.stride.get(c).ok_or(Error::InvalidData("plane"))?;
            if stride == 0 {
                return Err(Error::InvalidData("plane"));
            }
            let n = w * h;
            let rows = src.chunks_exact(stride).skip(y0).take(h);
            let samples = if depth_of(c) == 8 {
                let mut v: Vec<u8> = Vec::new();
                v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
                for row in rows {
                    let r = row.get(x0..x0 + w).ok_or(Error::InvalidData("plane"))?;
                    v.extend(r.iter().map(|&s| s as u8));
                }
                Samples::U8(v)
            } else {
                let mut v: Vec<u16> = Vec::new();
                v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
                for row in rows {
                    let r = row.get(x0..x0 + w).ok_or(Error::InvalidData("plane"))?;
                    v.extend_from_slice(r);
                }
                Samples::U16(v)
            };
            Ok(Plane {
                width: w as u32,
                height: h as u32,
                samples,
            })
        })
        .collect::<Result<Vec<Plane>>>()
        .inspect_err(|_| ledger.release(total))?;
    let pic = Picture {
        info,
        planes,
        poc,
        hash_ok,
        sei,
    };
    Ok((pic, total))
}

/// Whether the decoded picture matches the hash of the SEI (D.3.19), over the uncropped planes.
fn hash_matches(frame: &Frame, sps: &Sps, h: &PictureHash) -> bool {
    let n = frame.num_planes();
    if h.components != n {
        return false;
    }
    let planes: Vec<&[u16]> = frame.planes.iter().map(Vec::as_slice).collect();
    let depths = [
        sps.bit_depth_luma as u8,
        sps.bit_depth_chroma as u8,
        sps.bit_depth_chroma as u8,
    ];
    let got = sei::compute(
        h.kind,
        &planes,
        frame.width.as_slice(),
        frame.height.as_slice(),
        &depths,
    );
    got.iter().zip(&h.values).take(n).all(|(a, b)| a == b)
}

/// Decoder state of one call.
struct Decoder<'a> {
    ledger: &'a Ledger,
    opts: &'a Options,
    limits: &'a Limits,
    cancel: &'a AtomicBool,
    out: &'a mut Vec<Picture>,
    store: Store,
    cur: Option<Cur<'a>>,
    dpb: Vec<Waiting>,
    /// Pictures started so far (also the `max_pictures` count).
    pictures: usize,
    /// The next picture is the first of the stream.
    first_pic: bool,
    /// An end of sequence (or bitstream) NAL unit was seen since the last picture.
    after_eos: bool,
    /// The last IRAP picture had NoRaslOutputFlag = 1: its RASL pictures are skipped.
    skip_rasl: bool,
    /// (slice_pic_order_cnt_lsb, PicOrderCntMsb) of prevTid0Pic.
    prev_tid0: (i64, i64),
    /// Prefix SEI messages seen since the last picture started: they belong to the next one.
    pending: SeiSet,
}

impl<'a> Decoder<'a> {
    fn new(
        ledger: &'a Ledger,
        opts: &'a Options,
        limits: &'a Limits,
        cancel: &'a AtomicBool,
        out: &'a mut Vec<Picture>,
    ) -> Self {
        Self {
            ledger,
            opts,
            limits,
            cancel,
            out,
            store: Store::new(),
            cur: None,
            dpb: Vec::new(),
            pictures: 0,
            first_pic: true,
            after_eos: false,
            skip_rasl: false,
            prev_tid0: (0, 0),
            pending: SeiSet::default(),
        }
    }

    fn stream<'n>(&mut self, nals: impl Iterator<Item = Result<Nal<'n>>>) -> Result<()> {
        for nal in nals {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            let nal = nal?;
            if nal.layer_id > 0 {
                continue;
            }
            match nal.kind {
                32..=34 => self.store.param_set(&nal, self.ledger)?,
                // End of sequence: the picture is complete; the DPB is left alone, so the next
                // IRAP's C.5.2.2 handling decides between discard and flush.
                36 => {
                    self.finish_picture()?;
                    self.after_eos = true;
                }
                // End of bitstream: the picture is complete and the DPB is flushed.
                37 => {
                    self.finish_picture()?;
                    self.flush()?;
                    self.after_eos = true;
                }
                39 => self.sei_nal(&nal, false)?,
                40 => self.sei_nal(&nal, true)?,
                k if is_vcl(k) => self.vcl(&nal)?,
                _ => {}
            }
        }
        self.finish_picture()?;
        self.flush()?;
        if self.pictures == 0 {
            return Err(Error::NoPicture);
        }
        Ok(())
    }

    /// Prefix (type 39) or suffix (type 40) SEI (7.3.2.4). A suffix SEI belongs to the current
    /// picture (its decoded picture hash is checked); a prefix SEI waits for the next picture.
    /// With `Options::metadata`, every message is kept raw. A malformed SEI is ignored (the
    /// messages before the error are kept).
    fn sei_nal(&mut self, nal: &Nal<'_>, suffix: bool) -> Result<()> {
        if suffix && self.cur.is_none() || !suffix && !self.opts.metadata {
            return Ok(());
        }
        let (r, _held) = rbsp(nal.payload, self.ledger)?;
        let set = match self.cur.as_mut() {
            Some(cur) if suffix => {
                if let Some(h) = sei::parse(&r) {
                    cur.hash = Some(h);
                }
                &mut cur.sei
            }
            _ => &mut self.pending,
        };
        if self.opts.metadata {
            for (ty, payload) in sei::walk(&r) {
                set.add(suffix, ty, payload, self.limits.max_metadata, self.ledger)?;
            }
        }
        Ok(())
    }

    /// Drops the pending prefix SEI messages (their picture is not decoded).
    fn clear_pending(&mut self) {
        self.ledger.release(self.pending.charged);
        self.pending = SeiSet::default();
    }

    /// One slice segment NAL unit.
    fn vcl(&mut self, nal: &Nal<'_>) -> Result<()> {
        // 8.1: RASL pictures of an IRAP with NoRaslOutputFlag = 1 are not decoded (the previous
        // picture is complete, so a suffix SEI of the skipped picture cannot reach it).
        if matches!(nal.kind, 8 | 9) && self.skip_rasl {
            self.clear_pending();
            return self.finish_picture();
        }
        let (r, _held) = rbsp(nal.payload, self.ledger)?;
        // first_slice_segment_in_pic_flag is the first bit: the previous picture ends before
        // the header (which may be a P or B slice) is parsed.
        if r.first().is_some_and(|b| b & 0x80 != 0) {
            self.finish_picture()?;
        }
        let sh = self.store.slice_header(&r, nal.kind)?;
        if sh.first_slice_segment_in_pic {
            self.start_picture(nal, &sh)?;
        } else if let Some(cur) = self.cur.as_mut() {
            // A prefix SEI between two slice segments belongs to this picture (7.4.2.4.4).
            cur.sei
                .append(&mut self.pending, self.limits.max_metadata, self.ledger);
        }
        self.slice(&sh, &r)
    }

    /// Activation, POC, C.5.2.2 and the frame allocation of a new picture.
    fn start_picture(&mut self, nal: &Nal<'_>, sh: &SliceHeader) -> Result<()> {
        let ledger = self.ledger;
        let kind = nal.kind;
        let irap = (16..=23).contains(&kind);
        // NoRaslOutputFlag (8.1.3): IDR, BLA, and a CRA that is first in the stream or follows
        // an end of sequence NAL unit (HandleCraAsBlaFlag is 0).
        let no_rasl = irap && (kind <= 20 || self.first_pic || self.after_eos);
        let act = self
            .store
            .activate(sh.pps_id, self.limits, &mut self.pictures, ledger)?;
        let sps = Arc::clone(&act.sps);
        if irap {
            self.skip_rasl = no_rasl;
        }
        // 8.3.1.
        let max_lsb = 1i64 << sps.log2_max_poc_lsb;
        let lsb = i64::from(sh.poc_lsb);
        let msb = if irap && no_rasl {
            0
        } else {
            let (prev_lsb, prev_msb) = self.prev_tid0;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = match i32::try_from(msb + lsb) {
            Ok(p) => p,
            Err(_) => {
                ledger.release(act.charged);
                return Err(Error::InvalidData("PicOrderCntVal out of range"));
            }
        };
        // prevTid0Pic: TemporalId 0, not RASL, RADL or a sub-layer non-reference picture.
        let sub_layer_non_ref = kind <= 14 && kind.is_multiple_of(2);
        if nal.temporal_id == 0 && !(6..=9).contains(&kind) && !sub_layer_non_ref {
            self.prev_tid0 = (lsb, msb);
        }
        // C.5.2.2.
        let Some(ordering) = sps.ordering.last().copied() else {
            ledger.release(act.charged);
            return Err(Error::InvalidData("sub-layer ordering info"));
        };
        let freed = if irap && no_rasl && !self.first_pic {
            // A CRA here follows an end of sequence: NoOutputOfPriorPicsFlag is 1 for it.
            if kind == 21 || sh.no_output_of_prior_pics {
                self.discard_all();
                Ok(())
            } else {
                self.flush()
            }
        } else {
            let mut r = Ok(());
            while r.is_ok() && must_bump(&self.dpb, &ordering, true) {
                r = self.bump();
            }
            r
        };
        if let Err(e) = freed {
            ledger.release(act.charged);
            return Err(e);
        }
        let frame = match Frame::new(&act, ledger) {
            Ok(f) => f,
            Err(e) => {
                ledger.release(act.charged);
                return Err(e);
            }
        };
        self.first_pic = false;
        self.after_eos = false;
        self.cur = Some(Cur {
            frame,
            act,
            ledger,
            st: SliceState::default(),
            poc,
            output: sh.pic_output,
            pps_id: sh.pps_id,
            slice_addr_rs: sh.segment_address,
            hash: None,
            sei: std::mem::take(&mut self.pending),
        });
        Ok(())
    }

    /// Decodes one slice segment into the current picture.
    fn slice(&mut self, sh: &SliceHeader, rbsp: &[u8]) -> Result<()> {
        let ledger = self.ledger;
        let Some(cur) = self.cur.as_mut() else {
            return Err(Error::InvalidData("slice segment outside a picture"));
        };
        if sh.pps_id != cur.pps_id {
            return Err(Error::InvalidData("pps id changes within a picture"));
        }
        if !sh.dependent {
            cur.slice_addr_rs = sh.segment_address;
        }
        let pps = &cur.act.pps;
        let params = SliceParams {
            slice_addr_rs: cur.slice_addr_rs,
            qp_y: pps.init_qp.saturating_add(sh.qp_delta),
            cb_qp_offset: pps.cb_qp_offset + sh.cb_qp_offset,
            cr_qp_offset: pps.cr_qp_offset + sh.cr_qp_offset,
            deblocking_disabled: sh.deblocking_disabled,
            beta_offset_div2: sh.beta_offset_div2,
            tc_offset_div2: sh.tc_offset_div2,
            loop_filter_across_slices: sh.loop_filter_across_slices,
            sao_luma: sh.sao_luma,
            sao_chroma: sh.sao_chroma,
        };
        let idx = cur.frame.push_slice(params, ledger)?;
        let idx = u16::try_from(idx).map_err(|_| Error::InvalidData("too many slice segments"))?;
        ctu::decode_slice_segment(
            &mut cur.frame,
            &cur.act,
            sh,
            idx,
            rbsp,
            &mut cur.st,
            self.opts,
            self.cancel,
        )
    }

    /// Completes the current picture: coverage check, filters, hash check, then C.5.2.3.
    fn finish_picture(&mut self) -> Result<()> {
        let Some(mut cur) = self.cur.take() else {
            return Ok(());
        };
        if cur.frame.slice_idx.contains(&NO_SLICE) {
            return Err(Error::InvalidData(
                "picture is not fully covered by slice segments",
            ));
        }
        let sps = Arc::clone(&cur.act.sps);
        if self.opts.deblocking {
            deblock::deblock(&mut cur.frame, &cur.act, self.cancel)?;
        }
        if self.opts.sao && sps.sao_enabled {
            sao::sao(&mut cur.frame, &cur.act, self.ledger, self.cancel)?;
        }
        let ordering = sps
            .ordering
            .last()
            .copied()
            .ok_or(Error::InvalidData("sub-layer ordering info"))?;
        if cur.output {
            let hash_ok = cur.hash.as_ref().map(|h| hash_matches(&cur.frame, &sps, h));
            self.dpb.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
            let sei = std::mem::take(&mut cur.sei);
            let sei_charged = sei.charged;
            let (mut pic, charged) =
                to_picture(&cur.frame, &sps, cur.poc, hash_ok, sei.msgs, self.ledger)
                    .inspect_err(|_| self.ledger.release(sei_charged))?;
            let charged = charged + sei.charged;
            if !self.opts.metadata {
                pic.info.vui = None;
            }
            // C.5.2.3: latency counts of the pictures that follow the current one in output order.
            for w in &mut self.dpb {
                if w.pic.poc > cur.poc {
                    w.latency = w.latency.saturating_add(1);
                }
            }
            self.dpb.push(Waiting {
                pic,
                charged,
                latency: 0,
            });
        }
        while must_bump(&self.dpb, &ordering, false) {
            self.bump()?;
        }
        Ok(())
    }

    /// C.5.2.4: outputs the waiting picture with the smallest PicOrderCntVal.
    fn bump(&mut self) -> Result<()> {
        let Some(i) = self
            .dpb
            .iter()
            .enumerate()
            .min_by_key(|(_, w)| w.pic.poc)
            .map(|(i, _)| i)
        else {
            return Ok(());
        };
        self.out.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        let w = self.dpb.remove(i);
        self.out.push(w.pic);
        Ok(())
    }

    /// Outputs every waiting picture in POC order.
    fn flush(&mut self) -> Result<()> {
        while !self.dpb.is_empty() {
            self.bump()?;
        }
        Ok(())
    }

    /// Empties the DPB without output (NoOutputOfPriorPicsFlag = 1).
    fn discard_all(&mut self) {
        for w in self.dpb.drain(..) {
            self.ledger.release(w.charged);
        }
    }
}

/// The NAL loop shared by Annex B and length-prefixed input. On an error the pictures already
/// complete in the DPB are output into `out` (the picture being decoded is dropped).
fn run<'n>(
    nals: impl Iterator<Item = Result<Nal<'n>>>,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> Result<()> {
    let ledger = Ledger::default();
    ledger.reset(limits.max_alloc, 0)?;
    let mut d = Decoder::new(&ledger, opts, limits, cancel, out);
    let r = d.stream(nals);
    if r.is_err() {
        d.cur = None;
        // Out of memory while flushing leaves the rest of the DPB behind.
        let _ = d.flush();
    }
    r
}

/// See [`crate::decode_into`].
pub fn decode_into(
    stream: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> Result<()> {
    run(nal::annexb(stream), opts, limits, cancel, out)
}

/// See [`crate::decode`].
pub fn decode(
    stream: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Vec<Picture>> {
    let mut out = Vec::new();
    decode_into(stream, opts, limits, cancel, &mut out)?;
    Ok(out)
}

/// See [`crate::decode_hvcc_into`].
pub fn decode_hvcc_into(
    config: &[u8],
    samples: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> Result<()> {
    let hv = nal::parse_hvcc(config)?;
    let nals = hv
        .nals
        .iter()
        .copied()
        .map(nal::header)
        .chain(nal::length_prefixed(samples, hv.length_size));
    run(nals, opts, limits, cancel, out)
}

/// See [`crate::decode_hvcc`].
pub fn decode_hvcc(
    config: &[u8],
    samples: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Vec<Picture>> {
    let mut out = Vec::new();
    decode_hvcc_into(config, samples, opts, limits, cancel, &mut out)?;
    Ok(out)
}

/// Info of the SPS that the first slice of layer 0 uses (through its PPS), or of the first SPS
/// when there is no slice (an hvcC config alone); nothing is activated.
fn probe_nals<'a>(nals: impl Iterator<Item = Result<Nal<'a>>>, limits: &Limits) -> Result<Info> {
    let ledger = Ledger::default();
    ledger.reset(limits.max_alloc, 0)?;
    let mut sps_list: [Option<Sps>; 16] = Default::default();
    let mut pps_sps: [Option<u32>; 64] = [None; 64];
    let mut first: Option<u32> = None;
    for nal in nals {
        let nal = nal?;
        if nal.layer_id != 0 {
            continue;
        }
        match nal.kind {
            33 => {
                let (r, _held) = ps_rbsp(nal.payload, &ledger)?;
                let sps = ps::parse_sps(&r)?;
                first.get_or_insert(sps.sps_id);
                if let Some(slot) = sps_list.get_mut(sps.sps_id as usize) {
                    *slot = Some(sps);
                }
            }
            34 => {
                let (r, _held) = ps_rbsp(nal.payload, &ledger)?;
                let pps = ps::parse_pps(&r)?;
                if let Some(slot) = pps_sps.get_mut(pps.pps_id as usize) {
                    *slot = Some(pps.sps_id);
                }
            }
            0..=9 | 16..=21 => {
                // 7.3.6.1: first_slice_segment_in_pic_flag, no_output_of_prior_pics_flag (IRAP),
                // slice_pic_parameter_set_id. Three bytes of escaped payload are enough.
                let mut head = Vec::new();
                nal::unescape(nal.payload.get(..8).unwrap_or(nal.payload), &mut head);
                let mut r = crate::bits::BitReader::new(&head);
                r.flag()?;
                if (16..=23).contains(&nal.kind) {
                    r.flag()?;
                }
                let pps_id = r.ue()?;
                let sps_id = pps_sps
                    .get(pps_id as usize)
                    .copied()
                    .flatten()
                    .ok_or(Error::InvalidData("slice refers to a missing PPS"))?;
                let sps = sps_list
                    .get(sps_id as usize)
                    .and_then(Option::as_ref)
                    .ok_or(Error::InvalidData("PPS refers to a missing SPS"))?;
                sps.check_limits(limits)?;
                return info_of(sps);
            }
            _ => {}
        }
    }
    let sps = first
        .and_then(|id| sps_list.get(id as usize))
        .and_then(Option::as_ref)
        .ok_or(Error::InvalidData("no SPS in the stream"))?;
    sps.check_limits(limits)?;
    info_of(sps)
}

/// See [`crate::probe`].
pub fn probe(stream: &[u8], limits: &Limits) -> Result<Info> {
    probe_nals(nal::annexb(stream), limits)
}

/// See [`crate::probe_hvcc`].
pub fn probe_hvcc(config: &[u8], limits: &Limits) -> Result<Info> {
    let hv = nal::parse_hvcc(config)?;
    probe_nals(hv.nals.iter().copied().map(nal::header), limits)
}

/// See [`crate::intra_prefix`].
pub fn intra_prefix(stream: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let ledger = Ledger::default();
    ledger.reset(limits.max_alloc, 0)?;
    // Every NAL unit grows by at most one byte (3-byte start code to 4), and a NAL unit takes at
    // least 5 bytes of input, so this bound is never passed.
    let cap = stream
        .len()
        .checked_add(stream.len() / 4 + 8)
        .ok_or(Error::AllocLimit)?;
    ledger.reserve(cap)?;
    let mut out = Vec::new();
    out.try_reserve_exact(cap).map_err(|_| Error::OutOfMemory)?;
    let mut store = Store::new();
    let mut pic_start = 0usize;
    let mut pictures = 0usize;
    for nal in nal::annexb(stream) {
        let nal = nal?;
        if nal.layer_id > 0 {
            continue;
        }
        match nal.kind {
            32..=34 => store.param_set(&nal, &ledger)?,
            k if is_vcl(k) => {
                // first_slice_segment_in_pic_flag is the first bit of the slice header.
                if nal.payload.first().is_some_and(|b| b & 0x80 != 0) {
                    pic_start = out.len();
                    pictures += 1;
                }
                let (r, _held) = rbsp(nal.payload, &ledger)?;
                match store.slice_header(&r, nal.kind) {
                    Ok(_) => {}
                    Err(Error::InterSlice) => {
                        // Drop the whole picture that holds the inter slice, and the rest.
                        out.truncate(pic_start);
                        pictures = pictures.saturating_sub(1);
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            _ => {}
        }
        // The NAL header is rebuilt from the parsed fields (layer 0): 7.3.1.2.
        out.extend_from_slice(&[
            0,
            0,
            0,
            1,
            (nal.kind & 0x3f) << 1,
            nal.temporal_id.wrapping_add(1) & 7,
        ]);
        out.extend_from_slice(nal.payload);
    }
    if pictures == 0 {
        return Err(Error::NoPicture);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_and_cancel() {
        let limits = Limits::default();
        let cancel = AtomicBool::new(false);
        let opts = Options::default();
        assert_eq!(
            decode(&[], &opts, &limits, &cancel).unwrap_err(),
            Error::NoPicture
        );
        assert_eq!(intra_prefix(&[], &limits).unwrap_err(), Error::NoPicture);
        // One NAL unit (AUD, type 35) and the flag already set.
        cancel.store(true, Ordering::Relaxed);
        let s = [0, 0, 1, 35 << 1, 1, 0x50];
        assert_eq!(
            decode(&s, &opts, &limits, &cancel).unwrap_err(),
            Error::Cancelled
        );
    }

    #[test]
    fn sei_set_drops_past_max_metadata_and_charges() {
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        let before = ledger.left();
        let cost = |n: usize| n + size_of::<SeiMessage>();
        let max = cost(10) + cost(4);
        let mut s = SeiSet::default();
        s.add(false, 137, &[1; 10], max, &ledger).unwrap();
        // Would pass the cap: dropped, no error, nothing charged.
        s.add(true, 132, &[2; 5], max, &ledger).unwrap();
        s.add(true, 144, &[3; 4], max, &ledger).unwrap();
        let kept: Vec<_> = s.msgs.iter().map(|m| (m.suffix, m.payload_type)).collect();
        assert_eq!(kept, [(false, 137), (true, 144)]);
        assert_eq!(s.charged, max);
        assert_eq!(before - ledger.left(), max);
        // Over the ledger: AllocLimit, and nothing is kept.
        let small = Ledger::default();
        small.reset(crate::ledger::SMALL + 10, 0).unwrap();
        let mut t = SeiSet::default();
        assert_eq!(
            t.add(false, 137, &[0; 10], usize::MAX, &small).unwrap_err(),
            Error::AllocLimit
        );
        assert!(t.msgs.is_empty());
    }

    #[test]
    fn sei_set_append_keeps_order_and_charges() {
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        let before = ledger.left();
        let cost = |n: usize| n + size_of::<SeiMessage>();
        let max = cost(3) + cost(2);
        let mut cur = SeiSet::default();
        cur.add(false, 137, &[1; 3], max, &ledger).unwrap();
        let mut pending = SeiSet::default();
        pending.add(false, 130, &[2; 2], max, &ledger).unwrap();
        pending.add(false, 144, &[3; 4], max, &ledger).unwrap();
        cur.append(&mut pending, max, &ledger);
        let kept: Vec<_> = cur.msgs.iter().map(|m| m.payload_type).collect();
        assert_eq!(kept, [137, 130]);
        assert_eq!(cur.charged, max);
        assert!(pending.msgs.is_empty());
        assert_eq!(pending.charged, 0);
        // The dropped message's charge is released.
        assert_eq!(before - ledger.left(), max);
    }

    #[test]
    fn zero_budget_is_alloc_limit() {
        let limits = Limits {
            max_alloc: 0,
            ..Limits::default()
        };
        assert_eq!(probe(&[], &limits).unwrap_err(), Error::AllocLimit);
        assert_eq!(intra_prefix(&[], &limits).unwrap_err(), Error::AllocLimit);
    }
}
