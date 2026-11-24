//! The NAL loop: parameter sets, picture start and finish, POC (8.3.1), slice dispatch, the
//! in-loop filters, the decoded picture hash check and the DPB output process (C.5.2).

use crate::ctu::{self, SliceState};
use crate::error::{Error, Result};
use crate::frame::{Frame, NO_SLICE, SliceParams};
use crate::ledger::{Held, Ledger, grow_one};
use crate::limits::Limits;
use crate::nal::{self, Nal};
use crate::ps::{self, Active, Pps, Sps, SubLayerOrdering};
use crate::sei::{self, PictureHash};
use crate::slice::{self, SliceHeader};
use crate::{ChromaFormat, Info, Options, Picture, Plane, SeiMessage, crop, deblock, sao};
use std::collections::{BTreeMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, Mutex};

/// Largest parameter set NAL payload accepted (design.md, ps.rs).
const MAX_PS_BYTES: usize = 1 << 20;

/// VCL NAL unit types that carry a coded slice segment (7.4.2.2): TRAIL..RASL and IRAP. The
/// reserved VCL types are ignored.
fn is_vcl(kind: u8) -> bool {
    matches!(kind, 0..=9 | 16..=21)
}

/// Unescapes a NAL payload into a fresh buffer that is charged to the ledger until `Held` drops.
fn rbsp<'a>(payload: &[u8], ledger: &'a Ledger) -> Result<(Held<'a>, Vec<u8>)> {
    let held = ledger.charge(payload.len())?;
    let mut v = Vec::new();
    v.try_reserve_exact(payload.len())
        .map_err(|_| Error::OutOfMemory)?;
    nal::unescape(payload, &mut v);
    Ok((held, v))
}

/// Like [`rbsp`] for a parameter set: the payload size is bounded first.
fn ps_rbsp<'a>(payload: &[u8], ledger: &'a Ledger) -> Result<(Held<'a>, Vec<u8>)> {
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
    /// `max` (or cannot get memory) is dropped; the bytes booked for the dropped messages are
    /// returned for the caller to release (`other` already held their charge).
    fn append(&mut self, other: &mut SeiSet, max: usize) -> usize {
        let mut freed = 0usize;
        for m in std::mem::take(&mut other.msgs) {
            let cost = m.payload.len().saturating_add(size_of::<SeiMessage>());
            if self.charged.saturating_add(cost) > max || self.msgs.try_reserve_exact(1).is_err() {
                freed = freed.saturating_add(cost);
                continue;
            }
            self.msgs.push(m);
            self.charged += cost;
        }
        other.charged = 0;
        freed
    }
}

/// Parameter sets seen so far, and the last independent slice segment header of the picture.
struct Store {
    sps: [Option<Arc<Sps>>; 16],
    pps: [Option<Arc<Pps>>; 64],
    prev: Option<SliceHeader>,
    /// Bytes booked in the ledger for each stored set, released when the set is replaced.
    sps_cost: [usize; 16],
    pps_cost: [usize; 64],
}

/// Bytes an SPS holds: the struct, the `Arc` header and the heap vectors.
fn sps_cost(s: &Sps) -> usize {
    let st: usize = s
        .st_rps
        .iter()
        .map(|r| size_of::<ps::StRps>() + r.delta_poc.len() * 4 + r.used.len())
        .sum();
    size_of::<Sps>()
        + 2 * size_of::<usize>()
        + s.ordering.len() * size_of::<SubLayerOrdering>()
        + st
        + s.lt_ref_pic_poc_lsb_sps.len() * 4
        + s.used_by_curr_pic_lt_sps.len()
}

/// Bytes a PPS holds: the struct, the `Arc` header and the tile vectors.
pub(crate) fn pps_cost(p: &Pps) -> usize {
    size_of::<Pps>() + 2 * size_of::<usize>() + (p.column_width.len() + p.row_height.len()) * 4
}

/// Stores `set` in `slot`, booking `cost` bytes before the old set is dropped. Returns the old
/// set's `held` bytes, for the caller to release.
fn replace_set<T>(
    slot: &mut Option<Arc<T>>,
    held: &mut usize,
    set: T,
    cost: usize,
    ledger: &Ledger,
) -> Result<usize> {
    ledger.reserve(cost)?;
    *slot = Some(Arc::new(set));
    Ok(std::mem::replace(held, cost))
}

impl Store {
    fn new() -> Self {
        Self {
            sps: std::array::from_fn(|_| None),
            pps: std::array::from_fn(|_| None),
            prev: None,
            sps_cost: [0; 16],
            pps_cost: [0; 64],
        }
    }

    /// The SPS and PPS that a PPS id leads to.
    fn lookup(&self, pps_id: u32) -> Option<(Arc<Sps>, Arc<Pps>)> {
        let pps = self.pps.get(usize::try_from(pps_id).ok()?)?.as_ref()?;
        let sps = self.sps.get(usize::try_from(pps.sps_id).ok()?)?.as_ref()?;
        Some((Arc::clone(sps), Arc::clone(pps)))
    }

    /// VPS, SPS or PPS NAL unit (7.3.2.1 .. 7.3.2.3). A set replaces the one with its id.
    /// Returns the bytes of the replaced set, for the caller to release.
    fn param_set(&mut self, nal: &Nal<'_>, ledger: &Ledger) -> Result<usize> {
        let (_held, r) = ps_rbsp(nal.payload, ledger)?;
        match nal.kind {
            32 => {
                ps::parse_vps(&r)?;
                Ok(0)
            }
            33 => {
                let sps = ps::parse_sps(&r)?;
                let id = usize::try_from(sps.sps_id).map_err(|_| Error::InvalidData("sps id"))?;
                let slot = self.sps.get_mut(id).ok_or(Error::InvalidData("sps id"))?;
                let held = self
                    .sps_cost
                    .get_mut(id)
                    .ok_or(Error::InvalidData("sps id"))?;
                let cost = sps_cost(&sps);
                replace_set(slot, held, sps, cost, ledger)
            }
            _ => {
                let pps = ps::parse_pps(&r)?;
                let id = usize::try_from(pps.pps_id).map_err(|_| Error::InvalidData("pps id"))?;
                let slot = self.pps.get_mut(id).ok_or(Error::InvalidData("pps id"))?;
                let held = self
                    .pps_cost
                    .get_mut(id)
                    .ok_or(Error::InvalidData("pps id"))?;
                let cost = pps_cost(&pps);
                replace_set(slot, held, pps, cost, ledger)
            }
        }
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

/// A slice segment of the current picture that waits for the picture to finish: the front end
/// parses the header, `decode_picture` decodes the data.
struct Pending<'a> {
    sh: SliceHeader,
    /// Index of the segment in `Frame::slices`.
    idx: u16,
    rbsp: Vec<u8>,
    /// Keeps the bytes of `rbsp` booked until the segment drops.
    _held: Held<'a>,
}

/// The picture being decoded. Releases its ledger charges (frame buffers, slice records and
/// scan tables) after its buffers are freed (`booked`).
struct Cur<'a> {
    frame: Frame,
    act: Active,
    ledger: &'a Ledger,
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
    /// The slice segments received so far, not decoded yet.
    slices: Vec<Pending<'a>>,
    /// Every byte booked for this picture. Declared last: fields drop in order, so the bytes
    /// go back to the ledger after the buffers are freed (a worker must not reserve them while
    /// the memory is still there).
    booked: Booked<'a>,
}

/// The bytes booked for a [`Cur`] (frame, tables, SEI messages, `Pending` records and the box),
/// released when it drops. Not `Held`: the total changes while the picture is built.
struct Booked<'a> {
    ledger: &'a Ledger,
    n: usize,
}

impl Drop for Booked<'_> {
    fn drop(&mut self) {
        self.ledger.release(self.n);
    }
}

/// A decoded picture waiting for output in the DPB ("needed for output"). The picture itself is
/// the result of job `ticket` and may not be ready yet.
struct Waiting {
    ticket: usize,
    poc: i32,
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
    cancel: &AtomicBool,
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
    // The `Picture` and its plane list, plus the slack of the caller's output `Vec` (its capacity
    // is at most twice its length).
    total = total
        .checked_add(2 * size_of::<Picture>() + frame.num_planes() * size_of::<Plane>())
        .ok_or(Error::AllocLimit)?;
    ledger.reserve(total)?;
    let planes = (0..frame.num_planes())
        .map(|c| {
            let (x0, y0, w, h) = crop_rect(frame, sps, c)?;
            let src = frame.planes.get(c).ok_or(Error::InvalidData("plane"))?;
            let stride = *frame.stride.get(c).ok_or(Error::InvalidData("plane"))?;
            if stride == 0 {
                return Err(Error::InvalidData("plane"));
            }
            let samples = crop::crop_plane(src, stride, (x0, y0, w, h), depth_of(c) == 8, cancel)?;
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
/// `Error::Cancelled` when `cancel` is set (read once per row).
fn hash_matches(frame: &Frame, sps: &Sps, h: &PictureHash, cancel: &AtomicBool) -> Result<bool> {
    let n = frame.num_planes();
    if h.components != n {
        return Ok(false);
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
        cancel,
    )?;
    Ok(got.iter().zip(&h.values).take(n).all(|(a, b)| a == b))
}

/// Bands of CTB rows for the in-loop filters: at most one per thread (0 counts as 1), one per CTB
/// row, one per 64 Ki luma samples, and 64 (so the per-band bookkeeping stays within the small
/// reserve), so that a small picture is filtered on the calling thread
/// (a thread scope costs more than it saves there).
fn filter_workers(threads: usize, act: &Active) -> usize {
    let sps = &act.sps;
    let by_size =
        usize::try_from((u64::from(sps.width) * u64::from(sps.height)) >> 16).unwrap_or(usize::MAX);
    threads
        .min(sps.pic_height_in_ctbs_y as usize)
        .min(by_size)
        .clamp(1, 64)
}

/// Bytes booked with the box of each picture for its entry in `Decoder::spent` (a B-tree node
/// holds 5 to 11 entries of 16 bytes plus its links, so an entry costs under 64 bytes; the rest
/// is slack). Booked in a serial run too, and released with the picture.
const SPENT_ENTRY: usize = 128;

/// The outcome of one picture job: the output picture and the bytes booked for it (they stay
/// booked), or `None` when the picture is not output.
type Done = Result<Option<(Picture, usize)>>;

/// A job that ended: its ticket, its outcome, and the spent picture. The picture goes back to the
/// front end so that its bytes are released in ticket order (`Decoder::retire`).
type Finished<'a> = (usize, Done, Box<Cur<'a>>);

/// Decodes the slice segments received so far into the frame of `cur`. The segments stay in
/// `cur` (and booked) until it drops.
fn decode_slices(cur: &mut Cur<'_>, opts: &Options, cancel: &AtomicBool) -> Result<()> {
    // Scratch of the CTU loop (about 14 KiB of block buffers): on the stack of the thread that
    // decodes, so it never travels with the picture.
    let mut st = SliceState::default();
    for p in &cur.slices {
        ctu::decode_slice_segment(
            &mut cur.frame,
            &cur.act,
            &p.sh,
            p.idx,
            &p.rbsp,
            &mut st,
            opts,
            cancel,
        )?;
    }
    Ok(())
}

/// The second half of a picture, run on any thread: slice data, coverage check, filters, hash
/// check and conversion to the output picture. `cur` keeps its bytes booked: the caller drops it
/// in ticket order.
fn decode_picture(cur: &mut Cur<'_>, opts: &Options, cancel: &AtomicBool, workers: usize) -> Done {
    decode_slices(cur, opts, cancel)?;
    if cur.frame.slice_idx.contains(&NO_SLICE) {
        return Err(Error::InvalidData(
            "picture is not fully covered by slice segments",
        ));
    }
    let sps = Arc::clone(&cur.act.sps);
    if opts.deblocking {
        deblock::deblock(&mut cur.frame, &cur.act, cancel, workers)?;
    }
    if opts.sao && sps.sao_enabled {
        sao::sao(&mut cur.frame, &cur.act, cur.ledger, cancel, workers)?;
    }
    if !cur.output {
        return Ok(None);
    }
    // The hash and the crop read the cancel flag once per row.
    let hash_ok = if opts.check_hash {
        cur.hash
            .as_ref()
            .map(|h| hash_matches(&cur.frame, &sps, h, cancel))
            .transpose()?
    } else {
        None
    };
    let ledger = cur.ledger;
    let sei = std::mem::take(&mut cur.sei);
    let sei_charged = sei.charged;
    // The messages (and their bytes) now belong to the output picture.
    cur.booked.n -= sei_charged;
    let (mut pic, charged) =
        to_picture(&cur.frame, &sps, cur.poc, hash_ok, sei.msgs, ledger, cancel)
            .inspect_err(|_| ledger.release(sei_charged))?;
    if !opts.metadata {
        pic.info.vui = None;
    }
    Ok(Some((pic, charged + sei_charged)))
}

/// One finished picture (all its slice segments received) and its place in decode order.
struct Job<'a> {
    ticket: usize,
    cur: Box<Cur<'a>>,
}

/// The jobs queue of the workers.
type Jobs<'a> = Mutex<Receiver<Job<'a>>>;

/// A worker thread: takes jobs until the sender is dropped. A panic inside a job becomes an
/// error result, so the caller never waits for a result that cannot come.
fn worker<'a>(rx: &Jobs<'a>, done: &Sender<Finished<'a>>, opts: &Options, cancel: &AtomicBool) {
    loop {
        let job = match rx.lock() {
            Ok(g) => g.recv(),
            Err(_) => return,
        };
        let Ok(Job { ticket, mut cur }) = job else {
            return;
        };
        let r = catch_unwind(AssertUnwindSafe(|| {
            decode_picture(&mut cur, opts, cancel, 1)
        }))
        .unwrap_or(Err(Error::InvalidData("picture decoding panicked")));
        if done.send((ticket, r, cur)).is_err() {
            return;
        }
    }
}

/// Starts a closure on a new thread of the call's scope; false when the thread cannot be made.
type Spawn<'a, 'p> = &'p dyn Fn(Box<dyn FnOnce() + Send + 'a>) -> bool;

/// The threads of one call (docs/threading.md). The channels exist once the second picture of
/// the call has a good header, so a one-picture call never spawns. Workers start one at a time,
/// when a job finds none idle.
struct Pool<'a, 'p> {
    /// `Limits::max_threads` (at least 1), the calling thread included.
    threads: usize,
    /// None when `threads` is 1: no scope exists.
    spawn: Option<Spawn<'a, 'p>>,
    /// Some once the channels exist.
    tx: Option<SyncSender<Job<'a>>>,
    /// The receiving end of `tx`, shared by the workers.
    rx: Option<Arc<Jobs<'a>>>,
    /// Kept to give each new worker its sender. The workers never end before the call does, so
    /// `results` does not disconnect while a job is in flight.
    done_tx: Option<Sender<Finished<'a>>>,
    results: Option<Receiver<Finished<'a>>>,
    /// The most workers this call starts.
    cap: usize,
    /// Workers started.
    workers: usize,
    /// Jobs sent to a worker whose result has not been received yet.
    inflight: usize,
}

/// Decoder state of one call.
struct Decoder<'a, 'p> {
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
    pool: Pool<'a, 'p>,
    /// The ticket of the next picture (decode order).
    next_ticket: usize,
    /// Tickets of the bumped pictures, in output order, until they move to `out`.
    order: VecDeque<usize>,
    /// Finished output pictures by ticket, with the bytes booked for them.
    done: BTreeMap<usize, (Picture, usize)>,
    /// Spent pictures (their jobs are done) that wait for the earlier ones, by ticket.
    spent: BTreeMap<usize, Box<Cur<'a>>>,
    /// The ticket of the next picture to drop: every earlier one is dropped.
    retired: usize,
    /// Releases of the front end that wait for the jobs submitted before them, as (the ticket
    /// that must be retired first, bytes). At most one entry per ticket.
    late: VecDeque<(usize, usize)>,
    /// A worker job failed (or the workers are gone): the call must run again with one thread.
    replay: bool,
    /// A worker job returned `Cancelled`.
    cancelled: bool,
}

impl<'a, 'p> Decoder<'a, 'p> {
    fn new(
        ledger: &'a Ledger,
        opts: &'a Options,
        limits: &'a Limits,
        cancel: &'a AtomicBool,
        out: &'a mut Vec<Picture>,
        spawn: Option<Spawn<'a, 'p>>,
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
            pool: Pool {
                threads: limits.max_threads.max(1),
                spawn,
                tx: None,
                rx: None,
                done_tx: None,
                results: None,
                cap: 0,
                workers: 0,
                inflight: 0,
            },
            next_ticket: 0,
            order: VecDeque::new(),
            done: BTreeMap::new(),
            spent: BTreeMap::new(),
            retired: 0,
            late: VecDeque::new(),
            replay: false,
            cancelled: false,
        }
    }

    fn stream<'n>(&mut self, nals: impl Iterator<Item = Result<Nal<'n>>>) -> Result<()> {
        for nal in nals {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
            self.collect(false);
            self.check()?;
            let nal = nal?;
            if nal.layer_id > 0 {
                continue;
            }
            match nal.kind {
                32..=34 => {
                    let old = self.store.param_set(&nal, self.ledger)?;
                    self.release(old);
                }
                // End of sequence: the picture is complete; the DPB is left alone, so the next
                // IRAP's C.5.2.2 handling decides between discard and flush.
                36 => {
                    self.finish_picture(false)?;
                    self.after_eos = true;
                }
                // End of bitstream: the picture is complete and the DPB is flushed.
                37 => {
                    self.finish_picture(false)?;
                    self.flush()?;
                    self.after_eos = true;
                }
                39 => self.sei_nal(&nal, false)?,
                40 => self.sei_nal(&nal, true)?,
                k if is_vcl(k) => self.vcl(&nal)?,
                _ => {}
            }
        }
        self.finish_picture(false)?;
        self.flush()?;
        self.finish_jobs()?;
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
        let (_held, r) = rbsp(nal.payload, self.ledger)?;
        let (set, booked) = match self.cur.as_mut() {
            Some(cur) if suffix => {
                if let Some(h) = sei::parse(&r) {
                    cur.hash = Some(h);
                }
                (&mut cur.sei, Some(&mut cur.booked))
            }
            _ => (&mut self.pending, None),
        };
        let before = set.charged;
        let mut res = Ok(());
        if self.opts.metadata {
            for (ty, payload) in sei::walk(&r) {
                res = set.add(suffix, ty, payload, self.limits.max_metadata, self.ledger);
                if res.is_err() {
                    break;
                }
            }
        }
        // The messages kept so far stay booked with the picture, an error or not.
        if let Some(b) = booked {
            b.n += set.charged - before;
        }
        res
    }

    /// Drops the pending prefix SEI messages (their picture is not decoded).
    fn clear_pending(&mut self) {
        let n = self.pending.charged;
        self.release(n);
        self.pending = SeiSet::default();
    }

    /// One slice segment NAL unit.
    fn vcl(&mut self, nal: &Nal<'_>) -> Result<()> {
        // 8.1: RASL pictures of an IRAP with NoRaslOutputFlag = 1 are not decoded (the previous
        // picture is complete, so a suffix SEI of the skipped picture cannot reach it).
        if matches!(nal.kind, 8 | 9) && self.skip_rasl {
            self.clear_pending();
            return self.finish_picture(false);
        }
        let (held, r) = rbsp(nal.payload, self.ledger)?;
        // first_slice_segment_in_pic_flag is the first bit: the previous picture ends when the
        // next one starts. The header is parsed first (it does not depend on the picture): only
        // a good header with room for one more picture lets the workers start, so a P slice
        // after the intra prefix, or a one-picture call, spawns nothing. The errors come in the
        // same order as before: the job of the previous picture, then the header.
        let first = r.first().is_some_and(|b| b & 0x80 != 0);
        let sh = self.store.slice_header(&r, nal.kind);
        if first {
            let more = sh.is_ok() && self.pictures < self.limits.max_pictures;
            self.finish_picture(more).or_else(|e| self.fail(e))?;
        }
        let sh = sh.or_else(|e| self.fail(e))?;
        if sh.first_slice_segment_in_pic {
            self.start_picture(nal, &sh).or_else(|e| self.fail(e))?;
        } else if let Some(cur) = self.cur.as_mut() {
            // A prefix SEI between two slice segments belongs to this picture (7.4.2.4.4).
            let before = cur.sei.charged;
            let freed = cur.sei.append(&mut self.pending, self.limits.max_metadata);
            cur.booked.n += cur.sei.charged - before;
            self.release(freed);
        }
        self.slice(sh, r, held)
    }

    /// A front-end error: the jobs in flight finish first, so that the buffers the caller still
    /// holds (`rbsp`, `held`) go back to the ledger after them, as in a serial run.
    fn fail<T>(&mut self, e: Error) -> Result<T> {
        while self.pool.inflight > 0 {
            self.collect(true);
        }
        Err(e)
    }

    /// Gives `n` bytes of the front end back to the ledger. In a serial run the pictures before
    /// this point are done by now, so while their jobs are in flight (`retired < next_ticket`)
    /// the bytes wait for them: the memory the jobs see never drops below the serial one.
    fn release(&mut self, n: usize) {
        if self.retired >= self.next_ticket {
            self.ledger.release(n);
            return;
        }
        let barrier = self.next_ticket;
        if let Some(l) = self.late.back_mut()
            && l.0 == barrier
        {
            l.1 = l.1.saturating_add(n);
            return;
        }
        self.late.push_back((barrier, n));
    }

    /// Releases the waiting bytes whose jobs are all retired.
    fn apply_late(&mut self) {
        while let Some(&(barrier, n)) = self.late.front() {
            if barrier > self.retired {
                break;
            }
            self.late.pop_front();
            self.ledger.release(n);
        }
    }

    /// Drops the spent picture of job `ticket` once every earlier picture is dropped, so the
    /// bytes of a picture never go back to the ledger before those of an earlier one (a serial
    /// run drops them in this order, and the call must not succeed where it would hit
    /// `max_alloc`).
    fn retire(&mut self, ticket: usize, cur: Box<Cur<'a>>) {
        // `unbox` frees the box before the `Cur` (and its booking) drops, so the ledger never
        // gets bytes back that are still allocated.
        fn unbox<'b>(b: Box<Cur<'b>>) -> Cur<'b> {
            *b
        }
        if ticket == self.retired {
            drop(unbox(cur));
            self.retired += 1;
            while let Some(next) = self.spent.remove(&self.retired) {
                drop(unbox(next));
                self.retired += 1;
            }
        } else {
            self.spent.insert(ticket, cur);
        }
        self.apply_late();
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
                self.release(act.charged);
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
            self.release(act.charged);
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
            self.release(act.charged);
            return Err(e);
        }
        let frame = match Frame::new(&act, ledger) {
            Ok(f) => f,
            Err(e) => {
                self.release(act.charged);
                return Err(e);
            }
        };
        self.first_pic = false;
        self.after_eos = false;
        let sei = std::mem::take(&mut self.pending);
        let booked = Booked {
            ledger,
            n: frame.charged + act.charged + sei.charged,
        };
        self.cur = Some(Cur {
            frame,
            act,
            ledger,
            poc,
            output: sh.pic_output,
            pps_id: sh.pps_id,
            slice_addr_rs: sh.segment_address,
            hash: None,
            sei,
            slices: Vec::new(),
            booked,
        });
        Ok(())
    }

    /// Books one slice segment of the current picture in its frame and in `cur.slices`; the
    /// data is decoded when the picture finishes (`decode_picture`). Returns the segment index.
    fn book(&mut self, sh: &SliceHeader) -> Result<u16> {
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
        let before = cur.frame.charged;
        let idx = cur.frame.push_slice(params, ledger)?;
        cur.booked.n += cur.frame.charged - before;
        let idx = u16::try_from(idx).map_err(|_| Error::InvalidData("too many slice segments"))?;
        cur.booked.n += grow_one(&mut cur.slices, ledger)?;
        Ok(idx)
    }

    /// Queues one slice segment of the current picture (`book` made room for it).
    fn slice(&mut self, sh: SliceHeader, rbsp: Vec<u8>, held: Held<'a>) -> Result<()> {
        let idx = match self.book(&sh) {
            Ok(idx) => idx,
            Err(e) => return self.fail(e),
        };
        if let Some(cur) = self.cur.as_mut() {
            cur.slices.push(Pending {
                sh,
                idx,
                rbsp,
                _held: held,
            });
        }
        Ok(())
    }

    /// Completes the current picture: queues its job (`more`: another picture follows, so the
    /// workers may start), then C.5.2.3.
    fn finish_picture(&mut self, more: bool) -> Result<()> {
        let Some(mut cur) = self.cur.take() else {
            return Ok(());
        };
        let ordering = cur
            .act
            .sps
            .ordering
            .last()
            .copied()
            .ok_or(Error::InvalidData("sub-layer ordering info"))?;
        let (poc, output) = (cur.poc, cur.output);
        if output {
            self.dpb.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        }
        // The picture travels boxed (to a worker, back on the results channel, in `spent`): the
        // channel blocks and map nodes hold a pointer, not the picture. The box is booked here,
        // in a serial run too, and released when the picture drops.
        let n = size_of::<Cur<'_>>() + SPENT_ENTRY;
        self.ledger.reserve(n)?;
        cur.booked.n += n;
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.submit(
            Job {
                ticket,
                cur: Box::new(cur),
            },
            more,
        )?;
        if output {
            // C.5.2.3: latency counts of the pictures that follow the current one in output order.
            for w in &mut self.dpb {
                if w.poc > poc {
                    w.latency = w.latency.saturating_add(1);
                }
            }
            self.dpb.push(Waiting {
                ticket,
                poc,
                latency: 0,
            });
        }
        while must_bump(&self.dpb, &ordering, false) {
            self.bump()?;
        }
        self.check()
    }

    /// Runs a job on a worker, or on the calling thread when no worker is free (or none runs).
    /// A job that finds every worker busy starts one more, up to the cap.
    fn submit(&mut self, job: Job<'a>, more: bool) -> Result<()> {
        if self.pool.tx.is_none() && more {
            self.init_pool(&job);
        }
        self.collect(false);
        if self.pool.workers < self.pool.cap && self.pool.inflight >= self.pool.workers {
            self.add_worker();
        }
        let idle = self.pool.inflight < self.pool.workers;
        let sent = match self.pool.tx.as_ref() {
            None => Err(job),
            // A worker is free or about to be: wait for it rather than run the job here.
            Some(tx) if idle => tx.send(job).map_err(|e| e.0),
            Some(tx) => tx.try_send(job).map_err(|e| match e {
                TrySendError::Full(j) | TrySendError::Disconnected(j) => j,
            }),
        };
        match sent {
            Ok(()) => {
                self.pool.inflight += 1;
                Ok(())
            }
            Err(job) => self.run_inline(job),
        }
    }

    /// Makes the channels and the worker cap: `max_threads - 1`, and no more than the memory
    /// left holds frames for (two per thread). No thread starts here. Does nothing when the
    /// cap is zero.
    fn init_pool(&mut self, job: &Job<'a>) {
        if self.pool.spawn.is_none() {
            return;
        }
        let per = job
            .cur
            .frame
            .charged
            .saturating_add(job.cur.act.charged)
            .saturating_mul(2)
            .max(1);
        let threads = self.pool.threads.min(1 + self.ledger.left() / per);
        if threads < 2 {
            return;
        }
        let (tx, rx) = sync_channel::<Job<'a>>(0);
        let (done_tx, done_rx) = channel::<Finished<'a>>();
        self.pool.cap = threads - 1;
        self.pool.tx = Some(tx);
        self.pool.rx = Some(Arc::new(Mutex::new(rx)));
        self.pool.done_tx = Some(done_tx);
        self.pool.results = Some(done_rx);
    }

    /// Starts one worker. When the thread cannot be made, the cap drops to the workers there are.
    fn add_worker(&mut self) {
        let (Some(spawn), Some(rx), Some(done_tx)) = (
            self.pool.spawn,
            self.pool.rx.as_ref(),
            self.pool.done_tx.as_ref(),
        ) else {
            return;
        };
        let (rx, done_tx) = (Arc::clone(rx), done_tx.clone());
        let (opts, cancel) = (self.opts, self.cancel);
        if spawn(Box::new(move || worker(&rx, &done_tx, opts, cancel))) {
            self.pool.workers += 1;
        } else {
            self.pool.cap = self.pool.workers;
        }
    }

    /// Decodes a picture on the calling thread. An error is returned as it is; when workers
    /// ran, the front end was ahead of the job, so the call is replayed with one thread.
    fn run_inline(&mut self, job: Job<'a>) -> Result<()> {
        let Job { ticket, mut cur } = job;
        // The calling thread and its bands are the threads the picture workers leave free.
        let free = self.pool.threads.saturating_sub(self.pool.workers).max(1);
        let workers = filter_workers(free, &cur.act);
        let r = decode_picture(&mut cur, self.opts, self.cancel, workers);
        self.retire(ticket, cur);
        match r {
            Ok(res) => {
                self.keep(ticket, res);
                Ok(())
            }
            Err(e) => {
                if self.pool.workers > 0 && e != Error::Cancelled {
                    self.replay = true;
                }
                Err(e)
            }
        }
    }

    /// Stores the result of a job that did not fail.
    fn keep(&mut self, ticket: usize, res: Option<(Picture, usize)>) {
        if let Some(r) = res {
            self.done.insert(ticket, r);
        }
    }

    /// Takes the results that workers have sent; with `block`, waits for at least one when a job
    /// is in flight.
    fn collect(&mut self, mut block: bool) {
        while self.pool.inflight > 0 {
            let got = match self.pool.results.as_ref() {
                None => None,
                Some(rx) if block => rx.recv().ok(),
                Some(rx) => rx.try_recv().ok(),
            };
            let Some((ticket, r, cur)) = got else {
                if block {
                    // Every worker is gone: nothing more will come.
                    self.pool.inflight = 0;
                    self.replay = true;
                }
                return;
            };
            block = false;
            self.pool.inflight -= 1;
            self.retire(ticket, cur);
            match r {
                Ok(res) => self.keep(ticket, res),
                Err(Error::Cancelled) => self.cancelled = true,
                Err(_) => self.replay = true,
            }
        }
    }

    /// An error of a worker job: `Cancelled`, or a placeholder when the call is to be replayed.
    fn check(&self) -> Result<()> {
        if self.replay {
            Err(Error::InvalidData("worker failed"))
        } else if self.cancelled {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Moves the results of the bumped pictures to `out`, in output order, while the next one is
    /// there; with `wait`, until the jobs in flight are done.
    fn commit(&mut self, wait: bool) -> Result<()> {
        loop {
            self.collect(false);
            if self.replay {
                return Ok(());
            }
            let Some(&t) = self.order.front() else {
                return Ok(());
            };
            if !self.done.contains_key(&t) {
                if wait && self.pool.inflight > 0 {
                    self.collect(true);
                    continue;
                }
                return Ok(());
            }
            self.out.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
            self.order.pop_front();
            if let Some((pic, _)) = self.done.remove(&t) {
                self.out.push(pic);
            }
        }
    }

    /// Waits for every job and outputs every bumped picture.
    fn finish_jobs(&mut self) -> Result<()> {
        self.commit(true)?;
        self.check()
    }

    /// C.5.2.4: outputs the waiting picture with the smallest PicOrderCntVal.
    fn bump(&mut self) -> Result<()> {
        let Some(i) = self
            .dpb
            .iter()
            .enumerate()
            .min_by_key(|(_, w)| w.poc)
            .map(|(i, _)| i)
        else {
            return Ok(());
        };
        self.order.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        let w = self.dpb.remove(i);
        self.order.push_back(w.ticket);
        self.commit(false)
    }

    /// Outputs every waiting picture in POC order.
    fn flush(&mut self) -> Result<()> {
        while !self.dpb.is_empty() {
            self.bump()?;
        }
        Ok(())
    }

    /// Empties the DPB without output (NoOutputOfPriorPicsFlag = 1). The jobs in flight finish
    /// first: in a serial run the earlier pictures are done (and the results still booked) when
    /// the discard frees them.
    fn discard_all(&mut self) {
        while self.pool.inflight > 0 {
            self.collect(true);
        }
        for w in std::mem::take(&mut self.dpb) {
            if let Some((_, charged)) = self.done.remove(&w.ticket) {
                self.ledger.release(charged);
            }
        }
    }

    /// After the error `e` of the stream: the pictures decoded so far are output. The jobs come
    /// first (they precede the partial picture in decode order); then the slice segments of the
    /// partial picture are decoded, and an error there replaces `e`, as eager decoding would
    /// have raised it before `e`. Returns that error.
    fn recover(&mut self, e: &Error) -> Option<Error> {
        while self.pool.inflight > 0 {
            self.collect(true);
        }
        if self.replay {
            return None;
        }
        let mut over = None;
        if let Some(mut cur) = self.cur.take()
            && *e != Error::Cancelled
        {
            over = decode_slices(&mut cur, self.opts, self.cancel).err();
        }
        // Out of memory while flushing leaves the rest of the DPB behind.
        let _ = self.flush();
        let _ = self.commit(true);
        over
    }
}

/// Reads the stream and, on an error, outputs the pictures that completed. Returns the result
/// and whether the call must run again with one thread: a worker job failed or the memory ran
/// out while workers existed, so the serial run decides the result.
fn drive<'n>(
    d: &mut Decoder<'_, '_>,
    nals: impl Iterator<Item = Result<Nal<'n>>>,
) -> (Result<()>, bool) {
    let mut r = d.stream(nals);
    if let Err(e) = r.clone()
        && !d.replay
        && let Some(e2) = d.recover(&e)
    {
        r = Err(e2);
    }
    let replay = match &r {
        Ok(()) | Err(Error::Cancelled) => false,
        Err(Error::AllocLimit | Error::OutOfMemory) => d.pool.workers > 0,
        Err(_) => d.replay,
    };
    (r, replay)
}

/// One run of the NAL loop with a new ledger. Threads are used when `max_threads` is above 1.
fn attempt_with<'a, 'n, I: Iterator<Item = Result<Nal<'n>>>>(
    ledger: &'a Ledger,
    make: &impl Fn() -> I,
    opts: &'a Options,
    limits: &'a Limits,
    cancel: &'a AtomicBool,
    out: &'a mut Vec<Picture>,
) -> (Result<()>, bool) {
    if limits.max_threads <= 1 {
        let mut d = Decoder::new(ledger, opts, limits, cancel, out, None);
        return drive(&mut d, make());
    }
    // `move`: the references keep their full lifetime `'a`, which the boxed worker closures need.
    std::thread::scope(move |s| {
        let handles = std::cell::RefCell::new(Vec::new());
        let spawn = |f: Box<dyn FnOnce() + Send + 'a>| match std::thread::Builder::new()
            .spawn_scoped(s, f)
        {
            Ok(h) => {
                handles.borrow_mut().push(h);
                true
            }
            Err(_) => false,
        };
        let r = {
            let mut d = Decoder::new(ledger, opts, limits, cancel, out, Some(&spawn));
            // `d` drops at the end of this block: its sender closes and the workers return.
            drive(&mut d, make())
        };
        // Join every worker by its handle: the scope alone waits for the closures, not for the
        // OS threads to exit, and a following call must not see them (`max_threads` counts
        // live threads). A worker's panic is already an error result.
        for h in handles.take() {
            let _ = h.join();
        }
        r
    })
}

fn attempt<'n, I: Iterator<Item = Result<Nal<'n>>>>(
    make: &impl Fn() -> I,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> (Result<()>, bool) {
    let ledger = Ledger::default();
    if let Err(e) = ledger.reset(limits.max_alloc, 0) {
        return (Err(e), false);
    }
    attempt_with(&ledger, make, opts, limits, cancel, out)
}

/// The NAL loop shared by Annex B and length-prefixed input; `make` gives the NAL units again for
/// a replay. On an error the pictures already complete in the DPB are output into `out` (the
/// picture being decoded is dropped). The result and `out` do not depend on `max_threads`: when
/// a worker fails, or memory runs out while workers exist, the call runs again with one thread
/// (`out` is cut back to its length at entry first).
fn run<'n, I: Iterator<Item = Result<Nal<'n>>>>(
    make: impl Fn() -> I,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> Result<()> {
    let start = out.len();
    let (r, replay) = attempt(&make, opts, limits, cancel, out);
    if !replay {
        return r;
    }
    out.truncate(start);
    let serial = Limits {
        max_threads: 1,
        ..*limits
    };
    attempt(&make, opts, &serial, cancel, out).0
}

/// Decodes an Annex B byte stream like [`decode`]; the pictures output before an error stay in
/// `out`.
///
/// Limits enforced: `max_alloc`, `max_width`, `max_height`, `max_pixels`, `max_pictures`;
/// `max_metadata` drops SEI messages without an error. The cancel flag is read before every NAL
/// unit and every CTU.
///
/// # Errors
///
/// [`Error::InvalidData`], [`Error::Unsupported`], [`Error::InterSlice`] (the pictures before
/// the P or B slice are in `out`), [`Error::NoPicture`], [`Error::WidthLimit`],
/// [`Error::HeightLimit`], [`Error::PixelLimit`], [`Error::PictureLimit`],
/// [`Error::AllocLimit`], [`Error::OutOfMemory`] and [`Error::Cancelled`].
pub fn decode_into(
    stream: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> Result<()> {
    run(|| nal::annexb(stream), opts, limits, cancel, out)
}

/// Decodes an Annex B byte stream: every intra picture, in output order, cropped to the
/// conformance window.
///
/// Limits enforced: `max_alloc`, `max_width`, `max_height`, `max_pixels`, `max_pictures`;
/// `max_metadata` drops SEI messages without an error. The cancel flag is read before every NAL
/// unit and every CTU.
///
/// # Errors
///
/// [`Error::InvalidData`], [`Error::Unsupported`], [`Error::InterSlice`] (use [`decode_into`] to
/// keep the earlier pictures), [`Error::NoPicture`], [`Error::WidthLimit`],
/// [`Error::HeightLimit`], [`Error::PixelLimit`], [`Error::PictureLimit`],
/// [`Error::AllocLimit`], [`Error::OutOfMemory`] and [`Error::Cancelled`].
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

/// Decodes HEIF/MP4 samples like [`decode_hvcc`]; the pictures output before an error stay in
/// `out`.
///
/// `config` is the `hvcC` box payload, `samples` the length-prefixed NAL units. Limits and
/// errors are those of [`decode_into`]; a bad `hvcC` record is [`Error::InvalidData`].
///
/// # Errors
///
/// [`Error::InvalidData`], [`Error::Unsupported`], [`Error::InterSlice`], [`Error::NoPicture`],
/// [`Error::WidthLimit`], [`Error::HeightLimit`], [`Error::PixelLimit`],
/// [`Error::PictureLimit`], [`Error::AllocLimit`], [`Error::OutOfMemory`] and
/// [`Error::Cancelled`].
pub fn decode_hvcc_into(
    config: &[u8],
    samples: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    out: &mut Vec<Picture>,
) -> Result<()> {
    let hv = nal::parse_hvcc(config)?;
    let nals = || {
        hv.nals
            .iter()
            .copied()
            .map(nal::header)
            .chain(nal::length_prefixed(samples, hv.length_size))
    };
    run(nals, opts, limits, cancel, out)
}

/// Decodes HEIF/MP4 samples: `config` is the `hvcC` box payload, `samples` the length-prefixed
/// NAL units. Every intra picture, in output order.
///
/// Limits enforced: `max_alloc`, `max_width`, `max_height`, `max_pixels`, `max_pictures`;
/// `max_metadata` drops SEI messages without an error. The cancel flag is read before every NAL
/// unit and every CTU.
///
/// # Errors
///
/// [`Error::InvalidData`], [`Error::Unsupported`], [`Error::InterSlice`], [`Error::NoPicture`],
/// [`Error::WidthLimit`], [`Error::HeightLimit`], [`Error::PixelLimit`],
/// [`Error::PictureLimit`], [`Error::AllocLimit`], [`Error::OutOfMemory`] and
/// [`Error::Cancelled`].
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
    let mut sps_held = [0usize; 16];
    let mut pps_sps: [Option<u32>; 64] = [None; 64];
    let mut first: Option<u32> = None;
    for nal in nals {
        let nal = nal?;
        if nal.layer_id != 0 {
            continue;
        }
        match nal.kind {
            33 => {
                let (_held, r) = ps_rbsp(nal.payload, &ledger)?;
                let sps = ps::parse_sps(&r)?;
                first.get_or_insert(sps.sps_id);
                let id = sps.sps_id as usize;
                if let (Some(slot), Some(held)) = (sps_list.get_mut(id), sps_held.get_mut(id)) {
                    // Booked before the old SPS is dropped, as in `replace_set`.
                    let cost = sps_cost(&sps);
                    ledger.reserve(cost)?;
                    *slot = Some(sps);
                    ledger.release(std::mem::replace(held, cost));
                }
            }
            34 => {
                let (_held, r) = ps_rbsp(nal.payload, &ledger)?;
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

/// Reads the first SPS of an Annex B byte stream and returns its [`Info`]; nothing is decoded.
///
/// Limits enforced: `max_alloc`, `max_width`, `max_height`, `max_pixels` (on the SPS size).
///
/// # Errors
///
/// [`Error::InvalidData`], [`Error::Unsupported`], [`Error::NoPicture`] (no SPS),
/// [`Error::WidthLimit`], [`Error::HeightLimit`], [`Error::PixelLimit`], [`Error::AllocLimit`]
/// and [`Error::OutOfMemory`].
pub fn probe(stream: &[u8], limits: &Limits) -> Result<Info> {
    probe_nals(nal::annexb(stream), limits)
}

/// Like [`probe`] for the parameter sets of an `hvcC` record.
///
/// Limits enforced and errors: those of [`probe`].
///
/// # Errors
///
/// [`Error::InvalidData`], [`Error::Unsupported`], [`Error::NoPicture`], [`Error::WidthLimit`],
/// [`Error::HeightLimit`], [`Error::PixelLimit`], [`Error::AllocLimit`] and
/// [`Error::OutOfMemory`].
pub fn probe_hvcc(config: &[u8], limits: &Limits) -> Result<Info> {
    let hv = nal::parse_hvcc(config)?;
    probe_nals(hv.nals.iter().copied().map(nal::header), limits)
}

/// Test tooling for the parity oracle: the Annex B bytes of every layer 0 NAL unit before the
/// first picture that holds a P or B slice. Nothing is decoded.
///
/// Limits enforced: `max_alloc` (the output is charged up front).
///
/// # Errors
///
/// [`Error::NoPicture`] when no intra picture is left, [`Error::InvalidData`],
/// [`Error::AllocLimit`] and [`Error::OutOfMemory`].
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
            32..=34 => ledger.release(store.param_set(&nal, &ledger)?),
            k if is_vcl(k) => {
                // first_slice_segment_in_pic_flag is the first bit of the slice header.
                if nal.payload.first().is_some_and(|b| b & 0x80 != 0) {
                    pic_start = out.len();
                    pictures += 1;
                }
                let (_held, r) = rbsp(nal.payload, &ledger)?;
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
        ledger.release(cur.append(&mut pending, max));
        let kept: Vec<_> = cur.msgs.iter().map(|m| m.payload_type).collect();
        assert_eq!(kept, [137, 130]);
        assert_eq!(cur.charged, max);
        assert!(pending.msgs.is_empty());
        assert_eq!(pending.charged, 0);
        // The dropped message's charge is released.
        assert_eq!(before - ledger.left(), max);
    }

    #[test]
    fn stored_parameter_sets_are_charged_and_released_on_replace() {
        use crate::ps::tests::{pps_bits, sps_bits};
        let sps_payload = sps_bits(64, 64);
        let pps_payload = pps_bits(1);
        fn nal(kind: u8, payload: &[u8]) -> Nal<'_> {
            Nal {
                kind,
                layer_id: 0,
                temporal_id: 0,
                payload,
            }
        }
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        let start = ledger.left();
        let mut store = Store::new();
        store.param_set(&nal(33, &sps_payload), &ledger).unwrap();
        let one = start - ledger.left();
        assert!(one >= size_of::<Sps>());
        store.param_set(&nal(34, &pps_payload), &ledger).unwrap();
        let both = start - ledger.left();
        assert!(both > one);
        // The same ids again: the old charge is released, the total does not grow.
        let old = store.param_set(&nal(33, &sps_payload), &ledger).unwrap();
        ledger.release(old);
        let old = store.param_set(&nal(34, &pps_payload), &ledger).unwrap();
        ledger.release(old);
        assert_eq!(start - ledger.left(), both);
        // A budget that cannot hold the set: AllocLimit, nothing stored.
        let tiny = Ledger::default();
        tiny.reset(crate::ledger::SMALL + 100, 0).unwrap();
        let mut store = Store::new();
        assert_eq!(
            store.param_set(&nal(33, &sps_payload), &tiny).unwrap_err(),
            Error::AllocLimit
        );
        assert!(store.sps[0].is_none());
    }

    fn picture(poc: i32) -> Picture {
        Picture {
            info: Info {
                width: 1,
                height: 1,
                coded_width: 1,
                coded_height: 1,
                chroma_format: ChromaFormat::Mono,
                bit_depth_luma: 8,
                bit_depth_chroma: 8,
                profile_idc: 1,
                level_idc: 0,
                vui: None,
            },
            planes: Vec::new(),
            poc,
            hash_ok: None,
            sei: Vec::new(),
        }
    }

    #[test]
    fn results_commit_in_output_order_whatever_the_arrival_order() {
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        let (opts, limits, cancel) = (
            Options::default(),
            Limits::default(),
            AtomicBool::new(false),
        );
        let mut out = Vec::new();
        let mut d = Decoder::new(&ledger, &opts, &limits, &cancel, &mut out, None);
        // Decode order 0, 1, 2 with POC 2, 0, 1: the output order is tickets 1, 2, 0.
        for (ticket, poc) in [(0usize, 2), (1, 0), (2, 1)] {
            d.dpb.push(Waiting {
                ticket,
                poc,
                latency: 0,
            });
        }
        d.keep(2, Some((picture(1), 0)));
        d.flush().unwrap();
        // The head (ticket 1) has not arrived: nothing is output, nothing is skipped.
        assert!(d.out.is_empty());
        d.keep(1, Some((picture(0), 0)));
        d.commit(false).unwrap();
        assert_eq!(d.out.iter().map(|p| p.poc).collect::<Vec<_>>(), [0, 1]);
        d.keep(0, Some((picture(2), 0)));
        d.finish_jobs().unwrap();
        assert_eq!(d.out.iter().map(|p| p.poc).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(d.done.is_empty() && d.order.is_empty() && d.dpb.is_empty());
    }

    #[test]
    fn discarded_pictures_release_their_bytes() {
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        let before = ledger.left();
        let (opts, limits, cancel) = (
            Options::default(),
            Limits::default(),
            AtomicBool::new(false),
        );
        let mut out = Vec::new();
        let mut d = Decoder::new(&ledger, &opts, &limits, &cancel, &mut out, None);
        for ticket in [0usize, 1] {
            d.dpb.push(Waiting {
                ticket,
                poc: 0,
                latency: 0,
            });
        }
        ledger.reserve(150).unwrap();
        d.keep(0, Some((picture(0), 100)));
        d.keep(1, Some((picture(0), 50)));
        d.discard_all();
        assert_eq!(ledger.left(), before);
        assert!(d.done.is_empty() && d.dpb.is_empty() && d.out.is_empty());
    }

    #[test]
    fn front_end_releases_wait_for_the_jobs_submitted_before_them() {
        let ledger = Ledger::default();
        ledger.reset(1 << 20, 0).unwrap();
        ledger.reserve(30).unwrap();
        let before = ledger.left();
        let (opts, limits, cancel) = (
            Options::default(),
            Limits::default(),
            AtomicBool::new(false),
        );
        let mut out = Vec::new();
        let mut d = Decoder::new(&ledger, &opts, &limits, &cancel, &mut out, None);
        // Pictures 0 and 1 are submitted and not dropped yet.
        d.next_ticket = 2;
        d.release(10);
        d.release(20);
        assert_eq!(d.late.len(), 1);
        assert_eq!(ledger.left(), before);
        d.retired = 1;
        d.apply_late();
        assert_eq!(ledger.left(), before);
        d.retired = 2;
        d.apply_late();
        assert_eq!(ledger.left(), before + 30);
        assert!(d.late.is_empty());
        // Nothing outstanding: released at once.
        ledger.reserve(5).unwrap();
        d.release(5);
        assert_eq!(ledger.left(), before + 30);
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

    #[test]
    fn filter_workers_gates() {
        use crate::ps::tests::{ledger, pps_bits, sps_bits};
        use crate::ps::{activate, parse_pps, parse_sps};
        let workers = |w: u32, h: u32, threads: usize| {
            let sps = Arc::new(parse_sps(&sps_bits(w, h)).unwrap());
            let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
            let act = activate(sps, pps, &Limits::default(), &ledger()).unwrap();
            filter_workers(threads, &act)
        };
        assert_eq!(workers(512, 256, 0), 1);
        assert_eq!(workers(512, 256, 1), 1);
        assert_eq!(workers(256, 256, 8), 1);
        assert_eq!(workers(512, 256, 8), 2);
        assert_eq!(workers(512, 256, usize::MAX), 2);
        // Size would allow 4 bands, but there are only 2 CTB rows.
        assert_eq!(workers(4096, 64, 8), 2);
        assert_eq!(workers(1024, 1024, 3), 3);
        assert_eq!(workers(4096, 4096, usize::MAX), 64);
    }
}
