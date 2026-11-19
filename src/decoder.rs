//! The NAL loop. Phase 1: parameter sets, SEI and slice headers are parsed and the picture
//! limits are checked, but slice data is not decoded yet.

use crate::error::{Error, Result};
use crate::ledger::{Held, Ledger};
use crate::limits::Limits;
use crate::nal::{self, Nal};
use crate::ps::{self, Pps, Sps};
use crate::sei;
use crate::slice::{self, SliceHeader};
use crate::{ChromaFormat, Info, Options, Picture};
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
    })
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

    /// Prefix or suffix SEI NAL unit (7.3.2.4). A malformed SEI is ignored, so this cannot fail
    /// except on the ledger. Phase 1 walks the messages and drops the picture hash.
    fn sei(&mut self, nal: &Nal<'_>, ledger: &Ledger) -> Result<()> {
        let (r, _held) = rbsp(nal.payload, ledger)?;
        let _ = sei::parse(&r);
        Ok(())
    }

    /// Slice segment header (7.3.6.1) of a VCL NAL unit. Returns
    /// `(first_slice_segment_in_pic_flag, slice_pic_parameter_set_id)`. A P or B slice is
    /// `Error::InterSlice`.
    fn slice_header(&mut self, nal: &Nal<'_>, ledger: &Ledger) -> Result<(bool, u32)> {
        let (r, _held) = rbsp(nal.payload, ledger)?;
        let sh = slice::parse(&r, nal.kind, &self.sps, &self.pps, self.prev.as_ref())?;
        let out = (sh.first_slice_segment_in_pic, sh.pps_id);
        if !sh.dependent {
            self.prev = Some(sh);
        }
        Ok(out)
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

/// The NAL loop shared by Annex B and length-prefixed input.
fn run<'a>(
    nals: impl Iterator<Item = Result<Nal<'a>>>,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<()> {
    let ledger = Ledger::default();
    ledger.reset(limits.max_alloc, 0)?;
    let mut store = Store::new();
    let mut pictures = 0usize;
    for nal in nals {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let nal = nal?;
        if nal.layer_id > 0 {
            continue;
        }
        match nal.kind {
            32..=34 => store.param_set(&nal, &ledger)?,
            39 | 40 => store.sei(&nal, &ledger)?,
            k if is_vcl(k) => {
                let (first, pps_id) = store.slice_header(&nal, &ledger)?;
                if first {
                    let _active = store.activate(pps_id, limits, &mut pictures, &ledger)?;
                }
                // ponytail: phase 2 decodes the slice data here.
                return Err(Error::Unsupported(
                    "slice data decoding is not implemented yet",
                ));
            }
            _ => {}
        }
    }
    Err(Error::NoPicture)
}

/// See [`crate::decode_into`].
pub fn decode_into(
    stream: &[u8],
    _opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    _out: &mut Vec<Picture>,
) -> Result<()> {
    run(nal::annexb(stream), limits, cancel)
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
    _opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
    _out: &mut Vec<Picture>,
) -> Result<()> {
    let hv = nal::parse_hvcc(config)?;
    let nals = hv
        .nals
        .iter()
        .copied()
        .map(nal::header)
        .chain(nal::length_prefixed(samples, hv.length_size));
    run(nals, limits, cancel)
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

/// Info of the first SPS of layer 0 in `nals`; nothing is activated.
fn probe_nals<'a>(nals: impl Iterator<Item = Result<Nal<'a>>>, limits: &Limits) -> Result<Info> {
    let ledger = Ledger::default();
    ledger.reset(limits.max_alloc, 0)?;
    for nal in nals {
        let nal = nal?;
        if nal.layer_id == 0 && nal.kind == 33 {
            let (r, _held) = ps_rbsp(nal.payload, &ledger)?;
            let sps = ps::parse_sps(&r)?;
            sps.check_limits(limits)?;
            return info_of(&sps);
        }
    }
    Err(Error::InvalidData("no SPS in the stream"))
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
                match store.slice_header(&nal, &ledger) {
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
    fn zero_budget_is_alloc_limit() {
        let limits = Limits {
            max_alloc: 0,
            ..Limits::default()
        };
        assert_eq!(probe(&[], &limits).unwrap_err(), Error::AllocLimit);
        assert_eq!(intra_prefix(&[], &limits).unwrap_err(), Error::AllocLimit);
    }
}
