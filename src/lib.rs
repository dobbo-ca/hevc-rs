//! HEVC (ITU-T H.265) decoding for intra pictures only, bit-exact, under strict resource limits.
//!
//! The crate decodes Main, Main 10 and Main Still Picture streams (4:2:0 and 4:0:0, 8 to 12
//! bits) from an Annex B byte stream or from HEIF/MP4 samples plus an `hvcC` record. P and B
//! slices are rejected with [`Error::InterSlice`]. The build spec is `docs/design.md`.
//!
//! Status: phase 1. Parameter sets, SEI and slice headers are parsed and the limits are
//! enforced; decoding slice data is not implemented yet and returns [`Error::Unsupported`].
//!
//! # Resource limits
//!
//! There is no unlimited entry point: every call takes a [`Limits`]. `max_alloc` is the enforced
//! peak working memory of the call, charged before each allocation. `max_threads` caps the
//! threads of the call (1 spawns none; v1 is serial anyway). `max_width`, `max_height` and
//! `max_pixels` cap the coded picture size and `max_pictures` the pictures per call. The decode
//! entry points also read a cancel flag before every NAL unit.
//!
//! ```
//! use hevc::{Error, Limits, Options};
//! use std::sync::atomic::AtomicBool;
//!
//! let limits = Limits {
//!     max_threads: 2,
//!     max_alloc: 64 << 20,
//!     ..Limits::default()
//! };
//! let cancel = AtomicBool::new(false);
//! // An empty stream has no picture.
//! let r = hevc::decode(&[], &Options::default(), &limits, &cancel);
//! assert_eq!(r.unwrap_err(), Error::NoPicture);
//! // A NAL header with the forbidden bit set is not a stream.
//! assert!(hevc::probe(&[0, 0, 1, 0xff, 0xff], &limits).is_err());
//! // A zero memory budget fails at once.
//! let none = Limits { max_alloc: 0, ..limits };
//! let r = hevc::decode(&[], &Options::default(), &none, &cancel);
//! assert_eq!(r.unwrap_err(), Error::AllocLimit);
//! ```

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing
    )
)]
// ponytail: phase 1 leaves parts of the parsing modules unused until slice decoding lands;
// remove this allow with phase 2.
#![allow(dead_code)]

mod bits;
mod decoder;
mod error;
mod ledger;
mod limits;
mod nal;
mod ps;
mod scaling;
mod sei;
mod slice;

pub use decoder::{
    decode, decode_hvcc, decode_hvcc_into, decode_into, intra_prefix, probe, probe_hvcc,
};
pub use error::Error;
pub use limits::Limits;

/// Decode switches. Default: both in-loop filters on (conforming output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Run the deblocking filter (default true). false is for diagnosis only: the output then
    /// does not conform.
    pub deblocking: bool,
    /// Run SAO (default true). Same note.
    pub sao: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            deblocking: true,
            sao: true,
        }
    }
}

/// Chroma format of the pictures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 4:0:0, one plane.
    Mono,
    /// 4:2:0, three planes.
    Yuv420,
}

/// Stream facts from the first SPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// Width after the conformance window.
    pub width: u32,
    /// Height after the conformance window.
    pub height: u32,
    /// `pic_width_in_luma_samples`.
    pub coded_width: u32,
    /// `pic_height_in_luma_samples`.
    pub coded_height: u32,
    /// Chroma format.
    pub chroma_format: ChromaFormat,
    /// Luma bit depth, 8 to 12.
    pub bit_depth_luma: u8,
    /// Chroma bit depth, 8 to 12.
    pub bit_depth_chroma: u8,
    /// `general_profile_idc`.
    pub profile_idc: u8,
    /// `general_level_idc`.
    pub level_idc: u8,
}

/// Samples of one plane, tightly packed rows (stride = width of the plane). `U8` when the
/// component's bit depth is 8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Samples {
    /// 8-bit samples.
    U8(Vec<u8>),
    /// Samples of 9 to 12 bits.
    U16(Vec<u16>),
}

/// One plane of a picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// The samples, `width * height` of them.
    pub samples: Samples,
}

/// One output picture, cropped to the conformance window, in output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    /// Stream facts of the SPS this picture used.
    pub info: Info,
    /// 1 plane (Mono) or 3 (Y, Cb, Cr).
    pub planes: Vec<Plane>,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Some(true/false) when a decoded picture hash SEI was present and checked (over the
    /// uncropped decoded picture); None when absent.
    pub hash_ok: Option<bool>,
}

// The entry points live in `decoder.rs` and are re-exported above:
//
// decode(stream, opts, limits, cancel) -> Result<Vec<Picture>, Error>
//     Annex B byte stream, every intra picture in output order.
// decode_into(stream, opts, limits, cancel, out) -> Result<(), Error>
//     Same; pictures output before an error stay in `out`.
// decode_hvcc(config, samples, opts, limits, cancel) -> Result<Vec<Picture>, Error>
// decode_hvcc_into(config, samples, opts, limits, cancel, out) -> Result<(), Error>
//     HEIF/MP4 form: `config` is the hvcC box payload, `samples` length-prefixed NAL units.
// probe(stream, limits) -> Result<Info, Error>
// probe_hvcc(config, limits) -> Result<Info, Error>
//     Info of the first SPS, nothing decoded.
// intra_prefix(stream, limits) -> Result<Vec<u8>, Error>
//     Test tooling for the parity oracle: the Annex B bytes of every layer 0 NAL unit before the
//     first picture that holds a P or B slice. `Error::NoPicture` when none is left.
