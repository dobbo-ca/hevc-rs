//! HEVC (ITU-T H.265) decoding for intra pictures only, bit-exact, under strict resource limits.
//!
//! The crate decodes Main, Main 10 and Main Still Picture streams (4:2:0 and 4:0:0, 8 to 12
//! bits) from an Annex B byte stream or from HEIF/MP4 samples plus an `hvcC` record. P and B
//! slices are rejected with [`Error::InterSlice`]. The build spec is `docs/design.md`.
//!
//! Status: intra decoding works. Every I picture is decoded (CABAC, intra prediction, residual,
//! deblocking, SAO), checked against the decoded picture hash SEI when present, and output in
//! order with the conformance window applied.
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

mod bits;
mod cabac;
mod ctu;
mod deblock;
mod decoder;
mod error;
mod frame;
mod intra;
mod ledger;
mod limits;
mod nal;
mod ps;
mod residual;
mod sao;
mod scaling;
mod sei;
mod slice;
mod transform;

pub use decoder::{
    decode, decode_hvcc, decode_hvcc_into, decode_into, intra_prefix, probe, probe_hvcc,
};
pub use error::Error;
pub use limits::Limits;

/// Decode switches. Default: both in-loop filters on (conforming output), metadata kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Run the deblocking filter (default true). false is for diagnosis only: the output then
    /// does not conform.
    pub deblocking: bool,
    /// Run SAO (default true). Same note.
    pub sao: bool,
    /// Keep the metadata (default true): the SEI messages of each picture ([`Picture::sei`]) and
    /// the VUI ([`Info::vui`]). false: `Picture::sei` is empty and `Info::vui` is `None` in the
    /// pictures (`probe` takes no options and always returns the VUI).
    pub metadata: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            deblocking: true,
            sao: true,
            metadata: true,
        }
    }
}

/// The VUI (E.2.1) fields about aspect ratio, colour and display, as coded. A field whose flag
/// is absent has its spec default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vui {
    /// `aspect_ratio_idc` (Table E.1; 0, unspecified, when absent; 255 is `EXTENDED_SAR`).
    pub aspect_ratio_idc: u8,
    /// `(sar_width, sar_height)`, only when `aspect_ratio_idc` is 255.
    pub aspect_ratio: Option<(u16, u16)>,
    /// `video_format` (5, unspecified, when absent).
    pub video_format: u8,
    /// `video_full_range_flag` (false when absent).
    pub video_full_range: bool,
    /// `colour_primaries` (2, unspecified, when absent).
    pub colour_primaries: u8,
    /// `transfer_characteristics` (2 when absent).
    pub transfer_characteristics: u8,
    /// `matrix_coeffs` (2 when absent).
    pub matrix_coeffs: u8,
    /// `chroma_sample_loc_type_top_field` (0 when absent; values above 255 are clamped).
    pub chroma_sample_loc_type_top: u8,
    /// `chroma_sample_loc_type_bottom_field` (0 when absent; values above 255 are clamped).
    pub chroma_sample_loc_type_bottom: u8,
    /// `def_disp_win_left/right/top/bottom_offset` as coded (in chroma sample units); None when
    /// `default_display_window_flag` is 0.
    pub default_display_window: Option<[u32; 4]>,
}

/// One SEI message of a picture's access unit, raw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiMessage {
    /// true for a suffix SEI NAL unit, false for a prefix one.
    pub suffix: bool,
    /// `payloadType` (for example 132 decoded picture hash, 137 mastering display colour
    /// volume, 144 content light level, 147 alternative transfer characteristics).
    pub payload_type: u32,
    /// The `sei_payload` bytes, emulation prevention bytes removed.
    pub payload: Vec<u8>,
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
    /// The VUI of the SPS; None when `vui_parameters_present_flag` is 0, or when the picture was
    /// decoded with [`Options::metadata`] false.
    pub vui: Option<Vui>,
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
    /// Every SEI message of the picture's access unit (prefix ones before its first VCL NAL
    /// unit, suffix ones after), in bitstream order, raw. Empty when [`Options::metadata`] is
    /// false. Messages past [`Limits::max_metadata`] are dropped.
    pub sei: Vec<SeiMessage>,
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
