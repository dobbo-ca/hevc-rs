//! The one error enum.

use std::fmt;

/// Why a call failed. The limit variants are distinct so a caller can match on them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The bitstream breaks a syntax or semantic rule of H.265; the text says which.
    InvalidData(&'static str),
    /// Valid HEVC that this decoder does not handle (profile, chroma format, range extension
    /// tool); the text says which.
    Unsupported(&'static str),
    /// A P or B slice: this decoder handles intra pictures only. The `_into` entry points keep
    /// the pictures decoded before it.
    InterSlice,
    /// The stream holds no decodable picture.
    NoPicture,
    /// `pic_width_in_luma_samples` is over `Limits::max_width`.
    WidthLimit,
    /// `pic_height_in_luma_samples` is over `Limits::max_height`.
    HeightLimit,
    /// Coded width x height is over `Limits::max_pixels`.
    PixelLimit,
    /// More pictures than `Limits::max_pictures`.
    PictureLimit,
    /// The working memory would pass `Limits::max_alloc`.
    AllocLimit,
    /// The allocator refused a size that is within the limits.
    OutOfMemory,
    /// The cancel flag was set.
    Cancelled,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "hevc: invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "hevc: unsupported: {s}"),
            Self::InterSlice => f.write_str("hevc: P or B slice (only intra pictures are decoded)"),
            Self::NoPicture => f.write_str("hevc: no picture in the stream"),
            Self::WidthLimit => f.write_str("hevc: picture width over the limit"),
            Self::HeightLimit => f.write_str("hevc: picture height over the limit"),
            Self::PixelLimit => f.write_str("hevc: picture pixel count over the limit"),
            Self::PictureLimit => f.write_str("hevc: picture count over the limit"),
            Self::AllocLimit => f.write_str("hevc: memory limit exceeded"),
            Self::OutOfMemory => f.write_str("hevc: out of memory"),
            Self::Cancelled => f.write_str("hevc: cancelled"),
        }
    }
}

impl std::error::Error for Error {}

/// Result of every fallible internal function.
pub(crate) type Result<T> = std::result::Result<T, Error>;
