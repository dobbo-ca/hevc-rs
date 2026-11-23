//! HEIC glue: the primary image of a HEIC file (and its alpha plane), decoded with the HEVC
//! decoder on top of the isobmff-rs item graph. See `docs/design.md`, "HEIC glue".

use crate::error::{Error, Result};
use crate::ledger::filled;
use crate::{ChromaFormat, Limits, Options, Picture, Plane, Samples, SeiMessage, decode_hvcc_into};
use isobmff::{Codec, CodecConfig, Heif, ImageKind, Limit};
use std::sync::atomic::{AtomicBool, Ordering};

/// Mirror axis of an `imir` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Axis 0: the top and bottom swap (a vertical flip).
    TopBottom,
    /// Axis 1: the left and right swap (a horizontal flip).
    LeftRight,
}

/// A clean aperture (`clap`) as coded: the raw fractions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clap {
    /// `cleanApertureWidthN`.
    pub width_n: u32,
    /// `cleanApertureWidthD`.
    pub width_d: u32,
    /// `cleanApertureHeightN`.
    pub height_n: u32,
    /// `cleanApertureHeightD`.
    pub height_d: u32,
    /// `horizOffN` (signed).
    pub horiz_off_n: i32,
    /// `horizOffD`.
    pub horiz_off_d: u32,
    /// `vertOffN` (signed).
    pub vert_off_n: i32,
    /// `vertOffD`.
    pub vert_off_d: u32,
}

/// A transformative property of the primary image, raw. Not applied to [`HeicImage::image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// `clap`: crop.
    Clap(Clap),
    /// `irot`: rotate anticlockwise by 0, 90, 180 or 270 degrees.
    Rotate(u16),
    /// `imir`: mirror.
    Mirror(Axis),
}

/// The primary image of a HEIC file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeicImage {
    /// The primary image: one coded `hvc1` item, or a grid of `hvc1` tiles assembled (tiles
    /// row-major, each at `(col * tile_w, row * tile_h)`, cropped to the grid output size). Not
    /// rotated, mirrored or `clap`-cropped: see [`HeicImage::transforms`]. `sei` and `info.vui`
    /// are those of the first tile (empty / `None` with [`Options::metadata`] false).
    pub image: Picture,
    /// The luma plane of the auxiliary alpha image (itself coded or a grid).
    pub alpha: Option<Plane>,
    /// `clap`, `irot` and `imir` of the primary image, in file order.
    pub transforms: Vec<Transform>,
}

fn iso_err(e: isobmff::Error) -> Error {
    match e {
        isobmff::Error::Cancelled => Error::Cancelled,
        isobmff::Error::OutOfMemory => Error::OutOfMemory,
        isobmff::Error::LimitExceeded(Limit::Alloc) => Error::AllocLimit,
        isobmff::Error::LimitExceeded(Limit::Pixels) => Error::PixelLimit,
        // Valid HEIF that isobmff-rs does not handle (e.g. the `mini` box): outside the scope.
        isobmff::Error::Unsupported(what) => Error::Unsupported(what),
        // No `meta` box: e.g. the low-overhead `mini` form of HEIF, which isobmff-rs does not read.
        isobmff::Error::NoMeta => Error::Unsupported("HEIF file without a meta box (mini form)"),
        e => Error::InvalidData(e.class()),
    }
}

fn transform(t: &isobmff::Transform) -> Transform {
    match *t {
        isobmff::Transform::Clap(c) => Transform::Clap(Clap {
            width_n: c.width_n,
            width_d: c.width_d,
            height_n: c.height_n,
            height_d: c.height_d,
            horiz_off_n: c.horiz_off_n,
            horiz_off_d: c.horiz_off_d,
            vert_off_n: c.vert_off_n,
            vert_off_d: c.vert_off_d,
        }),
        isobmff::Transform::Rotate(d) => Transform::Rotate(d),
        isobmff::Transform::Mirror(a) => Transform::Mirror(match a {
            isobmff::Axis::TopBottom => Axis::TopBottom,
            isobmff::Axis::LeftRight => Axis::LeftRight,
        }),
    }
}

/// Copies `n` rows of `w` samples of `s` (row stride `sw`) into `d` (row stride `dw`) at `(x, y)`.
fn copy_rows<T: Copy>(
    d: &mut [T],
    dw: usize,
    s: &[T],
    sw: usize,
    (x, y): (usize, usize),
    (w, n): (usize, usize),
) -> Result<()> {
    for r in 0..n {
        let di = (y + r) * dw + x;
        let si = r * sw;
        match (d.get_mut(di..di + w), s.get(si..si + w)) {
            (Some(a), Some(b)) => a.copy_from_slice(b),
            _ => return Err(Error::InvalidData("plane size")),
        }
    }
    Ok(())
}

/// Copies `src` into `dst` at `(x, y)`, clipped to `dst`.
fn blit(dst: &mut Plane, src: &Plane, x: usize, y: usize) -> Result<()> {
    let (dw, sw) = (dst.width as usize, src.width as usize);
    let w = sw.min(dw.saturating_sub(x));
    let n = (src.height as usize).min((dst.height as usize).saturating_sub(y));
    if w == 0 || n == 0 {
        return Ok(());
    }
    match (&mut dst.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => copy_rows(d, dw, s, sw, (x, y), (w, n)),
        (Samples::U16(d), Samples::U16(s)) => copy_rows(d, dw, s, sw, (x, y), (w, n)),
        _ => Err(Error::InvalidData("grid tiles differ in sample size")),
    }
}

/// Places the planes of one tile (luma, then chroma at half the coordinates) on the canvas
/// planes at grid cell `(col, row)`. Planes the canvas lacks (luma only) are skipped.
fn place(canvas: &mut [Plane], tile: &[Plane], col: u32, row: u32) -> Result<()> {
    let (tw, th) = tile
        .first()
        .map(|p| (p.width, p.height))
        .ok_or(Error::InvalidData("tile has no plane"))?;
    let x = col.checked_mul(tw).ok_or(Error::InvalidData("grid size"))? as usize;
    let y = row.checked_mul(th).ok_or(Error::InvalidData("grid size"))? as usize;
    for (i, (c, t)) in canvas.iter_mut().zip(tile).enumerate() {
        let (x, y) = if i == 0 { (x, y) } else { (x / 2, y / 2) };
        blit(c, t, x, y)?;
    }
    Ok(())
}

/// A zeroed plane of `w` x `h` samples of the same type as `like`. The caller has charged it.
fn blank(like: &Plane, w: u32, h: u32) -> Result<Plane> {
    let n = (w as usize)
        .checked_mul(h as usize)
        .ok_or(Error::AllocLimit)?;
    let samples = match like.samples {
        Samples::U8(_) => Samples::U8(filled(n, 0u8)?),
        Samples::U16(_) => Samples::U16(filled(n, 0u16)?),
    };
    Ok(Plane {
        width: w,
        height: h,
        samples,
    })
}

/// Size of plane `i` of a `w` x `h` canvas.
fn canvas_dims(i: usize, w: u32, h: u32) -> (u32, u32) {
    if i == 0 {
        (w, h)
    } else {
        (w.div_ceil(2), h.div_ceil(2))
    }
}

fn plane_bytes(p: &Plane) -> usize {
    match &p.samples {
        Samples::U8(v) => v.len(),
        Samples::U16(v) => v.len().saturating_mul(2),
    }
}

/// Bytes of the planes of `p` that stay booked: `keep` canvas planes of `w` x `h`.
fn canvas_bytes(tile: &[Plane], keep: usize, w: u32, h: u32) -> usize {
    let mut total = 0usize;
    for (i, p) in tile.iter().take(keep).enumerate() {
        let (cw, ch) = canvas_dims(i, w, h);
        let sample = if matches!(p.samples, Samples::U8(_)) {
            1
        } else {
            2
        };
        total = total.saturating_add(
            (cw as usize)
                .saturating_mul(ch as usize)
                .saturating_mul(sample),
        );
    }
    total
}

fn sei_bytes(sei: &[SeiMessage]) -> usize {
    sei.iter().fold(0usize, |t, m| {
        t.saturating_add(m.payload.len() + size_of::<SeiMessage>())
    })
}

fn check_size(w: u32, h: u32, limits: &Limits) -> Result<()> {
    if w > limits.max_width {
        Err(Error::WidthLimit)
    } else if h > limits.max_height {
        Err(Error::HeightLimit)
    } else if u64::from(w) * u64::from(h) > limits.max_pixels {
        Err(Error::PixelLimit)
    } else {
        Ok(())
    }
}

/// `Some(false)` if any tile failed its hash, `Some(true)` if all passed, else `None`.
fn merge_hash(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

struct Ctx<'a> {
    heif: &'a Heif,
    file: &'a [u8],
    opts: &'a Options,
    limits: &'a Limits,
    cancel: &'a AtomicBool,
    /// The limits of every isobmff call (a parse, then one `item_bytes` at a time).
    iso: &'a isobmff::Limits,
    /// What the hevc side and the output may hold together.
    rest: usize,
}

impl Ctx<'_> {
    fn cancelled(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// One coded item: exactly one picture. `held` bytes of `rest` are already taken.
    fn coded(&self, id: u32, hv: &isobmff::Hvcc, held: usize) -> Result<Picture> {
        self.cancelled()?;
        if self.heif.item(id).is_some_and(|i| i.protection_index != 0) {
            return Err(Error::Unsupported("protected item"));
        }
        let config = Heif::slice(self.file, hv.raw).ok_or(Error::InvalidData("hvcC"))?;
        let sample = self
            .heif
            .item_bytes(self.file, id, self.iso, self.cancel)
            .map_err(iso_err)?;
        let lim = Limits {
            max_alloc: self.rest.checked_sub(held).ok_or(Error::AllocLimit)?,
            ..*self.limits
        };
        let mut out = Vec::new();
        decode_hvcc_into(config, &sample, self.opts, &lim, self.cancel, &mut out)?;
        match (out.pop(), out.is_empty()) {
            (Some(p), true) => Ok(p),
            _ => Err(Error::InvalidData("item is not exactly one picture")),
        }
    }

    /// A coded or grid image item; `what` is the `Unsupported` text for any other kind.
    fn image(&self, id: u32, held: usize, luma_only: bool, what: &'static str) -> Result<Picture> {
        let img = self.heif.image(id).map_err(iso_err)?;
        match &img.kind {
            ImageKind::Coded {
                codec: Codec::Hevc,
                config: CodecConfig::Hevc(hv),
            } => self.coded(id, hv, held),
            ImageKind::Grid(g) => self.grid(g, held, luma_only),
            _ => Err(Error::Unsupported(what)),
        }
    }

    fn grid(&self, g: &isobmff::Grid, held: usize, luma_only: bool) -> Result<Picture> {
        let (cols, rows) = (g.columns, g.rows);
        if cols == 0 || rows == 0 || u64::from(cols) * u64::from(rows) != g.tiles.len() as u64 {
            return Err(Error::InvalidData("grid tile count"));
        }
        check_size(g.width, g.height, self.limits)?;
        let mut held = held;
        let mut out: Option<Picture> = None;
        let (mut tw, mut th) = (0, 0);
        for (i, &tid) in g.tiles.iter().enumerate() {
            self.cancelled()?;
            let t = self.heif.image(tid).map_err(iso_err)?;
            let ImageKind::Coded {
                codec: Codec::Hevc,
                config: CodecConfig::Hevc(hv),
            } = &t.kind
            else {
                return Err(Error::Unsupported("grid tile is not HEVC"));
            };
            let mut tile = self.coded(tid, hv, held)?;
            let (col, row) = ((i % cols as usize) as u32, (i / cols as usize) as u32);
            let (pw, ph) = tile
                .planes
                .first()
                .map(|p| (p.width, p.height))
                .ok_or(Error::InvalidData("tile has no plane"))?;
            // The tile pitch is the `ispe` size (as in libheif); a decoded size that differs from
            // it would place the tiles elsewhere.
            if (pw, ph) != (t.width, t.height) {
                return Err(Error::InvalidData("grid tile size differs from ispe"));
            }
            if out.is_none() {
                (tw, th) = (pw, ph);
                let covers = u64::from(cols) * u64::from(tw) >= u64::from(g.width)
                    && u64::from(rows) * u64::from(th) >= u64::from(g.height);
                if !covers {
                    return Err(Error::InvalidData("grid tiles do not cover the output"));
                }
                let odd =
                    (cols > 1 && !tw.is_multiple_of(2)) || (rows > 1 && !th.is_multiple_of(2));
                if tile.info.chroma_format == ChromaFormat::Yuv420 && odd {
                    return Err(Error::InvalidData("odd tile size in a 4:2:0 grid"));
                }
                let keep = if luma_only { 1 } else { tile.planes.len() };
                let canvas = canvas_bytes(&tile.planes, keep, g.width, g.height);
                let sei = sei_bytes(&tile.sei);
                // The first tile is still alive while the canvas is allocated.
                let peak = [
                    held,
                    tile.planes.iter().map(plane_bytes).sum::<usize>(),
                    sei,
                    canvas,
                ]
                .iter()
                .try_fold(0usize, |a, &b| a.checked_add(b));
                if peak.is_none_or(|p| p > self.rest) {
                    return Err(Error::AllocLimit);
                }
                let planes = tile
                    .planes
                    .iter()
                    .take(keep)
                    .enumerate()
                    .map(|(k, p)| {
                        let (w, h) = canvas_dims(k, g.width, g.height);
                        blank(p, w, h)
                    })
                    .collect::<Result<Vec<_>>>()?;
                held += canvas + sei;
                let mut info = tile.info.clone();
                (info.width, info.height) = (g.width, g.height);
                (info.coded_width, info.coded_height) = (g.width, g.height);
                out = Some(Picture {
                    info,
                    planes,
                    poc: tile.poc,
                    hash_ok: tile.hash_ok,
                    sei: std::mem::take(&mut tile.sei),
                });
            } else if let Some(o) = out.as_mut() {
                let same = tile.info.chroma_format == o.info.chroma_format
                    && tile.info.bit_depth_luma == o.info.bit_depth_luma
                    && tile.info.bit_depth_chroma == o.info.bit_depth_chroma
                    && (pw, ph) == (tw, th);
                if !same {
                    return Err(Error::InvalidData("grid tiles differ in size or format"));
                }
                o.hash_ok = merge_hash(o.hash_ok, tile.hash_ok);
            }
            if let Some(o) = out.as_mut() {
                place(&mut o.planes, &tile.planes, col, row)?;
            }
        }
        out.ok_or(Error::InvalidData("grid tile count"))
    }
}

/// Decodes the primary image of a HEIC file, and its alpha plane when it has one.
///
/// The primary item is one `hvc1` image or a `grid` of `hvc1` tiles; each coded item is decoded
/// with [`decode_hvcc_into`] and must hold exactly one picture. The metadata (SEI messages, VUI)
/// is that of the first tile, kept unless [`Options::metadata`] is false. The result is not
/// rotated, mirrored or cropped by the `irot`, `imir` and `clap` properties; they are returned
/// raw in [`HeicImage::transforms`]. The alpha image (`urn:mpeg:hevc:2015:auxid:1` or
/// `urn:mpeg:mpegB:cicp:systems:auxiliary:alpha`, or `urn:mpeg:avc:2015:auxid:1`) is decoded the
/// same way and only its luma plane is kept.
///
/// Limits: `max_alloc` is the peak of the whole call. A quarter of it goes to isobmff (an eighth
/// for the parse, which stays alive, and an eighth for the item bytes of one item at a time);
/// the decoded pictures, the assembled planes and the decoder's working memory share the rest.
/// `max_threads` is 1 for isobmff. `max_width`, `max_height` and `max_pixels` apply to every coded
/// picture and to the output size of a grid. The cancel flag is read before each coded item, so
/// between tiles, and inside the decoder.
///
/// # Errors
///
/// [`Error::Unsupported`] when the primary (or alpha) image is not HEVC or a grid of HEVC tiles,
/// or when a coded item is protected (`protection_index` not 0); [`Error::InvalidData`] for a
/// container error (the text is the isobmff error class, for example `"truncated"`), an item that
/// is not exactly one picture, or tiles that differ in size or format, do not cover the grid or
/// decode to another size than their `ispe`; the errors of [`decode_hvcc_into`] (including
/// [`Error::InterSlice`] and [`Error::NoPicture`]); [`Error::WidthLimit`],
/// [`Error::HeightLimit`], [`Error::PixelLimit`], [`Error::AllocLimit`],
/// [`Error::OutOfMemory`] and [`Error::Cancelled`].
pub fn decode_heic(
    file: &[u8],
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<HeicImage> {
    let iso = isobmff::Limits {
        max_alloc: limits.max_alloc / 8,
        max_threads: 1,
        max_pixels: limits.max_pixels,
        ..isobmff::Limits::default()
    };
    let heif = isobmff::parse(file, &iso, cancel).map_err(iso_err)?;
    let primary = heif.primary_image().map_err(iso_err)?;
    let cx = Ctx {
        heif: &heif,
        file,
        opts,
        limits,
        cancel,
        iso: &iso,
        rest: limits.max_alloc - limits.max_alloc / 4,
    };
    let image = cx.image(primary.id, 0, false, "primary image is not HEVC")?;
    let alpha = match primary.alpha {
        Some(id) => {
            let held = image
                .planes
                .iter()
                .map(plane_bytes)
                .sum::<usize>()
                .saturating_add(sei_bytes(&image.sei));
            let a = cx.image(id, held, true, "alpha image is not HEVC")?;
            a.planes.into_iter().next()
        }
        None => None,
    };
    Ok(HeicImage {
        image,
        alpha,
        transforms: primary.transforms.iter().map(transform).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: u32, h: u32, v: u8) -> Plane {
        Plane {
            width: w,
            height: h,
            samples: Samples::U8(vec![v; (w * h) as usize]),
        }
    }

    fn rows(p: &Plane) -> Vec<Vec<u8>> {
        let Samples::U8(v) = &p.samples else {
            panic!("u8")
        };
        v.chunks(p.width as usize).map(<[u8]>::to_vec).collect()
    }

    #[test]
    fn grid_placement_row_major_and_clipped() {
        // 2 x 2 tiles of 2 x 2 on a 3 x 3 output: the right column and bottom row are clipped.
        let mut canvas = [blank(&plane(1, 1, 0), 3, 3).unwrap()];
        for (i, v) in [1u8, 2, 3, 4].into_iter().enumerate() {
            let (col, row) = ((i % 2) as u32, (i / 2) as u32);
            place(&mut canvas, &[plane(2, 2, v)], col, row).unwrap();
        }
        assert_eq!(
            rows(&canvas[0]),
            [vec![1, 1, 2], vec![1, 1, 2], vec![3, 3, 4]]
        );
    }

    #[test]
    fn column_outside_the_output_is_skipped() {
        // Columns 0..3 of 2-wide tiles on a 5 x 2 output: column 3 starts at x = 6, past it.
        let mut canvas = [blank(&plane(1, 1, 0), 5, 2).unwrap()];
        for col in 0..4 {
            place(&mut canvas, &[plane(2, 2, 1 + col as u8)], col, 0).unwrap();
        }
        assert_eq!(rows(&canvas[0]), [vec![1, 1, 2, 2, 3], vec![1, 1, 2, 2, 3]]);
    }

    #[test]
    fn chroma_goes_to_half_coordinates() {
        let mut canvas = [
            blank(&plane(1, 1, 0), 4, 2).unwrap(),
            blank(&plane(1, 1, 0), 2, 1).unwrap(),
        ];
        for col in 0..2 {
            let t = [plane(2, 2, 9), plane(1, 1, 5 + col as u8)];
            place(&mut canvas, &t, col, 0).unwrap();
        }
        assert_eq!(rows(&canvas[1]), [vec![5, 6]]);
        // A luma-only canvas skips the chroma plane.
        let mut luma = [blank(&plane(1, 1, 0), 2, 2).unwrap()];
        place(&mut luma, &[plane(2, 2, 7), plane(1, 1, 1)], 0, 0).unwrap();
        assert_eq!(rows(&luma[0]), [vec![7, 7], vec![7, 7]]);
    }

    #[test]
    fn mixed_sample_sizes_are_invalid() {
        let mut canvas = [blank(&plane(1, 1, 0), 2, 2).unwrap()];
        let wide = Plane {
            width: 2,
            height: 2,
            samples: Samples::U16(vec![0; 4]),
        };
        assert!(matches!(
            place(&mut canvas, &[wide], 0, 0),
            Err(Error::InvalidData(_))
        ));
    }

    #[test]
    fn transforms_are_mapped_in_order() {
        let c = isobmff::Clap {
            width_n: 1,
            width_d: 2,
            height_n: 3,
            height_d: 4,
            horiz_off_n: -5,
            horiz_off_d: 6,
            vert_off_n: 7,
            vert_off_d: 8,
        };
        let got: Vec<_> = [
            isobmff::Transform::Clap(c),
            isobmff::Transform::Rotate(270),
            isobmff::Transform::Mirror(isobmff::Axis::LeftRight),
            isobmff::Transform::Mirror(isobmff::Axis::TopBottom),
        ]
        .iter()
        .map(transform)
        .collect();
        let want = [
            Transform::Clap(Clap {
                width_n: 1,
                width_d: 2,
                height_n: 3,
                height_d: 4,
                horiz_off_n: -5,
                horiz_off_d: 6,
                vert_off_n: 7,
                vert_off_d: 8,
            }),
            Transform::Rotate(270),
            Transform::Mirror(Axis::LeftRight),
            Transform::Mirror(Axis::TopBottom),
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn errors_and_hashes_map() {
        use isobmff::Error as E;
        assert_eq!(iso_err(E::Cancelled), Error::Cancelled);
        assert_eq!(iso_err(E::LimitExceeded(Limit::Alloc)), Error::AllocLimit);
        assert_eq!(iso_err(E::LimitExceeded(Limit::Pixels)), Error::PixelLimit);
        assert_eq!(iso_err(E::Truncated), Error::InvalidData("truncated"));
        assert_eq!(iso_err(E::Unsupported("mini")), Error::Unsupported("mini"));
        assert_eq!(merge_hash(Some(true), Some(true)), Some(true));
        assert_eq!(merge_hash(Some(true), None), None);
        assert_eq!(merge_hash(Some(true), Some(false)), Some(false));
    }

    #[test]
    fn garbage_is_invalid_not_a_panic() {
        let cancel = AtomicBool::new(false);
        let r = decode_heic(&[0; 16], &Options::default(), &Limits::default(), &cancel);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }
}
