//! HEIC glue: the primary image of a HEIC file (and its alpha plane), decoded with the HEVC
//! decoder on top of the isobmff-rs item graph. See `docs/design.md`, "HEIC glue".

use crate::error::{Error, Result};
use crate::ledger::{SMALL, filled};
use crate::{
    ChromaFormat, Info, Limits, Options, Picture, Plane, Samples, SeiMessage, decode_hvcc_into,
};
use isobmff::{Codec, CodecConfig, Colour, Heif, ImageKind, Limit};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

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

/// A transformative property of the primary image, raw. Applied to [`HeicImage::image`] only with
/// [`Options::apply_transforms`].
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
    /// The primary image: one coded `hvc1` item, a grid of `hvc1` tiles assembled (tiles
    /// row-major, each at `(col * tile_w, row * tile_h)`, cropped to the grid output size), or an
    /// `iden`/`iovl` of those (see [`decode_heic`]). Not rotated, mirrored or `clap`-cropped
    /// unless [`HeicImage::transforms_applied`]: see [`HeicImage::transforms`]. `sei` and
    /// `info.vui` are those of the first tile (empty / `None` with [`Options::metadata`] false).
    pub image: Picture,
    /// The luma plane of the auxiliary alpha image (itself coded or a grid). With
    /// [`HeicImage::transforms_applied`], the alpha item's own transforms are applied to it.
    pub alpha: Option<Plane>,
    /// `clap`, `irot` and `imir` of the primary image, in file order. Always listed, also when
    /// they were applied.
    pub transforms: Vec<Transform>,
    /// True when [`Options::apply_transforms`] was set: `image` and `alpha` are already cropped,
    /// rotated and mirrored by their transforms, and `image.info.width`/`height` are the size
    /// after them. False: the pixels are as coded.
    pub transforms_applied: bool,
}

fn iso_err(e: isobmff::Error) -> Error {
    match e {
        isobmff::Error::Cancelled => Error::Cancelled,
        isobmff::Error::OutOfMemory => Error::OutOfMemory,
        isobmff::Error::LimitExceeded(Limit::Alloc) => Error::AllocLimit,
        isobmff::Error::LimitExceeded(Limit::Pixels) => Error::PixelLimit,
        // Valid HEIF that isobmff-rs does not handle (e.g. a `mini` box of another codec or
        // version): outside the scope.
        isobmff::Error::Unsupported(what) => Error::Unsupported(what),
        // `NoMeta` (neither `meta` nor `mini`) is a container error like the others.
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

/// Copies `n` rows of `w` samples of `s` (row stride `sw`, first sample at `s0`) into `d` (row
/// stride `dw`) at `(x, y)`.
fn copy_rows<T: Copy>(
    d: &mut [T],
    dw: usize,
    s: &[T],
    sw: usize,
    (x, y): (usize, usize),
    (w, n): (usize, usize),
    s0: usize,
) -> Result<()> {
    for r in 0..n {
        let di = (y + r) * dw + x;
        let si = s0 + r * sw;
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
        (Samples::U8(d), Samples::U8(s)) => copy_rows(d, dw, s, sw, (x, y), (w, n), 0),
        (Samples::U16(d), Samples::U16(s)) => copy_rows(d, dw, s, sw, (x, y), (w, n), 0),
        _ => Err(Error::InvalidData("grid tiles differ in sample size")),
    }
}

/// Copies `src` into `dst` with its top-left sample at `(x, y)` (either may be negative), clipped
/// to `dst`.
fn blit_at(dst: &mut Plane, src: &Plane, x: i64, y: i64) -> Result<()> {
    let (sx, sy) = ((-x).max(0), (-y).max(0));
    let (dx, dy) = (x.max(0), y.max(0));
    let w = (i64::from(src.width) - sx).min(i64::from(dst.width) - dx);
    let n = (i64::from(src.height) - sy).min(i64::from(dst.height) - dy);
    if w <= 0 || n <= 0 {
        return Ok(());
    }
    let u = |v: i64| usize::try_from(v).map_err(|_| Error::InvalidData("overlay offset"));
    let (sx, sy, dx, dy, w, n) = (u(sx)?, u(sy)?, u(dx)?, u(dy)?, u(w)?, u(n)?);
    let (dw, sw) = (dst.width as usize, src.width as usize);
    let s0 = sy * sw + sx;
    match (&mut dst.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => copy_rows(d, dw, s, sw, (dx, dy), (w, n), s0),
        (Samples::U16(d), Samples::U16(s)) => copy_rows(d, dw, s, sw, (dx, dy), (w, n), s0),
        _ => Err(Error::InvalidData("overlay inputs differ in sample size")),
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

/// Chroma format and bit depths: what the pieces of an assembled image must share.
type Fmt = (ChromaFormat, u8, u8);

fn fmt(i: &Info) -> Fmt {
    (i.chroma_format, i.bit_depth_luma, i.bit_depth_chroma)
}

/// The canvas of an assembled image of `w` x `h`: blank planes in the format of `first` (luma
/// only when `luma_only`), with its info (resized), POC, hash result and SEI messages (moved).
/// Returns it and the bytes it adds to `held`. `first` is alive meanwhile, so the peak counts both.
fn canvas(
    rest: usize,
    first: &mut Picture,
    w: u32,
    h: u32,
    luma_only: bool,
    held: usize,
) -> Result<(Picture, usize)> {
    let keep = if luma_only { 1 } else { first.planes.len() };
    let bytes = canvas_bytes(&first.planes, keep, w, h);
    let sei = sei_bytes(&first.sei);
    let peak = [
        held,
        first.planes.iter().map(plane_bytes).sum::<usize>(),
        sei,
        bytes,
    ]
    .iter()
    .try_fold(0usize, |a, &b| a.checked_add(b));
    if peak.is_none_or(|p| p > rest) {
        return Err(Error::AllocLimit);
    }
    let planes = first
        .planes
        .iter()
        .take(keep)
        .enumerate()
        .map(|(k, p)| {
            let (cw, ch) = canvas_dims(k, w, h);
            blank(p, cw, ch)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut info = first.info.clone();
    (info.width, info.height) = (w, h);
    (info.coded_width, info.coded_height) = (w, h);
    let out = Picture {
        info,
        planes,
        poc: first.poc,
        hash_ok: first.hash_ok,
        sei: std::mem::take(&mut first.sei),
    };
    Ok((out, bytes.saturating_add(sei)))
}

/// What the tile workers of one grid are told.
struct Fill<'a> {
    /// Tile columns.
    cols: usize,
    /// Tiles in all; tile 0 is already on the canvas.
    n: usize,
    /// Bytes the decodes of the other tiles may hold together.
    avail: usize,
    /// A guess at what one tile decode needs; it only bounds the workers (a worker that runs
    /// out of its share is redone alone with all of `avail`, so the result never depends on it).
    per: usize,
    /// `Limits::max_threads`, at least 1.
    threads: usize,
    cancel: &'a AtomicBool,
}

impl Fill<'_> {
    fn cancelled(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Grid cell of tile `i`.
    fn cell(&self, i: usize) -> (u32, u32) {
        let cols = self.cols.max(1);
        ((i % cols) as u32, (i / cols) as u32)
    }
}

/// The outcome of one tile of the parallel pass.
#[derive(Clone)]
enum Slot {
    /// Not done: skipped after an earlier error, or out of its share of the budget.
    Todo,
    /// Decoded and on the canvas; its hash result.
    Done(Option<bool>),
    Err(Error),
}

/// Decodes tiles `1..fill.n` with `dec(index, budget, workers)` and places them on `out`.
///
/// Up to `fill.threads` threads (the caller included) take tiles from a counter, each decode
/// with `avail / workers` bytes, and place the tile on the canvas themselves (the tiles do not
/// overlap, so the order does not matter). Then the results are read in tile order: the first
/// error wins, and a tile left `Todo` is decoded alone with the whole budget, so the output and
/// the error are those of the serial loop. One thread scope per call; no pool (see
/// `docs/threading.md`).
fn fill_tiles<F>(out: &mut Picture, fill: &Fill<'_>, dec: F) -> Result<()>
where
    F: Fn(usize, usize, usize) -> Result<Picture> + Sync,
{
    let n = fill.n;
    let slot_bytes = n.saturating_mul(size_of::<Slot>());
    let spare = fill.avail.saturating_sub(slot_bytes);
    let nw = fill
        .threads
        .min(n.saturating_sub(1))
        .min(spare / fill.per.max(1));
    let mut slots = Vec::new();
    if nw >= 2 {
        let budget = spare / nw;
        let next = AtomicUsize::new(1);
        let stop = AtomicUsize::new(usize::MAX);
        let shared = Mutex::new(filled(n, Slot::Todo)?);
        let cv = Mutex::new(out.planes.as_mut_slice());
        let work = || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n || i > stop.load(Ordering::Relaxed) {
                    break;
                }
                let (col, row) = fill.cell(i);
                let r = fill
                    .cancelled()
                    .and_then(|()| dec(i, budget, nw))
                    .and_then(|t| {
                        let mut g = cv.lock().unwrap_or_else(PoisonError::into_inner);
                        place(&mut g, &t.planes, col, row)?;
                        Ok(t.hash_ok)
                    });
                let s = match r {
                    Ok(h) => Slot::Done(h),
                    // Out of its share: left for the serial pass, which has all of it.
                    Err(Error::AllocLimit) => continue,
                    Err(e) => {
                        stop.fetch_min(i, Ordering::Relaxed);
                        Slot::Err(e)
                    }
                };
                if let Some(x) = shared
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get_mut(i)
                {
                    *x = s;
                }
            }
        };
        std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for _ in 1..nw {
                // A refused spawn only means fewer workers.
                match std::thread::Builder::new().spawn_scoped(sc, work) {
                    Ok(h) => handles.push(h),
                    Err(_) => break,
                }
            }
            work();
            for h in handles {
                let _ = h.join();
            }
        });
        slots = shared.into_inner().unwrap_or_else(PoisonError::into_inner);
    }
    for i in 1..n {
        match slots.get(i) {
            Some(Slot::Done(h)) => out.hash_ok = merge_hash(out.hash_ok, *h),
            Some(Slot::Err(e)) => return Err(e.clone()),
            Some(Slot::Todo) | None => {
                fill.cancelled()?;
                let t = dec(i, spare, 1)?;
                out.hash_ok = merge_hash(out.hash_ok, t.hash_ok);
                let (col, row) = fill.cell(i);
                place(&mut out.planes, &t.planes, col, row)?;
            }
        }
    }
    Ok(())
}

/// The Y, Cb and Cr sample values of the colour `rgb` (16 bits per channel, full range) for
/// `(matrix_coefficients, full_range)` and the luma and chroma bit depths. See `docs/design.md`.
fn fill_value(rgb: [u16; 3], (matrix, full): (u16, bool), bd_l: u8, bd_c: u8) -> [u32; 3] {
    let [r, g, b] = rgb.map(|v| f64::from(v) / 65535.0);
    let (kr, kb) = match matrix {
        1 => (0.2126, 0.0722),
        4 => (0.30, 0.11),
        7 => (0.212, 0.087),
        9 => (0.2627, 0.0593),
        _ => (0.299, 0.114),
    };
    // `e` is in 0..=1 for luma and in -0.5..=0.5 for the colour differences.
    let q = |e: f64, bd: u8, chroma: bool| -> u32 {
        let bd = u32::from(bd.clamp(8, 16));
        let max = f64::from((1u32 << bd) - 1);
        let v = match (full, chroma) {
            (true, false) => e * max,
            (true, true) => e * max + f64::from(1u32 << (bd - 1)),
            (false, false) => f64::from(1u32 << (bd - 8)) * (219.0 * e + 16.0),
            (false, true) => f64::from(1u32 << (bd - 8)) * (224.0 * e + 128.0),
        };
        v.round().clamp(0.0, max) as u32
    };
    if matrix == 0 {
        // Identity (GBR): the planes are G, B, R, quantised like luma.
        return [q(g, bd_l, false), q(b, bd_c, false), q(r, bd_c, false)];
    }
    let y = kr * r + (1.0 - kr - kb) * g + kb * b;
    [
        q(y, bd_l, false),
        q((b - y) / (2.0 * (1.0 - kb)), bd_c, true),
        q((r - y) / (2.0 * (1.0 - kr)), bd_c, true),
    ]
}

/// Fills the canvas planes with the `iovl` fill colour (R, G, B, A, 16 bits each). A luma-only
/// canvas (an alpha plane) gets A scaled to the luma bit depth, a colour canvas gets R, G, B
/// as [`fill_value`].
fn fill_canvas(out: &mut Picture, rgba: [u16; 4], luma_only: bool, cs: (u16, bool)) {
    let [r, g, b, a] = rgba;
    let (bd_l, bd_c) = (out.info.bit_depth_luma, out.info.bit_depth_chroma);
    let v = if luma_only {
        let max = f64::from((1u32 << u32::from(bd_l.clamp(8, 16))) - 1);
        [(f64::from(a) / 65535.0 * max).round() as u32, 0, 0]
    } else {
        fill_value([r, g, b], cs, bd_l, bd_c)
    };
    for (p, v) in out.planes.iter_mut().zip(v) {
        match &mut p.samples {
            Samples::U8(s) => s.fill(v as u8),
            Samples::U16(s) => s.fill(v as u16),
        }
    }
}

/// Paints `pic` on the canvas at the offset of `inp`, chroma at half the offset, clipped. A
/// 4:2:0 canvas with chroma needs even offsets.
fn paint(out: &mut Picture, pic: &Picture, inp: isobmff::OverlayInput) -> Result<()> {
    let (x, y) = (i64::from(inp.x), i64::from(inp.y));
    if out.info.chroma_format == ChromaFormat::Yuv420 && out.planes.len() > 1 && (x | y) & 1 != 0 {
        return Err(Error::Unsupported("overlay offset is odd in a 4:2:0 image"));
    }
    if out.info.chroma_format == ChromaFormat::Yuv420
        && out.planes.len() > 1
        && pic
            .planes
            .first()
            .is_some_and(|p| (p.width | p.height) & 1 != 0)
    {
        return Err(Error::Unsupported(
            "overlay input size is odd in a 4:2:0 image",
        ));
    }
    for (k, (c, p)) in out.planes.iter_mut().zip(&pic.planes).enumerate() {
        let (px, py) = if k == 0 { (x, y) } else { (x / 2, y / 2) };
        blit_at(c, p, px, py)?;
    }
    Ok(())
}

/// A decoded overlay input and the `nclx` (matrix coefficients, full range) of its item.
type Input = (Picture, Option<(u16, bool)>);

/// The first `nclx` (matrix coefficients, full range) of a `colr` list.
fn nclx_of(colour: &[Colour]) -> Option<(u16, bool)> {
    colour.iter().find_map(|c| match *c {
        Colour::Nclx {
            matrix, full_range, ..
        } => Some((matrix, full_range)),
        _ => None,
    })
}

/// Composes an `iovl` on a canvas of the overlay size, in the format of its inputs: the fill
/// colour, then the inputs in order (later on top), clipped. `dec(i, held)` decodes input `i`
/// with `held` bytes already taken. The colour of the fill is converted with `own` (the `nclx`
/// of the overlay item), else the `nclx` of the first input, else the VUI of its SPS, else
/// BT.601 limited range.
fn compose<F>(
    o: &isobmff::Overlay,
    own: Option<(u16, bool)>,
    rest: usize,
    held: usize,
    luma_only: bool,
    cancel: &AtomicBool,
    mut dec: F,
) -> Result<Picture>
where
    F: FnMut(usize, usize) -> Result<Input>,
{
    if o.width == 0 || o.height == 0 {
        return Err(Error::InvalidData("overlay size"));
    }
    let mut inputs = o.inputs.iter();
    let Some(&first_in) = inputs.next() else {
        return Err(Error::Unsupported("overlay without inputs"));
    };
    let (mut first, nclx) = dec(0, held)?;
    let (mut out, add) = canvas(rest, &mut first, o.width, o.height, luma_only, held)?;
    let held = held.saturating_add(add);
    let cs = own
        .or(nclx)
        .or_else(|| {
            first
                .info
                .vui
                .map(|v| (u16::from(v.matrix_coeffs), v.video_full_range))
        })
        .unwrap_or((6, false));
    fill_canvas(&mut out, o.fill, luma_only, cs);
    paint(&mut out, &first, first_in)?;
    drop(first);
    let want = fmt(&out.info);
    for (i, &inp) in inputs.enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let (pic, _) = dec(i + 1, held)?;
        if fmt(&pic.info) != want {
            return Err(Error::Unsupported(
                "overlay inputs differ in chroma format or bit depth",
            ));
        }
        out.hash_ok = merge_hash(out.hash_ok, pic.hash_ok);
        paint(&mut out, &pic, inp)?;
    }
    Ok(out)
}

/// The transforms of `img` in the order they apply: for an identity item, those of its input,
/// then its own.
fn iso_transforms(heif: &Heif, img: &isobmff::Image) -> Result<Vec<isobmff::Transform>> {
    let mut v = Vec::new();
    if let ImageKind::Identity { input } = &img.kind {
        let inner = heif.image(*input).map_err(iso_err)?;
        v.extend_from_slice(&inner.transforms);
    }
    v.extend_from_slice(&img.transforms);
    Ok(v)
}

/// The transforms of `img` as returned: [`iso_transforms`], mapped.
fn transforms_of(heif: &Heif, img: &isobmff::Image) -> Result<Vec<Transform>> {
    Ok(iso_transforms(heif, img)?.iter().map(transform).collect())
}

/// Bytes of the planes and SEI messages of `p`.
fn pic_bytes(p: &Picture) -> usize {
    p.planes
        .iter()
        .map(plane_bytes)
        .sum::<usize>()
        .saturating_add(sei_bytes(&p.sei))
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

    /// Applies `t` to `planes` (luma first, then the chroma planes of `cf`) with
    /// `isobmff::apply_transforms`, which may book what is left of `rest` after `held` bytes (the
    /// planes themselves included) for the buffers it makes.
    fn apply(
        &self,
        planes: Vec<Plane>,
        cf: ChromaFormat,
        t: &[isobmff::Transform],
        held: usize,
    ) -> Result<Vec<Plane>> {
        if t.is_empty() {
            return Ok(planes);
        }
        let (width, height) = planes
            .first()
            .map(|p| (p.width, p.height))
            .ok_or(Error::InvalidData("no plane"))?;
        let chroma = match cf {
            ChromaFormat::Mono => isobmff::Chroma::Mono,
            ChromaFormat::Yuv420 => isobmff::Chroma::Yuv420,
        };
        let planes = isobmff::Planes {
            width,
            height,
            chroma,
            planes: planes
                .into_iter()
                .map(|p| isobmff::Plane {
                    width: p.width,
                    height: p.height,
                    samples: match p.samples {
                        Samples::U8(v) => isobmff::PlaneSamples::U8(v),
                        Samples::U16(v) => isobmff::PlaneSamples::U16(v),
                    },
                })
                .collect(),
        };
        let iso = isobmff::Limits {
            max_alloc: self.rest.checked_sub(held).ok_or(Error::AllocLimit)?,
            max_threads: 1,
            max_pixels: self.limits.max_pixels,
            ..isobmff::Limits::default()
        };
        let out = isobmff::apply_transforms(planes, t, &iso, self.cancel).map_err(iso_err)?;
        // Only 4:2:2 turned by 90 or 270 degrees changes the format, and there is no 4:2:2 here.
        if out.chroma != chroma {
            return Err(Error::InvalidData("transform changed the chroma format"));
        }
        Ok(out
            .planes
            .into_iter()
            .map(|p| Plane {
                width: p.width,
                height: p.height,
                samples: match p.samples {
                    isobmff::PlaneSamples::U8(v) => Samples::U8(v),
                    isobmff::PlaneSamples::U16(v) => Samples::U16(v),
                },
            })
            .collect())
    }

    /// `pic` with `t` applied and its size set to the result; `other` bytes besides `pic` are
    /// held meanwhile.
    fn transformed(
        &self,
        mut pic: Picture,
        t: &[isobmff::Transform],
        other: usize,
    ) -> Result<Picture> {
        let held = pic_bytes(&pic).saturating_add(other);
        let cf = pic.info.chroma_format;
        pic.planes = self.apply(std::mem::take(&mut pic.planes), cf, t, held)?;
        if let Some(p) = pic.planes.first() {
            (pic.info.width, pic.info.height) = (p.width, p.height);
        }
        Ok(pic)
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

    /// A coded, grid, identity or overlay image item; `what` is the `Unsupported` text for any
    /// other kind (and for an identity or overlay input that is not coded or a grid).
    fn image(&self, id: u32, held: usize, luma_only: bool, what: &'static str) -> Result<Picture> {
        let img = self.heif.image(id).map_err(iso_err)?;
        match &img.kind {
            ImageKind::Identity { input } => {
                let inner = self.heif.image(*input).map_err(iso_err)?;
                self.leaf(&inner, held, luma_only, what)
            }
            ImageKind::Overlay(o) => self.overlay(o, &img.colour, held, luma_only, what),
            _ => self.leaf(&img, held, luma_only, what),
        }
    }

    /// A coded or grid image item.
    fn leaf(
        &self,
        img: &isobmff::Image,
        held: usize,
        luma_only: bool,
        what: &'static str,
    ) -> Result<Picture> {
        match &img.kind {
            ImageKind::Coded {
                codec: Codec::Hevc,
                config: CodecConfig::Hevc(hv),
            } => self.coded(img.id, hv, held),
            ImageKind::Grid(g) => self.grid(g, held, luma_only),
            _ => Err(Error::Unsupported(what)),
        }
    }

    /// Grid tile `i`: a coded HEVC item that decodes to its `ispe` size, and that size.
    fn grid_tile(&self, g: &isobmff::Grid, i: usize, held: usize) -> Result<(Picture, (u32, u32))> {
        self.cancelled()?;
        let tid = *g
            .tiles
            .get(i)
            .ok_or(Error::InvalidData("grid tile count"))?;
        let t = self.heif.image(tid).map_err(iso_err)?;
        let ImageKind::Coded {
            codec: Codec::Hevc,
            config: CodecConfig::Hevc(hv),
        } = &t.kind
        else {
            return Err(Error::Unsupported("grid tile is not HEVC"));
        };
        // libheif turns each tile by its own transforms before it is pasted; not done here.
        if self.opts.apply_transforms && !t.transforms.is_empty() {
            return Err(Error::Unsupported(
                "grid tile with transformative properties",
            ));
        }
        let tile = self.coded(tid, hv, held)?;
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
        Ok((tile, (pw, ph)))
    }

    fn grid(&self, g: &isobmff::Grid, held: usize, luma_only: bool) -> Result<Picture> {
        let (cols, rows) = (g.columns, g.rows);
        if cols == 0 || rows == 0 || u64::from(cols) * u64::from(rows) != g.tiles.len() as u64 {
            return Err(Error::InvalidData("grid tile count"));
        }
        check_size(g.width, g.height, self.limits)?;
        let (mut first, (tw, th)) = self.grid_tile(g, 0, held)?;
        let covers = u64::from(cols) * u64::from(tw) >= u64::from(g.width)
            && u64::from(rows) * u64::from(th) >= u64::from(g.height);
        if !covers {
            return Err(Error::InvalidData("grid tiles do not cover the output"));
        }
        let odd = (cols > 1 && !tw.is_multiple_of(2)) || (rows > 1 && !th.is_multiple_of(2));
        if first.info.chroma_format == ChromaFormat::Yuv420 && odd {
            return Err(Error::InvalidData("odd tile size in a 4:2:0 grid"));
        }
        let tile_bytes = first.planes.iter().map(plane_bytes).sum::<usize>();
        let (mut out, add) = canvas(self.rest, &mut first, g.width, g.height, luma_only, held)?;
        place(&mut out.planes, &first.planes, 0, 0)?;
        drop(first);
        let held = held.saturating_add(add);
        let want = fmt(&out.info);
        let one = Limits {
            max_threads: 1,
            ..*self.limits
        };
        // Tile `i` on `workers` threads of one call: each gets its share of the budget, of the
        // isobmff item bytes, and no threads of its own.
        let dec = |i: usize, budget: usize, workers: usize| -> Result<Picture> {
            let iso = isobmff::Limits {
                max_alloc: self.iso.max_alloc / workers.max(1),
                ..*self.iso
            };
            let cx = Ctx {
                iso: &iso,
                rest: budget,
                limits: if workers > 1 { &one } else { self.limits },
                ..*self
            };
            let (tile, dims) = cx.grid_tile(g, i, 0)?;
            if fmt(&tile.info) != want || dims != (tw, th) {
                return Err(Error::InvalidData("grid tiles differ in size or format"));
            }
            Ok(tile)
        };
        let fill = Fill {
            cols: cols as usize,
            n: g.tiles.len(),
            avail: self.rest.checked_sub(held).ok_or(Error::AllocLimit)?,
            per: tile_bytes.saturating_mul(3).saturating_add(SMALL),
            threads: self.limits.max_threads.max(1),
            cancel: self.cancel,
        };
        fill_tiles(&mut out, &fill, dec)?;
        Ok(out)
    }

    fn overlay(
        &self,
        o: &isobmff::Overlay,
        colour: &[Colour],
        held: usize,
        luma_only: bool,
        what: &'static str,
    ) -> Result<Picture> {
        check_size(o.width, o.height, self.limits)?;
        let own = nclx_of(colour);
        compose(
            o,
            own,
            self.rest,
            held,
            luma_only,
            self.cancel,
            |i, held| {
                let inp = o.inputs.get(i).ok_or(Error::InvalidData("overlay input"))?;
                let img = self.heif.image(inp.id).map_err(iso_err)?;
                // Pixels are not transformed here, so an input's own transforms cannot be applied
                // before it is placed.
                if !img.transforms.is_empty() {
                    return Err(Error::Unsupported(
                        "overlay input with transformative properties",
                    ));
                }
                Ok((
                    self.leaf(&img, held, luma_only, what)?,
                    nclx_of(&img.colour),
                ))
            },
        )
    }
}

/// Decodes the primary image of a HEIC file, and its alpha plane when it has one.
///
/// The file has a `meta` box or a `mini` box (ISO/IEC 23008-12 Annex O); isobmff-rs gives both
/// the same item graph, so a `mini` file's orientation comes back as `irot`/`imir` in
/// [`HeicImage::transforms`] and its alpha item is decoded like any other. The primary item is one `hvc1` image, a `grid` of `hvc1` tiles, an `iden` (identity) item
/// whose input is one of those, or an `iovl` (overlay) of those; each coded item is decoded
/// with [`decode_hvcc_into`] and must hold exactly one picture. The metadata (SEI messages, VUI)
/// is that of the first tile (first input of an overlay), kept unless [`Options::metadata`] is
/// false. By default the result is not rotated, mirrored or cropped by the `irot`, `imir` and
/// `clap` properties; they are returned raw in [`HeicImage::transforms`] (for an `iden` item:
/// those of its input, then its own). With [`Options::apply_transforms`] they are applied in that
/// order by `isobmff::apply_transforms` (the list is still returned, and
/// [`HeicImage::transforms_applied`] is true), and the alpha plane gets the alpha item's own
/// transforms; a grid whose tiles have transforms is then `Unsupported`. An overlay is composed in the YCbCr layout of its inputs: the canvas
/// is filled with the `iovl` fill colour converted to YCbCr, the inputs are painted in order at
/// their offsets (later on top, clipped, no blending); see `docs/design.md`. The alpha image
/// (`urn:mpeg:hevc:2015:auxid:1` or `urn:mpeg:mpegB:cicp:systems:auxiliary:alpha`, or
/// `urn:mpeg:avc:2015:auxid:1`) is decoded the same way and only its luma plane is kept.
///
/// Limits: `max_alloc` is the peak of the whole call. A quarter of it goes to isobmff (an eighth
/// for the parse, which stays alive, and an eighth for the item bytes of the items being decoded
/// at that moment, shared by the tile workers); the decoded pictures, the assembled planes and the
/// decoder's working memory share the rest, the tile workers of a grid splitting what is left.
/// With [`Options::apply_transforms`], the buffers the transforms make get what is left of that
/// share after the decoded image and alpha plane.
/// `max_threads` is the number of threads that decode the tiles of a grid (the calling thread
/// included; 1 spawns none) and is 1 for isobmff. The output and the error are the same for every
/// thread count. `max_width`, `max_height` and `max_pixels` apply to every coded picture and to
/// the output size of a grid or overlay. The cancel flag is read before each coded item, so
/// between tiles, and inside the decoder.
///
/// # Errors
///
/// [`Error::Unsupported`] when the primary (or alpha) image is not HEVC, a grid of HEVC tiles or
/// an identity or overlay of those, or when a coded item is protected (`protection_index` not 0),
/// for a grid tile with transformative properties when [`Options::apply_transforms`] is set,
/// or for an overlay with no input, with an input that has transformative properties, whose
/// inputs differ in chroma format or bit depth, or with an odd offset in a 4:2:0 image;
/// [`Error::InvalidData`] for a
/// container error (the text is the isobmff error class, for example `"truncated"`), an item that
/// is not exactly one picture, or tiles that differ in size or format, do not cover the grid or
/// decode to another size than their `ispe`, and (with [`Options::apply_transforms`]) a `clap` or
/// `irot` that `isobmff::apply_transforms` rejects (`"malformed"`); the errors of [`decode_hvcc_into`] (including
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
    let mut image = cx.image(primary.id, 0, false, "primary image is not HEVC")?;
    let mut alpha = match primary.alpha {
        Some(id) => {
            let a = cx.image(id, pic_bytes(&image), true, "alpha image is not HEVC")?;
            a.planes.into_iter().next()
        }
        None => None,
    };
    if opts.apply_transforms {
        let other = alpha.as_ref().map_or(0, plane_bytes);
        image = cx.transformed(image, &iso_transforms(&heif, &primary)?, other)?;
        // The alpha item takes its own transforms, as in libheif (docs/design.md).
        if let (Some(a), Some(id)) = (alpha.take(), primary.alpha) {
            let t = iso_transforms(&heif, &heif.image(id).map_err(iso_err)?)?;
            let held = pic_bytes(&image).saturating_add(plane_bytes(&a));
            alpha = cx.apply(vec![a], ChromaFormat::Mono, &t, held)?.pop();
        }
    }
    Ok(HeicImage {
        image,
        alpha,
        transforms: transforms_of(&heif, &primary)?,
        transforms_applied: opts.apply_transforms,
    })
}

/// Decodes image item `id` of `heif`, the parse of `file`, with no parse of its own.
///
/// The item is one `hvc1` image, a `grid`, an `iden` or an `iovl`, decoded as [`decode_heic`] decodes the
/// primary image: each coded item must hold exactly one picture, and the metadata (SEI messages,
/// VUI) is that of the first tile, kept unless [`Options::metadata`] is false. No alpha plane is
/// decoded. The `irot`, `imir` and `clap` properties are not applied (read them from
/// `heif.image(id)`) unless [`Options::apply_transforms`] is set: then the item's transforms (for
/// an `iden` item, those of its input first) are applied as in [`decode_heic`], and `info.width`
/// and `info.height` are the size after them.
///
/// Limits: `max_alloc` is the peak of the call. An eighth of it goes to the isobmff item bytes
/// (shared by the tile workers); the decoded pictures, the assembled planes and the decoder's working
/// memory share the rest (the buffers of the transforms get what is left of it after the decoded
/// picture). `heif` itself is not charged: the caller holds it. `max_width`,
/// `max_height` and `max_pixels` apply to every coded picture and to the output size of a grid or
/// overlay. The cancel flag is read before each coded item and inside the decoder.
///
/// # Errors
///
/// [`Error::InvalidData`] for an `id` that is not an item (the text is `"no-such-item"`) and for
/// the other errors of [`decode_heic`]; [`Error::Unsupported`] when the item is not HEVC, a grid, an
/// identity or an overlay of those, or is protected; the limit and cancel errors of [`decode_heic`].
pub fn decode_heic_item(
    heif: &Heif,
    file: &[u8],
    id: u32,
    opts: &Options,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Picture> {
    let iso = isobmff::Limits {
        max_alloc: limits.max_alloc / 8,
        max_threads: 1,
        max_pixels: limits.max_pixels,
        ..isobmff::Limits::default()
    };
    let cx = Ctx {
        heif,
        file,
        opts,
        limits,
        cancel,
        iso: &iso,
        rest: limits.max_alloc - limits.max_alloc / 8,
    };
    let pic = cx.image(id, 0, false, "image item is not HEVC")?;
    if !opts.apply_transforms {
        return Ok(pic);
    }
    let img = heif.image(id).map_err(iso_err)?;
    cx.transformed(pic, &iso_transforms(heif, &img)?, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Ledger;

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
        assert_eq!(iso_err(E::NoMeta), Error::InvalidData("no-meta"));
        assert_eq!(merge_hash(Some(true), Some(true)), Some(true));
        assert_eq!(merge_hash(Some(true), None), None);
        assert_eq!(merge_hash(Some(true), Some(false)), Some(false));
    }

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = (body.len() as u32 + 8).to_be_bytes().to_vec();
        v.extend_from_slice(kind);
        v.extend_from_slice(body);
        v
    }

    fn fbx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        bx(kind, &[&[0u8; 4][..], body].concat())
    }

    /// One `hvc1` item (id 1, 64 x 48, junk payload) in a hand-built HEIF.
    fn one_item_file() -> Vec<u8> {
        let hvcc = bx(
            b"hvcC",
            &[
                &[
                    1u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 0xF0, 0, 0xFC, 0xFD, 0xF8, 0xF8, 0,
                    0, 3, 1, 0xA0, 0, 1, 0, 3,
                ][..],
                &[0x40, 0x01, 0x02],
            ]
            .concat(),
        );
        let ispe = fbx(
            b"ispe",
            &[&64u32.to_be_bytes()[..], &48u32.to_be_bytes()].concat(),
        );
        let ftyp = bx(b"ftyp", b"heic\0\0\0\0heic");
        let meta = |offset: u32| {
            let hdlr = fbx(b"hdlr", &[&[0u8; 4][..], b"pict", &[0; 12], &[0]].concat());
            let pitm = fbx(b"pitm", &[0, 1]);
            let infe = fbx(b"infe", &[&[0u8, 1, 0, 0][..], b"hvc1", &[0]].concat());
            let iinf = fbx(b"iinf", &[&[0u8, 1][..], &infe].concat());
            let iloc = fbx(
                b"iloc",
                &[
                    &[0x44u8, 0, 0, 1, 0, 1, 0, 0, 0, 1][..],
                    &offset.to_be_bytes(),
                    &4u32.to_be_bytes(),
                ]
                .concat(),
            );
            let ipco = bx(b"ipco", &[&ispe[..], &hvcc].concat());
            let ipma = fbx(b"ipma", &[0u8, 0, 0, 1, 0, 1, 2, 1, 0x82]);
            let iprp = bx(b"iprp", &[&ipco[..], &ipma].concat());
            fbx(b"meta", &[&hdlr[..], &pitm, &iinf, &iloc, &iprp].concat())
        };
        let offset = (ftyp.len() + meta(0).len() + 8) as u32;
        [ftyp, meta(offset), bx(b"mdat", &[1, 2, 3, 4])].concat()
    }

    #[test]
    fn item_with_a_bad_id_or_bad_data_is_an_error() {
        let file = one_item_file();
        let cancel = AtomicBool::new(false);
        let iso = isobmff::Limits::default();
        let heif = isobmff::parse(&file, &iso, &cancel).expect("hand-built file parses");
        let (o, l) = (Options::default(), Limits::default());
        let r = decode_heic_item(&heif, &file, 99, &o, &l, &cancel);
        assert!(
            matches!(r, Err(Error::InvalidData("no-such-item"))),
            "{r:?}"
        );
        assert!(decode_heic_item(&heif, &file, 1, &o, &l, &cancel).is_err());
    }
    #[test]
    fn transforms_are_charged_and_cancelled() {
        let file = one_item_file();
        let cancel = AtomicBool::new(false);
        let iso = isobmff::Limits::default();
        let heif = isobmff::parse(&file, &iso, &cancel).expect("hand-built file parses");
        let (opts, limits) = (Options::default(), Limits::default());
        let p = pic(4, 2, ChromaFormat::Yuv420, [1, 2, 3]);
        let bytes = pic_bytes(&p);
        let rot = [isobmff::Transform::Rotate(90)];
        let big = Ctx {
            heif: &heif,
            file: &file,
            opts: &opts,
            limits: &limits,
            cancel: &cancel,
            iso: &iso,
            rest: 1 << 20,
        };
        let ok = big.transformed(p.clone(), &rot, 0).unwrap();
        assert_eq!((ok.info.width, ok.info.height), (2, 4));
        assert_eq!((ok.planes[1].width, ok.planes[1].height), (1, 2));
        // The new planes (12 bytes) and their list do not fit in what is left after the input.
        let tight = Ctx {
            rest: 2 * bytes,
            ..big
        };
        assert_eq!(
            tight.transformed(p.clone(), &rot, 0),
            Err(Error::AllocLimit)
        );
        // The bytes held besides the picture count.
        assert_eq!(
            big.transformed(p.clone(), &rot, 1 << 20),
            Err(Error::AllocLimit)
        );
        // An empty list needs nothing.
        let none = Ctx { rest: bytes, ..big };
        assert_eq!(none.transformed(p.clone(), &[], 0), Ok(p.clone()));
        let yes = AtomicBool::new(true);
        let stop = Ctx {
            cancel: &yes,
            ..big
        };
        assert_eq!(stop.transformed(p, &rot, 0), Err(Error::Cancelled));
    }

    #[test]
    fn garbage_is_invalid_not_a_panic() {
        let cancel = AtomicBool::new(false);
        let r = decode_heic(&[0; 16], &Options::default(), &Limits::default(), &cancel);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }

    fn pic(w: u32, h: u32, cf: ChromaFormat, v: [u8; 3]) -> Picture {
        let mut planes = vec![plane(w, h, v[0])];
        if cf == ChromaFormat::Yuv420 {
            planes.push(plane(w.div_ceil(2), h.div_ceil(2), v[1]));
            planes.push(plane(w.div_ceil(2), h.div_ceil(2), v[2]));
        }
        Picture {
            info: Info {
                width: w,
                height: h,
                coded_width: w,
                coded_height: h,
                chroma_format: cf,
                bit_depth_luma: 8,
                bit_depth_chroma: 8,
                profile_idc: 1,
                level_idc: 93,
                vui: None,
            },
            planes,
            poc: 0,
            hash_ok: Some(true),
            sei: Vec::new(),
        }
    }

    #[test]
    fn blit_at_clips_on_every_side() {
        let mut c = blank(&plane(1, 1, 0), 4, 4).unwrap();
        let src = Plane {
            width: 3,
            height: 3,
            samples: Samples::U8((1..=9).collect()),
        };
        // Top-left at (-1, -2): only source (1, 2) and (2, 2) show, at (0, 0) and (1, 0).
        blit_at(&mut c, &src, -1, -2).unwrap();
        // Bottom-right corner: only source (0, 0).
        blit_at(&mut c, &src, 3, 3).unwrap();
        // Wholly outside.
        for (x, y) in [(9, 0), (-3, 0), (0, 4), (0, -3)] {
            blit_at(&mut c, &plane(3, 3, 5), x, y).unwrap();
        }
        assert_eq!(
            rows(&c),
            [vec![8, 9, 0, 0], vec![0; 4], vec![0; 4], vec![0, 0, 0, 1]]
        );
    }

    #[test]
    fn fill_colour_conversion() {
        let w = [65535u16; 3];
        // BT.601 limited, 8 bit: white, black.
        assert_eq!(fill_value(w, (6, false), 8, 8), [235, 128, 128]);
        assert_eq!(fill_value([0; 3], (6, false), 8, 8), [16, 128, 128]);
        // 10 bit scales by 4.
        assert_eq!(fill_value(w, (1, false), 10, 10), [940, 512, 512]);
        assert_eq!(fill_value([0; 3], (1, false), 10, 10), [64, 512, 512]);
        // Full range BT.601 red: Y = 0.299 * 255, Cb = 128 - 0.1687 * 255, Cr clipped at 255.
        assert_eq!(fill_value([65535, 0, 0], (6, true), 8, 8), [76, 85, 255]);
        // Identity: G, B, R.
        assert_eq!(
            fill_value([65535, 0, 32768], (0, true), 8, 8),
            [0, 128, 255]
        );
        // The alpha of a luma-only canvas.
        let mut a = pic(2, 2, ChromaFormat::Mono, [0; 3]);
        fill_canvas(&mut a, [1, 2, 3, 65535], true, (6, false));
        assert_eq!(rows(&a.planes[0]), [vec![255; 2], vec![255; 2]]);
        fill_canvas(&mut a, [1, 2, 3, 0], true, (6, false));
        assert_eq!(rows(&a.planes[0]), [vec![0; 2], vec![0; 2]]);
    }

    fn overlay_of(w: u32, h: u32, inputs: &[(i32, i32)]) -> isobmff::Overlay {
        isobmff::Overlay {
            fill: [65535; 4],
            width: w,
            height: h,
            inputs: inputs
                .iter()
                .map(|&(x, y)| isobmff::OverlayInput { id: 0, x, y })
                .collect(),
        }
    }

    fn compose_of(o: &isobmff::Overlay, pics: &[Picture], rest: usize) -> Result<Picture> {
        let cancel = AtomicBool::new(false);
        compose(o, None, rest, 0, false, &cancel, |i, _| {
            pics.get(i)
                .cloned()
                .map(|p| (p, None))
                .ok_or(Error::InvalidData("input"))
        })
    }

    #[test]
    fn overlay_paints_in_order_clipped_in_yuv420() {
        let o = overlay_of(8, 8, &[(0, 0), (2, 2), (6, -2)]);
        let c = ChromaFormat::Yuv420;
        let pics = [
            pic(4, 4, c, [100, 50, 60]),
            pic(4, 4, c, [200, 51, 61]),
            pic(4, 4, c, [30, 52, 62]),
        ];
        let out = compose_of(&o, &pics, 1 << 20).unwrap();
        assert_eq!((out.info.width, out.info.height), (8, 8));
        assert_eq!((out.info.coded_width, out.info.coded_height), (8, 8));
        let top = vec![100, 100, 100, 100, 235, 235, 30, 30];
        let mid = vec![100, 100, 200, 200, 200, 200, 235, 235];
        let low = vec![235, 235, 200, 200, 200, 200, 235, 235];
        let rest = vec![235; 8];
        assert_eq!(
            rows(&out.planes[0]),
            [
                top.clone(),
                top,
                mid.clone(),
                mid,
                low.clone(),
                low,
                rest.clone(),
                rest
            ]
        );
        // The white fill is Cb = Cr = 128 in BT.601 limited range; chroma is at half offsets.
        assert_eq!(
            rows(&out.planes[1]),
            [
                vec![50, 50, 128, 52],
                vec![50, 51, 51, 128],
                vec![128, 51, 51, 128],
                vec![128; 4]
            ]
        );
        assert_eq!(out.planes[2].width, 4);
    }

    #[test]
    fn overlay_refusals() {
        let c = ChromaFormat::Yuv420;
        let a = pic(4, 4, c, [1, 2, 3]);
        let odd = overlay_of(8, 8, &[(0, 0), (1, 0)]);
        assert!(matches!(
            compose_of(&odd, &[a.clone(), a.clone()], 1 << 20),
            Err(Error::Unsupported(_))
        ));
        // An odd offset is fine without chroma.
        let m = pic(4, 4, ChromaFormat::Mono, [1, 0, 0]);
        assert!(compose_of(&odd, &[m.clone(), m.clone()], 1 << 20).is_ok());
        // An odd input size in 4:2:0.
        let o3 = pic(3, 4, c, [1, 2, 3]);
        let even0 = overlay_of(8, 8, &[(0, 0)]);
        assert!(matches!(
            compose_of(&even0, &[o3], 1 << 20),
            Err(Error::Unsupported(_))
        ));
        // Different formats.
        let even = overlay_of(8, 8, &[(0, 0), (2, 0)]);
        assert!(matches!(
            compose_of(&even, &[a.clone(), m], 1 << 20),
            Err(Error::Unsupported(_))
        ));
        let mut deep = a.clone();
        deep.info.bit_depth_luma = 10;
        assert!(matches!(
            compose_of(&even, &[a.clone(), deep], 1 << 20),
            Err(Error::Unsupported(_))
        ));
        // No input, no size, a canvas over the budget.
        assert!(matches!(
            compose_of(&overlay_of(8, 8, &[]), &[], 1 << 20),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            compose_of(
                &overlay_of(0, 8, &[(0, 0)]),
                std::slice::from_ref(&a),
                1 << 20
            ),
            Err(Error::InvalidData(_))
        ));
        let r = compose_of(&even, &[a.clone(), a], 100);
        assert_eq!(r, Err(Error::AllocLimit));
    }

    #[test]
    fn overlay_alpha_canvas_is_luma_only() {
        let o = overlay_of(4, 2, &[(1, 0)]);
        let m = pic(2, 2, ChromaFormat::Mono, [9, 0, 0]);
        let cancel = AtomicBool::new(false);
        let out = compose(&o, None, 1 << 20, 0, true, &cancel, |_, _| {
            Ok((m.clone(), None))
        })
        .unwrap();
        assert_eq!(out.planes.len(), 1);
        assert_eq!(rows(&out.planes[0]), vec![vec![255, 9, 9, 255]; 2]);
    }

    /// The result of `fill_tiles` on tiles `1..12` of a 3 x 4 grid of 2 x 2 tiles (tile `i` is all
    /// `i + 1`), decoded by a fake that needs `need` bytes of its budget and books them in a
    /// ledger; and what the fake saw: most decodes at once, most budget granted at once.
    struct Run {
        result: Result<()>,
        out: Picture,
        threads_seen: usize,
        granted_seen: usize,
        peak: u64,
    }

    fn run(threads: usize, avail: usize, per: usize, need: usize, fail: &[usize]) -> Run {
        let cancel = AtomicBool::new(false);
        run_with(&cancel, threads, avail, per, need, fail)
    }

    fn run_with(
        cancel: &AtomicBool,
        threads: usize,
        avail: usize,
        per: usize,
        need: usize,
        fail: &[usize],
    ) -> Run {
        let tile = |i: usize| {
            let mut p = pic(2, 2, ChromaFormat::Mono, [i as u8 + 1, 0, 0]);
            p.hash_ok = if i == 9 { Some(false) } else { Some(true) };
            p
        };
        let mut out = pic(6, 8, ChromaFormat::Mono, [0; 3]);
        place(&mut out.planes, &tile(0).planes, 0, 0).unwrap();
        let ledger = Ledger::default();
        ledger.reset(SMALL + avail, 0).unwrap();
        let (active, active_max) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let (granted, granted_max) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let dec = |i: usize, budget: usize, _workers: usize| -> Result<Picture> {
            active_max.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            granted_max.fetch_max(
                granted.fetch_add(budget, Ordering::SeqCst) + budget,
                Ordering::SeqCst,
            );
            let r = if fail.contains(&i) {
                Err(Error::InvalidData("tile"))
            } else if budget < need {
                Err(Error::AllocLimit)
            } else {
                // The booking is held until the end of the decode.
                ledger.charge(need).map(|held| (held, tile(i)))
            };
            granted.fetch_sub(budget, Ordering::SeqCst);
            active.fetch_sub(1, Ordering::SeqCst);
            r.map(|(_held, t)| t)
        };
        let fill = Fill {
            cols: 3,
            n: 12,
            avail,
            per,
            threads,
            cancel,
        };
        let result = fill_tiles(&mut out, &fill, dec);
        Run {
            result,
            out,
            threads_seen: active_max.load(Ordering::SeqCst),
            granted_seen: granted_max.load(Ordering::SeqCst),
            peak: ledger.peak(),
        }
    }

    #[test]
    fn tiles_are_the_same_for_every_thread_count_and_budget() {
        let need = 1000;
        let fails: [&[usize]; 5] = [&[], &[5], &[7, 3], &[11], &[1]];
        for fail in fails {
            for avail in [need, 3 * need, 5 * need + 100, 40 * need] {
                // Guesses of what a tile needs, from far under (many workers, every one out of
                // budget and redone alone) to exact.
                for per in [1, need, 2 * need] {
                    let want = run(1, avail, per, need, fail);
                    for threads in [0, 2, 4, 8] {
                        let got = run(threads, avail, per, need, fail);
                        assert_eq!(got.result, want.result, "{fail:?} {avail} {per} {threads}");
                        if want.result.is_ok() {
                            assert_eq!(got.out, want.out, "{fail:?} {avail} {per} {threads}");
                        }
                        assert!(got.threads_seen <= threads.max(1));
                        assert!(got.granted_seen <= avail, "{} > {avail}", got.granted_seen);
                        assert!(got.peak <= (SMALL + avail) as u64);
                    }
                }
            }
        }
        let ok = run(4, 40 * need, need, need, &[]);
        assert_eq!(ok.result, Ok(()));
        // Every tile is on the canvas, and one tile failed its hash.
        assert_eq!(
            rows(&ok.out.planes[0]),
            [
                vec![1, 1, 2, 2, 3, 3],
                vec![1, 1, 2, 2, 3, 3],
                vec![4, 4, 5, 5, 6, 6],
                vec![4, 4, 5, 5, 6, 6],
                vec![7, 7, 8, 8, 9, 9],
                vec![7, 7, 8, 8, 9, 9],
                vec![10, 10, 11, 11, 12, 12],
                vec![10, 10, 11, 11, 12, 12]
            ]
        );
        assert_eq!(ok.out.hash_ok, Some(false));
        // The first error in tile order wins, whichever thread met which first.
        let two = run(4, 40 * need, need, need, &[7, 3]);
        assert_eq!(two.result, Err(Error::InvalidData("tile")));
    }

    #[test]
    fn tiles_that_never_fit_fail_like_serial() {
        // One tile needs more than all that is available: AllocLimit with any thread count.
        for threads in [1, 2, 4] {
            let r = run(threads, 500, 1, 1000, &[]);
            assert_eq!(r.result, Err(Error::AllocLimit), "{threads}");
        }
    }

    #[test]
    fn tiles_stop_on_cancel() {
        let cancel = AtomicBool::new(true);
        for threads in [1, 2, 4] {
            let r = run_with(&cancel, threads, 1 << 20, 1, 1, &[]);
            assert_eq!(r.result, Err(Error::Cancelled), "{threads}");
        }
    }

    /// `hvc1` item 1 (junk payload, `imir`) and the `iden` item 2 (`irot`) that is the primary.
    fn iden_file() -> Vec<u8> {
        let hvcc = bx(
            b"hvcC",
            &[
                &[
                    1u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 0xF0, 0, 0xFC, 0xFD, 0xF8, 0xF8, 0,
                    0, 3, 1, 0xA0, 0, 1, 0, 3,
                ][..],
                &[0x40, 0x01, 0x02],
            ]
            .concat(),
        );
        let ispe = fbx(
            b"ispe",
            &[&64u32.to_be_bytes()[..], &48u32.to_be_bytes()].concat(),
        );
        let ftyp = bx(b"ftyp", b"heic\0\0\0\0heic");
        let meta = |offset: u32| {
            let hdlr = fbx(b"hdlr", &[&[0u8; 4][..], b"pict", &[0; 12], &[0]].concat());
            let pitm = fbx(b"pitm", &[0, 2]);
            let infe = |id: u8, kind: &[u8; 4]| {
                bx(
                    b"infe",
                    &[&[2u8, 0, 0, 0, 0, id, 0, 0][..], kind, &[0]].concat(),
                )
            };
            let iinf = fbx(
                b"iinf",
                &[&[0u8, 2][..], &infe(1, b"hvc1"), &infe(2, b"iden")].concat(),
            );
            let iloc = fbx(
                b"iloc",
                &[
                    &[0x44u8, 0, 0, 1, 0, 1, 0, 0, 0, 1][..],
                    &offset.to_be_bytes(),
                    &4u32.to_be_bytes(),
                ]
                .concat(),
            );
            let iref = fbx(b"iref", &bx(b"dimg", &[0, 2, 0, 1, 0, 1]));
            let ipco = bx(
                b"ipco",
                &[&ispe[..], &hvcc, &bx(b"irot", &[1]), &bx(b"imir", &[1])].concat(),
            );
            let ipma = fbx(
                b"ipma",
                &[0u8, 0, 0, 2, 0, 1, 3, 1, 0x82, 0x84, 0, 2, 1, 0x83],
            );
            let iprp = bx(b"iprp", &[&ipco[..], &ipma].concat());
            fbx(
                b"meta",
                &[&hdlr[..], &pitm, &iinf, &iloc, &iref, &iprp].concat(),
            )
        };
        let offset = (ftyp.len() + meta(0).len() + 8) as u32;
        [ftyp, meta(offset), bx(b"mdat", &[1, 2, 3, 4])].concat()
    }

    #[test]
    fn identity_decodes_its_input_and_appends_its_transforms() {
        let file = iden_file();
        let cancel = AtomicBool::new(false);
        let iso = isobmff::Limits::default();
        let heif = isobmff::parse(&file, &iso, &cancel).expect("hand-built file parses");
        let img = heif.primary_image().unwrap();
        assert!(matches!(img.kind, ImageKind::Identity { input: 1 }));
        // The input's own transforms first, then the identity item's.
        assert_eq!(
            transforms_of(&heif, &img).unwrap(),
            [Transform::Mirror(Axis::LeftRight), Transform::Rotate(90)]
        );
        // The input (junk) is decoded: the error comes from there, not from the item kind.
        let r = decode_heic(&file, &Options::default(), &Limits::default(), &cancel);
        assert!(r.is_err());
        assert!(
            !matches!(r, Err(Error::Unsupported("primary image is not HEVC"))),
            "{r:?}"
        );
    }

    /// A `mini` box file (ISO/IEC 23008-12 Annex O, `ftyp` minor version `heic`): a 64 x 48
    /// 4:2:0 8-bit `hvc1` image with the `hvcC` record of `one_item_file` and junk item data,
    /// orientation 5, an alpha item that shares the main configuration, and an Exif chunk.
    fn mini_file() -> Vec<u8> {
        let cfg = [
            1u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 0xF0, 0, 0xFC, 0xFD, 0xF8, 0xF8, 0, 0, 3, 1,
            0xA0, 0, 1, 0, 3, 0x40, 0x01, 0x02,
        ];
        let (alpha, main, exif): (&[u8], &[u8], &[u8]) = (b"ALPHA", &[1, 2, 3, 4], b"II*\0EXIF");
        let mut bits = Vec::new();
        let mut put = |v: usize, n: u32| bits.extend((0..n).rev().map(|i| (v >> i) & 1 == 1));
        put(0, 2); // version
        // explicit codec types, float, full range, alpha, explicit CICP, hdr, icc, exif, xmp
        put(0b0_0110_0010, 9);
        put(1, 2); // 4:2:0
        put(4, 3); // orientation 5
        put(0, 1); // small dimensions
        put(63, 7);
        put(47, 7);
        put(0, 2); // chroma not centred, horizontally and vertically
        put(0, 1); // 8 bits
        put(0, 1); // alpha not premultiplied
        put(0b010, 3); // large metadata 0, large codec config 1, large item data 0
        put(cfg.len(), 12);
        put(main.len() - 1, 15);
        put(alpha.len(), 15);
        put(0, 12); // the alpha item uses the main configuration
        put(0, 1); // Exif not compressed
        put(exif.len() - 1, 10);
        let mut payload: Vec<u8> = bits
            .chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u8, |b, (i, &x)| if x { b | (0x80 >> i) } else { b })
            })
            .collect();
        payload.extend_from_slice(&[&cfg[..], alpha, main, exif].concat());
        [bx(b"ftyp", b"mif3heic"), bx(b"mini", &payload)].concat()
    }

    #[test]
    fn mini_box_is_decoded_like_meta() {
        let file = mini_file();
        let cancel = AtomicBool::new(false);
        let iso = isobmff::Limits::default();
        let heif = isobmff::parse(&file, &iso, &cancel).expect("hand-built mini file parses");
        let img = heif.primary_image().unwrap();
        assert!(matches!(
            img.kind,
            ImageKind::Coded {
                codec: Codec::Hevc,
                config: CodecConfig::Hevc(_)
            }
        ));
        assert_eq!((img.width, img.height, img.alpha), (64, 48, Some(2)));
        // Orientation 5 as libheif maps it, returned raw.
        assert_eq!(
            transforms_of(&heif, &img).unwrap(),
            [Transform::Rotate(270), Transform::Mirror(Axis::LeftRight)]
        );
        // The Exif chunk stays readable as stored (a mini box has no exif_tiff_header_offset).
        let md = heif.metadata(&file, 1, &iso, &cancel).unwrap();
        assert_eq!(md.exif.as_deref(), Some(&b"II*\0EXIF"[..]));
        // The item bytes reach the decoder (the junk is one NAL length past the data), for the
        // primary and for the alpha item, which shares its configuration.
        let (o, l) = (Options::default(), Limits::default());
        let e = decode_heic(&file, &o, &l, &cancel).err();
        assert_eq!(e, Some(Error::InvalidData("NAL length beyond the data")));
        assert_eq!(e, decode_heic_item(&heif, &file, 2, &o, &l, &cancel).err());
    }
}
