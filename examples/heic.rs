//! HEIC parity oracle tool (docs/design.md, "HEIC glue").
//!
//! `heic <in-dir> <out-dir>` decodes the primary image of every `.heic`, `.heif` and `.hif`
//! file (any case) under in-dir and writes `<name>.yuv` (the planes of the image at native depth:
//! 1 byte per sample at depth 8, else 2 bytes little-endian; Y, Cb, Cr; mono is Y only),
//! `<name>.alpha` (the alpha plane, same sample format; only when there is one) and `<name>.info`
//! (`W H chroma depthY depthC alpha`, one line), or `<name>.err` (a token). `<name>` is the path
//! relative to in-dir, without the extension, with `/` replaced by `__`. The image is not
//! rotated, mirrored or cropped by its transforms.

use hevc::{ChromaFormat, Error, Limits, Options, Plane, Samples};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

const EXTENSIONS: [&str; 3] = ["heic", "heif", "hif"];

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(root, &p, out);
        } else if p.extension().is_some_and(|e| {
            EXTENSIONS
                .iter()
                .any(|x| e.eq_ignore_ascii_case(std::ffi::OsStr::new(x)))
        }) {
            let rel = p.strip_prefix(root).unwrap().with_extension("");
            let parts: Vec<_> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            out.push((parts.join("__"), p));
        }
    }
}

/// The text of `Error::InvalidData` for an isobmff error is its `class()`.
fn is_container(s: &str) -> bool {
    matches!(
        s,
        "unknown-format"
            | "no-meta"
            | "truncated"
            | "malformed"
            | "unsupported"
            | "no-such-item"
            | "out-of-memory"
            | "io"
    )
}

fn token(e: &Error) -> &'static str {
    match e {
        Error::InvalidData(s) if s.starts_with("limit-") => "limit",
        Error::InvalidData(s) if is_container(s) => "container",
        Error::InvalidData(_) => "invalid",
        Error::Unsupported(_) => "unsupported",
        Error::InterSlice => "inter",
        Error::NoPicture => "nopicture",
        Error::Cancelled => "cancelled",
        Error::WidthLimit
        | Error::HeightLimit
        | Error::PixelLimit
        | Error::PictureLimit
        | Error::AllocLimit
        | Error::OutOfMemory => "limit",
        _ => "invalid",
    }
}

fn chroma(c: ChromaFormat) -> u32 {
    match c {
        ChromaFormat::Mono => 400,
        ChromaFormat::Yuv420 => 420,
    }
}

fn push(out: &mut Vec<u8>, p: &Plane) {
    match &p.samples {
        Samples::U8(v) => out.extend_from_slice(v),
        Samples::U16(v) => out.extend(v.iter().flat_map(|s| s.to_le_bytes())),
    }
}

fn decode_one(name: &str, data: &[u8], outdir: &Path) {
    let cancel = AtomicBool::new(false);
    // Diagnosis only (tests/parity/run.sh): HEIC_FILTERS=none|nodeblock|nosao turns in-loop
    // filters off, to find which filter an oracle that differs left out.
    let mut opts = Options::default();
    match std::env::var("HEIC_FILTERS").as_deref() {
        Ok("none") => (opts.deblocking, opts.sao) = (false, false),
        Ok("nodeblock") => opts.deblocking = false,
        Ok("nosao") => opts.sao = false,
        _ => {}
    }
    // Bench only (tests/parity/bench.sh): HEIC_THREADS=N sets `max_threads`.
    let mut limits = Limits::default();
    if let Some(n) = std::env::var("HEIC_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        limits.max_threads = n;
    }
    let r = hevc::decode_heic(data, &opts, &limits, &cancel);
    let h = match r {
        Ok(h) => h,
        Err(e) => {
            std::fs::write(outdir.join(format!("{name}.err")), token(&e)).unwrap();
            std::fs::write(outdir.join(format!("{name}.msg")), e.to_string()).unwrap();
            return;
        }
    };
    // decode_heic_item on the parse of the same file must give the primary image of decode_heic.
    let iso = isobmff::Limits::default();
    let heif = isobmff::parse(data, &iso, &cancel).unwrap();
    let id = heif.primary_image().unwrap().id;
    let item = hevc::decode_heic_item(&heif, data, id, &opts, &Limits::default(), &cancel);
    assert!(
        item.is_ok_and(|p| p == h.image),
        "{name}: decode_heic_item differs"
    );
    let mut yuv = Vec::new();
    for p in &h.image.planes {
        push(&mut yuv, p);
    }
    std::fs::write(outdir.join(format!("{name}.yuv")), yuv).unwrap();
    if let Some(a) = &h.alpha {
        let mut bytes = Vec::new();
        push(&mut bytes, a);
        std::fs::write(outdir.join(format!("{name}.alpha")), bytes).unwrap();
    }
    let i = &h.image.info;
    let line = format!(
        "{} {} {} {} {} {}",
        i.width,
        i.height,
        chroma(i.chroma_format),
        i.bit_depth_luma,
        i.bit_depth_chroma,
        u8::from(h.alpha.is_some())
    );
    std::fs::write(outdir.join(format!("{name}.info")), line).unwrap();
}

/// Diagnosis only: when HEIC_ANNEXB is set, the primary image's coded item (one hvc1 item, not
/// a grid) is written as an Annex B stream `<name>.265` (the hvcC NAL units, then the item's NAL
/// units, each after a 4-byte start code), so dec265 can decode the same bytes.
fn write_annexb(name: &str, data: &[u8], outdir: &Path) {
    let limits = isobmff::Limits::default();
    let cancel = AtomicBool::new(false);
    let Ok(heif) = isobmff::parse(data, &limits, &cancel) else {
        return;
    };
    let Ok(img) = heif.primary_image() else {
        return;
    };
    let isobmff::ImageKind::Coded {
        config: isobmff::CodecConfig::Hevc(hv),
        ..
    } = &img.kind
    else {
        return;
    };
    let (Some(cfg), Ok(item)) = (
        isobmff::Heif::slice(data, hv.raw),
        heif.item_bytes(data, img.id, &limits, &cancel),
    ) else {
        return;
    };
    let mut out = Vec::new();
    // hvcC: 22 header bytes, numOfArrays, then per array 1 byte, u16 count, (u16 len, NAL)*.
    let Some((&arrays, mut d)) = cfg.get(22..).and_then(|r| r.split_first()) else {
        return;
    };
    for _ in 0..arrays {
        let Some(n) = d.get(1..3).map(|b| u16::from_be_bytes([b[0], b[1]])) else {
            return;
        };
        d = &d[3..];
        for _ in 0..n {
            let Some(len) = d
                .get(..2)
                .map(|b| usize::from(u16::from_be_bytes([b[0], b[1]])))
            else {
                return;
            };
            let Some(nal) = d.get(2..2 + len) else { return };
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
            d = &d[2 + len..];
        }
    }
    let ls = usize::from(hv.length_size_minus_one) + 1;
    let mut s: &[u8] = &item;
    while s.len() >= ls {
        let len = s[..ls]
            .iter()
            .fold(0usize, |a, &b| (a << 8) | usize::from(b));
        let Some(nal) = s.get(ls..ls + len) else {
            return;
        };
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        s = &s[ls + len..];
    }
    std::fs::write(outdir.join(format!("{name}.265")), out).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [indir, outdir] = &args[..] else {
        eprintln!("usage: heic <in-dir> <out-dir>");
        std::process::exit(2);
    };
    let (indir, outdir) = (Path::new(indir), Path::new(outdir));
    std::fs::create_dir_all(outdir).unwrap();
    let mut files = Vec::new();
    walk(indir, indir, &mut files);
    for (name, path) in files {
        let data = std::fs::read(&path).unwrap();
        decode_one(&name, &data, outdir);
        if std::env::var_os("HEIC_ANNEXB").is_some() {
            write_annexb(&name, &data, outdir);
        }
    }
}
