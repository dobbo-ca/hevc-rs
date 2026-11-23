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
    let r = hevc::decode_heic(data, &opts, &Limits::default(), &cancel);
    let h = match r {
        Ok(h) => h,
        Err(e) => {
            std::fs::write(outdir.join(format!("{name}.err")), token(&e)).unwrap();
            return;
        }
    };
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
        decode_one(&name, &std::fs::read(&path).unwrap(), outdir);
    }
}
