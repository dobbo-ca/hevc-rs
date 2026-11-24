//! Parity oracle (docs/design.md, "Oracles and parity").
//!
//! `oracle [--threads N] [--no-filters] <in-dir> <out-dir>` decodes every `.bit`, `.bin`,
//! `.hevc` and `.265` file (Annex B) under in-dir and writes `<name>.yuv`, `<name>.info` and
//! `<name>.sei` (informational: per picture, the SEI message count and `type:len` list), or
//! `<name>.err`. `oracle probe <in-dir> <out-dir>` writes `<name>.probe`. `oracle prefix
//! <in-dir> <out-dir>` writes `<name>.bit`, the NAL units before the first P or B picture.
//! `<name>` is the path relative to in-dir, without the extension, with `/` replaced by `__`.

use hevc::{ChromaFormat, Error, Limits, Options, Samples};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

const EXTENSIONS: [&str; 4] = ["bit", "bin", "hevc", "265"];

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(root, &p, out);
        } else if p
            .extension()
            .is_some_and(|e| EXTENSIONS.iter().any(|x| e == *x))
        {
            let rel = p.strip_prefix(root).unwrap().with_extension("");
            let parts: Vec<_> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            out.push((parts.join("__"), p));
        }
    }
}

fn token(e: &Error) -> &'static str {
    match e {
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

fn usage() -> ! {
    eprintln!(
        "usage: oracle [--threads N] [--no-filters] <in-dir> <out-dir>\n       oracle probe <in-dir> <out-dir>\n       oracle prefix <in-dir> <out-dir>"
    );
    std::process::exit(2);
}

fn decode_one(name: &str, data: &[u8], opts: &Options, limits: &Limits, outdir: &Path) {
    let cancel = AtomicBool::new(false);
    let mut pics = Vec::new();
    let r = hevc::decode_into(data, opts, limits, &cancel, &mut pics);
    // A P or B slice after at least one picture is the end of the intra prefix, not an error.
    let err = match r {
        Err(Error::InterSlice) if !pics.is_empty() => None,
        Err(e) => Some(e),
        Ok(()) => None,
    };
    if let Some(e) = err {
        std::fs::write(outdir.join(format!("{name}.err")), token(&e)).unwrap();
        return;
    }
    let mut yuv = Vec::new();
    let mut info = String::new();
    let mut sei = String::new();
    for p in &pics {
        for plane in &p.planes {
            match &plane.samples {
                Samples::U8(v) => yuv.extend_from_slice(v),
                Samples::U16(v) => yuv.extend(v.iter().flat_map(|s| s.to_le_bytes())),
            }
        }
        let hash = match p.hash_ok {
            Some(true) => "ok",
            Some(false) => "bad",
            None => "none",
        };
        write!(sei, "{}", p.sei.len()).unwrap();
        for m in &p.sei {
            write!(sei, " {}:{}", m.payload_type, m.payload.len()).unwrap();
        }
        sei.push('\n');
        let i = &p.info;
        writeln!(
            info,
            "{} {} {} {} {} {} {hash}",
            p.poc,
            i.width,
            i.height,
            chroma(i.chroma_format),
            i.bit_depth_luma,
            i.bit_depth_chroma
        )
        .unwrap();
    }
    std::fs::write(outdir.join(format!("{name}.yuv")), yuv).unwrap();
    std::fs::write(outdir.join(format!("{name}.info")), info).unwrap();
    std::fs::write(outdir.join(format!("{name}.sei")), sei).unwrap();
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.first().map(String::as_str) {
        Some(m @ ("probe" | "prefix")) => {
            let m = m.to_owned();
            args.remove(0);
            m
        }
        _ => String::new(),
    };
    let mut limits = Limits::default();
    // The parity gate G4 reads the hash column.
    let mut opts = Options {
        check_hash: true,
        ..Options::default()
    };
    let mut pos = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => {
                limits.max_threads = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--no-filters" => {
                opts.deblocking = false;
                opts.sao = false;
            }
            _ => pos.push(a),
        }
    }
    let [indir, outdir] = pos[..] else { usage() };
    let (indir, outdir) = (Path::new(indir), Path::new(outdir));
    std::fs::create_dir_all(outdir).unwrap();
    let mut files = Vec::new();
    walk(indir, indir, &mut files);
    for (name, path) in files {
        let data = std::fs::read(&path).unwrap();
        match mode.as_str() {
            "probe" => match hevc::probe(&data, &limits) {
                Ok(i) => {
                    let line = format!(
                        "{} {} {} {} {}",
                        i.width,
                        i.height,
                        chroma(i.chroma_format),
                        i.bit_depth_luma,
                        i.bit_depth_chroma
                    );
                    std::fs::write(outdir.join(format!("{name}.probe")), line).unwrap();
                }
                Err(e) => std::fs::write(outdir.join(format!("{name}.err")), token(&e)).unwrap(),
            },
            "prefix" => match hevc::intra_prefix(&data, &limits) {
                Ok(bytes) => std::fs::write(outdir.join(format!("{name}.bit")), bytes).unwrap(),
                // No file: the C oracles have nothing to decode.
                Err(e) => eprintln!("{name}: no prefix ({e})"),
            },
            _ => decode_one(&name, &data, &opts, &limits, outdir),
        }
    }
}
