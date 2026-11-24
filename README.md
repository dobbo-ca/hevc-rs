# hevc-rs

HEVC (H.265) decoding for intra pictures under strict resource limits, in Rust. It decodes Main, Main 10 and Main Still Picture streams (4:2:0 and 4:0:0, 8 to 12 bits per component) from an Annex B byte stream or from HEIF/MP4 samples plus an `hvcC` record, bit-exact with the spec; P and B slices are rejected with `Error::InterSlice`. It is a new crate written from the H.265 spec, not a port. Pure Rust, one dependency (isobmff-rs, for HEIC), `unsafe` only in the SIMD dispatch and load/store helpers (docs/safety.md). Apache-2.0; see `NOTICE`.

Dependency: `hevc = { git = "https://github.com/dobbo-ca/hevc-rs", rev = "5b860a0791fd5295aa2191b2c20b89d72d0cb9bb" }`

**PATENT CAUTION (owner):** HEVC is covered by patent pools. Low risk for a private app; the owner must review before Drive goes public.

Status: intra decoding works. Every I picture is decoded (CABAC, intra prediction, residual, deblocking, SAO), checked against its decoded picture hash SEI when present (with `Options { check_hash: true, .. }`; off by default), and bit-exact with libde265 and ffmpeg on the JCT-VC HEVC_v1 intra prefixes in CI. P and B slices are rejected with `Error::InterSlice`.

## Quick start

```rust
use std::sync::atomic::AtomicBool;
use hevc::{Limits, Options};

let data = std::fs::read("clip.265")?;
let limits = Limits {
    max_threads: 2,
    max_alloc: 256 << 20,
    ..Limits::default()
};
let cancel = AtomicBool::new(false);
let pictures = hevc::decode(&data, &Options::default(), &limits, &cancel)?;
println!("{} pictures", pictures.len());
```

`decode_into` keeps the pictures output before an error. `decode_hvcc` takes the `hvcC` payload and length-prefixed samples. `probe` reads the first SPS without decoding.

## Resource limits

There is no unlimited entry point. Every call takes a `Limits`.

| Field | Default | Unit | Over it |
|---|---|---|---|
| `max_alloc` | 1 GiB | bytes, peak of one call, all threads together | `Error::AllocLimit` (`Error::OutOfMemory` if the allocator fails) |
| `max_threads` | `available_parallelism()` | threads, calling thread included | no error; 0 means 1 |
| `max_width` | 16384 | luma samples (`pic_width_in_luma_samples`) | `Error::WidthLimit` |
| `max_height` | 16384 | luma rows | `Error::HeightLimit` |
| `max_pixels` | 268 435 456 (`1 << 28`) | coded width x height | `Error::PixelLimit` |
| `max_pictures` | 4096 | decoded pictures per call | `Error::PictureLimit` |
| `max_metadata` | 1 MiB | bytes of SEI messages kept per picture (payload plus the message struct) | no error; the message is dropped |

- `max_alloc` is the enforced peak working memory of one call, all threads together, including the nested isobmff-rs parse (an eighth of `max_alloc` each for the parse and the item bytes): allocations are charged before they are made and released on drop. The stored parameter sets, the output pictures and the planes are charged one by one; 64 KiB is booked up front for the small fixed allocations, so a `max_alloc` below that is `Error::AllocLimit` at once.
- Metadata (SEI messages, VUI) is kept by default and exposed raw; `Options { metadata: false, .. }` drops it. Kept SEI bytes are charged to `max_alloc` like the planes.
- `max_threads = 1` spawns no thread. With more, the tiles of a HEIC grid and the pictures of a call that holds several are decoded in parallel, and deblocking and SAO of a big picture run in parallel bands of CTB rows (one band per 64 Ki luma samples), all on up to `max_threads - 1` extra threads (a call with one picture, or one picture and a P slice, spawns no picture worker; a worker starts only when a job finds none free); the output and the errors do not depend on the value. No thread pool: workers are scoped per call (docs/threading.md, Known deviations). See `docs/threading.md`.
- `Options::check_hash` (default false): the decoded picture hash SEI is checked (`Picture::hash_ok` is `Some`) only when it is true; otherwise `hash_ok` is `None` and no MD5, CRC or checksum is computed.
- Cancellation: every decode takes `&AtomicBool`, read before each NAL unit and each CTU, and per row in deblocking, SAO, the picture hash and the crop. When set, the call returns `Error::Cancelled`.

## HEIC

`decode_heic` decodes the primary image of a HEIC file (one `hvc1` item or a grid of tiles, plus its alpha plane) on top of isobmff-rs. `irot`, `imir` and `clap` come back raw in `transforms`, not applied.
`decode_heic_item` decodes any image item of an already parsed `isobmff::Heif` (a coded item or a grid, no alpha, transforms not applied) without parsing again.

```rust
let file = std::fs::read("photo.heic")?;
let heic = hevc::decode_heic(&file, &Options::default(), &limits, &cancel)?;
println!("{}x{}, {} transforms", heic.image.info.width, heic.image.info.height, heic.transforms.len());
```

## Docs

- [docs/design.md](docs/design.md): the build spec
- [docs/limits.md](docs/limits.md): limits, charging, errors
- [docs/threading.md](docs/threading.md): grid tiles and pictures in parallel, the error and memory rules
- [docs/safety.md](docs/safety.md): the `unsafe` inventory (SIMD dispatch and helpers), kernels removed, hostile input
- [docs/dependencies.md](docs/dependencies.md): isobmff-rs
- [docs/parity.md](docs/parity.md): oracles, corpus, gates
- [docs/fuzzing.md](docs/fuzzing.md): fuzz targets
- [docs/benchmarks.md](docs/benchmarks.md): benchmarks
