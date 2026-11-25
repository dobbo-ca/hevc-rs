# hevc-rs

HEVC (H.265) decoding for intra pictures under strict resource limits, in Rust. It decodes Main, Main 10 and Main Still Picture streams (4:2:0 and 4:0:0, 8 to 12 bits per component) from an Annex B byte stream or from HEIF/MP4 samples plus an `hvcC` record, bit-exact with the spec and with libde265 and ffmpeg on the JCT-VC conformance intra prefixes in CI (docs/parity.md). Every I picture is decoded (CABAC, intra prediction, residual, deblocking, SAO); P and B slices are rejected with `Error::InterSlice`. It is a new crate written from the H.265 spec, not a port. Pure Rust, one dependency (isobmff-rs, for HEIC), `unsafe` only in the SIMD dispatch and load/store helpers (docs/safety.md). Apache-2.0; see `NOTICE`.

Dependency: `hevc = { git = "https://github.com/dobbo-ca/hevc-rs", rev = "55ea54ab68623d50c10362b537dcb16b11ac74e2" }`

**PATENT CAUTION (owner):** HEVC is covered by patent pools. Low risk for a private app; the owner must review before Drive goes public.

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

- `max_alloc` is the enforced peak working memory of one call, all threads together, including the nested isobmff-rs parse: allocations are charged before they are made and released on drop. 64 KiB is booked up front for small fixed allocations, so a `max_alloc` below that is `Error::AllocLimit` at once (docs/limits.md).
- `max_threads = 1` spawns no thread. With more, HEIC grid tiles, the pictures of a call that holds several, and the deblocking and SAO bands of a big picture run on up to `max_threads - 1` scoped threads; the output and the errors do not depend on the value. No thread pool: a shared pool gave no clear gain. Thread stacks are not charged to `max_alloc` (docs/threading.md).
- Metadata (SEI messages, VUI) is kept raw and charged to `max_alloc`; `Options { metadata: false, .. }` drops it. `Options::check_hash` (default false) checks the decoded picture hash SEI (`Picture::hash_ok`).
- Cancellation: every decode takes `&AtomicBool`, read before each NAL unit and each CTU, per 16 rows of vertical edges and per row of horizontal edges in deblocking, per CTB row in SAO, and per row in the picture hash and the crop. When set, the call returns `Error::Cancelled`.

## HEIC

`decode_heic` decodes the primary image of a HEIC file (`meta` or `mini` box; one `hvc1` item, a grid, `iden` or `iovl`, plus its alpha plane); `decode_heic_item` decodes any image item of a parsed `isobmff::Heif` (parse with the re-exported `hevc::isobmff`, the same rev). `irot`, `imir` and `clap` come back raw in `transforms` unless `Options { apply_transforms: true, .. }` applies them (docs/design.md, "HEIC glue" and "Transforms"). Exif and XMP: `isobmff::Heif::metadata`.

## Docs

[design](docs/design.md) (the build spec, HEIC), [limits](docs/limits.md) (charging, errors), [threading](docs/threading.md), [safety](docs/safety.md) (the `unsafe` inventory, hostile input), [dependencies](docs/dependencies.md) (isobmff-rs), [parity](docs/parity.md) (oracles, corpus, gates), [fuzzing](docs/fuzzing.md), [benchmarks](docs/benchmarks.md).
