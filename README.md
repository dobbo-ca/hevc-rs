# hevc-rs

HEVC (H.265) decoding for intra pictures under strict resource limits, in Rust. It decodes Main, Main 10 and Main Still Picture streams (4:2:0 and 4:0:0, 8 to 12 bits per component) from an Annex B byte stream or from HEIF/MP4 samples plus an `hvcC` record, bit-exact with the spec; P and B slices are rejected with `Error::InterSlice`. It is a new crate written from the H.265 spec, not a port. Pure Rust, no dependency, no `unsafe`. Apache-2.0; see `NOTICE`.

Dependency: `hevc = { git = "https://github.com/dobbo-ca/hevc-rs", rev = "<sha>" }`

**PATENT CAUTION (owner):** HEVC is covered by patent pools. Low risk for a private app; the owner must review before Drive goes public.

Status: intra decoding works. Every I picture is decoded (CABAC, intra prediction, residual, deblocking, SAO), checked against its decoded picture hash SEI when present, and bit-exact with libde265 and ffmpeg on the JCT-VC HEVC_v1 intra prefixes in CI. P and B slices are rejected with `Error::InterSlice`.

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

- `max_alloc` is the enforced peak working memory of one call, all threads together: allocations are charged before they are made and released on drop. The stored parameter sets, the output pictures and the planes are charged one by one; 64 KiB is booked up front for the small fixed allocations, so a `max_alloc` below that is `Error::AllocLimit` at once.
- Metadata (SEI messages, VUI) is kept by default and exposed raw; `Options { metadata: false, .. }` drops it. Kept SEI bytes are charged to `max_alloc` like the planes.
- `max_threads = 1` spawns no thread. v1 is serial, so no value spawns one.
- Cancellation: every decode takes `&AtomicBool`, read before each NAL unit and each CTU. When set, the call returns `Error::Cancelled`.

## Docs

- [docs/design.md](docs/design.md): the build spec
- [docs/limits.md](docs/limits.md): limits, charging, errors
- [docs/threading.md](docs/threading.md): serial in v1
- [docs/safety.md](docs/safety.md): no `unsafe`, hostile input
- [docs/dependencies.md](docs/dependencies.md): none
- [docs/parity.md](docs/parity.md): oracles, corpus, gates
- [docs/fuzzing.md](docs/fuzzing.md): fuzz targets
- [docs/benchmarks.md](docs/benchmarks.md): benchmarks
