# Dependencies

One: `isobmff` ([isobmff-rs](https://github.com/dobbo-ca/isobmff-rs), same owner, pure
Rust, no `unsafe`), pinned by git rev in `Cargo.toml` (`c4f748b`, which has the shared `apply_transforms` helper); it parses the HEIC container for
`decode_heic` and decodes no pixels; with `Options::apply_transforms` its `apply_transforms` crops, rotates and mirrors the decoded planes (docs/design.md, "Transforms"). CI fetches it with a read-only GitHub App token (see
`.github/ci/Dockerfile`). The build dependencies are
empty. No C code.

libde265 and ffmpeg are used only by the CI parity oracle, built or installed in the CI image
(see [parity.md](parity.md)). They are never linked into the crate and their code and tables are
not copied: both are LGPL, so the decoder is written from the H.265 spec.

`libfuzzer-sys` is used only by the separate `tests/fuzz/` crate, which also depends on isobmff-rs at the same rev (the `heic` target parses with `isobmff::parse` before `decode_heic_item`).
