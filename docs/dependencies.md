# Dependencies

One: `isobmff` ([isobmff-rs](https://github.com/dobbo-ca/isobmff-rs), private, same owner, pure
Rust, no `unsafe`), pinned by git rev in `Cargo.toml` (`06d94e6`, which reads the `mini` box); it parses the HEIC container for
`decode_heic` and decodes no pixels. CI fetches it with a read-only GitHub App token (see
`docs/conventions.md` in rust-ports, "Private git dependencies"). The build dependencies are
empty. No C code.

libde265 and ffmpeg are used only by the CI parity oracle, built or installed in the CI image
(see [parity.md](parity.md)). They are never linked into the crate and their code and tables are
not copied: both are LGPL, so the decoder is written from the H.265 spec.

`libfuzzer-sys` is used only by the separate `tests/fuzz/` crate.
