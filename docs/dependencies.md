# Dependencies

Two.

- `isobmff` ([isobmff-rs](https://github.com/dobbo-ca/isobmff-rs), private, same owner, pure
  Rust, no `unsafe`), pinned by git rev in `Cargo.toml` (`06d94e6`, which reads the `mini` box); it
  parses the HEIC container for `decode_heic` and decodes no pixels. CI fetches it with a
  read-only GitHub App token (see `docs/conventions.md` in rust-ports, "Private git
  dependencies").
- `rayon-core` (crates.io, MIT or Apache-2.0, pure Rust; it pulls `crossbeam-deque`,
  `crossbeam-epoch` and `crossbeam-utils`, same licences): the lazy thread pool that runs the
  deblocking and SAO bands ([threading.md](threading.md#the-shared-pool)). Only the pool is used
  (`ThreadPool`, `in_place_scope`, `spawn`), not rayon's parallel iterators. Owner decision
  2026-11-24: a pool crate is allowed when a CI bench shows it is faster than
  `std::thread::scope`. Bench: see below.

The build dependencies are empty. No C code.

libde265 and ffmpeg are used only by the CI parity oracle, built or installed in the CI image
(see [parity.md](parity.md)). They are never linked into the crate and their code and tables are
not copied: both are LGPL, so the decoder is written from the H.265 spec.

`libfuzzer-sys` is used only by the separate `tests/fuzz/` crate.
