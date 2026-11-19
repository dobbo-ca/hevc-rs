# Dependencies

None. `[dependencies]` is empty, and so are the build dependencies. No C code.

libde265 and ffmpeg are used only by the CI parity oracle, built or installed in the CI image
(see [parity.md](parity.md)). They are never linked into the crate and their code and tables are
not copied: both are LGPL, so the decoder is written from the H.265 spec.

`libfuzzer-sys` is used only by the separate `tests/fuzz/` crate.
