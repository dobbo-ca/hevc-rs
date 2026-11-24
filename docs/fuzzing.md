# Fuzzing

`tests/fuzz/` is a cargo-fuzz crate with its own `Cargo.lock`. Run with `--fuzz-dir tests/fuzz`.

| Target | What it calls |
|---|---|
| `annexb` | `intra_prefix`: the Annex B splitter, the parameter set store and slice header parsing, with a size bound on its output |
| `hvcc` | `probe_hvcc` and `decode_hvcc`; the first byte splits the input into the `hvcC` record and the samples |
| `params` | `probe` on one SPS and on one PPS NAL unit built from the input (start code and NAL header in front) |
| `probe` | `probe` on arbitrary bytes; an Ok `Info` must stay inside the limits |
| `decode` | `decode` with small `Limits`; the planes of every picture must match its `Info`. The low 4 bits of the last byte switch deblocking, SAO, metadata and the picture hash check off; the result and pictures with `max_threads` 3 must equal those with 1 |
| `sao_tiers` | `__fuzz_sao::tiers`: band/edge offset rows built from the input, every SIMD tier vs scalar, bit-exact |
| `heic` | `decode_heic` with small `Limits` (grids, overlays, identity items); an Ok image's luma plane must match its `Info` size, and the result with `max_threads` 4 must equal the one with 1 |
| `transform_tiers` | `__fuzz_transform::tiers_agree`: every SIMD tier equals scalar on one block of at most 32x32 built from the bytes |
| `intra_tiers` | `__fuzz_intra::run`: one intra block (size 4..32, mode 0..34, bit depth 8..16, luma/strong/raw flags) built from the bytes, every SIMD tier vs scalar, bit-exact |
| `deblock_tiers` | `__fuzz_deblock::tiers`: one batch of 4 deblocking edge segments (luma or chroma, vertical or horizontal edge, bit depth 8 to 10, beta, tC and kept sides) and a 24x20 plane built from the input, every SIMD tier vs scalar, bit-exact |

All targets run with `-rss_limit_mb=2048 -malloc_limit_mb=512`, so an allocation outside the
limits is a crash. The malloc limit needs the sanitizer's allocator hooks, so it applies to the
x86_64 jobs only; the `fuzz-arm64` jobs (no sanitizer, under qemu) have the RSS limit only.

Seeds are made in the image build (`.github/ci/Dockerfile`, stage `fuzz`), one directory per
target, and are not kept in git. `annexb`, `probe` and `decode` get a hand-written VPS, the start
of an SPS and an AUD, plus the first 64 KiB of the first 20 conformance streams from the data
cache. `heic` gets the first 20 files under 64 KiB of the HEIC corpus and its `mini` box files (C71 and both `lightning_mini.heif`). `hvcc` gets one minimal `hvcC` record with a sample, `params` the SPS bytes alone. The

`.github/workflows/fuzz.yml` runs weekly on Tuesday at 12:23 UTC (the slot is not shared with
another crate) and on `workflow_dispatch`, never on push or pull request. The dispatch input
`max_total_time` is at most 3000 s: the job times out at 60 minutes with the image build.

The `fuzz-arm64` job builds `transform_tiers`, `intra_tiers`, `sao_tiers` and `deblock_tiers` for aarch64 without a sanitizer (ASan does not run under qemu-user) and runs them under `qemu-aarch64`, so every NEON tier is compared with scalar. It has the same time input and `max-parallel: 2`.

## Runs

| Date | Ref | Targets x seconds | Crashes |
|---|---|---|---|
| 2026-11-23 | main 92cb413 | annexb, hvcc, params, probe, decode x 900 s | 0 |
| 2026-11-23 | main 6dbe2a7 | heic x 900 s | 0 |

Fuzz hours on main: 1.5 (6 targets x 900 s). `decode` runs slowly (about 5 exec/s: its seeds are whole
pictures), the parsers run 2000-5600 exec/s.
