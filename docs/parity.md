# Parity

All of this runs in CI only; nothing is built or run on a developer machine.

## Oracles

- `examples/oracle.rs`: `oracle [--threads N] [--no-filters] <in-dir> <out-dir>` decodes every
  `*.bit`, `*.bin`, `*.hevc` and `*.265` file (Annex B) and writes `<name>.yuv` (all output
  pictures, planes Y, Cb, Cr, cropped, 1 byte per sample at bit depth 8 else 2 bytes little-endian)
  and `<name>.info` (one line per picture: `poc W H chroma depthY depthC hash`, hash is
  ok/bad/none), or `<name>.err` with a token: `invalid`, `unsupported`, `inter`, `nopicture`,
  `limit`, `cancelled`. `InterSlice` after at least one picture is not an error.
- `oracle probe <in> <out>` writes `<name>.probe`: `W H chroma depthY depthC` (chroma 400 or 420).
- `oracle prefix <in> <out>` writes `<name>.bit`, the Annex B NAL units before the first picture
  with a P or B slice (`hevc::intra_prefix`); no file when no picture is left.
- libde265 v1.0.15 `dec265` (built from the tag) and Debian bookworm ffmpeg (`-f rawvideo`, native
  pix_fmt, `-fps_mode passthrough`) decode the same prefix files.
- `examples/heic.rs`: `heic <in-dir> <out-dir>` decodes the primary image of every `*.heic`,
  `*.heif` and `*.hif` file (`hevc::decode_heic`, default options and limits) and writes
  `<name>.yuv` (planes Y, Cb, Cr at native depth: 1 byte per sample at depth 8, else 2 bytes
  little-endian; monochrome is Y only), `<name>.alpha` (the alpha plane, same sample format; only
  when there is one) and `<name>.info` (`W H chroma depthY depthC alpha`, chroma 400 or 420, alpha
  0 or 1), or `<name>.err` with a token (`unsupported`, `invalid`, `container`, `limit`, ...). The
  image is not rotated, mirrored or cropped by its transforms.
- libheif v1.23.7 (`tests/parity/c/heic_oracle.c` linked statically to the tag, commit
  `12a259a5`, with libde265 v1.0.15 as the only codec) is the HEIC oracle `heic-oracle`:
  `heif_decode_image` of the primary image with `heif_colorspace_undefined`,
  `heif_chroma_undefined` and `ignore_transformations = 1`, so the native YCbCr layout and depth
  are kept, in the same output files. An RGB or interleaved result is `<name>.err` = `convert`;
  other failures are `decode`. For a monochrome image libheif has no chroma plane, so its depthC
  is written as depthY. libheif is LGPL: it is a CI-only binary, never linked into the crate.

## Gates

`tests/parity/run.sh full` writes `summary.md`; a gate passes on an empty diff. G2 and G4 are
"soft": they fail only when `PARITY_DECODE=1`, which is the default of the Dockerfile.

1. G1 `probe` against `ffprobe`: `W H chroma depthY` of every stream.
2. G2 every stream that hevc-rs decodes is confirmed: its `.yuv` equals dec265 output, or equals
   ffmpeg output (ffmpeg may drop pictures at a size change; every picture it writes must equal
   the matching one), or every picture has a decoded picture hash SEI that matched. A stream
   where an oracle differs while another check confirms hevc-rs is listed in
   `disagreements.txt`, not a failure.
3. G3 the per-stream dec265 and ffmpeg comparisons (`d-dec265.diff`, `d-ffmpeg.diff`,
   `streams.tsv`) that feed G2; reported, not a separate gate.
4. G4 decoded picture hash: no `.info` line is `bad`.
5. G5 `--threads 1` equals the default.
8. G8 (run after G5) `HEVC_SIMD=scalar` and `HEVC_SIMD=avx2` equal the default (best) tier,
   AVX-512 on the CI hosts. `parity-arm64` does the same for scalar against NEON.
6. G6 diagnostic, report only: `--no-filters` against dec265 with deblocking and SAO disabled.
7. G7 HEIC: hevc-rs against libheif on the HEIC corpus. Not soft: it fails whatever
   `PARITY_DECODE` is. For every file both decode, `.yuv` and `.alpha` are byte-equal and `.info`
   is equal (for a monochrome image the depthC field is not compared). Files only one side
   decodes:
   - libheif decodes, hevc-rs says `unsupported` (for example 4:2:2, 4:4:4, a protected item or a
     tool outside the scope) or `limit` (libheif runs with its security limits off, hevc-rs with
     `Limits::default()`): listed in `heic-disagreements.txt`, explained, not a failure;
   - hevc-rs decodes, libheif fails: listed in `heic-disagreements.txt`, not a failure;
   - libheif decodes and hevc-rs fails with any other token (`invalid`, `container`,
     `inter`, ...), a file with an `.alpha` on one side only, or a file with no output on a side:
     a line of `d-heic.diff`, G7 fails.

   An `iovl` primary decodes in the inputs' YCbCr, while libheif returns RGB for it (the oracle
   token is `convert`, a libheif failure): such a file is listed in `heic-disagreements.txt`, not
   compared. An `iden` primary is compared like any other (libheif runs with
   `ignore_transformations`).

   Coverage: G7 also fails when fewer than `PARITY_MIN_HEIC` (default 20) files are decoded by
   both, so a libheif without a HEVC decoder cannot pass with nothing compared (the oracle image
   also checks that libde265 is linked, and `heic-oracle` exits 2 without a HEVC decoder). Two
   corpus files with the same name once flattened and stripped of the extension (`a.heic` and
   `a.hif`) fail the run, since their outputs would overwrite each other.

   A file both reject is only counted. `heic-errors.txt` has the error token of every rejected
   file on each side. The summary has the counts: files, decoded by each, decoded by both, equal,
   alpha planes compared, rejected by both, explained.

## Counts

From the CI runs of 2026-11-23 with every gate green (main 6dbe2a7 and the PR runs before it):

| Count | Value |
|---|---|
| bitstreams (148 HEVC_v1 + 9 RExt zips) | 157 inputs, 152 intra prefixes |
| hevc-rs decoded / rejected (unsupported RExt tools) | 152 / 4 (`Bitdepth_A/B`, `IPCM_A/B_RExt`) |
| pictures decoded | 307 |
| streams equal to dec265 / ffmpeg | 150 / 149 (+1 by picture subset: `VPSSPSPPS_A`) |
| pictures with decoded picture hash ok / none / bad | 249 / 58 / 0 |
| streams confirmed (G2) | 152 / 152 |
| explained oracle disagreements | 3 (`disagreements.txt`: ffmpeg on `SAODBLK_A`; ffmpeg drops 4 of 6 pictures of `VPSSPSPPS_A` at its size changes; ffprobe size of `VPSSPSPPS_A`) |
| HEIC files / decoded by both / equal to libheif (yuv, alpha, info) | 176 / 56 / 56 (3 alpha planes) |
| HEIC files libheif decodes and hevc-rs rejects (explained) | 6 when measured: overlay/identity primaries (C008, C014, C039), 4:4:4 (MIAF003), the `mini` form (lightning_mini, C71); overlay and identity primaries are decoded since, re-measure |

Two oracle settings matter: ffmpeg runs with `-flags unaligned -autoscale 0` (else it aligns the
left crop and scales pictures that change size), and libheif with
`output_image_nclx_profile_passthrough = 1` (else it converts limited-range YCbCr to full range).
With the libheif default, a third witness (dec265 on the same item bytes) equalled hevc-rs on all
45 single-item files.

## Corpus

JCT-VC HEVC_v1 conformance streams (all 148 in `tests/parity/corpus.toml`, since each starts
with an intra picture), plus 9 RExt streams (4:0:0, 10/12-bit, PCM, bit depth) for 4:0:0 and
depth coverage and the rejection classes; fuzz-grown files are added when present. Fetched in CI into a BuildKit cache mount
(`id=hevc-rs-data`), never committed.

HEIC corpus (`tests/parity/heic-corpus.toml`): the HEVC-coded HEIC/HEIF sources of isobmff-rs's
corpus, at the same pinned commits, except the libheif fuzzing corpora (`fuzzing/data`: libheif
decodes hostile files that hevc-rs rejects on purpose, and the oracle runs with libheif's
security limits off, so they would be expected divergences; hevc-rs has its own `heic` fuzz
target): libheif (`tests/data`, `examples`),
MPEGGroup/FileFormatConformance (Git LFS), nokiatech/heif (gh-pages conformance files),
imazen/heic (`testdata/apple-hdr`) and aaron-rust/hdr-sample-images (Git LFS, the `3A*` and `3B*`
`.heic` files). `tests/parity/heic-fetch.sh` makes a blobless sparse checkout of each source at
its commit (verified after the fetch), pulls the LFS files, and keeps only `*.heic`, `*.heif` and
`*.hif` files, into a second BuildKit cache mount (`id=hevc-rs-heic-data`, files under
`/heic/x/<source>/...`), never committed. `run.sh` copies them flat (`/` becomes `__`) before the
two decoders run.
