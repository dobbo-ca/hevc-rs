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

## Gates

1. hevc-rs `.yuv` equals dec265 output for every stream.
2. Equals ffmpeg.
3. Every `.info` hash is ok or none.
4. `--threads 1` equals the default.
5. Phase 3 diagnostic: `--no-filters` against dec265 with deblocking and SAO disabled.

## Corpus

JCT-VC HEVC_v1 conformance streams (all, since each starts with an intra picture), selected RExt
streams for the rejection classes, public HEIC samples later, and fuzz-grown files. Fetched in CI
into a BuildKit cache mount (`id=hevc-rs-data`), never committed.
