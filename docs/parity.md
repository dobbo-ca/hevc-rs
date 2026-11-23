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
6. G6 diagnostic, report only: `--no-filters` against dec265 with deblocking and SAO disabled.

## Counts

From the CI run of main with all gates green (HEVC_v1 intra prefixes):

| Count | Value |
|---|---|
| streams (intra prefixes) | 147 |
| equal to dec265 | 145 |
| equal to ffmpeg | 145 |
| pictures with hash ok | 244 |
| pictures with hash bad | 0 |

The live numbers are in each run's `summary.md`; update this table when the corpus changes.

## Corpus

JCT-VC HEVC_v1 conformance streams (all 147 in `tests/parity/corpus.toml`, since each starts
with an intra picture). RExt streams for the rejection classes and public HEIC samples are
planned, not in the corpus yet; fuzz-grown files are added when present. Fetched in CI into a
BuildKit cache mount (`id=hevc-rs-data`), never committed.
