# Benchmarks

Information only: there is no speed bar and no CI gate. Nothing is run locally.

`examples/bench.rs`: `bench [--threads N] <stream>` decodes one Annex B file once and exits 0
when it yields a picture (`bench --tier` prints the SIMD tier). `tests/parity/bench.sh` times
that process against the reference decoders, best of 3, on the intra prefix of every corpus
stream and on every file of the HEIC corpus. `.github/workflows/bench.yml` runs on push to
`main` and on `workflow_dispatch` (also on a PR branch: `gh workflow run bench.yml --ref <branch>`).

- References: libde265 v1.0.15 `dec265` and Debian bookworm `ffmpeg` for Annex B streams;
  libheif v1.23.7 with libde265 (`heic-oracle`) for HEIC files.
- 1 thread: `dec265 -t1`, `ffmpeg -threads 1`, `bench --threads 1`, `HEIC_THREADS=1` (libheif's
  `heif_context_set_max_decoding_threads` and `max_threads` of `examples/heic`). Default threads:
  `dec265 -t$(nproc)`, `ffmpeg` with no `-threads`, the default `Limits::max_threads`.
- hevc-rs builds: the default release build (runtime SIMD dispatch), and the v4 build
  (`RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls"`, the
  flags of our release binaries). The 1-thread time is also given with `HEVC_SIMD=scalar` and
  `HEVC_SIMD=avx2`, so the gain of each tier shows.
- Classes, from `oracle probe`: `main` (8-bit 4:2:0), `main10` (10-bit 4:2:0), `rext` (12-bit,
  4:0:0), `heic`.

`.github/workflows/profile.yml` (dispatch only) prints where hevc-rs and dec265 spend their time
(`tests/parity/profile.sh`: perf, or callgrind when the host allows no perf events).

## Results

Main at 5b860a0 (the speed round), bench run 45312989791, best of 3, 16 CPUs, ms. Reference:
dec265 for main/main10/rext, libheif with libde265 for heic. Ratio: hevc-rs / reference (lower
is better). v4: hevc-rs built with `-C target-cpu=x86-64-v4` and the false-deps features.

| class | n | ref 1t | ffmpeg 1t | hevc 1t | v4 1t | scalar 1t | avx2 1t | ratio 1t | ratio v4 1t | ref nt | ffmpeg nt | hevc nt | v4 nt | ratio nt | ratio v4 nt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 135 | 3298.0 | 5952.3 | 3032.4 | 2734.6 | 4656.4 | 3008.4 | 0.92 | 0.83 | 2016.6 | 5695.6 | 1561.3 | 1403.7 | 0.77 | 0.70 |
| main10 | 11 | 40.4 | 317.3 | 25.1 | 23.2 | 32.1 | 25.1 | 0.62 | 0.57 | 36.5 | 347.7 | 25.0 | 23.4 | 0.68 | 0.64 |
| rext | 3 | 19.5 | 101.2 | 13.8 | 12.4 | 17.4 | 13.5 | 0.71 | 0.64 | 18.0 | 107.2 | 13.8 | 12.5 | 0.77 | 0.69 |
| all | 149 | 3357.9 | 6370.8 | 3071.4 | 2770.2 | 4706.0 | 3046.9 | 0.91 | 0.82 | 2071.1 | 6150.5 | 1600.1 | 1439.6 | 0.77 | 0.70 |
| heic | 58 | 6486.0 | 0.0 | 2478.4 | 2292.8 | 4087.3 | 2531.8 | 0.38 | 0.35 | 6794.9 | 0.0 | 2289.0 | 2080.0 | 0.34 | 0.31 |

Before the speed round (bench run 45280698331, main 947dccb, scalar and serial; the HEIC column of
that run timed the files' output writes too, so it is left out):

| class | n | dec265 1t | ffmpeg 1t | hevc 1t | ratio 1t | dec265 nt | hevc nt | ratio nt |
|---|---|---|---|---|---|---|---|---|
| main | 135 | 3290.4 | 5786.5 | 5890.3 | 1.79 | 2004.4 | 5891.8 | 2.94 |
| main10 | 11 | 40.0 | 309.0 | 42.3 | 1.06 | 35.5 | 42.1 | 1.18 |
| rext | 3 | 19.5 | 95.0 | 23.8 | 1.22 | 17.9 | 23.8 | 1.33 |
| all | 149 | 3349.9 | 6190.5 | 5956.3 | 1.78 | 2057.9 | 5957.6 | 2.90 |

What each PR gave (all classes, 1 thread unless noted; each run against the main before it):

| PR | change | before | after |
|---|---|---|---|
| #11 | SIMD dequant, inverse DCT/DST, residual add | 5912.4 | 5107.5 |
| #14 | SIMD intra prediction, cached availability, no per-block zeroing | 5107.5 | 4576.3 |
| #15 | CABAC engine and residual_coding in fast scalar | 5107.5 | 4678.2 |
| #17 | SIMD SAO | 4173.9 | 3520.0 |
| #19 | deblocking and SAO in CTB-row bands (default threads) | 3519.0 nt | 3226.8 nt |
| #18 | pictures in parallel; hash check opt-in (`Options::check_hash`) | 3523.8 / 3226.8 nt | 3106.5 / 1684.3 nt |
| #25 | deblocking in edge batches of 16 lines, AVX2 and NEON kernels (bench runs 45481933675 and 45483838642 vs main run 45466964176) | 3041.1 / 1622.6 nt | 2783.2 / 1532.7 nt (second run 2803.0 / 1545.4 nt) |

Kernels that gave no gain were removed (docs/safety.md, "Kernels removed"). Open for speed:
parallel decode by tile / WPP row inside one picture.
