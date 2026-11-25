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

These numbers come from the earlier 16-CPU self-hosted runner; runs on GitHub-hosted ubuntu-24.04 runners (4 vCPU) are a new baseline, so compare ratios, not ms.

Main at 55ea54a (after the follow-up round), bench run 45525817059, best of 3, 16 CPUs, ms.
Reference: dec265 for main/main10/rext, libheif with libde265 for heic. Ratio: hevc-rs / reference
(lower is better). v4: hevc-rs built with `-C target-cpu=x86-64-v4` and the false-deps features.
Before this round (main 4a8ef2a, run 45466964176): all 3041.1 ms 1t (ratio 0.90), 1622.6 nt
(0.78); heic 2395.2 1t (0.37), 2104.8 nt (0.34). The speed round (main 5b860a0, run 45312989791)
gave all 3071.4 ms 1t (ratio 0.91), 1600.1 nt (0.77).

| class | n | ref 1t | ffmpeg 1t | hevc 1t | v4 1t | scalar 1t | avx2 1t | ratio 1t | ratio v4 1t | ref nt | ffmpeg nt | hevc nt | v4 nt | ratio nt | ratio v4 nt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 135 | 3295.2 | 5760.7 | 2471.2 | 2375.4 | 3757.5 | 2454.9 | 0.75 | 0.72 | 2006.0 | 5502.7 | 1337.1 | 1299.1 | 0.67 | 0.65 |
| main10 | 11 | 39.5 | 306.4 | 22.8 | 22.0 | 27.4 | 22.4 | 0.58 | 0.56 | 35.5 | 332.5 | 22.7 | 22.0 | 0.64 | 0.62 |
| rext | 3 | 19.5 | 95.0 | 13.1 | 12.9 | 15.7 | 12.8 | 0.67 | 0.66 | 17.9 | 103.8 | 13.3 | 12.8 | 0.74 | 0.71 |
| all | 149 | 3354.1 | 6162.2 | 2507.0 | 2410.3 | 3800.6 | 2490.2 | 0.75 | 0.72 | 2059.3 | 5938.9 | 1373.1 | 1333.9 | 0.67 | 0.65 |
| heic | 60 | 6381.4 | 0.0 | 2168.7 | 2113.5 | 3399.6 | 2209.9 | 0.34 | 0.33 | 6242.9 | 0.0 | 1954.2 | 1898.1 | 0.31 | 0.30 |

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
| #22 | intra reference gather: availability per cell run and slice copies (`intra::gather`, `frame::Avail`) (bench run 45475372926 vs main run 45466964176) | 3041.1 / 1622.6 nt; v4 2748.8; heic 2395.2 | 2810.7 / 1479.1 nt; v4 2614.1; heic 2310.8 |
| #25 | deblocking in edge batches of 16 lines, AVX2 and NEON kernels (bench runs 45481933675 and 45483838642 vs main run 45466964176) | 3041.1 / 1622.6 nt | 2783.2 / 1532.7 nt (second run 2803.0 / 1545.4 nt) |
| #29 | sparse inverse transform: scaling, passes and clearing only on the extent of the non-zero coefficients, DC-only blocks as one value (bench runs 45510056582 and 45510870457 vs main run 45496374688) | 2606.8; heic 2390.8 / 2376.1 nt | 2528.5; heic 2144.9 / 1946.0 nt (second run 2528.9; heic 2176.4 / 2008.9 nt) |

PR #29 in more detail (1 thread, ms, main → run 1 / run 2): the v4 build of the Annex B classes did
not move (all 2396.8 → 2405.0 / 2408.8), the default build gained 3% (2606.8 → 2528.5 / 2528.9,
dec265 3398.5 → 3362.2 / 3404.4) and the scalar tier 18% (4653.6 → 3835.1 / 3833.9). HEIC gained
in every build: default 2390.8 → 2144.9 / 2176.4, avx2 2511.6 → 2179.2 / 2222.1, scalar 4267.6 →
3362.8 / 3428.8 (libheif 6603.9 → 6398.2 / 6482.8). The v4 HEIC time of the main run (2787.8) was
disturbed; earlier main runs gave 2232.9 to 2292.8, against 2131.4 / 2178.6 here. main10 and rext
(11 and 3 small streams) were 0.3 to 0.7 ms slower in total.

Kernels that gave no gain were removed (docs/safety.md, "Kernels removed"). Open for speed:
parallel decode by tile / WPP row inside one picture.

### Tried and dropped: residual_coding / CABAC fast paths (PR #26)

The 4a8ef2a profile (run 45317036877, callgrind) put residual_coding and the CABAC engine at
about 8% of the instructions on the large main-class streams and up to about 30% on the small,
high-rate ones (SDH_A, TUSIZE_A). Commit d2d4ff7 on branch `sp-cabc-v5n1` tried, all bit-exact
(test, arm64, parity and parity-arm64 green in run 45491008129):

- DecodeDecision without a branch on the MPS/LPS outcome, with one 256-entry rangeTabLps table
  and one 256-entry transition table indexed by the packed context byte, inlined into the
  per-coefficient loops of residual_coding.
- sig_coeff_flag context patterns precomputed in scan order per (scanIdx, prevCsbf or 4x4).
- coeff_abs_level_remaining read with one look at the next 16 bypass bins (one division)
  when the codeword fits, instead of a prefix run and a separate suffix read.
- Fixed-size clearing of the coefficient block per block size.

Two bench runs (45491992963, 45493071189) against main 4a8ef2a (45466964176), 1 thread, ms:

| class | build | main | run 1 | run 2 |
|---|---|---|---|---|
| all (149) | default | 3041.1 | 3017.2 | 3027.0 |
| all (149) | v4 | 2748.8 | 2705.9 | 2723.6 |
| all (149) | dec265 (reference) | 3364.4 | 3346.2 | 3366.4 |
| heic (58) | default | 2395.2 | 2431.0 | 2523.6 |
| heic (58) | v4 | 2232.9 | 2238.8 | 3507.1 |
| SDH_A_Orange_4 | default | 61.41 | 61.03 | 61.09 |
| SDH_A_Orange_4 | v4 | 56.47 | 54.71 | 55.54 |

The second run's HEIC column was disturbed (the v4 time rose 57%, a host effect). Per input,
the Annex B geometric mean moved by -1.1% to -1.7% (default) and -1.8% to -2.4% (v4) after
correcting for the reference's drift; the class totals moved by under 1% in the default build,
which is inside run-to-run noise, and HEIC got no faster. Not merged. The engine's remaining
cost is the serial dependency of one bin on the previous one; the next speed step for these
streams was a partial inverse transform for blocks whose coefficients sit in the top-left
corner, done in #29 (sparse inverse transform, above).
