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

A/B mode: `gh workflow run bench.yml --ref <branch> -f base=<ref>` also builds `<ref>` (default
build; its `Cargo.lock` must equal the branch's) and times it at 1 thread right after the branch's
default build, on the same host. The summary then has an "A/B" table with one row per class (base,
this, change, inputs faster), and column 14 of `all.tsv` / `heic.tsv` is the base time. The base
runs right after the branch's build, which gives a small bias against the branch.

`.github/workflows/profile.yml` (dispatch only) prints where hevc-rs and dec265 spend their time
(`tests/parity/profile.sh`: perf, or callgrind when the host allows no perf events; callgrind
cannot run AVX-512 code). With `-f lines='<symbol>,<symbol>'` (names as the report prints them) it
also prints the self time per source line of each symbol and the callers of the libc memset,
memmove and memcpy symbols (DWARF unwinding; it does not find every caller).

Every bench, profile and fuzz job prints the CPU model in its summary. A hosted `ubuntu-24.04`
runner is one of these CPUs (seen so far): AMD EPYC 7763 (Zen 3, no AVX-512), AMD EPYC 9V74 and
9V45 (Zen 4/5), Intel Xeon Platinum 8370C (Ice Lake), Intel Xeon Platinum 8573C (Emerald Rapids)
and Intel Xeon 6975P-C and 6973P-C (Granite Rapids). The dispatch input `require_avx512` stops the job at once
on a host without AVX-512 (fuzz: without VNNI too), so a rerun can land on one:
`gh workflow run bench.yml --ref <branch> -f require_avx512=true`.

The default tier runs first for each stream in `bench.sh`, and that gives it a small bias against
it: on main c6f4caf (run 45898645393, Xeon 6975P-C), with the same kernels in both tiers, the
default tier was faster than `HEVC_SIMD=avx2` on only 35 of 149 streams, +0.2% in total.

## AVX-512 kernels

Profile of main c6f4caf on an Intel Xeon 6975P-C (profile run 45896383881, perf, 1 thread, avx512
tier then running the AVX2 kernels), self time of the SIMD kernels: transform 7.0%
(`avx2::run::<8>` 3.0%, `add_row` 1.4%, dequant 1.3%, 16/32 passes 1.3%), intra 3.6%, SAO 2.8%,
deblocking 1.8%; about 15% in total. The rest is scalar (intra::predict 12.1%, the CTU and
residual parse 11.8%, CABAC 8.0%).

One PR per kernel family, each benched on an AVX-512 runner: before is `HEVC_SIMD=avx2`, after
the default (avx512) tier, same build and host, all Annex B classes at 1 thread, ms.

| PR | family | AVX-512 subsets | run, CPU | avx2 | avx512 | change | streams faster | heic change | result |
|---|---|---|---|---|---|---|---|---|---|
| #38 | transform: residual (VNNI dot products, vpermw operands, vpmovsdw clip), add_row (vpmovusdw) | F, BW, VL, VNNI (add_row: F, BW, VL) | 45900863802, Xeon 8370C | 5229.8 | 5198.3 | −0.6% | 76 / 149 | −1.6% | kept |
| #39 | intra: smooth, planar, angular, store | F, BW, VL | 45901715151, Xeon 8370C | 5361.1 | 5499.5 | +2.6% | 3 / 149 | +2.3% | dropped (retried on Zen 4 and Granite Rapids below: dropped again) |
| #40 | SAO: edge and band offset (vpermw lookup, mask compares, masked tails) | F, BW, VL | 45903926963, Xeon 6975P-C | 3896.8 | 3875.0 | −0.6% | 97 / 149 | −0.4% | kept |
| #41 | deblocking edge kernel (vpermt2w transposes, mask decisions) | F, BW, VL | 45904887986 (2 runs), Xeon 6975P-C | 3917.5 / 3951.5 | 3899.2 / 3944.4 | −0.5% / −0.2% | 61, 56 / 149 | +0.1% / +0.05% | dropped |

Intra and deblocking keep the AVX2 kernels (docs/safety.md, "Kernels removed"). A CPU with
AVX-512 F, BW and VL but no VNNI runs the AVX2 transform residual with the AVX-512 add_row and SAO
kernels (tested under SDE -skx).

Main 06717f1 (#38, #40 and the review fixes #42), `bench.yml` runs 45908276617 to 45918427976,
ms. The dec265 ratio is hevc-rs (default tier) / dec265, at 1 thread and at the default threads
(4 CPUs); heic is against libheif. A CPU with no AVX-512 (Zen 3) runs the unchanged AVX2 code: see
the hosted-runner baseline below (ratio 0.82 / 0.82).

| CPU | run | dec265 1t | avx2 1t | avx512 1t | change | ratio 1t | ratio nt | heic ratio 1t | heic ratio nt |
|---|---|---|---|---|---|---|---|---|---|
| AMD EPYC 9V74 (Zen 4) | 45908276617 | 5165.7 | 4031.7 | 3827.1 | −5.1% | 0.74 | 0.74 | 0.46 | 0.43 |
| AMD EPYC 9V74 (Zen 4) | 45913669608 | 5166.1 | 4030.2 | 3823.9 | −5.1% | 0.74 | 0.74 | 0.46 | 0.43 |
| Intel Xeon 6975P-C | 45911621683 | 4955.3 | 3980.4 | 3833.0 | −3.7% | 0.77 | 0.78 | 0.35 | 0.34 |
| Intel Xeon 6975P-C | 45915240857 | 4957.4 | 3962.6 | 3824.0 | −3.5% | 0.77 | 0.77 | 0.35 | 0.34 |
| Intel Xeon 6975P-C | 45916817418 | 4892.8 | 3912.2 | 3753.9 | −4.0% | 0.77 | 0.76 | 0.35 | 0.34 |
| Intel Xeon 6975P-C | 45918427976 | 4974.9 | 4001.2 | 3847.7 | −3.8% | 0.77 | 0.77 | 0.35 | 0.34 |
| Intel Xeon 8573C | 45909998606 | 5993.2 | 4855.3 | 4726.0 | −2.7% | 0.79 | 0.79 | 0.37 | 0.35 |

Before (main c6f4caf, run 45898645393, Xeon 6975P-C): ratio 0.80 at 1 thread and 0.79 at the
default threads, heic 0.36 / 0.35. Per family, the gain on Ice Lake (8370C, the host with the
largest frequency drop for 512-bit code) was smaller than the combined gain on Zen 4 and Granite
Rapids; no combined run landed on an Ice Lake host.

### AVX-512 intra kernels, retry on more CPUs

PR #39's intra kernels were benched only on Ice Lake (8370C), where 512-bit code can lower the
clock. They were rebased on main (branch `avx512-intra-ab-r8w3`, on top of the A/B tooling of #45)
and benched again with `require_avx512`, 8 dispatches in all. Rule: keep them only if they are
about 0.5% or more faster on every AVX-512 CPU reached and slower on none; no CPU-model gate.
A/B: base = main, whose avx512 tier runs the AVX2 intra kernels; this = the AVX-512 intra kernels;
both the default build and tier, same host, 1 thread, ms. Run 1 has no A/B columns (the A/B mode
came after it); its avx512-vs-avx2 gap compares with that of main on the same CPU (−5.1%, runs
45908276617 and 45913669608).

| # | run | CPU | base all | this all | change | faster | base heic | this heic | heic change | avx512 vs avx2 (this) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 46048849957 | AMD EPYC 9V74 (Zen 4) | – | 3821.0 | – | – | – | 3395.0 | – | −4.44% (main: −5.1%) |
| 2 | 46051008778 | Intel Xeon 6975P-C | 3798.9 | 3786.6 | −0.32% | 77 / 149 | 3273.3 | 3274.0 | +0.02% | −3.45% |
| 3 | 46053546606 | Intel Xeon 6975P-C | 3721.0 | 3702.6 | −0.49% | 102 / 149 | 3248.4 | 3244.3 | −0.12% | −4.06% |
| 4 | 46056134028 | Intel Xeon 6975P-C | 3771.1 | 3759.5 | −0.31% | 75 / 149 | 3261.5 | 3257.7 | −0.12% | −3.74% |
| 5 | 46058306279 | AMD EPYC 9V74 (Zen 4) | 3811.1 | 3828.2 | +0.45% | 43 / 149 | 3366.7 | 3393.0 | +0.78% | −4.45% |
| 6 | 46060785942 | AMD EPYC 7763 (Zen 3, no AVX-512) | – | – | – | – | – | – | – | stopped by `require_avx512` |
| 7 | 46063216181 | Intel Xeon 6975P-C | 3916.9 | 3893.0 | −0.61% | 82 / 149 | 3302.7 | 3292.8 | −0.30% | −2.50% |
| 8 | 46065541824 | Intel Xeon 6973P-C | 4425.7 | 4424.6 | −0.03% | 62 / 149 | 3928.7 | 3995.4 | +1.70% | +1.51% |

Result: dropped. They were slower on Zen 4 (run 5, and run 1 against main), under 0.5% on Granite
Rapids (6975P-C, four runs; heic within ±0.3%), and slower on heic on the 6973P-C. The Zen 5 host
(EPYC 9V45) was not reached in the 8 dispatches. On the 6973P-C (a CPU not seen before), the
whole avx512 tier was 1.5% slower than avx2 at 1 thread in that run. Intra keeps the AVX2 kernels.

## Scalar intra prediction

Profile of main + the A/B tooling (profile run 46053542360, AMD EPYC 7763, perf, 1 thread, the
per-line report of #45), self time: `intra::predict` 11.3%, `Avail::at` 4.9%,
`intra::compute_into` 4.7%, memset 2.4%, memmove 2.6%. Inside `intra::predict`: the left reference
column copy 33% (`StepBy::next_back` takes a remainder per sample: the `core::num::nonzero`
division is 10.7% of the symbol), the availability closure 8.6%, the checked `Frame::row_mut` of
the write-back about 12%, the top row copy 4%. Inside `compute_into`: the transpose of horizontal
angular blocks about 23% (two checked lookups per sample), the `ref[]` fill about 18%,
`is_x86_feature_detected!` 6%. Inside `Avail::at`: the MinTbAddrZs load and its checks about 45%.

One change per A/B (bench.yml A/B mode: base and this build in the same job, 1 thread, default
build and tier, ms); each base is main plus the changes kept before it. A change was kept at
about 1% or more on x86_64. There is no arm64 bench job, so no change was kept by the "faster on
both x86_64 and arm64" rule. The work stopped after three changes in a row under 1% (5, 6, 7).

| # | change | run, CPU | base all | this all | change | faster | heic change | result |
|---|---|---|---|---|---|---|---|---|
| 1 | left reference column by a forward `StepBy` (PR #46) | 46056191192, Xeon 6975P-C | 3854.3 | 3801.4 | −1.37% | 109 / 149 | −1.17% | kept |
| 2 | write-back of the block through one plane slice, not `row_mut` per row (branch `intra-writeback-r8w3`) | 46058302268, Xeon 6975P-C | 3755.3 | 3769.1 | +0.37% | 30 / 149 | −0.41% | dropped |
| 3 | no availability lookup for left / top / corner neighbours in the current CTB (branch `intra-avail-fast-r8w3`) | 46060782354, Xeon 6975P-C | 3729.7 | 3741.9 | +0.33% | 42 / 149 | +0.55% | dropped |
| 4 | transpose of horizontal angular blocks row by row of the output (PR #47) | 46063212985, EPYC 9V74 | 3851.8 | 3806.5 | −1.18% | 114 / 149 | −1.59% | kept |
| 5 | angular `ref[]` by slice copies (branch `intra-rf-r8w3`) | 46065537602, Xeon 6975P-C | 3695.2 | 3677.3 | −0.49% | 76 / 149 | +0.32% | dropped |
| 6 | availability z-scan order by bit interleave (6-10), no MinTbAddrZs load (branch `intra-avail-zs-r8w3`) | 46069795376, Xeon 6975P-C | 3693.8 | 3733.8 | +1.08% | 22 / 149 | +0.48% | dropped |
| 7 | reference arrays kept in `Scratch`, no zeroing per block (branch `intra-no-zero-r8w3`) | 46072274816, EPYC 7763 | 5168.0 | 5180.5 | +0.24% | 71 / 149 | +0.78% | dropped |

Profile after changes 1 and 4 (run 46067817526, EPYC 7763): `intra::predict` 8.5% (from 11.3%),
`Avail::at` 6.7%, `compute_into` 4.0%. No `unsafe` change was tried: the safe rewrites of the
checked write-back (2) and of the availability lookups (3, 6) gave no gain, so the bounds checks
there are not what costs the time.

Checks: `ci` (workflow_dispatch, all jobs, parity bit-exact) green on the stack of 1, 4 and 6 (run
46069819028) and of 1, 4 and 7 (run 46072277525). Fuzz, 900 s per target on the stack of 1 and 4:
annexb and hvcc (run 46074491189; its other x86_64 jobs failed in the image build, see below),
decode (46081512604), heic (46081514594), intra_tiers on x86_64 and arm64 (46084354147), and the
four tier targets on arm64 (46074491189): no crash. In one fuzz run of all targets, only the first
two x86_64 jobs fetched the corpus; the later jobs restored the `data` layer from the Actions
cache with an empty `/data` cache mount, and the seed step failed. One target per run
(`-f only=<target>`) avoids that.

Main before and after, `bench.yml` on push (default build and tier; ratio = hevc-rs / dec265, heic
against libheif; 1 thread / default threads, 4 CPUs). Main 3c52696 (#45) has the decoder code of
dc3aec9. The hosts differ between runs, so compare rows of the same CPU.

| main | run | CPU | dec265 1t | hevc 1t | ratio 1t | ratio nt | heic ratio 1t | heic ratio nt |
|---|---|---|---|---|---|---|---|---|
| dc3aec9 (before) | 46017392130 | Intel Xeon 6975P-C | 4894.5 | 3879.8 | 0.79 | 0.79 | 0.36 | 0.35 |
| 3c52696 (#45, before) | 46088427562 | AMD EPYC 7763 | 6351.7 | 5200.6 | 0.82 | 0.81 | 0.54 | 0.51 |
| 0373e66 (#46) | 46090267608 | AMD EPYC 7763 | 6339.0 | 5133.6 | 0.81 | 0.80 | 0.54 | 0.51 |
| 59e31b7 (#46, #47) | 46092051550 | AMD EPYC 9V45 (Zen 5) | 4186.1 | 3005.1 | 0.72 | 0.71 | 0.33 | 0.32 |

On the EPYC 9V45 (the first bench on Zen 5), the avx512 tier of main 59e31b7 (with the AVX2
intra kernels) was 5.1% faster than `HEVC_SIMD=avx2` (heic 4.1%).

## Hosted-runner baseline

Main 6de624c, `bench.yml` run 45718350632, on a GitHub-hosted `ubuntu-24.04` runner (4 CPUs, SIMD
tier avx2, no AVX-512, so the v4 columns are not run and show 0). This is the baseline for later
runs; compare ratios, not ms, with the older numbers below.

Reference: dec265 for main/main10/rext, libheif (with libde265) for heic. v4: hevc-rs built
with -C target-cpu=x86-64-v4 and the false-deps features. Ratio: hevc-rs / reference.

| class | n | ref 1t | ffmpeg 1t | hevc 1t | v4 1t | scalar 1t | avx2 1t | ratio 1t | ratio v4 1t | ref nt | ffmpeg nt | hevc nt | v4 nt | ratio nt | ratio v4 nt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 135 | 6276.4 | 13015.0 | 5191.4 | 0.0 | 7941.6 | 5180.5 | 0.83 | 0.00 | 4492.9 | 12703.2 | 3681.6 | 0.0 | 0.82 | 0.00 |
| main10 | 11 | 84.1 | 765.7 | 51.2 | 0.0 | 61.7 | 50.8 | 0.61 | 0.00 | 77.9 | 794.0 | 51.5 | 0.0 | 0.66 | 0.00 |
| rext | 3 | 37.9 | 229.0 | 27.7 | 0.0 | 33.9 | 27.5 | 0.73 | 0.00 | 35.8 | 239.9 | 28.0 | 0.0 | 0.78 | 0.00 |
| all | 149 | 6398.4 | 14009.7 | 5270.3 | 0.0 | 8037.3 | 5258.9 | 0.82 | 0.00 | 4606.6 | 13737.0 | 3761.1 | 0.0 | 0.82 | 0.00 |
| heic | 60 | 8565.3 | 0.0 | 4625.2 | 0.0 | 6994.6 | 4605.2 | 0.54 | 0.00 | 8380.1 | 0.0 | 4248.5 | 0.0 | 0.51 | 0.00 |

## Results (earlier self-hosted runner)

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
