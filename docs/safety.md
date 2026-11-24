# Safety

## `unsafe`

The crate root has `#![deny(unsafe_code, unsafe_op_in_unsafe_fn,
clippy::undocumented_unsafe_blocks)]`. `unsafe` is allowed in two places only, as in the rust-ports
safe-SIMD pattern: the load/store helpers of `src/simd/x86.rs` and `src/simd/arm.rs` (each takes a
fixed-size array reference, so its type proves the bounds), and the per-family dispatch that calls
a `#[target_feature]` kernel after `simd::tier()` / `simd::available()` found the features at run
time. The kernels themselves are safe functions. Each kernel family has a section below.

## Hostile input

- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!` or slice indexing in non-test code
  (clippy denies them; tests are exempt). Reads use `get`, iterators and fixed arrays indexed by
  masked values.
- Every size from the input uses `checked_*` arithmetic.
- Nothing is allocated from an input size before the ledger has booked it and the limits have
  passed; allocation uses `try_reserve_exact` (see [limits.md](limits.md)).
- NAL units of layer 1 and up, and reserved NAL unit types, are skipped, not trusted.
- Fuzzing runs with a memory cap so an allocation outside the limits is a crash (see
  [fuzzing.md](fuzzing.md)).

## Kernels removed

A kernel stays only when the bench shows a gain (`tests/parity/bench.sh`, corpus totals at one
thread, best of 3; `HEVC_SIMD` forces the tier). These were tried and removed:

| Kernel | Tiers | Bench | Result |
|---|---|---|---|
| Output pack `u16` to `u8` (`src/crop.rs`) | AVX2 (`packus` + lane permute), AVX-512 (`vpmovwb`), NEON (`vmovn`) | run 45286842132: scalar 5914.7 ms, avx2 5912.9 ms, avx512 5912.4 ms (dec265 3360.8 ms) | no measurable gain; the scalar narrowing loop stays |
| Deblocking luma/chroma line filters (`src/deblock.rs`, branch `sp-dblk-r5t1`, PR #16) | AVX2, NEON (Avx512 used AVX2) | run 45299840365 vs main run 45301270780: all classes 1t 4155.6 ms vs 4173.9 ms; ENTP_A 1055 vs 1053 ms, OPFLAG_B 949 vs 936 ms; the batched scalar path was slower (5273.4 vs 5216.7 ms) | no measurable gain; the scalar filters stay |
| AVX-512 inverse transform 16x16 / 32x32 and SAO rows of 32 lanes (branch `sp-a512-h4w9`, PR #20) | AVX-512 F/BW/VL | run 45307535320: all classes 1t, avx512 tier 3543.4 ms vs `HEVC_SIMD=avx2` 3515.5 ms | 0.8% slower; `Tier::Avx512` keeps the AVX2 kernels |

The output crop (`src/crop.rs`) and the picture hash (`sei::compute`) read the cancel flag before
every row and return `Error::Cancelled`.
## SIMD: transform (`src/transform.rs`, `src/transform/avx2.rs`, `src/transform/neon.rs`)

Scope: dequantisation (8.6.3), inverse DCT 4 to 32 and DST 4 (both passes, the i16 clip between
them, the bdShift rounding), the transform-skip shift, and the residual add with clip (8.6.7).
Scalar is the oracle and the fallback (`dequant_scalar`, `transform_scalar`, the loop in
`add_row`).

| Tier | Kernels |
|---|---|
| Scalar | oracle |
| AVX2 | `avx2::residual` (dequant, transform, transform-skip rounding), `avx2::add_row` |
| AVX-512 | no kernel of its own: dispatches to the AVX2 ones (the rows are 4 to 32 wide, 256 bits are enough) |
| NEON | `neon::residual`, `neon::add_row` |

`unsafe` blocks, all in `residual_simd` and `add_row` of `transform.rs` (`#[allow(unsafe_code)]`
on those two fns), plus the shared load/store helpers of `simd/x86.rs` and `simd/arm.rs`:

- Four calls of a `#[target_feature]` kernel. Invariant: the CPU has the feature. Checked by the
  match guard right before the call (`is_x86_feature_detected!("avx2")` /
  `is_aarch64_feature_detected!("neon")`), so even `residual_block(tier, ..)` with a tier the CPU
  lacks falls back to scalar. The detection is std's cached atomic load, not a CPUID per call.
`add_row` also runs scalar unless both slices have the same length 4, 8, 16 or 32. The kernels contain no `unsafe`: arrays come from slices with
  `as_chunks`, `first_chunk` and `chunks_exact`.
- Kernel inputs are validated in safe code before the call: `residual_block` checks size, bit
  depth, qP and the buffer length; `reconstruct` gets each frame row from `Frame::row_mut`.

Overflow: the dequantisation runs in i32 only when `max|c| * max(m) * scale + round` fits i32
(`dq_for`), else in i64 as scalar does. The transform inputs are i16 (clamped by the scaling), the
matrix entries are at most 90, so `madd` sums and the i32 accumulators cannot overflow and match
scalar. In the add the residual is clamped to +-65536 first, which does not change the clipped
sum for samples of at most 16 bits.

Tests: `transform::tests::tiers_match_scalar` (all sizes, depths 8/10/12/16, qP range, DCT, DST,
skip, bypass, flat and default scaling lists, extreme coefficients) and `fuzz_tiers_random_bytes`.
Fuzz: `tests/fuzz/fuzz_targets/transform_tiers.rs` calls `hevc::__fuzz_transform::tiers_agree`.
## SIMD: intra prediction (`src/intra.rs`, `src/intra/{x86,arm}.rs`)

Scalar is the oracle and the `Scalar` tier. The vector kernels work on 8 (AVX2) or 4 (NEON)
lanes of i32 over the same `[[i32; 32]; 32]` block the scalar code fills, so every tier gives the
same bits, including for out-of-range reference samples. They are safe `#[target_feature]` fns
with no `unsafe`; arrays come from slices with `chunks_exact` / `try_from` in safe code, loads
and stores are the `simd::{x86,arm}` helpers.

| Kernel | Does | Scalar left as is |
|---|---|---|
| `smooth` | the [1 2 1] reference filter | strong bilinear filter, gather and substitution, DC and its edge filter, boundary filters of modes 10 and 26 |
| `planar` | 8.4.4.2.4 | |
| `angular` | the (32-f)*a + f*b + 16 >> 5 interpolation | building `ref[]` (invAngle), the transpose of horizontal modes |
| `store` | clamp to the bit depth, narrow to u16 | |

Tiers: Avx2 (x86 AVX2), Avx512 runs the AVX2 kernels (a block is at most 32 wide, 16 i32 lanes
would not fill), Neon. `run` is the one dispatch.

`unsafe` inventory (besides the helpers of `simd/`): two blocks, both in `intra::run`, the calls of `x86::run` and
`arm::run`. Invariant: the CPU has AVX2 / NEON; checked by `is_x86_feature_detected!("avx2")` /
`is_aarch64_feature_detected!("neon")` in the match guard right before the call, so even a tier
value that was not taken from `simd::available()` falls back to scalar.

Bounds: block size is validated in `predict` (log2 2..=5, block inside the plane); the reference
line has 7 slots of zero padding after the 129 samples and `ref[]` has room for the 8-lane
loads, so the loads past the last sample stay inside the arrays. `arr` returns None instead of
reading out of range. Samples are u16 so the i32 arithmetic cannot overflow (at most 64 * 65535).

The reference gather walks the left column, the corner and the top row in runs of the samples of
one min-TB cell (6.4.1 gives one answer per cell) and asks availability once per run through
`frame::Avail`, which looks up the current block's z-scan address, slice and tile once and keeps
the CTB part of the answer for the last CTB asked. An available run is copied from the plane, an
unavailable run is filled with its predecessor (substitution). `tests::gather_matches_per_sample`
and `tests::gather_matches_per_sample_random` (random CTB and min-TB sizes, tiles, independent and
dependent slice segments, CTBs never started, aligned and unaligned blocks, luma and chroma) check
it against one `Frame::available` call per sample. constrained_intra_pred_flag cannot remove a
sample in an intra-only decoder (every neighbour is intra), so the gather does not read it.

Tests: `tiers_match_scalar` (all sizes, modes, bit depths 8/10/12/16, extreme and over-range
samples), `frames_match_scalar` (whole frames, luma and chroma, every aligned position),
`fuzz_entry`. Fuzz target `intra_tiers` (`__fuzz_intra::run`).
## SIMD: SAO (`src/sao_simd.rs`)

Row kernels for band offset (`bo_row`) and edge offset (`eo_row`) on 16-bit lanes. `sao_ctb`
(`src/sao.rs`) validates in safe code and hands a kernel only a run of one row that has no
PCM/bypass cell, whose edge-offset neighbours are inside the picture and a usable CTB (the first
and last column of the CTB, other rows, and everything on `Tier::Scalar` take the per-sample
scalar code, which is the oracle). Offsets equal to `i16::MIN` or tier Scalar skip the kernel.

- Tiers: Scalar (per-sample code in `sao_ctb`), AVX2 (16 lanes, `shuffle_epi8` offset lookup,
  saturating add/sub for the clip), AVX-512 (dispatched to the AVX2 kernel: a row kernel bound by
  loads gains nothing from 32 lanes), NEON (8 lanes, `vqtbl1q_u8` lookup).
- `unsafe` blocks: the two dispatch fns `eo_row` and `bo_row` (call the safe `target_feature`
  kernels; invariant: the tier came from `simd::tier()` / `simd::available()`, which return it
  only when the feature was detected). The load/store helpers are the shared ones in
  `src/simd/{x86,arm}.rs`; kernels get arrays from `as_chunks`, the tail runs the scalar row
  code.
- Tests: `sao_simd::tests::tiers_equal_scalar` (rows, bit depth 8/10/12/16, extreme offsets and
  samples) and `sao::tests::tiers_equal_scalar` (whole pictures with slices, PCM cells). Fuzz
  target `sao_tiers` (`__fuzz_sao::tiers`).
## SIMD: deblocking (`src/deblock.rs`, modules `x86k` and `armk`)

Each edge is filtered in batches of 4 segments of 4 lines (16 lines along the edge). The band
code first computes the parameters of the 4 segments (`Seg`: beta, tC, kept p or q side; tC 0
means nothing to filter) straight from the metadata grids, then filters. A full batch goes to the
kernel; a partial one (the right or bottom end of a plane) and everything on `Tier::Scalar` run
`luma_segment` / `chroma_segment`, the scalar oracle.

- Kernel layout: 16-bit lanes, one line per lane. A horizontal edge loads the 8 rows p3..q3 of
  16 columns as they are; a vertical edge loads 16 rows of 8 samples and transposes them (two
  8x8 transposes in the 128-bit halves on AVX2, two halves of 8 lines on NEON). dE, dEp, dEq and
  the strong/normal choice of a segment come from its lines 0 and 3 inside the vectors (lanes 0
  and 3 of each group of 4 added or and-ed, then broadcast) and are applied with blends, as are
  the kept sides (PCM with pcm_loop_filter_disabled, cu_transquant_bypass).
- Tiers: Scalar, AVX2, AVX-512 (dispatched to the AVX2 kernel), NEON. Bit depths above 10 run
  scalar: `9 * (q0 - p0)` of a 12-bit sample does not fit 16 bits. Up to 10 bits every
  intermediate is below 2^14.
- `unsafe` blocks: the two calls in `kernel` (invariant: the CPU has AVX2 / NEON; checked by
  `is_x86_feature_detected!("avx2")` in the match guard, and Neon comes only from
  `simd::tier()` / `simd::available()`). `kernel` cuts the window of the batch out of the plane in
  safe code (16 rows of 8 samples, or 8 rows of 16) and returns false when it does not fit; the
  kernels take their arrays from that window with `first_chunk`, load and store through the
  shared helpers of `src/simd/{x86,arm}.rs`.
- Tests: `deblock::tests::tiers_match_scalar` (whole pictures: random bS, QpY, two slices with
  random beta/tC offsets, PCM/bypass cells, bit depths 8, 9, 10 and 12, sizes with partial
  batches) and `fuzz_tiers_random_bytes`. Fuzz target `deblock_tiers` (`__fuzz_deblock::tiers`).
