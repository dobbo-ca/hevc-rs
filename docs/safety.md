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

The reference gather asks 6.4.1 once per run of samples in the same min-TB cell instead of once
per sample; `tests::gather_matches_per_sample` checks it against the per-sample version.

Tests: `tiers_match_scalar` (all sizes, modes, bit depths 8/10/12/16, extreme and over-range
samples), `frames_match_scalar` (whole frames, luma and chroma, every aligned position),
`fuzz_entry`. Fuzz target `intra_tiers` (`__fuzz_intra::run`).
## SIMD: deblocking (`src/deblock.rs`, `src/deblock/{x86,arm}.rs`)

The decisions (dE, dEp, dEq, tc, beta) stay scalar and per 4-line segment. The filter arithmetic
runs on batches of 8 lines (`Lines`, `CLines`: one 32-bit lane per line, samples as 8 or 4 rows
of 8 lanes). A pass queues the lines of its segments and filters 8 at a time; segments of one
pass touch disjoint samples, so the order does not matter. The gather and scatter between the
plane and the batch is safe code (`load` / `store`, bounds-checked `get`).

| Tier | Luma | Chroma |
|---|---|---|
| Scalar | `luma_line` per lane (the oracle) | `chroma_line` per lane |
| Avx2 | `x86::luma`, 8 lanes | `x86::chroma`, 8 lanes |
| Avx512 | the AVX2 kernels (a segment is 4 lines; 16 lanes would not pay) | same |
| Neon | `arm::luma`, two halves of 4 lanes | `arm::chroma` |

Same 32-bit integer expressions as the scalar code, branches as masks and blends: bit-exact,
no overflow difference (samples <= 65535, tc <= 24 * 256).

`unsafe` inventory:

- `deblock::filter_luma` and `filter_chroma` (`#[allow(unsafe_code)]`): call the safe
  `#[target_feature]` kernels. Invariant: the CPU has AVX2 (NEON). Checked in the match guard with
  `is_x86_feature_detected!("avx2")` (`is_aarch64_feature_detected!("neon")`), so a tier that the
  CPU lacks runs the scalar code.
- `simd::x86::ld_i32x8` / `st_i32x8`, `simd::arm::ld_i32x4` / `st_i32x4`: take `&[i32; N]`, the
  type proves the bounds.

Tests (`deblock::tests`): kernels on 20 000 random batches and on 0/max extremes for every tc,
all depths (8, 10, 12, 16), every tier of `simd::available()` against Scalar; whole 64x64
pictures with random QpY, offsets, edges and no_filter cells. Fuzz: `deblock_tiers`
(`__fuzz_deblock::tiers`).
