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
