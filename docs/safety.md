# Safety

## `unsafe`

The crate root has `#![forbid(unsafe_code)]`. The SIMD phase, if it comes, will use the
safe-SIMD pattern of the conventions and revisit this page.

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

## Crop (src/crop.rs)

The output crop copies the conformance window of a `u16` plane into `Samples`. `u16` rows are
`extend_from_slice`. 8-bit rows are packed to `u8` (`s as u8`, truncating) by one kernel family,
`pack_u8_with(tier, dst, src)`:

| Tier | Kernel | Notes |
|---|---|---|
| Scalar | `pack_scalar` | the oracle |
| Avx2 | `x86k::pack_avx2` | mask to the low byte, `packus_epi16`, `permute4x64(0xD8)`; 32 samples per step |
| Avx512 | `x86k::pack_avx512` | `_mm512_cvtepi16_epi8` (`vpmovwb`, truncating); 32 per step |
| Neon | `neonk::pack_neon` | `vmovn_u16` + `vmovn_high_u16`; 16 per step |

Tails (under one step) go through `pack_scalar`. Every tier gives the same bytes for every `u16`,
including values over 255.

`unsafe` inventory: the three calls in `pack_u8_with` (the only function with
`#[allow(unsafe_code)]` here), one per tier. Invariant: the CPU has the tier's features, which
`simd::available()` proves by runtime detection; the slices are trimmed to one length at the top
of the function. The kernels are safe `#[target_feature]` fns that take arrays from slices with
`as_chunks`; the loads and stores are the `simd::x86` / `simd::arm` helpers.

`crop_plane` checks the window against the plane per row (`get`, `checked_add`, `chunks_exact`)
and reads the cancel flag before every row. The picture hash (`sei::compute`) reads it before
every row too. Both return `Error::Cancelled`. Tests compare every available tier with Scalar and
with a naive reference; the fuzz target `crop_tiers` does the same on bounded inputs.
