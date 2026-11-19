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
