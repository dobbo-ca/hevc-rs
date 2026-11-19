# Fuzzing

`tests/fuzz/` is a cargo-fuzz crate with its own `Cargo.lock`. Run with `--fuzz-dir tests/fuzz`.

| Target | What it checks |
|---|---|
| `decode` | `decode_into` with small `Limits`; no panic, no allocation over the cap |
| `decode_hvcc` | the length-prefixed path with a fuzzed `hvcC` record |
| `probe` | `probe` and `intra_prefix` |

All targets run with `-rss_limit_mb=2048 -malloc_limit_mb=512`, so an allocation outside the
limits is a crash. Seeds are the conformance streams, copied at build time, not kept in git.

`.github/workflows/fuzz.yml` runs weekly on Thursday at 12:23 UTC and on `workflow_dispatch`,
never on push or pull request.
