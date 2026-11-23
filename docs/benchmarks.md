# Benchmarks

Information only: there is no speed bar and no CI gate.

`examples/bench.rs`: `bench [--threads N] <stream>` decodes one Annex B file once and exits 0
when it yields a picture. `tests/parity/bench.sh` times that process and `dec265` on the intra
prefix of every corpus stream, so the report compares dec265 with hevc-rs per stream. Both run
with one thread (`dec265 -t1`, `bench --threads 1`). Decode is serial in v1, so hevc-rs at its
default thread count is the same run; add the default-thread column (and `dec265` at its default)
with the threading phase. `.github/workflows/bench.yml` runs on push to `main` and on
`workflow_dispatch`. Nothing is run locally.
