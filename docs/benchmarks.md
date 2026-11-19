# Benchmarks

Information only: there is no speed bar and no CI gate.

`examples/bench.rs`: `bench [--threads N] <stream>` decodes one Annex B file once and exits 0
when it yields a picture. `tests/parity/bench.sh` times that process and `dec265` on the intra
prefix of every corpus stream, so the report compares dec265 with hevc-rs per stream. Decode is
serial in v1. Nothing is run locally.
