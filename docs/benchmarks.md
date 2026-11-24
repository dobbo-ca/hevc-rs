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

bench run on main 2026-11-23 (after #5), 150 streams timed by both tools, best of 3, one thread,
scalar code (no SIMD yet):

| | dec265 | hevc-rs | hevc-rs / dec265 |
|---|---|---|---|
| total, ms | 3372.93 | 5929.69 | 1.76 |
| per stream | | | 0.83 (DBLK_A_MAIN10) .. 2.23 (AMP_D) |

Next for speed: SIMD for the inverse transforms, and intra prediction (scalar code as
the oracle), then parallel decode by tile / WPP row.
