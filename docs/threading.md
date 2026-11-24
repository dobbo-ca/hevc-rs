# Threading

`Limits::max_threads` (default `available_parallelism`; 0 counts as 1) caps the threads of one call,
the calling thread included. With 1 no thread is spawned. Output and errors are the same for every
value; `oracle --threads 1` against the default is a parity gate (see [parity.md](parity.md)).

## What runs in parallel

Only the in-loop filters, after the picture is reconstructed (`finish_picture`):

- Deblocking: the vertical-edge pass, then the horizontal-edge pass, each plane split into bands of
  whole CTB rows (`frame::par_bands`). A vertical edge only touches samples of its own row. In the
  horizontal pass the band that owns the edges of CTB rows `[B, B')` is the slice `[B - 4, B' - 4)`
  of the plane (chroma too): an edge reads and writes at most 4 rows above and 3 below it (luma) or
  2 above and 1 below (chroma), and edges are independent of each other, so the result does not
  depend on the split. Every pass ends when its bands have joined, before the next one starts. The
  SIMD line batches are created and flushed inside one band.
- SAO: each plane is read from the deblocked copy (charged as before) and written in bands of whole
  CTB rows; the neighbour samples across a band edge come from the shared copy.

The number of bands is `min(max_threads, CTB rows, luma samples >> 16, 64)`, at least 1. One band per
64 Ki luma samples: pictures under 128 Ki luma samples (so 256x256, 352x288, 416x240) and
`max_threads = 1` filter on the calling thread. The band count is also capped at 64, so the per-band
bookkeeping (a few KiB) stays inside the small-allocation reserve. Band 0 runs on
the calling thread, so a call never has more than `max_threads` threads.

Reconstruction (CABAC, prediction, transform) is serial, as is the hash check and the cropping.

## Determinism, memory, cancel

- Output: the planes are byte-identical for every thread count (unit tests compare 1, 2, 3, 5 and 64
  workers on every SIMD tier).
- Errors: a band can only fail with `Cancelled` (every other error is raised before the split). All
  bands run to the end and the error of the lowest band is returned.
- Memory: the bands share the planes and the one deblocked copy that SAO already charged, so
  threads add no buffer to `max_alloc`. Thread stacks are not charged. A refused spawn runs that
  band on the calling thread.
- Cancel: every band reads the flag once per CTB row (SAO) or per row of 4 samples (deblocking).
  The planes are put back into the frame on every path.

## Known deviations from the conventions

- No lazy shared pool and no single scope per call. Borrowing the planes of a per-picture `Frame`
  needs `std::thread::scope`, a persistent pool cannot borrow them in safe Rust, and a pool crate
  would be a new dependency. So there is one scope per filter pass (up to 6 per picture for deblocking
  and 3 for SAO), each spawning `bands - 1` threads. The `>> 16` gate keeps small pictures serial; its
  value is a guess to tune from the bench.
- Not parallel yet: tiles, wavefront rows, and pictures of an all-intra stream.
