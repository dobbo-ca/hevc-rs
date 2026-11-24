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
  depend on the split. Every pass ends when its bands have joined, before the next one starts.
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
  workers; SAO on every SIMD tier, deblocking is scalar only).
- Errors: a band can only fail with `Cancelled` (every other error is raised before the split). All
  bands run to the end and the error of the lowest band is returned.
- Memory: the bands share the planes and the one deblocked copy that SAO already charged, so
  threads add no buffer to `max_alloc`. Thread stacks (512 KiB each, not the 2 MiB default) are not charged. A refused spawn runs that
  band on the calling thread.
- Cancel: every band reads the flag once per CTB row (SAO) or per row of 4 samples (deblocking).
  The planes are put back into the frame on every path.

## Known deviations from the conventions

- No lazy shared pool and no single scope per call. Borrowing the planes of a per-picture `Frame`
  needs `std::thread::scope`, a persistent pool cannot borrow them in safe Rust, and a pool crate
  would be a new dependency. So there is one scope per filter pass (up to 6 per picture for deblocking
  and 3 for SAO), each spawning `bands - 1` threads. The `>> 16` gate keeps small pictures serial; its
  value is a guess to tune from the bench.
- Not parallel yet: tiles and wavefront rows inside one picture, and pictures of an all-intra stream (a separate PR).

## HEIC grids

`decode_heic` and `decode_heic_item` decode the tiles of a grid in parallel. Tile 0 is
decoded alone (it gives the format and the canvas size). The other tiles are taken from an atomic
counter by up to `max_threads` workers (the caller is one of them); each worker decodes its tile
with `decode_hvcc_into` on one thread (`max_threads = 1` in the inner call) and copies it onto the
shared canvas under a mutex (tiles do not overlap, so the copy order does not matter). The results
are then read in tile order: the first error in tile order is the call's error, a tile that ran out
of its share of the memory budget is decoded again alone with all of it. So the output and the
error are those of the serial loop for every thread count (unit tests in `src/heic.rs` run 1, 2,
4 and 8 against the serial result, with errors, tight budgets and a cancel flag).

Memory: the workers split what the serial path would have had for one tile at a time
(`max_alloc` minus the isobmff share, minus the canvas, minus the result slots), each share is
charged by the decode's own ledger, and the isobmff item-bytes share is split the same way. The
number of workers is also bounded by a guess of what a tile needs (three times its planes);
the guess only changes how many workers run, never the result. Heap peak stays under `max_alloc`
for every thread count.

Overlays (`iovl`) and identity items (`iden`) decode their inputs one after the other on the
calling thread.

Known deviation from the conventions (lazy shared pool): there is no pool. A grid takes one
`std::thread::scope` per grid, and every grid (each grid input of an overlay, the primary, the
alpha) opens its own, one after the other, never nested (up to 65 536 for one overlay, plus the
alpha's), so the thread count never passes `max_threads`. A pool
would pay only for many small grids; revisit with the decoder's own parallelism.
