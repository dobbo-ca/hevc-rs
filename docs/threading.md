# Threading

`Limits::max_threads` is honoured (0 counts as 1): the most OS threads of one call, the calling
thread included. 1 spawns none.

Done: `decode_heic` and `decode_heic_item` decode the tiles of a grid in parallel. Tile 0 is
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

Planned: parallel decode by tile and by wavefront row inside one picture, and parallel deblocking
and SAO. The output and the errors will be byte-identical to the serial path; `oracle --threads 1`
against the default is a parity gate (see [parity.md](parity.md)).
