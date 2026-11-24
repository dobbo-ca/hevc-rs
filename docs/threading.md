# Threading

`Limits::max_threads` (default `available_parallelism`; 0 counts as 1) caps the threads that work
for one call at the same moment, the calling thread included. With 1 no thread is spawned and the
shared pool is not touched. The output and the errors are byte-identical to the serial path for
every value; `oracle --threads 1` against the default is a parity gate (see [parity.md](parity.md)).

Three things run in parallel, from the outside in:

1. The tiles of a HEIC grid ([HEIC grids](#heic-grids)).
2. The pictures of one call that holds several ([Pictures in parallel](#pictures-in-parallel)).
3. Deblocking and SAO inside one picture, in bands of CTB rows, as tasks on the shared pool
   ([Band-parallel filters](#band-parallel-filters)).

Reconstruction (CABAC, prediction, transform) of one picture is serial, as are the hash check and
the crop. [How the threads are split](#how-the-threads-are-split) says how the three share
`max_threads`.

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

## Pictures in parallel

All pictures are intra, so they share no samples. A call is split in two parts.

- Front end (the calling thread): the NAL loop, parameter sets, slice headers, POC, the
  DPB / C.5.2 bumping, SEI. A slice segment NAL is parsed and queued in the current picture
  (`Pending`: header, rbsp, charged to the ledger while queued).
- Job (`decode_picture`, any thread): slice data, coverage check, deblocking, SAO, hash check
  (only with `Options::check_hash`), crop to the output picture. One job per picture, including
  pictures with `pic_output_flag = 0` (their errors still count).

With `max_threads = 1` the job runs inline when the picture is complete and no scope exists.
With more, one `std::thread::scope` per call holds the workers. The channels are made when the
first slice of a second picture has a good header and the picture limit allows it, so a call with
one picture, or with one picture and then a P slice (the first frame of a video), spawns nothing.
The cap is `min(max_threads, 1 + free_budget / (2 * frame_bytes)) - 1` workers, and they start
one at a time: a job that finds no worker free starts one more until the cap, so a two-picture call
starts one, or two if the first job is still running when the second picture is complete. Jobs go over a `sync_channel(0)` to workers that share a `Mutex<Receiver>`; results
(with the spent picture) return on an `mpsc` channel. When no worker is free and the cap is
reached, the calling thread decodes the job itself, so there are never more than `max_threads`
threads and at most that many pictures being decoded.

Output order: every picture gets a ticket in decode order. The DPB holds tickets; a bump moves
a ticket to an output queue, and a result moves to `out` only when it is at the head of that
queue and has arrived. A picture discarded by `no_output_of_prior_pics` is released after the
jobs in flight have finished (the front end waits for them).


## Errors and determinism

- A job fails (other than `Cancelled`), or `AllocLimit` / `OutOfMemory` happens anywhere while
  workers exist: the workers are drained, `out` is cut back to its length at entry, the ledger
  is dropped and the whole call runs again with `max_threads = 1`. The serial run decides the
  result, so it is exact by construction. A corrupt stream pays twice, and the replay starts
  after the jobs in flight have ended (at most one picture per worker, in parallel). The memory
  check is one-sided: at every job the parallel run books at least what the serial run books
  at the same job, so it cannot succeed where the serial run hits `max_alloc`. Three rules keep
  this true, because a parallel run would otherwise give memory back earlier than a serial one:
  - A finished picture (`Cur`: frame, tables, slice segments, rbsp, SEI) goes back to the front
    end with its result and is dropped in ticket order (`Decoder::retire`), not when its job ends.
  - A release of the front end (replaced parameter set, dropped SEI, a failed activation)
    made while jobs are in flight waits until every picture submitted before it is retired
    (`Decoder::release`, `late`).
  - The discard of the DPB (`no_output_of_prior_pics`) and a front-end error wait for the jobs
    in flight first (`Decoder::discard_all`, `Decoder::fail`), so the buffers of the failing NAL
    unit are given back after them.
  The parallel run also holds the pictures finished out of order a little longer than needed;
  that costs memory, never a different result.
- An error in the front end (for example `InterSlice` after the intra prefix): the submitted
  jobs finish first (a failure there is the case above), then the received slice segments of
  the partial picture are decoded inline (an error there replaces the front-end error, as
  eager decoding would have raised it first), the finished pictures are output, and the error
  is returned. No replay: decoding the first frame of a video does not decode twice.
- `Cancelled`: the workers are drained, the pictures that are ready and in order are output,
  `Cancelled` is returned. `out` is then a prefix of the serial `out`.


## Band-parallel filters

After a picture is reconstructed, deblocking and SAO split each plane into bands of whole CTB rows
(`frame::par_bands`):

- Deblocking: the vertical-edge pass, then the horizontal-edge pass. A vertical edge only touches
  samples of its own row. In the horizontal pass the band that owns the edges of CTB rows
  `[B, B')` is the slice `[B - 4, B' - 4)` of the plane (chroma too): an edge reads and writes at
  most 4 rows above and 3 below it (luma) or 2 above and 1 below (chroma), and edges are
  independent of each other, so the result does not depend on the split. Every pass ends when its
  bands have joined, before the next one starts.
- SAO: each plane is read from the deblocked copy (charged as before) and written in bands of
  whole CTB rows; the neighbour samples across a band edge come from the shared copy.

The number of bands is `min(workers, CTB rows, luma samples >> 16, 64)`, at least 1 (`workers`:
`max_threads`). One band per 64 Ki luma samples: pictures under 128 Ki luma samples (256x256,
352x288, 416x240) filter on the thread that decodes them. The cap of 64 keeps the per-band
bookkeeping (a few KiB) inside the small-allocation reserve. Band tests run 1, 2, 3, 5 and 64
bands against one (SAO on every SIMD tier; deblocking on the tier of the process, and
`deblock::tests::tiers_match_scalar` compares every tier with scalar).

The bands of one pass (`frame::par_bands`) run on the thread that filters the picture and on the
shared pool. Band 0 runs on the filtering thread; the other bands are taken in order from an atomic
counter by the filtering thread and by up to `bands - 1` pool tasks, one for each thread the pass
can take from the call's budget (below). With no thread left in the budget, or no pool, the
filtering thread runs every band. The pass ends when every band is done (`in_place_scope`), and
its threads go back to the budget. The pool tasks borrow the planes for the length of the pass
only, which is what `std::thread::scope` allowed before, but no thread is started per pass.

## The shared pool

`pool::get` makes one `rayon_core::ThreadPool` on first use (a `OnceLock`) with
`available_parallelism` threads named `hevc-pool-N`, 512 KiB stack each, and keeps it for the life
of the process. It holds no other state. A call with `max_threads = 1` never makes or uses it.
There is no API for a caller-supplied pool. If the pool cannot be made, the bands run on the
filtering thread.

Only band tasks run on the pool: they are short and never block. Picture workers and grid workers
stay threads started per call (one `std::thread::scope` per call, or per grid): a picture worker
waits on the job channel for the whole call, and a blocked pool thread would be lost to every
other call; a grid worker decodes whole tiles. They start once per call, not once per pass.

## How the threads are split

`max_threads` counts the threads that work for the call at the same moment: the calling thread,
its picture workers or grid workers, and its band tasks that hold a pool thread. Idle pool
threads are not counted; they are alive whether or not a call runs (at most
`available_parallelism` of them in the process).

- Each call has a budget of `max_threads - 1` threads (`pool::Budget`, an atomic counter). A picture
  worker takes one when it starts and keeps it until the call ends. A band task takes one when its
  pass starts and gives it back when the pass ends. When the budget is empty, no worker starts
  (the job runs on the calling thread, as when every worker is busy) and the bands run on the
  filtering thread.
- Every job, on a picture worker or on the calling thread, splits the filters into up to
  `max_threads` bands by the size gate above; the budget decides how many threads they get. Two
  large pictures with `max_threads` 16 start 2 workers, and their bands use the 13 threads left.
- A tile decoded by a grid worker (two or more running) gets `max_threads = 1`: one thread, one
  band, no pool. Tile 0, a tile decoded again alone, and the tiles of a grid decoded serially get
  the full `max_threads`, because they run alone.

The split never changes a result: the picture and band paths are byte-identical for every thread
count (`tests/threads.rs`, unit tests). `frame::tests::par_bands_runs_at_most_max_threads_at_once`
counts the bands that run at the same moment against `max_threads`. The integration tests build
real streams with a small test encoder (`tests/common`); `tests/limits.rs` checks that
`max_threads = 1` never makes the pool, that the pool has at most `available_parallelism` threads,
that a one-picture call starts no thread outside the pool, that the threads outside the pool stay
within `max_threads`, and the heap peak against `max_alloc` while replaced parameter sets are held
by pictures in flight.


## Memory and cancel

- `max_alloc` is the peak of the whole call across threads: the frame, the pending slices, the
  SAO copy and the output picture of every job are charged before they are allocated, and so is
  the box a picture travels in (to a worker, back on the results channel, waiting in `spent`;
  by value, a channel block of 31 results or a map node of 11 would hold that many pictures of
  state uncharged), and its entry in `spent` (`SPENT_ENTRY`). The slice records of a picture
  (`Cur::slices`, `Frame::slices`) are booked by capacity (`ledger::grow_one`), not by element,
  and the copy of a PPS that `activate` makes for a Main profile is booked in
  `Active::charged`. A picture gives its bytes back after its buffers are freed (`Cur::booked`
  is its last field), so a worker cannot reserve memory that is still in use. The CTU scratch
  lives on the stack of the decoding thread.
- The bands share the planes and the one deblocked copy that SAO already charged, so band tasks
  add no buffer to `max_alloc`. A band can only fail with `Cancelled` (every other error is raised
  before the split); all bands run to the end and the error of the lowest band is returned. A
  band that panics fails with `InvalidData`.
- Thread stacks are not charged to `max_alloc` (pool threads 512 KiB, `pool::STACK`; picture and
  grid workers the std default). Nor is the pool's own memory: its thread records and queues, made
  once per process, and a few KiB that a pool thread allocates the first time it runs a task. The
  box rayon makes for each band task is a small fixed allocation, like the thread handles before.
- Cancel: the flag stops every thread. Every band reads it once per CTB row (SAO) or per 16 rows
  of vertical edges and per row of horizontal edges (deblocking); the planes are put back into
  the frame on every path. A panic inside a
  picture job is caught in the worker and becomes an error result (then the replay), so the
  calling thread never waits for a result that cannot come.
- A call joins its picture workers and grid workers by handle before it returns, and every filter
  pass waits for its band tasks, so no work of the call outlives it. The pool's threads stay, idle.

## Known deviations from the repository threading rule

- There is no single scope per call for everything. Picture jobs own their data (`Cur`), so they
  run on workers started inside one scope per call; the bands run on the shared pool (see above).
  The `>> 16` gate keeps small pictures on one thread; its value is a guess to tune from the
  bench. A HEIC grid opens one scope per grid, and every grid (each grid input of an overlay, the
  primary, the alpha) opens its own, one after the other, never nested (up to 65 536 for one
  overlay, plus the alpha's), so the thread count never passes `max_threads`.
- Small inputs use fewer workers (no picture worker for one picture, one more per job that finds
  none free, one band per 64 Ki luma samples).
- Thread stacks and the pool's own memory are not charged to `max_alloc`.
- A replay waits for the jobs in flight to end. They watch the caller's `cancel` flag only: a
  second, call-local stop flag would need every stage (CTU loop, deblocking, SAO, crop, hash) to
  read two flags, and a stop flag bridged by the front end would delay the caller's own cancel by
  a whole picture while the front end runs a job. The cost is at most one picture per worker, in
  parallel, on a stream that is corrupt anyway (it then pays the replay).
- Not parallel: tiles and wavefront rows inside one picture. They would keep the same rule:
  byte-identical output for every thread count.
