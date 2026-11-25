# Threading

`Limits::max_threads` (default `available_parallelism`; 0 counts as 1) caps the threads of one
call, the calling thread included. With 1 no thread is spawned. The output and the errors are
byte-identical to the serial path for every value; `oracle --threads 1` against the default is a
parity gate (see [parity.md](parity.md)).

Three things run in parallel, from the outside in:

1. The tiles of a HEIC grid ([HEIC grids](#heic-grids)).
2. The pictures of one call that holds several ([Pictures in parallel](#pictures-in-parallel)).
3. Deblocking and SAO inside one picture, in bands of CTB rows
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
  most 4 rows above and 3 below it, p3..q3, in both planes (the SIMD kernels load and store all 8
  rows, also for chroma), and edges are independent of each other, so the result does not depend
  on the split. Every pass ends when its bands have joined, before the next one starts.
- SAO: each plane is read from the deblocked copy (charged as before) and written in bands of
  whole CTB rows; the neighbour samples across a band edge come from the shared copy.

The number of bands is `min(workers, CTB rows, luma samples >> 16, 64)`, at least 1 (`workers`: the
threads the picture may use, see below). One band per 64 Ki luma samples: pictures under 128 Ki
luma samples (256x256, 352x288, 416x240) filter on the calling thread. The cap of 64 keeps the
per-band bookkeeping (a few KiB) inside the small-allocation reserve. Band 0 runs on the thread
that filters the picture. Band tests run 1, 2, 3, 5 and 64 bands against one (SAO on every SIMD
tier; deblocking on the tier of the process, and `deblock::tests::tiers_match_scalar` compares every
tier with scalar).

## How the threads are split

`max_threads` is the total of live threads of the call, the band threads included.

- A job on a picture worker filters with 1 band.
- A job the calling thread runs while no picture worker exists (a call with one picture, such as a
  HEIC item or a single still) filters with up to `max_threads` bands.
- Once picture workers exist, a job the calling thread runs uses up to
  `max(1, max_threads - spawned)` bands, where `spawned` is the number of picture workers started.
  The calling thread, its `bands - 1` band threads and the `spawned` workers (one band each) are at
  most `max_threads` threads. While the calling thread runs a job its front end waits,
  so the count holds at every moment.
- A tile decoded by a grid worker (two or more running) gets `max_threads = 1`: one thread, one band.
  Tile 0, a tile decoded again alone, and the tiles of a grid decoded serially get the full
  `max_threads`, because they run alone.

The split never changes a result: the picture and band paths are byte-identical for every thread
count (`tests/threads.rs`, unit tests). The integration tests build real streams with a small test
encoder (`tests/common`); `tests/limits.rs` also checks that a 512x512 picture runs band threads,
and the heap peak against `max_alloc` while replaced parameter sets are held by pictures in flight.


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
- The bands share the planes and the one deblocked copy that SAO already charged, so band threads
  add no buffer to `max_alloc`. A band can only fail with `Cancelled` (every other error is raised
  before the split); all bands run to the end and the error of the lowest band is returned. A
  refused spawn runs that band on the calling thread.
- Thread stacks are not charged to `max_alloc` (band threads 512 KiB, `frame::BAND_STACK`; picture and grid workers the std default). The README says so in "Resource limits".
- Cancel: the flag stops every thread. Every band reads it once per CTB row (SAO) or per 16 rows
  of vertical edges and per row of horizontal edges (deblocking); the planes are put back into
  the frame on every path. A panic inside a
  picture job is caught in the worker and becomes an error result (then the replay), so the
  calling thread never waits for a result that cannot come.
- A call joins its picture workers, band threads and grid workers by handle before it returns, so no thread of the call outlives it.

## Known deviations from the repository threading rule

- There is no lazy shared pool (`OnceLock`) and no single scope per call for the filters. A
  reused thread cannot borrow a per-picture `Frame` in safe Rust, and a pool crate (tried below)
  gave no clear gain. Picture jobs own their data (`Cur`), so they run on workers started inside one scope
  per call. The filters borrow the planes, so each filter pass opens its own scope (up to 6 per
  picture for deblocking and 3 for SAO), each spawning `bands - 1` threads. The `>> 16` gate keeps
  small pictures serial; its value is a guess to tune from the bench. A HEIC grid opens one scope
  per grid, and every grid (each grid input of an overlay, the primary, the alpha) opens its own,
  one after the other, never nested (up to 65 536 for one overlay, plus the alpha's), so the
  thread count never passes `max_threads`.
- Small inputs use fewer workers (no picture worker for one picture, one more per job that finds
  none free, one band per 64 Ki luma samples) instead of a pool.
- Thread stacks are not charged to `max_alloc`.
- A shared pool was tried and dropped (PR #27, branch `sp-pool-w9t5`, 2026-11-24; decision
  of that day: a pool crate only with a measured gain over `std::thread::scope`). The deblocking
  and SAO bands ran as tasks on one lazy `rayon-core` pool (a `OnceLock`, `available_parallelism`
  threads, 512 KiB stacks, made only when a pass had a thread for it, never with
  `max_threads = 1`), with a thread budget per call: an atomic counter of `max_threads - 1`, from
  which a picture worker took one thread for the call and a band task one for its pass, so the
  threads working for a call at once stayed within `max_threads` (idle pool threads not counted).
  The bands borrowed the planes through `in_place_scope`, so no thread was started per pass.
  Picture and grid workers stayed threads started per call: a picture worker blocks on the job
  channel, and a blocked pool thread would be lost to every other call. Output stayed
  byte-identical (test job green). Bench, default threads, best of 3, 16 CPUs, ms; ratio is
  hevc-rs / reference (dec265 for Annex B, libheif for HEIC), against two runs of main dcdc55d
  (45496374688, 45507433964) and two of the pool (45505669129, 45506541385, commit 48708ec):

  | input set | build | main ms | main ratio | pool ms | pool ratio |
  |---|---|---|---|---|---|
  | all 149, nt | default | 1474.3 / 1537.6 | 0.701 / 0.692 | 1445.7 / 1554.2 | 0.685 / 0.687 |
  | all 149, nt | v4 | 1346.7 / 1397.8 | 0.641 / 0.629 | 1303.8 / 1390.3 | 0.617 / 0.615 |
  | main 135, nt | default | 1438.1 / 1500.7 | 0.702 / 0.693 | 1409.4 / 1516.2 | 0.685 / 0.688 |
  | heic 60, nt | default | 2376.1 / 2154.7 | 0.372 / 0.332 | 2051.2 / 2003.2 | 0.316 / 0.316 |
  | heic 60, nt | v4 | 2649.6 / 1937.6 | 0.415 / 0.299 | 1948.7 / 1875.7 | 0.300 / 0.296 |
  | all 149, 1t (pool unused) | default | 2606.8 / 2671.5 | 0.767 / 0.766 | 2594.7 / 2692.4 | 0.763 / 0.769 |
  | heic 60, 1t (pool unused) | default | 2390.8 / 2336.5 | 0.362 / 0.354 | 2254.6 / 2213.0 | 0.343 / 0.340 |

  The pool runs beat both main runs by 1 to 2.5 points at default threads, about the spread
  between the two main runs themselves (0.9 to 1.2 points on Annex B, 4 to 12 on HEIC), and the
  HEIC 1-thread ratio, which the pool cannot touch, moved by 1.5 to 2 points too. Not clearly
  outside noise, so the crate stays on `std::thread::scope` with no new dependency. Each bench
  input is its own process, so every file paid the pool's start (16 threads); a long-lived caller
  would pay it once, which this bench does not measure. Two variants lost outright:
  - Bands for jobs on picture workers too (commit 7c3f61e, run 45503067302): the default-thread
    total of the 149 Annex B inputs rose from 1474.3 to 1571.0 ms (v4 1346.7 to 1446.3) while
    the 1-thread total moved from 2606.8 to 2691.6 ms (a slower host), and multi-picture streams
    lost most (OPFLAG_B 88.4 to 114.2 ms, IPRED_A 18.4 to 24.5 ms): bands on a worker held the
    budget threads that the next picture needed for its own worker, so that picture ran on the
    calling thread and the front end stalled.
  - One band per 16 Ki luma samples instead of 64 Ki (commit 93db140, run 45509186196): Annex B
    ratio at default threads 0.690 (v4 0.625), no better than the 64 Ki gate on the pool, and
    main10 at default threads took 24.7 ms (23.0 to 23.7 ms in the other runs): small pictures pay for
    waking pool threads.
- A job on a picture worker always filters with 1 band, even when threads are idle (two large
  pictures with `max_threads` 16 start 2 workers, 14 threads idle); main used up to 16 bands each.
- A replay waits for the jobs in flight to end. They watch the caller's `cancel` flag only: a
  second, call-local stop flag would need every stage (CTU loop, deblocking, SAO, crop, hash) to
  read two flags, and a stop flag bridged by the front end would delay the caller's own cancel by
  a whole picture while the front end runs a job. The cost is at most one picture per worker, in
  parallel, on a stream that is corrupt anyway (it then pays the replay).
- Not parallel: tiles and wavefront rows inside one picture. They would keep the same rule:
  byte-identical output for every thread count.
