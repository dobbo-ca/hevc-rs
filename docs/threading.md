# Threading

Decoding is serial in v1. `Limits::max_threads` is honoured: no thread is spawned for any value
(0 counts as 1), and there is no thread pool.

Planned: parallel decode by tile and by wavefront row, and parallel deblocking and SAO. The output
and the errors will be byte-identical to the serial path; `oracle --threads 1` against the default
is a parity gate (see [parity.md](parity.md)).
