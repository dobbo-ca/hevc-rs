# Limits

Every entry point takes `&Limits` (`src/limits.rs`); there is no unlimited call.

| Field | Default | Over it |
|---|---|---|
| `max_alloc` | 1 GiB | `Error::AllocLimit`; `Error::OutOfMemory` if the allocator itself fails |
| `max_threads` | `available_parallelism()` | no error; 0 is treated as 1 |
| `max_width` | 16384 | `Error::WidthLimit` |
| `max_height` | 16384 | `Error::HeightLimit` |
| `max_pixels` | `1 << 28` | `Error::PixelLimit` |
| `max_pictures` | 4096 | `Error::PictureLimit` |
| `max_metadata` | 1 MiB | no error; the SEI message is dropped |

## Charging

`src/ledger.rs` books every buffer that scales with the input before it is allocated, and
releases it when it is dropped. `max_alloc` bounds the peak. A call starts by booking a small
fixed amount for allocations that are not charged one by one, so `max_alloc` below it fails at
once. Allocation uses `try_reserve_exact`; a refusal inside the limit is `Error::OutOfMemory`.

Charged: the unescaped RBSP of each NAL unit, the sample planes and metadata grids of a picture,
the output pictures (planes and their SEI messages), and the buffer of `intra_prefix`. Parameter set NAL units over 1 MiB are
`Error::InvalidData`.

## Metadata

`max_metadata` caps the SEI bytes kept per picture: each kept message counts its payload plus
`size_of::<SeiMessage>()`. A message that would pass the cap is dropped, with no error (later,
smaller messages may still fit). The kept bytes are charged to `max_alloc` when the message is
collected and stay booked with the output picture. `Options::metadata = false` keeps nothing.

## Size checks

`max_width`, `max_height` and `max_pixels` apply to the coded size, when a picture activates an
SPS and in `probe`. `max_pictures` counts pictures started in one call.

## Cancellation

The cancel flag is read before each NAL unit and before each CTU; set means `Error::Cancelled`.

## decode_heic

`decode_heic` keeps its whole peak within `max_alloc`:

- isobmff gets `Limits { max_alloc: max_alloc / 8, max_threads: 1, max_pixels: ours, ..default }`
  twice over, never at once with each other's peak: the parse (its result, the item graph, stays
  alive for the call) and one `item_bytes` at a time (a copy only for an item with several
  extents). Together at most `max_alloc / 4`.
- The other `max_alloc - max_alloc / 4` is shared by the hevc decode of the current item
  (`max_alloc` of that call = what is left after the planes already kept), the assembled output
  planes and the first tile's SEI messages. A decoded tile is alive next to the output canvas;
  the check before the canvas is allocated counts both. `AllocLimit` when it does not fit.
- `max_width`, `max_height` and `max_pixels` apply to every coded item (in the decoder) and to
  the output size of a grid, before any tile is decoded. isobmff's own `max_pixels` is set to
  ours.
- isobmff limit errors other than alloc and pixels (boxes, items, grid tiles, ...) are
  `InvalidData("limit-<field>")`.
- Grid tiles in parallel (`max_threads` above 1): after tile 0 and the canvas, the other
  decodes split `max_alloc - max_alloc / 4 - held` (`held` = canvas, first tile's SEI, a small
  result table of 24 bytes per tile) equally between the workers, and the isobmff item-bytes
  share (`max_alloc / 8`) the same way, so the peak of all threads together is still within
  `max_alloc`. A tile that does not fit its share is decoded again alone with all of
  `max_alloc - max_alloc / 4 - held`: the result and the error never depend on `max_threads`.
  The decoder inside a worker runs with `max_threads = 1`.
- `iovl`: `max_width`, `max_height` and `max_pixels` are checked against the overlay output
  size before any input is decoded; the canvas (all planes) is charged before it is allocated,
  with the first input and its SEI still alive; each later input is decoded with what is left,
  one at a time.
