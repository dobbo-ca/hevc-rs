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

## Charging

`src/ledger.rs` books every buffer that scales with the input before it is allocated, and
releases it when it is dropped. `max_alloc` bounds the peak. A call starts by booking a small
fixed amount for allocations that are not charged one by one, so `max_alloc` below it fails at
once. Allocation uses `try_reserve_exact`; a refusal inside the limit is `Error::OutOfMemory`.

Charged: the unescaped RBSP of each NAL unit, the sample planes and metadata grids of a picture,
the output pictures, and the buffer of `intra_prefix`. Parameter set NAL units over 1 MiB are
`Error::InvalidData`.

## Size checks

`max_width`, `max_height` and `max_pixels` apply to the coded size, when a picture activates an
SPS and in `probe`. `max_pictures` counts pictures started in one call.

## Cancellation

The cancel flag is read before each NAL unit and before each CTU; set means `Error::Cancelled`.
