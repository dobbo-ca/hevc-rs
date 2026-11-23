//! Caps on one call.

/// Caps on one call. Every entry point takes `&Limits`; the decode entry points also take a
/// cancel flag. The defaults are finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Peak working memory of one call, in bytes, all threads together (default 1 GiB). Every
    /// buffer that scales with the input is charged before it is allocated and released when it
    /// is dropped. More is [`Error::AllocLimit`](crate::Error::AllocLimit).
    pub max_alloc: usize,
    /// Most OS threads of one call, the calling thread included (default
    /// `std::thread::available_parallelism`). 1 spawns no thread; 0 is treated as 1.
    pub max_threads: usize,
    /// Most luma samples per row of the coded picture, `pic_width_in_luma_samples` (default
    /// 16384). More is [`Error::WidthLimit`](crate::Error::WidthLimit).
    pub max_width: u32,
    /// Most luma rows of the coded picture (default 16384). More is
    /// [`Error::HeightLimit`](crate::Error::HeightLimit).
    pub max_height: u32,
    /// Most `width * height` luma samples of the coded picture (default `1 << 28`). More is
    /// [`Error::PixelLimit`](crate::Error::PixelLimit).
    pub max_pixels: u64,
    /// Most pictures decoded in one call (default 4096). More is
    /// [`Error::PictureLimit`](crate::Error::PictureLimit).
    pub max_pictures: usize,
    /// Most bytes of SEI messages kept per picture (default 1 MiB): the payload bytes plus
    /// `size_of::<SeiMessage>()` for each message. A message that would pass it is dropped, with
    /// no error. The kept bytes are charged to `max_alloc` like the planes.
    pub max_metadata: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_alloc: 1 << 30,
            max_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_width: 16384,
            max_height: 16384,
            max_pixels: 1 << 28,
            max_pictures: 4096,
            max_metadata: 1 << 20,
        }
    }
}
