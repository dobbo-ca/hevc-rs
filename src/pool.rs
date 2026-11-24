//! The lazy thread pool shared by every call, and the thread budget of one call
//! (docs/threading.md).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Stack of a pool thread: the band filters do not recurse, so far less than the 2 MiB default.
const STACK: usize = 512 * 1024;

static POOL: OnceLock<Option<rayon_core::ThreadPool>> = OnceLock::new();

/// The pool, made on first use with `available_parallelism` threads and kept for the life of the
/// process; `None` when it cannot be made (the work then runs on the calling thread).
pub(crate) fn get() -> Option<&'static rayon_core::ThreadPool> {
    POOL.get_or_init(|| {
        rayon_core::ThreadPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get()))
            .thread_name(|i| format!("hevc-pool-{i}"))
            .stack_size(STACK)
            .build()
            .ok()
    })
    .as_ref()
}

/// The threads one call may still add to the calling thread: `max_threads - 1` at the start.
/// A picture worker holds one for the rest of the call, a band task on the pool holds one until
/// its pass ends.
pub(crate) struct Budget(AtomicUsize);

impl Budget {
    pub(crate) const fn new(max_threads: usize) -> Self {
        Self(AtomicUsize::new(max_threads.saturating_sub(1)))
    }

    /// Takes one thread; false when none is left.
    pub(crate) fn take(&self) -> bool {
        self.0
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_ok()
    }

    /// Gives `n` threads back.
    pub(crate) fn give(&self, n: usize) {
        self.0.fetch_add(n, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_counts_extra_threads() {
        for max_threads in [0, 1] {
            assert!(!Budget::new(max_threads).take());
        }
        let b = Budget::new(3);
        assert!(b.take() && b.take() && !b.take());
        b.give(1);
        assert!(b.take() && !b.take());
    }
}
