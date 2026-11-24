//! The working memory of one call: every buffer that scales with the input is charged before it
//! is allocated, by all threads of the call together, and released when its guard drops.

use crate::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};

/// Booked at the start of every call for the small, short-lived allocations that do not scale
/// with the input and are not charged one by one: the iterator state of the NAL loop and the NAL
/// list of an `hvcC` record (at most 1024 slices of 16 bytes). The `Pps` copy that `ps::activate`
/// makes per picture is charged (`Active::charged`), and a parameter set replaced while a picture
/// may share it stays booked with that picture (`Cur::booked`). 64 KiB covers the rest. The
/// parameter set store, the output list and the pictures are charged one by one. A `max_alloc`
/// below this is `Error::AllocLimit` at once.
///
/// With threads, the bookkeeping of the front end is also not charged and also covered here:
/// the maps and queues of the tickets (`done`, `spent`, `late`, `order`; at most the pictures in
/// flight and in the DPB, and every output picture in `done` is charged for far more than its
/// map node), the channels (the results channel keeps blocks of 31 slots of a few hundred bytes:
/// a picture travels boxed, and its box is charged with it) and the `Arc`s and `Box`es of a
/// worker (a few hundred bytes each, against a thread stack of megabytes that is not charged
/// either). `BTreeMap` cannot report a failed allocation, so these stay small on purpose.
pub(crate) const SMALL: usize = 64 << 10;

/// Bytes booked against `Limits::max_alloc`.
#[derive(Debug, Default)]
pub(crate) struct Ledger {
    max: AtomicU64,
    used: AtomicU64,
    peak: AtomicU64,
}

impl Ledger {
    /// Starts a call: `max` bytes allowed, `preload` of them already taken by the caller, plus
    /// [`SMALL`] for the small allocations that are not charged one by one.
    pub(crate) fn reset(&self, max: usize, preload: usize) -> Result<(), Error> {
        self.max.store(max as u64, Ordering::Relaxed);
        self.used.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.reserve(preload.checked_add(SMALL).ok_or(Error::AllocLimit)?)
    }

    /// What may still be charged.
    pub(crate) fn left(&self) -> usize {
        let free = self
            .max
            .load(Ordering::Relaxed)
            .saturating_sub(self.used.load(Ordering::Relaxed));
        usize::try_from(free).unwrap_or(usize::MAX)
    }

    /// The most bytes booked at once in this call.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Books `n` bytes for the rest of the call.
    pub(crate) fn reserve(&self, n: usize) -> Result<(), Error> {
        let n = n as u64;
        let max = self.max.load(Ordering::Relaxed);
        let r = self
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |u| {
                u.checked_add(n).filter(|&t| t <= max)
            });
        match r {
            Ok(u) => {
                self.peak.fetch_max(u + n, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => Err(Error::AllocLimit),
        }
    }

    /// Books `n` bytes until the returned guard drops.
    pub(crate) fn charge(&self, n: usize) -> Result<Held<'_>, Error> {
        self.reserve(n)?;
        Ok(Held { l: self, n })
    }

    /// Gives back `n` bytes booked with [`Ledger::reserve`], for a buffer that was freed.
    pub(crate) fn release(&self, n: usize) {
        self.used.fetch_sub(n as u64, Ordering::AcqRel);
    }
}

/// Bytes booked in a [`Ledger`], released on drop.
#[derive(Debug)]
pub(crate) struct Held<'a> {
    l: &'a Ledger,
    n: usize,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.l.release(self.n);
    }
}

impl Held<'_> {
    /// Keeps the bytes booked (the caller now owns their release) and returns their number.
    pub(crate) fn keep(self) -> usize {
        let n = self.n;
        std::mem::forget(self);
        n
    }
}

/// Makes room for one more element in `v`, booking the real capacity growth (the `Vec` grows to
/// `max(4, 2 * capacity)`, not by one element). Returns the bytes booked, `0` when `v` has room.
pub(crate) fn grow_one<T>(v: &mut Vec<T>, ledger: &Ledger) -> Result<usize, Error> {
    if v.len() < v.capacity() {
        return Ok(0);
    }
    let add = v.capacity().checked_mul(2).ok_or(Error::AllocLimit)?.max(4) - v.len();
    let n = add.checked_mul(size_of::<T>()).ok_or(Error::AllocLimit)?;
    ledger.reserve(n)?;
    if v.try_reserve_exact(add).is_err() {
        ledger.release(n);
        return Err(Error::OutOfMemory);
    }
    Ok(n)
}

/// Sets `v` to `n` zero bytes and books `n` bytes for the rest of the call. A buffer that is
/// already big enough (a `Decoder` kept it) allocates nothing.
#[cfg(test)]
pub(crate) fn zeroed(v: &mut Vec<u8>, n: usize, ledger: &Ledger) -> Result<(), Error> {
    ledger.reserve(n)?;
    v.clear();
    if n > v.capacity() {
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    }
    v.resize(n, 0);
    Ok(())
}

/// A zeroed `Vec<u16>` of `len` samples (a picture plane), booked for the rest of the call.
#[cfg(test)]
pub(crate) fn alloc_u16_zeroed(len: usize, ledger: &Ledger) -> Result<Vec<u16>, Error> {
    ledger.reserve(len.checked_mul(2).ok_or(Error::AllocLimit)?)?;
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| Error::OutOfMemory)?;
    v.resize(len, 0);
    Ok(v)
}

/// A `Vec` of `len` copies of `v`. Does not charge: the caller has already reserved
/// `len * size_of::<T>()` bytes in the ledger (so one reservation can cover several buffers).
pub(crate) fn filled<T: Clone>(len: usize, v: T) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| Error::OutOfMemory)?;
    out.resize(len, v);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_release_and_peak() {
        let l = Ledger::default();
        l.reset(SMALL + 100, 0).unwrap();
        let a = l.charge(60).unwrap();
        assert_eq!(l.charge(50).unwrap_err(), Error::AllocLimit);
        assert_eq!(l.left(), 40);
        drop(a);
        assert_eq!(l.left(), 100);
        let _b = l.charge(100).unwrap();
        assert_eq!(l.peak(), SMALL as u64 + 100);
    }

    #[test]
    fn grow_one_books_the_capacity_growth() {
        let l = Ledger::default();
        l.reset(SMALL + 200, 0).unwrap();
        let mut v: Vec<u64> = Vec::new();
        let mut booked = 0;
        for i in 0..9 {
            booked += grow_one(&mut v, &l).unwrap();
            v.push(i);
            assert_eq!(booked, v.capacity() * 8);
            assert_eq!(l.left(), 200 - booked);
        }
        // A refusal books nothing and leaves the Vec as it was.
        let cap = v.capacity();
        while v.len() < cap {
            v.push(0);
        }
        assert_eq!(grow_one(&mut v, &l).unwrap_err(), Error::AllocLimit);
        assert_eq!((v.capacity(), l.left()), (cap, 200 - booked));
        // A `Held` that is kept stays booked.
        let n = l.charge(10).unwrap().keep();
        assert_eq!((n, l.left()), (10, 200 - booked - 10));
    }

    #[test]
    fn u16_plane_is_charged_in_bytes() {
        let l = Ledger::default();
        l.reset(SMALL + 10, 0).unwrap();
        assert_eq!(alloc_u16_zeroed(6, &l).unwrap_err(), Error::AllocLimit);
        assert_eq!(
            alloc_u16_zeroed(usize::MAX, &l).unwrap_err(),
            Error::AllocLimit
        );
        assert_eq!(alloc_u16_zeroed(5, &l).unwrap(), [0u16; 5]);
        assert_eq!(l.left(), 0);
    }

    #[test]
    fn reset_preload_and_zeroed() {
        let l = Ledger::default();
        assert_eq!(l.reset(SMALL + 10, 11).unwrap_err(), Error::AllocLimit);
        l.reset(SMALL + 10, 4).unwrap();
        assert_eq!(l.left(), 6);
        let mut v = Vec::new();
        assert_eq!(zeroed(&mut v, 7, &l).unwrap_err(), Error::AllocLimit);
        zeroed(&mut v, 6, &l).unwrap();
        assert_eq!(v, [0; 6]);
        assert_eq!(l.left(), 0);
        // A new call with the buffer kept: no allocation, same charge.
        l.reset(SMALL + 10, 4).unwrap();
        let cap = v.capacity();
        zeroed(&mut v, 6, &l).unwrap();
        assert_eq!(v.capacity(), cap);
        assert_eq!(l.left(), 0);
    }
}
