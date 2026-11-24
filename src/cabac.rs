//! CABAC arithmetic decoding engine (9.3.4.3) and context variable initialisation (9.3.2.2) for
//! I slices (initType 0).
//!
//! The engine is the spec model (`ivlCurrRange` 9 bits, `ivlOffset` 9 bits at initialisation and
//! one more bit per renormalisation step, over a [`BitReader`] on the slice segment RBSP) with
//! the offset kept scaled in a 64-bit register that is refilled 32 bits at a time. The spec bit
//! position stays exact, and so does the point where reading past the end of the data fails.
//! After a terminate bin equal to 1 the reader has consumed every bit the encoder flushed
//! (9.3.5.6), including the final 1 bit (rbsp_stop_one_bit or alignment_bit_equal_to_one). The
//! caller then reads the pcm_alignment_zero_bits / byte_alignment zero bits through
//! [`Cabac::reader`], reads raw PCM samples the same way and calls [`Cabac::reinit`] (9.3.2.6)
//! to restart the engine. The straightforward one-bit-at-a-time engine lives on in the tests as
//! the reference.

use crate::bits::BitReader;
use crate::error::{Error, Result};

/// Table 9-52: rangeTabLps[pStateIdx][qRangeIdx].
const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

/// Table 9-53, transIdxLps[pStateIdx].
const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, //
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24, //
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, //
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// State transitions of 9.3.4.3.2.2 for the packed context byte `pStateIdx << 1 | valMps`,
/// indexed by `lps << 7 | byte` (`lps` is 1 when the bin was the LPS). transIdxMps is
/// pStateIdx + 1 except 62 -> 62 and 63 -> 63; an LPS in state 0 flips valMps.
#[allow(clippy::indexing_slicing)] // evaluated at compile time
const fn build_next() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut s = 0usize;
    while s < 64 {
        let mut m = 0usize;
        while m < 2 {
            let up = if s < 62 { s + 1 } else { s };
            t[(s << 1) | m] = ((up << 1) | m) as u8;
            let nm = if s == 0 { m ^ 1 } else { m };
            t[128 | (s << 1) | m] = (((TRANS_IDX_LPS[s] as usize) << 1) | nm) as u8;
            m += 1;
        }
        s += 1;
    }
    t
}

static NEXT: [u8; 256] = build_next();

/// Table 9-52 flattened: rangeTabLps[pStateIdx][qRangeIdx] at `pStateIdx << 2 | qRangeIdx`,
/// which is `(byte & 0x7e) << 1 | qRangeIdx` for the packed context byte.
#[allow(clippy::indexing_slicing)] // evaluated at compile time
const fn build_lps() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = RANGE_TAB_LPS[i >> 2][i & 3];
        i += 1;
    }
    t
}

static LPS: [u8; 256] = build_lps();

/// One context variable: `pStateIdx` (0..=62 after initialisation) and `valMps` (0 or 1),
/// packed as `pStateIdx << 1 | valMps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ctx {
    s: u8,
}

impl Ctx {
    /// 9.3.2.2 (9-4) to (9-6): the context for table entry `init_value` and SliceQpY.
    pub(crate) fn init(init_value: u8, slice_qp_y: i32) -> Self {
        let slope_idx = i32::from(init_value >> 4);
        let offset_idx = i32::from(init_value & 15);
        let m = slope_idx * 5 - 45;
        let n = (offset_idx << 3) - 16;
        let pre = (((m * slice_qp_y.clamp(0, 51)) >> 4) + n).clamp(1, 126);
        let mps = pre > 63;
        let state = if mps { pre - 64 } else { 63 - pre };
        Self {
            s: ((state as u8) << 1) | u8::from(mps),
        }
    }

    /// valMps.
    #[cfg(test)]
    fn mps(self) -> bool {
        self.s & 1 == 1
    }

    /// The state transition of 9.3.4.3.2.2 after decoding `bin` (the engine inlines this; the
    /// tests' encoder and reference decoder use it).
    #[cfg(test)]
    fn update(&mut self, bin: bool) {
        let lps = usize::from(bin != self.mps());
        self.s = NEXT
            .get((lps << 7) | usize::from(self.s & 127))
            .copied()
            .unwrap_or(self.s);
    }

    /// rangeTabLps[pStateIdx][qRangeIdx] (9-65), for the tests' encoder and reference decoder.
    #[cfg(test)]
    fn lps_range(self, q_range_idx: u32) -> u32 {
        RANGE_TAB_LPS
            .get(usize::from((self.s >> 1) & 63))
            .and_then(|row| row.get(q_range_idx as usize))
            .map_or(2, |&v| u32::from(v))
    }
}

const TRUNCATED: Error = Error::InvalidData("truncated");

/// The arithmetic decoding engine over a slice segment's RBSP.
///
/// Same arithmetic as the spec model (9.3.4.3), with the 9-bit `ivlOffset` kept scaled inside a
/// wide register: `value = ivlOffset << k | next k bits of the stream`. Comparing the offset
/// with a range is then `value >= range << k`, renormalisation is just `k -= shift` and the
/// bits are fetched 32 at a time. `r` sits at the end of the fetched bits, so the spec's bit
/// position (what the model would have read) is `r.bit_pos() - k`.
///
/// Fetching past the end of the data supplies zero bits; the model would fail at the first
/// renormalisation bit beyond the end, which is detected exactly: the bits fetched beyond the
/// end are `floor`, and `k < floor` means the model has consumed one of them.
pub(crate) struct Cabac<'a> {
    r: BitReader<'a>,
    /// ivlCurrRange, 256..=510 between bins.
    range: u32,
    /// `ivlOffset << k | look-ahead bits`.
    value: u64,
    /// Look-ahead bits in `value`; at least 7 between bins (a renormalisation shifts by at
    /// most 7).
    k: u32,
    /// Bits fetched beyond the end of the data.
    floor: u32,
    /// `max(floor, 7)`: below this `k` needs attention ([`Cabac::settle`]).
    lim: u32,
}

impl<'a> Cabac<'a> {
    /// 9.3.2.6: initialises the engine reading 9 bits from `r`, which must sit at the first bit
    /// of the slice segment data (after byte_alignment()).
    pub(crate) fn new(r: BitReader<'a>) -> Result<Self> {
        let mut c = Self {
            r,
            range: 510,
            value: 0,
            k: 0,
            floor: 0,
            lim: 7,
        };
        c.reinit()?;
        Ok(c)
    }

    /// 9.3.2.6 again: ivlCurrRange = 510, ivlOffset = read_bits(9). For a new substream (tile,
    /// WPP row) after the byte_alignment() and for the data after PCM samples; the reader must
    /// already be at the right bit (see [`Cabac::reader`]).
    pub(crate) fn reinit(&mut self) -> Result<()> {
        self.range = 510;
        let v = self.r.u(9)?;
        if v >= 510 {
            return Err(Error::InvalidData("ivlOffset 510 or 511"));
        }
        self.value = u64::from(v);
        self.k = 0;
        self.refill();
        Ok(())
    }

    /// The underlying reader, for pcm_alignment_zero_bit, PCM samples and byte_alignment()
    /// after a terminate bin equal to 1. It is moved to the spec bit position first and the
    /// look-ahead is dropped, so the engine must be re-initialised with [`Cabac::reinit`]
    /// (or abandoned) before the next bin.
    pub(crate) fn reader(&mut self) -> &mut BitReader<'a> {
        self.r.rewind(self.k as usize);
        self.value >>= self.k;
        self.k = 0;
        self.floor = u32::MAX;
        self.lim = u32::MAX;
        &mut self.r
    }

    /// The spec bit position, for the tests.
    #[cfg(test)]
    pub(crate) fn spec_pos(&self) -> usize {
        self.r.bit_pos() - self.k as usize
    }

    /// Fetches 32 more look-ahead bits. Called with `k < 16`, so `value` (below
    /// `range << k`, 9 bits more than `k`) stays below 2^56.
    fn refill(&mut self) {
        self.value = (self.value << 32) | u64::from(self.r.load32());
        self.k += 32;
        self.floor = u32::try_from(self.r.overrun()).unwrap_or(u32::MAX);
        self.lim = self.floor.max(7);
    }

    /// Slow path after a bin left `k < lim`: the model read a bit beyond the end of the data
    /// (error), or the look-ahead is running low.
    #[cold]
    #[inline(never)]
    fn settle(&mut self) -> Result<()> {
        if self.k < self.floor {
            return Err(TRUNCATED);
        }
        if self.k < 7 {
            self.refill();
        }
        Ok(())
    }

    /// 9.3.4.3.2 DecodeDecision with the context `ctx`, which is updated.
    pub(crate) fn decode(&mut self, ctx: &mut Ctx) -> Result<bool> {
        self.decode_inline(ctx)
    }

    /// [`Cabac::decode`] inlined into the caller, for the per-coefficient loops of
    /// residual_coding. The MPS/LPS outcome selects values instead of branching: that branch
    /// is close to random for sig_coeff_flag and would mispredict often.
    #[inline(always)]
    pub(crate) fn decode_inline(&mut self, ctx: &mut Ctx) -> Result<bool> {
        let s = ctx.s;
        let q = ((self.range >> 6) & 3) as usize;
        let lps = LPS
            .get((usize::from(s & 0x7e) << 1) | q)
            .map_or(2, |&v| u32::from(v));
        // range >= 256 > lps.
        let mps_range = self.range - lps;
        let scaled = u64::from(mps_range) << self.k;
        let is_lps = self.value >= scaled;
        self.value -= scaled & 0u64.wrapping_sub(u64::from(is_lps));
        self.range = if is_lps { lps } else { mps_range };
        ctx.s = NEXT
            .get((usize::from(is_lps) << 7) | usize::from(s & 127))
            .copied()
            .unwrap_or(s);
        // Renormalisation: shift until the range is >= 256 (at most 7 bits).
        let sh = self.range.leading_zeros().saturating_sub(23);
        self.range <<= sh;
        self.k = self.k.saturating_sub(sh);
        if self.k < self.lim {
            self.settle()?;
        }
        Ok((s & 1 == 1) != is_lps)
    }

    /// 9.3.4.3.4 DecodeBypass.
    pub(crate) fn bypass(&mut self) -> Result<bool> {
        self.k = self.k.saturating_sub(1);
        let scaled = u64::from(self.range) << self.k;
        let bin = self.value >= scaled;
        if bin {
            self.value -= scaled;
        }
        if self.k < self.lim {
            self.settle()?;
        }
        Ok(bin)
    }

    /// `n` bypass bins (n <= 32), the first one as the most significant bit.
    pub(crate) fn bypass_bits(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(Error::InvalidData("bypass_bits over 32"));
        }
        Ok(self.bypass_wide(n)? as u32)
    }

    /// `n` bypass bins (n <= 64), the first one as the most significant bit. Up to 16 bins at
    /// a time: the bins are the quotient of the scaled value by `range << (k - m)` (long
    /// division, as the sequential compare-and-subtract of 9.3.4.3.4 does).
    pub(crate) fn bypass_wide(&mut self, n: u32) -> Result<u64> {
        let mut out = 0u64;
        let mut left = n.min(64);
        while left > 0 {
            let m = left.min(16);
            if self.k < m {
                self.refill();
            }
            self.k -= m;
            let scaled = u64::from(self.range) << self.k;
            let q = if m <= 3 {
                let mut q = 0u64;
                let mut sh = m;
                while sh > 0 {
                    sh -= 1;
                    let s = u64::from(self.range) << (self.k + sh);
                    let bit = u64::from(self.value >= s);
                    self.value -= s * bit;
                    q = (q << 1) | bit;
                }
                q
            } else {
                let q = self.value / scaled;
                self.value -= q * scaled;
                q & ((1u64 << m) - 1)
            };
            out = (out << m) | q;
            left -= m;
            if self.k < self.lim {
                self.settle()?;
            }
        }
        Ok(out)
    }

    /// Counts consecutive bypass bins equal to 1 up to the first 0 (consumed). `Some(count)`
    /// with `count <= max`, or `None` when `max + 1` ones were read (the model reads bins
    /// one at a time and gives up at the `max + 1`-th one; no 0 is read then).
    pub(crate) fn bypass_ones(&mut self, max: u32) -> Result<Option<u32>> {
        let mut count = 0u32;
        loop {
            // Bins that may still be read: count <= max here, so at least 1.
            let m = (max + 1 - count).clamp(1, 16);
            if self.k < m {
                self.refill();
            }
            let scaled = u64::from(self.range) << (self.k - m);
            let q = ((self.value / scaled) & ((1u64 << m) - 1)) as u32;
            // Ones at the top of the m-bit group; the zero-filled low bits end the run.
            let lead = (q << (32 - m)).leading_ones();
            let used = if lead < m { lead + 1 } else { m };
            // Commit `used` bins only: the quotient of their prefix.
            self.k -= used;
            let qc = u64::from(q >> (m - used));
            self.value -= qc * (u64::from(self.range) << self.k);
            if self.k < self.lim {
                self.settle()?;
            }
            if lead < m {
                return Ok(Some(count + lead));
            }
            count += m;
            if count > max {
                return Ok(None);
            }
        }
    }

    /// The next 16 bypass bins, the first one as the most significant bit, without consuming
    /// them; [`Cabac::bypass_take`] then consumes a prefix of them.
    #[inline(always)]
    pub(crate) fn bypass_peek16(&mut self) -> u32 {
        if self.k < 16 {
            self.refill();
        }
        let scaled = u64::from(self.range) << (self.k - 16);
        ((self.value / scaled) & 0xffff) as u32
    }

    /// Consumes the first `used` (1..=16) of the bins `q` returned by [`Cabac::bypass_peek16`].
    #[inline(always)]
    pub(crate) fn bypass_take(&mut self, q: u32, used: u32) -> Result<()> {
        self.k -= used;
        self.value -= u64::from(q >> (16 - used)) * (u64::from(self.range) << self.k);
        if self.k < self.lim {
            self.settle()?;
        }
        Ok(())
    }

    /// 9.3.4.3.5 DecodeTerminate (end_of_slice_segment_flag, end_of_subset_one_bit, pcm_flag).
    /// On true there is no renormalisation: the last bit read was the stop / alignment 1 bit.
    pub(crate) fn terminate(&mut self) -> Result<bool> {
        self.range -= 2;
        if self.value >= u64::from(self.range) << self.k {
            return Ok(true);
        }
        if self.range < 256 {
            self.range <<= 1;
            self.k = self.k.saturating_sub(1);
            if self.k < self.lim {
                self.settle()?;
            }
        }
        Ok(false)
    }
}

/// All context variables of a slice segment, initialised for initType 0 (I slices).
///
/// ctxIdx ranges used from Table 9-4 (initType 0 column) and the table holding the values:
/// sao_merge_left/up_flag 0 (9-5); sao_type_idx_luma/chroma 0 (9-6); split_cu_flag 0..2 (9-7);
/// cu_transquant_bypass_flag 0 (9-8); part_mode 0 (9-11, bin 0 only); prev_intra_luma_pred_flag
/// 0 (9-12); intra_chroma_pred_mode 0 (9-13); split_transform_flag 0..2 (9-20); cbf_luma 0..1
/// (9-21); cbf_cb/cbf_cr 0..3 and 12 (9-22); cu_qp_delta_abs 0..1 (9-24); transform_skip_flag
/// luma 0 and chroma 3 (9-25); last_sig_coeff_x/y_prefix 0..17 (9-26, 9-27);
/// coded_sub_block_flag 0..3 (9-28); sig_coeff_flag 0..41 and 126..127 (9-29);
/// coeff_abs_level_greater1_flag 0..23 (9-30); coeff_abs_level_greater2_flag 0..5 (9-31).
///
/// Array indexes are the ctxInc of 9.3.4.2 (Table 9-41); callers index with `get_mut`.
#[derive(Debug, Clone)]
pub(crate) struct Contexts {
    /// Shared by sao_merge_left_flag and sao_merge_up_flag.
    pub sao_merge: Ctx,
    /// Shared by sao_type_idx_luma and sao_type_idx_chroma (first bin only).
    pub sao_type_idx: Ctx,
    pub split_cu_flag: [Ctx; 3],
    pub cu_transquant_bypass_flag: Ctx,
    /// part_mode bin 0 (the only bin of an intra CU).
    pub part_mode: Ctx,
    pub prev_intra_luma_pred_flag: Ctx,
    pub intra_chroma_pred_mode: Ctx,
    pub split_transform_flag: [Ctx; 3],
    pub cbf_luma: [Ctx; 2],
    /// cbf_cb and cbf_cr, ctxInc = trafoDepth (0..=3); index 4 is the RExt fifth context
    /// (ctxIdx 12), unused.
    pub cbf_cb_cr: [Ctx; 5],
    pub cu_qp_delta_abs: [Ctx; 2],
    /// transform_skip_flag: [0] luma, [1] chroma.
    pub transform_skip_flag: [Ctx; 2],
    pub last_x_prefix: [Ctx; 18],
    pub last_y_prefix: [Ctx; 18],
    pub coded_sub_block_flag: [Ctx; 4],
    /// sig_coeff_flag ctxInc 0..=41; 42 and 43 are the transform_skip_context contexts
    /// (ctxIdx 126, 127; RExt), unused.
    pub sig_coeff_flag: [Ctx; 44],
    pub greater1: [Ctx; 24],
    pub greater2: [Ctx; 6],
}

fn ctxs<const N: usize>(values: [u8; N], qp: i32) -> [Ctx; N] {
    values.map(|v| Ctx::init(v, qp))
}

impl Contexts {
    /// 9.3.2.2 for every context, with `slice_qp_y` = SliceQpY (clipped to 0..=51 inside).
    pub(crate) fn init(slice_qp_y: i32) -> Self {
        let q = slice_qp_y;
        Self {
            sao_merge: Ctx::init(153, q),
            sao_type_idx: Ctx::init(200, q),
            split_cu_flag: ctxs([139, 141, 157], q),
            cu_transquant_bypass_flag: Ctx::init(154, q),
            part_mode: Ctx::init(184, q),
            prev_intra_luma_pred_flag: Ctx::init(184, q),
            intra_chroma_pred_mode: Ctx::init(63, q),
            split_transform_flag: ctxs([153, 138, 138], q),
            cbf_luma: ctxs([111, 141], q),
            cbf_cb_cr: ctxs([94, 138, 182, 154, 154], q),
            cu_qp_delta_abs: ctxs([154, 154], q),
            transform_skip_flag: ctxs([139, 139], q),
            last_x_prefix: ctxs(LAST_PREFIX, q),
            last_y_prefix: ctxs(LAST_PREFIX, q),
            coded_sub_block_flag: ctxs([91, 171, 134, 141], q),
            sig_coeff_flag: ctxs(SIG_COEFF, q),
            greater1: ctxs(GREATER1, q),
            greater2: ctxs([138, 153, 136, 167, 152, 152], q),
        }
    }
}

/// Tables 9-26 and 9-27 (identical), ctxIdx 0..=17.
const LAST_PREFIX: [u8; 18] = [
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
];

/// Table 9-29, ctxIdx 0..=41 then 126..=127.
const SIG_COEFF: [u8; 44] = [
    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, //
    125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, //
    136, 152, 136, 153, 136, 139, 111, 136, 139, 111, //
    141, 111,
];

/// Table 9-30, ctxIdx 0..=23.
const GREATER1: [u8; 24] = [
    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, //
    140, 179, 166, 182, 140, 227, 122, 197,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_of(state: u8, mps: u8) -> Ctx {
        Ctx {
            s: (state << 1) | mps,
        }
    }

    /// The one-bit-at-a-time engine of the spec model (9.3.4.3): the reference for [`Cabac`].
    struct RefCabac<'a> {
        r: BitReader<'a>,
        range: u32,
        offset: u32,
    }

    impl<'a> RefCabac<'a> {
        fn new(r: BitReader<'a>) -> Result<Self> {
            let mut c = Self {
                r,
                range: 510,
                offset: 0,
            };
            c.reinit()?;
            Ok(c)
        }

        fn reinit(&mut self) -> Result<()> {
            self.range = 510;
            self.offset = self.r.u(9)?;
            if self.offset >= 510 {
                return Err(Error::InvalidData("ivlOffset 510 or 511"));
            }
            Ok(())
        }

        fn renorm(&mut self) -> Result<()> {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.r.u(1)?;
            }
            Ok(())
        }

        fn decode(&mut self, ctx: &mut Ctx) -> Result<bool> {
            let lps = ctx.lps_range((self.range >> 6) & 3);
            self.range = self.range.saturating_sub(lps);
            let bin = if self.offset >= self.range {
                self.offset -= self.range;
                self.range = lps;
                !ctx.mps()
            } else {
                ctx.mps()
            };
            ctx.update(bin);
            self.renorm()?;
            Ok(bin)
        }

        fn bypass(&mut self) -> Result<bool> {
            self.offset = (self.offset << 1) | self.r.u(1)?;
            if self.offset >= self.range {
                self.offset -= self.range;
                Ok(true)
            } else {
                Ok(false)
            }
        }

        fn wide(&mut self, n: u32) -> Result<u64> {
            let mut v = 0u64;
            for _ in 0..n {
                v = (v << 1) | u64::from(self.bypass()?);
            }
            Ok(v)
        }

        /// The unary prefix loop of coeff_abs_level_remaining as it was written with `bypass`.
        fn ones(&mut self, max: u32) -> Result<Option<u32>> {
            let mut count = 0u32;
            while self.bypass()? {
                count += 1;
                if count > max {
                    return Ok(None);
                }
            }
            Ok(Some(count))
        }

        fn terminate(&mut self) -> Result<bool> {
            self.range = self.range.saturating_sub(2);
            if self.offset >= self.range {
                return Ok(true);
            }
            self.renorm()?;
            Ok(false)
        }
    }

    /// One random operation on both engines (and their contexts); the two results.
    fn step(
        r: u32,
        rf: &mut RefCabac,
        nw: &mut Cabac,
        a: &mut [Ctx],
        b: &mut [Ctx],
    ) -> (Result<u64>, Result<u64>) {
        let i = (r >> 8) as usize % a.len();
        let n = (r >> 12) % 41;
        match r % 8 {
            0..=2 => (
                rf.decode(&mut a[i]).map(u64::from),
                nw.decode(&mut b[i]).map(u64::from),
            ),
            3 => (rf.bypass().map(u64::from), nw.bypass().map(u64::from)),
            4 => (rf.wide(n.min(32)), nw.bypass_bits(n.min(32)).map(u64::from)),
            5 => (rf.wide(n), nw.bypass_wide(n)),
            6 => (
                rf.ones(n.min(33)).map(|o| o.map_or(u64::MAX, u64::from)),
                nw.bypass_ones(n.min(33))
                    .map(|o| o.map_or(u64::MAX, u64::from)),
            ),
            _ => (rf.terminate().map(u64::from), nw.terminate().map(u64::from)),
        }
    }

    #[test]
    fn matches_reference_engine_on_random_streams() {
        for seed in 1..400u64 {
            let mut s = seed;
            let len = (lcg(&mut s) % 48) as usize + if seed.is_multiple_of(5) { 600 } else { 0 };
            let mut data: Vec<u8> = (0..len).map(|_| (lcg(&mut s) >> 5) as u8).collect();
            // Some seeds start with 0xFF: ivlOffset 510/511 is rejected (9.3.2.5) by both engines.
            if let Some(b) = data.first_mut().filter(|_| seed.is_multiple_of(16)) {
                *b = 0xFF;
            }
            let a0 = RefCabac::new(BitReader::new(&data));
            let b0 = Cabac::new(BitReader::new(&data));
            assert_eq!(a0.is_err(), b0.is_err(), "seed {seed}");
            let (Ok(mut rf), Ok(mut nw)) = (a0, b0) else {
                continue;
            };
            let qp = (lcg(&mut s) % 52) as i32;
            let mut a: Vec<Ctx> = [153u8, 200, 139, 63, 79, 227, 91, 154]
                .iter()
                .map(|&v| Ctx::init(v, qp))
                .collect();
            let mut b = a.clone();
            for st in 0..800 {
                let r = lcg(&mut s);
                if r.is_multiple_of(61) {
                    // PCM style: reader at the spec position, then re-initialisation.
                    let n = (r >> 8) % 20;
                    assert_eq!(
                        rf.r.u(n).ok(),
                        nw.reader().u(n).ok(),
                        "seed {seed} step {st}"
                    );
                    assert_eq!(
                        rf.r.bit_pos(),
                        nw.reader().bit_pos(),
                        "seed {seed} step {st}"
                    );
                    let (x, y) = (rf.reinit(), nw.reinit());
                    assert_eq!(x, y, "seed {seed} step {st}");
                    if x.is_err() || rf.offset >= 510 {
                        break;
                    }
                    continue;
                }
                let (x, y) = step(r, &mut rf, &mut nw, &mut a, &mut b);
                assert_eq!(x, y, "seed {seed} step {st}");
                if x.is_err() {
                    break;
                }
                assert_eq!(a, b, "seed {seed} step {st}");
                assert_eq!(rf.r.bit_pos(), nw.spec_pos(), "seed {seed} step {st}");
                if r % 8 == 7 && x == Ok(1) {
                    break;
                }
            }
        }
    }

    #[test]
    fn bypass_ones_long_runs() {
        for run in [0u32, 1, 15, 16, 17, 31, 32, 33, 35] {
            let mut e = Enc::new();
            for _ in 0..run {
                e.bypass(true);
            }
            e.bypass(false);
            e.bypass(true);
            e.terminate(true);
            let data = e.bytes();
            for max in [0u32, 5, 16, 32, 40] {
                let mut rf = RefCabac::new(BitReader::new(&data)).unwrap();
                let mut nw = Cabac::new(BitReader::new(&data)).unwrap();
                assert_eq!(rf.ones(max), nw.bypass_ones(max), "run {run} max {max}");
                assert_eq!(rf.r.bit_pos(), nw.spec_pos(), "run {run} max {max}");
                assert_eq!(rf.bypass(), nw.bypass(), "run {run} max {max}");
            }
        }
    }

    /// The arithmetic encoder of 9.3.5 (informative), writing bits into a `Vec<bool>`.
    struct Enc {
        bits: Vec<bool>,
        low: u32,
        range: u32,
        first: bool,
        outstanding: u32,
    }

    impl Enc {
        fn new() -> Self {
            Self {
                bits: Vec::new(),
                low: 0,
                range: 510,
                first: true,
                outstanding: 0,
            }
        }

        /// PutBit (Figure 9-12).
        fn put_bit(&mut self, b: bool) {
            if self.first {
                self.first = false;
            } else {
                self.bits.push(b);
            }
            while self.outstanding > 0 {
                self.bits.push(!b);
                self.outstanding -= 1;
            }
        }

        /// RenormE (Figure 9-11).
        fn renorm(&mut self) {
            while self.range < 256 {
                if self.low < 256 {
                    self.put_bit(false);
                } else if self.low >= 512 {
                    self.low -= 512;
                    self.put_bit(true);
                } else {
                    self.low -= 256;
                    self.outstanding += 1;
                }
                self.range <<= 1;
                self.low <<= 1;
            }
        }

        /// EncodeDecision (Figure 9-10).
        fn decision(&mut self, ctx: &mut Ctx, bin: bool) {
            let lps = ctx.lps_range((self.range >> 6) & 3);
            self.range -= lps;
            if bin != ctx.mps() {
                self.low += self.range;
                self.range = lps;
            }
            ctx.update(bin);
            self.renorm();
        }

        /// EncodeBypass (Figure 9-13).
        fn bypass(&mut self, bin: bool) {
            self.low <<= 1;
            if bin {
                self.low += self.range;
            }
            if self.low >= 1024 {
                self.put_bit(true);
                self.low -= 1024;
            } else if self.low < 512 {
                self.put_bit(false);
            } else {
                self.low -= 512;
                self.outstanding += 1;
            }
        }

        /// EncodeTerminate (Figure 9-14) with EncodeFlush (Figure 9-15) for bin 1.
        fn terminate(&mut self, bin: bool) {
            self.range -= 2;
            if bin {
                self.low += self.range;
                self.range = 2;
                self.renorm();
                self.put_bit((self.low >> 9) & 1 == 1);
                let v = ((self.low >> 7) & 3) | 1;
                self.bits.push(v & 2 != 0);
                self.bits.push(v & 1 != 0);
            } else {
                self.renorm();
            }
        }

        /// The written bits packed MSB first, zero padded to whole bytes.
        fn bytes(&self) -> Vec<u8> {
            self.bits
                .chunks(8)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .fold(0u8, |b, (i, &x)| if x { b | (0x80 >> i) } else { b })
                })
                .collect()
        }
    }

    enum Op {
        Ctx(usize, bool),
        Bypass(bool),
        Terminate0,
    }

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*s >> 33) as u32
    }

    #[test]
    fn round_trip_random_bins_and_terminate_position() {
        for seed in 1..40u64 {
            let mut s = seed;
            let qp = (lcg(&mut s) % 52) as i32;
            let init = [153u8, 200, 139, 63, 79, 227, 91, 154];
            let mut ectx: Vec<Ctx> = init.iter().map(|&v| Ctx::init(v, qp)).collect();
            let mut dctx = ectx.clone();
            let n = 1 + lcg(&mut s) % 400;
            let mut ops = Vec::new();
            for _ in 0..n {
                let r = lcg(&mut s);
                // Skewed bins so that the contexts adapt and states move.
                let bin = (r >> 8).is_multiple_of(7);
                ops.push(match r % 8 {
                    0..=4 => Op::Ctx((r >> 12) as usize % 8, bin),
                    5..=6 => Op::Bypass((r >> 8) & 1 == 1),
                    _ => Op::Terminate0,
                });
            }
            let mut e = Enc::new();
            for op in &ops {
                match *op {
                    Op::Ctx(i, b) => e.decision(&mut ectx[i], b),
                    Op::Bypass(b) => e.bypass(b),
                    Op::Terminate0 => e.terminate(false),
                }
            }
            e.terminate(true);
            let data = e.bytes();
            let mut d = Cabac::new(BitReader::new(&data)).unwrap();
            for op in &ops {
                match *op {
                    Op::Ctx(i, b) => assert_eq!(d.decode(&mut dctx[i]).unwrap(), b, "seed {seed}"),
                    Op::Bypass(b) => assert_eq!(d.bypass().unwrap(), b, "seed {seed}"),
                    Op::Terminate0 => assert!(!d.terminate().unwrap(), "seed {seed}"),
                }
            }
            assert_eq!(dctx, ectx);
            assert!(d.terminate().unwrap(), "seed {seed}");
            // The decoder has read exactly the bits the encoder wrote, the last one being 1.
            assert_eq!(d.reader().bit_pos(), e.bits.len(), "seed {seed}");
            assert_eq!(e.bits.last(), Some(&true));
        }
    }

    #[test]
    fn bypass_bits_is_msb_first() {
        let mut e = Enc::new();
        for b in [true, false, true, true, false] {
            e.bypass(b);
        }
        e.terminate(true);
        let data = e.bytes();
        let mut d = Cabac::new(BitReader::new(&data)).unwrap();
        assert_eq!(d.bypass_bits(5).unwrap(), 0b10110);
        assert!(d.bypass_bits(33).is_err());
    }

    #[test]
    fn reinit_after_terminate_and_alignment() {
        // Two substreams, each flushed with terminate(1), the second starting at a byte boundary.
        let mut ctx_a = Ctx::init(154, 30);
        let mut e = Enc::new();
        e.decision(&mut ctx_a, true);
        e.terminate(true);
        let mut data = e.bytes();
        let first_bits = e.bits.len();
        let mut e2 = Enc::new();
        let mut ctx_b = Ctx::init(154, 30);
        e2.decision(&mut ctx_b, false);
        e2.terminate(true);
        data.extend(e2.bytes());
        let mut d = Cabac::new(BitReader::new(&data)).unwrap();
        let mut c = Ctx::init(154, 30);
        assert!(d.decode(&mut c).unwrap());
        assert!(d.terminate().unwrap());
        assert_eq!(d.reader().bit_pos(), first_bits);
        d.reader().align();
        d.reinit().unwrap();
        let mut c = Ctx::init(154, 30);
        assert!(!d.decode(&mut c).unwrap());
        assert!(d.terminate().unwrap());
    }

    #[test]
    fn invalid_initial_offset_is_an_error() {
        assert!(Cabac::new(BitReader::new(&[0xFF, 0x80, 0, 0])).is_err());
        assert!(RefCabac::new(BitReader::new(&[0xFF, 0x80, 0, 0])).is_err());
    }

    #[test]
    fn bins_after_reader_without_reinit_do_not_panic() {
        let data = [0u8; 16];
        let mut d = Cabac::new(BitReader::new(&data)).unwrap();
        let _ = d.reader().u(3);
        let mut ctx = Ctx::init(154, 30);
        assert!(d.decode(&mut ctx).is_err());
        let mut d = Cabac::new(BitReader::new(&data)).unwrap();
        d.reader();
        assert!(d.bypass().is_err());
        let mut d = Cabac::new(BitReader::new(&data)).unwrap();
        d.reader();
        let _ = d.terminate();
    }

    #[test]
    fn truncated_data_is_an_error() {
        assert!(Cabac::new(BitReader::new(&[0xFF])).is_err());
        // 16 bits: 9 at init, then 7 bypass bins, the 8th runs out of data.
        let data = [0u8, 0];
        let mut d = Cabac::new(BitReader::new(&data)).unwrap();
        for _ in 0..7 {
            assert!(!d.bypass().unwrap());
        }
        assert!(d.bypass().is_err());
    }

    #[test]
    fn init_values_by_hand() {
        // sao_type_idx, initValue 200: slopeIdx 12, offsetIdx 8, m = 15, n = 48.
        // QP 26: (15 * 26) >> 4 = 24, + 48 = 72 -> valMps 1, pStateIdx 8.
        assert_eq!(Ctx::init(200, 26), ctx_of(8, 1));
        // QP 51: (15 * 51) >> 4 = 47, + 48 = 95 -> valMps 1, pStateIdx 31.
        assert_eq!(Ctx::init(200, 51), ctx_of(31, 1));
        // split_cu_flag ctx 0, initValue 139: slopeIdx 8, offsetIdx 11, m = -5, n = 72.
        // QP 26: (-130) >> 4 = -9, + 72 = 63 -> valMps 0, pStateIdx 0.
        assert_eq!(Ctx::init(139, 26), ctx_of(0, 0));
        // QP 51: (-255) >> 4 = -16, + 72 = 56 -> valMps 0, pStateIdx 7.
        assert_eq!(Ctx::init(139, 51), ctx_of(7, 0));
        // SliceQpY is clipped to 0..=51. QP -7 acts as 0: (-5 * 0) >> 4 = 0, + 72 = 72 ->
        // valMps 1, pStateIdx 8. QP 60 acts as 51.
        assert_eq!(Ctx::init(139, -7), ctx_of(8, 1));
        assert_eq!(Ctx::init(139, 60), Ctx::init(139, 51));
        // initValue 154 (m = 0, n = 64): valMps 1, pStateIdx 0 for every QP.
        assert_eq!(Ctx::init(154, 33), ctx_of(0, 1));
        // The struct init uses the same tables.
        let c = Contexts::init(26);
        assert_eq!(c.sao_type_idx, Ctx::init(200, 26));
        assert_eq!(c.split_cu_flag[0], ctx_of(0, 0));
        assert_eq!(c.sig_coeff_flag.len(), 44);
    }

    #[test]
    fn tables_spot_checks() {
        assert_eq!(RANGE_TAB_LPS[0], [128, 176, 208, 240]);
        assert_eq!(RANGE_TAB_LPS[31], [29, 35, 41, 48]);
        assert_eq!(RANGE_TAB_LPS[32], [27, 33, 39, 45]);
        assert_eq!(RANGE_TAB_LPS[62], [6, 7, 8, 9]);
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
        for (s, w) in RANGE_TAB_LPS.windows(2).enumerate() {
            assert!(TRANS_IDX_LPS[s + 1] as usize <= s + 1);
            for (a, b) in w[0].iter().zip(&w[1]) {
                assert!(b <= a);
            }
        }
        assert_eq!(TRANS_IDX_LPS[62], 38);
        assert_eq!(TRANS_IDX_LPS[63], 63);
    }
}
