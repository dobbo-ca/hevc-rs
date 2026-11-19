//! Bit reader over an RBSP (H.265 7.2): most significant bit first.

use crate::error::{Error, Result};

const TRUNCATED: Error = Error::InvalidData("truncated");

/// Reads bits from a byte slice. `pos` is the bit position from the start of `data`.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// One bit, no end check beyond the slice itself.
    fn bit(&mut self) -> Result<u32> {
        let b = self.data.get(self.pos >> 3).ok_or(TRUNCATED)?;
        let v = (b >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(u32::from(v))
    }

    /// u(n): `n` bits as an unsigned number, `n <= 32`.
    pub fn u(&mut self, n: u32) -> Result<u32> {
        if n > 32 || (n as usize) > self.bits_left() {
            return Err(TRUNCATED);
        }
        let mut v: u64 = 0;
        for _ in 0..n {
            v = (v << 1) | u64::from(self.bit()?);
        }
        Ok(v as u32)
    }

    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// ue(v), 9.2: more than 31 leading zero bits is an error.
    pub fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(Error::InvalidData("ue(v) too long"));
            }
        }
        let rest = self.u(zeros)?;
        // zeros <= 31, so the sum is at most u32::MAX - 1.
        Ok(((1u32 << zeros) - 1) + rest)
    }

    /// se(v), 9.2.2: k maps to (-1)^(k+1) * Ceil(k / 2).
    pub fn se(&mut self) -> Result<i32> {
        let k = i64::from(self.ue()?);
        let v = if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) };
        i32::try_from(v).map_err(|_| Error::InvalidData("se(v) out of range"))
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        if n > self.bits_left() {
            return Err(TRUNCATED);
        }
        self.pos += n;
        Ok(())
    }

    pub fn byte_aligned(&self) -> bool {
        self.pos & 7 == 0
    }

    /// Moves to the next byte boundary (no-op when aligned).
    pub fn align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    pub fn bits_left(&self) -> usize {
        self.data.len().saturating_mul(8).saturating_sub(self.pos)
    }

    /// more_rbsp_data() of 7.2: true when a bit before the rbsp_stop_one_bit (the last 1 bit of
    /// the RBSP) is still to be read.
    pub fn more_rbsp_data(&self) -> bool {
        let Some(i) = self.data.iter().rposition(|&b| b != 0) else {
            return false;
        };
        let Some(&b) = self.data.get(i) else {
            return false;
        };
        let stop = i * 8 + 7 - b.trailing_zeros() as usize;
        self.pos < stop
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_bits_and_flags() {
        let mut r = BitReader::new(&[0b1010_1100, 0xFF, 0x00, 0x00, 0x00]);
        assert!(r.flag().unwrap());
        assert!(!r.flag().unwrap());
        assert_eq!(r.u(3).unwrap(), 0b101);
        assert_eq!(r.bit_pos(), 5);
        assert!(!r.byte_aligned());
        r.align();
        assert_eq!(r.bit_pos(), 8);
        assert_eq!(r.u(32).unwrap(), 0xFF00_0000);
        assert_eq!(r.u(1).unwrap_err(), Error::InvalidData("truncated"));
        assert_eq!(r.u(33).unwrap_err(), Error::InvalidData("truncated"));
    }

    #[test]
    fn ue_values() {
        // 1 | 010 | 011 | 00100 | 00101 | 0001000  -> 0 1 2 3 4 7
        let bits = 0b1010_0110_0100_0010_1000_1000u32;
        let bytes = bits.to_be_bytes();
        let mut r = BitReader::new(bytes.get(1..).unwrap());
        for want in [0, 1, 2, 3, 4, 7] {
            assert_eq!(r.ue().unwrap(), want);
        }
    }

    #[test]
    fn se_values() {
        // ue 1,2,3,4 -> se +1,-1,+2,-2
        // 010 011 00100 00101 -> 0100 1100 1000 0101
        let mut r = BitReader::new(&[0b0100_1100, 0b1000_0101]);
        for want in [1, -1, 2, -2] {
            assert_eq!(r.se().unwrap(), want);
        }
    }

    #[test]
    fn ue_limits() {
        // 31 zeros, a 1, 31 zero bits: value 2^31 - 1.
        let mut r = BitReader::new(&[0, 0, 0, 1, 0, 0, 0, 0]);
        assert_eq!(r.ue().unwrap(), (1 << 31) - 1);
        // 31 zeros, a 1, 31 one bits: u32::MAX - 1.
        let mut r = BitReader::new(&[0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(r.ue().unwrap(), u32::MAX - 1);
        // 32 zeros.
        let mut r = BitReader::new(&[0, 0, 0, 0, 0x80]);
        assert!(r.ue().is_err());
        // Truncated.
        let mut r = BitReader::new(&[0x00, 0x01]); // 15 zeros, a 1, then 15 bits missing
        assert!(r.ue().is_err());
        // se of the largest codes stays in range.
        let mut r = BitReader::new(&[0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(r.se().unwrap(), -i32::MAX);
        let mut r = BitReader::new(&[0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFC]);
        assert_eq!(r.se().unwrap(), i32::MAX);
    }

    #[test]
    fn skip_and_left() {
        let mut r = BitReader::new(&[0xAA, 0xBB]);
        r.skip(12).unwrap();
        assert_eq!(r.bits_left(), 4);
        assert_eq!(r.skip(5).unwrap_err(), Error::InvalidData("truncated"));
        assert_eq!(r.u(4).unwrap(), 0xB);
        assert_eq!(r.data().len(), 2);
    }

    #[test]
    fn more_rbsp_data() {
        // The stop bit is the last 1 bit, at bit position 3.
        let mut r = BitReader::new(&[0b1011_0000]);
        assert!(r.more_rbsp_data());
        r.skip(2).unwrap();
        assert!(r.more_rbsp_data());
        r.skip(1).unwrap();
        assert!(!r.more_rbsp_data());
        // Trailing zero bytes after the stop bit.
        let mut r = BitReader::new(&[0x80, 0x00, 0x00]);
        assert!(!r.more_rbsp_data());
        r.skip(0).unwrap();
        let r = BitReader::new(&[0x01, 0x80]);
        assert!(r.more_rbsp_data());
        // No 1 bit at all.
        assert!(!BitReader::new(&[0, 0]).more_rbsp_data());
        assert!(!BitReader::new(&[]).more_rbsp_data());
    }
}
