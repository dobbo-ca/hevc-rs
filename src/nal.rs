//! NAL units: Annex B and length-prefixed splitting, the NAL header (7.3.1.2), emulation
//! prevention (7.4.2) and the HEVCDecoderConfigurationRecord (ISO/IEC 14496-15, 8.3.3.1).

use crate::error::{Error, Result};

/// One NAL unit. `payload` is what follows the 2-byte header, still escaped.
pub(crate) struct Nal<'a> {
    pub kind: u8,
    pub layer_id: u8,
    pub temporal_id: u8,
    pub payload: &'a [u8],
}

/// Parses nal_unit_header() (7.3.1.2) of `nal`, which holds the whole NAL unit.
pub(crate) fn header(nal: &[u8]) -> Result<Nal<'_>> {
    let [b0, b1, payload @ ..] = nal else {
        return Err(Error::InvalidData("NAL unit shorter than its header"));
    };
    if b0 >> 7 != 0 {
        return Err(Error::InvalidData("forbidden_zero_bit set"));
    }
    let tid_plus1 = b1 & 7;
    if tid_plus1 == 0 {
        return Err(Error::InvalidData("nuh_temporal_id_plus1 is 0"));
    }
    Ok(Nal {
        kind: (b0 >> 1) & 0x3F,
        layer_id: ((b0 & 1) << 5) | (b1 >> 3),
        temporal_id: tid_plus1 - 1,
        payload,
    })
}

/// Splits an Annex B byte stream (B.2) at the 00 00 01 start codes. Bytes before the first start
/// code are ignored; zero bytes before a start code (the fourth byte of 00 00 00 01, trailing
/// zeros) are trimmed; empty NAL units are skipped.
pub(crate) struct AnnexB<'a> {
    data: &'a [u8],
    pos: usize,
    done: bool,
}

/// Index of the next 00 00 01 at or after `from`.
fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    let rest = data.get(from..)?;
    rest.windows(3)
        .position(|w| w == [0, 0, 1])
        .map(|p| p + from)
}

pub(crate) fn annexb(stream: &[u8]) -> AnnexB<'_> {
    match find_start_code(stream, 0) {
        Some(i) => AnnexB {
            data: stream,
            pos: i + 3,
            done: false,
        },
        None => AnnexB {
            data: stream,
            pos: 0,
            done: true,
        },
    }
}

impl<'a> Iterator for AnnexB<'a> {
    type Item = Result<Nal<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done {
                return None;
            }
            let start = self.pos;
            let end = match find_start_code(self.data, start) {
                Some(j) => {
                    self.pos = j + 3;
                    j
                }
                None => {
                    self.done = true;
                    self.data.len()
                }
            };
            let mut nal = self.data.get(start..end).unwrap_or(&[]);
            while let [rest @ .., 0] = nal {
                nal = rest;
            }
            if !nal.is_empty() {
                return Some(header(nal));
            }
        }
    }
}

/// Splits data made of NAL units, each preceded by a big-endian length of `len_size` bytes
/// (1, 2 or 4).
pub(crate) struct LengthPrefixed<'a> {
    data: &'a [u8],
    len_size: usize,
    done: bool,
}

pub(crate) fn length_prefixed(data: &[u8], len_size: usize) -> LengthPrefixed<'_> {
    LengthPrefixed {
        data,
        len_size,
        done: false,
    }
}

impl<'a> Iterator for LengthPrefixed<'a> {
    type Item = Result<Nal<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.data.is_empty() {
            return None;
        }
        // An error ends the iteration: the position after it is unknown.
        self.done = true;
        if !matches!(self.len_size, 1 | 2 | 4) {
            return Some(Err(Error::InvalidData("bad NAL length size")));
        }
        let Some((len_bytes, rest)) = self.data.split_at_checked(self.len_size) else {
            return Some(Err(Error::InvalidData("truncated NAL length")));
        };
        let len = len_bytes
            .iter()
            .fold(0usize, |a, &b| (a << 8) | usize::from(b));
        let Some((nal, rest)) = rest.split_at_checked(len) else {
            return Some(Err(Error::InvalidData("NAL length beyond the data")));
        };
        self.data = rest;
        let item = header(nal);
        self.done = item.is_err();
        Some(item)
    }
}

/// Removes the emulation_prevention_three_byte of every 00 00 03 (7.4.2) from `payload` into
/// `out`, which is cleared first. The caller reserves `out` (and charges it to the ledger).
pub(crate) fn unescape(payload: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let mut zeros = 0u32;
    for &b in payload {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
}

/// Parameter set and SEI NAL units of an hvcC record.
pub(crate) struct HvcC<'a> {
    pub length_size: usize,
    /// Each NAL unit including its 2-byte header.
    pub nals: Vec<&'a [u8]>,
}

/// Most NAL units accepted in one record.
const MAX_HVCC_NALS: usize = 1024;

fn take<'a>(d: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    let (a, b) = d
        .split_at_checked(n)
        .ok_or(Error::InvalidData("truncated hvcC"))?;
    *d = b;
    Ok(a)
}

fn take_u16(d: &mut &[u8]) -> Result<usize> {
    let b = take(d, 2)?;
    Ok(b.iter().fold(0usize, |a, &x| (a << 8) | usize::from(x)))
}

/// Parses an HEVCDecoderConfigurationRecord (ISO/IEC 14496-15, 8.3.3.1): byte 0 is
/// configurationVersion (must be 1), bytes 1..=20 are the profile, level and format fields
/// (skipped), the low 2 bits of byte 21 are lengthSizeMinusOne, byte 22 is numOfArrays.
pub(crate) fn parse_hvcc(config: &[u8]) -> Result<HvcC<'_>> {
    let mut d = config;
    let head = take(&mut d, 22)?;
    let [version, .., len_byte] = head else {
        return Err(Error::InvalidData("truncated hvcC"));
    };
    if *version != 1 {
        return Err(Error::InvalidData("hvcC configurationVersion is not 1"));
    }
    let length_size = usize::from(len_byte & 3) + 1;
    if length_size == 3 {
        return Err(Error::InvalidData("hvcC lengthSizeMinusOne is 2"));
    }
    let arrays = take(&mut d, 1)?.first().copied().unwrap_or(0);
    let mut nals = Vec::new();
    for _ in 0..arrays {
        take(&mut d, 1)?; // array_completeness, reserved, NAL_unit_type
        let num = take_u16(&mut d)?;
        if num > MAX_HVCC_NALS - nals.len() {
            return Err(Error::InvalidData("too many NAL units in hvcC"));
        }
        for _ in 0..num {
            let len = take_u16(&mut d)?;
            nals.push(take(&mut d, len)?);
        }
    }
    Ok(HvcC { length_size, nals })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fields() {
        // SPS (33), layer 0, tid_plus1 1.
        let n = header(&[0x42, 0x01, 0xAA, 0xBB]).unwrap();
        assert_eq!((n.kind, n.layer_id, n.temporal_id), (33, 0, 0));
        assert_eq!(n.payload, [0xAA, 0xBB]);
        // layer 63, tid_plus1 7: b0 = 0b0_100001_1, b1 = 0b11111_111.
        let n = header(&[0x43, 0xFF]).unwrap();
        assert_eq!((n.kind, n.layer_id, n.temporal_id), (33, 63, 6));
        assert!(n.payload.is_empty());
        assert!(header(&[0xC2, 0x01]).is_err()); // forbidden_zero_bit
        assert!(header(&[0x42, 0x00]).is_err()); // tid_plus1 = 0
        assert!(header(&[0x42]).is_err());
        assert!(header(&[]).is_err());
    }

    #[test]
    fn unescape_removes_emulation_bytes() {
        let mut out = vec![9];
        unescape(&[1, 0, 0, 3, 0, 0, 3, 3, 0, 0, 3], &mut out);
        assert_eq!(out, [1, 0, 0, 0, 0, 3, 0, 0]);
        // 00 03 is not an emulation prevention; 00 00 03 00 00 03: both removed.
        unescape(&[0, 3, 0, 0, 3, 1, 0, 0, 0], &mut out);
        assert_eq!(out, [0, 3, 0, 0, 1, 0, 0, 0]);
        unescape(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn annexb_three_and_four_byte_start_codes() {
        let s = [
            7, 7, // junk before the first start code
            0, 0, 1, 0x40, 0x01, 0x11, // 3-byte start code, VPS
            0, 0, 0, 1, 0x42, 0x01, 0x22, 0x33, 0,
            0, // 4-byte start code, SPS, trailing zeros
            0, 0, 1, // empty NAL
            0, 0, 1, 0x44, 0x01, 0x55, // PPS, ends the stream
        ];
        let nals: Vec<_> = annexb(&s).map(|n| n.unwrap()).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].kind, 32);
        assert_eq!(nals[0].payload, [0x11]);
        assert_eq!(nals[1].kind, 33);
        assert_eq!(nals[1].payload, [0x22, 0x33]);
        assert_eq!(nals[2].kind, 34);
        assert_eq!(nals[2].payload, [0x55]);
    }

    #[test]
    fn annexb_edge_cases() {
        assert_eq!(annexb(&[]).count(), 0);
        assert_eq!(annexb(&[1, 2, 3]).count(), 0);
        assert_eq!(annexb(&[0, 0, 1]).count(), 0);
        assert_eq!(annexb(&[0, 0, 0, 1, 0, 0, 0]).count(), 0);
        // One-byte NAL unit is invalid.
        let items: Vec<_> = annexb(&[0, 0, 1, 0x40]).collect();
        assert_eq!(items.len(), 1);
        assert!(items[0].is_err());
        // The error does not stop the following NAL unit.
        let items: Vec<_> = annexb(&[0, 0, 1, 0x40, 0, 0, 1, 0x42, 0x01]).collect();
        assert_eq!(items.len(), 2);
        assert!(items[0].is_err() && items[1].is_ok());
    }

    #[test]
    fn length_prefixed_sizes() {
        let nals = [[0x40u8, 0x01, 0xAA], [0x42, 0x01, 0xBB]];
        for len_size in [1usize, 2, 4] {
            let mut data = Vec::new();
            for n in &nals {
                let be = (n.len() as u32).to_be_bytes();
                data.extend_from_slice(be.get(4 - len_size..).unwrap());
                data.extend_from_slice(n);
            }
            let got: Vec<_> = length_prefixed(&data, len_size)
                .map(|n| n.unwrap())
                .collect();
            assert_eq!(got.len(), 2);
            assert_eq!(got[0].kind, 32);
            assert_eq!(got[1].payload, [0xBB]);
        }
    }

    #[test]
    fn length_prefixed_errors() {
        // Length beyond the data.
        let items: Vec<_> = length_prefixed(&[5, 0x40, 0x01], 1).collect();
        assert_eq!(items.len(), 1);
        assert!(items[0].is_err());
        // Truncated length field.
        let items: Vec<_> = length_prefixed(&[0, 0, 0], 4).collect();
        assert!(items.len() == 1 && items[0].is_err());
        // Bad length size.
        let items: Vec<_> = length_prefixed(&[2, 0x40, 0x01], 3).collect();
        assert!(items.len() == 1 && items[0].is_err());
        // Empty data.
        assert_eq!(length_prefixed(&[], 4).count(), 0);
    }

    /// Hand-built record: version 1, 20 filler bytes, length byte, then the arrays.
    fn record(len_byte: u8, arrays: &[(u8, &[&[u8]])]) -> Vec<u8> {
        let mut v = vec![1];
        v.extend_from_slice(&[0x11; 20]);
        v.push(0xFC | len_byte);
        v.push(arrays.len() as u8);
        for (kind, nals) in arrays {
            v.push(0x80 | kind);
            v.extend_from_slice(&(nals.len() as u16).to_be_bytes());
            for n in *nals {
                v.extend_from_slice(&(n.len() as u16).to_be_bytes());
                v.extend_from_slice(n);
            }
        }
        v
    }

    #[test]
    fn hvcc_round_trip() {
        let vps: &[u8] = &[0x40, 0x01, 0x0C];
        let sps: &[u8] = &[0x42, 0x01, 0x01, 0x02];
        let pps: &[u8] = &[0x44, 0x01, 0xC1];
        let pps2: &[u8] = &[0x44, 0x01, 0xC2];
        let rec = record(
            3,
            &[(32, &[vps][..]), (33, &[sps][..]), (34, &[pps, pps2][..])],
        );
        let h = parse_hvcc(&rec).unwrap();
        assert_eq!(h.length_size, 4);
        assert_eq!(h.nals, [vps, sps, pps, pps2]);
        assert_eq!(parse_hvcc(&record(1, &[])).unwrap().length_size, 2);
        assert_eq!(parse_hvcc(&record(0, &[])).unwrap().length_size, 1);
    }

    #[test]
    fn hvcc_errors() {
        let sps: &[u8] = &[0x42, 0x01, 0x01];
        let good = record(3, &[(33, &[sps][..])]);
        assert!(parse_hvcc(&good).is_ok());
        // Every truncation fails.
        for n in 0..good.len() {
            assert!(parse_hvcc(good.get(..n).unwrap()).is_err(), "len {n}");
        }
        // Bad version, bad length size.
        let mut bad = good.clone();
        *bad.first_mut().unwrap() = 2;
        assert!(parse_hvcc(&bad).is_err());
        assert!(parse_hvcc(&record(2, &[])).is_err());
        // More than 1024 NAL units.
        let many = vec![sps; 1025];
        assert!(parse_hvcc(&record(3, &[(33, many.as_slice())])).is_err());
        let ok = vec![sps; 1024];
        assert_eq!(
            parse_hvcc(&record(3, &[(33, ok.as_slice())]))
                .unwrap()
                .nals
                .len(),
            1024
        );
    }
}
