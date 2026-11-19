//! SEI messages: the decoded picture hash (payloadType 132, D.2.20 / D.3.19) and its computation.

/// Hash algorithm of a decoded picture hash SEI (hash_type 0, 1, 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HashKind {
    Md5,
    Crc,
    Checksum,
}

/// The hash values of one picture. Component `c` uses `values[c]`: MD5 fills all 16 bytes,
/// CRC the first 2 (big endian), checksum the first 4 (big endian); the rest is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PictureHash {
    pub kind: HashKind,
    /// 1 (monochrome) or 3 components.
    pub components: usize,
    pub values: [[u8; 16]; 3],
}

/// Walks the sei_message()s of one SEI RBSP and returns the decoded picture hash, if there is a
/// well formed one. Never fails: anything malformed gives `None` (or is skipped).
pub(crate) fn parse(rbsp: &[u8]) -> Option<PictureHash> {
    let mut pos = 0usize;
    // Stops at the rbsp trailing bits byte (0x80) or the end.
    while rbsp.get(pos).is_some_and(|&b| b != 0x80) {
        let mut ty = 0usize;
        while let Some(&b) = rbsp.get(pos) {
            pos += 1;
            ty = ty.checked_add(usize::from(b))?;
            if b != 0xFF {
                break;
            }
        }
        let mut size = 0usize;
        while let Some(&b) = rbsp.get(pos) {
            pos += 1;
            size = size.checked_add(usize::from(b))?;
            if b != 0xFF {
                break;
            }
        }
        let end = pos.checked_add(size)?;
        let payload = rbsp.get(pos..end)?;
        if ty == 132
            && let Some(h) = parse_hash(payload)
        {
            return Some(h);
        }
        pos = end;
    }
    None
}

fn parse_hash(p: &[u8]) -> Option<PictureHash> {
    let (&hash_type, rest) = p.split_first()?;
    let (kind, len) = match hash_type {
        0 => (HashKind::Md5, 16),
        1 => (HashKind::Crc, 2),
        2 => (HashKind::Checksum, 4),
        _ => return None,
    };
    let components = rest.len() / len;
    if rest.len() != components * len || (components != 1 && components != 3) {
        return None;
    }
    let mut values = [[0u8; 16]; 3];
    for (dst, src) in values.iter_mut().zip(rest.chunks_exact(len)) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s;
        }
    }
    Some(PictureHash {
        kind,
        components,
        values,
    })
}

/// Hash of the decoded (uncropped) picture per D.3.19. `planes[c]` is plane `c` with
/// `widths[c]` x `heights[c]` samples, tightly packed. Components beyond the shortest input
/// slice are left zero.
pub(crate) fn compute(
    kind: HashKind,
    planes: &[&[u16]],
    widths: &[usize],
    heights: &[usize],
    bit_depths: &[u8],
) -> [[u8; 16]; 3] {
    let mut out = [[0u8; 16]; 3];
    let comps = planes
        .iter()
        .zip(widths)
        .zip(heights)
        .zip(bit_depths)
        .zip(out.iter_mut());
    for ((((plane, &w), &h), &depth), dst) in comps {
        match kind {
            HashKind::Md5 => {
                let mut md5 = Md5::new();
                for_each_sample(plane, w, h, |_, _, s| {
                    md5.update(s as u8);
                    if depth > 8 {
                        md5.update((s >> 8) as u8);
                    }
                });
                *dst = md5.finish();
            }
            HashKind::Crc => {
                let mut crc = 0xFFFFu32;
                for_each_sample(plane, w, h, |_, _, s| {
                    crc = crc_byte(crc, s as u8);
                    if depth > 8 {
                        crc = crc_byte(crc, (s >> 8) as u8);
                    }
                });
                // The spec appends two zero bytes.
                crc = crc_byte(crc_byte(crc, 0), 0);
                for (d, b) in dst.iter_mut().zip((crc as u16).to_be_bytes()) {
                    *d = b;
                }
            }
            HashKind::Checksum => {
                let mut sum = 0u32;
                for_each_sample(plane, w, h, |x, y, s| {
                    let mask = ((x & 0xff) ^ (y & 0xff) ^ (x >> 8) ^ (y >> 8)) as u32;
                    sum = sum.wrapping_add(u32::from(s & 0xff) ^ mask);
                    if depth > 8 {
                        sum = sum.wrapping_add(u32::from(s >> 8) ^ mask);
                    }
                });
                for (d, b) in dst.iter_mut().zip(sum.to_be_bytes()) {
                    *d = b;
                }
            }
        }
    }
    out
}

/// Calls `f(x, y, sample)` for the samples of a w x h plane in raster order.
fn for_each_sample(plane: &[u16], w: usize, h: usize, mut f: impl FnMut(usize, usize, u16)) {
    if w == 0 {
        return;
    }
    for (y, row) in plane.chunks_exact(w).take(h).enumerate() {
        for (x, &s) in row.iter().enumerate() {
            f(x, y, s);
        }
    }
}

/// One byte of the D.3.19 CRC: polynomial 0x1021, MSB first, 16-bit register.
fn crc_byte(mut crc: u32, byte: u8) -> u32 {
    for bit in (0..8).rev() {
        let msb = (crc >> 15) & 1;
        let b = u32::from(byte >> bit) & 1;
        crc = (((crc << 1) + b) & 0xFFFF) ^ (msb * 0x1021);
    }
    crc
}

/// MD5 (RFC 1321), byte at a time: the pictures are hashed once, speed is not the point.
struct Md5 {
    state: [u32; 4],
    buf: [u8; 64],
    fill: usize,
    total: u64,
}

impl Md5 {
    fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buf: [0; 64],
            fill: 0,
            total: 0,
        }
    }

    fn update(&mut self, byte: u8) {
        if let Some(slot) = self.buf.get_mut(self.fill) {
            *slot = byte;
        }
        self.fill += 1;
        self.total = self.total.wrapping_add(1);
        if self.fill == 64 {
            self.block();
            self.fill = 0;
        }
    }

    fn finish(mut self) -> [u8; 16] {
        let bits = self.total.wrapping_mul(8);
        self.update(0x80);
        while self.fill != 56 {
            self.update(0);
        }
        for b in bits.to_le_bytes() {
            self.update(b);
        }
        let mut out = [0u8; 16];
        for (dst, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            for (d, b) in dst.iter_mut().zip(w.to_le_bytes()) {
                *d = b;
            }
        }
        out
    }

    fn block(&mut self) {
        let mut m = [0u32; 16];
        for (w, c) in m.iter_mut().zip(self.buf.as_chunks::<4>().0) {
            *w = c.iter().rev().fold(0, |a, &b| (a << 8) | u32::from(b));
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64usize {
            let (f, g) = match i >> 4 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) & 15),
                2 => (b ^ c ^ d, (3 * i + 5) & 15),
                _ => (c ^ (b | !d), (7 * i) & 15),
            };
            let k = MD5_K.get(i).copied().unwrap_or(0);
            let s = match i >> 4 {
                0 => [7, 12, 17, 22],
                1 => [5, 9, 14, 20],
                2 => [4, 11, 16, 23],
                _ => [6, 10, 15, 21],
            }
            .get(i & 3)
            .copied()
            .unwrap_or(0);
            let word = m.get(g).copied().unwrap_or(0);
            let t = f.wrapping_add(a).wrapping_add(k).wrapping_add(word);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(t.rotate_left(s));
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// MD5 sine table, K[i] = floor(2^32 * abs(sin(i + 1))), RFC 1321 section 3.4.
const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn md5_of(data: &[u8]) -> [u8; 16] {
        let mut m = Md5::new();
        for &b in data {
            m.update(b);
        }
        m.finish()
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn md5_vectors() {
        assert_eq!(hex(&md5_of(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(&md5_of(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        // 80 bytes: crosses a block boundary.
        let long =
            b"12345678901234567890123456789012345678901234567890123456789012345678901234567890";
        assert_eq!(hex(&md5_of(long)), "57edf4a22be3c955ac49da2e2107b67a");
    }

    #[test]
    fn md5_picture_bytes() {
        // 8-bit: one byte per sample; 10-bit: two little-endian bytes.
        let p8 = compute(HashKind::Md5, &[&[0x61, 0x62, 0x63]], &[3], &[1], &[8]);
        assert_eq!(p8[0], md5_of(b"abc"));
        let p10 = compute(HashKind::Md5, &[&[0x0261]], &[1], &[1], &[10]);
        assert_eq!(p10[0], md5_of(&[0x61, 0x02]));
    }

    #[test]
    fn checksum_tiny_plane() {
        // x = 0: mask 0 -> 1; x = 1: mask 1 -> 2 ^ 1 = 3.
        let h = compute(HashKind::Checksum, &[&[1, 2]], &[2], &[1], &[8]);
        assert_eq!(h[0][..4], 4u32.to_be_bytes());
        // 10-bit sample 0x0305 at (0, 0): 0x05 + 0x03.
        let h = compute(HashKind::Checksum, &[&[0x0305]], &[1], &[1], &[10]);
        assert_eq!(h[0][..4], 8u32.to_be_bytes());
    }

    #[test]
    fn crc_tiny_plane() {
        // Empty plane: 16 zero bits from 0xFFFF give 0x1D0F (worked by hand).
        let h = compute(HashKind::Crc, &[&[]], &[1], &[0], &[8]);
        assert_eq!(h[0][..2], [0x1D, 0x0F]);
        // "123456789": CRC-16/AUG-CCITT check value.
        let s: Vec<u16> = b"123456789".iter().map(|&b| u16::from(b)).collect();
        let h = compute(HashKind::Crc, &[&s], &[9], &[1], &[8]);
        assert_eq!(h[0][..2], [0xE5, 0xCC]);
    }

    #[test]
    fn sei_walk_extended_type_and_size() {
        // Unknown message type 255 + 1 = 256 with size 255 + 3 = 258 (0xFF extension),
        // then a CRC hash for one component, then trailing bits.
        let mut v = vec![0xFF, 0x01, 0xFF, 0x03];
        v.extend(std::iter::repeat_n(0xAA, 258));
        v.extend([132, 3, 1, 0xBE, 0xEF]);
        v.push(0x80);
        let h = parse(&v).unwrap();
        assert_eq!(h.kind, HashKind::Crc);
        assert_eq!(h.components, 1);
        assert_eq!(h.values[0][..2], [0xBE, 0xEF]);
    }

    #[test]
    fn sei_md5_three_components() {
        let mut v = vec![132, 49, 0];
        v.extend((0..48).map(|i| i as u8));
        v.push(0x80);
        let h = parse(&v).unwrap();
        assert_eq!(h.kind, HashKind::Md5);
        assert_eq!(h.components, 3);
        assert_eq!(h.values[2][15], 47);
    }

    #[test]
    fn sei_truncated_is_none() {
        assert!(parse(&[132, 4, 1, 0xBE]).is_none());
        assert!(parse(&[132]).is_none());
        assert!(parse(&[0xFF]).is_none());
        assert!(parse(&[]).is_none());
        // Wrong payload length for the hash type.
        assert!(parse(&[132, 2, 1, 0xBE, 0x80]).is_none());
    }
}
