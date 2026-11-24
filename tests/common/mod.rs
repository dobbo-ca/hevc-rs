//! A small HEVC intra encoder for the integration tests: real, decodable Main profile streams,
//! built in the test, so no fixture file or outside tool is needed. Every picture is 32x32 intra
//! CUs (one per CTB) with pseudo-random luma and chroma modes, DC-only residuals of random level
//! and sign, and random SAO parameters per CTB, so deblocking and SAO change real samples. The
//! arithmetic coder is the informative encoder of H.265 9.3.5 (as in the `cabac` unit tests); the
//! context init values are those of Tables 9-5 to 9-31 (initType 0), and the state tables are
//! Tables 9-52 and 9-53.

// Each test crate uses a part of this module.
#![allow(dead_code)]

/// Which stream [`stream`] builds.
#[derive(Clone, Copy, Default)]
pub struct Spec {
    /// Luma width and height, multiples of 32.
    pub width: u32,
    pub height: u32,
    /// IDR pictures in the stream.
    pub pictures: usize,
    /// The last `hidden` pictures have pic_output_flag 0 (decoded, never output).
    pub hidden: usize,
    /// The slice data of the last picture does not end at its last CTU
    /// (end_of_slice_segment_flag 0), so that picture fails once its CTUs are decoded.
    pub bad_last: bool,
    /// The SPS holds 64 short-term reference picture sets of 16 pictures and 32 long-term
    /// entries (IDR slices do not refer to them), so it is a large set to store.
    pub big_sps: bool,
    /// VPS, SPS and PPS before every picture, not only the first.
    pub repeat: bool,
}

/// SliceQpY of every slice.
const QP: i32 = 32;

/// Bit writer: u(n), ue(v), se(v).
#[derive(Default)]
struct W(Vec<bool>);

impl W {
    fn u(&mut self, n: u32, v: u64) {
        for i in (0..n).rev() {
            self.0.push((v >> i) & 1 == 1);
        }
    }
    fn flag(&mut self, v: bool) {
        self.0.push(v);
    }
    fn ue(&mut self, v: u32) {
        let x = u64::from(v) + 1;
        let len = 64 - x.leading_zeros();
        self.u(len - 1, 0);
        self.u(len, x);
    }
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }
    /// The bits, then a 1 bit (rbsp_stop_one_bit, or the 1 of byte_alignment()), zero padded
    /// to a byte.
    fn rbsp(mut self) -> Vec<u8> {
        self.flag(true);
        pack(&self.0)
    }
}

/// Bits packed MSB first, zero padded to whole bytes.
fn pack(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |b, (i, &x)| if x { b | (0x80 >> i) } else { b })
        })
        .collect()
}

/// xorshift64.
struct Rng(u64);

impl Rng {
    /// A value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) % n
    }
}

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

/// A context variable: pStateIdx and valMps.
#[derive(Clone, Copy)]
struct Ctx {
    state: u8,
    mps: bool,
}

impl Ctx {
    /// 9.3.2.2, (9-4) to (9-6).
    fn new(init_value: u8, qp: i32) -> Self {
        let m = i32::from(init_value >> 4) * 5 - 45;
        let n = (i32::from(init_value & 15) << 3) - 16;
        let pre = (((m * qp.clamp(0, 51)) >> 4) + n).clamp(1, 126);
        let mps = pre > 63;
        Self {
            state: (if mps { pre - 64 } else { 63 - pre }) as u8,
            mps,
        }
    }
}

/// The context variables the encoder uses, at the ctxInc it uses them with. Where luma and
/// chroma differ, index 0 is luma and 1 chroma.
struct Ctxs {
    sao_merge: Ctx,
    sao_type: Ctx,
    part_mode: Ctx,
    prev_intra: Ctx,
    chroma_mode: Ctx,
    /// cbf_luma, ctxInc 1 (trafoDepth 0).
    cbf_luma: Ctx,
    /// cbf_cb and cbf_cr share it, ctxInc 0 (trafoDepth 0).
    cbf_chroma: Ctx,
    /// last_sig_coeff_x_prefix and _y_prefix, bin 0: ctxInc 10 (32x32 luma) and 15 (chroma).
    last_x: [Ctx; 2],
    last_y: [Ctx; 2],
    /// coeff_abs_level_greater1_flag of the first coefficient: ctxInc 1 and 17.
    greater1: [Ctx; 2],
    /// coeff_abs_level_greater2_flag: ctxInc 0 and 4.
    greater2: [Ctx; 2],
}

impl Ctxs {
    fn new(qp: i32) -> Self {
        let c = |v| Ctx::new(v, qp);
        Self {
            sao_merge: c(153),
            sao_type: c(200),
            part_mode: c(184),
            prev_intra: c(184),
            chroma_mode: c(63),
            cbf_luma: c(141),
            cbf_chroma: c(94),
            last_x: [c(111), c(108)],
            last_y: [c(111), c(108)],
            greater1: [c(92), c(179)],
            greater2: [c(138), c(152)],
        }
    }
}

/// The arithmetic encoder of 9.3.5 (informative).
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

    /// EncodeDecision (Figure 9-10) with the state transition of 9.3.4.3.2.2.
    fn decision(&mut self, c: &mut Ctx, bin: bool) {
        let s = usize::from(c.state);
        let lps = u32::from(RANGE_TAB_LPS[s][((self.range >> 6) & 3) as usize]);
        self.range -= lps;
        if bin != c.mps {
            self.low += self.range;
            self.range = lps;
            if c.state == 0 {
                c.mps = !c.mps;
            }
            c.state = TRANS_IDX_LPS[s];
        } else if c.state < 62 {
            c.state += 1;
        }
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

    /// `n` bypass bins of `v`, MSB first.
    fn bypass_bits(&mut self, n: u32, v: u64) {
        for i in (0..n).rev() {
            self.bypass((v >> i) & 1 == 1);
        }
    }

    /// EncodeTerminate (Figure 9-14), with EncodeFlush (Figure 9-15) for bin 1. The last bit
    /// written is the rbsp_stop_one_bit.
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
}

/// sao() of 7.3.8.3 for CTB `rs` (raster address) of a picture `w` CTBs wide, in a slice that
/// starts at CTB 0: random merges, types, offsets, band positions and edge classes.
fn sao(e: &mut Enc, c: &mut Ctxs, rng: &mut Rng, rs: usize, w: usize) {
    let mut merge = false;
    if !rs.is_multiple_of(w) {
        merge = rng.below(4) == 0;
        e.decision(&mut c.sao_merge, merge);
    }
    if rs >= w && !merge {
        merge = rng.below(4) == 0;
        e.decision(&mut c.sao_merge, merge);
    }
    if merge {
        return;
    }
    let mut t = 0;
    for comp in 0..3 {
        // Cr takes the type (and edge class) of Cb.
        if comp < 2 {
            t = rng.below(3);
            e.decision(&mut c.sao_type, t != 0);
            if t != 0 {
                e.bypass(t == 2);
            }
        }
        if t == 0 {
            continue;
        }
        // sao_offset_abs: TR, cMax 7 at 8 bits.
        let abs: [u64; 4] = std::array::from_fn(|_| rng.below(8));
        for a in abs {
            for _ in 0..a {
                e.bypass(true);
            }
            if a < 7 {
                e.bypass(false);
            }
        }
        if t == 1 {
            for a in abs {
                if a != 0 {
                    e.bypass(rng.below(2) == 1);
                }
            }
            e.bypass_bits(5, rng.below(32));
        } else if comp < 2 {
            e.bypass_bits(2, rng.below(4));
        }
    }
}

/// residual_coding() (7.3.8.11) of a block whose one coefficient is the DC one, `level` 1..=6:
/// last position (0, 0), greater1 and greater2 flags, sign, and coeff_abs_level_remaining
/// (Rice parameter 0) for the levels above 3.
fn dc(e: &mut Enc, c: &mut Ctxs, chroma: bool, level: u64, negative: bool) {
    let k = usize::from(chroma);
    e.decision(&mut c.last_x[k], false);
    e.decision(&mut c.last_y[k], false);
    e.decision(&mut c.greater1[k], level > 1);
    if level > 1 {
        e.decision(&mut c.greater2[k], level > 2);
    }
    e.bypass(negative);
    if level > 2 {
        for _ in 3..level {
            e.bypass(true);
        }
        e.bypass(false);
    }
}

/// One 32x32 intra coding unit (7.3.8.5) with PART_2Nx2N and one transform unit.
fn cu(e: &mut Enc, c: &mut Ctxs, rng: &mut Rng) {
    e.decision(&mut c.part_mode, true);
    // prev_intra_luma_pred_flag, then mpm_idx (TR, bypass) or rem_intra_luma_pred_mode.
    let mpm = rng.below(2) == 0;
    e.decision(&mut c.prev_intra, mpm);
    if mpm {
        let idx = rng.below(3);
        e.bypass(idx > 0);
        if idx > 0 {
            e.bypass(idx == 2);
        }
    } else {
        e.bypass_bits(5, rng.below(32));
    }
    // intra_chroma_pred_mode: 0 is mode 4 (the luma mode), 1 then two bypass bins.
    let explicit = rng.below(2) == 0;
    e.decision(&mut c.chroma_mode, explicit);
    if explicit {
        e.bypass_bits(2, rng.below(4));
    }
    let cbf = [rng.below(4) != 0, rng.below(2) == 0, rng.below(2) == 0];
    e.decision(&mut c.cbf_chroma, cbf[1]);
    e.decision(&mut c.cbf_chroma, cbf[2]);
    e.decision(&mut c.cbf_luma, cbf[0]);
    for (comp, coded) in cbf.into_iter().enumerate() {
        if coded {
            let level = 1 + rng.below(6);
            dc(e, c, comp > 0, level, rng.below(2) == 1);
        }
    }
}

/// slice_segment_data() of a picture of `w` x `h` CTBs. With `bad_end` the last CTU is followed
/// by end_of_slice_segment_flag 0.
fn slice_data(w: usize, h: usize, rng: &mut Rng, bad_end: bool) -> Vec<u8> {
    let mut e = Enc::new();
    let mut c = Ctxs::new(QP);
    for rs in 0..w * h {
        sao(&mut e, &mut c, rng, rs, w);
        cu(&mut e, &mut c, rng);
        e.terminate(rs + 1 == w * h && !bad_end);
    }
    if bad_end {
        e.terminate(true);
    }
    pack(&e.bits)
}

/// A Main profile SPS: 4:2:0 8 bit, CTB 32 = minimum CB, transform blocks 4 to 32 with no
/// split, SAO on, no PCM, no VUI.
fn sps(width: u32, height: u32, big: bool) -> Vec<u8> {
    let mut b = W::default();
    b.u(4, 0); // vps id
    b.u(3, 0); // max_sub_layers_minus1
    b.flag(true);
    // profile_tier_level: Main, level 6.2.
    b.u(2, 0);
    b.flag(false);
    b.u(5, 1);
    b.u(32, 0x6000_0000);
    b.u(48, 0);
    b.u(8, 186);
    for v in [0, 1, width, height] {
        b.ue(v); // sps id, 4:2:0, size
    }
    b.flag(false); // conformance window
    for v in [0, 0, 4] {
        b.ue(v); // bit depths 8, log2_max_pic_order_cnt_lsb 8
    }
    b.flag(true);
    for v in [15, 0, 0] {
        b.ue(v); // max_dec_pic_buffering_minus1, num_reorder, latency
    }
    for v in [2, 0, 0, 3, 0, 0] {
        b.ue(v); // CB 32..32, TB 4..32, transform hierarchy depth 0 and 0
    }
    for f in [false, false, true, false] {
        b.flag(f); // scaling lists, amp, sao, pcm
    }
    if big {
        b.ue(64);
        for i in 0..64 {
            if i > 0 {
                b.flag(false); // inter_ref_pic_set_prediction_flag
            }
            b.ue(16);
            b.ue(0);
            for _ in 0..16 {
                b.ue(0); // delta_poc_s0_minus1
                b.flag(true);
            }
        }
        b.flag(true);
        b.ue(32);
        for i in 0..32 {
            b.u(8, i); // lt_ref_pic_poc_lsb_sps
            b.flag(true);
        }
    } else {
        b.ue(0);
        b.flag(false);
    }
    for f in [false, true, false, false] {
        b.flag(f); // temporal mvp, strong intra smoothing, vui, extension
    }
    b.rbsp()
}

/// A PPS: no tiles, no WPP, init_qp 24, sign data hiding, Cb / Cr QP offsets 1 and -1,
/// deblocking on with default offsets.
fn pps(output_flag_present: bool) -> Vec<u8> {
    let mut b = W::default();
    b.ue(0);
    b.ue(0);
    b.flag(false);
    b.flag(output_flag_present);
    b.u(3, 0);
    b.flag(true); // sign data hiding
    b.flag(false);
    b.ue(0);
    b.ue(0);
    b.se(-2); // init_qp 24
    for _ in 0..3 {
        b.flag(false); // constrained intra, transform skip, cu qp delta
    }
    b.se(1);
    b.se(-1);
    for _ in 0..6 {
        b.flag(false);
    }
    for _ in 0..3 {
        b.flag(false); // loop filter across slices, deblocking control, scaling list
    }
    b.flag(false);
    b.ue(0);
    b.flag(false);
    b.flag(false); // extension
    b.rbsp()
}

/// The slice segment header of an IDR picture's only slice; `output` is pic_output_flag when
/// the PPS has output_flag_present_flag.
fn slice_header(output: Option<bool>) -> W {
    let mut b = W::default();
    b.flag(true); // first_slice_segment_in_pic_flag
    b.flag(false); // no_output_of_prior_pics_flag
    b.ue(0); // pps id
    b.ue(2); // I
    if let Some(o) = output {
        b.flag(o);
    }
    b.flag(true); // slice_sao_luma_flag
    b.flag(true); // slice_sao_chroma_flag
    b.se(QP - 24);
    b
}

/// Escapes 00 00 0x sequences (7.4.2) so the payload holds no start code.
fn escape(p: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(p.len() + 8);
    let mut zeros = 0;
    for &x in p {
        if zeros >= 2 && x <= 3 {
            o.push(3);
            zeros = 0;
        }
        o.push(x);
        zeros = if x == 0 { zeros + 1 } else { 0 };
    }
    o
}

/// The Annex B stream that `s` describes: VPS, SPS and PPS (before every picture with
/// `repeat`), then one IDR_W_RADL picture after the other, each one slice.
pub fn stream(s: &Spec) -> Vec<u8> {
    let flagged = s.hidden > 0;
    let sets = [
        (32u8, vec![0x0C, 0xFF]),
        (33, sps(s.width, s.height, s.big_sps)),
        (34, pps(flagged)),
    ];
    let (w, h) = ((s.width / 32) as usize, (s.height / 32) as usize);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut nals = Vec::new();
    for i in 0..s.pictures {
        if i == 0 || s.repeat {
            nals.extend(sets.iter().cloned());
        }
        let last = i + 1 == s.pictures;
        let output = flagged.then_some(i + s.hidden < s.pictures);
        let mut p = slice_header(output).rbsp();
        p.extend(slice_data(w, h, &mut rng, s.bad_last && last));
        nals.push((19, p));
    }
    let mut out = Vec::new();
    for (kind, p) in nals {
        out.extend_from_slice(&[0, 0, 0, 1, kind << 1, 1]);
        out.extend_from_slice(&escape(&p));
    }
    out
}
