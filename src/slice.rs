//! Slice segment header of I slices (H.265 7.3.6.1, semantics 7.4.7.1).

use std::sync::Arc;

use crate::bits::BitReader;
use crate::error::{Error, Result};
use crate::ps::{Pps, Sps, parse_st_rps};

/// Parsed slice segment header; dependent segments carry the fields of their slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SliceHeader {
    pub first_slice_segment_in_pic: bool,
    pub no_output_of_prior_pics: bool,
    pub pps_id: u32,
    pub dependent: bool,
    /// slice_segment_address (CtbAddrInRs).
    pub segment_address: u32,
    pub pic_output: bool,
    pub poc_lsb: u32,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub qp_delta: i32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub deblocking_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub loop_filter_across_slices: bool,
    /// entry_point_offset_minus1 + 1 of each substream.
    pub entry_points: Vec<u32>,
    /// Byte offset of slice_segment_data in the RBSP.
    pub data_offset: usize,
}

/// Ceil(Log2(n)), 0 for n <= 1.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

fn check_range(v: i32, lo: i32, hi: i32, what: &'static str) -> Result<()> {
    if (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(Error::InvalidData(what))
    }
}

/// Parses one slice segment header from the unescaped NAL payload (after the 2-byte NAL
/// header). `prev` is the last independent slice segment header of the same picture; it is
/// required for a dependent segment. A slice_type other than I returns `Error::InterSlice`
/// before anything after slice_type is read.
pub(crate) fn parse(
    rbsp: &[u8],
    nal_kind: u8,
    sps_list: &[Option<Arc<Sps>>; 16],
    pps_list: &[Option<Arc<Pps>>; 64],
    prev: Option<&SliceHeader>,
) -> Result<SliceHeader> {
    let mut r = BitReader::new(rbsp);
    let first = r.flag()?;
    let no_output_of_prior_pics = if (16..=23).contains(&nal_kind) {
        r.flag()?
    } else {
        false
    };
    let pps_id = r.ue()?;
    if pps_id > 63 {
        return Err(Error::InvalidData("slice pps id out of range"));
    }
    let pps = pps_list
        .get(pps_id as usize)
        .and_then(|p| p.as_ref())
        .ok_or(Error::InvalidData("slice refers to a missing PPS"))?;
    let sps = sps_list
        .get(pps.sps_id as usize)
        .and_then(|s| s.as_ref())
        .ok_or(Error::InvalidData("slice refers to a missing SPS"))?;

    let mut dependent = false;
    let mut segment_address = 0u32;
    if !first {
        if pps.dependent_slice_segments_enabled {
            dependent = r.flag()?;
        }
        let pic_size = sps
            .pic_width_in_ctbs_y
            .checked_mul(sps.pic_height_in_ctbs_y)
            .ok_or(Error::InvalidData("picture size overflow"))?;
        segment_address = r.u(ceil_log2(pic_size))?;
        if segment_address >= pic_size {
            return Err(Error::InvalidData("slice_segment_address out of range"));
        }
    }

    let mut h = if dependent {
        let p = prev.ok_or(Error::InvalidData(
            "dependent slice segment without a slice",
        ))?;
        p.clone()
    } else {
        parse_independent(&mut r, nal_kind, sps, pps)?
    };
    h.first_slice_segment_in_pic = first;
    h.no_output_of_prior_pics = no_output_of_prior_pics;
    h.pps_id = pps_id;
    h.dependent = dependent;
    h.segment_address = segment_address;

    // Entry points (7.3.6.1), bounded by the substream count the PPS allows.
    h.entry_points = Vec::new();
    if pps.tiles_enabled || pps.entropy_coding_sync_enabled {
        let tiles = pps.num_tile_columns.saturating_mul(pps.num_tile_rows);
        let rows = sps.pic_height_in_ctbs_y;
        let max = match (pps.tiles_enabled, pps.entropy_coding_sync_enabled) {
            (true, true) => pps.num_tile_columns.saturating_mul(rows),
            (true, false) => tiles,
            _ => rows,
        }
        .saturating_sub(1);
        let n = r.ue()?;
        if n > max {
            return Err(Error::InvalidData("too many entry points"));
        }
        if n > 0 {
            let len = r.ue()?;
            if len > 31 {
                return Err(Error::InvalidData("offset_len_minus1 out of range"));
            }
            h.entry_points
                .try_reserve_exact(n as usize)
                .map_err(|_| Error::OutOfMemory)?;
            for _ in 0..n {
                let off = r.u(len + 1)?;
                h.entry_points.push(off.saturating_add(1));
            }
        }
    }

    if pps.slice_segment_header_extension_present {
        let len = r.ue()?;
        if len > 256 {
            return Err(Error::InvalidData("slice header extension too long"));
        }
        r.skip(len as usize * 8)?;
    }

    // byte_alignment(): one 1 bit, then 0 bits up to the byte boundary.
    if !r.flag()? {
        return Err(Error::InvalidData("slice header byte_alignment bit"));
    }
    while !r.byte_aligned() {
        if r.flag()? {
            return Err(Error::InvalidData("slice header alignment zero bits"));
        }
    }
    h.data_offset = r.bit_pos() / 8;
    Ok(h)
}

/// The fields present only in an independent slice segment.
fn parse_independent(r: &mut BitReader, nal_kind: u8, sps: &Sps, pps: &Pps) -> Result<SliceHeader> {
    r.skip(pps.num_extra_slice_header_bits as usize)?;
    // 7.4.7.1: 0 = B, 1 = P, 2 = I.
    let slice_type = r.ue()?;
    if slice_type > 2 {
        return Err(Error::InvalidData("slice_type out of range"));
    }
    if slice_type != 2 {
        return Err(Error::InterSlice);
    }
    let pic_output = if pps.output_flag_present {
        r.flag()?
    } else {
        true
    };

    let mut poc_lsb = 0;
    if nal_kind != 19 && nal_kind != 20 {
        poc_lsb = r.u(sps.log2_max_poc_lsb)?;
        let num_sets = sps.st_rps.len();
        if !r.flag()? {
            parse_st_rps(r, num_sets, &sps.st_rps)?;
        } else if num_sets == 0 {
            return Err(Error::InvalidData(
                "short_term_ref_pic_set_sps_flag without SPS sets",
            ));
        } else if num_sets > 1 {
            let idx = r.u(ceil_log2(num_sets as u32))? as usize;
            if idx >= num_sets {
                return Err(Error::InvalidData(
                    "short_term_ref_pic_set_idx out of range",
                ));
            }
        }
        if sps.long_term_ref_pics_present {
            let num_lt_sps = sps.lt_ref_pic_poc_lsb_sps.len() as u32;
            let from_sps = if num_lt_sps > 0 { r.ue()? } else { 0 };
            if from_sps > num_lt_sps {
                return Err(Error::InvalidData("num_long_term_sps out of range"));
            }
            let from_slice = r.ue()?;
            if from_slice > 32 {
                return Err(Error::InvalidData("num_long_term_pics out of range"));
            }
            for i in 0..from_sps + from_slice {
                if i < from_sps {
                    if num_lt_sps > 1 {
                        r.u(ceil_log2(num_lt_sps))?;
                    }
                } else {
                    r.u(sps.log2_max_poc_lsb)?;
                    r.flag()?;
                }
                if r.flag()? {
                    r.ue()?;
                }
            }
        }
        if sps.temporal_mvp_enabled {
            r.flag()?;
        }
    }

    let (mut sao_luma, mut sao_chroma) = (false, false);
    if sps.sao_enabled {
        sao_luma = r.flag()?;
        if sps.chroma_format_idc != 0 {
            sao_chroma = r.flag()?;
        }
    }

    // No P/B fields. 7.4.7.1: SliceQpY in -QpBdOffsetY..=51.
    let qp_delta = r.se()?;
    let slice_qp = pps.init_qp.saturating_add(qp_delta);
    check_range(slice_qp, -sps.qp_bd_offset_y, 51, "SliceQpY out of range")?;

    let (mut cb_qp_offset, mut cr_qp_offset) = (0, 0);
    if pps.slice_chroma_qp_offsets_present {
        cb_qp_offset = r.se()?;
        cr_qp_offset = r.se()?;
        check_range(cb_qp_offset, -12, 12, "slice_cb_qp_offset out of range")?;
        check_range(cr_qp_offset, -12, 12, "slice_cr_qp_offset out of range")?;
        check_range(
            pps.cb_qp_offset + cb_qp_offset,
            -12,
            12,
            "pps+slice cb qp offset",
        )?;
        check_range(
            pps.cr_qp_offset + cr_qp_offset,
            -12,
            12,
            "pps+slice cr qp offset",
        )?;
    }

    let override_flag = if pps.deblocking_override_enabled {
        r.flag()?
    } else {
        false
    };
    let mut deblocking_disabled = pps.deblocking_disabled;
    let mut beta_offset_div2 = pps.beta_offset_div2;
    let mut tc_offset_div2 = pps.tc_offset_div2;
    if override_flag {
        deblocking_disabled = r.flag()?;
        if !deblocking_disabled {
            beta_offset_div2 = r.se()?;
            tc_offset_div2 = r.se()?;
            check_range(
                beta_offset_div2,
                -6,
                6,
                "slice_beta_offset_div2 out of range",
            )?;
            check_range(tc_offset_div2, -6, 6, "slice_tc_offset_div2 out of range")?;
        }
    }

    let mut loop_filter_across_slices = pps.loop_filter_across_slices_enabled;
    if pps.loop_filter_across_slices_enabled && (sao_luma || sao_chroma || !deblocking_disabled) {
        loop_filter_across_slices = r.flag()?;
    }

    Ok(SliceHeader {
        first_slice_segment_in_pic: false,
        no_output_of_prior_pics: false,
        pps_id: 0,
        dependent: false,
        segment_address: 0,
        pic_output,
        poc_lsb,
        sao_luma,
        sao_chroma,
        qp_delta,
        cb_qp_offset,
        cr_qp_offset,
        deblocking_disabled,
        beta_offset_div2,
        tc_offset_div2,
        loop_filter_across_slices,
        entry_points: Vec::new(),
        data_offset: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_values() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(u32::MAX), 32);
    }

    #[test]
    fn missing_pps_is_invalid() {
        let sps: [Option<Arc<Sps>>; 16] = Default::default();
        let pps: [Option<Arc<Pps>>; 64] = std::array::from_fn(|_| None);
        // first_slice = 1, IDR, pps id ue(0) = '1'
        let r = parse(&[0b1011_0000], 19, &sps, &pps, None);
        assert!(matches!(r, Err(Error::InvalidData(_))));
    }
}
