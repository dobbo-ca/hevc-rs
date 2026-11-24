//! Parameter sets: VPS (id only), SPS (7.3.2.2), PPS (7.3.2.3), `profile_tier_level` (7.3.3),
//! `st_ref_pic_set` (7.3.7), VUI and HRD (E.2.1 to E.2.3; the VUI colour and display fields are
//! kept, the rest is parsed and dropped), and activation
//! (6.5.1 and 6.5.2 scan tables).

use std::sync::Arc;

use crate::Vui;
use crate::bits::BitReader;
use crate::error::{Error, Result};
use crate::ledger::Ledger;
use crate::limits::Limits;
use crate::scaling::ScalingList;

/// Parameter set payloads over this size are refused.
const MAX_PS_BYTES: usize = 1 << 20;
/// Most CTB columns or rows of the tile grid that is accepted at parse time (16384 / 16).
const MAX_TILES: u32 = 1024;

fn bad<T>(what: &'static str) -> Result<T> {
    Err(Error::InvalidData(what))
}

/// `ue(v)` that must be at most `max`.
fn ue_max(r: &mut BitReader, max: u32, what: &'static str) -> Result<u32> {
    let v = r.ue()?;
    if v > max { bad(what) } else { Ok(v) }
}

/// `se(v)` that must be in `lo..=hi`.
fn se_range(r: &mut BitReader, lo: i32, hi: i32, what: &'static str) -> Result<i32> {
    let v = r.se()?;
    if (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        bad(what)
    }
}

fn set_u32(v: &mut [u32], i: usize, val: u32) -> Result<()> {
    match v.get_mut(i) {
        Some(d) => {
            *d = val;
            Ok(())
        }
        None => bad("scan table index"),
    }
}

// ---------------------------------------------------------------------------------------------
// VPS

/// The only VPS field the decoder uses.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code, reason = "the VPS is parsed for validation only")]
pub(crate) struct Vps {
    pub id: u8,
}

/// 7.3.2.1, read as far as `vps_video_parameter_set_id`; the rest is not needed for intra
/// decoding of the base layer.
pub(crate) fn parse_vps(rbsp: &[u8]) -> Result<Vps> {
    if rbsp.len() > MAX_PS_BYTES {
        return bad("vps too large");
    }
    let mut r = BitReader::new(rbsp);
    Ok(Vps { id: r.u(4)? as u8 })
}

// ---------------------------------------------------------------------------------------------
// st_ref_pic_set

/// One short-term reference picture set (7.4.8). `delta_poc` holds DeltaPocS0 (the
/// `num_negative` entries) followed by DeltaPocS1 (`num_positive` entries); `used` is
/// UsedByCurrPicS0 / S1 in the same order. NumDeltaPocs is `delta_poc.len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StRps {
    pub num_negative: u32,
    pub num_positive: u32,
    pub delta_poc: Vec<i32>,
    pub used: Vec<bool>,
}

/// st_ref_pic_set(idx) of a slice header: `idx` is `num_short_term_ref_pic_sets` and `sets` are
/// all the SPS sets (so `delta_idx_minus1` is present when `idx == sets.len()`).
pub(crate) fn parse_st_rps(r: &mut BitReader, idx: usize, sets: &[StRps]) -> Result<StRps> {
    st_rps(r, idx, sets, idx == sets.len())
}

/// 7.3.7 and 7.4.8. `sets` are the sets before `idx`. `delta_idx_present` is true only for the
/// set in a slice header (stRpsIdx == num_short_term_ref_pic_sets).
fn st_rps(r: &mut BitReader, idx: usize, sets: &[StRps], delta_idx_present: bool) -> Result<StRps> {
    let inter = if idx != 0 { r.flag()? } else { false };
    let mut neg: Vec<(i32, bool)> = Vec::new();
    let mut pos: Vec<(i32, bool)> = Vec::new();
    if inter {
        let delta_idx = if delta_idx_present {
            r.ue()? as usize + 1
        } else {
            1
        };
        if delta_idx > idx {
            return bad("delta_idx_minus1");
        }
        let Some(rf) = sets.get(idx - delta_idx) else {
            return bad("reference rps");
        };
        let sign = r.flag()?;
        let abs = ue_max(r, 32767, "abs_delta_rps_minus1")? as i32 + 1;
        let delta_rps = if sign { -abs } else { abs };
        let n = rf.delta_poc.len();
        let mut used_flag = Vec::new();
        let mut use_delta = Vec::new();
        for _ in 0..=n {
            let u = r.flag()?;
            used_flag.push(u);
            use_delta.push(if u { true } else { r.flag()? });
        }
        let uf = |j: usize| used_flag.get(j).copied().unwrap_or(false);
        let ud = |j: usize| use_delta.get(j).copied().unwrap_or(false);
        let nn = rf.num_negative as usize;
        let np = rf.num_positive as usize;
        let dp = |k: usize| rf.delta_poc.get(k).copied().unwrap_or(0);
        for j in (0..np).rev() {
            let d = dp(nn + j) + delta_rps;
            if d < 0 && ud(nn + j) {
                neg.push((d, uf(nn + j)));
            }
        }
        if delta_rps < 0 && ud(n) {
            neg.push((delta_rps, uf(n)));
        }
        for j in 0..nn {
            let d = dp(j) + delta_rps;
            if d < 0 && ud(j) {
                neg.push((d, uf(j)));
            }
        }
        for j in (0..nn).rev() {
            let d = dp(j) + delta_rps;
            if d > 0 && ud(j) {
                pos.push((d, uf(j)));
            }
        }
        if delta_rps > 0 && ud(n) {
            pos.push((delta_rps, uf(n)));
        }
        for j in 0..np {
            let d = dp(nn + j) + delta_rps;
            if d > 0 && ud(nn + j) {
                pos.push((d, uf(nn + j)));
            }
        }
    } else {
        let nn = ue_max(r, 16, "num_negative_pics")?;
        let np = ue_max(r, 16, "num_positive_pics")?;
        if nn + np > 16 {
            return bad("too many pictures in a reference picture set");
        }
        let mut cur = 0i32;
        for _ in 0..nn {
            cur -= ue_max(r, 32767, "delta_poc_s0_minus1")? as i32 + 1;
            neg.push((cur, r.flag()?));
        }
        cur = 0;
        for _ in 0..np {
            cur += ue_max(r, 32767, "delta_poc_s1_minus1")? as i32 + 1;
            pos.push((cur, r.flag()?));
        }
    }
    if neg.len() + pos.len() > 16 {
        return bad("too many pictures in a reference picture set");
    }
    Ok(StRps {
        num_negative: neg.len() as u32,
        num_positive: pos.len() as u32,
        delta_poc: neg.iter().chain(pos.iter()).map(|&(d, _)| d).collect(),
        used: neg.iter().chain(pos.iter()).map(|&(_, u)| u).collect(),
    })
}

// ---------------------------------------------------------------------------------------------
// profile_tier_level, VUI, HRD

/// 7.3.3 with profilePresentFlag = 1. Returns (general_profile_idc, general_level_idc).
/// The profile is not checked here: the supported tools are checked field by field (Main,
/// Main 10 and Main Still Picture are idc 1 to 3; other profiles pass when they use no tool we
/// reject).
fn profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<(u8, u8)> {
    r.skip(3)?; // general_profile_space, general_tier_flag
    let profile_idc = r.u(5)? as u8;
    r.skip(32)?; // general_profile_compatibility_flag[0..32]
    r.skip(48)?; // progressive .. frame_only flags, 43 reserved/flag bits, inbld/reserved bit
    let level_idc = r.u(8)? as u8;
    let n = max_sub_layers_minus1 as usize;
    let mut sub = [(false, false); 8];
    for s in sub.iter_mut().take(n) {
        *s = (r.flag()?, r.flag()?);
    }
    if n > 0 {
        r.skip(2 * (8 - n))?; // reserved_zero_2bits
    }
    for &(profile_present, level_present) in sub.iter().take(n) {
        if profile_present {
            r.skip(88)?;
        }
        if level_present {
            r.skip(8)?;
        }
    }
    Ok((profile_idc, level_idc))
}

/// E.2.3 sub_layer_hrd_parameters.
fn sub_layer_hrd(r: &mut BitReader, cpb_cnt: u32, sub_pic: bool) -> Result<()> {
    for _ in 0..cpb_cnt {
        r.ue()?;
        r.ue()?;
        if sub_pic {
            r.ue()?;
            r.ue()?;
        }
        r.skip(1)?; // cbr_flag
    }
    Ok(())
}

/// E.2.2 hrd_parameters(1, max_sub_layers_minus1).
fn hrd_parameters(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<()> {
    let nal = r.flag()?;
    let vcl = r.flag()?;
    let mut sub_pic = false;
    if nal || vcl {
        sub_pic = r.flag()?;
        if sub_pic {
            r.skip(8 + 5 + 1 + 5)?;
        }
        r.skip(4 + 4)?;
        if sub_pic {
            r.skip(4)?;
        }
        r.skip(5 + 5 + 5)?;
    }
    for _ in 0..=max_sub_layers_minus1 {
        let general = r.flag()?;
        let within_cvs = if general { true } else { r.flag()? };
        let mut low_delay = false;
        if within_cvs {
            r.ue()?; // elemental_duration_in_tc_minus1
        } else {
            low_delay = r.flag()?;
        }
        let cpb_cnt = if low_delay {
            1
        } else {
            ue_max(r, 31, "cpb_cnt_minus1")? + 1
        };
        if nal {
            sub_layer_hrd(r, cpb_cnt, sub_pic)?;
        }
        if vcl {
            sub_layer_hrd(r, cpb_cnt, sub_pic)?;
        }
    }
    Ok(())
}

/// E.2.1 vui_parameters, read completely (the extension flags follow it). Returns the colour and
/// display fields; the timing, HRD and bitstream restriction parts are parsed and dropped.
fn vui_parameters(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<Vui> {
    let mut vui = Vui {
        aspect_ratio_idc: 0,
        aspect_ratio: None,
        video_format: 5,
        video_full_range: false,
        colour_primaries: 2,
        transfer_characteristics: 2,
        matrix_coeffs: 2,
        chroma_sample_loc_type_top: 0,
        chroma_sample_loc_type_bottom: 0,
        default_display_window: None,
    };
    if r.flag()? {
        vui.aspect_ratio_idc = r.u(8)? as u8;
        if vui.aspect_ratio_idc == 255 {
            vui.aspect_ratio = Some((r.u(16)? as u16, r.u(16)? as u16));
        }
    }
    if r.flag()? {
        r.skip(1)?; // overscan_appropriate_flag
    }
    if r.flag()? {
        vui.video_format = r.u(3)? as u8;
        vui.video_full_range = r.flag()?;
        if r.flag()? {
            vui.colour_primaries = r.u(8)? as u8;
            vui.transfer_characteristics = r.u(8)? as u8;
            vui.matrix_coeffs = r.u(8)? as u8;
        }
    }
    if r.flag()? {
        vui.chroma_sample_loc_type_top = u8::try_from(r.ue()?).unwrap_or(u8::MAX);
        vui.chroma_sample_loc_type_bottom = u8::try_from(r.ue()?).unwrap_or(u8::MAX);
    }
    r.skip(3)?; // neutral_chroma_indication, field_seq, frame_field_info_present
    if r.flag()? {
        vui.default_display_window = Some([r.ue()?, r.ue()?, r.ue()?, r.ue()?]);
    }
    if r.flag()? {
        r.skip(64)?; // num_units_in_tick, time_scale
        if r.flag()? {
            r.ue()?; // num_ticks_poc_diff_one_minus1
        }
        if r.flag()? {
            hrd_parameters(r, max_sub_layers_minus1)?;
        }
    }
    if r.flag()? {
        r.skip(3)?; // tiles_fixed_structure, motion_vectors_over_pic_boundaries, restricted_ref_pic_lists
        for _ in 0..5 {
            r.ue()?;
        }
    }
    Ok(vui)
}

// ---------------------------------------------------------------------------------------------
// SPS

/// sps_max_dec_pic_buffering_minus1, sps_max_num_reorder_pics and
/// sps_max_latency_increase_plus1 of one sub-layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubLayerOrdering {
    pub max_dec_pic_buffering_minus1: u32,
    pub max_num_reorder_pics: u32,
    pub max_latency_increase_plus1: u32,
}

/// PCM parameters (7.4.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pcm {
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_min_size: u32,
    pub log2_max_size: u32,
    pub loop_filter_disabled: bool,
}

/// A validated sequence parameter set.
#[derive(Debug, Clone)]
#[allow(
    dead_code,
    reason = "inter-only and informative fields are parsed but never read"
)]
pub(crate) struct Sps {
    pub vps_id: u8,
    pub sps_id: u32,
    pub max_sub_layers_minus1: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    /// 0 (4:0:0) or 1 (4:2:0); equals ChromaArrayType.
    pub chroma_format_idc: u32,
    pub width: u32,
    pub height: u32,
    /// Conformance window offsets in luma samples (already times SubWidthC / SubHeightC).
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_poc_lsb: u32,
    /// One entry per sub-layer (index 0..=max_sub_layers_minus1), filled from the highest when
    /// the stream sends only one.
    pub ordering: Vec<SubLayerOrdering>,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    /// None when scaling_list_enabled_flag is 0.
    pub scaling: Option<ScalingList>,
    pub amp_enabled: bool,
    pub sao_enabled: bool,
    pub pcm: Option<Pcm>,
    pub st_rps: Vec<StRps>,
    pub long_term_ref_pics_present: bool,
    pub lt_ref_pic_poc_lsb_sps: Vec<u32>,
    pub used_by_curr_pic_lt_sps: Vec<bool>,
    pub temporal_mvp_enabled: bool,
    pub strong_intra_smoothing_enabled: bool,
    pub high_precision_offsets_enabled: bool,
    pub vui: Option<Vui>,
    // Derived values (7.4.3.2).
    pub min_cb_log2_size_y: u32,
    pub ctb_log2_size_y: u32,
    pub ctb_size_y: u32,
    pub pic_width_in_ctbs_y: u32,
    pub pic_height_in_ctbs_y: u32,
    pub pic_width_in_min_cbs_y: u32,
    pub pic_height_in_min_cbs_y: u32,
    pub sub_width_c: u32,
    pub sub_height_c: u32,
    pub qp_bd_offset_y: i32,
    pub qp_bd_offset_c: i32,
    pub log2_min_trafo_size: u32,
    pub log2_max_trafo_size: u32,
}

impl Sps {
    /// Checks the coded size against the caller's limits (done when a picture activates the
    /// SPS, and by probe).
    pub(crate) fn check_limits(&self, limits: &Limits) -> Result<()> {
        if self.width > limits.max_width {
            return Err(Error::WidthLimit);
        }
        if self.height > limits.max_height {
            return Err(Error::HeightLimit);
        }
        if u64::from(self.width) * u64::from(self.height) > limits.max_pixels {
            return Err(Error::PixelLimit);
        }
        Ok(())
    }
}

/// 7.3.2.2 sequence_parameter_set_rbsp. `rbsp` is the payload after the NAL header with
/// emulation prevention bytes removed.
pub(crate) fn parse_sps(rbsp: &[u8]) -> Result<Sps> {
    if rbsp.len() > MAX_PS_BYTES {
        return bad("sps too large");
    }
    let mut r = BitReader::new(rbsp);
    let vps_id = r.u(4)? as u8;
    let max_sub_layers_minus1 = r.u(3)?;
    if max_sub_layers_minus1 > 6 {
        return bad("sps_max_sub_layers_minus1");
    }
    r.skip(1)?; // sps_temporal_id_nesting_flag
    let (profile_idc, level_idc) = profile_tier_level(&mut r, max_sub_layers_minus1)?;
    let sps_id = ue_max(&mut r, 15, "sps_seq_parameter_set_id")?;
    let chroma_format_idc = ue_max(&mut r, 3, "chroma_format_idc")?;
    if chroma_format_idc >= 2 {
        return Err(Error::Unsupported("chroma_format_idc 2 or 3"));
    }
    let width = r.ue()?;
    let height = r.ue()?;
    if width == 0 || height == 0 {
        return bad("empty picture");
    }
    let (sub_width_c, sub_height_c) = if chroma_format_idc == 1 {
        (2, 2)
    } else {
        (1, 1)
    };
    let mut crop = [0u32; 4];
    if r.flag()? {
        for c in &mut crop {
            *c = r.ue()?;
        }
    }
    let [cl, cr, ct, cb] = crop;
    let crop_l = u64::from(cl) * u64::from(sub_width_c);
    let crop_r = u64::from(cr) * u64::from(sub_width_c);
    let crop_t = u64::from(ct) * u64::from(sub_height_c);
    let crop_b = u64::from(cb) * u64::from(sub_height_c);
    if crop_l + crop_r >= u64::from(width) || crop_t + crop_b >= u64::from(height) {
        return bad("conformance window");
    }
    let bit_depth_luma = ue_max(&mut r, 8, "bit_depth_luma_minus8")? + 8;
    let bit_depth_chroma = ue_max(&mut r, 8, "bit_depth_chroma_minus8")? + 8;
    if bit_depth_luma > 12 || bit_depth_chroma > 12 {
        return Err(Error::Unsupported("bit depth over 12"));
    }
    let log2_max_poc_lsb = ue_max(&mut r, 12, "log2_max_pic_order_cnt_lsb_minus4")? + 4;

    let present = r.flag()?;
    let first = if present { 0 } else { max_sub_layers_minus1 };
    let mut ordering = Vec::new();
    for _ in first..=max_sub_layers_minus1 {
        let max_dec = ue_max(&mut r, 15, "sps_max_dec_pic_buffering_minus1")?;
        let reorder = ue_max(&mut r, max_dec, "sps_max_num_reorder_pics")?;
        let latency = r.ue()?;
        ordering.push(SubLayerOrdering {
            max_dec_pic_buffering_minus1: max_dec,
            max_num_reorder_pics: reorder,
            max_latency_increase_plus1: latency,
        });
    }
    if let Some(&last) = ordering.last() {
        while ordering.len() <= max_sub_layers_minus1 as usize {
            ordering.push(last);
        }
    }

    let min_cb_log2_size_y = ue_max(&mut r, 3, "log2_min_luma_coding_block_size_minus3")? + 3;
    let min_cb_mask = (1u32 << min_cb_log2_size_y) - 1;
    if width & min_cb_mask != 0 || height & min_cb_mask != 0 {
        return bad("picture size is not a multiple of MinCbSizeY");
    }
    let ctb_log2_size_y =
        min_cb_log2_size_y + ue_max(&mut r, 3, "log2_diff_max_min_luma_coding_block_size")?;
    if !(4..=6).contains(&ctb_log2_size_y) {
        return bad("CtbLog2SizeY");
    }
    let log2_min_trafo_size = ue_max(&mut r, 3, "log2_min_luma_transform_block_size_minus2")? + 2;
    if log2_min_trafo_size >= min_cb_log2_size_y {
        return bad("log2_min_luma_transform_block_size");
    }
    let log2_max_trafo_size =
        log2_min_trafo_size + ue_max(&mut r, 3, "log2_diff_max_min_luma_transform_block_size")?;
    if log2_max_trafo_size > ctb_log2_size_y.min(5) {
        return bad("log2_max_luma_transform_block_size");
    }
    let max_depth = ctb_log2_size_y - log2_min_trafo_size;
    let max_transform_hierarchy_depth_inter =
        ue_max(&mut r, max_depth, "max_transform_hierarchy_depth_inter")?;
    let max_transform_hierarchy_depth_intra =
        ue_max(&mut r, max_depth, "max_transform_hierarchy_depth_intra")?;

    let mut scaling = None;
    if r.flag()? {
        scaling = Some(if r.flag()? {
            ScalingList::parse(&mut r)?
        } else {
            ScalingList::default_lists()
        });
    }
    let amp_enabled = r.flag()?;
    let sao_enabled = r.flag()?;
    let mut pcm = None;
    if r.flag()? {
        let bl = r.u(4)? as u8 + 1;
        let bc = r.u(4)? as u8 + 1;
        if u32::from(bl) > bit_depth_luma || u32::from(bc) > bit_depth_chroma {
            return bad("pcm sample bit depth");
        }
        let pcm_max = ctb_log2_size_y.min(5);
        let log2_min_size = ue_max(&mut r, 2, "log2_min_pcm_luma_coding_block_size_minus3")? + 3;
        let log2_max_size =
            log2_min_size + ue_max(&mut r, 2, "log2_diff_max_min_pcm_luma_coding_block_size")?;
        if log2_min_size > pcm_max || log2_max_size > pcm_max {
            return bad("pcm coding block size");
        }
        pcm = Some(Pcm {
            bit_depth_luma: bl,
            bit_depth_chroma: bc,
            log2_min_size,
            log2_max_size,
            loop_filter_disabled: r.flag()?,
        });
    }

    let num_st = ue_max(&mut r, 64, "num_short_term_ref_pic_sets")? as usize;
    let mut st_rps_sets: Vec<StRps> = Vec::new();
    for i in 0..num_st {
        let s = st_rps(&mut r, i, &st_rps_sets, false)?;
        st_rps_sets.push(s);
    }
    let long_term_ref_pics_present = r.flag()?;
    let mut lt_ref_pic_poc_lsb_sps = Vec::new();
    let mut used_by_curr_pic_lt_sps = Vec::new();
    if long_term_ref_pics_present {
        let n = ue_max(&mut r, 32, "num_long_term_ref_pics_sps")?;
        for _ in 0..n {
            lt_ref_pic_poc_lsb_sps.push(r.u(log2_max_poc_lsb)?);
            used_by_curr_pic_lt_sps.push(r.flag()?);
        }
    }
    let temporal_mvp_enabled = r.flag()?;
    let strong_intra_smoothing_enabled = r.flag()?;
    let vui = if r.flag()? {
        Some(vui_parameters(&mut r, max_sub_layers_minus1)?)
    } else {
        None
    };
    let mut high_precision_offsets_enabled = false;
    // 7.4.3.2.1: in Main, Main 10 and Main Still Picture bitstreams the extension flags are 0,
    // and a decoder for those profiles ignores whatever follows sps_extension_present_flag
    // (version 1 streams carry sps_extension_data_flag bits there, e.g. PS_B_VIDYO_3).
    if r.flag()? && !(1..=3).contains(&profile_idc) {
        let range = r.flag()?;
        let multilayer = r.flag()?;
        let ext_3d = r.flag()?;
        let scc = r.flag()?;
        let ext_4bits = r.u(4)?;
        if multilayer || ext_3d || scc || ext_4bits != 0 {
            return Err(Error::Unsupported("sps multilayer, 3d or scc extension"));
        }
        if range {
            // 7.3.2.2.2: transform_skip_rotation, transform_skip_context, implicit_rdpcm,
            // explicit_rdpcm, extended_precision, intra_smoothing_disabled are rejected;
            // then high_precision_offsets (accepted), persistent_rice, cabac_bypass_alignment.
            let first_six = r.u(6)?;
            high_precision_offsets_enabled = r.flag()?;
            let last_two = r.u(2)?;
            if first_six != 0 || last_two != 0 {
                return Err(Error::Unsupported("sps range extension tool"));
            }
        }
    }

    Ok(Sps {
        vps_id,
        sps_id,
        max_sub_layers_minus1,
        profile_idc,
        level_idc,
        chroma_format_idc,
        width,
        height,
        crop_left: crop_l as u32,
        crop_right: crop_r as u32,
        crop_top: crop_t as u32,
        crop_bottom: crop_b as u32,
        bit_depth_luma,
        bit_depth_chroma,
        log2_max_poc_lsb,
        ordering,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling,
        amp_enabled,
        sao_enabled,
        pcm,
        st_rps: st_rps_sets,
        long_term_ref_pics_present,
        lt_ref_pic_poc_lsb_sps,
        used_by_curr_pic_lt_sps,
        temporal_mvp_enabled,
        strong_intra_smoothing_enabled,
        high_precision_offsets_enabled,
        vui,
        min_cb_log2_size_y,
        ctb_log2_size_y,
        ctb_size_y: 1 << ctb_log2_size_y,
        pic_width_in_ctbs_y: width.div_ceil(1 << ctb_log2_size_y),
        pic_height_in_ctbs_y: height.div_ceil(1 << ctb_log2_size_y),
        pic_width_in_min_cbs_y: width >> min_cb_log2_size_y,
        pic_height_in_min_cbs_y: height >> min_cb_log2_size_y,
        sub_width_c,
        sub_height_c,
        qp_bd_offset_y: 6 * (bit_depth_luma as i32 - 8),
        qp_bd_offset_c: 6 * (bit_depth_chroma as i32 - 8),
        log2_min_trafo_size,
        log2_max_trafo_size,
    })
}

// ---------------------------------------------------------------------------------------------
// PPS

/// A validated picture parameter set. Fields that depend on the SPS are checked in `activate`.
#[derive(Debug, Clone)]
#[allow(dead_code, reason = "inter-only fields are parsed but never read")]
pub(crate) struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present: bool,
    /// 26 + init_qp_minus26.
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip_enabled: bool,
    pub cu_qp_delta_enabled: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass_enabled: bool,
    pub tiles_enabled: bool,
    pub entropy_coding_sync_enabled: bool,
    pub num_tile_columns: u32,
    pub num_tile_rows: u32,
    pub uniform_spacing: bool,
    /// Explicit widths in CTBs of the first `num_tile_columns - 1` columns.
    pub column_width: Vec<u32>,
    /// Explicit heights in CTBs of the first `num_tile_rows - 1` rows.
    pub row_height: Vec<u32>,
    pub loop_filter_across_tiles_enabled: bool,
    pub loop_filter_across_slices_enabled: bool,
    pub deblocking_override_enabled: bool,
    pub deblocking_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    /// Some only when pps_scaling_list_data_present_flag is 1.
    pub scaling: Option<ScalingList>,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level: u32,
    pub slice_segment_header_extension_present: bool,
    pub log2_sao_offset_scale_luma: u32,
    pub log2_sao_offset_scale_chroma: u32,
    /// An unsupported or malformed pps_extension; ignored for profiles 1..=3 (see `activate`).
    pub extension_error: Option<Error>,
}

/// 7.3.2.3 pic_parameter_set_rbsp (with the range extension of 7.3.2.3.2).
pub(crate) fn parse_pps(rbsp: &[u8]) -> Result<Pps> {
    if rbsp.len() > MAX_PS_BYTES {
        return bad("pps too large");
    }
    let mut r = BitReader::new(rbsp);
    let pps_id = ue_max(&mut r, 63, "pps_pic_parameter_set_id")?;
    let sps_id = ue_max(&mut r, 15, "pps_seq_parameter_set_id")?;
    let dependent_slice_segments_enabled = r.flag()?;
    let output_flag_present = r.flag()?;
    let num_extra_slice_header_bits = r.u(3)?;
    let sign_data_hiding_enabled = r.flag()?;
    let cabac_init_present = r.flag()?;
    ue_max(&mut r, 14, "num_ref_idx_l0_default_active_minus1")?;
    ue_max(&mut r, 14, "num_ref_idx_l1_default_active_minus1")?;
    let init_qp = 26i32.saturating_add(r.se()?);
    let constrained_intra_pred = r.flag()?;
    let transform_skip_enabled = r.flag()?;
    let cu_qp_delta_enabled = r.flag()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled {
        ue_max(&mut r, 3, "diff_cu_qp_delta_depth")?
    } else {
        0
    };
    let cb_qp_offset = se_range(&mut r, -12, 12, "pps_cb_qp_offset")?;
    let cr_qp_offset = se_range(&mut r, -12, 12, "pps_cr_qp_offset")?;
    let slice_chroma_qp_offsets_present = r.flag()?;
    let weighted_pred = r.flag()?;
    let weighted_bipred = r.flag()?;
    let transquant_bypass_enabled = r.flag()?;
    let tiles_enabled = r.flag()?;
    let entropy_coding_sync_enabled = r.flag()?;

    let mut num_tile_columns = 1;
    let mut num_tile_rows = 1;
    let mut uniform_spacing = true;
    let mut column_width = Vec::new();
    let mut row_height = Vec::new();
    let mut loop_filter_across_tiles_enabled = true;
    if tiles_enabled {
        num_tile_columns = ue_max(&mut r, MAX_TILES - 1, "num_tile_columns_minus1")? + 1;
        num_tile_rows = ue_max(&mut r, MAX_TILES - 1, "num_tile_rows_minus1")? + 1;
        uniform_spacing = r.flag()?;
        if !uniform_spacing {
            for _ in 1..num_tile_columns {
                column_width.push(ue_max(&mut r, u32::MAX - 2, "column_width_minus1")? + 1);
            }
            for _ in 1..num_tile_rows {
                row_height.push(ue_max(&mut r, u32::MAX - 2, "row_height_minus1")? + 1);
            }
        }
        loop_filter_across_tiles_enabled = r.flag()?;
    }
    let loop_filter_across_slices_enabled = r.flag()?;
    let mut deblocking_override_enabled = false;
    let mut deblocking_disabled = false;
    let mut beta_offset_div2 = 0;
    let mut tc_offset_div2 = 0;
    if r.flag()? {
        deblocking_override_enabled = r.flag()?;
        deblocking_disabled = r.flag()?;
        if !deblocking_disabled {
            beta_offset_div2 = se_range(&mut r, -6, 6, "pps_beta_offset_div2")?;
            tc_offset_div2 = se_range(&mut r, -6, 6, "pps_tc_offset_div2")?;
        }
    }
    let scaling = if r.flag()? {
        Some(ScalingList::parse(&mut r)?)
    } else {
        None
    };
    let lists_modification_present = r.flag()?;
    let log2_parallel_merge_level = ue_max(&mut r, 4, "log2_parallel_merge_level_minus2")? + 2;
    let slice_segment_header_extension_present = r.flag()?;
    // The PPS does not know the profile. The extension is parsed here and judged in `activate`:
    // for profiles 1..=3 it is ignored (7.4.3.3.1), for others an error in it is returned there.
    let (mut log2_sao_offset_scale_luma, mut log2_sao_offset_scale_chroma) = (0, 0);
    let mut extension_error = None;
    if r.flag()? {
        match pps_extension(&mut r, transform_skip_enabled) {
            Ok((luma, chroma)) => {
                log2_sao_offset_scale_luma = luma;
                log2_sao_offset_scale_chroma = chroma;
            }
            Err(e) => extension_error = Some(e),
        }
    }
    Ok(Pps {
        pps_id,
        sps_id,
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled,
        cabac_init_present,
        init_qp,
        constrained_intra_pred,
        transform_skip_enabled,
        cu_qp_delta_enabled,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        weighted_pred,
        weighted_bipred,
        transquant_bypass_enabled,
        tiles_enabled,
        entropy_coding_sync_enabled,
        num_tile_columns,
        num_tile_rows,
        uniform_spacing,
        column_width,
        row_height,
        loop_filter_across_tiles_enabled,
        loop_filter_across_slices_enabled,
        deblocking_override_enabled,
        deblocking_disabled,
        beta_offset_div2,
        tc_offset_div2,
        scaling,
        lists_modification_present,
        log2_parallel_merge_level,
        slice_segment_header_extension_present,
        log2_sao_offset_scale_luma,
        log2_sao_offset_scale_chroma,
        extension_error,
    })
}

/// pps_extension_4bits etc. and pps_range_extension (7.3.2.3.2): the SAO offset scales, or the
/// error for an unsupported tool.
fn pps_extension(r: &mut BitReader, transform_skip_enabled: bool) -> Result<(u32, u32)> {
    let range = r.flag()?;
    let multilayer = r.flag()?;
    let ext_3d = r.flag()?;
    let scc = r.flag()?;
    let ext_4bits = r.u(4)?;
    if multilayer || ext_3d || scc || ext_4bits != 0 {
        return Err(Error::Unsupported("pps multilayer, 3d or scc extension"));
    }
    if !range {
        return Ok((0, 0));
    }
    if transform_skip_enabled && r.ue()? != 0 {
        return Err(Error::Unsupported(
            "log2_max_transform_skip_block_size_minus2",
        ));
    }
    if r.flag()? {
        return Err(Error::Unsupported("cross_component_prediction"));
    }
    if r.flag()? {
        return Err(Error::Unsupported("chroma_qp_offset_list"));
    }
    let luma = ue_max(r, 6, "log2_sao_offset_scale_luma")?;
    let chroma = ue_max(r, 6, "log2_sao_offset_scale_chroma")?;
    Ok((luma, chroma))
}

// ---------------------------------------------------------------------------------------------
// Activation

/// The SPS and PPS of the current picture with the scan tables of 6.5.1 and 6.5.2.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "col_bd and row_bd are only inputs of the scan tables"
)]
pub(crate) struct Active {
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    /// CtbAddrRsToTs, indexed by raster address.
    pub rs_to_ts: Vec<u32>,
    /// CtbAddrTsToRs, indexed by tile-scan address.
    pub ts_to_rs: Vec<u32>,
    /// TileId, indexed by tile-scan address.
    pub tile_id: Vec<u32>,
    /// colBd: `num_tile_columns + 1` boundaries in CTBs.
    pub col_bd: Vec<u32>,
    /// rowBd: `num_tile_rows + 1` boundaries in CTBs.
    pub row_bd: Vec<u32>,
    /// MinTbAddrZs, row-major with `min_tb_stride` entries per row, over the grid padded to
    /// whole CTBs.
    pub min_tb_addr_zs: Vec<u32>,
    pub min_tb_stride: usize,
    /// Bytes booked in the ledger for the tables; the caller releases them when it drops this.
    pub charged: usize,
}

impl Active {
    /// The scaling list in force: the PPS list if sent, else the SPS list; None when
    /// scaling_list_enabled_flag is 0.
    pub(crate) fn scaling(&self) -> Option<&ScalingList> {
        self.sps
            .scaling
            .as_ref()
            .map(|s| self.pps.scaling.as_ref().unwrap_or(s))
    }

    /// MinTbAddrZs[x][y] with (x, y) in minimum transform block units.
    pub(crate) fn min_tb_addr_zs(&self, x: usize, y: usize) -> Option<u32> {
        if x >= self.min_tb_stride {
            return None;
        }
        self.min_tb_addr_zs
            .get(y.checked_mul(self.min_tb_stride)?.checked_add(x)?)
            .copied()
    }
}

fn zeros(n: usize) -> Result<Vec<u32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, 0);
    Ok(v)
}

/// 6.5.1 (6-3, 6-4): the `n + 1` tile boundaries along one axis of `total` CTBs.
fn bounds(n: u32, total: u32, uniform: bool, explicit: &[u32], out: &mut [u32]) -> Result<()> {
    let (n64, total64) = (u64::from(n), u64::from(total));
    let mut acc = 0u64;
    for i in 0..n {
        let w = if uniform {
            (u64::from(i) + 1) * total64 / n64 - u64::from(i) * total64 / n64
        } else if i + 1 < n {
            u64::from(explicit.get(i as usize).copied().unwrap_or(0))
        } else {
            total64 - acc
        };
        if w == 0 || acc + w > total64 {
            return bad("tile size");
        }
        set_u32(out, i as usize, acc as u32)?;
        acc += w;
    }
    set_u32(out, n as usize, total)
}

/// Builds the tables for `sps` / `pps`. Everything is booked in `ledger` first (`charged`).
pub(crate) fn activate(
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    limits: &Limits,
    ledger: &Ledger,
) -> Result<Active> {
    if pps.sps_id != sps.sps_id {
        return bad("pps refers to another sps");
    }
    sps.check_limits(limits)?;
    // 7.4.3.3.1: a Main / Main 10 / Main Still Picture decoder ignores the PPS extension.
    let pps = if (1..=3).contains(&sps.profile_idc)
        && (pps.extension_error.is_some()
            || pps.log2_sao_offset_scale_luma != 0
            || pps.log2_sao_offset_scale_chroma != 0)
    {
        Arc::new(Pps {
            extension_error: None,
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            ..(*pps).clone()
        })
    } else if let Some(e) = &pps.extension_error {
        return Err(e.clone());
    } else {
        pps
    };
    // init_qp_minus26 in -(26 + QpBdOffsetY)..=25.
    if !(-sps.qp_bd_offset_y..=51).contains(&pps.init_qp) {
        return bad("init_qp_minus26");
    }
    if pps.diff_cu_qp_delta_depth > sps.ctb_log2_size_y - sps.min_cb_log2_size_y {
        return bad("diff_cu_qp_delta_depth");
    }
    if pps.log2_parallel_merge_level > sps.ctb_log2_size_y {
        return bad("log2_parallel_merge_level");
    }
    let max_scale = |depth: u32| depth.saturating_sub(10);
    if pps.log2_sao_offset_scale_luma > max_scale(sps.bit_depth_luma)
        || pps.log2_sao_offset_scale_chroma > max_scale(sps.bit_depth_chroma)
    {
        return bad("log2_sao_offset_scale");
    }
    let (w, h) = (sps.pic_width_in_ctbs_y, sps.pic_height_in_ctbs_y);
    let (cols, rows) = (pps.num_tile_columns, pps.num_tile_rows);
    if cols > w || rows > h {
        return bad("tile count over the picture size in CTBs");
    }
    let shift = sps.ctb_log2_size_y - sps.log2_min_trafo_size;
    let n_ctbs = usize::try_from(u64::from(w) * u64::from(h)).map_err(|_| Error::AllocLimit)?;
    let stride = usize::try_from(u64::from(w) << shift).map_err(|_| Error::AllocLimit)?;
    let grid_rows = usize::try_from(u64::from(h) << shift).map_err(|_| Error::AllocLimit)?;
    let grid = stride.checked_mul(grid_rows).ok_or(Error::AllocLimit)?;
    let words = n_ctbs
        .checked_mul(3)
        .and_then(|v| v.checked_add(grid))
        .and_then(|v| v.checked_add(cols as usize + rows as usize + 2))
        .ok_or(Error::AllocLimit)?;
    let charged = words.checked_mul(4).ok_or(Error::AllocLimit)?;
    ledger.reserve(charged)?;
    build(sps, pps, n_ctbs, stride, grid, charged).inspect_err(|_| ledger.release(charged))
}

fn build(
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    n_ctbs: usize,
    stride: usize,
    grid: usize,
    charged: usize,
) -> Result<Active> {
    let (w, h) = (sps.pic_width_in_ctbs_y, sps.pic_height_in_ctbs_y);
    let (cols, rows) = (pps.num_tile_columns, pps.num_tile_rows);
    let mut col_bd = zeros(cols as usize + 1)?;
    let mut row_bd = zeros(rows as usize + 1)?;
    bounds(cols, w, pps.uniform_spacing, &pps.column_width, &mut col_bd)?;
    bounds(rows, h, pps.uniform_spacing, &pps.row_height, &mut row_bd)?;
    let mut rs_to_ts = zeros(n_ctbs)?;
    let mut ts_to_rs = zeros(n_ctbs)?;
    let mut tile_id = zeros(n_ctbs)?;
    let mut ts = 0usize;
    let mut tid = 0u32;
    for tj in 0..rows as usize {
        for ti in 0..cols as usize {
            let (Some(&x0), Some(&x1), Some(&y0), Some(&y1)) = (
                col_bd.get(ti),
                col_bd.get(ti + 1),
                row_bd.get(tj),
                row_bd.get(tj + 1),
            ) else {
                return bad("tile bounds");
            };
            for y in y0..y1 {
                for x in x0..x1 {
                    let rs = y * w + x;
                    set_u32(&mut rs_to_ts, rs as usize, ts as u32)?;
                    set_u32(&mut ts_to_rs, ts, rs)?;
                    set_u32(&mut tile_id, ts, tid)?;
                    ts += 1;
                }
            }
            tid += 1;
        }
    }
    // 6.5.2: z-scan order address of every minimum transform block.
    let shift = sps.ctb_log2_size_y - sps.log2_min_trafo_size;
    let mut min_tb_addr_zs = zeros(grid)?;
    let grid_rows = grid.checked_div(stride).unwrap_or(0);
    for y in 0..grid_rows {
        for x in 0..stride {
            let rs = (y >> shift) * w as usize + (x >> shift);
            let Some(&t) = rs_to_ts.get(rs) else {
                return bad("min tb grid");
            };
            let mut v = u64::from(t) << (2 * shift);
            for i in 0..shift {
                let m = 1usize << i;
                if x & m != 0 {
                    v += (m * m) as u64;
                }
                if y & m != 0 {
                    v += (2 * m * m) as u64;
                }
            }
            set_u32(
                &mut min_tb_addr_zs,
                y * stride + x,
                u32::try_from(v).map_err(|_| Error::AllocLimit)?,
            )?;
        }
    }
    Ok(Active {
        sps,
        pps,
        rs_to_ts,
        ts_to_rs,
        tile_id,
        col_bd,
        row_bd,
        min_tb_addr_zs,
        min_tb_stride: stride,
        charged,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// MSB-first bit writer for hand-built parameter sets.
    #[derive(Default)]
    struct W {
        bits: Vec<bool>,
    }

    impl W {
        fn u(&mut self, n: u32, v: u64) {
            for i in (0..n).rev() {
                self.bits.push((v >> i) & 1 == 1);
            }
        }
        fn flag(&mut self, v: bool) {
            self.bits.push(v);
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

    /// A Main profile SPS: CTB 32, min CB 8, TB 4..32, 4:2:0 8 bit, crop of 2 columns and rows.
    pub(crate) fn sps_bits(w: u32, h: u32) -> Vec<u8> {
        sps_bits_with(w, h, 2, |b| b.flag(false)) // no vui
    }

    /// `sps_bits` with CTB 16.
    pub(crate) fn sps_bits_ctb16(w: u32, h: u32) -> Vec<u8> {
        sps_bits_with(w, h, 1, |b| b.flag(false))
    }

    /// `sps_bits` with CTB `8 << ctb_diff`, the vui_parameters_present_flag and the VUI written by `vui`.
    fn sps_bits_with(w: u32, h: u32, ctb_diff: u32, vui: impl FnOnce(&mut W)) -> Vec<u8> {
        let mut b = W::default();
        b.u(4, 0); // vps id
        b.u(3, 0); // max_sub_layers_minus1
        b.flag(true);
        b.u(2, 0);
        b.flag(false);
        b.u(5, 1); // profile idc
        b.u(32, 0x6000_0000);
        b.u(32, 0);
        b.u(16, 0); // 48 bits of flags
        b.u(8, 93); // level
        b.ue(0); // sps id
        b.ue(1); // 4:2:0
        b.ue(w);
        b.ue(h);
        b.flag(true); // conformance window
        for v in [0, 1, 0, 1] {
            b.ue(v);
        }
        b.ue(0);
        b.ue(0);
        b.ue(4); // log2_max_poc_lsb 8
        b.flag(true);
        b.ue(1);
        b.ue(0);
        b.ue(0);
        b.ue(0); // min cb 8
        b.ue(ctb_diff); // ctb 32 (diff 2) or 16 (diff 1)
        b.ue(0); // min tb 4
        b.ue(ctb_diff + 1); // max tb = ctb
        b.ue(1);
        b.ue(1);
        b.flag(false); // scaling lists
        b.flag(true); // amp
        b.flag(true); // sao
        b.flag(false); // pcm
        b.ue(0); // no st rps
        b.flag(false); // long term
        b.flag(false);
        b.flag(true); // strong intra smoothing
        vui(&mut b);
        b.flag(false); // extension
        b.bytes()
    }

    /// A PPS with `tiles` columns and rows (uniform) or none.
    pub(crate) fn pps_bits(tiles: u32) -> Vec<u8> {
        let mut b = W::default();
        b.ue(0);
        b.ue(0);
        b.flag(false);
        b.flag(false);
        b.u(3, 0);
        b.flag(true); // sign data hiding
        b.flag(false);
        b.ue(0);
        b.ue(0);
        b.se(-2); // init_qp 24
        b.flag(false);
        b.flag(false);
        b.flag(false); // cu_qp_delta
        b.se(1);
        b.se(-1);
        b.flag(false);
        b.flag(false);
        b.flag(false);
        b.flag(false);
        b.flag(tiles > 1);
        b.flag(false);
        if tiles > 1 {
            b.ue(tiles - 1);
            b.ue(tiles - 1);
            b.flag(true);
            b.flag(true);
        }
        b.flag(false); // loop filter across slices
        b.flag(false); // deblocking control
        b.flag(false); // scaling list
        b.flag(false);
        b.ue(0);
        b.flag(false);
        b.flag(false); // extension
        b.bytes()
    }

    pub(crate) fn ledger() -> Ledger {
        let l = Ledger::default();
        l.reset(1 << 30, 0).unwrap();
        l
    }

    #[test]
    fn minimal_sps_and_pps() {
        let sps = parse_sps(&sps_bits(64, 48)).unwrap();
        assert_eq!((sps.width, sps.height), (64, 48));
        assert_eq!(
            (sps.crop_left, sps.crop_right, sps.crop_top, sps.crop_bottom),
            (0, 2, 0, 2)
        );
        assert_eq!(
            (
                sps.ctb_size_y,
                sps.pic_width_in_ctbs_y,
                sps.pic_height_in_ctbs_y
            ),
            (32, 2, 2)
        );
        assert_eq!((sps.log2_min_trafo_size, sps.log2_max_trafo_size), (2, 5));
        assert_eq!(sps.log2_max_poc_lsb, 8);
        assert_eq!(sps.profile_idc, 1);
        assert_eq!(sps.level_idc, 93);
        assert!(sps.scaling.is_none() && sps.pcm.is_none() && sps.st_rps.is_empty());
        assert_eq!(sps.ordering.len(), 1);
        let pps = parse_pps(&pps_bits(1)).unwrap();
        assert_eq!(pps.init_qp, 24);
        assert_eq!((pps.cb_qp_offset, pps.cr_qp_offset), (1, -1));
        assert!(pps.sign_data_hiding_enabled && !pps.tiles_enabled);
        assert_eq!((pps.num_tile_columns, pps.num_tile_rows), (1, 1));
        assert_eq!(pps.log2_parallel_merge_level, 2);
    }

    #[test]
    fn vui_colour_description() {
        let bits = sps_bits_with(64, 48, 2, |b| {
            b.flag(true); // vui_parameters_present_flag
            b.flag(true); // aspect_ratio_info_present_flag
            b.u(8, 255);
            b.u(16, 4);
            b.u(16, 3);
            b.flag(false); // overscan_info_present_flag
            b.flag(true); // video_signal_type_present_flag
            b.u(3, 1);
            b.flag(true); // full range
            b.flag(true); // colour_description_present_flag
            b.u(8, 9);
            b.u(8, 16);
            b.u(8, 10);
            b.flag(true); // chroma_loc_info_present_flag
            b.ue(1);
            b.ue(2);
            b.u(3, 0);
            b.flag(true); // default_display_window_flag
            for v in [1, 2, 3, 4] {
                b.ue(v);
            }
            b.flag(false); // timing
            b.flag(false); // bitstream restriction
        });
        let vui = parse_sps(&bits).unwrap().vui.unwrap();
        assert_eq!(
            vui,
            Vui {
                aspect_ratio_idc: 255,
                aspect_ratio: Some((4, 3)),
                video_format: 1,
                video_full_range: true,
                colour_primaries: 9,
                transfer_characteristics: 16,
                matrix_coeffs: 10,
                chroma_sample_loc_type_top: 1,
                chroma_sample_loc_type_bottom: 2,
                default_display_window: Some([1, 2, 3, 4]),
            }
        );
        // Only the flags: the spec defaults.
        let bits = sps_bits_with(64, 48, 2, |b| {
            b.flag(true);
            for _ in 0..4 {
                b.flag(false); // aspect, overscan, video signal, chroma loc
            }
            b.u(3, 0);
            for _ in 0..3 {
                b.flag(false); // display window, timing, restriction
            }
        });
        let vui = parse_sps(&bits).unwrap().vui.unwrap();
        assert_eq!(
            (vui.aspect_ratio_idc, vui.aspect_ratio, vui.video_format),
            (0, None, 5)
        );
        assert_eq!(
            (
                vui.colour_primaries,
                vui.transfer_characteristics,
                vui.matrix_coeffs
            ),
            (2, 2, 2)
        );
        assert_eq!(vui.default_display_window, None);
        assert!(parse_sps(&sps_bits(64, 48)).unwrap().vui.is_none());
    }

    #[test]
    fn sps_rejections() {
        // Width not a multiple of MinCbSizeY.
        assert!(matches!(
            parse_sps(&sps_bits(60, 48)),
            Err(Error::InvalidData(_))
        ));
        assert!(matches!(parse_sps(&[0xff]), Err(Error::InvalidData(_))));
        let limits = Limits {
            max_width: 32,
            ..Limits::default()
        };
        assert_eq!(
            parse_sps(&sps_bits(64, 48)).unwrap().check_limits(&limits),
            Err(Error::WidthLimit)
        );
    }

    #[test]
    fn vps_id() {
        assert_eq!(parse_vps(&[0x70]).unwrap().id, 7);
        assert!(parse_vps(&[]).is_err());
    }

    #[test]
    fn st_rps_inter_prediction() {
        let sets = [StRps {
            num_negative: 2,
            num_positive: 0,
            delta_poc: vec![-2, -4],
            used: vec![true, true],
        }];
        let mut b = W::default();
        b.flag(true); // inter_ref_pic_set_prediction_flag
        b.ue(0); // delta_idx_minus1
        b.flag(true); // delta_rps_sign
        b.ue(0); // abs_delta_rps_minus1: deltaRps = -1
        for _ in 0..3 {
            b.flag(true);
        }
        let data = b.bytes();
        let mut r = BitReader::new(&data);
        let s = parse_st_rps(&mut r, 1, &sets).unwrap();
        assert_eq!((s.num_negative, s.num_positive), (3, 0));
        assert_eq!(s.delta_poc, [-1, -3, -5]);
        assert_eq!(s.used, [true; 3]);
    }

    #[test]
    fn st_rps_direct() {
        let mut b = W::default();
        b.ue(2);
        b.ue(1);
        b.ue(0);
        b.flag(true);
        b.ue(1);
        b.flag(false);
        b.ue(2);
        b.flag(true);
        let data = b.bytes();
        let mut r = BitReader::new(&data);
        let s = parse_st_rps(&mut r, 0, &[]).unwrap();
        assert_eq!(s.delta_poc, [-1, -3, 3]);
        assert_eq!(s.used, [true, false, true]);
        assert_eq!((s.num_negative, s.num_positive), (2, 1));
    }

    #[test]
    fn tile_scan_2x2() {
        // 128x128 luma = 4x4 CTBs, 2x2 uniform tiles.
        let sps = Arc::new(parse_sps(&sps_bits(128, 128)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(2)).unwrap());
        let l = ledger();
        let a = activate(sps, pps, &Limits::default(), &l).unwrap();
        assert_eq!(a.col_bd, [0, 2, 4]);
        assert_eq!(a.row_bd, [0, 2, 4]);
        assert_eq!(
            a.rs_to_ts,
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
        for (rs, &ts) in a.rs_to_ts.iter().enumerate() {
            assert_eq!(a.ts_to_rs[ts as usize] as usize, rs);
        }
        let want: Vec<u32> = (0..16).map(|ts| ts / 4).collect();
        assert_eq!(a.tile_id, want);
        // 6.5.2: shift 3 (CTB 32, min TB 4).
        assert_eq!(a.min_tb_addr_zs(0, 0), Some(0));
        assert_eq!(a.min_tb_addr_zs(1, 0), Some(1));
        assert_eq!(a.min_tb_addr_zs(0, 1), Some(2));
        assert_eq!(a.min_tb_addr_zs(1, 1), Some(3));
        assert_eq!(a.min_tb_addr_zs(8, 0), Some(64)); // CTB rs 1 = ts 1
        assert_eq!(a.min_tb_addr_zs(16, 0), Some(4 << 6)); // CTB rs 2 = ts 4
        assert_eq!(a.min_tb_addr_zs(0, 8), Some(2 << 6)); // CTB rs 4 = ts 2
        assert_eq!(a.min_tb_addr_zs(1000, 0), None);
        assert!(a.charged > 0);
        l.release(a.charged);
    }

    #[test]
    fn explicit_tiles_and_errors() {
        let mut v = [0u32; 3];
        bounds(2, 5, false, &[2], &mut v).unwrap();
        assert_eq!(v, [0, 2, 5]);
        assert!(bounds(2, 5, false, &[5], &mut v).is_err());
        bounds(2, 5, true, &[], &mut v).unwrap();
        assert_eq!(v, [0, 2, 5]);
        // More tile columns than CTB columns.
        let sps = Arc::new(parse_sps(&sps_bits(64, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(3)).unwrap());
        assert!(matches!(
            activate(sps, pps, &Limits::default(), &ledger()),
            Err(Error::InvalidData(_))
        ));
    }

    #[test]
    fn untiled_scan_is_identity() {
        let sps = Arc::new(parse_sps(&sps_bits(96, 64)).unwrap());
        let pps = Arc::new(parse_pps(&pps_bits(1)).unwrap());
        let a = activate(sps, pps, &Limits::default(), &ledger()).unwrap();
        assert_eq!(a.rs_to_ts, [0, 1, 2, 3, 4, 5]);
        assert_eq!(a.tile_id, [0; 6]);
        assert!(a.scaling().is_none());
    }
}
