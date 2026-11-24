# hevc-rs design (v1)

This is the build spec of the crate. It is binding for every module, oracle and workflow.
Conventions: `dobbo-ca/rust-ports` `docs/conventions.md` (all of it), `docs/new-crates.md`,
`docs/round-c.md`, `docs/quality.md`.

hevc-rs is a new crate, not a port: an HEVC (ITU-T H.265) decoder for intra pictures only,
bit-exact. HEIC stills and video keyframes are intra, so there is no inter prediction (no P/B
slices, no motion compensation, no reference picture handling beyond parsing).

## Hard rules

- No local builds: never run cargo build/test/check/clippy/bench/run/miri/fuzz, rustc, cc, gcc,
  python, perl, ffmpeg, dec265 or any other analysis tool on the owner's Mac. Only edit,
  `cargo fmt`, `cargo generate-lockfile`, git, gh. CI on ARC is the only build.
- No C dependency in the crate. libde265 and ffmpeg are CI-only oracle tools.
- Never panic on hostile input. Non-test code denies `clippy::unwrap_used`, `expect_used`,
  `panic`, `todo`, `unimplemented`, `indexing_slicing`. Use `get`, iterators, `chunks_exact`,
  fixed-size arrays indexed by values masked/clamped into range. `checked_*` on every size from
  the input.
- `#![forbid(unsafe_code)]` until the SIMD phase; then the safe-SIMD pattern of conventions.md.
- Bounded allocations: charge the ledger before allocating, `try_reserve_exact`, no `vec![0; n]`
  on an unchecked `n`.
- Source of truth: the H.265 spec (2023-09 or later edition). Tables come from the spec.
  libde265, ffmpeg and openHEVC are LGPL: design reference only, copy no code and no tables.
  Code may be ported only from MIT/BSD/Apache-2.0/ISC/zlib sources, with attribution in the file
  and in NOTICE.

## Scope

Supported: Main, Main 10, Main Still Picture (general_profile_idc 1, 2, 3, and compatible flags);
chroma_format_idc 1 (4:2:0) and 0 (4:0:0); bit depth 8..=12 per component (luma and chroma may
differ); tiles, wavefront (entry_coding_sync), dependent slice segments, multiple slices; PCM;
transform skip (4x4); sign data hiding; scaling lists (SPS and PPS, prediction, defaults);
cu_qp_delta; pps/slice chroma QP offsets; deblocking; SAO; conformance window cropping;
decoded picture hash SEI (MD5, CRC, checksum) checked when present and `Options::check_hash` is true.

Rejected with a distinct error:

- Any P or B slice: `Error::InterSlice`. Pictures before it are still returned by the `_into`
  entry points (see API).
- separate_colour_plane_flag = 1, chroma_format_idc 2 or 3: `Error::Unsupported`.
- Range extension tools (sps_range_extension / pps_range_extension flags):
  transform_skip_rotation, transform_skip_context, implicit_rdpcm, explicit_rdpcm,
  extended_precision_processing, intra_smoothing_disabled, persistent_rice_adaptation,
  cabac_bypass_alignment, log2_max_transform_skip_block_size_minus2 != 0,
  cross_component_prediction, chroma_qp_offset_list_enabled: `Error::Unsupported`.
  Accepted: high_precision_offsets_enabled (inter only), log2_sao_offset_scale_luma/chroma
  (one shift in SAO).
- sps_multilayer/3d/scc extension flags, pps multilayer/3d/scc extension flags:
  `Error::Unsupported`. NAL units with nuh_layer_id > 0 are skipped.

## Public API (exact; lib.rs re-exports)

```rust
pub use error::Error;
pub use limits::Limits;

/// Decode switches. Default: both in-loop filters on (conforming output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Run the deblocking filter (default true). false is for diagnosis only: the output then
    /// does not conform.
    pub deblocking: bool,
    /// Run SAO (default true). Same note.
    pub sao: bool,
    /// Keep the metadata (default true): `Picture::sei` and `Info::vui`. See "Metadata" below.
    pub metadata: bool,
    /// Check the decoded picture hash SEI (default false). false: `Picture::hash_ok` is `None` and
    /// no MD5, CRC or checksum is computed. `examples/oracle.rs` sets it, so the parity gate G4 is
    /// unchanged.
    pub check_hash: bool,
}
impl Default for Options { /* all true except check_hash */ }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat { Mono, Yuv420 }

/// Stream facts from the first SPS (and the first PPS is not needed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    pub width: u32,            // after the conformance window
    pub height: u32,
    pub coded_width: u32,      // pic_width_in_luma_samples
    pub coded_height: u32,
    pub chroma_format: ChromaFormat,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub profile_idc: u8,       // general_profile_idc
    pub level_idc: u8,         // general_level_idc
    pub vui: Option<Vui>,      // None when vui_parameters_present_flag = 0 (or Options::metadata false in pictures)
}

/// VUI colour, aspect ratio and display fields as coded (spec defaults when a flag is absent);
/// see src/lib.rs for the fields. Timing, HRD and restrictions are parsed and dropped.
pub struct Vui { /* aspect_ratio_idc, aspect_ratio, video_format, video_full_range,
                    colour_primaries, transfer_characteristics, matrix_coeffs,
                    chroma_sample_loc_type_top/bottom, default_display_window */ }

/// One SEI message, raw: `{ suffix: bool, payload_type: u32, payload: Vec<u8> }`.
pub struct SeiMessage { /* ... */ }

/// Samples of one plane, tightly packed rows (stride = width of the plane).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Samples { U8(Vec<u8>), U16(Vec<u16>) }   // U8 when the component's bit depth is 8

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane { pub width: u32, pub height: u32, pub samples: Samples }

/// One output picture, cropped to the conformance window, in output order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub info: Info,
    /// 1 plane (Mono) or 3 (Y, Cb, Cr).
    pub planes: Vec<Plane>,
    /// PicOrderCntVal.
    pub poc: i32,
    /// Some(true/false) when `Options::check_hash` is true and a decoded picture hash SEI was
    /// present (the check is over the
    /// uncropped decoded picture); None when absent or `check_hash` is false.
    pub hash_ok: Option<bool>,
    /// Every SEI message of the access unit in bitstream order; empty when metadata is off.
    pub sei: Vec<SeiMessage>,
}

/// Annex B byte stream (start codes). Every intra picture, in output order.
pub fn decode(stream: &[u8], opts: &Options, limits: &Limits, cancel: &AtomicBool)
    -> Result<Vec<Picture>, Error>;
/// Same, but pictures output before an error stay in `out` (the DPB is flushed into `out`
/// before the error returns). The picture that holds the failing slice is dropped.
pub fn decode_into(stream: &[u8], opts: &Options, limits: &Limits, cancel: &AtomicBool,
    out: &mut Vec<Picture>) -> Result<(), Error>;
/// HEIF/MP4 form: `config` is an HEVCDecoderConfigurationRecord (hvcC box payload), `samples`
/// one or more samples of length-prefixed NAL units (lengthSizeMinusOne + 1 bytes each).
pub fn decode_hvcc(config: &[u8], samples: &[u8], opts: &Options, limits: &Limits,
    cancel: &AtomicBool) -> Result<Vec<Picture>, Error>;
pub fn decode_hvcc_into(config: &[u8], samples: &[u8], opts: &Options, limits: &Limits,
    cancel: &AtomicBool, out: &mut Vec<Picture>) -> Result<(), Error>;
/// Info of the first SPS, no decode.
pub fn probe(stream: &[u8], limits: &Limits) -> Result<Info, Error>;
pub fn probe_hvcc(config: &[u8], limits: &Limits) -> Result<Info, Error>;
/// Test tooling for the parity oracle: the Annex B bytes (4-byte start codes) of every layer 0
/// NAL unit before the first picture that holds a P or B slice. That picture (all its VCL NAL
/// units) and everything after it is dropped. `Error::NoPicture` when no VCL NAL unit is left.
pub fn intra_prefix(stream: &[u8], limits: &Limits) -> Result<Vec<u8>, Error>;
```

`Limits` (src/limits.rs):

| Field | Default | Over it |
|---|---|---|
| `max_alloc: usize` | 1 GiB | `Error::AllocLimit` (`Error::OutOfMemory` if the allocator fails) |
| `max_threads: usize` | `available_parallelism()` | no error; 0 is treated as 1 |
| `max_width: u32` | 16384 (pic_width_in_luma_samples) | `Error::WidthLimit` |
| `max_height: u32` | 16384 | `Error::HeightLimit` |
| `max_pixels: u64` | `1 << 28` (coded width x height) | `Error::PixelLimit` |
| `max_pictures: usize` | 4096 (decoded pictures per call) | `Error::PictureLimit` |
| `max_metadata: usize` | 1 MiB (SEI bytes kept per picture, payload plus message struct) | no error; the message is dropped |

`Error` (src/error.rs), `#[non_exhaustive]`, `Clone + PartialEq + Eq + Debug + Display + Error`:

```rust
InvalidData(&'static str),  // malformed bitstream (any syntax or semantic violation)
Unsupported(&'static str),  // valid HEVC outside the scope above
InterSlice,                 // a P or B slice
NoPicture,                  // decode: the stream has no decodable picture (and no other error)
WidthLimit, HeightLimit, PixelLimit, PictureLimit, AllocLimit, OutOfMemory, Cancelled,
```

Parity error tokens (examples/oracle.rs): `invalid`, `unsupported`, `inter`, `nopicture`,
`limit`, `cancelled`.

Threading: with `max_threads` above 1 the pictures of one call are decoded in parallel, and
deblocking and SAO run in bands of whole CTB rows on scoped threads (`frame::par_bands`); the
tiles of a HEIC grid are decoded in parallel too. `max_threads = 1` spawns nothing. The output
and the errors are byte-identical to the serial path for every thread count; see
[threading.md](threading.md). Parallel decode by tile / WPP row is not done.

Cancellation: the flag is read before each NAL unit and before each CTU; set → `Cancelled`.

## Module layout and internal contracts

All modules are `pub(crate)` unless listed in the API. One `Result<T> = Result<T, Error>`.

```
src/lib.rs          crate docs (Resource limits section + doc-test), API fns, re-exports
src/error.rs        Error
src/limits.rs       Limits
src/ledger.rs       charge-on-allocate ledger (copied from png-rs); helpers alloc_u16 etc.
src/bits.rs         BitReader over an RBSP
src/nal.rs          Annex B and length-prefixed splitting, NAL header, emulation prevention, hvcC
src/ps.rs           VPS (minimal), SPS, PPS, profile_tier_level, st_ref_pic_set, VUI/HRD skip
src/scaling.rs      scaling_list_data, default lists, ScalingFactor derivation
src/slice.rs        slice_segment_header (I slices)
src/sei.rs          SEI message walk; decoded picture hash (payloadType 132); hash compute
src/cabac.rs        arithmetic decoding engine (spec model) + context state init (initType 0)
src/ctu.rs          slice_segment_data: CTU, SAO syntax, coding quadtree, CU, PCM, transform tree
src/residual.rs     residual_coding
src/intra.rs        intra sample prediction (8.4.4.2)
src/transform.rs    scaling (dequant) + inverse transform + transform skip + bypass (8.6)
src/deblock.rs      deblocking filter (8.7.2)
src/sao.rs          SAO (8.7.3)
src/frame.rs        per-picture state: sample planes and metadata grids, availability
src/decoder.rs      NAL loop, parameter set store, activation, picture start/finish, POC,
                    tiles/scan tables, slice dispatch, DPB output (C.5.2), cropping
```

### bits.rs

```rust
pub(crate) struct BitReader<'a> { data: &'a [u8], pos: usize /* bit position */ }
impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self;
    pub fn u(&mut self, n: u32) -> Result<u32>;   // n <= 32; past end -> InvalidData("truncated")
    pub fn flag(&mut self) -> Result<bool>;
    pub fn ue(&mut self) -> Result<u32>;          // > 31 leading zeros or value > u32::MAX - 1 -> InvalidData
    pub fn se(&mut self) -> Result<i32>;
    pub fn skip(&mut self, n: usize) -> Result<()>;
    pub fn byte_aligned(&self) -> bool;
    pub fn align(&mut self);                      // to the next byte boundary
    pub fn bit_pos(&self) -> usize;
    pub fn bits_left(&self) -> usize;
    pub fn more_rbsp_data(&self) -> bool;         // spec 7.2
    pub fn data(&self) -> &'a [u8];
}
```

### nal.rs

```rust
pub(crate) struct Nal<'a> { pub kind: u8, pub layer_id: u8, pub temporal_id: u8, pub payload: &'a [u8] }
// payload: bytes after the 2-byte NAL header, still escaped (emulation prevention bytes in).
pub(crate) fn annexb(stream: &[u8]) -> AnnexB<'_>;        // Iterator<Item = Result<Nal>>
pub(crate) fn length_prefixed(data: &[u8], len_size: usize) -> LengthPrefixed<'_>; // same Item
pub(crate) fn header(nal: &[u8]) -> Result<Nal<'_>>;       // forbidden_zero_bit must be 0
/// Removes emulation prevention bytes (00 00 03 -> 00 00) into `out` (cleared first; capacity
/// charged to the ledger by the caller through ledger::reserve_vec).
pub(crate) fn unescape(payload: &[u8], out: &mut Vec<u8>);
pub(crate) struct HvcC<'a> { pub length_size: usize, pub nals: Vec<&'a [u8]> }  // each NAL incl. header
pub(crate) fn parse_hvcc(config: &[u8]) -> Result<HvcC<'_>>;  // nals count bounded (<= 1024)
```

NAL unit types used: 0..=9 VCL non-IRAP (TRAIL_N/R, TSA, STSA, RADL, RASL), 16..=21 IRAP
(BLA_W_LP, BLA_W_RADL, BLA_N_LP, IDR_W_RADL, IDR_N_LP, CRA), 32 VPS, 33 SPS, 34 PPS, 35 AUD,
36 EOS, 37 EOB, 38 FD, 39 prefix SEI, 40 suffix SEI. Reserved VCL types 10..=15 and 22..=31:
skipped (spec: ignore). Others skipped.

### ps.rs

Parameter set ids: VPS 0..=15, SPS 0..=15, PPS 0..=63 (out of range → InvalidData). A new set
with the same id replaces the old one. Stored as parsed structs (`Vps`, `Sps`, `Pps`); the
decoder keeps `[Option<Arc<Sps>>; 16]`, `[Option<Arc<Pps>>; 64]` (or Box/Rc, no unsafe).
Parameter set NAL payload size is bounded (<= 1 MiB) and charged.

`Sps` holds every field the decoder needs, already validated (ranges from 7.4.3.2):
chroma_format_idc (0..=3; 2,3 → Unsupported), pic_width/height_in_luma_samples (multiple of
MinCbSizeY, non-zero, limits checked when the SPS is activated by a picture, and also by probe),
conformance window (offsets in luma samples after SubWidthC/SubHeightC scaling, must leave a
non-empty picture), bit_depth_luma/chroma (8..=12 else Unsupported; above 16 InvalidData),
log2_max_pic_order_cnt_lsb (4..=16), sub-layer ordering info (max_dec_pic_buffering_minus1,
max_num_reorder_pics, max_latency_increase_plus1 per sub-layer; when
sub_layer_ordering_info_present_flag is 0 every sub-layer takes the values of the highest),
log2_min_luma_coding_block_size (3..), CtbLog2SizeY (4..=6), log2_min/max transform block
(2..=5, min < MinCbLog2SizeY, max <= CtbLog2SizeY), max_transform_hierarchy_depth_intra,
scaling (Option<ScalingList>: None when scaling_list_enabled_flag = 0; Default lists when
enabled and sps_scaling_list_data_present_flag = 0), amp_enabled, sample_adaptive_offset_enabled,
pcm (Option<Pcm { bit_depth_luma, bit_depth_chroma, log2_min_size, log2_max_size,
loop_filter_disabled }> with 7.4.3.2 ranges: PCM bit depth <= component bit depth, log2 sizes in
3..=Min(CtbLog2SizeY, 5)), st_ref_pic_sets (all num_short_term_ref_pic_sets <= 64, each parsed
with inter RPS prediction so NumDeltaPocs is known; only NumDeltaPocs per set is kept),
long_term_ref_pics_present + num_long_term_ref_pics_sps (<= 32) + lt_ref_pic_poc_lsb_sps,
temporal_mvp_enabled, strong_intra_smoothing_enabled, VUI (colour, aspect ratio and display
fields kept in `Info::vui`, see "Metadata"; the rest must be parsed completely since extensions
follow it, HRD parameters too, and is dropped), extension flags (range extension
parsed; unsupported tools rejected as in Scope).

Derived on the struct: MinCbLog2SizeY, CtbLog2SizeY, CtbSizeY, PicWidthInCtbsY,
PicHeightInCtbsY, PicWidthInMinCbsY, PicHeightInMinCbsY, SubWidthC, SubHeightC,
QpBdOffsetY = 6 * (BitDepthY - 8), QpBdOffsetC, Log2MinTrafoSize, Log2MaxTrafoSize.

`Pps`: pps_id, sps_id, dependent_slice_segments_enabled, output_flag_present,
num_extra_slice_header_bits, sign_data_hiding_enabled, cabac_init_present, init_qp
(26 + init_qp_minus26; range -(26 + QpBdOffsetY)..=25 checked at activation),
constrained_intra_pred, transform_skip_enabled, cu_qp_delta_enabled, diff_cu_qp_delta_depth,
cb_qp_offset, cr_qp_offset (-12..=12), slice_chroma_qp_offsets_present, weighted flags,
transquant_bypass_enabled, tiles_enabled, entropy_coding_sync_enabled, tile spec
(num_tile_columns/rows, uniform_spacing, explicit column widths / row heights in CTBs),
loop_filter_across_tiles_enabled (default 1), loop_filter_across_slices_enabled,
deblocking control (override_enabled, pps_deblocking_filter_disabled, beta_offset_div2,
tc_offset_div2 in -6..=6), scaling (Option<ScalingList>), lists_modification_present,
log2_parallel_merge_level, slice_segment_header_extension_present, range extension (parsed,
rejected per Scope; log2_sao_offset_scale_luma/chroma kept).

Activation (decoder.rs, at the first slice segment of a picture) builds `Active` =
`{ sps: Arc<Sps>, pps: Arc<Pps>, scan tables }` with 6.5.1 (CtbAddrRsToTs, CtbAddrTsToRs,
TileId, column/row boundaries colBd/rowBd in CTBs, uniform spacing formula), 6.5.2 (MinTbAddrZs
over the min transform block grid), and checks the tile counts against the picture size
(num_tile_columns <= PicWidthInCtbsY, explicit widths sum < PicWidthInCtbsY, else InvalidData).

### scaling.rs

```rust
pub(crate) struct ScalingList { /* ScalingFactor for sizeId 0..=3, matrixId 0..=5, built at parse time */ }
impl ScalingList {
    pub fn default_lists() -> Self;                              // Table 7-5, 7-6
    pub fn parse(r: &mut BitReader) -> Result<Self>;             // 7.3.4 scaling_list_data
    /// m[x][y] of 8.6.4.2 for one TB: size 4..32, matrixId per 7.4.5 (cIdx and intra: matrixId = cIdx for intra).
    pub fn factor(&self, log2_size: u32, matrix_id: usize, x: usize, y: usize) -> u8;
}
```

scaling_list_pred_matrix_id_delta refers back (refMatrixId = matrixId - delta * (sizeId == 3 ? 3 : 1));
delta 0 means the default list. sizeId 3 has matrixId 0 and 3 only (in v1 syntax the loop steps by 3).
DC values for sizeId 2, 3. Up-right diagonal scan of 8x8 for coefficient order (6.5.3).

### slice.rs

```rust
pub(crate) struct SliceHeader {
    pub first_slice_segment_in_pic: bool,
    pub no_output_of_prior_pics: bool,
    pub pps_id: u32,
    pub dependent: bool,
    pub segment_address: u32,      // slice_segment_address (CtbAddrInRs)
    pub pic_output: bool,          // default true
    pub poc_lsb: u32,              // 0 for IDR
    pub sao_luma: bool, pub sao_chroma: bool,
    pub qp_delta: i32,
    pub cb_qp_offset: i32, pub cr_qp_offset: i32,
    pub deblocking_disabled: bool,
    pub beta_offset_div2: i32, pub tc_offset_div2: i32,
    pub loop_filter_across_slices: bool,
    pub data_offset: usize,        // byte offset of slice_segment_data in the RBSP
}
/// Parses one slice segment header. `prev` is the header of the previous independent slice
/// segment of the same picture (dependent segments copy its fields). slice_type != 2 →
/// Err(InterSlice), detected before anything after slice_type is parsed.
pub(crate) fn parse(rbsp: &[u8], nal_kind: u8, sps_of: impl Fn(u32)->..., ... ) -> Result<SliceHeader>;
```

A non-IDR I slice still carries short_term_ref_pic_set_sps_flag / st_ref_pic_set (with inter RPS
prediction against the SPS sets) / long-term fields / slice_temporal_mvp_enabled_flag: all are
parsed and dropped. `collocated`, num_ref_idx and weighted prediction are never present in an I
slice. slice_segment_header_extension bytes are skipped. After the header,
byte_alignment() (one 1 bit then 0 bits) must hold, else InvalidData.

### sei.rs

Walks sei_message()s of prefix and suffix SEI NALs: payloadType/payloadSize with 0xFF
continuation (each bounded to the NAL size). payloadType 132 (decoded_picture_hash, suffix)
is read for the check: hash_type 0 MD5 (16 bytes per component), 1 CRC (u16), 2 checksum (u32),
for 1 or 3 components. Every payload (132 too) is also kept raw in `Picture::sei` unless
`Options::metadata` is false, within `Limits::max_metadata`; a malformed SEI is ignored (never
an error).
`fn compute(kind, planes of the decoded (uncropped) picture, bit depths) -> hash values`
per spec D.3.19 (MD5 of the samples as bytes, 1 byte per sample when bit depth <= 8 else 2
little-endian bytes; CRC-CCITT as the spec pseudo-code; checksum with the xorMask formula).
MD5 is written from RFC 1321 (public domain algorithm), no dependency.

### cabac.rs

Spec model (9.3.4.3), exact bit position: `ivlCurrRange` 9 bits, `ivlOffset` read 9 bits at
init, 1 bit per renormalisation, through a BitReader over the slice RBSP. This keeps the bit
position exact for PCM samples and for the end_of_subset_one_bit re-initialisation: after a
terminate bin equal to 1 the reader has consumed every bit the encoder flushed, including the
final 1 bit (rbsp_stop_one_bit / alignment_bit_equal_to_one); pcm_alignment_zero_bits /
remaining byte_alignment zero bits follow.

```rust
pub(crate) struct Cabac<'a> { r: BitReader<'a>, range: u32, offset: u32 }
impl<'a> Cabac<'a> {
    pub fn new(r: BitReader<'a>) -> Result<Self>;   // 9.3.2.5 init: range 510, offset = read 9 bits
    pub fn decode(&mut self, ctx: &mut Ctx) -> Result<bool>;      // 9.3.4.3.2 (rangeTabLps, transIdx)
    pub fn bypass(&mut self) -> Result<bool>;                     // 9.3.4.3.4
    pub fn bypass_bits(&mut self, n: u32) -> Result<u32>;         // MSB first
    pub fn terminate(&mut self) -> Result<bool>;                  // 9.3.4.3.5
    pub fn reader(&mut self) -> &mut BitReader<'a>;               // for PCM / alignment after terminate
    pub fn reinit(&mut self) -> Result<()>;                       // read 9 bits again (after PCM, new substream)
}
#[derive(Clone, Copy)] pub(crate) struct Ctx { state: u8 /* pStateIdx */, mps: u8 }
/// All context variables of a slice, initialised for initType 0 (I slices) from SliceQpY
/// (9.3.2.2, Tables 9-5 .. 9-37, initType 0 columns only).
#[derive(Clone)] pub(crate) struct Contexts { pub sao_merge: Ctx, pub sao_type_idx: Ctx,
    pub split_cu_flag: [Ctx; 3], pub cu_transquant_bypass_flag: Ctx, pub part_mode: Ctx /* bin 0 only in I */,
    pub prev_intra_luma_pred_flag: Ctx, pub intra_chroma_pred_mode: Ctx,
    pub split_transform_flag: [Ctx; 3], pub cbf_luma: [Ctx; 2], pub cbf_cb_cr: [Ctx; 4 /*5 with RExt*/],
    pub cu_qp_delta_abs: [Ctx; 2], pub transform_skip_flag: [Ctx; 2],
    pub last_x_prefix: [Ctx; 18], pub last_y_prefix: [Ctx; 18],
    pub coded_sub_block_flag: [Ctx; 4], pub sig_coeff_flag: [Ctx; 44 /*42 + 2 TS ctx (RExt) unused*/],
    pub greater1: [Ctx; 24], pub greater2: [Ctx; 6], /* end_of_slice etc use terminate */ }
impl Contexts { pub fn init(slice_qp: i32) -> Self; }
```

Wavefront: after the 2nd CTB of a CTB row in a tile (CtbAddrInRs % PicWidthInCtbsY == ... per
9.3.1 / 9.3.2.4 storage rule) store Contexts; at the start of a row restore it when the
top-right CTB (x0 + CtbSizeY, y0 - CtbSizeY) is available, else re-init. Dependent slice
segments start from the Contexts stored at the end of the previous slice segment
(TableStateIdxDs). StatCoeff (persistent Rice) does not exist (rejected tool).

### ctu.rs and residual.rs

Implements 7.3.8 slice_segment_data for I slices with the ctxInc rules of 9.3.4.2 (Table 9-41,
9.3.4.2.2 split_cu_flag via ctDepth of left/above with availability, 9.3.4.2.3 .. 9.3.4.2.7
last_sig_coeff prefix ctx, coded_sub_block_flag, sig_coeff_flag ctxIdxMap and patterns,
greater1 ctxSet / greater1Ctx, greater2) and the binarizations of 9.3.3 (TR, EGk,
coeff_abs_level_remaining with cRiceParam update and the prefix > 3 escape, cu_qp_delta_abs
prefix TR cMax 5 + EG0 suffix, intra_chroma_pred_mode, part_mode (I: one bin: 2Nx2N vs NxN only
at the minimum CB size and log2CbSize > MinTbLog2SizeY), last_sig_coeff suffix FL).

Per CTU: end_of_slice_segment_flag (terminate); at a tile start or WPP row start inside the
slice segment: end_of_subset_one_bit (terminate, must be 1), byte_alignment, CABAC re-init
(contexts per WPP/tile rule). The entry point offsets are skipped in the slice header (the count
is bounded by the PPS and by the RBSP bits left, and nothing is stored): the serial decoder
follows the bit position.
When the slice data ends (end_of_slice_segment_flag = 1) the remaining bits must be
rbsp_slice_segment_trailing_bits (checked; else InvalidData).

SAO syntax (7.3.8.3) with merge left/up (only across CTBs in the same slice and tile), bypass
TR offsets cMax = (1 << (Min(bitDepth, 10) - 5)) - 1, band position FL 5 bits, EO class FL 2
bits; SaoOffsetVal = offset_sign * sao_offset_abs << log2OffsetScale.

Coding quadtree / coding unit / transform tree / transform unit exactly as 7.3.8.4 .. 7.3.8.10
for intra (no cu_skip_flag, no prediction_unit for inter). IsCuQpDeltaCoded / CuQpDeltaVal per
quantization group, QpY derivation 8.6.1 (qPY_PREV: SliceQpY at the first QG of a slice, a
tile, or a CTB row with WPP; else QpY of the previous QG in decoding order; qPY_A/qPY_B from the
left/above QG only when in the same CTB), Qp′Y, Qp′Cb/Qp′Cr through Table 8-10 (ChromaArrayType
1) with pps + slice offsets. Intra mode derivation 8.4.2 (MPM candidates from left/above PU;
above only within the same CTB), 8.4.3 chroma mode (Table 8-2; mode 4 = luma mode; 34 swap).
PCM sample reading (pcm_sample_luma / chroma at PcmBitDepth, stored shifted by
BitDepth - PcmBitDepth). cu_transquant_bypass. Intra 2Nx2N / NxN (NxN only at
log2CbSize == MinCbLog2SizeY; 4 luma PBs; chroma of 4:2:0 coded once with the mode of the first
PB). interSplitFlag is 0 (intra). Transform tree split inference (log2TrafoSize >
MaxTbLog2SizeY, IntraSplitFlag at depth 0). cbf_cb/cbf_cr at log2TrafoSize > 2 (4:2:0), chroma
4x4 blocks at blkIdx 3 of a split 8x8. Residual coding 7.3.8.11 with transform_skip_flag
(log2TrafoSize == 2 and enabled), sign data hiding (cu_transquant_bypass_flag = 0,
lastSigScanPos - firstSigScanPos > 3), scan orders 6.5.3 .. 6.5.5 (diagonal, horizontal,
vertical; scanIdx 8.4.4.? : intra, log2TrafoSize 2 or 3 (luma) / 2 (chroma 4:2:0) with mode
6..=14 → vertical scan (2), 22..=30 → horizontal (1), else diagonal).

Reconstruction happens in decoding order inside ctu.rs: for each TU, predict (intra.rs) then
add the residual (transform.rs) and clip, so later blocks see reconstructed neighbours.

### frame.rs

```rust
pub(crate) struct Frame {
    pub planes: Vec<Vec<u16>>,      // 1 or 3 planes, full coded size, stride = plane width
    pub width: [usize; 3], pub height: [usize; 3],
    // Metadata on the 4x4 luma grid (MinTbSize is >= 4):
    pub qp_y: Vec<i8>,              // QpY per 4x4
    pub intra_mode: Vec<u8>,        // IntraPredModeY per 4x4
    pub no_filter: Vec<u8>,         // bit0: pcm with pcm_loop_filter_disabled, bit1: cu_transquant_bypass
    pub edge_v: Vec<u8>, pub edge_h: Vec<u8>, // deblocking bS (0 or 2) at the left / top edge of each 4x4
    pub ct_depth: Vec<u8>,          // CtDepth per 4x4 (for split_cu_flag ctxInc)
    // Per CTB:
    pub slice_idx: Vec<u16>,        // index into `slices`
    pub sao: Vec<SaoParams>,
    pub slices: Vec<SliceParams>,   // deblocking/SAO/QP params of each slice segment, addr of its slice
}
pub(crate) fn available(&self, act: &Active, x_cur: i32, y_cur: i32, x_nb: i32, y_nb: i32) -> bool; // 6.4.1
```

All grids are allocated once per picture through the ledger.

### intra.rs

8.4.4.2: reference sample availability (6.4.1 per neighbouring sample, 4x4 granularity is
enough because availability is constant per min TB), substitution (8.4.4.2.2), filtering
(8.4.4.2.3: filterFlag by intraHorVerDistThres[nTbS] = 7 (8x8), 1 (16x16), 0 (32x32), none for
4x4 and DC; chroma is never filtered for ChromaArrayType != 3; strong intra smoothing for luma
32x32 with the threshold 1 << (BitDepthY - 5)), planar (8.4.4.2.5), DC with the edge filter for
luma nTbS < 32 (8.4.4.2.6), angular with intraPredAngle / invAngle (Table 8-5, 8-6) and the
boundary filter for modes 10 and 26 when cIdx = 0 and nTbS < 32 (disableIntraBoundaryFilter is
0 without RExt).

```rust
pub(crate) fn predict(frame: &mut Frame, act: &Active, c_idx: usize, x0: usize, y0: usize,
    log2_size: u32, mode: u32);   // x0/y0 in samples of component c_idx; writes the prediction
```

### transform.rs

8.6.2 .. 8.6.4: scaling with m[x][y] (16 when scaling lists are off or transform_skip with
nTbS > 4), levelScale = {40, 45, 51, 57, 64, 72}, bdShift = BitDepth + Log2(nTbS) + 10 - 15
(v1, extended precision off), clip to -32768..=32767; transform skip: r = d << 7 (tsShift = 5 +
Log2(nTbS) = 7 for 4x4); inverse transform: DST-VII 4x4 for luma intra 4x4, else DCT with the
spec 32x32 matrix transMatrix (8.6.4.2, coefficients written from the spec equations),
first stage (vertical) then clip to 16 bits after shift 7, second stage shift 20 - BitDepth;
cu_transquant_bypass: residual = TransCoeffLevel. Residual added to the prediction and clipped
to 0..=(1 << BitDepth) - 1.

```rust
pub(crate) fn reconstruct(frame: &mut Frame, c_idx: usize, x0: usize, y0: usize, log2_size: u32,
    coeffs: &mut [i32] /* nTbS*nTbS, raster */, p: &TuParams /* qp, bit_depth, transform_skip,
    bypass, dst, scaling list + matrix_id */);
```

### deblock.rs and sao.rs

Run once per picture after all slices are decoded (the picture is complete before the filters
read across slice and tile edges). Deblocking 8.7.2: all vertical edges of the picture first
(8x8 luma grid, 8x8 chroma-sample grid i.e. 16 luma for 4:2:0, bS from `edge_v`, chroma only
where bS == 2), then all horizontal edges on the result. bS is 2 on every transform-block and
coding-block edge on the 8x8 grid (all blocks are intra); filterEdgeFlag 0 at picture edges,
at tile edges when loop_filter_across_tiles_enabled_flag = 0, at slice edges when
slice_loop_filter_across_slices_enabled_flag of the slice that contains q0 is 0 (and per the
spec text for the upper/left slice boundary), and for every edge of a coding unit in a slice
with slice_deblocking_filter_disabled_flag = 1. These are set into edge_v/edge_h while decoding
(ctu.rs). β and tC from Table 8-12 with the slice offsets of the slice that contains q0,0; QpC
for chroma from Table 8-10 with cQpPicOffset = pps_cb_qp_offset / pps_cr_qp_offset (not the
slice offsets). Samples with no_filter set are not modified (nDp/nDq = 0).

SAO 8.7.3 on the deblocked picture into a new buffer (copy, then per CTB per component),
band offset and edge offset (hPos/vPos per class), SaoOffsetVal clipping, samples with
no_filter set unchanged, picture-boundary and slice/tile neighbour rules (MinTbAddrZs
comparison and the slice_loop_filter_across_slices_enabled_flag of the right slice,
loop_filter_across_tiles_enabled_flag). bitDepth-based band shift = bitDepth - 5.

Both filters run in bands of whole CTB rows (`workers` bands, `decoder::filter_workers`; 1 = the
calling thread, no spawn; a job on a picture worker uses 1 band, a job on the calling thread uses `filter_workers(max(1, max_threads - spawned))`, see threading.md). Vertical edges never leave their row. A horizontal edge reaches 4 rows
above and 3 below it, so the band that owns the edges of CTB rows `[B, B')` is the plane slice
`[B - 4, B' - 4)`; a pass ends when its bands have joined. SAO bands write their own rows and read
the shared deblocked copy. Details and the deviation from the pool convention: threading.md.

### decoder.rs

- NAL loop over the stream (Annex B or length-prefixed), skip layer_id > 0, store parameter
  sets, collect SEI hashes (suffix SEI after the picture's VCL NALs), decode slice segments.
- First slice segment of a picture (first_slice_segment_in_pic_flag = 1): finish the previous
  picture (queue its decode job), activate SPS/PPS, check limits
  (max_width/height/pixels against the coded size), allocate the Frame (ledger), compute POC
  (8.3.1, prevTid0Pic rules; NoRaslOutputFlag = 1 for IDR/BLA/first picture/after EOS), skip
  RASL pictures when the associated IRAP has NoRaslOutputFlag = 1, apply C.5.2.2 (IRAP with
  NoRaslOutputFlag = 1 and not the first picture: no_output_of_prior_pics handling (inferred 1
  for CRA), else empty the DPB by bumping) and the C.5.2.2 "additional bumping" before
  decoding the current picture: while the number of pictures waiting for output >
  sps_max_num_reorder_pics[HighestTid], or latency (sps_max_latency_increase_plus1) is
  exceeded, or DPB fullness >= sps_max_dec_pic_buffering_minus1 + 1, bump (output the smallest
  POC). HighestTid = sps_max_sub_layers_minus1. After the picture is decoded (C.5.2.3), apply
  the latency count and the reorder/latency bumping again. At the end of the stream bump
  everything. Pictures with pic_output_flag = 0 (or RASL skipped) are not output. All pictures
  are intra, so no picture is kept for reference. KNOWN DEVIATION: the C.5.2.2 fullness
  condition counts only pictures waiting for output; pictures kept only as references by a
  later picture's RPS (8.3.2) are not counted, so the fullness bump can fire later than in the
  spec (observable only when a later IRAP discards the DPB). EOS (type 36) does not flush: the
  next IRAP applies C.5.2.2 (CRA after EOS discards); EOB (37) and end of stream flush.
- A picture is complete when the next picture starts, at EOS/EOB, or at the end of the stream.
  A picture whose CTBs are not all covered by slices → InvalidData.
- Front end and jobs: the NAL loop (parameter sets, slice headers, POC, DPB / C.5.2) is serial.
  A slice segment NAL is only parsed and queued (`Pending`, its rbsp stays charged); the slice
  data, the coverage check, the filters, the hash check (`Options::check_hash`) and the output
  conversion run in `decode_picture`, one job per picture, when the picture is complete. With
  `max_threads = 1` the job runs inline at that point. See `docs/threading.md` for the workers,
  the output order (tickets) and the error rules.
- Slices of the picture must use the same PPS id? (no: PPS may differ per slice only in
  allowed fields; v1: require the same pps_id within a picture, else InvalidData).
- Output: crop to the conformance window and convert to `Samples::U8` (bit depth 8) or
  `Samples::U16`, charged.

## Oracles and parity (CI only)

- `examples/oracle.rs`: `oracle [--threads N] [--no-filters] <in-dir> <out-dir>` decodes every
  `*.bit` / `*.bin` / `*.hevc` / `*.265` file (Annex B) and writes `<name>.yuv` (all output
  pictures, planes Y then Cb then Cr, cropped, 1 byte per sample at bit depth 8 else 2 bytes
  little-endian) plus `<name>.info` (one line per picture: `poc W H chroma depthY depthC hash`
  where hash is ok/bad/none), or `<name>.err` (token). `InterSlice` after at least one picture
  is not an error: the pictures before it are the output. `oracle probe <in> <out>` writes
  `<name>.probe` (Info fields). `oracle prefix <in> <out>` writes, per stream, the Annex B
  prefix that holds every NAL unit before the first picture with a P or B slice (parameter
  sets included, nuh_layer_id > 0 dropped): the input of the C oracles.
- libde265 v1.0.15 `dec265` (built from the tag in CI) and Debian bookworm ffmpeg
  (`-f rawvideo`, native pix_fmt, `-fps_mode passthrough`) decode the same prefix files.
- Gates (`tests/parity/run.sh`, see `docs/parity.md`): G1 `probe` vs `ffprobe`; G2 every stream
  hevc-rs decodes is confirmed (equal to dec265, or equal to ffmpeg, or every picture hash
  ok); G4 no decoded picture hash is bad; G5 `--threads 1` == default; G6 report only: with
  `--no-filters` against dec265 with deblocking and SAO disabled. G2 and G4 are soft: they gate
  when `PARITY_DECODE=1` (the default).
- Corpus: JCT-VC HEVC_v1 conformance streams (all 147, since every one starts with an intra
  picture); RExt streams for rejection classes and public HEIC samples are planned; fuzz-grown
  files. Fetched in CI into a BuildKit cache mount (`id=hevc-rs-data`), never committed.

## Metadata (owner rule, 2026-11-19)

Keep metadata by default and expose it raw: SEI messages (for example mastering display colour
volume, content light level) and the colour information of the VUI. An option may drop them.
Implemented:

- `Options::metadata` (default true). false: `Picture::sei` is empty and `Info::vui` is `None`
  in the pictures. `probe` takes no options and always returns the VUI.
- `Info::vui: Option<Vui>`: `None` when `vui_parameters_present_flag` is 0. `Vui` holds
  aspect_ratio_idc (and `(sar_width, sar_height)` when it is 255), video_format,
  video_full_range, colour_primaries, transfer_characteristics, matrix_coeffs, the two chroma
  sample location types and the default display window offsets as coded; an absent flag gives
  the E.2.1 default (video_format 5, colour description 2, the rest 0 / false / `None`). The
  timing, HRD and restriction parts are parsed and dropped.
- `Picture::sei: Vec<SeiMessage>`: every sei_message() of the access unit in bitstream order,
  all payload types: prefix SEI NAL units (type 39) seen since the previous picture's last VCL
  NAL unit (those before the first VCL NAL unit of the picture, and those between its slice
  segments), and suffix SEI NAL units (type 40) up to the next picture. `SeiMessage { suffix, payload_type, payload }`, payload without emulation
  prevention bytes. Prefix SEI before a skipped RASL picture is dropped. The decoded picture
  hash (132) is also kept in the list; it is checked as before.
- A malformed SEI keeps the messages parsed before the error and never fails the decode
  (`sei::walk`).
- `Limits::max_metadata` (default 1 MiB): payload bytes plus `size_of::<SeiMessage>()` per
  message kept per picture; a message that would pass it is dropped, no error. The bytes are
  charged to the ledger when collected and stay booked with the output picture, like its planes
  (the ledger refusing is `Error::AllocLimit`).

## HEIC glue

`src/heic.rs`, on top of isobmff-rs (private git dependency, `docs/dependencies.md`). isobmff
gives the item graph; this crate decodes the pixels. A file with the low-overhead `mini` box
(ISO/IEC 23008-12 Annex O) instead of `meta` gives the same item graph (isobmff-rs
docs/design.md, "The `mini` box"), so it takes the same path: item 1 is the primary `hvc1`
image, item 2 its alpha (`auxl`), the orientation is `irot`/`imir` (returned raw like any other
transform), and the Exif and XMP items stay in the graph for `Heif::metadata` (the Exif of a
`mini` box is returned as stored: it has no `exif_tiff_header_offset`). A file with neither
`meta` nor `mini` is `InvalidData("no-meta")`; a `mini` of another codec (the VVC `.hmg` files)
is `Unsupported`.

```rust
pub fn decode_heic(file: &[u8], opts: &Options, limits: &Limits, cancel: &AtomicBool)
    -> Result<HeicImage, Error>;
pub fn decode_heic_item(heif: &isobmff::Heif, file: &[u8], id: u32, opts: &Options, limits: &Limits,
    cancel: &AtomicBool) -> Result<Picture, Error>;   // any image item of a parsed file; no alpha, no transforms applied
pub struct HeicImage { pub image: Picture, pub alpha: Option<Plane>, pub transforms: Vec<Transform> }
pub enum Transform { Clap(Clap), Rotate(u16), Mirror(Axis) }   // own types, same fields as isobmff's
```

- `image` is the primary item: one coded `hvc1` item, a `grid` of `hvc1` tiles, or an `iden` or
  `iovl` of those (below). A coded item
  is `Heif::item_bytes` plus its `hvcC` record, passed as one sample to `decode_hvcc_into`; it
  must give exactly one picture, else `InvalidData`. Grid tiles go row-major, tile `(col, row)`
  at `(col * tile_w, row * tile_h)` (chroma at half), cropped to the grid `output_width x
  output_height`. Tiles must have the same size and format (chroma, bit depths) and cover the
  output, else `InvalidData`; 4:2:0 tiles must have even sides unless that axis has one tile.
  The tile pitch is the tiles' `ispe` size (as libheif places them): a tile that decodes to
  another size is `InvalidData`. A grid column or row wholly outside the output is skipped.
- A protected item (`protection_index` not 0, for example CENC) is `Unsupported("protected
  item")`: its slice data is encrypted, so it is not decoded as if clear.
  `place` and `blit` are pure functions over planes (unit tested without fixtures).
- Not applied: `irot`, `imir`, `clap` come back raw and in file order in `transforms`. The caller
  applies them.
- Metadata is kept raw (the owner rule above): `image.sei`, `image.info.vui` and `poc` are the
  first tile's; `hash_ok` is `Some(true)` only if every tile's hash was checked and passed (so `None` unless `Options::check_hash` is true).
  `Options::metadata` false drops them.
- `alpha` is the luma plane of the auxiliary alpha item (`Image::alpha` of isobmff:
  `urn:mpeg:hevc:2015:auxid:1`, `urn:mpeg:mpegB:cicp:systems:auxiliary:alpha` or
  `urn:mpeg:avc:2015:auxid:1`), itself coded
  or a grid (a grid of alpha decodes luma only).
- `iden` (identity) primary or alpha: the input item (coded or grid) is decoded and returned as
  is; `transforms` is the input's own `irot`/`imir`/`clap` in file order, then the identity
  item's. (The alpha plane has no transform list, so an alpha `iden` loses its transforms, as
  every alpha image does.)
- `iovl` (overlay) primary or alpha: composed in the native YCbCr layout of the inputs on a canvas
  of `output_width x output_height` (`max_width`, `max_height`, `max_pixels` and the ledger apply
  before the canvas is allocated). The canvas is filled with the fill colour, then the inputs are
  painted in `dimg` order at `(horizontal_offset, vertical_offset)`, later inputs on top,
  clipped to the canvas, chroma at half the offset; there is no blending (an HEVC input has no
  alpha). Inputs are coded or grid items and must share chroma format and bit depths, and have no
  transformative properties of their own (the pixels are not transformed here); in 4:2:0 every
  offset and every input width and height must be even. Else `Unsupported` with the reason; an overlay with no input is
  `Unsupported` too (no format to compose in). `sei`, `info.vui`, `poc` are the first input's,
  `hash_ok` is the merge over inputs as for grid tiles. An input wholly outside the canvas is
  still decoded (and so still checked for the format).
- Overlay fill colour (R, G, B, A, 16 bits each, full range): the colour space is the `nclx`
  `colr` of the `iovl` item itself, else of the first input item (`matrix_coefficients`,
  `full_range_flag`), else the VUI of the
  first input's SPS (`matrix_coeffs`, `video_full_range_flag`; absent with `metadata` off),
  else BT.601 limited. With `E'R = R/65535` (same for G, B), `Kr`, `Kb` from the matrix
  (1: 0.2126, 0.0722; 4: 0.30, 0.11; 7: 0.212, 0.087; 9: 0.2627, 0.0593; anything else
  including 2, 5, 6: 0.299, 0.114) and `Kg = 1 - Kr - Kb`:
  `E'Y = Kr E'R + Kg E'G + Kb E'B`, `E'Pb = (E'B - E'Y) / (2 (1 - Kb))`,
  `E'Pr = (E'R - E'Y) / (2 (1 - Kr))`, quantised as H.273 (E-3 to E-5) with bit depth `b`:
  limited `Y = round(2^(b-8) (219 E'Y + 16))`, `C = round(2^(b-8) (224 E'P + 128))`;
  full `Y = round((2^b - 1) E'Y)`, `C = round((2^b - 1) E'P + 2^(b-1))`; all clipped to
  `0..2^b - 1`, `b` the luma depth for Y and the chroma depth for Cb, Cr. Matrix 0 (identity,
  GBR): the planes are G, B, R, quantised like luma. The fill R, G, B are sRGB in ISO/IEC
  23008-12; only the matrix and range are applied here, not the primaries or transfer, so the
  fill is approximate for BT.2020/PQ content (a known approximation). A monochrome canvas takes only Y. The
  alpha `A` is used when the overlay is an alpha image (luma-only canvas): `round(A / 65535 *
  (2^b - 1))`. libheif (from memory of its design, not read for this change) composes an `iovl`
  in interleaved RGB(A) and alpha-blends its inputs, so its output is RGB, not the inputs' YCbCr;
  we do not match that (native layout, no blending), and the HEIC parity oracle reports `convert`
  for such files.
- Grid tiles are decoded in parallel (`Limits::max_threads`, see `docs/threading.md`): tile 0
  first and alone (it gives the format and sizes the canvas), the others on up to `max_threads`
  threads (the caller included) that each place their tile on the canvas. The output and the
  error are those of the serial loop: the first error in tile order wins.
- A primary (or alpha) image that is not `hvc1`, a grid of `hvc1`, or an `iden` or `iovl` of those
  (AV1, other) is `Unsupported("primary image is not HEVC")`.
- Errors: isobmff `Cancelled`, `OutOfMemory`, `LimitExceeded(Alloc)` and `LimitExceeded(Pixels)`
  map to `Cancelled`, `OutOfMemory`, `AllocLimit` and `PixelLimit`, `Unsupported(what)` to
  `Unsupported(what)`; every other isobmff error (`NoMeta` included) is
  `InvalidData(err.class())`.
- Limits: see `docs/limits.md`. The cancel flag is read before each coded item (between tiles).
- libheif is the CI oracle only (`examples/heic.rs` is the Rust side of the comparison); it is LGPL
  and never linked.
