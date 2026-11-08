//! Sequence and picture parameter sets (ISO/IEC 14496-10 §7.3.2.1-2;
//! FFmpeg `h264_ps.c`).

use super::consts::{NORM_ADJUST_4X4, NORM_ADJUST_8X8, ZIGZAG_4X4, ZIGZAG_8X8, max_dpb_mbs};
use super::nal::more_rbsp_data;
use super::tables::{DEFAULT_SCALING_4X4, DEFAULT_SCALING_8X8};
use crate::codecs::bitstream::BitReader;

pub const MAX_SPS: usize = 32;
pub const MAX_PPS: usize = 256;
/// Largest picture accepted (8K UHD), which bounds every allocation.
const MAX_MBS: u32 = (8192 / 16) * (4352 / 16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

/// Scaling matrices in raster order: six 4x4 lists (Y/Cb/Cr intra, Y/Cb/Cr
/// inter) and two 8x8 lists (Y intra, Y inter).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingLists {
	pub l4: [[u8; 16]; 6],
	pub l8: [[u8; 64]; 2],
}

impl ScalingLists {
	pub fn flat() -> Self {
		Self { l4: [[16; 16]; 6], l8: [[16; 64]; 2] }
	}
}

#[derive(Debug, Clone)]
pub struct Sps {
	pub profile_idc: u8,
	pub level_idc: u8,
	pub id: usize,
	pub chroma_format_idc: u32,
	pub scaling: Option<ScalingLists>,
	pub log2_max_frame_num: u32,
	pub poc_type: u32,
	pub log2_max_poc_lsb: u32,
	pub delta_pic_order_always_zero: bool,
	pub offset_for_non_ref_pic: i32,
	pub offset_for_top_to_bottom_field: i32,
	pub offset_for_ref_frame: Vec<i32>,
	pub max_num_ref_frames: u32,
	pub gaps_in_frame_num_allowed: bool,
	pub mb_width: u32,
	pub mb_height: u32,
	pub direct_8x8_inference: bool,
	/// Crop in luma samples: left, right, top, bottom.
	pub crop: [u32; 4],
	pub max_num_reorder_frames: Option<u32>,
	pub max_dec_frame_buffering: Option<u32>,
}

impl Sps {
	pub fn width(&self) -> u32 {
		self.mb_width * 16 - self.crop[0] - self.crop[1]
	}

	pub fn height(&self) -> u32 {
		self.mb_height * 16 - self.crop[2] - self.crop[3]
	}

	pub fn max_frame_num(&self) -> u32 {
		1 << self.log2_max_frame_num
	}

	/// DPB capacity in frames (A.3.1 item h, or the VUI value).
	pub fn dpb_frames(&self) -> usize {
		let by_level = max_dpb_mbs(self.level_idc) / (self.mb_width * self.mb_height).max(1);
		let frames = self.max_dec_frame_buffering.unwrap_or(by_level.min(16));
		frames.max(self.max_num_ref_frames).clamp(1, 16) as usize
	}

	/// Pictures that may be held back for reordering before output.
	pub fn reorder_depth(&self) -> usize {
		let depth = self.max_num_reorder_frames.unwrap_or(self.dpb_frames() as u32);
		(depth as usize).min(self.dpb_frames())
	}
}

fn scaling_list<const N: usize>(bs: &mut BitReader, scan: &[u8; N], default: &[u8; N]) -> [u8; N] {
	let mut list = [0u8; N];
	let mut last = 8i32;
	let mut next = 8i32;
	for (j, &pos) in scan.iter().enumerate() {
		if next != 0 {
			let delta = bs.se();
			next = (last + delta + 256).rem_euclid(256);
			if j == 0 && next == 0 {
				return *default;
			}
		}
		let value = if next == 0 { last } else { next };
		list[usize::from(pos)] = value as u8;
		last = value;
	}
	list
}

/// Parses the scaling-list syntax for an SPS (`fallback` = None, rule A) or
/// a PPS (`fallback` = the SPS lists, rule B), Table 7-2.
fn parse_scaling(
	bs: &mut BitReader,
	count8: usize,
	fallback: Option<&ScalingLists>,
) -> ScalingLists {
	let mut s = ScalingLists::flat();
	for i in 0..6 {
		let default = &DEFAULT_SCALING_4X4[i / 3];
		if bs.bit() {
			s.l4[i] = scaling_list(bs, &ZIGZAG_4X4, default);
		} else if i == 0 || i == 3 {
			s.l4[i] = fallback.map_or(*default, |f| f.l4[i]);
		} else {
			s.l4[i] = s.l4[i - 1];
		}
	}
	for i in 0..2 {
		let default = &DEFAULT_SCALING_8X8[i];
		if i < count8 && bs.bit() {
			s.l8[i] = scaling_list(bs, &ZIGZAG_8X8, default);
		} else {
			s.l8[i] = fallback.map_or(*default, |f| f.l8[i]);
		}
	}
	s
}

fn skip_hrd(bs: &mut BitReader) {
	let count = bs.ue().min(31) + 1;
	bs.skip(8);
	for _ in 0..count {
		bs.ue();
		bs.ue();
		bs.skip(1);
	}
	bs.skip(20);
}

fn parse_vui(bs: &mut BitReader, sps: &mut Sps) {
	if bs.bit() && bs.bits(8) == 255 {
		bs.skip(32); // aspect_ratio_idc == Extended_SAR
	}
	if bs.bit() {
		bs.skip(1); // overscan
	}
	if bs.bit() {
		bs.skip(4);
		if bs.bit() {
			bs.skip(24);
		}
	}
	if bs.bit() {
		bs.ue();
		bs.ue();
	}
	if bs.bit() {
		bs.skip(65); // timing info
	}
	let nal_hrd = bs.bit();
	if nal_hrd {
		skip_hrd(bs);
	}
	let vcl_hrd = bs.bit();
	if vcl_hrd {
		skip_hrd(bs);
	}
	if nal_hrd || vcl_hrd {
		bs.skip(1);
	}
	bs.skip(1); // pic_struct_present_flag
	if bs.bit() {
		bs.skip(1);
		for _ in 0..4 {
			bs.ue();
		}
		let reorder = bs.ue();
		let buffering = bs.ue();
		if !bs.overrun() && reorder <= 16 && buffering <= 16 {
			sps.max_num_reorder_frames = Some(reorder);
			sps.max_dec_frame_buffering = Some(buffering.max(reorder));
		}
	}
}

/// Parses a sequence parameter set RBSP.
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, Unsupported> {
	let bad = Unsupported("invalid h264 sps");
	let mut bs = BitReader::new(rbsp);
	let profile_idc = bs.bits(8) as u8;
	bs.skip(8); // constraint flags
	let level_idc = bs.bits(8) as u8;
	let id = bs.ue() as usize;
	if id >= MAX_SPS {
		return Err(bad);
	}
	let mut chroma_format_idc = 1;
	let mut scaling = None;
	if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135)
	{
		chroma_format_idc = bs.ue();
		if chroma_format_idc == 3 {
			bs.skip(1);
		}
		let depth_luma = bs.ue() + 8;
		let depth_chroma = bs.ue() + 8;
		if depth_luma != 8 || depth_chroma != 8 {
			return Err(Unsupported("only 8-bit h264 is supported"));
		}
		if bs.bit() {
			return Err(Unsupported("lossless h264 (transform bypass) is not supported"));
		}
		if bs.bit() {
			scaling = Some(parse_scaling(&mut bs, if chroma_format_idc == 3 { 6 } else { 2 }, None));
		}
	}
	if chroma_format_idc != 1 {
		return Err(Unsupported("only 4:2:0 h264 is supported"));
	}
	let log2_max_frame_num = bs.ue() + 4;
	if log2_max_frame_num > 16 {
		return Err(bad);
	}
	let poc_type = bs.ue();
	let mut sps = Sps {
		profile_idc,
		level_idc,
		id,
		chroma_format_idc,
		scaling,
		log2_max_frame_num,
		poc_type,
		log2_max_poc_lsb: 0,
		delta_pic_order_always_zero: false,
		offset_for_non_ref_pic: 0,
		offset_for_top_to_bottom_field: 0,
		offset_for_ref_frame: Vec::new(),
		max_num_ref_frames: 0,
		gaps_in_frame_num_allowed: false,
		mb_width: 0,
		mb_height: 0,
		direct_8x8_inference: false,
		crop: [0; 4],
		max_num_reorder_frames: None,
		max_dec_frame_buffering: None,
	};
	match poc_type {
		0 => {
			sps.log2_max_poc_lsb = bs.ue() + 4;
			if sps.log2_max_poc_lsb > 16 {
				return Err(bad);
			}
		}
		1 => {
			sps.delta_pic_order_always_zero = bs.bit();
			sps.offset_for_non_ref_pic = bs.se();
			sps.offset_for_top_to_bottom_field = bs.se();
			let n = bs.ue();
			if n > 255 {
				return Err(bad);
			}
			sps.offset_for_ref_frame = (0..n).map(|_| bs.se()).collect();
		}
		2 => {}
		_ => return Err(bad),
	}
	sps.max_num_ref_frames = bs.ue();
	if sps.max_num_ref_frames > 16 {
		return Err(bad);
	}
	sps.gaps_in_frame_num_allowed = bs.bit();
	sps.mb_width = bs.ue().saturating_add(1);
	sps.mb_height = bs.ue().saturating_add(1);
	if !bs.bit() {
		return Err(Unsupported("interlaced h264 (field/mbaff coding) is not supported"));
	}
	if sps.mb_width.checked_mul(sps.mb_height).is_none_or(|n| n > MAX_MBS) {
		return Err(Unsupported("h264 picture too large"));
	}
	sps.direct_8x8_inference = bs.bit();
	if bs.bit() {
		let crop: [u32; 4] = [bs.ue(), bs.ue(), bs.ue(), bs.ue()];
		// 4:2:0 progressive: crop units are 2 luma samples.
		let crop = crop.map(|c| c.saturating_mul(2));
		if crop[0] + crop[1] >= sps.mb_width * 16 || crop[2] + crop[3] >= sps.mb_height * 16 {
			return Err(bad);
		}
		sps.crop = crop;
	}
	if bs.overrun() {
		return Err(bad);
	}
	// A truncated VUI is tolerated, as in FFmpeg: only the fields above are
	// required, and parse_vui keeps reorder values only when fully read.
	if bs.bit() {
		parse_vui(&mut bs, &mut sps);
	}
	Ok(sps)
}

#[derive(Debug, Clone)]
pub struct Pps {
	pub id: usize,
	pub sps_id: usize,
	pub cabac: bool,
	pub bottom_field_pic_order_present: bool,
	pub num_ref_idx_default: [u32; 2],
	pub weighted_pred: bool,
	pub weighted_bipred_idc: u32,
	pub init_qp: i32,
	pub chroma_qp_offset: [i32; 2],
	pub deblocking_control_present: bool,
	pub constrained_intra_pred: bool,
	pub redundant_pic_cnt_present: bool,
	pub transform_8x8_mode: bool,
	/// Effective scaling matrices (PPS over SPS over flat).
	pub scaling: ScalingLists,
	/// Dequantisation factors `LevelScale(m, i, j)`, see [`Dequant`].
	pub dequant: Box<Dequant>,
}

/// Precomputed `LevelScale4x4/8x8[list][qp % 6][raster]` (8.5.9).
#[derive(Debug, Clone)]
pub struct Dequant {
	pub l4: [[[i32; 16]; 6]; 6],
	pub l8: [[[i32; 64]; 6]; 2],
}

impl Dequant {
	pub fn new(s: &ScalingLists) -> Self {
		let mut d = Dequant { l4: [[[0; 16]; 6]; 6], l8: [[[0; 64]; 6]; 2] };
		for list in 0..6 {
			for m in 0..6 {
				for pos in 0..16 {
					let (i, j) = (pos / 4, pos % 4);
					let v = if i % 2 == 0 && j % 2 == 0 {
						NORM_ADJUST_4X4[m][0]
					} else if i % 2 == 1 && j % 2 == 1 {
						NORM_ADJUST_4X4[m][1]
					} else {
						NORM_ADJUST_4X4[m][2]
					};
					d.l4[list][m][pos] = i32::from(s.l4[list][pos]) * i32::from(v);
				}
			}
		}
		for list in 0..2 {
			for m in 0..6 {
				for pos in 0..64 {
					let (i, j) = (pos / 8, pos % 8);
					let idx = if i % 4 == 0 && j % 4 == 0 {
						0
					} else if i % 2 == 1 && j % 2 == 1 {
						1
					} else if i % 4 == 2 && j % 4 == 2 {
						2
					} else if (i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0) {
						3
					} else if (i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0) {
						4
					} else {
						5
					};
					d.l8[list][m][pos] = i32::from(s.l8[list][pos]) * i32::from(NORM_ADJUST_8X8[m][idx]);
				}
			}
		}
		d
	}
}

/// Parses a picture parameter set RBSP. `sps_lookup` gives the SPS it
/// refers to (needed for scaling-list fall-back).
pub fn parse_pps(
	rbsp: &[u8],
	sps_lookup: impl Fn(usize) -> Option<Sps>,
) -> Result<Pps, Unsupported> {
	let bad = Unsupported("invalid h264 pps");
	let mut bs = BitReader::new(rbsp);
	let id = bs.ue() as usize;
	let sps_id = bs.ue() as usize;
	if id >= MAX_PPS || sps_id >= MAX_SPS {
		return Err(bad);
	}
	let sps = sps_lookup(sps_id).ok_or(Unsupported("h264 pps refers to a missing sps"))?;
	let cabac = bs.bit();
	let bottom_field_pic_order_present = bs.bit();
	if bs.ue() != 0 {
		return Err(Unsupported("h264 slice groups (fmo) are not supported"));
	}
	let num_ref_idx_default = [bs.ue() + 1, bs.ue() + 1];
	if num_ref_idx_default.iter().any(|&n| n > 32) {
		return Err(bad);
	}
	let weighted_pred = bs.bit();
	let weighted_bipred_idc = bs.bits(2);
	let init_qp = 26 + bs.se();
	let _init_qs = bs.se();
	let chroma_offset = bs.se();
	if !(0..=51).contains(&init_qp) || !(-12..=12).contains(&chroma_offset) {
		return Err(bad);
	}
	let deblocking_control_present = bs.bit();
	let constrained_intra_pred = bs.bit();
	let redundant_pic_cnt_present = bs.bit();
	let mut transform_8x8_mode = false;
	let mut second_offset = chroma_offset;
	let mut scaling = sps.scaling.clone().unwrap_or_else(ScalingLists::flat);
	if more_rbsp_data(rbsp, bs.pos()) {
		transform_8x8_mode = bs.bit();
		if bs.bit() {
			let count8 = if transform_8x8_mode { 2 } else { 0 };
			scaling = parse_scaling(&mut bs, count8, sps.scaling.as_ref());
		}
		second_offset = bs.se();
		if !(-12..=12).contains(&second_offset) {
			return Err(bad);
		}
	}
	if bs.overrun() {
		return Err(bad);
	}
	let dequant = Box::new(Dequant::new(&scaling));
	Ok(Pps {
		id,
		sps_id,
		cabac,
		bottom_field_pic_order_present,
		num_ref_idx_default,
		weighted_pred,
		weighted_bipred_idc,
		init_qp,
		chroma_qp_offset: [chroma_offset, second_offset],
		deblocking_control_present,
		constrained_intra_pred,
		redundant_pic_cnt_present,
		transform_8x8_mode,
		scaling,
		dequant,
	})
}
