//! Slice header parsing (ISO/IEC 14496-10 §7.3.3; FFmpeg
//! `h264_slice.c: h264_slice_header_parse`, `h264_parse.c`).

use super::params::{Pps, Sps};
use crate::codecs::bitstream::BitReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
	P,
	B,
	I,
}

/// One `modification_of_pic_nums_idc` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefModification {
	/// idc 0 / 1: subtract / add `abs_diff_pic_num_minus1 + 1`.
	ShortTerm {
		add: bool,
		abs_diff: u32,
	},
	LongTerm(u32),
}

/// `memory_management_control_operation` (Table 7-9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mmco {
	ForgetShort { diff: u32 },
	ForgetLong { long_term_pic_num: u32 },
	ShortToLong { diff: u32, long_term_frame_idx: u32 },
	MaxLongIdx { max_plus1: u32 },
	Reset,
	CurrentToLong { long_term_frame_idx: u32 },
}

/// Explicit weighted prediction for one reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Weight {
	pub luma: (i32, i32),
	pub chroma: [(i32, i32); 2],
	/// `true` if any weight/offset was transmitted for this reference.
	pub explicit: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WeightTable {
	pub luma_log2_denom: u32,
	pub chroma_log2_denom: u32,
	pub weights: [Vec<Weight>; 2],
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
	pub first_mb: u32,
	pub slice_type: SliceType,
	pub pps_id: usize,
	pub frame_num: u32,
	pub idr: bool,
	pub nal_ref_idc: u8,
	pub idr_pic_id: u32,
	pub poc_lsb: u32,
	pub delta_poc_bottom: i32,
	pub delta_poc: [i32; 2],
	pub redundant_pic_cnt: u32,
	pub direct_spatial: bool,
	pub num_ref_idx: [usize; 2],
	pub ref_mods: [Vec<RefModification>; 2],
	pub weights: Option<WeightTable>,
	pub no_output_of_prior_pics: bool,
	pub long_term_reference: bool,
	pub mmco: Option<Vec<Mmco>>,
	pub cabac_init_idc: usize,
	pub qp: i32,
	pub disable_deblocking: u32,
	pub alpha_offset: i32,
	pub beta_offset: i32,
	/// Bit offset of slice_data() within the RBSP.
	pub data_bit_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BadSlice(pub &'static str);

/// Reads `first_mb_in_slice` and `pic_parameter_set_id`, enough to find the
/// parameter sets before the full parse.
pub fn peek_pps_id(rbsp: &[u8]) -> Option<usize> {
	let mut bs = BitReader::new(rbsp);
	bs.ue();
	bs.ue();
	let id = bs.ue() as usize;
	(!bs.overrun()).then_some(id)
}

fn parse_ref_mods(bs: &mut BitReader) -> Result<Vec<RefModification>, BadSlice> {
	let mut mods = Vec::new();
	if !bs.bit() {
		return Ok(mods);
	}
	loop {
		match bs.ue() {
			0 => mods.push(RefModification::ShortTerm { add: false, abs_diff: bs.ue() }),
			1 => mods.push(RefModification::ShortTerm { add: true, abs_diff: bs.ue() }),
			2 => mods.push(RefModification::LongTerm(bs.ue())),
			3 => return Ok(mods),
			_ => return Err(BadSlice("invalid ref_pic_list_modification")),
		}
		if mods.len() > 33 || bs.overrun() {
			return Err(BadSlice("too many ref_pic_list_modifications"));
		}
	}
}

fn parse_weights(
	bs: &mut BitReader,
	counts: [usize; 2],
	lists: usize,
) -> Result<WeightTable, BadSlice> {
	let luma_log2_denom = bs.ue();
	let chroma_log2_denom = bs.ue();
	if luma_log2_denom > 7 || chroma_log2_denom > 7 {
		return Err(BadSlice("invalid weight denominator"));
	}
	let mut table =
		WeightTable { luma_log2_denom, chroma_log2_denom, weights: [Vec::new(), Vec::new()] };
	for (list, &count) in counts.iter().enumerate().take(lists) {
		for _ in 0..count {
			let mut w = Weight {
				luma: (1 << luma_log2_denom, 0),
				chroma: [(1 << chroma_log2_denom, 0); 2],
				explicit: false,
			};
			if bs.bit() {
				w.luma = (bs.se(), bs.se());
				w.explicit = true;
			}
			if bs.bit() {
				for c in &mut w.chroma {
					*c = (bs.se(), bs.se());
				}
				w.explicit = true;
			}
			let in_range =
				|(wt, off): (i32, i32)| (-128..=127).contains(&wt) && (-128..=127).contains(&off);
			if !in_range(w.luma) || !w.chroma.iter().all(|&c| in_range(c)) {
				return Err(BadSlice("weight out of range"));
			}
			table.weights[list].push(w);
		}
	}
	Ok(table)
}

fn parse_mmco(bs: &mut BitReader) -> Result<Vec<Mmco>, BadSlice> {
	let mut ops = Vec::new();
	loop {
		let op = match bs.ue() {
			0 => return Ok(ops),
			1 => Mmco::ForgetShort { diff: bs.ue() },
			2 => Mmco::ForgetLong { long_term_pic_num: bs.ue() },
			3 => Mmco::ShortToLong { diff: bs.ue(), long_term_frame_idx: bs.ue() },
			4 => Mmco::MaxLongIdx { max_plus1: bs.ue() },
			5 => Mmco::Reset,
			6 => Mmco::CurrentToLong { long_term_frame_idx: bs.ue() },
			_ => return Err(BadSlice("invalid mmco")),
		};
		ops.push(op);
		if ops.len() > 66 || bs.overrun() {
			return Err(BadSlice("too many mmco operations"));
		}
	}
}

/// Parses a slice header. `nal_kind` / `nal_ref_idc` come from the NAL
/// header.
pub fn parse(
	rbsp: &[u8],
	nal_kind: u8,
	nal_ref_idc: u8,
	sps: &Sps,
	pps: &Pps,
) -> Result<SliceHeader, BadSlice> {
	let mut bs = BitReader::new(rbsp);
	let first_mb = bs.ue();
	let raw_type = bs.ue();
	let slice_type = match raw_type % 5 {
		0 => SliceType::P,
		1 => SliceType::B,
		2 => SliceType::I,
		_ => return Err(BadSlice("sp/si slices are not supported")),
	};
	if raw_type > 9 {
		return Err(BadSlice("invalid slice_type"));
	}
	let pps_id = bs.ue() as usize;
	let frame_num = bs.bits(sps.log2_max_frame_num);
	let idr = nal_kind == super::nal::kind::IDR;
	let idr_pic_id = if idr { bs.ue() } else { 0 };
	let mut poc_lsb = 0;
	let mut delta_poc_bottom = 0;
	let mut delta_poc = [0; 2];
	if sps.poc_type == 0 {
		poc_lsb = bs.bits(sps.log2_max_poc_lsb);
		if pps.bottom_field_pic_order_present {
			delta_poc_bottom = bs.se();
		}
	} else if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
		delta_poc[0] = bs.se();
		if pps.bottom_field_pic_order_present {
			delta_poc[1] = bs.se();
		}
	}
	let redundant_pic_cnt = if pps.redundant_pic_cnt_present { bs.ue() } else { 0 };
	let direct_spatial = slice_type == SliceType::B && bs.bit();
	let mut num_ref_idx = [pps.num_ref_idx_default[0] as usize, pps.num_ref_idx_default[1] as usize];
	if slice_type != SliceType::I && bs.bit() {
		num_ref_idx[0] = bs.ue() as usize + 1;
		if slice_type == SliceType::B {
			num_ref_idx[1] = bs.ue() as usize + 1;
		}
	}
	if num_ref_idx.iter().any(|&n| n > 32) {
		return Err(BadSlice("num_ref_idx_active out of range"));
	}
	let lists = match slice_type {
		SliceType::I => 0,
		SliceType::P => 1,
		SliceType::B => 2,
	};
	if lists < 2 {
		num_ref_idx[1] = 0;
	}
	if lists < 1 {
		num_ref_idx[0] = 0;
	}
	let mut ref_mods = [Vec::new(), Vec::new()];
	for m in ref_mods.iter_mut().take(lists) {
		*m = parse_ref_mods(&mut bs)?;
	}
	let weighted = (pps.weighted_pred && slice_type == SliceType::P)
		|| (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B);
	let weights = if weighted { Some(parse_weights(&mut bs, num_ref_idx, lists)?) } else { None };
	let mut no_output_of_prior_pics = false;
	let mut long_term_reference = false;
	let mut mmco = None;
	if nal_ref_idc != 0 {
		if idr {
			no_output_of_prior_pics = bs.bit();
			long_term_reference = bs.bit();
		} else if bs.bit() {
			mmco = Some(parse_mmco(&mut bs)?);
		}
	}
	let mut cabac_init_idc = 0;
	if pps.cabac && slice_type != SliceType::I {
		cabac_init_idc = bs.ue() as usize;
		if cabac_init_idc > 2 {
			return Err(BadSlice("invalid cabac_init_idc"));
		}
	}
	let qp = pps.init_qp + bs.se();
	if !(0..=51).contains(&qp) {
		return Err(BadSlice("slice qp out of range"));
	}
	let mut disable_deblocking = 0;
	let mut alpha_offset = 0;
	let mut beta_offset = 0;
	if pps.deblocking_control_present {
		disable_deblocking = bs.ue();
		if disable_deblocking > 2 {
			return Err(BadSlice("invalid disable_deblocking_filter_idc"));
		}
		if disable_deblocking != 1 {
			alpha_offset = bs.se() * 2;
			beta_offset = bs.se() * 2;
			if !(-12..=12).contains(&alpha_offset) || !(-12..=12).contains(&beta_offset) {
				return Err(BadSlice("deblocking offset out of range"));
			}
		}
	}
	if bs.overrun() {
		return Err(BadSlice("truncated slice header"));
	}
	Ok(SliceHeader {
		first_mb,
		slice_type,
		pps_id,
		frame_num,
		idr,
		nal_ref_idc,
		idr_pic_id,
		poc_lsb,
		delta_poc_bottom,
		delta_poc,
		redundant_pic_cnt,
		direct_spatial,
		num_ref_idx,
		ref_mods,
		weights,
		no_output_of_prior_pics,
		long_term_reference,
		mmco,
		cabac_init_idc,
		qp,
		disable_deblocking,
		alpha_offset,
		beta_offset,
		data_bit_offset: bs.pos(),
	})
}
