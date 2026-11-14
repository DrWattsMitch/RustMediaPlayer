//! Slice segment header (ITU-T H.265 §7.3.6; FFmpeg `hls_slice_header`).

use super::ps::{Pps, ShortTermRps, Sps, parse_short_term_rps};
use crate::codecs::bitstream::BitReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
	B = 0,
	P = 1,
	I = 2,
}

/// Explicit weighted prediction for one reference.
#[derive(Debug, Clone, Copy, Default)]
pub struct Weight {
	pub luma_weight: i32,
	pub luma_offset: i32,
	pub chroma_weight: [i32; 2],
	pub chroma_offset: [i32; 2],
}

#[derive(Debug, Clone)]
pub struct PredWeights {
	pub luma_log2_denom: u32,
	pub chroma_log2_denom: u32,
	pub l: [[Weight; 16]; 2],
}

#[derive(Debug, Clone)]
pub struct LongTermRefs {
	pub count: usize,
	/// PocLsbLt.
	pub poc_lsb: [i32; 32],
	/// DeltaPocMsbCycleLt (accumulated).
	pub msb_cycle: [i32; 32],
	pub used: [bool; 32],
	/// `true` when the full POC is known (delta_poc_msb_present_flag).
	pub has_msb: [bool; 32],
}

#[derive(Debug, Clone)]
pub struct SliceHeader {
	pub first_slice_in_pic: bool,
	pub no_output_of_prior_pics: bool,
	pub pps_id: usize,
	pub dependent: bool,
	pub segment_address: u32,
	pub slice_type: SliceType,
	pub pic_output: bool,
	pub poc_lsb: u32,
	pub short_term_rps: Option<ShortTermRps>,
	/// Bits of the slice header's own st_ref_pic_set (0 when from SPS).
	pub long_term: LongTermRefs,
	pub temporal_mvp_enabled: bool,
	pub sao_luma: bool,
	pub sao_chroma: bool,
	pub num_ref_idx: [u32; 2],
	pub list_entry: [Option<[u32; 16]>; 2],
	pub mvd_l1_zero: bool,
	pub cabac_init: bool,
	pub collocated_from_l0: bool,
	pub collocated_ref_idx: u32,
	pub weights: Option<PredWeights>,
	pub max_num_merge_cand: u32,
	pub slice_qp: i32,
	pub cb_qp_offset: i32,
	pub cr_qp_offset: i32,
	pub deblocking_disabled: bool,
	pub beta_offset: i32,
	pub tc_offset: i32,
	pub loop_filter_across_slices: bool,
	/// Byte offset of slice_data() in the unescaped NAL unit.
	pub data_offset: usize,
	/// Address (raster scan) of the first CTB of the independent slice this
	/// segment belongs to (`SliceAddrRs`).
	pub slice_addr_rs: u32,
}

#[derive(Debug)]
pub struct SliceError(pub &'static str);

fn check(cond: bool, what: &'static str) -> Result<(), SliceError> {
	if cond { Ok(()) } else { Err(SliceError(what)) }
}

/// NAL unit types (Table 7-1).
pub mod nal {
	pub const TRAIL_N: u8 = 0;
	pub const RASL_N: u8 = 8;
	pub const RASL_R: u8 = 9;
	pub const BLA_W_LP: u8 = 16;
	pub const BLA_W_RADL: u8 = 17;
	pub const BLA_N_LP: u8 = 18;
	pub const IDR_W_RADL: u8 = 19;
	pub const IDR_N_LP: u8 = 20;
	pub const CRA_NUT: u8 = 21;
	pub const VPS: u8 = 32;
	pub const SPS: u8 = 33;
	pub const PPS: u8 = 34;
	pub const AUD: u8 = 35;
	pub const EOS: u8 = 36;
	pub const EOB: u8 = 37;
}

pub fn is_irap(t: u8) -> bool {
	(16..=23).contains(&t)
}

pub fn is_idr(t: u8) -> bool {
	t == nal::IDR_W_RADL || t == nal::IDR_N_LP
}

/// Sub-layer non-reference picture (§7.4.2.2).
pub fn is_sub_layer_non_ref(t: u8) -> bool {
	t <= 14 && t.is_multiple_of(2)
}

/// Parses a slice segment header. `prev` is the header of the preceding
/// independent slice segment (dependent segments inherit from it).
pub fn parse(
	rbsp: &[u8],
	nal_type: u8,
	sps_list: &[Option<std::sync::Arc<Sps>>],
	pps_list: &[Option<std::sync::Arc<Pps>>],
	prev: Option<&SliceHeader>,
) -> Result<SliceHeader, SliceError> {
	let mut bs = BitReader::new(rbsp);
	bs.skip(16);
	let first_slice_in_pic = bs.bit();
	let no_output_of_prior_pics = if is_irap(nal_type) { bs.bit() } else { false };
	let pps_id = bs.ue() as usize;
	let pps = pps_list
		.get(pps_id)
		.and_then(|p| p.as_ref())
		.ok_or(SliceError("hevc: slice refers to missing pps"))?;
	let sps =
		sps_list.get(pps.sps_id).and_then(|s| s.as_ref()).ok_or(SliceError("hevc: missing sps"))?;
	let mut dependent = false;
	let mut segment_address = 0;
	if !first_slice_in_pic {
		if pps.dependent_slice_segments_enabled {
			dependent = bs.bit();
		}
		let n = sps.ctb_width * sps.ctb_height;
		let bits = 32 - (n - 1).leading_zeros();
		segment_address = if bits > 0 { bs.bits(bits) } else { 0 };
		check(segment_address < n, "hevc: invalid slice address")?;
	}
	if dependent {
		let p = prev.ok_or(SliceError("hevc: dependent slice without independent slice"))?;
		check(p.pps_id == pps_id, "hevc: pps change in dependent slice")?;
		let mut h = p.clone();
		h.first_slice_in_pic = first_slice_in_pic;
		h.dependent = true;
		h.segment_address = segment_address;
		finish_entry_points(&mut bs, pps, &mut h)?;
		return Ok(h);
	}
	bs.skip(pps.num_extra_slice_header_bits as usize);
	let slice_type = match bs.ue() {
		0 => SliceType::B,
		1 => SliceType::P,
		2 => SliceType::I,
		_ => return Err(SliceError("hevc: invalid slice type")),
	};
	let pic_output = if pps.output_flag_present { bs.bit() } else { true };
	let mut poc_lsb = 0;
	let mut short_term_rps = None;
	let mut long_term = LongTermRefs {
		count: 0,
		poc_lsb: [0; 32],
		msb_cycle: [0; 32],
		used: [false; 32],
		has_msb: [false; 32],
	};
	let mut temporal_mvp_enabled = false;
	if !is_idr(nal_type) {
		poc_lsb = bs.bits(sps.log2_max_poc_lsb);
		let from_sps = bs.bit();
		if !from_sps {
			let n = sps.short_term_rps.len();
			short_term_rps = Some(
				parse_short_term_rps(&mut bs, n, n, &sps.short_term_rps, true)
					.map_err(|e| SliceError(e.0))?,
			);
		} else {
			check(!sps.short_term_rps.is_empty(), "hevc: no short-term rps in sps")?;
			let n = sps.short_term_rps.len() as u32;
			let bits = 32 - (n - 1).leading_zeros();
			let idx = if n > 1 { bs.bits(bits) as usize } else { 0 };
			short_term_rps =
				Some(sps.short_term_rps.get(idx).cloned().ok_or(SliceError("hevc: invalid rps idx"))?);
		}
		if sps.long_term_refs_present {
			let num_sps = if sps.lt_ref_pic_poc_lsb.is_empty() { 0 } else { bs.ue() as usize };
			let num_pics = bs.ue() as usize;
			check(
				num_sps + num_pics <= 32 && num_sps <= sps.lt_ref_pic_poc_lsb.len(),
				"hevc: too many lt refs",
			)?;
			long_term.count = num_sps + num_pics;
			let mut prev_msb = 0i32;
			for i in 0..long_term.count {
				let (lsb, used) = if i < num_sps {
					let n = sps.lt_ref_pic_poc_lsb.len() as u32;
					let bits = 32 - (n - 1).leading_zeros();
					let idx = if n > 1 { bs.bits(bits) as usize } else { 0 };
					(sps.lt_ref_pic_poc_lsb[idx] as i32, sps.used_by_curr_pic_lt[idx])
				} else {
					(bs.bits(sps.log2_max_poc_lsb) as i32, bs.bit())
				};
				let msb_present = bs.bit();
				if msb_present {
					let mut delta = bs.ue() as i32;
					if i != 0 && i != num_sps {
						delta += prev_msb;
					}
					prev_msb = delta;
					long_term.msb_cycle[i] = delta;
				}
				long_term.poc_lsb[i] = lsb;
				long_term.used[i] = used;
				long_term.has_msb[i] = msb_present;
			}
		}
		if sps.temporal_mvp_enabled {
			temporal_mvp_enabled = bs.bit();
		}
	}
	let (mut sao_luma, mut sao_chroma) = (false, false);
	if sps.sao_enabled {
		sao_luma = bs.bit();
		sao_chroma = bs.bit();
	}
	let mut num_ref_idx = [0u32; 2];
	let mut list_entry = [None, None];
	let mut mvd_l1_zero = false;
	let mut cabac_init = false;
	let mut collocated_from_l0 = true;
	let mut collocated_ref_idx = 0;
	let mut weights = None;
	let mut max_num_merge_cand = 5;
	if slice_type != SliceType::I {
		num_ref_idx = [pps.num_ref_idx_l0_default, pps.num_ref_idx_l1_default];
		if bs.bit() {
			num_ref_idx[0] = bs.ue() + 1;
			if slice_type == SliceType::B {
				num_ref_idx[1] = bs.ue() + 1;
			}
		}
		if slice_type == SliceType::P {
			num_ref_idx[1] = 0;
		}
		check(num_ref_idx[0] <= 16 && num_ref_idx[1] <= 16, "hevc: too many references")?;
		let num_pic_total_curr = num_pic_total_curr(short_term_rps.as_ref(), &long_term);
		if pps.lists_modification_present && num_pic_total_curr > 1 {
			let bits = 32 - (num_pic_total_curr - 1).leading_zeros();
			for l in 0..if slice_type == SliceType::B { 2 } else { 1 } {
				if bs.bit() {
					let mut e = [0u32; 16];
					for v in e.iter_mut().take(num_ref_idx[l] as usize) {
						*v = bs.bits(bits);
					}
					list_entry[l] = Some(e);
				}
			}
		}
		if slice_type == SliceType::B {
			mvd_l1_zero = bs.bit();
		}
		if pps.cabac_init_present {
			cabac_init = bs.bit();
		}
		if temporal_mvp_enabled {
			if slice_type == SliceType::B {
				collocated_from_l0 = bs.bit();
			}
			let list = if collocated_from_l0 { 0 } else { 1 };
			if num_ref_idx[list] > 1 {
				collocated_ref_idx = bs.ue();
				check(collocated_ref_idx < num_ref_idx[list], "hevc: invalid collocated_ref_idx")?;
			}
		}
		if (pps.weighted_pred && slice_type == SliceType::P)
			|| (pps.weighted_bipred && slice_type == SliceType::B)
		{
			weights = Some(pred_weight_table(&mut bs, sps, slice_type, &num_ref_idx)?);
		}
		max_num_merge_cand = 5 - bs.ue();
		check((1..=5).contains(&max_num_merge_cand), "hevc: invalid merge candidate count")?;
	}
	let slice_qp = pps.init_qp + bs.se();
	check(slice_qp >= -sps.qp_bd_offset && slice_qp <= 51, "hevc: invalid slice qp")?;
	let (mut cb_qp_offset, mut cr_qp_offset) = (0, 0);
	if pps.slice_chroma_qp_offsets_present {
		cb_qp_offset = bs.se();
		cr_qp_offset = bs.se();
		check(
			(-12..=12).contains(&cb_qp_offset) && (-12..=12).contains(&cr_qp_offset),
			"hevc: chroma qp offset",
		)?;
	}
	let mut deblocking_disabled = pps.deblocking_disabled;
	let mut beta_offset = pps.beta_offset;
	let mut tc_offset = pps.tc_offset;
	if pps.deblocking_override_enabled && bs.bit() {
		deblocking_disabled = bs.bit();
		if !deblocking_disabled {
			beta_offset = bs.se() * 2;
			tc_offset = bs.se() * 2;
			check(
				(-12..=12).contains(&beta_offset) && (-12..=12).contains(&tc_offset),
				"hevc: deblock offsets",
			)?;
		}
	}
	let mut loop_filter_across_slices = pps.loop_filter_across_slices;
	if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
		loop_filter_across_slices = bs.bit();
	}
	let mut h = SliceHeader {
		first_slice_in_pic,
		no_output_of_prior_pics,
		pps_id,
		dependent: false,
		segment_address,
		slice_type,
		pic_output,
		poc_lsb,
		short_term_rps,
		long_term,
		temporal_mvp_enabled,
		sao_luma,
		sao_chroma,
		num_ref_idx,
		list_entry,
		mvd_l1_zero,
		cabac_init,
		collocated_from_l0,
		collocated_ref_idx,
		weights,
		max_num_merge_cand,
		slice_qp,
		cb_qp_offset,
		cr_qp_offset,
		deblocking_disabled,
		beta_offset,
		tc_offset,
		loop_filter_across_slices,
		data_offset: 0,
		slice_addr_rs: segment_address,
	};
	finish_entry_points(&mut bs, pps, &mut h)?;
	Ok(h)
}

/// Entry points, header extension and byte alignment.
fn finish_entry_points(
	bs: &mut BitReader,
	pps: &Pps,
	h: &mut SliceHeader,
) -> Result<(), SliceError> {
	if pps.tiles_enabled || pps.entropy_coding_sync {
		let n = bs.ue();
		check(n <= 4400, "hevc: too many entry points")?;
		if n > 0 {
			let len = bs.ue() + 1;
			check(len <= 32, "hevc: invalid entry point size")?;
			bs.skip(n as usize * len as usize);
		}
	}
	if pps.slice_header_extension_present {
		let len = bs.ue() as usize;
		bs.skip(8 * len);
	}
	// byte_alignment(): a 1 then zeros.
	bs.skip(1);
	bs.align();
	check(!bs.overrun(), "hevc: truncated slice header")?;
	h.data_offset = bs.pos() / 8;
	Ok(())
}

/// NumPicTotalCurr (7-55).
pub fn num_pic_total_curr(st: Option<&ShortTermRps>, lt: &LongTermRefs) -> u32 {
	let mut n = 0;
	if let Some(st) = st {
		n += st.used[..st.num_delta_pocs].iter().filter(|&&u| u).count() as u32;
	}
	n += lt.used[..lt.count].iter().filter(|&&u| u).count() as u32;
	n
}

fn pred_weight_table(
	bs: &mut BitReader,
	sps: &Sps,
	slice_type: SliceType,
	num_ref_idx: &[u32; 2],
) -> Result<PredWeights, SliceError> {
	let luma_log2_denom = bs.ue();
	check(luma_log2_denom <= 7, "hevc: invalid luma weight denominator")?;
	let chroma_log2_denom = (luma_log2_denom as i32 + bs.se()) as u32;
	check(chroma_log2_denom <= 7, "hevc: invalid chroma weight denominator")?;
	let mut l = [[Weight::default(); 16]; 2];
	let lists = if slice_type == SliceType::B { 2 } else { 1 };
	let high = 1 << (sps.bit_depth - 1).min(7); // WpOffsetHalfRangeC (no high precision offsets)
	let shift = sps.bit_depth - 8;
	for list in 0..lists {
		let n = num_ref_idx[list] as usize;
		let mut luma_flags = [false; 16];
		let mut chroma_flags = [false; 16];
		for f in luma_flags.iter_mut().take(n) {
			*f = bs.bit();
		}
		for f in chroma_flags.iter_mut().take(n) {
			*f = bs.bit();
		}
		for i in 0..n {
			let w = &mut l[list][i];
			w.luma_weight = 1 << luma_log2_denom;
			w.chroma_weight = [1 << chroma_log2_denom; 2];
			if luma_flags[i] {
				let dw = bs.se();
				check((-128..=127).contains(&dw), "hevc: invalid luma weight")?;
				w.luma_weight += dw;
				let off = bs.se();
				check((-128..=127).contains(&off), "hevc: invalid luma offset")?;
				w.luma_offset = off << shift;
			}
			if chroma_flags[i] {
				for c in 0..2 {
					let dw = bs.se();
					check((-128..=127).contains(&dw), "hevc: invalid chroma weight")?;
					w.chroma_weight[c] += dw;
					let doff = bs.se();
					check((-512..=511).contains(&doff), "hevc: invalid chroma offset")?;
					let off =
						(doff - ((128 * w.chroma_weight[c]) >> chroma_log2_denom) + 128).clamp(-128, 127);
					w.chroma_offset[c] = off << shift;
				}
			}
		}
	}
	let _ = high;
	Ok(PredWeights { luma_log2_denom, chroma_log2_denom, l })
}
