//! Per-block mode info (spec §6.4.5 – §6.4.20, §6.4.22; libvpx
//! `vp9_decodemv.c`, `vp9_pred_common.c`, `vp9_mvref_common.h`).

use super::decoder::{BlockPos, FrameDecoder};
use super::header::{
	COMPOUND_REFERENCE, REFERENCE_MODE_SELECT, SEG_REF_FRAME, SEG_SKIP, TX_MODE_SELECT,
};
use super::probs::MvComponentProbs;
use super::range::BoolDecoder;
use super::tables;
use super::{ALTREF_FRAME, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME, Mv, NO_REF, bs, pm};
use super::{INTER_MODE_TREE, INTERP_FILTER_TREE, INTRA_MODE_TREE, SEGMENT_TREE};
use super::{MAX_TX_SIZE, MV_CLASS_TREE, MV_FP_TREE, MV_JOINT_TREE, SIZE_GROUP};

/// Mode info of one block, replicated over the 8x8 cells it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeInfo {
	pub sb_type: u8,
	/// Luma intra mode or inter mode (of the last sub-block below 8x8).
	pub mode: u8,
	pub uv_mode: u8,
	/// Luma modes of the four 4x4 sub-blocks (below 8x8).
	pub sub_modes: [u8; 4],
	pub ref_frame: [i8; 2],
	pub mv: [Mv; 2],
	/// Motion vectors of the four 4x4 sub-blocks (below 8x8).
	pub sub_mv: [[Mv; 2]; 4],
	pub tx_size: u8,
	pub skip: bool,
	pub segment_id: u8,
	pub seg_id_predicted: bool,
	/// Interpolation filter (`INTERP_FILTER`: regular, smooth, sharp,
	/// bilinear); intra blocks hold 3, which is also the "no neighbour"
	/// value of the filter context.
	pub interp_filter: u8,
}

impl Default for ModeInfo {
	fn default() -> Self {
		Self {
			sb_type: bs::B8X8,
			mode: pm::DC,
			uv_mode: pm::DC,
			sub_modes: [pm::DC; 4],
			ref_frame: [INTRA_FRAME, NO_REF],
			mv: [Mv::ZERO; 2],
			sub_mv: [[Mv::ZERO; 2]; 4],
			tx_size: 0,
			skip: false,
			segment_id: 0,
			seg_id_predicted: false,
			interp_filter: 3,
		}
	}
}

impl ModeInfo {
	pub fn is_inter(&self) -> bool {
		self.ref_frame[0] > INTRA_FRAME
	}

	pub fn has_second_ref(&self) -> bool {
		self.ref_frame[1] > INTRA_FRAME
	}

	/// Luma mode of sub-block `b` (`get_y_mode`).
	fn y_mode(&self, b: usize) -> u8 {
		if self.sb_type < bs::B8X8 { self.sub_modes[b] } else { self.mode }
	}
}

/// Motion info kept per 8x8 cell for the next frame's candidate search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MvRef {
	pub ref_frame: [i8; 2],
	pub mv: [Mv; 2],
}

/// Neighbour positions (row, col) searched for candidate motion vectors,
/// by block size (`mv_ref_blocks`).
const MV_REF_BLOCKS: [[(i8, i8); 8]; 13] = [
	[(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
	[(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
	[(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
	[(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
	[(0, -1), (-1, 0), (1, -1), (-1, -1), (0, -2), (-2, 0), (-2, -1), (-1, -2)],
	[(-1, 0), (0, -1), (-1, 1), (-1, -1), (-2, 0), (0, -2), (-1, -2), (-2, -1)],
	[(-1, 0), (0, -1), (-1, 1), (1, -1), (-1, -1), (-3, 0), (0, -3), (-3, -3)],
	[(0, -1), (-1, 0), (2, -1), (-1, -1), (-1, 1), (0, -3), (-3, 0), (-3, -3)],
	[(-1, 0), (0, -1), (-1, 2), (-1, -1), (1, -1), (-3, 0), (0, -3), (-3, -3)],
	[(-1, 1), (1, -1), (-1, 2), (2, -1), (-1, -1), (-3, 0), (0, -3), (-3, -3)],
	[(0, -1), (-1, 0), (4, -1), (-1, 2), (-1, -1), (0, -3), (-3, 0), (2, -1)],
	[(-1, 0), (0, -1), (-1, 4), (2, -1), (-1, -1), (-3, 0), (0, -3), (-1, 2)],
	[(-1, 3), (3, -1), (-1, 4), (4, -1), (-1, -1), (-1, 0), (0, -1), (-1, 6)],
];

/// Weights for the inter mode context: 9 per intra, 3 per ZEROMV, 1 per
/// NEWMV neighbour (`mode_2_counter`) …
const MODE_2_COUNTER: [u8; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
/// … and the context of each sum (`counter_to_context`; 9 = invalid).
const COUNTER_TO_CONTEXT: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];

/// Sub-block of a below-8x8 neighbour whose vector a sub-block candidate
/// uses: `[block][neighbour is left]` (`idx_n_column_to_subblock`).
const IDX_N_COLUMN_TO_SUBBLOCK: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

fn use_mv_hp(mv: Mv) -> bool {
	i32::from(mv.row).abs() < 64 && i32::from(mv.col).abs() < 64
}

fn lower_mv_precision(mv: &mut Mv, allow_hp: bool) {
	if !(allow_hp && use_mv_hp(*mv)) {
		if mv.row & 1 != 0 {
			mv.row += if mv.row > 0 { -1 } else { 1 };
		}
		if mv.col & 1 != 0 {
			mv.col += if mv.col > 0 { -1 } else { 1 };
		}
	}
}

impl FrameDecoder {
	fn above_mi(&self, p: &BlockPos) -> Option<ModeInfo> {
		p.have_above.then(|| self.mi[(p.row - 1) * self.mi_cols + p.col])
	}

	fn left_mi(&self, p: &BlockPos) -> Option<ModeInfo> {
		p.have_left.then(|| self.mi[p.row * self.mi_cols + p.col - 1])
	}

	/// Reads the mode info of a block (`vp9_read_mode_info`).
	pub(super) fn read_mode_info(
		&mut self,
		r: &mut BoolDecoder,
		p: &BlockPos,
		sb_type: u8,
		sub_w: usize,
		sub_h: usize,
	) -> crate::Result<ModeInfo> {
		let mut mi = ModeInfo { sb_type, ..Default::default() };
		let above = self.above_mi(p);
		let left = self.left_mi(p);
		if self.h.intra_only_frame() {
			mi.segment_id = self.read_intra_segment_id(r, p);
			mi.skip = self.read_skip(r, mi.segment_id, above, left);
			mi.tx_size = self.read_tx_size(r, sb_type, true, above, left);
			mi.ref_frame = [INTRA_FRAME, NO_REF];
			let probs = |mi: &ModeInfo, b: usize| {
				let a = if b < 2 {
					above.filter(|m| !m.is_inter()).map_or(pm::DC, |m| m.y_mode(b + 2))
				} else {
					mi.sub_modes[b - 2]
				};
				let l = if b & 1 == 0 {
					left.filter(|m| !m.is_inter()).map_or(pm::DC, |m| m.y_mode(b + 1))
				} else {
					mi.sub_modes[b - 1]
				};
				&tables::KF_Y_MODE_PROBS[usize::from(a)][usize::from(l)]
			};
			match sb_type {
				bs::B4X4 => {
					for b in 0..4 {
						mi.sub_modes[b] = r.tree(&INTRA_MODE_TREE, probs(&mi, b));
					}
					mi.mode = mi.sub_modes[3];
				}
				bs::B4X8 => {
					let m = r.tree(&INTRA_MODE_TREE, probs(&mi, 0));
					mi.sub_modes[0] = m;
					mi.sub_modes[2] = m;
					let m = r.tree(&INTRA_MODE_TREE, probs(&mi, 1));
					mi.sub_modes[1] = m;
					mi.sub_modes[3] = m;
					mi.mode = m;
				}
				bs::B8X4 => {
					let m = r.tree(&INTRA_MODE_TREE, probs(&mi, 0));
					mi.sub_modes[0] = m;
					mi.sub_modes[1] = m;
					let m = r.tree(&INTRA_MODE_TREE, probs(&mi, 2));
					mi.sub_modes[2] = m;
					mi.sub_modes[3] = m;
					mi.mode = m;
				}
				_ => {
					mi.mode = r.tree(&INTRA_MODE_TREE, probs(&mi, 0));
					mi.sub_modes = [mi.mode; 4];
				}
			}
			mi.uv_mode = r.tree(&INTRA_MODE_TREE, &tables::KF_UV_MODE_PROBS[usize::from(mi.mode)]);
			return Ok(mi);
		}

		mi.segment_id = self.read_inter_segment_id(r, p, &mut mi, above, left);
		mi.skip = self.read_skip(r, mi.segment_id, above, left);
		let inter = self.read_is_inter(r, mi.segment_id, above, left);
		mi.tx_size = self.read_tx_size(r, sb_type, !mi.skip || !inter, above, left);
		if inter {
			self.read_inter_block(r, p, &mut mi, above, left, sub_w, sub_h)?;
		} else {
			self.read_intra_block(r, &mut mi);
		}
		// Motion info for the next frame's candidate search.
		let entry = MvRef { ref_frame: mi.ref_frame, mv: mi.mv };
		for y in 0..p.y_mis {
			let at = (p.row + y) * self.mi_cols + p.col;
			self.cur_mvs[at..at + p.x_mis].fill(entry);
		}
		Ok(mi)
	}

	fn read_intra_segment_id(&mut self, r: &mut BoolDecoder, p: &BlockPos) -> u8 {
		let seg = &self.seg;
		if !seg.enabled {
			return 0;
		}
		if !seg.update_map {
			self.copy_segment_ids(p);
			return 0;
		}
		let id = r.tree(&SEGMENT_TREE, &seg.tree_probs);
		self.set_segment_ids(p, id);
		id
	}

	fn read_inter_segment_id(
		&mut self,
		r: &mut BoolDecoder,
		p: &BlockPos,
		mi: &mut ModeInfo,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
	) -> u8 {
		if !self.seg.enabled {
			return 0;
		}
		let mut predicted = 7u8;
		for y in 0..p.y_mis {
			for x in 0..p.x_mis {
				let at = (p.row + y) * self.mi_cols + p.col + x;
				predicted = predicted.min(self.last_seg_map.get(at).copied().unwrap_or(0));
			}
		}
		if !self.seg.update_map {
			self.copy_segment_ids(p);
			return predicted;
		}
		let id = if self.seg.temporal_update {
			let ctx = usize::from(above.is_some_and(|m| m.seg_id_predicted))
				+ usize::from(left.is_some_and(|m| m.seg_id_predicted));
			mi.seg_id_predicted = r.read(self.seg.pred_probs[ctx]);
			if mi.seg_id_predicted { predicted } else { r.tree(&SEGMENT_TREE, &self.seg.tree_probs) }
		} else {
			r.tree(&SEGMENT_TREE, &self.seg.tree_probs)
		};
		self.set_segment_ids(p, id);
		id
	}

	fn copy_segment_ids(&mut self, p: &BlockPos) {
		for y in 0..p.y_mis {
			let at = (p.row + y) * self.mi_cols + p.col;
			for x in at..at + p.x_mis {
				self.seg_map[x] = self.last_seg_map.get(x).copied().unwrap_or(0);
			}
		}
	}

	fn set_segment_ids(&mut self, p: &BlockPos, id: u8) {
		for y in 0..p.y_mis {
			let at = (p.row + y) * self.mi_cols + p.col;
			self.seg_map[at..at + p.x_mis].fill(id);
		}
	}

	fn read_skip(
		&mut self,
		r: &mut BoolDecoder,
		segment: u8,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
	) -> bool {
		if self.seg.active(segment, SEG_SKIP) {
			return true;
		}
		let ctx =
			usize::from(above.is_some_and(|m| m.skip)) + usize::from(left.is_some_and(|m| m.skip));
		let skip = r.read(self.fc.skip[ctx]);
		if self.counting {
			self.counts.skip[ctx][usize::from(skip)] += 1;
		}
		skip
	}

	fn read_tx_size(
		&mut self,
		r: &mut BoolDecoder,
		sb_type: u8,
		allow_select: bool,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
	) -> u8 {
		let max = MAX_TX_SIZE[usize::from(sb_type)];
		let tx_mode = self.ch.tx_mode;
		if !(allow_select && tx_mode == TX_MODE_SELECT && sb_type >= bs::B8X8) {
			return max.min(tx_mode.min(3));
		}
		let side = |m: Option<ModeInfo>| m.map(|m| if m.skip { max } else { m.tx_size });
		let (mut a, mut l) = (side(above), side(left));
		if l.is_none() {
			l = a;
		}
		if a.is_none() {
			a = l;
		}
		let ctx = usize::from(a.unwrap_or(max) + l.unwrap_or(max) > max);
		let probs: &[u8] = match max {
			1 => &self.fc.tx8[ctx],
			2 => &self.fc.tx16[ctx],
			_ => &self.fc.tx32[ctx],
		};
		let mut tx = u8::from(r.read(probs[0]));
		if tx != 0 && max >= 2 {
			tx += u8::from(r.read(probs[1]));
			if tx != 1 && max >= 3 {
				tx += u8::from(r.read(probs[2]));
			}
		}
		if self.counting {
			match max {
				1 => self.counts.tx8[ctx][usize::from(tx)] += 1,
				2 => self.counts.tx16[ctx][usize::from(tx)] += 1,
				_ => self.counts.tx32[ctx][usize::from(tx)] += 1,
			}
		}
		tx
	}

	fn read_is_inter(
		&mut self,
		r: &mut BoolDecoder,
		segment: u8,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
	) -> bool {
		if self.seg.active(segment, SEG_REF_FRAME) {
			return self.seg.data(segment, SEG_REF_FRAME) != i16::from(INTRA_FRAME);
		}
		let ctx = match (above, left) {
			(Some(a), Some(l)) => {
				let (ai, li) = (!a.is_inter(), !l.is_inter());
				if ai && li { 3 } else { usize::from(ai || li) }
			}
			(Some(m), None) | (None, Some(m)) => 2 * usize::from(!m.is_inter()),
			(None, None) => 0,
		};
		let inter = r.read(self.fc.intra_inter[ctx]);
		if self.counting {
			self.counts.intra_inter[ctx][usize::from(inter)] += 1;
		}
		inter
	}

	fn read_intra_block(&mut self, r: &mut BoolDecoder, mi: &mut ModeInfo) {
		let y_mode = |r: &mut BoolDecoder, group: usize, this: &mut Self| {
			let m = r.tree(&INTRA_MODE_TREE, &this.fc.y_mode[group]);
			if this.counting {
				this.counts.y_mode[group][usize::from(m)] += 1;
			}
			m
		};
		match mi.sb_type {
			bs::B4X4 => {
				for b in 0..4 {
					mi.sub_modes[b] = y_mode(r, 0, self);
				}
				mi.mode = mi.sub_modes[3];
			}
			bs::B4X8 => {
				let m = y_mode(r, 0, self);
				mi.sub_modes[0] = m;
				mi.sub_modes[2] = m;
				let m = y_mode(r, 0, self);
				mi.sub_modes[1] = m;
				mi.sub_modes[3] = m;
				mi.mode = m;
			}
			bs::B8X4 => {
				let m = y_mode(r, 0, self);
				mi.sub_modes[0] = m;
				mi.sub_modes[1] = m;
				let m = y_mode(r, 0, self);
				mi.sub_modes[2] = m;
				mi.sub_modes[3] = m;
				mi.mode = m;
			}
			t => {
				mi.mode = y_mode(r, usize::from(SIZE_GROUP[usize::from(t)]), self);
				mi.sub_modes = [mi.mode; 4];
			}
		}
		mi.uv_mode = r.tree(&INTRA_MODE_TREE, &self.fc.uv_mode[usize::from(mi.mode)]);
		if self.counting {
			self.counts.uv_mode[usize::from(mi.mode)][usize::from(mi.uv_mode)] += 1;
		}
		mi.interp_filter = 3;
		mi.ref_frame = [INTRA_FRAME, NO_REF];
	}

	// --- Reference frames (vp9_pred_common.c) ---

	fn reference_mode_context(&self, above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
		let fixed = self.ch.comp_fixed_ref;
		match (above, left) {
			(Some(a), Some(l)) => {
				if !a.has_second_ref() && !l.has_second_ref() {
					usize::from((a.ref_frame[0] == fixed) ^ (l.ref_frame[0] == fixed))
				} else if !a.has_second_ref() {
					2 + usize::from(a.ref_frame[0] == fixed || !a.is_inter())
				} else if !l.has_second_ref() {
					2 + usize::from(l.ref_frame[0] == fixed || !l.is_inter())
				} else {
					4
				}
			}
			(Some(m), None) | (None, Some(m)) => {
				if !m.has_second_ref() {
					usize::from(m.ref_frame[0] == fixed)
				} else {
					3
				}
			}
			(None, None) => 1,
		}
	}

	fn comp_ref_context(&self, above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
		let fixed = self.ch.comp_fixed_ref;
		let var = self.ch.comp_var_ref;
		let fix_idx = usize::from(self.h.sign_bias[fixed as usize]);
		let var_idx = 1 - fix_idx;
		match (above, left) {
			(Some(a), Some(l)) => {
				let (ai, li) = (!a.is_inter(), !l.is_inter());
				if ai && li {
					2
				} else if ai || li {
					let e = if ai { l } else { a };
					if !e.has_second_ref() {
						1 + 2 * usize::from(e.ref_frame[0] != var[1])
					} else {
						1 + 2 * usize::from(e.ref_frame[var_idx] != var[1])
					}
				} else {
					let l_sg = !l.has_second_ref();
					let a_sg = !a.has_second_ref();
					let vrfa = if a_sg { a.ref_frame[0] } else { a.ref_frame[var_idx] };
					let vrfl = if l_sg { l.ref_frame[0] } else { l.ref_frame[var_idx] };
					if vrfa == vrfl && var[1] == vrfa {
						0
					} else if l_sg && a_sg {
						if (vrfa == fixed && vrfl == var[0]) || (vrfl == fixed && vrfa == var[0]) {
							4
						} else if vrfa == vrfl {
							3
						} else {
							1
						}
					} else if l_sg || a_sg {
						let vrfc = if l_sg { vrfa } else { vrfl };
						let rfs = if a_sg { vrfa } else { vrfl };
						if vrfc == var[1] && rfs != var[1] {
							1
						} else if rfs == var[1] && vrfc != var[1] {
							2
						} else {
							4
						}
					} else if vrfa == vrfl {
						4
					} else {
						2
					}
				}
			}
			(Some(e), None) | (None, Some(e)) => {
				if !e.is_inter() {
					2
				} else if e.has_second_ref() {
					4 * usize::from(e.ref_frame[var_idx] != var[1])
				} else {
					3 * usize::from(e.ref_frame[0] != var[1])
				}
			}
			(None, None) => 2,
		}
	}

	fn single_ref_p1_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
		let has_last = |m: &ModeInfo| m.ref_frame[0] == LAST_FRAME || m.ref_frame[1] == LAST_FRAME;
		match (above, left) {
			(Some(a), Some(l)) => {
				let (ai, li) = (!a.is_inter(), !l.is_inter());
				if ai && li {
					2
				} else if ai || li {
					let e = if ai { l } else { a };
					if !e.has_second_ref() {
						4 * usize::from(e.ref_frame[0] == LAST_FRAME)
					} else {
						1 + usize::from(has_last(&e))
					}
				} else {
					let (a2, l2) = (a.has_second_ref(), l.has_second_ref());
					if a2 && l2 {
						1 + usize::from(has_last(&a) || has_last(&l))
					} else if a2 || l2 {
						let rfs = if !a2 { a.ref_frame[0] } else { l.ref_frame[0] };
						let c = if a2 { a } else { l };
						if rfs == LAST_FRAME {
							3 + usize::from(has_last(&c))
						} else {
							usize::from(has_last(&c))
						}
					} else {
						2 * usize::from(a.ref_frame[0] == LAST_FRAME)
							+ 2 * usize::from(l.ref_frame[0] == LAST_FRAME)
					}
				}
			}
			(Some(e), None) | (None, Some(e)) => {
				if !e.is_inter() {
					2
				} else if !e.has_second_ref() {
					4 * usize::from(e.ref_frame[0] == LAST_FRAME)
				} else {
					1 + usize::from(has_last(&e))
				}
			}
			(None, None) => 2,
		}
	}

	fn single_ref_p2_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
		let has_golden =
			|m: &ModeInfo| m.ref_frame[0] == GOLDEN_FRAME || m.ref_frame[1] == GOLDEN_FRAME;
		match (above, left) {
			(Some(a), Some(l)) => {
				let (ai, li) = (!a.is_inter(), !l.is_inter());
				if ai && li {
					2
				} else if ai || li {
					let e = if ai { l } else { a };
					if !e.has_second_ref() {
						if e.ref_frame[0] == LAST_FRAME {
							3
						} else {
							4 * usize::from(e.ref_frame[0] == GOLDEN_FRAME)
						}
					} else {
						1 + 2 * usize::from(has_golden(&e))
					}
				} else {
					let (a2, l2) = (a.has_second_ref(), l.has_second_ref());
					let (a0, a1, l0, l1) = (a.ref_frame[0], a.ref_frame[1], l.ref_frame[0], l.ref_frame[1]);
					if a2 && l2 {
						if a0 == l0 && a1 == l1 { 3 * usize::from(has_golden(&a) || has_golden(&l)) } else { 2 }
					} else if a2 || l2 {
						let rfs = if !a2 { a0 } else { l0 };
						let c = if a2 { a } else { l };
						if rfs == GOLDEN_FRAME {
							3 + usize::from(has_golden(&c))
						} else if rfs == ALTREF_FRAME {
							usize::from(has_golden(&c))
						} else {
							1 + 2 * usize::from(has_golden(&c))
						}
					} else if a0 == LAST_FRAME && l0 == LAST_FRAME {
						3
					} else if a0 == LAST_FRAME || l0 == LAST_FRAME {
						let edge = if a0 == LAST_FRAME { l0 } else { a0 };
						4 * usize::from(edge == GOLDEN_FRAME)
					} else {
						2 * usize::from(a0 == GOLDEN_FRAME) + 2 * usize::from(l0 == GOLDEN_FRAME)
					}
				}
			}
			(Some(e), None) | (None, Some(e)) => {
				if !e.is_inter() || (e.ref_frame[0] == LAST_FRAME && !e.has_second_ref()) {
					2
				} else if !e.has_second_ref() {
					4 * usize::from(e.ref_frame[0] == GOLDEN_FRAME)
				} else {
					3 * usize::from(has_golden(&e))
				}
			}
			(None, None) => 2,
		}
	}

	fn read_ref_frames(
		&mut self,
		r: &mut BoolDecoder,
		segment: u8,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
	) -> [i8; 2] {
		if self.seg.active(segment, SEG_REF_FRAME) {
			return [self.seg.data(segment, SEG_REF_FRAME) as i8, NO_REF];
		}
		let mode = if self.ch.reference_mode == REFERENCE_MODE_SELECT {
			let ctx = self.reference_mode_context(above, left);
			let m = r.read(self.fc.comp_inter[ctx]);
			if self.counting {
				self.counts.comp_inter[ctx][usize::from(m)] += 1;
			}
			if m { COMPOUND_REFERENCE } else { 0 }
		} else {
			self.ch.reference_mode
		};
		if mode == COMPOUND_REFERENCE {
			let idx = usize::from(self.h.sign_bias[self.ch.comp_fixed_ref as usize]);
			let ctx = self.comp_ref_context(above, left);
			let bit = r.read(self.fc.comp_ref[ctx]);
			if self.counting {
				self.counts.comp_ref[ctx][usize::from(bit)] += 1;
			}
			let mut refs = [0i8; 2];
			refs[idx] = self.ch.comp_fixed_ref;
			refs[1 - idx] = self.ch.comp_var_ref[usize::from(bit)];
			refs
		} else {
			let ctx0 = Self::single_ref_p1_context(above, left);
			let bit0 = r.read(self.fc.single_ref[ctx0][0]);
			if self.counting {
				self.counts.single_ref[ctx0][0][usize::from(bit0)] += 1;
			}
			if bit0 {
				let ctx1 = Self::single_ref_p2_context(above, left);
				let bit1 = r.read(self.fc.single_ref[ctx1][1]);
				if self.counting {
					self.counts.single_ref[ctx1][1][usize::from(bit1)] += 1;
				}
				[if bit1 { ALTREF_FRAME } else { GOLDEN_FRAME }, NO_REF]
			} else {
				[LAST_FRAME, NO_REF]
			}
		}
	}

	// --- Inter blocks ---

	fn read_inter_mode(&mut self, r: &mut BoolDecoder, ctx: usize) -> u8 {
		let m = r.tree(&INTER_MODE_TREE, &self.fc.inter_mode[ctx]);
		if self.counting {
			self.counts.inter_mode[ctx][usize::from(m)] += 1;
		}
		pm::NEARESTMV + m
	}

	fn mi_at(&self, p: &BlockPos, (dr, dc): (i8, i8)) -> Option<ModeInfo> {
		let row = p.row as isize + isize::from(dr);
		let col = p.col as isize + isize::from(dc);
		let inside = row >= 0
			&& (row as usize) < self.mi_rows
			&& col >= self.tile.mi_col_start as isize
			&& (col as usize) < self.tile.mi_col_end;
		inside.then(|| self.mi[row as usize * self.mi_cols + col as usize])
	}

	fn mode_context(&self, p: &BlockPos, sb_type: u8) -> usize {
		let search = &MV_REF_BLOCKS[usize::from(sb_type)];
		let counter: usize = search[..2]
			.iter()
			.filter_map(|&pos| self.mi_at(p, pos))
			.map(|m| usize::from(MODE_2_COUNTER[usize::from(m.mode)]))
			.sum();
		usize::from(COUNTER_TO_CONTEXT[counter])
	}

	fn clamp_mv_ref(p: &BlockPos, mv: Mv) -> Mv {
		const BORDER: i32 = 16 << 3;
		Mv::new(
			i32::from(mv.row).clamp(p.to_top - BORDER, p.to_bottom + BORDER),
			i32::from(mv.col).clamp(p.to_left - BORDER, p.to_right + BORDER),
		)
	}

	/// Candidate motion vectors for `ref_frame` (`dec_find_mv_refs`).
	/// Returns the list and how many entries are meaningful.
	fn find_mv_refs(
		&self,
		p: &BlockPos,
		sb_type: u8,
		mode: u8,
		ref_frame: i8,
		block: Option<usize>,
	) -> ([Mv; 2], usize) {
		let bias = &self.h.sign_bias;
		let search = &MV_REF_BLOCKS[usize::from(sb_type)];
		let early_break = mode != pm::NEARMV;
		let mut list = [Mv::ZERO; 2];
		let mut count = 0usize;
		let prev = self.prev_mvs.as_ref().map(|m| m[p.row * self.mi_cols + p.col]);

		// Adds a candidate; returns true when the search is complete.
		let add = |mv: Mv, list: &mut [Mv; 2], count: &mut usize| -> bool {
			if *count > 0 {
				if mv != list[0] {
					list[1] = mv;
					*count = 2;
					return true;
				}
				false
			} else {
				list[0] = mv;
				*count = 1;
				early_break
			}
		};
		let scale = |m: &ModeInfo, i: usize| -> Mv {
			let mv = m.mv[i];
			if bias[m.ref_frame[i] as usize] != bias[ref_frame as usize] { mv.negated() } else { mv }
		};

		let done = 'search: {
			let mut different_ref_found = false;
			let mut start = 0;
			if let Some(block) = block {
				for (i, &pos) in search.iter().enumerate().take(2) {
					if let Some(c) = self.mi_at(p, pos) {
						different_ref_found = true;
						let sub = |which: usize| {
							if c.sb_type < bs::B8X8 {
								c.sub_mv[IDX_N_COLUMN_TO_SUBBLOCK[block][usize::from(pos.1 == 0)]][which]
							} else {
								c.mv[which]
							}
						};
						if c.ref_frame[0] == ref_frame {
							if add(sub(0), &mut list, &mut count) {
								break 'search true;
							}
						} else if c.ref_frame[1] == ref_frame && add(sub(1), &mut list, &mut count) {
							break 'search true;
						}
					}
					start = i + 1;
				}
			}
			for &pos in &search[start..] {
				if let Some(c) = self.mi_at(p, pos) {
					different_ref_found = true;
					if c.ref_frame[0] == ref_frame {
						if add(c.mv[0], &mut list, &mut count) {
							break 'search true;
						}
					} else if c.ref_frame[1] == ref_frame && add(c.mv[1], &mut list, &mut count) {
						break 'search true;
					}
				}
			}
			if let Some(prev) = prev {
				if prev.ref_frame[0] == ref_frame {
					if add(prev.mv[0], &mut list, &mut count) {
						break 'search true;
					}
				} else if prev.ref_frame[1] == ref_frame && add(prev.mv[1], &mut list, &mut count) {
					break 'search true;
				}
			}
			if different_ref_found {
				for &pos in search {
					if let Some(c) = self.mi_at(p, pos) {
						if c.is_inter() {
							if c.ref_frame[0] != ref_frame && add(scale(&c, 0), &mut list, &mut count) {
								break 'search true;
							}
							if c.has_second_ref()
								&& c.ref_frame[1] != ref_frame
								&& c.mv[1] != c.mv[0]
								&& add(scale(&c, 1), &mut list, &mut count)
							{
								break 'search true;
							}
						}
					}
				}
			}
			if let Some(prev) = prev {
				if prev.ref_frame[0] != ref_frame && prev.ref_frame[0] > INTRA_FRAME {
					let mut mv = prev.mv[0];
					if bias[prev.ref_frame[0] as usize] != bias[ref_frame as usize] {
						mv = mv.negated();
					}
					if add(mv, &mut list, &mut count) {
						break 'search true;
					}
				}
				if prev.ref_frame[1] > INTRA_FRAME
					&& prev.ref_frame[1] != ref_frame
					&& prev.mv[1] != prev.mv[0]
				{
					let mut mv = prev.mv[1];
					if bias[prev.ref_frame[1] as usize] != bias[ref_frame as usize] {
						mv = mv.negated();
					}
					if add(mv, &mut list, &mut count) {
						break 'search true;
					}
				}
			}
			false
		};
		if !done {
			count = if mode == pm::NEARMV { 2 } else { 1 };
		}
		for mv in list.iter_mut().take(count) {
			*mv = Self::clamp_mv_ref(p, *mv);
		}
		(list, count)
	}

	/// Candidate for sub-block `block` (`append_sub8x8_mvs_for_idx`).
	fn sub8x8_mv(&self, p: &BlockPos, mi: &ModeInfo, mode: u8, block: usize, which: usize) -> Mv {
		let ref_frame = mi.ref_frame[which];
		let sub = |b: usize| mi.sub_mv[b][which];
		match block {
			0 => {
				let (list, count) = self.find_mv_refs(p, mi.sb_type, mode, ref_frame, Some(0));
				list[count - 1]
			}
			1 | 2 => {
				if mode == pm::NEARESTMV {
					sub(0)
				} else {
					let (list, _) = self.find_mv_refs(p, mi.sb_type, mode, ref_frame, Some(block));
					list.into_iter().find(|&mv| mv != sub(0)).unwrap_or(Mv::ZERO)
				}
			}
			_ => {
				if mode == pm::NEARESTMV {
					sub(2)
				} else if sub(2) != sub(1) {
					sub(1)
				} else if sub(2) != sub(0) {
					sub(0)
				} else {
					let (list, _) = self.find_mv_refs(p, mi.sb_type, mode, ref_frame, Some(block));
					list.into_iter().find(|&mv| mv != sub(2)).unwrap_or(Mv::ZERO)
				}
			}
		}
	}

	fn read_mv_component(r: &mut BoolDecoder, c: &MvComponentProbs, use_hp: bool) -> i32 {
		let sign = r.read(c.sign);
		let class = r.tree(&MV_CLASS_TREE, &c.classes);
		let (d, mut mag) = if class == 0 {
			(u32::from(r.read(c.class0[0])), 0)
		} else {
			let mut d = 0u32;
			for i in 0..usize::from(class) {
				d |= u32::from(r.read(c.bits[i])) << i;
			}
			(d, 2i32 << (class + 2))
		};
		let fr = r.tree(&MV_FP_TREE, if class == 0 { &c.class0_fp[d as usize] } else { &c.fp });
		let hp = if use_hp { r.read(if class == 0 { c.class0_hp } else { c.hp }) } else { true };
		mag += ((d << 3) | (u32::from(fr) << 1) | u32::from(hp)) as i32 + 1;
		if sign { -mag } else { mag }
	}

	/// Reads a new motion vector relative to `reference`; `None` when out
	/// of range (a corrupt block).
	fn read_mv(&mut self, r: &mut BoolDecoder, reference: Mv) -> Option<Mv> {
		let joint = r.tree(&MV_JOINT_TREE, &self.fc.mv_joints);
		let use_hp = self.h.allow_hp && use_mv_hp(reference);
		let row =
			if joint == 2 || joint == 3 { Self::read_mv_component(r, &self.fc.mv[0], use_hp) } else { 0 };
		let col =
			if joint == 1 || joint == 3 { Self::read_mv_component(r, &self.fc.mv[1], use_hp) } else { 0 };
		if self.counting {
			self.counts.add_mv(row, col);
		}
		let row = i32::from(reference.row) + row;
		let col = i32::from(reference.col) + col;
		const LIMIT: i32 = 1 << 14;
		(row > -LIMIT && row < LIMIT - 1 && col > -LIMIT && col < LIMIT - 1).then(|| Mv::new(row, col))
	}

	#[allow(clippy::too_many_arguments)]
	fn read_inter_block(
		&mut self,
		r: &mut BoolDecoder,
		p: &BlockPos,
		mi: &mut ModeInfo,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
		sub_w: usize,
		sub_h: usize,
	) -> crate::Result<()> {
		let corrupt = || crate::message::Message::Codec("vp9: invalid motion vector");
		mi.ref_frame = self.read_ref_frames(r, mi.segment_id, above, left);
		let compound = mi.has_second_ref();
		let ctx = self.mode_context(p, mi.sb_type);
		if self.seg.active(mi.segment_id, SEG_SKIP) {
			mi.mode = pm::ZEROMV;
			if mi.sb_type < bs::B8X8 {
				return Err(crate::message::Message::Codec("vp9: segment skip on a block below 8x8"));
			}
		} else if mi.sb_type >= bs::B8X8 {
			mi.mode = self.read_inter_mode(r, ctx);
		}
		mi.interp_filter = if self.h.interp_filter == super::FILTER_SWITCHABLE {
			let lt = left.map_or(3, |m| m.interp_filter);
			let at = above.map_or(3, |m| m.interp_filter);
			let fctx = if lt == at {
				lt
			} else if lt == 3 {
				at
			} else if at == 3 {
				lt
			} else {
				3
			};
			let f = r.tree(&INTERP_FILTER_TREE, &self.fc.interp_filter[usize::from(fctx)]);
			if self.counting {
				self.counts.interp_filter[usize::from(fctx)][usize::from(f)] += 1;
			}
			f
		} else {
			self.h.interp_filter
		};

		let refs = 1 + usize::from(compound);
		if mi.sb_type < bs::B8X8 {
			let mut best_ref = [Mv::ZERO; 2];
			let mut got_new_refs = false;
			let mut b_mode = pm::ZEROMV;
			for idy in (0..2).step_by(sub_h) {
				for idx in (0..2).step_by(sub_w) {
					let j = idy * 2 + idx;
					b_mode = self.read_inter_mode(r, ctx);
					let mut mvs = [Mv::ZERO; 2];
					match b_mode {
						pm::NEARESTMV | pm::NEARMV => {
							for (k, mv) in mvs.iter_mut().enumerate().take(refs) {
								*mv = self.sub8x8_mv(p, mi, b_mode, j, k);
							}
						}
						pm::NEWMV => {
							if !got_new_refs {
								for (k, best) in best_ref.iter_mut().enumerate().take(refs) {
									let (list, _) =
										self.find_mv_refs(p, mi.sb_type, pm::NEWMV, mi.ref_frame[k], None);
									let mut mv = list[0];
									lower_mv_precision(&mut mv, self.h.allow_hp);
									*best = mv;
								}
								got_new_refs = true;
							}
							for k in 0..refs {
								mvs[k] = self.read_mv(r, best_ref[k]).ok_or_else(corrupt)?;
							}
						}
						_ => {}
					}
					mi.sub_mv[j] = mvs;
					if sub_h == 2 {
						mi.sub_mv[j + 2] = mvs;
					}
					if sub_w == 2 {
						mi.sub_mv[j + 1] = mvs;
					}
				}
			}
			mi.mode = b_mode;
			mi.mv = mi.sub_mv[3];
		} else {
			let mut best = [Mv::ZERO; 2];
			if mi.mode != pm::ZEROMV {
				for (k, b) in best.iter_mut().enumerate().take(refs) {
					let (list, count) = self.find_mv_refs(p, mi.sb_type, mi.mode, mi.ref_frame[k], None);
					let mut mv = list[count - 1];
					lower_mv_precision(&mut mv, self.h.allow_hp);
					*b = mv;
				}
			}
			match mi.mode {
				pm::NEWMV => {
					for (mv, &b) in mi.mv.iter_mut().zip(&best).take(refs) {
						*mv = self.read_mv(r, b).ok_or_else(corrupt)?;
					}
				}
				pm::NEARESTMV | pm::NEARMV => mi.mv = best,
				_ => mi.mv = [Mv::ZERO; 2],
			}
			mi.sub_mv = [mi.mv; 4];
		}
		Ok(())
	}
}
