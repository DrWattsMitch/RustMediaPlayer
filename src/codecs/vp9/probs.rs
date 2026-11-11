//! Probability contexts (spec §8.4.2, §8.4.3; libvpx `vp9_entropymode.c`,
//! `vp9_entropy.c`, `vp9_entropymv.c`, `vpx_dsp/prob.h`).
//!
//! A frame decodes with a [`FrameContext`] loaded from one of four saved
//! contexts and patched by the compressed header. Unless the frame is
//! error resilient or in frame-parallel mode, the symbol [`Counts`] it
//! collects are then merged back into the probabilities ("backward
//! adaptation") before the context is saved.

use super::range::BoolDecoder;
use super::tables;
use super::{INTER_MODE_TREE, INTERP_FILTER_TREE, INTRA_MODE_TREE, PARTITION_TREE, Tree};
use super::{MV_CLASS_TREE, MV_CLASS0_TREE, MV_FP_TREE, MV_JOINT_TREE};

/// `[tx size][plane type][ref][band][context][node]`.
pub type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];
/// Token counts with the same indices: ZERO, ONE, TWO+ and EOB.
pub type CoefCounts = [[[[[[u32; 4]; 6]; 6]; 2]; 2]; 4];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvComponentProbs {
	pub sign: u8,
	pub classes: [u8; 10],
	pub class0: [u8; 1],
	pub bits: [u8; 10],
	pub class0_fp: [[u8; 3]; 2],
	pub fp: [u8; 3],
	pub class0_hp: u8,
	pub hp: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameContext {
	pub y_mode: [[u8; 9]; 4],
	pub uv_mode: [[u8; 9]; 10],
	pub partition: [[u8; 3]; 16],
	pub coef: CoefProbs,
	pub skip: [u8; 3],
	pub inter_mode: [[u8; 3]; 7],
	pub interp_filter: [[u8; 2]; 4],
	pub intra_inter: [u8; 4],
	pub comp_inter: [u8; 5],
	pub single_ref: [[u8; 2]; 5],
	pub comp_ref: [u8; 5],
	pub tx8: [[u8; 1]; 2],
	pub tx16: [[u8; 2]; 2],
	pub tx32: [[u8; 3]; 2],
	pub mv_joints: [u8; 3],
	pub mv: [MvComponentProbs; 2],
}

impl Default for FrameContext {
	fn default() -> Self {
		let mv_bits = [136, 140, 148, 160, 176, 192, 224, 234, 234, 240];
		let component = |classes: [u8; 10], class0: u8| MvComponentProbs {
			sign: 128,
			classes,
			class0: [class0],
			bits: mv_bits,
			class0_fp: [[128, 128, 64], [96, 112, 64]],
			fp: [64, 96, 64],
			class0_hp: 160,
			hp: 128,
		};
		Self {
			y_mode: tables::DEFAULT_Y_MODE_PROBS,
			uv_mode: tables::DEFAULT_UV_MODE_PROBS,
			partition: tables::DEFAULT_PARTITION_PROBS,
			coef: tables::DEFAULT_COEF_PROBS,
			skip: tables::DEFAULT_SKIP_PROBS,
			inter_mode: tables::DEFAULT_INTER_MODE_PROBS,
			interp_filter: tables::DEFAULT_INTERP_FILTER_PROBS,
			intra_inter: tables::DEFAULT_INTRA_INTER_PROBS,
			comp_inter: tables::DEFAULT_COMP_INTER_PROBS,
			single_ref: tables::DEFAULT_SINGLE_REF_PROBS,
			comp_ref: tables::DEFAULT_COMP_REF_PROBS,
			tx8: [[100], [66]],
			tx16: [[20, 152], [15, 101]],
			tx32: [[3, 136, 37], [5, 52, 13]],
			mv_joints: [32, 64, 96],
			mv: [
				component([224, 144, 192, 168, 192, 176, 192, 198, 198, 245], 216),
				component([216, 128, 176, 160, 176, 176, 192, 198, 198, 208], 208),
			],
		}
	}
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MvComponentCounts {
	pub sign: [u32; 2],
	pub classes: [u32; 11],
	pub class0: [u32; 2],
	pub bits: [[u32; 2]; 10],
	pub class0_fp: [[u32; 4]; 2],
	pub fp: [u32; 4],
	pub class0_hp: [u32; 2],
	pub hp: [u32; 2],
}

/// Symbol counts of one frame (libvpx `FRAME_COUNTS`).
#[derive(Debug, Clone, Default)]
pub struct Counts {
	pub y_mode: [[u32; 10]; 4],
	pub uv_mode: [[u32; 10]; 10],
	pub partition: [[u32; 4]; 16],
	pub coef: CoefCounts,
	/// Times the EOB branch was coded, same indices.
	pub eob_branch: [[[[[u32; 6]; 6]; 2]; 2]; 4],
	pub skip: [[u32; 2]; 3],
	/// Indexed by mode - NEARESTMV.
	pub inter_mode: [[u32; 4]; 7],
	pub interp_filter: [[u32; 3]; 4],
	pub intra_inter: [[u32; 2]; 4],
	pub comp_inter: [[u32; 2]; 5],
	pub single_ref: [[[u32; 2]; 2]; 5],
	pub comp_ref: [[u32; 2]; 5],
	pub tx8: [[u32; 2]; 2],
	pub tx16: [[u32; 3]; 2],
	pub tx32: [[u32; 4]; 2],
	pub mv_joints: [u32; 4],
	pub mv: [MvComponentCounts; 2],
}

impl Counts {
	/// Counts a decoded motion vector difference (libvpx `vp9_inc_mv`):
	/// the high-precision bit is counted even when it was not coded (it
	/// then reads as 1), as libvpx does.
	pub fn add_mv(&mut self, row: i32, col: i32) {
		let joint = usize::from(row != 0) * 2 + usize::from(col != 0);
		self.mv_joints[joint] += 1;
		for (comp, v) in [(0, row), (1, col)] {
			if v == 0 {
				continue;
			}
			let c = &mut self.mv[comp];
			c.sign[usize::from(v < 0)] += 1;
			let z = v.unsigned_abs() as usize - 1;
			let class = mv_class(z);
			c.classes[class] += 1;
			let offset = z - if class == 0 { 0 } else { 2 << (class + 2) };
			let (d, f, e) = (offset >> 3, (offset >> 1) & 3, offset & 1);
			if class == 0 {
				c.class0[d] += 1;
				c.class0_fp[d][f] += 1;
				c.class0_hp[e] += 1;
			} else {
				for i in 0..class {
					c.bits[i][(d >> i) & 1] += 1;
				}
				c.fp[f] += 1;
				c.hp[e] += 1;
			}
		}
	}
}

/// Magnitude class of `z = |v| - 1` (`vp9_get_mv_class`).
fn mv_class(z: usize) -> usize {
	if z >= 2 * 4096 {
		return 10;
	}
	let x = z >> 3;
	if x == 0 { 0 } else { (usize::BITS - 1 - x.leading_zeros()) as usize }
}

/// `inv_map_table`: the first 20 entries are coarse steps of 13, then
/// every other probability in order.
const INV_MAP: [u8; 255] = {
	let mut t = [0u8; 255];
	let mut i = 0;
	while i < 20 {
		t[i] = (7 + 13 * i) as u8;
		i += 1;
	}
	let mut v = 1;
	while v < 254 {
		if (v - 7) % 13 != 0 || v < 7 {
			t[i] = v as u8;
			i += 1;
		}
		v += 1;
	}
	t[254] = 253;
	t
};

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
	if v > 2 * m {
		v
	} else if v & 1 == 1 {
		m - ((v + 1) >> 1)
	} else {
		m + (v >> 1)
	}
}

/// A differential probability update (spec §9.3.3 `diff_update_prob`,
/// libvpx `vp9_diff_update_prob`).
pub fn diff_update(r: &mut BoolDecoder, p: &mut u8) {
	if !r.read(252) {
		return;
	}
	let d = if !r.bit() {
		r.literal(4)
	} else if !r.bit() {
		r.literal(4) + 16
	} else if !r.bit() {
		r.literal(5) + 32
	} else {
		let v = r.literal(7);
		(if v < 65 { v } else { (v << 1) - 65 + u32::from(r.bit()) }) + 64
	};
	let v = i32::from(INV_MAP[(d as usize).min(254)]);
	let m = i32::from(*p);
	*p = if m <= 128 {
		1 + inv_recenter_nonneg(v, m - 1)
	} else {
		255 - inv_recenter_nonneg(v, 255 - m)
	} as u8;
}

/// Motion vector probability updates use a plain 7-bit value.
fn mv_update(r: &mut BoolDecoder, p: &mut u8) {
	if r.read(252) {
		*p = ((r.literal(7) << 1) | 1) as u8;
	}
}

/// Coefficient probability updates for every transform size up to
/// `max_tx` (`read_coef_probs`).
pub fn read_coef_updates(r: &mut BoolDecoder, fc: &mut FrameContext, max_tx: usize) {
	for tx in 0..=max_tx {
		if !r.bit() {
			continue;
		}
		for plane in &mut fc.coef[tx] {
			for reference in plane.iter_mut() {
				for (band, contexts) in reference.iter_mut().enumerate() {
					let n = if band == 0 { 3 } else { 6 };
					for ctx in contexts.iter_mut().take(n) {
						for p in ctx.iter_mut() {
							diff_update(r, p);
						}
					}
				}
			}
		}
	}
}

pub fn read_mv_updates(r: &mut BoolDecoder, fc: &mut FrameContext, allow_hp: bool) {
	for p in fc.mv_joints.iter_mut() {
		mv_update(r, p);
	}
	for comp in fc.mv.iter_mut() {
		mv_update(r, &mut comp.sign);
		for p in comp.classes.iter_mut() {
			mv_update(r, p);
		}
		mv_update(r, &mut comp.class0[0]);
		for p in comp.bits.iter_mut() {
			mv_update(r, p);
		}
	}
	for comp in fc.mv.iter_mut() {
		for row in comp.class0_fp.iter_mut() {
			for p in row.iter_mut() {
				mv_update(r, p);
			}
		}
		for p in comp.fp.iter_mut() {
			mv_update(r, p);
		}
	}
	if allow_hp {
		for comp in fc.mv.iter_mut() {
			mv_update(r, &mut comp.class0_hp);
			mv_update(r, &mut comp.hp);
		}
	}
}

// --- Adaptation (vpx_dsp/prob.h) ---

fn get_prob(num: u32, den: u32) -> u8 {
	let p = ((u64::from(num) * 256 + u64::from(den >> 1)) / u64::from(den)) as i64;
	p.clamp(1, 255) as u8
}

fn weighted_prob(a: u8, b: u8, factor: u32) -> u8 {
	((u32::from(a) * (256 - factor) + u32::from(b) * factor + 128) >> 8) as u8
}

/// `merge_probs` with an explicit saturation and update factor.
fn merge_probs(pre: u8, ct: [u32; 2], count_sat: u32, max_factor: u32) -> u8 {
	let den = ct[0] + ct[1];
	let prob = if den == 0 { 128 } else { get_prob(ct[0], den) };
	let count = den.min(count_sat);
	weighted_prob(pre, prob, max_factor * count / count_sat)
}

const COUNT_TO_FACTOR: [u32; 21] =
	[0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128];

/// `mode_mv_merge_probs`.
fn mode_merge(pre: u8, ct: [u32; 2]) -> u8 {
	let den = ct[0] + ct[1];
	if den == 0 {
		return pre;
	}
	weighted_prob(pre, get_prob(ct[0], den), COUNT_TO_FACTOR[den.min(20) as usize])
}

/// `vpx_tree_merge_probs`: adapts every node of a tree; returns the count
/// under node `i`.
fn tree_merge(tree: &Tree, pre: &[u8], counts: &[u32], out: &mut [u8], i: usize) -> u32 {
	let side = |node: i8, out: &mut [u8]| -> u32 {
		if node <= 0 {
			counts[(-node) as usize]
		} else {
			tree_merge(tree, pre, counts, out, node as usize)
		}
	};
	let left = side(tree[i], out);
	let right = side(tree[i + 1], out);
	out[i >> 1] = mode_merge(pre[i >> 1], [left, right]);
	left + right
}

fn merge_tree(tree: &Tree, pre: &[u8], counts: &[u32], out: &mut [u8]) {
	tree_merge(tree, pre, counts, out, 0);
}

/// Coefficient adaptation (`vp9_adapt_coef_probs`). `key` is an intra
/// frame, `after_key` an inter frame following a key frame.
pub fn adapt_coef(
	fc: &mut FrameContext,
	pre: &FrameContext,
	counts: &Counts,
	key: bool,
	after_key: bool,
) {
	let (sat, factor) = if key {
		(24, 112)
	} else if after_key {
		(24, 128)
	} else {
		(24, 112)
	};
	for tx in 0..4 {
		for plane in 0..2 {
			for reference in 0..2 {
				for band in 0..6 {
					for ctx in 0..if band == 0 { 3 } else { 6 } {
						let c = &counts.coef[tx][plane][reference][band][ctx];
						let eob = counts.eob_branch[tx][plane][reference][band][ctx];
						let branch = [[c[3], eob.saturating_sub(c[3])], [c[0], c[1] + c[2]], [c[1], c[2]]];
						for (node, ct) in branch.into_iter().enumerate() {
							let p = pre.coef[tx][plane][reference][band][ctx][node];
							fc.coef[tx][plane][reference][band][ctx][node] = merge_probs(p, ct, sat, factor);
						}
					}
				}
			}
		}
	}
}

/// Mode adaptation (`vp9_adapt_mode_probs`).
pub fn adapt_modes(
	fc: &mut FrameContext,
	pre: &FrameContext,
	counts: &Counts,
	switchable_filter: bool,
	tx_select: bool,
) {
	for i in 0..4 {
		fc.intra_inter[i] = mode_merge(pre.intra_inter[i], counts.intra_inter[i]);
	}
	for i in 0..5 {
		fc.comp_inter[i] = mode_merge(pre.comp_inter[i], counts.comp_inter[i]);
		fc.comp_ref[i] = mode_merge(pre.comp_ref[i], counts.comp_ref[i]);
		for j in 0..2 {
			fc.single_ref[i][j] = mode_merge(pre.single_ref[i][j], counts.single_ref[i][j]);
		}
	}
	for i in 0..7 {
		// The tree's leaves are offsets from NEARESTMV: counts are indexed
		// the same way.
		merge_tree(&INTER_MODE_TREE, &pre.inter_mode[i], &counts.inter_mode[i], &mut fc.inter_mode[i]);
	}
	for i in 0..4 {
		merge_tree(&INTRA_MODE_TREE, &pre.y_mode[i], &counts.y_mode[i], &mut fc.y_mode[i]);
	}
	for i in 0..10 {
		merge_tree(&INTRA_MODE_TREE, &pre.uv_mode[i], &counts.uv_mode[i], &mut fc.uv_mode[i]);
	}
	for i in 0..16 {
		merge_tree(&PARTITION_TREE, &pre.partition[i], &counts.partition[i], &mut fc.partition[i]);
	}
	if switchable_filter {
		for i in 0..4 {
			merge_tree(
				&INTERP_FILTER_TREE,
				&pre.interp_filter[i],
				&counts.interp_filter[i],
				&mut fc.interp_filter[i],
			);
		}
	}
	if tx_select {
		for i in 0..2 {
			let c = counts.tx8[i];
			fc.tx8[i][0] = mode_merge(pre.tx8[i][0], [c[0], c[1]]);
			let c = counts.tx16[i];
			fc.tx16[i][0] = mode_merge(pre.tx16[i][0], [c[0], c[1] + c[2]]);
			fc.tx16[i][1] = mode_merge(pre.tx16[i][1], [c[1], c[2]]);
			let c = counts.tx32[i];
			fc.tx32[i][0] = mode_merge(pre.tx32[i][0], [c[0], c[1] + c[2] + c[3]]);
			fc.tx32[i][1] = mode_merge(pre.tx32[i][1], [c[1], c[2] + c[3]]);
			fc.tx32[i][2] = mode_merge(pre.tx32[i][2], [c[2], c[3]]);
		}
	}
	for i in 0..3 {
		fc.skip[i] = mode_merge(pre.skip[i], counts.skip[i]);
	}
}

/// Motion vector adaptation (`vp9_adapt_mv_probs`).
pub fn adapt_mv(fc: &mut FrameContext, pre: &FrameContext, counts: &Counts, allow_hp: bool) {
	merge_tree(&MV_JOINT_TREE, &pre.mv_joints, &counts.mv_joints, &mut fc.mv_joints);
	for i in 0..2 {
		let (p, c, out) = (&pre.mv[i], &counts.mv[i], &mut fc.mv[i]);
		out.sign = mode_merge(p.sign, c.sign);
		merge_tree(&MV_CLASS_TREE, &p.classes, &c.classes, &mut out.classes);
		merge_tree(&MV_CLASS0_TREE, &p.class0, &c.class0, &mut out.class0);
		for j in 0..10 {
			out.bits[j] = mode_merge(p.bits[j], c.bits[j]);
		}
		for j in 0..2 {
			merge_tree(&MV_FP_TREE, &p.class0_fp[j], &c.class0_fp[j], &mut out.class0_fp[j]);
		}
		merge_tree(&MV_FP_TREE, &p.fp, &c.fp, &mut out.fp);
		if allow_hp {
			out.class0_hp = mode_merge(p.class0_hp, c.class0_hp);
			out.hp = mode_merge(p.hp, c.hp);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn inverse_map_matches_libvpx() {
		assert_eq!(
			&INV_MAP[..21],
			&[
				7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254,
				1
			]
		);
		assert_eq!(&INV_MAP[250..], &[249, 250, 251, 252, 253, 253][1..]);
		assert_eq!(INV_MAP[26], 8);
	}

	#[test]
	fn mv_classes() {
		assert_eq!(mv_class(0), 0);
		assert_eq!(mv_class(15), 0);
		assert_eq!(mv_class(16), 1);
		assert_eq!(mv_class(31), 1);
		assert_eq!(mv_class(32), 2);
		assert_eq!(mv_class(8191), 9);
		assert_eq!(mv_class(8192), 10);
	}

	#[test]
	fn merging() {
		assert_eq!(get_prob(0, 10), 1);
		assert_eq!(get_prob(10, 10), 255);
		assert_eq!(mode_merge(100, [0, 0]), 100);
		// 20+ counts move the probability halfway.
		assert_eq!(mode_merge(100, [20, 20]), 114);
	}
}
