//! Macroblock layer writing (ISO/IEC 14496-10 §7.3.5) for both entropy
//! coders: the decoder's `h264::mb` parsing inverted, with the same
//! neighbour-dependent context derivations (§9.2.1 nC, §9.3.3.1.1
//! ctxIdxInc) computed from the same [`MbInfo`] records.
//!
//! A candidate macroblock is described by an [`MbSyn`] (the syntax
//! elements that are not already in its `MbInfo`) plus its coefficient
//! levels; [`MbWriter`] serialises it into CAVLC bits or CABAC bins. The
//! CABAC path is generic over [`Bins`], so the same code both prices a
//! candidate ([`super::cabac::CabacCost`]) and writes the chosen one.
//!
//! FFmpeg / x264: `x264/encoder/cabac.c` (`cabac_mb_type`,
//! `cabac_mvd`, `cabac_cbp_luma`, ...), `cavlc.c`.

use super::cabac::{self, Bins};
use super::vlc::{cbp_code, write_block};
use crate::codecs::bitstream::BitWriter;
use crate::codecs::h264::cabac::{BlockCat, ctx};
use crate::codecs::h264::consts::{BLK_XY, ZIGZAG_4X4, ZIGZAG_8X8};
use crate::codecs::h264::mb::{MbInfo, MbKind, Pred, Residual, Shape};

/// Slice type of the picture being coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
	I,
	P,
	B,
}

/// `(shape, pred0, pred1)` for B mb_type 1..=21 (Table 7-14), as in the
/// decoder.
pub const B_TYPES: [(Shape, Pred, Pred); 21] = {
	use Pred::*;
	use Shape::*;
	[
		(S16x16, L0, L0),
		(S16x16, L1, L1),
		(S16x16, Bi, Bi),
		(S16x8, L0, L0),
		(S8x16, L0, L0),
		(S16x8, L1, L1),
		(S8x16, L1, L1),
		(S16x8, L0, L1),
		(S8x16, L0, L1),
		(S16x8, L1, L0),
		(S8x16, L1, L0),
		(S16x8, L0, Bi),
		(S8x16, L0, Bi),
		(S16x8, L1, Bi),
		(S8x16, L1, Bi),
		(S16x8, Bi, L0),
		(S8x16, Bi, L0),
		(S16x8, Bi, L1),
		(S8x16, Bi, L1),
		(S16x8, Bi, Bi),
		(S8x16, Bi, Bi),
	]
};

/// B mb_type for a shape and the predictions of its partitions.
pub fn b_mb_type(shape: Shape, p0: Pred, p1: Pred) -> u32 {
	let p1 = if shape == Shape::S16x16 { p0 } else { p1 };
	B_TYPES.iter().position(|&(s, a, b)| s == shape && a == p0 && b == p1).map_or(1, |i| i as u32 + 1)
}

/// A partition as (x4, y4, w4, h4) in 4x4 units.
pub type Part = (usize, usize, usize, usize);

/// Partitions of a shape.
pub fn parts(shape: Shape) -> &'static [Part] {
	match shape {
		Shape::S16x16 => &[(0, 0, 4, 4)],
		Shape::S16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
		Shape::S8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
		Shape::S8x8 => &[(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
	}
}

/// The macroblock type as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynType {
	PSkip,
	BSkip,
	/// I_NxN with 4x4 transform.
	I4x4,
	/// I_NxN with transform_size_8x8_flag.
	I8x8,
	I16x16 {
		pred: u8,
	},
	/// P_L0_16x16 / 16x8 / 8x16 / P_8x8 (sub-type P_L0_8x8 only).
	P(Shape),
	/// B mb_type 1..=21.
	B(u32),
	BDirect,
}

/// The syntax elements of one macroblock beyond its `MbInfo`.
#[derive(Debug, Clone, Copy)]
pub struct MbSyn {
	pub t: SynType,
	/// rem_intra_pred_mode per 4x4 (or 8x8) block in decoding order, or
	/// -1 when the predicted mode is used.
	pub intra_rem: [i8; 16],
	/// ref_idx per list and partition (per 8x8 block for 8x8 types).
	pub ref_idx: [[i8; 4]; 2],
	/// mvd per list and partition.
	pub mvd: [[[i32; 2]; 4]; 2],
}

impl MbSyn {
	pub fn new(t: SynType) -> Self {
		Self { t, intra_rem: [-1; 16], ref_idx: [[0; 4]; 2], mvd: [[[0; 2]; 4]; 2] }
	}

	fn is_intra(&self) -> bool {
		matches!(self.t, SynType::I4x4 | SynType::I8x8 | SynType::I16x16 { .. })
	}

	/// Partitions and their predictions for the non-8x8 inter types.
	fn inter_parts(&self) -> (&'static [Part], [Pred; 2]) {
		match self.t {
			SynType::P(shape) => (parts(shape), [Pred::L0; 2]),
			SynType::B(t) => {
				let (s, a, b) = B_TYPES[(t as usize).clamp(1, 21) - 1];
				(parts(s), [a, b])
			}
			_ => (&[], [Pred::L0; 2]),
		}
	}
}

/// Neighbour access over the macroblock records of the picture being
/// coded (single slice, progressive), as `SliceDecoder::nb_mb` / `nb4`.
#[derive(Clone, Copy)]
pub struct Nbr<'a> {
	pub mbs: &'a [MbInfo],
	pub mb_w: usize,
	pub mb_h: usize,
	pub mbx: usize,
	pub mby: usize,
}

impl<'a> Nbr<'a> {
	#[inline]
	pub fn nb_mb(&self, dx: i32, dy: i32) -> Option<&'a MbInfo> {
		let x = self.mbx as i32 + dx;
		let y = self.mby as i32 + dy;
		if x < 0 || y < 0 || x >= self.mb_w as i32 || y >= self.mb_h as i32 {
			return None;
		}
		let mb = &self.mbs[y as usize * self.mb_w + x as usize];
		(mb.slice != 0).then_some(mb)
	}

	#[inline]
	pub fn nb4<'b>(&self, cur: &'b MbInfo, x4: i32, y4: i32) -> Option<(&'b MbInfo, usize)>
	where
		'a: 'b,
	{
		let (dx, dy) = (x4.div_euclid(4), y4.div_euclid(4));
		let idx = (y4.rem_euclid(4) * 4 + x4.rem_euclid(4)) as usize;
		if dx == 0 && dy == 0 {
			return Some((cur, idx));
		}
		self.nb_mb(dx, dy).map(|m| (m, idx))
	}

	pub fn nb_chroma<'b>(
		&self,
		cur: &'b MbInfo,
		comp: usize,
		x2: i32,
		y2: i32,
	) -> Option<(&'b MbInfo, usize)>
	where
		'a: 'b,
	{
		let (dx, dy) = (x2.div_euclid(2), y2.div_euclid(2));
		let idx = 16 + 8 * comp + (y2.rem_euclid(2) * 2 + x2.rem_euclid(2)) as usize;
		if dx == 0 && dy == 0 {
			return Some((cur, idx));
		}
		self.nb_mb(dx, dy).map(|m| (m, idx))
	}

	pub fn nc_luma(&self, cur: &MbInfo, x4: i32, y4: i32) -> i32 {
		let get = |n: Option<(&MbInfo, usize)>| n.map(|(m, i)| i32::from(m.nnz[i]));
		combine_nc(get(self.nb4(cur, x4 - 1, y4)), get(self.nb4(cur, x4, y4 - 1)))
	}

	pub fn nc_chroma(&self, cur: &MbInfo, comp: usize, x2: i32, y2: i32) -> i32 {
		let get = |n: Option<(&MbInfo, usize)>| n.map(|(m, i)| i32::from(m.nnz[i]));
		combine_nc(
			get(self.nb_chroma(cur, comp, x2 - 1, y2)),
			get(self.nb_chroma(cur, comp, x2, y2 - 1)),
		)
	}
}

fn combine_nc(a: Option<i32>, b: Option<i32>) -> i32 {
	match (a, b) {
		(Some(a), Some(b)) => (a + b + 1) >> 1,
		(Some(a), None) => a,
		(None, Some(b)) => b,
		(None, None) => 0,
	}
}

#[inline]
fn b8_of(idx: usize) -> usize {
	(idx / 8) * 2 + (idx % 4) / 2
}

/// A raster 4x4 block's levels from `start` in zig-zag order.
fn scan4(b: &[i32; 16], start: usize) -> ([i32; 16], usize) {
	let mut out = [0i32; 16];
	for k in start..16 {
		out[k - start] = b[usize::from(ZIGZAG_4X4[k])];
	}
	(out, 16 - start)
}

fn scan8(b: &[i32; 64]) -> [i32; 64] {
	std::array::from_fn(|k| b[usize::from(ZIGZAG_8X8[k])])
}

/// Writes macroblocks of one slice.
#[derive(Clone, Copy)]
pub struct MbWriter<'a> {
	pub nb: Nbr<'a>,
	pub slice: SliceKind,
	/// num_ref_idx_lX_active.
	pub num_ref: [usize; 2],
	pub pps_8x8: bool,
}

impl<'a> MbWriter<'a> {
	// -------------------------------------------------------------------
	// CAVLC

	/// `macroblock_layer()` in CAVLC (the skip run is the caller's).
	pub fn cavlc(&self, w: &mut BitWriter, info: &MbInfo, syn: &MbSyn, res: &Residual) {
		let off = match self.slice {
			SliceKind::I => 0,
			SliceKind::P => 5,
			SliceKind::B => 23,
		};
		match syn.t {
			SynType::PSkip | SynType::BSkip => return,
			SynType::I4x4 | SynType::I8x8 => {
				w.ue(off);
				if self.pps_8x8 {
					w.bit(syn.t == SynType::I8x8);
				}
				let n = if syn.t == SynType::I8x8 { 4 } else { 16 };
				for &rem in &syn.intra_rem[..n] {
					if rem < 0 {
						w.bit(true);
					} else {
						w.bit(false);
						w.put(rem as u32, 3);
					}
				}
				w.ue(u32::from(info.chroma_pred));
			}
			SynType::I16x16 { pred } => {
				w.ue(off + i16_type(pred, info.cbp));
				w.ue(u32::from(info.chroma_pred));
			}
			SynType::P(shape) => {
				w.ue(match shape {
					Shape::S16x16 => 0,
					Shape::S16x8 => 1,
					Shape::S8x16 => 2,
					Shape::S8x8 => 3,
				});
				if shape == Shape::S8x8 {
					for _ in 0..4 {
						w.ue(0);
					}
					self.cavlc_sub_motion(w, syn);
				} else {
					self.cavlc_motion(w, syn);
				}
			}
			SynType::B(t) => {
				w.ue(t);
				self.cavlc_motion(w, syn);
			}
			SynType::BDirect => w.ue(0),
		}
		if !matches!(syn.t, SynType::I16x16 { .. }) {
			w.ue(cbp_code(info.cbp, syn.is_intra()));
			if info.cbp & 15 != 0 && self.inter_8x8_flag(syn) {
				w.bit(info.t8x8);
			}
		}
		if info.cbp != 0 || matches!(syn.t, SynType::I16x16 { .. }) {
			w.se(0); // mb_qp_delta
			self.cavlc_residual(w, info, res);
		}
	}

	/// Whether an inter macroblock carries transform_size_8x8_flag.
	fn inter_8x8_flag(&self, syn: &MbSyn) -> bool {
		self.pps_8x8
			&& match syn.t {
				// P_8x8 uses only 8x8 sub-partitions.
				SynType::P(_) | SynType::B(_) | SynType::BDirect => true,
				_ => false,
			}
	}

	fn cavlc_ref(&self, w: &mut BitWriter, list: usize, r: i8) {
		match self.num_ref[list] {
			0 | 1 => {}
			2 => w.bit(r == 0),
			_ => w.ue(r as u32),
		}
	}

	fn cavlc_motion(&self, w: &mut BitWriter, syn: &MbSyn) {
		let (ps, pred) = syn.inter_parts();
		for list in 0..2 {
			for p in 0..ps.len() {
				if pred[p].uses(list) {
					self.cavlc_ref(w, list, syn.ref_idx[list][p]);
				}
			}
		}
		for list in 0..2 {
			for p in 0..ps.len() {
				if pred[p].uses(list) {
					w.se(syn.mvd[list][p][0]);
					w.se(syn.mvd[list][p][1]);
				}
			}
		}
	}

	/// P_8x8 (sub-type P_L0_8x8): list 0 only.
	fn cavlc_sub_motion(&self, w: &mut BitWriter, syn: &MbSyn) {
		for b8 in 0..4 {
			self.cavlc_ref(w, 0, syn.ref_idx[0][b8]);
		}
		for b8 in 0..4 {
			w.se(syn.mvd[0][b8][0]);
			w.se(syn.mvd[0][b8][1]);
		}
	}

	fn cavlc_residual(&self, w: &mut BitWriter, info: &MbInfo, res: &Residual) {
		let nb = &self.nb;
		let i16 = info.kind == MbKind::I16x16;
		if i16 {
			let nc = nb.nc_luma(info, 0, 0);
			let (l, n) = scan4(&res.dc, 0);
			write_block(w, nc, &l[..n]);
		}
		for b8 in 0..4usize {
			if info.cbp & (1 << b8) == 0 {
				continue;
			}
			for i in 0..4 {
				let (x4, y4) = (usize::from(BLK_XY[b8 * 4 + i].0), usize::from(BLK_XY[b8 * 4 + i].1));
				let raster = y4 * 4 + x4;
				let nc = nb.nc_luma(info, x4 as i32, y4 as i32);
				if info.t8x8 {
					let s = scan8(&res.luma8[b8]);
					let sub: [i32; 16] = std::array::from_fn(|k| s[4 * k + i]);
					write_block(w, nc, &sub);
				} else {
					let (l, n) = scan4(&res.luma[raster], usize::from(i16));
					write_block(w, nc, &l[..n]);
				}
			}
		}
		let cbp_c = info.cbp >> 4;
		if cbp_c != 0 {
			for c in 0..2 {
				write_block(w, -1, &res.chroma_dc[c][..4]);
			}
		}
		if cbp_c == 2 {
			for c in 0..2 {
				for blk in 0..4 {
					let nc = nb.nc_chroma(info, c, (blk % 2) as i32, (blk / 2) as i32);
					let (l, n) = scan4(&res.chroma_ac[c][blk], 1);
					write_block(w, nc, &l[..n]);
				}
			}
		}
	}

	// -------------------------------------------------------------------
	// CABAC

	/// mb_skip_flag (P and B slices) and, unless skipped, the
	/// `macroblock_layer()`. end_of_slice_flag is the caller's.
	pub fn cabac<B: Bins>(&self, b: &mut B, info: &MbInfo, syn: &MbSyn, res: &Residual) {
		let nb = &self.nb;
		if self.slice != SliceKind::I {
			let cond = |m: Option<&MbInfo>| m.is_some_and(|m| !m.is_skip()) as usize;
			let inc = cond(nb.nb_mb(-1, 0)) + cond(nb.nb_mb(0, -1));
			let base = if self.slice == SliceKind::B { ctx::MB_SKIP_B } else { ctx::MB_SKIP_P };
			let skip = matches!(syn.t, SynType::PSkip | SynType::BSkip);
			b.decision(base + inc, skip);
			if skip {
				return;
			}
		}
		self.cabac_mb_type(b, info, syn);
		match syn.t {
			SynType::I4x4 | SynType::I8x8 => {
				if self.pps_8x8 {
					self.cabac_t8x8(b, syn.t == SynType::I8x8);
				}
				let n = if syn.t == SynType::I8x8 { 4 } else { 16 };
				for &rem in &syn.intra_rem[..n] {
					b.decision(ctx::PREV_INTRA, rem < 0);
					if rem >= 0 {
						for i in 0..3 {
							b.decision(ctx::REM_INTRA, (rem >> i) & 1 != 0);
						}
					}
				}
				self.cabac_chroma_pred(b, info.chroma_pred);
			}
			SynType::I16x16 { .. } => self.cabac_chroma_pred(b, info.chroma_pred),
			SynType::P(Shape::S8x8) => {
				for _ in 0..4 {
					b.decision(ctx::SUB_MB_P, true); // P_L0_8x8
				}
				self.cabac_sub_motion(b, info, syn);
			}
			SynType::P(_) | SynType::B(_) => self.cabac_motion(b, info, syn),
			_ => {}
		}
		if !matches!(syn.t, SynType::I16x16 { .. }) {
			self.cabac_cbp(b, info);
			if info.cbp & 15 != 0 && self.inter_8x8_flag(syn) {
				self.cabac_t8x8(b, info.t8x8);
			}
		}
		if info.cbp != 0 || matches!(syn.t, SynType::I16x16 { .. }) {
			// mb_qp_delta = 0; the previous macroblock's delta is always 0.
			b.decision(ctx::QP_DELTA, false);
			self.cabac_residual(b, info, res);
		}
	}

	fn cabac_t8x8<B: Bins>(&self, b: &mut B, flag: bool) {
		let cond = |m: Option<&MbInfo>| m.is_some_and(|m| m.t8x8) as usize;
		let inc = cond(self.nb.nb_mb(-1, 0)) + cond(self.nb.nb_mb(0, -1));
		b.decision(ctx::TRANSFORM_8X8 + inc, flag);
	}

	fn cabac_chroma_pred<B: Bins>(&self, b: &mut B, v: u8) {
		let cond = |m: Option<&MbInfo>| {
			m.is_some_and(|m| m.is_intra() && m.kind != MbKind::Pcm && m.chroma_pred != 0) as usize
		};
		let inc = cond(self.nb.nb_mb(-1, 0)) + cond(self.nb.nb_mb(0, -1));
		b.decision(ctx::CHROMA_PRED + inc, v > 0);
		let mut k = 1;
		while v >= k && k < 3 {
			b.decision(ctx::CHROMA_PRED + 3, v > k);
			k += 1;
		}
	}

	fn cabac_intra_type<B: Bins>(
		b: &mut B,
		base: usize,
		inc: usize,
		intra_slice: bool,
		t: SynType,
		cbp: u8,
	) {
		let SynType::I16x16 { pred } = t else {
			b.decision(base + inc, false);
			return;
		};
		b.decision(base + inc, true);
		b.terminate(false); // not I_PCM
		let s = if intra_slice { base + 2 } else { base };
		let i = usize::from(intra_slice);
		b.decision(s + 1, cbp & 15 != 0);
		let chroma = cbp >> 4;
		b.decision(s + 2, chroma != 0);
		if chroma != 0 {
			b.decision(s + 2 + i, chroma == 2);
		}
		b.decision(s + 3 + i, pred >> 1 != 0);
		b.decision(s + 3 + 2 * i, pred & 1 != 0);
	}

	fn cabac_mb_type<B: Bins>(&self, b: &mut B, info: &MbInfo, syn: &MbSyn) {
		let nb = &self.nb;
		match self.slice {
			SliceKind::I => {
				let cond = |m: Option<&MbInfo>| {
					m.is_some_and(|m| !matches!(m.kind, MbKind::I4x4 | MbKind::I8x8)) as usize
				};
				let inc = cond(nb.nb_mb(-1, 0)) + cond(nb.nb_mb(0, -1));
				Self::cabac_intra_type(b, ctx::MB_TYPE_I, inc, true, syn.t, info.cbp);
			}
			SliceKind::P => {
				if syn.is_intra() {
					b.decision(ctx::MB_TYPE_P, true);
					Self::cabac_intra_type(b, ctx::MB_TYPE_P_INTRA, 0, false, syn.t, info.cbp);
					return;
				}
				b.decision(ctx::MB_TYPE_P, false);
				let SynType::P(shape) = syn.t else { return };
				match shape {
					Shape::S16x16 => {
						b.decision(ctx::MB_TYPE_P + 1, false);
						b.decision(ctx::MB_TYPE_P + 2, false);
					}
					Shape::S8x8 => {
						b.decision(ctx::MB_TYPE_P + 1, false);
						b.decision(ctx::MB_TYPE_P + 2, true);
					}
					Shape::S16x8 => {
						b.decision(ctx::MB_TYPE_P + 1, true);
						b.decision(ctx::MB_TYPE_P + 3, true);
					}
					Shape::S8x16 => {
						b.decision(ctx::MB_TYPE_P + 1, true);
						b.decision(ctx::MB_TYPE_P + 3, false);
					}
				}
			}
			SliceKind::B => {
				let cond = |m: Option<&MbInfo>| m.is_some_and(|m| !m.direct16) as usize;
				let inc = cond(nb.nb_mb(-1, 0)) + cond(nb.nb_mb(0, -1));
				let base = ctx::MB_TYPE_B;
				if syn.t == SynType::BDirect {
					b.decision(base + inc, false);
					return;
				}
				b.decision(base + inc, true);
				let t = match syn.t {
					SynType::B(t) => t,
					_ => 0, // intra
				};
				if t == 1 || t == 2 {
					b.decision(base + 3, false);
					b.decision(base + 5, t == 2);
					return;
				}
				b.decision(base + 3, true);
				let (bits, extra) = match t {
					0 => (13, None),
					3..=10 => (t - 3, None),
					11 => (14, None),
					22 => (15, None),
					_ => ((t + 4) >> 1, Some((t + 4) & 1 != 0)),
				};
				b.decision(base + 4, bits & 8 != 0);
				b.decision(base + 5, bits & 4 != 0);
				b.decision(base + 5, bits & 2 != 0);
				b.decision(base + 5, bits & 1 != 0);
				if let Some(e) = extra {
					b.decision(base + 5, e);
				}
				if t == 0 {
					Self::cabac_intra_type(b, ctx::MB_TYPE_B_INTRA, 0, false, syn.t, info.cbp);
				}
			}
		}
	}

	fn cabac_ref<B: Bins>(&self, b: &mut B, cur: &MbInfo, list: usize, x4: usize, y4: usize, r: i8) {
		if self.num_ref[list] <= 1 {
			return;
		}
		let cond = |n: Option<(&MbInfo, usize)>| -> usize {
			let Some((m, idx)) = n else { return 0 };
			let b8 = b8_of(idx);
			if m.is_intra() || m.direct8 & (1 << b8) != 0 {
				return 0;
			}
			usize::from(m.ref_idx[list][b8] > 0)
		};
		let inc = cond(self.nb.nb4(cur, x4 as i32 - 1, y4 as i32))
			+ 2 * cond(self.nb.nb4(cur, x4 as i32, y4 as i32 - 1));
		let v = r as u32;
		b.decision(ctx::REF_IDX + inc, v > 0);
		if v == 0 {
			return;
		}
		b.decision(ctx::REF_IDX + 4, v > 1);
		if v == 1 {
			return;
		}
		for _ in 2..v {
			b.decision(ctx::REF_IDX + 5, true);
		}
		if v < 32 {
			b.decision(ctx::REF_IDX + 5, false);
		}
	}

	#[allow(clippy::too_many_arguments)]
	fn cabac_mvd<B: Bins>(
		&self,
		b: &mut B,
		cur: &MbInfo,
		list: usize,
		x4: usize,
		y4: usize,
		mvd: [i32; 2],
	) {
		for comp in 0..2 {
			let get =
				|n: Option<(&MbInfo, usize)>| n.map_or(0u32, |(m, i)| u32::from(m.mvd[list][i][comp]));
			let sum = get(self.nb.nb4(cur, x4 as i32 - 1, y4 as i32))
				+ get(self.nb.nb4(cur, x4 as i32, y4 as i32 - 1));
			let inc = if sum < 3 {
				0
			} else if sum > 32 {
				2
			} else {
				1
			};
			let base = if comp == 0 { ctx::MVD_X } else { ctx::MVD_Y };
			let d = mvd[comp];
			let v = d.unsigned_abs();
			b.decision(base + inc, v > 0);
			if v == 0 {
				continue;
			}
			const INCS: [usize; 8] = [3, 4, 5, 6, 6, 6, 6, 6];
			for j in 1..v.min(9) {
				b.decision(base + INCS[(j - 1) as usize], true);
			}
			if v < 9 {
				b.decision(base + INCS[(v - 1) as usize], false);
			} else {
				cabac::exp_golomb_bypass(b, v - 9, 3);
			}
			b.bypass(d < 0);
		}
	}

	/// ref_idx and mvd of a 1- or 2-partition macroblock. The neighbour
	/// derivations read `cur`, which already holds every partition's
	/// final motion: a partition's left and upper neighbours always
	/// precede it, so what they see equals what a decoder has parsed.
	fn cabac_motion<B: Bins>(&self, b: &mut B, cur: &MbInfo, syn: &MbSyn) {
		let (ps, pred) = syn.inter_parts();
		for list in 0..2 {
			for (p, &(x4, y4, _, _)) in ps.iter().enumerate() {
				if pred[p].uses(list) {
					self.cabac_ref(b, cur, list, x4, y4, syn.ref_idx[list][p]);
				}
			}
		}
		for list in 0..2 {
			for (p, &(x4, y4, _, _)) in ps.iter().enumerate() {
				if pred[p].uses(list) {
					self.cabac_mvd(b, cur, list, x4, y4, syn.mvd[list][p]);
				}
			}
		}
	}

	/// P_8x8 (sub-type P_L0_8x8): list 0 only.
	fn cabac_sub_motion<B: Bins>(&self, b: &mut B, cur: &MbInfo, syn: &MbSyn) {
		for b8 in 0..4 {
			self.cabac_ref(b, cur, 0, (b8 % 2) * 2, (b8 / 2) * 2, syn.ref_idx[0][b8]);
		}
		for b8 in 0..4 {
			self.cabac_mvd(b, cur, 0, (b8 % 2) * 2, (b8 / 2) * 2, syn.mvd[0][b8]);
		}
	}

	fn cabac_cbp<B: Bins>(&self, b: &mut B, cur: &MbInfo) {
		let nb = &self.nb;
		let mut cbp = 0u8;
		for b8 in 0..4usize {
			let (x4, y4) = ((b8 % 2) as i32 * 2, (b8 / 2) as i32 * 2);
			let cond = |n: Option<(&MbInfo, usize)>, same: bool| -> usize {
				let Some((m, idx)) = n else { return 0 };
				if m.kind == MbKind::Pcm {
					return 0;
				}
				let bits = if same { cbp } else { m.cbp };
				if !m.is_skip() && bits & (1 << b8_of(idx)) != 0 { 0 } else { 1 }
			};
			let inc = cond(nb.nb4(cur, x4 - 1, y4), x4 > 0) + 2 * cond(nb.nb4(cur, x4, y4 - 1), y4 > 0);
			let bit = cur.cbp & (1 << b8) != 0;
			b.decision(ctx::CBP_LUMA + inc, bit);
			if bit {
				cbp |= 1 << b8;
			}
		}
		let cond = |m: Option<&MbInfo>, need: u8| -> usize {
			m.is_some_and(|m| m.kind == MbKind::Pcm || (!m.is_skip() && (m.cbp >> 4) >= need)) as usize
		};
		let (a, bb) = (nb.nb_mb(-1, 0), nb.nb_mb(0, -1));
		let chroma = cur.cbp >> 4;
		b.decision(ctx::CBP_CHROMA + cond(a, 1) + 2 * cond(bb, 1), chroma != 0);
		if chroma != 0 {
			b.decision(ctx::CBP_CHROMA + cond(a, 2) + 2 * cond(bb, 2) + 4, chroma == 2);
		}
	}

	/// coded_block_flag ctxIdxInc for a luma 4x4 block.
	pub fn cbf_luma_inc(&self, cur: &MbInfo, x4: i32, y4: i32) -> usize {
		let intra = cur.is_intra();
		let cond = |n: Option<(&MbInfo, usize)>| -> usize {
			match n {
				None => usize::from(intra),
				Some((m, idx)) => {
					if m.kind == MbKind::Pcm {
						1
					} else if m.is_skip() {
						0
					} else {
						usize::from(m.cbp & (1 << b8_of(idx)) != 0 && m.nnz[idx] != 0)
					}
				}
			}
		};
		cond(self.nb.nb4(cur, x4 - 1, y4)) + 2 * cond(self.nb.nb4(cur, x4, y4 - 1))
	}

	pub fn cbf_dc_inc(&self, cur: &MbInfo, comp: usize) -> usize {
		let intra = cur.is_intra();
		let cond = |m: Option<&MbInfo>| -> usize {
			match m {
				None => usize::from(intra),
				Some(m) if m.kind == MbKind::Pcm => 1,
				Some(m) if m.is_skip() => 0,
				Some(m) => {
					if comp == 0 {
						usize::from(m.kind == MbKind::I16x16 && m.dc_cbf & 1 != 0)
					} else {
						usize::from(m.cbp >> 4 != 0 && m.dc_cbf & (1 << comp) != 0)
					}
				}
			}
		};
		cond(self.nb.nb_mb(-1, 0)) + 2 * cond(self.nb.nb_mb(0, -1))
	}

	pub fn cbf_chroma_ac_inc(&self, cur: &MbInfo, comp: usize, x2: i32, y2: i32) -> usize {
		let intra = cur.is_intra();
		let cond = |n: Option<(&MbInfo, usize)>| -> usize {
			match n {
				None => usize::from(intra),
				Some((m, _)) if m.kind == MbKind::Pcm => 1,
				Some((m, _)) if m.is_skip() => 0,
				Some((m, idx)) => usize::from(m.cbp >> 4 == 2 && m.nnz[idx] != 0),
			}
		};
		cond(self.nb.nb_chroma(cur, comp, x2 - 1, y2))
			+ 2 * cond(self.nb.nb_chroma(cur, comp, x2, y2 - 1))
	}

	fn cabac_residual<B: Bins>(&self, b: &mut B, info: &MbInfo, res: &Residual) {
		let i16 = info.kind == MbKind::I16x16;
		if i16 {
			let (l, n) = scan4(&res.dc, 0);
			cabac::residual(b, BlockCat::LumaDc, Some(self.cbf_dc_inc(info, 0)), &l[..n]);
		}
		for b8 in 0..4usize {
			if info.cbp & (1 << b8) == 0 {
				continue;
			}
			if info.t8x8 {
				cabac::residual(b, BlockCat::Luma8x8, None, &scan8(&res.luma8[b8]));
				continue;
			}
			for i in 0..4 {
				let (x4, y4) = (usize::from(BLK_XY[b8 * 4 + i].0), usize::from(BLK_XY[b8 * 4 + i].1));
				let raster = y4 * 4 + x4;
				let inc = self.cbf_luma_inc(info, x4 as i32, y4 as i32);
				let (l, n) = scan4(&res.luma[raster], usize::from(i16));
				let cat = if i16 { BlockCat::LumaAc } else { BlockCat::Luma4x4 };
				cabac::residual(b, cat, Some(inc), &l[..n]);
			}
		}
		let cbp_c = info.cbp >> 4;
		if cbp_c != 0 {
			for c in 0..2 {
				let inc = self.cbf_dc_inc(info, c + 1);
				cabac::residual(b, BlockCat::ChromaDc, Some(inc), &res.chroma_dc[c][..4]);
			}
		}
		if cbp_c == 2 {
			for c in 0..2 {
				for blk in 0..4 {
					let inc = self.cbf_chroma_ac_inc(info, c, (blk % 2) as i32, (blk / 2) as i32);
					let (l, n) = scan4(&res.chroma_ac[c][blk], 1);
					cabac::residual(b, BlockCat::ChromaAc, Some(inc), &l[..n]);
				}
			}
		}
	}
}

/// Intra_16x16 mb_type (1..=24) for a prediction mode and cbp.
pub fn i16_type(pred: u8, cbp: u8) -> u32 {
	1 + u32::from(pred) + 4 * u32::from(cbp >> 4) + if cbp & 15 != 0 { 12 } else { 0 }
}
