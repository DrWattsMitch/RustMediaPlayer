//! Motion vector prediction (§8.4.1.3), P_Skip (§8.4.1.1), spatial and
//! temporal direct prediction (§8.4.1.2) and motion-compensated
//! prediction with weighting (§8.4.2). FFmpeg: `h264_mvpred.h`,
//! `h264_direct.c`, `h264_mb.c` (`mc_part`).

use super::decode::SliceDecoder;
use super::mb::{BadMb, MbInfo, MbKind, MbResult, MbType, Pred, Shape, blk_index};
use super::recon::{self, Weighting};
use super::slice::SliceType;

/// Motion data of one neighbouring partition.
#[derive(Debug, Clone, Copy)]
struct Nb {
	avail: bool,
	ref_idx: i8,
	mv: [i16; 2],
}

const UNAVAILABLE: Nb = Nb { avail: false, ref_idx: -1, mv: [0, 0] };

fn median(a: i16, b: i16, c: i16) -> i16 {
	a.max(b).min(a.min(b).max(c))
}

fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
	v.clamp(lo, hi)
}

/// Raster 4x4 index → 8x8 block index.
#[inline]
fn b8_of(idx: usize) -> usize {
	(idx / 8) * 2 + (idx % 4) / 2
}

impl<'a> SliceDecoder<'a> {
	/// Neighbour partition at 4x4 position (x, y) relative to the current
	/// macroblock, seen from the partition starting at (px, py).
	fn mv_nb(&self, cur: &MbInfo, list: usize, x: i32, y: i32, px: usize, py: usize) -> Nb {
		if (0..4).contains(&x)
			&& (0..4).contains(&y)
			&& blk_index(x as usize, y as usize) >= blk_index(px, py)
		{
			return UNAVAILABLE; // not yet decoded
		}
		if x >= 4 && y >= 0 {
			return UNAVAILABLE;
		}
		match self.nb4(cur, x, y) {
			None => UNAVAILABLE,
			Some((m, idx)) => {
				if m.is_intra() {
					return Nb { avail: true, ref_idx: -1, mv: [0, 0] };
				}
				let r = m.ref_idx[list][b8_of(idx)];
				if r < 0 {
					Nb { avail: true, ref_idx: -1, mv: [0, 0] }
				} else {
					Nb { avail: true, ref_idx: r, mv: m.mv[list][idx] }
				}
			}
		}
	}

	/// Luma motion vector prediction (8.4.1.3) for a partition at (x4, y4)
	/// of size w4 x h4 (4x4 units).
	#[allow(clippy::too_many_arguments)]
	pub fn predict_mv(
		&self,
		cur: &MbInfo,
		list: usize,
		x4: usize,
		y4: usize,
		w4: usize,
		h4: usize,
		ref_idx: i8,
		shape: Shape,
		part: usize,
	) -> [i16; 2] {
		let (x, y) = (x4 as i32, y4 as i32);
		let a = self.mv_nb(cur, list, x - 1, y, x4, y4);
		let b = self.mv_nb(cur, list, x, y - 1, x4, y4);
		let mut c = self.mv_nb(cur, list, x + w4 as i32, y - 1, x4, y4);
		if !c.avail {
			c = self.mv_nb(cur, list, x - 1, y - 1, x4, y4);
		}
		match (shape, part) {
			(Shape::S16x8, 0) if b.ref_idx == ref_idx => return b.mv,
			(Shape::S16x8, 1) if a.ref_idx == ref_idx => return a.mv,
			(Shape::S8x16, 0) if a.ref_idx == ref_idx => return a.mv,
			(Shape::S8x16, 1) if c.ref_idx == ref_idx => return c.mv,
			_ => {}
		}
		let _ = h4;
		let (b, c) = if !b.avail && !c.avail && a.avail { (a, a) } else { (b, c) };
		let matches = [a.ref_idx == ref_idx, b.ref_idx == ref_idx, c.ref_idx == ref_idx];
		match matches {
			[true, false, false] => a.mv,
			[false, true, false] => b.mv,
			[false, false, true] => c.mv,
			_ => [median(a.mv[0], b.mv[0], c.mv[0]), median(a.mv[1], b.mv[1], c.mv[1])],
		}
	}

	/// P_Skip motion (8.4.1.1).
	pub fn predict_p_skip(&self, cur: &mut MbInfo) {
		let a = self.mv_nb(cur, 0, -1, 0, 0, 0);
		let b = self.mv_nb(cur, 0, 0, -1, 0, 0);
		let zero = !a.avail
			|| !b.avail
			|| (a.ref_idx == 0 && a.mv == [0, 0])
			|| (b.ref_idx == 0 && b.mv == [0, 0]);
		let mv = if zero { [0, 0] } else { self.predict_mv(cur, 0, 0, 0, 4, 4, 0, Shape::S16x16, 0) };
		cur.ref_idx = [[0; 4], [-1; 4]];
		cur.mv[0] = [mv; 16];
		cur.mv[1] = [[0; 2]; 16];
	}

	/// Direct prediction (spatial or temporal) for the 8x8 blocks in `mask`.
	pub fn predict_direct(&self, cur: &mut MbInfo, mask: u8) -> MbResult<()> {
		let col_pic = self.lists[1]
			.first()
			.and_then(|r| r.as_ref())
			.ok_or(BadMb("direct mode without RefPicList1[0]"))?;
		let addr = self.mby * self.f.mb_w + self.mbx;
		let col = col_pic.pic.motion.get(addr).copied().unwrap_or_default();
		let inference = self.sps.direct_8x8_inference;
		if self.h.direct_spatial {
			let mut refs = [-1i8; 2];
			let mut mvp = [[0i16; 2]; 2];
			for (list, r) in refs.iter_mut().enumerate() {
				let a = self.mv_nb(cur, list, -1, 0, 0, 0);
				let b = self.mv_nb(cur, list, 0, -1, 0, 0);
				let mut c = self.mv_nb(cur, list, 4, -1, 0, 0);
				if !c.avail {
					c = self.mv_nb(cur, list, -1, -1, 0, 0);
				}
				let min_pos = |x: i8, y: i8| if x >= 0 && y >= 0 { x.min(y) } else { x.max(y) };
				*r = min_pos(a.ref_idx, min_pos(b.ref_idx, c.ref_idx));
			}
			let direct_zero = refs[0] < 0 && refs[1] < 0;
			if direct_zero {
				refs = [0, 0];
			} else {
				for list in 0..2 {
					if refs[list] >= 0 {
						mvp[list] = self.predict_mv(cur, list, 0, 0, 4, 4, refs[list], Shape::S16x16, 0);
					}
				}
			}
			let col_short = !col_pic.long;
			for b8 in 0..4 {
				if mask & (1 << b8) == 0 {
					continue;
				}
				let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
				for list in 0..2 {
					cur.ref_idx[list][b8] = if refs[list] >= 0 { refs[list] } else { -1 };
				}
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					let idx = (by + dy) * 4 + bx + dx;
					let col_idx = if inference { (b8 / 2) * 12 + (b8 % 2) * 3 } else { idx };
					let col_zero = col_short && !col.intra && {
						let (l, r) = if col.ref_idx[0][b8] >= 0 {
							(0, col.ref_idx[0][b8])
						} else {
							(1, col.ref_idx[1][b8])
						};
						let mv = col.mv[l][col_idx];
						r == 0 && (-1..=1).contains(&mv[0]) && (-1..=1).contains(&mv[1])
					};
					for list in 0..2 {
						cur.mv[list][idx] = if refs[list] < 0 || direct_zero || (refs[list] == 0 && col_zero) {
							[0, 0]
						} else {
							mvp[list]
						};
						cur.mvd[list][idx] = [0, 0];
					}
				}
			}
		} else {
			for b8 in 0..4 {
				if mask & (1 << b8) == 0 {
					continue;
				}
				let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
				// Reference: the colocated block's reference, mapped into L0.
				let (col_list, ref_l0) = if col.intra {
					(0, 0i8)
				} else {
					let l = if col.ref_idx[0][b8] >= 0 { 0 } else { 1 };
					let id = col.ref_id[l][b8];
					let mapped = self.lists[0]
						.iter()
						.position(|r| r.as_ref().is_some_and(|r| r.pic.id == id))
						.unwrap_or(0);
					(l, mapped as i8)
				};
				cur.ref_idx[0][b8] = ref_l0;
				cur.ref_idx[1][b8] = 0;
				let pic0 = self.lists[0]
					.get(ref_l0 as usize)
					.and_then(|r| r.as_ref())
					.ok_or(BadMb("missing temporal direct reference"))?;
				let tb = clip3(-128, 127, self.cur_poc.wrapping_sub(pic0.pic.poc));
				let td = clip3(-128, 127, col_pic.pic.poc.wrapping_sub(pic0.pic.poc));
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					let idx = (by + dy) * 4 + bx + dx;
					let col_idx = if inference { (b8 / 2) * 12 + (b8 % 2) * 3 } else { idx };
					let mv_col = if col.intra { [0, 0] } else { col.mv[col_list][col_idx] };
					let (mv0, mv1) = if pic0.long || td == 0 {
						(mv_col, [0, 0])
					} else {
						let tx = (16384 + (td / 2).abs()) / td;
						let dsf = clip3(-1024, 1023, (tb * tx + 32) >> 6);
						let s = |v: i16| ((dsf * i32::from(v) + 128) >> 8) as i16;
						let mv0 = [s(mv_col[0]), s(mv_col[1])];
						(mv0, [mv0[0].wrapping_sub(mv_col[0]), mv0[1].wrapping_sub(mv_col[1])])
					};
					cur.mv[0][idx] = mv0;
					cur.mv[1][idx] = mv1;
					cur.mvd[0][idx] = [0, 0];
					cur.mvd[1][idx] = [0, 0];
				}
			}
		}
		Ok(())
	}

	/// Weighting for a block predicted from `refs` (8.4.2.3).
	fn weighting(&self, refs: [i8; 2], comp: usize) -> Weighting {
		let explicit = match self.h.slice_type {
			SliceType::P => self.pps.weighted_pred,
			SliceType::B => self.pps.weighted_bipred_idc == 1,
			SliceType::I => false,
		};
		if explicit {
			let Some(table) = self.h.weights.as_ref() else {
				return Weighting::Default;
			};
			let log_wd = if comp == 0 { table.luma_log2_denom } else { table.chroma_log2_denom };
			let mut w = [1 << log_wd, 1 << log_wd];
			let mut o = [0, 0];
			for list in 0..2 {
				if refs[list] < 0 {
					continue;
				}
				if let Some(e) = table.weights[list].get(refs[list] as usize) {
					let (wt, off) = if comp == 0 { e.luma } else { e.chroma[comp - 1] };
					w[list] = wt;
					o[list] = off;
				}
			}
			return Weighting::Weighted { log_wd, w, o };
		}
		if self.h.slice_type == SliceType::B
			&& self.pps.weighted_bipred_idc == 2
			&& refs[0] >= 0
			&& refs[1] >= 0
		{
			let (Some(Some(r0)), Some(Some(r1))) =
				(self.lists[0].get(refs[0] as usize), self.lists[1].get(refs[1] as usize))
			else {
				return Weighting::Default;
			};
			let tb = clip3(-128, 127, self.cur_poc.wrapping_sub(r0.pic.poc));
			let td = clip3(-128, 127, r1.pic.poc.wrapping_sub(r0.pic.poc));
			let (w0, w1) = if td == 0 || r0.long || r1.long {
				(32, 32)
			} else {
				let tx = (16384 + (td / 2).abs()) / td;
				let dsf = clip3(-1024, 1023, (tb * tx + 32) >> 6);
				if !(-64..=128).contains(&(dsf >> 2)) { (32, 32) } else { (64 - (dsf >> 2), dsf >> 2) }
			};
			return Weighting::Weighted { log_wd: 5, w: [w0, w1], o: [0, 0] };
		}
		Weighting::Default
	}

	/// Predicts one rectangle (4x4 units) from its motion into the frame.
	fn mc_rect(&mut self, cur: &MbInfo, x4: usize, y4: usize, w4: usize, h4: usize) -> MbResult<()> {
		let idx = y4 * 4 + x4;
		let b8 = b8_of(idx);
		let refs = [cur.ref_idx[0][b8], cur.ref_idx[1][b8]];
		let (bw, bh) = (w4 * 4, h4 * 4);
		let px = self.mbx * 16 + x4 * 4;
		let py = self.mby * 16 + y4 * 4;
		let mut luma = [[0u8; 256]; 2];
		let mut chroma = [[[0u8; 64]; 2]; 2];
		for list in 0..2 {
			if refs[list] < 0 {
				continue;
			}
			let r = self.lists[list]
				.get(refs[list] as usize)
				.and_then(|r| r.as_ref())
				.ok_or(BadMb("reference picture missing"))?;
			let mv = cur.mv[list][idx];
			let (mx, my) = (i32::from(mv[0]), i32::from(mv[1]));
			recon::mc_luma(
				&r.pic.y,
				px as i32 + (mx >> 2),
				py as i32 + (my >> 2),
				mx & 3,
				my & 3,
				bw,
				bh,
				&mut luma[list],
			);
			let (cx, cy) = ((px / 2) as i32 + (mx >> 3), (py / 2) as i32 + (my >> 3));
			recon::mc_chroma(&r.pic.cb, cx, cy, mx & 7, my & 7, bw / 2, bh / 2, &mut chroma[0][list]);
			recon::mc_chroma(&r.pic.cr, cx, cy, mx & 7, my & 7, bw / 2, bh / 2, &mut chroma[1][list]);
		}
		let w_luma = self.weighting(refs, 0);
		let (l0, l1) = (&luma[0][..], &luma[1][..]);
		recon::store_prediction(
			&mut self.f.y,
			px,
			py,
			bw,
			bh,
			(refs[0] >= 0).then_some(l0),
			(refs[1] >= 0).then_some(l1),
			w_luma,
		);
		for c in 0..2 {
			let wt = self.weighting(refs, c + 1);
			let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
			let (l0, l1) = (&chroma[c][0], &chroma[c][1]);
			recon::store_prediction(
				plane,
				px / 2,
				py / 2,
				bw / 2,
				bh / 2,
				(refs[0] >= 0).then_some(&l0[..]),
				(refs[1] >= 0).then_some(&l1[..]),
				wt,
			);
		}
		Ok(())
	}

	/// Motion-compensated prediction of a whole inter macroblock.
	pub fn motion_compensate(&mut self, cur: &MbInfo, mb_type: Option<MbType>) -> MbResult<()> {
		let direct_rects = |s: &mut Self, b8: usize| -> MbResult<()> {
			let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
			if s.sps.direct_8x8_inference {
				s.mc_rect(cur, bx, by, 2, 2)
			} else {
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					s.mc_rect(cur, bx + dx, by + dy, 1, 1)?;
				}
				Ok(())
			}
		};
		match (cur.kind, mb_type) {
			(MbKind::PSkip, _) => self.mc_rect(cur, 0, 0, 4, 4),
			(MbKind::BSkip, _) | (_, Some(MbType::Direct)) => {
				for b8 in 0..4 {
					direct_rects(self, b8)?;
				}
				Ok(())
			}
			(_, Some(MbType::Inter { shape: Shape::S16x16, .. })) => self.mc_rect(cur, 0, 0, 4, 4),
			(_, Some(MbType::Inter { shape: Shape::S16x8, .. })) => {
				self.mc_rect(cur, 0, 0, 4, 2)?;
				self.mc_rect(cur, 0, 2, 4, 2)
			}
			(_, Some(MbType::Inter { shape: Shape::S8x16, .. })) => {
				self.mc_rect(cur, 0, 0, 2, 4)?;
				self.mc_rect(cur, 2, 0, 2, 4)
			}
			(_, Some(MbType::Inter { shape: Shape::S8x8, .. })) => {
				let subs = self.sub_types;
				for (b8, s) in subs.iter().enumerate() {
					let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
					if s.pred == Pred::Direct {
						direct_rects(self, b8)?;
						continue;
					}
					for part in 0..s.parts {
						let (x4, y4) = match (s.w, s.h) {
							(2, 2) => (bx, by),
							(2, 1) => (bx, by + part),
							(1, 2) => (bx + part, by),
							_ => (bx + part % 2, by + part / 2),
						};
						self.mc_rect(cur, x4, y4, s.w, s.h)?;
					}
				}
				Ok(())
			}
			_ => Ok(()),
		}
	}
}
