//! Motion vector prediction (§8.4.1.3), P_Skip (§8.4.1.1), spatial and
//! temporal direct prediction (§8.4.1.2) and motion-compensated
//! prediction with weighting (§8.4.2). FFmpeg: `h264_mvpred.h`,
//! `h264_direct.c`, `h264_mb.c` (`mc_part`).

use super::decode::SliceDecoder;
use super::dpb::{MbMotion, RefPic};
use super::mb::{BadMb, MbInfo, MbKind, MbResult, MbType, Pred, Shape, blk_index};
use super::recon::{self, Pixel, Plane, Weighting};
use super::slice::SliceType;

/// Motion data of one neighbouring partition.
#[derive(Debug, Clone, Copy)]
struct Nb {
	avail: bool,
	ref_idx: i8,
	mv: [i16; 2],
}

const UNAVAILABLE: Nb = Nb { avail: false, ref_idx: -1, mv: [0, 0] };

/// One colocated 4x4 block (8.4.1.2.1).
#[derive(Debug, Clone, Copy, Default)]
struct ColBlock {
	intra: bool,
	ref_idx: [i8; 2],
	ref_id: [u32; 2],
	mv: [[i16; 2]; 2],
}

/// vertMvScale (Table 8-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertScale {
	One,
	FrmToFld,
	FldToFrm,
}

/// The field of a colocated field pair whose order count is closer to the
/// current frame's (8-188; the bottom field on a tie, as in FFmpeg).
fn col_parity<T: Pixel>(col: &RefPic<T>, cur_poc: i32) -> usize {
	let d = |p: usize| (i64::from(col.field_poc[p]) - i64::from(cur_poc)).abs();
	usize::from(d(0) >= d(1))
}

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

impl<'a, T: Pixel> SliceDecoder<'a, T> {
	/// Neighbour partition at 4x4 position (x, y) relative to the current
	/// macroblock, seen from the partition starting at (px, py).
	fn mv_nb(&self, cur: &MbInfo, list: usize, x: i32, y: i32, px: usize, py: usize) -> Nb {
		self.mv_nb_row(cur, list, x, y, px, py, 0)
	}

	/// [`Self::mv_nb`] for sample row `4 * y + sub` of the 4x4 block (see
	/// `nb4_row`).
	#[allow(clippy::too_many_arguments)]
	fn mv_nb_row(
		&self,
		cur: &MbInfo,
		list: usize,
		x: i32,
		y: i32,
		px: usize,
		py: usize,
		sub: i32,
	) -> Nb {
		if (0..4).contains(&x)
			&& (0..4).contains(&y)
			&& blk_index(x as usize, y as usize) >= blk_index(px, py)
		{
			return UNAVAILABLE; // not yet decoded
		}
		if x >= 4 && y >= 0 {
			return UNAVAILABLE;
		}
		match self.nb4_row(cur, x, y, sub) {
			None => UNAVAILABLE,
			Some((m, idx)) => {
				if m.is_intra() {
					return Nb { avail: true, ref_idx: -1, mv: [0, 0] };
				}
				let r = m.ref_idx[list][b8_of(idx)];
				if r < 0 {
					return Nb { avail: true, ref_idx: -1, mv: [0, 0] };
				}
				let mv = m.mv[list][idx];
				// MBAFF frame/field neighbours (8.4.1.3.1; C division).
				match self.nb_scale(m) {
					1 => Nb { avail: true, ref_idx: r * 2, mv: [mv[0], mv[1] / 2] },
					-1 => Nb { avail: true, ref_idx: r >> 1, mv: [mv[0], mv[1].wrapping_mul(2)] },
					_ => Nb { avail: true, ref_idx: r, mv },
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
			c = self.mv_nb_row(cur, list, x - 1, y - 1, x4, y4, 3);
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

	/// The motion of macroblock `idx` (storage index) of the colocated
	/// picture: read from the frame being decoded when it is the first
	/// field of the current pair.
	fn col_motion(&self, col: &RefPic<T>, idx: usize, row: usize) -> MbMotion {
		if col.pic.id == self.cur_id {
			return self.f.mbs.get(idx).map(MbInfo::motion).unwrap_or_default();
		}
		let data = &col.pic.data;
		data.with(data.need_motion(row), |p| p.motion.get(idx).copied()).unwrap_or_default()
	}

	/// The colocated 4x4 blocks of the current macroblock (8.4.1.2.1,
	/// Table 8-6) and how their vertical motion scales. `col` is
	/// RefPicList1[0] as seen by the current macroblock (a field for field
	/// macroblocks).
	fn colocated(&self, col: &RefPic<T>) -> ([ColBlock; 16], VertScale) {
		let w = self.f.mb_w;
		let inference = self.sps.direct_8x8_inference;
		// The current macroblock's storage row (field rows interleaved).
		let cur_row = match self.field {
			Some(p) => 2 * self.mby + p,
			None => self.mby,
		};
		let cur_field = self.cur_field();
		let probe = if col.pic.field_coded {
			None
		} else {
			Some(self.col_motion(col, cur_row * w + self.mbx, cur_row))
		};
		// A field-coded colocated picture, or a field pair of an MBAFF one.
		let col_field = probe.is_none_or(|m| m.field);
		let scale = match (cur_field, col_field) {
			(true, false) => VertScale::FrmToFld,
			(false, true) => VertScale::FldToFrm,
			_ => VertScale::One,
		};
		let fld_parity = if scale == VertScale::FldToFrm { col_parity(col, self.cur_poc()) } else { 0 };
		// (xCol, yCol) of each 4x4 block: its own position, or the corner of
		// its 8x8 block with direct_8x8_inference.
		let pos = |idx: usize| -> (usize, usize) {
			let (x4, y4) = (idx % 4, idx / 4);
			if inference { ((x4 / 2) * 3, (y4 / 2) * 3) } else { (x4, y4) }
		};
		let mut out = [ColBlock::default(); 16];
		// Fetch at most two macroblocks.
		let mut cache: [(usize, MbMotion); 2] = [(usize::MAX, MbMotion::default()); 2];
		if let Some(m) = probe {
			cache[0] = (cur_row * w + self.mbx, m);
		}
		let mut fetch = |s: &Self, idx: usize, row: usize| -> MbMotion {
			for c in cache.iter() {
				if c.0 == idx {
					return c.1;
				}
			}
			let m = s.col_motion(col, idx, row);
			if cache[0].0 == usize::MAX {
				cache[0] = (idx, m)
			} else {
				cache[1] = (idx, m)
			}
			m
		};
		for (idx, b) in out.iter_mut().enumerate() {
			let (xc, yc) = pos(idx);
			// Colocated macroblock row (storage) and 4x4 row inside it.
			let (row, y4) = match scale {
				// Same structure; a field of a field-coded picture is read in
				// its own parity (FFmpeg's col_fieldoff).
				VertScale::One => match (self.field, col.pic.field_coded, col.parity) {
					(Some(_), true, Some(q)) => ((cur_row & !1) + q, yc),
					_ => (cur_row, yc),
				},
				// Field from a frame: the two frame macroblocks of the pair.
				VertScale::FrmToFld => ((cur_row & !1) + yc / 2, (2 * yc) % 4),
				// Frame from a field pair: the field macroblock of the field
				// with the closer count (8-188).
				VertScale::FldToFrm => ((cur_row & !1) + fld_parity, 2 * (cur_row & 1) + yc / 2),
			};
			let m = fetch(self, row * w + self.mbx, row);
			let blk = y4 * 4 + xc;
			let b8 = (y4 / 2) * 2 + xc / 2;
			*b = ColBlock {
				intra: m.intra,
				ref_idx: [m.ref_idx[0][b8], m.ref_idx[1][b8]],
				ref_id: [m.ref_id[0][b8], m.ref_id[1][b8]],
				mv: [m.mv[0][blk], m.mv[1][blk]],
			};
		}
		(out, scale)
	}

	/// Direct prediction (spatial or temporal) for the 8x8 blocks in `mask`.
	pub fn predict_direct(&self, cur: &mut MbInfo, mask: u8) -> MbResult<()> {
		let col_pic = self
			.list(1)
			.first()
			.and_then(|r| r.as_ref())
			.ok_or(BadMb("direct mode without RefPicList1[0]"))?;
		let (col, scale) = self.colocated(col_pic);
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
					let cb = &col[idx];
					let col_zero = col_short && !cb.intra && {
						let (l, r) = if cb.ref_idx[0] >= 0 { (0, cb.ref_idx[0]) } else { (1, cb.ref_idx[1]) };
						let mv = cb.mv[l];
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
			let cur_parity = self.cur_parity();
			for b8 in 0..4 {
				if mask & (1 << b8) == 0 {
					continue;
				}
				let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
				// Reference: the colocated block's reference, mapped into L0
				// (8-191): the same frame or field; a frame's field of the
				// current parity (frame to field); a field's frame (field to
				// frame).
				let corner = &col[by * 4 + bx];
				let (col_list, ref_l0) = if corner.intra {
					(0, 0i8)
				} else {
					let l = if corner.ref_idx[0] >= 0 { 0 } else { 1 };
					let key = corner.ref_id[l];
					let key = match scale {
						VertScale::One => key,
						VertScale::FrmToFld => (key & !3) | (1 + cur_parity as u32),
						VertScale::FldToFrm => key | 3,
					};
					let mapped = self
						.list(0)
						.iter()
						.position(|r| r.as_ref().is_some_and(|r| r.key() == key))
						.unwrap_or(0);
					(l, mapped as i8)
				};
				cur.ref_idx[0][b8] = ref_l0;
				cur.ref_idx[1][b8] = 0;
				let pic0 = self
					.list(0)
					.get(ref_l0 as usize)
					.and_then(|r| r.as_ref())
					.ok_or(BadMb("missing temporal direct reference"))?;
				let tb = clip3(-128, 127, self.cur_poc().wrapping_sub(pic0.poc));
				let td = clip3(-128, 127, col_pic.poc.wrapping_sub(pic0.poc));
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					let idx = (by + dy) * 4 + bx + dx;
					let cb = &col[idx];
					let mut mv_col = if cb.intra { [0, 0] } else { cb.mv[col_list] };
					mv_col[1] = match scale {
						VertScale::One => mv_col[1],
						VertScale::FrmToFld => mv_col[1] / 2,
						VertScale::FldToFrm => mv_col[1].wrapping_mul(2),
					};
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
				// MBAFF field macroblocks use the weights of the frame
				// (refIdxL0WP = refIdxL0 >> 1, 8-268).
				let wp_idx = (refs[list] as usize) >> usize::from(self.mbaff && self.mb_field);
				if let Some(e) = table.weights[list].get(wp_idx) {
					let (wt, off) = if comp == 0 { e.luma } else { e.chroma[comp - 1] };
					w[list] = wt;
					// Offsets are in 8-bit units (8-273, 8-274).
					o[list] = off << (self.bd - 8);
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
				(self.list(0).get(refs[0] as usize), self.list(1).get(refs[1] as usize))
			else {
				return Weighting::Default;
			};
			let tb = clip3(-128, 127, self.cur_poc().wrapping_sub(r0.poc));
			let td = clip3(-128, 127, r1.poc.wrapping_sub(r0.poc));
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
		let (field, mx0, my0) = self.mb_view();
		let px = mx0 + x4 * 4;
		let py = my0 + y4 * 4;
		let c422 = self.c422;
		// Chroma block: half width; half height for 4:2:0, full for 4:2:2.
		let (cbw, cbh) = (bw / 2, if c422 { bh } else { bh / 2 });
		let (cpx, cpy) = (px / 2, if c422 { py } else { py / 2 });
		let max = self.max;
		let mut luma = [[T::default(); 256]; 2];
		let mut chroma = [[[T::default(); 128]; 2]; 2];
		for list in 0..2 {
			if refs[list] < 0 {
				continue;
			}
			let r = self
				.list(list)
				.get(refs[list] as usize)
				.and_then(|r| r.as_ref())
				.ok_or(BadMb("reference picture missing"))?;
			let mv = cur.mv[list][idx];
			let (mx, my) = (i32::from(mv[0]), i32::from(mv[1]));
			let (lx, ly) = (px as i32 + (mx >> 2), py as i32 + (my >> 2));
			// 4:2:0 field prediction from the other parity: the chroma
			// sample grids are offset by a quarter sample (Table 8-10).
			let rpar = r.parity;
			let mcy = match (c422, field, rpar) {
				(false, Some(cp), Some(rp)) => my + 2 * (cp as i32 - rp as i32),
				_ => my,
			};
			// Chroma vectors: eighth samples; vertically quarter samples for
			// 4:2:2 (8.4.1.4, 8.4.2.2.2).
			let (cx, cfx) = (cpx as i32 + (mx >> 3), mx & 7);
			let (cy, cfy) = if c422 {
				(cpy as i32 + (mcy >> 2), (mcy & 3) << 1)
			} else {
				(cpy as i32 + (mcy >> 3), mcy & 7)
			};
			let (l, c) = (&mut luma[list], &mut chroma);
			let mut predict = |y: &Plane<T>, cb: &Plane<T>, cr: &Plane<T>, c: &mut [[[T; 128]; 2]; 2]| {
				recon::mc_luma(&y.src(rpar, max), lx, ly, mx & 3, my & 3, bw, bh, l);
				let (cb, cr) = (cb.src(rpar, max), cr.src(rpar, max));
				recon::mc_chroma(&cb, cx, cy, cfx, cfy, cbw, cbh, &mut c[0][list]);
				recon::mc_chroma(&cr, cx, cy, cfx, cfy, cbw, cbh, &mut c[1][list]);
			};
			if r.pic.id == self.cur_id {
				// The first field of the frame being decoded.
				predict(&self.f.y, &self.f.cb, &self.f.cr, c);
				continue;
			}
			// Frame threading (FFmpeg `await_references`): wait until the
			// reference rows this block reads are final. The 6-tap window
			// reaches 3 rows below the block, the bilinear chroma one 1 row;
			// coordinates outside the picture are clamped onto its edge. A
			// field's rows are every other row of the frame.
			let data = &r.pic.data;
			let (rows, crows) = match rpar {
				Some(_) => (self.f.y.height / 2, self.f.cb.height / 2),
				None => (self.f.y.height, self.f.cb.height),
			};
			let frame_row = |v: i32, n: usize| -> usize {
				let v = v.clamp(0, n as i32 - 1) as usize;
				match rpar {
					Some(p) => 2 * v + p,
					None => v,
				}
			};
			let luma_bottom = frame_row(ly + bh as i32 + 2, rows);
			let chroma_bottom = frame_row(cy + cbh as i32, crows);
			let need = data.need_luma(luma_bottom).max(data.need_chroma(chroma_bottom));
			data.with(need, |p| predict(&p.y, &p.cb, &p.cr, c));
		}
		let w_luma = self.weighting(refs, 0);
		let (l0, l1) = (&luma[0][..], &luma[1][..]);
		recon::store_prediction(
			&mut self.f.y.dst(field, max),
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
				&mut plane.dst(field, max),
				cpx,
				cpy,
				cbw,
				cbh,
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
