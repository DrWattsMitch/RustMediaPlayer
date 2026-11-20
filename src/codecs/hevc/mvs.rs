//! Motion vector prediction: merge mode, AMVP and temporal candidates
//! (ITU-T H.265 §8.5.3.2; FFmpeg `hevc/mvs.c`).
//!
//! This is a child module of `ctu` so it can extend `SliceDecoder`.

use super::{PartMode, SliceDecoder};
use crate::codecs::hevc::picture::{MvField, PF_BI, PF_INTRA, PF_L0, PF_L1};
use crate::codecs::hevc::slice::SliceType;

/// A prediction block and the coding block that contains it.
#[derive(Clone, Copy)]
pub(super) struct Pb {
	pub xcb: i32,
	pub ycb: i32,
	pub ncbs: i32,
	pub x: i32,
	pub y: i32,
	pub w: i32,
	pub h: i32,
	pub part_idx: usize,
	pub part: PartMode,
}

#[inline]
fn scale_mv(mv: [i16; 2], td: i32, tb: i32) -> [i16; 2] {
	let td = if td == 0 { 1 } else { td.clamp(-128, 127) };
	let tb = tb.clamp(-128, 127);
	let tx = (16384 + (td.abs() >> 1)) / td;
	let f = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
	let s = |v: i16| {
		let p = f * i32::from(v);
		(p.signum() * ((p.abs() + 127) >> 8)).clamp(-32768, 32767) as i16
	};
	[s(mv[0]), s(mv[1])]
}

impl SliceDecoder<'_, '_> {
	/// Prediction block availability (6.4.2), including the intra check.
	fn pb_available(&self, pb: &Pb, xn: i32, yn: i32) -> bool {
		let same_cb = pb.xcb <= xn && pb.ycb <= yn && pb.xcb + pb.ncbs > xn && pb.ycb + pb.ncbs > yn;
		let avail = if same_cb {
			!((pb.w << 1) == pb.ncbs
				&& (pb.h << 1) == pb.ncbs
				&& pb.part_idx == 1
				&& pb.ycb + pb.h <= yn
				&& pb.xcb + pb.w > xn)
		} else {
			self.available(pb.x, pb.y, xn, yn)
		};
		avail && self.pic.mvf[self.i4(xn, yn)].pred != PF_INTRA
	}

	#[inline]
	fn mvf_at(&self, x: i32, y: i32) -> MvField {
		self.pic.mvf[self.i4(x, y)]
	}

	// -----------------------------------------------------------------
	// Temporal motion vector prediction (8.5.3.2.8-9)
	// -----------------------------------------------------------------

	fn temporal_mv(
		&self,
		x: i32,
		y: i32,
		w: i32,
		h: i32,
		ref_idx: usize,
		list: usize,
	) -> Option<[i16; 2]> {
		if !self.sh.temporal_mvp_enabled {
			return None;
		}
		let col = self.refs.collocated.as_ref()?;
		let sps = &*self.sps;
		let (xbr, ybr) = (x + w, y + h);
		if (y >> sps.log2_ctb_size) == (ybr >> sps.log2_ctb_size)
			&& ybr < sps.height as i32
			&& xbr < sps.width as i32
		{
			if let Some(mv) = self.col_mv(col, (xbr >> 4) << 4, (ybr >> 4) << 4, ref_idx, list) {
				return Some(mv);
			}
		}
		let (xc, yc) = (((x + (w >> 1)) >> 4) << 4, ((y + (h >> 1)) >> 4) << 4);
		self.col_mv(col, xc, yc, ref_idx, list)
	}

	fn col_mv(
		&self,
		col: &crate::codecs::hevc::picture::RefPic,
		x: i32,
		y: i32,
		ref_idx: usize,
		list: usize,
	) -> Option<[i16; 2]> {
		// The collocated picture may still be decoding on another thread.
		col.with_motion(y as usize, |f| {
			let m = *f.mv_at(x as usize, y as usize);
			if m.pred == PF_INTRA {
				return None;
			}
			let list_col = if m.pred & PF_L0 == 0 {
				1
			} else if m.pred & PF_L1 == 0 {
				0
			} else if self.refs.no_backward_pred {
				list
			} else {
				usize::from(self.sh.collocated_from_l0)
			};
			let col_refs = f.refs_at(x as usize, y as usize);
			let ri = m.ref_idx[list_col] as usize;
			let cur = &self.refs.info;
			let cur_lt = cur.long_term[list][ref_idx];
			if cur_lt != col_refs.long_term[list_col][ri] {
				return None;
			}
			let mv = m.mv[list_col];
			let col_diff = col.poc - col_refs.poc[list_col][ri];
			let cur_diff = self.pic.poc - cur.poc[list][ref_idx];
			if cur_lt || col_diff == cur_diff || col_diff == 0 {
				Some(mv)
			} else {
				Some(scale_mv(mv, col_diff, cur_diff))
			}
		})
	}

	// -----------------------------------------------------------------
	// Merge mode (8.5.3.2.2-5)
	// -----------------------------------------------------------------

	pub(super) fn merge_mode(&self, pb0: Pb, merge_idx: usize) -> MvField {
		let (orig_w, orig_h) = (pb0.w, pb0.h);
		let mut pb = pb0;
		let par = self.pps.log2_parallel_merge_level;
		let single = par > 2 && pb.ncbs == 8;
		if single {
			pb = Pb { x: pb.xcb, y: pb.ycb, w: pb.ncbs, h: pb.ncbs, part_idx: 0, ..pb };
		}
		let (x, y, w, h) = (pb.x, pb.y, pb.w, pb.h);
		let par_ok = |xn: i32, yn: i32| !((x >> par) == (xn >> par) && (y >> par) == (yn >> par));
		let mut cands: [MvField; 5] = [MvField::default(); 5];
		let mut n = 0usize;
		let max = self.sh.max_num_merge_cand as usize;

		let finish = |mut m: MvField| {
			if m.pred == PF_BI && orig_w + orig_h == 12 {
				m.pred = PF_L0;
				m.ref_idx[1] = -1;
				m.mv[1] = [0, 0];
			}
			m
		};

		// A1
		let (xa1, ya1) = (x - 1, y + h - 1);
		let a1_ok = !(!single
			&& pb.part_idx == 1
			&& matches!(pb.part, PartMode::PNx2N | PartMode::PnLx2N | PartMode::PnRx2N))
			&& par_ok(xa1, ya1)
			&& self.pb_available(&pb, xa1, ya1);
		let a1 = if a1_ok { Some(self.mvf_at(xa1, ya1)) } else { None };
		if let Some(m) = a1 {
			cands[n] = m;
			if n == merge_idx {
				return finish(m);
			}
			n += 1;
		}
		// B1
		let (xb1, yb1) = (x + w - 1, y - 1);
		let b1_ok = !(!single
			&& pb.part_idx == 1
			&& matches!(pb.part, PartMode::P2NxN | PartMode::P2NxnU | PartMode::P2NxnD))
			&& par_ok(xb1, yb1)
			&& self.pb_available(&pb, xb1, yb1);
		let b1 = if b1_ok { Some(self.mvf_at(xb1, yb1)) } else { None };
		if let Some(m) = b1 {
			if a1 != Some(m) {
				cands[n] = m;
				if n == merge_idx {
					return finish(m);
				}
				n += 1;
			}
		}
		// B0
		let (xb0, yb0) = (x + w, y - 1);
		if par_ok(xb0, yb0) && self.pb_available(&pb, xb0, yb0) {
			let m = self.mvf_at(xb0, yb0);
			if b1 != Some(m) {
				cands[n] = m;
				if n == merge_idx {
					return finish(m);
				}
				n += 1;
			}
		}
		// A0
		let (xa0, ya0) = (x - 1, y + h);
		if par_ok(xa0, ya0) && self.pb_available(&pb, xa0, ya0) {
			let m = self.mvf_at(xa0, ya0);
			if a1 != Some(m) {
				cands[n] = m;
				if n == merge_idx {
					return finish(m);
				}
				n += 1;
			}
		}
		// B2
		let (xb2, yb2) = (x - 1, y - 1);
		if n != 4 && par_ok(xb2, yb2) && self.pb_available(&pb, xb2, yb2) {
			let m = self.mvf_at(xb2, yb2);
			if a1 != Some(m) && b1 != Some(m) {
				cands[n] = m;
				if n == merge_idx {
					return finish(m);
				}
				n += 1;
			}
		}
		// Temporal.
		let is_b = self.sh.slice_type == SliceType::B;
		if n < max {
			let mut m = MvField { ref_idx: [0, -1], ..Default::default() };
			if let Some(mv) = self.temporal_mv(x, y, w, h, 0, 0) {
				m.pred |= PF_L0;
				m.mv[0] = mv;
			}
			if is_b {
				if let Some(mv) = self.temporal_mv(x, y, w, h, 0, 1) {
					m.pred |= PF_L1;
					m.mv[1] = mv;
					m.ref_idx[1] = 0;
				}
			}
			if m.pred != PF_INTRA {
				if m.pred & PF_L0 == 0 {
					m.ref_idx[0] = -1;
				}
				cands[n] = m;
				if n == merge_idx {
					return finish(m);
				}
				n += 1;
			}
		}
		// Combined bi-predictive candidates.
		let mut out = Vec::with_capacity(max);
		out.extend_from_slice(&cands[..n]);
		if is_b && n > 1 && n < max {
			const L0: [usize; 12] = [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3];
			const L1: [usize; 12] = [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2];
			let orig = n;
			let info = &self.refs.info;
			for comb in 0..orig * (orig - 1) {
				if out.len() >= max {
					break;
				}
				let (c0, c1) = (cands[L0[comb]], cands[L1[comb]]);
				if c0.pred & PF_L0 != 0 && c1.pred & PF_L1 != 0 {
					let p0 = info.poc[0][c0.ref_idx[0] as usize];
					let p1 = info.poc[1][c1.ref_idx[1] as usize];
					if p0 != p1 || c0.mv[0] != c1.mv[1] {
						out.push(MvField {
							mv: [c0.mv[0], c1.mv[1]],
							ref_idx: [c0.ref_idx[0], c1.ref_idx[1]],
							pred: PF_BI,
						});
						if out.len() - 1 == merge_idx {
							return finish(out[merge_idx]);
						}
					}
				}
			}
		}
		// Zero candidates.
		let num_ref = if is_b {
			self.sh.num_ref_idx[0].min(self.sh.num_ref_idx[1])
		} else {
			self.sh.num_ref_idx[0]
		} as usize;
		let mut zero = 0usize;
		while out.len() <= merge_idx {
			let r = if zero < num_ref { zero as i8 } else { 0 };
			out.push(if is_b {
				MvField { mv: [[0; 2]; 2], ref_idx: [r, r], pred: PF_BI }
			} else {
				MvField { mv: [[0; 2]; 2], ref_idx: [r, -1], pred: PF_L0 }
			});
			zero += 1;
		}
		finish(out[merge_idx])
	}

	// -----------------------------------------------------------------
	// AMVP (8.5.3.2.6-7)
	// -----------------------------------------------------------------

	/// The two motion vector predictor candidates for list `lx`.
	pub(super) fn mvp(&self, pb: Pb, lx: usize, ref_idx: usize) -> [[i16; 2]; 2] {
		let info = &self.refs.info;
		let ly = 1 - lx;
		let target_poc = info.poc[lx][ref_idx];
		let target_lt = info.long_term[lx][ref_idx];
		let cur_poc = self.pic.poc;
		let (x, y, w, h) = (pb.x, pb.y, pb.w, pb.h);

		// Unscaled match: same reference picture.
		let direct = |m: &MvField| -> Option<[i16; 2]> {
			if m.pred & (1 << lx) != 0 && info.poc[lx][m.ref_idx[lx] as usize] == target_poc {
				// Long-term status always agrees when the picture is the same.
				Some(m.mv[lx])
			} else if m.pred & (1 << ly) != 0 && info.poc[ly][m.ref_idx[ly] as usize] == target_poc {
				Some(m.mv[ly])
			} else {
				None
			}
		};
		// Scaled match: same long-term status, scaled when both short-term.
		let scaled = |m: &MvField| -> Option<[i16; 2]> {
			for l in [lx, ly] {
				if m.pred & (1 << l) != 0 {
					let ri = m.ref_idx[l] as usize;
					if info.long_term[l][ri] == target_lt {
						let mv = m.mv[l];
						let poc = info.poc[l][ri];
						if target_lt || poc == target_poc {
							return Some(mv);
						}
						return Some(scale_mv(mv, cur_poc - poc, cur_poc - target_poc));
					}
				}
			}
			None
		};

		let a0 = (x - 1, y + h);
		let a1 = (x - 1, y + h - 1);
		let av_a0 = self.pb_available(&pb, a0.0, a0.1);
		let av_a1 = self.pb_available(&pb, a1.0, a1.1);
		let is_scaled = av_a0 || av_a1;
		let ma: Vec<MvField> = [(av_a0, a0), (av_a1, a1)]
			.iter()
			.filter(|(a, _)| *a)
			.map(|&(_, p)| self.mvf_at(p.0, p.1))
			.collect();
		let mut mv_a = ma.iter().find_map(direct);
		if mv_a.is_none() {
			mv_a = ma.iter().find_map(scaled);
		}

		let bpos = [(x + w, y - 1), (x + w - 1, y - 1), (x - 1, y - 1)];
		let mb: Vec<MvField> = bpos
			.iter()
			.filter(|p| self.pb_available(&pb, p.0, p.1))
			.map(|p| self.mvf_at(p.0, p.1))
			.collect();
		let mut mv_b = mb.iter().find_map(direct);
		if !is_scaled {
			if mv_b.is_some() && mv_a.is_none() {
				mv_a = mv_b;
			}
			mv_b = mb.iter().find_map(scaled);
		}

		let mut list: [[i16; 2]; 2] = [[0; 2]; 2];
		let mut n = 0;
		if let Some(a) = mv_a {
			list[n] = a;
			n += 1;
		}
		if let Some(b) = mv_b {
			if mv_a != Some(b) {
				list[n] = b;
				n += 1;
			}
		}
		if n < 2 {
			if let Some(t) = self.temporal_mv(x, y, w, h, ref_idx, lx) {
				list[n] = t;
			}
		}
		list
	}
}
