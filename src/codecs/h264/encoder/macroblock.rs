//! Macroblock analysis, coding and reconstruction (ISO/IEC 14496-10 §7.3.5
//! syntax, written instead of read).
//!
//! Every candidate is coded for real: prediction, transform, quantisation
//! (deadzone, or trellis with CABAC), reconstruction with the decoder's own
//! `recon` functions, and pricing of its syntax with the active entropy
//! coder ([`super::syntax::MbWriter`] into a scratch CAVLC writer or the
//! CABAC cost model). The macroblock type is chosen by rate-distortion
//! cost `SSD + λ·bits` (x264's `--subme 7`-style final decision), so the
//! choice always reflects what will actually be sent.
//!
//! Candidates:
//!
//! * intra: Intra16x16 (4 modes), Intra4x4 and, with the 8x8 transform,
//!   Intra8x8 (9 modes per block, mode by SATD + mode bits);
//! * P: P_Skip, P_L0 16x16, 16x8, 8x16 and 8x8, each partition with its
//!   own reference picture from list 0;
//! * B: B_Skip and B_Direct_16x16 (spatial direct), B 16x16 / 16x8 / 8x16
//!   with L0, L1 or bi-prediction per partition;
//! * inter residuals with the 4x4 or the 8x8 transform, by RD.
//!
//! Neighbour derivations (availability, intra-mode prediction, motion
//! vector prediction, P_Skip and direct motion) mirror `h264::mb` and
//! `h264::inter` clause for clause; the encoder's reconstruction must equal
//! a decoder's.

use super::cabac::{Bins, CabacCost, CabacWriter};
use super::me::{self, Padded};
use super::syntax::{MbSyn, MbWriter, Nbr, SliceKind, SynType, b_mb_type, parts};
use super::transform::{
	Quant, decimate_score, decimate_score64, fdct4, fdct8, hadamard2, hadamard4, satd4, trellis_cabac,
};
use crate::codecs::bitstream::{BitWriter, se_bits, ue_bits};
use crate::codecs::h264::cabac::BlockCat;
use crate::codecs::h264::consts::{BLK_XY, ZIGZAG_4X4, ZIGZAG_8X8, chroma_qp};
use crate::codecs::h264::decode::Frame;
use crate::codecs::h264::dpb::MbMotion;
use crate::codecs::h264::mb::{MbInfo, MbKind, Pred, Residual, Shape, blk_index};
use crate::codecs::h264::params::Dequant;
use crate::codecs::h264::recon::{self, Avail, Plane, Weighting};

/// Source picture, padded to whole macroblocks.
#[derive(Debug, Clone)]
pub struct Source {
	pub y: Plane,
	pub cb: Plane,
	pub cr: Plane,
}

/// A decoded (deblocked) reference picture.
#[derive(Debug)]
pub struct Reference {
	/// Identity used in `MbInfo::ref_id` (non-zero, unique per picture).
	pub id: u32,
	pub poc: i32,
	pub y: Plane,
	pub cb: Plane,
	pub cr: Plane,
	pub padded: Padded,
	pub half: me::HalfPel,
	/// Motion of every macroblock (the colocated data of direct mode).
	pub motion: Vec<MbMotion>,
}

/// Motion-vector neighbour (8.4.1.3.2).
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

/// A fully coded macroblock candidate.
struct Coded {
	info: MbInfo,
	syn: MbSyn,
	res: Box<Residual>,
	ssd: u64,
	/// Reconstructed samples: 256 luma, 64 Cb, 64 Cr.
	recon: Box<[u8; 384]>,
	/// Bits of its syntax with the active entropy coder.
	bits: f64,
}

/// Motion of one inter partition.
#[derive(Debug, Clone, Copy)]
struct PartMotion {
	pred: Pred,
	refs: [i8; 2],
	mvs: [[i16; 2]; 2],
}

/// The entropy coder of the slice.
pub enum SliceOut {
	Cavlc(BitWriter),
	Cabac(Box<CabacWriter>),
}

/// Coding tools and parameters of one picture.
pub struct PicParams<'a> {
	pub kind: SliceKind,
	pub l0: Vec<&'a Reference>,
	pub l1: Vec<&'a Reference>,
	pub qp: i32,
	pub transform_8x8: bool,
	pub trellis: bool,
	/// Largest vertical motion (in luma samples) the level allows.
	pub max_mv_y: i32,
}

/// Per-picture encoding context.
pub struct PicEncoder<'a> {
	pub src: &'a Source,
	pub f: &'a mut Frame,
	pub p: PicParams<'a>,
	pub dq: &'a Dequant,
	qp: i32,
	qpc: i32,
	/// λ for SATD-domain decisions and λ² for SSD-domain RD.
	lambda: u32,
	lambda2: f64,
	pub out: SliceOut,
	skip_run: u32,
	mbx: usize,
	mby: usize,
	/// Statistics for the caller.
	pub stats: MbStats,
}

/// Macroblock counts of one picture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MbStats {
	pub intra: u32,
	pub intra8x8: u32,
	pub skipped: u32,
	/// Inter macroblocks with the 8x8 transform.
	pub inter8x8: u32,
	/// Inter macroblocks with a reference index above 0.
	pub multi_ref: u32,
	/// Macroblocks with a bi-predicted or list-1 partition.
	pub list1: u32,
	pub direct: u32,
}

impl<'a> PicEncoder<'a> {
	pub fn new(
		src: &'a Source,
		f: &'a mut Frame,
		p: PicParams<'a>,
		dq: &'a Dequant,
		out: SliceOut,
	) -> Self {
		let qp = p.qp;
		// x264's lambda tables: λ = 2^((qp-12)/6), λ² = 0.85·2^((qp-12)/3).
		let lf = 2f64.powf(f64::from(qp - 12) / 6.0);
		Self {
			src,
			f,
			p,
			dq,
			qp,
			qpc: i32::from(chroma_qp(qp, 0, 0)),
			lambda: lf.round().max(1.0) as u32,
			lambda2: 0.85 * lf * lf,
			out,
			skip_run: 0,
			mbx: 0,
			mby: 0,
			stats: MbStats::default(),
		}
	}

	fn cabac(&self) -> bool {
		matches!(self.out, SliceOut::Cabac(_))
	}

	/// Codes every macroblock of the picture (one slice) into `self.out`.
	pub fn encode_slice(&mut self) {
		let total = self.f.mb_w * self.f.mb_h;
		for mby in 0..self.f.mb_h {
			for mbx in 0..self.f.mb_w {
				self.mbx = mbx;
				self.mby = mby;
				let best = match self.p.kind {
					SliceKind::I => self.best_intra(),
					SliceKind::P => self.decide_p(),
					SliceKind::B => self.decide_b(),
				};
				let last = mby * self.f.mb_w + mbx + 1 == total;
				self.commit(best, last);
			}
		}
		if let SliceOut::Cavlc(w) = &mut self.out {
			if self.p.kind != SliceKind::I && self.skip_run > 0 {
				w.ue(self.skip_run);
			}
		}
	}

	fn writer(&self) -> MbWriter<'_> {
		MbWriter {
			nb: self.nbr(),
			slice: self.p.kind,
			num_ref: [self.p.l0.len(), self.p.l1.len()],
			pps_8x8: self.p.transform_8x8,
		}
	}

	fn nbr(&self) -> Nbr<'_> {
		Nbr { mbs: &self.f.mbs, mb_w: self.f.mb_w, mb_h: self.f.mb_h, mbx: self.mbx, mby: self.mby }
	}

	fn commit(&mut self, c: Coded, last: bool) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		self.put_recon(&c.recon, px, py);
		let addr = self.mby * self.f.mb_w + self.mbx;
		let skip = c.info.is_skip();
		let st = &mut self.stats;
		st.intra += u32::from(c.info.is_intra());
		st.intra8x8 += u32::from(c.info.kind == MbKind::I8x8);
		st.skipped += u32::from(skip);
		st.inter8x8 += u32::from(c.info.kind == MbKind::Inter && c.info.t8x8);
		st.multi_ref +=
			u32::from(!c.info.is_intra() && c.info.ref_idx.iter().flatten().any(|&r| r > 0));
		st.list1 += u32::from(!c.info.is_intra() && c.info.ref_idx[1].iter().any(|&r| r >= 0));
		st.direct += u32::from(c.info.direct16);
		// Written with the neighbours as they stand (this macroblock not
		// yet stored), as a decoder parses it.
		let mut out = std::mem::replace(&mut self.out, SliceOut::Cavlc(BitWriter::new()));
		match &mut out {
			SliceOut::Cavlc(w) => {
				if skip {
					self.skip_run += 1;
				} else {
					if self.p.kind != SliceKind::I {
						w.ue(self.skip_run);
						self.skip_run = 0;
					}
					self.writer().cavlc(w, &c.info, &c.syn, &c.res);
				}
			}
			SliceOut::Cabac(cw) => {
				self.writer().cabac(cw.as_mut(), &c.info, &c.syn, &c.res);
				if !last {
					cw.terminate(false);
				}
			}
		}
		self.out = out;
		self.f.mbs[addr] = c.info;
	}

	/// Bits of a candidate's syntax with the active entropy coder.
	fn price(&self, info: &MbInfo, syn: &MbSyn, res: &Residual) -> f64 {
		match &self.out {
			SliceOut::Cavlc(_) => {
				if info.is_skip() {
					return 0.0;
				}
				let mut w = BitWriter::new();
				self.writer().cavlc(&mut w, info, syn, res);
				// The skip run before a coded macroblock.
				let run = if self.p.kind == SliceKind::I { 0 } else { ue_bits(self.skip_run) };
				f64::from(w.bits() as u32 + run)
			}
			SliceOut::Cabac(cw) => {
				let mut c = CabacCost::new(cw.states());
				self.writer().cabac(&mut c, info, syn, res);
				c.bits()
			}
		}
	}

	// -----------------------------------------------------------------------
	// Neighbours

	fn nb_mb(&self, dx: i32, dy: i32) -> Option<&MbInfo> {
		let x = self.mbx as i32 + dx;
		let y = self.mby as i32 + dy;
		if x < 0 || y < 0 || x >= self.f.mb_w as i32 || y >= self.f.mb_h as i32 {
			return None;
		}
		let mb = &self.f.mbs[y as usize * self.f.mb_w + x as usize];
		(mb.slice != 0).then_some(mb)
	}

	fn nb4<'b>(&'b self, cur: &'b MbInfo, x4: i32, y4: i32) -> Option<(&'b MbInfo, usize)> {
		let (dx, dy) = (x4.div_euclid(4), y4.div_euclid(4));
		let idx = (y4.rem_euclid(4) * 4 + x4.rem_euclid(4)) as usize;
		if dx == 0 && dy == 0 {
			return Some((cur, idx));
		}
		self.nb_mb(dx, dy).map(|m| (m, idx))
	}

	/// [left, top, top-right, top-left] macroblock availability.
	fn mb_avail(&self) -> [bool; 4] {
		[
			self.nb_mb(-1, 0).is_some(),
			self.nb_mb(0, -1).is_some(),
			self.nb_mb(1, -1).is_some(),
			self.nb_mb(-1, -1).is_some(),
		]
	}

	/// Intra 4x4 / 8x8 neighbour availability (the decoder's
	/// `block_avail`; `size` 1 or 2 in 4x4 units).
	fn block_avail(x4: usize, y4: usize, size: usize, mb: [bool; 4]) -> Avail {
		let [l, t, tr, tl] = mb;
		let left = if x4 > 0 { true } else { l };
		let top = if y4 > 0 { true } else { t };
		let top_left = match (x4 > 0, y4 > 0) {
			(true, true) => true,
			(false, true) => l,
			(true, false) => t,
			(false, false) => tl,
		};
		let rx = x4 + size;
		let top_right = if y4 == 0 {
			if rx < 4 { t } else { tr }
		} else if rx >= 4 {
			false
		} else {
			blk_index(rx, y4 - 1) < blk_index(x4, y4)
		};
		Avail { left, top, top_right, top_left }
	}

	/// Predicted Intra_4x4 / Intra_8x8 mode (8.3.1.1, 8.3.2.1).
	fn predicted_intra(&self, cur: &MbInfo, x4: usize, y4: usize) -> u8 {
		let mode_of = |n: Option<(&MbInfo, usize)>| -> Option<u8> {
			let (m, idx) = n?;
			Some(if matches!(m.kind, MbKind::I4x4 | MbKind::I8x8) { m.ipred[idx] } else { 2 })
		};
		let (x, y) = (x4 as i32, y4 as i32);
		match (mode_of(self.nb4(cur, x - 1, y)), mode_of(self.nb4(cur, x, y - 1))) {
			(Some(a), Some(b)) => a.min(b),
			_ => 2,
		}
	}

	fn mv_nb(&self, cur: &MbInfo, list: usize, x: i32, y: i32, px: usize, py: usize) -> Nb {
		if (0..4).contains(&x)
			&& (0..4).contains(&y)
			&& blk_index(x as usize, y as usize) >= blk_index(px, py)
		{
			return UNAVAILABLE;
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
				let r = m.ref_idx[list][(idx / 8) * 2 + (idx % 4) / 2];
				if r < 0 {
					Nb { avail: true, ref_idx: -1, mv: [0, 0] }
				} else {
					Nb { avail: true, ref_idx: r, mv: m.mv[list][idx] }
				}
			}
		}
	}

	/// Motion vector predictor (8.4.1.3) for a partition of list `list`
	/// with reference `ref_idx`.
	#[allow(clippy::too_many_arguments)]
	fn predict_mv(
		&self,
		cur: &MbInfo,
		list: usize,
		x4: usize,
		y4: usize,
		w4: usize,
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
		let (b, c) = if !b.avail && !c.avail && a.avail { (a, a) } else { (b, c) };
		match [a.ref_idx == ref_idx, b.ref_idx == ref_idx, c.ref_idx == ref_idx] {
			[true, false, false] => a.mv,
			[false, true, false] => b.mv,
			[false, false, true] => c.mv,
			_ => [median(a.mv[0], b.mv[0], c.mv[0]), median(a.mv[1], b.mv[1], c.mv[1])],
		}
	}

	/// P_Skip motion (8.4.1.1).
	fn skip_mv(&self) -> [i16; 2] {
		let cur = MbInfo::default();
		let a = self.mv_nb(&cur, 0, -1, 0, 0, 0);
		let b = self.mv_nb(&cur, 0, 0, -1, 0, 0);
		let zero = !a.avail
			|| !b.avail
			|| (a.ref_idx == 0 && a.mv == [0, 0])
			|| (b.ref_idx == 0 && b.mv == [0, 0]);
		if zero { [0, 0] } else { self.predict_mv(&cur, 0, 0, 0, 4, 0, Shape::S16x16, 0) }
	}

	/// Spatial direct prediction (8.4.1.2.2) of the whole macroblock with
	/// `direct_8x8_inference_flag`, into `cur`'s references and vectors.
	fn predict_direct(&self, cur: &mut MbInfo) {
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
					mvp[list] = self.predict_mv(cur, list, 0, 0, 4, refs[list], Shape::S16x16, 0);
				}
			}
		}
		let addr = self.mby * self.f.mb_w + self.mbx;
		let col = self.p.l1.first().and_then(|r| r.motion.get(addr)).copied().unwrap_or_default();
		for b8 in 0..4 {
			let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
			for list in 0..2 {
				cur.ref_idx[list][b8] = if refs[list] >= 0 { refs[list] } else { -1 };
			}
			for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
				let idx = (by + dy) * 4 + bx + dx;
				// direct_8x8_inference: the corner block of the 8x8.
				let (xc, yc) = (((bx + dx) / 2) * 3, ((by + dy) / 2) * 3);
				let blk = yc * 4 + xc;
				let cb8 = (yc / 2) * 2 + xc / 2;
				let col_zero = !col.intra && {
					let (l, r) = if col.ref_idx[0][cb8] >= 0 {
						(0, col.ref_idx[0][cb8])
					} else {
						(1, col.ref_idx[1][cb8])
					};
					let mv = col.mv[l][blk];
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
	}

	// -----------------------------------------------------------------------
	// Sample helpers

	fn diff4(&self, comp: usize, x: usize, y: usize) -> [i32; 16] {
		let (s, r) = match comp {
			0 => (&self.src.y, &self.f.y),
			1 => (&self.src.cb, &self.f.cb),
			_ => (&self.src.cr, &self.f.cr),
		};
		let mut d = [0i32; 16];
		for j in 0..4 {
			for i in 0..4 {
				let k = (y + j) * s.width + x + i;
				d[j * 4 + i] = i32::from(s.data[k]) - i32::from(r.data[k]);
			}
		}
		d
	}

	fn diff8(&self, x: usize, y: usize) -> [i32; 64] {
		let (s, r) = (&self.src.y, &self.f.y);
		let mut d = [0i32; 64];
		for j in 0..8 {
			for i in 0..8 {
				let k = (y + j) * s.width + x + i;
				d[j * 8 + i] = i32::from(s.data[k]) - i32::from(r.data[k]);
			}
		}
		d
	}

	fn satd8(&self, x: usize, y: usize) -> u32 {
		(0..4).map(|q| satd4(&self.diff4(0, x + (q % 2) * 4, y + (q / 2) * 4))).sum()
	}

	fn ssd(&self) -> u64 {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mut sum = 0u64;
		let mut plane = |s: &Plane, r: &Plane, x: usize, y: usize, n: usize| {
			for j in 0..n {
				let k = (y + j) * s.width + x;
				for i in 0..n {
					let d = i64::from(s.data[k + i]) - i64::from(r.data[k + i]);
					sum += (d * d) as u64;
				}
			}
		};
		plane(&self.src.y, &self.f.y, px, py, 16);
		plane(&self.src.cb, &self.f.cb, px / 2, py / 2, 8);
		plane(&self.src.cr, &self.f.cr, px / 2, py / 2, 8);
		sum
	}

	fn take_recon(&self) -> Box<[u8; 384]> {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mut out = Box::new([0u8; 384]);
		for j in 0..16 {
			let k = (py + j) * self.f.y.width + px;
			out[j * 16..j * 16 + 16].copy_from_slice(&self.f.y.data[k..k + 16]);
		}
		for (c, plane) in [&self.f.cb, &self.f.cr].into_iter().enumerate() {
			for j in 0..8 {
				let k = (py / 2 + j) * plane.width + px / 2;
				out[256 + 64 * c + j * 8..256 + 64 * c + j * 8 + 8].copy_from_slice(&plane.data[k..k + 8]);
			}
		}
		out
	}

	fn put_recon(&mut self, r: &[u8; 384], px: usize, py: usize) {
		for j in 0..16 {
			let k = (py + j) * self.f.y.width + px;
			self.f.y.data[k..k + 16].copy_from_slice(&r[j * 16..j * 16 + 16]);
		}
		for c in 0..2 {
			let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
			for j in 0..8 {
				let k = (py / 2 + j) * plane.width + px / 2;
				plane.data[k..k + 8].copy_from_slice(&r[256 + 64 * c + j * 8..256 + 64 * c + j * 8 + 8]);
			}
		}
	}

	fn base_info(&self, kind: MbKind) -> MbInfo {
		let mut info = MbInfo { slice: 1, kind, ipred: [2; 16], ..MbInfo::default() };
		info.qp = self.qp as i8;
		info.qpc = [self.qpc as i8; 2];
		info.ref_idx = [[-1; 4]; 2];
		info
	}

	fn finish(&self, info: MbInfo, syn: MbSyn, res: Box<Residual>) -> Coded {
		let bits = self.price(&info, &syn, &res);
		Coded { info, syn, res, ssd: self.ssd(), recon: self.take_recon(), bits }
	}

	fn rd_cost(&self, c: &Coded) -> f64 {
		c.ssd as f64 + self.lambda2 * c.bits
	}

	/// Trellis λ: 1/256 bit in units of squared level error at `qp`.
	fn trellis_lambda(&self, qp: i32) -> f64 {
		let step = 0.625 * 2f64.powf(f64::from(qp) / 6.0);
		self.lambda2 / (step * step) / 256.0
	}

	fn trellis_on(&self) -> bool {
		self.p.trellis && self.cabac()
	}

	/// Quantises a 4x4 block (raster) from `start`: trellis with CABAC
	/// when enabled, deadzone otherwise. `cbf_inc` prices the block's
	/// coded_block_flag in the trellis.
	#[allow(clippy::too_many_arguments)]
	fn quant4(
		&self,
		coef: &mut [i32; 16],
		quant: &Quant,
		qp: i32,
		start: usize,
		cat: BlockCat,
		cbf_inc: usize,
	) {
		let SliceOut::Cabac(cw) = &self.out else {
			quant.block4(coef, start == 1);
			return;
		};
		if !self.p.trellis {
			quant.block4(coef, start == 1);
			return;
		}
		let n = 16 - start;
		let mut z = [0f64; 16];
		for k in 0..n {
			let pos = usize::from(ZIGZAG_4X4[k + start]);
			z[k] = quant.real4(coef[pos], pos);
		}
		let mut lv = [0i32; 16];
		trellis_cabac(&z[..n], cat, Some(cbf_inc), cw.states(), self.trellis_lambda(qp), &mut lv);
		for k in 0..n {
			coef[usize::from(ZIGZAG_4X4[k + start])] = lv[k];
		}
	}

	fn quant8(&self, coef: &mut [i32; 64], quant: &Quant) {
		let SliceOut::Cabac(cw) = &self.out else {
			quant.block8(coef);
			return;
		};
		if !self.p.trellis {
			quant.block8(coef);
			return;
		}
		let mut z = [0f64; 64];
		for (k, zk) in z.iter_mut().enumerate() {
			let pos = usize::from(ZIGZAG_8X8[k]);
			*zk = quant.real8(coef[pos], pos);
		}
		let mut lv = [0i32; 64];
		trellis_cabac(&z, BlockCat::Luma8x8, None, cw.states(), self.trellis_lambda(self.qp), &mut lv);
		for k in 0..64 {
			coef[usize::from(ZIGZAG_8X8[k])] = lv[k];
		}
	}

	/// Approximate coded_block_flag ctxIdxInc of a luma 4x4 block while
	/// the macroblock is being built (its cbp counted as all set).
	fn cbf_inc_luma(&self, info: &MbInfo, x4: usize, y4: usize) -> usize {
		if !self.trellis_on() {
			return 0;
		}
		let mut tmp = *info;
		tmp.cbp |= 15;
		self.writer().cbf_luma_inc(&tmp, x4 as i32, y4 as i32)
	}

	// -----------------------------------------------------------------------
	// Residual coding shared by every macroblock type

	/// Chroma prediction is in the frame: transform, quantise and
	/// reconstruct both components; sets `cbp` bits 4-5, `dc_cbf` bits 1-2
	/// and `nnz[16..]`.
	fn code_chroma(&mut self, info: &mut MbInfo, res: &mut Residual, intra: bool) {
		let (cx, cy) = (self.mbx * 8, self.mby * 8);
		let quant = Quant::new(self.qpc, intra);
		let mut any_dc = false;
		let mut any_ac = false;
		let mut ac_nnz = [[0u8; 4]; 2];
		let trellis = self.trellis_on();
		for c in 0..2 {
			let mut dc = [0i32; 4];
			for blk in 0..4 {
				let d = self.diff4(c + 1, cx + (blk % 2) * 4, cy + (blk / 2) * 4);
				let mut coef = fdct4(&d);
				dc[blk] = coef[0];
				let inc = if trellis {
					let mut tmp = *info;
					tmp.cbp |= 0x20;
					self.writer().cbf_chroma_ac_inc(&tmp, c, (blk % 2) as i32, (blk / 2) as i32)
				} else {
					0
				};
				self.quant4(&mut coef, &quant, self.qpc, 1, BlockCat::ChromaAc, inc);
				coef[0] = 0;
				res.chroma_ac[c][blk] = coef;
				info.nnz[16 + 8 * c + blk] = coef.iter().filter(|&&v| v != 0).count() as u8;
			}
			let mut dcq = hadamard2(&dc);
			any_dc |= quant.dc(&mut dcq);
			res.chroma_dc[c][..4].copy_from_slice(&dcq);
			if !intra {
				let score: u32 = (0..4).map(|b| decimate_score(&scan_ac(&res.chroma_ac[c][b]))).sum();
				if score < 7 {
					for b in 0..4 {
						res.chroma_ac[c][b] = [0; 16];
					}
				}
			}
			for blk in 0..4 {
				let n = res.chroma_ac[c][blk].iter().filter(|&&v| v != 0).count() as u8;
				ac_nnz[c][blk] = n;
				any_ac |= n != 0;
			}
		}
		let cbp_c: u8 = if any_ac {
			2
		} else if any_dc {
			1
		} else {
			0
		};
		info.cbp |= cbp_c << 4;
		info.dc_cbf &= 1;
		for c in 0..2 {
			if cbp_c != 0 && res.chroma_dc[c][..4].iter().any(|&v| v != 0) {
				info.dc_cbf |= 2 << c;
			}
			for blk in 0..4 {
				info.nnz[16 + 8 * c + blk] = if cbp_c == 2 { ac_nnz[c][blk] } else { 0 };
				if cbp_c != 2 {
					res.chroma_ac[c][blk] = [0; 16];
				}
			}
		}
		if cbp_c == 0 {
			res.chroma_dc = [[0; 8]; 2];
			return;
		}
		// Reconstruct exactly as the decoder does.
		for c in 0..2 {
			let list = if intra { 1 + c } else { 4 + c };
			let scale = &self.dq.l4[list][(self.qpc % 6) as usize];
			let dc = recon::chroma_dc_dequant(&res.chroma_dc[c], self.qpc, scale[0], 8);
			let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
			for blk in 0..4 {
				let mut block = res.chroma_ac[c][blk];
				let nz = info.nnz[16 + 8 * c + blk] != 0;
				if nz {
					recon::dequant4(&mut block, self.qpc, scale, true, 8);
				}
				block[0] = dc[blk];
				if nz || dc[blk] != 0 {
					recon::idct4_add(&mut plane.d8(), cx + (blk % 2) * 4, cy + (blk / 2) * 4, &block);
				}
			}
		}
	}

	/// Inter luma residual with the 4x4 transform (with x264-style
	/// decimation). Sets `cbp` bits 0-3 and `nnz[..16]`.
	fn code_inter_luma(&mut self, info: &mut MbInfo, res: &mut Residual) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let quant = Quant::new(self.qp, false);
		let mut scores = [0u32; 4];
		for blk in 0..16 {
			let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
			let raster = y4 * 4 + x4;
			let mut coef = fdct4(&self.diff4(0, px + x4 * 4, py + y4 * 4));
			let inc = self.cbf_inc_luma(info, x4, y4);
			self.quant4(&mut coef, &quant, self.qp, 0, BlockCat::Luma4x4, inc);
			res.luma[raster] = coef;
			info.nnz[raster] = coef.iter().filter(|&&v| v != 0).count() as u8;
			scores[(y4 / 2) * 2 + x4 / 2] += decimate_score(&scan_full(&coef));
		}
		let total: u32 = scores.iter().map(|&s| s.min(9)).sum();
		info.cbp &= !15;
		for raster in 0..16 {
			let (x4, y4) = (raster % 4, raster / 4);
			let b8 = (y4 / 2) * 2 + x4 / 2;
			if scores[b8] < 4 || total < 6 {
				res.luma[raster] = [0; 16];
			}
			let n = res.luma[raster].iter().filter(|&&v| v != 0).count() as u8;
			info.nnz[raster] = n;
			if n != 0 {
				info.cbp |= 1 << b8;
			}
		}
		let scale = &self.dq.l4[3][(self.qp % 6) as usize];
		for raster in 0..16 {
			if info.nnz[raster] != 0 {
				let mut block = res.luma[raster];
				recon::dequant4(&mut block, self.qp, scale, false, 8);
				recon::idct4_add(&mut self.f.y.d8(), px + (raster % 4) * 4, py + (raster / 4) * 4, &block);
			}
		}
	}

	/// `nnz` of an 8x8-transform block as the decoder records it: the
	/// block's count in all four 4x4 entries with CABAC, the count of each
	/// interleaved 4x4 sub-block with CAVLC.
	fn set_nnz8(&self, info: &mut MbInfo, b8: usize, coef: &[i32; 64]) {
		let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
		if self.cabac() {
			let n = coef.iter().filter(|&&v| v != 0).count() as u8;
			for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
				info.nnz[(y4 + dy) * 4 + x4 + dx] = n;
			}
		} else {
			for i in 0..4 {
				let (bx, by) = (usize::from(BLK_XY[b8 * 4 + i].0), usize::from(BLK_XY[b8 * 4 + i].1));
				let n = (0..16).filter(|&k| coef[usize::from(ZIGZAG_8X8[4 * k + i])] != 0).count() as u8;
				info.nnz[by * 4 + bx] = n;
			}
		}
	}

	/// Inter luma residual with the 8x8 transform (x264 decimation of 8x8
	/// blocks). Sets `t8x8`, `cbp` bits 0-3 and `nnz[..16]`.
	fn code_inter_luma8(&mut self, info: &mut MbInfo, res: &mut Residual) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let quant = Quant::new(self.qp, false);
		info.t8x8 = true;
		let mut scores = [0u32; 4];
		for b8 in 0..4 {
			let mut coef = fdct8(&self.diff8(px + (b8 % 2) * 8, py + (b8 / 2) * 8));
			self.quant8(&mut coef, &quant);
			let scan: [i32; 64] = std::array::from_fn(|k| coef[usize::from(ZIGZAG_8X8[k])]);
			scores[b8] = decimate_score64(&scan);
			res.luma8[b8] = coef;
		}
		let total: u32 = scores.iter().map(|&s| s.min(9)).sum();
		info.cbp &= !15;
		for b8 in 0..4 {
			if scores[b8] < 4 || total < 6 {
				res.luma8[b8] = [0; 64];
			}
			let coef = res.luma8[b8];
			self.set_nnz8(info, b8, &coef);
			if coef.iter().any(|&v| v != 0) {
				info.cbp |= 1 << b8;
			}
		}
		let scale = &self.dq.l8[1][(self.qp % 6) as usize];
		for b8 in 0..4 {
			if info.cbp & (1 << b8) != 0 {
				let mut block = res.luma8[b8];
				recon::dequant8(&mut block, self.qp, scale, 8);
				recon::idct8_add(&mut self.f.y.d8(), px + (b8 % 2) * 8, py + (b8 / 2) * 8, &block);
			}
		}
	}

	// -----------------------------------------------------------------------
	// Intra

	/// Chroma intra mode by SATD over both components (syntax value: 0 DC,
	/// 1 horizontal, 2 vertical, 3 plane).
	fn choose_chroma_mode(&mut self, av: Avail) -> u8 {
		let (cx, cy) = (self.mbx * 8, self.mby * 8);
		let mut best = (u32::MAX, 0u8);
		for syntax in 0..4u8 {
			let mode = chroma_mode(syntax);
			if !block_mode_ok(mode, av) {
				continue;
			}
			let mut cost = 0;
			for c in 0..2 {
				let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
				recon::pred_block(&mut plane.d8(), cx, cy, 8, 8, mode, av, true);
				for blk in 0..4 {
					cost += satd4(&self.diff4(c + 1, cx + (blk % 2) * 4, cy + (blk / 2) * 4));
				}
			}
			cost += self.lambda * ue_bits(u32::from(syntax));
			if cost < best.0 {
				best = (cost, syntax);
			}
		}
		best.1
	}

	fn predict_chroma(&mut self, syntax: u8, av: Avail) {
		let (cx, cy) = (self.mbx * 8, self.mby * 8);
		let mode = chroma_mode(syntax);
		recon::pred_block(&mut self.f.cb.d8(), cx, cy, 8, 8, mode, av, true);
		recon::pred_block(&mut self.f.cr.d8(), cx, cy, 8, 8, mode, av, true);
	}

	/// Best Intra_16x16 mode by SATD, and its cost.
	fn choose_i16_mode(&mut self, av: Avail) -> (u8, u32) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mut best = (0u8, u32::MAX);
		for mode in 0..4u8 {
			if !block_mode_ok(mode, av) {
				continue;
			}
			recon::pred_block(&mut self.f.y.d8(), px, py, 16, 16, mode, av, false);
			let cost: u32 =
				(0..16).map(|r| satd4(&self.diff4(0, px + (r % 4) * 4, py + (r / 4) * 4))).sum();
			if cost < best.1 {
				best = (mode, cost);
			}
		}
		best
	}

	fn encode_i16(&mut self, mode: u8, chroma: u8) -> Coded {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mb = self.mb_avail();
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		let mut info = self.base_info(MbKind::I16x16);
		let mut res = Box::<Residual>::default();
		recon::pred_block(&mut self.f.y.d8(), px, py, 16, 16, mode, av, false);
		let quant = Quant::new(self.qp, true);
		let mut dc = [0i32; 16];
		let mut any_ac = false;
		for blk in 0..16 {
			let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
			let raster = y4 * 4 + x4;
			let mut coef = fdct4(&self.diff4(0, px + x4 * 4, py + y4 * 4));
			dc[raster] = coef[0];
			let inc = self.cbf_inc_luma(&info, x4, y4);
			self.quant4(&mut coef, &quant, self.qp, 1, BlockCat::LumaAc, inc);
			coef[0] = 0;
			info.nnz[raster] = coef.iter().filter(|&&v| v != 0).count() as u8;
			any_ac |= info.nnz[raster] != 0;
			res.luma[raster] = coef;
		}
		let mut dcq = hadamard4(&dc);
		if quant.dc(&mut dcq) {
			info.dc_cbf |= 1;
		}
		res.dc = dcq;
		let cbp_l: u8 = if any_ac { 15 } else { 0 };
		info.cbp = cbp_l;
		for raster in 0..16 {
			if cbp_l == 0 {
				res.luma[raster] = [0; 16];
			}
			info.nnz[raster] = res.luma[raster].iter().filter(|&&v| v != 0).count() as u8;
		}
		let scale = &self.dq.l4[0][(self.qp % 6) as usize];
		let dcd = recon::luma_dc_dequant(&res.dc, self.qp, scale[0], 8);
		for raster in 0..16 {
			let mut block = res.luma[raster];
			let nz = info.nnz[raster] != 0;
			if nz {
				recon::dequant4(&mut block, self.qp, scale, true, 8);
			}
			block[0] = dcd[raster];
			if nz || dcd[raster] != 0 {
				recon::idct4_add(&mut self.f.y.d8(), px + (raster % 4) * 4, py + (raster / 4) * 4, &block);
			}
		}
		self.predict_chroma(chroma, av);
		info.chroma_pred = chroma;
		self.code_chroma(&mut info, &mut res, true);
		self.finish(info, MbSyn::new(SynType::I16x16 { pred: mode }), res)
	}

	/// Mode-signalling cost in bits of `mode` against the predicted mode.
	fn mode_bits(mode: u8, pred_mode: u8) -> u32 {
		if mode == pred_mode { 1 } else { 4 }
	}

	fn rem_of(mode: u8, pred_mode: u8) -> i8 {
		if mode == pred_mode {
			-1
		} else if mode < pred_mode {
			mode as i8
		} else {
			mode as i8 - 1
		}
	}

	fn encode_i4(&mut self, chroma: u8) -> Coded {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mb = self.mb_avail();
		let mut info = self.base_info(MbKind::I4x4);
		let mut syn = MbSyn::new(SynType::I4x4);
		let mut res = Box::<Residual>::default();
		let quant = Quant::new(self.qp, true);
		let scale = &self.dq.l4[0][(self.qp % 6) as usize];
		for blk in 0..16 {
			let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
			let raster = y4 * 4 + x4;
			let (x, y) = (px + x4 * 4, py + y4 * 4);
			let av = Self::block_avail(x4, y4, 1, mb);
			let pred_mode = self.predicted_intra(&info, x4, y4);
			let mut best = (u32::MAX, 2u8);
			for mode in 0..9u8 {
				if !intra4_mode_ok(mode, av) {
					continue;
				}
				recon::pred4x4(&mut self.f.y.d8(), x, y, mode, av);
				let cost = satd4(&self.diff4(0, x, y)) + self.lambda * Self::mode_bits(mode, pred_mode);
				if cost < best.0 {
					best = (cost, mode);
				}
			}
			let mode = best.1;
			recon::pred4x4(&mut self.f.y.d8(), x, y, mode, av);
			info.ipred[raster] = mode;
			syn.intra_rem[blk] = Self::rem_of(mode, pred_mode);
			let mut coef = fdct4(&self.diff4(0, x, y));
			let inc = self.cbf_inc_luma(&info, x4, y4);
			self.quant4(&mut coef, &quant, self.qp, 0, BlockCat::Luma4x4, inc);
			res.luma[raster] = coef;
			let n = coef.iter().filter(|&&v| v != 0).count() as u8;
			info.nnz[raster] = n;
			if n != 0 {
				info.cbp |= 1 << ((y4 / 2) * 2 + x4 / 2);
				let mut block = coef;
				recon::dequant4(&mut block, self.qp, scale, false, 8);
				recon::idct4_add(&mut self.f.y.d8(), x, y, &block);
			}
		}
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		self.predict_chroma(chroma, av);
		info.chroma_pred = chroma;
		self.code_chroma(&mut info, &mut res, true);
		self.finish(info, syn, res)
	}

	fn encode_i8(&mut self, chroma: u8) -> Coded {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mb = self.mb_avail();
		let mut info = self.base_info(MbKind::I8x8);
		info.t8x8 = true;
		let mut syn = MbSyn::new(SynType::I8x8);
		let mut res = Box::<Residual>::default();
		let quant = Quant::new(self.qp, true);
		let scale = &self.dq.l8[0][(self.qp % 6) as usize];
		for b8 in 0..4 {
			let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
			let (x, y) = (px + x4 * 4, py + y4 * 4);
			let av = Self::block_avail(x4, y4, 2, mb);
			let pred_mode = self.predicted_intra(&info, x4, y4);
			let mut best = (u32::MAX, 2u8);
			for mode in 0..9u8 {
				if !intra4_mode_ok(mode, av) {
					continue;
				}
				recon::pred8x8l(&mut self.f.y.d8(), x, y, mode, av);
				let cost = self.satd8(x, y) + self.lambda * Self::mode_bits(mode, pred_mode);
				if cost < best.0 {
					best = (cost, mode);
				}
			}
			let mode = best.1;
			recon::pred8x8l(&mut self.f.y.d8(), x, y, mode, av);
			for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
				info.ipred[(y4 + dy) * 4 + x4 + dx] = mode;
			}
			syn.intra_rem[b8] = Self::rem_of(mode, pred_mode);
			let mut coef = fdct8(&self.diff8(x, y));
			self.quant8(&mut coef, &quant);
			res.luma8[b8] = coef;
			self.set_nnz8(&mut info, b8, &coef);
			if coef.iter().any(|&v| v != 0) {
				info.cbp |= 1 << b8;
				let mut block = coef;
				recon::dequant8(&mut block, self.qp, scale, 8);
				recon::idct8_add(&mut self.f.y.d8(), x, y, &block);
			}
		}
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		self.predict_chroma(chroma, av);
		info.chroma_pred = chroma;
		self.code_chroma(&mut info, &mut res, true);
		self.finish(info, syn, res)
	}

	/// Best intra candidate (I16x16, I4x4 or I8x8) by RD cost.
	fn best_intra(&mut self) -> Coded {
		let mb = self.mb_avail();
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		let chroma = self.choose_chroma_mode(av);
		let (mode, _) = self.choose_i16_mode(av);
		let mut best = self.encode_i16(mode, chroma);
		let i4 = self.encode_i4(chroma);
		if self.rd_cost(&i4) < self.rd_cost(&best) {
			best = i4;
		}
		if self.p.transform_8x8 {
			let i8 = self.encode_i8(chroma);
			if self.rd_cost(&i8) < self.rd_cost(&best) {
				best = i8;
			}
		}
		best
	}

	// -----------------------------------------------------------------------
	// Inter prediction

	fn list(&self, l: usize) -> &[&'a Reference] {
		if l == 0 { &self.p.l0 } else { &self.p.l1 }
	}

	/// Motion-compensates rectangle (x4, y4, w4, h4) from `info`'s
	/// references and vectors into the frame (the decoder's `mc_rect`).
	fn mc_rect(&mut self, info: &MbInfo, x4: usize, y4: usize, w4: usize, h4: usize) {
		let idx = y4 * 4 + x4;
		let b8 = (idx / 8) * 2 + (idx % 4) / 2;
		let refs = [info.ref_idx[0][b8], info.ref_idx[1][b8]];
		let (bw, bh) = (w4 * 4, h4 * 4);
		let px = self.mbx * 16 + x4 * 4;
		let py = self.mby * 16 + y4 * 4;
		let mut luma = [[0u8; 256]; 2];
		let mut chroma = [[[0u8; 64]; 2]; 2];
		for list in 0..2 {
			if refs[list] < 0 {
				continue;
			}
			let Some(r) = self.list(list).get(refs[list] as usize).copied() else { continue };
			let mv = info.mv[list][idx];
			let (mx, my) = (i32::from(mv[0]), i32::from(mv[1]));
			recon::mc_luma(
				&r.y.s8(),
				px as i32 + (mx >> 2),
				py as i32 + (my >> 2),
				mx & 3,
				my & 3,
				bw,
				bh,
				&mut luma[list],
			);
			let (cx, cy) = ((px / 2) as i32 + (mx >> 3), (py / 2) as i32 + (my >> 3));
			for (c, rp) in [&r.cb, &r.cr].into_iter().enumerate() {
				recon::mc_chroma(&rp.s8(), cx, cy, mx & 7, my & 7, bw / 2, bh / 2, &mut chroma[c][list]);
			}
		}
		let (l0, l1) = (&luma[0][..bw * bh], &luma[1][..bw * bh]);
		recon::store_prediction(
			&mut self.f.y.d8(),
			px,
			py,
			bw,
			bh,
			(refs[0] >= 0).then_some(l0),
			(refs[1] >= 0).then_some(l1),
			Weighting::Default,
		);
		for c in 0..2 {
			let n = bw * bh / 4;
			let (l0, l1) = (&chroma[c][0][..n], &chroma[c][1][..n]);
			let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
			recon::store_prediction(
				&mut plane.d8(),
				px / 2,
				py / 2,
				bw / 2,
				bh / 2,
				(refs[0] >= 0).then_some(l0),
				(refs[1] >= 0).then_some(l1),
				Weighting::Default,
			);
		}
	}

	/// Whether the prediction in the frame leaves a residual that
	/// quantises to nothing after decimation, so skipping loses nothing a
	/// coded macroblock would keep.
	fn skip_is_free(&self) -> bool {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let quant = Quant::new(self.qp, false);
		let mut score = 0;
		for raster in 0..16 {
			let mut coef = fdct4(&self.diff4(0, px + (raster % 4) * 4, py + (raster / 4) * 4));
			quant.block4(&mut coef, false);
			score += decimate_score(&scan_full(&coef));
			if score >= 6 {
				return false;
			}
		}
		let cq = Quant::new(self.qpc, false);
		for c in 0..2 {
			let mut dc = [0i32; 4];
			let mut ac_score = 0;
			for blk in 0..4 {
				let mut coef = fdct4(&self.diff4(c + 1, px / 2 + (blk % 2) * 4, py / 2 + (blk / 2) * 4));
				dc[blk] = coef[0];
				cq.block4(&mut coef, true);
				coef[0] = 0;
				ac_score += decimate_score(&scan_ac(&coef));
			}
			let mut dcq = hadamard2(&dc);
			if cq.dc(&mut dcq) || ac_score >= 7 {
				return false;
			}
		}
		true
	}

	fn encode_p_skip(&mut self) -> Coded {
		let mv = self.skip_mv();
		let mut info = self.base_info(MbKind::PSkip);
		info.ref_idx = [[0; 4], [-1; 4]];
		info.mv[0] = [mv; 16];
		self.set_ref_ids(&mut info);
		self.mc_rect(&info, 0, 0, 4, 4);
		self.finish(info, MbSyn::new(SynType::PSkip), Box::default())
	}

	/// B_Skip (`skip`) or B_Direct_16x16 with a residual.
	fn encode_direct(&mut self, skip: bool, t8: bool) -> Coded {
		let mut info = self.base_info(if skip { MbKind::BSkip } else { MbKind::Inter });
		info.direct16 = true;
		info.direct8 = 0xF;
		self.predict_direct(&mut info);
		self.set_ref_ids(&mut info);
		for b8 in 0..4 {
			self.mc_rect(&info, (b8 % 2) * 2, (b8 / 2) * 2, 2, 2);
		}
		if skip {
			return self.finish(info, MbSyn::new(SynType::BSkip), Box::default());
		}
		let mut res = Box::<Residual>::default();
		self.code_residual_inter(&mut info, &mut res, t8);
		self.finish(info, MbSyn::new(SynType::BDirect), res)
	}

	/// `ref_id` of every 8x8 block: the referenced picture's identity.
	fn set_ref_ids(&self, info: &mut MbInfo) {
		for list in 0..2 {
			for b8 in 0..4 {
				let r = info.ref_idx[list][b8];
				info.ref_id[list][b8] =
					if r < 0 { 0 } else { self.list(list).get(r as usize).map_or(0, |p| p.id) };
			}
		}
	}

	fn code_residual_inter(&mut self, info: &mut MbInfo, res: &mut Residual, t8: bool) {
		if t8 {
			self.code_inter_luma8(info, res);
		} else {
			self.code_inter_luma(info, res);
		}
		self.code_chroma(info, res, false);
		if info.cbp & 15 == 0 {
			// transform_size_8x8_flag is only sent with luma coefficients.
			info.t8x8 = false;
		}
	}

	/// Codes an inter macroblock of type `t` whose partitions (of
	/// `shape`) have the given motion; `t8` selects the 8x8 transform.
	fn encode_inter(&mut self, t: SynType, shape: Shape, pm: &[PartMotion], t8: bool) -> Coded {
		let mut info = self.base_info(MbKind::Inter);
		let mut syn = MbSyn::new(t);
		let ps = parts(shape);
		// References first (all lists), as they are parsed.
		for (p, &(x4, y4, w4, h4)) in ps.iter().enumerate() {
			for list in 0..2 {
				if pm[p].pred.uses(list) {
					for y in (y4..y4 + h4).step_by(2) {
						for x in (x4..x4 + w4).step_by(2) {
							info.ref_idx[list][(y / 2) * 2 + x / 2] = pm[p].refs[list];
						}
					}
					syn.ref_idx[list][p] = pm[p].refs[list];
				}
			}
		}
		for list in 0..2 {
			for (p, &(x4, y4, w4, h4)) in ps.iter().enumerate() {
				if !pm[p].pred.uses(list) {
					continue;
				}
				let mv = pm[p].mvs[list];
				let mvp = self.predict_mv(&info, list, x4, y4, w4, pm[p].refs[list], shape, p);
				let mvd = [i32::from(mv[0]) - i32::from(mvp[0]), i32::from(mv[1]) - i32::from(mvp[1])];
				syn.mvd[list][p] = mvd;
				for y in y4..y4 + h4 {
					for x in x4..x4 + w4 {
						info.mv[list][y * 4 + x] = mv;
						info.mvd[list][y * 4 + x] =
							[mvd[0].unsigned_abs().min(255) as u8, mvd[1].unsigned_abs().min(255) as u8];
					}
				}
			}
		}
		self.set_ref_ids(&mut info);
		for &(x4, y4, w4, h4) in ps {
			self.mc_rect(&info, x4, y4, w4, h4);
		}
		let mut res = Box::<Residual>::default();
		self.code_residual_inter(&mut info, &mut res, t8);
		self.finish(info, syn, res)
	}

	/// The better of the 4x4- and (when enabled) 8x8-transform codings.
	fn encode_inter_best(&mut self, t: SynType, shape: Shape, pm: &[PartMotion]) -> Coded {
		let c4 = self.encode_inter(t, shape, pm, false);
		if !self.p.transform_8x8 || c4.info.cbp & 15 == 0 {
			return c4;
		}
		let c8 = self.encode_inter(t, shape, pm, true);
		if self.rd_cost(&c8) < self.rd_cost(&c4) { c8 } else { c4 }
	}

	fn ref_bits(&self, list: usize, r: i8) -> u32 {
		match self.list(list).len() {
			0 | 1 => 0,
			2 => 1,
			_ => ue_bits(r as u32),
		}
	}

	fn mv_bits(mv: [i32; 2], mvp: [i16; 2]) -> u32 {
		se_bits(mv[0] - i32::from(mvp[0])) + se_bits(mv[1] - i32::from(mvp[1]))
	}

	/// Diamond integer search followed by half- and quarter-sample
	/// refinement (x264 `--me dia --subme 7`-like) in reference `r`.
	/// Returns the vector (in quarter samples) and its SATD + λ·bits cost.
	#[allow(clippy::too_many_arguments)]
	fn search(
		&self,
		r: &Reference,
		x4: usize,
		y4: usize,
		bw: usize,
		bh: usize,
		mvp: [i16; 2],
		extra: &[[i16; 2]],
	) -> ([i16; 2], u32) {
		let (px, py) = (self.mbx * 16 + x4 * 4, self.mby * 16 + y4 * 4);
		let lambda = self.lambda;
		let max_y = self.p.max_mv_y;
		let in_range = |mx: i32, my: i32| {
			r.padded.fits(px as i32 + mx, py as i32 + my, bw, bh) && my.abs() <= max_y && mx.abs() <= 2040
		};
		let int_cost = |mx: i32, my: i32| -> u32 {
			r.padded.sad(&self.src.y, px, py, px as i32 + mx, py as i32 + my, bw, bh)
				+ lambda * Self::mv_bits([mx * 4, my * 4], mvp)
		};
		let mut best = (u32::MAX, 0i32, 0i32);
		let round = |v: i16| (i32::from(v) + 2) >> 2;
		let mut starts = vec![[round(mvp[0]), round(mvp[1])], [0, 0]];
		starts.extend(extra.iter().map(|m| [round(m[0]), round(m[1])]));
		for s in starts {
			if in_range(s[0], s[1]) {
				let c = int_cost(s[0], s[1]);
				if c < best.0 {
					best = (c, s[0], s[1]);
				}
			}
		}
		if best.0 == u32::MAX {
			return ([0, 0], u32::MAX);
		}
		for step in [4i32, 2, 1] {
			for _ in 0..64 {
				let (c0, cx, cy) = best;
				let mut moved = false;
				for (dx, dy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
					let (mx, my) = (cx + dx, cy + dy);
					if in_range(mx, my) {
						let c = int_cost(mx, my);
						if c < best.0 {
							best = (c, mx, my);
							moved = true;
						}
					}
				}
				if !moved || best.0 == c0 {
					break;
				}
			}
		}
		let mut q = [best.1 * 4, best.2 * 4];
		let cost = |mv: [i32; 2]| {
			me::satd_mv(&r.padded, &r.half, &self.src.y, px, py, bw, bh, mv)
				+ lambda * Self::mv_bits(mv, mvp)
		};
		let mut qcost = cost(q);
		for step in [2, 1] {
			let centre = q;
			for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
				let mv = [centre[0] + dx * step, centre[1] + dy * step];
				if mv[1].abs() > max_y * 4
					|| !r.padded.fits(
						px as i32 + (mv[0] >> 2) - 3,
						py as i32 + (mv[1] >> 2) - 3,
						bw + 6,
						bh + 6,
					) {
					continue;
				}
				let c = cost(mv);
				if c < qcost {
					qcost = c;
					q = mv;
				}
			}
		}
		([q[0] as i16, q[1] as i16], qcost)
	}

	/// Prediction samples of a block from `r` with `mv` (stride `bw`).
	#[allow(clippy::too_many_arguments)]
	fn pred_block(
		&self,
		r: &Reference,
		x4: usize,
		y4: usize,
		bw: usize,
		bh: usize,
		mv: [i16; 2],
		out: &mut [u8],
	) {
		let (px, py) = (self.mbx * 16 + x4 * 4, self.mby * 16 + y4 * 4);
		let (mx, my) = (i32::from(mv[0]), i32::from(mv[1]));
		recon::mc_luma(
			&r.y.s8(),
			px as i32 + (mx >> 2),
			py as i32 + (my >> 2),
			mx & 3,
			my & 3,
			bw,
			bh,
			out,
		);
	}

	/// SATD of the bi-prediction (rounded average) of two blocks.
	#[allow(clippy::too_many_arguments)]
	fn bi_satd(
		&self,
		r0: &Reference,
		mv0: [i16; 2],
		r1: &Reference,
		mv1: [i16; 2],
		x4: usize,
		y4: usize,
		bw: usize,
		bh: usize,
	) -> u32 {
		let (mut a, mut b) = ([0u8; 256], [0u8; 256]);
		self.pred_block(r0, x4, y4, bw, bh, mv0, &mut a);
		self.pred_block(r1, x4, y4, bw, bh, mv1, &mut b);
		for i in 0..bw * bh {
			a[i] = ((u16::from(a[i]) + u16::from(b[i]) + 1) >> 1) as u8;
		}
		me::satd_block(&self.src.y, self.mbx * 16 + x4 * 4, self.mby * 16 + y4 * 4, &a, bw, bh)
	}

	/// Best motion of partition `part` of `shape` from list `list`, over
	/// the reference indices `refs`. `info` holds the motion of earlier
	/// partitions (for prediction). Returns (ref, mv, cost).
	fn search_part(
		&self,
		info: &MbInfo,
		list: usize,
		shape: Shape,
		part: usize,
		refs: &[i8],
		seeds: &[[i16; 2]],
	) -> (i8, [i16; 2], u32) {
		let (x4, y4, w4, h4) = parts(shape)[part];
		let mut best = (0i8, [0i16; 2], u32::MAX);
		for &r in refs {
			let Some(rf) = self.list(list).get(r as usize).copied() else { continue };
			let mvp = self.predict_mv(info, list, x4, y4, w4, r, shape, part);
			let (mv, cost) = self.search(rf, x4, y4, w4 * 4, h4 * 4, mvp, seeds);
			let cost = cost.saturating_add(self.lambda * self.ref_bits(list, r));
			if cost < best.2 {
				best = (r, mv, cost);
			}
		}
		best
	}

	/// Records a partition's motion in `info` (for the next partition's
	/// prediction).
	fn note_part(info: &mut MbInfo, shape: Shape, part: usize, list: usize, r: i8, mv: [i16; 2]) {
		let (x4, y4, w4, h4) = parts(shape)[part];
		for y in y4..y4 + h4 {
			for x in x4..x4 + w4 {
				info.mv[list][y * 4 + x] = mv;
			}
		}
		for y in (y4..y4 + h4).step_by(2) {
			for x in (x4..x4 + w4).step_by(2) {
				info.ref_idx[list][(y / 2) * 2 + x / 2] = r;
			}
		}
	}

	/// L0-only motion search for every partition of `shape`.
	fn search_p_shape(
		&self,
		shape: Shape,
		refs: &[i8],
		seeds: &[[i16; 2]],
	) -> (Vec<PartMotion>, u32) {
		let mut info = self.base_info(MbKind::Inter);
		let header = match shape {
			Shape::S16x16 => 1,
			Shape::S16x8 | Shape::S8x16 => 3,
			Shape::S8x8 => 5 + 4,
		};
		let mut total = self.lambda * header;
		let mut out = Vec::new();
		for part in 0..parts(shape).len() {
			let (r, mv, cost) = self.search_part(&info, 0, shape, part, refs, seeds);
			if cost == u32::MAX {
				return (out, u32::MAX);
			}
			total = total.saturating_add(cost);
			Self::note_part(&mut info, shape, part, 0, r, mv);
			out.push(PartMotion { pred: Pred::L0, refs: [r, -1], mvs: [mv, [0, 0]] });
		}
		(out, total)
	}

	fn seeds(&self, first: [i16; 2]) -> Vec<[i16; 2]> {
		let mut seeds = vec![first];
		if let Some(m) = self.nb_mb(0, -1).filter(|m| !m.is_intra()) {
			seeds.push(m.mv[0][12]);
		}
		if let Some(m) = self.nb_mb(-1, 0).filter(|m| !m.is_intra()) {
			seeds.push(m.mv[0][3]);
		}
		seeds
	}

	/// Intra when its SATD estimate competes with the inter match.
	fn try_intra(&mut self, best: Coded, inter_satd: u32) -> Coded {
		let mb = self.mb_avail();
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		let (_, i16_satd) = self.choose_i16_mode(av);
		if inter_satd == u32::MAX || i16_satd + 6 * self.lambda < inter_satd + inter_satd / 4 {
			let intra = self.best_intra();
			if self.rd_cost(&intra) < self.rd_cost(&best) {
				return intra;
			}
		}
		best
	}

	fn decide_p(&mut self) -> Coded {
		let skip = self.encode_p_skip();
		let skip_mv = skip.info.mv[0][0];
		if self.skip_is_free() {
			return skip;
		}
		let mut seeds = self.seeds(skip_mv);
		let all_refs: Vec<i8> = (0..self.p.l0.len() as i8).collect();
		let (pm16, cost16) = self.search_p_shape(Shape::S16x16, &all_refs, &seeds);
		let mut best_shape = (Shape::S16x16, pm16, cost16);
		// Smaller partitions only when the 16x16 match is poor; they try
		// the 16x16 winner's reference and the nearest one.
		if cost16 != u32::MAX && cost16 > 24 * self.lambda + 256 {
			let r16 = best_shape.1[0].refs[0];
			let mut refs = vec![r16];
			if r16 != 0 {
				refs.push(0);
			}
			seeds.push(best_shape.1[0].mvs[0]);
			for shape in [Shape::S8x8, Shape::S16x8, Shape::S8x16] {
				let (pm, cost) = self.search_p_shape(shape, &refs, &seeds);
				if cost < best_shape.2 {
					best_shape = (shape, pm, cost);
				}
			}
		}
		let mut best = skip;
		if best_shape.2 != u32::MAX {
			let inter = self.encode_inter_best(SynType::P(best_shape.0), best_shape.0, &best_shape.1);
			if self.rd_cost(&inter) < self.rd_cost(&best) {
				best = inter;
			}
		}
		self.try_intra(best, best_shape.2)
	}

	/// Best of L0, L1 and bi-prediction for every partition of `shape`.
	fn search_b_shape(
		&self,
		shape: Shape,
		refs0: &[i8],
		seeds: &[[i16; 2]],
	) -> (Vec<PartMotion>, u32) {
		let mut info = self.base_info(MbKind::Inter);
		let header = if shape == Shape::S16x16 { 3 } else { 6 };
		let mut total = self.lambda * header;
		let mut out = Vec::new();
		let refs1: Vec<i8> = (0..self.p.l1.len() as i8).take(1).collect();
		for part in 0..parts(shape).len() {
			let (x4, y4, w4, h4) = parts(shape)[part];
			let (r0, mv0, c0) = self.search_part(&info, 0, shape, part, refs0, seeds);
			let (r1, mv1, c1) = self.search_part(&info, 1, shape, part, &refs1, seeds);
			if c0 == u32::MAX && c1 == u32::MAX {
				return (out, u32::MAX);
			}
			let mut choice = if c0 <= c1 {
				(PartMotion { pred: Pred::L0, refs: [r0, -1], mvs: [mv0, [0, 0]] }, c0)
			} else {
				(PartMotion { pred: Pred::L1, refs: [-1, r1], mvs: [[0, 0], mv1] }, c1)
			};
			if c0 != u32::MAX && c1 != u32::MAX {
				let (f0, f1) = (self.p.l0[r0 as usize], self.p.l1[r1 as usize]);
				let mvp0 = self.predict_mv(&info, 0, x4, y4, w4, r0, shape, part);
				let mvp1 = self.predict_mv(&info, 1, x4, y4, w4, r1, shape, part);
				let bits = Self::mv_bits([i32::from(mv0[0]), i32::from(mv0[1])], mvp0)
					+ Self::mv_bits([i32::from(mv1[0]), i32::from(mv1[1])], mvp1)
					+ self.ref_bits(0, r0)
					+ self.ref_bits(1, r1);
				let cb = self.bi_satd(f0, mv0, f1, mv1, x4, y4, w4 * 4, h4 * 4) + self.lambda * (bits + 1);
				if cb < choice.1 {
					choice = (PartMotion { pred: Pred::Bi, refs: [r0, r1], mvs: [mv0, mv1] }, cb);
				}
			}
			let pm = choice.0;
			for list in 0..2 {
				if pm.pred.uses(list) {
					Self::note_part(&mut info, shape, part, list, pm.refs[list], pm.mvs[list]);
				}
			}
			total = total.saturating_add(choice.1);
			out.push(pm);
		}
		(out, total)
	}

	fn decide_b(&mut self) -> Coded {
		// Direct prediction first: B_Skip when its residual vanishes.
		let skip = self.encode_direct(true, false);
		if self.skip_is_free() {
			return skip;
		}
		let mut best = skip;
		let direct = self.encode_direct(false, false);
		let direct_luma = direct.info.cbp & 15 != 0;
		if self.rd_cost(&direct) < self.rd_cost(&best) {
			best = direct;
		}
		if self.p.transform_8x8 && direct_luma {
			let d8 = self.encode_direct(false, true);
			if self.rd_cost(&d8) < self.rd_cost(&best) {
				best = d8;
			}
		}
		let seeds = self.seeds([0, 0]);
		let refs0: Vec<i8> = (0..self.p.l0.len() as i8).collect();
		let (pm16, cost16) = self.search_b_shape(Shape::S16x16, &refs0, &seeds);
		let mut best_shape = (Shape::S16x16, pm16, cost16);
		if cost16 != u32::MAX && cost16 > 24 * self.lambda + 256 {
			let r16 = best_shape.1[0].refs[0].max(0);
			let refs = vec![r16];
			for shape in [Shape::S16x8, Shape::S8x16] {
				let (pm, cost) = self.search_b_shape(shape, &refs, &seeds);
				if cost < best_shape.2 {
					best_shape = (shape, pm, cost);
				}
			}
		}
		if best_shape.2 != u32::MAX {
			let pm = best_shape.1.clone();
			let t = b_mb_type(best_shape.0, pm[0].pred, pm.get(1).map_or(pm[0].pred, |p| p.pred));
			let inter = self.encode_inter_best(SynType::B(t), best_shape.0, &pm);
			if self.rd_cost(&inter) < self.rd_cost(&best) {
				best = inter;
			}
		}
		self.try_intra(best, best_shape.2)
	}
}

/// Chroma prediction mode for `pred_block` from the syntax value.
fn chroma_mode(syntax: u8) -> u8 {
	match syntax {
		0 => 2,
		1 => 1,
		2 => 0,
		_ => 3,
	}
}

fn intra4_mode_ok(mode: u8, a: Avail) -> bool {
	match mode {
		0 | 3 | 7 => a.top,
		1 | 8 => a.left,
		4..=6 => a.top && a.left && a.top_left,
		_ => true,
	}
}

fn block_mode_ok(mode: u8, a: Avail) -> bool {
	match mode {
		0 => a.top,
		1 => a.left,
		3 => a.top && a.left && a.top_left,
		_ => true,
	}
}

/// A raster 4x4 block in zig-zag order.
fn scan_full(b: &[i32; 16]) -> [i32; 16] {
	std::array::from_fn(|k| b[usize::from(ZIGZAG_4X4[k])])
}

/// The 15 AC levels of a raster 4x4 block in zig-zag order.
fn scan_ac(b: &[i32; 16]) -> [i32; 15] {
	std::array::from_fn(|k| b[usize::from(ZIGZAG_4X4[k + 1])])
}
