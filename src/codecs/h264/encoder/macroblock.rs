//! Macroblock analysis, coding and reconstruction (ISO/IEC 14496-10 §7.3.5
//! syntax, written instead of read).
//!
//! Every candidate is coded for real: prediction, transform, quantisation,
//! reconstruction with the decoder's own `recon` functions, and CAVLC into
//! a scratch writer. The macroblock type is then chosen by rate-distortion
//! cost `SSD + λ·bits` (x264's `--subme 7`-style final decision, without
//! trellis), so the choice always reflects what will actually be sent.
//!
//! Neighbour derivations (availability, intra-mode prediction, nC, motion
//! vector prediction, P_Skip motion) mirror `h264::mb` / `h264::inter`
//! clause for clause; the encoder's reconstruction must equal a decoder's.

use super::me::{self, Padded};
use super::transform::{Quant, decimate_score, fdct4, hadamard2, hadamard4, satd4};
use super::vlc::{cbp_code, write_block};
use crate::codecs::bitstream::{BitWriter, se_bits, ue_bits};
use crate::codecs::h264::consts::{BLK_XY, ZIGZAG_4X4, chroma_qp};
use crate::codecs::h264::decode::Frame;
use crate::codecs::h264::mb::{MbInfo, MbKind, Residual, blk_index};
use crate::codecs::h264::params::Dequant;
use crate::codecs::h264::recon::{self, Avail, Plane, Weighting};

/// Identity of the single reference picture in `MbInfo::ref_id` (only
/// compared within one picture by the deblocking filter).
const REF_ID: u32 = 1;

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
	pub y: Plane,
	pub cb: Plane,
	pub cr: Plane,
	pub padded: Padded,
	pub half: me::HalfPel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
	S16x16,
	S16x8,
	S8x16,
	S8x8,
}

impl Shape {
	/// Partitions as (x4, y4, w4, h4).
	fn parts(self) -> &'static [(usize, usize, usize, usize)] {
		match self {
			Shape::S16x16 => &[(0, 0, 4, 4)],
			Shape::S16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
			Shape::S8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
			Shape::S8x8 => &[(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
		}
	}
}

/// Motion-vector neighbour (8.4.1.3.2), list 0 only.
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
	bits: BitWriter,
	ssd: u64,
	/// Reconstructed samples: 256 luma, 64 Cb, 64 Cr.
	recon: Box<[u8; 384]>,
}

/// Per-picture encoding context.
pub struct PicEncoder<'a> {
	pub src: &'a Source,
	pub f: &'a mut Frame,
	pub reference: Option<&'a Reference>,
	pub dq: &'a Dequant,
	pub qp: i32,
	qpc: i32,
	/// λ for SATD-domain decisions and λ² for SSD-domain RD.
	lambda: u32,
	lambda2: f64,
	/// Largest vertical motion (in luma samples) the level allows.
	max_mv_y: i32,
	pub w: BitWriter,
	skip_run: u32,
	mbx: usize,
	mby: usize,
	/// Statistics for the caller.
	pub skipped: u32,
	pub intra_mbs: u32,
}

impl<'a> PicEncoder<'a> {
	pub fn new(
		src: &'a Source,
		f: &'a mut Frame,
		reference: Option<&'a Reference>,
		dq: &'a Dequant,
		qp: i32,
		max_mv_y: i32,
		w: BitWriter,
	) -> Self {
		// x264's lambda tables: λ = 2^((qp-12)/6), λ² = 0.85·2^((qp-12)/3).
		let lf = 2f64.powf(f64::from(qp - 12) / 6.0);
		Self {
			src,
			f,
			reference,
			dq,
			qp,
			qpc: i32::from(chroma_qp(qp, 0)),
			lambda: lf.round().max(1.0) as u32,
			lambda2: 0.85 * lf * lf,
			max_mv_y,
			w,
			skip_run: 0,
			mbx: 0,
			mby: 0,
			skipped: 0,
			intra_mbs: 0,
		}
	}

	/// Codes every macroblock of the picture (one slice) into `self.w`.
	pub fn encode_slice(&mut self) {
		let p_slice = self.reference.is_some();
		for mby in 0..self.f.mb_h {
			for mbx in 0..self.f.mb_w {
				self.mbx = mbx;
				self.mby = mby;
				let best = if p_slice { self.decide_p() } else { self.decide_i() };
				self.commit(best, p_slice);
			}
		}
		if p_slice && self.skip_run > 0 {
			self.w.ue(self.skip_run);
		}
	}

	fn commit(&mut self, c: Coded, p_slice: bool) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		self.put_recon(&c.recon, px, py);
		let addr = self.mby * self.f.mb_w + self.mbx;
		if c.info.is_intra() {
			self.intra_mbs += 1;
		}
		if c.info.kind == MbKind::PSkip {
			self.skip_run += 1;
			self.skipped += 1;
		} else {
			if p_slice {
				self.w.ue(self.skip_run);
				self.skip_run = 0;
			}
			self.w.append(&c.bits);
		}
		self.f.mbs[addr] = c.info;
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

	fn nb_chroma<'b>(
		&'b self,
		cur: &'b MbInfo,
		comp: usize,
		x2: i32,
		y2: i32,
	) -> Option<(&'b MbInfo, usize)> {
		let (dx, dy) = (x2.div_euclid(2), y2.div_euclid(2));
		let idx = 16 + 4 * comp + (y2.rem_euclid(2) * 2 + x2.rem_euclid(2)) as usize;
		if dx == 0 && dy == 0 {
			return Some((cur, idx));
		}
		self.nb_mb(dx, dy).map(|m| (m, idx))
	}

	fn nc_luma(&self, cur: &MbInfo, x4: i32, y4: i32) -> i32 {
		let get = |n: Option<(&MbInfo, usize)>| n.map(|(m, i)| i32::from(m.nnz[i]));
		combine_nc(get(self.nb4(cur, x4 - 1, y4)), get(self.nb4(cur, x4, y4 - 1)))
	}

	fn nc_chroma(&self, cur: &MbInfo, comp: usize, x2: i32, y2: i32) -> i32 {
		let get = |n: Option<(&MbInfo, usize)>| n.map(|(m, i)| i32::from(m.nnz[i]));
		combine_nc(
			get(self.nb_chroma(cur, comp, x2 - 1, y2)),
			get(self.nb_chroma(cur, comp, x2, y2 - 1)),
		)
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

	/// Intra 4x4 neighbour availability (as the decoder's `block_avail`).
	fn block_avail(x4: usize, y4: usize, mb: [bool; 4]) -> Avail {
		let [l, t, tr, tl] = mb;
		let left = if x4 > 0 { true } else { l };
		let top = if y4 > 0 { true } else { t };
		let top_left = match (x4 > 0, y4 > 0) {
			(true, true) => true,
			(false, true) => l,
			(true, false) => t,
			(false, false) => tl,
		};
		let rx = x4 + 1;
		let top_right = if y4 == 0 {
			if rx < 4 { t } else { tr }
		} else if rx >= 4 {
			false
		} else {
			blk_index(rx, y4 - 1) < blk_index(x4, y4)
		};
		Avail { left, top, top_right, top_left }
	}

	/// Predicted Intra_4x4 mode (8.3.1.1).
	fn predicted_intra4(&self, cur: &MbInfo, x4: usize, y4: usize) -> u8 {
		let mode_of = |n: Option<(&MbInfo, usize)>| -> Option<u8> {
			let (m, idx) = n?;
			Some(if m.kind == MbKind::I4x4 { m.ipred[idx] } else { 2 })
		};
		let (x, y) = (x4 as i32, y4 as i32);
		match (mode_of(self.nb4(cur, x - 1, y)), mode_of(self.nb4(cur, x, y - 1))) {
			(Some(a), Some(b)) => a.min(b),
			_ => 2,
		}
	}

	fn mv_nb(&self, cur: &MbInfo, x: i32, y: i32, px: usize, py: usize) -> Nb {
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
				let r = m.ref_idx[0][(idx / 8) * 2 + (idx % 4) / 2];
				if r < 0 {
					Nb { avail: true, ref_idx: -1, mv: [0, 0] }
				} else {
					Nb { avail: true, ref_idx: r, mv: m.mv[0][idx] }
				}
			}
		}
	}

	/// Motion vector predictor for a list-0 partition with ref_idx 0.
	fn predict_mv(
		&self,
		cur: &MbInfo,
		x4: usize,
		y4: usize,
		w4: usize,
		shape: Shape,
		part: usize,
	) -> [i16; 2] {
		let (x, y) = (x4 as i32, y4 as i32);
		let a = self.mv_nb(cur, x - 1, y, x4, y4);
		let b = self.mv_nb(cur, x, y - 1, x4, y4);
		let mut c = self.mv_nb(cur, x + w4 as i32, y - 1, x4, y4);
		if !c.avail {
			c = self.mv_nb(cur, x - 1, y - 1, x4, y4);
		}
		match (shape, part) {
			(Shape::S16x8, 0) if b.ref_idx == 0 => return b.mv,
			(Shape::S16x8, 1) if a.ref_idx == 0 => return a.mv,
			(Shape::S8x16, 0) if a.ref_idx == 0 => return a.mv,
			(Shape::S8x16, 1) if c.ref_idx == 0 => return c.mv,
			_ => {}
		}
		let (b, c) = if !b.avail && !c.avail && a.avail { (a, a) } else { (b, c) };
		match [a.ref_idx == 0, b.ref_idx == 0, c.ref_idx == 0] {
			[true, false, false] => a.mv,
			[false, true, false] => b.mv,
			[false, false, true] => c.mv,
			_ => [median(a.mv[0], b.mv[0], c.mv[0]), median(a.mv[1], b.mv[1], c.mv[1])],
		}
	}

	/// P_Skip motion (8.4.1.1).
	fn skip_mv(&self) -> [i16; 2] {
		let cur = MbInfo::default();
		let a = self.mv_nb(&cur, -1, 0, 0, 0);
		let b = self.mv_nb(&cur, 0, -1, 0, 0);
		let zero = !a.avail
			|| !b.avail
			|| (a.ref_idx == 0 && a.mv == [0, 0])
			|| (b.ref_idx == 0 && b.mv == [0, 0]);
		if zero { [0, 0] } else { self.predict_mv(&cur, 0, 0, 4, Shape::S16x16, 0) }
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
		info.qp = self.qp as u8;
		info.qpc = [self.qpc as u8; 2];
		if kind == MbKind::I4x4 || kind == MbKind::I16x16 {
			info.ref_idx = [[-1; 4]; 2];
		} else {
			info.ref_idx = [[0; 4], [-1; 4]];
			info.ref_id = [[REF_ID; 4], [0; 4]];
		}
		info
	}

	fn finish(&self, info: MbInfo, bits: BitWriter) -> Coded {
		Coded { info, bits, ssd: self.ssd(), recon: self.take_recon() }
	}

	fn rd_cost(&self, c: &Coded, extra_bits: usize) -> f64 {
		c.ssd as f64 + self.lambda2 * (c.bits.bits() + extra_bits) as f64
	}

	// -----------------------------------------------------------------------
	// Residual coding shared by every macroblock type

	/// Chroma prediction is in the frame: transform, quantise and
	/// reconstruct both components; sets `cbp` bits 4-5 and `nnz[16..]`.
	fn code_chroma(&mut self, info: &mut MbInfo, res: &mut Residual, intra: bool) {
		let (cx, cy) = (self.mbx * 8, self.mby * 8);
		let quant = Quant::new(self.qpc, intra);
		let mut any_dc = false;
		let mut any_ac = false;
		let mut ac_nnz = [[0u8; 4]; 2];
		for c in 0..2 {
			let mut dc = [0i32; 4];
			for blk in 0..4 {
				let d = self.diff4(c + 1, cx + (blk % 2) * 4, cy + (blk / 2) * 4);
				let mut coef = fdct4(&d);
				dc[blk] = coef[0];
				quant.block4(&mut coef, true);
				coef[0] = 0;
				res.chroma_ac[c][blk] = coef;
			}
			let mut dcq = hadamard2(&dc);
			any_dc |= quant.dc(&mut dcq);
			res.chroma_dc[c] = dcq;
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
		for c in 0..2 {
			for blk in 0..4 {
				info.nnz[16 + 4 * c + blk] = if cbp_c == 2 { ac_nnz[c][blk] } else { 0 };
				if cbp_c != 2 {
					res.chroma_ac[c][blk] = [0; 16];
				}
			}
		}
		if cbp_c == 0 {
			return;
		}
		// Reconstruct exactly as the decoder does.
		for c in 0..2 {
			let list = if intra { 1 + c } else { 4 + c };
			let scale = &self.dq.l4[list][(self.qpc % 6) as usize];
			let dc = recon::chroma_dc_dequant(&res.chroma_dc[c], self.qpc, scale[0]);
			let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
			for blk in 0..4 {
				let mut block = res.chroma_ac[c][blk];
				let nz = info.nnz[16 + 4 * c + blk] != 0;
				if nz {
					recon::dequant4(&mut block, self.qpc, scale, true);
				}
				block[0] = dc[blk];
				if nz || dc[blk] != 0 {
					recon::idct4_add(plane, cx + (blk % 2) * 4, cy + (blk / 2) * 4, &block);
				}
			}
		}
	}

	/// Inter luma residual: transform, quantise (with x264-style
	/// decimation) and reconstruct. Sets `cbp` bits 0-3 and `nnz[..16]`.
	fn code_inter_luma(&mut self, info: &mut MbInfo, res: &mut Residual) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let quant = Quant::new(self.qp, false);
		let mut scores = [0u32; 4];
		for raster in 0..16 {
			let (x4, y4) = (raster % 4, raster / 4);
			let mut coef = fdct4(&self.diff4(0, px + x4 * 4, py + y4 * 4));
			quant.block4(&mut coef, false);
			res.luma[raster] = coef;
			scores[(y4 / 2) * 2 + x4 / 2] += decimate_score(&scan_full(&coef));
		}
		let total: u32 = scores.iter().map(|&s| s.min(9)).sum();
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
				recon::dequant4(&mut block, self.qp, scale, false);
				recon::idct4_add(&mut self.f.y, px + (raster % 4) * 4, py + (raster / 4) * 4, &block);
			}
		}
	}

	/// `residual()` for the macroblock (7.3.5.3), CAVLC.
	fn write_residual(&self, w: &mut BitWriter, info: &MbInfo, res: &Residual) {
		let i16 = info.kind == MbKind::I16x16;
		if i16 {
			let nc = self.nc_luma(info, 0, 0);
			write_block(w, nc, &scan_full(&res.dc));
		}
		for b8 in 0..4usize {
			if info.cbp & (1 << b8) == 0 {
				continue;
			}
			for i in 0..4 {
				let (x4, y4) = (usize::from(BLK_XY[b8 * 4 + i].0), usize::from(BLK_XY[b8 * 4 + i].1));
				let raster = y4 * 4 + x4;
				let nc = self.nc_luma(info, x4 as i32, y4 as i32);
				if i16 {
					write_block(w, nc, &scan_ac(&res.luma[raster]));
				} else {
					write_block(w, nc, &scan_full(&res.luma[raster]));
				}
			}
		}
		let cbp_c = info.cbp >> 4;
		if cbp_c != 0 {
			for c in 0..2 {
				write_block(w, -1, &res.chroma_dc[c]);
			}
		}
		if cbp_c == 2 {
			for c in 0..2 {
				for blk in 0..4 {
					let nc = self.nc_chroma(info, c, (blk % 2) as i32, (blk / 2) as i32);
					write_block(w, nc, &scan_ac(&res.chroma_ac[c][blk]));
				}
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
				recon::pred_block(plane, cx, cy, 8, mode, av, true);
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
		recon::pred_block(&mut self.f.cb, cx, cy, 8, mode, av, true);
		recon::pred_block(&mut self.f.cr, cx, cy, 8, mode, av, true);
	}

	/// Best Intra_16x16 mode by SATD, and its cost.
	fn choose_i16_mode(&mut self, av: Avail) -> (u8, u32) {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mut best = (0u8, u32::MAX);
		for mode in 0..4u8 {
			if !block_mode_ok(mode, av) {
				continue;
			}
			recon::pred_block(&mut self.f.y, px, py, 16, mode, av, false);
			let cost: u32 =
				(0..16).map(|r| satd4(&self.diff4(0, px + (r % 4) * 4, py + (r / 4) * 4))).sum();
			if cost < best.1 {
				best = (mode, cost);
			}
		}
		best
	}

	fn encode_i16(&mut self, mode: u8, chroma: u8, p_slice: bool) -> Coded {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mb = self.mb_avail();
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		let mut info = self.base_info(MbKind::I16x16);
		let mut res = Residual::default();
		recon::pred_block(&mut self.f.y, px, py, 16, mode, av, false);
		let quant = Quant::new(self.qp, true);
		let mut dc = [0i32; 16];
		let mut any_ac = false;
		for raster in 0..16 {
			let mut coef = fdct4(&self.diff4(0, px + (raster % 4) * 4, py + (raster / 4) * 4));
			dc[raster] = coef[0];
			any_ac |= quant.block4(&mut coef, true);
			coef[0] = 0;
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
		let dcd = recon::luma_dc_dequant(&res.dc, self.qp, scale[0]);
		for raster in 0..16 {
			let mut block = res.luma[raster];
			let nz = info.nnz[raster] != 0;
			if nz {
				recon::dequant4(&mut block, self.qp, scale, true);
			}
			block[0] = dcd[raster];
			if nz || dcd[raster] != 0 {
				recon::idct4_add(&mut self.f.y, px + (raster % 4) * 4, py + (raster / 4) * 4, &block);
			}
		}
		self.predict_chroma(chroma, av);
		info.chroma_pred = chroma;
		self.code_chroma(&mut info, &mut res, true);

		let mut w = BitWriter::new();
		let cbp_c = u32::from(info.cbp >> 4);
		let mb_type = 1 + u32::from(mode) + 4 * cbp_c + if cbp_l != 0 { 12 } else { 0 };
		w.ue(mb_type + if p_slice { 5 } else { 0 });
		w.ue(u32::from(chroma));
		w.se(0); // mb_qp_delta
		self.write_residual(&mut w, &info, &res);
		self.finish(info, w)
	}

	fn encode_i4(&mut self, chroma: u8, p_slice: bool) -> Coded {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mb = self.mb_avail();
		let mut info = self.base_info(MbKind::I4x4);
		let mut res = Residual::default();
		let quant = Quant::new(self.qp, true);
		let scale = &self.dq.l4[0][(self.qp % 6) as usize];
		let mut mode_bits = BitWriter::new();
		for blk in 0..16 {
			let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
			let raster = y4 * 4 + x4;
			let (x, y) = (px + x4 * 4, py + y4 * 4);
			let av = Self::block_avail(x4, y4, mb);
			let pred_mode = self.predicted_intra4(&info, x4, y4);
			let mut best = (u32::MAX, 2u8);
			for mode in 0..9u8 {
				if !intra4_mode_ok(mode, av) {
					continue;
				}
				recon::pred4x4(&mut self.f.y, x, y, mode, av);
				let bits = if mode == pred_mode { 1 } else { 4 };
				let cost = satd4(&self.diff4(0, x, y)) + self.lambda * bits;
				if cost < best.0 {
					best = (cost, mode);
				}
			}
			let mode = best.1;
			recon::pred4x4(&mut self.f.y, x, y, mode, av);
			info.ipred[raster] = mode;
			if mode == pred_mode {
				mode_bits.bit(true);
			} else {
				mode_bits.bit(false);
				mode_bits.put(u32::from(if mode < pred_mode { mode } else { mode - 1 }), 3);
			}
			let mut coef = fdct4(&self.diff4(0, x, y));
			quant.block4(&mut coef, false);
			res.luma[raster] = coef;
			let n = coef.iter().filter(|&&v| v != 0).count() as u8;
			info.nnz[raster] = n;
			if n != 0 {
				info.cbp |= 1 << ((y4 / 2) * 2 + x4 / 2);
				let mut block = coef;
				recon::dequant4(&mut block, self.qp, scale, false);
				recon::idct4_add(&mut self.f.y, x, y, &block);
			}
		}
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		self.predict_chroma(chroma, av);
		info.chroma_pred = chroma;
		self.code_chroma(&mut info, &mut res, true);

		let mut w = BitWriter::new();
		w.ue(if p_slice { 5 } else { 0 });
		w.append(&mode_bits);
		w.ue(u32::from(chroma));
		w.ue(cbp_code(info.cbp, true));
		if info.cbp != 0 {
			w.se(0);
		}
		self.write_residual(&mut w, &info, &res);
		self.finish(info, w)
	}

	/// Best intra candidate (I16x16 or I4x4) by RD cost.
	fn best_intra(&mut self, p_slice: bool) -> Coded {
		let mb = self.mb_avail();
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		let chroma = self.choose_chroma_mode(av);
		let (mode, _) = self.choose_i16_mode(av);
		let i16 = self.encode_i16(mode, chroma, p_slice);
		let i4 = self.encode_i4(chroma, p_slice);
		if self.rd_cost(&i4, 0) < self.rd_cost(&i16, 0) { i4 } else { i16 }
	}

	fn decide_i(&mut self) -> Coded {
		self.best_intra(false)
	}

	// -----------------------------------------------------------------------
	// Inter

	/// Motion-compensates partition (x4, y4, w4, h4) with `mv` into the frame.
	fn mc_part(&mut self, x4: usize, y4: usize, w4: usize, h4: usize, mv: [i16; 2]) {
		let Some(r) = self.reference else { return };
		let (bw, bh) = (w4 * 4, h4 * 4);
		let px = self.mbx * 16 + x4 * 4;
		let py = self.mby * 16 + y4 * 4;
		let (mx, my) = (i32::from(mv[0]), i32::from(mv[1]));
		let mut luma = [0u8; 256];
		recon::mc_luma(
			&r.y,
			px as i32 + (mx >> 2),
			py as i32 + (my >> 2),
			mx & 3,
			my & 3,
			bw,
			bh,
			&mut luma,
		);
		recon::store_prediction(
			&mut self.f.y,
			px,
			py,
			bw,
			bh,
			Some(&luma[..bw * bh]),
			None,
			Weighting::Default,
		);
		let (cx, cy) = ((px / 2) as i32 + (mx >> 3), (py / 2) as i32 + (my >> 3));
		for (c, rp) in [&r.cb, &r.cr].into_iter().enumerate() {
			let mut chroma = [0u8; 64];
			recon::mc_chroma(rp, cx, cy, mx & 7, my & 7, bw / 2, bh / 2, &mut chroma);
			let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
			recon::store_prediction(
				plane,
				px / 2,
				py / 2,
				bw / 2,
				bh / 2,
				Some(&chroma[..bw * bh / 4]),
				None,
				Weighting::Default,
			);
		}
	}

	fn encode_skip(&mut self) -> Coded {
		let mv = self.skip_mv();
		let mut info = self.base_info(MbKind::PSkip);
		info.mv[0] = [mv; 16];
		self.mc_part(0, 0, 4, 4, mv);
		self.finish(info, BitWriter::new())
	}

	/// Codes an inter macroblock with one motion vector per 8x8 block.
	fn encode_inter(&mut self, shape: Shape, mvs: [[i16; 2]; 4]) -> Coded {
		let mut info = self.base_info(MbKind::Inter);
		let mut mvds = Vec::with_capacity(4);
		for (part, &(x4, y4, w4, h4)) in shape.parts().iter().enumerate() {
			let mv = mvs[(y4 / 2) * 2 + x4 / 2];
			let mvp = self.predict_mv(&info, x4, y4, w4, shape, part);
			let mvd = [i32::from(mv[0]) - i32::from(mvp[0]), i32::from(mv[1]) - i32::from(mvp[1])];
			mvds.push(mvd);
			for y in y4..y4 + h4 {
				for x in x4..x4 + w4 {
					info.mv[0][y * 4 + x] = mv;
					info.mvd[0][y * 4 + x] =
						[mvd[0].unsigned_abs().min(255) as u8, mvd[1].unsigned_abs().min(255) as u8];
				}
			}
			self.mc_part(x4, y4, w4, h4, mv);
		}
		let mut res = Residual::default();
		self.code_inter_luma(&mut info, &mut res);
		self.code_chroma(&mut info, &mut res, false);

		let mut w = BitWriter::new();
		w.ue(match shape {
			Shape::S16x16 => 0,
			Shape::S16x8 => 1,
			Shape::S8x16 => 2,
			Shape::S8x8 => 3,
		});
		if shape == Shape::S8x8 {
			for _ in 0..4 {
				w.ue(0); // sub_mb_type P_L0_8x8
			}
		}
		for mvd in &mvds {
			w.se(mvd[0]);
			w.se(mvd[1]);
		}
		w.ue(cbp_code(info.cbp, false));
		if info.cbp != 0 {
			w.se(0);
		}
		self.write_residual(&mut w, &info, &res);
		self.finish(info, w)
	}

	fn mv_bits(mv: [i32; 2], mvp: [i16; 2]) -> u32 {
		se_bits(mv[0] - i32::from(mvp[0])) + se_bits(mv[1] - i32::from(mvp[1]))
	}

	/// Diamond integer search followed by half- and quarter-sample
	/// refinement (x264 `--me dia --subme 7`-like). Returns the vector (in
	/// quarter samples) and its SATD + λ·bits cost.
	fn search(
		&self,
		x4: usize,
		y4: usize,
		bw: usize,
		bh: usize,
		mvp: [i16; 2],
		extra: &[[i16; 2]],
	) -> ([i16; 2], u32) {
		let Some(r) = self.reference else { return ([0, 0], u32::MAX) };
		let (px, py) = (self.mbx * 16 + x4 * 4, self.mby * 16 + y4 * 4);
		let lambda = self.lambda;
		let max_y = self.max_mv_y;
		let in_range = |mx: i32, my: i32| {
			r.padded.fits(px as i32 + mx, py as i32 + my, bw, bh) && my.abs() <= max_y && mx.abs() <= 2040
		};
		let int_cost = |mx: i32, my: i32| -> u32 {
			r.padded.sad(&self.src.y, px, py, px as i32 + mx, py as i32 + my, bw, bh)
				+ lambda * Self::mv_bits([mx * 4, my * 4], mvp)
		};
		// Starting point: the best of the predictor, zero and the extra
		// candidates (rounded to whole samples).
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
		// Large then small diamond until the centre wins.
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
		// Sub-sample refinement on SATD of the real interpolation.
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

	/// Motion search for a partitioning: one vector per partition, each
	/// predicted from the ones before it. Returns vectors per 8x8 block
	/// and the SATD + λ·bits cost.
	fn search_shape(&self, shape: Shape, seed: &[[i16; 2]]) -> ([[i16; 2]; 4], u32) {
		let mut info = self.base_info(MbKind::Inter);
		let mut mvs = [[0i16; 2]; 4];
		let header = match shape {
			Shape::S16x16 => 1,
			Shape::S16x8 | Shape::S8x16 => 3,
			Shape::S8x8 => 5 + 4,
		};
		let mut total = self.lambda * header;
		for (part, &(x4, y4, w4, h4)) in shape.parts().iter().enumerate() {
			let mvp = self.predict_mv(&info, x4, y4, w4, shape, part);
			let (mv, cost) = self.search(x4, y4, w4 * 4, h4 * 4, mvp, seed);
			if cost == u32::MAX {
				return (mvs, u32::MAX);
			}
			total = total.saturating_add(cost);
			for y in y4..y4 + h4 {
				for x in x4..x4 + w4 {
					info.mv[0][y * 4 + x] = mv;
				}
			}
			for y in (y4..y4 + h4).step_by(2) {
				for x in (x4..x4 + w4).step_by(2) {
					mvs[(y / 2) * 2 + x / 2] = mv;
				}
			}
		}
		(mvs, total)
	}

	fn decide_p(&mut self) -> Coded {
		// Skip first: if its residual would quantise away, take it.
		let skip = self.encode_skip();
		let skip_mv = skip.info.mv[0][0];
		if self.skip_is_free() {
			return skip;
		}
		let mut seeds = vec![skip_mv];
		if let Some(m) = self.nb_mb(0, -1).filter(|m| !m.is_intra()) {
			seeds.push(m.mv[0][12]);
		}
		if let Some(m) = self.nb_mb(-1, 0).filter(|m| !m.is_intra()) {
			seeds.push(m.mv[0][3]);
		}
		let (mvs16, cost16) = self.search_shape(Shape::S16x16, &seeds);
		let mut best_shape = (Shape::S16x16, mvs16, cost16);
		// Smaller partitions only when the 16x16 match is poor.
		if cost16 != u32::MAX && cost16 > 24 * self.lambda + 256 {
			seeds.push(mvs16[0]);
			for shape in [Shape::S8x8, Shape::S16x8, Shape::S8x16] {
				let (mvs, cost) = self.search_shape(shape, &seeds);
				if cost < best_shape.2 {
					best_shape = (shape, mvs, cost);
				}
			}
		}
		let mut best = skip;
		let mut best_cost = self.rd_cost(&best, 0) + self.lambda2;
		if best_shape.2 != u32::MAX {
			let inter = self.encode_inter(best_shape.0, best_shape.1);
			let c = self.rd_cost(&inter, 1);
			if c < best_cost {
				best = inter;
				best_cost = c;
			}
		}
		// Intra when its SATD estimate competes with the inter match.
		let mb = self.mb_avail();
		let av = Avail { left: mb[0], top: mb[1], top_right: false, top_left: mb[3] };
		let (_, i16_satd) = self.choose_i16_mode(av);
		if best_shape.2 == u32::MAX || i16_satd + 6 * self.lambda < best_shape.2 + best_shape.2 / 4 {
			let intra = self.best_intra(true);
			let c = self.rd_cost(&intra, 1);
			if c < best_cost {
				best = intra;
			}
		}
		best
	}

	/// Whether the P_Skip prediction (already in the frame) leaves a
	/// residual that quantises to nothing after decimation, so skipping
	/// loses nothing a coded macroblock would keep.
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
}

fn combine_nc(a: Option<i32>, b: Option<i32>) -> i32 {
	match (a, b) {
		(Some(a), Some(b)) => (a + b + 1) >> 1,
		(Some(a), None) => a,
		(None, Some(b)) => b,
		(None, None) => 0,
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
