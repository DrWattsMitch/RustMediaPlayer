//! Slice data decoding (§7.3.4) and macroblock reconstruction (§8.3-8.5).
//! FFmpeg: `h264_slice.c: decode_slice`, `h264_mb.c: hl_decode_mb`.

use super::cabac::Cabac;
use super::consts::BLK_XY;
use super::dpb::RefPic;
use super::mb::{BadMb, Entropy, MbInfo, MbKind, MbResult, MbType, Pred, Residual, SubType};
use super::nal::more_rbsp_data;
use super::params::{Pps, Sps};
use super::recon::{self, Avail, Plane};
use super::slice::{SliceHeader, SliceType};
use crate::codecs::bitstream::BitReader;

/// Per-slice deblocking parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct SliceFilter {
	pub disable: u32,
	pub alpha: i32,
	pub beta: i32,
}

/// The picture being decoded.
#[derive(Debug, Clone)]
pub struct Frame {
	pub y: Plane,
	pub cb: Plane,
	pub cr: Plane,
	pub mbs: Vec<MbInfo>,
	pub mb_w: usize,
	pub mb_h: usize,
	/// Deblocking parameters by slice number - 1.
	pub slices: Vec<SliceFilter>,
}

impl Frame {
	pub fn new(mb_w: usize, mb_h: usize) -> Self {
		Self {
			y: Plane::new(mb_w * 16, mb_h * 16, 0),
			cb: Plane::new(mb_w * 8, mb_h * 8, 128),
			cr: Plane::new(mb_w * 8, mb_h * 8, 128),
			mbs: vec![MbInfo::default(); mb_w * mb_h],
			mb_w,
			mb_h,
			slices: Vec::new(),
		}
	}
}

pub struct SliceDecoder<'a> {
	pub f: &'a mut Frame,
	pub sps: &'a Sps,
	pub pps: &'a Pps,
	pub h: &'a SliceHeader,
	pub lists: &'a [Vec<Option<RefPic>>; 2],
	pub slice_num: u32,
	pub cur_poc: i32,
	pub mbx: usize,
	pub mby: usize,
	pub qp: i32,
	pub prev_qp_delta_nonzero: bool,
	pub res: Option<Box<Residual>>,
	pub sub_types: [SubType; 4],
	pub entropy: Entropy<'a>,
}

impl<'a> SliceDecoder<'a> {
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		f: &'a mut Frame,
		sps: &'a Sps,
		pps: &'a Pps,
		h: &'a SliceHeader,
		lists: &'a [Vec<Option<RefPic>>; 2],
		rbsp: &'a [u8],
		slice_num: u32,
		cur_poc: i32,
	) -> Self {
		let entropy = if pps.cabac {
			Entropy::Cabac(Box::new(Cabac::new(
				rbsp,
				h.data_bit_offset,
				h.slice_type,
				h.cabac_init_idc,
				h.qp,
			)))
		} else {
			let mut bs = BitReader::new(rbsp);
			bs.skip(h.data_bit_offset);
			Entropy::Cavlc(bs)
		};
		Self {
			f,
			sps,
			pps,
			h,
			lists,
			slice_num,
			cur_poc,
			mbx: 0,
			mby: 0,
			qp: h.qp,
			prev_qp_delta_nonzero: false,
			res: Some(Box::default()),
			sub_types: [SubType { pred: Pred::L0, parts: 1, w: 2, h: 2 }; 4],
			entropy,
		}
	}

	fn set_addr(&mut self, addr: usize) {
		self.mbx = addr % self.f.mb_w;
		self.mby = addr / self.f.mb_w;
	}

	/// Decodes all macroblocks of the slice. On error, the macroblocks
	/// decoded so far are kept.
	pub fn decode(&mut self, rbsp: &[u8]) -> MbResult<()> {
		let count = self.f.mb_w * self.f.mb_h;
		let mut addr = self.h.first_mb as usize;
		let cabac = matches!(self.entropy, Entropy::Cabac(_));
		let intra_slice = self.h.slice_type == SliceType::I;
		loop {
			if addr >= count {
				return Err(BadMb("slice runs past the last macroblock"));
			}
			let mut more = true;
			if !intra_slice {
				if cabac {
					self.set_addr(addr);
					if self.cabac_skip_flag() {
						self.skip_mb(addr)?;
						more = false;
					}
				} else {
					let run = match &mut self.entropy {
						Entropy::Cavlc(bs) => bs.ue() as usize,
						Entropy::Cabac(_) => 0,
					};
					for _ in 0..run {
						if addr >= count {
							return Err(BadMb("mb_skip_run past the last macroblock"));
						}
						self.skip_mb(addr)?;
						addr += 1;
					}
					if run > 0 {
						more = self.cavlc_more(rbsp);
					}
				}
			}
			if more {
				if addr >= count {
					return Err(BadMb("slice runs past the last macroblock"));
				}
				self.set_addr(addr);
				self.macroblock(addr)?;
			}
			let more = match &mut self.entropy {
				Entropy::Cavlc(bs) => {
					if bs.overrun() {
						return Err(BadMb("truncated slice data"));
					}
					more_rbsp_data(rbsp, bs.pos())
				}
				Entropy::Cabac(c) => {
					if c.overrun() {
						return Err(BadMb("truncated slice data"));
					}
					!c.terminate()
				}
			};
			addr += 1;
			if !more {
				return Ok(());
			}
		}
	}

	fn cavlc_more(&self, rbsp: &[u8]) -> bool {
		match &self.entropy {
			Entropy::Cavlc(bs) => more_rbsp_data(rbsp, bs.pos()),
			Entropy::Cabac(_) => true,
		}
	}

	fn finish_mb(&mut self, addr: usize, mut cur: MbInfo) {
		for list in 0..2 {
			for b8 in 0..4 {
				let r = cur.ref_idx[list][b8];
				cur.ref_id[list][b8] = if r < 0 {
					0
				} else {
					self.lists[list].get(r as usize).and_then(|r| r.as_ref()).map_or(0, |r| r.pic.id)
				};
			}
		}
		self.f.mbs[addr] = cur;
	}

	fn skip_mb(&mut self, addr: usize) -> MbResult<()> {
		self.set_addr(addr);
		let mut cur = MbInfo { slice: self.slice_num, ipred: [2; 16], ..MbInfo::default() };
		cur.qp = self.qp as u8;
		cur.qpc = [
			super::consts::chroma_qp(self.qp, self.pps.chroma_qp_offset[0]),
			super::consts::chroma_qp(self.qp, self.pps.chroma_qp_offset[1]),
		];
		self.prev_qp_delta_nonzero = false;
		if self.h.slice_type == SliceType::B {
			cur.kind = MbKind::BSkip;
			cur.direct16 = true;
			cur.direct8 = 0xF;
			cur.ref_idx = [[-1; 4]; 2];
			self.predict_direct(&mut cur, 0xF)?;
		} else {
			cur.kind = MbKind::PSkip;
			self.predict_p_skip(&mut cur);
		}
		self.finish_mb(addr, cur);
		let cur = self.f.mbs[addr];
		self.motion_compensate(&cur, None)
	}

	fn macroblock(&mut self, addr: usize) -> MbResult<()> {
		let mut cur = MbInfo { slice: self.slice_num, ..MbInfo::default() };
		let mb_type = self.parse_macroblock(&mut cur)?;
		self.finish_mb(addr, cur);
		let cur = self.f.mbs[addr];
		let result = self.reconstruct(&cur, mb_type);
		if let Some(res) = self.res.as_mut() {
			clear_residual(res, &cur);
		}
		result
	}

	// -----------------------------------------------------------------------
	// Reconstruction

	/// Neighbour availability for intra prediction (6.4.11, with
	/// constrained_intra_pred).
	fn intra_nb(&self, dx: i32, dy: i32) -> bool {
		self.nb_mb(dx, dy).is_some_and(|m| !self.pps.constrained_intra_pred || m.is_intra())
	}

	fn block_avail(&self, x4: usize, y4: usize, size: usize, mb: [bool; 4]) -> Avail {
		// mb = [left, top, top_right, top_left] macroblock availability.
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
			super::mb::blk_index(rx, y4 - 1) < super::mb::blk_index(x4, y4)
		};
		Avail { left, top, top_right, top_left }
	}

	fn reconstruct(&mut self, cur: &MbInfo, mb_type: MbType) -> MbResult<()> {
		let (px, py) = (self.mbx * 16, self.mby * 16);
		let mb_av =
			[self.intra_nb(-1, 0), self.intra_nb(0, -1), self.intra_nb(1, -1), self.intra_nb(-1, -1)];
		let qp = i32::from(cur.qp);
		let dq = &self.pps.dequant;
		let Some(mut res) = self.res.take() else {
			return Err(BadMb("residual buffer missing"));
		};
		let result = (|| -> MbResult<()> {
			match cur.kind {
				MbKind::Pcm => {
					for (i, &s) in res.pcm[..256].iter().enumerate() {
						self.f.y.data[(py + i / 16) * self.f.y.width + px + i % 16] = s;
					}
					for c in 0..2 {
						let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
						for (i, &s) in res.pcm[256 + 64 * c..320 + 64 * c].iter().enumerate() {
							plane.data[(py / 2 + i / 8) * plane.width + px / 2 + i % 8] = s;
						}
					}
					return Ok(());
				}
				MbKind::I4x4 => {
					for blk in 0..16 {
						let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
						let raster = y4 * 4 + x4;
						let av = self.block_avail(x4, y4, 1, mb_av);
						let mode = cur.ipred[raster];
						if mode > 8 || !intra4_mode_ok(mode, av) {
							return Err(BadMb("intra 4x4 mode needs unavailable samples"));
						}
						recon::pred4x4(&mut self.f.y, px + x4 * 4, py + y4 * 4, mode, av);
						if cur.nnz[raster] != 0 {
							recon::dequant4(&mut res.luma[raster], qp, &dq.l4[0][(qp % 6) as usize], false);
							recon::idct4_add(&mut self.f.y, px + x4 * 4, py + y4 * 4, &res.luma[raster]);
						}
					}
				}
				MbKind::I8x8 => {
					for b8 in 0..4 {
						let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
						let av = self.block_avail(x4, y4, 2, mb_av);
						let mode = cur.ipred[y4 * 4 + x4];
						if mode > 8 || !intra4_mode_ok(mode, av) {
							return Err(BadMb("intra 8x8 mode needs unavailable samples"));
						}
						recon::pred8x8l(&mut self.f.y, px + x4 * 4, py + y4 * 4, mode, av);
						if cur.cbp & (1 << b8) != 0 {
							recon::dequant8(&mut res.luma8[b8], qp, &dq.l8[0][(qp % 6) as usize]);
							recon::idct8_add(&mut self.f.y, px + x4 * 4, py + y4 * 4, &res.luma8[b8]);
						}
					}
				}
				MbKind::I16x16 => {
					let MbType::I16x16 { pred, .. } = mb_type else {
						return Err(BadMb("intra 16x16 type mismatch"));
					};
					let av = Avail { left: mb_av[0], top: mb_av[1], top_right: false, top_left: mb_av[3] };
					if !block_mode_ok(pred, av) {
						return Err(BadMb("intra 16x16 mode needs unavailable samples"));
					}
					recon::pred_block(&mut self.f.y, px, py, 16, pred, av, false);
					let dc = recon::luma_dc_dequant(&res.dc, qp, dq.l4[0][(qp % 6) as usize][0]);
					for raster in 0..16 {
						let block = &mut res.luma[raster];
						if cur.nnz[raster] != 0 {
							recon::dequant4(block, qp, &dq.l4[0][(qp % 6) as usize], true);
						}
						block[0] = dc[raster];
						if cur.nnz[raster] != 0 || dc[raster] != 0 {
							let (x4, y4) = (raster % 4, raster / 4);
							recon::idct4_add(&mut self.f.y, px + x4 * 4, py + y4 * 4, block);
						}
					}
				}
				_ => {
					self.motion_compensate(cur, Some(mb_type))?;
					if cur.t8x8 {
						for b8 in 0..4 {
							if cur.cbp & (1 << b8) != 0 {
								let (x, y) = (px + (b8 % 2) * 8, py + (b8 / 2) * 8);
								recon::dequant8(&mut res.luma8[b8], qp, &dq.l8[1][(qp % 6) as usize]);
								recon::idct8_add(&mut self.f.y, x, y, &res.luma8[b8]);
							}
						}
					} else {
						for raster in 0..16 {
							if cur.nnz[raster] != 0 {
								let (x4, y4) = (raster % 4, raster / 4);
								recon::dequant4(&mut res.luma[raster], qp, &dq.l4[3][(qp % 6) as usize], false);
								recon::idct4_add(&mut self.f.y, px + x4 * 4, py + y4 * 4, &res.luma[raster]);
							}
						}
					}
				}
			}
			// Chroma.
			let intra = cur.is_intra();
			if intra {
				let av = Avail { left: mb_av[0], top: mb_av[1], top_right: false, top_left: mb_av[3] };
				let mode = match cur.chroma_pred {
					0 => 2, // DC
					1 => 1, // horizontal
					2 => 0, // vertical
					_ => 3, // plane
				};
				if !block_mode_ok(mode, av) {
					return Err(BadMb("chroma intra mode needs unavailable samples"));
				}
				recon::pred_block(&mut self.f.cb, px / 2, py / 2, 8, mode, av, true);
				recon::pred_block(&mut self.f.cr, px / 2, py / 2, 8, mode, av, true);
			}
			if cur.cbp >> 4 != 0 {
				for c in 0..2 {
					let qpc = i32::from(cur.qpc[c]);
					let list = if intra { 1 + c } else { 4 + c };
					let scale = &dq.l4[list][(qpc % 6) as usize];
					let dc = recon::chroma_dc_dequant(&res.chroma_dc[c], qpc, scale[0]);
					let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
					for blk in 0..4 {
						let block = &mut res.chroma_ac[c][blk];
						if cur.nnz[16 + 4 * c + blk] != 0 {
							recon::dequant4(block, qpc, scale, true);
						}
						block[0] = dc[blk];
						if cur.nnz[16 + 4 * c + blk] != 0 || dc[blk] != 0 {
							recon::idct4_add(plane, px / 2 + (blk % 2) * 4, py / 2 + (blk / 2) * 4, block);
						}
					}
				}
			}
			Ok(())
		})();
		self.res = Some(res);
		result
	}
}

/// Intra 4x4/8x8 modes that need samples which are not available would
/// read outside the picture; such streams are invalid.
fn intra4_mode_ok(mode: u8, a: Avail) -> bool {
	match mode {
		0 | 3 | 7 => a.top,
		1 | 8 => a.left,
		4..=6 => a.top && a.left && a.top_left,
		_ => true,
	}
}

/// Same check for 16x16 / chroma modes (0 V, 1 H, 2 DC, 3 plane).
fn block_mode_ok(mode: u8, a: Avail) -> bool {
	match mode {
		0 => a.top,
		1 => a.left,
		3 => a.top && a.left && a.top_left,
		_ => true,
	}
}

/// Zeroes the coefficient blocks a macroblock used, ready for the next.
fn clear_residual(res: &mut Residual, cur: &MbInfo) {
	res.dc = [0; 16];
	for b in res.luma.iter_mut() {
		*b = [0; 16];
	}
	if cur.t8x8 {
		for b in res.luma8.iter_mut() {
			*b = [0; 64];
		}
	}
	res.chroma_dc = [[0; 4]; 2];
	for comp in res.chroma_ac.iter_mut() {
		for b in comp.iter_mut() {
			*b = [0; 16];
		}
	}
}
