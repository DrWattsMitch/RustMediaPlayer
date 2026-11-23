//! Slice data decoding (§7.3.4) and macroblock reconstruction (§8.3-8.5).
//! FFmpeg: `h264_slice.c: decode_slice`, `h264_mb.c: hl_decode_mb`.

use super::cabac::Cabac;
use super::consts::BLK_XY;
use super::dpb::{MbMotion, RefList, RefPic};
use super::mb::{BadMb, Entropy, MbInfo, MbKind, MbResult, MbType, Pred, Residual, SubType};
use super::nal::more_rbsp_data;
use super::params::{Pps, Sps};
use super::recon::{self, Avail, Pixel, Plane};
use super::slice::{SliceHeader, SliceType};
use crate::codecs::bitstream::BitReader;
use crate::core::{Pooled, VecPool};

/// Per-slice deblocking parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct SliceFilter {
	pub disable: u32,
	pub alpha: i32,
	pub beta: i32,
}

/// The picture being decoded.
#[derive(Debug, Clone)]
pub struct Frame<T = u8> {
	pub y: Plane<T>,
	pub cb: Plane<T>,
	pub cr: Plane<T>,
	pub mbs: Pooled<MbInfo>,
	pub mb_w: usize,
	pub mb_h: usize,
	/// Deblocking parameters by slice number - 1.
	pub slices: Vec<SliceFilter>,
	pub bit_depth: u32,
	/// 4:2:2 (chroma planes as tall as luma) rather than 4:2:0.
	pub chroma_422: bool,
}

impl<T: Pixel> Frame<T> {
	/// An 8-bit-range 4:2:0 picture (the encoder's).
	pub fn new(mb_w: usize, mb_h: usize) -> Self {
		let ch = mb_h * 8;
		Self {
			y: Plane::new(mb_w * 16, mb_h * 16, T::default()),
			cb: Plane::new(mb_w * 8, ch, T::of(128)),
			cr: Plane::new(mb_w * 8, ch, T::of(128)),
			mbs: Pooled::detached(vec![MbInfo::default(); mb_w * mb_h]),
			mb_w,
			mb_h,
			slices: Vec::new(),
			bit_depth: 8,
			chroma_422: false,
		}
	}

	/// A picture with every buffer taken from `pools` (so that decoding a
	/// picture allocates nothing once the pools are warm).
	pub fn pooled(
		pools: &DecodePools<T>,
		mb_w: usize,
		mb_h: usize,
		bit_depth: u32,
		chroma_422: bool,
	) -> Self {
		let ch = mb_h * if chroma_422 { 16 } else { 8 };
		let grey = T::of(1 << (bit_depth - 1));
		Self {
			y: Plane::pooled(&pools.planes, mb_w * 16, mb_h * 16, T::default()),
			cb: Plane::pooled(&pools.planes, mb_w * 8, ch, grey),
			cr: Plane::pooled(&pools.planes, mb_w * 8, ch, grey),
			mbs: pools.mbs.take_filled(mb_w * mb_h, MbInfo::default()),
			mb_w,
			mb_h,
			slices: Vec::new(),
			bit_depth,
			chroma_422,
		}
	}

	/// Largest sample value.
	#[inline]
	pub fn max(&self) -> i32 {
		(1 << self.bit_depth) - 1
	}

	/// Chroma rows per macroblock (8 for 4:2:0, 16 for 4:2:2).
	#[inline]
	pub fn mb_chroma_h(&self) -> usize {
		if self.chroma_422 { 16 } else { 8 }
	}
}

/// Buffer pools shared by the pictures of one decoder (and its frame
/// threads): samples, macroblock info and stored motion.
#[derive(Debug, Clone)]
pub struct DecodePools<T = u8> {
	pub planes: VecPool<T>,
	pub mbs: VecPool<MbInfo>,
	pub motion: VecPool<MbMotion>,
}

impl<T> Default for DecodePools<T> {
	fn default() -> Self {
		Self { planes: VecPool::default(), mbs: VecPool::default(), motion: VecPool::default() }
	}
}

/// Called by [`SliceDecoder`] as rows (or MBAFF pair rows) complete.
pub type RowHook<'a, T> = &'a mut dyn FnMut(&mut Frame<T>);

pub struct SliceDecoder<'a, T: Pixel = u8> {
	pub f: &'a mut Frame<T>,
	pub sps: &'a Sps,
	pub pps: &'a Pps,
	pub h: &'a SliceHeader,
	pub lists: &'a [RefList<T>; 2],
	pub slice_num: u32,
	pub cur_poc: i32,
	pub mbx: usize,
	pub mby: usize,
	pub qp: i32,
	pub prev_qp_delta_nonzero: bool,
	pub res: Option<Box<Residual>>,
	pub sub_types: [SubType; 4],
	pub entropy: Entropy<'a>,
	/// Called each time decoding crosses into a new macroblock row, so the
	/// frame-threaded decoder can deblock and publish finished rows.
	pub row_hook: Option<RowHook<'a, T>>,
	/// Largest sample value, bit depth and QpBdOffset (luma = chroma).
	pub max: i32,
	pub bd: u32,
	pub qp_bd: i32,
	/// 4:2:2 chroma.
	pub c422: bool,
	/// Parity of the field picture being decoded (`None` for a frame).
	/// Field macroblocks are stored interleaved: field row `r` at frame
	/// macroblock row `2r + parity`.
	pub field: Option<usize>,
	/// Macroblock rows of the picture (half the frame's for a field).
	pub rows: usize,
	/// Id of the picture being decoded: the second field of a pair may
	/// predict from the first one, which is read from `f` directly.
	pub cur_id: u32,
	/// Coefficient scans of the current macroblock (frame or field).
	pub scan4: &'static [u8; 16],
	pub scan8: &'static [u8; 64],
	/// MBAFF frame: macroblock pairs, each frame or field coded.
	pub mbaff: bool,
	/// mb_field_decoding_flag of the current pair (MBAFF).
	pub mb_field: bool,
	/// MBAFF: the reference lists of field macroblocks, per parity of the
	/// macroblock: index `2i` is the same-parity field of frame `i`, `2i + 1`
	/// the other one (8.4.2.1).
	pub field_lists: Option<Box<[[RefList<T>; 2]; 2]>>,
	/// Top and bottom field order counts of the current frame (its own
	/// count twice for a field).
	pub field_poc: [i32; 2],
}

impl<'a, T: Pixel> SliceDecoder<'a, T> {
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		f: &'a mut Frame<T>,
		sps: &'a Sps,
		pps: &'a Pps,
		h: &'a SliceHeader,
		lists: &'a [RefList<T>; 2],
		rbsp: &'a [u8],
		slice_num: u32,
		cur_poc: i32,
	) -> Self {
		let field = h.parity();
		let entropy = if pps.cabac {
			let mut c = Cabac::new(rbsp, h.data_bit_offset, h.slice_type, h.cabac_init_idc, h.qp);
			c.field = field.is_some();
			Entropy::Cabac(Box::new(c))
		} else {
			let mut bs = BitReader::new(rbsp);
			bs.skip(h.data_bit_offset);
			Entropy::Cavlc(bs)
		};
		let bd = sps.bit_depth;
		let rows = if field.is_some() { f.mb_h / 2 } else { f.mb_h };
		let (scan4, scan8) = if field.is_some() {
			(&super::consts::FIELD_SCAN_4X4, &super::consts::FIELD_SCAN_8X8)
		} else {
			(&super::consts::ZIGZAG_4X4, &super::consts::ZIGZAG_8X8)
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
			row_hook: None,
			max: (1 << bd) - 1,
			bd,
			qp_bd: 6 * (bd as i32 - 8),
			c422: sps.chroma_format_idc == 2,
			field,
			rows,
			cur_id: 0,
			scan4,
			scan8,
			mbaff: h.mbaff,
			mb_field: false,
			field_lists: h.mbaff.then(|| {
				let derive = |p: usize, l: usize| -> RefList<T> {
					lists[l]
						.iter()
						.flat_map(|r| [r.as_ref().map(|r| r.field(p)), r.as_ref().map(|r| r.field(1 - p))])
						.collect()
				};
				Box::new([[derive(0, 0), derive(0, 1)], [derive(1, 0), derive(1, 1)]])
			}),
			field_poc: [cur_poc; 2],
		}
	}

	/// Sets the field order counts of the current frame.
	pub fn with_field_poc(mut self, field_poc: [i32; 2]) -> Self {
		self.field_poc = field_poc;
		self
	}

	/// The current macroblock is a field macroblock (field picture or
	/// MBAFF field pair).
	#[inline(always)]
	pub fn cur_field(&self) -> bool {
		self.field.is_some() || self.mb_field
	}

	/// Parity of the current macroblock when it is a field macroblock.
	#[inline(always)]
	pub fn cur_parity(&self) -> usize {
		match self.field {
			Some(p) => p,
			None => self.mby & 1,
		}
	}

	/// Reference list `l` as seen by the current macroblock (the field list
	/// for an MBAFF field macroblock).
	#[inline]
	pub fn list(&self, l: usize) -> &[Option<RefPic<T>>] {
		match (&self.field_lists, self.mb_field) {
			(Some(f), true) => &f[self.mby & 1][l],
			_ => &self.lists[l],
		}
	}

	/// POC of the current frame, field, or (MBAFF field macroblock) field of
	/// the current frame.
	#[inline]
	pub fn cur_poc(&self) -> i32 {
		if self.mbaff && self.mb_field { self.field_poc[self.mby & 1] } else { self.cur_poc }
	}

	/// Sets the id of the picture being decoded.
	pub fn with_picture_id(mut self, id: u32) -> Self {
		self.cur_id = id;
		self
	}

	/// Index into `f.mbs` of macroblock (x, y) of the current picture.
	#[inline]
	pub fn store_idx(&self, x: usize, y: usize) -> usize {
		match self.field {
			None => y * self.f.mb_w + x,
			Some(p) => (2 * y + p) * self.f.mb_w + x,
		}
	}

	/// Index into `f.mbs` of macroblock address `addr`.
	#[inline]
	fn store_of(&self, addr: usize) -> usize {
		let (x, y) = self.pos_of(addr);
		self.store_idx(x, y)
	}

	/// mb_field_decoding_flag inferred for a pair whose flag is not (yet)
	/// decoded: the left pair's, else the above pair's (both in this
	/// slice), else frame (7.4.4).
	fn infer_field(&self) -> bool {
		let w = self.f.mb_w;
		let top_row = self.mby & !1;
		let same = |m: &MbInfo| m.slice == self.slice_num;
		if self.mbx > 0 {
			let left = &self.f.mbs[top_row * w + self.mbx - 1];
			if same(left) {
				return left.field;
			}
		}
		if top_row >= 2 {
			let above = &self.f.mbs[(top_row - 2) * w + self.mbx];
			if same(above) {
				return above.field;
			}
		}
		false
	}

	/// mb_field_decoding_flag (CABAC context from the left and above pairs,
	/// 9.3.3.1.1.2).
	fn read_field_flag(&mut self) -> bool {
		let w = self.f.mb_w;
		let top_row = self.mby & !1;
		let cond = |m: &MbInfo| usize::from(m.slice == self.slice_num && m.field);
		let a = if self.mbx > 0 { cond(&self.f.mbs[top_row * w + self.mbx - 1]) } else { 0 };
		let b = if top_row >= 2 { cond(&self.f.mbs[(top_row - 2) * w + self.mbx]) } else { 0 };
		match &mut self.entropy {
			Entropy::Cavlc(bs) => bs.bit(),
			Entropy::Cabac(c) => c.decision(super::cabac::ctx::MB_FIELD + a + b),
		}
	}

	/// Selects frame or field coefficient scans and CABAC contexts for the
	/// current macroblock.
	fn set_mb_coding(&mut self) {
		let field = self.cur_field();
		(self.scan4, self.scan8) = if field {
			(&super::consts::FIELD_SCAN_4X4, &super::consts::FIELD_SCAN_8X8)
		} else {
			(&super::consts::ZIGZAG_4X4, &super::consts::ZIGZAG_8X8)
		};
		if let Entropy::Cabac(c) = &mut self.entropy {
			c.field = field;
		}
	}

	/// Decodes the macroblock pairs of an MBAFF slice (7.3.4 with
	/// MbaffFrameFlag): the field flag of a pair precedes its first coded
	/// macroblock; a skipped top macroblock takes the flag of a coded bottom
	/// one, whose skip flag and field flag are read first (as FFmpeg does).
	fn decode_mbaff(&mut self, rbsp: &[u8]) -> MbResult<()> {
		let count = self.f.mb_w * self.f.mb_h;
		let mut addr = self.h.first_mb as usize;
		let cabac = matches!(self.entropy, Entropy::Cabac(_));
		let intra_slice = self.h.slice_type == SliceType::I;
		let mut skip_run: Option<usize> = None;
		let (mut top_skipped, mut next_skipped, mut field_read) = (false, false, false);
		loop {
			if addr >= count {
				return Err(BadMb("slice runs past the last macroblock"));
			}
			self.set_addr(addr);
			let bottom = addr % 2 == 1;
			if !bottom {
				self.mb_field = self.infer_field();
				(top_skipped, field_read) = (false, false);
			}
			let mut skipped = false;
			let mut more_after_skip = true;
			if !intra_slice {
				if cabac {
					skipped = if bottom && top_skipped { next_skipped } else { self.cabac_skip_flag() };
					if skipped && !bottom {
						self.mby += 1;
						next_skipped = self.cabac_skip_flag();
						self.mby -= 1;
						if !next_skipped {
							self.mb_field = self.read_field_flag();
							field_read = true;
						}
					}
				} else {
					let run = match skip_run {
						Some(r) => r,
						None => match &mut self.entropy {
							Entropy::Cavlc(bs) => bs.ue() as usize,
							Entropy::Cabac(_) => 0,
						},
					};
					if run > 0 {
						skipped = true;
						skip_run = Some(run - 1);
						if run == 1 {
							more_after_skip = self.cavlc_more(rbsp);
							if !bottom && more_after_skip {
								self.mb_field = self.read_field_flag();
								field_read = true;
							}
						}
					} else {
						skip_run = None;
					}
				}
			}
			if skipped {
				self.set_mb_coding();
				self.skip_mb(addr)?;
			} else {
				if (!bottom || top_skipped) && !field_read {
					self.mb_field = self.read_field_flag();
					field_read = true;
				}
				self.set_mb_coding();
				self.macroblock(addr)?;
			}
			if !bottom {
				top_skipped = skipped;
			}
			let more = match &mut self.entropy {
				Entropy::Cavlc(bs) => {
					if bs.overrun() {
						return Err(BadMb("truncated slice data"));
					}
					if skipped {
						skip_run != Some(0) || more_after_skip
					} else {
						more_rbsp_data(rbsp, bs.pos())
					}
				}
				Entropy::Cabac(c) => {
					if c.overrun() {
						return Err(BadMb("truncated slice data"));
					}
					!bottom || !c.terminate()
				}
			};
			addr += 1;
			if addr.is_multiple_of(2 * self.f.mb_w) {
				if let Some(hook) = self.row_hook.as_mut() {
					hook(self.f);
				}
			}
			if !more {
				return Ok(());
			}
		}
	}

	/// QPc of both chroma components for luma QP `qp` (8.5.8, without the
	/// bit-depth offset).
	pub fn chroma_qps(&self, qp: i32) -> [i8; 2] {
		[
			super::consts::chroma_qp(qp, self.pps.chroma_qp_offset[0], self.qp_bd),
			super::consts::chroma_qp(qp, self.pps.chroma_qp_offset[1], self.qp_bd),
		]
	}

	/// Chroma rows per macroblock.
	#[inline]
	pub fn mb_ch(&self) -> usize {
		if self.c422 { 16 } else { 8 }
	}

	/// Runs `hook` whenever a macroblock row has been finished.
	pub fn with_row_hook(mut self, hook: RowHook<'a, T>) -> Self {
		self.row_hook = Some(hook);
		self
	}

	#[inline]
	fn row_check(&mut self, next_addr: usize) {
		if next_addr.is_multiple_of(self.f.mb_w) {
			if let Some(hook) = self.row_hook.as_mut() {
				hook(self.f);
			}
		}
	}

	/// Position (x, row) of macroblock address `addr`: the field row for a
	/// field picture, the frame row for frames (MBAFF addresses count pairs:
	/// top, bottom, next pair).
	#[inline]
	fn pos_of(&self, addr: usize) -> (usize, usize) {
		let w = self.f.mb_w;
		if self.mbaff {
			let pair = addr / 2;
			(pair % w, 2 * (pair / w) + addr % 2)
		} else {
			(addr % w, addr / w)
		}
	}

	fn set_addr(&mut self, addr: usize) {
		(self.mbx, self.mby) = self.pos_of(addr);
	}

	/// Decodes all macroblocks of the slice. On error, the macroblocks
	/// decoded so far are kept.
	pub fn decode(&mut self, rbsp: &[u8]) -> MbResult<()> {
		if self.mbaff {
			return self.decode_mbaff(rbsp);
		}
		let count = self.f.mb_w * self.rows;
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
						self.row_check(addr);
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
			self.row_check(addr);
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
					self.list(list).get(r as usize).and_then(|r| r.as_ref()).map_or(0, |r| r.key())
				};
			}
		}
		cur.field = self.cur_field();
		let i = self.store_of(addr);
		self.f.mbs[i] = cur;
	}

	fn skip_mb(&mut self, addr: usize) -> MbResult<()> {
		self.set_addr(addr);
		let mut cur = MbInfo {
			slice: self.slice_num,
			ipred: [2; 16],
			field: self.cur_field(),
			..MbInfo::default()
		};
		cur.qp = self.qp as i8;
		cur.qpc = self.chroma_qps(self.qp);
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
		let cur = self.f.mbs[self.store_of(addr)];
		self.motion_compensate(&cur, None)
	}

	fn macroblock(&mut self, addr: usize) -> MbResult<()> {
		let mut cur = MbInfo { slice: self.slice_num, field: self.cur_field(), ..MbInfo::default() };
		let mb_type = self.parse_macroblock(&mut cur)?;
		self.finish_mb(addr, cur);
		let cur = self.f.mbs[self.store_of(addr)];
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

	fn block_avail(x4: usize, y4: usize, size: usize, mb: [bool; 4]) -> Avail {
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

	/// The view the current macroblock's samples are addressed in (`None`
	/// = the frame, `Some(parity)` = one field of it) and its luma origin
	/// in that view.
	#[inline]
	pub fn mb_view(&self) -> (Option<usize>, usize, usize) {
		if self.mbaff && self.mb_field {
			return (Some(self.mby & 1), self.mbx * 16, (self.mby >> 1) * 16);
		}
		(self.field, self.mbx * 16, self.mby * 16)
	}

	fn reconstruct(&mut self, cur: &MbInfo, mb_type: MbType) -> MbResult<()> {
		let (field, px, py) = self.mb_view();
		let ch = self.mb_ch();
		let (cx, cy) = (px / 2, py / 16 * ch);
		let mb_av =
			[self.intra_nb(-1, 0), self.intra_nb(0, -1), self.intra_nb(1, -1), self.intra_nb(-1, -1)];
		let qp = i32::from(cur.qp) + self.qp_bd;
		let bd = self.bd;
		let max = self.max;
		let dq = &self.pps.dequant;
		let Some(mut res) = self.res.take() else {
			return Err(BadMb("residual buffer missing"));
		};
		let result = (|| -> MbResult<()> {
			let mut y = self.f.y.dst(field, max);
			match cur.kind {
				MbKind::Pcm => {
					for (i, &s) in res.pcm[..256].iter().enumerate() {
						y.set(px + i % 16, py + i / 16, T::of(i32::from(s)));
					}
					let n = 8 * ch;
					for c in 0..2 {
						let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
						let mut d = plane.dst(field, max);
						for (i, &s) in res.pcm[256 + n * c..256 + n * (c + 1)].iter().enumerate() {
							d.set(cx + i % 8, cy + i / 8, T::of(i32::from(s)));
						}
					}
					return Ok(());
				}
				MbKind::I4x4 => {
					for blk in 0..16 {
						let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
						let raster = y4 * 4 + x4;
						let av = Self::block_avail(x4, y4, 1, mb_av);
						let mode = cur.ipred[raster];
						if mode > 8 || !intra4_mode_ok(mode, av) {
							return Err(BadMb("intra 4x4 mode needs unavailable samples"));
						}
						recon::pred4x4(&mut y, px + x4 * 4, py + y4 * 4, mode, av);
						if cur.nnz[raster] != 0 {
							recon::dequant4(&mut res.luma[raster], qp, &dq.l4[0][(qp % 6) as usize], false, bd);
							recon::idct4_add(&mut y, px + x4 * 4, py + y4 * 4, &res.luma[raster]);
						}
					}
				}
				MbKind::I8x8 => {
					for b8 in 0..4 {
						let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
						let av = Self::block_avail(x4, y4, 2, mb_av);
						let mode = cur.ipred[y4 * 4 + x4];
						if mode > 8 || !intra4_mode_ok(mode, av) {
							return Err(BadMb("intra 8x8 mode needs unavailable samples"));
						}
						recon::pred8x8l(&mut y, px + x4 * 4, py + y4 * 4, mode, av);
						if cur.cbp & (1 << b8) != 0 {
							recon::dequant8(&mut res.luma8[b8], qp, &dq.l8[0][(qp % 6) as usize], bd);
							recon::idct8_add(&mut y, px + x4 * 4, py + y4 * 4, &res.luma8[b8]);
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
					recon::pred_block(&mut y, px, py, 16, 16, pred, av, false);
					let dc = recon::luma_dc_dequant(&res.dc, qp, dq.l4[0][(qp % 6) as usize][0], bd);
					for raster in 0..16 {
						let block = &mut res.luma[raster];
						if cur.nnz[raster] != 0 {
							recon::dequant4(block, qp, &dq.l4[0][(qp % 6) as usize], true, bd);
						}
						block[0] = dc[raster];
						if cur.nnz[raster] != 0 || dc[raster] != 0 {
							let (x4, y4) = (raster % 4, raster / 4);
							recon::idct4_add(&mut y, px + x4 * 4, py + y4 * 4, block);
						}
					}
				}
				_ => {
					self.motion_compensate(cur, Some(mb_type))?;
					let mut y = self.f.y.dst(field, max);
					if cur.t8x8 {
						for b8 in 0..4 {
							if cur.cbp & (1 << b8) != 0 {
								let (x, yy) = (px + (b8 % 2) * 8, py + (b8 / 2) * 8);
								recon::dequant8(&mut res.luma8[b8], qp, &dq.l8[1][(qp % 6) as usize], bd);
								recon::idct8_add(&mut y, x, yy, &res.luma8[b8]);
							}
						}
					} else {
						for raster in 0..16 {
							if cur.nnz[raster] != 0 {
								let (x4, y4) = (raster % 4, raster / 4);
								recon::dequant4(&mut res.luma[raster], qp, &dq.l4[3][(qp % 6) as usize], false, bd);
								recon::idct4_add(&mut y, px + x4 * 4, py + y4 * 4, &res.luma[raster]);
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
				recon::pred_block(&mut self.f.cb.dst(field, max), cx, cy, 8, ch, mode, av, true);
				recon::pred_block(&mut self.f.cr.dst(field, max), cx, cy, 8, ch, mode, av, true);
			}
			if cur.cbp >> 4 != 0 {
				let blocks = ch / 2;
				for c in 0..2 {
					let qpc = i32::from(cur.qpc[c]) + self.qp_bd;
					let list = if intra { 1 + c } else { 4 + c };
					let scale = &dq.l4[list][(qpc % 6) as usize];
					let mut dc = [0i32; 8];
					if self.c422 {
						let q = qpc + 3;
						let s = dq.l4[list][(q % 6) as usize][0];
						dc = recon::chroma422_dc_dequant(&res.chroma_dc[c], q, s, bd);
					} else {
						dc[..4].copy_from_slice(&recon::chroma_dc_dequant(
							&res.chroma_dc[c],
							qpc,
							scale[0],
							bd,
						));
					}
					let plane = if c == 0 { &mut self.f.cb } else { &mut self.f.cr };
					let mut d = plane.dst(field, max);
					for blk in 0..blocks {
						let block = &mut res.chroma_ac[c][blk];
						let nz = cur.nnz[16 + 8 * c + blk] != 0;
						if nz {
							recon::dequant4(block, qpc, scale, true, bd);
						}
						block[0] = dc[blk];
						if nz || dc[blk] != 0 {
							recon::idct4_add(&mut d, cx + (blk % 2) * 4, cy + (blk / 2) * 4, block);
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
	res.chroma_dc = [[0; 8]; 2];
	for comp in res.chroma_ac.iter_mut() {
		for b in comp.iter_mut() {
			*b = [0; 16];
		}
	}
}
