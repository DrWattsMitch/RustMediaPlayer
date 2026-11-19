//! VP8 frame decoding: headers, modes and motion vectors, residual tokens,
//! reconstruction, the loop filter and reference buffers, and the
//! [`crate::core::Decoder`] implementation (RFC 6386; libvpx
//! `vp8/decoder/decodeframe.c`, `decodemv.c`, `detokenize.c`, `onyxd_if.c`).

// Block, partition and context indices are the bitstream's own.
#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use super::dsp::{self, B_PRED, DC_PRED, EdgeLimits, H_PRED, RefPlane, TM_PRED, V_PRED};
use super::range::BoolDecoder;
use super::tables::*;
use crate::core::{
	self, BufferPool, CodecPrivate, CodecState, DecodeOut, Frame, FrameBuffer, Keyframe, Packet,
	Parameters, PlaneRef, Pooled, State, VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

fn corrupt(what: &'static str) -> Message {
	Message::Codec(what)
}

// Inter modes (after the intra ones in libvpx's MB_PREDICTION_MODE).
const NEARESTMV: u8 = 5;
const NEARMV: u8 = 6;
const ZEROMV: u8 = 7;
const NEWMV: u8 = 8;
const SPLITMV: u8 = 9;

const INTRA_FRAME: u8 = 0;
const LAST_FRAME: u8 = 1;
const GOLDEN_FRAME: u8 = 2;
const ALTREF_FRAME: u8 = 3;

// Trees (`entropymode.c`, `entropymv.c`): pairs of children, leaves as -value.
const YMODE_TREE: [i8; 8] =
	[-(DC_PRED as i8), 2, 4, 6, -(V_PRED as i8), -(H_PRED as i8), -(TM_PRED as i8), -(B_PRED as i8)];
const KF_YMODE_TREE: [i8; 8] =
	[-(B_PRED as i8), 2, 4, 6, -(DC_PRED as i8), -(V_PRED as i8), -(H_PRED as i8), -(TM_PRED as i8)];
const UV_MODE_TREE: [i8; 6] =
	[-(DC_PRED as i8), 2, -(V_PRED as i8), 4, -(H_PRED as i8), -(TM_PRED as i8)];
const BMODE_TREE: [i8; 18] = [
	-(dsp::B_DC_PRED as i8),
	2,
	-(dsp::B_TM_PRED as i8),
	4,
	-(dsp::B_VE_PRED as i8),
	6,
	8,
	12,
	-(dsp::B_HE_PRED as i8),
	10,
	-(dsp::B_RD_PRED as i8),
	-(dsp::B_VR_PRED as i8),
	-(dsp::B_LD_PRED as i8),
	14,
	-(dsp::B_VL_PRED as i8),
	16,
	-(dsp::B_HD_PRED as i8),
	-(dsp::B_HU_PRED as i8),
];
const SMALL_MV_TREE: [i8; 14] = [2, 8, 4, 6, -0, -1, -2, -3, 10, 12, -4, -5, -6, -7];

/// Sub-MV reference probabilities by left/above context
/// (`vp8_sub_mv_ref_prob3`, indexed `above_zero << 2 | left_zero << 1 | equal`).
const SUB_MV_REF_PROBS: [[u8; 3]; 8] = [
	[147, 136, 18],
	[223, 1, 34],
	[106, 145, 1],
	[208, 1, 1],
	[179, 121, 1],
	[223, 1, 34],
	[179, 121, 1],
	[208, 1, 1],
];
/// First block of each partition, by split type (`vp8_mbsplit_offset`).
const MBSPLIT_OFFSET: [[u8; 16]; 4] = [
	[0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 2, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];
/// Which partition each 4x4 block belongs to (`vp8_mbsplits`).
const MBSPLITS: [[u8; 16]; 4] = [
	[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
	[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
	[0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3],
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];
const MBSPLIT_COUNT: [usize; 4] = [2, 2, 4, 16];

// Tokens (`detokenize.c`).
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const CAT3: &[u8] = &[173, 148, 140];
const CAT4: &[u8] = &[176, 155, 140, 135];
const CAT5: &[u8] = &[180, 157, 141, 134, 130];
const CAT6: &[u8] = &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];
const CAT3456: [&[u8]; 4] = [CAT3, CAT4, CAT5, CAT6];

/// Margins (1/8 pel) by which motion vectors may point outside the frame
/// before prediction needs clamping.
const MARGIN: i32 = 16 << 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mv {
	pub row: i16,
	pub col: i16,
}

impl Mv {
	const ZERO: Mv = Mv { row: 0, col: 0 };

	fn is_zero(self) -> bool {
		self == Self::ZERO
	}
}

/// Per-macroblock mode information (libvpx `MODE_INFO`).
#[derive(Debug, Clone, Copy)]
struct MbInfo {
	y_mode: u8,
	uv_mode: u8,
	ref_frame: u8,
	segment: u8,
	/// No non-zero coefficient (`mb_skip_coeff`).
	skip: bool,
	need_clamp: bool,
	partitioning: u8,
	mv: Mv,
	bmodes: [u8; 16],
	bmvs: [Mv; 16],
}

impl Default for MbInfo {
	/// The state of the macroblocks outside the frame.
	fn default() -> Self {
		Self {
			y_mode: DC_PRED,
			uv_mode: DC_PRED,
			ref_frame: INTRA_FRAME,
			segment: 0,
			skip: false,
			need_clamp: false,
			partitioning: 0,
			mv: Mv::ZERO,
			bmodes: [dsp::B_DC_PRED; 16],
			bmvs: [Mv::ZERO; 16],
		}
	}
}

/// Probabilities that persist between frames (libvpx `FRAME_CONTEXT`).
#[derive(Debug, Clone)]
struct Probs {
	coef: [[[[u8; 11]; 3]; 8]; 4],
	ymode: [u8; 4],
	uv_mode: [u8; 3],
	mv: [[u8; 19]; 2],
}

impl Default for Probs {
	fn default() -> Self {
		Self {
			coef: DEFAULT_COEF_PROBS,
			ymode: YMODE_PROBS,
			uv_mode: UV_MODE_PROBS,
			mv: DEFAULT_MV_PROBS,
		}
	}
}

/// One plane of a picture, sized to whole macroblocks.
#[derive(Debug, Clone)]
pub struct Plane {
	pub data: Pooled<u8>,
	pub stride: usize,
	/// Visible size.
	pub width: usize,
	pub height: usize,
}

/// A decoded picture (planes cover whole macroblocks).
#[derive(Debug, Clone)]
pub struct Picture {
	pub planes: [Plane; 3],
	pub width: u32,
	pub height: u32,
	pub key_frame: bool,
}

impl Picture {
	fn new(pool: &BufferPool, mb_cols: usize, mb_rows: usize, width: u32, height: u32) -> Self {
		let plane = |sub: usize, w: usize, h: usize| {
			let stride = (mb_cols * 16) >> sub;
			Plane {
				data: pool.take_filled(stride * ((mb_rows * 16) >> sub), 0),
				stride,
				width: w,
				height: h,
			}
		};
		let (w, h) = (width as usize, height as usize);
		let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
		Self {
			planes: [plane(0, w, h), plane(1, cw, ch), plane(1, cw, ch)],
			width,
			height,
			key_frame: false,
		}
	}

	fn reference(&self, plane: usize) -> RefPlane<'_> {
		let p = &self.planes[plane];
		RefPlane { data: &p.data, stride: p.stride, width: p.stride, height: p.data.len() / p.stride }
	}

	/// The visible picture as a 4:2:0 frame; the planes are views into
	/// this picture (no copy).
	pub fn to_frame(self: &Arc<Self>, pts: Option<core::Time>) -> VideoFrame {
		let keyframe = if self.key_frame { Keyframe::Key } else { Keyframe::NonKey };
		fn plane0(p: &Picture) -> &[u8] {
			&p.planes[0].data
		}
		fn plane1(p: &Picture) -> &[u8] {
			&p.planes[1].data
		}
		fn plane2(p: &Picture) -> &[u8] {
			&p.planes[2].data
		}
		let project: [fn(&Picture) -> &[u8]; 3] = [plane0, plane1, plane2];
		let planes = self.planes.iter().zip(project).filter_map(|(p, f)| {
			PlaneRef::new(FrameBuffer::project(self.clone(), f), 0, p.stride, p.width, p.height)
		});
		let pixel = video::Pixel { depth: 8, format: video::PixelFormat::YUV420 };
		VideoFrame::from_planes(planes, self.width, self.height, pixel, keyframe, pts)
	}

	/// The visible picture as planar 8-bit YUV 4:2:0.
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut out = Vec::new();
		for p in &self.planes {
			for r in 0..p.height {
				out.extend_from_slice(&p.data[r * p.stride..r * p.stride + p.width]);
			}
		}
		out
	}
}

/// Frame-level header fields used during reconstruction.
#[derive(Debug, Clone, Default)]
struct FrameHeader {
	key_frame: bool,
	version: u8,
	show_frame: bool,
	filter_simple: bool,
	filter_level: u8,
	sharpness: u8,
	partitions: usize,
	base_q: i32,
	q_deltas: [i32; 5],
	refresh_golden: bool,
	refresh_altref: bool,
	copy_to_golden: u8,
	copy_to_altref: u8,
	refresh_last: bool,
	refresh_probs: bool,
	skip_enabled: bool,
	prob_skip: u8,
	prob_intra: u8,
	prob_last: u8,
	prob_golden: u8,
}

/// Segmentation and loop filter delta state that persists between frames.
#[derive(Debug, Clone, Default)]
struct Segmentation {
	enabled: bool,
	update_map: bool,
	absolute: bool,
	quant: [i32; 4],
	filter: [i32; 4],
	tree_probs: [u8; 3],
	deltas_enabled: bool,
	ref_deltas: [i32; 4],
	mode_deltas: [i32; 4],
}

/// Dequantization factors of one quantizer index: `[dc, ac]` for Y1, Y2
/// and chroma.
#[derive(Debug, Clone, Copy, Default)]
struct Dequant {
	y1: [i16; 2],
	y2: [i16; 2],
	uv: [i16; 2],
}

fn dequant(q: i32, d: &[i32; 5]) -> Dequant {
	let at = |delta: i32| (q + delta).clamp(0, 127) as usize;
	let y2ac = (AC_QLOOKUP[at(d[2])] * 101_581) >> 16;
	Dequant {
		y1: [DC_QLOOKUP[at(d[0])] as i16, AC_QLOOKUP[at(0)] as i16],
		y2: [(DC_QLOOKUP[at(d[1])] * 2) as i16, y2ac.max(8) as i16],
		uv: [DC_QLOOKUP[at(d[3])].min(132) as i16, AC_QLOOKUP[at(d[4])] as i16],
	}
}

/// The decoder: references, persistent probabilities and per-macroblock
/// state (FFmpeg `VP8Context`, libvpx `VP8D_COMP`).
pub struct Vp8DecoderState {
	pool: BufferPool,
	width: u32,
	height: u32,
	mb_cols: usize,
	mb_rows: usize,
	mbs: Vec<MbInfo>,
	probs: Probs,
	seg: Segmentation,
	sign_bias: [bool; 4],
	last: Option<Arc<Picture>>,
	golden: Option<Arc<Picture>>,
	altref: Option<Arc<Picture>>,
	got_keyframe: bool,
}

impl std::fmt::Debug for Vp8DecoderState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Vp8DecoderState")
			.field("width", &self.width)
			.field("height", &self.height)
			.finish()
	}
}

impl Clone for Vp8DecoderState {
	fn clone(&self) -> Self {
		Self {
			pool: self.pool.clone(),
			width: self.width,
			height: self.height,
			mb_cols: self.mb_cols,
			mb_rows: self.mb_rows,
			mbs: self.mbs.clone(),
			probs: self.probs.clone(),
			seg: self.seg.clone(),
			sign_bias: self.sign_bias,
			last: self.last.clone(),
			golden: self.golden.clone(),
			altref: self.altref.clone(),
			got_keyframe: self.got_keyframe,
		}
	}
}

impl Default for Vp8DecoderState {
	fn default() -> Self {
		Self::new()
	}
}

/// Coefficients and end-of-block positions of one macroblock (blocks 0-15
/// luma, 16-23 chroma, 24 Y2).
struct Residual {
	coeffs: [[i16; 16]; 25],
	eobs: [u8; 25],
}

impl Vp8DecoderState {
	pub fn new() -> Self {
		Self {
			pool: BufferPool::new(4),
			width: 0,
			height: 0,
			mb_cols: 0,
			mb_rows: 0,
			mbs: Vec::new(),
			probs: Probs::default(),
			seg: Segmentation::default(),
			sign_bias: [false; 4],
			last: None,
			golden: None,
			altref: None,
			got_keyframe: false,
		}
	}

	/// Decodes one frame. Returns the picture to show, if the frame is
	/// shown (alt-ref frames usually aren't).
	pub fn decode_frame(&mut self, data: &[u8]) -> Result<Option<Arc<Picture>>> {
		if data.len() < 3 {
			return Err(corrupt("vp8: truncated frame"));
		}
		let tag = u32::from(data[0]) | u32::from(data[1]) << 8 | u32::from(data[2]) << 16;
		let mut h = FrameHeader {
			key_frame: tag & 1 == 0,
			version: ((tag >> 1) & 7) as u8,
			show_frame: (tag >> 4) & 1 == 1,
			..Default::default()
		};
		let first_size = (tag >> 5) as usize;
		let mut rest = &data[3..];
		if h.key_frame {
			if rest.len() < 7 {
				return Err(corrupt("vp8: truncated key frame header"));
			}
			if rest[..3] != [0x9d, 0x01, 0x2a] {
				return Err(corrupt("vp8: invalid key frame start code"));
			}
			let width = u32::from(u16::from_le_bytes([rest[3], rest[4]]) & 0x3fff);
			let height = u32::from(u16::from_le_bytes([rest[5], rest[6]]) & 0x3fff);
			if width == 0 || height == 0 {
				return Err(corrupt("vp8: zero frame size"));
			}
			rest = &rest[7..];
			if width != self.width || height != self.height || self.mbs.is_empty() {
				self.width = width;
				self.height = height;
				self.mb_cols = width.div_ceil(16) as usize;
				self.mb_rows = height.div_ceil(16) as usize;
				self.mbs = vec![MbInfo::default(); self.mb_cols * self.mb_rows];
			}
		} else if !self.got_keyframe {
			return Err(corrupt("vp8: stream does not start with a key frame"));
		}
		if first_size == 0 || first_size > rest.len() {
			return Err(corrupt("vp8: invalid first partition size"));
		}
		// libvpx `vp8_setup_version`: versions 1-3 use bilinear prediction,
		// 3 full-pixel chroma; 4-7 decode as 0.
		let bilinear = matches!(h.version, 1..=3);
		let full_pixel = h.version == 3;

		if h.key_frame {
			self.probs = Probs::default();
			self.seg.quant = [0; 4];
			self.seg.filter = [0; 4];
			self.seg.absolute = false;
			self.seg.ref_deltas = [0; 4];
			self.seg.mode_deltas = [0; 4];
			self.sign_bias = [false; 4];
		}

		// The first partition's decoder runs over the rest of the packet,
		// as libvpx's does.
		let mut bc = BoolDecoder::new(rest);
		if h.key_frame {
			bc.bit(); // colour space
			bc.bit(); // clamping type: pixels are always clamped
		}
		self.read_segmentation(&mut bc);
		h.filter_simple = bc.bit();
		h.filter_level = bc.literal(6) as u8;
		h.sharpness = bc.literal(3) as u8;
		self.read_lf_deltas(&mut bc);
		h.partitions = 1 << bc.literal(2);
		let parts = token_partitions(&rest[first_size..], h.partitions)?;

		h.base_q = bc.literal(7) as i32;
		for d in &mut h.q_deltas {
			*d = if bc.bit() { bc.signed_literal(4) } else { 0 };
		}
		if h.key_frame {
			h.refresh_golden = true;
			h.refresh_altref = true;
		} else {
			h.refresh_golden = bc.bit();
			h.refresh_altref = bc.bit();
			if !h.refresh_golden {
				h.copy_to_golden = bc.literal(2) as u8;
			}
			if !h.refresh_altref {
				h.copy_to_altref = bc.literal(2) as u8;
			}
			self.sign_bias[GOLDEN_FRAME as usize] = bc.bit();
			self.sign_bias[ALTREF_FRAME as usize] = bc.bit();
		}
		h.refresh_probs = bc.bit();
		let saved = (!h.refresh_probs).then(|| self.probs.clone());
		h.refresh_last = h.key_frame || bc.bit();

		for (i, block) in self.probs.coef.iter_mut().enumerate() {
			for (j, band) in block.iter_mut().enumerate() {
				for (k, ctx) in band.iter_mut().enumerate() {
					for (l, p) in ctx.iter_mut().enumerate() {
						if bc.read(COEF_UPDATE_PROBS[i][j][k][l]) {
							*p = bc.literal(8) as u8;
						}
					}
				}
			}
		}

		h.skip_enabled = bc.bit();
		if h.skip_enabled {
			h.prob_skip = bc.literal(8) as u8;
		}
		if !h.key_frame {
			h.prob_intra = bc.literal(8) as u8;
			h.prob_last = bc.literal(8) as u8;
			h.prob_golden = bc.literal(8) as u8;
			if bc.bit() {
				for p in &mut self.probs.ymode {
					*p = bc.literal(8) as u8;
				}
			}
			if bc.bit() {
				for p in &mut self.probs.uv_mode {
					*p = bc.literal(8) as u8;
				}
			}
			for (comp, probs) in self.probs.mv.iter_mut().enumerate() {
				for (p, &up) in probs.iter_mut().zip(&MV_UPDATE_PROBS[comp]) {
					if bc.read(up) {
						let x = bc.literal(7) as u8;
						*p = if x != 0 { x << 1 } else { 1 };
					}
				}
			}
		}

		self.read_modes(&mut bc, &h);
		let mut pic = Picture::new(&self.pool, self.mb_cols, self.mb_rows, self.width, self.height);
		pic.key_frame = h.key_frame;
		self.reconstruct(&mut pic, &h, parts, bilinear, full_pixel)?;
		if h.filter_level != 0 {
			self.loop_filter(&mut pic, &h);
		}

		if let Some(p) = saved {
			self.probs = p;
		}
		if h.key_frame {
			self.got_keyframe = true;
		}
		let pic = Arc::new(pic);
		// Reference updates (`swap_frame_buffers`): copies first (the
		// golden copy sees the new alt-ref), then refreshes.
		match h.copy_to_altref {
			1 => self.altref = self.last.clone(),
			2 => self.altref = self.golden.clone(),
			_ => {}
		}
		match h.copy_to_golden {
			1 => self.golden = self.last.clone(),
			2 => self.golden = self.altref.clone(),
			_ => {}
		}
		if h.refresh_golden {
			self.golden = Some(pic.clone());
		}
		if h.refresh_altref {
			self.altref = Some(pic.clone());
		}
		if h.refresh_last {
			self.last = Some(pic.clone());
		}
		Ok(h.show_frame.then_some(pic))
	}

	fn read_segmentation(&mut self, bc: &mut BoolDecoder) {
		let seg = &mut self.seg;
		seg.enabled = bc.bit();
		seg.update_map = false;
		if !seg.enabled {
			return;
		}
		seg.update_map = bc.bit();
		let update_data = bc.bit();
		if update_data {
			seg.absolute = bc.bit();
			for q in &mut seg.quant {
				*q = if bc.bit() { bc.signed_literal(7) } else { 0 };
			}
			for f in &mut seg.filter {
				*f = if bc.bit() { bc.signed_literal(6) } else { 0 };
			}
		}
		if seg.update_map {
			for p in &mut seg.tree_probs {
				*p = if bc.bit() { bc.literal(8) as u8 } else { 255 };
			}
		}
	}

	fn read_lf_deltas(&mut self, bc: &mut BoolDecoder) {
		let seg = &mut self.seg;
		seg.deltas_enabled = bc.bit();
		if seg.deltas_enabled && bc.bit() {
			for d in seg.ref_deltas.iter_mut().chain(seg.mode_deltas.iter_mut()) {
				if bc.bit() {
					*d = bc.signed_literal(6);
				}
			}
		}
	}

	fn mb(&self, row: isize, col: isize) -> MbInfo {
		if row < 0 || col < 0 || col as usize >= self.mb_cols {
			return MbInfo::default();
		}
		self.mbs[row as usize * self.mb_cols + col as usize]
	}

	/// Modes and motion vectors of every macroblock (`vp8_decode_mode_mvs`).
	fn read_modes(&mut self, bc: &mut BoolDecoder, h: &FrameHeader) {
		let to_bottom_start = ((self.mb_rows as i32 - 1) * 16) << 3;
		let to_right_start = ((self.mb_cols as i32 - 1) * 16) << 3;
		for row in 0..self.mb_rows {
			for col in 0..self.mb_cols {
				let edges = Edges {
					left: -((col as i32 * 16) << 3),
					right: to_right_start - ((col as i32 * 16) << 3),
					top: -((row as i32 * 16) << 3),
					bottom: to_bottom_start - ((row as i32 * 16) << 3),
				};
				let idx = row * self.mb_cols + col;
				let mut mb = self.mbs[idx];
				if self.seg.update_map {
					let p = self.seg.tree_probs;
					mb.segment =
						if bc.read(p[0]) { 2 + u8::from(bc.read(p[2])) } else { u8::from(bc.read(p[1])) };
				} else if h.key_frame {
					mb.segment = 0;
				}
				mb.skip = h.skip_enabled && bc.read(h.prob_skip);
				if h.key_frame {
					self.read_kf_modes(bc, &mut mb, row as isize, col as isize);
				} else {
					self.read_inter_modes(bc, h, &mut mb, row as isize, col as isize, &edges);
				}
				self.mbs[idx] = mb;
			}
		}
	}

	fn read_kf_modes(&self, bc: &mut BoolDecoder, mb: &mut MbInfo, row: isize, col: isize) {
		mb.ref_frame = INTRA_FRAME;
		mb.mv = Mv::ZERO;
		mb.y_mode = bc.tree(&KF_YMODE_TREE, &KF_YMODE_PROBS);
		if mb.y_mode == B_PRED {
			let above = self.mb(row - 1, col);
			let left = self.mb(row, col - 1);
			for i in 0..16 {
				let a = if i < 4 { block_mode(&above, i + 12) } else { mb.bmodes[i - 4] };
				let l = if i & 3 == 0 { block_mode(&left, i + 3) } else { mb.bmodes[i - 1] };
				mb.bmodes[i] = bc.tree(&BMODE_TREE, &KF_BMODE_PROBS[a as usize][l as usize]);
			}
		}
		mb.uv_mode = bc.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS);
	}

	#[allow(clippy::too_many_arguments)]
	fn read_inter_modes(
		&self,
		bc: &mut BoolDecoder,
		h: &FrameHeader,
		mb: &mut MbInfo,
		row: isize,
		col: isize,
		e: &Edges,
	) {
		mb.need_clamp = false;
		if !bc.read(h.prob_intra) {
			mb.ref_frame = INTRA_FRAME;
			mb.mv = Mv::ZERO;
			mb.y_mode = bc.tree(&YMODE_TREE, &self.probs.ymode);
			if mb.y_mode == B_PRED {
				for m in &mut mb.bmodes {
					*m = bc.tree(&BMODE_TREE, &BMODE_PROBS);
				}
			}
			mb.uv_mode = bc.tree(&UV_MODE_TREE, &self.probs.uv_mode);
			return;
		}
		mb.ref_frame = if bc.read(h.prob_last) {
			GOLDEN_FRAME + u8::from(bc.read(h.prob_golden))
		} else {
			LAST_FRAME
		};
		let above = self.mb(row - 1, col);
		let left = self.mb(row, col - 1);
		let above_left = self.mb(row - 1, col - 1);

		// Near motion vector search (`read_mb_modes_mv`).
		let mut near = [Mv::ZERO; 4];
		let mut cnt = [0usize; 4];
		let mut n = 0usize; // index of the last stored candidate
		let sign_bias = self.sign_bias;
		let bias = |m: &MbInfo| {
			let mut v = m.mv;
			if sign_bias[m.ref_frame as usize] != sign_bias[mb.ref_frame as usize] {
				v.row = v.row.wrapping_neg();
				v.col = v.col.wrapping_neg();
			}
			v
		};
		if above.ref_frame != INTRA_FRAME {
			if !above.mv.is_zero() {
				n += 1;
				near[n] = bias(&above);
				cnt[n] += 2;
			} else {
				cnt[0] += 2;
			}
		}
		for (m, weight) in [(&left, 2), (&above_left, 1)] {
			if m.ref_frame == INTRA_FRAME {
				continue;
			}
			if !m.mv.is_zero() {
				let v = bias(m);
				if v != near[n] {
					n += 1;
					near[n] = v;
				}
				cnt[n] += weight;
			} else {
				cnt[0] += weight;
			}
		}

		if !bc.read(MODE_CONTEXTS[cnt[0]][0]) {
			mb.y_mode = ZEROMV;
			mb.mv = Mv::ZERO;
			return;
		}
		// Merge the above-left candidate into the nearest when they match.
		if cnt[3] > 0 && near[n] == near[1] {
			cnt[1] += 1;
		}
		if cnt[2] > cnt[1] {
			cnt.swap(1, 2);
			near.swap(1, 2);
		}
		if !bc.read(MODE_CONTEXTS[cnt[1]][1]) {
			mb.y_mode = NEARESTMV;
			mb.mv = clamp_mv2(near[1], e);
			return;
		}
		if !bc.read(MODE_CONTEXTS[cnt[2]][2]) {
			mb.y_mode = NEARMV;
			mb.mv = clamp_mv2(near[2], e);
			return;
		}
		let best = clamp_mv2(near[usize::from(cnt[1] >= cnt[0])], e);
		let split_cnt = usize::from(above.y_mode == SPLITMV) * 2
			+ usize::from(left.y_mode == SPLITMV) * 2
			+ usize::from(above_left.y_mode == SPLITMV);
		let bounds = Edges {
			left: e.left - MARGIN,
			right: e.right + MARGIN,
			top: e.top - MARGIN,
			bottom: e.bottom + MARGIN,
		};
		if bc.read(MODE_CONTEXTS[split_cnt][3]) {
			self.read_split_mv(bc, mb, &left, &above, best, &bounds);
			mb.mv = mb.bmvs[15];
			mb.y_mode = SPLITMV;
		} else {
			let r = read_mv_component(bc, &self.probs.mv[0]) * 2;
			let c = read_mv_component(bc, &self.probs.mv[1]) * 2;
			mb.mv = Mv { row: (r as i16).wrapping_add(best.row), col: (c as i16).wrapping_add(best.col) };
			mb.need_clamp = out_of_bounds(mb.mv, &bounds);
			mb.y_mode = NEWMV;
		}
	}

	fn read_split_mv(
		&self,
		bc: &mut BoolDecoder,
		mb: &mut MbInfo,
		left: &MbInfo,
		above: &MbInfo,
		best: Mv,
		bounds: &Edges,
	) {
		let s = if bc.read(110) { if bc.read(111) { usize::from(bc.read(150)) } else { 2 } } else { 3 };
		for j in 0..MBSPLIT_COUNT[s] {
			let k = usize::from(MBSPLIT_OFFSET[s][j]);
			let left_mv = if k & 3 == 0 {
				if left.y_mode == SPLITMV { left.bmvs[k + 3] } else { left.mv }
			} else {
				mb.bmvs[k - 1]
			};
			let above_mv = if k >> 2 == 0 {
				if above.y_mode == SPLITMV { above.bmvs[k + 12] } else { above.mv }
			} else {
				mb.bmvs[k - 4]
			};
			let ctx = usize::from(above_mv.is_zero()) << 2
				| usize::from(left_mv.is_zero()) << 1
				| usize::from(left_mv == above_mv);
			let p = SUB_MV_REF_PROBS[ctx];
			let mv = if !bc.read(p[0]) {
				left_mv
			} else if !bc.read(p[1]) {
				above_mv
			} else if !bc.read(p[2]) {
				Mv::ZERO
			} else {
				let r = read_mv_component(bc, &self.probs.mv[0]) * 2;
				let c = read_mv_component(bc, &self.probs.mv[1]) * 2;
				Mv { row: (r as i16).wrapping_add(best.row), col: (c as i16).wrapping_add(best.col) }
			};
			mb.need_clamp |= out_of_bounds(mv, bounds);
			for b in 0..16 {
				if usize::from(MBSPLITS[s][b]) == j {
					mb.bmvs[b] = mv;
				}
			}
		}
		mb.partitioning = s as u8;
	}

	/// Decodes residuals and reconstructs every macroblock.
	fn reconstruct(
		&mut self,
		pic: &mut Picture,
		h: &FrameHeader,
		parts: Vec<&[u8]>,
		bilinear: bool,
		full_pixel: bool,
	) -> Result<()> {
		let mut decoders: Vec<BoolDecoder> = parts.into_iter().map(BoolDecoder::new).collect();
		let mut above_ctx = vec![[0u8; 9]; self.mb_cols];
		let seg_dequant: Vec<Dequant> = (0..4)
			.map(|s| {
				let q = if self.seg.enabled {
					let v = self.seg.quant[s];
					if self.seg.absolute { v } else { h.base_q + v }
				} else {
					h.base_q
				};
				dequant(q.clamp(0, 127), &h.q_deltas)
			})
			.collect();
		let refs = [None, self.last.clone(), self.golden.clone(), self.altref.clone()];
		if !h.key_frame && refs[1..].iter().any(Option::is_none) {
			return Err(corrupt("vp8: missing reference frame"));
		}
		let mut res = Residual { coeffs: [[0; 16]; 25], eobs: [0; 25] };
		for row in 0..self.mb_rows {
			let bc = &mut decoders[row % h.partitions];
			let mut left_ctx = [0u8; 9];
			for col in 0..self.mb_cols {
				let idx = row * self.mb_cols + col;
				let mut mb = self.mbs[idx];
				let has_y2 = mb.y_mode != B_PRED && mb.y_mode != SPLITMV;
				if mb.skip {
					// `vp8_reset_mb_tokens_context`
					let keep = if has_y2 { 9 } else { 8 };
					above_ctx[col][..keep].fill(0);
					left_ctx[..keep].fill(0);
					res.eobs = [0; 25];
				} else {
					let total =
						read_tokens(bc, &self.probs.coef, &mut above_ctx[col], &mut left_ctx, has_y2, &mut res);
					mb.skip = total == 0;
					self.mbs[idx].skip = mb.skip;
				}
				let dq = seg_dequant[usize::from(mb.segment)];
				if mb.ref_frame == INTRA_FRAME {
					intra_mb(pic, &mb, row, col, &mut res, &dq, self.mb_cols);
				} else {
					let r =
						refs[usize::from(mb.ref_frame)].as_deref().ok_or(corrupt("vp8: missing reference"))?;
					inter_mb(pic, r, &mb, row, col, self.mb_cols, self.mb_rows, bilinear, full_pixel);
				}
				if !mb.skip {
					add_residual(pic, &mb, row, col, &mut res, &dq);
				}
			}
		}
		Ok(())
	}

	/// The loop filter over the whole frame, in macroblock order
	/// (`vp8_loop_filter_frame`).
	fn loop_filter(&self, pic: &mut Picture, h: &FrameHeader) {
		// Levels per segment, reference and mode class.
		let mut levels = [[[0u8; 4]; 4]; 4];
		for (s, seg_levels) in levels.iter_mut().enumerate() {
			let mut lvl = i32::from(h.filter_level);
			if self.seg.enabled {
				lvl = if self.seg.absolute { self.seg.filter[s] } else { lvl + self.seg.filter[s] };
				lvl = lvl.clamp(0, 63);
			}
			if !self.seg.deltas_enabled {
				*seg_levels = [[lvl as u8; 4]; 4];
				continue;
			}
			let r = lvl + self.seg.ref_deltas[0];
			seg_levels[0][0] = (r + self.seg.mode_deltas[0]).clamp(0, 63) as u8;
			seg_levels[0][1] = r.clamp(0, 63) as u8;
			for rf in 1..4 {
				let r = lvl + self.seg.ref_deltas[rf];
				for m in 1..4 {
					seg_levels[rf][m] = (r + self.seg.mode_deltas[m]).clamp(0, 63) as u8;
				}
			}
		}
		let limits = |level: u8| {
			let sharp = u32::from(h.sharpness);
			let mut interior = u32::from(level) >> u32::from(sharp > 0) >> u32::from(sharp > 4);
			if sharp > 0 {
				interior = interior.min(9 - sharp);
			}
			let interior = interior.max(1);
			let hev = match (h.key_frame, level) {
				(_, 0..=14) => 0,
				(true, 15..=39) => 1,
				(true, _) => 2,
				(false, 15..=19) => 1,
				(false, 20..=39) => 2,
				(false, _) => 3,
			};
			EdgeLimits {
				mb_limit: ((u32::from(level) + 2) * 2 + interior) as u8,
				sub_limit: (u32::from(level) * 2 + interior) as u8,
				interior: interior as u8,
				hev_thresh: hev,
			}
		};
		for row in 0..self.mb_rows {
			for col in 0..self.mb_cols {
				let mb = &self.mbs[row * self.mb_cols + col];
				let mode_class = match mb.y_mode {
					B_PRED => 0,
					SPLITMV => 3,
					NEARESTMV | NEARMV | NEWMV => 2,
					_ => 1,
				};
				let level = levels[usize::from(mb.segment)][usize::from(mb.ref_frame)][mode_class];
				if level == 0 {
					continue;
				}
				let inner = !(mb.y_mode != B_PRED && mb.y_mode != SPLITMV && mb.skip);
				let lim = limits(level);
				for p in 0..if h.filter_simple { 1 } else { 3 } {
					let (n, stride) =
						if p == 0 { (16, pic.planes[0].stride) } else { (8, pic.planes[p].stride) };
					let buf: &mut [u8] = &mut pic.planes[p].data;
					let origin = row * n * stride + col * n;
					let count = n;
					if h.filter_simple {
						if col > 0 {
							dsp::simple_edge(buf, origin, 1, stride, lim.mb_limit);
						}
						if inner {
							for x in [4, 8, 12] {
								dsp::simple_edge(buf, origin + x, 1, stride, lim.sub_limit);
							}
						}
						if row > 0 {
							dsp::simple_edge(buf, origin, stride, 1, lim.mb_limit);
						}
						if inner {
							for y in [4, 8, 12] {
								dsp::simple_edge(buf, origin + y * stride, stride, 1, lim.sub_limit);
							}
						}
						continue;
					}
					let inner_edges: &[usize] = if n == 16 { &[4, 8, 12] } else { &[4] };
					if col > 0 {
						dsp::filter_edge(buf, origin, 1, stride, count, &lim, true);
					}
					if inner {
						for &x in inner_edges {
							dsp::filter_edge(buf, origin + x, 1, stride, count, &lim, false);
						}
					}
					if row > 0 {
						dsp::filter_edge(buf, origin, stride, 1, count, &lim, true);
					}
					if inner {
						for &y in inner_edges {
							dsp::filter_edge(buf, origin + y * stride, stride, 1, count, &lim, false);
						}
					}
				}
			}
		}
	}
}

/// Distances (1/8 pel) from a macroblock to the frame edges.
#[derive(Debug, Clone, Copy)]
struct Edges {
	left: i32,
	right: i32,
	top: i32,
	bottom: i32,
}

/// `vp8_clamp_mv2`.
fn clamp_mv2(mv: Mv, e: &Edges) -> Mv {
	let clamp = |v: i16, lo: i32, hi: i32| {
		let v = i32::from(v);
		(if v < lo {
			lo
		} else if v > hi {
			hi
		} else {
			v
		}) as i16
	};
	Mv {
		row: clamp(mv.row, e.top - MARGIN, e.bottom + MARGIN),
		col: clamp(mv.col, e.left - MARGIN, e.right + MARGIN),
	}
}

/// `vp8_check_mv_bounds`.
fn out_of_bounds(mv: Mv, b: &Edges) -> bool {
	let (r, c) = (i32::from(mv.row), i32::from(mv.col));
	c < b.left || c > b.right || r < b.top || r > b.bottom
}

/// The sub-block mode a neighbouring macroblock implies for context
/// (`above_block_mode` / `left_block_mode`).
fn block_mode(m: &MbInfo, b: usize) -> u8 {
	match m.y_mode {
		B_PRED => m.bmodes[b],
		V_PRED => dsp::B_VE_PRED,
		H_PRED => dsp::B_HE_PRED,
		TM_PRED => dsp::B_TM_PRED,
		_ => dsp::B_DC_PRED,
	}
}

/// One motion vector component (`read_mvcomponent`), in quarter pixels.
fn read_mv_component(bc: &mut BoolDecoder, p: &[u8; 19]) -> i32 {
	const IS_SHORT: usize = 0;
	const SIGN: usize = 1;
	const SHORT: usize = 2;
	const BITS: usize = 9;
	let mut x: i32;
	if bc.read(p[IS_SHORT]) {
		x = 0;
		for i in 0..3 {
			x += i32::from(bc.read(p[BITS + i])) << i;
		}
		for i in (4..=9).rev() {
			x += i32::from(bc.read(p[BITS + i])) << i;
		}
		if x & 0xfff0 == 0 || bc.read(p[BITS + 3]) {
			x += 8;
		}
	} else {
		x = i32::from(bc.tree(&SMALL_MV_TREE, &p[SHORT..SHORT + 7]));
	}
	if x != 0 && bc.read(p[SIGN]) {
		x = -x;
	}
	x
}

/// Splits the token data into its partitions (sizes are 3-byte
/// little-endian, the last partition takes the rest).
fn token_partitions(data: &[u8], count: usize) -> Result<Vec<&[u8]>> {
	let sizes_len = 3 * (count - 1);
	if data.len() < sizes_len {
		return Err(corrupt("vp8: truncated partition sizes"));
	}
	let (sizes, mut rest) = data.split_at(sizes_len);
	let mut parts = Vec::with_capacity(count);
	for i in 0..count - 1 {
		let s = &sizes[3 * i..3 * i + 3];
		let size = usize::from(s[0]) | usize::from(s[1]) << 8 | usize::from(s[2]) << 16;
		if size > rest.len() {
			return Err(corrupt("vp8: truncated partition"));
		}
		let (p, r) = rest.split_at(size);
		parts.push(p);
		rest = r;
	}
	parts.push(rest);
	Ok(parts)
}

/// Coefficients of one block (`GetCoeffs`): returns the position after the
/// last decoded coefficient (0 for none). `n` is the first position.
fn read_coeffs(
	bc: &mut BoolDecoder,
	probs: &[[[u8; 11]; 3]; 8],
	ctx: usize,
	mut n: usize,
	out: &mut [i16; 16],
) -> usize {
	let mut p = &probs[n][ctx];
	if !bc.read(p[0]) {
		return 0;
	}
	loop {
		n += 1;
		if !bc.read(p[1]) {
			p = &probs[BANDS[n]][0];
		} else {
			let v: i32;
			if !bc.read(p[2]) {
				p = &probs[BANDS[n]][1];
				v = 1;
			} else {
				if !bc.read(p[3]) {
					if !bc.read(p[4]) {
						v = 2;
					} else {
						v = 3 + i32::from(bc.read(p[5]));
					}
				} else if !bc.read(p[6]) {
					if !bc.read(p[7]) {
						v = 5 + i32::from(bc.read(159));
					} else {
						v = 7 + 2 * i32::from(bc.read(165)) + i32::from(bc.read(145));
					}
				} else {
					let bit1 = usize::from(bc.read(p[8]));
					let bit0 = usize::from(bc.read(p[9 + bit1]));
					let cat = 2 * bit1 + bit0;
					let mut e = 0;
					for &t in CAT3456[cat] {
						e = 2 * e + i32::from(bc.read(t));
					}
					v = e + 3 + (8 << cat);
				}
				p = &probs[BANDS[n]][2];
			}
			out[ZIGZAG[n - 1]] = bc.sign(v) as i16;
			if n == 16 || !bc.read(p[0]) {
				return n;
			}
		}
		if n == 16 {
			return 16;
		}
	}
}

/// All coefficients of a macroblock (`vp8_decode_mb_tokens`); returns the
/// total of end-of-block positions (0 when the macroblock has none).
fn read_tokens(
	bc: &mut BoolDecoder,
	probs: &[[[[u8; 11]; 3]; 8]; 4],
	above: &mut [u8; 9],
	left: &mut [u8; 9],
	has_y2: bool,
	res: &mut Residual,
) -> i32 {
	res.coeffs = [[0; 16]; 25];
	let mut total = 0i32;
	let (y_type, first) = if has_y2 {
		let ctx = usize::from(above[8] + left[8]);
		let nz = read_coeffs(bc, &probs[1], ctx, 0, &mut res.coeffs[24]);
		above[8] = u8::from(nz > 0);
		left[8] = above[8];
		res.eobs[24] = nz as u8;
		total += nz as i32 - 16;
		(0, 1)
	} else {
		(3, 0)
	};
	for i in 0..16 {
		let (a, l) = (i & 3, i >> 2);
		let ctx = usize::from(above[a] + left[l]);
		let nz = read_coeffs(bc, &probs[y_type], ctx, first, &mut res.coeffs[i]);
		above[a] = u8::from(nz > 0);
		left[l] = above[a];
		let nz = nz + first;
		res.eobs[i] = nz as u8;
		total += nz as i32;
	}
	for i in 16..24 {
		let a = 4 + (usize::from(i > 19) << 1) + (i & 1);
		let l = 4 + (usize::from(i > 19) << 1) + usize::from((i & 3) > 1);
		let ctx = usize::from(above[a] + left[l]);
		let nz = read_coeffs(bc, &probs[2], ctx, 0, &mut res.coeffs[i]);
		above[a] = u8::from(nz > 0);
		left[l] = above[a];
		res.eobs[i] = nz as u8;
		total += nz as i32;
	}
	total
}

/// Edge pixels of a macroblock for intra prediction, following libvpx's
/// frame setup: 127 above the frame (above-right included), 129 left of
/// it, the corner 127 on the first row and 129 below it; past the right
/// edge the above-right pixels repeat the row's last pixel.
struct IntraEdges {
	/// Corner, then `n` pixels above, then 4 above-right (luma).
	above: [u8; 21],
	left: [u8; 16],
}

fn intra_edges(plane: &Plane, n: usize, row: usize, col: usize, mb_cols: usize) -> IntraEdges {
	let mut e = IntraEdges { above: [127; 21], left: [129; 16] };
	let (x0, y0) = (col * n, row * n);
	let s = plane.stride;
	if row > 0 {
		let above = &plane.data[(y0 - 1) * s..y0 * s];
		e.above[1..=n].copy_from_slice(&above[x0..x0 + n]);
		if n == 16 {
			if col + 1 < mb_cols {
				e.above[17..21].copy_from_slice(&above[x0 + 16..x0 + 20]);
			} else {
				e.above[17..21].fill(above[s - 1]);
			}
		}
		e.above[0] = if col > 0 { above[x0 - 1] } else { 129 };
	}
	if col > 0 {
		for (i, l) in e.left[..n].iter_mut().enumerate() {
			*l = plane.data[(y0 + i) * s + x0 - 1];
		}
	}
	e
}

fn intra_mb(
	pic: &mut Picture,
	mb: &MbInfo,
	row: usize,
	col: usize,
	res: &mut Residual,
	dq: &Dequant,
	mb_cols: usize,
) {
	let (have_above, have_left) = (row > 0, col > 0);
	// Chroma first, as libvpx does (it doesn't matter: planes are apart).
	for p in 1..3 {
		let e = intra_edges(&pic.planes[p], 8, row, col, mb_cols);
		let mut out = [0u8; 64];
		dsp::predict_mb(
			mb.uv_mode,
			8,
			&e.above[1..9],
			&e.left[..8],
			e.above[0],
			have_above,
			have_left,
			&mut out,
		);
		store(&mut pic.planes[p], &out, 8, row, col);
	}
	let e = intra_edges(&pic.planes[0], 16, row, col, mb_cols);
	if mb.y_mode != B_PRED {
		let mut out = [0u8; 256];
		dsp::predict_mb(
			mb.y_mode,
			16,
			&e.above[1..17],
			&e.left,
			e.above[0],
			have_above,
			have_left,
			&mut out,
		);
		store(&mut pic.planes[0], &out, 16, row, col);
		return;
	}
	// Sub-blocks: a 17-row work area whose row 0 is the edge above (with
	// the above-right pixels copied down to rows 4, 8 and 12, libvpx's
	// `intra_prediction_down_copy`) and column 0 the edge to the left.
	const W: usize = 21;
	let mut buf = [0u8; 17 * W];
	buf[..W].copy_from_slice(&e.above);
	for i in 0..16 {
		buf[(i + 1) * W] = e.left[i];
	}
	for r in [4, 8, 12] {
		let ar = [buf[17], buf[18], buf[19], buf[20]];
		buf[r * W + 17..r * W + 21].copy_from_slice(&ar);
	}
	for b in 0..16 {
		let (bx, by) = (b & 3, b >> 2);
		let (x, y) = (4 * bx, 4 * by); // top-left of the corner pixel
		let mut above = [0u8; 9];
		above.copy_from_slice(&buf[y * W + x..y * W + x + 9]);
		let mut left = [0u8; 4];
		for (i, l) in left.iter_mut().enumerate() {
			*l = buf[(y + 1 + i) * W + x];
		}
		let mut out = [[0u8; 4]; 4];
		dsp::predict_4x4(mb.bmodes[b], &above, &left, &mut out);
		let at = (y + 1) * W + x + 1;
		for (r, line) in out.iter().enumerate() {
			buf[at + r * W..at + r * W + 4].copy_from_slice(line);
		}
		if !mb.skip {
			let eob = res.eobs[b];
			if eob > 1 {
				let c = dequantize(&res.coeffs[b], dq.y1);
				dsp::idct_add(&c, &mut buf, at, W);
			} else if eob == 1 {
				dsp::dc_add(mul16(res.coeffs[b][0], dq.y1[0]), &mut buf, at, W);
			}
		}
	}
	let plane = &mut pic.planes[0];
	let s = plane.stride;
	for r in 0..16 {
		let at = (row * 16 + r) * s + col * 16;
		plane.data[at..at + 16].copy_from_slice(&buf[(r + 1) * W + 1..(r + 1) * W + 17]);
	}
}

fn store(plane: &mut Plane, block: &[u8], n: usize, row: usize, col: usize) {
	let s = plane.stride;
	for r in 0..n {
		let at = (row * n + r) * s + col * n;
		plane.data[at..at + n].copy_from_slice(&block[r * n..r * n + n]);
	}
}

#[inline]
fn mul16(a: i16, b: i16) -> i16 {
	(i32::from(a) * i32::from(b)) as i16
}

fn dequantize(c: &[i16; 16], dq: [i16; 2]) -> [i16; 16] {
	let mut out = [0i16; 16];
	out[0] = mul16(c[0], dq[0]);
	for i in 1..16 {
		out[i] = mul16(c[i], dq[1]);
	}
	out
}

/// `clamp_mv_to_umv_border`.
fn clamp_umv(mv: Mv, e: &Edges) -> Mv {
	let (mut r, mut c) = (i32::from(mv.row), i32::from(mv.col));
	if c < e.left - (19 << 3) {
		c = e.left - (16 << 3);
	} else if c > e.right + (18 << 3) {
		c = e.right + (16 << 3);
	}
	if r < e.top - (19 << 3) {
		r = e.top - (16 << 3);
	} else if r > e.bottom + (18 << 3) {
		r = e.bottom + (16 << 3);
	}
	Mv { row: r as i16, col: c as i16 }
}

/// `clamp_uvmv_to_umv_border`.
fn clamp_uv_umv(mv: Mv, e: &Edges) -> Mv {
	let (mut r, mut c) = (i32::from(mv.row), i32::from(mv.col));
	if 2 * c < e.left - (19 << 3) {
		c = (e.left - (16 << 3)) >> 1;
	}
	if 2 * c > e.right + (18 << 3) {
		c = (e.right + (16 << 3)) >> 1;
	}
	if 2 * r < e.top - (19 << 3) {
		r = (e.top - (16 << 3)) >> 1;
	}
	if 2 * r > e.bottom + (18 << 3) {
		r = (e.bottom + (16 << 3)) >> 1;
	}
	Mv { row: r as i16, col: c as i16 }
}

#[allow(clippy::too_many_arguments)]
fn predict_block(
	dst: &mut Plane,
	src: &RefPlane,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mv: Mv,
	bilinear: bool,
) {
	let (mr, mc) = (i32::from(mv.row), i32::from(mv.col));
	let sx = x as isize + (mc >> 3) as isize;
	let sy = y as isize + (mr >> 3) as isize;
	let s = dst.stride;
	dsp::RefPlane::predict(
		src,
		sx,
		sy,
		(mc & 7) as usize,
		(mr & 7) as usize,
		w,
		h,
		bilinear,
		&mut dst.data[y * s + x..],
		s,
	);
}

#[allow(clippy::too_many_arguments)]
fn inter_mb(
	pic: &mut Picture,
	r: &Picture,
	mb: &MbInfo,
	row: usize,
	col: usize,
	mb_cols: usize,
	mb_rows: usize,
	bilinear: bool,
	full_pixel: bool,
) {
	let e = Edges {
		left: -((col as i32 * 16) << 3),
		right: (((mb_cols - 1 - col) * 16) as i32) << 3,
		top: -((row as i32 * 16) << 3),
		bottom: (((mb_rows - 1 - row) * 16) as i32) << 3,
	};
	let mask = if full_pixel { !7 } else { !0 };
	let (x0, y0) = (col * 16, row * 16);
	let luma = r.reference(0);
	if mb.y_mode != SPLITMV {
		let mut mv = mb.mv;
		if mb.need_clamp {
			mv = clamp_umv(mv, &e);
		}
		predict_block(&mut pic.planes[0], &luma, x0, y0, 16, 16, mv, bilinear);
		// Chroma: half, rounded away from zero.
		let half = |v: i16| {
			let v = i32::from(v);
			let v = (v + (1 | (v >> 31))) / 2;
			(v & mask) as i16
		};
		let uv = clamp_uv_umv(Mv { row: half(mv.row), col: half(mv.col) }, &e);
		for p in 1..3 {
			let src = r.reference(p);
			predict_block(&mut pic.planes[p], &src, x0 / 2, y0 / 2, 8, 8, uv, bilinear);
		}
		return;
	}
	let clamp = |mv: Mv| if mb.need_clamp { clamp_umv(mv, &e) } else { mv };
	if mb.partitioning < 3 {
		for b in [0, 2, 8, 10] {
			let (bx, by) = (b & 3, b >> 2);
			predict_block(
				&mut pic.planes[0],
				&luma,
				x0 + 4 * bx,
				y0 + 4 * by,
				8,
				8,
				clamp(mb.bmvs[b]),
				bilinear,
			);
		}
	} else {
		for b in 0..16 {
			let (bx, by) = (b & 3, b >> 2);
			predict_block(
				&mut pic.planes[0],
				&luma,
				x0 + 4 * bx,
				y0 + 4 * by,
				4,
				4,
				clamp(mb.bmvs[b]),
				bilinear,
			);
		}
	}
	// Chroma: each 4x4 block takes the average of its four luma MVs
	// (`build_4x4uvmvs`).
	for i in 0..2 {
		for j in 0..2 {
			let y = i * 8 + j * 2;
			let avg = |f: fn(&Mv) -> i16| {
				let t: i32 = [y, y + 1, y + 4, y + 5].iter().map(|&k| i32::from(f(&mb.bmvs[k]))).sum();
				let t = t + 4 + ((t >> 31) * 8);
				((t / 8) & mask) as i16
			};
			let mut uv = Mv { row: avg(|m| m.row), col: avg(|m| m.col) };
			if mb.need_clamp {
				uv = clamp_uv_umv(uv, &e);
			}
			for p in 1..3 {
				let src = r.reference(p);
				predict_block(&mut pic.planes[p], &src, x0 / 2 + 4 * j, y0 / 2 + 4 * i, 4, 4, uv, bilinear);
			}
		}
	}
}

/// Adds the residual of a non-skipped macroblock (`decode_macroblock`);
/// B_PRED luma was added during prediction.
fn add_residual(
	pic: &mut Picture,
	mb: &MbInfo,
	row: usize,
	col: usize,
	res: &mut Residual,
	dq: &Dequant,
) {
	if mb.y_mode != B_PRED {
		let mut y_dq = dq.y1;
		if mb.y_mode != SPLITMV {
			let dcs = if res.eobs[24] > 1 {
				let c = dequantize(&res.coeffs[24], dq.y2);
				dsp::inv_walsh(&c)
			} else {
				[dsp::inv_walsh_dc(mul16(res.coeffs[24][0], dq.y2[0])); 16]
			};
			for (b, dc) in dcs.iter().enumerate() {
				res.coeffs[b][0] = *dc;
			}
			// The DCs are already dequantized.
			y_dq[0] = 1;
		}
		let plane = &mut pic.planes[0];
		let s = plane.stride;
		for b in 0..16 {
			let at = (row * 16 + 4 * (b >> 2)) * s + col * 16 + 4 * (b & 3);
			if res.eobs[b] > 1 {
				let c = dequantize(&res.coeffs[b], y_dq);
				dsp::idct_add(&c, &mut plane.data, at, s);
			} else {
				dsp::dc_add(mul16(res.coeffs[b][0], y_dq[0]), &mut plane.data, at, s);
			}
		}
	}
	for p in 1..3 {
		let plane = &mut pic.planes[p];
		let s = plane.stride;
		for k in 0..4 {
			let b = 16 + (p - 1) * 4 + k;
			let at = (row * 8 + 4 * (k >> 1)) * s + col * 8 + 4 * (k & 1);
			if res.eobs[b] > 1 {
				let c = dequantize(&res.coeffs[b], dq.uv);
				dsp::idct_add(&c, &mut plane.data, at, s);
			} else {
				dsp::dc_add(mul16(res.coeffs[b][0], dq.uv[0]), &mut plane.data, at, s);
			}
		}
	}
}

#[derive(Debug, Clone, Default)]
pub struct Vp8Decoder;

impl Vp8Decoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for Vp8Decoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		State::Video(VideoState {
			width: parameters.width.unwrap_or(0),
			height: parameters.height.unwrap_or(0),
			pixel: video::Pixel { depth: 8, format: video::PixelFormat::YUV420 },
			reorder: Vec::new(),
			has_keyframe: false,
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			private: CodecPrivate::Vp8(Box::default()),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Video(video) = state else {
			return Err(Message::Codec("vp8 decoder requires a video state"));
		};
		if !matches!(video.private, CodecPrivate::Vp8(_)) {
			video.private = CodecPrivate::Vp8(Box::default());
		}
		let CodecPrivate::Vp8(dec) = &mut video.private else {
			return Err(Message::Codec("vp8 decoder state missing"));
		};
		let mut frames = Vec::new();
		if let Some(pic) = dec.decode_frame(&packet.data)? {
			video.width = pic.width;
			video.height = pic.height;
			video.has_keyframe |= pic.key_frame;
			frames.push(Frame::Video(pic.to_frame(packet.pts)));
		}
		Ok(DecodeOut { value: frames })
	}

	fn flush(&self, _state: &mut State) -> Result<DecodeOut> {
		Ok(DecodeOut { value: Vec::new() })
	}
}
