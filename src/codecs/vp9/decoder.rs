//! Frame decoding: reference slots, tiles, partitions, block
//! reconstruction and the [`crate::core::Decoder`] implementation (spec
//! §6.4, §8.4 – §8.10; libvpx `vp9_decodeframe.c`, `vp9_decoder.c`).

use std::sync::Arc;

use super::header::{
	self, ColorConfig, CompressedHeader, FrameHeader, Persistent, RefInfo, SEG_ALT_LF, SEG_ALT_Q,
	Segmentation,
};
use super::inter::{self, RefPlane};
use super::intra::{self, Edges, MODE_TO_TX_TYPE};
use super::loopfilter;
use super::mode::{ModeInfo, MvRef};
use super::pixel::Px;
use super::probs::{self, Counts, FrameContext};
use super::range::BoolDecoder;
use super::residual::{self, DCT_DCT, TokenCounts};
use super::tables::{AC_QLOOKUP, DC_QLOOKUP, KF_PARTITION_PROBS};
use super::transform;
use super::{Mv, PARTITION_CONTEXT, PARTITION_TREE, SS_SIZE, SUBSIZE, bs, uv_tx_size};
use crate::core::{
	self, BufferPool, CodecPrivate, CodecState, DecodeOut, Frame, FrameBuffer, Keyframe, Packet,
	Parameters, PlaneRef, Pooled, State, VecPool, VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

#[path = "tile_threads.rs"]
mod tile_threads;
pub(super) use tile_threads::TilePool;

fn corrupt(what: &'static str) -> Message {
	Message::Codec(what)
}

/// One plane of a picture. The buffer covers whole superblocks, so blocks
/// that cross the picture's edge can be written in full. The samples come
/// from the decoder's pools and go back to them when the last reference to
/// the picture (slot or output frame) is dropped. `P` is `u8` for 8-bit
/// streams and `u16` for 10- and 12-bit ones.
#[derive(Debug, Clone)]
pub struct Plane<P = u8> {
	pub data: Pooled<P>,
	pub stride: usize,
	/// Visible size.
	pub width: usize,
	pub height: usize,
	/// Size rounded up to 8 luma pixels (what intra prediction sees).
	pub aligned_w: usize,
	pub aligned_h: usize,
}

/// A decoded picture.
#[derive(Debug, Clone)]
pub struct Picture<P = u8> {
	pub planes: [Plane<P>; 3],
	pub width: u32,
	pub height: u32,
	pub ss_x: usize,
	pub ss_y: usize,
	pub bit_depth: u8,
	pub key_frame: bool,
	/// sRGB colour space: the planes hold G, B and R.
	pub rgb: bool,
}

impl<P: Px> Picture<P> {
	pub(super) fn new(
		pool: &VecPool<P>,
		width: u32,
		height: u32,
		ss_x: usize,
		ss_y: usize,
		bit_depth: u8,
	) -> Self {
		let (w, h) = (width as usize, height as usize);
		let (aw, ah) = (w.next_multiple_of(8), h.next_multiple_of(8));
		let (sw, sh) = (w.next_multiple_of(64), h.next_multiple_of(64));
		let plane = |sx: usize, sy: usize| {
			let stride = sw >> sx;
			Plane {
				data: pool.take_filled(stride * (sh >> sy), P::default()),
				stride,
				width: (w + sx) >> sx,
				height: (h + sy) >> sy,
				aligned_w: aw >> sx,
				aligned_h: ah >> sy,
			}
		};
		Self {
			planes: [plane(0, 0), plane(ss_x, ss_y), plane(ss_x, ss_y)],
			width,
			height,
			ss_x,
			ss_y,
			bit_depth,
			key_frame: false,
			rgb: false,
		}
	}

	/// The visible picture as bytes: planar YUV (two bytes per sample,
	/// little-endian, above 8 bits), or for sRGB streams packed RGB24 at 8
	/// bits and planar G, B, R (FFmpeg's `gbrp10le` / `gbrp12le`) above.
	pub(crate) fn bytes(&self) -> Vec<u8> {
		if self.rgb && !P::WIDE {
			let [g, b, r] = &self.planes;
			let mut out = Vec::with_capacity(g.width * g.height * 3);
			for y in 0..g.height {
				for x in 0..g.width {
					let at = |p: &Plane<P>| p.data[y * p.stride + x].into() as u8;
					out.extend_from_slice(&[at(r), at(g), at(b)]);
				}
			}
			return out;
		}
		let size: usize = self.planes.iter().map(|p| p.width * p.height).sum();
		let mut out = Vec::with_capacity(size * if P::WIDE { 2 } else { 1 });
		for p in &self.planes {
			for r in 0..p.height {
				let row = &p.data[r * p.stride..r * p.stride + p.width];
				if P::WIDE {
					for &v in row {
						out.extend_from_slice(&(v.into() as u16).to_le_bytes());
					}
				} else {
					out.extend(row.iter().map(|&v| v.into() as u8));
				}
			}
		}
		out
	}
}

impl Picture<u8> {
	/// The visible picture as an output frame. YUV planes are views into
	/// this picture (no copy); sRGB pictures (planes G, B, R) are packed
	/// into RGB24, which needs one pass over the samples.
	fn frame(self: &Arc<Self>, pixel: video::Pixel, pts: Option<crate::core::Time>) -> VideoFrame {
		let keyframe = if self.key_frame { Keyframe::Key } else { Keyframe::NonKey };
		if self.rgb {
			return VideoFrame::from_packed(self.bytes(), self.width, self.height, pixel, keyframe, pts);
		}
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
		VideoFrame::from_planes(planes, self.width, self.height, pixel, keyframe, pts)
	}
}

/// A decoded picture of either sample type: what reference slots hold and
/// what [`Vp9DecoderState::decode_packet`] returns. Cloning is cheap (the
/// samples are shared).
#[derive(Debug, Clone)]
pub enum DecodedPicture {
	/// Profiles 0 and 1.
	Narrow(Arc<Picture<u8>>),
	/// Profiles 2 and 3 (10 or 12 bits in `u16` samples).
	Wide(Arc<Picture<u16>>),
}

macro_rules! either {
	($self:expr, $p:ident => $e:expr) => {
		match $self {
			DecodedPicture::Narrow($p) => $e,
			DecodedPicture::Wide($p) => $e,
		}
	};
}

impl DecodedPicture {
	pub fn width(&self) -> u32 {
		either!(self, p => p.width)
	}

	pub fn height(&self) -> u32 {
		either!(self, p => p.height)
	}

	pub fn bit_depth(&self) -> u8 {
		either!(self, p => p.bit_depth)
	}

	/// Chroma subsampling `(ss_x, ss_y)`.
	pub fn subsampling(&self) -> (usize, usize) {
		either!(self, p => (p.ss_x, p.ss_y))
	}

	pub fn key_frame(&self) -> bool {
		either!(self, p => p.key_frame)
	}

	/// sRGB colour space (planes G, B, R).
	pub fn rgb(&self) -> bool {
		either!(self, p => p.rgb)
	}

	/// The RMP pixel layout of [`Self::to_frame`], if there is one (there
	/// is no 4:4:0 layout, nor a high-bit-depth RGB one).
	pub fn pixel(&self) -> Option<video::Pixel> {
		let depth = self.bit_depth();
		if self.rgb() {
			return (depth == 8).then_some(video::Pixel { depth: 8, format: video::PixelFormat::RGB24 });
		}
		let format = match self.subsampling() {
			(1, 1) => video::PixelFormat::YUV420,
			(1, 0) => video::PixelFormat::YUV422,
			(0, 0) => video::PixelFormat::YUV444,
			_ => return None,
		};
		Some(video::Pixel { depth, format })
	}

	/// The visible picture as bytes: planar YUV (two bytes per sample,
	/// little-endian, above 8 bits; FFmpeg's `yuv420p10le` and so on), or
	/// for sRGB streams packed RGB24 at 8 bits and planar G, B, R
	/// (`gbrp10le` / `gbrp12le`) above.
	pub fn to_bytes(&self) -> Vec<u8> {
		either!(self, p => p.bytes())
	}

	/// The visible picture as an output frame. 8-bit YUV planes are views
	/// into the picture (no copy); 10/12-bit samples are written out as
	/// little-endian bytes, RMP's layout for deep samples.
	pub fn to_frame(&self, pixel: video::Pixel, pts: Option<crate::core::Time>) -> VideoFrame {
		match self {
			DecodedPicture::Narrow(p) => p.frame(pixel, pts),
			DecodedPicture::Wide(p) => {
				let keyframe = if p.key_frame { Keyframe::Key } else { Keyframe::NonKey };
				VideoFrame::from_packed(p.bytes(), p.width, p.height, pixel, keyframe, pts)
			}
		}
	}
}

/// A block being decoded: position (in 8x8 units), size and distances to
/// the picture edges in 1/8 pixels (libvpx `mb_to_*_edge`).
#[derive(Debug, Clone, Copy)]
pub struct BlockPos {
	pub row: usize,
	pub col: usize,
	pub bw: usize,
	pub bh: usize,
	pub x_mis: usize,
	pub y_mis: usize,
	pub to_left: i32,
	pub to_right: i32,
	pub to_top: i32,
	pub to_bottom: i32,
	pub have_above: bool,
	pub have_left: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TileBounds {
	pub mi_row_start: usize,
	pub mi_row_end: usize,
	pub mi_col_start: usize,
	pub mi_col_end: usize,
}

fn tile_offset(i: usize, mis: usize, log2: u32) -> usize {
	let sb = mis.div_ceil(8);
	(((i * sb) >> log2) << 3).min(mis)
}

/// libvpx's `struct scale_factors`: how a reference of another size maps
/// onto the frame being decoded (`vp9_setup_scale_factors_for_frame`).
/// Positions scale by `ref_size / frame_size` in 14-bit fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefScale {
	x_fp: i64,
	y_fp: i64,
	/// Source step per output pixel, in 1/16 (16 when not scaled).
	pub step_x: i32,
	pub step_y: i32,
	/// Within 2x smaller and 16x larger than the reference.
	pub valid: bool,
}

const REF_SCALE_SHIFT: u32 = 14;
const REF_NO_SCALE: i64 = 1 << REF_SCALE_SHIFT;

impl Default for RefScale {
	fn default() -> Self {
		Self { x_fp: REF_NO_SCALE, y_fp: REF_NO_SCALE, step_x: 16, step_y: 16, valid: true }
	}
}

impl RefScale {
	/// Scale factors of a `ref_w` x `ref_h` reference for a `w` x `h` frame.
	pub fn new(ref_w: u32, ref_h: u32, w: u32, h: u32) -> Self {
		let valid = 2 * w >= ref_w && 2 * h >= ref_h && w <= 16 * ref_w && h <= 16 * ref_h;
		if !valid || w == 0 || h == 0 {
			return Self { valid: false, ..Self::default() };
		}
		let x_fp = (i64::from(ref_w) << REF_SCALE_SHIFT) / i64::from(w);
		let y_fp = (i64::from(ref_h) << REF_SCALE_SHIFT) / i64::from(h);
		let mut s = Self { x_fp, y_fp, step_x: 16, step_y: 16, valid };
		s.step_x = s.x(16) as i32;
		s.step_y = s.y(16) as i32;
		s
	}

	/// `vp9_is_scaled`.
	pub fn scaled(&self) -> bool {
		self.valid && (self.x_fp != REF_NO_SCALE || self.y_fp != REF_NO_SCALE)
	}

	/// `scaled_x`: a horizontal position or offset in reference units.
	#[inline]
	pub fn x(&self, v: i64) -> i64 {
		(v * self.x_fp) >> REF_SCALE_SHIFT
	}

	/// `scaled_y`.
	#[inline]
	pub fn y(&self, v: i64) -> i64 {
		(v * self.y_fp) >> REF_SCALE_SHIFT
	}
}

/// State of the frame being decoded.
pub struct FrameDecoder<P = u8> {
	pub h: FrameHeader,
	pub ch: CompressedHeader,
	pub fc: FrameContext,
	pub seg: Segmentation,
	pub mi_rows: usize,
	pub mi_cols: usize,
	pub mi: Vec<ModeInfo>,
	/// Loop filter inputs per 8x8 cell.
	pub(super) lf_cells: Vec<loopfilter::LfCell>,
	pub cur_mvs: Vec<MvRef>,
	pub prev_mvs: Option<Arc<Vec<MvRef>>>,
	pub seg_map: Vec<u8>,
	pub last_seg_map: Arc<Vec<u8>>,
	pub counts: Box<Counts>,
	pub counting: bool,
	pub tile: TileBounds,
	pub(super) above_nz: [Vec<u8>; 3],
	pub(super) left_nz: [[u8; 16]; 3],
	pub(super) above_partition: Vec<u8>,
	pub(super) left_partition: [u8; 8],
	pub(super) pic: Picture<P>,
	pub(super) refs: [Option<Arc<Picture<P>>>; 3],
	/// Scale factors of the three references.
	pub(super) ref_scale: [RefScale; 3],
	pub(super) bit_depth: u8,
	/// Largest sample value, `(1 << bit_depth) - 1`.
	pub(super) max: i32,
	/// `[segment][luma, chroma][dc, ac]`.
	pub(super) dequant: [[[i16; 2]; 2]; 8],
	/// Loop filter level `[segment][ref frame][mode delta]`.
	pub(super) lf_table: [[[u8; 2]; 4]; 8],
	pub(super) coefs: Vec<i32>,
	pub(super) scratch: inter::Scratch<P>,
}

impl<P: Px> FrameDecoder<P> {
	pub(super) fn plane_ss(&self, plane: usize) -> (usize, usize) {
		if plane == 0 { (0, 0) } else { (self.pic.ss_x, self.pic.ss_y) }
	}

	/// Splits the tile data into one buffer per tile, in raster order.
	fn split_tiles<'d>(&self, data: &'d [u8]) -> Result<Vec<&'d [u8]>> {
		let tile_cols = 1usize << self.h.log2_tile_cols;
		let tile_rows = 1usize << self.h.log2_tile_rows;
		let mut buffers: Vec<&[u8]> = Vec::with_capacity(tile_cols * tile_rows);
		let mut at = 0usize;
		for i in 0..tile_rows * tile_cols {
			let size = if i == tile_rows * tile_cols - 1 {
				data.len() - at
			} else {
				let bytes = data.get(at..at + 4).ok_or(corrupt("vp9: truncated tile size"))?;
				at += 4;
				u32::from_be_bytes(bytes.try_into().unwrap_or([0; 4])) as usize
			};
			let tile = data.get(at..at.saturating_add(size)).ok_or(corrupt("vp9: truncated tile"))?;
			buffers.push(tile);
			at += size;
		}
		Ok(buffers)
	}

	/// Mode-info rows of tile row `tile_row`.
	fn tile_mi_rows(&self, tile_row: usize) -> (usize, usize) {
		(
			tile_offset(tile_row, self.mi_rows, self.h.log2_tile_rows),
			tile_offset(tile_row + 1, self.mi_rows, self.h.log2_tile_rows),
		)
	}

	/// Mode-info columns of tile column `tile_col`.
	fn tile_mi_cols(&self, tile_col: usize) -> (usize, usize) {
		(
			tile_offset(tile_col, self.mi_cols, self.h.log2_tile_cols),
			tile_offset(tile_col + 1, self.mi_cols, self.h.log2_tile_cols),
		)
	}

	/// Decodes one superblock row of one tile.
	fn decode_sb_row(
		&mut self,
		r: &mut BoolDecoder,
		rows: (usize, usize),
		tile_col: usize,
		mi_row: usize,
	) -> Result<()> {
		let cols = self.tile_mi_cols(tile_col);
		self.tile = TileBounds {
			mi_row_start: rows.0,
			mi_row_end: rows.1,
			mi_col_start: cols.0,
			mi_col_end: cols.1,
		};
		self.left_nz = [[0; 16]; 3];
		self.left_partition = [0; 8];
		let mut mi_col = self.tile.mi_col_start;
		while mi_col < self.tile.mi_col_end {
			self.decode_partition(r, mi_row, mi_col, bs::B64X64, 4)?;
			mi_col += 8;
		}
		if r.overrun() {
			return Err(corrupt("vp9: truncated tile data"));
		}
		Ok(())
	}

	/// Decodes every tile (`decode_tiles`); `data` follows the headers.
	fn decode_tiles(&mut self, data: &[u8]) -> Result<()> {
		let tile_cols = 1usize << self.h.log2_tile_cols;
		let tile_rows = 1usize << self.h.log2_tile_rows;
		let buffers = self.split_tiles(data)?;
		for tile_row in 0..tile_rows {
			let mut readers: Vec<BoolDecoder> = (0..tile_cols)
				.map(|c| {
					BoolDecoder::new(buffers[tile_row * tile_cols + c])
						.ok_or(corrupt("vp9: tile marker bit set"))
				})
				.collect::<Result<_>>()?;
			let rows = self.tile_mi_rows(tile_row);
			let mut mi_row = rows.0;
			while mi_row < rows.1 {
				for (tile_col, r) in readers.iter_mut().enumerate() {
					self.decode_sb_row(r, rows, tile_col, mi_row)?;
				}
				mi_row += 8;
			}
		}
		Ok(())
	}

	fn decode_partition(
		&mut self,
		r: &mut BoolDecoder,
		mi_row: usize,
		mi_col: usize,
		bsize: u8,
		n4x4_l2: u32,
	) -> Result<()> {
		if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
			return Ok(());
		}
		let n8x8_l2 = n4x4_l2 - 1;
		let num_8x8 = 1usize << n8x8_l2;
		let hbs = num_8x8 >> 1;
		let has_rows = mi_row + hbs < self.mi_rows;
		let has_cols = mi_col + hbs < self.mi_cols;

		let bsl = n8x8_l2 as usize;
		let above = usize::from((self.above_partition[mi_col] >> bsl) & 1);
		let left = usize::from((self.left_partition[mi_row & 7] >> bsl) & 1);
		let ctx = left * 2 + above + bsl * 4;
		let probs =
			if self.h.intra_only_frame() { KF_PARTITION_PROBS[ctx] } else { self.fc.partition[ctx] };
		let partition = if has_rows && has_cols {
			r.tree(&PARTITION_TREE, &probs)
		} else if !has_rows && has_cols {
			if r.read(probs[1]) { 3 } else { 1 }
		} else if has_rows && !has_cols {
			if r.read(probs[2]) { 3 } else { 2 }
		} else {
			3
		};
		if self.counting {
			self.counts.partition[ctx][usize::from(partition)] += 1;
		}
		let subsize = SUBSIZE[usize::from(partition)][usize::from(bsize)];
		if hbs == 0 {
			// 8x8: sub-8x8 blocks are coded as one block with 4x4 sub-blocks.
			let sub_w = if partition & 2 != 0 { 1 } else { 2 };
			let sub_h = if partition & 1 != 0 { 1 } else { 2 };
			self.decode_block(r, mi_row, mi_col, subsize, 1, 1, sub_w, sub_h)?;
		} else {
			match partition {
				0 => self.decode_block(r, mi_row, mi_col, subsize, n4x4_l2, n4x4_l2, 2, 2)?,
				1 => {
					self.decode_block(r, mi_row, mi_col, subsize, n4x4_l2, n8x8_l2, 2, 2)?;
					if has_rows {
						self.decode_block(r, mi_row + hbs, mi_col, subsize, n4x4_l2, n8x8_l2, 2, 2)?;
					}
				}
				2 => {
					self.decode_block(r, mi_row, mi_col, subsize, n8x8_l2, n4x4_l2, 2, 2)?;
					if has_cols {
						self.decode_block(r, mi_row, mi_col + hbs, subsize, n8x8_l2, n4x4_l2, 2, 2)?;
					}
				}
				_ => {
					self.decode_partition(r, mi_row, mi_col, subsize, n8x8_l2)?;
					self.decode_partition(r, mi_row, mi_col + hbs, subsize, n8x8_l2)?;
					self.decode_partition(r, mi_row + hbs, mi_col, subsize, n8x8_l2)?;
					self.decode_partition(r, mi_row + hbs, mi_col + hbs, subsize, n8x8_l2)?;
				}
			}
		}
		if bsize >= bs::B8X8 && (bsize == bs::B8X8 || partition != 3) {
			let (a, l) = PARTITION_CONTEXT[usize::from(subsize)];
			self.above_partition[mi_col..mi_col + num_8x8].fill(a);
			let lr = mi_row & 7;
			self.left_partition[lr..(lr + num_8x8).min(8)].fill(l);
		}
		Ok(())
	}

	#[allow(clippy::too_many_arguments)]
	fn decode_block(
		&mut self,
		r: &mut BoolDecoder,
		mi_row: usize,
		mi_col: usize,
		bsize: u8,
		bwl: u32,
		bhl: u32,
		sub_w: usize,
		sub_h: usize,
	) -> Result<()> {
		let bw = 1usize << (bwl - 1);
		let bh = 1usize << (bhl - 1);
		let p = BlockPos {
			row: mi_row,
			col: mi_col,
			bw,
			bh,
			x_mis: bw.min(self.mi_cols - mi_col),
			y_mis: bh.min(self.mi_rows - mi_row),
			to_left: -((mi_col * 8 * 8) as i32),
			to_right: ((self.mi_cols as i32 - bw as i32 - mi_col as i32) * 8) * 8,
			to_top: -((mi_row * 8 * 8) as i32),
			to_bottom: ((self.mi_rows as i32 - bh as i32 - mi_row as i32) * 8) * 8,
			have_above: mi_row > 0,
			have_left: mi_col > self.tile.mi_col_start,
		};
		let (ss_x, ss_y) = (self.pic.ss_x, self.pic.ss_y);
		if bsize >= bs::B8X8 && SS_SIZE[usize::from(bsize)][ss_x][ss_y] == bs::INVALID {
			return Err(corrupt("vp9: invalid block size for the chroma subsampling"));
		}
		let mut mi = self.read_mode_info(r, &p, bsize, sub_w, sub_h)?;
		if mi.skip {
			for plane in 0..3 {
				let (sx, sy) = self.plane_ss(plane);
				let (x, y) = ((mi_col * 2) >> sx, ((mi_row & 7) * 2) >> sy);
				let (n4w, n4h) = ((bw * 2) >> sx, (bh * 2) >> sy);
				let a = &mut self.above_nz[plane];
				let end = (x + n4w).min(a.len());
				a[x..end].fill(0);
				let l = &mut self.left_nz[plane];
				l[y..(y + n4h).min(16)].fill(0);
			}
		}

		if !mi.is_inter() {
			for plane in 0..3 {
				self.reconstruct_plane(r, &p, &mi, plane, true)?;
			}
		} else {
			self.predict_inter(&p, &mi)?;
			if !mi.skip {
				let mut eobs = 0;
				for plane in 0..3 {
					eobs += self.reconstruct_plane(r, &p, &mi, plane, false)?;
				}
				if bsize >= bs::B8X8 && eobs == 0 {
					// No residual: the loop filter treats the block as skipped
					// (and so do the next blocks' skip contexts).
					mi.skip = true;
				}
			}
		}

		let cell = loopfilter::LfCell::new(&mi, loopfilter::level(&self.lf_table, &mi));
		for y in 0..p.y_mis {
			let at = (mi_row + y) * self.mi_cols + mi_col;
			self.mi[at..at + p.x_mis].fill(mi);
			self.lf_cells[at..at + p.x_mis].fill(cell);
		}
		Ok(())
	}

	/// Intra prediction (when `intra`) and residual of one plane, transform
	/// block by transform block. Returns the sum of the end-of-blocks.
	fn reconstruct_plane(
		&mut self,
		r: &mut BoolDecoder,
		p: &BlockPos,
		mi: &ModeInfo,
		plane: usize,
		intra: bool,
	) -> Result<usize> {
		let (sx, sy) = self.plane_ss(plane);
		let tx_size = if plane == 0 { mi.tx_size } else { uv_tx_size(mi.sb_type, mi.tx_size, sx, sy) };
		let n4w = (p.bw * 2) >> sx;
		let n4h = (p.bh * 2) >> sy;
		let n4_wl = (p.bw * 2).trailing_zeros() as usize - sx;
		let step = 1usize << tx_size;
		let max_w = if p.to_right >= 0 { n4w as i32 } else { n4w as i32 + (p.to_right >> (5 + sx)) };
		let max_h = if p.to_bottom >= 0 { n4h as i32 } else { n4h as i32 + (p.to_bottom >> (5 + sy)) };
		let clip_w = if p.to_right >= 0 { 0 } else { max_w.max(0) as usize };
		let clip_h = if p.to_bottom >= 0 { 0 } else { max_h.max(0) as usize };
		let x0 = (p.col * 8) >> sx;
		let y0 = (p.row * 8) >> sy;
		let lossless = self.h.lossless;
		let mut eobs = 0;
		let mut row = 0;
		while (row as i32) < max_h {
			let mut col = 0;
			while (col as i32) < max_w {
				let x = x0 + col * 4;
				let y = y0 + row * 4;
				let mut tx_type = DCT_DCT;
				if intra {
					let mode = if plane == 0 {
						if mi.sb_type < bs::B8X8 { mi.sub_modes[(row << 1) + col] } else { mi.mode }
					} else {
						mi.uv_mode
					};
					let pl = &mut self.pic.planes[plane];
					let edges = Edges {
						have_top: row > 0 || p.have_above,
						have_left: col > 0 || p.have_left,
						have_right: col + step < (1 << n4_wl),
						clip_right: p.to_right < 0,
						clip_bottom: p.to_bottom < 0,
						frame_width: pl.aligned_w,
						frame_height: pl.aligned_h,
					};
					let stride = pl.stride;
					intra::predict(&mut pl.data, stride, x, y, tx_size, mode, &edges, self.bit_depth);
					if plane == 0 && !lossless {
						tx_type = MODE_TO_TX_TYPE[usize::from(mode)];
					}
				}
				if !mi.skip {
					let eob = self.decode_tokens(r, p, mi, plane, col, row, tx_size, tx_type, clip_w, clip_h);
					if eob > 0 {
						let n = 4usize << tx_size;
						let pl = &mut self.pic.planes[plane];
						let stride = pl.stride;
						let coefs = &mut self.coefs[..n * n];
						transform::inverse_transform_add(
							coefs,
							&mut pl.data[y * stride + x..],
							stride,
							tx_size,
							tx_type,
							lossless,
							eob,
							self.max,
						);
						// A DC-only block wrote just its first coefficient.
						if eob == 1 {
							coefs[0] = 0;
						} else {
							coefs.fill(0);
						}
					}
					eobs += eob;
				}
				col += step;
			}
			row += step;
		}
		Ok(eobs)
	}

	#[allow(clippy::too_many_arguments)]
	fn decode_tokens(
		&mut self,
		r: &mut BoolDecoder,
		p: &BlockPos,
		mi: &ModeInfo,
		plane: usize,
		col: usize,
		row: usize,
		tx_size: u8,
		tx_type: u8,
		clip_w: usize,
		clip_h: usize,
	) -> usize {
		let (sx, sy) = self.plane_ss(plane);
		let ax = ((p.col * 2) >> sx) + col;
		let ly = (((p.row & 7) * 2) >> sy) + row;
		let n = 1usize << tx_size;
		let above_ctx =
			self.above_nz[plane][ax..(ax + n).min(self.above_nz[plane].len())].iter().any(|&v| v != 0);
		let left_ctx = self.left_nz[plane][ly..(ly + n).min(16)].iter().any(|&v| v != 0);
		let ctx = usize::from(above_ctx) + usize::from(left_ctx);
		let plane_type = usize::from(plane > 0);
		let reference = usize::from(mi.is_inter());
		let sc = residual::scan(tx_size, tx_type);
		let dq = self.dequant[usize::from(mi.segment_id)][plane_type];
		let probs = &self.fc.coef[usize::from(tx_size)][plane_type][reference];
		let counts = self.counting.then(|| TokenCounts {
			coef: &mut self.counts.coef[usize::from(tx_size)][plane_type][reference],
			eob_branch: &mut self.counts.eob_branch[usize::from(tx_size)][plane_type][reference],
		});
		let cat6 = residual::cat6(self.bit_depth);
		let eob =
			residual::decode_coefs(r, probs, counts, &mut self.coefs, tx_size, dq, ctx, &sc, cat6);
		let nz = u8::from(eob > 0);
		for k in 0..n {
			let v = if tx_size > 0 && clip_w != 0 && col + k >= clip_w { 0 } else { nz };
			if let Some(a) = self.above_nz[plane].get_mut(ax + k) {
				*a = v;
			}
			let v = if tx_size > 0 && clip_h != 0 && row + k >= clip_h { 0 } else { nz };
			if let Some(l) = self.left_nz[plane].get_mut(ly + k) {
				*l = v;
			}
		}
		eob
	}

	/// Motion-compensated prediction of a whole block, every plane
	/// (`dec_build_inter_predictors_sb`).
	fn predict_inter(&mut self, p: &BlockPos, mi: &ModeInfo) -> Result<()> {
		let refs = 1 + usize::from(mi.has_second_ref());
		for k in 0..refs {
			let idx = (mi.ref_frame[k] - 1) as usize;
			let reference =
				self.refs[idx].clone().ok_or(corrupt("vp9: block uses a missing reference"))?;
			let scale = self.ref_scale[idx];
			if !scale.valid {
				return Err(corrupt("vp9: reference frame has invalid dimensions"));
			}
			let scaled = scale.scaled();
			for plane in 0..3 {
				let (sx, sy) = self.plane_ss(plane);
				let src = &reference.planes[plane];
				let src =
					RefPlane { data: &src.data, stride: src.stride, width: src.width, height: src.height };
				let dst = &mut self.pic.planes[plane];
				let stride = dst.stride;
				let x0 = (p.col * 8) >> sx;
				let y0 = (p.row * 8) >> sy;
				let n4w = (p.bw * 2) >> sx;
				let n4h = (p.bh * 2) >> sy;
				let filter = mi.interp_filter;
				let avg = k == 1;
				let max = self.max;
				let one = |sc: &mut inter::Scratch<P>, dst: &mut [P], x: usize, y: usize, w, h, mv| {
					if scaled {
						let pos = scaled_position(p, &scale, mv, sx, sy, n4w * 4, n4h * 4, x, y);
						inter::predict_scaled(
							sc,
							&src,
							dst,
							stride,
							x0 + x,
							y0 + y,
							w,
							h,
							pos,
							filter,
							avg,
							max,
						);
					} else {
						let (mr, mc) =
							(i32::from(mv.row) * (2 - sy as i32), i32::from(mv.col) * (2 - sx as i32));
						inter::predict(sc, &src, dst, stride, x0 + x, y0 + y, w, h, mr, mc, filter, avg, max);
					}
				};
				if mi.sb_type < bs::B8X8 {
					let mut i = 0;
					for y in 0..n4h {
						for x in 0..n4w {
							let mv = average_split_mv(mi, k, i, sx, sy);
							i += 1;
							one(&mut self.scratch, &mut dst.data, 4 * x, 4 * y, 4, 4, mv);
						}
					}
				} else {
					one(&mut self.scratch, &mut dst.data, 0, 0, n4w * 4, n4h * 4, mi.mv[k]);
				}
			}
		}
		Ok(())
	}

	pub(super) fn loop_filter(&mut self) {
		let thr = loopfilter::thresholds(self.h.sharpness);
		let mut mi_row = 0;
		while mi_row < self.mi_rows {
			self.filter_sb_row(mi_row, &thr);
			mi_row += 8;
		}
	}

	/// Loop-filters one superblock row. Rows must be filtered top to
	/// bottom; row `r` reads only rows `r - 1` (already filtered) and `r`.
	fn filter_sb_row(&mut self, mi_row: usize, thr: &[loopfilter::Thresholds; 64]) {
		let mut mi_col = 0;
		while mi_col < self.mi_cols {
			for plane in 0..3 {
				let (sx, sy) = self.plane_ss(plane);
				loopfilter::filter_superblock(
					&mut self.pic.planes[plane],
					&self.lf_cells,
					self.mi_rows,
					self.mi_cols,
					mi_row,
					mi_col,
					sx,
					sy,
					thr,
					u32::from(self.bit_depth - 8),
				);
			}
			mi_col += 8;
		}
	}
}

/// Where the prediction of the `w` x `h` part at offset (`x`, `y`) of a
/// block's plane starts in a scaled reference (libvpx
/// `dec_build_inter_predictors`, `is_scaled` path). `bw` x `bh` is the
/// block's size in the plane.
///
/// The motion vector is first clamped so it points at most 4 pixels plus
/// the block size past the picture (`clamp_mv_to_umv_border_sb`). The
/// block's position is scaled as an integer, the vector in 1/16, and the
/// phase of the scaled position added to the vector; for chroma that phase
/// comes from the *luma* block position plus the chroma offset
/// (`vp9_scale_mv(&mv_q4, mi_x + x, mi_y + y, sf)`), a libvpx quirk every
/// decoder reproduces.
#[allow(clippy::too_many_arguments)]
fn scaled_position(
	p: &BlockPos,
	scale: &RefScale,
	mv: Mv,
	sx: usize,
	sy: usize,
	bw: usize,
	bh: usize,
	x: usize,
	y: usize,
) -> inter::ScaledPos {
	// C's clamp(v, lo, hi): lo wins if the bounds cross.
	let clamp = |v: i32, lo: i32, hi: i32| if v < lo { lo } else { v.min(hi) };
	let spel_left = (4 + bw as i32) << 4;
	let spel_top = (4 + bh as i32) << 4;
	let (mx, my) = (1i32 << (1 - sx), 1i32 << (1 - sy));
	let col =
		clamp(i32::from(mv.col) * mx, p.to_left * mx - spel_left, p.to_right * mx + spel_left - 16);
	let row =
		clamp(i32::from(mv.row) * my, p.to_top * my - spel_top, p.to_bottom * my + spel_top - 16);
	let x_start = ((-p.to_left) >> (3 + sx)) as i64 + x as i64;
	let y_start = ((-p.to_top) >> (3 + sy)) as i64 + y as i64;
	let (mi_x, mi_y) = ((p.col * 8 + x) as i64, (p.row * 8 + y) as i64);
	let mv_col = scale.x(i64::from(col)) + (scale.x(mi_x << 4) & 15);
	let mv_row = scale.y(i64::from(row)) + (scale.y(mi_y << 4) & 15);
	inter::ScaledPos {
		x: scale.x(x_start) + (mv_col >> 4),
		y: scale.y(y_start) + (mv_row >> 4),
		phase_x: (mv_col & 15) as i32,
		phase_y: (mv_row & 15) as i32,
		step_x: scale.step_x,
		step_y: scale.step_y,
	}
}

/// Chroma motion vector of a sub-8x8 block (`average_split_mvs`),
/// including libvpx's choice of sub-blocks for 4:2:2.
fn average_split_mv(mi: &ModeInfo, k: usize, block: usize, sx: usize, sy: usize) -> Mv {
	let mv = |b: usize| mi.sub_mv[b.min(3)][k];
	let q2 = |a: Mv, b: Mv| {
		let r = |v: i32| (if v < 0 { v - 1 } else { v + 1 }) / 2;
		Mv::new(r(i32::from(a.row) + i32::from(b.row)), r(i32::from(a.col) + i32::from(b.col)))
	};
	match (sx > 0, sy > 0) {
		(false, false) => mv(block),
		(false, true) => q2(mv(block), mv(block + 2)),
		(true, false) => q2(mv(block), mv(block + 1)),
		(true, true) => {
			let r = |v: i32| (if v < 0 { v - 2 } else { v + 2 }) / 4;
			let sum = |f: fn(Mv) -> i16| (0..4).map(|b| i32::from(f(mv(b)))).sum::<i32>();
			Mv::new(r(sum(|m| m.row)), r(sum(|m| m.col)))
		}
	}
}

/// Decoder state carried between packets.
#[derive(Debug, Clone, Default)]
pub struct Vp9DecoderState {
	persistent: Box<Persistent>,
	slots: [Option<DecodedPicture>; 8],
	color: ColorConfig,
	/// Size, type and visibility of the previous decoded frame.
	last_size: (u32, u32),
	last_show_frame: bool,
	last_key_frame: bool,
	/// libvpx's `intra_only`, which key frames leave unchanged.
	intra_only: bool,
	last_intra_only: bool,
	prev_mvs: Option<Arc<Vec<MvRef>>>,
	seg_maps: [Arc<Vec<u8>>; 2],
	frames_decoded: u64,
	/// Tile / row threads (FFmpeg `thread_type=slice`); 0 or 1 decodes on
	/// the calling thread.
	threads: usize,
	/// Per-tile-column buffers reused from frame to frame.
	pub(super) tile_pool: TilePool<u8>,
	pub(super) tile_pool_wide: TilePool<u16>,
	/// Picture planes, recycled once no slot or output frame uses them.
	pub(super) planes: BufferPool,
	pub(super) planes_wide: VecPool<u16>,
	/// Per-frame arrays reused from frame to frame.
	scratch: FrameScratch,
	/// Also pipeline decode and loop filter for single-tile-column frames.
	/// Off by default: the filter is only ~20% of the work and the row
	/// hand-over costs about as much (see `docs/THREADING.md`).
	row_pipeline: bool,
}

/// Per-frame arrays kept between frames so decoding a frame doesn't
/// allocate them. Cloning a decoder state gives the clone empty ones.
#[derive(Debug, Default)]
struct FrameScratch {
	mi: Vec<ModeInfo>,
	lf_cells: Vec<loopfilter::LfCell>,
	mvs: Vec<MvRef>,
}

impl Clone for FrameScratch {
	fn clone(&self) -> Self {
		Self::default()
	}
}

/// `v` emptied and refilled with `len` copies of `value`, keeping its
/// allocation.
fn refill<T: Clone>(mut v: Vec<T>, len: usize, value: T) -> Vec<T> {
	v.clear();
	v.resize(len, value);
	v
}

impl RefInfo for Vp9DecoderState {
	fn ref_size(&self, slot: usize) -> Option<(u32, u32)> {
		self.slots.get(slot)?.as_ref().map(|p| (p.width(), p.height()))
	}
}

impl Vp9DecoderState {
	pub fn new() -> Self {
		Self::default()
	}

	/// Like [`Self::new`], decoding tile columns on up to `threads` threads
	/// while the loop filter follows row by row. Output is identical for
	/// every thread count.
	pub fn with_threads(threads: usize) -> Self {
		Self { threads: threads.clamp(1, crate::codecs::threading::MAX_THREADS), ..Self::default() }
	}

	pub fn threads(&self) -> usize {
		self.threads.max(1)
	}

	/// Allocation counters of the picture-plane pools, 8-bit and 10/12-bit
	/// together (see [`crate::codecs::h264::H264DecoderState::pool_stats`]).
	pub fn pool_stats(&self) -> crate::core::PoolStats {
		let (a, b) = (self.planes.stats(), self.planes_wide.stats());
		crate::core::PoolStats {
			allocated: a.allocated + b.allocated,
			reused: a.reused + b.reused,
			idle: a.idle + b.idle,
		}
	}

	/// Runs the loop filter on its own thread, one superblock row behind
	/// decoding, even when a frame has a single tile column.
	pub fn with_row_pipeline(mut self, on: bool) -> Self {
		self.row_pipeline = on;
		self
	}

	/// Forgets every reference (after a seek, decoding restarts at the next
	/// key frame).
	pub fn reset(&mut self) {
		*self = Self::with_threads(self.threads).with_row_pipeline(self.row_pipeline);
	}

	/// Decodes one packet (a frame or a superframe); returns the pictures
	/// to show, in order.
	pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<DecodedPicture>> {
		let mut shown = Vec::new();
		for frame in header::split_superframe(data) {
			if let Some(pic) = self.decode_frame(frame)? {
				shown.push(pic);
			}
		}
		Ok(shown)
	}

	fn decode_frame(&mut self, data: &[u8]) -> Result<Option<DecodedPicture>> {
		let mut persistent = std::mem::take(&mut self.persistent);
		let parsed = header::parse_uncompressed(data, &mut persistent, self, self.color);
		let h = match parsed {
			Ok(h) => h,
			Err(e) => {
				self.persistent = persistent;
				return Err(e);
			}
		};
		self.persistent = persistent;
		if let Some(slot) = h.show_existing {
			let pic = self.slots[slot].clone().ok_or(corrupt("vp9: shown frame slot is empty"))?;
			return Ok(Some(pic));
		}
		if !h.key_frame && self.frames_decoded == 0 && !h.intra_only {
			return Err(corrupt("vp9: stream does not start with a key frame"));
		}
		if h.intra_only_frame() {
			self.color = h.color;
		}
		if h.color.bit_depth == 8 {
			self.decode_samples::<u8>(data, h)
		} else {
			self.decode_samples::<u16>(data, h)
		}
	}

	/// The rest of [`Self::decode_frame`], once the sample type is known.
	fn decode_samples<P: Px>(
		&mut self,
		data: &[u8],
		h: FrameHeader,
	) -> Result<Option<DecodedPicture>> {
		let (ss_x, ss_y) = (usize::from(h.color.ss_x), usize::from(h.color.ss_y));
		let mi_cols = h.width.div_ceil(8) as usize;
		let mi_rows = h.height.div_ceil(8) as usize;
		let cells = mi_rows * mi_cols;

		// Segment maps: reset with the decoding size and on independence.
		if self.last_size != (h.width, h.height) || self.seg_maps[1].len() != cells {
			self.seg_maps = [Arc::new(vec![0; cells]), Arc::new(vec![0; cells])];
		}
		if h.intra_only_frame() || h.error_resilient {
			self.seg_maps = [Arc::new(vec![0; cells]), Arc::new(vec![0; cells])];
		}
		let use_prev = !h.error_resilient
			&& (h.width, h.height) == self.last_size
			&& !self.last_intra_only
			&& self.last_show_frame
			&& !self.last_key_frame;

		let mut refs: [Option<Arc<Picture<P>>>; 3] = [None, None, None];
		let mut ref_scale = [RefScale::default(); 3];
		if !h.intra_only_frame() {
			for (i, slot) in h.ref_slot.iter().enumerate() {
				let r = self.slots[*slot].as_ref().ok_or(corrupt("vp9: missing reference frame"))?;
				let r = P::unwrap(r)
					.filter(|r| r.bit_depth == h.color.bit_depth)
					.ok_or(corrupt("vp9: reference frame has another bit depth"))?;
				if (r.ss_x, r.ss_y) != (ss_x, ss_y) {
					return Err(corrupt("vp9: reference frame has another chroma format"));
				}
				ref_scale[i] = RefScale::new(r.width, r.height, h.width, h.height);
				refs[i] = Some(r.clone());
			}
			// At least one reference must be usable; blocks that use an
			// unusable one fail (libvpx `setup_frame_size_with_refs`).
			if !ref_scale.iter().any(|s| s.valid) {
				return Err(corrupt("vp9: referenced frames have invalid sizes"));
			}
		}

		// Compressed header.
		let mut fc = self.persistent.contexts[h.context_idx].clone();
		let start = h.header_size;
		let end = start.checked_add(h.compressed_size).filter(|&e| e <= data.len());
		let end = end.ok_or(corrupt("vp9: truncated compressed header"))?;
		let ch = header::parse_compressed(&data[start..end], &h, &mut fc)?;

		let seg = self.persistent.seg.clone();
		let mut fd = FrameDecoder {
			ch,
			fc,
			mi_rows,
			mi_cols,
			// Allocated by whichever path decodes the tiles (the threaded one
			// keeps mode info in its tile workspaces only).
			mi: Vec::new(),
			lf_cells: refill(std::mem::take(&mut self.scratch.lf_cells), cells, Default::default()),
			cur_mvs: refill(std::mem::take(&mut self.scratch.mvs), cells, MvRef::default()),
			prev_mvs: if use_prev { self.prev_mvs.clone().filter(|m| m.len() == cells) } else { None },
			seg_map: (*self.seg_maps[0]).clone(),
			last_seg_map: self.seg_maps[1].clone(),
			counts: Box::default(),
			counting: !h.error_resilient && !h.parallel_mode,
			tile: TileBounds::default(),
			above_nz: std::array::from_fn(|p| {
				let shift = if p == 0 { 0 } else { ss_x };
				vec![0; ((mi_cols.next_multiple_of(8) * 2) >> shift) + 16]
			}),
			left_nz: [[0; 16]; 3],
			above_partition: vec![0; mi_cols.next_multiple_of(8) + 8],
			left_partition: [0; 8],
			pic: Picture::new(P::pools(self).0, h.width, h.height, ss_x, ss_y, h.color.bit_depth),
			refs,
			ref_scale,
			bit_depth: h.color.bit_depth,
			max: (1 << h.color.bit_depth) - 1,
			dequant: [[[0; 2]; 2]; 8],
			lf_table: [[[0; 2]; 4]; 8],
			coefs: vec![0; 32 * 32],
			scratch: inter::Scratch::default(),
			seg,
			h,
		};
		fd.setup_segment_tables(&self.persistent);
		let threads = self.threads.max(1);
		if threads > 1 && (fd.h.log2_tile_cols > 0 || (self.row_pipeline && fd.h.filter_level != 0)) {
			// Tile columns on worker threads, loop filter here, row by row.
			fd.decode_tiles_threaded(&data[end..], threads, P::pools(self).1)?;
		} else {
			fd.mi = refill(std::mem::take(&mut self.scratch.mi), cells, ModeInfo::default());
			fd.decode_tiles(&data[end..])?;
			if fd.h.filter_level != 0 {
				fd.loop_filter();
			}
		}

		// Backward adaptation, then save the context.
		let h = &fd.h;
		if !h.error_resilient && !h.parallel_mode {
			let pre = &self.persistent.contexts[h.context_idx];
			probs::adapt_coef(&mut fd.fc, pre, &fd.counts, h.intra_only_frame(), self.last_key_frame);
			if !h.intra_only_frame() {
				let switchable = h.interp_filter == super::FILTER_SWITCHABLE;
				let tx_select = fd.ch.tx_mode == header::TX_MODE_SELECT;
				probs::adapt_modes(&mut fd.fc, pre, &fd.counts, switchable, tx_select);
				probs::adapt_mv(&mut fd.fc, pre, &fd.counts, h.allow_hp);
			}
		}
		if h.refresh_context {
			self.persistent.contexts[h.context_idx] = fd.fc.clone();
		}

		let FrameDecoder { mut pic, mi, lf_cells, cur_mvs, seg_map, seg, h, .. } = fd;
		self.scratch.mi = mi;
		self.scratch.lf_cells = lf_cells;
		pic.key_frame = h.key_frame;
		pic.rgb = h.color.color_space == 7;
		let pic = P::wrap(Arc::new(pic));
		for (i, slot) in self.slots.iter_mut().enumerate() {
			if h.refresh >> i & 1 == 1 {
				*slot = Some(pic.clone());
			}
		}
		if seg.enabled {
			self.seg_maps = [self.seg_maps[1].clone(), Arc::new(seg_map)];
		}
		// The motion vectors two frames back are no longer needed: reuse
		// their array if nothing else holds it.
		if let Some(old) = self.prev_mvs.replace(Arc::new(cur_mvs)) {
			if let Ok(v) = Arc::try_unwrap(old) {
				self.scratch.mvs = v;
			}
		}
		self.last_show_frame = h.show_frame;
		self.last_size = (h.width, h.height);
		self.last_key_frame = h.key_frame;
		if !h.key_frame {
			self.intra_only = h.intra_only;
		}
		self.last_intra_only = self.intra_only;
		self.frames_decoded += 1;
		Ok(h.show_frame.then_some(pic))
	}
}

impl<P: Px> FrameDecoder<P> {
	/// Per-segment quantizers and loop filter levels
	/// (`setup_segmentation_dequant`, `vp9_loop_filter_frame_init`).
	pub(super) fn setup_segment_tables(&mut self, persistent: &Persistent) {
		let h = &self.h;
		let seg = &self.seg;
		let q = |base: i32, delta: i8, table: &[i16; 256]| {
			table[(base + i32::from(delta)).clamp(0, 255) as usize]
		};
		let scale = 1i32 << (h.filter_level >> 5);
		for s in 0..8u8 {
			let mut qindex = i32::from(h.base_q);
			if seg.active(s, SEG_ALT_Q) {
				let d = i32::from(seg.data(s, SEG_ALT_Q));
				qindex = (if seg.abs_delta { d } else { qindex + d }).clamp(0, 255);
			}
			let depth = usize::from((h.color.bit_depth - 8) / 2).min(2);
			let (dc, ac) = (&DC_QLOOKUP[depth], &AC_QLOOKUP[depth]);
			self.dequant[usize::from(s)] = [
				[q(qindex, h.y_dc_delta, dc), q(qindex, 0, ac)],
				[q(qindex, h.uv_dc_delta, dc), q(qindex, h.uv_ac_delta, ac)],
			];

			let mut lvl = i32::from(h.filter_level);
			if seg.active(s, SEG_ALT_LF) {
				let d = i32::from(seg.data(s, SEG_ALT_LF));
				lvl = (if seg.abs_delta { d } else { lvl + d }).clamp(0, 63);
			}
			let table = &mut self.lf_table[usize::from(s)];
			if !h.lf_delta_enabled {
				*table = [[lvl as u8; 2]; 4];
			} else {
				let rd = persistent.ref_deltas.map(i32::from);
				let md = persistent.mode_deltas.map(i32::from);
				table[0][0] = (lvl + rd[0] * scale).clamp(0, 63) as u8;
				for (rf, row) in table.iter_mut().enumerate().skip(1) {
					for (m, v) in row.iter_mut().enumerate() {
						*v = (lvl + rd[rf] * scale + md[m] * scale).clamp(0, 63) as u8;
					}
				}
			}
		}
	}
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Vp9Decoder;

impl Vp9Decoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for Vp9Decoder {
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
			private: CodecPrivate::Vp9(Box::new(Vp9DecoderState::with_threads(
				crate::codecs::threading::default_threads(),
			))),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Video(video) = state else {
			return Err(Message::Codec("vp9 decoder requires a video state"));
		};
		if !matches!(video.private, CodecPrivate::Vp9(_)) {
			video.private = CodecPrivate::Vp9(Box::new(Vp9DecoderState::with_threads(
				crate::codecs::threading::default_threads(),
			)));
		}
		let CodecPrivate::Vp9(dec) = &mut video.private else {
			return Err(Message::Codec("vp9 decoder state missing"));
		};
		let pictures = dec.decode_packet(&packet.data)?;
		let mut frames = Vec::with_capacity(pictures.len());
		for pic in pictures {
			let pixel = pic.pixel().ok_or(Message::Codec(if pic.rgb() {
				"vp9: 10/12-bit sRGB output is not supported"
			} else {
				"vp9: 4:4:0 output is not supported"
			}))?;
			video.width = pic.width();
			video.height = pic.height();
			video.pixel = pixel.clone();
			video.has_keyframe |= pic.key_frame();
			frames.push(Frame::Video(pic.to_frame(pixel, packet.pts)));
		}
		Ok(DecodeOut { value: frames })
	}

	fn flush(&self, _state: &mut State) -> Result<DecodeOut> {
		Ok(DecodeOut { value: Vec::new() })
	}
}
