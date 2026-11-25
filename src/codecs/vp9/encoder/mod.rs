//! VP9 encoder: profile 0 (8-bit 4:2:0), key and inter frames.
//!
//! * [`writer`] — boolean encoder, bit-cost model, probability updates.
//! * [`fdct`] — forward transforms, the exact inverse of the decoder's.
//! * [`pack`] — partitions, mode info, motion vectors and tokens, written
//!   (or priced) the way the decoder parses them.
//! * [`frame`] — rate-distortion search and reconstruction of one frame.
//!
//! The encoder runs on the decoder's own frame state and kernels (intra
//! and inter prediction, inverse transforms, loop filter, motion-vector
//! candidates, probability adaptation), so its reconstruction is what a
//! decoder reconstructs, bit for bit (the tests check every frame against
//! RMP's decoder).
//!
//! Stream structure: a key frame every GOP (`-g`, default 128), inter
//! frames predicted from the previous frame (LAST; GOLDEN and ALTREF hold
//! the key frame). Every frame is shown; no alt-ref, no superframes.
//!
//! Coding tools:
//!
//! | | |
//! |---|---|
//! | blocks | 64x64 to 8x8 squares (PARTITION_NONE / SPLIT, by RD) |
//! | transform | largest for the block (`ALLOW_32X32`); DCT/ADST by intra mode |
//! | intra | all 10 modes, luma and chroma |
//! | inter | NEARESTMV, NEARMV, ZEROMV, NEWMV (quarter-pel), regular 8-tap |
//! | entropy | forward coefficient probability updates per frame, backward adaptation |
//! | loop filter | level from the quantiser (libvpx `LPF_PICK_FROM_Q`) |
//!
//! Rate control: a constant quantiser (`-qp` as libvpx's 0-63 scale; key
//! frames a little finer) or a target bit rate (`-b:v`) with a per-frame
//! complexity model.
//!
//! libvpx counterparts: `vp9_encoder.c`, `vp9_encodeframe.c`,
//! `vp9_rdopt.c`, `vp9_bitstream.c`, `vp9_tokenize.c`, `vp9_subexp.c`,
//! `vp9_mcomp.c`, `vp9_picklpf.c`; FFmpeg `libvpxenc.c` for packets.

mod fdct;
mod frame;
mod pack;
mod writer;

use std::sync::Arc;

pub use frame::Search;

use super::decoder::{FrameDecoder, Picture, RefScale, TileBounds};
use super::header::{
	self, ColorConfig, CompressedHeader, FrameHeader, Persistent, SINGLE_REFERENCE, Segmentation,
};
use super::inter;
use super::loopfilter;
use super::mode::{ModeInfo, MvRef};
use super::probs::{self, Counts, FrameContext};
use super::tables::{AC_QLOOKUP, DC_QLOOKUP};
use crate::codecs::bitstream::BitWriter;
use crate::core::{
	self, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters, Pixel, PixelFormat, State,
	StreamId, Time, TimeBase, VecPool, VideoFrame, VideoState,
};
use crate::message::{Message, Result};
use frame::{FrameEncoder, Source};
use writer::{BoolEncoder, Null};

/// libvpx's `quantizer_to_qindex`: the 0-63 scale of `--min-q` / `-qp`.
pub const QUANTIZER_TO_QINDEX: [u8; 64] = [
	0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96,
	100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172,
	176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 249,
	255,
];

/// Default quantizer (0-63).
pub const DEFAULT_QUANTIZER: u8 = 30;
pub const DEFAULT_GOP: u32 = 128;
/// `TX_MODE_SELECT`: each block chooses its transform size.
const TX_MODE: u8 = header::TX_MODE_SELECT;

/// Rate control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateMode {
	/// Constant quantizer on libvpx's 0-63 scale.
	Quantizer(u8),
	/// Average bit rate in bits per second.
	Bitrate(u32),
}

#[derive(Debug, Clone, Copy)]
pub struct Settings {
	pub mode: RateMode,
	pub gop: u32,
	pub frame_rate: f64,
	pub search: Search,
}

impl Default for Settings {
	fn default() -> Self {
		Self {
			mode: RateMode::Quantizer(DEFAULT_QUANTIZER),
			gop: DEFAULT_GOP,
			frame_rate: 25.0,
			search: Search::default(),
		}
	}
}

/// One coded frame.
#[derive(Debug, Clone)]
pub struct CodedFrame {
	pub data: Vec<u8>,
	pub key: bool,
	pub pts: Option<Time>,
	pub qindex: u8,
}

/// Bit-rate control: `bits ≈ complexity / ac_quant(q)` per frame type,
/// steered towards the target over about a second.
#[derive(Debug, Clone)]
struct RateControl {
	/// Complexity estimates: key, inter.
	complexity: [f64; 2],
	known: [bool; 2],
	/// Bits produced minus bits budgeted so far.
	error: f64,
}

impl RateControl {
	fn new() -> Self {
		Self { complexity: [0.0; 2], known: [false; 2], error: 0.0 }
	}

	fn qindex(&self, key: bool, bitrate: u32, fps: f64) -> u8 {
		let k = usize::from(!key);
		let per_frame = f64::from(bitrate) / fps.max(1.0);
		let mut target = per_frame * if key { 4.0 } else { 0.92 };
		// Pay back the error over one second.
		target = (target - self.error / fps.max(1.0)).max(per_frame * 0.1);
		if !self.known[k] {
			// First frame of a type: a middle quantiser scaled by bits per pixel.
			return if key { 100 } else { 120 };
		}
		let c = self.complexity[k];
		(1..=255u8)
			.min_by(|&a, &b| {
				let fa = (c / f64::from(AC_QLOOKUP[0][usize::from(a)]) - target).abs();
				let fb = (c / f64::from(AC_QLOOKUP[0][usize::from(b)]) - target).abs();
				fa.total_cmp(&fb)
			})
			.unwrap_or(120)
	}

	fn update(&mut self, key: bool, qindex: u8, bits: f64, bitrate: u32, fps: f64) {
		let k = usize::from(!key);
		let c = bits * f64::from(AC_QLOOKUP[0][usize::from(qindex)]);
		self.complexity[k] = if self.known[k] { 0.6 * self.complexity[k] + 0.4 * c } else { c };
		self.known[k] = true;
		self.error += bits - f64::from(bitrate) / fps.max(1.0);
	}
}

/// Encoder state carried from frame to frame.
#[derive(Debug, Clone)]
pub struct Vp9EncoderState {
	pub width: u32,
	pub height: u32,
	pub settings: Settings,
	persistent: Persistent,
	slots: [Option<Arc<Picture<u8>>>; 8],
	pool: VecPool<u8>,
	prev_mvs: Option<Arc<Vec<MvRef>>>,
	last_show_frame: bool,
	last_key_frame: bool,
	last_size: (u32, u32),
	frames: u64,
	since_key: u64,
	rc: RateControl,
	/// The last frame's reconstruction (what a decoder shows).
	pub recon: Option<Arc<Picture<u8>>>,
}

fn tile_offset(i: usize, mis: usize, log2: u32) -> usize {
	let sb = mis.div_ceil(8);
	(((i * sb) >> log2) << 3).min(mis)
}

/// Smallest and largest tile-column log2 for a width (spec §6.2.12).
fn tile_col_bounds(width: u32) -> (u32, u32) {
	let sb_cols = header::sb64_cols(width.div_ceil(8));
	let mut min_log2 = 0;
	while (64 << min_log2) < sb_cols {
		min_log2 += 1;
	}
	let mut max_log2 = 1;
	while (sb_cols >> max_log2) >= 4 {
		max_log2 += 1;
	}
	(min_log2, max_log2 - 1)
}

impl Vp9EncoderState {
	pub fn new(width: u32, height: u32, settings: Settings) -> Result<Self> {
		if width == 0 || height == 0 || width > 16384 || height > 16384 {
			return Err(Message::Codec("vp9 encoder: invalid picture size"));
		}
		if u64::from(width) * u64::from(height) > header::MAX_AREA {
			return Err(Message::Codec("vp9 encoder: picture too large"));
		}
		Ok(Self {
			width,
			height,
			settings,
			persistent: Persistent::default(),
			slots: Default::default(),
			pool: VecPool::new(4),
			prev_mvs: None,
			last_show_frame: false,
			last_key_frame: false,
			last_size: (0, 0),
			frames: 0,
			since_key: 0,
			rc: RateControl::new(),
			recon: None,
		})
	}

	/// The last frame's reconstruction as packed I420 (empty before the
	/// first frame).
	pub fn recon_bytes(&self) -> Vec<u8> {
		self.recon.as_ref().map(|p| p.bytes()).unwrap_or_default()
	}

	/// Codes a frame (any strides, 4:2:0, chroma `(w + 1) / 2` wide).
	pub fn encode_frame(&mut self, frame: &VideoFrame) -> Result<CodedFrame> {
		let plane = |i: usize| frame.plane(i).map_or((&[][..], 0), |p| (p.bytes(), p.stride()));
		self.encode_planes([plane(0), plane(1), plane(2)], frame.pts)
	}

	/// Codes a packed I420 picture.
	pub fn encode_picture(&mut self, data: &[u8], pts: Option<Time>) -> Result<CodedFrame> {
		let (w, h) = (self.width as usize, self.height as usize);
		let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
		if data.len() < w * h + 2 * cw * ch {
			return Err(Message::Codec("vp9 encoder: frame data shorter than its yuv420p size"));
		}
		let (y, rest) = data.split_at(w * h);
		let (u, v) = rest.split_at(cw * ch);
		self.encode_planes([(y, w), (u, cw), (v, cw)], pts)
	}

	fn encode_planes(
		&mut self,
		planes: [(&[u8], usize); 3],
		pts: Option<Time>,
	) -> Result<CodedFrame> {
		let (w, h) = (self.width as usize, self.height as usize);
		for (i, &(data, stride)) in planes.iter().enumerate() {
			let s = usize::from(i > 0);
			let (pw, ph) = ((w + s) >> s, (h + s) >> s);
			if stride < pw || data.len() < stride * (ph - 1) + pw {
				return Err(Message::Codec("vp9 encoder: plane smaller than the picture"));
			}
		}
		let src = Source::new(planes, w, h);
		let key = self.frames == 0 || self.since_key >= u64::from(self.settings.gop.max(1));
		let qindex = match self.settings.mode {
			RateMode::Quantizer(q) => {
				let q = QUANTIZER_TO_QINDEX[usize::from(q.min(63))];
				if key { q.saturating_sub(q / 5).max(1) } else { q.max(1) }
			}
			RateMode::Bitrate(b) => self.rc.qindex(key, b, self.settings.frame_rate),
		};
		let data = self.code(src, key, qindex);
		if let RateMode::Bitrate(b) = self.settings.mode {
			self.rc.update(key, qindex, data.len() as f64 * 8.0, b, self.settings.frame_rate);
		}
		self.since_key = if key { 1 } else { self.since_key + 1 };
		self.frames += 1;
		Ok(CodedFrame { data, key, pts, qindex })
	}

	/// The uncompressed header of a frame (spec §6.2).
	fn write_uncompressed(&self, h: &FrameHeader, compressed_size: usize) -> Vec<u8> {
		let mut b = BitWriter::new();
		b.put(2, 2); // frame marker
		b.put(0, 2); // profile 0
		b.bit(false); // show_existing_frame
		b.bit(!h.key_frame);
		b.bit(h.show_frame);
		b.bit(h.error_resilient);
		let size = |b: &mut BitWriter| {
			b.put(h.width - 1, 16);
			b.put(h.height - 1, 16);
			b.bit(false); // render size = frame size
		};
		if h.key_frame {
			b.put(0x49_8342, 24);
			b.put(u32::from(h.color.color_space), 3);
			b.bit(h.color.full_range);
			size(&mut b);
		} else {
			b.put(u32::from(h.reset_context), 2);
			b.put(u32::from(h.refresh), 8);
			for i in 0..3 {
				b.put(h.ref_slot[i] as u32, 3);
				b.bit(h.sign_bias[i + 1]);
			}
			b.bit(true); // size from LAST
			b.bit(false); // render size = frame size
			b.bit(h.allow_hp);
			b.bit(false); // not switchable
			// Literal order: smooth, regular, sharp, bilinear.
			b.put([1u32, 0, 2, 3][usize::from(h.interp_filter)], 2);
		}
		b.bit(h.refresh_context);
		b.bit(h.parallel_mode);
		b.put(h.context_idx as u32, 2);
		b.put(u32::from(h.filter_level), 6);
		b.put(u32::from(h.sharpness), 3);
		b.bit(false); // no loop filter deltas
		b.put(u32::from(h.base_q), 8);
		b.put(0, 3); // no quantizer deltas
		b.bit(false); // no segmentation
		let (min_log2, max_log2) = tile_col_bounds(h.width);
		let mut l = min_log2;
		while l < max_log2 {
			let more = l < h.log2_tile_cols;
			b.bit(more);
			if !more {
				break;
			}
			l += 1;
		}
		b.bit(false); // one tile row
		b.put(compressed_size as u32, 16);
		b.align_zero();
		b.finish()
	}

	/// The compressed header: transform mode and probability updates from
	/// `old` to `new` (only coefficient probabilities change).
	fn write_compressed(h: &FrameHeader, old: &FrameContext, new: &FrameContext) -> Vec<u8> {
		use writer::{Bits, write_diff_update};
		let mut w = BoolEncoder::new();
		w.literal(u32::from(TX_MODE.min(3)), 2);
		if TX_MODE >= 3 {
			w.put(TX_MODE == header::TX_MODE_SELECT, 128);
		}
		if TX_MODE == header::TX_MODE_SELECT {
			// tx8, tx16, tx32 probabilities: no update.
			for _ in 0..2 * (1 + 2 + 3) {
				w.put(false, writer::UPDATE_PROB);
			}
		}
		for tx in 0..=header::tx_mode_max(TX_MODE) {
			let changed = old.coef[tx] != new.coef[tx];
			w.put(changed, 128);
			if !changed {
				continue;
			}
			for plane in 0..2 {
				for r in 0..2 {
					for band in 0..6 {
						for ctx in 0..if band == 0 { 3 } else { 6 } {
							for node in 0..3 {
								let (o, n) = (
									old.coef[tx][plane][r][band][ctx][node],
									new.coef[tx][plane][r][band][ctx][node],
								);
								write_diff_update(&mut w, o, n);
							}
						}
					}
				}
			}
		}
		let none = |w: &mut BoolEncoder, n: usize| {
			for _ in 0..n {
				w.put(false, writer::UPDATE_PROB);
			}
		};
		none(&mut w, 3); // skip
		if !h.intra_only_frame() {
			none(&mut w, 7 * 3); // inter modes
			none(&mut w, 4); // intra / inter
			none(&mut w, 5 * 2); // single reference
			none(&mut w, 4 * 9); // luma modes
			none(&mut w, 16 * 3); // partitions
			// Motion vectors: joints, then per component sign, classes,
			// class0, bits; then class0_fp and fp; then hp.
			none(&mut w, 3 + 2 * (1 + 10 + 1 + 10) + 2 * (2 * 3 + 3));
			if h.allow_hp {
				none(&mut w, 4);
			}
		}
		w.finish()
	}

	/// Chooses forward coefficient probability updates from the frame's
	/// counts (`vp9_update_coef_probs`, one pass).
	fn coef_updates(fc: &FrameContext, counts: &Counts) -> FrameContext {
		let mut new = fc.clone();
		for tx in 0..=header::tx_mode_max(TX_MODE) {
			let mut saved = 0i64;
			let mut cand = new.coef[tx];
			for plane in 0..2 {
				for r in 0..2 {
					for band in 0..6 {
						for ctx in 0..if band == 0 { 3 } else { 6 } {
							let c = &counts.coef[tx][plane][r][band][ctx];
							let eob = counts.eob_branch[tx][plane][r][band][ctx];
							let branch = [[c[3], eob.saturating_sub(c[3])], [c[0], c[1] + c[2]], [c[1], c[2]]];
							for (node, ct) in branch.into_iter().enumerate() {
								let old = fc.coef[tx][plane][r][band][ctx][node];
								if let Some(p) = writer::best_update(ct, old) {
									let before = writer::branch_cost(ct, old) as i64;
									let after = writer::branch_cost(ct, p) as i64;
									saved += before - after;
									cand[plane][r][band][ctx][node] = p;
								}
							}
						}
					}
				}
			}
			// The per-size flag costs a bit; the update codes were already
			// weighed per probability.
			if saved > 256 {
				new.coef[tx] = cand;
			}
		}
		new
	}

	/// Codes one frame; returns its bytes.
	fn code(&mut self, src: Source, key: bool, qindex: u8) -> Vec<u8> {
		let (width, height) = (self.width, self.height);
		let mi_cols = width.div_ceil(8) as usize;
		let mi_rows = height.div_ceil(8) as usize;
		let cells = mi_rows * mi_cols;
		let ac = i32::from(AC_QLOOKUP[0][usize::from(qindex)]);
		let filter_level = {
			let guess = (ac * 20723 + 1_015_158 + (1 << 17)) >> 18;
			(if key { guess - 4 } else { guess }).clamp(0, 63) as u8
		};
		let mut h = FrameHeader {
			profile: 0,
			key_frame: key,
			show_frame: true,
			color: ColorConfig {
				bit_depth: 8,
				color_space: 1,
				full_range: false,
				ss_x: true,
				ss_y: true,
			},
			refresh: if key { 0xFF } else { 0x01 },
			ref_slot: [0, 1, 2],
			width,
			height,
			render_size: (width, height),
			interp_filter: 0,
			refresh_context: true,
			filter_level,
			base_q: qindex,
			log2_tile_cols: tile_col_bounds(width).0,
			..FrameHeader::default()
		};
		// What the decoder's header parsing does to its persistent state.
		if key {
			self.persistent.seg = Segmentation::default();
			self.persistent.ref_deltas = [1, 0, -1, -1];
			self.persistent.mode_deltas = [0, 0];
			self.persistent.contexts = std::array::from_fn(|_| FrameContext::default());
			h.context_idx = 0;
		}
		let use_prev =
			(width, height) == self.last_size && self.last_show_frame && !self.last_key_frame;
		let refs = if key { [None, None, None] } else { h.ref_slot.map(|s| self.slots[s].clone()) };
		let ch =
			CompressedHeader { tx_mode: TX_MODE, reference_mode: SINGLE_REFERENCE, ..Default::default() };
		let fc = self.persistent.contexts[h.context_idx].clone();
		let mut fd = FrameDecoder {
			ch,
			fc,
			seg: self.persistent.seg.clone(),
			mi_rows,
			mi_cols,
			mi: vec![ModeInfo::default(); cells],
			lf_cells: vec![Default::default(); cells],
			cur_mvs: vec![MvRef::default(); cells],
			prev_mvs: if use_prev { self.prev_mvs.clone().filter(|m| m.len() == cells) } else { None },
			seg_map: vec![0; cells],
			last_seg_map: Arc::new(vec![0; cells]),
			counts: Box::default(),
			counting: false,
			tile: TileBounds::default(),
			above_nz: std::array::from_fn(|p| {
				let shift = usize::from(p > 0);
				vec![0; ((mi_cols.next_multiple_of(8) * 2) >> shift) + 16]
			}),
			left_nz: [[0; 16]; 3],
			above_partition: vec![0; mi_cols.next_multiple_of(8) + 8],
			left_partition: [0; 8],
			pic: Picture::new(&self.pool, width, height, 1, 1, 8),
			refs,
			ref_scale: [RefScale::default(); 3],
			bit_depth: 8,
			max: 255,
			dequant: [[[0; 2]; 2]; 8],
			lf_table: [[[0; 2]; 4]; 8],
			coefs: vec![0; 32 * 32],
			scratch: inter::Scratch::default(),
			h,
		};
		fd.setup_segment_tables(&self.persistent);
		let dc = f64::from(DC_QLOOKUP[0][usize::from(qindex)]);
		let lambda = 88.0 * dc * dc / (24.0 * 2048.0);
		let sad_per_bit = 0.0418 * f64::from(ac) / 4.0 + 2.4;
		let mut enc = FrameEncoder::new(fd, src, lambda, sad_per_bit, self.settings.search);

		// Search and reconstruct, in the decoder's order.
		let log2 = enc.fd.h.log2_tile_cols;
		let tile_cols = 1usize << log2;
		let tile = |c: usize| TileBounds {
			mi_row_start: 0,
			mi_row_end: mi_rows,
			mi_col_start: tile_offset(c, mi_cols, log2),
			mi_col_end: tile_offset(c + 1, mi_cols, log2),
		};
		for mi_row in (0..mi_rows).step_by(8) {
			for c in 0..tile_cols {
				enc.fd.tile = tile(c);
				enc.analyse_sb_row(mi_row);
			}
		}
		{
			let fd = &mut enc.fd;
			for (cell, mi) in fd.lf_cells.iter_mut().zip(&fd.mi) {
				*cell = loopfilter::LfCell::new(mi, loopfilter::level(&fd.lf_table, mi));
			}
			if fd.h.filter_level != 0 {
				fd.loop_filter();
			}
		}

		// Pack: count, choose updates, write.
		let reset = |enc: &mut FrameEncoder| {
			let fd = &mut enc.fd;
			fd.mi.fill(ModeInfo::default());
			fd.cur_mvs.fill(MvRef::default());
			for a in fd.above_nz.iter_mut() {
				a.fill(0);
			}
			fd.above_partition.fill(0);
			enc.cursor = (0, 0);
		};
		reset(&mut enc);
		enc.fd.counting = true;
		for mi_row in (0..mi_rows).step_by(8) {
			for c in 0..tile_cols {
				enc.fd.tile = tile(c);
				enc.pack_sb_row(&mut Null, mi_row);
			}
		}
		let counts = std::mem::take(&mut enc.fd.counts);

		enc.fd.counting = false;
		let old = enc.fd.fc.clone();
		let new = Self::coef_updates(&old, &counts);
		let compressed = Self::write_compressed(&enc.fd.h, &old, &new);
		enc.fd.fc = new;
		reset(&mut enc);
		let mut writers: Vec<BoolEncoder> = (0..tile_cols).map(|_| BoolEncoder::new()).collect();
		for mi_row in (0..mi_rows).step_by(8) {
			for (c, w) in writers.iter_mut().enumerate() {
				enc.fd.tile = tile(c);
				enc.pack_sb_row(w, mi_row);
			}
		}
		let mut out = self.write_uncompressed(&enc.fd.h, compressed.len());
		out.extend_from_slice(&compressed);
		let n = writers.len();
		for (i, w) in writers.into_iter().enumerate() {
			let data = w.finish();
			if i + 1 < n {
				out.extend_from_slice(&(data.len() as u32).to_be_bytes());
			}
			out.extend_from_slice(&data);
		}

		// Backward adaptation and the saved context, as the decoder does.
		let FrameEncoder { fd, .. } = enc;
		let mut fd = fd;
		let h = &fd.h;
		if !h.error_resilient && !h.parallel_mode {
			let pre = &self.persistent.contexts[h.context_idx];
			probs::adapt_coef(&mut fd.fc, pre, &counts, h.intra_only_frame(), self.last_key_frame);
			if !h.intra_only_frame() {
				probs::adapt_modes(&mut fd.fc, pre, &counts, false, TX_MODE == header::TX_MODE_SELECT);
				probs::adapt_mv(&mut fd.fc, pre, &counts, h.allow_hp);
			}
		}
		if h.refresh_context {
			self.persistent.contexts[h.context_idx] = fd.fc.clone();
		}
		let FrameDecoder { mut pic, cur_mvs, h, .. } = fd;
		pic.key_frame = h.key_frame;
		let pic = Arc::new(pic);
		for (i, slot) in self.slots.iter_mut().enumerate() {
			if h.refresh >> i & 1 == 1 {
				*slot = Some(pic.clone());
			}
		}
		self.recon = Some(pic);
		self.prev_mvs = Some(Arc::new(cur_mvs));
		self.last_show_frame = h.show_frame;
		self.last_key_frame = h.key_frame;
		self.last_size = (width, height);
		out
	}
}

/// VP9 encoder (`-c:v vp9` / `libvpx-vp9`).
#[derive(Debug, Clone, Copy)]
pub struct Vp9Encoder {
	stream_id: StreamId,
	settings: Settings,
}

impl Vp9Encoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self { stream_id, settings: Settings::default() }
	}

	/// `-qp` (quantizer 0-63) takes precedence over `-b:v`.
	pub fn with_rate(mut self, quantizer: Option<u8>, bitrate: Option<u32>) -> Self {
		self.settings.mode = match (quantizer, bitrate) {
			(Some(q), _) => RateMode::Quantizer(q.min(63)),
			(None, Some(b)) if b > 0 => RateMode::Bitrate(b),
			_ => RateMode::Quantizer(DEFAULT_QUANTIZER),
		};
		self
	}

	pub fn with_gop(mut self, gop: Option<u32>) -> Self {
		self.settings.gop = gop.filter(|&g| g > 0).unwrap_or(DEFAULT_GOP);
		self
	}

	pub fn with_time_base(mut self, tb: TimeBase) -> Self {
		if tb.num > 0 && tb.den > 0 {
			self.settings.frame_rate = f64::from(tb.den) / f64::from(tb.num);
		}
		self
	}

	fn state<'a>(&self, state: &'a mut State) -> Result<&'a mut Vp9EncoderState> {
		let State::Video(video) = state else {
			return Err(Message::Codec("vp9 encoder requires a video state"));
		};
		match &mut video.private {
			CodecPrivate::Vp9Encoder(enc) => Ok(enc),
			_ => Err(Message::Codec("vp9 encoder: invalid picture size (up to 16384x16384)")),
		}
	}
}

impl core::Encoder for Vp9Encoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let (w, h) = (parameters.width.unwrap_or(0), parameters.height.unwrap_or(0));
		let enc = Vp9EncoderState::new(w, h, self.settings).ok();
		State::Video(VideoState {
			width: w,
			height: h,
			pixel: Pixel { depth: 8, format: PixelFormat::YUV420 },
			reorder: Vec::new(),
			has_keyframe: false,
			codec_state: CodecState { extradata: Vec::new(), pts: None, dts: None, last_pts: None },
			private: enc.map_or(CodecPrivate::None, |e| CodecPrivate::Vp9Encoder(Box::new(e))),
		})
	}

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut> {
		let Frame::Video(frame) = frame else {
			return Err(Message::Codec("vp9 encoder requires video frames"));
		};
		let enc = self.state(state)?;
		if frame.pixel.depth != 8 || frame.pixel.format != PixelFormat::YUV420 {
			return Err(Message::Codec(
				"vp9 encoder needs yuv420p frames (convert with -pix_fmt yuv420p)",
			));
		}
		if frame.width != enc.width || frame.height != enc.height {
			return Err(Message::Codec("vp9 encoder: frame size does not match the stream"));
		}
		let coded = enc.encode_frame(&frame)?;
		let mut packet = Packet::new(self.stream_id, coded.data);
		packet.pts = coded.pts;
		packet.dts = coded.pts;
		packet.keyframe = coded.key;
		if let State::Video(v) = state {
			v.has_keyframe |= coded.key;
		}
		Ok(EncodeOut { value: vec![packet] })
	}

	fn flush(&self, _state: &mut State) -> Result<EncodeOut> {
		Ok(EncodeOut { value: Vec::new() })
	}
}

#[cfg(test)]
mod tests;
