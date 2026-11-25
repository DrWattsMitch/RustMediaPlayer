//! AV1 encoder: Main profile, 8-bit 4:2:0, key and inter frames.
//!
//! * [`msac`] — the multi-symbol range encoder and a bit-cost model.
//! * [`cdf`], [`tables`] — default CDFs, scans, quantizers, filters
//!   (generated from rav1d by `tools/gen_av1_tables.py`).
//! * [`coef`] — block contexts and coefficient coding.
//! * [`itx`], [`itx_1d`] — inverse transforms (rav1d's kernels) and the
//!   forward transforms that invert them.
//! * [`ipred`], [`mc`] — intra and inter prediction.
//! * [`refmvs`] — motion-vector candidates.
//! * [`frame`] — rate-distortion search, reconstruction, tile writing.
//!
//! Every decoding step the encoder depends on — symbol adaptation, the
//! contexts of each syntax element, the candidate list, prediction and
//! inverse transforms — follows rav1d's code, so the encoder's
//! reconstruction is what rav1d decodes (the tests decode every frame with
//! rav1d and compare).
//!
//! The stream: a sequence header (64x64 superblocks; no CDEF, loop
//! restoration, superres, film grain, intra edge filter, filter intra,
//! inter-intra, masked compound, warped motion, dual filter or order hints)
//! and one frame OBU per temporal unit. A key frame every GOP (`-g`,
//! default 128); inter frames predict from the previous frame. Each frame
//! starts from the default CDFs (`primary_ref_frame` none) and adapts them
//! symbol by symbol; the deblocking filter is off.
//!
//! Coding tools:
//!
//! | | |
//! |---|---|
//! | blocks | 64x64 to 8x8 squares (PARTITION_NONE / SPLIT, by RD) |
//! | transform | the block's size (`TX_MODE_LARGEST`), reduced set: DCT, ADST and identity combinations for small intra blocks |
//! | intra | the 13 luma modes (directional ones without angle delta), same for chroma |
//! | inter | NEARESTMV, NEARMV (all candidates), GLOBALMV, NEWMV (quarter-pel, best predictor), regular 8-tap |
//!
//! libaom / FFmpeg counterparts: `av1/encoder/bitstream.c` (headers,
//! OBUs), `encodeframe.c`, `rdopt.c`, `aom_dsp/entenc.c`; FFmpeg
//! `libaomenc.c` (`av1C`, packets).

pub mod cdf;
mod coef;
mod frame;
mod ipred;
mod itx;
mod itx_1d;
mod mc;
pub mod msac;
mod refmvs;
pub mod tables;

use std::sync::Arc;

pub use frame::{Picture, Search};

use crate::codecs::bitstream::BitWriter;
use crate::core::{
	self, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters, Pixel, PixelFormat, State,
	StreamId, Time, TimeBase, VideoFrame, VideoState,
};
use crate::message::{Message, Result};
use frame::{FrameEncoder, Plane};
use msac::Encoder as RangeEncoder;
use refmvs::TileRange;

/// libaom's `quantizer_to_qindex` (the 0-63 scale of `--min-q` / `-crf`).
pub const QUANTIZER_TO_QINDEX: [u8; 64] = crate::codecs::vp9::encoder::QUANTIZER_TO_QINDEX;
pub const DEFAULT_QUANTIZER: u8 = 30;
pub const DEFAULT_GOP: u32 = 128;

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME: u8 = 6;

#[derive(Debug, Clone, Copy)]
pub struct Settings {
	/// Quantizer on libaom's 0-63 scale (ignored with a bit rate).
	pub quantizer: u8,
	/// Average bit rate in bits per second.
	pub bitrate: Option<u32>,
	pub gop: u32,
	pub frame_rate: f64,
	pub search: Search,
}

impl Default for Settings {
	fn default() -> Self {
		Self {
			quantizer: DEFAULT_QUANTIZER,
			bitrate: None,
			gop: DEFAULT_GOP,
			frame_rate: 25.0,
			search: Search::default(),
		}
	}
}

/// One coded temporal unit.
#[derive(Debug, Clone)]
pub struct CodedFrame {
	/// OBUs (sequence header on key frames, then the frame), without a
	/// temporal delimiter (as MP4 and Matroska store AV1).
	pub data: Vec<u8>,
	pub key: bool,
	pub pts: Option<Time>,
	pub qindex: u8,
}

fn leb128(mut v: usize, out: &mut Vec<u8>) {
	loop {
		let b = (v & 0x7f) as u8;
		v >>= 7;
		if v == 0 {
			out.push(b);
			return;
		}
		out.push(b | 0x80);
	}
}

/// An OBU with its size field.
fn obu(kind: u8, payload: &[u8]) -> Vec<u8> {
	let mut out = vec![(kind << 3) | 0x02];
	leb128(payload.len(), &mut out);
	out.extend_from_slice(payload);
	out
}

/// A temporal delimiter OBU (for raw streams and IVF).
pub fn temporal_delimiter() -> Vec<u8> {
	obu(OBU_TEMPORAL_DELIMITER, &[])
}

/// `tile_log2`.
fn tile_log2(blk: usize, target: usize) -> u32 {
	let mut k = 0;
	while (blk << k) < target {
		k += 1;
	}
	k
}

/// Tile layout of a frame (uniform spacing, the fewest tiles allowed).
#[derive(Debug, Clone)]
struct Tiles {
	cols_log2: u32,
	rows_log2: u32,
	min_cols_log2: u32,
	max_cols_log2: u32,
	min_rows_log2: u32,
	max_rows_log2: u32,
	/// Starts in 4x4 units, with the end appended.
	col_starts: Vec<usize>,
	row_starts: Vec<usize>,
}

impl Tiles {
	fn new(mi_cols: usize, mi_rows: usize) -> Self {
		let (sb_cols, sb_rows) = (mi_cols.div_ceil(16), mi_rows.div_ceil(16));
		let min_cols_log2 = tile_log2(64, sb_cols);
		let max_cols_log2 = tile_log2(1, sb_cols.min(64));
		let max_rows_log2 = tile_log2(1, sb_rows.min(64));
		let min_log2_tiles = min_cols_log2.max(tile_log2(2304, sb_rows * sb_cols));
		let cols_log2 = min_cols_log2;
		let starts = |sbs: usize, log2: u32, mis: usize| {
			let size = (sbs + (1 << log2) - 1) >> log2;
			let mut v: Vec<usize> = (0..sbs).step_by(size.max(1)).map(|s| s << 4).collect();
			v.push(mis);
			v
		};
		let col_starts = starts(sb_cols, cols_log2, mi_cols);
		let min_rows_log2 = min_log2_tiles.saturating_sub(cols_log2);
		let rows_log2 = min_rows_log2;
		let row_starts = starts(sb_rows, rows_log2, mi_rows);
		Self {
			cols_log2,
			rows_log2,
			min_cols_log2,
			max_cols_log2,
			min_rows_log2,
			max_rows_log2,
			col_starts,
			row_starts,
		}
	}

	fn count(&self) -> usize {
		(self.col_starts.len() - 1) * (self.row_starts.len() - 1)
	}

	fn write(&self, b: &mut BitWriter) {
		b.bit(true); // uniform spacing
		if self.cols_log2 < self.max_cols_log2 {
			b.bit(false);
		}
		let _ = self.min_cols_log2;
		if self.rows_log2 < self.max_rows_log2 {
			b.bit(false);
		}
		let _ = self.min_rows_log2;
		if self.cols_log2 > 0 || self.rows_log2 > 0 {
			b.put(0, self.cols_log2 + self.rows_log2); // context_update_tile_id
			b.put(3, 2); // 4-byte tile sizes
		}
	}
}

/// Bit-rate control (as the VP9 encoder's): `bits ≈ complexity / ac_q`.
#[derive(Debug, Clone, Default)]
struct RateControl {
	complexity: [f64; 2],
	known: [bool; 2],
	error: f64,
}

impl RateControl {
	fn qindex(&self, key: bool, bitrate: u32, fps: f64) -> u8 {
		let k = usize::from(!key);
		let per_frame = f64::from(bitrate) / fps.max(1.0);
		let target =
			(per_frame * if key { 4.0 } else { 0.92 } - self.error / fps.max(1.0)).max(per_frame * 0.1);
		if !self.known[k] {
			return if key { 100 } else { 120 };
		}
		let c = self.complexity[k];
		(1..=255u8)
			.min_by(|&a, &b| {
				let ea = (c / f64::from(tables::DQ_8BIT[usize::from(a)][1]) - target).abs();
				let eb = (c / f64::from(tables::DQ_8BIT[usize::from(b)][1]) - target).abs();
				ea.total_cmp(&eb)
			})
			.unwrap_or(120)
	}

	fn update(&mut self, key: bool, qindex: u8, bits: f64, bitrate: u32, fps: f64) {
		let k = usize::from(!key);
		let c = bits * f64::from(tables::DQ_8BIT[usize::from(qindex)][1]);
		self.complexity[k] = if self.known[k] { 0.6 * self.complexity[k] + 0.4 * c } else { c };
		self.known[k] = true;
		self.error += bits - f64::from(bitrate) / fps.max(1.0);
	}
}

/// Encoder state carried from frame to frame.
#[derive(Debug, Clone)]
pub struct Av1EncoderState {
	pub width: u32,
	pub height: u32,
	pub settings: Settings,
	last: Option<Arc<Picture>>,
	frames: u64,
	since_key: u64,
	rc: RateControl,
	/// The last frame's reconstruction (what a decoder shows).
	pub recon: Option<Arc<Picture>>,
}

/// `seq_level_idx` for a picture size (by luma samples, Annex A).
fn level(width: u32, height: u32) -> u32 {
	let samples = u64::from(width) * u64::from(height);
	let levels: [(u64, u32); 6] =
		[(147_456, 0), (278_784, 1), (665_856, 4), (1_065_024, 5), (2_359_296, 8), (8_912_896, 12)];
	levels
		.iter()
		.find(|&&(max, _)| samples <= max)
		.map_or(if samples <= 35_651_584 { 16 } else { 31 }, |&(_, l)| l)
}

impl Av1EncoderState {
	pub fn new(width: u32, height: u32, settings: Settings) -> Result<Self> {
		if width == 0 || height == 0 || width > 65536 || height > 65536 {
			return Err(Message::Codec("av1 encoder: invalid picture size"));
		}
		Ok(Self {
			width,
			height,
			settings,
			last: None,
			frames: 0,
			since_key: 0,
			rc: RateControl::default(),
			recon: None,
		})
	}

	/// The sequence header OBU.
	pub fn sequence_header(&self) -> Vec<u8> {
		let (w, h) = (self.width, self.height);
		let bits = |v: u32| (32 - (v - 1).leading_zeros()).max(1);
		let (wb, hb) = (bits(w), bits(h));
		let lvl = level(w, h);
		let mut b = BitWriter::new();
		b.put(0, 3); // profile
		b.bit(false); // still_picture
		b.bit(false); // reduced_still_picture_header
		b.bit(false); // timing_info_present
		b.bit(false); // initial_display_delay_present
		b.put(0, 5); // one operating point
		b.put(0, 12); // operating_point_idc
		b.put(lvl, 5);
		if lvl > 7 {
			b.bit(false); // tier
		}
		b.put(wb - 1, 4);
		b.put(hb - 1, 4);
		b.put(w - 1, wb);
		b.put(h - 1, hb);
		b.bit(false); // frame ids
		b.bit(false); // 128x128 superblocks
		for _ in 0..7 {
			// filter intra, intra edge filter, inter-intra, masked compound,
			// warped motion, dual filter, order hints
			b.bit(false);
		}
		b.bit(false); // seq_choose_screen_content_tools
		b.bit(false); // seq_force_screen_content_tools = 0
		b.bit(false); // superres
		b.bit(false); // CDEF
		b.bit(false); // loop restoration
		b.bit(false); // high bit depth
		b.bit(false); // monochrome
		b.bit(false); // colour description
		b.bit(false); // limited range
		b.put(0, 2); // chroma sample position unknown
		b.bit(false); // separate_uv_delta_q
		b.bit(false); // film grain
		b.bit(true); // trailing bits
		b.align_zero();
		obu(OBU_SEQUENCE_HEADER, &b.finish())
	}

	/// The `av1C` record (ISOBMFF / Matroska `CodecPrivate`).
	pub fn av1c(&self) -> Vec<u8> {
		let lvl = level(self.width, self.height) as u8;
		let mut out = vec![0x81, lvl, 0b0000_1100, 0];
		out.extend(self.sequence_header());
		out
	}

	/// The last reconstruction as packed I420 (empty before the first frame).
	pub fn recon_bytes(&self) -> Vec<u8> {
		self.recon.as_ref().map(|p| p.bytes()).unwrap_or_default()
	}

	/// Codes a frame (any strides, 4:2:0).
	pub fn encode_frame(&mut self, frame: &VideoFrame) -> Result<CodedFrame> {
		let plane = |i: usize| frame.plane(i).map_or((&[][..], 0), |p| (p.bytes(), p.stride()));
		self.encode_planes([plane(0), plane(1), plane(2)], frame.pts)
	}

	/// Codes a packed I420 picture.
	pub fn encode_picture(&mut self, data: &[u8], pts: Option<Time>) -> Result<CodedFrame> {
		let (w, h) = (self.width as usize, self.height as usize);
		let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
		if data.len() < w * h + 2 * cw * ch {
			return Err(Message::Codec("av1 encoder: frame data shorter than its yuv420p size"));
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
		let mut src: [Plane; 3] = std::array::from_fn(|i| {
			let s = usize::from(i > 0);
			Plane::new((w + s) >> s, (h + s) >> s, s)
		});
		for (i, &(data, stride)) in planes.iter().enumerate() {
			let p = &mut src[i];
			let (pw, ph) = (p.width, p.height);
			if stride < pw || data.len() < stride * (ph - 1) + pw {
				return Err(Message::Codec("av1 encoder: plane smaller than the picture"));
			}
			// Copy, repeating the edges into the padding.
			let rows = p.data.len() / p.stride;
			for y in 0..rows {
				let row = &data[y.min(ph - 1) * stride..];
				let out = &mut p.data[y * p.stride..(y + 1) * p.stride];
				out[..pw].copy_from_slice(&row[..pw]);
				let last = row[pw - 1];
				out[pw..].fill(last);
			}
		}
		let key = self.frames == 0 || self.since_key >= u64::from(self.settings.gop.max(1));
		let qidx = match self.settings.bitrate {
			Some(b) if b > 0 => self.rc.qindex(key, b, self.settings.frame_rate),
			_ => {
				let q = QUANTIZER_TO_QINDEX[usize::from(self.settings.quantizer.min(63))];
				if key { q.saturating_sub(q / 5).max(1) } else { q.max(1) }
			}
		};
		let mut data = Vec::new();
		if key {
			data.extend(self.sequence_header());
		}
		data.extend(self.code(src, key, qidx));
		if let Some(b) = self.settings.bitrate.filter(|&b| b > 0) {
			self.rc.update(key, qidx, data.len() as f64 * 8.0, b, self.settings.frame_rate);
		}
		self.since_key = if key { 1 } else { self.since_key + 1 };
		self.frames += 1;
		Ok(CodedFrame { data, key, pts, qindex: qidx })
	}

	/// The frame header (`uncompressed_header`) up to the tile info, and the
	/// rest after it.
	fn frame_header(&self, key: bool, qidx: u8, tiles: &Tiles) -> BitWriter {
		let mut b = BitWriter::new();
		b.bit(false); // show_existing_frame
		b.put(if key { 0 } else { 1 }, 2);
		b.bit(true); // show_frame
		if !key {
			b.bit(false); // error_resilient_mode
		}
		b.bit(false); // disable_cdf_update
		b.bit(false); // frame_size_override_flag
		if !key {
			b.put(7, 3); // primary_ref_frame: none
			b.put(0x01, 8); // refresh LAST's slot
			for _ in 0..7 {
				b.put(0, 3); // every reference is slot 0
			}
		}
		b.bit(false); // render size = frame size
		if !key {
			b.bit(false); // allow_high_precision_mv
			b.bit(false); // interpolation filter not switchable
			b.put(0, 2); // EIGHTTAP
			b.bit(false); // is_motion_mode_switchable
		}
		b.bit(true); // disable_frame_end_update_cdf
		tiles.write(&mut b);
		b.put(u32::from(qidx), 8);
		b.put(0, 3); // no DC / chroma deltas
		b.bit(false); // no quantizer matrices
		b.bit(false); // segmentation
		b.bit(false); // delta_q_present (qidx > 0)
		b.put(0, 12); // loop filter levels 0, 0
		b.put(0, 3); // sharpness
		b.bit(false); // loop_filter_delta_enabled
		b.bit(false); // TX_MODE_LARGEST
		if !key {
			b.bit(false); // single reference
		}
		b.bit(true); // reduced_tx_set
		if !key {
			for _ in 0..7 {
				b.bit(false); // no global motion
			}
		}
		b
	}

	fn code(&mut self, src: [Plane; 3], key: bool, qidx: u8) -> Vec<u8> {
		let dq = tables::DQ_8BIT[usize::from(qidx)];
		let dq = [i32::from(dq[0]), i32::from(dq[1])];
		let dc = f64::from(dq[0]);
		let lambda = 88.0 * dc * dc / (24.0 * 2048.0);
		let sad_per_bit = 0.0418 * f64::from(dq[1]) / 4.0 + 2.4;
		let refp = if key { None } else { self.last.clone() };
		let mut fe =
			FrameEncoder::new(key, src, refp, qidx, dq, lambda, sad_per_bit, self.settings.search);
		let tiles = Tiles::new(fe.bw, fe.bh);
		let mut tile_data = Vec::with_capacity(tiles.count());
		for tr in 0..tiles.row_starts.len() - 1 {
			for tc in 0..tiles.col_starts.len() - 1 {
				let t = TileRange {
					col_start: tiles.col_starts[tc] as i32,
					col_end: tiles.col_starts[tc + 1] as i32,
					row_start: tiles.row_starts[tr] as i32,
					row_end: tiles.row_starts[tr + 1] as i32,
				};
				fe.start_tile(t);
				let mut e = RangeEncoder::new();
				for y4 in (tiles.row_starts[tr]..tiles.row_starts[tr + 1]).step_by(16) {
					fe.start_sb_row();
					for x4 in (tiles.col_starts[tc]..tiles.col_starts[tc + 1]).step_by(16) {
						fe.code_sb(&mut e, x4, y4);
					}
				}
				tile_data.push(e.finish());
			}
		}
		let mut b = self.frame_header(key, qidx, &tiles);
		b.align_zero();
		let mut payload = b.finish();
		if tile_data.len() > 1 {
			payload.push(0); // tile_start_and_end_present_flag = 0, aligned
		}
		let n = tile_data.len();
		for (i, t) in tile_data.into_iter().enumerate() {
			if i + 1 < n {
				payload.extend_from_slice(&((t.len() - 1) as u32).to_le_bytes());
			}
			payload.extend(t);
		}
		let pic = Arc::new(Picture { planes: fe.rec });
		self.last = Some(pic.clone());
		self.recon = Some(pic);
		obu(OBU_FRAME, &payload)
	}
}

/// AV1 encoder (`-c:v av1` / `libaom-av1`).
#[derive(Debug, Clone, Copy)]
pub struct Av1Encoder {
	stream_id: StreamId,
	settings: Settings,
}

impl Av1Encoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self { stream_id, settings: Settings::default() }
	}

	/// `-crf` (quantizer 0-63) takes precedence over `-b:v`.
	pub fn with_rate(mut self, quantizer: Option<u8>, bitrate: Option<u32>) -> Self {
		match (quantizer, bitrate) {
			(Some(q), _) => {
				self.settings.quantizer = q.min(63);
				self.settings.bitrate = None;
			}
			(None, Some(b)) if b > 0 => self.settings.bitrate = Some(b),
			_ => {}
		}
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

	/// `av1C` extradata for `parameters`.
	pub fn extradata(&self, parameters: &Parameters) -> Result<Vec<u8>> {
		let (w, h) = (parameters.width.unwrap_or(0), parameters.height.unwrap_or(0));
		Ok(Av1EncoderState::new(w, h, self.settings)?.av1c())
	}

	fn state<'a>(&self, state: &'a mut State) -> Result<&'a mut Av1EncoderState> {
		let State::Video(video) = state else {
			return Err(Message::Codec("av1 encoder requires a video state"));
		};
		match &mut video.private {
			CodecPrivate::Av1Encoder(enc) => Ok(enc),
			_ => Err(Message::Codec("av1 encoder: invalid picture size")),
		}
	}
}

impl core::Encoder for Av1Encoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let (w, h) = (parameters.width.unwrap_or(0), parameters.height.unwrap_or(0));
		let enc = Av1EncoderState::new(w, h, self.settings).ok();
		State::Video(VideoState {
			width: w,
			height: h,
			pixel: Pixel { depth: 8, format: PixelFormat::YUV420 },
			reorder: Vec::new(),
			has_keyframe: false,
			codec_state: CodecState {
				extradata: enc.as_ref().map(Av1EncoderState::av1c).unwrap_or_default(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			private: enc.map_or(CodecPrivate::None, |e| CodecPrivate::Av1Encoder(Box::new(e))),
		})
	}

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut> {
		let Frame::Video(frame) = frame else {
			return Err(Message::Codec("av1 encoder requires video frames"));
		};
		let enc = self.state(state)?;
		if frame.pixel.depth != 8 || frame.pixel.format != PixelFormat::YUV420 {
			return Err(Message::Codec(
				"av1 encoder needs yuv420p frames (convert with -pix_fmt yuv420p)",
			));
		}
		if frame.width != enc.width || frame.height != enc.height {
			return Err(Message::Codec("av1 encoder: frame size does not match the stream"));
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
