//! H.264 / AVC encoder (ISO/IEC 14496-10), Constrained Baseline profile.
//!
//! * [`transform`] — forward 4x4 transform, Hadamard DC transforms,
//!   deadzone quantisation, SATD, x264-style decimation.
//! * [`vlc`] — CAVLC residual writer (the decoder's `cavlc` inverted).
//! * [`me`] — padded reference, SAD/SATD block costs.
//! * [`macroblock`] — mode decision and coding of every macroblock.
//!
//! What it produces: progressive 8-bit 4:2:0, CAVLC, one slice per
//! picture, IDR + P pictures with one reference, Intra16x16 / Intra4x4 /
//! P_Skip / P_L0 16x16, 16x8, 8x16 and 8x8 macroblocks, quarter-sample
//! motion, the in-loop deblocking filter. That is the Constrained Baseline
//! subset every H.264 decoder supports (phones, browsers, hardware).
//!
//! Pictures are reconstructed with the decoder's own `recon` and
//! `deblock` code, so the encoder's references are exactly what a decoder
//! reconstructs and there is no drift (the tests check this bit for bit
//! against RMP's decoder, and the docs against FFmpeg's).
//!
//! Rate control: constant QP (`-qp`, P pictures; IDR pictures 3 lower, as
//! x264's default `ipratio`), or a target bit rate (`-b:v`) with a
//! per-picture-type complexity model (`bits ≈ K / Qstep`) and a one-second
//! correction window. Without either, QP 23 is used.
//!
//! Output: `avcC` extradata (SPS + PPS) and one access unit per packet of
//! 4-byte length-prefixed NAL units, as MP4 and Matroska store them.
//! Timestamps are passed through; there is no reordering (pts = dts).
//!
//! FFmpeg / x264 counterparts: `x264/encoder/encoder.c` (picture loop),
//! `set.c` (SPS/PPS), `macroblock.c`, `analyse.c`, `me.c`, `ratecontrol.c`,
//! `cavlc.c`; FFmpeg `libavcodec/libx264.c` (packetisation, `avcC`).

mod macroblock;
mod me;
mod transform;
mod vlc;

use std::sync::{Arc, OnceLock};

use super::deblock::deblock_frame;
use super::decode::{Frame as Picture, SliceFilter};
use super::params::{Dequant, ScalingLists};
use super::recon::Plane;
use crate::codecs::bitstream::BitWriter;
use crate::core::{
	self, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters, Pixel, PixelFormat, State,
	StreamId, TimeBase, VideoState,
};
use crate::message::{Message, Result};
use macroblock::{PicEncoder, Reference, Source};

/// QP for P pictures when neither `-qp` nor `-b:v` is given.
pub const DEFAULT_QP: u8 = 23;
/// IDR interval when `-g` isn't given (x264's default `keyint`).
pub const DEFAULT_GOP: u32 = 250;
/// IDR pictures use a QP this much lower than P pictures.
const IP_OFFSET: i32 = 3;
/// `log2_max_frame_num`.
const LOG2_MAX_FRAME_NUM: u32 = 8;
/// Largest picture: same cap as the decoder.
const MAX_WIDTH: u32 = 8192;
const MAX_HEIGHT: u32 = 4352;

fn dequant() -> &'static Dequant {
	static D: OnceLock<Dequant> = OnceLock::new();
	D.get_or_init(|| Dequant::new(&ScalingLists::flat()))
}

/// Table A-1 limits: (level_idc, MaxMBPS, MaxFS, MaxBR in kb/s for
/// Baseline).
const LEVELS: [(u8, u64, u64, u64); 15] = [
	(10, 1485, 99, 64),
	(11, 3000, 396, 192),
	(12, 6000, 396, 384),
	(13, 11880, 396, 768),
	(20, 11880, 396, 2000),
	(21, 19800, 792, 4000),
	(22, 20250, 1620, 4000),
	(30, 40500, 1620, 10000),
	(31, 108000, 3600, 14000),
	(32, 216000, 5120, 20000),
	(40, 245760, 8192, 20000),
	(41, 245760, 8192, 50000),
	(42, 522240, 8704, 50000),
	(50, 589824, 22080, 135000),
	(51, 983040, 36864, 240000),
];

/// Smallest level that fits the picture size, macroblock rate and bit
/// rate (`bitrate` 0 = unknown).
fn choose_level(mb_w: u32, mb_h: u32, fps: f64, bitrate: u32) -> u8 {
	let fs = u64::from(mb_w * mb_h);
	let mbps = (fs as f64 * fps).ceil() as u64;
	for &(idc, max_mbps, max_fs, max_br) in &LEVELS {
		let side = ((8 * max_fs) as f64).sqrt() as u64;
		if fs <= max_fs
			&& u64::from(mb_w) <= side
			&& u64::from(mb_h) <= side
			&& mbps <= max_mbps
			&& u64::from(bitrate) <= max_br * 1000
		{
			return idc;
		}
	}
	52
}

/// Vertical motion vector limit in luma samples (Table A-1), minus the
/// quarter sample the range excludes.
fn max_mv_y(level: u8) -> i32 {
	match level {
		0..=10 => 63,
		11..=20 => 127,
		21..=30 => 255,
		_ => 511,
	}
}

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateMode {
	/// Constant QP for P pictures.
	Qp(u8),
	/// Average bit rate in bits per second.
	Bitrate(u32),
}

#[derive(Debug, Clone)]
struct RateControl {
	mode: RateMode,
	fps: f64,
	bits_per_frame: f64,
	spent: f64,
	budget: f64,
	/// Complexity `K = bits · Qstep` per picture type (0 = IDR, 1 = P).
	complexity: [Option<f64>; 2],
	last_qp: [Option<f64>; 2],
	/// QP the first picture starts from (bits-per-pixel heuristic).
	initial_qp: f64,
}

fn qstep(qp: f64) -> f64 {
	0.625 * 2f64.powf(qp / 6.0)
}

impl RateControl {
	fn new(mode: RateMode, fps: f64, pixels: u32) -> Self {
		let (bits_per_frame, initial_qp) = match mode {
			RateMode::Bitrate(b) => {
				let bpf = f64::from(b) / fps;
				let bpp = bpf / f64::from(pixels.max(1));
				(bpf, (26.0 - 6.0 * (bpp / 0.1).log2()).clamp(12.0, 45.0))
			}
			RateMode::Qp(q) => (0.0, f64::from(q)),
		};
		Self {
			mode,
			fps,
			bits_per_frame,
			spent: 0.0,
			budget: 0.0,
			complexity: [None, None],
			last_qp: [None, None],
			initial_qp,
		}
	}

	fn qp(&self, idr: bool) -> i32 {
		let qp = match self.mode {
			RateMode::Qp(q) => f64::from(q) - if idr { f64::from(IP_OFFSET) } else { 0.0 },
			RateMode::Bitrate(_) => {
				// Spread over- or under-spending over about a second.
				let error = self.spent - self.budget;
				let target = (self.bits_per_frame - error / self.fps.max(1.0))
					.clamp(0.25 * self.bits_per_frame, 2.0 * self.bits_per_frame);
				let p_qp = match (self.complexity[1], self.last_qp[1]) {
					(Some(k), last) => {
						let q = 6.0 * (k / (target * 0.625)).log2();
						match last {
							Some(l) => q.clamp(l - 4.0, l + 4.0),
							None => q,
						}
					}
					(None, _) => self.last_qp[0].map_or(self.initial_qp, |q| q + f64::from(IP_OFFSET)),
				};
				if idr { p_qp - f64::from(IP_OFFSET) } else { p_qp }
			}
		};
		qp.round().clamp(if matches!(self.mode, RateMode::Qp(_)) { 0.0 } else { 10.0 }, 51.0) as i32
	}

	fn update(&mut self, idr: bool, qp: i32, bits: usize) {
		let t = usize::from(!idr);
		let k = bits as f64 * qstep(f64::from(qp));
		self.complexity[t] = Some(match self.complexity[t] {
			Some(old) => 0.5 * old + 0.5 * k,
			None => k,
		});
		self.last_qp[t] = Some(f64::from(qp));
		self.spent += bits as f64;
		self.budget += self.bits_per_frame;
	}
}

/// Encoder state: parameter sets, the reference picture, GOP position and
/// rate control.
#[derive(Debug, Clone)]
pub struct H264EncoderState {
	pub width: u32,
	pub height: u32,
	mb_w: usize,
	mb_h: usize,
	gop: u32,
	level: u8,
	sps: Vec<u8>,
	pps: Vec<u8>,
	frames: u64,
	since_idr: u32,
	frame_num: u32,
	idr_id: u32,
	reference: Option<Arc<Reference>>,
	rc: RateControl,
	/// QP of each coded picture (for tests and statistics).
	pub last_qp: i32,
}

impl H264EncoderState {
	pub fn new(width: u32, height: u32, fps: f64, mode: RateMode, gop: u32) -> Result<Self> {
		if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
			return Err(Message::Codec("h264 encoder: width and height must be even and non-zero"));
		}
		if width > MAX_WIDTH || height > MAX_HEIGHT {
			return Err(Message::Codec("h264 encoder: picture larger than 8192x4352"));
		}
		let fps = if fps.is_finite() && fps > 0.0 { fps } else { 25.0 };
		let (mb_w, mb_h) = (width.div_ceil(16), height.div_ceil(16));
		let bitrate = match mode {
			RateMode::Bitrate(b) => b,
			RateMode::Qp(_) => 0,
		};
		let level = choose_level(mb_w, mb_h, fps, bitrate);
		let mut s = Self {
			width,
			height,
			mb_w: mb_w as usize,
			mb_h: mb_h as usize,
			gop: gop.max(1),
			level,
			sps: Vec::new(),
			pps: Vec::new(),
			frames: 0,
			since_idr: 0,
			frame_num: 0,
			idr_id: 0,
			reference: None,
			rc: RateControl::new(mode, fps, width * height),
			last_qp: 0,
		};
		s.sps = nal(7, 3, &s.write_sps(fps));
		s.pps = nal(8, 3, &write_pps());
		Ok(s)
	}

	/// `seq_parameter_set_rbsp()` (7.3.2.1.1): Constrained Baseline, POC
	/// type 2, one reference, cropping to the real size, and VUI timing plus
	/// `bitstream_restriction` (no reordering, so decoders output at once).
	fn write_sps(&self, fps: f64) -> Vec<u8> {
		let mut w = BitWriter::new();
		w.put(66, 8); // profile_idc: Baseline
		w.put(0b1100_0000, 8); // constraint_set0 + set1: Constrained Baseline
		w.put(u32::from(self.level), 8);
		w.ue(0); // seq_parameter_set_id
		w.ue(LOG2_MAX_FRAME_NUM - 4);
		w.ue(2); // pic_order_cnt_type
		w.ue(1); // max_num_ref_frames
		w.bit(false); // gaps_in_frame_num_value_allowed_flag
		w.ue(self.mb_w as u32 - 1);
		w.ue(self.mb_h as u32 - 1);
		w.bit(true); // frame_mbs_only_flag
		w.bit(true); // direct_8x8_inference_flag
		let (crop_r, crop_b) =
			((self.mb_w as u32 * 16 - self.width) / 2, (self.mb_h as u32 * 16 - self.height) / 2);
		let crop = crop_r != 0 || crop_b != 0;
		w.bit(crop);
		if crop {
			w.ue(0);
			w.ue(crop_r);
			w.ue(0);
			w.ue(crop_b);
		}
		w.bit(true); // vui_parameters_present_flag
		w.bit(false); // aspect_ratio_info_present_flag
		w.bit(false); // overscan_info_present_flag
		w.bit(false); // video_signal_type_present_flag
		w.bit(false); // chroma_loc_info_present_flag
		// Timing: time_scale / (2 · num_units_in_tick) = fps.
		let (num, scale) = frame_rate_fraction(fps);
		w.bit(true);
		w.put(num, 32);
		w.put(scale.saturating_mul(2), 32);
		w.bit(false); // fixed_frame_rate_flag
		w.bit(false); // nal_hrd_parameters_present_flag
		w.bit(false); // vcl_hrd_parameters_present_flag
		w.bit(false); // pic_struct_present_flag
		w.bit(true); // bitstream_restriction_flag
		w.bit(true); // motion_vectors_over_pic_boundaries_flag
		w.ue(0); // max_bytes_per_pic_denom
		w.ue(0); // max_bits_per_mb_denom
		w.ue(15); // log2_max_mv_length_horizontal
		w.ue(15); // log2_max_mv_length_vertical
		w.ue(0); // max_num_reorder_frames
		w.ue(1); // max_dec_frame_buffering
		w.rbsp_trailing();
		w.finish()
	}

	/// `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §5.3.3.1).
	pub fn avcc(&self) -> Vec<u8> {
		let sps = &self.sps;
		let pps = &self.pps;
		let mut out = vec![1, sps[1], sps[2], sps[3], 0xFF, 0xE1];
		out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
		out.extend_from_slice(sps);
		out.push(1);
		out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
		out.extend_from_slice(pps);
		out
	}

	/// Pads a planar 4:2:0 picture to whole macroblocks by edge
	/// replication.
	fn source(&self, data: &[u8]) -> Result<Source> {
		let (w, h) = (self.width as usize, self.height as usize);
		let (cw, ch) = (w / 2, h / 2);
		if data.len() < w * h + 2 * cw * ch {
			return Err(Message::Codec("h264 encoder: frame data shorter than its yuv420p size"));
		}
		let pad = |plane: &[u8], pw: usize, ph: usize, ow: usize, oh: usize| -> Plane {
			let mut p = Plane::new(ow, oh, 0);
			for y in 0..oh {
				let sy = y.min(ph - 1);
				let src = &plane[sy * pw..(sy + 1) * pw];
				let row = &mut p.data[y * ow..(y + 1) * ow];
				row[..pw].copy_from_slice(src);
				row[pw..].fill(src[pw - 1]);
			}
			p
		};
		let (yw, yh) = (self.mb_w * 16, self.mb_h * 16);
		let u0 = w * h;
		let v0 = u0 + cw * ch;
		Ok(Source {
			y: pad(&data[..u0], w, h, yw, yh),
			cb: pad(&data[u0..v0], cw, ch, yw / 2, yh / 2),
			cr: pad(&data[v0..v0 + cw * ch], cw, ch, yw / 2, yh / 2),
		})
	}

	/// Codes one picture; returns the access unit (length-prefixed NAL
	/// units) and whether it is an IDR picture.
	pub fn encode_picture(&mut self, data: &[u8]) -> Result<(Vec<u8>, bool)> {
		let src = self.source(data)?;
		let idr = self.reference.is_none() || self.since_idr >= self.gop;
		if idr {
			self.since_idr = 0;
			self.frame_num = 0;
		}
		let qp = self.rc.qp(idr);
		let mut pic = Picture::new(self.mb_w, self.mb_h);
		pic.slices.push(SliceFilter { disable: 0, alpha: 0, beta: 0 });

		let mut w = BitWriter::with_capacity(self.mb_w * self.mb_h * 32);
		self.write_slice_header(&mut w, idr, qp);
		let reference = if idr { None } else { self.reference.as_deref() };
		let max_mv = max_mv_y(self.level);
		let mut enc = PicEncoder::new(&src, &mut pic, reference, dequant(), qp, max_mv, w);
		enc.encode_slice();
		let mut w = std::mem::take(&mut enc.w);
		drop(enc);
		w.rbsp_trailing();
		let rbsp = w.finish();

		deblock_frame(&mut pic);
		let padded = me::Padded::new(&pic.y);
		let half = me::HalfPel::new(&padded);
		self.reference = Some(Arc::new(Reference { y: pic.y, cb: pic.cb, cr: pic.cr, padded, half }));

		let slice = nal(if idr { 5 } else { 1 }, 3, &rbsp);
		let mut out = Vec::with_capacity(slice.len() + 4);
		out.extend_from_slice(&(slice.len() as u32).to_be_bytes());
		out.extend_from_slice(&slice);
		self.rc.update(idr, qp, out.len() * 8);
		self.last_qp = qp;
		if idr {
			self.idr_id = (self.idr_id + 1) % 65536;
		}
		self.frame_num = (self.frame_num + 1) % (1 << LOG2_MAX_FRAME_NUM);
		self.since_idr += 1;
		self.frames += 1;
		Ok((out, idr))
	}

	/// `slice_header()` (7.3.3) for a single-slice I (IDR) or P picture.
	fn write_slice_header(&self, w: &mut BitWriter, idr: bool, qp: i32) {
		w.ue(0); // first_mb_in_slice
		w.ue(if idr { 7 } else { 5 }); // slice_type: I / P (all slices)
		w.ue(0); // pic_parameter_set_id
		w.put(self.frame_num, LOG2_MAX_FRAME_NUM);
		if idr {
			w.ue(self.idr_id);
		}
		if !idr {
			w.bit(false); // num_ref_idx_active_override_flag
			w.bit(false); // ref_pic_list_modification_flag_l0
		}
		// dec_ref_pic_marking()
		if idr {
			w.bit(false); // no_output_of_prior_pics_flag
			w.bit(false); // long_term_reference_flag
		} else {
			w.bit(false); // adaptive_ref_pic_marking_mode_flag
		}
		w.se(qp - 26); // slice_qp_delta
		w.ue(0); // disable_deblocking_filter_idc
		w.se(0); // slice_alpha_c0_offset_div2
		w.se(0); // slice_beta_offset_div2
	}
}

/// `pic_parameter_set_rbsp()` (7.3.2.2): CAVLC, one reference, deblocking
/// control present, QP 26 base.
fn write_pps() -> Vec<u8> {
	let mut w = BitWriter::new();
	w.ue(0); // pic_parameter_set_id
	w.ue(0); // seq_parameter_set_id
	w.bit(false); // entropy_coding_mode_flag: CAVLC
	w.bit(false); // bottom_field_pic_order_in_frame_present_flag
	w.ue(0); // num_slice_groups_minus1
	w.ue(0); // num_ref_idx_l0_default_active_minus1
	w.ue(0); // num_ref_idx_l1_default_active_minus1
	w.bit(false); // weighted_pred_flag
	w.put(0, 2); // weighted_bipred_idc
	w.se(0); // pic_init_qp_minus26
	w.se(0); // pic_init_qs_minus26
	w.se(0); // chroma_qp_index_offset
	w.bit(true); // deblocking_filter_control_present_flag
	w.bit(false); // constrained_intra_pred_flag
	w.bit(false); // redundant_pic_cnt_present_flag
	w.rbsp_trailing();
	w.finish()
}

/// `num_units_in_tick` and `time_scale / 2` for a frame rate, exact for
/// the usual NTSC rates.
fn frame_rate_fraction(fps: f64) -> (u32, u32) {
	for (num, den) in [(1001u32, 24_000u32), (1001, 30_000), (1001, 60_000)] {
		if (fps - f64::from(den) / f64::from(num)).abs() < 0.001 {
			return (num, den);
		}
	}
	let scale = (fps * 1000.0).round().clamp(1.0, 2.0e9) as u32;
	(1000, scale)
}

/// Wraps an RBSP in a NAL unit header with emulation prevention (7.4.1).
fn nal(kind: u8, ref_idc: u8, rbsp: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 2);
	out.push((ref_idc << 5) | kind);
	let mut zeros = 0;
	for &b in rbsp {
		if zeros >= 2 && b <= 3 {
			out.push(3);
			zeros = 0;
		}
		out.push(b);
		zeros = if b == 0 { zeros + 1 } else { 0 };
	}
	out
}

/// H.264 encoder (`-c:v h264`).
#[derive(Debug, Clone, Copy)]
pub struct H264Encoder {
	stream_id: StreamId,
	mode: RateMode,
	gop: u32,
	frame_rate: f64,
}

impl H264Encoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self { stream_id, mode: RateMode::Qp(DEFAULT_QP), gop: DEFAULT_GOP, frame_rate: 25.0 }
	}

	/// `-qp` (constant QP) takes precedence over `-b:v`.
	pub fn with_rate(mut self, qp: Option<u8>, bitrate: Option<u32>) -> Self {
		self.mode = match (qp, bitrate) {
			(Some(q), _) => RateMode::Qp(q.min(51)),
			(None, Some(b)) if b > 0 => RateMode::Bitrate(b),
			_ => RateMode::Qp(DEFAULT_QP),
		};
		self
	}

	pub fn with_gop(mut self, gop: Option<u32>) -> Self {
		self.gop = gop.filter(|&g| g > 0).unwrap_or(DEFAULT_GOP);
		self
	}

	/// Frame rate from the stream's per-frame time base (e.g. 1/30).
	pub fn with_time_base(mut self, tb: TimeBase) -> Self {
		if tb.num > 0 && tb.den > 0 {
			self.frame_rate = f64::from(tb.den) / f64::from(tb.num);
		}
		self
	}

	fn new_state(&self, parameters: &Parameters) -> Result<H264EncoderState> {
		let (w, h) = (parameters.width.unwrap_or(0), parameters.height.unwrap_or(0));
		H264EncoderState::new(w, h, self.frame_rate, self.mode, self.gop)
	}

	/// `avcC` extradata for `parameters`.
	pub fn extradata(&self, parameters: &Parameters) -> Result<Vec<u8>> {
		Ok(self.new_state(parameters)?.avcc())
	}

	fn state<'a>(&self, state: &'a mut State) -> Result<&'a mut H264EncoderState> {
		let State::Video(video) = state else {
			return Err(Message::Codec("h264 encoder requires a video state"));
		};
		match &mut video.private {
			CodecPrivate::H264Encoder(enc) => Ok(enc),
			_ => Err(Message::Codec(
				"h264 encoder: invalid picture size (even width and height, up to 8192x4352)",
			)),
		}
	}
}

impl core::Encoder for H264Encoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let enc = self.new_state(parameters).ok();
		State::Video(VideoState {
			width: parameters.width.unwrap_or(0),
			height: parameters.height.unwrap_or(0),
			pixel: Pixel { depth: 8, format: PixelFormat::YUV420 },
			reorder: Vec::new(),
			has_keyframe: false,
			codec_state: CodecState {
				extradata: enc.as_ref().map(H264EncoderState::avcc).unwrap_or_default(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			private: enc.map_or(CodecPrivate::None, |e| CodecPrivate::H264Encoder(Box::new(e))),
		})
	}

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut<State>> {
		let Frame::Video(frame) = frame else {
			return Err(Message::Codec("h264 encoder requires video frames"));
		};
		let enc = self.state(state)?;
		if frame.pixel.depth != 8 || frame.pixel.format != PixelFormat::YUV420 {
			return Err(Message::Codec(
				"h264 encoder needs yuv420p frames (convert with -pix_fmt yuv420p)",
			));
		}
		if frame.width != enc.width || frame.height != enc.height {
			return Err(Message::Codec("h264 encoder: frame size does not match the stream"));
		}
		let (data, idr) = enc.encode_picture(&frame.data)?;
		let mut packet = Packet::new(self.stream_id, data);
		packet.pts = frame.pts;
		packet.dts = frame.pts;
		packet.keyframe = idr;
		if let State::Video(v) = state {
			v.has_keyframe |= idr;
		}
		Ok(EncodeOut { value: vec![packet], state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<EncodeOut<State>> {
		Ok(EncodeOut { value: Vec::new(), state: state.clone() })
	}
}

#[cfg(test)]
mod tests;
