//! H.264 / AVC encoder (ISO/IEC 14496-10): Constrained Baseline, Main and
//! High profiles.
//!
//! * [`transform`] — forward 4x4 and 8x8 transforms, Hadamard DC
//!   transforms, deadzone and trellis (RD) quantisation, SATD, x264-style
//!   decimation.
//! * [`vlc`] — CAVLC residual writer (the decoder's `cavlc` inverted).
//! * [`cabac`] — CABAC arithmetic encoder and bit-cost model.
//! * [`syntax`] — the macroblock layer for both entropy coders.
//! * [`me`] — padded reference, SAD/SATD block costs.
//! * [`macroblock`] — mode decision and coding of every macroblock.
//! * [`lookahead`] — half-resolution analysis: B-frame placement and scene
//!   cuts.
//!
//! What each profile uses (x264's defaults for that profile):
//!
//! | | Baseline | Main | High |
//! |---|---|---|---|
//! | entropy coding | CAVLC | CABAC | CABAC |
//! | B pictures (adaptive, up to) | — | 3 | 3 |
//! | reference pictures | 1 | 3 | 3 |
//! | 8x8 transform, Intra8x8 | — | — | yes |
//! | trellis quantisation | — | yes | yes |
//!
//! All produce progressive 8-bit 4:2:0, one slice per picture, IDR / P /
//! (non-reference) B pictures, Intra16x16 / Intra4x4 (/ Intra8x8), P_Skip,
//! P 16x16 to 8x8 with a reference per partition, B_Skip / B_Direct_16x16
//! (spatial direct), B 16x16 / 16x8 / 8x16 with L0, L1 or bi-prediction,
//! quarter-sample motion and the in-loop deblocking filter.
//!
//! Pictures are reconstructed with the decoder's own `recon` and
//! `deblock` code, so the encoder's references are exactly what a decoder
//! reconstructs and there is no drift (the tests check this bit for bit
//! against RMP's decoder for every profile).
//!
//! GOP structure: the lookahead holds up to `bframes + 1` pictures. The
//! first of them becomes an IDR picture at the GOP interval (`-g`) or at a
//! scene cut; otherwise x264's fast adaptive B-frame decision picks the
//! next P picture, which is coded before the B pictures that precede it in
//! display order. B pictures are not used for reference, so the reorder
//! delay is one picture (`max_num_reorder_frames` 1) and packets carry
//! presentation times only (muxers derive decode times).
//!
//! Rate control: constant QP (`-qp`, P pictures; IDR 3 lower and B 2
//! higher, x264's default `ipratio` / `pbratio`), or a target bit rate
//! (`-b:v`) with a per-picture-type complexity model (`bits ≈ K / Qstep`)
//! and a one-second correction window. Without either, QP 23 is used.
//!
//! Output: `avcC` extradata (SPS + PPS) and one access unit per packet of
//! 4-byte length-prefixed NAL units, as MP4 and Matroska store them.
//!
//! FFmpeg / x264 counterparts: `x264/encoder/encoder.c` (picture loop,
//! reference lists), `set.c` (SPS/PPS), `slicetype.c` (lookahead),
//! `macroblock.c`, `analyse.c`, `me.c`, `rdo.c`, `ratecontrol.c`,
//! `cavlc.c`, `cabac.c`; FFmpeg `libavcodec/libx264.c` (packetisation,
//! `avcC`).

mod cabac;
mod lookahead;
mod macroblock;
mod me;
mod syntax;
mod transform;
mod vlc;

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use super::consts::max_dpb_mbs;
use super::deblock::deblock_frame;
use super::decode::{Frame as Picture, SliceFilter};
use super::params::{Dequant, ScalingLists};
use super::recon::Plane;
use crate::codecs::bitstream::BitWriter;
use crate::core::{
	self, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters, Pixel, PixelFormat, State,
	StreamId, Time, TimeBase, VideoFrame, VideoState,
};
use crate::message::{Message, Result};
use lookahead::Lowres;
pub use macroblock::MbStats;
use macroblock::{PicEncoder, PicParams, Reference, SliceOut, Source};
use syntax::SliceKind;

/// QP for P pictures when neither `-qp` nor `-b:v` is given.
pub const DEFAULT_QP: u8 = 23;
/// IDR interval when `-g` isn't given (x264's default `keyint`).
pub const DEFAULT_GOP: u32 = 250;
/// Most B pictures in a row (`-bf`).
pub const MAX_BFRAMES: u32 = 8;
/// Most reference pictures (`-refs`).
pub const MAX_REFS: u32 = 16;
/// IDR pictures use a QP this much lower than P pictures.
const IP_OFFSET: i32 = 3;
/// B pictures use a QP this much higher than P pictures.
const PB_OFFSET: i32 = 2;
/// `log2_max_frame_num`.
const LOG2_MAX_FRAME_NUM: u32 = 8;
/// `log2_max_pic_order_cnt_lsb` (POC type 0).
const LOG2_MAX_POC_LSB: u32 = 8;
/// Largest picture: same cap as the decoder.
const MAX_WIDTH: u32 = 8192;
const MAX_HEIGHT: u32 = 4352;

fn dequant() -> &'static Dequant {
	static D: OnceLock<Dequant> = OnceLock::new();
	D.get_or_init(|| Dequant::new(&ScalingLists::flat()))
}

/// Profile of the stream, which decides the coding tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
	/// Constrained Baseline: CAVLC, I/P, every decoder supports it.
	Baseline,
	/// Main: CABAC, B pictures, multiple references.
	Main,
	/// High: Main plus the 8x8 transform and Intra8x8.
	High,
}

impl Profile {
	/// `-profile:v` names (FFmpeg / x264 spelling).
	pub fn from_name(name: &str) -> Option<Self> {
		match name.to_ascii_lowercase().as_str() {
			"baseline" | "constrained_baseline" => Some(Self::Baseline),
			"main" => Some(Self::Main),
			"high" => Some(Self::High),
			_ => None,
		}
	}

	fn idc(self) -> u8 {
		match self {
			Self::Baseline => 66,
			Self::Main => 77,
			Self::High => 100,
		}
	}
}

/// Coding tools of an encoder instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
	pub profile: Profile,
	/// Most consecutive B pictures (0 = none; always 0 for Baseline).
	pub bframes: u32,
	/// Reference pictures P pictures may use.
	pub refs: u32,
	/// Trellis quantisation (CABAC profiles).
	pub trellis: bool,
}

impl Settings {
	/// x264's defaults for a profile.
	pub fn for_profile(profile: Profile) -> Self {
		match profile {
			Profile::Baseline => Self { profile, bframes: 0, refs: 1, trellis: false },
			Profile::Main | Profile::High => Self { profile, bframes: 3, refs: 3, trellis: true },
		}
	}

	fn cabac(&self) -> bool {
		self.profile != Profile::Baseline
	}

	fn transform_8x8(&self) -> bool {
		self.profile == Profile::High
	}

	/// The settings a profile allows.
	fn sanitised(mut self) -> Self {
		self.refs = self.refs.clamp(1, MAX_REFS);
		self.bframes =
			if self.profile == Profile::Baseline { 0 } else { self.bframes.min(MAX_BFRAMES) };
		self.trellis &= self.cabac();
		self
	}

	/// `max_num_ref_frames`: B pictures need a past and a future one.
	fn dpb_refs(&self) -> u32 {
		self.refs.max(if self.bframes > 0 { 2 } else { 1 })
	}
}

impl Default for Settings {
	fn default() -> Self {
		Self::for_profile(Profile::High)
	}
}

/// Table A-1 limits: (level_idc, MaxMBPS, MaxFS, MaxBR in kb/s for
/// Baseline/Main).
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

/// Smallest level that fits the picture size, macroblock rate, bit rate
/// (`bitrate` 0 = unknown) and decoded picture buffer.
fn choose_level(mb_w: u32, mb_h: u32, fps: f64, bitrate: u32, dpb_frames: u32) -> u8 {
	let fs = u64::from(mb_w * mb_h);
	let mbps = (fs as f64 * fps).ceil() as u64;
	for &(idc, max_mbps, max_fs, max_br) in &LEVELS {
		let side = ((8 * max_fs) as f64).sqrt() as u64;
		if fs <= max_fs
			&& u64::from(mb_w) <= side
			&& u64::from(mb_h) <= side
			&& mbps <= max_mbps
			&& u64::from(bitrate) <= max_br * 1000
			&& fs * u64::from(dpb_frames) <= u64::from(max_dpb_mbs(idc))
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

/// Picture type index for rate control: 0 = I, 1 = P, 2 = B.
fn type_index(kind: SliceKind) -> usize {
	match kind {
		SliceKind::I => 0,
		SliceKind::P => 1,
		SliceKind::B => 2,
	}
}

fn type_offset(kind: SliceKind) -> f64 {
	match kind {
		SliceKind::I => -f64::from(IP_OFFSET),
		SliceKind::P => 0.0,
		SliceKind::B => f64::from(PB_OFFSET),
	}
}

#[derive(Debug, Clone)]
struct RateControl {
	mode: RateMode,
	fps: f64,
	bits_per_frame: f64,
	spent: f64,
	budget: f64,
	/// Complexity `K = bits · Qstep` per picture type.
	complexity: [Option<f64>; 3],
	last_qp: [Option<f64>; 3],
	/// Recent share of P and B pictures (decaying counts).
	mix: [f64; 3],
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
			complexity: [None; 3],
			last_qp: [None; 3],
			mix: [0.0; 3],
			initial_qp,
		}
	}

	fn qp(&self, kind: SliceKind) -> i32 {
		let qp = match self.mode {
			RateMode::Qp(q) => f64::from(q) + type_offset(kind),
			RateMode::Bitrate(_) => {
				// Spread over- or under-spending over about a second.
				let error = self.spent - self.budget;
				let target = (self.bits_per_frame - error / self.fps.max(1.0))
					.clamp(0.25 * self.bits_per_frame, 2.0 * self.bits_per_frame);
				let p_qp = match (self.complexity[1], self.last_qp[1]) {
					(Some(kp), last) => {
						// The P quantiser that makes the recent mix of P and
						// B pictures (B at +PB_OFFSET) average to the target:
						// Σ share·K·2^(-offset/6) / Qstep(qp) = target.
						let kb = self.complexity[2].unwrap_or(kp);
						let (fp, fb) = (self.mix[1].max(1e-3), self.mix[2]);
						let k = (fp * kp + fb * kb * 2f64.powf(-f64::from(PB_OFFSET) / 6.0)) / (fp + fb);
						let q = 6.0 * (k / (target * 0.625)).log2();
						match last {
							Some(l) => q.clamp(l - 4.0, l + 4.0),
							None => q,
						}
					}
					(None, _) => self.last_qp[0].map_or(self.initial_qp, |q| q + f64::from(IP_OFFSET)),
				};
				p_qp + type_offset(kind)
			}
		};
		qp.round().clamp(if matches!(self.mode, RateMode::Qp(_)) { 0.0 } else { 10.0 }, 51.0) as i32
	}

	fn update(&mut self, kind: SliceKind, qp: i32, bits: usize) {
		let t = type_index(kind);
		let k = bits as f64 * qstep(f64::from(qp));
		self.complexity[t] = Some(match self.complexity[t] {
			Some(old) => 0.5 * old + 0.5 * k,
			None => k,
		});
		self.last_qp[t] = Some(f64::from(qp));
		for m in &mut self.mix {
			*m *= 0.9;
		}
		self.mix[t] += 1.0;
		self.spent += bits as f64;
		self.budget += self.bits_per_frame;
	}
}

/// One coded picture.
#[derive(Debug, Clone)]
pub struct CodedPicture {
	/// The access unit: length-prefixed NAL units.
	pub data: Vec<u8>,
	pub idr: bool,
	/// Slice type: 'I', 'P' or 'B'.
	pub kind: char,
	/// Position of the picture in display (input) order.
	pub index: u64,
	pub pts: Option<Time>,
	pub qp: i32,
	pub stats: MbStats,
	/// The encoder's reconstruction, cropped I420 (only with
	/// [`H264EncoderState::keep_recon`]).
	pub recon: Option<Vec<u8>>,
}

/// A picture waiting in the lookahead.
#[derive(Debug, Clone)]
struct Pending {
	src: Source,
	low: Lowres,
	pts: Option<Time>,
	index: u64,
}

/// Encoder state: parameter sets, lookahead, reference pictures, GOP
/// position and rate control.
#[derive(Debug, Clone)]
pub struct H264EncoderState {
	pub width: u32,
	pub height: u32,
	mb_w: usize,
	mb_h: usize,
	gop: u32,
	level: u8,
	pub settings: Settings,
	sps: Vec<u8>,
	pps: Vec<u8>,
	queue: VecDeque<Pending>,
	/// Lowres copy of the last coded I/P picture.
	anchor_low: Option<Lowres>,
	inputs: u64,
	/// Display index of the last IDR picture.
	idr_index: u64,
	prev_ref_frame_num: u32,
	idr_id: u32,
	/// Short-term reference pictures in coding order.
	refs: Vec<Arc<Reference>>,
	next_id: u32,
	rc: RateControl,
	/// QP of the last coded picture (for tests and statistics).
	pub last_qp: i32,
	/// Return each picture's reconstruction (tests).
	pub keep_recon: bool,
}

impl H264EncoderState {
	pub fn new(width: u32, height: u32, fps: f64, mode: RateMode, gop: u32) -> Result<Self> {
		Self::with_settings(width, height, fps, mode, gop, Settings::default())
	}

	pub fn with_settings(
		width: u32,
		height: u32,
		fps: f64,
		mode: RateMode,
		gop: u32,
		settings: Settings,
	) -> Result<Self> {
		if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
			return Err(Message::Codec("h264 encoder: width and height must be even and non-zero"));
		}
		if width > MAX_WIDTH || height > MAX_HEIGHT {
			return Err(Message::Codec("h264 encoder: picture larger than 8192x4352"));
		}
		let settings = settings.sanitised();
		let fps = if fps.is_finite() && fps > 0.0 { fps } else { 25.0 };
		let (mb_w, mb_h) = (width.div_ceil(16), height.div_ceil(16));
		let bitrate = match mode {
			RateMode::Bitrate(b) => b,
			RateMode::Qp(_) => 0,
		};
		let level = choose_level(mb_w, mb_h, fps, bitrate, settings.dpb_refs());
		let mut s = Self {
			width,
			height,
			mb_w: mb_w as usize,
			mb_h: mb_h as usize,
			gop: gop.max(1),
			level,
			settings,
			sps: Vec::new(),
			pps: Vec::new(),
			queue: VecDeque::new(),
			anchor_low: None,
			inputs: 0,
			idr_index: 0,
			prev_ref_frame_num: 0,
			idr_id: 0,
			refs: Vec::new(),
			next_id: 1,
			rc: RateControl::new(mode, fps, width * height),
			last_qp: 0,
			keep_recon: false,
		};
		s.sps = nal(7, 3, &s.write_sps(fps));
		s.pps = nal(8, 3, &s.write_pps());
		Ok(s)
	}

	fn poc_type(&self) -> u32 {
		if self.settings.bframes > 0 { 0 } else { 2 }
	}

	/// `seq_parameter_set_rbsp()` (7.3.2.1.1): the profile's parameters,
	/// cropping to the real size, and VUI timing plus
	/// `bitstream_restriction` (reorder depth, so decoders output early).
	fn write_sps(&self, fps: f64) -> Vec<u8> {
		let mut w = BitWriter::new();
		let profile = self.settings.profile;
		w.put(u32::from(profile.idc()), 8);
		let constraints = match profile {
			Profile::Baseline => 0b1100_0000, // constraint_set0 + set1: Constrained Baseline
			Profile::Main => 0b0100_0000,     // constraint_set1: Main
			Profile::High => 0,
		};
		w.put(constraints, 8);
		w.put(u32::from(self.level), 8);
		w.ue(0); // seq_parameter_set_id
		if profile == Profile::High {
			w.ue(1); // chroma_format_idc: 4:2:0
			w.ue(0); // bit_depth_luma_minus8
			w.ue(0); // bit_depth_chroma_minus8
			w.bit(false); // qpprime_y_zero_transform_bypass_flag
			w.bit(false); // seq_scaling_matrix_present_flag
		}
		w.ue(LOG2_MAX_FRAME_NUM - 4);
		w.ue(self.poc_type());
		if self.poc_type() == 0 {
			w.ue(LOG2_MAX_POC_LSB - 4);
		}
		w.ue(self.settings.dpb_refs()); // max_num_ref_frames
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
		w.ue(u32::from(self.settings.bframes > 0)); // max_num_reorder_frames
		w.ue(self.settings.dpb_refs()); // max_dec_frame_buffering
		w.rbsp_trailing();
		w.finish()
	}

	/// `pic_parameter_set_rbsp()` (7.3.2.2): the entropy coder, deblocking
	/// control present, QP 26 base, the 8x8 transform for High.
	fn write_pps(&self) -> Vec<u8> {
		let mut w = BitWriter::new();
		w.ue(0); // pic_parameter_set_id
		w.ue(0); // seq_parameter_set_id
		w.bit(self.settings.cabac()); // entropy_coding_mode_flag
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
		if self.settings.transform_8x8() {
			w.bit(true); // transform_8x8_mode_flag
			w.bit(false); // pic_scaling_matrix_present_flag
			w.se(0); // second_chroma_qp_index_offset
		}
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
		if self.settings.profile == Profile::High {
			// High-profile extension: chroma format, bit depths, no SPS ext.
			out.extend_from_slice(&[0xFC | 1, 0xF8, 0xF8, 0]);
		}
		out
	}

	/// Pads a planar 4:2:0 picture to whole macroblocks by edge
	/// replication. `planes` are (bytes, stride) of Y, Cb and Cr, each
	/// starting at its first visible sample.
	fn source(&self, planes: [(&[u8], usize); 3]) -> Result<Source> {
		let (w, h) = (self.width as usize, self.height as usize);
		let (cw, ch) = (w / 2, h / 2);
		let fits = |(data, stride): (&[u8], usize), pw: usize, ph: usize| {
			stride >= pw && (ph == 0 || data.len() >= stride * (ph - 1) + pw)
		};
		if !(fits(planes[0], w, h) && fits(planes[1], cw, ch) && fits(planes[2], cw, ch)) {
			return Err(Message::Codec("h264 encoder: frame data shorter than its yuv420p size"));
		}
		let pad = |(plane, stride): (&[u8], usize), pw: usize, ph: usize, ow: usize, oh: usize| {
			let mut p = Plane::new(ow, oh, 0);
			for y in 0..oh {
				let sy = y.min(ph - 1);
				let src = &plane[sy * stride..sy * stride + pw];
				let row = &mut p.data[y * ow..(y + 1) * ow];
				row[..pw].copy_from_slice(src);
				row[pw..].fill(src[pw - 1]);
			}
			p
		};
		let (yw, yh) = (self.mb_w * 16, self.mb_h * 16);
		Ok(Source {
			y: pad(planes[0], w, h, yw, yh),
			cb: pad(planes[1], cw, ch, yw / 2, yh / 2),
			cr: pad(planes[2], cw, ch, yw / 2, yh / 2),
		})
	}

	/// Queues `frame` (any stride) and returns the pictures that could be
	/// coded, in decoding order.
	pub fn push_frame(&mut self, frame: &VideoFrame) -> Result<Vec<CodedPicture>> {
		let plane = |i: usize| frame.plane(i).map_or((&[][..], 0), |p| (p.bytes(), p.stride()));
		let src = self.source([plane(0), plane(1), plane(2)])?;
		self.push_source(src, frame.pts)
	}

	/// Like [`Self::push_frame`] for a packed I420 picture.
	pub fn push_picture(&mut self, data: &[u8], pts: Option<Time>) -> Result<Vec<CodedPicture>> {
		let (w, h) = (self.width as usize, self.height as usize);
		let (luma, chroma) = (w * h, (w / 2) * (h / 2));
		if data.len() < luma + 2 * chroma {
			return Err(Message::Codec("h264 encoder: frame data shorter than its yuv420p size"));
		}
		let src = self.source([
			(&data[..luma], w),
			(&data[luma..luma + chroma], w / 2),
			(&data[luma + chroma..luma + 2 * chroma], w / 2),
		])?;
		self.push_source(src, pts)
	}

	fn push_source(&mut self, src: Source, pts: Option<Time>) -> Result<Vec<CodedPicture>> {
		let low = Lowres::new(&src);
		self.queue.push_back(Pending { src, low, pts, index: self.inputs });
		self.inputs += 1;
		let mut out = Vec::new();
		while self.queue.len() > self.settings.bframes as usize {
			out.extend(self.code_next());
		}
		Ok(out)
	}

	/// Codes every queued picture (end of stream).
	pub fn flush(&mut self) -> Vec<CodedPicture> {
		let mut out = Vec::new();
		while !self.queue.is_empty() {
			out.extend(self.code_next());
		}
		out
	}

	/// Decides the next anchor and codes it with the B pictures before it.
	fn code_next(&mut self) -> Vec<CodedPicture> {
		let Some(first) = self.queue.front() else { return Vec::new() };
		let since_idr = (first.index - self.idr_index) as u32;
		let force_idr = self.refs.is_empty()
			|| since_idr >= self.gop
			|| self
				.anchor_low
				.as_ref()
				.is_some_and(|a| lookahead::is_scenecut(a, &first.low, since_idr, self.gop));
		if force_idr {
			let Some(p) = self.queue.pop_front() else { return Vec::new() };
			return vec![self.code_picture(p, SliceKind::I)];
		}
		// Candidates: up to bframes + 1 pictures, before any forced IDR or
		// scene cut.
		let mut n = self.queue.len().min(self.settings.bframes as usize + 1);
		n = n.min((self.gop - since_idr) as usize).max(1);
		for k in 1..n {
			let idx = (self.queue[k].index - self.idr_index) as u32;
			if lookahead::is_scenecut(&self.queue[k - 1].low, &self.queue[k].low, idx, self.gop) {
				n = k;
				break;
			}
		}
		let nb = match &self.anchor_low {
			Some(anchor) if n >= 2 => {
				let mut frames = vec![anchor];
				frames.extend(self.queue.iter().take(n).map(|p| &p.low));
				lookahead::decide(&frames, self.settings.bframes as usize).min(n - 1)
			}
			_ => 0,
		};
		let mut batch: Vec<Pending> = self.queue.drain(..=nb).collect();
		let Some(anchor) = batch.pop() else { return Vec::new() };
		let mut out = vec![self.code_picture(anchor, SliceKind::P)];
		for b in batch {
			out.push(self.code_picture(b, SliceKind::B));
		}
		out
	}

	/// Default reference lists (8.2.4.2) for a picture of order count
	/// `poc`, truncated to the active sizes.
	fn ref_lists(&self, kind: SliceKind, poc: i32) -> (Vec<Arc<Reference>>, Vec<Arc<Reference>>) {
		match kind {
			SliceKind::I => (Vec::new(), Vec::new()),
			SliceKind::P => {
				// Descending FrameNumWrap: most recently coded first.
				let l0: Vec<_> =
					self.refs.iter().rev().take(self.settings.refs as usize).cloned().collect();
				(l0, Vec::new())
			}
			SliceKind::B => {
				let mut before: Vec<_> = self.refs.iter().filter(|r| r.poc <= poc).cloned().collect();
				let mut after: Vec<_> = self.refs.iter().filter(|r| r.poc > poc).cloned().collect();
				before.sort_by_key(|r| std::cmp::Reverse(r.poc));
				after.sort_by_key(|r| r.poc);
				let mut l0: Vec<_> = before.iter().chain(after.iter()).cloned().collect();
				let mut l1: Vec<_> = after.iter().chain(before.iter()).cloned().collect();
				if l1.len() > 1 && l0.iter().zip(&l1).all(|(a, b)| a.id == b.id) {
					l1.swap(0, 1);
				}
				l0.truncate(self.settings.refs as usize);
				l1.truncate(1);
				(l0, l1)
			}
		}
	}

	fn code_picture(&mut self, p: Pending, kind: SliceKind) -> CodedPicture {
		let idr = kind == SliceKind::I;
		let is_ref = kind != SliceKind::B;
		if idr {
			self.idr_index = p.index;
			self.refs.clear();
		}
		let frame_num = if idr { 0 } else { (self.prev_ref_frame_num + 1) % (1 << LOG2_MAX_FRAME_NUM) };
		let poc = (2 * (p.index - self.idr_index)) as i32;
		let qp = self.rc.qp(kind);
		let (l0, l1) = self.ref_lists(kind, poc);

		let mut pic = Picture::new(self.mb_w, self.mb_h);
		pic.slices.push(SliceFilter { disable: 0, alpha: 0, beta: 0 });
		let mut w = BitWriter::with_capacity(self.mb_w * self.mb_h * 32);
		self.write_slice_header(&mut w, kind, is_ref, frame_num, poc, qp, l0.len(), l1.len());
		let out = if self.settings.cabac() {
			let states = cabac::init_states(kind == SliceKind::I, 0, qp);
			SliceOut::Cabac(Box::new(cabac::CabacWriter::new(w, states)))
		} else {
			SliceOut::Cavlc(w)
		};
		let params = PicParams {
			kind,
			l0: l0.iter().map(|r| r.as_ref()).collect(),
			l1: l1.iter().map(|r| r.as_ref()).collect(),
			qp,
			transform_8x8: self.settings.transform_8x8(),
			trellis: self.settings.trellis,
			max_mv_y: max_mv_y(self.level),
		};
		let mut enc = PicEncoder::new(&p.src, &mut pic, params, dequant(), out);
		enc.encode_slice();
		let stats = enc.stats;
		let out = std::mem::replace(&mut enc.out, SliceOut::Cavlc(BitWriter::new()));
		drop(enc);
		let rbsp = match out {
			SliceOut::Cavlc(mut w) => {
				w.rbsp_trailing();
				w.finish()
			}
			SliceOut::Cabac(cw) => cw.finish(),
		};
		deblock_frame(&mut pic);

		let nal_type = if idr { 5 } else { 1 };
		let ref_idc = match kind {
			SliceKind::I => 3,
			SliceKind::P => 2,
			SliceKind::B => 0,
		};
		let slice = nal(nal_type, ref_idc, &rbsp);
		let mut data = Vec::with_capacity(slice.len() + 4);
		data.extend_from_slice(&(slice.len() as u32).to_be_bytes());
		data.extend_from_slice(&slice);
		self.rc.update(kind, qp, data.len() * 8);
		self.last_qp = qp;

		let recon = self.keep_recon.then(|| self.crop(&pic));
		if is_ref {
			let padded = me::Padded::new(&pic.y);
			let half = me::HalfPel::new(&padded);
			let motion = pic.mbs.iter().map(|m| m.motion()).collect();
			let r =
				Reference { id: self.next_id, poc, y: pic.y, cb: pic.cb, cr: pic.cr, padded, half, motion };
			self.next_id = self.next_id.wrapping_add(1).max(1);
			self.refs.push(Arc::new(r));
			// Sliding window (8.2.5.3): drop the oldest short-term picture.
			if self.refs.len() > self.settings.dpb_refs() as usize {
				self.refs.remove(0);
			}
			self.prev_ref_frame_num = frame_num;
			self.anchor_low = Some(p.low);
		}
		if idr {
			self.idr_id = (self.idr_id + 1) % 65536;
		}
		CodedPicture {
			data,
			idr,
			kind: match kind {
				SliceKind::I => 'I',
				SliceKind::P => 'P',
				SliceKind::B => 'B',
			},
			index: p.index,
			pts: p.pts,
			qp,
			stats,
			recon,
		}
	}

	/// The visible part of a reconstructed picture as packed I420.
	fn crop(&self, pic: &Picture) -> Vec<u8> {
		let (w, h) = (self.width as usize, self.height as usize);
		let mut out = Vec::with_capacity(w * h * 3 / 2);
		for y in 0..h {
			out.extend_from_slice(&pic.y.data[y * pic.y.width..y * pic.y.width + w]);
		}
		for p in [&pic.cb, &pic.cr] {
			for y in 0..h / 2 {
				out.extend_from_slice(&p.data[y * p.width..y * p.width + w / 2]);
			}
		}
		out
	}

	/// `slice_header()` (7.3.3) for a single-slice picture.
	#[allow(clippy::too_many_arguments)]
	fn write_slice_header(
		&self,
		w: &mut BitWriter,
		kind: SliceKind,
		is_ref: bool,
		frame_num: u32,
		poc: i32,
		qp: i32,
		n0: usize,
		n1: usize,
	) {
		let idr = kind == SliceKind::I;
		w.ue(0); // first_mb_in_slice
		w.ue(match kind {
			SliceKind::P => 5,
			SliceKind::B => 6,
			SliceKind::I => 7,
		}); // slice_type (all slices of the picture)
		w.ue(0); // pic_parameter_set_id
		w.put(frame_num, LOG2_MAX_FRAME_NUM);
		if idr {
			w.ue(self.idr_id);
		}
		if self.poc_type() == 0 {
			w.put(poc as u32 % (1 << LOG2_MAX_POC_LSB), LOG2_MAX_POC_LSB);
		}
		if kind == SliceKind::B {
			w.bit(true); // direct_spatial_mv_pred_flag
		}
		if kind != SliceKind::I {
			w.bit(true); // num_ref_idx_active_override_flag
			w.ue(n0.max(1) as u32 - 1);
			if kind == SliceKind::B {
				w.ue(n1.max(1) as u32 - 1);
			}
			w.bit(false); // ref_pic_list_modification_flag_l0
			if kind == SliceKind::B {
				w.bit(false); // ref_pic_list_modification_flag_l1
			}
		}
		if is_ref {
			// dec_ref_pic_marking()
			if idr {
				w.bit(false); // no_output_of_prior_pics_flag
				w.bit(false); // long_term_reference_flag
			} else {
				w.bit(false); // adaptive_ref_pic_marking_mode_flag
			}
		}
		if self.settings.cabac() && kind != SliceKind::I {
			w.ue(0); // cabac_init_idc
		}
		w.se(qp - 26); // slice_qp_delta
		w.ue(0); // disable_deblocking_filter_idc
		w.se(0); // slice_alpha_c0_offset_div2
		w.se(0); // slice_beta_offset_div2
	}
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
	settings: Settings,
}

impl H264Encoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self {
			stream_id,
			mode: RateMode::Qp(DEFAULT_QP),
			gop: DEFAULT_GOP,
			frame_rate: 25.0,
			settings: Settings::default(),
		}
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

	/// `-profile:v`: the profile's default tools (x264's). Apply before
	/// [`Self::with_bframes`] / [`Self::with_refs`].
	pub fn with_profile(mut self, profile: Option<Profile>) -> Self {
		if let Some(p) = profile {
			self.settings = Settings::for_profile(p);
		}
		self
	}

	/// `-bf`: most consecutive B pictures.
	pub fn with_bframes(mut self, bframes: Option<u32>) -> Self {
		if let Some(b) = bframes {
			self.settings.bframes = b.min(MAX_BFRAMES);
		}
		self
	}

	/// `-refs`: reference pictures for P pictures.
	pub fn with_refs(mut self, refs: Option<u32>) -> Self {
		if let Some(r) = refs {
			self.settings.refs = r.clamp(1, MAX_REFS);
		}
		self
	}

	pub fn settings(&self) -> Settings {
		self.settings.sanitised()
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
		H264EncoderState::with_settings(w, h, self.frame_rate, self.mode, self.gop, self.settings)
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

	fn packets(&self, state: &mut State, pictures: Vec<CodedPicture>, reorder: bool) -> EncodeOut {
		let mut out = Vec::with_capacity(pictures.len());
		for p in pictures {
			let mut packet = Packet::new(self.stream_id, p.data);
			packet.pts = p.pts;
			// With B pictures, decode times differ from presentation times;
			// muxers derive them from the sorted presentation times.
			packet.dts = if reorder { None } else { p.pts };
			packet.keyframe = p.idr;
			if let State::Video(v) = state {
				v.has_keyframe |= p.idr;
			}
			out.push(packet);
		}
		EncodeOut { value: out }
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

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut> {
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
		let reorder = enc.settings.bframes > 0;
		let pictures = enc.push_frame(&frame)?;
		Ok(self.packets(state, pictures, reorder))
	}

	fn flush(&self, state: &mut State) -> Result<EncodeOut> {
		let Ok(enc) = self.state(state) else {
			return Ok(EncodeOut { value: Vec::new() });
		};
		let reorder = enc.settings.bframes > 0;
		let pictures = enc.flush();
		Ok(self.packets(state, pictures, reorder))
	}
}

#[cfg(test)]
mod tests;
