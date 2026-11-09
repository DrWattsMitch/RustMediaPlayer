//! FLAC encoder (FFmpeg: `libavcodec/flacenc.c`).
//!
//! Per frame:
//!
//! 1. stereo decorrelation: the cheapest of independent, left/side,
//!    right/side and mid/side, estimated from second-order residuals;
//! 2. per channel: wasted-bit removal, then the smallest of constant,
//!    verbatim, fixed predictors (orders 0-4) and LPC (Levinson-Durbin on a
//!    Tukey-windowed autocorrelation, quantised coefficients);
//! 3. residuals: Rice coding with the best partition order and per-partition
//!    parameter, computed exactly rather than estimated.
//!
//! Compression levels 0-12 follow FFmpeg's presets (block time, maximum
//! LPC order, maximum partition order, order search). Output is a valid
//! FLAC stream that FFmpeg decodes bit-exactly back to the input.
//!
//! Frames are emitted as one packet each, with `pts` = first sample index in
//! a `1 / sample_rate` time base. The MD5 and final frame-size bounds in
//! STREAMINFO are filled in by the FLAC muxer (`container::flac::muxer`).

use super::frame_header::crc8;
use super::lpc;
use super::streaminfo::StreamInfo;
use crate::codecs::flac::decoder::crc16;
use crate::codecs::pcm::convert::for_each_sample;
use crate::core::{
	self, AudioState, Channels, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters,
	SampleFormat, SampleLayout, State, StreamId, Time, TimeBase,
};
use crate::message::{Message, Result};

/// FFmpeg's default compression level.
pub const DEFAULT_LEVEL: u8 = 5;
/// Bit depths the encoder writes.
pub const SUPPORTED_BITS: [u8; 5] = [8, 12, 16, 20, 24];

/// Block sizes with a 4-bit code in the frame header.
const BLOCK_SIZES: [u32; 13] =
	[192, 576, 1152, 2304, 4608, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

/// Per-level settings, as FFmpeg's `flac_encode_init` table.
#[derive(Debug, Clone, Copy)]
struct Preset {
	block_time_ms: u32,
	max_lpc_order: usize,
	max_partition_order: u32,
	/// 0 = fixed predictors only; 1 = estimate the LPC order; 2 = also try
	/// neighbouring orders; 3 = try every order.
	search: u8,
}

fn preset(level: u8) -> Preset {
	let level = usize::from(level.min(12));
	Preset {
		block_time_ms: [27, 27, 27, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105][level],
		max_lpc_order: [0, 0, 0, 6, 8, 8, 8, 8, 12, 12, 12, 32, 32][level],
		max_partition_order: [2, 2, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8, 8][level],
		search: [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3][level],
	}
}

/// FFmpeg's `select_blocksize`: the largest coded size at most
/// `block_time_ms` long.
pub fn select_block_size(sample_rate: u32, level: u8) -> u32 {
	let target = u64::from(sample_rate) * u64::from(preset(level).block_time_ms) / 1000;
	BLOCK_SIZES.iter().copied().filter(|&b| u64::from(b) <= target).max().unwrap_or(192)
}

/// LPC coefficient precision. FFmpeg uses 15 bits; for streams of up to 16
/// bits it is capped so `bits + precision + log2(32)` stays within 32 bits,
/// which lets every decoder use 32-bit accumulators.
fn lpc_precision(bits: u8) -> u32 {
	if bits <= 16 { 15u32.min(32 - u32::from(bits) - 5) } else { 15 }
}

/// MSB-first bit writer.
#[derive(Default)]
struct BitWriter {
	out: Vec<u8>,
	acc: u64,
	used: u32,
}

impl BitWriter {
	#[inline]
	fn put(&mut self, value: u64, bits: u32) {
		debug_assert!(bits <= 57);
		if bits == 0 {
			return;
		}
		self.acc = (self.acc << bits) | (value & ((1u64 << bits) - 1));
		self.used += bits;
		while self.used >= 8 {
			self.used -= 8;
			self.out.push((self.acc >> self.used) as u8);
		}
	}

	fn put_signed(&mut self, value: i64, bits: u32) {
		if bits > 32 {
			self.put((value >> 32) as u64, bits - 32);
			self.put(value as u64 & 0xFFFF_FFFF, 32);
		} else {
			self.put(value as u64, bits);
		}
	}

	#[inline]
	fn put_rice(&mut self, value: i64, k: u32) {
		let u = ((value << 1) ^ (value >> 63)) as u64;
		let mut q = u >> k;
		while q >= 32 {
			self.put(0, 32);
			q -= 32;
		}
		self.put(1, q as u32 + 1);
		self.put(u, k);
	}

	fn align(&mut self) {
		if self.used > 0 {
			self.put(0, 8 - self.used);
		}
	}
}

/// How a subframe is coded.
#[derive(Debug, Clone)]
enum Coding {
	Constant,
	Verbatim,
	Fixed(usize),
	Lpc { coefs: Vec<i32>, shift: u32, precision: u32 },
}

/// Rice partitioning of a residual: method (0 = 4-bit, 1 = 5-bit
/// parameters), partition order and one parameter per partition.
#[derive(Debug, Clone)]
struct RiceChoice {
	order: u32,
	params: Vec<u32>,
	bits: u64,
}

fn zigzag(v: i64) -> u64 {
	((v << 1) ^ (v >> 63)) as u64
}

/// Exact cost of coding `values` with parameter `k`.
fn rice_cost(n: u64, sum_shifted: impl Fn(u32) -> u64, k: u32) -> u64 {
	n * u64::from(k + 1) + sum_shifted(k)
}

/// Best parameter for one partition, searching around the mean's log2.
fn best_param(values: &[u64], max_param: u32) -> (u32, u64) {
	let n = values.len() as u64;
	if n == 0 {
		return (0, 0);
	}
	let sum: u64 = values.iter().sum();
	let mean = sum / n;
	let guess = if mean == 0 { 0 } else { 63 - mean.leading_zeros() };
	let mut best = (0, u64::MAX);
	for k in guess.saturating_sub(1)..=(guess + 1).min(max_param) {
		let cost = rice_cost(n, |k| values.iter().map(|v| v >> k).sum(), k);
		if cost < best.1 {
			best = (k, cost);
		}
	}
	best
}

/// Chooses partition order and parameters for `residual` (which excludes
/// the `warmup` samples).
fn choose_rice(residual: &[i64], block: usize, warmup: usize, max_order: u32) -> RiceChoice {
	let values: Vec<u64> = residual.iter().map(|&v| zigzag(v)).collect();
	let mut best = RiceChoice { order: 0, params: vec![0], bits: u64::MAX };
	for order in 0..=max_order {
		let parts = 1usize << order;
		if !block.is_multiple_of(parts) || block / parts < warmup || (order > 0 && block / parts < 16) {
			break;
		}
		let per = block / parts;
		let mut params = Vec::with_capacity(parts);
		let mut bits = 0u64;
		let mut at = 0usize;
		for p in 0..parts {
			let len = if p == 0 { per - warmup } else { per };
			let (k, cost) = best_param(&values[at..at + len], 30);
			params.push(k);
			bits += cost;
			at += len;
		}
		let wide = params.iter().any(|&k| k > 14);
		bits += parts as u64 * if wide { 5 } else { 4 } + 2 + 4;
		if bits < best.bits {
			best = RiceChoice { order, params, bits };
		}
	}
	best
}

fn fixed_residual(x: &[i64], order: usize) -> Vec<i64> {
	(order..x.len())
		.map(|i| match order {
			0 => x[i],
			1 => x[i] - x[i - 1],
			2 => x[i] - 2 * x[i - 1] + x[i - 2],
			3 => x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3],
			_ => x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4],
		})
		.collect()
}

fn lpc_residual(x: &[i64], coefs: &[i32], shift: u32) -> Option<Vec<i64>> {
	let order = coefs.len();
	let mut out = Vec::with_capacity(x.len() - order);
	for i in order..x.len() {
		let mut sum = 0i64;
		for (j, &c) in coefs.iter().enumerate() {
			sum += i64::from(c) * x[i - 1 - j];
		}
		let r = x[i] - (sum >> shift);
		// The format requires residuals to fit in 32 bits.
		if r < i64::from(i32::MIN) + 1 || r > i64::from(i32::MAX) {
			return None;
		}
		out.push(r);
	}
	Some(out)
}

/// A fully chosen subframe.
struct Subframe {
	coding: Coding,
	wasted: u32,
	/// Samples after removing wasted bits.
	samples: Vec<i64>,
	residual: Vec<i64>,
	rice: Option<RiceChoice>,
	bits: u64,
}

fn analyse(x: &[i64], bits: u32, preset: &Preset, window: &[f64]) -> Subframe {
	let n = x.len();
	if x.iter().all(|&v| v == x[0]) {
		return Subframe {
			coding: Coding::Constant,
			wasted: 0,
			samples: x.to_vec(),
			residual: Vec::new(),
			rice: None,
			bits: 8 + u64::from(bits),
		};
	}
	let or = x.iter().fold(0i64, |acc, &v| acc | v);
	let wasted = if or == 0 { 0 } else { (or.trailing_zeros()).min(bits - 1) };
	let samples: Vec<i64> = x.iter().map(|&v| v >> wasted).collect();
	let bits = bits - wasted;
	let header = 8 + if wasted > 0 { u64::from(wasted) } else { 0 };

	let mut best = Subframe {
		coding: Coding::Verbatim,
		wasted,
		samples: Vec::new(),
		residual: Vec::new(),
		rice: None,
		bits: header + n as u64 * u64::from(bits),
	};
	let consider =
		|coding: Coding, residual: Vec<i64>, warmup: usize, extra: u64, best: &mut Subframe| {
			let rice = choose_rice(&residual, n, warmup, preset.max_partition_order);
			let total = header + warmup as u64 * u64::from(bits) + extra + rice.bits;
			if total < best.bits {
				*best =
					Subframe { coding, wasted, samples: Vec::new(), residual, rice: Some(rice), bits: total };
			}
		};

	let max_fixed = if preset.search == 0 && preset.max_partition_order <= 2 { 3 } else { 4 };
	for order in 0..=max_fixed.min(n.saturating_sub(1)) {
		consider(Coding::Fixed(order), fixed_residual(&samples, order), order, 0, &mut best);
	}

	let max_order = preset.max_lpc_order.min(n.saturating_sub(1));
	if max_order > 0 && n > 32 {
		let r = lpc::autocorrelation(&samples, window, max_order);
		let models = lpc::levinson(&r, max_order);
		let precision = lpc_precision(bits as u8);
		let orders: Vec<usize> = match preset.search {
			3 => (1..=models.len()).collect(),
			_ => {
				// Estimated bits: n/2 · log2(error / n) + order · precision.
				let estimate = |(i, (_, err)): (usize, &(Vec<f64>, f64))| {
					let order = i + 1;
					let e = (err / n as f64).max(1e-9);
					(order, 0.5 * (n - order) as f64 * e.log2() + (order as u32 * precision) as f64)
				};
				let best_order = models
					.iter()
					.enumerate()
					.map(estimate)
					.min_by(|a, b| a.1.total_cmp(&b.1))
					.map_or(1, |(o, _)| o);
				if preset.search == 2 {
					(best_order.saturating_sub(2).max(1)..=(best_order + 2).min(models.len())).collect()
				} else {
					vec![best_order]
				}
			}
		};
		for order in orders {
			let Some((coefs, shift)) = lpc::quantize(&models[order - 1].0, precision) else { continue };
			let Some(residual) = lpc_residual(&samples, &coefs, shift) else { continue };
			let extra = 4 + 5 + u64::from(precision) * order as u64;
			consider(Coding::Lpc { coefs, shift, precision }, residual, order, extra, &mut best);
		}
	}
	best.samples = samples;
	best
}

fn write_subframe(w: &mut BitWriter, s: &Subframe, bits: u32) {
	let bits = bits - s.wasted;
	let kind: u64 = match &s.coding {
		Coding::Constant => 0,
		Coding::Verbatim => 1,
		Coding::Fixed(order) => 8 + *order as u64,
		Coding::Lpc { coefs, .. } => 31 + coefs.len() as u64,
	};
	w.put(0, 1);
	w.put(kind, 6);
	if s.wasted > 0 {
		w.put(1, 1);
		w.put(1, s.wasted); // unary (wasted - 1) zeros, then a one
	} else {
		w.put(0, 1);
	}
	match &s.coding {
		Coding::Constant => w.put_signed(s.samples[0], bits),
		Coding::Verbatim => {
			for &v in &s.samples {
				w.put_signed(v, bits);
			}
		}
		Coding::Fixed(order) => {
			for &v in &s.samples[..*order] {
				w.put_signed(v, bits);
			}
		}
		Coding::Lpc { coefs, shift, precision } => {
			for &v in &s.samples[..coefs.len()] {
				w.put_signed(v, bits);
			}
			w.put(u64::from(precision - 1), 4);
			w.put(u64::from(*shift), 5);
			for &c in coefs {
				w.put_signed(i64::from(c), *precision);
			}
		}
	}
	if let Some(rice) = &s.rice {
		let wide = rice.params.iter().any(|&k| k > 14);
		w.put(u64::from(wide), 2);
		w.put(u64::from(rice.order), 4);
		let parts = rice.params.len();
		let block = s.samples.len();
		let warmup = block - s.residual.len();
		let per = block / parts;
		let mut at = 0;
		for (p, &k) in rice.params.iter().enumerate() {
			w.put(u64::from(k), if wide { 5 } else { 4 });
			let len = if p == 0 { per - warmup } else { per };
			for &v in &s.residual[at..at + len] {
				w.put_rice(v, k);
			}
			at += len;
		}
	}
}

/// Cost estimate of a channel for stereo-mode selection: Rice bits of its
/// second-order fixed residual (FFmpeg's `estimate_stereo_mode`).
fn stereo_cost(x: &[i64]) -> u64 {
	let sum: u64 = x.windows(3).map(|w| (w[2] - 2 * w[1] + w[0]).unsigned_abs()).sum();
	let n = x.len().max(1) as u64;
	let mean = sum / n;
	let k = if mean == 0 { 0 } else { 63 - u64::from(mean.leading_zeros()) };
	n * (k + 1) + (sum >> k)
}

/// Writes a frame number in FLAC's UTF-8-like code (up to 36 bits).
fn put_coded_number(w: &mut BitWriter, n: u64) {
	if n < 0x80 {
		w.put(n, 8);
		return;
	}
	// `c` continuation bytes carry 6 bits each; the lead byte 6 - c bits.
	let mut c = 1u32;
	while c < 6 && n >= 1u64 << (5 * c + 6) {
		c += 1;
	}
	let lead = (0xFF00u16 >> (c + 1)) as u8;
	w.put(u64::from(lead) | (n >> (6 * c)), 8);
	for i in (0..c).rev() {
		w.put(0x80 | ((n >> (6 * i)) & 0x3F), 8);
	}
}

/// Encoder settings and pending samples.
#[derive(Debug, Clone)]
pub struct FlacEncoderState {
	pub info: StreamInfo,
	pub level: u8,
	/// Input not yet coded, per channel.
	pending: Vec<Vec<i64>>,
	frame_number: u64,
	samples_coded: u64,
	window: Vec<f64>,
}

impl FlacEncoderState {
	pub fn new(info: StreamInfo, level: u8) -> Self {
		Self {
			info,
			level,
			pending: vec![Vec::new(); usize::from(info.channels)],
			frame_number: 0,
			samples_coded: 0,
			window: lpc::tukey_window(usize::from(info.max_block_size)),
		}
	}

	/// Codes one frame from the first `block` pending samples.
	fn encode_frame(&mut self, block: usize) -> Vec<u8> {
		let info = self.info;
		let bits = u32::from(info.bits_per_sample);
		let preset = preset(self.level);
		let window =
			if block == self.window.len() { self.window.clone() } else { lpc::tukey_window(block) };
		let channels: Vec<Vec<i64>> =
			self.pending.iter_mut().map(|c| c.drain(..block).collect()).collect();

		// Channel assignment code and the signals to code.
		let (assignment, signals): (u8, Vec<(Vec<i64>, u32)>) = if channels.len() == 2 {
			let (l, r) = (&channels[0], &channels[1]);
			let side: Vec<i64> = l.iter().zip(r).map(|(a, b)| a - b).collect();
			let mid: Vec<i64> = l.iter().zip(r).map(|(a, b)| (a + b) >> 1).collect();
			let (cl, cr, cs, cm) =
				(stereo_cost(l), stereo_cost(r), stereo_cost(&side), stereo_cost(&mid));
			let options = [(1u8, cl + cr), (8, cl + cs), (9, cr + cs), (10, cm + cs)];
			let mode = options.iter().min_by_key(|o| o.1).map_or(1, |o| o.0);
			match mode {
				8 => (8, vec![(l.clone(), bits), (side, bits + 1)]),
				9 => (9, vec![(side, bits + 1), (r.clone(), bits)]),
				10 => (10, vec![(mid, bits), (side, bits + 1)]),
				_ => (1, vec![(l.clone(), bits), (r.clone(), bits)]),
			}
		} else {
			(channels.len() as u8 - 1, channels.into_iter().map(|c| (c, bits)).collect())
		};

		let mut w = BitWriter::default();
		self.write_header(&mut w, block as u32, assignment);
		for (signal, depth) in &signals {
			let sub = analyse(signal, *depth, &preset, &window);
			write_subframe(&mut w, &sub, *depth);
		}
		w.align();
		let crc = crc16(&w.out);
		w.put(u64::from(crc), 16);
		self.frame_number += 1;
		self.samples_coded += block as u64;
		w.out
	}

	fn write_header(&self, w: &mut BitWriter, block: u32, assignment: u8) {
		let info = &self.info;
		w.put(0x7FFC, 15);
		w.put(0, 1); // fixed block size
		let (block_code, block_extra) = match BLOCK_SIZES.iter().position(|&b| b == block) {
			Some(i) => (if i < 5 { i as u64 + 1 } else { i as u64 + 3 }, None),
			None if block <= 256 => (6, Some((u64::from(block - 1), 8))),
			None => (7, Some((u64::from(block - 1), 16))),
		};
		let rates =
			[88_200, 176_400, 192_000, 8000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000];
		let rate = info.sample_rate;
		let (rate_code, rate_extra) = match rates.iter().position(|&r| r == rate) {
			Some(i) => (i as u64 + 1, None),
			None if rate.is_multiple_of(1000) && rate / 1000 <= 255 => {
				(12, Some((u64::from(rate / 1000), 8)))
			}
			None if rate <= 65_535 => (13, Some((u64::from(rate), 16))),
			None if rate.is_multiple_of(10) && rate / 10 <= 65_535 => {
				(14, Some((u64::from(rate / 10), 16)))
			}
			None => (0, None),
		};
		let depth_code = match info.bits_per_sample {
			8 => 1,
			12 => 2,
			16 => 4,
			20 => 5,
			24 => 6,
			32 => 7,
			_ => 0,
		};
		w.put(block_code, 4);
		w.put(rate_code, 4);
		w.put(u64::from(assignment), 4);
		w.put(depth_code, 3);
		w.put(0, 1);
		put_coded_number(w, self.frame_number);
		if let Some((v, b)) = block_extra {
			w.put(v, b);
		}
		if let Some((v, b)) = rate_extra {
			w.put(v, b);
		}
		let crc = crc8(&w.out);
		w.put(u64::from(crc), 8);
	}
}

#[derive(Debug, Clone, Copy)]
pub struct FlacEncoder {
	stream_id: StreamId,
	level: u8,
}

impl FlacEncoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self { stream_id, level: DEFAULT_LEVEL }
	}

	/// Compression level 0 (fastest) to 12 (smallest), as FFmpeg's
	/// `-compression_level`.
	pub fn with_level(mut self, level: u8) -> Self {
		self.level = level.min(12);
		self
	}

	/// Bit depth used for `parameters`: `bits_per_sample` when it is one of
	/// [`SUPPORTED_BITS`], otherwise 16.
	pub fn bits_for(parameters: &Parameters) -> u8 {
		parameters.bits_per_sample.filter(|b| SUPPORTED_BITS.contains(b)).unwrap_or(16)
	}

	/// STREAMINFO the encoder will produce (frame sizes, sample count and
	/// MD5 are left unknown; the FLAC muxer fills them in at the end). Use
	/// it as the output stream's extradata before adding it to a muxer.
	pub fn stream_info(&self, parameters: &Parameters) -> Result<StreamInfo> {
		let sample_rate = parameters.sample_rate.filter(|&r| r > 0 && r < 1 << 20);
		let channels = parameters.channels.filter(|&c| (1..=8).contains(&c));
		let (Some(sample_rate), Some(channels)) = (sample_rate, channels) else {
			return Err(Message::Codec("flac encoder needs 1-8 channels and a valid sample rate"));
		};
		let block = select_block_size(sample_rate, self.level) as u16;
		Ok(StreamInfo {
			min_block_size: block,
			max_block_size: block,
			min_frame_size: 0,
			max_frame_size: 0,
			sample_rate,
			channels,
			bits_per_sample: Self::bits_for(parameters),
			total_samples: 0,
			md5: [0; 16],
		})
	}

	/// Codec extradata (34-byte STREAMINFO) for `parameters`.
	pub fn extradata(&self, parameters: &Parameters) -> Result<Vec<u8>> {
		Ok(self.stream_info(parameters)?.to_bytes().to_vec())
	}

	fn packet(&self, data: Vec<u8>, first: u64, block: usize, rate: u32) -> Packet {
		let tb = TimeBase::new(1, rate);
		let mut packet = Packet::new(self.stream_id, data);
		packet.pts = Some(Time::new(first, tb));
		packet.dts = packet.pts;
		packet.duration = Some(Time::new(block as u64, tb));
		packet
	}

	fn state<'a>(&self, state: &'a mut State) -> Result<&'a mut FlacEncoderState> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("flac encoder requires an audio state"));
		};
		match &mut audio.private {
			CodecPrivate::FlacEncoder(enc) => Ok(enc),
			_ => Err(Message::Codec("flac encoder: invalid parameters (sample rate / channels)")),
		}
	}
}

impl core::Encoder for FlacEncoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let info = self.stream_info(parameters).ok();
		let private = info.map_or(CodecPrivate::None, |i| {
			CodecPrivate::FlacEncoder(Box::new(FlacEncoderState::new(i, self.level)))
		});
		State::Audio(AudioState {
			sample_rate: parameters.sample_rate.unwrap_or(44_100),
			channels: Channels::from_count(parameters.channels.unwrap_or(2)),
			format: if Self::bits_for(parameters) > 16 { SampleFormat::S32 } else { SampleFormat::S16 },
			layout: SampleLayout::Interleaved,
			frame_size: info.map_or(4096, |i| usize::from(i.max_block_size)),
			codec_state: CodecState {
				extradata: info.map(|i| i.to_bytes().to_vec()).unwrap_or_default(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private,
		})
	}

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut<State>> {
		let Frame::Audio(frame) = frame else {
			return Err(Message::Codec("flac encoder requires audio frames"));
		};
		let enc = self.state(state)?;
		let channels = usize::from(enc.info.channels);
		if usize::from(frame.channels.count()) != channels {
			return Err(Message::Codec("flac encoder: frame channel count does not match the stream"));
		}
		if frame.sample_rate != enc.info.sample_rate {
			return Err(Message::Codec(
				"flac encoder: frame sample rate does not match the stream (resample first)",
			));
		}
		// Samples are taken at the stream depth the way FFmpeg feeds its
		// encoder: through S16 for <= 16 bits, through S32 above.
		let bits = u32::from(enc.info.bits_per_sample);
		let mut ch = 0usize;
		for_each_sample(&frame, |s| {
			let v = if bits <= 16 {
				i64::from(s.to_i16() >> (16 - bits))
			} else {
				i64::from(s.to_i32() >> (32 - bits))
			};
			enc.pending[ch].push(v);
			ch = (ch + 1) % channels;
		})?;

		let block = usize::from(enc.info.max_block_size);
		let rate = enc.info.sample_rate;
		let mut packets = Vec::new();
		while enc.pending[0].len() >= block {
			let first = enc.samples_coded;
			let data = enc.encode_frame(block);
			packets.push(self.packet(data, first, block, rate));
		}
		if let State::Audio(audio) = state {
			audio.samples_consumed += frame.nb_samples as u64;
		}
		Ok(EncodeOut { value: packets, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<EncodeOut<State>> {
		let enc = self.state(state)?;
		let mut packets = Vec::new();
		let rest = enc.pending[0].len();
		if rest > 0 {
			let first = enc.samples_coded;
			let rate = enc.info.sample_rate;
			let data = enc.encode_frame(rest);
			packets.push(self.packet(data, first, rest, rate));
		}
		Ok(EncodeOut { value: packets, state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::super::decoder::decode_frame;
	use super::*;

	#[test]
	fn block_sizes_follow_ffmpeg() {
		assert_eq!(select_block_size(44_100, 5), 4608);
		assert_eq!(select_block_size(48_000, 5), 4608);
		assert_eq!(select_block_size(96_000, 5), 8192);
		assert_eq!(select_block_size(8000, 5), 576);
		assert_eq!(select_block_size(44_100, 0), 1152);
	}

	#[test]
	fn frames_decode_back_to_the_input() {
		let n = 4608usize;
		let left: Vec<i64> = (0..n).map(|i| ((i as f64 * 0.031).sin() * 12_000.0) as i64).collect();
		let right: Vec<i64> =
			left.iter().enumerate().map(|(i, v)| v / 2 + (i as i64 % 7) - 3).collect();
		for level in [0u8, 5, 12] {
			for bits in [16u8, 24] {
				let scale = if bits == 24 { 256 } else { 1 };
				let info = StreamInfo {
					min_block_size: n as u16,
					max_block_size: n as u16,
					min_frame_size: 0,
					max_frame_size: 0,
					sample_rate: 44_100,
					channels: 2,
					bits_per_sample: bits,
					total_samples: 0,
					md5: [0; 16],
				};
				let mut enc = FlacEncoderState::new(info, level);
				enc.pending[0] = left.iter().map(|v| v * scale).collect();
				enc.pending[1] = right.iter().map(|v| v * scale).collect();
				let frame = enc.encode_frame(n);
				let decoded = decode_frame(&frame, Some(&info)).unwrap();
				assert!(decoded.crc_ok);
				assert_eq!(
					decoded.channels[0],
					left.iter().map(|v| (v * scale) as i32).collect::<Vec<_>>()
				);
				assert_eq!(
					decoded.channels[1],
					right.iter().map(|v| (v * scale) as i32).collect::<Vec<_>>()
				);
				assert!(frame.len() < n * 2 * usize::from(bits) / 8 / 2, "level {level}: {}", frame.len());
			}
		}
	}

	#[test]
	fn large_frame_numbers_are_utf8_coded() {
		let info = StreamInfo {
			min_block_size: 576,
			max_block_size: 576,
			min_frame_size: 0,
			max_frame_size: 0,
			sample_rate: 44_100,
			channels: 1,
			bits_per_sample: 16,
			total_samples: 0,
			md5: [0; 16],
		};
		for number in
			[0u64, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x1_0000, 0x7FF_FFFF, 0x7FFF_FFFF, (1 << 36) - 1]
		{
			let mut enc = FlacEncoderState::new(info, 0);
			enc.frame_number = number;
			enc.pending[0] = vec![5; 576];
			let frame = enc.encode_frame(576);
			let header = super::super::FrameHeader::parse(&frame).unwrap();
			assert_eq!(header.number, number);
		}
	}
}
