//! ALAC frame decoding (FFmpeg `alac.c`, `alacdsp.c`).

use crate::codecs::bitstream::BitReader;
use crate::core::{
	AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Decoder, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State,
};
use crate::message::{Message, Result};

const MAX_CHANNELS: usize = 8;
/// Bitstream element types (the AAC syntax ALAC borrows).
#[cfg(test)]
const TYPE_SCE: u32 = 0;
const TYPE_CPE: u32 = 1;
const TYPE_LFE: u32 = 3;
const TYPE_END: u32 = 7;

/// Where each coded channel goes in FFmpeg's channel order, by channel
/// count (ALAC codes C, L, R, ...; FFmpeg's layouts are L, R, C, LFE, ...).
const CHANNEL_OFFSETS: [[u8; MAX_CHANNELS]; MAX_CHANNELS] = [
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 0, 0],
	[2, 0, 1, 0, 0, 0, 0, 0],
	[2, 0, 1, 3, 0, 0, 0, 0],
	[2, 0, 1, 3, 4, 0, 0, 0],
	[2, 0, 1, 4, 5, 3, 0, 0],
	[2, 0, 1, 4, 5, 6, 3, 0],
	[2, 6, 7, 0, 1, 4, 5, 3],
];

fn corrupt(what: &'static str) -> Message {
	Message::Codec(what)
}

/// `ALACSpecificConfig` (Apple's `ALACMagicCookieDescription.txt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlacConfig {
	pub frame_length: u32,
	pub bit_depth: u8,
	pub rice_history_mult: u8,
	pub rice_initial_history: u8,
	pub rice_limit: u8,
	pub channels: u8,
	pub max_run: u16,
	pub max_frame_bytes: u32,
	pub avg_bit_rate: u32,
	pub sample_rate: u32,
}

impl AlacConfig {
	/// Parses the config from extradata in any of its wrappings (see the
	/// module docs).
	pub fn from_extradata(data: &[u8]) -> Option<Self> {
		let config = if data.len() >= 36 && &data[4..8] == b"alac" {
			&data[12..]
		} else if data.len() >= 28 && data[..4] == [0, 0, 0, 0] {
			// Version and flags of the `alac` box; a bare config can't start
			// with four zero bytes (its frame length is never 0).
			&data[4..]
		} else {
			data
		};
		Self::parse(config)
	}

	/// Parses a bare 24-byte `ALACSpecificConfig`.
	pub fn parse(c: &[u8]) -> Option<Self> {
		if c.len() < 24 {
			return None;
		}
		let be32 = |at: usize| u32::from_be_bytes([c[at], c[at + 1], c[at + 2], c[at + 3]]);
		let config = Self {
			frame_length: be32(0),
			bit_depth: c[5],
			rice_history_mult: c[6],
			rice_initial_history: c[7],
			rice_limit: c[8],
			channels: c[9],
			max_run: u16::from_be_bytes([c[10], c[11]]),
			max_frame_bytes: be32(12),
			avg_bit_rate: be32(16),
			sample_rate: be32(20),
		};
		let valid = config.frame_length > 0
			&& config.frame_length <= 4096 * 4096
			&& matches!(config.bit_depth, 16 | 20 | 24 | 32)
			&& (1..=MAX_CHANNELS as u8).contains(&config.channels);
		valid.then_some(config)
	}

	/// The 24-byte config.
	pub fn to_bytes(&self) -> [u8; 24] {
		let mut out = [0u8; 24];
		out[0..4].copy_from_slice(&self.frame_length.to_be_bytes());
		out[5] = self.bit_depth;
		out[6] = self.rice_history_mult;
		out[7] = self.rice_initial_history;
		out[8] = self.rice_limit;
		out[9] = self.channels;
		out[10..12].copy_from_slice(&self.max_run.to_be_bytes());
		out[12..16].copy_from_slice(&self.max_frame_bytes.to_be_bytes());
		out[16..20].copy_from_slice(&self.avg_bit_rate.to_be_bytes());
		out[20..24].copy_from_slice(&self.sample_rate.to_be_bytes());
		out
	}

	/// FFmpeg's extradata layout: the 36-byte `alac` atom (size, `alac`,
	/// version and flags, then the config).
	pub fn to_atom(&self) -> Vec<u8> {
		let mut out = 36u32.to_be_bytes().to_vec();
		out.extend(b"alac");
		out.extend([0; 4]);
		out.extend(self.to_bytes());
		out
	}

	/// The body of an MP4 `alac` box (version and flags, then the config)
	/// from extradata in any wrapping.
	pub fn box_body(extradata: &[u8]) -> Option<Vec<u8>> {
		let config = Self::from_extradata(extradata)?;
		let mut out = vec![0; 4];
		out.extend(config.to_bytes());
		Some(out)
	}

	pub fn sample_format(&self) -> SampleFormat {
		if self.bit_depth == 16 { SampleFormat::S16 } else { SampleFormat::S32 }
	}
}

/// FFmpeg's `sign_extend`: the low `bits` bits of `v` as a signed value.
#[inline]
fn sign_extend(v: u32, bits: u32) -> i32 {
	let shift = 32 - bits;
	((v << shift) as i32) >> shift
}

#[inline]
fn sign_only(v: i32) -> i32 {
	v.signum()
}

/// `av_log2`: index of the highest set bit (0 for 0).
#[inline]
fn log2(v: u32) -> u32 {
	31 - (v | 1).leading_zeros()
}

/// One adaptive-Golomb value (`decode_scalar`): a unary prefix of up to 8
/// ones, then `k` bits coding `x * (2^k - 1) + extra`; nine ones escape to
/// a raw `bps`-bit value.
#[inline]
fn decode_scalar(br: &mut BitReader, k: u32, bps: u32) -> u32 {
	let mut x = 0;
	while x < 9 && br.bit() {
		x += 1;
	}
	if x > 8 {
		return br.bits(bps);
	}
	if k != 1 {
		let extra = br.peek(k);
		x = (x << k) - x;
		if extra > 1 {
			x += extra - 1;
			br.skip(k as usize);
		} else {
			br.skip(k as usize - 1);
		}
	}
	x
}

/// Residuals of one channel (`rice_decompress`).
fn rice_decompress(
	br: &mut BitReader,
	out: &mut [i32],
	bps: u32,
	history_mult: u32,
	config: &AlacConfig,
) -> Result<()> {
	let n = out.len();
	let limit = u32::from(config.rice_limit);
	let mut history = u32::from(config.rice_initial_history);
	let mut sign_modifier = 0u32;
	let mut i = 0;
	while i < n {
		if br.bits_left() == 0 {
			return Err(corrupt("alac: residuals run past the packet"));
		}
		let k = log2((history >> 9) + 3).min(limit);
		let x = decode_scalar(br, k, bps).wrapping_add(sign_modifier);
		sign_modifier = 0;
		out[i] = ((x >> 1) as i32) ^ -((x & 1) as i32);
		if x > 0xffff {
			history = 0xffff;
		} else {
			history = history
				.wrapping_add(x.wrapping_mul(history_mult))
				.wrapping_sub(history.wrapping_mul(history_mult) >> 9);
		}
		// Runs of zeros.
		if history < 128 && i + 1 < n {
			let k = (7 - log2(history) + ((history + 16) >> 6)).min(limit);
			let mut block = decode_scalar(br, k, 16) as usize;
			if block > 0 {
				if block >= n - i {
					block = n - i - 1;
				}
				out[i + 1..i + 1 + block].fill(0);
				i += block;
			}
			if block <= 0xffff {
				sign_modifier = 1;
			}
			history = 0;
		}
		i += 1;
	}
	Ok(())
}

/// The adaptive FIR predictor (`lpc_prediction`), in place: `buf` holds the
/// residuals on entry and the samples on return. All arithmetic wraps at
/// 32 bits as in the reference decoder.
fn lpc_prediction(buf: &mut [i32], bps: u32, coefs: &mut [i16], order: usize, quant: u32) {
	let n = buf.len();
	if n <= 1 || order == 0 {
		return;
	}
	if order == 31 {
		for i in 1..n {
			buf[i] = sign_extend((buf[i - 1] as u32).wrapping_add(buf[i] as u32), bps);
		}
		return;
	}
	let mut i = 1;
	while i <= order && i < n {
		buf[i] = sign_extend((buf[i - 1] as u32).wrapping_add(buf[i] as u32), bps);
		i += 1;
	}
	let round = 1i64 << (quant - 1);
	while i < n {
		let base = i - order - 1;
		let d = buf[base] as u32;
		let mut error = buf[i] as u32;
		let mut acc = 0u32;
		for j in 0..order {
			let p = buf[base + 1 + j] as u32;
			acc = acc.wrapping_add(p.wrapping_sub(d).wrapping_mul(coefs[j] as i32 as u32));
		}
		let val = ((i64::from(acc as i32) + round) >> quant) as i32 as u32;
		buf[i] = sign_extend(val.wrapping_add(d).wrapping_add(error), bps);

		let error_sign = sign_only(error as i32);
		if error_sign != 0 {
			let mut j = 0;
			while j < order && (error.wrapping_mul(error_sign as u32) as i32) > 0 {
				let v = (d as i32).wrapping_sub(buf[base + 1 + j]);
				let sign = sign_only(v) * error_sign;
				coefs[j] = coefs[j].wrapping_sub(sign as i16);
				let v = v.wrapping_mul(sign);
				error = error.wrapping_sub(((v >> quant) as u32).wrapping_mul(j as u32 + 1));
				j += 1;
			}
		}
		i += 1;
	}
}

/// Decoder state: the configuration and per-channel work buffers.
#[derive(Debug, Clone)]
pub struct AlacDecoderState {
	pub config: Option<AlacConfig>,
	/// Samples per channel, in coded order within an element.
	samples: [Vec<i32>; 2],
	extra: [Vec<u32>; 2],
}

impl AlacDecoderState {
	pub fn new(extradata: &[u8]) -> Self {
		Self {
			config: AlacConfig::from_extradata(extradata),
			samples: [Vec::new(), Vec::new()],
			extra: [Vec::new(), Vec::new()],
		}
	}

	/// Decodes one packet (one ALAC frame, every channel) into an
	/// interleaved frame.
	pub fn decode_packet(&mut self, data: &[u8]) -> Result<AudioFrame> {
		let config = self.config.ok_or(corrupt("alac: missing or invalid ALACSpecificConfig"))?;
		let channels = usize::from(config.channels);
		let format = config.sample_format();
		let mut br = BitReader::new(data);
		let mut nb_samples = 0usize;
		let mut ch = 0usize;
		let mut got_end = false;
		// Decoded samples, interleaved in FFmpeg's channel order; sized once
		// the first element gives the sample count.
		let mut out: Vec<i32> = Vec::new();
		while br.bits_left() >= 3 {
			let element = br.bits(3);
			if element == TYPE_END {
				got_end = true;
				break;
			}
			if element > TYPE_CPE && element != TYPE_LFE {
				return Err(corrupt("alac: unsupported channel element"));
			}
			let count = if element == TYPE_CPE { 2 } else { 1 };
			if ch + count > channels {
				return Err(corrupt("alac: more channels than the config declares"));
			}
			let offset = usize::from(CHANNEL_OFFSETS[channels - 1][ch]);
			if offset + count > channels {
				return Err(corrupt("alac: invalid channel layout"));
			}
			let n = self.decode_element(&mut br, &config, count, nb_samples)?;
			if nb_samples == 0 {
				nb_samples = n;
				out = vec![0; n * channels];
			}
			for c in 0..count {
				let shift = match config.bit_depth {
					20 => 12,
					24 => 8,
					_ => 0,
				};
				for (i, &s) in self.samples[c][..n].iter().enumerate() {
					out[i * channels + offset + c] = ((s as u32) << shift) as i32;
				}
			}
			ch += count;
		}
		if !got_end {
			return Err(corrupt("alac: no end tag (incomplete packet)"));
		}
		if ch != channels || nb_samples == 0 {
			return Err(corrupt("alac: not every channel was decoded"));
		}
		let bytes = match format {
			SampleFormat::S16 => out.iter().flat_map(|&s| (s as i16).to_le_bytes()).collect(),
			_ => out.iter().flat_map(|&s| s.to_le_bytes()).collect(),
		};
		Ok(AudioFrame::new(
			bytes,
			config.sample_rate,
			Channels::from_count(config.channels),
			format,
			SampleLayout::Interleaved,
		))
	}

	/// Decodes one SCE/CPE/LFE element of `channels` (1 or 2) channels into
	/// `self.samples`; returns its sample count.
	fn decode_element(
		&mut self,
		br: &mut BitReader,
		config: &AlacConfig,
		channels: usize,
		expected: usize,
	) -> Result<usize> {
		br.skip(4); // element instance tag
		br.skip(12); // unused header bits
		let has_size = br.bit();
		let mut extra_bits = br.bits(2) << 3;
		let sample_size = u32::from(config.bit_depth);
		let bps = (sample_size + channels as u32 - 1)
			.checked_sub(extra_bits)
			.filter(|&b| (1..=32).contains(&b))
			.ok_or(corrupt("alac: invalid sample size"))?;
		let compressed = !br.bit();
		let n = if has_size { br.bits(32) } else { config.frame_length };
		if n == 0 || n > config.frame_length {
			return Err(corrupt("alac: invalid sample count"));
		}
		let n = n as usize;
		if expected != 0 && n != expected {
			return Err(corrupt("alac: elements disagree on the sample count"));
		}
		for c in 0..channels {
			self.samples[c].resize(n, 0);
		}
		let (mut decorr_shift, mut decorr_left_weight) = (0u32, 0i32);
		if compressed {
			if config.rice_limit == 0 {
				return Err(corrupt("alac: rice limit of 0"));
			}
			decorr_shift = br.bits(8);
			decorr_left_weight = br.bits(8) as i32;
			if channels == 2 && decorr_left_weight != 0 && decorr_shift > 31 {
				return Err(corrupt("alac: invalid stereo decorrelation"));
			}
			let mut prediction = [0u32; 2];
			let mut quant = [0u32; 2];
			let mut history_mult = [0u32; 2];
			let mut order = [0usize; 2];
			let mut coefs = [[0i16; 32]; 2];
			for c in 0..channels {
				prediction[c] = br.bits(4);
				quant[c] = br.bits(4);
				history_mult[c] = br.bits(3);
				order[c] = br.bits(5) as usize;
				if order[c] >= config.frame_length as usize || quant[c] == 0 {
					return Err(corrupt("alac: invalid predictor"));
				}
				for i in (0..order[c]).rev() {
					coefs[c][i] = br.signed(16) as i16;
				}
			}
			if extra_bits != 0 {
				for c in 0..channels {
					self.extra[c].resize(n, 0);
				}
				for i in 0..n {
					if br.bits_left() == 0 {
						return Err(corrupt("alac: extra bits run past the packet"));
					}
					for c in 0..channels {
						self.extra[c][i] = br.bits(extra_bits);
					}
				}
			}
			for c in 0..channels {
				let mult = history_mult[c] * u32::from(config.rice_history_mult) / 4;
				rice_decompress(br, &mut self.samples[c][..n], bps, mult, config)?;
				if prediction[c] == 15 {
					// Two passes: first-order, then the coded predictor
					// (unused by Apple's encoder).
					lpc_prediction(&mut self.samples[c][..n], bps, &mut [], 31, 0);
				}
				lpc_prediction(&mut self.samples[c][..n], bps, &mut coefs[c], order[c], quant[c]);
			}
		} else {
			for i in 0..n {
				if br.bits_left() == 0 {
					return Err(corrupt("alac: samples run past the packet"));
				}
				for c in 0..channels {
					self.samples[c][i] = br.signed(sample_size);
				}
			}
			extra_bits = 0;
		}
		if br.overrun() {
			return Err(corrupt("alac: element runs past the packet"));
		}

		if channels == 2 && decorr_left_weight != 0 {
			let (left, right) = self.samples.split_at_mut(1);
			for (a, b) in left[0][..n].iter_mut().zip(&mut right[0][..n]) {
				let mut x = *a as u32;
				let mut y = *b as u32;
				x = x.wrapping_sub(
					((y.wrapping_mul(decorr_left_weight as u32) as i32) >> decorr_shift) as u32,
				);
				y = y.wrapping_add(x);
				*a = y as i32;
				*b = x as i32;
			}
		}
		if extra_bits != 0 {
			for c in 0..channels {
				for (s, &e) in self.samples[c][..n].iter_mut().zip(&self.extra[c][..n]) {
					*s = (((*s as u32) << extra_bits) | e) as i32;
				}
			}
		}
		Ok(n)
	}
}

#[derive(Debug, Clone, Default)]
pub struct AlacDecoder;

impl AlacDecoder {
	pub fn new() -> Self {
		Self
	}
}

impl Decoder for AlacDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec = AlacDecoderState::new(&parameters.extradata);
		let config = dec.config;
		State::Audio(AudioState {
			sample_rate: config.map_or(parameters.sample_rate.unwrap_or(44_100), |c| c.sample_rate),
			channels: Channels::from_count(
				config.map_or(parameters.channels.unwrap_or(2), |c| c.channels),
			),
			format: config.map_or(SampleFormat::S16, |c| c.sample_format()),
			layout: SampleLayout::Interleaved,
			frame_size: config.map_or(4096, |c| c.frame_length as usize),
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::Alac(Box::new(dec)),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("alac decoder requires an audio state"));
		};
		if !matches!(audio.private, CodecPrivate::Alac(_)) {
			audio.private =
				CodecPrivate::Alac(Box::new(AlacDecoderState::new(&audio.codec_state.extradata)));
		}
		let CodecPrivate::Alac(dec) = &mut audio.private else {
			return Err(Message::Codec("alac decoder state missing"));
		};
		if packet.data.is_empty() {
			return Ok(DecodeOut { value: Vec::new() });
		}
		let mut frame = dec.decode_packet(&packet.data)?;
		frame.pts = packet.pts;
		audio.sample_rate = frame.sample_rate;
		audio.channels = frame.channels;
		audio.format = frame.format;
		audio.samples_consumed += packet.data.len() as u64;
		audio.samples_produced += frame.nb_samples as u64;
		Ok(DecodeOut { value: vec![Frame::Audio(frame)] })
	}

	fn flush(&self, _state: &mut State) -> Result<DecodeOut> {
		Ok(DecodeOut { value: Vec::new() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn config_wrappings() {
		let c = AlacConfig {
			frame_length: 4096,
			bit_depth: 24,
			rice_history_mult: 40,
			rice_initial_history: 10,
			rice_limit: 14,
			channels: 2,
			max_run: 255,
			max_frame_bytes: 0,
			avg_bit_rate: 0,
			sample_rate: 48_000,
		};
		let bare = c.to_bytes();
		assert_eq!(AlacConfig::from_extradata(&bare), Some(c));
		let mut boxed = vec![0, 0, 0, 0];
		boxed.extend(bare);
		assert_eq!(AlacConfig::from_extradata(&boxed), Some(c));
		let mut atom = 36u32.to_be_bytes().to_vec();
		atom.extend(b"alac");
		atom.extend(boxed);
		assert_eq!(AlacConfig::from_extradata(&atom), Some(c));
		assert_eq!(AlacConfig::from_extradata(&bare[..20]), None);
	}

	#[test]
	fn verbatim_frame() {
		// One mono 16-bit element, uncompressed, 3 samples, then END.
		let c = AlacConfig {
			frame_length: 3,
			bit_depth: 16,
			rice_history_mult: 40,
			rice_initial_history: 10,
			rice_limit: 14,
			channels: 1,
			max_run: 255,
			max_frame_bytes: 0,
			avg_bit_rate: 0,
			sample_rate: 8000,
		};
		let mut w = crate::codecs::bitstream::BitWriter::new();
		w.put(TYPE_SCE, 3);
		w.put(0, 4);
		w.put(0, 12);
		w.put(0, 1); // no size
		w.put(0, 2); // no extra bits
		w.put(1, 1); // not compressed
		for s in [1i16, -2, 300] {
			w.put(s as u16 as u32, 16);
		}
		w.put(TYPE_END, 3);
		let data = w.finish();
		let mut dec = AlacDecoderState::new(&c.to_bytes());
		let f = dec.decode_packet(&data).unwrap();
		let s: Vec<i16> = f.data.chunks(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
		assert_eq!(s, [1, -2, 300]);
		assert!(dec.decode_packet(&data[..2]).is_err());
	}
}
