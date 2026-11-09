//! FLAC frame decoder (FLAC format specification / RFC 9639 §9-§10;
//! FFmpeg: `libavcodec/flacdec.c`, `flacdsp_template.c`).
//!
//! One frame per call: frame header → one subframe per channel (constant,
//! verbatim, fixed predictor or LPC, each with Rice-coded residuals and
//! optional wasted bits) → inter-channel decorrelation → CRC-16 footer.
//!
//! Output is interleaved, in FFmpeg's sample formats: S16 for streams of
//! up to 16 bits, S32 above, with samples left-aligned (a 24-bit sample is
//! stored as `x << 8`), exactly as `ffmpeg -f s16le` / `-f s32le` writes
//! them.

use super::frame_header::{ChannelAssignment, FrameHeader};
use super::streaminfo::StreamInfo;
use crate::codecs::bitstream::BitReader;
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State,
};
use crate::message::{Message, Result};

/// Largest block size a FLAC frame can declare.
pub const MAX_BLOCK_SIZE: u32 = 65_535;
/// Largest supported sample depth (RFC 9639 allows 32 bits).
pub const MAX_BITS: u8 = 32;
/// Most channels a frame can carry.
pub const MAX_CHANNELS: usize = 8;

/// Why a frame could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invalid {
	Header,
	Subframe,
}

/// CRC-16, polynomial 0x8005 (the frame footer checksum).
pub fn crc16(data: &[u8]) -> u16 {
	static TABLE: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
	let table = TABLE.get_or_init(|| {
		let mut t = [0u16; 256];
		for (i, slot) in t.iter_mut().enumerate() {
			let mut c = (i as u16) << 8;
			for _ in 0..8 {
				c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
			}
			*slot = c;
		}
		t
	});
	data.iter().fold(0u16, |crc, &b| (crc << 8) ^ table[usize::from((crc >> 8) as u8 ^ b)])
}

/// One decoded frame: per-channel samples (right-aligned, `bits` deep).
#[derive(Debug, Clone)]
pub struct DecodedFrame {
	pub header: FrameHeader,
	pub sample_rate: u32,
	pub bits: u8,
	pub channels: Vec<Vec<i32>>,
	/// Bytes the frame occupied in the packet.
	pub len: usize,
	pub crc_ok: bool,
}

/// Reads a unary-coded value (count of zero bits before a one).
#[inline]
fn unary(bs: &mut BitReader) -> Option<u32> {
	let mut count = 0u32;
	loop {
		let window = bs.peek(32);
		if window != 0 {
			let zeros = window.leading_zeros();
			bs.skip(zeros as usize + 1);
			return count.checked_add(zeros);
		}
		if bs.bits_left() == 0 {
			return None;
		}
		bs.skip(32);
		count = count.checked_add(32).filter(|&c| c < (1 << 30))?;
	}
}

/// Reads `bits` (up to 33) as a signed value.
#[inline]
fn signed_wide(bs: &mut BitReader, bits: u32) -> i64 {
	if bits <= 32 {
		return i64::from(bs.signed(bits));
	}
	let high = i64::from(bs.signed(bits - 32));
	(high << 32) | i64::from(bs.bits(32))
}

/// Decodes the Rice-coded residual of one subframe into `out[order..]`.
fn decode_residual(bs: &mut BitReader, out: &mut [i64], order: usize) -> Option<()> {
	let block = out.len();
	let (param_bits, escape) = match bs.bits(2) {
		0 => (4, 15),
		1 => (5, 31),
		_ => return None,
	};
	let partition_order = bs.bits(4);
	let partitions = 1usize << partition_order;
	let per_partition = block >> partition_order;
	if per_partition << partition_order != block || per_partition < order {
		return None;
	}
	let mut at = order;
	for p in 0..partitions {
		let end = (p + 1) * per_partition;
		let k = bs.bits(param_bits);
		if k == escape {
			let n = bs.bits(5);
			for slot in &mut out[at..end] {
				*slot = if n == 0 { 0 } else { i64::from(bs.signed(n)) };
			}
		} else {
			for slot in &mut out[at..end] {
				let q = u64::from(unary(bs)?);
				let v = (q << k) | u64::from(bs.bits(k));
				*slot = (v >> 1) as i64 ^ -((v & 1) as i64);
			}
		}
		at = end;
	}
	(!bs.overrun()).then_some(())
}

/// Fixed predictor coefficients for orders 0-4 (§10.2.6).
const FIXED: [&[i64]; 5] = [&[], &[1], &[2, -1], &[3, -3, 1], &[4, -6, 4, -1]];

/// Restores samples from residuals with an integer predictor:
/// `x[i] = r[i] + (Σ c[j]·x[i-1-j]) >> shift`. Wrapping arithmetic keeps
/// hostile streams from overflowing (conforming ones never do).
fn predict(samples: &mut [i64], coefs: &[i64], shift: u32) {
	let order = coefs.len();
	for i in order..samples.len() {
		let mut sum: i64 = 0;
		for (j, &c) in coefs.iter().enumerate() {
			sum = sum.wrapping_add(c.wrapping_mul(samples[i - 1 - j]));
		}
		samples[i] = samples[i].wrapping_add(sum >> shift);
	}
}

/// Decodes one subframe of `bits` bits per sample into `out`.
fn decode_subframe(bs: &mut BitReader, out: &mut [i64], bits: u32) -> Option<()> {
	if bs.bit() {
		return None; // zero padding bit
	}
	let kind = bs.bits(6);
	let wasted = if bs.bit() { unary(bs)? + 1 } else { 0 };
	if wasted >= bits {
		return None;
	}
	let bits = bits - wasted;
	match kind {
		0 => {
			let v = signed_wide(bs, bits);
			out.fill(v);
		}
		1 => {
			for slot in out.iter_mut() {
				*slot = signed_wide(bs, bits);
			}
		}
		8..=12 => {
			let order = (kind - 8) as usize;
			if order > out.len() {
				return None;
			}
			for slot in &mut out[..order] {
				*slot = signed_wide(bs, bits);
			}
			decode_residual(bs, out, order)?;
			predict(out, FIXED[order], 0);
		}
		32..=63 => {
			let order = (kind - 31) as usize;
			if order > out.len() {
				return None;
			}
			for slot in &mut out[..order] {
				*slot = signed_wide(bs, bits);
			}
			let precision = bs.bits(4) + 1;
			if precision == 16 {
				return None;
			}
			let shift = bs.signed(5);
			if shift < 0 {
				return None;
			}
			let mut coefs = [0i64; 32];
			for c in &mut coefs[..order] {
				*c = i64::from(bs.signed(precision));
			}
			decode_residual(bs, out, order)?;
			predict(out, &coefs[..order], shift as u32);
		}
		_ => return None,
	}
	if bs.overrun() {
		return None;
	}
	if wasted > 0 {
		for v in out.iter_mut() {
			*v = v.wrapping_shl(wasted);
		}
	}
	Some(())
}

/// Decodes the frame at the start of `data`.
pub fn decode_frame(
	data: &[u8],
	info: Option<&StreamInfo>,
) -> std::result::Result<DecodedFrame, Invalid> {
	let header = FrameHeader::parse(data).ok_or(Invalid::Header)?;
	let bits = match header.bits_per_sample {
		0 => info.map(|i| i.bits_per_sample).ok_or(Invalid::Header)?,
		b => b,
	};
	let sample_rate = match header.sample_rate {
		0 => info.map(|i| i.sample_rate).ok_or(Invalid::Header)?,
		r => r,
	};
	if !(4..=MAX_BITS).contains(&bits) {
		return Err(Invalid::Header);
	}
	let block = header.block_size as usize;
	let channels = usize::from(header.channels.channels());
	let mut bs = BitReader::new(data);
	bs.skip(header.len * 8);

	let mut decoded: Vec<Vec<i64>> = Vec::with_capacity(channels);
	for ch in 0..channels {
		let side = match header.channels {
			ChannelAssignment::LeftSide | ChannelAssignment::MidSide => ch == 1,
			ChannelAssignment::RightSide => ch == 0,
			ChannelAssignment::Independent(_) => false,
		};
		let mut out = vec![0i64; block];
		decode_subframe(&mut bs, &mut out, u32::from(bits) + u32::from(side))
			.ok_or(Invalid::Subframe)?;
		decoded.push(out);
	}
	bs.align();
	let footer = bs.pos() / 8;
	let stored = bs.bits(16) as u16;
	if bs.overrun() {
		return Err(Invalid::Subframe);
	}
	let crc_ok = crc16(&data[..footer]) == stored;

	if channels == 2 {
		let (a, b) = decoded.split_at_mut(1);
		let (a, b) = (&mut a[0], &mut b[0]);
		match header.channels {
			ChannelAssignment::LeftSide => {
				for (l, s) in a.iter().zip(b.iter_mut()) {
					*s = l.wrapping_sub(*s);
				}
			}
			ChannelAssignment::RightSide => {
				for (s, r) in a.iter_mut().zip(b.iter()) {
					*s = s.wrapping_add(*r);
				}
			}
			ChannelAssignment::MidSide => {
				for (m, s) in a.iter_mut().zip(b.iter_mut()) {
					let mid = m.wrapping_shl(1) | (*s & 1);
					*m = mid.wrapping_add(*s) >> 1;
					*s = mid.wrapping_sub(*s) >> 1;
				}
			}
			ChannelAssignment::Independent(_) => {}
		}
	}
	let channels = decoded.into_iter().map(|c| c.into_iter().map(|v| v as i32).collect()).collect();
	Ok(DecodedFrame { header, sample_rate, bits, channels, len: footer + 2, crc_ok })
}

#[derive(Debug, Clone, Default)]
pub struct FlacDecoderState {
	pub info: Option<StreamInfo>,
	/// Frames replaced by silence because they could not be decoded.
	pub concealed_frames: u64,
	/// Frames whose CRC-16 did not match (decoded anyway, as FFmpeg does
	/// unless `-err_detect crccheck` is set).
	pub crc_errors: u64,
}

impl FlacDecoderState {
	pub fn new(extradata: &[u8]) -> Self {
		Self { info: StreamInfo::from_extradata(extradata), ..Default::default() }
	}

	/// Output sample format for a depth.
	pub fn format_for(bits: u8) -> SampleFormat {
		if bits <= 16 { SampleFormat::S16 } else { SampleFormat::S32 }
	}

	/// Decodes every frame in `data` into interleaved output frames.
	pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<AudioFrame>> {
		let mut frames = Vec::new();
		let mut rest = data;
		while rest.len() > 2 {
			match decode_frame(rest, self.info.as_ref()) {
				Ok(frame) => {
					self.crc_errors += u64::from(!frame.crc_ok);
					let len = frame.len.min(rest.len());
					frames.push(interleave(&frame));
					rest = &rest[len..];
				}
				Err(Invalid::Header) if frames.is_empty() => {
					return Err(Message::Codec("invalid flac frame header"));
				}
				Err(Invalid::Header) => break,
				Err(Invalid::Subframe) => {
					// The header parsed, so the frame's size is known: conceal
					// it with silence to keep timestamps continuous.
					let Some(header) = FrameHeader::parse(rest) else { break };
					self.concealed_frames += 1;
					let bits = match header.bits_per_sample {
						0 => self.info.map_or(16, |i| i.bits_per_sample),
						b => b,
					};
					let rate = match header.sample_rate {
						0 => self.info.map_or(44_100, |i| i.sample_rate),
						r => r,
					};
					let format = Self::format_for(bits);
					let channels = header.channels.channels();
					let bytes = header.block_size as usize * usize::from(channels) * format.bytes();
					frames.push(AudioFrame::new(
						vec![0; bytes],
						rate,
						Channels::from_count(channels),
						format,
						SampleLayout::Interleaved,
					));
					break;
				}
			}
		}
		Ok(frames)
	}
}

/// Interleaves a decoded frame into FFmpeg's left-aligned S16 / S32 layout.
pub fn interleave(frame: &DecodedFrame) -> AudioFrame {
	let format = FlacDecoderState::format_for(frame.bits);
	let channels = frame.channels.len();
	let block = frame.channels.first().map_or(0, Vec::len);
	let mut data = Vec::with_capacity(block * channels * format.bytes());
	if format == SampleFormat::S16 {
		let shift = 16 - u32::from(frame.bits);
		for i in 0..block {
			for ch in &frame.channels {
				data.extend_from_slice(&((ch[i] << shift) as i16).to_le_bytes());
			}
		}
	} else {
		let shift = 32 - u32::from(frame.bits);
		for i in 0..block {
			for ch in &frame.channels {
				data.extend_from_slice(&(ch[i].wrapping_shl(shift)).to_le_bytes());
			}
		}
	}
	AudioFrame::new(
		data,
		frame.sample_rate,
		Channels::from_count(channels as u8),
		format,
		SampleLayout::Interleaved,
	)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FlacDecoder;

impl FlacDecoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for FlacDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec = FlacDecoderState::new(&parameters.extradata);
		let bits = dec.info.map_or(parameters.bits_per_sample.unwrap_or(16), |i| i.bits_per_sample);
		State::Audio(AudioState {
			sample_rate: dec.info.map_or(parameters.sample_rate.unwrap_or(44_100), |i| i.sample_rate),
			channels: Channels::from_count(
				dec.info.map_or(parameters.channels.unwrap_or(2), |i| i.channels),
			),
			format: FlacDecoderState::format_for(bits),
			layout: SampleLayout::Interleaved,
			frame_size: dec.info.map_or(4096, |i| usize::from(i.max_block_size)),
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::Flac(Box::new(dec)),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("flac decoder requires an audio state"));
		};
		if !matches!(audio.private, CodecPrivate::Flac(_)) {
			audio.private =
				CodecPrivate::Flac(Box::new(FlacDecoderState::new(&audio.codec_state.extradata)));
		}
		let CodecPrivate::Flac(dec) = &mut audio.private else {
			return Err(Message::Codec("flac decoder state missing"));
		};
		let decoded = dec.decode_packet(&packet.data)?;
		let mut frames = Vec::with_capacity(decoded.len());
		let mut pts = packet.pts;
		for mut frame in decoded {
			frame.pts = pts;
			pts = pts.and_then(|t| {
				let step =
					core::Time::new(frame.nb_samples as u64, core::TimeBase::new(1, frame.sample_rate));
				step.rescale(t.base()).and_then(|s| t.checked_add(s))
			});
			audio.sample_rate = frame.sample_rate;
			audio.channels = frame.channels;
			audio.format = frame.format;
			audio.samples_produced += frame.nb_samples as u64;
			frames.push(Frame::Audio(frame));
		}
		audio.samples_consumed += packet.data.len() as u64;
		Ok(DecodeOut { value: frames, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		Ok(DecodeOut { value: Vec::new(), state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn crc16_check_value() {
		// CRC-16/UMTS ("BUYPASS") check value for "123456789".
		assert_eq!(crc16(b"123456789"), 0xFEE8);
	}

	#[test]
	fn rejects_garbage() {
		assert_eq!(decode_frame(&[0xFF, 0xF8, 0, 0], None).err(), Some(Invalid::Header));
		assert!(FlacDecoderState::default().decode_packet(b"nonsense").is_err());
	}
}
