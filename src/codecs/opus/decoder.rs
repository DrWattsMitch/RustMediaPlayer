//! Opus decoder: packet parsing, mode switching and multistream mapping
//! (RFC 6716 §3-§4.5, RFC 7845 §5.1.1; libopus `src/opus_decoder.c`,
//! `src/opus.c`, `src/opus_multistream_decoder.c`).
//!
//! Each Opus stream is decoded by a [`StreamDecoder`] (SILK, CELT or both);
//! packets with several streams (mapping families 1 and 255) are split by
//! self-delimited framing and mixed into the output channels by the
//! `OpusHead` channel mapping. Output is 48 kHz interleaved `F32`, in
//! FFmpeg's channel order (family-1 surround is reordered from Vorbis
//! order, as FFmpeg's decoders do).
//!
//! The decoder returns every decoded sample: dropping the pre-skip (the
//! stream's `encoder_delay`) is the caller's job, as for AAC and MP3.

use super::celt::CeltDecoder;
use super::header::{OpusHead, frame_samples};
use super::range::RangeDecoder;
use super::silk::SilkDecoder;
use super::tables::WINDOW120;
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State, Time, TimeBase,
};
use crate::message::{Message, Result};

/// Output sample rate: Opus always decodes at 48 kHz here (FFmpeg's choice).
pub const SAMPLE_RATE: u32 = 48_000;
/// Longest packet: 120 ms.
const MAX_FRAME_SAMPLES: usize = 5760;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
	SilkOnly,
	Hybrid,
	CeltOnly,
}

/// Audio bandwidth from the TOC (RFC 6716 Table 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bandwidth {
	Narrow,
	Medium,
	Wide,
	SuperWide,
	Full,
}

pub fn toc_mode(toc: u8) -> Mode {
	if toc & 0x80 != 0 {
		Mode::CeltOnly
	} else if toc & 0x60 == 0x60 {
		Mode::Hybrid
	} else {
		Mode::SilkOnly
	}
}

pub fn toc_bandwidth(toc: u8) -> Bandwidth {
	if toc & 0x80 != 0 {
		match (toc >> 5) & 3 {
			0 => Bandwidth::Narrow,
			1 => Bandwidth::Wide,
			2 => Bandwidth::SuperWide,
			_ => Bandwidth::Full,
		}
	} else if toc & 0x60 == 0x60 {
		if toc & 0x10 != 0 { Bandwidth::Full } else { Bandwidth::SuperWide }
	} else {
		match (toc >> 5) & 3 {
			0 => Bandwidth::Narrow,
			1 => Bandwidth::Medium,
			_ => Bandwidth::Wide,
		}
	}
}

/// Frames of one packet: `(toc, frame byte ranges, bytes used)`.
pub struct ParsedPacket {
	pub toc: u8,
	pub frames: Vec<(usize, usize)>,
	/// Bytes the packet occupies (for self-delimited packets inside a
	/// multistream packet).
	pub len: usize,
}

fn parse_size(data: &[u8]) -> Option<(usize, usize)> {
	let first = usize::from(*data.first()?);
	if first < 252 { Some((first, 1)) } else { Some((4 * usize::from(*data.get(1)?) + first, 2)) }
}

/// `opus_packet_parse_impl`: splits a packet into frames.
pub fn parse_packet(data: &[u8], self_delimited: bool) -> Option<ParsedPacket> {
	let toc = *data.first()?;
	let framesize = frame_samples(toc) as usize;
	let mut at = 1usize;
	let mut len = data.len() - 1;
	let mut sizes: Vec<usize> = Vec::new();
	let mut last_size = len;
	let mut cbr = false;
	let mut pad = 0usize;
	let count;
	match toc & 3 {
		0 => count = 1,
		1 => {
			count = 2;
			cbr = true;
			if !self_delimited {
				if len & 1 != 0 {
					return None;
				}
				last_size = len / 2;
				sizes.push(last_size);
			}
		}
		2 => {
			count = 2;
			let (size, bytes) = parse_size(&data[at..])?;
			len = len.checked_sub(bytes)?;
			if size > len {
				return None;
			}
			at += bytes;
			sizes.push(size);
			last_size = len - size;
		}
		_ => {
			if len < 1 {
				return None;
			}
			let ch = data[at];
			at += 1;
			len -= 1;
			count = usize::from(ch & 0x3F);
			if count == 0 || framesize * count > MAX_FRAME_SAMPLES {
				return None;
			}
			if ch & 0x40 != 0 {
				loop {
					let p = usize::from(*data.get(at)?);
					at += 1;
					len = len.checked_sub(1)?;
					let tmp = if p == 255 { 254 } else { p };
					len = len.checked_sub(tmp)?;
					pad += tmp;
					if p != 255 {
						break;
					}
				}
			}
			cbr = ch & 0x80 == 0;
			if !cbr {
				last_size = len;
				for _ in 0..count - 1 {
					let (size, bytes) = parse_size(data.get(at..)?)?;
					len = len.checked_sub(bytes)?;
					if size > len {
						return None;
					}
					at += bytes;
					sizes.push(size);
					last_size = last_size.checked_sub(bytes + size)?;
				}
			} else if !self_delimited {
				last_size = len / count;
				if last_size * count != len {
					return None;
				}
				sizes.extend(std::iter::repeat_n(last_size, count - 1));
			}
		}
	}
	if self_delimited {
		let (size, bytes) = parse_size(data.get(at..)?)?;
		len = len.checked_sub(bytes)?;
		if size > len {
			return None;
		}
		at += bytes;
		if cbr {
			if size * count > len {
				return None;
			}
			sizes = vec![size; count];
		} else {
			if bytes + size > last_size {
				return None;
			}
			sizes.push(size);
		}
	} else {
		if last_size > 1275 {
			return None;
		}
		sizes.push(last_size);
	}
	let mut frames = Vec::with_capacity(count);
	for &size in &sizes {
		let end = at.checked_add(size)?;
		if end > data.len() {
			return None;
		}
		frames.push((at, end));
		at = end;
	}
	Some(ParsedPacket { toc, frames, len: at + pad })
}

fn smooth_fade(in1: &[f32], in2: &[f32], out: &mut [f32], overlap: usize, channels: usize) {
	for c in 0..channels {
		for i in 0..overlap {
			let w = WINDOW120[i] * WINDOW120[i];
			out[i * channels + c] = w * in2[i * channels + c] + (1.0 - w) * in1[i * channels + c];
		}
	}
}

/// One Opus stream (mono or stereo): `OpusDecoder`.
#[derive(Debug, Clone)]
pub struct StreamDecoder {
	pub channels: usize,
	celt: CeltDecoder,
	silk: SilkDecoder,
	mode: Option<Mode>,
	prev_mode: Option<Mode>,
	prev_redundancy: bool,
	frame_size: usize,
	bandwidth: Bandwidth,
	stream_channels: usize,
	/// Final range of the last frame (conformance testing).
	pub range_final: u32,
}

impl StreamDecoder {
	pub fn new(channels: usize) -> Self {
		Self {
			channels,
			celt: CeltDecoder::new(channels),
			silk: SilkDecoder::new(channels),
			mode: None,
			prev_mode: None,
			prev_redundancy: false,
			frame_size: 960,
			bandwidth: Bandwidth::Full,
			stream_channels: channels,
			range_final: 0,
		}
	}

	pub fn reset(&mut self) {
		*self = Self::new(self.channels);
	}

	/// Concealment for a missing frame of `samples` (`opus_decode_frame`
	/// with no data): SILK or CELT PLC depending on the last mode.
	fn decode_lost(&mut self, pcm: &mut [f32], samples: usize) {
		let channels = self.channels;
		let mode = if self.prev_redundancy { Some(Mode::CeltOnly) } else { self.prev_mode };
		pcm[..samples * channels].fill(0.0);
		match mode {
			Some(Mode::CeltOnly) => {
				let mut done = 0;
				while done < samples {
					let n =
						[960usize, 480, 240, 120].into_iter().find(|&n| n <= samples - done).unwrap_or(120);
					let mut tmp = vec![0f32; n * channels];
					self.celt.decode_lost(n.trailing_zeros().saturating_sub(3) as usize, &mut tmp);
					let take = n.min(samples - done);
					pcm[done * channels..(done + take) * channels].copy_from_slice(&tmp[..take * channels]);
					done += take;
				}
			}
			Some(_) => {
				// SILK concealment is not implemented: silence.
			}
			None => {}
		}
	}

	/// `opus_decode_frame` for one frame with data.
	#[allow(clippy::too_many_lines)]
	fn decode_frame(
		&mut self,
		data: &[u8],
		pcm: &mut [f32],
	) -> std::result::Result<usize, &'static str> {
		let channels = self.channels;
		let f20 = 960usize;
		let f10 = 480usize;
		let f5 = 240usize;
		let f2_5 = 120usize;
		let audiosize = self.frame_size;
		let mode = self.mode.ok_or("opus decoder has no mode")?;
		let bandwidth = self.bandwidth;
		let mut len = data.len();
		if len <= 1 {
			self.decode_lost(pcm, audiosize);
			return Ok(audiosize);
		}
		let mut dec = RangeDecoder::new(data);

		let mut transition = false;
		let mut pcm_transition: Vec<f32> = Vec::new();
		if let Some(prev) = self.prev_mode {
			if (mode == Mode::CeltOnly && prev != Mode::CeltOnly && !self.prev_redundancy)
				|| (mode != Mode::CeltOnly && prev == Mode::CeltOnly)
			{
				transition = true;
				if mode == Mode::CeltOnly {
					pcm_transition = vec![0f32; f5 * channels];
					let n = f5.min(audiosize);
					self.decode_lost(&mut pcm_transition, n);
				}
			}
		}
		let frame_size = audiosize;

		// SILK.
		let mut pcm_silk: Vec<f32> = Vec::new();
		if mode != Mode::CeltOnly {
			pcm_silk = vec![0f32; f10.max(frame_size) * channels];
			if self.prev_mode == Some(Mode::CeltOnly) {
				self.silk.reset();
			}
			let internal_rate = match (mode, bandwidth) {
				(Mode::SilkOnly, Bandwidth::Narrow) => 8000,
				(Mode::SilkOnly, Bandwidth::Medium) => 12000,
				_ => 16000,
			};
			let payload_ms = (1000 * audiosize / 48_000).max(10);
			let mut decoded = 0usize;
			while decoded < frame_size {
				let first = decoded == 0;
				let out = &mut pcm_silk[decoded * channels..];
				let n = self.silk.decode(
					&mut dec,
					self.stream_channels,
					internal_rate,
					payload_ms,
					first,
					out,
				)?;
				if n == 0 {
					return Err("silk decoder produced no samples");
				}
				decoded += n;
			}
		}

		let mut start_band = 0;
		let mut redundancy = false;
		let mut redundancy_bytes = 0usize;
		let mut celt_to_silk = false;
		if mode != Mode::CeltOnly
			&& dec.tell() + 17 + 20 * i32::from(mode == Mode::Hybrid) <= 8 * len as i32
		{
			redundancy = if mode == Mode::Hybrid { dec.bit_logp(12) } else { true };
			if redundancy {
				celt_to_silk = dec.bit_logp(1);
				redundancy_bytes = if mode == Mode::Hybrid {
					dec.uint(256) as usize + 2
				} else {
					len.saturating_sub(((dec.tell() + 7) >> 3) as usize)
				};
				if redundancy_bytes > len || ((len - redundancy_bytes) * 8) < dec.tell() as usize {
					len = 0;
					redundancy_bytes = 0;
					redundancy = false;
				} else {
					len -= redundancy_bytes;
				}
				dec.shrink(redundancy_bytes);
			}
		}
		if mode != Mode::CeltOnly {
			start_band = 17;
		}
		if redundancy {
			transition = false;
		}
		if transition && mode != Mode::CeltOnly {
			pcm_transition = vec![0f32; f5 * channels];
			let n = f5.min(audiosize);
			self.decode_lost(&mut pcm_transition, n);
		}

		let end_band = match bandwidth {
			Bandwidth::Narrow => 13,
			Bandwidth::Medium | Bandwidth::Wide => 17,
			Bandwidth::SuperWide => 19,
			Bandwidth::Full => 21,
		};
		self.celt.end = end_band;
		self.celt.stream_channels = self.stream_channels;

		let redundant_start = len;
		let redundant = &data[redundant_start..(redundant_start + redundancy_bytes).min(data.len())];
		let mut redundant_audio = vec![0f32; f5 * channels];
		let mut redundant_rng = 0u32;
		if redundancy && celt_to_silk {
			self.celt.start = 0;
			let _ = self.celt.decode(redundant, &mut redundant_audio, f5);
			redundant_rng = self.celt.rng;
		}
		self.celt.start = start_band;

		if mode != Mode::SilkOnly {
			let celt_frame = f20.min(frame_size);
			if Some(mode) != self.prev_mode && self.prev_mode.is_some() && !self.prev_redundancy {
				self.celt.reset();
			}
			self.celt.decode_with(&mut dec, len, pcm, celt_frame)?;
		} else {
			pcm[..frame_size * channels].fill(0.0);
			if self.prev_mode == Some(Mode::Hybrid)
				&& !(redundancy && celt_to_silk && self.prev_redundancy)
			{
				self.celt.start = 0;
				let mut tmp = vec![0f32; f2_5 * channels];
				let _ = self.celt.decode(&[0xFF, 0xFF], &mut tmp, f2_5);
				pcm[..f2_5 * channels].copy_from_slice(&tmp);
			}
		}
		if mode != Mode::CeltOnly {
			for i in 0..frame_size * channels {
				pcm[i] += pcm_silk[i];
			}
		}

		if redundancy && !celt_to_silk {
			self.celt.reset();
			self.celt.start = 0;
			let _ = self.celt.decode(redundant, &mut redundant_audio, f5);
			redundant_rng = self.celt.rng;
			let at = channels * (frame_size - f2_5);
			let head = pcm[at..at + f2_5 * channels].to_vec();
			smooth_fade(&head, &redundant_audio[channels * f2_5..], &mut pcm[at..], f2_5, channels);
		}
		if redundancy
			&& celt_to_silk
			&& (self.prev_mode != Some(Mode::SilkOnly) || self.prev_redundancy)
		{
			pcm[..f2_5 * channels].copy_from_slice(&redundant_audio[..f2_5 * channels]);
			let tail = pcm[channels * f2_5..channels * 2 * f2_5].to_vec();
			smooth_fade(
				&redundant_audio[channels * f2_5..],
				&tail,
				&mut pcm[channels * f2_5..],
				f2_5,
				channels,
			);
		}
		if transition && !pcm_transition.is_empty() {
			if audiosize >= f5 {
				pcm[..channels * f2_5].copy_from_slice(&pcm_transition[..channels * f2_5]);
				let cur = pcm[channels * f2_5..channels * 2 * f2_5].to_vec();
				smooth_fade(
					&pcm_transition[channels * f2_5..],
					&cur,
					&mut pcm[channels * f2_5..],
					f2_5,
					channels,
				);
			} else {
				let cur = pcm[..channels * f2_5].to_vec();
				smooth_fade(&pcm_transition, &cur, pcm, f2_5, channels);
			}
		}
		self.range_final = if len <= 1 { 0 } else { dec.range() ^ redundant_rng };
		self.prev_mode = Some(mode);
		self.prev_redundancy = redundancy && !celt_to_silk;
		Ok(audiosize)
	}

	/// Decodes one (non-self-delimited unless `self_delimited`) packet.
	/// Returns the samples per channel written to `pcm`.
	pub fn decode_packet(
		&mut self,
		data: &[u8],
		self_delimited: bool,
		pcm: &mut [f32],
	) -> std::result::Result<(usize, usize), &'static str> {
		let parsed = parse_packet(data, self_delimited).ok_or("invalid opus packet")?;
		let toc = parsed.toc;
		let frame = frame_samples(toc) as usize;
		self.mode = Some(toc_mode(toc));
		self.bandwidth = toc_bandwidth(toc);
		self.frame_size = frame;
		self.stream_channels = if toc & 0x4 != 0 { 2 } else { 1 };
		let mut written = 0;
		for &(a, b) in &parsed.frames {
			let out = &mut pcm[written * self.channels..];
			let n = self.decode_frame(&data[a..b], out)?;
			written += n;
		}
		Ok((written, parsed.len))
	}
}

/// Full Opus decoder for one `OpusHead` configuration.
#[derive(Debug, Clone)]
pub struct OpusDecoderState {
	pub head: OpusHead,
	streams: Vec<StreamDecoder>,
	/// FFmpeg output channel → decoded channel (255 = silence).
	map: Vec<u8>,
	gain: f32,
	/// Packets that failed to decode and were replaced by silence.
	pub concealed_frames: u64,
}

/// Vorbis → FFmpeg channel order for mapping family 1
/// (`ff_vorbis_channel_layout_offsets`).
const VORBIS_ORDER: [&[usize]; 8] = [
	&[0],
	&[0, 1],
	&[0, 2, 1],
	&[0, 1, 2, 3],
	&[0, 2, 1, 3, 4],
	&[0, 2, 1, 5, 3, 4],
	&[0, 2, 1, 6, 5, 3, 4],
	&[0, 2, 1, 7, 5, 6, 3, 4],
];

impl OpusDecoderState {
	pub fn new(head: OpusHead) -> Self {
		let m = &head.mapping;
		let streams = (0..usize::from(m.streams))
			.map(|s| StreamDecoder::new(if s < usize::from(m.coupled) { 2 } else { 1 }))
			.collect();
		let channels = usize::from(head.channels);
		let map: Vec<u8> = if m.family == 1 && channels <= 8 {
			VORBIS_ORDER[channels - 1].iter().map(|&i| m.mapping[i]).collect()
		} else {
			m.mapping.clone()
		};
		let gain = if head.output_gain != 0 {
			10f32.powf(f32::from(head.output_gain) / (20.0 * 256.0))
		} else {
			1.0
		};
		Self { head, streams, map, gain, concealed_frames: 0 }
	}

	pub fn from_extradata(extradata: &[u8], channels: u8) -> Self {
		Self::new(
			OpusHead::parse(extradata)
				.unwrap_or_else(|| OpusHead::new(channels.clamp(1, 2), 0, SAMPLE_RATE)),
		)
	}

	pub fn channels(&self) -> usize {
		usize::from(self.head.channels)
	}

	pub fn reset(&mut self) {
		for s in &mut self.streams {
			s.reset();
		}
	}

	/// Decodes a packet into interleaved samples (`opus_multistream_decode_float`).
	pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<f32>> {
		let channels = self.channels();
		let mut at = 0usize;
		let count = self.streams.len();
		let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(count);
		let mut samples = None;
		for (s, stream) in self.streams.iter_mut().enumerate() {
			let last = s + 1 == count;
			let mut pcm = vec![0f32; MAX_FRAME_SAMPLES * stream.channels];
			let (n, used) = stream
				.decode_packet(data.get(at..).unwrap_or(&[]), !last, &mut pcm)
				.map_err(Message::Codec)?;
			if samples.is_some_and(|m| m != n) {
				return Err(Message::Codec("opus streams disagree on frame duration"));
			}
			samples = Some(n);
			pcm.truncate(n * stream.channels);
			decoded.push(pcm);
			at += used;
		}
		let n = samples.unwrap_or(0);
		let coupled = usize::from(self.head.mapping.coupled);
		let mut out = vec![0f32; n * channels];
		for (c, &m) in self.map.iter().enumerate() {
			if m == 255 {
				continue;
			}
			let m = usize::from(m);
			let (stream, sub) =
				if m < 2 * coupled { (m / 2, m % 2) } else { (coupled + m - 2 * coupled, 0) };
			let Some(src) = decoded.get(stream) else { continue };
			let stride = self.streams[stream].channels;
			for i in 0..n {
				out[i * channels + c] = src[i * stride + sub] * self.gain;
			}
		}
		Ok(out)
	}
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OpusDecoder;

impl OpusDecoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for OpusDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec =
			OpusDecoderState::from_extradata(&parameters.extradata, parameters.channels.unwrap_or(2));
		State::Audio(AudioState {
			sample_rate: SAMPLE_RATE,
			channels: Channels::from_count(dec.head.channels),
			format: SampleFormat::F32,
			layout: SampleLayout::Interleaved,
			frame_size: 960,
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::Opus(Box::new(dec)),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("opus decoder requires an audio state"));
		};
		if !matches!(audio.private, CodecPrivate::Opus(_)) {
			let dec =
				OpusDecoderState::from_extradata(&audio.codec_state.extradata, audio.channels.count());
			audio.private = CodecPrivate::Opus(Box::new(dec));
		}
		let CodecPrivate::Opus(dec) = &mut audio.private else {
			return Err(Message::Codec("opus decoder state missing"));
		};
		let channels = dec.channels();
		let samples = match dec.decode_packet(&packet.data) {
			Ok(s) => s,
			Err(_) => {
				// Conceal with silence of the packet's nominal duration.
				dec.concealed_frames += 1;
				let n = super::header::packet_samples(&packet.data).unwrap_or(960) as usize;
				dec.reset();
				vec![0f32; n * channels]
			}
		};
		let data: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
		let mut frame = AudioFrame::new(
			data,
			SAMPLE_RATE,
			Channels::from_count(channels as u8),
			SampleFormat::F32,
			SampleLayout::Interleaved,
		);
		frame.pts = packet
			.pts
			.and_then(|t| Time::new(t.ticks(), t.base()).rescale(TimeBase::new(1, SAMPLE_RATE)));
		audio.samples_produced += frame.nb_samples as u64;
		audio.samples_consumed += packet.data.len() as u64;
		Ok(DecodeOut { value: vec![Frame::Audio(frame)], state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		Ok(DecodeOut { value: Vec::new(), state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_frame_packings() {
		// Code 0: one frame.
		let p = parse_packet(&[0xFC, 1, 2, 3], false).unwrap();
		assert_eq!(p.frames, vec![(1, 4)]);
		// Code 1: two equal frames.
		let p = parse_packet(&[0xFD, 1, 2, 3, 4], false).unwrap();
		assert_eq!(p.frames, vec![(1, 3), (3, 5)]);
		assert!(parse_packet(&[0xFD, 1, 2, 3], false).is_none());
		// Code 2: sizes 1 and 2.
		let p = parse_packet(&[0xFE, 1, 9, 8, 7], false).unwrap();
		assert_eq!(p.frames, vec![(2, 3), (3, 5)]);
		// Code 3, CBR, 3 frames with 2 bytes of padding.
		let p = parse_packet(&[0xFF, 0x43, 2, 1, 1, 2, 2, 3, 3, 0, 0], false).unwrap();
		assert_eq!(p.frames, vec![(3, 5), (5, 7), (7, 9)]);
		// Self-delimited code 0: explicit size.
		let p = parse_packet(&[0xFC, 2, 5, 6, 0xFC, 7], true).unwrap();
		assert_eq!((p.frames, p.len), (vec![(2, 4)], 4));
		assert!(parse_packet(&[], false).is_none());
	}
}
