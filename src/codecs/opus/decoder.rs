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
use super::silk::{SilkDecoder, SilkMode};
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
	/// Mode, bandwidth, frame size and coded channels of the current packet.
	mode: Option<Mode>,
	bandwidth: Bandwidth,
	frame_size: usize,
	stream_channels: usize,
	prev_mode: Option<Mode>,
	prev_redundancy: bool,
	/// SILK's channel count and internal rate from the last packet, reused
	/// when concealing (libopus' `DecControl`).
	silk_channels: usize,
	silk_rate: u32,
	/// Samples produced by the last decode or concealment call.
	pub last_packet_duration: usize,
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
			bandwidth: Bandwidth::Full,
			frame_size: 960,
			stream_channels: channels,
			prev_mode: None,
			prev_redundancy: false,
			silk_channels: channels,
			silk_rate: 16_000,
			last_packet_duration: 0,
			range_final: 0,
		}
	}

	/// `OPUS_RESET_STATE`.
	pub fn reset(&mut self) {
		*self = Self::new(self.channels);
	}

	/// `opus_decode_frame`: decodes one frame of `data` into `pcm`, or with
	/// `None` (or a 0/1-byte DTX payload) conceals up to `frame_size`
	/// samples in the last mode. With `decode_fec`, `data` is the packet
	/// after a lost one and its SILK LBRR copy of that packet is decoded.
	/// Returns the samples per channel written.
	#[allow(clippy::too_many_lines)]
	fn decode_frame(
		&mut self,
		data: Option<&[u8]>,
		pcm: &mut [f32],
		frame_size: usize,
		decode_fec: bool,
	) -> std::result::Result<usize, &'static str> {
		let channels = self.channels;
		let f20 = 960usize;
		let f10 = 480usize;
		let f5 = 240usize;
		let f2_5 = 120usize;
		if frame_size < f2_5 {
			return Err("opus output buffer too small");
		}
		let mut frame_size = frame_size.min(MAX_FRAME_SAMPLES);
		// Payloads of 0 or 1 byte (DTX) trigger the PLC, for no more than
		// the TOC's duration.
		let data = data.filter(|d| d.len() > 1);
		if data.is_none() {
			frame_size = frame_size.min(self.frame_size);
		}
		let mut audiosize;
		let mode;
		let bandwidth;
		if data.is_some() {
			audiosize = self.frame_size;
			mode = self.mode.ok_or("opus decoder has no mode")?;
			bandwidth = Some(self.bandwidth);
		} else {
			audiosize = frame_size;
			// Conceal in the last mode (CELT if it ended with CELT redundancy).
			let last = if self.prev_redundancy { Some(Mode::CeltOnly) } else { self.prev_mode };
			let Some(last) = last else {
				// Nothing decoded yet: all we can do is silence.
				pcm[..audiosize * channels].fill(0.0);
				return Ok(audiosize);
			};
			mode = last;
			bandwidth = None;
			// The PLC only runs on 2.5, 5, 10 and 20 ms.
			if audiosize > f20 {
				let mut done = 0;
				while done < audiosize {
					let n = self.decode_frame(
						None,
						&mut pcm[done * channels..],
						(audiosize - done).min(f20),
						false,
					)?;
					if n == 0 {
						break;
					}
					done += n;
				}
				return Ok(frame_size);
			} else if audiosize < f20 {
				if audiosize > f10 {
					audiosize = f10;
				} else if mode != Mode::SilkOnly && audiosize > f5 && audiosize < f10 {
					audiosize = f5;
				}
			}
		}
		let empty: &[u8] = &[];
		let payload = data.unwrap_or(empty);
		let mut len = payload.len();
		let mut dec = RangeDecoder::new(payload);

		// Mode transitions cross-fade from a 5 ms concealment of the old mode.
		let mut transition = false;
		let mut pcm_transition: Vec<f32> = Vec::new();
		if let (Some(_), Some(prev)) = (data, self.prev_mode) {
			if (mode == Mode::CeltOnly && prev != Mode::CeltOnly && !self.prev_redundancy)
				|| (mode != Mode::CeltOnly && prev == Mode::CeltOnly)
			{
				transition = true;
				if mode == Mode::CeltOnly {
					pcm_transition = vec![0f32; f5 * channels];
					// A failed concealment only weakens the cross-fade.
					let _ = self.decode_frame(None, &mut pcm_transition, f5.min(audiosize), false);
				}
			}
		}
		if audiosize > frame_size {
			return Err("opus output buffer too small");
		}
		frame_size = audiosize;

		// SILK.
		let mut pcm_silk: Vec<f32> = Vec::new();
		if mode != Mode::CeltOnly {
			pcm_silk = vec![0f32; f10.max(frame_size) * channels];
			if self.prev_mode == Some(Mode::CeltOnly) {
				self.silk.reset();
			}
			// SILK cannot conceal less than 10 ms.
			let payload_ms = (1000 * audiosize / 48_000).max(10);
			if data.is_some() {
				self.silk_channels = self.stream_channels;
				self.silk_rate = match (mode, bandwidth) {
					(Mode::SilkOnly, Some(Bandwidth::Narrow)) => 8000,
					(Mode::SilkOnly, Some(Bandwidth::Medium)) => 12000,
					_ => 16000,
				};
			}
			let silk_mode = match (data, decode_fec) {
				(None, _) => SilkMode::Lost,
				(Some(_), true) => SilkMode::Lbrr,
				(Some(_), false) => SilkMode::Normal,
			};
			let mut decoded = 0usize;
			while decoded < frame_size {
				let first = decoded == 0;
				let out = &mut pcm_silk[decoded * channels..];
				match self.silk.decode(
					&mut dec,
					silk_mode,
					self.silk_channels,
					self.silk_rate,
					payload_ms,
					first,
					out,
				) {
					Ok(0) => return Err("silk decoder produced no samples"),
					Ok(n) => decoded += n,
					// A concealment failure is not fatal: silence instead.
					Err(_) if silk_mode != SilkMode::Normal => {
						let end = ((decoded + frame_size) * channels).min(pcm_silk.len());
						pcm_silk[decoded * channels..end].fill(0.0);
						decoded += frame_size;
					}
					Err(e) => return Err(e),
				}
			}
		}

		// Redundant CELT frame for SILK/hybrid ↔ CELT switches.
		let mut start_band = 0;
		let mut redundancy = false;
		let mut redundancy_bytes = 0usize;
		let mut celt_to_silk = false;
		if !decode_fec
			&& mode != Mode::CeltOnly
			&& data.is_some()
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
			let _ = self.decode_frame(None, &mut pcm_transition, f5.min(audiosize), false);
		}

		if let Some(bandwidth) = bandwidth {
			self.celt.end = match bandwidth {
				Bandwidth::Narrow => 13,
				Bandwidth::Medium | Bandwidth::Wide => 17,
				Bandwidth::SuperWide => 19,
				Bandwidth::Full => 21,
			};
		}
		self.celt.stream_channels = self.stream_channels;

		let redundant = payload.get(len..len + redundancy_bytes).unwrap_or(empty);
		let mut redundant_audio = vec![0f32; f5 * channels];
		let mut redundant_rng = 0u32;
		if redundancy && celt_to_silk {
			// Decoded even when stale (its first frame may have been lost):
			// the final range is still needed.
			self.celt.start = 0;
			let _ = self.celt.decode(redundant, &mut redundant_audio, f5);
			redundant_rng = self.celt.rng;
		}
		// Must come after the transition PLC.
		self.celt.start = start_band;

		if mode != Mode::SilkOnly {
			let celt_frame = f20.min(frame_size);
			// Discard any previous CELT state.
			if Some(mode) != self.prev_mode && self.prev_mode.is_some() && !self.prev_redundancy {
				self.celt.reset();
			}
			// A zero length makes CELT conceal (lost packets and FEC).
			let celt_len = if decode_fec { 0 } else { len };
			self.celt.decode_with(&mut dec, celt_len, pcm, celt_frame)?;
		} else {
			pcm[..frame_size * channels].fill(0.0);
			// Hybrid → SILK: let the CELT MDCT fade out over a silence frame.
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
		// Ignored when the previous frame did not use CELT (the first
		// redundancy frame of a switch from SILK may have been lost).
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
				// Too short for a clean transition; fade anyway.
				let cur = pcm[..channels * f2_5].to_vec();
				smooth_fade(&pcm_transition, &cur, pcm, f2_5, channels);
			}
		}
		self.range_final = if len <= 1 { 0 } else { dec.range() ^ redundant_rng };
		self.prev_mode = Some(mode);
		self.prev_redundancy = redundancy && !celt_to_silk;
		Ok(audiosize)
	}

	/// `opus_decode_native`. With `None`, conceals `frame_size` samples
	/// (a multiple of 2.5 ms). With `decode_fec`, recovers `frame_size`
	/// samples lost before `data`: concealment for all but the last frame's
	/// worth, then that frame from `data`'s in-band FEC (concealment when
	/// it has none). Otherwise decodes `data` (self-delimited when
	/// `self_delimited`). Returns the samples per channel written to `pcm`
	/// and the bytes of `data` used.
	pub fn decode_native(
		&mut self,
		data: Option<&[u8]>,
		pcm: &mut [f32],
		frame_size: usize,
		decode_fec: bool,
		self_delimited: bool,
	) -> std::result::Result<(usize, usize), &'static str> {
		let channels = self.channels;
		let plc = data.is_none_or(<[u8]>::is_empty);
		if (decode_fec || plc) && !frame_size.is_multiple_of(PLC_STEP) {
			return Err("opus concealment must be a multiple of 2.5 ms");
		}
		if pcm.len() < frame_size * channels {
			return Err("opus output buffer too small");
		}
		let data = match data {
			Some(d) if !d.is_empty() => d,
			_ => {
				let mut count = 0;
				while count < frame_size {
					let n =
						self.decode_frame(None, &mut pcm[count * channels..], frame_size - count, false)?;
					if n == 0 {
						return Err("opus concealment produced no samples");
					}
					count += n;
				}
				self.last_packet_duration = count;
				return Ok((count, 0));
			}
		};
		let parsed = parse_packet(data, self_delimited).ok_or("invalid opus packet")?;
		let toc = parsed.toc;
		let packet_mode = toc_mode(toc);
		let packet_frame = frame_samples(toc) as usize;
		let set_packet_state = |st: &mut Self| {
			st.mode = Some(packet_mode);
			st.bandwidth = toc_bandwidth(toc);
			st.frame_size = packet_frame;
			st.stream_channels = if toc & 0x4 != 0 { 2 } else { 1 };
		};

		if decode_fec {
			// No FEC can be present: plain concealment.
			if frame_size < packet_frame
				|| packet_mode == Mode::CeltOnly
				|| self.mode == Some(Mode::CeltOnly)
			{
				let (n, _) = self.decode_native(None, pcm, frame_size, false, false)?;
				return Ok((n, parsed.len));
			}
			// Conceal everything except the frame the FEC may cover.
			let duration = self.last_packet_duration;
			if frame_size > packet_frame {
				if let Err(e) = self.decode_native(None, pcm, frame_size - packet_frame, false, false) {
					self.last_packet_duration = duration;
					return Err(e);
				}
			}
			set_packet_state(self);
			let (a, b) = parsed.frames[0];
			let out = &mut pcm[(frame_size - packet_frame) * channels..];
			self.decode_frame(Some(&data[a..b]), out, packet_frame, true)?;
			self.last_packet_duration = frame_size;
			return Ok((frame_size, parsed.len));
		}

		if parsed.frames.len() * packet_frame > frame_size {
			return Err("opus output buffer too small");
		}
		// Only now that the packet is known to be valid.
		set_packet_state(self);
		let mut written = 0;
		for &(a, b) in &parsed.frames {
			let out = &mut pcm[written * channels..];
			written += self.decode_frame(Some(&data[a..b]), out, frame_size - written, false)?;
		}
		self.last_packet_duration = written;
		Ok((written, parsed.len))
	}
}

/// Full Opus decoder for one `OpusHead` configuration.
///
/// # Packet loss
///
/// Losses are recovered as libopus does: [`conceal`](Self::conceal) runs the
/// packet-loss concealment (`opus_decode` with no data) and
/// [`decode_fec`](Self::decode_fec) additionally recovers the last lost
/// frame from the next packet's in-band FEC. Through [`core::Decoder`] this
/// happens automatically: a packet that cannot be decoded (or is empty) is
/// concealed for its duration, and a gap in the packet timestamps (a
/// dropped packet or Ogg page) is concealed, with FEC, before the next
/// packet is decoded. The concealment continues the signal for the first
/// lost frames and then fades into comfort noise at the level of the
/// background, so damaged streams never drop to digital silence.
#[derive(Debug, Clone)]
pub struct OpusDecoderState {
	pub head: OpusHead,
	streams: Vec<StreamDecoder>,
	/// FFmpeg output channel → decoded channel (255 = silence).
	map: Vec<u8>,
	gain: f32,
	/// Conceal gaps in packet timestamps (default on).
	pub conceal_gaps: bool,
	/// Use in-band FEC when concealing a gap (default on).
	pub use_fec: bool,
	/// Longest gap, in 48 kHz samples, that is concealed; longer jumps are
	/// treated as discontinuities (a seek or a splice). Default 2 s.
	pub max_gap: usize,
	/// Expected pts (1/48000) of the next packet.
	next_pts: Option<u64>,
	/// Packets that could not be decoded and were concealed instead.
	pub concealed_frames: u64,
	/// Samples produced by concealment (lost packets, gaps and DTX aside).
	pub concealed_samples: u64,
}

/// Default for [`OpusDecoderState::max_gap`]: 2 s.
pub const DEFAULT_MAX_GAP: usize = 2 * SAMPLE_RATE as usize;
/// Concealment granularity: 2.5 ms.
const PLC_STEP: usize = 120;

/// How [`OpusDecoderState::run`] drives each stream.
#[derive(Clone, Copy)]
enum Request<'a> {
	Packet(&'a [u8]),
	Conceal(usize),
	Fec(&'a [u8], usize),
}

impl OpusDecoderState {
	pub fn new(head: OpusHead) -> Self {
		let m = &head.mapping;
		let streams = (0..usize::from(m.streams))
			.map(|s| StreamDecoder::new(if s < usize::from(m.coupled) { 2 } else { 1 }))
			.collect();
		let channels = usize::from(head.channels);
		let map: Vec<u8> = if m.family == 1 && channels <= 8 {
			crate::codecs::vorbis::CHANNEL_ORDER[channels - 1].iter().map(|&i| m.mapping[i]).collect()
		} else {
			m.mapping.clone()
		};
		let gain = if head.output_gain != 0 {
			10f32.powf(f32::from(head.output_gain) / (20.0 * 256.0))
		} else {
			1.0
		};
		Self {
			head,
			streams,
			map,
			gain,
			conceal_gaps: true,
			use_fec: true,
			max_gap: DEFAULT_MAX_GAP,
			next_pts: None,
			concealed_frames: 0,
			concealed_samples: 0,
		}
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
		self.next_pts = None;
	}

	/// Samples per channel of the last decode or concealment
	/// (`OPUS_GET_LAST_PACKET_DURATION`).
	pub fn last_packet_duration(&self) -> usize {
		self.streams.first().map_or(0, |s| s.last_packet_duration)
	}

	/// Decodes a packet into interleaved samples (`opus_multistream_decode_float`).
	pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<f32>> {
		self.run(Request::Packet(data))
	}

	/// Conceals `samples` lost samples per channel (a multiple of 120, i.e.
	/// 2.5 ms): `opus_multistream_decode_float` with no data, in calls of at
	/// most 120 ms.
	pub fn conceal(&mut self, samples: usize) -> Result<Vec<f32>> {
		self.recover(None, samples)
	}

	/// Recovers `samples` lost samples per channel (a multiple of 120) that
	/// precede `next`, the first packet received after the loss: the last
	/// lost frame comes from `next`'s in-band FEC when it carries any, the
	/// rest is concealed (`opus_multistream_decode_float` with
	/// `decode_fec = 1` for the final 120 ms at most). `next` must still be
	/// decoded afterwards.
	pub fn decode_fec(&mut self, next: &[u8], samples: usize) -> Result<Vec<f32>> {
		self.recover(Some(next), samples)
	}

	/// Splits a recovery into libopus-sized calls: 120 ms at most each, the
	/// last one covering the gap's final 120 ms or less (and using FEC).
	fn recover(&mut self, next: Option<&[u8]>, samples: usize) -> Result<Vec<f32>> {
		if !samples.is_multiple_of(PLC_STEP) {
			return Err(Message::Codec("opus concealment must be a multiple of 2.5 ms"));
		}
		let mut out = Vec::with_capacity(samples * self.channels());
		let mut left = samples;
		while left > 0 {
			let last = left <= MAX_FRAME_SAMPLES;
			let chunk = if last { left } else { (left - MAX_FRAME_SAMPLES).min(MAX_FRAME_SAMPLES) };
			let request = match next {
				Some(packet) if last => Request::Fec(packet, chunk),
				_ => Request::Conceal(chunk),
			};
			out.extend(self.run(request)?);
			left -= chunk;
		}
		Ok(out)
	}

	/// `opus_multistream_decode_native`: runs every stream and mixes them
	/// into the output channels.
	fn run(&mut self, request: Request<'_>) -> Result<Vec<f32>> {
		let channels = self.channels();
		let mut at = 0usize;
		let count = self.streams.len();
		let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(count);
		let mut samples = None;
		for (s, stream) in self.streams.iter_mut().enumerate() {
			let self_delimited = s + 1 != count;
			let (data, frame_size, fec) = match request {
				// An empty payload would mean "conceal" to libopus.
				Request::Packet(d) if d.len() <= at => {
					return Err(Message::Codec("empty opus packet"));
				}
				Request::Packet(d) => (Some(d.get(at..).unwrap_or(&[])), MAX_FRAME_SAMPLES, false),
				Request::Conceal(n) => (None, n, false),
				Request::Fec(d, n) => (Some(d.get(at..).unwrap_or(&[])), n, true),
			};
			let mut pcm = vec![0f32; frame_size * stream.channels];
			let (n, used) = stream
				.decode_native(data, &mut pcm, frame_size, fec, self_delimited)
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

	/// Decodes one demuxed packet with loss recovery: conceals the gap
	/// before it (if its `pts`, in 1/48000, is later than expected), then
	/// decodes it, concealing it instead when it is damaged. `duration` is
	/// the container's duration for the packet, used when the payload
	/// cannot tell. Returns the samples and the pts of the first one.
	pub fn decode_with_recovery(
		&mut self,
		data: &[u8],
		pts: Option<u64>,
		duration: Option<usize>,
	) -> (Vec<f32>, Option<u64>) {
		let channels = self.channels().max(1);
		let mut out = Vec::new();
		let mut start = pts;
		if let (true, Some(pts), Some(expected)) = (self.conceal_gaps, pts, self.next_pts) {
			// A lost packet leaves a gap of about a packet's duration; less
			// than half of one is timestamp jitter (Matroska stores
			// milliseconds, and muxers round). The gap is concealed to the
			// nearest 2.5 ms.
			let gap = pts.saturating_sub(expected) as usize;
			let threshold = (self.last_packet_duration() / 2).max(PLC_STEP);
			let lost = (gap + PLC_STEP / 2) / PLC_STEP * PLC_STEP;
			if gap >= threshold && lost <= self.max_gap {
				let recovered = if self.use_fec { self.decode_fec(data, lost) } else { self.conceal(lost) };
				if let Ok(samples) = recovered.or_else(|_| self.conceal(lost)) {
					out = samples;
					start = Some(expected);
					self.concealed_samples += lost as u64;
				}
			}
		}
		match self.decode_packet(data) {
			Ok(samples) => out.extend(samples),
			Err(_) => {
				// Damaged or empty: conceal what the packet should have held.
				self.concealed_frames += 1;
				let nominal = super::header::packet_samples(data)
					.map(|n| n as usize)
					.or(duration)
					.unwrap_or_else(|| self.last_packet_duration());
				let n = nominal.clamp(PLC_STEP, MAX_FRAME_SAMPLES).div_ceil(PLC_STEP) * PLC_STEP;
				match self.conceal(n) {
					Ok(samples) => {
						self.concealed_samples += n as u64;
						out.extend(samples);
					}
					Err(_) => {
						self.reset();
						out.resize(out.len() + n * channels, 0.0);
					}
				}
			}
		}
		self.next_pts = start.map(|p| p + (out.len() / channels) as u64);
		(out, start)
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
		let rate = TimeBase::new(1, SAMPLE_RATE);
		let pts = packet.pts.and_then(|t| t.rescale(rate)).map(|t| t.ticks());
		let duration = packet.duration.and_then(|t| t.rescale(rate)).map(|t| t.ticks() as usize);
		let (samples, start) = dec.decode_with_recovery(&packet.data, pts, duration);
		let data: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
		let mut frame = AudioFrame::new(
			data,
			SAMPLE_RATE,
			Channels::from_count(channels as u8),
			SampleFormat::F32,
			SampleLayout::Interleaved,
		);
		frame.pts = start.map(|t| Time::new(t, rate));
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
