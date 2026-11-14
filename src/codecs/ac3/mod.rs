//! Dolby Digital (AC-3, ATSC A/52) and Dolby Digital Plus (E-AC-3, A/52
//! Annex E) decoding.
//!
//! FFmpeg: `libavcodec/ac3dec.c`, `eac3dec.c`, `ac3.c`, `ac3_parser.c`.
//! Supported: all AC-3 modes (coupling, rematrixing, block switching, delta
//! bit allocation, dual mono), and E-AC-3 independent + dependent substreams
//! (up to 7.1), spectral extension, adaptive hybrid transform (AHT). Like
//! FFmpeg, enhanced coupling and substreams other than 0 are not decoded.

// The decoder mirrors the specification's per-channel arrays.
#![allow(clippy::needless_range_loop)]

mod bitalloc;
pub mod decoder;
pub mod header;
mod tables;

pub use decoder::{Ac3DecoderState, DecodedFrame, find_sync, layout_of};
pub use header::{Header, HeaderError};

use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State, Time, TimeBase,
};
use crate::message::{Message, Result};

/// [`core::Decoder`] for `ac3` and `eac3`: one packet (one or more
/// syncframes) in, one planar `f32` [`AudioFrame`] per access unit out.
#[derive(Debug, Clone, Default)]
pub struct Ac3Decoder;

impl Ac3Decoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for Ac3Decoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		State::Audio(AudioState {
			sample_rate: parameters.sample_rate.unwrap_or(48_000),
			channels: Channels::from_count(parameters.channels.unwrap_or(2)),
			format: SampleFormat::F32,
			layout: SampleLayout::Planar,
			frame_size: 1536,
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::Ac3(Box::default()),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("ac-3 decoder requires an audio state"));
		};
		if !matches!(audio.private, CodecPrivate::Ac3(_)) {
			audio.private = CodecPrivate::Ac3(Box::default());
		}
		let CodecPrivate::Ac3(dec) = &mut audio.private else {
			return Err(Message::Codec("ac-3 decoder state missing"));
		};
		let decoded = dec.decode_packet(&packet.data);
		let mut frames = Vec::with_capacity(decoded.len());
		let mut pts = packet.pts;
		for f in decoded {
			let n = f.samples();
			let channels = f.planes.len();
			let mut data = Vec::with_capacity(n * channels * 4);
			for plane in &f.planes {
				for v in plane {
					data.extend_from_slice(&v.to_le_bytes());
				}
			}
			let mut out = AudioFrame::new(
				data,
				f.sample_rate,
				Channels::from_count(channels as u8),
				SampleFormat::F32,
				SampleLayout::Planar,
			);
			out.pts = pts;
			pts = pts.and_then(|t| {
				Time::new(n as u64, TimeBase::new(1, f.sample_rate))
					.rescale(t.base())
					.and_then(|s| t.checked_add(s))
			});
			audio.sample_rate = f.sample_rate;
			audio.channels = Channels::from_count(channels as u8);
			audio.samples_produced += n as u64;
			frames.push(Frame::Audio(out));
		}
		audio.samples_consumed += packet.data.len() as u64;
		Ok(DecodeOut { value: frames })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut> {
		if let State::Audio(audio) = state {
			if let CodecPrivate::Ac3(dec) = &mut audio.private {
				dec.reset();
			}
		}
		Ok(DecodeOut { value: Vec::new() })
	}
}
