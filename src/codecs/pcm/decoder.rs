//! PCM decoder: stored layout -> native interleaved samples.

use super::{PcmCodec, g711};
use crate::Message;
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleLayout, State,
};
use crate::message::Result;

#[derive(Debug, Clone, Copy)]
pub struct PcmDecoder {
	codec: PcmCodec,
}

impl PcmDecoder {
	pub fn new(codec: PcmCodec) -> Self {
		Self { codec }
	}

	/// Decoder for an FFmpeg PCM codec ID, if it is one.
	pub fn for_id(id: &str) -> Option<Self> {
		PcmCodec::from_id(id).map(Self::new)
	}

	pub fn codec(&self) -> PcmCodec {
		self.codec
	}
}

/// Converts whole stored samples of `data` to the codec's native format.
/// Trailing bytes that do not make a whole sample are dropped (as FFmpeg's
/// `pcm_decode_frame` does).
pub fn decode_samples(codec: PcmCodec, data: &[u8]) -> Vec<u8> {
	let size = codec.coded_bytes();
	let whole = &data[..data.len() / size * size];
	let mut out = Vec::with_capacity(whole.len() / size * codec.sample_format().bytes());
	let samples = whole.chunks_exact(size);
	match codec {
		PcmCodec::U8
		| PcmCodec::S16Le
		| PcmCodec::S32Le
		| PcmCodec::S64Le
		| PcmCodec::F32Le
		| PcmCodec::F64Le => {
			// Already native: one copy.
			out.extend_from_slice(whole);
		}
		PcmCodec::S8 => out.extend(whole.iter().map(|b| b ^ 0x80)),
		PcmCodec::S16Be | PcmCodec::S32Be | PcmCodec::S64Be | PcmCodec::F32Be | PcmCodec::F64Be => {
			for s in samples {
				out.extend(s.iter().rev());
			}
		}
		PcmCodec::S24Le => {
			for s in samples {
				out.extend_from_slice(&[0, s[0], s[1], s[2]]);
			}
		}
		PcmCodec::S24Be => {
			for s in samples {
				out.extend_from_slice(&[0, s[2], s[1], s[0]]);
			}
		}
		PcmCodec::ALaw => {
			for b in whole {
				out.extend_from_slice(&g711::alaw_to_linear(*b).to_le_bytes());
			}
		}
		PcmCodec::MuLaw => {
			for b in whole {
				out.extend_from_slice(&g711::mulaw_to_linear(*b).to_le_bytes());
			}
		}
	}
	out
}

impl core::Decoder for PcmDecoder {
	fn create_state(&self, p: &Parameters) -> State {
		State::Audio(AudioState {
			sample_rate: p.sample_rate.unwrap_or(44_100),
			channels: Channels::from_count(p.channels.unwrap_or(2)),
			format: self.codec.sample_format(),
			layout: SampleLayout::Interleaved,
			frame_size: 0,
			codec_state: CodecState { extradata: Vec::new(), pts: None, dts: None, last_pts: None },
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::None,
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("pcm decoder requires an audio state"));
		};
		let channels = usize::from(audio.channels.count());
		if channels == 0 {
			return Err(Message::Codec("pcm stream has no channels"));
		}
		// Keep whole sample frames (one sample per channel) only.
		let frame_bytes = self.codec.coded_bytes() * channels;
		let usable = packet.data.len() / frame_bytes * frame_bytes;
		let mut frames = Vec::new();
		if usable > 0 {
			let data = decode_samples(self.codec, &packet.data[..usable]);
			let mut frame = AudioFrame::new(
				data,
				audio.sample_rate,
				audio.channels,
				self.codec.sample_format(),
				SampleLayout::Interleaved,
			);
			frame.pts = packet.pts;
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
	fn unpacks_24_bit_to_top_of_s32() {
		let le = decode_samples(PcmCodec::S24Le, &[0x56, 0x34, 0x12, 0xFF, 0xFF, 0xFF, 0x00]);
		assert_eq!(le, [0, 0x56, 0x34, 0x12, 0, 0xFF, 0xFF, 0xFF]);
		let be = decode_samples(PcmCodec::S24Be, &[0x12, 0x34, 0x56]);
		assert_eq!(i32::from_le_bytes(be.try_into().unwrap()), 0x1234_5600);
	}

	#[test]
	fn swaps_big_endian_and_flips_signed_8_bit() {
		assert_eq!(decode_samples(PcmCodec::S16Be, &[0x12, 0x34]), [0x34, 0x12]);
		assert_eq!(decode_samples(PcmCodec::F32Be, &1.5f32.to_be_bytes()), 1.5f32.to_le_bytes());
		assert_eq!(decode_samples(PcmCodec::S8, &[0x00, 0x80, 0x7F]), [0x80, 0x00, 0xFF]);
	}
}
