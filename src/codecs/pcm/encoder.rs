//! PCM encoder: frames in any sample format -> one stored PCM layout.
//!
//! The input frame's format does not have to match the target: an `S32`
//! frame can be written as `pcm_s16le`, or `F32` as `pcm_alaw`, using the
//! conversions in [`super::convert`].

use super::convert::{Sample, for_each_sample};
use super::{PcmCodec, g711};
use crate::Message;
use crate::core::{
	self, AudioState, Channels, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters,
	SampleLayout, State, StreamId,
};
use crate::message::Result;

#[derive(Debug, Clone, Copy)]
pub struct PcmEncoder {
	stream_id: StreamId,
	codec: PcmCodec,
}

impl PcmEncoder {
	pub fn new(stream_id: StreamId, codec: PcmCodec) -> Self {
		Self { stream_id, codec }
	}

	pub fn for_id(stream_id: StreamId, id: &str) -> Option<Self> {
		PcmCodec::from_id(id).map(|codec| Self::new(stream_id, codec))
	}
}

/// Writes one sample. Each codec first converts to its native sample format
/// (as FFmpeg's encoder input would be), then packs: so `pcm_s24le` from a
/// float is `f32 -> s32` with rounding, then `>> 8`, exactly as FFmpeg does.
fn store(codec: PcmCodec, s: Sample, out: &mut Vec<u8>) {
	match codec {
		PcmCodec::U8 => out.push((s.to_int(8) as i8 as u8) ^ 0x80),
		PcmCodec::S8 => out.push(s.to_int(8) as i8 as u8),
		PcmCodec::S16Le => out.extend_from_slice(&s.to_i16().to_le_bytes()),
		PcmCodec::S16Be => out.extend_from_slice(&s.to_i16().to_be_bytes()),
		PcmCodec::S24Le => out.extend_from_slice(&(s.to_i32() >> 8).to_le_bytes()[..3]),
		PcmCodec::S24Be => out.extend_from_slice(&(s.to_i32() >> 8).to_be_bytes()[1..]),
		PcmCodec::S32Le => out.extend_from_slice(&s.to_i32().to_le_bytes()),
		PcmCodec::S32Be => out.extend_from_slice(&s.to_i32().to_be_bytes()),
		PcmCodec::S64Le => out.extend_from_slice(&s.to_int(64).to_le_bytes()),
		PcmCodec::S64Be => out.extend_from_slice(&s.to_int(64).to_be_bytes()),
		PcmCodec::F32Le => out.extend_from_slice(&(s.to_float() as f32).to_le_bytes()),
		PcmCodec::F32Be => out.extend_from_slice(&(s.to_float() as f32).to_be_bytes()),
		PcmCodec::F64Le => out.extend_from_slice(&s.to_float().to_le_bytes()),
		PcmCodec::F64Be => out.extend_from_slice(&s.to_float().to_be_bytes()),
		PcmCodec::ALaw => out.push(g711::linear_to_alaw(s.to_i16())),
		PcmCodec::MuLaw => out.push(g711::linear_to_mulaw(s.to_i16())),
	}
}

impl core::Encoder for PcmEncoder {
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

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut<State>> {
		let Frame::Audio(frame) = frame else {
			return Err(Message::Codec("pcm encoder requires audio frames"));
		};
		let mut data =
			Vec::with_capacity(frame.data.len() / frame.format.bytes() * self.codec.coded_bytes());
		for_each_sample(&frame, |s| store(self.codec, s, &mut data))?;
		if let State::Audio(audio) = state {
			audio.samples_consumed += frame.nb_samples as u64;
		}
		let mut packet = Packet::new(self.stream_id, data);
		packet.pts = frame.pts;
		packet.dts = frame.pts;
		Ok(EncodeOut { value: vec![packet], state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<EncodeOut<State>> {
		Ok(EncodeOut { value: Vec::new(), state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::super::decoder::decode_samples;
	use super::*;
	use crate::core::{AudioFrame, Decoder, Encoder, SampleFormat};

	fn s32_frame(values: &[i32]) -> AudioFrame {
		let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
		AudioFrame::new(data, 48_000, Channels::Mono, SampleFormat::S32, SampleLayout::Interleaved)
	}

	#[test]
	fn every_codec_round_trips_its_own_output() {
		let values = [0, 1 << 24, -(1 << 24), 0x1234_5600, i32::MIN, 0x7FFF_FF00];
		for codec in PcmCodec::ALL {
			let encoder = PcmEncoder::new(StreamId(0), codec);
			let params =
				crate::core::Stream::audio(StreamId(0), 48_000, 1, crate::core::CodecId::new(codec.id()))
					.parameters;
			let mut state = encoder.create_state(&params);
			let packets = encoder.encode(&mut state, Frame::Audio(s32_frame(&values))).unwrap().value;
			let coded = &packets[0].data;
			assert_eq!(coded.len(), values.len() * codec.coded_bytes(), "{}", codec.id());

			// decode -> encode must reproduce the stored bytes exactly.
			let decoder = super::super::PcmDecoder::new(codec);
			let mut dstate = decoder.create_state(&params);
			let frames =
				decoder.decode(&mut dstate, Packet::new(StreamId(0), coded.clone())).unwrap().value;
			let Frame::Audio(decoded) = &frames[0] else { panic!() };
			assert_eq!(decoded.format, codec.sample_format());
			let again = encoder.encode(&mut state, Frame::Audio(decoded.clone())).unwrap().value;
			assert_eq!(&again[0].data, coded, "{}", codec.id());
			assert_eq!(decode_samples(codec, coded), decoded.data);
		}
	}

	#[test]
	fn planar_input_is_interleaved() {
		let data = [1i16, 2, 3, 10, 20, 30].iter().flat_map(|v| v.to_le_bytes()).collect();
		let frame =
			AudioFrame::new(data, 8000, Channels::Stereo, SampleFormat::S16, SampleLayout::Planar);
		let encoder = PcmEncoder::new(StreamId(0), PcmCodec::S16Le);
		let params =
			crate::core::Stream::audio(StreamId(0), 8000, 2, crate::core::CodecId::new("pcm_s16le"))
				.parameters;
		let mut state = encoder.create_state(&params);
		let out = encoder.encode(&mut state, Frame::Audio(frame)).unwrap().value;
		let samples: Vec<i16> =
			out[0].data.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
		assert_eq!(samples, [1, 10, 2, 20, 3, 30]);
	}
}
