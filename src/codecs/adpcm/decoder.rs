//! [`core::Decoder`] for the WAV ADPCM codecs: one block per packet in, one
//! interleaved S16 frame out.

use super::{AdpcmCodec, ima, ms};
use crate::Message;
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State,
};
use crate::message::Result;

#[derive(Debug, Clone, Copy)]
pub struct AdpcmDecoder {
	codec: AdpcmCodec,
}

impl AdpcmDecoder {
	pub fn new(codec: AdpcmCodec) -> Self {
		Self { codec }
	}

	pub fn for_id(id: &str) -> Option<Self> {
		AdpcmCodec::from_id(id).map(Self::new)
	}
}

/// Decodes one block to interleaved samples.
pub fn decode_block(codec: AdpcmCodec, block: &[u8], channels: usize) -> Result<Vec<i16>> {
	let mut out = Vec::new();
	match codec {
		AdpcmCodec::ImaWav => ima::decode_block(block, channels, &mut out)?,
		AdpcmCodec::Ms => ms::decode_block(block, channels, &mut out)?,
	};
	Ok(out)
}

impl core::Decoder for AdpcmDecoder {
	fn create_state(&self, p: &Parameters) -> State {
		State::Audio(AudioState {
			sample_rate: p.sample_rate.unwrap_or(44_100),
			channels: Channels::from_count(p.channels.unwrap_or(2)),
			format: SampleFormat::S16,
			layout: SampleLayout::Interleaved,
			frame_size: 0,
			codec_state: CodecState {
				extradata: p.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::None,
		})
	}

	/// Damaged blocks are an error for that packet only: every block starts
	/// from its own header, so the next packet decodes normally.
	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("adpcm decoder requires an audio state"));
		};
		let channels = usize::from(audio.channels.count());
		let pcm = decode_block(self.codec, &packet.data, channels)?;
		let data: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
		let mut frame = AudioFrame::new(
			data,
			audio.sample_rate,
			audio.channels,
			SampleFormat::S16,
			SampleLayout::Interleaved,
		);
		frame.pts = packet.pts;
		audio.samples_consumed += packet.data.len() as u64;
		audio.samples_produced += frame.nb_samples as u64;
		Ok(DecodeOut { value: vec![Frame::Audio(frame)], state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		Ok(DecodeOut { value: Vec::new(), state: state.clone() })
	}
}
