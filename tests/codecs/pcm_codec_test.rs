//! PCM decode and encode against FFmpeg. Every `wav/pcm/<codec>.wav` is
//! FFmpeg's encode of `wav/pcm/s16le.wav`; `<codec>.ffmpeg.s16le` is FFmpeg's
//! decode of it to S16.

use crate::common::{decode_all, fixture, memory, wav_data};
use rmp::codecs::pcm::convert::convert;
use rmp::codecs::pcm::{PcmCodec, PcmEncoder};
use rmp::container::riff::demuxer::RiffDemuxer;
use rmp::core::{
	AudioFrame, Channels, Demuxer, Encoder, Frame, SampleFormat, SampleLayout, StreamId,
};

const CODECS: [&str; 9] =
	["u8", "s16le", "s24le", "s32le", "s64le", "f32le", "f64le", "alaw", "mulaw"];

fn as_s16(codec: PcmCodec, native: Vec<u8>) -> Vec<u8> {
	let frame =
		AudioFrame::new(native, 8000, Channels::Mono, codec.sample_format(), SampleLayout::Interleaved);
	convert(&frame, SampleFormat::S16).unwrap()
}

#[test]
fn decodes_every_wav_layout_like_ffmpeg() {
	for name in CODECS {
		let mut demuxer = RiffDemuxer::new(memory(fixture(&format!("wav/pcm/{name}.wav")))).unwrap();
		let id = demuxer.streams().streams[0].parameters.codec.id.clone();
		assert_eq!(id, format!("pcm_{name}"));
		let codec = PcmCodec::from_id(&id).unwrap();
		let native = decode_all(&mut demuxer);
		assert_eq!(native.len(), 800 * codec.sample_format().bytes(), "{name}");
		assert_eq!(as_s16(codec, native), fixture(&format!("wav/pcm/{name}.ffmpeg.s16le")), "{name}");
	}
}

#[test]
fn encodes_s16_to_every_wav_layout_like_ffmpeg() {
	let source = wav_data(&fixture("wav/pcm/s16le.wav")).to_vec();
	for name in CODECS {
		let codec = PcmCodec::from_id(&format!("pcm_{name}")).unwrap();
		let encoder = PcmEncoder::new(StreamId(0), codec);
		let stream =
			rmp::core::Stream::audio(StreamId(0), 8000, 1, rmp::core::CodecId::new(codec.id()));
		let mut state = encoder.create_state(&stream.parameters);
		let frame = AudioFrame::new(
			source.clone(),
			8000,
			Channels::Mono,
			SampleFormat::S16,
			SampleLayout::Interleaved,
		);
		let packets = encoder.encode(&mut state, Frame::Audio(frame)).unwrap().value;
		let expected = fixture(&format!("wav/pcm/{name}.wav"));
		assert_eq!(packets[0].data, wav_data(&expected), "{name}");
	}
}

#[test]
fn packets_are_timed_in_samples() {
	let mut demuxer = RiffDemuxer::new(memory(fixture("wav/pcm/s24le.wav"))).unwrap();
	let mut expected = 0;
	while let Some(packet) = demuxer.read().unwrap() {
		assert_eq!(packet.pts.map(|t| t.ticks()), Some(expected));
		assert_eq!(packet.data.len() % 3, 0);
		expected += (packet.data.len() / 3) as u64;
	}
	assert_eq!(expected, 800);
	assert_eq!(demuxer.duration().ticks(), 800);
}

#[test]
fn partial_trailing_sample_is_dropped() {
	use rmp::codecs::pcm::PcmDecoder;
	use rmp::core::{Decoder, Packet};
	let decoder = PcmDecoder::new(PcmCodec::S24Le);
	let stream = rmp::core::Stream::audio(StreamId(0), 8000, 2, rmp::core::CodecId::new("pcm_s24le"));
	let mut state = decoder.create_state(&stream.parameters);
	// 1.5 stereo sample frames: only the first is decoded.
	let out = decoder.decode(&mut state, Packet::new(StreamId(0), vec![1; 9])).unwrap().value;
	let Frame::Audio(frame) = &out[0] else { panic!() };
	assert_eq!(frame.nb_samples, 1);
	assert!(
		decoder.decode(&mut state, Packet::new(StreamId(0), vec![1; 5])).unwrap().value.is_empty()
	);
}
