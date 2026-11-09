//! Re-encodes the audio of any input RMP can decode into a WAV file with a
//! chosen PCM codec, like `ffmpeg -i in -c:a pcm_s24le out.wav`.
//!
//! cargo run --release --example pcm_convert -- in.wav out.wav pcm_alaw
//!
//! WAV can hold pcm_u8, pcm_s16le, pcm_s24le, pcm_s32le, pcm_s64le,
//! pcm_f32le, pcm_f64le, pcm_alaw and pcm_mulaw.
use rmp::Result;
use rmp::core::traits::*;
use rmp::core::{CodecId, Resolver, StreamKind};
use rmp::io::{Input, Output};

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let (Some(input), Some(output), Some(codec)) = (args.get(1), args.get(2), args.get(3)) else {
		return Err("usage: pcm_convert <input> <output.wav> <pcm codec>".into());
	};

	let mut demuxer = Input::open(input)?;
	let source = demuxer
		.streams()
		.iter()
		.find(|s| s.kind == StreamKind::Audio)
		.cloned()
		.ok_or("input has no audio stream")?;

	let mut target = source.clone();
	target.parameters.codec = CodecId::new(codec);
	target.parameters.extradata.clear();

	let resolver = Resolver::new();
	let decoder = resolver.decoder_for(&source).ok_or("no decoder")?;
	let encoder = resolver.encoder_for(&target).ok_or("no encoder")?;
	let mut dec_state = decoder.create_state(&source.parameters);
	let mut enc_state = encoder.create_state(&target.parameters);

	let mut muxer = Output::create(output)?;
	muxer.set_metadata(demuxer.metadata().clone())?;
	muxer.add(&target)?;

	while let Some(packet) = demuxer.read()? {
		if packet.stream_id != source.id {
			continue;
		}
		for frame in decoder.decode(&mut dec_state, packet)?.value {
			for packet in encoder.encode(&mut enc_state, frame)?.value {
				muxer.write(packet)?;
			}
		}
	}
	for frame in decoder.flush(&mut dec_state)?.value {
		for packet in encoder.encode(&mut enc_state, frame)?.value {
			muxer.write(packet)?;
		}
	}
	muxer.finish()
}
