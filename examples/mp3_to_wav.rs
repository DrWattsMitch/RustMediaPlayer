//! Decodes an MP3 file to a 16-bit PCM WAV file.
//!
//! cargo run --release --example mp3_to_wav -- input.mp3 output.wav
use rmp::Result;
use rmp::core::traits::*;
use rmp::core::{CodecId, Frame, Packet, Resolver, Stream};
use rmp::io::{Input, Output};

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
		return Err("usage: mp3_to_wav <input.mp3> <output.wav>".into());
	};

	let mut demuxer = Input::open(input)?;
	let stream = demuxer.streams().iter().next().cloned().ok_or("no audio stream")?;
	println!("{}", demuxer.metadata());
	println!("duration: {}", demuxer.duration());

	let decoder = Resolver::new().decoder_for(&stream).ok_or("no decoder")?;
	let mut state = decoder.create_state(&stream.parameters);

	let mut muxer = Output::create(output)?;
	let sample_rate = stream.parameters.sample_rate.unwrap_or(44_100);
	let channels = stream.parameters.channels.unwrap_or(2);
	muxer.add(&Stream::audio(stream.id, sample_rate, channels, CodecId::new("pcm_s16le")))?;

	while let Some(packet) = demuxer.read()? {
		for frame in decoder.decode(&mut state, packet)?.value {
			if let Frame::Audio(frame) = frame {
				muxer.write(Packet::new(stream.id, frame.data))?;
			}
		}
	}
	muxer.finish()
}
