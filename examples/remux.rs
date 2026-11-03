//! Copies every stream of a file into another container without re-encoding
//! (like `ffmpeg -i in -c copy out`). The output format is chosen from the
//! output extension, e.g. MP4 → M4A/MP4, MP3 → MP3.
//!
//! cargo run --release --example remux -- input.mp4 output.mp4
use rmp::Result;
use rmp::core::traits::*;
use rmp::io::{Input, Output};

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
		return Err("usage: remux <input> <output>".into());
	};

	let mut demuxer = Input::open(input)?;
	let mut muxer = Output::create(output)?;
	muxer.set_metadata(demuxer.metadata().clone())?;
	for stream in demuxer.streams().iter() {
		println!("stream {:?}: {:?} {}", stream.id, stream.kind, stream.parameters.codec.id);
		muxer.add(stream)?;
	}
	println!("duration: {}", demuxer.duration());

	let mut packets = 0u64;
	while let Some(packet) = demuxer.read()? {
		muxer.write(packet)?;
		packets += 1;
	}
	println!("packets: {packets}");
	muxer.finish()
}
