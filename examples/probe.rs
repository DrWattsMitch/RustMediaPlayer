//! Prints container, stream and metadata information (like `ffprobe`).
//!
//! cargo run --release --example probe -- input.mp4
use rmp::Result;
use rmp::core::traits::*;
use rmp::io::Input;

pub fn main() -> Result<()> {
	let path = std::env::args().nth(1).ok_or("usage: probe <input>")?;
	let input = Input::open(&path)?;
	println!("duration: {} ({:.3}s)", input.duration(), input.duration().as_seconds());
	for stream in input.streams().iter() {
		let p = &stream.parameters;
		println!(
			"stream {}: {:?} codec={} rate={:?} channels={:?} size={:?}x{:?} bitrate={:?} extradata={}B",
			stream.id.0,
			stream.kind,
			p.codec.id,
			p.sample_rate,
			p.channels,
			p.width,
			p.height,
			p.bitrate,
			p.extradata.len()
		);
		for (key, value) in &stream.metadata.raw {
			if let rmp::core::RawValue::String(v) = value {
				println!("  {key}: {v}");
			}
		}
	}
	print!("{}", input.metadata());
	Ok(())
}
