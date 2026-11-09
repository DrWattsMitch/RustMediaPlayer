//! Prints one line per demuxed packet, like `ffmpeg -c copy -f framecrc -`:
//! stream index, dts, pts, duration, size and the Adler-32 of the payload.
//!
//! cargo run --release --example framecrc -- input.mkv
//!
//! Missing timestamps print as `-`. Comparing with FFmpeg's output is the
//! quickest way to check a demuxer packet for packet.
use rmp::Result;
use rmp::core::traits::*;
use rmp::io::Input;

fn adler32(data: &[u8]) -> u32 {
	// FFmpeg seeds with 0 (`av_adler32_update(0, ...)`), not the standard 1.
	let (mut a, mut b) = (0u32, 0u32);
	for chunk in data.chunks(5552) {
		for &byte in chunk {
			a += u32::from(byte);
			b += a;
		}
		a %= 65521;
		b %= 65521;
	}
	(b << 16) | a
}

pub fn main() -> Result<()> {
	let path = std::env::args().nth(1).ok_or("usage: framecrc <input>")?;
	let mut input = Input::open(&path)?;
	for stream in input.streams().iter() {
		let tb = stream.time_base;
		println!("#tb {}: {}/{}", stream.id.0, tb.num, tb.den);
		println!("#codec_id {}: {}", stream.id.0, stream.parameters.codec.id);
		if !stream.parameters.extradata.is_empty() {
			let e = &stream.parameters.extradata;
			println!("#extradata {}: {:8}, 0x{:08x}", stream.id.0, e.len(), adler32(e));
		}
	}
	let show = |t: Option<rmp::core::Time>| t.map_or("-".to_string(), |t| t.ticks().to_string());
	while let Some(packet) = input.read()? {
		println!(
			"{}, {:>10}, {:>10}, {:>8}, {:>8}, 0x{:08x}{}",
			packet.stream_id.0,
			show(packet.dts),
			show(packet.pts),
			show(packet.duration),
			packet.data.len(),
			adler32(&packet.data),
			if packet.keyframe { "" } else { ", F=0x0" }
		);
	}
	Ok(())
}
