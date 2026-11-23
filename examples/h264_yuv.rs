//! Decodes a raw H.264 Annex B stream to planar YUV (8-bit samples as
//! bytes, deeper samples as 16-bit little-endian), like
//! `ffmpeg -i in.264 -f rawvideo out.yuv`.
//!
//! cargo run --release --example h264_yuv -- in.264 out.yuv [threads]
//!
//! Prints the picture count, size and decode time. Comparing the output
//! with FFmpeg's (`cmp`) checks the decoder bit for bit.
use std::io::Write;
use std::time::Instant;

use rmp::codecs::h264::H264DecoderState;

pub fn main() -> rmp::Result<()> {
	let mut args = std::env::args().skip(1);
	let (Some(input), Some(output)) = (args.next(), args.next()) else {
		return Err("usage: h264_yuv <in.264> <out.yuv> [threads]".into());
	};
	let threads = args.next().and_then(|t| t.parse().ok()).unwrap_or(1);
	let data = std::fs::read(&input)?;
	let start = Instant::now();
	let mut dec = H264DecoderState::with_threads(&[], threads);
	let mut pictures = dec.decode_packet(&data, None)?;
	pictures.extend(dec.flush());
	let elapsed = start.elapsed();
	let mut out = std::io::BufWriter::new(std::fs::File::create(&output)?);
	for p in &pictures {
		out.write_all(&p.data())?;
	}
	out.flush()?;
	let (w, h) = pictures.first().map_or((0, 0), |p| (p.width, p.height));
	println!(
		"{} pictures {}x{} in {:.3}s, {} damaged slices{}",
		pictures.len(),
		w,
		h,
		elapsed.as_secs_f64(),
		dec.damaged_slices,
		dec.unsupported.map_or(String::new(), |u| format!(", unsupported: {u}"))
	);
	Ok(())
}
