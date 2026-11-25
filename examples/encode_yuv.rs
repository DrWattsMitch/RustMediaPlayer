//! Encodes raw I420 video with RMP's VP9 or AV1 encoder into an IVF file,
//! like `ffmpeg -f rawvideo -s WxH -i in.yuv -c:v libvpx-vp9 out.ivf`.
//!
//! cargo run --release --example encode_yuv -- vp9|av1 in.yuv WxH out.ivf [quantizer] [gop]
//!
//! Prints the size, bit rate at 25 fps, encode speed and PSNR of the
//! reconstruction (which is what a decoder shows).
use std::time::Instant;

pub fn main() -> rmp::Result<()> {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let usage = "usage: encode_yuv <vp9|av1> <in.yuv> <WxH> <out.ivf> [quantizer] [gop]";
	if args.len() < 4 {
		return Err(usage.into());
	}
	let (codec, input, size, output) = (&args[0], &args[1], &args[2], &args[3]);
	let (w, h) = size.split_once('x').ok_or(usage)?;
	let (w, h): (usize, usize) = (w.parse().map_err(|_| usage)?, h.parse().map_err(|_| usage)?);
	let quantizer: Option<u8> = args.get(4).and_then(|q| q.parse().ok());
	let gop: Option<u32> = args.get(5).and_then(|g| g.parse().ok());
	let data = std::fs::read(input)?;
	let frame_size = w * h + 2 * w.div_ceil(2) * h.div_ceil(2);
	let frames: Vec<&[u8]> = data.chunks_exact(frame_size).collect();

	let mut packets: Vec<Vec<u8>> = Vec::new();
	let mut sse = 0f64;
	let start = Instant::now();
	match codec.as_str() {
		"vp9" => {
			use rmp::codecs::vp9::encoder::{RateMode, Settings, Vp9EncoderState};
			let mut s = Settings::default();
			if let Some(q) = quantizer {
				s.mode = RateMode::Quantizer(q);
			}
			if let Some(g) = gop {
				s.gop = g;
			}
			let mut enc = Vp9EncoderState::new(w as u32, h as u32, s)?;
			for f in &frames {
				packets.push(enc.encode_picture(f, None)?.data);
				let rec = enc.recon_bytes();
				sse += rec.iter().zip(*f).map(|(&a, &b)| f64::from(a.abs_diff(b)).powi(2)).sum::<f64>();
			}
		}
		#[cfg(feature = "av1")]
		"av1" => {
			use rmp::codecs::av1::encoder::{Av1EncoderState, Settings};
			let mut s = Settings::default();
			if let Some(q) = quantizer {
				s.quantizer = q;
			}
			if let Some(g) = gop {
				s.gop = g;
			}
			let mut enc = Av1EncoderState::new(w as u32, h as u32, s)?;
			for f in &frames {
				// IVF carries each temporal unit with its delimiter.
				let mut tu = rmp::codecs::av1::encoder::temporal_delimiter();
				tu.extend(enc.encode_picture(f, None)?.data);
				packets.push(tu);
				let rec = enc.recon_bytes();
				sse += rec.iter().zip(*f).map(|(&a, &b)| f64::from(a.abs_diff(b)).powi(2)).sum::<f64>();
			}
		}
		_ => return Err(usage.into()),
	}
	let elapsed = start.elapsed().as_secs_f64();

	let fourcc: &[u8; 4] = if codec == "vp9" { b"VP90" } else { b"AV01" };
	let mut ivf = Vec::new();
	ivf.extend_from_slice(b"DKIF");
	ivf.extend_from_slice(&0u16.to_le_bytes());
	ivf.extend_from_slice(&32u16.to_le_bytes());
	ivf.extend_from_slice(fourcc);
	ivf.extend_from_slice(&(w as u16).to_le_bytes());
	ivf.extend_from_slice(&(h as u16).to_le_bytes());
	ivf.extend_from_slice(&25u32.to_le_bytes());
	ivf.extend_from_slice(&1u32.to_le_bytes());
	ivf.extend_from_slice(&(packets.len() as u32).to_le_bytes());
	ivf.extend_from_slice(&0u32.to_le_bytes());
	for (i, p) in packets.iter().enumerate() {
		ivf.extend_from_slice(&(p.len() as u32).to_le_bytes());
		ivf.extend_from_slice(&(i as u64).to_le_bytes());
		ivf.extend_from_slice(p);
	}
	std::fs::write(output, &ivf)?;
	let bytes: usize = packets.iter().map(Vec::len).sum();
	let mse = sse / (frames.len() * frame_size).max(1) as f64;
	println!(
		"{} frames {w}x{h}: {bytes} bytes ({:.1} kb/s at 25 fps), {:.2} frames/s, PSNR {:.2} dB",
		frames.len(),
		bytes as f64 * 8.0 * 25.0 / frames.len().max(1) as f64 / 1000.0,
		frames.len() as f64 / elapsed,
		10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
	);
	Ok(())
}
