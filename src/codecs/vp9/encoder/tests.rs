//! The encoder's reconstruction must be what RMP's decoder decodes from the
//! encoder's packets, bit for bit, for every frame.

use super::*;
use crate::codecs::vp9::Vp9DecoderState;

/// A test sequence: gradients, a textured background, moving and fading
/// objects, some noise (`I420`).
pub(crate) fn synthetic(w: usize, h: usize, frames: usize, seed: u32) -> Vec<Vec<u8>> {
	let mut s = seed.wrapping_mul(2_654_435_761).max(1);
	let mut noise = move || {
		s ^= s << 13;
		s ^= s >> 17;
		s ^= s << 5;
		(s % 9) as i32 - 4
	};
	let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
	(0..frames)
		.map(|t| {
			let t = t as i32;
			let mut out = vec![0u8; w * h + 2 * cw * ch];
			for y in 0..h {
				for x in 0..w {
					let (xi, yi) = (x as i32, y as i32);
					let mut v = (xi * 2 + yi + t * 3) % 200 + 20;
					// Textured band.
					if (yi / 6 + xi / 9 + t / 2) % 3 == 0 {
						v += ((xi * 7 + yi * 3) % 23) - 11;
					}
					// A square moving right and down.
					let (ox, oy) = (5 + t * 3, 4 + t * 2);
					if xi >= ox && xi < ox + w as i32 / 3 && yi >= oy && yi < oy + h as i32 / 3 {
						v = 230 - ((xi - ox) ^ (yi - oy)) % 40;
					}
					// A disc moving left.
					let (cx, cy) = (w as i32 - 10 - t * 4, h as i32 / 2);
					if (xi - cx).pow(2) + (yi - cy).pow(2) < (h as i32 / 5).pow(2) {
						v = 40 + t * 5;
					}
					out[y * w + x] = (v + noise()).clamp(0, 255) as u8;
				}
			}
			for y in 0..ch {
				for x in 0..cw {
					let (xi, yi) = (x as i32, y as i32);
					out[w * h + y * cw + x] = (128 + (xi - yi + t) % 50 - 25 + noise()).clamp(0, 255) as u8;
					out[w * h + cw * ch + y * cw + x] =
						(110 + (xi * yi / 7 + t * 2) % 60 + noise()).clamp(0, 255) as u8;
				}
			}
			out
		})
		.collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
	let mse =
		a.iter().zip(b).map(|(&x, &y)| f64::from(x.abs_diff(y)).powi(2)).sum::<f64>() / a.len() as f64;
	if mse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

/// Encodes `frames`, decodes every packet with RMP's decoder and checks the
/// decoder's pictures against the encoder's reconstruction. Returns the
/// packets and the mean luma+chroma PSNR against the source.
fn run(w: usize, h: usize, frames: &[Vec<u8>], settings: Settings) -> (Vec<CodedFrame>, f64) {
	let mut enc = Vp9EncoderState::new(w as u32, h as u32, settings).unwrap();
	let mut dec = Vp9DecoderState::new();
	let mut out = Vec::new();
	let mut total = 0.0;
	for (i, f) in frames.iter().enumerate() {
		let coded = enc.encode_picture(f, None).unwrap();
		let pics = dec.decode_packet(&coded.data).unwrap_or_else(|e| panic!("frame {i}: {e:?}"));
		assert_eq!(pics.len(), 1, "frame {i}");
		let decoded = pics[0].to_bytes();
		let recon = enc.recon.as_ref().unwrap().bytes();
		assert_eq!(decoded.len(), recon.len());
		if decoded != recon {
			let at = decoded.iter().zip(&recon).position(|(a, b)| a != b).unwrap();
			panic!(
				"{w}x{h} frame {i} (key {}): reconstruction differs from the decoder at byte {at}",
				coded.key
			);
		}
		total += psnr(&decoded, f);
		out.push(coded);
	}
	(out, total / frames.len() as f64)
}

fn q(quantizer: u8, gop: u32) -> Settings {
	Settings { mode: RateMode::Quantizer(quantizer), gop, ..Settings::default() }
}

#[test]
fn key_and_inter_frames_are_bit_exact() {
	let frames = synthetic(96, 64, 6, 1);
	let (coded, psnr) = run(96, 64, &frames, q(30, 128));
	assert!(coded[0].key && coded[1..].iter().all(|c| !c.key));
	assert!(psnr > 32.0, "psnr {psnr}");
	// Inter frames are smaller than the key frame (the content is noisy).
	let inter: usize = coded[1..].iter().map(|c| c.data.len()).sum::<usize>() / (coded.len() - 1);
	assert!(inter < coded[0].data.len(), "key {} inter {inter}", coded[0].data.len());
}

#[test]
fn odd_sizes_and_frame_edges() {
	for (w, h) in [(33, 17), (8, 8), (65, 71), (130, 9)] {
		let frames = synthetic(w, h, 3, w as u32);
		run(w, h, &frames, q(28, 2));
	}
}

#[test]
fn every_quantizer_range() {
	let frames = synthetic(64, 48, 3, 9);
	let mut last = usize::MAX;
	for quantizer in [0u8, 10, 30, 50, 63] {
		let (coded, psnr) = run(64, 48, &frames, q(quantizer, 128));
		let size: usize = coded.iter().map(|c| c.data.len()).sum();
		assert!(size < last || quantizer == 0, "q {quantizer}: {size} bytes");
		last = size;
		if quantizer <= 10 {
			assert!(psnr > 42.0, "q {quantizer}: psnr {psnr}");
		}
	}
}

#[test]
fn several_tile_columns() {
	// 4096+ pixels wide needs two tile columns at least; keep it short.
	let (w, h) = (4160, 16);
	let frames = synthetic(w, h, 2, 3);
	let mut enc = Vp9EncoderState::new(w as u32, h as u32, q(40, 128)).unwrap();
	let mut dec = Vp9DecoderState::new();
	for f in &frames {
		let coded = enc.encode_picture(f, None).unwrap();
		let pics = dec.decode_packet(&coded.data).unwrap();
		assert_eq!(pics[0].to_bytes(), enc.recon.as_ref().unwrap().bytes());
	}
}

#[test]
fn bitrate_mode_tracks_the_target() {
	let frames = synthetic(128, 96, 20, 5);
	let bitrate = 300_000;
	let settings =
		Settings { mode: RateMode::Bitrate(bitrate), frame_rate: 25.0, ..Settings::default() };
	let (coded, _) = run(128, 96, &frames, settings);
	let bits: usize = coded.iter().map(|c| c.data.len() * 8).sum();
	let rate = bits as f64 * 25.0 / coded.len() as f64;
	assert!(rate > f64::from(bitrate) * 0.5 && rate < f64::from(bitrate) * 1.6, "{rate} b/s");
}

#[test]
fn ffmpeg_decodes_the_stream_when_available() {
	// Cross-check with libvpx through FFmpeg, if it is installed.
	let Ok(out) = std::process::Command::new("ffmpeg").arg("-version").output() else { return };
	if !out.status.success() {
		return;
	}
	let (w, h) = (80, 48);
	let frames = synthetic(w, h, 5, 11);
	let (coded, _) = run(w, h, &frames, q(30, 3));
	let dir = tempfile::tempdir().unwrap();
	// IVF container.
	let mut ivf = Vec::new();
	ivf.extend_from_slice(b"DKIF");
	ivf.extend_from_slice(&0u16.to_le_bytes());
	ivf.extend_from_slice(&32u16.to_le_bytes());
	ivf.extend_from_slice(b"VP90");
	ivf.extend_from_slice(&(w as u16).to_le_bytes());
	ivf.extend_from_slice(&(h as u16).to_le_bytes());
	ivf.extend_from_slice(&25u32.to_le_bytes());
	ivf.extend_from_slice(&1u32.to_le_bytes());
	ivf.extend_from_slice(&(coded.len() as u32).to_le_bytes());
	ivf.extend_from_slice(&0u32.to_le_bytes());
	for (i, c) in coded.iter().enumerate() {
		ivf.extend_from_slice(&(c.data.len() as u32).to_le_bytes());
		ivf.extend_from_slice(&(i as u64).to_le_bytes());
		ivf.extend_from_slice(&c.data);
	}
	let input = dir.path().join("in.ivf");
	let output = dir.path().join("out.yuv");
	std::fs::write(&input, ivf).unwrap();
	let status = std::process::Command::new("ffmpeg")
		.args(["-v", "error", "-i"])
		.arg(&input)
		.args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
		.arg(&output)
		.status()
		.unwrap();
	assert!(status.success());
	let yuv = std::fs::read(&output).unwrap();
	let mut dec = Vp9DecoderState::new();
	let mut ours = Vec::new();
	for c in &coded {
		ours.extend(dec.decode_packet(&c.data).unwrap()[0].to_bytes());
	}
	assert_eq!(yuv, ours, "FFmpeg decodes the stream differently");
}
