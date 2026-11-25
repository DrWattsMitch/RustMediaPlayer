//! Encoder tests: every stream is decoded with rav1d and compared with the
//! encoder's own reconstruction.

use super::*;
use crate::codecs::av1::Av1DecoderState;

/// A moving test pattern (packed I420).
pub(crate) fn pattern(w: usize, h: usize, t: usize) -> Vec<u8> {
	let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
	let mut out = vec![0u8; w * h + 2 * cw * ch];
	for y in 0..h {
		for x in 0..w {
			let (fx, fy) = ((x + 2 * t) as f64, (y + t) as f64);
			let v = 128.0 + 60.0 * (fx / 9.0).sin() * (fy / 13.0).cos() + 30.0 * ((fx + fy) / 23.0).sin();
			let sq = if ((x + 3 * t) / 16 + y / 16).is_multiple_of(2) { 20.0 } else { -20.0 };
			out[y * w + x] = (v + sq).clamp(0.0, 255.0) as u8;
		}
	}
	for y in 0..ch {
		for x in 0..cw {
			out[w * h + y * cw + x] = (128.0 + 40.0 * ((x + t) as f64 / 7.0).sin()) as u8;
			out[w * h + cw * ch + y * cw + x] = (128.0 + 40.0 * ((y + t) as f64 / 5.0).cos()) as u8;
		}
	}
	out
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
	let mse =
		a.iter().zip(b).map(|(&x, &y)| f64::from(x.abs_diff(y)).powi(2)).sum::<f64>() / a.len() as f64;
	if mse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

/// Encodes `frames` frames, decodes each with rav1d, checks bit-exactness;
/// returns (total bytes, mean PSNR).
fn round_trip(w: usize, h: usize, frames: usize, settings: Settings) -> (usize, f64) {
	let mut enc = Av1EncoderState::new(w as u32, h as u32, settings).unwrap();
	let mut dec = Av1DecoderState::new(&enc.av1c(), 1);
	let mut bytes = 0;
	let mut total_psnr = 0.0;
	for t in 0..frames {
		let src = pattern(w, h, t);
		let coded = enc.encode_picture(&src, None).unwrap();
		assert_eq!(coded.key, (t as u32).is_multiple_of(settings.gop), "frame {t}");
		bytes += coded.data.len();
		let mut pics = dec.decode_packet(&coded.data, t as i64).unwrap();
		if pics.is_empty() {
			pics = dec.drain().unwrap();
		}
		assert_eq!(pics.len(), 1, "frame {t}: one picture out");
		let p = &pics[0];
		assert_eq!((p.width, p.height), (w, h));
		let recon = enc.recon_bytes();
		if p.data != recon {
			let first = p.data.iter().zip(&recon).position(|(a, b)| a != b).unwrap();
			panic!("frame {t} ({w}x{h}): rav1d output differs from the reconstruction at byte {first}");
		}
		total_psnr += psnr(&src, &recon);
	}
	(bytes, total_psnr / frames as f64)
}

fn settings(q: u8, gop: u32) -> Settings {
	Settings { quantizer: q, gop, ..Settings::default() }
}

#[test]
fn key_frame_bit_exact() {
	let (bytes, psnr) = round_trip(64, 64, 1, settings(30, 128));
	assert!(bytes > 20);
	assert!(psnr > 30.0, "psnr {psnr}");
}

#[test]
fn inter_frames_bit_exact() {
	let (_, psnr) = round_trip(176, 144, 6, settings(30, 128));
	assert!(psnr > 30.0, "psnr {psnr}");
}

#[test]
fn odd_sizes_bit_exact() {
	for (w, h) in [(1, 1), (7, 5), (33, 17), (100, 70), (130, 66)] {
		round_trip(w, h, 3, settings(28, 2));
	}
}

#[test]
fn quantizers_bit_exact() {
	let mut last = usize::MAX;
	for q in [2, 20, 40, 63] {
		let (bytes, _) = round_trip(96, 64, 3, settings(q, 128));
		assert!(bytes < last, "q {q}: size should fall with the quantizer");
		last = bytes;
	}
}

#[test]
fn several_tiles_bit_exact() {
	// Wider than 4096 forces two tile columns.
	round_trip(4160, 16, 2, settings(40, 128));
}

#[test]
fn bitrate_mode_bit_exact() {
	let s = Settings { bitrate: Some(200_000), frame_rate: 25.0, ..settings(30, 128) };
	round_trip(96, 96, 5, s);
}

#[test]
fn inter_smaller_than_key() {
	let mut enc = Av1EncoderState::new(176, 144, settings(30, 128)).unwrap();
	let key = enc.encode_picture(&pattern(176, 144, 0), None).unwrap().data.len();
	let inter = enc.encode_picture(&pattern(176, 144, 1), None).unwrap().data.len();
	assert!(inter * 2 < key, "inter {inter} vs key {key}");
}
