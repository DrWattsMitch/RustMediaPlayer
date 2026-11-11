use super::*;
use crate::codecs::h264::H264DecoderState;
use crate::codecs::h264::params::{parse_pps, parse_sps};

/// A moving test picture: gradients, a textured block that moves, and a
/// noise-like region (I420, `w` x `h`).
fn picture(w: usize, h: usize, t: usize) -> Vec<u8> {
	let mut data = vec![0u8; w * h * 3 / 2];
	for y in 0..h {
		for x in 0..w {
			let mut v = (x * 2 + y + t * 3) as i32 % 256;
			let (bx, by) = (x as i32 - (t as i32 * 3) % w as i32, y as i32 - (t as i32 * 2) % h as i32);
			if (0..24).contains(&bx) && (0..20).contains(&by) {
				v = if (bx / 4 + by / 4) % 2 == 0 { 230 } else { 20 };
			}
			if x > w * 2 / 3 && y > h / 2 {
				v = ((x * 7919 + y * 104_729 + t) % 251) as i32;
			}
			data[y * w + x] = v as u8;
		}
	}
	let (cw, ch) = (w / 2, h / 2);
	for y in 0..ch {
		for x in 0..cw {
			data[w * h + y * cw + x] = (128 + (x as i32 - cw as i32 / 2) / 2 + (t % 7) as i32) as u8;
			data[w * h + cw * ch + y * cw + x] = (128 + (y as i32 - ch as i32 / 2) / 2) as u8;
		}
	}
	data
}

/// The encoder's reconstruction of its last picture, cropped (I420).
fn recon(enc: &H264EncoderState) -> Vec<u8> {
	let r = enc.reference.as_ref().expect("reference");
	let (w, h) = (enc.width as usize, enc.height as usize);
	let mut out = Vec::with_capacity(w * h * 3 / 2);
	for y in 0..h {
		out.extend_from_slice(&r.y.data[y * r.y.width..y * r.y.width + w]);
	}
	for p in [&r.cb, &r.cr] {
		for y in 0..h / 2 {
			out.extend_from_slice(&p.data[y * p.width..y * p.width + w / 2]);
		}
	}
	out
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
	let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>()
		/ a.len() as f64;
	10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
}

fn run(w: usize, h: usize, frames: usize, mode: RateMode, gop: u32) -> (usize, f64) {
	let mut enc = H264EncoderState::new(w as u32, h as u32, 30.0, mode, gop).expect("state");
	let mut dec = H264DecoderState::new(&enc.avcc());
	let mut bytes = 0;
	let mut worst = f64::MAX;
	for t in 0..frames {
		let src = picture(w, h, t);
		let (au, idr) = enc.encode_picture(&src).expect("encodes");
		assert_eq!(idr, (t as u32).is_multiple_of(gop), "frame {t}");
		bytes += au.len();
		let mine = recon(&enc);
		let out = dec.decode_packet(&au, None).expect("decodes");
		assert_eq!(out.len(), 1, "frame {t}: decoder output count");
		assert_eq!(dec.damaged_slices, 0, "frame {t}");
		assert_eq!(out[0].data.len(), mine.len());
		if out[0].data != mine {
			let first = out[0].data.iter().zip(&mine).position(|(a, b)| a != b).unwrap_or(0);
			panic!(
				"frame {t} ({w}x{h}): decoder differs from encoder reconstruction at byte {first} (x {}, y {})",
				first % w,
				first / w
			);
		}
		worst = worst.min(psnr(&src, &mine));
	}
	(bytes, worst)
}

#[test]
fn parameter_sets_parse() {
	let enc = H264EncoderState::new(176, 100, 30.0, RateMode::Qp(26), 10).expect("state");
	let sps = parse_sps(&crate::codecs::h264::nal::unescape(&enc.sps[1..])).expect("sps");
	assert_eq!((sps.width(), sps.height()), (176, 100));
	let pps = parse_pps(&crate::codecs::h264::nal::unescape(&enc.pps[1..]), |_| Some(sps.clone()))
		.expect("pps");
	assert!(!pps.cabac);
}

#[test]
fn bit_exact_with_the_decoder() {
	for (w, h, frames, qp, gop) in
		[(64, 48, 6, 26, 4), (176, 144, 8, 30, 250), (98, 70, 5, 12, 3), (48, 32, 4, 45, 2)]
	{
		let (bytes, worst) = run(w, h, frames, RateMode::Qp(qp), gop);
		assert!(bytes > 0);
		assert!(worst > 20.0, "{w}x{h} qp {qp}: psnr {worst}");
	}
}

#[test]
fn extreme_quantisers_stay_exact() {
	run(32, 32, 3, RateMode::Qp(0), 2);
	run(32, 32, 3, RateMode::Qp(51), 2);
}

#[test]
fn bitrate_mode_converges() {
	let (w, h, frames) = (160, 96, 40);
	let target = 300_000;
	let (bytes, _) = run(w, h, frames, RateMode::Bitrate(target), 250);
	let rate = bytes as f64 * 8.0 * 30.0 / frames as f64;
	assert!((rate / f64::from(target) - 1.0).abs() < 0.35, "rate {rate}");
}

#[test]
fn odd_sizes_are_rejected() {
	assert!(H264EncoderState::new(101, 64, 30.0, RateMode::Qp(26), 10).is_err());
	assert!(H264EncoderState::new(0, 64, 30.0, RateMode::Qp(26), 10).is_err());
}
