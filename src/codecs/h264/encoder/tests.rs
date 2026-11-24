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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
	let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>()
		/ a.len() as f64;
	10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
}

struct Run {
	bytes: usize,
	worst_psnr: f64,
	pictures: Vec<CodedPicture>,
}

/// Encodes `frames` pictures from `make`, decodes every access unit in
/// decoding order with RMP's decoder and checks that each output picture
/// (display order) equals the encoder's reconstruction bit for bit.
fn run_with(
	w: usize,
	h: usize,
	frames: usize,
	mode: RateMode,
	gop: u32,
	settings: Settings,
	make: &dyn Fn(usize) -> Vec<u8>,
) -> Run {
	let mut enc =
		H264EncoderState::with_settings(w as u32, h as u32, 30.0, mode, gop, settings).expect("state");
	enc.keep_recon = true;
	let mut dec = H264DecoderState::new(&enc.avcc());
	let mut coded = Vec::new();
	let mut decoded = Vec::new();
	let sources: Vec<Vec<u8>> = (0..frames).map(make).collect();
	let mut feed =
		|pics: Vec<CodedPicture>, dec: &mut H264DecoderState, coded: &mut Vec<CodedPicture>| {
			for p in pics {
				let out = dec.decode_packet(&p.data, None).expect("decodes");
				assert_eq!(dec.damaged_slices, 0, "picture {} ({})", p.index, p.kind);
				decoded.extend(out.into_iter().map(|d| d.data().to_vec()));
				coded.push(p);
			}
		};
	for src in &sources {
		let pics = enc.push_picture(src, None).expect("encodes");
		feed(pics, &mut dec, &mut coded);
	}
	feed(enc.flush(), &mut dec, &mut coded);
	decoded.extend(dec.flush().into_iter().map(|d| d.data().to_vec()));
	assert_eq!(coded.len(), frames, "every picture coded once");
	assert_eq!(decoded.len(), frames, "decoder output count");
	let mut by_index: Vec<&CodedPicture> = coded.iter().collect();
	by_index.sort_by_key(|p| p.index);
	let mut worst = f64::MAX;
	for (t, p) in by_index.iter().enumerate() {
		assert_eq!(p.index, t as u64);
		let mine = p.recon.as_ref().expect("recon");
		let theirs = &decoded[t];
		assert_eq!(theirs.len(), mine.len());
		if theirs != mine {
			let first = theirs.iter().zip(mine).position(|(a, b)| a != b).unwrap_or(0);
			panic!(
				"{:?}: picture {t} ({}, {w}x{h}): decoder differs from the encoder's reconstruction at byte {first} (x {}, y {})",
				settings.profile,
				p.kind,
				first % w,
				first / w
			);
		}
		worst = worst.min(psnr(&sources[t], mine));
	}
	Run { bytes: coded.iter().map(|p| p.data.len()).sum(), worst_psnr: worst, pictures: coded }
}

fn run(w: usize, h: usize, frames: usize, mode: RateMode, gop: u32, settings: Settings) -> Run {
	run_with(w, h, frames, mode, gop, settings, &|t| picture(w, h, t))
}

fn profiles() -> [Settings; 4] {
	[
		Settings::for_profile(Profile::Baseline),
		Settings::for_profile(Profile::Main),
		Settings::for_profile(Profile::High),
		Settings { trellis: false, ..Settings::for_profile(Profile::High) },
	]
}

#[test]
fn parameter_sets_parse() {
	for settings in profiles() {
		let enc = H264EncoderState::with_settings(176, 100, 30.0, RateMode::Qp(26), 10, settings)
			.expect("state");
		let sps = parse_sps(&crate::codecs::h264::nal::unescape(&enc.sps[1..])).expect("sps");
		assert_eq!((sps.width(), sps.height()), (176, 100));
		assert_eq!(sps.reorder_depth(), usize::from(settings.bframes > 0));
		let pps = parse_pps(&crate::codecs::h264::nal::unescape(&enc.pps[1..]), |_| Some(sps.clone()))
			.expect("pps");
		assert_eq!(pps.cabac, settings.profile != Profile::Baseline);
		assert_eq!(pps.transform_8x8_mode, settings.profile == Profile::High);
	}
}

#[test]
fn bit_exact_with_the_decoder() {
	for settings in profiles() {
		for (w, h, frames, qp, gop) in
			[(64, 48, 9, 26, 6), (176, 144, 10, 30, 250), (98, 70, 7, 12, 4), (48, 32, 6, 45, 3)]
		{
			let r = run(w, h, frames, RateMode::Qp(qp), gop, settings);
			assert!(r.bytes > 0);
			assert!(r.worst_psnr > 20.0, "{:?} {w}x{h} qp {qp}: psnr {}", settings.profile, r.worst_psnr);
		}
	}
}

#[test]
fn extreme_quantisers_stay_exact() {
	for settings in profiles() {
		run(32, 32, 5, RateMode::Qp(0), 4, settings);
		run(32, 32, 5, RateMode::Qp(51), 4, settings);
	}
}

#[test]
fn high_profile_uses_its_tools() {
	let s = Settings::for_profile(Profile::High);
	let r = run(176, 144, 12, RateMode::Qp(24), 250, s);
	let kinds: String = {
		let mut p: Vec<&CodedPicture> = r.pictures.iter().collect();
		p.sort_by_key(|p| p.index);
		p.iter().map(|p| p.kind).collect()
	};
	assert!(kinds.starts_with('I'), "{kinds}");
	assert!(kinds.contains('B'), "B pictures in {kinds}");
	let sum = |f: fn(&MbStats) -> u32| r.pictures.iter().map(|p| f(&p.stats)).sum::<u32>();
	assert!(sum(|s| s.intra8x8) > 0, "Intra8x8 used");
	assert!(sum(|s| s.inter8x8) > 0, "inter 8x8 transform used");
	assert!(sum(|s| s.list1) > 0, "list 1 / bi-prediction used");
	assert!(sum(|s| s.direct) > 0, "direct prediction used");
	// Decoding order: every B picture comes after the P picture that
	// follows it in display order.
	let mut seen_max = 0u64;
	for p in &r.pictures {
		if p.kind == 'B' {
			assert!(p.index < seen_max, "B picture {} coded before its anchor", p.index);
		} else {
			seen_max = seen_max.max(p.index);
		}
	}
}

#[test]
fn multiple_references_pay_off_on_alternating_content() {
	// Two scenes alternate: the picture two back predicts far better.
	let (w, h) = (96, 64);
	let make = |t: usize| picture(w, h, if t.is_multiple_of(2) { t } else { 40 + t });
	let one = Settings { refs: 1, bframes: 0, ..Settings::for_profile(Profile::Main) };
	let three = Settings { refs: 3, bframes: 0, ..Settings::for_profile(Profile::Main) };
	let a = run_with(w, h, 10, RateMode::Qp(26), 250, one, &make);
	let b = run_with(w, h, 10, RateMode::Qp(26), 250, three, &make);
	let multi: u32 = b.pictures.iter().map(|p| p.stats.multi_ref).sum();
	assert!(multi > 0, "reference index > 0 used");
	assert!(b.bytes < a.bytes, "3 refs {} bytes vs 1 ref {}", b.bytes, a.bytes);
}

#[test]
fn trellis_lowers_the_rate() {
	let (w, h) = (176, 144);
	let with = Settings::for_profile(Profile::High);
	let without = Settings { trellis: false, ..with };
	let a = run(w, h, 6, RateMode::Qp(26), 250, without);
	let b = run(w, h, 6, RateMode::Qp(26), 250, with);
	// Fewer bits at about the same quality.
	assert!(b.bytes < a.bytes, "trellis {} bytes vs deadzone {}", b.bytes, a.bytes);
	assert!(b.worst_psnr > a.worst_psnr - 1.0, "psnr {} vs {}", b.worst_psnr, a.worst_psnr);
}

#[test]
fn cabac_beats_cavlc() {
	let (w, h) = (176, 144);
	let base = Settings::for_profile(Profile::Baseline);
	let main =
		Settings { bframes: 0, refs: 1, trellis: false, ..Settings::for_profile(Profile::Main) };
	let a = run(w, h, 6, RateMode::Qp(28), 250, base);
	let b = run(w, h, 6, RateMode::Qp(28), 250, main);
	assert!(b.bytes < a.bytes, "cabac {} bytes vs cavlc {}", b.bytes, a.bytes);
}

#[test]
fn scene_cuts_start_a_new_gop() {
	let (w, h) = (96, 64);
	let make = |t: usize| {
		let mut p = picture(w, h, t);
		if t >= 7 {
			for (i, v) in p.iter_mut().enumerate().take(w * h) {
				*v = ((i * 7919 + (i / w) * 31) % 199) as u8;
			}
		}
		p
	};
	for settings in [Settings::for_profile(Profile::Baseline), Settings::for_profile(Profile::High)] {
		let r = run_with(w, h, 12, RateMode::Qp(26), 250, settings, &make);
		let mut idrs: Vec<u64> = r.pictures.iter().filter(|p| p.idr).map(|p| p.index).collect();
		idrs.sort_unstable();
		assert_eq!(idrs, vec![0, 7], "{:?}", settings.profile);
	}
}

#[test]
fn gop_interval_is_kept_with_b_frames() {
	let r = run(64, 48, 14, RateMode::Qp(26), 5, Settings::for_profile(Profile::High));
	let mut idrs: Vec<u64> = r.pictures.iter().filter(|p| p.idr).map(|p| p.index).collect();
	idrs.sort_unstable();
	assert_eq!(idrs, vec![0, 5, 10]);
}

#[test]
fn bitrate_mode_converges() {
	for settings in [Settings::for_profile(Profile::Baseline), Settings::for_profile(Profile::High)] {
		let (w, h, frames) = (160, 96, 40);
		let target = 120_000;
		let r = run(w, h, frames, RateMode::Bitrate(target), 250, settings);
		let rate = r.bytes as f64 * 8.0 * 30.0 / frames as f64;
		assert!((rate / f64::from(target) - 1.0).abs() < 0.35, "{:?}: rate {rate}", settings.profile);
	}
}

#[test]
fn odd_sizes_are_rejected() {
	assert!(H264EncoderState::new(101, 64, 30.0, RateMode::Qp(26), 10).is_err());
	assert!(H264EncoderState::new(0, 64, 30.0, RateMode::Qp(26), 10).is_err());
}

/// Busier content: several objects moving at different speeds over a
/// textured background, with fades.
fn busy(w: usize, h: usize, t: usize) -> Vec<u8> {
	let mut data = picture(w, h, t);
	for y in 0..h {
		for x in 0..w {
			let (fx, fy) = (x as f64, y as f64);
			let tt = t as f64;
			let mut v = 128.0 + 50.0 * ((fx + 2.3 * tt) * 0.15).sin() * ((fy - 1.1 * tt) * 0.12).cos();
			let (cx, cy) = (20.0 + 3.7 * tt, 30.0 + 1.3 * tt);
			if (fx - cx).powi(2) + (fy - cy).powi(2) < 120.0 {
				v = 240.0 - ((fx * 13.0 + fy * 7.0) as usize % 40) as f64;
			}
			if x % 37 < 9 && y > (t * 5) % h {
				v = f64::from(data[y * w + x]);
			}
			data[y * w + x] = (v * (1.0 - 0.02 * (t % 10) as f64)) as u8;
		}
	}
	data
}

#[test]
fn busy_content_stays_exact() {
	let (w, h) = (200, 120);
	for settings in [
		Settings { refs: 4, bframes: 3, ..Settings::for_profile(Profile::High) },
		Settings { refs: 2, bframes: 2, ..Settings::for_profile(Profile::Main) },
		Settings { refs: 1, ..Settings::for_profile(Profile::Baseline) },
	] {
		for qp in [18, 34] {
			let r = run_with(w, h, 20, RateMode::Qp(qp), 12, settings, &|t| busy(w, h, t));
			assert!(r.worst_psnr > 25.0, "{:?} qp {qp}: {}", settings.profile, r.worst_psnr);
		}
	}
}
