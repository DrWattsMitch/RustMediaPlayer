//! Processing layer against FFmpeg 6.1 (`libswresample`, `libswscale`)
//! references in `tests/fixtures/process` (see `tools/make_fixtures.sh`).

use crate::common::{fixture, memory, wav_data};
use rmp::container::riff::demuxer::RiffDemuxer;
use rmp::container::y4m::demuxer::Y4mDemuxer;
use rmp::core::{
	AudioFrame, Channels, Demuxer, Frame, Pixel, Resolver, SampleFormat, SampleLayout, VideoFrame,
};
use rmp::process::{AudioConverter, AudioSpec, VideoConverter, VideoOptions, VideoSpec};

fn source_picture() -> VideoFrame {
	let mut demuxer = Y4mDemuxer::new(memory(fixture("process/testsrc_96x64.y4m"))).unwrap();
	let stream = demuxer.streams().streams[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let packet = demuxer.read().unwrap().unwrap();
	match decoder.decode(&mut state, packet).unwrap().value.remove(0) {
		Frame::Video(v) => v,
		Frame::Audio(_) => panic!("video expected"),
	}
}

fn psnr(a: &[u8], b: &[u8]) -> (f64, u8) {
	assert_eq!(a.len(), b.len());
	let mse = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>()
		/ a.len() as f64;
	let max = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
	(10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10(), max)
}

fn convert(frame: &VideoFrame, w: u32, h: u32, pixel: &str) -> VideoFrame {
	let out = VideoSpec::new(w, h, Pixel::from_name(pixel).unwrap());
	VideoConverter::new(VideoSpec::of(frame), out, VideoOptions::default())
		.unwrap()
		.convert(frame)
		.unwrap()
}

/// Scaling with the default bicubic kernel is within a few code values of
/// swscale (measured: 59-64 dB PSNR, max difference 1-5).
#[test]
fn scaling_matches_swscale() {
	let src = source_picture();
	for (w, h, reference) in
		[(100, 75, "process/testsrc_100x75.yuv"), (48, 32, "process/testsrc_48x32.yuv")]
	{
		let ours = convert(&src, w, h, "yuv420p");
		let (db, max) = psnr(&ours.data, &fixture(reference));
		eprintln!("scale {w}x{h}: {db:.1} dB, max {max}");
		assert!(db > 50.0 && max <= 8, "{w}x{h}: {db:.1} dB, max {max}");
	}
}

/// YUV ↔ RGB with BT.601 limited range matches swscale's accurate path.
#[test]
fn color_conversion_matches_swscale() {
	let src = source_picture();
	let rgb = convert(&src, 96, 64, "rgb24");
	let (db, _) = psnr(&rgb.data, &fixture("process/testsrc_96x64.accurate.rgb24"));
	eprintln!("yuv420p -> rgb24: {db:.1} dB");
	assert!(db > 35.0, "yuv420p -> rgb24: {db:.1} dB");
	let mut from_rgb = rgb.clone();
	from_rgb.data = fixture("process/testsrc_96x64.rgb24");
	let yuv = convert(&from_rgb, 96, 64, "yuv420p");
	let (db, _) = psnr(&yuv.data, &fixture("process/from_rgb_96x64.accurate.yuv"));
	eprintln!("rgb24 -> yuv420p: {db:.1} dB");
	assert!(db > 40.0, "rgb24 -> yuv420p: {db:.1} dB");
	// Packed orders are consistent.
	let bgra = convert(&src, 96, 64, "bgra");
	for (px, rgb_px) in bgra.data.chunks(4).zip(rgb.data.chunks(3)) {
		assert_eq!([px[2], px[1], px[0], px[3]], [rgb_px[0], rgb_px[1], rgb_px[2], 255]);
	}
}

fn f32_samples(raw: &[u8]) -> Vec<f32> {
	raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
	let signal: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
	let noise: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum();
	10.0 * (signal / noise.max(1e-30)).log10()
}

fn resample(out_rate: u32, out_channels: u8) -> Vec<f32> {
	let wav = fixture("process/tone_44k_stereo.wav");
	let pcm = wav_data(&wav).to_vec();
	let input = AudioSpec::new(44_100, 2, SampleFormat::S16, SampleLayout::Interleaved);
	let output = AudioSpec::new(out_rate, out_channels, SampleFormat::F32, SampleLayout::Interleaved);
	let mut converter = AudioConverter::new(input, output).unwrap();
	let mut out = Vec::new();
	for chunk in pcm.chunks(4 * 1000) {
		let frame = AudioFrame::new(
			chunk.to_vec(),
			44_100,
			Channels::Stereo,
			SampleFormat::S16,
			SampleLayout::Interleaved,
		);
		if let Some(f) = converter.convert(&frame).unwrap() {
			out.extend(f32_samples(&f.data));
		}
	}
	if let Some(f) = converter.flush().unwrap() {
		out.extend(f32_samples(&f.data));
	}
	out
}

/// Resampling matches swresample sample for sample away from the first and
/// last half-filter (where the two pad the stream edges differently):
/// measured ~97 dB SNR, under half a 16-bit LSB.
#[test]
fn resampling_matches_swresample() {
	for (rate, channels, reference) in [
		(48_000, 2, "process/tone_48k_stereo.ffmpeg.f32le"),
		(8000, 1, "process/tone_8k_mono.ffmpeg.f32le"),
	] {
		let ours = resample(rate, channels);
		let expected = f32_samples(&fixture(reference));
		assert_eq!(ours.len(), expected.len(), "{rate} Hz: sample count");
		let edge = 64 * usize::from(channels);
		let n = ours.len();
		let s = snr(&ours[edge..n - edge], &expected[edge..n - edge]);
		eprintln!("resample {rate} Hz: {s:.1} dB");
		assert!(s > 80.0, "{rate} Hz: {s:.1} dB");
	}
}

/// The WAV demuxer + PCM decoder + converter chain keeps timestamps.
#[test]
fn converted_audio_keeps_timestamps() {
	let mut demuxer = RiffDemuxer::new(memory(fixture("process/tone_44k_stereo.wav"))).unwrap();
	let stream = demuxer.streams().streams[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let output = AudioSpec::new(22_050, 1, SampleFormat::S16, SampleLayout::Planar);
	let mut converter: Option<AudioConverter> = None;
	let mut expected_pts = None;
	let mut total = 0u64;
	while let Some(packet) = demuxer.read().unwrap() {
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(a) = frame else { continue };
			let c =
				converter.get_or_insert_with(|| AudioConverter::new(AudioSpec::of(&a), output).unwrap());
			if let Some(out) = c.convert(&a).unwrap() {
				let pts = out.pts.unwrap().ticks();
				assert_eq!(Some(pts), expected_pts.or(Some(0)));
				expected_pts = Some(pts + out.nb_samples as u64);
				total += out.nb_samples as u64;
			}
		}
	}
	total += converter.unwrap().flush().unwrap().map_or(0, |f| f.nb_samples as u64);
	// 11 025 input samples → 5 512.5, rounded up.
	assert_eq!(total, 5513);
}
