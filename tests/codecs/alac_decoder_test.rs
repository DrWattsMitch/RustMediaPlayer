//! ALAC decoder tests. Streams are FFmpeg `alac` encodes in MP4 and
//! Matroska (16-bit mono/stereo, 24-bit stereo and 5.1, and verbatim
//! frames from `-compression_level 0` on white noise); references are the
//! MD5 of FFmpeg's decoded PCM (`ffmpeg -i X -c:a pcm_s16le|pcm_s32le -f md5`),
//! so a match is bit-exact, channel order included.

use crate::common::{fixture, memory};
use rmp::codecs::alac::AlacConfig;
use rmp::codecs::flac::md5::Md5;
use rmp::container::detect;
use rmp::core::{Frame, Resolver, SampleFormat, StreamKind};

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decodes the file's ALAC stream; returns (format, channels, rate,
/// samples per channel, MD5 of the PCM).
fn decode(name: &str) -> (SampleFormat, u8, u32, usize, String) {
	let data = fixture(&format!("alac/{name}"));
	let mut demuxer = detect(&data, memory(data.clone())).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
	assert_eq!(stream.parameters.codec.id, "alac");
	assert!(AlacConfig::from_extradata(&stream.parameters.extradata).is_some(), "{name}: config");
	let resolver = Resolver::new();
	assert!(resolver.has_decoder("alac"));
	let decoder = resolver.decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut md5 = Md5::new();
	let mut info = None;
	let mut samples = 0;
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id != stream.id {
			continue;
		}
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(a) = frame else { panic!("audio expected") };
			assert!(!a.is_planar());
			md5.update(&a.data);
			samples += a.nb_samples;
			info = Some((a.format, a.channels.count(), a.sample_rate));
		}
	}
	let (format, channels, rate) = info.expect("no audio decoded");
	(format, channels, rate, samples, hex(&md5.finish()))
}

fn reference(name: &str) -> String {
	let text = String::from_utf8(fixture(&format!("alac/{name}.md5"))).unwrap();
	text.trim().trim_start_matches("MD5=").to_string()
}

fn assert_bit_exact(file: &str, format: SampleFormat, channels: u8, rate: u32, seconds: f64) {
	let (f, ch, r, samples, md5) = decode(file);
	assert_eq!((f, ch, r), (format, channels, rate), "{file}");
	assert_eq!(samples, (seconds * f64::from(rate)).round() as usize, "{file}: sample count");
	let stem = file.rsplit_once('.').unwrap().0;
	assert_eq!(md5, reference(stem), "{file} differs from FFmpeg");
}

#[test]
fn s16_stereo_mp4_matches_ffmpeg() {
	assert_bit_exact("s16_stereo.m4a", SampleFormat::S16, 2, 44_100, 0.6);
}

#[test]
fn s16_mono_matroska_matches_ffmpeg() {
	assert_bit_exact("s16_mono.mkv", SampleFormat::S16, 1, 44_100, 0.6);
}

#[test]
fn s24_stereo_with_extra_bits_matches_ffmpeg() {
	assert_bit_exact("s24_stereo.m4a", SampleFormat::S32, 2, 48_000, 0.5);
}

#[test]
fn s24_surround_channel_order_matches_ffmpeg() {
	assert_bit_exact("s24_51.mkv", SampleFormat::S32, 6, 32_000, 0.4);
}

#[test]
fn verbatim_frames_match_ffmpeg() {
	assert_bit_exact("verbatim.m4a", SampleFormat::S16, 2, 22_050, 0.4);
}

#[test]
fn damaged_packets_fail_cleanly() {
	let data = fixture("alac/s16_stereo.m4a");
	let mut demuxer = detect(&data, memory(data.clone())).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut seed = 0x1234_5678u32;
	while let Some(mut packet) = demuxer.read().unwrap() {
		for _ in 0..8 {
			seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
			let at = seed as usize % packet.data.len();
			packet.data[at] ^= (seed >> 16) as u8 | 1;
		}
		let cut = packet.data.len() * 3 / 4;
		let mut short = packet.clone();
		short.data.truncate(cut);
		// Errors are fine; panics are not.
		let _ = decoder.decode(&mut state, packet);
		let _ = decoder.decode(&mut state, short);
	}
}
