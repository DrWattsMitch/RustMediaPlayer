//! HEVC decoder tests. References are FFmpeg `framecrc` dumps (FFmpeg's
//! own HEVC decoder), or the encoder's reconstruction where FFmpeg is not
//! a reliable reference, so a match is bit-exact.

use crate::common::{fixture, memory};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Demuxer, Frame, Keyframe, PixelFormat, Resolver, StreamKind, VideoFrame};

fn adler32(data: &[u8]) -> u32 {
	let (mut a, mut b) = (0u32, 0u32);
	for &byte in data {
		a = (a + u32::from(byte)) % 65521;
		b = (b + a) % 65521;
	}
	(b << 16) | a
}

fn reference(name: &str) -> Vec<(usize, u32)> {
	String::from_utf8(fixture(&format!("hevc/{name}.framecrc")))
		.unwrap()
		.lines()
		.filter(|l| !l.starts_with('#'))
		.map(|l| {
			let f: Vec<&str> = l.split(',').map(str::trim).collect();
			(f[4].parse().unwrap(), u32::from_str_radix(f[5].trim_start_matches("0x"), 16).unwrap())
		})
		.collect()
}

fn decode(demuxer: &mut dyn Demuxer) -> Vec<VideoFrame> {
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	assert_eq!(stream.parameters.codec.id, "hevc");
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut frames = Vec::new();
	let mut push = |out: Vec<Frame>| {
		for frame in out {
			let Frame::Video(v) = frame else { panic!("video expected") };
			frames.push(v);
		}
	};
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id == stream.id {
			push(decoder.decode(&mut state, packet).unwrap().value);
		}
	}
	push(decoder.flush(&mut state).unwrap().value);
	frames
}

fn check(name: &str, frames: &[VideoFrame], size: (u32, u32), depth: u8) {
	let reference = reference(name);
	assert_eq!(frames.len(), reference.len(), "{name}: frame count");
	for (i, (frame, &(bytes, hash))) in frames.iter().zip(&reference).enumerate() {
		assert_eq!((frame.width, frame.height), size, "{name}");
		assert_eq!((frame.pixel.format, frame.pixel.depth), (PixelFormat::YUV420, depth));
		let data = frame.data();
		assert_eq!(data.len(), bytes, "{name} frame {i}: size");
		assert_eq!(adler32(&data), hash, "{name} frame {i} differs from the reference");
	}
	assert_eq!(frames[0].keyframe, Keyframe::Key);
	let pts: Vec<_> = frames.iter().map(|f| f.pts.map(|t| t.ticks())).collect();
	assert!(pts.windows(2).all(|w| w[0] < w[1]), "{name}: pts in presentation order");
}

/// x265 Main: B-pyramid, open GOP (CRA + RASL pictures), AMP and
/// rectangular partitions, explicit weighted prediction (fade), WPP, SAO.
#[test]
fn hevc_main_in_mp4_is_bit_exact() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_main.mp4"))).unwrap();
	check("x265_main", &decode(&mut d), (160, 96), 8);
}

/// x265 Main 10 in Matroska, two slices per picture.
#[test]
fn hevc_main10_in_matroska_is_bit_exact() {
	let mut d = MkvDemuxer::new(memory(fixture("hevc/x265_main10.mkv"))).unwrap();
	check("x265_main10", &decode(&mut d), (192, 192), 10);
}

/// Transform skip, default scaling lists, per-CU lossless (transquant
/// bypass), 32x32 CTBs with 16x16 QP groups, chroma QP offsets.
#[test]
fn hevc_coding_tools_are_bit_exact() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_tools.mp4"))).unwrap();
	check("x265_tools", &decode(&mut d), (128, 96), 8);
}

/// Kvazaar, 2x2 tiles with wavefront parallel processing in each tile.
/// FFmpeg mis-decodes tiles combined with WPP, so the reference is the
/// encoder's own reconstruction.
#[test]
fn hevc_tiles_with_wpp_match_the_encoder() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/kvazaar_tiles_wpp.mp4"))).unwrap();
	check("kvazaar_tiles_wpp", &decode(&mut d), (192, 128), 8);
}

/// Corrupted packets must be reported or concealed, never panic.
#[test]
fn damaged_hevc_packets_do_not_panic() {
	for name in ["hevc/x265_main.mp4", "hevc/x265_tools.mp4", "hevc/kvazaar_tiles_wpp.mp4"] {
		for seed in 0..40u32 {
			let mut d = Mp4Demuxer::new(memory(fixture(name))).unwrap();
			let stream = d.streams().by_kind(StreamKind::Video)[0].clone();
			let decoder = Resolver::new().decoder_for(&stream).unwrap();
			let mut state = decoder.create_state(&stream.parameters);
			let mut n: u32 = seed;
			while let Some(mut packet) = d.read().unwrap() {
				for b in packet.data.iter_mut().skip(4) {
					n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
					if (n >> 16).is_multiple_of(61) {
						*b ^= (n >> 8) as u8;
					}
				}
				let _ = decoder.decode(&mut state, packet);
			}
			let _ = decoder.flush(&mut state);
		}
	}
}
