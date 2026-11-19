//! VP8 decoder tests. References are FFmpeg `framecrc` dumps (adler32 of
//! every shown frame, `ffmpeg -i X -f framecrc`; libvpx's decoder gives
//! the same), so a match is bit-exact. Streams are libvpx encodes
//! covering hidden alt-ref frames with sign bias (2-pass), odd sizes,
//! four and eight token partitions, profiles 1 and 3 (bilinear filters,
//! simple loop filter, full-pixel chroma), q = 0, segmentation with error
//! resilience and frame-local probabilities, temporal layers, and a fast
//! pan with motion vectors past the frame edge.

use crate::common::{fixture, memory, read_all};
use rmp::codecs::vp8::Vp8DecoderState;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::core::{Demuxer, Frame, Keyframe, PixelFormat, Resolver, StreamKind, VideoFrame};

/// framecrc's adler32 (`av_adler32_update` seeded with 0, not 1).
fn adler32(data: &[u8]) -> u32 {
	let (mut a, mut b) = (0u32, 0u32);
	for &byte in data {
		a = (a + u32::from(byte)) % 65521;
		b = (b + a) % 65521;
	}
	(b << 16) | a
}

/// `(size, adler32)` of each frame in a framecrc file.
fn reference(name: &str) -> Vec<(usize, u32)> {
	String::from_utf8(fixture(&format!("vp8/{name}.framecrc")))
		.unwrap()
		.lines()
		.filter(|l| !l.starts_with('#'))
		.map(|l| {
			let f: Vec<&str> = l.split(',').map(str::trim).collect();
			(f[4].parse().unwrap(), u32::from_str_radix(f[5].trim_start_matches("0x"), 16).unwrap())
		})
		.collect()
}

fn packets(name: &str) -> Vec<rmp::core::Packet> {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("vp8/{name}.webm")))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	assert_eq!(stream.parameters.codec.id, "vp8");
	read_all(&mut demuxer).into_iter().filter(|p| p.stream_id == stream.id).collect()
}

fn decode(name: &str) -> Vec<VideoFrame> {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("vp8/{name}.webm")))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	let resolver = Resolver::new();
	assert!(resolver.has_decoder("vp8"));
	let decoder = resolver.decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut frames = Vec::new();
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id != stream.id {
			continue;
		}
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Video(v) = frame else { panic!("video expected") };
			frames.push(v);
		}
	}
	frames
}

fn assert_bit_exact(name: &str, size: (u32, u32)) {
	let frames = decode(name);
	let reference = reference(name);
	assert_eq!(frames.len(), reference.len(), "{name}: frame count");
	for (i, (frame, &(bytes, hash))) in frames.iter().zip(&reference).enumerate() {
		assert_eq!(
			(frame.width, frame.height, frame.pixel.format),
			(size.0, size.1, PixelFormat::YUV420),
			"{name}"
		);
		let data = frame.data();
		assert_eq!(data.len(), bytes, "{name} frame {i}: size");
		assert_eq!(adler32(&data), hash, "{name} frame {i} differs from FFmpeg");
	}
	assert_eq!(frames[0].keyframe, Keyframe::Key);
	let pts: Vec<u64> = frames.iter().map(|f| f.pts.unwrap().ticks()).collect();
	assert!(pts.windows(2).all(|w| w[0] < w[1]), "{name}: pts {pts:?}");
}

#[test]
fn basic_stream_is_bit_exact() {
	assert_bit_exact("basic", (176, 144));
}

#[test]
fn hidden_altref_frames_with_sign_bias_are_bit_exact() {
	assert_bit_exact("altref2pass", (144, 112));
	// Four alt-ref frames are decoded but not shown.
	let mut dec = Vp8DecoderState::new();
	let packets = packets("altref2pass");
	let shown = packets.iter().filter(|p| dec.decode_frame(&p.data).unwrap().is_some()).count();
	assert_eq!(shown + 4, packets.len());
}

#[test]
fn odd_size_is_bit_exact() {
	assert_bit_exact("odd", (99, 67));
}

#[test]
fn token_partitions_are_bit_exact() {
	assert_bit_exact("parts", (160, 120));
}

#[test]
fn bilinear_profiles_and_simple_loop_filter_are_bit_exact() {
	assert_bit_exact("prof1", (128, 96));
	assert_bit_exact("prof3", (128, 96));
}

#[test]
fn quantizer_zero_is_bit_exact() {
	assert_bit_exact("lossless", (96, 80));
}

#[test]
fn segmentation_and_frame_local_probabilities_are_bit_exact() {
	assert_bit_exact("er", (128, 128));
	assert_bit_exact("temporal", (202, 150));
}

#[test]
fn motion_past_the_frame_edge_is_bit_exact() {
	assert_bit_exact("pan", (112, 80));
}

#[test]
fn needs_a_key_frame_and_survives_damage() {
	let basic = packets("basic");
	let pan = packets("pan");
	let mut dec = Vp8DecoderState::new();
	assert!(dec.decode_frame(&basic[1].data).is_err());
	assert!(dec.decode_frame(&[]).is_err());
	// Damaged and truncated packets: errors or garbage, never a panic.
	let mut seed = 0x9e37_79b9u32;
	for round in 0..4 {
		let mut dec = Vp8DecoderState::new();
		for (i, p) in basic.iter().chain(pan.iter()).enumerate() {
			let mut data = p.data.clone();
			if (i + round) % 3 != 0 {
				for _ in 0..1 + round * 3 {
					seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
					// Keep the key frame's size bytes, so a flip can't ask
					// for a huge picture.
					let at = 10 + seed as usize % data.len().saturating_sub(10).max(1);
					if at < data.len() {
						data[at] ^= (seed >> 16) as u8 | 1;
					}
				}
			}
			if round == 3 && i % 2 == 1 {
				data.truncate(data.len() / 2);
			}
			let _ = dec.decode_frame(&data);
		}
	}
}

#[test]
fn key_frames_may_change_the_size() {
	let mut dec = Vp8DecoderState::new();
	let mut sizes = Vec::new();
	for name in ["basic", "odd", "basic"] {
		for p in packets(name).iter().take(5) {
			if let Some(pic) = dec.decode_frame(&p.data).unwrap() {
				sizes.push((pic.width, pic.height));
			}
		}
	}
	assert_eq!(sizes.len(), 15);
	assert_eq!((sizes[0], sizes[5], sizes[10]), ((176, 144), (99, 67), (176, 144)));
	// The re-opened stream decodes like a fresh decoder.
	let fresh: Vec<_> = {
		let mut d = Vp8DecoderState::new();
		packets("basic")
			.iter()
			.take(5)
			.filter_map(|p| d.decode_frame(&p.data).unwrap())
			.map(|p| p.to_bytes())
			.collect()
	};
	let again: Vec<_> = packets("basic")
		.iter()
		.take(5)
		.filter_map(|p| dec.decode_frame(&p.data).unwrap())
		.map(|p| p.to_bytes())
		.collect();
	assert_eq!(fresh, again);
}
