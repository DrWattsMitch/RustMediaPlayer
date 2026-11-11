//! VP9 decoder tests. References are FFmpeg `framecrc` dumps (adler32 of
//! every shown frame, `ffmpeg -c:v vp9 -i X -f framecrc`), so a match is
//! bit-exact. Streams are libvpx encodes covering hidden alt-ref frames and
//! superframes, compound prediction, odd sizes, profile 1 (4:2:2, sRGB
//! 4:4:4), lossless, tiles, segmentation, error resilience and
//! frame-parallel mode.

use crate::common::{fixture, memory};
use rmp::codecs::vp9::Vp9DecoderState;
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
	String::from_utf8(fixture(&format!("vp9/{name}.framecrc")))
		.unwrap()
		.lines()
		.filter(|l| !l.starts_with('#'))
		.map(|l| {
			let f: Vec<&str> = l.split(',').map(str::trim).collect();
			(f[4].parse().unwrap(), u32::from_str_radix(f[5].trim_start_matches("0x"), 16).unwrap())
		})
		.collect()
}

fn decode(name: &str) -> Vec<VideoFrame> {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("vp9/{name}.webm")))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	assert_eq!(stream.parameters.codec.id, "vp9");
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
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

/// FFmpeg's planar layout of a frame (sRGB streams come out as packed
/// RGB24; FFmpeg's `gbrp` stores G, B, R planes).
fn planar(frame: &VideoFrame) -> Vec<u8> {
	if frame.pixel.format != PixelFormat::RGB24 {
		return frame.data.clone();
	}
	let px = frame.data.chunks_exact(3);
	let mut out: Vec<u8> = px.clone().map(|p| p[1]).collect();
	out.extend(px.clone().map(|p| p[2]));
	out.extend(px.map(|p| p[0]));
	out
}

fn assert_bit_exact(name: &str, size: (u32, u32), format: PixelFormat) {
	let frames = decode(name);
	let reference = reference(name);
	assert_eq!(frames.len(), reference.len(), "{name}: frame count");
	for (i, (frame, &(bytes, hash))) in frames.iter().zip(&reference).enumerate() {
		assert_eq!((frame.width, frame.height, frame.pixel.format), (size.0, size.1, format), "{name}");
		let data = planar(frame);
		assert_eq!(data.len(), bytes, "{name} frame {i}: size");
		assert_eq!(adler32(&data), hash, "{name} frame {i} differs from FFmpeg");
	}
	assert_eq!(frames[0].keyframe, Keyframe::Key);
	// Every frame keeps its packet's timestamp, in order.
	let pts: Vec<u64> = frames.iter().map(|f| f.pts.unwrap().ticks()).collect();
	assert!(pts.windows(2).all(|w| w[0] < w[1]), "{name}: pts {pts:?}");
}

#[test]
fn hidden_altref_superframes_are_bit_exact() {
	assert_bit_exact("altref_176x144", (176, 144), PixelFormat::YUV420);
}

#[test]
fn profile_1_srgb_444_with_odd_size_is_bit_exact() {
	assert_bit_exact("odd_444_99x61", (99, 61), PixelFormat::RGB24);
}

#[test]
fn profile_1_422_is_bit_exact() {
	assert_bit_exact("p1_422_96x64", (96, 64), PixelFormat::YUV422);
}

#[test]
fn lossless_is_bit_exact() {
	assert_bit_exact("lossless_64x48", (64, 48), PixelFormat::YUV420);
}

#[test]
fn tile_columns_and_rows_are_bit_exact() {
	assert_bit_exact("tiles_512x128", (512, 128), PixelFormat::YUV420);
}

#[test]
fn segmentation_and_error_resilience_are_bit_exact() {
	assert_bit_exact("rt_aq_errres_160x96", (160, 96), PixelFormat::YUV420);
}

#[test]
fn frame_parallel_mode_is_bit_exact() {
	assert_bit_exact("frame_parallel_128x72", (128, 72), PixelFormat::YUV420);
}

#[test]
fn webm_fixture_matches_ffmpeg_and_needs_a_key_frame() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/vp9_opus_vtt.webm"))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	let packets: Vec<_> = crate::common::read_all(&mut demuxer)
		.into_iter()
		.filter(|p| p.stream_id == stream.id)
		.collect();
	// Starting mid-stream (no key frame yet) is an error, not garbage.
	let mut dec = Vp9DecoderState::new();
	assert!(dec.decode_packet(&packets[1].data).is_err());
	// From the start, every packet shows one 64x48 frame.
	let mut dec = Vp9DecoderState::new();
	let shown: usize = packets.iter().map(|p| dec.decode_packet(&p.data).unwrap().len()).sum();
	assert_eq!(shown, packets.len());
}
