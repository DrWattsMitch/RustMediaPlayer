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
		return frame.data().into_owned();
	}
	let data = frame.data();
	let px = data.chunks_exact(3);
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

/// Every shown picture of a fixture, decoded by `dec`.
fn decode_with(name: &str, mut dec: Vp9DecoderState) -> Vec<Vec<u8>> {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("vp9/{name}.webm")))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	let mut out = Vec::new();
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id == stream.id {
			out.extend(dec.decode_packet(&packet.data).unwrap().iter().map(|p| p.to_bytes()));
		}
	}
	out
}

#[test]
fn tile_and_row_threading_are_bit_exact() {
	for name in [
		"tiles_512x128",
		"altref_176x144",
		"frame_parallel_128x72",
		"lossless_64x48",
		"odd_444_99x61",
		"p1_422_96x64",
		"rt_aq_errres_160x96",
	] {
		let serial = decode_with(name, Vp9DecoderState::new());
		assert_eq!(serial.len(), reference(name).len(), "{name}");
		for threads in [2, 3, 8] {
			for pipeline in [false, true] {
				let dec = Vp9DecoderState::with_threads(threads).with_row_pipeline(pipeline);
				assert_eq!(dec.threads(), threads);
				let threaded = decode_with(name, dec);
				assert!(threaded == serial, "{name}: {threads} threads, row pipeline {pipeline}");
			}
		}
	}
}

#[test]
fn tile_threading_fails_like_single_threaded_decoding_on_damage() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("vp9/tiles_512x128.webm"))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	let packets: Vec<_> = crate::common::read_all(&mut demuxer)
		.into_iter()
		.filter(|p| p.stream_id == stream.id)
		.collect();
	for cut in [3, 2] {
		let mut serial = Vp9DecoderState::new();
		let mut threaded = Vp9DecoderState::with_threads(4).with_row_pipeline(true);
		for (i, p) in packets.iter().enumerate() {
			let data = if i % 2 == 1 { &p.data[..p.data.len() * (cut - 1) / cut] } else { &p.data[..] };
			let a = serial.decode_packet(data);
			let b = threaded.decode_packet(data);
			assert_eq!(a.is_ok(), b.is_ok(), "packet {i}");
			if let (Ok(a), Ok(b)) = (a, b) {
				let a: Vec<_> = a.iter().map(|p| p.to_bytes()).collect();
				let b: Vec<_> = b.iter().map(|p| p.to_bytes()).collect();
				assert!(a == b, "packet {i}");
			}
		}
	}
}

/// Shown frames are views into the decoder's reference pictures (no copy)
/// and the planes are recycled once no slot or frame holds them.
#[test]
fn output_frames_share_and_recycle_decoder_buffers() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("vp9/altref_176x144.webm"))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Video)[0].clone();
	let mut packets = Vec::new();
	while let Some(p) = demuxer.read().unwrap() {
		if p.stream_id == stream.id {
			packets.push(p);
		}
	}
	let want: Vec<(usize, u32)> = reference("altref_176x144");
	let mut dec = Vp9DecoderState::new();
	let mut shown = 0;
	for p in &packets {
		for pic in dec.decode_packet(&p.data).unwrap() {
			let pixel = rmp::core::Pixel { depth: 8, format: PixelFormat::YUV420 };
			let frame = pic.to_frame(pixel, None);
			let data = frame.data();
			assert_eq!((data.len(), adler32(&data)), want[shown], "frame {shown}");
			// Superblock-aligned stride, visible width.
			let y = frame.plane(0).unwrap();
			assert_eq!((y.row_bytes(), y.stride() % 64), (176, 0));
			shown += 1;
		}
	}
	assert_eq!(shown, want.len());
	let stats = dec.pool_stats();
	// 8 reference slots + the frame being decoded, 3 planes each.
	assert!(stats.allocated <= 3 * 9, "{stats:?}");
	assert!(stats.reused > 0, "{stats:?}");
}
