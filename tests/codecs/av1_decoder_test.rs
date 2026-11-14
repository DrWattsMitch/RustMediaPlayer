//! AV1 decoder tests (rav1d behind RMP's `Decoder` trait). References are
//! FFmpeg `framecrc` dumps of libdav1d's output, so a match is bit-exact.

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
	String::from_utf8(fixture(&format!("av1/{name}.framecrc")))
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
	assert_eq!(stream.parameters.codec.id, "av1");
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
		assert_eq!(adler32(&data), hash, "{name} frame {i} differs from FFmpeg");
	}
	assert_eq!(frames[0].keyframe, Keyframe::Key);
	let pts: Vec<_> = frames.iter().map(|f| f.pts.map(|t| t.ticks())).collect();
	assert!(pts.windows(2).all(|w| w[0] < w[1]), "{name}: pts in presentation order");
}

#[test]
fn av1_8bit_in_mp4_is_bit_exact() {
	let mut d = Mp4Demuxer::new(memory(fixture("av1/aom_96x64.mp4"))).unwrap();
	check("aom_96x64", &decode(&mut d), (96, 64), 8);
}

#[test]
fn av1_10bit_in_matroska_is_bit_exact() {
	let mut d = MkvDemuxer::new(memory(fixture("av1/aom_10bit.mkv"))).unwrap();
	check("aom_10bit", &decode(&mut d), (64, 48), 10);
}

#[test]
fn damaged_av1_packets_error_without_panicking() {
	let mut d = Mp4Demuxer::new(memory(fixture("av1/aom_96x64.mp4"))).unwrap();
	let stream = d.streams().by_kind(StreamKind::Video)[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	while let Some(mut packet) = d.read().unwrap() {
		for (i, b) in packet.data.iter_mut().enumerate() {
			if i % 7 == 3 {
				*b ^= 0xA5;
			}
		}
		let _ = decoder.decode(&mut state, packet);
	}
}
