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
	check_format(name, frames, size, depth, PixelFormat::YUV420);
}

fn check_format(
	name: &str,
	frames: &[VideoFrame],
	size: (u32, u32),
	depth: u8,
	format: PixelFormat,
) {
	let reference = reference(name);
	assert_eq!(frames.len(), reference.len(), "{name}: frame count");
	for (i, (frame, &(bytes, hash))) in frames.iter().zip(&reference).enumerate() {
		assert_eq!((frame.width, frame.height), size, "{name}");
		assert_eq!((frame.pixel.format, frame.pixel.depth), (format, depth), "{name}");
		let data = frame.data();
		assert_eq!(data.len(), bytes, "{name} frame {i}: size");
		assert_eq!(adler32(&data), hash, "{name} frame {i} differs from the reference");
	}
	assert_eq!(frames[0].keyframe, Keyframe::Key);
	// Raw x265 streams remuxed by FFmpeg carry no B-frame timestamps.
	if name.ends_with("_ctu16") || name.ends_with("_odd") {
		return;
	}
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
	for name in [
		"hevc/x265_main.mp4",
		"hevc/x265_tools.mp4",
		"hevc/kvazaar_tiles_wpp.mp4",
		"hevc/x265_422_ctu16.mp4",
		"hevc/x265_444_12.mp4",
		"hevc/x265_400_12.mp4",
		"hevc/rext_444.mp4",
		"hevc/rext_422_12.mp4",
	] {
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

/// x265 with WPP over several CTB rows (416x240: 4 rows of 7 CTBs),
/// B-pyramid, three references.
#[test]
fn hevc_wpp_rows_are_bit_exact() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_wpp.mp4"))).unwrap();
	check("x265_wpp", &decode(&mut d), (416, 240), 8);
}

/// Main 10, 16x16 CTBs (13 rows), three slices per picture with WPP,
/// weighted bi-prediction.
#[test]
fn hevc_wpp_with_slices_and_small_ctbs_is_bit_exact() {
	let mut d = MkvDemuxer::new(memory(fixture("hevc/x265_wpp_ctu16_slices.mkv"))).unwrap();
	check("x265_wpp_ctu16_slices", &decode(&mut d), (352, 208), 10);
}

/// x265 Monochrome 12: 4:0:0, B-frames, WPP.
#[test]
fn hevc_monochrome_12_is_bit_exact() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_400_12.mp4"))).unwrap();
	check_format("x265_400_12", &decode(&mut d), (160, 96), 12, PixelFormat::Gray);
}

/// x265 Main 4:2:2 10: B-pyramid, transform skip, default scaling lists,
/// weighted bi-prediction (fade), AMP, two slices.
#[test]
fn hevc_422_10_is_bit_exact() {
	let mut d = MkvDemuxer::new(memory(fixture("hevc/x265_422_10.mkv"))).unwrap();
	check_format("x265_422_10", &decode(&mut d), (192, 128), 10, PixelFormat::YUV422);
}

/// x265 4:2:2 with 16x16 CTBs, SAO and lossless CUs. FFmpeg 6.1 differs
/// from the encoder here (SAO of 4:2:2 chroma at CTB edges, and lossless
/// chroma it does not restore after SAO), so the reference is the
/// encoder's own reconstruction.
#[test]
fn hevc_422_small_ctbs_match_the_encoder() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_422_ctu16.mp4"))).unwrap();
	check_format("x265_422_ctu16", &decode(&mut d), (176, 144), 8, PixelFormat::YUV422);
}

/// x265 Main 4:4:4 12: intra NxN with a chroma mode per block, transform
/// skip, constrained intra prediction, AMP, weighted prediction.
#[test]
fn hevc_444_12_is_bit_exact() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_444_12.mp4"))).unwrap();
	check_format("x265_444_12", &decode(&mut d), (128, 96), 12, PixelFormat::YUV444);
}

/// 4:4:4 at an odd size (cropped 131x71) with lossless CUs and WPP;
/// reference: the encoder's reconstruction.
#[test]
fn hevc_444_odd_size_matches_the_encoder() {
	let mut d = Mp4Demuxer::new(memory(fixture("hevc/x265_444_odd.mp4"))).unwrap();
	check_format("x265_444_odd", &decode(&mut d), (131, 71), 8, PixelFormat::YUV444);
}

/// Range extension coding tools, from `tools/hevc_rext_streams.py`
/// (headers that enable them, random CABAC slice data): cross-component
/// prediction, implicit and explicit RDPCM, transform skip up to 16x16
/// with rotation and its own contexts, persistent Rice adaptation, CU
/// chroma QP offsets, SAO offset scaling, intra smoothing disabled,
/// lossless. References: FFmpeg.
#[test]
fn hevc_range_extension_tools_are_bit_exact() {
	for (name, depth, format) in [
		("rext_444", 8, PixelFormat::YUV444),
		("rext_444_lossless", 10, PixelFormat::YUV444),
		("rext_422_12", 12, PixelFormat::YUV422),
		("rext_420_10", 10, PixelFormat::YUV420),
		("rext_400_8", 8, PixelFormat::Gray),
	] {
		let mut d = Mp4Demuxer::new(memory(fixture(&format!("hevc/{name}.mp4")))).unwrap();
		check_format(name, &decode(&mut d), (64, 64), depth, format);
	}
}

/// Profiles beyond 12 bits are refused, not decoded wrongly.
#[test]
fn hevc_16_bit_tools_are_rejected() {
	use rmp::codecs::hevc::ps::Sps;
	// Main 4:4:4 16 Intra SPS with extended_precision_processing_flag.
	let mut sps = sps_with_range_extension(0b0_0001_0000);
	assert!(Sps::parse(&sps).unwrap_err().0.contains("unsupported"));
	sps = sps_with_range_extension(0b0_0000_0001);
	assert!(Sps::parse(&sps).unwrap_err().0.contains("unsupported"));
	sps = sps_with_range_extension(0b1_1100_0110);
	let sps = Sps::parse(&sps).unwrap();
	assert!(sps.transform_skip_rotation && sps.implicit_rdpcm && !sps.explicit_rdpcm);
	assert!(sps.persistent_rice_adaptation && sps.high_precision_offsets);
	assert_eq!((sps.chroma_format_idc, sps.hshift, sps.vshift), (3, [0; 3], [0; 3]));
}

/// A 64x64 4:4:4 8-bit SPS whose range extension flags are the 9 low bits
/// of `flags` (first flag in the highest bit).
fn sps_with_range_extension(flags: u32) -> Vec<u8> {
	let mut b = rmp::codecs::bitstream::BitWriter::new();
	b.put(0x4201, 16); // NAL header: SPS
	b.put(0, 4);
	b.put(0, 3);
	b.put(1, 1);
	// profile_tier_level: RExt.
	b.put(4, 8);
	b.put(1 << 27, 32);
	b.put(0b1001, 4);
	b.put(0, 32);
	b.put(0, 11);
	b.put(0, 1);
	b.put(120, 8);
	b.ue(0);
	b.ue(3);
	b.put(0, 1); // separate_colour_plane_flag
	b.ue(64);
	b.ue(64);
	b.put(0, 1);
	b.ue(0);
	b.ue(0);
	b.ue(4);
	b.put(1, 1);
	b.ue(0);
	b.ue(0);
	b.ue(0);
	b.ue(0); // min CB 8
	b.ue(3); // CTB 64
	b.ue(0); // min TB 4
	b.ue(3); // max TB 32
	b.ue(1);
	b.ue(1);
	for _ in 0..4 {
		b.put(0, 1); // scaling lists, AMP, SAO, PCM
	}
	b.ue(0); // no short-term RPS
	b.put(0, 1);
	b.put(0, 1);
	b.put(0, 1);
	b.put(0, 1); // VUI
	b.put(1, 1); // sps_extension_present_flag
	b.put(0b1000_0000, 8);
	b.put(flags, 9);
	b.rbsp_trailing();
	b.finish()
}

fn demuxer_for(name: &str) -> Box<dyn Demuxer> {
	let data = memory(fixture(name));
	if name.ends_with(".mkv") {
		Box::new(MkvDemuxer::new(data).unwrap())
	} else {
		Box::new(Mp4Demuxer::new(data).unwrap())
	}
}

/// Decodes `name` with `threads` threads; `damage` flips bits as in
/// [`damaged_hevc_packets_do_not_panic`] (seed), errors ignored.
fn decode_threads(name: &str, threads: usize, damage: Option<u32>) -> Vec<(Vec<u8>, Option<u64>)> {
	rmp::codecs::threading::with_default_threads(threads, || {
		let mut d = demuxer_for(name);
		let stream = d.streams().by_kind(StreamKind::Video)[0].clone();
		let decoder = Resolver::new().decoder_for(&stream).unwrap();
		let mut state = decoder.create_state(&stream.parameters);
		let mut out = Vec::new();
		let mut push = |r: rmp::Result<rmp::core::DecodeOut>| {
			for f in r.map(|o| o.value).unwrap_or_default() {
				let Frame::Video(v) = f else { panic!("video expected") };
				out.push((v.data().to_vec(), v.pts.map(|t| t.ticks())));
			}
		};
		let mut n = damage.unwrap_or(0);
		while let Some(mut packet) = d.read().unwrap() {
			if packet.stream_id != stream.id {
				continue;
			}
			if damage.is_some() {
				for b in packet.data.iter_mut().skip(4) {
					n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
					if (n >> 16).is_multiple_of(61) {
						*b ^= (n >> 8) as u8;
					}
				}
			}
			push(decoder.decode(&mut state, packet));
		}
		push(decoder.flush(&mut state));
		out
	})
}

const ALL_FIXTURES: [&str; 13] = [
	"hevc/x265_main.mp4",
	"hevc/x265_main10.mkv",
	"hevc/x265_tools.mp4",
	"hevc/kvazaar_tiles_wpp.mp4",
	"hevc/x265_wpp.mp4",
	"hevc/x265_wpp_ctu16_slices.mkv",
	"hevc/x265_400_12.mp4",
	"hevc/x265_422_10.mkv",
	"hevc/x265_422_ctu16.mp4",
	"hevc/x265_444_12.mp4",
	"hevc/x265_444_odd.mp4",
	"hevc/rext_444.mp4",
	"hevc/rext_444_lossless.mp4",
];

/// Frame (and WPP row) threading never changes the output.
#[test]
fn hevc_output_is_identical_for_every_thread_count() {
	for name in ALL_FIXTURES {
		let one = decode_threads(name, 1, None);
		assert!(!one.is_empty());
		for threads in [2, 3, 8] {
			let many = decode_threads(name, threads, None);
			assert_eq!(many.len(), one.len(), "{name}: frame count at {threads} threads");
			for (i, (a, b)) in one.iter().zip(&many).enumerate() {
				assert!(a == b, "{name}: frame {i} differs at {threads} threads");
			}
		}
	}
}

/// Damaged streams decode the same way at every thread count too.
#[test]
fn damaged_hevc_output_is_identical_for_every_thread_count() {
	for name in ALL_FIXTURES {
		for seed in 0..6u32 {
			let one = decode_threads(name, 1, Some(seed));
			for threads in [2, 4] {
				let many = decode_threads(name, threads, Some(seed));
				assert!(one == many, "{name} seed {seed}: output differs at {threads} threads");
			}
		}
	}
}

/// Decodes `name` through [`HevcDecoderState`] with explicit frame and
/// WPP row thread counts.
fn decode_threading(
	name: &str,
	frames: usize,
	rows: usize,
	damage: Option<u32>,
) -> (Vec<Vec<u8>>, u64) {
	use rmp::codecs::hevc::HevcDecoderState;
	let mut d = demuxer_for(name);
	let stream = d.streams().by_kind(StreamKind::Video)[0].clone();
	let mut dec = HevcDecoderState::with_threading(&stream.parameters.extradata, frames, rows);
	let mut out = Vec::new();
	let mut n = damage.unwrap_or(0);
	while let Some(mut packet) = d.read().unwrap() {
		if packet.stream_id != stream.id {
			continue;
		}
		if damage.is_some() {
			for b in packet.data.iter_mut().skip(4) {
				n = n.wrapping_mul(1_103_515_245).wrapping_add(12345);
				if (n >> 16).is_multiple_of(61) {
					*b ^= (n >> 8) as u8;
				}
			}
		}
		out.extend(dec.decode_packet(&packet.data, packet.pts).into_iter().map(|p| p.data));
	}
	out.extend(dec.drain().into_iter().map(|p| p.data));
	(out, dec.damaged_slices)
}

/// WPP rows on several threads, with and without frame threads, give the
/// single-threaded output, also on damaged input (same damage count too).
#[test]
fn hevc_wpp_row_threads_match_one_thread() {
	for name in ALL_FIXTURES {
		let one = decode_threading(name, 1, 1, None);
		for (f, r) in [(1, 2), (1, 3), (1, 8), (2, 2), (3, 1), (4, 4)] {
			let many = decode_threading(name, f, r, None);
			assert!(one == many, "{name}: differs with {f} frame / {r} row threads");
		}
		for seed in 0..8u32 {
			let one = decode_threading(name, 1, 1, Some(seed));
			for (f, r) in [(1, 3), (3, 2)] {
				let many = decode_threading(name, f, r, Some(seed));
				assert!(one == many, "{name} seed {seed}: differs with {f} frame / {r} row threads");
			}
		}
	}
}
