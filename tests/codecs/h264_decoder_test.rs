use crate::common::{fixture, memory, read_all};
use rmp::codecs::h264::{H264Decoder, H264DecoderState};
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Decoder, Demuxer, Frame, Keyframe, Packet, Stream, StreamKind};

fn video(path: &str) -> (Stream, Vec<Packet>) {
	let mut demuxer = Mp4Demuxer::new(memory(fixture(path))).unwrap();
	let stream = demuxer.streams().iter().find(|s| s.kind == StreamKind::Video).unwrap().clone();
	let packets = read_all(&mut demuxer).into_iter().filter(|p| p.stream_id == stream.id).collect();
	(stream, packets)
}

/// Decodes through the Decoder trait, returning every frame in output order.
fn decode(path: &str) -> Vec<rmp::core::VideoFrame> {
	let (stream, packets) = video(path);
	let decoder = H264Decoder::new();
	let mut state = decoder.create_state(&stream.parameters);
	let mut frames = Vec::new();
	for packet in packets {
		frames.extend(decoder.decode(&mut state, packet).unwrap().value);
	}
	frames.extend(decoder.flush(&mut state).unwrap().value);
	frames
		.into_iter()
		.map(|f| match f {
			Frame::Video(v) => v,
			_ => panic!("video expected"),
		})
		.collect()
}

fn assert_bit_exact(path: &str, reference: &str, width: u32, height: u32, count: usize) {
	let frames = decode(path);
	let reference = fixture(reference);
	let size = (width * height * 3 / 2) as usize;
	assert_eq!(frames.len(), count);
	assert_eq!(reference.len(), size * count);
	for (i, (frame, want)) in frames.iter().zip(reference.chunks_exact(size)).enumerate() {
		assert_eq!((frame.width, frame.height), (width, height));
		assert!(frame.data() == want, "frame {i} differs from FFmpeg");
	}
	// Display order: presentation timestamps strictly increase.
	let pts: Vec<u64> = frames.iter().map(|f| f.pts.unwrap().ticks()).collect();
	assert!(pts.windows(2).all(|w| w[0] < w[1]), "pts not increasing: {pts:?}");
	assert_eq!(frames[0].keyframe, Keyframe::Key);
}

#[test]
fn cavlc_baseline_with_slices_is_bit_exact() {
	assert_bit_exact("mp4/h264_cavlc_baseline.mp4", "mp4/h264_cavlc_baseline.ffmpeg.yuv", 64, 48, 12);
}

#[test]
fn high_profile_tools_are_bit_exact() {
	// CABAC, B-pyramid, temporal direct, explicit + implicit weighted
	// prediction, 8x8 transform, JVT scaling matrices, 2 slices, cropping.
	assert_bit_exact("mp4/h264_high_tools.mp4", "mp4/h264_high_tools.ffmpeg.yuv", 72, 40, 12);
}

#[test]
fn reorders_b_frames_and_keeps_all_frames() {
	let frames = decode("mp4/h264_bframes_aac.mp4");
	assert_eq!(frames.len(), 25);
	assert!(
		frames.iter().all(|f| f.width == 64 && f.height == 48 && f.data().len() == 64 * 48 * 3 / 2)
	);
	let pts: Vec<u64> = frames.iter().map(|f| f.pts.unwrap().ticks()).collect();
	assert!(pts.windows(2).all(|w| w[0] < w[1]));
}

/// Re-frames length-prefixed MP4 packets as an Annex B byte stream.
fn to_annex_b(extradata: &[u8], packets: &[Packet]) -> Vec<Vec<u8>> {
	let cfg = rmp::codecs::h264::nal::AvcConfig::parse(extradata).unwrap();
	let mut out = Vec::new();
	for (i, p) in packets.iter().enumerate() {
		let mut au = Vec::new();
		if i == 0 {
			for ps in cfg.sps.iter().chain(cfg.pps.iter()) {
				au.extend_from_slice(&[0, 0, 0, 1]);
				au.extend_from_slice(ps);
			}
		}
		for nal in rmp::codecs::h264::nal::split_length_prefixed(&p.data, cfg.nal_length_size) {
			au.extend_from_slice(&[0, 0, 0, 1]);
			au.extend_from_slice(nal);
		}
		out.push(au);
	}
	out
}

#[test]
fn annex_b_input_without_extradata_decodes_identically() {
	let (stream, packets) = video("mp4/h264_high_tools.mp4");
	let mut dec = H264DecoderState::new(&[]);
	let mut pictures = Vec::new();
	for (au, p) in to_annex_b(&stream.parameters.extradata, &packets).iter().zip(&packets) {
		pictures.extend(dec.decode_packet(au, p.pts).unwrap());
	}
	pictures.extend(dec.flush());
	let reference = fixture("mp4/h264_high_tools.ffmpeg.yuv");
	assert_eq!(pictures.len(), 12);
	for (pic, want) in pictures.iter().zip(reference.chunks_exact(72 * 40 * 3 / 2)) {
		assert!(pic.data() == want);
	}
}

#[test]
fn reset_after_seek_resumes_at_next_keyframe() {
	let (stream, packets) = video("mp4/h264_bframes_aac.mp4");
	let mut dec = H264DecoderState::new(&stream.parameters.extradata);
	for p in &packets[..5] {
		dec.decode_packet(&p.data, p.pts).unwrap();
	}
	dec.reset();
	let key = packets.iter().position(|p| p.keyframe).unwrap();
	let mut out = Vec::new();
	for p in &packets[key..] {
		out.extend(dec.decode_packet(&p.data, p.pts).unwrap());
	}
	out.extend(dec.flush());
	assert_eq!(out.len(), packets.len() - key);
}

#[test]
fn garbage_without_parameter_sets_produces_nothing() {
	let mut dec = H264DecoderState::new(&[]);
	let out = dec.decode_packet(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00], None).unwrap_or_default();
	assert!(out.is_empty());
	assert!(dec.flush().is_empty());
}

/// Decodes `packets` with `threads` frame threads; returns the pictures and
/// the damaged-slice count.
fn decode_threaded(stream: &Stream, packets: &[Packet], threads: usize) -> (Vec<Vec<u8>>, u64) {
	let mut dec = H264DecoderState::with_threads(&stream.parameters.extradata, threads);
	assert_eq!(dec.threads(), threads.max(1));
	let mut out = Vec::new();
	for p in packets {
		out.extend(dec.decode_packet(&p.data, p.pts).unwrap_or_default().into_iter().map(|p| p.data()));
	}
	out.extend(dec.flush().into_iter().map(|p| p.data()));
	(out, dec.damaged_slices)
}

#[test]
fn frame_threading_is_bit_exact_with_single_threaded_decoding() {
	for path in ["mp4/h264_cavlc_baseline.mp4", "mp4/h264_high_tools.mp4", "mp4/h264_bframes_aac.mp4"]
	{
		let (stream, packets) = video(path);
		let (serial, damaged) = decode_threaded(&stream, &packets, 1);
		assert!(!serial.is_empty());
		for threads in [2, 3, 8] {
			let (threaded, d) = decode_threaded(&stream, &packets, threads);
			assert_eq!(threaded.len(), serial.len(), "{path}: {threads} threads");
			assert!(threaded == serial, "{path}: {threads} threads differ");
			assert_eq!(d, damaged, "{path}: {threads} threads");
		}
	}
}

#[test]
fn frame_threading_conceals_damage_like_single_threaded_decoding() {
	// Corrupt the second half of every third access unit (slices then fail
	// or decode garbage, and some are lost): the pictures that reference
	// them must still wait for and match them.
	for path in ["mp4/h264_high_tools.mp4", "mp4/h264_bframes_aac.mp4", "mp4/h264_cavlc_baseline.mp4"]
	{
		let (stream, mut packets) = video(path);
		let (clean, _) = decode_threaded(&stream, &packets, 1);
		for (i, p) in packets.iter_mut().enumerate() {
			if i % 3 == 1 {
				let n = p.data.len();
				for (k, b) in p.data[n / 2..].iter_mut().enumerate() {
					*b ^= (k as u8).wrapping_mul(37) | 1;
				}
			}
		}
		let (serial, damaged) = decode_threaded(&stream, &packets, 1);
		assert!(serial != clean, "{path}: the corruption should change the output");
		for threads in [2, 4] {
			let (threaded, d) = decode_threaded(&stream, &packets, threads);
			assert!(threaded == serial, "{path}: {threads} threads differ");
			assert_eq!(d, damaged);
		}
	}
}

/// Frames are views into the decoder's pictures: no copy on output, and the
/// sample buffers are recycled once neither the DPB nor the caller holds
/// them, so a long decode allocates a bounded number of planes.
#[test]
fn output_frames_share_and_recycle_decoder_buffers() {
	let (stream, packets) = video("mp4/h264_high_tools.mp4");
	let want = decode("mp4/h264_high_tools.mp4");
	for threads in [1, 3] {
		let mut dec = H264DecoderState::with_threads(&stream.parameters.extradata, threads);
		let mut pictures = 0;
		let mut i = 0;
		// Decode the clip three times over (an IDR restarts each loop), dropping
		// every frame as soon as it has been checked.
		for _ in 0..3 {
			for p in &packets {
				for pic in dec.decode_packet(&p.data, p.pts).unwrap() {
					let frame = pic.into_frame();
					assert!(frame.data() == want[i % want.len()].data(), "{threads} threads: frame {i}");
					// Cropped views into a macroblock-aligned picture.
					let y = frame.plane(0).unwrap();
					assert_eq!(y.stride() % 16, 0);
					assert!(y.stride() >= y.row_bytes());
					let copy = frame.clone();
					assert!(copy.plane(0).unwrap().buffer().ptr_eq(y.buffer()), "clone shares the planes");
					i += 1;
					pictures += 1;
				}
			}
		}
		pictures += dec.flush().len();
		let stats = dec.pool_stats();
		// 3 planes per picture; the DPB of this clip holds a handful of
		// pictures, frame threads keep `threads` more in flight (each with a
		// published copy), plus a few transient ones.
		let bound = 3 * (2 * (16 + threads as u64) + 4);
		assert!(stats.allocated <= bound, "{threads} threads: {stats:?} for {pictures} pictures");
		assert!(stats.reused > stats.allocated, "{threads} threads: buffers are reused ({stats:?})");
	}
}

/// Output frames keep their samples alive after the decoder is gone.
#[test]
fn frames_outlive_the_decoder() {
	let (stream, packets) = video("mp4/h264_cavlc_baseline.mp4");
	let mut dec = H264DecoderState::with_threads(&stream.parameters.extradata, 2);
	let mut frames = Vec::new();
	for p in &packets {
		frames.extend(dec.decode_packet(&p.data, p.pts).unwrap().into_iter().map(|p| p.into_frame()));
	}
	frames.extend(dec.flush().into_iter().map(|p| p.into_frame()));
	drop(dec);
	let want = fixture("mp4/h264_cavlc_baseline.ffmpeg.yuv");
	let size = frames[0].packed_len();
	for (i, f) in frames.iter().enumerate() {
		assert!(f.data()[..] == want[i * size..(i + 1) * size], "frame {i}");
	}
}

/// FNV-1a 64 of the decoded planar output (as `tools/make_h264_fixtures.sh`
/// computes it from FFmpeg's).
fn fnv1a(data: &[u8]) -> u64 {
	data
		.iter()
		.fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// Decodes a raw Annex B stream (one packet) with `threads` frame threads.
fn decode_annex_b(path: &str, threads: usize) -> (Vec<rmp::codecs::h264::DecodedPicture>, u64) {
	let data = fixture(path);
	let mut dec = H264DecoderState::with_threads(&[], threads);
	let mut pictures = dec.decode_packet(&data, None).unwrap();
	pictures.extend(dec.flush());
	(pictures, dec.damaged_slices)
}

struct Expect {
	path: &'static str,
	size: (u32, u32),
	bit_depth: u32,
	chroma_422: bool,
	/// FNV-1a of FFmpeg's planar output.
	hash: u64,
}

const HIGH_PROFILES: &[Expect] = &[
	// High 10: CABAC, B-pyramid, explicit weighted prediction, cropping.
	Expect {
		path: "h264/h264_high10.264",
		size: (120, 88),
		bit_depth: 10,
		chroma_422: false,
		hash: 0xccd6_1130_14b4_9b5d,
	},
	// High 4:2:2, 10-bit: 2x4 chroma DC (CABAC), 8x16 chroma prediction.
	Expect {
		path: "h264/h264_high422_10.264",
		size: (120, 88),
		bit_depth: 10,
		chroma_422: true,
		hash: 0x900f_a4d8_9b72_df50,
	},
	// High 4:2:2, 8-bit CAVLC: the 4:2:2 chroma DC coeff_token tables.
	Expect {
		path: "h264/h264_high422_cavlc.264",
		size: (120, 88),
		bit_depth: 8,
		chroma_422: true,
		hash: 0xb255_576a_a291_506e,
	},
];

const INTERLACED: &[Expect] = &[
	// MBAFF frames (adaptive frame/field pairs), CABAC, B-pyramid, weights.
	Expect {
		path: "h264/h264_mbaff.264",
		size: (128, 96),
		bit_depth: 8,
		chroma_422: false,
		hash: 0xddd0_0037_be5d_2494,
	},
	// MBAFF, CAVLC, 4:2:2 10-bit, temporal direct (broadcast contribution).
	Expect {
		path: "h264/h264_mbaff_422_10_cavlc.264",
		size: (128, 96),
		bit_depth: 10,
		chroma_422: true,
		hash: 0x37bd_f980_dae5_9382,
	},
	// PAFF field pictures, bottom field first, B fields with temporal direct.
	Expect {
		path: "h264/h264_paff_bff.264",
		size: (128, 96),
		bit_depth: 8,
		chroma_422: false,
		hash: 0x360a_44b6_4bf8_8170,
	},
	// MBAFF frames and field pairs referencing each other (colocated
	// macroblocks across structures).
	Expect {
		path: "h264/h264_paff_mbaff_mix.264",
		size: (128, 96),
		bit_depth: 8,
		chroma_422: false,
		hash: 0xf143_c5ec_ee76_8a2d,
	},
];

fn check(e: &Expect) {
	let (pictures, damaged) = decode_annex_b(e.path, 1);
	assert_eq!(pictures.len(), 12, "{}", e.path);
	assert_eq!(damaged, 0, "{}", e.path);
	let mut out = Vec::new();
	for p in &pictures {
		assert_eq!((p.width, p.height), e.size, "{}", e.path);
		assert_eq!((p.bit_depth, p.chroma_422), (e.bit_depth, e.chroma_422), "{}", e.path);
		out.extend(p.data());
	}
	let bytes = if e.bit_depth > 8 { 2 } else { 1 };
	let (w, h) = (e.size.0 as usize, e.size.1 as usize);
	let chroma = if e.chroma_422 { w * h } else { w * h / 2 };
	assert_eq!(out.len(), 12 * (w * h + chroma) * bytes, "{}", e.path);
	assert_eq!(fnv1a(&out), e.hash, "{} differs from FFmpeg", e.path);
}

#[test]
fn high10_and_high422_are_bit_exact() {
	HIGH_PROFILES.iter().for_each(check);
}

#[test]
fn interlaced_paff_and_mbaff_are_bit_exact() {
	INTERLACED.iter().for_each(check);
}

#[test]
fn interlaced_and_high_bit_depth_frame_threading_is_bit_exact() {
	for e in HIGH_PROFILES.iter().chain(INTERLACED) {
		let (serial, _) = decode_annex_b(e.path, 1);
		let serial: Vec<Vec<u8>> = serial.iter().map(|p| p.data()).collect();
		for threads in [2, 4] {
			let (threaded, damaged) = decode_annex_b(e.path, threads);
			let threaded: Vec<Vec<u8>> = threaded.iter().map(|p| p.data()).collect();
			assert!(threaded == serial, "{}: {threads} threads differ", e.path);
			assert_eq!(damaged, 0);
		}
	}
}

/// Through the Decoder trait the frames carry their pixel format.
#[test]
fn decoder_trait_reports_depth_and_chroma_format() {
	use rmp::core::video::PixelFormat;
	for (path, depth, format) in [
		("h264/h264_high422_10.264", 10, PixelFormat::YUV422),
		("h264/h264_mbaff.264", 8, PixelFormat::YUV420),
	] {
		let decoder = H264Decoder::new();
		let id = rmp::core::StreamId(0);
		let stream = Stream::video(id, 0, 0, "yuv420p", rmp::core::CodecId::new("h264"));
		let mut state = decoder.create_state(&stream.parameters);
		let packet = Packet::new(id, fixture(path));
		let mut frames = decoder.decode(&mut state, packet).unwrap().value;
		frames.extend(decoder.flush(&mut state).unwrap().value);
		assert_eq!(frames.len(), 12, "{path}");
		for f in frames {
			let Frame::Video(v) = f else { panic!("video expected") };
			assert_eq!((v.pixel.depth, v.pixel.format), (depth, format), "{path}");
			let bytes = if depth > 8 { 2 } else { 1 };
			assert_eq!(v.plane(0).unwrap().row_bytes(), v.width as usize * bytes, "{path}");
		}
	}
}

/// Damaged interlaced and high bit depth streams: no panic, and frame
/// threads conceal exactly like one thread.
#[test]
fn damaged_interlaced_streams_conceal_identically_with_threads() {
	let mut rng = crate::common::Rng::new(7);
	for e in HIGH_PROFILES.iter().chain(INTERLACED) {
		for round in 0..3 {
			let mut data = fixture(e.path);
			let n = data.len();
			for _ in 0..8 + round * 8 {
				let at = 64 + (rng.next() as usize) % (n - 64);
				data[at] ^= 1 << (rng.next() % 8);
			}
			let run = |threads: usize| {
				let mut dec = H264DecoderState::with_threads(&[], threads);
				let mut out: Vec<Vec<u8>> =
					dec.decode_packet(&data, None).unwrap_or_default().iter().map(|p| p.data()).collect();
				out.extend(dec.flush().iter().map(|p| p.data()));
				out
			};
			let serial = run(1);
			assert!(run(3) == serial, "{} round {round}: threads differ", e.path);
		}
	}
}
