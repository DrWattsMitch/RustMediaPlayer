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
