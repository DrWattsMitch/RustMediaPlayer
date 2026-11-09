use crate::common::{fixture, memory, read_all};
use rmp::container::y4m::demuxer::Y4mDemuxer;
use rmp::core::{Demuxer, Frame, RawValue, Resolver, StreamKind, TimeBase};

#[test]
fn reads_header_and_frames_like_ffmpeg() {
	let mut demuxer = Y4mDemuxer::new(memory(fixture("y4m/yuv420p_34x18.y4m"))).unwrap();
	let stream = demuxer.streams().streams[0].clone();
	assert_eq!(stream.kind, StreamKind::Video);
	assert_eq!(stream.parameters.codec.id, "rawvideo");
	assert_eq!(stream.parameters.pixel_format.as_deref(), Some("yuv420p"));
	assert_eq!((stream.parameters.width, stream.parameters.height), (Some(34), Some(18)));
	assert_eq!(stream.time_base, TimeBase::new(1001, 30000));
	assert!(
		matches!(stream.metadata.raw.get("chroma_location"), Some(RawValue::String(v)) if v == "center")
	);
	assert_eq!(demuxer.duration().ticks(), 3);

	let packets = read_all(&mut demuxer);
	assert_eq!(packets.len(), 3);
	for (i, p) in packets.iter().enumerate() {
		assert_eq!(p.pts.map(|t| t.ticks()), Some(i as u64));
		assert_eq!(p.data.len(), 34 * 18 + 2 * 17 * 9);
	}
	let all: Vec<u8> = packets.into_iter().flat_map(|p| p.data).collect();
	assert_eq!(all, fixture("y4m/yuv420p_34x18.ffmpeg.yuv"));
}

#[test]
fn high_bit_depth_frames_decode_through_the_raw_codec() {
	let mut demuxer = Y4mDemuxer::new(memory(fixture("y4m/yuv422p10_16x8.y4m"))).unwrap();
	let stream = demuxer.streams().streams[0].clone();
	assert_eq!(stream.parameters.pixel_format.as_deref(), Some("yuv422p10le"));

	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let packet = demuxer.read().unwrap().unwrap();
	let Frame::Video(frame) = &decoder.decode(&mut state, packet).unwrap().value[0] else {
		panic!("expected video");
	};
	assert_eq!((frame.width, frame.height, frame.pixel.depth), (16, 8, 10));
	assert_eq!(frame.pixel.name(), "yuv422p10le");
	assert_eq!(frame.data, fixture("y4m/yuv422p10_16x8.ffmpeg.yuv")[..512]);
}

#[test]
fn seeks_by_frame() {
	let reference = fixture("y4m/yuv420p_34x18.ffmpeg.yuv");
	let frame = reference.len() / 3;
	let mut demuxer = Y4mDemuxer::new(memory(fixture("y4m/yuv420p_34x18.y4m"))).unwrap();
	demuxer.seek(2.0 * 1001.0 / 30000.0 + 1e-6).unwrap();
	let packet = demuxer.read().unwrap().unwrap();
	assert_eq!(packet.pts.map(|t| t.ticks()), Some(2));
	assert_eq!(packet.data, reference[2 * frame..]);
	assert!(demuxer.read().unwrap().is_none());

	demuxer.seek(0.0).unwrap();
	assert_eq!(read_all(&mut demuxer).len(), 3);
	demuxer.seek(1e9).unwrap();
	assert!(demuxer.read().unwrap().is_none());
	assert!(demuxer.seek(f64::NAN).is_err());
}

#[test]
fn truncated_file_ends_at_last_whole_frame() {
	let mut data = fixture("y4m/yuv420p_34x18.y4m");
	data.truncate(data.len() - 10);
	let mut demuxer = Y4mDemuxer::new(memory(data)).unwrap();
	assert_eq!(read_all(&mut demuxer).len(), 2);
}

#[test]
fn per_frame_parameters_are_skipped_and_bad_frame_headers_rejected() {
	let mut data = b"YUV4MPEG2 W2 H2 F25:1 C444\nFRAME Ixyz\n".to_vec();
	data.extend_from_slice(&[1; 12]);
	data.extend_from_slice(b"FRAMX\n");
	data.extend_from_slice(&[2; 12]);
	let mut demuxer = Y4mDemuxer::new(memory(data)).unwrap();
	assert_eq!(demuxer.read().unwrap().unwrap().data, vec![1; 12]);
	assert!(demuxer.read().is_err());
}

#[test]
fn rejects_invalid_streams() {
	for data in [
		&b""[..],
		b"YUV4MPEG2 W2 H2",
		b"YUV4MPEG2 W2\n",
		b"YUV4MPEG2 W2 H2 Cnope\n",
		b"YUV4MPEG2 W65536 H65536\n",
	] {
		assert!(Y4mDemuxer::new(memory(data.to_vec())).is_err(), "{}", String::from_utf8_lossy(data));
	}
	let long = [b"YUV4MPEG2 W2 H2 X".as_slice(), &[b'a'; 2000], b"\n"].concat();
	assert!(Y4mDemuxer::new(memory(long)).is_err());
}
