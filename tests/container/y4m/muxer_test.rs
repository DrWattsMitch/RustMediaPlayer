use crate::common::{fixture, memory, read_all};
use rmp::container::y4m::demuxer::Y4mDemuxer;
use rmp::container::y4m::muxer::Y4mMuxer;
use rmp::core::{CodecId, Demuxer, Muxer, Packet, RawValue, Stream, StreamId, TimeBase};
use rmp::io::MemoryIo;

#[test]
fn remux_is_byte_identical_to_ffmpeg() {
	for path in ["y4m/yuv420p_34x18.y4m", "y4m/yuv422p10_16x8.y4m"] {
		let source = fixture(path);
		let mut demuxer = Y4mDemuxer::new(memory(source.clone())).unwrap();
		let out = MemoryIo::new();
		let mut muxer = Y4mMuxer::new(Box::new(out.clone()));
		muxer.add(&demuxer.streams().streams[0]).unwrap();
		for packet in read_all(&mut demuxer) {
			muxer.write(packet).unwrap();
		}
		muxer.finish().unwrap();
		assert_eq!(out.contents(), source, "{path}");
	}
}

#[test]
fn writes_header_from_stream_fields() {
	let mut stream = Stream::video(StreamId(0), 4, 2, "yuv420p", CodecId::new("rawvideo"));
	stream.time_base = TimeBase::new(2002, 60000);
	for (k, v) in [
		("field_order", "tt"),
		("sample_aspect_ratio", "4:3"),
		("chroma_location", "topleft"),
		("color_range", "pc"),
	] {
		stream.metadata.raw.insert(k.into(), RawValue::String(v.into()));
	}
	let out = MemoryIo::new();
	let mut muxer = Y4mMuxer::new(Box::new(out.clone()));
	muxer.add(&stream).unwrap();
	muxer.write(Packet::new(StreamId(0), vec![9; 12])).unwrap();
	assert!(muxer.write(Packet::new(StreamId(0), vec![9; 11])).is_err());
	muxer.finish().unwrap();
	let expected =
		b"YUV4MPEG2 W4 H2 F30000:1001 It A4:3 C420paldv XYSCSS=420PALDV XCOLORRANGE=FULL\nFRAME\n";
	assert_eq!(out.contents(), [expected.as_slice(), &[9; 12]].concat());
}

#[test]
fn rejects_streams_y4m_cannot_carry() {
	let mut muxer = Y4mMuxer::new(Box::new(MemoryIo::new()));
	let h264 = Stream::video(StreamId(0), 4, 4, "yuv420p", CodecId::new("h264"));
	assert!(muxer.add(&h264).is_err());
	let audio = Stream::audio(StreamId(0), 48_000, 2, CodecId::new("pcm_s16le"));
	assert!(muxer.add(&audio).is_err());
	let rgb = Stream::video(StreamId(0), 4, 4, "rgb24", CodecId::new("rawvideo"));
	assert!(muxer.add(&rgb).is_err());
	// The raw codec's default id is the pixel format name.
	let raw = Stream::video(StreamId(0), 4, 4, "gray", CodecId::new("yuv420p"));
	assert!(muxer.add(&raw).is_ok());
	assert!(muxer.add(&raw).is_err(), "second stream");
}
