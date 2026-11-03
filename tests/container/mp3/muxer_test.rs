use crate::common::{fixture, memory, read_all};
use rmp::container::mp3::demuxer::Mp3Demuxer;
use rmp::container::mp3::muxer::Mp3Muxer;
use rmp::core::{CodecId, Demuxer, Muxer, Packet, Stream, StreamId};
use rmp::io::MemoryIo;

fn remux(path: &str) -> (Mp3Demuxer, Vec<u8>) {
	let mut source = Mp3Demuxer::new(memory(fixture(path))).unwrap();
	let output = MemoryIo::new();
	let mut muxer = Mp3Muxer::new(Box::new(output.clone()));
	muxer.set_metadata(source.metadata().clone()).unwrap();
	muxer.add(source.streams().get(StreamId(0)).unwrap()).unwrap();
	while let Some(packet) = source.read().unwrap() {
		muxer.write(packet).unwrap();
	}
	muxer.finish().unwrap();
	source.seek(0.0).unwrap();
	(source, output.contents())
}

#[test]
fn round_trip_preserves_frames_metadata_and_duration() {
	for path in ["mp3/cbr_44k_stereo_tagged.mp3", "mp3/vbr_44k_xing_id3v1.mp3"] {
		let (mut source, bytes) = remux(path);
		let mut copy = Mp3Demuxer::new(memory(bytes)).unwrap();

		let original = read_all(&mut source);
		let remuxed = read_all(&mut copy);
		assert_eq!(original.len(), remuxed.len(), "{path}");
		for (a, b) in original.iter().zip(&remuxed) {
			assert_eq!(a.data, b.data);
			assert_eq!(a.pts, b.pts);
		}
		assert_eq!(copy.duration(), source.duration(), "info frame gives exact duration");
		assert_eq!(copy.metadata().title, source.metadata().title);
		assert_eq!(copy.metadata().track_number, source.metadata().track_number);
		let raw = |d: &Mp3Demuxer| {
			d.streams().get(StreamId(0)).unwrap().metadata.raw.get("encoder_delay").cloned()
		};
		assert_eq!(format!("{:?}", raw(&copy)), format!("{:?}", raw(&source)), "gapless info kept");
	}
}

#[test]
fn rejects_non_mpeg_streams_and_packets() {
	let mut muxer = Mp3Muxer::new(Box::new(MemoryIo::new()));
	let pcm = Stream::audio(StreamId(0), 44_100, 2, CodecId::new("pcm_s16le"));
	assert!(muxer.add(&pcm).is_err());

	let mp3 = Stream::audio(StreamId(0), 44_100, 2, CodecId::new("mp3"));
	muxer.add(&mp3).unwrap();
	assert!(muxer.add(&mp3).is_err(), "only one stream");
	assert!(muxer.write(Packet::new(StreamId(0), vec![1, 2, 3, 4, 5])).is_err());
}
