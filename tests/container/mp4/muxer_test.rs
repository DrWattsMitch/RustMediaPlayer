use crate::common::{fixture, memory, read_all};
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::mp4::muxer::Mp4Muxer;
use rmp::core::{CodecId, Demuxer, Muxer, Packet, Stream, StreamId, Time, TimeBase};
use rmp::io::MemoryIo;

fn remux(path: &str) -> (Mp4Demuxer, Mp4Demuxer) {
	let mut source = Mp4Demuxer::new(memory(fixture(path))).unwrap();
	let output = MemoryIo::new();
	let mut muxer = Mp4Muxer::new(Box::new(output.clone()));
	muxer.set_metadata(source.metadata().clone()).unwrap();
	for stream in source.streams().iter() {
		muxer.add(stream).unwrap();
	}
	while let Some(packet) = source.read().unwrap() {
		muxer.write(packet).unwrap();
	}
	muxer.finish().unwrap();
	source.seek(0.0).unwrap();
	let copy = Mp4Demuxer::new(memory(output.contents())).unwrap();
	(source, copy)
}

#[test]
fn round_trip_preserves_samples_timing_and_config() {
	let (mut source, mut copy) = remux("mp4/h264_bframes_aac.mp4");
	for (a, b) in source.streams().iter().zip(copy.streams().iter()) {
		assert_eq!(a.parameters.codec, b.parameters.codec);
		assert_eq!(a.parameters.extradata, b.parameters.extradata);
		assert_eq!(a.parameters.sample_rate, b.parameters.sample_rate);
		assert_eq!(a.parameters.width, b.parameters.width);
	}
	let original = read_all(&mut source);
	let remuxed = read_all(&mut copy);
	assert_eq!(original.len(), remuxed.len());
	for stream in [StreamId(0), StreamId(1)] {
		let a: Vec<_> = original.iter().filter(|p| p.stream_id == stream).collect();
		let b: Vec<_> = remuxed.iter().filter(|p| p.stream_id == stream).collect();
		assert_eq!(a.len(), b.len());
		for (x, y) in a.iter().zip(&b) {
			assert_eq!(x.data, y.data);
			assert_eq!(x.keyframe, y.keyframe);
			assert_eq!(x.pts.map(|t| t.as_seconds()), y.pts.map(|t| t.as_seconds()));
		}
	}
	assert!((source.duration().as_seconds() - copy.duration().as_seconds()).abs() < 0.002);
	assert_eq!(copy.metadata().title.as_deref(), Some("Clip"));
}

#[test]
fn round_trip_preserves_itunes_metadata() {
	let (source, copy) = remux("mp4/aac_tagged_cover.m4a");
	let (a, b) = (source.metadata(), copy.metadata());
	assert_eq!(a.title, b.title);
	assert_eq!(a.album, b.album);
	assert_eq!((a.track_number, a.tracks_total), (b.track_number, b.tracks_total));
	assert_eq!((a.disc_number, a.discs_total), (b.disc_number, b.discs_total));
	assert_eq!(a.images.len(), b.images.len());
	assert_eq!(a.images[0].data, b.images[0].data);
	assert_eq!(source.duration(), copy.duration(), "edit list (priming + padding) kept");
}

#[test]
fn muxes_mp3_frames_into_mp4() {
	let mut source =
		rmp::container::mp3::demuxer::Mp3Demuxer::new(memory(fixture("mp3/cbr_44k_stereo_tagged.mp3")))
			.unwrap();
	let output = MemoryIo::new();
	let mut muxer = Mp4Muxer::new(Box::new(output.clone()));
	muxer.add(source.streams().get(StreamId(0)).unwrap()).unwrap();
	let mut count = 0;
	while let Some(packet) = source.read().unwrap() {
		muxer.write(packet).unwrap();
		count += 1;
	}
	muxer.finish().unwrap();

	let mut copy = Mp4Demuxer::new(memory(output.contents())).unwrap();
	assert_eq!(copy.streams().get(StreamId(0)).unwrap().parameters.codec.id, "mp3");
	assert_eq!(read_all(&mut copy).len(), count);
}

#[test]
fn rejects_invalid_input() {
	let mut muxer = Mp4Muxer::new(Box::new(MemoryIo::new()));
	let pcm = Stream::audio(StreamId(0), 48_000, 2, CodecId::new("pcm_s16le"));
	assert!(muxer.add(&pcm).is_err(), "codec without an mp4 sample entry");

	let mut aac = Stream::audio(StreamId(0), 48_000, 2, CodecId::new("aac"));
	aac.parameters.extradata = vec![0x11, 0x90];
	muxer.add(&aac).unwrap();
	assert!(muxer.add(&aac).is_err(), "duplicate stream id");

	let tb = TimeBase::new(1, 48_000);
	let mut packet = Packet::new(StreamId(0), vec![0; 10]);
	packet.dts = Some(Time::new(2048, tb));
	muxer.write(packet.clone()).unwrap();
	packet.dts = Some(Time::new(1024, tb));
	assert!(muxer.write(packet).is_err(), "non-monotonic dts");
	assert!(muxer.write(Packet::new(StreamId(9), vec![0])).is_err(), "unknown stream");
}
