use crate::common::{fixture, memory, read_all};
use rmp::codecs::mp3::FrameHeader;
use rmp::container::mp3::demuxer::Mp3Demuxer;
use rmp::core::{Demuxer, RawValue, StreamId};

#[test]
fn reads_stream_parameters_and_all_frames() {
	let mut demuxer = Mp3Demuxer::new(memory(fixture("mp3/cbr_44k_stereo_tagged.mp3"))).unwrap();
	let stream = demuxer.streams().get(StreamId(0)).unwrap().clone();
	assert_eq!(stream.parameters.codec.id, "mp3");
	assert_eq!(stream.parameters.sample_rate, Some(44_100));
	assert_eq!(stream.parameters.channels, Some(2));

	let packets = read_all(&mut demuxer);
	// ffprobe -count_packets reports 40 (the Xing frame is not a packet).
	assert_eq!(packets.len(), 40);
	for (i, packet) in packets.iter().enumerate() {
		let header = FrameHeader::parse(&packet.data).expect("packet starts with a frame header");
		assert_eq!(header.frame_size, packet.data.len());
		assert_eq!(packet.pts.unwrap().ticks(), i as u64 * 1152);
		assert_eq!(packet.duration.unwrap().ticks(), 1152);
		assert!(packet.keyframe);
	}
	// Exact duration from the Xing frame count.
	assert_eq!(demuxer.duration().ticks(), 40 * 1152);
}

#[test]
fn reads_id3v2_metadata_and_gapless_info() {
	let demuxer = Mp3Demuxer::new(memory(fixture("mp3/cbr_44k_stereo_tagged.mp3"))).unwrap();
	let m = demuxer.metadata();
	assert_eq!(m.title.as_deref(), Some("RMP Fixture"));
	assert_eq!(m.artist.as_deref(), Some("RMP Tests"));
	assert_eq!(m.album.as_deref(), Some("Fixtures"));
	assert_eq!(m.genre.as_deref(), Some("Electronic"));
	assert_eq!(m.date.as_deref(), Some("2026"));
	assert_eq!(m.comment.as_deref(), Some("cbr fixture"));
	assert_eq!((m.track_number, m.tracks_total), (Some(3), Some(7)));

	let stream = demuxer.streams().get(StreamId(0)).unwrap();
	// LAME delay 576 + decoder delay 529.
	assert!(
		matches!(stream.metadata.raw.get("encoder_delay"), Some(RawValue::String(v)) if v == "1105")
	);
}

#[test]
fn excludes_id3v1_tag_from_audio_data() {
	let mut demuxer = Mp3Demuxer::new(memory(fixture("mp3/vbr_44k_xing_id3v1.mp3"))).unwrap();
	assert_eq!(demuxer.metadata().title.as_deref(), Some("VBR"));
	let packets = read_all(&mut demuxer);
	assert_eq!(packets.len(), 40);
	assert!(packets.iter().all(|p| FrameHeader::parse(&p.data).is_some()));
}

#[test]
fn stream_without_tags_or_xing_uses_cbr_estimate() {
	let mut demuxer = Mp3Demuxer::new(memory(fixture("mp3/mpeg2_22k_mono_raw.mp3"))).unwrap();
	let stream = demuxer.streams().get(StreamId(0)).unwrap().clone();
	assert_eq!(stream.parameters.sample_rate, Some(22_050));
	assert_eq!(stream.parameters.channels, Some(1));
	let packets = read_all(&mut demuxer);
	assert_eq!(packets.len(), 41);
	let estimated = demuxer.duration().as_seconds();
	let actual = 41.0 * 576.0 / 22_050.0;
	assert!((estimated - actual).abs() < 0.05, "estimate {estimated} vs {actual}");
}

#[test]
fn seek_lands_on_frame_boundary_near_target() {
	for path in ["mp3/cbr_44k_stereo_tagged.mp3", "mp3/vbr_44k_xing_id3v1.mp3"] {
		let mut demuxer = Mp3Demuxer::new(memory(fixture(path))).unwrap();
		demuxer.seek(0.5).unwrap();
		let packet = demuxer.read().unwrap().unwrap();
		assert!(FrameHeader::parse(&packet.data).is_some());
		let pts = packet.pts.unwrap().as_seconds();
		assert!((pts - 0.5).abs() <= 1152.0 / 44_100.0, "{path}: pts {pts}");

		demuxer.seek(0.0).unwrap();
		assert_eq!(demuxer.read().unwrap().unwrap().pts.unwrap().ticks(), 0);

		demuxer.seek(1000.0).unwrap();
		assert!(demuxer.read().unwrap().is_none());
		assert!(demuxer.seek(-1.0).is_err());
		assert!(demuxer.seek(f64::NAN).is_err());
	}
}
