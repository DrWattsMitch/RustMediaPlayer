use crate::common::{fixture, memory, read_all};
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Demuxer, RawValue, StreamId, StreamKind};

#[test]
fn reads_tracks_and_codec_configuration() {
	let demuxer = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	let video = demuxer.streams().get(StreamId(0)).unwrap();
	assert_eq!(video.kind, StreamKind::Video);
	assert_eq!(video.parameters.codec.id, "h264");
	assert_eq!((video.parameters.width, video.parameters.height), (Some(64), Some(48)));
	assert_eq!(video.parameters.extradata.first(), Some(&1), "avcC configurationVersion");

	let audio = demuxer.streams().get(StreamId(1)).unwrap();
	assert_eq!(audio.kind, StreamKind::Audio);
	assert_eq!(audio.parameters.codec.id, "aac");
	assert_eq!(audio.parameters.sample_rate, Some(48_000));
	assert_eq!(audio.parameters.channels, Some(1));
	assert!(!audio.parameters.extradata.is_empty(), "AudioSpecificConfig");
	assert!(
		matches!(audio.metadata.raw.get("encoder_delay"), Some(RawValue::String(v)) if v == "1024")
	);

	assert!((demuxer.duration().as_seconds() - 1.0).abs() < 0.05);
	assert_eq!(demuxer.metadata().title.as_deref(), Some("Clip"));
}

#[test]
fn packets_are_interleaved_and_ordered() {
	let mut demuxer = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	let packets = read_all(&mut demuxer);
	let video: Vec<_> = packets.iter().filter(|p| p.stream_id == StreamId(0)).collect();
	let audio: Vec<_> = packets.iter().filter(|p| p.stream_id == StreamId(1)).collect();
	// Counts reported by `ffprobe -count_packets`.
	assert_eq!(video.len(), 25);
	assert_eq!(audio.len(), 48);
	assert!(video[0].keyframe);
	assert!(video.iter().any(|p| !p.keyframe), "stss marks non-key frames");

	for stream in [&video, &audio] {
		for pair in stream.windows(2) {
			assert!(pair[0].dts.unwrap().ticks() <= pair[1].dts.unwrap().ticks());
		}
	}
	// B-frames: presentation order differs from decode order.
	let pts: Vec<u64> = video.iter().map(|p| p.pts.unwrap().ticks()).collect();
	assert!(pts.windows(2).any(|w| w[1] < w[0]));
	assert_eq!(pts.iter().min(), Some(&0), "edit list removes the B-frame delay");

	// Interleaving: decode times across tracks never go backwards by more
	// than one packet duration.
	let seconds: Vec<f64> = packets.iter().map(|p| p.dts.unwrap().as_seconds()).collect();
	assert!(seconds.windows(2).all(|w| w[1] + 0.05 >= w[0]));
}

#[test]
fn reads_itunes_metadata_and_cover() {
	let demuxer = Mp4Demuxer::new(memory(fixture("mp4/aac_tagged_cover.m4a"))).unwrap();
	let m = demuxer.metadata();
	assert_eq!(m.title.as_deref(), Some("Song"));
	assert_eq!(m.artist.as_deref(), Some("Band"));
	assert_eq!(m.album.as_deref(), Some("Record"));
	assert_eq!(m.genre.as_deref(), Some("Jazz"));
	assert_eq!(m.date.as_deref(), Some("2023"));
	assert_eq!(m.comment.as_deref(), Some("m4a fixture"));
	assert_eq!((m.track_number, m.tracks_total), (Some(2), Some(9)));
	assert_eq!((m.disc_number, m.discs_total), (Some(1), Some(2)));
	assert_eq!(m.images.len(), 1);
	assert_eq!(m.images[0].mime_type, "image/png");
	assert!(m.images[0].data.starts_with(b"\x89PNG"));
}

#[test]
fn seek_starts_video_on_keyframe() {
	let mut demuxer = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	demuxer.seek(0.7).unwrap();
	let packets = read_all(&mut demuxer);
	let first_video = packets.iter().find(|p| p.stream_id == StreamId(0)).unwrap();
	assert!(first_video.keyframe);
	let pts = first_video.pts.unwrap().as_seconds();
	assert!((0.3..=0.7).contains(&pts), "keyframe at {pts}"); // GOP of 10 frames at 25 fps
	let first_audio = packets.iter().find(|p| p.stream_id == StreamId(1)).unwrap();
	assert!(first_audio.dts.unwrap().as_seconds() <= pts + 0.05);

	assert!(demuxer.seek(f64::INFINITY).is_err());
}
