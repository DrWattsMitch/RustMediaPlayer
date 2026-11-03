use crate::common::fixture;
use rmp::container::{detect, mp3::Mp3Probe, mp4::Mp4Probe, select_muxer};
use rmp::core::{Hint, Probe};
use rmp::io::MemoryIo;

#[test]
fn detects_each_format_from_its_first_bytes() {
	for (path, expected) in [
		("mp3/cbr_44k_stereo_tagged.mp3", "mp3"),
		("mp3/mpeg2_22k_mono_raw.mp3", "mp3"),
		("mp4/h264_bframes_aac.mp4", "h264"),
		("mp4/aac_tagged_cover.m4a", "aac"),
		("wav/wav_mono_8khz_tone_sample.wav", "pcm_s16le"),
	] {
		let data = fixture(path);
		let head = data[..data.len().min(4096)].to_vec();
		let demuxer = detect(&head, Box::new(MemoryIo::from_vec(data))).unwrap();
		assert_eq!(demuxer.streams().streams[0].parameters.codec.id, expected, "{path}");
	}
}

#[test]
fn probes_do_not_claim_other_formats() {
	let wav = fixture("wav/wav_mono_8khz_tone_sample.wav");
	let mp4 = fixture("mp4/h264_bframes_aac.mp4");
	let mp3 = fixture("mp3/mpeg2_22k_mono_raw.mp3");
	assert!(!Mp3Probe::demuxer_matches(&wav[..4096]));
	assert!(!Mp4Probe::demuxer_matches(&wav[..4096]));
	assert!(!Mp4Probe::demuxer_matches(&mp3[..4096]));
	assert!(!Mp3Probe::demuxer_matches(&[0u8; 4096]));
	assert!(!Mp3Probe::demuxer_matches(&mp4[..4096]) || Mp4Probe::demuxer_matches(&mp4[..4096]));
}

#[test]
fn selects_muxers_by_extension_and_mime() {
	for ext in ["mp4", "m4a", "MP4", "mp3"] {
		let hint = Hint { extension: Some(ext.into()), ..Default::default() };
		assert!(select_muxer(hint, Box::new(MemoryIo::new())).is_ok(), "{ext}");
	}
	assert!(Mp3Probe::muxer_matches(&Hint {
		mime_type: Some("audio/mpeg".into()),
		..Default::default()
	}));
	assert!(Mp4Probe::muxer_matches(&Hint {
		mime_type: Some("video/mp4".into()),
		..Default::default()
	}));
}
