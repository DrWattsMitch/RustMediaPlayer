use crate::common::{fixture, memory, read_all};
use rmp::codecs::aac::{AacDecoder, AacDecoderState};
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Decoder, Demuxer, Frame, SampleFormat, StreamKind};

fn samples(bytes: &[u8]) -> Vec<i16> {
	bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Asserts every sample is within one LSB of FFmpeg and the SNR is high.
fn assert_matches(ours: &[i16], reference: &[i16]) {
	assert_eq!(ours.len(), reference.len(), "sample count");
	let (mut signal, mut noise) = (0f64, 0f64);
	for (i, (&a, &b)) in ours.iter().zip(reference).enumerate() {
		let diff = i32::from(a) - i32::from(b);
		assert!(diff.abs() <= 1, "sample {i} differs by {diff}");
		signal += f64::from(b).powi(2);
		noise += f64::from(diff).powi(2);
	}
	let snr = 10.0 * (signal / noise.max(1.0)).log10();
	assert!(snr > 90.0, "SNR {snr:.1} dB");
}

/// Decodes the first audio stream of an MP4 file through the Decoder trait.
fn decode_mp4(path: &str) -> (Vec<i16>, u8) {
	let mut demuxer = Mp4Demuxer::new(memory(fixture(path))).unwrap();
	let stream = demuxer.streams().iter().find(|s| s.kind == StreamKind::Audio).unwrap().clone();
	let decoder = AacDecoder::new();
	let mut state = decoder.create_state(&stream.parameters);
	let mut out = Vec::new();
	let mut channels = 0;
	for packet in read_all(&mut demuxer) {
		if packet.stream_id != stream.id {
			continue;
		}
		let pts = packet.pts;
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(frame) = frame else { panic!("audio expected") };
			assert_eq!(frame.format, SampleFormat::S16);
			assert_eq!(frame.nb_samples, 1024);
			assert_eq!(frame.pts, pts);
			channels = frame.channels.count();
			out.extend(samples(&frame.data));
		}
	}
	(out, channels)
}

#[test]
fn mono_lc_matches_ffmpeg() {
	let (ours, channels) = decode_mp4("mp4/aac_tagged_cover.m4a");
	assert_eq!(channels, 1);
	assert_matches(&ours, &samples(&fixture("mp4/aac_tagged_cover.ffmpeg.s16le")));
}

#[test]
fn surround_51_matches_ffmpeg_channel_order() {
	let (ours, channels) = decode_mp4("mp4/aac_51.m4a");
	assert_eq!(channels, 6);
	assert_matches(&ours, &samples(&fixture("mp4/aac_51.ffmpeg.s16le")));
}

#[test]
fn adts_stream_with_pns_intensity_and_tns_matches_ffmpeg() {
	// The ADTS file is a sequence of self-describing frames: no extradata.
	let mut dec = AacDecoderState::default();
	let frames = dec.decode_packet(&fixture("mp4/aac_tools_22k_stereo.aac")).unwrap();
	assert!(frames.iter().all(|f| f.complete && f.channels == 2 && f.sample_rate == 22050));
	let ours: Vec<i16> = frames.iter().flat_map(|f| f.pcm.iter().copied()).collect();
	assert_matches(&ours, &samples(&fixture("mp4/aac_tools_22k_stereo.ffmpeg.s16le")));
	assert_eq!(dec.concealed_frames, 0);
}

#[test]
fn audio_in_h264_mp4_decodes() {
	let (ours, channels) = decode_mp4("mp4/h264_bframes_aac.mp4");
	assert_eq!(channels, 1);
	// 1 s of FFmpeg's `sine` source (amplitude 1/8) at 48 kHz, plus priming.
	assert!(ours.len() >= 48_000);
	let peak = ours.iter().map(|s| s.saturating_abs()).max().unwrap_or(0);
	assert!((3_500..=5_000).contains(&peak), "peak {peak}");
}

#[test]
fn missing_configuration_is_an_error() {
	assert!(AacDecoderState::default().decode_packet(&[0x21, 0x10, 0x04]).is_err());
	assert!(AacDecoderState::from_extradata(&[0xFF]).is_none());
}

#[test]
fn damaged_frames_are_concealed_with_silence() {
	let mut dec = AacDecoderState::from_extradata(&[0x12, 0x10]).unwrap();
	let frames = dec.decode_packet(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
	assert_eq!(frames.len(), 1);
	assert!(!frames[0].complete);
	assert_eq!(frames[0].pcm.len(), 2048);
	assert!(frames[0].pcm.iter().all(|&s| s == 0));
	assert_eq!(dec.concealed_frames, 1);
}
