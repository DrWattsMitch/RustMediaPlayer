use crate::common::{fixture, memory};
use rmp::codecs::mp3::{Mp3Decoder, Mp3DecoderState, decoder::decode_packet};
use rmp::container::mp3::demuxer::Mp3Demuxer;
use rmp::core::{Decoder, Demuxer, Frame, SampleFormat};

/// Reference PCM produced by FFmpeg (`-flags2 +skip_manual`, no trimming).
fn reference() -> Vec<i16> {
	fixture("mp3/mpeg2_22k_mono_raw.ffmpeg.s16le")
		.chunks_exact(2)
		.map(|c| i16::from_le_bytes([c[0], c[1]]))
		.collect()
}

#[test]
fn matches_ffmpeg_within_one_lsb() {
	let mut decoder = Mp3DecoderState::new();
	let frames = decode_packet(&mut decoder, &fixture("mp3/mpeg2_22k_mono_raw.mp3")).unwrap();
	let ours: Vec<i16> = frames.iter().flat_map(|f| f.pcm.iter().copied()).collect();
	let reference = reference();
	assert_eq!(ours.len(), reference.len());

	let mut signal = 0f64;
	let mut noise = 0f64;
	for (&a, &b) in ours.iter().zip(&reference) {
		let diff = i32::from(a) - i32::from(b);
		assert!(diff.abs() <= 1, "sample differs by {diff}");
		signal += f64::from(b).powi(2);
		noise += f64::from(diff).powi(2);
	}
	let snr = 10.0 * (signal / noise.max(1.0)).log10();
	assert!(snr > 90.0, "SNR {snr:.1} dB");
	assert_eq!(decoder.concealed_frames, 0);
}

#[test]
fn decoder_trait_produces_timed_s16_frames() {
	let mut demuxer = Mp3Demuxer::new(memory(fixture("mp3/cbr_44k_stereo_tagged.mp3"))).unwrap();
	let stream = demuxer.streams().streams[0].clone();
	let decoder = Mp3Decoder::new();
	let mut state = decoder.create_state(&stream.parameters);

	let mut total = 0usize;
	let mut peak = 0i16;
	while let Some(packet) = demuxer.read().unwrap() {
		let pts = packet.pts;
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(frame) = frame else { panic!("audio expected") };
			assert_eq!(frame.format, SampleFormat::S16);
			assert_eq!(frame.channels.count(), 2);
			assert_eq!(frame.nb_samples, 1152);
			assert_eq!(frame.pts, pts);
			total += frame.nb_samples;
			for c in frame.data.chunks_exact(2) {
				peak = peak.max(i16::from_le_bytes([c[0], c[1]]).saturating_abs());
			}
		}
	}
	assert_eq!(total, 40 * 1152);
	// 0.4 + 0.1 full-scale sines: the decoded peak must be in that range.
	assert!((10_000..=17_000).contains(&peak), "peak {peak}");
}

#[test]
fn rejects_non_mp3_packets() {
	assert!(decode_packet(&mut Mp3DecoderState::new(), b"definitely not audio").is_err());
}
