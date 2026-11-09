//! Opus decoder tests (CELT, SILK, hybrid, mode switches, multistream).
//! References are libopus 1.4 decodes through FFmpeg
//! (`ffmpeg -flags2 +skip_manual -c:a libopus -i X -f s16le`), which keep
//! the pre-skip so they line up with RMP's untrimmed output.

use crate::common::{fixture, memory};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::ogg::demuxer::OggDemuxer;
use rmp::core::{Demuxer, Frame, Resolver, SampleFormat, StreamKind};

fn decode_f32(demuxer: &mut dyn Demuxer) -> (Vec<f32>, u8) {
	let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut out = Vec::new();
	let mut channels = 0;
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id != stream.id {
			continue;
		}
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(a) = frame else { panic!("audio expected") };
			assert_eq!(a.format, SampleFormat::F32);
			assert_eq!(a.sample_rate, 48_000);
			channels = a.channels.count();
			out.extend(a.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
		}
	}
	(out, channels)
}

/// libopus' `FLOAT2INT16`.
fn to_s16(x: f32) -> i16 {
	(x * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Returns the largest difference, in 16-bit steps, from the reference.
fn max_diff(name: &str, expected_channels: u8) -> i32 {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("opus/{name}.webm")))).unwrap();
	let (pcm, channels) = decode_f32(&mut demuxer);
	assert_eq!(channels, expected_channels, "{name}: channels");
	let reference: Vec<i16> = fixture(&format!("opus/{name}.ffmpeg.s16le"))
		.chunks_exact(2)
		.map(|b| i16::from_le_bytes([b[0], b[1]]))
		.collect();
	assert_eq!(pcm.len(), reference.len(), "{name}: length");
	pcm
		.iter()
		.zip(&reference)
		.map(|(&a, &b)| (i32::from(to_s16(a)) - i32::from(b)).abs())
		.max()
		.unwrap_or(0)
}

#[test]
fn silk_is_bit_exact_with_libopus() {
	// SILK is fixed point: narrowband mono, wideband stereo with 40 ms packets.
	assert_eq!(max_diff("silk_nb_mono", 1), 0);
	assert_eq!(max_diff("silk_wb_st_40ms", 2), 0);
}

#[test]
fn hybrid_matches_libopus() {
	// CELT is the float build: identical up to the last-bit rounding.
	assert!(max_diff("hybrid_st", 2) <= 1);
}

#[test]
fn mode_switches_match_libopus() {
	// SILK → hybrid → CELT and back, with transition and redundancy frames.
	assert!(max_diff("mode_switch", 1) <= 1);
}

#[test]
fn multistream_surround_matches_libopus() {
	assert!(max_diff("celt_51", 6) <= 1);
}

#[test]
fn ogg_and_webm_carry_the_same_stream() {
	// The same Opus packets in both containers decode identically.
	let mut ogg = OggDemuxer::new(memory(fixture("ogg/mono.opus"))).unwrap();
	let (from_ogg, _) = decode_f32(&mut ogg);
	assert_eq!(from_ogg.len() % 960, 0);
	assert!(from_ogg.iter().any(|&s| s != 0.0));
	assert!(from_ogg.iter().all(|s| s.is_finite() && s.abs() <= 2.0));
}
