//! Vorbis decoder tests. References are FFmpeg's native float decoder
//! (`ffmpeg -c:a vorbis -i X -f s16le`), which drops the first packet's
//! priming samples as RMP does. RMP's float output, rounded the way
//! FFmpeg converts to S16, stays within one step of it.

use crate::common::{fixture, memory};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::ogg::demuxer::OggDemuxer;
use rmp::core::{Demuxer, Frame, Resolver, SampleFormat, SampleLayout, StreamKind};

/// Decodes the first audio stream to interleaved S16 (FFmpeg's
/// `lrintf(x * 32768)` with clipping); returns the samples, channel count
/// and sample rate.
fn decode_s16(demuxer: &mut dyn Demuxer) -> (Vec<i16>, usize, u32) {
	let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let (mut out, mut channels, mut rate) = (Vec::new(), 0, 0);
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id != stream.id {
			continue;
		}
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(a) = frame else { panic!("audio expected") };
			assert_eq!((a.format, a.layout), (SampleFormat::F32, SampleLayout::Planar));
			channels = usize::from(a.channels.count());
			rate = a.sample_rate;
			let plane = a.nb_samples * 4;
			for i in 0..a.nb_samples {
				for c in 0..channels {
					let at = c * plane + i * 4;
					let v = f32::from_le_bytes(a.data[at..at + 4].try_into().unwrap());
					out.push((v * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) as i16);
				}
			}
		}
	}
	(out, channels, rate)
}

fn reference(name: &str) -> Vec<i16> {
	fixture(&format!("vorbis/{name}.ffmpeg.s16le"))
		.chunks_exact(2)
		.map(|b| i16::from_le_bytes([b[0], b[1]]))
		.collect()
}

fn check(name: &str, demuxer: &mut dyn Demuxer, expected: (usize, u32)) {
	let (pcm, channels, rate) = decode_s16(demuxer);
	assert_eq!((channels, rate), expected, "{name}: layout");
	let reference = reference(name);
	assert_eq!(pcm.len(), reference.len(), "{name}: length");
	let worst = pcm.iter().zip(&reference).map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs()).max();
	assert!(worst.unwrap_or(0) <= 1, "{name}: max difference {worst:?}");
}

fn ogg(name: &str) -> OggDemuxer {
	OggDemuxer::new(memory(fixture(&format!("vorbis/{name}.ogg")))).unwrap()
}

#[test]
fn low_quality_mono_matches_ffmpeg() {
	check("mono_22k_q-1", &mut ogg("mono_22k_q-1"), (1, 22_050));
}

#[test]
fn surround_matches_ffmpeg_in_ffmpeg_order() {
	// 5.1: Vorbis codes L C R Ls Rs LFE, FFmpeg outputs L R C LFE Ls Rs.
	check("51_22k", &mut ogg("51_22k"), (6, 22_050));
}

#[test]
fn high_quality_stereo_from_webm_matches_ffmpeg() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("vorbis/st_44k_q8.webm"))).unwrap();
	check("st_44k_q8", &mut demuxer, (2, 44_100));
}

#[test]
fn ogg_vorbis_decodes_to_its_granule_length() {
	// The last granule position is the stream length: 2 s at 48 kHz.
	let mut demuxer = OggDemuxer::new(memory(fixture("ogg/vorbis.ogg"))).unwrap();
	let (pcm, channels, _) = decode_s16(&mut demuxer);
	assert_eq!(pcm.len() / channels, 96_000);
}
