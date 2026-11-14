//! AC-3 / E-AC-3 decoder tests. References are FFmpeg's float decoder
//! (`ffmpeg -i X -f s16le`, edit lists ignored), which dithers zero-bit
//! mantissas with the same seeded generator RMP uses, so RMP's float output,
//! rounded the way FFmpeg converts to S16, stays within one step of it.

use crate::common::{fixture, memory};
use rmp::container::ac3::demuxer::Ac3Demuxer;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Demuxer, Frame, Resolver, SampleFormat, SampleLayout, StreamKind};

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

fn check(name: &str, demuxer: &mut dyn Demuxer, expected: (usize, u32)) {
	let (pcm, channels, rate) = decode_s16(demuxer);
	assert_eq!((channels, rate), expected, "{name}: layout");
	let reference: Vec<i16> = fixture(&format!("ac3/{name}.ffmpeg.s16le"))
		.chunks_exact(2)
		.map(|b| i16::from_le_bytes([b[0], b[1]]))
		.collect();
	assert_eq!(pcm.len(), reference.len(), "{name}: length");
	let worst = pcm.iter().zip(&reference).map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs()).max();
	assert!(worst.unwrap_or(0) <= 1, "{name}: max difference {worst:?}");
}

#[test]
fn ac3_stereo_with_coupling_raw_stream() {
	let mut d = Ac3Demuxer::new(memory(fixture("ac3/stereo_cpl.ac3"))).unwrap();
	assert_eq!(d.streams().by_kind(StreamKind::Audio)[0].parameters.codec.id, "ac3");
	check("stereo_cpl", &mut d, (2, 44_100));
}

#[test]
fn ac3_surround_51_in_matroska() {
	let mut d = MkvDemuxer::new(memory(fixture("ac3/surround51.mkv"))).unwrap();
	check("surround51", &mut d, (6, 48_000));
}

#[test]
fn eac3_surround_51_in_mp4() {
	let mut d = Mp4Demuxer::new(memory(fixture("ac3/eac3_51.mp4"))).unwrap();
	check("eac3_51", &mut d, (6, 48_000));
}

#[test]
fn eac3_mono_32k_raw_stream() {
	let mut d = Ac3Demuxer::new(memory(fixture("ac3/mono_32k.eac3"))).unwrap();
	assert_eq!(d.streams().by_kind(StreamKind::Audio)[0].parameters.codec.id, "eac3");
	check("mono_32k", &mut d, (1, 32_000));
}

#[test]
fn damaged_frames_are_concealed_not_fatal() {
	let mut data = fixture("ac3/stereo_cpl.ac3");
	for i in (400..data.len()).step_by(97) {
		data[i] ^= 0x5A;
	}
	let mut d = Ac3Demuxer::new(memory(data)).unwrap();
	let (pcm, channels, _) = decode_s16(&mut d);
	assert_eq!(channels, 2);
	assert!(!pcm.is_empty());
}
