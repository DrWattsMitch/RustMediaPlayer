//! FLAC decoder and encoder tests. References are FFmpeg 6.1 decodes of the
//! fixtures (`-f s16le` up to 16 bits, `-f s32le` above).

use crate::common::{fixture, memory, read_all};
use rmp::codecs::flac::md5::Md5;
use rmp::codecs::flac::{FlacEncoder, StreamInfo};
use rmp::container::flac::demuxer::FlacDemuxer;
use rmp::container::flac::muxer::FlacMuxer;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::core::{
	AudioFrame, Channels, Demuxer, Encoder, Frame, Muxer, Resolver, SampleFormat, SampleLayout,
};
use rmp::io::MemoryIo;

const CASES: [(&str, &str); 6] = [
	("s16_stereo_44k", "s16le"),
	("s16_ms_48k_lvl8", "s16le"),
	("noise_22k", "s16le"),
	("wasted_mono_8k_lvl0", "s16le"),
	("s24_51_96k_lvl12", "s32le"),
	("s24_rs_32k", "s32le"),
];

fn decode(demuxer: &mut dyn Demuxer) -> (Vec<u8>, SampleFormat) {
	let stream = demuxer.streams().by_kind(rmp::core::StreamKind::Audio)[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut out = Vec::new();
	let mut format = SampleFormat::S16;
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id != stream.id {
			continue;
		}
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(a) = frame else { panic!("audio expected") };
			format = a.format;
			out.extend_from_slice(&a.data);
		}
	}
	(out, format)
}

#[test]
fn decodes_bit_exact_with_ffmpeg() {
	for (name, raw) in CASES {
		let mut demuxer = FlacDemuxer::new(memory(fixture(&format!("flac/{name}.flac")))).unwrap();
		let (pcm, format) = decode(&mut demuxer);
		let expected = fixture(&format!("flac/{name}.ffmpeg.{raw}"));
		assert_eq!(
			format,
			if raw == "s16le" { SampleFormat::S16 } else { SampleFormat::S32 },
			"{name}"
		);
		assert_eq!(pcm.len(), expected.len(), "{name}: length");
		assert!(pcm == expected, "{name}: samples differ");
	}
}

#[test]
fn flac_in_matroska_decodes_like_native() {
	let mut mkv = MkvDemuxer::new(memory(fixture("mkv/h264_flac_srt.mkv"))).unwrap();
	let (pcm, _) = decode(&mut mkv);
	assert_eq!(pcm.len(), 2 * 44_100 * 2);
}

#[test]
fn demuxer_reads_streaminfo_tags_and_timestamps() {
	let mut demuxer = FlacDemuxer::new(memory(fixture("flac/s16_stereo_44k.flac"))).unwrap();
	let stream = &demuxer.streams().streams[0];
	assert_eq!(stream.parameters.codec.id, "flac");
	assert_eq!(stream.parameters.extradata.len(), 34);
	assert_eq!(stream.parameters.bits_per_sample, Some(16));
	assert_eq!((stream.parameters.sample_rate, stream.parameters.channels), (Some(44_100), Some(2)));
	let m = demuxer.metadata().clone();
	assert_eq!(m.title.as_deref(), Some("FLAC Song"));
	assert_eq!(m.artist.as_deref(), Some("RMP Tests"));
	assert_eq!((m.track_number, m.tracks_total), (Some(4), Some(12)));
	assert_eq!(demuxer.duration().ticks(), 22_050);
	let packets = read_all(&mut demuxer);
	let pts: Vec<u64> = packets.iter().map(|p| p.pts.unwrap().ticks()).collect();
	assert_eq!(pts, vec![0, 4608, 9216, 13824, 18432]);
	assert_eq!(packets.last().unwrap().duration.unwrap().ticks(), 22_050 - 18_432);
}

#[test]
fn seeks_to_the_containing_frame() {
	let mut demuxer = FlacDemuxer::new(memory(fixture("flac/wasted_mono_8k_lvl0.flac"))).unwrap();
	let all = read_all(&mut demuxer);
	for (seconds, expected) in [(0.0, 0usize), (0.2, 8), (0.31, 12), (0.49, 20)] {
		demuxer.seek(seconds).unwrap();
		let first = demuxer.read().unwrap().unwrap();
		assert_eq!(first.pts, all[expected].pts, "seek {seconds}");
		assert_eq!(first.data, all[expected].data, "seek {seconds}");
	}
}

/// Interleaved samples of a raw reference as `i32` frames of `depth` bits.
fn samples_of(raw: &[u8], format: SampleFormat) -> Vec<i32> {
	match format {
		SampleFormat::S16 => {
			raw.chunks_exact(2).map(|c| i32::from(i16::from_le_bytes([c[0], c[1]]))).collect()
		}
		_ => raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
	}
}

/// Encodes `pcm` (FFmpeg's decode of a fixture) with RMP's encoder, muxes,
/// demuxes and decodes it again. Returns the file and the decoded PCM.
fn encode_round_trip(
	pcm: &[u8],
	format: SampleFormat,
	info: &StreamInfo,
	level: u8,
) -> (Vec<u8>, Vec<u8>) {
	let mut stream = rmp::core::Stream::audio(
		rmp::core::StreamId(0),
		info.sample_rate,
		info.channels,
		rmp::core::CodecId::new("flac"),
	);
	stream.parameters.bits_per_sample = Some(info.bits_per_sample);
	let encoder = FlacEncoder::new(stream.id).with_level(level);
	stream.parameters.extradata = encoder.extradata(&stream.parameters).unwrap();
	let mut state = encoder.create_state(&stream.parameters);

	let out = MemoryIo::new();
	let mut muxer = FlacMuxer::new(Box::new(out.clone()));
	muxer.add(&stream).unwrap();
	// Feed in uneven chunks to exercise buffering across frames.
	let frame_bytes = format.bytes() * usize::from(info.channels);
	for chunk in pcm.chunks(frame_bytes * 1000) {
		let frame = AudioFrame::new(
			chunk.to_vec(),
			info.sample_rate,
			Channels::from_count(info.channels),
			format,
			SampleLayout::Interleaved,
		);
		for packet in encoder.encode(&mut state, Frame::Audio(frame)).unwrap().value {
			muxer.write(packet).unwrap();
		}
	}
	for packet in encoder.flush(&mut state).unwrap().value {
		muxer.write(packet).unwrap();
	}
	muxer.finish().unwrap();

	let file = out.contents();
	let mut demuxer = FlacDemuxer::new(memory(file.clone())).unwrap();
	let (decoded, _) = decode(&mut demuxer);
	(file, decoded)
}

#[test]
fn encoder_is_lossless_and_writes_exact_streaminfo() {
	for (name, raw) in CASES {
		let source = fixture(&format!("flac/{name}.flac"));
		let info = StreamInfo::from_extradata(&source[4..]).unwrap();
		let format = if raw == "s16le" { SampleFormat::S16 } else { SampleFormat::S32 };
		let pcm = fixture(&format!("flac/{name}.ffmpeg.{raw}"));
		for level in [0u8, 5, 8] {
			let (file, decoded) = encode_round_trip(&pcm, format, &info, level);
			assert!(decoded == pcm, "{name} level {level}: not lossless");

			// STREAMINFO was rewritten with the real totals and MD5.
			let written = StreamInfo::from_extradata(&file[4..]).unwrap();
			let samples = samples_of(&pcm, format);
			assert_eq!(written.total_samples, (samples.len() / usize::from(info.channels)) as u64);
			let shift =
				if format == SampleFormat::S16 { 16 } else { 32 } - u32::from(info.bits_per_sample);
			let bytes = usize::from(info.bits_per_sample).div_ceil(8);
			let mut md5 = Md5::new();
			for s in samples {
				md5.update(&(s >> shift).to_le_bytes()[..bytes]);
			}
			assert_eq!(written.md5, md5.finish(), "{name} level {level}: md5");
			assert_eq!(written.md5, info.md5, "{name} level {level}: md5 differs from FFmpeg's");
			assert!(written.max_frame_size >= written.min_frame_size && written.min_frame_size > 0);
		}
	}
}

/// At FFmpeg's default level, RMP's files are about the size of FFmpeg's.
#[test]
fn compression_is_close_to_ffmpeg() {
	for (name, raw) in CASES {
		// Only fixtures FFmpeg encoded at its default level are comparable.
		if name.contains("lvl") {
			continue;
		}
		let source = fixture(&format!("flac/{name}.flac"));
		let info = StreamInfo::from_extradata(&source[4..]).unwrap();
		let format = if raw == "s16le" { SampleFormat::S16 } else { SampleFormat::S32 };
		let (file, _) =
			encode_round_trip(&fixture(&format!("flac/{name}.ffmpeg.{raw}")), format, &info, 5);
		// FFmpeg's file has 8 KiB of padding and similar tags; compare audio.
		let ratio = file.len() as f64 / source.len() as f64;
		eprintln!("{name}: RMP {} bytes, FFmpeg {} bytes ({ratio:.3})", file.len(), source.len());
		assert!(ratio < 1.03, "{name}: {} vs {} bytes ({ratio:.3})", file.len(), source.len());
	}
}

#[test]
fn encoder_accepts_float_and_planar_input() {
	let rate = 48_000;
	let n = 10_000;
	let mut data = Vec::new();
	for ch in 0..2 {
		for i in 0..n {
			let v = ((i as f32 * 0.01 * (ch + 1) as f32).sin() * 0.5).to_le_bytes();
			data.extend_from_slice(&v);
		}
	}
	let frame =
		AudioFrame::new(data, rate, Channels::Stereo, SampleFormat::F32, SampleLayout::Planar);
	let mut stream =
		rmp::core::Stream::audio(rmp::core::StreamId(0), rate, 2, rmp::core::CodecId::new("flac"));
	stream.parameters.bits_per_sample = Some(24);
	let encoder = FlacEncoder::new(stream.id);
	let mut state = encoder.create_state(&stream.parameters);
	let mut packets = encoder.encode(&mut state, Frame::Audio(frame)).unwrap().value;
	packets.extend(encoder.flush(&mut state).unwrap().value);
	let total: u64 = packets.iter().map(|p| p.duration.unwrap().ticks()).sum();
	assert_eq!(total, n as u64);
	assert_eq!(packets[1].pts.unwrap().ticks(), 4608);
}
