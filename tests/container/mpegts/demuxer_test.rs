//! MPEG-TS demuxer tests. References are `ffprobe -show_packets` dumps with
//! decode times (`*.packets.csv`, see `tools/make_fixtures.sh`).

use crate::common::{fixture, memory, read_all};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::mpegts::TsDemuxer;
use rmp::core::{Demuxer, Muxer, Packet, RawValue, StreamKind};
use rmp::io::MemoryIo;

pub fn adler32(data: &[u8]) -> u32 {
	let (mut a, mut b) = (1u32, 0u32);
	for &byte in data {
		a = (a + u32::from(byte)) % 65521;
		b = (b + a) % 65521;
	}
	(b << 16) | a
}

/// One ffprobe packet line: stream, pts, dts, duration, size, key, hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
	pub stream: usize,
	pub pts: Option<i64>,
	pub dts: Option<i64>,
	pub duration: Option<u64>,
	pub size: usize,
	pub keyframe: bool,
	pub hash: u32,
}

pub fn reference(path: &str) -> Vec<Reference> {
	String::from_utf8(fixture(path))
		.unwrap()
		.lines()
		.map(|line| {
			let f: Vec<&str> = line.split(',').collect();
			Reference {
				stream: f[0].parse().unwrap(),
				pts: f[1].parse().ok(),
				dts: f[2].parse().ok(),
				duration: f[3].parse().ok(),
				size: f[4].parse().unwrap(),
				keyframe: f[5].starts_with('K'),
				hash: u32::from_str_radix(f[6].trim_start_matches("adler32:"), 16).unwrap(),
			}
		})
		.collect()
}

pub fn as_reference(p: &Packet) -> Reference {
	Reference {
		stream: p.stream_id.0,
		pts: p.pts.map(|t| t.ticks() as i64),
		dts: p.dts.map(|t| t.ticks() as i64),
		duration: p.duration.map(|t| t.ticks()),
		size: p.data.len(),
		keyframe: p.keyframe,
		hash: adler32(&p.data),
	}
}

/// Packets of each stream, in order (FFmpeg and RMP may interleave streams
/// differently; within a stream the order is the file's).
pub fn per_stream(packets: impl IntoIterator<Item = Reference>) -> Vec<Vec<Reference>> {
	let mut out: Vec<Vec<Reference>> = Vec::new();
	for p in packets {
		if out.len() <= p.stream {
			out.resize(p.stream + 1, Vec::new());
		}
		out[p.stream].push(p);
	}
	out
}

fn open(name: &str) -> TsDemuxer {
	TsDemuxer::new(memory(fixture(&format!("mpegts/{name}")))).unwrap()
}

/// Every packet matches FFmpeg's: pts, dts, duration, size, keyframe flag
/// and payload.
fn assert_matches_ffmpeg(name: &str) {
	let mut demuxer = open(name);
	let got = per_stream(read_all(&mut demuxer).iter().map(as_reference));
	let stem = name.rsplit_once('.').unwrap().0;
	let want = per_stream(reference(&format!("mpegts/{stem}.packets.csv")));
	assert_eq!(got.len(), want.len(), "{name}: stream count");
	for (s, (g, w)) in got.iter().zip(&want).enumerate() {
		assert_eq!(g.len(), w.len(), "{name} stream {s}: packet count");
		for (i, (g, w)) in g.iter().zip(w).enumerate() {
			assert_eq!(g, w, "{name} stream {s} packet {i}");
		}
	}
}

#[test]
fn h264_and_two_aac_tracks_match_ffmpeg() {
	// 48 kHz and 44.1 kHz AAC: the second checks FFmpeg's timestamp
	// interpolation between PES timestamps.
	assert_matches_ffmpeg("h264_aac_2audio.ts");
}

#[test]
fn hevc_and_ac3_match_ffmpeg() {
	assert_matches_ffmpeg("hevc_ac3.ts");
}

#[test]
fn mpeg2_video_and_mp2_match_ffmpeg() {
	assert_matches_ffmpeg("mpeg2_mp2.ts");
}

#[test]
fn m2ts_with_unlabelled_mp3_matches_ffmpeg() {
	// 192-byte packets; the MP3 stream is private data (type 0x06) with no
	// descriptor, identified from its content as FFmpeg does.
	assert_matches_ffmpeg("h264_mp3.m2ts");
}

#[test]
fn opus_matches_ffmpeg() {
	// Control headers stripped, timestamps per access unit. RMP applies the
	// end trim to the last packet's duration (as its Ogg demuxer does);
	// FFmpeg reports it as skip side data instead.
	let mut demuxer = open("opus_51.ts");
	let s = &demuxer.streams().streams[0];
	assert_eq!((s.parameters.codec.id.as_str(), s.parameters.channels), ("opus", Some(6)));
	let head = rmp::codecs::opus::OpusHead::parse(&s.parameters.extradata).unwrap();
	assert_eq!((head.mapping.family, head.mapping.streams, head.mapping.coupled), (1, 4, 2));
	assert!(matches!(s.metadata.raw.get("encoder_delay"), Some(RawValue::String(d)) if d == "312"));
	let mut got = read_all(&mut demuxer).iter().map(as_reference).collect::<Vec<_>>();
	let want = reference("mpegts/opus_51.packets.csv");
	assert_eq!(got.len(), want.len());
	let last = got.len() - 1;
	assert!(got[last].duration < want[last].duration);
	got[last].duration = want[last].duration;
	assert_eq!(got, want);
}

#[test]
fn streams_and_metadata() {
	let demuxer = open("h264_aac_2audio.ts");
	let s = &demuxer.streams().streams;
	assert_eq!(s.len(), 3);
	assert_eq!(s[0].kind, StreamKind::Video);
	assert_eq!((s[0].parameters.width, s[0].parameters.height), (Some(96), Some(64)));
	assert!(s[0].parameters.extradata.starts_with(&[0, 0, 0, 1, 0x67]));
	assert_eq!(s[1].parameters.sample_rate, Some(48_000));
	assert_eq!(s[2].parameters.sample_rate, Some(44_100));
	let lang = |i: usize| match s[i].metadata.raw.get("language") {
		Some(RawValue::String(l)) => l.clone(),
		_ => String::new(),
	};
	assert_eq!((lang(1), lang(2)), ("eng".to_string(), "fra".to_string()));
	assert!(
		matches!(demuxer.metadata().raw.get("service_name"), Some(RawValue::String(n)) if n == "RMPTest")
	);
	let d = demuxer.duration().as_seconds();
	assert!((1.19..1.3).contains(&d), "duration {d}");
	let hevc = open("hevc_ac3.ts");
	assert_eq!(hevc.streams().streams[1].parameters.codec.id, "ac3");
	let mp2 = open("mpeg2_mp2.ts");
	assert_eq!(mp2.streams().streams[1].parameters.codec.id, "mp2");
}

#[test]
fn timestamps_continue_across_the_33_bit_wrap() {
	// FFmpeg moves the timestamps before the wrap below zero and leaves
	// those after it as they are; RMP's are unsigned and keep increasing
	// through it. Either way time runs on evenly: RMP's are FFmpeg's + 2^33.
	let mut demuxer = open("wrap.ts");
	let got = per_stream(read_all(&mut demuxer).iter().map(as_reference));
	let want = per_stream(reference("mpegts/wrap.packets.csv"));
	let unwrap = |t: Option<i64>| t.map(|t| t + (1 << 33));
	for (g, w) in got.iter().zip(&want) {
		assert_eq!(g.len(), w.len());
		assert!(w.iter().any(|p| p.pts.is_some_and(|t| t < 0)), "fixture crosses the wrap");
		assert!(w.iter().any(|p| p.pts.is_some_and(|t| t > 0)), "fixture crosses the wrap");
		for (g, w) in g.iter().zip(w) {
			assert_eq!(g.pts, unwrap(w.pts));
			assert_eq!(g.dts, unwrap(w.dts));
			assert_eq!((g.size, g.hash, g.keyframe), (w.size, w.hash, w.keyframe));
		}
	}
}

#[test]
fn seeking_lands_on_the_last_keyframe_before_the_target() {
	let mut demuxer = open("h264_aac_2audio.ts");
	let all = read_all(&mut demuxer);
	let keys: Vec<f64> = all
		.iter()
		.filter(|p| p.stream_id.0 == 0 && p.keyframe)
		.map(|p| p.dts.unwrap().as_seconds())
		.collect();
	assert!(keys.len() >= 2);
	for target in [keys[0], keys[1] + 0.05, keys[keys.len() - 1] + 0.01, 0.0] {
		demuxer.seek(target).unwrap();
		let first_video =
			std::iter::from_fn(|| demuxer.read().unwrap()).find(|p| p.stream_id.0 == 0).unwrap();
		assert!(first_video.keyframe);
		let t = first_video.dts.unwrap().as_seconds();
		let expected = keys.iter().copied().rfind(|&k| k <= target + 1e-9).unwrap_or(keys[0]);
		assert!((t - expected).abs() < 1e-6, "seek {target}: got {t}, want {expected}");
	}
}

#[test]
fn detected_by_content() {
	let data = fixture("mpegts/h264_mp3.m2ts");
	let mut demuxer = rmp::container::detect(&data[..4096], memory(data.clone())).unwrap();
	assert_eq!(demuxer.streams().streams.len(), 2);
	assert!(demuxer.read().unwrap().is_some());
	// A file cut mid-packet still syncs.
	let cut = data[100..].to_vec();
	let mut demuxer = TsDemuxer::new(memory(cut)).unwrap();
	assert!(!read_all(&mut demuxer).is_empty());
}

/// Stream copy of a TS into MP4 and Matroska converts Annex B H.264 to
/// `avcC` and ADTS to raw AAC; the packets read back carry the same
/// payloads (after that conversion) and timestamps.
#[test]
fn remux_to_mp4_and_matroska() {
	let mut ts = open("h264_aac_2audio.ts");
	let source = read_all(&mut ts);
	for mp4 in [true, false] {
		let out = MemoryIo::new();
		let mut muxer: Box<dyn Muxer> = if mp4 {
			Box::new(rmp::container::mp4::muxer::Mp4Muxer::new(Box::new(out.clone())))
		} else {
			Box::new(rmp::container::mkv::muxer::MkvMuxer::new(Box::new(out.clone()), false))
		};
		for s in ts.streams().iter() {
			muxer.add(s).unwrap();
		}
		for p in &source {
			muxer.write(p.clone()).unwrap();
		}
		muxer.finish().unwrap();
		let mut back: Box<dyn Demuxer> = if mp4 {
			Box::new(Mp4Demuxer::new(memory(out.contents())).unwrap())
		} else {
			Box::new(MkvDemuxer::new(memory(out.contents())).unwrap())
		};
		let streams = back.streams().streams.clone();
		assert_eq!(streams.len(), 3);
		let avcc = rmp::codecs::h264::nal::AvcConfig::parse(&streams[0].parameters.extradata).unwrap();
		assert_eq!(avcc.nal_length_size, 4);
		assert_eq!(streams[1].parameters.extradata.len(), 2, "AudioSpecificConfig");
		let packets = read_all(back.as_mut());
		for stream in 0..3 {
			let src: Vec<&Packet> = source.iter().filter(|p| p.stream_id.0 == stream).collect();
			let got: Vec<&Packet> = packets.iter().filter(|p| p.stream_id.0 == stream).collect();
			assert_eq!(src.len(), got.len(), "mp4={mp4} stream {stream}");
			for (s, g) in src.iter().zip(&got) {
				let expected = if stream == 0 {
					rmp::container::bsf::annex_b_to_length_prefixed(&s.data)
				} else {
					let h = rmp::codecs::aac::AdtsHeader::parse(&s.data).unwrap();
					s.data[h.header_length..].to_vec()
				};
				assert_eq!(g.data, expected);
				assert_eq!(g.keyframe, s.keyframe || stream != 0);
			}
		}
	}
}
