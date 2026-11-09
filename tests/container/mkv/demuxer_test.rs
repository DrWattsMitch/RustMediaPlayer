//! Matroska / WebM demuxer tests. References are `ffprobe -show_packets`
//! dumps of the fixtures (`*.packets.csv`, see `tools/make_fixtures.sh`).

use crate::common::{fixture, memory, read_all};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Demuxer, Frame, Packet, RawValue, Resolver, StreamKind};

fn adler32(data: &[u8]) -> u32 {
	let (mut a, mut b) = (1u32, 0u32);
	for &byte in data {
		a = (a + u32::from(byte)) % 65521;
		b = (b + a) % 65521;
	}
	(b << 16) | a
}

struct Reference {
	stream: usize,
	pts: i64,
	duration: Option<u64>,
	size: usize,
	keyframe: bool,
	hash: u32,
}

fn reference(path: &str) -> Vec<Reference> {
	let text = String::from_utf8(fixture(path)).unwrap();
	text
		.lines()
		.map(|line| {
			let f: Vec<&str> = line.split(',').collect();
			Reference {
				stream: f[0].parse().unwrap(),
				pts: f[1].parse().unwrap(),
				duration: f[2].parse().ok(),
				size: f[3].parse().unwrap(),
				keyframe: f[4].starts_with('K'),
				hash: u32::from_str_radix(f[5].trim_start_matches("adler32:"), 16).unwrap(),
			}
		})
		.collect()
}

fn delay_ticks(demuxer: &MkvDemuxer, stream: usize) -> i64 {
	let s = &demuxer.streams().streams[stream];
	let Some(RawValue::String(v)) = s.metadata.raw.get("encoder_delay") else { return 0 };
	let samples: u64 = v.parse().unwrap();
	let rate =
		if s.parameters.codec.id == "opus" { 48_000 } else { s.parameters.sample_rate.unwrap() };
	rmp::core::rescale_ticks(samples, rmp::core::TimeBase::new(1, rate), s.time_base).unwrap() as i64
}

/// Every packet matches FFmpeg's demuxer: order, stream, pts, duration,
/// size, keyframe flag and payload. FFmpeg subtracts the Opus CodecDelay
/// from pts; RMP reports it as `encoder_delay` instead (as for MP4 priming).
fn assert_matches_ffmpeg(name: &str) {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("mkv/{name}")))).unwrap();
	let stem = name.rsplit_once('.').unwrap().0;
	let expected = reference(&format!("mkv/{stem}.packets.csv"));
	let packets = read_all(&mut demuxer);
	assert_eq!(packets.len(), expected.len(), "{name}: packet count");
	for (i, (p, e)) in packets.iter().zip(&expected).enumerate() {
		let at = format!("{name} packet {i}");
		assert_eq!(p.stream_id.0, e.stream, "{at}");
		let pts = p.pts.expect("pts").ticks() as i64;
		assert_eq!(pts, e.pts + delay_ticks(&demuxer, e.stream), "{at}: pts");
		if let Some(d) = e.duration {
			assert_eq!(p.duration.map(|t| t.ticks()), Some(d), "{at}: duration");
		}
		assert_eq!(p.data.len(), e.size, "{at}: size");
		assert_eq!(p.keyframe, e.keyframe, "{at}: keyframe");
		assert_eq!(adler32(&p.data), e.hash, "{at}: payload");
	}
}

#[test]
fn h264_aac_packets_match_ffmpeg() {
	assert_matches_ffmpeg("h264_aac.mkv");
}

#[test]
fn h264_flac_subrip_packets_match_ffmpeg() {
	assert_matches_ffmpeg("h264_flac_srt.mkv");
}

#[test]
fn webm_vp9_opus_webvtt_packets_match_ffmpeg() {
	assert_matches_ffmpeg("vp9_opus_vtt.webm");
}

#[test]
fn live_webm_with_unknown_sizes_matches_ffmpeg() {
	assert_matches_ffmpeg("live_vp9_opus.webm");
}

#[test]
fn streams_and_metadata() {
	let demuxer = MkvDemuxer::new(memory(fixture("mkv/h264_flac_srt.mkv"))).unwrap();
	let streams = &demuxer.streams().streams;
	let kinds: Vec<_> = streams.iter().map(|s| (s.kind, s.parameters.codec.id.as_str())).collect();
	assert_eq!(
		kinds,
		vec![
			(StreamKind::Video, "h264"),
			(StreamKind::Audio, "flac"),
			(StreamKind::Subtitle, "subrip")
		]
	);
	assert_eq!((streams[0].parameters.width, streams[0].parameters.height), (Some(64), Some(48)));
	// FLAC extradata is normalised to the 34-byte STREAMINFO.
	assert_eq!(streams[1].parameters.extradata.len(), 34);
	assert_eq!(streams[1].parameters.sample_rate, Some(44_100));
	assert!(
		matches!(streams[1].metadata.raw.get("language"), Some(RawValue::String(l)) if l == "fra")
	);
	let m = demuxer.metadata();
	assert_eq!(m.title.as_deref(), Some("MKV Song"));
	assert_eq!(m.artist.as_deref(), Some("RMP Tests"));
	assert_eq!(m.album.as_deref(), Some("Fixtures"));
	assert!((demuxer.duration().as_seconds() - 2.0).abs() < 0.01);
	assert!(demuxer.cue_count() > 0);

	let webm = MkvDemuxer::new(memory(fixture("mkv/vp9_opus_vtt.webm"))).unwrap();
	let opus = &webm.streams().streams[1];
	assert_eq!(opus.parameters.codec.id, "opus");
	assert!(opus.parameters.extradata.starts_with(b"OpusHead"));
	assert!(
		matches!(opus.metadata.raw.get("encoder_delay"), Some(RawValue::String(v)) if v == "312")
	);
	assert_eq!(webm.metadata().title.as_deref(), Some("WebM clip"));
}

fn decode_stream(demuxer: &mut dyn Demuxer, kind: StreamKind) -> Vec<u8> {
	let stream = demuxer.streams().iter().find(|s| s.kind == kind).unwrap().clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut out = Vec::new();
	let mut push = |frames: Vec<Frame>| {
		for frame in frames {
			match frame {
				Frame::Audio(a) => out.extend_from_slice(&a.data),
				Frame::Video(v) => out.extend_from_slice(&v.data),
			}
		}
	};
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id == stream.id {
			push(decoder.decode(&mut state, packet).unwrap().value);
		}
	}
	push(decoder.flush(&mut state).unwrap().value);
	out
}

/// The same H.264 + AAC bitstreams decode identically from MKV and MP4:
/// the Matroska path feeds the existing decoders without any conversion.
#[test]
fn decodes_like_the_mp4_source() {
	for kind in [StreamKind::Video, StreamKind::Audio] {
		let mut mkv = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
		let mut mp4 = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
		let a = decode_stream(&mut mkv, kind);
		let b = decode_stream(&mut mp4, kind);
		assert!(!a.is_empty());
		assert_eq!(a, b, "{kind:?}");
	}
}

#[test]
fn seeks_with_cues_to_a_keyframe() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/h264_flac_srt.mkv"))).unwrap();
	let all = read_all(&mut demuxer);
	demuxer.seek(1.3).unwrap();
	let after = read_all(&mut demuxer);
	let first_video = after.iter().find(|p| p.stream_id.0 == 0).unwrap();
	assert!(first_video.keyframe);
	let start = first_video.pts.unwrap().ticks();
	assert!(start <= 1300 && start > 0, "{start}");
	// Audio resumes at the cue time and the tail of the file is unchanged.
	let audio = after.iter().find(|p| p.stream_id.0 == 1).unwrap();
	assert!(audio.pts.unwrap().ticks() >= start.saturating_sub(105));
	let last = |ps: &[Packet]| ps.last().map(|p| (p.stream_id.0, p.pts, p.data.len()));
	assert_eq!(last(&all), last(&after));
	demuxer.seek(0.0).unwrap();
	assert_eq!(read_all(&mut demuxer).len(), all.len());
}

#[test]
fn seeks_without_cues_by_cluster_index() {
	// The live WebM has no cues and unknown-size clusters.
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/live_vp9_opus.webm"))).unwrap();
	assert_eq!(demuxer.cue_count(), 0);
	let total = read_all(&mut demuxer).len();
	demuxer.seek(0.5).unwrap();
	let after = read_all(&mut demuxer);
	assert!(!after.is_empty() && after.len() <= total);
	if let Some(v) = after.iter().find(|p| p.stream_id.0 == 0) {
		assert!(v.keyframe);
	}
}

// --- Synthetic files: lacing, header stripping, BlockGroups, damage. ---

fn vint_size(n: usize) -> Vec<u8> {
	assert!(n < (1 << 28) - 1);
	let v = (n as u32) | (1 << 28);
	v.to_be_bytes().to_vec()
}

fn element(id: u32, body: &[u8]) -> Vec<u8> {
	let mut out: Vec<u8> = id.to_be_bytes().iter().copied().skip_while(|&b| b == 0).collect();
	out.extend(vint_size(body.len()));
	out.extend_from_slice(body);
	out
}

fn uint(id: u32, v: u64) -> Vec<u8> {
	element(id, &v.to_be_bytes())
}

fn block(track: u8, relative: i16, flags: u8, payload: &[u8]) -> Vec<u8> {
	let mut b = vec![0x80 | track];
	b.extend(relative.to_be_bytes());
	b.push(flags);
	b.extend_from_slice(payload);
	b
}

fn synthetic() -> Vec<u8> {
	let ebml = element(0x1A45_DFA3, &element(0x4282, b"matroska"));
	let info = element(0x1549_A966, &uint(0x2AD7B1, 1_000_000));
	// Track 1: PCM with header stripping (prefix "PRE"), 10 ms per frame.
	let encoding = element(
		0x6D80,
		&element(0x6240, &element(0x5034, &[uint(0x4254, 3), element(0x4255, b"PRE")].concat())),
	);
	let audio =
		element(0xE1, &[uint(0x9F, 1), element(0xB5, &8000f32.to_bits().to_be_bytes())].concat());
	let track1 = element(
		0xAE,
		&[
			uint(0xD7, 1),
			uint(0x73C5, 11),
			uint(0x83, 2),
			element(0x86, b"A_PCM/INT/LIT"),
			uint(0x23E383, 10_000_000),
			audio,
			encoding,
		]
		.concat(),
	);
	// Track 2: subtitles in BlockGroups.
	let track2 =
		element(0xAE, &[uint(0xD7, 2), uint(0x83, 17), element(0x86, b"S_TEXT/UTF8")].concat());
	let tracks = element(0x1654_AE6B, &[track1, track2].concat());

	// Xiph lacing: 3 frames of 2, 3 and 1 bytes.
	let xiph = block(1, 0, 0x80 | 0x02, &[2, 2, 3, b'a', b'a', b'b', b'b', b'b', b'c']);
	// EBML lacing: 2 frames of 4 and 2 bytes.
	let ebml_lace = block(1, 30, 0x80 | 0x06, &[1, 0x84, b'd', b'd', b'd', b'd', b'e', b'e']);
	// Fixed lacing: 2 frames of 2 bytes.
	let fixed = block(1, 50, 0x80 | 0x04, &[1, b'f', b'f', b'g', b'g']);
	let group = element(
		0xA0,
		&[element(0xA1, &block(2, 5, 0, b"Hi")), uint(0x9B, 700), element(0xFB, &[0xFF])].concat(),
	);
	let cluster1 = element(
		0x1F43_B675,
		&[uint(0xE7, 0), element(0xA3, &xiph), element(0xA3, &ebml_lace), group].concat(),
	);
	let cluster2 = element(0x1F43_B675, &[uint(0xE7, 1000), element(0xA3, &fixed)].concat());
	let garbage = vec![0x00, 0x13, 0x37, 0x00, 0xFF, 0xFF];
	let segment = [info, tracks, cluster1, garbage, cluster2].concat();
	[ebml, element(0x1853_8067, &segment)].concat()
}

/// `(stream, pts, duration, payload, keyframe)`.
type Summary = (usize, Option<u64>, Option<u64>, Vec<u8>, bool);

#[test]
fn synthetic_lacing_stripping_and_resync() {
	let mut demuxer = MkvDemuxer::new(memory(synthetic())).unwrap();
	assert_eq!(demuxer.streams().streams[0].parameters.codec.id, "pcm_s16le");
	let packets = read_all(&mut demuxer);
	let summary: Vec<Summary> = packets
		.iter()
		.map(|p| {
			(
				p.stream_id.0,
				p.pts.map(|t| t.ticks()),
				p.duration.map(|t| t.ticks()),
				p.data.clone(),
				p.keyframe,
			)
		})
		.collect();
	let pre = |s: &[u8]| [b"PRE".as_slice(), s].concat();
	assert_eq!(
		summary,
		vec![
			(0, Some(0), Some(10), pre(b"aa"), true),
			(0, Some(10), Some(10), pre(b"bbb"), true),
			(0, Some(20), Some(10), pre(b"c"), true),
			(0, Some(30), Some(10), pre(b"dddd"), true),
			(0, Some(40), Some(10), pre(b"ee"), true),
			(1, Some(5), Some(700), b"Hi".to_vec(), false),
			(0, Some(1050), Some(10), pre(b"ff"), true),
			(0, Some(1060), Some(10), pre(b"gg"), true),
		]
	);
}

#[test]
fn rejects_non_matroska_and_trackless_files() {
	assert!(MkvDemuxer::new(memory(b"not matroska at all".to_vec())).is_err());
	let ebml = element(0x1A45_DFA3, &element(0x4282, b"matroska"));
	let file = [ebml.clone(), element(0x1853_8067, &[])].concat();
	assert!(MkvDemuxer::new(memory(file)).is_err());
	let wrong_doc =
		[element(0x1A45_DFA3, &element(0x4282, b"other")), element(0x1853_8067, &[])].concat();
	assert!(MkvDemuxer::new(memory(wrong_doc)).is_err());
}

#[test]
fn truncated_file_ends_cleanly() {
	let data = fixture("mkv/h264_flac_srt.mkv");
	let full = read_all(&mut MkvDemuxer::new(memory(data.clone())).unwrap()).len();
	let mut cut = MkvDemuxer::new(memory(data[..data.len() * 2 / 3].to_vec())).unwrap();
	let partial = read_all(&mut cut).len();
	assert!(partial > 0 && partial < full);
}
