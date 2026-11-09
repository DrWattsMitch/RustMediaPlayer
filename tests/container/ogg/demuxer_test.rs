//! Ogg demuxer tests. References are `ffprobe -show_packets -show_data_hash
//! adler32` dumps (`*.packets.csv`, see `tools/make_fixtures.sh`). FFmpeg
//! starts Opus at `-pre_skip`; RMP starts at 0 and reports the pre-skip as
//! `encoder_delay`.

use crate::common::{fixture, memory, read_all};
use rmp::container::ogg::demuxer::OggDemuxer;
use rmp::core::{Demuxer, RawValue};

fn adler32(data: &[u8]) -> u32 {
	let (mut a, mut b) = (1u32, 0u32);
	for &byte in data {
		a = (a + u32::from(byte)) % 65521;
		b = (b + a) % 65521;
	}
	(b << 16) | a
}

/// `(pts, duration, size, adler32)` per packet of stream 0 (FFmpeg shows a
/// `METADATA_BLOCK_PICTURE` cover as an extra attached-picture stream; RMP
/// reports it in `metadata.images`).
fn reference(name: &str) -> Vec<(i64, Option<u64>, usize, u32)> {
	String::from_utf8(fixture(&format!("ogg/{name}.packets.csv")))
		.unwrap()
		.lines()
		.filter(|line| line.starts_with("0,"))
		.map(|line| {
			let f: Vec<&str> = line.split(',').collect();
			(
				f[1].parse().unwrap(),
				f[2].parse().ok(),
				f[3].parse().unwrap(),
				u32::from_str_radix(f[5].trim_start_matches("adler32:"), 16).unwrap(),
			)
		})
		.collect()
}

fn delay(demuxer: &OggDemuxer) -> i64 {
	match demuxer.streams().streams[0].metadata.raw.get("encoder_delay") {
		Some(RawValue::String(v)) => v.parse().unwrap(),
		_ => 0,
	}
}

fn open(file: &str) -> OggDemuxer {
	OggDemuxer::new(memory(fixture(&format!("ogg/{file}")))).unwrap()
}

/// Packets, timestamps and durations match FFmpeg.
fn assert_matches_ffmpeg(file: &str, check_timing: bool) {
	let mut demuxer = open(file);
	let shift = delay(&demuxer);
	let expected = reference(file.rsplit_once('.').unwrap().0);
	let packets = read_all(&mut demuxer);
	assert_eq!(packets.len(), expected.len(), "{file}: packet count");
	for (i, (p, &(pts, duration, size, hash))) in packets.iter().zip(&expected).enumerate() {
		let at = format!("{file} packet {i}");
		assert_eq!(p.data.len(), size, "{at}: size");
		assert_eq!(adler32(&p.data), hash, "{at}: payload");
		if check_timing {
			assert_eq!(p.pts.map(|t| t.ticks() as i64), Some(pts + shift), "{at}: pts");
			assert_eq!(p.duration.map(|t| t.ticks()), duration, "{at}: duration");
		}
	}
}

#[test]
fn opus_packets_match_ffmpeg() {
	assert_matches_ffmpeg("mono.opus", true);
	assert_matches_ffmpeg("st_cover.opus", true);
	assert_matches_ffmpeg("51.opus", true);
}

#[test]
fn flac_packets_match_ffmpeg() {
	assert_matches_ffmpeg("flac.oga", true);
}

#[test]
fn vorbis_packets_match_ffmpeg() {
	// RMP has no Vorbis parser for per-packet durations; payloads match.
	assert_matches_ffmpeg("vorbis.ogg", false);
}

#[test]
fn opus_stream_parameters_and_tags() {
	let demuxer = open("mono.opus");
	let s = &demuxer.streams().streams[0];
	assert_eq!(s.parameters.codec.id, "opus");
	assert_eq!(s.parameters.sample_rate, Some(48_000));
	assert_eq!(s.parameters.channels, Some(1));
	assert!(s.parameters.extradata.starts_with(b"OpusHead"));
	assert_eq!(delay(&demuxer), 312);
	assert_eq!(demuxer.metadata().title.as_deref(), Some("Ogg Test"));
	assert_eq!(demuxer.metadata().artist.as_deref(), Some("RMP"));
	// Duration is the last granule position (pre-skip included), as FFmpeg.
	assert_eq!(demuxer.duration().ticks(), 96_312);

	let cover = open("st_cover.opus");
	assert_eq!(cover.metadata().album.as_deref(), Some("Album"));
	let image = &cover.metadata().images[0];
	assert_eq!(image.mime_type, "image/png");
	assert!(image.data.starts_with(b"\x89PNG"));
}

#[test]
fn flac_and_vorbis_extradata() {
	let flac = open("flac.oga");
	let p = &flac.streams().streams[0].parameters;
	assert_eq!(
		(p.codec.id.as_str(), p.sample_rate, p.channels, p.extradata.len()),
		("flac", Some(44_100), Some(2), 34)
	);
	assert_eq!(flac.metadata().title.as_deref(), Some("FlacOgg"));

	let vorbis = open("vorbis.ogg");
	let p = &vorbis.streams().streams[0].parameters;
	assert_eq!(p.codec.id, "vorbis");
	// FFmpeg's layout: 2, two Xiph-laced sizes, then the three headers.
	assert_eq!(p.extradata[0], 2);
	assert_eq!(&p.extradata[3..10], b"\x01vorbis");
}

#[test]
fn seek_lands_before_target_and_continues_in_order() {
	let mut demuxer = open("st_cover.opus");
	let all = read_all(&mut demuxer);
	for target in [0.0, 0.5, 1.37, 2.9] {
		demuxer.seek(target).unwrap();
		let first = demuxer.read().unwrap().expect("packet after seek");
		let pts = first.pts.unwrap().ticks();
		let goal = (target * 48_000.0) as u64 + 312;
		// At or before the target, within the 80 ms pre-roll plus a page.
		assert!(pts <= goal.max(312), "seek {target}: {pts} > {goal}");
		assert!(goal.saturating_sub(pts) <= 3840 + 48_000, "seek {target}: too early ({pts})");
		// The packets that follow are the original ones, in order.
		let at = all.iter().position(|p| p.pts == first.pts).expect("known packet");
		assert_eq!(all[at].data, first.data);
		for expected in &all[at + 1..(at + 10).min(all.len())] {
			let next = demuxer.read().unwrap().unwrap();
			assert_eq!((next.pts, &next.data), (expected.pts, &expected.data));
		}
	}
}

#[test]
fn damaged_page_is_skipped() {
	let mut data = fixture("ogg/mono.opus");
	// Corrupt a byte in the middle: the page fails its CRC and is dropped,
	// the rest of the stream still demuxes.
	let mid = data.len() / 2;
	data[mid] ^= 0x55;
	let mut demuxer = OggDemuxer::new(memory(data)).unwrap();
	let packets = read_all(&mut demuxer);
	let total = reference("mono").len();
	assert!(packets.len() < total && packets.len() > total / 2);
}
