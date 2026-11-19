//! HLS output: presentations written by RMP's HLS muxer (TS and fMP4
//! segments, AES-128, SAMPLE-AES, WebVTT renditions, sliding windows), read
//! back by RMP's HLS demuxer and compared with the source.

use super::mpegts_demuxer_test::{Reference, as_reference, per_stream};
use crate::common::{fixture, memory, read_all};
use rmp::container::hls;
use rmp::container::hls::muxer::{HlsMuxer, HlsOptions};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::mpegts::TsDemuxer;
use rmp::container::segment::StreamingOptions;
use rmp::core::{Demuxer, Muxer, RawValue, StreamKind};
use std::path::Path;

fn write_hls(
	source: &mut dyn Demuxer,
	dir: &Path,
	options: &[(&str, &str)],
	keep: impl Fn(StreamKind) -> bool,
) {
	let mut muxer = HlsMuxer::create(dir.join("out.m3u8")).unwrap();
	for (k, v) in options {
		muxer.set_option(k, v).unwrap();
	}
	let mut kept = Vec::new();
	for s in source.streams().iter() {
		if keep(s.kind) {
			muxer.add(s).unwrap();
			kept.push(s.id);
		}
	}
	while let Some(p) = source.read().unwrap() {
		if kept.contains(&p.stream_id) {
			muxer.write(p).unwrap();
		}
	}
	muxer.finish().unwrap();
}

fn all(_: StreamKind) -> bool {
	true
}

fn ts_source() -> TsDemuxer {
	TsDemuxer::new(memory(fixture("mpegts/h264_aac_2audio.ts"))).unwrap()
}

fn read_back(playlist: &Path) -> Vec<Vec<Reference>> {
	let mut d = hls::open(playlist.to_str().unwrap(), &StreamingOptions::default()).unwrap();
	per_stream(read_all(&mut d).iter().map(as_reference))
}

/// Same packets, one timestamp shift for everything.
fn assert_same(got: &[Vec<Reference>], want: &[Vec<Reference>], what: &str) {
	assert_eq!(got.len(), want.len(), "{what}: streams");
	let shift = got[0][0].dts.unwrap() - want[0][0].dts.unwrap();
	for (s, (g, w)) in got.iter().zip(want).enumerate() {
		assert_eq!(g.len(), w.len(), "{what} stream {s}");
		for (i, (g, w)) in g.iter().zip(w).enumerate() {
			let moved =
				Reference { pts: w.pts.map(|t| t + shift), dts: w.dts.map(|t| t + shift), ..w.clone() };
			// A packet's duration is the next decode time, so the last one of
			// a segment can't see it.
			assert_eq!(
				Reference { duration: None, ..g.clone() },
				Reference { duration: None, ..moved },
				"{what} {s}/{i}"
			);
		}
	}
}

fn segments(playlist: &Path) -> Vec<String> {
	std::fs::read_to_string(playlist)
		.unwrap()
		.lines()
		.filter(|l| !l.starts_with('#'))
		.map(str::to_string)
		.collect()
}

#[test]
fn mpegts_segments_read_back_as_the_source() {
	let dir = tempfile::tempdir().unwrap();
	write_hls(&mut ts_source(), dir.path(), &[("hls_time", "0.4")], all);
	let playlist = std::fs::read_to_string(dir.path().join("out.m3u8")).unwrap();
	assert!(
		playlist.starts_with("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n"),
		"{playlist}"
	);
	assert!(playlist.ends_with("#EXT-X-ENDLIST\n"));
	let names = segments(&dir.path().join("out.m3u8"));
	assert_eq!(names, ["out0.ts", "out1.ts", "out2.ts"]);
	// Every segment starts with the tables and a keyframe.
	for name in &names {
		let data = std::fs::read(dir.path().join(name)).unwrap();
		let mut seg = TsDemuxer::new(memory(data)).unwrap();
		let first_video = read_all(&mut seg).into_iter().find(|p| p.stream_id.0 == 0).unwrap();
		assert!(first_video.keyframe, "{name}");
	}
	let want = per_stream(read_all(&mut ts_source()).iter().map(as_reference));
	assert_same(&read_back(&dir.path().join("out.m3u8")), &want, "hls ts");
	// Languages carried in the PMT.
	let d =
		hls::open(dir.path().join("out.m3u8").to_str().unwrap(), &StreamingOptions::default()).unwrap();
	let lang = |i: usize| d.streams().streams[i].metadata.raw.get("language").cloned();
	assert!(matches!(lang(2), Some(RawValue::String(l)) if l == "fra"));
}

#[test]
fn fmp4_segments_read_back_as_the_source() {
	let dir = tempfile::tempdir().unwrap();
	let open = || Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	write_hls(
		&mut open(),
		dir.path(),
		&[("hls_time", "0.4"), ("hls_segment_type", "fmp4"), ("hls_playlist_type", "vod")],
		all,
	);
	let playlist = std::fs::read_to_string(dir.path().join("out.m3u8")).unwrap();
	assert!(
		playlist.contains("#EXT-X-VERSION:7\n") && playlist.contains("#EXT-X-MAP:URI=\"init.mp4\"\n")
	);
	assert!(playlist.contains("#EXT-X-PLAYLIST-TYPE:VOD\n"));
	assert!(segments(&dir.path().join("out.m3u8")).iter().all(|s| s.ends_with(".m4s")));
	let got = read_back(&dir.path().join("out.m3u8"));
	let want = per_stream(read_all(&mut open()).iter().map(as_reference));
	for (g, w) in got.iter().zip(&want) {
		let payload =
			|r: &[Reference]| r.iter().map(|p| (p.size, p.hash, p.keyframe)).collect::<Vec<_>>();
		assert_eq!(payload(g), payload(w));
	}
}

#[test]
fn encrypted_segments_read_back_as_the_clear_ones() {
	let clear = tempfile::tempdir().unwrap();
	write_hls(&mut ts_source(), clear.path(), &[("hls_time", "0.4")], all);
	let want = read_back(&clear.path().join("out.m3u8"));
	for method in ["aes-128", "sample-aes"] {
		let dir = tempfile::tempdir().unwrap();
		let options = [
			("hls_time", "0.4"),
			("hls_enc", "1"),
			("hls_enc_key", "000102030405060708090a0b0c0d0e0f"),
			("hls_enc_method", method),
		];
		write_hls(&mut ts_source(), dir.path(), &options, all);
		let playlist = std::fs::read_to_string(dir.path().join("out.m3u8")).unwrap();
		let line = if method == "aes-128" {
			"METHOD=AES-128,URI=\"out.key\""
		} else {
			"METHOD=SAMPLE-AES,URI=\"out.key\""
		};
		assert!(playlist.contains(line), "{playlist}");
		assert_eq!(std::fs::read(dir.path().join("out.key")).unwrap(), (0u8..16).collect::<Vec<_>>());
		// The segments really are encrypted...
		let first = std::fs::read(dir.path().join("out0.ts")).unwrap();
		let clear_first = std::fs::read(clear.path().join("out0.ts")).unwrap();
		assert_ne!(first, clear_first, "{method}");
		// ...and decrypt to the same packets.
		assert_eq!(read_back(&dir.path().join("out.m3u8")), want, "{method}");
	}
	// A key info file: URI, key file, IV.
	let dir = tempfile::tempdir().unwrap();
	std::fs::write(dir.path().join("k.bin"), [9u8; 16]).unwrap();
	std::fs::write(
		dir.path().join("info"),
		"https://keys.example/k.bin\nk.bin\n0x00000000000000000000000000000001\n",
	)
	.unwrap();
	let info = dir.path().join("info");
	write_hls(
		&mut ts_source(),
		dir.path(),
		&[("hls_time", "0.4"), ("hls_key_info_file", info.to_str().unwrap())],
		all,
	);
	let playlist = std::fs::read_to_string(dir.path().join("out.m3u8")).unwrap();
	assert_eq!(playlist.matches("#EXT-X-KEY:").count(), 1);
	assert!(playlist.contains("#EXT-X-KEY:METHOD=AES-128,URI=\"https://keys.example/k.bin\",IV=0x00000000000000000000000000000001"));
	// SAMPLE-AES has no fMP4 form here.
	let mut muxer = HlsMuxer::create(dir.path().join("x.m3u8")).unwrap();
	for (k, v) in [("hls_enc", "1"), ("hls_enc_method", "sample-aes"), ("hls_segment_type", "fmp4")] {
		muxer.set_option(k, v).unwrap();
	}
	muxer.add(&ts_source().streams().streams[0]).unwrap();
	let packet = read_all(&mut ts_source()).remove(0);
	assert!(muxer.write(packet).is_err());
}

#[test]
fn subtitles_become_a_webvtt_rendition() {
	let dir = tempfile::tempdir().unwrap();
	let open = || MkvDemuxer::new(memory(fixture("mkv/h264_flac_srt.mkv"))).unwrap();
	// Video and the SRT track (FLAC has no place in TS).
	write_hls(&mut open(), dir.path(), &[("hls_time", "1")], |k| k != StreamKind::Audio);
	let master = std::fs::read_to_string(dir.path().join("out_master.m3u8")).unwrap();
	assert!(master.contains("#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs\""), "{master}");
	assert!(master.contains("URI=\"out_vtt.m3u8\""), "{master}");
	assert!(master.contains(",SUBTITLES=\"subs\"\nout.m3u8\n"), "{master}");
	assert!(master.contains("CODECS=\"avc1."), "{master}");
	let vtt = segments(&dir.path().join("out_vtt.m3u8"));
	assert_eq!(vtt.len(), segments(&dir.path().join("out.m3u8")).len());
	let first = std::fs::read_to_string(dir.path().join(&vtt[0])).unwrap();
	assert!(
		first.starts_with("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000\n"),
		"{first}"
	);

	// Read back from the master playlist: the cues, on the video's clock.
	let mut back =
		hls::open(dir.path().join("out_master.m3u8").to_str().unwrap(), &StreamingOptions::default())
			.unwrap();
	let kinds: Vec<StreamKind> = back.streams().iter().map(|s| s.kind).collect();
	assert_eq!(kinds, [StreamKind::Video, StreamKind::Subtitle]);
	assert_eq!(back.streams().streams[1].parameters.codec.id, "webvtt");
	let got = read_all(&mut back);
	let mut source = open();
	let want = read_all(&mut source);
	let seconds = |p: &rmp::core::Packet| p.pts.unwrap().as_seconds();
	let first_video = |ps: &[rmp::core::Packet], id: usize| {
		ps.iter().filter(|p| p.stream_id.0 == id).map(seconds).fold(f64::MAX, f64::min)
	};
	let cues = |ps: &[rmp::core::Packet], id: usize, base: f64| -> Vec<(i64, String)> {
		ps.iter()
			.filter(|p| p.stream_id.0 == id)
			.map(|p| {
				(
					((seconds(p) - base) * 1000.0).round() as i64,
					String::from_utf8_lossy(&p.data).into_owned(),
				)
			})
			.collect()
	};
	let got_cues = cues(&got, 1, first_video(&got, 0));
	let want_cues = cues(&want, 2, first_video(&want, 0));
	assert_eq!(got_cues.len(), want_cues.len());
	for (g, w) in got_cues.iter().zip(&want_cues) {
		assert_eq!(g.1, w.1);
		assert!((g.0 - w.0).abs() <= 1, "cue at {} ms, source {} ms", g.0, w.0);
	}
}

#[test]
fn sliding_window_deletes_old_segments() {
	let dir = tempfile::tempdir().unwrap();
	let options = [
		("hls_time", "0.4"),
		("hls_list_size", "2"),
		("hls_flags", "delete_segments+independent_segments"),
		("start_number", "5"),
	];
	write_hls(&mut ts_source(), dir.path(), &options, all);
	let playlist = std::fs::read_to_string(dir.path().join("out.m3u8")).unwrap();
	assert!(
		playlist.contains("#EXT-X-MEDIA-SEQUENCE:6\n")
			&& playlist.contains("#EXT-X-INDEPENDENT-SEGMENTS\n"),
		"{playlist}"
	);
	assert_eq!(segments(&dir.path().join("out.m3u8")), ["out6.ts", "out7.ts"]);
	assert!(!dir.path().join("out5.ts").exists());
}

#[test]
fn options_are_checked() {
	let mut o = HlsOptions::default();
	assert!(o.set("hls_time", "4").unwrap());
	assert!(o.set("hls_flags", "+omit_endlist+split_by_time").unwrap());
	assert!(o.flags.omit_endlist && o.flags.split_by_time);
	assert!(o.set("hls_flags", "-split_by_time").unwrap());
	assert!(!o.flags.split_by_time && o.flags.omit_endlist);
	assert!(o.set("hls_flags", "delete_segments").unwrap());
	assert!(o.flags.delete_segments && !o.flags.omit_endlist);
	assert!(o.set("mpegts_service_id", "3").unwrap(), "TS options pass through");
	assert!(o.set("hls_time", "-1").is_err());
	assert!(o.set("hls_segment_type", "webm").is_err());
	assert!(o.set("hls_flags", "+bogus").is_err());
	assert!(o.set("hls_enc_key", "0011").is_err());
	assert!(!o.set("frobnicate", "1").unwrap());
}
