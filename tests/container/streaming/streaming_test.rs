//! HLS and DASH: playlists and MPDs written by FFmpeg, demuxed segment by
//! segment, compared with `ffprobe` of the same manifest; live playlists
//! over HTTP from a local server.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use super::mpegts_demuxer_test::{Reference, as_reference, per_stream, reference};
use crate::common::read_all;
use rmp::container::segment::{SegmentedDemuxer, StreamingOptions};
use rmp::container::{dash, hls};
use rmp::core::{Demuxer, RawValue, StreamKind};
use rmp::io::Input;

fn path(rel: &str) -> String {
	format!("{}/tests/fixtures/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn packets(demuxer: &mut dyn Demuxer) -> Vec<Vec<Reference>> {
	per_stream(read_all(demuxer).iter().map(as_reference))
}

fn assert_identical(got: &[Vec<Reference>], want: &[Vec<Reference>], what: &str) {
	assert_eq!(got.len(), want.len(), "{what}: streams");
	for (s, (g, w)) in got.iter().zip(want).enumerate() {
		assert_eq!(g.len(), w.len(), "{what} stream {s}: packets");
		for (i, (g, w)) in g.iter().zip(w).enumerate() {
			// FFmpeg leaves some first-packet durations unset.
			let duration_ok = w.duration.is_none() || g.duration == w.duration;
			assert!(
				duration_ok && Reference { duration: g.duration, ..w.clone() } == *g,
				"{what} stream {s} packet {i}: {g:?} vs {w:?}"
			);
		}
	}
}

/// Same payloads, flags and order; timestamps shifted by one constant per
/// stream (RMP's MP4 demuxer keeps times non-negative where FFmpeg's go
/// below zero).
fn assert_same_media(got: &[Vec<Reference>], want: &[Vec<Reference>], what: &str) {
	assert_eq!(got.len(), want.len(), "{what}: streams");
	for (s, (g, w)) in got.iter().zip(want).enumerate() {
		assert_eq!(g.len(), w.len(), "{what} stream {s}: packets");
		let shift = g[0].pts.unwrap() - w[0].pts.unwrap();
		for (g, w) in g.iter().zip(w) {
			assert_eq!((g.size, g.hash, g.keyframe), (w.size, w.hash, w.keyframe), "{what} stream {s}");
			assert_eq!(g.pts.unwrap() - w.pts.unwrap(), shift, "{what} stream {s}: pts");
		}
	}
}

fn open_hls(rel: &str) -> SegmentedDemuxer {
	hls::open(&path(rel), &StreamingOptions::default()).unwrap()
}

#[test]
fn hls_mpegts_segments_match_ffmpeg() {
	let mut d = open_hls("hls/ts/index.m3u8");
	assert_eq!(d.streams().streams.len(), 2);
	assert!((d.duration().as_seconds() - 1.2).abs() < 0.01);
	assert_identical(&packets(&mut d), &per_stream(reference("hls/ts/index.packets.csv")), "hls ts");
}

/// RMP's MP4 demuxer over the init segment and the media segments joined
/// into one file: what segment-by-segment demuxing must reproduce.
fn joined_mp4(files: &[String]) -> Vec<Vec<Reference>> {
	let data: Vec<u8> = files.iter().flat_map(|f| std::fs::read(path(f)).unwrap()).collect();
	let mut d = rmp::container::mp4::demuxer::Mp4Demuxer::new(crate::common::memory(data)).unwrap();
	packets(&mut d)
}

#[test]
fn hls_fmp4_segments_match_ffmpeg() {
	let mut d = open_hls("hls/fmp4/index.m3u8");
	let got = packets(&mut d);
	let files: Vec<String> =
		["init.mp4", "s0.m4s", "s1.m4s", "s2.m4s"].iter().map(|f| format!("hls/fmp4/{f}")).collect();
	assert_eq!(got, joined_mp4(&files));
	// FFmpeg agrees on everything but the last AAC duration, which RMP's
	// MP4 demuxer trims to the edit list.
	let mut want = per_stream(reference("hls/fmp4/index.packets.csv"));
	for s in &mut want {
		if let Some(last) = s.last_mut() {
			last.duration = None;
		}
	}
	assert_identical(&got, &want, "hls fmp4");
}

#[test]
fn hls_aes128_segments_are_decrypted() {
	let mut d = open_hls("hls/aes/index.m3u8");
	assert_identical(
		&packets(&mut d),
		&per_stream(reference("hls/aes/index.packets.csv")),
		"hls aes",
	);
}

#[test]
fn hls_byte_ranges_of_a_single_file() {
	// Same encode as hls/ts, cut by EXT-X-BYTERANGE instead of files.
	let mut single = open_hls("hls/single/index.m3u8");
	let mut files = open_hls("hls/ts/index.m3u8");
	assert_eq!(packets(&mut single), packets(&mut files));
}

#[test]
fn hls_variant_and_audio_rendition_choice() {
	let location = path("hls/master/master.m3u8");
	let mut d = hls::open(&location, &StreamingOptions::default()).unwrap();
	let s = d.streams().streams.clone();
	assert_eq!(s.len(), 2);
	assert_eq!((s[0].kind, s[0].parameters.width), (StreamKind::Video, Some(96)));
	assert!(matches!(s[1].metadata.raw.get("language"), Some(RawValue::String(l)) if l == "eng"));
	// The audio comes from its own rendition playlist, interleaved by time.
	let all = read_all(&mut d);
	assert!(all.iter().any(|p| p.stream_id.0 == 1));
	let times: Vec<f64> = all.iter().map(|p| p.dts.unwrap().as_seconds()).collect();
	assert!(times.windows(2).all(|w| w[1] >= w[0] - 0.2), "interleaved by decode time");

	let low = StreamingOptions { max_bandwidth: Some(150_000), ..Default::default() };
	let d = hls::open(&location, &low).unwrap();
	assert_eq!(d.streams().streams[0].parameters.width, Some(48));
	let by_index = StreamingOptions { variant: Some(1), ..Default::default() };
	let d = hls::open(&location, &by_index).unwrap();
	assert_eq!(d.streams().streams[0].parameters.width, Some(48));
}

#[test]
fn hls_seek_restarts_at_the_segment() {
	let mut d = open_hls("hls/ts/index.m3u8");
	let all = read_all(&mut d);
	let start = all[0].dts.unwrap().as_seconds();
	d.seek(start + 0.85).unwrap();
	let first = d.read().unwrap().unwrap();
	let t = first.dts.or(first.pts).unwrap().as_seconds();
	// Segments are 0.4 s: the third one starts at +0.8 s.
	assert!((t - (start + 0.8)).abs() < 0.08, "seek landed at {t}");
	d.seek(0.0).unwrap();
	assert_eq!(read_all(&mut d).len(), all.len());
}

fn open_dash(rel: &str) -> SegmentedDemuxer {
	dash::open(&path(rel), &StreamingOptions::default()).unwrap()
}

#[test]
fn dash_template_and_timeline_match_ffmpeg() {
	let mut d = open_dash("dash/template/manifest.mpd");
	let got = packets(&mut d);
	// Each representation equals its segments joined into one file.
	for (stream, chunks) in [(0usize, 3), (1, 4)] {
		let mut files = vec![format!("dash/template/init-stream{stream}.m4s")];
		files.extend((1..=chunks).map(|n| format!("dash/template/chunk-stream{stream}-{n:05}.m4s")));
		let mut want = joined_mp4(&files).remove(0);
		for p in &mut want {
			p.stream = stream;
		}
		assert_eq!(got[stream], want, "representation {stream}");
	}
	// FFmpeg 6.1's dash demuxer drops the last timeline segment (one audio
	// packet); everything before it matches.
	let mut want = per_stream(reference("dash/template/manifest.packets.csv"));
	assert_eq!(got[1].len(), want[1].len() + 1);
	want[1].push(got[1].last().unwrap().clone());
	let shift = got[1][0].pts.unwrap() - want[1][0].pts.unwrap();
	want[1].last_mut().unwrap().pts = got[1].last().unwrap().pts.map(|t| t - shift);
	assert_same_media(&got, &want, "dash");
}

#[test]
fn dash_segment_list_with_media_ranges() {
	let mut list = open_dash("dash/list/manifest.mpd");
	let mut template = open_dash("dash/template/manifest.mpd");
	assert_eq!(packets(&mut list), packets(&mut template));
}

#[test]
fn dash_segment_base_with_sidx() {
	let mut d = open_dash("dash/base/manifest.mpd");
	assert_same_media(
		&packets(&mut d),
		&per_stream(reference("dash/base/v.packets.csv")),
		"dash base",
	);
}

#[test]
fn input_recognises_manifests() {
	for (rel, streams) in
		[("hls/ts/index.m3u8", 2), ("hls/master/master.m3u8", 2), ("dash/template/manifest.mpd", 2)]
	{
		let mut input = Input::open(path(rel)).unwrap();
		assert_eq!(input.streams().streams.len(), streams, "{rel}");
		assert!(input.read().unwrap().is_some());
	}
}

/// A small HTTP/1.1 file server with `Range` support. Files can be changed
/// while it runs (a live playlist).
struct Server {
	base: String,
	files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl Server {
	fn start() -> Self {
		let listener = TcpListener::bind("127.0.0.1:0").unwrap();
		let base = format!("http://{}", listener.local_addr().unwrap());
		let files: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::default();
		let shared = files.clone();
		std::thread::spawn(move || {
			for stream in listener.incoming() {
				let Ok(mut stream) = stream else { continue };
				let mut reader = BufReader::new(stream.try_clone().unwrap());
				let mut request = String::new();
				if reader.read_line(&mut request).is_err() {
					continue;
				}
				let target = request.split_whitespace().nth(1).unwrap_or("/").to_string();
				let mut range = None;
				loop {
					let mut line = String::new();
					if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
						break;
					}
					if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
						let (a, b) = v.trim().split_once('-').unwrap();
						range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().ok()));
					}
				}
				let body = shared.lock().unwrap().get(&target).cloned();
				let response = match body {
					None => {
						b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
					}
					Some(body) => {
						let (status, slice) = match range {
							Some((a, b)) => {
								let end = b.map_or(body.len(), |b| (b + 1).min(body.len()));
								("206 Partial Content", body[a.min(end)..end].to_vec())
							}
							None => ("200 OK", body),
						};
						let mut r = format!(
							"HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
							slice.len()
						)
						.into_bytes();
						r.extend(slice);
						r
					}
				};
				let _ = stream.write_all(&response);
			}
		});
		Self { base, files }
	}

	fn put(&self, name: &str, data: Vec<u8>) {
		self.files.lock().unwrap().insert(format!("/{name}"), data);
	}
}

#[test]
fn hls_over_http_with_byte_ranges() {
	let server = Server::start();
	for f in ["index.m3u8", "index.ts"] {
		server.put(f, std::fs::read(path(&format!("hls/single/{f}"))).unwrap());
	}
	let mut remote =
		hls::open(&format!("{}/index.m3u8", server.base), &StreamingOptions::default()).unwrap();
	let mut local = open_hls("hls/single/index.m3u8");
	assert_eq!(packets(&mut remote), packets(&mut local));
}

#[test]
fn live_hls_playlist_is_reloaded_until_it_ends() {
	let server = Server::start();
	let text = String::from_utf8(std::fs::read(path("hls/ts/index.m3u8")).unwrap()).unwrap();
	for i in 0..3 {
		server.put(&format!("s{i}.ts"), std::fs::read(path(&format!("hls/ts/s{i}.ts"))).unwrap());
	}
	// A live playlist: no PLAYLIST-TYPE or ENDLIST, one segment so far, a
	// target duration of 1 s (reloads every 0.5 s).
	let lines: Vec<&str> = text
		.lines()
		.filter(|l| !l.starts_with("#EXT-X-PLAYLIST-TYPE") && !l.starts_with("#EXT-X-ENDLIST"))
		.map(|l| if l.starts_with("#EXT-X-TARGETDURATION") { "#EXT-X-TARGETDURATION:1" } else { l })
		.collect();
	let first_segment_end = lines.iter().position(|l| *l == "s0.ts").unwrap() + 1;
	server.put("live.m3u8", lines[..first_segment_end].join("\n").into_bytes());
	let files = server.files.clone();
	let full = lines.join("\n") + "\n#EXT-X-ENDLIST\n";
	std::thread::spawn(move || {
		std::thread::sleep(std::time::Duration::from_millis(700));
		files.lock().unwrap().insert("/live.m3u8".into(), full.into_bytes());
	});
	let mut live =
		hls::open(&format!("{}/live.m3u8", server.base), &StreamingOptions::default()).unwrap();
	assert_eq!(live.duration().ticks(), 0, "live: duration unknown");
	let got = packets(&mut live);
	assert_identical(&got, &per_stream(reference("hls/ts/index.packets.csv")), "live hls");
}

#[test]
fn broken_manifests_are_errors() {
	let dir = tempfile::tempdir().unwrap();
	let write = |name: &str, text: &str| {
		let p = dir.path().join(name);
		std::fs::write(&p, text).unwrap();
		p.to_string_lossy().into_owned()
	};
	let options = StreamingOptions::default();
	assert!(
		hls::open(&write("a.m3u8", "#EXTM3U\n#EXTINF:1,\nmissing.ts\n#EXT-X-ENDLIST\n"), &options)
			.is_err()
	);
	assert!(
		hls::open(
			&write("b.m3u8", "#EXTM3U\n#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"k\"\n#EXTINF:1,\nx.ts\n"),
			&options
		)
		.is_err()
	);
	assert!(dash::open(&write("c.mpd", "<MPD><Period></Period></MPD>"), &options).is_err());
	assert!(dash::open(&write("d.mpd", "<MPD><Period>"), &options).is_err());
}
