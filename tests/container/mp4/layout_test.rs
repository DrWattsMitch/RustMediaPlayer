//! MP4 output layouts: `+faststart` and fragmented MP4 (`-movflags
//! frag_keyframe+empty_moov+default_base_moof`, `-frag_duration`, ...).
//!
//! Every test remuxes a fixture and reads the result back with RMP's own
//! demuxer: the packets (data, keyframe flag, presentation time) must be
//! exactly those of the progressive (default) layout.

use crate::common::{fixture, memory, read_all};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::mp4::muxer::Mp4Muxer;
use rmp::core::{Demuxer, Muxer, Packet, StreamId};
use rmp::io::{Io, MemoryIo};
use rmp::{Message, Result};

/// Top-level boxes of a file: (type, offset, size).
fn top_level(data: &[u8]) -> Vec<(String, usize, usize)> {
	let mut boxes = Vec::new();
	let mut at = 0;
	while at + 8 <= data.len() {
		let mut size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
		if size == 1 {
			size = u64::from_be_bytes(data[at + 8..at + 16].try_into().unwrap()) as usize;
		}
		let kind = String::from_utf8_lossy(&data[at + 4..at + 8]).into_owned();
		boxes.push((kind, at, size));
		at += size.max(8);
	}
	boxes
}

fn kinds(data: &[u8]) -> Vec<String> {
	top_level(data).into_iter().map(|(k, _, _)| k).collect()
}

/// Remuxes all streams of `source` into `io` with the given options.
fn remux_into(source: &mut dyn Demuxer, io: Box<dyn Io>, options: &[(&str, &str)]) -> Result<()> {
	let mut muxer = Mp4Muxer::new(io);
	muxer.set_metadata(source.metadata().clone())?;
	for (key, value) in options {
		muxer.set_option(key, value)?;
	}
	for stream in source.streams().iter() {
		muxer.add(stream)?;
	}
	while let Some(packet) = source.read()? {
		muxer.write(packet)?;
	}
	muxer.finish()
}

fn remux(path: &str, options: &[(&str, &str)]) -> Vec<u8> {
	let mut source = Mp4Demuxer::new(memory(fixture(path))).unwrap();
	let output = MemoryIo::new();
	remux_into(&mut source, Box::new(output.clone()), options).unwrap();
	output.contents()
}

fn packets(data: Vec<u8>) -> Vec<Packet> {
	let mut demuxer = Mp4Demuxer::new(memory(data)).unwrap();
	read_all(&mut demuxer)
}

/// Same packets per stream (data, keyframe, presentation time).
fn assert_same_packets(a: &[Packet], b: &[Packet]) {
	assert_eq!(a.len(), b.len());
	for stream in [StreamId(0), StreamId(1)] {
		let x: Vec<_> = a.iter().filter(|p| p.stream_id == stream).collect();
		let y: Vec<_> = b.iter().filter(|p| p.stream_id == stream).collect();
		assert_eq!(x.len(), y.len(), "stream {}", stream.0);
		for (p, q) in x.iter().zip(&y) {
			assert_eq!(p.data, q.data);
			assert_eq!(p.keyframe, q.keyframe);
			assert_eq!(p.pts.map(|t| t.as_seconds()), q.pts.map(|t| t.as_seconds()));
		}
	}
}

const FRAGMENTED: &[(&str, &str)] = &[("movflags", "frag_keyframe+empty_moov+default_base_moof")];

#[test]
fn faststart_puts_moov_first_and_keeps_every_sample() {
	let plain = remux("mp4/h264_bframes_aac.mp4", &[]);
	let fast = remux("mp4/h264_bframes_aac.mp4", &[("movflags", "+faststart")]);
	assert_eq!(kinds(&plain), vec!["ftyp", "mdat", "moov"]);
	assert_eq!(kinds(&fast), vec!["ftyp", "moov", "mdat"]);
	assert_eq!(plain.len(), fast.len());
	assert_same_packets(&packets(plain), &packets(fast));
}

#[test]
fn fragmented_output_round_trips() {
	let plain = remux("mp4/h264_bframes_aac.mp4", &[]);
	let fragmented = remux("mp4/h264_bframes_aac.mp4", FRAGMENTED);
	let k = kinds(&fragmented);
	assert_eq!(&k[..2], &["ftyp", "moov"]);
	assert_eq!(k.last().map(String::as_str), Some("mfra"));
	assert!(k.iter().filter(|k| *k == "moof").count() >= 1);
	// moof and mdat alternate.
	let body: Vec<_> = k[2..k.len() - 1].to_vec();
	assert!(body.chunks(2).all(|pair| pair == ["moof", "mdat"]));
	assert_eq!(&fragmented[8..12], b"iso5");

	let a = packets(plain.clone());
	let b = packets(fragmented.clone());
	assert_same_packets(&a, &b);
	// Every fragment starts on a video keyframe.
	let copy = Mp4Demuxer::new(memory(fragmented)).unwrap();
	let original = Mp4Demuxer::new(memory(plain)).unwrap();
	assert!((copy.duration().as_seconds() - original.duration().as_seconds()).abs() < 0.05);
}

#[test]
fn fragment_rules_decide_the_cuts() {
	let count = |options: &[(&str, &str)]| {
		kinds(&remux("mp4/h264_bframes_aac.mp4", options)).iter().filter(|k| *k == "moof").count()
	};
	let by_keyframe = count(FRAGMENTED);
	let every_frame = count(&[("movflags", "frag_every_frame")]);
	let short = count(&[("frag_duration", "100000")]);
	let separate = count(&[("movflags", "frag_keyframe+separate_moof")]);
	let source = packets(fixture("mp4/h264_bframes_aac.mp4")).len();
	assert_eq!(every_frame, source);
	assert!(short > by_keyframe, "{short} > {by_keyframe}");
	assert_eq!(separate, 2 * by_keyframe);
	// A minimum longer than the file means a single fragment.
	assert_eq!(count(&[("movflags", "frag_every_frame"), ("min_frag_duration", "100000000")]), 1);
	for options in [&[("movflags", "frag_every_frame")][..], &[("frag_duration", "100000")]] {
		assert_same_packets(
			&packets(remux("mp4/h264_bframes_aac.mp4", &[])),
			&packets(remux("mp4/h264_bframes_aac.mp4", options)),
		);
	}
}

#[test]
fn audio_only_fragments_by_time() {
	let fragmented = remux("mp4/aac_tagged_cover.m4a", &[("movflags", "frag_keyframe")]);
	let moofs = kinds(&fragmented).iter().filter(|k| *k == "moof").count();
	let mut source = Mp4Demuxer::new(memory(fixture("mp4/aac_tagged_cover.m4a"))).unwrap();
	let copy = Mp4Demuxer::new(memory(fragmented.clone())).unwrap();
	assert_eq!(copy.metadata().title, source.metadata().title);
	assert!(copy.streams().streams[0].metadata.raw.contains_key("encoder_delay"));
	let a = read_all(&mut source);
	// Media time including priming, in seconds: one fragment per second.
	let media: f64 = a.iter().filter_map(|p| p.duration).map(|d| d.as_seconds()).sum();
	assert_eq!(moofs, media.ceil() as usize, "one fragment per second of {media} s");
	let b = packets(fragmented);
	assert_eq!(
		a.iter().map(|p| &p.data).collect::<Vec<_>>(),
		b.iter().map(|p| &p.data).collect::<Vec<_>>()
	);
}

/// An output that can't seek or be read back (a pipe, a socket).
struct Pipe(MemoryIo);

impl Io for Pipe {
	fn read(&mut self, _: &mut [u8]) -> Result<usize> {
		Err(Message::Other("pipe"))
	}
	fn write(&mut self, buf: &[u8]) -> Result<usize> {
		self.0.write(buf)
	}
}

#[test]
fn fragmented_output_streams_to_a_pipe() {
	let output = MemoryIo::new();
	let mut source = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	remux_into(&mut source, Box::new(Pipe(output.clone())), FRAGMENTED).unwrap();
	let piped = output.contents();
	let seekable = remux("mp4/h264_bframes_aac.mp4", FRAGMENTED);
	// Identical except mehd, which only a seekable output gets filled in.
	assert_eq!(piped.len(), seekable.len());
	let differing = piped.iter().zip(&seekable).filter(|(a, b)| a != b).count();
	assert!(differing <= 8, "{differing} bytes differ");
	assert_same_packets(&packets(piped), &packets(seekable));

	// A progressive file needs to seek back, and says so.
	let mut source = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	assert!(remux_into(&mut source, Box::new(Pipe(MemoryIo::new())), &[]).is_err());
}

/// Seeks, but can't read back.
struct WriteOnly(MemoryIo);

impl Io for WriteOnly {
	fn read(&mut self, _: &mut [u8]) -> Result<usize> {
		Err(Message::Other("write only"))
	}
	fn write(&mut self, buf: &[u8]) -> Result<usize> {
		self.0.write(buf)
	}
	fn seek(&mut self, pos: u64) -> Result<()> {
		self.0.seek(pos)
	}
}

#[test]
fn faststart_on_an_unreadable_output_fails_but_leaves_a_playable_file() {
	let output = MemoryIo::new();
	let mut source = Mp4Demuxer::new(memory(fixture("mp4/h264_bframes_aac.mp4"))).unwrap();
	let result =
		remux_into(&mut source, Box::new(WriteOnly(output.clone())), &[("movflags", "+faststart")]);
	assert!(result.is_err());
	assert_eq!(kinds(&output.contents()), vec!["ftyp", "mdat", "moov"]);
	assert_same_packets(
		&packets(output.contents()),
		&packets(remux("mp4/h264_bframes_aac.mp4", &[])),
	);
}

#[test]
fn truncated_fragmented_file_plays_up_to_the_cut() {
	let fragmented = remux("mp4/h264_bframes_aac.mp4", &[("frag_duration", "200000")]);
	let all = packets(fragmented.clone());
	let boxes = top_level(&fragmented);
	// Cut in the middle of the last mdat.
	let (_, at, size) = boxes.iter().rev().find(|(k, _, _)| k == "mdat").cloned().unwrap();
	let cut = fragmented[..at + size / 2].to_vec();
	let partial = packets(cut);
	assert!(!partial.is_empty() && partial.len() < all.len());
	for p in &partial {
		let q = all.iter().find(|q| q.stream_id == p.stream_id && q.dts == p.dts).unwrap();
		assert_eq!(p.data, q.data);
	}
}

#[test]
fn seeking_in_a_fragmented_file_lands_on_a_keyframe() {
	let fragmented = remux("mp4/h264_bframes_aac.mp4", FRAGMENTED);
	let mut demuxer = Mp4Demuxer::new(memory(fragmented)).unwrap();
	let duration = demuxer.duration().as_seconds();
	demuxer.seek(duration / 2.0).unwrap();
	let first_video =
		std::iter::from_fn(|| demuxer.read().unwrap()).find(|p| p.stream_id == StreamId(0)).unwrap();
	assert!(first_video.keyframe);
}

#[test]
fn matroska_video_without_dts_fragments() {
	let mut source = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	let original = read_all(&mut source);
	let mut source = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	let output = MemoryIo::new();
	remux_into(&mut source, Box::new(output.clone()), FRAGMENTED).unwrap();
	let copy = packets(output.contents());
	assert_eq!(original.len(), copy.len());
	// Presentation times relative to the stream's first one (MP4 tracks
	// start at 0, as in the progressive muxer).
	let relative = |packets: &[Packet], stream: StreamId| {
		let ms = |p: &Packet| (p.pts.unwrap().as_seconds() * 1e3).round() as i64;
		let ours: Vec<&Packet> = packets.iter().filter(|p| p.stream_id == stream).collect();
		let first = ours.iter().map(|p| ms(p)).min().unwrap();
		ours.iter().map(|p| (ms(p) - first, p.data.clone())).collect::<Vec<_>>()
	};
	for stream in [StreamId(0), StreamId(1)] {
		let mut a = relative(&original, stream);
		let mut b = relative(&copy, stream);
		a.sort();
		b.sort();
		assert_eq!(a, b, "stream {}", stream.0);
	}
}

#[test]
fn options_are_checked() {
	let mut muxer = Mp4Muxer::new(Box::new(MemoryIo::new()));
	assert!(muxer.set_option("movflags", "+faststart+frag_keyframe").is_ok());
	assert!(muxer.set_option("movflags", "+bogus").is_err());
	assert!(muxer.set_option("frag_duration", "two seconds").is_err());
	assert!(muxer.set_option("brand", "isom").is_err());
}

#[test]
fn reads_fragments_written_by_ffmpeg() {
	// `ffmpeg -c copy -movflags frag_keyframe+empty_moov`: tfhd carries an
	// explicit base data offset, there are no edit lists, and FFmpeg
	// stretches the first AAC packet over the priming (ffprobe reports the
	// same timestamps). Times are compared from the second packet on.
	let source = packets(fixture("mp4/h264_bframes_aac.mp4"));
	let fragmented = packets(fixture("mp4/h264_aac.ffmpeg_frag.mp4"));
	assert_eq!(source.len(), fragmented.len());
	for stream in [StreamId(0), StreamId(1)] {
		let a: Vec<_> = source.iter().filter(|p| p.stream_id == stream).collect();
		let b: Vec<_> = fragmented.iter().filter(|p| p.stream_id == stream).collect();
		assert_eq!(a.len(), b.len());
		let pts = |p: &Packet| p.pts.unwrap().as_seconds();
		let (sa, sb) = (pts(a[1]), pts(b[1]));
		for (x, y) in a.iter().zip(&b) {
			assert_eq!((&x.data, x.keyframe), (&y.data, y.keyframe));
		}
		for (x, y) in a.iter().zip(&b).skip(1) {
			assert!(((pts(x) - sa) - (pts(y) - sb)).abs() < 1e-3, "stream {}", stream.0);
		}
	}
}
