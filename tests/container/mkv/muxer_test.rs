//! Matroska / WebM muxer tests. Each fixture is remuxed into memory and
//! read back with RMP's demuxer; the packets, codec configuration and tags
//! must survive unchanged. `tools/make_fixtures.sh` holds the FFmpeg
//! commands behind the fixtures; the FFmpeg-side checks (ffprobe packet
//! dumps and decodes of the output) are described in
//! `docs/MKV_OGG_MUXING_CLI.md`.

use crate::common::{fixture, memory, read_all};
use rmp::container::flac::demuxer::FlacDemuxer;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mkv::muxer::MkvMuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::ogg::demuxer::OggDemuxer;
use rmp::core::{Demuxer, Metadata, Muxer, Packet, RawValue, StreamKind, TimeBase};
use rmp::io::{Io, MemoryIo};

/// Remuxes every stream of `demuxer` into a new Matroska file.
fn remux(demuxer: &mut dyn Demuxer, webm: bool) -> (Vec<u8>, Vec<Packet>) {
	let out = MemoryIo::new();
	let mut muxer = MkvMuxer::new(Box::new(out.clone()), webm);
	muxer.set_metadata(demuxer.metadata().clone()).unwrap();
	for stream in demuxer.streams().iter() {
		muxer.add(stream).unwrap();
	}
	let packets = read_all(demuxer);
	for p in &packets {
		muxer.write(p.clone()).unwrap();
	}
	muxer.finish().unwrap();
	(out.contents(), packets)
}

fn ms(t: Option<rmp::core::Time>) -> Option<u64> {
	t.and_then(|t| t.rescale(TimeBase::new(1, 1000))).map(|t| t.ticks())
}

/// Packets read back match the source: order, stream, timestamps (at the
/// 1 ms Matroska resolution), keyframe flag and payload.
fn assert_same_packets(name: &str, source: &[Packet], back: &[Packet]) {
	assert_eq!(source.len(), back.len(), "{name}: packet count");
	for (i, (a, b)) in source.iter().zip(back).enumerate() {
		let at = format!("{name} packet {i}");
		assert_eq!(a.stream_id, b.stream_id, "{at}: stream");
		assert_eq!(ms(a.pts.or(a.dts)), ms(b.pts), "{at}: pts");
		assert_eq!(a.keyframe, b.keyframe, "{at}: keyframe");
		assert_eq!(a.data, b.data, "{at}: payload");
	}
}

fn round_trip_mkv(name: &str, webm: bool) {
	let source = fixture(&format!("mkv/{name}"));
	let mut demuxer = MkvDemuxer::new(memory(source)).unwrap();
	let (bytes, packets) = remux(&mut demuxer, webm);
	let mut back = MkvDemuxer::new(memory(bytes)).unwrap();
	let original = MkvDemuxer::new(memory(fixture(&format!("mkv/{name}")))).unwrap();
	for (a, b) in original.streams().iter().zip(back.streams().iter()) {
		assert_eq!(a.kind, b.kind, "{name}");
		assert_eq!(a.parameters.codec, b.parameters.codec, "{name}");
		assert_eq!(a.parameters.extradata, b.parameters.extradata, "{name}: extradata");
		assert_eq!(a.parameters.sample_rate, b.parameters.sample_rate, "{name}");
		assert_eq!(a.parameters.channels, b.parameters.channels, "{name}");
		assert_eq!(a.parameters.width, b.parameters.width, "{name}");
		assert_eq!(a.parameters.height, b.parameters.height, "{name}");
		for key in ["language", "encoder_delay"] {
			let get = |s: &rmp::core::Stream| match s.metadata.raw.get(key) {
				Some(RawValue::String(v)) => Some(v.clone()),
				_ => None,
			};
			assert_eq!(get(a), get(b), "{name}: stream {key}");
		}
	}
	assert_eq!(original.metadata().title, back.metadata().title, "{name}: title");
	assert_eq!(original.metadata().artist, back.metadata().artist, "{name}: artist");
	if original.duration().ticks() > 0 {
		assert!(
			(original.duration().as_seconds() - back.duration().as_seconds()).abs() < 0.05,
			"{name}"
		);
	} else {
		// Live files have no Duration; RMP writes the real one.
		assert!(back.duration().as_seconds() > 0.5, "{name}: duration written");
	}
	let read_back = read_all(&mut back);
	assert_same_packets(name, &packets, &read_back);
	// Durations survive too, including Opus end trimming (DiscardPadding).
	for (i, (a, b)) in packets.iter().zip(&read_back).enumerate() {
		if a.duration.is_some_and(|d| d.base() == TimeBase::new(1, 48_000)) {
			assert_eq!(a.duration, b.duration, "{name} packet {i}: trimmed duration");
		} else if a.duration.is_some() && b.duration.is_some() {
			assert_eq!(ms(a.duration), ms(b.duration), "{name} packet {i}: duration");
		}
	}
}

#[test]
fn h264_aac_round_trips() {
	round_trip_mkv("h264_aac.mkv", false);
}

#[test]
fn h264_flac_subrip_round_trips() {
	round_trip_mkv("h264_flac_srt.mkv", false);
}

#[test]
fn webm_vp9_opus_webvtt_round_trips() {
	round_trip_mkv("vp9_opus_vtt.webm", true);
}

#[test]
fn live_webm_round_trips_as_mkv() {
	round_trip_mkv("live_vp9_opus.webm", false);
}

#[test]
fn layout_has_seek_head_cues_and_exact_sizes() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	let (bytes, _) = remux(&mut demuxer, false);
	assert_eq!(&bytes[..4], &[0x1A, 0x45, 0xDF, 0xA3]);
	assert!(bytes.windows(8).any(|w| w == b"matroska"));
	let back = MkvDemuxer::new(memory(bytes.clone())).unwrap();
	assert!(back.cue_count() > 0, "cues for video keyframes");
	// The segment size was patched to the real size.
	let seg = back.segment_start();
	let size_at = (seg - 8) as usize;
	let size = u64::from_be_bytes(bytes[size_at..size_at + 8].try_into().unwrap()) & ((1 << 56) - 1);
	assert_eq!(seg + size, bytes.len() as u64);
	// The SeekHead replaced the reserved Void.
	assert_eq!(&bytes[seg as usize..seg as usize + 4], &[0x11, 0x4D, 0x9B, 0x74]);
}

#[test]
fn seeking_a_muxed_file_lands_on_keyframes() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	let (bytes, _) = remux(&mut demuxer, false);
	let mut back = MkvDemuxer::new(memory(bytes)).unwrap();
	back.seek(0.5).unwrap();
	let first_video = std::iter::from_fn(|| back.read().unwrap())
		.find(|p| back_kind(p) == StreamKind::Video)
		.expect("video after seek");
	assert!(first_video.keyframe);
}

fn back_kind(p: &Packet) -> StreamKind {
	// h264_aac.mkv: stream 0 is video, stream 1 audio.
	if p.stream_id.0 == 0 { StreamKind::Video } else { StreamKind::Audio }
}

#[test]
fn mp4_with_cover_art_becomes_mka_with_attachment() {
	let mut demuxer = Mp4Demuxer::new(memory(fixture("mp4/aac_tagged_cover.m4a"))).unwrap();
	let original_meta = demuxer.metadata().clone();
	let (bytes, packets) = remux(&mut demuxer, false);
	let mut back = MkvDemuxer::new(memory(bytes)).unwrap();
	let m = back.metadata().clone();
	assert_eq!(m.title, original_meta.title);
	assert_eq!(m.artist, original_meta.artist);
	assert_eq!(m.album, original_meta.album);
	assert_eq!(m.track_number, original_meta.track_number);
	assert_eq!(m.images.len(), original_meta.images.len());
	assert_eq!(m.images[0].data, original_meta.images[0].data);
	assert_eq!(m.images[0].mime_type, original_meta.images[0].mime_type);
	// AAC priming survives as CodecDelay → encoder_delay.
	let delay = |s: &rmp::core::Stream| match s.metadata.raw.get("encoder_delay") {
		Some(RawValue::String(v)) => v.clone(),
		_ => String::new(),
	};
	let source = Mp4Demuxer::new(memory(fixture("mp4/aac_tagged_cover.m4a"))).unwrap();
	assert_eq!(delay(&back.streams().streams[0]), delay(&source.streams().streams[0]));
	assert_eq!(
		back.streams().streams[0].parameters.extradata,
		source.streams().streams[0].parameters.extradata
	);
	assert_same_packets("aac_tagged_cover", &packets, &read_all(&mut back));
}

#[test]
fn flac_streaminfo_is_patched_with_the_real_values() {
	let source = FlacDemuxer::new(memory(fixture("flac/s16_stereo_44k.flac"))).unwrap();
	let exact = source.streams().streams[0].parameters.extradata.clone();
	let mut demuxer = FlacDemuxer::new(memory(fixture("flac/s16_stereo_44k.flac"))).unwrap();
	// Start from a provisional STREAMINFO, as an encoder would: no sample
	// count, no MD5, no frame size bounds.
	let mut stream = demuxer.streams().streams[0].clone();
	let mut info = rmp::codecs::flac::StreamInfo::from_extradata(&exact).unwrap();
	info.total_samples = 0;
	info.md5 = [0; 16];
	info.min_frame_size = 0;
	info.max_frame_size = 0;
	stream.parameters.extradata = info.to_bytes().to_vec();
	let out = MemoryIo::new();
	let mut muxer = MkvMuxer::new(Box::new(out.clone()), false);
	muxer.add(&stream).unwrap();
	for p in read_all(&mut demuxer) {
		muxer.write(p).unwrap();
	}
	muxer.finish().unwrap();
	let back = MkvDemuxer::new(memory(out.contents())).unwrap();
	assert_eq!(back.streams().streams[0].parameters.extradata, exact);
}

#[test]
fn ogg_opus_into_webm_keeps_codec_delay_and_trimming() {
	let mut demuxer = OggDemuxer::new(memory(fixture("ogg/mono.opus"))).unwrap();
	let (bytes, packets) = remux(&mut demuxer, true);
	assert!(bytes.windows(4).any(|w| w == b"webm"));
	let mut back = MkvDemuxer::new(memory(bytes)).unwrap();
	let s = &back.streams().streams[0];
	assert!(matches!(s.metadata.raw.get("encoder_delay"), Some(RawValue::String(v)) if v == "312"));
	let read_back = read_all(&mut back);
	assert_same_packets("mono.opus", &packets, &read_back);
	let (last_in, last_out) = (packets.last().unwrap(), read_back.last().unwrap());
	assert_eq!(
		last_in.duration.and_then(|d| d.rescale(TimeBase::new(1, 48_000))),
		last_out.duration.and_then(|d| d.rescale(TimeBase::new(1, 48_000)))
	);
}

#[test]
fn webm_refuses_codecs_outside_its_subset() {
	let demuxer = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	let mut muxer = MkvMuxer::new(Box::new(MemoryIo::new()), true);
	assert!(muxer.add(&demuxer.streams().streams[0]).is_err(), "h264 in webm");
	assert!(muxer.add(&demuxer.streams().streams[1]).is_err(), "aac in webm");
	let mut mkv = MkvMuxer::new(Box::new(MemoryIo::new()), false);
	assert!(mkv.add(&demuxer.streams().streams[0]).is_ok());
}

#[test]
fn streams_must_come_before_packets() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	let mut muxer = MkvMuxer::new(Box::new(MemoryIo::new()), false);
	muxer.add(&demuxer.streams().streams[0]).unwrap();
	let video = std::iter::from_fn(|| demuxer.read().unwrap()).find(|p| p.stream_id.0 == 0).unwrap();
	muxer.write(video).unwrap();
	assert!(muxer.add(&demuxer.streams().streams[1]).is_err());
	let mut stray = Packet::new(rmp::core::StreamId(7), vec![1, 2, 3]);
	stray.pts = None;
	assert!(muxer.write(stray).is_err());
}

/// An output that can't seek (a pipe): no back-patching, still a valid
/// live-style file.
struct Pipe(MemoryIo);

impl Io for Pipe {
	fn read(&mut self, buf: &mut [u8]) -> rmp::Result<usize> {
		self.0.read(buf)
	}
	fn write(&mut self, buf: &[u8]) -> rmp::Result<usize> {
		self.0.write(buf)
	}
}

#[test]
fn unseekable_output_is_still_readable() {
	let mut demuxer = MkvDemuxer::new(memory(fixture("mkv/vp9_opus_vtt.webm"))).unwrap();
	let sink = MemoryIo::new();
	let mut muxer = MkvMuxer::new(Box::new(Pipe(sink.clone())), true);
	muxer.set_metadata(Metadata::default()).unwrap();
	for s in demuxer.streams().iter() {
		muxer.add(s).unwrap();
	}
	let packets = read_all(&mut demuxer);
	for p in &packets {
		muxer.write(p.clone()).unwrap();
	}
	muxer.finish().unwrap();
	let mut back = MkvDemuxer::new(memory(sink.contents())).unwrap();
	assert_same_packets("unseekable", &packets, &read_all(&mut back));
}

#[test]
fn raw_video_round_trips_through_v_uncompressed() {
	use rmp::container::y4m::demuxer::Y4mDemuxer;
	let mut demuxer = Y4mDemuxer::new(memory(fixture("y4m/yuv420p_34x18.y4m"))).unwrap();
	let (bytes, packets) = remux(&mut demuxer, false);
	let mut back = MkvDemuxer::new(memory(bytes)).unwrap();
	let p = &back.streams().streams[0].parameters;
	assert_eq!(p.codec.id, "rawvideo");
	assert_eq!(p.pixel_format.as_deref(), Some("yuv420p"));
	assert_eq!((p.width, p.height), (Some(34), Some(18)));
	let read_back = read_all(&mut back);
	assert_eq!(packets.len(), read_back.len());
	for (a, b) in packets.iter().zip(&read_back) {
		assert_eq!(a.data, b.data);
	}
}
