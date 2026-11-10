//! Ogg muxer tests. Remuxing an FFmpeg-made fixture must give back the
//! same packets and timestamps, and the same page structure (flags,
//! granule positions, segment counts) FFmpeg writes for the audio pages.

use crate::common::{fixture, memory, read_all};
use rmp::codecs::vorbis::VorbisParser;
use rmp::container::flac::demuxer::FlacDemuxer;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::ogg::demuxer::{OggDemuxer, crc32, parse_page};
use rmp::container::ogg::muxer::OggMuxer;
use rmp::core::{Demuxer, Muxer, Packet, RawValue, TimeBase};
use rmp::io::MemoryIo;

fn remux(demuxer: &mut dyn Demuxer, opus_only: bool) -> (Vec<u8>, Vec<Packet>) {
	let out = MemoryIo::new();
	let mut muxer = OggMuxer::new(Box::new(out.clone()), opus_only);
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

/// `(flags, granule, segments)` of every page.
fn pages(data: &[u8]) -> Vec<(u8, i64, usize)> {
	let mut out = Vec::new();
	let mut at = 0;
	while at < data.len() {
		let (page, len) = parse_page(&data[at..]).expect("valid page with a correct CRC");
		let granule = page.granule.map_or(-1, |g| g as i64);
		out.push((page.flags, granule, page.lacing.len()));
		at += len;
	}
	out
}

fn round_trip(name: &str) {
	let source = fixture(&format!("ogg/{name}"));
	let mut demuxer = OggDemuxer::new(memory(source.clone())).unwrap();
	let original_stream = demuxer.streams().streams[0].clone();
	let original_meta = demuxer.metadata().clone();
	let (bytes, packets) = remux(&mut demuxer, name.ends_with(".opus"));
	let mut back = OggDemuxer::new(memory(bytes.clone())).unwrap();
	let s = &back.streams().streams[0];
	assert_eq!(s.parameters.codec, original_stream.parameters.codec, "{name}");
	assert_eq!(s.parameters.sample_rate, original_stream.parameters.sample_rate, "{name}");
	assert_eq!(s.parameters.channels, original_stream.parameters.channels, "{name}");
	match s.parameters.codec.id.as_str() {
		// Vorbis extradata holds the comment header, which is rebuilt.
		"vorbis" => {}
		// FFmpeg's Ogg FLAC keeps the encoder's provisional STREAMINFO (no
		// sample count or MD5); RMP rewrites the header page with the
		// real values.
		"flac" => {
			use rmp::codecs::flac::StreamInfo;
			let ours = StreamInfo::from_extradata(&s.parameters.extradata).unwrap();
			let theirs = StreamInfo::from_extradata(&original_stream.parameters.extradata).unwrap();
			assert_eq!(
				(ours.sample_rate, ours.channels, ours.bits_per_sample),
				(theirs.sample_rate, theirs.channels, theirs.bits_per_sample)
			);
			assert_eq!(theirs.total_samples, 0);
			assert_eq!(ours.total_samples, 88_200, "{name}: 2 s at 44.1 kHz");
			assert_ne!(ours.md5, [0; 16]);
		}
		_ => {
			assert_eq!(s.parameters.extradata, original_stream.parameters.extradata, "{name}: extradata")
		}
	}
	assert_eq!(back.duration(), demuxer_duration(&source), "{name}: duration");
	assert_eq!(back.metadata().title, original_meta.title, "{name}: title");
	assert_eq!(back.metadata().artist, original_meta.artist, "{name}: artist");
	assert_eq!(back.metadata().album, original_meta.album, "{name}: album");
	assert_eq!(back.metadata().images.len(), original_meta.images.len(), "{name}: pictures");

	let read_back = read_all(&mut back);
	assert_eq!(packets.len(), read_back.len(), "{name}: packet count");
	for (i, (a, b)) in packets.iter().zip(&read_back).enumerate() {
		let at = format!("{name} packet {i}");
		assert_eq!(a.data, b.data, "{at}: payload");
		assert_eq!(a.pts, b.pts, "{at}: pts");
		assert_eq!(a.duration, b.duration, "{at}: duration");
	}

	// Audio pages: identical to FFmpeg's (the header pages differ only by
	// the comment header's size).
	let ours = pages(&bytes);
	let ffmpeg = pages(&source);
	let headers = |p: &[(u8, i64, usize)]| p.iter().take_while(|(_, g, _)| *g == 0).count();
	assert_eq!(&ours[headers(&ours)..], &ffmpeg[headers(&ffmpeg)..], "{name}: audio pages");
	assert_eq!(ours[0].0, 2, "{name}: BOS page first");
	assert_eq!(ours.last().unwrap().0 & 4, 4, "{name}: EOS page last");
}

fn demuxer_duration(data: &[u8]) -> rmp::core::Time {
	OggDemuxer::new(memory(data.to_vec())).unwrap().duration()
}

#[test]
fn opus_mono_round_trips() {
	round_trip("mono.opus");
}

#[test]
fn opus_with_cover_round_trips() {
	round_trip("st_cover.opus");
}

#[test]
fn opus_51_round_trips() {
	round_trip("51.opus");
}

#[test]
fn flac_round_trips() {
	round_trip("flac.oga");
}

#[test]
fn vorbis_round_trips() {
	round_trip("vorbis.ogg");
}

#[test]
fn vorbis_parser_matches_granules() {
	let demuxer = OggDemuxer::new(memory(fixture("ogg/vorbis.ogg"))).unwrap();
	let extradata = demuxer.streams().streams[0].parameters.extradata.clone();
	let mut parser = VorbisParser::from_extradata(&extradata).unwrap();
	let mut demuxer = OggDemuxer::new(memory(fixture("ogg/vorbis.ogg"))).unwrap();
	let packets = read_all(&mut demuxer);
	let durations: Vec<u64> =
		packets.iter().map(|p| u64::from(parser.packet_samples(&p.data).unwrap())).collect();
	// The first packet only primes the decoder; the rest add up to the last
	// granule position (96000 = 2 s at 48 kHz).
	assert_eq!(durations[1..].iter().sum::<u64>(), 96_000);
	assert_eq!(durations[0], 128);
	assert!(durations.iter().all(|&d| [128, 576, 1024].contains(&d)));
}

#[test]
fn webm_opus_becomes_sample_exact_ogg_opus() {
	// celt_51.webm ends with DiscardPadding; its timestamps are 1 ms ticks.
	let mut demuxer = MkvDemuxer::new(memory(fixture("opus/celt_51.webm"))).unwrap();
	let (bytes, packets) = remux(&mut demuxer, true);
	let last = pages(&bytes).last().copied().unwrap();
	// 26 packets of 20 ms, the last trimmed to 312 samples, plus pre-skip.
	let kept: u64 = packets
		.iter()
		.map(|p| p.duration.unwrap().rescale(TimeBase::new(1, 48_000)).unwrap().ticks())
		.sum();
	assert_eq!(kept, 25 * 960 + 312);
	assert_eq!(last.1, kept as i64);
	let back = OggDemuxer::new(memory(bytes)).unwrap();
	assert!(
		matches!(back.streams().streams[0].metadata.raw.get("encoder_delay"), Some(RawValue::String(v)) if v == "312")
	);
}

#[test]
fn flac_header_page_gets_the_final_streaminfo() {
	let exact =
		FlacDemuxer::new(memory(fixture("flac/s16_stereo_44k.flac"))).unwrap().streams().streams[0]
			.parameters
			.extradata
			.clone();
	let mut demuxer = FlacDemuxer::new(memory(fixture("flac/s16_stereo_44k.flac"))).unwrap();
	let mut stream = demuxer.streams().streams[0].clone();
	let mut info = rmp::codecs::flac::StreamInfo::from_extradata(&exact).unwrap();
	info.total_samples = 0;
	info.md5 = [0; 16];
	stream.parameters.extradata = info.to_bytes().to_vec();
	let out = MemoryIo::new();
	let mut muxer = OggMuxer::new(Box::new(out.clone()), false);
	muxer.add(&stream).unwrap();
	for p in read_all(&mut demuxer) {
		muxer.write(p).unwrap();
	}
	muxer.finish().unwrap();
	let bytes = out.contents();
	pages(&bytes); // every CRC, including the rewritten page's, is valid
	let back = OggDemuxer::new(memory(bytes)).unwrap();
	assert_eq!(back.streams().streams[0].parameters.extradata, exact);
}

#[test]
fn opus_files_hold_only_opus() {
	let flac = OggDemuxer::new(memory(fixture("ogg/flac.oga"))).unwrap();
	let mut opus = OggMuxer::new(Box::new(MemoryIo::new()), true);
	assert!(opus.add(&flac.streams().streams[0]).is_err());
	let mut ogg = OggMuxer::new(Box::new(MemoryIo::new()), false);
	assert!(ogg.add(&flac.streams().streams[0]).is_ok());
	let mkv = MkvDemuxer::new(memory(fixture("mkv/h264_aac.mkv"))).unwrap();
	assert!(ogg.add(&mkv.streams().streams[0]).is_err(), "video in ogg");
}

#[test]
fn page_crc_is_the_ogg_polynomial() {
	let (bytes, _) = remux(&mut OggDemuxer::new(memory(fixture("ogg/mono.opus"))).unwrap(), true);
	let len = 27
		+ usize::from(bytes[26])
		+ bytes[27..27 + usize::from(bytes[26])].iter().map(|&l| usize::from(l)).sum::<usize>();
	let mut page = bytes[..len].to_vec();
	let stored = u32::from_le_bytes(page[22..26].try_into().unwrap());
	page[22..26].fill(0);
	assert_eq!(crc32(&page), stored);
}
