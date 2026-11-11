//! Corrupted and truncated input must produce errors or end-of-stream,
//! never a panic (release builds use `panic = "abort"`).

use crate::common::{Rng, fixture, memory};
use rmp::codecs::mp3::{Mp3Decoder, Mp3DecoderState, decoder::decode_packet};
use rmp::container::flac::demuxer::FlacDemuxer;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp3::demuxer::Mp3Demuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::ogg::demuxer::OggDemuxer;
use rmp::container::riff::demuxer::RiffDemuxer;
use rmp::container::y4m::demuxer::Y4mDemuxer;
use rmp::core::{Decoder, Demuxer, Resolver};

/// Override with `RMP_FUZZ_ROUNDS=20000 cargo test robustness` for a deeper run.
fn rounds() -> u64 {
	std::env::var("RMP_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150)
}

fn seed() -> u64 {
	std::env::var("RMP_FUZZ_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x5EED)
}

fn drain(demuxer: &mut dyn Demuxer) {
	for _ in 0..10_000 {
		match demuxer.read() {
			Ok(Some(_)) => {}
			Ok(None) | Err(_) => break,
		}
	}
	let _ = demuxer.seek(0.3);
	let _ = demuxer.read();
}

fn mutate(data: &[u8], rng: &mut Rng) -> Vec<u8> {
	let mut out = data.to_vec();
	match rng.below(3) {
		0 => out.truncate(rng.below(data.len())),
		1 => {
			for _ in 0..1 + rng.below(32) {
				let i = rng.below(out.len());
				out[i] = rng.next() as u8;
			}
		}
		_ => {
			// Overwrite a run with 0xFF / 0x00 (sync words, sizes).
			let start = rng.below(out.len());
			let len = rng.below(16).min(out.len() - start);
			let value = if rng.below(2) == 0 { 0xFF } else { 0x00 };
			out[start..start + len].fill(value);
		}
	}
	out
}

#[test]
fn mp3_demuxer_and_decoder_survive_corruption() {
	let sources = [fixture("mp3/cbr_44k_stereo_tagged.mp3"), fixture("mp3/mpeg2_22k_mono_raw.mp3")];
	let mut rng = Rng::new(seed() ^ 3);
	for round in 0..rounds() {
		let data = mutate(&sources[(round % 2) as usize], &mut rng);
		// Decoder on raw corrupted bytes.
		let _ = decode_packet(&mut Mp3DecoderState::new(), &data);
		// Demuxer + decoder pipeline.
		if let Ok(mut demuxer) = Mp3Demuxer::new(memory(data)) {
			let stream = demuxer.streams().streams[0].clone();
			let decoder = Mp3Decoder::new();
			let mut state = decoder.create_state(&stream.parameters);
			while let Ok(Some(packet)) = demuxer.read() {
				let _ = decoder.decode(&mut state, packet);
			}
			drain(&mut demuxer);
		}
	}
}

#[test]
fn mp4_demuxer_survives_corruption() {
	let sources = [
		fixture("mp4/h264_bframes_aac.mp4"),
		fixture("mp4/aac_tagged_cover.m4a"),
		fixture("mp4/h264_aac.ffmpeg_frag.mp4"),
	];
	let mut rng = Rng::new(seed() ^ 4);
	for round in 0..rounds() * 3 {
		let data = mutate(&sources[(round % 3) as usize], &mut rng);
		if let Ok(mut demuxer) = Mp4Demuxer::new(memory(data)) {
			drain(&mut demuxer);
		}
	}
}

/// Demuxes and decodes every packet of every stream, ignoring errors.
fn decode_every_stream(demuxer: &mut dyn Demuxer) {
	let resolver = Resolver::new();
	let mut decoders: Vec<_> = demuxer
		.streams()
		.iter()
		.filter_map(|s| resolver.decoder_for(s).map(|d| (s.id, d.create_state(&s.parameters), d)))
		.collect();
	for _ in 0..10_000 {
		match demuxer.read() {
			Ok(Some(packet)) => {
				if let Some((_, state, decoder)) =
					decoders.iter_mut().find(|(id, _, _)| *id == packet.stream_id)
				{
					let _ = decoder.decode(state, packet);
				}
			}
			Ok(None) | Err(_) => break,
		}
	}
	for (_, state, decoder) in decoders.iter_mut() {
		let _ = decoder.flush(state);
	}
	drain(demuxer);
}

#[test]
fn mkv_demuxer_and_decoders_survive_corruption() {
	let sources = [
		fixture("mkv/h264_aac.mkv"),
		fixture("mkv/h264_flac_srt.mkv"),
		fixture("mkv/vp9_opus_vtt.webm"),
		fixture("mkv/live_vp9_opus.webm"),
	];
	let mut rng = Rng::new(seed() ^ 9);
	for round in 0..rounds() * 2 {
		let data = mutate(&sources[round as usize % sources.len()], &mut rng);
		if let Ok(mut demuxer) = MkvDemuxer::new(memory(data)) {
			decode_every_stream(&mut demuxer);
		}
	}
}

#[test]
fn flac_demuxer_and_decoder_survive_corruption() {
	let sources = [
		fixture("flac/s16_stereo_44k.flac"),
		fixture("flac/wasted_mono_8k_lvl0.flac"),
		fixture("flac/s24_51_96k_lvl12.flac"),
		fixture("flac/s24_rs_32k.flac"),
	];
	let mut rng = Rng::new(seed() ^ 10);
	for round in 0..rounds() * 2 {
		let data = mutate(&sources[round as usize % sources.len()], &mut rng);
		if let Ok(mut demuxer) = FlacDemuxer::new(memory(data.clone())) {
			decode_every_stream(&mut demuxer);
		}
		// Raw corrupted bytes straight into the frame decoder.
		let tail = &data[data.len().min(42)..];
		let _ = rmp::codecs::flac::decoder::decode_frame(tail, None);
		let _ = rmp::codecs::flac::FlacDecoderState::default().decode_packet(tail);
	}
}

#[test]
fn vp9_decoder_survives_corrupted_frames() {
	use rmp::codecs::vp9::Vp9DecoderState;
	let names = [
		"altref_176x144",
		"odd_444_99x61",
		"p1_422_96x64",
		"lossless_64x48",
		"tiles_512x128",
		"rt_aq_errres_160x96",
	];
	let mut rng = Rng::new(seed() ^ 17);
	for (n, name) in names.iter().enumerate() {
		let data = fixture(&format!("vp9/{name}.webm"));
		let mut demuxer = MkvDemuxer::new(memory(data)).unwrap();
		let packets: Vec<Vec<u8>> =
			std::iter::from_fn(|| demuxer.read().ok().flatten()).map(|p| p.data).collect();
		for round in 0..rounds() / 6 + 1 {
			// Damage a few packets (bit flips, truncation, garbage) in an
			// otherwise valid stream, so later frames reference broken ones.
			let mut dec = Vp9DecoderState::new();
			for (i, packet) in packets.iter().enumerate() {
				let mut p = packet.clone();
				if (i as u64 + round + n as u64).is_multiple_of(3) && !p.is_empty() {
					match rng.below(3) {
						0 => p.truncate(rng.below(p.len())),
						1 => {
							for _ in 0..1 + rng.below(16) {
								let at = rng.below(p.len());
								p[at] ^= 1 << rng.below(8);
							}
						}
						_ => {
							let at = rng.below(p.len());
							for b in &mut p[at..] {
								*b = rng.next() as u8;
							}
						}
					}
				}
				let _ = dec.decode_packet(&p);
			}
		}
	}
	// Random bytes behind a valid key frame header byte.
	let mut dec = Vp9DecoderState::new();
	for _ in 0..rounds() * 2 {
		let len = 1 + rng.below(300);
		let mut p: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
		p[0] = 0x82 | (p[0] & 0x30);
		let _ = dec.decode_packet(&p);
	}
	// Whole corrupted WebM files through the demuxer and decoder.
	for round in 0..rounds() / 2 {
		let name = names[round as usize % names.len()];
		if let Ok(mut demuxer) =
			MkvDemuxer::new(memory(mutate(&fixture(&format!("vp9/{name}.webm")), &mut rng)))
		{
			decode_every_stream(&mut demuxer);
		}
	}
}

#[test]
fn vorbis_headers_and_packets_survive_corruption() {
	let sources = [fixture("vorbis/mono_22k_q-1.ogg"), fixture("vorbis/51_22k.ogg")];
	let mut rng = Rng::new(seed() ^ 13);
	for round in 0..rounds() {
		let data = mutate(&sources[round as usize % sources.len()], &mut rng);
		if let Ok(mut demuxer) = OggDemuxer::new(memory(data)) {
			decode_every_stream(&mut demuxer);
		}
	}
	let webm = fixture("vorbis/st_44k_q8.webm");
	for _ in 0..rounds() / 2 {
		if let Ok(mut demuxer) = MkvDemuxer::new(memory(mutate(&webm, &mut rng))) {
			decode_every_stream(&mut demuxer);
		}
	}
	// Damaged setup headers, and random audio packets for a valid setup.
	use rmp::codecs::vorbis::VorbisDecoderState;
	let demuxer = OggDemuxer::new(memory(sources[1].clone())).unwrap();
	let extradata = demuxer.streams().streams[0].parameters.extradata.clone();
	let headers = rmp::container::xiph::vorbis_headers(&extradata).unwrap();
	for _ in 0..rounds() * 2 {
		let mut setup = headers[2].clone();
		for _ in 0..1 + rng.below(8) {
			let at = 7 + rng.below(setup.len() - 7);
			setup[at] ^= 1 << rng.below(8);
		}
		if rng.below(4) == 0 {
			setup.truncate(7 + rng.below(setup.len() - 7));
		}
		if let Ok(mut state) = VorbisDecoderState::new(&headers[0], &setup) {
			for _ in 0..8 {
				let len = rng.below(300);
				let packet: Vec<u8> = (0..len).map(|_| rng.next() as u8 & !1).collect();
				let _ = state.decode_packet(&packet);
			}
		}
	}
	let mut state = VorbisDecoderState::new(&headers[0], &headers[2]).unwrap();
	for _ in 0..rounds() * 4 {
		let len = rng.below(600);
		let packet: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
		let _ = state.decode_packet(&packet);
	}
}

#[test]
fn ogg_demuxer_and_opus_decoder_survive_corruption() {
	let sources = [
		fixture("ogg/mono.opus"),
		fixture("ogg/st_cover.opus"),
		fixture("ogg/51.opus"),
		fixture("ogg/flac.oga"),
		fixture("ogg/vorbis.ogg"),
	];
	let mut rng = Rng::new(seed() ^ 11);
	for round in 0..rounds() * 2 {
		let data = mutate(&sources[round as usize % sources.len()], &mut rng);
		if let Ok(mut demuxer) = OggDemuxer::new(memory(data)) {
			decode_every_stream(&mut demuxer);
		}
	}
	// Corrupted SILK / hybrid / CELT / multistream packets.
	let webm = [
		"silk_nb_mono",
		"silk_wb_st_40ms",
		"hybrid_st",
		"mode_switch",
		"celt_51",
		"plc_silk_fec",
		"plc_hybrid_fec_st",
		"plc_celt",
	];
	for round in 0..rounds() * 2 {
		let data =
			mutate(&fixture(&format!("opus/{}.webm", webm[round as usize % webm.len()])), &mut rng);
		if let Ok(mut demuxer) = MkvDemuxer::new(memory(data)) {
			decode_every_stream(&mut demuxer);
		}
	}
	// Random packets straight into the decoder, for each channel layout.
	use rmp::codecs::opus::OpusDecoderState;
	for (extradata, channels) in [(fixture("ogg/mono.opus")[28..47].to_vec(), 1), (Vec::new(), 2)] {
		let mut state = OpusDecoderState::from_extradata(&extradata, channels);
		for _ in 0..rounds() * 4 {
			let len = rng.below(400);
			let packet: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
			let _ = state.decode_packet(&packet);
		}
	}
	// Real packets with random losses, FEC recoveries, damaged packets and
	// timestamp jumps, exercising every concealment path from every mode.
	for name in ["plc_silk_fec", "plc_hybrid_fec_st", "plc_celt", "mode_switch", "celt_51"] {
		let Ok(demuxer) = MkvDemuxer::new(memory(fixture(&format!("opus/{name}.webm")))) else {
			continue;
		};
		let extradata = demuxer.streams().streams[0].parameters.extradata.clone();
		let mut demuxer = demuxer;
		let mut packets = Vec::new();
		while let Ok(Some(packet)) = demuxer.read() {
			packets.push(packet.data);
		}
		for _ in 0..rounds().div_ceil(20) {
			let mut state = OpusDecoderState::from_extradata(&extradata, 2);
			let mut pts = 0u64;
			for packet in &packets {
				let mut packet = packet.clone();
				match rng.below(10) {
					0 => {
						let _ = state.conceal(120 * (1 + rng.below(100)));
					}
					1 => {
						let _ = state.decode_fec(&packet, 120 * (1 + rng.below(60)));
					}
					2 => {
						let at = rng.below(packet.len());
						packet[at] ^= 1 << rng.below(8);
						packet.truncate(at + 1);
					}
					3 => pts += rng.below(96_000) as u64,
					_ => {}
				}
				let (out, _) = state.decode_with_recovery(&packet, Some(pts), None);
				assert!(out.iter().all(|s| s.is_finite()), "{name}: non-finite output");
				pts += (out.len() / state.channels()) as u64;
			}
		}
	}
}

/// Demuxes and decodes every packet of the first stream, ignoring errors.
fn decode_everything(demuxer: &mut dyn Demuxer) {
	let Some(stream) = demuxer.streams().streams.first().cloned() else {
		return;
	};
	let Some(decoder) = Resolver::new().decoder_for(&stream) else {
		return;
	};
	let mut state = decoder.create_state(&stream.parameters);
	for _ in 0..10_000 {
		match demuxer.read() {
			Ok(Some(packet)) => {
				let _ = decoder.decode(&mut state, packet);
			}
			Ok(None) | Err(_) => break,
		}
	}
	drain(demuxer);
}

#[test]
fn wav_demuxer_and_pcm_adpcm_decoders_survive_corruption() {
	let sources = [
		fixture("wav/adpcm/ima_stereo_22k.wav"),
		fixture("wav/adpcm/ms_mono_8k.wav"),
		fixture("wav/adpcm/ms_stereo_11k.wav"),
		fixture("wav/pcm/s24le.wav"),
		fixture("wav/pcm/f64le.wav"),
		fixture("wav/pcm/alaw.wav"),
	];
	let mut rng = Rng::new(seed() ^ 5);
	for round in 0..rounds() * 2 {
		let data = mutate(&sources[round as usize % sources.len()], &mut rng);
		if let Ok(mut demuxer) = RiffDemuxer::new(memory(data.clone())) {
			decode_everything(&mut demuxer);
		}
		// Raw corrupted bytes straight into the block decoders.
		use rmp::codecs::adpcm::{AdpcmCodec, decoder::decode_block};
		let tail = &data[data.len().min(44)..];
		for codec in [AdpcmCodec::ImaWav, AdpcmCodec::Ms] {
			for channels in 1..=3 {
				let _ = decode_block(codec, tail, channels);
			}
		}
	}
}

#[test]
fn y4m_demuxer_survives_corruption() {
	let sources = [fixture("y4m/yuv420p_34x18.y4m"), fixture("y4m/yuv422p10_16x8.y4m")];
	let mut rng = Rng::new(seed() ^ 6);
	for round in 0..rounds() * 2 {
		let data = mutate(&sources[(round % 2) as usize], &mut rng);
		if let Ok(mut demuxer) = Y4mDemuxer::new(memory(data)) {
			decode_everything(&mut demuxer);
		}
	}
}

#[test]
fn random_bytes_are_rejected_cleanly() {
	let mut rng = Rng::new(42);
	for len in [0usize, 1, 7, 8, 16, 100, 5000] {
		let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
		assert!(Mp4Demuxer::new(memory(data.clone())).is_err() || len >= 8);
		let _ = Mp3Demuxer::new(memory(data.clone()));
		let _ = decode_packet(&mut Mp3DecoderState::new(), &data);
		let _ = RiffDemuxer::new(memory(data.clone()));
		let _ = Y4mDemuxer::new(memory(data.clone()));
		let _ = MkvDemuxer::new(memory(data.clone()));
		let _ = FlacDemuxer::new(memory(data.clone()));
		let _ = OggDemuxer::new(memory(data.clone()));
	}
	// Valid magic followed by noise.
	for magic in [
		&b"RIFF\xff\xff\xff\xffWAVE"[..],
		b"YUV4MPEG2 ",
		b"\x1a\x45\xdf\xa3\x88\x42\x82\x88matroska",
		b"fLaC\x80\x00\x00\x22",
		b"OggS\x00\x02",
	] {
		for len in [0usize, 8, 64, 3000] {
			let mut data = magic.to_vec();
			data.extend((0..len).map(|_| rng.next() as u8));
			if let Ok(mut d) = RiffDemuxer::new(memory(data.clone())) {
				decode_everything(&mut d);
			}
			if let Ok(mut d) = Y4mDemuxer::new(memory(data.clone())) {
				decode_everything(&mut d);
			}
			if let Ok(mut d) = MkvDemuxer::new(memory(data.clone())) {
				decode_every_stream(&mut d);
			}
			if let Ok(mut d) = FlacDemuxer::new(memory(data.clone())) {
				decode_every_stream(&mut d);
			}
			if let Ok(mut d) = OggDemuxer::new(memory(data)) {
				decode_every_stream(&mut d);
			}
		}
	}
}

/// Remuxes whatever a (possibly corrupted) demuxer yields into Matroska,
/// WebM and Ogg, then reads each result back. Muxers may refuse streams or
/// packets with an error, but must never panic, and what they write must
/// stay readable.
fn remux_everywhere(
	data: &[u8],
	open: &dyn Fn(Vec<u8>) -> Option<Box<dyn Demuxer>>,
	rng: &mut Rng,
) {
	use rmp::container::mkv::muxer::MkvMuxer;
	use rmp::container::ogg::muxer::OggMuxer;
	use rmp::core::Muxer;
	use rmp::io::MemoryIo;
	let Some(mut demuxer) = open(data.to_vec()) else { return };
	let mut packets = Vec::new();
	for _ in 0..4000 {
		match demuxer.read() {
			Ok(Some(mut p)) => {
				// Damage timestamps too: missing, reordered or huge.
				match rng.below(40) {
					0 => p.pts = None,
					1 => p.duration = None,
					2 => {
						p.pts = Some(rmp::core::Time::new(rng.next() >> 20, rmp::core::TimeBase::new(1, 1000)))
					}
					_ => {}
				}
				packets.push(p);
			}
			_ => break,
		}
	}
	for kind in 0..3 {
		let out = MemoryIo::new();
		let mut muxer: Box<dyn Muxer> = match kind {
			0 => Box::new(MkvMuxer::new(Box::new(out.clone()), false)),
			1 => Box::new(MkvMuxer::new(Box::new(out.clone()), true)),
			_ => Box::new(OggMuxer::new(Box::new(out.clone()), false)),
		};
		let _ = muxer.set_metadata(demuxer.metadata().clone());
		let mut added = Vec::new();
		for s in demuxer.streams().iter() {
			if muxer.add(s).is_ok() {
				added.push(s.id);
			}
		}
		if added.is_empty() {
			continue;
		}
		for p in packets.iter().filter(|p| added.contains(&p.stream_id)) {
			let _ = muxer.write(p.clone());
		}
		if muxer.finish().is_err() {
			continue;
		}
		let bytes = out.contents();
		let back: Option<Box<dyn Demuxer>> = match kind {
			0 | 1 => MkvDemuxer::new(memory(bytes)).ok().map(|d| Box::new(d) as Box<dyn Demuxer>),
			_ => OggDemuxer::new(memory(bytes)).ok().map(|d| Box::new(d) as Box<dyn Demuxer>),
		};
		let mut back = back.expect("a finished muxer output reopens");
		drain(back.as_mut());
	}
}

#[test]
fn mkv_and_ogg_muxers_survive_corrupted_input() {
	let mkv_sources = [
		fixture("mkv/h264_aac.mkv"),
		fixture("mkv/h264_flac_srt.mkv"),
		fixture("mkv/vp9_opus_vtt.webm"),
		fixture("opus/celt_51.webm"),
	];
	let ogg_sources =
		[fixture("ogg/st_cover.opus"), fixture("ogg/flac.oga"), fixture("ogg/vorbis.ogg")];
	let open_mkv =
		|d: Vec<u8>| MkvDemuxer::new(memory(d)).ok().map(|d| Box::new(d) as Box<dyn Demuxer>);
	let open_ogg =
		|d: Vec<u8>| OggDemuxer::new(memory(d)).ok().map(|d| Box::new(d) as Box<dyn Demuxer>);
	let open_mp4 =
		|d: Vec<u8>| Mp4Demuxer::new(memory(d)).ok().map(|d| Box::new(d) as Box<dyn Demuxer>);
	let mut rng = Rng::new(seed() ^ 21);
	for round in 0..rounds() {
		let i = round as usize;
		let data = mutate(&mkv_sources[i % mkv_sources.len()], &mut rng);
		remux_everywhere(&data, &open_mkv, &mut rng);
		let data = mutate(&ogg_sources[i % ogg_sources.len()], &mut rng);
		remux_everywhere(&data, &open_ogg, &mut rng);
		let data = mutate(&fixture("mp4/aac_tagged_cover.m4a"), &mut rng);
		remux_everywhere(&data, &open_mp4, &mut rng);
	}
}
