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
	let sources = [fixture("mp4/h264_bframes_aac.mp4"), fixture("mp4/aac_tagged_cover.m4a")];
	let mut rng = Rng::new(seed() ^ 4);
	for round in 0..rounds() * 2 {
		let data = mutate(&sources[(round % 2) as usize], &mut rng);
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
	let webm = ["silk_nb_mono", "silk_wb_st_40ms", "hybrid_st", "mode_switch", "celt_51"];
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
