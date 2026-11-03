//! Corrupted and truncated input must produce errors or end-of-stream,
//! never a panic (release builds use `panic = "abort"`).

use crate::common::{Rng, fixture, memory};
use rmp::codecs::mp3::{Mp3Decoder, Mp3DecoderState, decoder::decode_packet};
use rmp::container::mp3::demuxer::Mp3Demuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::core::{Decoder, Demuxer};

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

#[test]
fn random_bytes_are_rejected_cleanly() {
	let mut rng = Rng::new(42);
	for len in [0usize, 1, 7, 8, 16, 100, 5000] {
		let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
		assert!(Mp4Demuxer::new(memory(data.clone())).is_err() || len >= 8);
		let _ = Mp3Demuxer::new(memory(data.clone()));
		let _ = decode_packet(&mut Mp3DecoderState::new(), &data);
	}
}
