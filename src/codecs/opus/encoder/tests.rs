use super::*;
use crate::codecs::opus::OpusDecoderState;
use crate::core::Encoder as _;

/// Test signals, interleaved, unit scale.
fn signal(kind: &str, frames: usize, channels: usize) -> Vec<f32> {
	let mut seed = 0x1234_5678u32;
	let mut noise = move || {
		seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
		(seed >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
	};
	let mut out = Vec::with_capacity(frames * channels);
	for i in 0..frames {
		let t = i as f32 / 48_000.0;
		for ch in 0..channels {
			let v = match kind {
				"tone" => 0.5 * (2.0 * std::f32::consts::PI * (440.0 + 110.0 * ch as f32) * t).sin(),
				"chord" => {
					let f = [220.0, 277.2, 329.6, 1760.0, 5000.0];
					f.iter()
						.enumerate()
						.map(|(k, &f)| 0.12 * (2.0 * std::f32::consts::PI * f * t + k as f32 + ch as f32).sin())
						.sum()
				}
				"noise" => 0.3 * noise(),
				"clicks" => {
					// Silence with sharp bursts every 50 ms.
					if i % 2400 < 64 { 0.8 * noise() } else { 0.002 * noise() }
				}
				"chirp" => 0.4 * (2.0 * std::f32::consts::PI * (100.0 + 8000.0 * t) * t).sin(),
				"silence" => 0.0,
				_ => unreachable!(),
			};
			out.push(v.clamp(-1.0, 1.0));
		}
	}
	out
}

/// Encodes `pcm` and decodes every packet with RMP's decoder, checking
/// that each packet's final range-coder state matches. Returns the
/// decoded samples and the packets.
fn round_trip(
	pcm: &[f32],
	channels: u8,
	bitrate: u32,
	frame: usize,
) -> (Vec<f32>, Vec<Vec<u8>>, OpusEncoderState) {
	let mut enc = OpusEncoderState::new(channels, bitrate, frame).expect("encoder");
	let mut dec = OpusDecoderState::new(enc.head());
	let mut out = Vec::new();
	let mut packets = Vec::new();
	enc.push(pcm);
	let mut check = |p: Vec<u8>, range: u32, out: &mut Vec<f32>| {
		assert_eq!(p.len(), enc_bytes(bitrate, frame) + 1);
		let samples = dec.decode_packet(&p).expect("decodes");
		assert_eq!(samples.len(), frame * usize::from(channels));
		assert_eq!(dec.final_range(), range, "packet {}: final range", packets.len());
		assert_eq!(dec.concealed_frames, 0);
		out.extend(samples);
		packets.push(p);
	};
	while let Some(p) = enc.next_packet().or_else(|| enc.flush_packet()) {
		let r = enc.final_range();
		check(p, r, &mut out);
	}
	(out, packets, enc)
}

fn enc_bytes(bitrate: u32, frame: usize) -> usize {
	OpusEncoderState::new(1, bitrate, frame).expect("encoder").frame_bytes()
}

/// Signal-to-noise ratio of `decoded` against `input` delayed by the
/// pre-skip, over the first channel.
fn snr(input: &[f32], decoded: &[f32], channels: usize) -> f64 {
	let d = usize::from(PRE_SKIP);
	let n = input.len() / channels;
	let (mut sig, mut err) = (0f64, 0f64);
	for i in 2400..n.saturating_sub(2400) {
		let a = f64::from(input[i * channels]);
		let b = f64::from(decoded[(i + d) * channels]);
		sig += a * a;
		err += (a - b) * (a - b);
	}
	10.0 * (sig / err.max(1e-12)).log10()
}

#[test]
fn final_range_matches_the_decoder() {
	for channels in [1u8, 2] {
		for kind in ["tone", "chord", "noise", "clicks", "chirp", "silence"] {
			for (bitrate, frame) in [
				(12_000, 960),
				(32_000, 960),
				(96_000, 960),
				(256_000, 960),
				(64_000, 480),
				(64_000, 240),
				(128_000, 120),
				(510_000, 960),
			] {
				let pcm = signal(kind, 48_000 / 5, usize::from(channels));
				round_trip(&pcm, channels, bitrate * u32::from(channels), frame);
			}
		}
	}
}

#[test]
fn decoded_audio_tracks_the_input() {
	for (kind, channels, bitrate, min_snr) in
		[("tone", 1u8, 64_000, 20.0), ("chord", 2, 128_000, 15.0), ("chirp", 1, 96_000, 12.0)]
	{
		let pcm = signal(kind, 48_000, usize::from(channels));
		let (out, _, _) = round_trip(&pcm, channels, bitrate, 960);
		let s = snr(&pcm, &out, usize::from(channels));
		assert!(s > min_snr, "{kind}: SNR {s:.1} dB");
	}
	// More bits, less error.
	let pcm = signal("chord", 48_000, 1);
	let (lo, _, _) = round_trip(&pcm, 1, 24_000, 960);
	let (hi, _, _) = round_trip(&pcm, 1, 160_000, 960);
	assert!(snr(&pcm, &hi, 1) > snr(&pcm, &lo, 1) + 6.0);
}

#[test]
fn attacks_use_short_blocks() {
	let pcm = signal("clicks", 48_000, 1);
	let (_, _, enc) = round_trip(&pcm, 1, 64_000, 960);
	assert!(enc.celt().transients > 5, "transient frames: {}", enc.celt().transients);
	let pcm = signal("tone", 48_000, 1);
	let (_, _, enc) = round_trip(&pcm, 1, 64_000, 960);
	// Only the tone's onset and its cut-off at the end are attacks.
	assert!(enc.celt().transients <= 2, "transient frames: {}", enc.celt().transients);
}

#[test]
fn encoder_trait_packetises_with_pre_skip() {
	use crate::core::{AudioFrame, Channels, CodecId, SampleFormat, SampleLayout, Stream, StreamId};
	let enc = OpusEncoder::new(StreamId(3)).with_bitrate(Some(96_000));
	let mut params = Stream::audio(StreamId(3), 48_000, 2, CodecId::new("opus")).parameters;
	let head = OpusHead::parse(&enc.extradata(&params).expect("extradata")).expect("head");
	assert_eq!((head.channels, head.pre_skip), (2, PRE_SKIP));
	let mut state = enc.create_state(&params);
	let pcm = signal("chord", 5_000, 2);
	let mut packets = Vec::new();
	for chunk in pcm.chunks(1400) {
		let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
		let frame = AudioFrame::new(
			bytes,
			48_000,
			Channels::from_count(2),
			SampleFormat::F32,
			SampleLayout::Interleaved,
		);
		packets.extend(enc.encode(&mut state, Frame::Audio(frame)).expect("encodes").value);
	}
	packets.extend(enc.flush(&mut state).expect("flushes").value);
	// 5000 samples + 120 of delay: six 20 ms packets, the last one short.
	assert_eq!(packets.len(), 6);
	for (k, p) in packets.iter().enumerate() {
		assert_eq!(p.stream_id, StreamId(3));
		assert_eq!(p.pts.map(|t| t.ticks()), Some(k as u64 * 960));
	}
	assert_eq!(packets[5].duration.map(|t| t.ticks()), Some(5_120 - 5 * 960));
	let mut dec = OpusDecoderState::new(head);
	for p in &packets {
		assert_eq!(dec.decode_packet(&p.data).expect("decodes").len(), 960 * 2);
	}
	// Wrong rate: no state, and encode fails cleanly.
	params.sample_rate = Some(44_100);
	assert!(enc.extradata(&params).is_err());
	let mut bad = enc.create_state(&params);
	let frame = AudioFrame::new(
		vec![0; 64],
		44_100,
		Channels::from_count(2),
		SampleFormat::S16,
		SampleLayout::Interleaved,
	);
	assert!(enc.encode(&mut bad, Frame::Audio(frame)).is_err());
}
