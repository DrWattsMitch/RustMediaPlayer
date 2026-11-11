//! Opus packet-loss recovery: concealment (SILK PLC and comfort noise, CELT
//! pitch and noise PLC, hybrid), in-band FEC, timestamp gaps and damaged
//! packets.
//!
//! References are libopus 1.4 decodes through `tools/opus_plc_ref.c`, which
//! drops the packets chosen by [`lost`] and conceals each gap the way RMP
//! does (`tools/make_fixtures.sh` regenerates them).

use crate::common::{fixture, memory};
use rmp::codecs::opus::OpusDecoderState;
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::core::{CodecPrivate, Demuxer, Frame, Packet, Resolver, State, StreamKind};

/// The packets the tests drop: single losses, a pair, a 160 ms burst (long
/// enough for CELT to switch to comfort noise), three in a row and a late
/// single loss. The last packet is kept (nothing would reveal the gap).
fn lost(i: usize, count: usize) -> bool {
	i + 1 < count
		&& ([5, 9, 10].contains(&i)
			|| (15..23).contains(&i)
			|| (30..33).contains(&i)
			|| i == count * 3 / 4)
}

fn packets(name: &str) -> (Vec<u8>, Vec<Packet>) {
	let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("opus/{name}.webm")))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
	let mut packets = Vec::new();
	while let Some(packet) = demuxer.read().unwrap() {
		if packet.stream_id == stream.id {
			packets.push(packet);
		}
	}
	(stream.parameters.extradata.clone(), packets)
}

fn samples_of(packet: &[u8]) -> usize {
	rmp::codecs::opus::packet_samples(packet).unwrap() as usize
}

/// Decodes `name` with [`OpusDecoderState`]'s explicit recovery calls: each
/// run of lost packets is concealed (with FEC from the next packet when
/// `fec`), as an application that knows what it lost would.
fn decode_explicit(name: &str, fec: bool, drop: impl Fn(usize, usize) -> bool) -> Vec<f32> {
	let (extradata, packets) = packets(name);
	let mut dec = OpusDecoderState::from_extradata(&extradata, 2);
	let count = packets.len();
	let mut out = Vec::new();
	let mut gap = 0;
	for (i, packet) in packets.iter().enumerate() {
		if drop(i, count) {
			gap += samples_of(&packet.data);
			continue;
		}
		if gap > 0 {
			let recovered = if fec { dec.decode_fec(&packet.data, gap) } else { dec.conceal(gap) };
			out.extend(recovered.unwrap());
			gap = 0;
		}
		out.extend(dec.decode_packet(&packet.data).unwrap());
	}
	out
}

/// Decodes `name` through the `Decoder` trait with the lost packets left
/// out, as a demuxer that skipped damaged data delivers them: the decoder
/// finds the gaps from the timestamps. Returns the samples, the frames'
/// `(pts, samples per channel)` and the final decoder state.
fn decode_with_gaps(name: &str, fec: bool) -> (Vec<f32>, Vec<(u64, usize)>, OpusDecoderState) {
	let demuxer = MkvDemuxer::new(memory(fixture(&format!("opus/{name}.webm")))).unwrap();
	let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	if let State::Audio(audio) = &mut state
		&& let CodecPrivate::Opus(dec) = &mut audio.private
	{
		dec.use_fec = fec;
	}
	let (_, packets) = packets(name);
	let count = packets.len();
	let mut out = Vec::new();
	let mut frames = Vec::new();
	for (i, packet) in packets.into_iter().enumerate() {
		if lost(i, count) {
			continue;
		}
		for frame in decoder.decode(&mut state, packet).unwrap().value {
			let Frame::Audio(a) = frame else { panic!("audio expected") };
			let samples = a.data.len() / 4 / usize::from(a.channels.count());
			frames.push((a.pts.unwrap().ticks(), samples));
			out.extend(a.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
		}
	}
	let State::Audio(audio) = state else { unreachable!() };
	let CodecPrivate::Opus(dec) = audio.private else { unreachable!() };
	(out, frames, *dec)
}

/// libopus' `FLOAT2INT16`.
fn to_s16(x: f32) -> i16 {
	(x * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Largest difference in 16-bit steps and the SNR in dB of `pcm` against
/// `reference`.
fn distance(pcm: &[f32], reference: &[i16]) -> (i32, f64) {
	assert_eq!(pcm.len(), reference.len(), "length");
	let mut signal = 0f64;
	let mut noise = 0f64;
	let mut max_diff = 0;
	for (&a, &b) in pcm.iter().zip(reference) {
		let d = i32::from(to_s16(a)) - i32::from(b);
		max_diff = max_diff.max(d.abs());
		signal += f64::from(b) * f64::from(b);
		noise += f64::from(d) * f64::from(d);
	}
	(max_diff, if noise == 0.0 { f64::INFINITY } else { 10.0 * (signal / noise).log10() })
}

fn reference(name: &str, mode: &str) -> Vec<i16> {
	fixture(&format!("opus/{name}.loss_{mode}.s16le"))
		.chunks_exact(2)
		.map(|b| i16::from_le_bytes([b[0], b[1]]))
		.collect()
}

fn against_libopus(name: &str, fec: bool) -> (i32, f64) {
	let pcm = decode_explicit(name, fec, lost);
	distance(&pcm, &reference(name, if fec { "fec" } else { "plc" }))
}

#[test]
fn silk_concealment_and_comfort_noise_are_bit_exact() {
	// SILK is fixed point: PLC, CNG and the fade-in after a loss match
	// libopus exactly (wideband voiced speech; 40 ms stereo packets).
	assert_eq!(against_libopus("plc_silk_fec", false).0, 0);
	assert_eq!(against_libopus("silk_wb_st_40ms", false).0, 0);
}

#[test]
fn silk_fec_is_bit_exact_and_recovers_the_loss() {
	assert_eq!(against_libopus("plc_silk_fec", true).0, 0);
	// The LBRR copies bring the output closer to a loss-free decode than
	// concealment alone.
	let clean = decode_explicit("plc_silk_fec", false, |_, _| false);
	let clean: Vec<i16> = clean.iter().map(|&x| to_s16(x)).collect();
	let (_, plc) = distance(&decode_explicit("plc_silk_fec", false, lost), &clean);
	let (_, fec) = distance(&decode_explicit("plc_silk_fec", true, lost), &clean);
	assert!(fec > plc + 1.0, "FEC {fec:.1} dB vs PLC {plc:.1} dB");
}

#[test]
fn hybrid_concealment_and_fec_match_libopus() {
	// SILK below 8 kHz (exact) plus CELT noise concealment above (float).
	for (name, fec) in
		[("hybrid_st", false), ("plc_hybrid_fec_st", false), ("plc_hybrid_fec_st", true)]
	{
		let (max_diff, snr) = against_libopus(name, fec);
		assert!(max_diff <= 1, "{name} (fec {fec}): {max_diff} steps, {snr:.1} dB");
	}
}

#[test]
fn celt_pitch_concealment_matches_libopus() {
	// The pitch-based PLC runs long LPC filters over the extrapolated
	// excitation, so float rounding grows: libopus' own SIMD and scalar
	// builds differ by 13 steps (74 dB) on this stream.
	let (max_diff, snr) = against_libopus("plc_celt", false);
	assert!(max_diff <= 16 && snr > 70.0, "{max_diff} steps, {snr:.1} dB");
	// Multistream 5.1 and mode switches with losses.
	for name in ["celt_51", "mode_switch"] {
		let (max_diff, snr) = against_libopus(name, false);
		assert!(max_diff <= 2 && snr > 90.0, "{name}: {max_diff} steps, {snr:.1} dB");
	}
}

#[test]
fn timestamp_gaps_are_concealed_like_explicit_losses() {
	for (name, fec) in [("plc_celt", false), ("plc_silk_fec", true), ("plc_hybrid_fec_st", true)] {
		let (pcm, frames, dec) = decode_with_gaps(name, fec);
		assert_eq!(pcm, decode_explicit(name, fec, lost), "{name}");
		// The output timeline has no holes (millisecond timestamps aside).
		for pair in frames.windows(2) {
			let (pts, n) = pair[0];
			let drift = pair[1].0 as i64 - (pts + n as u64) as i64;
			assert!(drift.abs() < 60, "{name}: frame at {} drifts {drift}", pair[1].0);
		}
		assert_eq!(dec.concealed_frames, 0);
		let (_, all) = packets(name);
		let lost_samples: usize = all
			.iter()
			.enumerate()
			.filter(|(i, _)| lost(*i, all.len()))
			.map(|(_, p)| samples_of(&p.data))
			.sum();
		assert_eq!(dec.concealed_samples, lost_samples as u64, "{name}");
	}
}

fn rms(x: &[f32]) -> f32 {
	(x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

#[test]
fn damaged_packets_are_concealed_instead_of_silenced() {
	for name in ["plc_celt", "plc_silk_fec", "plc_hybrid_fec_st"] {
		let mut demuxer = MkvDemuxer::new(memory(fixture(&format!("opus/{name}.webm")))).unwrap();
		let stream = demuxer.streams().by_kind(StreamKind::Audio)[0].clone();
		let decoder = Resolver::new().decoder_for(&stream).unwrap();
		let mut state = decoder.create_state(&stream.parameters);
		let mut damaged = Vec::new();
		let mut i = 0;
		while let Some(mut packet) = demuxer.read().unwrap() {
			if packet.stream_id != stream.id {
				continue;
			}
			// Packet 12 is inside a voiced segment: an invalid TOC code 3 with
			// zero frames, then an empty packet.
			if i == 12 {
				packet.data = vec![packet.data[0] | 3, 0];
			} else if i == 13 {
				packet.data.clear();
			}
			for frame in decoder.decode(&mut state, packet).unwrap().value {
				let Frame::Audio(a) = frame else { panic!("audio expected") };
				let pcm: Vec<f32> =
					a.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
				assert!(pcm.iter().all(|s| s.is_finite() && s.abs() <= 2.0), "{name}");
				if i == 12 || i == 13 {
					assert_eq!(pcm.len(), 960 * usize::from(a.channels.count()), "{name}: duration");
					damaged.push(rms(&pcm));
				}
			}
			i += 1;
		}
		assert!(damaged.iter().all(|&r| r > 1e-3), "{name}: concealment is silent: {damaged:?}");
		let State::Audio(audio) = &state else { unreachable!() };
		let CodecPrivate::Opus(dec) = &audio.private else { unreachable!() };
		assert_eq!(dec.concealed_frames, 2, "{name}");
	}
}

#[test]
fn long_losses_fade_into_comfort_noise() {
	for name in ["plc_celt", "plc_silk_fec", "plc_hybrid_fec_st"] {
		let (extradata, packets) = packets(name);
		let mut dec = OpusDecoderState::from_extradata(&extradata, 2);
		let mut last = Vec::new();
		// Up to voiced audio after the first pause (from which SILK's comfort
		// noise learns the background; without one it fades to silence, as
		// libopus does).
		for packet in &packets[..60] {
			last = dec.decode_packet(&packet.data).unwrap();
		}
		// One second lost.
		let concealed = dec.conceal(48_000).unwrap();
		let ch = dec.channels();
		assert_eq!(concealed.len(), 48_000 * ch);
		assert!(concealed.iter().all(|s| s.is_finite() && s.abs() <= 2.0), "{name}");
		let start = rms(&concealed[..960 * ch]);
		let end = rms(&concealed[concealed.len() - 4800 * ch..]);
		// Continues the signal at first, then decays, but never to silence.
		assert!(start > 0.2 * rms(&last), "{name}: start {start} vs {}", rms(&last));
		assert!(end < 0.5 * start, "{name}: no fade ({start} → {end})");
		assert!(end > 1e-4, "{name}: concealment went silent");
		// Decoding resumes normally afterwards.
		let resumed = dec.decode_packet(&packets[60].data).unwrap();
		assert!(resumed.iter().all(|s| s.is_finite() && s.abs() <= 2.0), "{name}");
	}
}

#[test]
fn concealment_edge_cases() {
	let (extradata, packets) = packets("plc_celt");
	let mut dec = OpusDecoderState::from_extradata(&extradata, 2);
	// Nothing decoded yet: silence, as libopus.
	assert!(dec.conceal(960).unwrap().iter().all(|&s| s == 0.0));
	// Concealment comes in steps of 2.5 ms.
	assert!(dec.conceal(100).is_err());
	assert!(dec.decode_fec(&packets[0].data, 1000).is_err());
	// Gaps longer than `max_gap` are discontinuities: nothing is invented.
	dec.decode_packet(&packets[0].data).unwrap();
	let (out, start) = dec.decode_with_recovery(&packets[1].data, Some(960), None);
	assert_eq!((out.len(), start), (960, Some(960)));
	let (out, start) = dec.decode_with_recovery(&packets[2].data, Some(960 * 1000), None);
	assert_eq!((out.len(), start), (960, Some(960 * 1000)));
	// With gap concealment off, a gap is passed through.
	dec.conceal_gaps = false;
	let (out, _) = dec.decode_with_recovery(&packets[3].data, Some(960 * 1003), None);
	assert_eq!(out.len(), 960);
}
