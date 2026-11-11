//! Vorbis audio packet decoding (Vorbis I §4.3; FFmpeg's
//! `vorbis_parse_audio_packet`).
//!
//! Each packet is one MDCT block: per channel a floor (spectral envelope)
//! and a residue (detail), decoupled from magnitude/angle pairs, multiplied
//! together, inverse-transformed, windowed and overlap-added with the
//! previous block. A packet returns the samples between the centres of the
//! previous and current blocks: `previous / 4 + current / 4`.
//!
//! The first packet after the headers (or a reset) only primes the
//! overlap: like FFmpeg, its samples (silence ramping in) are not
//! returned, which is what Ogg granule positions and FFmpeg's negative
//! first timestamp describe. A packet whose duration (in samples) is
//! shorter than its block marks end trimming and is cut to it.
//!
//! Output is planar `F32` in FFmpeg's channel order.

use std::sync::Arc;

use super::bits::Bits;
use super::floor::FloorData;
use super::setup::{Identification, Setup};
use super::{CHANNEL_ORDER, ilog};
use crate::codecs::dsp::fft::Complex;
use crate::codecs::dsp::mdct::Imdct;
use crate::container::xiph;
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State, Time, TimeBase,
};
use crate::message::{Message, Result};

/// Header-derived tables shared between clones of the decoder state.
#[derive(Debug)]
struct Tables {
	id: Identification,
	setup: Setup,
	imdct: [Imdct; 2],
	/// Rising window slope of the short and long window (`size / 2`
	/// values, §4.3.1).
	slope: [Vec<f32>; 2],
}

#[derive(Debug, Clone)]
pub struct VorbisDecoderState {
	tables: Arc<Tables>,
	/// Right half of the previous block, per coded channel; empty before
	/// the first packet.
	saved: Vec<Vec<f32>>,
	/// Where the next output starts, in samples, when known.
	next_pts: Option<u64>,
}

/// §4.3.1 window slope: `sin(π/2 · sin²((i + ½) / half · π/2))`.
fn window_slope(size: usize) -> Vec<f32> {
	let half = size / 2;
	(0..half)
		.map(|i| {
			let x = (i as f64 + 0.5) / half as f64 * std::f64::consts::FRAC_PI_2;
			(std::f64::consts::FRAC_PI_2 * x.sin().powi(2)).sin() as f32
		})
		.collect()
}

impl VorbisDecoderState {
	/// From the three header packets.
	pub fn new(identification: &[u8], setup: &[u8]) -> Result<Self> {
		let id = Identification::parse(identification)
			.ok_or(Message::Codec("vorbis: invalid identification header"))?;
		let setup = Setup::parse(setup, &id).ok_or(Message::Codec("vorbis: invalid setup header"))?;
		let imdct = id.blocksize.map(|n| Imdct::new(n, 1.0));
		let slope = id.blocksize.map(window_slope);
		Ok(Self {
			tables: Arc::new(Tables { id, setup, imdct, slope }),
			saved: Vec::new(),
			next_pts: None,
		})
	}

	/// From FFmpeg-style extradata (Xiph-laced or length-prefixed headers).
	pub fn from_extradata(extradata: &[u8]) -> Result<Self> {
		let headers = xiph::vorbis_headers(extradata)
			.ok_or(Message::Codec("vorbis: extradata holds no headers"))?;
		Self::new(&headers[0], &headers[2])
	}

	pub fn channels(&self) -> usize {
		usize::from(self.tables.id.channels)
	}

	pub fn sample_rate(&self) -> u32 {
		self.tables.id.sample_rate
	}

	/// Forgets the overlap (after a seek): the next packet primes again.
	pub fn reset(&mut self) {
		self.saved.clear();
		self.next_pts = None;
	}

	/// Decodes one audio packet into per-channel samples (coded channel
	/// order). Returns no samples for the priming packet and for header or
	/// empty packets.
	pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Vec<Vec<f32>>> {
		let tables = Arc::clone(&self.tables);
		let channels = self.channels();
		if packet.is_empty() || packet[0] & 1 == 1 {
			// Empty packets carry no audio; header packets may repeat.
			return Ok(Vec::new());
		}
		let mut b = Bits::new(packet);
		let invalid = || Message::Codec("vorbis: invalid audio packet");
		b.read(1).ok_or_else(invalid)?;
		let setup = &tables.setup;
		let mode_bits = ilog(setup.modes.len() as u32 - 1);
		let mode_index = b.read(mode_bits).ok_or_else(invalid)? as usize;
		let mode = *setup.modes.get(mode_index).ok_or_else(invalid)?;
		let long = mode.long;
		let n = tables.id.blocksize[usize::from(long)];
		let (prev_long, next_long) = if long {
			let p = b.flag().ok_or_else(invalid)?;
			let q = b.flag().ok_or_else(invalid)?;
			(p, q)
		} else {
			(false, false)
		};
		let mapping = &setup.mappings[mode.mapping];
		let half = n / 2;

		// Floors (§4.3.2); an unused floor (or the packet ending) leaves the
		// channel silent.
		let floors: Vec<Option<FloorData>> = (0..channels)
			.map(|ch| {
				let floor = &setup.floors[mapping.submap_floor[mapping.mux[ch]]];
				floor.decode(&mut b, &setup.codebooks)
			})
			.collect();
		let mut no_residue: Vec<bool> = floors.iter().map(Option::is_none).collect();
		for &(m, a) in mapping.coupling.iter().rev() {
			if !(no_residue[m] && no_residue[a]) {
				no_residue[m] = false;
				no_residue[a] = false;
			}
		}

		// Residues, one submap at a time (§4.3.4).
		let mut spectrum: Vec<Vec<f32>> = vec![vec![0f32; half]; channels];
		for (submap, &residue) in mapping.submap_residue.iter().enumerate() {
			let members: Vec<usize> = (0..channels).filter(|&ch| mapping.mux[ch] == submap).collect();
			if members.is_empty() {
				continue;
			}
			let mut vectors: Vec<Vec<f32>> =
				members.iter().map(|&ch| std::mem::take(&mut spectrum[ch])).collect();
			let decode: Vec<bool> = members.iter().map(|&ch| !no_residue[ch]).collect();
			setup.residues[residue].decode(&mut b, &setup.codebooks, &mut vectors, &decode, half);
			for (&ch, v) in members.iter().zip(vectors) {
				spectrum[ch] = v;
			}
		}

		// Inverse coupling (§4.3.5), last step first.
		for &(m, a) in mapping.coupling.iter().rev() {
			let (mag, ang) = pair_mut(&mut spectrum, m, a);
			for (m, a) in mag.iter_mut().zip(ang.iter_mut()) {
				let (old_m, old_a) = (*m, *a);
				let (new_m, new_a) = if old_m > 0.0 {
					if old_a > 0.0 { (old_m, old_m - old_a) } else { (old_m + old_a, old_m) }
				} else if old_a > 0.0 {
					(old_m, old_m + old_a)
				} else {
					(old_m - old_a, old_m)
				};
				*m = new_m;
				*a = new_a;
			}
		}

		// Floor × residue, inverse MDCT, windowing (§4.3.6 – §4.3.8).
		let imdct = &tables.imdct[usize::from(long)];
		let mut scratch = vec![0f32; half];
		let mut buf = vec![Complex::default(); n / 4];
		let left = if long && prev_long { 1 } else { 0 };
		let right = if long && next_long { 1 } else { 0 };
		let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(channels);
		for ch in 0..channels {
			let mut block = vec![0f32; n];
			if let Some(data) = &floors[ch] {
				let floor = &setup.floors[mapping.submap_floor[mapping.mux[ch]]];
				floor.apply(data, long, &mut spectrum[ch]);
				imdct.inverse(&spectrum[ch], &mut block, &mut scratch, &mut buf);
				apply_window(&mut block, &tables.slope[left], &tables.slope[right]);
			}
			blocks.push(block);
		}

		// Overlap-add with the previous block's right half.
		let out = if self.saved.len() == channels {
			let prev_half = self.saved[0].len();
			let len = prev_half / 2 + n / 4;
			// Offset of the current block in the output: the overlap
			// centres (previous 3/4, current 1/4) coincide.
			let shift = (prev_half / 2) as isize - (n / 4) as isize;
			let mut out = Vec::with_capacity(channels);
			for (saved, block) in self.saved.iter().zip(&blocks) {
				let samples: Vec<f32> = (0..len)
					.map(|k| {
						let prev = saved.get(k).copied().unwrap_or(0.0);
						let at = k as isize - shift;
						let cur = if (0..half as isize).contains(&at) { block[at as usize] } else { 0.0 };
						prev + cur
					})
					.collect();
				out.push(samples);
			}
			out
		} else {
			Vec::new()
		};
		self.saved = blocks.into_iter().map(|mut b| b.split_off(half)).collect();
		Ok(out)
	}
}

/// Windows a block of `n` samples: zero, rising slope centred on `n / 4`,
/// flat, falling slope centred on `3n / 4`, zero.
fn apply_window(block: &mut [f32], left: &[f32], right: &[f32]) {
	let n = block.len();
	let (lh, rh) = (left.len(), right.len());
	let left_start = n / 4 - lh / 2;
	block[..left_start].fill(0.0);
	for (x, &w) in block[left_start..left_start + lh].iter_mut().zip(left) {
		*x *= w;
	}
	let right_start = 3 * n / 4 - rh / 2;
	for (x, &w) in block[right_start..right_start + rh].iter_mut().zip(right.iter().rev()) {
		*x *= w;
	}
	block[right_start + rh..].fill(0.0);
}

/// Two distinct elements of `v`, mutably.
fn pair_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
	debug_assert_ne!(a, b);
	if a < b {
		let (lo, hi) = v.split_at_mut(b);
		(&mut lo[a], &mut hi[0])
	} else {
		let (lo, hi) = v.split_at_mut(a);
		(&mut hi[0], &mut lo[b])
	}
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VorbisDecoder;

impl VorbisDecoder {
	pub fn new() -> Self {
		Self
	}
}

fn channel_layout(channels: usize) -> Channels {
	Channels::from_count(channels.min(255) as u8)
}

impl core::Decoder for VorbisDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec = VorbisDecoderState::from_extradata(&parameters.extradata).ok();
		let channels =
			dec.as_ref().map_or(usize::from(parameters.channels.unwrap_or(2)), |d| d.channels());
		let rate = dec.as_ref().map_or(parameters.sample_rate.unwrap_or(44_100), |d| d.sample_rate());
		State::Audio(AudioState {
			sample_rate: rate,
			channels: channel_layout(channels),
			format: SampleFormat::F32,
			layout: SampleLayout::Planar,
			frame_size: 0,
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: dec.map_or(CodecPrivate::None, |d| CodecPrivate::Vorbis(Box::new(d))),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("vorbis decoder requires an audio state"));
		};
		if !matches!(audio.private, CodecPrivate::Vorbis(_)) {
			let dec = VorbisDecoderState::from_extradata(&audio.codec_state.extradata)?;
			audio.private = CodecPrivate::Vorbis(Box::new(dec));
		}
		let CodecPrivate::Vorbis(dec) = &mut audio.private else {
			return Err(Message::Codec("vorbis decoder state missing"));
		};
		let rate = dec.sample_rate();
		let base = TimeBase::new(1, rate);
		let mut samples = dec.decode_packet(&packet.data)?;
		let mut frames = Vec::new();
		if let Some(len) = samples.first().map(Vec::len).filter(|&l| l > 0) {
			// End trimming: a duration in samples shorter than the block.
			let keep =
				packet.duration.filter(|d| d.base() == base).map_or(len, |d| (d.ticks() as usize).min(len));
			let pts = packet.pts.and_then(|t| t.rescale(base)).map(|t| t.ticks()).or(dec.next_pts);
			let channels = samples.len();
			let mut data = Vec::with_capacity(keep * channels * 4);
			let order: Vec<usize> =
				if channels <= 8 { CHANNEL_ORDER[channels - 1].to_vec() } else { (0..channels).collect() };
			for coded in order {
				let plane = &mut samples[coded];
				plane.truncate(keep);
				data.extend(plane.iter().flat_map(|v| v.to_le_bytes()));
			}
			let mut frame = AudioFrame::new(
				data,
				rate,
				channel_layout(channels),
				SampleFormat::F32,
				SampleLayout::Planar,
			);
			frame.pts = pts.map(|t| Time::new(t, base));
			dec.next_pts = pts.map(|t| t + keep as u64);
			audio.samples_produced += keep as u64;
			frames.push(Frame::Audio(frame));
		}
		audio.samples_consumed += packet.data.len() as u64;
		Ok(DecodeOut { value: frames, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		Ok(DecodeOut { value: Vec::new(), state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn window_is_power_complementary() {
		// Princen-Bradley: w[i]² + w[half - 1 - i]² = 1.
		for size in [64usize, 256, 2048] {
			let w = window_slope(size);
			for i in 0..w.len() {
				let s = w[i] * w[i] + w[w.len() - 1 - i] * w[w.len() - 1 - i];
				assert!((s - 1.0).abs() < 1e-6, "{size} {i}: {s}");
			}
		}
	}

	#[test]
	fn window_shapes_short_inside_long() {
		let mut block = vec![1.0f32; 16];
		let short = vec![0.5f32; 2];
		let long = vec![0.25f32; 8];
		apply_window(&mut block, &short, &long);
		// Left: zero up to 16/4 - 1 = 3, slope 3..5, flat to the right slope 8..16.
		assert_eq!(&block[..3], &[0.0; 3]);
		assert_eq!(&block[3..5], &[0.5; 2]);
		assert_eq!(&block[5..8], &[1.0; 3]);
		assert_eq!(&block[8..16], &[0.25; 8]);
	}
}
