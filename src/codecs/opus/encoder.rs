//! Opus encoder, CELT mode (RFC 6716 §5.3; libopus 1.4 `celt/celt_encoder.c`,
//! `quant_bands.c`, `bands.c`, `vq.c`, `cwrs.c`, `rate.c`, float build).
//!
//! Every packet is a single fullband CELT frame (TOC configs 28-31: 2.5, 5,
//! 10 or 20 ms) at a constant size from the bit rate. One frame:
//!
//! 1. pre-emphasis, transient analysis (`transient_analysis`: short blocks
//!    for attacks), forward MDCT ([`Mdct::forward`]);
//! 2. band energies, normalisation, coarse energy (Laplace-coded
//!    prediction residual, intra on the first frame), spreading decision
//!    (`spreading_decision`), dynamic allocation boosts for bands above
//!    their neighbours (`dynalloc_analysis`), allocation trim from the
//!    spectral tilt and stereo correlation (`alloc_trim_analysis`);
//! 3. the bit allocation, fine energy, PVQ shapes of every band with band
//!    splitting, mid/side or intensity stereo and folding, anti-collapse,
//!    final energy bits.
//!
//! Steps 2-3 run the decoder's own allocation and band code
//! ([`super::celt::compute_allocation`], [`super::celt::quant_all_bands`])
//! through the [`Coder`] trait, which is how libopus shares `quant_*`
//! between its encoder and decoder: the encoder writes the value it chose
//! where the decoder reads one, and reconstructs what the decoder will.
//! So the encoder's state stays in step with any decoder's, which the
//! tests check with Opus's own conformance measure: the final range-coder
//! state after each packet equals the decoder's.
//!
//! Not implemented (libopus tunings, not format features): SILK and
//! hybrid modes, the pitch pre-filter, TF analysis, the tonality and
//! masking analyses, VBR, dual stereo, two-pass intra energy.
//!
//! The encoder delays the signal by the MDCT overlap: 120 samples, the
//! `pre_skip` of its `OpusHead`.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::OpusHead;
use super::celt::{
	AllocChoices, BETA_COEF, BETA_INTRA, MAX_FINE_BITS, MAX_LM, NB_EBANDS, OVERLAP, PRED_COEF,
	PREEMPH, SHORT_MDCT, SMALL_ENERGY_ICDF, SPREAD_ICDF, SPREAD_NORMAL, TF_SELECT_TABLE, TRIM_ICDF,
	compute_allocation, init_caps, quant_all_bands,
};
use super::mdct::Mdct;
use super::range::{BITRES, RangeEncoder};
use super::tables::{E_MEANS, E_PROB_MODEL, EBANDS, WINDOW120};
use crate::codecs::pcm::convert::for_each_sample;
use crate::core::{
	self, AudioState, Channels, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters,
	SampleFormat, SampleLayout, State, StreamId, Time, TimeBase,
};
use crate::message::{Message, Result};

/// Encoder delay in samples: the MDCT overlap (`OpusHead.pre_skip`).
pub const PRE_SKIP: u16 = OVERLAP as u16;
/// Default frame: 20 ms.
pub const DEFAULT_FRAME: usize = 960;
/// Largest Opus frame payload.
const MAX_BYTES: usize = 1275;

/// Default bit rate: 64 kb/s per channel (libopus' fullband default is
/// close to this for music).
pub fn default_bitrate(channels: u8) -> u32 {
	64_000 * u32::from(channels.clamp(1, 2))
}

/// Intensity-stereo start band by bit rate in kb/s (`intensity_thresholds`).
const INTENSITY_THRESHOLDS: [i32; 21] =
	[1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134];
const INTENSITY_HYSTERESIS: [i32; 21] =
	[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8];

fn hysteresis_decision(val: i32, thresholds: &[i32], hysteresis: &[i32], prev: usize) -> usize {
	let n = thresholds.len();
	let mut i = thresholds.iter().position(|&t| val < t).unwrap_or(n);
	if i > prev && prev < n && val < thresholds[prev] + hysteresis[prev] {
		i = prev;
	}
	if i < prev && prev > 0 && val > thresholds[prev - 1] - hysteresis[prev - 1] {
		i = prev;
	}
	i
}

/// `transient_analysis` (float, without weak transients): whether the
/// frame has an attack that needs short blocks. `inp` holds `len` samples
/// per channel.
fn transient_analysis(inp: &[f32], len: usize, c: usize) -> bool {
	const INV_TABLE: [u8; 128] = [
		255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
		16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
		8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
		5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
	];
	// Forward masking: 6.7 dB/ms.
	let forward_decay = 0.0625f32;
	let len2 = len / 2;
	let mut mask_metric = 0i32;
	let mut tmp = vec![0f32; len];
	for ch in 0..c {
		let x = &inp[ch * len..(ch + 1) * len];
		// High-pass filter: (1 - 2z^-1 + z^-2) / (1 - z^-1 + .5z^-2).
		let (mut mem0, mut mem1) = (0f32, 0f32);
		for i in 0..len {
			let y = mem0 + x[i];
			mem0 = mem1 + y - 2.0 * x[i];
			mem1 = x[i] - 0.5 * y;
			tmp[i] = y;
		}
		tmp[..12].fill(0.0);
		let mut mean = 0f32;
		let mut mem0 = 0f32;
		// Forward pass: post-echo threshold (pairs of samples).
		for i in 0..len2 {
			let x2 = tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1];
			mean += x2;
			tmp[i] = mem0 + forward_decay * (x2 - mem0);
			mem0 = tmp[i];
		}
		// Backward pass: pre-echo threshold (13.9 dB/ms).
		let mut mem0 = 0f32;
		let mut max_e = 0f32;
		for i in (0..len2).rev() {
			tmp[i] = mem0 + 0.125 * (tmp[i] - mem0);
			mem0 = tmp[i];
			max_e = max_e.max(mem0);
		}
		// Frame energy (geometric mean of the energy and half the
		// maximum) over the harmonic mean of the envelope.
		let mean = (mean * max_e * 0.5 * len2 as f32).sqrt();
		let norm = len2 as f32 / (1e-15 + mean);
		let mut unmask = 0i32;
		let mut i = 12;
		while i + 5 < len2 {
			let id = (64.0 * norm * (tmp[i] + 1e-15)).floor().clamp(0.0, 127.0) as usize;
			unmask += i32::from(INV_TABLE[id]);
			i += 4;
		}
		let unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
		mask_metric = mask_metric.max(unmask);
	}
	mask_metric > 200
}

const SPREAD_NONE: i32 = 0;
const SPREAD_LIGHT: i32 = 1;
const SPREAD_AGGRESSIVE: i32 = 3;

/// `spreading_decision`: how much the PVQ codewords are rotated, from how
/// peaky the normalised bands are (tonal signals want no spreading).
/// `average` carries the recursive average between frames.
fn spreading_decision(
	x: &[f32],
	average: &mut i32,
	last: i32,
	end: usize,
	c: usize,
	m: usize,
) -> i32 {
	let n0 = m * SHORT_MDCT;
	let width = |i: usize| m * (EBANDS[i + 1] - EBANDS[i]) as usize;
	if width(end - 1) <= 8 {
		return SPREAD_NONE;
	}
	let (mut sum, mut bands) = (0i32, 0i32);
	for ch in 0..c {
		for i in 0..end {
			let n = width(i);
			if n <= 8 {
				continue;
			}
			let band = &x[ch * n0 + m * EBANDS[i] as usize..][..n];
			let mut tcount = [0i32; 3];
			for &v in band {
				let x2n = v * v * n as f32;
				tcount[0] += i32::from(x2n < 0.25);
				tcount[1] += i32::from(x2n < 0.0625);
				tcount[2] += i32::from(x2n < 0.015625);
			}
			let n = n as i32;
			sum += i32::from(2 * tcount[2] >= n)
				+ i32::from(2 * tcount[1] >= n)
				+ i32::from(2 * tcount[0] >= n);
			bands += 1;
		}
	}
	if bands == 0 {
		return SPREAD_NORMAL;
	}
	let sum = (sum << 8) / bands;
	let sum = (sum + *average) >> 1;
	*average = sum;
	// Hysteresis around the last decision.
	let sum = (3 * sum + (((3 - last) << 7) + 64) + 2) >> 2;
	if sum < 80 {
		SPREAD_AGGRESSIVE
	} else if sum < 256 {
		SPREAD_NORMAL
	} else if sum < 384 {
		SPREAD_LIGHT
	} else {
		SPREAD_NONE
	}
}

/// `alloc_trim_analysis` (without the tonality analysis): tilts the
/// allocation towards low or high bands from the spectral tilt and, for
/// stereo, the inter-channel correlation.
fn alloc_trim_analysis(
	x: &[f32],
	band_log: &[f32; 2 * NB_EBANDS],
	end: usize,
	lm: usize,
	c: usize,
	intensity: usize,
	equiv_rate: i32,
) -> usize {
	let mut trim = 5.0f32;
	if equiv_rate < 64_000 {
		trim = 4.0;
	} else if equiv_rate < 80_000 {
		trim = 4.0 + (1.0 / 16.0) * ((equiv_rate - 64_000) >> 10) as f32;
	}
	if c == 2 {
		let n0 = SHORT_MDCT << lm;
		let corr = |i: usize| {
			let (lo, hi) = ((EBANDS[i] as usize) << lm, (EBANDS[i + 1] as usize) << lm);
			x[lo..hi].iter().zip(&x[n0 + lo..n0 + hi]).map(|(a, b)| a * b).sum::<f32>()
		};
		let mut sum = 0f32;
		for i in 0..8 {
			sum += corr(i);
		}
		let sum = (sum / 8.0).abs().min(1.0);
		let mut min_xc = sum;
		for i in 8..intensity.min(end) {
			min_xc = min_xc.min(corr(i).abs());
		}
		let _ = min_xc.min(1.0);
		let log_xc = (1.001 - sum * sum).log2();
		trim += (-4.0f32).max(0.75 * log_xc);
	}
	// Spectral tilt.
	let mut diff = 0f32;
	for ch in 0..c {
		for i in 0..end - 1 {
			diff += band_log[ch * NB_EBANDS + i] * (2 + 2 * i as i32 - end as i32) as f32;
		}
	}
	diff /= (c * (end - 1)) as f32;
	trim -= (-2.0f32).max(2.0f32.min((diff + 1.0) / 6.0));
	((0.5 + trim).floor() as i32).clamp(0, 10) as usize
}

fn median_of_5(x: &[f32]) -> f32 {
	let mut v = [x[0], x[1], x[2], x[3], x[4]];
	v.sort_by(|a, b| a.total_cmp(b));
	v[2]
}

fn median_of_3(x: &[f32]) -> f32 {
	let mut v = [x[0], x[1], x[2]];
	v.sort_by(|a, b| a.total_cmp(b));
	v[1]
}

/// `dynalloc_analysis` (energy follower part): extra bits for bands that
/// stand out above a smoothed envelope of their neighbours. Returns the
/// boost per band in allocation quanta.
fn dynalloc_analysis(
	band_log: &[f32; 2 * NB_EBANDS],
	old_band_e: &[f32; 2 * NB_EBANDS],
	end: usize,
	c: usize,
	lm: usize,
	transient: bool,
	effective_bytes: i32,
) -> [i32; NB_EBANDS] {
	use super::tables::LOGN400;
	let mut offsets = [0i32; NB_EBANDS];
	if effective_bytes < 30 + 5 * lm as i32 {
		return offsets;
	}
	let lsb_depth = 24.0f32;
	let noise_floor: Vec<f32> = (0..end)
		.map(|i| {
			0.0625 * f32::from(LOGN400[i]) + 0.5 + (9.0 - lsb_depth) - E_MEANS[i]
				+ 0.0062 * ((i + 5) * (i + 5)) as f32
		})
		.collect();
	let mut follower = [0f32; 2 * NB_EBANDS];
	for ch in 0..c {
		let mut e3: Vec<f32> = band_log[ch * NB_EBANDS..ch * NB_EBANDS + end].to_vec();
		if lm == 0 {
			// 2.5 ms: the first bands have one bin; use the previous energy too.
			for i in 0..8.min(end) {
				e3[i] = e3[i].max(old_band_e[ch * NB_EBANDS + i]);
			}
		}
		let f = &mut follower[ch * NB_EBANDS..ch * NB_EBANDS + end];
		f[0] = e3[0];
		let mut last = 0;
		for i in 1..end {
			if e3[i] > e3[i - 1] + 0.5 {
				last = i;
			}
			f[i] = (f[i - 1] + 1.5).min(e3[i]);
		}
		for i in (0..last).rev() {
			f[i] = f[i].min((f[i + 1] + 2.0).min(e3[i]));
		}
		// Median filter so dynalloc doesn't trigger on noise.
		let offset = 1.0;
		for i in 2..end - 2 {
			f[i] = f[i].max(median_of_5(&e3[i - 2..]) - offset);
		}
		let tmp = median_of_3(&e3[0..]) - offset;
		f[0] = f[0].max(tmp);
		f[1] = f[1].max(tmp);
		let tmp = median_of_3(&e3[end - 3..]) - offset;
		f[end - 2] = f[end - 2].max(tmp);
		f[end - 1] = f[end - 1].max(tmp);
		for i in 0..end {
			f[i] = f[i].max(noise_floor[i]);
		}
	}
	if c == 2 {
		for i in 0..end {
			// 24 dB of cross-talk.
			follower[NB_EBANDS + i] = follower[NB_EBANDS + i].max(follower[i] - 4.0);
			follower[i] = follower[i].max(follower[NB_EBANDS + i] - 4.0);
			follower[i] = 0.5
				* ((band_log[i] - follower[i]).max(0.0)
					+ (band_log[NB_EBANDS + i] - follower[NB_EBANDS + i]).max(0.0));
		}
	} else {
		for i in 0..end {
			follower[i] = (band_log[i] - follower[i]).max(0.0);
		}
	}
	for i in 0..end {
		// CBR: half the contribution outside transients.
		if !transient {
			follower[i] *= 0.5;
		}
		if i < 8 {
			follower[i] *= 2.0;
		}
		if i >= 12 {
			follower[i] *= 0.5;
		}
	}
	let mut tot_boost = 0i32;
	for i in 0..end {
		let f = follower[i].min(4.0);
		let width = (c as i32 * (EBANDS[i + 1] - EBANDS[i]) as i32) << lm;
		let (boost, boost_bits) = if width < 6 {
			let b = f as i32;
			(b, (b * width) << BITRES)
		} else if width > 48 {
			let b = (f * 8.0) as i32;
			(b, ((b * width) << BITRES) / 8)
		} else {
			let b = (f * width as f32 / 6.0) as i32;
			(b, (b * 6) << BITRES)
		};
		// CBR: at most 2/3 of the bits go to boosts.
		if (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes / 3 {
			let cap = (2 * effective_bytes / 3) << BITRES << 3;
			offsets[i] = cap - tot_boost;
			break;
		}
		offsets[i] = boost;
		tot_boost += boost_bits;
	}
	offsets
}

/// CELT encoder state (`OpusCustomEncoder`).
#[derive(Debug, Clone)]
pub struct CeltEncoder {
	pub channels: usize,
	/// Pre-emphasised history: the last `OVERLAP` samples per channel.
	in_mem: Vec<f32>,
	preemph_mem: [f32; 2],
	/// Quantised band energies, as the decoder holds them.
	old_band_e: [f32; 2 * NB_EBANDS],
	intensity: usize,
	prev_coded: usize,
	consec_transient: u32,
	spread_decision: i32,
	tonal_average: i32,
	/// Final range of the previous frame (the decoder's folding seed).
	rng: u32,
	frames: u64,
	/// Frames coded with short blocks (statistics).
	pub transients: u64,
}

impl CeltEncoder {
	pub fn new(channels: usize) -> Self {
		Self {
			channels,
			in_mem: vec![0.0; channels * OVERLAP],
			preemph_mem: [0.0; 2],
			old_band_e: [0.0; 2 * NB_EBANDS],
			intensity: 0,
			prev_coded: 0,
			consec_transient: 0,
			spread_decision: SPREAD_NORMAL,
			tonal_average: 256,
			rng: 0,
			frames: 0,
			transients: 0,
		}
	}

	/// Final range of the last frame (`OPUS_GET_FINAL_RANGE`).
	pub fn final_range(&self) -> u32 {
		self.rng
	}

	/// Codes one frame of `120 << lm` interleaved samples (unit scale) into
	/// exactly `bytes` bytes.
	#[allow(clippy::too_many_lines)]
	pub fn encode(&mut self, pcm: &[f32], lm: usize, bytes: usize) -> Vec<u8> {
		let c = self.channels;
		let n = SHORT_MDCT << lm;
		let m = 1usize << lm;
		let ov = OVERLAP;
		let (start, end) = (0usize, NB_EBANDS);
		let stride = n + ov;

		// Pre-emphasis into [history | new samples] per channel.
		let mut inp = vec![0f32; c * stride];
		for ch in 0..c {
			let row = &mut inp[ch * stride..(ch + 1) * stride];
			row[..ov].copy_from_slice(&self.in_mem[ch * ov..(ch + 1) * ov]);
			let mut mem = self.preemph_mem[ch];
			for j in 0..n {
				let x = pcm.get(j * c + ch).copied().unwrap_or(0.0) * 32768.0;
				row[ov + j] = x - mem;
				mem = PREEMPH * x;
			}
			self.preemph_mem[ch] = mem;
		}

		let mut enc = RangeEncoder::new(bytes);
		let total_bits = (bytes * 8) as i32;
		if enc.tell() == 1 {
			enc.bit_logp(false, 15); // silence
		}
		if start == 0 && enc.tell() + 16 <= total_bits {
			enc.bit_logp(false, 1); // no pitch post-filter
		}
		let mut transient = lm > 0 && transient_analysis(&inp, stride, c);
		if lm > 0 && enc.tell() + 3 <= total_bits {
			enc.bit_logp(transient, 3);
		} else {
			transient = false;
		}

		// MDCT: one long block, or 2^lm interleaved short ones.
		let (blocks, nb, shift) = if transient { (m, SHORT_MDCT, MAX_LM) } else { (1, n, MAX_LM - lm) };
		let mdct = Mdct::celt();
		let mut freq = vec![0f32; c * n];
		for ch in 0..c {
			for b in 0..blocks {
				let input = &inp[ch * stride + b * nb..];
				mdct.forward(input, &mut freq[ch * n + b..], &WINDOW120, ov, shift, blocks);
			}
		}

		// Band energies (amplitudes and log2 relative to the means) and
		// the normalised spectrum.
		let mut band_e = [0f32; 2 * NB_EBANDS];
		let mut band_log = [0f32; 2 * NB_EBANDS];
		let mut x = vec![0f32; c * n];
		for ch in 0..c {
			for i in start..end {
				let (lo, hi) = (m * EBANDS[i] as usize, m * EBANDS[i + 1] as usize);
				let f = &freq[ch * n + lo..ch * n + hi];
				let e = (1e-27 + f.iter().map(|v| v * v).sum::<f32>()).sqrt();
				band_e[ch * NB_EBANDS + i] = e;
				band_log[ch * NB_EBANDS + i] = e.log2() - E_MEANS[i];
				let g = 1.0 / (1e-27 + e);
				for j in lo..hi {
					x[ch * n + j] = freq[ch * n + j] * g;
				}
			}
		}

		let intra = self.frames == 0;
		let intra = if enc.tell() + 3 <= total_bits {
			enc.bit_logp(intra, 3);
			intra
		} else {
			false
		};
		let mut error = [0f32; 2 * NB_EBANDS];
		self.quant_coarse_energy(&band_log, start, end, intra, &mut enc, lm, bytes, &mut error);

		// TF: no changes (tf_select 0).
		let mut tf_res = [0i32; NB_EBANDS];
		{
			let mut budget = (bytes * 8) as u32;
			let mut tell = enc.tell() as u32;
			let mut logp = if transient { 2 } else { 4 };
			let tf_select_rsv = lm > 0 && tell + logp < budget;
			budget -= u32::from(tf_select_rsv);
			for _ in start..end {
				if tell + logp <= budget {
					enc.bit_logp(false, logp);
					tell = enc.tell() as u32;
				}
				logp = if transient { 4 } else { 5 };
			}
			let t = usize::from(transient) * 4;
			if tf_select_rsv && TF_SELECT_TABLE[lm][t] != TF_SELECT_TABLE[lm][t + 2] {
				enc.bit_logp(false, 1);
			}
			for slot in tf_res.iter_mut().take(end).skip(start) {
				*slot = i32::from(TF_SELECT_TABLE[lm][t]);
			}
		}

		let mut spread = SPREAD_NORMAL;
		if enc.tell() + 4 <= total_bits {
			if !transient && bytes >= 10 * c {
				self.spread_decision =
					spreading_decision(&x, &mut self.tonal_average, self.spread_decision, end, c, m);
			} else {
				self.spread_decision = SPREAD_NORMAL;
			}
			spread = self.spread_decision;
			enc.icdf(spread as usize, &SPREAD_ICDF, 5);
		}

		// Dynamic allocation: boosts in quanta, signalled band by band
		// exactly as the decoder reads them.
		let cap = init_caps(lm, c);
		let wanted =
			dynalloc_analysis(&band_log, &self.old_band_e, end, c, lm, transient, bytes as i32);
		let mut offsets = [0i32; NB_EBANDS];
		let mut total_frac = total_bits << BITRES;
		let mut tell_frac = enc.tell_frac() as i32;
		let mut dynalloc_logp = 6;
		for i in start..end {
			let width = ((c as i32) * (EBANDS[i + 1] - EBANDS[i]) as i32) << lm;
			let quanta = (width << BITRES).min((6 << BITRES).max(width));
			let mut loop_logp = dynalloc_logp;
			let mut boost = 0;
			let mut j = 0;
			while tell_frac + (loop_logp << BITRES) < total_frac && boost < cap[i] {
				let flag = j < wanted[i];
				enc.bit_logp(flag, loop_logp as u32);
				tell_frac = enc.tell_frac() as i32;
				if !flag {
					break;
				}
				boost += quanta;
				total_frac -= quanta;
				loop_logp = 1;
				j += 1;
			}
			offsets[i] = boost;
			if boost > 0 {
				dynalloc_logp = (dynalloc_logp - 1).max(2);
			}
		}
		let equiv_rate = (bytes * 8 * 48_000 / n) as i32;
		if c == 2 {
			self.intensity = hysteresis_decision(
				equiv_rate / 1000,
				&INTENSITY_THRESHOLDS,
				&INTENSITY_HYSTERESIS,
				self.intensity,
			);
		}
		// The decoder assumes 5 when there's no room to signal it.
		let mut alloc_trim = 5;
		if tell_frac + (6 << BITRES) <= total_frac {
			alloc_trim = alloc_trim_analysis(&x, &band_log, end, lm, c, self.intensity, equiv_rate);
			enc.icdf(alloc_trim, &TRIM_ICDF, 7);
		}

		let mut bits = (((bytes * 8) as i32) << BITRES) - enc.tell_frac() as i32 - 1;
		let anti_collapse_rsv =
			if transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) { 1 << BITRES } else { 0 };
		bits -= anti_collapse_rsv;
		let choices = AllocChoices {
			intensity: self.intensity.min(end),
			dual_stereo: false,
			signal_bandwidth: end - 1,
			prev_coded: self.prev_coded,
		};
		let alloc = compute_allocation(
			start,
			end,
			&offsets,
			&cap,
			alloc_trim as i32,
			bits,
			c,
			lm,
			&mut enc,
			Some(choices),
		);
		self.prev_coded = alloc.coded_bands;
		self.quant_fine_energy(start, end, &alloc.fine_quant, &mut error, &mut enc);

		let mut collapse_masks = vec![0u8; c * NB_EBANDS];
		{
			let (xs, ys) = x.split_at_mut(n);
			let y = if c == 2 { Some(ys) } else { None };
			let mut seed = self.rng;
			quant_all_bands(
				&mut enc,
				Some(&band_e),
				start,
				end,
				xs,
				y,
				&mut collapse_masks,
				&alloc.pulses,
				transient,
				spread,
				alloc.dual_stereo,
				alloc.intensity,
				&tf_res,
				((bytes * 8) as i32) * (1 << BITRES) - anti_collapse_rsv,
				alloc.balance,
				lm,
				alloc.coded_bands,
				&mut seed,
				c == 1,
			);
		}
		if anti_collapse_rsv > 0 {
			enc.bits(u32::from(self.consec_transient < 2), 1);
		}
		let bits_left = (bytes * 8) as i32 - enc.tell();
		self.quant_energy_finalise(
			start,
			end,
			&alloc.fine_quant,
			&alloc.fine_priority,
			bits_left,
			&mut error,
			&mut enc,
		);

		self.consec_transient = if transient { self.consec_transient + 1 } else { 0 };
		self.transients += u64::from(transient);
		if c == 1 {
			let (a, b) = self.old_band_e.split_at_mut(NB_EBANDS);
			b.copy_from_slice(a);
		}
		for ch in 0..c {
			self.in_mem[ch * ov..(ch + 1) * ov].copy_from_slice(&inp[ch * stride + n..(ch + 1) * stride]);
		}
		self.rng = enc.range();
		self.frames += 1;
		enc.finish()
	}

	/// `quant_coarse_energy` (single pass): codes the energies' prediction
	/// residual and updates `old_band_e` exactly as the decoder's
	/// `unquant_coarse_energy` does.
	fn quant_coarse_energy(
		&mut self,
		target: &[f32; 2 * NB_EBANDS],
		start: usize,
		end: usize,
		intra: bool,
		enc: &mut RangeEncoder,
		lm: usize,
		bytes: usize,
		error: &mut [f32; 2 * NB_EBANDS],
	) {
		let c = self.channels;
		let model = &E_PROB_MODEL[(lm * 2 + usize::from(intra)) * 42..][..42];
		let (coef, beta) = if intra { (0.0, BETA_INTRA) } else { (PRED_COEF[lm], BETA_COEF[lm]) };
		let mut prev = [0f32; 2];
		let budget = (enc.storage() * 8) as i32;
		// Energy may not fall faster than this per frame (for bands that
		// hold one bin).
		let max_decay = (16.0f32).min(0.125 * bytes as f32);
		for i in start..end {
			for ch in 0..c {
				let k = i + ch * NB_EBANDS;
				let x = target[k];
				let old_e = self.old_band_e[k].max(-9.0);
				let f = x - coef * old_e - prev[ch];
				let mut qi = (0.5 + f).floor() as i32;
				let decay_bound = (-28.0f32).max(self.old_band_e[k] - max_decay);
				if qi < 0 && x < decay_bound {
					qi += (decay_bound - x) as i32;
					qi = qi.min(0);
				}
				let tell = enc.tell();
				let bits_left = budget - tell - 3 * c as i32 * (end - i) as i32;
				if i != start && bits_left < 30 {
					if bits_left < 24 {
						qi = qi.min(1);
					}
					if bits_left < 16 {
						qi = qi.max(-1);
					}
				}
				if budget - tell >= 15 {
					let pi = 2 * i.min(20);
					qi = enc.laplace(qi, u32::from(model[pi]) << 7, i32::from(model[pi + 1]) << 6);
				} else if budget - tell >= 2 {
					qi = qi.clamp(-1, 1);
					enc.icdf(((2 * qi) ^ -i32::from(qi < 0)) as usize, &SMALL_ENERGY_ICDF, 2);
				} else if budget - tell >= 1 {
					qi = qi.min(0);
					enc.bit_logp(qi != 0, 1);
				} else {
					qi = -1;
				}
				error[k] = f - qi as f32;
				let q = qi as f32;
				self.old_band_e[k] = coef * old_e + prev[ch] + q;
				prev[ch] = prev[ch] + q - beta * q;
			}
		}
	}

	fn quant_fine_energy(
		&mut self,
		start: usize,
		end: usize,
		fine: &[i32; NB_EBANDS],
		error: &mut [f32; 2 * NB_EBANDS],
		enc: &mut RangeEncoder,
	) {
		for i in start..end {
			if fine[i] <= 0 {
				continue;
			}
			let frac = 1i32 << fine[i];
			for ch in 0..self.channels {
				let k = i + ch * NB_EBANDS;
				let q2 = (((error[k] + 0.5) * frac as f32).floor() as i32).clamp(0, frac - 1);
				enc.bits(q2 as u32, fine[i] as u32);
				let offset = (q2 as f32 + 0.5) * (1 << (14 - fine[i])) as f32 * (1.0 / 16384.0) - 0.5;
				self.old_band_e[k] += offset;
				error[k] -= offset;
			}
		}
	}

	fn quant_energy_finalise(
		&mut self,
		start: usize,
		end: usize,
		fine: &[i32; NB_EBANDS],
		priority: &[i32; NB_EBANDS],
		mut bits_left: i32,
		error: &mut [f32; 2 * NB_EBANDS],
		enc: &mut RangeEncoder,
	) {
		let c = self.channels;
		for prio in 0..2 {
			let mut i = start;
			while i < end && bits_left >= c as i32 {
				if fine[i] >= MAX_FINE_BITS || priority[i] != prio {
					i += 1;
					continue;
				}
				for ch in 0..c {
					let k = i + ch * NB_EBANDS;
					let q2 = u32::from(error[k] >= 0.0);
					enc.bits(q2, 1);
					let offset = (q2 as f32 - 0.5) * (1 << (14 - fine[i] - 1)) as f32 * (1.0 / 16384.0);
					self.old_band_e[k] += offset;
					error[k] -= offset;
					bits_left -= 1;
				}
				i += 1;
			}
		}
	}
}

/// Opus stream encoder: CELT frames in single-frame packets.
#[derive(Debug, Clone)]
pub struct OpusEncoderState {
	pub channels: u8,
	pub bitrate: u32,
	/// Samples per channel per packet (120, 240, 480 or 960).
	pub frame_size: usize,
	celt: CeltEncoder,
	/// Interleaved input not yet coded.
	input: Vec<f32>,
	pub samples_in: u64,
	frames_out: u64,
}

impl OpusEncoderState {
	pub fn new(channels: u8, bitrate: u32, frame_size: usize) -> Result<Self> {
		if !(1..=2).contains(&channels) {
			return Err(Message::Codec("opus encoder: 1 or 2 channels are supported"));
		}
		if ![120, 240, 480, 960].contains(&frame_size) {
			return Err(Message::Codec("opus encoder: frame size must be 2.5, 5, 10 or 20 ms"));
		}
		Ok(Self {
			channels,
			bitrate: bitrate.clamp(6_000, 510_000),
			frame_size,
			celt: CeltEncoder::new(usize::from(channels)),
			input: Vec::new(),
			samples_in: 0,
			frames_out: 0,
		})
	}

	/// `OpusHead` for this stream (the extradata Ogg, Matroska and MP4
	/// carry).
	pub fn head(&self) -> OpusHead {
		OpusHead::new(self.channels, PRE_SKIP, 48_000)
	}

	/// CELT bytes of each frame (the packet has one more, the TOC).
	pub fn frame_bytes(&self) -> usize {
		let bytes = (u64::from(self.bitrate) * self.frame_size as u64 / (8 * 48_000)) as usize;
		bytes.saturating_sub(1).clamp(8, MAX_BYTES)
	}

	/// Final range of the last packet (equal to a decoder's after it).
	pub fn final_range(&self) -> u32 {
		self.celt.final_range()
	}

	/// The CELT encoder (statistics).
	pub fn celt(&self) -> &CeltEncoder {
		&self.celt
	}

	/// Queues interleaved samples (unit scale).
	pub fn push(&mut self, samples: &[f32]) {
		self.input.extend_from_slice(samples);
		self.samples_in += (samples.len() / usize::from(self.channels)) as u64;
	}

	/// Codes one packet from the queue if a whole frame is there.
	pub fn next_packet(&mut self) -> Option<Vec<u8>> {
		let need = self.frame_size * usize::from(self.channels);
		if self.input.len() < need {
			return None;
		}
		let pcm: Vec<f32> = self.input.drain(..need).collect();
		Some(self.code(&pcm))
	}

	/// Packets needed so a decoder outputs every input sample after the
	/// pre-skip.
	fn frames_needed(&self) -> u64 {
		(self.samples_in + u64::from(PRE_SKIP)).div_ceil(self.frame_size as u64)
	}

	fn code(&mut self, pcm: &[f32]) -> Vec<u8> {
		let lm = (0..=MAX_LM).find(|&l| SHORT_MDCT << l == self.frame_size).unwrap_or(MAX_LM);
		let config = 28 + lm as u8; // CELT-only, fullband
		let toc = (config << 3) | (u8::from(self.channels == 2) << 2);
		let bytes = self.frame_bytes();
		let mut packet = Vec::with_capacity(bytes + 1);
		packet.push(toc);
		packet.extend(self.celt.encode(pcm, lm, bytes));
		self.frames_out += 1;
		packet
	}

	/// Codes one packet of the remaining input, padded with silence, while
	/// a decoder still lacks samples.
	pub fn flush_packet(&mut self) -> Option<Vec<u8>> {
		if self.frames_out >= self.frames_needed() {
			return None;
		}
		let need = self.frame_size * usize::from(self.channels);
		if self.input.len() < need {
			self.input.resize(need, 0.0);
		}
		let pcm: Vec<f32> = self.input.drain(..need).collect();
		Some(self.code(&pcm))
	}

	/// Codes the rest of the input, padded with silence.
	pub fn flush(&mut self) -> Vec<Vec<u8>> {
		std::iter::from_fn(|| self.flush_packet()).collect()
	}
}

/// Opus encoder (`-c:a opus` / `libopus`).
#[derive(Debug, Clone, Copy)]
pub struct OpusEncoder {
	stream_id: StreamId,
	bitrate: Option<u32>,
	frame_size: usize,
}

impl OpusEncoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self { stream_id, bitrate: None, frame_size: DEFAULT_FRAME }
	}

	/// `-b:a` (default 64 kb/s per channel).
	pub fn with_bitrate(mut self, bitrate: Option<u32>) -> Self {
		self.bitrate = bitrate.filter(|&b| b > 0);
		self
	}

	/// Frame length in samples at 48 kHz (120, 240, 480 or 960).
	pub fn with_frame_size(mut self, samples: usize) -> Self {
		self.frame_size = samples;
		self
	}

	fn new_state(&self, parameters: &Parameters) -> Result<OpusEncoderState> {
		if parameters.sample_rate != Some(48_000) {
			return Err(Message::Codec("opus encoder: the input must be 48 kHz (use -ar 48000)"));
		}
		let channels = parameters.channels.unwrap_or(0);
		let bitrate = self.bitrate.unwrap_or_else(|| default_bitrate(channels));
		OpusEncoderState::new(channels, bitrate, self.frame_size)
	}

	/// `OpusHead` extradata for `parameters`.
	pub fn extradata(&self, parameters: &Parameters) -> Result<Vec<u8>> {
		Ok(self.new_state(parameters)?.head().to_bytes())
	}

	/// Bit rate the encoder will use (after clamping).
	pub fn bitrate_for(&self, parameters: &Parameters) -> u32 {
		self.new_state(parameters).map_or(0, |s| s.bitrate)
	}

	fn state<'a>(&self, state: &'a mut State) -> Result<&'a mut OpusEncoderState> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("opus encoder requires an audio state"));
		};
		match &mut audio.private {
			CodecPrivate::OpusEncoder(enc) => Ok(enc),
			_ => Err(Message::Codec("opus encoder: unsupported parameters (48 kHz, 1 or 2 channels)")),
		}
	}

	fn packet(&self, data: Vec<u8>, index: u64, enc: &OpusEncoderState) -> Packet {
		let tb = TimeBase::new(1, 48_000);
		let n = enc.frame_size as u64;
		let start = index * n;
		// The last packet ends with the input (plus the pre-skip).
		let total = enc.samples_in + u64::from(PRE_SKIP);
		let duration = total.saturating_sub(start).clamp(1, n);
		let mut packet = Packet::new(self.stream_id, data);
		packet.pts = Some(Time::new(start, tb));
		packet.dts = packet.pts;
		packet.duration = Some(Time::new(duration, tb));
		packet.keyframe = true;
		packet
	}
}

impl core::Encoder for OpusEncoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let enc = self.new_state(parameters).ok();
		State::Audio(AudioState {
			sample_rate: 48_000,
			channels: Channels::from_count(parameters.channels.unwrap_or(2)),
			format: SampleFormat::F32,
			layout: SampleLayout::Interleaved,
			frame_size: self.frame_size,
			codec_state: CodecState {
				extradata: enc.as_ref().map(|e| e.head().to_bytes()).unwrap_or_default(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: enc.map_or(CodecPrivate::None, |e| CodecPrivate::OpusEncoder(Box::new(e))),
		})
	}

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut> {
		let Frame::Audio(frame) = frame else {
			return Err(Message::Codec("opus encoder requires audio frames"));
		};
		let enc = self.state(state)?;
		if frame.channels.count() != enc.channels {
			return Err(Message::Codec("opus encoder: frame channel count does not match the stream"));
		}
		if frame.sample_rate != 48_000 {
			return Err(Message::Codec("opus encoder: frames must be 48 kHz (resample first)"));
		}
		let channels = usize::from(enc.channels);
		let mut samples = Vec::with_capacity(frame.nb_samples * channels);
		for_each_sample(&frame, |s| samples.push(s.to_float() as f32))?;
		let count = (samples.len() / channels) as u64;
		enc.push(&samples);
		let mut packets = Vec::new();
		while let Some(data) = enc.next_packet() {
			let index = enc.frames_out - 1;
			packets.push(self.packet(data, index, enc));
		}
		if let State::Audio(audio) = state {
			audio.samples_consumed += count;
		}
		Ok(EncodeOut { value: packets })
	}

	fn flush(&self, state: &mut State) -> Result<EncodeOut> {
		let enc = self.state(state)?;
		let first = enc.frames_out;
		let data = enc.flush();
		let packets =
			data.into_iter().enumerate().map(|(k, d)| self.packet(d, first + k as u64, enc)).collect();
		Ok(EncodeOut { value: packets })
	}
}

#[cfg(test)]
mod tests;
