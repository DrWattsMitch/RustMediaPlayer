//! Psychoacoustic analysis: attack detection for block switching, masking
//! thresholds per scalefactor band, and perceptual entropy.
//!
//! This is a compact version of the model in 3GPP TS 26.403 §5.4, which
//! FFmpeg's default `aacpsy.c` also follows:
//!
//! - **Block switching.** The next frame's short-window region is
//!   high-pass filtered and cut into 128-sample blocks. An attack is a block
//!   whose energy jumps well above the recent average (FFmpeg's
//!   `psy_lame_window` uses the same idea with its own filter).
//! - **Thresholds.** Band energies are spread across neighbouring bands on
//!   the Bark scale (−15 dB/Bark towards higher bands, −27 dB/Bark towards
//!   lower ones, applied to band energies as 3GPP does), lowered by a fixed SNR and floored by the absolute
//!   threshold of hearing. The rate loop then scales all thresholds by one
//!   common factor until the frame fits its bit budget, so the model only
//!   has to get the *shape* right.
//! - **Perceptual entropy** (`Σ width · log2(1 + E/T)`) steers how many bits
//!   a frame takes from the reservoir.

use crate::codecs::aac::tables::{SWB_OFFSET_LONG, SWB_OFFSET_SHORT};

/// Masking below the spread energy, in dB.
const SNR_DB: f32 = 29.0;
/// Spreading slopes in dB per Bark.
const SPREAD_UP_DB: f32 = 15.0;
const SPREAD_DOWN_DB: f32 = 27.0;
/// A block whose high-passed energy exceeds this multiple of the recent
/// average is an attack.
const ATTACK_RATIO: f32 = 10.0;
/// Upper limit of the threshold in quiet, dB SPL.
const ATH_CAP_DB: f32 = 20.0;
/// Attacks quieter than this (energy of a 128-sample high-passed block, at
/// 16-bit sample scale) don't need short windows.
const ATTACK_FLOOR: f32 = 128.0 * 40.0 * 40.0;

fn bark(f: f32) -> f32 {
	13.0 * (0.00076 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

/// Absolute threshold of hearing in dB SPL (Terhardt), capped at
/// [`ATH_CAP_DB`]: the curve's steep rise at both ends assumes a known
/// playback level, and taken literally it would drop all content above
/// ~16 kHz (as LAME and FFmpeg, RMP limits its influence).
fn ath_db(f: f32) -> f32 {
	let k = (f.max(10.0)) / 1000.0;
	let db = 3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4);
	db.min(ATH_CAP_DB)
}

/// Band layout and the per-band constants of the model for one window
/// length at one sample rate.
#[derive(Debug, Clone)]
pub struct BandModel {
	pub offsets: &'static [u16],
	/// Spreading factors from band b-1 into b, and from b+1 into b.
	spread_up: Vec<f32>,
	spread_down: Vec<f32>,
	/// Threshold-in-quiet per band (energy).
	ath: Vec<f32>,
}

impl BandModel {
	pub fn new(rate_index: usize, sample_rate: u32, short: bool) -> Self {
		let band_index = rate_index.min(11);
		let offsets = if short { SWB_OFFSET_SHORT[band_index] } else { SWB_OFFSET_LONG[band_index] };
		let lines = if short { 128.0 } else { 1024.0 };
		let hz_per_line = sample_rate as f32 / 2.0 / lines;
		let n = offsets.len() - 1;
		let centre: Vec<f32> = (0..n)
			.map(|b| bark((f32::from(offsets[b]) + f32::from(offsets[b + 1])) * 0.5 * hz_per_line))
			.collect();
		let factor = |db_per_bark: f32, d: f32| 10f32.powf(-db_per_bark * d / 10.0);
		let spread_up = (0..n)
			.map(|b| if b == 0 { 0.0 } else { factor(SPREAD_UP_DB, centre[b] - centre[b - 1]) })
			.collect();
		let spread_down = (0..n)
			.map(|b| if b + 1 == n { 0.0 } else { factor(SPREAD_DOWN_DB, centre[b + 1] - centre[b]) })
			.collect();
		// Calibration: a full-scale sine is 96 dB SPL, and its MDCT peak is
		// about amplitude · N/2 with the encoder's scaling (2 · Σ w·x·cos).
		let peak = 32768.0 * lines;
		let full_scale = peak * peak;
		let ath = (0..n)
			.map(|b| {
				let (lo, hi) = (usize::from(offsets[b]), usize::from(offsets[b + 1]));
				let db = (lo..hi).map(|k| ath_db((k as f32 + 0.5) * hz_per_line)).fold(f32::MAX, f32::min);
				let per_line = full_scale * 10f32.powf((db - 96.0) / 10.0);
				per_line * (hi - lo) as f32
			})
			.collect();
		Self { offsets, spread_up, spread_down, ath }
	}

	pub fn bands(&self) -> usize {
		self.offsets.len() - 1
	}

	/// Masking thresholds for `energy` (one window), into `out`.
	pub fn thresholds(&self, energy: &[f32], out: &mut [f32]) {
		let n = energy.len().min(out.len());
		// Band energies are spread directly (3GPP TS 26.403 §5.4.3):
		// spreading densities instead would let one loud narrow band
		// raise a wide high band's threshold by the ratio of their widths.
		out[..n].copy_from_slice(&energy[..n]);
		for b in 1..n {
			out[b] = out[b].max(out[b - 1] * self.spread_up[b]);
		}
		for b in (0..n.saturating_sub(1)).rev() {
			out[b] = out[b].max(out[b + 1] * self.spread_down[b]);
		}
		let snr = 10f32.powf(-SNR_DB / 10.0);
		for b in 0..n {
			out[b] = (out[b] * snr).max(self.ath[b]);
		}
	}
}

/// Perceptual entropy of bands with `energy` and `threshold`.
pub fn perceptual_entropy(energy: f32, threshold: f32, width: usize) -> f32 {
	if energy > threshold && threshold > 0.0 {
		width as f32 * (energy / threshold).log2().min(30.0)
	} else {
		0.0
	}
}

/// `true` if `x` (a stretch of samples; the first `context` of them only
/// prime the average) contains an attack.
pub fn has_attack(x: &[f32], context: usize) -> bool {
	const BLOCK: usize = 128;
	if x.len() < 2 {
		return false;
	}
	// Second-order high-pass (difference of differences) removes the low
	// end that dominates energy without causing pre-echo.
	let mut energies = Vec::with_capacity(x.len() / BLOCK + 1);
	let mut e = 0.0f32;
	let mut count = 0;
	for i in 2..x.len() {
		let h = x[i] - 2.0 * x[i - 1] + x[i - 2];
		e += h * h;
		count += 1;
		if count == BLOCK {
			energies.push(e);
			e = 0.0;
			count = 0;
		}
	}
	let first = context / BLOCK;
	let mut average = energies.iter().take(first.max(1)).sum::<f32>() / first.max(1) as f32;
	for (i, &e) in energies.iter().enumerate().skip(first) {
		if e > ATTACK_FLOOR && e > ATTACK_RATIO * average {
			return true;
		}
		// Slowly rising signals keep raising the reference.
		average = if i == 0 { e } else { 0.6 * average + 0.4 * e };
	}
	false
}

/// Short-window grouping as the 7-bit `scale_factor_grouping`: windows of
/// similar energy share a group (and its scalefactors); a jump of more than
/// 3x either way starts a new one.
pub fn grouping(window_energy: &[f32; 8]) -> u8 {
	let mut bits = 0u8;
	let (mut sum, mut count) = (window_energy[0], 1.0f32);
	for w in 1..8 {
		let avg = sum / count;
		let e = window_energy[w];
		let similar = e <= 3.0 * avg + 1.0 && 3.0 * e + 1.0 >= avg;
		if similar {
			bits |= 1 << (7 - w);
			sum += e;
			count += 1.0;
		} else {
			sum = e;
			count = 1.0;
		}
	}
	bits
}

/// Window group lengths from the grouping bits.
pub fn group_lengths(grouping: u8) -> Vec<usize> {
	let mut groups = vec![1usize];
	for w in 1..8 {
		if grouping & (1 << (7 - w)) != 0 {
			if let Some(last) = groups.last_mut() {
				*last += 1;
			}
		} else {
			groups.push(1);
		}
	}
	groups
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn detects_a_click_but_not_a_steady_tone() {
		let tone: Vec<f32> = (0..1536).map(|i| 8000.0 * (i as f32 * 0.05).sin()).collect();
		assert!(!has_attack(&tone, 384));
		let mut click = vec![0.0f32; 1536];
		for (i, v) in click.iter_mut().enumerate().skip(1000) {
			*v = 20000.0 * ((i * 7919 % 101) as f32 / 50.0 - 1.0);
		}
		assert!(has_attack(&click, 384));
	}

	#[test]
	fn grouping_round_trips() {
		let g = grouping(&[1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0]);
		assert_eq!(group_lengths(g), vec![3, 2, 3]);
	}

	#[test]
	fn thresholds_respect_ath_and_spread() {
		let m = BandModel::new(4, 44100, false);
		let mut e = vec![0.0; m.bands()];
		e[10] = 1e12;
		let mut t = vec![0.0; m.bands()];
		m.thresholds(&e, &mut t);
		assert!(t[11] > m.ath[11] && t[11] < t[10]);
		assert!(t[40] == m.ath[40]);
	}
}
