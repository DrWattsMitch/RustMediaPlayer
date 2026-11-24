//! Temporal noise shaping analysis (ISO/IEC 14496-3 §4.6.9, 3GPP TS 26.403
//! §5.4.3, FFmpeg `libavcodec/aacenc_tns.c`).
//!
//! An LPC predictor run *across frequency* has a prediction gain that grows
//! with how uneven the signal's envelope is *in time* inside the frame
//! (speech pitch pulses, castanets). Coding the prediction residual instead
//! of the spectrum makes the decoder's all-pole synthesis filter shape the
//! quantisation noise like the signal's temporal envelope, so the noise
//! hides under the loud parts instead of spreading as pre-echo.
//!
//! Per window, one filter covers the bands from [`START_HZ`] up to the TNS
//! limit of the sampling rate (Table 4.155) or the bandwidth, whichever is
//! lower. The filter is found by Levinson-Durbin on a lag-windowed
//! autocorrelation of that region, its reflection coefficients are
//! quantised with 4-bit resolution exactly as the decoder dequantises them
//! (`decode_tns`), and the residual is computed with the *quantised*
//! coefficients so the decoder's filter is the exact inverse.
//!
//! The filter always runs upwards in frequency (`direction` 0). The decoder
//! clips the region to its `max_sfb`, which can be below the encoder's
//! band limit when the top bands quantise to nothing; an upward filter on a
//! prefix of the region gives the same values as on the whole region, so
//! that clipping never changes the reconstruction.

use crate::codecs::aac::ics::{TNS_MAX_BANDS_LONG, TNS_MAX_BANDS_SHORT, parcor_to_lpc};

/// Lowest frequency TNS filters (3GPP TS 26.403 `tnsStartFreq`).
const START_HZ: [f32; 2] = [1275.0, 2750.0];
/// Maximum order for AAC-LC: 12 for long windows, 7 for short ones.
const MAX_ORDER: [usize; 2] = [12, 7];
/// Smallest prediction gain worth the side information (FFmpeg's
/// `TNS_GAIN_THRESHOLD_LOW`, about 1.5 dB).
const MIN_GAIN: f64 = 1.4;
/// Gaussian lag window (bandwidth expansion), 3GPP `tnsAcfWindow`.
const LAG_ALPHA: f64 = 0.04;
/// 4-bit coefficients (`coef_res` = 1).
const RES_BITS: u32 = 4;

/// One window's filter as written in `tns_data()`.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
	/// `length` in bands, counted down from the top band of the window.
	pub length: u8,
	/// Quantised reflection coefficient indices (`order` of them).
	pub index: Vec<i8>,
	/// 3-bit coefficients (`coef_compress`).
	pub compress: bool,
	/// Energy of the spectrum over the residual in the filtered region.
	pub gain: f32,
	/// Filtered lines of the window.
	pub start: usize,
	pub end: usize,
}

/// TNS of one channel and frame: a filter (or none) per window.
#[derive(Debug, Clone, Default)]
pub struct Tns {
	pub short: bool,
	pub windows: Vec<Option<Filter>>,
}

impl Tns {
	pub fn present(&self) -> bool {
		self.windows.iter().any(Option::is_some)
	}

	/// Bits of `tns_data()` (Table 4.48).
	pub fn bits(&self) -> usize {
		let (n_filt, length, order) = if self.short { (1, 4, 3) } else { (2, 6, 5) };
		self
			.windows
			.iter()
			.map(|w| match w {
				None => n_filt,
				Some(f) => {
					let coef = RES_BITS as usize - usize::from(f.compress);
					n_filt + 1 + length + order + 2 + f.index.len() * coef
				}
			})
			.sum()
	}

	pub fn write(&self, w: &mut crate::codecs::bitstream::BitWriter) {
		let (n_filt, length, order) = if self.short { (1, 4, 3) } else { (2, 6, 5) };
		for win in &self.windows {
			match win {
				None => w.put(0, n_filt),
				Some(f) => {
					w.put(1, n_filt);
					w.put(RES_BITS - 3, 1); // coef_res: 4 bits
					w.put(u32::from(f.length), length);
					w.put(f.index.len() as u32, order);
					w.bit(false); // direction: upwards
					w.bit(f.compress);
					let bits = RES_BITS - u32::from(f.compress);
					let mask = (1u32 << bits) - 1;
					for &i in &f.index {
						w.put(i as i32 as u32 & mask, bits);
					}
				}
			}
		}
	}
}

/// Quantisation steps of §4.6.9.3 for 4-bit coefficients.
fn steps() -> (f64, f64) {
	let half = f64::from(1u32 << (RES_BITS - 1));
	((half - 0.5) / std::f64::consts::FRAC_PI_2, (half + 0.5) / std::f64::consts::FRAC_PI_2)
}

/// The decoder's dequantisation of an index (`decode_tns`).
fn dequantise(index: i8) -> f32 {
	let (pos, neg) = steps();
	let v = f64::from(index);
	(v / if v >= 0.0 { pos } else { neg }).sin() as f32
}

fn quantise(k: f64) -> i8 {
	let (pos, neg) = steps();
	let a = k.clamp(-0.999, 0.999).asin();
	let max = (1i32 << (RES_BITS - 1)) - 1;
	((a * if a >= 0.0 { pos } else { neg }).round() as i32).clamp(-max - 1, max) as i8
}

/// Levinson-Durbin: reflection coefficients of the prediction-error filter
/// `1 + Σ a_i z^-i` (the convention of the decoder's step-up recursion) and
/// the prediction gain.
fn levinson(r: &[f64]) -> (Vec<f64>, f64) {
	let order = r.len() - 1;
	let mut a = vec![0.0f64; order + 1];
	let mut k = vec![0.0f64; order];
	let mut err = r[0];
	for m in 1..=order {
		if err <= r[0] * 1e-9 {
			break;
		}
		let mut acc = r[m];
		for j in 1..m {
			acc += a[j] * r[m - j];
		}
		let km = -acc / err;
		k[m - 1] = km;
		let prev = a.clone();
		a[m] = km;
		for j in 1..m {
			a[j] = prev[j] + km * prev[m - j];
		}
		err *= 1.0 - km * km;
	}
	(k, if err > 0.0 { r[0] / err } else { 1.0 })
}

/// Finds and applies the filters for one channel's spectrum (`spec` is
/// window after window, `win_len` lines each), replacing it by the
/// residual. `offsets` are the window's band offsets and `bands` the
/// number of bands the encoder may code (bandwidth limit).
pub fn analyse(
	spec: &mut [f32],
	short: bool,
	offsets: &[u16],
	bands: usize,
	rate_index: usize,
	sample_rate: u32,
) -> Tns {
	let s = usize::from(short);
	let (windows, win_len) = if short { (8, 128) } else { (1, 1024) };
	let num_swb = offsets.len() - 1;
	let limit = if short { TNS_MAX_BANDS_SHORT } else { TNS_MAX_BANDS_LONG };
	let top = usize::from(limit[rate_index.min(12)]).min(bands).min(num_swb);
	let hz_per_line = sample_rate as f32 / 2.0 / win_len as f32;
	let start_line = (START_HZ[s] / hz_per_line) as usize;
	let Some(start_band) = (0..top).find(|&b| usize::from(offsets[b]) >= start_line) else {
		return Tns { short, windows: vec![None; windows] };
	};
	let (lo, hi) = (usize::from(offsets[start_band]), usize::from(offsets[top]));
	let mut out = Tns { short, windows: Vec::with_capacity(windows) };
	for w in 0..windows {
		let region = &mut spec[w * win_len + lo..w * win_len + hi];
		out.windows.push(window_filter(region, s, (num_swb - start_band) as u8, lo, hi));
	}
	out
}

fn window_filter(x: &mut [f32], s: usize, length: u8, start: usize, end: usize) -> Option<Filter> {
	let n = x.len();
	let order = MAX_ORDER[s].min(n / 8);
	if order == 0 {
		return None;
	}
	let mut r = vec![0.0f64; order + 1];
	for (lag, rv) in r.iter_mut().enumerate() {
		*rv = (lag..n).map(|i| f64::from(x[i]) * f64::from(x[i - lag])).sum::<f64>();
		*rv *= (-0.5 * (LAG_ALPHA * lag as f64).powi(2)).exp();
	}
	if r[0] < 1e-3 * n as f64 {
		return None;
	}
	let (k, gain) = levinson(&r);
	if gain < MIN_GAIN {
		return None;
	}
	let mut index: Vec<i8> = k.iter().map(|&v| quantise(v)).collect();
	while index.last() == Some(&0) {
		index.pop();
	}
	if index.is_empty() {
		return None;
	}
	let parcor: Vec<f32> = index.iter().map(|&i| dequantise(i)).collect();
	let lpc = parcor_to_lpc(&parcor);
	let order = index.len();
	// Residual r[n] = x[n] + Σ lpc[i]·x[n-i]; from the top down so every
	// read sees the original spectrum. The decoder inverts it with
	// y[n] = r[n] - Σ lpc[i]·y[n-i].
	let energy: f32 = x.iter().map(|v| v * v).sum();
	for m in (0..n).rev() {
		let mut acc = x[m];
		for i in 1..=m.min(order) {
			acc += lpc[i - 1] * x[m - i];
		}
		x[m] = acc;
	}
	let residual: f32 = x.iter().map(|v| v * v).sum();
	let compress = index.iter().all(|&i| (-4..=3).contains(&i));
	Some(Filter {
		length,
		index,
		compress,
		gain: if residual > 0.0 { (energy / residual).max(1.0) } else { 1.0 },
		start,
		end,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The decoder's synthesis (`apply_tns`) for one upward filter.
	fn synthesise(x: &mut [f32], index: &[i8]) {
		let parcor: Vec<f32> = index.iter().map(|&i| dequantise(i)).collect();
		let lpc = parcor_to_lpc(&parcor);
		for m in 0..x.len() {
			let mut acc = x[m];
			for i in 1..=m.min(index.len()) {
				acc -= x[m - i] * lpc[i - 1];
			}
			x[m] = acc;
		}
	}

	#[test]
	fn residual_inverts_through_the_decoder_filter() {
		// A click at the start of a long window: the spectrum is a slow
		// cosine across frequency, which TNS predicts well.
		let mut spec: Vec<f32> = (0..1024)
			.map(|k| 5000.0 * ((k as f32 + 0.5) * 0.07).cos() + 300.0 * ((k * 13 % 7) as f32 - 3.0))
			.collect();
		let original = spec.clone();
		let offsets = crate::codecs::aac::tables::SWB_OFFSET_LONG[4];
		let tns = analyse(&mut spec, false, offsets, offsets.len() - 1, 4, 44_100);
		let f = tns.windows[0].as_ref().expect("filter for an impulsive window");
		assert!(f.gain > 2.0, "{}", f.gain);
		// Prefix clipping (decoder max_sfb below the encoder's limit) gives
		// the same values on the prefix.
		let mut prefix = spec.clone();
		synthesise(&mut prefix[f.start..f.end - 40], &f.index);
		synthesise(&mut spec[f.start..f.end], &f.index);
		for (k, (a, b)) in original.iter().zip(&spec).enumerate() {
			assert!((a - b).abs() < 0.05, "{k}: {a} {b}");
			if k < f.end - 40 {
				assert_eq!(*b, prefix[k]);
			}
		}
		let mut bits = crate::codecs::bitstream::BitWriter::new();
		tns.write(&mut bits);
		assert_eq!(bits.bits(), tns.bits());
	}

	#[test]
	fn steady_tone_gets_no_filter() {
		let mut spec = vec![0.0f32; 1024];
		spec[300] = 10000.0;
		spec[301] = -6000.0;
		for (k, v) in spec.iter_mut().enumerate() {
			*v += 20.0 * ((k * 7919 % 13) as f32 - 6.0);
		}
		let offsets = crate::codecs::aac::tables::SWB_OFFSET_LONG[4];
		let tns = analyse(&mut spec, false, offsets, offsets.len() - 1, 4, 44_100);
		assert!(!tns.present());
	}

	#[test]
	fn quantiser_matches_the_decoder_steps() {
		for i in -8i8..=7 {
			assert_eq!(quantise(f64::from(dequantise(i))), i);
		}
	}
}
