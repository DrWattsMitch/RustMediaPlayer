//! Dithering and noise shaping for integer output, after libswresample's
//! `-dither_method` / `-dither_scale` (`libswresample/dither.c`).
//!
//! Rounding a high-resolution signal to 16 (or 8, or 24) bits leaves an
//! error that follows the signal: quiet passages and fades turn into
//! harmonic distortion and "granular" noise. Adding a little random noise
//! before rounding (dither) turns that error into a steady, signal-
//! independent hiss, about 1 LSB in size, and keeps detail below the last
//! bit audible. Noise shaping goes further: it feeds the rounding error
//! back through a filter so the hiss moves out of the 2-5 kHz region the
//! ear is most sensitive to, towards the top of the band.
//!
//! | Method | Noise | Where |
//! | --- | --- | --- |
//! | `rectangular` | uniform, ±0.5 LSB | flat |
//! | `triangular` | sum of two uniforms (TPDF), ±1 LSB | flat — the standard choice |
//! | `triangular_hp` | difference of successive uniforms, ±1 LSB | rising with frequency |
//! | `lipshitz` | TPDF + 5-tap error feedback (Lipshitz, Vanderkooy & Wannamaker, 1991) | shaped away from 2-5 kHz; 44.1 / 48 kHz only |
//!
//! The noise generator is FFmpeg's linear congruential one, seeded per
//! channel, so output is deterministic: the same input always gives the
//! same bytes.

use crate::message::{Message, Result};

/// `-dither_method` names, for errors and `--help`.
pub const DITHER_METHODS: &str = "none, rectangular, triangular, triangular_hp, lipshitz";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitherMethod {
	/// Plain rounding (or truncation, for integer → narrower integer, as
	/// `swr_convert` does without dither).
	#[default]
	None,
	Rectangular,
	Triangular,
	TriangularHighPass,
	Lipshitz,
}

impl DitherMethod {
	pub fn from_name(name: &str) -> Option<Self> {
		Some(match name {
			"none" | "0" => Self::None,
			"rectangular" => Self::Rectangular,
			"triangular" => Self::Triangular,
			"triangular_hp" => Self::TriangularHighPass,
			"lipshitz" => Self::Lipshitz,
			_ => return None,
		})
	}

	pub fn name(self) -> &'static str {
		match self {
			Self::None => "none",
			Self::Rectangular => "rectangular",
			Self::Triangular => "triangular",
			Self::TriangularHighPass => "triangular_hp",
			Self::Lipshitz => "lipshitz",
		}
	}
}

/// A dither method and its noise level (`-dither_scale`, 1 = the
/// method's nominal amplitude in output LSBs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dither {
	pub method: DitherMethod,
	pub scale: f64,
}

impl Default for Dither {
	fn default() -> Self {
		Self::NONE
	}
}

impl Dither {
	pub const NONE: Dither = Dither { method: DitherMethod::None, scale: 1.0 };
	pub const TRIANGULAR: Dither = Dither { method: DitherMethod::Triangular, scale: 1.0 };

	pub fn new(method: DitherMethod) -> Self {
		Self { method, scale: 1.0 }
	}

	pub fn is_none(&self) -> bool {
		self.method == DitherMethod::None || self.scale == 0.0
	}

	/// Checks that the method can run at `sample_rate`.
	pub fn check(&self, sample_rate: u32) -> Result<()> {
		if !self.scale.is_finite() || self.scale < 0.0 {
			return Err(Message::Other("dither scale must be a finite number >= 0"));
		}
		if self.method == DitherMethod::Lipshitz && !matches!(sample_rate, 44_100 | 48_000) {
			return Err(Message::Other(
				"lipshitz noise shaping needs a 44100 or 48000 Hz output (use -dither_method triangular)",
			));
		}
		Ok(())
	}
}

/// Error-feedback coefficients of the Lipshitz "minimally audible"
/// E-weighted shaper, designed for 44.1 kHz. The noise transfer function
/// is `1 - Σ c[k]·z^-(k+1)`: about −17 dB at DC, a dip near 4 kHz and the
/// excess pushed above 15 kHz. At 48 kHz the same curve sits 9% higher in
/// frequency, which keeps it out of the sensitive band.
const LIPSHITZ: [f64; 5] = [2.033, -2.165, 1.959, -1.590, 0.6149];

#[derive(Debug, Clone)]
struct Channel {
	seed: u32,
	/// Previous uniform value (`triangular_hp`).
	previous: f64,
	/// Last rounding errors, newest first (`lipshitz`).
	errors: [f64; 5],
}

impl Channel {
	/// FFmpeg's LCG (`seed * 1664525 + 1013904223`), as a uniform value in
	/// [-0.5, 0.5).
	fn uniform(&mut self) -> f64 {
		self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
		f64::from(self.seed) / 4_294_967_296.0 - 0.5
	}
}

/// Quantises full-scale samples (`-1.0..1.0`) to `bits`-bit integers with
/// dither, one channel state per channel.
#[derive(Debug, Clone)]
pub struct Quantizer {
	dither: Dither,
	/// 2^(bits-1): one full-scale unit in output LSBs.
	scale: f64,
	min: f64,
	max: f64,
	channels: Vec<Channel>,
}

impl Quantizer {
	pub fn new(dither: Dither, bits: u32, channels: usize, sample_rate: u32) -> Result<Self> {
		if !(2..=32).contains(&bits) {
			return Err(Message::Other("dither: output precision must be 2 to 32 bits"));
		}
		dither.check(sample_rate)?;
		let full = (1u64 << (bits - 1)) as f64;
		let channels = (0..channels)
			.map(|c| Channel {
				// Distinct, fixed seeds per channel: uncorrelated noise
				// between channels, identical output on every run.
				seed: 0x9E37_79B9 ^ (c as u32).wrapping_mul(0x85EB_CA6B),
				previous: 0.0,
				errors: [0.0; 5],
			})
			.collect();
		Ok(Self { dither, scale: full, min: -full, max: full - 1.0, channels })
	}

	/// One sample of `channel` → a signed integer of the output precision,
	/// saturated.
	pub fn quantize(&mut self, channel: usize, x: f64) -> i64 {
		let x = if x.is_nan() { 0.0 } else { x };
		let Some(state) = self.channels.get_mut(channel) else {
			return (x * self.scale).round_ties_even().clamp(self.min, self.max) as i64;
		};
		let v = x * self.scale;
		let amount = self.dither.scale;
		let (target, noise) = match self.dither.method {
			DitherMethod::None => (v, 0.0),
			DitherMethod::Rectangular => (v, state.uniform() * amount),
			DitherMethod::Triangular => (v, (state.uniform() + state.uniform()) * amount),
			DitherMethod::TriangularHighPass => {
				let u = state.uniform();
				let d = u - state.previous;
				state.previous = u;
				(v, d * amount)
			}
			DitherMethod::Lipshitz => {
				let feedback: f64 = LIPSHITZ.iter().zip(&state.errors).map(|(c, e)| c * e).sum();
				(v - feedback, (state.uniform() + state.uniform()) * amount)
			}
		};
		let q = (target + noise).round_ties_even();
		if self.dither.method == DitherMethod::Lipshitz {
			// The error before clipping: it stays within ±1.5 LSB whatever
			// the signal does, so the feedback loop can't run away.
			state.errors.rotate_right(1);
			state.errors[0] = q - target;
		}
		q.clamp(self.min, self.max) as i64
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Power of a signal in a band, by direct DFT over a few bins
	/// (enough for a test; no FFT needed).
	fn band_power(x: &[f64], rate: f64, from: f64, to: f64) -> f64 {
		let n = x.len();
		let mut total = 0.0;
		let first = (from / rate * n as f64) as usize;
		let last = (to / rate * n as f64) as usize;
		for k in (first.max(1)..=last).step_by(8) {
			let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
			let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (i, &v)| {
				(re + v * (w * i as f64).cos(), im - v * (w * i as f64).sin())
			});
			total += re * re + im * im;
		}
		total
	}

	fn error_signal(method: DitherMethod, input: impl Fn(usize) -> f64, n: usize) -> Vec<f64> {
		let mut q = Quantizer::new(Dither::new(method), 16, 1, 44_100).unwrap();
		(0..n).map(|i| q.quantize(0, input(i)) as f64 - input(i) * 32768.0).collect()
	}

	#[test]
	fn names_round_trip() {
		for name in DITHER_METHODS.split(", ") {
			assert_eq!(DitherMethod::from_name(name).unwrap().name(), name);
		}
		assert!(DitherMethod::from_name("shibata").is_none());
	}

	#[test]
	fn noise_is_bounded_and_unbiased() {
		for method in
			[DitherMethod::Rectangular, DitherMethod::Triangular, DitherMethod::TriangularHighPass]
		{
			let e = error_signal(method, |i| (i as f64 * 0.001).sin() * 0.25, 20_000);
			let peak = e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
			let mean = e.iter().sum::<f64>() / e.len() as f64;
			let limit = if method == DitherMethod::Rectangular { 1.0 } else { 1.5 };
			assert!(peak <= limit, "{method:?} peak {peak}");
			assert!(mean.abs() < 0.05, "{method:?} mean {mean}");
		}
	}

	#[test]
	fn tpdf_makes_the_error_independent_of_the_signal() {
		// A sine of 0.4 LSB amplitude: rounding alone deletes it entirely,
		// dither keeps it (the average output follows the input).
		let input = |i: usize| (i as f64 * 0.01).sin() * 0.4 / 32768.0;
		let mut plain = Quantizer::new(Dither::NONE, 16, 1, 44_100).unwrap();
		assert!((0..10_000).all(|i| plain.quantize(0, input(i)) == 0));
		let mut q = Quantizer::new(Dither::TRIANGULAR, 16, 1, 44_100).unwrap();
		let out: Vec<f64> = (0..200_000).map(|i| q.quantize(0, input(i)) as f64).collect();
		let correlation: f64 =
			out.iter().enumerate().map(|(i, &v)| v * input(i) * 32768.0).sum::<f64>()
				/ (0..200_000).map(|i| (input(i) * 32768.0).powi(2)).sum::<f64>();
		assert!((correlation - 1.0).abs() < 0.1, "gain {correlation}");
	}

	#[test]
	fn lipshitz_moves_noise_out_of_the_midrange() {
		let n = 8192;
		let input = |i: usize| (i as f64 * 0.0137).sin() * 0.3;
		let flat = error_signal(DitherMethod::Triangular, input, n);
		let shaped = error_signal(DitherMethod::Lipshitz, input, n);
		let mid = |e: &[f64]| band_power(e, 44_100.0, 2_000.0, 5_000.0);
		let high = |e: &[f64]| band_power(e, 44_100.0, 16_000.0, 21_000.0);
		assert!(mid(&shaped) < mid(&flat) / 4.0, "midrange noise at least 6 dB lower");
		assert!(high(&shaped) > high(&flat) * 4.0, "and pushed to the top of the band");
	}

	#[test]
	fn saturates_and_stays_stable_when_clipping() {
		let mut q = Quantizer::new(Dither::new(DitherMethod::Lipshitz), 16, 2, 48_000).unwrap();
		for i in 0..10_000 {
			let x = if i % 2 == 0 { 1.5 } else { -1.5 };
			let v = q.quantize(i % 2, x);
			assert!((-32768..=32767).contains(&v));
		}
		// Back to silence, the output settles to within a few LSB.
		let tail: Vec<i64> = (0..1000).map(|_| q.quantize(0, 0.0)).collect();
		assert!(tail[500..].iter().all(|v| v.abs() <= 8), "{:?}", &tail[500..520]);
		assert!(q.quantize(0, f64::NAN).abs() <= 8);
	}

	#[test]
	fn deterministic_and_checked() {
		let run = || {
			let mut q = Quantizer::new(Dither::TRIANGULAR, 16, 2, 44_100).unwrap();
			(0..1000).map(|i| q.quantize(i % 2, 0.1)).collect::<Vec<_>>()
		};
		assert_eq!(run(), run());
		assert!(Quantizer::new(Dither::new(DitherMethod::Lipshitz), 16, 1, 96_000).is_err());
		assert!(
			Quantizer::new(Dither { method: DitherMethod::Triangular, scale: f64::NAN }, 16, 1, 8000)
				.is_err()
		);
		assert!(Quantizer::new(Dither::TRIANGULAR, 1, 1, 8000).is_err());
	}
}
