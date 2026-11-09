//! Rational sample-rate conversion by `L / M` (upsample by `L`, low-pass,
//! downsample by `M`), computed directly in polyphase form.
//!
//! Output sample `n` sits at input position `n·M / L`. Its integer part
//! selects the input window and its fractional part the filter phase. When
//! `L` is at most [`MAX_EXACT_PHASES`] every phase is stored and the result
//! is exact; otherwise 1024 phases are stored and neighbouring phases are
//! interpolated linearly, as `libswresample` does (`phase_shift` 10,
//! `linear_interp`).
//!
//! Output is aligned with the input: output sample 0 is input sample 0, with
//! no filter delay, and a stream of `N` input samples yields
//! `ceil(N·L / M)` output samples once flushed — FFmpeg's `aresample`
//! behaviour.

use std::sync::Arc;

use super::delay_line::DelayLine;
use super::filter_bank::FilterBank;

/// Taps of the prototype filter when not downsampling (swresample's
/// `filter_size`).
pub const FILTER_SIZE: usize = 32;
/// Cutoff relative to the lower Nyquist frequency (swresample's `cutoff`).
pub const CUTOFF: f64 = 0.97;
/// Kaiser window β (swresample's `kaiser_beta`).
pub const KAISER_BETA: f64 = 9.0;
/// Largest `L` for which every phase is precomputed.
pub const MAX_EXACT_PHASES: usize = 1024;

fn gcd(mut a: u64, mut b: u64) -> u64 {
	while b != 0 {
		(a, b) = (b, a % b);
	}
	a
}

/// One channel of rate conversion. Channels of the same stream share their
/// [`FilterBank`] through [`Resampler::new_like`].
#[derive(Debug, Clone)]
pub struct Resampler {
	l: u64,
	m: u64,
	filter: Arc<FilterBank>,
	/// Whether phases are interpolated (`L > MAX_EXACT_PHASES`).
	interpolate: bool,
	delay: DelayLine,
	/// Next output index.
	produced: u64,
	/// Input samples received.
	consumed: u64,
}

impl Resampler {
	pub fn new(input_rate: u32, output_rate: u32) -> Option<Self> {
		if input_rate == 0 || output_rate == 0 {
			return None;
		}
		let g = gcd(u64::from(input_rate), u64::from(output_rate));
		let (l, m) = (u64::from(output_rate) / g, u64::from(input_rate) / g);
		let factor = (l as f64 / m as f64).min(1.0);
		let interpolate = l as usize > MAX_EXACT_PHASES;
		let phases = if interpolate { MAX_EXACT_PHASES } else { l as usize };
		let filter = Arc::new(FilterBank::new(factor, CUTOFF, FILTER_SIZE, phases, KAISER_BETA));
		let delay = DelayLine::new(filter.phase_length() / 2 - 1);
		Some(Self { l, m, filter, interpolate, delay, produced: 0, consumed: 0 })
	}

	/// A fresh resampler for another channel, sharing this one's filter.
	pub fn new_like(&self) -> Self {
		Self {
			delay: DelayLine::new(self.filter.phase_length() / 2 - 1),
			produced: 0,
			consumed: 0,
			filter: Arc::clone(&self.filter),
			..*self
		}
	}

	/// `(L, M)` after reduction.
	pub fn ratio(&self) -> (u64, u64) {
		(self.l, self.m)
	}

	/// Output samples a stream of `input` samples produces in total.
	pub fn output_len(&self, input: u64) -> u64 {
		((u128::from(input) * u128::from(self.l)).div_ceil(u128::from(self.m))) as u64
	}

	fn compute(&self, n: u64) -> Option<f64> {
		let half = (self.filter.phase_length() / 2) as i64;
		let pos = u128::from(n) * u128::from(self.m);
		let q = (pos / u128::from(self.l)) as i64;
		let rem = (pos % u128::from(self.l)) as u64;
		let window = self.delay.window(q - half + 1, self.filter.phase_length())?;
		let dot = |taps: &[f64]| taps.iter().zip(window).map(|(c, x)| c * x).sum::<f64>();
		if self.interpolate {
			let exact = rem as f64 * self.filter.phase_count() as f64 / self.l as f64;
			let p = exact.floor() as usize;
			let t = exact - p as f64;
			let a = dot(self.filter.phase(p));
			let b = dot(self.filter.phase(p + 1));
			Some(a + (b - a) * t)
		} else {
			Some(dot(self.filter.phase(rem as usize)))
		}
	}

	/// Produces every output sample whose input window is complete.
	fn drain(&mut self, limit: u64, out: &mut Vec<f64>) {
		let half = (self.filter.phase_length() / 2) as i64;
		while self.produced < limit {
			let q = (u128::from(self.produced) * u128::from(self.m) / u128::from(self.l)) as i64;
			if q + half > self.delay.end() {
				break;
			}
			match self.compute(self.produced) {
				Some(v) => out.push(v),
				None => break,
			}
			self.produced += 1;
		}
		let next_q = (u128::from(self.produced) * u128::from(self.m) / u128::from(self.l)) as i64;
		self.delay.discard(next_q - half + 1);
	}

	/// Feeds input samples; returns the output samples now available.
	pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
		if self.l == self.m {
			// Same rate: pass through (a low-pass would only colour the signal).
			self.consumed += input.len() as u64;
			self.produced += input.len() as u64;
			return input.to_vec();
		}
		self.delay.push(input);
		self.consumed += input.len() as u64;
		let mut out = Vec::with_capacity(input.len() * self.l as usize / self.m as usize + 2);
		self.drain(u64::MAX, &mut out);
		out
	}

	/// Ends the stream: pads with silence and returns the remaining output
	/// (the total is then exactly [`Resampler::output_len`] of the input).
	pub fn flush(&mut self) -> Vec<f64> {
		if self.l == self.m {
			return Vec::new();
		}
		let total = self.output_len(self.consumed);
		self.delay.push(&vec![0.0; self.filter.phase_length()]);
		let mut out = Vec::new();
		self.drain(total, &mut out);
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn sine(rate: u32, freq: f64, n: usize) -> Vec<f64> {
		(0..n)
			.map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin() * 0.5)
			.collect()
	}

	fn snr(a: &[f64], b: &[f64]) -> f64 {
		let signal: f64 = b.iter().map(|x| x * x).sum();
		let noise: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
		10.0 * (signal / noise.max(1e-30)).log10()
	}

	#[test]
	fn converts_sines_accurately() {
		for (from, to) in [
			(44_100, 48_000),
			(48_000, 44_100),
			(48_000, 8000),
			(8000, 48_000),
			(44_100, 96_000),
			(22_050, 44_100),
		] {
			let freq = 440.0;
			let n = from as usize; // one second
			let mut r = Resampler::new(from, to).unwrap();
			let mut out = Vec::new();
			for chunk in sine(from, freq, n).chunks(1000) {
				out.extend(r.process(chunk));
			}
			out.extend(r.flush());
			assert_eq!(out.len() as u64, r.output_len(n as u64), "{from}->{to}");
			let expected = sine(to, freq, out.len());
			// Ignore the edges, where the input starts and stops abruptly.
			let edge = 200;
			let s = snr(&out[edge..out.len() - edge], &expected[edge..out.len() - edge]);
			eprintln!("{from}->{to}: {s:.1} dB");
			assert!(s > 80.0, "{from}->{to}: {s:.1} dB");
		}
	}

	#[test]
	fn same_rate_is_identity_and_large_ratios_interpolate() {
		let mut r = Resampler::new(48_000, 48_000).unwrap();
		let input = sine(48_000, 1000.0, 2000);
		let mut out = r.process(&input);
		out.extend(r.flush());
		assert_eq!(out, input);
		let r = Resampler::new(44_100, 44_101).unwrap();
		assert!(r.interpolate);
		assert_eq!(r.ratio(), (44_101, 44_100));
	}
}
