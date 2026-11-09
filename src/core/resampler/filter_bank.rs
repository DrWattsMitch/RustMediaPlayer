//! Polyphase windowed-sinc filter bank.
//!
//! Phase `p` of `P` holds the taps that interpolate a value `p / P` of an
//! input sample past an integer position. The prototype is a sinc low-pass
//! at `cutoff × min(1, L / M)` of the input Nyquist frequency, shaped by a
//! Kaiser window (β = 9) and normalised to unity gain per phase, with the
//! same defaults as FFmpeg's `libswresample` (32 taps at the input rate,
//! cutoff 0.97).

/// Zeroth-order modified Bessel function of the first kind.
fn bessel_i0(x: f64) -> f64 {
	let (mut sum, mut term) = (1.0f64, 1.0f64);
	let half = x / 2.0;
	for k in 1..64 {
		term *= (half / k as f64) * (half / k as f64);
		sum += term;
		if term < sum * 1e-17 {
			break;
		}
	}
	sum
}

#[derive(Debug, Clone)]
pub struct FilterBank {
	/// `phase_count + 1` rows of `phase_length` taps (the extra row lets
	/// interpolation between the last phase and the next sample work).
	phases: Vec<f64>,
	phase_length: usize,
	phase_count: usize,
}

impl FilterBank {
	/// `factor` is `min(1, L / M)`; `taps_at_unity` the filter length when
	/// not downsampling; `phase_count` the number of fractional positions.
	pub fn new(
		factor: f64,
		cutoff: f64,
		taps_at_unity: usize,
		phase_count: usize,
		beta: f64,
	) -> Self {
		let fc = (factor * cutoff).clamp(1e-6, 1.0);
		let phase_length = ((taps_at_unity as f64 / factor).ceil() as usize).max(2).next_multiple_of(2);
		let half = phase_length as f64 / 2.0;
		let i0_beta = bessel_i0(beta);
		let mut phases = Vec::with_capacity((phase_count + 1) * phase_length);
		for p in 0..=phase_count {
			let frac = p as f64 / phase_count as f64;
			let start = phases.len();
			for k in 0..phase_length {
				// Distance from the interpolated position to tap k.
				let x = (k as f64 - (half - 1.0)) - frac;
				let sinc = if x == 0.0 {
					1.0
				} else {
					let a = std::f64::consts::PI * fc * x;
					a.sin() / a
				};
				let r = x / half;
				let window =
					if r.abs() >= 1.0 { 0.0 } else { bessel_i0(beta * (1.0 - r * r).sqrt()) / i0_beta };
				phases.push(sinc * window);
			}
			let sum: f64 = phases[start..].iter().sum();
			if sum.abs() > 1e-12 {
				for c in &mut phases[start..] {
					*c /= sum;
				}
			}
		}
		Self { phases, phase_length, phase_count }
	}

	pub fn phase_length(&self) -> usize {
		self.phase_length
	}

	pub fn phase_count(&self) -> usize {
		self.phase_count
	}

	/// Taps of phase `p` (`0..=phase_count`).
	pub fn phase(&self, p: usize) -> &[f64] {
		&self.phases[p * self.phase_length..(p + 1) * self.phase_length]
	}
}
