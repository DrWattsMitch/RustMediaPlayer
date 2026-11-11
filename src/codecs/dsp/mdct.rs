//! Inverse MDCT via a half-length DCT-IV computed with a quarter-length
//! complex FFT (the factorisation behind FFmpeg's `ff_imdct_calc` / `av_tx`).
//!
//! For `N` outputs and `N/2` inputs:
//!
//! ```text
//! y[n] = Σ_k X[k] · cos(2π/N · (n + N/4 + 1/2) · (k + 1/2))
//! ```
//!
//! which equals a DCT-IV `u` of size `L = N/2` mapped as
//! `y[n] = u[n + N/4]` (first quarter), `-u[3N/4 - 1 - n]` (middle half) and
//! `-u[n - 3N/4]` (last quarter). The DCT-IV uses one `L/2`-point FFT with a
//! pre- and post-twiddle. Cost: O(N log N), no allocation per call.

use super::fft::{Complex, Fft};

#[derive(Debug, Clone)]
pub struct Imdct {
	/// Number of output samples (`N`).
	n: usize,
	fft: Fft,
	pre: Vec<Complex>,
	post: Vec<Complex>,
	scale: f32,
}

impl Imdct {
	/// Creates an IMDCT producing `n` samples (a power of two ≥ 16), with
	/// every output multiplied by `scale`.
	pub fn new(n: usize, scale: f32) -> Self {
		let l = n / 2;
		let quarter = l / 2;
		let pi = std::f64::consts::PI;
		let pre = (0..quarter)
			.map(|k| {
				let a = -pi * (4 * k + 1) as f64 / (4 * l) as f64;
				Complex::new(a.cos() as f32, a.sin() as f32)
			})
			.collect();
		let post = (0..quarter)
			.map(|k| {
				let a = -pi * k as f64 / l as f64;
				Complex::new(a.cos() as f32, a.sin() as f32)
			})
			.collect();
		Self { n, fft: Fft::new(quarter), pre, post, scale }
	}

	pub fn len(&self) -> usize {
		self.n
	}

	pub fn is_empty(&self) -> bool {
		self.n == 0
	}

	/// DCT-IV of `input` (length `N/2`) into `out` (length `N/2`), scaled.
	/// `buf` is scratch of length `N/4`.
	fn dct4(&self, input: &[f32], out: &mut [f32], buf: &mut [Complex]) {
		let l = self.n / 2;
		let quarter = l / 2;
		for (k, slot) in buf.iter_mut().enumerate().take(quarter) {
			let z = Complex::new(input[2 * k], input[l - 1 - 2 * k]);
			*slot = z.times(self.pre[k]);
		}
		self.fft.forward(&mut buf[..quarter]);
		for k in 0..quarter {
			let c = buf[k].times(self.post[k]);
			out[2 * k] = c.re * self.scale;
			out[l - 1 - 2 * k] = -c.im * self.scale;
		}
	}

	/// Full IMDCT: `input` has `N/2` coefficients, `output` receives `N`
	/// samples. `scratch` must hold at least `N/2` floats and `buf` `N/4`
	/// complex values.
	pub fn inverse(
		&self,
		input: &[f32],
		output: &mut [f32],
		scratch: &mut [f32],
		buf: &mut [Complex],
	) {
		let n = self.n;
		let (q, h, tq) = (n / 4, n / 2, 3 * n / 4);
		let u = &mut scratch[..h];
		self.dct4(&input[..h], u, buf);
		output[..q].copy_from_slice(&u[q..h]);
		for i in q..tq {
			output[i] = -u[tq - 1 - i];
		}
		for i in tq..n {
			output[i] = -u[i - tq];
		}
	}
}

/// Forward MDCT: `N` windowed samples to `N/2` coefficients,
///
/// ```text
/// X[k] = scale · Σ_n x[n] · cos(2π/N · (n + N/4 + 1/2) · (k + 1/2))
/// ```
///
/// computed by folding the input into a half-length DCT-IV (the inverse of
/// the mapping used by [`Imdct`]) and reusing its quarter-length FFT.
/// FFmpeg: `ff_mdct_calc` / `av_tx` (`AV_TX_FLOAT_MDCT`).
#[derive(Debug, Clone)]
pub struct Mdct {
	inner: Imdct,
}

impl Mdct {
	/// Creates an MDCT taking `n` samples (a power of two ≥ 16).
	pub fn new(n: usize, scale: f32) -> Self {
		Self { inner: Imdct::new(n, scale) }
	}

	pub fn len(&self) -> usize {
		self.inner.n
	}

	pub fn is_empty(&self) -> bool {
		self.inner.n == 0
	}

	/// `input` has `N` samples; `output` receives `N/2` coefficients.
	/// `scratch` must hold at least `N/2` floats and `buf` `N/4` complex
	/// values.
	pub fn forward(
		&self,
		input: &[f32],
		output: &mut [f32],
		scratch: &mut [f32],
		buf: &mut [Complex],
	) {
		let n = self.inner.n;
		let (q, h, tq) = (n / 4, n / 2, 3 * n / 4);
		let u = &mut scratch[..h];
		// Time-domain aliasing fold: with x = [a b c d] (quarters),
		// u = [-c_r - d, a - b_r].
		for i in 0..q {
			u[i] = -input[tq - 1 - i] - input[tq + i];
			u[q + i] = input[i] - input[h - 1 - i];
		}
		self.inner.dct4(u, &mut output[..h], buf);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn forward_matches_direct_formula() {
		for n in [16usize, 64, 256, 2048] {
			let mdct = Mdct::new(n, 1.0);
			let input: Vec<f32> = (0..n).map(|i| ((i * 7919) % 31) as f32 - 15.0).collect();
			let mut out = vec![0.0; n / 2];
			let mut scratch = vec![0.0; n / 2];
			let mut buf = vec![Complex::default(); n / 4];
			mdct.forward(&input, &mut out, &mut scratch, &mut buf);
			for (k, &y) in out.iter().enumerate() {
				let mut want = 0.0f64;
				for (i, &x) in input.iter().enumerate() {
					let arg = 2.0 * std::f64::consts::PI / n as f64
						* (i as f64 + n as f64 / 4.0 + 0.5)
						* (k as f64 + 0.5);
					want += f64::from(x) * arg.cos();
				}
				assert!((f64::from(y) - want).abs() < 2e-3 * (n as f64), "n={n} k={k} {y} {want}");
			}
		}
	}

	#[test]
	fn matches_direct_formula() {
		for n in [16usize, 64, 256] {
			let imdct = Imdct::new(n, 1.0);
			let input: Vec<f32> = (0..n / 2).map(|k| ((k * 7919) % 23) as f32 - 11.0).collect();
			let mut out = vec![0.0; n];
			let mut scratch = vec![0.0; n / 2];
			let mut buf = vec![Complex::default(); n / 4];
			imdct.inverse(&input, &mut out, &mut scratch, &mut buf);
			for (i, &y) in out.iter().enumerate() {
				let mut want = 0.0f64;
				for (k, &x) in input.iter().enumerate() {
					let arg = 2.0 * std::f64::consts::PI / n as f64
						* (i as f64 + n as f64 / 4.0 + 0.5)
						* (k as f64 + 0.5);
					want += f64::from(x) * arg.cos();
				}
				assert!((f64::from(y) - want).abs() < 1e-3 * (n as f64), "n={n} i={i} {y} {want}");
			}
		}
	}
}
