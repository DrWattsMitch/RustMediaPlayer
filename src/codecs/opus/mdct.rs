//! CELT inverse MDCT (libopus `celt/mdct.c`, `clt_mdct_backward`) on top of
//! a mixed-radix complex FFT.
//!
//! CELT's MDCT sizes are 1920 >> shift (shift 0-3), so the FFTs are 480,
//! 240, 120 and 60 points: products of 2, 3, 4 and 5. The transform is
//! computed the way libopus computes it — pre-rotation, an N/4-point
//! forward FFT, post-rotation, then the TDAC mirror with the window — so the
//! output equals libopus's up to floating-point rounding.

use std::f64::consts::PI;

use crate::simd::F32x4;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cpx {
	pub re: f32,
	pub im: f32,
}

impl Cpx {
	#[inline]
	fn mul(self, o: Cpx) -> Cpx {
		Cpx { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
	}
	#[inline]
	fn add(self, o: Cpx) -> Cpx {
		Cpx { re: self.re + o.re, im: self.im + o.im }
	}
}

/// Forward complex FFT of one size, `X[k] = Σ x[n]·e^(-2πi·nk/N)`, unscaled.
#[derive(Debug, Clone)]
pub struct Fft {
	n: usize,
	factors: Vec<usize>,
	twiddles: Vec<Cpx>,
}

impl Fft {
	pub fn new(n: usize) -> Self {
		let mut factors = Vec::new();
		let mut m = n;
		for p in [4usize, 2, 3, 5] {
			while m.is_multiple_of(p) && m > 1 {
				factors.push(p);
				m /= p;
			}
		}
		assert!(m == 1, "fft size must factor into 2, 3, 5");
		let twiddles = (0..n)
			.map(|i| {
				let phase = -2.0 * PI * i as f64 / n as f64;
				Cpx { re: phase.cos() as f32, im: phase.sin() as f32 }
			})
			.collect();
		Self { n, factors, twiddles }
	}

	/// Transforms `data` (length `n`) in place.
	pub fn process(&self, data: &mut [Cpx]) {
		self.process_with(data, crate::simd::enabled());
	}

	/// [`Fft::process`] with the vector loop on or off (tests compare the
	/// two).
	pub(crate) fn process_with(&self, data: &mut [Cpx], vector: bool) {
		let input = data.to_vec();
		self.recurse(&input, 1, data, 0, vector);
	}

	/// Decimation in time: `out` receives the DFT of `x[0], x[stride],
	/// x[2·stride], ...` (`out.len()` points).
	fn recurse(&self, x: &[Cpx], stride: usize, out: &mut [Cpx], level: usize, vector: bool) {
		let len = out.len();
		if len == 1 {
			out[0] = x[0];
			return;
		}
		let p = self.factors[level];
		let m = len / p;
		for r in 0..p {
			self.recurse(&x[r * stride..], stride * p, &mut out[r * m..(r + 1) * m], level + 1, vector);
		}
		// W_len^e is twiddles[e · n / len]; r·k < len, so no wrap is needed.
		let step = self.n / len;
		let w = |e: usize| self.twiddles[e * step];
		// The combining twiddles W_len^(r·q·m) depend only on (r, q).
		let mut wq = [[Cpx::default(); 5]; 5];
		for (r, row) in wq[..p].iter_mut().enumerate() {
			for (q, v) in row[..p].iter_mut().enumerate() {
				*v = w((r * q * m) % len);
			}
		}
		let mut k = 0;
		if m >= 4 && vector {
			k = butterflies(out, p, m, step, &self.twiddles, &wq);
		}
		let mut terms = [Cpx::default(); 5];
		for k in k..m {
			for (r, t) in terms[..p].iter_mut().enumerate() {
				*t = out[r * m + k].mul(w(r * k));
			}
			for q in 0..p {
				let mut acc = terms[0];
				for (r, t) in terms[..p].iter().enumerate().skip(1) {
					acc = acc.add(t.mul(wq[r][q]));
				}
				out[q * m + k] = acc;
			}
		}
	}
}

/// Four lanes of complex values as (real parts, imaginary parts).
type Cv = (F32x4, F32x4);

/// `Cpx::mul` lane-wise.
#[inline(always)]
fn cmul(a: Cv, b: Cv) -> Cv {
	(a.0.mul(b.0).sub(a.1.mul(b.1)), a.0.mul(b.1).add(a.1.mul(b.0)))
}

#[inline(always)]
fn gather(c: [Cpx; 4]) -> Cv {
	(F32x4::from_array(c.map(|v| v.re)), F32x4::from_array(c.map(|v| v.im)))
}

crate::multiversion! {
	/// The combining loop of [`Fft::recurse`] for `k` in steps of four, with
	/// each lane doing exactly the scalar operations. Returns the first `k`
	/// left for the scalar loop.
	fn butterflies(out: &mut [Cpx], p: usize, m: usize, step: usize, tw: &[Cpx], wq: &[[Cpx; 5]; 5]) -> usize {
		let mut k = 0;
		while k + 4 <= m {
			let mut terms = [(F32x4::splat(0.0), F32x4::splat(0.0)); 5];
			for (r, t) in terms[..p].iter_mut().enumerate() {
				let x = gather(std::array::from_fn(|j| out[r * m + k + j]));
				let w = gather(std::array::from_fn(|j| tw[r * (k + j) * step]));
				*t = cmul(x, w);
			}
			for q in 0..p {
				let mut acc = terms[0];
				for r in 1..p {
					let w = (F32x4::splat(wq[r][q].re), F32x4::splat(wq[r][q].im));
					let v = cmul(terms[r], w);
					acc = (acc.0.add(v.0), acc.1.add(v.1));
				}
				let (re, im) = (acc.0.to_array(), acc.1.to_array());
				for j in 0..4 {
					out[q * m + k + j] = Cpx { re: re[j], im: im[j] };
				}
			}
			k += 4;
		}
		k
	}
}

/// The inverse MDCT for CELT's 48 kHz / 960-sample mode.
#[derive(Debug)]
pub struct Mdct {
	/// Largest transform size (1920).
	n: usize,
	/// One FFT per shift (N/4 points each).
	ffts: Vec<Fft>,
	/// `cos(2π(i + 1/8) / N)` for `i < N/2`, concatenated for each shift.
	trig: Vec<f32>,
}

impl Mdct {
	pub fn new(n: usize, max_shift: usize) -> Self {
		let mut trig = Vec::new();
		let mut ffts = Vec::new();
		for shift in 0..=max_shift {
			let size = n >> shift;
			for i in 0..size / 2 {
				trig.push((2.0 * PI * (i as f64 + 0.125) / size as f64).cos() as f32);
			}
			ffts.push(Fft::new(size / 4));
		}
		Self { n, ffts, trig }
	}

	/// The shared instance for CELT (N = 1920, shifts 0-3).
	pub fn celt() -> &'static Mdct {
		static M: OnceLock<Mdct> = OnceLock::new();
		M.get_or_init(|| Mdct::new(1920, 3))
	}

	/// `clt_mdct_backward`: transforms `input[0], input[stride], ...` (N/2
	/// coefficients) into `out`, overlap-adding with the previous block
	/// already present at the start of `out` through the TDAC mirror.
	/// `out` must hold at least `overlap / 2 + N / 2` samples... plus the
	/// overlap, i.e. `N / 2 + overlap`.
	pub fn backward(
		&self,
		input: &[f32],
		out: &mut [f32],
		window: &[f32],
		overlap: usize,
		shift: usize,
		stride: usize,
	) {
		let mut offset = 0;
		let mut n = self.n;
		for _ in 0..shift {
			n >>= 1;
			offset += n;
		}
		let trig = &self.trig[offset..offset + n / 2];
		let n2 = n / 2;
		let n4 = n / 4;

		// Pre-rotation (real and imaginary swapped: an FFT stands in for an IFFT).
		let mut buf = vec![Cpx::default(); n4];
		for (i, slot) in buf.iter_mut().enumerate() {
			let x1 = input[2 * stride * i];
			let x2 = input[stride * (n2 - 1 - 2 * i)];
			let yr = x2 * trig[i] + x1 * trig[n4 + i];
			let yi = x1 * trig[i] - x2 * trig[n4 + i];
			*slot = Cpx { re: yi, im: yr };
		}
		self.ffts[shift].process(&mut buf);

		// Post-rotation, de-shuffling from both ends.
		let base = overlap / 2;
		{
			let yp = &mut out[base..base + n2];
			for (i, c) in buf.iter().enumerate() {
				yp[2 * i] = c.re;
				yp[2 * i + 1] = c.im;
			}
			let mut i0 = 0usize;
			let mut i1 = n2 - 2;
			for i in 0..n4.div_ceil(2) {
				let re = yp[i0 + 1];
				let im = yp[i0];
				let (t0, t1) = (trig[i], trig[n4 + i]);
				let yr = re * t0 + im * t1;
				let yi = re * t1 - im * t0;
				let re2 = yp[i1 + 1];
				let im2 = yp[i1];
				yp[i0] = yr;
				yp[i1 + 1] = yi;
				let (t0, t1) = (trig[n4 - i - 1], trig[n2 - i - 1]);
				let yr = re2 * t0 + im2 * t1;
				let yi = re2 * t1 - im2 * t0;
				yp[i1] = yr;
				yp[i0 + 1] = yi;
				i0 += 2;
				i1 = i1.wrapping_sub(2);
			}
		}

		// Mirror on both sides for TDAC.
		for i in 0..overlap / 2 {
			let x1 = out[overlap - 1 - i];
			let x2 = out[i];
			let (w1, w2) = (window[i], window[overlap - 1 - i]);
			out[i] = w2 * x2 - w1 * x1;
			out[overlap - 1 - i] = w1 * x2 + w2 * x1;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn vector_fft_is_bit_identical_to_scalar() {
		for n in [60usize, 120, 240, 480, 8, 30] {
			let fft = Fft::new(n);
			let input: Vec<Cpx> = (0..n)
				.map(|i| Cpx {
					re: ((i * 7919) % 1013) as f32 / 7.0 - 70.0,
					im: ((i * 31) % 97) as f32 / 3.0,
				})
				.collect();
			let mut a = input.clone();
			let mut b = input;
			fft.process_with(&mut a, false);
			fft.process_with(&mut b, true);
			let bits =
				|v: &[Cpx]| v.iter().flat_map(|c| [c.re.to_bits(), c.im.to_bits()]).collect::<Vec<_>>();
			assert_eq!(bits(&a), bits(&b), "n={n}");
		}
	}

	#[test]
	#[allow(clippy::needless_range_loop)]
	fn fft_matches_a_direct_dft() {
		for n in [60usize, 120, 240, 480, 8, 30] {
			let input: Vec<Cpx> =
				(0..n).map(|i| Cpx { re: ((i * 7) % 13) as f32 - 6.0, im: ((i * 3) % 5) as f32 }).collect();
			let mut fast = input.clone();
			Fft::new(n).process(&mut fast);
			for k in 0..n {
				let (mut re, mut im) = (0f64, 0f64);
				for (j, x) in input.iter().enumerate() {
					let ph = -2.0 * PI * (j * k) as f64 / n as f64;
					re += f64::from(x.re) * ph.cos() - f64::from(x.im) * ph.sin();
					im += f64::from(x.re) * ph.sin() + f64::from(x.im) * ph.cos();
				}
				assert!(
					(f64::from(fast[k].re) - re).abs() < 1e-2 && (f64::from(fast[k].im) - im).abs() < 1e-2,
					"n={n} k={k}"
				);
			}
		}
	}
}
