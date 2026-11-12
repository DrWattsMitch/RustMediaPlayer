//! In-place iterative radix-2 complex FFT with precomputed twiddles and
//! bit-reversal permutation.

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct Complex {
	pub re: f32,
	pub im: f32,
}

impl Complex {
	#[inline]
	pub const fn new(re: f32, im: f32) -> Self {
		Self { re, im }
	}

	#[inline]
	pub fn times(self, o: Complex) -> Complex {
		Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
	}
}

#[derive(Debug, Clone)]
pub struct Fft {
	n: usize,
	/// `exp(-2πi·k/n)` for `k < n/2`.
	twiddles: Vec<Complex>,
	/// Bit-reversed index for every position.
	rev: Vec<u32>,
	/// The twiddles of each stage with `half >= 4`, contiguous: for a stage
	/// of `half` butterflies, `half` real parts then `half` imaginary parts
	/// (`twiddles[k * step]`), for the vector butterflies.
	stage_twiddles: Vec<f32>,
}

impl Fft {
	/// `n` must be a power of two (≥ 1).
	pub fn new(n: usize) -> Self {
		let n = n.max(1).next_power_of_two();
		let bits = n.trailing_zeros();
		let twiddles = (0..n / 2)
			.map(|k| {
				let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
				Complex::new(a.cos() as f32, a.sin() as f32)
			})
			.collect();
		let rev =
			(0..n as u32).map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (32 - bits) }).collect();
		let twiddles: Vec<Complex> = twiddles;
		let mut stage_twiddles = Vec::new();
		let mut size = 8;
		while size <= n {
			let (half, step) = (size / 2, n / size);
			stage_twiddles.extend((0..half).map(|k| twiddles[k * step].re));
			stage_twiddles.extend((0..half).map(|k| twiddles[k * step].im));
			size *= 2;
		}
		Self { n, twiddles, rev, stage_twiddles }
	}

	/// Forward transform `X[k] = Σ x[j]·exp(-2πi·jk/n)`. `data.len()` must
	/// equal the size passed to [`Fft::new`].
	pub fn forward(&self, data: &mut [Complex]) {
		self.forward_with(data, crate::simd::enabled());
	}

	/// [`Fft::forward`] with the vector stages on or off (tests compare
	/// the two).
	pub(crate) fn forward_with(&self, data: &mut [Complex], vector: bool) {
		let n = self.n;
		if data.len() != n {
			return;
		}
		for i in 0..n {
			let j = self.rev[i] as usize;
			if i < j {
				data.swap(i, j);
			}
		}
		let mut size = 2;
		let mut tw_at = 0;
		while size <= n {
			let half = size / 2;
			if vector && half >= 4 {
				let tw = &self.stage_twiddles[tw_at..tw_at + 2 * half];
				super::simd::fft_stage(data, size, tw);
				tw_at += 2 * half;
				size *= 2;
				continue;
			}
			let step = n / size;
			for start in (0..n).step_by(size) {
				let (lo, hi) = data[start..start + size].split_at_mut(half);
				for k in 0..half {
					let w = self.twiddles[k * step];
					let t = hi[k].times(w);
					let u = lo[k];
					lo[k] = Complex::new(u.re + t.re, u.im + t.im);
					hi[k] = Complex::new(u.re - t.re, u.im - t.im);
				}
			}
			size *= 2;
		}
	}
}
