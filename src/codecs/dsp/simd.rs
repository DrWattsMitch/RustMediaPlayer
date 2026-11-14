//! Vector versions of the shared transform loops, written with the
//! portable types of [`crate::simd`]. Every lane performs exactly the
//! scalar code's operations in the same order (no fused multiply-add), so
//! the results are bit-identical to the scalar path.

use super::fft::Complex;
use crate::multiversion;
use crate::simd::F32x4;

/// Four complex values as (real parts, imaginary parts).
#[inline(always)]
fn load4(c: &[Complex]) -> (F32x4, F32x4) {
	let a = F32x4::from_array([c[0].re, c[0].im, c[1].re, c[1].im]);
	let b = F32x4::from_array([c[2].re, c[2].im, c[3].re, c[3].im]);
	a.unzip(b)
}

#[inline(always)]
fn store4(c: &mut [Complex], re: F32x4, im: F32x4) {
	let a = re.zip_lo(im).to_array();
	let b = re.zip_hi(im).to_array();
	c[..4].copy_from_slice(&[
		Complex::new(a[0], a[1]),
		Complex::new(a[2], a[3]),
		Complex::new(b[0], b[1]),
		Complex::new(b[2], b[3]),
	]);
}

multiversion! {
	/// One radix-2 stage of [`super::Fft::forward`] with `size / 2 >= 4`
	/// butterflies per group, four at a time. `tw` holds the stage's
	/// twiddle real parts then imaginary parts.
	pub fn fft_stage(data: &mut [Complex], size: usize, tw: &[f32]) {
		let half = size / 2;
		let (w_re, w_im) = tw.split_at(half);
		for group in data.chunks_exact_mut(size) {
			let (lo, hi) = group.split_at_mut(half);
			for k in (0..half).step_by(4) {
				let (h_re, h_im) = load4(&hi[k..]);
				let (u_re, u_im) = load4(&lo[k..]);
				let (wr, wi) = (F32x4::load(&w_re[k..]), F32x4::load(&w_im[k..]));
				// `Complex::times`: (a.re b.re - a.im b.im, a.re b.im + a.im b.re).
				let t_re = h_re.mul(wr).sub(h_im.mul(wi));
				let t_im = h_re.mul(wi).add(h_im.mul(wr));
				store4(&mut lo[k..], u_re.add(t_re), u_im.add(t_im));
				store4(&mut hi[k..], u_re.sub(t_re), u_im.sub(t_im));
			}
		}
	}
}

multiversion! {
	/// The DCT-IV pre-twiddle of [`super::Imdct`]: `buf[k] =
	/// (x[2k], x[l - 1 - 2k]) * pre[k]`, four `k` at a time.
	pub fn dct4_pre(input: &[f32], pre: &[Complex], buf: &mut [Complex]) {
		let l = input.len();
		for k in (0..buf.len()).step_by(4) {
			// Even samples from the front, odd ones from the back (reversed).
			let (z_re, _) = F32x4::load(&input[2 * k..]).unzip(F32x4::load(&input[2 * k + 4..]));
			let back = l - 8 - 2 * k;
			let (_, z_im) = F32x4::load(&input[back..]).unzip(F32x4::load(&input[back + 4..]));
			let z_im = z_im.reverse();
			let (p_re, p_im) = load4(&pre[k..]);
			store4(&mut buf[k..], z_re.mul(p_re).sub(z_im.mul(p_im)), z_re.mul(p_im).add(z_im.mul(p_re)));
		}
	}
}

multiversion! {
	/// The DCT-IV post-twiddle of [`super::Imdct`]: `c = buf[k] *
	/// post[k]`, `out[2k] = c.re * scale`, `out[l - 1 - 2k] = -c.im * scale`.
	pub fn dct4_post(buf: &[Complex], post: &[Complex], out: &mut [f32], scale: f32) {
		let l = out.len();
		let s = F32x4::splat(scale);
		for k in (0..buf.len()).step_by(4) {
			let (b_re, b_im) = load4(&buf[k..]);
			let (p_re, p_im) = load4(&post[k..]);
			let c_re = b_re.mul(p_re).sub(b_im.mul(p_im));
			let c_im = b_re.mul(p_im).add(b_im.mul(p_re));
			let even = c_re.mul(s).to_array();
			let odd = c_im.neg().mul(s).to_array();
			for j in 0..4 {
				out[2 * (k + j)] = even[j];
				out[l - 1 - 2 * (k + j)] = odd[j];
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::super::fft::{Complex, Fft};
	use super::super::mdct::Imdct;

	#[test]
	fn vector_dct4_is_bit_identical_to_scalar() {
		let mut seed = 0x99u64;
		let mut next = || {
			seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			((seed >> 40) as i32 - (1 << 23)) as f32 / 4096.0
		};
		for n in [64usize, 128, 256, 512, 2048, 4096] {
			let imdct = Imdct::new(n, 2.0 / n as f32);
			for _ in 0..20 {
				let input: Vec<f32> = (0..n / 2).map(|_| next()).collect();
				let mut a = vec![0f32; n / 2];
				let mut b = vec![0f32; n / 2];
				let mut buf = vec![Complex::default(); n / 4];
				imdct.dct4_with(&input, &mut a, &mut buf, false);
				imdct.dct4_with(&input, &mut b, &mut buf, true);
				assert_eq!(
					a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
					b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
					"n={n}"
				);
			}
		}
	}

	#[test]
	fn vector_fft_is_bit_identical_to_scalar() {
		let mut seed = 0x77u64;
		let mut next = || {
			seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			((seed >> 40) as i32 - (1 << 23)) as f32 / 4096.0
		};
		for n in [8usize, 16, 32, 64, 128, 256, 512, 1024] {
			let fft = Fft::new(n);
			for _ in 0..50 {
				let input: Vec<Complex> = (0..n).map(|_| Complex::new(next(), next())).collect();
				let mut a = input.clone();
				let mut b = input;
				fft.forward_with(&mut a, false);
				fft.forward_with(&mut b, true);
				let bits = |v: &[Complex]| {
					v.iter().flat_map(|c| [c.re.to_bits(), c.im.to_bits()]).collect::<Vec<_>>()
				};
				assert_eq!(bits(&a), bits(&b), "n={n}");
			}
		}
	}
}
