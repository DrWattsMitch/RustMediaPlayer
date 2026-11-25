//! Forward transforms.
//!
//! Only the inverse transforms are normative: whatever the encoder puts in
//! a block, the decoder's butterflies define what it reconstructs to. So
//! rather than port libvpx's forward butterflies (`vpx_fdct*`, `fadst*`),
//! the forward transform is computed as the exact inverse of the decoder's
//! own 1-D transforms: each 1-D inverse is measured once (it is linear up
//! to rounding) into a matrix, which is inverted. A block's coefficients
//! are then `2^shift · Fc · R · Frᵀ`, which the decoder's
//! `(Mc · X · Mrᵀ + round) >> shift` maps back to the residual `R` up to
//! the quantisation. This matches the decoder's scaling for every size and
//! type (DCT, ADST, 32x32's halved dequantisation) by construction.

use std::sync::LazyLock;

use super::super::transform::inverse_1d;

/// `F = M⁻¹` of one 1-D inverse transform, row-major `n x n`.
struct Forward {
	n: usize,
	f: Vec<f32>,
}

fn measure(n: usize, adst: bool) -> Forward {
	// Columns of the inverse: the response to each unit coefficient, scaled
	// up so rounding inside the butterflies is negligible.
	const AMP: f64 = 65536.0;
	let mut m = vec![0f64; n * n];
	let mut input = vec![0i32; n];
	let mut output = vec![0i32; n];
	for k in 0..n {
		input.fill(0);
		input[k] = AMP as i32;
		inverse_1d(n, adst, &input, &mut output);
		for i in 0..n {
			m[i * n + k] = f64::from(output[i]) / AMP;
		}
	}
	// Gauss-Jordan inversion with partial pivoting.
	let mut a = m;
	let mut inv: Vec<f64> = (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect();
	for c in 0..n {
		let pivot =
			(c..n).max_by(|&x, &y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs())).unwrap_or(c);
		if pivot != c {
			for j in 0..n {
				a.swap(c * n + j, pivot * n + j);
				inv.swap(c * n + j, pivot * n + j);
			}
		}
		let d = a[c * n + c];
		for j in 0..n {
			a[c * n + j] /= d;
			inv[c * n + j] /= d;
		}
		for r in 0..n {
			if r != c {
				let f = a[r * n + c];
				if f != 0.0 {
					for j in 0..n {
						a[r * n + j] -= f * a[c * n + j];
						inv[r * n + j] -= f * inv[c * n + j];
					}
				}
			}
		}
	}
	Forward { n, f: inv.into_iter().map(|v| v as f32).collect() }
}

/// `[size][adst]`; 32x32 has no ADST.
static FORWARD: LazyLock<[[Forward; 2]; 4]> = LazyLock::new(|| {
	std::array::from_fn(|t| {
		let n = 4usize << t;
		[measure(n, false), measure(n, n < 32)]
	})
});

/// Rounding shift of each size's inverse 2-D transform.
const SHIFT: [u32; 4] = [4, 5, 6, 6];

/// Forward 2-D transform of the `n x n` residual `res` (row-major) into
/// `out` (raster order, the coefficient scale the decoder's inverse
/// expects). `tx_type` as the decoder's: bit 0 vertical ADST, bit 1
/// horizontal ADST.
pub fn forward(res: &[i32], out: &mut [f32], tx_size: u8, tx_type: u8) {
	let t = usize::from(tx_size);
	let n = 4usize << t;
	let fr = &FORWARD[t][usize::from(tx_type & 2 != 0 && n < 32)];
	let fc = &FORWARD[t][usize::from(tx_type & 1 != 0 && n < 32)];
	debug_assert!(fr.n == n && fc.n == n);
	let scale = (1u32 << SHIFT[t]) as f32;
	// T = R · Frᵀ (rows), then X = Fc · T (columns).
	let mut tmp = [0f32; 1024];
	for i in 0..n {
		let row = &res[i * n..(i + 1) * n];
		if row.iter().all(|&v| v == 0) {
			tmp[i * n..(i + 1) * n].fill(0.0);
			continue;
		}
		for k in 0..n {
			let f = &fr.f[k * n..(k + 1) * n];
			let mut s = 0f32;
			for j in 0..n {
				s += row[j] as f32 * f[j];
			}
			tmp[i * n + k] = s;
		}
	}
	for m in 0..n {
		let f = &fc.f[m * n..(m + 1) * n];
		for k in 0..n {
			let mut s = 0f32;
			for i in 0..n {
				s += f[i] * tmp[i * n + k];
			}
			out[m * n + k] = s * scale;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::super::super::transform::inverse_transform_add;
	use super::*;

	/// Forward then the decoder's inverse gives the residual back (up to the
	/// rounding of the coefficients), every size and type.
	#[test]
	fn forward_inverts_the_decoder() {
		let mut seed = 7u32;
		let mut next = || {
			seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			(seed >> 24) as i32 - 128
		};
		for tx_size in 0..4u8 {
			let n = 4usize << tx_size;
			for tx_type in 0..4u8 {
				let res: Vec<i32> = (0..n * n).map(|_| next() / 2).collect();
				let mut c = vec![0f32; n * n];
				forward(&res, &mut c, tx_size, tx_type);
				let coefs: Vec<i32> = c.iter().map(|v| v.round() as i32).collect();
				let mut dst = vec![128u8; n * n];
				inverse_transform_add(&coefs, &mut dst, n, tx_size, tx_type, false, n * n, 255);
				let err = dst.iter().zip(&res).map(|(&d, &r)| (i32::from(d) - 128 - r).abs()).max();
				assert!(err.unwrap_or(0) <= 1, "{n}x{n} type {tx_type}: error {err:?}");
			}
		}
	}
}
