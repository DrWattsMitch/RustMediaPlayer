//! Inverse transforms (rav1d's `inv_txfm_add` on the vendored 1-D kernels,
//! 8-bit, square sizes) and forward transforms.
//!
//! As for VP9, only the inverse is normative. The forward transform is the
//! exact inverse of the decoder's: each 1-D kernel is measured into a
//! matrix once and inverted, and the 2-D forward transform undoes the
//! decoder's row pass, intermediate rounding shift and final `>> 4`. 64-point
//! transforms keep their 32 lowest frequencies, which is all AV1 codes.

use std::num::NonZeroUsize;
use std::sync::LazyLock;

use super::itx_1d::*;

/// Transform types used here (`TxfmType`).
pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;
pub const IDTX: u8 = 9;

type Itx1d = fn(&mut [i32], NonZeroUsize, i32, i32);

fn kernel(n: usize, adst: bool, identity: bool) -> Itx1d {
	match (n, adst, identity) {
		(4, _, true) => rav1d_inv_identity4_1d_c,
		(8, _, true) => rav1d_inv_identity8_1d_c,
		(16, _, true) => rav1d_inv_identity16_1d_c,
		(32, _, true) => rav1d_inv_identity32_1d_c,
		(4, true, _) => rav1d_inv_adst4_1d_c,
		(8, true, _) => rav1d_inv_adst8_1d_c,
		(16, true, _) => rav1d_inv_adst16_1d_c,
		(4, _, _) => rav1d_inv_dct4_1d_c,
		(8, _, _) => rav1d_inv_dct8_1d_c,
		(16, _, _) => rav1d_inv_dct16_1d_c,
		(32, _, _) => rav1d_inv_dct32_1d_c,
		_ => rav1d_inv_dct64_1d_c,
	}
}

/// (vertical, horizontal) kernels of a transform type: (is ADST, is identity).
fn kinds(txtp: u8) -> ((bool, bool), (bool, bool)) {
	match txtp {
		ADST_DCT => ((true, false), (false, false)),
		DCT_ADST => ((false, false), (true, false)),
		ADST_ADST => ((true, false), (true, false)),
		IDTX => ((false, true), (false, true)),
		_ => ((false, false), (false, false)),
	}
}

/// Intermediate rounding shift of a square size (`log2` of the side in 4
/// units: 0 = 4x4 ... 4 = 64x64).
const SHIFT: [u32; 5] = [0, 1, 2, 2, 2];

/// Adds the inverse transform of `coef` (rav1d's layout: column-major over
/// the coded `min(n, 32)` square, `coef[x * h + y]`) to the `n x n` block at
/// `dst`. `eob` is the last coded scan position; the coefficients are
/// cleared. 8-bit samples.
pub fn inverse_add(
	coef: &mut [i32],
	dst: &mut [u8],
	stride: usize,
	lw: usize,
	txtp: u8,
	eob: usize,
) {
	let n = 4usize << lw;
	let shift = SHIFT[lw];
	let rnd = (1i32 << shift) >> 1;
	if eob == 0 && txtp == DCT_DCT {
		// DC only: rav1d's shortcut (same result as the full transform).
		let mut dc = coef[0];
		coef[0] = 0;
		dc = (dc * 181 + 128) >> 8;
		dc = (dc + rnd) >> shift;
		dc = (dc * 181 + 128 + 2048) >> 12;
		for y in 0..n {
			for p in &mut dst[y * stride..y * stride + n] {
				*p = (i32::from(*p) + dc).clamp(0, 255) as u8;
			}
		}
		return;
	}
	let s = n.min(32);
	let (min, max) = (i32::from(i16::MIN), i32::from(i16::MAX));
	let ((v_adst, v_id), (h_adst, h_id)) = kinds(txtp);
	let first = kernel(n, h_adst, h_id);
	let second = kernel(n, v_adst, v_id);
	let mut tmp = vec![0i32; n * n];
	for y in 0..s {
		let row = &mut tmp[y * n..(y + 1) * n];
		for x in 0..s {
			row[x] = coef[y + x * s];
		}
		first(row, NonZeroUsize::MIN, min, max);
	}
	coef[..s * s].fill(0);
	for v in &mut tmp[..n * s] {
		*v = ((*v + rnd) >> shift).clamp(min, max);
	}
	let stride_nz = NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
	for x in 0..n {
		second(&mut tmp[x..], stride_nz, min, max);
	}
	for y in 0..n {
		for x in 0..n {
			let p = &mut dst[y * stride + x];
			*p = (i32::from(*p) + ((tmp[y * n + x] + 8) >> 4)).clamp(0, 255) as u8;
		}
	}
}

/// The forward matrix of one 1-D kernel: `F = (MᵀM)⁻¹Mᵀ` (`k x n`,
/// row-major), the least-squares inverse of the kernel's `n x k` matrix `M`
/// over the `k = min(n, 32)` inputs it reads. For `k = n` this is `M⁻¹`.
fn measure(n: usize, adst: bool, identity: bool) -> Vec<f32> {
	const AMP: f64 = 8192.0;
	let k = n.min(32);
	let kern = kernel(n, adst, identity);
	let mut m = vec![0f64; n * k];
	for j in 0..k {
		let mut v = vec![0i32; n];
		v[j] = AMP as i32;
		kern(&mut v, NonZeroUsize::MIN, i32::MIN, i32::MAX);
		for i in 0..n {
			m[i * k + j] = f64::from(v[i]) / AMP;
		}
	}
	// G = MᵀM, then Gauss-Jordan on [G | Mᵀ].
	let mut g = vec![0f64; k * k];
	for a in 0..k {
		for b in 0..k {
			g[a * k + b] = (0..n).map(|i| m[i * k + a] * m[i * k + b]).sum();
		}
	}
	let mut f: Vec<f64> = (0..k * n).map(|idx| m[(idx % n) * k + idx / n]).collect();
	for c in 0..k {
		let p = (c..k).max_by(|&a, &b| g[a * k + c].abs().total_cmp(&g[b * k + c].abs())).unwrap_or(c);
		for j in 0..k {
			g.swap(c * k + j, p * k + j);
		}
		for j in 0..n {
			f.swap(c * n + j, p * n + j);
		}
		let d = g[c * k + c];
		for j in 0..k {
			g[c * k + j] /= d;
		}
		for j in 0..n {
			f[c * n + j] /= d;
		}
		for r in 0..k {
			if r != c {
				let x = g[r * k + c];
				if x != 0.0 {
					for j in 0..k {
						g[r * k + j] -= x * g[c * k + j];
					}
					for j in 0..n {
						f[r * n + j] -= x * f[c * n + j];
					}
				}
			}
		}
	}
	f.into_iter().map(|v| v as f32).collect()
}

/// Forward matrices `[lw][dct, adst, identity]`.
static FORWARD: LazyLock<Vec<[Vec<f32>; 3]>> = LazyLock::new(|| {
	(0..5)
		.map(|lw| {
			let n = 4usize << lw;
			[measure(n, false, false), measure(n, n <= 16, false), measure(n, false, n <= 32)]
		})
		.collect()
});

fn forward_matrix(lw: usize, (adst, identity): (bool, bool)) -> &'static [f32] {
	&FORWARD[lw][if identity { 2 } else { usize::from(adst) }]
}

/// Forward transform of the `n x n` residual `res` (row-major) into `out`
/// in rav1d's coefficient layout (`out[x * s + y]`, `s = min(n, 32)`),
/// at the scale the decoder's inverse expects.
pub fn forward(res: &[i32], out: &mut [f32], lw: usize, txtp: u8) {
	let n = 4usize << lw;
	let s = n.min(32);
	let (v, h) = kinds(txtp);
	let fh = forward_matrix(lw, h);
	let fv = forward_matrix(lw, v);
	// res = (Mv · (X · Mhᵀ / 2^shift)) / 16, so X = 16 · 2^shift · Fv · res · Fhᵀ.
	let scale = (16u32 << SHIFT[lw]) as f32;
	let mut t = vec![0f32; n * s];
	// t = res · Fhᵀ (rows), keeping the first s horizontal frequencies.
	for y in 0..n {
		let row = &res[y * n..(y + 1) * n];
		if row.iter().all(|&v| v == 0) {
			continue;
		}
		for k in 0..s {
			let f = &fh[k * n..(k + 1) * n];
			t[y * s + k] = row.iter().zip(f).map(|(&r, &c)| r as f32 * c).sum();
		}
	}
	for m in 0..s {
		let f = &fv[m * n..(m + 1) * n];
		for k in 0..s {
			let mut acc = 0f32;
			for y in 0..n {
				acc += f[y] * t[y * s + k];
			}
			out[k * s + m] = acc * scale;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn forward_inverts_the_decoder() {
		let mut seed = 3u32;
		let mut next = || {
			seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			(seed >> 24) as i32 - 128
		};
		for lw in 0..5 {
			let n = 4usize << lw;
			let s = n.min(32);
			let types: &[u8] =
				if n <= 16 { &[DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST, IDTX] } else { &[DCT_DCT] };
			for &txtp in types {
				// Smooth content (64-point transforms drop high frequencies).
				let res: Vec<i32> = (0..n * n)
					.map(|i| {
						let (y, x) = ((i / n) as i32, (i % n) as i32);
						if lw == 4 { (x - y) / 2 + 10 } else { next() / 2 }
					})
					.collect();
				let mut c = vec![0f32; s * s];
				forward(&res, &mut c, lw, txtp);
				let mut coef: Vec<i32> = c.iter().map(|v| v.round() as i32).collect();
				let mut dst = vec![128u8; n * n];
				inverse_add(&mut coef, &mut dst, n, lw, txtp, s * s - 1);
				let err =
					dst.iter().zip(&res).map(|(&d, &r)| (i32::from(d) - 128 - r).abs()).max().unwrap();
				assert!(err <= 2, "{n}x{n} type {txtp}: error {err}");
			}
		}
	}

	#[test]
	fn dc_shortcut_matches_the_full_transform() {
		for lw in 0..5 {
			let n = 4usize << lw;
			let s = n.min(32);
			for dc in [-3000, -7, 1, 100, 2500] {
				let mut a = vec![0i32; s * s];
				a[0] = dc;
				let mut b = a.clone();
				let mut da = vec![120u8; n * n];
				let mut db = da.clone();
				inverse_add(&mut a, &mut da, n, lw, DCT_DCT, 0);
				inverse_add(&mut b, &mut db, n, lw, DCT_DCT, 5);
				assert_eq!(da, db, "{n}x{n} dc {dc}");
			}
		}
	}
}
