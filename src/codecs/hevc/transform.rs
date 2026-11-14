//! Inverse transforms (ITU-T H.265 §8.6.4; FFmpeg `hevc/dsp_template.c`).
//!
//! Two-stage integer DCT (4 to 32 points) and the 4x4 DST, with the
//! intermediate clipping to 16 bits the standard requires. Coefficients
//! beyond the last significant row/column are skipped.

/// The 32x32 DCT basis (Equation 8-317); smaller transforms use every
/// `32 / N`-th row.
pub const DCT_MATRIX: [[i8; 32]; 32] = {
	const ROWS: [[i16; 16]; 32] = [
		[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
		[90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4],
		[90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90],
		[90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13],
		[89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
		[88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22],
		[87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87],
		[85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31],
		[83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
		[82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38],
		[80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80],
		[78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46],
		[75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
		[73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54],
		[70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70],
		[67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61],
		[64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
		[61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67],
		[57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57],
		[54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73],
		[50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
		[46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78],
		[43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43],
		[38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82],
		[36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
		[31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85],
		[25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25],
		[22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88],
		[18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
		[13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90],
		[9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9],
		[4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90],
	];
	// Rows are even (symmetric) or odd (antisymmetric) about the middle.
	let mut m = [[0i8; 32]; 32];
	let mut k = 0;
	while k < 32 {
		let mut i = 0;
		while i < 16 {
			let v = ROWS[k][i];
			m[k][i] = v as i8;
			m[k][31 - i] = if k % 2 == 0 { v as i8 } else { (-v) as i8 };
			i += 1;
		}
		k += 1;
	}
	m
};

const DST_MATRIX: [[i32; 4]; 4] =
	[[29, 55, 74, 84], [74, 74, 0, -74], [84, -29, -74, 55], [55, -84, 74, -29]];

#[inline]
fn clip16(v: i32) -> i32 {
	v.clamp(-32768, 32767)
}

/// One inverse 1-D DCT of size `n` over `src` (stride `src_stride`),
/// using only the first `nz` coefficients.
#[inline(always)]
fn idct_1d(src: &[i32], src_stride: usize, n: usize, nz: usize, shift: u32, out: &mut [i32; 32]) {
	let step = 32 / n;
	let add = 1i32 << (shift - 1);
	let half = n / 2;
	// Even/odd decomposition: E from even rows, O from odd rows.
	let mut e = [0i32; 16];
	let mut o = [0i32; 16];
	for k in (1..nz).step_by(2) {
		let c = src[k * src_stride];
		if c == 0 {
			continue;
		}
		let row = &DCT_MATRIX[k * step];
		for i in 0..half {
			o[i] += c * i32::from(row[i]);
		}
	}
	for k in (0..nz).step_by(2) {
		let c = src[k * src_stride];
		if c == 0 {
			continue;
		}
		let row = &DCT_MATRIX[k * step];
		for i in 0..half {
			e[i] += c * i32::from(row[i]);
		}
	}
	for i in 0..half {
		out[i] = clip16((e[i] + o[i] + add) >> shift);
		out[n - 1 - i] = clip16((e[i] - o[i] + add) >> shift);
	}
}

/// In-place inverse DCT of an `n`x`n` block of dequantised coefficients
/// into residuals. `max_x`/`max_y` bound the non-zero coefficients;
/// `tmp` and `tmp2` are scratch space of at least `n * n`.
#[inline]
pub fn idct(
	coeffs: &mut [i32],
	n: usize,
	max_x: usize,
	max_y: usize,
	bit_depth: u32,
	tmp: &mut [i32],
	tmp2: &mut [i32],
) {
	if (max_x | max_y) != 0 && crate::simd::enabled() {
		super::simd::idct(coeffs, n, max_x, max_y, bit_depth, tmp, tmp2);
	} else {
		idct_scalar(coeffs, n, max_x, max_y, bit_depth, tmp);
	}
}

crate::multiversion! {
/// Scalar [`idct`] (`tmp`: scratch space of at least `n * n`).
pub fn idct_scalar(coeffs: &mut [i32], n: usize, max_x: usize, max_y: usize, bit_depth: u32, tmp: &mut [i32]) {
	if max_x == 0 && max_y == 0 {
		idct_dc(coeffs, n, bit_depth);
		return;
	}
	let tmp = &mut tmp[..n * n];
	let mut col = [0i32; 32];
	let nz_rows = (max_y + 1).min(n);
	// Stage 1: columns (vertical), shift 7.
	for x in 0..=max_x.min(n - 1) {
		idct_1d(&coeffs[x..], n, n, nz_rows, 7, &mut col);
		for y in 0..n {
			tmp[y * n + x] = col[y];
		}
	}
	// Stage 2: rows, shift 20 - bitDepth.
	let shift = 20 - bit_depth;
	let nz_cols = (max_x + 1).min(n);
	for y in 0..n {
		idct_1d(&tmp[y * n..], 1, n, nz_cols, shift, &mut col);
		coeffs[y * n..y * n + n].copy_from_slice(&col[..n]);
	}
}
}

/// DC-only inverse transform.
pub fn idct_dc(coeffs: &mut [i32], n: usize, bit_depth: u32) {
	let shift = 20 - bit_depth;
	let v = clip16((coeffs[0] * 64 + 64) >> 7);
	let v = clip16((v * 64 + (1 << (shift - 1))) >> shift);
	coeffs[..n * n].fill(v);
}

/// 4x4 inverse DST (intra luma 4x4).
pub fn idst4(coeffs: &mut [i32], bit_depth: u32) {
	let mut tmp = [0i32; 16];
	for x in 0..4 {
		for y in 0..4 {
			let mut s = 0;
			for k in 0..4 {
				s += coeffs[k * 4 + x] * DST_MATRIX[k][y];
			}
			tmp[y * 4 + x] = clip16((s + 64) >> 7);
		}
	}
	let shift = 20 - bit_depth;
	let add = 1 << (shift - 1);
	for y in 0..4 {
		for x in 0..4 {
			let mut s = 0;
			for k in 0..4 {
				s += tmp[y * 4 + k] * DST_MATRIX[k][x];
			}
			coeffs[y * 4 + x] = clip16((s + add) >> shift);
		}
	}
}

/// Transform skip (version 1: 4x4 only): `(c << 7)` then the second-stage
/// rounding shift.
pub fn transform_skip(coeffs: &mut [i32], n: usize, bit_depth: u32) {
	let shift = 20 - bit_depth;
	let add = 1 << (shift - 1);
	for c in &mut coeffs[..n * n] {
		*c = (*c * 128 + add) >> shift;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn dc_matches_full_transform() {
		for n in [4, 8, 16, 32] {
			let mut a = vec![0i32; n * n];
			a[0] = 100;
			let mut b = a.clone();
			idct_scalar(&mut a, n, 0, 1, 8, &mut [0; 1024]); // full path (max_y forces it)
			idct_dc(&mut b, n, 8);
			assert_eq!(a, b, "n={n}");
		}
	}

	#[test]
	fn matrix_rows_are_orthogonal_enough() {
		// Rows 0 and 1 of the DCT basis are orthogonal.
		let dot: i32 = (0..32).map(|i| i32::from(DCT_MATRIX[0][i]) * i32::from(DCT_MATRIX[1][i])).sum();
		assert_eq!(dot, 0);
	}
}
