//! Inter prediction (spec §8.5.2; libvpx `dec_build_inter_predictors`,
//! `vpx_dsp/vpx_convolve.c`).
//!
//! An 8-tap separable filter at 1/16-pixel phase: a horizontal pass into a
//! temporary block (rounded and clipped to 8 bits), then a vertical pass.
//! A pass whose phase is 0 is a plain copy (its kernel is the unit tap), so
//! it is skipped. Reference pixels outside the picture repeat its edge
//! (libvpx's `build_mc_border`). Compound prediction averages the second
//! prediction with the first.

use super::tables::SUBPEL_FILTERS;

/// A reference plane: pixels and the visible size.
pub struct RefPlane<'a> {
	pub data: &'a [u8],
	pub stride: usize,
	pub width: usize,
	pub height: usize,
}

/// Largest source block: 64 pixels plus the 7-pixel filter margin.
pub(super) const MAX: usize = 64 + 7;

/// Reusable buffers for [`predict`].
pub struct Scratch {
	pub(super) block: Vec<u8>,
	pub(super) temp: Vec<u8>,
}

impl Default for Scratch {
	fn default() -> Self {
		Self { block: vec![0; MAX * MAX], temp: vec![0; MAX * 64] }
	}
}

#[inline]
fn tap8(px: &[u8], k: &[i16; 8]) -> u8 {
	let s = i32::from(px[0]) * i32::from(k[0])
		+ i32::from(px[1]) * i32::from(k[1])
		+ i32::from(px[2]) * i32::from(k[2])
		+ i32::from(px[3]) * i32::from(k[3])
		+ i32::from(px[4]) * i32::from(k[4])
		+ i32::from(px[5]) * i32::from(k[5])
		+ i32::from(px[6]) * i32::from(k[6])
		+ i32::from(px[7]) * i32::from(k[7]);
	((s + 64) >> 7).clamp(0, 255) as u8
}

/// Predicts a `w` x `h` block at (`x`, `y`) of `dst` from `src`, displaced
/// by (`mv_col`, `mv_row`) in 1/16 pixels. `avg` averages with what `dst`
/// already holds.
#[allow(clippy::too_many_arguments)]
pub fn predict(
	scratch: &mut Scratch,
	src: &RefPlane,
	dst: &mut [u8],
	dst_stride: usize,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mv_row: i32,
	mv_col: i32,
	filter: u8,
	avg: bool,
) {
	if crate::simd::enabled() && (w == 4 || w.is_multiple_of(8)) {
		super::simd::predict(scratch, src, dst, dst_stride, x, y, w, h, mv_row, mv_col, filter, avg);
	} else {
		predict_scalar(scratch, src, dst, dst_stride, x, y, w, h, mv_row, mv_col, filter, avg);
	}
}

/// Scalar version of [`predict`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_scalar(
	scratch: &mut Scratch,
	src: &RefPlane,
	dst: &mut [u8],
	dst_stride: usize,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mv_row: i32,
	mv_col: i32,
	filter: u8,
	avg: bool,
) {
	let kernels = &SUBPEL_FILTERS[usize::from(filter.min(3))];
	let (phase_x, phase_y) = ((mv_col & 15) as usize, (mv_row & 15) as usize);
	let x0 = x as i64 + i64::from(mv_col >> 4) - 3;
	let y0 = y as i64 + i64::from(mv_row >> 4) - 3;
	let (sw, sh) = (src.width as i64, src.height as i64);
	let (bw, bh) = (w + 7, h + 7);

	// The source block with a 3/4-pixel margin: straight from the picture
	// when inside it, else built with the edges repeated.
	let inside = x0 >= 0 && y0 >= 0 && x0 + bw as i64 <= sw && y0 + bh as i64 <= sh;
	let (block, base, stride): (&[u8], usize, usize) = if inside {
		(src.data, y0 as usize * src.stride + x0 as usize, src.stride)
	} else {
		for r in 0..bh {
			let sy = (y0 + r as i64).clamp(0, sh - 1) as usize;
			let row = &src.data[sy * src.stride..];
			let out = &mut scratch.block[r * MAX..r * MAX + bw];
			for (c, o) in out.iter_mut().enumerate() {
				*o = row[(x0 + c as i64).clamp(0, sw - 1) as usize];
			}
		}
		(&scratch.block, 0, MAX)
	};

	// Horizontal pass into `temp` (rows -3..h+4), or straight through.
	let temp = &mut scratch.temp;
	let (rows_from, rows_base, rows_stride): (&[u8], usize, usize) = if phase_x != 0 {
		let k = &kernels[phase_x];
		let (r0, r1) = if phase_y != 0 { (0, bh) } else { (3, 3 + h) };
		for r in r0..r1 {
			let row = &block[base + r * stride..];
			let out = &mut temp[r * 64..r * 64 + w];
			for (c, o) in out.iter_mut().enumerate() {
				*o = tap8(&row[c..c + 8], k);
			}
		}
		(&temp[..], 0, 64)
	} else {
		// No horizontal filter: the source columns 3.. are the pixels.
		(block, base + 3, stride)
	};

	let k = &kernels[phase_y];
	for r in 0..h {
		let out = &mut dst[(y + r) * dst_stride + x..(y + r) * dst_stride + x + w];
		if phase_y != 0 {
			let mut col = [0u8; 8];
			for (c, o) in out.iter_mut().enumerate() {
				for (t, v) in col.iter_mut().enumerate() {
					*v = rows_from[rows_base + (r + t) * rows_stride + c];
				}
				let v = tap8(&col, k);
				*o = if avg { ((u16::from(*o) + u16::from(v) + 1) >> 1) as u8 } else { v };
			}
		} else {
			let row = &rows_from[rows_base + (r + 3) * rows_stride..];
			for (c, o) in out.iter_mut().enumerate() {
				let v = row[c];
				*o = if avg { ((u16::from(*o) + u16::from(v) + 1) >> 1) as u8 } else { v };
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn whole_pixel_motion_copies_with_edge_repeat() {
		// 4x4 picture 0..16; a 4x4 block moved one pixel left and up.
		let pixels: Vec<u8> = (0..16).collect();
		let src = RefPlane { data: &pixels, stride: 4, width: 4, height: 4 };
		let mut dst = vec![0u8; 16];
		let mut s = Scratch::default();
		predict(&mut s, &src, &mut dst, 4, 0, 0, 4, 4, -16, -16, 0, false);
		assert_eq!(&dst[..4], &[0, 0, 1, 2]);
		assert_eq!(&dst[4..8], &[0, 0, 1, 2]);
		assert_eq!(&dst[8..12], &[4, 4, 5, 6]);
		// Averaging with itself is idempotent.
		let before = dst.clone();
		predict(&mut s, &src, &mut dst, 4, 0, 0, 4, 4, -16, -16, 0, true);
		assert_eq!(dst, before);
	}

	#[test]
	fn skipped_passes_match_the_full_filter() {
		// A unit-tap pass is exact, so phase 0 shortcuts change nothing.
		let pixels: Vec<u8> = (0..32 * 32).map(|i| ((i * 37) % 251) as u8).collect();
		let src = RefPlane { data: &pixels, stride: 32, width: 32, height: 32 };
		let mut s = Scratch::default();
		for (mr, mc) in [(0, 5), (7, 0), (0, 0), (-9, 13)] {
			let mut a = vec![0u8; 32 * 32];
			predict(&mut s, &src, &mut a, 32, 8, 8, 8, 8, mr, mc, 0, false);
			// Reference: explicit 2-D filter with both passes.
			let k = &SUBPEL_FILTERS[0];
			let (fx, fy) = (&k[(mc & 15) as usize], &k[(mr & 15) as usize]);
			for r in 0..8i32 {
				for c in 0..8i32 {
					let px = |yy: i32, xx: i32| pixels[(yy.clamp(0, 31) * 32 + xx.clamp(0, 31)) as usize];
					let col: Vec<u8> = (0..8)
						.map(|t| {
							let yy = 8 + r + (mr >> 4) - 3 + t;
							let row: Vec<u8> = (0..8).map(|u| px(yy, 8 + c + (mc >> 4) - 3 + u)).collect();
							tap8(&row, fx)
						})
						.collect();
					assert_eq!(a[((8 + r) * 32 + 8 + c) as usize], tap8(&col, fy), "{mr},{mc} {r},{c}");
				}
			}
		}
	}
}
