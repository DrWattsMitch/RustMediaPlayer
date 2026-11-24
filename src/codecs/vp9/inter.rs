//! Inter prediction (spec §8.5.2; libvpx `dec_build_inter_predictors`,
//! `vpx_dsp/vpx_convolve.c`).
//!
//! An 8-tap separable filter at 1/16-pixel phase: a horizontal pass into a
//! temporary block (rounded and clipped to 8 bits), then a vertical pass.
//! A pass whose phase is 0 is a plain copy (its kernel is the unit tap), so
//! it is skipped. Reference pixels outside the picture repeat its edge
//! (libvpx's `build_mc_border`). Compound prediction averages the second
//! prediction with the first. High bit depths use the same filters with
//! results clipped to `(1 << bit_depth) - 1` (`vpx_highbd_convolve8`).
//!
//! References of another size than the frame go through
//! [`predict_scaled`], whose source position advances by a fractional step
//! per output pixel (libvpx `vpx_scaled_2d`).

use super::pixel::Px;
use super::tables::SUBPEL_FILTERS;

/// A reference plane: pixels and the visible size.
pub struct RefPlane<'a, P = u8> {
	pub data: &'a [P],
	pub stride: usize,
	pub width: usize,
	pub height: usize,
}

/// Largest source block: 64 pixels plus the 7-pixel filter margin.
pub(super) const MAX: usize = 64 + 7;

/// Rows of the scaled filter's intermediate block: a 64-row block read from
/// a reference twice as large spans `(64 - 1) * 32 / 16` rows, plus the
/// phase and the 8 taps (libvpx `vpx_convolve8_c`'s `temp[64 * 135]`).
const SCALED_ROWS: usize = 135;

/// Reusable buffers for [`predict_scalar`] and [`predict_scaled`].
pub struct Scratch<P = u8> {
	pub(super) block: Vec<P>,
	pub(super) temp: Vec<P>,
}

impl<P: Px> Default for Scratch<P> {
	fn default() -> Self {
		Self { block: vec![P::default(); MAX * MAX], temp: vec![P::default(); SCALED_ROWS * 64] }
	}
}

#[inline]
fn tap8<P: Px>(px: &[P], k: &[i16; 8], max: i32) -> P {
	let s = px[0].into() * i32::from(k[0])
		+ px[1].into() * i32::from(k[1])
		+ px[2].into() * i32::from(k[2])
		+ px[3].into() * i32::from(k[3])
		+ px[4].into() * i32::from(k[4])
		+ px[5].into() * i32::from(k[5])
		+ px[6].into() * i32::from(k[6])
		+ px[7].into() * i32::from(k[7]);
	P::clip((s + 64) >> 7, max)
}

#[inline]
fn average<P: Px>(a: P, b: P) -> P {
	P::of((a.into() + b.into() + 1) >> 1)
}

/// Predicts a `w` x `h` block at (`x`, `y`) of `dst` from `src`, displaced
/// by (`mv_col`, `mv_row`) in 1/16 pixels. `avg` averages with what `dst`
/// already holds; `max` is the largest sample value.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn predict<P: Px>(
	scratch: &mut Scratch<P>,
	src: &RefPlane<P>,
	dst: &mut [P],
	dst_stride: usize,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mv_row: i32,
	mv_col: i32,
	filter: u8,
	avg: bool,
	max: i32,
) {
	P::predict(scratch, src, dst, dst_stride, x, y, w, h, mv_row, mv_col, filter, avg, max);
}

/// Scalar version of [`predict`]: the 10/12-bit path, the 8-bit fallback
/// and the reference the vector code is tested against.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_scalar<P: Px>(
	scratch: &mut Scratch<P>,
	src: &RefPlane<P>,
	dst: &mut [P],
	dst_stride: usize,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mv_row: i32,
	mv_col: i32,
	filter: u8,
	avg: bool,
	max: i32,
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
	let (block, base, stride): (&[P], usize, usize) = if inside {
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
	let (rows_from, rows_base, rows_stride): (&[P], usize, usize) = if phase_x != 0 {
		let k = &kernels[phase_x];
		let (r0, r1) = if phase_y != 0 { (0, bh) } else { (3, 3 + h) };
		for r in r0..r1 {
			let row = &block[base + r * stride..];
			let out = &mut temp[r * 64..r * 64 + w];
			for (c, o) in out.iter_mut().enumerate() {
				*o = tap8(&row[c..c + 8], k, max);
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
			let mut col = [P::default(); 8];
			for (c, o) in out.iter_mut().enumerate() {
				for (t, v) in col.iter_mut().enumerate() {
					*v = rows_from[rows_base + (r + t) * rows_stride + c];
				}
				let v = tap8(&col, k, max);
				*o = if avg { average(*o, v) } else { v };
			}
		} else {
			let row = &rows_from[rows_base + (r + 3) * rows_stride..];
			for (c, o) in out.iter_mut().enumerate() {
				let v = row[c];
				*o = if avg { average(*o, v) } else { v };
			}
		}
	}
}

/// Where a block's prediction starts in a scaled reference: the integer
/// sample (`x`, `y`) of the first output pixel and its phase, plus the
/// step from one output pixel to the next, all in the reference's samples
/// (phases and steps in 1/16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaledPos {
	pub x: i64,
	pub y: i64,
	pub phase_x: i32,
	pub phase_y: i32,
	pub step_x: i32,
	pub step_y: i32,
}

/// Prediction from a reference of another size (libvpx `vpx_scaled_2d` /
/// `vpx_convolve8_c` with steps other than 16): every output pixel steps
/// its source position by `step` sixteenths, and takes the 8-tap kernel of
/// its own phase. Horizontal pass into an intermediate block (rounded and
/// clipped), then vertical. A phase-0 kernel is the unit tap, so libvpx's
/// one-direction variants (`vpx_scaled_horiz` / `_vert`) give the same
/// result. Reads outside the reference repeat its edge, which is what
/// libvpx's border extension (`extend_and_predict`) amounts to.
#[allow(clippy::too_many_arguments)]
pub fn predict_scaled<P: Px>(
	scratch: &mut Scratch<P>,
	src: &RefPlane<P>,
	dst: &mut [P],
	dst_stride: usize,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	pos: ScaledPos,
	filter: u8,
	avg: bool,
	max: i32,
) {
	let kernels = &SUBPEL_FILTERS[usize::from(filter.min(3))];
	let (sw, sh) = (src.width as i64, src.height as i64);
	let (step_x, step_y) = (pos.step_x as i64, pos.step_y as i64);
	let rows = ((((h as i64 - 1) * step_y + i64::from(pos.phase_y)) >> 4) + 8) as usize;
	let rows = rows.min(SCALED_ROWS);
	let x_last = pos.x + ((i64::from(pos.phase_x) + (w as i64 - 1) * step_x) >> 4) + 4;
	let inside_x = pos.x - 3 >= 0 && x_last < sw;

	let temp = &mut scratch.temp;
	for r in 0..rows {
		let sy = (pos.y - 3 + r as i64).clamp(0, sh - 1) as usize;
		let row = &src.data[sy * src.stride..sy * src.stride + src.width];
		let out = &mut temp[r * 64..r * 64 + w];
		let mut q4 = i64::from(pos.phase_x);
		for o in out.iter_mut() {
			let k = &kernels[(q4 & 15) as usize];
			let at = pos.x + (q4 >> 4) - 3;
			*o = if inside_x {
				tap8(&row[at as usize..at as usize + 8], k, max)
			} else {
				let px: [P; 8] = std::array::from_fn(|t| row[(at + t as i64).clamp(0, sw - 1) as usize]);
				tap8(&px, k, max)
			};
			q4 += step_x;
		}
	}

	let mut q4 = i64::from(pos.phase_y);
	for r in 0..h {
		let k = &kernels[(q4 & 15) as usize];
		let top = (q4 >> 4) as usize;
		let out = &mut dst[(y + r) * dst_stride + x..(y + r) * dst_stride + x + w];
		for (c, o) in out.iter_mut().enumerate() {
			let col: [P; 8] = std::array::from_fn(|t| temp[(top + t).min(rows - 1) * 64 + c]);
			let v = tap8(&col, k, max);
			*o = if avg { average(*o, v) } else { v };
		}
		q4 += step_y;
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
		predict(&mut s, &src, &mut dst, 4, 0, 0, 4, 4, -16, -16, 0, false, 255);
		assert_eq!(&dst[..4], &[0, 0, 1, 2]);
		assert_eq!(&dst[4..8], &[0, 0, 1, 2]);
		assert_eq!(&dst[8..12], &[4, 4, 5, 6]);
		// Averaging with itself is idempotent.
		let before = dst.clone();
		predict(&mut s, &src, &mut dst, 4, 0, 0, 4, 4, -16, -16, 0, true, 255);
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
			predict(&mut s, &src, &mut a, 32, 8, 8, 8, 8, mr, mc, 0, false, 255);
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
							tap8(&row, fx, 255)
						})
						.collect();
					assert_eq!(a[((8 + r) * 32 + 8 + c) as usize], tap8(&col, fy, 255), "{mr},{mc} {r},{c}");
				}
			}
		}
	}
}
