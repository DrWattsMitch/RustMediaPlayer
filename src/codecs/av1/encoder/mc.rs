//! Motion-compensated prediction, 8-bit, single reference, unscaled
//! (rav1d `recon.rs: mc` and `mc.rs: put_8tap_rust`): reads outside the
//! reference repeat its edge samples, the sub-pixel filter is the
//! sequence's regular 8-tap (its 4-tap version for blocks 4 samples wide or
//! tall), intermediate precision 4 bits.

use super::tables::SUBPEL_FILTERS;

/// A reference plane and its visible size.
pub struct RefPlane<'a> {
	pub data: &'a [u8],
	pub stride: usize,
	pub width: usize,
	pub height: usize,
}

fn filter(phase: usize, len: usize) -> Option<&'static [i8; 8]> {
	let m = phase.checked_sub(1)?;
	// Regular (0); its 4-tap version (3) for 4-sample blocks.
	Some(&SUBPEL_FILTERS[if len > 4 { 0 } else { 3 }][m])
}

/// Predicts the `w x h` block of plane `pl` (0 luma, 1-2 chroma, 4:2:0) at
/// (`x`, `y`) samples into `dst` from `src`, displaced by `mv` (row, col)
/// in 1/8 luma samples.
#[allow(clippy::too_many_arguments)]
pub fn predict(
	src: &RefPlane,
	dst: &mut [u8],
	dst_stride: usize,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mv: (i32, i32),
	chroma: bool,
) {
	let ss = usize::from(chroma);
	let (mvy, mvx) = mv;
	let mx = (mvx & (15 >> (1 - ss))) as usize;
	let my = (mvy & (15 >> (1 - ss))) as usize;
	let dx = x as i64 + i64::from(mvx >> (3 + ss));
	let dy = y as i64 + i64::from(mvy >> (3 + ss));
	let (mx, my) = (mx << (1 - ss), my << (1 - ss));
	// Source block with a 3/4-sample margin, edges repeated.
	const S: usize = 64 + 8;
	let mut block = [0u8; S * S];
	let (sw, sh) = (src.width as i64, src.height as i64);
	for r in 0..h + 7 {
		let sy = (dy + r as i64 - 3).clamp(0, sh - 1) as usize;
		let row = &src.data[sy * src.stride..];
		for c in 0..w + 7 {
			block[r * S + c] = row[(dx + c as i64 - 3).clamp(0, sw - 1) as usize];
		}
	}
	let px = |r: usize, c: usize| i32::from(block[(r + 3) * S + c + 3]);
	let fh = filter(mx, w);
	let fv = filter(my, h);
	let tap = |f: &[i8; 8], get: &dyn Fn(usize) -> i32| -> i32 {
		(0..8).map(|k| i32::from(f[k]) * get(k)).sum()
	};
	match (fh, fv) {
		(Some(fh), Some(fv)) => {
			let mut mid = [0i16; (64 + 7) * 64];
			for r in 0..h + 7 {
				for c in 0..w {
					let v = tap(fh, &|k| i32::from(block[r * S + c + k]));
					mid[r * 64 + c] = ((v + 2) >> 2) as i16;
				}
			}
			for r in 0..h {
				for c in 0..w {
					let v = tap(fv, &|k| i32::from(mid[(r + k) * 64 + c]));
					dst[(y + r) * dst_stride + x + c] = ((v + 512) >> 10).clamp(0, 255) as u8;
				}
			}
		}
		(Some(fh), None) => {
			for r in 0..h {
				for c in 0..w {
					let v = tap(fh, &|k| i32::from(block[(r + 3) * S + c + k]));
					dst[(y + r) * dst_stride + x + c] = ((v + 34) >> 6).clamp(0, 255) as u8;
				}
			}
		}
		(None, Some(fv)) => {
			for r in 0..h {
				for c in 0..w {
					let v = tap(fv, &|k| i32::from(block[(r + k) * S + c + 3]));
					dst[(y + r) * dst_stride + x + c] = ((v + 32) >> 6).clamp(0, 255) as u8;
				}
			}
		}
		(None, None) => {
			for r in 0..h {
				for c in 0..w {
					dst[(y + r) * dst_stride + x + c] = px(r, c) as u8;
				}
			}
		}
	}
}
