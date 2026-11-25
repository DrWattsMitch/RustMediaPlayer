//! Intra prediction, 8-bit (rav1d `ipred_prepare.rs` / `ipred.rs`, as the
//! decoder runs them when the sequence disables the intra edge filter).
//!
//! [`predict`] gathers a transform block's edges from the reconstruction
//! (`rav1d_prepare_intra_edges`: missing edges extended per the spec, the
//! mode converted to the predictor actually run) and predicts the block in
//! place.

use super::tables::{DR_INTRA_DERIVATIVE, SM_WEIGHTS};

pub const DC_PRED: u8 = 0;
pub const VERT_PRED: u8 = 1;
pub const HOR_PRED: u8 = 2;
pub const VERT_LEFT_PRED: u8 = 8;
pub const SMOOTH_PRED: u8 = 9;
pub const SMOOTH_V_PRED: u8 = 10;
pub const SMOOTH_H_PRED: u8 = 11;
pub const PAETH_PRED: u8 = 12;

// Implementation modes.
const LEFT_DC_PRED: u8 = 3;
const TOP_DC_PRED: u8 = 4;
const DC_128_PRED: u8 = 5;
const Z1_PRED: u8 = 6;
const Z2_PRED: u8 = 7;
const Z3_PRED: u8 = 8;

const LEFT: u8 = 1;
const TOP: u8 = 2;
const TOP_LEFT: u8 = 4;
const TOP_RIGHT: u8 = 8;
const BOTTOM_LEFT: u8 = 16;

/// Edges each implementation mode reads.
const NEEDS: [u8; 13] = [
	TOP | LEFT,                    // DC
	TOP,                           // V
	LEFT,                          // H
	LEFT,                          // left DC
	TOP,                           // top DC
	0,                             // DC 128
	TOP | TOP_RIGHT | TOP_LEFT,    // Z1
	LEFT | TOP | TOP_LEFT,         // Z2
	LEFT | BOTTOM_LEFT | TOP_LEFT, // Z3
	LEFT | TOP,                    // smooth
	LEFT | TOP,                    // smooth V
	LEFT | TOP,                    // smooth H
	LEFT | TOP | TOP_LEFT,         // Paeth
];

const MODE_TO_ANGLE: [i32; 8] = [90, 180, 45, 135, 113, 157, 203, 67];

/// Where a transform block sits (all in 4-sample units of its plane).
#[derive(Debug, Clone, Copy)]
pub struct Pos {
	pub x: usize,
	pub y: usize,
	/// Transform size (width = height), log2 of 4-sample units.
	pub lw: usize,
	pub have_left: bool,
	pub have_top: bool,
	/// Right edge of the tile and bottom of the picture (aligned to 8 luma
	/// samples).
	pub w_end: usize,
	pub h_end: usize,
	pub top_has_right: bool,
	pub left_has_bottom: bool,
}

/// Predicts the transform block at `p` of `plane` (stride `stride`) with
/// `mode` and angle delta `delta` (directional modes).
pub fn predict(plane: &mut [u8], stride: usize, p: &Pos, mode: u8, delta: i8) {
	let t = 1usize << p.lw;
	let sz = 4 * t;
	let (have_left, have_top) = (p.have_left, p.have_top);
	let mut angle = 0i32;
	let mut m = mode;
	match mode {
		VERT_PRED..=VERT_LEFT_PRED => {
			angle = MODE_TO_ANGLE[usize::from(mode - VERT_PRED)] + 3 * i32::from(delta);
			m = if angle <= 90 {
				if angle < 90 && have_top { Z1_PRED } else { VERT_PRED }
			} else if angle < 180 {
				Z2_PRED
			} else if angle > 180 && have_left {
				Z3_PRED
			} else {
				HOR_PRED
			};
		}
		DC_PRED => {
			m = match (have_left, have_top) {
				(false, false) => DC_128_PRED,
				(false, true) => TOP_DC_PRED,
				(true, false) => LEFT_DC_PRED,
				(true, true) => DC_PRED,
			};
		}
		PAETH_PRED => {
			m = match (have_left, have_top) {
				(false, false) => DC_128_PRED,
				(false, true) => VERT_PRED,
				(true, false) => HOR_PRED,
				(true, true) => PAETH_PRED,
			};
		}
		_ => {}
	}
	let needs = NEEDS[usize::from(m)];
	let (px, py) = (4 * p.x, 4 * p.y);
	let at = |x: usize, y: usize| plane[y * stride + x];

	// edge[128] is the top-left sample; edge[129..] the top row and
	// top-right; edge[..128] the left column upwards from the bottom.
	const O: usize = 128;
	let mut edge = [0u8; 2 * O + 1];
	let top_row = |i: usize| at(px + i, py - 1);

	if needs & LEFT != 0 {
		if have_left {
			let have = sz.min(4 * (p.h_end - p.y));
			for i in 0..have {
				edge[O - 1 - i] = at(px - 1, py + i);
			}
			let fill = edge[O - have];
			for i in have..sz {
				edge[O - 1 - i] = fill;
			}
		} else {
			let v = if have_top { top_row(0) } else { 129 };
			for i in 0..sz {
				edge[O - 1 - i] = v;
			}
		}
		if needs & BOTTOM_LEFT != 0 {
			let have_bl = have_left && p.y + t < p.h_end && p.left_has_bottom;
			if have_bl {
				let have = sz.min(4 * (p.h_end - p.y - t));
				for i in 0..have {
					edge[O - 1 - sz - i] = at(px - 1, py + sz + i);
				}
				let fill = edge[O - sz - have];
				for i in have..sz {
					edge[O - 1 - sz - i] = fill;
				}
			} else {
				let fill = edge[O - sz];
				for i in 0..sz {
					edge[O - 1 - sz - i] = fill;
				}
			}
		}
	}
	if needs & TOP != 0 {
		if have_top {
			let have = sz.min(4 * (p.w_end - p.x));
			for i in 0..have {
				edge[O + 1 + i] = top_row(i);
			}
			let fill = edge[O + have];
			for i in have..sz {
				edge[O + 1 + i] = fill;
			}
		} else {
			let v = if have_left { at(px - 1, py) } else { 127 };
			for i in 0..sz {
				edge[O + 1 + i] = v;
			}
		}
		if needs & TOP_RIGHT != 0 {
			let have_tr = have_top && p.x + t < p.w_end && p.top_has_right;
			if have_tr {
				let have = sz.min(4 * (p.w_end - p.x - t));
				for i in 0..have {
					edge[O + 1 + sz + i] = top_row(sz + i);
				}
				let fill = edge[O + sz + have];
				for i in have..sz {
					edge[O + 1 + sz + i] = fill;
				}
			} else {
				let fill = edge[O + sz];
				for i in 0..sz {
					edge[O + 1 + sz + i] = fill;
				}
			}
		}
	}
	if needs & TOP_LEFT != 0 {
		edge[O] = if have_top {
			if have_left { at(px - 1, py - 1) } else { top_row(0) }
		} else if have_left {
			at(px - 1, py)
		} else {
			128
		};
	}

	let n = sz;
	let mut out = |x: usize, y: usize, v: i32| plane[(py + y) * stride + px + x] = v as u8;
	let top = |i: usize| i32::from(edge[O + 1 + i]);
	let left = |i: usize| i32::from(edge[O - 1 - i]);
	let shift = n.trailing_zeros();
	match m {
		DC_PRED | LEFT_DC_PRED | TOP_DC_PRED | DC_128_PRED => {
			let dc = match m {
				DC_PRED => {
					let mut s = n as i32;
					for i in 0..n {
						s += top(i) + left(i);
					}
					s >> (shift + 1)
				}
				TOP_DC_PRED => ((n as i32 >> 1) + (0..n).map(top).sum::<i32>()) >> shift,
				LEFT_DC_PRED => ((n as i32 >> 1) + (0..n).map(left).sum::<i32>()) >> shift,
				_ => 128,
			};
			for y in 0..n {
				for x in 0..n {
					out(x, y, dc);
				}
			}
		}
		VERT_PRED => {
			for y in 0..n {
				for x in 0..n {
					out(x, y, top(x));
				}
			}
		}
		HOR_PRED => {
			for y in 0..n {
				for x in 0..n {
					out(x, y, left(y));
				}
			}
		}
		PAETH_PRED => {
			let tl = i32::from(edge[O]);
			for y in 0..n {
				let l = left(y);
				for x in 0..n {
					let tp = top(x);
					let base = l + tp - tl;
					let (ld, td, tld) = ((l - base).abs(), (tp - base).abs(), (tl - base).abs());
					out(
						x,
						y,
						if ld <= td && ld <= tld {
							l
						} else if td <= tld {
							tp
						} else {
							tl
						},
					);
				}
			}
		}
		SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED => {
			let w = &SM_WEIGHTS[n..2 * n];
			let right = top(n - 1);
			let bottom = left(n - 1);
			for y in 0..n {
				for x in 0..n {
					let (wv, wh) = (i32::from(w[y]), i32::from(w[x]));
					let v = match m {
						SMOOTH_PRED => {
							(wv * top(x) + (256 - wv) * bottom + wh * left(y) + (256 - wh) * right + 256) >> 9
						}
						SMOOTH_V_PRED => (wv * top(x) + (256 - wv) * bottom + 128) >> 8,
						_ => (wh * left(y) + (256 - wh) * right + 128) >> 8,
					};
					out(x, y, v);
				}
			}
		}
		Z1_PRED => {
			let dx = i32::from(DR_INTRA_DERIVATIVE[(angle >> 1) as usize]);
			let max_base_x = 2 * n - 1;
			for y in 0..n {
				let xpos = (y as i32 + 1) * dx;
				let frac = xpos & 0x3e;
				for x in 0..n {
					let base = (xpos >> 6) as usize + x;
					if base < max_base_x {
						out(x, y, (top(base) * (64 - frac) + top(base + 1) * frac + 32) >> 6);
					} else {
						for xx in x..n {
							out(xx, y, top(max_base_x));
						}
						break;
					}
				}
			}
		}
		Z2_PRED => {
			let dy = i32::from(DR_INTRA_DERIVATIVE[((angle - 90) >> 1) as usize]);
			let dx = i32::from(DR_INTRA_DERIVATIVE[((180 - angle) >> 1) as usize]);
			// e(i): i = 0 top-left, i > 0 top row, i < 0 left column.
			let e = |i: i32| i32::from(edge[(O as i32 + i) as usize]);
			for y in 0..n {
				let xpos = (1 << 6) - dx * (y as i32 + 1);
				let base_x = xpos >> 6;
				let frac_x = xpos & 0x3e;
				for x in 0..n {
					let bx = base_x + x as i32;
					let v = if bx >= 0 {
						e(bx) * (64 - frac_x) + e(bx + 1) * frac_x
					} else {
						let ypos = ((y as i32) << 6) - dy * (x as i32 + 1);
						let base_y = ypos >> 6;
						let frac_y = ypos & 0x3e;
						// edge[left - base_y] with left = topleft - 1.
						e(-1 - base_y) * (64 - frac_y) + e(-1 - (base_y + 1)) * frac_y
					};
					out(x, y, (v + 32) >> 6);
				}
			}
		}
		Z3_PRED => {
			let dy = i32::from(DR_INTRA_DERIVATIVE[((270 - angle) >> 1) as usize]) as usize;
			let max_base_y = 2 * n - 1;
			// left[left_off - base] = left(base).
			for x in 0..n {
				let ypos = dy * (x + 1);
				let frac = (ypos & 0x3e) as i32;
				for y in 0..n {
					let base = (ypos >> 6) + y;
					if base < max_base_y {
						out(x, y, (left(base) * (64 - frac) + left(base + 1) * frac + 32) >> 6);
					} else {
						for yy in y..n {
							out(x, yy, left(max_base_y));
						}
						break;
					}
				}
			}
		}
		_ => unreachable!("av1: intra mode"),
	}
}
