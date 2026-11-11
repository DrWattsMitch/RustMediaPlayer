//! The loop (deblocking) filter (spec §8.8; libvpx `vp9_loopfilter.c`
//! `vp9_filter_block_plane_non420`, `vpx_dsp/loopfilter.c`).
//!
//! Runs after the whole frame is reconstructed, superblock by superblock in
//! raster order, and in each superblock plane by plane: first every
//! vertical edge (left to right in each 8-pixel row), then every
//! horizontal edge. Edges follow transform boundaries; the filter length
//! (4, 8 or 16 taps) comes from the transform size, and each 8x8 block's
//! level from its segment, reference frame and mode.

use super::decoder::Plane;
use super::mode::ModeInfo;
use super::{NUM_4X4_HIGH, NUM_4X4_WIDE, NUM_8X8_HIGH, NUM_8X8_WIDE, uv_tx_size};

/// Limits for one filter level.
#[derive(Debug, Clone, Copy, Default)]
pub struct Thresholds {
	pub mblim: u8,
	pub lim: u8,
	pub hev: u8,
}

/// Limits of every level for a sharpness (`update_sharpness`).
pub fn thresholds(sharpness: u8) -> [Thresholds; 64] {
	std::array::from_fn(|lvl| {
		let lvl = lvl as i32;
		let s = i32::from(sharpness);
		let mut inside = lvl >> (i32::from(s > 0) + i32::from(s > 4));
		if s > 0 {
			inside = inside.min(9 - s);
		}
		inside = inside.max(1);
		Thresholds { mblim: (2 * (lvl + 2) + inside) as u8, lim: inside as u8, hev: (lvl >> 4) as u8 }
	})
}

/// Mode delta index of each mode: 0 for intra and ZEROMV, 1 otherwise.
const MODE_LF_LUT: [usize; 14] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1];

/// Filter level of a block from the frame's `[segment][ref][mode]` table.
pub fn level(levels: &[[[u8; 2]; 4]; 8], mi: &ModeInfo) -> u8 {
	levels[usize::from(mi.segment_id)][mi.ref_frame[0].max(0) as usize]
		[MODE_LF_LUT[usize::from(mi.mode)]]
}

#[inline]
fn d(a: u8, b: u8) -> u8 {
	a.abs_diff(b)
}

#[inline]
fn sclamp(v: i32) -> i32 {
	v.clamp(-128, 127)
}

/// The 4-tap filter (`filter4`) on `[p1, p0, q0, q1]`, once the mask passed.
#[inline]
fn filter4(px: &mut [u8], hev_thr: u8) {
	let [p1, p0, q0, q1] = [px[0], px[1], px[2], px[3]];
	let hev = d(p1, p0) > hev_thr || d(q1, q0) > hev_thr;
	let s = |v: u8| i32::from(v) - 128;
	let (ps1, ps0, qs0, qs1) = (s(p1), s(p0), s(q0), s(q1));
	let mut filter = if hev { sclamp(ps1 - qs1) } else { 0 };
	filter = sclamp(filter + 3 * (qs0 - ps0));
	let filter1 = sclamp(filter + 4) >> 3;
	let filter2 = sclamp(filter + 3) >> 3;
	px[2] = (sclamp(qs0 - filter1) + 128) as u8;
	px[1] = (sclamp(ps0 + filter2) + 128) as u8;
	if !hev {
		let f = (filter1 + 1) >> 1;
		px[3] = (sclamp(qs1 - f) + 128) as u8;
		px[0] = (sclamp(ps1 + f) + 128) as u8;
	}
}

/// Filters across one edge. `px` holds `p3..p0 q0..q3` (8 pixels) or, for
/// the 16-wide filter, `p7..p0 q0..q7` (16 pixels).
#[inline]
fn filter_edge(px: &mut [u8], t: &Thresholds, taps: u8) {
	let c = px.len() / 2;
	let p = |i: usize| px[c - 1 - i];
	let q = |i: usize| px[c + i];
	let lim = t.lim;
	let mask = d(p(3), p(2)) <= lim
		&& d(p(2), p(1)) <= lim
		&& d(p(1), p(0)) <= lim
		&& d(q(1), q(0)) <= lim
		&& d(q(2), q(1)) <= lim
		&& d(q(3), q(2)) <= lim
		&& u32::from(d(p(0), q(0))) * 2 + u32::from(d(p(1), q(1))) / 2 <= u32::from(t.mblim);
	if !mask {
		return;
	}
	let (p0, q0) = (p(0), q(0));
	let flat = taps >= 8 && (1..=3).all(|i| d(p(i), p0) <= 1 && d(q(i), q0) <= 1);
	if taps == 16 && flat && (4..=7).all(|i| d(p(i), p0) <= 1 && d(q(i), q0) <= 1) {
		// 15-tap filter [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]: a
		// running sum over the window, with the ends repeated.
		let v: [u32; 16] = std::array::from_fn(|i| u32::from(px[i]));
		let mut sum = v[0] * 7 + v[1] * 2 + v[2] + v[3] + v[4] + v[5] + v[6] + v[7] + v[8];
		let mut out = [0u8; 14];
		out[0] = ((sum + 8) >> 4) as u8;
		for k in 1..14 {
			// The window of pixel k + 1 drops pixel k - 7 (repeating p7),
			// takes in pixel k + 8 (repeating q7), and its doubled centre
			// moves from pixel k to k + 1.
			sum = sum + v[(k + 8).min(15)] - v[k.saturating_sub(7)] + v[k + 1] - v[k];
			out[k] = ((sum + 8) >> 4) as u8;
		}
		px[1..15].copy_from_slice(&out);
	} else if flat {
		let b = c - 4;
		let [p3, p2, p1, p0, q0, q1, q2, q3] =
			std::array::from_fn::<u32, 8, _>(|i| u32::from(px[b + i]));
		let r = |v: u32| ((v + 4) >> 3) as u8;
		px[b + 1] = r(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0);
		px[b + 2] = r(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1);
		px[b + 3] = r(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2);
		px[b + 4] = r(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3);
		px[b + 5] = r(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3);
		px[b + 6] = r(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3);
	} else {
		filter4(&mut px[c - 2..c + 2], t.hev);
	}
}

/// Pixels on each side of an edge the filter reads.
fn reach(taps: u8) -> usize {
	if taps == 16 { 8 } else { 4 }
}

/// Filters an 8-pixel stretch of a vertical edge at (`x`, `y`).
fn vertical(plane: &mut Plane, x: usize, y: usize, t: &Thresholds, taps: u8) {
	let n = reach(taps);
	let stride = plane.stride;
	for r in 0..8 {
		let at = (y + r) * stride + x;
		filter_edge(&mut plane.data[at - n..at + n], t, taps);
	}
}

/// Filters an 8-pixel stretch of a horizontal edge at (`x`, `y`).
fn horizontal(plane: &mut Plane, x: usize, y: usize, t: &Thresholds, taps: u8) {
	let n = reach(taps);
	let stride = plane.stride;
	let mut px = [0u8; 16];
	for c in 0..8 {
		let base = (y - n) * stride + x + c;
		for (i, v) in px[..2 * n].iter_mut().enumerate() {
			*v = plane.data[base + i * stride];
		}
		filter_edge(&mut px[..2 * n], t, taps);
		for (i, &v) in px[..2 * n].iter().enumerate() {
			plane.data[base + i * stride] = v;
		}
	}
}

/// Filters one plane of the superblock at (`mi_row`, `mi_col`).
#[allow(clippy::too_many_arguments)]
pub fn filter_superblock(
	plane: &mut Plane,
	mi: &[ModeInfo],
	levels: &[u8],
	mi_rows: usize,
	mi_cols: usize,
	mi_row: usize,
	mi_col: usize,
	ss_x: usize,
	ss_y: usize,
	thr: &[Thresholds; 64],
) {
	let row_step = 1 << ss_y;
	let col_step = 1 << ss_x;
	let mut mask_16 = [0u32; 8];
	let mut mask_8 = [0u32; 8];
	let mut mask_4 = [0u32; 8];
	let mut mask_4_int = [0u32; 8];
	let mut lfl = [0u8; 64];
	let x_base = (mi_col * 8) >> ss_x;
	let y_base = (mi_row * 8) >> ss_y;

	let mut r = 0;
	while r < 8 && mi_row + r < mi_rows {
		let (mut m16c, mut m8c, mut m4c) = (0u32, 0u32, 0u32);
		let mut c = 0;
		while c < 8 && mi_col + c < mi_cols {
			let at = (mi_row + r) * mi_cols + mi_col + c;
			let m = &mi[at];
			let bt = usize::from(m.sb_type);
			let skip_this = m.skip && m.is_inter();
			let edge_left =
				if NUM_4X4_WIDE[bt] > 1 { c & (usize::from(NUM_8X8_WIDE[bt]) - 1) == 0 } else { true };
			let skip_c = skip_this && !edge_left;
			let edge_above =
				if NUM_4X4_HIGH[bt] > 1 { r & (usize::from(NUM_8X8_HIGH[bt]) - 1) == 0 } else { true };
			let skip_r = skip_this && !edge_above;
			let tx = if ss_x == 0 && ss_y == 0 {
				m.tx_size
			} else {
				uv_tx_size(m.sb_type, m.tx_size, ss_x, ss_y)
			};
			let border_c = ss_x == 1 && mi_col + c == mi_cols - 1;
			let border_r = ss_y == 1 && mi_row + r == mi_rows - 1;
			let k = c >> ss_x;
			let bit = 1u32 << k;
			let level = levels[at];
			lfl[(r << 3) + k] = level;
			if level != 0 {
				match tx {
					3 | 2 => {
						let period = if tx == 3 { 3 } else { 1 };
						if !skip_c && (k & period) == 0 {
							if border_c {
								m8c |= bit;
							} else {
								m16c |= bit;
							}
						}
						if !skip_r && ((r >> ss_y) & period) == 0 {
							if border_r {
								mask_8[r] |= bit;
							} else {
								mask_16[r] |= bit;
							}
						}
					}
					_ => {
						if !skip_c {
							if tx == 1 || k & 3 == 0 {
								m8c |= bit;
							} else {
								m4c |= bit;
							}
						}
						if !skip_r {
							if tx == 1 || (r >> ss_y) & 3 == 0 {
								mask_8[r] |= bit;
							} else {
								mask_4[r] |= bit;
							}
						}
						if !skip_this && tx == 0 && !border_c {
							mask_4_int[r] |= bit;
						}
					}
				}
			}
			c += col_step;
		}
		// No filtering on the picture's left edge.
		let border = if mi_col == 0 { !1u32 } else { !0 };
		let (m16c, m8c, m4c) = (m16c & border, m8c & border, m4c & border);
		let y = y_base + (r >> ss_y) * 8;
		let all = m16c | m8c | m4c | mask_4_int[r];
		for k in 0..8 {
			if all >> k == 0 {
				break;
			}
			let t = &thr[usize::from(lfl[(r << 3) + k])];
			let x = x_base + 8 * k;
			if m16c >> k & 1 == 1 {
				vertical(plane, x, y, t, 16);
			} else if m8c >> k & 1 == 1 {
				vertical(plane, x, y, t, 8);
			} else if m4c >> k & 1 == 1 {
				vertical(plane, x, y, t, 4);
			}
			if mask_4_int[r] >> k & 1 == 1 {
				vertical(plane, x + 4, y, t, 4);
			}
		}
		r += row_step;
	}

	let mut r = 0;
	while r < 8 && mi_row + r < mi_rows {
		let border_r = ss_y == 1 && mi_row + r == mi_rows - 1;
		let int = if border_r { 0 } else { mask_4_int[r] };
		let (m16, m8, m4) =
			if mi_row + r == 0 { (0, 0, 0) } else { (mask_16[r], mask_8[r], mask_4[r]) };
		let y = y_base + (r >> ss_y) * 8;
		for k in 0..8 {
			if (m16 | m8 | m4 | int) >> k == 0 {
				break;
			}
			let t = &thr[usize::from(lfl[(r << 3) + k])];
			let x = x_base + 8 * k;
			if m16 >> k & 1 == 1 {
				horizontal(plane, x, y, t, 16);
			} else if m8 >> k & 1 == 1 {
				horizontal(plane, x, y, t, 8);
			} else if m4 >> k & 1 == 1 {
				horizontal(plane, x, y, t, 4);
			}
			if int >> k & 1 == 1 {
				horizontal(plane, x, y + 4, t, 4);
			}
		}
		r += row_step;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn sharpness_limits() {
		let t = thresholds(0);
		assert_eq!((t[10].lim, t[10].mblim, t[10].hev), (10, 34, 0));
		assert_eq!(t[0].lim, 1);
		let t = thresholds(5);
		// 40 >> 2 = 10, capped at 9 - 5.
		assert_eq!((t[40].lim, t[40].hev), (4, 2));
	}

	#[test]
	fn flat_step_is_smoothed() {
		// A flat 8-tap edge: p = 10, q = 12 → the 7-tap filter blends.
		let mut row = [10u8, 10, 10, 10, 12, 12, 12, 12];
		let t = Thresholds { mblim: 30, lim: 5, hev: 0 };
		filter_edge(&mut row, &t, 8);
		assert_eq!(row, [10, 10, 11, 11, 11, 12, 12, 12]);
	}

	#[test]
	fn wide_filter_matches_libvpx_formulas() {
		let mut px = [20u8, 20, 21, 20, 21, 20, 20, 21, 22, 22, 21, 22, 22, 21, 22, 22];
		let v: Vec<u32> = px.iter().map(|&x| u32::from(x)).collect();
		let (p, q) = (|i: usize| v[7 - i], |i: usize| v[8 + i]);
		let r4 = |s: u32| ((s + 8) >> 4) as u8;
		let expect_p6 = r4(p(7) * 7 + p(6) * 2 + p(5) + p(4) + p(3) + p(2) + p(1) + p(0) + q(0));
		let expect_p0 = r4(
			p(7)
				+ p(6)
				+ p(5)
				+ p(4)
				+ p(3)
				+ p(2)
				+ p(1)
				+ p(0) * 2
				+ q(0)
				+ q(1)
				+ q(2)
				+ q(3)
				+ q(4)
				+ q(5)
				+ q(6),
		);
		let expect_q3 =
			r4(p(3) + p(2) + p(1) + p(0) + q(0) + q(1) + q(2) + q(3) * 2 + q(4) + q(5) + q(6) + q(7) * 4);
		let expect_q6 = r4(p(0) + q(0) + q(1) + q(2) + q(3) + q(4) + q(5) + q(6) * 2 + q(7) * 7);
		filter_edge(&mut px, &Thresholds { mblim: 60, lim: 10, hev: 0 }, 16);
		assert_eq!((px[1], px[7], px[11], px[14]), (expect_p6, expect_p0, expect_q3, expect_q6));
	}
}
