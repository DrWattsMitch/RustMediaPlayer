//! Intra prediction (spec §8.5.1; libvpx `vp9_reconintra.c`,
//! `vpx_dsp/intrapred.c`).
//!
//! Edges come from the reconstructed (not yet loop-filtered) frame. Missing
//! edges are 127 (above) or 129 (left); past the frame's 8-aligned size the
//! last pixel repeats; "above-right" pixels are real only for 4x4
//! transforms inside the block, and repeat the last above pixel otherwise.

#![allow(clippy::needless_range_loop)] // index arithmetic mirrors the specification

use super::pm;

/// Where a transform block sits and which edges exist.
pub struct Edges {
	pub have_top: bool,
	pub have_left: bool,
	pub have_right: bool,
	/// The block crosses the frame's right / bottom edge.
	pub clip_right: bool,
	pub clip_bottom: bool,
	/// Plane size rounded to 8 luma pixels (libvpx `y_width` / `uv_width`).
	pub frame_width: usize,
	pub frame_height: usize,
}

const NEED_LEFT: u8 = 2;
const NEED_ABOVE: u8 = 4;
const NEED_ABOVERIGHT: u8 = 8;

const EXTEND_MODES: [u8; 10] = [
	NEED_ABOVE | NEED_LEFT,
	NEED_ABOVE,
	NEED_LEFT,
	NEED_ABOVERIGHT,
	NEED_LEFT | NEED_ABOVE,
	NEED_LEFT | NEED_ABOVE,
	NEED_LEFT | NEED_ABOVE,
	NEED_LEFT,
	NEED_ABOVERIGHT,
	NEED_LEFT | NEED_ABOVE,
];

/// Transform type implied by a luma intra mode (`intra_mode_to_tx_type_lookup`).
pub const MODE_TO_TX_TYPE: [u8; 10] = [0, 1, 2, 0, 3, 1, 2, 2, 1, 3];

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
	((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
	((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

/// Predicts the `4 << tx_size` square at (`x`, `y`) of a plane.
pub fn predict(
	plane: &mut [u8],
	stride: usize,
	x: usize,
	y: usize,
	tx_size: u8,
	mode: u8,
	e: &Edges,
) {
	let bs = 4usize << tx_size;
	let mut left = [0u8; 32];
	// above[0] is the top-left pixel; the row starts at above[1].
	let mut above_buf = [0u8; 65];
	let need = EXTEND_MODES[usize::from(mode)];

	if need & NEED_LEFT != 0 {
		if e.have_left {
			let rows = if e.clip_bottom && y + bs > e.frame_height { e.frame_height - y } else { bs };
			for (i, l) in left.iter_mut().enumerate().take(bs) {
				let r = i.min(rows - 1);
				*l = plane[(y + r) * stride + x - 1];
			}
		} else {
			left[..bs].fill(129);
		}
	}
	let above_at = |i: usize| plane[(y - 1) * stride + x + i];
	if need & (NEED_ABOVE | NEED_ABOVERIGHT) != 0 {
		let width = if need & NEED_ABOVERIGHT != 0 { 2 * bs } else { bs };
		if e.have_top {
			let above = &mut above_buf[1..];
			// Real pixels available along the row.
			let real = if need & NEED_ABOVERIGHT != 0 && bs == 4 && e.have_right { 2 * bs } else { bs };
			let limit = if e.clip_right { e.frame_width.saturating_sub(x) } else { usize::MAX };
			let n = real.min(limit).max(1);
			for (i, a) in above.iter_mut().enumerate().take(n) {
				*a = above_at(i);
			}
			let last = above[n - 1];
			above[n..width].fill(last);
			above_buf[0] = if e.have_left { plane[(y - 1) * stride + x - 1] } else { 129 };
		} else {
			above_buf[..=width].fill(127);
		}
	}
	let top_left = above_buf[0];
	let above = &above_buf[1..];

	let mut out = [0u8; 32 * 32];
	let mut put = |c: usize, r: usize, v: u8| out[r * 32 + c] = v;
	match mode {
		pm::DC => {
			let sum_above: u32 = above[..bs].iter().map(|&v| u32::from(v)).sum();
			let sum_left: u32 = left[..bs].iter().map(|&v| u32::from(v)).sum();
			let n = bs as u32;
			let v = match (e.have_left, e.have_top) {
				(true, true) => (sum_above + sum_left + n) / (2 * n),
				(true, false) => (sum_left + n / 2) / n,
				(false, true) => (sum_above + n / 2) / n,
				(false, false) => 128,
			} as u8;
			for r in 0..bs {
				for c in 0..bs {
					put(c, r, v);
				}
			}
		}
		pm::V => {
			for r in 0..bs {
				for c in 0..bs {
					put(c, r, above[c]);
				}
			}
		}
		pm::H => {
			for r in 0..bs {
				for c in 0..bs {
					put(c, r, left[r]);
				}
			}
		}
		pm::TM => {
			for r in 0..bs {
				for c in 0..bs {
					let v = i32::from(left[r]) + i32::from(above[c]) - i32::from(top_left);
					put(c, r, v.clamp(0, 255) as u8);
				}
			}
		}
		_ if bs == 4 => predict_4x4(mode, above, top_left, &left, &mut put),
		pm::D207 => {
			let mut col = |c: usize, r: usize, v: u8| out[r * 32 + c] = v;
			for r in 0..bs - 1 {
				col(0, r, avg2(left[r], left[r + 1]));
			}
			col(0, bs - 1, left[bs - 1]);
			for r in 0..bs - 2 {
				col(1, r, avg3(left[r], left[r + 1], left[r + 2]));
			}
			col(1, bs - 2, avg3(left[bs - 2], left[bs - 1], left[bs - 1]));
			col(1, bs - 1, left[bs - 1]);
			for c in 0..bs - 2 {
				out[(bs - 1) * 32 + 2 + c] = left[bs - 1];
			}
			for r in (0..bs - 1).rev() {
				for c in 0..bs - 2 {
					out[r * 32 + 2 + c] = out[(r + 1) * 32 + c];
				}
			}
		}
		pm::D63 => {
			for c in 0..bs {
				out[c] = avg2(above[c], above[c + 1]);
				out[32 + c] = avg3(above[c], above[c + 1], above[c + 2]);
			}
			let mut size = bs - 2;
			let mut r = 2;
			while r < bs {
				for c in 0..size {
					out[r * 32 + c] = out[(r >> 1) + c];
					out[(r + 1) * 32 + c] = out[32 + (r >> 1) + c];
				}
				for c in size..bs {
					out[r * 32 + c] = above[bs - 1];
					out[(r + 1) * 32 + c] = above[bs - 1];
				}
				r += 2;
				size -= 1;
			}
		}
		pm::D45 => {
			let above_right = above[bs - 1];
			for c in 0..bs - 1 {
				out[c] = avg3(above[c], above[c + 1], above[c + 2]);
			}
			out[bs - 1] = above_right;
			let mut size = bs - 2;
			for r in 1..bs {
				for c in 0..size {
					out[r * 32 + c] = out[r + c];
				}
				for c in size..bs {
					out[r * 32 + c] = above_right;
				}
				size = size.saturating_sub(1);
			}
		}
		pm::D117 => {
			for c in 0..bs {
				out[c] = avg2(if c == 0 { top_left } else { above[c - 1] }, above[c]);
			}
			out[32] = avg3(left[0], top_left, above[0]);
			for c in 1..bs {
				out[32 + c] = avg3(if c == 1 { top_left } else { above[c - 2] }, above[c - 1], above[c]);
			}
			out[2 * 32] = avg3(top_left, left[0], left[1]);
			for r in 3..bs {
				out[r * 32] = avg3(left[r - 3], left[r - 2], left[r - 1]);
			}
			for r in 2..bs {
				for c in 1..bs {
					out[r * 32 + c] = out[(r - 2) * 32 + c - 1];
				}
			}
		}
		pm::D135 => {
			let mut border = [0u8; 63];
			for i in 0..bs - 2 {
				border[i] = avg3(left[bs - 3 - i], left[bs - 2 - i], left[bs - 1 - i]);
			}
			border[bs - 2] = avg3(top_left, left[0], left[1]);
			border[bs - 1] = avg3(left[0], top_left, above[0]);
			border[bs] = avg3(top_left, above[0], above[1]);
			for i in 0..bs - 2 {
				border[bs + 1 + i] = avg3(above[i], above[i + 1], above[i + 2]);
			}
			for r in 0..bs {
				out[r * 32..r * 32 + bs].copy_from_slice(&border[bs - 1 - r..2 * bs - 1 - r]);
			}
		}
		_ => {
			// D153.
			out[0] = avg2(top_left, left[0]);
			for r in 1..bs {
				out[r * 32] = avg2(left[r - 1], left[r]);
			}
			out[1] = avg3(left[0], top_left, above[0]);
			out[32 + 1] = avg3(top_left, left[0], left[1]);
			for r in 2..bs {
				out[r * 32 + 1] = avg3(left[r - 2], left[r - 1], left[r]);
			}
			for c in 0..bs - 2 {
				out[2 + c] = avg3(if c == 0 { top_left } else { above[c - 1] }, above[c], above[c + 1]);
			}
			for r in 1..bs {
				for c in 0..bs - 2 {
					out[r * 32 + 2 + c] = out[(r - 1) * 32 + c];
				}
			}
		}
	}
	for r in 0..bs {
		plane[(y + r) * stride + x..(y + r) * stride + x + bs]
			.copy_from_slice(&out[r * 32..r * 32 + bs]);
	}
}

/// The 4x4 directional predictors, which libvpx specialises (they use the
/// above-right pixels differently from the larger sizes).
fn predict_4x4(
	mode: u8,
	above: &[u8],
	top_left: u8,
	left: &[u8],
	put: &mut impl FnMut(usize, usize, u8),
) {
	let (a, b, c, d, e, f, g, h) =
		(above[0], above[1], above[2], above[3], above[4], above[5], above[6], above[7]);
	let (i, j, k, l) = (left[0], left[1], left[2], left[3]);
	let x = top_left;
	match mode {
		pm::D207 => {
			put(0, 0, avg2(i, j));
			put(2, 0, avg2(j, k));
			put(0, 1, avg2(j, k));
			put(2, 1, avg2(k, l));
			put(0, 2, avg2(k, l));
			put(1, 0, avg3(i, j, k));
			put(3, 0, avg3(j, k, l));
			put(1, 1, avg3(j, k, l));
			put(3, 1, avg3(k, l, l));
			put(1, 2, avg3(k, l, l));
			for (cc, rr) in [(3, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)] {
				put(cc, rr, l);
			}
		}
		pm::D63 => {
			put(0, 0, avg2(a, b));
			put(1, 0, avg2(b, c));
			put(0, 2, avg2(b, c));
			put(2, 0, avg2(c, d));
			put(1, 2, avg2(c, d));
			put(3, 0, avg2(d, e));
			put(2, 2, avg2(d, e));
			put(3, 2, avg2(e, f));
			put(0, 1, avg3(a, b, c));
			put(1, 1, avg3(b, c, d));
			put(0, 3, avg3(b, c, d));
			put(2, 1, avg3(c, d, e));
			put(1, 3, avg3(c, d, e));
			put(3, 1, avg3(d, e, f));
			put(2, 3, avg3(d, e, f));
			put(3, 3, avg3(e, f, g));
		}
		pm::D45 => {
			put(0, 0, avg3(a, b, c));
			for (cc, rr) in [(1, 0), (0, 1)] {
				put(cc, rr, avg3(b, c, d));
			}
			for (cc, rr) in [(2, 0), (1, 1), (0, 2)] {
				put(cc, rr, avg3(c, d, e));
			}
			for (cc, rr) in [(3, 0), (2, 1), (1, 2), (0, 3)] {
				put(cc, rr, avg3(d, e, f));
			}
			for (cc, rr) in [(3, 1), (2, 2), (1, 3)] {
				put(cc, rr, avg3(e, f, g));
			}
			for (cc, rr) in [(3, 2), (2, 3)] {
				put(cc, rr, avg3(f, g, h));
			}
			put(3, 3, h);
		}
		pm::D117 => {
			for (cc, rr) in [(0, 0), (1, 2)] {
				put(cc, rr, avg2(x, a));
			}
			for (cc, rr) in [(1, 0), (2, 2)] {
				put(cc, rr, avg2(a, b));
			}
			for (cc, rr) in [(2, 0), (3, 2)] {
				put(cc, rr, avg2(b, c));
			}
			put(3, 0, avg2(c, d));
			put(0, 3, avg3(k, j, i));
			put(0, 2, avg3(j, i, x));
			for (cc, rr) in [(0, 1), (1, 3)] {
				put(cc, rr, avg3(i, x, a));
			}
			for (cc, rr) in [(1, 1), (2, 3)] {
				put(cc, rr, avg3(x, a, b));
			}
			for (cc, rr) in [(2, 1), (3, 3)] {
				put(cc, rr, avg3(a, b, c));
			}
			put(3, 1, avg3(b, c, d));
		}
		pm::D135 => {
			put(0, 3, avg3(j, k, l));
			for (cc, rr) in [(1, 3), (0, 2)] {
				put(cc, rr, avg3(i, j, k));
			}
			for (cc, rr) in [(2, 3), (1, 2), (0, 1)] {
				put(cc, rr, avg3(x, i, j));
			}
			for (cc, rr) in [(3, 3), (2, 2), (1, 1), (0, 0)] {
				put(cc, rr, avg3(a, x, i));
			}
			for (cc, rr) in [(3, 2), (2, 1), (1, 0)] {
				put(cc, rr, avg3(b, a, x));
			}
			for (cc, rr) in [(3, 1), (2, 0)] {
				put(cc, rr, avg3(c, b, a));
			}
			put(3, 0, avg3(d, c, b));
		}
		_ => {
			// D153.
			for (cc, rr) in [(0, 0), (2, 1)] {
				put(cc, rr, avg2(i, x));
			}
			for (cc, rr) in [(0, 1), (2, 2)] {
				put(cc, rr, avg2(j, i));
			}
			for (cc, rr) in [(0, 2), (2, 3)] {
				put(cc, rr, avg2(k, j));
			}
			put(0, 3, avg2(l, k));
			put(3, 0, avg3(a, b, c));
			put(2, 0, avg3(x, a, b));
			for (cc, rr) in [(1, 0), (3, 1)] {
				put(cc, rr, avg3(i, x, a));
			}
			for (cc, rr) in [(1, 1), (3, 2)] {
				put(cc, rr, avg3(j, i, x));
			}
			for (cc, rr) in [(1, 2), (3, 3)] {
				put(cc, rr, avg3(k, j, i));
			}
			put(1, 3, avg3(l, k, j));
		}
	}
}
