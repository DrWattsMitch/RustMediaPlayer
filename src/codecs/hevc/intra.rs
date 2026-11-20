//! Intra sample prediction (ITU-T H.265 §8.4.4.2; FFmpeg
//! `hevc/pred_template.c`): reference sample substitution and smoothing,
//! then planar, DC or one of 33 angular modes.

use super::picture::Plane;

pub const MODE_PLANAR: u32 = 0;
pub const MODE_DC: u32 = 1;

const INTRA_PRED_ANGLE: [i32; 35] = [
	0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13,
	-9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];
const INV_ANGLE: [i32; 15] =
	[-4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096];

/// Reference samples: `[0]` is the corner p[-1][-1]; `left[1 + y]` is
/// p[-1][y] and `top[1 + x]` is p[x][-1], for y, x in 0..2N.
pub struct Refs {
	pub left: [u16; 129],
	pub top: [u16; 129],
}

/// Gathers and substitutes the reference samples of an `n`x`n` block at
/// component position (`x0`, `y0`). `avail(xn, yn)` tells whether the
/// component sample (xn, yn) may be used; it is asked once per `unit`
/// samples.
pub fn build_refs(
	plane: &Plane,
	x0: usize,
	y0: usize,
	n: usize,
	unit: usize,
	bit_depth: u32,
	mut avail: impl FnMut(isize, isize) -> bool,
) -> Refs {
	let mut r = Refs { left: [0; 129], top: [0; 129] };
	let n2 = 2 * n;
	// Availability per unit, in substitution scan order: left from the
	// bottom up, the corner, then the top from left to right.
	let units_side = n2.div_ceil(unit);
	let mut left_ok = [false; 64];
	let mut top_ok = [false; 64];
	let (xi, yi) = (x0 as isize, y0 as isize);
	let mut any = false;
	for u in 0..units_side {
		left_ok[u] = avail(xi - 1, yi + (u * unit) as isize);
		top_ok[u] = avail(xi + (u * unit) as isize, yi - 1);
		any |= left_ok[u] | top_ok[u];
	}
	let corner_ok = avail(xi - 1, yi - 1);
	any |= corner_ok;
	if !any {
		let v = 1u16 << (bit_depth - 1);
		r.left[..=n2].fill(v);
		r.top[..=n2].fill(v);
		return r;
	}
	let row = plane.idx(xi - 1, yi - 1);
	let s = plane.stride;
	let d = &plane.data;
	if corner_ok {
		r.left[0] = d[row];
	}
	for u in 0..units_side {
		if left_ok[u] {
			for k in u * unit..((u + 1) * unit).min(n2) {
				r.left[1 + k] = d[row + (k + 1) * s];
			}
		}
		if top_ok[u] {
			for k in u * unit..((u + 1) * unit).min(n2) {
				r.top[1 + k] = d[row + 1 + k];
			}
		}
	}
	// Substitution (8.4.4.2.2) along the scan bottom-left -> corner -> top-right.
	let ok_left = |k: usize| left_ok[k / unit];
	let ok_top = |k: usize| top_ok[k / unit];
	// First available sample in scan order.
	let mut prev: u16;
	if ok_left(n2 - 1) {
		prev = r.left[n2];
	} else {
		let mut found = None;
		for k in (0..n2).rev() {
			if ok_left(k) {
				found = Some(r.left[1 + k]);
				break;
			}
		}
		if found.is_none() && corner_ok {
			found = Some(r.left[0]);
		}
		if found.is_none() {
			for k in 0..n2 {
				if ok_top(k) {
					found = Some(r.top[1 + k]);
					break;
				}
			}
		}
		prev = found.unwrap_or(1 << (bit_depth - 1));
		r.left[n2] = prev;
	}
	for k in (0..n2 - 1).rev() {
		if ok_left(k) {
			prev = r.left[1 + k];
		} else {
			r.left[1 + k] = prev;
		}
	}
	if corner_ok {
		prev = r.left[0];
	} else {
		r.left[0] = prev;
	}
	r.top[0] = r.left[0];
	for k in 0..n2 {
		if ok_top(k) {
			prev = r.top[1 + k];
		} else {
			r.top[1 + k] = prev;
		}
	}
	r
}

/// Reference smoothing (8.4.4.2.3) for luma blocks.
pub fn filter_refs(r: &mut Refs, n: usize, mode: u32, strong_smoothing: bool, bit_depth: u32) {
	if mode == MODE_DC || n == 4 {
		return;
	}
	let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
	let thres = match n {
		8 => 7,
		16 => 1,
		_ => 0,
	};
	if min_dist <= thres {
		return;
	}
	let n2 = 2 * n;
	let corner = i32::from(r.left[0]);
	let top_end = i32::from(r.top[n2]);
	let left_end = i32::from(r.left[n2]);
	if strong_smoothing
		&& n == 32
		&& (corner + top_end - 2 * i32::from(r.top[n])).abs() < (1 << (bit_depth - 5))
		&& (corner + left_end - 2 * i32::from(r.left[n])).abs() < (1 << (bit_depth - 5))
	{
		for i in 1..64 {
			r.top[i] = (((64 - i as i32) * corner + i as i32 * top_end + 32) >> 6) as u16;
			r.left[i] = (((64 - i as i32) * corner + i as i32 * left_end + 32) >> 6) as u16;
		}
		return;
	}
	let (l, t) = (r.left, r.top);
	let c = i32::from(l[0]);
	let f0 = ((i32::from(l[1]) + 2 * c + i32::from(t[1]) + 2) >> 2) as u16;
	for i in 1..n2 {
		r.left[i] = ((i32::from(l[i - 1]) + 2 * i32::from(l[i]) + i32::from(l[i + 1]) + 2) >> 2) as u16;
		r.top[i] = ((i32::from(t[i - 1]) + 2 * i32::from(t[i]) + i32::from(t[i + 1]) + 2) >> 2) as u16;
	}
	r.left[0] = f0;
	r.top[0] = f0;
}

/// Writes the prediction of an `n`x`n` block at plane index `at`.
/// `edge_filters` enables the DC / horizontal / vertical boundary filters
/// (luma blocks smaller than 32x32).
pub fn predict(
	plane: &mut Plane,
	at: usize,
	n: usize,
	mode: u32,
	r: &Refs,
	edge_filters: bool,
	bit_depth: u32,
) {
	if mode == MODE_DC || !crate::simd::enabled() {
		return predict_scalar(plane, at, n, mode, r, edge_filters, bit_depth);
	}
	let s = plane.stride;
	if mode == MODE_PLANAR {
		super::simd::intra_planar(&r.top, &r.left, n, &mut plane.data, at, s);
		return;
	}
	let angle = INTRA_PRED_ANGLE[mode as usize];
	let vertical = mode >= 18;
	let (main, side) = if vertical { (&r.top, &r.left) } else { (&r.left, &r.top) };
	let mut ref_buf = [0i16; 3 * 64 + 1 + 16];
	let base = 64; // ref_buf[base + k] = ref[k], k in -n..=2n
	let len = if angle < 0 { n } else { 2 * n };
	for k in 0..=len {
		ref_buf[base + k] = main[k] as i16;
	}
	if angle < 0 {
		let inv = INV_ANGLE[(mode - 11) as usize];
		let last = (n as i32 * angle) >> 5;
		if last < -1 {
			let mut x = -1;
			while x >= last {
				let idx = (((x * inv) + 128) >> 8) as usize;
				ref_buf[(base as i32 + x) as usize] = side[idx] as i16;
				x -= 1;
			}
		}
	}
	super::simd::intra_angular(&ref_buf, base, n, angle, !vertical, &mut plane.data, at, s);
	if edge_filters && angle == 0 && n < 32 {
		// Mode 26 (vertical) or 10 (horizontal): gradient on the first
		// column / row.
		let max = (1i32 << bit_depth) - 1;
		for k in 0..n {
			let v = i32::from(main[1]) + ((i32::from(side[1 + k]) - i32::from(side[0])) >> 1);
			let o = if vertical { at + k * s } else { at + k };
			plane.data[o] = v.clamp(0, max) as u16;
		}
	}
}

/// [`predict`] without SIMD (the reference for its kernels).
pub fn predict_scalar(
	plane: &mut Plane,
	at: usize,
	n: usize,
	mode: u32,
	r: &Refs,
	edge_filters: bool,
	bit_depth: u32,
) {
	let s = plane.stride;
	let d = &mut plane.data;
	let max = (1i32 << bit_depth) - 1;
	match mode {
		MODE_PLANAR => {
			let shift = n.trailing_zeros() + 1;
			let tr = i32::from(r.top[1 + n]);
			let bl = i32::from(r.left[1 + n]);
			for y in 0..n {
				let row = at + y * s;
				for x in 0..n {
					let v = (n - 1 - x) as i32 * i32::from(r.left[1 + y])
						+ (x + 1) as i32 * tr
						+ (n - 1 - y) as i32 * i32::from(r.top[1 + x])
						+ (y + 1) as i32 * bl
						+ n as i32;
					d[row + x] = (v >> shift) as u16;
				}
			}
		}
		MODE_DC => {
			let mut sum = n as i32;
			for k in 0..n {
				sum += i32::from(r.top[1 + k]) + i32::from(r.left[1 + k]);
			}
			let dc = sum >> (n.trailing_zeros() + 1);
			for y in 0..n {
				d[at + y * s..at + y * s + n].fill(dc as u16);
			}
			if edge_filters && n < 32 {
				d[at] = ((i32::from(r.left[1]) + 2 * dc + i32::from(r.top[1]) + 2) >> 2) as u16;
				for x in 1..n {
					d[at + x] = ((i32::from(r.top[1 + x]) + 3 * dc + 2) >> 2) as u16;
				}
				for y in 1..n {
					d[at + y * s] = ((i32::from(r.left[1 + y]) + 3 * dc + 2) >> 2) as u16;
				}
			}
		}
		_ => {
			let angle = INTRA_PRED_ANGLE[mode as usize];
			let mut ref_buf = [0i32; 3 * 64 + 1];
			let base = 64; // ref_buf[base + k] = ref[k], k in -n..=2n
			let vertical = mode >= 18;
			let (main, side) = if vertical { (&r.top, &r.left) } else { (&r.left, &r.top) };
			for k in 0..=n {
				ref_buf[base + k] = i32::from(main[k]);
			}
			if angle < 0 {
				let inv = INV_ANGLE[(mode - 11) as usize];
				let last = (n as i32 * angle) >> 5;
				if last < -1 {
					let mut x = -1;
					while x >= last {
						let idx = (((x * inv) + 128) >> 8) as usize;
						ref_buf[(base as i32 + x) as usize] = i32::from(side[idx]);
						x -= 1;
					}
				}
			} else {
				for k in n + 1..=2 * n {
					ref_buf[base + k] = i32::from(main[k]);
				}
			}
			for y in 0..n {
				let pos = (y as i32 + 1) * angle;
				let idx = pos >> 5;
				let fact = pos & 31;
				for x in 0..n {
					let i = (base as i32 + x as i32 + idx + 1) as usize;
					let v = if fact != 0 {
						((32 - fact) * ref_buf[i] + fact * ref_buf[i + 1] + 16) >> 5
					} else {
						ref_buf[i]
					};
					// Transposed for horizontal modes.
					let o = if vertical { at + y * s + x } else { at + x * s + y };
					d[o] = v as u16;
				}
			}
			if edge_filters && angle == 0 && n < 32 {
				// Mode 26 (vertical) or 10 (horizontal): gradient on the
				// first column / row.
				for k in 0..n {
					let v = i32::from(main[1]) + ((i32::from(side[1 + k]) - i32::from(side[0])) >> 1);
					let o = if vertical { at + k * s } else { at + k };
					d[o] = v.clamp(0, max) as u16;
				}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The vector planar and angular kernels give the scalar prediction.
	#[test]
	fn simd_prediction_matches_scalar() {
		let mut seed = 0x2545_f491u32;
		let mut next = move || {
			seed ^= seed << 13;
			seed ^= seed >> 17;
			seed ^= seed << 5;
			seed
		};
		for bit_depth in [8u32, 10, 12] {
			let max = (1u32 << bit_depth) - 1;
			for log2 in 2..=5 {
				let n = 1usize << log2;
				for round in 0..6 {
					let mut r = Refs { left: [0; 129], top: [0; 129] };
					for k in 0..=2 * n {
						// Smooth, noisy and extreme references.
						let v = match round % 3 {
							0 => next() & max,
							1 => (k as u32 * 7 + 100) & max,
							_ => {
								if next() & 1 == 0 {
									0
								} else {
									max
								}
							}
						};
						r.left[k] = v as u16;
						r.top[k] = (next() & max) as u16;
					}
					r.top[0] = r.left[0];
					for mode in 0..35 {
						for edge in [false, true] {
							let mut a = Plane::new(80, 80, 8, 7);
							let mut b = a.clone();
							let at = a.idx(4, 4);
							predict(&mut a, at, n, mode, &r, edge, bit_depth);
							predict_scalar(&mut b, at, n, mode, &r, edge, bit_depth);
							assert!(a.data == b.data, "mode {mode} n {n} bd {bit_depth}");
						}
					}
				}
			}
		}
	}

	#[test]
	fn flat_references_predict_flat_blocks() {
		let mut p = Plane::new(64, 64, 8, 0);
		let r = Refs { left: [100; 129], top: [100; 129] };
		for mode in 0..35 {
			let at = p.idx(8, 8);
			predict(&mut p, at, 8, mode, &r, true, 8);
			for y in 0..8 {
				for x in 0..8 {
					assert_eq!(p.get(8 + x, 8 + y), 100, "mode {mode}");
				}
			}
		}
	}
}
