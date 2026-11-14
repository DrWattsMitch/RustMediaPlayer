//! Fractional-sample interpolation and weighted sample prediction (ITU-T
//! H.265 §8.5.3.3; FFmpeg `hevc/dsp_template.c` `put_hevc_qpel`/`epel`).
//!
//! Predictions are produced at 14-bit intermediate precision into `i16`
//! buffers, then combined (uni, bi, or explicitly weighted) into the
//! picture. Reference planes are edge-padded, so block positions are only
//! clamped to stay inside the padding.

use super::picture::Plane;

const LUMA_FILTER: [[i32; 8]; 4] = [
	[0, 0, 0, 64, 0, 0, 0, 0],
	[-1, 4, -10, 58, 17, -5, 1, 0],
	[-1, 4, -11, 40, 40, -11, 4, -1],
	[0, 1, -5, 17, 58, -10, 4, -1],
];
const CHROMA_FILTER: [[i32; 4]; 8] = [
	[0, 64, 0, 0],
	[-2, 58, 10, -2],
	[-4, 54, 16, -2],
	[-6, 46, 28, -4],
	[-4, 36, 36, -4],
	[-4, 28, 46, -6],
	[-2, 16, 54, -4],
	[-2, 10, 58, -2],
];

pub const MAX_PB: usize = 64;

/// Interpolates a `w`x`h` block of `plane` whose top-left integer sample is
/// (`x`, `y`) with fractional offset (`fx`, `fy`) (quarter samples for
/// luma, eighth for chroma) into `out` (stride `w`), 14-bit precision.
/// `tmp` holds intermediate rows (at least [`TMP_LEN`]).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn interpolate(
	plane: &Plane,
	x: isize,
	y: isize,
	fx: usize,
	fy: usize,
	w: usize,
	h: usize,
	chroma: bool,
	bit_depth: u32,
	out: &mut [i16],
	tmp: &mut [i16],
) {
	if !w.is_multiple_of(4) || !crate::simd::enabled() {
		return interpolate_scalar(plane, x, y, fx, fy, w, h, chroma, bit_depth, out, tmp);
	}
	let (x, y) = clamp_position(plane, x, y, w, h, if chroma { 4 } else { 8 });
	let base = plane.idx(x, y);
	if chroma {
		let (fh, fv) = (CHROMA_FILTER[fx], CHROMA_FILTER[fy]);
		super::simd::interp_chroma(
			&plane.data,
			plane.stride,
			base,
			&fh,
			&fv,
			fx != 0,
			fy != 0,
			w,
			h,
			bit_depth,
			out,
			tmp,
		);
	} else {
		let (fh, fv) = (LUMA_FILTER[fx], LUMA_FILTER[fy]);
		super::simd::interp_luma(
			&plane.data,
			plane.stride,
			base,
			&fh,
			&fv,
			fx != 0,
			fy != 0,
			w,
			h,
			bit_depth,
			out,
			tmp,
		);
	}
}

/// Keeps a block (with its `taps`-tap filter support) inside the padding;
/// the padding replicates the edges, so this changes no sample.
#[inline]
fn clamp_position(
	plane: &Plane,
	x: isize,
	y: isize,
	w: usize,
	h: usize,
	taps: usize,
) -> (isize, isize) {
	let half = (taps / 2 - 1) as isize;
	let pad = plane.pad as isize;
	(
		x.clamp(-pad + half, plane.width as isize + pad - w as isize - taps as isize),
		y.clamp(-pad + half, plane.height as isize + pad - h as isize - taps as isize),
	)
}

/// Writes a uni-directional prediction (default or explicit weighting).
#[inline]
pub fn put_uni(
	dst: &mut Plane,
	at: usize,
	src: &[i16],
	w: usize,
	h: usize,
	bit_depth: u32,
	wp: Option<(Wp, u32)>,
) {
	if wp.is_none() && w.is_multiple_of(4) && crate::simd::enabled() {
		let s = dst.stride;
		super::simd::put_uni(&mut dst.data, s, at, src, w, h, bit_depth);
	} else {
		put_uni_scalar(dst, at, src, w, h, bit_depth, wp);
	}
}

/// Writes a bi-directional prediction (average or explicit weighting).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn put_bi(
	dst: &mut Plane,
	at: usize,
	a: &[i16],
	b: &[i16],
	w: usize,
	h: usize,
	bit_depth: u32,
	wp: Option<(Wp, Wp, u32)>,
) {
	if wp.is_none() && w.is_multiple_of(4) && crate::simd::enabled() {
		let s = dst.stride;
		super::simd::put_bi(&mut dst.data, s, at, a, b, w, h, bit_depth);
	} else {
		put_bi_scalar(dst, at, a, b, w, h, bit_depth, wp);
	}
}

/// Scratch rows for the separable (horizontal then vertical) case.
pub const TMP_LEN: usize = (MAX_PB + 7) * MAX_PB;

crate::multiversion! {
/// Scalar [`interpolate`].
#[allow(clippy::too_many_arguments)]
pub fn interpolate_scalar(
	plane: &Plane,
	x: isize,
	y: isize,
	fx: usize,
	fy: usize,
	w: usize,
	h: usize,
	chroma: bool,
	bit_depth: u32,
	out: &mut [i16],
	tmp: &mut [i16],
) {
	if chroma {
		let f = |i: usize| CHROMA_FILTER[i];
		interp::<4>(plane, x, y, f(fx), f(fy), fx != 0, fy != 0, w, h, bit_depth, out, tmp);
	} else {
		let f = |i: usize| LUMA_FILTER[i];
		interp::<8>(plane, x, y, f(fx), f(fy), fx != 0, fy != 0, w, h, bit_depth, out, tmp);
	}
}
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn interp<const T: usize>(
	plane: &Plane,
	x: isize,
	y: isize,
	fh: [i32; T],
	fv: [i32; T],
	frac_x: bool,
	frac_y: bool,
	w: usize,
	h: usize,
	bit_depth: u32,
	out: &mut [i16],
	tmp: &mut [i16],
) {
	let half = T / 2 - 1; // samples before the current one
	let (x, y) = clamp_position(plane, x, y, w, h, T);
	let s = plane.stride;
	let d = &plane.data[..];
	let shift1 = bit_depth - 8;
	let out = &mut out[..w * h];
	if !frac_x && !frac_y {
		let shift3 = 14 - bit_depth;
		for (r, o) in out.chunks_exact_mut(w).enumerate() {
			let src = plane.idx(x, y + r as isize);
			for (o, &v) in o.iter_mut().zip(&d[src..src + w]) {
				*o = (v << shift3) as i16;
			}
		}
		return;
	}
	if !frac_y {
		for (r, o) in out.chunks_exact_mut(w).enumerate() {
			let src = plane.idx(x - half as isize, y + r as isize);
			hfilter::<T>(&d[src..src + w + T - 1], &fh, shift1, o);
		}
		return;
	}
	if !frac_x {
		for (r, o) in out.chunks_exact_mut(w).enumerate() {
			let src = plane.idx(x, y + r as isize - half as isize);
			let mut acc = [0i32; MAX_PB];
			let acc = &mut acc[..w];
			for (t, &f) in fv.iter().enumerate() {
				let row = &d[src + t * s..src + t * s + w];
				for (a, &v) in acc.iter_mut().zip(row) {
					*a += f * i32::from(v);
				}
			}
			for (o, &a) in o.iter_mut().zip(acc.iter()) {
				*o = (a >> shift1) as i16;
			}
		}
		return;
	}
	// Separable: horizontal into `tmp` (h + T - 1 rows), then vertical.
	let rows = h + T - 1;
	let tmp = &mut tmp[..rows * w];
	for (r, t) in tmp.chunks_exact_mut(w).enumerate() {
		let src = plane.idx(x - half as isize, y + r as isize - half as isize);
		hfilter::<T>(&d[src..src + w + T - 1], &fh, shift1, t);
	}
	for (r, o) in out.chunks_exact_mut(w).enumerate() {
		let mut acc = [0i32; MAX_PB];
		let acc = &mut acc[..w];
		for (t, &f) in fv.iter().enumerate() {
			let row = &tmp[(r + t) * w..(r + t + 1) * w];
			for (a, &v) in acc.iter_mut().zip(row) {
				*a += f * i32::from(v);
			}
		}
		for (o, &a) in o.iter_mut().zip(acc.iter()) {
			*o = (a >> 6) as i16;
		}
	}
}

/// One row of a horizontal `T`-tap filter: `src` holds `out.len() + T - 1`
/// samples.
#[inline(always)]
fn hfilter<const T: usize>(src: &[u16], f: &[i32; T], shift: u32, out: &mut [i16]) {
	let w = out.len();
	let mut acc = [0i32; MAX_PB];
	let acc = &mut acc[..w];
	for (t, &f) in f.iter().enumerate() {
		for (a, &v) in acc.iter_mut().zip(&src[t..t + w]) {
			*a += f * i32::from(v);
		}
	}
	for (o, &a) in out.iter_mut().zip(acc.iter()) {
		*o = (a >> shift) as i16;
	}
}

/// Explicit weighting parameters for one list.
#[derive(Clone, Copy)]
pub struct Wp {
	pub weight: i32,
	pub offset: i32,
}

crate::multiversion! {
/// Scalar [`put_uni`].
pub fn put_uni_scalar(
	dst: &mut Plane,
	at: usize,
	src: &[i16],
	w: usize,
	h: usize,
	bit_depth: u32,
	wp: Option<(Wp, u32)>,
) {
	let s = dst.stride;
	let max = (1i32 << bit_depth) - 1;
	let shift = 14 - bit_depth;
	match wp {
		None => {
			let add = if shift > 0 { 1 << (shift - 1) } else { 0 };
			for r in 0..h {
				let row = &mut dst.data[at + r * s..at + r * s + w];
				for (o, &v) in row.iter_mut().zip(&src[r * w..r * w + w]) {
					*o = ((i32::from(v) + add) >> shift).clamp(0, max) as u16;
				}
			}
		}
		Some((p, denom)) => {
			let log2wd = denom + shift;
			for r in 0..h {
				let row = &mut dst.data[at + r * s..at + r * s + w];
				for (o, &v) in row.iter_mut().zip(&src[r * w..r * w + w]) {
					let v = i32::from(v);
					let x = if log2wd >= 1 {
						((v * p.weight + (1 << (log2wd - 1))) >> log2wd) + p.offset
					} else {
						v * p.weight + p.offset
					};
					*o = x.clamp(0, max) as u16;
				}
			}
		}
	}
}
}

crate::multiversion! {
/// Scalar [`put_bi`].
#[allow(clippy::too_many_arguments)]
pub fn put_bi_scalar(
	dst: &mut Plane,
	at: usize,
	a: &[i16],
	b: &[i16],
	w: usize,
	h: usize,
	bit_depth: u32,
	wp: Option<(Wp, Wp, u32)>,
) {
	let s = dst.stride;
	let max = (1i32 << bit_depth) - 1;
	let shift = 15 - bit_depth;
	match wp {
		None => {
			let add = 1 << (shift - 1);
			for r in 0..h {
				let row = &mut dst.data[at + r * s..at + r * s + w];
				let (ra, rb) = (&a[r * w..r * w + w], &b[r * w..r * w + w]);
				for ((o, &x), &y) in row.iter_mut().zip(ra).zip(rb) {
					*o = ((i32::from(x) + i32::from(y) + add) >> shift).clamp(0, max) as u16;
				}
			}
		}
		Some((p0, p1, denom)) => {
			let log2wd = denom + 14 - bit_depth;
			let round = (p0.offset + p1.offset + 1) << log2wd;
			for r in 0..h {
				let row = &mut dst.data[at + r * s..at + r * s + w];
				let (ra, rb) = (&a[r * w..r * w + w], &b[r * w..r * w + w]);
				for ((o, &x), &y) in row.iter_mut().zip(ra).zip(rb) {
					let v = (i32::from(x) * p0.weight + i32::from(y) * p1.weight + round) >> (log2wd + 1);
					*o = v.clamp(0, max) as u16;
				}
			}
		}
	}
}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn full_pel_copy_round_trips() {
		let mut p = Plane::new(16, 16, 80, 0);
		for y in 0..16 {
			for x in 0..16 {
				let i = p.idx(x, y);
				p.data[i] = (x * 10 + y) as u16;
			}
		}
		p.extend_edges();
		let mut buf = [0i16; 64];
		interpolate(&p, 2, 3, 0, 0, 8, 8, false, 8, &mut buf, &mut [0; TMP_LEN]);
		let mut out = Plane::new(16, 16, 8, 0);
		let at = out.idx(0, 0);
		put_uni(&mut out, at, &buf, 8, 8, 8, None);
		assert_eq!(out.get(0, 0), p.get(2, 3));
		assert_eq!(out.get(7, 7), p.get(9, 10));
	}

	#[test]
	fn half_pel_of_flat_plane_is_flat() {
		let mut p = Plane::new(16, 16, 80, 77);
		p.extend_edges();
		let mut buf = [0i16; 64];
		interpolate(&p, 0, 0, 2, 2, 8, 8, false, 8, &mut buf, &mut [0; TMP_LEN]);
		assert!(buf.iter().all(|&v| v == 77 << 6));
	}
}
