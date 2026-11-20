//! SIMD kernels for HEVC motion compensation, written against the portable
//! `crate::simd` types (SSE2 / AVX2 / NEON / scalar). Each is bit-exact with
//! the scalar code in `mc.rs` it replaces (differential tests below).
//!
//! Interpolation filters use `madd` on interleaved neighbour pairs: for
//! eight outputs `o[i] = sum_t f[t] * s[i + t]`, the pairs
//! `(s[i + 2p], s[i + 2p + 1])` times `(f[2p], f[2p + 1])` summed over `p`
//! give four 32-bit outputs per `madd`, exactly, for any 16-bit input.

use crate::multiversion;
use crate::simd::{I16x8, I32x4};

/// Coefficient pairs `(f[2p], f[2p + 1])` splatted across a vector.
#[inline(always)]
fn pairs<const T: usize>(f: &[i32; T]) -> [I16x8; 4] {
	let mut out = [I16x8::splat(0); 4];
	for p in 0..T / 2 {
		let (a, b) = (f[2 * p] as i16, f[2 * p + 1] as i16);
		out[p] = I16x8::from_array([a, b, a, b, a, b, a, b]);
	}
	out
}

/// Low 16 bits of each lane, sign-extended (the `as i16` wrap of the
/// scalar code), so the saturating pack below is exact.
#[inline(always)]
fn wrap16(v: I32x4) -> I32x4 {
	v.shl(16).shr(16)
}

/// Eight outputs of a `T`-tap filter over `T` input vectors (the input
/// shifted by 0..T samples, or T consecutive rows).
#[inline(always)]
fn taps8<const T: usize>(v: &[I16x8; 8], c: &[I16x8; 4], shift: u32, wrap: bool) -> I16x8 {
	let mut lo = I32x4::splat(0);
	let mut hi = I32x4::splat(0);
	for p in 0..T / 2 {
		lo = lo.add(v[2 * p].zip_lo(v[2 * p + 1]).madd(c[p]));
		hi = hi.add(v[2 * p].zip_hi(v[2 * p + 1]).madd(c[p]));
	}
	let (mut lo, mut hi) = (lo.shr(shift), hi.shr(shift));
	if wrap {
		lo = wrap16(lo);
		hi = wrap16(hi);
	}
	lo.pack_i16(hi)
}

/// Stores the first `n` (4 or 8) lanes.
#[inline(always)]
fn store_n(v: I16x8, out: &mut [i16], n: usize) {
	if n == 8 {
		v.store(out);
	} else {
		out[..n].copy_from_slice(&v.to_array()[..n]);
	}
}

#[inline(always)]
fn store_n_u16(v: I16x8, out: &mut [u16], n: usize) {
	if n == 8 {
		v.store_u16(out);
	} else {
		let a = v.to_array();
		for (o, &x) in out[..n].iter_mut().zip(&a[..n]) {
			*o = x as u16;
		}
	}
}

/// Horizontal filter of one row: `src` holds `out.len() + T - 1` samples
/// (and readable samples up to 8 past the last one), `out.len()` is a
/// multiple of 4.
#[inline(always)]
fn h_row<const T: usize>(src: &[u16], c: &[I16x8; 4], shift: u32, out: &mut [i16]) {
	let w = out.len();
	let mut x = 0;
	while x < w {
		// One 16-sample window per 8 outputs: the 8 shifted loads below are
		// then provably in bounds (no per-load checks).
		let win = &src[x..x + 16];
		let v: [I16x8; 8] =
			std::array::from_fn(
				|t| if t < T { I16x8::load_u16(&win[t..t + 8]) } else { I16x8::splat(0) },
			);
		store_n(taps8::<T>(&v, c, shift, false), &mut out[x..], (w - x).min(8));
		x += 8;
	}
}

/// Interpolation of a block whose width is a multiple of 4; same contract
/// as `mc::interp` after clamping (`base` is the index of the block's
/// top-left integer sample in `data`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn interp<const T: usize>(
	data: &[u16],
	stride: usize,
	base: usize,
	fh: &[i32; T],
	fv: &[i32; T],
	frac_x: bool,
	frac_y: bool,
	w: usize,
	h: usize,
	bit_depth: u32,
	out: &mut [i16],
	tmp: &mut [i16],
) {
	let half = T / 2 - 1;
	let shift1 = bit_depth - 8;
	let out = &mut out[..w * h];
	if !frac_x && !frac_y {
		let shift3 = 14 - bit_depth;
		for (r, o) in out.chunks_exact_mut(w).enumerate() {
			let row = &data[base + r * stride..];
			let mut x = 0;
			while x < w {
				store_n(I16x8::load_u16(&row[x..]).shl(shift3), &mut o[x..], (w - x).min(8));
				x += 8;
			}
		}
		return;
	}
	let ch = pairs(fh);
	let cv = pairs(fv);
	if !frac_y {
		for (r, o) in out.chunks_exact_mut(w).enumerate() {
			let at = base + r * stride - half;
			h_row::<T>(&data[at..], &ch, shift1, o);
		}
		return;
	}
	// Rows are read in whole groups of 8 (the padding holds the extra 4 of
	// a 4-wide block); slicing each row once removes the per-load checks.
	let w8 = (w + 7) & !7;
	if !frac_x {
		for (r, o) in out.chunks_exact_mut(w).enumerate() {
			let at = base + (r * stride) - half * stride;
			let rows: [&[u16]; 8] =
				std::array::from_fn(
					|t| if t < T { &data[at + t * stride..at + t * stride + w8] } else { &[] },
				);
			for g in 0..w8 / 8 {
				let x = g * 8;
				let v: [I16x8; 8] = std::array::from_fn(|t| {
					if t < T { I16x8::load_u16(&rows[t][x..x + 8]) } else { I16x8::splat(0) }
				});
				store_n(taps8::<T>(&v, &cv, shift1, false), &mut o[x..], (w - x).min(8));
			}
		}
		return;
	}
	let rows = h + T - 1;
	for (r, t) in tmp[..rows * w].chunks_exact_mut(w).enumerate() {
		let at = base + r * stride - half * stride - half;
		h_row::<T>(&data[at..], &ch, shift1, t);
	}
	for (r, o) in out.chunks_exact_mut(w).enumerate() {
		let rows: [&[i16]; 8] =
			std::array::from_fn(|t| if t < T { &tmp[(r + t) * w..(r + t) * w + w8] } else { &[] });
		for g in 0..w8 / 8 {
			let x = g * 8;
			let v: [I16x8; 8] =
				std::array::from_fn(
					|t| if t < T { I16x8::load(&rows[t][x..x + 8]) } else { I16x8::splat(0) },
				);
			store_n(taps8::<T>(&v, &cv, 6, true), &mut o[x..], (w - x).min(8));
		}
	}
}

multiversion! {
	/// 8-tap luma interpolation (`w.is_multiple_of(4)`; `out` and `tmp` readable 8
	/// past each row).
	#[allow(clippy::too_many_arguments)]
	pub fn interp_luma(data: &[u16], stride: usize, base: usize, fh: &[i32; 8], fv: &[i32; 8], frac_x: bool, frac_y: bool, w: usize, h: usize, bit_depth: u32, out: &mut [i16], tmp: &mut [i16]) {
		interp::<8>(data, stride, base, fh, fv, frac_x, frac_y, w, h, bit_depth, out, tmp)
	}
}

multiversion! {
	/// 4-tap chroma interpolation (as [`interp_luma`]).
	#[allow(clippy::too_many_arguments)]
	pub fn interp_chroma(data: &[u16], stride: usize, base: usize, fh: &[i32; 4], fv: &[i32; 4], frac_x: bool, frac_y: bool, w: usize, h: usize, bit_depth: u32, out: &mut [i16], tmp: &mut [i16]) {
		interp::<4>(data, stride, base, fh, fv, frac_x, frac_y, w, h, bit_depth, out, tmp)
	}
}

multiversion! {
	/// Default-weighted uni-prediction store (`w.is_multiple_of(4)`, `src` readable
	/// 4 past the block when `!w.is_multiple_of(8)`).
	///
	/// `(v + round) >> shift` clamped to `[0, max]`: the saturating add is
	/// exact because `max << shift` stays below `i16::MAX` for every bit
	/// depth up to 12, so any saturated sum clamps to `max` (or 0) anyway.
	pub fn put_uni(dst: &mut [u16], stride: usize, at: usize, src: &[i16], w: usize, h: usize, bit_depth: u32) {
		let shift = 14 - bit_depth;
		let round = I16x8::splat(if shift > 0 { 1 << (shift - 1) } else { 0 });
		let (zero, max) = (I16x8::splat(0), I16x8::splat(((1i32 << bit_depth) - 1) as i16));
		let w8 = (w + 7) & !7;
		for r in 0..h {
			let row = &mut dst[at + r * stride..at + r * stride + w8];
			let s = &src[r * w..r * w + w8];
			for g in 0..w8 / 8 {
				let x = g * 8;
				let v = I16x8::load(&s[x..x + 8]).saturating_add(round).shr(shift).clamp(zero, max);
				store_n_u16(v, &mut row[x..x + 8], (w - x).min(8));
			}
		}
	}
}

multiversion! {
	/// Default-weighted bi-prediction store (as [`put_uni`]); exact for the
	/// same reason as [`put_uni`] (`max << (15 - bit_depth)` < `i16::MAX`).
	#[allow(clippy::too_many_arguments)]
	pub fn put_bi(dst: &mut [u16], stride: usize, at: usize, a: &[i16], b: &[i16], w: usize, h: usize, bit_depth: u32) {
		let shift = 15 - bit_depth;
		let round = I16x8::splat(1 << (shift - 1));
		let (zero, max) = (I16x8::splat(0), I16x8::splat(((1i32 << bit_depth) - 1) as i16));
		let w8 = (w + 7) & !7;
		for r in 0..h {
			let row = &mut dst[at + r * stride..at + r * stride + w8];
			let (ra, rb) = (&a[r * w..r * w + w8], &b[r * w..r * w + w8]);
			for g in 0..w8 / 8 {
				let x = g * 8;
				let v = I16x8::load(&ra[x..x + 8]).saturating_add(I16x8::load(&rb[x..x + 8])).saturating_add(round);
				store_n_u16(v.shr(shift).clamp(zero, max), &mut row[x..x + 8], (w - x).min(8));
			}
		}
	}
}

multiversion! {
	/// Edge-offset SAO of `n` samples (`n.is_multiple_of(8)`) starting at `at`; see
	/// `filter::sao_edge_run`.
	#[allow(clippy::too_many_arguments)]
	pub fn sao_edge(src: &[u16], dst: &mut [u16], at: usize, n: usize, d0: isize, d1: isize, offs: &[i16; 5], max: i32) {
		let (zero, maxv) = (I16x8::splat(0), I16x8::splat(max as i16));
		let two = I16x8::splat(2);
		let o: [I16x8; 5] = std::array::from_fn(|k| I16x8::splat(offs[k]));
		let (a0, b0) = (at.wrapping_add_signed(d0), at.wrapping_add_signed(d1));
		let mut i = 0;
		while i < n {
			let cur = I16x8::load_u16(&src[at + i..]);
			let a = I16x8::load_u16(&src[a0 + i..]);
			let b = I16x8::load_u16(&src[b0 + i..]);
			// sign(cur - x) as (x < cur) - (cur < x), masks being -1.
			let e = two.add(cur.lt(a)).sub(a.lt(cur)).add(cur.lt(b)).sub(b.lt(cur));
			let mut off = o[0];
			for k in 1..5 {
				off = e.eq(I16x8::splat(k as i16)).select(o[k], off);
			}
			cur.add(off).clamp(zero, maxv).store_u16(&mut dst[at + i..]);
			i += 8;
		}
	}
}

multiversion! {
	/// Band-offset SAO of a row of `n` samples (`n.is_multiple_of(8)`): bands
	/// `pos..pos + 4` (mod 32) get `offs[0..4]`.
	pub fn sao_band(row: &mut [u16], n: usize, shift: u32, pos: u8, offs: &[i16; 4], max: i32) {
		let (zero, maxv) = (I16x8::splat(0), I16x8::splat(max as i16));
		let (p, m31) = (I16x8::splat(i16::from(pos)), I16x8::splat(31));
		let o: [I16x8; 4] = std::array::from_fn(|k| I16x8::splat(offs[k]));
		let mut i = 0;
		while i < n {
			let v = I16x8::load_u16(&row[i..]);
			let k = v.shr(shift).sub(p).and(m31);
			let mut off = zero;
			for (j, oj) in o.iter().enumerate() {
				off = k.eq(I16x8::splat(j as i16)).select(*oj, off);
			}
			v.add(off).clamp(zero, maxv).store_u16(&mut row[i..]);
			i += 8;
		}
	}
}

/// One 1-D inverse DCT pass over groups of four columns: column `x` of
/// `dst` is the transform of column `x` of `src` (both `n`x`n`, row-major),
/// using its first `nz` coefficients, for `x < cols` (`cols % 4 == 0`).
#[inline(always)]
fn idct_cols(src: &[i32], dst: &mut [i32], n: usize, cols: usize, nz: usize, shift: u32) {
	use super::transform::DCT_MATRIX;
	let step = 32 / n;
	let half = n / 2;
	let add = I32x4::splat(1 << (shift - 1));
	let (lo, hi) = (I32x4::splat(-32768), I32x4::splat(32767));
	let mut x = 0;
	while x < cols {
		let mut e = [I32x4::splat(0); 16];
		let mut o = [I32x4::splat(0); 16];
		for k in 0..nz {
			let c = I32x4::load(&src[k * n + x..]);
			if !c.any() {
				continue;
			}
			let row = &DCT_MATRIX[k * step];
			let acc = if k % 2 == 1 { &mut o } else { &mut e };
			for i in 0..half {
				acc[i] = acc[i].add(c.mul(I32x4::splat(i32::from(row[i]))));
			}
		}
		for i in 0..half {
			let (a, b) = (e[i].add(add), o[i]);
			a.add(b).shr(shift).clamp(lo, hi).store(&mut dst[i * n + x..]);
			a.sub(b).shr(shift).clamp(lo, hi).store(&mut dst[(n - 1 - i) * n + x..]);
		}
		x += 4;
	}
}

multiversion! {
	/// [`super::transform::idct`] for `n >= 4` with at least one AC
	/// coefficient (bit-exact: 32-bit lanes, same operation order).
	#[allow(clippy::too_many_arguments)]
	pub fn idct(coeffs: &mut [i32], n: usize, max_x: usize, max_y: usize, bit_depth: u32, tmp: &mut [i32], tmp2: &mut [i32]) {
		let (tmp, tmp2) = (&mut tmp[..n * n], &mut tmp2[..n * n]);
		let cols = (max_x + 4) & !3;
		// Stage 1: columns, shift 7.
		idct_cols(coeffs, tmp, n, cols.min(n), (max_y + 1).min(n), 7);
		// Stage 2 on the transpose: rows become columns.
		for y in 0..n {
			for x in 0..cols.min(n) {
				tmp2[x * n + y] = tmp[y * n + x];
			}
		}
		idct_cols(tmp2, tmp, n, n, (max_x + 1).min(n), 20 - bit_depth);
		for y in 0..n {
			for x in 0..n {
				coeffs[y * n + x] = tmp[x * n + y];
			}
		}
	}
}

/// Luma deblocking of 8 lines across one edge (two 4-line segments): `v`
/// holds p3, p2, p1, p0, q0, q1, q2, q3, lane `i` being line `i`. Same
/// decisions and arithmetic as `filter::luma_edge`; valid for bit depths up
/// to 10 (the weak filter's `9 * (q0 - p0)` must fit 16 bits).
#[inline(always)]
fn luma_lines(
	v: &mut [I16x8; 8],
	beta: [i32; 2],
	tc: [i32; 2],
	no_p: [bool; 2],
	no_q: [bool; 2],
	max: i32,
) {
	let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
	let two = I16x8::splat(2);
	let dpv = p2.sub(p1.mul(two)).add(p0).abs().to_array();
	let dqv = q2.sub(q1.mul(two)).add(q0).abs().to_array();
	let sv = p3.sub(p0).abs().add(q3.sub(q0).abs()).to_array();
	let pqv = p0.sub(q0).abs().to_array();
	// Per-segment decisions, as lane masks.
	let mut strong = [0i16; 8];
	let mut weak = [0i16; 8];
	let mut ndp = [0i16; 8];
	let mut ndq = [0i16; 8];
	let mut tcl = [0i16; 8];
	for j in 0..2 {
		let (l0, l3) = (4 * j, 4 * j + 3);
		let (b, t) = (beta[j], tc[j]);
		if t == 0 {
			continue;
		}
		let (dp0, dq0, dp3, dq3) =
			(i32::from(dpv[l0]), i32::from(dqv[l0]), i32::from(dpv[l3]), i32::from(dqv[l3]));
		let (d0, d3) = (dp0 + dq0, dp3 + dq3);
		if d0 + d3 >= b {
			continue;
		}
		let tc25 = (t * 5 + 1) >> 1;
		let strong_line = |l: usize, d: i32| {
			i32::from(sv[l]) < (b >> 3) && i32::from(pqv[l]) < tc25 && (d << 1) < (b >> 2)
		};
		let is_strong = strong_line(l0, d0) && strong_line(l3, d3);
		let side = (b + (b >> 1)) >> 3;
		for l in l0..=l3 {
			tcl[l] = t as i16;
			if is_strong {
				strong[l] = -1;
			} else {
				weak[l] = -1;
				ndp[l] = if dp0 + dp3 < side { -1 } else { 0 };
				ndq[l] = if dq0 + dq3 < side { -1 } else { 0 };
			}
		}
	}
	let strong = I16x8::from_array(strong);
	let weak = I16x8::from_array(weak);
	if !strong.or(weak).any() {
		return;
	}
	let lanes =
		|m: [bool; 2]| I16x8::from_array(std::array::from_fn(|l| if m[l / 4] { -1 } else { 0 }));
	let (keep_p, keep_q) = (lanes(no_p), lanes(no_q));
	let tc = I16x8::from_array(tcl);
	let (zero, maxv) = (I16x8::splat(0), I16x8::splat(max as i16));
	let c = |x: i16| I16x8::splat(x);
	// Strong filter.
	let tc2 = tc.add(tc);
	let sp0 =
		p2.add(p1.add(p0).add(q0).mul(two)).add(q1).add(c(4)).shr(3).clamp(p0.sub(tc2), p0.add(tc2));
	let sp1 = p2.add(p1).add(p0).add(q0).add(c(2)).shr(2).clamp(p1.sub(tc2), p1.add(tc2));
	let sp2 = p3
		.mul(two)
		.add(p2.mul(c(3)))
		.add(p1)
		.add(p0)
		.add(q0)
		.add(c(4))
		.shr(3)
		.clamp(p2.sub(tc2), p2.add(tc2));
	let sq0 =
		p1.add(p0.add(q0).add(q1).mul(two)).add(q2).add(c(4)).shr(3).clamp(q0.sub(tc2), q0.add(tc2));
	let sq1 = p0.add(q0).add(q1).add(q2).add(c(2)).shr(2).clamp(q1.sub(tc2), q1.add(tc2));
	let sq2 = p0
		.add(q0)
		.add(q1)
		.add(q2.mul(c(3)))
		.add(q3.mul(two))
		.add(c(4))
		.shr(3)
		.clamp(q2.sub(tc2), q2.add(tc2));
	// Weak filter.
	let delta = q0.sub(p0).mul(c(9)).sub(q1.sub(p1).mul(c(3))).add(c(8)).shr(4);
	let weak = weak.and(delta.abs().lt(tc.mul(c(10))));
	let delta = delta.clamp(zero.sub(tc), tc);
	let tc_2 = tc.shr(1);
	let wp0 = p0.add(delta).clamp(zero, maxv);
	let wq0 = q0.sub(delta).clamp(zero, maxv);
	let dp = p2.add(p0).add(c(1)).shr(1).sub(p1).add(delta).shr(1).clamp(zero.sub(tc_2), tc_2);
	let dq = q2.add(q0).add(c(1)).shr(1).sub(q1).sub(delta).shr(1).clamp(zero.sub(tc_2), tc_2);
	let wp1 = p1.add(dp).clamp(zero, maxv);
	let wq1 = q1.add(dq).clamp(zero, maxv);
	let pick = |orig: I16x8, s: I16x8, w: Option<(I16x8, I16x8)>, keep: I16x8| -> I16x8 {
		let mut r = strong.select(s, orig);
		if let Some((wv, wm)) = w {
			r = wm.select(wv, r);
		}
		keep.select(orig, r)
	};
	v[1] = pick(p2, sp2, None, keep_p);
	v[2] = pick(p1, sp1, Some((wp1, weak.and(I16x8::from_array(ndp)))), keep_p);
	v[3] = pick(p0, sp0, Some((wp0, weak)), keep_p);
	v[4] = pick(q0, sq0, Some((wq0, weak)), keep_q);
	v[5] = pick(q1, sq1, Some((wq1, weak.and(I16x8::from_array(ndq)))), keep_q);
	v[6] = pick(q2, sq2, None, keep_q);
}

multiversion! {
	/// Luma deblocking of a horizontal edge: 8 columns starting at `at`
	/// (the first q0 sample), rows above are p.
	#[allow(clippy::too_many_arguments)]
	pub fn deblock_luma_h(data: &mut [u16], stride: usize, at: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
		let top = at - 4 * stride;
		let mut v: [I16x8; 8] = std::array::from_fn(|r| I16x8::load_u16(&data[top + r * stride..]));
		luma_lines(&mut v, beta, tc, no_p, no_q, max);
		for r in 1..7 {
			v[r].store_u16(&mut data[top + r * stride..]);
		}
	}
}

multiversion! {
	/// Luma deblocking of a vertical edge: 8 rows starting at `at` (the
	/// first q0 sample), columns to the left are p.
	#[allow(clippy::too_many_arguments)]
	pub fn deblock_luma_v(data: &mut [u16], stride: usize, at: usize, beta: [i32; 2], tc: [i32; 2], no_p: [bool; 2], no_q: [bool; 2], max: i32) {
		let left = at - 4;
		let rows: [I16x8; 8] = std::array::from_fn(|r| I16x8::load_u16(&data[left + r * stride..]));
		let mut v = crate::simd::transpose8x8(rows);
		luma_lines(&mut v, beta, tc, no_p, no_q, max);
		let rows = crate::simd::transpose8x8(v);
		for (r, row) in rows.iter().enumerate() {
			row.store_u16(&mut data[left + r * stride..]);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::super::mc;
	use super::super::picture::Plane;

	struct Rng(u64);
	impl Rng {
		fn next(&mut self) -> u32 {
			self.0 ^= self.0 << 13;
			self.0 ^= self.0 >> 7;
			self.0 ^= self.0 << 17;
			(self.0 >> 16) as u32
		}
	}

	fn plane(rng: &mut Rng, bit_depth: u32, extreme: bool) -> Plane {
		let mut p = Plane::new(64, 48, 80, 0);
		let max = (1u32 << bit_depth) - 1;
		for y in 0..48 {
			for x in 0..64 {
				let i = p.idx(x, y);
				p.data[i] =
					if extreme { if rng.next() & 1 == 0 { 0 } else { max } } else { rng.next() % (max + 1) }
						as u16;
			}
		}
		p.extend_edges();
		p
	}

	/// Every fraction, width and bit depth, at positions inside, across and
	/// far outside the picture, against the scalar kernels.
	#[test]
	fn interpolation_matches_scalar() {
		let mut rng = Rng(0x1234_5678_9ABC);
		let widths = [4, 8, 12, 16, 24, 32, 48, 64];
		for bit_depth in [8, 10, 12] {
			for extreme in [false, true] {
				let p = plane(&mut rng, bit_depth, extreme);
				for chroma in [false, true] {
					let fracs = if chroma { 8 } else { 4 };
					for fx in 0..fracs {
						for fy in 0..fracs {
							for &w in &widths {
								let h = [4, 8, 16, 64][rng.next() as usize % 4];
								let x = (rng.next() % 200) as isize - 100;
								let y = (rng.next() % 200) as isize - 100;
								let mut a = vec![0i16; w * h];
								let mut b = vec![0i16; w * h];
								let mut tmp = vec![0i16; mc::TMP_LEN];
								mc::interpolate(&p, x, y, fx, fy, w, h, chroma, bit_depth, &mut a, &mut tmp);
								mc::interpolate_scalar(&p, x, y, fx, fy, w, h, chroma, bit_depth, &mut b, &mut tmp);
								assert_eq!(
									a, b,
									"bd {bit_depth} chroma {chroma} f {fx},{fy} w {w} h {h} at {x},{y}"
								);
							}
						}
					}
				}
			}
		}
	}

	#[test]
	fn sao_matches_scalar() {
		let mut rng = Rng(0xABCD_1234);
		for bit_depth in [8u32, 10, 12] {
			let max = (1i32 << bit_depth) - 1;
			for _ in 0..300 {
				let src: Vec<u16> = (0..64 * 3).map(|_| (rng.next() % (max as u32 + 1)) as u16).collect();
				let lim = 1i32 << (bit_depth.min(10) - 5);
				let offs: [i16; 5] = std::array::from_fn(|k| {
					if k == 2 { 0 } else { ((rng.next() % (2 * lim as u32)) as i32 - lim) as i16 }
				});
				let d = [-1isize, 1, -64, 64, -65, 65, -63, 63][rng.next() as usize % 8];
				let (mut a, mut b) = (src.clone(), src.clone());
				super::sao_edge(&src, &mut a, 64 + 8, 40, d, -d, &offs, max);
				for i in 72..112 {
					let cur = i32::from(src[i]);
					let x = i32::from(src[i.wrapping_add_signed(d)]);
					let y = i32::from(src[i.wrapping_add_signed(-d)]);
					let e = (2 + (cur - x).signum() + (cur - y).signum()) as usize;
					b[i] = (cur + i32::from(offs[e])).clamp(0, max) as u16;
				}
				assert_eq!(a, b, "edge bd {bit_depth}");
				let pos = (rng.next() % 32) as u8;
				let bo: [i16; 4] = std::array::from_fn(|k| offs[if k < 2 { k } else { k + 1 }]);
				let (mut a, mut b) = (src.clone(), src.clone());
				super::sao_band(&mut a, 64, bit_depth - 5, pos, &bo, max);
				for v in &mut b[..64] {
					let k = ((u32::from(*v) >> (bit_depth - 5)) + 32 - u32::from(pos)) & 31;
					let o = if k < 4 { i32::from(bo[k as usize]) } else { 0 };
					*v = (i32::from(*v) + o).clamp(0, max) as u16;
				}
				assert_eq!(a, b, "band bd {bit_depth}");
			}
		}
	}

	#[test]
	fn add_residual_matches_scalar() {
		let mut rng = Rng(0x1234_9876);
		for bit_depth in [8u32, 10, 12] {
			let max = (1i32 << bit_depth) - 1;
			for n in [4usize, 8, 16, 32] {
				let stride = 48;
				let mut px: Vec<u16> = (0..stride * 40).map(|_| (rng.next() as i32 & max) as u16).collect();
				let res: Vec<i32> = (0..n * n)
					.map(|_| {
						if rng.next().is_multiple_of(5) {
							rng.next() as i16 as i32
						} else {
							(rng.next() % 600) as i32 - 300
						}
					})
					.collect();
				let mut want = px.clone();
				for y in 0..n {
					for x in 0..n {
						let v = &mut want[3 + y * stride + x];
						*v = (i32::from(*v) + res[y * n + x]).clamp(0, max) as u16;
					}
				}
				super::add_residual(&mut px, stride, 3, &res, n, max);
				assert_eq!(px, want, "n {n} bd {bit_depth}");
			}
		}
	}

	#[test]
	fn idct_matches_scalar() {
		let mut rng = Rng(0x5151_7272);
		for bit_depth in [8u32, 10, 12] {
			for log2 in 2..=5 {
				let n = 1usize << log2;
				for _ in 0..200 {
					let (mx, my) = (rng.next() as usize % n, rng.next() as usize % n);
					if mx == 0 && my == 0 {
						continue;
					}
					let big = rng.next().is_multiple_of(4);
					let mut c = vec![0i32; n * n];
					for y in 0..=my {
						for x in 0..=mx {
							if rng.next().is_multiple_of(3) {
								c[y * n + x] =
									if big { rng.next() as i16 as i32 } else { (rng.next() % 512) as i32 - 256 };
							}
						}
					}
					let (mut a, mut b) = (c.clone(), c.clone());
					let (mut t1, mut t2) = (vec![0; 1024], vec![0; 1024]);
					let mut d = c.clone();
					super::idct(&mut a, n, mx, my, bit_depth, &mut t1, &mut t2);
					crate::codecs::hevc::transform::idct_scalar(&mut b, n, mx, my, bit_depth, &mut t1);
					assert_eq!(a, b, "n {n} bd {bit_depth}");
					if n >= 8 {
						let (mut s1, mut s2) = (vec![0i16; 1024], vec![0i16; 1024]);
						super::idct16(&mut d, n, mx, my, bit_depth, &mut s1, &mut s2);
						assert_eq!(d, b, "idct16 n {n} bd {bit_depth} max {mx},{my}");
					}
				}
			}
		}
	}

	/// Blocky content near the edge so every filter path runs.
	#[test]
	fn deblocking_matches_scalar() {
		let mut rng = Rng(0x0DEB_10C4);
		let (mut strong, mut weak, mut changed) = (0, 0, 0);
		for bit_depth in [8u32, 10] {
			let max = (1i32 << bit_depth) - 1;
			for round in 0..3000u32 {
				let stride = 32;
				let flat = !rng.next().is_multiple_of(4);
				let lvl = (rng.next() % (max as u32 + 1)) as i32;
				let step = (rng.next() % 40) as i32 - 20;
				let data: Vec<u16> = (0..stride * 16)
					.map(|i| {
						let side = if (i % stride) >= 8 || (i / stride) >= 8 { step } else { 0 };
						let noise =
							if flat { (rng.next() % 3) as i32 - 1 } else { (rng.next() % 64) as i32 - 32 };
						(lvl + side + noise).clamp(0, max) as u16
					})
					.collect();
				let beta = [
					((rng.next() % 64) as i32) << (bit_depth - 8),
					((rng.next() % 64) as i32) << (bit_depth - 8),
				];
				let tc = [
					((rng.next() % 25) as i32) << (bit_depth - 8),
					((rng.next() % 25) as i32) << (bit_depth - 8),
				];
				let no_p = [rng.next().is_multiple_of(8), rng.next().is_multiple_of(8)];
				let no_q = [rng.next().is_multiple_of(8), rng.next().is_multiple_of(8)];
				let vertical = round.is_multiple_of(2);
				let at = if vertical { 4 * stride + 8 } else { 8 * stride + 4 };
				let (mut a, mut b) = (data.clone(), data.clone());
				if vertical {
					super::deblock_luma_v(&mut a, stride, at, beta, tc, no_p, no_q, max);
				} else {
					super::deblock_luma_h(&mut a, stride, at, beta, tc, no_p, no_q, max);
				}
				for j in 0..2 {
					let (xs, ys) = if vertical { (1, stride) } else { (stride, 1) };
					let seg = at + 4 * j * ys;
					let r = crate::codecs::hevc::filter::luma_edge_probe(
						&mut b, seg, xs, ys, beta[j], tc[j], no_p[j], no_q[j], max,
					);
					match r {
						2 => strong += 1,
						1 => weak += 1,
						_ => {}
					}
				}
				if a != data {
					changed += 1;
				}
				assert_eq!(a, b, "bd {bit_depth} round {round}");
			}
		}
		assert!(strong > 50 && weak > 50 && changed > 500, "{strong} {weak} {changed}");
	}

	#[test]
	fn stores_match_scalar() {
		let mut rng = Rng(0xFEED_FACE);
		for bit_depth in [8, 10, 12] {
			for _ in 0..200 {
				let (w, h) = ([4, 8, 12, 16, 32, 64][rng.next() as usize % 6], 1 + rng.next() as usize % 8);
				// Full i16 range, including values far beyond 14-bit.
				let a: Vec<i16> = (0..w * h + 8).map(|_| rng.next() as i16).collect();
				let b: Vec<i16> = (0..w * h + 8).map(|_| (rng.next() % 20000) as i16 - 2000).collect();
				let mut p1 = Plane::new(64, 8, 8, 0);
				let mut p2 = p1.clone();
				let at = p1.idx(0, 0);
				mc::put_uni(&mut p1, at, &a, w, h, bit_depth, None);
				mc::put_uni_scalar(&mut p2, at, &a, w, h, bit_depth, None);
				assert!(p1.data == p2.data, "uni bd {bit_depth}");
				mc::put_bi(&mut p1, at, &a, &b, w, h, bit_depth, None);
				mc::put_bi_scalar(&mut p2, at, &a, &b, w, h, bit_depth, None);
				assert!(p1.data == p2.data, "bi bd {bit_depth}");
			}
		}
	}
}

// ---------------------------------------------------------------------
// Inverse DCT in 16-bit lanes (n >= 8)
// ---------------------------------------------------------------------

/// DCT coefficient pairs for [`idct_pass16`]: `[parity][j][i]` holds
/// `(M[k][i], M[k + 2][i])` four times, `k = parity + 4j`, where
/// `M[k][i] = DCT_MATRIX[k * 32 / n][i]` (zero past the matrix).
type Pairs = [[[[i16; 8]; 16]; 8]; 2];

fn pair_table(n: usize) -> &'static Pairs {
	use std::sync::OnceLock;
	static T: OnceLock<Vec<Pairs>> = OnceLock::new();
	let all = T.get_or_init(|| {
		use super::transform::DCT_MATRIX;
		[8usize, 16, 32]
			.iter()
			.map(|&n| {
				let step = 32 / n;
				let m = |k: usize, i: usize| if k < n { DCT_MATRIX[k * step][i] as i16 } else { 0 };
				let mut t = [[[[0i16; 8]; 16]; 8]; 2];
				for (parity, tp) in t.iter_mut().enumerate() {
					for (j, tj) in tp.iter_mut().enumerate() {
						let k = parity + 4 * j;
						for (i, v) in tj.iter_mut().enumerate().take(n / 2) {
							let (a, b) = (m(k, i), m(k + 2, i));
							*v = [a, b, a, b, a, b, a, b];
						}
					}
				}
				t
			})
			.collect()
	});
	&all[match n {
		8 => 0,
		16 => 1,
		_ => 2,
	}]
}

/// One 1-D inverse DCT pass over groups of eight columns of 16-bit
/// coefficients: column `x` of `dst` is the transform of column `x` of
/// `src` (both `n`x`n`, row-major), using its first `nz` coefficients, for
/// `x < cols` (`cols % 8 == 0`). The even and odd halves are accumulated
/// with `madd` on interleaved pairs of coefficient rows; the results are
/// exact 32-bit sums, as in the scalar code, and the saturating pack is
/// its clip to 16 bits.
#[inline(always)]
fn idct_pass16(src: &[i16], dst: &mut [i16], n: usize, cols: usize, nz: usize, shift: u32) {
	let t = pair_table(n);
	let half = n / 2;
	let add = I32x4::splat(1 << (shift - 1));
	let zero = I16x8::splat(0);
	let z4 = I32x4::splat(0);
	let mut x = 0;
	while x < cols {
		let mut e = [(z4, z4); 16];
		let mut o = [(z4, z4); 16];
		for parity in 0..2 {
			let acc = if parity == 0 { &mut e } else { &mut o };
			let mut j = 0;
			while parity + 4 * j < nz {
				let k = parity + 4 * j;
				let a = I16x8::load(&src[k * n + x..]);
				let b = if k + 2 < nz { I16x8::load(&src[(k + 2) * n + x..]) } else { zero };
				if a.or(b).any() {
					let (lo, hi) = (a.zip_lo(b), a.zip_hi(b));
					for (i, m) in t[parity][j].iter().enumerate().take(half) {
						let m = I16x8::from_array(*m);
						acc[i].0 = acc[i].0.add(lo.madd(m));
						acc[i].1 = acc[i].1.add(hi.madd(m));
					}
				}
				j += 1;
			}
		}
		for i in 0..half {
			let (elo, ehi) = (e[i].0.add(add), e[i].1.add(add));
			let (olo, ohi) = o[i];
			elo.add(olo).shr(shift).pack_i16(ehi.add(ohi).shr(shift)).store(&mut dst[i * n + x..]);
			elo
				.sub(olo)
				.shr(shift)
				.pack_i16(ehi.sub(ohi).shr(shift))
				.store(&mut dst[(n - 1 - i) * n + x..]);
		}
		x += 8;
	}
}

/// Transposes the `rows`x`cols` top-left part of `src` (stride `n`) into
/// `dst` (stride `n`), 8x8 blocks at a time.
#[inline(always)]
fn transpose16(src: &[i16], dst: &mut [i16], n: usize, rows: usize, cols: usize) {
	for by in (0..rows).step_by(8) {
		for bx in (0..cols).step_by(8) {
			let r: [I16x8; 8] = std::array::from_fn(|k| I16x8::load(&src[(by + k) * n + bx..]));
			let t = crate::simd::transpose8x8(r);
			for (k, v) in t.iter().enumerate() {
				v.store(&mut dst[(bx + k) * n + by..]);
			}
		}
	}
}

multiversion! {
	/// [`super::transform::idct`] for `n >= 8` with at least one AC
	/// coefficient, on 16-bit lanes (bit-exact: same sums, same clipping).
	/// `a` and `b` are scratch space of `32 * 32` (every sample read is
	/// written first).
	pub fn idct16(coeffs: &mut [i32], n: usize, max_x: usize, max_y: usize, bit_depth: u32, a: &mut [i16], b: &mut [i16]) {
		let cols = ((max_x + 8) & !7).min(n);
		let nz = (max_y + 1).min(n);
		// Dequantised levels are already within 16 bits.
		for y in 0..nz {
			for x in (0..cols).step_by(8) {
				let at = y * n + x;
				let v = I32x4::load(&coeffs[at..]).pack_i16(I32x4::load(&coeffs[at + 4..]));
				v.store(&mut a[at..]);
			}
		}
		// Columns (shift 7), then the rows as columns of the transpose.
		idct_pass16(a, b, n, cols, nz, 7);
		transpose16(b, a, n, n, cols);
		idct_pass16(a, b, n, n, (max_x + 1).min(n), 20 - bit_depth);
		for by in (0..n).step_by(8) {
			for bx in (0..n).step_by(8) {
				let r: [I16x8; 8] = std::array::from_fn(|k| I16x8::load(&b[(by + k) * n + bx..]));
				let t = crate::simd::transpose8x8(r);
				for (k, v) in t.iter().enumerate() {
					let at = (bx + k) * n + by;
					v.widen_lo().store(&mut coeffs[at..]);
					v.widen_hi().store(&mut coeffs[at + 4..]);
				}
			}
		}
	}
}

multiversion! {
	/// Adds an `n`x`n` residual (16-bit range) to the prediction at `at`
	/// and clips to `0..=max`. Rows must have 8 readable samples from every
	/// multiple of 8 (padded planes do); only the block is written. The
	/// 16-bit saturating add clips exactly like the scalar 32-bit add.
	pub fn add_residual(dst: &mut [u16], stride: usize, at: usize, res: &[i32], n: usize, max: i32) {
		let (zero, maxv) = (I16x8::splat(0), I16x8::splat(max as i16));
		for y in 0..n {
			let row = &mut dst[at + y * stride..];
			let r = &res[y * n..];
			let mut x = 0;
			while x < n {
				let rv = I32x4::load(&r[x..]).pack_i16(if x + 4 < n {
					I32x4::load(&r[x + 4..])
				} else {
					I32x4::splat(0)
				});
				let v = I16x8::load_u16(&row[x..]).saturating_add(rv).clamp(zero, maxv);
				store_n_u16(v, &mut row[x..], (n - x).min(8));
				x += 8;
			}
		}
	}
}

// ---------------------------------------------------------------------
// Intra prediction
// ---------------------------------------------------------------------

/// One row of an angular prediction: eight outputs from `row[x..]`.
#[inline(always)]
fn angular8(row: &[i16], x: usize, w: I16x8) -> I16x8 {
	let r16 = I32x4::splat(16);
	let a = I16x8::load(&row[x..]);
	let b = I16x8::load(&row[x + 1..]);
	let lo = a.zip_lo(b).madd(w).add(r16).shr(5);
	let hi = a.zip_hi(b).madd(w).add(r16).shr(5);
	lo.pack_i16(hi)
}

/// `(32 - fact, fact)` weights and the start of row `y`'s references.
#[inline(always)]
fn angular_row(base: usize, y: usize, angle: i32) -> (I16x8, usize) {
	let pos = (y as i32 + 1) * angle;
	let (idx, fact) = (pos >> 5, (pos & 31) as i16);
	let w = I16x8::from_array([32 - fact, fact, 32 - fact, fact, 32 - fact, fact, 32 - fact, fact]);
	(w, (base as i32 + idx + 1) as usize)
}

multiversion! {
	/// Angular intra prediction (8.4.4.2.6) of an `n`x`n` block at `at` of
	/// `dst` (stride `s`), transposed for the horizontal modes.
	/// `refs[base + k]` is the extended main reference `ref[k]`, readable 8
	/// samples past `2n + 1`. Each output is
	/// `((32 - fact) * ref[i] + fact * ref[i + 1] + 16) >> 5`, computed with
	/// `madd` on (ref[i], ref[i + 1]) pairs, exactly (fact 0 gives ref[i]).
	/// Rows of `dst` must be readable 8 samples from `at`.
	#[allow(clippy::too_many_arguments)]
	pub fn intra_angular(refs: &[i16], base: usize, n: usize, angle: i32, transpose: bool, dst: &mut [u16], at: usize, s: usize) {
		if !transpose {
			for y in 0..n {
				let (w, start) = angular_row(base, y, angle);
				let row = &refs[start..];
				let o = &mut dst[at + y * s..];
				let mut x = 0;
				while x < n {
					store_n_u16(angular8(row, x, w), &mut o[x..], (n - x).min(8));
					x += 8;
				}
			}
		} else if n == 4 {
			let v: [[i16; 8]; 4] = std::array::from_fn(|y| {
				let (w, start) = angular_row(base, y, angle);
				angular8(&refs[start..], 0, w).to_array()
			});
			for (y, vy) in v.iter().enumerate() {
				for (x, &p) in vy.iter().take(4).enumerate() {
					dst[at + x * s + y] = p as u16;
				}
			}
		} else {
			// Eight rows at a time, transposed in registers into columns.
			for y in (0..n).step_by(8) {
				let ws: [(I16x8, usize); 8] = std::array::from_fn(|k| angular_row(base, y + k, angle));
				for x in (0..n).step_by(8) {
					let rows: [I16x8; 8] = std::array::from_fn(|k| angular8(&refs[ws[k].1..], x, ws[k].0));
					let t = crate::simd::transpose8x8(rows);
					for (k, v) in t.iter().enumerate() {
						v.store_u16(&mut dst[at + (x + k) * s + y..]);
					}
				}
			}
		}
	}
}

multiversion! {
	/// Planar intra prediction (8.4.4.2.5) of an `n`x`n` block at `at` of
	/// `dst` (stride `s`):
	/// `(n-1-x) L[y] + (x+1) TR + (n-1-y) T[x] + (y+1) BL + n >> log2(n)+1`,
	/// as two `madd`s of sample pairs by weight pairs, exactly.
	pub fn intra_planar(top: &[u16; 129], left: &[u16; 129], n: usize, dst: &mut [u16], at: usize, s: usize) {
		let shift = n.trailing_zeros() + 1;
		let (tr, bl) = (top[1 + n] as i16, left[1 + n] as i16);
		let rnd = I32x4::splat(n as i32);
		let blv = I16x8::splat(bl);
		let wx = |x0: usize| -> I16x8 {
			I16x8::from_array(std::array::from_fn(|k| {
				let xx = (x0 + k / 2) as i16;
				if k % 2 == 0 { n as i16 - 1 - xx } else { xx + 1 }
			}))
		};
		let wxs: [(I16x8, I16x8); 4] = std::array::from_fn(|g| (wx(8 * g), wx(8 * g + 4)));
		for y in 0..n {
			let ly = left[1 + y] as i16;
			let lt = I16x8::from_array([ly, tr, ly, tr, ly, tr, ly, tr]);
			let (wy0, wy1) = ((n - 1 - y) as i16, (y + 1) as i16);
			let wy = I16x8::from_array([wy0, wy1, wy0, wy1, wy0, wy1, wy0, wy1]);
			let o = &mut dst[at + y * s..];
			let mut x = 0;
			while x < n {
				let (w0, w1) = wxs[x / 8];
				let t = I16x8::load_u16(&top[1 + x..]);
				let lo = lt.madd(w0).add(t.zip_lo(blv).madd(wy)).add(rnd).shr(shift);
				let hi = lt.madd(w1).add(t.zip_hi(blv).madd(wy)).add(rnd).shr(shift);
				store_n_u16(lo.pack_i16(hi), &mut o[x..], (n - x).min(8));
				x += 8;
			}
		}
	}
}
