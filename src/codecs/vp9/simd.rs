//! Vector versions of the VP9 hot loops, written with the portable types of
//! [`crate::simd`] (SSE2 / NEON, AVX2 by run-time dispatch): the loop
//! filter and the 8-tap sub-pixel convolution. Each is bit-exact with its
//! scalar original in `loopfilter.rs` / `inter.rs`, which stays the
//! fallback and the reference the tests compare against. libvpx:
//! `vpx_dsp/x86/loopfilter_sse2.c`, `vpx_subpixel_8t_intrin_ssse3.c`.

use super::decoder::Plane;
use super::inter::{MAX, RefPlane, Scratch};
use super::loopfilter::Thresholds;
use super::tables::SUBPEL_FILTERS;
use crate::multiversion;
use crate::simd::{I16x8, I32x4, U8x16, transpose8x8};

// ---------------------------------------------------------------------------
// Loop filter

/// `v.clamp(-128, 127)`.
#[inline(always)]
fn sclamp(v: I16x8) -> I16x8 {
	v.clamp(I16x8::splat(-128), I16x8::splat(127))
}

/// Filters 8 lines across an edge, one line per lane. `px` holds
/// `p3..p0 q0..q3` (`taps` 4 or 8) or `p7..p0 q0..q7` (`taps` 16) in its
/// first `2 * n` entries. Returns `false` when no line passed the mask.
#[inline(always)]
fn filter_lines(px: &mut [I16x8; 16], n: usize, t: &Thresholds, taps: u8) -> bool {
	let p = |i: usize| px[n - 1 - i];
	let q = |i: usize| px[n + i];
	let d = |a: I16x8, b: I16x8| a.sub(b).abs();
	let le = |a: I16x8, lim: I16x8| a.gt(lim).not();
	let lim = I16x8::splat(i16::from(t.lim));
	let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));
	let (q0, q1, q2, q3) = (q(0), q(1), q(2), q(3));
	let mask = le(d(p3, p2), lim)
		.and(le(d(p2, p1), lim))
		.and(le(d(p1, p0), lim))
		.and(le(d(q1, q0), lim))
		.and(le(d(q2, q1), lim))
		.and(le(d(q3, q2), lim))
		.and(le(d(p0, q0).shl(1).add(d(p1, q1).shr(1)), I16x8::splat(i16::from(t.mblim))));
	if !mask.any() {
		return false;
	}
	let one = I16x8::splat(1);
	let flat = if taps >= 8 {
		mask
			.and(le(d(p1, p0), one))
			.and(le(d(q1, q0), one))
			.and(le(d(p2, p0), one))
			.and(le(d(q2, q0), one))
			.and(le(d(p3, p0), one))
			.and(le(d(q3, q0), one))
	} else {
		I16x8::splat(0)
	};
	let flat2 = if taps == 16 && flat.any() {
		(4..8).fold(flat, |m, i| m.and(le(d(p(i), p0), one)).and(le(d(q(i), q0), one)))
	} else {
		I16x8::splat(0)
	};

	// filter4 on p1 p0 q0 q1, signed around 128.
	let s = |v: I16x8| v.sub(I16x8::splat(128));
	let u = |v: I16x8| v.add(I16x8::splat(128));
	let hev_thr = I16x8::splat(i16::from(t.hev));
	let hev = d(p1, p0).gt(hev_thr).or(d(q1, q0).gt(hev_thr));
	let (ps1, ps0, qs0, qs1) = (s(p1), s(p0), s(q0), s(q1));
	let base = hev.and(sclamp(ps1.sub(qs1)));
	let filter = sclamp(base.add(qs0.sub(ps0).mul(I16x8::splat(3))));
	let filter1 = sclamp(filter.add(I16x8::splat(4))).shr(3);
	let filter2 = sclamp(filter.add(I16x8::splat(3))).shr(3);
	let f4_q0 = u(sclamp(qs0.sub(filter1)));
	let f4_p0 = u(sclamp(ps0.add(filter2)));
	let f = filter1.add(one).shr(1);
	let f4_q1 = hev.select(q1, u(sclamp(qs1.sub(f))));
	let f4_p1 = hev.select(p1, u(sclamp(ps1.add(f))));

	// The 7-tap flat filter.
	let r3 = |v: I16x8| v.add(I16x8::splat(4)).shr(3);
	let (f8_p2, f8_p1, f8_p0, f8_q0, f8_q1, f8_q2) = if flat.any() {
		let two = |v: I16x8| v.shl(1);
		(
			r3(p3.add(p3).add(p3).add(two(p2)).add(p1).add(p0).add(q0)),
			r3(p3.add(p3).add(p2).add(two(p1)).add(p0).add(q0).add(q1)),
			r3(p3.add(p2).add(p1).add(two(p0)).add(q0).add(q1).add(q2)),
			r3(p2.add(p1).add(p0).add(two(q0)).add(q1).add(q2).add(q3)),
			r3(p1.add(p0).add(q0).add(two(q1)).add(q2).add(q3).add(q3)),
			r3(p0.add(q0).add(q1).add(two(q2)).add(q3).add(q3).add(q3)),
		)
	} else {
		(p2, p1, p0, q0, q1, q2)
	};

	// The 15-tap filter: a running sum over the 16-sample window.
	let mut wide = [I16x8::splat(0); 14];
	if flat2.any() {
		let v = *px;
		let mut sum = v[0].mul(I16x8::splat(7)).add(v[1].shl(1));
		for x in &v[2..9] {
			sum = sum.add(*x);
		}
		let r4 = |s: I16x8| s.add(I16x8::splat(8)).shr(4);
		wide[0] = r4(sum);
		for k in 1..14 {
			sum = sum.add(v[(k + 8).min(15)]).sub(v[k.saturating_sub(7)]).add(v[k + 1]).sub(v[k]);
			wide[k] = r4(sum);
		}
	}

	// Per lane: wide where flat2, else 7-tap where flat, else filter4,
	// and nothing where the mask failed.
	let pick = |f4: I16x8, f8: I16x8, idx: usize, orig: I16x8| {
		let v = flat.select(f8, f4);
		let v = if taps == 16 { flat2.select(wide[idx - 1], v) } else { v };
		mask.select(v, orig)
	};
	let c = n;
	let new_p1 = pick(f4_p1, f8_p1, 6, p1);
	let new_p0 = pick(f4_p0, f8_p0, 7, p0);
	let new_q0 = pick(f4_q0, f8_q0, 8, q0);
	let new_q1 = pick(f4_q1, f8_q1, 9, q1);
	let new_p2 = pick(p2, f8_p2, 5, p2);
	let new_q2 = pick(q2, f8_q2, 10, q2);
	if taps == 16 {
		// Outer samples p6..p3 and q3..q6 change only under flat2.
		for (k, w) in wide.iter().enumerate() {
			let i = k + 1;
			if !(c - 3..c + 3).contains(&i) {
				px[i] = mask.and(flat2).select(*w, px[i]);
			}
		}
	}
	px[c - 3] = new_p2;
	px[c - 2] = new_p1;
	px[c - 1] = new_p0;
	px[c] = new_q0;
	px[c + 1] = new_q1;
	px[c + 2] = new_q2;
	true
}

/// Filter reach on each side of an edge.
#[inline(always)]
fn reach(taps: u8) -> usize {
	if taps == 16 { 8 } else { 4 }
}

multiversion! {
	/// [`super::loopfilter`]'s vertical edge: 8 rows at (x, y).
	pub fn vertical(plane: &mut Plane, x: usize, y: usize, t: &Thresholds, taps: u8) {
		let n = reach(taps);
		let stride = plane.stride;
		let mut px = [I16x8::splat(0); 16];
		// Rows hold the lines; transpose so lanes do.
		if n == 4 {
			let rows: [I16x8; 8] = std::array::from_fn(|r| I16x8::load_u8(&plane.data[(y + r) * stride + x - 4..]));
			px[..8].copy_from_slice(&transpose8x8(rows));
		} else {
			let rows: [U8x16; 8] = std::array::from_fn(|r| U8x16::load(&plane.data[(y + r) * stride + x - 8..]));
			px[..8].copy_from_slice(&transpose8x8(rows.map(U8x16::widen_lo)));
			px[8..].copy_from_slice(&transpose8x8(rows.map(U8x16::widen_hi)));
		}
		if !filter_lines(&mut px, n, t, taps) {
			return;
		}
		if n == 4 {
			let rows = transpose8x8(std::array::from_fn(|i| px[i]));
			for (r, v) in rows.iter().enumerate() {
				v.pack_u8(*v).store8(&mut plane.data[(y + r) * stride + x - 4..]);
			}
		} else {
			let lo = transpose8x8(std::array::from_fn(|i| px[i]));
			let hi = transpose8x8(std::array::from_fn(|i| px[8 + i]));
			for r in 0..8 {
				lo[r].pack_u8(hi[r]).store(&mut plane.data[(y + r) * stride + x - 8..]);
			}
		}
	}
}

multiversion! {
	/// [`super::loopfilter`]'s horizontal edge: 8 columns at (x, y).
	pub fn horizontal(plane: &mut Plane, x: usize, y: usize, t: &Thresholds, taps: u8) {
		let n = reach(taps);
		let stride = plane.stride;
		let base = (y - n) * stride + x;
		let mut px = [I16x8::splat(0); 16];
		for (i, v) in px[..2 * n].iter_mut().enumerate() {
			*v = I16x8::load_u8(&plane.data[base + i * stride..]);
		}
		if !filter_lines(&mut px, n, t, taps) {
			return;
		}
		// The outermost sample on each side never changes.
		for (i, v) in px[..2 * n].iter().enumerate().take(2 * n - 1).skip(1) {
			v.pack_u8(*v).store8(&mut plane.data[base + i * stride..]);
		}
	}
}

// ---------------------------------------------------------------------------
// Inverse transform

/// Adds `v` to every pixel of the `n` x `n` block, clipped: a saturating
/// add or subtract of `|v|` (capped at 255, which already saturates).
pub fn add_constant(dst: &mut [u8], stride: usize, n: usize, v: i64) {
	let m = U8x16::splat(v.unsigned_abs().min(255) as u8);
	let f = |p: U8x16| if v >= 0 { p.saturating_add(m) } else { p.saturating_sub(m) };
	for r in 0..n {
		let row = &mut dst[r * stride..];
		match n {
			4 => f(U8x16::load4(row)).store4(row),
			8 => f(U8x16::load8(row)).store8(row),
			_ => {
				for c in (0..n).step_by(16) {
					f(U8x16::load(&row[c..])).store(&mut row[c..]);
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Inter prediction

/// An 8-tap kernel as four pairs for `madd`.
#[derive(Clone, Copy)]
struct Taps([I16x8; 4]);

impl Taps {
	#[inline(always)]
	fn new(k: &[i16; 8]) -> Self {
		Self(std::array::from_fn(|p| I16x8::from_array(std::array::from_fn(|i| k[2 * p + i % 2]))))
	}

	/// `clip((sum_t k[t] * a[t] + 64) >> 7)` lane-wise, as 16-bit lanes
	/// (packing to bytes does the clip). Sums are exact in 32 bits.
	#[inline(always)]
	fn apply(&self, a: &[I16x8; 8]) -> I16x8 {
		let mut lo = I32x4::splat(64);
		let mut hi = I32x4::splat(64);
		for p in 0..4 {
			lo = lo.add(a[2 * p].zip_lo(a[2 * p + 1]).madd(self.0[p]));
			hi = hi.add(a[2 * p].zip_hi(a[2 * p + 1]).madd(self.0[p]));
		}
		lo.shr(7).pack_i16(hi.shr(7))
	}
}

/// Filters 16 outputs from eight 16-byte loads `at(t)` (t = 0..8).
#[inline(always)]
fn conv16(k: &Taps, at: impl Fn(usize) -> U8x16) -> U8x16 {
	let v: [U8x16; 8] = std::array::from_fn(at);
	let lo = k.apply(&v.map(U8x16::widen_lo));
	let hi = k.apply(&v.map(U8x16::widen_hi));
	lo.pack_u8(hi)
}

/// Filters 8 outputs (in the low half) from eight 8-byte loads.
#[inline(always)]
fn conv8(k: &Taps, at: impl Fn(usize) -> U8x16) -> U8x16 {
	let v = k.apply(&std::array::from_fn(|t| at(t).widen_lo()));
	v.pack_u8(v)
}

/// Stores `w` (4, 8 or 16) bytes, averaging with what is there if `avg`.
#[inline(always)]
fn put(v: U8x16, out: &mut [u8], w: usize, avg: bool) {
	let v = match (avg, w) {
		(false, _) => v,
		(true, 16) => v.avg(U8x16::load(out)),
		(true, 8) => v.avg(U8x16::load8(out)),
		(true, _) => v.avg(U8x16::load4(out)),
	};
	match w {
		16 => v.store(out),
		8 => v.store8(out),
		_ => v.store4(out),
	}
}

multiversion! {
	/// [`super::inter::predict`] for `w` of 4 or a multiple of 8.
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
		let kernels = &SUBPEL_FILTERS[usize::from(filter.min(3))];
		let (phase_x, phase_y) = ((mv_col & 15) as usize, (mv_row & 15) as usize);
		let x0 = x as i64 + i64::from(mv_col >> 4) - 3;
		let y0 = y as i64 + i64::from(mv_row >> 4) - 3;
		let (sw, sh) = (src.width as i64, src.height as i64);
		// A 4-wide block is filtered 8 lanes at a time, so it reads 8 + 7
		// columns.
		let (bw, bh) = (w.max(8) + 7, h + 7);
		let inside = x0 >= 0 && y0 >= 0 && x0 + bw as i64 <= sw && y0 + bh as i64 <= sh;
		let Scratch { block: emu, temp } = scratch;
		let (block, base, stride): (&[u8], usize, usize) = if inside {
			(src.data, y0 as usize * src.stride + x0 as usize, src.stride)
		} else {
			for r in 0..bh {
				let sy = (y0 + r as i64).clamp(0, sh - 1) as usize;
				let row = &src.data[sy * src.stride..];
				for (c, o) in emu[r * MAX..r * MAX + bw].iter_mut().enumerate() {
					*o = row[(x0 + c as i64).clamp(0, sw - 1) as usize];
				}
			}
			(&emu[..], 0, MAX)
		};

		// Horizontal pass into `temp` (64-wide rows), or straight through.
		let (rows_from, rows_base, rows_stride): (&[u8], usize, usize) = if phase_x != 0 {
			let k = Taps::new(&kernels[phase_x]);
			let (r0, r1) = if phase_y != 0 { (0, bh) } else { (3, 3 + h) };
			for r in r0..r1 {
				let row = &block[base + r * stride..];
				let out = &mut temp[r * 64..];
				let mut c = 0;
				while c + 16 <= w {
					conv16(&k, |t| U8x16::load(&row[c + t..])).store(&mut out[c..]);
					c += 16;
				}
				if c < w {
					conv8(&k, |t| U8x16::load8(&row[c + t..])).store8(&mut out[c..]);
				}
			}
			(&temp[..], 0, 64)
		} else {
			(block, base + 3, stride)
		};

		let k = Taps::new(&kernels[phase_y]);
		for r in 0..h {
			let out = &mut dst[(y + r) * dst_stride + x..];
			let at = |t: usize, c: usize| rows_base + (r + t) * rows_stride + c;
			let mut c = 0;
			while c < w {
				let n = match w - c {
					16.. => 16,
					8.. => 8,
					_ => 4,
				};
				let v = match (phase_y != 0, n) {
					(true, 16) => conv16(&k, |t| U8x16::load(&rows_from[at(t, c)..])),
					(true, _) => conv8(&k, |t| U8x16::load8(&rows_from[at(t, c)..])),
					(false, 16) => U8x16::load(&rows_from[at(3, c)..]),
					(false, _) => U8x16::load8(&rows_from[at(3, c)..]),
				};
				put(v, &mut out[c..], n, avg);
				c += n;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::super::inter::predict_scalar;
	use super::super::loopfilter::{horizontal_scalar, thresholds, vertical_scalar};
	use super::*;

	struct Rng(u64);
	impl Rng {
		fn next(&mut self) -> u32 {
			self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			(self.0 >> 33) as u32
		}
		fn range(&mut self, lo: i32, hi: i32) -> i32 {
			lo + (self.next() % (hi - lo + 1) as u32) as i32
		}
	}

	fn plane(w: usize, h: usize, f: impl FnMut(usize, usize) -> u8) -> Plane {
		let mut f = f;
		let data: Vec<u8> = (0..w * h).map(|i| f(i % w, i / w)).collect();
		Plane { data: data.into(), stride: w, width: w, height: h, aligned_w: w, aligned_h: h }
	}

	/// Flat 8x8 blocks with noise of a random small amplitude: steps that
	/// the 4-, 8- and 16-wide filters all see.
	fn blocky(rng: &mut Rng, w: usize, h: usize) -> Plane {
		let levels: Vec<i32> = (0..(w / 8 + 1) * (h / 8 + 1)).map(|_| rng.range(0, 255)).collect();
		let step = rng.range(0, 6);
		let amp = rng.range(0, 2);
		let mut noise = Rng(u64::from(rng.next()));
		plane(w, h, |x, y| {
			let base = levels[(y / 8) * (w / 8 + 1) + x / 8] % (step * 8 + 1) + 100;
			(base + noise.range(-amp, amp)).clamp(0, 255) as u8
		})
	}

	#[test]
	fn loop_filter_matches_scalar_for_every_length_and_direction() {
		let mut rng = Rng(7);
		let mut changed = [[0usize; 3]; 2];
		for round in 0..6000 {
			let taps = [4u8, 8, 16][round % 3];
			let vertical = (round / 3) % 2 == 0;
			let sharp = rng.range(0, 7) as u8;
			let t = thresholds(sharp)[rng.range(0, 63) as usize];
			let mut a = blocky(&mut rng, 48, 48);
			let orig = a.data.clone();
			let mut b = a.clone();
			let (x, y) = (16 + 8 * rng.range(0, 2) as usize, 16 + 8 * rng.range(0, 2) as usize);
			if vertical {
				vertical_scalar(&mut a, x, y, &t, taps, 0);
				super::vertical(&mut b, x, y, &t, taps);
			} else {
				horizontal_scalar(&mut a, x, y, &t, taps, 0);
				super::horizontal(&mut b, x, y, &t, taps);
			}
			assert_eq!(a.data, b.data, "taps {taps} vertical {vertical} {t:?}");
			if a.data != orig {
				changed[usize::from(vertical)][round % 3] += 1;
			}
		}
		assert!(changed.iter().flatten().all(|&n| n > 200), "{changed:?}");
	}

	#[test]
	fn wide_filter_path_is_exercised() {
		// Near-flat sides with a small step: flat2 holds on every line.
		let mut rng = Rng(8);
		let t = thresholds(0)[40];
		for _ in 0..200 {
			let l = rng.range(50, 200);
			let mut a = plane(32, 32, |x, _| (l + if x >= 16 { 3 } else { 0 }) as u8);
			let mut b = a.clone();
			vertical_scalar(&mut a, 16, 8, &t, 16, 0);
			super::vertical(&mut b, 16, 8, &t, 16);
			assert_eq!(a.data, b.data);
			// p3 (x = 12) changes only under the 15-tap filter.
			assert_eq!(a.data[8 * 32 + 12], l as u8 + 1, "the 15-tap filter should have smoothed p3");
		}
	}

	#[test]
	fn inter_prediction_matches_scalar() {
		let mut rng = Rng(9);
		let src_px: Vec<u8> = (0..96 * 80).map(|_| rng.next() as u8).collect();
		let src = RefPlane { data: &src_px, stride: 96, width: 96, height: 80 };
		let sizes = [
			(4, 4),
			(4, 8),
			(8, 4),
			(8, 8),
			(16, 8),
			(8, 16),
			(16, 16),
			(32, 16),
			(32, 32),
			(64, 32),
			(64, 64),
		];
		let mut s1 = Scratch::default();
		let mut s2 = Scratch::default();
		for round in 0..3000 {
			let (w, h) = sizes[round % sizes.len()];
			let filter = rng.range(0, 3) as u8;
			let avg = rng.next().is_multiple_of(2);
			let (x, y) = (8 * rng.range(0, 4) as usize, 8 * rng.range(0, 2) as usize);
			let mv_row = rng.range(-60 * 16, 60 * 16);
			let mv_col = rng.range(-60 * 16, 60 * 16);
			// Whole-pixel and half-way phases are special cases.
			let (mv_row, mv_col) = match round % 7 {
				0 => (mv_row & !15, mv_col),
				1 => (mv_row, mv_col & !15),
				2 => (mv_row & !15, mv_col & !15),
				_ => (mv_row, mv_col),
			};
			let init: Vec<u8> = (0..96 * 80).map(|_| rng.next() as u8).collect();
			let mut a = init.clone();
			let mut b = init;
			predict_scalar(&mut s1, &src, &mut a, 96, x, y, w, h, mv_row, mv_col, filter, avg, 255);
			super::predict(&mut s2, &src, &mut b, 96, x, y, w, h, mv_row, mv_col, filter, avg);
			assert_eq!(a, b, "{w}x{h} mv ({mv_row},{mv_col}) filter {filter} avg {avg}");
		}
	}
}
