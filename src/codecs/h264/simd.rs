//! Vector versions of the H.264 reconstruction hot loops, written with the
//! portable types of [`crate::simd`] (SSE2 / NEON, AVX2 by run-time
//! dispatch). Each one is bit-exact with its scalar original in `recon.rs`
//! or `deblock.rs`, which stays the fallback and the reference the tests
//! compare against. FFmpeg: `h264_qpel_8bit.asm`, `h264_chromamc.asm`,
//! `h264_deblock.asm`, `h264_idct.asm`.

use super::recon::{Plane, Weighting, fetch};
use crate::multiversion;
use crate::simd::{I16x8, I32x4, U8x16, transpose4x4, transpose8x8};

/// A source block: sample (i, j) of the window is
/// `data[base + j * stride + i]`.
#[derive(Clone, Copy)]
struct Window<'a> {
	data: &'a [u8],
	base: usize,
	stride: usize,
}

impl<'a> Window<'a> {
	#[inline(always)]
	fn row(&self, j: usize, i: usize) -> &'a [u8] {
		&self.data[self.base + j * self.stride + i..]
	}

	/// 8 samples of row `j` from column `i`, widened.
	#[inline(always)]
	fn px(&self, j: usize, i: usize) -> I16x8 {
		I16x8::load_u8(self.row(j, i))
	}
}

/// Scratch for an edge-emulated window (at most 21 x 21 samples).
const EMU: usize = 24;

/// Whether the `w` x `h` area at (x, y) lies inside the plane.
#[inline(always)]
fn inside(r: &Plane, x: i32, y: i32, w: usize, h: usize) -> bool {
	x >= 0 && y >= 0 && x as usize + w <= r.width && y as usize + h <= r.height
}

/// The `w` x `h` area at (x, y) with the picture edges repeated.
#[inline(never)]
fn emulated(r: &Plane, x: i32, y: i32, w: usize, h: usize) -> [u8; EMU * EMU] {
	let mut emu = [0u8; EMU * EMU];
	fetch(r, x, y, w, h, &mut emu, EMU);
	emu
}

/// The window at (x, y): borrowed straight from the plane when inside,
/// else built in `emu` (only then, so the common case does no copying).
macro_rules! window {
	($r:expr, $x:expr, $y:expr, $w:expr, $h:expr, $emu:ident) => {
		if inside($r, $x, $y, $w, $h) {
			Window { data: &$r.data, base: $y as usize * $r.width + $x as usize, stride: $r.width }
		} else {
			$emu = emulated($r, $x, $y, $w, $h);
			Window { data: &$emu, base: 0, stride: EMU }
		}
	};
}

/// `a + f - 5 (b + e) + 20 (c + d)` lane-wise (fits in 16 bits for 8-bit
/// samples: -2550..=10710).
#[inline(always)]
fn tap6(v: [I16x8; 6]) -> I16x8 {
	let five = I16x8::splat(5);
	let twenty = I16x8::splat(20);
	v[0].add(v[5]).sub(v[1].add(v[4]).mul(five)).add(v[2].add(v[3]).mul(twenty))
}

/// Unrounded horizontal 6-tap of window row `j`, columns `c..c + 8`.
#[inline(always)]
fn h_tap(w: &Window, j: usize, c: usize) -> I16x8 {
	tap6(std::array::from_fn(|k| w.px(j, c + k)))
}

/// `clip((v + 16) >> 5)` into the low 8 bytes.
#[inline(always)]
fn round5(v: I16x8) -> U8x16 {
	let r = v.add(I16x8::splat(16)).shr(5);
	r.pack_u8(r)
}

/// The centre sample `j` from six unrounded horizontal intermediates:
/// `clip((t0 + t5 - 5 (t1 + t4) + 20 (t2 + t3) + 512) >> 10)`, in 32 bits.
#[inline(always)]
fn centre(t: &[I16x8; 6]) -> U8x16 {
	let s05 = t[0].add(t[5]);
	let s14 = t[1].add(t[4]);
	let s23 = t[2].add(t[3]);
	let k = I16x8::from_array([-5, 20, -5, 20, -5, 20, -5, 20]);
	let r = I32x4::splat(512);
	let lo = s14.zip_lo(s23).madd(k).add(s05.widen_lo()).add(r).shr(10);
	let hi = s14.zip_hi(s23).madd(k).add(s05.widen_hi()).add(r).shr(10);
	let v = lo.pack_i16(hi);
	v.pack_u8(v)
}

#[inline(always)]
fn store_n(v: U8x16, out: &mut [u8], n: usize) {
	match n {
		16 => v.store(out),
		8 => v.store8(out),
		_ => v.store4(out),
	}
}

multiversion! {
	/// [`super::recon::mc_luma`] for `bw` in {4, 8, 16}.
	#[allow(clippy::too_many_arguments)]
	pub fn mc_luma(r: &Plane, x: i32, y: i32, fx: i32, fy: i32, bw: usize, bh: usize, out: &mut [u8]) {
		// Groups of 8 columns read 13 samples per row, so a 4-wide block
		// still needs a 13-wide window.
		let cols = bw.max(8) + 5;
		let emu;
		let w = window!(r, x - 2, y - 2, cols, bh + 5, emu);
		luma_block(&w, fx, fy, bw, bh, out);
	}
}

/// The body of [`mc_luma`] on a window whose (0, 0) is the block's
/// (-2, -2).
#[inline(always)]
fn luma_block(w: &Window, fx: i32, fy: i32, bw: usize, bh: usize, out: &mut [u8]) {
	let n = bw.min(8);
	let need_t = matches!((fx, fy), (2, 1..=3) | (1 | 3, 2));
	let need_v2 = matches!((fx, fy), (0, 1..=3) | (1, 1 | 3));
	let need_v3 = matches!((fx, fy), (3, 1 | 3));
	let mut c = 0;
	while c < bw {
		// Sliding windows over rows j..j+6: unrounded horizontal taps
		// (for the centre), and raw columns c+2 / c+3 (for `h` / `m`).
		let mut t: [I16x8; 6] =
			if need_t { std::array::from_fn(|k| h_tap(w, k, c)) } else { [I16x8::splat(0); 6] };
		let mut v2: [I16x8; 6] =
			if need_v2 { std::array::from_fn(|k| w.px(k, c + 2)) } else { [I16x8::splat(0); 6] };
		let mut v3: [I16x8; 6] =
			if need_v3 { std::array::from_fn(|k| w.px(k, c + 3)) } else { [I16x8::splat(0); 6] };
		for j in 0..bh {
			let g = |dx: usize, dy: usize| U8x16::load8(w.row(j + 2 + dy, c + 2 + dx));
			let b = || if need_t { round5(t[2]) } else { round5(h_tap(w, j + 2, c)) };
			let s = || if need_t { round5(t[3]) } else { round5(h_tap(w, j + 3, c)) };
			let h = || round5(tap6(v2));
			let m = || round5(tap6(v3));
			let v = match (fx, fy) {
				(0, 0) => g(0, 0),
				(1, 0) => g(0, 0).avg(b()),
				(2, 0) => b(),
				(3, 0) => b().avg(g(1, 0)),
				(0, 1) => g(0, 0).avg(h()),
				(0, 2) => h(),
				(0, 3) => h().avg(g(0, 1)),
				(2, 2) => centre(&t),
				(2, 1) => b().avg(centre(&t)),
				(2, 3) => centre(&t).avg(s()),
				(1, 2) => {
					let hh = round5(tap6(std::array::from_fn(|k| w.px(j + k, c + 2))));
					hh.avg(centre(&t))
				}
				(3, 2) => {
					let mm = round5(tap6(std::array::from_fn(|k| w.px(j + k, c + 3))));
					centre(&t).avg(mm)
				}
				(1, 1) => b().avg(h()),
				(3, 1) => b().avg(m()),
				(1, 3) => h().avg(s()),
				_ => m().avg(s()),
			};
			store_n(v, &mut out[j * bw + c..], n);
			if j + 1 < bh {
				if need_t {
					t.copy_within(1.., 0);
					t[5] = h_tap(w, j + 6, c);
				}
				if need_v2 {
					v2.copy_within(1.., 0);
					v2[5] = w.px(j + 6, c + 2);
				}
				if need_v3 {
					v3.copy_within(1.., 0);
					v3[5] = w.px(j + 6, c + 3);
				}
			}
		}
		c += 8;
	}
}

multiversion! {
	/// [`super::recon::mc_chroma`] for `bw` in {4, 8}.
	#[allow(clippy::too_many_arguments)]
	pub fn mc_chroma(r: &Plane, x: i32, y: i32, fx: i32, fy: i32, bw: usize, bh: usize, out: &mut [u8]) {
		// Each row reads 9 samples (8 lanes and their right neighbours).
		let emu;
		let w = window!(r, x, y, 9, bh + 1, emu);
		match bw {
			8 => chroma_block::<8>(&w, fx, fy, bh, out),
			_ => chroma_block::<4>(&w, fx, fy, bh, out),
		}
	}
}

#[inline(always)]
fn chroma_block<const BW: usize>(w: &Window, fx: i32, fy: i32, bh: usize, out: &mut [u8]) {
	let k = |v: i32| I16x8::splat(v as i16);
	let (w00, w10, w01, w11) =
		(k((8 - fx) * (8 - fy)), k(fx * (8 - fy)), k((8 - fx) * fy), k(fx * fy));
	let mut top = (w.px(0, 0), w.px(0, 1));
	for j in 0..bh {
		let bot = (w.px(j + 1, 0), w.px(j + 1, 1));
		let v = w00
			.mul(top.0)
			.add(w10.mul(top.1))
			.add(w01.mul(bot.0))
			.add(w11.mul(bot.1))
			.add(I16x8::splat(32))
			.shr(6);
		store_n(v.pack_u8(v), &mut out[j * BW..], BW);
		top = bot;
	}
}

multiversion! {
	/// [`super::recon::store_prediction`] for `bw` in {4, 8, 16}, except
	/// weights outside the 8-bit range (which the scalar code handles).
	#[allow(clippy::too_many_arguments)]
	pub fn store_prediction(
		p: &mut Plane,
		x: usize,
		y: usize,
		bw: usize,
		bh: usize,
		l0: Option<&[u8]>,
		l1: Option<&[u8]>,
		wt: Weighting,
	) {
		match bw {
			16 => store_rows::<16>(p, x, y, bh, l0, l1, wt),
			8 => store_rows::<8>(p, x, y, bh, l0, l1, wt),
			_ => store_rows::<4>(p, x, y, bh, l0, l1, wt),
		}
	}
}

/// One prediction row of `BW` samples.
#[inline(always)]
fn load_row<const BW: usize>(s: &[u8], j: usize) -> U8x16 {
	match BW {
		16 => U8x16::load(&s[j * 16..]),
		8 => U8x16::load8(&s[j * 8..]),
		_ => U8x16::load4(&s[j * 4..]),
	}
}

#[inline(always)]
fn store_rows<const BW: usize>(
	p: &mut Plane,
	x: usize,
	y: usize,
	bh: usize,
	l0: Option<&[u8]>,
	l1: Option<&[u8]>,
	wt: Weighting,
) {
	let stride = p.width;
	let data = &mut p.data[y * stride + x..];
	let mut put = |j: usize, v: U8x16| store_n(v, &mut data[j * stride..], BW);
	match (l0, l1, wt) {
		(Some(a), Some(b), Weighting::Default) => {
			for j in 0..bh {
				put(j, load_row::<BW>(a, j).avg(load_row::<BW>(b, j)));
			}
		}
		(Some(a), None, Weighting::Default) | (None, Some(a), Weighting::Default) => {
			for j in 0..bh {
				put(j, load_row::<BW>(a, j));
			}
		}
		(Some(a), Some(b), Weighting::Weighted { log_wd, w, o }) => {
			// Products of 8-bit samples and weights overflow 16 bits, so
			// pairs (a, b) x (w0, w1) are summed by `madd` in 32 bits.
			let ws =
				I16x8::from_array([w[0], w[1], w[0], w[1], w[0], w[1], w[0], w[1]].map(|v| v as i16));
			let round = I32x4::splat(1 << log_wd);
			let off = I32x4::splat((o[0] + o[1] + 1) >> 1);
			let half = |a: I16x8, b: I16x8| {
				let f = |pairs: I16x8| pairs.madd(ws).add(round).shr(log_wd + 1).add(off);
				f(a.zip_lo(b)).pack_i16(f(a.zip_hi(b)))
			};
			for j in 0..bh {
				let (va, vb) = (load_row::<BW>(a, j), load_row::<BW>(b, j));
				let lo = half(va.widen_lo(), vb.widen_lo());
				let hi = half(va.widen_hi(), vb.widen_hi());
				put(j, lo.pack_u8(hi));
			}
		}
		(Some(a), None, Weighting::Weighted { log_wd, w, o })
		| (None, Some(a), Weighting::Weighted { log_wd, w, o }) => {
			let list = usize::from(l0.is_none());
			// |v * w| <= 255 * 128 and the rounding add stay in 16 bits.
			let ww = I16x8::splat(w[list] as i16);
			let oo = I16x8::splat(o[list] as i16);
			let round = I16x8::splat(if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 });
			let f = |v: I16x8| v.mul(ww).add(round).shr(log_wd).add(oo);
			for j in 0..bh {
				let v = load_row::<BW>(a, j);
				put(j, f(v.widen_lo()).pack_u8(f(v.widen_hi())));
			}
		}
		(None, None, _) => {
			for j in 0..bh {
				put(j, U8x16::splat(128));
			}
		}
	}
}

/// Whether [`store_prediction`] covers these weights exactly: 8-bit
/// weights and offsets, and `log_wd <= 7`, so that the unipred path stays
/// within 16 bits and the bipred one within 32.
pub fn weights_supported(wt: Weighting) -> bool {
	match wt {
		Weighting::Default => true,
		Weighting::Weighted { log_wd, w, o } => {
			log_wd <= 7
				&& w.iter().chain(&o).all(|v| (-128..=128).contains(v))
				&& o.iter().all(|&v| v < 128)
		}
	}
}

// ---------------------------------------------------------------------------
// Deblocking

/// Filters 8 lines across one edge (8.7.2.3-4). `px` holds the samples
/// `p3 p2 p1 p0 q0 q1 q2 q3` of each line, one line per lane; `bs` and
/// `tc0` are per line. Returns the filtered samples.
#[inline(always)]
fn filter_lines(
	px: [I16x8; 8],
	bs: I16x8,
	tc0: I16x8,
	alpha: i16,
	beta: i16,
	chroma: bool,
) -> [I16x8; 8] {
	let [p3, p2, p1, p0, q0, q1, q2, q3] = px;
	let zero = I16x8::splat(0);
	let one = I16x8::splat(1);
	let two = I16x8::splat(2);
	let four = I16x8::splat(4);
	let alpha_v = I16x8::splat(alpha);
	let beta_v = I16x8::splat(beta);
	let max = I16x8::splat(255);
	let filter = bs
		.gt(zero)
		.and(p0.sub(q0).abs().lt(alpha_v))
		.and(p1.sub(p0).abs().lt(beta_v))
		.and(q1.sub(q0).abs().lt(beta_v));
	if !filter.any() {
		return px;
	}
	let strong_bs = bs.eq(four);
	// Normal filter (bS < 4) delta, shared by luma and chroma.
	let normal = |tc: I16x8| {
		let delta = q0.sub(p0).shl(2).add(p1.sub(q1)).add(four).shr(3).clamp(zero.sub(tc), tc);
		(p0.add(delta).clamp(zero, max), q0.sub(delta).clamp(zero, max))
	};
	if chroma {
		let (np0, nq0) = normal(tc0.add(one));
		let sp0 = p1.shl(1).add(p0).add(q1).add(two).shr(2);
		let sq0 = q1.shl(1).add(q0).add(p1).add(two).shr(2);
		let p0n = filter.select(strong_bs.select(sp0, np0), p0);
		let q0n = filter.select(strong_bs.select(sq0, nq0), q0);
		return [p3, p2, p1, p0n, q0n, q1, q2, q3];
	}
	let ap = p2.sub(p0).abs().lt(beta_v);
	let aq = q2.sub(q0).abs().lt(beta_v);
	// bS < 4. The masks are -1, so subtracting them counts.
	let tc = tc0.sub(ap).sub(aq);
	let (np0, nq0) = normal(tc);
	let avg0 = p0.add(q0).add(one).shr(1);
	let ntc0 = zero.sub(tc0);
	let np1 = p1.add(p2.add(avg0).sub(p1.shl(1)).shr(1).clamp(ntc0, tc0));
	let nq1 = q1.add(q2.add(avg0).sub(q1.shl(1)).shr(1).clamp(ntc0, tc0));
	let np1 = ap.select(np1, p1);
	let nq1 = aq.select(nq1, q1);
	// bS == 4.
	let strong = p0.sub(q0).abs().lt(I16x8::splat((alpha >> 2) + 2));
	let sp = ap.and(strong);
	let sq = aq.and(strong);
	let pq0 = p0.add(q0);
	let sp0 = p2.add(p1.shl(1)).add(pq0.shl(1)).add(q1).add(four).shr(3);
	let sp1 = p2.add(p1).add(pq0).add(two).shr(2);
	let sp2 = p3.shl(1).add(p2.mul(I16x8::splat(3))).add(p1).add(pq0).add(four).shr(3);
	let wp0 = p1.shl(1).add(p0).add(q1).add(two).shr(2);
	let sq0 = q2.add(q1.shl(1)).add(pq0.shl(1)).add(p1).add(four).shr(3);
	let sq1 = q2.add(q1).add(pq0).add(two).shr(2);
	let sq2 = q3.shl(1).add(q2.mul(I16x8::splat(3))).add(q1).add(pq0).add(four).shr(3);
	let wq0 = q1.shl(1).add(q0).add(p1).add(two).shr(2);
	let s_p0 = sp.select(sp0, wp0);
	let s_p1 = sp.select(sp1, p1);
	let s_p2 = sp.select(sp2, p2);
	let s_q0 = sq.select(sq0, wq0);
	let s_q1 = sq.select(sq1, q1);
	let s_q2 = sq.select(sq2, q2);
	let pick = |s: I16x8, n: I16x8, orig: I16x8| filter.select(strong_bs.select(s, n), orig);
	[
		p3,
		pick(s_p2, p2, p2),
		pick(s_p1, np1, p1),
		pick(s_p0, np0, p0),
		pick(s_q0, nq0, q0),
		pick(s_q1, nq1, q1),
		pick(s_q2, q2, q2),
		q3,
	]
}

multiversion! {
	/// [`super::deblock`]'s edge filter: `len` lines (16 luma or 8 chroma)
	/// starting at (x, y), with one strength and `tc0` per segment.
	#[allow(clippy::too_many_arguments)]
	pub fn filter_edge(
		plane: &mut Plane,
		x: usize,
		y: usize,
		len: usize,
		vertical: bool,
		bs: &[u8; 4],
		tc0: &[i32; 4],
		alpha: i32,
		beta: i32,
		chroma: bool,
	) {
		let w = plane.width;
		let seg = len / 4;
		for g in 0..len / 8 {
			let line = |l: usize| (g * 8 + l) / seg;
			if (0..8).all(|l| bs[line(l)] == 0) {
				continue;
			}
			let bs_v = I16x8::from_array(std::array::from_fn(|l| i16::from(bs[line(l)])));
			let tc_v = I16x8::from_array(std::array::from_fn(|l| tc0[line(l)] as i16));
			let first = g * 8;
			if vertical {
				// Lines are rows: 8 bytes p3..q3 per row, transposed.
				let rows: [I16x8; 8] = std::array::from_fn(|l| I16x8::load_u8(&plane.data[(y + first + l) * w + x - 4..]));
				let out = filter_lines(transpose8x8(rows), bs_v, tc_v, alpha as i16, beta as i16, chroma);
				let back = transpose8x8(out);
				for (l, r) in back.iter().enumerate() {
					r.pack_u8(*r).store8(&mut plane.data[(y + first + l) * w + x - 4..]);
				}
			} else {
				// Lines are columns: each sample row is contiguous. Chroma
				// reads only p1..q1.
				let base = x + first;
				let (lo, hi) = if chroma { (2, 6) } else { (0, 8) };
				let mut px = [I16x8::splat(0); 8];
				for k in lo..hi {
					px[k] = I16x8::load_u8(&plane.data[(y + k - 4) * w + base..]);
				}
				let out = filter_lines(px, bs_v, tc_v, alpha as i16, beta as i16, chroma);
				let (lo, hi) = if chroma { (3, 5) } else { (1, 7) };
				for k in lo..hi {
					out[k].pack_u8(out[k]).store8(&mut plane.data[(y + k - 4) * w + base..]);
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Inverse transforms

/// One 4-point butterfly (8.5.12.2) on four vectors.
#[inline(always)]
fn idct4_1d(s: [I32x4; 4]) -> [I32x4; 4] {
	let e0 = s[0].add(s[2]);
	let e1 = s[0].sub(s[2]);
	let e2 = s[1].shr(1).sub(s[3]);
	let e3 = s[1].add(s[3].shr(1));
	[e0.add(e3), e1.add(e2), e1.sub(e2), e0.sub(e3)]
}

/// Adds `(v + 32) >> 6` of four 32-bit lanes to 4 pixels, clipped.
#[inline(always)]
fn add_row4(dst: &mut [u8], v: I32x4) {
	let r = v.add(I32x4::splat(32)).shr(6);
	let px = U8x16::load4(dst).widen_lo().widen_lo();
	let s = px.add(r).pack_i16(px.add(r));
	s.pack_u8(s).store4(dst);
}

/// [`super::recon::idct4_add`].
pub fn idct4_add(p: &mut Plane, x: usize, y: usize, c: &[i32; 16]) {
	// Lanes are rows for the row pass, so the coefficients go in
	// transposed; the result is transposed back so lanes are columns.
	let cols = transpose4x4(std::array::from_fn(|i| I32x4::load(&c[i * 4..])));
	let rows = transpose4x4(idct4_1d(cols));
	let out = idct4_1d(rows);
	for (i, v) in out.iter().enumerate() {
		add_row4(&mut p.data[(y + i) * p.width + x..], *v);
	}
}

/// One 8-point butterfly (8.5.13.2), lane-wise.
#[inline(always)]
fn idct8_1d(s: [I32x4; 8]) -> [I32x4; 8] {
	let a0 = s[0].add(s[4]);
	let a4 = s[0].sub(s[4]);
	let a2 = s[2].shr(1).sub(s[6]);
	let a6 = s[2].add(s[6].shr(1));
	let b0 = a0.add(a6);
	let b2 = a4.add(a2);
	let b4 = a4.sub(a2);
	let b6 = a0.sub(a6);
	let a1 = s[5].sub(s[3]).sub(s[7]).sub(s[7].shr(1));
	let a3 = s[1].add(s[7]).sub(s[3]).sub(s[3].shr(1));
	let a5 = s[7].sub(s[1]).add(s[5]).add(s[5].shr(1));
	let a7 = s[3].add(s[5]).add(s[1]).add(s[1].shr(1));
	let b1 = a1.add(a7.shr(2));
	let b7 = a7.sub(a1.shr(2));
	let b3 = a3.add(a5.shr(2));
	let b5 = a3.shr(2).sub(a5);
	[b0.add(b7), b2.add(b5), b4.add(b3), b6.add(b1), b6.sub(b1), b4.sub(b3), b2.sub(b5), b0.sub(b7)]
}

/// Transposes an 8x8 block held as `[row][half]` of 4-lane vectors.
#[inline(always)]
fn transpose8x8_i32(m: [[I32x4; 2]; 8]) -> [[I32x4; 2]; 8] {
	let mut out = [[I32x4::splat(0); 2]; 8];
	for bi in 0..2 {
		for bj in 0..2 {
			let t = transpose4x4(std::array::from_fn(|k| m[bi * 4 + k][bj]));
			for k in 0..4 {
				out[bj * 4 + k][bi] = t[k];
			}
		}
	}
	out
}

/// Runs [`idct8_1d`] on the columns of `m` (lanes across each row).
#[inline(always)]
fn idct8_cols(m: [[I32x4; 2]; 8]) -> [[I32x4; 2]; 8] {
	let lo = idct8_1d(std::array::from_fn(|i| m[i][0]));
	let hi = idct8_1d(std::array::from_fn(|i| m[i][1]));
	std::array::from_fn(|i| [lo[i], hi[i]])
}

multiversion! {
	/// [`super::recon::idct8_add`].
	pub fn idct8_add(p: &mut Plane, x: usize, y: usize, c: &[i32; 64]) {
		let m: [[I32x4; 2]; 8] = std::array::from_fn(|i| [I32x4::load(&c[i * 8..]), I32x4::load(&c[i * 8 + 4..])]);
		// Row pass on the transpose, then the column pass in place.
		let t = transpose8x8_i32(idct8_cols(transpose8x8_i32(m)));
		let out = idct8_cols(t);
		let round = I32x4::splat(32);
		for (i, [lo, hi]) in out.iter().enumerate() {
			let at = (y + i) * p.width + x;
			let px = U8x16::load8(&p.data[at..]).widen_lo();
			let r = lo.add(round).shr(6).pack_i16(hi.add(round).shr(6));
			// Residuals are within 16 bits only after saturation; the sum
			// is done in 16 bits with saturation, which clips identically
			// once packed to 8 bits.
			let s = px.saturating_add(r);
			s.pack_u8(s).store8(&mut p.data[at..]);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::super::consts::{ALPHA, BETA, TC0};
	use super::super::deblock::filter_edge_scalar;
	use super::super::recon::{self, Plane, Weighting};
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

	fn noise_plane(rng: &mut Rng, w: usize, h: usize) -> Plane {
		let mut p = Plane::new(w, h, 0);
		for v in &mut p.data {
			*v = rng.next() as u8;
		}
		p
	}

	/// A plane of flat 4x4 blocks plus small noise, so that edge filters
	/// actually trigger.
	fn blocky_plane(rng: &mut Rng, w: usize, h: usize) -> Plane {
		let levels: Vec<i32> = (0..(w / 4 + 1) * (h / 4 + 1)).map(|_| rng.range(0, 255)).collect();
		let mut p = Plane::new(w, h, 0);
		let amp = rng.range(0, 6);
		for y in 0..h {
			for x in 0..w {
				let base = levels[(y / 4) * (w / 4 + 1) + x / 4];
				p.data[y * w + x] = (base + rng.range(-amp, amp)).clamp(0, 255) as u8;
			}
		}
		p
	}

	#[test]
	fn mc_luma_matches_scalar_at_every_fraction_and_edge() {
		let mut rng = Rng(1);
		let r = noise_plane(&mut rng, 48, 40);
		for (bw, bh) in [(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
			for fy in 0..4 {
				for fx in 0..4 {
					for _ in 0..12 {
						let (x, y) = (rng.range(-24, 56), rng.range(-24, 48));
						let mut a = [0u8; 256];
						let mut b = [0u8; 256];
						recon::mc_luma_scalar(&r, x, y, fx, fy, bw, bh, &mut a);
						mc_luma(&r, x, y, fx, fy, bw, bh, &mut b);
						assert_eq!(a[..], b[..], "{bw}x{bh} frac ({fx},{fy}) at ({x},{y})");
					}
				}
			}
		}
	}

	#[test]
	fn mc_chroma_matches_scalar() {
		let mut rng = Rng(2);
		let r = noise_plane(&mut rng, 24, 20);
		for (bw, bh) in [(8, 8), (8, 4), (4, 8), (4, 4), (4, 2)] {
			for fy in 0..8 {
				for fx in 0..8 {
					for _ in 0..6 {
						let (x, y) = (rng.range(-12, 28), rng.range(-12, 24));
						let mut a = [0u8; 64];
						let mut b = [0u8; 64];
						recon::mc_chroma_scalar(&r, x, y, fx, fy, bw, bh, &mut a);
						mc_chroma(&r, x, y, fx, fy, bw, bh, &mut b);
						assert_eq!(a, b, "{bw}x{bh} frac ({fx},{fy}) at ({x},{y})");
					}
				}
			}
		}
	}

	#[test]
	fn store_prediction_matches_scalar_for_all_weightings() {
		let mut rng = Rng(3);
		for round in 0..3000 {
			let bw = [4, 8, 16][round % 3];
			let bh = [4, 8, 16][(round / 3) % 3];
			let l0: Vec<u8> = (0..256).map(|_| rng.next() as u8).collect();
			let l1: Vec<u8> = (0..256).map(|_| rng.next() as u8).collect();
			let wt = if round % 4 == 0 {
				Weighting::Default
			} else {
				Weighting::Weighted {
					log_wd: rng.range(0, 7) as u32,
					w: [rng.range(-128, 128), rng.range(-128, 128)],
					o: [rng.range(-128, 127), rng.range(-128, 127)],
				}
			};
			let lists = [(true, true), (true, false), (false, true), (false, false)][(round / 9) % 4];
			let (a, b) = (lists.0.then_some(&l0[..]), lists.1.then_some(&l1[..]));
			let mut p1 = noise_plane(&mut rng, 32, 32);
			let mut p2 = p1.clone();
			assert!(weights_supported(wt));
			recon::store_prediction_scalar(&mut p1, 8, 4, bw, bh, a, b, wt);
			store_prediction(&mut p2, 8, 4, bw, bh, a, b, wt);
			assert_eq!(p1.data, p2.data, "{bw}x{bh} {wt:?} {lists:?}");
		}
	}

	#[test]
	fn deblock_edge_matches_scalar() {
		let mut rng = Rng(4);
		let mut changed = [0usize; 4];
		for round in 0..4000 {
			let chroma = round % 2 == 1;
			let vertical = (round / 2) % 2 == 1;
			let len = if chroma { 8 } else { 16 };
			let mut p1 = blocky_plane(&mut rng, 32, 32);
			let orig = p1.clone();
			let mut p2 = p1.clone();
			let qp = rng.range(10, 51) as usize;
			let index_a = (qp as i32 + rng.range(-12, 12)).clamp(0, 51) as usize;
			let index_b = (qp as i32 + rng.range(-12, 12)).clamp(0, 51) as usize;
			let (alpha, beta) = (i32::from(ALPHA[index_a]), i32::from(BETA[index_b]));
			let bs: [u8; 4] = std::array::from_fn(|_| [0, 1, 2, 3, 4, 4][rng.range(0, 5) as usize]);
			let tc0: [i32; 4] = bs
				.map(|s| if (1..4).contains(&s) { i32::from(TC0[index_a][usize::from(s) - 1]) } else { 0 });
			let (x, y) = (8, 8);
			filter_edge_scalar(&mut p1, x, y, len, vertical, &bs, &tc0, alpha, beta, chroma);
			filter_edge(&mut p2, x, y, len, vertical, &bs, &tc0, alpha, beta, chroma);
			assert_eq!(
				p1.data, p2.data,
				"chroma {chroma} vertical {vertical} bs {bs:?} a {alpha} b {beta}"
			);
			if p1.data != orig.data {
				changed[round % 4] += 1;
			}
		}
		// Every combination must actually have filtered a good share.
		assert!(changed.iter().all(|&n| n > 300), "{changed:?}");
	}

	#[test]
	fn idct_matches_scalar_including_overflowing_coefficients() {
		let mut rng = Rng(5);
		for round in 0..4000 {
			let big = round % 5 == 0;
			let lim = if big { 1 << 20 } else { 1 << 12 };
			let mut c4 = [0i32; 16];
			let mut c8 = [0i32; 64];
			for v in c4.iter_mut().chain(c8.iter_mut()) {
				if !rng.next().is_multiple_of(3) {
					*v = rng.range(-lim, lim);
				}
			}
			let mut p1 = noise_plane(&mut rng, 16, 16);
			let mut p2 = p1.clone();
			recon::idct4_add_scalar(&mut p1, 4, 8, &c4);
			idct4_add(&mut p2, 4, 8, &c4);
			assert_eq!(p1.data, p2.data, "idct4 {c4:?}");
			recon::idct8_add_scalar(&mut p1, 8, 0, &c8);
			idct8_add(&mut p2, 8, 0, &c8);
			assert_eq!(p1.data, p2.data, "idct8 {c8:?}");
		}
	}
}
