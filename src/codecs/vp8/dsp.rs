//! VP8 signal processing: inverse transforms, intra predictors, sub-pixel
//! interpolation and the loop filters (RFC 6386 sections 12, 14, 15 and
//! 18; libvpx `vp8/common/idctllm.c`, `vpx_dsp/intrapred.c`,
//! `vp8/common/filter.c`, `loopfilter_filters.c`). 16-bit intermediate
//! values are truncated exactly as libvpx's `short` arithmetic does.

const COS_PI8_SQRT2_MINUS1: i32 = 20091;
const SIN_PI8_SQRT2: i32 = 35468;

/// Inverse 4x4 DCT of dequantized `coeffs`, added to the 4x4 block at
/// `dst[at..]` with `stride` (`vp8_short_idct4x4llm_c`).
pub fn idct_add(coeffs: &[i16; 16], dst: &mut [u8], at: usize, stride: usize) {
	let mut tmp = [0i16; 16];
	for i in 0..4 {
		let ip = |k: usize| i32::from(coeffs[i + k]);
		let a1 = ip(0) + ip(8);
		let b1 = ip(0) - ip(8);
		let temp1 = (ip(4) * SIN_PI8_SQRT2) >> 16;
		let temp2 = ip(12) + ((ip(12) * COS_PI8_SQRT2_MINUS1) >> 16);
		let c1 = temp1 - temp2;
		let temp1 = ip(4) + ((ip(4) * COS_PI8_SQRT2_MINUS1) >> 16);
		let temp2 = (ip(12) * SIN_PI8_SQRT2) >> 16;
		let d1 = temp1 + temp2;
		tmp[i] = (a1 + d1) as i16;
		tmp[12 + i] = (a1 - d1) as i16;
		tmp[4 + i] = (b1 + c1) as i16;
		tmp[8 + i] = (b1 - c1) as i16;
	}
	let mut out = [0i16; 16];
	for i in 0..4 {
		let ip = |k: usize| i32::from(tmp[4 * i + k]);
		let a1 = ip(0) + ip(2);
		let b1 = ip(0) - ip(2);
		let temp1 = (ip(1) * SIN_PI8_SQRT2) >> 16;
		let temp2 = ip(3) + ((ip(3) * COS_PI8_SQRT2_MINUS1) >> 16);
		let c1 = temp1 - temp2;
		let temp1 = ip(1) + ((ip(1) * COS_PI8_SQRT2_MINUS1) >> 16);
		let temp2 = (ip(3) * SIN_PI8_SQRT2) >> 16;
		let d1 = temp1 + temp2;
		out[4 * i] = ((a1 + d1 + 4) >> 3) as i16;
		out[4 * i + 3] = ((a1 - d1 + 4) >> 3) as i16;
		out[4 * i + 1] = ((b1 + c1 + 4) >> 3) as i16;
		out[4 * i + 2] = ((b1 - c1 + 4) >> 3) as i16;
	}
	for r in 0..4 {
		let row = &mut dst[at + r * stride..at + r * stride + 4];
		for (c, px) in row.iter_mut().enumerate() {
			*px = (i32::from(out[4 * r + c]) + i32::from(*px)).clamp(0, 255) as u8;
		}
	}
}

/// Adds a DC-only inverse transform (`vp8_dc_only_idct_add_c`).
pub fn dc_add(dc: i16, dst: &mut [u8], at: usize, stride: usize) {
	let a1 = (i32::from(dc) + 4) >> 3;
	for r in 0..4 {
		for px in &mut dst[at + r * stride..at + r * stride + 4] {
			*px = (a1 + i32::from(*px)).clamp(0, 255) as u8;
		}
	}
}

/// Inverse Walsh-Hadamard transform of the Y2 block; returns the 16 luma
/// DC values (`vp8_short_inv_walsh4x4_c`).
pub fn inv_walsh(input: &[i16; 16]) -> [i16; 16] {
	let mut tmp = [0i16; 16];
	for i in 0..4 {
		let ip = |k: usize| i32::from(input[i + k]);
		let a1 = ip(0) + ip(12);
		let b1 = ip(4) + ip(8);
		let c1 = ip(4) - ip(8);
		let d1 = ip(0) - ip(12);
		tmp[i] = (a1 + b1) as i16;
		tmp[4 + i] = (c1 + d1) as i16;
		tmp[8 + i] = (a1 - b1) as i16;
		tmp[12 + i] = (d1 - c1) as i16;
	}
	let mut out = [0i16; 16];
	for i in 0..4 {
		let ip = |k: usize| i32::from(tmp[4 * i + k]);
		let a1 = ip(0) + ip(3);
		let b1 = ip(1) + ip(2);
		let c1 = ip(1) - ip(2);
		let d1 = ip(0) - ip(3);
		out[4 * i] = ((a1 + b1 + 3) >> 3) as i16;
		out[4 * i + 1] = ((c1 + d1 + 3) >> 3) as i16;
		out[4 * i + 2] = ((a1 - b1 + 3) >> 3) as i16;
		out[4 * i + 3] = ((d1 - c1 + 3) >> 3) as i16;
	}
	out
}

/// DC-only Walsh transform (`vp8_short_inv_walsh4x4_1_c`).
pub fn inv_walsh_dc(dc: i16) -> i16 {
	((i32::from(dc) + 3) >> 3) as i16
}

// ---------------------------------------------------------------------------
// Intra prediction

/// Macroblock-level modes.
pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const TM_PRED: u8 = 3;
pub const B_PRED: u8 = 4;

/// Sub-block modes.
pub const B_DC_PRED: u8 = 0;
pub const B_TM_PRED: u8 = 1;
pub const B_VE_PRED: u8 = 2;
pub const B_HE_PRED: u8 = 3;
pub const B_LD_PRED: u8 = 4;
pub const B_RD_PRED: u8 = 5;
pub const B_VR_PRED: u8 = 6;
pub const B_VL_PRED: u8 = 7;
pub const B_HD_PRED: u8 = 8;
pub const B_HU_PRED: u8 = 9;

/// Predicts an `n`x`n` block (16 or 8) into `dst` (stride `n`). `above` and
/// `left` hold `n` edge pixels, `top_left` the corner; `have_above` /
/// `have_left` select the DC variant at frame edges.
#[allow(clippy::too_many_arguments)]
pub fn predict_mb(
	mode: u8,
	n: usize,
	above: &[u8],
	left: &[u8],
	top_left: u8,
	have_above: bool,
	have_left: bool,
	dst: &mut [u8],
) {
	match mode {
		V_PRED => {
			for r in 0..n {
				dst[r * n..r * n + n].copy_from_slice(&above[..n]);
			}
		}
		H_PRED => {
			for r in 0..n {
				dst[r * n..r * n + n].fill(left[r]);
			}
		}
		TM_PRED => {
			for r in 0..n {
				for c in 0..n {
					let v = i32::from(left[r]) + i32::from(above[c]) - i32::from(top_left);
					dst[r * n + c] = v.clamp(0, 255) as u8;
				}
			}
		}
		_ => {
			let shift = if n == 16 { 4 } else { 3 };
			let sum = |e: &[u8]| e[..n].iter().map(|&p| u32::from(p)).sum::<u32>();
			let dc = match (have_above, have_left) {
				(true, true) => (sum(above) + sum(left) + n as u32) >> (shift + 1),
				(true, false) => (sum(above) + (n as u32 >> 1)) >> shift,
				(false, true) => (sum(left) + (n as u32 >> 1)) >> shift,
				(false, false) => 128,
			};
			dst[..n * n].fill(dc as u8);
		}
	}
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
	((u32::from(a) + 2 * u32::from(b) + u32::from(c) + 2) >> 2) as u8
}

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
	((u32::from(a) + u32::from(b) + 1) >> 1) as u8
}

/// Predicts a 4x4 sub-block. `above` holds the corner then 8 pixels (4
/// above, 4 above-right); `left` 4 pixels.
pub fn predict_4x4(mode: u8, above: &[u8; 9], left: &[u8; 4], out: &mut [[u8; 4]; 4]) {
	let x = above[0];
	let a = &above[1..];
	let (i, j, k, l) = (left[0], left[1], left[2], left[3]);
	let mut set = |col: usize, row: usize, v: u8| out[row][col] = v;
	match mode {
		B_DC_PRED => {
			let s: u32 = a[..4].iter().chain(left).map(|&p| u32::from(p)).sum();
			let dc = ((s + 4) >> 3) as u8;
			*out = [[dc; 4]; 4];
		}
		B_TM_PRED => {
			for (r, row) in out.iter_mut().enumerate() {
				for (c, px) in row.iter_mut().enumerate() {
					let v = i32::from(left[r]) + i32::from(a[c]) - i32::from(x);
					*px = v.clamp(0, 255) as u8;
				}
			}
		}
		B_VE_PRED => {
			let row = [
				avg3(x, a[0], a[1]),
				avg3(a[0], a[1], a[2]),
				avg3(a[1], a[2], a[3]),
				avg3(a[2], a[3], a[4]),
			];
			*out = [row; 4];
		}
		B_HE_PRED => {
			*out = [[avg3(x, i, j); 4], [avg3(i, j, k); 4], [avg3(j, k, l); 4], [avg3(k, l, l); 4]];
		}
		B_LD_PRED => {
			// vpx_d45e_predictor_4x4
			let (aa, b, c, d, e, f, g, h) = (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
			set(0, 0, avg3(aa, b, c));
			let v = avg3(b, c, d);
			set(1, 0, v);
			set(0, 1, v);
			let v = avg3(c, d, e);
			set(2, 0, v);
			set(1, 1, v);
			set(0, 2, v);
			let v = avg3(d, e, f);
			set(3, 0, v);
			set(2, 1, v);
			set(1, 2, v);
			set(0, 3, v);
			let v = avg3(e, f, g);
			set(3, 1, v);
			set(2, 2, v);
			set(1, 3, v);
			let v = avg3(f, g, h);
			set(3, 2, v);
			set(2, 3, v);
			set(3, 3, avg3(g, h, h));
		}
		B_RD_PRED => {
			// vpx_d135_predictor_4x4
			let (aa, b, c, d) = (a[0], a[1], a[2], a[3]);
			set(0, 3, avg3(j, k, l));
			let v = avg3(i, j, k);
			set(1, 3, v);
			set(0, 2, v);
			let v = avg3(x, i, j);
			set(2, 3, v);
			set(1, 2, v);
			set(0, 1, v);
			let v = avg3(aa, x, i);
			set(3, 3, v);
			set(2, 2, v);
			set(1, 1, v);
			set(0, 0, v);
			let v = avg3(b, aa, x);
			set(3, 2, v);
			set(2, 1, v);
			set(1, 0, v);
			let v = avg3(c, b, aa);
			set(3, 1, v);
			set(2, 0, v);
			set(3, 0, avg3(d, c, b));
		}
		B_VR_PRED => {
			// vpx_d117_predictor_4x4
			let (aa, b, c, d) = (a[0], a[1], a[2], a[3]);
			let v = avg2(x, aa);
			set(0, 0, v);
			set(1, 2, v);
			let v = avg2(aa, b);
			set(1, 0, v);
			set(2, 2, v);
			let v = avg2(b, c);
			set(2, 0, v);
			set(3, 2, v);
			set(3, 0, avg2(c, d));
			set(0, 3, avg3(k, j, i));
			set(0, 2, avg3(j, i, x));
			let v = avg3(i, x, aa);
			set(0, 1, v);
			set(1, 3, v);
			let v = avg3(x, aa, b);
			set(1, 1, v);
			set(2, 3, v);
			let v = avg3(aa, b, c);
			set(2, 1, v);
			set(3, 3, v);
			set(3, 1, avg3(b, c, d));
		}
		B_VL_PRED => {
			// vpx_d63e_predictor_4x4
			let (aa, b, c, d, e, f, g, h) = (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
			set(0, 0, avg2(aa, b));
			let v = avg2(b, c);
			set(1, 0, v);
			set(0, 2, v);
			let v = avg2(c, d);
			set(2, 0, v);
			set(1, 2, v);
			let v = avg2(d, e);
			set(3, 0, v);
			set(2, 2, v);
			set(3, 2, avg3(e, f, g));
			set(0, 1, avg3(aa, b, c));
			let v = avg3(b, c, d);
			set(1, 1, v);
			set(0, 3, v);
			let v = avg3(c, d, e);
			set(2, 1, v);
			set(1, 3, v);
			let v = avg3(d, e, f);
			set(3, 1, v);
			set(2, 3, v);
			set(3, 3, avg3(f, g, h));
		}
		B_HD_PRED => {
			// vpx_d153_predictor_4x4
			let (aa, b, c) = (a[0], a[1], a[2]);
			let v = avg2(i, x);
			set(0, 0, v);
			set(2, 1, v);
			let v = avg2(j, i);
			set(0, 1, v);
			set(2, 2, v);
			let v = avg2(k, j);
			set(0, 2, v);
			set(2, 3, v);
			set(0, 3, avg2(l, k));
			set(3, 0, avg3(aa, b, c));
			set(2, 0, avg3(x, aa, b));
			let v = avg3(i, x, aa);
			set(1, 0, v);
			set(3, 1, v);
			let v = avg3(j, i, x);
			set(1, 1, v);
			set(3, 2, v);
			let v = avg3(k, j, i);
			set(1, 2, v);
			set(3, 3, v);
			set(1, 3, avg3(l, k, j));
		}
		_ => {
			// B_HU_PRED, vpx_d207_predictor_4x4
			set(0, 0, avg2(i, j));
			let v = avg2(j, k);
			set(2, 0, v);
			set(0, 1, v);
			let v = avg2(k, l);
			set(2, 1, v);
			set(0, 2, v);
			set(1, 0, avg3(i, j, k));
			let v = avg3(j, k, l);
			set(3, 0, v);
			set(1, 1, v);
			let v = avg3(k, l, l);
			set(3, 1, v);
			set(1, 2, v);
			for (cc, rr) in [(3, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)] {
				set(cc, rr, l);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Inter prediction

const SUBPEL_FILTERS: [[i32; 6]; 8] = [
	[0, 0, 128, 0, 0, 0],
	[0, -6, 123, 12, -1, 0],
	[2, -11, 108, 36, -8, 1],
	[0, -9, 93, 50, -6, 0],
	[3, -16, 77, 77, -16, 3],
	[0, -6, 50, 93, -9, 0],
	[1, -8, 36, 108, -11, 2],
	[0, -1, 12, 123, -6, 0],
];

const BILINEAR_FILTERS: [[i32; 2]; 8] =
	[[128, 0], [112, 16], [96, 32], [80, 48], [64, 64], [48, 80], [32, 96], [16, 112]];

/// A reference plane, read with its edge pixels repeated without bound
/// (libvpx extends its 32-pixel border; motion vectors that reach further
/// are clamped to positions with the same result).
pub struct RefPlane<'a> {
	pub data: &'a [u8],
	pub stride: usize,
	pub width: usize,
	pub height: usize,
}

/// Source window of a predicted block: `w`x`h` plus 2 pixels before and 3
/// after on each axis for the filter taps.
const MAX_WIN: usize = 16 + 5;

impl RefPlane<'_> {
	/// Copies the `(w + 5) x (h + 5)` window whose block starts at
	/// `(x, y)` (full pixels, may be outside the plane).
	fn window(&self, x: isize, y: isize, w: usize, h: usize, out: &mut [u8; MAX_WIN * MAX_WIN]) {
		let (x0, y0) = (x - 2, y - 2);
		let (ww, wh) = (w + 5, h + 5);
		let inside =
			x0 >= 0 && y0 >= 0 && (x0 as usize + ww) <= self.width && (y0 as usize + wh) <= self.height;
		for r in 0..wh {
			let dst = &mut out[r * MAX_WIN..r * MAX_WIN + ww];
			if inside {
				let at = (y0 as usize + r) * self.stride + x0 as usize;
				dst.copy_from_slice(&self.data[at..at + ww]);
			} else {
				let sy = (y0 + r as isize).clamp(0, self.height as isize - 1) as usize;
				let row = &self.data[sy * self.stride..sy * self.stride + self.width];
				for (c, px) in dst.iter_mut().enumerate() {
					let sx = (x0 + c as isize).clamp(0, self.width as isize - 1) as usize;
					*px = row[sx];
				}
			}
		}
	}

	/// Predicts the `w`x`h` block at full-pixel `(x, y)` with sub-pixel
	/// offsets `fx`, `fy` (eighths) into `dst` (stride `dst_stride`).
	#[allow(clippy::too_many_arguments)]
	pub fn predict(
		&self,
		x: isize,
		y: isize,
		fx: usize,
		fy: usize,
		w: usize,
		h: usize,
		bilinear: bool,
		dst: &mut [u8],
		dst_stride: usize,
	) {
		let mut win = [0u8; MAX_WIN * MAX_WIN];
		self.window(x, y, w, h, &mut win);
		let src = |r: usize, c: usize| i32::from(win[(r + 2) * MAX_WIN + c + 2]);
		if fx == 0 && fy == 0 {
			for r in 0..h {
				for c in 0..w {
					dst[r * dst_stride + c] = src(r, c) as u8;
				}
			}
			return;
		}
		if bilinear {
			// First pass: h + 1 rows.
			let hf = BILINEAR_FILTERS[fx];
			let vf = BILINEAR_FILTERS[fy];
			let mut tmp = [0i32; 17 * 16];
			for r in 0..=h {
				for c in 0..w {
					tmp[r * w + c] = (src(r, c) * hf[0] + src(r, c + 1) * hf[1] + 64) >> 7;
				}
			}
			for r in 0..h {
				for c in 0..w {
					let v = (tmp[r * w + c] * vf[0] + tmp[(r + 1) * w + c] * vf[1] + 64) >> 7;
					dst[r * dst_stride + c] = v as u8;
				}
			}
			return;
		}
		// Six-tap: first pass over rows -2 .. h + 3, then vertically.
		let hf = SUBPEL_FILTERS[fx];
		let vf = SUBPEL_FILTERS[fy];
		let mut tmp = [0i32; MAX_WIN * 16];
		for r in 0..h + 5 {
			for c in 0..w {
				let s = |k: usize| i32::from(win[r * MAX_WIN + c + k]);
				let v = s(0) * hf[0]
					+ s(1) * hf[1]
					+ s(2) * hf[2]
					+ s(3) * hf[3]
					+ s(4) * hf[4]
					+ s(5) * hf[5]
					+ 64;
				tmp[r * w + c] = (v >> 7).clamp(0, 255);
			}
		}
		for r in 0..h {
			for c in 0..w {
				let t = |k: usize| tmp[(r + k) * w + c];
				let v = t(0) * vf[0]
					+ t(1) * vf[1]
					+ t(2) * vf[2]
					+ t(3) * vf[3]
					+ t(4) * vf[4]
					+ t(5) * vf[5]
					+ 64;
				dst[r * dst_stride + c] = (v >> 7).clamp(0, 255) as u8;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Loop filter

/// Thresholds of one filter level.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeLimits {
	/// Macroblock-edge limit (`mblim`).
	pub mb_limit: u8,
	/// Sub-block-edge limit (`blim`).
	pub sub_limit: u8,
	/// Interior limit (`lim`).
	pub interior: u8,
	/// High edge variance threshold.
	pub hev_thresh: u8,
}

#[inline]
fn c8(v: i32) -> i32 {
	v.clamp(-128, 127)
}

#[inline]
fn u2s(v: u8) -> i32 {
	i32::from(v) - 128
}

#[inline]
fn s2u(v: i32) -> u8 {
	(c8(v) + 128) as u8
}

/// The pixels across one edge position: `p3 p2 p1 p0 | q0 q1 q2 q3` at
/// `at + k * step`, k = -4..3.
#[inline]
fn tap(buf: &[u8], at: usize, step: usize, k: isize) -> u8 {
	buf[(at as isize + k * step as isize) as usize]
}

#[inline]
fn filter_mask(buf: &[u8], at: usize, step: usize, limit: u8, blimit: u8) -> bool {
	let p = |k| i32::from(tap(buf, at, step, k));
	let l = i32::from(limit);
	(p(-4) - p(-3)).abs() <= l
		&& (p(-3) - p(-2)).abs() <= l
		&& (p(-2) - p(-1)).abs() <= l
		&& (p(1) - p(0)).abs() <= l
		&& (p(2) - p(1)).abs() <= l
		&& (p(3) - p(2)).abs() <= l
		&& (p(-1) - p(0)).abs() * 2 + (p(-2) - p(1)).abs() / 2 <= i32::from(blimit)
}

#[inline]
fn hev(buf: &[u8], at: usize, step: usize, thresh: u8) -> bool {
	let p = |k| i32::from(tap(buf, at, step, k));
	(p(-2) - p(-1)).abs() > i32::from(thresh) || (p(1) - p(0)).abs() > i32::from(thresh)
}

/// Normal sub-block edge filter (`vp8_filter`).
#[inline]
fn inner_filter(buf: &mut [u8], at: usize, step: usize, hev: bool) {
	let i = |k: isize| (at as isize + k * step as isize) as usize;
	let ps1 = u2s(buf[i(-2)]);
	let ps0 = u2s(buf[i(-1)]);
	let qs0 = u2s(buf[i(0)]);
	let qs1 = u2s(buf[i(1)]);
	let mut f = if hev { c8(ps1 - qs1) } else { 0 };
	f = c8(f + 3 * (qs0 - ps0));
	let f1 = c8(f + 4) >> 3;
	let f2 = c8(f + 3) >> 3;
	buf[i(0)] = s2u(qs0 - f1);
	buf[i(-1)] = s2u(ps0 + f2);
	if !hev {
		let a = (f1 + 1) >> 1;
		buf[i(1)] = s2u(qs1 - a);
		buf[i(-2)] = s2u(ps1 + a);
	}
}

/// Normal macroblock edge filter (`vp8_mbfilter`).
#[inline]
fn mb_filter(buf: &mut [u8], at: usize, step: usize, hev: bool) {
	let i = |k: isize| (at as isize + k * step as isize) as usize;
	let ps2 = u2s(buf[i(-3)]);
	let ps1 = u2s(buf[i(-2)]);
	let ps0 = u2s(buf[i(-1)]);
	let qs0 = u2s(buf[i(0)]);
	let qs1 = u2s(buf[i(1)]);
	let qs2 = u2s(buf[i(2)]);
	let f = c8(c8(ps1 - qs1) + 3 * (qs0 - ps0));
	if hev {
		let f1 = c8(f + 4) >> 3;
		let f2 = c8(f + 3) >> 3;
		buf[i(0)] = s2u(qs0 - f1);
		buf[i(-1)] = s2u(ps0 + f2);
		return;
	}
	let u = c8((63 + f * 27) >> 7);
	buf[i(0)] = s2u(qs0 - u);
	buf[i(-1)] = s2u(ps0 + u);
	let u = c8((63 + f * 18) >> 7);
	buf[i(1)] = s2u(qs1 - u);
	buf[i(-2)] = s2u(ps1 + u);
	let u = c8((63 + f * 9) >> 7);
	buf[i(2)] = s2u(qs2 - u);
	buf[i(-3)] = s2u(ps2 + u);
}

/// Filters `count` positions of an edge. `at` is the first q0 pixel,
/// `step` the distance across the edge, `along` the distance between
/// positions. `mb` selects the macroblock-edge filter.
pub fn filter_edge(
	buf: &mut [u8],
	mut at: usize,
	step: usize,
	along: usize,
	count: usize,
	lim: &EdgeLimits,
	mb: bool,
) {
	let blimit = if mb { lim.mb_limit } else { lim.sub_limit };
	for _ in 0..count {
		if filter_mask(buf, at, step, lim.interior, blimit) {
			let h = hev(buf, at, step, lim.hev_thresh);
			if mb {
				mb_filter(buf, at, step, h);
			} else {
				inner_filter(buf, at, step, h);
			}
		}
		at += along;
	}
}

/// The simple filter (luma only, `vp8_loop_filter_simple_*`): 16
/// positions.
pub fn simple_edge(buf: &mut [u8], mut at: usize, step: usize, along: usize, blimit: u8) {
	for _ in 0..16 {
		let i = |k: isize| (at as isize + k * step as isize) as usize;
		let (p1, p0, q0, q1) = (buf[i(-2)], buf[i(-1)], buf[i(0)], buf[i(1)]);
		let d = (i32::from(p0) - i32::from(q0)).abs() * 2 + (i32::from(p1) - i32::from(q1)).abs() / 2;
		if d <= i32::from(blimit) {
			let (p1, p0, q0, q1) = (u2s(p1), u2s(p0), u2s(q0), u2s(q1));
			let f = c8(c8(p1 - q1) + 3 * (q0 - p0));
			let f1 = c8(f + 4) >> 3;
			buf[i(0)] = s2u(q0 - f1);
			let f2 = c8(f + 3) >> 3;
			buf[i(-1)] = s2u(p0 + f2);
		}
		at += along;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn transforms() {
		// A lone DC through the Walsh and the DCT is a flat block.
		let mut y2 = [0i16; 16];
		y2[0] = 800;
		let dcs = inv_walsh(&y2);
		assert!(dcs.iter().all(|&d| d == 100));
		assert_eq!(inv_walsh_dc(800), 100);
		let mut c = [0i16; 16];
		c[0] = 80;
		let mut px = [100u8; 16];
		idct_add(&c, &mut px, 0, 4);
		assert!(px.iter().all(|&p| p == 110));
		let mut px2 = [100u8; 16];
		dc_add(80, &mut px2, 0, 4);
		assert_eq!(px, px2);
	}

	#[test]
	fn filters_flat_inputs_unchanged() {
		let data = [77u8; 40 * 40];
		let plane = RefPlane { data: &data, stride: 40, width: 40, height: 40 };
		let mut out = [0u8; 256];
		for (fx, fy, bil) in [(2, 0, false), (0, 6, false), (4, 4, false), (3, 5, true)] {
			plane.predict(-30, 35, fx, fy, 16, 16, bil, &mut out, 16);
			assert!(out.iter().all(|&p| p == 77));
		}
	}
}
