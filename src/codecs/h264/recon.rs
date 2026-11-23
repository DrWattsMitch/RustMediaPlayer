//! Reconstruction: inverse transforms (§8.5.12-13), intra prediction
//! (§8.3), fractional-sample interpolation (§8.4.2.2) and weighted sample
//! prediction (§8.4.2.3). FFmpeg: `h264idct_template.c`,
//! `h264pred_template.c`, `h264qpel_template.c`, `h264chroma_template.c`,
//! `h264dsp_template.c`.
//!
//! Every routine is generic over the sample type ([`Pixel`]: `u8` for
//! 8-bit streams, `u16` for 9 to 14 bits) and works on strided views
//! ([`Src`], [`Dst`]) of a [`Plane`]: the whole frame, or one field of it
//! (every other row), which is how field pictures and field macroblocks
//! address an interleaved frame. The 8-bit frame case dispatches to the
//! vector code in `simd.rs`; that dispatch compiles away for `u16`.

use std::fmt::Debug;

use crate::core::{Pooled, VecPool};

/// A sample type: `u8` (8-bit video) or `u16` (9 to 14 bits).
pub trait Pixel: Copy + Default + Send + Sync + PartialEq + Debug + Into<i32> + 'static {
	/// `true` for `u16`.
	const WIDE: bool;
	/// `v` clipped to `0..=max` (`max` is ignored for `u8`: always 255).
	fn clip(v: i32, max: i32) -> Self;
	/// `v`, known to be in range.
	fn of(v: i32) -> Self;
	/// The same view as 8-bit samples, if this is the 8-bit type (lets
	/// generic code use the 8-bit vector routines).
	fn narrow_dst<'b>(d: &'b mut Dst<'_, Self>) -> Option<Dst<'b, u8>>;
	fn narrow_src<'b>(s: &Src<'b, Self>) -> Option<Src<'b, u8>>;
	fn narrow(s: &[Self]) -> Option<&[u8]>;
	fn narrow_mut(s: &mut [Self]) -> Option<&mut [u8]>;
}

impl Pixel for u8 {
	const WIDE: bool = false;
	#[inline(always)]
	fn clip(v: i32, _max: i32) -> u8 {
		v.clamp(0, 255) as u8
	}
	#[inline(always)]
	fn of(v: i32) -> u8 {
		v as u8
	}
	#[inline(always)]
	fn narrow_dst<'b>(d: &'b mut Dst<'_, u8>) -> Option<Dst<'b, u8>> {
		Some(d.reborrow())
	}
	#[inline(always)]
	fn narrow_src<'b>(s: &Src<'b, u8>) -> Option<Src<'b, u8>> {
		Some(*s)
	}
	#[inline(always)]
	fn narrow(s: &[u8]) -> Option<&[u8]> {
		Some(s)
	}
	#[inline(always)]
	fn narrow_mut(s: &mut [u8]) -> Option<&mut [u8]> {
		Some(s)
	}
}

impl Pixel for u16 {
	const WIDE: bool = true;
	#[inline(always)]
	fn clip(v: i32, max: i32) -> u16 {
		v.clamp(0, max) as u16
	}
	#[inline(always)]
	fn of(v: i32) -> u16 {
		v as u16
	}
	#[inline(always)]
	fn narrow_dst<'b>(_: &'b mut Dst<'_, u16>) -> Option<Dst<'b, u8>> {
		None
	}
	#[inline(always)]
	fn narrow_src<'b>(_: &Src<'b, u16>) -> Option<Src<'b, u8>> {
		None
	}
	#[inline(always)]
	fn narrow(_: &[u16]) -> Option<&[u8]> {
		None
	}
	#[inline(always)]
	fn narrow_mut(_: &mut [u16]) -> Option<&mut [u8]> {
		None
	}
}

/// One sample plane. The samples may come from a [`VecPool`], to which
/// they return when the plane is dropped.
#[derive(Debug, Clone, Default)]
pub struct Plane<T = u8> {
	pub data: Pooled<T>,
	pub width: usize,
	pub height: usize,
}

impl<T: Pixel> Plane<T> {
	pub fn new(width: usize, height: usize, fill: T) -> Self {
		Self { data: Pooled::detached(vec![fill; width * height]), width, height }
	}

	/// Like [`Self::new`], with the samples taken from `pool`.
	pub fn pooled(pool: &VecPool<T>, width: usize, height: usize, fill: T) -> Self {
		Self { data: pool.take_filled(width * height, fill), width, height }
	}

	/// A plane from `pool` whose samples are left over from its previous
	/// use: only for planes every sample of which is written before it is
	/// read.
	pub fn recycled(pool: &VecPool<T>, width: usize, height: usize) -> Self {
		Self { data: pool.take_recycled(width * height), width, height }
	}

	#[inline]
	pub fn at(&self, x: usize, y: usize) -> T {
		self.data[y * self.width + x]
	}

	/// A read view of the frame (`field` = `None`) or of one field
	/// (`Some(0)` top, `Some(1)` bottom). `max` is the largest sample value.
	#[inline]
	pub fn src(&self, field: Option<usize>, max: i32) -> Src<'_, T> {
		let (base, stride, height) = match field {
			None => (0, self.width, self.height),
			Some(p) => (p * self.width, 2 * self.width, self.height / 2),
		};
		Src { data: &self.data, base, stride, width: self.width, height, max }
	}

	/// A write view, see [`Self::src`].
	#[inline]
	pub fn dst(&mut self, field: Option<usize>, max: i32) -> Dst<'_, T> {
		let (base, stride, height) = match field {
			None => (0, self.width, self.height),
			Some(p) => (p * self.width, 2 * self.width, self.height / 2),
		};
		Dst { data: &mut self.data, base, stride, width: self.width, height, max }
	}
}

impl Plane<u8> {
	/// The whole 8-bit frame as a write view.
	#[inline]
	pub fn d8(&mut self) -> Dst<'_, u8> {
		self.dst(None, 255)
	}

	/// The whole 8-bit frame as a read view.
	#[inline]
	pub fn s8(&self) -> Src<'_, u8> {
		self.src(None, 255)
	}
}

/// A read-only, strided view of a plane: sample (x, y) is
/// `data[base + y * stride + x]`, for `x < width` and `y < height`.
#[derive(Debug, Clone, Copy)]
pub struct Src<'a, T> {
	pub data: &'a [T],
	pub base: usize,
	pub stride: usize,
	pub width: usize,
	pub height: usize,
	/// Largest sample value (`(1 << bit_depth) - 1`).
	pub max: i32,
}

impl<T: Pixel> Src<'_, T> {
	#[inline(always)]
	pub fn idx(&self, x: usize, y: usize) -> usize {
		self.base + y * self.stride + x
	}

	/// Sample with coordinates clamped into the view (reference picture
	/// edge extension, 8-228/8-229).
	#[inline]
	pub fn clamped(&self, x: i32, y: i32) -> T {
		let x = x.clamp(0, self.width as i32 - 1) as usize;
		let y = y.clamp(0, self.height as i32 - 1) as usize;
		self.data[self.idx(x, y)]
	}
}

/// A writable, strided view of a plane, see [`Src`].
#[derive(Debug)]
pub struct Dst<'a, T> {
	pub data: &'a mut [T],
	pub base: usize,
	pub stride: usize,
	pub width: usize,
	pub height: usize,
	pub max: i32,
}

impl<T: Pixel> Dst<'_, T> {
	#[inline(always)]
	pub fn idx(&self, x: usize, y: usize) -> usize {
		self.base + y * self.stride + x
	}

	#[inline(always)]
	pub fn get(&self, x: usize, y: usize) -> T {
		self.data[self.idx(x, y)]
	}

	#[inline(always)]
	pub fn set(&mut self, x: usize, y: usize, v: T) {
		let i = self.idx(x, y);
		self.data[i] = v;
	}

	/// The same view, borrowed for a shorter time.
	#[inline(always)]
	pub fn reborrow(&mut self) -> Dst<'_, T> {
		Dst {
			data: &mut *self.data,
			base: self.base,
			stride: self.stride,
			width: self.width,
			height: self.height,
			max: self.max,
		}
	}

	/// Row `y` from column `x`, `n` samples long.
	#[inline(always)]
	pub fn row_mut(&mut self, x: usize, y: usize, n: usize) -> &mut [T] {
		let i = self.idx(x, y);
		&mut self.data[i..i + n]
	}
}

/// Mid-grey of a bit depth whose largest value is `max`.
#[inline(always)]
pub fn mid(max: i32) -> i32 {
	(max + 1) >> 1
}

// ---------------------------------------------------------------------------
// Inverse transforms

/// Adds a 4x4 inverse-transformed residual (8.5.12.2) at (x, y).
#[inline]
pub fn idct4_add<T: Pixel>(p: &mut Dst<T>, x: usize, y: usize, c: &[i32; 16]) {
	if crate::simd::enabled() {
		if let Some(mut d) = T::narrow_dst(p) {
			return super::simd::idct4_add(&mut d, x, y, c);
		}
	}
	idct4_add_scalar(p, x, y, c)
}

/// Scalar version of [`idct4_add`]: the fallback and the reference.
pub(super) fn idct4_add_scalar<T: Pixel>(p: &mut Dst<T>, x: usize, y: usize, c: &[i32; 16]) {
	let mut t = [0i32; 16];
	for i in 0..4 {
		let r = &c[i * 4..i * 4 + 4];
		let e0 = r[0] + r[2];
		let e1 = r[0] - r[2];
		let e2 = (r[1] >> 1) - r[3];
		let e3 = r[1] + (r[3] >> 1);
		t[i * 4] = e0 + e3;
		t[i * 4 + 1] = e1 + e2;
		t[i * 4 + 2] = e1 - e2;
		t[i * 4 + 3] = e0 - e3;
	}
	let max = p.max;
	for j in 0..4 {
		let e0 = t[j] + t[8 + j];
		let e1 = t[j] - t[8 + j];
		let e2 = (t[4 + j] >> 1) - t[12 + j];
		let e3 = t[4 + j] + (t[12 + j] >> 1);
		let col = [e0 + e3, e1 + e2, e1 - e2, e0 - e3];
		for (i, &v) in col.iter().enumerate() {
			let at = p.idx(x + j, y + i);
			p.data[at] = T::clip(p.data[at].into() + ((v + 32) >> 6), max);
		}
	}
}

fn idct8_1d(s: [i32; 8]) -> [i32; 8] {
	let a0 = s[0] + s[4];
	let a4 = s[0] - s[4];
	let a2 = (s[2] >> 1) - s[6];
	let a6 = s[2] + (s[6] >> 1);
	let b0 = a0 + a6;
	let b2 = a4 + a2;
	let b4 = a4 - a2;
	let b6 = a0 - a6;
	let a1 = -s[3] + s[5] - s[7] - (s[7] >> 1);
	let a3 = s[1] + s[7] - s[3] - (s[3] >> 1);
	let a5 = -s[1] + s[7] + s[5] + (s[5] >> 1);
	let a7 = s[3] + s[5] + s[1] + (s[1] >> 1);
	let b1 = a1 + (a7 >> 2);
	let b7 = a7 - (a1 >> 2);
	let b3 = a3 + (a5 >> 2);
	let b5 = (a3 >> 2) - a5;
	[b0 + b7, b2 + b5, b4 + b3, b6 + b1, b6 - b1, b4 - b3, b2 - b5, b0 - b7]
}

/// Adds an 8x8 inverse-transformed residual (8.5.13.2) at (x, y).
#[inline]
pub fn idct8_add<T: Pixel>(p: &mut Dst<T>, x: usize, y: usize, c: &[i32; 64]) {
	if crate::simd::enabled() {
		if let Some(mut d) = T::narrow_dst(p) {
			return super::simd::idct8_add(&mut d, x, y, c);
		}
	}
	idct8_add_scalar(p, x, y, c)
}

/// Scalar version of [`idct8_add`]: the fallback and the reference.
pub(super) fn idct8_add_scalar<T: Pixel>(p: &mut Dst<T>, x: usize, y: usize, c: &[i32; 64]) {
	let mut t = [0i32; 64];
	for i in 0..8 {
		let mut row = [0i32; 8];
		row.copy_from_slice(&c[i * 8..i * 8 + 8]);
		t[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d(row));
	}
	let max = p.max;
	for j in 0..8 {
		let col = idct8_1d(std::array::from_fn(|i| t[i * 8 + j]));
		for (i, &v) in col.iter().enumerate() {
			let at = p.idx(x + j, y + i);
			p.data[at] = T::clip(p.data[at].into() + ((v + 32) >> 6), max);
		}
	}
}

/// Transform inputs of a conforming stream fit in `8 + bit_depth` bits
/// (8.5.12.1); clamping keeps damaged streams from overflowing the integer
/// transforms.
#[inline]
fn clamp_coeff(v: i64, bit_depth: u32) -> i32 {
	let lim = 1i64 << (7 + bit_depth);
	v.clamp(-lim, lim - 1) as i32
}

/// Inverse 4x4 Hadamard of Intra16x16 luma DC coefficients (8.5.10),
/// followed by dequantisation with `qp` = QP'Y. `c` is in raster order of
/// the 4x4 DC array.
pub fn luma_dc_dequant(c: &[i32; 16], qp: i32, scale: i32, bit_depth: u32) -> [i32; 16] {
	let mut t = [0i32; 16];
	for i in 0..4 {
		let r = &c[i * 4..i * 4 + 4];
		let (a, b) = (r[0] + r[1], r[0] - r[1]);
		let (cc, d) = (r[2] + r[3], r[2] - r[3]);
		t[i * 4] = a + cc;
		t[i * 4 + 1] = a - cc;
		t[i * 4 + 2] = b - d;
		t[i * 4 + 3] = b + d;
	}
	let mut f = [0i32; 16];
	for j in 0..4 {
		let (a, b) = (t[j] + t[4 + j], t[j] - t[4 + j]);
		let (cc, d) = (t[8 + j] + t[12 + j], t[8 + j] - t[12 + j]);
		f[j] = a + cc;
		f[4 + j] = a - cc;
		f[8 + j] = b - d;
		f[12 + j] = b + d;
	}
	let shift = i64::from(qp / 6);
	for v in &mut f {
		let x = i64::from(*v) * i64::from(scale);
		*v = clamp_coeff(
			if shift >= 6 { x << (shift - 6) } else { (x + (1 << (5 - shift))) >> (6 - shift) },
			bit_depth,
		);
	}
	f
}

/// Inverse 2x2 transform and dequantisation of 4:2:0 chroma DC (8.5.11),
/// `qp` = QP'C.
pub fn chroma_dc_dequant(c: &[i32], qp: i32, scale: i32, bit_depth: u32) -> [i32; 4] {
	let f = [
		c[0] + c[1] + c[2] + c[3],
		c[0] - c[1] + c[2] - c[3],
		c[0] + c[1] - c[2] - c[3],
		c[0] - c[1] - c[2] + c[3],
	];
	f.map(|v| clamp_coeff(((i64::from(v) * i64::from(scale)) << (qp / 6)) >> 5, bit_depth))
}

/// Inverse 2x4 transform and dequantisation of 4:2:2 chroma DC (8.5.11.1,
/// 8.5.11.2). `c` is the 2-wide, 4-tall DC array in raster order; `qp` is
/// QP'C + 3 and `scale` LevelScale4x4(qp % 6, 0, 0).
pub fn chroma422_dc_dequant(c: &[i32], qp: i32, scale: i32, bit_depth: u32) -> [i32; 8] {
	// Columns: 4-point Hadamard (A4 * c); rows: 2-point (* A2).
	let mut t = [0i32; 8];
	for col in 0..2 {
		let v = [c[col], c[2 + col], c[4 + col], c[6 + col]];
		let (a, b) = (v[0] + v[1], v[0] - v[1]);
		let (cc, d) = (v[2] + v[3], v[2] - v[3]);
		t[col] = a + cc;
		t[2 + col] = a - cc;
		t[4 + col] = b - d;
		t[6 + col] = b + d;
	}
	let mut f = [0i32; 8];
	for row in 0..4 {
		f[row * 2] = t[row * 2] + t[row * 2 + 1];
		f[row * 2 + 1] = t[row * 2] - t[row * 2 + 1];
	}
	let shift = i64::from(qp / 6);
	f.map(|v| {
		let x = i64::from(v) * i64::from(scale);
		clamp_coeff(
			if shift >= 6 { x << (shift - 6) } else { (x + (1 << (5 - shift))) >> (6 - shift) },
			bit_depth,
		)
	})
}

/// Dequantises a 4x4 block in place (8.5.12.1), `qp` = QP'. `skip_dc`
/// leaves coefficient 0 untouched (already-dequantised DC).
pub fn dequant4(c: &mut [i32; 16], qp: i32, scale: &[i32; 16], skip_dc: bool, bit_depth: u32) {
	let shift = i64::from(qp / 6);
	for (i, v) in c.iter_mut().enumerate() {
		if (skip_dc && i == 0) || *v == 0 {
			continue;
		}
		let x = i64::from(*v) * i64::from(scale[i]);
		*v = clamp_coeff(
			if shift >= 4 { x << (shift - 4) } else { (x + (1 << (3 - shift))) >> (4 - shift) },
			bit_depth,
		);
	}
}

/// Dequantises an 8x8 block in place (8.5.13.1), `qp` = QP'Y.
pub fn dequant8(c: &mut [i32; 64], qp: i32, scale: &[i32; 64], bit_depth: u32) {
	let shift = i64::from(qp / 6);
	for (i, v) in c.iter_mut().enumerate() {
		if *v == 0 {
			continue;
		}
		let x = i64::from(*v) * i64::from(scale[i]);
		*v = clamp_coeff(
			if shift >= 6 { x << (shift - 6) } else { (x + (1 << (5 - shift))) >> (6 - shift) },
			bit_depth,
		);
	}
}

// ---------------------------------------------------------------------------
// Intra prediction

/// Which neighbouring samples are available for an intra block.
#[derive(Debug, Clone, Copy, Default)]
pub struct Avail {
	pub left: bool,
	pub top: bool,
	pub top_right: bool,
	pub top_left: bool,
}

/// Intra_4x4 prediction (8.3.1.2) written into the view at (x, y).
pub fn pred4x4<T: Pixel>(p: &mut Dst<T>, x: usize, y: usize, mode: u8, a: Avail) {
	let dflt = mid(p.max);
	let mut top = [dflt; 8];
	let mut left = [dflt; 4];
	let mut q = dflt;
	if a.top {
		for i in 0..4 {
			top[i] = p.get(x + i, y - 1).into();
		}
		for i in 4..8 {
			top[i] = if a.top_right { p.get(x + i, y - 1).into() } else { top[3] };
		}
	}
	if a.left {
		for (i, l) in left.iter_mut().enumerate() {
			*l = p.get(x - 1, y + i).into();
		}
	}
	if a.top_left {
		q = p.get(x - 1, y - 1).into();
	}
	let t = |i: usize| top[i];
	let l = |i: usize| left[i];
	let mut out = [[0i32; 4]; 4];
	match mode {
		0 => {
			for row in &mut out {
				row.copy_from_slice(&top[..4]);
			}
		}
		1 => {
			for (r, row) in out.iter_mut().enumerate() {
				*row = [left[r]; 4];
			}
		}
		2 => {
			let dc = match (a.top, a.left) {
				(true, true) => (t(0) + t(1) + t(2) + t(3) + l(0) + l(1) + l(2) + l(3) + 4) >> 3,
				(true, false) => (t(0) + t(1) + t(2) + t(3) + 2) >> 2,
				(false, true) => (l(0) + l(1) + l(2) + l(3) + 2) >> 2,
				_ => dflt,
			};
			out = [[dc; 4]; 4];
		}
		3 => {
			for r in 0..4 {
				for c in 0..4 {
					out[r][c] = if r == 3 && c == 3 {
						(t(6) + 3 * t(7) + 2) >> 2
					} else {
						(t(r + c) + 2 * t(r + c + 1) + t(r + c + 2) + 2) >> 2
					};
				}
			}
		}
		4..=6 => {
			// e[4] = p[-1,-1]; p[k,-1] = e[5 + k]; p[-1,k] = e[3 - k].
			let e = [l(3), l(2), l(1), l(0), q, t(0), t(1), t(2), t(3)];
			let f3 = |i: i32| (e[(i - 1) as usize] + 2 * e[i as usize] + e[(i + 1) as usize] + 2) >> 2;
			let f2 = |i: i32| (e[i as usize] + e[(i + 1) as usize] + 1) >> 1;
			for r in 0..4i32 {
				for c in 0..4i32 {
					out[r as usize][c as usize] = match mode {
						// Diagonal_Down_Right (8.3.1.2.5).
						4 => f3(4 + c - r),
						// Vertical_Right (8.3.1.2.6), zVR = 2x - y.
						5 => match 2 * c - r {
							z if z >= 0 && z % 2 == 0 => f2(4 + c - (r >> 1)),
							z if z >= 0 => f3(4 + c - (r >> 1)),
							-1 => f3(4),
							_ => f3(5 - r),
						},
						// Horizontal_Down (8.3.1.2.7), zHD = 2y - x.
						_ => match 2 * r - c {
							z if z >= 0 && z % 2 == 0 => f2(3 - r + (c >> 1)),
							z if z >= 0 => f3(4 - r + (c >> 1)),
							-1 => f3(4),
							_ => f3(3 + c),
						},
					};
				}
			}
		}
		7 => {
			for r in 0..4 {
				for c in 0..4 {
					let i = c + (r >> 1);
					out[r][c] = if r % 2 == 0 {
						(t(i) + t(i + 1) + 1) >> 1
					} else {
						(t(i) + 2 * t(i + 1) + t(i + 2) + 2) >> 2
					};
				}
			}
		}
		_ => {
			for r in 0..4 {
				for c in 0..4 {
					let z = c + 2 * r;
					out[r][c] = match z {
						0 | 2 | 4 => (l(z / 2) + l(z / 2 + 1) + 1) >> 1,
						1 | 3 => (l(z / 2) + 2 * l(z / 2 + 1) + l(z / 2 + 2) + 2) >> 2,
						5 => (l(2) + 3 * l(3) + 2) >> 2,
						_ => l(3),
					};
				}
			}
		}
	}
	for (r, row) in out.iter().enumerate() {
		for (c, &v) in row.iter().enumerate() {
			p.set(x + c, y + r, T::of(v));
		}
	}
}

/// Intra_8x8 prediction with reference sample filtering (8.3.2.2).
pub fn pred8x8l<T: Pixel>(p: &mut Dst<T>, x: usize, y: usize, mode: u8, a: Avail) {
	let dflt = mid(p.max);
	// Unfiltered references.
	let mut top = [0i32; 16];
	let mut left = [0i32; 8];
	let mut q = 0i32;
	if a.top {
		for i in 0..8 {
			top[i] = p.get(x + i, y - 1).into();
		}
		for i in 8..16 {
			top[i] = if a.top_right { p.get(x + i, y - 1).into() } else { top[7] };
		}
	}
	if a.left {
		for (i, l) in left.iter_mut().enumerate() {
			*l = p.get(x - 1, y + i).into();
		}
	}
	if a.top_left {
		q = p.get(x - 1, y - 1).into();
	}
	// Filtered references (8-78 .. 8-92).
	let mut ft = [0i32; 16];
	let mut fl = [0i32; 8];
	let mut fq = q;
	if a.top {
		ft[0] =
			if a.top_left { (q + 2 * top[0] + top[1] + 2) >> 2 } else { (3 * top[0] + top[1] + 2) >> 2 };
		for i in 1..15 {
			ft[i] = (top[i - 1] + 2 * top[i] + top[i + 1] + 2) >> 2;
		}
		ft[15] = (top[14] + 3 * top[15] + 2) >> 2;
	}
	if a.top_left {
		fq = match (a.top, a.left) {
			(true, true) => (top[0] + 2 * q + left[0] + 2) >> 2,
			(true, false) => (3 * q + top[0] + 2) >> 2,
			(false, true) => (3 * q + left[0] + 2) >> 2,
			_ => q,
		};
	}
	if a.left {
		fl[0] = if a.top_left {
			(q + 2 * left[0] + left[1] + 2) >> 2
		} else {
			(3 * left[0] + left[1] + 2) >> 2
		};
		for i in 1..7 {
			fl[i] = (left[i - 1] + 2 * left[i] + left[i + 1] + 2) >> 2;
		}
		fl[7] = (left[6] + 3 * left[7] + 2) >> 2;
	}
	let (t, l) = (&ft, &fl);
	let mut out = [[0i32; 8]; 8];
	match mode {
		0 => {
			for row in &mut out {
				row.copy_from_slice(&t[..8]);
			}
		}
		1 => {
			for (r, row) in out.iter_mut().enumerate() {
				*row = [l[r]; 8];
			}
		}
		2 => {
			let st: i32 = t[..8].iter().sum();
			let sl: i32 = l.iter().sum();
			let dc = match (a.top, a.left) {
				(true, true) => (st + sl + 8) >> 4,
				(true, false) => (st + 4) >> 3,
				(false, true) => (sl + 4) >> 3,
				_ => dflt,
			};
			out = [[dc; 8]; 8];
		}
		3 => {
			for r in 0..8 {
				for c in 0..8 {
					out[r][c] = if r == 7 && c == 7 {
						(t[14] + 3 * t[15] + 2) >> 2
					} else {
						(t[r + c] + 2 * t[r + c + 1] + t[r + c + 2] + 2) >> 2
					};
				}
			}
		}
		_ => {
			// e[8] = p'[-1,-1]; p'[k,-1] = e[9 + k]; p'[-1,k] = e[7 - k].
			let mut e = [0i32; 17];
			for i in 0..8 {
				e[i] = l[7 - i];
				e[9 + i] = t[i];
			}
			e[8] = fq;
			let f3 = |i: i32| (e[(i - 1) as usize] + 2 * e[i as usize] + e[(i + 1) as usize] + 2) >> 2;
			let f2 = |i: i32| (e[i as usize] + e[(i + 1) as usize] + 1) >> 1;
			for r in 0..8i32 {
				for c in 0..8i32 {
					out[r as usize][c as usize] = match mode {
						// Diagonal_Down_Right (8.3.2.2.6).
						4 => f3(8 + c - r),
						// Vertical_Right (8.3.2.2.7), zVR = 2x - y.
						5 => match 2 * c - r {
							z if z >= 0 && z & 1 == 0 => f2(8 + c - (r >> 1)),
							z if z >= 0 => f3(8 + c - (r >> 1)),
							-1 => f3(8),
							_ => f3(9 - r + 2 * c),
						},
						// Horizontal_Down (8.3.2.2.8), zHD = 2y - x.
						6 => match 2 * r - c {
							z if z >= 0 && z & 1 == 0 => f2(7 - r + (c >> 1)),
							z if z >= 0 => f3(8 - r + (c >> 1)),
							-1 => f3(8),
							_ => f3(7 + c - 2 * r),
						},
						7 => {
							let i = (c + (r >> 1)) as usize;
							if r % 2 == 0 {
								(t[i] + t[i + 1] + 1) >> 1
							} else {
								(t[i] + 2 * t[i + 1] + t[i + 2] + 2) >> 2
							}
						}
						_ => {
							let z = c + 2 * r;
							if z < 13 && z & 1 == 0 {
								(l[(r + (c >> 1)) as usize] + l[(r + (c >> 1) + 1) as usize] + 1) >> 1
							} else if z < 13 {
								let i = (r + (c >> 1)) as usize;
								(l[i] + 2 * l[i + 1] + l[i + 2] + 2) >> 2
							} else if z == 13 {
								(l[6] + 3 * l[7] + 2) >> 2
							} else {
								l[7]
							}
						}
					};
				}
			}
		}
	}
	for (r, row) in out.iter().enumerate() {
		for (c, &v) in row.iter().enumerate() {
			p.set(x + c, y + r, T::of(v));
		}
	}
}

/// Intra_16x16 (16x16) and chroma (8x8 for 4:2:0, 8x16 for 4:2:2)
/// whole-block prediction: mode 0 vertical, 1 horizontal, 2 DC, 3 plane
/// (8.3.3, 8.3.4). For chroma, DC is computed per 4x4 sub-block
/// (8.3.4.1-3).
#[allow(clippy::too_many_arguments)]
pub fn pred_block<T: Pixel>(
	p: &mut Dst<T>,
	x: usize,
	y: usize,
	w: usize,
	h: usize,
	mode: u8,
	a: Avail,
	chroma: bool,
) {
	let max = p.max;
	let dflt = mid(max);
	let mut tp = [0i32; 16];
	let mut lf = [0i32; 16];
	if a.top || mode == 3 {
		for (i, v) in tp[..w].iter_mut().enumerate() {
			*v = p.get(x + i, y - 1).into();
		}
	}
	if a.left || mode == 3 {
		for (i, v) in lf[..h].iter_mut().enumerate() {
			*v = p.get(x - 1, y + i).into();
		}
	}
	let top = |i: usize| tp[i];
	let left = |i: usize| lf[i];
	// At most 16x16: a stack buffer, not a heap allocation per block.
	let mut buf = [0i32; 256];
	let out = &mut buf[..w * h];
	match mode {
		0 => {
			for r in 0..h {
				out[r * w..r * w + w].copy_from_slice(&tp[..w]);
			}
		}
		1 => {
			for r in 0..h {
				out[r * w..r * w + w].fill(left(r));
			}
		}
		2 if !chroma => {
			let st: i32 = if a.top { tp[..16].iter().sum() } else { 0 };
			let sl: i32 = if a.left { lf[..16].iter().sum() } else { 0 };
			let dc = match (a.top, a.left) {
				(true, true) => (st + sl + 16) >> 5,
				(true, false) => (st + 8) >> 4,
				(false, true) => (sl + 8) >> 4,
				_ => dflt,
			};
			out.fill(dc);
		}
		2 => {
			for by in 0..h / 4 {
				for bx in 0..w / 4 {
					let st: i32 = if a.top { (0..4).map(|i| top(bx * 4 + i)).sum() } else { 0 };
					let sl: i32 = if a.left { (0..4).map(|i| left(by * 4 + i)).sum() } else { 0 };
					// Corner and interior blocks prefer both; edge blocks
					// prefer the adjacent side (8.3.4.1-3).
					let prefer_top = by == 0 && bx != 0;
					let prefer_left = bx == 0 && by != 0;
					let dc = if prefer_top {
						if a.top {
							(st + 2) >> 2
						} else if a.left {
							(sl + 2) >> 2
						} else {
							dflt
						}
					} else if prefer_left {
						if a.left {
							(sl + 2) >> 2
						} else if a.top {
							(st + 2) >> 2
						} else {
							dflt
						}
					} else {
						match (a.top, a.left) {
							(true, true) => (st + sl + 4) >> 3,
							(true, false) => (st + 2) >> 2,
							(false, true) => (sl + 2) >> 2,
							_ => dflt,
						}
					};
					for r in 0..4 {
						out[(by * 4 + r) * w + bx * 4..(by * 4 + r) * w + bx * 4 + 4].fill(dc);
					}
				}
			}
		}
		_ => {
			// Plane (8.3.3.4, 8.3.4.4): xCF / yCF extend the 8-sample chroma
			// sums to 16 for a 16-sample side.
			let q: i32 = p.get(x - 1, y - 1).into();
			let tq = |i: i32| if i < 0 { q } else { top(i as usize) };
			let lq = |i: i32| if i < 0 { q } else { left(i as usize) };
			let (hw, hh) = ((w / 2) as i32, (h / 2) as i32);
			let mut sh = 0;
			let mut sv = 0;
			for i in 0..hw {
				sh += (i + 1) * (tq(hw + i) - tq(hw - 2 - i));
			}
			for i in 0..hh {
				sv += (i + 1) * (lq(hh + i) - lq(hh - 2 - i));
			}
			let a_ = 16 * (left(h - 1) + top(w - 1));
			let k = |n: usize| if n == 16 { 5 } else { 34 };
			let b = (k(w) * sh + 32) >> 6;
			let c = (k(h) * sv + 32) >> 6;
			for r in 0..h as i32 {
				for cc in 0..w as i32 {
					out[(r * w as i32 + cc) as usize] = (a_ + b * (cc - hw + 1) + c * (r - hh + 1) + 16) >> 5;
				}
			}
		}
	}
	for r in 0..h {
		let row = p.row_mut(x, y + r, w);
		for (d, &v) in row.iter_mut().zip(&out[r * w..r * w + w]) {
			*d = T::clip(v, max);
		}
	}
}

// ---------------------------------------------------------------------------
// Inter prediction

/// Window width for luma interpolation: a 16-sample block plus 5 taps.
const WIN: usize = 16 + 5;

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
	a + f - 5 * (b + e) + 20 * (c + d)
}

#[inline(always)]
fn avg<T: Pixel>(a: T, b: T) -> T {
	T::of((a.into() + b.into() + 1) >> 1)
}

/// Copies the (w x h) area at (x, y) of `r` into `dst` (row stride
/// `stride`), clamping coordinates at the view's edges.
pub(super) fn fetch<T: Pixel>(
	r: &Src<T>,
	x: i32,
	y: i32,
	w: usize,
	h: usize,
	dst: &mut [T],
	stride: usize,
) {
	let inside =
		x >= 0 && y >= 0 && x + w as i32 <= r.width as i32 && y + h as i32 <= r.height as i32;
	for j in 0..h {
		let row = &mut dst[j * stride..j * stride + w];
		if inside {
			let start = r.idx(x as usize, y as usize + j);
			row.copy_from_slice(&r.data[start..start + w]);
		} else {
			let sy = (y + j as i32).clamp(0, r.height as i32 - 1) as usize;
			let base = r.idx(0, sy);
			for (i, v) in row.iter_mut().enumerate() {
				*v = r.data[base + (x + i as i32).clamp(0, r.width as i32 - 1) as usize];
			}
		}
	}
}

/// Luma sample interpolation (8.4.2.2.1) for a `bw` x `bh` block whose
/// top-left is at integer position (x, y) with quarter-sample fraction
/// (fx, fy). Output rows have stride `bw`.
///
/// Separable passes over a fetched window: half-sample rows `b`/`s`,
/// columns `h`/`m` and the centre `j` are computed only when the fraction
/// needs them, then averaged for quarter positions.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn mc_luma<T: Pixel>(
	r: &Src<T>,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [T],
) {
	if crate::simd::enabled() && matches!(bw, 4 | 8 | 16) {
		if let (Some(r8), Some(o8)) = (T::narrow_src(r), T::narrow_mut(out)) {
			return super::simd::mc_luma(&r8, x, y, fx, fy, bw, bh, o8);
		}
	}
	mc_luma_scalar(r, x, y, fx, fy, bw, bh, out)
}

/// Scalar version of [`mc_luma`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn mc_luma_scalar<T: Pixel>(
	r: &Src<T>,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [T],
) {
	let max = r.max;
	let mut win = [T::default(); WIN * WIN];
	fetch(r, x - 2, y - 2, bw + 5, bh + 5, &mut win, WIN);
	let g = |i: usize, j: usize| win[(j + 2) * WIN + i + 2];
	if fx == 0 && fy == 0 {
		for j in 0..bh {
			out[j * bw..j * bw + bw].copy_from_slice(&win[(j + 2) * WIN + 2..(j + 2) * WIN + 2 + bw]);
		}
		return;
	}
	let need_b = matches!((fx, fy), (1..=3, 0 | 1));
	let need_s = matches!((fx, fy), (1..=3, 3));
	let need_h = matches!((fx, fy), (0, 1..=3) | (1, 1..=3));
	let need_m = matches!((fx, fy), (3, 1..=3));
	let need_j = matches!((fx, fy), (2, 1..=3) | (1 | 3, 2));
	let px = |v: T| -> i32 { v.into() };

	// Horizontal half samples for rows 0..=bh (row bh is `s` of the last row).
	let mut hb = [T::default(); 16 * 17];
	if need_b || need_s {
		for j in 0..=bh {
			let row = &win[(j + 2) * WIN..(j + 2) * WIN + WIN];
			for i in 0..bw {
				let v = tap6(
					px(row[i]),
					px(row[i + 1]),
					px(row[i + 2]),
					px(row[i + 3]),
					px(row[i + 4]),
					px(row[i + 5]),
				);
				hb[j * 16 + i] = T::clip((v + 16) >> 5, max);
			}
		}
	}
	// Vertical half samples for columns 0..=bw (column bw is `m`).
	let mut vb = [T::default(); 17 * 16];
	if need_h || need_m {
		for j in 0..bh {
			for i in 0..=bw {
				let c = i + 2;
				let v = tap6(
					px(win[j * WIN + c]),
					px(win[(j + 1) * WIN + c]),
					px(win[(j + 2) * WIN + c]),
					px(win[(j + 3) * WIN + c]),
					px(win[(j + 4) * WIN + c]),
					px(win[(j + 5) * WIN + c]),
				);
				vb[j * 17 + i] = T::clip((v + 16) >> 5, max);
			}
		}
	}
	// Centre samples from unrounded horizontal intermediates.
	let mut jb = [T::default(); 256];
	if need_j {
		let mut tmp = [0i32; WIN * 16];
		for rr in 0..bh + 5 {
			let row = &win[rr * WIN..rr * WIN + WIN];
			for i in 0..bw {
				tmp[rr * 16 + i] = tap6(
					px(row[i]),
					px(row[i + 1]),
					px(row[i + 2]),
					px(row[i + 3]),
					px(row[i + 4]),
					px(row[i + 5]),
				);
			}
		}
		for j in 0..bh {
			for i in 0..bw {
				let t = |k: usize| tmp[(j + k) * 16 + i];
				let v = t(0) + t(5) - 5 * (t(1) + t(4)) + 20 * (t(2) + t(3));
				jb[j * 16 + i] = T::clip((v + 512) >> 10, max);
			}
		}
	}
	let b = |i: usize, j: usize| hb[j * 16 + i];
	let s = |i: usize, j: usize| hb[(j + 1) * 16 + i];
	let h = |i: usize, j: usize| vb[j * 17 + i];
	let m = |i: usize, j: usize| vb[j * 17 + i + 1];
	let c = |i: usize, j: usize| jb[j * 16 + i];
	// One monomorphised loop per fractional position.
	match (fx, fy) {
		(1, 0) => fill_block(out, bw, bh, |i, j| avg(g(i, j), b(i, j))),
		(2, 0) => fill_block(out, bw, bh, b),
		(3, 0) => fill_block(out, bw, bh, |i, j| avg(b(i, j), g(i + 1, j))),
		(0, 1) => fill_block(out, bw, bh, |i, j| avg(g(i, j), h(i, j))),
		(0, 2) => fill_block(out, bw, bh, h),
		(0, 3) => fill_block(out, bw, bh, |i, j| avg(h(i, j), g(i, j + 1))),
		(2, 2) => fill_block(out, bw, bh, c),
		(2, 1) => fill_block(out, bw, bh, |i, j| avg(b(i, j), c(i, j))),
		(2, 3) => fill_block(out, bw, bh, |i, j| avg(c(i, j), s(i, j))),
		(1, 2) => fill_block(out, bw, bh, |i, j| avg(h(i, j), c(i, j))),
		(3, 2) => fill_block(out, bw, bh, |i, j| avg(c(i, j), m(i, j))),
		(1, 1) => fill_block(out, bw, bh, |i, j| avg(b(i, j), h(i, j))),
		(3, 1) => fill_block(out, bw, bh, |i, j| avg(b(i, j), m(i, j))),
		(1, 3) => fill_block(out, bw, bh, |i, j| avg(h(i, j), s(i, j))),
		_ => fill_block(out, bw, bh, |i, j| avg(m(i, j), s(i, j))),
	}
}

#[inline(always)]
fn fill_block<T: Pixel>(out: &mut [T], bw: usize, bh: usize, f: impl Fn(usize, usize) -> T) {
	for j in 0..bh {
		for (i, v) in out[j * bw..j * bw + bw].iter_mut().enumerate() {
			*v = f(i, j);
		}
	}
}

/// Chroma sample interpolation (8.4.2.2.2) with eighth-sample fraction
/// (fx, fy).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn mc_chroma<T: Pixel>(
	r: &Src<T>,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [T],
) {
	if crate::simd::enabled() && matches!(bw, 4 | 8) {
		if let (Some(r8), Some(o8)) = (T::narrow_src(r), T::narrow_mut(out)) {
			return super::simd::mc_chroma(&r8, x, y, fx, fy, bw, bh, o8);
		}
	}
	mc_chroma_scalar(r, x, y, fx, fy, bw, bh, out)
}

/// Scalar version of [`mc_chroma`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn mc_chroma_scalar<T: Pixel>(
	r: &Src<T>,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [T],
) {
	const CW: usize = 9;
	let mut win = [T::default(); CW * 17];
	fetch(r, x, y, bw + 1, bh + 1, &mut win, CW);
	let (w00, w10, w01, w11) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
	for j in 0..bh {
		let r0 = &win[j * CW..j * CW + CW];
		let r1 = &win[(j + 1) * CW..(j + 1) * CW + CW];
		for i in 0..bw {
			let v =
				w00 * r0[i].into() + w10 * r0[i + 1].into() + w01 * r1[i].into() + w11 * r1[i + 1].into();
			out[j * bw + i] = T::of((v + 32) >> 6);
		}
	}
}

/// Weighting parameters for one block and colour component.
#[derive(Debug, Clone, Copy)]
pub enum Weighting {
	/// Default: average for bi-prediction, copy otherwise.
	Default,
	/// Explicit/implicit: `log_wd`, weights and offsets per list (offsets
	/// already scaled to the bit depth).
	Weighted { log_wd: u32, w: [i32; 2], o: [i32; 2] },
}

/// Combines one or two prediction blocks into the view (8.4.2.3).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn store_prediction<T: Pixel>(
	p: &mut Dst<T>,
	x: usize,
	y: usize,
	bw: usize,
	bh: usize,
	l0: Option<&[T]>,
	l1: Option<&[T]>,
	wt: Weighting,
) {
	if crate::simd::enabled() && matches!(bw, 4 | 8 | 16) && super::simd::weights_supported(wt) {
		let a = l0.map(T::narrow);
		let b = l1.map(T::narrow);
		if !matches!(a, Some(None)) && !matches!(b, Some(None)) {
			if let Some(mut d) = T::narrow_dst(p) {
				return super::simd::store_prediction(&mut d, x, y, bw, bh, a.flatten(), b.flatten(), wt);
			}
		}
	}
	store_prediction_scalar(p, x, y, bw, bh, l0, l1, wt)
}

/// Scalar version of [`store_prediction`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn store_prediction_scalar<T: Pixel>(
	p: &mut Dst<T>,
	x: usize,
	y: usize,
	bw: usize,
	bh: usize,
	l0: Option<&[T]>,
	l1: Option<&[T]>,
	wt: Weighting,
) {
	let max = p.max;
	#[inline(always)]
	fn rows_impl<T: Pixel>(
		p: &mut Dst<T>,
		x: usize,
		y: usize,
		bw: usize,
		bh: usize,
		f: impl Fn(usize) -> T,
	) {
		for j in 0..bh {
			let dst = p.row_mut(x, y + j, bw);
			for (i, d) in dst.iter_mut().enumerate() {
				*d = f(j * bw + i);
			}
		}
	}
	let px = |v: T| -> i32 { v.into() };
	match (l0, l1, wt) {
		(Some(a), Some(b), Weighting::Default) => rows_impl(p, x, y, bw, bh, |k| avg(a[k], b[k])),
		(Some(a), None, Weighting::Default) | (None, Some(a), Weighting::Default) => {
			for j in 0..bh {
				p.row_mut(x, y + j, bw).copy_from_slice(&a[j * bw..j * bw + bw]);
			}
		}
		(Some(a), Some(b), Weighting::Weighted { log_wd, w: wt, o }) => {
			let round = 1 << log_wd;
			let off = (o[0] + o[1] + 1) >> 1;
			rows_impl(p, x, y, bw, bh, |k| {
				T::clip(((px(a[k]) * wt[0] + px(b[k]) * wt[1] + round) >> (log_wd + 1)) + off, max)
			})
		}
		(Some(a), None, Weighting::Weighted { log_wd, w: wt, o }) => {
			rows_impl(p, x, y, bw, bh, |k| T::clip(uni_weight(px(a[k]), log_wd, wt[0], o[0]), max))
		}
		(None, Some(a), Weighting::Weighted { log_wd, w: wt, o }) => {
			rows_impl(p, x, y, bw, bh, |k| T::clip(uni_weight(px(a[k]), log_wd, wt[1], o[1]), max))
		}
		(None, None, _) => rows_impl(p, x, y, bw, bh, |_| T::of(mid(max))),
	}
}

#[inline]
fn uni_weight(v: i32, log_wd: u32, w: i32, o: i32) -> i32 {
	if log_wd >= 1 { ((v * w + (1 << (log_wd - 1))) >> log_wd) + o } else { v * w + o }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn idct4_dc_only() {
		let mut p = Plane::new(4, 4, 100u8);
		let mut c = [0i32; 16];
		c[0] = 64; // DC 64 -> +64 * ... >> 6 per stage: (64 + 32) >> 6 = 1
		idct4_add(&mut p.d8(), 0, 0, &c);
		assert!(p.data.iter().all(|&v| v == 101));
	}

	#[test]
	fn full_pel_mc_copies() {
		let mut r = Plane::new(32, 32, 0u8);
		for (i, v) in r.data.iter_mut().enumerate() {
			*v = (i % 251) as u8;
		}
		let mut out = [0u8; 16];
		mc_luma(&r.s8(), 5, 7, 0, 0, 4, 4, &mut out);
		assert_eq!(out[0], r.at(5, 7));
		assert_eq!(out[15], r.at(8, 10));
		// Far outside: clamps to the corner sample.
		mc_luma(&r.s8(), -100, -100, 0, 0, 4, 4, &mut out);
		assert!(out.iter().all(|&v| v == r.at(0, 0)));
	}

	#[test]
	fn field_views_address_alternate_rows() {
		let mut p = Plane::new(4, 8, 0u16);
		for (i, v) in p.data.iter_mut().enumerate() {
			*v = i as u16;
		}
		let bottom = p.src(Some(1), 1023);
		assert_eq!((bottom.height, bottom.clamped(1, 0)), (4, 5));
		assert_eq!(bottom.clamped(2, 3), p.at(2, 7));
		// Clamping stays inside the field.
		assert_eq!(bottom.clamped(0, 9), p.at(0, 7));
		let mut top = p.dst(Some(0), 1023);
		top.set(3, 1, 999);
		assert_eq!(p.at(3, 2), 999);
	}

	#[test]
	fn high_bit_depth_clips_to_its_range() {
		let mut p = Plane::new(4, 4, 1000u16);
		let mut c = [0i32; 16];
		c[0] = 64 * 100;
		idct4_add(&mut p.dst(None, 1023), 0, 0, &c);
		assert!(p.data.iter().all(|&v| v == 1023));
	}

	#[test]
	fn chroma422_dc_transform_of_a_single_dc() {
		// One DC coefficient spreads evenly over all eight blocks.
		let mut c = [0i32; 8];
		c[0] = 4;
		let out = chroma422_dc_dequant(&c, 36, 16, 8);
		assert!(out.iter().all(|&v| v == out[0]) && out[0] != 0, "{out:?}");
	}
}
