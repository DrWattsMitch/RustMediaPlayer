//! Reconstruction: inverse transforms (§8.5.12-13), intra prediction
//! (§8.3), fractional-sample interpolation (§8.4.2.2) and weighted sample
//! prediction (§8.4.2.3). FFmpeg: `h264idct_template.c`,
//! `h264pred_template.c`, `h264qpel_template.c`, `h264chroma_template.c`,
//! `h264dsp_template.c`.

/// One 8-bit sample plane.
#[derive(Debug, Clone)]
pub struct Plane {
	pub data: Vec<u8>,
	pub width: usize,
	pub height: usize,
}

impl Plane {
	pub fn new(width: usize, height: usize, fill: u8) -> Self {
		Self { data: vec![fill; width * height], width, height }
	}

	#[inline]
	pub fn at(&self, x: usize, y: usize) -> u8 {
		self.data[y * self.width + x]
	}

	/// Sample with coordinates clamped into the plane (reference picture
	/// edge extension, 8-228/8-229).
	#[inline]
	pub fn clamped(&self, x: i32, y: i32) -> u8 {
		let x = x.clamp(0, self.width as i32 - 1) as usize;
		let y = y.clamp(0, self.height as i32 - 1) as usize;
		self.data[y * self.width + x]
	}
}

#[inline]
fn clip(v: i32) -> u8 {
	v.clamp(0, 255) as u8
}

// ---------------------------------------------------------------------------
// Inverse transforms

/// Adds a 4x4 inverse-transformed residual (8.5.12.2) at (x, y).
pub fn idct4_add(p: &mut Plane, x: usize, y: usize, c: &[i32; 16]) {
	if crate::simd::enabled() {
		super::simd::idct4_add(p, x, y, c)
	} else {
		idct4_add_scalar(p, x, y, c)
	}
}

/// Scalar version of [`idct4_add`]: the fallback and the reference.
pub(super) fn idct4_add_scalar(p: &mut Plane, x: usize, y: usize, c: &[i32; 16]) {
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
	for j in 0..4 {
		let e0 = t[j] + t[8 + j];
		let e1 = t[j] - t[8 + j];
		let e2 = (t[4 + j] >> 1) - t[12 + j];
		let e3 = t[4 + j] + (t[12 + j] >> 1);
		let col = [e0 + e3, e1 + e2, e1 - e2, e0 - e3];
		for (i, &v) in col.iter().enumerate() {
			let at = (y + i) * p.width + x + j;
			p.data[at] = clip(i32::from(p.data[at]) + ((v + 32) >> 6));
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
pub fn idct8_add(p: &mut Plane, x: usize, y: usize, c: &[i32; 64]) {
	if crate::simd::enabled() {
		super::simd::idct8_add(p, x, y, c)
	} else {
		idct8_add_scalar(p, x, y, c)
	}
}

/// Scalar version of [`idct8_add`]: the fallback and the reference.
pub(super) fn idct8_add_scalar(p: &mut Plane, x: usize, y: usize, c: &[i32; 64]) {
	let mut t = [0i32; 64];
	for i in 0..8 {
		let mut row = [0i32; 8];
		row.copy_from_slice(&c[i * 8..i * 8 + 8]);
		t[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d(row));
	}
	for j in 0..8 {
		let col = idct8_1d(std::array::from_fn(|i| t[i * 8 + j]));
		for (i, &v) in col.iter().enumerate() {
			let at = (y + i) * p.width + x + j;
			p.data[at] = clip(i32::from(p.data[at]) + ((v + 32) >> 6));
		}
	}
}

/// Inverse 4x4 Hadamard of Intra16x16 luma DC coefficients (8.5.10),
/// followed by dequantisation. `c` is in raster order of the 4x4 DC array.
pub fn luma_dc_dequant(c: &[i32; 16], qp: i32, scale: i32) -> [i32; 16] {
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
		*v = clamp_coeff(if shift >= 6 {
			x << (shift - 6)
		} else {
			(x + (1 << (5 - shift))) >> (6 - shift)
		});
	}
	f
}

/// Transform inputs of a conforming 8-bit stream fit in 16 bits (8.5.12.1);
/// clamping keeps damaged streams from overflowing the integer transforms.
#[inline]
fn clamp_coeff(v: i64) -> i32 {
	v.clamp(-(1 << 15), (1 << 15) - 1) as i32
}

/// Inverse 2x2 transform and dequantisation of 4:2:0 chroma DC (8.5.11).
pub fn chroma_dc_dequant(c: &[i32; 4], qp: i32, scale: i32) -> [i32; 4] {
	let f = [
		c[0] + c[1] + c[2] + c[3],
		c[0] - c[1] + c[2] - c[3],
		c[0] + c[1] - c[2] - c[3],
		c[0] - c[1] - c[2] + c[3],
	];
	f.map(|v| clamp_coeff(((i64::from(v) * i64::from(scale)) << (qp / 6)) >> 5))
}

/// Dequantises a 4x4 block in place (8.5.12.1). `skip_dc` leaves
/// coefficient 0 untouched (already-dequantised DC).
pub fn dequant4(c: &mut [i32; 16], qp: i32, scale: &[i32; 16], skip_dc: bool) {
	let shift = i64::from(qp / 6);
	for (i, v) in c.iter_mut().enumerate() {
		if (skip_dc && i == 0) || *v == 0 {
			continue;
		}
		let x = i64::from(*v) * i64::from(scale[i]);
		*v = clamp_coeff(if shift >= 4 {
			x << (shift - 4)
		} else {
			(x + (1 << (3 - shift))) >> (4 - shift)
		});
	}
}

/// Dequantises an 8x8 block in place (8.5.13.1).
pub fn dequant8(c: &mut [i32; 64], qp: i32, scale: &[i32; 64]) {
	let shift = i64::from(qp / 6);
	for (i, v) in c.iter_mut().enumerate() {
		if *v == 0 {
			continue;
		}
		let x = i64::from(*v) * i64::from(scale[i]);
		*v = clamp_coeff(if shift >= 6 {
			x << (shift - 6)
		} else {
			(x + (1 << (5 - shift))) >> (6 - shift)
		});
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

/// Intra_4x4 prediction (8.3.1.2) written into the plane at (x, y).
pub fn pred4x4(p: &mut Plane, x: usize, y: usize, mode: u8, a: Avail) {
	let w = p.width;
	let mut top = [128u8; 8];
	let mut left = [128u8; 4];
	let mut tl = 128u8;
	if a.top {
		for i in 0..4 {
			top[i] = p.data[(y - 1) * w + x + i];
		}
		for i in 4..8 {
			top[i] = if a.top_right { p.data[(y - 1) * w + x + i] } else { top[3] };
		}
	}
	if a.left {
		for (i, l) in left.iter_mut().enumerate() {
			*l = p.data[(y + i) * w + x - 1];
		}
	}
	if a.top_left {
		tl = p.data[(y - 1) * w + x - 1];
	}
	let t = |i: usize| i32::from(top[i]);
	let l = |i: usize| i32::from(left[i]);
	let q = i32::from(tl);
	let mut out = [[0u8; 4]; 4];
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
				_ => 128,
			};
			out = [[dc as u8; 4]; 4];
		}
		3 => {
			for r in 0..4 {
				for c in 0..4 {
					let v = if r == 3 && c == 3 {
						(t(6) + 3 * t(7) + 2) >> 2
					} else {
						(t(r + c) + 2 * t(r + c + 1) + t(r + c + 2) + 2) >> 2
					};
					out[r][c] = v as u8;
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
					let v = match mode {
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
					out[r as usize][c as usize] = v as u8;
				}
			}
		}
		7 => {
			for r in 0..4 {
				for c in 0..4 {
					let i = c + (r >> 1);
					let v = if r % 2 == 0 {
						(t(i) + t(i + 1) + 1) >> 1
					} else {
						(t(i) + 2 * t(i + 1) + t(i + 2) + 2) >> 2
					};
					out[r][c] = v as u8;
				}
			}
		}
		_ => {
			for r in 0..4 {
				for c in 0..4 {
					let z = c + 2 * r;
					let v = match z {
						0 | 2 | 4 => (l(z / 2) + l(z / 2 + 1) + 1) >> 1,
						1 | 3 => (l(z / 2) + 2 * l(z / 2 + 1) + l(z / 2 + 2) + 2) >> 2,
						5 => (l(2) + 3 * l(3) + 2) >> 2,
						_ => l(3),
					};
					out[r][c] = v as u8;
				}
			}
		}
	}
	for (r, row) in out.iter().enumerate() {
		p.data[(y + r) * w + x..(y + r) * w + x + 4].copy_from_slice(row);
	}
}

/// Intra_8x8 prediction with reference sample filtering (8.3.2.2).
pub fn pred8x8l(p: &mut Plane, x: usize, y: usize, mode: u8, a: Avail) {
	let w = p.width;
	// Unfiltered references.
	let mut top = [0i32; 16];
	let mut left = [0i32; 8];
	let mut q = 0i32;
	if a.top {
		for i in 0..8 {
			top[i] = i32::from(p.data[(y - 1) * w + x + i]);
		}
		for i in 8..16 {
			top[i] = if a.top_right { i32::from(p.data[(y - 1) * w + x + i]) } else { top[7] };
		}
	}
	if a.left {
		for (i, l) in left.iter_mut().enumerate() {
			*l = i32::from(p.data[(y + i) * w + x - 1]);
		}
	}
	if a.top_left {
		q = i32::from(p.data[(y - 1) * w + x - 1]);
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
				_ => 128,
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
					let v = match mode {
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
					out[r as usize][c as usize] = v;
				}
			}
		}
	}
	for (r, row) in out.iter().enumerate() {
		for (c, &v) in row.iter().enumerate() {
			p.data[(y + r) * w + x + c] = v as u8;
		}
	}
}

/// Intra_16x16 (n = 16) and chroma (n = 8) whole-block prediction:
/// mode 0 vertical, 1 horizontal, 2 DC, 3 plane (8.3.3, 8.3.4).
/// For chroma, DC is computed per 4x4 sub-block (8.3.4.1-3).
pub fn pred_block(p: &mut Plane, x: usize, y: usize, n: usize, mode: u8, a: Avail, chroma: bool) {
	let w = p.width;
	let top = |i: usize| i32::from(p.data[(y - 1) * w + x + i]);
	let left = |i: usize| i32::from(p.data[(y + i) * w + x - 1]);
	let mut out = vec![0i32; n * n];
	match mode {
		0 => {
			for r in 0..n {
				for c in 0..n {
					out[r * n + c] = top(c);
				}
			}
		}
		1 => {
			for r in 0..n {
				for c in 0..n {
					out[r * n + c] = left(r);
				}
			}
		}
		2 if !chroma => {
			let st: i32 = if a.top { (0..n).map(top).sum() } else { 0 };
			let sl: i32 = if a.left { (0..n).map(left).sum() } else { 0 };
			let dc = match (a.top, a.left) {
				(true, true) => (st + sl + 16) >> 5,
				(true, false) => (st + 8) >> 4,
				(false, true) => (sl + 8) >> 4,
				_ => 128,
			};
			out.fill(dc);
		}
		2 => {
			for by in 0..n / 4 {
				for bx in 0..n / 4 {
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
							128
						}
					} else if prefer_left {
						if a.left {
							(sl + 2) >> 2
						} else if a.top {
							(st + 2) >> 2
						} else {
							128
						}
					} else {
						match (a.top, a.left) {
							(true, true) => (st + sl + 4) >> 3,
							(true, false) => (st + 2) >> 2,
							(false, true) => (sl + 2) >> 2,
							_ => 128,
						}
					};
					for r in 0..4 {
						for c in 0..4 {
							out[(by * 4 + r) * n + bx * 4 + c] = dc;
						}
					}
				}
			}
		}
		_ => {
			let q = i32::from(p.data[(y - 1) * w + x - 1]);
			let half = (n / 2) as i32;
			let tq = |i: i32| if i < 0 { q } else { top(i as usize) };
			let lq = |i: i32| if i < 0 { q } else { left(i as usize) };
			let mut hh = 0;
			let mut vv = 0;
			for i in 0..half {
				hh += (i + 1) * (tq(half + i) - tq(half - 2 - i));
				vv += (i + 1) * (lq(half + i) - lq(half - 2 - i));
			}
			let n_i = n as i32;
			let a_ = 16 * (left(n - 1) + top(n - 1));
			let (b, c) = if chroma {
				((34 * hh + 32) >> 6, (34 * vv + 32) >> 6)
			} else {
				((5 * hh + 32) >> 6, (5 * vv + 32) >> 6)
			};
			for r in 0..n_i {
				for cc in 0..n_i {
					out[(r * n_i + cc) as usize] = (a_ + b * (cc - half + 1) + c * (r - half + 1) + 16) >> 5;
				}
			}
		}
	}
	for r in 0..n {
		for c in 0..n {
			p.data[(y + r) * w + x + c] = clip(out[r * n + c]);
		}
	}
}

// ---------------------------------------------------------------------------
// Inter prediction

/// Window width for luma interpolation: a 16-sample block plus 5 taps.
const WIN: usize = 16 + 5;

#[inline(always)]
fn tap6(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8) -> i32 {
	i32::from(a) + i32::from(f) - 5 * (i32::from(b) + i32::from(e))
		+ 20 * (i32::from(c) + i32::from(d))
}

#[inline(always)]
fn avg(a: u8, b: u8) -> u8 {
	((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// Copies the (w x h) area at (x, y) of `r` into `dst` (row stride
/// `stride`), clamping coordinates at the picture edges.
pub(super) fn fetch(r: &Plane, x: i32, y: i32, w: usize, h: usize, dst: &mut [u8], stride: usize) {
	let inside =
		x >= 0 && y >= 0 && x + w as i32 <= r.width as i32 && y + h as i32 <= r.height as i32;
	for j in 0..h {
		let row = &mut dst[j * stride..j * stride + w];
		if inside {
			let start = (y as usize + j) * r.width + x as usize;
			row.copy_from_slice(&r.data[start..start + w]);
		} else {
			let sy = (y + j as i32).clamp(0, r.height as i32 - 1) as usize * r.width;
			for (i, v) in row.iter_mut().enumerate() {
				*v = r.data[sy + (x + i as i32).clamp(0, r.width as i32 - 1) as usize];
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
pub fn mc_luma(r: &Plane, x: i32, y: i32, fx: i32, fy: i32, bw: usize, bh: usize, out: &mut [u8]) {
	if crate::simd::enabled() && matches!(bw, 4 | 8 | 16) {
		super::simd::mc_luma(r, x, y, fx, fy, bw, bh, out)
	} else {
		mc_luma_scalar(r, x, y, fx, fy, bw, bh, out)
	}
}

/// Scalar version of [`mc_luma`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn mc_luma_scalar(
	r: &Plane,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [u8],
) {
	let mut win = [0u8; WIN * WIN];
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

	// Horizontal half samples for rows 0..=bh (row bh is `s` of the last row).
	let mut hb = [0u8; 16 * 17];
	if need_b || need_s {
		for j in 0..=bh {
			let row = &win[(j + 2) * WIN..(j + 2) * WIN + WIN];
			for i in 0..bw {
				let v = tap6(row[i], row[i + 1], row[i + 2], row[i + 3], row[i + 4], row[i + 5]);
				hb[j * 16 + i] = clip((v + 16) >> 5);
			}
		}
	}
	// Vertical half samples for columns 0..=bw (column bw is `m`).
	let mut vb = [0u8; 17 * 16];
	if need_h || need_m {
		for j in 0..bh {
			for i in 0..=bw {
				let c = i + 2;
				let v = tap6(
					win[j * WIN + c],
					win[(j + 1) * WIN + c],
					win[(j + 2) * WIN + c],
					win[(j + 3) * WIN + c],
					win[(j + 4) * WIN + c],
					win[(j + 5) * WIN + c],
				);
				vb[j * 17 + i] = clip((v + 16) >> 5);
			}
		}
	}
	// Centre samples from unrounded horizontal intermediates.
	let mut jb = [0u8; 256];
	if need_j {
		let mut tmp = [0i32; WIN * 16];
		for rr in 0..bh + 5 {
			let row = &win[rr * WIN..rr * WIN + WIN];
			for i in 0..bw {
				tmp[rr * 16 + i] = tap6(row[i], row[i + 1], row[i + 2], row[i + 3], row[i + 4], row[i + 5]);
			}
		}
		for j in 0..bh {
			for i in 0..bw {
				let t = |k: usize| tmp[(j + k) * 16 + i];
				let v = t(0) + t(5) - 5 * (t(1) + t(4)) + 20 * (t(2) + t(3));
				jb[j * 16 + i] = clip((v + 512) >> 10);
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
fn fill_block(out: &mut [u8], bw: usize, bh: usize, f: impl Fn(usize, usize) -> u8) {
	for j in 0..bh {
		for (i, v) in out[j * bw..j * bw + bw].iter_mut().enumerate() {
			*v = f(i, j);
		}
	}
}

/// Chroma sample interpolation (8.4.2.2.2) for 4:2:0 with eighth-sample
/// fraction (fx, fy).
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(
	r: &Plane,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [u8],
) {
	if crate::simd::enabled() && matches!(bw, 4 | 8) {
		super::simd::mc_chroma(r, x, y, fx, fy, bw, bh, out)
	} else {
		mc_chroma_scalar(r, x, y, fx, fy, bw, bh, out)
	}
}

/// Scalar version of [`mc_chroma`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn mc_chroma_scalar(
	r: &Plane,
	x: i32,
	y: i32,
	fx: i32,
	fy: i32,
	bw: usize,
	bh: usize,
	out: &mut [u8],
) {
	const CW: usize = 9;
	let mut win = [0u8; CW * CW];
	fetch(r, x, y, bw + 1, bh + 1, &mut win, CW);
	let (w00, w10, w01, w11) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
	for j in 0..bh {
		let r0 = &win[j * CW..j * CW + CW];
		let r1 = &win[(j + 1) * CW..(j + 1) * CW + CW];
		for i in 0..bw {
			let v = w00 * i32::from(r0[i])
				+ w10 * i32::from(r0[i + 1])
				+ w01 * i32::from(r1[i])
				+ w11 * i32::from(r1[i + 1]);
			out[j * bw + i] = ((v + 32) >> 6) as u8;
		}
	}
}

/// Weighting parameters for one block and colour component.
#[derive(Debug, Clone, Copy)]
pub enum Weighting {
	/// Default: average for bi-prediction, copy otherwise.
	Default,
	/// Explicit/implicit: `log_wd`, weights and offsets per list.
	Weighted { log_wd: u32, w: [i32; 2], o: [i32; 2] },
}

/// Combines one or two prediction blocks into the plane (8.4.2.3).
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
	if crate::simd::enabled() && matches!(bw, 4 | 8 | 16) && super::simd::weights_supported(wt) {
		super::simd::store_prediction(p, x, y, bw, bh, l0, l1, wt)
	} else {
		store_prediction_scalar(p, x, y, bw, bh, l0, l1, wt)
	}
}

/// Scalar version of [`store_prediction`]: the fallback and the reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn store_prediction_scalar(
	p: &mut Plane,
	x: usize,
	y: usize,
	bw: usize,
	bh: usize,
	l0: Option<&[u8]>,
	l1: Option<&[u8]>,
	wt: Weighting,
) {
	let w = p.width;
	#[inline(always)]
	fn rows_impl(p: &mut Plane, x: usize, y: usize, bw: usize, bh: usize, f: impl Fn(usize) -> u8) {
		let w = p.width;
		for j in 0..bh {
			let dst = &mut p.data[(y + j) * w + x..(y + j) * w + x + bw];
			for (i, d) in dst.iter_mut().enumerate() {
				*d = f(j * bw + i);
			}
		}
	}
	match (l0, l1, wt) {
		(Some(a), Some(b), Weighting::Default) => rows_impl(p, x, y, bw, bh, |k| avg(a[k], b[k])),
		(Some(a), None, Weighting::Default) | (None, Some(a), Weighting::Default) => {
			for j in 0..bh {
				p.data[(y + j) * w + x..(y + j) * w + x + bw].copy_from_slice(&a[j * bw..j * bw + bw]);
			}
		}
		(Some(a), Some(b), Weighting::Weighted { log_wd, w: wt, o }) => {
			let round = 1 << log_wd;
			let off = (o[0] + o[1] + 1) >> 1;
			rows_impl(p, x, y, bw, bh, |k| {
				clip(((i32::from(a[k]) * wt[0] + i32::from(b[k]) * wt[1] + round) >> (log_wd + 1)) + off)
			})
		}
		(Some(a), None, Weighting::Weighted { log_wd, w: wt, o }) => {
			rows_impl(p, x, y, bw, bh, |k| clip(uni_weight(a[k], log_wd, wt[0], o[0])))
		}
		(None, Some(a), Weighting::Weighted { log_wd, w: wt, o }) => {
			rows_impl(p, x, y, bw, bh, |k| clip(uni_weight(a[k], log_wd, wt[1], o[1])))
		}
		(None, None, _) => rows_impl(p, x, y, bw, bh, |_| 128),
	}
}

#[inline]
fn uni_weight(v: u8, log_wd: u32, w: i32, o: i32) -> i32 {
	if log_wd >= 1 {
		((i32::from(v) * w + (1 << (log_wd - 1))) >> log_wd) + o
	} else {
		i32::from(v) * w + o
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn idct4_dc_only() {
		let mut p = Plane::new(4, 4, 100);
		let mut c = [0i32; 16];
		c[0] = 64; // DC 64 -> +64 * ... >> 6 per stage: (64 + 32) >> 6 = 1
		idct4_add(&mut p, 0, 0, &c);
		assert!(p.data.iter().all(|&v| v == 101));
	}

	#[test]
	fn full_pel_mc_copies() {
		let mut r = Plane::new(32, 32, 0);
		for (i, v) in r.data.iter_mut().enumerate() {
			*v = (i % 251) as u8;
		}
		let mut out = [0u8; 16];
		mc_luma(&r, 5, 7, 0, 0, 4, 4, &mut out);
		assert_eq!(out[0], r.at(5, 7));
		assert_eq!(out[15], r.at(8, 10));
		// Far outside: clamps to the corner sample.
		mc_luma(&r, -100, -100, 0, 0, 4, 4, &mut out);
		assert!(out.iter().all(|&v| v == r.at(0, 0)));
	}
}
