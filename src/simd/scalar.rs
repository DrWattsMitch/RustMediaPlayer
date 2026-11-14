//! Plain-array backend: the vector types on targets without SSE2/NEON, and
//! the reference model the native backends are tested against. No `unsafe`.

// Methods are named `add`, `mul`, `shr`, ... rather than implementing
// `std::ops`: these wrap (or saturate, where named so), and `a + b` would
// read as Rust's overflow-checked arithmetic.
#![allow(clippy::should_implement_trait)]
#![allow(clippy::needless_range_loop)]

macro_rules! lanes {
	($n:expr, |$i:ident| $e:expr) => {
		std::array::from_fn::<_, $n, _>(|$i| $e)
	};
}

fn mask16(b: bool) -> i16 {
	if b { -1 } else { 0 }
}

fn mask32(b: bool) -> i32 {
	if b { -1 } else { 0 }
}

/// 16 lanes of `u8`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U8x16(pub [u8; 16]);

impl U8x16 {
	#[inline(always)]
	pub fn splat(v: u8) -> Self {
		Self([v; 16])
	}

	#[inline(always)]
	pub fn load(s: &[u8]) -> Self {
		Self(lanes!(16, |i| s[..16][i]))
	}

	#[inline(always)]
	pub fn store(self, d: &mut [u8]) {
		d[..16].copy_from_slice(&self.0);
	}

	#[inline(always)]
	pub fn load8(s: &[u8]) -> Self {
		let s = &s[..8];
		Self(lanes!(16, |i| if i < 8 { s[i] } else { 0 }))
	}

	#[inline(always)]
	pub fn store8(self, d: &mut [u8]) {
		d[..8].copy_from_slice(&self.0[..8]);
	}

	#[inline(always)]
	pub fn load4(s: &[u8]) -> Self {
		let s = &s[..4];
		Self(lanes!(16, |i| if i < 4 { s[i] } else { 0 }))
	}

	#[inline(always)]
	pub fn store4(self, d: &mut [u8]) {
		d[..4].copy_from_slice(&self.0[..4]);
	}

	#[inline(always)]
	pub fn to_array(self) -> [u8; 16] {
		self.0
	}

	#[inline(always)]
	pub fn avg(self, o: Self) -> Self {
		Self(lanes!(16, |i| ((u16::from(self.0[i]) + u16::from(o.0[i]) + 1) >> 1) as u8))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		Self(lanes!(16, |i| self.0[i].min(o.0[i])))
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		Self(lanes!(16, |i| self.0[i].max(o.0[i])))
	}

	#[inline(always)]
	pub fn abs_diff(self, o: Self) -> Self {
		Self(lanes!(16, |i| self.0[i].abs_diff(o.0[i])))
	}

	#[inline(always)]
	pub fn saturating_add(self, o: Self) -> Self {
		Self(lanes!(16, |i| self.0[i].saturating_add(o.0[i])))
	}

	#[inline(always)]
	pub fn saturating_sub(self, o: Self) -> Self {
		Self(lanes!(16, |i| self.0[i].saturating_sub(o.0[i])))
	}

	#[inline(always)]
	pub fn widen_lo(self) -> I16x8 {
		I16x8(lanes!(8, |i| i16::from(self.0[i])))
	}

	#[inline(always)]
	pub fn widen_hi(self) -> I16x8 {
		I16x8(lanes!(8, |i| i16::from(self.0[i + 8])))
	}
}

/// 8 lanes of `i16`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct I16x8(pub [i16; 8]);

impl I16x8 {
	#[inline(always)]
	pub fn splat(v: i16) -> Self {
		Self([v; 8])
	}

	#[inline(always)]
	pub fn from_array(a: [i16; 8]) -> Self {
		Self(a)
	}

	#[inline(always)]
	pub fn load(s: &[i16]) -> Self {
		Self(lanes!(8, |i| s[..8][i]))
	}

	#[inline(always)]
	pub fn store(self, d: &mut [i16]) {
		d[..8].copy_from_slice(&self.0);
	}

	#[inline(always)]
	pub fn load_u8(s: &[u8]) -> Self {
		Self(lanes!(8, |i| i16::from(s[..8][i])))
	}

	#[inline(always)]
	pub fn load_u16(s: &[u16]) -> Self {
		Self(lanes!(8, |i| s[..8][i] as i16))
	}

	#[inline(always)]
	pub fn store_u16(self, d: &mut [u16]) {
		for (d, &v) in d[..8].iter_mut().zip(&self.0) {
			*d = v as u16;
		}
	}

	#[inline(always)]
	pub fn to_array(self) -> [i16; 8] {
		self.0
	}

	fn map2(self, o: Self, f: impl Fn(i16, i16) -> i16) -> Self {
		Self(lanes!(8, |i| f(self.0[i], o.0[i])))
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		self.map2(o, i16::wrapping_add)
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		self.map2(o, i16::wrapping_sub)
	}

	#[inline(always)]
	pub fn saturating_add(self, o: Self) -> Self {
		self.map2(o, i16::saturating_add)
	}

	#[inline(always)]
	pub fn saturating_sub(self, o: Self) -> Self {
		self.map2(o, i16::saturating_sub)
	}

	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		self.map2(o, i16::wrapping_mul)
	}

	#[inline(always)]
	pub fn shr(self, n: u32) -> Self {
		Self(self.0.map(|v| v >> n))
	}

	#[inline(always)]
	pub fn shr_logical(self, n: u32) -> Self {
		Self(self.0.map(|v| ((v as u16) >> n) as i16))
	}

	#[inline(always)]
	pub fn shl(self, n: u32) -> Self {
		Self(self.0.map(|v| ((v as u16) << n) as i16))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		self.map2(o, i16::min)
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		self.map2(o, i16::max)
	}

	#[inline(always)]
	pub fn clamp(self, lo: Self, hi: Self) -> Self {
		self.max(lo).min(hi)
	}

	#[inline(always)]
	pub fn abs(self) -> Self {
		Self(self.0.map(i16::wrapping_abs))
	}

	#[inline(always)]
	pub fn gt(self, o: Self) -> Self {
		self.map2(o, |a, b| mask16(a > b))
	}

	#[inline(always)]
	pub fn lt(self, o: Self) -> Self {
		self.map2(o, |a, b| mask16(a < b))
	}

	#[inline(always)]
	pub fn eq(self, o: Self) -> Self {
		self.map2(o, |a, b| mask16(a == b))
	}

	#[inline(always)]
	pub fn and(self, o: Self) -> Self {
		self.map2(o, |a, b| a & b)
	}

	#[inline(always)]
	pub fn or(self, o: Self) -> Self {
		self.map2(o, |a, b| a | b)
	}

	#[inline(always)]
	pub fn xor(self, o: Self) -> Self {
		self.map2(o, |a, b| a ^ b)
	}

	#[inline(always)]
	pub fn and_not(self, o: Self) -> Self {
		self.map2(o, |a, b| a & !b)
	}

	#[inline(always)]
	pub fn not(self) -> Self {
		Self(self.0.map(|v| !v))
	}

	#[inline(always)]
	pub fn select(self, a: Self, b: Self) -> Self {
		Self(lanes!(8, |i| (self.0[i] & a.0[i]) | (!self.0[i] & b.0[i])))
	}

	#[inline(always)]
	pub fn any(self) -> bool {
		self.0.iter().any(|&v| v != 0)
	}

	#[inline(always)]
	pub fn widen_lo(self) -> I32x4 {
		I32x4(lanes!(4, |i| i32::from(self.0[i])))
	}

	#[inline(always)]
	pub fn widen_hi(self) -> I32x4 {
		I32x4(lanes!(4, |i| i32::from(self.0[i + 4])))
	}

	#[inline(always)]
	pub fn madd(self, o: Self) -> I32x4 {
		let p = |k: usize| i32::from(self.0[k]) * i32::from(o.0[k]);
		I32x4(lanes!(4, |i| p(2 * i).wrapping_add(p(2 * i + 1))))
	}

	#[inline(always)]
	pub fn pack_u8(self, hi: Self) -> U8x16 {
		let s = |v: i16| v.clamp(0, 255) as u8;
		U8x16(lanes!(16, |i| if i < 8 { s(self.0[i]) } else { s(hi.0[i - 8]) }))
	}

	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(lanes!(8, |i| if i % 2 == 0 { self.0[i / 2] } else { o.0[i / 2] }))
	}

	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(lanes!(8, |i| if i % 2 == 0 { self.0[4 + i / 2] } else { o.0[4 + i / 2] }))
	}

	/// `[a0, a1, b0, b1, a2, a3, b2, b3]` (32-bit pairs interleaved).
	#[inline(always)]
	pub fn zip_lo32(self, o: Self) -> Self {
		zip32(self, o, 0)
	}

	/// `[a4, a5, b4, b5, a6, a7, b6, b7]`.
	#[inline(always)]
	pub fn zip_hi32(self, o: Self) -> Self {
		zip32(self, o, 4)
	}

	#[inline(always)]
	pub fn zip_lo64(self, o: Self) -> Self {
		Self(lanes!(8, |i| if i < 4 { self.0[i] } else { o.0[i - 4] }))
	}

	#[inline(always)]
	pub fn zip_hi64(self, o: Self) -> Self {
		Self(lanes!(8, |i| if i < 4 { self.0[i + 4] } else { o.0[i] }))
	}
}

fn zip32(s: I16x8, o: I16x8, b: usize) -> I16x8 {
	let (s, o) = (s.0, o.0);
	I16x8([s[b], s[b + 1], o[b], o[b + 1], s[b + 2], s[b + 3], o[b + 2], o[b + 3]])
}

/// 4 lanes of `i32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct I32x4(pub [i32; 4]);

impl I32x4 {
	#[inline(always)]
	pub fn splat(v: i32) -> Self {
		Self([v; 4])
	}

	#[inline(always)]
	pub fn from_array(a: [i32; 4]) -> Self {
		Self(a)
	}

	#[inline(always)]
	pub fn load(s: &[i32]) -> Self {
		Self(lanes!(4, |i| s[..4][i]))
	}

	#[inline(always)]
	pub fn store(self, d: &mut [i32]) {
		d[..4].copy_from_slice(&self.0);
	}

	#[inline(always)]
	pub fn to_array(self) -> [i32; 4] {
		self.0
	}

	fn map2(self, o: Self, f: impl Fn(i32, i32) -> i32) -> Self {
		Self(lanes!(4, |i| f(self.0[i], o.0[i])))
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		self.map2(o, i32::wrapping_add)
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		self.map2(o, i32::wrapping_sub)
	}

	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		self.map2(o, i32::wrapping_mul)
	}

	#[inline(always)]
	pub fn shr(self, n: u32) -> Self {
		Self(self.0.map(|v| v >> n))
	}

	#[inline(always)]
	pub fn shl(self, n: u32) -> Self {
		Self(self.0.map(|v| ((v as u32) << n) as i32))
	}

	#[inline(always)]
	pub fn gt(self, o: Self) -> Self {
		self.map2(o, |a, b| mask32(a > b))
	}

	#[inline(always)]
	pub fn lt(self, o: Self) -> Self {
		self.map2(o, |a, b| mask32(a < b))
	}

	#[inline(always)]
	pub fn eq(self, o: Self) -> Self {
		self.map2(o, |a, b| mask32(a == b))
	}

	#[inline(always)]
	pub fn and(self, o: Self) -> Self {
		self.map2(o, |a, b| a & b)
	}

	#[inline(always)]
	pub fn or(self, o: Self) -> Self {
		self.map2(o, |a, b| a | b)
	}

	#[inline(always)]
	pub fn select(self, a: Self, b: Self) -> Self {
		Self(lanes!(4, |i| (self.0[i] & a.0[i]) | (!self.0[i] & b.0[i])))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		self.map2(o, i32::min)
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		self.map2(o, i32::max)
	}

	#[inline(always)]
	pub fn clamp(self, lo: Self, hi: Self) -> Self {
		self.max(lo).min(hi)
	}

	#[inline(always)]
	pub fn any(self) -> bool {
		self.0.iter().any(|&v| v != 0)
	}

	#[inline(always)]
	pub fn pack_i16(self, hi: Self) -> I16x8 {
		let s = |v: i32| v.clamp(-32768, 32767) as i16;
		I16x8(lanes!(8, |i| if i < 4 { s(self.0[i]) } else { s(hi.0[i - 4]) }))
	}

	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self([self.0[0], o.0[0], self.0[1], o.0[1]])
	}

	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self([self.0[2], o.0[2], self.0[3], o.0[3]])
	}
	#[inline(always)]
	pub fn zip_lo64(self, o: Self) -> Self {
		Self([self.0[0], self.0[1], o.0[0], o.0[1]])
	}

	#[inline(always)]
	pub fn zip_hi64(self, o: Self) -> Self {
		Self([self.0[2], self.0[3], o.0[2], o.0[3]])
	}
}

/// 4 lanes of `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32x4(pub [f32; 4]);

impl F32x4 {
	#[inline(always)]
	pub fn splat(v: f32) -> Self {
		Self([v; 4])
	}

	#[inline(always)]
	pub fn from_array(a: [f32; 4]) -> Self {
		Self(a)
	}

	#[inline(always)]
	pub fn load(s: &[f32]) -> Self {
		Self(lanes!(4, |i| s[..4][i]))
	}

	#[inline(always)]
	pub fn store(self, d: &mut [f32]) {
		d[..4].copy_from_slice(&self.0);
	}

	#[inline(always)]
	pub fn to_array(self) -> [f32; 4] {
		self.0
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(lanes!(4, |i| self.0[i] + o.0[i]))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(lanes!(4, |i| self.0[i] - o.0[i]))
	}

	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		Self(lanes!(4, |i| self.0[i] * o.0[i]))
	}

	#[inline(always)]
	pub fn neg(self) -> Self {
		Self(self.0.map(|v| -v))
	}

	#[inline(always)]
	pub fn reverse(self) -> Self {
		let a = self.0;
		Self([a[3], a[2], a[1], a[0]])
	}

	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self([self.0[0], o.0[0], self.0[1], o.0[1]])
	}

	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self([self.0[2], o.0[2], self.0[3], o.0[3]])
	}

	#[inline(always)]
	pub fn unzip(self, o: Self) -> (Self, Self) {
		let (a, b) = (self.0, o.0);
		(Self([a[0], a[2], b[0], b[2]]), Self([a[1], a[3], b[1], b[3]]))
	}
}
