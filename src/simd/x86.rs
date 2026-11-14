//! SSE2 backend (x86_64 baseline). See the module docs of [`super`] for why
//! each `unsafe` here is sound: `sse!` wraps value intrinsics, which are
//! only `unsafe` because they need SSE2, and this file is compiled only when
//! SSE2 is a target feature; loads and stores re-slice to the exact length
//! they access before using the pointer.

// Methods are named `add`, `mul`, `shr`, ... rather than implementing
// `std::ops`: these wrap (or saturate, where named so), and `a + b` would
// read as Rust's overflow-checked arithmetic.
#![allow(clippy::should_implement_trait)]

use core::arch::x86_64::*;

/// Calls SSE2 value intrinsics. SAFETY: this module is compiled only with
/// `target_feature = "sse2"` (see `mod.rs`), the intrinsics' one
/// requirement.
macro_rules! sse {
	($e:expr) => {
		unsafe { $e }
	};
}

#[inline(always)]
fn zero() -> __m128i {
	sse!(_mm_setzero_si128())
}

#[inline(always)]
fn count(n: u32) -> __m128i {
	sse!(_mm_cvtsi32_si128(n as i32))
}

/// Lane-wise select: `m` lanes are all ones or all zeros.
#[inline(always)]
fn blend(m: __m128i, a: __m128i, b: __m128i) -> __m128i {
	sse!(_mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b)))
}

// ---------------------------------------------------------------------------

/// 16 lanes of `u8`.
#[derive(Clone, Copy)]
pub struct U8x16(__m128i);

impl U8x16 {
	#[inline(always)]
	pub fn splat(v: u8) -> Self {
		Self(sse!(_mm_set1_epi8(v as i8)))
	}

	/// Loads `s[..16]`.
	#[inline(always)]
	pub fn load(s: &[u8]) -> Self {
		let s = &s[..16];
		// SAFETY: `s` is 16 readable bytes; `loadu` has no alignment need.
		Self(unsafe { _mm_loadu_si128(s.as_ptr().cast()) })
	}

	/// Stores into `d[..16]`.
	#[inline(always)]
	pub fn store(self, d: &mut [u8]) {
		let d = &mut d[..16];
		// SAFETY: `d` is 16 writable bytes; `storeu` has no alignment need.
		unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), self.0) }
	}

	/// Loads `s[..8]` into the low half; the high half is zero.
	#[inline(always)]
	pub fn load8(s: &[u8]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 8 readable bytes, exactly what `loadl_epi64` reads.
		Self(unsafe { _mm_loadl_epi64(s.as_ptr().cast()) })
	}

	/// Stores the low half into `d[..8]`.
	#[inline(always)]
	pub fn store8(self, d: &mut [u8]) {
		let d = &mut d[..8];
		// SAFETY: `d` is 8 writable bytes, exactly what `storel_epi64` writes.
		unsafe { _mm_storel_epi64(d.as_mut_ptr().cast(), self.0) }
	}

	/// Loads `s[..4]` into lanes 0..4; the rest are zero.
	#[inline(always)]
	pub fn load4(s: &[u8]) -> Self {
		let b: [u8; 4] = std::array::from_fn(|i| s[i]);
		Self(sse!(_mm_cvtsi32_si128(i32::from_le_bytes(b))))
	}

	/// Stores lanes 0..4 into `d[..4]`.
	#[inline(always)]
	pub fn store4(self, d: &mut [u8]) {
		let v = sse!(_mm_cvtsi128_si32(self.0));
		d[..4].copy_from_slice(&v.to_le_bytes());
	}

	#[inline(always)]
	pub fn to_array(self) -> [u8; 16] {
		let mut a = [0u8; 16];
		self.store(&mut a);
		a
	}

	/// `(a + b + 1) >> 1`.
	#[inline(always)]
	pub fn avg(self, o: Self) -> Self {
		Self(sse!(_mm_avg_epu8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		Self(sse!(_mm_min_epu8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		Self(sse!(_mm_max_epu8(self.0, o.0)))
	}

	/// `|a - b|`.
	#[inline(always)]
	pub fn abs_diff(self, o: Self) -> Self {
		Self(sse!(_mm_or_si128(_mm_subs_epu8(self.0, o.0), _mm_subs_epu8(o.0, self.0))))
	}

	#[inline(always)]
	pub fn saturating_add(self, o: Self) -> Self {
		Self(sse!(_mm_adds_epu8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_sub(self, o: Self) -> Self {
		Self(sse!(_mm_subs_epu8(self.0, o.0)))
	}

	/// Lanes 0..8 zero-extended.
	#[inline(always)]
	pub fn widen_lo(self) -> I16x8 {
		I16x8(sse!(_mm_unpacklo_epi8(self.0, zero())))
	}

	/// Lanes 8..16 zero-extended.
	#[inline(always)]
	pub fn widen_hi(self) -> I16x8 {
		I16x8(sse!(_mm_unpackhi_epi8(self.0, zero())))
	}
}

// ---------------------------------------------------------------------------

/// 8 lanes of `i16`.
#[derive(Clone, Copy)]
pub struct I16x8(__m128i);

impl I16x8 {
	#[inline(always)]
	pub fn splat(v: i16) -> Self {
		Self(sse!(_mm_set1_epi16(v)))
	}

	#[inline(always)]
	pub fn from_array(a: [i16; 8]) -> Self {
		Self::load(&a)
	}

	/// Loads `s[..8]`.
	#[inline(always)]
	pub fn load(s: &[i16]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 16 readable bytes; `loadu` has no alignment need.
		Self(unsafe { _mm_loadu_si128(s.as_ptr().cast()) })
	}

	/// Stores into `d[..8]`.
	#[inline(always)]
	pub fn store(self, d: &mut [i16]) {
		let d = &mut d[..8];
		// SAFETY: `d` is 16 writable bytes; `storeu` has no alignment need.
		unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), self.0) }
	}

	/// Loads `s[..8]` zero-extended from bytes.
	#[inline(always)]
	pub fn load_u8(s: &[u8]) -> Self {
		U8x16::load8(s).widen_lo()
	}

	/// Loads `s[..8]`, reinterpreting each `u16` as the `i16` with the same
	/// bits.
	#[inline(always)]
	pub fn load_u16(s: &[u16]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 16 readable bytes; `loadu` has no alignment need.
		Self(unsafe { _mm_loadu_si128(s.as_ptr().cast()) })
	}

	/// Stores into `d[..8]`, each lane's bits as a `u16`.
	#[inline(always)]
	pub fn store_u16(self, d: &mut [u16]) {
		let d = &mut d[..8];
		// SAFETY: `d` is 16 writable bytes; `storeu` has no alignment need.
		unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), self.0) }
	}

	#[inline(always)]
	pub fn to_array(self) -> [i16; 8] {
		let mut a = [0i16; 8];
		self.store(&mut a);
		a
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(sse!(_mm_add_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(sse!(_mm_sub_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_add(self, o: Self) -> Self {
		Self(sse!(_mm_adds_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_sub(self, o: Self) -> Self {
		Self(sse!(_mm_subs_epi16(self.0, o.0)))
	}

	/// Low 16 bits of the product.
	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		Self(sse!(_mm_mullo_epi16(self.0, o.0)))
	}

	/// Arithmetic shift right (`n < 16`).
	#[inline(always)]
	pub fn shr(self, n: u32) -> Self {
		Self(sse!(_mm_sra_epi16(self.0, count(n))))
	}

	/// Logical shift right (`n < 16`).
	#[inline(always)]
	pub fn shr_logical(self, n: u32) -> Self {
		Self(sse!(_mm_srl_epi16(self.0, count(n))))
	}

	/// Shift left (`n < 16`).
	#[inline(always)]
	pub fn shl(self, n: u32) -> Self {
		Self(sse!(_mm_sll_epi16(self.0, count(n))))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		Self(sse!(_mm_min_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		Self(sse!(_mm_max_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn clamp(self, lo: Self, hi: Self) -> Self {
		self.max(lo).min(hi)
	}

	/// Wrapping absolute value (`i16::MIN` stays `i16::MIN`).
	#[inline(always)]
	pub fn abs(self) -> Self {
		Self(sse!(_mm_max_epi16(self.0, _mm_sub_epi16(zero(), self.0))))
	}

	/// All-ones lanes where `self > o`.
	#[inline(always)]
	pub fn gt(self, o: Self) -> Self {
		Self(sse!(_mm_cmpgt_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn lt(self, o: Self) -> Self {
		Self(sse!(_mm_cmplt_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn eq(self, o: Self) -> Self {
		Self(sse!(_mm_cmpeq_epi16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn and(self, o: Self) -> Self {
		Self(sse!(_mm_and_si128(self.0, o.0)))
	}

	#[inline(always)]
	pub fn or(self, o: Self) -> Self {
		Self(sse!(_mm_or_si128(self.0, o.0)))
	}

	#[inline(always)]
	pub fn xor(self, o: Self) -> Self {
		Self(sse!(_mm_xor_si128(self.0, o.0)))
	}

	/// `self & !o`.
	#[inline(always)]
	pub fn and_not(self, o: Self) -> Self {
		Self(sse!(_mm_andnot_si128(o.0, self.0)))
	}

	#[inline(always)]
	pub fn not(self) -> Self {
		Self(sse!(_mm_xor_si128(self.0, _mm_set1_epi32(-1))))
	}

	/// `a` where the mask lane is set, else `b`.
	#[inline(always)]
	pub fn select(self, a: Self, b: Self) -> Self {
		Self(blend(self.0, a.0, b.0))
	}

	/// Whether any bit of any lane is set.
	#[inline(always)]
	pub fn any(self) -> bool {
		sse!(_mm_movemask_epi8(_mm_cmpeq_epi8(self.0, zero()))) != 0xFFFF
	}

	/// Lanes 0..4 sign-extended.
	#[inline(always)]
	pub fn widen_lo(self) -> I32x4 {
		I32x4(sse!(_mm_srai_epi32::<16>(_mm_unpacklo_epi16(self.0, self.0))))
	}

	/// Lanes 4..8 sign-extended.
	#[inline(always)]
	pub fn widen_hi(self) -> I32x4 {
		I32x4(sse!(_mm_srai_epi32::<16>(_mm_unpackhi_epi16(self.0, self.0))))
	}

	/// `a[2i] * b[2i] + a[2i+1] * b[2i+1]` in 32 bits (wrapping).
	#[inline(always)]
	pub fn madd(self, o: Self) -> I32x4 {
		I32x4(sse!(_mm_madd_epi16(self.0, o.0)))
	}

	/// Saturates `self` (lanes 0..8) and `hi` (lanes 8..16) to `u8`.
	#[inline(always)]
	pub fn pack_u8(self, hi: Self) -> U8x16 {
		U8x16(sse!(_mm_packus_epi16(self.0, hi.0)))
	}

	/// `[a0, b0, a1, b1, a2, b2, a3, b3]`.
	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(sse!(_mm_unpacklo_epi16(self.0, o.0)))
	}

	/// `[a4, b4, ..., a7, b7]`.
	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(sse!(_mm_unpackhi_epi16(self.0, o.0)))
	}

	/// [`Self::zip_lo`] on 32-bit lane pairs.
	#[inline(always)]
	pub fn zip_lo32(self, o: Self) -> Self {
		Self(sse!(_mm_unpacklo_epi32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn zip_hi32(self, o: Self) -> Self {
		Self(sse!(_mm_unpackhi_epi32(self.0, o.0)))
	}

	/// Low 64 bits of `self`, then low 64 bits of `o`.
	#[inline(always)]
	pub fn zip_lo64(self, o: Self) -> Self {
		Self(sse!(_mm_unpacklo_epi64(self.0, o.0)))
	}

	#[inline(always)]
	pub fn zip_hi64(self, o: Self) -> Self {
		Self(sse!(_mm_unpackhi_epi64(self.0, o.0)))
	}
}

// ---------------------------------------------------------------------------

/// 4 lanes of `i32`.
#[derive(Clone, Copy)]
pub struct I32x4(__m128i);

impl I32x4 {
	#[inline(always)]
	pub fn splat(v: i32) -> Self {
		Self(sse!(_mm_set1_epi32(v)))
	}

	#[inline(always)]
	pub fn from_array(a: [i32; 4]) -> Self {
		Self::load(&a)
	}

	/// Loads `s[..4]`.
	#[inline(always)]
	pub fn load(s: &[i32]) -> Self {
		let s = &s[..4];
		// SAFETY: `s` is 16 readable bytes; `loadu` has no alignment need.
		Self(unsafe { _mm_loadu_si128(s.as_ptr().cast()) })
	}

	/// Stores into `d[..4]`.
	#[inline(always)]
	pub fn store(self, d: &mut [i32]) {
		let d = &mut d[..4];
		// SAFETY: `d` is 16 writable bytes; `storeu` has no alignment need.
		unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), self.0) }
	}

	#[inline(always)]
	pub fn to_array(self) -> [i32; 4] {
		let mut a = [0i32; 4];
		self.store(&mut a);
		a
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(sse!(_mm_add_epi32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(sse!(_mm_sub_epi32(self.0, o.0)))
	}

	/// Low 32 bits of the product (SSE2 has no `pmulld`: two 32x32->64
	/// multiplies of the even and odd lanes, low halves recombined).
	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		sse!({
			let even = _mm_mul_epu32(self.0, o.0);
			let odd = _mm_mul_epu32(_mm_srli_epi64::<32>(self.0), _mm_srli_epi64::<32>(o.0));
			Self(_mm_unpacklo_epi32(
				_mm_shuffle_epi32::<0b00_00_10_00>(even),
				_mm_shuffle_epi32::<0b00_00_10_00>(odd),
			))
		})
	}

	/// Arithmetic shift right (`n < 32`).
	#[inline(always)]
	pub fn shr(self, n: u32) -> Self {
		Self(sse!(_mm_sra_epi32(self.0, count(n))))
	}

	/// Shift left (`n < 32`).
	#[inline(always)]
	pub fn shl(self, n: u32) -> Self {
		Self(sse!(_mm_sll_epi32(self.0, count(n))))
	}

	#[inline(always)]
	pub fn gt(self, o: Self) -> Self {
		Self(sse!(_mm_cmpgt_epi32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn lt(self, o: Self) -> Self {
		Self(sse!(_mm_cmplt_epi32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn eq(self, o: Self) -> Self {
		Self(sse!(_mm_cmpeq_epi32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn and(self, o: Self) -> Self {
		Self(sse!(_mm_and_si128(self.0, o.0)))
	}

	#[inline(always)]
	pub fn or(self, o: Self) -> Self {
		Self(sse!(_mm_or_si128(self.0, o.0)))
	}

	#[inline(always)]
	pub fn select(self, a: Self, b: Self) -> Self {
		Self(blend(self.0, a.0, b.0))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		self.gt(o).select(o, self)
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		self.gt(o).select(self, o)
	}

	#[inline(always)]
	pub fn clamp(self, lo: Self, hi: Self) -> Self {
		self.max(lo).min(hi)
	}

	#[inline(always)]
	pub fn any(self) -> bool {
		sse!(_mm_movemask_epi8(_mm_cmpeq_epi8(self.0, zero()))) != 0xFFFF
	}

	/// Saturates `self` (lanes 0..4) and `hi` (lanes 4..8) to `i16`.
	#[inline(always)]
	pub fn pack_i16(self, hi: Self) -> I16x8 {
		I16x8(sse!(_mm_packs_epi32(self.0, hi.0)))
	}

	/// `[a0, b0, a1, b1]`.
	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(sse!(_mm_unpacklo_epi32(self.0, o.0)))
	}

	/// `[a2, b2, a3, b3]`.
	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(sse!(_mm_unpackhi_epi32(self.0, o.0)))
	}
	/// `[a0, a1, b0, b1]`.
	#[inline(always)]
	pub fn zip_lo64(self, o: Self) -> Self {
		Self(sse!(_mm_unpacklo_epi64(self.0, o.0)))
	}

	/// `[a2, a3, b2, b3]`.
	#[inline(always)]
	pub fn zip_hi64(self, o: Self) -> Self {
		Self(sse!(_mm_unpackhi_epi64(self.0, o.0)))
	}
}

// ---------------------------------------------------------------------------

/// 4 lanes of `f32`.
#[derive(Clone, Copy)]
pub struct F32x4(__m128);

impl F32x4 {
	#[inline(always)]
	pub fn splat(v: f32) -> Self {
		Self(sse!(_mm_set1_ps(v)))
	}

	#[inline(always)]
	pub fn from_array(a: [f32; 4]) -> Self {
		Self::load(&a)
	}

	/// Loads `s[..4]`.
	#[inline(always)]
	pub fn load(s: &[f32]) -> Self {
		let s = &s[..4];
		// SAFETY: `s` is 16 readable bytes; `loadu` has no alignment need.
		Self(unsafe { _mm_loadu_ps(s.as_ptr()) })
	}

	/// Stores into `d[..4]`.
	#[inline(always)]
	pub fn store(self, d: &mut [f32]) {
		let d = &mut d[..4];
		// SAFETY: `d` is 16 writable bytes; `storeu` has no alignment need.
		unsafe { _mm_storeu_ps(d.as_mut_ptr(), self.0) }
	}

	#[inline(always)]
	pub fn to_array(self) -> [f32; 4] {
		let mut a = [0f32; 4];
		self.store(&mut a);
		a
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(sse!(_mm_add_ps(self.0, o.0)))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(sse!(_mm_sub_ps(self.0, o.0)))
	}

	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		Self(sse!(_mm_mul_ps(self.0, o.0)))
	}

	/// Flips the sign bit (exactly `-x`, including zeros and NaNs).
	#[inline(always)]
	pub fn neg(self) -> Self {
		Self(sse!(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))))
	}

	/// `[a3, a2, a1, a0]`.
	#[inline(always)]
	pub fn reverse(self) -> Self {
		Self(sse!(_mm_shuffle_ps::<0b00_01_10_11>(self.0, self.0)))
	}

	/// `[a0, b0, a1, b1]`.
	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(sse!(_mm_unpacklo_ps(self.0, o.0)))
	}

	/// `[a2, b2, a3, b3]`.
	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(sse!(_mm_unpackhi_ps(self.0, o.0)))
	}

	/// `([a0, a2, b0, b2], [a1, a3, b1, b3])`.
	#[inline(always)]
	pub fn unzip(self, o: Self) -> (Self, Self) {
		sse!((
			Self(_mm_shuffle_ps::<0b10_00_10_00>(self.0, o.0)),
			Self(_mm_shuffle_ps::<0b11_01_11_01>(self.0, o.0))
		))
	}
}

macro_rules! debug_via_array {
	($($t:ty),*) => {$(
		impl std::fmt::Debug for $t {
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				self.to_array().fmt(f)
			}
		}
	)*};
}
debug_via_array!(U8x16, I16x8, I32x4, F32x4);
