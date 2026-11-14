//! NEON backend (aarch64 baseline). See the module docs of [`super`] for
//! why each `unsafe` here is sound: `neon!` wraps value intrinsics, which
//! are only `unsafe` because they need NEON, and this file is compiled only
//! when NEON is a target feature; loads and stores re-slice to the exact
//! length they access before using the pointer.

// Methods are named `add`, `mul`, `shr`, ... rather than implementing
// `std::ops`: these wrap (or saturate, where named so), and `a + b` would
// read as Rust's overflow-checked arithmetic.
#![allow(clippy::should_implement_trait)]

use core::arch::aarch64::*;

/// Calls NEON value intrinsics. SAFETY: this module is compiled only with
/// `target_feature = "neon"` (see `mod.rs`), the intrinsics' one
/// requirement.
macro_rules! neon {
	($e:expr) => {
		unsafe { $e }
	};
}

// ---------------------------------------------------------------------------

/// 16 lanes of `u8`.
#[derive(Clone, Copy)]
pub struct U8x16(uint8x16_t);

impl U8x16 {
	#[inline(always)]
	pub fn splat(v: u8) -> Self {
		Self(neon!(vdupq_n_u8(v)))
	}

	/// Loads `s[..16]`.
	#[inline(always)]
	pub fn load(s: &[u8]) -> Self {
		let s = &s[..16];
		// SAFETY: `s` is 16 readable bytes; `vld1q` has no alignment need.
		Self(unsafe { vld1q_u8(s.as_ptr()) })
	}

	/// Stores into `d[..16]`.
	#[inline(always)]
	pub fn store(self, d: &mut [u8]) {
		let d = &mut d[..16];
		// SAFETY: `d` is 16 writable bytes; `vst1q` has no alignment need.
		unsafe { vst1q_u8(d.as_mut_ptr(), self.0) }
	}

	/// Loads `s[..8]` into the low half; the high half is zero.
	#[inline(always)]
	pub fn load8(s: &[u8]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 8 readable bytes, exactly what `vld1_u8` reads.
		let lo = unsafe { vld1_u8(s.as_ptr()) };
		Self(neon!(vcombine_u8(lo, vdup_n_u8(0))))
	}

	/// Stores the low half into `d[..8]`.
	#[inline(always)]
	pub fn store8(self, d: &mut [u8]) {
		let d = &mut d[..8];
		// SAFETY: `d` is 8 writable bytes, exactly what `vst1_u8` writes.
		unsafe { vst1_u8(d.as_mut_ptr(), vget_low_u8(self.0)) }
	}

	/// Loads `s[..4]` into lanes 0..4; the rest are zero.
	#[inline(always)]
	pub fn load4(s: &[u8]) -> Self {
		let b: [u8; 4] = std::array::from_fn(|i| s[i]);
		let v = u32::from_le_bytes(b);
		Self(neon!(vreinterpretq_u8_u32(vsetq_lane_u32::<0>(v, vdupq_n_u32(0)))))
	}

	/// Stores lanes 0..4 into `d[..4]`.
	#[inline(always)]
	pub fn store4(self, d: &mut [u8]) {
		let v = neon!(vgetq_lane_u32::<0>(vreinterpretq_u32_u8(self.0)));
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
		Self(neon!(vrhaddq_u8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		Self(neon!(vminq_u8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		Self(neon!(vmaxq_u8(self.0, o.0)))
	}

	/// `|a - b|`.
	#[inline(always)]
	pub fn abs_diff(self, o: Self) -> Self {
		Self(neon!(vabdq_u8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_add(self, o: Self) -> Self {
		Self(neon!(vqaddq_u8(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_sub(self, o: Self) -> Self {
		Self(neon!(vqsubq_u8(self.0, o.0)))
	}

	/// Lanes 0..8 zero-extended.
	#[inline(always)]
	pub fn widen_lo(self) -> I16x8 {
		I16x8(neon!(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(self.0)))))
	}

	/// Lanes 8..16 zero-extended.
	#[inline(always)]
	pub fn widen_hi(self) -> I16x8 {
		I16x8(neon!(vreinterpretq_s16_u16(vmovl_high_u8(self.0))))
	}
}

// ---------------------------------------------------------------------------

/// 8 lanes of `i16`.
#[derive(Clone, Copy)]
pub struct I16x8(int16x8_t);

impl I16x8 {
	#[inline(always)]
	pub fn splat(v: i16) -> Self {
		Self(neon!(vdupq_n_s16(v)))
	}

	#[inline(always)]
	pub fn from_array(a: [i16; 8]) -> Self {
		Self::load(&a)
	}

	/// Loads `s[..8]`.
	#[inline(always)]
	pub fn load(s: &[i16]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 8 readable lanes; `vld1q` has no alignment need.
		Self(unsafe { vld1q_s16(s.as_ptr()) })
	}

	/// Stores into `d[..8]`.
	#[inline(always)]
	pub fn store(self, d: &mut [i16]) {
		let d = &mut d[..8];
		// SAFETY: `d` is 8 writable lanes; `vst1q` has no alignment need.
		unsafe { vst1q_s16(d.as_mut_ptr(), self.0) }
	}

	/// Loads `s[..8]` zero-extended from bytes.
	#[inline(always)]
	pub fn load_u8(s: &[u8]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 8 readable bytes, exactly what `vld1_u8` reads.
		let b = unsafe { vld1_u8(s.as_ptr()) };
		Self(neon!(vreinterpretq_s16_u16(vmovl_u8(b))))
	}

	/// Loads `s[..8]`, reinterpreting each `u16` as the `i16` with the same
	/// bits.
	#[inline(always)]
	pub fn load_u16(s: &[u16]) -> Self {
		let s = &s[..8];
		// SAFETY: `s` is 8 readable lanes; `vld1q` has no alignment need.
		Self(unsafe { vreinterpretq_s16_u16(vld1q_u16(s.as_ptr())) })
	}

	/// Stores into `d[..8]`, each lane's bits as a `u16`.
	#[inline(always)]
	pub fn store_u16(self, d: &mut [u16]) {
		let d = &mut d[..8];
		// SAFETY: `d` is 8 writable lanes; `vst1q` has no alignment need.
		unsafe { vst1q_u16(d.as_mut_ptr(), vreinterpretq_u16_s16(self.0)) }
	}

	#[inline(always)]
	pub fn to_array(self) -> [i16; 8] {
		let mut a = [0i16; 8];
		self.store(&mut a);
		a
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(neon!(vaddq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(neon!(vsubq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_add(self, o: Self) -> Self {
		Self(neon!(vqaddq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn saturating_sub(self, o: Self) -> Self {
		Self(neon!(vqsubq_s16(self.0, o.0)))
	}

	/// Low 16 bits of the product.
	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		Self(neon!(vmulq_s16(self.0, o.0)))
	}

	/// Arithmetic shift right (`n < 16`).
	#[inline(always)]
	pub fn shr(self, n: u32) -> Self {
		Self(neon!(vshlq_s16(self.0, vdupq_n_s16(-(n as i16)))))
	}

	/// Logical shift right (`n < 16`).
	#[inline(always)]
	pub fn shr_logical(self, n: u32) -> Self {
		Self(neon!(vreinterpretq_s16_u16(vshlq_u16(
			vreinterpretq_u16_s16(self.0),
			vdupq_n_s16(-(n as i16))
		))))
	}

	/// Shift left (`n < 16`).
	#[inline(always)]
	pub fn shl(self, n: u32) -> Self {
		Self(neon!(vshlq_s16(self.0, vdupq_n_s16(n as i16))))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		Self(neon!(vminq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		Self(neon!(vmaxq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn clamp(self, lo: Self, hi: Self) -> Self {
		self.max(lo).min(hi)
	}

	/// Wrapping absolute value (`i16::MIN` stays `i16::MIN`).
	#[inline(always)]
	pub fn abs(self) -> Self {
		Self(neon!(vabsq_s16(self.0)))
	}

	/// All-ones lanes where `self > o`.
	#[inline(always)]
	pub fn gt(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_u16(vcgtq_s16(self.0, o.0))))
	}

	#[inline(always)]
	pub fn lt(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_u16(vcltq_s16(self.0, o.0))))
	}

	#[inline(always)]
	pub fn eq(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_u16(vceqq_s16(self.0, o.0))))
	}

	#[inline(always)]
	pub fn and(self, o: Self) -> Self {
		Self(neon!(vandq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn or(self, o: Self) -> Self {
		Self(neon!(vorrq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn xor(self, o: Self) -> Self {
		Self(neon!(veorq_s16(self.0, o.0)))
	}

	/// `self & !o`.
	#[inline(always)]
	pub fn and_not(self, o: Self) -> Self {
		Self(neon!(vbicq_s16(self.0, o.0)))
	}

	#[inline(always)]
	pub fn not(self) -> Self {
		Self(neon!(vmvnq_s16(self.0)))
	}

	/// `a` where the mask lane is set, else `b`.
	#[inline(always)]
	pub fn select(self, a: Self, b: Self) -> Self {
		Self(neon!(vbslq_s16(vreinterpretq_u16_s16(self.0), a.0, b.0)))
	}

	/// Whether any bit of any lane is set.
	#[inline(always)]
	pub fn any(self) -> bool {
		neon!(vmaxvq_u16(vreinterpretq_u16_s16(self.0))) != 0
	}

	/// Lanes 0..4 sign-extended.
	#[inline(always)]
	pub fn widen_lo(self) -> I32x4 {
		I32x4(neon!(vmovl_s16(vget_low_s16(self.0))))
	}

	/// Lanes 4..8 sign-extended.
	#[inline(always)]
	pub fn widen_hi(self) -> I32x4 {
		I32x4(neon!(vmovl_high_s16(self.0)))
	}

	/// `a[2i] * b[2i] + a[2i+1] * b[2i+1]` in 32 bits (wrapping).
	#[inline(always)]
	pub fn madd(self, o: Self) -> I32x4 {
		neon!({
			let lo = vmull_s16(vget_low_s16(self.0), vget_low_s16(o.0));
			let hi = vmull_high_s16(self.0, o.0);
			I32x4(vpaddq_s32(lo, hi))
		})
	}

	/// Saturates `self` (lanes 0..8) and `hi` (lanes 8..16) to `u8`.
	#[inline(always)]
	pub fn pack_u8(self, hi: Self) -> U8x16 {
		U8x16(neon!(vcombine_u8(vqmovun_s16(self.0), vqmovun_s16(hi.0))))
	}

	/// `[a0, b0, a1, b1, a2, b2, a3, b3]`.
	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(neon!(vzip1q_s16(self.0, o.0)))
	}

	/// `[a4, b4, ..., a7, b7]`.
	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(neon!(vzip2q_s16(self.0, o.0)))
	}

	/// [`Self::zip_lo`] on 32-bit lane pairs.
	#[inline(always)]
	pub fn zip_lo32(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_s32(vzip1q_s32(
			vreinterpretq_s32_s16(self.0),
			vreinterpretq_s32_s16(o.0)
		))))
	}

	#[inline(always)]
	pub fn zip_hi32(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_s32(vzip2q_s32(
			vreinterpretq_s32_s16(self.0),
			vreinterpretq_s32_s16(o.0)
		))))
	}

	/// Low 64 bits of `self`, then low 64 bits of `o`.
	#[inline(always)]
	pub fn zip_lo64(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_s64(vzip1q_s64(
			vreinterpretq_s64_s16(self.0),
			vreinterpretq_s64_s16(o.0)
		))))
	}

	#[inline(always)]
	pub fn zip_hi64(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s16_s64(vzip2q_s64(
			vreinterpretq_s64_s16(self.0),
			vreinterpretq_s64_s16(o.0)
		))))
	}
}

// ---------------------------------------------------------------------------

/// 4 lanes of `i32`.
#[derive(Clone, Copy)]
pub struct I32x4(int32x4_t);

impl I32x4 {
	#[inline(always)]
	pub fn splat(v: i32) -> Self {
		Self(neon!(vdupq_n_s32(v)))
	}

	#[inline(always)]
	pub fn from_array(a: [i32; 4]) -> Self {
		Self::load(&a)
	}

	/// Loads `s[..4]`.
	#[inline(always)]
	pub fn load(s: &[i32]) -> Self {
		let s = &s[..4];
		// SAFETY: `s` is 4 readable lanes; `vld1q` has no alignment need.
		Self(unsafe { vld1q_s32(s.as_ptr()) })
	}

	/// Stores into `d[..4]`.
	#[inline(always)]
	pub fn store(self, d: &mut [i32]) {
		let d = &mut d[..4];
		// SAFETY: `d` is 4 writable lanes; `vst1q` has no alignment need.
		unsafe { vst1q_s32(d.as_mut_ptr(), self.0) }
	}

	#[inline(always)]
	pub fn to_array(self) -> [i32; 4] {
		let mut a = [0i32; 4];
		self.store(&mut a);
		a
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(neon!(vaddq_s32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(neon!(vsubq_s32(self.0, o.0)))
	}

	/// Low 32 bits of the product.
	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		Self(neon!(vmulq_s32(self.0, o.0)))
	}

	/// Arithmetic shift right (`n < 32`).
	#[inline(always)]
	pub fn shr(self, n: u32) -> Self {
		Self(neon!(vshlq_s32(self.0, vdupq_n_s32(-(n as i32)))))
	}

	/// Shift left (`n < 32`).
	#[inline(always)]
	pub fn shl(self, n: u32) -> Self {
		Self(neon!(vshlq_s32(self.0, vdupq_n_s32(n as i32))))
	}

	#[inline(always)]
	pub fn gt(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s32_u32(vcgtq_s32(self.0, o.0))))
	}

	#[inline(always)]
	pub fn lt(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s32_u32(vcltq_s32(self.0, o.0))))
	}

	#[inline(always)]
	pub fn eq(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s32_u32(vceqq_s32(self.0, o.0))))
	}

	#[inline(always)]
	pub fn and(self, o: Self) -> Self {
		Self(neon!(vandq_s32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn or(self, o: Self) -> Self {
		Self(neon!(vorrq_s32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn select(self, a: Self, b: Self) -> Self {
		Self(neon!(vbslq_s32(vreinterpretq_u32_s32(self.0), a.0, b.0)))
	}

	#[inline(always)]
	pub fn min(self, o: Self) -> Self {
		Self(neon!(vminq_s32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn max(self, o: Self) -> Self {
		Self(neon!(vmaxq_s32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn clamp(self, lo: Self, hi: Self) -> Self {
		self.max(lo).min(hi)
	}

	#[inline(always)]
	pub fn any(self) -> bool {
		neon!(vmaxvq_u32(vreinterpretq_u32_s32(self.0))) != 0
	}

	/// Saturates `self` (lanes 0..4) and `hi` (lanes 4..8) to `i16`.
	#[inline(always)]
	pub fn pack_i16(self, hi: Self) -> I16x8 {
		I16x8(neon!(vcombine_s16(vqmovn_s32(self.0), vqmovn_s32(hi.0))))
	}

	/// `[a0, b0, a1, b1]`.
	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(neon!(vzip1q_s32(self.0, o.0)))
	}

	/// `[a2, b2, a3, b3]`.
	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(neon!(vzip2q_s32(self.0, o.0)))
	}
	/// `[a0, a1, b0, b1]`.
	#[inline(always)]
	pub fn zip_lo64(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s32_s64(vzip1q_s64(
			vreinterpretq_s64_s32(self.0),
			vreinterpretq_s64_s32(o.0)
		))))
	}

	/// `[a2, a3, b2, b3]`.
	#[inline(always)]
	pub fn zip_hi64(self, o: Self) -> Self {
		Self(neon!(vreinterpretq_s32_s64(vzip2q_s64(
			vreinterpretq_s64_s32(self.0),
			vreinterpretq_s64_s32(o.0)
		))))
	}
}

// ---------------------------------------------------------------------------

/// 4 lanes of `f32`.
#[derive(Clone, Copy)]
pub struct F32x4(float32x4_t);

impl F32x4 {
	#[inline(always)]
	pub fn splat(v: f32) -> Self {
		Self(neon!(vdupq_n_f32(v)))
	}

	#[inline(always)]
	pub fn from_array(a: [f32; 4]) -> Self {
		Self::load(&a)
	}

	/// Loads `s[..4]`.
	#[inline(always)]
	pub fn load(s: &[f32]) -> Self {
		let s = &s[..4];
		// SAFETY: `s` is 4 readable lanes; `vld1q` has no alignment need.
		Self(unsafe { vld1q_f32(s.as_ptr()) })
	}

	/// Stores into `d[..4]`.
	#[inline(always)]
	pub fn store(self, d: &mut [f32]) {
		let d = &mut d[..4];
		// SAFETY: `d` is 4 writable lanes; `vst1q` has no alignment need.
		unsafe { vst1q_f32(d.as_mut_ptr(), self.0) }
	}

	#[inline(always)]
	pub fn to_array(self) -> [f32; 4] {
		let mut a = [0f32; 4];
		self.store(&mut a);
		a
	}

	#[inline(always)]
	pub fn add(self, o: Self) -> Self {
		Self(neon!(vaddq_f32(self.0, o.0)))
	}

	#[inline(always)]
	pub fn sub(self, o: Self) -> Self {
		Self(neon!(vsubq_f32(self.0, o.0)))
	}

	/// Plain multiply (never fused with a following add).
	#[inline(always)]
	pub fn mul(self, o: Self) -> Self {
		Self(neon!(vmulq_f32(self.0, o.0)))
	}

	/// Flips the sign bit (exactly `-x`, including zeros and NaNs).
	#[inline(always)]
	pub fn neg(self) -> Self {
		Self(neon!(vnegq_f32(self.0)))
	}

	/// `[a3, a2, a1, a0]`.
	#[inline(always)]
	pub fn reverse(self) -> Self {
		neon!({
			let r = vrev64q_f32(self.0);
			Self(vextq_f32::<2>(r, r))
		})
	}

	/// `[a0, b0, a1, b1]`.
	#[inline(always)]
	pub fn zip_lo(self, o: Self) -> Self {
		Self(neon!(vzip1q_f32(self.0, o.0)))
	}

	/// `[a2, b2, a3, b3]`.
	#[inline(always)]
	pub fn zip_hi(self, o: Self) -> Self {
		Self(neon!(vzip2q_f32(self.0, o.0)))
	}

	/// `([a0, a2, b0, b2], [a1, a3, b1, b3])`.
	#[inline(always)]
	pub fn unzip(self, o: Self) -> (Self, Self) {
		neon!((Self(vuzp1q_f32(self.0, o.0)), Self(vuzp2q_f32(self.0, o.0))))
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
