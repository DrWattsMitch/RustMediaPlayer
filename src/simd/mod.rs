//! Portable, safe SIMD for the codec hot loops.
//!
//! This module is the **only** place in RMP that contains `unsafe` code
//! (`lib.rs` denies `unsafe_code` everywhere else and allows it here). Its
//! job is to expose a handful of small vector types whose every method is a
//! safe function, so the kernels in the codecs are written once, in safe
//! Rust, and compile to SSE2 on x86_64, NEON on aarch64, and plain array
//! loops anywhere else.
//!
//! # Types
//!
//! | type      | lanes      | x86_64 (SSE2) | aarch64 (NEON) | other        |
//! |-----------|------------|---------------|----------------|--------------|
//! | [`U8x16`] | 16 x `u8`  | `__m128i`     | `uint8x16_t`   | `[u8; 16]`   |
//! | [`I16x8`] | 8 x `i16`  | `__m128i`     | `int16x8_t`    | `[i16; 8]`   |
//! | [`I32x4`] | 4 x `i32`  | `__m128i`     | `int32x4_t`    | `[i32; 4]`   |
//! | [`F32x4`] | 4 x `f32`  | `__m128`      | `float32x4_t`  | `[f32; 4]`   |
//!
//! Integer arithmetic wraps (like `wrapping_add`) unless the method says it
//! saturates, and float arithmetic is plain IEEE-754 per lane with no fused
//! multiply-add, so a kernel that keeps the scalar code's operation order
//! produces bit-identical output. Every SIMD kernel in RMP is tested
//! against its scalar original for exactly that.
//!
//! # Why the `unsafe` here is sound
//!
//! There are only two kinds of `unsafe` operation in the backends:
//!
//! 1. **Value intrinsics** (`_mm_add_epi16`, `vaddq_s16`, ...). They touch
//!    no memory; they are `unsafe` to call only because they require a CPU
//!    feature. Each backend module is compiled only when its feature is part
//!    of the compilation target (`#[cfg(target_feature = "sse2")]`, which
//!    every x86_64 CPU has, and `#[cfg(target_feature = "neon")]`, which
//!    every aarch64 CPU has), so the requirement is proven at compile time.
//! 2. **Loads and stores** through a raw pointer. Each one first re-slices
//!    its argument to exactly the number of bytes it touches (`&s[..16]`),
//!    so the bounds check runs in safe code and the pointer is valid for the
//!    whole access. All loads and stores are the unaligned variants, so
//!    alignment never matters.
//!
//! The third use is [`multiversion!`]: it compiles a kernel a second time
//! with AVX2 enabled and calls that copy only when [`level`] says AVX2 was
//! detected at run time. [`set_level`] can lower the level, never raise it
//! above what the CPU reported, so that call is always on a CPU that has
//! the feature.
//!
//! # Dispatch
//!
//! [`level`] is detected once (CPUID on x86_64) and cached. The `RMP_SIMD`
//! environment variable caps it (`scalar`, `simd`/`sse2`/`neon`, `avx2`),
//! which is how the scalar fallback is exercised on a SIMD machine; tests
//! use [`set_level`]. Kernels check [`enabled`] and fall back to the
//! original scalar code when it is `false`.

use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod x86;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub use x86::{F32x4, I16x8, I32x4, U8x16};

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
mod neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
pub use neon::{F32x4, I16x8, I32x4, U8x16};

/// Plain-array backend: the implementation on other targets, and the
/// reference model the native backends are tested against.
#[cfg(any(
	test,
	not(any(
		all(target_arch = "x86_64", target_feature = "sse2"),
		all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")
	))
))]
pub mod scalar;
#[cfg(not(any(
	all(target_arch = "x86_64", target_feature = "sse2"),
	all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")
)))]
pub use scalar::{F32x4, I16x8, I32x4, U8x16};

#[cfg(test)]
mod tests;

/// Which code path the kernels take. Ordered: a higher level includes the
/// lower ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
	/// The original scalar code.
	Scalar = 1,
	/// 128-bit vectors: SSE2 on x86_64, NEON on aarch64.
	Simd128 = 2,
	/// 128-bit kernels recompiled with AVX2 (x86_64 only, detected at run
	/// time).
	Avx2 = 3,
}

impl Level {
	fn from_u8(v: u8) -> Self {
		match v {
			3 => Level::Avx2,
			2 => Level::Simd128,
			_ => Level::Scalar,
		}
	}

	/// Parses an `RMP_SIMD` value.
	pub fn parse(s: &str) -> Option<Self> {
		match s.trim().to_ascii_lowercase().as_str() {
			"scalar" | "none" | "off" | "0" => Some(Level::Scalar),
			"simd" | "simd128" | "sse2" | "neon" | "128" => Some(Level::Simd128),
			"avx2" | "max" | "on" | "auto" => Some(Level::Avx2),
			_ => None,
		}
	}

	/// Short name, naming the instruction set on this target.
	pub fn name(self) -> &'static str {
		match self {
			Level::Scalar => "scalar",
			Level::Simd128 if cfg!(target_arch = "aarch64") => "neon",
			Level::Simd128 if cfg!(target_arch = "x86_64") => "sse2",
			Level::Simd128 => "simd128",
			Level::Avx2 => "avx2",
		}
	}
}

impl std::fmt::Display for Level {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(self.name())
	}
}

const UNSET: u8 = 0;
static LEVEL: AtomicU8 = AtomicU8::new(UNSET);

/// The best level this CPU supports.
pub fn detected() -> Level {
	#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
	{
		if std::arch::is_x86_feature_detected!("avx2") { Level::Avx2 } else { Level::Simd128 }
	}
	#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
	{
		Level::Simd128
	}
	#[cfg(not(any(
		all(target_arch = "x86_64", target_feature = "sse2"),
		all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")
	)))]
	{
		Level::Scalar
	}
}

#[cold]
fn init() -> Level {
	let hw = detected();
	let wanted = std::env::var("RMP_SIMD").ok().and_then(|v| Level::parse(&v));
	let level = wanted.map_or(hw, |w| w.min(hw));
	LEVEL.store(level as u8, Ordering::Relaxed);
	level
}

/// The level kernels run at: [`detected`], capped by `RMP_SIMD` or
/// [`set_level`].
#[inline]
pub fn level() -> Level {
	match LEVEL.load(Ordering::Relaxed) {
		UNSET => init(),
		v => Level::from_u8(v),
	}
}

/// Whether the vector kernels are in use (any level above scalar).
#[inline]
pub fn enabled() -> bool {
	level() >= Level::Simd128
}

/// Caps the level for the whole process (benchmarks, tests). The result is
/// `wanted` limited to what the CPU supports, and is returned.
pub fn set_level(wanted: Level) -> Level {
	let level = wanted.min(detected());
	LEVEL.store(level as u8, Ordering::Relaxed);
	level
}

/// Transposes an 8x8 block of 16-bit lanes (row `i` lane `j` becomes row
/// `j` lane `i`).
#[inline(always)]
pub fn transpose8x8(r: [I16x8; 8]) -> [I16x8; 8] {
	// Pairs of rows interleaved at 16, 32 and 64 bits, as in the classic
	// SSE2 sequence (`punpck{l,h}{wd,dq,qdq}`; NEON `zip1`/`zip2`).
	let a0 = r[0].zip_lo(r[1]);
	let a1 = r[0].zip_hi(r[1]);
	let a2 = r[2].zip_lo(r[3]);
	let a3 = r[2].zip_hi(r[3]);
	let a4 = r[4].zip_lo(r[5]);
	let a5 = r[4].zip_hi(r[5]);
	let a6 = r[6].zip_lo(r[7]);
	let a7 = r[6].zip_hi(r[7]);
	let b0 = a0.zip_lo32(a2);
	let b1 = a0.zip_hi32(a2);
	let b2 = a1.zip_lo32(a3);
	let b3 = a1.zip_hi32(a3);
	let b4 = a4.zip_lo32(a6);
	let b5 = a4.zip_hi32(a6);
	let b6 = a5.zip_lo32(a7);
	let b7 = a5.zip_hi32(a7);
	[
		b0.zip_lo64(b4),
		b0.zip_hi64(b4),
		b1.zip_lo64(b5),
		b1.zip_hi64(b5),
		b2.zip_lo64(b6),
		b2.zip_hi64(b6),
		b3.zip_lo64(b7),
		b3.zip_hi64(b7),
	]
}

/// Transposes a 4x4 block of 32-bit lanes.
#[inline(always)]
pub fn transpose4x4(r: [I32x4; 4]) -> [I32x4; 4] {
	let a0 = r[0].zip_lo(r[1]);
	let a1 = r[0].zip_hi(r[1]);
	let a2 = r[2].zip_lo(r[3]);
	let a3 = r[2].zip_hi(r[3]);
	[a0.zip_lo64(a2), a0.zip_hi64(a2), a1.zip_lo64(a3), a1.zip_hi64(a3)]
}

/// Defines a function whose body is compiled twice on x86_64: once for the
/// baseline (SSE2) and once with AVX2 enabled, picked at run time by
/// [`level`]. On other targets it is an ordinary function.
///
/// The body should be written with the vector types of this module; under
/// AVX2 LLVM re-encodes them with VEX three-operand forms and fuses the
/// SSE2 emulations (absolute value, 32-bit multiply, min/max) into the
/// native SSSE3/SSE4.1 instructions. Arguments must be plain identifiers.
#[macro_export]
macro_rules! multiversion {
	(
		$(#[$meta:meta])*
		$vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
	) => {
		$(#[$meta])*
		#[inline]
		#[allow(unsafe_code)]
		$vis fn $name($($arg: $ty),*) $(-> $ret)? {
			#[inline(always)]
			fn body($($arg: $ty),*) $(-> $ret)? $body

			#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
			{
				#[target_feature(enable = "avx2")]
				fn avx2($($arg: $ty),*) $(-> $ret)? {
					body($($arg),*)
				}
				if $crate::simd::level() == $crate::simd::Level::Avx2 {
					// SAFETY: `level()` only reports `Avx2` after
					// `is_x86_feature_detected!("avx2")` returned true
					// (`set_level` cannot raise it above that).
					return unsafe { avx2($($arg),*) };
				}
			}
			body($($arg),*)
		}
	};
}
