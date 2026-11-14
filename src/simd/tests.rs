//! Every method of the native backend checked lane-for-lane against the
//! plain-array model in `scalar.rs`, on random and edge-case inputs.

use super::scalar as s;
use super::*;

/// xorshift64*; deterministic so failures reproduce.
struct Rng(u64);

impl Rng {
	fn next(&mut self) -> u64 {
		self.0 ^= self.0 >> 12;
		self.0 ^= self.0 << 25;
		self.0 ^= self.0 >> 27;
		self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
	}

	/// Mostly uniform, with a bias towards the type's extremes and small
	/// values, where SIMD emulations go wrong.
	fn pick<T: Copy>(&mut self, edges: &[T], uniform: impl Fn(u64) -> T) -> T {
		let r = self.next();
		if r.is_multiple_of(4) { edges[(r >> 8) as usize % edges.len()] } else { uniform(r >> 16) }
	}

	fn u8s(&mut self) -> [u8; 16] {
		std::array::from_fn(|_| self.pick(&[0, 1, 127, 128, 254, 255], |r| r as u8))
	}

	fn i16s(&mut self) -> [i16; 8] {
		std::array::from_fn(|_| self.pick(&[0, 1, -1, 255, -256, i16::MIN, i16::MAX], |r| r as i16))
	}

	fn i32s(&mut self) -> [i32; 4] {
		std::array::from_fn(|_| self.pick(&[0, 1, -1, 65535, i32::MIN, i32::MAX], |r| r as i32))
	}

	fn f32s(&mut self) -> [f32; 4] {
		std::array::from_fn(|_| {
			self.pick(&[0.0, -0.0, 1.0, -1.5, f32::MAX, f32::MIN_POSITIVE, 1e-40], |r| {
				(r as i32 as f32) / 65536.0
			})
		})
	}
}

const ROUNDS: usize = 2000;

fn bits(a: [f32; 4]) -> [u32; 4] {
	a.map(f32::to_bits)
}

#[test]
fn u8x16_matches_model() {
	let mut rng = Rng(0x1234_5678);
	for _ in 0..ROUNDS {
		let (a, b) = (rng.u8s(), rng.u8s());
		let (na, nb) = (U8x16::load(&a), U8x16::load(&b));
		let (sa, sb) = (s::U8x16(a), s::U8x16(b));
		macro_rules! same {
			($($m:ident),*) => {$(
				assert_eq!(na.$m(nb).to_array(), sa.$m(sb).to_array(), "{} {a:?} {b:?}", stringify!($m));
			)*};
		}
		same!(avg, min, max, abs_diff, saturating_add, saturating_sub);
		assert_eq!(na.widen_lo().to_array(), sa.widen_lo().to_array());
		assert_eq!(na.widen_hi().to_array(), sa.widen_hi().to_array());
		assert_eq!(U8x16::load8(&a).to_array(), s::U8x16::load8(&a).to_array());
		assert_eq!(U8x16::load4(&a).to_array(), s::U8x16::load4(&a).to_array());
		let mut d = b;
		na.store8(&mut d);
		let mut e = b;
		sa.store8(&mut e);
		assert_eq!(d, e);
		na.store4(&mut d);
		sa.store4(&mut e);
		assert_eq!(d, e);
		assert_eq!(U8x16::splat(a[3]).to_array(), [a[3]; 16]);
	}
}

#[test]
fn i16x8_matches_model() {
	let mut rng = Rng(0x9E37_79B9);
	for _ in 0..ROUNDS {
		let (a, b, m) = (rng.i16s(), rng.i16s(), rng.i16s());
		let (na, nb) = (I16x8::load(&a), I16x8::load(&b));
		let (sa, sb) = (s::I16x8(a), s::I16x8(b));
		macro_rules! same {
			($($m:ident),*) => {$(
				assert_eq!(na.$m(nb).to_array(), sa.$m(sb).to_array(), "{} {a:?} {b:?}", stringify!($m));
			)*};
		}
		same!(add, sub, saturating_add, saturating_sub, mul, min, max, gt, lt, eq, and, or, xor);
		same!(and_not, zip_lo, zip_hi, zip_lo32, zip_hi32, zip_lo64, zip_hi64);
		assert_eq!(na.madd(nb).to_array(), sa.madd(sb).to_array(), "madd {a:?} {b:?}");
		assert_eq!(na.pack_u8(nb).to_array(), sa.pack_u8(sb).to_array());
		assert_eq!(na.abs().to_array(), sa.abs().to_array());
		assert_eq!(na.not().to_array(), sa.not().to_array());
		assert_eq!(na.widen_lo().to_array(), sa.widen_lo().to_array());
		assert_eq!(na.widen_hi().to_array(), sa.widen_hi().to_array());
		assert_eq!(na.any(), sa.any());
		let (lo, hi) = (na.min(nb), na.max(nb));
		let (slo, shi) = (sa.min(sb), sa.max(sb));
		let nm = I16x8::load(&m);
		assert_eq!(nm.clamp(lo, hi).to_array(), s::I16x8(m).clamp(slo, shi).to_array());
		let mask = nm.gt(na);
		let smask = s::I16x8(m).gt(sa);
		assert_eq!(mask.select(na, nb).to_array(), smask.select(sa, sb).to_array());
		for n in [0, 1, 2, 3, 5, 7, 10, 15] {
			assert_eq!(na.shr(n).to_array(), sa.shr(n).to_array(), "shr {n}");
			assert_eq!(na.shl(n).to_array(), sa.shl(n).to_array(), "shl {n}");
			assert_eq!(na.shr_logical(n).to_array(), sa.shr_logical(n).to_array(), "shrl {n}");
		}
		let bytes: [u8; 8] = std::array::from_fn(|i| a[i] as u8);
		assert_eq!(I16x8::load_u8(&bytes).to_array(), s::I16x8::load_u8(&bytes).to_array());
		let words: [u16; 8] = std::array::from_fn(|i| a[i] as u16);
		assert_eq!(I16x8::load_u16(&words).to_array(), s::I16x8::load_u16(&words).to_array());
		let (mut w1, mut w2) = ([0u16; 8], [0u16; 8]);
		na.store_u16(&mut w1);
		sa.store_u16(&mut w2);
		assert_eq!(w1, w2);
	}
	assert!(!I16x8::splat(0).any());
	let mut one = [0i16; 8];
	one[7] = 1;
	assert!(I16x8::load(&one).any());
}

#[test]
fn i32x4_matches_model() {
	let mut rng = Rng(0xDEAD_BEEF);
	for _ in 0..ROUNDS {
		let (a, b, m) = (rng.i32s(), rng.i32s(), rng.i32s());
		let (na, nb) = (I32x4::load(&a), I32x4::load(&b));
		let (sa, sb) = (s::I32x4(a), s::I32x4(b));
		macro_rules! same {
			($($m:ident),*) => {$(
				assert_eq!(na.$m(nb).to_array(), sa.$m(sb).to_array(), "{} {a:?} {b:?}", stringify!($m));
			)*};
		}
		same!(add, sub, mul, gt, lt, eq, and, or, min, max, zip_lo, zip_hi, zip_lo64, zip_hi64);
		assert_eq!(na.pack_i16(nb).to_array(), sa.pack_i16(sb).to_array());
		assert_eq!(na.any(), sa.any());
		let nm = I32x4::load(&m);
		let sm = s::I32x4(m);
		assert_eq!(
			nm.clamp(na.min(nb), na.max(nb)).to_array(),
			sm.clamp(sa.min(sb), sa.max(sb)).to_array()
		);
		assert_eq!(nm.lt(na).select(na, nb).to_array(), sm.lt(sa).select(sa, sb).to_array());
		for n in [0, 1, 6, 13, 14, 31] {
			assert_eq!(na.shr(n).to_array(), sa.shr(n).to_array(), "shr {n}");
			assert_eq!(na.shl(n).to_array(), sa.shl(n).to_array(), "shl {n}");
		}
	}
}

#[test]
fn f32x4_matches_model_bit_for_bit() {
	let mut rng = Rng(0x0BAD_F00D);
	for _ in 0..ROUNDS {
		let (a, b) = (rng.f32s(), rng.f32s());
		let (na, nb) = (F32x4::load(&a), F32x4::load(&b));
		let (sa, sb) = (s::F32x4(a), s::F32x4(b));
		macro_rules! same {
			($($m:ident),*) => {$(
				assert_eq!(bits(na.$m(nb).to_array()), bits(sa.$m(sb).to_array()), "{}", stringify!($m));
			)*};
		}
		same!(add, sub, mul, zip_lo, zip_hi);
		assert_eq!(bits(na.neg().to_array()), bits(sa.neg().to_array()));
		assert_eq!(bits(na.reverse().to_array()), bits(sa.reverse().to_array()));
		let (e, o) = na.unzip(nb);
		let (se, so) = sa.unzip(sb);
		assert_eq!(bits(e.to_array()), bits(se.to_array()));
		assert_eq!(bits(o.to_array()), bits(so.to_array()));
	}
}

#[test]
fn transpose_is_a_transpose() {
	let rows: [I16x8; 8] =
		std::array::from_fn(|i| I16x8::from_array(std::array::from_fn(|j| (i * 8 + j) as i16)));
	let t = transpose8x8(rows);
	for (j, row) in t.iter().enumerate() {
		assert_eq!(row.to_array(), std::array::from_fn(|i| (i * 8 + j) as i16));
	}
}

#[test]
fn transpose4x4_is_a_transpose() {
	let rows: [I32x4; 4] =
		std::array::from_fn(|i| I32x4::from_array(std::array::from_fn(|j| (i * 4 + j) as i32)));
	let t = transpose4x4(rows);
	for (j, row) in t.iter().enumerate() {
		assert_eq!(row.to_array(), std::array::from_fn(|i| (i * 4 + j) as i32));
	}
}

#[test]
fn level_never_exceeds_hardware() {
	let hw = detected();
	assert_eq!(set_level(Level::Avx2), hw);
	assert_eq!(set_level(Level::Scalar), Level::Scalar);
	assert!(!enabled());
	assert_eq!(set_level(hw), hw);
	assert_eq!(Level::parse("NEON"), Some(Level::Simd128));
	assert_eq!(Level::parse("scalar"), Some(Level::Scalar));
	assert_eq!(Level::parse("bogus"), None);
}

#[test]
fn short_slices_panic_instead_of_reading_out_of_bounds() {
	assert!(std::panic::catch_unwind(|| U8x16::load(&[0u8; 15])).is_err());
	assert!(std::panic::catch_unwind(|| U8x16::load8(&[0u8; 7])).is_err());
	assert!(std::panic::catch_unwind(|| I16x8::load(&[0i16; 7])).is_err());
	assert!(std::panic::catch_unwind(|| F32x4::load(&[0f32; 3])).is_err());
	assert!(
		std::panic::catch_unwind(|| {
			let mut d = [0u8; 7];
			U8x16::splat(1).store8(&mut d);
		})
		.is_err()
	);
}
