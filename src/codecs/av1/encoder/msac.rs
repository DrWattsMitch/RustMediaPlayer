//! AV1's multi-symbol arithmetic coder, encoder side (AV1 spec §8.2;
//! libaom `aom_dsp/entenc.c`), and a bit-cost model with the same
//! interface.
//!
//! CDFs are kept in rav1d's in-memory form: for an `n + 1`-symbol CDF,
//! `cdf[0..n]` hold `32768 - P(symbol <= i)` (decreasing) and `cdf[n]` the
//! adaptation counter. Every adaptive symbol updates its CDF exactly as
//! rav1d's `decode_symbol_adapt` does, so encoder and decoder stay in step.

use std::sync::LazyLock;

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

/// Something AV1 symbols can be written to.
pub trait Ec {
	/// Writes `s` (`0..=n`) with the non-adapted inverted CDF bounds of
	/// that symbol: `fl` (32768 for the first symbol), `fh` (0 for the last).
	fn put(&mut self, s: usize, n: usize, fl: u32, fh: u32);

	/// Whether writing adapts the CDFs (the cost model leaves them alone).
	const ADAPT: bool;

	/// An adaptive symbol `s` of an `n + 1`-symbol CDF.
	#[inline]
	fn symbol(&mut self, s: usize, cdf: &mut [u16], n: usize) {
		debug_assert!(s <= n);
		let fl = if s > 0 { u32::from(cdf[s - 1]) } else { 32768 };
		let fh = if s < n { u32::from(cdf[s]) } else { 0 };
		self.put(s, n, fl, fh);
		if Self::ADAPT {
			adapt(cdf, s, n);
		}
	}

	/// An adaptive boolean.
	#[inline]
	fn bool_adapt(&mut self, b: bool, cdf: &mut [u16; 2]) {
		self.symbol(usize::from(b), cdf, 1);
	}

	/// A boolean with a fixed probability: `f` is the inverted probability
	/// of `false` (rav1d `decode_bool`).
	#[inline]
	fn bool_f(&mut self, b: bool, f: u32) {
		self.put(usize::from(b), 1, if b { f } else { 32768 }, if b { 0 } else { f });
	}

	/// An equiprobable bit.
	#[inline]
	fn equi(&mut self, b: bool) {
		self.bool_f(b, 16384);
	}

	/// `n` equiprobable bits, most significant first.
	fn bits(&mut self, v: u32, n: u32) {
		for i in (0..n).rev() {
			self.equi((v >> i) & 1 == 1);
		}
	}

	/// Exp-Golomb code of `v` (rav1d `read_golomb`).
	fn golomb(&mut self, v: u32) {
		let x = v + 1;
		let len = 31 - x.leading_zeros();
		for _ in 0..len {
			self.equi(false);
		}
		self.equi(true);
		for i in (0..len).rev() {
			self.equi((x >> i) & 1 == 1);
		}
	}
}

/// rav1d's CDF update after coding `s`.
#[inline]
pub fn adapt(cdf: &mut [u16], s: usize, n: usize) {
	let count = cdf[n];
	let rate = 4 + (count >> 4) + u16::from(n > 2);
	for c in &mut cdf[..s] {
		*c += (((1u32 << 15) - u32::from(*c)) >> rate) as u16;
	}
	for c in &mut cdf[s..n] {
		*c -= *c >> rate;
	}
	cdf[n] = count + u16::from(count < 32);
}

/// The range encoder (libaom `od_ec_enc`, 32-bit window).
#[derive(Debug, Clone)]
pub struct Encoder {
	low: u32,
	rng: u32,
	cnt: i32,
	/// Output before carry propagation, 16 bits per entry.
	precarry: Vec<u16>,
}

impl Default for Encoder {
	fn default() -> Self {
		Self::new()
	}
}

impl Encoder {
	pub fn new() -> Self {
		Self { low: 0, rng: 0x8000, cnt: -9, precarry: Vec::new() }
	}

	#[inline]
	fn normalize(&mut self, mut low: u32, rng: u32) {
		debug_assert!(rng > 0 && rng <= 65535);
		let d = 16 - (32 - rng.leading_zeros()) as i32;
		let mut c = self.cnt;
		let mut s = c + d;
		if s >= 0 {
			c += 16;
			let mut m = (1u32 << c) - 1;
			if s >= 8 {
				self.precarry.push((low >> c) as u16);
				low &= m;
				c -= 8;
				m >>= 8;
			}
			self.precarry.push((low >> c) as u16);
			s = c + d - 24;
			low &= m;
		}
		self.low = low << d;
		self.rng = rng << d;
		self.cnt = s;
	}

	/// Terminates the stream (`od_ec_enc_done`): the fewest bits that decode
	/// correctly whatever follows.
	pub fn finish(mut self) -> Vec<u8> {
		let l = self.low;
		let mut c = self.cnt;
		let mut s = 10 + c;
		let m = 0x3FFFu32;
		let mut e = ((l + m) & !m) | (m + 1);
		if s > 0 {
			let mut n = (1u32 << (c + 16)) - 1;
			while s > 0 {
				self.precarry.push((e >> (c + 16)) as u16);
				e &= n;
				s -= 8;
				c -= 8;
				n >>= 8;
			}
		}
		let mut out = vec![0u8; self.precarry.len()];
		let mut carry = 0u32;
		for i in (0..out.len()).rev() {
			carry += u32::from(self.precarry[i]);
			out[i] = carry as u8;
			carry >>= 8;
		}
		out
	}
}

impl Ec for Encoder {
	const ADAPT: bool = true;

	#[inline]
	fn put(&mut self, s: usize, n: usize, fl: u32, fh: u32) {
		let (mut l, mut r) = (self.low, self.rng);
		let v =
			(((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s) as u32;
		if s > 0 {
			let u = (((r >> 8) * (fl >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
				+ EC_MIN_PROB * (n + 1 - s) as u32;
			l += r - u;
			r = u - v;
		} else {
			r -= v;
		}
		self.normalize(l, r);
	}
}

/// `-log2(p / 32768)` in 1/256 bit for every 15-bit probability.
static COST: LazyLock<Vec<u16>> = LazyLock::new(|| {
	(0..=32768u32)
		.map(|p| (-(f64::from(p.max(1)) / 32768.0).log2() * 256.0).round().min(65535.0) as u16)
		.collect()
});

/// Cost in 1/256 bit of a symbol whose probability is `(fl - fh) / 32768`.
#[inline]
pub fn cost(fl: u32, fh: u32) -> u32 {
	u32::from(COST[(fl.saturating_sub(fh)) as usize])
}

/// Sums what is written, in 1/256 bit; CDFs are not adapted.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cost(pub u32);

impl Ec for Cost {
	const ADAPT: bool = false;

	#[inline]
	fn put(&mut self, _s: usize, _n: usize, fl: u32, fh: u32) {
		self.0 += cost(fl, fh);
	}

	#[inline]
	fn equi(&mut self, _b: bool) {
		self.0 += 256;
	}

	fn bits(&mut self, _v: u32, n: u32) {
		self.0 += 256 * n;
	}
}

#[cfg(test)]
pub(crate) mod tests {
	use super::*;

	/// A plain transcription of rav1d's `msac` decoder (spec §8.2.2), to
	/// check the encoder.
	pub struct Decoder<'a> {
		data: &'a [u8],
		pos: usize,
		dif: u64,
		rng: u32,
		cnt: i32,
	}

	impl<'a> Decoder<'a> {
		pub fn new(data: &'a [u8]) -> Self {
			let mut d = Self { data, pos: 0, dif: 0, rng: 0x8000, cnt: -15 };
			d.refill();
			d
		}

		fn refill(&mut self) {
			let mut c = 64 - 24 - self.cnt;
			let mut dif = self.dif;
			loop {
				if self.pos >= self.data.len() {
					dif |= !(!0xffu64 << c);
					break;
				}
				dif |= u64::from(self.data[self.pos] ^ 0xff) << c;
				self.pos += 1;
				c -= 8;
				if c < 0 {
					break;
				}
			}
			self.dif = dif;
			self.cnt = 64 - 24 - c;
		}

		fn norm(&mut self, dif: u64, rng: u32) {
			let d = 15 ^ (31 ^ rng.leading_zeros() as i32);
			let cnt = self.cnt;
			self.dif = dif << d;
			self.rng = rng << d;
			self.cnt = cnt - d;
			if (cnt as u32) < (d as u32) {
				self.refill();
			}
		}

		pub fn symbol(&mut self, cdf: &mut [u16], n: usize) -> usize {
			let c = (self.dif >> 48) as u32;
			let r = self.rng >> 8;
			let mut u;
			let mut v = self.rng;
			let mut val = 0usize;
			loop {
				u = v;
				v = (r * (u32::from(cdf[val]) >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
				v += EC_MIN_PROB * (n - val) as u32;
				if c >= v {
					break;
				}
				val += 1;
			}
			self.norm(self.dif.wrapping_sub(u64::from(v) << 48), u - v);
			adapt(cdf, val, n);
			val
		}

		pub fn bool_f(&mut self, f: u32) -> bool {
			let r = self.rng;
			let mut dif = self.dif;
			let mut v = (((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
			let vw = u64::from(v) << 48;
			let ret = dif >= vw;
			if ret {
				dif -= vw;
				v = r - v;
			}
			self.norm(dif, v);
			!ret
		}
	}

	#[test]
	fn symbols_round_trip() {
		let mut seed = 1u32;
		let mut next = || {
			seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
			seed >> 8
		};
		let mut enc_cdfs: Vec<Vec<u16>> = (2..=16)
			.map(|n: usize| {
				let mut c: Vec<u16> = (1..n).map(|i| (32768 - i * 32768 / n) as u16).collect();
				c.push(0);
				c
			})
			.collect();
		let mut dec_cdfs = enc_cdfs.clone();
		let mut ops = Vec::new();
		let mut e = Encoder::new();
		for _ in 0..20000 {
			let k = (next() % 16) as usize;
			if k == 15 {
				let b = next() % 3 == 0;
				let f = 1 + next() % 32766;
				e.bool_f(b, f);
				ops.push((usize::MAX, usize::from(b), f));
			} else {
				let n = k + 1;
				// Skewed symbols so the CDFs move.
				let s = ((next() % 100) as usize * (n + 1) / 140).min(n);
				e.symbol(s, &mut enc_cdfs[k], n);
				ops.push((k, s, 0));
			}
		}
		e.golomb(12345);
		let data = e.finish();
		let mut d = Decoder::new(&data);
		for (i, &(k, s, f)) in ops.iter().enumerate() {
			if k == usize::MAX {
				assert_eq!(usize::from(d.bool_f(f)), s, "op {i}");
			} else {
				assert_eq!(d.symbol(&mut dec_cdfs[k], k + 1), s, "op {i}");
			}
		}
		let mut len = 0;
		while !d.bool_f(16384) {
			len += 1;
		}
		let mut v = 1u32;
		for _ in 0..len {
			v = (v << 1) | u32::from(d.bool_f(16384));
		}
		assert_eq!(v - 1, 12345);
		assert_eq!(enc_cdfs, dec_cdfs);
	}
}
