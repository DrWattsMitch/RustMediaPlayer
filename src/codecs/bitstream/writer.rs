//! MSB-first bit writer shared by the AAC and H.264 encoders (FFmpeg's
//! `PutBitContext`, `put_bits.h` / `put_golomb.h`).
//!
//! Bits accumulate in a 64-bit register and are flushed a byte at a time,
//! so writes never fail and never need bounds checks. [`BitWriter::bits`]
//! gives the exact length written so far, which the encoders' rate control
//! uses to account for every bit.

#[derive(Debug, Clone, Default)]
pub struct BitWriter {
	out: Vec<u8>,
	acc: u64,
	/// Bits held in `acc` (always < 8 between calls).
	used: u32,
}

impl BitWriter {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_capacity(bytes: usize) -> Self {
		Self { out: Vec::with_capacity(bytes), acc: 0, used: 0 }
	}

	/// Writes the low `n` bits of `value` (`n` <= 32).
	#[inline]
	pub fn put(&mut self, value: u32, n: u32) {
		debug_assert!(n <= 32);
		if n == 0 {
			return;
		}
		let mask = if n == 32 { u64::from(u32::MAX) } else { (1u64 << n) - 1 };
		self.acc = (self.acc << n) | (u64::from(value) & mask);
		self.used += n;
		while self.used >= 8 {
			self.used -= 8;
			self.out.push((self.acc >> self.used) as u8);
		}
	}

	#[inline]
	pub fn bit(&mut self, b: bool) {
		self.put(u32::from(b), 1);
	}

	/// Unsigned Exp-Golomb code `ue(v)` (ISO/IEC 14496-10 §9.1).
	pub fn ue(&mut self, v: u32) {
		let x = u64::from(v) + 1;
		let len = 64 - x.leading_zeros(); // bits of x
		// (len - 1) zeros, then x in len bits; split to stay within 32.
		let total = 2 * len - 1;
		if total <= 32 {
			self.put(x as u32, total);
		} else {
			self.put(0, len - 1);
			self.put(x as u32, len);
		}
	}

	/// Signed Exp-Golomb code `se(v)`: 1 → 1, -1 → 2, 2 → 3, ...
	pub fn se(&mut self, v: i32) {
		let k = if v > 0 { (v as u32) * 2 - 1 } else { v.unsigned_abs() * 2 };
		self.ue(k);
	}

	/// Bits written so far.
	pub fn bits(&self) -> usize {
		self.out.len() * 8 + self.used as usize
	}

	pub fn is_aligned(&self) -> bool {
		self.used == 0
	}

	/// Pads with zero bits to a byte boundary.
	pub fn align_zero(&mut self) {
		if self.used > 0 {
			self.put(0, 8 - self.used);
		}
	}

	/// `rbsp_trailing_bits()`: a one bit, then zeros to a byte boundary.
	pub fn rbsp_trailing(&mut self) {
		self.bit(true);
		self.align_zero();
	}

	/// Appends whole bytes (the writer must be byte-aligned).
	pub fn put_bytes(&mut self, bytes: &[u8]) {
		debug_assert!(self.is_aligned());
		self.out.extend_from_slice(bytes);
	}

	/// Appends everything written to `other` (any alignment).
	pub fn append(&mut self, other: &BitWriter) {
		if self.used == 0 {
			self.out.extend_from_slice(&other.out);
		} else {
			for &b in &other.out {
				self.put(u32::from(b), 8);
			}
		}
		if other.used > 0 {
			self.put(other.acc as u32 & ((1 << other.used) - 1), other.used);
		}
	}

	/// Pads to a byte boundary with zeros and returns the bytes.
	pub fn finish(mut self) -> Vec<u8> {
		self.align_zero();
		self.out
	}
}

/// Number of bits of `ue(v)`.
#[inline]
pub fn ue_bits(v: u32) -> u32 {
	let x = u64::from(v) + 1;
	2 * (64 - x.leading_zeros()) - 1
}

/// Number of bits of `se(v)`.
#[inline]
pub fn se_bits(v: i32) -> u32 {
	ue_bits(if v > 0 { (v as u32) * 2 - 1 } else { v.unsigned_abs() * 2 })
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::bitstream::BitReader;

	#[test]
	fn round_trips_with_reader() {
		let mut w = BitWriter::new();
		w.put(0b101, 3);
		w.ue(0);
		w.ue(7);
		w.se(-3);
		w.se(4);
		w.ue(u32::MAX - 1);
		w.put(0xDEAD_BEEF, 32);
		let expected_bits = 3 + 1 + ue_bits(7) + se_bits(-3) + se_bits(4) + ue_bits(u32::MAX - 1) + 32;
		assert_eq!(w.bits(), expected_bits as usize);
		let data = w.finish();
		let mut r = BitReader::new(&data);
		assert_eq!(r.bits(3), 0b101);
		assert_eq!(r.ue(), 0);
		assert_eq!(r.ue(), 7);
		assert_eq!(r.se(), -3);
		assert_eq!(r.se(), 4);
		r.ue();
		assert_eq!(r.bits(32), 0xDEAD_BEEF);
	}

	#[test]
	fn append_keeps_bit_order() {
		let mut a = BitWriter::new();
		a.put(0b101, 3);
		let mut b = BitWriter::new();
		b.put(0xABC, 12);
		b.put(0b1, 1);
		a.append(&b);
		assert_eq!(a.bits(), 16);
		assert_eq!(a.finish(), vec![0b1011_0101, 0b0111_1001]);
	}
}
