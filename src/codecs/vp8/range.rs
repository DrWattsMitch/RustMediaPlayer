//! The VP8 boolean entropy decoder (RFC 6386 section 7; libvpx
//! `vp8/decoder/dboolhuff.[ch]`), with libvpx's 64-bit window so that the
//! bits read past the end of a partition (zeros) match it exactly.

const VALUE_SIZE: i32 = 64;
/// Added to `count` once the data runs out (libvpx `VP8_LOTS_OF_BITS`).
const LOTS_OF_BITS: i32 = 0x4000_0000;

#[derive(Clone)]
pub struct BoolDecoder<'a> {
	data: &'a [u8],
	pos: usize,
	value: u64,
	/// Bits in `value` beyond the top byte, minus 8 (libvpx's `count`).
	count: i32,
	range: u32,
}

impl<'a> BoolDecoder<'a> {
	pub fn new(data: &'a [u8]) -> Self {
		let mut d = Self { data, pos: 0, value: 0, count: -8, range: 255 };
		d.fill();
		d
	}

	fn fill(&mut self) {
		let mut shift = VALUE_SIZE - 8 - (self.count + 8);
		let bytes_left = self.data.len() - self.pos;
		let bits_left = (bytes_left * 8).min(i32::MAX as usize / 2) as i32;
		let x = shift + 8 - bits_left;
		let mut loop_end = 0;
		if x >= 0 {
			self.count += LOTS_OF_BITS;
			loop_end = x;
		}
		if x < 0 || bits_left != 0 {
			while shift >= loop_end {
				self.count += 8;
				self.value |= u64::from(self.data[self.pos]) << shift;
				self.pos += 1;
				shift -= 8;
			}
		}
	}

	/// `true` once bits past the end of the data have been decoded.
	pub fn overrun(&self) -> bool {
		self.count > VALUE_SIZE && self.count < LOTS_OF_BITS
	}

	/// One bool with `prob / 256` probability of being 0.
	#[inline]
	pub fn read(&mut self, prob: u8) -> bool {
		let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
		if self.count < 0 {
			self.fill();
		}
		let bigsplit = u64::from(split) << (VALUE_SIZE - 8);
		let bit = self.value >= bigsplit;
		let range = if bit {
			self.value -= bigsplit;
			self.range - split
		} else {
			split
		};
		let shift = (range as u8).leading_zeros();
		self.range = range << shift;
		self.value <<= shift;
		self.count -= shift as i32;
		bit
	}

	#[inline]
	pub fn bit(&mut self) -> bool {
		self.read(128)
	}

	/// An unsigned `n`-bit literal, most significant bit first.
	pub fn literal(&mut self, n: u32) -> u32 {
		let mut v = 0;
		for _ in 0..n {
			v = (v << 1) | u32::from(self.bit());
		}
		v
	}

	/// A magnitude-then-sign value (`n` bits, then the sign bit).
	pub fn signed_literal(&mut self, n: u32) -> i32 {
		let v = self.literal(n) as i32;
		if self.bit() { -v } else { v }
	}

	/// Reads a tree-coded value (`vp8_treed_read`): `tree` holds pairs of
	/// child indices, leaves as `-value`.
	#[inline]
	pub fn tree(&mut self, tree: &[i8], probs: &[u8]) -> u8 {
		let mut i = 0usize;
		loop {
			let next = tree[i + usize::from(self.read(probs[i >> 1]))];
			if next <= 0 {
				return (-next) as u8;
			}
			i = next as usize;
		}
	}

	/// The sign of a coefficient (`GetSigned` in `detokenize.c`): an even
	/// split, normalised by exactly one bit.
	#[inline]
	pub fn sign(&mut self, v: i32) -> i32 {
		let split = (self.range + 1) >> 1;
		if self.count < 0 {
			self.fill();
		}
		let bigsplit = u64::from(split) << (VALUE_SIZE - 8);
		let out = if self.value < bigsplit {
			self.range = split;
			v
		} else {
			self.range -= split;
			self.value -= bigsplit;
			-v
		};
		self.range += self.range;
		self.value = self.value.wrapping_add(self.value);
		self.count -= 1;
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A minimal boolean encoder (RFC 6386 section 7.3).
	struct Encoder {
		out: Vec<u8>,
		range: u32,
		bottom: u32,
		bit_count: i32,
	}

	impl Encoder {
		fn new() -> Self {
			Self { out: Vec::new(), range: 255, bottom: 0, bit_count: 24 }
		}

		fn add_one(&mut self) {
			let mut i = self.out.len();
			while i > 0 {
				i -= 1;
				if self.out[i] == 255 {
					self.out[i] = 0;
				} else {
					self.out[i] += 1;
					break;
				}
			}
		}

		fn write(&mut self, prob: u8, bit: bool) {
			let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
			if bit {
				self.bottom = self.bottom.wrapping_add(split);
				self.range -= split;
			} else {
				self.range = split;
			}
			while self.range < 128 {
				self.range <<= 1;
				if self.bottom & (1 << 31) != 0 {
					self.add_one();
				}
				self.bottom <<= 1;
				self.bit_count -= 1;
				if self.bit_count == 0 {
					self.out.push((self.bottom >> 24) as u8);
					self.bottom &= (1 << 24) - 1;
					self.bit_count = 8;
				}
			}
		}

		fn finish(mut self) -> Vec<u8> {
			for _ in 0..32 {
				self.write(128, false);
			}
			self.out
		}
	}

	#[test]
	fn round_trip() {
		let mut e = Encoder::new();
		let mut seed = 7u32;
		let mut bits = Vec::new();
		for i in 0..5000 {
			seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
			let prob = ((seed >> 8) % 255 + 1) as u8;
			let bit = (seed >> 20) % 256 >= u32::from(prob) || i % 97 == 0;
			e.write(prob, bit);
			bits.push((prob, bit));
		}
		let data = e.finish();
		let mut d = BoolDecoder::new(&data);
		for (i, &(prob, bit)) in bits.iter().enumerate() {
			assert_eq!(d.read(prob), bit, "bit {i}");
		}
		assert!(!d.overrun());
		// Past the end, zeros.
		let mut d = BoolDecoder::new(&[]);
		assert!(!d.read(128));
		for _ in 0..80 {
			d.read(1);
		}
		assert!(d.overrun());
	}
}
