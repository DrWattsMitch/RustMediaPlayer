//! MSB-first bit reader shared by the AAC and H.264 decoders (FFmpeg's
//! `GetBitContext`, `get_bits.h` / `golomb.h`).
//!
//! Reads past the end yield zero bits and set a sticky overrun flag instead
//! of failing, so hot loops stay branch-light. Callers check
//! [`BitReader::overrun`] at syntax-element boundaries and turn an overrun
//! into a decoding error.

#[derive(Clone)]
pub struct BitReader<'a> {
	data: &'a [u8],
	/// Position in bits from the start of `data`.
	pos: usize,
	/// Size of `data` in bits.
	limit: usize,
}

impl<'a> BitReader<'a> {
	pub fn new(data: &'a [u8]) -> Self {
		Self { data, pos: 0, limit: data.len() * 8 }
	}

	#[inline]
	pub fn pos(&self) -> usize {
		self.pos
	}

	#[inline]
	pub fn limit(&self) -> usize {
		self.limit
	}

	/// Bits left before the end of the data (0 once overrun).
	#[inline]
	pub fn bits_left(&self) -> usize {
		self.limit.saturating_sub(self.pos)
	}

	/// `true` once any read went past the end of the data.
	#[inline]
	pub fn overrun(&self) -> bool {
		self.pos > self.limit
	}

	#[inline]
	pub fn data(&self) -> &'a [u8] {
		self.data
	}

	/// Returns the next `n` (0..=32) bits without consuming them; bits past
	/// the end read as zero.
	#[inline]
	pub fn peek(&self, n: u32) -> u32 {
		debug_assert!(n <= 32);
		if n == 0 {
			return 0;
		}
		let byte = self.pos >> 3;
		let window = match self.data.get(byte..byte + 8) {
			Some(&[a, b, c, d, e, f, g, h]) => u64::from_be_bytes([a, b, c, d, e, f, g, h]),
			_ => {
				let mut bytes = [0u8; 8];
				for (i, slot) in bytes.iter_mut().enumerate() {
					*slot = byte.checked_add(i).and_then(|at| self.data.get(at)).copied().unwrap_or(0);
				}
				u64::from_be_bytes(bytes)
			}
		};
		((window << (self.pos & 7)) >> (64 - n)) as u32
	}

	#[inline]
	pub fn skip(&mut self, n: usize) {
		self.pos = self.pos.saturating_add(n);
	}

	/// Reads `n` (0..=32) bits.
	#[inline]
	pub fn bits(&mut self, n: u32) -> u32 {
		let value = self.peek(n);
		self.pos = self.pos.saturating_add(n as usize);
		value
	}

	#[inline]
	pub fn bit(&mut self) -> bool {
		self.bits(1) != 0
	}

	/// Reads `n` bits as a two's-complement signed value.
	#[inline]
	pub fn signed(&mut self, n: u32) -> i32 {
		if n == 0 {
			return 0;
		}
		let v = self.bits(n);
		((v << (32 - n)) as i32) >> (32 - n)
	}

	/// Unsigned Exp-Golomb code `ue(v)`; at most 32 leading zeros.
	#[inline]
	pub fn ue(&mut self) -> u32 {
		let window = self.peek(32);
		let zeros = window.leading_zeros();
		if zeros < 16 {
			let len = 2 * zeros + 1;
			self.skip(len as usize);
			return (window >> (32 - len)) - 1;
		}
		if zeros >= 32 {
			// Invalid in every syntax we parse; consume and flag via overrun.
			self.skip(self.bits_left() + 1);
			return 0;
		}
		self.skip(zeros as usize);
		let value = self.bits(zeros + 1);
		value - 1
	}

	/// Signed Exp-Golomb code `se(v)`.
	#[inline]
	pub fn se(&mut self) -> i32 {
		let k = self.ue();
		let magnitude = (k / 2 + (k & 1)) as i32;
		if k & 1 == 1 { magnitude } else { -magnitude }
	}

	/// Skips to the next byte boundary.
	#[inline]
	pub fn align(&mut self) {
		self.pos = (self.pos + 7) & !7;
	}

	#[inline]
	pub fn is_aligned(&self) -> bool {
		self.pos & 7 == 0
	}
}

#[cfg(test)]
mod tests {
	use super::BitReader;

	#[test]
	fn reads_bits_and_golomb_codes() {
		// 1 | 010 | 011 | 00100 | 00101 = ue 0, 1, 2, 3, se(4) = -2
		let mut bs = BitReader::new(&[0b1010_0110, 0b0100_0010, 0b1000_0000]);
		assert_eq!(bs.ue(), 0);
		assert_eq!(bs.ue(), 1);
		assert_eq!(bs.ue(), 2);
		assert_eq!(bs.ue(), 3);
		assert_eq!(bs.se(), -2);
		assert!(!bs.overrun());
		bs.bits(16);
		assert!(bs.overrun());
	}

	#[test]
	fn signed_reads() {
		let mut bs = BitReader::new(&[0b1110_0000]);
		assert_eq!(bs.signed(3), -1);
	}
}
