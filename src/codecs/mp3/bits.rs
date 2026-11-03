//! MSB-first bit reader used by the Layer III side-info, scalefactor and
//! Huffman stages.
//!
//! Reads past the end of the data yield zero bits instead of failing; the
//! callers enforce their own limits (`part2_3_length`, frame size), exactly
//! like the reference decoder. This keeps the hot Huffman loop branch-light
//! while remaining memory-safe.

pub struct BitReader<'a> {
	data: &'a [u8],
	/// Current position in bits from the start of `data`.
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

	#[inline]
	pub fn set_pos(&mut self, pos: usize) {
		self.pos = pos;
	}

	/// Returns the next `n` (0..=32) bits without consuming them.
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
				// Slow path near the end of the buffer: pad with zeros.
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
	pub fn skip(&mut self, n: u32) {
		self.pos += n as usize;
	}

	/// Reads `n` bits. Like the reference `get_bits`, a read that would cross
	/// `limit` advances the position and returns 0, so callers detect overrun
	/// by comparing `pos()` with `limit()`.
	#[inline]
	pub fn get_bits(&mut self, n: u32) -> u32 {
		if self.pos + n as usize > self.limit {
			self.pos += n as usize;
			return 0;
		}
		let value = self.peek(n);
		self.pos += n as usize;
		value
	}
}

#[cfg(test)]
mod tests {
	use super::BitReader;

	#[test]
	fn reads_msb_first_and_pads_with_zeros() {
		let mut bs = BitReader::new(&[0b1010_1100, 0xFF]);
		assert_eq!(bs.get_bits(3), 0b101);
		assert_eq!(bs.get_bits(5), 0b01100);
		assert_eq!(bs.peek(12), 0xFF0);
		assert_eq!(bs.get_bits(8), 0xFF);
		assert_eq!(bs.get_bits(4), 0); // past the end
		assert!(bs.pos() > bs.limit());
	}
}
