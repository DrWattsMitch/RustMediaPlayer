//! LSB-first bit reader (Vorbis I §2.1, libogg's `oggpack_*`).
//!
//! Vorbis packs values starting from the least significant bit of each
//! byte. Reading past the end of the packet is not an error in itself:
//! the specification defines what an early end means at each point
//! (§4.3.1, §7.2.2, §8.6.2), so reads return `None` and the caller
//! decides.

#[derive(Clone)]
pub struct Bits<'a> {
	data: &'a [u8],
	/// Position in bits.
	pos: usize,
}

impl<'a> Bits<'a> {
	pub fn new(data: &'a [u8]) -> Self {
		Self { data, pos: 0 }
	}

	pub fn left(&self) -> usize {
		(self.data.len() * 8).saturating_sub(self.pos)
	}

	/// The next `n` bits (at most 32) without consuming them; bits past
	/// the end read as zero.
	pub fn peek(&self, n: u32) -> u32 {
		debug_assert!(n <= 32);
		if n == 0 {
			return 0;
		}
		let byte = self.pos / 8;
		let shift = self.pos % 8;
		let mut window = 0u64;
		for i in 0..5 {
			let b = self.data.get(byte + i).copied().unwrap_or(0);
			window |= u64::from(b) << (8 * i);
		}
		((window >> shift) & ((1u64 << n) - 1)) as u32
	}

	pub fn skip(&mut self, n: usize) {
		self.pos += n;
	}

	/// Reads `n` bits (at most 32); `None` if the packet ends first.
	pub fn read(&mut self, n: u32) -> Option<u32> {
		if (n as usize) > self.left() {
			self.pos = self.data.len() * 8;
			return None;
		}
		let v = self.peek(n);
		self.pos += n as usize;
		Some(v)
	}

	pub fn flag(&mut self) -> Option<bool> {
		self.read(1).map(|b| b == 1)
	}
}

#[cfg(test)]
mod tests {
	use super::Bits;

	#[test]
	fn reads_lsb_first() {
		// Spec §2.1.4 example: 0b1100 (4 bits), 0b10001 (5 bits) ...
		let mut b = Bits::new(&[0b0001_1100, 0b0000_0001, 0xFF]);
		assert_eq!(b.read(4), Some(0b1100));
		assert_eq!(b.read(5), Some(0b10001));
		assert_eq!(b.read(7), Some(0));
		assert_eq!(b.read(8), Some(0xFF));
		assert_eq!(b.read(1), None);
		assert_eq!(b.left(), 0);
	}
}
