//! Table-driven variable-length-code decoder (FFmpeg's `VLC` / `get_vlc2`).
//!
//! A code set is turned into multi-level lookup tables: the first level is
//! indexed by the next `root_bits` bits; longer codes chain into sub-tables.
//! Decoding is one or two table lookups for every code used by AAC and
//! H.264 CAVLC.

use super::BitReader;

#[derive(Debug, Clone, Copy, Default)]
struct Entry {
	/// Symbol (leaf) or sub-table offset (link).
	value: u32,
	/// > 0: leaf, bits to consume. < 0: link, sub-table index bits. 0: invalid.
	len: i8,
}

#[derive(Debug, Clone)]
pub struct Vlc {
	table: Vec<Entry>,
	root_bits: u32,
}

/// One code: `len` bits of `code` (MSB-first) decode to `symbol`.
#[derive(Debug, Clone, Copy)]
pub struct Code {
	pub code: u32,
	pub len: u8,
	pub symbol: u32,
}

impl Vlc {
	/// Builds the lookup tables. Returns `None` if a code is longer than 32
	/// bits, empty, or the set is not prefix-free.
	pub fn new(codes: &[Code], root_bits: u32) -> Option<Self> {
		let mut vlc = Self { table: Vec::new(), root_bits };
		let mut sorted: Vec<Code> = codes.to_vec();
		if sorted.iter().any(|c| c.len == 0 || c.len > 32) {
			return None;
		}
		// Left-align codes so prefix grouping is a simple shift.
		sorted.sort_by_key(|c| (u64::from(c.code) << (32 - u32::from(c.len)), c.len));
		vlc.build(&sorted, root_bits, 0)?;
		Some(vlc)
	}

	/// Builds the table for `codes` (all sharing a `consumed`-bit prefix that
	/// is already stripped conceptually) and returns its offset.
	fn build(&mut self, codes: &[Code], bits: u32, consumed: u32) -> Option<usize> {
		let base = self.table.len();
		let size = 1usize << bits;
		self.table.resize(base + size, Entry::default());

		let mut i = 0;
		while i < codes.len() {
			let c = codes[i];
			let len = u32::from(c.len) - consumed;
			// Code bits below the consumed prefix.
			let rest = c.code & mask(len);
			if len <= bits {
				let start = (rest << (bits - len)) as usize;
				let count = 1usize << (bits - len);
				for slot in &mut self.table[base + start..base + start + count] {
					if slot.len != 0 {
						return None;
					}
					*slot = Entry { value: c.symbol, len: len as i8 };
				}
				i += 1;
			} else {
				// Group every code sharing this `bits`-bit prefix.
				let prefix = rest >> (len - bits);
				let mut j = i;
				let mut max_len = 0;
				while j < codes.len() {
					let cj = codes[j];
					let lj = u32::from(cj.len) - consumed;
					if lj <= bits || (cj.code & mask(lj)) >> (lj - bits) != prefix {
						break;
					}
					max_len = max_len.max(lj - bits);
					j += 1;
				}
				let sub_bits = max_len.min(self.root_bits).max(1);
				let slot = base + prefix as usize;
				if self.table[slot].len != 0 {
					return None;
				}
				let offset = self.build(&codes[i..j], sub_bits, consumed + bits)?;
				self.table[slot] = Entry { value: offset as u32, len: -(sub_bits as i8) };
				i = j;
			}
		}
		Some(base)
	}

	/// A table that rejects every code. Used as a fallback so table setup
	/// never needs to panic; generated tables are verified by tests.
	pub fn empty() -> Self {
		Self { table: vec![Entry::default(); 2], root_bits: 1 }
	}

	/// Decodes one symbol. Invalid codes return `None` (the reader is left
	/// past the bits examined).
	#[inline]
	pub fn decode(&self, bs: &mut BitReader) -> Option<u32> {
		let mut bits = self.root_bits;
		let mut offset = 0usize;
		loop {
			let index = offset + bs.peek(bits) as usize;
			let e = *self.table.get(index)?;
			if e.len > 0 {
				bs.skip(e.len as usize);
				return Some(e.value);
			}
			if e.len == 0 {
				bs.skip(bits as usize);
				return None;
			}
			bs.skip(bits as usize);
			bits = u32::from(e.len.unsigned_abs());
			offset = e.value as usize;
		}
	}
}

#[inline]
fn mask(len: u32) -> u32 {
	if len >= 32 { u32::MAX } else { (1u32 << len) - 1 }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn decodes_short_and_long_codes() {
		// 0 -> a, 10 -> b, 110 -> c, 1110 -> d, 11110 -> e, 11111 -> f
		let codes = [
			Code { code: 0b0, len: 1, symbol: 0 },
			Code { code: 0b10, len: 2, symbol: 1 },
			Code { code: 0b110, len: 3, symbol: 2 },
			Code { code: 0b1110, len: 4, symbol: 3 },
			Code { code: 0b11110, len: 5, symbol: 4 },
			Code { code: 0b11111, len: 5, symbol: 5 },
		];
		let vlc = Vlc::new(&codes, 2).unwrap();
		// 11111 11110 1110 110 10 0 -> 5 4 3 2 1 0
		let data = [0b1111_1111, 0b1011_1011, 0b0100_0000];
		let mut bs = BitReader::new(&data);
		for want in [5, 4, 3, 2, 1, 0] {
			assert_eq!(vlc.decode(&mut bs), Some(want));
		}
		assert_eq!(bs.pos(), 20);
	}

	#[test]
	fn rejects_ambiguous_codes() {
		let codes = [Code { code: 0, len: 1, symbol: 0 }, Code { code: 0, len: 2, symbol: 1 }];
		assert!(Vlc::new(&codes, 4).is_none());
	}
}
