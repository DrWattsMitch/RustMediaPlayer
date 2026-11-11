//! VP9's boolean decoder (spec §9.2; libvpx `vpx_dsp/bitreader.h`).
//!
//! A binary arithmetic decoder with 8-bit probabilities: `split = 1 +
//! (((range - 1) * prob) >> 8)`, renormalised a bit at a time. Bytes past
//! the end of the buffer read as zero; [`BoolDecoder::overrun`] reports
//! whether the decoder consumed more than the buffer holds, which libvpx
//! treats as a corrupt tile.

use super::Tree;

pub struct BoolDecoder<'a> {
	data: &'a [u8],
	pos: usize,
	/// Undecoded bits, MSB-aligned.
	value: u64,
	/// Valid bits in `value`.
	bits: i32,
	range: u32,
	/// Zero bytes appended past the end.
	padding: usize,
}

impl<'a> BoolDecoder<'a> {
	/// Starts decoding `data`; `None` if the marker bit is set (spec §9.2.1).
	pub fn new(data: &'a [u8]) -> Option<Self> {
		let mut d = Self { data, pos: 0, value: 0, bits: 0, range: 255, padding: 0 };
		d.fill();
		if d.read(128) { None } else { Some(d) }
	}

	fn fill(&mut self) {
		while self.bits <= 56 {
			let byte = match self.data.get(self.pos) {
				Some(&b) => {
					self.pos += 1;
					b
				}
				None => {
					self.padding += 1;
					0
				}
			};
			self.value |= u64::from(byte) << (56 - self.bits);
			self.bits += 8;
		}
	}

	/// Decodes one bool whose probability of being 0 is `prob / 256`.
	#[inline]
	pub fn read(&mut self, prob: u8) -> bool {
		let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
		if self.bits < 8 {
			self.fill();
		}
		let big = u64::from(split) << 56;
		let bit = self.value >= big;
		if bit {
			self.range -= split;
			self.value -= big;
		} else {
			self.range = split;
		}
		let shift = self.range.leading_zeros() - 24;
		self.range <<= shift;
		self.value <<= shift;
		self.bits -= shift as i32;
		bit
	}

	pub fn bit(&mut self) -> bool {
		self.read(128)
	}

	/// An unsigned `n`-bit literal, MSB first.
	pub fn literal(&mut self, n: u32) -> u32 {
		(0..n).fold(0, |v, _| (v << 1) | u32::from(self.bit()))
	}

	/// A tree-coded symbol (`vpx_read_tree`).
	pub fn tree(&mut self, tree: &Tree, probs: &[u8]) -> u8 {
		let mut i = 0usize;
		loop {
			let next = tree[i + usize::from(self.read(probs[i >> 1]))];
			if next <= 0 {
				return (-next) as u8;
			}
			i = next as usize;
		}
	}

	/// True once more bits were consumed than the buffer holds (libvpx's
	/// `vpx_reader_has_error`, which allows the 8 bits of look-ahead).
	pub fn overrun(&self) -> bool {
		let consumed = (self.pos + self.padding) * 8 - self.bits.max(0) as usize;
		consumed > self.data.len() * 8 + 8
	}
}

#[cfg(test)]
mod tests {
	use super::BoolDecoder;

	/// A minimal boolean encoder (libvpx `vpx_dsp/bitwriter.c`).
	struct Writer {
		low: u32,
		range: u32,
		count: i32,
		out: Vec<u8>,
	}

	impl Writer {
		fn new() -> Self {
			Self { low: 0, range: 255, count: -24, out: Vec::new() }
		}

		fn write(&mut self, bit: bool, prob: u8) {
			let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
			let (mut low, mut range) = (self.low, split);
			if bit {
				low += split;
				range = self.range - split;
			}
			let mut shift = range.leading_zeros() as i32 - 24;
			range <<= shift;
			self.count += shift;
			if self.count >= 0 {
				let offset = shift - self.count;
				if (low << (offset - 1)) & 0x8000_0000 != 0 {
					let mut x = self.out.len();
					while x > 0 && self.out[x - 1] == 0xFF {
						self.out[x - 1] = 0;
						x -= 1;
					}
					self.out[x - 1] += 1;
				}
				self.out.push((low >> (24 - offset)) as u8);
				low <<= offset;
				shift = self.count;
				low &= 0xFF_FFFF;
				self.count -= 8;
			}
			low <<= shift;
			self.low = low;
			self.range = range;
		}

		fn finish(mut self) -> Vec<u8> {
			for _ in 0..32 {
				self.write(false, 128);
			}
			self.out
		}
	}

	#[test]
	fn round_trips_an_encoder() {
		let mut w = Writer::new();
		w.write(false, 128); // marker
		let symbols: Vec<(bool, u8)> =
			(0..2000u32).map(|i| ((i * 7919) % 5 < 2, ((i * 31) % 254 + 1) as u8)).collect();
		for &(bit, prob) in &symbols {
			w.write(bit, prob);
		}
		let data = w.finish();
		let mut r = BoolDecoder::new(&data).unwrap();
		for (i, &(bit, prob)) in symbols.iter().enumerate() {
			assert_eq!(r.read(prob), bit, "symbol {i}");
		}
		assert!(!r.overrun());
	}
}
