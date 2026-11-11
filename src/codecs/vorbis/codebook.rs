//! Vorbis codebooks (Vorbis I §3): Huffman entropy codes over entry
//! numbers, optionally mapped to vectors of floats (VQ lookup types 1
//! and 2).
//!
//! Codeword lengths are stored per entry; the codewords themselves are
//! assigned in entry order, each taking the lowest free node of the tree
//! (libvorbis' `_make_words`). Decoding uses a table on the next
//! [`FAST_BITS`] bits, falling back to a walk of the explicit tree for the
//! longer codewords.

use super::bits::Bits;
use super::ilog;

/// Bits looked up at once (codewords up to this length decode in one step).
const FAST_BITS: u32 = 10;
/// Largest VQ table built (entries × dimensions), as a guard against
/// hostile headers: real streams stay far below.
const MAX_VQ_VALUES: usize = 1 << 24;

/// One slot of the fast table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fast {
	/// No codeword starts with these bits.
	Invalid,
	/// A codeword of `len` bits for `entry`.
	Leaf { entry: u32, len: u8 },
	/// A longer codeword: continue the tree walk at `node` after
	/// [`FAST_BITS`] bits.
	Node(u32),
}

/// Tree node children: `>= 0` is another node, `< 0` is `-(entry + 1)`,
/// `i32::MIN` is absent.
const ABSENT: i32 = i32::MIN;

#[derive(Debug, Clone)]
pub struct Codebook {
	pub dimensions: usize,
	pub entries: usize,
	fast: Vec<Fast>,
	tree: Vec<[i32; 2]>,
	/// `entries × dimensions` VQ values, when the book has a lookup table.
	vq: Option<Vec<f32>>,
}

/// Floats in the codebook header (§9.2.2 `float32_unpack`).
fn float32_unpack(x: u32) -> f32 {
	let mantissa = f64::from(x & 0x1F_FFFF);
	let exponent = ((x >> 21) & 0x3FF) as i32 - 788;
	let value = mantissa * 2f64.powi(exponent.clamp(-63, 63));
	(if x & 0x8000_0000 != 0 { -value } else { value }) as f32
}

/// §9.2.3 `lookup1_values`: the largest `r` with `r^dimensions <= entries`.
fn lookup1_values(entries: usize, dimensions: usize) -> usize {
	let fits = |r: usize| {
		let mut acc: usize = 1;
		for _ in 0..dimensions {
			acc = match acc.checked_mul(r) {
				Some(v) if v <= entries => v,
				_ => return false,
			};
		}
		true
	};
	let mut r = (entries as f64).powf(1.0 / dimensions as f64).floor() as usize;
	while r > 0 && !fits(r) {
		r -= 1;
	}
	while fits(r + 1) {
		r += 1;
	}
	r
}

/// Codewords for `lengths` (0 = unused entry), MSB-first as the tree
/// reads them. `None` for an over- or under-specified tree, except the
/// single-entry book whose only codeword is `0` (libvorbis accepts it).
fn make_codewords(lengths: &[u8]) -> Option<Vec<u32>> {
	let mut marker = [0u32; 33];
	let mut words = vec![0u32; lengths.len()];
	let mut used = 0usize;
	for (i, &len) in lengths.iter().enumerate() {
		if len == 0 {
			continue;
		}
		let len = usize::from(len);
		let mut entry = marker[len];
		if len < 32 && entry >> len != 0 {
			return None;
		}
		words[i] = entry;
		used += 1;
		for j in (1..=len).rev() {
			if marker[j] & 1 != 0 {
				if j == 1 {
					marker[1] += 1;
				} else {
					marker[j] = marker[j - 1] << 1;
				}
				break;
			}
			marker[j] += 1;
		}
		for j in len + 1..33 {
			if marker[j] >> 1 == entry {
				entry = marker[j];
				marker[j] = marker[j - 1] << 1;
			} else {
				break;
			}
		}
	}
	let single = used == 1 && marker[2] == 2;
	if !single && (1..33).any(|i| marker[i] & (u32::MAX >> (32 - i)) != 0) {
		return None;
	}
	Some(words)
}

impl Codebook {
	/// Reads one codebook from the setup header (§3.2.1).
	pub(super) fn read(b: &mut Bits) -> Option<Self> {
		if b.read(24)? != 0x56_4342 {
			return None;
		}
		let dimensions = b.read(16)? as usize;
		let entries = b.read(24)? as usize;
		if dimensions == 0 || entries == 0 {
			return None;
		}
		let mut lengths = vec![0u8; entries];
		if b.flag()? {
			// Ordered: runs of entries with increasing lengths.
			let mut length = b.read(5)? + 1;
			let mut current = 0usize;
			while current < entries {
				let count = b.read(ilog((entries - current) as u32))? as usize;
				if length > 32 || current + count > entries {
					return None;
				}
				lengths[current..current + count].fill(length as u8);
				current += count;
				length += 1;
			}
		} else {
			let sparse = b.flag()?;
			for len in lengths.iter_mut() {
				if !sparse || b.flag()? {
					*len = b.read(5)? as u8 + 1;
				}
			}
		}

		let lookup = b.read(4)?;
		let vq = match lookup {
			0 => None,
			1 | 2 => {
				let minimum = float32_unpack(b.read(32)?);
				let delta = float32_unpack(b.read(32)?);
				let value_bits = b.read(4)? + 1;
				let sequence = b.flag()?;
				let values = if lookup == 1 {
					lookup1_values(entries, dimensions)
				} else {
					entries.checked_mul(dimensions)?
				};
				if values == 0 || entries.checked_mul(dimensions)? > MAX_VQ_VALUES {
					return None;
				}
				if b.left() < values * value_bits as usize {
					return None;
				}
				let multiplicands: Vec<u32> =
					(0..values).map(|_| b.read(value_bits)).collect::<Option<_>>()?;
				Some(build_vq(&multiplicands, entries, dimensions, lookup, minimum, delta, sequence))
			}
			_ => return None,
		};

		let words = make_codewords(&lengths)?;
		let (fast, tree) = build_decoder(&lengths, &words);
		Some(Self { dimensions, entries, fast, tree, vq })
	}

	pub fn has_vq(&self) -> bool {
		self.vq.is_some()
	}

	/// Decodes one entry number; `None` at the end of the packet or on an
	/// invalid codeword.
	pub(super) fn scalar(&self, b: &mut Bits) -> Option<u32> {
		let left = b.left();
		match self.fast[b.peek(FAST_BITS) as usize] {
			Fast::Leaf { entry, len } if usize::from(len) <= left => {
				b.skip(usize::from(len));
				Some(entry)
			}
			Fast::Node(mut node) if left > FAST_BITS as usize => {
				b.skip(FAST_BITS as usize);
				loop {
					let child = self.tree[node as usize][b.read(1)? as usize];
					match child {
						ABSENT => return None,
						c if c < 0 => return Some((-(c + 1)) as u32),
						c => node = c as u32,
					}
				}
			}
			_ => {
				// Near the end of the packet: consume what is left.
				b.skip(left);
				None
			}
		}
	}

	/// Decodes one entry and returns its VQ vector (`dimensions` values).
	pub(super) fn vector(&self, b: &mut Bits) -> Option<&[f32]> {
		let entry = self.scalar(b)? as usize;
		let vq = self.vq.as_ref()?;
		vq.get(entry * self.dimensions..(entry + 1) * self.dimensions)
	}
}

/// §3.2.1 vector lookup, for every entry (`lookup` 1: lattice; 2: one
/// multiplicand per value).
fn build_vq(
	multiplicands: &[u32],
	entries: usize,
	dimensions: usize,
	lookup: u32,
	minimum: f32,
	delta: f32,
	sequence: bool,
) -> Vec<f32> {
	let values = multiplicands.len();
	let mut vq = Vec::with_capacity(entries * dimensions);
	for entry in 0..entries {
		let mut last = 0f32;
		let mut divisor = 1usize;
		for i in 0..dimensions {
			let offset = if lookup == 1 {
				let o = (entry / divisor) % values;
				divisor = divisor.saturating_mul(values);
				o
			} else {
				entry * dimensions + i
			};
			let value = multiplicands[offset] as f32 * delta + minimum + last;
			if sequence {
				last = value;
			}
			vq.push(value);
		}
	}
	vq
}

/// Builds the fast table and the tree from the codewords.
fn build_decoder(lengths: &[u8], words: &[u32]) -> (Vec<Fast>, Vec<[i32; 2]>) {
	let mut tree: Vec<[i32; 2]> = vec![[ABSENT, ABSENT]];
	let mut fast = vec![Fast::Invalid; 1 << FAST_BITS];
	for (entry, (&len, &word)) in lengths.iter().zip(words).enumerate() {
		if len == 0 {
			continue;
		}
		let len32 = u32::from(len);
		// Codeword bits in stream (LSB-first) order.
		let reversed = word.reverse_bits() >> (32 - len32);
		if len32 <= FAST_BITS {
			let step = 1usize << len32;
			let mut i = reversed as usize;
			while i < fast.len() {
				fast[i] = Fast::Leaf { entry: entry as u32, len };
				i += step;
			}
			continue;
		}
		// Longer: the first FAST_BITS bits select a slot, the rest walk
		// the tree from that slot's node.
		let slot = (reversed & ((1 << FAST_BITS) - 1)) as usize;
		let mut node = match fast[slot] {
			Fast::Node(n) => n as usize,
			_ => {
				tree.push([ABSENT, ABSENT]);
				let n = tree.len() - 1;
				fast[slot] = Fast::Node(n as u32);
				n
			}
		};
		for depth in FAST_BITS..len32 {
			let bit = ((reversed >> depth) & 1) as usize;
			if depth + 1 == len32 {
				tree[node][bit] = -(entry as i32) - 1;
			} else {
				let next = tree[node][bit];
				node = if next >= 0 {
					next as usize
				} else {
					tree.push([ABSENT, ABSENT]);
					let n = tree.len() - 1;
					tree[node][bit] = n as i32;
					n
				};
			}
		}
	}
	(fast, tree)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn codewords_follow_the_specification_example() {
		// §3.2.1: lengths 2,4,4,4,4,2,3,3 → 00, 0100, 0101, 0110, 0111, 10,
		// 110, 111.
		let words = make_codewords(&[2, 4, 4, 4, 4, 2, 3, 3]).unwrap();
		assert_eq!(words, vec![0b00, 0b0100, 0b0101, 0b0110, 0b0111, 0b10, 0b110, 0b111]);
		// Over- and under-specified trees are rejected; one entry is fine.
		assert!(make_codewords(&[1, 1, 1]).is_none());
		assert!(make_codewords(&[1, 2]).is_none());
		assert!(make_codewords(&[1]).is_some());
	}

	#[test]
	fn lattice_size() {
		assert_eq!(lookup1_values(81, 4), 3);
		assert_eq!(lookup1_values(80, 4), 2);
		assert_eq!(lookup1_values(1, 8), 1);
		assert_eq!(lookup1_values(625, 2), 25);
	}

	#[test]
	fn float_unpack() {
		// 1.0: mantissa 1 << 20, exponent 788 + 1 - 21 ... encoded as libvorbis does.
		let encode = |mantissa: u32, exp: i32| ((exp + 788) as u32) << 21 | mantissa;
		assert_eq!(float32_unpack(encode(1, 0)), 1.0);
		assert_eq!(float32_unpack(encode(3, -1)), 1.5);
		assert_eq!(float32_unpack(encode(1, 0) | 0x8000_0000), -1.0);
	}

	#[test]
	fn decodes_short_and_long_codewords() {
		// Twelve entries: 1, 2, ..., 11, 11 bits (a complete tree).
		let mut lengths: Vec<u8> = (1..=11).collect();
		lengths.push(11);
		let words = make_codewords(&lengths).unwrap();
		let (fast, tree) = build_decoder(&lengths, &words);
		let book = Codebook { dimensions: 1, entries: 12, fast, tree, vq: None };
		// Write entries 0, 10, 11, 3 MSB-first into an LSB-first stream.
		let mut bits: Vec<bool> = Vec::new();
		for e in [0usize, 10, 11, 3] {
			for k in (0..lengths[e]).rev() {
				bits.push((words[e] >> k) & 1 == 1);
			}
		}
		let mut bytes = vec![0u8; bits.len().div_ceil(8)];
		for (i, &bit) in bits.iter().enumerate() {
			bytes[i / 8] |= u8::from(bit) << (i % 8);
		}
		let mut b = Bits::new(&bytes);
		let got: Vec<u32> = (0..4).map(|_| book.scalar(&mut b).unwrap()).collect();
		assert_eq!(got, vec![0, 10, 11, 3]);
	}
}
