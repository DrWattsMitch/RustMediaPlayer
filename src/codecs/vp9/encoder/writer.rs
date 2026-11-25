//! The boolean encoder (libvpx `vpx_dsp/bitwriter.c`), a bit-cost model
//! with the same interface, and the probability-update coding of the
//! compressed header (libvpx `vp9_subexp.c`).
//!
//! Everything that writes symbols is generic over [`Bits`], so the bits an
//! encoder decision is priced at are the bits the bitstream would carry.

use std::sync::LazyLock;

use super::super::Tree;

/// Something boolean symbols can be written to.
pub trait Bits {
	/// Writes `bit`, whose probability of being 0 is `prob / 256`.
	fn put(&mut self, bit: bool, prob: u8);

	/// An `n`-bit literal, MSB first (`vpx_write_literal`).
	fn literal(&mut self, v: u32, n: u32) {
		for i in (0..n).rev() {
			self.put((v >> i) & 1 == 1, 128);
		}
	}

	/// A tree-coded symbol (`vpx_write_tree`).
	fn tree(&mut self, tree: &Tree, probs: &[u8], symbol: u8) {
		let (path, len) = tree_path(tree, symbol);
		let mut i = 0usize;
		for k in (0..len).rev() {
			let b = (path >> k) & 1;
			self.put(b == 1, probs[i >> 1]);
			let next = tree[i + b as usize];
			if next > 0 {
				i = next as usize;
			}
		}
	}
}

/// Branch decisions from the root of `tree` to the leaf of `symbol`, as
/// (bits MSB first, count).
pub fn tree_path(tree: &Tree, symbol: u8) -> (u32, u32) {
	fn go(tree: &Tree, i: usize, symbol: i8, path: u32, len: u32) -> Option<(u32, u32)> {
		for b in 0..2 {
			let node = tree[i + b];
			let p = (path << 1) | b as u32;
			if node <= 0 {
				if -node == symbol {
					return Some((p, len + 1));
				}
			} else if let Some(found) = go(tree, node as usize, symbol, p, len + 1) {
				return Some(found);
			}
		}
		None
	}
	go(tree, 0, symbol as i8, 0, 0).unwrap_or((0, 0))
}

/// libvpx's `vpx_writer`.
#[derive(Debug, Clone)]
pub struct BoolEncoder {
	low: u32,
	range: u32,
	count: i32,
	out: Vec<u8>,
}

impl BoolEncoder {
	/// Starts a buffer with its (zero) marker bit (`vpx_start_encode`).
	pub fn new() -> Self {
		let mut w = Self { low: 0, range: 255, count: -24, out: Vec::new() };
		w.put(false, 128);
		w
	}

	/// Flushes the coder (`vpx_stop_encode`).
	pub fn finish(mut self) -> Vec<u8> {
		for _ in 0..32 {
			self.put(false, 128);
		}
		// No ambiguity with a superframe index marker byte.
		if self.out.last().is_some_and(|&b| b & 0xe0 == 0xc0) {
			self.out.push(0);
		}
		self.out
	}
}

impl Bits for BoolEncoder {
	#[inline]
	fn put(&mut self, bit: bool, prob: u8) {
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
				// Carry into the bytes already written.
				let mut x = self.out.len();
				while x > 0 && self.out[x - 1] == 0xFF {
					self.out[x - 1] = 0;
					x -= 1;
				}
				if x > 0 {
					self.out[x - 1] += 1;
				}
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
}

/// Discards everything (a pass that only counts symbols).
#[derive(Debug, Default)]
pub struct Null;

impl Bits for Null {
	#[inline]
	fn put(&mut self, _bit: bool, _prob: u8) {}
}

/// Cost of a bit in 1/256 bit, by probability of the coded value
/// (`vp9_prob_cost`, at 8 fractional bits).
static PROB_COST: LazyLock<[u16; 257]> = LazyLock::new(|| {
	let mut t = [0u16; 257];
	for (p, c) in t.iter_mut().enumerate().skip(1) {
		*c = (-(p as f64 / 256.0).log2() * 256.0).round() as u16;
	}
	t[0] = t[1];
	t
});

/// Cost of writing `bit` with probability `prob` (of a 0), 1/256 bit.
#[inline]
pub fn bit_cost(bit: bool, prob: u8) -> u32 {
	let p = if bit { 256 - usize::from(prob) } else { usize::from(prob) };
	u32::from(PROB_COST[p])
}

/// Sums the cost of what is written, in 1/256 bit.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cost(pub u32);

impl Bits for Cost {
	#[inline]
	fn put(&mut self, bit: bool, prob: u8) {
		self.0 += bit_cost(bit, prob);
	}

	#[inline]
	fn literal(&mut self, _v: u32, n: u32) {
		self.0 += 256 * n;
	}
}

/// Cost of a symbol of `tree`.
pub fn tree_cost(tree: &Tree, probs: &[u8], symbol: u8) -> u32 {
	let mut c = Cost(0);
	c.tree(tree, probs, symbol);
	c.0
}

// --- Probability updates (vp9_subexp.c) ---

/// The new probability a coded delta `d` gives from `old`, exactly as the
/// decoder computes it (`probs::diff_update`).
fn apply_delta(old: u8, d: u32) -> u8 {
	let mut r = old;
	super::super::probs::apply_diff(&mut r, d);
	r
}

/// The delta that turns `old` into `new` (`remap_prob`), found by trying
/// every code: the table is small and this is exactly the decoder's map.
fn delta_for(old: u8, new: u8) -> Option<u32> {
	(0..254u32).find(|&d| apply_delta(old, d) == new)
}

/// `encode_term_subexp`.
fn write_subexp<B: Bits>(w: &mut B, d: u32) {
	if d < 16 {
		w.put(false, 128);
		w.literal(d, 4);
	} else if d < 32 {
		w.put(true, 128);
		w.put(false, 128);
		w.literal(d - 16, 4);
	} else if d < 64 {
		w.put(true, 128);
		w.put(true, 128);
		w.put(false, 128);
		w.literal(d - 32, 5);
	} else {
		w.put(true, 128);
		w.put(true, 128);
		w.put(true, 128);
		let v = d - 64;
		const M: u32 = 65;
		if v < M {
			w.literal(v, 7);
		} else {
			w.literal(M + ((v - M) >> 1), 7);
			w.literal((v - M) & 1, 1);
		}
	}
}

/// Bits of the update code of `d` (after the flag).
fn subexp_bits(d: u32) -> u32 {
	if d < 16 {
		5
	} else if d < 32 {
		6
	} else if d < 64 {
		8
	} else if d - 64 < 65 {
		10
	} else {
		11
	}
}

/// The update flag (probability 252).
pub const UPDATE_PROB: u8 = 252;

/// Writes an optional update of `old` to `new` (`vp9_cond_prob_diff_update`
/// once decided). `new == old` writes the "no update" flag.
pub fn write_diff_update<B: Bits>(w: &mut B, old: u8, new: u8) {
	match (new != old).then(|| delta_for(old, new)).flatten() {
		Some(d) => {
			w.put(true, UPDATE_PROB);
			write_subexp(w, d);
		}
		None => w.put(false, UPDATE_PROB),
	}
}

/// Cost in 1/256 bit of `ct` (zeros, ones) at probability `p`.
pub fn branch_cost(ct: [u32; 2], p: u8) -> u64 {
	u64::from(ct[0]) * u64::from(bit_cost(false, p)) + u64::from(ct[1]) * u64::from(bit_cost(true, p))
}

/// The probability to code `ct` with when updating from `old`, if an update
/// saves bits overall (`vp9_prob_diff_update_savings_search`): the
/// candidates between the optimum and the old value are tried.
pub fn best_update(ct: [u32; 2], old: u8) -> Option<u8> {
	let den = ct[0] + ct[1];
	if den == 0 {
		return None;
	}
	let target = ((u64::from(ct[0]) * 256 + u64::from(den / 2)) / u64::from(den)).clamp(1, 255) as u8;
	let old_cost = branch_cost(ct, old) + u64::from(bit_cost(false, UPDATE_PROB));
	let mut best: Option<(u64, u8)> = None;
	let (lo, hi) = if target < old { (target, old) } else { (old, target) };
	for p in lo..=hi {
		if p == old {
			continue;
		}
		let Some(d) = delta_for(old, p) else { continue };
		let cost =
			branch_cost(ct, p) + u64::from(bit_cost(true, UPDATE_PROB)) + 256 * u64::from(subexp_bits(d));
		if cost < old_cost && best.is_none_or(|(c, _)| cost < c) {
			best = Some((cost, p));
		}
	}
	best.map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
	use super::super::super::probs;
	use super::super::super::range::BoolDecoder;
	use super::*;

	#[test]
	fn symbols_round_trip_through_the_decoder() {
		let mut w = BoolEncoder::new();
		let symbols: Vec<(bool, u8)> =
			(0..5000u32).map(|i| ((i * 7919) % 7 < 3, ((i * 31) % 255 + 1) as u8)).collect();
		for &(b, p) in &symbols {
			w.put(b, p);
		}
		w.literal(0x2A5, 10);
		w.tree(&super::super::super::INTRA_MODE_TREE, &[100; 9], super::super::super::pm::D63);
		let data = w.finish();
		let mut r = BoolDecoder::new(&data).unwrap();
		for (i, &(b, p)) in symbols.iter().enumerate() {
			assert_eq!(r.read(p), b, "symbol {i}");
		}
		assert_eq!(r.literal(10), 0x2A5);
		assert_eq!(
			r.tree(&super::super::super::INTRA_MODE_TREE, &[100; 9]),
			super::super::super::pm::D63
		);
		assert!(!r.overrun());
	}

	#[test]
	fn probability_updates_decode_to_the_new_value() {
		for old in 1..=255u8 {
			for new in [1u8, 2, 17, 64, 127, 128, 129, 200, 254, 255] {
				let mut w = BoolEncoder::new();
				write_diff_update(&mut w, old, new);
				let data = w.finish();
				let mut r = BoolDecoder::new(&data).unwrap();
				let mut p = old;
				probs::diff_update(&mut r, &mut p);
				// Every probability but the extremes is reachable.
				if delta_for(old, new).is_some() || new == old {
					assert_eq!(p, new, "{old} -> {new}");
				}
			}
		}
	}

	#[test]
	fn costs_follow_the_probability() {
		assert_eq!(bit_cost(false, 128), 256);
		assert!(bit_cost(false, 250) < 10);
		assert!(bit_cost(true, 250) > 1300);
		assert_eq!(tree_path(&super::super::super::PARTITION_TREE, 0), (0, 1));
		assert_eq!(tree_path(&super::super::super::PARTITION_TREE, 3), (0b111, 3));
	}
}
