//! HEVC CABAC decoding engine (ITU-T H.265 §9.3.4.3; FFmpeg `cabac.h`,
//! `cabac_functions.h`, `hevc/cabac.c`).
//!
//! The engine keeps `low` scaled by 2^17 and refills two bytes at a time,
//! as FFmpeg's does, so most bins cost no bitstream read. The byte position
//! is recoverable after a terminate bin ([`Cabac::reinit_after_terminate`]),
//! which is how PCM samples and new tiles / WPP rows are located without
//! entry point offsets.

use super::tables::{INIT_VALUES, NUM_CTX};
use crate::codecs::h264::tables::{RANGE_LPS, TRANS_IDX};
use std::sync::OnceLock;

const CABAC_BITS: u32 = 16;
const CABAC_MASK: u32 = (1 << CABAC_BITS) - 1;

struct EngineTables {
	/// `lps_range[(range & 0xC0) * 2 + state]`, state = pStateIdx << 1 | valMps.
	lps_range: [u8; 512],
	/// Next state after a bin: `mlps[128 + state]` after an MPS,
	/// `mlps[127 - state]` after an LPS (FFmpeg `ff_h264_mlps_state`).
	mlps: [u8; 256],
	norm_shift: [u8; 512],
}

fn tables() -> &'static EngineTables {
	static T: OnceLock<EngineTables> = OnceLock::new();
	T.get_or_init(|| {
		let mut lps_range = [0u8; 512];
		for (i, row) in RANGE_LPS.iter().enumerate() {
			for (q, &v) in row.iter().enumerate() {
				lps_range[q * 128 + 2 * i] = v;
				lps_range[q * 128 + 2 * i + 1] = v;
			}
		}
		let mut mlps = [0u8; 256];
		for (i, &(lps, mps)) in TRANS_IDX.iter().enumerate() {
			mlps[128 + 2 * i] = 2 * mps;
			mlps[128 + 2 * i + 1] = 2 * mps + 1;
			if i > 0 {
				mlps[127 - 2 * i] = 2 * lps;
				mlps[127 - 2 * i - 1] = 2 * lps + 1;
			} else {
				mlps[127] = 1;
				mlps[126] = 0;
			}
		}
		let mut norm_shift = [0u8; 512];
		norm_shift[0] = 9;
		for (i, v) in norm_shift.iter_mut().enumerate().skip(1) {
			*v = (8 - (31 - (i as u32).leading_zeros())) as u8;
		}
		EngineTables { lps_range, mlps, norm_shift }
	})
}

/// Context states plus the persistent Rice statistics (saved together for
/// WPP).
#[derive(Clone)]
pub struct Contexts {
	pub states: [u8; NUM_CTX],
}

impl Contexts {
	/// 9.3.2.2: initialise every context for `init_type` and `qp`.
	pub fn new(init_type: usize, qp: i32) -> Self {
		let mut states = [0u8; NUM_CTX];
		let qp = qp.clamp(0, 51);
		for (s, &iv) in states.iter_mut().zip(INIT_VALUES[init_type].iter()) {
			let m = i32::from(iv >> 4) * 5 - 45;
			let n = (i32::from(iv & 15) << 3) - 16;
			let mut pre = 2 * (((m * qp) >> 4) + n) - 127;
			pre ^= pre >> 31;
			if pre > 124 {
				pre = 124 + (pre & 1);
			}
			*s = pre as u8;
		}
		Self { states }
	}
}

pub struct Cabac<'a> {
	data: &'a [u8],
	/// Next byte to load.
	pos: usize,
	low: u32,
	range: u32,
	pub ctx: Contexts,
}

impl<'a> Cabac<'a> {
	/// Starts the arithmetic decoder at byte `start` of `data`.
	pub fn new(data: &'a [u8], start: usize, ctx: Contexts) -> Self {
		let mut c = Self { data, pos: start, low: 0, range: 0x1FE, ctx };
		c.init_engine(start);
		c
	}

	#[inline]
	fn byte(&self, at: usize) -> u32 {
		self.data.get(at).copied().map_or(0, u32::from)
	}

	fn init_engine(&mut self, start: usize) {
		self.pos = start;
		self.low = (self.byte(start) << 18) | (self.byte(start + 1) << 10) | (1 << 9);
		self.pos += 2;
		self.range = 0x1FE;
	}

	/// `true` once the decoder has consumed bytes past the end of the data.
	pub fn overrun(&self) -> bool {
		self.pos > self.data.len() + 2
	}

	#[inline]
	fn refill(&mut self) {
		self.low = self.low.wrapping_add((self.byte(self.pos) << 9) + (self.byte(self.pos + 1) << 1));
		self.low = self.low.wrapping_sub(CABAC_MASK);
		self.pos += 2;
	}

	#[inline]
	fn refill2(&mut self) {
		let x = self.low ^ (self.low.wrapping_sub(1));
		let t = tables();
		let i = 7 - i32::from(t.norm_shift[(x >> (CABAC_BITS - 1)) as usize & 511]);
		let fill =
			((self.byte(self.pos) << 9) + (self.byte(self.pos + 1) << 1)).wrapping_sub(CABAC_MASK);
		self.low = self.low.wrapping_add(fill << i);
		self.pos += 2;
	}

	/// DecodeDecision with context `idx`.
	#[inline]
	pub fn decode(&mut self, idx: usize) -> bool {
		let t = tables();
		let s = u32::from(self.ctx.states[idx]);
		let lps = u32::from(t.lps_range[(2 * (self.range & 0xC0) + s) as usize]);
		self.range -= lps;
		let scaled = self.range << (CABAC_BITS + 1);
		let s = if self.low < scaled {
			// MPS path.
			self.ctx.states[idx] = t.mlps[(128 + s) as usize];
			s
		} else {
			self.low -= scaled;
			self.range = lps;
			self.ctx.states[idx] = t.mlps[(127 - s) as usize];
			s ^ 1
		};
		let shift = u32::from(t.norm_shift[self.range as usize]);
		self.range <<= shift;
		self.low <<= shift;
		if self.low & CABAC_MASK == 0 {
			self.refill2();
		}
		s & 1 != 0
	}

	/// DecodeBypass.
	#[inline]
	pub fn bypass(&mut self) -> bool {
		self.low <<= 1;
		if self.low & CABAC_MASK == 0 {
			self.refill();
		}
		let r = self.range << (CABAC_BITS + 1);
		if self.low < r {
			false
		} else {
			self.low -= r;
			true
		}
	}

	/// `n` bypass bins, MSB first.
	#[inline]
	pub fn bypass_bits(&mut self, n: u32) -> u32 {
		let mut v = 0;
		for _ in 0..n {
			v = (v << 1) | u32::from(self.bypass());
		}
		v
	}

	/// DecodeTerminate.
	#[inline]
	pub fn terminate(&mut self) -> bool {
		self.range -= 2;
		if self.low < self.range << (CABAC_BITS + 1) {
			let shift = (self.range.wrapping_sub(0x100)) >> 31;
			self.range <<= shift;
			self.low <<= shift;
			if self.low & CABAC_MASK == 0 {
				self.refill();
			}
			false
		} else {
			true
		}
	}

	/// Byte position of the bitstream right after a terminate bin that
	/// returned 1 (FFmpeg `skip_bytes(c, 0)`).
	pub fn byte_position(&self) -> usize {
		let mut p = self.pos;
		if self.low & 1 != 0 {
			p -= 1;
		}
		if self.low & 0x1FF != 0 {
			p -= 1;
		}
		p
	}

	/// Restarts the engine at the byte after the last terminate (new
	/// substream: tile or WPP row), keeping the contexts.
	pub fn reinit_after_terminate(&mut self) {
		let p = self.byte_position();
		self.init_engine(p);
	}

	/// Reads `n` raw bytes after a terminate (PCM samples), returning them
	/// and restarting the engine after them.
	pub fn take_raw(&mut self, n: usize) -> &'a [u8] {
		let p = self.byte_position().min(self.data.len());
		let end = (p + n).min(self.data.len());
		let raw = &self.data[p..end];
		self.init_engine(p + n);
		raw
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn init_states_are_in_range() {
		for t in 0..3 {
			for qp in [0, 26, 51] {
				let c = Contexts::new(t, qp);
				assert!(c.states.iter().all(|&s| s < 128));
			}
		}
	}

	#[test]
	fn bypass_reads_raw_bits_after_init() {
		// With range 510 the first bypass bins follow the data's top bits
		// closely; this just checks the engine runs off the end safely.
		let data = [0xA5u8; 8];
		let mut c = Cabac::new(&data, 0, Contexts::new(0, 26));
		for _ in 0..200 {
			c.bypass();
		}
		assert!(c.overrun());
	}
}
