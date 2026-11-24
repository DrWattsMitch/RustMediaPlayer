//! CABAC arithmetic encoding (ISO/IEC 14496-10 §9.3.4) and a bit-cost
//! estimator with the same interface.
//!
//! [`CabacWriter`] is the bit-exact encoder of §9.3.4.2 (`codILow`,
//! `codIRange`, `bitsOutstanding`), the inverse of the decoder's engine
//! in `h264::cabac`. [`CabacCost`] runs the same context state machine but
//! only accumulates `-log2(p)` of every bin (in 1/256 bit), which is how
//! the encoder prices a macroblock candidate or a trellis path without
//! producing bits (x264 `cabac_size_decision`, `x264_cabac_entropy`).
//!
//! Both implement [`Bins`]; every syntax element writer is generic over
//! it, so the bits that are priced are exactly the bits that are sent.
//!
//! FFmpeg / x264 counterparts: `x264/common/cabac.c` (`cabac_encode_*`,
//! `cabac_encode_flush`), `x264/encoder/cabac.c` (residual blocks).

use crate::codecs::bitstream::BitWriter;
use crate::codecs::h264::cabac::{BlockCat, NUM_CTX, ctx};
use crate::codecs::h264::consts::{LAST_8X8_CTX, SIG_8X8_CTX};
use crate::codecs::h264::tables::{CABAC_INIT, RANGE_LPS, TRANS_IDX};
use std::sync::OnceLock;

const CBF_OFFSET: [usize; 5] = [0, 4, 8, 12, 16];
const SIG_OFFSET: [usize; 5] = [0, 15, 29, 44, 47];
const ABS_OFFSET: [usize; 5] = [0, 10, 20, 30, 39];

/// Context states at the start of a slice (9.3.1.1), packed as the
/// decoder packs them: `pStateIdx << 1 | valMPS`.
pub fn init_states(intra_slice: bool, init_idc: usize, qp: i32) -> [u8; NUM_CTX] {
	let column = if intra_slice { 0 } else { init_idc + 1 };
	let qp = qp.clamp(0, 51);
	let mut states = [0u8; NUM_CTX];
	for (state, init) in states.iter_mut().zip(CABAC_INIT.iter()) {
		let (m, n) = init[column];
		let pre = (((i32::from(m) * qp) >> 4) + i32::from(n)).clamp(1, 126);
		*state = if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 };
	}
	states
}

/// Next state after coding `bin` in a context in state `s`.
#[inline]
fn next_state(s: u8, bin: bool) -> u8 {
	let state = usize::from(s >> 1);
	let mut mps = s & 1;
	if u8::from(bin) == mps {
		(TRANS_IDX[state].1 << 1) | mps
	} else {
		if state == 0 {
			mps ^= 1;
		}
		(TRANS_IDX[state].0 << 1) | mps
	}
}

/// Cost in 1/256 bit of coding the MPS / LPS in each probability state.
fn entropy_table() -> &'static [[u32; 2]; 64] {
	static T: OnceLock<[[u32; 2]; 64]> = OnceLock::new();
	T.get_or_init(|| {
		let mut t = [[0u32; 2]; 64];
		// Midpoints of the four codIRange quarters the LPS table is indexed by.
		let mids = [288.0, 352.0, 416.0, 479.0];
		for (s, slot) in t.iter_mut().enumerate() {
			let p_lps: f64 = (0..4).map(|q| f64::from(RANGE_LPS[s][q]) / mids[q]).sum::<f64>() / 4.0;
			let p_lps = p_lps.clamp(1e-4, 0.5);
			slot[0] = (-(1.0 - p_lps).log2() * 256.0).round() as u32;
			slot[1] = (-p_lps.log2() * 256.0).round() as u32;
		}
		t
	})
}

/// Cost in 1/256 bit of coding `bin` with a context in state `s`.
#[inline]
pub fn bin_cost(s: u8, bin: bool) -> u32 {
	let e = &entropy_table()[usize::from(s >> 1)];
	if u8::from(bin) == (s & 1) { e[0] } else { e[1] }
}

/// A sink for CABAC bins: the real encoder or the cost estimator.
pub trait Bins {
	fn states(&self) -> &[u8; NUM_CTX];
	fn decision(&mut self, ctx_idx: usize, bin: bool);
	fn bypass(&mut self, bin: bool);
	fn terminate(&mut self, bin: bool);
}

/// The arithmetic encoder of §9.3.4.2, writing into a [`BitWriter`].
#[derive(Debug, Clone)]
pub struct CabacWriter {
	low: u32,
	range: u32,
	outstanding: u32,
	first: bool,
	pub states: [u8; NUM_CTX],
	pub w: BitWriter,
}

impl CabacWriter {
	/// Starts slice data: `w` holds the slice header; it is padded with
	/// `cabac_alignment_one_bit`s first (7.3.4).
	pub fn new(mut w: BitWriter, states: [u8; NUM_CTX]) -> Self {
		while !w.is_aligned() {
			w.bit(true);
		}
		Self { low: 0, range: 510, outstanding: 0, first: true, states, w }
	}

	#[inline]
	fn put_bit(&mut self, b: bool) {
		if self.first {
			self.first = false;
		} else {
			self.w.bit(b);
		}
		while self.outstanding > 0 {
			self.w.bit(!b);
			self.outstanding -= 1;
		}
	}

	#[inline]
	fn renorm(&mut self) {
		while self.range < 256 {
			if self.low < 256 {
				self.put_bit(false);
			} else if self.low >= 512 {
				self.low -= 512;
				self.put_bit(true);
			} else {
				self.low -= 256;
				self.outstanding += 1;
			}
			self.range <<= 1;
			self.low <<= 1;
		}
	}

	/// Codes `end_of_slice_flag = 1`, flushes (EncodeFlush, whose last bit
	/// is the `rbsp_stop_one_bit`) and returns the byte-aligned RBSP.
	pub fn finish(mut self) -> Vec<u8> {
		self.terminate(true);
		self.w.finish()
	}
}

impl Bins for CabacWriter {
	fn states(&self) -> &[u8; NUM_CTX] {
		&self.states
	}

	#[inline]
	fn decision(&mut self, ctx_idx: usize, bin: bool) {
		let s = self.states[ctx_idx];
		let state = usize::from(s >> 1);
		let lps = u32::from(RANGE_LPS[state][((self.range >> 6) & 3) as usize]);
		self.range -= lps;
		if u8::from(bin) != (s & 1) {
			self.low += self.range;
			self.range = lps;
		}
		self.states[ctx_idx] = next_state(s, bin);
		self.renorm();
	}

	#[inline]
	fn bypass(&mut self, bin: bool) {
		self.low <<= 1;
		if bin {
			self.low += self.range;
		}
		if self.low >= 1024 {
			self.put_bit(true);
			self.low -= 1024;
		} else if self.low < 512 {
			self.put_bit(false);
		} else {
			self.low -= 512;
			self.outstanding += 1;
		}
	}

	fn terminate(&mut self, bin: bool) {
		self.range -= 2;
		if bin {
			self.low += self.range;
			// EncodeFlush (9.3.4.5).
			self.range = 2;
			self.renorm();
			self.put_bit((self.low >> 9) & 1 != 0);
			self.w.put(((self.low >> 7) & 3) | 1, 2);
		} else {
			self.renorm();
		}
	}
}

/// Context-state machine that only accumulates the cost of the bins.
#[derive(Debug, Clone)]
pub struct CabacCost {
	pub states: [u8; NUM_CTX],
	/// Accumulated cost in 1/256 bit.
	pub cost: u32,
}

impl CabacCost {
	pub fn new(states: &[u8; NUM_CTX]) -> Self {
		Self { states: *states, cost: 0 }
	}

	/// Cost in (fractional) bits.
	pub fn bits(&self) -> f64 {
		f64::from(self.cost) / 256.0
	}
}

impl Bins for CabacCost {
	fn states(&self) -> &[u8; NUM_CTX] {
		&self.states
	}

	#[inline]
	fn decision(&mut self, ctx_idx: usize, bin: bool) {
		let s = self.states[ctx_idx];
		self.cost += bin_cost(s, bin);
		self.states[ctx_idx] = next_state(s, bin);
	}

	#[inline]
	fn bypass(&mut self, _bin: bool) {
		self.cost += 256;
	}

	#[inline]
	fn terminate(&mut self, _bin: bool) {}
}

/// k-th order Exp-Golomb suffix in bypass bins (UEGk, 9.3.2.3).
pub fn exp_golomb_bypass<B: Bins>(b: &mut B, mut v: u32, mut k: u32) {
	while v >= 1 << k {
		b.bypass(true);
		v -= 1 << k;
		k += 1;
	}
	b.bypass(false);
	while k > 0 {
		k -= 1;
		b.bypass((v >> k) & 1 != 0);
	}
}

/// Context bases of a residual block category: (significance, last,
/// level).
pub fn residual_bases(cat: BlockCat) -> (usize, usize, usize) {
	let c = cat as usize;
	if cat == BlockCat::Luma8x8 {
		(ctx::SIG_8X8, ctx::LAST_8X8, ctx::ABS_8X8)
	} else {
		(ctx::SIG + SIG_OFFSET[c], ctx::LAST + SIG_OFFSET[c], ctx::ABS + ABS_OFFSET[c])
	}
}

/// coded_block_flag context index for a category and ctxIdxInc.
pub fn cbf_ctx(cat: BlockCat, inc: usize) -> usize {
	ctx::CBF + CBF_OFFSET[(cat as usize).min(4)] + inc
}

/// significant_coeff_flag / last_significant_coeff_flag ctxIdxInc of
/// scan position `i`.
#[inline]
pub fn sig_last_inc(cat: BlockCat, i: usize) -> (usize, usize) {
	match cat {
		BlockCat::ChromaDc => (i.min(2), i.min(2)),
		BlockCat::Luma8x8 => (usize::from(SIG_8X8_CTX[i]), usize::from(LAST_8X8_CTX[i])),
		_ => (i, i),
	}
}

/// Codes one residual block (§7.3.5.3.3 with CABAC, the decoder's
/// `Cabac::residual` inverted). `levels` are in scan order, `levels.len()`
/// is maxNumCoeff. `cbf_inc` is `None` where coded_block_flag is not sent
/// (8x8 luma blocks); such blocks must have a non-zero level.
pub fn residual<B: Bins>(b: &mut B, cat: BlockCat, cbf_inc: Option<usize>, levels: &[i32]) {
	let max = levels.len();
	let last = levels.iter().rposition(|&v| v != 0);
	if let Some(inc) = cbf_inc {
		b.decision(cbf_ctx(cat, inc), last.is_some());
	}
	let Some(last) = last else { return };
	let (sig_base, last_base, abs_base) = residual_bases(cat);
	for i in 0..max - 1 {
		let (si, li) = sig_last_inc(cat, i);
		let nz = levels[i] != 0;
		b.decision(sig_base + si, nz);
		if nz {
			b.decision(last_base + li, i == last);
			if i == last {
				break;
			}
		}
	}
	let gt1_limit = if cat == BlockCat::ChromaDc { 3 } else { 4 };
	let (mut gt1, mut eq1) = (0usize, 0usize);
	for &v in levels[..=last].iter().rev() {
		if v == 0 {
			continue;
		}
		let minus1 = v.unsigned_abs() - 1;
		let first = abs_base + if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
		b.decision(first, minus1 > 0);
		if minus1 > 0 {
			let rest = abs_base + 5 + gt1.min(gt1_limit);
			for _ in 1..minus1.min(14) {
				b.decision(rest, true);
			}
			if minus1 < 14 {
				b.decision(rest, false);
			} else {
				exp_golomb_bypass(b, minus1 - 14, 0);
			}
			gt1 += 1;
		} else {
			eq1 += 1;
		}
		b.bypass(v < 0);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::h264::cabac::Cabac;
	use crate::codecs::h264::consts::{ZIGZAG_4X4, ZIGZAG_8X8};
	use crate::codecs::h264::slice::SliceType;

	/// Pseudo-random bins and residual blocks round-trip through the
	/// decoder's engine.
	#[test]
	fn engine_round_trips_through_the_decoder() {
		let qp = 30;
		let states = init_states(false, 1, qp);
		let mut w = BitWriter::new();
		w.put(0b101, 3); // a fake header, not byte aligned
		let mut enc = CabacWriter::new(w, states);
		let mut seed = 12345u32;
		let mut rnd = || {
			seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
			seed >> 16
		};
		let mut bins = Vec::new();
		let mut blocks = Vec::new();
		for n in 0..400 {
			let ctx_idx = (rnd() % 60) as usize;
			let bin = rnd() % 5 == 0;
			let bypass = rnd() % 4 == 0;
			bins.push((ctx_idx, bin, bypass));
			if bypass {
				enc.bypass(bin);
			} else {
				enc.decision(ctx_idx, bin);
			}
			if n % 20 == 0 {
				let cat = [BlockCat::Luma4x4, BlockCat::ChromaDc, BlockCat::Luma8x8][n / 20 % 3];
				let max = match cat {
					BlockCat::ChromaDc => 4,
					BlockCat::Luma8x8 => 64,
					_ => 16,
				};
				let mut levels = vec![0i32; max];
				for (i, l) in levels.iter_mut().enumerate() {
					if rnd() % 3 == 0 || i == 0 {
						*l = (rnd() % 40) as i32 - 20;
					}
				}
				if levels.iter().all(|&v| v == 0) {
					levels[0] = 3;
				}
				let inc = (cat != BlockCat::Luma8x8).then_some((rnd() % 4) as usize);
				residual(&mut enc, cat, inc, &levels);
				blocks.push((n, cat, inc, levels));
			}
			if n % 50 == 49 {
				enc.terminate(false);
			}
		}
		let data = enc.finish();
		let mut dec = Cabac::new(&data, 3, SliceType::P, 1, qp);
		let mut bi = blocks.iter().peekable();
		for (n, &(ctx_idx, bin, bypass)) in bins.iter().enumerate() {
			let got = if bypass { dec.bypass() } else { dec.decision(ctx_idx) };
			assert_eq!(got, bin, "bin {n}");
			if let Some((_, cat, inc, levels)) = bi.next_if(|b| b.0 == n) {
				let max = levels.len();
				let scan: Vec<u8> = match max {
					4 => vec![0, 1, 2, 3],
					64 => ZIGZAG_8X8.to_vec(),
					_ => ZIGZAG_4X4.to_vec(),
				};
				let mut out = vec![0i32; max];
				let count = dec.residual(*cat, *inc, &scan, 0, max, &mut out);
				let want: Vec<i32> = (0..max).map(|k| levels[k]).collect();
				let got: Vec<i32> = (0..max).map(|k| out[usize::from(scan[k])]).collect();
				assert_eq!(got, want, "block at bin {n}");
				assert_eq!(usize::from(count), levels.iter().filter(|&&v| v != 0).count());
			}
			if n % 50 == 49 {
				assert!(!dec.terminate());
			}
		}
		assert!(dec.terminate(), "end of slice");
	}

	#[test]
	fn cost_tracks_probability() {
		// A context at its most probable state costs well under a bit for
		// the MPS and several bits for the LPS.
		let s = (62u8 << 1) | 1;
		assert!(bin_cost(s, true) < 16);
		assert!(bin_cost(s, false) > 256 * 5);
		let s0 = 0u8;
		assert!((bin_cost(s0, false) as i32 - 256).abs() < 40);
	}
}
