//! CABAC arithmetic decoding engine and residual decoding (ISO/IEC 14496-10
//! §9.3; FFmpeg `cabac.c`, `h264_cabac.c`).
//!
//! The engine is the bit-exact form of §9.3.3.2: a 9-bit `codIRange`
//! and `codIOffset`, renormalised one bit at a time (several bits per
//! read). Context states are stored as `pStateIdx << 1 | valMPS`.

use super::slice::SliceType;
use super::tables::{CABAC_INIT, RANGE_LPS, TRANS_IDX};
use crate::codecs::bitstream::BitReader;

pub const NUM_CTX: usize = 460;
/// Coefficient levels beyond this cannot occur in conforming 8-bit streams
/// (the spec bounds them to 16 bits); damaged data is clamped here.
pub const MAX_LEVEL: u32 = 1 << 16;

/// ctxIdxOffsets (Table 9-34).
pub mod ctx {
	pub const MB_TYPE_I: usize = 3;
	pub const MB_SKIP_P: usize = 11;
	pub const MB_TYPE_P: usize = 14;
	pub const MB_TYPE_P_INTRA: usize = 17;
	pub const SUB_MB_P: usize = 21;
	pub const MB_SKIP_B: usize = 24;
	pub const MB_TYPE_B: usize = 27;
	pub const MB_TYPE_B_INTRA: usize = 32;
	pub const SUB_MB_B: usize = 36;
	pub const MVD_X: usize = 40;
	pub const MVD_Y: usize = 47;
	pub const REF_IDX: usize = 54;
	pub const QP_DELTA: usize = 60;
	pub const CHROMA_PRED: usize = 64;
	pub const PREV_INTRA: usize = 68;
	pub const REM_INTRA: usize = 69;
	pub const CBP_LUMA: usize = 73;
	pub const CBP_CHROMA: usize = 77;
	pub const CBF: usize = 85;
	pub const SIG: usize = 105;
	pub const LAST: usize = 166;
	pub const ABS: usize = 227;
	pub const TRANSFORM_8X8: usize = 399;
	pub const SIG_8X8: usize = 402;
	pub const LAST_8X8: usize = 417;
	pub const ABS_8X8: usize = 426;
}

/// Residual block categories (`ctxBlockCat`, Table 9-42).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockCat {
	LumaDc = 0,
	LumaAc = 1,
	Luma4x4 = 2,
	ChromaDc = 3,
	ChromaAc = 4,
	Luma8x8 = 5,
}

const CBF_OFFSET: [usize; 5] = [0, 4, 8, 12, 16];
const SIG_OFFSET: [usize; 5] = [0, 15, 29, 44, 47];
const ABS_OFFSET: [usize; 5] = [0, 10, 20, 30, 39];

pub struct Cabac<'a> {
	bs: BitReader<'a>,
	range: u32,
	offset: u32,
	states: [u8; NUM_CTX],
}

impl<'a> Cabac<'a> {
	/// Initialises contexts (9.3.1.1) and the decoding engine (9.3.1.2) at
	/// the byte-aligned start of slice data.
	pub fn new(
		data: &'a [u8],
		bit_pos: usize,
		slice_type: SliceType,
		init_idc: usize,
		qp: i32,
	) -> Self {
		let mut bs = BitReader::new(data);
		bs.skip(bit_pos);
		bs.align();
		let column = if slice_type == SliceType::I { 0 } else { init_idc + 1 };
		let qp = qp.clamp(0, 51);
		let mut states = [0u8; NUM_CTX];
		for (state, init) in states.iter_mut().zip(CABAC_INIT.iter()) {
			let (m, n) = init[column];
			let pre = (((i32::from(m) * qp) >> 4) + i32::from(n)).clamp(1, 126);
			*state = if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 };
		}
		let offset = bs.bits(9);
		Self { bs, range: 510, offset, states }
	}

	/// `true` once the engine has read past the end of the slice data.
	pub fn overrun(&self) -> bool {
		self.bs.overrun()
	}

	#[inline]
	fn renorm(&mut self) {
		if self.range < 256 {
			let shift = self.range.leading_zeros() - 23;
			self.range <<= shift;
			self.offset = (self.offset << shift) | self.bs.bits(shift);
		}
	}

	/// DecodeDecision (9.3.3.2.1).
	#[inline]
	pub fn decision(&mut self, ctx_idx: usize) -> bool {
		let Some(&s) = self.states.get(ctx_idx) else {
			return false;
		};
		let state = usize::from(s >> 1);
		let mut mps = s & 1;
		let lps = u32::from(RANGE_LPS[state][((self.range >> 6) & 3) as usize]);
		self.range -= lps;
		let bin;
		let next;
		if self.offset >= self.range {
			bin = mps ^ 1;
			self.offset -= self.range;
			self.range = lps;
			if state == 0 {
				mps ^= 1;
			}
			next = TRANS_IDX[state].0;
		} else {
			bin = mps;
			next = TRANS_IDX[state].1;
		}
		self.states[ctx_idx] = (next << 1) | mps;
		self.renorm();
		bin != 0
	}

	/// DecodeBypass (9.3.3.2.3).
	#[inline]
	pub fn bypass(&mut self) -> bool {
		self.offset = (self.offset << 1) | self.bs.bits(1);
		if self.offset >= self.range {
			self.offset -= self.range;
			true
		} else {
			false
		}
	}

	/// DecodeTerminate (9.3.3.2.2).
	pub fn terminate(&mut self) -> bool {
		self.range -= 2;
		if self.offset >= self.range {
			true
		} else {
			self.renorm();
			false
		}
	}

	/// Bit position of the underlying reader (for I_PCM samples).
	pub fn reader(&mut self) -> &mut BitReader<'a> {
		&mut self.bs
	}

	/// Re-initialises the decoding engine after I_PCM samples (9.3.1.2).
	pub fn restart(&mut self) {
		self.range = 510;
		self.offset = self.bs.bits(9);
	}

	/// k-th order Exp-Golomb bypass suffix (UEGk, 9.3.2.3).
	pub fn exp_golomb_bypass(&mut self, mut k: u32) -> u32 {
		let mut value = 0u32;
		while self.bypass() {
			value = value.saturating_add(1 << k);
			k += 1;
			if k >= 31 {
				return value;
			}
		}
		while k > 0 {
			k -= 1;
			value = value.saturating_add(u32::from(self.bypass()) << k);
		}
		value
	}

	/// Truncated-unary value with a first-bin context and a shared context
	/// for the remaining bins.
	pub fn unary(&mut self, first: usize, rest: usize, max: u32) -> u32 {
		if !self.decision(first) {
			return 0;
		}
		let mut v = 1;
		while v < max && self.decision(rest) {
			v += 1;
		}
		v
	}

	/// Unary value whose first, second and remaining bins use separate
	/// contexts (ref_idx, mb_qp_delta).
	pub fn unary3(&mut self, first: usize, second: usize, rest: usize, max: u32) -> u32 {
		if !self.decision(first) {
			return 0;
		}
		if max <= 1 || !self.decision(second) {
			return 1;
		}
		let mut v = 2;
		while v < max && self.decision(rest) {
			v += 1;
		}
		v
	}

	/// Decodes one residual block (§7.3.5.3.3 with CABAC).
	///
	/// `cbf_inc` is the coded_block_flag ctxIdxInc, or `None` when the flag
	/// is not coded (8x8 luma blocks outside 4:4:4). Levels are stored at
	/// `out[scan[start + i]]`. Returns the number of non-zero coefficients.
	pub fn residual(
		&mut self,
		cat: BlockCat,
		cbf_inc: Option<usize>,
		scan: &[u8],
		start: usize,
		max: usize,
		out: &mut [i32],
	) -> u8 {
		let c = cat as usize;
		if let Some(inc) = cbf_inc {
			if !self.decision(ctx::CBF + CBF_OFFSET[c.min(4)] + inc) {
				return 0;
			}
		}
		let (sig_base, last_base, abs_base) = if cat == BlockCat::Luma8x8 {
			(ctx::SIG_8X8, ctx::LAST_8X8, ctx::ABS_8X8)
		} else {
			(ctx::SIG + SIG_OFFSET[c], ctx::LAST + SIG_OFFSET[c], ctx::ABS + ABS_OFFSET[c])
		};
		let mut sig = [0u8; 64];
		let mut count = 0usize;
		let mut i = 0usize;
		while i + 1 < max {
			let (si, li) = match cat {
				BlockCat::ChromaDc => (i.min(2), i.min(2)),
				BlockCat::Luma8x8 => {
					(usize::from(super::consts::SIG_8X8_CTX[i]), usize::from(super::consts::LAST_8X8_CTX[i]))
				}
				_ => (i, i),
			};
			if self.decision(sig_base + si) {
				sig[count] = i as u8;
				count += 1;
				if self.decision(last_base + li) {
					break;
				}
			}
			i += 1;
		}
		if i + 1 == max {
			sig[count] = (max - 1) as u8;
			count += 1;
		}
		let gt1_limit = if cat == BlockCat::ChromaDc { 3 } else { 4 };
		let (mut gt1, mut eq1) = (0usize, 0usize);
		for &pos in sig[..count].iter().rev() {
			let first = abs_base + if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
			let mut minus1 = 0u32;
			if self.decision(first) {
				let rest = abs_base + 5 + gt1.min(gt1_limit);
				minus1 = 1;
				while minus1 < 14 && self.decision(rest) {
					minus1 += 1;
				}
				if minus1 == 14 {
					minus1 = minus1.saturating_add(self.exp_golomb_bypass(0));
				}
			}
			let magnitude = minus1.saturating_add(1).min(MAX_LEVEL) as i32;
			let level = if self.bypass() { -magnitude } else { magnitude };
			if magnitude == 1 {
				eq1 += 1;
			} else {
				gt1 += 1;
			}
			if let Some(&raster) = scan.get(start + usize::from(pos)) {
				if let Some(slot) = out.get_mut(usize::from(raster)) {
					*slot = level;
				}
			}
		}
		count as u8
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn contexts_initialise_in_range() {
		let c = Cabac::new(&[0xFF; 8], 0, SliceType::P, 1, 26);
		assert!(c.states.iter().all(|&s| (s >> 1) <= 62));
	}
}
