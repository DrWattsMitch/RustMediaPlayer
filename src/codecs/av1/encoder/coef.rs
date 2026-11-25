//! Block contexts and coefficient coding.
//!
//! [`Ctx`] holds what rav1d keeps per 4x4 column above and per 4x4 row to
//! the left of the block being coded (`BlockContext`), for the syntax this
//! encoder writes. [`write_coefs`] is rav1d's `decode_coefs` inverted:
//! all-zero flag, transform type, end of block, base and range tokens in
//! reverse scan order (with rav1d's level-context layout), then signs and
//! Golomb remainders in scan order. The same function prices a block with
//! a [`Cost`](super::msac::Cost) writer.

use super::cdf::CdfContext;
use super::itx::{ADST_ADST, ADST_DCT, DCT_ADST, DCT_DCT, IDTX};
use super::msac::Ec;
use super::tables::{SCAN_4X4, SCAN_8X8, SCAN_16X16, SCAN_32X32};

/// Contexts along one side of the block (rav1d `BlockContext`).
#[derive(Debug, Clone)]
pub struct Side {
	pub mode: Vec<u8>,
	pub intra: Vec<u8>,
	pub skip: Vec<u8>,
	pub r#ref: Vec<i8>,
	pub uvmode: Vec<u8>,
	pub tx_intra: Vec<i8>,
	pub lcoef: Vec<u8>,
	pub ccoef: [Vec<u8>; 2],
	/// In 8-sample units.
	pub partition: Vec<u8>,
}

impl Side {
	pub fn new(n4: usize) -> Self {
		Self {
			mode: vec![0; n4],
			intra: vec![0; n4],
			skip: vec![0; n4],
			r#ref: vec![-1; n4],
			uvmode: vec![0; n4],
			tx_intra: vec![-1; n4],
			lcoef: vec![0x40; n4],
			ccoef: [vec![0x40; n4], vec![0x40; n4]],
			partition: vec![0; n4 / 2 + 1],
		}
	}

	/// The half-resolution range (chroma, partition) of a luma range.
	fn half(r: &std::ops::Range<usize>) -> std::ops::Range<usize> {
		r.start / 2..r.end.div_ceil(2)
	}

	/// `reset_context` over the luma 4x4 positions `range` (and the chroma
	/// and partition entries under them).
	pub fn reset(&mut self, keyframe: bool, range: std::ops::Range<usize>) {
		let r = range.clone();
		let h = Self::half(&r);
		self.intra[r.clone()].fill(u8::from(keyframe));
		self.uvmode[h.clone()].fill(0);
		self.mode[r.clone()].fill(0); // DC_PRED and NEARESTMV are both 0
		self.skip[r.clone()].fill(0);
		self.tx_intra[r.clone()].fill(-1);
		if !keyframe {
			self.r#ref[r.clone()].fill(-1);
		}
		self.lcoef[r].fill(0x40);
		self.ccoef[0][h.clone()].fill(0x40);
		self.ccoef[1][h.clone()].fill(0x40);
		self.partition[h].fill(0);
	}

	/// A copy of the entries under the luma range `range`.
	pub fn save(&self, range: std::ops::Range<usize>) -> Side {
		let h = Self::half(&range);
		Side {
			mode: self.mode[range.clone()].to_vec(),
			intra: self.intra[range.clone()].to_vec(),
			skip: self.skip[range.clone()].to_vec(),
			r#ref: self.r#ref[range.clone()].to_vec(),
			uvmode: self.uvmode[h.clone()].to_vec(),
			tx_intra: self.tx_intra[range.clone()].to_vec(),
			lcoef: self.lcoef[range].to_vec(),
			ccoef: [self.ccoef[0][h.clone()].to_vec(), self.ccoef[1][h.clone()].to_vec()],
			partition: self.partition[h].to_vec(),
		}
	}

	/// Puts back what [`Self::save`] took from the luma range at `start`.
	pub fn restore(&mut self, start: usize, s: &Side) {
		let r = start..start + s.mode.len();
		let h = start / 2..start / 2 + s.uvmode.len();
		self.mode[r.clone()].copy_from_slice(&s.mode);
		self.intra[r.clone()].copy_from_slice(&s.intra);
		self.skip[r.clone()].copy_from_slice(&s.skip);
		self.r#ref[r.clone()].copy_from_slice(&s.r#ref);
		self.uvmode[h.clone()].copy_from_slice(&s.uvmode);
		self.tx_intra[r.clone()].copy_from_slice(&s.tx_intra);
		self.lcoef[r].copy_from_slice(&s.lcoef);
		self.ccoef[0][h.clone()].copy_from_slice(&s.ccoef[0]);
		self.ccoef[1][h.clone()].copy_from_slice(&s.ccoef[1]);
		self.partition[h].copy_from_slice(&s.partition);
	}
}

/// Above (frame-wide, by absolute 4x4 column) and left (by row within the
/// superblock) contexts.
#[derive(Debug, Clone)]
pub struct Ctx {
	pub a: Side,
	pub l: Side,
}

/// Scan of a square transform (`lw` = log2 of its side in 4-sample units).
pub fn scan(lw: usize) -> &'static [u16] {
	match lw {
		0 => &SCAN_4X4,
		1 => &SCAN_8X8,
		2 => &SCAN_16X16,
		_ => &SCAN_32X32,
	}
}

/// rav1d `dav1d_skip_ctx`.
const SKIP_CTX: [[u8; 5]; 5] =
	[[1, 2, 2, 2, 3], [2, 4, 4, 4, 5], [2, 4, 4, 4, 5], [2, 4, 4, 4, 5], [3, 5, 5, 5, 6]];
/// rav1d `dav1d_lo_ctx_offsets[0]` (square transforms).
const LO_CTX_OFFSETS: [[u8; 5]; 5] = [
	[0, 1, 6, 6, 21],
	[1, 6, 6, 21, 21],
	[6, 6, 21, 21, 21],
	[6, 21, 21, 21, 21],
	[21, 21, 21, 21, 21],
];
/// The reduced intra transform set (`dav1d_tx_types_per_set[0..5]`).
pub const INTRA_SET: [u8; 5] = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST];

/// Transform type of a chroma intra block (`dav1d_txtp_from_uvmode`).
pub const TXTP_FROM_UVMODE: [u8; 14] = [
	DCT_DCT, ADST_DCT, DCT_ADST, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST, DCT_ADST, ADST_DCT,
	ADST_ADST, ADST_DCT, DCT_ADST, ADST_ADST, DCT_DCT,
];

/// What a transform block is coded within.
#[derive(Debug, Clone, Copy)]
pub struct TxInfo {
	/// Transform side, log2 of 4-sample units (0 = 4x4 ... 4 = 64x64).
	pub lw: usize,
	pub chroma: bool,
	/// Block side, log2 of luma 4-sample units.
	pub block_lw: usize,
	/// Intra block, and its luma mode (for the transform-type CDF).
	pub intra: bool,
	pub y_mode: u8,
}

fn skip_ctx(t: &TxInfo, a: &[u8], l: &[u8]) -> usize {
	let n = 1usize << t.lw;
	if t.chroma {
		let b = t.block_lw.saturating_sub(1);
		let not_one = usize::from(b > t.lw);
		let ca = usize::from(a[..n].iter().any(|&v| v != 0x40));
		let cl = usize::from(l[..n].iter().any(|&v| v != 0x40));
		7 + not_one * 3 + ca + cl
	} else if t.block_lw == t.lw {
		0
	} else {
		let merge = |d: &[u8]| usize::from(d[..n].iter().fold(0u8, |acc, &v| acc | v) & 0x3f).min(4);
		usize::from(SKIP_CTX[merge(a)][merge(l)])
	}
}

fn dc_sign_ctx(n: usize, a: &[u8], l: &[u8]) -> usize {
	let s: i32 = a[..n].iter().chain(&l[..n]).map(|&v| i32::from(v >> 6) - 1).sum();
	usize::from(s != 0) + usize::from(s > 0)
}

/// Writes a range token `tok` (3..=15) as up to four 4-ary symbols
/// (`decode_hi_tok`).
fn write_hi<E: Ec>(e: &mut E, tok: u32, cdf: &mut [u16; 4]) {
	let mut rem = tok - 3;
	for _ in 0..4 {
		let s = rem.min(3);
		e.symbol(s as usize, cdf, 3);
		if s < 3 {
			break;
		}
		rem -= 3;
	}
}

/// Writes one transform block. `levels` are the quantized coefficients in
/// scan order up to the last non-zero one (empty: an all-zero block),
/// `txtp` its transform type. Returns the context the block leaves
/// (`res_ctx`: summed magnitude up to 63, DC sign in bits 6-7).
pub fn write_coefs<E: Ec>(
	e: &mut E,
	cdf: &mut CdfContext,
	t: &TxInfo,
	a: &[u8],
	l: &[u8],
	levels: &[i32],
	txtp: u8,
) -> u8 {
	let n = 1usize << t.lw;
	let ctx_t = t.lw; // `t_dim.ctx` of square sizes
	let chroma = usize::from(t.chroma);
	let sctx = skip_ctx(t, a, l);
	let all_skip = levels.is_empty();
	e.bool_adapt(all_skip, &mut cdf.coef_skip[ctx_t][sctx]);
	if all_skip {
		return 0x40;
	}
	if !t.chroma {
		if t.intra {
			if t.lw < 3 {
				let idx = INTRA_SET.iter().position(|&x| x == txtp).unwrap_or(1);
				e.symbol(idx, &mut cdf.txtp_intra2[t.lw][usize::from(t.y_mode)], 4);
			}
		} else if t.lw < 4 {
			e.bool_adapt(txtp == DCT_DCT, &mut cdf.txtp_inter3[t.lw]);
		}
	}

	// End of block.
	let eob = levels.len() - 1;
	// eob in [2^(bin-1), 2^bin) from bin 2 on.
	let bin = if eob < 2 { eob } else { (usize::BITS - eob.leading_zeros()) as usize };
	match t.lw.min(3) {
		0 => e.symbol(bin, &mut cdf.eob_bin_16[chroma][0], 4),
		1 => e.symbol(bin, &mut cdf.eob_bin_64[chroma][0], 6),
		2 => e.symbol(bin, &mut cdf.eob_bin_256[chroma][0], 8),
		_ => e.symbol(bin, &mut cdf.eob_bin_1024[chroma], 10),
	}
	if bin > 1 {
		let hi = (eob >> (bin - 2)) & 1 == 1;
		e.bool_adapt(hi, &mut cdf.eob_hi_bit[ctx_t][chroma][bin]);
		e.bits((eob & ((1 << (bin - 2)) - 1)) as u32, (bin - 2) as u32);
	}

	// Base and range tokens, last coefficient first.
	let s = n.min(8);
	let stride = 4 * s;
	let shift = if t.lw < 3 { t.lw + 2 } else { 5 };
	let mask = 4 * s - 1;
	let sc = scan(t.lw);
	let hi_t = ctx_t.min(3);
	let mut lv = [0u8; 32 * (32 + 2)];
	let lv = &mut lv[..stride * (4 * s + 2)];
	let abs = |i: usize| levels[i].unsigned_abs();
	if eob > 0 {
		let rc = usize::from(sc[eob]);
		let (x, y) = (rc >> shift, rc & mask);
		let ctx = 1 + usize::from(eob > s * s * 2) + usize::from(eob > s * s * 4);
		let v = abs(eob);
		let eob_tok = v.min(3) - 1;
		e.symbol(eob_tok as usize, &mut cdf.eob_base_tok[ctx_t][chroma][ctx], 2);
		let mut level = (eob_tok + 1) * 0x41;
		if eob_tok == 2 {
			let c = if (x | y) > 1 { 14 } else { 7 };
			let tok = v.min(15);
			write_hi(e, tok, &mut cdf.br_tok[hi_t][chroma][c]);
			level = tok + (3 << 6);
		}
		lv[x * stride + y] = level as u8;
		for i in (1..eob).rev() {
			let rc = usize::from(sc[i]);
			let (x, y) = (rc >> shift, rc & mask);
			let pos = x * stride + y;
			let at = |k: usize| u32::from(lv[pos + k]);
			let mut mag = at(1) + at(stride);
			mag += at(stride + 1);
			let hi_mag = mag;
			mag += at(2) + at(2 * stride);
			let ctx = usize::from(LO_CTX_OFFSETS[y.min(4)][x.min(4)])
				+ if mag > 512 { 4 } else { ((mag + 64) >> 7) as usize };
			let v = abs(i);
			let tok = v.min(3);
			e.symbol(tok as usize, &mut cdf.base_tok[ctx_t][chroma][ctx], 3);
			if tok == 3 {
				let m = hi_mag & 63;
				let c = if (y | x) > 1 { 14 } else { 7 } + if m > 12 { 6 } else { ((m + 1) >> 1) as usize };
				let tok = v.min(15);
				write_hi(e, tok, &mut cdf.br_tok[hi_t][chroma][c]);
				lv[pos] = (tok + (3 << 6)) as u8;
			} else {
				lv[pos] = (tok * 0x41) as u8;
			}
		}
		// DC.
		let v = abs(0);
		let tok = v.min(3);
		e.symbol(tok as usize, &mut cdf.base_tok[ctx_t][chroma][0], 3);
		if tok == 3 {
			let m = (u32::from(lv[1]) + u32::from(lv[stride]) + u32::from(lv[stride + 1])) & 63;
			let c = if m > 12 { 6 } else { ((m + 1) >> 1) as usize };
			write_hi(e, v.min(15), &mut cdf.br_tok[hi_t][chroma][c]);
		}
	} else {
		let v = abs(0);
		let tok = v.min(3) - 1;
		e.symbol(tok as usize, &mut cdf.eob_base_tok[ctx_t][chroma][0], 2);
		if tok == 2 {
			write_hi(e, v.min(15), &mut cdf.br_tok[hi_t][chroma][0]);
		}
	}

	// Signs and remainders, scan order.
	let mut cul: u32 = 0;
	let dc = levels[0];
	let dc_level = if dc == 0 {
		1 << 6
	} else {
		let neg = dc < 0;
		e.bool_adapt(neg, &mut cdf.dc_sign[chroma][dc_sign_ctx(n, a, l)]);
		if abs(0) >= 15 {
			e.golomb(abs(0) - 15);
		}
		cul += abs(0);
		if neg { 0 } else { 2 << 6 }
	};
	for i in 1..=eob {
		let v = levels[i];
		if v != 0 {
			e.equi(v < 0);
			if abs(i) >= 15 {
				e.golomb(abs(i) - 15);
			}
			cul = cul.wrapping_add(abs(i));
		}
	}
	(cul.min(63) | dc_level) as u8
}

/// Dequantizes a level (`decode_coefs`): `(|level| * q) >> shift`, at most
/// `32767` (`32768` when negative).
#[inline]
pub fn dequant(level: i32, q: i32, shift: u32) -> i32 {
	let a = level.unsigned_abs();
	let mut d = u64::from(a) * q as u64;
	if a >= 15 {
		d &= 0xff_ffff;
	}
	let d = ((d >> shift) as i64).min(32767 + i64::from(level < 0)) as i32;
	if level < 0 { -d } else { d }
}
