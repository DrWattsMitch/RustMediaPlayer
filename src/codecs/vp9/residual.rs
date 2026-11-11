//! Coefficient tokens (spec §6.4.24, §9.3; libvpx `vp9_detokenize.c`).
//!
//! Tokens are coded along a scan order with a context from two already
//! decoded neighbours. The first three tree nodes (EOB, ZERO, ONE) have
//! adaptive probabilities; the rest come from the Pareto model of the
//! third. Values are dequantized as they are read (32x32 halves them).

use super::range::BoolDecoder;
use super::tables;

const CAT1: [u8; 1] = [159];
const CAT2: [u8; 2] = [165, 145];
const CAT3: [u8; 3] = [173, 148, 140];
const CAT4: [u8; 4] = [176, 155, 140, 135];
const CAT5: [u8; 5] = [180, 157, 141, 134, 130];
const CAT6: [u8; 14] = [254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129];

/// A scan order and its context neighbours.
pub struct Scan {
	pub scan: &'static [u16],
	pub neighbors: &'static [[u16; 2]],
}

/// Transform types (`TX_TYPE`): vertical / horizontal 1-D transforms.
pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;

/// Scan of a transform size and type (`vp9_scan_orders`): ADST_DCT uses the
/// row scan, DCT_ADST the column scan, 32x32 always the default scan.
pub fn scan(tx_size: u8, tx_type: u8) -> Scan {
	use tables::*;
	let (scan, neighbors): (&'static [u16], &'static [[u16; 2]]) = match (tx_size, tx_type) {
		(0, ADST_DCT) => (&ROW_SCAN_4X4, &ROW_SCAN_4X4_NEIGHBORS),
		(0, DCT_ADST) => (&COL_SCAN_4X4, &COL_SCAN_4X4_NEIGHBORS),
		(0, _) => (&DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NEIGHBORS),
		(1, ADST_DCT) => (&ROW_SCAN_8X8, &ROW_SCAN_8X8_NEIGHBORS),
		(1, DCT_ADST) => (&COL_SCAN_8X8, &COL_SCAN_8X8_NEIGHBORS),
		(1, _) => (&DEFAULT_SCAN_8X8, &DEFAULT_SCAN_8X8_NEIGHBORS),
		(2, ADST_DCT) => (&ROW_SCAN_16X16, &ROW_SCAN_16X16_NEIGHBORS),
		(2, DCT_ADST) => (&COL_SCAN_16X16, &COL_SCAN_16X16_NEIGHBORS),
		(2, _) => (&DEFAULT_SCAN_16X16, &DEFAULT_SCAN_16X16_NEIGHBORS),
		_ => (&DEFAULT_SCAN_32X32, &DEFAULT_SCAN_32X32_NEIGHBORS),
	};
	Scan { scan, neighbors }
}

fn extra(r: &mut BoolDecoder, probs: &[u8]) -> i32 {
	probs.iter().fold(0, |v, &p| (v << 1) | i32::from(r.read(p)))
}

/// Where token counts go (`None` in frame-parallel mode).
pub struct TokenCounts<'a> {
	pub coef: &'a mut [[[u32; 4]; 6]; 6],
	pub eob_branch: &'a mut [[u32; 6]; 6],
}

/// Decodes the tokens of one transform block into `coefs` (raster order,
/// dequantized); returns the end of block. `probs` are the block's
/// `[band][context][node]` probabilities, `dq` its DC and AC quantizers.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs(
	r: &mut BoolDecoder,
	probs: &[[[u8; 3]; 6]; 6],
	mut counts: Option<TokenCounts>,
	coefs: &mut [i32],
	tx_size: u8,
	dq: [i16; 2],
	mut ctx: usize,
	sc: &Scan,
) -> usize {
	let max_eob = 16usize << (tx_size << 1);
	let band_of: &[u8] = if tx_size == 0 { &tables::COEFBAND_4X4 } else { &tables::COEFBAND_8X8PLUS };
	let shift = u32::from(tx_size == 3);
	let mut token_cache = [0u8; 1024];
	let mut dqv = i32::from(dq[0]);
	let mut c = 0usize;
	let context = |cache: &[u8; 1024], c: usize| {
		let n = sc.neighbors[c];
		(1 + usize::from(cache[usize::from(n[0])]) + usize::from(cache[usize::from(n[1])])) >> 1
	};
	while c < max_eob {
		let mut band = usize::from(band_of[c]);
		let mut p = &probs[band][ctx];
		if let Some(k) = counts.as_mut() {
			k.eob_branch[band][ctx] += 1;
		}
		if !r.read(p[0]) {
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][3] += 1;
			}
			break;
		}
		while !r.read(p[1]) {
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][0] += 1;
			}
			dqv = i32::from(dq[1]);
			token_cache[usize::from(sc.scan[c])] = 0;
			c += 1;
			if c >= max_eob {
				return c;
			}
			ctx = context(&token_cache, c);
			band = usize::from(band_of[c]);
			p = &probs[band][ctx];
		}
		let pos = usize::from(sc.scan[c]);
		let v = if !r.read(p[2]) {
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][1] += 1;
			}
			token_cache[pos] = 1;
			dqv >> shift
		} else {
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][2] += 1;
			}
			let pareto = &tables::PARETO8[usize::from(p[2]).max(1) - 1];
			if r.read(pareto[0]) {
				let val = if r.read(pareto[3]) {
					token_cache[pos] = 5;
					if r.read(pareto[5]) {
						if r.read(pareto[7]) { 67 + extra(r, &CAT6) } else { 35 + extra(r, &CAT5) }
					} else if r.read(pareto[6]) {
						19 + extra(r, &CAT4)
					} else {
						11 + extra(r, &CAT3)
					}
				} else {
					token_cache[pos] = 4;
					if r.read(pareto[4]) { 7 + extra(r, &CAT2) } else { 5 + extra(r, &CAT1) }
				};
				((i64::from(val) * i64::from(dqv)) >> shift) as i32
			} else if r.read(pareto[1]) {
				token_cache[pos] = 3;
				((3 + i32::from(r.read(pareto[2]))) * dqv) >> shift
			} else {
				token_cache[pos] = 2;
				(2 * dqv) >> shift
			}
		};
		coefs[pos] = if r.bit() { -v } else { v };
		c += 1;
		if c < max_eob {
			ctx = context(&token_cache, c);
		}
		dqv = i32::from(dq[1]);
	}
	c
}
