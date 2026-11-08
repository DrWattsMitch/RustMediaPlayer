//! CAVLC residual decoding (ISO/IEC 14496-10 §9.2; FFmpeg `h264_cavlc.c`).

use std::sync::OnceLock;

use super::tables::{COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS, TOTAL_ZEROS_CHROMA_DC};
use crate::codecs::bitstream::{BitReader, Code, Vlc};

struct Tables {
	/// coeff_token per nC class (0-1, 2-3, 4-7, 8+, chroma DC); symbol =
	/// `total_coeff * 4 + trailing_ones`.
	coeff_token: Vec<Vlc>,
	/// total_zeros per total_coeff (index 1..=15).
	total_zeros: Vec<Vlc>,
	total_zeros_dc: Vec<Vlc>,
	/// run_before per min(zeros_left, 7) (index 1..=7).
	run_before: Vec<Vlc>,
}

fn build(codes: impl Iterator<Item = (u32, u8, u32)>, root: u32) -> Vlc {
	let list: Vec<Code> =
		codes.filter(|c| c.1 > 0).map(|(code, len, symbol)| Code { code, len, symbol }).collect();
	Vlc::new(&list, root).unwrap_or_else(Vlc::empty)
}

fn tables() -> &'static Tables {
	static T: OnceLock<Tables> = OnceLock::new();
	T.get_or_init(|| Tables {
		coeff_token: COEFF_TOKEN
			.iter()
			.map(|t| {
				build(
					t.iter().enumerate().flat_map(|(tc, row)| {
						row
							.iter()
							.enumerate()
							.map(move |(t1, &(code, len))| (u32::from(code), len, (tc * 4 + t1) as u32))
					}),
					8,
				)
			})
			.collect(),
		total_zeros: TOTAL_ZEROS
			.iter()
			.map(|row| build(row.iter().enumerate().map(|(z, &(c, l))| (u32::from(c), l, z as u32)), 9))
			.collect(),
		total_zeros_dc: TOTAL_ZEROS_CHROMA_DC
			.iter()
			.map(|row| build(row.iter().enumerate().map(|(z, &(c, l))| (u32::from(c), l, z as u32)), 3))
			.collect(),
		run_before: RUN_BEFORE
			.iter()
			.map(|row| build(row.iter().enumerate().map(|(r, &(c, l))| (u32::from(c), l, r as u32)), 6))
			.collect(),
	})
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BadResidual;

/// Decodes one `residual_block_cavlc()`.
///
/// `nc` selects the coeff_token table (-1 = 4:2:0 chroma DC). Levels are
/// written to `out[scan[start + i]]` for `i < max`. Returns `total_coeff`.
pub fn residual(
	bs: &mut BitReader,
	nc: i32,
	scan: &[u8],
	start: usize,
	max: usize,
	out: &mut [i32],
) -> Result<u8, BadResidual> {
	let t = tables();
	let table = match nc {
		-1 => 4,
		0..=1 => 0,
		2..=3 => 1,
		4..=7 => 2,
		_ => 3,
	};
	let token = t.coeff_token[table].decode(bs).ok_or(BadResidual)? as usize;
	let (total, trailing) = (token / 4, token % 4);
	if total == 0 {
		return Ok(0);
	}
	if total > max || trailing > total {
		return Err(BadResidual);
	}

	let mut levels = [0i32; 16];
	let mut suffix_len: u32 = if total > 10 && trailing < 3 { 1 } else { 0 };
	for (i, level) in levels.iter_mut().enumerate().take(total) {
		if i < trailing {
			*level = if bs.bit() { -1 } else { 1 };
			continue;
		}
		let mut prefix = 0u32;
		while !bs.bit() {
			prefix += 1;
			if prefix > 28 || bs.overrun() {
				return Err(BadResidual);
			}
		}
		let suffix_size = if prefix == 14 && suffix_len == 0 {
			4
		} else if prefix >= 15 {
			prefix - 3
		} else {
			suffix_len
		};
		let mut code: i64 = i64::from(prefix.min(15)) << suffix_len;
		if suffix_size > 0 {
			code += i64::from(bs.bits(suffix_size));
		}
		if prefix >= 15 && suffix_len == 0 {
			code += 15;
		}
		if prefix >= 16 {
			code += (1i64 << (prefix - 3)) - 4096;
		}
		if i == trailing && trailing < 3 {
			code += 2;
		}
		let value = if code % 2 == 0 { (code + 2) >> 1 } else { (-code - 1) >> 1 };
		*level = value.clamp(-(1 << 16), 1 << 16) as i32;
		if suffix_len == 0 {
			suffix_len = 1;
		}
		if level.unsigned_abs() > (3 << (suffix_len - 1)) && suffix_len < 6 {
			suffix_len += 1;
		}
	}

	let mut zeros_left = 0usize;
	if total < max {
		let vlc = if nc == -1 { t.total_zeros_dc.get(total) } else { t.total_zeros.get(total) };
		zeros_left = vlc.and_then(|v| v.decode(bs)).ok_or(BadResidual)? as usize;
	}
	if total + zeros_left > max {
		return Err(BadResidual);
	}
	// Place levels from the highest frequency down.
	let mut pos = total + zeros_left;
	for (i, &level) in levels.iter().enumerate().take(total) {
		let run = if i + 1 < total && zeros_left > 0 {
			let r = t.run_before[zeros_left.min(7)].decode(bs).ok_or(BadResidual)? as usize;
			if r > zeros_left {
				return Err(BadResidual);
			}
			r
		} else if i + 1 == total {
			zeros_left
		} else {
			0
		};
		zeros_left -= run.min(zeros_left);
		pos -= 1;
		if let Some(&raster) = scan.get(start + pos) {
			if let Some(slot) = out.get_mut(usize::from(raster)) {
				*slot = level;
			}
		}
		pos -= run.min(pos);
	}
	if bs.overrun() {
		return Err(BadResidual);
	}
	Ok(total as u8)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn total_coeff_zero() {
		let t = tables();
		assert_eq!(t.coeff_token.len(), 5);
		// With 0 <= nC < 2, the 1-bit code "1" means TotalCoeff = 0.
		let mut bs = BitReader::new(&[0x80]);
		let mut out = [0i32; 16];
		assert_eq!(residual(&mut bs, 0, &super::super::consts::ZIGZAG_4X4, 0, 16, &mut out), Ok(0));
		assert_eq!(bs.pos(), 1);
	}

	#[test]
	fn decodes_single_trailing_one() {
		// coeff_token TC=1,T1=1 with nC=0 is "01"; sign "1" = -1; total_zeros
		// for TC=1 value 0 is "1". Coefficient at scan position 0 = -1.
		let mut bs = BitReader::new(&[0b0111_0000]);
		let mut out = [0i32; 16];
		assert_eq!(residual(&mut bs, 0, &super::super::consts::ZIGZAG_4X4, 0, 16, &mut out), Ok(1));
		assert_eq!(out[0], -1);
	}
}
