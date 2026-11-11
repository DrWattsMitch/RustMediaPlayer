//! CAVLC residual coding (ISO/IEC 14496-10 §9.2, encoder side; x264
//! `cavlc.c` `cavlc_block_residual`, FFmpeg's decoder `h264_cavlc.c`
//! inverted). Every function writes exactly what
//! [`crate::codecs::h264::cavlc::residual`] reads.

use crate::codecs::bitstream::BitWriter;
use crate::codecs::h264::consts::{GOLOMB_TO_INTER_CBP, GOLOMB_TO_INTRA_CBP};
use crate::codecs::h264::tables::{COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS, TOTAL_ZEROS_CHROMA_DC};

/// Writes one `residual_block_cavlc()` from levels in scan order
/// (`levels.len()` = maxNumCoeff: 16, 15 or 4). `nc` = -1 selects the
/// chroma DC tables. Returns `TotalCoeff`.
pub fn write_block(w: &mut BitWriter, nc: i32, levels: &[i32]) -> u8 {
	let max = levels.len();
	let table = match nc {
		-1 => 4,
		0..=1 => 0,
		2..=3 => 1,
		4..=7 => 2,
		_ => 3,
	};
	// Non-zero levels from the highest frequency down, with the zero run
	// below each.
	let mut nz = [0i32; 16];
	let mut runs = [0usize; 16];
	let mut total = 0usize;
	let mut total_zeros = 0usize;
	if let Some(last) = levels.iter().rposition(|&v| v != 0) {
		let mut i = last as isize;
		while i >= 0 {
			let v = levels[i as usize];
			i -= 1;
			let mut run = 0;
			while i >= 0 && levels[i as usize] == 0 {
				run += 1;
				i -= 1;
			}
			nz[total] = v;
			runs[total] = run;
			total += 1;
		}
		// Every zero below the highest level; the lowest level's run is
		// whatever remains, so it's never coded.
		total_zeros = last + 1 - total;
	}
	let mut trailing = 0usize;
	while trailing < total.min(3) && nz[trailing].abs() == 1 {
		trailing += 1;
	}
	let (code, len) = COEFF_TOKEN[table][total][trailing];
	w.put(u32::from(code), u32::from(len));
	if total == 0 {
		return 0;
	}
	for &v in &nz[..trailing] {
		w.bit(v < 0);
	}
	let mut suffix_len: u32 = if total > 10 && trailing < 3 { 1 } else { 0 };
	for i in trailing..total {
		let v = nz[i];
		let mut code = if v > 0 { 2 * (v as i64) - 2 } else { -2 * (v as i64) - 1 };
		if i == trailing && trailing < 3 {
			code -= 2;
		}
		write_level(w, code, suffix_len);
		if suffix_len == 0 {
			suffix_len = 1;
		}
		if v.unsigned_abs() > (3 << (suffix_len - 1)) && suffix_len < 6 {
			suffix_len += 1;
		}
	}
	if total < max {
		let (c, l) = if nc == -1 {
			TOTAL_ZEROS_CHROMA_DC[total][total_zeros]
		} else {
			TOTAL_ZEROS[total][total_zeros]
		};
		w.put(u32::from(c), u32::from(l));
	}
	// run_before for every level but the lowest, while zeros remain.
	let mut zeros_left = total_zeros;
	for &run in runs.iter().take(total.saturating_sub(1)) {
		if zeros_left == 0 {
			break;
		}
		let (c, l) = RUN_BEFORE[zeros_left.min(7)][run];
		w.put(u32::from(c), u32::from(l));
		zeros_left -= run;
	}
	total as u8
}

/// `level_prefix` / `level_suffix` for one levelCode (§9.2.2.1 inverted,
/// including the escapes of prefixes 14, 15 and 16+).
fn write_level(w: &mut BitWriter, code: i64, suffix_len: u32) {
	let put_prefix = |w: &mut BitWriter, prefix: u32| {
		// `prefix` zeros then a one.
		if prefix < 32 {
			w.put(1, prefix + 1);
		} else {
			w.put(0, prefix);
			w.put(1, 1);
		}
	};
	if suffix_len == 0 {
		if code < 14 {
			put_prefix(w, code as u32);
			return;
		}
		if code < 30 {
			put_prefix(w, 14);
			w.put((code - 14) as u32, 4);
			return;
		}
	} else if code < (15i64 << suffix_len) {
		put_prefix(w, (code >> suffix_len) as u32);
		w.put((code & ((1 << suffix_len) - 1)) as u32, suffix_len);
		return;
	}
	// Escape: prefix 15 with a 12-bit suffix, or longer prefixes.
	let base = (15i64 << suffix_len) + if suffix_len == 0 { 15 } else { 0 };
	let rem = code - base;
	if rem < 4096 {
		put_prefix(w, 15);
		w.put(rem as u32, 12);
	} else {
		let x = rem + 4096;
		let bits = 63 - x.leading_zeros(); // floor(log2 x) = prefix - 3
		put_prefix(w, bits + 3);
		w.put((x - (1i64 << bits)) as u32, bits);
	}
}

/// `me(v)` codeNum of a coded_block_pattern.
pub fn cbp_code(cbp: u8, intra: bool) -> u32 {
	let table = if intra { &GOLOMB_TO_INTRA_CBP } else { &GOLOMB_TO_INTER_CBP };
	table.iter().position(|&c| c == cbp).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::bitstream::BitReader;
	use crate::codecs::h264::cavlc::residual;

	fn round_trip(levels: &[i32], nc: i32) {
		let mut w = BitWriter::new();
		let total = write_block(&mut w, nc, levels);
		let bits = w.bits();
		let data = w.finish();
		let mut bs = BitReader::new(&data);
		let identity: Vec<u8> = (0..levels.len() as u8).collect();
		let mut out = vec![0i32; levels.len()];
		let n = residual(&mut bs, nc, &identity, 0, levels.len(), &mut out).expect("decodes");
		assert_eq!(n, total);
		assert_eq!(out, levels, "nc {nc}");
		assert_eq!(bs.pos(), bits, "nc {nc}: {levels:?}");
	}

	#[test]
	fn blocks_round_trip_through_the_decoder() {
		let cases: Vec<Vec<i32>> = vec![
			vec![0; 16],
			{
				let mut v = vec![0; 16];
				v[0] = 1;
				v
			},
			vec![7, -3, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			vec![30, 12, -9, 5, 3, -2, 2, 1, 1, 0, -1, 0, 0, 1, 0, 0],
			(0..16).map(|i| if i % 2 == 0 { 40 - i } else { -i }).collect(),
			vec![3000, -2500, 900, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
			vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
			vec![1; 16],
			vec![20000, -16000, 8000, 5000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		];
		for levels in &cases {
			for nc in [0, 2, 5, 9] {
				round_trip(levels, nc);
				round_trip(&levels[1..], nc); // AC blocks (15)
			}
		}
		for dc in [
			vec![0, 0, 0, 0],
			vec![5, -1, 0, 1],
			vec![0, 0, 0, -3],
			vec![1, 1, 1, 1],
			vec![300, -2, 0, 0],
		] {
			round_trip(&dc, -1);
		}
	}

	#[test]
	fn cbp_codes_invert_the_decoder_tables() {
		for cbp in 0..48u8 {
			assert_eq!(GOLOMB_TO_INTRA_CBP[cbp_code(cbp, true) as usize], cbp);
			assert_eq!(GOLOMB_TO_INTER_CBP[cbp_code(cbp, false) as usize], cbp);
		}
	}
}
