//! ITU-T G.711 A-law and µ-law companding.
//!
//! Expansion is the closed-form formula from the recommendation. Compression
//! maps a 16-bit sample to the code whose expanded value is nearest, using
//! decision thresholds halfway between adjacent reconstruction levels. The
//! thresholds are rounded to the same 14-bit grid FFmpeg's `pcm_tablegen.h`
//! uses, so encoded bytes match FFmpeg's `pcm_alaw` / `pcm_mulaw` encoders.

use std::sync::OnceLock;

/// Even-bit inversion applied to every A-law code.
const ALAW_XOR: u8 = 0x55;
/// Codes are inverted in µ-law.
const MULAW_XOR: u8 = 0xFF;
const MULAW_BIAS: i32 = 0x84;
/// The compression table is indexed by `(sample + 32768) >> 2`.
const TABLE_SIZE: usize = 1 << 14;
const TABLE_ZERO: usize = TABLE_SIZE / 2;

/// Expands an A-law code to a 16-bit sample.
pub const fn alaw_to_linear(code: u8) -> i16 {
	let a = code ^ ALAW_XOR;
	let mantissa = (a & 0x0F) as i32;
	let segment = ((a & 0x70) >> 4) as u32;
	let magnitude =
		if segment == 0 { (mantissa << 4) + 8 } else { ((mantissa << 4) + 0x108) << (segment - 1) };
	(if a & 0x80 != 0 { magnitude } else { -magnitude }) as i16
}

/// Expands a µ-law code to a 16-bit sample.
pub const fn mulaw_to_linear(code: u8) -> i16 {
	let u = !code;
	let magnitude = ((((u & 0x0F) as i32) << 3) + MULAW_BIAS) << ((u & 0x70) >> 4);
	(if u & 0x80 != 0 { MULAW_BIAS - magnitude } else { magnitude - MULAW_BIAS }) as i16
}

/// Builds the sample -> code table for one law. `positive(i)` is the code of
/// the i-th smallest positive reconstruction level (i in 0..128); flipping
/// bit 7 gives its negative twin.
fn build(expand: fn(u8) -> i16, positive: fn(u8) -> u8) -> Box<[u8; TABLE_SIZE]> {
	let mut table = Box::new([0u8; TABLE_SIZE]);
	// thresholds[i]: first grid magnitude that quantises to level i + 1.
	let mut thresholds = [0usize; 127];
	for (i, t) in thresholds.iter_mut().enumerate() {
		let lo = i32::from(expand(positive(i as u8)));
		let hi = i32::from(expand(positive(i as u8 + 1)));
		*t = ((lo + hi + 4) >> 3) as usize;
	}
	let mut level = 0usize;
	table[TABLE_ZERO] = positive(0);
	for magnitude in 1..TABLE_ZERO {
		while level < thresholds.len() && magnitude >= thresholds[level] {
			level += 1;
		}
		let code = positive(level as u8);
		table[TABLE_ZERO + magnitude] = code;
		table[TABLE_ZERO - magnitude] = code ^ 0x80;
	}
	// -32768 (index 0) saturates to the largest negative level.
	table[0] = table[1];
	table
}

fn alaw_table() -> &'static [u8; TABLE_SIZE] {
	static TABLE: OnceLock<Box<[u8; TABLE_SIZE]>> = OnceLock::new();
	// Positive A-law codes have bit 7 set before the even-bit inversion.
	TABLE.get_or_init(|| build(alaw_to_linear, |i| (i | 0x80) ^ ALAW_XOR))
}

fn mulaw_table() -> &'static [u8; TABLE_SIZE] {
	static TABLE: OnceLock<Box<[u8; TABLE_SIZE]>> = OnceLock::new();
	// Positive µ-law codes have bit 7 clear before the inversion.
	TABLE.get_or_init(|| build(mulaw_to_linear, |i| i ^ MULAW_XOR))
}

fn index(sample: i16) -> usize {
	((i32::from(sample) + 32768) >> 2) as usize
}

pub fn linear_to_alaw(sample: i16) -> u8 {
	alaw_table()[index(sample)]
}

pub fn linear_to_mulaw(sample: i16) -> u8 {
	mulaw_table()[index(sample)]
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn reference_points() {
		// G.711 Table 1/2 extremes and the codes for silence.
		assert_eq!(alaw_to_linear(0xD5), 8);
		assert_eq!(alaw_to_linear(0x55), -8);
		assert_eq!(alaw_to_linear(0xAA), 32256);
		assert_eq!(alaw_to_linear(0x2A), -32256);
		assert_eq!(mulaw_to_linear(0xFF), 0);
		assert_eq!(mulaw_to_linear(0x80), 32124);
		assert_eq!(mulaw_to_linear(0x00), -32124);
	}

	#[test]
	fn compress_inverts_expand() {
		for code in 0..=255u8 {
			assert_eq!(linear_to_alaw(alaw_to_linear(code)), code, "alaw {code:#x}");
			// µ-law has two zeros (0x7F and 0xFF); both expand to 0.
			if code != 0x7F {
				assert_eq!(linear_to_mulaw(mulaw_to_linear(code)), code, "mulaw {code:#x}");
			}
		}
	}

	#[test]
	fn compression_is_monotonic() {
		let mut last = i32::MIN;
		for s in (i16::MIN..=i16::MAX).step_by(4) {
			let v = i32::from(alaw_to_linear(linear_to_alaw(s)));
			assert!(v >= last);
			last = v;
		}
	}
}
