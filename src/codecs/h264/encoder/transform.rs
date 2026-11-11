//! Forward transforms, quantisation and SATD (ISO/IEC 14496-10 §8.5
//! inverted; the forward side is encoder-defined).
//!
//! FFmpeg / x264: `x264/common/dct.c` (`sub4x4_dct`, `dct4x4dc`,
//! `dct2x2dc`), `quant.c`, `pixel.c` (`satd_4x4`). Reconstruction never
//! happens here: the encoder rebuilds pictures with the decoder's own
//! `recon` functions, so its references match any conforming decoder.

/// Quantiser multipliers `MF(qp % 6)` for the three coefficient classes:
/// (even, even), (odd, odd), mixed — the reciprocals of the decoder's
/// `NORM_ADJUST_4X4` scaled by 2^15 (x264 `quant4_mf`, flat matrices).
const MF: [[i32; 3]; 6] = [
	[13107, 5243, 8066],
	[11916, 4660, 7490],
	[10082, 4194, 6554],
	[9362, 3647, 5825],
	[8192, 3355, 5243],
	[7282, 2893, 4559],
];

#[inline]
fn class(pos: usize) -> usize {
	let (i, j) = (pos / 4, pos % 4);
	if i % 2 == 0 && j % 2 == 0 {
		0
	} else if i % 2 == 1 && j % 2 == 1 {
		1
	} else {
		2
	}
}

/// Forward 4x4 core transform of a raster residual block (8.5.12 inverse).
pub fn fdct4(d: &[i32; 16]) -> [i32; 16] {
	let mut t = [0i32; 16];
	for i in 0..4 {
		let r = &d[i * 4..i * 4 + 4];
		let (s03, d03) = (r[0] + r[3], r[0] - r[3]);
		let (s12, d12) = (r[1] + r[2], r[1] - r[2]);
		t[i * 4] = s03 + s12;
		t[i * 4 + 1] = 2 * d03 + d12;
		t[i * 4 + 2] = s03 - s12;
		t[i * 4 + 3] = d03 - 2 * d12;
	}
	let mut out = [0i32; 16];
	for j in 0..4 {
		let (s03, d03) = (t[j] + t[12 + j], t[j] - t[12 + j]);
		let (s12, d12) = (t[4 + j] + t[8 + j], t[4 + j] - t[8 + j]);
		out[j] = s03 + s12;
		out[4 + j] = 2 * d03 + d12;
		out[8 + j] = s03 - s12;
		out[12 + j] = d03 - 2 * d12;
	}
	out
}

/// Forward 4x4 Hadamard of the Intra16x16 DC coefficients, halved
/// (x264 `dct4x4dc`).
pub fn hadamard4(d: &[i32; 16]) -> [i32; 16] {
	let mut t = [0i32; 16];
	for i in 0..4 {
		let r = &d[i * 4..i * 4 + 4];
		let (a, b, c, e) = (r[0] + r[1], r[0] - r[1], r[2] + r[3], r[2] - r[3]);
		t[i * 4] = a + c;
		t[i * 4 + 1] = a - c;
		t[i * 4 + 2] = b - e;
		t[i * 4 + 3] = b + e;
	}
	let mut out = [0i32; 16];
	for j in 0..4 {
		let (a, b, c, e) =
			(t[j] + t[4 + j], t[j] - t[4 + j], t[8 + j] + t[12 + j], t[8 + j] - t[12 + j]);
		out[j] = (a + c + 1) >> 1;
		out[4 + j] = (a - c + 1) >> 1;
		out[8 + j] = (b - e + 1) >> 1;
		out[12 + j] = (b + e + 1) >> 1;
	}
	out
}

/// Forward 2x2 Hadamard of chroma DC coefficients.
pub fn hadamard2(d: &[i32; 4]) -> [i32; 4] {
	let (a, b, c, e) = (d[0] + d[1], d[0] - d[1], d[2] + d[3], d[2] - d[3]);
	[a + c, b + e, a - c, b - e]
}

/// Deadzone quantiser settings: rounding offset as a fraction of the step
/// (JM / x264 defaults: 1/3 for intra, 1/6 for inter).
#[derive(Debug, Clone, Copy)]
pub struct Quant {
	qp: usize,
	intra: bool,
}

impl Quant {
	pub fn new(qp: i32, intra: bool) -> Self {
		Self { qp: qp.clamp(0, 51) as usize, intra }
	}

	#[inline]
	fn qbits(&self) -> u32 {
		15 + (self.qp / 6) as u32
	}

	#[inline]
	fn offset(&self, qbits: u32) -> i64 {
		let unit = 1i64 << qbits;
		if self.intra { unit / 3 } else { unit / 6 }
	}

	#[inline]
	fn level(v: i32, mf: i32, f: i64, qbits: u32) -> i32 {
		let q = ((i64::from(v.unsigned_abs()) * i64::from(mf) + f) >> qbits) as i32;
		if v < 0 { -q } else { q }
	}

	/// Quantises a 4x4 block in place (raster positions); `skip_dc` leaves
	/// position 0 alone for blocks whose DC is coded separately. Returns
	/// `true` if any level is non-zero.
	pub fn block4(&self, c: &mut [i32; 16], skip_dc: bool) -> bool {
		let qbits = self.qbits();
		let f = self.offset(qbits);
		let mf = &MF[self.qp % 6];
		let mut any = false;
		for (pos, v) in c.iter_mut().enumerate() {
			if skip_dc && pos == 0 {
				continue;
			}
			*v = Self::level(*v, mf[class(pos)], f, qbits);
			any |= *v != 0;
		}
		any
	}

	/// Quantises Hadamard-transformed DC coefficients (luma 4x4 or chroma
	/// 2x2): one more shift than AC (8.5.10 / 8.5.11 inverted).
	pub fn dc(&self, c: &mut [i32]) -> bool {
		let qbits = self.qbits() + 1;
		let f = 2 * self.offset(qbits - 1);
		let mf = MF[self.qp % 6][0];
		let mut any = false;
		for v in c.iter_mut() {
			*v = Self::level(*v, mf, f, qbits);
			any |= *v != 0;
		}
		any
	}
}

/// Sum of absolute Hadamard-transformed differences of a 4x4 block,
/// halved (x264's `satd_4x4`).
pub fn satd4(d: &[i32; 16]) -> u32 {
	let mut t = [0i32; 16];
	for i in 0..4 {
		let r = &d[i * 4..i * 4 + 4];
		let (a, b, c, e) = (r[0] + r[1], r[0] - r[1], r[2] + r[3], r[2] - r[3]);
		t[i * 4] = a + c;
		t[i * 4 + 1] = a - c;
		t[i * 4 + 2] = b - e;
		t[i * 4 + 3] = b + e;
	}
	let mut sum = 0u32;
	for j in 0..4 {
		let (a, b, c, e) =
			(t[j] + t[4 + j], t[j] - t[4 + j], t[8 + j] + t[12 + j], t[8 + j] - t[12 + j]);
		sum += (a + c).unsigned_abs()
			+ (a - c).unsigned_abs()
			+ (b - e).unsigned_abs()
			+ (b + e).unsigned_abs();
	}
	sum.div_ceil(2)
}

/// x264's decimation score of a 4x4 block in scan order (`decimate_score`):
/// isolated ±1 levels after long zero runs cost more bits than they're
/// worth in inter blocks. 9 or more means "keep".
pub fn decimate_score(scan: &[i32]) -> u32 {
	const TABLE: [u32; 16] = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	let Some(last) = scan.iter().rposition(|&v| v != 0) else { return 0 };
	let mut idx = last as isize;
	let mut score = 0;
	while idx >= 0 {
		if scan[idx as usize].abs() > 1 {
			return 9;
		}
		idx -= 1;
		let mut run = 0;
		while idx >= 0 && scan[idx as usize] == 0 {
			idx -= 1;
			run += 1;
		}
		score += TABLE[run.min(15)];
	}
	score
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::h264::params::{Dequant, ScalingLists};
	use crate::codecs::h264::recon::{self, Plane};

	#[test]
	fn transform_quant_round_trips_through_the_decoder_path() {
		let dq = Dequant::new(&ScalingLists::flat());
		let src: [i32; 16] = std::array::from_fn(|i| ((i * 37) % 23) as i32 - 11);
		for qp in [0, 10, 22, 28, 40] {
			let mut c = fdct4(&src);
			Quant::new(qp, true).block4(&mut c, false);
			recon::dequant4(&mut c, qp, &dq.l4[0][(qp % 6) as usize], false);
			let mut p = Plane::new(4, 4, 128);
			recon::idct4_add(&mut p, 0, 0, &c);
			let err: i32 =
				(0..16).map(|i| (i32::from(p.data[i]) - 128 - src[i]).abs()).max().unwrap_or(0);
			let step = (0.625 * 2f64.powf(f64::from(qp) / 6.0)) as i32 + 1;
			assert!(err <= step, "qp {qp}: max error {err}");
		}
	}

	#[test]
	fn decimation_scores() {
		assert_eq!(decimate_score(&[0; 16]), 0);
		let mut b = [0i32; 16];
		b[15] = 1;
		assert_eq!(decimate_score(&b), 0); // one level after 15 zeros
		b[0] = 1;
		b[15] = 0;
		assert_eq!(decimate_score(&b), 3);
		b[3] = -2;
		assert_eq!(decimate_score(&b), 9);
	}

	#[test]
	fn satd_of_flat_difference() {
		assert_eq!(satd4(&[2; 16]), 16);
	}
}
