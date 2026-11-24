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

/// 8x8 quantiser multipliers per normAdjust8x8 class (x264
/// `quant8_scale`); with `qbits = 16 + qp / 6` they invert the decoder's
/// `LevelScale8x8` and the 8x8 transform's norm.
const MF8: [[i32; 6]; 6] = [
	[13107, 11428, 20972, 12222, 16777, 15481],
	[11916, 10826, 19174, 11058, 14980, 14290],
	[10082, 8943, 15978, 9675, 12710, 11985],
	[9362, 8228, 14913, 8931, 11984, 11259],
	[8192, 7346, 13159, 7740, 10486, 9777],
	[7282, 6428, 11570, 6830, 9118, 8640],
];

/// normAdjust8x8 class of raster position `pos` (8-318).
#[inline]
fn class8(pos: usize) -> usize {
	let (i, j) = (pos / 8, pos % 8);
	if i % 4 == 0 && j % 4 == 0 {
		0
	} else if i % 2 == 1 && j % 2 == 1 {
		1
	} else if i % 4 == 2 && j % 4 == 2 {
		2
	} else if (i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0) {
		3
	} else if (i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0) {
		4
	} else {
		5
	}
}

fn dct8_1d(s: [i32; 8]) -> [i32; 8] {
	let (s07, s16, s25, s34) = (s[0] + s[7], s[1] + s[6], s[2] + s[5], s[3] + s[4]);
	let (a0, a1, a2, a3) = (s07 + s34, s16 + s25, s07 - s34, s16 - s25);
	let (d07, d16, d25, d34) = (s[0] - s[7], s[1] - s[6], s[2] - s[5], s[3] - s[4]);
	let a4 = d16 + d25 + (d07 + (d07 >> 1));
	let a5 = d07 - d34 - (d25 + (d25 >> 1));
	let a6 = d07 + d34 - (d16 + (d16 >> 1));
	let a7 = d16 - d25 + (d34 + (d34 >> 1));
	[
		a0 + a1,
		a4 + (a7 >> 2),
		a2 + (a3 >> 1),
		a5 + (a6 >> 2),
		a0 - a1,
		a6 - (a5 >> 2),
		(a2 >> 1) - a3,
		(a4 >> 2) - a7,
	]
}

/// Forward 8x8 integer transform of a raster residual block (x264
/// `sub8x8_dct8`; the inverse of the decoder's `idct8_add`).
pub fn fdct8(d: &[i32; 64]) -> [i32; 64] {
	let mut t = [0i32; 64];
	for i in 0..8 {
		let r = dct8_1d(std::array::from_fn(|k| d[i * 8 + k]));
		t[i * 8..i * 8 + 8].copy_from_slice(&r);
	}
	let mut out = [0i32; 64];
	for j in 0..8 {
		let c = dct8_1d(std::array::from_fn(|k| t[k * 8 + j]));
		for k in 0..8 {
			out[k * 8 + j] = c[k];
		}
	}
	out
}

impl Quant {
	/// Quantises an 8x8 block in place (raster). Returns `true` if any
	/// level is non-zero.
	pub fn block8(&self, c: &mut [i32; 64]) -> bool {
		let qbits = self.qbits() + 1;
		let f = self.offset(qbits);
		let mf = &MF8[self.qp % 6];
		let mut any = false;
		for (pos, v) in c.iter_mut().enumerate() {
			*v = Self::level(*v, mf[class8(pos)], f, qbits);
			any |= *v != 0;
		}
		any
	}

	/// The unrounded level `c · MF / 2^qbits` of a 4x4 coefficient.
	#[inline]
	pub fn real4(&self, v: i32, pos: usize) -> f64 {
		f64::from(v) * f64::from(MF[self.qp % 6][class(pos)]) / f64::from(1u32 << self.qbits())
	}

	/// The unrounded level of an 8x8 coefficient.
	#[inline]
	pub fn real8(&self, v: i32, pos: usize) -> f64 {
		f64::from(v) * f64::from(MF8[self.qp % 6][class8(pos)]) / f64::from(1u32 << (self.qbits() + 1))
	}
}

/// Rate-distortion optimal levels of one block for CABAC (trellis
/// quantisation, x264 `quant_trellis_cabac`).
///
/// `z` holds the unrounded levels in scan order (signed). Distortion is
/// measured as `(|z| - level)²`, which is proportional to the squared
/// error of the reconstruction (the quantiser step is uniform in the
/// orthonormal transform domain), and rate as the CABAC cost of the
/// block from the current context states: significance and last flags,
/// `coeff_abs_level_minus1` bins and signs, and coded_block_flag when
/// the block has one. `lambda` converts 1/256 bits into units of
/// squared level error. Each coefficient may keep its rounded level,
/// drop by one, or become zero; the dynamic programme runs over the
/// eight level-context states of §9.3.3.1.3 (plus "nothing coded yet")
/// in the order CABAC codes levels (reverse scan).
pub fn trellis_cabac(
	z: &[f64],
	cat: crate::codecs::h264::cabac::BlockCat,
	cbf_inc: Option<usize>,
	states: &[u8; crate::codecs::h264::cabac::NUM_CTX],
	lambda: f64,
	out: &mut [i32],
) {
	use super::cabac::{bin_cost, cbf_ctx, residual_bases, sig_last_inc};
	use crate::codecs::h264::cabac::BlockCat;
	const NOT: usize = 8;
	let n = z.len();
	out[..n].fill(0);
	let round = |v: f64| (v.abs() + 0.5).floor() as i32;
	let Some(last) = (0..n).rev().find(|&i| round(z[i]) > 0) else { return };
	let (sig_base, last_base, abs_base) = residual_bases(cat);
	let limit = if cat == BlockCat::ChromaDc { 3 } else { 4 };
	let st = |c: usize| states[c];
	// Bits of a level of magnitude `l` coded in state `s` (no sign).
	let level_cost = |s: usize, l: i32| -> u32 {
		let (gt1, eq1) = if s < 4 { (0, s) } else { (s - 3, 0) };
		let first = abs_base + if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
		let minus1 = (l - 1) as u32;
		let mut c = bin_cost(st(first), minus1 > 0);
		if minus1 > 0 {
			let rest = st(abs_base + 5 + gt1.min(limit));
			let ones = minus1.min(14) - 1;
			c += ones * bin_cost(rest, true);
			if minus1 < 14 {
				c += bin_cost(rest, false);
			} else {
				let v = minus1 - 14;
				c += 256 * (2 * (32 - (v + 1).leading_zeros()) - 1);
			}
		}
		c + 256
	};
	let next = |s: usize, l: i32| -> usize {
		match (s < 4, l == 1) {
			(true, true) => (s + 1).min(3),
			(true, false) => 4,
			(false, true) => s,
			(false, false) => (s + 1).min(7),
		}
	};
	let mut cost = [f64::INFINITY; 9];
	cost[NOT] = 0.0;
	// (previous state, level) per position and new state.
	let mut path = vec![[(0u8, 0i32); 9]; last + 1];
	for i in (0..=last).rev() {
		let zi = z[i].abs();
		let r = round(z[i]);
		let (si, li) = if i + 1 < n { sig_last_inc(cat, i) } else { (0, 0) };
		let coded_flags = i + 1 < n;
		let sig0 = if coded_flags { bin_cost(st(sig_base + si), false) } else { 0 };
		let sig1 = if coded_flags { bin_cost(st(sig_base + si), true) } else { 0 };
		let last1 = if coded_flags { bin_cost(st(last_base + li), true) } else { 0 };
		let last0 = if coded_flags { bin_cost(st(last_base + li), false) } else { 0 };
		let mut new = [f64::INFINITY; 9];
		let mut choice = [(0u8, 0i32); 9];
		for s in 0..9 {
			if !cost[s].is_finite() {
				continue;
			}
			let try_level = |l: i32, new: &mut [f64; 9], choice: &mut [(u8, i32); 9]| {
				let d = (zi - f64::from(l)) * (zi - f64::from(l));
				let (rate, ns) = if l == 0 {
					(if s == NOT { 0 } else { sig0 }, s)
				} else {
					let s0 = if s == NOT { 0 } else { s };
					let flags = if s == NOT { sig1 + last1 } else { sig1 + last0 };
					(flags + level_cost(s0, l), next(s0, l))
				};
				let j = cost[s] + d + lambda * f64::from(rate);
				if j < new[ns] {
					new[ns] = j;
					choice[ns] = (s as u8, l);
				}
			};
			try_level(0, &mut new, &mut choice);
			if r >= 1 {
				try_level(r, &mut new, &mut choice);
			}
			if r >= 2 {
				try_level(r - 1, &mut new, &mut choice);
			}
		}
		cost = new;
		path[i] = choice;
	}
	let mut best = (f64::INFINITY, NOT);
	for (s, &c) in cost.iter().enumerate() {
		if !c.is_finite() {
			continue;
		}
		let cbf = cbf_inc.map_or(0, |inc| bin_cost(st(cbf_ctx(cat, inc)), s != NOT));
		let total = c + lambda * f64::from(cbf);
		if total < best.0 {
			best = (total, s);
		}
	}
	let mut s = best.1;
	for i in 0..=last {
		let (prev, l) = path[i][s];
		out[i] = if z[i] < 0.0 { -l } else { l };
		s = usize::from(prev);
	}
}

/// x264's decimation score of an 8x8 block in scan order (`decimate_score64`).
pub fn decimate_score64(scan: &[i32]) -> u32 {
	const TABLE: [u32; 64] = {
		let mut t = [0u32; 64];
		let v = [3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
		let mut i = 0;
		while i < v.len() {
			t[i] = v[i];
			i += 1;
		}
		t
	};
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
		score += TABLE[run.min(63)];
	}
	score
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
			recon::dequant4(&mut c, qp, &dq.l4[0][(qp % 6) as usize], false, 8);
			let mut p = Plane::new(4, 4, 128u8);
			recon::idct4_add(&mut p.d8(), 0, 0, &c);
			let err: i32 =
				(0..16).map(|i| (i32::from(p.data[i]) - 128 - src[i]).abs()).max().unwrap_or(0);
			let step = (0.625 * 2f64.powf(f64::from(qp) / 6.0)) as i32 + 1;
			assert!(err <= step, "qp {qp}: max error {err}");
		}
	}

	#[test]
	fn transform8_quant_round_trips_through_the_decoder_path() {
		let dq = Dequant::new(&ScalingLists::flat());
		let src: [i32; 64] = std::array::from_fn(|i| ((i * 37) % 23) as i32 - 11);
		for qp in [0, 10, 22, 28, 40] {
			let mut c = fdct8(&src);
			Quant::new(qp, true).block8(&mut c);
			recon::dequant8(&mut c, qp, &dq.l8[0][(qp % 6) as usize], 8);
			let mut p = Plane::new(8, 8, 128u8);
			recon::idct8_add(&mut p.d8(), 0, 0, &c);
			let err: i32 =
				(0..64).map(|i| (i32::from(p.data[i]) - 128 - src[i]).abs()).max().unwrap_or(0);
			let step = (0.625 * 2f64.powf(f64::from(qp) / 6.0)) as i32 + 1;
			assert!(err <= step, "qp {qp}: max error {err}");
		}
	}

	#[test]
	fn trellis_never_costs_more_than_rounding() {
		use crate::codecs::h264::cabac::BlockCat;
		use crate::codecs::h264::encoder::cabac::{CabacCost, init_states, residual};
		let states = init_states(false, 0, 26);
		let lambda = 0.136 / 256.0 * 4.0;
		let z: Vec<f64> = (0..16).map(|i| ((i * 7919) % 13) as f64 / 3.0 - 2.0).collect();
		let mut t = [0i32; 16];
		trellis_cabac(&z, BlockCat::Luma4x4, Some(1), &states, lambda, &mut t);
		let rounded: Vec<i32> =
			z.iter().map(|&v| v.signum() as i32 * (v.abs() + 0.5).floor() as i32).collect();
		let j = |levels: &[i32]| {
			let mut c = CabacCost::new(&states);
			residual(&mut c, BlockCat::Luma4x4, Some(1), levels);
			let d: f64 = z.iter().zip(levels).map(|(&a, &b)| (a - f64::from(b)).powi(2)).sum();
			d + lambda * f64::from(c.cost)
		};
		assert!(j(&t) <= j(&rounded) + 1e-9, "{t:?} vs {rounded:?}");
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
