//! CELT decoder (RFC 6716 §4.3), ported from libopus 1.4's float build
//! (BSD-3-Clause): `celt/celt_decoder.c`, `bands.c`, `rate.c`,
//! `quant_bands.c`, `vq.c`, `cwrs.c`, `celt.c`.
//!
//! Only the standard 48 kHz / 960-sample mode exists in Opus, so its
//! parameters are constants here: 21 bands, 120-sample short MDCTs and
//! overlap, frame sizes of 120 << LM samples (LM 0-3).
//!
//! The decode of one frame:
//!
//! 1. global flags: silence, post-filter, transient, intra energy;
//! 2. coarse energy (Laplace), TF changes, spreading, dynamic allocation
//!    boosts, allocation trim;
//! 3. bit allocation ([`compute_allocation`]) and fine energy;
//! 4. PVQ shape decoding with band splitting, stereo and folding
//!    ([`quant_all_bands`]), anti-collapse, final fine energy;
//! 5. denormalisation, inverse MDCT, pitch post-filter, de-emphasis.
//!
//! A lost frame is concealed as in `celt_decode_lost`: after at least two
//! good frames in a row, by repeating the last pitch period of the
//! decoded signal through an LPC model of it, decaying as the signal did
//! ([`CeltDecoder::decode_lost`], pitch branch); after 40 lost units
//! (100 ms), in hybrid mode, or right after a loss or reset, by noise
//! shaped by the last band energies that fade towards the background
//! level (comfort noise).
//!
//! All arithmetic matches the float build's, so output equals libopus up to
//! floating-point rounding (FMA and summation order).

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::mdct::Mdct;
use super::pitch::{self, LPC_ORDER};
use super::range::{BITRES, RangeDecoder, ilog};
use super::tables::{
	BAND_ALLOCATION, CACHE_BITS50, CACHE_CAPS50, CACHE_INDEX50, E_MEANS, E_PROB_MODEL, EBANDS,
	LOGN400, ORDERY_TABLE, WINDOW120,
};

pub const NB_EBANDS: usize = 21;
pub const SHORT_MDCT: usize = 120;
pub const OVERLAP: usize = 120;
pub const MAX_LM: usize = 3;
const DECODE_BUFFER_SIZE: usize = 2048;
const MAX_PERIOD: usize = 1024;
const PLC_PITCH_LAG_MAX: usize = 720;
const PLC_PITCH_LAG_MIN: usize = 100;
const NB_ALLOC_VECTORS: usize = 11;
const PREEMPH: f32 = 0.850_006_1;
const VERY_SMALL: f32 = 1e-30;
const EPSILON: f32 = 1e-15;
const COMBFILTER_MINPERIOD: i32 = 15;
const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const LOG_MAX_PSEUDO: usize = 6;
const ALLOC_STEPS: i32 = 6;

const SPREAD_NONE: i32 = 0;
const SPREAD_NORMAL: i32 = 2;
const SPREAD_AGGRESSIVE: i32 = 3;

const LOG2_FRAC_TABLE: [u8; 24] =
	[0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37];
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];
const TF_SELECT_TABLE: [[i8; 8]; 4] = [
	[0, -1, 0, -1, 0, -1, 0, -1],
	[0, -1, 0, -2, 1, 0, 1, -1],
	[0, -2, 0, -3, 2, 0, 1, -1],
	[0, -2, 0, -3, 3, 0, 1, -1],
];
const PRED_COEF: [f32; 4] =
	[29440.0 / 32768.0, 26112.0 / 32768.0, 21248.0 / 32768.0, 16384.0 / 32768.0];
const BETA_COEF: [f32; 4] =
	[30147.0 / 32768.0, 22282.0 / 32768.0, 12124.0 / 32768.0, 6554.0 / 32768.0];
const BETA_INTRA: f32 = 4915.0 / 32768.0;
const COMB_GAINS: [[f32; 3]; 3] = [
	[0.306_640_63, 0.217_041_02, 0.129_638_67],
	[0.463_867_2, 0.268_066_4, 0.0],
	[0.799_804_7, 0.100_097_66, 0.0],
];

#[inline]
fn eband(i: usize) -> i32 {
	i32::from(EBANDS[i])
}

#[inline]
fn celt_exp2(x: f32) -> f32 {
	(std::f64::consts::LN_2 * f64::from(x)).exp() as f32
}

#[inline]
fn lcg_rand(seed: u32) -> u32 {
	seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

// ---------------------------------------------------------------------
// PVQ codeword counts (cwrs.c). U(N, K) is symmetric and obeys
// U(N, K) = U(N-1, K) + U(N, K-1) + U(N-1, K-1), U(0, 0) = 1.

const PVQ_DIM: usize = 180;

fn pvq_table() -> &'static Vec<u32> {
	static T: OnceLock<Vec<u32>> = OnceLock::new();
	T.get_or_init(|| {
		let mut u = vec![0u64; PVQ_DIM * PVQ_DIM];
		u[0] = 1;
		for n in 1..PVQ_DIM {
			for k in 1..PVQ_DIM {
				let v = u[(n - 1) * PVQ_DIM + k] + u[n * PVQ_DIM + k - 1] + u[(n - 1) * PVQ_DIM + k - 1];
				u[n * PVQ_DIM + k] = v.min(u64::from(u32::MAX) * 4);
			}
		}
		u.into_iter().map(|v| v.min(u64::from(u32::MAX)) as u32).collect()
	})
}

#[inline]
fn pvq_u(n: usize, k: usize) -> u32 {
	let (a, b) = (n.min(k), n.max(k));
	if b >= PVQ_DIM {
		return u32::MAX;
	}
	pvq_table()[a * PVQ_DIM + b]
}

#[inline]
fn pvq_v(n: usize, k: usize) -> u32 {
	pvq_u(n, k).saturating_add(pvq_u(n, k + 1))
}

/// Decodes a PVQ codeword index into pulse positions (`cwrsi`). Returns
/// the squared norm of `y`.
fn cwrsi(mut n: usize, mut k: usize, mut i: u32, y: &mut [i32]) -> f32 {
	let mut yy = 0f32;
	let mut at = 0;
	while n > 2 {
		let val;
		if k >= n {
			let p = pvq_u(n, k + 1);
			let s: i32 = if i >= p { -1 } else { 0 };
			i = i.wrapping_sub(p & (s as u32));
			let k0 = k;
			let q = pvq_u(n, n);
			let mut p;
			if q > i {
				k = n;
				loop {
					k -= 1;
					p = pvq_u(k, n);
					if p <= i || k == 0 {
						break;
					}
				}
			} else {
				p = pvq_u(n, k);
				while p > i && k > 0 {
					k -= 1;
					p = pvq_u(n, k);
				}
			}
			i = i.wrapping_sub(p);
			val = ((k0 - k) as i32 + s) ^ s;
		} else {
			let p = pvq_u(k, n);
			let q = pvq_u(k + 1, n);
			if p <= i && i < q {
				i -= p;
				val = 0;
			} else {
				let s: i32 = if i >= q { -1 } else { 0 };
				i = i.wrapping_sub(q & (s as u32));
				let k0 = k;
				let mut p;
				loop {
					if k == 0 {
						p = 0;
						break;
					}
					k -= 1;
					p = pvq_u(k, n);
					if p <= i {
						break;
					}
				}
				i = i.wrapping_sub(p);
				val = ((k0 - k) as i32 + s) ^ s;
			}
		}
		y[at] = val;
		at += 1;
		yy += (val * val) as f32;
		n -= 1;
	}
	// n == 2
	let p = 2 * k as u32 + 1;
	let s: i32 = if i >= p { -1 } else { 0 };
	i = i.wrapping_sub(p & (s as u32));
	let k0 = k;
	k = ((i.wrapping_add(1)) >> 1) as usize;
	if k > 0 {
		i = i.wrapping_sub(2 * k as u32 - 1);
	}
	let val = ((k0 as i32 - k as i32) + s) ^ s;
	y[at] = val;
	yy += (val * val) as f32;
	// n == 1
	let s = -(i as i32);
	let val = (k as i32 + s) ^ s;
	y[at + 1] = val;
	yy += (val * val) as f32;
	yy
}

// ---------------------------------------------------------------------
// Allocation (rate.c).

fn cache_for(band: usize, lm: i32) -> &'static [u8] {
	let index = CACHE_INDEX50[((lm + 1) as usize) * NB_EBANDS + band];
	if index < 0 {
		return &[0];
	}
	&CACHE_BITS50[index as usize..]
}

#[inline]
fn get_pulses(i: i32) -> i32 {
	if i < 8 { i } else { (8 + (i & 7)) << ((i >> 3) - 1) }
}

fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
	let cache = cache_for(band, lm);
	let mut lo = 0i32;
	let mut hi = i32::from(cache[0]);
	let bits = bits - 1;
	for _ in 0..LOG_MAX_PSEUDO {
		let mid = (lo + hi + 1) >> 1;
		if i32::from(cache[mid as usize]) >= bits {
			hi = mid;
		} else {
			lo = mid;
		}
	}
	let lo_bits = if lo == 0 { -1 } else { i32::from(cache[lo as usize]) };
	if bits - lo_bits <= i32::from(cache[hi as usize]) - bits { lo } else { hi }
}

fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
	let cache = cache_for(band, lm);
	if pulses == 0 { 0 } else { i32::from(cache[pulses as usize]) + 1 }
}

fn init_caps(lm: usize, c: usize) -> [i32; NB_EBANDS] {
	let mut cap = [0; NB_EBANDS];
	for (i, slot) in cap.iter_mut().enumerate() {
		let n = (eband(i + 1) - eband(i)) << lm;
		*slot = ((i32::from(CACHE_CAPS50[NB_EBANDS * (2 * lm + c - 1) + i]) + 64) * c as i32 * n) >> 2;
	}
	cap
}

/// Result of the bit allocation.
struct Allocation {
	coded_bands: usize,
	intensity: usize,
	dual_stereo: bool,
	balance: i32,
	pulses: [i32; NB_EBANDS],
	fine_quant: [i32; NB_EBANDS],
	fine_priority: [i32; NB_EBANDS],
}

#[allow(clippy::too_many_lines)]
fn compute_allocation(
	start: usize,
	end: usize,
	offsets: &[i32; NB_EBANDS],
	cap: &[i32; NB_EBANDS],
	alloc_trim: i32,
	total: i32,
	c: usize,
	lm: usize,
	dec: &mut RangeDecoder,
) -> Allocation {
	let ci = c as i32;
	let lmi = lm as i32;
	let mut total = total.max(0);
	let mut skip_start = start;
	let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
	total -= skip_rsv;
	let mut intensity_rsv = 0;
	let mut dual_stereo_rsv = 0;
	if c == 2 {
		intensity_rsv = i32::from(LOG2_FRAC_TABLE[end - start]);
		if intensity_rsv > total {
			intensity_rsv = 0;
		} else {
			total -= intensity_rsv;
			dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
			total -= dual_stereo_rsv;
		}
	}
	let mut thresh = [0i32; NB_EBANDS];
	let mut trim_offset = [0i32; NB_EBANDS];
	for j in start..end {
		let n = eband(j + 1) - eband(j);
		thresh[j] = (ci << BITRES).max(((3 * n) << lmi << BITRES) >> 4);
		trim_offset[j] = (ci
			* n * (alloc_trim - 5 - lmi)
			* (end as i32 - j as i32 - 1)
			* (1 << (lmi + BITRES as i32)))
			>> 6;
		if n << lmi == 1 {
			trim_offset[j] -= ci << BITRES;
		}
	}
	let alloc = |v: usize, j: usize| i32::from(BAND_ALLOCATION[v * NB_EBANDS + j]);
	let mut lo = 1i32;
	let mut hi = NB_ALLOC_VECTORS as i32 - 1;
	loop {
		let mut done = false;
		let mut psum = 0i32;
		let mid = (lo + hi) >> 1;
		for j in (start..end).rev() {
			let n = eband(j + 1) - eband(j);
			let mut bitsj = (ci * n * alloc(mid as usize, j)) << lmi >> 2;
			if bitsj > 0 {
				bitsj = (bitsj + trim_offset[j]).max(0);
			}
			bitsj += offsets[j];
			if bitsj >= thresh[j] || done {
				done = true;
				psum += bitsj.min(cap[j]);
			} else if bitsj >= ci << BITRES {
				psum += ci << BITRES;
			}
		}
		if psum > total {
			hi = mid - 1;
		} else {
			lo = mid + 1;
		}
		if lo > hi {
			break;
		}
	}
	hi = lo;
	lo -= 1;
	let mut bits1 = [0i32; NB_EBANDS];
	let mut bits2 = [0i32; NB_EBANDS];
	for j in start..end {
		let n = eband(j + 1) - eband(j);
		let mut bits1j = (ci * n * alloc(lo as usize, j)) << lmi >> 2;
		let mut bits2j = if hi as usize >= NB_ALLOC_VECTORS {
			cap[j]
		} else {
			(ci * n * alloc(hi as usize, j)) << lmi >> 2
		};
		if bits1j > 0 {
			bits1j = (bits1j + trim_offset[j]).max(0);
		}
		if bits2j > 0 {
			bits2j = (bits2j + trim_offset[j]).max(0);
		}
		if lo > 0 {
			bits1j += offsets[j];
		}
		bits2j += offsets[j];
		if offsets[j] > 0 {
			skip_start = j;
		}
		bits2j = (bits2j - bits1j).max(0);
		bits1[j] = bits1j;
		bits2[j] = bits2j;
	}

	// interp_bits2pulses
	let alloc_floor = ci << BITRES;
	let stereo = c > 1;
	let log_m = lmi << BITRES;
	let (mut lo, mut hi) = (0i32, 1i32 << ALLOC_STEPS);
	for _ in 0..ALLOC_STEPS {
		let mid = (lo + hi) >> 1;
		let mut psum = 0;
		let mut done = false;
		for j in (start..end).rev() {
			let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
			if tmp >= thresh[j] || done {
				done = true;
				psum += tmp.min(cap[j]);
			} else if tmp >= alloc_floor {
				psum += alloc_floor;
			}
		}
		if psum > total {
			hi = mid;
		} else {
			lo = mid;
		}
	}
	let mut psum = 0;
	let mut done = false;
	let mut bits = [0i32; NB_EBANDS];
	for j in (start..end).rev() {
		let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
		if tmp < thresh[j] && !done {
			tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
		} else {
			done = true;
		}
		tmp = tmp.min(cap[j]);
		bits[j] = tmp;
		psum += tmp;
	}

	let mut coded_bands = end;
	loop {
		let j = coded_bands - 1;
		if j <= skip_start {
			total += skip_rsv;
			break;
		}
		let mut left = total - psum;
		let span = eband(coded_bands) - eband(start);
		let percoeff = if span > 0 { left / span } else { 0 };
		left -= span * percoeff;
		let rem = (left - (eband(j) - eband(start))).max(0);
		let band_width = eband(coded_bands) - eband(j);
		let mut band_bits = bits[j] + percoeff * band_width + rem;
		if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
			if dec.bit_logp(1) {
				break;
			}
			psum += 1 << BITRES;
			band_bits -= 1 << BITRES;
		}
		psum -= bits[j] + intensity_rsv;
		if intensity_rsv > 0 {
			intensity_rsv = i32::from(LOG2_FRAC_TABLE[j - start]);
		}
		psum += intensity_rsv;
		if band_bits >= alloc_floor {
			psum += alloc_floor;
			bits[j] = alloc_floor;
		} else {
			bits[j] = 0;
		}
		coded_bands -= 1;
	}

	let mut intensity = 0usize;
	if intensity_rsv > 0 {
		intensity = start + dec.uint((coded_bands + 1 - start) as u32) as usize;
	}
	if intensity <= start {
		total += dual_stereo_rsv;
		dual_stereo_rsv = 0;
	}
	let dual_stereo = dual_stereo_rsv > 0 && dec.bit_logp(1);

	let mut left = total - psum;
	let span = eband(coded_bands) - eband(start);
	let percoeff = if span > 0 { left / span } else { 0 };
	left -= span * percoeff;
	for j in start..coded_bands {
		bits[j] += percoeff * (eband(j + 1) - eband(j));
	}
	for j in start..coded_bands {
		let tmp = left.min(eband(j + 1) - eband(j));
		bits[j] += tmp;
		left -= tmp;
	}

	let mut ebits = [0i32; NB_EBANDS];
	let mut fine_priority = [0i32; NB_EBANDS];
	let mut balance = 0i32;
	let mut j = start;
	while j < coded_bands {
		let n0 = eband(j + 1) - eband(j);
		let n = n0 << lmi;
		let bit = bits[j] + balance;
		let excess;
		if n > 1 {
			excess = (bit - cap[j]).max(0);
			bits[j] = bit - excess;
			let den = ci * n + i32::from(c == 2 && n > 2 && !dual_stereo && j < intensity);
			let nclogn = den * (i32::from(LOGN400[j]) + log_m);
			let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
			if n == 2 {
				offset += den << BITRES >> 2;
			}
			if bits[j] + offset < (den * 2) << BITRES {
				offset += nclogn >> 2;
			} else if bits[j] + offset < (den * 3) << BITRES {
				offset += nclogn >> 3;
			}
			ebits[j] = (bits[j] + offset + (den << (BITRES - 1))).max(0);
			ebits[j] = (ebits[j] / den) >> BITRES;
			if ci * ebits[j] > (bits[j] >> BITRES) {
				ebits[j] = bits[j] >> i32::from(stereo) >> BITRES;
			}
			ebits[j] = ebits[j].min(MAX_FINE_BITS);
			fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
			bits[j] -= (ci * ebits[j]) << BITRES;
		} else {
			excess = (bit - (ci << BITRES)).max(0);
			bits[j] = bit - excess;
			ebits[j] = 0;
			fine_priority[j] = 1;
		}
		let mut excess = excess;
		if excess > 0 {
			let extra_fine =
				(excess >> (i32::from(stereo) + BITRES as i32)).min(MAX_FINE_BITS - ebits[j]);
			ebits[j] += extra_fine;
			let extra_bits = (extra_fine * ci) << BITRES;
			fine_priority[j] = i32::from(extra_bits >= excess - balance);
			excess -= extra_bits;
		}
		balance = excess;
		j += 1;
	}
	while j < end {
		ebits[j] = bits[j] >> i32::from(stereo) >> BITRES;
		bits[j] = 0;
		fine_priority[j] = i32::from(ebits[j] < 1);
		j += 1;
	}
	Allocation {
		coded_bands,
		intensity,
		dual_stereo,
		balance,
		pulses: bits,
		fine_quant: ebits,
		fine_priority,
	}
}

// ---------------------------------------------------------------------
// Energy (quant_bands.c).

fn unquant_coarse_energy(
	old: &mut [f32; 2 * NB_EBANDS],
	start: usize,
	end: usize,
	intra: bool,
	dec: &mut RangeDecoder,
	c: usize,
	lm: usize,
) {
	let model = &E_PROB_MODEL[(lm * 2 + usize::from(intra)) * 42..][..42];
	let (coef, beta) = if intra { (0.0, BETA_INTRA) } else { (PRED_COEF[lm], BETA_COEF[lm]) };
	let mut prev = [0f32; 2];
	let budget = (dec.storage() * 8) as i32;
	for i in start..end {
		for ch in 0..c {
			let tell = dec.tell();
			let qi = if budget - tell >= 15 {
				let pi = 2 * i.min(20);
				dec.laplace(u32::from(model[pi]) << 7, i32::from(model[pi + 1]) << 6)
			} else if budget - tell >= 2 {
				let q = dec.icdf(&SMALL_ENERGY_ICDF, 2) as i32;
				(q >> 1) ^ -(q & 1)
			} else if budget - tell >= 1 {
				-i32::from(dec.bit_logp(1))
			} else {
				-1
			};
			let q = qi as f32;
			let e = &mut old[i + ch * NB_EBANDS];
			*e = e.max(-9.0);
			let tmp = coef * *e + prev[ch] + q;
			*e = tmp;
			prev[ch] = prev[ch] + q - beta * q;
		}
	}
}

fn unquant_fine_energy(
	old: &mut [f32; 2 * NB_EBANDS],
	start: usize,
	end: usize,
	fine: &[i32; NB_EBANDS],
	dec: &mut RangeDecoder,
	c: usize,
) {
	for i in start..end {
		if fine[i] <= 0 {
			continue;
		}
		for ch in 0..c {
			let q2 = dec.bits(fine[i] as u32);
			let offset = (q2 as f32 + 0.5) * (1 << (14 - fine[i])) as f32 * (1.0 / 16384.0) - 0.5;
			old[i + ch * NB_EBANDS] += offset;
		}
	}
}

fn unquant_energy_finalise(
	old: &mut [f32; 2 * NB_EBANDS],
	start: usize,
	end: usize,
	fine: &[i32; NB_EBANDS],
	priority: &[i32; NB_EBANDS],
	mut bits_left: i32,
	dec: &mut RangeDecoder,
	c: usize,
) {
	for prio in 0..2 {
		let mut i = start;
		while i < end && bits_left >= c as i32 {
			if fine[i] >= MAX_FINE_BITS || priority[i] != prio {
				i += 1;
				continue;
			}
			for ch in 0..c {
				let q2 = dec.bits(1);
				let offset = (q2 as f32 - 0.5) * (1 << (14 - fine[i] - 1)) as f32 * (1.0 / 16384.0);
				old[i + ch * NB_EBANDS] += offset;
				bits_left -= 1;
			}
			i += 1;
		}
	}
}

// ---------------------------------------------------------------------
// Vector quantisation helpers (vq.c, bands.c).

fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
	let ms = -s;
	if len <= stride {
		return;
	}
	for i in 0..len - stride {
		let x1 = x[i];
		let x2 = x[i + stride];
		x[i + stride] = c * x2 + s * x1;
		x[i] = c * x1 + ms * x2;
	}
	if len < 2 * stride + 1 {
		return;
	}
	for i in (0..=len - 2 * stride - 1).rev() {
		let x1 = x[i];
		let x2 = x[i + stride];
		x[i + stride] = c * x2 + s * x1;
		x[i] = c * x1 + ms * x2;
	}
}

fn exp_rotation(x: &mut [f32], len: usize, dir: i32, stride: usize, k: i32, spread: i32) {
	const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
	if 2 * k >= len as i32 || spread == SPREAD_NONE {
		return;
	}
	let factor = SPREAD_FACTOR[(spread - 1) as usize];
	let gain = len as f32 / (len as i32 + factor * k) as f32;
	let theta = 0.5 * (gain * gain);
	let c = (f64::from(0.5 * std::f32::consts::PI) * f64::from(theta)).cos() as f32;
	let s = (f64::from(0.5 * std::f32::consts::PI) * f64::from(1.0 - theta)).cos() as f32;
	let mut stride2 = 0usize;
	if len >= 8 * stride {
		stride2 = 1;
		while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
			stride2 += 1;
		}
	}
	let sub = len / stride;
	for i in 0..stride {
		let block = &mut x[i * sub..(i + 1) * sub];
		if dir < 0 {
			if stride2 > 0 {
				exp_rotation1(block, sub, stride2, s, c);
			}
			exp_rotation1(block, sub, 1, c, s);
		} else {
			exp_rotation1(block, sub, 1, c, -s);
			if stride2 > 0 {
				exp_rotation1(block, sub, stride2, s, -c);
			}
		}
	}
}

fn renormalise_vector(x: &mut [f32], gain: f32) {
	let e = EPSILON + x.iter().map(|v| v * v).sum::<f32>();
	let g = (1.0 / e.sqrt()) * gain;
	for v in x.iter_mut() {
		*v *= g;
	}
}

fn alg_unquant(
	x: &mut [f32],
	n: usize,
	k: i32,
	spread: i32,
	b: usize,
	dec: &mut RangeDecoder,
	gain: f32,
) -> u32 {
	let mut iy = vec![0i32; n];
	let v = pvq_v(n, k as usize);
	let index = dec.uint(v);
	let ryy = cwrsi(n, k as usize, index, &mut iy);
	let g = (1.0 / ryy.sqrt()) * gain;
	for (xv, &yv) in x.iter_mut().zip(&iy) {
		*xv = g * yv as f32;
	}
	exp_rotation(x, n, -1, b, k, spread);
	if b <= 1 {
		return 1;
	}
	let n0 = n / b;
	let mut mask = 0u32;
	for i in 0..b {
		if iy[i * n0..(i + 1) * n0].iter().any(|&v| v != 0) {
			mask |= 1 << i;
		}
	}
	mask
}

fn haar1(x: &mut [f32], n0: usize, stride: usize) {
	let n0 = n0 >> 1;
	let s = std::f32::consts::FRAC_1_SQRT_2;
	for i in 0..stride {
		for j in 0..n0 {
			let t1 = s * x[stride * 2 * j + i];
			let t2 = s * x[stride * (2 * j + 1) + i];
			x[stride * 2 * j + i] = t1 + t2;
			x[stride * (2 * j + 1) + i] = t1 - t2;
		}
	}
}

fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
	let n = n0 * stride;
	let mut tmp = vec![0f32; n];
	if hadamard {
		let ordery = &ORDERY_TABLE[stride - 2..];
		for i in 0..stride {
			for j in 0..n0 {
				tmp[ordery[i] as usize * n0 + j] = x[j * stride + i];
			}
		}
	} else {
		for i in 0..stride {
			for j in 0..n0 {
				tmp[i * n0 + j] = x[j * stride + i];
			}
		}
	}
	x[..n].copy_from_slice(&tmp);
}

fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
	let n = n0 * stride;
	let mut tmp = vec![0f32; n];
	if hadamard {
		let ordery = &ORDERY_TABLE[stride - 2..];
		for i in 0..stride {
			for j in 0..n0 {
				tmp[j * stride + i] = x[ordery[i] as usize * n0 + j];
			}
		}
	} else {
		for i in 0..stride {
			for j in 0..n0 {
				tmp[j * stride + i] = x[i * n0 + j];
			}
		}
	}
	x[..n].copy_from_slice(&tmp);
}

#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
	(16384 + i32::from(a as i16) * i32::from(b as i16)) >> 15
}

fn bitexact_cos(x: i16) -> i16 {
	let tmp = (4096 + i32::from(x) * i32::from(x)) >> 13;
	let x2 = tmp;
	let x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
	(1 + x2) as i16
}

fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
	let lc = ilog(icos as u32) as i32;
	let ls = ilog(isin as u32) as i32;
	let icos = icos << (15 - lc);
	let isin = isin << (15 - ls);
	(ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
		- frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

fn isqrt32(v: u32) -> u32 {
	(f64::from(v).sqrt()) as u32
}

fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
	const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
	let mut n2 = 2 * n - 1;
	if stereo && n == 2 {
		n2 -= 1;
	}
	// celt_sudiv: integer division truncating towards zero, like `/`.
	let mut qb = ((b + n2 * offset) / n2).min(b - pulse_cap - (4 << BITRES));
	qb = qb.min(8 << BITRES);
	if qb < (1 << BITRES >> 1) {
		1
	} else {
		let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
		(qn + 1) >> 1 << 1
	}
}

struct BandCtx<'a, 'b> {
	dec: &'a mut RangeDecoder<'b>,
	band: usize,
	intensity: usize,
	spread: i32,
	tf_change: i32,
	remaining_bits: i32,
	seed: u32,
	disable_inv: bool,
	avoid_split_noise: bool,
}

struct Split {
	inv: bool,
	imid: i32,
	iside: i32,
	delta: i32,
	itheta: i32,
	qalloc: i32,
}

fn compute_theta(
	ctx: &mut BandCtx,
	n: i32,
	b: &mut i32,
	bb: i32,
	b0: i32,
	lm: i32,
	stereo: bool,
	fill: &mut u32,
) -> Split {
	let i = ctx.band;
	let pulse_cap = i32::from(LOGN400[i]) + lm * (1 << BITRES);
	let offset =
		(pulse_cap >> 1) - if stereo && n == 2 { QTHETA_OFFSET_TWOPHASE } else { QTHETA_OFFSET };
	let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
	if stereo && i >= ctx.intensity {
		qn = 1;
	}
	let tell = ctx.dec.tell_frac() as i32;
	let mut itheta = 0i32;
	let mut inv = false;
	if qn != 1 {
		if stereo && n > 2 {
			let p0 = 3;
			let x0 = qn / 2;
			let ft = p0 * (x0 + 1) + x0;
			let fs = ctx.dec.decode(ft as u32) as i32;
			let x = if fs < (x0 + 1) * p0 { fs / p0 } else { x0 + 1 + (fs - (x0 + 1) * p0) };
			let (fl, fh) = if x <= x0 {
				(p0 * x, p0 * (x + 1))
			} else {
				((x - 1 - x0) + (x0 + 1) * p0, (x - x0) + (x0 + 1) * p0)
			};
			ctx.dec.update(fl as u32, fh as u32, ft as u32);
			itheta = x;
		} else if b0 > 1 || stereo {
			itheta = ctx.dec.uint((qn + 1) as u32) as i32;
		} else {
			let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
			let fm = ctx.dec.decode(ft as u32) as i32;
			let (fs, fl);
			if fm < (((qn >> 1) * ((qn >> 1) + 1)) >> 1) {
				itheta = (isqrt32(8 * fm as u32 + 1) as i32 - 1) >> 1;
				fs = itheta + 1;
				fl = (itheta * (itheta + 1)) >> 1;
			} else {
				itheta = (2 * (qn + 1) - isqrt32(8 * (ft - fm - 1) as u32 + 1) as i32) >> 1;
				fs = qn + 1 - itheta;
				fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
			}
			ctx.dec.update(fl as u32, (fl + fs) as u32, ft as u32);
		}
		itheta = itheta * 16384 / qn;
	} else if stereo {
		if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
			inv = ctx.dec.bit_logp(2);
		}
		if ctx.disable_inv {
			inv = false;
		}
		itheta = 0;
	}
	let qalloc = ctx.dec.tell_frac() as i32 - tell;
	*b -= qalloc;
	let (imid, iside, delta);
	if itheta == 0 {
		imid = 32767;
		iside = 0;
		*fill &= (1 << bb) - 1;
		delta = -16384;
	} else if itheta == 16384 {
		imid = 0;
		iside = 32767;
		*fill &= ((1 << bb) - 1) << bb;
		delta = 16384;
	} else {
		imid = i32::from(bitexact_cos(itheta as i16));
		iside = i32::from(bitexact_cos((16384 - itheta) as i16));
		delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
	}
	Split { inv, imid, iside, delta, itheta, qalloc }
}

fn quant_band_n1(
	ctx: &mut BandCtx,
	x: &mut [f32],
	y: Option<&mut [f32]>,
	lowband_out: Option<&mut [f32]>,
) -> u32 {
	let decode_sign = |ctx: &mut BandCtx, v: &mut [f32]| {
		let mut sign = 0;
		if ctx.remaining_bits >= 1 << BITRES {
			sign = ctx.dec.bits(1);
			ctx.remaining_bits -= 1 << BITRES;
		}
		v[0] = if sign != 0 { -1.0 } else { 1.0 };
	};
	decode_sign(ctx, x);
	if let Some(y) = y {
		decode_sign(ctx, y);
	}
	if let Some(out) = lowband_out {
		out[0] = x[0];
	}
	1
}

fn quant_partition(
	ctx: &mut BandCtx,
	x: &mut [f32],
	n: usize,
	mut b: i32,
	mut bb: i32,
	lowband: Option<&[f32]>,
	mut lm: i32,
	gain: f32,
	mut fill: u32,
) -> u32 {
	let i = ctx.band;
	let b0 = bb;
	let cache = cache_for(i, lm);
	if lm != -1 && b > i32::from(cache[usize::from(cache[0])]) + 12 && n > 2 {
		let half = n >> 1;
		let (xs, ys) = x.split_at_mut(half);
		lm -= 1;
		if bb == 1 {
			fill = (fill & 1) | (fill << 1);
		}
		bb = (bb + 1) >> 1;
		let split = compute_theta(ctx, half as i32, &mut b, bb, b0, lm, false, &mut fill);
		let mid = (1.0 / 32768.0) * split.imid as f32;
		let side = (1.0 / 32768.0) * split.iside as f32;
		let mut delta = split.delta;
		if b0 > 1 && (split.itheta & 0x3fff) != 0 {
			if split.itheta > 8192 {
				delta -= delta >> (4 - lm);
			} else {
				delta = (delta + ((half as i32) << BITRES >> (5 - lm))).min(0);
			}
		}
		let mbits = (b - delta) / 2;
		let mbits = mbits.min(b).max(0);
		let mut sbits = b - mbits;
		let mut mbits = mbits;
		ctx.remaining_bits -= split.qalloc;
		let next_lowband2 = lowband.map(|l| &l[half.min(l.len())..]);
		let mut rebalance = ctx.remaining_bits;
		let cm;
		if mbits >= sbits {
			let c1 = quant_partition(ctx, xs, half, mbits, bb, lowband, lm, gain * mid, fill);
			rebalance = mbits - (rebalance - ctx.remaining_bits);
			if rebalance > 3 << BITRES && split.itheta != 0 {
				sbits += rebalance - (3 << BITRES);
			}
			let c2 =
				quant_partition(ctx, ys, half, sbits, bb, next_lowband2, lm, gain * side, fill >> bb);
			cm = c1 | c2 << (b0 >> 1);
		} else {
			let c2 =
				quant_partition(ctx, ys, half, sbits, bb, next_lowband2, lm, gain * side, fill >> bb);
			rebalance = sbits - (rebalance - ctx.remaining_bits);
			if rebalance > 3 << BITRES && split.itheta != 16384 {
				mbits += rebalance - (3 << BITRES);
			}
			let c1 = quant_partition(ctx, xs, half, mbits, bb, lowband, lm, gain * mid, fill);
			cm = c1 | c2 << (b0 >> 1);
		}
		return cm;
	}
	let mut q = bits2pulses(i, lm, b);
	let mut curr_bits = pulses2bits(i, lm, q);
	ctx.remaining_bits -= curr_bits;
	while ctx.remaining_bits < 0 && q > 0 {
		ctx.remaining_bits += curr_bits;
		q -= 1;
		curr_bits = pulses2bits(i, lm, q);
		ctx.remaining_bits -= curr_bits;
	}
	if q != 0 {
		let k = get_pulses(q);
		return alg_unquant(&mut x[..n], n, k, ctx.spread, bb as usize, ctx.dec, gain);
	}
	let cm_mask = ((1u64 << bb) - 1) as u32;
	fill &= cm_mask;
	if fill == 0 {
		x[..n].fill(0.0);
		return 0;
	}
	let cm;
	match lowband {
		None => {
			for v in x[..n].iter_mut() {
				ctx.seed = lcg_rand(ctx.seed);
				*v = ((ctx.seed as i32) >> 20) as f32;
			}
			cm = cm_mask;
		}
		Some(l) => {
			for j in 0..n {
				ctx.seed = lcg_rand(ctx.seed);
				let tmp = if ctx.seed & 0x8000 != 0 { 1.0 / 256.0 } else { -1.0 / 256.0 };
				x[j] = l.get(j).copied().unwrap_or(0.0) + tmp;
			}
			cm = fill;
		}
	}
	renormalise_vector(&mut x[..n], gain);
	cm
}

fn quant_band(
	ctx: &mut BandCtx,
	x: &mut [f32],
	n: usize,
	b: i32,
	mut bb: i32,
	lowband: Option<&[f32]>,
	lm: i32,
	lowband_out: Option<&mut [f32]>,
	gain: f32,
	mut fill: u32,
) -> u32 {
	const BIT_INTERLEAVE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
	const BIT_DEINTERLEAVE: [u8; 16] = [
		0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
	];
	let n0 = n;
	let mut n_b = n;
	let b0 = bb;
	let long_blocks = b0 == 1;
	n_b /= bb as usize;
	if n == 1 {
		return quant_band_n1(ctx, x, None, lowband_out);
	}
	let mut tf_change = ctx.tf_change;
	let recombine = tf_change.max(0);
	// Always work on a copy of the folding source (the reference copies it
	// to scratch space whenever it modifies it).
	let mut low: Option<Vec<f32>> = lowband.map(|l| l[..n.min(l.len())].to_vec());
	if let Some(l) = low.as_mut() {
		l.resize(n, 0.0);
	}
	for k in 0..recombine {
		if let Some(l) = low.as_mut() {
			haar1(l, n >> k, 1 << k);
		}
		fill = u32::from(BIT_INTERLEAVE[(fill & 0xF) as usize])
			| u32::from(BIT_INTERLEAVE[((fill >> 4) & 0xF) as usize]) << 2;
	}
	bb >>= recombine;
	n_b <<= recombine;
	let mut time_divide = 0;
	while (n_b & 1) == 0 && tf_change < 0 {
		if let Some(l) = low.as_mut() {
			haar1(l, n_b, bb as usize);
		}
		fill |= fill << bb;
		bb <<= 1;
		n_b >>= 1;
		time_divide += 1;
		tf_change += 1;
	}
	let b0b = bb;
	let n_b0 = n_b;
	if b0b > 1 {
		if let Some(l) = low.as_mut() {
			deinterleave_hadamard(l, n_b >> recombine, (b0b << recombine) as usize, long_blocks);
		}
	}
	let mut cm = quant_partition(ctx, x, n, b, bb, low.as_deref(), lm, gain, fill);
	if b0b > 1 {
		interleave_hadamard(x, n_b >> recombine, (b0b << recombine) as usize, long_blocks);
	}
	let mut n_b = n_b0;
	bb = b0b;
	for _ in 0..time_divide {
		bb >>= 1;
		n_b <<= 1;
		cm |= cm >> bb;
		haar1(x, n_b, bb as usize);
	}
	for k in 0..recombine {
		cm = u32::from(BIT_DEINTERLEAVE[(cm & 0xF) as usize]);
		haar1(x, n0 >> k, 1 << k);
	}
	bb <<= recombine;
	if let Some(out) = lowband_out {
		let scale = (n0 as f32).sqrt();
		for j in 0..n0 {
			out[j] = scale * x[j];
		}
	}
	let _ = b0;
	cm & ((1u32 << bb) - 1)
}

fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
	let mut xp = 0f32;
	let mut side = 0f32;
	for j in 0..n {
		xp += y[j] * x[j];
		side += y[j] * y[j];
	}
	xp *= mid;
	let mid2 = mid;
	let el = mid2 * mid2 + side - 2.0 * xp;
	let er = mid2 * mid2 + side + 2.0 * xp;
	if er < 6e-4 || el < 6e-4 {
		y[..n].copy_from_slice(&x[..n]);
		return;
	}
	let lgain = 1.0 / el.sqrt();
	let rgain = 1.0 / er.sqrt();
	for j in 0..n {
		let l = mid * x[j];
		let r = y[j];
		x[j] = lgain * (l - r);
		y[j] = rgain * (l + r);
	}
}

fn quant_band_stereo(
	ctx: &mut BandCtx,
	x: &mut [f32],
	y: &mut [f32],
	n: usize,
	mut b: i32,
	bb: i32,
	lowband: Option<&[f32]>,
	lm: i32,
	lowband_out: Option<&mut [f32]>,
	mut fill: u32,
) -> u32 {
	if n == 1 {
		return quant_band_n1(ctx, x, Some(y), lowband_out);
	}
	let orig_fill = fill;
	let split = compute_theta(ctx, n as i32, &mut b, bb, bb, lm, true, &mut fill);
	let mid = (1.0 / 32768.0) * split.imid as f32;
	let side = (1.0 / 32768.0) * split.iside as f32;
	let itheta = split.itheta;
	let mut cm;
	if n == 2 {
		let mut mbits = b;
		let mut sbits = 0;
		if itheta != 0 && itheta != 16384 {
			sbits = 1 << BITRES;
		}
		mbits -= sbits;
		let c = itheta > 8192;
		ctx.remaining_bits -= split.qalloc + sbits;
		let mut sign = 0;
		if sbits != 0 {
			sign = ctx.dec.bits(1) as i32;
		}
		let sign = 1 - 2 * sign;
		let (x2, y2): (&mut [f32], &mut [f32]) = if c { (y, x) } else { (x, y) };
		cm = quant_band(ctx, x2, n, mbits, bb, lowband, lm, lowband_out, 1.0, orig_fill);
		y2[0] = -(sign as f32) * x2[1];
		y2[1] = sign as f32 * x2[0];
		let (x, y) = if c { (y2, x2) } else { (x2, y2) };
		x[0] *= mid;
		x[1] *= mid;
		y[0] *= side;
		y[1] *= side;
		let tmp = x[0];
		x[0] = tmp - y[0];
		y[0] += tmp;
		let tmp = x[1];
		x[1] = tmp - y[1];
		y[1] += tmp;
		if split.inv {
			y[..n].iter_mut().for_each(|v| *v = -*v);
		}
		return cm;
	}
	let mbits = ((b - split.delta) / 2).min(b).max(0);
	let mut sbits = b - mbits;
	let mut mbits = mbits;
	ctx.remaining_bits -= split.qalloc;
	let mut rebalance = ctx.remaining_bits;
	if mbits >= sbits {
		cm = quant_band(ctx, x, n, mbits, bb, lowband, lm, lowband_out, 1.0, fill);
		rebalance = mbits - (rebalance - ctx.remaining_bits);
		if rebalance > 3 << BITRES && itheta != 0 {
			sbits += rebalance - (3 << BITRES);
		}
		cm |= quant_band(ctx, y, n, sbits, bb, None, lm, None, side, fill >> bb);
	} else {
		cm = quant_band(ctx, y, n, sbits, bb, None, lm, None, side, fill >> bb);
		rebalance = sbits - (rebalance - ctx.remaining_bits);
		if rebalance > 3 << BITRES && itheta != 16384 {
			mbits += rebalance - (3 << BITRES);
		}
		cm |= quant_band(ctx, x, n, mbits, bb, lowband, lm, lowband_out, 1.0, fill);
	}
	stereo_merge(x, y, mid, n);
	if split.inv {
		y[..n].iter_mut().for_each(|v| *v = -*v);
	}
	cm
}

/// Decodes the normalised shape of every band into `x` (and `y` for
/// stereo). Returns the collapse masks.
fn quant_all_bands(
	dec: &mut RangeDecoder,
	start: usize,
	end: usize,
	x: &mut [f32],
	mut y: Option<&mut [f32]>,
	collapse_masks: &mut [u8],
	pulses: &[i32; NB_EBANDS],
	short_blocks: bool,
	spread: i32,
	mut dual_stereo: bool,
	intensity: usize,
	tf_res: &[i32; NB_EBANDS],
	total_bits: i32,
	mut balance: i32,
	lm: usize,
	coded_bands: usize,
	seed: &mut u32,
	disable_inv: bool,
) {
	let m = 1usize << lm;
	let bb: i32 = if short_blocks { m as i32 } else { 1 };
	let c = if y.is_some() { 2 } else { 1 };
	let norm_offset = m * EBANDS[start] as usize;
	let norm_len = m * EBANDS[NB_EBANDS - 1] as usize - norm_offset;
	let mut norm = vec![0f32; norm_len];
	let mut norm2 = vec![0f32; norm_len];
	let mut lowband_offset = 0usize;
	let mut update_lowband = true;
	let mut ctx = BandCtx {
		dec,
		band: 0,
		intensity,
		spread,
		tf_change: 0,
		remaining_bits: 0,
		seed: *seed,
		disable_inv,
		avoid_split_noise: bb > 1,
	};
	for i in start..end {
		ctx.band = i;
		let last = i == end - 1;
		let band_start = m * EBANDS[i] as usize;
		let n = m * (EBANDS[i + 1] - EBANDS[i]) as usize;
		let tell = ctx.dec.tell_frac() as i32;
		if i != start {
			balance -= tell;
		}
		let remaining_bits = total_bits - tell - 1;
		ctx.remaining_bits = remaining_bits;
		let b = if i < coded_bands {
			let curr_balance = balance / (coded_bands as i32 - i as i32).min(3);
			(remaining_bits + 1).min(pulses[i] + curr_balance).clamp(0, 16383)
		} else {
			0
		};
		if (band_start as i32 - n as i32 >= (m * EBANDS[start] as usize) as i32 || i == start + 1)
			&& (update_lowband || lowband_offset == 0)
		{
			lowband_offset = i;
		}
		if i == start + 1 {
			// special_hybrid_folding
			let n1 = m * (EBANDS[start + 1] - EBANDS[start]) as usize;
			let n2 = m * (EBANDS[start + 2] - EBANDS[start + 1]) as usize;
			if n2 > n1 {
				let src = 2 * n1 - n2;
				norm.copy_within(src..src + (n2 - n1), n1);
				if dual_stereo {
					norm2.copy_within(src..src + (n2 - n1), n1);
				}
			}
		}
		ctx.tf_change = tf_res[i];
		let mut effective_lowband: Option<usize> = None;
		let (mut x_cm, mut y_cm);
		if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || bb > 1 || tf_res[i] < 0) {
			let eff =
				(m as i32 * EBANDS[lowband_offset] as i32 - norm_offset as i32 - n as i32).max(0) as usize;
			effective_lowband = Some(eff);
			let mut fold_start = lowband_offset;
			loop {
				fold_start -= 1;
				if m * EBANDS[fold_start] as usize <= eff + norm_offset {
					break;
				}
			}
			let mut fold_end = lowband_offset - 1;
			loop {
				fold_end += 1;
				if !(fold_end < i && m * (EBANDS[fold_end] as usize) < eff + norm_offset + n) {
					break;
				}
			}
			x_cm = 0u32;
			y_cm = 0u32;
			let mut fold_i = fold_start;
			loop {
				x_cm |= u32::from(collapse_masks[fold_i * c]);
				y_cm |= u32::from(collapse_masks[fold_i * c + c - 1]);
				fold_i += 1;
				if fold_i >= fold_end {
					break;
				}
			}
		} else {
			x_cm = ((1u64 << bb) - 1) as u32;
			y_cm = x_cm;
		}
		if dual_stereo && i == intensity {
			dual_stereo = false;
			let upto = band_start - norm_offset;
			for j in 0..upto {
				norm[j] = 0.5 * (norm[j] + norm2[j]);
			}
		}
		let out_at = band_start - norm_offset;
		let xb = &mut x[band_start..band_start + n];
		if dual_stereo {
			let low: Option<Vec<f32>> =
				effective_lowband.map(|e| norm[e..(e + n).min(norm.len())].to_vec());
			let mut out = vec![0f32; n];
			x_cm = quant_band(
				&mut ctx,
				xb,
				n,
				b / 2,
				bb,
				low.as_deref(),
				lm as i32,
				(!last).then_some(&mut out[..]),
				1.0,
				x_cm,
			);
			if !last {
				norm[out_at..out_at + n].copy_from_slice(&out);
			}
			let low2: Option<Vec<f32>> =
				effective_lowband.map(|e| norm2[e..(e + n).min(norm2.len())].to_vec());
			let yb = &mut y.as_deref_mut().unwrap_or(&mut [])[band_start..band_start + n];
			y_cm = quant_band(
				&mut ctx,
				yb,
				n,
				b / 2,
				bb,
				low2.as_deref(),
				lm as i32,
				(!last).then_some(&mut out[..]),
				1.0,
				y_cm,
			);
			if !last {
				norm2[out_at..out_at + n].copy_from_slice(&out);
			}
		} else {
			let low: Option<Vec<f32>> =
				effective_lowband.map(|e| norm[e..(e + n).min(norm.len())].to_vec());
			let mut out = vec![0f32; n];
			let out_ref = (!last).then_some(&mut out[..]);
			x_cm = match y.as_deref_mut() {
				Some(yall) => {
					let yb = &mut yall[band_start..band_start + n];
					quant_band_stereo(
						&mut ctx,
						xb,
						yb,
						n,
						b,
						bb,
						low.as_deref(),
						lm as i32,
						out_ref,
						x_cm | y_cm,
					)
				}
				None => {
					quant_band(&mut ctx, xb, n, b, bb, low.as_deref(), lm as i32, out_ref, 1.0, x_cm | y_cm)
				}
			};
			if !last {
				norm[out_at..out_at + n].copy_from_slice(&out);
			}
			y_cm = x_cm;
		}
		collapse_masks[i * c] = x_cm as u8;
		collapse_masks[i * c + c - 1] = y_cm as u8;
		balance += pulses[i] + tell;
		update_lowband = b > (n as i32) << BITRES;
		ctx.avoid_split_noise = false;
	}
	*seed = ctx.seed;
}

fn anti_collapse(
	x: &mut [f32],
	collapse_masks: &[u8],
	lm: usize,
	c: usize,
	size: usize,
	start: usize,
	end: usize,
	log_e: &[f32; 2 * NB_EBANDS],
	prev1: &[f32; 2 * NB_EBANDS],
	prev2: &[f32; 2 * NB_EBANDS],
	pulses: &[i32; NB_EBANDS],
	mut seed: u32,
) {
	for i in start..end {
		let n0 = (EBANDS[i + 1] - EBANDS[i]) as usize;
		let depth = ((1 + pulses[i]) / n0 as i32) >> lm;
		let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
		let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();
		for ch in 0..c {
			let mut p1 = prev1[ch * NB_EBANDS + i];
			let mut p2 = prev2[ch * NB_EBANDS + i];
			if c == 1 {
				p1 = p1.max(prev1[NB_EBANDS + i]);
				p2 = p2.max(prev2[NB_EBANDS + i]);
			}
			let ediff = (log_e[ch * NB_EBANDS + i] - p1.min(p2)).max(0.0);
			let mut r = 2.0 * celt_exp2(-ediff);
			if lm == 3 {
				r *= std::f32::consts::SQRT_2;
			}
			r = thresh.min(r) * sqrt_1;
			let base = ch * size + ((EBANDS[i] as usize) << lm);
			let mut renormalize = false;
			for k in 0..1usize << lm {
				if collapse_masks[i * c + ch] & (1 << k) == 0 {
					for j in 0..n0 {
						seed = lcg_rand(seed);
						x[base + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
					}
					renormalize = true;
				}
			}
			if renormalize {
				renormalise_vector(&mut x[base..base + (n0 << lm)], 1.0);
			}
		}
	}
}

fn denormalise_bands(
	x: &[f32],
	freq: &mut [f32],
	band_log_e: &[f32],
	start: usize,
	end: usize,
	m: usize,
	silence: bool,
) {
	let n = m * SHORT_MDCT;
	let (start, end, bound) =
		if silence { (0, 0, 0) } else { (start, end, m * EBANDS[end] as usize) };
	freq[..m * EBANDS[start] as usize].fill(0.0);
	for i in start..end {
		let lg = band_log_e[i] + E_MEANS[i];
		let g = celt_exp2(lg.min(32.0));
		for j in m * EBANDS[i] as usize..m * EBANDS[i + 1] as usize {
			freq[j] = x[j] * g;
		}
	}
	freq[bound..n].fill(0.0);
}

/// Pitch post-filter (`comb_filter`), applied in place over `buf[at..at + n]`.
/// Because the reference runs it with `y == x`, samples written earlier in
/// the call feed later ones: the filter is recursive (IIR), reading at
/// least `COMBFILTER_MINPERIOD - 2` samples back.
fn comb_filter(
	buf: &mut [f32],
	at: usize,
	t0: i32,
	t1: i32,
	n: usize,
	g0: f32,
	g1: f32,
	tapset0: usize,
	tapset1: usize,
	overlap: usize,
) {
	if g0 == 0.0 && g1 == 0.0 {
		return;
	}
	let t0 = t0.max(COMBFILTER_MINPERIOD) as usize;
	let t1 = t1.max(COMBFILTER_MINPERIOD) as usize;
	let g00 = g0 * COMB_GAINS[tapset0][0];
	let g01 = g0 * COMB_GAINS[tapset0][1];
	let g02 = g0 * COMB_GAINS[tapset0][2];
	let g10 = g1 * COMB_GAINS[tapset1][0];
	let g11 = g1 * COMB_GAINS[tapset1][1];
	let g12 = g1 * COMB_GAINS[tapset1][2];
	let mut x1 = buf[at + 1 - t1];
	let mut x2 = buf[at - t1];
	let mut x3 = buf[at - t1 - 1];
	let mut x4 = buf[at - t1 - 2];
	let overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 { 0 } else { overlap.min(n) };
	let w = &WINDOW120;
	let mut i = 0usize;
	while i < overlap {
		let p = at + i;
		let x0 = buf[p + 2 - t1];
		let f = w[i] * w[i];
		let v = buf[p]
			+ ((1.0 - f) * g00) * buf[p - t0]
			+ ((1.0 - f) * g01) * (buf[p + 1 - t0] + buf[p - t0 - 1])
			+ ((1.0 - f) * g02) * (buf[p + 2 - t0] + buf[p - t0 - 2])
			+ (f * g10) * x2
			+ (f * g11) * (x1 + x3)
			+ (f * g12) * (x0 + x4);
		buf[p] = v;
		x4 = x3;
		x3 = x2;
		x2 = x1;
		x1 = x0;
		i += 1;
	}
	if g1 == 0.0 {
		return;
	}
	// The constant-filter part (`comb_filter_const`).
	let p = at + i;
	let mut x4 = buf[p - t1 - 2];
	let mut x3 = buf[p - t1 - 1];
	let mut x2 = buf[p - t1];
	let mut x1 = buf[p + 1 - t1];
	while i < n {
		let p = at + i;
		let x0 = buf[p + 2 - t1];
		buf[p] = buf[p] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
		x4 = x3;
		x3 = x2;
		x2 = x1;
		x1 = x0;
		i += 1;
	}
}

/// `comb_filter` with distinct input and output and a constant filter
/// (`comb_filter_const`): returns `n` samples of
/// `buf[at + i] + g * (taps around buf[at + i - t])`.
fn comb_filter_fir(buf: &[f32], at: usize, t: i32, n: usize, g: f32, tapset: usize) -> Vec<f32> {
	if g == 0.0 {
		return buf[at..at + n].to_vec();
	}
	let t = t.max(COMBFILTER_MINPERIOD) as usize;
	let g10 = g * COMB_GAINS[tapset][0];
	let g11 = g * COMB_GAINS[tapset][1];
	let g12 = g * COMB_GAINS[tapset][2];
	(0..n)
		.map(|i| {
			let p = at + i;
			buf[p]
				+ g10 * buf[p - t]
				+ g11 * (buf[p + 1 - t] + buf[p - t - 1])
				+ g12 * (buf[p + 2 - t] + buf[p - t - 2])
		})
		.collect()
}

fn tf_decode(
	start: usize,
	end: usize,
	is_transient: bool,
	tf_res: &mut [i32; NB_EBANDS],
	lm: usize,
	dec: &mut RangeDecoder,
) {
	let mut budget = (dec.storage() * 8) as u32;
	let mut tell = dec.tell() as u32;
	let mut logp = if is_transient { 2 } else { 4 };
	let tf_select_rsv = lm > 0 && tell + logp < budget;
	budget -= u32::from(tf_select_rsv);
	let mut tf_changed = 0;
	let mut curr = 0;
	for slot in tf_res.iter_mut().take(end).skip(start) {
		if tell + logp <= budget {
			curr ^= i32::from(dec.bit_logp(logp));
			tell = dec.tell() as u32;
			tf_changed |= curr;
		}
		*slot = curr;
		logp = if is_transient { 4 } else { 5 };
	}
	let t = usize::from(is_transient) * 4;
	let mut tf_select = 0;
	if tf_select_rsv
		&& TF_SELECT_TABLE[lm][t + tf_changed as usize]
			!= TF_SELECT_TABLE[lm][t + 2 + tf_changed as usize]
	{
		tf_select = usize::from(dec.bit_logp(1));
	}
	for slot in tf_res.iter_mut().take(end).skip(start) {
		*slot = i32::from(TF_SELECT_TABLE[lm][t + 2 * tf_select + *slot as usize]);
	}
}

/// CELT decoder state (`OpusCustomDecoder`).
#[derive(Debug, Clone)]
pub struct CeltDecoder {
	/// Output channels (1 or 2).
	pub channels: usize,
	/// Channels coded in the current stream (may differ from `channels`).
	pub stream_channels: usize,
	pub start: usize,
	pub end: usize,
	disable_inv: bool,
	pub rng: u32,
	/// Pitch period found at the start of the current loss.
	last_pitch_index: usize,
	/// Concealed duration since the last good frame, in 2.5 ms units.
	loss_duration: i32,
	/// Set until two good frames follow a loss or reset: the pitch-based
	/// PLC needs a clean history.
	skip_plc: bool,
	/// Per-channel LPC model of the signal before the loss.
	lpc: [[f32; LPC_ORDER]; 2],
	postfilter_period: i32,
	postfilter_period_old: i32,
	postfilter_gain: f32,
	postfilter_gain_old: f32,
	postfilter_tapset: usize,
	postfilter_tapset_old: usize,
	preemph_mem: [f32; 2],
	decode_mem: Vec<f32>,
	old_band_e: [f32; 2 * NB_EBANDS],
	old_log_e: [f32; 2 * NB_EBANDS],
	old_log_e2: [f32; 2 * NB_EBANDS],
	background_log_e: [f32; 2 * NB_EBANDS],
}

impl CeltDecoder {
	pub fn new(channels: usize) -> Self {
		let mut d = Self {
			channels,
			stream_channels: channels,
			start: 0,
			end: NB_EBANDS,
			disable_inv: channels == 1,
			rng: 0,
			last_pitch_index: 0,
			loss_duration: 0,
			skip_plc: true,
			lpc: [[0.0; LPC_ORDER]; 2],
			postfilter_period: 0,
			postfilter_period_old: 0,
			postfilter_gain: 0.0,
			postfilter_gain_old: 0.0,
			postfilter_tapset: 0,
			postfilter_tapset_old: 0,
			preemph_mem: [0.0; 2],
			decode_mem: vec![0.0; channels * (DECODE_BUFFER_SIZE + OVERLAP)],
			old_band_e: [0.0; 2 * NB_EBANDS],
			old_log_e: [-28.0; 2 * NB_EBANDS],
			old_log_e2: [-28.0; 2 * NB_EBANDS],
			background_log_e: [0.0; 2 * NB_EBANDS],
		};
		d.reset();
		d
	}

	/// `OPUS_RESET_STATE`.
	pub fn reset(&mut self) {
		self.rng = 0;
		self.last_pitch_index = 0;
		self.loss_duration = 0;
		self.skip_plc = true;
		self.lpc = [[0.0; LPC_ORDER]; 2];
		self.postfilter_period = 0;
		self.postfilter_period_old = 0;
		self.postfilter_gain = 0.0;
		self.postfilter_gain_old = 0.0;
		self.postfilter_tapset = 0;
		self.postfilter_tapset_old = 0;
		self.preemph_mem = [0.0; 2];
		self.decode_mem.fill(0.0);
		self.old_band_e = [0.0; 2 * NB_EBANDS];
		self.old_log_e = [-28.0; 2 * NB_EBANDS];
		self.old_log_e2 = [-28.0; 2 * NB_EBANDS];
		self.background_log_e = [0.0; 2 * NB_EBANDS];
	}

	fn channel_base(&self, c: usize) -> usize {
		c * (DECODE_BUFFER_SIZE + OVERLAP)
	}

	/// Denormalises `x` and runs the inverse MDCTs into the synthesis
	/// buffers (`celt_synthesis`).
	fn synthesis(
		&mut self,
		x: &[f32],
		start: usize,
		eff_end: usize,
		c: usize,
		is_transient: bool,
		lm: usize,
		silence: bool,
	) {
		let cc = self.channels;
		let n = SHORT_MDCT << lm;
		let m = 1 << lm;
		let (b_count, nb, shift) =
			if is_transient { (m, SHORT_MDCT, MAX_LM) } else { (1, SHORT_MDCT << lm, MAX_LM - lm) };
		let mdct = Mdct::celt();
		let mut freq = vec![0f32; n];
		let run = |dec: &mut Self, ch: usize, freq: &[f32]| {
			let base = dec.channel_base(ch) + DECODE_BUFFER_SIZE - n;
			for b in 0..b_count {
				let out = &mut dec.decode_mem[base + nb * b..];
				mdct.backward(&freq[b..], out, &WINDOW120, OVERLAP, shift, b_count);
			}
		};
		if cc == 2 && c == 1 {
			denormalise_bands(x, &mut freq, &self.old_band_e[..NB_EBANDS], start, eff_end, m, silence);
			let copy = freq.clone();
			run(self, 0, &copy);
			run(self, 1, &copy);
		} else if cc == 1 && c == 2 {
			let mut freq2 = vec![0f32; n];
			denormalise_bands(
				&x[..n],
				&mut freq,
				&self.old_band_e[..NB_EBANDS],
				start,
				eff_end,
				m,
				silence,
			);
			denormalise_bands(
				&x[n..],
				&mut freq2,
				&self.old_band_e[NB_EBANDS..],
				start,
				eff_end,
				m,
				silence,
			);
			for i in 0..n {
				freq[i] = 0.5 * freq[i] + 0.5 * freq2[i];
			}
			run(self, 0, &freq);
		} else {
			for ch in 0..cc {
				denormalise_bands(
					&x[ch * n..(ch + 1) * n],
					&mut freq,
					&self.old_band_e[ch * NB_EBANDS..(ch + 1) * NB_EBANDS],
					start,
					eff_end,
					m,
					silence,
				);
				run(self, ch, &freq);
			}
		}
	}

	fn deemphasis(&mut self, n: usize, pcm: &mut [f32]) {
		let cc = self.channels;
		for ch in 0..cc {
			let base = self.channel_base(ch) + DECODE_BUFFER_SIZE - n;
			let mut mem = self.preemph_mem[ch];
			for j in 0..n {
				let tmp = self.decode_mem[base + j] + VERY_SMALL + mem;
				mem = PREEMPH * tmp;
				pcm[j * cc + ch] = tmp * (1.0 / 32768.0);
			}
			self.preemph_mem[ch] = mem;
		}
	}

	fn shift_buffers(&mut self, n: usize) {
		for ch in 0..self.channels {
			let base = self.channel_base(ch);
			self.decode_mem.copy_within(base + n..base + DECODE_BUFFER_SIZE + OVERLAP / 2, base);
		}
	}

	/// Conceals a missing frame of `120 << lm` samples (`celt_decode_lost`)
	/// and writes it to `pcm` (interleaved, unit scale).
	pub fn decode_lost(&mut self, lm: usize, pcm: &mut [f32]) {
		let n = SHORT_MDCT << lm;
		let noise_based = self.loss_duration >= 40 || self.start != 0 || self.skip_plc;
		if noise_based {
			self.conceal_with_noise(lm);
		} else {
			self.conceal_with_pitch(n);
		}
		self.deemphasis(n, pcm);
		// Saturate to avoid wrap-around.
		self.loss_duration = (self.loss_duration + (1 << lm)).min(10_000);
	}

	/// Noise-based PLC / comfort noise: random spectra at the last band
	/// energies, decaying towards the background noise estimate.
	fn conceal_with_noise(&mut self, lm: usize) {
		let n = SHORT_MDCT << lm;
		let c = self.channels;
		let start = self.start;
		let end = self.end;
		let eff_end = start.max(end.min(NB_EBANDS));
		self.shift_buffers(n);
		let decay = if self.loss_duration == 0 { 1.5 } else { 0.5 };
		for ch in 0..c {
			for i in start..end {
				let k = ch * NB_EBANDS + i;
				self.old_band_e[k] = self.background_log_e[k].max(self.old_band_e[k] - decay);
			}
		}
		let mut seed = self.rng;
		let mut x = vec![0f32; c * n];
		for ch in 0..c {
			for i in start..eff_end {
				let boffs = n * ch + ((EBANDS[i] as usize) << lm);
				let blen = ((EBANDS[i + 1] - EBANDS[i]) as usize) << lm;
				for j in 0..blen {
					seed = lcg_rand(seed);
					x[boffs + j] = ((seed as i32) >> 20) as f32;
				}
				renormalise_vector(&mut x[boffs..boffs + blen], 1.0);
			}
		}
		self.rng = seed;
		let saved = self.stream_channels;
		self.stream_channels = c;
		self.synthesis(&x, start, eff_end, c, false, lm, false);
		self.stream_channels = saved;
	}

	/// `celt_plc_pitch_search`: the pitch period of the decoded history,
	/// between `PLC_PITCH_LAG_MIN` and `PLC_PITCH_LAG_MAX`.
	fn plc_pitch_search(&self) -> usize {
		let channels: Vec<&[f32]> = (0..self.channels)
			.map(|ch| {
				let base = self.channel_base(ch);
				&self.decode_mem[base..base + DECODE_BUFFER_SIZE]
			})
			.collect();
		let lp = pitch::downsample(&channels, DECODE_BUFFER_SIZE);
		let lag = pitch::search(
			&lp[PLC_PITCH_LAG_MAX >> 1..],
			&lp,
			DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
			PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
		);
		PLC_PITCH_LAG_MAX.saturating_sub(lag).clamp(PLC_PITCH_LAG_MIN, PLC_PITCH_LAG_MAX)
	}

	/// Pitch-based PLC: repeats the last pitch period of the LPC
	/// excitation, decaying as the signal did, and re-synthesises it.
	#[allow(clippy::too_many_lines)]
	fn conceal_with_pitch(&mut self, n: usize) {
		let first_loss = self.loss_duration == 0;
		let (pitch_index, fade) = if first_loss {
			self.last_pitch_index = self.plc_pitch_search();
			(self.last_pitch_index, 1.0f32)
		} else {
			(self.last_pitch_index, 0.8)
		};
		// Always in range once set; the clamp guards a never-searched state.
		let pitch_index = pitch_index.clamp(PLC_PITCH_LAG_MIN, PLC_PITCH_LAG_MAX);
		// Two pitch periods of excitation, to see whether it decays, but no
		// more than MAX_PERIOD.
		let exc_length = (2 * pitch_index).min(MAX_PERIOD);
		let extrapolation_offset = MAX_PERIOD - pitch_index;
		// Enough to cover a complete MDCT window, overlap included.
		let extrapolation_len = n + OVERLAP;
		let window = &WINDOW120[..OVERLAP];
		// exc[i] is `exc_buf[LPC_ORDER + i]`, with LPC_ORDER samples of history.
		let mut exc_buf = vec![0f32; MAX_PERIOD + LPC_ORDER];
		let mut fir_tmp = vec![0f32; exc_length];
		for ch in 0..self.channels {
			let base = self.channel_base(ch);
			let buf_end = base + DECODE_BUFFER_SIZE;
			exc_buf.copy_from_slice(&self.decode_mem[buf_end - MAX_PERIOD - LPC_ORDER..buf_end]);
			if first_loss {
				// LPC model of the last MAX_PERIOD samples before the loss.
				let mut ac = pitch::autocorr(&exc_buf[LPC_ORDER..], window, LPC_ORDER);
				// Noise floor at -40 dB, then lag windowing.
				ac[0] *= 1.0001;
				for i in 1..=LPC_ORDER {
					ac[i] -= ac[i] * (0.008 * 0.008) * i as f32 * i as f32;
				}
				let lpc = pitch::lpc(&ac, LPC_ORDER);
				self.lpc[ch].copy_from_slice(&lpc);
			}
			let lpc = self.lpc[ch];
			// The excitation for the exc_length samples before the loss.
			let at = LPC_ORDER + MAX_PERIOD - exc_length;
			pitch::fir(&exc_buf, at, &lpc, &mut fir_tmp);
			exc_buf[at..at + exc_length].copy_from_slice(&fir_tmp);
			let exc = &exc_buf[LPC_ORDER..];

			// How fast the waveform decays, so concealment adds no energy.
			let decay = {
				let mut e1 = 1f32;
				let mut e2 = 1f32;
				let decay_length = exc_length >> 1;
				for i in 0..decay_length {
					let e = exc[MAX_PERIOD - decay_length + i];
					e1 += e * e;
					let e = exc[MAX_PERIOD - 2 * decay_length + i];
					e2 += e * e;
				}
				(e1.min(e2) / e2).sqrt()
			};

			// Make room for the new frame (the overlap past the buffer end is
			// not needed).
			self.decode_mem.copy_within(base + n..buf_end, base);
			let buf = &mut self.decode_mem[base..buf_end + OVERLAP];
			let out_at = DECODE_BUFFER_SIZE - n;

			// Repeat the last pitch period, one more decay per period.
			let mut attenuation = fade * decay;
			let mut s1 = 0f32;
			let mut j = 0usize;
			for i in 0..extrapolation_len {
				if j >= pitch_index {
					j -= pitch_index;
					attenuation *= decay;
				}
				buf[out_at + i] = attenuation * exc[extrapolation_offset + j];
				// Energy of the decoded signal whose excitation is copied.
				let tmp = buf[DECODE_BUFFER_SIZE - MAX_PERIOD - n + extrapolation_offset + j];
				s1 += tmp * tmp;
				j += 1;
			}

			// Back to the signal domain, continuing from the last samples.
			let mut lpc_mem = [0f32; LPC_ORDER];
			for (i, m) in lpc_mem.iter_mut().enumerate() {
				*m = buf[out_at - 1 - i];
			}
			pitch::iir(&mut buf[out_at..out_at + extrapolation_len], &lpc, &mut lpc_mem);

			// Attenuate if the synthesis came out louder than expected, which
			// the signal changing within the window can cause.
			let s2: f32 = buf[out_at..out_at + extrapolation_len].iter().map(|v| v * v).sum();
			// Written so that a NaN from the filter also mutes.
			if s1.partial_cmp(&(0.2 * s2)) != Some(std::cmp::Ordering::Greater) {
				buf[out_at..out_at + extrapolation_len].fill(0.0);
			} else if s1 < s2 {
				let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
				for i in 0..OVERLAP {
					buf[out_at + i] *= 1.0 - window[i] * (1.0 - ratio);
				}
				for v in &mut buf[out_at + OVERLAP..out_at + extrapolation_len] {
					*v *= ratio;
				}
			}

			// Pre-filter the overlap for the next frame, whose post-filter
			// runs over it again, then simulate TDAC so the concealment blends
			// with the next frame's MDCT.
			let etmp = comb_filter_fir(
				buf,
				DECODE_BUFFER_SIZE,
				self.postfilter_period,
				OVERLAP,
				-self.postfilter_gain,
				self.postfilter_tapset,
			);
			for i in 0..OVERLAP / 2 {
				buf[DECODE_BUFFER_SIZE + i] =
					window[i] * etmp[OVERLAP - 1 - i] + window[OVERLAP - i - 1] * etmp[i];
			}
		}
	}

	/// Decodes one CELT frame of `frame_size` samples (120 << LM) per
	/// channel from `data`. Output is interleaved float at unit scale.
	pub fn decode(
		&mut self,
		data: &[u8],
		pcm: &mut [f32],
		frame_size: usize,
	) -> Result<usize, &'static str> {
		let lm =
			(0..=MAX_LM).find(|&l| SHORT_MDCT << l == frame_size).ok_or("invalid celt frame size")?;
		if data.len() <= 1 {
			self.decode_lost(lm, pcm);
			return Ok(frame_size);
		}
		let mut dec = RangeDecoder::new(data);
		self.decode_with(&mut dec, data.len(), pcm, frame_size)
	}

	/// Decodes a frame whose first `len` bytes are being read by `dec` (in
	/// hybrid mode SILK has already used the start of the same range coder).
	#[allow(clippy::too_many_lines)]
	pub fn decode_with(
		&mut self,
		dec: &mut RangeDecoder,
		len: usize,
		pcm: &mut [f32],
		frame_size: usize,
	) -> Result<usize, &'static str> {
		let lm =
			(0..=MAX_LM).find(|&l| SHORT_MDCT << l == frame_size).ok_or("invalid celt frame size")?;
		let m = 1usize << lm;
		let n = m * SHORT_MDCT;
		if len > 1275 {
			return Err("celt frame too large");
		}
		if len <= 1 {
			self.decode_lost(lm, pcm);
			return Ok(frame_size);
		}
		// The pitch-based PLC needs two good frames in a row.
		self.skip_plc = self.loss_duration != 0;
		let cc = self.channels;
		let c = self.stream_channels;
		let start = self.start;
		let end = self.end;
		let eff_end = end.min(NB_EBANDS);

		if c == 1 {
			for i in 0..NB_EBANDS {
				self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[NB_EBANDS + i]);
			}
		}

		let mut total_bits = (len * 8) as i32;
		let mut tell = dec.tell();
		let silence = if tell >= total_bits {
			true
		} else if tell == 1 {
			dec.bit_logp(15)
		} else {
			false
		};
		if silence {
			// Pretend every remaining bit was read.
			tell = (len * 8) as i32;
			dec.add_bits(tell - dec.tell());
		}

		let mut postfilter_gain = 0f32;
		let mut postfilter_pitch = 0i32;
		let mut postfilter_tapset = 0usize;
		if start == 0 && tell + 16 <= total_bits {
			if dec.bit_logp(1) {
				let octave = dec.uint(6);
				postfilter_pitch = ((16 << octave) + dec.bits(4 + octave)) as i32 - 1;
				let qg = dec.bits(3);
				if dec.tell() + 2 <= total_bits {
					postfilter_tapset = dec.icdf(&TAPSET_ICDF, 2);
				}
				postfilter_gain = 0.09375 * (qg + 1) as f32;
			}
			tell = dec.tell();
		}
		let is_transient = if lm > 0 && tell + 3 <= total_bits {
			let t = dec.bit_logp(3);
			tell = dec.tell();
			t
		} else {
			false
		};
		let short_blocks = is_transient;
		let intra = tell + 3 <= total_bits && dec.bit_logp(3);
		unquant_coarse_energy(&mut self.old_band_e, start, end, intra, dec, c, lm);

		let mut tf_res = [0i32; NB_EBANDS];
		tf_decode(start, end, is_transient, &mut tf_res, lm, dec);

		tell = dec.tell();
		let spread =
			if tell + 4 <= total_bits { dec.icdf(&SPREAD_ICDF, 5) as i32 } else { SPREAD_NORMAL };

		let cap = init_caps(lm, c);
		let mut offsets = [0i32; NB_EBANDS];
		let mut dynalloc_logp = 6;
		total_bits <<= BITRES;
		let mut tell_frac = dec.tell_frac() as i32;
		for i in start..end {
			let width = ((c as i32) * (eband(i + 1) - eband(i))) << lm;
			let quanta = (width << BITRES).min((6 << BITRES).max(width));
			let mut loop_logp = dynalloc_logp;
			let mut boost = 0;
			while tell_frac + (loop_logp << BITRES) < total_bits && boost < cap[i] {
				let flag = dec.bit_logp(loop_logp as u32);
				tell_frac = dec.tell_frac() as i32;
				if !flag {
					break;
				}
				boost += quanta;
				total_bits -= quanta;
				loop_logp = 1;
			}
			offsets[i] = boost;
			if boost > 0 {
				dynalloc_logp = (dynalloc_logp - 1).max(2);
			}
		}
		let alloc_trim =
			if tell_frac + (6 << BITRES) <= total_bits { dec.icdf(&TRIM_ICDF, 7) as i32 } else { 5 };

		let mut bits = (((len * 8) as i32) << BITRES) - dec.tell_frac() as i32 - 1;
		let anti_collapse_rsv =
			if is_transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) { 1 << BITRES } else { 0 };
		bits -= anti_collapse_rsv;
		let alloc = compute_allocation(start, end, &offsets, &cap, alloc_trim, bits, c, lm, dec);
		unquant_fine_energy(&mut self.old_band_e, start, end, &alloc.fine_quant, dec, c);

		self.shift_buffers(n);

		let mut x = vec![0f32; c * n];
		let mut collapse_masks = vec![0u8; c * NB_EBANDS];
		{
			let (xs, ys) = x.split_at_mut(n);
			let y = if c == 2 { Some(ys) } else { None };
			let mut rng = self.rng;
			quant_all_bands(
				dec,
				start,
				end,
				xs,
				y,
				&mut collapse_masks,
				&alloc.pulses,
				short_blocks,
				spread,
				alloc.dual_stereo,
				alloc.intensity,
				&tf_res,
				((len * 8) as i32) * (1 << BITRES) - anti_collapse_rsv,
				alloc.balance,
				lm,
				alloc.coded_bands,
				&mut rng,
				self.disable_inv,
			);
			self.rng = rng;
		}
		let anti_collapse_on = anti_collapse_rsv > 0 && dec.bits(1) != 0;
		let bits_left = (len * 8) as i32 - dec.tell();
		unquant_energy_finalise(
			&mut self.old_band_e,
			start,
			end,
			&alloc.fine_quant,
			&alloc.fine_priority,
			bits_left,
			dec,
			c,
		);
		if anti_collapse_on {
			anti_collapse(
				&mut x,
				&collapse_masks,
				lm,
				c,
				n,
				start,
				end,
				&self.old_band_e,
				&self.old_log_e,
				&self.old_log_e2,
				&alloc.pulses,
				self.rng,
			);
		}
		if silence {
			self.old_band_e = [-28.0; 2 * NB_EBANDS];
		}
		self.synthesis(&x, start, eff_end, c, is_transient, lm, silence);

		for ch in 0..cc {
			self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
			self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
			let at = self.channel_base(ch) + DECODE_BUFFER_SIZE - n;
			comb_filter(
				&mut self.decode_mem,
				at,
				self.postfilter_period_old,
				self.postfilter_period,
				SHORT_MDCT,
				self.postfilter_gain_old,
				self.postfilter_gain,
				self.postfilter_tapset_old,
				self.postfilter_tapset,
				OVERLAP,
			);
			if lm != 0 {
				comb_filter(
					&mut self.decode_mem,
					at + SHORT_MDCT,
					self.postfilter_period,
					postfilter_pitch,
					n - SHORT_MDCT,
					self.postfilter_gain,
					postfilter_gain,
					self.postfilter_tapset,
					postfilter_tapset,
					OVERLAP,
				);
			}
		}
		self.postfilter_period_old = self.postfilter_period;
		self.postfilter_gain_old = self.postfilter_gain;
		self.postfilter_tapset_old = self.postfilter_tapset;
		self.postfilter_period = postfilter_pitch;
		self.postfilter_gain = postfilter_gain;
		self.postfilter_tapset = postfilter_tapset;
		if lm != 0 {
			self.postfilter_period_old = self.postfilter_period;
			self.postfilter_gain_old = self.postfilter_gain;
			self.postfilter_tapset_old = self.postfilter_tapset;
		}

		if c == 1 {
			let (a, b) = self.old_band_e.split_at_mut(NB_EBANDS);
			b.copy_from_slice(a);
		}
		if !is_transient {
			self.old_log_e2 = self.old_log_e;
			self.old_log_e = self.old_band_e;
		} else {
			for i in 0..2 * NB_EBANDS {
				self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
			}
		}
		let max_background_increase = (self.loss_duration + m as i32).min(160) as f32 * 0.001;
		for i in 0..2 * NB_EBANDS {
			self.background_log_e[i] =
				(self.background_log_e[i] + max_background_increase).min(self.old_band_e[i]);
		}
		for ch in 0..2 {
			for i in (0..start).chain(end..NB_EBANDS) {
				self.old_band_e[ch * NB_EBANDS + i] = 0.0;
				self.old_log_e[ch * NB_EBANDS + i] = -28.0;
				self.old_log_e2[ch * NB_EBANDS + i] = -28.0;
			}
		}
		self.rng = dec.range();
		self.deemphasis(n, pcm);
		self.loss_duration = 0;
		Ok(frame_size)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn pvq_counts() {
		// V(N, K): number of integer vectors of dimension N with L1 norm K.
		assert_eq!(pvq_v(2, 1), 4);
		assert_eq!(pvq_v(2, 2), 8);
		assert_eq!(pvq_v(3, 1), 6);
		assert_eq!(pvq_v(3, 2), 18);
		assert_eq!(pvq_v(4, 3), 88);
		// Decoding every index of V(3, 2) gives distinct vectors of norm 2.
		let mut seen = std::collections::HashSet::new();
		for i in 0..pvq_v(3, 2) {
			let mut y = [0i32; 3];
			let yy = cwrsi(3, 2, i, &mut y);
			assert_eq!(y.iter().map(|v| v.abs()).sum::<i32>(), 2);
			assert_eq!(yy, y.iter().map(|v| (v * v) as f32).sum::<f32>());
			assert!(seen.insert(y));
		}
	}

	#[test]
	fn bitexact_trig() {
		assert!(bitexact_cos(16383) > 0 && bitexact_cos(16383) < 64);
		assert!((bitexact_cos(8192) - 23170).abs() < 4);
		assert_eq!(bitexact_log2tan(23170, 23170), 0);
	}
}
