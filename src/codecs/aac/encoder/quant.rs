//! Quantisation, Huffman bit counting, sectioning and scalefactor selection
//! (ISO/IEC 14496-3 §4.6.1-4.6.3, encoder side).
//!
//! FFmpeg: `aaccoder.c` (`quantize_and_encode_band_cost`,
//! `encode_window_bands_info`), `aaccoder_twoloop.h`, `aacenc_quantization.h`.
//!
//! The quantiser is the specification's
//! `q = int(|x|^(3/4) · 2^(-3(sf-100)/16) + 0.4054)`; the decoder inverts
//! it as `|q|^(4/3) · 2^((sf-100)/4)`.
//!
//! Every bit count here is exact (the codeword lengths of the normative
//! tables), so the rate loop's estimate equals what the writer emits.

use crate::codecs::aac::ics::tables as decoder_tables;
use crate::codecs::aac::tables::{SCALEFACTOR_CODES, SPECTRUM_CODES};
use crate::codecs::bitstream::BitWriter;
use std::sync::OnceLock;

/// Rounding offset of the quantiser (3GPP TS 26.403 / FFmpeg `ROUND_STANDARD`).
pub const MAGIC: f32 = 0.4054;
/// Scalefactor that gives a quantiser step of 1.
pub const SF_OFFSET: i32 = 100;
/// Largest quantised magnitude (13-bit escape).
pub const MAX_Q: i32 = 8191;
/// Largest scalefactor difference the scalefactor codebook can code.
pub const MAX_SF_DIFF: i32 = 60;
const SF_DIFF_ZERO: usize = 60;

/// Largest absolute value (LAV) per codebook, index 0 = zero band.
pub const LAV: [i32; 12] = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16];

/// Step tables per scalefactor 0..=255.
struct Steps {
	/// `2^(-3(sf-100)/16)`, applied to `|x|^(3/4)`.
	inv: [f32; 256],
	/// `2^((sf-100)/4)`, applied to `|q|^(4/3)`.
	deq: [f32; 256],
}

fn steps() -> &'static Steps {
	static S: OnceLock<Steps> = OnceLock::new();
	S.get_or_init(|| {
		let mut s = Steps { inv: [0.0; 256], deq: [0.0; 256] };
		for sf in 0..256 {
			let e = f64::from(sf as i32 - SF_OFFSET);
			s.inv[sf] = 2f64.powf(-3.0 * e / 16.0) as f32;
			s.deq[sf] = 2f64.powf(e / 4.0) as f32;
		}
		s
	})
}

/// Per-band analysis that doesn't change during the rate loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct BandStats {
	/// Start and end in the channel's grouped coefficient array.
	pub start: usize,
	pub end: usize,
	pub energy: f32,
	/// Masking threshold (allowed noise energy) before the global offset.
	pub threshold: f32,
	/// `max |x|^(3/4)`.
	pub max34: f32,
	/// `Σ |x|^(1/2)`, for the noise estimate.
	pub sum_sqrt: f32,
}

impl BandStats {
	pub fn new(x: &[f32], x34: &[f32], start: usize, end: usize, threshold: f32) -> Self {
		let (mut energy, mut max34, mut sum_sqrt) = (0.0f32, 0.0f32, 0.0f32);
		for i in start..end {
			energy += x[i] * x[i];
			max34 = max34.max(x34[i]);
			sum_sqrt += x[i].abs().sqrt();
		}
		Self { start, end, energy, threshold, max34, sum_sqrt }
	}

	/// Smallest scalefactor whose quantised values stay within 8191.
	pub fn min_sf(&self) -> i32 {
		if self.max34 <= 0.0 {
			return 0;
		}
		let need = self.max34 / (MAX_Q as f32 - MAGIC);
		let sf = SF_OFFSET as f32 + 16.0 / 3.0 * need.log2();
		(sf.ceil() as i32).clamp(0, 255)
	}
}

/// `|x|^(3/4)` of every coefficient.
pub fn abs34(x: &[f32], out: &mut [f32]) {
	for (o, &v) in out.iter_mut().zip(x) {
		let a = v.abs();
		*o = (a * a.sqrt()).sqrt();
	}
}

/// Quantises `x` (with `x34 = |x|^(3/4)`) at `sf` into `q`; returns the
/// squared error after dequantisation.
pub fn quantize(x: &[f32], x34: &[f32], sf: i32, q: &mut [i32]) -> f32 {
	let s = steps();
	let sf = sf.clamp(0, 255) as usize;
	let (inv, deq) = (s.inv[sf], s.deq[sf]);
	let pow43 = &decoder_tables().pow43;
	let mut err = 0.0f32;
	for i in 0..x.len() {
		let m = ((x34[i] * inv + MAGIC) as i32).min(MAX_Q);
		let r = pow43[m as usize] * deq;
		let d = x[i].abs() - r;
		err += d * d;
		q[i] = if x[i] < 0.0 { -m } else { m };
	}
	err
}

/// Picks the coarsest scalefactor whose quantisation noise stays within
/// `allowed` (starting from the analytic estimate
/// `noise ≈ (4/27)·Δ²·Σ|x|^(1/2)` of a uniform quantiser in the `|x|^(3/4)`
/// domain, then refining against the real error). `None`: the band can be
/// left out (quantises to zero or is below the threshold).
pub fn choose_sf(
	x: &[f32],
	x34: &[f32],
	b: &BandStats,
	allowed: f32,
	q: &mut [i32],
) -> Option<i32> {
	if b.energy <= allowed || b.max34 <= 0.0 {
		return None;
	}
	let (lo, hi) = (b.start, b.end);
	let (x, x34, q) = (&x[lo..hi], &x34[lo..hi], &mut q[lo..hi]);
	let min_sf = b.min_sf();
	let estimate = SF_OFFSET as f32
		+ 8.0 / 3.0 * (27.0 * allowed / (4.0 * b.sum_sqrt.max(1e-9))).max(1e-30).log2();
	let mut sf = (estimate.floor() as i32).clamp(min_sf, 255);
	let mut err = quantize(x, x34, sf, q);
	if err > allowed {
		while sf > min_sf && err > allowed {
			sf -= 1;
			err = quantize(x, x34, sf, q);
		}
	} else {
		// Coarser while still within the allowance; on the first failure,
		// requantise at the last good step.
		for _ in 0..4 {
			if sf >= 255 {
				break;
			}
			if quantize(x, x34, sf + 1, q) > allowed {
				quantize(x, x34, sf, q);
				break;
			}
			sf += 1;
		}
	}
	if q.iter().all(|&v| v == 0) {
		return None;
	}
	Some(sf)
}

#[inline]
fn code_len(cb: u8, idx: usize) -> u32 {
	u32::from(SPECTRUM_CODES[usize::from(cb) - 1][idx].1)
}

/// Escape bits for a codebook-11 magnitude `v >= 16`: N ones, a zero, then
/// N + 4 bits.
#[inline]
fn escape_bits(v: i32) -> u32 {
	let n = (31 - (v as u32).leading_zeros()) - 4;
	2 * n + 5
}

/// Bits to code `q` with spectral codebook `cb` (1-11). The caller ensures
/// every magnitude is within the codebook's LAV.
pub fn band_bits(cb: u8, q: &[i32]) -> u32 {
	let mut bits = 0u32;
	match cb {
		1 | 2 => {
			for v in q.chunks_exact(4) {
				let idx = 27 * (v[0] + 1) + 9 * (v[1] + 1) + 3 * (v[2] + 1) + (v[3] + 1);
				bits += code_len(cb, idx as usize);
			}
		}
		3 | 4 => {
			for v in q.chunks_exact(4) {
				let a = [v[0].abs(), v[1].abs(), v[2].abs(), v[3].abs()];
				let idx = 27 * a[0] + 9 * a[1] + 3 * a[2] + a[3];
				bits += code_len(cb, idx as usize) + a.iter().filter(|&&x| x != 0).count() as u32;
			}
		}
		5 | 6 => {
			for v in q.chunks_exact(2) {
				bits += code_len(cb, (9 * (v[0] + 4) + (v[1] + 4)) as usize);
			}
		}
		_ => {
			let m = match cb {
				7 | 8 => 8,
				9 | 10 => 13,
				_ => 17,
			};
			for v in q.chunks_exact(2) {
				let (a, b) = (v[0].abs(), v[1].abs());
				let (ca, cb_) = if cb == 11 { (a.min(16), b.min(16)) } else { (a, b) };
				bits += code_len(cb, (m * ca + cb_) as usize);
				bits += u32::from(a != 0) + u32::from(b != 0);
				if cb == 11 {
					if a >= 16 {
						bits += escape_bits(a);
					}
					if b >= 16 {
						bits += escape_bits(b);
					}
				}
			}
		}
	}
	bits
}

fn put_code(w: &mut BitWriter, cb: u8, idx: usize) {
	let (code, len) = SPECTRUM_CODES[usize::from(cb) - 1][idx];
	w.put(code, u32::from(len));
}

fn put_escape(w: &mut BitWriter, v: i32) {
	let n = (31 - (v as u32).leading_zeros()) - 4;
	w.put((1u32 << n) - 1, n); // n ones
	w.put(0, 1);
	w.put(v as u32 - (1u32 << (n + 4)), n + 4);
}

/// Writes `q` with codebook `cb` (the inverse of the decoder's
/// `decode_band`).
pub fn write_band(w: &mut BitWriter, cb: u8, q: &[i32]) {
	let signs = |w: &mut BitWriter, v: &[i32]| {
		for &x in v {
			if x != 0 {
				w.bit(x < 0);
			}
		}
	};
	match cb {
		1 | 2 => {
			for v in q.chunks_exact(4) {
				let idx = 27 * (v[0] + 1) + 9 * (v[1] + 1) + 3 * (v[2] + 1) + (v[3] + 1);
				put_code(w, cb, idx as usize);
			}
		}
		3 | 4 => {
			for v in q.chunks_exact(4) {
				let a = [v[0].abs(), v[1].abs(), v[2].abs(), v[3].abs()];
				put_code(w, cb, (27 * a[0] + 9 * a[1] + 3 * a[2] + a[3]) as usize);
				signs(w, v);
			}
		}
		5 | 6 => {
			for v in q.chunks_exact(2) {
				put_code(w, cb, (9 * (v[0] + 4) + (v[1] + 4)) as usize);
			}
		}
		_ => {
			let m = match cb {
				7 | 8 => 8,
				9 | 10 => 13,
				_ => 17,
			};
			for v in q.chunks_exact(2) {
				let (a, b) = (v[0].abs(), v[1].abs());
				let (ca, cb_) = if cb == 11 { (a.min(16), b.min(16)) } else { (a, b) };
				put_code(w, cb, (m * ca + cb_) as usize);
				signs(w, v);
				if cb == 11 {
					if a >= 16 {
						put_escape(w, a);
					}
					if b >= 16 {
						put_escape(w, b);
					}
				}
			}
		}
	}
}

/// Bits of a scalefactor difference.
#[inline]
pub fn sf_diff_bits(diff: i32) -> u32 {
	u32::from(SCALEFACTOR_CODES[(diff + SF_DIFF_ZERO as i32) as usize].1)
}

pub fn write_sf_diff(w: &mut BitWriter, diff: i32) {
	let (code, len) = SCALEFACTOR_CODES[(diff + SF_DIFF_ZERO as i32) as usize];
	w.put(code, u32::from(len));
}

/// Bits of a section length `len` with `bits`-bit fields escaped at
/// `2^bits - 1` (§4.6.2.3).
pub fn section_len_bits(len: usize, bits: u32) -> u32 {
	let esc = (1usize << bits) - 1;
	(len / esc + 1) as u32 * bits
}

/// Optimal codebook per band of one window group (FFmpeg's
/// `encode_window_bands_info` trellis): `bits[b][cb]` is `u32::MAX` where
/// the codebook can't code band `b`. A new section costs its 4-bit codebook
/// plus a length field; escapes of long sections are added exactly when the
/// result is written, and are rare enough not to steer the search.
pub fn section(bits: &[[u32; 12]], len_bits: u32) -> Vec<u8> {
	let n = bits.len();
	if n == 0 {
		return Vec::new();
	}
	let start_cost = 4 + len_bits;
	let mut cost = vec![[u32::MAX; 12]; n];
	let mut from = vec![[0u8; 12]; n];
	for cb in 0..12 {
		if bits[0][cb] != u32::MAX {
			cost[0][cb] = bits[0][cb] + start_cost;
			from[0][cb] = cb as u8;
		}
	}
	for b in 1..n {
		let (best_prev, best_cb) = (0..12).map(|c| (cost[b - 1][c], c)).min().unwrap_or((u32::MAX, 0));
		for cb in 0..12 {
			if bits[b][cb] == u32::MAX {
				continue;
			}
			let stay = cost[b - 1][cb];
			let switch = best_prev.saturating_add(start_cost);
			let (c, f) = if stay <= switch { (stay, cb) } else { (switch, best_cb) };
			if c != u32::MAX {
				cost[b][cb] = c.saturating_add(bits[b][cb]);
				from[b][cb] = f as u8;
			}
		}
	}
	let mut out = vec![0u8; n];
	let mut cb = (0..12).min_by_key(|&c| cost[n - 1][c]).unwrap_or(11);
	for b in (0..n).rev() {
		out[b] = cb as u8;
		cb = usize::from(from[b][cb]);
	}
	out
}

/// Exact bits of `section_data` for codebooks `cbs` of one group.
pub fn section_bits(cbs: &[u8], len_bits: u32) -> u32 {
	let mut bits = 0;
	let mut i = 0;
	while i < cbs.len() {
		let mut j = i;
		while j < cbs.len() && cbs[j] == cbs[i] {
			j += 1;
		}
		bits += 4 + section_len_bits(j - i, len_bits);
		i = j;
	}
	bits
}

pub fn write_sections(w: &mut BitWriter, cbs: &[u8], len_bits: u32) {
	let esc = (1u32 << len_bits) - 1;
	let mut i = 0;
	while i < cbs.len() {
		let mut j = i;
		while j < cbs.len() && cbs[j] == cbs[i] {
			j += 1;
		}
		w.put(u32::from(cbs[i]), 4);
		let mut len = (j - i) as u32;
		while len >= esc {
			w.put(esc, len_bits);
			len -= esc;
		}
		w.put(len, len_bits);
		i = j;
	}
}

/// Bits of `q` for each codebook (`u32::MAX` where it doesn't fit).
/// Uncoded bands only take codebook 0.
pub fn codebook_costs(q: &[i32], coded: bool) -> [u32; 12] {
	let mut out = [u32::MAX; 12];
	if !coded {
		out[0] = 0;
		return out;
	}
	let max = q.iter().map(|v| v.abs()).max().unwrap_or(0);
	for cb in 1..12u8 {
		if LAV[usize::from(cb)] >= max || cb == 11 {
			out[usize::from(cb)] = band_bits(cb, q);
		}
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::bitstream::BitReader;

	#[test]
	fn band_bits_match_written_length() {
		let q: Vec<i32> = (0..32).map(|i| ((i * 37) % 41) - 20).collect();
		let small: Vec<i32> = (0..32).map(|i| ((i * 7) % 3) - 1).collect();
		for (vals, books) in [(&q, &[11u8][..]), (&small, &[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..])] {
			for &cb in books {
				let mut w = BitWriter::new();
				write_band(&mut w, cb, vals);
				assert_eq!(w.bits() as u32, band_bits(cb, vals), "cb {cb}");
			}
		}
	}

	#[test]
	fn escape_round_trip() {
		for v in [16, 17, 31, 32, 100, 1000, 8191] {
			let mut w = BitWriter::new();
			put_escape(&mut w, v);
			assert_eq!(w.bits() as u32, escape_bits(v));
			let data = w.finish();
			let mut r = BitReader::new(&data);
			let mut n = 0;
			while r.bit() {
				n += 1;
			}
			assert_eq!((1 << (n + 4)) + r.bits(n + 4) as i32, v);
		}
	}

	#[test]
	fn sectioning_prefers_fewer_sections() {
		// Two bands where codebook 1 is slightly cheaper for the second
		// alone but switching costs a whole section header.
		let mut bits = vec![[u32::MAX; 12]; 2];
		bits[0][5] = 20;
		bits[1][5] = 12;
		bits[1][1] = 10;
		assert_eq!(section(&bits, 5), vec![5, 5]);
	}
}
