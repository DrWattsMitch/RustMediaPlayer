//! Individual channel stream parsing and inverse quantisation
//! (ISO/IEC 14496-3 §4.4.2.7, §4.6.1-4.6.3; FFmpeg `aac/aacdec.c`
//! `decode_ics`, `decode_spectrum_and_dequant`).
//!
//! The result of [`decode_ics`] is a [`Channel`] holding dequantised
//! spectral coefficients (noise substitution already applied), band types
//! and scalefactors for the stereo and TNS tools that follow.

use std::sync::OnceLock;

use super::tables::{SCALEFACTOR_CODES, SPECTRUM_CODES, SWB_OFFSET_LONG, SWB_OFFSET_SHORT};
use crate::codecs::bitstream::{BitReader, Code, Vlc};

/// Coefficients per frame and channel.
pub const FRAME_LEN: usize = 1024;
/// Coefficients per short window.
pub const SHORT_LEN: usize = 128;
/// Maximum scalefactor bands of a window group (long windows: 51).
pub const MAX_SFB: usize = 51;
/// Maximum (group × band) entries: 8 groups × 15 short bands = 120 < 8×16.
const MAX_BANDS: usize = 8 * 16;
/// `scalefactor - SF_OFFSET` is the gain exponent in quarter steps.
const SF_OFFSET: i32 = 100;
/// The scalefactor Huffman alphabet is centred on this index.
const SCALE_DIFF_ZERO: i32 = 60;
/// Noise energy starts at `global_gain - NOISE_OFFSET` and the first noise
/// band adds a 9-bit PCM value biased by `NOISE_PRE`.
const NOISE_OFFSET: i32 = 90;
const NOISE_PRE: i32 = 256;
const NOISE_PRE_BITS: u32 = 9;
/// Largest escape-coded magnitude (13 bits).
const MAX_QUANT: usize = 8191;
pub const TNS_MAX_ORDER: usize = 20;

/// Maximum TNS bands per sampling-frequency index (Table 4.155).
const TNS_MAX_BANDS_LONG: [u8; 13] = [31, 31, 34, 40, 42, 51, 46, 46, 42, 42, 42, 39, 39];
const TNS_MAX_BANDS_SHORT: [u8; 13] = [9, 9, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowSequence {
	#[default]
	OnlyLong,
	LongStart,
	EightShort,
	LongStop,
}

/// Band types (`sect_cb`): 0 zero, 1-11 spectral codebooks, 13 noise,
/// 14/15 intensity (out of phase / in phase).
pub const ZERO_BT: u8 = 0;
pub const ESC_BT: u8 = 11;
pub const NOISE_BT: u8 = 13;
pub const INTENSITY_BT2: u8 = 14;
pub const INTENSITY_BT: u8 = 15;

#[derive(Debug, Clone, Default)]
pub struct IcsInfo {
	pub window_sequence: WindowSequence,
	/// `true` = Kaiser-Bessel-derived window, `false` = sine window.
	pub kbd_window: bool,
	pub max_sfb: usize,
	pub num_windows: usize,
	pub num_window_groups: usize,
	pub group_len: [u8; 8],
	pub num_swb: usize,
	pub swb_offset: &'static [u16],
	pub tns_max_bands: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TnsFilter {
	pub length: u8,
	pub order: u8,
	pub direction: bool,
	/// Reflection (PARCOR) coefficients.
	pub coef: [f32; TNS_MAX_ORDER],
}

#[derive(Debug, Clone, Default)]
pub struct Tns {
	pub present: bool,
	pub n_filt: [u8; 8],
	pub filters: [[TnsFilter; 4]; 8],
}

/// One decoded channel of a frame.
#[derive(Debug, Clone)]
pub struct Channel {
	pub ics: IcsInfo,
	/// Band type per (group, sfb), `g * max_sfb + sfb`.
	pub band_type: [u8; MAX_BANDS],
	/// Scalefactor, noise energy or intensity position per (group, sfb).
	pub sf: [i32; MAX_BANDS],
	pub tns: Tns,
	pub coef: [f32; FRAME_LEN],
	quant: [i32; FRAME_LEN],
}

impl Default for Channel {
	fn default() -> Self {
		Self {
			ics: IcsInfo::default(),
			band_type: [0; MAX_BANDS],
			sf: [0; MAX_BANDS],
			tns: Tns::default(),
			coef: [0.0; FRAME_LEN],
			quant: [0; FRAME_LEN],
		}
	}
}

/// Decoding failure inside an element; the frame is concealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Invalid(pub &'static str);

pub type IcsResult<T> = Result<T, Invalid>;

/// Static lookup tables built once per process.
pub struct Tables {
	pub spectrum: Vec<Vlc>,
	pub scalefactor: Vlc,
	/// `|q|^(4/3)` for `q` in `0..=8191`.
	pub pow43: Vec<f32>,
}

pub fn tables() -> &'static Tables {
	static TABLES: OnceLock<Tables> = OnceLock::new();
	TABLES.get_or_init(|| {
		let build = |codes: &[(u32, u8)]| {
			let list: Vec<Code> = codes
				.iter()
				.enumerate()
				.map(|(i, &(code, len))| Code { code, len, symbol: i as u32 })
				.collect();
			// The tables are generated and checked prefix-free by the generator.
			Vlc::new(&list, 8).unwrap_or_else(Vlc::empty)
		};
		Tables {
			spectrum: SPECTRUM_CODES.iter().map(|c| build(c)).collect(),
			scalefactor: build(&SCALEFACTOR_CODES),
			pow43: (0..=MAX_QUANT).map(|q| (q as f64).powf(4.0 / 3.0) as f32).collect(),
		}
	})
}

/// `2^(e/4)` for a quarter-step exponent.
#[inline]
pub fn pow2_quarter(e: i32) -> f32 {
	(2.0f64).powf(f64::from(e) * 0.25) as f32
}

/// Parses `ics_info()` (Table 4.6).
pub fn decode_ics_info(bs: &mut BitReader, rate_index: usize, ics: &mut IcsInfo) -> IcsResult<()> {
	if bs.bit() {
		// ics_reserved_bit; FFmpeg tolerates it, so do we.
	}
	ics.window_sequence = match bs.bits(2) {
		0 => WindowSequence::OnlyLong,
		1 => WindowSequence::LongStart,
		2 => WindowSequence::EightShort,
		_ => WindowSequence::LongStop,
	};
	ics.kbd_window = bs.bit();
	let band_index = rate_index.min(11);
	if ics.window_sequence == WindowSequence::EightShort {
		ics.max_sfb = bs.bits(4) as usize;
		let grouping = bs.bits(7);
		ics.num_windows = 8;
		ics.num_window_groups = 1;
		ics.group_len = [0; 8];
		ics.group_len[0] = 1;
		for i in 0..7 {
			if grouping & (1 << (6 - i)) != 0 {
				ics.group_len[ics.num_window_groups - 1] += 1;
			} else {
				ics.num_window_groups += 1;
				ics.group_len[ics.num_window_groups - 1] = 1;
			}
		}
		ics.swb_offset = SWB_OFFSET_SHORT[band_index];
		ics.tns_max_bands = usize::from(TNS_MAX_BANDS_SHORT[rate_index.min(12)]);
	} else {
		ics.max_sfb = bs.bits(6) as usize;
		ics.num_windows = 1;
		ics.num_window_groups = 1;
		ics.group_len = [1, 0, 0, 0, 0, 0, 0, 0];
		ics.swb_offset = SWB_OFFSET_LONG[band_index];
		ics.tns_max_bands = usize::from(TNS_MAX_BANDS_LONG[rate_index.min(12)]);
		if bs.bit() {
			return Err(Invalid("aac main prediction and ltp are not supported"));
		}
	}
	ics.num_swb = ics.swb_offset.len() - 1;
	if ics.max_sfb > ics.num_swb {
		return Err(Invalid("aac max_sfb exceeds the band count"));
	}
	Ok(())
}

/// Parses `section_data()` (Table 4.52).
fn decode_band_types(bs: &mut BitReader, ch: &mut Channel) -> IcsResult<()> {
	let ics = &ch.ics;
	let (bits, esc) =
		if ics.window_sequence == WindowSequence::EightShort { (3, 7) } else { (5, 31) };
	for g in 0..ics.num_window_groups {
		let mut k = 0;
		while k < ics.max_sfb {
			let cb = bs.bits(4) as u8;
			if cb == 12 {
				return Err(Invalid("aac reserved codebook"));
			}
			let mut len = 0usize;
			loop {
				let incr = bs.bits(bits) as usize;
				len += incr;
				if incr != esc || bs.overrun() {
					break;
				}
			}
			if bs.overrun() || k + len > ics.max_sfb {
				return Err(Invalid("aac section exceeds max_sfb"));
			}
			for slot in &mut ch.band_type[g * ics.max_sfb + k..g * ics.max_sfb + k + len] {
				*slot = cb;
			}
			k += len;
			if len == 0 && bs.overrun() {
				return Err(Invalid("aac truncated section data"));
			}
		}
	}
	Ok(())
}

/// Parses `scale_factor_data()` (Table 4.53).
fn decode_scalefactors(bs: &mut BitReader, ch: &mut Channel, global_gain: i32) -> IcsResult<()> {
	let t = tables();
	let mut sf = global_gain;
	let mut noise = global_gain - NOISE_OFFSET;
	let mut is_pos = 0i32;
	let mut first_noise = true;
	let ics = &ch.ics;
	for idx in 0..ics.num_window_groups * ics.max_sfb {
		let diff = |bs: &mut BitReader| -> IcsResult<i32> {
			t.scalefactor
				.decode(bs)
				.map(|v| v as i32 - SCALE_DIFF_ZERO)
				.ok_or(Invalid("aac invalid scalefactor code"))
		};
		ch.sf[idx] = match ch.band_type[idx] {
			ZERO_BT => 0,
			INTENSITY_BT | INTENSITY_BT2 => {
				is_pos += diff(bs)?;
				is_pos = is_pos.clamp(-155, 100);
				is_pos
			}
			NOISE_BT => {
				if first_noise {
					first_noise = false;
					noise += bs.bits(NOISE_PRE_BITS) as i32 - NOISE_PRE;
				} else {
					noise += diff(bs)?;
				}
				noise = noise.clamp(-100, 155);
				noise
			}
			_ => {
				sf += diff(bs)?;
				if !(0..=255).contains(&sf) {
					return Err(Invalid("aac scalefactor out of range"));
				}
				sf
			}
		};
	}
	Ok(())
}

#[derive(Debug, Default)]
struct Pulse {
	count: usize,
	start: usize,
	offset: [u8; 4],
	amp: [u8; 4],
}

fn decode_pulse(bs: &mut BitReader, ics: &IcsInfo) -> IcsResult<Pulse> {
	let count = bs.bits(2) as usize + 1;
	let start_sfb = bs.bits(6) as usize;
	if start_sfb >= ics.num_swb {
		return Err(Invalid("aac pulse start band out of range"));
	}
	let mut p = Pulse { count, start: usize::from(ics.swb_offset[start_sfb]), ..Pulse::default() };
	for i in 0..count {
		p.offset[i] = bs.bits(5) as u8;
		p.amp[i] = bs.bits(4) as u8;
	}
	Ok(p)
}

/// Parses `tns_data()` (Table 4.48) and dequantises the reflection
/// coefficients (§4.6.9.3).
fn decode_tns(bs: &mut BitReader, ics: &IcsInfo, object_type: u8, tns: &mut Tns) -> IcsResult<()> {
	let short = ics.window_sequence == WindowSequence::EightShort;
	let max_order = if short {
		7
	} else if object_type == super::config::AOT_AAC_MAIN {
		20
	} else {
		12
	};
	tns.present = true;
	for w in 0..ics.num_windows {
		let n_filt = bs.bits(if short { 1 } else { 2 }) as usize;
		tns.n_filt[w] = n_filt as u8;
		if n_filt == 0 {
			continue;
		}
		let coef_res = bs.bits(1);
		for filt in &mut tns.filters[w][..n_filt] {
			filt.length = bs.bits(if short { 4 } else { 6 }) as u8;
			filt.order = bs.bits(if short { 3 } else { 5 }) as u8;
			if usize::from(filt.order) > max_order {
				return Err(Invalid("aac tns order too high"));
			}
			if filt.order == 0 {
				continue;
			}
			filt.direction = bs.bit();
			let compress = bs.bits(1);
			let res_bits = coef_res + 3;
			let len = res_bits - compress;
			let half = f64::from(1u32 << (res_bits - 1));
			let iq_pos = (half - 0.5) / std::f64::consts::FRAC_PI_2;
			let iq_neg = (half + 0.5) / std::f64::consts::FRAC_PI_2;
			for c in &mut filt.coef[..usize::from(filt.order)] {
				let v = f64::from(bs.signed(len));
				*c = (v / if v >= 0.0 { iq_pos } else { iq_neg }).sin() as f32;
			}
		}
	}
	Ok(())
}

/// Decodes `individual_channel_stream()` (Table 4.50) into `ch`.
///
/// `common_window` means `ch.ics` was already filled by the CPE. `random`
/// is the decoder's noise generator state (FFmpeg-compatible LCG).
pub fn decode_ics(
	bs: &mut BitReader,
	ch: &mut Channel,
	rate_index: usize,
	object_type: u8,
	common_window: bool,
	random: &mut u32,
) -> IcsResult<()> {
	let global_gain = bs.bits(8) as i32;
	if !common_window {
		decode_ics_info(bs, rate_index, &mut ch.ics)?;
	}
	decode_band_types(bs, ch)?;
	decode_scalefactors(bs, ch, global_gain)?;

	let short = ch.ics.window_sequence == WindowSequence::EightShort;
	let mut pulse = None;
	if bs.bit() {
		if short {
			return Err(Invalid("aac pulse data in short window"));
		}
		pulse = Some(decode_pulse(bs, &ch.ics)?);
	}
	ch.tns.present = false;
	if bs.bit() {
		decode_tns(bs, &ch.ics, object_type, &mut ch.tns)?;
	}
	if bs.bit() {
		return Err(Invalid("aac gain control (ssr) is not supported"));
	}
	if bs.overrun() {
		return Err(Invalid("aac truncated channel header"));
	}
	decode_spectrum(bs, ch)?;
	if let Some(p) = pulse {
		apply_pulses(ch, &p)?;
	}
	dequantise(ch, random);
	if bs.overrun() {
		return Err(Invalid("aac truncated spectral data"));
	}
	Ok(())
}

/// Huffman-decodes every band into `ch.quant` (FFmpeg decodes and
/// dequantises in one pass; RMP keeps integers first so pulses can be added
/// exactly as the specification describes).
fn decode_spectrum(bs: &mut BitReader, ch: &mut Channel) -> IcsResult<()> {
	let t = tables();
	ch.quant.fill(0);
	let ics = &ch.ics;
	let offsets = ics.swb_offset;
	let mut window = 0usize;
	for g in 0..ics.num_window_groups {
		let glen = usize::from(ics.group_len[g]);
		for sfb in 0..ics.max_sfb {
			let cb = ch.band_type[g * ics.max_sfb + sfb];
			if cb == ZERO_BT || cb >= NOISE_BT {
				continue;
			}
			let vlc = &t.spectrum[usize::from(cb) - 1];
			let (lo, hi) = (usize::from(offsets[sfb]), usize::from(offsets[sfb + 1]));
			for w in window..window + glen {
				let base = w * SHORT_LEN;
				let band =
					ch.quant.get_mut(base + lo..base + hi).ok_or(Invalid("aac band outside frame"))?;
				decode_band(bs, vlc, cb, band)?;
			}
		}
		window += glen;
	}
	Ok(())
}

/// Decodes one band of one window with codebook `cb` (1-11).
#[inline]
fn decode_band(bs: &mut BitReader, vlc: &Vlc, cb: u8, band: &mut [i32]) -> IcsResult<()> {
	let bad = Invalid("aac invalid spectral code");
	match cb {
		1 | 2 => {
			for quad in band.chunks_exact_mut(4) {
				let idx = vlc.decode(bs).ok_or(bad)? as i32;
				quad[0] = idx / 27 - 1;
				quad[1] = (idx / 9) % 3 - 1;
				quad[2] = (idx / 3) % 3 - 1;
				quad[3] = idx % 3 - 1;
			}
		}
		3 | 4 => {
			for quad in band.chunks_exact_mut(4) {
				let idx = vlc.decode(bs).ok_or(bad)? as i32;
				quad[0] = idx / 27;
				quad[1] = (idx / 9) % 3;
				quad[2] = (idx / 3) % 3;
				quad[3] = idx % 3;
				apply_signs(bs, quad);
			}
		}
		5 | 6 => {
			for pair in band.chunks_exact_mut(2) {
				let idx = vlc.decode(bs).ok_or(bad)? as i32;
				pair[0] = idx / 9 - 4;
				pair[1] = idx % 9 - 4;
			}
		}
		_ => {
			let modulo = match cb {
				7 | 8 => 8,
				9 | 10 => 13,
				_ => 17,
			};
			for pair in band.chunks_exact_mut(2) {
				let idx = vlc.decode(bs).ok_or(bad)? as i32;
				pair[0] = idx / modulo;
				pair[1] = idx % modulo;
				apply_signs(bs, pair);
				if cb == ESC_BT {
					for v in pair.iter_mut() {
						if v.abs() == 16 {
							let e = decode_escape(bs)?;
							*v = if *v < 0 { -e } else { e };
						}
					}
				}
			}
		}
	}
	if bs.overrun() {
		return Err(Invalid("aac truncated spectral data"));
	}
	Ok(())
}

#[inline]
fn apply_signs(bs: &mut BitReader, values: &mut [i32]) {
	for v in values.iter_mut() {
		if *v != 0 && bs.bit() {
			*v = -*v;
		}
	}
}

/// Escape sequence: `N` ones, a zero, then `N + 4` bits (§4.6.3.3).
fn decode_escape(bs: &mut BitReader) -> IcsResult<i32> {
	let mut n = 0u32;
	while bs.bit() {
		n += 1;
		if n > 8 {
			return Err(Invalid("aac escape sequence too long"));
		}
	}
	Ok(((1u32 << (n + 4)) + bs.bits(n + 4)) as i32)
}

fn apply_pulses(ch: &mut Channel, p: &Pulse) -> IcsResult<()> {
	let mut pos = p.start;
	for i in 0..p.count {
		pos += usize::from(p.offset[i]);
		let q = ch.quant.get_mut(pos).ok_or(Invalid("aac pulse position out of range"))?;
		let amp = i32::from(p.amp[i]);
		*q += if *q > 0 { amp } else { -amp };
		if q.unsigned_abs() as usize > MAX_QUANT {
			return Err(Invalid("aac pulse overflow"));
		}
	}
	Ok(())
}

/// FFmpeg's noise generator (`lcg_random`), so PNS output matches FFmpeg.
#[inline]
fn lcg_random(state: u32) -> u32 {
	state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// Inverse quantisation and noise substitution (§4.6.1.3, §4.6.13).
fn dequantise(ch: &mut Channel, random: &mut u32) {
	let t = tables();
	let ics = &ch.ics;
	let offsets = ics.swb_offset;
	let win_len = FRAME_LEN / ics.num_windows;
	ch.coef.fill(0.0);
	let mut window = 0usize;
	for g in 0..ics.num_window_groups {
		let glen = usize::from(ics.group_len[g]);
		for sfb in 0..ics.max_sfb {
			let idx = g * ics.max_sfb + sfb;
			let (lo, hi) = (usize::from(offsets[sfb]), usize::from(offsets[sfb + 1]));
			if lo >= win_len {
				break;
			}
			let hi = hi.min(win_len);
			match ch.band_type[idx] {
				ZERO_BT | INTENSITY_BT | INTENSITY_BT2 => {}
				NOISE_BT => {
					let gain = pow2_quarter(ch.sf[idx]);
					for w in window..window + glen {
						let band = &mut ch.coef[w * SHORT_LEN + lo..w * SHORT_LEN + hi];
						let mut energy = 0.0f32;
						for c in band.iter_mut() {
							*random = lcg_random(*random);
							*c = *random as i32 as f32;
							energy += *c * *c;
						}
						let scale = if energy > 0.0 { gain / energy.sqrt() } else { 0.0 };
						for c in band.iter_mut() {
							*c *= scale;
						}
					}
				}
				_ => {
					let gain = pow2_quarter(ch.sf[idx] - SF_OFFSET);
					for w in window..window + glen {
						let base = w * SHORT_LEN;
						for k in base + lo..base + hi {
							let q = ch.quant[k];
							let m = t.pow43[(q.unsigned_abs() as usize).min(MAX_QUANT)] * gain;
							ch.coef[k] = if q < 0 { -m } else { m };
						}
					}
				}
			}
		}
		window += glen;
	}
}

/// Applies the TNS all-pole filters in place (§4.6.9.3).
pub fn apply_tns(ch: &mut Channel) {
	if !ch.tns.present {
		return;
	}
	let ics = &ch.ics;
	let mmm = ics.tns_max_bands.min(ics.max_sfb);
	if mmm == 0 {
		return;
	}
	for w in 0..ics.num_windows {
		let mut bottom = ics.num_swb;
		for f in &ch.tns.filters[w][..usize::from(ch.tns.n_filt[w])] {
			let top = bottom;
			bottom = top.saturating_sub(usize::from(f.length));
			let order = usize::from(f.order);
			if order == 0 {
				continue;
			}
			let lpc = parcor_to_lpc(&f.coef[..order]);
			let start = usize::from(ics.swb_offset[bottom.min(mmm)]);
			let end = usize::from(ics.swb_offset[top.min(mmm)]);
			if end <= start {
				continue;
			}
			let region = &mut ch.coef[w * SHORT_LEN + start..w * SHORT_LEN + end];
			let size = region.len();
			// y[n] = x[n] - Σ lpc[i] · y[n - i], walking up or down.
			for m in 0..size {
				let at = if f.direction { size - 1 - m } else { m };
				let mut acc = region[at];
				for i in 1..=m.min(order) {
					let prev = if f.direction { at + i } else { at - i };
					acc -= region[prev] * lpc[i - 1];
				}
				region[at] = acc;
			}
		}
	}
}

/// Step-up recursion from reflection to direct-form coefficients.
fn parcor_to_lpc(parcor: &[f32]) -> [f32; TNS_MAX_ORDER] {
	let mut lpc = [0.0f32; TNS_MAX_ORDER];
	for (i, &r) in parcor.iter().enumerate() {
		lpc[i] = r;
		for j in 0..i.div_ceil(2) {
			let f = lpc[j];
			let b = lpc[i - 1 - j];
			lpc[j] = f + r * b;
			lpc[i - 1 - j] = b + r * f;
		}
	}
	lpc
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn codebooks_build() {
		let t = tables();
		assert_eq!(t.spectrum.len(), 11);
		// Scalefactor difference 0 is the 1-bit code "0".
		let mut bs = BitReader::new(&[0x00]);
		assert_eq!(t.scalefactor.decode(&mut bs), Some(60));
		assert_eq!(bs.pos(), 1);
	}

	#[test]
	fn parcor_step_up() {
		let lpc = parcor_to_lpc(&[0.5, 0.25]);
		assert!((lpc[0] - 0.625).abs() < 1e-6);
		assert!((lpc[1] - 0.25).abs() < 1e-6);
	}
}
