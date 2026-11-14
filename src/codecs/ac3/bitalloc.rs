//! The AC-3 parametric bit allocation (ATSC A/52 §7.2; FFmpeg `ac3.c`
//! and `ac3dsp.c`). Integer-exact: encoder and decoder must agree on
//! every bap.

use super::tables::{BAND_START, BIN_TO_BAND, HEARING_THRESHOLD, LOG_ADD};

pub const CRITICAL_BANDS: usize = 50;
pub const MAX_COEFS: usize = 256;

/// Delta bit allocation modes (`deltbae`).
pub const DBA_REUSE: u8 = 0;
pub const DBA_NEW: u8 = 1;
pub const DBA_NONE: u8 = 2;
pub const DBA_RESERVED: u8 = 3;

pub const SLOW_DECAY: [u8; 4] = [0x0f, 0x11, 0x13, 0x15];
pub const FAST_DECAY: [u8; 4] = [0x3f, 0x53, 0x67, 0x7b];
pub const SLOW_GAIN: [u16; 4] = [0x540, 0x4d8, 0x478, 0x410];
pub const DB_PER_BIT: [u16; 4] = [0x000, 0x700, 0x900, 0xb00];
pub const FLOOR: [i16; 8] = [0x2f0, 0x2b0, 0x270, 0x230, 0x1f0, 0x170, 0x0f0, -2048];
pub const FAST_GAIN: [u16; 8] = [0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400];

/// Frame-level bit allocation parameters (`sdcycod`, `fdcycod`, ...).
#[derive(Debug, Clone, Copy, Default)]
pub struct Params {
	pub sr_code: u8,
	pub sr_shift: u8,
	pub slow_gain: i32,
	pub slow_decay: i32,
	pub fast_decay: i32,
	pub db_per_bit: i32,
	pub floor: i32,
	pub cpl_fast_leak: i32,
	pub cpl_slow_leak: i32,
}

/// Delta bit allocation for one channel.
#[derive(Debug, Clone, Copy)]
pub struct Dba {
	pub mode: u8,
	pub nsegs: usize,
	pub offsets: [u8; 8],
	pub lengths: [u8; 8],
	pub values: [u8; 8],
}

impl Default for Dba {
	fn default() -> Self {
		Self { mode: DBA_NONE, nsegs: 0, offsets: [0; 8], lengths: [0; 8], values: [0; 8] }
	}
}

/// Exponent mapping into PSD and PSD integration (§7.2.2.2-3).
pub fn calc_psd(
	exp: &[i8; MAX_COEFS],
	start: usize,
	end: usize,
	psd: &mut [i16; MAX_COEFS],
	band_psd: &mut [i16; CRITICAL_BANDS],
) {
	for bin in start..end {
		psd[bin] = 3072 - (i16::from(exp[bin]) << 7);
	}
	let mut bin = start;
	while bin < end.min(28) {
		band_psd[bin] = psd[bin];
		bin += 1;
	}
	if bin >= end {
		return;
	}
	let mut band = usize::from(BIN_TO_BAND[bin]);
	loop {
		let mut v = i32::from(psd[bin]);
		bin += 1;
		let band_end = usize::from(BAND_START[band + 1]).min(end);
		while bin < band_end {
			let p = i32::from(psd[bin]);
			let max = v.max(p);
			let adr = (max - ((v + p + 1) >> 1)).min(255) as usize;
			v = max + i32::from(LOG_ADD[adr]);
			bin += 1;
		}
		band_psd[band] = v as i16;
		band += 1;
		if end <= usize::from(BAND_START[band]) {
			break;
		}
	}
}

#[inline]
fn lowcomp1(a: i32, b0: i32, b1: i32, c: i32) -> i32 {
	if b0 + 256 == b1 {
		c
	} else if b0 > b1 {
		(a - 64).max(0)
	} else {
		a
	}
}

#[inline]
fn lowcomp(a: i32, b0: i32, b1: i32, bin: usize) -> i32 {
	if bin < 7 {
		lowcomp1(a, b0, b1, 384)
	} else if bin < 20 {
		lowcomp1(a, b0, b1, 320)
	} else {
		(a - 128).max(0)
	}
}

/// Excitation, masking curve and delta bit allocation (§7.2.2.4-6).
/// Returns `false` for an invalid delta bit allocation.
#[allow(clippy::too_many_arguments)]
pub fn calc_mask(
	p: &Params,
	band_psd: &[i16; CRITICAL_BANDS],
	start: usize,
	end: usize,
	fast_gain: i32,
	is_lfe: bool,
	dba: &Dba,
	mask: &mut [i16; CRITICAL_BANDS],
) -> bool {
	if end == 0 || start >= end {
		return false;
	}
	let bp = |b: usize| i32::from(band_psd[b]);
	let mut excite = [0i32; CRITICAL_BANDS];
	let band_start = usize::from(BIN_TO_BAND[start]);
	let band_end = usize::from(BIN_TO_BAND[end - 1]) + 1;
	let begin;
	let (mut fastleak, mut slowleak);
	if band_start == 0 {
		let mut lc = lowcomp1(0, bp(0), bp(1), 384);
		excite[0] = bp(0) - fast_gain - lc;
		lc = lowcomp1(lc, bp(1), bp(2), 384);
		excite[1] = bp(1) - fast_gain - lc;
		let mut b = 7;
		fastleak = 0;
		slowleak = 0;
		for band in 2..7 {
			if !(is_lfe && band == 6) {
				lc = lowcomp1(lc, bp(band), bp(band + 1), 384);
			}
			fastleak = bp(band) - fast_gain;
			slowleak = bp(band) - p.slow_gain;
			excite[band] = fastleak - lc;
			if !(is_lfe && band == 6) && bp(band) <= bp(band + 1) {
				b = band + 1;
				break;
			}
		}
		let end1 = band_end.min(22);
		for band in b..end1 {
			if !(is_lfe && band == 6) {
				lc = lowcomp(lc, bp(band), bp(band + 1), band);
			}
			fastleak = (fastleak - p.fast_decay).max(bp(band) - fast_gain);
			slowleak = (slowleak - p.slow_decay).max(bp(band) - p.slow_gain);
			excite[band] = (fastleak - lc).max(slowleak);
		}
		begin = 22;
	} else {
		begin = band_start;
		fastleak = (p.cpl_fast_leak << 8) + 768;
		slowleak = (p.cpl_slow_leak << 8) + 768;
	}
	for band in begin..band_end {
		fastleak = (fastleak - p.fast_decay).max(bp(band) - fast_gain);
		slowleak = (slowleak - p.slow_decay).max(bp(band) - p.slow_gain);
		excite[band] = fastleak.max(slowleak);
	}
	for band in band_start..band_end {
		let tmp = p.db_per_bit - bp(band);
		if tmp > 0 {
			excite[band] += tmp >> 2;
		}
		let hth = i32::from(HEARING_THRESHOLD[band >> p.sr_shift][usize::from(p.sr_code)]);
		mask[band] = hth.max(excite[band]) as i16;
	}
	if dba.mode == DBA_REUSE || dba.mode == DBA_NEW {
		if dba.nsegs > 8 {
			return false;
		}
		let mut band = band_start;
		for seg in 0..dba.nsegs {
			band += usize::from(dba.offsets[seg]);
			let len = usize::from(dba.lengths[seg]);
			if band >= CRITICAL_BANDS || len > CRITICAL_BANDS - band {
				return false;
			}
			let v = i32::from(dba.values[seg]);
			let delta = if v >= 4 { (v - 3) * 128 } else { (v - 4) * 128 };
			for m in &mut mask[band..band + len] {
				*m = (i32::from(*m) + delta) as i16;
			}
			band += len;
		}
	}
	true
}

/// Bit allocation pointers from the mask and PSD (§7.2.2.7).
#[allow(clippy::too_many_arguments)]
pub fn calc_bap(
	mask: &[i16; CRITICAL_BANDS],
	psd: &[i16; MAX_COEFS],
	start: usize,
	end: usize,
	snr_offset: i32,
	floor: i32,
	bap_tab: &[u8; 64],
	bap: &mut [u8; MAX_COEFS],
) {
	if snr_offset == -960 {
		bap.fill(0);
		return;
	}
	let masked = |m: i16| ((i32::from(m) - snr_offset - floor).max(0) & 0x1FE0) + floor;
	let address = |psd: i16, m: i32| ((i32::from(psd) - m) >> 5).clamp(0, 63) as usize;
	let mut bin = start;
	while bin < end.min(28) {
		let m = masked(mask[bin]);
		bap[bin] = bap_tab[address(psd[bin], m)];
		bin += 1;
	}
	if bin >= end {
		return;
	}
	let mut band = usize::from(BIN_TO_BAND[bin]);
	loop {
		let m = masked(mask[band]);
		band += 1;
		let band_end = usize::from(BAND_START[band]).min(end);
		while bin < band_end {
			bap[bin] = bap_tab[address(psd[bin], m)];
			bin += 1;
		}
		if end <= band_end {
			break;
		}
	}
}
