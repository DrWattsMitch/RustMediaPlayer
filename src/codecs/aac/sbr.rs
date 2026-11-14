//! Spectral Band Replication (ISO/IEC 14496-3 §4.6.18; FFmpeg
//! `aacsbr.c`, `aacsbr_template.c`, `sbrdsp.c`).
//!
//! Per AAC channel element: the 1024 core samples go through a 32-band
//! complex QMF analysis, the high band is regenerated from the low band
//! (patching, inverse filtering, envelope adjustment with noise and
//! sinusoids), optionally parametric stereo is applied, and a 64-band QMF
//! synthesis produces 2048 samples at twice the core rate.
//!
//! The arithmetic follows FFmpeg's float decoder step for step so the output
//! matches it; the core input is RMP's ±32768-scaled float PCM.

use super::ps::Ps;
use super::sbr_tables::*;
use crate::codecs::bitstream::BitReader;
use crate::codecs::bitstream::{Code, Vlc};
use crate::codecs::dsp::{Complex, Imdct};
use std::sync::OnceLock;

const ENVELOPE_ADJUSTMENT_OFFSET: usize = 2;
const NOISE_FLOOR_OFFSET: i32 = 6;
const SYNTHESIS_BUF_SIZE: usize = (1280 - 128) * 2;
const TIME_SLOTS: usize = 16;
const QMF_SLOTS: usize = 2 * TIME_SLOTS;

const FIXFIX: u32 = 0;
const FIXVAR: u32 = 1;
const VARFIX: u32 = 2;
#[allow(dead_code)]
const VARVAR: u32 = 3;

// Huffman table indices (FFmpeg `ff_aac_sbr_vlc` order).
const T_ENV_1_5DB: usize = 0;
const F_ENV_1_5DB: usize = 1;
const T_ENV_BAL_1_5DB: usize = 2;
const F_ENV_BAL_1_5DB: usize = 3;
const T_ENV_3_0DB: usize = 4;
const F_ENV_3_0DB: usize = 5;
const T_ENV_BAL_3_0DB: usize = 6;
const F_ENV_BAL_3_0DB: usize = 7;
const T_NOISE_3_0DB: usize = 8;
const T_NOISE_BAL_3_0DB: usize = 9;

const EXTENSION_ID_PS: u32 = 2;

pub type Cplx = [f32; 2];

/// Builds VLC tables from FFmpeg-style `(symbol, length)` lists in code
/// order (`ff_vlc_init_from_lengths`).
pub(crate) fn vlcs_from_lengths(entries: &[(u8, u8)], sizes: &[u8], offsets: &[i8]) -> Vec<Vlc> {
	let mut tables = Vec::with_capacity(sizes.len());
	let mut at = 0;
	for (&n, &off) in sizes.iter().zip(offsets) {
		let mut acc: u64 = 0;
		let mut codes = Vec::with_capacity(usize::from(n));
		for &(sym, len) in &entries[at..at + usize::from(n)] {
			let shift = 32 - u32::from(len);
			codes.push(Code {
				code: (acc >> shift) as u32,
				len,
				symbol: (i32::from(sym) + i32::from(off)) as u32,
			});
			acc += 1u64 << shift;
		}
		tables.push(Vlc::new(&codes, 9).expect("valid sbr/ps huffman table"));
		at += usize::from(n);
	}
	tables
}

#[inline]
fn read_vlc(bs: &mut BitReader, vlc: &Vlc) -> i32 {
	vlc.decode(bs).map_or(0, |s| s as i32)
}

struct Shared {
	vlc: Vec<Vlc>,
	imdct: Imdct,
	analysis_scale: f32,
	synthesis_scale: f32,
}

fn shared() -> &'static Shared {
	static S: OnceLock<Shared> = OnceLock::new();
	S.get_or_init(|| Shared {
		vlc: vlcs_from_lengths(&SBR_HUFFMAN, &SBR_HUFFMAN_SIZES, &SBR_HUFFMAN_OFFSETS),
		// A 64-coefficient IMDCT; FFmpeg scales the analysis by -2*32768 for
		// ±1 input and the synthesis by 1/(64*32768). RMP's PCM is already
		// ±32768, so the 32768 factors drop out.
		imdct: Imdct::new(128, 1.0),
		analysis_scale: -2.0,
		synthesis_scale: 1.0 / 64.0,
	})
}

/// 2^x for integer x, exact (FFmpeg `exp2fi`).
fn exp2fi(x: i32) -> f32 {
	if (-126..=128).contains(&x) {
		f32::from_bits(((x + 127) as u32) << 23)
	} else if x > 128 {
		f32::INFINITY
	} else if x > -150 {
		f32::from_bits(1 << (x + 149))
	} else {
		0.0
	}
}

#[derive(Clone)]
pub struct SbrChannel {
	bs_frame_class: u32,
	bs_add_harmonic_flag: bool,
	bs_num_env: usize,
	bs_freq_res: [u8; 9],
	bs_num_noise: usize,
	bs_df_env: [u8; 9],
	bs_df_noise: [u8; 2],
	bs_invf_mode: [[u8; 5]; 2],
	bs_add_harmonic: [u8; 48],
	bs_amp_res: bool,
	synthesis: Vec<f32>,
	synthesis_offset: usize,
	analysis: Vec<f32>,
	e_a: [i32; 2],
	bw_array: [f32; 5],
	w: Box<[[[Cplx; 32]; 32]; 2]>,
	ypos: usize,
	y: Box<[[[Cplx; 64]; 38]; 2]>,
	g_temp: Box<[[f32; 48]; 42]>,
	q_temp: Box<[[f32; 48]; 42]>,
	s_indexmapped: [[u8; 48]; 9],
	env_facs_q: [[u8; 48]; 9],
	env_facs: [[f32; 48]; 9],
	noise_facs_q: [[u8; 5]; 3],
	noise_facs: [[f32; 5]; 3],
	t_env: [u8; 9],
	t_env_num_env_old: u8,
	t_q: [u8; 3],
	f_indexnoise: u32,
	f_indexsine: u32,
}

impl Default for SbrChannel {
	fn default() -> Self {
		Self {
			bs_frame_class: 0,
			bs_add_harmonic_flag: false,
			bs_num_env: 0,
			bs_freq_res: [0; 9],
			bs_num_noise: 0,
			bs_df_env: [0; 9],
			bs_df_noise: [0; 2],
			bs_invf_mode: [[0; 5]; 2],
			bs_add_harmonic: [0; 48],
			bs_amp_res: false,
			synthesis: vec![0.0; SYNTHESIS_BUF_SIZE],
			synthesis_offset: SYNTHESIS_BUF_SIZE - (1280 - 128),
			analysis: vec![0.0; 1312],
			e_a: [0, -1],
			bw_array: [0.0; 5],
			w: Box::new([[[[0.0; 2]; 32]; 32]; 2]),
			ypos: 0,
			y: Box::new([[[[0.0; 2]; 64]; 38]; 2]),
			g_temp: Box::new([[0.0; 48]; 42]),
			q_temp: Box::new([[0.0; 48]; 42]),
			s_indexmapped: [[0; 48]; 9],
			env_facs_q: [[0; 48]; 9],
			env_facs: [[0.0; 48]; 9],
			noise_facs_q: [[0; 5]; 3],
			noise_facs: [[0.0; 5]; 3],
			t_env: [0; 9],
			t_env_num_env_old: 0,
			t_q: [0; 3],
			f_indexnoise: 0,
			f_indexsine: 0,
		}
	}
}

/// Bitstream spectrum parameters; a change forces a frequency table reset.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Spectrum {
	start_freq: u8,
	stop_freq: u8,
	xover_band: u8,
	freq_scale: u8,
	alter_scale: u8,
	noise_bands: u8,
}

const SPECTRUM_UNSET: Spectrum = Spectrum {
	start_freq: 255,
	stop_freq: 255,
	xover_band: 255,
	freq_scale: 255,
	alter_scale: 255,
	noise_bands: 255,
};

/// SBR state of one channel element (FFmpeg `SpectralBandReplication`).
#[derive(Clone)]
pub struct Sbr {
	pub sample_rate: u32,
	start: bool,
	ready_for_dequant: bool,
	/// `true` for a channel pair element.
	pub is_cpe: bool,
	reset: bool,
	spectrum: Spectrum,
	bs_amp_res_header: bool,
	bs_limiter_bands: u32,
	bs_limiter_gains: u32,
	bs_interpol_freq: bool,
	bs_smoothing_mode: bool,
	bs_coupling: bool,
	k: [i32; 5],
	kx: [i32; 2],
	m: [i32; 2],
	kx_and_m_pushed: bool,
	n_master: i32,
	data: [SbrChannel; 2],
	pub ps: Box<Ps>,
	n: [i32; 2],
	n_q: i32,
	n_lim: i32,
	f_master: [u16; 49],
	f_tablelow: [u16; 25],
	f_tablehigh: [u16; 49],
	f_tablenoise: [u16; 6],
	f_tablelim: [u16; 30],
	num_patches: usize,
	patch_num_subbands: [u8; 6],
	patch_start_subband: [u8; 6],
	x_low: Box<[[Cplx; 40]; 32]>,
	x_high: Box<[[Cplx; 40]; 64]>,
	/// Reconstructed QMF signal per channel: `[re/im][slot][band]`.
	x: Box<[[[[f32; 64]; 38]; 2]; 2]>,
	alpha0: [Cplx; 64],
	alpha1: [Cplx; 64],
	e_origmapped: [[f32; 48]; 8],
	q_mapped: [[f32; 48]; 8],
	s_mapped: [[u8; 48]; 8],
	e_curr: [[f32; 48]; 8],
	q_m: [[f32; 48]; 8],
	s_m: [[f32; 48]; 8],
	gain: [[f32; 48]; 8],
	scratch: Vec<f32>,
	mdct_buf: [[f32; 64]; 2],
	work: Vec<f32>,
	cbuf: Vec<Complex>,
}

impl std::fmt::Debug for Sbr {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Sbr")
			.field("start", &self.start)
			.field("kx", &self.kx)
			.field("m", &self.m)
			.finish_non_exhaustive()
	}
}

impl Sbr {
	pub fn new(is_cpe: bool) -> Self {
		let mut s = Self {
			sample_rate: 0,
			start: false,
			ready_for_dequant: false,
			is_cpe,
			reset: false,
			spectrum: SPECTRUM_UNSET,
			bs_amp_res_header: false,
			bs_limiter_bands: 0,
			bs_limiter_gains: 0,
			bs_interpol_freq: false,
			bs_smoothing_mode: false,
			bs_coupling: false,
			k: [0; 5],
			kx: [32, 32],
			m: [0; 2],
			kx_and_m_pushed: false,
			n_master: 0,
			data: [SbrChannel::default(), SbrChannel::default()],
			ps: Box::default(),
			n: [0; 2],
			n_q: 0,
			n_lim: 0,
			f_master: [0; 49],
			f_tablelow: [0; 25],
			f_tablehigh: [0; 49],
			f_tablenoise: [0; 6],
			f_tablelim: [0; 30],
			num_patches: 0,
			patch_num_subbands: [0; 6],
			patch_start_subband: [0; 6],
			x_low: Box::new([[[0.0; 2]; 40]; 32]),
			x_high: Box::new([[[0.0; 2]; 40]; 64]),
			x: Box::new([[[[0.0; 64]; 38]; 2]; 2]),
			alpha0: [[0.0; 2]; 64],
			alpha1: [[0.0; 2]; 64],
			e_origmapped: [[0.0; 48]; 8],
			q_mapped: [[0.0; 48]; 8],
			s_mapped: [[0; 48]; 8],
			e_curr: [[0.0; 48]; 8],
			q_m: [[0.0; 48]; 8],
			s_m: [[0.0; 48]; 8],
			gain: [[0.0; 48]; 8],
			scratch: vec![0.0; 320],
			mdct_buf: [[0.0; 64]; 2],
			work: vec![0.0; 128],
			cbuf: vec![Complex::default(); 32],
		};
		s.turnoff();
		s
	}

	/// Pure upsampling mode (`sbr_turnoff`).
	fn turnoff(&mut self) {
		self.start = false;
		self.ready_for_dequant = false;
		self.kx[1] = 32;
		self.m[1] = 0;
		self.data[0].e_a[1] = -1;
		self.data[1].e_a[1] = -1;
		self.spectrum = SPECTRUM_UNSET;
	}

	// ---------------------------------------------------------------
	// Frequency tables
	// ---------------------------------------------------------------

	fn make_f_tablelim(&mut self) {
		if self.bs_limiter_bands > 0 {
			const WARPED: [f32; 3] = [1.327_151_8, 1.185_092_8, 1.119_871_6];
			let lim = WARPED[(self.bs_limiter_bands - 1) as usize];
			let mut patch_borders = [0i32; 7];
			patch_borders[0] = self.kx[1];
			for k in 1..=self.num_patches {
				patch_borders[k] = patch_borders[k - 1] + i32::from(self.patch_num_subbands[k - 1]);
			}
			let n0 = self.n[0] as usize;
			let mut table: Vec<u16> = self.f_tablelow[..=n0].to_vec();
			if self.num_patches > 1 {
				table.extend(patch_borders[1..self.num_patches].iter().map(|&b| b as u16));
			}
			table.sort_unstable();
			let in_borders = |v: u16| patch_borders[..=self.num_patches].contains(&i32::from(v));
			let mut n_lim = (self.n[0] + self.num_patches as i32 - 1) as usize;
			let mut out = 0usize;
			let mut inp = 1usize;
			while out < n_lim {
				let (iv, ov) = (table[inp], table[out]);
				if f32::from(iv) >= f32::from(ov) * lim {
					out += 1;
					table[out] = iv;
					inp += 1;
				} else if iv == ov || !in_borders(iv) {
					inp += 1;
					n_lim -= 1;
				} else if !in_borders(ov) {
					table[out] = iv;
					inp += 1;
					n_lim -= 1;
				} else {
					out += 1;
					table[out] = iv;
					inp += 1;
				}
			}
			self.n_lim = n_lim as i32;
			let len = table.len().min(30);
			self.f_tablelim[..len].copy_from_slice(&table[..len]);
		} else {
			self.f_tablelim[0] = self.f_tablelow[0];
			self.f_tablelim[1] = self.f_tablelow[self.n[0] as usize];
			self.n_lim = 1;
		}
	}

	fn make_f_master(&mut self) -> Option<()> {
		let sp = self.spectrum;
		let offsets = match self.sample_rate {
			16000 => &SBR_OFFSET[0],
			22050 => &SBR_OFFSET[1],
			24000 => &SBR_OFFSET[2],
			32000 => &SBR_OFFSET[3],
			44100 | 48000 | 64000 => &SBR_OFFSET[4],
			88200 | 96000 | 128000 | 176400 | 192000 => &SBR_OFFSET[5],
			_ => return None,
		};
		let sr = self.sample_rate;
		let temp: u32 = if sr < 32000 {
			3000
		} else if sr < 64000 {
			4000
		} else {
			5000
		};
		let start_min = ((temp << 7) + (sr >> 1)) / sr;
		let stop_min = ((temp << 8) + (sr >> 1)) / sr;
		self.k[0] = start_min as i32 + i32::from(offsets[usize::from(sp.start_freq & 15)]);
		if sp.stop_freq < 14 {
			self.k[2] = stop_min as i32;
			let mut stop_dk = make_bands(stop_min as i32, 64, 13);
			stop_dk.sort_unstable();
			for d in &stop_dk[..usize::from(sp.stop_freq)] {
				self.k[2] += i32::from(*d);
			}
		} else if sp.stop_freq == 14 {
			self.k[2] = 2 * self.k[0];
		} else if sp.stop_freq == 15 {
			self.k[2] = 3 * self.k[0];
		} else {
			return None;
		}
		self.k[2] = self.k[2].min(64);
		let max_qmf: i32 = if sr <= 32000 {
			48
		} else if sr == 44100 {
			35
		} else if sr >= 48000 {
			32
		} else {
			return None;
		};
		if self.k[2] - self.k[0] > max_qmf || self.k[0] <= 0 {
			return None;
		}
		if sp.freq_scale == 0 {
			let dk = i32::from(sp.alter_scale) + 1;
			self.n_master = ((self.k[2] - self.k[0] + (dk & 2)) >> dk) << 1;
			if self.n_master <= 0 || i32::from(sp.xover_band) >= self.n_master || self.n_master > 48 {
				return None;
			}
			let nm = self.n_master as usize;
			let mut fm = [0i32; 49];
			for v in &mut fm[1..=nm] {
				*v = dk;
			}
			let k2diff = self.k[2] - self.k[0] - self.n_master * dk;
			if k2diff < 0 {
				fm[1] -= 1;
				fm[2] -= i32::from(k2diff < -1);
			} else if k2diff != 0 {
				fm[nm] += 1;
			}
			fm[0] = self.k[0];
			for k in 1..=nm {
				fm[k] += fm[k - 1];
			}
			for k in 0..=nm {
				self.f_master[k] = fm[k] as u16;
			}
		} else {
			let half_bands = 7 - i32::from(sp.freq_scale);
			let two_regions = 49 * self.k[2] > 110 * self.k[0];
			self.k[1] = if two_regions { 2 * self.k[0] } else { self.k[2] };
			let num_bands_0 = ((half_bands as f32 * (self.k[1] as f32 / self.k[0] as f32).log2())
				.round_ties_even() as i32)
				* 2;
			if num_bands_0 <= 0 || num_bands_0 > 48 {
				return None;
			}
			let mut vk0 = vec![0i32; num_bands_0 as usize + 1];
			let mut bands = make_bands(self.k[0], self.k[1], num_bands_0);
			bands.sort_unstable();
			let vdk0_max = i32::from(*bands.last()?);
			vk0[0] = self.k[0];
			for k in 1..=num_bands_0 as usize {
				let d = i32::from(bands[k - 1]);
				if d <= 0 {
					return None;
				}
				vk0[k] = vk0[k - 1] + d;
			}
			if two_regions {
				let invwarp = if sp.alter_scale != 0 { 0.769_230_8f32 } else { 1.0 };
				let num_bands_1 =
					((half_bands as f32 * invwarp * (self.k[2] as f32 / self.k[1] as f32).log2())
						.round_ties_even() as i32)
						* 2;
				if num_bands_1 <= 0 {
					return None;
				}
				let mut vk1: Vec<i32> =
					make_bands(self.k[1], self.k[2], num_bands_1).iter().map(|&v| i32::from(v)).collect();
				let vdk1_min = *vk1.iter().min()?;
				if vdk1_min < vdk0_max {
					vk1.sort_unstable();
					let n1 = vk1.len();
					let change = (vdk0_max - vk1[0]).min((vk1[n1 - 1] - vk1[0]) >> 1);
					vk1[0] += change;
					vk1[n1 - 1] -= change;
				}
				vk1.sort_unstable();
				let mut acc = self.k[1];
				let mut cum = Vec::with_capacity(vk1.len());
				for d in vk1 {
					if d <= 0 {
						return None;
					}
					acc += d;
					cum.push(acc);
				}
				self.n_master = num_bands_0 + num_bands_1;
				if i32::from(sp.xover_band) >= self.n_master || self.n_master > 48 {
					return None;
				}
				for (k, v) in vk0.iter().enumerate() {
					self.f_master[k] = *v as u16;
				}
				for (k, v) in cum.iter().enumerate() {
					self.f_master[num_bands_0 as usize + 1 + k] = *v as u16;
				}
			} else {
				self.n_master = num_bands_0;
				if i32::from(sp.xover_band) >= self.n_master {
					return None;
				}
				for (k, v) in vk0.iter().enumerate() {
					self.f_master[k] = *v as u16;
				}
			}
		}
		Some(())
	}

	fn calc_npatches(&mut self) -> Option<()> {
		let mut msb = self.k[0];
		let mut usb = self.kx[1];
		let goal_sb = (((1000 << 11) + (self.sample_rate >> 1)) / self.sample_rate) as i32;
		self.num_patches = 0;
		let mut k: i32 = if goal_sb < self.kx[1] + self.m[1] {
			let mut k = 0;
			while k < self.n_master && i32::from(self.f_master[k as usize]) < goal_sb {
				k += 1;
			}
			k
		} else {
			self.n_master
		};
		let (mut last_k, mut last_msb, mut sb) = (-1, -1, 0i32);
		loop {
			let mut odd = 0;
			if k == last_k && msb == last_msb {
				return None;
			}
			last_k = k;
			last_msb = msb;
			let mut i = k;
			while i == k || sb > self.k[0] - 1 + msb - odd {
				if i < 0 {
					return None;
				}
				sb = i32::from(self.f_master[i as usize]);
				odd = (sb + self.k[0]) & 1;
				i -= 1;
			}
			if self.num_patches > 5 {
				return None;
			}
			let n = (sb - usb).max(0);
			self.patch_num_subbands[self.num_patches] = n as u8;
			self.patch_start_subband[self.num_patches] = (self.k[0] - odd - n) as u8;
			if n > 0 {
				usb = sb;
				msb = sb;
				self.num_patches += 1;
			} else {
				msb = self.kx[1];
			}
			if i32::from(self.f_master[k as usize]) - sb < 3 {
				k = self.n_master;
			}
			if sb == self.kx[1] + self.m[1] {
				break;
			}
		}
		if self.num_patches > 1 && self.patch_num_subbands[self.num_patches - 1] < 3 {
			self.num_patches -= 1;
		}
		Some(())
	}

	fn make_f_derived(&mut self) -> Option<()> {
		let xover = i32::from(self.spectrum.xover_band);
		self.n[1] = self.n_master - xover;
		self.n[0] = (self.n[1] + 1) >> 1;
		let n1 = self.n[1] as usize;
		let xo = xover as usize;
		self.f_tablehigh[..=n1].copy_from_slice(&self.f_master[xo..=xo + n1]);
		self.m[1] = i32::from(self.f_tablehigh[n1]) - i32::from(self.f_tablehigh[0]);
		self.kx[1] = i32::from(self.f_tablehigh[0]);
		if self.kx[1] + self.m[1] > 64 || self.kx[1] > 32 || self.m[1] > 48 {
			return None;
		}
		self.f_tablelow[0] = self.f_tablehigh[0];
		let temp = (self.n[1] & 1) as usize;
		for k in 1..=self.n[0] as usize {
			self.f_tablelow[k] = self.f_tablehigh[2 * k - temp];
		}
		self.n_q = ((f32::from(self.spectrum.noise_bands)
			* (self.k[2] as f32 / self.kx[1] as f32).log2())
		.round_ties_even() as i32)
			.max(1);
		if self.n_q > 5 {
			self.n_q = 1;
			return None;
		}
		self.f_tablenoise[0] = self.f_tablelow[0];
		let mut temp = 0i32;
		for k in 1..=self.n_q {
			temp += (self.n[0] - temp) / (self.n_q + 1 - k);
			self.f_tablenoise[k as usize] = self.f_tablelow[temp as usize];
		}
		self.calc_npatches()?;
		self.make_f_tablelim();
		self.data[0].f_indexnoise = 0;
		self.data[1].f_indexnoise = 0;
		Some(())
	}

	fn sbr_reset(&mut self) {
		if self.make_f_master().and_then(|_| self.make_f_derived()).is_none() {
			self.turnoff();
		}
	}

	// ---------------------------------------------------------------
	// Bitstream
	// ---------------------------------------------------------------

	fn read_header(&mut self, bs: &mut BitReader) {
		let old_limiter = self.bs_limiter_bands;
		let old = self.spectrum;
		self.start = true;
		self.ready_for_dequant = false;
		self.bs_amp_res_header = bs.bit();
		self.spectrum.start_freq = bs.bits(4) as u8;
		self.spectrum.stop_freq = bs.bits(4) as u8;
		self.spectrum.xover_band = bs.bits(3) as u8;
		bs.skip(2);
		let extra1 = bs.bit();
		let extra2 = bs.bit();
		if extra1 {
			self.spectrum.freq_scale = bs.bits(2) as u8;
			self.spectrum.alter_scale = bs.bits(1) as u8;
			self.spectrum.noise_bands = bs.bits(2) as u8;
		} else {
			self.spectrum.freq_scale = 2;
			self.spectrum.alter_scale = 1;
			self.spectrum.noise_bands = 2;
		}
		if old != self.spectrum {
			self.reset = true;
		}
		if extra2 {
			self.bs_limiter_bands = bs.bits(2);
			self.bs_limiter_gains = bs.bits(2);
			self.bs_interpol_freq = bs.bit();
			self.bs_smoothing_mode = bs.bit();
		} else {
			self.bs_limiter_bands = 2;
			self.bs_limiter_gains = 2;
			self.bs_interpol_freq = true;
			self.bs_smoothing_mode = true;
		}
		if self.bs_limiter_bands != old_limiter && !self.reset {
			self.make_f_tablelim();
		}
	}

	fn read_grid(&mut self, bs: &mut BitReader, ch: usize) -> Option<()> {
		const CEIL_LOG2: [u32; 6] = [0, 1, 2, 2, 3, 3];
		let amp_res_header = self.bs_amp_res_header;
		let d = &mut self.data[ch];
		let mut bs_pointer = 0u32;
		let mut abs_bord_trail = TIME_SLOTS as u32;
		let old_num_env = d.bs_num_env;
		d.bs_freq_res[0] = d.bs_freq_res[old_num_env];
		d.bs_amp_res = amp_res_header;
		d.t_env_num_env_old = d.t_env[old_num_env];
		let frame_class = bs.bits(2);
		match frame_class {
			FIXFIX => {
				let n = 1usize << bs.bits(2);
				if n > 5 {
					return None;
				}
				d.bs_num_env = n;
				if n == 1 {
					d.bs_amp_res = false;
				}
				d.t_env[0] = 0;
				d.t_env[n] = abs_bord_trail as u8;
				let step = (abs_bord_trail + (n as u32 >> 1)) / n as u32;
				for i in 0..n - 1 {
					d.t_env[i + 1] = d.t_env[i] + step as u8;
				}
				d.bs_freq_res[1] = bs.bits(1) as u8;
				for i in 1..n {
					d.bs_freq_res[i + 1] = d.bs_freq_res[1];
				}
			}
			FIXVAR => {
				abs_bord_trail += bs.bits(2);
				let num_rel_trail = bs.bits(2) as usize;
				d.bs_num_env = num_rel_trail + 1;
				d.t_env[0] = 0;
				d.t_env[d.bs_num_env] = abs_bord_trail as u8;
				for i in (1..=num_rel_trail).rev() {
					let v = i32::from(d.t_env[i + 1]) - 2 * bs.bits(2) as i32 - 2;
					d.t_env[i] = v as u8;
				}
				bs_pointer = bs.bits(CEIL_LOG2[d.bs_num_env]);
				for i in 0..d.bs_num_env {
					d.bs_freq_res[d.bs_num_env - i] = bs.bits(1) as u8;
				}
			}
			VARFIX => {
				d.t_env[0] = bs.bits(2) as u8;
				let num_rel_lead = bs.bits(2) as usize;
				d.bs_num_env = num_rel_lead + 1;
				d.t_env[d.bs_num_env] = abs_bord_trail as u8;
				for i in 0..num_rel_lead {
					d.t_env[i + 1] = d.t_env[i] + 2 * bs.bits(2) as u8 + 2;
				}
				bs_pointer = bs.bits(CEIL_LOG2[d.bs_num_env]);
				for i in 0..d.bs_num_env {
					d.bs_freq_res[i + 1] = bs.bits(1) as u8;
				}
			}
			_ => {
				d.t_env[0] = bs.bits(2) as u8;
				abs_bord_trail += bs.bits(2);
				let num_rel_lead = bs.bits(2) as usize;
				let num_rel_trail = bs.bits(2) as usize;
				let n = num_rel_lead + num_rel_trail + 1;
				if n > 5 {
					return None;
				}
				d.bs_num_env = n;
				d.t_env[n] = abs_bord_trail as u8;
				for i in 0..num_rel_lead {
					d.t_env[i + 1] = d.t_env[i] + 2 * bs.bits(2) as u8 + 2;
				}
				for i in (n - num_rel_trail..n).rev() {
					let v = i32::from(d.t_env[i + 1]) - 2 * bs.bits(2) as i32 - 2;
					d.t_env[i] = v as u8;
				}
				bs_pointer = bs.bits(CEIL_LOG2[n]);
				for i in 0..n {
					d.bs_freq_res[i + 1] = bs.bits(1) as u8;
				}
			}
		}
		d.bs_frame_class = frame_class;
		if bs_pointer as usize > d.bs_num_env + 1 {
			return None;
		}
		for i in 1..=d.bs_num_env {
			if d.t_env[i - 1] >= d.t_env[i] {
				return None;
			}
		}
		d.bs_num_noise = usize::from(d.bs_num_env > 1) + 1;
		d.t_q[0] = d.t_env[0];
		d.t_q[d.bs_num_noise] = d.t_env[d.bs_num_env];
		if d.bs_num_noise > 1 {
			let n = d.bs_num_env as i32;
			let p = bs_pointer as i32;
			let idx = if d.bs_frame_class == FIXFIX {
				n >> 1
			} else if d.bs_frame_class & 1 != 0 {
				n - (p - 1).max(1)
			} else if p == 0 {
				1
			} else if p == 1 {
				n - 1
			} else {
				p - 1
			};
			d.t_q[1] = d.t_env[idx.clamp(0, 8) as usize];
		}
		d.e_a[0] = -i32::from(d.e_a[1] != old_num_env as i32);
		d.e_a[1] = -1;
		if d.bs_frame_class & 1 != 0 && bs_pointer != 0 {
			d.e_a[1] = d.bs_num_env as i32 + 1 - bs_pointer as i32;
		} else if d.bs_frame_class == 2 && bs_pointer > 1 {
			d.e_a[1] = bs_pointer as i32 - 1;
		}
		Some(())
	}

	fn copy_grid(&mut self) {
		let [src, dst] = &mut self.data;
		dst.bs_freq_res[0] = dst.bs_freq_res[dst.bs_num_env];
		dst.t_env_num_env_old = dst.t_env[dst.bs_num_env];
		dst.e_a[0] = -i32::from(dst.e_a[1] != dst.bs_num_env as i32);
		dst.bs_freq_res[1..].copy_from_slice(&src.bs_freq_res[1..]);
		dst.t_env = src.t_env;
		dst.t_q = src.t_q;
		dst.bs_num_env = src.bs_num_env;
		dst.bs_amp_res = src.bs_amp_res;
		dst.bs_num_noise = src.bs_num_noise;
		dst.bs_frame_class = src.bs_frame_class;
		dst.e_a[1] = src.e_a[1];
	}

	fn read_dtdf(bs: &mut BitReader, d: &mut SbrChannel) {
		for i in 0..d.bs_num_env {
			d.bs_df_env[i] = bs.bits(1) as u8;
		}
		for i in 0..d.bs_num_noise {
			d.bs_df_noise[i] = bs.bits(1) as u8;
		}
	}

	fn read_invf(bs: &mut BitReader, d: &mut SbrChannel, n_q: usize) {
		d.bs_invf_mode[1] = d.bs_invf_mode[0];
		for i in 0..n_q {
			d.bs_invf_mode[0][i] = bs.bits(2) as u8;
		}
	}

	fn read_envelope(&mut self, bs: &mut BitReader, ch: usize) -> Option<()> {
		let sh = shared();
		let coupled = self.bs_coupling && ch == 1;
		let delta = if coupled { 2 } else { 1 };
		let odd = (self.n[1] & 1) as usize;
		let n = self.n;
		let d = &mut self.data[ch];
		let (bits, t, f) = match (coupled, d.bs_amp_res) {
			(true, true) => (5, T_ENV_BAL_3_0DB, F_ENV_BAL_3_0DB),
			(true, false) => (6, T_ENV_BAL_1_5DB, F_ENV_BAL_1_5DB),
			(false, true) => (6, T_ENV_3_0DB, F_ENV_3_0DB),
			(false, false) => (7, T_ENV_1_5DB, F_ENV_1_5DB),
		};
		let (t, f) = (&sh.vlc[t], &sh.vlc[f]);
		for i in 0..d.bs_num_env {
			let res = usize::from(d.bs_freq_res[i + 1]);
			let count = n[res] as usize;
			if d.bs_df_env[i] != 0 {
				for j in 0..count {
					let k = if d.bs_freq_res[i + 1] == d.bs_freq_res[i] {
						j
					} else if res != 0 {
						(j + odd) >> 1
					} else if j == 0 {
						0
					} else {
						2 * j - odd
					};
					let v = i32::from(d.env_facs_q[i][k.min(47)]) + delta * read_vlc(bs, t);
					if !(0..=127).contains(&v) {
						return None;
					}
					d.env_facs_q[i + 1][j] = v as u8;
				}
			} else {
				d.env_facs_q[i + 1][0] = (delta * bs.bits(bits) as i32) as u8;
				for j in 1..count {
					let v = i32::from(d.env_facs_q[i + 1][j - 1]) + delta * read_vlc(bs, f);
					if !(0..=127).contains(&v) {
						return None;
					}
					d.env_facs_q[i + 1][j] = v as u8;
				}
			}
		}
		d.env_facs_q[0] = d.env_facs_q[d.bs_num_env];
		Some(())
	}

	fn read_noise(&mut self, bs: &mut BitReader, ch: usize) -> Option<()> {
		let sh = shared();
		let coupled = self.bs_coupling && ch == 1;
		let delta = if coupled { 2 } else { 1 };
		let (t, f) = if coupled {
			(&sh.vlc[T_NOISE_BAL_3_0DB], &sh.vlc[F_ENV_BAL_3_0DB])
		} else {
			(&sh.vlc[T_NOISE_3_0DB], &sh.vlc[F_ENV_3_0DB])
		};
		let n_q = self.n_q as usize;
		let d = &mut self.data[ch];
		for i in 0..d.bs_num_noise {
			if d.bs_df_noise[i] != 0 {
				for j in 0..n_q {
					let v = i32::from(d.noise_facs_q[i][j]) + delta * read_vlc(bs, t);
					if !(0..=30).contains(&v) {
						return None;
					}
					d.noise_facs_q[i + 1][j] = v as u8;
				}
			} else {
				d.noise_facs_q[i + 1][0] = (delta * bs.bits(5) as i32) as u8;
				for j in 1..n_q {
					let v = i32::from(d.noise_facs_q[i + 1][j - 1]) + delta * read_vlc(bs, f);
					if !(0..=30).contains(&v) {
						return None;
					}
					d.noise_facs_q[i + 1][j] = v as u8;
				}
			}
		}
		d.noise_facs_q[0] = d.noise_facs_q[d.bs_num_noise];
		Some(())
	}

	fn read_harmonics(&mut self, bs: &mut BitReader, ch: usize) {
		let n1 = self.n[1] as usize;
		let d = &mut self.data[ch];
		d.bs_add_harmonic_flag = bs.bit();
		if d.bs_add_harmonic_flag {
			for v in &mut d.bs_add_harmonic[..n1] {
				*v = bs.bits(1) as u8;
			}
		}
	}

	fn read_sce(&mut self, bs: &mut BitReader) -> Option<()> {
		if bs.bit() {
			bs.skip(4);
		}
		self.read_grid(bs, 0)?;
		Self::read_dtdf(bs, &mut self.data[0]);
		Self::read_invf(bs, &mut self.data[0], self.n_q as usize);
		self.read_envelope(bs, 0)?;
		self.read_noise(bs, 0)?;
		self.read_harmonics(bs, 0);
		Some(())
	}

	fn read_cpe(&mut self, bs: &mut BitReader) -> Option<()> {
		if bs.bit() {
			bs.skip(8);
		}
		let nq = self.n_q as usize;
		self.bs_coupling = bs.bit();
		if self.bs_coupling {
			self.read_grid(bs, 0)?;
			self.copy_grid();
			Self::read_dtdf(bs, &mut self.data[0]);
			Self::read_dtdf(bs, &mut self.data[1]);
			Self::read_invf(bs, &mut self.data[0], nq);
			self.data[1].bs_invf_mode[1] = self.data[1].bs_invf_mode[0];
			self.data[1].bs_invf_mode[0] = self.data[0].bs_invf_mode[0];
			self.read_envelope(bs, 0)?;
			self.read_noise(bs, 0)?;
			self.read_envelope(bs, 1)?;
			self.read_noise(bs, 1)?;
		} else {
			self.read_grid(bs, 0)?;
			self.read_grid(bs, 1)?;
			Self::read_dtdf(bs, &mut self.data[0]);
			Self::read_dtdf(bs, &mut self.data[1]);
			Self::read_invf(bs, &mut self.data[0], nq);
			Self::read_invf(bs, &mut self.data[1], nq);
			self.read_envelope(bs, 0)?;
			self.read_envelope(bs, 1)?;
			self.read_noise(bs, 0)?;
			self.read_noise(bs, 1)?;
		}
		self.read_harmonics(bs, 0);
		self.read_harmonics(bs, 1);
		Some(())
	}

	fn read_data(&mut self, bs: &mut BitReader, is_cpe: bool, ps_allowed: bool) -> bool {
		let start = bs.pos();
		self.is_cpe = is_cpe;
		self.ready_for_dequant = true;
		let ok = if is_cpe { self.read_cpe(bs) } else { self.read_sce(bs) };
		if ok.is_none() {
			self.turnoff();
			return false;
		}
		let mut found_ps = false;
		if bs.bit() {
			let mut left = bs.bits(4) as i64;
			if left == 15 {
				left += i64::from(bs.bits(8));
			}
			left <<= 3;
			while left > 7 {
				left -= 2;
				let id = bs.bits(2);
				if id == EXTENSION_ID_PS && ps_allowed {
					let used = self.ps.read_data(bs, left as usize);
					left -= used as i64;
					found_ps = true;
				} else {
					bs.skip(left as usize);
					left = 0;
				}
			}
			if left > 0 {
				bs.skip(left as usize);
			}
		}
		let _ = start;
		found_ps
	}

	/// Parses an `sbr_extension_data()` from a FIL element of `cnt` bytes
	/// (the 4-bit extension type already read). Returns `true` when a PS
	/// extension was present.
	pub fn decode_extension(
		&mut self,
		bs: &mut BitReader,
		crc: bool,
		cnt: usize,
		is_cpe: bool,
		core_rate: u32,
		ps_allowed: bool,
	) -> bool {
		let end = bs.pos() + cnt * 8 - 4;
		let mut gb = bs.clone();
		self.reset = false;
		if self.sample_rate == 0 {
			self.sample_rate = 2 * core_rate;
		}
		if crc {
			gb.skip(10);
		}
		self.kx[0] = self.kx[1];
		self.m[0] = self.m[1];
		self.kx_and_m_pushed = true;
		if gb.bit() {
			self.read_header(&mut gb);
		}
		if self.reset {
			self.sbr_reset();
		}
		let mut ps = false;
		if self.start {
			ps = self.read_data(&mut gb, is_cpe, ps_allowed);
		}
		if gb.pos() > end {
			self.turnoff();
		}
		bs.skip(end - bs.pos());
		ps
	}

	// ---------------------------------------------------------------
	// Signal processing
	// ---------------------------------------------------------------

	fn dequant(&mut self) {
		const EXP2_TAB: [f32; 2] = [1.0, std::f32::consts::SQRT_2];
		if self.is_cpe && self.bs_coupling {
			let n = self.n;
			let n_q = self.n_q as usize;
			let [d0, d1] = &mut self.data;
			let pan_offset = if d0.bs_amp_res { 12 } else { 24 };
			for e in 1..=d0.bs_num_env {
				for k in 0..n[usize::from(d0.bs_freq_res[e])] as usize {
					let q0 = i32::from(d0.env_facs_q[e][k]);
					let q1 = i32::from(d1.env_facs_q[e][k]);
					let (mut t1, t2) = if d0.bs_amp_res {
						(exp2fi(q0 + 7), exp2fi(pan_offset - q1))
					} else {
						(
							exp2fi((q0 >> 1) + 7) * EXP2_TAB[(q0 & 1) as usize],
							exp2fi((pan_offset - q1) >> 1) * EXP2_TAB[((pan_offset - q1) & 1) as usize],
						)
					};
					if t1 > 1e20 {
						t1 = 1.0;
					}
					let fac = t1 / (1.0 + t2);
					d0.env_facs[e][k] = fac;
					d1.env_facs[e][k] = fac * t2;
				}
			}
			for e in 1..=d0.bs_num_noise {
				for k in 0..n_q {
					let t1 = exp2fi(NOISE_FLOOR_OFFSET - i32::from(d0.noise_facs_q[e][k]) + 1);
					let t2 = exp2fi(12 - i32::from(d1.noise_facs_q[e][k]));
					let fac = t1 / (1.0 + t2);
					d0.noise_facs[e][k] = fac;
					d1.noise_facs[e][k] = fac * t2;
				}
			}
		} else {
			let n = self.n;
			let n_q = self.n_q as usize;
			for ch in 0..if self.is_cpe { 2 } else { 1 } {
				let d = &mut self.data[ch];
				for e in 1..=d.bs_num_env {
					for k in 0..n[usize::from(d.bs_freq_res[e])] as usize {
						let q = i32::from(d.env_facs_q[e][k]);
						let mut v = if d.bs_amp_res {
							exp2fi(q + 6)
						} else {
							exp2fi((q >> 1) + 6) * EXP2_TAB[(q & 1) as usize]
						};
						if v > 1e20 {
							v = 1.0;
						}
						d.env_facs[e][k] = v;
					}
				}
				for e in 1..=d.bs_num_noise {
					for k in 0..n_q {
						d.noise_facs[e][k] = exp2fi(NOISE_FLOOR_OFFSET - i32::from(d.noise_facs_q[e][k]));
					}
				}
			}
		}
	}

	fn qmf_analysis(&mut self, input: &[f32], ch: usize) {
		let sh = shared();
		let d = &mut self.data[ch];
		let buf_idx = d.ypos;
		let x = &mut d.analysis;
		let nb = TIME_SLOTS * 64;
		x.copy_within(nb..nb + 288, 0);
		x[288..288 + nb].copy_from_slice(&input[..nb]);
		let z = &mut self.scratch;
		for i in 0..QMF_SLOTS {
			let xs = &x[32 * i..32 * i + 320];
			for (k, zv) in z.iter_mut().enumerate() {
				*zv = QMF_WINDOW_DS[k] * xs[319 - k];
			}
			for k in 0..64 {
				z[k] = z[k] + z[k + 64] + z[k + 128] + z[k + 192] + z[k + 256];
			}
			// qmf_pre_shuffle: z[64..128] from z[0..64].
			let mut pre = [0f32; 64];
			pre[0] = z[0];
			pre[1] = z[1];
			let mut k = 1;
			while k < 31 {
				pre[2 * k] = -z[64 - k];
				pre[2 * k + 1] = z[k + 1];
				pre[2 * k + 2] = -z[63 - k];
				pre[2 * k + 3] = z[k + 2];
				k += 2;
			}
			pre[62] = -z[64 - 31];
			pre[63] = z[32];
			let mut out = [0f32; 64];
			sh.imdct.inverse_half(&pre, &mut out, &mut self.work, &mut self.cbuf);
			for v in &mut out {
				*v *= sh.analysis_scale;
			}
			// qmf_post_shuffle.
			let w = &mut d.w[buf_idx][i];
			let mut k = 0;
			while k < 32 {
				w[k] = [-out[63 - k], out[k]];
				w[k + 1] = [-out[62 - k], out[k + 1]];
				k += 2;
			}
		}
	}

	fn lf_gen(&mut self, ch: usize) {
		let d = &self.data[ch];
		let buf_idx = d.ypos;
		for row in self.x_low.iter_mut() {
			*row = [[0.0; 2]; 40];
		}
		for k in 0..self.kx[1].clamp(0, 32) as usize {
			for i in 8..40 {
				self.x_low[k][i] = d.w[buf_idx][i - 8][k];
			}
		}
		for k in 0..self.kx[0].clamp(0, 32) as usize {
			for i in 0..8 {
				self.x_low[k][i] = d.w[1 - buf_idx][i + QMF_SLOTS - 8][k];
			}
		}
	}

	fn hf_inverse_filter(&mut self) {
		for k in 0..self.k[0].clamp(0, 32) as usize {
			let x = &self.x_low[k];
			let phi = autocorrelate(x);
			let dk = phi[2][1][0] * phi[1][0][0]
				- (phi[1][1][0] * phi[1][1][0] + phi[1][1][1] * phi[1][1][1]) / 1.000001;
			let a1 = if dk == 0.0 {
				[0.0, 0.0]
			} else {
				let re =
					phi[0][0][0] * phi[1][1][0] - phi[0][0][1] * phi[1][1][1] - phi[0][1][0] * phi[1][0][0];
				let im =
					phi[0][0][0] * phi[1][1][1] + phi[0][0][1] * phi[1][1][0] - phi[0][1][1] * phi[1][0][0];
				[re / dk, im / dk]
			};
			let a0 = if phi[1][0][0] == 0.0 {
				[0.0, 0.0]
			} else {
				let re = phi[0][0][0] + a1[0] * phi[1][1][0] + a1[1] * phi[1][1][1];
				let im = phi[0][0][1] + a1[1] * phi[1][1][0] - a1[0] * phi[1][1][1];
				[-re / phi[1][0][0], -im / phi[1][0][0]]
			};
			if a1[0] * a1[0] + a1[1] * a1[1] >= 16.0 || a0[0] * a0[0] + a0[1] * a0[1] >= 16.0 {
				self.alpha1[k] = [0.0; 2];
				self.alpha0[k] = [0.0; 2];
			} else {
				self.alpha1[k] = a1;
				self.alpha0[k] = a0;
			}
		}
	}

	fn chirp(&mut self, ch: usize) {
		const BW_TAB: [f32; 4] = [0.0, 0.75, 0.9, 0.98];
		let d = &mut self.data[ch];
		for i in 0..self.n_q as usize {
			let mut new_bw = if d.bs_invf_mode[0][i] + d.bs_invf_mode[1][i] == 1 {
				0.6
			} else {
				BW_TAB[usize::from(d.bs_invf_mode[0][i] & 3)]
			};
			if new_bw < d.bw_array[i] {
				new_bw = 0.75 * new_bw + 0.25 * d.bw_array[i];
			} else {
				new_bw = 0.90625 * new_bw + 0.09375 * d.bw_array[i];
			}
			d.bw_array[i] = if new_bw < 0.015625 { 0.0 } else { new_bw };
		}
	}

	fn hf_gen(&mut self, ch: usize) -> Option<()> {
		let d = &self.data[ch];
		let (start, end) = (2 * usize::from(d.t_env[0]), 2 * usize::from(d.t_env[d.bs_num_env]));
		let mut k = self.kx[1] as usize;
		let mut g: i32 = 0;
		for j in 0..self.num_patches {
			for x in 0..usize::from(self.patch_num_subbands[j]) {
				let p = usize::from(self.patch_start_subband[j]) + x;
				while g <= self.n_q && k >= usize::from(self.f_tablenoise[g as usize]) {
					g += 1;
				}
				g -= 1;
				if g < 0 || k >= 64 || p >= 32 {
					return None;
				}
				let bw = d.bw_array[g as usize];
				let (a0, a1) = (self.alpha0[p], self.alpha1[p]);
				let alpha = [a1[0] * bw * bw, a1[1] * bw * bw, a0[0] * bw, a0[1] * bw];
				let low = &self.x_low[p];
				let high = &mut self.x_high[k];
				for i in start..end.min(38) {
					let (l0, l1, l2) = (low[i], low[i + 1], low[i + 2]);
					high[i + 2][0] =
						l0[0] * alpha[0] - l0[1] * alpha[1] + l1[0] * alpha[2] - l1[1] * alpha[3] + l2[0];
					high[i + 2][1] =
						l0[1] * alpha[0] + l0[0] * alpha[1] + l1[1] * alpha[2] + l1[0] * alpha[3] + l2[1];
				}
				k += 1;
			}
		}
		let top = (self.m[1] + self.kx[1]).clamp(0, 64) as usize;
		for row in self.x_high[k.min(64)..top.max(k.min(64))].iter_mut() {
			*row = [[0.0; 2]; 40];
		}
		Some(())
	}

	fn mapping(&mut self, ch: usize) -> Option<()> {
		let kx1 = self.kx[1] as usize;
		let d = &mut self.data[ch];
		for row in &mut d.s_indexmapped[1..8] {
			*row = [0; 48];
		}
		let e_a = d.e_a;
		for e in 0..d.bs_num_env {
			let res = usize::from(d.bs_freq_res[e + 1]);
			let ilim = self.n[res] as usize;
			let table: &[u16] = if res != 0 { &self.f_tablehigh } else { &self.f_tablelow };
			if usize::from(table[0]) != kx1 {
				return None;
			}
			for i in 0..ilim {
				for m in usize::from(table[i])..usize::from(table[i + 1]) {
					self.e_origmapped[e][m - kx1] = d.env_facs[e + 1][i];
				}
			}
			let k = usize::from(d.bs_num_noise > 1 && d.t_env[e] >= d.t_q[1]);
			for i in 0..self.n_q as usize {
				for m in usize::from(self.f_tablenoise[i])..usize::from(self.f_tablenoise[i + 1]) {
					self.q_mapped[e][m - kx1] = d.noise_facs[k + 1][i];
				}
			}
			for i in 0..self.n[1] as usize {
				if d.bs_add_harmonic_flag {
					let mid = (usize::from(self.f_tablehigh[i]) + usize::from(self.f_tablehigh[i + 1])) >> 1;
					let on = e as i32 >= e_a[1] || d.s_indexmapped[0][mid - kx1] == 1;
					d.s_indexmapped[e + 1][mid - kx1] = d.bs_add_harmonic[i] * u8::from(on);
				}
			}
			for i in 0..ilim {
				let (lo, hi) = (usize::from(table[i]) - kx1, usize::from(table[i + 1]) - kx1);
				let present = d.s_indexmapped[e + 1][lo..hi].iter().any(|&v| v != 0);
				self.s_mapped[e][lo..hi].fill(u8::from(present));
			}
		}
		d.s_indexmapped[0] = d.s_indexmapped[d.bs_num_env];
		Some(())
	}

	fn env_estimate(&mut self, ch: usize) {
		let d = &self.data[ch];
		let kx1 = self.kx[1] as usize;
		if self.bs_interpol_freq {
			for e in 0..d.bs_num_env {
				let recip = 0.5 / f32::from(d.t_env[e + 1] - d.t_env[e]);
				let ilb = usize::from(d.t_env[e]) * 2 + ENVELOPE_ADJUSTMENT_OFFSET;
				let iub = usize::from(d.t_env[e + 1]) * 2 + ENVELOPE_ADJUSTMENT_OFFSET;
				if ilb >= 40 {
					return;
				}
				for m in 0..self.m[1] as usize {
					let sum = sum_square(&self.x_high[m + kx1][ilb..iub.min(40)]);
					self.e_curr[e][m] = sum * recip;
				}
			}
		} else {
			for e in 0..d.bs_num_env {
				let env_size = 2 * i32::from(d.t_env[e + 1] - d.t_env[e]);
				let ilb = usize::from(d.t_env[e]) * 2 + ENVELOPE_ADJUSTMENT_OFFSET;
				let iub = usize::from(d.t_env[e + 1]) * 2 + ENVELOPE_ADJUSTMENT_OFFSET;
				if ilb >= 40 {
					return;
				}
				let res = usize::from(d.bs_freq_res[e + 1]);
				let table: &[u16] = if res != 0 { &self.f_tablehigh } else { &self.f_tablelow };
				for p in 0..self.n[res] as usize {
					let (lo, hi) = (usize::from(table[p]), usize::from(table[p + 1]));
					let den = env_size * (hi - lo) as i32;
					let mut sum = 0.0f32;
					for k in lo..hi {
						sum += sum_square(&self.x_high[k][ilb..iub.min(40)]);
					}
					sum /= den as f32;
					for k in lo..hi {
						self.e_curr[e][k - kx1] = sum;
					}
				}
			}
		}
	}

	fn gain_calc(&mut self, ch: usize) {
		const LIMGAIN: [f32; 4] = [0.70795, 1.0, 1.41254, 10_000_000_000.0];
		let d = &self.data[ch];
		let e_a = d.e_a;
		let kx1 = self.kx[1] as usize;
		for e in 0..d.bs_num_env {
			let delta = !(e as i32 == e_a[1] || e as i32 == e_a[0]);
			let deltaf = if delta { 1.0 } else { 0.0 };
			for k in 0..self.n_lim as usize {
				let lo = usize::from(self.f_tablelim[k]) - kx1;
				let hi = usize::from(self.f_tablelim[k + 1]) - kx1;
				for m in lo..hi {
					let temp = self.e_origmapped[e][m] / (1.0 + self.q_mapped[e][m]);
					self.q_m[e][m] = (temp * self.q_mapped[e][m]).sqrt();
					self.s_m[e][m] = (temp * f32::from(d.s_indexmapped[e + 1][m])).sqrt();
					self.gain[e][m] = if self.s_mapped[e][m] == 0 {
						(self.e_origmapped[e][m]
							/ ((1.0 + self.e_curr[e][m]) * (1.0 + self.q_mapped[e][m] * deltaf)))
							.sqrt()
					} else {
						(self.e_origmapped[e][m] * self.q_mapped[e][m]
							/ ((1.0 + self.e_curr[e][m]) * (1.0 + self.q_mapped[e][m])))
							.sqrt()
					};
					self.gain[e][m] += f32::MIN_POSITIVE;
				}
				let (mut s0, mut s1) = (0.0f32, 0.0f32);
				for m in lo..hi {
					s0 += self.e_origmapped[e][m];
					s1 += self.e_curr[e][m];
				}
				let mut gain_max = LIMGAIN[self.bs_limiter_gains as usize & 3]
					* ((f32::EPSILON + s0) / (f32::EPSILON + s1)).sqrt();
				gain_max = gain_max.min(100_000.0);
				for m in lo..hi {
					let q_m_max = self.q_m[e][m] * gain_max / self.gain[e][m];
					self.q_m[e][m] = self.q_m[e][m].min(q_m_max);
					self.gain[e][m] = self.gain[e][m].min(gain_max);
				}
				let (mut s0, mut s1) = (0.0f32, 0.0f32);
				for m in lo..hi {
					s0 += self.e_origmapped[e][m];
					let noise = if delta && self.s_m[e][m] == 0.0 { 1.0 } else { 0.0 };
					s1 += self.e_curr[e][m] * self.gain[e][m] * self.gain[e][m]
						+ self.s_m[e][m] * self.s_m[e][m]
						+ noise * self.q_m[e][m] * self.q_m[e][m];
				}
				let boost = ((f32::EPSILON + s0) / (f32::EPSILON + s1)).sqrt().min(1.584_893_2);
				for m in lo..hi {
					self.gain[e][m] *= boost;
					self.q_m[e][m] *= boost;
					self.s_m[e][m] *= boost;
				}
			}
		}
	}

	fn hf_assemble(&mut self, ch: usize) {
		const H_SMOOTH: [f32; 5] = [0.333_333_33, 0.301_502_83, 0.218_169_5, 0.115_163_83, 0.031_830_5];
		let h_sl = if self.bs_smoothing_mode { 0 } else { 4 };
		let kx = self.kx[1] as usize;
		let m_max = self.m[1] as usize;
		let reset = self.reset;
		let d = &mut self.data[ch];
		let e_a = d.e_a;
		let mut indexnoise = d.f_indexnoise;
		let mut indexsine = d.f_indexsine;
		let t0 = 2 * usize::from(d.t_env[0]);
		if reset {
			for i in 0..h_sl {
				d.g_temp[i + t0][..m_max].copy_from_slice(&self.gain[0][..m_max]);
				d.q_temp[i + t0][..m_max].copy_from_slice(&self.q_m[0][..m_max]);
			}
		} else if h_sl != 0 {
			let old = 2 * usize::from(d.t_env_num_env_old);
			for i in 0..4 {
				d.g_temp[i + t0] = d.g_temp[i + old];
				d.q_temp[i + t0] = d.q_temp[i + old];
			}
		}
		for e in 0..d.bs_num_env {
			for i in 2 * usize::from(d.t_env[e])..2 * usize::from(d.t_env[e + 1]) {
				d.g_temp[h_sl + i][..m_max].copy_from_slice(&self.gain[e][..m_max]);
				d.q_temp[h_sl + i][..m_max].copy_from_slice(&self.q_m[e][..m_max]);
			}
		}
		let y1 = &mut d.y[d.ypos];
		for e in 0..d.bs_num_env {
			let is_ea = e as i32 == e_a[0] || e as i32 == e_a[1];
			for i in 2 * usize::from(d.t_env[e])..2 * usize::from(d.t_env[e + 1]) {
				let mut g_tab = [0f32; 48];
				let mut q_tab = [0f32; 48];
				let (g_filt, q_filt): (&[f32; 48], &[f32; 48]) = if h_sl != 0 && !is_ea {
					let idx1 = i + h_sl;
					for m in 0..m_max {
						let (mut g, mut q) = (0.0f32, 0.0f32);
						for j in 0..=h_sl {
							g += d.g_temp[idx1 - j][m] * H_SMOOTH[j];
							q += d.q_temp[idx1 - j][m] * H_SMOOTH[j];
						}
						g_tab[m] = g;
						q_tab[m] = q;
					}
					(&g_tab, &q_tab)
				} else {
					(&d.g_temp[i + h_sl], &d.q_temp[i])
				};
				let row = &mut y1[i];
				for m in 0..m_max {
					let xh = self.x_high[kx + m][i + ENVELOPE_ADJUSTMENT_OFFSET];
					row[kx + m] = [xh[0] * g_filt[m], xh[1] * g_filt[m]];
				}
				let s_m = &self.s_m[e];
				if !is_ea {
					let phi = 1.0 - 2.0 * (kx & 1) as f32;
					let (ps0, mut ps1) = match indexsine {
						0 => (1.0, 0.0),
						1 => (0.0, phi),
						2 => (-1.0, 0.0),
						_ => (0.0, -phi),
					};
					let mut noise = indexnoise;
					for m in 0..m_max {
						let y = &mut row[kx + m];
						noise = (noise + 1) & 0x1ff;
						if s_m[m] != 0.0 {
							y[0] += s_m[m] * ps0;
							y[1] += s_m[m] * ps1;
						} else {
							let n = NOISE_TABLE[noise as usize];
							y[0] += q_filt[m] * n[0];
							y[1] += q_filt[m] * n[1];
						}
						ps1 = -ps1;
					}
				} else {
					let idx = (indexsine & 1) as usize;
					let a: f32 = (1 - ((indexsine as i32 + (kx & 1) as i32) & 2)) as f32;
					let b: f32 = if idx == 1 { -a } else { a };
					let mut m = 0;
					while m + 1 < m_max {
						row[kx + m][idx] += s_m[m] * a;
						row[kx + m + 1][idx] += s_m[m + 1] * b;
						m += 2;
					}
					if m_max & 1 != 0 {
						row[kx + m][idx] += s_m[m] * a;
					}
				}
				indexnoise = (indexnoise + m_max as u32) & 0x1ff;
				indexsine = (indexsine + 1) & 3;
			}
		}
		d.f_indexnoise = indexnoise;
		d.f_indexsine = indexsine;
	}

	fn x_gen(&mut self, ch: usize) {
		let d = &self.data[ch];
		let i_f = QMF_SLOTS;
		let i_temp = (2 * i32::from(d.t_env_num_env_old) - i_f as i32).max(0) as usize;
		let x = &mut self.x[ch];
		*x = [[[0.0; 64]; 38]; 2];
		let y0 = &d.y[1 - d.ypos];
		let y1 = &d.y[d.ypos];
		let (kx0, m0) = (self.kx[0].clamp(0, 64) as usize, self.m[0].max(0) as usize);
		let (kx1, m1) = (self.kx[1].clamp(0, 64) as usize, self.m[1].max(0) as usize);
		for k in 0..kx0.min(32) {
			for i in 0..i_temp.min(38) {
				let v = self.x_low[k][i + ENVELOPE_ADJUSTMENT_OFFSET];
				x[0][i][k] = v[0];
				x[1][i][k] = v[1];
			}
		}
		// i_temp <= 6, so i + i_f stays inside the 38 slots of Y.
		for k in kx0..(kx0 + m0).min(64) {
			for i in 0..i_temp.min(38 - i_f) {
				x[0][i][k] = y0[i + i_f][k][0];
				x[1][i][k] = y0[i + i_f][k][1];
			}
		}
		for k in 0..kx1.min(32) {
			for i in i_temp..38 {
				let v = self.x_low[k][i + ENVELOPE_ADJUSTMENT_OFFSET];
				x[0][i][k] = v[0];
				x[1][i][k] = v[1];
			}
		}
		for k in kx1..(kx1 + m1).min(64) {
			for i in i_temp..i_f {
				x[0][i][k] = y1[i][k][0];
				x[1][i][k] = y1[i][k][1];
			}
		}
	}

	fn qmf_synthesis(&mut self, ch: usize, out: &mut [f32], downsampled: bool) {
		let sh = shared();
		let div = usize::from(downsampled);
		let window: &[f32] = if downsampled { &QMF_WINDOW_DS } else { &QMF_WINDOW_US };
		let step = 128 >> div;
		let x = &mut self.x[ch];
		let d = &mut self.data[ch];
		let mut out_at = 0;
		for i in 0..QMF_SLOTS {
			if d.synthesis_offset < step {
				let saved = (1280 - 128) >> div;
				d.synthesis.copy_within(0..saved, SYNTHESIS_BUF_SIZE - saved);
				d.synthesis_offset = SYNTHESIS_BUF_SIZE - saved - step;
			} else {
				d.synthesis_offset -= step;
			}
			let v_off = d.synthesis_offset;
			if downsampled {
				let mut inp = [0f32; 64];
				for n in 0..32 {
					inp[n] = -x[0][i][n];
					inp[32 + n] = x[1][i][31 - n];
				}
				sh.imdct.inverse_half(&inp, &mut self.mdct_buf[0], &mut self.work, &mut self.cbuf);
				let src = &self.mdct_buf[0];
				let v = &mut d.synthesis[v_off..];
				for k in 0..32 {
					v[k] = src[63 - 2 * k] * sh.synthesis_scale;
					v[63 - k] = -src[63 - 2 * k - 1] * sh.synthesis_scale;
				}
			} else {
				let mut x1 = x[1][i];
				let mut k = 1;
				while k < 64 {
					x1[k] = -x1[k];
					x1[k + 2] = -x1[k + 2];
					k += 4;
				}
				let (b0, b1) = self.mdct_buf.split_at_mut(1);
				sh.imdct.inverse_half(&x[0][i], &mut b0[0], &mut self.work, &mut self.cbuf);
				sh.imdct.inverse_half(&x1, &mut b1[0], &mut self.work, &mut self.cbuf);
				let (s0, s1) = (&self.mdct_buf[1], &self.mdct_buf[0]);
				let v = &mut d.synthesis[v_off..];
				let sc = sh.synthesis_scale;
				for k in 0..64 {
					v[k] = (s0[k] - s1[63 - k]) * sc;
					v[127 - k] = (s0[k] + s1[63 - k]) * sc;
				}
			}
			let n = 64 >> div;
			let v = &d.synthesis[v_off..];
			let o = &mut out[out_at..out_at + n];
			const TAPS: [usize; 10] = [0, 192, 256, 448, 512, 704, 768, 960, 1024, 1216];
			for (t, &vo) in TAPS.iter().enumerate() {
				let vs = &v[vo >> div..(vo >> div) + n];
				let ws = &window[(t * 64) >> div..((t * 64) >> div) + n];
				if t == 0 {
					for j in 0..n {
						o[j] = vs[j] * ws[j];
					}
				} else {
					for j in 0..n {
						o[j] += vs[j] * ws[j];
					}
				}
			}
			out_at += n;
		}
	}

	/// Runs SBR (and PS when `ps` is set) on one element. `left`/`right`
	/// hold the 1024 core samples; the output (2048 samples per channel, or
	/// 1024 when `downsampled`) goes to `out_l`/`out_r`.
	#[allow(clippy::too_many_arguments)]
	pub fn apply(
		&mut self,
		is_cpe: bool,
		left: &[f32],
		right: Option<&[f32]>,
		out_l: &mut [f32],
		out_r: Option<&mut [f32]>,
		ps: bool,
		downsampled: bool,
	) {
		if is_cpe != self.is_cpe {
			self.turnoff();
		}
		if self.start && !self.ready_for_dequant {
			self.turnoff();
		}
		if !self.kx_and_m_pushed {
			self.kx[0] = self.kx[1];
			self.m[0] = self.m[1];
		} else {
			self.kx_and_m_pushed = false;
		}
		if self.start {
			self.dequant();
			self.ready_for_dequant = false;
		}
		let nch = if is_cpe { 2 } else { 1 };
		for ch in 0..nch {
			let input = if ch == 0 { left } else { right.unwrap_or(left) };
			self.qmf_analysis(input, ch);
			self.lf_gen(ch);
			self.data[ch].ypos ^= 1;
			if self.start {
				self.hf_inverse_filter();
				self.chirp(ch);
				if self.data[ch].bs_num_env > 0 && self.hf_gen(ch).is_some() && self.mapping(ch).is_some() {
					self.env_estimate(ch);
					self.gain_calc(ch);
					self.hf_assemble(ch);
				}
			}
			self.x_gen(ch);
		}
		let mut stereo = is_cpe;
		if ps {
			if self.ps.common.start {
				let top = (self.kx[1] + self.m[1]) as usize;
				let [l, r] = &mut *self.x;
				self.ps.apply(l, r, top);
			} else {
				self.x[1] = self.x[0];
			}
			stereo = true;
		}
		self.qmf_synthesis(0, out_l, downsampled);
		if stereo {
			if let Some(r) = out_r {
				self.qmf_synthesis(1, r, downsampled);
			}
		}
	}
}

fn make_bands(start: i32, stop: i32, num_bands: i32) -> Vec<i16> {
	let base = (stop as f32 / start as f32).powf(1.0 / num_bands as f32);
	let mut prod = start as f32;
	let mut previous = start;
	let mut out = Vec::with_capacity(num_bands as usize);
	for _ in 0..num_bands - 1 {
		prod *= base;
		let present = prod.round_ties_even() as i32;
		out.push((present - previous) as i16);
		previous = present;
	}
	out.push((stop - previous) as i16);
	out
}

#[inline]
fn sum_square(x: &[Cplx]) -> f32 {
	let (mut s0, mut s1) = (0.0f32, 0.0f32);
	let mut i = 0;
	while i + 1 < x.len() {
		s0 += x[i][0] * x[i][0];
		s1 += x[i][1] * x[i][1];
		s0 += x[i + 1][0] * x[i + 1][0];
		s1 += x[i + 1][1] * x[i + 1][1];
		i += 2;
	}
	s0 + s1
}

/// `sbr_autocorrelate_c`: phi[lag-ish][row][re/im].
fn autocorrelate(x: &[Cplx; 40]) -> [[[f32; 2]; 2]; 3] {
	let mut real_sum2 = x[0][0] * x[2][0] + x[0][1] * x[2][1];
	let mut imag_sum2 = x[0][0] * x[2][1] - x[0][1] * x[2][0];
	let (mut real_sum1, mut imag_sum1, mut real_sum0) = (0.0f32, 0.0f32, 0.0f32);
	for i in 1..38 {
		real_sum0 += x[i][0] * x[i][0] + x[i][1] * x[i][1];
		real_sum1 += x[i][0] * x[i + 1][0] + x[i][1] * x[i + 1][1];
		imag_sum1 += x[i][0] * x[i + 1][1] - x[i][1] * x[i + 1][0];
		real_sum2 += x[i][0] * x[i + 2][0] + x[i][1] * x[i + 2][1];
		imag_sum2 += x[i][0] * x[i + 2][1] - x[i][1] * x[i + 2][0];
	}
	let mut phi = [[[0.0f32; 2]; 2]; 3];
	phi[0][1] = [real_sum2, imag_sum2];
	phi[2][1][0] = real_sum0 + x[0][0] * x[0][0] + x[0][1] * x[0][1];
	phi[1][0][0] = real_sum0 + x[38][0] * x[38][0] + x[38][1] * x[38][1];
	phi[1][1] = [
		real_sum1 + x[0][0] * x[1][0] + x[0][1] * x[1][1],
		imag_sum1 + x[0][0] * x[1][1] - x[0][1] * x[1][0],
	];
	phi[0][0] = [
		real_sum1 + x[38][0] * x[39][0] + x[38][1] * x[39][1],
		imag_sum1 + x[38][0] * x[39][1] - x[38][1] * x[39][0],
	];
	phi
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn huffman_tables_build() {
		assert_eq!(shared().vlc.len(), 10);
	}

	#[test]
	fn exp2fi_is_exact() {
		assert_eq!(exp2fi(0), 1.0);
		assert_eq!(exp2fi(10), 1024.0);
		assert_eq!(exp2fi(-3), 0.125);
	}
}
