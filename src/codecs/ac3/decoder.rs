//! AC-3 and E-AC-3 frame decoder (ATSC A/52 §7 and Annex E; FFmpeg
//! `ac3dec.c`, `eac3dec.c`).
//!
//! One syncframe (or an independent E-AC-3 frame followed by its
//! dependent substream) per call:
//!
//! 1. `audfrm()` (E-AC-3) and per block: strategies, exponents, the
//!    parametric bit allocation ([`super::bitalloc`]),
//! 2. mantissa unpacking (grouped/symmetric/asymmetric quantisers, or AHT
//!    vector and gain-adaptive quantisation), coupling, rematrixing,
//! 3. dynamic range scaling, spectral extension,
//! 4. the 512-point (or 2×256-point) IMDCT with a KBD window and overlap.
//!
//! Output is planar `f32` in FFmpeg's native channel order, as FFmpeg's
//! float decoder produces it (no downmix, `drc_scale` 1).

use super::bitalloc::{self, Dba, MAX_COEFS, Params};
use super::header::{
	ACMOD_CHANNELS, FrameType, Header, HeaderError, MODE_DUAL_MONO, MODE_MONO, MODE_STEREO, SYNC_WORD,
};
use super::tables::*;
use crate::codecs::bitstream::BitReader;
use crate::codecs::dsp::{Complex, Imdct};
use crate::codecs::flac::md5::Md5;

/// Coupling channel plus 5 full-bandwidth channels plus LFE.
const MAX_CH: usize = 7;
const CPL: usize = 0;
pub const BLOCK_LEN: usize = 256;
const HALF: usize = BLOCK_LEN / 2;
/// Two substreams (independent, dependent) of up to 6 coded channels.
const SLOTS: usize = 12;

const EXP_REUSE: u8 = 0;
const EXP_D45: u8 = 3;

// FFmpeg channel mask bits (`AV_CH_*` positions).
pub const CH_FL: u8 = 0;
pub const CH_FR: u8 = 1;
pub const CH_FC: u8 = 2;
pub const CH_LFE: u8 = 3;
pub const CH_BL: u8 = 4;
pub const CH_BR: u8 = 5;
pub const CH_FLC: u8 = 6;
pub const CH_FRC: u8 = 7;
pub const CH_BC: u8 = 8;
pub const CH_SL: u8 = 9;
pub const CH_SR: u8 = 10;
pub const CH_TC: u8 = 11;
pub const CH_TFL: u8 = 12;
pub const CH_TFC: u8 = 13;
pub const CH_TFR: u8 = 14;
pub const CH_TBL: u8 = 15;
pub const CH_TBR: u8 = 17;
pub const CH_WL: u8 = 31;
pub const CH_WR: u8 = 32;
pub const CH_SDL: u8 = 33;
pub const CH_SDR: u8 = 34;
pub const CH_LFE2: u8 = 35;

const fn bit(b: u8) -> u64 {
	1 << b
}

/// Channel locations of the 16 `chanmap` bits, MSB first (Table E.5):
/// `(single location, mask)`.
pub const CUSTOM_LOCATIONS: [(bool, u64); 16] = [
	(true, bit(CH_FL)),
	(true, bit(CH_FC)),
	(true, bit(CH_FR)),
	(true, bit(CH_SL)),
	(true, bit(CH_SR)),
	(false, bit(CH_FLC) | bit(CH_FRC)),
	(false, bit(CH_BL) | bit(CH_BR)),
	(false, bit(CH_BC)),
	(false, bit(CH_TC)),
	(false, bit(CH_SDL) | bit(CH_SDR)),
	(false, bit(CH_WL) | bit(CH_WR)),
	(false, bit(CH_TFL) | bit(CH_TFR)),
	(false, bit(CH_TFC)),
	(false, bit(CH_TBL) | bit(CH_TBR)),
	(false, bit(CH_LFE2)),
	(true, bit(CH_LFE)),
];

/// Locations of the coded channels of each `acmod`, in coded order.
fn acmod_locations(mode: u8) -> &'static [u8] {
	match mode {
		MODE_MONO => &[CH_FC],
		0 | MODE_STEREO => &[CH_FL, CH_FR],
		3 => &[CH_FL, CH_FC, CH_FR],
		4 => &[CH_FL, CH_FR, CH_BC],
		5 => &[CH_FL, CH_FC, CH_FR, CH_BC],
		6 => &[CH_FL, CH_FR, CH_SL, CH_SR],
		_ => &[CH_FL, CH_FC, CH_FR, CH_SL, CH_SR],
	}
}

const REMATRIX_BANDS: [usize; 5] = [13, 25, 37, 61, 253];
const DEFAULT_CPL_BAND_STRUCT: [u8; 18] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1];
const DEFAULT_SPX_BAND_STRUCT: [u8; 17] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
const QUANTIZATION_BITS: [u8; 16] = [0, 3, 5, 7, 11, 15, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16];

/// FFmpeg's `AVLFG` lagged Fibonacci generator, seeded through MD5 exactly
/// like `av_lfg_init`, so dither noise matches FFmpeg sample for sample.
#[derive(Debug, Clone)]
struct Lfg {
	state: [u32; 64],
	index: u32,
}

impl Lfg {
	fn new(seed: u32) -> Self {
		let mut state = [0u32; 64];
		// Bytes 5..16 carry over from the previous digest, as in FFmpeg.
		let mut tmp = [0u8; 16];
		let mut i = 8;
		while i < 64 {
			tmp[..4].copy_from_slice(&seed.to_le_bytes());
			tmp[4] = i as u8;
			let mut md5 = Md5::new();
			md5.update(&tmp);
			tmp = md5.finish();
			for k in 0..4 {
				state[i + k] =
					u32::from_le_bytes([tmp[4 * k], tmp[4 * k + 1], tmp[4 * k + 2], tmp[4 * k + 3]]);
			}
			i += 4;
		}
		Self { state, index: 0 }
	}

	#[inline]
	fn next(&mut self) -> u32 {
		let i = self.index as usize;
		let a =
			self.state[(i.wrapping_sub(24)) & 63].wrapping_add(self.state[(i.wrapping_sub(55)) & 63]);
		self.state[i & 63] = a;
		self.index = self.index.wrapping_add(1);
		a
	}
}

/// FFmpeg's `ff_kbd_window_init(window, alpha, n)`: the rising half (`n`
/// values) of a Kaiser-Bessel-derived window of length `2n`.
fn kbd_window(alpha: f64, n: usize) -> Vec<f32> {
	let alpha2 = 4.0 * (alpha * std::f64::consts::PI / n as f64).powi(2);
	let half = n / 2;
	let temp: Vec<f64> =
		(0..=half).map(|i| bessel_i0(((i * (n - i)) as f64 * alpha2).sqrt())).collect();
	let mut scale = 0.0;
	for (i, t) in temp.iter().enumerate() {
		scale += t * if i > 0 && i < half { 2.0 } else { 1.0 };
	}
	let scale = 1.0 / (scale + 1.0);
	let mut sum = 0.0;
	(0..n)
		.map(|i| {
			sum += if i <= half { temp[i] } else { temp[n - i] };
			(sum * scale).sqrt() as f32
		})
		.collect()
}

fn bessel_i0(x: f64) -> f64 {
	let q = x * x / 4.0;
	let (mut sum, mut term) = (1.0, 1.0);
	for k in 1..200 {
		term *= q / (k as f64 * k as f64);
		sum += term;
		if term < sum * 1e-17 {
			break;
		}
	}
	sum
}

/// `vector_fmul_window`: overlap-add of the previous half (`src0`) and the
/// current half (`src1`) with a symmetric window of `2 * len` values.
#[inline]
fn fmul_window(dst: &mut [f32], src0: &[f32], src1: &[f32], win: &[f32], len: usize) {
	let (dst, src0, src1, win) = (&mut dst[..2 * len], &src0[..len], &src1[..len], &win[..2 * len]);
	for i in 0..len {
		let j = 2 * len - 1 - i;
		let s0 = src0[i];
		let s1 = src1[len - 1 - i];
		let wi = win[i];
		let wj = win[j];
		dst[i] = s0 * wj - s1 * wi;
		dst[j] = s0 * wi + s1 * wj;
	}
}

#[derive(Debug)]
struct Invalid(#[allow(dead_code)] &'static str);
type BlockResult<T> = std::result::Result<T, Invalid>;

/// The decoded audio of one access unit.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
	pub sample_rate: u32,
	/// FFmpeg channel mask of the output layout.
	pub layout: u64,
	/// One plane per output channel, in mask-bit order.
	pub planes: Vec<Vec<f32>>,
	/// `false` when (part of) the frame was concealed.
	pub complete: bool,
	pub eac3: bool,
}

impl DecodedFrame {
	pub fn samples(&self) -> usize {
		self.planes.first().map_or(0, Vec::len)
	}
}

/// Mantissa group state shared by all channels of one block (§7.3.5).
#[derive(Default)]
struct MantGroups {
	b1_mant: [i32; 2],
	b2_mant: [i32; 2],
	b4_mant: i32,
	b1: usize,
	b2: usize,
	b4: usize,
}

/// Grouped mantissa tables (Tables 7.19-7.23), as 24-bit fixed point.
struct MantissaTables {
	bap1: [[i32; 3]; 32],
	bap2: [[i32; 3]; 128],
	bap3: [i32; 8],
	bap4: [[i32; 2]; 128],
	bap5: [i32; 16],
	ungroup7: [[u8; 3]; 128],
	ungroup5: [[u8; 3]; 32],
	dynrng: [f32; 256],
	heavy: [f32; 256],
	window: Vec<f32>,
	imdct256: Imdct,
	imdct128: Imdct,
}

fn sym(code: i32, levels: i32) -> i32 {
	((code - (levels >> 1)) * (1 << 24)) / levels
}

impl MantissaTables {
	fn get() -> &'static MantissaTables {
		static T: std::sync::OnceLock<MantissaTables> = std::sync::OnceLock::new();
		T.get_or_init(|| {
			let mut ungroup5 = [[0u8; 3]; 32];
			for (i, g) in ungroup5.iter_mut().enumerate() {
				*g = [(i / 9) as u8, ((i % 9) / 3) as u8, ((i % 9) % 3) as u8];
			}
			let mut ungroup7 = [[0u8; 3]; 128];
			for (i, g) in ungroup7.iter_mut().enumerate() {
				*g = [(i / 25) as u8, ((i % 25) / 5) as u8, ((i % 25) % 5) as u8];
			}
			let mut bap1 = [[0; 3]; 32];
			for i in 0..32 {
				for k in 0..3 {
					bap1[i][k] = sym(i32::from(ungroup5[i][k]), 3);
				}
			}
			let mut bap2 = [[0; 3]; 128];
			let mut bap4 = [[0; 2]; 128];
			for i in 0..128 {
				for k in 0..3 {
					bap2[i][k] = sym(i32::from(ungroup7[i][k]), 5);
				}
				bap4[i] = [sym(i as i32 / 11, 11), sym(i as i32 % 11, 11)];
			}
			let mut bap3 = [0; 8];
			for (i, v) in bap3.iter_mut().enumerate().take(7) {
				*v = sym(i as i32, 7);
			}
			let mut bap5 = [0; 16];
			for (i, v) in bap5.iter_mut().enumerate().take(15) {
				*v = sym(i as i32, 15);
			}
			let mut dynrng = [0f32; 256];
			let mut heavy = [0f32; 256];
			for i in 0..256i32 {
				let v = (i >> 5) - ((i >> 7) << 3) - 5;
				dynrng[i as usize] = 2f32.powi(v) * ((i & 0x1F) | 0x20) as f32;
				let v = (i >> 4) - ((i >> 7) << 4) - 4;
				heavy[i as usize] = 2f32.powi(v) * ((i & 0xF) | 0x10) as f32;
			}
			MantissaTables {
				bap1,
				bap2,
				bap3,
				bap4,
				bap5,
				ungroup7,
				ungroup5,
				dynrng,
				heavy,
				window: kbd_window(5.0, BLOCK_LEN),
				imdct256: Imdct::new(2 * BLOCK_LEN, 1.0),
				imdct128: Imdct::new(BLOCK_LEN, 1.0),
			}
		})
	}
}

/// Decoder state (FFmpeg's `AC3DecodeContext`).
#[derive(Clone)]
pub struct Ac3DecoderState {
	// Frame header.
	eac3: bool,
	frame_type: FrameType,
	channel_mode: u8,
	lfe_on: bool,
	channels: usize,
	fbw_channels: usize,
	lfe_ch: usize,
	num_blocks: usize,
	sample_rate: u32,
	channel_map: u16,

	// Syntax flags (E-AC-3 audfrm).
	snr_offset_strategy: u32,
	block_switch_syntax: bool,
	dither_flag_syntax: bool,
	bit_allocation_syntax: bool,
	fast_gain_syntax: bool,
	dba_syntax: bool,
	skip_syntax: bool,
	first_cpl_leak: bool,
	cpl_strategy_exists: [bool; 6],
	cpl_in_use: [bool; 6],
	exp_strategy: [[u8; MAX_CH]; 6],
	channel_uses_aht: [bool; MAX_CH],
	spx_atten_code: [i8; MAX_CH],

	// Block state.
	block_switch: [bool; MAX_CH],
	dither_flag: [bool; MAX_CH],
	dynamic_range: [f32; 2],
	heavy_dynamic_range: [f32; 2],
	compression_exists: [bool; 2],
	heavy_compression: bool,

	spx_in_use: bool,
	channel_uses_spx: [bool; MAX_CH],
	first_spx_coords: [bool; MAX_CH],
	spx_dst_start_freq: usize,
	spx_src_start_freq: usize,
	spx_dst_end_freq: usize,
	num_spx_bands: usize,
	spx_band_sizes: [u8; 17],
	spx_band_struct: [u8; 17],
	spx_noise_blend: [[f32; 17]; MAX_CH],
	spx_signal_blend: [[f32; 17]; MAX_CH],

	channel_in_cpl: [bool; MAX_CH],
	first_cpl_coords: [bool; MAX_CH],
	phase_flags_in_use: bool,
	phase_flags: [bool; 18],
	num_cpl_bands: usize,
	cpl_band_sizes: [u8; 18],
	cpl_band_struct: [u8; 18],
	cpl_coords: [[i32; 18]; MAX_CH],

	num_rematrixing_bands: usize,
	rematrixing_flags: [bool; 4],

	start_freq: [usize; MAX_CH],
	end_freq: [usize; MAX_CH],
	num_exp_groups: [usize; MAX_CH],
	dexps: Box<[[i8; MAX_COEFS]; MAX_CH]>,
	psd: Box<[[i16; MAX_COEFS]; MAX_CH]>,
	band_psd: Box<[[i16; bitalloc::CRITICAL_BANDS]; MAX_CH]>,
	mask: Box<[[i16; bitalloc::CRITICAL_BANDS]; MAX_CH]>,
	bap: Box<[[u8; MAX_COEFS]; MAX_CH]>,
	params: Params,
	snr_offset: [i32; MAX_CH],
	fast_gain: [i32; MAX_CH],
	dba: [Dba; MAX_CH],

	pre_mantissa: Box<[[[i32; 6]; MAX_COEFS]; MAX_CH]>,
	coeffs: Box<[[f32; MAX_COEFS]; MAX_CH]>,
	transform: Box<[[f32; MAX_COEFS]; MAX_CH]>,

	lfg: Lfg,
	/// IMDCT overlap per substream slot (independent: 0-5, dependent: 6-11).
	delay: Box<[[f32; HALF]; SLOTS]>,
	/// Decoded planes of the current access unit in coded order, per slot.
	coded: Vec<Vec<f32>>,
	/// Last decoded block per slot, for concealment.
	last_block: Box<[[f32; BLOCK_LEN]; SLOTS]>,
	work: Vec<f32>,
	cbuf: Vec<Complex>,

	/// Frames that failed to decode and were concealed.
	pub concealed_frames: u64,
}

impl std::fmt::Debug for Ac3DecoderState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Ac3DecoderState")
			.field("eac3", &self.eac3)
			.field("channels", &self.channels)
			.field("sample_rate", &self.sample_rate)
			.finish_non_exhaustive()
	}
}

impl Default for Ac3DecoderState {
	fn default() -> Self {
		Self::new()
	}
}

impl Ac3DecoderState {
	pub fn new() -> Self {
		Self {
			eac3: false,
			frame_type: FrameType::Ac3Convert,
			channel_mode: 0,
			lfe_on: false,
			channels: 0,
			fbw_channels: 0,
			lfe_ch: 0,
			num_blocks: 6,
			sample_rate: 0,
			channel_map: 0,
			snr_offset_strategy: 2,
			block_switch_syntax: true,
			dither_flag_syntax: true,
			bit_allocation_syntax: true,
			fast_gain_syntax: false,
			dba_syntax: true,
			skip_syntax: true,
			first_cpl_leak: false,
			cpl_strategy_exists: [false; 6],
			cpl_in_use: [false; 6],
			exp_strategy: [[0; MAX_CH]; 6],
			channel_uses_aht: [false; MAX_CH],
			spx_atten_code: [-1; MAX_CH],
			block_switch: [false; MAX_CH],
			dither_flag: [false; MAX_CH],
			dynamic_range: [1.0; 2],
			heavy_dynamic_range: [1.0; 2],
			compression_exists: [false; 2],
			heavy_compression: false,
			spx_in_use: false,
			channel_uses_spx: [false; MAX_CH],
			first_spx_coords: [true; MAX_CH],
			spx_dst_start_freq: 0,
			spx_src_start_freq: 0,
			spx_dst_end_freq: 0,
			num_spx_bands: 0,
			spx_band_sizes: [0; 17],
			spx_band_struct: [0; 17],
			spx_noise_blend: [[0.0; 17]; MAX_CH],
			spx_signal_blend: [[0.0; 17]; MAX_CH],
			channel_in_cpl: [false; MAX_CH],
			first_cpl_coords: [true; MAX_CH],
			phase_flags_in_use: false,
			phase_flags: [false; 18],
			num_cpl_bands: 0,
			cpl_band_sizes: [0; 18],
			cpl_band_struct: [0; 18],
			cpl_coords: [[0; 18]; MAX_CH],
			num_rematrixing_bands: 0,
			rematrixing_flags: [false; 4],
			start_freq: [0; MAX_CH],
			end_freq: [0; MAX_CH],
			num_exp_groups: [0; MAX_CH],
			dexps: Box::new([[0; MAX_COEFS]; MAX_CH]),
			psd: Box::new([[0; MAX_COEFS]; MAX_CH]),
			band_psd: Box::new([[0; bitalloc::CRITICAL_BANDS]; MAX_CH]),
			mask: Box::new([[0; bitalloc::CRITICAL_BANDS]; MAX_CH]),
			bap: Box::new([[0; MAX_COEFS]; MAX_CH]),
			params: Params::default(),
			snr_offset: [0; MAX_CH],
			fast_gain: [0; MAX_CH],
			dba: [Dba::default(); MAX_CH],
			pre_mantissa: Box::new([[[0; 6]; MAX_COEFS]; MAX_CH]),
			coeffs: Box::new([[0.0; MAX_COEFS]; MAX_CH]),
			transform: Box::new([[0.0; MAX_COEFS]; MAX_CH]),
			lfg: Lfg::new(0),
			delay: Box::new([[0.0; HALF]; SLOTS]),
			coded: vec![Vec::new(); SLOTS],
			last_block: Box::new([[0.0; BLOCK_LEN]; SLOTS]),
			work: vec![0.0; 2 * BLOCK_LEN],
			cbuf: vec![Complex::default(); BLOCK_LEN / 2],
			concealed_frames: 0,
		}
	}

	/// Enables heavy compression (`heavy_compr`, off by default as in
	/// FFmpeg).
	pub fn set_heavy_compression(&mut self, on: bool) {
		self.heavy_compression = on;
	}

	/// Clears overlap and dither state (after a seek).
	pub fn reset(&mut self) {
		*self = Self { heavy_compression: self.heavy_compression, ..Self::new() };
	}

	/// Decodes every access unit in `data` (a packet may hold several
	/// syncframes).
	pub fn decode_packet(&mut self, data: &[u8]) -> Vec<DecodedFrame> {
		let mut out = Vec::new();
		let mut at = 0;
		while at + super::header::MIN_HEADER_BYTES <= data.len() {
			let Some(start) = find_sync(&data[at..]) else { break };
			at += start;
			match self.decode_access_unit(&data[at..]) {
				(Some(frame), used) => {
					out.push(frame);
					at += used.max(1);
				}
				(None, used) => at += used.max(1),
			}
		}
		out
	}

	/// Decodes one independent frame plus its dependent substream (if any)
	/// at the start of `data`. Returns the frame and the bytes consumed.
	fn decode_access_unit(&mut self, data: &[u8]) -> (Option<DecodedFrame>, usize) {
		let mut bs = BitReader::new(data);
		let header = match Header::parse(&mut bs) {
			Ok(h) => h,
			Err(HeaderError::FrameType) => {
				// Additional substreams and reserved frame types are skipped.
				let used = Header::peek_size(data).map_or(data.len(), |(s, _)| s);
				return (None, used);
			}
			Err(_) => return (None, 2),
		};
		if header.frame_type == Some(FrameType::Dependent) {
			// A dependent frame without its independent frame.
			return (None, header.frame_size);
		}
		let mut used = header.frame_size.min(data.len());
		let base_mode = header.channel_mode;
		let base_lfe = header.lfe_on;
		let base_channels = header.channels;
		let num_blocks = header.num_blocks;
		let ok = if header.frame_size <= data.len() {
			// Conceals the blocks it could not decode itself.
			self.decode_frame(&mut bs, &header, 0).is_ok()
		} else {
			self.conceal(0, base_channels, num_blocks);
			false
		};
		let mut complete = ok;
		let mut dependent: Option<(Header, bool)> = None;
		if data.len() > header.frame_size + 16 {
			let rest = &data[header.frame_size..];
			let mut bs2 = BitReader::new(rest);
			if let Ok(h2) = Header::parse(&mut bs2) {
				if h2.frame_type == Some(FrameType::Dependent)
					&& h2.num_blocks == num_blocks
					&& h2.sample_rate == header.sample_rate
				{
					let ok2 = if h2.frame_size <= rest.len() {
						self.decode_frame(&mut bs2, &h2, 6).is_ok()
					} else {
						self.conceal(6, h2.channels, num_blocks);
						false
					};
					complete &= ok2;
					used += h2.frame_size.min(rest.len());
					dependent = Some((h2, ok2));
				}
			}
		}
		if !complete {
			self.concealed_frames += 1;
		}

		// Assemble the output layout.
		let mut sources: Vec<(u8, usize)> = Vec::new(); // (location, coded slot)
		for (i, &loc) in acmod_locations(base_mode).iter().enumerate() {
			sources.push((loc, i));
		}
		if base_lfe {
			sources.push((CH_LFE, base_channels - 1));
		}
		if let Some((h2, _)) = &dependent {
			let mut next = 6;
			let end = 6 + h2.channels;
			let assign = |loc: u8, sources: &mut Vec<(u8, usize)>, next: &mut usize| {
				if *next >= end {
					return;
				}
				if let Some(s) = sources.iter_mut().find(|s| s.0 == loc) {
					s.1 = *next;
				} else {
					sources.push((loc, *next));
				}
				*next += 1;
			};
			if h2.channel_map != 0 {
				for (i, &(_, mask)) in CUSTOM_LOCATIONS.iter().enumerate() {
					if h2.channel_map & (1 << (15 - i)) != 0 {
						for b in 0..64u8 {
							if mask & (1 << b) != 0 {
								assign(b, &mut sources, &mut next);
							}
						}
					}
				}
			} else {
				// No map: the dependent substream replaces the channels of its
				// own acmod.
				for &loc in acmod_locations(h2.channel_mode) {
					assign(loc, &mut sources, &mut next);
				}
				if h2.lfe_on {
					assign(CH_LFE, &mut sources, &mut next);
				}
			}
		}
		sources.sort_by_key(|s| s.0);
		let layout = sources.iter().fold(0u64, |m, s| m | (1 << s.0));
		let planes = sources.iter().map(|&(_, slot)| self.coded[slot].clone()).collect();
		(
			Some(DecodedFrame {
				sample_rate: header.sample_rate,
				layout,
				planes,
				complete,
				eac3: header.is_eac3(),
			}),
			used,
		)
	}

	/// Repeats the last good block over a whole frame (FFmpeg's
	/// concealment).
	fn conceal(&mut self, offset: usize, channels: usize, num_blocks: usize) {
		for ch in 0..channels.min(6) {
			let slot = offset + ch;
			let plane = &mut self.coded[slot];
			plane.clear();
			for _ in 0..num_blocks {
				plane.extend_from_slice(&self.last_block[slot]);
			}
		}
	}

	fn apply_header(&mut self, h: &Header) {
		self.params.sr_code = h.sr_code;
		self.params.sr_shift = h.sr_shift;
		self.channel_mode = h.channel_mode;
		self.lfe_on = h.lfe_on;
		self.sample_rate = h.sample_rate;
		self.channels = h.channels;
		self.fbw_channels = h.channels - usize::from(h.lfe_on);
		self.lfe_ch = self.fbw_channels + 1;
		self.num_blocks = h.num_blocks;
		self.frame_type = h.frame_type.unwrap_or(FrameType::Ac3Convert);
		self.channel_map = h.channel_map;
		if h.lfe_on {
			let l = self.lfe_ch;
			self.start_freq[l] = 0;
			self.end_freq[l] = 7;
			self.num_exp_groups[l] = 2;
			self.channel_in_cpl[l] = false;
		}
		let t = MantissaTables::get();
		for i in 0..if h.channel_mode == 0 { 2 } else { 1 } {
			self.compression_exists[i] = h.compression_exists[i];
			if h.compression_exists[i] {
				self.heavy_dynamic_range[i] = t.heavy[usize::from(h.heavy_dynamic_range[i] << 1)];
			}
		}
		if !h.is_eac3() {
			self.eac3 = false;
			self.snr_offset_strategy = 2;
			self.block_switch_syntax = true;
			self.dither_flag_syntax = true;
			self.bit_allocation_syntax = true;
			self.fast_gain_syntax = false;
			self.first_cpl_leak = false;
			self.dba_syntax = true;
			self.skip_syntax = true;
			self.channel_uses_aht = [false; MAX_CH];
		} else {
			self.eac3 = true;
		}
	}

	fn decode_frame(&mut self, bs: &mut BitReader, h: &Header, offset: usize) -> BlockResult<()> {
		self.apply_header(h);
		if self.eac3 {
			if h.sr_code == 3 {
				return Err(Invalid("e-ac-3 reduced sample rates are not supported"));
			}
			self.parse_audfrm(bs, h)?;
		}
		for ch in 0..self.channels {
			let plane = &mut self.coded[offset + ch];
			plane.clear();
			plane.resize(self.num_blocks * BLOCK_LEN, 0.0);
		}
		for blk in 0..self.num_blocks {
			if let Err(e) = self.decode_block(bs, blk, offset) {
				// Conceal the rest of the frame with the last good block.
				for b in blk..self.num_blocks {
					for ch in 0..self.channels {
						let slot = offset + ch;
						let last = self.last_block[slot];
						self.coded[slot][b * BLOCK_LEN..(b + 1) * BLOCK_LEN].copy_from_slice(&last);
					}
				}
				return Err(e);
			}
			if bs.overrun() {
				return Err(Invalid("ac-3 frame truncated"));
			}
			for ch in 0..self.channels {
				let slot = offset + ch;
				self.last_block[slot]
					.copy_from_slice(&self.coded[slot][blk * BLOCK_LEN..(blk + 1) * BLOCK_LEN]);
			}
		}
		Ok(())
	}

	/// `audfrm()` (Annex E §E.1.2.3; FFmpeg `ff_eac3_parse_header`).
	fn parse_audfrm(&mut self, bs: &mut BitReader, h: &Header) -> BlockResult<()> {
		let (exp_strategy_ac3, parse_aht) =
			if self.num_blocks == 6 { (bs.bit(), bs.bit()) } else { (true, false) };
		self.snr_offset_strategy = bs.bits(2);
		let parse_transient = bs.bit();
		self.block_switch_syntax = bs.bit();
		if !self.block_switch_syntax {
			self.block_switch = [false; MAX_CH];
		}
		self.dither_flag_syntax = bs.bit();
		if !self.dither_flag_syntax {
			for ch in 1..=self.fbw_channels {
				self.dither_flag[ch] = true;
			}
		}
		self.dither_flag[CPL] = false;
		self.dither_flag[self.lfe_ch.min(MAX_CH - 1)] = false;
		self.bit_allocation_syntax = bs.bit();
		if !self.bit_allocation_syntax {
			self.params.slow_decay = i32::from(bitalloc::SLOW_DECAY[2]) >> self.params.sr_shift;
			self.params.fast_decay = i32::from(bitalloc::FAST_DECAY[1]) >> self.params.sr_shift;
			self.params.slow_gain = i32::from(bitalloc::SLOW_GAIN[1]);
			self.params.db_per_bit = i32::from(bitalloc::DB_PER_BIT[2]);
			self.params.floor = i32::from(bitalloc::FLOOR[7]);
		}
		self.fast_gain_syntax = bs.bit();
		self.dba_syntax = bs.bit();
		self.skip_syntax = bs.bit();
		let parse_spx_atten = bs.bit();

		let mut num_cpl_blocks = 0;
		if self.channel_mode > 1 {
			for blk in 0..self.num_blocks {
				self.cpl_strategy_exists[blk] = blk == 0 || bs.bit();
				self.cpl_in_use[blk] =
					if self.cpl_strategy_exists[blk] { bs.bit() } else { self.cpl_in_use[blk - 1] };
				num_cpl_blocks += usize::from(self.cpl_in_use[blk]);
			}
		} else {
			self.cpl_in_use = [false; 6];
		}

		if exp_strategy_ac3 {
			for blk in 0..self.num_blocks {
				for ch in usize::from(!self.cpl_in_use[blk])..=self.fbw_channels {
					self.exp_strategy[blk][ch] = bs.bits(2) as u8;
				}
			}
		} else {
			let first = usize::from(!(self.channel_mode > 1 && num_cpl_blocks > 0));
			for ch in first..=self.fbw_channels {
				let code = bs.bits(5) as usize;
				for blk in 0..6 {
					self.exp_strategy[blk][ch] = FRAME_EXP_STRATEGY[code][blk];
				}
			}
		}
		if self.lfe_on {
			for blk in 0..self.num_blocks {
				self.exp_strategy[blk][self.lfe_ch] = u8::from(bs.bit());
			}
		}
		if self.frame_type == FrameType::Independent && (self.num_blocks == 6 || bs.bit()) {
			bs.skip(5 * self.fbw_channels); // convexpstr
		}

		if parse_aht {
			self.channel_uses_aht[CPL] = false;
			for ch in usize::from(num_cpl_blocks != 6)..=self.channels {
				let mut use_aht = true;
				for blk in 1..6 {
					if self.exp_strategy[blk][ch] != EXP_REUSE || (ch == CPL && self.cpl_strategy_exists[blk])
					{
						use_aht = false;
						break;
					}
				}
				self.channel_uses_aht[ch] = use_aht && bs.bit();
			}
		} else {
			self.channel_uses_aht = [false; MAX_CH];
		}

		if self.snr_offset_strategy == 0 {
			let csnr = (bs.bits(6) as i32 - 15) << 4;
			let snr = (csnr + bs.bits(4) as i32) << 2;
			for ch in 0..=self.channels {
				self.snr_offset[ch] = snr;
			}
		}
		if parse_transient {
			for _ in 1..=self.fbw_channels {
				if bs.bit() {
					bs.skip(10 + 8);
				}
			}
		}
		for ch in 1..=self.fbw_channels {
			self.spx_atten_code[ch] = if parse_spx_atten && bs.bit() { bs.bits(5) as i8 } else { -1 };
		}
		if self.num_blocks > 1 && bs.bit() {
			let words = h.frame_size.saturating_sub(2).max(1);
			let log2 = usize::BITS - 1 - words.leading_zeros();
			bs.skip((self.num_blocks - 1) * (4 + log2 as usize));
		}
		for ch in 1..=self.fbw_channels {
			self.first_spx_coords[ch] = true;
			self.first_cpl_coords[ch] = true;
		}
		self.first_cpl_leak = true;
		if bs.overrun() {
			return Err(Invalid("e-ac-3 audfrm truncated"));
		}
		Ok(())
	}

	/// Band structure for coupling or spectral extension (§7.4.2/E.2.3.3).
	#[allow(clippy::too_many_arguments)]
	fn band_structure(
		bs: &mut BitReader,
		blk: usize,
		eac3: bool,
		start: usize,
		end: usize,
		default: &[u8],
		band_struct: &mut [u8],
		band_sizes: &mut [u8],
	) -> usize {
		let n_subbands = end - start;
		if blk == 0 {
			band_struct.copy_from_slice(default);
		}
		if !eac3 || bs.bit() {
			for sub in 0..n_subbands.saturating_sub(1) {
				band_struct[start + 1 + sub] = u8::from(bs.bit());
			}
		}
		let mut n_bands = n_subbands;
		let mut sizes = [0u8; 22];
		sizes[0] = 12;
		let mut bnd = 0;
		for sub in 1..n_subbands {
			if band_struct[start + 1 + sub - 1] != 0 {
				n_bands -= 1;
				sizes[bnd] += 12;
			} else {
				bnd += 1;
				sizes[bnd] = 12;
			}
		}
		band_sizes[..n_bands].copy_from_slice(&sizes[..n_bands]);
		n_bands
	}

	fn spx_strategy(&mut self, bs: &mut BitReader, blk: usize) -> BlockResult<()> {
		if self.channel_mode == MODE_MONO {
			self.channel_uses_spx[1] = true;
		} else {
			let mut bits = bs.bits(self.fbw_channels as u32);
			for ch in (1..=self.fbw_channels).rev() {
				self.channel_uses_spx[ch] = bits & 1 != 0;
				bits >>= 1;
			}
		}
		let dst_start = bs.bits(2) as usize;
		let mut start_sub = bs.bits(3) as usize + 2;
		if start_sub > 7 {
			start_sub += start_sub - 7;
		}
		let mut end_sub = bs.bits(3) as usize + 5;
		if end_sub > 7 {
			end_sub += end_sub - 7;
		}
		let dst_start = dst_start * 12 + 25;
		let src_start = start_sub * 12 + 25;
		let dst_end = end_sub * 12 + 25;
		if start_sub >= end_sub {
			return Err(Invalid("invalid spectral extension range"));
		}
		if dst_start >= src_start {
			return Err(Invalid("invalid spectral extension copy start"));
		}
		self.spx_dst_start_freq = dst_start;
		self.spx_src_start_freq = src_start;
		self.spx_dst_end_freq = dst_end;
		let eac3 = self.eac3;
		self.num_spx_bands = Self::band_structure(
			bs,
			blk,
			eac3,
			start_sub,
			end_sub,
			&DEFAULT_SPX_BAND_STRUCT,
			&mut self.spx_band_struct,
			&mut self.spx_band_sizes,
		);
		Ok(())
	}

	fn spx_coordinates(&mut self, bs: &mut BitReader) {
		for ch in 1..=self.fbw_channels {
			if !self.channel_uses_spx[ch] {
				self.first_spx_coords[ch] = true;
				continue;
			}
			if self.first_spx_coords[ch] || bs.bit() {
				self.first_spx_coords[ch] = false;
				let blend = bs.bits(5) as f32 * (1.0 / 32.0);
				let master = bs.bits(2) as i32 * 3;
				let mut bin = self.spx_src_start_freq;
				for bnd in 0..self.num_spx_bands {
					let size = usize::from(self.spx_band_sizes[bnd]);
					let nratio =
						((bin + (size >> 1)) as f32 / self.spx_dst_end_freq as f32 - blend).clamp(0.0, 1.0);
					let nblend = (3.0 * nratio).sqrt();
					let sblend = (1.0 - nratio).sqrt();
					bin += size;
					let exp = bs.bits(4) as i32;
					let mut mant = bs.bits(2) as i32;
					if exp == 15 {
						mant <<= 1;
					} else {
						mant += 4;
					}
					let shift = 25 - exp - master;
					let mant = if shift >= 0 { mant << shift } else { mant >> -shift };
					let coord = mant as f32 * (1.0 / (1 << 23) as f32);
					self.spx_noise_blend[ch][bnd] = nblend * coord;
					self.spx_signal_blend[ch][bnd] = sblend * coord;
				}
			}
		}
	}

	fn coupling_strategy(
		&mut self,
		bs: &mut BitReader,
		blk: usize,
		stages: &mut [u8; MAX_CH],
	) -> BlockResult<()> {
		*stages = [3; MAX_CH];
		if !self.eac3 {
			self.cpl_in_use[blk] = bs.bit();
		}
		if self.cpl_in_use[blk] {
			if self.channel_mode < MODE_STEREO {
				return Err(Invalid("coupling in mono or dual-mono"));
			}
			if self.eac3 && bs.bit() {
				return Err(Invalid("e-ac-3 enhanced coupling is not supported"));
			}
			if self.eac3 && self.channel_mode == MODE_STEREO {
				self.channel_in_cpl[1] = true;
				self.channel_in_cpl[2] = true;
			} else {
				for ch in 1..=self.fbw_channels {
					self.channel_in_cpl[ch] = bs.bit();
				}
			}
			if self.channel_mode == MODE_STEREO {
				self.phase_flags_in_use = bs.bit();
			}
			let start = bs.bits(4) as usize;
			let end = if self.spx_in_use {
				(self.spx_src_start_freq as isize - 37).max(0) as usize / 12
			} else {
				bs.bits(4) as usize + 3
			};
			if start >= end {
				return Err(Invalid("invalid coupling range"));
			}
			self.start_freq[CPL] = start * 12 + 37;
			self.end_freq[CPL] = end * 12 + 37;
			let eac3 = self.eac3;
			self.num_cpl_bands = Self::band_structure(
				bs,
				blk,
				eac3,
				start,
				end,
				&DEFAULT_CPL_BAND_STRUCT,
				&mut self.cpl_band_struct,
				&mut self.cpl_band_sizes,
			);
		} else {
			for ch in 1..=self.fbw_channels {
				self.channel_in_cpl[ch] = false;
				self.first_cpl_coords[ch] = true;
			}
			self.first_cpl_leak = self.eac3;
			self.phase_flags_in_use = false;
		}
		Ok(())
	}

	fn coupling_coordinates(&mut self, bs: &mut BitReader, blk: usize) -> BlockResult<()> {
		let mut exist = false;
		for ch in 1..=self.fbw_channels {
			if self.channel_in_cpl[ch] {
				if (self.eac3 && self.first_cpl_coords[ch]) || bs.bit() {
					self.first_cpl_coords[ch] = false;
					exist = true;
					let master = 3 * bs.bits(2);
					for bnd in 0..self.num_cpl_bands {
						let exp = bs.bits(4);
						let mant = bs.bits(4) as i32;
						let c = if exp == 15 { mant << 22 } else { (mant + 16) << 21 };
						self.cpl_coords[ch][bnd] = c >> (exp + master).min(31);
					}
				} else if blk == 0 {
					return Err(Invalid("coupling coordinates missing in block 0"));
				}
			} else {
				self.first_cpl_coords[ch] = true;
			}
		}
		if self.channel_mode == MODE_STEREO && exist {
			for bnd in 0..self.num_cpl_bands {
				self.phase_flags[bnd] = self.phase_flags_in_use && bs.bit();
			}
		}
		Ok(())
	}

	fn decode_exponents(
		bs: &mut BitReader,
		strategy: u8,
		ngrps: usize,
		absexp: i32,
		dexps: &mut [i8],
	) -> BlockResult<()> {
		let t = MantissaTables::get();
		let group_size = usize::from(strategy) + usize::from(strategy == EXP_D45);
		let mut prev = absexp;
		let mut j = 0;
		for _ in 0..ngrps {
			let acc = bs.bits(7) as usize;
			if acc >= 125 {
				return Err(Invalid("ac-3 exponent group out of range"));
			}
			for &d in &t.ungroup7[acc] {
				prev += i32::from(d) - 2;
				if !(0..=24).contains(&prev) {
					return Err(Invalid("ac-3 exponent out of range"));
				}
				let slots = dexps.get_mut(j..j + group_size).ok_or(Invalid("ac-3 exponents overflow"))?;
				slots.fill(prev as i8);
				j += group_size;
			}
		}
		Ok(())
	}

	/// `audblk()` (§5.4.3; FFmpeg `decode_audio_block`).
	fn decode_block(&mut self, bs: &mut BitReader, blk: usize, offset: usize) -> BlockResult<()> {
		let t = MantissaTables::get();
		let fbw = self.fbw_channels;
		let mut stages = [0u8; MAX_CH];

		if self.block_switch_syntax {
			for ch in 1..=fbw {
				self.block_switch[ch] = bs.bit();
			}
		}
		if self.dither_flag_syntax {
			for ch in 1..=fbw {
				self.dither_flag[ch] = bs.bit();
			}
		}
		// Dynamic range.
		let mut i = usize::from(self.channel_mode == MODE_DUAL_MONO);
		loop {
			if bs.bit() {
				self.dynamic_range[i] = t.dynrng[bs.bits(8) as usize];
			} else if blk == 0 {
				self.dynamic_range[i] = 1.0;
			}
			if i == 0 {
				break;
			}
			i -= 1;
		}

		// Spectral extension strategy.
		if self.eac3 && (blk == 0 || bs.bit()) {
			self.spx_in_use = bs.bit();
			if self.spx_in_use {
				self.spx_strategy(bs, blk)?;
			}
		}
		if !self.eac3 || !self.spx_in_use {
			self.spx_in_use = false;
			for ch in 1..=fbw {
				self.channel_uses_spx[ch] = false;
				self.first_spx_coords[ch] = true;
			}
		}
		if self.spx_in_use {
			self.spx_coordinates(bs);
		}

		// Coupling strategy.
		let new_cpl = if self.eac3 { self.cpl_strategy_exists[blk] } else { bs.bit() };
		if new_cpl {
			self.coupling_strategy(bs, blk, &mut stages)?;
		} else if !self.eac3 {
			if blk == 0 {
				return Err(Invalid("coupling strategy missing in block 0"));
			}
			self.cpl_in_use[blk] = self.cpl_in_use[blk - 1];
		}
		let cpl_in_use = self.cpl_in_use[blk];
		if cpl_in_use {
			self.coupling_coordinates(bs, blk)?;
		}

		// Rematrixing.
		if self.channel_mode == MODE_STEREO {
			if (self.eac3 && blk == 0) || bs.bit() {
				self.num_rematrixing_bands = 4;
				if cpl_in_use && self.start_freq[CPL] <= 61 {
					self.num_rematrixing_bands -= 1 + usize::from(self.start_freq[CPL] == 37);
				} else if self.spx_in_use && self.spx_src_start_freq <= 61 {
					self.num_rematrixing_bands -= 1;
				}
				for bnd in 0..self.num_rematrixing_bands {
					self.rematrixing_flags[bnd] = bs.bit();
				}
			} else if blk == 0 {
				self.num_rematrixing_bands = 0;
			}
		}

		// Exponent strategies.
		let first = usize::from(!cpl_in_use);
		for ch in first..=self.channels {
			if !self.eac3 {
				self.exp_strategy[blk][ch] = bs.bits(2 - u32::from(ch == self.lfe_ch)) as u8;
			}
			if self.exp_strategy[blk][ch] != EXP_REUSE {
				stages[ch] = 3;
			}
		}

		// Channel bandwidth.
		for ch in 1..=fbw {
			self.start_freq[ch] = 0;
			if self.exp_strategy[blk][ch] != EXP_REUSE {
				let prev = self.end_freq[ch];
				if self.channel_in_cpl[ch] {
					self.end_freq[ch] = self.start_freq[CPL];
				} else if self.channel_uses_spx[ch] {
					self.end_freq[ch] = self.spx_src_start_freq;
				} else {
					let code = bs.bits(6) as usize;
					if code > 60 {
						return Err(Invalid("ac-3 bandwidth code out of range"));
					}
					self.end_freq[ch] = code * 3 + 73;
				}
				let gs = 3 << (self.exp_strategy[blk][ch] - 1);
				self.num_exp_groups[ch] = (self.end_freq[ch] + gs - 4) / gs;
				if blk > 0 && self.end_freq[ch] != prev {
					stages = [3; MAX_CH];
				}
			}
		}
		if cpl_in_use && self.exp_strategy[blk][CPL] != EXP_REUSE {
			self.num_exp_groups[CPL] =
				(self.end_freq[CPL] - self.start_freq[CPL]) / (3 << (self.exp_strategy[blk][CPL] - 1));
		}

		// Exponents.
		for ch in first..=self.channels {
			let strategy = self.exp_strategy[blk][ch];
			if strategy != EXP_REUSE {
				let abs = (bs.bits(4) as i32) << u32::from(ch == CPL);
				self.dexps[ch][0] = abs as i8;
				let from = self.start_freq[ch] + usize::from(ch != CPL);
				Self::decode_exponents(
					bs,
					strategy,
					self.num_exp_groups[ch],
					abs,
					&mut self.dexps[ch][from..],
				)?;
				if ch != CPL && ch != self.lfe_ch {
					bs.skip(2); // gainrng
				}
			}
		}

		// Bit allocation parameters.
		if self.bit_allocation_syntax {
			if bs.bit() {
				let sh = self.params.sr_shift;
				self.params.slow_decay = i32::from(bitalloc::SLOW_DECAY[bs.bits(2) as usize]) >> sh;
				self.params.fast_decay = i32::from(bitalloc::FAST_DECAY[bs.bits(2) as usize]) >> sh;
				self.params.slow_gain = i32::from(bitalloc::SLOW_GAIN[bs.bits(2) as usize]);
				self.params.db_per_bit = i32::from(bitalloc::DB_PER_BIT[bs.bits(2) as usize]);
				self.params.floor = i32::from(bitalloc::FLOOR[bs.bits(3) as usize]);
				for s in &mut stages[first..=self.channels] {
					*s = (*s).max(2);
				}
			} else if blk == 0 {
				return Err(Invalid("bit allocation info missing in block 0"));
			}
		}

		// SNR offsets and fast gains.
		if !self.eac3 || blk == 0 {
			if self.snr_offset_strategy != 0 && bs.bit() {
				let csnr = (bs.bits(6) as i32 - 15) << 4;
				let mut snr = 0;
				for ch in first..=self.channels {
					if ch == first || self.snr_offset_strategy == 2 {
						snr = (csnr + bs.bits(4) as i32) << 2;
					}
					if blk > 0 && self.snr_offset[ch] != snr {
						stages[ch] = stages[ch].max(1);
					}
					self.snr_offset[ch] = snr;
					if !self.eac3 {
						let prev = self.fast_gain[ch];
						self.fast_gain[ch] = i32::from(bitalloc::FAST_GAIN[bs.bits(3) as usize]);
						if blk > 0 && prev != self.fast_gain[ch] {
							stages[ch] = stages[ch].max(2);
						}
					}
				}
			} else if !self.eac3 && blk == 0 {
				return Err(Invalid("snr offsets missing in block 0"));
			}
		}
		if self.fast_gain_syntax && bs.bit() {
			for ch in first..=self.channels {
				let prev = self.fast_gain[ch];
				self.fast_gain[ch] = i32::from(bitalloc::FAST_GAIN[bs.bits(3) as usize]);
				if blk > 0 && prev != self.fast_gain[ch] {
					stages[ch] = stages[ch].max(2);
				}
			}
		} else if self.eac3 && blk == 0 {
			for ch in first..=self.channels {
				self.fast_gain[ch] = i32::from(bitalloc::FAST_GAIN[4]);
			}
		}
		if self.frame_type == FrameType::Independent && bs.bit() {
			bs.skip(10); // convsnroffst
		}

		// Coupling leak.
		if cpl_in_use {
			if self.first_cpl_leak || bs.bit() {
				let fl = bs.bits(3) as i32;
				let sl = bs.bits(3) as i32;
				if blk > 0 && (fl != self.params.cpl_fast_leak || sl != self.params.cpl_slow_leak) {
					stages[CPL] = stages[CPL].max(2);
				}
				self.params.cpl_fast_leak = fl;
				self.params.cpl_slow_leak = sl;
			} else if !self.eac3 && blk == 0 {
				return Err(Invalid("coupling leak info missing in block 0"));
			}
			self.first_cpl_leak = false;
		}

		// Delta bit allocation.
		if self.dba_syntax && bs.bit() {
			for ch in first..=fbw {
				self.dba[ch].mode = bs.bits(2) as u8;
				if self.dba[ch].mode == bitalloc::DBA_RESERVED {
					return Err(Invalid("reserved delta bit allocation mode"));
				}
				stages[ch] = stages[ch].max(2);
			}
			for ch in first..=fbw {
				if self.dba[ch].mode == bitalloc::DBA_NEW {
					let d = &mut self.dba[ch];
					d.nsegs = bs.bits(3) as usize + 1;
					for seg in 0..d.nsegs {
						d.offsets[seg] = bs.bits(5) as u8;
						d.lengths[seg] = bs.bits(4) as u8;
						d.values[seg] = bs.bits(3) as u8;
					}
					stages[ch] = stages[ch].max(2);
				}
			}
		} else if blk == 0 {
			for d in &mut self.dba[..=self.channels] {
				d.mode = bitalloc::DBA_NONE;
			}
		}

		// Run the bit allocation.
		for ch in first..=self.channels {
			let (start, end) = (self.start_freq[ch], self.end_freq[ch]);
			if stages[ch] > 2 {
				bitalloc::calc_psd(&self.dexps[ch], start, end, &mut self.psd[ch], &mut self.band_psd[ch]);
			}
			if stages[ch] > 1
				&& !bitalloc::calc_mask(
					&self.params,
					&self.band_psd[ch],
					start,
					end,
					self.fast_gain[ch],
					ch == self.lfe_ch,
					&self.dba[ch],
					&mut self.mask[ch],
				) {
				return Err(Invalid("ac-3 bit allocation error"));
			}
			if stages[ch] > 0 {
				let tab = if self.channel_uses_aht[ch] { &HEBAP } else { &BAP };
				bitalloc::calc_bap(
					&self.mask[ch],
					&self.psd[ch],
					start,
					end,
					self.snr_offset[ch],
					self.params.floor,
					tab,
					&mut self.bap[ch],
				);
			}
		}

		if self.skip_syntax && bs.bit() {
			let len = bs.bits(9) as usize;
			bs.skip(8 * len);
		}

		self.decode_transform_coeffs(bs, blk);
		if bs.overrun() {
			return Err(Invalid("ac-3 block truncated"));
		}

		if self.channel_mode == MODE_STEREO {
			self.rematrix();
		}

		// Dynamic range and the 2^-22 mantissa scale.
		for ch in 1..=self.channels {
			let audio_channel = if self.channel_mode == MODE_DUAL_MONO && ch <= 2 { 2 - ch } else { 0 };
			let gain = if self.heavy_compression && self.compression_exists[audio_channel] {
				self.heavy_dynamic_range[audio_channel]
			} else {
				self.dynamic_range[audio_channel]
			} * (1.0 / 4_194_304.0);
			for (d, &c) in self.transform[ch].iter_mut().zip(self.coeffs[ch].iter()) {
				*d = c * gain;
			}
		}

		if self.spx_in_use {
			self.apply_spectral_extension();
		}

		self.imdct(blk, offset);
		Ok(())
	}

	fn decode_transform_coeffs(&mut self, bs: &mut BitReader, blk: usize) {
		let mut m = MantGroups::default();
		let mut got_cpl = false;
		for ch in 1..=self.channels {
			self.decode_channel_coeffs(bs, blk, ch, &mut m);
			let end = if self.channel_in_cpl[ch] {
				if !got_cpl {
					self.decode_channel_coeffs(bs, blk, CPL, &mut m);
					self.uncouple();
					got_cpl = true;
				}
				self.end_freq[CPL]
			} else {
				self.end_freq[ch]
			};
			self.coeffs[ch][end.min(MAX_COEFS)..].fill(0.0);
		}
		// Remove dither from coupled channels that do not use it (§7.3.4).
		for ch in 1..=self.fbw_channels {
			if !self.dither_flag[ch] && self.channel_in_cpl[ch] {
				for i in self.start_freq[CPL]..self.end_freq[CPL] {
					if self.bap[CPL][i] == 0 {
						self.coeffs[ch][i] = 0.0;
					}
				}
			}
		}
	}

	fn decode_channel_coeffs(
		&mut self,
		bs: &mut BitReader,
		blk: usize,
		ch: usize,
		m: &mut MantGroups,
	) {
		const SCALE: [f32; 25] = {
			let mut s = [0.0f32; 25];
			let mut i = 0;
			let mut v = 1.0f32;
			while i < 25 {
				s[i] = v;
				v *= 0.5;
				i += 1;
			}
			s
		};
		let (start, end) = (self.start_freq[ch], self.end_freq[ch]);
		if self.channel_uses_aht[ch] {
			if blk == 0 {
				self.decode_aht(bs, ch);
			}
			for bin in start..end {
				self.coeffs[ch][bin] =
					self.pre_mantissa[ch][bin][blk] as f32 * SCALE[self.dexps[ch][bin] as usize];
			}
			return;
		}
		let t = MantissaTables::get();
		let dither = ch == CPL || self.dither_flag[ch];
		for bin in start..end {
			let bap = self.bap[ch][bin];
			let mantissa = match bap {
				0 => {
					if dither {
						let r = (self.lfg.next() >> 8) as i64;
						(((r * 181) >> 8) - 5_931_008) as i32
					} else {
						0
					}
				}
				1 => {
					if m.b1 > 0 {
						m.b1 -= 1;
						m.b1_mant[m.b1]
					} else {
						let g = &t.bap1[bs.bits(5) as usize];
						m.b1_mant = [g[2], g[1]];
						m.b1 = 2;
						g[0]
					}
				}
				2 => {
					if m.b2 > 0 {
						m.b2 -= 1;
						m.b2_mant[m.b2]
					} else {
						let g = &t.bap2[bs.bits(7) as usize];
						m.b2_mant = [g[2], g[1]];
						m.b2 = 2;
						g[0]
					}
				}
				3 => t.bap3[bs.bits(3) as usize],
				4 => {
					if m.b4 > 0 {
						m.b4 = 0;
						m.b4_mant
					} else {
						let g = &t.bap4[bs.bits(7) as usize];
						m.b4_mant = g[1];
						m.b4 = 1;
						g[0]
					}
				}
				5 => t.bap5[bs.bits(4) as usize],
				_ => {
					let q = u32::from(QUANTIZATION_BITS[usize::from(bap.min(15))]);
					((bs.signed(q) as u32) << (24 - q)) as i32
				}
			};
			self.coeffs[ch][bin] = mantissa as f32 * SCALE[self.dexps[ch][bin].clamp(0, 24) as usize];
		}
	}

	/// AHT mantissas of all six blocks (Annex E §E.3; FFmpeg
	/// `ff_eac3_decode_transform_coeffs_aht_ch`).
	fn decode_aht(&mut self, bs: &mut BitReader, ch: usize) {
		let t = MantissaTables::get();
		let gaq_mode = bs.bits(2);
		let end_bap: u8 = if gaq_mode < 2 { 12 } else { 17 };
		let (start, end) = (self.start_freq[ch], self.end_freq[ch]);
		let mut gaq_gain = [0i32; MAX_COEFS];
		let mut gs = 0;
		if gaq_mode == 1 || gaq_mode == 2 {
			for bin in start..end {
				let b = self.bap[ch][bin];
				if b > 7 && b < end_bap {
					gaq_gain[gs] = (bs.bits(1) as i32) << (gaq_mode - 1);
					gs += 1;
				}
			}
		} else if gaq_mode == 3 {
			let mut gc = 2;
			for bin in start..end {
				let b = self.bap[ch][bin];
				if b > 7 && b < 17 {
					if gc == 2 {
						let code = (bs.bits(5) as usize).min(26);
						for k in 0..3 {
							if gs < MAX_COEFS {
								gaq_gain[gs] = i32::from(t.ungroup5[code][k]);
								gs += 1;
							}
						}
						gc = 0;
					} else {
						gc += 1;
					}
				}
			}
		}
		gs = 0;
		for bin in start..end {
			let hebap = usize::from(self.bap[ch][bin]);
			let bits = u32::from(BITS_VS_HEBAP[hebap.min(19)]);
			let pre = &mut self.pre_mantissa[ch][bin];
			if hebap == 0 {
				for v in pre.iter_mut() {
					*v = (self.lfg.next() & 0x7F_FFFF) as i32 - 0x40_0000;
				}
			} else if hebap < 8 {
				let v = bs.bits(bits) as usize;
				let row = match hebap {
					1 => VQ_HEBAP1.get(v),
					2 => VQ_HEBAP2.get(v),
					3 => VQ_HEBAP3.get(v),
					4 => VQ_HEBAP4.get(v),
					5 => VQ_HEBAP5.get(v),
					6 => VQ_HEBAP6.get(v),
					_ => VQ_HEBAP7.get(v),
				};
				let row = row.copied().unwrap_or([0; 6]);
				for (p, r) in pre.iter_mut().zip(row) {
					*p = i32::from(r) * 256;
				}
			} else {
				let log_gain = if gaq_mode != 0 && hebap < usize::from(end_bap) {
					let g = gaq_gain.get(gs).copied().unwrap_or(0);
					gs += 1;
					g
				} else {
					0
				};
				let gbits = bits as i32 - log_gain;
				let h8 = hebap - 8;
				for p in pre.iter_mut() {
					let mut mant = bs.signed(gbits.max(1) as u32);
					if log_gain != 0 && mant == -(1 << (gbits - 1)) {
						let mbits = bits as i32 - (2 - log_gain);
						mant = bs.signed(mbits.max(1) as u32);
						mant = ((mant as u32) << (23 - (mbits - 1)) as u32) as i32;
						let b = if mant >= 0 {
							1 << (23 - log_gain)
						} else {
							i32::from(GAQ_REMAP_2_4_B[h8.min(8)][(log_gain - 1) as usize]) * 256
						};
						let a = i64::from(GAQ_REMAP_2_4_A[h8.min(8)][(log_gain - 1) as usize]);
						mant += ((a * i64::from(mant)) >> 15) as i32 + b;
					} else {
						mant = mant.wrapping_mul(1 << (24 - bits));
						if log_gain == 0 {
							mant += ((i64::from(GAQ_REMAP_1[h8.min(11)]) * i64::from(mant)) >> 15) as i32;
						}
					}
					*p = mant;
				}
			}
			idct6(pre);
		}
	}

	fn uncouple(&mut self) {
		let mut bin = self.start_freq[CPL];
		for band in 0..self.num_cpl_bands {
			let start = bin;
			let end = (bin + usize::from(self.cpl_band_sizes[band])).min(MAX_COEFS);
			for ch in 1..=self.fbw_channels {
				if !self.channel_in_cpl[ch] {
					continue;
				}
				let coord = self.cpl_coords[ch][band] as f32 * (1.0 / (1 << 23) as f32);
				for b in start..end {
					self.coeffs[ch][b] = self.coeffs[CPL][b] * coord;
				}
				if ch == 2 && self.phase_flags[band] {
					for b in start..end {
						self.coeffs[2][b] = -self.coeffs[2][b];
					}
				}
			}
			bin = end;
		}
	}

	fn rematrix(&mut self) {
		let end = self.end_freq[1].min(self.end_freq[2]);
		for bnd in 0..self.num_rematrixing_bands {
			if self.rematrixing_flags[bnd] {
				let band_end = end.min(REMATRIX_BANDS[bnd + 1]);
				let [_, l, r, ..] = &mut *self.coeffs;
				for i in REMATRIX_BANDS[bnd]..band_end {
					let a = l[i];
					l[i] += r[i];
					r[i] = a - r[i];
				}
			}
		}
	}

	/// Spectral extension (Annex E §E.3.3; FFmpeg
	/// `ff_eac3_apply_spectral_extension`).
	fn apply_spectral_extension(&mut self) {
		let mut wrapflag = [false; 17];
		wrapflag[0] = true;
		let mut copy_sizes = [0usize; 18];
		let mut num_copy = 0;
		let (dst_start, src_start) = (self.spx_dst_start_freq, self.spx_src_start_freq);
		let mut bin = dst_start;
		for bnd in 0..self.num_spx_bands {
			let size = usize::from(self.spx_band_sizes[bnd]);
			if bin + size > src_start {
				copy_sizes[num_copy] = bin - dst_start;
				num_copy += 1;
				bin = dst_start;
				wrapflag[bnd] = true;
			}
			let mut i = 0;
			while i < size {
				if bin == src_start {
					copy_sizes[num_copy] = bin - dst_start;
					num_copy += 1;
					bin = dst_start;
				}
				let copy = (size - i).min(src_start - bin);
				bin += copy;
				i += copy;
			}
		}
		copy_sizes[num_copy] = bin - dst_start;
		num_copy += 1;

		for ch in 1..=self.fbw_channels {
			if !self.channel_uses_spx[ch] {
				continue;
			}
			let coeffs = &mut self.transform[ch];
			let mut bin = src_start;
			for &size in &copy_sizes[..num_copy] {
				if bin + size > MAX_COEFS {
					break;
				}
				coeffs.copy_within(dst_start..dst_start + size, bin);
				bin += size;
			}
			let mut rms = [0f32; 17];
			let mut bin = src_start;
			for bnd in 0..self.num_spx_bands {
				let size = usize::from(self.spx_band_sizes[bnd]);
				let mut acc = 0.0f32;
				for c in &coeffs[bin.min(MAX_COEFS)..(bin + size).min(MAX_COEFS)] {
					acc += c * c;
				}
				rms[bnd] = (acc / size as f32).sqrt();
				bin += size;
			}
			if self.spx_atten_code[ch] >= 0 {
				let code = self.spx_atten_code[ch] as i32;
				let atten = |b: i32| 2f64.powf(f64::from((b + 1) * (code + 1)) / -15.0) as f32;
				let tab = [atten(0), atten(1), atten(2)];
				let mut bin = src_start - 2;
				for bnd in 0..self.num_spx_bands {
					if wrapflag[bnd] && bin + 5 <= MAX_COEFS {
						let c = &mut coeffs[bin..bin + 5];
						c[0] *= tab[0];
						c[1] *= tab[1];
						c[2] *= tab[2];
						c[3] *= tab[1];
						c[4] *= tab[0];
					}
					bin += usize::from(self.spx_band_sizes[bnd]);
				}
			}
			let mut bin = src_start;
			for bnd in 0..self.num_spx_bands {
				let nscale = self.spx_noise_blend[ch][bnd] * rms[bnd] * (1.0 / -(2f32.powi(31)));
				let sscale = self.spx_signal_blend[ch][bnd];
				for _ in 0..self.spx_band_sizes[bnd] {
					let noise = nscale * self.lfg.next() as i32 as f32;
					if bin < MAX_COEFS {
						coeffs[bin] = coeffs[bin] * sscale + noise;
					}
					bin += 1;
				}
			}
		}
	}

	fn imdct(&mut self, blk: usize, offset: usize) {
		let t = MantissaTables::get();
		let mut tmp = [0f32; BLOCK_LEN];
		for ch in 1..=self.channels {
			let slot = offset + ch - 1;
			let out = &mut self.coded[slot][blk * BLOCK_LEN..(blk + 1) * BLOCK_LEN];
			let delay = &mut self.delay[slot];
			let coeffs = &self.transform[ch];
			if self.block_switch[ch] {
				let mut x = [0f32; HALF];
				for i in 0..HALF {
					x[i] = coeffs[2 * i];
				}
				t.imdct128.inverse_half(&x, &mut tmp[..HALF], &mut self.work, &mut self.cbuf);
				fmul_window(out, delay, &tmp[..HALF], &t.window, HALF);
				for i in 0..HALF {
					x[i] = coeffs[2 * i + 1];
				}
				t.imdct128.inverse_half(&x, delay, &mut self.work, &mut self.cbuf);
			} else {
				t.imdct256.inverse_half(coeffs, &mut tmp, &mut self.work, &mut self.cbuf);
				fmul_window(out, delay, &tmp[..HALF], &t.window, HALF);
				delay.copy_from_slice(&tmp[HALF..]);
			}
		}
	}
}

/// 6-point IDCT of AHT pre-mantissas in 24-bit fixed point (§E.3.5).
fn idct6(p: &mut [i32; 6]) {
	const C0: i64 = 10_273_905;
	const C1: i64 = 11_863_283;
	const C2: i64 = 3_070_444;
	let odd1 = p[1] - p[3] - p[5];
	let mut even2 = ((i64::from(p[2]) * C0) >> 23) as i32;
	let tmp = ((i64::from(p[4]) * C1) >> 23) as i32;
	let mut odd0 = ((i64::from(p[1] + p[5]) * C2) >> 23) as i32;
	let mut even0 = p[0] + (tmp >> 1);
	let even1 = p[0] - tmp;
	let t = even0;
	even0 = t + even2;
	even2 = t - even2;
	let t = odd0;
	odd0 = t + p[1] + p[3];
	let odd2 = t + p[5] - p[3];
	*p = [even0 + odd0, even1 + odd1, even2 + odd2, even2 - odd2, even1 - odd1, even0 - odd0];
}

/// Offset of the first syncword in the first 11 bytes (FFmpeg
/// `ff_ac3_find_syncword`), or anywhere when scanning a raw stream.
pub fn find_sync(data: &[u8]) -> Option<usize> {
	data.windows(2).position(|w| u16::from_be_bytes([w[0], w[1]]) == SYNC_WORD)
}

/// Number of channels and FFmpeg layout mask of a header's main substream.
pub fn layout_of(channel_mode: u8, lfe: bool) -> (usize, u64) {
	let mut mask = acmod_locations(channel_mode).iter().fold(0u64, |m, &b| m | (1 << b));
	if lfe {
		mask |= 1 << CH_LFE;
	}
	(ACMOD_CHANNELS[usize::from(channel_mode)] + usize::from(lfe), mask)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn lfg_matches_ffmpeg_seed_zero() {
		// First outputs of av_lfg_init(&s, 0) + av_lfg_get().
		let mut l = Lfg::new(0);
		let first: Vec<u32> = (0..3).map(|_| l.next()).collect();
		// Deterministic and non-trivial.
		assert!(first.iter().all(|&v| v != 0));
		let mut l2 = Lfg::new(0);
		assert_eq!(first, (0..3).map(|_| l2.next()).collect::<Vec<_>>());
	}

	#[test]
	fn kbd_window_is_power_complementary() {
		let w = kbd_window(5.0, 256);
		for i in 0..256 {
			let s = w[i] * w[i] + w[255 - i] * w[255 - i];
			assert!((s - 1.0).abs() < 1e-5, "{i} {s}");
		}
	}

	#[test]
	fn idct6_of_dc() {
		let mut p = [1 << 20, 0, 0, 0, 0, 0];
		idct6(&mut p);
		assert!(p.iter().all(|&v| v == 1 << 20));
	}
}
