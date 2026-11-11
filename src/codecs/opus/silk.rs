//! SILK decoder (RFC 6716 §4.2), ported from libopus 1.4 (BSD-3-Clause):
//! `silk/dec_API.c`, `decode_frame.c`, `decode_indices.c`,
//! `decode_pulses.c`, `shell_coder.c`, `code_signs.c`,
//! `decode_parameters.c`, `gain_quant.c`, `NLSF_decode.c`, `NLSF2A.c`,
//! `NLSF_stabilize.c`, `LPC_inv_pred_gain.c`, `LPC_fit.c`, `bwexpander_32.c`,
//! `decode_pitch.c`, `decode_core.c`, `stereo_*.c`, `resampler*.c`, and the
//! loss handling: `PLC.c` (packet-loss concealment), `CNG.c` (comfort
//! noise) and the LBRR (in-band FEC) path of `dec_API.c`.
//!
//! SILK is specified in fixed point, so this port is bit-exact with libopus:
//! the fixed-point helpers below reproduce `SigProc_FIX.h` / `Inlines.h`
//! exactly (with wrapping where the reference relies on two's complement).
//!
//! Loss handling follows libopus. Every decoded frame updates the PLC state
//! (last pitch lag, LTP and LPC filters, gains) and, during silence, the CNG
//! estimate (smoothed NLSFs, gain and an excitation buffer). A lost frame is
//! extrapolated from the last excitation with a decaying pitch pulse train
//! and attenuated noise ([`plc_conceal`]); comfort noise shaped by the
//! smoothed spectrum is added on top ([`cng`]); and the first good frame
//! after a loss is faded in if it is louder than the concealment
//! ([`plc_glue_frames`]). With [`SilkMode::Lbrr`] the low-bitrate
//! redundant copy of the previous packet, carried at the start of the
//! current one, is decoded instead of the packet itself.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::range::RangeDecoder;
use super::tables::*;

const MAX_NB_SUBFR: usize = 4;
const MAX_LPC_ORDER: usize = 16;
const LTP_ORDER: usize = 5;
const MAX_FRAME_LENGTH: usize = 320;
const MAX_SUB_FRAME_LENGTH: usize = 80;
const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const N_LEVELS_QGAIN: i32 = 64;
const MIN_QGAIN_DB: i32 = 2;
const MAX_QGAIN_DB: i32 = 88;
const MIN_DELTA_GAIN_QUANT: i32 = -4;
const MAX_DELTA_GAIN_QUANT: i32 = 36;
const SHELL_CODEC_FRAME_LENGTH: usize = 16;
const SILK_MAX_PULSES: usize = 16;
const N_RATE_LEVELS: usize = 10;
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
const STEREO_INTERP_LEN_MS: i32 = 8;
const LTP_MEM_LENGTH_MS: usize = 20;
const SUB_FRAME_LENGTH_MS: usize = 5;
const PE_MIN_LAG_MS: i32 = 2;
const PE_MAX_LAG_MS: i32 = 18;
const MAX_LPC_STABILIZE_ITERATIONS: usize = 16;
const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
const TYPE_VOICED: i32 = 2;
const CODE_INDEPENDENTLY: i32 = 0;
const CODE_INDEPENDENTLY_NO_LTP_SCALING: i32 = 1;
const CODE_CONDITIONALLY: i32 = 2;
const RESAMPLER_ORDER_FIR_12: usize = 8;
const RESAMPLER_MAX_BATCH_SIZE_MS: i32 = 10;
const UP2_HQ_0: [i32; 3] = [1746, 14986, 39083 - 65536];
const UP2_HQ_1: [i32; 3] = [6854, 25769, 55542 - 65536];

// PLC.h / PLC.c.
const BWE_COEF_Q16: i32 = 64881; // SILK_FIX_CONST(0.99, 16)
const BWE_AFTER_LOSS_Q16: i32 = 63570; // define.h
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469; // 0.7
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565; // 0.95
const MAX_PITCH_LAG_MS: i32 = 18;
const RAND_BUF_SIZE: usize = 128;
const RAND_BUF_MASK: i32 = RAND_BUF_SIZE as i32 - 1;
const LOG2_INV_LPC_GAIN_HIGH_THRES: u32 = 3; // 8 dB LPC gain
const LOG2_INV_LPC_GAIN_LOW_THRES: u32 = 8; // 24 dB LPC gain
const PITCH_DRIFT_FAC_Q16: i32 = 655; // 0.01
const NB_ATT: usize = 2;
const HARM_ATT_Q15: [i32; NB_ATT] = [32440, 31130]; // 0.99, 0.95
const PLC_RAND_ATTENUATE_V_Q15: [i32; NB_ATT] = [31130, 26214]; // 0.95, 0.8
const PLC_RAND_ATTENUATE_UV_Q15: [i32; NB_ATT] = [32440, 29491]; // 0.99, 0.9

// define.h (comfort noise).
const CNG_BUF_MASK_MAX: i32 = 255;
const CNG_GAIN_SMTH_Q16: i32 = 4634; // 0.25^(1/4)
const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46396; // -3 dB
const CNG_NLSF_SMTH_Q16: i32 = 16348; // 0.25
const CNG_RAND_SEED: i32 = 3_176_576;

/// How [`SilkDecoder::decode`] treats the current frame (libopus'
/// `lostFlag`: `FLAG_DECODE_NORMAL`, `FLAG_PACKET_LOST`, `FLAG_DECODE_LBRR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkMode {
	/// Decode the frame from the packet.
	Normal,
	/// The packet is missing: conceal the frame.
	Lost,
	/// Decode the LBRR (FEC) copy of the previous packet's frame from the
	/// current packet; frames without LBRR data are concealed.
	Lbrr,
}

// --- Fixed-point helpers (SigProc_FIX.h, macros.h, Inlines.h). ---

#[inline]
fn smulbb(a: i32, b: i32) -> i32 {
	i32::from(a as i16) * i32::from(b as i16)
}
#[inline]
fn smlabb(a: i32, b: i32, c: i32) -> i32 {
	a.wrapping_add(smulbb(b, c))
}
#[inline]
fn smulwb(a: i32, b: i32) -> i32 {
	((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}
#[inline]
fn smlawb(a: i32, b: i32, c: i32) -> i32 {
	a.wrapping_add(smulwb(b, c))
}
#[inline]
fn smulww(a: i32, b: i32) -> i32 {
	((i64::from(a) * i64::from(b)) >> 16) as i32
}
#[inline]
fn smlaww(a: i32, b: i32, c: i32) -> i32 {
	a.wrapping_add(smulww(b, c))
}
#[inline]
fn smmul(a: i32, b: i32) -> i32 {
	((i64::from(a) * i64::from(b)) >> 32) as i32
}
#[inline]
fn rshift_round(a: i32, shift: u32) -> i32 {
	if shift == 1 { (a >> 1) + (a & 1) } else { ((a >> (shift - 1)) + 1) >> 1 }
}
#[inline]
fn rshift_round64(a: i64, shift: u32) -> i64 {
	if shift == 1 { (a >> 1) + (a & 1) } else { ((a >> (shift - 1)) + 1) >> 1 }
}
#[inline]
fn sat16(a: i32) -> i32 {
	a.clamp(i32::from(i16::MIN), i32::from(i16::MAX))
}
#[inline]
fn add_sat32(a: i32, b: i32) -> i32 {
	a.saturating_add(b)
}
#[inline]
fn sub_sat32(a: i32, b: i32) -> i32 {
	a.saturating_sub(b)
}
#[inline]
fn lshift_sat32(a: i32, shift: u32) -> i32 {
	a.clamp(i32::MIN >> shift, i32::MAX >> shift) << shift
}
#[inline]
fn limit(a: i32, l1: i32, l2: i32) -> i32 {
	if l1 > l2 { a.clamp(l2, l1) } else { a.clamp(l1, l2) }
}
#[inline]
fn clz32(a: i32) -> i32 {
	(a as u32).leading_zeros() as i32
}
#[inline]
fn silk_rand(seed: i32) -> i32 {
	907_633_515i32.wrapping_add(seed.wrapping_mul(196_314_165))
}

/// `silk_SQRT_APPROX`: approximate square root in the input's Q domain / 2.
fn sqrt_approx(x: i32) -> i32 {
	if x <= 0 {
		return 0;
	}
	// silk_CLZ_FRAC: leading zeros and the 7 bits after the leading one.
	let lz = clz32(x);
	let rot = 24 - lz;
	let rotated = if rot >= 0 {
		(x as u32).rotate_right(rot as u32)
	} else {
		(x as u32).rotate_left((-rot) as u32)
	};
	let frac_q7 = (rotated & 0x7F) as i32;
	let mut y = if lz & 1 != 0 { 32768 } else { 46214 }; // 46214 = sqrt(2) * 32768
	y >>= lz >> 1;
	smlawb(y, y, smulbb(213, frac_q7))
}

/// `silk_sum_sqr_shift`: energy of `x` and the right shift applied to keep
/// it within 32 bits with two bits of headroom.
fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
	let len = x.len();
	let pass = |init: i32, shift: i32| -> i32 {
		let shift = shift as u32;
		let mut nrg = init as u32;
		let mut i = 0;
		while i + 1 < len {
			let t = (smulbb(i32::from(x[i]), i32::from(x[i])) as u32)
				.wrapping_add(smulbb(i32::from(x[i + 1]), i32::from(x[i + 1])) as u32);
			nrg = nrg.wrapping_add(t >> shift);
			i += 2;
		}
		if i < len {
			nrg = nrg.wrapping_add((smulbb(i32::from(x[i]), i32::from(x[i])) as u32) >> shift);
		}
		nrg as i32
	};
	let first_shift = 31 - clz32(len as i32);
	let nrg = pass(len as i32, first_shift);
	let shift = (first_shift + 3 - clz32(nrg)).max(0);
	(pass(0, shift), shift)
}

/// `silk_bwexpander`: chirps a 16-bit AR filter in place.
fn bwexpander(ar: &mut [i16], mut chirp_q16: i32) {
	let Some(last) = ar.len().checked_sub(1) else { return };
	let chirp_minus_one_q16 = chirp_q16 - 65536;
	for v in &mut ar[..last] {
		*v = rshift_round(chirp_q16.wrapping_mul(i32::from(*v)), 16) as i16;
		chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
	}
	ar[last] = rshift_round(chirp_q16.wrapping_mul(i32::from(ar[last])), 16) as i16;
}

fn div32_var_q(a32: i32, b32: i32, qres: i32) -> i32 {
	let a_headrm = clz32(a32.wrapping_abs()) - 1;
	let mut a32_nrm = a32.wrapping_shl(a_headrm as u32);
	let b_headrm = clz32(b32.wrapping_abs()) - 1;
	let b32_nrm = b32.wrapping_shl(b_headrm as u32);
	let denom = b32_nrm >> 16;
	let b32_inv = if denom == 0 { i32::MAX } else { (i32::MAX >> 2) / denom };
	let mut result = smulwb(a32_nrm, b32_inv);
	a32_nrm = a32_nrm.wrapping_sub(smmul(b32_nrm, result).wrapping_shl(3));
	result = smlawb(result, a32_nrm, b32_inv);
	let lshift = 29 + a_headrm - b_headrm - qres;
	if lshift < 0 {
		lshift_sat32(result, (-lshift) as u32)
	} else if lshift < 32 {
		result >> lshift
	} else {
		0
	}
}

fn inverse32_var_q(b32: i32, qres: i32) -> i32 {
	let b_headrm = clz32(b32.wrapping_abs()) - 1;
	let b32_nrm = b32.wrapping_shl(b_headrm as u32);
	let denom = b32_nrm >> 16;
	let b32_inv = if denom == 0 { i32::MAX } else { (i32::MAX >> 2) / denom };
	let mut result = b32_inv.wrapping_shl(16);
	let err_q32 = ((1i32 << 29).wrapping_sub(smulwb(b32_nrm, b32_inv))).wrapping_shl(3);
	result = smlaww(result, err_q32, b32_inv);
	let lshift = 61 - b_headrm - qres;
	if lshift <= 0 {
		lshift_sat32(result, (-lshift) as u32)
	} else if lshift < 32 {
		result >> lshift
	} else {
		0
	}
}

fn log2lin(in_log_q7: i32) -> i32 {
	if in_log_q7 < 0 {
		return 0;
	}
	if in_log_q7 >= 3967 {
		return i32::MAX;
	}
	let out = 1i32 << (in_log_q7 >> 7);
	let frac_q7 = in_log_q7 & 0x7F;
	let poly = smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174);
	if in_log_q7 < 2048 {
		out.wrapping_add(out.wrapping_mul(poly) >> 7)
	} else {
		out.wrapping_add((out >> 7).wrapping_mul(poly))
	}
}

// --- NLSF codebooks. ---

struct NlsfCb {
	n_vectors: usize,
	order: usize,
	quant_step_size_q16: i32,
	cb1_nlsf_q8: &'static [u8],
	cb1_wght_q9: &'static [i16],
	cb1_icdf: &'static [u8],
	pred_q8: &'static [u8],
	ec_sel: &'static [u8],
	ec_icdf: &'static [u8],
	delta_min_q15: &'static [i16],
}

static NLSF_CB_NB_MB: NlsfCb = NlsfCb {
	n_vectors: 32,
	order: 10,
	quant_step_size_q16: 11796, // SILK_FIX_CONST(0.18, 16)
	cb1_nlsf_q8: &NLSF_CB1_NB_MB_Q8,
	cb1_wght_q9: &NLSF_CB1_WGHT_Q9,
	cb1_icdf: &NLSF_CB1_I_CDF_NB_MB,
	pred_q8: &NLSF_PRED_NB_MB_Q8,
	ec_sel: &NLSF_CB2_SELECT_NB_MB,
	ec_icdf: &NLSF_CB2_I_CDF_NB_MB,
	delta_min_q15: &NLSF_DELTA_MIN_NB_MB_Q15,
};

static NLSF_CB_WB: NlsfCb = NlsfCb {
	n_vectors: 32,
	order: 16,
	quant_step_size_q16: 9830, // SILK_FIX_CONST(0.15, 16)
	cb1_nlsf_q8: &NLSF_CB1_WB_Q8,
	cb1_wght_q9: &NLSF_CB1_WB_WGHT_Q9,
	cb1_icdf: &NLSF_CB1_I_CDF_WB,
	pred_q8: &NLSF_PRED_WB_Q8,
	ec_sel: &NLSF_CB2_SELECT_WB,
	ec_icdf: &NLSF_CB2_I_CDF_WB,
	delta_min_q15: &NLSF_DELTA_MIN_WB_Q15,
};

fn ltp_gain_icdf(per: usize) -> &'static [u8] {
	match per {
		0 => &LTP_GAIN_I_CDF_0,
		1 => &LTP_GAIN_I_CDF_1,
		_ => &LTP_GAIN_I_CDF_2,
	}
}

fn ltp_vq(per: usize) -> &'static [i8] {
	match per {
		0 => &LTP_GAIN_VQ_0,
		1 => &LTP_GAIN_VQ_1,
		_ => &LTP_GAIN_VQ_2,
	}
}

// --- Resampler (only the paths used when decoding to 48 kHz). ---

#[derive(Debug, Clone, Default)]
struct Resampler {
	s_iir: [i32; 6],
	s_fir: [i16; RESAMPLER_ORDER_FIR_12],
	delay_buf: [i16; 32],
	/// 0 copy, 1 up2 HQ, 2 IIR/FIR.
	function: u8,
	batch_size: i32,
	inv_ratio_q16: i32,
	fs_in_khz: i32,
	fs_out_khz: i32,
	input_delay: i32,
}

fn rate_id(r: i32) -> usize {
	((((r >> 12) - i32::from(r > 16000)) >> i32::from(r > 24000)) - 1) as usize
}

impl Resampler {
	fn new(fs_in: i32, fs_out: i32) -> Self {
		const DELAY_MATRIX_DEC: [[i32; 5]; 3] = [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];
		let mut s = Self {
			input_delay: DELAY_MATRIX_DEC[rate_id(fs_in)][rate_id(fs_out)],
			fs_in_khz: fs_in / 1000,
			fs_out_khz: fs_out / 1000,
			..Default::default()
		};
		s.batch_size = s.fs_in_khz * RESAMPLER_MAX_BATCH_SIZE_MS;
		let mut up2x = 0;
		if fs_out > fs_in {
			if fs_out == fs_in * 2 {
				s.function = 1;
			} else {
				s.function = 2;
				up2x = 1;
			}
		} else {
			s.function = 0;
		}
		s.inv_ratio_q16 = ((fs_in << (14 + up2x)) / fs_out) << 2;
		while smulww(s.inv_ratio_q16, fs_out) < (fs_in << up2x) {
			s.inv_ratio_q16 += 1;
		}
		s
	}

	fn up2_hq(st: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
		for (k, &x) in input.iter().enumerate() {
			let in32 = i32::from(x) << 10;
			let mut y = in32.wrapping_sub(st[0]);
			let mut xv = smulwb(y, UP2_HQ_0[0]);
			let mut out32_1 = st[0].wrapping_add(xv);
			st[0] = in32.wrapping_add(xv);
			y = out32_1.wrapping_sub(st[1]);
			xv = smulwb(y, UP2_HQ_0[1]);
			let mut out32_2 = st[1].wrapping_add(xv);
			st[1] = out32_1.wrapping_add(xv);
			y = out32_2.wrapping_sub(st[2]);
			xv = smlawb(y, y, UP2_HQ_0[2]);
			out32_1 = st[2].wrapping_add(xv);
			st[2] = out32_2.wrapping_add(xv);
			out[2 * k] = sat16(rshift_round(out32_1, 10)) as i16;

			y = in32.wrapping_sub(st[3]);
			xv = smulwb(y, UP2_HQ_1[0]);
			out32_1 = st[3].wrapping_add(xv);
			st[3] = in32.wrapping_add(xv);
			y = out32_1.wrapping_sub(st[4]);
			xv = smulwb(y, UP2_HQ_1[1]);
			out32_2 = st[4].wrapping_add(xv);
			st[4] = out32_1.wrapping_add(xv);
			y = out32_2.wrapping_sub(st[5]);
			xv = smlawb(y, y, UP2_HQ_1[2]);
			out32_1 = st[5].wrapping_add(xv);
			st[5] = out32_2.wrapping_add(xv);
			out[2 * k + 1] = sat16(rshift_round(out32_1, 10)) as i16;
		}
	}

	fn iir_fir(&mut self, out: &mut Vec<i16>, mut input: &[i16]) {
		let mut buf = vec![0i16; 2 * self.batch_size as usize + RESAMPLER_ORDER_FIR_12];
		buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&self.s_fir);
		let inc = self.inv_ratio_q16;
		let mut n_in;
		loop {
			n_in = input.len().min(self.batch_size as usize);
			Self::up2_hq(
				&mut self.s_iir,
				&mut buf[RESAMPLER_ORDER_FIR_12..RESAMPLER_ORDER_FIR_12 + 2 * n_in],
				&input[..n_in],
			);
			let max_index_q16 = (n_in as i32) << 17;
			let mut index_q16 = 0i32;
			while index_q16 < max_index_q16 {
				let table_index = smulwb(index_q16 & 0xFFFF, 12) as usize;
				let b = &buf[(index_q16 >> 16) as usize..];
				let f = |row: usize, col: usize| i32::from(RESAMPLER_FRAC_FIR_12[row * 4 + col]);
				let mut res = smulbb(i32::from(b[0]), f(table_index, 0));
				res = smlabb(res, i32::from(b[1]), f(table_index, 1));
				res = smlabb(res, i32::from(b[2]), f(table_index, 2));
				res = smlabb(res, i32::from(b[3]), f(table_index, 3));
				res = smlabb(res, i32::from(b[4]), f(11 - table_index, 3));
				res = smlabb(res, i32::from(b[5]), f(11 - table_index, 2));
				res = smlabb(res, i32::from(b[6]), f(11 - table_index, 1));
				res = smlabb(res, i32::from(b[7]), f(11 - table_index, 0));
				out.push(sat16(rshift_round(res, 15)) as i16);
				index_q16 += inc;
			}
			input = &input[n_in..];
			if !input.is_empty() {
				buf.copy_within(n_in << 1..(n_in << 1) + RESAMPLER_ORDER_FIR_12, 0);
			} else {
				break;
			}
		}
		self.s_fir.copy_from_slice(&buf[n_in << 1..(n_in << 1) + RESAMPLER_ORDER_FIR_12]);
	}

	fn run_block(&mut self, out: &mut Vec<i16>, input: &[i16]) {
		match self.function {
			1 => {
				let start = out.len();
				out.resize(start + 2 * input.len(), 0);
				Self::up2_hq(&mut self.s_iir, &mut out[start..], input);
			}
			2 => self.iir_fir(out, input),
			_ => out.extend_from_slice(input),
		}
	}

	/// `silk_resampler`: `input.len() >= fs_in_khz`.
	fn process(&mut self, input: &[i16]) -> Vec<i16> {
		let fs_in = self.fs_in_khz as usize;
		let delay = self.input_delay as usize;
		let n = fs_in - delay;
		let mut out = Vec::with_capacity(input.len() * (self.fs_out_khz as usize) / fs_in.max(1) + 4);
		self.delay_buf[delay..delay + n].copy_from_slice(&input[..n]);
		let head = self.delay_buf[..fs_in].to_vec();
		self.run_block(&mut out, &head);
		self.run_block(&mut out, &input[n..input.len() - delay]);
		self.delay_buf[..delay].copy_from_slice(&input[input.len() - delay..]);
		out
	}
}

// --- Per-channel decoder state. ---

#[derive(Debug, Clone, Default)]
struct Indices {
	gains: [i32; MAX_NB_SUBFR],
	ltp_index: [usize; MAX_NB_SUBFR],
	nlsf: [i32; MAX_LPC_ORDER + 1],
	lag_index: i32,
	contour_index: usize,
	signal_type: i32,
	quant_offset_type: i32,
	nlsf_interp_coef_q2: i32,
	per_index: usize,
	ltp_scale_index: usize,
	seed: i32,
}

#[derive(Debug, Clone)]
struct ChannelState {
	prev_gain_q16: i32,
	exc_q14: [i32; MAX_FRAME_LENGTH],
	s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
	out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
	lag_prev: i32,
	last_gain_index: i32,
	fs_khz: i32,
	fs_api_hz: i32,
	nb_subfr: usize,
	frame_length: usize,
	subfr_length: usize,
	ltp_mem_length: usize,
	lpc_order: usize,
	prev_nlsf_q15: [i16; MAX_LPC_ORDER],
	first_frame_after_reset: bool,
	wideband: bool,
	n_frames_decoded: usize,
	n_frames_per_packet: usize,
	ec_prev_signal_type: i32,
	ec_prev_lag_index: i32,
	vad_flags: [bool; 3],
	lbrr_flag: bool,
	lbrr_flags: [bool; 3],
	resampler: Resampler,
	indices: Indices,
	prev_signal_type: i32,
	/// Consecutive concealed frames (`lossCnt`).
	loss_cnt: i32,
	plc: PlcState,
	cng: CngState,
}

/// Packet-loss concealment state (`silk_PLC_struct`).
#[derive(Debug, Clone)]
struct PlcState {
	/// Pitch lag to use for voiced concealment (Q8).
	pitch_l_q8: i32,
	/// LTP filter for voiced concealment (only the centre tap is non-zero).
	ltp_coef_q14: [i16; LTP_ORDER],
	prev_lpc_q12: [i16; MAX_LPC_ORDER],
	last_frame_lost: bool,
	rand_seed: i32,
	rand_scale_q14: i16,
	/// Energy of the last concealed frame, for fading in the next good one.
	conc_energy: i32,
	conc_energy_shift: i32,
	prev_ltp_scale_q14: i16,
	prev_gain_q16: [i32; 2],
	fs_khz: i32,
	nb_subfr: usize,
	subfr_length: usize,
}

impl PlcState {
	/// The state after `silk_init_decoder` (zeroed, then `silk_PLC_Reset`
	/// with no frame length yet).
	fn new() -> Self {
		Self {
			pitch_l_q8: 0,
			ltp_coef_q14: [0; LTP_ORDER],
			prev_lpc_q12: [0; MAX_LPC_ORDER],
			last_frame_lost: false,
			rand_seed: 0,
			rand_scale_q14: 0,
			conc_energy: 0,
			conc_energy_shift: 0,
			prev_ltp_scale_q14: 0,
			prev_gain_q16: [1 << 16; 2],
			fs_khz: 0,
			nb_subfr: 2,
			subfr_length: 20,
		}
	}
}

/// Comfort-noise state (`silk_CNG_struct`).
#[derive(Debug, Clone)]
struct CngState {
	exc_buf_q14: [i32; MAX_FRAME_LENGTH],
	smth_nlsf_q15: [i16; MAX_LPC_ORDER],
	synth_state: [i32; MAX_LPC_ORDER],
	smth_gain_q16: i32,
	rand_seed: i32,
	fs_khz: i32,
}

impl CngState {
	fn new() -> Self {
		Self {
			exc_buf_q14: [0; MAX_FRAME_LENGTH],
			smth_nlsf_q15: [0; MAX_LPC_ORDER],
			synth_state: [0; MAX_LPC_ORDER],
			smth_gain_q16: 0,
			rand_seed: CNG_RAND_SEED,
			fs_khz: 0,
		}
	}
}

impl ChannelState {
	fn new() -> Self {
		Self {
			prev_gain_q16: 65536,
			exc_q14: [0; MAX_FRAME_LENGTH],
			s_lpc_q14_buf: [0; MAX_LPC_ORDER],
			out_buf: [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
			lag_prev: 0,
			last_gain_index: 0,
			fs_khz: 0,
			fs_api_hz: 0,
			nb_subfr: 0,
			frame_length: 0,
			subfr_length: 0,
			ltp_mem_length: 0,
			lpc_order: 0,
			prev_nlsf_q15: [0; MAX_LPC_ORDER],
			first_frame_after_reset: true,
			wideband: false,
			n_frames_decoded: 0,
			n_frames_per_packet: 0,
			ec_prev_signal_type: 0,
			ec_prev_lag_index: 0,
			vad_flags: [false; 3],
			lbrr_flag: false,
			lbrr_flags: [false; 3],
			resampler: Resampler::default(),
			indices: Indices::default(),
			prev_signal_type: 0,
			loss_cnt: 0,
			plc: PlcState::new(),
			cng: CngState::new(),
		}
	}

	fn nlsf_cb(&self) -> &'static NlsfCb {
		if self.wideband { &NLSF_CB_WB } else { &NLSF_CB_NB_MB }
	}

	fn pitch_lag_low_bits_icdf(&self) -> &'static [u8] {
		match self.fs_khz {
			16 => &UNIFORM8_I_CDF,
			12 => &UNIFORM6_I_CDF,
			_ => &UNIFORM4_I_CDF,
		}
	}

	fn pitch_contour_icdf(&self) -> &'static [u8] {
		match (self.fs_khz == 8, self.nb_subfr == MAX_NB_SUBFR) {
			(true, true) => &PITCH_CONTOUR_NB_I_CDF,
			(true, false) => &PITCH_CONTOUR_10_MS_NB_I_CDF,
			(false, true) => &PITCH_CONTOUR_I_CDF,
			(false, false) => &PITCH_CONTOUR_10_MS_I_CDF,
		}
	}

	/// `silk_decoder_set_fs`.
	fn set_fs(&mut self, fs_khz: i32, fs_api_hz: i32) {
		self.subfr_length = SUB_FRAME_LENGTH_MS * fs_khz as usize;
		let frame_length = self.nb_subfr * self.subfr_length;
		if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
			self.resampler = Resampler::new(fs_khz * 1000, fs_api_hz);
			self.fs_api_hz = fs_api_hz;
		}
		if self.fs_khz != fs_khz || frame_length != self.frame_length {
			if self.fs_khz != fs_khz {
				self.ltp_mem_length = LTP_MEM_LENGTH_MS * fs_khz as usize;
				self.wideband = fs_khz == 16;
				self.lpc_order = if self.wideband { 16 } else { 10 };
				self.first_frame_after_reset = true;
				self.lag_prev = 100;
				self.last_gain_index = 10;
				self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
				self.out_buf.fill(0);
				self.s_lpc_q14_buf.fill(0);
			}
			self.fs_khz = fs_khz;
			self.frame_length = frame_length;
		}
	}
}

// --- Indices, pulses, parameters. ---

fn decode_indices(
	st: &mut ChannelState,
	dec: &mut RangeDecoder,
	frame_index: usize,
	decode_lbrr: bool,
	cond: i32,
) {
	let ix = if decode_lbrr || st.vad_flags[frame_index] {
		dec.icdf(&TYPE_OFFSET_VAD_I_CDF, 8) as i32 + 2
	} else {
		dec.icdf(&TYPE_OFFSET_NO_VAD_I_CDF, 8) as i32
	};
	let ind = &mut st.indices;
	ind.signal_type = ix >> 1;
	ind.quant_offset_type = ix & 1;
	if cond == CODE_CONDITIONALLY {
		ind.gains[0] = dec.icdf(&DELTA_GAIN_I_CDF, 8) as i32;
	} else {
		let row = &GAIN_I_CDF[ind.signal_type as usize * 8..];
		ind.gains[0] = (dec.icdf(row, 8) as i32) << 3;
		ind.gains[0] += dec.icdf(&UNIFORM8_I_CDF, 8) as i32;
	}
	for i in 1..st.nb_subfr {
		st.indices.gains[i] = dec.icdf(&DELTA_GAIN_I_CDF, 8) as i32;
	}
	let cb = st.nlsf_cb();
	let ind = &mut st.indices;
	ind.nlsf[0] = dec.icdf(&cb.cb1_icdf[(ind.signal_type as usize >> 1) * cb.n_vectors..], 8) as i32;
	let (ec_ix, _) = nlsf_unpack(cb, ind.nlsf[0] as usize);
	for i in 0..cb.order {
		let mut ix = dec.icdf(&cb.ec_icdf[ec_ix[i]..], 8) as i32;
		if ix == 0 {
			ix -= dec.icdf(&NLSF_EXT_I_CDF, 8) as i32;
		} else if ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
			ix += dec.icdf(&NLSF_EXT_I_CDF, 8) as i32;
		}
		ind.nlsf[i + 1] = ix - NLSF_QUANT_MAX_AMPLITUDE;
	}
	ind.nlsf_interp_coef_q2 = if st.nb_subfr == MAX_NB_SUBFR {
		dec.icdf(&NLSF_INTERPOLATION_FACTOR_I_CDF, 8) as i32
	} else {
		4
	};
	if st.indices.signal_type == TYPE_VOICED {
		let mut absolute = true;
		if cond == CODE_CONDITIONALLY && st.ec_prev_signal_type == TYPE_VOICED {
			let mut delta = dec.icdf(&PITCH_DELTA_I_CDF, 8) as i32;
			if delta > 0 {
				delta -= 9;
				st.indices.lag_index = st.ec_prev_lag_index + delta;
				absolute = false;
			}
		}
		if absolute {
			st.indices.lag_index = dec.icdf(&PITCH_LAG_I_CDF, 8) as i32 * (st.fs_khz >> 1);
			st.indices.lag_index += dec.icdf(st.pitch_lag_low_bits_icdf(), 8) as i32;
		}
		st.ec_prev_lag_index = st.indices.lag_index;
		st.indices.contour_index = dec.icdf(st.pitch_contour_icdf(), 8);
		st.indices.per_index = dec.icdf(&LTP_PER_INDEX_I_CDF, 8);
		for k in 0..st.nb_subfr {
			st.indices.ltp_index[k] = dec.icdf(ltp_gain_icdf(st.indices.per_index), 8);
		}
		st.indices.ltp_scale_index =
			if cond == CODE_INDEPENDENTLY { dec.icdf(&LTPSCALE_I_CDF, 8) } else { 0 };
	}
	st.ec_prev_signal_type = st.indices.signal_type;
	st.indices.seed = dec.icdf(&UNIFORM4_I_CDF, 8) as i32;
}

fn shell_decoder(out: &mut [i32], dec: &mut RangeDecoder, pulses4: i32) {
	let split = |dec: &mut RangeDecoder, p: i32, table: &[u8]| -> (i32, i32) {
		if p > 0 {
			let a = dec.icdf(&table[usize::from(SHELL_CODE_TABLE_OFFSETS[p as usize])..], 8) as i32;
			(a, p - a)
		} else {
			(0, 0)
		}
	};
	let (p3a, p3b) = split(dec, pulses4, &SHELL_CODE_TABLE3);
	let mut p2 = [0i32; 4];
	let mut p1 = [0i32; 8];
	(p2[0], p2[1]) = split(dec, p3a, &SHELL_CODE_TABLE2);
	(p1[0], p1[1]) = split(dec, p2[0], &SHELL_CODE_TABLE1);
	(out[0], out[1]) = split(dec, p1[0], &SHELL_CODE_TABLE0);
	(out[2], out[3]) = split(dec, p1[1], &SHELL_CODE_TABLE0);
	(p1[2], p1[3]) = split(dec, p2[1], &SHELL_CODE_TABLE1);
	(out[4], out[5]) = split(dec, p1[2], &SHELL_CODE_TABLE0);
	(out[6], out[7]) = split(dec, p1[3], &SHELL_CODE_TABLE0);
	(p2[2], p2[3]) = split(dec, p3b, &SHELL_CODE_TABLE2);
	(p1[4], p1[5]) = split(dec, p2[2], &SHELL_CODE_TABLE1);
	(out[8], out[9]) = split(dec, p1[4], &SHELL_CODE_TABLE0);
	(out[10], out[11]) = split(dec, p1[5], &SHELL_CODE_TABLE0);
	(p1[6], p1[7]) = split(dec, p2[3], &SHELL_CODE_TABLE1);
	(out[12], out[13]) = split(dec, p1[6], &SHELL_CODE_TABLE0);
	(out[14], out[15]) = split(dec, p1[7], &SHELL_CODE_TABLE0);
}

fn decode_pulses(
	dec: &mut RangeDecoder,
	pulses: &mut [i32],
	signal_type: i32,
	quant_offset_type: i32,
	frame_length: usize,
) {
	let rate_level = dec.icdf(&RATE_LEVELS_I_CDF[(signal_type as usize >> 1) * 9..], 8);
	let mut iter = frame_length / SHELL_CODEC_FRAME_LENGTH;
	if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
		iter += 1;
	}
	let mut sum_pulses = [0i32; MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH + 1];
	let mut n_lshifts = [0i32; MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH + 1];
	for i in 0..iter {
		sum_pulses[i] = dec.icdf(&PULSES_PER_BLOCK_I_CDF[rate_level * 18..], 8) as i32;
		while sum_pulses[i] == SILK_MAX_PULSES as i32 + 1 {
			n_lshifts[i] += 1;
			let offset = (N_RATE_LEVELS - 1) * 18 + usize::from(n_lshifts[i] == 10);
			sum_pulses[i] = dec.icdf(&PULSES_PER_BLOCK_I_CDF[offset..], 8) as i32;
			if n_lshifts[i] > 10 {
				break;
			}
		}
	}
	for i in 0..iter {
		let block = &mut pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH];
		if sum_pulses[i] > 0 {
			shell_decoder(block, dec, sum_pulses[i]);
		} else {
			block.fill(0);
		}
	}
	for i in 0..iter {
		if n_lshifts[i] > 0 {
			let nls = n_lshifts[i];
			for k in 0..SHELL_CODEC_FRAME_LENGTH {
				let mut abs_q = pulses[i * SHELL_CODEC_FRAME_LENGTH + k];
				for _ in 0..nls {
					abs_q = (abs_q << 1) + dec.icdf(&LSB_I_CDF, 8) as i32;
				}
				pulses[i * SHELL_CODEC_FRAME_LENGTH + k] = abs_q;
			}
			sum_pulses[i] |= nls << 5;
		}
	}
	// silk_decode_signs
	let base = 7 * (quant_offset_type + (signal_type << 1)) as usize;
	let length = (frame_length + SHELL_CODEC_FRAME_LENGTH / 2) / SHELL_CODEC_FRAME_LENGTH;
	for i in 0..length {
		let p = sum_pulses[i];
		if p > 0 {
			let icdf = [SIGN_I_CDF[base + (p & 0x1F).min(6) as usize], 0];
			for j in 0..SHELL_CODEC_FRAME_LENGTH {
				let q = &mut pulses[i * SHELL_CODEC_FRAME_LENGTH + j];
				if *q > 0 {
					*q *= ((dec.icdf(&icdf, 8) as i32) << 1) - 1;
				}
			}
		}
	}
}

fn nlsf_unpack(cb: &NlsfCb, cb1_index: usize) -> ([usize; MAX_LPC_ORDER], [u8; MAX_LPC_ORDER]) {
	let mut ec_ix = [0usize; MAX_LPC_ORDER];
	let mut pred = [0u8; MAX_LPC_ORDER];
	let sel = &cb.ec_sel[cb1_index * cb.order / 2..];
	for i in (0..cb.order).step_by(2) {
		let entry = sel[i / 2];
		ec_ix[i] = (usize::from(entry >> 1) & 7) * (2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1);
		pred[i] = cb.pred_q8[i + usize::from(entry & 1) * (cb.order - 1)];
		ec_ix[i + 1] = (usize::from(entry >> 5) & 7) * (2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1);
		pred[i + 1] = cb.pred_q8[i + usize::from((entry >> 4) & 1) * (cb.order - 1) + 1];
	}
	(ec_ix, pred)
}

fn nlsf_stabilize(nlsf: &mut [i16], delta_min: &[i16], l: usize) {
	for _ in 0..20 {
		let mut min_diff = i32::from(nlsf[0]) - i32::from(delta_min[0]);
		let mut index = 0usize;
		for i in 1..l {
			let diff = i32::from(nlsf[i]) - (i32::from(nlsf[i - 1]) + i32::from(delta_min[i]));
			if diff < min_diff {
				min_diff = diff;
				index = i;
			}
		}
		let diff = (1 << 15) - (i32::from(nlsf[l - 1]) + i32::from(delta_min[l]));
		if diff < min_diff {
			min_diff = diff;
			index = l;
		}
		if min_diff >= 0 {
			return;
		}
		if index == 0 {
			nlsf[0] = delta_min[0];
		} else if index == l {
			nlsf[l - 1] = ((1 << 15) - i32::from(delta_min[l])) as i16;
		} else {
			let mut min_center: i32 = delta_min[..index].iter().map(|&v| i32::from(v)).sum();
			min_center += i32::from(delta_min[index]) >> 1;
			let mut max_center = 1i32 << 15;
			for k in (index + 1..=l).rev() {
				max_center -= i32::from(delta_min[k]);
			}
			max_center -= i32::from(delta_min[index]) >> 1;
			let center = limit(
				rshift_round(i32::from(nlsf[index - 1]) + i32::from(nlsf[index]), 1),
				min_center,
				max_center,
			);
			nlsf[index - 1] = (center - (i32::from(delta_min[index]) >> 1)) as i16;
			nlsf[index] = (i32::from(nlsf[index - 1]) + i32::from(delta_min[index])) as i16;
		}
	}
	// Fall back: sort and force the minimum spacing.
	nlsf[..l].sort_unstable();
	nlsf[0] = nlsf[0].max(delta_min[0]);
	for i in 1..l {
		nlsf[i] = nlsf[i].max(nlsf[i - 1].saturating_add(delta_min[i]));
	}
	nlsf[l - 1] = (i32::from(nlsf[l - 1])).min((1 << 15) - i32::from(delta_min[l])) as i16;
	for i in (0..l - 1).rev() {
		nlsf[i] = (i32::from(nlsf[i])).min(i32::from(nlsf[i + 1]) - i32::from(delta_min[i + 1])) as i16;
	}
}

fn nlsf_decode(indices: &[i32; MAX_LPC_ORDER + 1], cb: &NlsfCb) -> [i16; MAX_LPC_ORDER] {
	let cb1 = indices[0] as usize;
	let (_, pred) = nlsf_unpack(cb, cb1);
	let order = cb.order;
	let mut res_q10 = [0i32; MAX_LPC_ORDER];
	let mut out_q10 = 0i32;
	for i in (0..order).rev() {
		let pred_q10 = smulbb(out_q10, i32::from(pred[i])) >> 8;
		out_q10 = indices[i + 1] << 10;
		if out_q10 > 0 {
			out_q10 -= 102; // SILK_FIX_CONST(NLSF_QUANT_LEVEL_ADJ, 10)
		} else if out_q10 < 0 {
			out_q10 += 102;
		}
		out_q10 = smlawb(pred_q10, out_q10, cb.quant_step_size_q16);
		res_q10[i] = out_q10;
	}
	let mut nlsf = [0i16; MAX_LPC_ORDER];
	for i in 0..order {
		let w = i32::from(cb.cb1_wght_q9[cb1 * order + i]);
		let v =
			((res_q10[i] as i16 as i32) << 14) / w + (i32::from(cb.cb1_nlsf_q8[cb1 * order + i]) << 7);
		nlsf[i] = v.clamp(0, 32767) as i16;
	}
	nlsf_stabilize(&mut nlsf, cb.delta_min_q15, order);
	nlsf
}

fn bwexpander_32(ar: &mut [i32], mut chirp_q16: i32) {
	let d = ar.len();
	let chirp_minus_one = chirp_q16 - 65536;
	for i in 0..d - 1 {
		ar[i] = smulww(chirp_q16, ar[i]);
		chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one), 16);
	}
	ar[d - 1] = smulww(chirp_q16, ar[d - 1]);
}

fn lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: u32, qin: u32, d: usize) {
	let mut i = 0;
	while i < 10 {
		let mut maxabs = 0i32;
		let mut idx = 0usize;
		for k in 0..d {
			let absval = a_qin[k].wrapping_abs();
			if absval > maxabs {
				maxabs = absval;
				idx = k;
			}
		}
		maxabs = rshift_round(maxabs, qin - qout);
		if maxabs > i32::from(i16::MAX) {
			maxabs = maxabs.min(163_838);
			let chirp = 65470
				- ((maxabs - i32::from(i16::MAX)) << 14) / ((maxabs.wrapping_mul(idx as i32 + 1)) >> 2);
			bwexpander_32(&mut a_qin[..d], chirp);
		} else {
			break;
		}
		i += 1;
	}
	if i == 10 {
		for k in 0..d {
			a_qout[k] = sat16(rshift_round(a_qin[k], qin - qout)) as i16;
			a_qin[k] = i32::from(a_qout[k]) << (qin - qout);
		}
	} else {
		for k in 0..d {
			a_qout[k] = rshift_round(a_qin[k], qin - qout) as i16;
		}
	}
}

fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
	const QA: u32 = 24;
	const A_LIMIT: i32 = 16_773_022; // SILK_FIX_CONST(0.99975, 24)
	const MIN_INV_GAIN: i32 = 107_374; // SILK_FIX_CONST(1/1e4, 30)
	let order = a_q12.len();
	let mut a = [0i32; MAX_LPC_ORDER];
	let mut dc = 0i32;
	for k in 0..order {
		dc += i32::from(a_q12[k]);
		a[k] = i32::from(a_q12[k]) << (QA - 12);
	}
	if dc >= 4096 {
		return 0;
	}
	let mul32_frac_q = |a: i32, b: i32, q: u32| rshift_round64(i64::from(a) * i64::from(b), q) as i32;
	let mut inv_gain_q30 = 1i32 << 30;
	let mut k = order - 1;
	while k > 0 {
		if a[k] > A_LIMIT || a[k] < -A_LIMIT {
			return 0;
		}
		let rc_q31 = -(a[k] << (31 - QA));
		let rc_mult1_q30 = (1i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
		inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
		if inv_gain_q30 < MIN_INV_GAIN {
			return 0;
		}
		let mult2q = 32 - clz32(rc_mult1_q30.wrapping_abs());
		let rc_mult2 = inverse32_var_q(rc_mult1_q30, mult2q + 30);
		for n in 0..(k + 1) >> 1 {
			let tmp1 = a[n];
			let tmp2 = a[k - n - 1];
			let t = rshift_round64(
				i64::from(sub_sat32(tmp1, mul32_frac_q(tmp2, rc_q31, 31))) * i64::from(rc_mult2),
				mult2q as u32,
			);
			if t > i64::from(i32::MAX) || t < i64::from(i32::MIN) {
				return 0;
			}
			a[n] = t as i32;
			let t = rshift_round64(
				i64::from(sub_sat32(tmp2, mul32_frac_q(tmp1, rc_q31, 31))) * i64::from(rc_mult2),
				mult2q as u32,
			);
			if t > i64::from(i32::MAX) || t < i64::from(i32::MIN) {
				return 0;
			}
			a[k - n - 1] = t as i32;
		}
		k -= 1;
	}
	if a[0] > A_LIMIT || a[0] < -A_LIMIT {
		return 0;
	}
	let rc_q31 = -(a[0] << (31 - QA));
	let rc_mult1_q30 = (1i32 << 30).wrapping_sub(smmul(rc_q31, rc_q31));
	inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
	if inv_gain_q30 < MIN_INV_GAIN {
		return 0;
	}
	inv_gain_q30
}

fn nlsf2a(nlsf: &[i16], d: usize) -> [i16; MAX_LPC_ORDER] {
	const QA: u32 = 16;
	const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
	const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
	let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };
	let mut cos_lsf = [0i32; MAX_LPC_ORDER];
	for k in 0..d {
		let v = i32::from(nlsf[k]);
		let f_int = (v >> 8) as usize;
		let f_frac = v - ((f_int as i32) << 8);
		let cos_val = i32::from(LSFCOS_TAB_FIX_Q12[f_int]);
		let delta = i32::from(LSFCOS_TAB_FIX_Q12[f_int + 1]) - cos_val;
		cos_lsf[ordering[k]] = rshift_round((cos_val << 8) + delta * f_frac, 20 - QA);
	}
	let dd = d >> 1;
	let find_poly = |c: &[i32], step_start: usize| -> [i32; MAX_LPC_ORDER / 2 + 1] {
		let mut out = [0i32; MAX_LPC_ORDER / 2 + 1];
		out[0] = 1 << QA;
		out[1] = -c[step_start];
		for k in 1..dd {
			let ftmp = c[step_start + 2 * k];
			out[k + 1] = (out[k - 1] << 1)
				.wrapping_sub(rshift_round64(i64::from(ftmp) * i64::from(out[k]), QA) as i32);
			for n in (2..=k).rev() {
				out[n] = out[n].wrapping_add(
					out[n - 2]
						.wrapping_sub(rshift_round64(i64::from(ftmp) * i64::from(out[n - 1]), QA) as i32),
				);
			}
			out[1] = out[1].wrapping_sub(ftmp);
		}
		out
	};
	let p = find_poly(&cos_lsf, 0);
	let q = find_poly(&cos_lsf, 1);
	let mut a32 = [0i32; MAX_LPC_ORDER];
	for k in 0..dd {
		let ptmp = p[k + 1].wrapping_add(p[k]);
		let qtmp = q[k + 1].wrapping_sub(q[k]);
		a32[k] = (-qtmp).wrapping_sub(ptmp);
		a32[d - k - 1] = qtmp.wrapping_sub(ptmp);
	}
	let mut a_q12 = [0i16; MAX_LPC_ORDER];
	lpc_fit(&mut a_q12, &mut a32, 12, QA + 1, d);
	let mut i = 0;
	while lpc_inverse_pred_gain(&a_q12[..d]) == 0 && i < MAX_LPC_STABILIZE_ITERATIONS {
		bwexpander_32(&mut a32[..d], 65536 - (2 << i));
		for k in 0..d {
			a_q12[k] = rshift_round(a32[k], QA + 1 - 12) as i16;
		}
		i += 1;
	}
	a_q12
}

#[derive(Default)]
struct DecoderControl {
	pitch_l: [i32; MAX_NB_SUBFR],
	gains_q16: [i32; MAX_NB_SUBFR],
	pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
	ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
	ltp_scale_q14: i32,
}

fn decode_parameters(st: &mut ChannelState, cond: i32) -> DecoderControl {
	let mut ctrl = DecoderControl {
		pitch_l: [0; MAX_NB_SUBFR],
		gains_q16: [0; MAX_NB_SUBFR],
		pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
		ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
		ltp_scale_q14: 0,
	};
	// silk_gains_dequant
	const OFFSET: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
	const INV_SCALE_Q16: i32 =
		(65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1);
	let conditional = cond == CODE_CONDITIONALLY;
	for k in 0..st.nb_subfr {
		let ind = st.indices.gains[k];
		if k == 0 && !conditional {
			st.last_gain_index = ind.max(st.last_gain_index - 16);
		} else {
			let ind_tmp = ind + MIN_DELTA_GAIN_QUANT;
			let threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + st.last_gain_index;
			if ind_tmp > threshold {
				st.last_gain_index += (ind_tmp << 1) - threshold;
			} else {
				st.last_gain_index += ind_tmp;
			}
		}
		st.last_gain_index = st.last_gain_index.clamp(0, N_LEVELS_QGAIN - 1);
		ctrl.gains_q16[k] = log2lin((smulwb(INV_SCALE_Q16, st.last_gain_index) + OFFSET).min(3967));
	}

	let cb = st.nlsf_cb();
	let nlsf = nlsf_decode(&st.indices.nlsf, cb);
	let order = st.lpc_order;
	ctrl.pred_coef_q12[1] = nlsf2a(&nlsf, order);
	if st.first_frame_after_reset {
		st.indices.nlsf_interp_coef_q2 = 4;
	}
	if st.indices.nlsf_interp_coef_q2 < 4 {
		let mut nlsf0 = [0i16; MAX_LPC_ORDER];
		for i in 0..order {
			let prev = i32::from(st.prev_nlsf_q15[i]);
			nlsf0[i] =
				(prev + ((st.indices.nlsf_interp_coef_q2 * (i32::from(nlsf[i]) - prev)) >> 2)) as i16;
		}
		ctrl.pred_coef_q12[0] = nlsf2a(&nlsf0, order);
	} else {
		ctrl.pred_coef_q12[0] = ctrl.pred_coef_q12[1];
	}
	st.prev_nlsf_q15 = nlsf;

	// After a packet loss, bandwidth-expand the LPC filters.
	if st.loss_cnt != 0 {
		bwexpander(&mut ctrl.pred_coef_q12[0][..order], BWE_AFTER_LOSS_Q16);
		bwexpander(&mut ctrl.pred_coef_q12[1][..order], BWE_AFTER_LOSS_Q16);
	}

	if st.indices.signal_type == TYPE_VOICED {
		// silk_decode_pitch
		let fs = st.fs_khz;
		let (cb_lags, cbk_size): (&[i8], usize) = match (fs == 8, st.nb_subfr == MAX_NB_SUBFR) {
			(true, true) => (&CB_LAGS_STAGE2, 11),
			(true, false) => (&CB_LAGS_STAGE2_10_MS, 3),
			(false, true) => (&CB_LAGS_STAGE3, 34),
			(false, false) => (&CB_LAGS_STAGE3_10_MS, 12),
		};
		let min_lag = PE_MIN_LAG_MS * fs;
		let max_lag = PE_MAX_LAG_MS * fs;
		let lag = min_lag + st.indices.lag_index;
		let contour = st.indices.contour_index.min(cbk_size - 1);
		for k in 0..st.nb_subfr {
			ctrl.pitch_l[k] = limit(lag + i32::from(cb_lags[k * cbk_size + contour]), min_lag, max_lag);
		}
		let vq = ltp_vq(st.indices.per_index);
		for k in 0..st.nb_subfr {
			let ix = st.indices.ltp_index[k];
			for i in 0..LTP_ORDER {
				ctrl.ltp_coef_q14[k * LTP_ORDER + i] =
					(i16::from(vq.get(ix * LTP_ORDER + i).copied().unwrap_or(0))) << 7;
			}
		}
		ctrl.ltp_scale_q14 = i32::from(LTPSCALES_TABLE_Q14[st.indices.ltp_scale_index.min(2)]);
	} else {
		st.indices.per_index = 0;
	}
	ctrl
}

fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16], len: usize, d: usize) {
	for ix in d..len {
		let p = ix - 1;
		let mut acc = smulbb(i32::from(input[p]), i32::from(b[0]));
		for j in 1..d {
			acc = acc.wrapping_add(smulbb(i32::from(input[p - j]), i32::from(b[j])));
		}
		let v = (i32::from(input[p + 1]) << 12).wrapping_sub(acc);
		out[ix] = sat16(rshift_round(v, 12)) as i16;
	}
	out[..d].fill(0);
}

fn decode_core(st: &mut ChannelState, ctrl: &mut DecoderControl, xq: &mut [i16], pulses: &[i32]) {
	let ltp_mem = st.ltp_mem_length;
	let frame_length = st.frame_length;
	let subfr_length = st.subfr_length;
	let order = st.lpc_order;
	let mut s_ltp = vec![0i16; ltp_mem];
	let mut s_ltp_q15 = vec![0i32; ltp_mem + frame_length];
	let mut res_q14 = vec![0i32; subfr_length];
	let mut s_lpc_q14 = vec![0i32; subfr_length + MAX_LPC_ORDER];
	let offset_q10 = i32::from(
		QUANTIZATION_OFFSETS_Q10
			[(st.indices.signal_type as usize >> 1) * 2 + st.indices.quant_offset_type as usize],
	);
	let nlsf_interpolation = st.indices.nlsf_interp_coef_q2 < 4;

	let mut rand_seed = st.indices.seed;
	for i in 0..frame_length {
		rand_seed = silk_rand(rand_seed);
		let mut e = pulses[i] << 14;
		if e > 0 {
			e -= QUANT_LEVEL_ADJUST_Q10 << 4;
		} else if e < 0 {
			e += QUANT_LEVEL_ADJUST_Q10 << 4;
		}
		e += offset_q10 << 4;
		if rand_seed < 0 {
			e = -e;
		}
		st.exc_q14[i] = e;
		rand_seed = rand_seed.wrapping_add(pulses[i]);
	}
	s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&st.s_lpc_q14_buf);
	let mut ltp_buf_idx = ltp_mem;
	let mut lag = 0usize;
	for k in 0..st.nb_subfr {
		let a_q12 = ctrl.pred_coef_q12[k >> 1];
		let mut b_q14: [i16; LTP_ORDER] =
			ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER].try_into().unwrap_or([0; LTP_ORDER]);
		let mut signal_type = st.indices.signal_type;
		let gain_q10 = ctrl.gains_q16[k] >> 6;
		let mut inv_gain_q31 = inverse32_var_q(ctrl.gains_q16[k], 47);
		let gain_adj_q16;
		if ctrl.gains_q16[k] != st.prev_gain_q16 {
			gain_adj_q16 = div32_var_q(st.prev_gain_q16, ctrl.gains_q16[k], 16);
			for v in s_lpc_q14[..MAX_LPC_ORDER].iter_mut() {
				*v = smulww(gain_adj_q16, *v);
			}
		} else {
			gain_adj_q16 = 1 << 16;
		}
		st.prev_gain_q16 = ctrl.gains_q16[k];

		// Avoid an abrupt switch from voiced concealment to unvoiced decoding.
		if st.loss_cnt != 0
			&& st.prev_signal_type == TYPE_VOICED
			&& st.indices.signal_type != TYPE_VOICED
			&& k < MAX_NB_SUBFR / 2
		{
			b_q14 = [0; LTP_ORDER];
			b_q14[LTP_ORDER / 2] = 4096; // 0.25 in Q14
			signal_type = TYPE_VOICED;
			ctrl.pitch_l[k] = st.lag_prev;
		}

		if signal_type == TYPE_VOICED {
			lag = ctrl.pitch_l[k] as usize;
			if k == 0 || (k == 2 && nlsf_interpolation) {
				let start_idx = ltp_mem - lag - order - LTP_ORDER / 2;
				if k == 2 {
					st.out_buf[ltp_mem..ltp_mem + 2 * subfr_length].copy_from_slice(&xq[..2 * subfr_length]);
				}
				let src: Vec<i16> = st.out_buf
					[start_idx + k * subfr_length..start_idx + k * subfr_length + ltp_mem - start_idx]
					.to_vec();
				lpc_analysis_filter(&mut s_ltp[start_idx..], &src, &a_q12, ltp_mem - start_idx, order);
				if k == 0 {
					inv_gain_q31 = smulwb(inv_gain_q31, ctrl.ltp_scale_q14) << 2;
				}
				for i in 0..lag + LTP_ORDER / 2 {
					s_ltp_q15[ltp_buf_idx - i - 1] = smulwb(inv_gain_q31, i32::from(s_ltp[ltp_mem - i - 1]));
				}
			} else if gain_adj_q16 != 1 << 16 {
				for i in 0..lag + LTP_ORDER / 2 {
					let at = ltp_buf_idx - i - 1;
					s_ltp_q15[at] = smulww(gain_adj_q16, s_ltp_q15[at]);
				}
			}
			let pred_start = ltp_buf_idx - lag + LTP_ORDER / 2;
			for (i, pred_at) in (pred_start..pred_start + subfr_length).enumerate() {
				let mut ltp_pred_q13 = 2i32;
				ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_at], i32::from(b_q14[0]));
				ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_at - 1], i32::from(b_q14[1]));
				ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_at - 2], i32::from(b_q14[2]));
				ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_at - 3], i32::from(b_q14[3]));
				ltp_pred_q13 = smlawb(ltp_pred_q13, s_ltp_q15[pred_at - 4], i32::from(b_q14[4]));
				res_q14[i] = st.exc_q14[k * subfr_length + i].wrapping_add(ltp_pred_q13 << 1);
				s_ltp_q15[ltp_buf_idx] = res_q14[i] << 1;
				ltp_buf_idx += 1;
			}
		} else {
			res_q14.copy_from_slice(&st.exc_q14[k * subfr_length..(k + 1) * subfr_length]);
		}

		for i in 0..subfr_length {
			let mut pred_q10 = (order >> 1) as i32;
			for j in 0..order {
				pred_q10 = smlawb(pred_q10, s_lpc_q14[MAX_LPC_ORDER + i - j - 1], i32::from(a_q12[j]));
			}
			s_lpc_q14[MAX_LPC_ORDER + i] = add_sat32(res_q14[i], lshift_sat32(pred_q10, 4));
			xq[k * subfr_length + i] =
				sat16(rshift_round(smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10), 8)) as i16;
		}
		s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
	}
	st.s_lpc_q14_buf.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
	let _ = lag;
}

/// `silk_decode_frame`: decodes (or conceals) one frame into `out` and
/// returns its length.
fn decode_frame(
	st: &mut ChannelState,
	dec: &mut RangeDecoder,
	out: &mut [i16],
	mode: SilkMode,
	cond: i32,
) -> usize {
	let l = st.frame_length;
	let mut ctrl = DecoderControl::default();
	let has_data = match mode {
		SilkMode::Normal => true,
		SilkMode::Lbrr => st.lbrr_flags[st.n_frames_decoded.min(2)],
		SilkMode::Lost => false,
	};
	if has_data {
		let mut pulses =
			vec![0i32; (l + SHELL_CODEC_FRAME_LENGTH - 1) & !(SHELL_CODEC_FRAME_LENGTH - 1)];
		decode_indices(st, dec, st.n_frames_decoded, mode == SilkMode::Lbrr, cond);
		decode_pulses(dec, &mut pulses, st.indices.signal_type, st.indices.quant_offset_type, l);
		ctrl = decode_parameters(st, cond);
		decode_core(st, &mut ctrl, out, &pulses);
		plc(st, &mut ctrl, out, false);
		st.loss_cnt = 0;
		st.prev_signal_type = st.indices.signal_type;
		st.first_frame_after_reset = false;
	} else {
		plc(st, &mut ctrl, out, true);
	}
	let mv = st.ltp_mem_length - l;
	st.out_buf.copy_within(l..l + mv, 0);
	st.out_buf[mv..mv + l].copy_from_slice(&out[..l]);
	cng(st, &ctrl, &mut out[..l]);
	plc_glue_frames(st, &mut out[..l]);
	st.lag_prev = ctrl.pitch_l[st.nb_subfr - 1];
	l
}

// --- Packet-loss concealment (PLC.c). ---

/// `silk_PLC_Reset`.
fn plc_reset(st: &mut ChannelState) {
	st.plc.pitch_l_q8 = (st.frame_length as i32) << 7;
	st.plc.prev_gain_q16 = [1 << 16; 2];
	st.plc.subfr_length = 20;
	st.plc.nb_subfr = 2;
}

/// `silk_PLC`: conceals a lost frame into `frame`, or learns from a good one.
fn plc(st: &mut ChannelState, ctrl: &mut DecoderControl, frame: &mut [i16], lost: bool) {
	if st.fs_khz != st.plc.fs_khz {
		plc_reset(st);
		st.plc.fs_khz = st.fs_khz;
	}
	if lost {
		plc_conceal(st, ctrl, frame);
		st.loss_cnt += 1;
	} else {
		plc_update(st, ctrl);
	}
}

/// `silk_PLC_update`: keeps the parameters of the last good frame.
fn plc_update(st: &mut ChannelState, ctrl: &DecoderControl) {
	let nb_subfr = st.nb_subfr;
	let subfr_length = st.subfr_length as i32;
	st.prev_signal_type = st.indices.signal_type;
	let plc = &mut st.plc;
	if st.indices.signal_type == TYPE_VOICED {
		// The strongest LTP filter among the subframes within one pitch
		// period of the end.
		let mut ltp_gain_q14 = 0i32;
		let mut j = 0usize;
		while (j as i32) * subfr_length < ctrl.pitch_l[nb_subfr - 1] && j < nb_subfr {
			let sub = nb_subfr - 1 - j;
			let taps = &ctrl.ltp_coef_q14[sub * LTP_ORDER..(sub + 1) * LTP_ORDER];
			let gain: i32 = taps.iter().map(|&t| i32::from(t)).sum();
			if gain > ltp_gain_q14 {
				ltp_gain_q14 = gain;
				plc.ltp_coef_q14.copy_from_slice(taps);
				plc.pitch_l_q8 = ctrl.pitch_l[sub] << 8;
			}
			j += 1;
		}
		// As in libopus, only the summed gain is kept, on the centre tap.
		plc.ltp_coef_q14 = [0; LTP_ORDER];
		plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;
		if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
			let scale_q10 = (V_PITCH_GAIN_START_MIN_Q14 << 10) / ltp_gain_q14.max(1);
			for c in &mut plc.ltp_coef_q14 {
				*c = (smulbb(i32::from(*c), scale_q10) >> 10) as i16;
			}
		} else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
			let scale_q14 = (V_PITCH_GAIN_START_MAX_Q14 << 14) / ltp_gain_q14.max(1);
			for c in &mut plc.ltp_coef_q14 {
				*c = (smulbb(i32::from(*c), scale_q14) >> 14) as i16;
			}
		}
	} else {
		plc.pitch_l_q8 = smulbb(st.fs_khz, 18) << 8;
		plc.ltp_coef_q14 = [0; LTP_ORDER];
	}
	let order = st.lpc_order;
	plc.prev_lpc_q12[..order].copy_from_slice(&ctrl.pred_coef_q12[1][..order]);
	plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14 as i16;
	plc.prev_gain_q16 = [ctrl.gains_q16[nb_subfr - 2], ctrl.gains_q16[nb_subfr - 1]];
	plc.subfr_length = st.subfr_length;
	plc.nb_subfr = nb_subfr;
}

/// `silk_PLC_energy`: energies of the last two subframes' excitation.
fn plc_energy(st: &ChannelState, prev_gain_q10: [i32; 2]) -> (i32, i32, i32, i32) {
	let subfr_length = st.subfr_length;
	let mut exc_buf = vec![0i16; 2 * subfr_length];
	for k in 0..2 {
		let src = (k + st.nb_subfr - 2) * subfr_length;
		for i in 0..subfr_length {
			exc_buf[k * subfr_length + i] =
				sat16(smulww(st.exc_q14[src + i], prev_gain_q10[k]) >> 8) as i16;
		}
	}
	let (energy1, shift1) = sum_sqr_shift(&exc_buf[..subfr_length]);
	let (energy2, shift2) = sum_sqr_shift(&exc_buf[subfr_length..]);
	(energy1, shift1, energy2, shift2)
}

/// `silk_PLC_conceal`: extrapolates a lost frame from the last excitation
/// (pitch pulses for voiced speech, random excitation otherwise), through
/// the last LPC filter, with decaying gains.
fn plc_conceal(st: &mut ChannelState, ctrl: &mut DecoderControl, frame: &mut [i16]) {
	let ltp_mem = st.ltp_mem_length;
	let frame_length = st.frame_length;
	let order = st.lpc_order;
	let mut s_ltp_q14 = vec![0i32; ltp_mem + frame_length];
	let mut s_ltp = vec![0i16; ltp_mem];
	let prev_gain_q10 = [st.plc.prev_gain_q16[0] >> 6, st.plc.prev_gain_q16[1] >> 6];
	if st.first_frame_after_reset {
		st.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
	}
	let (energy1, shift1, energy2, shift2) = plc_energy(st, prev_gain_q10);
	// Use the quieter of the last two subframes as the noise source.
	let plc_frame = (st.plc.nb_subfr * st.plc.subfr_length) as i32;
	let rand_at = if (energy1 >> shift2) < (energy2 >> shift1) {
		(plc_frame - st.plc.subfr_length as i32 - RAND_BUF_SIZE as i32).max(0)
	} else {
		(plc_frame - RAND_BUF_SIZE as i32).max(0)
	} as usize;

	let mut rand_scale_q14 = st.plc.rand_scale_q14;
	let att = (st.loss_cnt as usize).min(NB_ATT - 1);
	let harm_gain_q15 = HARM_ATT_Q15[att];
	let mut rand_gain_q15 = if st.prev_signal_type == TYPE_VOICED {
		PLC_RAND_ATTENUATE_V_Q15[att]
	} else {
		PLC_RAND_ATTENUATE_UV_Q15[att]
	};
	bwexpander(&mut st.plc.prev_lpc_q12[..order], BWE_COEF_Q16);
	let a_q12 = st.plc.prev_lpc_q12;

	if st.loss_cnt == 0 {
		// First lost frame.
		rand_scale_q14 = 1 << 14;
		if st.prev_signal_type == TYPE_VOICED {
			// Less noise for voiced frames.
			for &b in &st.plc.ltp_coef_q14 {
				rand_scale_q14 = rand_scale_q14.wrapping_sub(b);
			}
			rand_scale_q14 = rand_scale_q14.max(3277); // 0.2
			rand_scale_q14 =
				(smulbb(i32::from(rand_scale_q14), i32::from(st.plc.prev_ltp_scale_q14)) >> 14) as i16;
		} else {
			// Less noise for unvoiced frames with a high LPC gain.
			let inv_gain_q30 = lpc_inverse_pred_gain(&st.plc.prev_lpc_q12[..order]);
			let down_scale_q30 = inv_gain_q30.clamp(
				(1i32 << 30) >> LOG2_INV_LPC_GAIN_LOW_THRES,
				(1i32 << 30) >> LOG2_INV_LPC_GAIN_HIGH_THRES,
			) << LOG2_INV_LPC_GAIN_HIGH_THRES;
			rand_gain_q15 = smulwb(down_scale_q30, rand_gain_q15) >> 14;
		}
	}

	let mut rand_seed = st.plc.rand_seed;
	// The pitch lag never exceeds MAX_PITCH_LAG_MS (and libopus asserts the
	// filter history fits); the clamp only keeps damaged state in bounds.
	const MIN_PLC_LAG: i32 = LTP_ORDER as i32 / 2 + 1;
	let max_lag = (ltp_mem - order - LTP_ORDER / 2 - 1) as i32;
	let mut lag = rshift_round(st.plc.pitch_l_q8, 8).clamp(MIN_PLC_LAG, max_lag) as usize;
	let mut s_ltp_buf_idx = ltp_mem;

	// Re-whiten the LTP state.
	let idx = ltp_mem - lag - order - LTP_ORDER / 2;
	lpc_analysis_filter(&mut s_ltp[idx..], &st.out_buf[idx..], &a_q12, ltp_mem - idx, order);
	let inv_gain_q30 = inverse32_var_q(st.plc.prev_gain_q16[1], 46).min(i32::MAX >> 1);
	for i in idx + order..ltp_mem {
		s_ltp_q14[i] = smulwb(inv_gain_q30, i32::from(s_ltp[i]));
	}

	// LTP synthesis.
	let b_q14 = &mut st.plc.ltp_coef_q14;
	for _ in 0..st.nb_subfr {
		let pred_start = s_ltp_buf_idx - lag + LTP_ORDER / 2;
		for pred_at in pred_start..pred_start + st.subfr_length {
			let mut ltp_pred_q12 = 2i32;
			for (tap, &b) in b_q14.iter().enumerate() {
				ltp_pred_q12 = smlawb(ltp_pred_q12, s_ltp_q14[pred_at - tap], i32::from(b));
			}
			rand_seed = silk_rand(rand_seed);
			let r = ((rand_seed >> 25) & RAND_BUF_MASK) as usize;
			s_ltp_q14[s_ltp_buf_idx] =
				smlawb(ltp_pred_q12, st.exc_q14[rand_at + r], i32::from(rand_scale_q14)).wrapping_shl(2);
			s_ltp_buf_idx += 1;
		}
		for b in b_q14.iter_mut() {
			*b = (smulbb(harm_gain_q15, i32::from(*b)) >> 15) as i16;
		}
		rand_scale_q14 = (smulbb(i32::from(rand_scale_q14), rand_gain_q15) >> 15) as i16;
		// Slowly lengthen the pitch period.
		st.plc.pitch_l_q8 = smlawb(st.plc.pitch_l_q8, st.plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
		st.plc.pitch_l_q8 = st.plc.pitch_l_q8.min(smulbb(MAX_PITCH_LAG_MS, st.fs_khz) << 8);
		lag = rshift_round(st.plc.pitch_l_q8, 8).clamp(MIN_PLC_LAG, max_lag) as usize;
	}

	// LPC synthesis, in place over the excitation.
	let base = ltp_mem - MAX_LPC_ORDER;
	s_ltp_q14[base..ltp_mem].copy_from_slice(&st.s_lpc_q14_buf);
	for i in 0..frame_length {
		let at = base + MAX_LPC_ORDER + i;
		let mut pred_q10 = (order >> 1) as i32;
		for j in 0..order {
			pred_q10 = smlawb(pred_q10, s_ltp_q14[at - j - 1], i32::from(a_q12[j]));
		}
		s_ltp_q14[at] = add_sat32(s_ltp_q14[at], lshift_sat32(pred_q10, 4));
		frame[i] = sat16(rshift_round(smulww(s_ltp_q14[at], prev_gain_q10[1]), 8)) as i16;
	}
	st.s_lpc_q14_buf
		.copy_from_slice(&s_ltp_q14[base + frame_length..base + frame_length + MAX_LPC_ORDER]);

	st.plc.rand_seed = rand_seed;
	st.plc.rand_scale_q14 = rand_scale_q14;
	ctrl.pitch_l = [lag as i32; MAX_NB_SUBFR];
}

/// `silk_PLC_glue_frames`: remembers the energy of concealed frames and
/// fades in the first good frame when it is louder.
fn plc_glue_frames(st: &mut ChannelState, frame: &mut [i16]) {
	let plc = &mut st.plc;
	if st.loss_cnt != 0 {
		(plc.conc_energy, plc.conc_energy_shift) = sum_sqr_shift(frame);
		plc.last_frame_lost = true;
		return;
	}
	if plc.last_frame_lost {
		let (mut energy, energy_shift) = sum_sqr_shift(frame);
		if energy_shift > plc.conc_energy_shift {
			plc.conc_energy >>= energy_shift - plc.conc_energy_shift;
		} else if energy_shift < plc.conc_energy_shift {
			energy >>= plc.conc_energy_shift - energy_shift;
		}
		if energy > plc.conc_energy {
			let lz = clz32(plc.conc_energy) - 1;
			plc.conc_energy = plc.conc_energy.wrapping_shl(lz as u32);
			energy >>= (24 - lz).max(0);
			let frac_q24 = plc.conc_energy / energy.max(1);
			let mut gain_q16 = sqrt_approx(frac_q24) << 4;
			// Four times steeper than linear, so onsets after DTX survive.
			let slope_q16 = (((1 << 16) - gain_q16) / frame.len() as i32) << 2;
			for v in frame.iter_mut() {
				*v = smulwb(gain_q16, i32::from(*v)) as i16;
				gain_q16 += slope_q16;
				if gain_q16 > 1 << 16 {
					break;
				}
			}
		}
	}
	plc.last_frame_lost = false;
}

// --- Comfort noise (CNG.c). ---

/// `silk_CNG_Reset`.
fn cng_reset(st: &mut ChannelState) {
	let order = st.lpc_order;
	let step_q15 = i32::from(i16::MAX) / (order as i32 + 1);
	let mut acc_q15 = 0;
	for v in &mut st.cng.smth_nlsf_q15[..order] {
		acc_q15 += step_q15;
		*v = acc_q15 as i16;
	}
	st.cng.smth_gain_q16 = 0;
	st.cng.rand_seed = CNG_RAND_SEED;
}

/// `silk_CNG`: tracks the background noise during inactive frames and adds
/// it to concealed frames.
fn cng(st: &mut ChannelState, ctrl: &DecoderControl, frame: &mut [i16]) {
	if st.fs_khz != st.cng.fs_khz {
		cng_reset(st);
		st.cng.fs_khz = st.fs_khz;
	}
	let order = st.lpc_order;
	let cng = &mut st.cng;
	if st.loss_cnt == 0 && st.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
		for i in 0..order {
			let smth = i32::from(cng.smth_nlsf_q15[i]);
			cng.smth_nlsf_q15[i] =
				smth.wrapping_add(smulwb(i32::from(st.prev_nlsf_q15[i]) - smth, CNG_NLSF_SMTH_Q16)) as i16;
		}
		// The excitation of the loudest subframe feeds the noise buffer.
		let mut max_gain_q16 = 0;
		let mut subfr = 0;
		for i in 0..st.nb_subfr {
			if ctrl.gains_q16[i] > max_gain_q16 {
				max_gain_q16 = ctrl.gains_q16[i];
				subfr = i;
			}
		}
		let sub = st.subfr_length;
		cng.exc_buf_q14.copy_within(..(st.nb_subfr - 1) * sub, sub);
		cng.exc_buf_q14[..sub].copy_from_slice(&st.exc_q14[subfr * sub..(subfr + 1) * sub]);
		for i in 0..st.nb_subfr {
			let gain = ctrl.gains_q16[i];
			cng.smth_gain_q16 = cng
				.smth_gain_q16
				.wrapping_add(smulwb(gain.wrapping_sub(cng.smth_gain_q16), CNG_GAIN_SMTH_Q16));
			// Adapt quickly when the smoothed gain is 3 dB above this one.
			if smulww(cng.smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16) > gain {
				cng.smth_gain_q16 = gain;
			}
		}
	}
	if st.loss_cnt == 0 {
		cng.synth_state[..order].fill(0);
		return;
	}
	// Noise gain: what the smoothed background has beyond the concealment.
	let smultt = |a: i32, b: i32| (a >> 16).wrapping_mul(b >> 16);
	let mut gain_q16 = smulww(i32::from(st.plc.rand_scale_q14), st.plc.prev_gain_q16[1]);
	if gain_q16 >= 1 << 21 || cng.smth_gain_q16 > 1 << 23 {
		gain_q16 = smultt(gain_q16, gain_q16);
		gain_q16 = smultt(cng.smth_gain_q16, cng.smth_gain_q16).wrapping_sub(gain_q16.wrapping_shl(5));
		gain_q16 = sqrt_approx(gain_q16).wrapping_shl(16);
	} else {
		gain_q16 = smulww(gain_q16, gain_q16);
		gain_q16 = smulww(cng.smth_gain_q16, cng.smth_gain_q16).wrapping_sub(gain_q16.wrapping_shl(5));
		gain_q16 = sqrt_approx(gain_q16).wrapping_shl(8);
	}
	let gain_q10 = gain_q16 >> 6;

	// silk_CNG_exc: random excitation drawn from the buffer.
	let length = frame.len();
	let mut sig_q14 = vec![0i32; length + MAX_LPC_ORDER];
	let mut exc_mask = CNG_BUF_MASK_MAX;
	while exc_mask > length as i32 {
		exc_mask >>= 1;
	}
	let mut seed = cng.rand_seed;
	for v in &mut sig_q14[MAX_LPC_ORDER..] {
		seed = silk_rand(seed);
		*v = cng.exc_buf_q14[((seed >> 24) & exc_mask) as usize];
	}
	cng.rand_seed = seed;

	let a_q12 = nlsf2a(&cng.smth_nlsf_q15, order);
	sig_q14[..MAX_LPC_ORDER].copy_from_slice(&cng.synth_state);
	for i in 0..length {
		let at = MAX_LPC_ORDER + i;
		let mut pred_q10 = (order >> 1) as i32;
		for j in 0..order {
			pred_q10 = smlawb(pred_q10, sig_q14[at - j - 1], i32::from(a_q12[j]));
		}
		sig_q14[at] = add_sat32(sig_q14[at], lshift_sat32(pred_q10, 4));
		let noise = sat16(rshift_round(smulww(sig_q14[at], gain_q10), 8));
		frame[i] = sat16(i32::from(frame[i]) + noise) as i16;
	}
	cng.synth_state.copy_from_slice(&sig_q14[length..length + MAX_LPC_ORDER]);
}

fn stereo_decode_pred(dec: &mut RangeDecoder) -> [i32; 2] {
	let n = dec.icdf(&STEREO_PRED_JOINT_I_CDF, 8) as i32;
	let mut ix = [[0i32; 3]; 2];
	ix[0][2] = n / 5;
	ix[1][2] = n - 5 * ix[0][2];
	for row in ix.iter_mut() {
		row[0] = dec.icdf(&UNIFORM3_I_CDF, 8) as i32;
		row[1] = dec.icdf(&UNIFORM5_I_CDF, 8) as i32;
	}
	let mut pred = [0i32; 2];
	for n in 0..2 {
		ix[n][0] += 3 * ix[n][2];
		let i0 = (ix[n][0] as usize).min(STEREO_PRED_QUANT_Q13.len() - 2);
		let low = i32::from(STEREO_PRED_QUANT_Q13[i0]);
		let step = smulwb(i32::from(STEREO_PRED_QUANT_Q13[i0 + 1]) - low, 6554); // 0.5 / 5 in Q16
		pred[n] = smlabb(low, step, 2 * ix[n][1] + 1);
	}
	pred[0] -= pred[1];
	pred
}

#[derive(Debug, Clone, Default)]
struct StereoState {
	pred_prev_q13: [i32; 2],
	s_mid: [i16; 2],
	s_side: [i16; 2],
}

fn stereo_ms_to_lr(
	state: &mut StereoState,
	x1: &mut [i16],
	x2: &mut [i16],
	pred_q13: [i32; 2],
	fs_khz: i32,
	frame_length: usize,
) {
	x1[..2].copy_from_slice(&state.s_mid);
	x2[..2].copy_from_slice(&state.s_side);
	state.s_mid.copy_from_slice(&x1[frame_length..frame_length + 2]);
	state.s_side.copy_from_slice(&x2[frame_length..frame_length + 2]);
	let mut pred0 = state.pred_prev_q13[0];
	let mut pred1 = state.pred_prev_q13[1];
	let interp = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
	let denom_q16 = (1i32 << 16) / (STEREO_INTERP_LEN_MS * fs_khz);
	let delta0 = rshift_round(smulbb(pred_q13[0] - state.pred_prev_q13[0], denom_q16), 16);
	let delta1 = rshift_round(smulbb(pred_q13[1] - state.pred_prev_q13[1], denom_q16), 16);
	let mix = |x1: &[i16], x2: &mut [i16], n: usize, p0: i32, p1: i32| {
		let mut sum =
			((i32::from(x1[n]) + i32::from(x1[n + 2])).wrapping_add(i32::from(x1[n + 1]) << 1)) << 9;
		sum = smlawb(i32::from(x2[n + 1]) << 8, sum, p0);
		sum = smlawb(sum, i32::from(x1[n + 1]) << 11, p1);
		x2[n + 1] = sat16(rshift_round(sum, 8)) as i16;
	};
	for n in 0..interp.min(frame_length) {
		pred0 += delta0;
		pred1 += delta1;
		mix(x1, x2, n, pred0, pred1);
	}
	for n in interp..frame_length {
		mix(x1, x2, n, pred_q13[0], pred_q13[1]);
	}
	state.pred_prev_q13 = pred_q13;
	for n in 0..frame_length {
		let sum = i32::from(x1[n + 1]) + i32::from(x2[n + 1]);
		let diff = i32::from(x1[n + 1]) - i32::from(x2[n + 1]);
		x1[n + 1] = sat16(sum) as i16;
		x2[n + 1] = sat16(diff) as i16;
	}
}

/// SILK decoder for one Opus stream (`silk_decoder`).
#[derive(Debug, Clone)]
pub struct SilkDecoder {
	/// Output channels (1 or 2).
	channels_api: usize,
	channel_state: [ChannelState; 2],
	stereo: StereoState,
	channels_internal: usize,
	prev_decode_only_middle: bool,
}

impl SilkDecoder {
	pub fn new(channels: usize) -> Self {
		Self {
			channels_api: channels,
			channel_state: [ChannelState::new(), ChannelState::new()],
			stereo: StereoState::default(),
			channels_internal: 0,
			prev_decode_only_middle: false,
		}
	}

	/// `silk_InitDecoder`.
	pub fn reset(&mut self) {
		*self = Self::new(self.channels_api);
	}

	/// `silk_Decode` for one SILK frame (10 or 20 ms). Writes interleaved
	/// samples at 48 kHz (unit scale) to `out` and returns the samples per
	/// channel.
	///
	/// With [`SilkMode::Lost`], `dec` is not read and the frame is concealed;
	/// the caller passes the channel count and rate of the last packet (as
	/// libopus keeps them in `DecControl`) and the concealed duration as
	/// `payload_ms` (10 or 20 ms).
	#[allow(clippy::too_many_lines)]
	pub fn decode(
		&mut self,
		dec: &mut RangeDecoder,
		mode: SilkMode,
		channels_internal: usize,
		internal_rate: u32,
		payload_ms: usize,
		new_packet: bool,
		out: &mut [f32],
	) -> Result<usize, &'static str> {
		const API_RATE: i32 = 48_000;
		let ci = channels_internal.clamp(1, 2);
		if new_packet {
			for n in 0..ci {
				self.channel_state[n].n_frames_decoded = 0;
			}
		}
		if ci > self.channels_internal {
			self.channel_state[1] = ChannelState::new();
		}
		let stereo_to_mono = ci == 1
			&& self.channels_internal == 2
			&& internal_rate as i32 == 1000 * self.channel_state[0].fs_khz;
		if self.channel_state[0].n_frames_decoded == 0 {
			for n in 0..ci {
				let st = &mut self.channel_state[n];
				// A lost payload (0 ms) is concealed as 10 ms.
				let (frames, subfr) = match payload_ms {
					0 | 10 => (1, 2),
					20 => (1, 4),
					40 => (2, 4),
					60 => (3, 4),
					_ => return Err("invalid silk frame size"),
				};
				st.n_frames_per_packet = frames;
				st.nb_subfr = subfr;
				let fs_khz = ((internal_rate as i32) >> 10) + 1;
				if !matches!(fs_khz, 8 | 12 | 16) {
					return Err("invalid silk sample rate");
				}
				st.set_fs(fs_khz, API_RATE);
			}
		}
		if self.channels_api == 2 && ci == 2 && self.channels_internal == 1 {
			self.stereo.pred_prev_q13 = [0; 2];
			self.stereo.s_side = [0; 2];
			self.channel_state[1].resampler = self.channel_state[0].resampler.clone();
		}
		self.channels_internal = ci;

		let mut ms_pred_q13 = [0i32; 2];
		let mut decode_only_middle = false;
		if mode != SilkMode::Lost && self.channel_state[0].n_frames_decoded == 0 {
			// First call for this payload: VAD and LBRR flags.
			for n in 0..ci {
				let st = &mut self.channel_state[n];
				for i in 0..st.n_frames_per_packet {
					st.vad_flags[i] = dec.bit_logp(1);
				}
				st.lbrr_flag = dec.bit_logp(1);
			}
			for n in 0..ci {
				let st = &mut self.channel_state[n];
				st.lbrr_flags = [false; 3];
				if st.lbrr_flag {
					if st.n_frames_per_packet == 1 {
						st.lbrr_flags[0] = true;
					} else {
						let table: &[u8] =
							if st.n_frames_per_packet == 2 { &LBRR_FLAGS_2_I_CDF } else { &LBRR_FLAGS_3_I_CDF };
						let symbol = dec.icdf(table, 8) + 1;
						for i in 0..st.n_frames_per_packet {
							st.lbrr_flags[i] = (symbol >> i) & 1 != 0;
						}
					}
				}
			}
			if mode == SilkMode::Normal {
				// Regular decoding skips the LBRR (FEC) data.
				let frames = self.channel_state[0].n_frames_per_packet;
				for i in 0..frames {
					for n in 0..ci {
						if !self.channel_state[n].lbrr_flags[i] {
							continue;
						}
						if ci == 2 && n == 0 {
							ms_pred_q13 = stereo_decode_pred(dec);
							if !self.channel_state[1].lbrr_flags[i] {
								decode_only_middle = dec.icdf(&STEREO_ONLY_CODE_MID_I_CDF, 8) != 0;
							}
						}
						let cond = if i > 0 && self.channel_state[n].lbrr_flags[i - 1] {
							CODE_CONDITIONALLY
						} else {
							CODE_INDEPENDENTLY
						};
						let st = &mut self.channel_state[n];
						decode_indices(st, dec, i, true, cond);
						let mut pulses = vec![0i32; MAX_FRAME_LENGTH];
						decode_pulses(
							dec,
							&mut pulses,
							st.indices.signal_type,
							st.indices.quant_offset_type,
							st.frame_length,
						);
					}
				}
			}
		}

		// Mid/side predictor and the mid-only flag.
		if ci == 2 {
			let fi = self.channel_state[0].n_frames_decoded.min(2);
			let coded = match mode {
				SilkMode::Normal => true,
				SilkMode::Lbrr => self.channel_state[0].lbrr_flags[fi],
				SilkMode::Lost => false,
			};
			if coded {
				ms_pred_q13 = stereo_decode_pred(dec);
				let side_missing = match mode {
					SilkMode::Normal => !self.channel_state[1].vad_flags[fi],
					_ => !self.channel_state[1].lbrr_flags[fi],
				};
				decode_only_middle = side_missing && dec.icdf(&STEREO_ONLY_CODE_MID_I_CDF, 8) != 0;
			} else {
				ms_pred_q13 = self.stereo.pred_prev_q13;
			}
		}
		// Reset the side channel's prediction memory for its first coded frame.
		if ci == 2 && !decode_only_middle && self.prev_decode_only_middle {
			let st = &mut self.channel_state[1];
			st.out_buf.fill(0);
			st.s_lpc_q14_buf.fill(0);
			st.lag_prev = 100;
			st.last_gain_index = 10;
			st.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
			st.first_frame_after_reset = true;
		}

		let frame_length = self.channel_state[0].frame_length;
		if frame_length > MAX_FRAME_LENGTH || frame_length == 0 {
			return Err("invalid silk frame length");
		}
		let mut tmp = [vec![0i16; frame_length + 2], vec![0i16; frame_length + 2]];
		let has_side = match mode {
			SilkMode::Normal => !decode_only_middle,
			_ => {
				!self.prev_decode_only_middle
					|| (ci == 2
						&& mode == SilkMode::Lbrr
						&& self.channel_state[1].lbrr_flags[self.channel_state[1].n_frames_decoded.min(2)])
			}
		};
		let mut n_out_dec = frame_length;
		for n in 0..ci {
			if n == 0 || has_side {
				let frame_index = self.channel_state[0].n_frames_decoded as i32 - n as i32;
				let cond = if frame_index <= 0 {
					CODE_INDEPENDENTLY
				} else if mode == SilkMode::Lbrr {
					if self.channel_state[n].lbrr_flags[(frame_index - 1).min(2) as usize] {
						CODE_CONDITIONALLY
					} else {
						CODE_INDEPENDENTLY
					}
				} else if n > 0 && self.prev_decode_only_middle {
					CODE_INDEPENDENTLY_NO_LTP_SCALING
				} else {
					CODE_CONDITIONALLY
				};
				let st = &mut self.channel_state[n];
				if st.frame_length != frame_length {
					return Err("invalid silk frame length");
				}
				n_out_dec = decode_frame(st, dec, &mut tmp[n][2..], mode, cond);
			} else {
				tmp[n][2..2 + n_out_dec].fill(0);
			}
			self.channel_state[n].n_frames_decoded += 1;
		}
		if self.channels_api == 2 && ci == 2 {
			let fs = self.channel_state[0].fs_khz;
			let (a, b) = tmp.split_at_mut(1);
			stereo_ms_to_lr(&mut self.stereo, &mut a[0], &mut b[0], ms_pred_q13, fs, n_out_dec);
		} else {
			tmp[0][..2].copy_from_slice(&self.stereo.s_mid);
			self.stereo.s_mid.copy_from_slice(&tmp[0][n_out_dec..n_out_dec + 2]);
		}

		let n_out = n_out_dec * API_RATE as usize / (self.channel_state[0].fs_khz as usize * 1000);
		let api = self.channels_api;
		for n in 0..api.min(ci) {
			let resampled = self.channel_state[n].resampler.process(&tmp[n][1..1 + n_out_dec]);
			for (i, &v) in resampled.iter().take(n_out).enumerate() {
				out[n + api * i] = f32::from(v) * (1.0 / 32768.0);
			}
		}
		if api == 2 && ci == 1 {
			if stereo_to_mono {
				let resampled = self.channel_state[1].resampler.process(&tmp[0][1..1 + n_out_dec]);
				for (i, &v) in resampled.iter().take(n_out).enumerate() {
					out[1 + 2 * i] = f32::from(v) * (1.0 / 32768.0);
				}
			} else {
				for i in 0..n_out {
					out[1 + 2 * i] = out[2 * i];
				}
			}
		}
		if mode == SilkMode::Lost {
			// Remove the gain clamping so the energy does not "bounce back"
			// when packets are lost while it is going down.
			for st in &mut self.channel_state[..ci] {
				st.last_gain_index = 10;
			}
		} else {
			self.prev_decode_only_middle = decode_only_middle;
		}
		Ok(n_out)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn fixed_point_helpers() {
		assert_eq!(smulwb(1 << 16, 3), 3);
		assert_eq!(smulwb(-(1 << 16), 3), -3);
		assert_eq!(rshift_round(5, 1), 3);
		assert_eq!(rshift_round(-5, 2), -1);
		assert_eq!(log2lin(0), 1);
		assert_eq!(log2lin(7 << 7), 128);
		// Like libopus, the reciprocal is an approximation (within 1 LSB here).
		assert!((inverse32_var_q(1 << 16, 32) - (1 << 16)).abs() <= 1);
		assert!((div32_var_q(3 << 16, 1 << 16, 16) - (3 << 16)).abs() <= 2);
		assert_eq!(silk_rand(0), 907_633_515);
	}

	#[test]
	fn nlsf_to_lpc_is_stable() {
		// Evenly spread NLSFs describe a flat spectrum: a stable, small filter.
		let nlsf: Vec<i16> = (1..=16).map(|i| (i * 32768 / 17) as i16).collect();
		let a = nlsf2a(&nlsf, 16);
		assert!(lpc_inverse_pred_gain(&a[..16]) > 0);
		assert!(a.iter().all(|&c| c.abs() < 4096));
	}
}
