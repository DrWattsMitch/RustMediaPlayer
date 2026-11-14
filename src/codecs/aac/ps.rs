//! Parametric Stereo (ISO/IEC 14496-3 §8.6.4; FFmpeg `aacps.c`,
//! `aacps_common.c`, `aacpsdsp_template.c`, `aacps_tablegen.h`).
//!
//! Turns the mono SBR QMF signal of HE-AAC v2 into stereo: hybrid
//! analysis splits the low QMF bands further, a decorrelator builds a
//! second signal, and per-band mixing matrices (from the transmitted
//! intensity, coherence and phase parameters) produce left and right.

use super::sbr::{Cplx, vlcs_from_lengths};
use super::sbr_tables::{K_TO_I_20, K_TO_I_34, PS_HUFFMAN, PS_HUFFMAN_OFFSETS, PS_HUFFMAN_SIZES};
use crate::codecs::bitstream::BitReader;
use crate::codecs::bitstream::Vlc;
use std::f32::consts::{FRAC_1_SQRT_2, SQRT_2};
use std::sync::OnceLock;

const MAX_NUM_ENV: usize = 5;
const MAX_NR_IIDICC: usize = 34;
const MAX_SSB: usize = 91;
const MAX_AP_BANDS: usize = 50;
const QMF_TIME_SLOTS: usize = 32;
const MAX_DELAY: usize = 14;
const AP_LINKS: usize = 3;
const MAX_AP_DELAY: usize = 5;
const NUM_QMF_SLOTS: i32 = 32;

const NR_PAR_BANDS: [usize; 2] = [20, 34];
const NR_IPDOPD_BANDS: [usize; 2] = [11, 17];
const NR_BANDS: [usize; 2] = [71, 91];
const DECAY_CUTOFF: [i32; 2] = [10, 32];
const NR_ALLPASS_BANDS: [usize; 2] = [30, 50];
const SHORT_DELAY_BAND: [usize; 2] = [42, 62];
const DECAY_SLOPE: f32 = 0.05;

const NUM_ENV_TAB: [[usize; 4]; 2] = [[0, 1, 2, 4], [1, 2, 3, 4]];
const NR_IIDICC_PAR_TAB: [usize; 6] = [10, 20, 34, 10, 20, 34];
const NR_IIDOPD_PAR_TAB: [usize; 6] = [5, 11, 17, 5, 11, 17];

// Huffman table indices.
const IID_DF1: usize = 0;
const IID_DT1: usize = 1;
const IID_DF0: usize = 2;
const IID_DT0: usize = 3;
const ICC_DF: usize = 4;
const ICC_DT: usize = 5;
const IPD_DF: usize = 6;
const IPD_DT: usize = 7;
const OPD_DF: usize = 8;
const OPD_DT: usize = 9;
const HUFF_IID: [usize; 4] = [IID_DF0, IID_DF1, IID_DT0, IID_DT1];

type Filter = [[f32; 2]; 8];

struct Tables {
	vlc: Vec<Vlc>,
	pd_re_smooth: Vec<f32>,
	pd_im_smooth: Vec<f32>,
	ha: Box<[[[f32; 4]; 8]; 46]>,
	hb: Box<[[[f32; 4]; 8]; 46]>,
	f20_0_8: [Filter; 8],
	f34_0_12: [Filter; 12],
	f34_1_8: [Filter; 8],
	f34_2_4: [Filter; 4],
	q_fract_allpass: Box<[[[Cplx; 3]; 50]; 2]>,
	phi_fract: Box<[[Cplx; 50]; 2]>,
}

fn filters<const N: usize>(proto: &[f32; 7]) -> [Filter; N] {
	let mut f = [[[0.0; 2]; 8]; N];
	for (q, row) in f.iter_mut().enumerate() {
		for n in 0..7 {
			let theta = 2.0 * std::f64::consts::PI * (q as f64 + 0.5) * (n as f64 - 6.0) / N as f64;
			row[n] =
				[(f64::from(proto[n]) * theta.cos()) as f32, (f64::from(proto[n]) * -theta.sin()) as f32];
		}
	}
	f
}

fn tables() -> &'static Tables {
	static T: OnceLock<Tables> = OnceLock::new();
	T.get_or_init(|| {
		const S: f32 = FRAC_1_SQRT_2;
		let ipdopd_sin = [0.0, S, 1.0, S, 0.0, -S, -1.0, -S];
		let ipdopd_cos = [1.0, S, 0.0, -S, -1.0, -S, 0.0, S];
		let mut pd_re = vec![0.0f32; 512];
		let mut pd_im = vec![0.0f32; 512];
		for p0 in 0..8 {
			for p1 in 0..8 {
				for p2 in 0..8 {
					let re: f32 = 0.25 * ipdopd_cos[p0] + 0.5 * ipdopd_cos[p1] + ipdopd_cos[p2];
					let im: f32 = 0.25 * ipdopd_sin[p0] + 0.5 * ipdopd_sin[p1] + ipdopd_sin[p2];
					let mag = (1.0 / f64::from(im).hypot(f64::from(re))) as f32;
					pd_re[p0 * 64 + p1 * 8 + p2] = re * mag;
					pd_im[p0 * 64 + p1 * 8 + p2] = im * mag;
				}
			}
		}
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const IID_DEQUANT: [f32; 46] = [
			0.05623413251903,
			0.12589254117942,
			0.19952623149689,
			0.31622776601684,
			0.44668359215096,
			0.63095734448019,
			0.79432823472428,
			1.0,
			1.25892541179417,
			1.58489319246111,
			2.23872113856834,
			3.16227766016838,
			5.01187233627272,
			7.94328234724282,
			17.7827941003892,
			0.00316227766017,
			0.00562341325190,
			0.01,
			0.01778279410039,
			0.03162277660168,
			0.05623413251903,
			0.07943282347243,
			0.11220184543020,
			0.15848931924611,
			0.22387211385683,
			0.31622776601684,
			0.39810717055350,
			0.50118723362727,
			0.63095734448019,
			0.79432823472428,
			1.0,
			1.25892541179417,
			1.58489319246111,
			1.99526231496888,
			2.51188643150958,
			3.16227766016838,
			4.46683592150963,
			6.30957344480193,
			8.91250938133745,
			12.5892541179417,
			17.7827941003892,
			31.6227766016838,
			56.2341325190349,
			100.0,
			177.827941003892,
			316.227766016837,
		];
		const ICC_INVQ: [f32; 8] = [1.0, 0.937, 0.84118, 0.60092, 0.36764, 0.0, -0.589, -1.0];
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const ACOS_ICC_INVQ: [f32; 8] = [
			0.0,
			0.35685527,
			0.57133466,
			0.92614472,
			1.1943263,
			std::f32::consts::FRAC_PI_2,
			2.2006171,
			std::f32::consts::PI,
		];
		let mut ha = Box::new([[[0.0f32; 4]; 8]; 46]);
		let mut hb = Box::new([[[0.0f32; 4]; 8]; 46]);
		for iid in 0..46 {
			let c = IID_DEQUANT[iid];
			let c1 = SQRT_2 / (1.0 + c * c).sqrt();
			let c2 = c * c1;
			for icc in 0..8 {
				let alpha = 0.5 * ACOS_ICC_INVQ[icc];
				let beta = alpha * (c1 - c2) * FRAC_1_SQRT_2;
				ha[iid][icc] = [
					c2 * (beta + alpha).cos(),
					c1 * (beta - alpha).cos(),
					c2 * (beta + alpha).sin(),
					c1 * (beta - alpha).sin(),
				];
				let rho = ICC_INVQ[icc].max(0.05);
				let mut alpha = 0.5 * (2.0 * c * rho).atan2(c * c - 1.0);
				let mut mu = c + 1.0 / c;
				mu = (1.0 + (4.0 * rho * rho - 4.0) / (mu * mu)).sqrt();
				let gamma = ((1.0 - mu) / (1.0 + mu)).sqrt().atan();
				if alpha < 0.0 {
					alpha += std::f32::consts::FRAC_PI_2;
				}
				let (ac, as_, gc, gs) = (alpha.cos(), alpha.sin(), gamma.cos(), gamma.sin());
				hb[iid][icc] = [SQRT_2 * ac * gc, SQRT_2 * as_ * gc, -SQRT_2 * as_ * gs, SQRT_2 * ac * gs];
			}
		}
		const F_CENTER_20: [i8; 10] = [-3, -1, 1, 3, 5, 7, 10, 14, 18, 22];
		const F_CENTER_34: [i8; 32] = [
			2, 6, 10, 14, 18, 22, 26, 30, 34, -10, -6, -2, 51, 57, 15, 21, 27, 33, 39, 45, 54, 66, 78,
			42, 102, 66, 78, 90, 102, 114, 126, 90,
		];
		const FRACTIONAL_DELAY_LINKS: [f32; 3] = [0.43, 0.75, 0.347];
		const FRACTIONAL_DELAY_GAIN: f32 = 0.39;
		let mut q_fract = Box::new([[[[0.0f32; 2]; 3]; 50]; 2]);
		let mut phi_fract = Box::new([[[0.0f32; 2]; 50]; 2]);
		for is34 in 0..2 {
			for k in 0..NR_ALLPASS_BANDS[is34] {
				let f_center: f64 = if is34 == 0 {
					if k < 10 { f64::from(F_CENTER_20[k]) * 0.125 } else { f64::from(k as f32 - 6.5) }
				} else if k < 32 {
					f64::from(F_CENTER_34[k]) / 24.0
				} else {
					f64::from(k as f32 - 26.5)
				};
				for m in 0..AP_LINKS {
					let theta = -std::f64::consts::PI * f64::from(FRACTIONAL_DELAY_LINKS[m]) * f_center;
					q_fract[is34][k][m] = [theta.cos() as f32, theta.sin() as f32];
				}
				let theta = -std::f64::consts::PI * f64::from(FRACTIONAL_DELAY_GAIN) * f_center;
				phi_fract[is34][k] = [theta.cos() as f32, theta.sin() as f32];
			}
		}
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const G0_Q8: [f32; 7] = [
			0.00746082949812,
			0.02270420949825,
			0.04546865930473,
			0.07266113929591,
			0.09885108575264,
			0.11793710567217,
			0.125,
		];
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const G0_Q12: [f32; 7] = [
			0.04081179924692,
			0.03812810994926,
			0.05144908135699,
			0.06399831151592,
			0.07428313801106,
			0.08100347892914,
			0.08333333333333,
		];
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const G1_Q8: [f32; 7] = [
			0.01565675600122,
			0.03752716391991,
			0.05417891378782,
			0.08417044116767,
			0.10307344158036,
			0.12222452249753,
			0.125,
		];
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const G2_Q4: [f32; 7] = [
			-0.05908211155639,
			-0.04871498374946,
			0.0,
			0.07778723915851,
			0.16486303567403,
			0.23279856662996,
			0.25,
		];
		Tables {
			vlc: vlcs_from_lengths(&PS_HUFFMAN, &PS_HUFFMAN_SIZES, &PS_HUFFMAN_OFFSETS),
			pd_re_smooth: pd_re,
			pd_im_smooth: pd_im,
			ha,
			hb,
			f20_0_8: filters::<8>(&G0_Q8),
			f34_0_12: filters::<12>(&G0_Q12),
			f34_1_8: filters::<8>(&G1_Q8),
			f34_2_4: filters::<4>(&G2_Q4),
			q_fract_allpass: q_fract,
			phi_fract,
		}
	})
}

type ParSet = [[i8; MAX_NR_IIDICC]; MAX_NUM_ENV];

/// Parsed PS parameters (FFmpeg `PSCommonContext`).
#[derive(Clone)]
pub struct PsCommon {
	pub start: bool,
	enable_iid: bool,
	iid_quant: usize,
	nr_iid_par: usize,
	nr_ipdopd_par: usize,
	enable_icc: bool,
	icc_mode: u32,
	nr_icc_par: usize,
	enable_ext: bool,
	frame_class: usize,
	num_env_old: usize,
	num_env: usize,
	enable_ipdopd: bool,
	border_position: [i32; MAX_NUM_ENV + 1],
	iid_par: ParSet,
	icc_par: ParSet,
	ipd_par: ParSet,
	opd_par: ParSet,
	is34bands: bool,
	is34bands_old: bool,
}

impl Default for PsCommon {
	fn default() -> Self {
		Self {
			start: false,
			enable_iid: false,
			iid_quant: 0,
			nr_iid_par: 0,
			nr_ipdopd_par: 0,
			enable_icc: false,
			icc_mode: 0,
			nr_icc_par: 0,
			enable_ext: false,
			frame_class: 0,
			num_env_old: 0,
			num_env: 0,
			enable_ipdopd: false,
			border_position: [0; MAX_NUM_ENV + 1],
			iid_par: [[0; MAX_NR_IIDICC]; MAX_NUM_ENV],
			icc_par: [[0; MAX_NR_IIDICC]; MAX_NUM_ENV],
			ipd_par: [[0; MAX_NR_IIDICC]; MAX_NUM_ENV],
			opd_par: [[0; MAX_NR_IIDICC]; MAX_NUM_ENV],
			is34bands: false,
			is34bands_old: false,
		}
	}
}

#[derive(Clone)]
pub struct Ps {
	pub common: PsCommon,
	in_buf: Box<[[Cplx; 44]; 5]>,
	delay: Box<[[Cplx; QMF_TIME_SLOTS + MAX_DELAY]; MAX_SSB]>,
	ap_delay: Box<[[[Cplx; QMF_TIME_SLOTS + MAX_AP_DELAY]; AP_LINKS]; MAX_AP_BANDS]>,
	peak_decay_nrg: [f32; 34],
	power_smooth: [f32; 34],
	peak_decay_diff_smooth: [f32; 34],
	h11: [[[f32; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
	h12: [[[f32; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
	h21: [[[f32; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
	h22: [[[f32; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
	lbuf: Box<[[Cplx; 32]; 91]>,
	rbuf: Box<[[Cplx; 32]; 91]>,
	opd_hist: [i8; MAX_NR_IIDICC],
	ipd_hist: [i8; MAX_NR_IIDICC],
}

impl Default for Ps {
	fn default() -> Self {
		Self {
			common: PsCommon::default(),
			in_buf: Box::new([[[0.0; 2]; 44]; 5]),
			delay: Box::new([[[0.0; 2]; QMF_TIME_SLOTS + MAX_DELAY]; MAX_SSB]),
			ap_delay: Box::new([[[[0.0; 2]; QMF_TIME_SLOTS + MAX_AP_DELAY]; AP_LINKS]; MAX_AP_BANDS]),
			peak_decay_nrg: [0.0; 34],
			power_smooth: [0.0; 34],
			peak_decay_diff_smooth: [0.0; 34],
			h11: [[[0.0; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
			h12: [[[0.0; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
			h21: [[[0.0; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
			h22: [[[0.0; MAX_NR_IIDICC]; MAX_NUM_ENV + 1]; 2],
			lbuf: Box::new([[[0.0; 2]; 32]; 91]),
			rbuf: Box::new([[[0.0; 2]; 32]; 91]),
			opd_hist: [0; MAX_NR_IIDICC],
			ipd_hist: [0; MAX_NR_IIDICC],
		}
	}
}

enum ParKind {
	Iid,
	Icc,
	IpdOpd,
}

fn read_par(
	bs: &mut BitReader,
	ps: &PsCommon,
	par: &mut ParSet,
	table: usize,
	e: usize,
	dt: bool,
	kind: ParKind,
) -> bool {
	let vlc = &tables().vlc[table];
	let num = match kind {
		ParKind::Iid => ps.nr_iid_par,
		ParKind::Icc => ps.nr_icc_par,
		ParKind::IpdOpd => ps.nr_ipdopd_par,
	};
	let mask = matches!(kind, ParKind::IpdOpd);
	let bad = |v: i32| match kind {
		ParKind::Iid => v.abs() > 7 + 8 * ps.iid_quant as i32,
		ParKind::Icc => !(0..=7).contains(&v),
		ParKind::IpdOpd => false,
	};
	let read = |bs: &mut BitReader| vlc.decode(bs).map_or(0, |s| s as i32);
	if dt {
		let e_prev = if e > 0 { e - 1 } else { ps.num_env_old.saturating_sub(1) };
		for b in 0..num {
			let mut v = i32::from(par[e_prev][b]) + read(bs);
			if mask {
				v &= 7;
			}
			par[e][b] = v as i8;
			if bad(i32::from(par[e][b])) {
				return false;
			}
		}
	} else {
		let mut v = 0i32;
		for b in 0..num {
			v += read(bs);
			if mask {
				v &= 7;
			}
			par[e][b] = v as i8;
			if bad(i32::from(par[e][b])) {
				return false;
			}
		}
	}
	true
}

impl Ps {
	/// `ps_data()`; returns the bits consumed (all of `bits_left` on error).
	pub fn read_data(&mut self, bs: &mut BitReader, bits_left: usize) -> usize {
		let start = bs.pos();
		let mut gb = bs.clone();
		let ok = self.parse(&mut gb);
		let consumed = gb.pos() - start;
		if ok && consumed <= bits_left {
			bs.skip(consumed);
			return consumed;
		}
		let ps = &mut self.common;
		ps.start = false;
		ps.iid_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		ps.icc_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		ps.ipd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		ps.opd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		bs.skip(bits_left);
		bits_left
	}

	fn parse(&mut self, gb: &mut BitReader) -> bool {
		let ps = &mut self.common;
		let header = gb.bit();
		if header {
			ps.enable_iid = gb.bit();
			if ps.enable_iid {
				let mode = gb.bits(3) as usize;
				if mode > 5 {
					return false;
				}
				ps.nr_iid_par = NR_IIDICC_PAR_TAB[mode];
				ps.iid_quant = usize::from(mode > 2);
				ps.nr_ipdopd_par = NR_IIDOPD_PAR_TAB[mode];
			}
			ps.enable_icc = gb.bit();
			if ps.enable_icc {
				ps.icc_mode = gb.bits(3);
				if ps.icc_mode > 5 {
					return false;
				}
				ps.nr_icc_par = NR_IIDICC_PAR_TAB[ps.icc_mode as usize];
			}
			ps.enable_ext = gb.bit();
		}
		ps.frame_class = gb.bits(1) as usize;
		ps.num_env_old = ps.num_env;
		ps.num_env = NUM_ENV_TAB[ps.frame_class][gb.bits(2) as usize];
		ps.border_position[0] = -1;
		if ps.frame_class != 0 {
			for e in 1..=ps.num_env {
				ps.border_position[e] = gb.bits(5) as i32;
				if ps.border_position[e] < ps.border_position[e - 1] {
					return false;
				}
			}
		} else {
			let log2 = match ps.num_env {
				1 => 0,
				2 => 1,
				4 => 2,
				_ => 0,
			};
			for e in 1..=ps.num_env {
				ps.border_position[e] = ((e as i32 * NUM_QMF_SLOTS) >> log2) - 1;
			}
		}
		if ps.enable_iid {
			for e in 0..ps.num_env {
				let dt = gb.bit();
				let mut par = ps.iid_par;
				let ok = read_par(
					gb,
					ps,
					&mut par,
					HUFF_IID[2 * usize::from(dt) + ps.iid_quant],
					e,
					dt,
					ParKind::Iid,
				);
				ps.iid_par = par;
				if !ok {
					return false;
				}
			}
		} else {
			ps.iid_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		}
		if ps.enable_icc {
			for e in 0..ps.num_env {
				let dt = gb.bit();
				let mut par = ps.icc_par;
				let ok = read_par(gb, ps, &mut par, if dt { ICC_DT } else { ICC_DF }, e, dt, ParKind::Icc);
				ps.icc_par = par;
				if !ok {
					return false;
				}
			}
		} else {
			ps.icc_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		}
		if ps.enable_ext {
			let mut cnt = gb.bits(4) as i64;
			if cnt == 15 {
				cnt += i64::from(gb.bits(8));
			}
			cnt *= 8;
			while cnt > 7 {
				let id = gb.bits(2);
				let before = gb.pos();
				if id == 0 {
					ps.enable_ipdopd = gb.bit();
					if ps.enable_ipdopd {
						for e in 0..ps.num_env {
							let dt = gb.bit();
							let mut par = ps.ipd_par;
							read_par(gb, ps, &mut par, if dt { IPD_DT } else { IPD_DF }, e, dt, ParKind::IpdOpd);
							ps.ipd_par = par;
							let dt = gb.bit();
							let mut par = ps.opd_par;
							read_par(gb, ps, &mut par, if dt { OPD_DT } else { OPD_DF }, e, dt, ParKind::IpdOpd);
							ps.opd_par = par;
						}
					}
					gb.skip(1);
				}
				cnt -= 2 + (gb.pos() - before) as i64;
			}
			if cnt < 0 {
				return false;
			}
			gb.skip(cnt as usize);
		}
		// Fix up envelopes: the last border must reach the end of the frame.
		if ps.num_env == 0 || ps.border_position[ps.num_env] < NUM_QMF_SLOTS - 1 {
			let source = if ps.num_env > 0 { ps.num_env as i32 - 1 } else { ps.num_env_old as i32 - 1 };
			let ne = ps.num_env;
			if ne >= MAX_NUM_ENV {
				return false;
			}
			if source >= 0 && source as usize != ne {
				let s = source as usize;
				if ps.enable_iid {
					ps.iid_par[ne] = ps.iid_par[s];
				}
				if ps.enable_icc {
					ps.icc_par[ne] = ps.icc_par[s];
				}
				if ps.enable_ipdopd {
					ps.ipd_par[ne] = ps.ipd_par[s];
					ps.opd_par[ne] = ps.opd_par[s];
				}
			}
			if ps.enable_iid
				&& ps.iid_par[ne][..ps.nr_iid_par]
					.iter()
					.any(|&v| i32::from(v).abs() > 7 + 8 * ps.iid_quant as i32)
			{
				return false;
			}
			if ps.enable_icc && ps.icc_par[ne][..ps.nr_iid_par].iter().any(|&v| !(0..=7).contains(&v)) {
				return false;
			}
			ps.num_env += 1;
			ps.border_position[ps.num_env] = NUM_QMF_SLOTS - 1;
		}
		ps.is34bands_old = ps.is34bands;
		if ps.enable_iid || ps.enable_icc {
			ps.is34bands =
				(ps.enable_iid && ps.nr_iid_par == 34) || (ps.enable_icc && ps.nr_icc_par == 34);
		}
		if !ps.enable_ipdopd {
			ps.ipd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
			ps.opd_par = [[0; MAX_NR_IIDICC]; MAX_NUM_ENV];
		}
		if header {
			ps.start = true;
		}
		true
	}

	/// Applies PS to the QMF signal `l` (mono in, left out) producing `r`.
	/// Both are `[re/im][slot][band]`.
	pub fn apply(&mut self, l: &mut [[[f32; 64]; 38]; 2], r: &mut [[[f32; 64]; 38]; 2], top: usize) {
		let is34 = usize::from(self.common.is34bands);
		let top = (top + NR_BANDS[is34]).saturating_sub(64);
		for row in self.delay[top.min(NR_BANDS[is34])..NR_BANDS[is34]].iter_mut() {
			*row = [[0.0; 2]; QMF_TIME_SLOTS + MAX_DELAY];
		}
		if top < NR_ALLPASS_BANDS[is34] {
			for row in self.ap_delay[top..NR_ALLPASS_BANDS[is34]].iter_mut() {
				*row = [[[0.0; 2]; QMF_TIME_SLOTS + MAX_AP_DELAY]; AP_LINKS];
			}
		}
		self.hybrid_analysis(l, is34 == 1);
		self.decorrelation(is34);
		self.stereo_processing(is34);
		hybrid_synthesis(l, &self.lbuf, is34 == 1);
		hybrid_synthesis(r, &self.rbuf, is34 == 1);
	}

	fn hybrid_analysis(&mut self, l: &[[[f32; 64]; 38]; 2], is34: bool) {
		let t = tables();
		let len = 32;
		for i in 0..5 {
			for j in 0..38 {
				self.in_buf[i][j + 6] = [l[0][j][i], l[1][j][i]];
			}
		}
		let out = &mut self.lbuf;
		if is34 {
			hybrid_cx(&self.in_buf[0], out, 0, &t.f34_0_12, len);
			hybrid_cx(&self.in_buf[1], out, 12, &t.f34_1_8, len);
			hybrid_cx(&self.in_buf[2], out, 20, &t.f34_2_4, len);
			hybrid_cx(&self.in_buf[3], out, 24, &t.f34_2_4, len);
			hybrid_cx(&self.in_buf[4], out, 28, &t.f34_2_4, len);
			ileave(out, 27, l, 5, len);
		} else {
			hybrid6_cx(&self.in_buf[0], out, &t.f20_0_8, len);
			hybrid2_re(&self.in_buf[1], out, 6, len, true);
			hybrid2_re(&self.in_buf[2], out, 8, len, false);
			ileave(out, 7, l, 3, len);
		}
		for row in self.in_buf.iter_mut() {
			row.copy_within(32..38, 0);
		}
	}

	fn decorrelation(&mut self, is34: usize) {
		let t = tables();
		let k_to_i: &[u8] = if is34 == 1 { &K_TO_I_34 } else { &K_TO_I_20 };
		let mut power = [[0f32; QMF_TIME_SLOTS]; 34];
		let mut transient_gain = [[0f32; QMF_TIME_SLOTS]; 34];
		let n_l = 32;
		const PEAK_DECAY_FACTOR: f32 = 0.765_928_3;
		const TRANSIENT_IMPACT: f32 = 1.5;
		const A_SMOOTH: f32 = 0.25;
		if self.common.is34bands != self.common.is34bands_old {
			self.peak_decay_nrg = [0.0; 34];
			self.power_smooth = [0.0; 34];
			self.peak_decay_diff_smooth = [0.0; 34];
			for row in self.delay.iter_mut() {
				*row = [[0.0; 2]; QMF_TIME_SLOTS + MAX_DELAY];
			}
			for row in self.ap_delay.iter_mut() {
				*row = [[[0.0; 2]; QMF_TIME_SLOTS + MAX_AP_DELAY]; AP_LINKS];
			}
		}
		let s = &self.lbuf;
		for k in 0..NR_BANDS[is34] {
			let i = usize::from(k_to_i[k]);
			for n in 0..n_l {
				power[i][n] += s[k][n][0] * s[k][n][0] + s[k][n][1] * s[k][n][1];
			}
		}
		for i in 0..NR_PAR_BANDS[is34] {
			for n in 0..n_l {
				let decayed = PEAK_DECAY_FACTOR * self.peak_decay_nrg[i];
				self.peak_decay_nrg[i] = decayed.max(power[i][n]);
				self.power_smooth[i] += A_SMOOTH * (power[i][n] - self.power_smooth[i]);
				self.peak_decay_diff_smooth[i] +=
					A_SMOOTH * (self.peak_decay_nrg[i] - power[i][n] - self.peak_decay_diff_smooth[i]);
				let denom = TRANSIENT_IMPACT * self.peak_decay_diff_smooth[i];
				transient_gain[i][n] =
					if denom > self.power_smooth[i] { self.power_smooth[i] / denom } else { 1.0 };
			}
		}
		#[allow(clippy::excessive_precision)] // FFmpeg's values, digit for digit
		const A: [f32; 3] = [0.651_439_06, 0.564_718_1, 0.489_541_66];
		let out = &mut self.rbuf;
		let mut k = 0;
		while k < NR_ALLPASS_BANDS[is34] {
			let b = usize::from(k_to_i[k]);
			let g_decay_slope =
				(1.0 - DECAY_SLOPE * (k as i32 - DECAY_CUTOFF[is34]) as f32).clamp(0.0, 1.0);
			let delay = &mut self.delay[k];
			delay.copy_within(n_l..n_l + MAX_DELAY, 0);
			delay[MAX_DELAY..MAX_DELAY + 32].copy_from_slice(&s[k]);
			let ap = &mut self.ap_delay[k];
			for m in 0..AP_LINKS {
				ap[m].copy_within(32..37, 0);
			}
			let phi = t.phi_fract[is34][k];
			let q = &t.q_fract_allpass[is34][k];
			let ag = [A[0] * g_decay_slope, A[1] * g_decay_slope, A[2] * g_decay_slope];
			for n in 0..n_l {
				let dl = delay[MAX_DELAY - 2 + n];
				let mut in_re = dl[0] * phi[0] - dl[1] * phi[1];
				let mut in_im = dl[0] * phi[1] + dl[1] * phi[0];
				for m in 0..AP_LINKS {
					let a_re = ag[m] * in_re;
					let a_im = ag[m] * in_im;
					let link = ap[m][n + 2 - m];
					let (apd_re, apd_im) = (in_re, in_im);
					in_re = link[0] * q[m][0] - link[1] * q[m][1];
					in_re -= a_re;
					in_im = link[0] * q[m][1] + link[1] * q[m][0];
					in_im -= a_im;
					ap[m][n + 5] = [apd_re + ag[m] * in_re, apd_im + ag[m] * in_im];
				}
				out[k][n] = [transient_gain[b][n] * in_re, transient_gain[b][n] * in_im];
			}
			k += 1;
		}
		while k < NR_BANDS[is34] {
			let i = usize::from(k_to_i[k]);
			let delay = &mut self.delay[k];
			delay.copy_within(n_l..n_l + MAX_DELAY, 0);
			delay[MAX_DELAY..MAX_DELAY + 32].copy_from_slice(&s[k]);
			let d = if k < SHORT_DELAY_BAND[is34] { 14 } else { 1 };
			for n in 0..n_l {
				let v = delay[MAX_DELAY - d + n];
				out[k][n] = [v[0] * transient_gain[i][n], v[1] * transient_gain[i][n]];
			}
			k += 1;
		}
	}

	fn stereo_processing(&mut self, is34: usize) {
		let t = tables();
		let ps2 = &self.common;
		let k_to_i: &[u8] = if is34 == 1 { &K_TO_I_34 } else { &K_TO_I_20 };
		let h_lut = if ps2.icc_mode < 3 { &t.ha } else { &t.hb };
		let ne_old = ps2.num_env_old;
		if ne_old != 0 {
			for h in [&mut self.h11, &mut self.h12, &mut self.h21, &mut self.h22] {
				h[0][0] = h[0][ne_old.min(MAX_NUM_ENV)];
				h[1][0] = h[1][ne_old.min(MAX_NUM_ENV)];
			}
		}
		let remap = |par: &ParSet, num_par: usize, full: bool| -> ParSet {
			let mut out = *par;
			for e in 0..ps2.num_env.min(MAX_NUM_ENV) {
				if is34 == 1 {
					if num_par == 20 || num_par == 11 {
						map_idx_20_to_34(&mut out[e], &par[e], full);
					} else if num_par == 10 || num_par == 5 {
						map_idx_10_to_34(&mut out[e], &par[e], full);
					}
				} else if num_par == 34 || num_par == 17 {
					map_idx_34_to_20(&mut out[e], &par[e], full);
				} else if num_par == 10 || num_par == 5 {
					map_idx_10_to_20(&mut out[e], &par[e], full);
				}
			}
			out
		};
		let iid_mapped = remap(&ps2.iid_par, ps2.nr_iid_par, true);
		let icc_mapped = remap(&ps2.icc_par, ps2.nr_icc_par, true);
		let (ipd_mapped, opd_mapped) = if ps2.enable_ipdopd {
			(remap(&ps2.ipd_par, ps2.nr_ipdopd_par, false), remap(&ps2.opd_par, ps2.nr_ipdopd_par, false))
		} else {
			(ps2.ipd_par, ps2.opd_par)
		};
		if is34 == 1 && !ps2.is34bands_old {
			for h in [&mut self.h11, &mut self.h12, &mut self.h21, &mut self.h22] {
				map_val_20_to_34(&mut h[0][0]);
				map_val_20_to_34(&mut h[1][0]);
			}
			self.ipd_hist = [0; MAX_NR_IIDICC];
			self.opd_hist = [0; MAX_NR_IIDICC];
		} else if is34 == 0 && ps2.is34bands_old {
			for h in [&mut self.h11, &mut self.h12, &mut self.h21, &mut self.h22] {
				map_val_34_to_20(&mut h[0][0]);
				map_val_34_to_20(&mut h[1][0]);
			}
			self.ipd_hist = [0; MAX_NR_IIDICC];
			self.opd_hist = [0; MAX_NR_IIDICC];
		}
		let ipdopd = ps2.enable_ipdopd;
		let iid_quant = ps2.iid_quant;
		let num_env = ps2.num_env;
		let borders = ps2.border_position;
		for e in 0..num_env.min(MAX_NUM_ENV) {
			for b in 0..NR_PAR_BANDS[is34] {
				let iid = (i32::from(iid_mapped[e][b]) + 7 + 23 * iid_quant as i32).clamp(0, 45) as usize;
				let icc = (icc_mapped[e][b] as usize).min(7);
				let [mut h11, mut h12, mut h21, mut h22] = h_lut[iid][icc];
				if ipdopd && b < NR_IPDOPD_BANDS[is34] {
					let opd_idx =
						(i32::from(self.opd_hist[b]) * 8 + i32::from(opd_mapped[e][b])) as usize & 511;
					let ipd_idx =
						(i32::from(self.ipd_hist[b]) * 8 + i32::from(ipd_mapped[e][b])) as usize & 511;
					let (opd_re, opd_im) = (t.pd_re_smooth[opd_idx], t.pd_im_smooth[opd_idx]);
					let (ipd_re, ipd_im) = (t.pd_re_smooth[ipd_idx], t.pd_im_smooth[ipd_idx]);
					self.opd_hist[b] = (opd_idx & 0x3F) as i8;
					self.ipd_hist[b] = (ipd_idx & 0x3F) as i8;
					let adj_re = opd_re * ipd_re + opd_im * ipd_im;
					let adj_im = opd_im * ipd_re - opd_re * ipd_im;
					let h11i = h11 * opd_im;
					h11 *= opd_re;
					let h12i = h12 * adj_im;
					h12 *= adj_re;
					let h21i = h21 * opd_im;
					h21 *= opd_re;
					let h22i = h22 * adj_im;
					h22 *= adj_re;
					self.h11[1][e + 1][b] = h11i;
					self.h12[1][e + 1][b] = h12i;
					self.h21[1][e + 1][b] = h21i;
					self.h22[1][e + 1][b] = h22i;
				}
				self.h11[0][e + 1][b] = h11;
				self.h12[0][e + 1][b] = h12;
				self.h21[0][e + 1][b] = h21;
				self.h22[0][e + 1][b] = h22;
			}
			for k in 0..NR_BANDS[is34] {
				let start = borders[e];
				let stop = borders[e + 1];
				let width = 1.0 / if stop - start != 0 { (stop - start) as f32 } else { 1.0 };
				let b = usize::from(k_to_i[k]);
				let mut h = [[0f32; 4]; 2];
				h[0] = [self.h11[0][e][b], self.h12[0][e][b], self.h21[0][e][b], self.h22[0][e][b]];
				if ipdopd {
					let neg = (is34 == 1 && (9..=13).contains(&k)) || (is34 == 0 && k <= 1);
					let s = if neg { -1.0 } else { 1.0 };
					h[1] = [
						s * self.h11[1][e][b],
						s * self.h12[1][e][b],
						s * self.h21[1][e][b],
						s * self.h22[1][e][b],
					];
				}
				let next = [
					[
						self.h11[0][e + 1][b],
						self.h12[0][e + 1][b],
						self.h21[0][e + 1][b],
						self.h22[0][e + 1][b],
					],
					[
						self.h11[1][e + 1][b],
						self.h12[1][e + 1][b],
						self.h21[1][e + 1][b],
						self.h22[1][e + 1][b],
					],
				];
				let mut step = [[0f32; 4]; 2];
				for j in 0..4 {
					step[0][j] = (next[0][j] - h[0][j]) * width;
					if ipdopd {
						step[1][j] = (next[1][j] - h[1][j]) * width;
					}
				}
				if stop - start != 0 {
					let lo = (1 + start) as usize;
					let n = (stop - start) as usize;
					let (l, r) = (&mut self.lbuf[k], &mut self.rbuf[k]);
					stereo_interpolate(&mut l[lo..lo + n], &mut r[lo..lo + n], h, step, ipdopd);
				}
			}
		}
	}
}

fn stereo_interpolate(
	l: &mut [Cplx],
	r: &mut [Cplx],
	h: [[f32; 4]; 2],
	step: [[f32; 4]; 2],
	ipdopd: bool,
) {
	let [mut h00, mut h01, mut h02, mut h03] = h[0];
	let [mut h10, mut h11, mut h12, mut h13] = h[1];
	for (lv, rv) in l.iter_mut().zip(r.iter_mut()) {
		let (l_re, l_im, r_re, r_im) = (lv[0], lv[1], rv[0], rv[1]);
		h00 += step[0][0];
		h01 += step[0][1];
		h02 += step[0][2];
		h03 += step[0][3];
		if ipdopd {
			h10 += step[1][0];
			h11 += step[1][1];
			h12 += step[1][2];
			h13 += step[1][3];
			lv[0] = h00 * l_re + h02 * r_re - h10 * l_im - h12 * r_im;
			lv[1] = h00 * l_im + h02 * r_im + h10 * l_re + h12 * r_re;
			rv[0] = h01 * l_re + h03 * r_re - h11 * l_im - h13 * r_im;
			rv[1] = h01 * l_im + h03 * r_im + h11 * l_re + h13 * r_re;
		} else {
			lv[0] = h00 * l_re + h02 * r_re;
			lv[1] = h00 * l_im + h02 * r_im;
			rv[0] = h01 * l_re + h03 * r_re;
			rv[1] = h01 * l_im + h03 * r_im;
		}
	}
}

/// `ps_hybrid_analysis_c` for one input slot window.
#[inline]
fn hybrid_filter(input: &[Cplx], filter: &[Filter], out: &mut [Cplx]) {
	let mut inre0 = [0f32; 6];
	let mut inre1 = [0f32; 6];
	let mut inim0 = [0f32; 6];
	let mut inim1 = [0f32; 6];
	for j in 0..6 {
		inre0[j] = input[j][0] + input[12 - j][0];
		inre1[j] = input[j][1] - input[12 - j][1];
		inim0[j] = input[j][1] + input[12 - j][1];
		inim1[j] = input[j][0] - input[12 - j][0];
	}
	for (i, f) in filter.iter().enumerate() {
		let mut sum_re = f[6][0] * input[6][0];
		let mut sum_im = f[6][0] * input[6][1];
		for j in 0..6 {
			sum_re += f[j][0] * inre0[j] - f[j][1] * inre1[j];
			sum_im += f[j][0] * inim0[j] + f[j][1] * inim1[j];
		}
		out[i] = [sum_re, sum_im];
	}
}

fn hybrid_cx(
	input: &[Cplx; 44],
	out: &mut [[Cplx; 32]; 91],
	base: usize,
	filter: &[Filter],
	len: usize,
) {
	let mut tmp = [[0f32; 2]; 12];
	for i in 0..len {
		hybrid_filter(&input[i..i + 13], filter, &mut tmp[..filter.len()]);
		for (q, v) in tmp[..filter.len()].iter().enumerate() {
			out[base + q][i] = *v;
		}
	}
}

fn hybrid6_cx(input: &[Cplx; 44], out: &mut [[Cplx; 32]; 91], filter: &[Filter; 8], len: usize) {
	let mut t = [[0f32; 2]; 8];
	for i in 0..len {
		hybrid_filter(&input[i..i + 13], filter, &mut t);
		out[0][i] = t[6];
		out[1][i] = t[7];
		out[2][i] = t[0];
		out[3][i] = t[1];
		out[4][i] = [t[2][0] + t[5][0], t[2][1] + t[5][1]];
		out[5][i] = [t[3][0] + t[4][0], t[3][1] + t[4][1]];
	}
}

fn hybrid2_re(
	input: &[Cplx; 44],
	out: &mut [[Cplx; 32]; 91],
	base: usize,
	len: usize,
	reverse: bool,
) {
	const G1_Q2: [f32; 7] = [0.0, 0.018_994_875, 0.0, -0.072_931_39, 0.0, 0.305_966_3, 0.5];
	for i in 0..len {
		let inp = &input[i..i + 13];
		let re_in = G1_Q2[6] * inp[6][0];
		let im_in = G1_Q2[6] * inp[6][1];
		let (mut re_op, mut im_op) = (0.0f32, 0.0f32);
		let mut j = 0;
		while j < 6 {
			re_op += G1_Q2[j + 1] * (inp[j + 1][0] + inp[12 - j - 1][0]);
			im_op += G1_Q2[j + 1] * (inp[j + 1][1] + inp[12 - j - 1][1]);
			j += 2;
		}
		let (a, b) = if reverse { (base + 1, base) } else { (base, base + 1) };
		out[a][i] = [re_in + re_op, im_in + im_op];
		out[b][i] = [re_in - re_op, im_in - im_op];
	}
}

fn ileave(
	out: &mut [[Cplx; 32]; 91],
	first: usize,
	l: &[[[f32; 64]; 38]; 2],
	i0: usize,
	len: usize,
) {
	for i in i0..64 {
		for j in 0..len {
			out[first + i][j] = [l[0][j][i], l[1][j][i]];
		}
	}
}

fn hybrid_synthesis(out: &mut [[[f32; 64]; 38]; 2], input: &[[Cplx; 32]; 91], is34: bool) {
	let len = 32;
	let sum =
		|rng: std::ops::Range<usize>, n: usize, c: usize| rng.map(|i| input[i][n][c]).sum::<f32>();
	if is34 {
		for n in 0..len {
			for c in 0..2 {
				out[c][n][0] = sum(0..12, n, c);
				out[c][n][1] = sum(12..20, n, c);
				out[c][n][2] = sum(20..24, n, c);
				out[c][n][3] = sum(24..28, n, c);
				out[c][n][4] = sum(28..32, n, c);
			}
		}
		for i in 5..64 {
			for n in 0..len {
				out[0][n][i] = input[27 + i][n][0];
				out[1][n][i] = input[27 + i][n][1];
			}
		}
	} else {
		for n in 0..len {
			for c in 0..2 {
				out[c][n][0] = input[0][n][c]
					+ input[1][n][c]
					+ input[2][n][c]
					+ input[3][n][c]
					+ input[4][n][c]
					+ input[5][n][c];
				out[c][n][1] = input[6][n][c] + input[7][n][c];
				out[c][n][2] = input[8][n][c] + input[9][n][c];
			}
		}
		for i in 3..64 {
			for n in 0..len {
				out[0][n][i] = input[7 + i][n][0];
				out[1][n][i] = input[7 + i][n][1];
			}
		}
	}
}

fn map_idx_10_to_20(out: &mut [i8; 34], par: &[i8; 34], full: bool) {
	let mut b: i32 = if full { 9 } else { 4 };
	if !full {
		out[10] = 0;
	}
	while b >= 0 {
		let v = par[b as usize];
		out[2 * b as usize + 1] = v;
		out[2 * b as usize] = v;
		b -= 1;
	}
}

fn map_idx_34_to_20(out: &mut [i8; 34], p: &[i8; 34], full: bool) {
	let p = |i: usize| i32::from(p[i]);
	out[0] = ((2 * p(0) + p(1)) / 3) as i8;
	out[1] = ((p(1) + 2 * p(2)) / 3) as i8;
	out[2] = ((2 * p(3) + p(4)) / 3) as i8;
	out[3] = ((p(4) + 2 * p(5)) / 3) as i8;
	out[4] = ((p(6) + p(7)) / 2) as i8;
	out[5] = ((p(8) + p(9)) / 2) as i8;
	out[6] = p(10) as i8;
	out[7] = p(11) as i8;
	out[8] = ((p(12) + p(13)) / 2) as i8;
	out[9] = ((p(14) + p(15)) / 2) as i8;
	out[10] = p(16) as i8;
	if full {
		out[11] = p(17) as i8;
		out[12] = p(18) as i8;
		out[13] = p(19) as i8;
		out[14] = ((p(20) + p(21)) / 2) as i8;
		out[15] = ((p(22) + p(23)) / 2) as i8;
		out[16] = ((p(24) + p(25)) / 2) as i8;
		out[17] = ((p(26) + p(27)) / 2) as i8;
		out[18] = ((p(28) + p(29) + p(30) + p(31)) / 4) as i8;
		out[19] = ((p(32) + p(33)) / 2) as i8;
	}
}

fn map_idx_10_to_34(out: &mut [i8; 34], p: &[i8; 34], full: bool) {
	if full {
		for i in 28..34 {
			out[i] = p[9];
		}
		for i in 24..28 {
			out[i] = p[8];
		}
		for i in 20..24 {
			out[i] = p[7];
		}
		out[19] = p[6];
		out[18] = p[6];
		out[17] = p[5];
		out[16] = p[5];
	} else {
		out[16] = 0;
	}
	for i in 12..16 {
		out[i] = p[4];
	}
	out[11] = p[3];
	out[10] = p[3];
	for i in 6..10 {
		out[i] = p[2];
	}
	for i in 3..6 {
		out[i] = p[1];
	}
	for i in 0..3 {
		out[i] = p[0];
	}
}

fn map_idx_20_to_34(out: &mut [i8; 34], p: &[i8; 34], full: bool) {
	if full {
		out[33] = p[19];
		out[32] = p[19];
		out[31] = p[18];
		out[30] = p[18];
		out[29] = p[18];
		out[28] = p[18];
		out[27] = p[17];
		out[26] = p[17];
		out[25] = p[16];
		out[24] = p[16];
		out[23] = p[15];
		out[22] = p[15];
		out[21] = p[14];
		out[20] = p[14];
		out[19] = p[13];
		out[18] = p[12];
		out[17] = p[11];
	}
	out[16] = p[10];
	out[15] = p[9];
	out[14] = p[9];
	out[13] = p[8];
	out[12] = p[8];
	out[11] = p[7];
	out[10] = p[6];
	out[9] = p[5];
	out[8] = p[5];
	out[7] = p[4];
	out[6] = p[4];
	out[5] = p[3];
	out[4] = ((i32::from(p[2]) + i32::from(p[3])) / 2) as i8;
	out[3] = p[2];
	out[2] = p[1];
	out[1] = ((i32::from(p[0]) + i32::from(p[1])) / 2) as i8;
	out[0] = p[0];
}

fn map_val_34_to_20(p: &mut [f32; 34]) {
	p[0] = (2.0 * p[0] + p[1]) * 0.333_333_33;
	p[1] = (p[1] + 2.0 * p[2]) * 0.333_333_33;
	p[2] = (2.0 * p[3] + p[4]) * 0.333_333_33;
	p[3] = (p[4] + 2.0 * p[5]) * 0.333_333_33;
	p[4] = (p[6] + p[7]) * 0.5;
	p[5] = (p[8] + p[9]) * 0.5;
	p[6] = p[10];
	p[7] = p[11];
	p[8] = (p[12] + p[13]) * 0.5;
	p[9] = (p[14] + p[15]) * 0.5;
	p[10] = p[16];
	p[11] = p[17];
	p[12] = p[18];
	p[13] = p[19];
	p[14] = (p[20] + p[21]) * 0.5;
	p[15] = (p[22] + p[23]) * 0.5;
	p[16] = (p[24] + p[25]) * 0.5;
	p[17] = (p[26] + p[27]) * 0.5;
	p[18] = (p[28] + p[29] + p[30] + p[31]) * 0.25;
	p[19] = (p[32] + p[33]) * 0.5;
}

fn map_val_20_to_34(p: &mut [f32; 34]) {
	p[33] = p[19];
	p[32] = p[19];
	p[31] = p[18];
	p[30] = p[18];
	p[29] = p[18];
	p[28] = p[18];
	p[27] = p[17];
	p[26] = p[17];
	p[25] = p[16];
	p[24] = p[16];
	p[23] = p[15];
	p[22] = p[15];
	p[21] = p[14];
	p[20] = p[14];
	p[19] = p[13];
	p[18] = p[12];
	p[17] = p[11];
	p[16] = p[10];
	p[15] = p[9];
	p[14] = p[9];
	p[13] = p[8];
	p[12] = p[8];
	p[11] = p[7];
	p[10] = p[6];
	p[9] = p[5];
	p[8] = p[5];
	p[7] = p[4];
	p[6] = p[4];
	p[5] = p[3];
	p[4] = (p[2] + p[3]) * 0.5;
	p[3] = p[2];
	p[2] = p[1];
	p[1] = (p[0] + p[1]) * 0.5;
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tables_build() {
		let t = tables();
		assert_eq!(t.vlc.len(), 10);
		// Unit-magnitude phase smoothing.
		let (re, im) = (t.pd_re_smooth[0], t.pd_im_smooth[0]);
		assert!((re * re + im * im - 1.0).abs() < 1e-5);
	}
}
