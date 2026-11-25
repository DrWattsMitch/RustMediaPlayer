//! The adaptive CDFs a tile codes with (the subset of rav1d's
//! `CdfContext` this encoder's tools use), initialised to the defaults for
//! a frame's quantizer (`primary_ref_frame` = none).

use super::tables as t;

#[derive(Clone)]
pub struct MvComponentCdf {
	pub classes: [u16; 11],
	pub class_0_fp: [[u16; 4]; 2],
	pub class_n_fp: [u16; 4],
	pub class_0_hp: [u16; 2],
	pub class_n_hp: [u16; 2],
	pub class_0: [u16; 2],
	pub class_n: [[u16; 2]; 10],
	pub sign: [u16; 2],
}

#[derive(Clone)]
pub struct CdfContext {
	// Coefficients (for the frame's quantizer context).
	pub eob_bin_16: [[[u16; 5]; 2]; 2],
	pub eob_bin_64: [[[u16; 7]; 2]; 2],
	pub eob_bin_256: [[[u16; 9]; 2]; 2],
	pub eob_bin_1024: [[u16; 11]; 2],
	pub eob_base_tok: [[[[u16; 3]; 4]; 2]; 5],
	pub base_tok: [[[[u16; 4]; 41]; 2]; 5],
	pub br_tok: [[[[u16; 4]; 21]; 2]; 4],
	pub eob_hi_bit: [[[[u16; 2]; 11]; 2]; 5],
	pub coef_skip: [[[u16; 2]; 13]; 5],
	pub dc_sign: [[[u16; 2]; 3]; 2],
	// Modes.
	pub kf_y_mode: [[[u16; 13]; 5]; 5],
	pub y_mode: [[u16; 13]; 4],
	pub uv_mode_cfl: [[u16; 14]; 13],
	pub uv_mode_no_cfl: [[u16; 13]; 13],
	pub angle_delta: [[u16; 7]; 8],
	pub partition_64: [[u16; 10]; 4],
	pub partition_32: [[u16; 10]; 4],
	pub partition_16: [[u16; 10]; 4],
	pub partition_8: [[u16; 4]; 4],
	pub txtp_intra2: [[[u16; 5]; 13]; 3],
	pub txtp_inter3: [[u16; 2]; 4],
	pub txsz: [[[u16; 3]; 3]; 4],
	pub skip: [[u16; 2]; 3],
	pub intra: [[u16; 2]; 4],
	pub r#ref: [[[u16; 2]; 3]; 6],
	pub newmv_mode: [[u16; 2]; 6],
	pub globalmv_mode: [[u16; 2]; 2],
	pub refmv_mode: [[u16; 2]; 6],
	pub drl_bit: [[u16; 2]; 3],
	pub mv_joint: [u16; 4],
	pub mv: [MvComponentCdf; 2],
}

impl CdfContext {
	/// Defaults for a frame whose base quantizer index is `qidx`.
	pub fn new(qidx: u8) -> Self {
		let q = usize::from(qidx > 20) + usize::from(qidx > 60) + usize::from(qidx > 120);
		let comp = MvComponentCdf {
			classes: t::MV_CLASSES,
			class_0_fp: t::MV_CLASS_0_FP,
			class_n_fp: t::MV_CLASS_N_FP,
			class_0_hp: t::MV_CLASS_0_HP,
			class_n_hp: t::MV_CLASS_N_HP,
			class_0: t::MV_CLASS_0,
			class_n: t::MV_CLASS_N,
			sign: t::MV_SIGN,
		};
		Self {
			eob_bin_16: t::COEF_EOB_BIN_16[q],
			eob_bin_64: t::COEF_EOB_BIN_64[q],
			eob_bin_256: t::COEF_EOB_BIN_256[q],
			eob_bin_1024: t::COEF_EOB_BIN_1024[q],
			eob_base_tok: t::COEF_EOB_BASE_TOK[q],
			base_tok: t::COEF_BASE_TOK[q],
			br_tok: t::COEF_BR_TOK[q],
			eob_hi_bit: t::COEF_EOB_HI_BIT[q],
			coef_skip: t::COEF_SKIP[q],
			dc_sign: t::COEF_DC_SIGN[q],
			kf_y_mode: t::KF_Y_MODE,
			y_mode: t::MI_Y_MODE,
			uv_mode_cfl: t::UV_MODE_CFL,
			uv_mode_no_cfl: t::UV_MODE_NO_CFL,
			angle_delta: t::M_ANGLE_DELTA,
			partition_64: t::PARTITION_64,
			partition_32: t::PARTITION_32,
			partition_16: t::PARTITION_16,
			partition_8: t::PARTITION_8,
			txtp_intra2: t::M_TXTP_INTRA2,
			txtp_inter3: t::M_TXTP_INTER3,
			txsz: t::M_TXSZ,
			skip: t::M_SKIP,
			intra: t::MI_INTRA,
			r#ref: t::MI_REF,
			newmv_mode: t::MI_NEWMV_MODE,
			globalmv_mode: t::MI_GLOBALMV_MODE,
			refmv_mode: t::MI_REFMV_MODE,
			drl_bit: t::MI_DRL_BIT,
			mv_joint: t::MV_JOINT,
			mv: [comp.clone(), comp],
		}
	}

	/// The partition CDF of a block level (1 = 64x64 ... 4 = 8x8) and
	/// context, and its symbol count minus one.
	pub fn partition(&mut self, bl: usize, ctx: usize) -> (&mut [u16], usize) {
		match bl {
			1 => (&mut self.partition_64[ctx][..], 9),
			2 => (&mut self.partition_32[ctx][..], 9),
			3 => (&mut self.partition_16[ctx][..], 9),
			_ => (&mut self.partition_8[ctx][..], 3),
		}
	}
}
