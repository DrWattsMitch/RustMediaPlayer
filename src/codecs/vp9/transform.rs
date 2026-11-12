//! Inverse transforms (spec §8.7.1; libvpx `vpx_dsp/inv_txfm.c`,
//! `vp9/common/vp9_idct.c`).
//!
//! The 1-D DCT and ADST butterflies are a direct transcription of libvpx's
//! C reference (generated from it, then reviewed), computed in 64-bit with
//! libvpx's 32-bit wrapping (`WRAPLOW`) so every stream decodes bit-exactly.
//! The 2-D transforms run rows then columns and add the rounded result to
//! the prediction. Lossless frames use the 4x4 Walsh-Hadamard transform.
#![allow(clippy::too_many_lines, unused_assignments, unused_mut, clippy::identity_op)]

const COSPI_1_64: i64 = 16364;
const COSPI_2_64: i64 = 16305;
const COSPI_3_64: i64 = 16207;
const COSPI_4_64: i64 = 16069;
const COSPI_5_64: i64 = 15893;
const COSPI_6_64: i64 = 15679;
const COSPI_7_64: i64 = 15426;
const COSPI_8_64: i64 = 15137;
const COSPI_9_64: i64 = 14811;
const COSPI_10_64: i64 = 14449;
const COSPI_11_64: i64 = 14053;
const COSPI_12_64: i64 = 13623;
const COSPI_13_64: i64 = 13160;
const COSPI_14_64: i64 = 12665;
const COSPI_15_64: i64 = 12140;
const COSPI_16_64: i64 = 11585;
const COSPI_17_64: i64 = 11003;
const COSPI_18_64: i64 = 10394;
const COSPI_19_64: i64 = 9760;
const COSPI_20_64: i64 = 9102;
const COSPI_21_64: i64 = 8423;
const COSPI_22_64: i64 = 7723;
const COSPI_23_64: i64 = 7005;
const COSPI_24_64: i64 = 6270;
const COSPI_25_64: i64 = 5520;
const COSPI_26_64: i64 = 4756;
const COSPI_27_64: i64 = 3981;
const COSPI_28_64: i64 = 3196;
const COSPI_29_64: i64 = 2404;
const COSPI_30_64: i64 = 1606;
const COSPI_31_64: i64 = 804;
const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

/// `dct_const_round_shift`: rounds away 14 fractional bits.
#[inline]
fn rs(x: i64) -> i64 {
	(x + (1 << 13)) >> 14
}

/// `WRAPLOW`: libvpx keeps intermediate values in 32 bits.
#[inline]
fn wl(x: i64) -> i64 {
	i64::from(x as i32)
}

fn idct4(input: &[i32], output: &mut [i32]) {
	let mut step = [0i64; 4];
	let mut temp1: i64 = 0;
	let mut temp2: i64 = 0;
	temp1 = (i64::from(input[0]) + i64::from(input[2])) * COSPI_16_64;
	temp2 = (i64::from(input[0]) - i64::from(input[2])) * COSPI_16_64;
	step[0] = wl(rs(temp1));
	step[1] = wl(rs(temp2));
	temp1 = i64::from(input[1]) * COSPI_24_64 - i64::from(input[3]) * COSPI_8_64;
	temp2 = i64::from(input[1]) * COSPI_8_64 + i64::from(input[3]) * COSPI_24_64;
	step[2] = wl(rs(temp1));
	step[3] = wl(rs(temp2));
	output[0] = wl(step[0] + step[3]) as i32;
	output[1] = wl(step[1] + step[2]) as i32;
	output[2] = wl(step[1] - step[2]) as i32;
	output[3] = wl(step[0] - step[3]) as i32;
}

fn iadst4(input: &[i32], output: &mut [i32]) {
	let mut s0: i64 = 0;
	let mut s1: i64 = 0;
	let mut s2: i64 = 0;
	let mut s3: i64 = 0;
	let mut s4: i64 = 0;
	let mut s5: i64 = 0;
	let mut s6: i64 = 0;
	let mut s7: i64 = 0;
	let mut x0: i64 = i64::from(input[0]);
	let mut x1: i64 = i64::from(input[1]);
	let mut x2: i64 = i64::from(input[2]);
	let mut x3: i64 = i64::from(input[3]);
	if (x0 | x1 | x2 | x3) == 0 {
		output[..4].fill(0);
		return;
	}
	s0 = SINPI_1_9 * x0;
	s1 = SINPI_2_9 * x0;
	s2 = SINPI_3_9 * x1;
	s3 = SINPI_4_9 * x2;
	s4 = SINPI_1_9 * x2;
	s5 = SINPI_2_9 * x3;
	s6 = SINPI_4_9 * x3;
	s7 = wl(x0 - x2 + x3);
	s0 = s0 + s3 + s5;
	s1 = s1 - s4 - s6;
	s3 = s2;
	s2 = SINPI_3_9 * s7;
	output[0] = wl(rs(s0 + s3)) as i32;
	output[1] = wl(rs(s1 + s3)) as i32;
	output[2] = wl(rs(s2)) as i32;
	output[3] = wl(rs(s0 + s1 - s3)) as i32;
}

fn idct8(input: &[i32], output: &mut [i32]) {
	let mut step1 = [0i64; 8];
	let mut step2 = [0i64; 8];
	let mut temp1: i64 = 0;
	let mut temp2: i64 = 0;
	step1[0] = i64::from(input[0]);
	step1[2] = i64::from(input[4]);
	step1[1] = i64::from(input[2]);
	step1[3] = i64::from(input[6]);
	temp1 = i64::from(input[1]) * COSPI_28_64 - i64::from(input[7]) * COSPI_4_64;
	temp2 = i64::from(input[1]) * COSPI_4_64 + i64::from(input[7]) * COSPI_28_64;
	step1[4] = wl(rs(temp1));
	step1[7] = wl(rs(temp2));
	temp1 = i64::from(input[5]) * COSPI_12_64 - i64::from(input[3]) * COSPI_20_64;
	temp2 = i64::from(input[5]) * COSPI_20_64 + i64::from(input[3]) * COSPI_12_64;
	step1[5] = wl(rs(temp1));
	step1[6] = wl(rs(temp2));
	temp1 = (step1[0] + step1[2]) * COSPI_16_64;
	temp2 = (step1[0] - step1[2]) * COSPI_16_64;
	step2[0] = wl(rs(temp1));
	step2[1] = wl(rs(temp2));
	temp1 = step1[1] * COSPI_24_64 - step1[3] * COSPI_8_64;
	temp2 = step1[1] * COSPI_8_64 + step1[3] * COSPI_24_64;
	step2[2] = wl(rs(temp1));
	step2[3] = wl(rs(temp2));
	step2[4] = wl(step1[4] + step1[5]);
	step2[5] = wl(step1[4] - step1[5]);
	step2[6] = wl(-step1[6] + step1[7]);
	step2[7] = wl(step1[6] + step1[7]);
	step1[0] = wl(step2[0] + step2[3]);
	step1[1] = wl(step2[1] + step2[2]);
	step1[2] = wl(step2[1] - step2[2]);
	step1[3] = wl(step2[0] - step2[3]);
	step1[4] = step2[4];
	temp1 = (step2[6] - step2[5]) * COSPI_16_64;
	temp2 = (step2[5] + step2[6]) * COSPI_16_64;
	step1[5] = wl(rs(temp1));
	step1[6] = wl(rs(temp2));
	step1[7] = step2[7];
	output[0] = wl(step1[0] + step1[7]) as i32;
	output[1] = wl(step1[1] + step1[6]) as i32;
	output[2] = wl(step1[2] + step1[5]) as i32;
	output[3] = wl(step1[3] + step1[4]) as i32;
	output[4] = wl(step1[3] - step1[4]) as i32;
	output[5] = wl(step1[2] - step1[5]) as i32;
	output[6] = wl(step1[1] - step1[6]) as i32;
	output[7] = wl(step1[0] - step1[7]) as i32;
}

fn iadst8(input: &[i32], output: &mut [i32]) {
	let mut s0: i64 = 0;
	let mut s1: i64 = 0;
	let mut s2: i64 = 0;
	let mut s3: i64 = 0;
	let mut s4: i64 = 0;
	let mut s5: i64 = 0;
	let mut s6: i64 = 0;
	let mut s7: i64 = 0;
	let mut x0: i64 = i64::from(input[7]);
	let mut x1: i64 = i64::from(input[0]);
	let mut x2: i64 = i64::from(input[5]);
	let mut x3: i64 = i64::from(input[2]);
	let mut x4: i64 = i64::from(input[3]);
	let mut x5: i64 = i64::from(input[4]);
	let mut x6: i64 = i64::from(input[1]);
	let mut x7: i64 = i64::from(input[6]);
	if (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7) == 0 {
		output[..8].fill(0);
		return;
	}
	s0 = wl(COSPI_2_64 * x0 + COSPI_30_64 * x1);
	s1 = wl(COSPI_30_64 * x0 - COSPI_2_64 * x1);
	s2 = wl(COSPI_10_64 * x2 + COSPI_22_64 * x3);
	s3 = wl(COSPI_22_64 * x2 - COSPI_10_64 * x3);
	s4 = wl(COSPI_18_64 * x4 + COSPI_14_64 * x5);
	s5 = wl(COSPI_14_64 * x4 - COSPI_18_64 * x5);
	s6 = wl(COSPI_26_64 * x6 + COSPI_6_64 * x7);
	s7 = wl(COSPI_6_64 * x6 - COSPI_26_64 * x7);
	x0 = wl(rs(s0 + s4));
	x1 = wl(rs(s1 + s5));
	x2 = wl(rs(s2 + s6));
	x3 = wl(rs(s3 + s7));
	x4 = wl(rs(s0 - s4));
	x5 = wl(rs(s1 - s5));
	x6 = wl(rs(s2 - s6));
	x7 = wl(rs(s3 - s7));
	s0 = wl(x0);
	s1 = wl(x1);
	s2 = wl(x2);
	s3 = wl(x3);
	s4 = wl(COSPI_8_64 * x4 + COSPI_24_64 * x5);
	s5 = wl(COSPI_24_64 * x4 - COSPI_8_64 * x5);
	s6 = wl(-COSPI_24_64 * x6 + COSPI_8_64 * x7);
	s7 = wl(COSPI_8_64 * x6 + COSPI_24_64 * x7);
	x0 = wl(s0 + s2);
	x1 = wl(s1 + s3);
	x2 = wl(s0 - s2);
	x3 = wl(s1 - s3);
	x4 = wl(rs(s4 + s6));
	x5 = wl(rs(s5 + s7));
	x6 = wl(rs(s4 - s6));
	x7 = wl(rs(s5 - s7));
	s2 = wl(COSPI_16_64 * (x2 + x3));
	s3 = wl(COSPI_16_64 * (x2 - x3));
	s6 = wl(COSPI_16_64 * (x6 + x7));
	s7 = wl(COSPI_16_64 * (x6 - x7));
	x2 = wl(rs(s2));
	x3 = wl(rs(s3));
	x6 = wl(rs(s6));
	x7 = wl(rs(s7));
	output[0] = wl(x0) as i32;
	output[1] = wl(-x4) as i32;
	output[2] = wl(x6) as i32;
	output[3] = wl(-x2) as i32;
	output[4] = wl(x3) as i32;
	output[5] = wl(-x7) as i32;
	output[6] = wl(x5) as i32;
	output[7] = wl(-x1) as i32;
}

fn idct16(input: &[i32], output: &mut [i32]) {
	let mut step1 = [0i64; 16];
	let mut step2 = [0i64; 16];
	let mut temp1: i64 = 0;
	let mut temp2: i64 = 0;
	step1[0] = i64::from(input[0]);
	step1[1] = i64::from(input[8]);
	step1[2] = i64::from(input[4]);
	step1[3] = i64::from(input[12]);
	step1[4] = i64::from(input[2]);
	step1[5] = i64::from(input[10]);
	step1[6] = i64::from(input[6]);
	step1[7] = i64::from(input[14]);
	step1[8] = i64::from(input[1]);
	step1[9] = i64::from(input[9]);
	step1[10] = i64::from(input[5]);
	step1[11] = i64::from(input[13]);
	step1[12] = i64::from(input[3]);
	step1[13] = i64::from(input[11]);
	step1[14] = i64::from(input[7]);
	step1[15] = i64::from(input[15]);
	step2[0] = step1[0];
	step2[1] = step1[1];
	step2[2] = step1[2];
	step2[3] = step1[3];
	step2[4] = step1[4];
	step2[5] = step1[5];
	step2[6] = step1[6];
	step2[7] = step1[7];
	temp1 = step1[8] * COSPI_30_64 - step1[15] * COSPI_2_64;
	temp2 = step1[8] * COSPI_2_64 + step1[15] * COSPI_30_64;
	step2[8] = wl(rs(temp1));
	step2[15] = wl(rs(temp2));
	temp1 = step1[9] * COSPI_14_64 - step1[14] * COSPI_18_64;
	temp2 = step1[9] * COSPI_18_64 + step1[14] * COSPI_14_64;
	step2[9] = wl(rs(temp1));
	step2[14] = wl(rs(temp2));
	temp1 = step1[10] * COSPI_22_64 - step1[13] * COSPI_10_64;
	temp2 = step1[10] * COSPI_10_64 + step1[13] * COSPI_22_64;
	step2[10] = wl(rs(temp1));
	step2[13] = wl(rs(temp2));
	temp1 = step1[11] * COSPI_6_64 - step1[12] * COSPI_26_64;
	temp2 = step1[11] * COSPI_26_64 + step1[12] * COSPI_6_64;
	step2[11] = wl(rs(temp1));
	step2[12] = wl(rs(temp2));
	step1[0] = step2[0];
	step1[1] = step2[1];
	step1[2] = step2[2];
	step1[3] = step2[3];
	temp1 = step2[4] * COSPI_28_64 - step2[7] * COSPI_4_64;
	temp2 = step2[4] * COSPI_4_64 + step2[7] * COSPI_28_64;
	step1[4] = wl(rs(temp1));
	step1[7] = wl(rs(temp2));
	temp1 = step2[5] * COSPI_12_64 - step2[6] * COSPI_20_64;
	temp2 = step2[5] * COSPI_20_64 + step2[6] * COSPI_12_64;
	step1[5] = wl(rs(temp1));
	step1[6] = wl(rs(temp2));
	step1[8] = wl(step2[8] + step2[9]);
	step1[9] = wl(step2[8] - step2[9]);
	step1[10] = wl(-step2[10] + step2[11]);
	step1[11] = wl(step2[10] + step2[11]);
	step1[12] = wl(step2[12] + step2[13]);
	step1[13] = wl(step2[12] - step2[13]);
	step1[14] = wl(-step2[14] + step2[15]);
	step1[15] = wl(step2[14] + step2[15]);
	temp1 = (step1[0] + step1[1]) * COSPI_16_64;
	temp2 = (step1[0] - step1[1]) * COSPI_16_64;
	step2[0] = wl(rs(temp1));
	step2[1] = wl(rs(temp2));
	temp1 = step1[2] * COSPI_24_64 - step1[3] * COSPI_8_64;
	temp2 = step1[2] * COSPI_8_64 + step1[3] * COSPI_24_64;
	step2[2] = wl(rs(temp1));
	step2[3] = wl(rs(temp2));
	step2[4] = wl(step1[4] + step1[5]);
	step2[5] = wl(step1[4] - step1[5]);
	step2[6] = wl(-step1[6] + step1[7]);
	step2[7] = wl(step1[6] + step1[7]);
	step2[8] = step1[8];
	step2[15] = step1[15];
	temp1 = -step1[9] * COSPI_8_64 + step1[14] * COSPI_24_64;
	temp2 = step1[9] * COSPI_24_64 + step1[14] * COSPI_8_64;
	step2[9] = wl(rs(temp1));
	step2[14] = wl(rs(temp2));
	temp1 = -step1[10] * COSPI_24_64 - step1[13] * COSPI_8_64;
	temp2 = -step1[10] * COSPI_8_64 + step1[13] * COSPI_24_64;
	step2[10] = wl(rs(temp1));
	step2[13] = wl(rs(temp2));
	step2[11] = step1[11];
	step2[12] = step1[12];
	step1[0] = wl(step2[0] + step2[3]);
	step1[1] = wl(step2[1] + step2[2]);
	step1[2] = wl(step2[1] - step2[2]);
	step1[3] = wl(step2[0] - step2[3]);
	step1[4] = step2[4];
	temp1 = (step2[6] - step2[5]) * COSPI_16_64;
	temp2 = (step2[5] + step2[6]) * COSPI_16_64;
	step1[5] = wl(rs(temp1));
	step1[6] = wl(rs(temp2));
	step1[7] = step2[7];
	step1[8] = wl(step2[8] + step2[11]);
	step1[9] = wl(step2[9] + step2[10]);
	step1[10] = wl(step2[9] - step2[10]);
	step1[11] = wl(step2[8] - step2[11]);
	step1[12] = wl(-step2[12] + step2[15]);
	step1[13] = wl(-step2[13] + step2[14]);
	step1[14] = wl(step2[13] + step2[14]);
	step1[15] = wl(step2[12] + step2[15]);
	step2[0] = wl(step1[0] + step1[7]);
	step2[1] = wl(step1[1] + step1[6]);
	step2[2] = wl(step1[2] + step1[5]);
	step2[3] = wl(step1[3] + step1[4]);
	step2[4] = wl(step1[3] - step1[4]);
	step2[5] = wl(step1[2] - step1[5]);
	step2[6] = wl(step1[1] - step1[6]);
	step2[7] = wl(step1[0] - step1[7]);
	step2[8] = step1[8];
	step2[9] = step1[9];
	temp1 = (-step1[10] + step1[13]) * COSPI_16_64;
	temp2 = (step1[10] + step1[13]) * COSPI_16_64;
	step2[10] = wl(rs(temp1));
	step2[13] = wl(rs(temp2));
	temp1 = (-step1[11] + step1[12]) * COSPI_16_64;
	temp2 = (step1[11] + step1[12]) * COSPI_16_64;
	step2[11] = wl(rs(temp1));
	step2[12] = wl(rs(temp2));
	step2[14] = step1[14];
	step2[15] = step1[15];
	output[0] = wl(step2[0] + step2[15]) as i32;
	output[1] = wl(step2[1] + step2[14]) as i32;
	output[2] = wl(step2[2] + step2[13]) as i32;
	output[3] = wl(step2[3] + step2[12]) as i32;
	output[4] = wl(step2[4] + step2[11]) as i32;
	output[5] = wl(step2[5] + step2[10]) as i32;
	output[6] = wl(step2[6] + step2[9]) as i32;
	output[7] = wl(step2[7] + step2[8]) as i32;
	output[8] = wl(step2[7] - step2[8]) as i32;
	output[9] = wl(step2[6] - step2[9]) as i32;
	output[10] = wl(step2[5] - step2[10]) as i32;
	output[11] = wl(step2[4] - step2[11]) as i32;
	output[12] = wl(step2[3] - step2[12]) as i32;
	output[13] = wl(step2[2] - step2[13]) as i32;
	output[14] = wl(step2[1] - step2[14]) as i32;
	output[15] = wl(step2[0] - step2[15]) as i32;
}

fn iadst16(input: &[i32], output: &mut [i32]) {
	let mut s0: i64 = 0;
	let mut s1: i64 = 0;
	let mut s2: i64 = 0;
	let mut s3: i64 = 0;
	let mut s4: i64 = 0;
	let mut s5: i64 = 0;
	let mut s6: i64 = 0;
	let mut s7: i64 = 0;
	let mut s8: i64 = 0;
	let mut s9: i64 = 0;
	let mut s10: i64 = 0;
	let mut s11: i64 = 0;
	let mut s12: i64 = 0;
	let mut s13: i64 = 0;
	let mut s14: i64 = 0;
	let mut s15: i64 = 0;
	let mut x0: i64 = i64::from(input[15]);
	let mut x1: i64 = i64::from(input[0]);
	let mut x2: i64 = i64::from(input[13]);
	let mut x3: i64 = i64::from(input[2]);
	let mut x4: i64 = i64::from(input[11]);
	let mut x5: i64 = i64::from(input[4]);
	let mut x6: i64 = i64::from(input[9]);
	let mut x7: i64 = i64::from(input[6]);
	let mut x8: i64 = i64::from(input[7]);
	let mut x9: i64 = i64::from(input[8]);
	let mut x10: i64 = i64::from(input[5]);
	let mut x11: i64 = i64::from(input[10]);
	let mut x12: i64 = i64::from(input[3]);
	let mut x13: i64 = i64::from(input[12]);
	let mut x14: i64 = i64::from(input[1]);
	let mut x15: i64 = i64::from(input[14]);
	if (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15) == 0 {
		output[..16].fill(0);
		return;
	}
	s0 = x0 * COSPI_1_64 + x1 * COSPI_31_64;
	s1 = x0 * COSPI_31_64 - x1 * COSPI_1_64;
	s2 = x2 * COSPI_5_64 + x3 * COSPI_27_64;
	s3 = x2 * COSPI_27_64 - x3 * COSPI_5_64;
	s4 = x4 * COSPI_9_64 + x5 * COSPI_23_64;
	s5 = x4 * COSPI_23_64 - x5 * COSPI_9_64;
	s6 = x6 * COSPI_13_64 + x7 * COSPI_19_64;
	s7 = x6 * COSPI_19_64 - x7 * COSPI_13_64;
	s8 = x8 * COSPI_17_64 + x9 * COSPI_15_64;
	s9 = x8 * COSPI_15_64 - x9 * COSPI_17_64;
	s10 = x10 * COSPI_21_64 + x11 * COSPI_11_64;
	s11 = x10 * COSPI_11_64 - x11 * COSPI_21_64;
	s12 = x12 * COSPI_25_64 + x13 * COSPI_7_64;
	s13 = x12 * COSPI_7_64 - x13 * COSPI_25_64;
	s14 = x14 * COSPI_29_64 + x15 * COSPI_3_64;
	s15 = x14 * COSPI_3_64 - x15 * COSPI_29_64;
	x0 = wl(rs(s0 + s8));
	x1 = wl(rs(s1 + s9));
	x2 = wl(rs(s2 + s10));
	x3 = wl(rs(s3 + s11));
	x4 = wl(rs(s4 + s12));
	x5 = wl(rs(s5 + s13));
	x6 = wl(rs(s6 + s14));
	x7 = wl(rs(s7 + s15));
	x8 = wl(rs(s0 - s8));
	x9 = wl(rs(s1 - s9));
	x10 = wl(rs(s2 - s10));
	x11 = wl(rs(s3 - s11));
	x12 = wl(rs(s4 - s12));
	x13 = wl(rs(s5 - s13));
	x14 = wl(rs(s6 - s14));
	x15 = wl(rs(s7 - s15));
	s0 = x0;
	s1 = x1;
	s2 = x2;
	s3 = x3;
	s4 = x4;
	s5 = x5;
	s6 = x6;
	s7 = x7;
	s8 = x8 * COSPI_4_64 + x9 * COSPI_28_64;
	s9 = x8 * COSPI_28_64 - x9 * COSPI_4_64;
	s10 = x10 * COSPI_20_64 + x11 * COSPI_12_64;
	s11 = x10 * COSPI_12_64 - x11 * COSPI_20_64;
	s12 = -x12 * COSPI_28_64 + x13 * COSPI_4_64;
	s13 = x12 * COSPI_4_64 + x13 * COSPI_28_64;
	s14 = -x14 * COSPI_12_64 + x15 * COSPI_20_64;
	s15 = x14 * COSPI_20_64 + x15 * COSPI_12_64;
	x0 = wl(s0 + s4);
	x1 = wl(s1 + s5);
	x2 = wl(s2 + s6);
	x3 = wl(s3 + s7);
	x4 = wl(s0 - s4);
	x5 = wl(s1 - s5);
	x6 = wl(s2 - s6);
	x7 = wl(s3 - s7);
	x8 = wl(rs(s8 + s12));
	x9 = wl(rs(s9 + s13));
	x10 = wl(rs(s10 + s14));
	x11 = wl(rs(s11 + s15));
	x12 = wl(rs(s8 - s12));
	x13 = wl(rs(s9 - s13));
	x14 = wl(rs(s10 - s14));
	x15 = wl(rs(s11 - s15));
	s0 = x0;
	s1 = x1;
	s2 = x2;
	s3 = x3;
	s4 = x4 * COSPI_8_64 + x5 * COSPI_24_64;
	s5 = x4 * COSPI_24_64 - x5 * COSPI_8_64;
	s6 = -x6 * COSPI_24_64 + x7 * COSPI_8_64;
	s7 = x6 * COSPI_8_64 + x7 * COSPI_24_64;
	s8 = x8;
	s9 = x9;
	s10 = x10;
	s11 = x11;
	s12 = x12 * COSPI_8_64 + x13 * COSPI_24_64;
	s13 = x12 * COSPI_24_64 - x13 * COSPI_8_64;
	s14 = -x14 * COSPI_24_64 + x15 * COSPI_8_64;
	s15 = x14 * COSPI_8_64 + x15 * COSPI_24_64;
	x0 = wl(s0 + s2);
	x1 = wl(s1 + s3);
	x2 = wl(s0 - s2);
	x3 = wl(s1 - s3);
	x4 = wl(rs(s4 + s6));
	x5 = wl(rs(s5 + s7));
	x6 = wl(rs(s4 - s6));
	x7 = wl(rs(s5 - s7));
	x8 = wl(s8 + s10);
	x9 = wl(s9 + s11);
	x10 = wl(s8 - s10);
	x11 = wl(s9 - s11);
	x12 = wl(rs(s12 + s14));
	x13 = wl(rs(s13 + s15));
	x14 = wl(rs(s12 - s14));
	x15 = wl(rs(s13 - s15));
	s2 = (-COSPI_16_64) * (x2 + x3);
	s3 = COSPI_16_64 * (x2 - x3);
	s6 = COSPI_16_64 * (x6 + x7);
	s7 = COSPI_16_64 * (-x6 + x7);
	s10 = COSPI_16_64 * (x10 + x11);
	s11 = COSPI_16_64 * (-x10 + x11);
	s14 = (-COSPI_16_64) * (x14 + x15);
	s15 = COSPI_16_64 * (x14 - x15);
	x2 = wl(rs(s2));
	x3 = wl(rs(s3));
	x6 = wl(rs(s6));
	x7 = wl(rs(s7));
	x10 = wl(rs(s10));
	x11 = wl(rs(s11));
	x14 = wl(rs(s14));
	x15 = wl(rs(s15));
	output[0] = wl(x0) as i32;
	output[1] = wl(-x8) as i32;
	output[2] = wl(x12) as i32;
	output[3] = wl(-x4) as i32;
	output[4] = wl(x6) as i32;
	output[5] = wl(x14) as i32;
	output[6] = wl(x10) as i32;
	output[7] = wl(x2) as i32;
	output[8] = wl(x3) as i32;
	output[9] = wl(x11) as i32;
	output[10] = wl(x15) as i32;
	output[11] = wl(x7) as i32;
	output[12] = wl(x5) as i32;
	output[13] = wl(-x13) as i32;
	output[14] = wl(x9) as i32;
	output[15] = wl(-x1) as i32;
}

fn idct32(input: &[i32], output: &mut [i32]) {
	let mut step1 = [0i64; 32];
	let mut step2 = [0i64; 32];
	let mut temp1: i64 = 0;
	let mut temp2: i64 = 0;
	step1[0] = i64::from(input[0]);
	step1[1] = i64::from(input[16]);
	step1[2] = i64::from(input[8]);
	step1[3] = i64::from(input[24]);
	step1[4] = i64::from(input[4]);
	step1[5] = i64::from(input[20]);
	step1[6] = i64::from(input[12]);
	step1[7] = i64::from(input[28]);
	step1[8] = i64::from(input[2]);
	step1[9] = i64::from(input[18]);
	step1[10] = i64::from(input[10]);
	step1[11] = i64::from(input[26]);
	step1[12] = i64::from(input[6]);
	step1[13] = i64::from(input[22]);
	step1[14] = i64::from(input[14]);
	step1[15] = i64::from(input[30]);
	temp1 = i64::from(input[1]) * COSPI_31_64 - i64::from(input[31]) * COSPI_1_64;
	temp2 = i64::from(input[1]) * COSPI_1_64 + i64::from(input[31]) * COSPI_31_64;
	step1[16] = wl(rs(temp1));
	step1[31] = wl(rs(temp2));
	temp1 = i64::from(input[17]) * COSPI_15_64 - i64::from(input[15]) * COSPI_17_64;
	temp2 = i64::from(input[17]) * COSPI_17_64 + i64::from(input[15]) * COSPI_15_64;
	step1[17] = wl(rs(temp1));
	step1[30] = wl(rs(temp2));
	temp1 = i64::from(input[9]) * COSPI_23_64 - i64::from(input[23]) * COSPI_9_64;
	temp2 = i64::from(input[9]) * COSPI_9_64 + i64::from(input[23]) * COSPI_23_64;
	step1[18] = wl(rs(temp1));
	step1[29] = wl(rs(temp2));
	temp1 = i64::from(input[25]) * COSPI_7_64 - i64::from(input[7]) * COSPI_25_64;
	temp2 = i64::from(input[25]) * COSPI_25_64 + i64::from(input[7]) * COSPI_7_64;
	step1[19] = wl(rs(temp1));
	step1[28] = wl(rs(temp2));
	temp1 = i64::from(input[5]) * COSPI_27_64 - i64::from(input[27]) * COSPI_5_64;
	temp2 = i64::from(input[5]) * COSPI_5_64 + i64::from(input[27]) * COSPI_27_64;
	step1[20] = wl(rs(temp1));
	step1[27] = wl(rs(temp2));
	temp1 = i64::from(input[21]) * COSPI_11_64 - i64::from(input[11]) * COSPI_21_64;
	temp2 = i64::from(input[21]) * COSPI_21_64 + i64::from(input[11]) * COSPI_11_64;
	step1[21] = wl(rs(temp1));
	step1[26] = wl(rs(temp2));
	temp1 = i64::from(input[13]) * COSPI_19_64 - i64::from(input[19]) * COSPI_13_64;
	temp2 = i64::from(input[13]) * COSPI_13_64 + i64::from(input[19]) * COSPI_19_64;
	step1[22] = wl(rs(temp1));
	step1[25] = wl(rs(temp2));
	temp1 = i64::from(input[29]) * COSPI_3_64 - i64::from(input[3]) * COSPI_29_64;
	temp2 = i64::from(input[29]) * COSPI_29_64 + i64::from(input[3]) * COSPI_3_64;
	step1[23] = wl(rs(temp1));
	step1[24] = wl(rs(temp2));
	step2[0] = step1[0];
	step2[1] = step1[1];
	step2[2] = step1[2];
	step2[3] = step1[3];
	step2[4] = step1[4];
	step2[5] = step1[5];
	step2[6] = step1[6];
	step2[7] = step1[7];
	temp1 = step1[8] * COSPI_30_64 - step1[15] * COSPI_2_64;
	temp2 = step1[8] * COSPI_2_64 + step1[15] * COSPI_30_64;
	step2[8] = wl(rs(temp1));
	step2[15] = wl(rs(temp2));
	temp1 = step1[9] * COSPI_14_64 - step1[14] * COSPI_18_64;
	temp2 = step1[9] * COSPI_18_64 + step1[14] * COSPI_14_64;
	step2[9] = wl(rs(temp1));
	step2[14] = wl(rs(temp2));
	temp1 = step1[10] * COSPI_22_64 - step1[13] * COSPI_10_64;
	temp2 = step1[10] * COSPI_10_64 + step1[13] * COSPI_22_64;
	step2[10] = wl(rs(temp1));
	step2[13] = wl(rs(temp2));
	temp1 = step1[11] * COSPI_6_64 - step1[12] * COSPI_26_64;
	temp2 = step1[11] * COSPI_26_64 + step1[12] * COSPI_6_64;
	step2[11] = wl(rs(temp1));
	step2[12] = wl(rs(temp2));
	step2[16] = wl(step1[16] + step1[17]);
	step2[17] = wl(step1[16] - step1[17]);
	step2[18] = wl(-step1[18] + step1[19]);
	step2[19] = wl(step1[18] + step1[19]);
	step2[20] = wl(step1[20] + step1[21]);
	step2[21] = wl(step1[20] - step1[21]);
	step2[22] = wl(-step1[22] + step1[23]);
	step2[23] = wl(step1[22] + step1[23]);
	step2[24] = wl(step1[24] + step1[25]);
	step2[25] = wl(step1[24] - step1[25]);
	step2[26] = wl(-step1[26] + step1[27]);
	step2[27] = wl(step1[26] + step1[27]);
	step2[28] = wl(step1[28] + step1[29]);
	step2[29] = wl(step1[28] - step1[29]);
	step2[30] = wl(-step1[30] + step1[31]);
	step2[31] = wl(step1[30] + step1[31]);
	step1[0] = step2[0];
	step1[1] = step2[1];
	step1[2] = step2[2];
	step1[3] = step2[3];
	temp1 = step2[4] * COSPI_28_64 - step2[7] * COSPI_4_64;
	temp2 = step2[4] * COSPI_4_64 + step2[7] * COSPI_28_64;
	step1[4] = wl(rs(temp1));
	step1[7] = wl(rs(temp2));
	temp1 = step2[5] * COSPI_12_64 - step2[6] * COSPI_20_64;
	temp2 = step2[5] * COSPI_20_64 + step2[6] * COSPI_12_64;
	step1[5] = wl(rs(temp1));
	step1[6] = wl(rs(temp2));
	step1[8] = wl(step2[8] + step2[9]);
	step1[9] = wl(step2[8] - step2[9]);
	step1[10] = wl(-step2[10] + step2[11]);
	step1[11] = wl(step2[10] + step2[11]);
	step1[12] = wl(step2[12] + step2[13]);
	step1[13] = wl(step2[12] - step2[13]);
	step1[14] = wl(-step2[14] + step2[15]);
	step1[15] = wl(step2[14] + step2[15]);
	step1[16] = step2[16];
	step1[31] = step2[31];
	temp1 = -step2[17] * COSPI_4_64 + step2[30] * COSPI_28_64;
	temp2 = step2[17] * COSPI_28_64 + step2[30] * COSPI_4_64;
	step1[17] = wl(rs(temp1));
	step1[30] = wl(rs(temp2));
	temp1 = -step2[18] * COSPI_28_64 - step2[29] * COSPI_4_64;
	temp2 = -step2[18] * COSPI_4_64 + step2[29] * COSPI_28_64;
	step1[18] = wl(rs(temp1));
	step1[29] = wl(rs(temp2));
	step1[19] = step2[19];
	step1[20] = step2[20];
	temp1 = -step2[21] * COSPI_20_64 + step2[26] * COSPI_12_64;
	temp2 = step2[21] * COSPI_12_64 + step2[26] * COSPI_20_64;
	step1[21] = wl(rs(temp1));
	step1[26] = wl(rs(temp2));
	temp1 = -step2[22] * COSPI_12_64 - step2[25] * COSPI_20_64;
	temp2 = -step2[22] * COSPI_20_64 + step2[25] * COSPI_12_64;
	step1[22] = wl(rs(temp1));
	step1[25] = wl(rs(temp2));
	step1[23] = step2[23];
	step1[24] = step2[24];
	step1[27] = step2[27];
	step1[28] = step2[28];
	temp1 = (step1[0] + step1[1]) * COSPI_16_64;
	temp2 = (step1[0] - step1[1]) * COSPI_16_64;
	step2[0] = wl(rs(temp1));
	step2[1] = wl(rs(temp2));
	temp1 = step1[2] * COSPI_24_64 - step1[3] * COSPI_8_64;
	temp2 = step1[2] * COSPI_8_64 + step1[3] * COSPI_24_64;
	step2[2] = wl(rs(temp1));
	step2[3] = wl(rs(temp2));
	step2[4] = wl(step1[4] + step1[5]);
	step2[5] = wl(step1[4] - step1[5]);
	step2[6] = wl(-step1[6] + step1[7]);
	step2[7] = wl(step1[6] + step1[7]);
	step2[8] = step1[8];
	step2[15] = step1[15];
	temp1 = -step1[9] * COSPI_8_64 + step1[14] * COSPI_24_64;
	temp2 = step1[9] * COSPI_24_64 + step1[14] * COSPI_8_64;
	step2[9] = wl(rs(temp1));
	step2[14] = wl(rs(temp2));
	temp1 = -step1[10] * COSPI_24_64 - step1[13] * COSPI_8_64;
	temp2 = -step1[10] * COSPI_8_64 + step1[13] * COSPI_24_64;
	step2[10] = wl(rs(temp1));
	step2[13] = wl(rs(temp2));
	step2[11] = step1[11];
	step2[12] = step1[12];
	step2[16] = wl(step1[16] + step1[19]);
	step2[17] = wl(step1[17] + step1[18]);
	step2[18] = wl(step1[17] - step1[18]);
	step2[19] = wl(step1[16] - step1[19]);
	step2[20] = wl(-step1[20] + step1[23]);
	step2[21] = wl(-step1[21] + step1[22]);
	step2[22] = wl(step1[21] + step1[22]);
	step2[23] = wl(step1[20] + step1[23]);
	step2[24] = wl(step1[24] + step1[27]);
	step2[25] = wl(step1[25] + step1[26]);
	step2[26] = wl(step1[25] - step1[26]);
	step2[27] = wl(step1[24] - step1[27]);
	step2[28] = wl(-step1[28] + step1[31]);
	step2[29] = wl(-step1[29] + step1[30]);
	step2[30] = wl(step1[29] + step1[30]);
	step2[31] = wl(step1[28] + step1[31]);
	step1[0] = wl(step2[0] + step2[3]);
	step1[1] = wl(step2[1] + step2[2]);
	step1[2] = wl(step2[1] - step2[2]);
	step1[3] = wl(step2[0] - step2[3]);
	step1[4] = step2[4];
	temp1 = (step2[6] - step2[5]) * COSPI_16_64;
	temp2 = (step2[5] + step2[6]) * COSPI_16_64;
	step1[5] = wl(rs(temp1));
	step1[6] = wl(rs(temp2));
	step1[7] = step2[7];
	step1[8] = wl(step2[8] + step2[11]);
	step1[9] = wl(step2[9] + step2[10]);
	step1[10] = wl(step2[9] - step2[10]);
	step1[11] = wl(step2[8] - step2[11]);
	step1[12] = wl(-step2[12] + step2[15]);
	step1[13] = wl(-step2[13] + step2[14]);
	step1[14] = wl(step2[13] + step2[14]);
	step1[15] = wl(step2[12] + step2[15]);
	step1[16] = step2[16];
	step1[17] = step2[17];
	temp1 = -step2[18] * COSPI_8_64 + step2[29] * COSPI_24_64;
	temp2 = step2[18] * COSPI_24_64 + step2[29] * COSPI_8_64;
	step1[18] = wl(rs(temp1));
	step1[29] = wl(rs(temp2));
	temp1 = -step2[19] * COSPI_8_64 + step2[28] * COSPI_24_64;
	temp2 = step2[19] * COSPI_24_64 + step2[28] * COSPI_8_64;
	step1[19] = wl(rs(temp1));
	step1[28] = wl(rs(temp2));
	temp1 = -step2[20] * COSPI_24_64 - step2[27] * COSPI_8_64;
	temp2 = -step2[20] * COSPI_8_64 + step2[27] * COSPI_24_64;
	step1[20] = wl(rs(temp1));
	step1[27] = wl(rs(temp2));
	temp1 = -step2[21] * COSPI_24_64 - step2[26] * COSPI_8_64;
	temp2 = -step2[21] * COSPI_8_64 + step2[26] * COSPI_24_64;
	step1[21] = wl(rs(temp1));
	step1[26] = wl(rs(temp2));
	step1[22] = step2[22];
	step1[23] = step2[23];
	step1[24] = step2[24];
	step1[25] = step2[25];
	step1[30] = step2[30];
	step1[31] = step2[31];
	step2[0] = wl(step1[0] + step1[7]);
	step2[1] = wl(step1[1] + step1[6]);
	step2[2] = wl(step1[2] + step1[5]);
	step2[3] = wl(step1[3] + step1[4]);
	step2[4] = wl(step1[3] - step1[4]);
	step2[5] = wl(step1[2] - step1[5]);
	step2[6] = wl(step1[1] - step1[6]);
	step2[7] = wl(step1[0] - step1[7]);
	step2[8] = step1[8];
	step2[9] = step1[9];
	temp1 = (-step1[10] + step1[13]) * COSPI_16_64;
	temp2 = (step1[10] + step1[13]) * COSPI_16_64;
	step2[10] = wl(rs(temp1));
	step2[13] = wl(rs(temp2));
	temp1 = (-step1[11] + step1[12]) * COSPI_16_64;
	temp2 = (step1[11] + step1[12]) * COSPI_16_64;
	step2[11] = wl(rs(temp1));
	step2[12] = wl(rs(temp2));
	step2[14] = step1[14];
	step2[15] = step1[15];
	step2[16] = wl(step1[16] + step1[23]);
	step2[17] = wl(step1[17] + step1[22]);
	step2[18] = wl(step1[18] + step1[21]);
	step2[19] = wl(step1[19] + step1[20]);
	step2[20] = wl(step1[19] - step1[20]);
	step2[21] = wl(step1[18] - step1[21]);
	step2[22] = wl(step1[17] - step1[22]);
	step2[23] = wl(step1[16] - step1[23]);
	step2[24] = wl(-step1[24] + step1[31]);
	step2[25] = wl(-step1[25] + step1[30]);
	step2[26] = wl(-step1[26] + step1[29]);
	step2[27] = wl(-step1[27] + step1[28]);
	step2[28] = wl(step1[27] + step1[28]);
	step2[29] = wl(step1[26] + step1[29]);
	step2[30] = wl(step1[25] + step1[30]);
	step2[31] = wl(step1[24] + step1[31]);
	step1[0] = wl(step2[0] + step2[15]);
	step1[1] = wl(step2[1] + step2[14]);
	step1[2] = wl(step2[2] + step2[13]);
	step1[3] = wl(step2[3] + step2[12]);
	step1[4] = wl(step2[4] + step2[11]);
	step1[5] = wl(step2[5] + step2[10]);
	step1[6] = wl(step2[6] + step2[9]);
	step1[7] = wl(step2[7] + step2[8]);
	step1[8] = wl(step2[7] - step2[8]);
	step1[9] = wl(step2[6] - step2[9]);
	step1[10] = wl(step2[5] - step2[10]);
	step1[11] = wl(step2[4] - step2[11]);
	step1[12] = wl(step2[3] - step2[12]);
	step1[13] = wl(step2[2] - step2[13]);
	step1[14] = wl(step2[1] - step2[14]);
	step1[15] = wl(step2[0] - step2[15]);
	step1[16] = step2[16];
	step1[17] = step2[17];
	step1[18] = step2[18];
	step1[19] = step2[19];
	temp1 = (-step2[20] + step2[27]) * COSPI_16_64;
	temp2 = (step2[20] + step2[27]) * COSPI_16_64;
	step1[20] = wl(rs(temp1));
	step1[27] = wl(rs(temp2));
	temp1 = (-step2[21] + step2[26]) * COSPI_16_64;
	temp2 = (step2[21] + step2[26]) * COSPI_16_64;
	step1[21] = wl(rs(temp1));
	step1[26] = wl(rs(temp2));
	temp1 = (-step2[22] + step2[25]) * COSPI_16_64;
	temp2 = (step2[22] + step2[25]) * COSPI_16_64;
	step1[22] = wl(rs(temp1));
	step1[25] = wl(rs(temp2));
	temp1 = (-step2[23] + step2[24]) * COSPI_16_64;
	temp2 = (step2[23] + step2[24]) * COSPI_16_64;
	step1[23] = wl(rs(temp1));
	step1[24] = wl(rs(temp2));
	step1[28] = step2[28];
	step1[29] = step2[29];
	step1[30] = step2[30];
	step1[31] = step2[31];
	output[0] = wl(step1[0] + step1[31]) as i32;
	output[1] = wl(step1[1] + step1[30]) as i32;
	output[2] = wl(step1[2] + step1[29]) as i32;
	output[3] = wl(step1[3] + step1[28]) as i32;
	output[4] = wl(step1[4] + step1[27]) as i32;
	output[5] = wl(step1[5] + step1[26]) as i32;
	output[6] = wl(step1[6] + step1[25]) as i32;
	output[7] = wl(step1[7] + step1[24]) as i32;
	output[8] = wl(step1[8] + step1[23]) as i32;
	output[9] = wl(step1[9] + step1[22]) as i32;
	output[10] = wl(step1[10] + step1[21]) as i32;
	output[11] = wl(step1[11] + step1[20]) as i32;
	output[12] = wl(step1[12] + step1[19]) as i32;
	output[13] = wl(step1[13] + step1[18]) as i32;
	output[14] = wl(step1[14] + step1[17]) as i32;
	output[15] = wl(step1[15] + step1[16]) as i32;
	output[16] = wl(step1[15] - step1[16]) as i32;
	output[17] = wl(step1[14] - step1[17]) as i32;
	output[18] = wl(step1[13] - step1[18]) as i32;
	output[19] = wl(step1[12] - step1[19]) as i32;
	output[20] = wl(step1[11] - step1[20]) as i32;
	output[21] = wl(step1[10] - step1[21]) as i32;
	output[22] = wl(step1[9] - step1[22]) as i32;
	output[23] = wl(step1[8] - step1[23]) as i32;
	output[24] = wl(step1[7] - step1[24]) as i32;
	output[25] = wl(step1[6] - step1[25]) as i32;
	output[26] = wl(step1[5] - step1[26]) as i32;
	output[27] = wl(step1[4] - step1[27]) as i32;
	output[28] = wl(step1[3] - step1[28]) as i32;
	output[29] = wl(step1[2] - step1[29]) as i32;
	output[30] = wl(step1[1] - step1[30]) as i32;
	output[31] = wl(step1[0] - step1[31]) as i32;
}

type Transform1d = fn(&[i32], &mut [i32]);

#[inline]
fn add_clip(dst: &mut u8, v: i64) {
	*dst = (i64::from(*dst) + v).clamp(0, 255) as u8;
}

/// 4x4 inverse Walsh-Hadamard transform of lossless frames
/// (`vpx_iwht4x4_16_add`).
fn iwht4x4(input: &[i32], dst: &mut [u8], stride: usize) {
	let mut out = [0i64; 16];
	for i in 0..4 {
		let ip = &input[4 * i..4 * i + 4];
		let mut a1 = i64::from(ip[0] >> 2);
		let mut c1 = i64::from(ip[1] >> 2);
		let mut d1 = i64::from(ip[2] >> 2);
		let mut b1 = i64::from(ip[3] >> 2);
		a1 += c1;
		d1 -= b1;
		let e1 = (a1 - d1) >> 1;
		b1 = e1 - b1;
		c1 = e1 - c1;
		a1 -= b1;
		d1 += c1;
		out[4 * i..4 * i + 4].copy_from_slice(&[wl(a1), wl(b1), wl(c1), wl(d1)]);
	}
	for i in 0..4 {
		let mut a1 = out[i];
		let mut c1 = out[4 + i];
		let mut d1 = out[8 + i];
		let mut b1 = out[12 + i];
		a1 += c1;
		d1 -= b1;
		let e1 = (a1 - d1) >> 1;
		b1 = e1 - b1;
		c1 = e1 - c1;
		a1 -= b1;
		d1 += c1;
		for (j, v) in [a1, b1, c1, d1].into_iter().enumerate() {
			add_clip(&mut dst[j * stride + i], wl(v));
		}
	}
}

/// Adds `v` to every pixel of the `n` x `n` block, clipped.
fn add_constant(dst: &mut [u8], stride: usize, n: usize, v: i64) {
	if crate::simd::enabled() {
		super::simd::add_constant(dst, stride, n, v);
		return;
	}
	for r in 0..n {
		for p in &mut dst[r * stride..r * stride + n] {
			add_clip(p, v);
		}
	}
}

/// Adds the inverse transform of `coefs` (raster order) to `dst`.
/// `eob` is the number of coefficients decoded in scan order (only used to
/// spot DC-only blocks; the result is the same as the full transform).
/// `tx_type` selects ADST in the vertical (1) or horizontal (2) direction
/// or both (3); 32x32 is always a DCT.
pub fn inverse_transform_add(
	coefs: &[i32],
	dst: &mut [u8],
	stride: usize,
	tx_size: u8,
	tx_type: u8,
	lossless: bool,
	eob: usize,
) {
	if lossless {
		iwht4x4(coefs, dst, stride);
		return;
	}
	if eob == 1 && tx_type == 0 {
		// Only the DC coefficient (every scan starts there): the 2-D DCT
		// is a constant, libvpx's `vpx_idctNxN_1_add`.
		let n = 4usize << tx_size.min(3);
		let shift = if tx_size == 0 {
			4
		} else if tx_size == 1 {
			5
		} else {
			6
		};
		let a = wl(rs(i64::from(coefs[0]) * COSPI_16_64));
		let b = wl(rs(a * COSPI_16_64));
		let v = (b + (1 << (shift - 1))) >> shift;
		add_constant(dst, stride, n, v);
		return;
	}
	let (n, shift) = match tx_size {
		0 => (4usize, 4u32),
		1 => (8, 5),
		2 => (16, 6),
		_ => (32, 6),
	};
	let (dct, adst): (Transform1d, Transform1d) = match n {
		4 => (idct4, iadst4),
		8 => (idct8, iadst8),
		16 => (idct16, iadst16),
		_ => (idct32, idct32),
	};
	let cols = if tx_type & 1 != 0 { adst } else { dct };
	let rows = if tx_type & 2 != 0 { adst } else { dct };
	// Scratch sized to the transform: zeroing a 32x32 buffer for every 4x4
	// block was the decoder's single largest `memset`.
	match n {
		4 => transform_2d::<4, 16>(coefs, dst, stride, rows, cols, shift),
		8 => transform_2d::<8, 64>(coefs, dst, stride, rows, cols, shift),
		16 => transform_2d::<16, 256>(coefs, dst, stride, rows, cols, shift),
		_ => transform_2d::<32, 1024>(coefs, dst, stride, rows, cols, shift),
	}
}

/// Rows then columns of an `N` x `N` (`NN` = `N * N`) transform, added to
/// `dst` with rounding `shift`.
fn transform_2d<const N: usize, const NN: usize>(
	coefs: &[i32],
	dst: &mut [u8],
	stride: usize,
	rows: Transform1d,
	cols: Transform1d,
	shift: u32,
) {
	let n = N;
	let mut out = [0i32; NN];
	for i in 0..n {
		let row = &coefs[i * n..(i + 1) * n];
		// The transform of a zero row is zero.
		if row.iter().any(|&c| c != 0) {
			rows(row, &mut out[i * n..(i + 1) * n]);
		}
	}
	let mut temp_in = [0i32; N];
	let mut temp_out = [0i32; N];
	let round = 1i64 << (shift - 1);
	for i in 0..n {
		for j in 0..n {
			temp_in[j] = out[j * n + i];
		}
		cols(&temp_in, &mut temp_out);
		for j in 0..n {
			add_clip(&mut dst[j * stride + i], (i64::from(temp_out[j]) + round) >> shift);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn dc_only_matches_libvpx_shortcut() {
		// vpx_idct8x8_1_add: two rounded multiplications by cos(pi/4), then
		// a shift by 5.
		for dc in [-4000i32, -17, 1, 64, 1023, 9999] {
			let mut coefs = [0i32; 64];
			coefs[0] = dc;
			let mut dst = [128u8; 64];
			inverse_transform_add(&coefs, &mut dst, 8, 1, 0, false, 64);
			let out = rs(i64::from(dc) * COSPI_16_64);
			let out = rs(out * COSPI_16_64);
			let a1 = (out + 16) >> 5;
			let expect = (128 + a1).clamp(0, 255) as u8;
			assert!(dst.iter().all(|&p| p == expect), "dc {dc}");
		}
	}

	#[test]
	fn walsh_hadamard_dc() {
		let mut coefs = [0i32; 16];
		coefs[0] = 40;
		let mut dst = [100u8; 16];
		inverse_transform_add(&coefs, &mut dst, 4, 0, 0, true, 16);
		// vpx_iwht4x4_1_add: a1 = 10, e1 = 5 → row 0 [5, 5, 5, 5] ... first
		// column split again: [3, 2, 2, 2] for 5 and [3, 2, 2, 2] for e1 = 5.
		assert_eq!(dst[0], 103);
		assert_eq!(dst[1], 103);
		assert_eq!(dst[4], 102);
	}

	#[test]
	fn dc_only_shortcut_equals_the_full_transform() {
		let mut seed = 0x51u64;
		let mut next = || {
			seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			(seed >> 33) as i32
		};
		for tx_size in 0..4u8 {
			let n = 4usize << tx_size;
			for round in 0..3000 {
				let dc = match round % 3 {
					0 => next() % 64,
					1 => next() % 70000,
					_ => next() % (1 << 24),
				};
				let mut coefs = vec![0i32; n * n];
				coefs[0] = dc;
				let init: Vec<u8> = (0..n * n).map(|_| next() as u8).collect();
				let mut full = init.clone();
				let mut fast = init;
				// eob = n * n forces the full transform.
				inverse_transform_add(&coefs, &mut full, n, tx_size, 0, false, n * n);
				inverse_transform_add(&coefs, &mut fast, n, tx_size, 0, false, 1);
				assert_eq!(full, fast, "{n}x{n} dc {dc}");
			}
		}
	}
}
