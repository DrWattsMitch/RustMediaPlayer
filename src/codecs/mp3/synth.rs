//! Polyphase synthesis filterbank (ISO/IEC 11172-3 §2.4.3.2 / Annex A).
//!
//! Turns 32 subband samples per time slot back into PCM: a 32-point DCT-II
//! followed by the 512-tap windowed overlap. The arithmetic is minimp3's
//! fast factorisation (public domain), written with explicit index math over
//! fixed-size buffers.

#![allow(clippy::excessive_precision)]

use super::layer3::GRANULE_SIZE;
use super::tables::{DCT_SEC, SYNTH_WIN};

/// History the synthesis window keeps between granules (per stream).
pub const QMF_STATE_LEN: usize = 15 * 64;

/// Scratch space for one granule's synthesis: history + 18 new time slots.
pub const SYNTH_SCRATCH_LEN: usize = (18 + 15) * 64;

/// In-place 32-point DCT-II across subbands, for the first `n` time slots.
/// `grbuf` is laid out as `[subband][18 time slots]`.
fn dct_ii(grbuf: &mut [f32], n: usize) {
	for k in 0..n {
		let mut t = [[0.0f32; 8]; 4];
		for i in 0..8 {
			let x0 = grbuf[k + i * 18];
			let x1 = grbuf[k + (15 - i) * 18];
			let x2 = grbuf[k + (16 + i) * 18];
			let x3 = grbuf[k + (31 - i) * 18];
			let t0 = x0 + x3;
			let t1 = x1 + x2;
			let t2 = (x1 - x2) * DCT_SEC[3 * i];
			let t3 = (x0 - x3) * DCT_SEC[3 * i + 1];
			t[0][i] = t0 + t1;
			t[1][i] = (t0 - t1) * DCT_SEC[3 * i + 2];
			t[2][i] = t3 + t2;
			t[3][i] = (t3 - t2) * DCT_SEC[3 * i + 2];
		}
		for x in t.iter_mut() {
			let [mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7] = *x;
			let mut xt = x0 - x7;
			x0 += x7;
			x7 = x1 - x6;
			x1 += x6;
			x6 = x2 - x5;
			x2 += x5;
			x5 = x3 - x4;
			x3 += x4;
			x4 = x0 - x3;
			x0 += x3;
			x3 = x1 - x2;
			x1 += x2;
			x[0] = x0 + x1;
			x[4] = (x0 - x1) * 0.70710677;
			x5 += x6;
			x6 = (x6 + x7) * 0.70710677;
			x7 += xt;
			x3 = (x3 + x4) * 0.70710677;
			x5 -= x7 * 0.198912367; // rotate by PI/8
			x7 += x5 * 0.382683432;
			x5 -= x7 * 0.198912367;
			x0 = xt - x6;
			xt += x6;
			x[1] = (xt + x7) * 0.50979561;
			x[2] = (x4 + x3) * 0.54119611;
			x[3] = (x0 - x5) * 0.60134488;
			x[5] = (x0 + x5) * 0.89997619;
			x[6] = (x4 - x3) * 1.30656302;
			x[7] = (xt - x7) * 2.56291556;
		}
		let mut y = k;
		for i in 0..7 {
			grbuf[y] = t[0][i];
			grbuf[y + 18] = t[2][i] + t[3][i] + t[3][i + 1];
			grbuf[y + 2 * 18] = t[1][i] + t[1][i + 1];
			grbuf[y + 3 * 18] = t[2][i + 1] + t[3][i] + t[3][i + 1];
			y += 4 * 18;
		}
		grbuf[y] = t[0][7];
		grbuf[y + 18] = t[2][7] + t[3][7];
		grbuf[y + 2 * 18] = t[1][7];
		grbuf[y + 3 * 18] = t[3][7];
	}
}

/// Rounds to the nearest integer and saturates to the 16-bit range.
#[inline]
fn scale_pcm(sample: f32) -> i16 {
	if sample >= 32766.5 {
		return i16::MAX;
	}
	if sample <= -32767.5 {
		return i16::MIN;
	}
	sample.round() as i16
}

/// Produces the two "edge" outputs of a time-slot pair (rows 0 and 16).
fn synth_pair(pcm: &mut [i16], at: usize, nch: usize, lins: &[f32], z: usize) {
	let l = |k: usize| lins[z + k * 64];
	let mut a = (l(14) - l(0)) * 29.0;
	a += (l(1) + l(13)) * 213.0;
	a += (l(12) - l(2)) * 459.0;
	a += (l(3) + l(11)) * 2037.0;
	a += (l(10) - l(4)) * 5153.0;
	a += (l(5) + l(9)) * 6574.0;
	a += (l(8) - l(6)) * 37489.0;
	a += l(7) * 75038.0;
	pcm[at] = scale_pcm(a);

	let l = |k: usize| lins[z + 2 + k * 64];
	let mut a = l(14) * 104.0;
	a += l(12) * 1567.0;
	a += l(10) * 9727.0;
	a += l(8) * 64019.0;
	a += l(6) * -9975.0;
	a += l(4) * -45.0;
	a += l(2) * 146.0;
	a += l(0) * -5.0;
	pcm[at + 16 * nch] = scale_pcm(a);
}

/// Synthesises two consecutive time slots (64 output samples per channel).
///
/// * `x` — offset of the time slot in `grbuf` (left channel; right is +576)
/// * `out` — offset of the first output sample in interleaved `pcm`
/// * `base` — offset of this slot pair's window in `lins`
fn synth(
	grbuf: &[f32],
	x: usize,
	nch: usize,
	pcm: &mut [i16],
	out: usize,
	lins: &mut [f32],
	base: usize,
) {
	let xl = x;
	let xr = x + GRANULE_SIZE * (nch - 1);
	let dstl = out;
	let dstr = out + nch - 1;
	let zlin = base + 15 * 64;

	lins[zlin + 4 * 15] = grbuf[xl + 18 * 16];
	lins[zlin + 4 * 15 + 1] = grbuf[xr + 18 * 16];
	lins[zlin + 4 * 15 + 2] = grbuf[xl];
	lins[zlin + 4 * 15 + 3] = grbuf[xr];

	lins[zlin + 4 * 31] = grbuf[xl + 1 + 18 * 16];
	lins[zlin + 4 * 31 + 1] = grbuf[xr + 1 + 18 * 16];
	lins[zlin + 4 * 31 + 2] = grbuf[xl + 1];
	lins[zlin + 4 * 31 + 3] = grbuf[xr + 1];

	synth_pair(pcm, dstr, nch, lins, base + 4 * 15 + 1);
	synth_pair(pcm, dstr + 32 * nch, nch, lins, base + 4 * 15 + 64 + 1);
	synth_pair(pcm, dstl, nch, lins, base + 4 * 15);
	synth_pair(pcm, dstl + 32 * nch, nch, lins, base + 4 * 15 + 64);

	let mut w = 0usize;
	for i in (0..15).rev() {
		lins[zlin + 4 * i] = grbuf[xl + 18 * (31 - i)];
		lins[zlin + 4 * i + 1] = grbuf[xr + 18 * (31 - i)];
		lins[zlin + 4 * i + 2] = grbuf[xl + 1 + 18 * (31 - i)];
		lins[zlin + 4 * i + 3] = grbuf[xr + 1 + 18 * (31 - i)];
		lins[zlin + 4 * (i + 16)] = grbuf[xl + 1 + 18 * (1 + i)];
		lins[zlin + 4 * (i + 16) + 1] = grbuf[xr + 1 + 18 * (1 + i)];
		lins[zlin + 4 * i - 64 + 2] = grbuf[xl + 18 * (1 + i)];
		lins[zlin + 4 * i - 64 + 3] = grbuf[xr + 18 * (1 + i)];

		let mut a = [0.0f32; 4];
		let mut b = [0.0f32; 4];
		for k in 0..8 {
			let w0 = SYNTH_WIN[w];
			let w1 = SYNTH_WIN[w + 1];
			w += 2;
			// vz = zlin + 4i - 64k, vy = zlin + 4i - 64(15 - k)
			let vz = base + (15 - k) * 64 + 4 * i;
			let vy = base + k * 64 + 4 * i;
			for j in 0..4 {
				let z = lins[vz + j];
				let y = lins[vy + j];
				if k == 0 {
					b[j] = z * w1 + y * w0;
					a[j] = z * w0 - y * w1;
				} else if k % 2 == 0 {
					b[j] += z * w1 + y * w0;
					a[j] += z * w0 - y * w1;
				} else {
					b[j] += z * w1 + y * w0;
					a[j] += y * w1 - z * w0;
				}
			}
		}

		pcm[dstr + (15 - i) * nch] = scale_pcm(a[1]);
		pcm[dstr + (17 + i) * nch] = scale_pcm(b[1]);
		pcm[dstl + (15 - i) * nch] = scale_pcm(a[0]);
		pcm[dstl + (17 + i) * nch] = scale_pcm(b[0]);
		pcm[dstr + (47 - i) * nch] = scale_pcm(a[3]);
		pcm[dstr + (49 + i) * nch] = scale_pcm(b[3]);
		pcm[dstl + (47 - i) * nch] = scale_pcm(a[2]);
		pcm[dstl + (49 + i) * nch] = scale_pcm(b[2]);
	}
}

/// Runs the synthesis filterbank for one granule (18 time slots × 32
/// subbands per channel) and writes `576 * nch` interleaved samples to `pcm`.
///
/// `grbuf` holds both channels back to back (`2 * 576`), `qmf_state` is the
/// persistent window history and `lins` is scratch space.
pub fn synth_granule(
	qmf_state: &mut [f32; QMF_STATE_LEN],
	grbuf: &mut [f32; 2 * GRANULE_SIZE],
	nch: usize,
	pcm: &mut [i16],
	lins: &mut [f32; SYNTH_SCRATCH_LEN],
) {
	debug_assert!(nch == 1 || nch == 2);
	debug_assert!(pcm.len() >= GRANULE_SIZE * nch);
	for ch in 0..nch {
		dct_ii(&mut grbuf[ch * GRANULE_SIZE..(ch + 1) * GRANULE_SIZE], 18);
	}
	lins[..QMF_STATE_LEN].copy_from_slice(qmf_state);
	for i in (0..18).step_by(2) {
		synth(grbuf, i, nch, pcm, 32 * nch * i, lins, i * 64);
	}
	let history = &lins[18 * 64..18 * 64 + QMF_STATE_LEN];
	if nch == 1 {
		// Mono only uses the even (left) lanes; keep odd lanes untouched,
		// matching the reference decoder's output bit-for-bit.
		for i in (0..QMF_STATE_LEN).step_by(2) {
			qmf_state[i] = history[i];
		}
	} else {
		qmf_state.copy_from_slice(history);
	}
}
