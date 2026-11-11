//! Pitch and LPC analysis for CELT packet-loss concealment, ported from
//! libopus 1.4's float build (BSD-3-Clause): `celt/pitch.c`
//! (`pitch_downsample`, `pitch_search`, `find_best_pitch`, `celt_fir5`) and
//! `celt/celt_lpc.c` (`_celt_autocorr`, `_celt_lpc`, `celt_fir`, `celt_iir`).
//!
//! Sums run in libopus' C order (its SIMD kernels may add in another order,
//! so results agree up to float rounding).

#![allow(clippy::needless_range_loop)]

/// Order of the LPC filter CELT's PLC works through.
pub const LPC_ORDER: usize = 24;

fn inner_prod(x: &[f32], y: &[f32]) -> f32 {
	x.iter().zip(y).fold(0.0, |acc, (&a, &b)| acc + a * b)
}

/// `celt_pitch_xcorr`: `xcorr[i] = <x, y[i..]>` over `len` samples.
fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize) {
	for (i, v) in xcorr.iter_mut().enumerate() {
		*v = inner_prod(&x[..len], &y[i..i + len]);
	}
}

/// `_celt_autocorr`: lags `0..=lag` of `x`, optionally windowed at both
/// ends over `window.len()` samples.
pub fn autocorr(x: &[f32], window: &[f32], lag: usize) -> Vec<f32> {
	let n = x.len();
	let mut xx = x.to_vec();
	for (i, &w) in window.iter().enumerate() {
		xx[i] = x[i] * w;
		xx[n - i - 1] = x[n - i - 1] * w;
	}
	let fast_n = n - lag;
	let mut ac = vec![0f32; lag + 1];
	pitch_xcorr(&xx, &xx, &mut ac, fast_n);
	for k in 0..=lag {
		let mut d = 0f32;
		for i in k + fast_n..n {
			d += xx[i] * xx[i - k];
		}
		ac[k] += d;
	}
	ac
}

/// `_celt_lpc`: Levinson-Durbin, stopping at 30 dB of prediction gain.
pub fn lpc(ac: &[f32], p: usize) -> Vec<f32> {
	let mut lpc = vec![0f32; p];
	let mut error = ac[0];
	if ac[0] > 1e-10 {
		for i in 0..p {
			let mut rr = 0f32;
			for j in 0..i {
				rr += lpc[j] * ac[i - j];
			}
			rr += ac[i + 1];
			let r = -(rr / error);
			lpc[i] = r;
			for j in 0..(i + 1) >> 1 {
				let tmp1 = lpc[j];
				let tmp2 = lpc[i - 1 - j];
				lpc[j] = tmp1 + r * tmp2;
				lpc[i - 1 - j] = tmp2 + r * tmp1;
			}
			error -= r * r * error;
			if error <= 0.001 * ac[0] {
				break;
			}
		}
	}
	lpc
}

/// `celt_fir`: `y[i] = x[at + i] + sum_k num[k] * x[at + i - k - 1]`; `x`
/// must hold `num.len()` samples of history before `at`.
pub fn fir(x: &[f32], at: usize, num: &[f32], y: &mut [f32]) {
	let ord = num.len();
	for (i, out) in y.iter_mut().enumerate() {
		let p = at + i;
		let mut sum = x[p];
		// libopus walks the reversed filter from the oldest sample.
		for j in 0..ord {
			sum += num[ord - j - 1] * x[p + j - ord];
		}
		*out = sum;
	}
}

/// `celt_iir`, in place: `buf[i] -= sum_k den[k] * out[i - k - 1]`, with
/// `mem` holding the previous outputs (most recent first) and updated on
/// return. Mirrors libopus' four-at-a-time evaluation.
pub fn iir(buf: &mut [f32], den: &[f32], mem: &mut [f32]) {
	let ord = den.len();
	let n = buf.len();
	let rden: Vec<f32> = den.iter().rev().copied().collect();
	// y holds the negated outputs, preceded by the negated memory.
	let mut y = vec![0f32; n + ord];
	for i in 0..ord {
		y[i] = -mem[ord - i - 1];
	}
	let mut i = 0;
	while i + 3 < n {
		let mut sum = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
		for j in 0..ord {
			for (k, s) in sum.iter_mut().enumerate() {
				*s += rden[j] * y[i + j + k];
			}
		}
		y[i + ord] = -sum[0];
		buf[i] = sum[0];
		sum[1] += y[i + ord] * den[0];
		y[i + ord + 1] = -sum[1];
		buf[i + 1] = sum[1];
		sum[2] += y[i + ord + 1] * den[0];
		sum[2] += y[i + ord] * den[1];
		y[i + ord + 2] = -sum[2];
		buf[i + 2] = sum[2];
		sum[3] += y[i + ord + 2] * den[0];
		sum[3] += y[i + ord + 1] * den[1];
		sum[3] += y[i + ord] * den[2];
		y[i + ord + 3] = -sum[3];
		buf[i + 3] = sum[3];
		i += 4;
	}
	while i < n {
		let mut sum = buf[i];
		for j in 0..ord {
			sum -= rden[j] * y[i + j];
		}
		y[i + ord] = sum;
		buf[i] = sum;
		i += 1;
	}
	for k in 0..ord.min(n) {
		mem[k] = buf[n - k - 1];
	}
}

/// `pitch_downsample`: sums the channels, halves the rate and whitens the
/// result with a 4th-order LPC plus a zero (`celt_fir5`).
pub fn downsample(channels: &[&[f32]], len: usize) -> Vec<f32> {
	let half = len >> 1;
	let mut x_lp = vec![0f32; half];
	for (c, x) in channels.iter().enumerate() {
		for i in 1..half {
			let v = 0.25 * x[2 * i - 1] + 0.25 * x[2 * i + 1] + 0.5 * x[2 * i];
			x_lp[i] = if c == 0 { v } else { x_lp[i] + v };
		}
		let v = 0.25 * x[1] + 0.5 * x[0];
		x_lp[0] = if c == 0 { v } else { x_lp[0] + v };
	}
	let mut ac = autocorr(&x_lp, &[], 4);
	// Noise floor at -40 dB, then lag windowing.
	ac[0] *= 1.0001;
	for i in 1..=4 {
		let w = 0.008 * i as f32;
		ac[i] -= ac[i] * w * w;
	}
	let mut lpc4 = lpc(&ac, 4);
	let mut tmp = 1.0f32;
	for v in &mut lpc4 {
		tmp *= 0.9;
		*v *= tmp;
	}
	let c1 = 0.8f32;
	let num = [
		lpc4[0] + 0.8,
		lpc4[1] + c1 * lpc4[0],
		lpc4[2] + c1 * lpc4[1],
		lpc4[3] + c1 * lpc4[2],
		c1 * lpc4[3],
	];
	let mut mem = [0f32; 5];
	for v in &mut x_lp {
		let mut sum = *v;
		for k in 0..5 {
			sum += num[k] * mem[k];
		}
		mem.copy_within(0..4, 1);
		mem[0] = *v;
		*v = sum;
	}
	x_lp
}

/// `find_best_pitch`: the two lags with the best normalised correlation.
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize) -> [usize; 2] {
	let mut syy = 1f32;
	let mut best_num = [-1f32; 2];
	let mut best_den = [0f32; 2];
	let mut best = [0usize, 1];
	for &v in &y[..len] {
		syy += v * v;
	}
	for (i, &xc) in xcorr.iter().enumerate() {
		if xc > 0.0 {
			// Scaled to avoid under- and overflow when squaring.
			let xcorr16 = xc * 1e-12;
			let num = xcorr16 * xcorr16;
			if num * best_den[1] > best_num[1] * syy {
				if num * best_den[0] > best_num[0] * syy {
					best_num[1] = best_num[0];
					best_den[1] = best_den[0];
					best[1] = best[0];
					best_num[0] = num;
					best_den[0] = syy;
					best[0] = i;
				} else {
					best_num[1] = num;
					best_den[1] = syy;
					best[1] = i;
				}
			}
		}
		syy += y[i + len] * y[i + len] - y[i] * y[i];
		syy = syy.max(1.0);
	}
	best
}

/// `pitch_search`: the lag (in `x_lp`'s samples, `0..max_pitch`) where `y`
/// best matches `x_lp`, searched at a quarter and then half the rate.
pub fn search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
	let lag = len + max_pitch;
	let x_lp4: Vec<f32> = (0..len >> 2).map(|j| x_lp[2 * j]).collect();
	let y_lp4: Vec<f32> = (0..lag >> 2).map(|j| y[2 * j]).collect();
	let mut xcorr = vec![0f32; max_pitch >> 1];
	// Coarse search with 4x decimation.
	pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr[..max_pitch >> 2], len >> 2);
	let best = find_best_pitch(&xcorr[..max_pitch >> 2], &y_lp4, len >> 2);
	// Finer search with 2x decimation around the two candidates.
	for i in 0..max_pitch >> 1 {
		xcorr[i] = 0.0;
		let near = |b: usize| (i as i32 - 2 * b as i32).abs() <= 2;
		if !near(best[0]) && !near(best[1]) {
			continue;
		}
		xcorr[i] = inner_prod(&x_lp[..len >> 1], &y[i..i + (len >> 1)]).max(-1.0);
	}
	let best = find_best_pitch(&xcorr, y, len >> 1);
	// Refine by pseudo-interpolation.
	let b0 = best[0];
	let offset = if b0 > 0 && b0 < (max_pitch >> 1) - 1 {
		let (a, b, c) = (xcorr[b0 - 1], xcorr[b0], xcorr[b0 + 1]);
		if c - a > 0.7 * (b - a) {
			1
		} else if a - c > 0.7 * (b - c) {
			-1
		} else {
			0
		}
	} else {
		0
	};
	(2 * b0 as i32 - offset).max(0) as usize
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn finds_the_period_of_a_pulse_train() {
		// Period 300 at 48 kHz: PLC searches lags 100-720.
		let x: Vec<f32> = (0..2048)
			.map(|i| (if i % 300 == 0 { 10_000.0 } else { 0.0 }) + ((i * 7919) % 13) as f32)
			.collect();
		let lp = downsample(&[&x], 2048);
		let found = 720 - search(&lp[360..], &lp, 2048 - 720, 620);
		assert!((found as i32 - 300).abs() <= 2, "found {found}");
	}

	#[test]
	fn lpc_and_filters_invert_each_other() {
		let x: Vec<f32> =
			(0..256).map(|i| ((i as f32) * 0.07).sin() * 1000.0 + (i % 5) as f32).collect();
		let a = lpc(&autocorr(&x, &[], 4), 4);
		let mut padded = vec![0f32; 4];
		padded.extend_from_slice(&x);
		let mut residual = vec![0f32; x.len()];
		fir(&padded, 4, &a, &mut residual);
		let mut mem = [0f32; 4];
		iir(&mut residual, &a, &mut mem);
		for (r, v) in residual.iter().zip(&x) {
			assert!((r - v).abs() < 0.05, "{r} vs {v}");
		}
	}
}
