//! Linear prediction analysis for the FLAC encoder (FFmpeg:
//! `libavcodec/lpc.c`; libFLAC: `lpc.c`).
//!
//! Windowed autocorrelation → Levinson-Durbin recursion gives predictor
//! coefficients for every order up to the maximum, plus the prediction
//! error of each order (used to estimate the best order). Coefficients are
//! then quantised to integers with error feedback.

/// Tukey(0.5) window, as libFLAC and FFmpeg's default `-lpc_type levinson`
/// analysis use (a Welch window was FFmpeg's older choice).
pub fn tukey_window(n: usize) -> Vec<f64> {
	let p = 0.5;
	let taper = ((p / 2.0) * n as f64) as usize;
	(0..n)
		.map(|i| {
			if taper == 0 {
				1.0
			} else if i < taper {
				0.5 - 0.5 * (std::f64::consts::PI * i as f64 / taper as f64).cos()
			} else if i >= n - taper {
				0.5 - 0.5 * (std::f64::consts::PI * (n - 1 - i) as f64 / taper as f64).cos()
			} else {
				1.0
			}
		})
		.collect()
}

/// Autocorrelation `r[0..=max_lag]` of the windowed signal.
pub fn autocorrelation(samples: &[i64], window: &[f64], max_lag: usize) -> Vec<f64> {
	let x: Vec<f64> = samples.iter().zip(window).map(|(&s, &w)| s as f64 * w).collect();
	(0..=max_lag).map(|lag| x.iter().zip(x.iter().skip(lag)).map(|(a, b)| a * b).sum()).collect()
}

/// Levinson-Durbin recursion. Returns, for each order `1..=max_order`, the
/// predictor coefficients and the residual energy. Stops early when the
/// error vanishes (a perfectly predictable signal).
pub fn levinson(r: &[f64], max_order: usize) -> Vec<(Vec<f64>, f64)> {
	let mut out = Vec::with_capacity(max_order);
	if r.is_empty() || r[0] <= 0.0 {
		return out;
	}
	let mut a = vec![0.0f64; max_order + 1];
	let mut err = r[0];
	for m in 1..=max_order.min(r.len() - 1) {
		let mut acc = r[m];
		for j in 1..m {
			acc -= a[j] * r[m - j];
		}
		let k = acc / err;
		let prev = a.clone();
		a[m] = k;
		for j in 1..m {
			a[j] = prev[j] - k * prev[m - j];
		}
		err *= 1.0 - k * k;
		out.push((a[1..=m].to_vec(), err.max(0.0)));
		if err <= 0.0 {
			break;
		}
	}
	out
}

/// Quantises coefficients to `precision`-bit integers with a right shift
/// (libFLAC's `FLAC__lpc_quantize_coefficients`). Returns `(coefs, shift)`,
/// or `None` if the coefficients can't be represented.
pub fn quantize(coefs: &[f64], precision: u32) -> Option<(Vec<i32>, u32)> {
	let cmax = coefs.iter().fold(0.0f64, |m, c| m.max(c.abs()));
	if !cmax.is_finite() || cmax <= 0.0 {
		return None;
	}
	let max_q = (1i64 << (precision - 1)) - 1;
	let min_q = -(1i64 << (precision - 1));
	let log2cmax = cmax.log2().floor() as i32 + 1;
	let shift = (precision as i32 - 1 - log2cmax).clamp(0, 15) as u32;
	let scale = f64::from(1u32 << shift);
	let mut error = 0.0;
	let mut out = Vec::with_capacity(coefs.len());
	for &c in coefs {
		error += c * scale;
		let q = (error.round() as i64).clamp(min_q, max_q);
		error -= q as f64;
		out.push(q as i32);
	}
	Some((out, shift))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn predicts_a_sine() {
		let n = 4096;
		let x: Vec<i64> = (0..n).map(|i| (10_000.0 * (i as f64 * 0.05).sin()).round() as i64).collect();
		let r = autocorrelation(&x, &tukey_window(n), 8);
		let orders = levinson(&r, 8);
		// A sinusoid is almost perfectly predicted by order 2.
		let (a2, e2) = &orders[1];
		assert!(*e2 < orders[0].1 / 100.0, "{e2}");
		assert!((a2[0] - 2.0 * 0.05f64.cos()).abs() < 0.01);
		let (q, shift) = quantize(a2, 15).unwrap();
		assert!(shift > 0 && q.iter().all(|&c| c.abs() < 1 << 14));
	}
}
