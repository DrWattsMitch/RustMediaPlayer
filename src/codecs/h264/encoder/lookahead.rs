//! Lookahead: frame-type decisions from half-resolution analysis (x264
//! `encoder/slicetype.c`).
//!
//! Every input picture gets a half-resolution luma copy ([`Lowres`]); one
//! 8x8 block of it stands for one macroblock. [`frame_cost`] estimates
//! what a picture costs coded as P (from a past picture) or as B (from a
//! past and a future one): per block, the cheapest of an intra estimate,
//! a small motion search in either reference, and their average, in SATD
//! plus a motion-vector term. x264 calls this `slicetype_frame_cost`.
//!
//! [`decide`] then picks the next anchor and how many B pictures precede
//! it with x264's fast adaptive B-frame algorithm (`--b-adapt 1`), and
//! [`is_scenecut`] finds pictures that predict so badly from the previous
//! one that they should start a new GOP (`scenecut_internal`, threshold
//! 40).

use super::macroblock::Source;
use super::transform::satd4;

/// Half-resolution luma of one picture, with its intra cost estimate.
#[derive(Debug, Clone)]
pub struct Lowres {
	pub w: usize,
	pub h: usize,
	pub data: Vec<u8>,
	/// Intra estimate per 8x8 block.
	pub intra: Vec<u32>,
}

/// Search range (in lowres samples) of the lookahead motion search.
const RANGE: i32 = 16;
/// Per-block bias of inter over intra (x264 adds a bit cost to intra).
const INTRA_PENALTY: u32 = 5 * 4;

impl Lowres {
	pub fn new(src: &Source) -> Self {
		let (w, h) = (src.y.width / 2, src.y.height / 2);
		let mut data = vec![0u8; w * h];
		for y in 0..h {
			for x in 0..w {
				let s =
					|dx: usize, dy: usize| u32::from(src.y.data[(2 * y + dy) * src.y.width + 2 * x + dx]);
				data[y * w + x] = ((s(0, 0) + s(1, 0) + s(0, 1) + s(1, 1) + 2) >> 2) as u8;
			}
		}
		let mut l = Self { w, h, data, intra: Vec::new() };
		l.intra = l.intra_costs();
		l
	}

	pub fn blocks(&self) -> usize {
		(self.w / 8) * (self.h / 8)
	}

	#[inline]
	fn at(&self, x: i32, y: i32) -> u8 {
		let x = x.clamp(0, self.w as i32 - 1) as usize;
		let y = y.clamp(0, self.h as i32 - 1) as usize;
		self.data[y * self.w + x]
	}

	/// SATD of the 8x8 block at (bx, by) against `pred` (64 samples).
	fn satd8(&self, bx: usize, by: usize, pred: &[u8; 64]) -> u32 {
		let mut sum = 0;
		for q in 0..4 {
			let (ox, oy) = ((q % 2) * 4, (q / 2) * 4);
			let mut d = [0i32; 16];
			for j in 0..4 {
				for i in 0..4 {
					let s = self.data[(by * 8 + oy + j) * self.w + bx * 8 + ox + i];
					d[j * 4 + i] = i32::from(s) - i32::from(pred[(oy + j) * 8 + ox + i]);
				}
			}
			sum += satd4(&d);
		}
		sum
	}

	/// DC, horizontal and vertical prediction from the neighbouring
	/// source samples; the cheapest SATD per block.
	fn intra_costs(&self) -> Vec<u32> {
		let (bw, bh) = (self.w / 8, self.h / 8);
		let mut out = Vec::with_capacity(bw * bh);
		for by in 0..bh {
			for bx in 0..bw {
				let (x0, y0) = ((bx * 8) as i32, (by * 8) as i32);
				let top: [u8; 8] = std::array::from_fn(|i| self.at(x0 + i as i32, y0 - 1));
				let left: [u8; 8] = std::array::from_fn(|j| self.at(x0 - 1, y0 + j as i32));
				let dc = match (bx > 0, by > 0) {
					(true, true) => (top.iter().chain(&left).map(|&v| u32::from(v)).sum::<u32>() + 8) >> 4,
					(false, true) => (top.iter().map(|&v| u32::from(v)).sum::<u32>() + 4) >> 3,
					(true, false) => (left.iter().map(|&v| u32::from(v)).sum::<u32>() + 4) >> 3,
					(false, false) => 128,
				} as u8;
				let mut best = self.satd8(bx, by, &[dc; 64]);
				if by > 0 {
					best = best.min(self.satd8(bx, by, &std::array::from_fn(|k| top[k % 8])));
				}
				if bx > 0 {
					best = best.min(self.satd8(bx, by, &std::array::from_fn(|k| left[k / 8])));
				}
				out.push(best + INTRA_PENALTY);
			}
		}
		out
	}

	fn block(&self, x: i32, y: i32) -> [u8; 64] {
		std::array::from_fn(|k| self.at(x + (k % 8) as i32, y + (k / 8) as i32))
	}

	fn sad8(&self, other: &Lowres, bx: usize, by: usize, mx: i32, my: i32) -> u32 {
		let (x0, y0) = ((bx * 8) as i32, (by * 8) as i32);
		let mut sum = 0;
		for j in 0..8 {
			for i in 0..8 {
				let s = self.data[(by * 8 + j) * self.w + bx * 8 + i];
				let r = other.at(x0 + i as i32 + mx, y0 + j as i32 + my);
				sum += u32::from(s.abs_diff(r));
			}
		}
		sum
	}
}

fn mv_cost(mv: [i32; 2], pred: [i32; 2]) -> u32 {
	// About 2·log2 bits per component at the lookahead's λ (4).
	let bits = |d: i32| 2 * (32 - (d.unsigned_abs() * 2 + 1).leading_zeros());
	4 * (bits(mv[0] - pred[0]) + bits(mv[1] - pred[1]))
}

/// Integer diamond search of block (bx, by) of `cur` in `r`; returns the
/// vector and its SATD + vector cost.
fn search(cur: &Lowres, r: &Lowres, bx: usize, by: usize, pred: [i32; 2]) -> ([i32; 2], u32) {
	let cost = |mv: [i32; 2]| cur.sad8(r, bx, by, mv[0], mv[1]) + mv_cost(mv, pred);
	let mut best = ([0, 0], cost([0, 0]));
	let p = [pred[0].clamp(-RANGE, RANGE), pred[1].clamp(-RANGE, RANGE)];
	let cp = cost(p);
	if cp < best.1 {
		best = (p, cp);
	}
	for step in [4, 2, 1] {
		loop {
			let centre = best;
			for (dx, dy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
				let mv = [centre.0[0] + dx, centre.0[1] + dy];
				if mv[0].abs() > RANGE || mv[1].abs() > RANGE {
					continue;
				}
				let c = cost(mv);
				if c < best.1 {
					best = (mv, c);
				}
			}
			if best.0 == centre.0 {
				break;
			}
		}
	}
	let (x0, y0) = ((bx * 8) as i32, (by * 8) as i32);
	let pred_blk = r.block(x0 + best.0[0], y0 + best.0[1]);
	(best.0, cur.satd8(bx, by, &pred_blk) + mv_cost(best.0, pred))
}

/// Estimated cost of `b` predicted from `p0` (past) and, for a B
/// picture, `p1` (future), with the number of blocks that chose intra.
pub fn frame_cost(p0: &Lowres, p1: Option<&Lowres>, b: &Lowres) -> (u64, usize) {
	let (bw, bh) = (b.w / 8, b.h / 8);
	let mut mvs0 = vec![[0i32; 2]; bw * bh];
	let mut mvs1 = vec![[0i32; 2]; bw * bh];
	let mut total = 0u64;
	let mut intra = 0usize;
	for by in 0..bh {
		for bx in 0..bw {
			let i = by * bw + bx;
			let pred = |mvs: &[[i32; 2]]| -> [i32; 2] {
				match (bx > 0, by > 0) {
					(true, true) => {
						let (a, c) = (mvs[i - 1], mvs[i - bw]);
						[(a[0] + c[0]) / 2, (a[1] + c[1]) / 2]
					}
					(true, false) => mvs[i - 1],
					(false, true) => mvs[i - bw],
					_ => [0, 0],
				}
			};
			let (mv0, c0) = search(b, p0, bx, by, pred(&mvs0));
			mvs0[i] = mv0;
			let mut best = c0;
			if let Some(p1) = p1 {
				let (mv1, c1) = search(b, p1, bx, by, pred(&mvs1));
				mvs1[i] = mv1;
				best = best.min(c1);
				let (x0, y0) = ((bx * 8) as i32, (by * 8) as i32);
				let a = p0.block(x0 + mv0[0], y0 + mv0[1]);
				let c = p1.block(x0 + mv1[0], y0 + mv1[1]);
				let avg: [u8; 64] =
					std::array::from_fn(|k| ((u16::from(a[k]) + u16::from(c[k]) + 1) >> 1) as u8);
				let cb = b.satd8(bx, by, &avg) + mv_cost(mv0, [0, 0]) / 2 + mv_cost(mv1, [0, 0]) / 2;
				best = best.min(cb);
			}
			if b.intra[i] < best {
				best = b.intra[i];
				intra += 1;
			}
			total += u64::from(best);
		}
	}
	(total, intra)
}

/// Whether `cur`, predicted from `prev`, is a scene cut. `since_idr` is
/// its distance from the last IDR picture and `gop` the maximum.
pub fn is_scenecut(prev: &Lowres, cur: &Lowres, since_idr: u32, gop: u32) -> bool {
	let icost: u64 = cur.intra.iter().map(|&c| u64::from(c)).sum();
	let (pcost, _) = frame_cost(prev, None, cur);
	let (t_max, t_min) = (0.40f64, 0.10f64);
	let keyint_min = (gop / 10).max(1);
	let bias = if since_idr <= keyint_min / 4 {
		t_min / 4.0
	} else if since_idr <= keyint_min {
		t_min * f64::from(since_idr) / f64::from(keyint_min)
	} else {
		t_min
			+ (t_max - t_min) * f64::from(since_idr - keyint_min) / f64::from((gop - keyint_min).max(1))
	};
	pcost as f64 >= (1.0 - bias) * icost as f64
}

/// The number of B pictures before the next anchor (x264 `--b-adapt 1`).
/// `frames[0]` is the last coded anchor, `frames[1..]` the queued pictures
/// in display order (at most `bframes + 1` of them are looked at).
pub fn decide(frames: &[&Lowres], bframes: usize) -> usize {
	let n = frames.len() - 1;
	if bframes == 0 || n < 2 {
		return 0;
	}
	let mbs = frames[0].blocks() as u64;
	let (cost2p1, intra2) = frame_cost(frames[0], None, frames[2]);
	if intra2 as u64 > mbs / 2 {
		return 0;
	}
	let (cost1b1, _) = frame_cost(frames[0], Some(frames[2]), frames[1]);
	let (cost1p0, _) = frame_cost(frames[0], None, frames[1]);
	let (cost2p0, _) = frame_cost(frames[1], None, frames[2]);
	if cost1p0 + cost2p0 < cost1b1 + cost2p1 {
		return 0;
	}
	// frames[1] is a B picture; extend the run while P pictures further
	// out stay cheap.
	const INTER_THRESH: i64 = 300;
	const P_SENS_BIAS: i64 = 50;
	let mut j = 2;
	while j <= bframes.min(n - 1) {
		let pthresh = (INTER_THRESH - P_SENS_BIAS * (j as i64 - 1)).max(INTER_THRESH / 10);
		let (pcost, intra) = frame_cost(frames[0], None, frames[j + 1]);
		if pcost as i64 > pthresh * mbs as i64 || intra as u64 > mbs / 3 {
			break;
		}
		j += 1;
	}
	j - 1
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::h264::recon::Plane;

	fn source(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Source {
		let mut y = Plane::new(w, h, 0);
		for j in 0..h {
			for i in 0..w {
				y.data[j * w + i] = f(i, j);
			}
		}
		Source { y, cb: Plane::new(w / 2, h / 2, 128), cr: Plane::new(w / 2, h / 2, 128) }
	}

	#[test]
	fn smooth_motion_prefers_b_frames_and_cuts_are_found() {
		// A texture panning one sample per picture.
		let tex = |t: usize| {
			move |x: usize, y: usize| {
				let (u, v) = ((x + t) as f64, y as f64);
				(128.0 + 60.0 * (u * 0.3).sin() * (v * 0.2).cos() + 30.0 * (u * 0.07 + v * 0.11).sin())
					as u8
			}
		};
		let frames: Vec<Lowres> = (0..5).map(|t| Lowres::new(&source(64, 64, tex(t)))).collect();
		let refs: Vec<&Lowres> = frames.iter().collect();
		assert!(decide(&refs, 3) >= 1);
		let other = Lowres::new(&source(64, 64, |x, y| ((x * x + y * 31) % 251) as u8));
		assert!(is_scenecut(&frames[0], &other, 30, 250));
		assert!(!is_scenecut(&frames[0], &frames[1], 30, 250));
	}
}
