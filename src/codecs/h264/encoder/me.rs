//! Motion estimation helpers: an edge-padded copy of the reference luma
//! for branch-free integer SAD, and the block cost functions.
//!
//! x264: `common/frame.c` (`x264_frame_expand_border`), `pixel.c`
//! (`pixel_sad_*`, `pixel_satd_*`), `encoder/me.c` (diamond search and
//! sub-pixel refinement).

use super::transform::satd4;
use crate::codecs::h264::recon::Plane;

/// Border of the padded reference, in pixels. Integer search keeps blocks
/// inside it; sub-pixel positions use the decoder's interpolation, which
/// clamps at the picture edge exactly as a decoder does.
pub const PAD: i32 = 48;

/// Reference luma with [`PAD`] samples of edge replication on every side.
#[derive(Debug, Clone)]
pub struct Padded {
	pub data: Vec<u8>,
	pub stride: usize,
	pub width: i32,
	pub height: i32,
}

impl Padded {
	pub fn new(p: &Plane) -> Self {
		let stride = p.width + 2 * PAD as usize;
		let rows = p.height + 2 * PAD as usize;
		let mut data = vec![0u8; stride * rows];
		for y in 0..rows {
			let sy = (y as i32 - PAD).clamp(0, p.height as i32 - 1) as usize;
			let src = &p.data[sy * p.width..(sy + 1) * p.width];
			let row = &mut data[y * stride..(y + 1) * stride];
			let (left, rest) = row.split_at_mut(PAD as usize);
			let (mid, right) = rest.split_at_mut(p.width);
			left.fill(src[0]);
			mid.copy_from_slice(src);
			right.fill(src[p.width - 1]);
		}
		Self { data, stride, width: p.width as i32, height: p.height as i32 }
	}

	/// Whether a `bw` x `bh` block at (x, y) lies inside the padding.
	#[inline]
	pub fn fits(&self, x: i32, y: i32, bw: usize, bh: usize) -> bool {
		x >= -PAD
			&& y >= -PAD
			&& x + bw as i32 <= self.width + PAD
			&& y + bh as i32 <= self.height + PAD
	}

	/// Sum of absolute differences between `src` at (sx, sy) and the
	/// reference at (x, y). The caller ensures [`Padded::fits`].
	#[allow(clippy::too_many_arguments)]
	pub fn sad(
		&self,
		src: &Plane,
		sx: usize,
		sy: usize,
		x: i32,
		y: i32,
		bw: usize,
		bh: usize,
	) -> u32 {
		let base = ((y + PAD) as usize) * self.stride + (x + PAD) as usize;
		let mut sum = 0u32;
		for j in 0..bh {
			let r = &self.data[base + j * self.stride..base + j * self.stride + bw];
			let s = &src.data[(sy + j) * src.width + sx..(sy + j) * src.width + sx + bw];
			for (a, b) in r.iter().zip(s) {
				sum += u32::from(a.abs_diff(*b));
			}
		}
		sum
	}
}

/// Half-sample planes of a padded reference (8.4.2.2.1): `b` (between a
/// sample and its right neighbour), `h` (and the one below) and `j` (the
/// centre), each indexed like [`Padded::data`]. With them, every
/// quarter-sample prediction is one or two lookups instead of a 6-tap
/// filter per candidate (x264's `hpel` planes); the values are exactly
/// those of the decoder's `mc_luma`.
#[derive(Debug, Clone)]
pub struct HalfPel {
	pub b: Vec<u8>,
	pub h: Vec<u8>,
	pub j: Vec<u8>,
}

/// A sample plane and an offset from the integer position.
type Tap<'a> = (&'a [u8], usize);

#[inline]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
	a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline]
fn clip8(v: i32) -> u8 {
	v.clamp(0, 255) as u8
}

impl HalfPel {
	pub fn new(p: &Padded) -> Self {
		let (stride, rows) = (p.stride, p.data.len() / p.stride);
		let at = |x: isize, y: isize| -> i32 {
			let x = x.clamp(0, stride as isize - 1) as usize;
			let y = y.clamp(0, rows as isize - 1) as usize;
			i32::from(p.data[y * stride + x])
		};
		let n = p.data.len();
		let (mut b, mut h, mut j) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
		// Unrounded horizontal intermediates b1, needed for j.
		let mut b1 = vec![0i32; n];
		for y in 0..rows as isize {
			for x in 0..stride as isize {
				let v =
					tap6(at(x - 2, y), at(x - 1, y), at(x, y), at(x + 1, y), at(x + 2, y), at(x + 3, y));
				let k = y as usize * stride + x as usize;
				b1[k] = v;
				b[k] = clip8((v + 16) >> 5);
				let v =
					tap6(at(x, y - 2), at(x, y - 1), at(x, y), at(x, y + 1), at(x, y + 2), at(x, y + 3));
				h[k] = clip8((v + 16) >> 5);
			}
		}
		let b1_at = |x: usize, y: isize| b1[y.clamp(0, rows as isize - 1) as usize * stride + x];
		for y in 0..rows as isize {
			for x in 0..stride {
				let v = tap6(
					b1_at(x, y - 2),
					b1_at(x, y - 1),
					b1_at(x, y),
					b1_at(x, y + 1),
					b1_at(x, y + 2),
					b1_at(x, y + 3),
				);
				j[y as usize * stride + x] = clip8((v + 512) >> 10);
			}
		}
		Self { b, h, j }
	}

	/// Quarter-sample prediction of a `bw` x `bh` block at integer (x, y)
	/// with fraction (fx, fy), as `mc_luma` computes it (Table 8-12). The
	/// block plus one sample right and below must lie inside the padding.
	#[allow(clippy::too_many_arguments)]
	pub fn predict(
		&self,
		p: &Padded,
		x: i32,
		y: i32,
		fx: i32,
		fy: i32,
		bw: usize,
		bh: usize,
		out: &mut [u8],
	) {
		let s = p.stride;
		let base = ((y + PAD) as usize) * s + (x + PAD) as usize;
		let (g, b, h, j) = (&p.data[..], &self.b[..], &self.h[..], &self.j[..]);
		// Each position is one plane or the rounded average of two
		// (Table 8-12), at fixed offsets from the integer sample.
		let (one, two): (Tap, Option<Tap>) = match (fx, fy) {
			(0, 0) => ((g, 0), None),
			(1, 0) => ((g, 0), Some((b, 0))),
			(2, 0) => ((b, 0), None),
			(3, 0) => ((g, 1), Some((b, 0))),
			(0, 1) => ((g, 0), Some((h, 0))),
			(0, 2) => ((h, 0), None),
			(0, 3) => ((g, s), Some((h, 0))),
			(2, 2) => ((j, 0), None),
			(2, 1) => ((b, 0), Some((j, 0))),
			(2, 3) => ((j, 0), Some((b, s))),
			(1, 2) => ((h, 0), Some((j, 0))),
			(3, 2) => ((j, 0), Some((h, 1))),
			(1, 1) => ((b, 0), Some((h, 0))),
			(3, 1) => ((b, 0), Some((h, 1))),
			(1, 3) => ((h, 0), Some((b, s))),
			_ => ((h, 1), Some((b, s))),
		};
		for r in 0..bh {
			let row = &mut out[r * bw..(r + 1) * bw];
			let k = base + r * s;
			let a = &one.0[k + one.1..k + one.1 + bw];
			match two {
				None => row.copy_from_slice(a),
				Some((plane, off)) => {
					let c = &plane[k + off..k + off + bw];
					for ((o, &x), &y) in row.iter_mut().zip(a).zip(c) {
						*o = ((u16::from(x) + u16::from(y) + 1) >> 1) as u8;
					}
				}
			}
		}
	}
}

/// SATD between `src` at (sx, sy) and a `bw` x `bh` prediction block
/// (stride `bw`).
pub fn satd_block(src: &Plane, sx: usize, sy: usize, pred: &[u8], bw: usize, bh: usize) -> u32 {
	let mut sum = 0;
	for by in (0..bh).step_by(4) {
		for bx in (0..bw).step_by(4) {
			let mut d = [0i32; 16];
			for j in 0..4 {
				for i in 0..4 {
					let s = src.data[(sy + by + j) * src.width + sx + bx + i];
					d[j * 4 + i] = i32::from(s) - i32::from(pred[(by + j) * bw + bx + i]);
				}
			}
			sum += satd4(&d);
		}
	}
	sum
}

/// SATD of the quarter-sample motion vector `mv` for a block at (px, py),
/// predicted from the half-sample planes. The block (with one extra
/// sample right and below) must lie inside the padding.
#[allow(clippy::too_many_arguments)]
pub fn satd_mv(
	p: &Padded,
	hp: &HalfPel,
	src: &Plane,
	px: usize,
	py: usize,
	bw: usize,
	bh: usize,
	mv: [i32; 2],
) -> u32 {
	let mut pred = [0u8; 256];
	hp.predict(
		p,
		px as i32 + (mv[0] >> 2),
		py as i32 + (mv[1] >> 2),
		mv[0] & 3,
		mv[1] & 3,
		bw,
		bh,
		&mut pred,
	);
	satd_block(src, px, py, &pred, bw, bh)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn padded_sad_matches_clamped_reads() {
		let mut p = Plane::new(32, 16, 0);
		for (i, v) in p.data.iter_mut().enumerate() {
			*v = (i * 7 % 251) as u8;
		}
		let pad = Padded::new(&p);
		let src = Plane::new(32, 16, 100);
		for (x, y) in [(0, 0), (-20, -10), (20, 5), (30, 14)] {
			let want: u32 = (0..8)
				.flat_map(|j| (0..8).map(move |i| (i, j)))
				.map(|(i, j)| u32::from(p.clamped(x + i, y + j).abs_diff(100)))
				.sum();
			assert_eq!(pad.sad(&src, 0, 0, x, y, 8, 8), want);
		}
	}

	#[test]
	fn half_pel_planes_match_mc_luma() {
		let mut p = Plane::new(40, 24, 0);
		for (i, v) in p.data.iter_mut().enumerate() {
			*v = ((i * 7919) % 256) as u8;
		}
		let pad = Padded::new(&p);
		let hp = HalfPel::new(&pad);
		for (x, y) in [(0, 0), (5, 3), (-10, -8), (30, 18), (38, 22), (-40, 30)] {
			for fy in 0..4 {
				for fx in 0..4 {
					let (mut want, mut got) = ([0u8; 64], [0u8; 64]);
					crate::codecs::h264::recon::mc_luma(&p, x, y, fx, fy, 8, 8, &mut want);
					hp.predict(&pad, x, y, fx, fy, 8, 8, &mut got);
					assert_eq!(want, got, "({x}, {y}) frac ({fx}, {fy})");
				}
			}
		}
	}
}
