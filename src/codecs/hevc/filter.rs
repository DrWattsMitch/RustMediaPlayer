//! In-loop filters: deblocking (§8.7.2) and sample adaptive offset
//! (§8.7.3), run CTB row by CTB row as decoding completes the rows, a row
//! or two behind (FFmpeg `hevc/filter.c`, `hevc/dsp_template.c`). The
//! result equals filtering the whole picture at once; see
//! [`PicState::advance`].

use super::ctu::{PicState, chroma_qp};
use super::picture::Plane;

const TC_TABLE: [i32; 54] = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
	3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];
const BETA_TABLE: [i32; 52] = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

impl PicState {
	#[inline]
	fn qp_at(&self, x: usize, y: usize) -> i32 {
		let s = self.sps.log2_min_cb_size;
		i32::from(self.qp_y[(y >> s) * self.sps.min_cb_width as usize + (x >> s)])
	}

	#[inline]
	fn slice_at(&self, x: usize, y: usize) -> &super::ctu::SliceInfo {
		let l = self.sps.log2_ctb_size;
		let ctb = (y >> l) * self.sps.ctb_width as usize + (x >> l);
		&self.slices[self.ctb_slice[ctb] as usize]
	}

	#[inline]
	fn nf(&self, x: usize, y: usize) -> bool {
		self.no_filter[(y >> 2) * self.w4 + (x >> 2)] != 0
	}

	/// Deblocks the edges in luma rows `y0..y1` (a band of whole CTB rows):
	/// the vertical edges of those rows, then the horizontal edges at
	/// `y0 <= y < y1`. Running the bands top to bottom gives the same
	/// result as filtering every vertical edge of the picture first: a
	/// horizontal edge at `y` reads rows `y - 4..y + 4`, which the band and
	/// the one above have already filtered vertically, and modifies only
	/// rows `y - 3..y + 3`, which no later vertical edge reads.
	pub fn deblock_rows(&mut self, y0: usize, y1: usize) {
		let (w, h) = (self.sps.width as usize, y1.min(self.sps.height as usize));
		let bd = self.sps.bit_depth;
		let max = (1i32 << bd) - 1;
		let s = self.planes[0].stride;
		for dir in 0..2 {
			// Luma.
			if bd <= 10 && crate::simd::enabled() {
				self.deblock_luma_simd(dir, max, y0, h);
			} else {
				let (xs, ys, xstep, ystep) = if dir == 0 { (8, 0, 8, 4) } else { (0, 8, 4, 8) };
				for y in (ys.max(y0)..h).step_by(ystep) {
					for x in (xs..w).step_by(xstep) {
						let i4 = (y >> 2) * self.w4 + (x >> 2);
						let bs = i32::from(if dir == 0 { self.bs_v[i4] } else { self.bs_h[i4] });
						if bs == 0 {
							continue;
						}
						let (xp, yp) = if dir == 0 { (x - 1, y) } else { (x, y - 1) };
						let qp = (self.qp_at(xp, yp) + self.qp_at(x, y) + 1) >> 1;
						let sl = self.slice_at(x, y);
						let beta = BETA_TABLE[(qp + sl.beta_offset).clamp(0, 51) as usize] << (bd - 8);
						let tc = TC_TABLE[(qp + 2 * (bs - 1) + sl.tc_offset).clamp(0, 53) as usize] << (bd - 8);
						let (no_p, no_q) = (self.nf(xp, yp), self.nf(x, y));
						let plane = &mut self.planes[0];
						let at = plane.idx(x as isize, y as isize);
						let (xstr, ystr) = if dir == 0 { (1, s) } else { (s, 1) };
						luma_edge(&mut plane.data, at, xstr, ystr, beta, tc, no_p, no_q, max);
					}
				}
			}
			// Chroma (4:2:0): edges on the 8x8 chroma grid, bS == 2 only.
			for c in 1..3 {
				let cs = self.planes[c].stride;
				let (xs, ys, xstep, ystep) = if dir == 0 { (16, 0, 16, 8) } else { (0, 16, 8, 16) };
				let qp_off = if c == 1 { self.pps.cb_qp_offset } else { self.pps.cr_qp_offset };
				for y in (ys.max(y0)..h).step_by(ystep) {
					for x in (xs..w).step_by(xstep) {
						let i4 = (y >> 2) * self.w4 + (x >> 2);
						let bs = if dir == 0 { self.bs_v[i4] } else { self.bs_h[i4] };
						if bs != 2 {
							continue;
						}
						let (xp, yp) = if dir == 0 { (x - 1, y) } else { (x, y - 1) };
						let qp = (self.qp_at(xp, yp) + self.qp_at(x, y) + 1) >> 1;
						let tc_off = self.slice_at(x, y).tc_offset;
						let qpc = chroma_qp((qp + qp_off).clamp(0, 57));
						let tc = TC_TABLE[(qpc + 2 + tc_off).clamp(0, 53) as usize] << (bd - 8);
						if tc == 0 {
							continue;
						}
						let (no_p, no_q) = (self.nf(xp, yp), self.nf(x, y));
						let plane = &mut self.planes[c];
						let at = plane.idx((x / 2) as isize, (y / 2) as isize);
						let (xstr, ystr) = if dir == 0 { (1, cs) } else { (cs, 1) };
						chroma_edge(&mut plane.data, at, xstr, ystr, tc, no_p, no_q, max);
					}
				}
			}
		}
	}

	/// Luma edges of one direction, two 4-line segments at a time.
	fn deblock_luma_simd(&mut self, dir: usize, max: i32, y0: usize, h: usize) {
		let w = self.sps.width as usize;
		let bd = self.sps.bit_depth;
		let s = self.planes[0].stride;
		let (xs, ys) = if dir == 0 { (8, 0) } else { (0, 8) };
		for y in (ys.max(y0)..h).step_by(8) {
			for x in (xs..w).step_by(8) {
				let mut beta = [0; 2];
				let mut tc = [0; 2];
				let mut no_p = [false; 2];
				let mut no_q = [false; 2];
				for j in 0..2 {
					// Segment j: 4 lines along the edge.
					let (sx, sy) = if dir == 0 { (x, y + 4 * j) } else { (x + 4 * j, y) };
					let i4 = (sy >> 2) * self.w4 + (sx >> 2);
					let bs = i32::from(if dir == 0 { self.bs_v[i4] } else { self.bs_h[i4] });
					if bs == 0 {
						continue;
					}
					let (xp, yp) = if dir == 0 { (sx - 1, sy) } else { (sx, sy - 1) };
					let qp = (self.qp_at(xp, yp) + self.qp_at(sx, sy) + 1) >> 1;
					let sl = self.slice_at(sx, sy);
					beta[j] = BETA_TABLE[(qp + sl.beta_offset).clamp(0, 51) as usize] << (bd - 8);
					tc[j] = TC_TABLE[(qp + 2 * (bs - 1) + sl.tc_offset).clamp(0, 53) as usize] << (bd - 8);
					no_p[j] = self.nf(xp, yp);
					no_q[j] = self.nf(sx, sy);
				}
				if tc == [0, 0] {
					continue;
				}
				let plane = &mut self.planes[0];
				let at = plane.idx(x as isize, y as isize);
				if dir == 0 {
					super::simd::deblock_luma_v(&mut plane.data, s, at, beta, tc, no_p, no_q, max);
				} else {
					super::simd::deblock_luma_h(&mut plane.data, s, at, beta, tc, no_p, no_q, max);
				}
			}
		}
	}

	/// Applies SAO to CTB row `r`. Rows above must have been done already,
	/// and the picture deblocked down to the row below: SAO reads the
	/// deblocked samples one row outside the band, so they are kept in
	/// `sao_src` (rows are copied in as they are first needed, before SAO
	/// of the band above has changed them).
	pub fn sao_row(&mut self, r: usize) {
		let log2 = self.sps.log2_ctb_size as usize;
		let (cw, ch) = (self.sps.ctb_width as usize, self.sps.ctb_height as usize);
		let band = &self.sao[r * cw..(r + 1) * cw];
		if !band.iter().any(|s| s.type_idx.iter().any(|&t| t != 0)) {
			return;
		}
		let bd = self.sps.bit_depth;
		let max = (1i32 << bd) - 1;
		let (y0l, y1l) = (r << log2, ((r + 1) << log2).min(self.sps.height as usize));
		let any_nf =
			self.no_filter[(y0l >> 2) * self.w4..(y1l.div_ceil(4)) * self.w4].iter().any(|&v| v != 0);
		for c in 0..3 {
			if !self.sao[r * cw..(r + 1) * cw].iter().any(|s| s.type_idx[c] != 0) {
				continue;
			}
			let sh = usize::from(c > 0);
			let (pw, ph) = (self.planes[c].width, self.planes[c].height);
			let ctb = (1usize << log2) >> sh;
			// The deblocked rows y0 - 1 ..= y1 of this band, the rows SAO reads.
			// Row y0 - 1 has already had SAO applied with the band above, so
			// it comes from the copy kept then.
			let (y0, y1) = (r * ctb, ((r + 1) * ctb).min(ph));
			let plane = &self.planes[c];
			let stride = plane.stride;
			let mut src =
				BandSrc { data: std::mem::take(&mut self.sao_band[c]), stride, pad: plane.pad, y0 };
			src.data.resize((ctb + 2) * stride, 0);
			if y0 > 0 {
				let row = &mut src.data[..stride];
				if self.sao_last_row[c] == Some(y0 - 1) {
					row.copy_from_slice(&self.sao_last[c]);
				} else {
					row.copy_from_slice(&plane.data[plane.rows_range(y0 as isize - 1, y0 as isize)]);
				}
			}
			let to = (y1 + 1).min(ph);
			src.data[stride..(to - y0 + 1) * stride]
				.copy_from_slice(&plane.data[plane.rows_range(y0 as isize, to as isize)]);
			self.sao_last[c].clear();
			self.sao_last[c].extend_from_slice(&src.data[(y1 - y0) * stride..(y1 - y0 + 1) * stride]);
			self.sao_last_row[c] = Some(y1 - 1);
			// Destination indexed like `src`.
			let boff = (y0 + plane.pad - 1) * stride;
			let cy = r;
			for cx in 0..cw {
				let rs = cy * cw + cx;
				let sao = self.sao[rs];
				let t = sao.type_idx[c];
				if t == 0 {
					continue;
				}
				let x0 = cx * ctb;
				let y0 = cy * ctb;
				let x1 = (x0 + ctb).min(pw);
				let y1 = (y0 + ctb).min(ph);
				let off = &sao.offset[c];
				if t == 1 {
					let shift = bd - 5;
					let mut table = [0i32; 32];
					for k in 0..4 {
						table[(k + sao.band_position[c] as usize) & 31] = i32::from(off[k + 1]);
					}
					let dst = &mut self.planes[c];
					let n = x1 - x0;
					let fast = if crate::simd::enabled() { n & !7 } else { 0 };
					let bo = [off[1], off[2], off[3], off[4]];
					for y in y0..y1 {
						let row = dst.idx(x0 as isize, y as isize);
						if fast > 0 {
							super::simd::sao_band(
								&mut dst.data[row..],
								fast,
								shift,
								sao.band_position[c],
								&bo,
								max,
							);
						}
						for v in &mut dst.data[row + fast..row + n] {
							let o = table[(*v >> shift) as usize];
							*v = (i32::from(*v) + o).clamp(0, max) as u16;
						}
					}
				} else {
					// Usability of the eight neighbouring CTBs.
					let mut ok = [[true; 3]; 3];
					for (dy, row) in ok.iter_mut().enumerate() {
						for (dx, v) in row.iter_mut().enumerate() {
							let (nx, ny) = (cx as isize + dx as isize - 1, cy as isize + dy as isize - 1);
							*v = !(nx < 0 || ny < 0 || nx >= cw as isize || ny >= ch as isize)
								&& (dx == 1 && dy == 1
									|| self.sao_neighbour_ok(rs, ny as usize * cw + nx as usize));
						}
					}
					let (hp, vp): ([isize; 2], [isize; 2]) = match sao.eo_class[c] {
						0 => ([-1, 1], [0, 0]),
						1 => ([0, 0], [-1, 1]),
						2 => ([-1, 1], [-1, 1]),
						_ => ([1, -1], [-1, 1]),
					};
					let s = src.stride as isize;
					let d0 = vp[0] * s + hp[0];
					let d1 = vp[1] * s + hp[1];
					let dst = &mut self.planes[c].data[boff..];
					// Offsets by raw edgeIdx 2 + sign + sign (0..=4).
					let offs = [off[1], off[2], 0, off[3], off[4]];
					let usable = |x: usize, y: usize| -> bool {
						(0..2).all(|k| {
							let nx = x as isize + hp[k];
							let ny = y as isize + vp[k];
							if nx < 0 || ny < 0 || nx >= pw as isize || ny >= ph as isize {
								return false;
							}
							let dx = if nx < x0 as isize {
								0
							} else if nx >= (x0 + ctb) as isize {
								2
							} else {
								1
							};
							let dy = if ny < y0 as isize {
								0
							} else if ny >= (y0 + ctb) as isize {
								2
							} else {
								1
							};
							ok[dy][dx]
						})
					};
					for y in y0..y1 {
						let base = src.idx(x0 as isize, y as isize);
						// Interior samples share the row's neighbour CTBs;
						// only the two ends can reach a CTB to the side.
						if x1 - x0 > 2 && usable(x0 + 1, y) {
							sao_edge_run(&src.data, dst, base + 1, x1 - x0 - 2, d0, d1, &offs, max);
						} else {
							for x in x0 + 1..x1.saturating_sub(1) {
								if usable(x, y) {
									sao_edge_run(&src.data, dst, base + x - x0, 1, d0, d1, &offs, max);
								}
							}
						}
						for x in [x0, x1 - 1] {
							if (x == x0 || x1 - x0 > 1) && usable(x, y) {
								sao_edge_run(&src.data, dst, base + x - x0, 1, d0, d1, &offs, max);
							}
						}
					}
				}
				if any_nf {
					// Restore PCM / bypass samples.
					let dst = &mut self.planes[c].data[boff..];
					for y in y0..y1 {
						for x in x0..x1 {
							let (lx, ly) = (x << sh, y << sh);
							if self.no_filter[(ly >> 2) * self.w4 + (lx >> 2)] != 0 {
								let at = src.idx(x as isize, y as isize);
								dst[at] = src.data[at];
							}
						}
					}
				}
			}
			self.sao_band[c] = src.data;
		}
	}
}

/// The deblocked rows `y0 - 1 ..= y0 + ctb` of a plane around the band
/// SAO is applied to, with the plane's stride and padding.
struct BandSrc {
	data: Vec<u16>,
	stride: usize,
	pad: usize,
	y0: usize,
}

impl BandSrc {
	#[inline]
	fn idx(&self, x: isize, y: isize) -> usize {
		((y + 1 - self.y0 as isize) as usize) * self.stride + (x + self.pad as isize) as usize
	}
}

/// Edge-offset SAO of `n` consecutive samples starting at `at`: `d0` and
/// `d1` are the offsets of the two neighbours, `offs` the offsets by raw
/// edge index `2 + sign(cur - a) + sign(cur - b)`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn sao_edge_run(
	src: &[u16],
	dst: &mut [u16],
	at: usize,
	n: usize,
	d0: isize,
	d1: isize,
	offs: &[i16; 5],
	max: i32,
) {
	let mut done = 0;
	if n >= 8 && crate::simd::enabled() {
		done = n & !7;
		super::simd::sao_edge(src, dst, at, done, d0, d1, offs, max);
	}
	for i in at + done..at + n {
		let cur = i32::from(src[i]);
		let a = i32::from(src[i.wrapping_add_signed(d0)]);
		let b = i32::from(src[i.wrapping_add_signed(d1)]);
		let e = (2 + (cur - a).signum() + (cur - b).signum()) as usize;
		dst[i] = (cur + i32::from(offs[e])).clamp(0, max) as u16;
	}
}

impl PicState {
	/// Whether SAO of CTB `cur` may read samples of CTB `n` (slice and tile
	/// boundary flags).
	fn sao_neighbour_ok(&self, cur: usize, n: usize) -> bool {
		let l = &*self.layout;
		let (sa, sb) = (self.ctb_addr[cur], self.ctb_addr[n]);
		if sb == u32::MAX {
			return false;
		}
		if sa != sb {
			let ta = l.ctb_addr_rs_to_ts[sa as usize];
			let tb = l.ctb_addr_rs_to_ts[sb as usize];
			let later = if tb < ta { cur } else { n };
			if !self.slices[self.ctb_slice[later] as usize].lf_across_slices {
				return false;
			}
		}
		if !self.pps.loop_filter_across_tiles
			&& l.tile_id[l.ctb_addr_rs_to_ts[cur] as usize] != l.tile_id[l.ctb_addr_rs_to_ts[n] as usize]
		{
			return false;
		}
		true
	}
}

/// Filters one 4-sample luma edge segment, twice (8 lines in total is the
/// caller's 2x4 split; here 4 lines).
#[allow(clippy::too_many_arguments)]
#[inline]
fn luma_edge(
	px: &mut [u16],
	at: usize,
	xs: usize,
	ys: usize,
	beta: i32,
	tc: i32,
	no_p: bool,
	no_q: bool,
	max: i32,
) -> u8 {
	let g = |px: &[u16], line: usize, k: isize| -> i32 {
		i32::from(px[(at + line * ys).wrapping_add_signed(k * xs as isize)])
	};
	let dp0 = (g(px, 0, -3) - 2 * g(px, 0, -2) + g(px, 0, -1)).abs();
	let dq0 = (g(px, 0, 2) - 2 * g(px, 0, 1) + g(px, 0, 0)).abs();
	let dp3 = (g(px, 3, -3) - 2 * g(px, 3, -2) + g(px, 3, -1)).abs();
	let dq3 = (g(px, 3, 2) - 2 * g(px, 3, 1) + g(px, 3, 0)).abs();
	let d0 = dp0 + dq0;
	let d3 = dp3 + dq3;
	if d0 + d3 >= beta || tc == 0 {
		return 0;
	}
	let b3 = beta >> 3;
	let b2 = beta >> 2;
	let tc25 = (tc * 5 + 1) >> 1;
	let strong_line = |px: &[u16], l: usize, d: i32| {
		(g(px, l, -4) - g(px, l, -1)).abs() + (g(px, l, 3) - g(px, l, 0)).abs() < b3
			&& (g(px, l, -1) - g(px, l, 0)).abs() < tc25
			&& (d << 1) < b2
	};
	if strong_line(px, 0, d0) && strong_line(px, 3, d3) {
		let tc2 = tc << 1;
		for l in 0..4 {
			let base = at + l * ys;
			let p = |k: usize| i32::from(px[base - (k + 1) * xs]);
			let q = |k: usize| i32::from(px[base + k * xs]);
			let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));
			let (q0, q1, q2, q3) = (q(0), q(1), q(2), q(3));
			if !no_p {
				px[base - xs] =
					((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - tc2, p0 + tc2) as u16;
				px[base - 2 * xs] = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - tc2, p1 + tc2) as u16;
				px[base - 3 * xs] =
					((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc2, p2 + tc2) as u16;
			}
			if !no_q {
				px[base] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - tc2, q0 + tc2) as u16;
				px[base + xs] = ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - tc2, q1 + tc2) as u16;
				px[base + 2 * xs] =
					((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc2, q2 + tc2) as u16;
			}
		}
		2
	} else {
		let side = (beta + (beta >> 1)) >> 3;
		let nd_p = dp0 + dp3 < side;
		let nd_q = dq0 + dq3 < side;
		let tc_2 = tc >> 1;
		for l in 0..4 {
			let base = at + l * ys;
			let p = |k: usize| i32::from(px[base - (k + 1) * xs]);
			let q = |k: usize| i32::from(px[base + k * xs]);
			let (p0, p1, p2) = (p(0), p(1), p(2));
			let (q0, q1, q2) = (q(0), q(1), q(2));
			let mut delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
			if delta.abs() >= tc * 10 {
				continue;
			}
			delta = delta.clamp(-tc, tc);
			if !no_p {
				px[base - xs] = (p0 + delta).clamp(0, max) as u16;
				if nd_p {
					let dp = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-tc_2, tc_2);
					px[base - 2 * xs] = (p1 + dp).clamp(0, max) as u16;
				}
			}
			if !no_q {
				px[base] = (q0 - delta).clamp(0, max) as u16;
				if nd_q {
					let dq = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-tc_2, tc_2);
					px[base + xs] = (q1 + dq).clamp(0, max) as u16;
				}
			}
		}
		1
	}
}

/// [`luma_edge`] for tests elsewhere in the decoder: returns 2 for the
/// strong filter, 1 for the weak one, 0 for none.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_edge_probe(
	px: &mut [u16],
	at: usize,
	xs: usize,
	ys: usize,
	beta: i32,
	tc: i32,
	no_p: bool,
	no_q: bool,
	max: i32,
) -> u8 {
	luma_edge(px, at, xs, ys, beta, tc, no_p, no_q, max)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn chroma_edge(
	px: &mut [u16],
	at: usize,
	xs: usize,
	ys: usize,
	tc: i32,
	no_p: bool,
	no_q: bool,
	max: i32,
) {
	for l in 0..4 {
		let base = at + l * ys;
		let p0 = i32::from(px[base - xs]);
		let p1 = i32::from(px[base - 2 * xs]);
		let q0 = i32::from(px[base]);
		let q1 = i32::from(px[base + xs]);
		let delta = ((((q0 - p0) * 4) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
		if !no_p {
			px[base - xs] = (p0 + delta).clamp(0, max) as u16;
		}
		if !no_q {
			px[base] = (q0 - delta).clamp(0, max) as u16;
		}
	}
}

#[allow(dead_code)]
fn _plane_unused(_: &Plane) {}

impl PicState {
	/// Runs the in-loop filters as far as the decoded rows allow, and
	/// publishes the rows that became final (FFmpeg filters and reports
	/// progress the same way, a CTB row behind decoding).
	///
	/// * CTB row `r` is deblocked once row `r + 1` is decoded: intra
	///   prediction in row `r + 1` reads row `r`'s last line unfiltered.
	/// * SAO of row `r` runs once row `r + 1` is deblocked: it reads one
	///   deblocked row below the band.
	///
	/// `finished`: the picture has no more slices; filter everything.
	pub fn advance(&mut self, finished: bool) {
		let (cw, ch) = (self.sps.ctb_width as usize, self.sps.ctb_height as usize);
		while self.rows_complete < ch && self.row_ctbs[self.rows_complete] as usize == cw {
			self.rows_complete += 1;
		}
		let log2 = self.sps.log2_ctb_size;
		let h = self.sps.height as usize;
		let deblock_to = if finished { ch } else { self.rows_complete.saturating_sub(1) };
		while self.rows_deblocked < deblock_to {
			let r = self.rows_deblocked;
			self.deblock_rows(r << log2, ((r + 1) << log2).min(h));
			self.rows_deblocked += 1;
		}
		let final_to = if finished { ch } else { self.rows_deblocked.saturating_sub(1) };
		let prev = self.rows_final;
		while self.rows_final < final_to {
			self.sao_row(self.rows_final);
			self.rows_final += 1;
		}
		if self.rows_final > prev {
			let last = self.rows_final == ch;
			for (c, p) in self.planes.iter_mut().enumerate() {
				let sh = u32::from(c > 0);
				p.extend_rows((prev << log2) >> sh, (self.rows_final << log2) >> sh);
				if prev == 0 {
					p.extend_top();
				}
				if last {
					p.extend_bottom();
				}
			}
		}
		if !finished && (self.rows_final > prev || self.rows_complete > self.motion_published) {
			self.publish(prev);
		}
	}

	/// Copies newly final rows (from CTB row `prev`) and newly final motion
	/// into the shared picture, and reports the progress.
	fn publish(&mut self, prev: usize) {
		let Some(out) = self.out.clone() else { return };
		let log2 = self.sps.log2_ctb_size;
		let h = self.sps.height as usize;
		let cw = self.sps.ctb_width as usize;
		let rows = ((self.rows_final << log2).min(h)) as u32;
		let (m0, m1) = (self.motion_published, self.rows_complete);
		let final_now = self.rows_final;
		out.publish(rows, m1 as u32, |live| {
			if final_now > prev {
				for c in 0..3 {
					let (src, dst) = (&self.planes[c], &mut live.planes[c]);
					let sh = u32::from(c > 0);
					let y0 = if prev == 0 { -(src.pad as isize) } else { ((prev << log2) >> sh) as isize };
					let y1 = ((final_now << log2) >> sh).min(src.height) as isize;
					let r = src.rows_range(y0, y1);
					dst.data[r.clone()].copy_from_slice(&src.data[r]);
				}
			}
			if m1 > m0 {
				let w4 = self.w4;
				let (a, b) = ((m0 << log2) >> 2, ((m1 << log2) >> 2).min(self.h4));
				live.mvf[a * w4..b * w4].copy_from_slice(&self.mvf[a * w4..b * w4]);
				live.ctb_slice[m0 * cw..m1 * cw].copy_from_slice(&self.ctb_slice[m0 * cw..m1 * cw]);
				if live.slice_refs.len() != self.slices.len() {
					live.slice_refs = self.slices.iter().map(|s| s.refs.clone()).collect();
				}
			}
		});
		self.motion_published = m1;
	}
}
