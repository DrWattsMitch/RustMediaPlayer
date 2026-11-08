//! Deblocking filter for progressive frames (ISO/IEC 14496-10 §8.7;
//! FFmpeg `h264_loopfilter.c`, `h264dsp_template.c`).

use super::consts::{ALPHA, BETA, TC0};
use super::decode::{Frame, SliceFilter};
use super::mb::MbInfo;
use super::recon::Plane;

/// `true` if the luma 4x4 block (raster `idx`) of `m` has coefficients.
fn has_coeffs(m: &MbInfo, idx: usize) -> bool {
	if m.t8x8 {
		let (bx, by) = ((idx % 4) / 2 * 2, idx / 8 * 2);
		[0, 1, 4, 5].iter().any(|&o| m.nnz[by * 4 + bx + o] != 0)
	} else {
		m.nnz[idx] != 0
	}
}

#[inline]
fn b8_of(idx: usize) -> usize {
	(idx / 8) * 2 + (idx % 4) / 2
}

fn mv_differs(a: [i16; 2], b: [i16; 2]) -> bool {
	(i32::from(a[0]) - i32::from(b[0])).abs() >= 4 || (i32::from(a[1]) - i32::from(b[1])).abs() >= 4
}

/// Boundary strength from motion (bS 1 or 0), 8.7.2.1.
fn bs_motion(p: &MbInfo, pi: usize, q: &MbInfo, qi: usize) -> u8 {
	let (pb, qb) = (b8_of(pi), b8_of(qi));
	let pr = [p.ref_id[0][pb], p.ref_id[1][pb]];
	let qr = [q.ref_id[0][qb], q.ref_id[1][qb]];
	let pn = pr.iter().filter(|&&r| r != 0).count();
	let qn = qr.iter().filter(|&&r| r != 0).count();
	if pn != qn {
		return 1;
	}
	let pm = [p.mv[0][pi], p.mv[1][pi]];
	let qm = [q.mv[0][qi], q.mv[1][qi]];
	if pn == 1 {
		let (pl, ql) = (usize::from(pr[0] == 0), usize::from(qr[0] == 0));
		if pr[pl] != qr[ql] {
			return 1;
		}
		return u8::from(mv_differs(pm[pl], qm[ql]));
	}
	// Two motion vectors each.
	let same_set = (pr[0] == qr[0] && pr[1] == qr[1]) || (pr[0] == qr[1] && pr[1] == qr[0]);
	if !same_set {
		return 1;
	}
	if pr[0] != pr[1] {
		if pr[0] == qr[0] {
			u8::from(mv_differs(pm[0], qm[0]) || mv_differs(pm[1], qm[1]))
		} else {
			u8::from(mv_differs(pm[0], qm[1]) || mv_differs(pm[1], qm[0]))
		}
	} else {
		let straight = mv_differs(pm[0], qm[0]) || mv_differs(pm[1], qm[1]);
		let crossed = mv_differs(pm[0], qm[1]) || mv_differs(pm[1], qm[0]);
		u8::from(straight && crossed)
	}
}

fn boundary_strength(p: &MbInfo, pi: usize, q: &MbInfo, qi: usize, mb_edge: bool) -> u8 {
	if p.is_intra() || q.is_intra() {
		return if mb_edge { 4 } else { 3 };
	}
	if has_coeffs(p, pi) || has_coeffs(q, qi) {
		return 2;
	}
	bs_motion(p, pi, q, qi)
}

/// Filters one line of samples across an edge. `pix(k)` addresses sample
/// p_k for k < 0 (p0 = -1) and q_k for k >= 0 along the line.
#[inline]
#[allow(clippy::too_many_arguments)]
fn filter_line(
	data: &mut [u8],
	base: usize,
	step: isize,
	bs: u8,
	alpha: i32,
	beta: i32,
	tc0: i32,
	chroma: bool,
) {
	let at = |k: isize| (base as isize + k * step) as usize;
	let p0 = i32::from(data[at(-1)]);
	let q0 = i32::from(data[at(0)]);
	let p1 = i32::from(data[at(-2)]);
	let q1 = i32::from(data[at(1)]);
	if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
		return;
	}
	let clip1 = |v: i32| v.clamp(0, 255) as u8;
	if chroma {
		if bs < 4 {
			let tc = tc0 + 1;
			let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
			data[at(-1)] = clip1(p0 + delta);
			data[at(0)] = clip1(q0 - delta);
		} else {
			data[at(-1)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
			data[at(0)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
		}
		return;
	}
	let p2 = i32::from(data[at(-3)]);
	let q2 = i32::from(data[at(2)]);
	let ap = (p2 - p0).abs();
	let aq = (q2 - q0).abs();
	if bs < 4 {
		let tc = tc0 + i32::from(ap < beta) + i32::from(aq < beta);
		let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
		data[at(-1)] = clip1(p0 + delta);
		data[at(0)] = clip1(q0 - delta);
		if ap < beta {
			data[at(-2)] = (p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0)) as u8;
		}
		if aq < beta {
			data[at(1)] = (q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0)) as u8;
		}
	} else {
		let p3 = i32::from(data[at(-4)]);
		let q3 = i32::from(data[at(3)]);
		let strong = (p0 - q0).abs() < (alpha >> 2) + 2;
		if ap < beta && strong {
			data[at(-1)] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
			data[at(-2)] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
			data[at(-3)] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
		} else {
			data[at(-1)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
		}
		if aq < beta && strong {
			data[at(0)] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u8;
			data[at(1)] = ((p0 + q0 + q1 + q2 + 2) >> 2) as u8;
			data[at(2)] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
		} else {
			data[at(0)] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
		}
	}
}

/// Filters a whole edge of one plane. `bs[i]` is the strength of segment
/// `i` (4 luma rows/columns, or 2 chroma ones). `vertical` = edge between
/// left and right samples.
#[allow(clippy::too_many_arguments)]
fn filter_edge(
	plane: &mut Plane,
	x: usize,
	y: usize,
	len: usize,
	vertical: bool,
	bs: &[u8; 4],
	qp: i32,
	alpha_off: i32,
	beta_off: i32,
	chroma: bool,
) {
	let index_a = (qp + alpha_off).clamp(0, 51) as usize;
	let index_b = (qp + beta_off).clamp(0, 51) as usize;
	let alpha = i32::from(ALPHA[index_a]);
	let beta = i32::from(BETA[index_b]);
	if alpha == 0 || beta == 0 {
		return;
	}
	let w = plane.width;
	let seg = len / 4;
	for i in 0..len {
		let s = bs[i / seg];
		if s == 0 {
			continue;
		}
		let tc0 = if s < 4 { i32::from(TC0[index_a][usize::from(s) - 1]) } else { 0 };
		let (base, step) =
			if vertical { ((y + i) * w + x, 1isize) } else { (y * w + x + i, w as isize) };
		filter_line(&mut plane.data, base, step, s, alpha, beta, tc0, chroma);
	}
}

/// Deblocks the whole frame in macroblock raster order.
pub fn deblock_frame(f: &mut Frame) {
	let Frame { y, cb, cr, mbs, mb_w, mb_h, slices } = f;
	let mut planes = Planes { y, cb, cr };
	for mby in 0..*mb_h {
		for mbx in 0..*mb_w {
			deblock_mb(mbs, slices, *mb_w, &mut planes, mbx, mby);
		}
	}
}

struct Planes<'a> {
	y: &'a mut Plane,
	cb: &'a mut Plane,
	cr: &'a mut Plane,
}

fn deblock_mb(
	mbs: &[MbInfo],
	slices: &[SliceFilter],
	mb_w: usize,
	planes: &mut Planes,
	mbx: usize,
	mby: usize,
) {
	let addr = mby * mb_w + mbx;
	let q = &mbs[addr];
	if q.slice == 0 {
		return;
	}
	let Some(&params) = slices.get(q.slice as usize - 1) else {
		return;
	};
	if params.disable == 1 {
		return;
	}
	let edge_ok = |p: &&MbInfo| p.slice != 0 && (params.disable != 2 || p.slice == q.slice);
	let left = (mbx > 0).then(|| &mbs[addr - 1]).filter(edge_ok);
	let top = (mby > 0).then(|| &mbs[addr - mb_w]).filter(edge_ok);
	let (px, py) = (mbx * 16, mby * 16);

	for vertical in [true, false] {
		let neighbour = if vertical { left } else { top };
		let mut luma_bs = [[0u8; 4]; 4];
		for e in 0..4 {
			let p = match (e, neighbour) {
				(0, None) => continue,
				(0, Some(n)) => n,
				_ => q,
			};
			if e > 0 && q.t8x8 && e != 2 {
				continue;
			}
			let mut bs = [0u8; 4];
			for (i, s) in bs.iter_mut().enumerate() {
				let (qi, pi) = if vertical {
					(i * 4 + e, i * 4 + (e + 3) % 4)
				} else {
					(e * 4 + i, ((e + 3) % 4) * 4 + i)
				};
				*s = boundary_strength(p, pi, q, qi, e == 0);
			}
			luma_bs[e] = bs;
			if bs.iter().all(|&s| s == 0) {
				continue;
			}
			let qp = (i32::from(p.qp) + i32::from(q.qp) + 1) >> 1;
			let (x, y) = if vertical { (px + e * 4, py) } else { (px, py + e * 4) };
			filter_edge(planes.y, x, y, 16, vertical, &bs, qp, params.alpha, params.beta, false);
		}
		// Chroma edges 0 and 4 use the luma strengths of luma edges 0 and 2.
		for (ce, le) in [(0usize, 0usize), (1, 2)] {
			let p = match (ce, neighbour) {
				(0, None) => continue,
				(0, Some(n)) => n,
				_ => q,
			};
			let bs = luma_bs[le];
			if bs.iter().all(|&s| s == 0) {
				continue;
			}
			for c in 0..2 {
				let qp = (i32::from(p.qpc[c]) + i32::from(q.qpc[c]) + 1) >> 1;
				let (x, y) = if vertical { (px / 2 + ce * 4, py / 2) } else { (px / 2, py / 2 + ce * 4) };
				let plane = if c == 0 { &mut *planes.cb } else { &mut *planes.cr };
				filter_edge(plane, x, y, 8, vertical, &bs, qp, params.alpha, params.beta, true);
			}
		}
	}
}
