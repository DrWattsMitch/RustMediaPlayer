//! Deblocking filter for progressive frames (ISO/IEC 14496-10 §8.7;
//! FFmpeg `h264_loopfilter.c`, `h264dsp_template.c`).

use super::consts::{ALPHA, BETA, TC0};
use super::decode::{Frame, SliceFilter};
use super::mb::MbInfo;
use super::recon::{Dst, Pixel, Plane};

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

/// Motion vectors differ by a quarter-pel distance of 4 (vertically 4 in
/// frame units: 2 between field macroblocks, `mvy_limit`).
#[inline]
fn mv_differs(a: [i16; 2], b: [i16; 2], mvy_limit: i32) -> bool {
	(i32::from(a[0]) - i32::from(b[0])).abs() >= 4
		|| (i32::from(a[1]) - i32::from(b[1])).abs() >= mvy_limit
}

/// Boundary strength from motion (bS 1 or 0), 8.7.2.1.
fn bs_motion(p: &MbInfo, pi: usize, q: &MbInfo, qi: usize, mvy_limit: i32) -> u8 {
	let mv_differs = |a, b| mv_differs(a, b, mvy_limit);
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

/// bS of the edge between 4x4 blocks `pi` of `p` and `qi` of `q`
/// (8.7.2.1). Intra gives 4 on macroblock edges, except horizontal ones
/// next to a field macroblock; an MBAFF edge between a frame and a field
/// macroblock (mixedModeEdgeFlag) is at least 1 and compares no motion.
fn boundary_strength(
	p: &MbInfo,
	pi: usize,
	q: &MbInfo,
	qi: usize,
	mb_edge: bool,
	vertical: bool,
) -> u8 {
	if p.is_intra() || q.is_intra() {
		return if mb_edge && (vertical || (!p.field && !q.field)) { 4 } else { 3 };
	}
	if has_coeffs(p, pi) || has_coeffs(q, qi) {
		return 2;
	}
	if p.field != q.field {
		return 1;
	}
	bs_motion(p, pi, q, qi, if q.field { 2 } else { 4 })
}

/// Filters one line of samples across an edge. `at(k)` addresses sample
/// p_k for k < 0 (p0 = -1) and q_k for k >= 0 along the line.
#[inline]
#[allow(clippy::too_many_arguments)]
fn filter_line<T: Pixel>(
	data: &mut [T],
	base: usize,
	step: isize,
	bs: u8,
	alpha: i32,
	beta: i32,
	tc0: i32,
	chroma: bool,
	max: i32,
) {
	let at = |k: isize| (base as isize + k * step) as usize;
	let px = |v: T| -> i32 { v.into() };
	let p0 = px(data[at(-1)]);
	let q0 = px(data[at(0)]);
	let p1 = px(data[at(-2)]);
	let q1 = px(data[at(1)]);
	if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
		return;
	}
	let clip1 = |v: i32| T::clip(v, max);
	if chroma {
		if bs < 4 {
			let tc = tc0 + 1;
			let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
			data[at(-1)] = clip1(p0 + delta);
			data[at(0)] = clip1(q0 - delta);
		} else {
			data[at(-1)] = T::of((2 * p1 + p0 + q1 + 2) >> 2);
			data[at(0)] = T::of((2 * q1 + q0 + p1 + 2) >> 2);
		}
		return;
	}
	let p2 = px(data[at(-3)]);
	let q2 = px(data[at(2)]);
	let ap = (p2 - p0).abs();
	let aq = (q2 - q0).abs();
	if bs < 4 {
		let tc = tc0 + i32::from(ap < beta) + i32::from(aq < beta);
		let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
		data[at(-1)] = clip1(p0 + delta);
		data[at(0)] = clip1(q0 - delta);
		if ap < beta {
			data[at(-2)] = T::of(p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0));
		}
		if aq < beta {
			data[at(1)] = T::of(q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0));
		}
	} else {
		let p3 = px(data[at(-4)]);
		let q3 = px(data[at(3)]);
		let strong = (p0 - q0).abs() < (alpha >> 2) + 2;
		if ap < beta && strong {
			data[at(-1)] = T::of((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
			data[at(-2)] = T::of((p2 + p1 + p0 + q0 + 2) >> 2);
			data[at(-3)] = T::of((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
		} else {
			data[at(-1)] = T::of((2 * p1 + p0 + q1 + 2) >> 2);
		}
		if aq < beta && strong {
			data[at(0)] = T::of((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
			data[at(1)] = T::of((p0 + q0 + q1 + q2 + 2) >> 2);
			data[at(2)] = T::of((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
		} else {
			data[at(0)] = T::of((2 * q1 + q0 + p1 + 2) >> 2);
		}
	}
}

/// Filters a whole edge of one plane. `bs[i]` is the strength of segment
/// `i` (`len / 4` lines each). `vertical` = edge between left and right
/// samples. `qp` is the average QP (without the bit-depth offset); the
/// thresholds are scaled to the view's bit depth (8.7.2.2).
#[allow(clippy::too_many_arguments)]
pub(super) fn filter_edge<T: Pixel>(
	plane: &mut Dst<T>,
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
	// (1 << (bit_depth - 8)), from the view's largest sample.
	let scale = (plane.max + 1) >> 8;
	let alpha = i32::from(ALPHA[index_a]) * scale;
	let beta = i32::from(BETA[index_b]) * scale;
	if alpha == 0 || beta == 0 {
		return;
	}
	let mut tc0 = [0i32; 4];
	for (t, &s) in tc0.iter_mut().zip(bs) {
		if (1..4).contains(&s) {
			*t = i32::from(TC0[index_a][usize::from(s) - 1]) * scale;
		}
	}
	if crate::simd::enabled() && len.is_multiple_of(8) {
		if let Some(mut d) = T::narrow_dst(plane) {
			return super::simd::filter_edge(&mut d, x, y, len, vertical, bs, &tc0, alpha, beta, chroma);
		}
	}
	filter_edge_scalar(plane, x, y, len, vertical, bs, &tc0, alpha, beta, chroma);
}

/// Scalar edge filter: the fallback and the reference for
/// [`super::simd::filter_edge`].
#[allow(clippy::too_many_arguments)]
pub(super) fn filter_edge_scalar<T: Pixel>(
	plane: &mut Dst<T>,
	x: usize,
	y: usize,
	len: usize,
	vertical: bool,
	bs: &[u8; 4],
	tc0: &[i32; 4],
	alpha: i32,
	beta: i32,
	chroma: bool,
) {
	let seg = (len / 4).max(1);
	let max = plane.max;
	for i in 0..len {
		let s = bs[(i / seg).min(3)];
		if s == 0 {
			continue;
		}
		let (base, step) = if vertical {
			(plane.idx(x, y + i), 1isize)
		} else {
			(plane.idx(x + i, y), plane.stride as isize)
		};
		filter_line(plane.data, base, step, s, alpha, beta, tc0[(i / seg).min(3)], chroma, max);
	}
}

/// Deblocks the whole frame in macroblock raster order.
pub fn deblock_frame<T: Pixel>(f: &mut Frame<T>) {
	let rows = 0..f.mb_h;
	deblock_rows(f, rows);
}

/// The views a macroblock is filtered in, and its origin in them.
struct Planes<'a, T> {
	y: Dst<'a, T>,
	cb: Dst<'a, T>,
	cr: Dst<'a, T>,
	c422: bool,
	/// Luma origin; chroma origin.
	px: usize,
	py: usize,
	cx: usize,
	cy: usize,
}

impl<'a, T: Pixel> Planes<'a, T> {
	#[allow(clippy::too_many_arguments)]
	fn new(
		y: &'a mut Plane<T>,
		cb: &'a mut Plane<T>,
		cr: &'a mut Plane<T>,
		field: Option<usize>,
		max: i32,
		c422: bool,
		mbx: usize,
		row: usize,
	) -> Self {
		let ch = if c422 { 16 } else { 8 };
		Self {
			y: y.dst(field, max),
			cb: cb.dst(field, max),
			cr: cr.dst(field, max),
			c422,
			px: mbx * 16,
			py: row * 16,
			cx: mbx * 8,
			cy: row * ch,
		}
	}
}

/// Deblocks the macroblock rows in `rows` (in raster order, as
/// [`deblock_frame`] does; rows must be deblocked top to bottom).
pub fn deblock_rows<T: Pixel>(f: &mut Frame<T>, rows: std::ops::Range<usize>) {
	let max = f.max();
	let c422 = f.chroma_422;
	let Frame { y, cb, cr, mbs, mb_w, mb_h, slices, .. } = f;
	let w = *mb_w;
	let mut planes = Planes::new(y, cb, cr, None, max, c422, 0, 0);
	let ch = if c422 { 16 } else { 8 };
	for mby in rows.start..rows.end.min(*mb_h) {
		for mbx in 0..w {
			let addr = mby * w + mbx;
			let Some((q, params)) = filtered(mbs, slices, addr) else { continue };
			(planes.px, planes.py, planes.cx, planes.cy) = (mbx * 16, mby * 16, mbx * 8, mby * ch);
			let left = (mbx > 0).then(|| &mbs[addr - 1]).filter(|p| edge_ok(p, q, params));
			let top = (mby > 0).then(|| &mbs[addr - w]).filter(|p| edge_ok(p, q, params));
			deblock_dir(&mut planes, q, left, true, params);
			deblock_dir(&mut planes, q, top, false, params);
		}
	}
}

/// Deblocks field `parity` of a frame decoded as two field pictures (in
/// the field: macroblock rows of that parity, every other sample row).
pub fn deblock_field<T: Pixel>(f: &mut Frame<T>, parity: usize) {
	let max = f.max();
	let c422 = f.chroma_422;
	let Frame { y, cb, cr, mbs, mb_w, mb_h, slices, .. } = f;
	let w = *mb_w;
	let mut planes = Planes::new(y, cb, cr, Some(parity), max, c422, 0, 0);
	let ch = if c422 { 16 } else { 8 };
	let store = |x: usize, r: usize| (2 * r + parity) * w + x;
	for row in 0..*mb_h / 2 {
		for mbx in 0..w {
			let addr = store(mbx, row);
			let Some((q, params)) = filtered(mbs, slices, addr) else { continue };
			(planes.px, planes.py, planes.cx, planes.cy) = (mbx * 16, row * 16, mbx * 8, row * ch);
			let left = (mbx > 0).then(|| &mbs[addr - 1]).filter(|p| edge_ok(p, q, params));
			let top = (row > 0).then(|| &mbs[store(mbx, row - 1)]).filter(|p| edge_ok(p, q, params));
			deblock_dir(&mut planes, q, left, true, params);
			deblock_dir(&mut planes, q, top, false, params);
		}
	}
}

/// Deblocks an MBAFF frame (8.7 with MbaffFrameFlag) in macroblock address
/// order: each pair's top macroblock, then its bottom one. Frame
/// macroblocks are filtered in the frame, field macroblocks in their field;
/// edges between a frame and a field macroblock follow FFmpeg's
/// `ff_h264_filter_mb`.
///
/// `pair_rows` are deblocked top to bottom, like [`deblock_rows`].
pub fn deblock_mbaff<T: Pixel>(f: &mut Frame<T>, pair_rows: std::ops::Range<usize>) {
	let max = f.max();
	let c422 = f.chroma_422;
	let ch = if c422 { 16 } else { 8 };
	let Frame { y, cb, cr, mbs, mb_w, mb_h, slices, .. } = f;
	let w = *mb_w;
	for pair_row in pair_rows.start..pair_rows.end.min(*mb_h / 2) {
		for mbx in 0..w {
			for bottom in 0..2 {
				let row = 2 * pair_row + bottom;
				let addr = row * w + mbx;
				let Some((q, params)) = filtered(mbs, slices, addr) else { continue };
				let field = q.field.then_some(bottom);
				let view_row = if q.field { pair_row } else { row };
				// Left: the pair beside, whose kind may differ.
				let (la, lb) = (
					&mbs[2 * pair_row * w + mbx - usize::from(mbx > 0)],
					&mbs[(2 * pair_row + 1) * w + mbx - usize::from(mbx > 0)],
				);
				let left_ok = mbx > 0 && edge_ok(la, q, params);
				let mixed_left = left_ok && la.field != q.field;
				if mixed_left {
					let mut planes = Planes::new(y, cb, cr, field, max, c422, mbx, view_row);
					filter_mixed_left(&mut planes, q, [la, lb], bottom, params);
				}
				// Above (Table 6-4): a frame macroblock sees the macroblock
				// right above it (in its pair for the bottom one); a field
				// macroblock the same-parity one of the pair above, or its
				// bottom frame macroblock.
				let above = match (q.field, bottom) {
					(false, 1) => Some(addr - w),
					(_, _) if pair_row == 0 => None,
					(false, _) => Some(addr - w),
					(true, 1) => Some(addr - 2 * w),
					(true, _) => {
						let top_of_above = (2 * pair_row - 2) * w + mbx;
						Some(if mbs[top_of_above].field { top_of_above } else { top_of_above + w })
					}
				};
				let top = above.map(|i| &mbs[i]).filter(|p| edge_ok(p, q, params));
				// A frame macroblock under a field pair filters its top edge
				// twice, once per field.
				let special_top = !q.field && bottom == 0 && top.is_some_and(|p| p.field);
				let mut planes = Planes::new(y, cb, cr, field, max, c422, mbx, view_row);
				planes.cy = view_row * ch;
				let left = if left_ok && !mixed_left { Some(&mbs[addr - 1]) } else { None };
				deblock_dir(&mut planes, q, left, true, params);
				if special_top {
					for parity in 0..2 {
						let p = &mbs[(2 * pair_row - 2 + parity) * w + mbx];
						let mut fp = Planes::new(y, cb, cr, Some(parity), max, c422, mbx, pair_row);
						filter_field_top(&mut fp, q, p, params);
					}
					let mut planes = Planes::new(y, cb, cr, field, max, c422, mbx, view_row);
					deblock_dir(&mut planes, q, None, false, params);
				} else {
					deblock_dir(&mut planes, q, top, false, params);
				}
			}
		}
	}
}

/// The macroblock at `addr` and its slice's filter parameters, if it is
/// filtered at all.
fn filtered<'m>(
	mbs: &'m [MbInfo],
	slices: &[SliceFilter],
	addr: usize,
) -> Option<(&'m MbInfo, SliceFilter)> {
	let q = &mbs[addr];
	if q.slice == 0 {
		return None;
	}
	let params = *slices.get(q.slice as usize - 1)?;
	(params.disable != 1).then_some((q, params))
}

/// Whether the edge between `p` and `q` is filtered
/// (disable_deblocking_filter_idc 2 stops at slice edges).
fn edge_ok(p: &MbInfo, q: &MbInfo, params: SliceFilter) -> bool {
	p.slice != 0 && (params.disable != 2 || p.slice == q.slice)
}

/// Filters the luma and chroma edges of `q` in one direction: the
/// macroblock edge (when `neighbour` is given) and the internal ones.
fn deblock_dir<T: Pixel>(
	planes: &mut Planes<T>,
	q: &MbInfo,
	neighbour: Option<&MbInfo>,
	vertical: bool,
	params: SliceFilter,
) {
	let (px, py, cx, cy) = (planes.px, planes.py, planes.cx, planes.cy);
	let c422 = planes.c422;
	let ch = if c422 { 16 } else { 8 };
	let mut luma_bs = [[0u8; 4]; 4];
	for e in 0..4 {
		let p = match (e, neighbour) {
			(0, None) => continue,
			(0, Some(n)) => n,
			_ => q,
		};
		// Luma edges inside an 8x8 transform are not filtered; 4:2:2
		// chroma still filters its horizontal edges at rows 4 and 12
		// with those strengths.
		let luma_edge = e == 0 || !q.t8x8 || e == 2;
		if !luma_edge && (vertical || !c422) {
			continue;
		}
		let mut bs = [0u8; 4];
		for (i, s) in bs.iter_mut().enumerate() {
			let (qi, pi) = if vertical {
				(i * 4 + e, i * 4 + (e + 3) % 4)
			} else {
				(e * 4 + i, ((e + 3) % 4) * 4 + i)
			};
			*s = boundary_strength(p, pi, q, qi, e == 0, vertical);
		}
		luma_bs[e] = bs;
		if !luma_edge || bs.iter().all(|&s| s == 0) {
			continue;
		}
		let qp = (i32::from(p.qp) + i32::from(q.qp) + 1) >> 1;
		let (x, y) = if vertical { (px + e * 4, py) } else { (px, py + e * 4) };
		filter_edge(&mut planes.y, x, y, 16, vertical, &bs, qp, params.alpha, params.beta, false);
	}
	// Chroma edges: every other luma edge across 8 chroma samples, and
	// for 4:2:2 every horizontal luma edge (16 chroma rows).
	let chroma_edges: &[(usize, usize)] =
		if c422 && !vertical { &[(0, 0), (1, 1), (2, 2), (3, 3)] } else { &[(0, 0), (1, 2)] };
	for &(ce, le) in chroma_edges {
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
			let (x, y, len) = if vertical { (cx + ce * 4, cy, ch) } else { (cx, cy + ce * 4, 8) };
			let plane = if c == 0 { &mut planes.cb } else { &mut planes.cr };
			filter_edge(plane, x, y, len, vertical, &bs, qp, params.alpha, params.beta, true);
		}
	}
}

/// Filters single lines across a vertical edge at column `x`, rows
/// `y0..y0 + rows` of `plane`, each with its own strength and QP.
#[allow(clippy::too_many_arguments)]
fn filter_rows<T: Pixel>(
	plane: &mut Dst<T>,
	x: usize,
	y0: usize,
	rows: usize,
	chroma: bool,
	params: SliceFilter,
	line: impl Fn(usize) -> (u8, i32),
) {
	let scale = (plane.max + 1) >> 8;
	let max = plane.max;
	for k in 0..rows {
		let (bs, qp) = line(k);
		if bs == 0 {
			continue;
		}
		let index_a = (qp + params.alpha).clamp(0, 51) as usize;
		let index_b = (qp + params.beta).clamp(0, 51) as usize;
		let alpha = i32::from(ALPHA[index_a]) * scale;
		let beta = i32::from(BETA[index_b]) * scale;
		if alpha == 0 || beta == 0 {
			continue;
		}
		let tc0 = if bs < 4 { i32::from(TC0[index_a][usize::from(bs) - 1]) * scale } else { 0 };
		let base = plane.idx(x, y0 + k);
		filter_line(plane.data, base, 1, bs, alpha, beta, tc0, chroma, max);
	}
}

/// The left edge of an MBAFF macroblock whose left pair `[top, bottom]`
/// is of the other kind (frame vs field): every row has the neighbour that
/// physically holds it, its own strength (4 for intra, else 1 or 2) and
/// QP. Chroma rows take the strength of FFmpeg's corresponding luma rows
/// (`filter_mb_mbaff_edgecv`).
fn filter_mixed_left<T: Pixel>(
	planes: &mut Planes<T>,
	q: &MbInfo,
	left: [&MbInfo; 2],
	bottom: usize,
	params: SliceFilter,
) {
	let c422 = planes.c422;
	let ch = if c422 { 16 } else { 8 };
	// For luma row k of q: the left macroblock and its 4x4 block.
	let luma = |k: usize| -> (&MbInfo, usize) {
		if q.field {
			// Field row k is pair row 2k + bottom of the frame pair beside.
			let r = 2 * k + bottom;
			(left[r / 16], ((r % 16) / 4) * 4 + 3)
		} else {
			// Frame row k is pair row 16 * bottom + k of the field pair.
			let r = 16 * bottom + k;
			(left[r % 2], ((r / 2) / 4) * 4 + 3)
		}
	};
	let bs = |k: usize| -> u8 {
		let (p, pi) = luma(k);
		let qi = (k / 4) * 4;
		if q.is_intra() || p.is_intra() {
			4
		} else {
			1 + u8::from(has_coeffs(q, qi) || has_coeffs(p, pi))
		}
	};
	let (px, py, cx, cy) = (planes.px, planes.py, planes.cx, planes.cy);
	filter_rows(&mut planes.y, px, py, 16, false, params, |k| {
		let p = luma(k).0;
		(bs(k), (i32::from(p.qp) + i32::from(q.qp) + 1) >> 1)
	});
	// The luma row whose strength and neighbour a chroma row uses.
	let chroma_row = |c: usize| -> usize {
		match (c422, q.field) {
			(true, _) => c,
			(false, true) => 2 * c,
			(false, false) => 4 * (c / 2) + c % 2,
		}
	};
	for comp in 0..2 {
		let plane = if comp == 0 { &mut planes.cb } else { &mut planes.cr };
		filter_rows(plane, cx, cy, ch, true, params, |c| {
			let k = chroma_row(c);
			let p = luma(k).0;
			(bs(k), (i32::from(p.qpc[comp]) + i32::from(q.qpc[comp]) + 1) >> 1)
		});
	}
}

/// One field of the top edge of a frame macroblock (the top one of its
/// pair) under a field pair: the field rows of `q` against field
/// macroblock `p` of that parity, strength 3 for intra, else 1 or 2.
fn filter_field_top<T: Pixel>(planes: &mut Planes<T>, q: &MbInfo, p: &MbInfo, params: SliceFilter) {
	let mut bs = [0u8; 4];
	for (i, s) in bs.iter_mut().enumerate() {
		*s = if q.is_intra() || p.is_intra() {
			3
		} else {
			1 + u8::from(has_coeffs(q, i) || has_coeffs(p, 12 + i))
		};
	}
	let (px, py, cx, cy) = (planes.px, planes.py, planes.cx, planes.cy);
	let qp = (i32::from(p.qp) + i32::from(q.qp) + 1) >> 1;
	filter_edge(&mut planes.y, px, py, 16, false, &bs, qp, params.alpha, params.beta, false);
	for c in 0..2 {
		let qpc = (i32::from(p.qpc[c]) + i32::from(q.qpc[c]) + 1) >> 1;
		let plane = if c == 0 { &mut planes.cb } else { &mut planes.cr };
		filter_edge(plane, cx, cy, 8, false, &bs, qpc, params.alpha, params.beta, true);
	}
}
