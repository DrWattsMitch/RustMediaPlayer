//! Motion-vector candidates (rav1d `refmvs.rs: rav1d_refmvs_find`, for
//! single-reference blocks of the sequences this encoder writes: no
//! temporal candidates, identity global motion, one reference frame).
//!
//! The candidate list and the mode contexts it yields are the decoder's, so
//! NEARESTMV / NEARMV / NEWMV mean the same vectors to both sides.

/// A motion vector in 1/8 samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mv {
	pub y: i16,
	pub x: i16,
}

impl Mv {
	pub const ZERO: Mv = Mv { y: 0, x: 0 };
	pub const INVALID: Mv = Mv { y: i16::MIN, x: i16::MIN };

	pub fn new(y: i32, x: i32) -> Self {
		Self { y: y as i16, x: x as i16 }
	}
}

/// What each 4x4 cell remembers (`RefMvsBlock`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
	pub mv: [Mv; 2],
	/// 0 intra, 1 LAST ...; -1 none.
	pub r#ref: [i8; 2],
	/// Block size (rav1d `BlockSize`).
	pub bs: u8,
	/// Motion flags: 1 global motion, 2 new motion vector.
	pub mf: u8,
}

impl Default for Cell {
	fn default() -> Self {
		Self { mv: [Mv::INVALID, Mv::ZERO], r#ref: [0, -1], bs: BS_8X8, mf: 0 }
	}
}

/// rav1d `BlockSize` values of the square sizes.
pub const BS_64X64: u8 = 3;
pub const BS_32X32: u8 = 7;
pub const BS_16X16: u8 = 12;
pub const BS_8X8: u8 = 17;

/// `[width, height]` in 4-sample units of every rav1d `BlockSize`.
pub const BLOCK_DIMS: [[u8; 2]; 22] = [
	[32, 32],
	[32, 16],
	[16, 32],
	[16, 16],
	[16, 8],
	[16, 4],
	[8, 16],
	[8, 8],
	[8, 4],
	[8, 2],
	[4, 16],
	[4, 8],
	[4, 4],
	[4, 2],
	[4, 1],
	[2, 8],
	[2, 4],
	[2, 2],
	[2, 1],
	[1, 4],
	[1, 2],
	[1, 1],
];

/// The 4x4-cell grid of a frame.
#[derive(Debug, Clone)]
pub struct Grid {
	pub cells: Vec<Cell>,
	pub stride: usize,
	/// Frame size in 4x4 cells (`(w + 3) >> 2`).
	pub iw4: i32,
	pub ih4: i32,
}

impl Grid {
	pub fn new(width: usize, height: usize) -> Self {
		let (iw4, ih4) = (width.div_ceil(4), height.div_ceil(4));
		let stride = iw4.next_multiple_of(16) + 16;
		Self {
			cells: vec![Cell::default(); stride * (ih4.next_multiple_of(16) + 16)],
			stride,
			iw4: iw4 as i32,
			ih4: ih4 as i32,
		}
	}

	#[inline]
	fn at(&self, x: i32, y: i32) -> Cell {
		self.cells[y as usize * self.stride + x as usize]
	}

	/// Stores `c` over a block (`splat_mv`), clipped to the grid.
	pub fn splat(&mut self, x4: usize, y4: usize, bw4: usize, bh4: usize, c: Cell) {
		for y in y4..y4 + bh4 {
			let at = y * self.stride + x4;
			self.cells[at..at + bw4].fill(c);
		}
	}
}

/// A candidate.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cand {
	pub mv: Mv,
	pub weight: i32,
}

/// The tile in 4x4 cells.
#[derive(Debug, Clone, Copy)]
pub struct TileRange {
	pub col_start: i32,
	pub col_end: i32,
	pub row_start: i32,
	pub row_end: i32,
}

fn add_spatial(
	stack: &mut [Cand; 8],
	cnt: &mut usize,
	weight: i32,
	b: Cell,
	have_newmv: &mut i32,
	have_refmv: &mut i32,
) {
	if b.mv[0] == Mv::INVALID {
		return;
	}
	for n in 0..2 {
		if b.r#ref[n] == 1 {
			let cand_mv = b.mv[n];
			*have_refmv = 1;
			*have_newmv |= i32::from(b.mf >> 1);
			if let Some(c) = stack[..*cnt].iter_mut().find(|c| c.mv == cand_mv) {
				c.weight += weight;
				return;
			}
			if *cnt < 8 {
				stack[*cnt] = Cand { mv: cand_mv, weight };
				*cnt += 1;
			}
			return;
		}
	}
}

#[allow(clippy::too_many_arguments)]
fn scan_row(
	g: &Grid,
	stack: &mut [Cand; 8],
	cnt: &mut usize,
	x0: i32,
	y: i32,
	bw4: i32,
	w4: i32,
	max_rows: i32,
	step: i32,
	newmv: &mut i32,
	refmv: &mut i32,
) -> i32 {
	let mut cand = g.at(x0, y);
	let first = BLOCK_DIMS[usize::from(cand.bs)];
	let mut cand_bw4 = i32::from(first[0]);
	let mut len = step.max(bw4.min(cand_bw4));
	if bw4 <= cand_bw4 {
		let weight = if bw4 == 1 { 2 } else { 2.max((2 * max_rows).min(i32::from(first[1]))) };
		add_spatial(stack, cnt, len * weight, cand, newmv, refmv);
		return weight >> 1;
	}
	let mut x = 0;
	loop {
		add_spatial(stack, cnt, len * 2, cand, newmv, refmv);
		x += len;
		if x >= w4 {
			return 1;
		}
		cand = g.at(x0 + x, y);
		cand_bw4 = i32::from(BLOCK_DIMS[usize::from(cand.bs)][0]);
		len = step.max(cand_bw4);
	}
}

#[allow(clippy::too_many_arguments)]
fn scan_col(
	g: &Grid,
	stack: &mut [Cand; 8],
	cnt: &mut usize,
	x: i32,
	y0: i32,
	bh4: i32,
	h4: i32,
	max_cols: i32,
	step: i32,
	newmv: &mut i32,
	refmv: &mut i32,
) -> i32 {
	let mut cand = g.at(x, y0);
	let first = BLOCK_DIMS[usize::from(cand.bs)];
	let mut cand_bh4 = i32::from(first[1]);
	let mut len = step.max(bh4.min(cand_bh4));
	if bh4 <= cand_bh4 {
		let weight = if bh4 == 1 { 2 } else { 2.max((2 * max_cols).min(i32::from(first[0]))) };
		add_spatial(stack, cnt, len * weight, cand, newmv, refmv);
		return weight >> 1;
	}
	let mut y = 0;
	loop {
		add_spatial(stack, cnt, len * 2, cand, newmv, refmv);
		y += len;
		if y >= h4 {
			return 1;
		}
		cand = g.at(x, y0 + y);
		cand_bh4 = i32::from(BLOCK_DIMS[usize::from(cand.bs)][1]);
		len = step.max(cand_bh4);
	}
}

/// The candidates of a LAST-reference block of size `bs` at (`bx4`, `by4`):
/// the stack, how many entries it has, and the mode context
/// (`newmv | globalmv << 3 | refmv << 4`). `top_has_right` is the block's
/// edge flag.
pub fn find(
	g: &Grid,
	tile: &TileRange,
	bs: u8,
	bx4: i32,
	by4: i32,
	top_has_right: bool,
) -> ([Cand; 8], usize, i32) {
	let dims = BLOCK_DIMS[usize::from(bs)];
	let (bw4, bh4) = (i32::from(dims[0]), i32::from(dims[1]));
	let w4 = bw4.min(16).min(tile.col_end.min(g.iw4) - bx4);
	let h4 = bh4.min(16).min(tile.row_end.min(g.ih4) - by4);
	let mut stack = [Cand::default(); 8];
	let mut cnt = 0usize;
	let (mut have_newmv, mut have_col_mvs, mut have_row_mvs) = (0, 0, 0);

	// Top.
	let (max_rows, mut n_rows) = if by4 > tile.row_start {
		let max_rows = ((by4 - tile.row_start + 1) >> 1).min(2 + i32::from(bh4 > 1));
		let n = scan_row(
			g,
			&mut stack,
			&mut cnt,
			bx4,
			by4 - 1,
			bw4,
			w4,
			max_rows,
			if bw4 >= 16 { 4 } else { 1 },
			&mut have_newmv,
			&mut have_row_mvs,
		);
		(max_rows, n as u32)
	} else {
		(0, !0u32)
	};
	// Left.
	let (max_cols, mut n_cols) = if bx4 > tile.col_start {
		let max_cols = ((bx4 - tile.col_start + 1) >> 1).min(2 + i32::from(bw4 > 1));
		let n = scan_col(
			g,
			&mut stack,
			&mut cnt,
			bx4 - 1,
			by4,
			bh4,
			h4,
			max_cols,
			if bh4 >= 16 { 4 } else { 1 },
			&mut have_newmv,
			&mut have_col_mvs,
		);
		(max_cols, n as u32)
	} else {
		(0, !0u32)
	};
	// Top-right.
	if n_rows != !0 && top_has_right && bw4.max(bh4) <= 16 && bw4 + bx4 < tile.col_end.min(g.iw4) {
		add_spatial(
			&mut stack,
			&mut cnt,
			4,
			g.at(bx4 + bw4, by4 - 1),
			&mut have_newmv,
			&mut have_row_mvs,
		);
	}
	let nearest_match = have_col_mvs + have_row_mvs;
	let nearest_cnt = cnt;
	for c in &mut stack[..nearest_cnt] {
		c.weight += 640;
	}
	let globalmv_ctx = 0; // no temporal candidates (use_ref_frame_mvs = 0)

	// Top-left.
	let mut dummy = 0;
	if n_rows | n_cols != !0 {
		add_spatial(&mut stack, &mut cnt, 4, g.at(bx4 - 1, by4 - 1), &mut dummy, &mut have_row_mvs);
	}
	// Secondary rows and columns (8x8 resolution).
	for n in 2..=3u32 {
		if n > n_rows && n <= max_rows as u32 {
			let y = (by4 - 2 * n as i32 + 1) | 1;
			n_rows = n_rows.wrapping_add(scan_row(
				g,
				&mut stack,
				&mut cnt,
				bx4 | 1,
				y,
				bw4,
				w4,
				1 + max_rows - n as i32,
				if bw4 >= 16 { 4 } else { 2 },
				&mut dummy,
				&mut have_row_mvs,
			) as u32);
		}
		if n > n_cols && n <= max_cols as u32 {
			let x = (bx4 - n as i32 * 2 + 1) | 1;
			n_cols = n_cols.wrapping_add(scan_col(
				g,
				&mut stack,
				&mut cnt,
				x,
				by4 | 1,
				bh4,
				h4,
				1 + max_cols - n as i32,
				if bh4 >= 16 { 4 } else { 2 },
				&mut dummy,
				&mut have_col_mvs,
			) as u32);
		}
	}
	let ref_match_count = have_col_mvs + have_row_mvs;
	let (refmv_ctx, newmv_ctx) = match nearest_match {
		0 => (2.min(ref_match_count), i32::from(ref_match_count > 0)),
		1 => ((ref_match_count * 3).min(4), 3 - have_newmv),
		_ => (5, 5 - have_newmv),
	};
	// Stable sort by weight, nearest first then the rest.
	stack[..nearest_cnt].sort_by_key(|c| -c.weight);
	stack[nearest_cnt..cnt].sort_by_key(|c| -c.weight);

	if cnt < 2 {
		// Extended candidates: other references' vectors (all LAST here, so
		// only intra neighbours fail) from the nearest row and column.
		let sz4 = w4.min(h4);
		let extend = |cand: Cell, stack: &mut [Cand; 8], cnt: &mut usize| {
			for n in 0..2 {
				let r = cand.r#ref[n];
				if r <= 0 {
					break;
				}
				let mv = cand.mv[n];
				if !stack[..*cnt].iter().any(|c| c.mv == mv) {
					stack[*cnt] = Cand { mv, weight: 2 };
					*cnt += 1;
				}
			}
		};
		if n_rows != !0 {
			let mut x = 0;
			while x < sz4 && cnt < 2 {
				let c = g.at(bx4 + x, by4 - 1);
				extend(c, &mut stack, &mut cnt);
				x += i32::from(BLOCK_DIMS[usize::from(c.bs)][0]);
			}
		}
		if n_cols != !0 {
			let mut y = 0;
			while y < sz4 && cnt < 2 {
				let c = g.at(bx4 - 1, by4 + y);
				extend(c, &mut stack, &mut cnt);
				y += i32::from(BLOCK_DIMS[usize::from(c.bs)][1]);
			}
		}
	}
	// Clamping.
	if cnt != 0 {
		let left = -(bx4 + bw4 + 4) * 4 * 8;
		let right = (g.iw4 - bx4 + 4) * 4 * 8;
		let top = -(by4 + bh4 + 4) * 4 * 8;
		let bottom = (g.ih4 - by4 + 4) * 4 * 8;
		for c in &mut stack[..cnt] {
			c.mv.x = i32::from(c.mv.x).clamp(left, right) as i16;
			c.mv.y = i32::from(c.mv.y).clamp(top, bottom) as i16;
		}
	}
	for c in &mut stack[cnt.min(2)..2] {
		c.mv = Mv::ZERO;
	}
	(stack, cnt, refmv_ctx << 4 | globalmv_ctx << 3 | newmv_ctx)
}

/// `get_drl_context`.
pub fn drl_ctx(stack: &[Cand; 8], idx: usize) -> usize {
	if stack[idx].weight >= 640 {
		usize::from(stack[idx + 1].weight < 640)
	} else if stack[idx + 1].weight < 640 {
		2
	} else {
		0
	}
}

/// `fix_mv_precision` without high precision: even 1/8 values.
pub fn fix_precision(mv: Mv) -> Mv {
	let f = |v: i16| (v - (v >> 15)) & !1;
	Mv { y: f(mv.y), x: f(mv.x) }
}
