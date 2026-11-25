//! One frame: rate-distortion search with reconstruction, then the tile
//! bitstream.
//!
//! Each superblock is first searched: candidates are predicted, transformed,
//! quantised, priced and reconstructed for real, and rolled back
//! ([`Checkpoint`]) when they lose; the decisions are recorded. The
//! contexts are then put back as they were at the start of the superblock
//! and the decisions written with the tile's range encoder by
//! [`FrameEncoder::write_sb`], which mirrors rav1d's `decode_sb` /
//! `decode_b` (and updates contexts and CDFs exactly as they do). The
//! search prices symbols with the tile's CDFs as adapted so far.

use std::sync::Arc;

use super::cdf::CdfContext;
use super::coef::{Ctx, INTRA_SET, Side, TXTP_FROM_UVMODE, TxInfo, dequant, scan, write_coefs};
use super::ipred::{self, Pos};
use super::itx::{self, DCT_DCT};
use super::mc::{self, RefPlane};
use super::msac::{Cost, Ec};
use super::refmvs::{self, BS_8X8, BS_16X16, BS_32X32, BS_64X64, Cand, Cell, Grid, Mv, TileRange};

/// A picture plane.
#[derive(Debug, Clone)]
pub struct Plane {
	pub data: Vec<u8>,
	pub stride: usize,
	/// Visible size.
	pub width: usize,
	pub height: usize,
}

impl Plane {
	/// A plane of `width x height` visible samples, its buffer covering
	/// whole superblocks plus a margin.
	pub fn new(width: usize, height: usize, sub: usize) -> Self {
		let stride = (width.next_multiple_of(64 >> sub)) + (64 >> sub);
		let rows = height.next_multiple_of(64 >> sub) + (64 >> sub);
		Self { data: vec![0; stride * rows], stride, width, height }
	}

	pub fn as_ref(&self) -> RefPlane<'_> {
		RefPlane { data: &self.data, stride: self.stride, width: self.width, height: self.height }
	}
}

/// A reconstructed frame kept as reference.
#[derive(Debug, Clone)]
pub struct Picture {
	pub planes: [Plane; 3],
}

impl Picture {
	/// The visible picture as packed I420.
	pub fn bytes(&self) -> Vec<u8> {
		let mut out = Vec::new();
		for p in &self.planes {
			for y in 0..p.height {
				out.extend_from_slice(&p.data[y * p.stride..y * p.stride + p.width]);
			}
		}
		out
	}
}

pub const NEARESTMV: u8 = 0;
pub const NEARMV: u8 = 1;
pub const GLOBALMV: u8 = 2;
pub const NEWMV: u8 = 3;

/// A block's modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
	Intra {
		y: u8,
		uv: u8,
		y_angle: i8,
		uv_angle: i8,
		txtp: u8,
	},
	/// `drl`: candidate index (0 nearest, 1-3 near; NEWMV: its predictor).
	Inter {
		mode: u8,
		drl: u8,
		mv: Mv,
	},
}

/// A coded block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blk {
	/// rav1d `BlockSize`.
	pub bs: u8,
	/// Side, log2 of 4-sample units (1 = 8x8 ... 4 = 64x64).
	pub lw: usize,
	pub skip: bool,
	pub mode: Mode,
}

#[derive(Debug, Clone, Copy)]
pub enum Dec {
	/// NONE (0) or SPLIT (3).
	Part(u8),
	Blk(Blk),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Rd {
	pub dist: u64,
	/// 1/256 bit.
	pub rate: u32,
}

impl Rd {
	pub fn cost(&self, lambda: f64) -> f64 {
		self.dist as f64 + lambda * f64::from(self.rate) / 256.0
	}

	fn add(&mut self, o: Rd) {
		self.dist += o.dist;
		self.rate += o.rate;
	}
}

/// rav1d `BlockSize` of a square block level (1 = 64x64 ... 4 = 8x8).
fn bs_of(bl: usize) -> u8 {
	[BS_64X64, BS_64X64, BS_32X32, BS_16X16, BS_8X8][bl]
}

/// rav1d `dav1d_intra_mode_context`.
const INTRA_MODE_CTX: [u8; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
/// rav1d `dav1d_ymode_size_context` of 64, 32, 16 and 8-sample squares
/// (by `lw`).
const YMODE_SIZE_CTX: [usize; 5] = [0, 1, 2, 3, 3];
/// rav1d `dav1d_al_part_ctx[_][bl][NONE]`.
const AL_PART_NONE: [u8; 5] = [0x00, 0x10, 0x18, 0x1c, 0x1e];

/// Edge flags of child `n` of a split node (rav1d `intra_edge.rs`).
fn child_edges(n: usize, thr: bool, lhb: bool) -> (bool, bool) {
	(!(n == 3 || (n == 1 && !thr)), n == 0 || (n == 2 && lhb))
}

/// 4x4 Hadamard SATD.
fn satd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
	let mut total = 0u32;
	for by in (0..h).step_by(4) {
		for bx in (0..w).step_by(4) {
			let mut d = [0i32; 16];
			for y in 0..4 {
				for x in 0..4 {
					d[y * 4 + x] =
						i32::from(a[(by + y) * a_stride + bx + x]) - i32::from(b[(by + y) * b_stride + bx + x]);
				}
			}
			for y in 0..4 {
				let r = &mut d[y * 4..y * 4 + 4];
				let (s0, s1, s2, s3) = (r[0] + r[1], r[0] - r[1], r[2] + r[3], r[2] - r[3]);
				r[0] = s0 + s2;
				r[1] = s1 + s3;
				r[2] = s0 - s2;
				r[3] = s1 - s3;
			}
			let mut sum = 0;
			for x in 0..4 {
				let (a0, a1, a2, a3) = (d[x], d[4 + x], d[8 + x], d[12 + x]);
				let (s0, s1, s2, s3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
				sum += (s0 + s2).abs() + (s1 + s3).abs() + (s0 - s2).abs() + (s1 - s3).abs();
			}
			total += (sum as u32).div_ceil(2);
		}
	}
	total
}

fn sad(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
	(0..h)
		.map(|y| {
			a[y * a_stride..y * a_stride + w]
				.iter()
				.zip(&b[y * b_stride..y * b_stride + w])
				.map(|(&p, &q)| u32::from(p.abs_diff(q)))
				.sum::<u32>()
		})
		.sum()
}

/// Saved state of a square region.
pub struct Checkpoint {
	x4: usize,
	y4: usize,
	n4: usize,
	pix: [Vec<u8>; 3],
	a: Side,
	l: Side,
	cells: Vec<Cell>,
	dec_len: usize,
	coef_len: usize,
	decs: Vec<Dec>,
	coefs: Vec<i32>,
}

/// Search effort.
#[derive(Debug, Clone, Copy)]
pub struct Search {
	pub intra_candidates: usize,
	pub inter_candidates: usize,
	pub range: i32,
}

impl Default for Search {
	fn default() -> Self {
		Self { intra_candidates: 3, inter_candidates: 2, range: 64 }
	}
}

/// Luma intra modes tried (no CfL, palette or filter intra).
const Y_MODES: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

pub struct FrameEncoder {
	pub key: bool,
	/// Frame size in 4x4 units, aligned to 8 samples (`MiCols`, `MiRows`).
	pub bw: usize,
	pub bh: usize,
	pub qidx: u8,
	/// DC and AC quantizers.
	pub dq: [i32; 2],
	pub lambda: f64,
	pub sad_per_bit: f64,
	pub search: Search,
	pub src: [Plane; 3],
	pub rec: [Plane; 3],
	pub refp: Option<Arc<Picture>>,
	pub grid: Grid,
	pub ctx: Ctx,
	pub tile: TileRange,
	pub cdf: CdfContext,
	decisions: Vec<Dec>,
	coefs: Vec<i32>,
	cursor: (usize, usize),
	pad_ref: Option<(Vec<u8>, usize)>,
	pred: [Vec<u8>; 3],
	mcbuf: Vec<u8>,
}

const BORDER: i32 = 96;

impl FrameEncoder {
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		key: bool,
		src: [Plane; 3],
		refp: Option<Arc<Picture>>,
		qidx: u8,
		dq: [i32; 2],
		lambda: f64,
		sad_per_bit: f64,
		search: Search,
	) -> Self {
		let (w, h) = (src[0].width, src[0].height);
		let bw = w.div_ceil(8) * 2;
		let bh = h.div_ceil(8) * 2;
		let rec = [
			Plane::new(w, h, 0),
			Plane::new(w.div_ceil(2), h.div_ceil(2), 1),
			Plane::new(w.div_ceil(2), h.div_ceil(2), 1),
		];
		let n4 = bw.next_multiple_of(16) + 32;
		let pad_ref = refp.as_ref().map(|r| {
			let p = &r.planes[0];
			let stride = p.stride + 2 * BORDER as usize;
			let rows = p.data.len() / p.stride + 2 * BORDER as usize;
			let mut data = vec![0u8; stride * rows];
			for y in 0..rows {
				let sy = (y as i32 - BORDER).clamp(0, p.height as i32 - 1) as usize;
				for x in 0..stride {
					let sx = (x as i32 - BORDER).clamp(0, p.width as i32 - 1) as usize;
					data[y * stride + x] = p.data[sy * p.stride + sx];
				}
			}
			(data, stride)
		});
		Self {
			key,
			bw,
			bh,
			qidx,
			dq,
			lambda,
			sad_per_bit,
			search,
			src,
			rec,
			refp,
			grid: Grid::new(w, h),
			ctx: Ctx { a: Side::new(n4), l: Side::new(32) },
			tile: TileRange { col_start: 0, col_end: 0, row_start: 0, row_end: 0 },
			cdf: CdfContext::new(qidx),
			decisions: Vec::new(),
			coefs: Vec::new(),
			cursor: (0, 0),
			pad_ref,
			pred: [vec![0; 64 * 64], vec![0; 32 * 32], vec![0; 32 * 32]],
			mcbuf: vec![0; 64 * 64],
		}
	}

	/// Starts a tile: fresh CDFs and above contexts.
	pub fn start_tile(&mut self, tile: TileRange) {
		self.tile = tile;
		self.cdf = CdfContext::new(self.qidx);
		let (s, e) = (tile.col_start as usize, (tile.col_end as usize).next_multiple_of(16));
		self.ctx.a.reset(self.key, s..e);
	}

	/// Starts a superblock row of the current tile.
	pub fn start_sb_row(&mut self) {
		self.ctx.l.reset(self.key, 0..32);
	}

	/// Searches, reconstructs and writes one superblock.
	pub fn code_sb<E: Ec>(&mut self, e: &mut E, x4: usize, y4: usize) {
		let saved_a = self.ctx.a.save(x4..x4 + 16);
		let saved_l = self.ctx.l.clone();
		self.decisions.clear();
		self.coefs.clear();
		self.search_part(1, x4, y4, true, false);
		self.ctx.a.restore(x4, &saved_a);
		self.ctx.l = saved_l;
		self.cursor = (0, 0);
		self.write_sb(e, 1, x4, y4, true, false);
	}

	// --- Writing (rav1d decode_sb / decode_b mirrored) ---

	fn partition_ctx(&self, bl: usize, x4: usize, y4: usize) -> usize {
		let has = |v: u8| usize::from((v >> (4 - bl)) & 1);
		has(self.ctx.a.partition[x4 >> 1]) + 2 * has(self.ctx.l.partition[(y4 & 15) >> 1])
	}

	fn set_partition_ctx(&mut self, bl: usize, x4: usize, y4: usize) {
		let hsz = 16 >> bl;
		let v = AL_PART_NONE[bl];
		let (ax, ly) = (x4 >> 1, (y4 & 15) >> 1);
		self.ctx.a.partition[ax..ax + hsz].fill(v);
		let end = (ly + hsz).min(self.ctx.l.partition.len());
		self.ctx.l.partition[ly..end].fill(v);
	}

	/// Writes partition `p` (0 none, 3 split) of a `bl` node.
	fn write_partition<E: Ec>(&mut self, e: &mut E, bl: usize, x4: usize, y4: usize, p: u8) {
		let hsz = 16 >> bl;
		let have_h = self.bw > x4 + hsz;
		let have_v = self.bh > y4 + hsz;
		let ctx = self.partition_ctx(bl, x4, y4);
		let (cdf, n) = self.cdf.partition(bl, ctx);
		if have_h && have_v {
			e.symbol(usize::from(p), cdf, n);
		} else if have_h {
			// gather_top_partition_prob: H against the splits.
			let f = i32::from(cdf[1]) - i32::from(cdf[4]) + i32::from(cdf[5]) + i32::from(cdf[8])
				- i32::from(cdf[7]);
			e.bool_f(p == 3, f as u32);
		} else {
			// gather_left_partition_prob: V against the splits.
			let f = i32::from(cdf[0]) - i32::from(cdf[1]) + i32::from(cdf[2]) - i32::from(cdf[6])
				+ i32::from(cdf[7])
				- i32::from(cdf[8]);
			e.bool_f(p == 3, f as u32);
		}
	}

	fn next(&mut self) -> Dec {
		let d = self.decisions[self.cursor.0];
		self.cursor.0 += 1;
		d
	}

	fn write_sb<E: Ec>(&mut self, e: &mut E, bl: usize, x4: usize, y4: usize, thr: bool, lhb: bool) {
		let hsz = 16 >> bl;
		let have_h = self.bw > x4 + hsz;
		let have_v = self.bh > y4 + hsz;
		if !have_h && !have_v {
			let (t, l) = child_edges(0, thr, lhb);
			self.write_sb(e, bl + 1, x4, y4, t, l);
			return;
		}
		let Dec::Part(p) = self.next() else { unreachable!("av1 encoder: decisions out of order") };
		self.write_partition(e, bl, x4, y4, p);
		if p == 0 {
			let Dec::Blk(b) = self.next() else { unreachable!("av1 encoder: decisions out of order") };
			self.write_block(e, &b, x4, y4, thr, lhb);
			self.set_partition_ctx(bl, x4, y4);
		} else {
			let children: &[usize] = if have_h && have_v {
				&[0, 1, 2, 3]
			} else if have_h {
				&[0, 1]
			} else {
				&[0, 2]
			};
			for &n in children {
				let (t, l) = child_edges(n, thr, lhb);
				self.write_sb(e, bl + 1, x4 + (n & 1) * hsz, y4 + (n >> 1) * hsz, t, l);
			}
		}
	}

	fn have(&self, x4: usize, y4: usize) -> (bool, bool) {
		(x4 as i32 > self.tile.col_start, y4 as i32 > self.tile.row_start)
	}

	fn intra_ctx(&self, x4: usize, y4: usize) -> usize {
		let (hl, ht) = self.have(x4, y4);
		let (a, l) = (self.ctx.a.intra[x4], self.ctx.l.intra[y4 & 15]);
		usize::from(match (hl, ht) {
			(true, true) => {
				let c = l + a;
				c + u8::from(c == 2)
			}
			(true, false) => l * 2,
			(false, true) => a * 2,
			(false, false) => 0,
		})
	}

	/// Context of the single-reference bits when every inter block uses LAST
	/// (`av1_get_ref_ctx` and the forward-reference contexts agree then).
	fn ref_ctx(&self, x4: usize, y4: usize) -> usize {
		let (hl, ht) = self.have(x4, y4);
		let n = u8::from(ht && self.ctx.a.intra[x4] == 0 && self.ctx.a.r#ref[x4] >= 0)
			+ u8::from(hl && self.ctx.l.intra[y4 & 15] == 0 && self.ctx.l.r#ref[y4 & 15] >= 0);
		if n == 0 { 1 } else { 2 }
	}

	/// The candidate list of an inter block.
	fn mv_stack(&self, b: &Blk, x4: usize, y4: usize, thr: bool) -> ([Cand; 8], usize, i32) {
		refmvs::find(&self.grid, &self.tile, b.bs, x4 as i32, y4 as i32, thr)
	}

	/// The motion vector an inter mode and candidate index give, and for
	/// NEWMV its predictor.
	fn mode_mv(stack: &[Cand; 8], n_mvs: usize, mode: u8, drl: u8) -> Mv {
		match mode {
			GLOBALMV => Mv::ZERO,
			NEARESTMV => refmvs::fix_precision(stack[0].mv),
			NEARMV => {
				let mv = stack[usize::from(drl)].mv;
				if drl < 2 { refmvs::fix_precision(mv) } else { mv }
			}
			_ => {
				if n_mvs > 1 {
					stack[usize::from(drl)].mv
				} else {
					refmvs::fix_precision(stack[0].mv)
				}
			}
		}
	}

	fn write_mv_component<E: Ec>(e: &mut E, c: &mut super::cdf::MvComponentCdf, v: i32) {
		e.bool_adapt(v < 0, &mut c.sign);
		let z = v.unsigned_abs() - 1;
		let cl = if z < 16 { 0 } else { (31 - (z >> 3).leading_zeros()) as usize };
		e.symbol(cl, &mut c.classes, 10);
		let fp = ((z >> 1) & 3) as usize;
		if cl == 0 {
			let up = (z >> 3) as usize;
			e.bool_adapt(up == 1, &mut c.class_0);
			e.symbol(fp, &mut c.class_0_fp[up], 3);
		} else {
			let up = (z >> 3) - (1 << cl);
			for n in 0..cl {
				e.bool_adapt((up >> n) & 1 == 1, &mut c.class_n[n]);
			}
			e.symbol(fp, &mut c.class_n_fp, 3);
		}
	}

	fn write_mv<E: Ec>(e: &mut E, cdf: &mut CdfContext, diff: Mv) {
		let (y, x) = (i32::from(diff.y), i32::from(diff.x));
		let joint = usize::from(y != 0) * 2 + usize::from(x != 0);
		e.symbol(joint, &mut cdf.mv_joint, 3);
		if y != 0 {
			Self::write_mv_component(e, &mut cdf.mv[0], y);
		}
		if x != 0 {
			Self::write_mv_component(e, &mut cdf.mv[1], x);
		}
	}

	/// Mode info of a block (everything before its coefficients).
	fn write_mode_info<E: Ec>(&mut self, e: &mut E, b: &Blk, x4: usize, y4: usize, thr: bool) {
		let ly = y4 & 15;
		let sctx = usize::from(self.ctx.a.skip[x4] + self.ctx.l.skip[ly]);
		e.bool_adapt(b.skip, &mut self.cdf.skip[sctx]);
		let intra = matches!(b.mode, Mode::Intra { .. });
		if !self.key {
			let ictx = self.intra_ctx(x4, y4);
			e.bool_adapt(!intra, &mut self.cdf.intra[ictx]);
		}
		match b.mode {
			Mode::Intra { y, uv, y_angle, uv_angle, .. } => {
				let cdf = if self.key {
					let ca = INTRA_MODE_CTX[usize::from(self.ctx.a.mode[x4])];
					let cl = INTRA_MODE_CTX[usize::from(self.ctx.l.mode[ly])];
					&mut self.cdf.kf_y_mode[usize::from(ca)][usize::from(cl)]
				} else {
					&mut self.cdf.y_mode[YMODE_SIZE_CTX[b.lw]]
				};
				e.symbol(usize::from(y), cdf, 12);
				if b.lw >= 1 && (1..=8).contains(&y) {
					e.symbol((y_angle + 3) as usize, &mut self.cdf.angle_delta[usize::from(y - 1)], 6);
				}
				if b.lw <= 3 {
					e.symbol(usize::from(uv), &mut self.cdf.uv_mode_cfl[usize::from(y)], 13);
				} else {
					e.symbol(usize::from(uv), &mut self.cdf.uv_mode_no_cfl[usize::from(y)], 12);
				}
				if b.lw >= 1 && (1..=8).contains(&uv) {
					e.symbol((uv_angle + 3) as usize, &mut self.cdf.angle_delta[usize::from(uv - 1)], 6);
				}
			}
			Mode::Inter { mode, drl, mv } => {
				let rctx = self.ref_ctx(x4, y4);
				e.bool_adapt(false, &mut self.cdf.r#ref[0][rctx]);
				e.bool_adapt(false, &mut self.cdf.r#ref[2][rctx]);
				e.bool_adapt(false, &mut self.cdf.r#ref[3][rctx]);
				let (stack, n_mvs, ctx) = self.mv_stack(b, x4, y4, thr);
				let cdf = &mut self.cdf;
				e.bool_adapt(mode != NEWMV, &mut cdf.newmv_mode[(ctx & 7) as usize]);
				if mode == NEWMV {
					if n_mvs > 1 {
						e.bool_adapt(drl > 0, &mut cdf.drl_bit[refmvs::drl_ctx(&stack, 0)]);
						if drl > 0 && n_mvs > 2 {
							e.bool_adapt(drl > 1, &mut cdf.drl_bit[refmvs::drl_ctx(&stack, 1)]);
						}
					}
					let pred = Self::mode_mv(&stack, n_mvs, NEWMV, drl);
					Self::write_mv(
						e,
						cdf,
						Mv::new(i32::from(mv.y) - i32::from(pred.y), i32::from(mv.x) - i32::from(pred.x)),
					);
				} else {
					e.bool_adapt(mode != GLOBALMV, &mut cdf.globalmv_mode[((ctx >> 3) & 1) as usize]);
					if mode != GLOBALMV {
						e.bool_adapt(mode == NEARMV, &mut cdf.refmv_mode[((ctx >> 4) & 15) as usize]);
						if mode == NEARMV && n_mvs > 2 {
							e.bool_adapt(drl > 1, &mut cdf.drl_bit[refmvs::drl_ctx(&stack, 1)]);
							if drl > 1 && n_mvs > 3 {
								e.bool_adapt(drl > 2, &mut cdf.drl_bit[refmvs::drl_ctx(&stack, 2)]);
							}
						}
					}
				}
			}
		}
	}

	/// The transform blocks of a block in coding order: (plane, x, y in the
	/// plane's 4x4 units, transform `lw`, top-has-right, left-has-bottom).
	fn tx_blocks(
		&self,
		b: &Blk,
		x4: usize,
		y4: usize,
		thr: bool,
		lhb: bool,
	) -> Vec<(usize, usize, usize, usize, bool, bool)> {
		let mut out = Vec::new();
		let bw4 = 1usize << b.lw;
		let w4 = bw4.min(self.bw - x4);
		let h4 = bw4.min(self.bh - y4);
		let tl = b.lw.min(4);
		let tw = 1usize << tl;
		let mut y = 0;
		while y < h4 {
			let mut x = 0;
			while x < w4 {
				let t = !((y > 0 || !thr) && x + tw >= w4);
				let l = !(x > 0 || (!lhb && y + tw >= h4));
				out.push((0, x4 + x, y4 + y, tl, t, l));
				x += tw;
			}
			y += tw;
		}
		let (cw4, ch4) = ((w4 + 1) >> 1, (h4 + 1) >> 1);
		let ul = b.lw - 1;
		let uw = 1usize << ul;
		for pl in 1..3 {
			let mut y = 0;
			while y < ch4 {
				let mut x = 0;
				while x < cw4 {
					let t = !((y > 0 || !thr) && x + uw >= cw4);
					let l = !(x > 0 || (!lhb && y + uw >= ch4));
					out.push((pl, (x4 >> 1) + x, (y4 >> 1) + y, ul, t, l));
					x += uw;
				}
				y += uw;
			}
		}
		out
	}

	fn tx_info(&self, b: &Blk, pl: usize, lw: usize) -> TxInfo {
		let (intra, y_mode) = match b.mode {
			Mode::Intra { y, .. } => (true, y),
			Mode::Inter { .. } => (false, 0),
		};
		TxInfo { lw, chroma: pl > 0, block_lw: b.lw, intra, y_mode }
	}

	/// The transform type of a transform block.
	fn txtp(&self, b: &Blk, pl: usize, lw: usize) -> u8 {
		match b.mode {
			Mode::Intra { .. } if lw >= 3 => DCT_DCT,
			Mode::Intra { uv, .. } if pl > 0 => TXTP_FROM_UVMODE[usize::from(uv)],
			Mode::Intra { txtp, .. } => txtp,
			Mode::Inter { .. } => DCT_DCT,
		}
	}

	/// The coefficient contexts a transform block is coded with.
	fn coef_ctx(&self, pl: usize, x: usize, y: usize, lw: usize) -> (Vec<u8>, Vec<u8>) {
		let n = 1usize << lw;
		let (ly, a, l) = if pl == 0 {
			(y & 15, &self.ctx.a.lcoef, &self.ctx.l.lcoef)
		} else {
			(y & 7, &self.ctx.a.ccoef[pl - 1], &self.ctx.l.ccoef[pl - 1])
		};
		(a[x..x + n].to_vec(), l[ly..ly + n].to_vec())
	}

	/// Stores a transform block's result context (clipped to the frame).
	fn set_coef_ctx(&mut self, pl: usize, x: usize, y: usize, lw: usize, v: u8) {
		let n = 1usize << lw;
		let (bw, bh) =
			if pl == 0 { (self.bw, self.bh) } else { ((self.bw + 1) >> 1, (self.bh + 1) >> 1) };
		let (nw, nh) = (n.min(bw - x), n.min(bh - y));
		if pl == 0 {
			self.ctx.a.lcoef[x..x + nw].fill(v);
			let ly = y & 15;
			self.ctx.l.lcoef[ly..ly + nh].fill(v);
		} else {
			self.ctx.a.ccoef[pl - 1][x..x + nw].fill(v);
			let ly = y & 7;
			self.ctx.l.ccoef[pl - 1][ly..ly + nh].fill(v);
		}
	}

	/// Contexts of a skipped block's transform blocks.
	fn skip_coef_ctx(&mut self, b: &Blk, x4: usize, y4: usize, thr: bool, lhb: bool) {
		let bw4 = 1usize << b.lw;
		if matches!(b.mode, Mode::Inter { .. }) {
			let ly = y4 & 15;
			self.ctx.a.lcoef[x4..x4 + bw4].fill(0x40);
			self.ctx.l.lcoef[ly..ly + bw4].fill(0x40);
			let (cx, cy, cw) = (x4 >> 1, (y4 & 15) >> 1, (bw4 + 1) >> 1);
			for pl in 0..2 {
				self.ctx.a.ccoef[pl][cx..cx + cw].fill(0x40);
				self.ctx.l.ccoef[pl][cy..cy + cw].fill(0x40);
			}
		} else {
			for (pl, x, y, lw, _, _) in self.tx_blocks(b, x4, y4, thr, lhb) {
				let n = 1usize << lw;
				if pl == 0 {
					self.ctx.a.lcoef[x..x + n].fill(0x40);
					let ly = y & 15;
					self.ctx.l.lcoef[ly..ly + n].fill(0x40);
				} else {
					self.ctx.a.ccoef[pl - 1][x..x + n].fill(0x40);
					let ly = y & 7;
					self.ctx.l.ccoef[pl - 1][ly..ly + n].fill(0x40);
				}
			}
		}
	}

	/// The block's own contexts and motion-field cells (end of `decode_b`).
	fn update_block_ctx(&mut self, b: &Blk, x4: usize, y4: usize) {
		let n = 1usize << b.lw;
		let (ax, ly) = (x4, y4 & 15);
		let (cx, cy, cn) = (x4 >> 1, (y4 & 15) >> 1, (n + 1) >> 1);
		let skip = u8::from(b.skip);
		let key = self.key;
		let (a, l) = (&mut self.ctx.a, &mut self.ctx.l);
		match b.mode {
			Mode::Intra { y, uv, .. } => {
				for (s, at) in [(&mut *a, ax), (&mut *l, ly)] {
					s.tx_intra[at..at + n].fill(b.lw.min(4) as i8);
					s.mode[at..at + n].fill(y);
					s.intra[at..at + n].fill(1);
					s.skip[at..at + n].fill(skip);
					if !key {
						s.r#ref[at..at + n].fill(-1);
					}
				}
				a.uvmode[cx..cx + cn].fill(uv);
				l.uvmode[cy..cy + cn].fill(uv);
				if !key {
					self.grid.splat(
						x4,
						y4,
						n,
						n,
						Cell { mv: [Mv::INVALID, Mv::ZERO], r#ref: [0, -1], bs: b.bs, mf: 0 },
					);
				}
			}
			Mode::Inter { mode, mv, .. } => {
				for (s, at) in [(&mut *a, ax), (&mut *l, ly)] {
					s.intra[at..at + n].fill(0);
					s.skip[at..at + n].fill(skip);
					s.tx_intra[at..at + n].fill(b.lw as i8);
					s.mode[at..at + n].fill(mode);
					s.r#ref[at..at + n].fill(0);
				}
				a.uvmode[cx..cx + cn].fill(0);
				l.uvmode[cy..cy + cn].fill(0);
				let mf = u8::from(mode == GLOBALMV && n >= 2) | (u8::from(mode == NEWMV) << 1);
				self.grid.splat(x4, y4, n, n, Cell { mv: [mv, Mv::ZERO], r#ref: [1, -1], bs: b.bs, mf });
			}
		}
	}

	fn write_block<E: Ec>(&mut self, e: &mut E, b: &Blk, x4: usize, y4: usize, thr: bool, lhb: bool) {
		self.write_mode_info(e, b, x4, y4, thr);
		if b.skip {
			self.skip_coef_ctx(b, x4, y4, thr, lhb);
		} else {
			for (pl, x, y, lw, _, _) in self.tx_blocks(b, x4, y4, thr, lhb) {
				let at = self.cursor.1;
				let len = self.coefs[at] as usize;
				let txtp = self.coefs[at + 1] as u8;
				self.cursor.1 = at + 2 + len;
				let (a, l) = self.coef_ctx(pl, x, y, lw);
				let t = self.tx_info(b, pl, lw);
				let levels = std::mem::take(&mut self.coefs);
				let c = write_coefs(e, &mut self.cdf, &t, &a, &l, &levels[at + 2..at + 2 + len], txtp);
				self.coefs = levels;
				self.set_coef_ctx(pl, x, y, lw, c);
			}
		}
		self.update_block_ctx(b, x4, y4);
	}

	// --- Search ---

	fn checkpoint(
		&self,
		x4: usize,
		y4: usize,
		n4: usize,
		tail_from: Option<&Checkpoint>,
	) -> Checkpoint {
		let pix = std::array::from_fn(|pl| {
			let sub = usize::from(pl > 0);
			let p = &self.rec[pl];
			let (x, y, n) = ((x4 * 4) >> sub, (y4 * 4) >> sub, (n4 * 4) >> sub);
			let mut v = Vec::with_capacity(n * n);
			for r in 0..n {
				v.extend_from_slice(&p.data[(y + r) * p.stride + x..(y + r) * p.stride + x + n]);
			}
			v
		});
		let mut cells = Vec::with_capacity(n4 * n4);
		for r in 0..n4 {
			let at = (y4 + r) * self.grid.stride + x4;
			cells.extend_from_slice(&self.grid.cells[at..at + n4]);
		}
		let (dec_len, coef_len) = match tail_from {
			Some(b) => (b.dec_len, b.coef_len),
			None => (self.decisions.len(), self.coefs.len()),
		};
		Checkpoint {
			x4,
			y4,
			n4,
			pix,
			a: self.ctx.a.save(x4..x4 + n4),
			l: self.ctx.l.save(0..32),
			cells,
			dec_len,
			coef_len,
			decs: self.decisions[dec_len..].to_vec(),
			coefs: self.coefs[coef_len..].to_vec(),
		}
	}

	fn restore(&mut self, c: &Checkpoint) {
		for pl in 0..3 {
			let sub = usize::from(pl > 0);
			let p = &mut self.rec[pl];
			let (x, y, n) = ((c.x4 * 4) >> sub, (c.y4 * 4) >> sub, (c.n4 * 4) >> sub);
			for r in 0..n {
				let at = (y + r) * p.stride + x;
				p.data[at..at + n].copy_from_slice(&c.pix[pl][r * n..(r + 1) * n]);
			}
		}
		self.ctx.a.restore(c.x4, &c.a);
		self.ctx.l.restore(0, &c.l);
		for r in 0..c.n4 {
			let at = (c.y4 + r) * self.grid.stride + c.x4;
			self.grid.cells[at..at + c.n4].copy_from_slice(&c.cells[r * c.n4..(r + 1) * c.n4]);
		}
		self.decisions.truncate(c.dec_len);
		self.decisions.extend_from_slice(&c.decs);
		self.coefs.truncate(c.coef_len);
		self.coefs.extend_from_slice(&c.coefs);
	}

	fn partition_cost(&mut self, bl: usize, x4: usize, y4: usize, p: u8) -> u32 {
		let mut c = Cost(0);
		self.write_partition(&mut c, bl, x4, y4, p);
		c.0
	}

	fn search_part(&mut self, bl: usize, x4: usize, y4: usize, thr: bool, lhb: bool) -> Rd {
		let hsz = 16 >> bl;
		let have_h = self.bw > x4 + hsz;
		let have_v = self.bh > y4 + hsz;
		if !have_h && !have_v {
			let (t, l) = child_edges(0, thr, lhb);
			return self.search_part(bl + 1, x4, y4, t, l);
		}
		let n4 = 2 * hsz;
		let can_none = have_h && have_v;
		let can_split = bl < 4;
		let before = (can_none && can_split).then(|| self.checkpoint(x4, y4, n4, None));
		let mut none = None;
		if can_none {
			let mut rd = Rd { dist: 0, rate: self.partition_cost(bl, x4, y4, 0) };
			self.decisions.push(Dec::Part(0));
			rd.add(self.code_best_block(bl, x4, y4, thr, lhb));
			self.set_partition_ctx(bl, x4, y4);
			if !can_split {
				return rd;
			}
			let b = self.grid.cells[y4 * self.grid.stride + x4];
			let area = (n4 * 4 * n4 * 4) as f64;
			if !self.key && b.r#ref[0] == 1 && (rd.dist as f64) < area * self.lambda * 0.02 {
				return rd;
			}
			none = Some(rd);
		}
		let after_none = before.as_ref().map(|b| self.checkpoint(x4, y4, n4, Some(b)));
		if let Some(b) = &before {
			self.restore(b);
		}
		let limit = none.map(|r| r.cost(self.lambda));
		let mut rd = Rd { dist: 0, rate: self.partition_cost(bl, x4, y4, 3) };
		self.decisions.push(Dec::Part(3));
		let children: &[usize] = if have_h && have_v {
			&[0, 1, 2, 3]
		} else if have_h {
			&[0, 1]
		} else {
			&[0, 2]
		};
		for &n in children {
			let (t, l) = child_edges(n, thr, lhb);
			rd.add(self.search_part(bl + 1, x4 + (n & 1) * hsz, y4 + (n >> 1) * hsz, t, l));
			if limit.is_some_and(|l| rd.cost(self.lambda) >= l) {
				break;
			}
		}
		match (none, after_none) {
			(Some(n), Some(a)) if n.cost(self.lambda) <= rd.cost(self.lambda) => {
				self.restore(&a);
				n
			}
			_ => rd,
		}
	}

	/// SSE between source and reconstruction over the visible part.
	fn ssd(&self, pl: usize, x: usize, y: usize, w: usize, h: usize) -> u64 {
		let (s, r) = (&self.src[pl], &self.rec[pl]);
		if x >= s.width || y >= s.height {
			return 0;
		}
		let (w, h) = (w.min(s.width - x), h.min(s.height - y));
		let mut sum = 0u64;
		for row in y..y + h {
			let a = &s.data[row * s.stride + x..row * s.stride + x + w];
			let b = &r.data[row * r.stride + x..row * r.stride + x + w];
			sum += a.iter().zip(b).map(|(&p, &q)| u64::from(p.abs_diff(q)).pow(2)).sum::<u64>();
		}
		sum
	}

	fn intra_pos(&self, pl: usize, x: usize, y: usize, lw: usize, thr: bool, lhb: bool) -> Pos {
		let sub = usize::from(pl > 0);
		let t = &self.tile;
		Pos {
			x,
			y,
			lw,
			have_left: x as i32 > t.col_start >> sub,
			have_top: y as i32 > t.row_start >> sub,
			w_end: (t.col_end >> sub) as usize,
			h_end: (t.row_end >> sub) as usize,
			top_has_right: thr,
			left_has_bottom: lhb,
		}
	}

	/// Predicts the block (inter: every plane; intra: nothing, as intra
	/// prediction happens per transform block).
	fn predict_inter(&mut self, mv: Mv, x4: usize, y4: usize, n4: usize) {
		let Some(r) = self.refp.clone() else { return };
		for pl in 0..3 {
			let sub = usize::from(pl > 0);
			let (x, y, n) = ((x4 * 4) >> sub, (y4 * 4) >> sub, (n4 * 4) >> sub);
			let src = r.planes[pl].as_ref();
			let dst = &mut self.rec[pl];
			let stride = dst.stride;
			mc::predict(
				&src,
				&mut dst.data,
				stride,
				x,
				y,
				n,
				n,
				(i32::from(mv.y), i32::from(mv.x)),
				pl > 0,
			);
		}
	}

	/// Codes a block with the given modes (search side): predicts,
	/// quantises, reconstructs, prices, appends the coefficients and updates
	/// the contexts. Returns its cost and the final block (skip decided).
	fn code_block(&mut self, mut b: Blk, x4: usize, y4: usize, thr: bool, lhb: bool) -> (Rd, Blk) {
		let coefs_at = self.coefs.len();
		let n4 = 1usize << b.lw;
		let inter = matches!(b.mode, Mode::Inter { .. });
		let mut pred_dist = 0;
		if let Mode::Inter { mv, .. } = b.mode {
			self.predict_inter(mv, x4, y4, n4);
			for pl in 0..3 {
				let sub = usize::from(pl > 0);
				let (x, y, n) = ((x4 * 4) >> sub, (y4 * 4) >> sub, (n4 * 4) >> sub);
				let p = &self.rec[pl];
				for r in 0..n {
					self.pred[pl][r * n..(r + 1) * n]
						.copy_from_slice(&p.data[(y + r) * p.stride + x..(y + r) * p.stride + x + n]);
				}
				pred_dist += self.ssd(pl, x, y, n, n);
			}
		}
		let saved_a = self.ctx.a.save(x4..x4 + n4);
		let saved_l = self.ctx.l.save(0..32);
		let mut rd = Rd::default();
		let mut any = false;
		for (pl, x, y, lw, t, l) in self.tx_blocks(&b, x4, y4, thr, lhb) {
			let n = 4usize << lw;
			let (px, py) = (x * 4, y * 4);
			if let Mode::Intra { y: ym, uv, y_angle, uv_angle, .. } = b.mode {
				let (mode, angle) = if pl == 0 { (ym, y_angle) } else { (uv, uv_angle) };
				let pos = self.intra_pos(pl, x, y, lw, t, l);
				let p = &mut self.rec[pl];
				let stride = p.stride;
				ipred::predict(&mut p.data, stride, &pos, mode, angle);
			}
			let txtp = self.txtp(&b, pl, lw);
			let (c, levels) = self.quantize(pl, px, py, lw, txtp, inter);
			let (ca, cl) = self.coef_ctx(pl, x, y, lw);
			let ti = self.tx_info(&b, pl, lw);
			let mut cost = Cost(0);
			let ctxv = write_coefs(&mut cost, &mut self.cdf, &ti, &ca, &cl, &levels, txtp);
			rd.rate += cost.0;
			self.set_coef_ctx(pl, x, y, lw, ctxv);
			self.coefs.push(levels.len() as i32);
			self.coefs.push(i32::from(txtp));
			self.coefs.extend_from_slice(&levels);
			if !levels.is_empty() {
				any = true;
				let mut cf = c;
				let p = &mut self.rec[pl];
				let stride = p.stride;
				itx::inverse_add(
					&mut cf,
					&mut p.data[py * stride + px..],
					stride,
					lw,
					txtp,
					levels.len() - 1,
				);
			}
			rd.dist += self.ssd(pl, px, py, n, n);
		}
		b.skip = !any;
		let coded = {
			let mut c = Cost(0);
			self.write_mode_info(&mut c, &b, x4, y4, thr);
			Rd { dist: rd.dist, rate: rd.rate + c.0 }
		};
		let mut result = coded;
		if inter && any {
			let mut s = b;
			s.skip = true;
			let mut c = Cost(0);
			self.write_mode_info(&mut c, &s, x4, y4, thr);
			let skipped = Rd { dist: pred_dist, rate: c.0 };
			if skipped.cost(self.lambda) <= coded.cost(self.lambda) {
				b.skip = true;
				result = skipped;
				for pl in 0..3 {
					let sub = usize::from(pl > 0);
					let (x, y, n) = ((x4 * 4) >> sub, (y4 * 4) >> sub, (n4 * 4) >> sub);
					let p = &mut self.rec[pl];
					for r in 0..n {
						let at = (y + r) * p.stride + x;
						p.data[at..at + n].copy_from_slice(&self.pred[pl][r * n..(r + 1) * n]);
					}
				}
			}
		}
		if b.skip {
			self.coefs.truncate(coefs_at);
			self.ctx.a.restore(x4, &saved_a);
			self.ctx.l.restore(0, &saved_l);
			self.skip_coef_ctx(&b, x4, y4, thr, lhb);
		}
		self.update_block_ctx(&b, x4, y4);
		(result, b)
	}

	/// Residual of a transform block, forward transform and quantisation
	/// (with a drop-the-block test). Returns the dequantised coefficients
	/// in rav1d's layout and the levels in scan order.
	fn quantize(
		&self,
		pl: usize,
		px: usize,
		py: usize,
		lw: usize,
		txtp: u8,
		inter: bool,
	) -> (Vec<i32>, Vec<i32>) {
		let n = 4usize << lw;
		let s = n.min(32);
		let (src, rec) = (&self.src[pl], &self.rec[pl]);
		let mut res = vec![0i32; n * n];
		let mut energy = 0u64;
		for r in 0..n {
			for c in 0..n {
				if px + c < src.width && py + r < src.height {
					let v = i32::from(src.data[(py + r) * src.stride + px + c])
						- i32::from(rec.data[(py + r) * rec.stride + px + c]);
					res[r * n + c] = v;
					energy += (v * v) as u64;
				}
			}
		}
		let mut cf = vec![0i32; s * s];
		if energy == 0 {
			return (cf, Vec::new());
		}
		let mut f = vec![0f32; s * s];
		itx::forward(&res, &mut f, lw, txtp);
		let shift = lw.saturating_sub(2) as u32;
		let round: f32 = if inter { 0.30 } else { 0.38 };
		let sc = scan(lw);
		let mut levels = Vec::with_capacity(s * s);
		let mut last = 0;
		let mut qerr = 0f64;
		for (i, &rc) in sc.iter().enumerate().take(s * s) {
			let rc = usize::from(rc);
			let q = if rc == 0 { self.dq[0] } else { self.dq[1] };
			let step = q as f32 / (1u32 << shift) as f32;
			let c = f[rc];
			let mut lv = ((c.abs() / step) + round).floor() as i32;
			lv = lv.min((1 << 20) - 1);
			let v = if c < 0.0 { -lv } else { lv };
			let d = dequant(v, q, shift);
			qerr += f64::from((c - d as f32) * (c - d as f32));
			cf[rc] = d;
			levels.push(v);
			if v != 0 {
				last = i + 1;
			}
		}
		levels.truncate(last);
		// Coefficient-domain error to pixels: the transform's energy gain.
		let gain = f.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / energy as f64;
		if last > 0 && energy as f64 <= qerr / gain.max(1e-9) + self.lambda * 2.0 {
			return (vec![0; s * s], Vec::new());
		}
		(cf, levels)
	}

	fn base_blk(bl: usize, mode: Mode) -> Blk {
		Blk { bs: bs_of(bl), lw: 5 - bl, skip: false, mode }
	}

	/// Chooses and codes the block's modes.
	fn code_best_block(&mut self, bl: usize, x4: usize, y4: usize, thr: bool, lhb: bool) -> Rd {
		let n4 = 16 >> (bl - 1);
		let start = self.checkpoint(x4, y4, n4, None);
		let lambda = self.lambda;
		let mut best: Option<(f64, Blk)> = None;
		let consider = |rd: Rd, b: Blk, best: &mut Option<(f64, Blk)>| {
			let c = rd.cost(lambda);
			if best.is_none_or(|(x, _)| c < x) {
				*best = Some((c, b));
			}
		};
		let mut inter_satd = u32::MAX;
		if !self.key {
			let mut cands = self.inter_candidates(bl, x4, y4, thr);
			cands.sort_by_key(|c| c.0);
			inter_satd = cands.first().map_or(u32::MAX, |c| c.0);
			for &(_, mode) in cands.iter().take(self.search.inter_candidates) {
				let (rd, b) = self.code_block(Self::base_blk(bl, mode), x4, y4, thr, lhb);
				consider(rd, b, &mut best);
				self.restore(&start);
			}
		}
		let (y_modes, intra_satd) = self.rank_intra(bl, x4, y4, thr, lhb, 0, 0);
		if intra_satd < inter_satd.saturating_add(inter_satd / 4) || best.is_none() {
			let mk = |y: u8, uv: u8| Mode::Intra { y, uv, y_angle: 0, uv_angle: 0, txtp: DCT_DCT };
			// Luma (with DC chroma), then chroma for the best luma.
			let mut best_y: Option<(f64, Mode)> = None;
			for &y in y_modes.iter().take(self.search.intra_candidates) {
				for txtp in self.luma_types(bl) {
					let mode = match mk(y, 0) {
						Mode::Intra { y, uv, y_angle, uv_angle, .. } => {
							Mode::Intra { y, uv, y_angle, uv_angle, txtp }
						}
						m => m,
					};
					let (rd, _) = self.code_block(Self::base_blk(bl, mode), x4, y4, thr, lhb);
					let c = rd.cost(lambda);
					if best_y.is_none_or(|(x, _)| c < x) {
						best_y = Some((c, mode));
					}
					self.restore(&start);
				}
			}
			let Some((_, Mode::Intra { y, txtp, .. })) = best_y else { unreachable!() };
			let (uv_modes, _) = self.rank_intra(bl, x4, y4, thr, lhb, 1, y);
			for &uv in uv_modes.iter().take(self.search.intra_candidates) {
				let mode = Mode::Intra { y, uv, y_angle: 0, uv_angle: 0, txtp };
				let (rd, b) = self.code_block(Self::base_blk(bl, mode), x4, y4, thr, lhb);
				consider(rd, b, &mut best);
				self.restore(&start);
			}
		}
		let (_, b) = best.unwrap_or((
			0.0,
			Self::base_blk(bl, Mode::Intra { y: 0, uv: 0, y_angle: 0, uv_angle: 0, txtp: DCT_DCT }),
		));
		let (rd, b) = self.code_block(b, x4, y4, thr, lhb);
		self.decisions.push(Dec::Blk(b));
		rd
	}

	/// Luma transform types worth trying for an intra block.
	fn luma_types(&self, bl: usize) -> Vec<u8> {
		if 5 - bl >= 3 { vec![DCT_DCT] } else { INTRA_SET.to_vec() }
	}

	/// Intra modes for luma (`plane` 0) or chroma (1) ranked by SATD plus
	/// mode bits, and the best SATD. Blocks use one transform, so each
	/// prediction depends only on the block's neighbours.
	#[allow(clippy::too_many_arguments)]
	fn rank_intra(
		&mut self,
		bl: usize,
		x4: usize,
		y4: usize,
		thr: bool,
		lhb: bool,
		plane: usize,
		y_mode: u8,
	) -> (Vec<u8>, u32) {
		let lw = 5 - bl;
		let b =
			Self::base_blk(bl, Mode::Intra { y: y_mode, uv: 0, y_angle: 0, uv_angle: 0, txtp: DCT_DCT });
		let blocks: Vec<_> = self
			.tx_blocks(&b, x4, y4, thr, lhb)
			.into_iter()
			.filter(|t| (t.0 == 0) == (plane == 0))
			.collect();
		let mut scores: Vec<(f64, u8, u32)> = Vec::new();
		let modes: &[u8] = &Y_MODES;
		for &m in modes {
			let mut total = 0u32;
			for &(pl, x, y, tl, t, l) in &blocks {
				let pos = self.intra_pos(pl, x, y, tl, t, l);
				let n = 4usize << tl;
				let p = &mut self.rec[pl];
				let stride = p.stride;
				let (px, py) = (x * 4, y * 4);
				// Predict into the reconstruction, score, put it back.
				let saved: Vec<u8> = (0..n)
					.flat_map(|r| p.data[(py + r) * stride + px..(py + r) * stride + px + n].to_vec())
					.collect();
				ipred::predict(&mut p.data, stride, &pos, m, 0);
				let s = &self.src[pl];
				total +=
					satd(&s.data[py * s.stride + px..], s.stride, &p.data[py * stride + px..], stride, n, n);
				for r in 0..n {
					p.data[(py + r) * stride + px..(py + r) * stride + px + n]
						.copy_from_slice(&saved[r * n..(r + 1) * n]);
				}
			}
			// Mode bits (approximate: the mode symbol alone).
			let bits = {
				let mut c = Cost(0);
				let mut cdf = if plane == 0 {
					if self.key {
						let ca = INTRA_MODE_CTX[usize::from(self.ctx.a.mode[x4])];
						let cl = INTRA_MODE_CTX[usize::from(self.ctx.l.mode[y4 & 15])];
						self.cdf.kf_y_mode[usize::from(ca)][usize::from(cl)].to_vec()
					} else {
						self.cdf.y_mode[YMODE_SIZE_CTX[lw]].to_vec()
					}
				} else if lw <= 3 {
					self.cdf.uv_mode_cfl[usize::from(y_mode)].to_vec()
				} else {
					self.cdf.uv_mode_no_cfl[usize::from(y_mode)].to_vec()
				};
				let n = cdf.len() - 1;
				c.symbol(usize::from(m), &mut cdf, n);
				c.0
			};
			scores.push((f64::from(total) + self.sad_per_bit * f64::from(bits) / 256.0, m, total));
		}
		scores.sort_by(|a, b| a.0.total_cmp(&b.0));
		(scores.iter().map(|s| s.1).collect(), scores[0].2)
	}

	/// Luma SATD of predicting the block from `mv`.
	fn luma_satd(&mut self, mv: Mv, x4: usize, y4: usize, n4: usize) -> u32 {
		let Some(r) = self.refp.clone() else { return u32::MAX };
		let (x, y, n) = (x4 * 4, y4 * 4, n4 * 4);
		mc::predict(
			&r.planes[0].as_ref(),
			&mut self.mcbuf,
			64,
			0,
			0,
			n,
			n,
			(i32::from(mv.y) + (y as i32) * 8, i32::from(mv.x) + (x as i32) * 8),
			false,
		);
		let s = &self.src[0];
		satd(&s.data[y * s.stride + x..], s.stride, &self.mcbuf, 64, n, n)
	}

	fn mv_bits(&mut self, mv: Mv, pred: Mv) -> u32 {
		let mut c = Cost(0);
		// The cost model leaves the CDFs alone.
		Self::write_mv(
			&mut c,
			&mut self.cdf,
			Mv::new(i32::from(mv.y) - i32::from(pred.y), i32::from(mv.x) - i32::from(pred.x)),
		);
		c.0
	}

	/// Inter candidates as (SATD + bits, mode).
	fn inter_candidates(&mut self, bl: usize, x4: usize, y4: usize, thr: bool) -> Vec<(u32, Mode)> {
		let n4 = 1usize << (5 - bl);
		let b = Self::base_blk(bl, Mode::Inter { mode: GLOBALMV, drl: 0, mv: Mv::ZERO });
		let (stack, n_mvs, _) = self.mv_stack(&b, x4, y4, thr);
		let mut list: Vec<Mode> = vec![
			Mode::Inter { mode: GLOBALMV, drl: 0, mv: Mv::ZERO },
			Mode::Inter { mode: NEARESTMV, drl: 0, mv: Self::mode_mv(&stack, n_mvs, NEARESTMV, 0) },
		];
		for drl in 1..n_mvs.clamp(2, 4) as u8 {
			list.push(Mode::Inter { mode: NEARMV, drl, mv: Self::mode_mv(&stack, n_mvs, NEARMV, drl) });
		}
		// NEWMV: search from the first predictor, then the cheapest predictor.
		let preds: Vec<(u8, Mv)> =
			(0..n_mvs.clamp(1, 3) as u8).map(|d| (d, Self::mode_mv(&stack, n_mvs, NEWMV, d))).collect();
		let new = self.motion_search(
			x4,
			y4,
			n4,
			preds[0].1,
			&[Self::mode_mv(&stack, n_mvs, NEARESTMV, 0), Mv::ZERO],
		);
		let (drl, _) =
			preds.iter().copied().min_by_key(|&(_, p)| self.mv_bits(new, p)).unwrap_or((0, Mv::ZERO));
		list.push(Mode::Inter { mode: NEWMV, drl, mv: new });
		let mut out = Vec::new();
		let mut seen: Vec<Mv> = Vec::new();
		for m in list {
			let Mode::Inter { mv, .. } = m else { continue };
			if seen.contains(&mv) || !mv_ok(mv) {
				continue;
			}
			seen.push(mv);
			let s = self.luma_satd(mv, x4, y4, n4);
			let mut c = Cost(0);
			let blk = Blk { skip: false, ..Self::base_blk(bl, m) };
			self.write_mode_info(&mut c, &blk, x4, y4, thr);
			out.push(((f64::from(s) + self.sad_per_bit * f64::from(c.0) / 256.0) as u32, m));
		}
		out
	}

	/// Full-pel diamond search, then half- and quarter-pel refinement.
	fn motion_search(&mut self, x4: usize, y4: usize, n4: usize, pred: Mv, starts: &[Mv]) -> Mv {
		let Some((padded, pstride)) = self.pad_ref.take() else { return pred };
		let (x0, y0) = ((x4 * 4) as i32, (y4 * 4) as i32);
		let n = n4 * 4;
		let ss = self.src[0].stride;
		let src_at = y0 as usize * ss + x0 as usize;
		let (fw, fh) = ((self.bw * 4) as i32, (self.bh * 4) as i32);
		let range = self.search.range;
		let (min_x, max_x) =
			((-x0 - BORDER + 8).max(-range * 2), (fw - x0 + BORDER - n as i32 - 8).min(range * 2));
		let (min_y, max_y) =
			((-y0 - BORDER + 8).max(-range * 2), (fh - y0 + BORDER - n as i32 - 8).min(range * 2));
		let cost_at = |enc: &mut Self, dx: i32, dy: i32| -> f64 {
			let at = (y0 + dy + BORDER) as usize * pstride + (x0 + dx + BORDER) as usize;
			let s = sad(&enc.src[0].data[src_at..], ss, &padded[at..], pstride, n, n);
			f64::from(s) + enc.sad_per_bit * f64::from(enc.mv_bits(Mv::new(dy * 8, dx * 8), pred)) / 256.0
		};
		let mut best = (0, 0);
		let mut best_cost = f64::MAX;
		for &m in std::iter::once(&pred).chain(starts) {
			let (dx, dy) = (
				((i32::from(m.x) + 4) >> 3).clamp(min_x, max_x),
				((i32::from(m.y) + 4) >> 3).clamp(min_y, max_y),
			);
			let c = cost_at(self, dx, dy);
			if c < best_cost {
				best_cost = c;
				best = (dx, dy);
			}
		}
		let mut step = 16;
		while step >= 1 {
			let mut moved = true;
			while moved {
				moved = false;
				let (bx, by) = best;
				for (ox, oy) in [(step, 0), (-step, 0), (0, step), (0, -step)] {
					let (dx, dy) = (bx + ox, by + oy);
					if dx < min_x || dx > max_x || dy < min_y || dy > max_y {
						continue;
					}
					let c = cost_at(self, dx, dy);
					if c < best_cost {
						best_cost = c;
						best = (dx, dy);
						moved = true;
					}
				}
			}
			step /= 2;
		}
		self.pad_ref = Some((padded, pstride));
		let mut mv = Mv::new(best.1 * 8, best.0 * 8);
		let mut cost = f64::from(self.luma_satd(mv, x4, y4, n4))
			+ self.sad_per_bit * f64::from(self.mv_bits(mv, pred)) / 256.0;
		for d in [4i32, 2] {
			let center = mv;
			for (oy, ox) in [(-d, -d), (-d, 0), (-d, d), (0, -d), (0, d), (d, -d), (d, 0), (d, d)] {
				let m = Mv::new(i32::from(center.y) + oy, i32::from(center.x) + ox);
				let c = f64::from(self.luma_satd(m, x4, y4, n4))
					+ self.sad_per_bit * f64::from(self.mv_bits(m, pred)) / 256.0;
				if c < cost {
					cost = c;
					mv = m;
				}
			}
		}
		mv
	}
}

/// Motion vectors the syntax can carry.
fn mv_ok(mv: Mv) -> bool {
	let ok = |v: i16| i32::from(v).abs() < (1 << 14) - 1;
	ok(mv.x) && ok(mv.y)
}
