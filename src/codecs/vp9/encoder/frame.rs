//! One frame: the rate-distortion search over partitions and modes, with
//! reconstruction.
//!
//! The encoder works on the decoder's own frame state ([`FrameDecoder`]):
//! the same mode-info array, motion-vector candidates, non-zero and
//! partition contexts, picture buffer and loop-filter cells. Every
//! candidate is coded for real — predicted with the decoder's
//! `intra::predict` / `inter::predict`, its residual transformed,
//! quantised, priced with the token writer and reconstructed with the
//! decoder's `inverse_transform_add` — and the state is rolled back
//! ([`Checkpoint`]) when a candidate loses. What is left once a superblock
//! is decided is exactly what a decoder will have reconstructed.
//!
//! Search (libvpx `vp9_rdopt.c` / `vp9_encodeframe.c` in spirit):
//!
//! - **Partition:** each square from 64x64 down to 8x8 is coded whole and
//!   split into four; the cheaper of `D + λ·R` wins.
//! - **Intra:** all ten luma modes ranked by SATD (neighbours taken from
//!   the source inside the block), the best few fully coded; the same for
//!   the chroma mode.
//! - **Inter (LAST):** NEARESTMV, NEARMV, ZEROMV and NEWMV from a full-pel
//!   diamond search plus half- and quarter-pel refinement, ranked by SATD
//!   plus motion-vector bits, the best two fully coded; residual or skip.

use std::sync::Arc;

use super::super::decoder::{BlockPos, FrameDecoder, Picture};
use super::super::inter::{self, RefPlane};
use super::super::intra::{self, Edges};
use super::super::mode::ModeInfo;
use super::super::residual;
use super::super::transform;
use super::super::{INTRA_MODE_TREE, LAST_FRAME, MAX_TX_SIZE, Mv, NO_REF, SIZE_GROUP};
use super::super::{SUBSIZE, bs, pm, tables};
use super::fdct;
use super::pack::MAX_LEVEL;
use super::writer::{Cost, tree_cost};

/// A decision, in coding order (what the pack pass replays).
#[derive(Debug, Clone, Copy)]
pub enum Decision {
	Partition(u8),
	Block(ModeInfo),
}

/// The picture being coded, its planes padded to whole superblocks with
/// the edge samples repeated, on the reconstruction's strides.
pub struct Source {
	pub planes: [Vec<u8>; 3],
	pub stride: [usize; 3],
	pub width: [usize; 3],
	pub height: [usize; 3],
}

impl Source {
	/// From planes of `width` x `height` 4:2:0 samples.
	pub fn new(planes: [(&[u8], usize); 3], width: usize, height: usize) -> Self {
		let (sw, sh) = (width.next_multiple_of(64), height.next_multiple_of(64));
		let mut out = Source {
			planes: [Vec::new(), Vec::new(), Vec::new()],
			stride: [0; 3],
			width: [0; 3],
			height: [0; 3],
		};
		for (i, &(data, stride)) in planes.iter().enumerate() {
			let s = usize::from(i > 0);
			let (w, h) = ((width + s) >> s, (height + s) >> s);
			let (pw, ph) = (sw >> s, sh >> s);
			let mut plane = vec![0u8; pw * ph];
			for y in 0..ph {
				let row = &data[y.min(h - 1) * stride..];
				let out_row = &mut plane[y * pw..(y + 1) * pw];
				out_row[..w].copy_from_slice(&row[..w]);
				let last = row[w - 1];
				out_row[w..].fill(last);
			}
			out.planes[i] = plane;
			out.stride[i] = pw;
			out.width[i] = w;
			out.height[i] = h;
		}
		out
	}
}

/// Rate and distortion of a coded part (rate in 1/256 bit, distortion as
/// the sum of squared errors over visible samples).
#[derive(Debug, Clone, Copy, Default)]
pub struct Rd {
	pub dist: u64,
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

/// The reference's luma with a wide repeated border, for full-pel search.
struct PaddedLuma {
	data: Vec<u8>,
	stride: usize,
}

/// Border of [`PaddedLuma`]; full-pel candidates stay this far inside it.
const BORDER: i32 = 96;
const MARGIN: i32 = 8;

/// Saved state of a square region (for trying alternatives).
pub struct Checkpoint {
	x8: usize,
	y8: usize,
	n8: usize,
	pix: [Vec<u8>; 3],
	above_nz: [Vec<u8>; 3],
	left_nz: [[u8; 16]; 3],
	above_partition: Vec<u8>,
	left_partition: [u8; 8],
	mi: Vec<ModeInfo>,
	mvs: Vec<super::super::mode::MvRef>,
	decisions_len: usize,
	tokens_len: usize,
	decisions: Vec<Decision>,
	tokens: Vec<i16>,
}

/// Encoder settings that shape the search.
#[derive(Debug, Clone, Copy)]
pub struct Search {
	/// Luma (and chroma) intra modes fully coded after the SATD ranking.
	pub intra_candidates: usize,
	/// Inter candidates fully coded.
	pub inter_candidates: usize,
	/// Full-pel search range (pixels).
	pub range: i32,
}

impl Default for Search {
	fn default() -> Self {
		Self { intra_candidates: 3, inter_candidates: 2, range: 64 }
	}
}

pub struct FrameEncoder {
	pub fd: FrameDecoder<u8>,
	pub src: Source,
	/// SSE per bit.
	pub lambda: f64,
	/// SAD / SATD per bit.
	pub sad_per_bit: f64,
	pub decisions: Vec<Decision>,
	/// Per coded transform block: the end of block, then its levels in scan
	/// order.
	pub tokens: Vec<i16>,
	/// Pack-pass read positions in `decisions` and `tokens`.
	pub cursor: (usize, usize),
	pub search: Search,
	ref_luma: Option<PaddedLuma>,
	res: Vec<i32>,
	coef: Vec<f32>,
	levels: Vec<i16>,
	pred: [Vec<u8>; 3],
	mc: Vec<u8>,
}

/// 4x4 Hadamard SATD of an `n x n` block (x264's, halved).
fn satd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
	let mut total = 0u32;
	for by in (0..h).step_by(4) {
		for bx in (0..w).step_by(4) {
			let mut d = [0i32; 16];
			for y in 0..4 {
				for x in 0..4 {
					let i = (by + y) * a_stride + bx + x;
					let j = (by + y) * b_stride + bx + x;
					d[y * 4 + x] = i32::from(a[i]) - i32::from(b[j]);
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
	let mut s = 0u32;
	for y in 0..h {
		let ra = &a[y * a_stride..y * a_stride + w];
		let rb = &b[y * b_stride..y * b_stride + w];
		s += ra.iter().zip(rb).map(|(&x, &y)| u32::from(x.abs_diff(y))).sum::<u32>();
	}
	s
}

impl FrameEncoder {
	pub fn new(
		fd: FrameDecoder<u8>,
		src: Source,
		lambda: f64,
		sad_per_bit: f64,
		search: Search,
	) -> Self {
		let ref_luma = fd.refs[0].as_ref().map(|r| {
			let p = &r.planes[0];
			let stride = p.stride + 2 * BORDER as usize;
			let rows = (fd.mi_rows * 8).next_multiple_of(64) + 2 * BORDER as usize;
			let mut data = vec![0u8; stride * rows];
			for y in 0..rows {
				let sy = (y as i32 - BORDER).clamp(0, p.height as i32 - 1) as usize;
				let row = &p.data[sy * p.stride..];
				for x in 0..stride {
					let sx = (x as i32 - BORDER).clamp(0, p.width as i32 - 1) as usize;
					data[y * stride + x] = row[sx];
				}
			}
			PaddedLuma { data, stride }
		});
		Self {
			fd,
			src,
			lambda,
			sad_per_bit,
			decisions: Vec::new(),
			tokens: Vec::new(),
			cursor: (0, 0),
			search,
			ref_luma,
			res: vec![0; 32 * 32],
			coef: vec![0.0; 32 * 32],
			levels: Vec::with_capacity(1024),
			pred: [vec![0; 64 * 64], vec![0; 32 * 32], vec![0; 32 * 32]],
			mc: vec![0; 64 * 64],
		}
	}

	// --- State save / restore ---

	pub fn checkpoint(
		&self,
		mi_row: usize,
		mi_col: usize,
		n8: usize,
		with_tail: Option<&Checkpoint>,
	) -> Checkpoint {
		let fd = &self.fd;
		let pix = std::array::from_fn(|plane| {
			let (sx, sy) = fd.plane_ss(plane);
			let pl = &fd.pic.planes[plane];
			let (x, y, n) = ((mi_col * 8) >> sx, (mi_row * 8) >> sy, (n8 * 8) >> sx);
			let mut v = Vec::with_capacity(n * n);
			for r in 0..n {
				v.extend_from_slice(&pl.data[(y + r) * pl.stride + x..(y + r) * pl.stride + x + n]);
			}
			v
		});
		let above_nz = std::array::from_fn(|plane| {
			let (sx, _) = fd.plane_ss(plane);
			let a = &fd.above_nz[plane];
			let x = (mi_col * 2) >> sx;
			a[x..(x + ((n8 * 2) >> sx)).min(a.len())].to_vec()
		});
		let (rows, cols) = (n8.min(fd.mi_rows - mi_row), n8.min(fd.mi_cols - mi_col));
		let mut mi = Vec::with_capacity(rows * cols);
		let mut mvs = Vec::with_capacity(rows * cols);
		for r in 0..rows {
			let at = (mi_row + r) * fd.mi_cols + mi_col;
			mi.extend_from_slice(&fd.mi[at..at + cols]);
			mvs.extend_from_slice(&fd.cur_mvs[at..at + cols]);
		}
		let (decisions_len, tokens_len) = match with_tail {
			Some(base) => (base.decisions_len, base.tokens_len),
			None => (self.decisions.len(), self.tokens.len()),
		};
		Checkpoint {
			x8: mi_col,
			y8: mi_row,
			n8,
			pix,
			above_nz,
			left_nz: fd.left_nz,
			above_partition: fd.above_partition[mi_col..mi_col + n8].to_vec(),
			left_partition: fd.left_partition,
			mi,
			mvs,
			decisions_len,
			tokens_len,
			decisions: self.decisions[decisions_len..].to_vec(),
			tokens: self.tokens[tokens_len..].to_vec(),
		}
	}

	pub fn restore(&mut self, c: &Checkpoint) {
		let (mi_row, mi_col, n8) = (c.y8, c.x8, c.n8);
		let fd = &mut self.fd;
		for plane in 0..3 {
			let (sx, sy) = fd.plane_ss(plane);
			let pl = &mut fd.pic.planes[plane];
			let (x, y, n) = ((mi_col * 8) >> sx, (mi_row * 8) >> sy, (n8 * 8) >> sx);
			for r in 0..n {
				let at = (y + r) * pl.stride + x;
				pl.data[at..at + n].copy_from_slice(&c.pix[plane][r * n..(r + 1) * n]);
			}
			let x = (mi_col * 2) >> sx;
			let a = &mut fd.above_nz[plane];
			let len = c.above_nz[plane].len();
			a[x..x + len].copy_from_slice(&c.above_nz[plane]);
		}
		fd.left_nz = c.left_nz;
		fd.above_partition[mi_col..mi_col + n8].copy_from_slice(&c.above_partition);
		fd.left_partition = c.left_partition;
		let (rows, cols) = (n8.min(fd.mi_rows - mi_row), n8.min(fd.mi_cols - mi_col));
		for r in 0..rows {
			let at = (mi_row + r) * fd.mi_cols + mi_col;
			fd.mi[at..at + cols].copy_from_slice(&c.mi[r * cols..(r + 1) * cols]);
			fd.cur_mvs[at..at + cols].copy_from_slice(&c.mvs[r * cols..(r + 1) * cols]);
		}
		self.decisions.truncate(c.decisions_len);
		self.decisions.extend_from_slice(&c.decisions);
		self.tokens.truncate(c.tokens_len);
		self.tokens.extend_from_slice(&c.tokens);
	}

	// --- Superblocks and partitions ---

	/// Searches and codes one superblock row of the current tile.
	pub fn analyse_sb_row(&mut self, mi_row: usize) {
		self.fd.left_nz = [[0; 16]; 3];
		self.fd.left_partition = [0; 8];
		let mut mi_col = self.fd.tile.mi_col_start;
		while mi_col < self.fd.tile.mi_col_end {
			self.search_partition(mi_row, mi_col, bs::B64X64, 3);
			mi_col += 8;
		}
	}

	fn partition_cost(&mut self, mi_row: usize, mi_col: usize, bsl: usize, partition: u8) -> u32 {
		let mut c = Cost(0);
		self.write_partition(&mut c, mi_row, mi_col, bsl, partition);
		c.0
	}

	/// Codes the block at (`mi_row`, `mi_col`) of `bsize` (`1 << bsl` 8x8
	/// units) the cheapest way; returns its cost.
	fn search_partition(&mut self, mi_row: usize, mi_col: usize, bsize: u8, bsl: usize) -> Rd {
		let fd = &self.fd;
		if mi_row >= fd.mi_rows || mi_col >= fd.mi_cols {
			return Rd::default();
		}
		let hbs = (1usize << bsl) >> 1;
		let has_rows = mi_row + hbs < fd.mi_rows;
		let has_cols = mi_col + hbs < fd.mi_cols;
		let can_none = bsize == bs::B8X8 || (has_rows && has_cols);
		let can_split = bsize > bs::B8X8;
		let n8 = 1usize << bsl;
		let before = (can_none && can_split).then(|| self.checkpoint(mi_row, mi_col, n8, None));

		let mut none = None;
		if can_none {
			let mut rd = Rd { dist: 0, rate: self.partition_cost(mi_row, mi_col, bsl, 0) };
			self.decisions.push(Decision::Partition(0));
			rd.add(self.code_best_block(mi_row, mi_col, bsize));
			self.set_partition_ctx(mi_row, mi_col, bsize);
			if !can_split {
				return rd;
			}
			// A block coded without residual and with little error is not
			// worth splitting.
			let area = (n8 * 8 * n8 * 8) as f64;
			let mi = self.fd.mi[mi_row * self.fd.mi_cols + mi_col];
			if mi.skip && mi.is_inter() && (rd.dist as f64) < area * self.lambda * 0.02 {
				return rd;
			}
			none = Some(rd);
		}
		let after_none = before.as_ref().map(|b| self.checkpoint(mi_row, mi_col, n8, Some(b)));
		if let Some(b) = &before {
			self.restore(b);
		}
		let limit = none.map(|r| r.cost(self.lambda));
		let mut rd = Rd { dist: 0, rate: self.partition_cost(mi_row, mi_col, bsl, 3) };
		self.decisions.push(Decision::Partition(3));
		let sub = SUBSIZE[3][usize::from(bsize)];
		for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
			rd.add(self.search_partition(mi_row + dr, mi_col + dc, sub, bsl - 1));
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

	// --- Blocks ---

	fn base_mi(&self, bsize: u8) -> ModeInfo {
		let tx_mode_max = super::super::header::tx_mode_max(self.fd.ch.tx_mode) as u8;
		ModeInfo {
			sb_type: bsize,
			tx_size: MAX_TX_SIZE[usize::from(bsize)].min(tx_mode_max),
			..ModeInfo::default()
		}
	}

	/// Transform sizes worth trying for a block, largest first.
	fn tx_sizes(&self, bsize: u8) -> std::iter::Rev<std::ops::RangeInclusive<u8>> {
		let max = MAX_TX_SIZE[usize::from(bsize)];
		if self.fd.ch.tx_mode == super::super::header::TX_MODE_SELECT {
			(max.saturating_sub(2)..=max).rev()
		} else {
			(max..=max).rev()
		}
	}

	/// Cost of the transform size of an intra block (counting is off).
	fn tx_size_cost(&mut self, p: &BlockPos, mi: &ModeInfo) -> u32 {
		let mut c = Cost(0);
		let mut m = *mi;
		m.skip = false;
		self.write_mode_info(&mut c, p, &m);
		let mut d = Cost(0);
		let mut n = m;
		n.tx_size = MAX_TX_SIZE[usize::from(mi.sb_type)];
		self.write_mode_info(&mut d, p, &n);
		c.0.saturating_sub(d.0)
	}

	/// Cost of a luma intra mode symbol for the block at `p`.
	fn y_mode_cost(&self, p: &BlockPos, bsize: u8, mode: u8) -> u32 {
		if self.fd.h.intra_only_frame() {
			let a = self.fd.above_mi(p).filter(|m| !m.is_inter()).map_or(pm::DC, |m| m.y_mode(2));
			let l = self.fd.left_mi(p).filter(|m| !m.is_inter()).map_or(pm::DC, |m| m.y_mode(1));
			tree_cost(&INTRA_MODE_TREE, &tables::KF_Y_MODE_PROBS[usize::from(a)][usize::from(l)], mode)
		} else {
			let group = usize::from(SIZE_GROUP[usize::from(bsize)]);
			tree_cost(&INTRA_MODE_TREE, &self.fd.fc.y_mode[group], mode)
		}
	}

	fn uv_mode_cost(&self, y_mode: u8, uv_mode: u8) -> u32 {
		let probs = if self.fd.h.intra_only_frame() {
			&tables::KF_UV_MODE_PROBS[usize::from(y_mode)]
		} else {
			&self.fd.fc.uv_mode[usize::from(y_mode)]
		};
		tree_cost(&INTRA_MODE_TREE, probs, uv_mode)
	}

	/// Chooses and codes the block's modes; leaves it reconstructed and its
	/// decision recorded.
	fn code_best_block(&mut self, mi_row: usize, mi_col: usize, bsize: u8) -> Rd {
		let p = self.pos(mi_row, mi_col, bsize);
		let n8 = usize::from(super::super::NUM_8X8_WIDE[usize::from(bsize)]);
		let start = self.checkpoint(mi_row, mi_col, n8, None);
		let mut best: Option<(f64, ModeInfo)> = None;
		let lambda = self.lambda;
		let consider = |rd: Rd, mi: ModeInfo, best: &mut Option<(f64, ModeInfo)>| {
			let c = rd.cost(lambda);
			if best.is_none_or(|(b, _)| c < b) {
				*best = Some((c, mi));
			}
		};

		let mut inter_satd = u32::MAX;
		if !self.fd.h.intra_only_frame() {
			let mut cands = self.inter_candidates(&p, bsize);
			cands.sort_by_key(|c| c.0);

			inter_satd = cands.first().map_or(u32::MAX, |c| c.0);
			for &(_, mode, mv) in cands.iter().take(self.search.inter_candidates) {
				let mut mi = self.base_mi(bsize);
				mi.mode = mode;
				mi.ref_frame = [LAST_FRAME, NO_REF];
				mi.mv = [mv, Mv::ZERO];
				mi.sub_mv = [mi.mv; 4];
				mi.interp_filter = self.fd.h.interp_filter;
				let mut prev = f64::MAX;
				for tx in self.tx_sizes(bsize) {
					mi.tx_size = tx;
					let (rd, coded) = self.code_block(&p, mi);
					self.restore(&start);
					let c = rd.cost(lambda);
					consider(rd, coded, &mut best);
					// Smaller transforms only while they help; a skipped
					// block has no transform.
					if c >= prev || coded.skip {
						break;
					}
					prev = c;
				}
			}
		}
		let (y_modes, intra_satd) = self.rank_intra_modes(&p, bsize, 0, pm::DC);
		if intra_satd < inter_satd.saturating_add(inter_satd / 4) || best.is_none() {
			let mut base = self.base_mi(bsize);
			// Luma first.
			let mut best_y: Option<(f64, u8, u8)> = None;
			for &mode in y_modes.iter().take(self.search.intra_candidates) {
				base.mode = mode;
				let mut prev = f64::MAX;
				for tx in self.tx_sizes(bsize) {
					base.tx_size = tx;
					let mut rd = self.code_plane(&p, &base, 0, true);
					rd.rate += self.y_mode_cost(&p, bsize, mode) + self.tx_size_cost(&p, &base);
					let c = rd.cost(lambda);
					if best_y.is_none_or(|(b, _, _)| c < b) {
						best_y = Some((c, mode, tx));
					}
					self.restore(&start);
					if c >= prev {
						break;
					}
					prev = c;
				}
			}
			(base.mode, base.tx_size) = best_y.map_or((pm::DC, base.tx_size), |b| (b.1, b.2));
			// Then chroma, over the chosen luma.
			let (uv_modes, _) = self.rank_intra_modes(&p, bsize, 1, base.mode);
			let mut best_uv: Option<(f64, u8)> = None;
			if uv_modes.len() > 1 {
				for &mode in uv_modes.iter().take(self.search.intra_candidates) {
					base.uv_mode = mode;
					let mut rd = self.code_plane(&p, &base, 1, true);
					rd.add(self.code_plane(&p, &base, 2, true));
					rd.rate += self.uv_mode_cost(base.mode, mode);
					let c = rd.cost(lambda);
					if best_uv.is_none_or(|(b, _)| c < b) {
						best_uv = Some((c, mode));
					}
					self.restore(&start);
				}
			}
			base.uv_mode = best_uv.map_or(uv_modes[0], |b| b.1);
			let (rd, mi) = self.code_block(&p, base);
			consider(rd, mi, &mut best);
			self.restore(&start);
		}

		let (_, mi) = best.unwrap_or((0.0, self.base_mi(bsize)));
		let (rd, mi) = self.code_block(&p, mi);
		self.decisions.push(Decision::Block(mi));
		self.commit_mi(&p, &mi);
		rd
	}

	/// Codes a block with the given modes (prediction, residual or skip,
	/// mode info); returns its cost and final mode info.
	fn code_block(&mut self, p: &BlockPos, mut mi: ModeInfo) -> (Rd, ModeInfo) {
		let tokens_at = self.tokens.len();
		let mut rd = Rd::default();
		if mi.is_inter() {
			self.predict_inter(p, &mi);
			let pred_dist = self.save_prediction(p);
			let mut coded = Rd::default();
			for plane in 0..3 {
				coded.add(self.code_plane(p, &mi, plane, false));
			}
			mi.skip = !self.has_coefficients(tokens_at);
			let coded_cost = {
				let mut c = Cost(0);
				self.write_mode_info(&mut c, p, &mi);
				Rd { dist: coded.dist, rate: coded.rate + c.0 }
			};
			let mut skipped = mi;
			skipped.skip = true;
			skipped.tx_size = MAX_TX_SIZE[usize::from(mi.sb_type)];
			let skip_cost = {
				let mut c = Cost(0);
				self.write_mode_info(&mut c, p, &skipped);
				Rd { dist: pred_dist, rate: c.0 }
			};
			if mi.skip || skip_cost.cost(self.lambda) <= coded_cost.cost(self.lambda) {
				mi = skipped;
				self.restore_prediction(p);
				self.tokens.truncate(tokens_at);
				self.clear_nz(p);
				rd = skip_cost;
			} else {
				rd = coded_cost;
			}
		} else {
			for plane in 0..3 {
				rd.add(self.code_plane(p, &mi, plane, true));
			}
			mi.skip = !self.has_coefficients(tokens_at);
			if mi.skip {
				self.tokens.truncate(tokens_at);
			}
			let mut c = Cost(0);
			self.write_mode_info(&mut c, p, &mi);
			rd.rate += c.0;
		}
		if mi.skip {
			// What the decoder does for a skipped block (the token rate was
			// counted with the contexts as they were before).
			self.clear_nz(p);
		}
		(rd, mi)
	}

	fn has_coefficients(&self, from: usize) -> bool {
		let mut at = from;
		while at < self.tokens.len() {
			let eob = self.tokens[at] as usize;
			if eob > 0 {
				return true;
			}
			at += 1 + eob;
		}
		false
	}

	/// Copies the block's prediction (all planes) aside; returns its
	/// distortion.
	fn save_prediction(&mut self, p: &BlockPos) -> u64 {
		let mut dist = 0;
		for plane in 0..3 {
			let (sx, sy) = self.fd.plane_ss(plane);
			let (x, y) = ((p.col * 8) >> sx, (p.row * 8) >> sy);
			let n = (p.bw * 8) >> sx;
			let pl = &self.fd.pic.planes[plane];
			for r in 0..n {
				let at = (y + r) * pl.stride + x;
				self.pred[plane][r * n..(r + 1) * n].copy_from_slice(&pl.data[at..at + n]);
			}
			dist += self.ssd(plane, x, y, n, n);
		}
		dist
	}

	fn restore_prediction(&mut self, p: &BlockPos) {
		for plane in 0..3 {
			let (sx, sy) = self.fd.plane_ss(plane);
			let (x, y) = ((p.col * 8) >> sx, (p.row * 8) >> sy);
			let n = (p.bw * 8) >> sx;
			let pl = &mut self.fd.pic.planes[plane];
			for r in 0..n {
				let at = (y + r) * pl.stride + x;
				pl.data[at..at + n].copy_from_slice(&self.pred[plane][r * n..(r + 1) * n]);
			}
		}
	}

	/// Squared error between source and reconstruction over the visible
	/// part of a rectangle.
	fn ssd(&self, plane: usize, x: usize, y: usize, w: usize, h: usize) -> u64 {
		let (pw, ph) = (self.src.width[plane], self.src.height[plane]);
		if x >= pw || y >= ph {
			return 0;
		}
		let (w, h) = (w.min(pw - x), h.min(ph - y));
		let pl = &self.fd.pic.planes[plane];
		let s = &self.src.planes[plane];
		let ss = self.src.stride[plane];
		let mut sum = 0u64;
		for r in 0..h {
			let a = &s[(y + r) * ss + x..(y + r) * ss + x + w];
			let b = &pl.data[(y + r) * pl.stride + x..(y + r) * pl.stride + x + w];
			sum += a.iter().zip(b).map(|(&u, &v)| u64::from(u.abs_diff(v)).pow(2)).sum::<u64>();
		}
		sum
	}

	fn edges(&self, p: &BlockPos, plane: usize, col: usize, row: usize, tx_size: u8) -> Edges {
		let (sx, _) = self.fd.plane_ss(plane);
		let n4_wl = (p.bw * 2).trailing_zeros() as usize - sx;
		let step = 1usize << tx_size;
		let pl = &self.fd.pic.planes[plane];
		Edges {
			have_top: row > 0 || p.have_above,
			have_left: col > 0 || p.have_left,
			have_right: col + step < (1 << n4_wl),
			clip_right: p.to_right < 0,
			clip_bottom: p.to_bottom < 0,
			frame_width: pl.aligned_w,
			frame_height: pl.aligned_h,
		}
	}

	/// Codes one plane of a block, transform block by transform block
	/// (intra prediction first when `intra`); appends the tokens, updates
	/// the contexts and reconstructs.
	fn code_plane(&mut self, p: &BlockPos, mi: &ModeInfo, plane: usize, intra: bool) -> Rd {
		let mut rd = Rd::default();
		let (sx, sy) = self.fd.plane_ss(plane);
		let (x0, y0) = ((p.col * 8) >> sx, (p.row * 8) >> sy);
		let mode = if plane == 0 { mi.mode } else { mi.uv_mode };
		let lambda = self.lambda;
		let inter_round = !intra;
		self.for_each_tx(p, mi, plane, |enc, col, row, tx_size, tx_type, clip| {
			let (x, y) = (x0 + col * 4, y0 + row * 4);
			let n = 4usize << tx_size;
			if intra {
				let e = enc.edges(p, plane, col, row, tx_size);
				let pl = &mut enc.fd.pic.planes[plane];
				let stride = pl.stride;
				intra::predict(&mut pl.data, stride, x, y, tx_size, mode, &e, 8);
			}
			// Residual (zero outside the picture).
			let (pw, ph) = (enc.src.width[plane], enc.src.height[plane]);
			let ss = enc.src.stride[plane];
			let pl = &enc.fd.pic.planes[plane];
			let mut energy = 0u64;
			for r in 0..n {
				for c in 0..n {
					let v = if x + c < pw && y + r < ph {
						i32::from(enc.src.planes[plane][(y + r) * ss + x + c])
							- i32::from(pl.data[(y + r) * pl.stride + x + c])
					} else {
						0
					};
					enc.res[r * n + c] = v;
					energy += (v * v) as u64;
				}
			}
			let ctx = enc.token_ctx(p, plane, col, row, tx_size);
			let plane_type = usize::from(plane > 0);
			let reference = usize::from(mi.is_inter());
			let sc = residual::scan(tx_size, tx_type);
			let dq = enc.fd.dequant[usize::from(mi.segment_id)][plane_type];
			enc.levels.clear();
			let mut eob = 0;
			let mut qerr = 0f64;
			if energy > 0 {
				fdct::forward(&enc.res[..n * n], &mut enc.coef[..n * n], tx_size, tx_type);
				let shift = i32::from(tx_size == 3);
				let round: f32 = if inter_round { 0.30 } else { 0.38 };
				for (k, &pos) in sc.scan.iter().enumerate().take(n * n) {
					let pos = usize::from(pos);
					let q = i32::from(if k == 0 { dq[0] } else { dq[1] });
					let c = enc.coef[pos];
					let step = q as f32 / (1 << shift) as f32;
					let level = ((c.abs() / step) + round).floor().min(MAX_LEVEL as f32) as i32;
					let level = if c < 0.0 { -level } else { level };
					let rec = ((level.abs() * q) >> shift) as f32 * level.signum() as f32;
					qerr += f64::from((c - rec) * (c - rec));
					enc.levels.push(level as i16);
					if level != 0 {
						eob = k + 1;
					}
				}
				enc.levels.truncate(eob);
			}
			// Drop the block if its coefficients cost more than they save
			// (distortion estimated in the transform domain).
			if eob > 0 {
				let probs = &enc.fd.fc.coef[usize::from(tx_size)][plane_type][reference];
				let mut coded = Cost(0);
				super::pack::write_coefs(
					&mut coded,
					probs,
					None,
					&enc.levels,
					tx_size,
					ctx,
					&sc,
					residual::cat6(8),
				);
				let mut none = Cost(0);
				super::pack::write_coefs(&mut none, probs, None, &[], tx_size, ctx, &sc, residual::cat6(8));
				let scale = if tx_size == 3 { 16.0 } else { 64.0 };
				let d_coded = qerr / scale;
				if energy as f64 + lambda * f64::from(none.0) / 256.0
					<= d_coded + lambda * f64::from(coded.0) / 256.0
				{
					eob = 0;
					enc.levels.clear();
				}
			}
			let mut cost = Cost(0);
			let levels = std::mem::take(&mut enc.levels);
			enc.write_tokens(&mut cost, p, mi, plane, col, row, tx_size, tx_type, clip, &levels);
			rd.rate += cost.0;
			enc.tokens.push(eob as i16);
			enc.tokens.extend_from_slice(&levels);
			if eob > 0 {
				let shift = u32::from(tx_size == 3);
				for (k, &l) in levels.iter().enumerate() {
					if l != 0 {
						let q = i32::from(if k == 0 { dq[0] } else { dq[1] });
						let v = (i32::from(l.unsigned_abs()) * q) >> shift;
						enc.fd.coefs[usize::from(sc.scan[k])] = if l < 0 { -v } else { v };
					}
				}
				let pl = &mut enc.fd.pic.planes[plane];
				let stride = pl.stride;
				transform::inverse_transform_add(
					&enc.fd.coefs[..n * n],
					&mut pl.data[y * stride + x..],
					stride,
					tx_size,
					tx_type,
					false,
					eob,
					255,
				);
				enc.fd.coefs[..n * n].fill(0);
			}
			enc.levels = levels;
			rd.dist += enc.ssd(plane, x, y, n, n);
		});
		rd
	}

	/// The token context of a transform block (above / left non-zero).
	fn token_ctx(&self, p: &BlockPos, plane: usize, col: usize, row: usize, tx_size: u8) -> usize {
		let (sx, sy) = self.fd.plane_ss(plane);
		let ax = ((p.col * 2) >> sx) + col;
		let ly = (((p.row & 7) * 2) >> sy) + row;
		let n = 1usize << tx_size;
		let a = &self.fd.above_nz[plane];
		let above = a[ax..(ax + n).min(a.len())].iter().any(|&v| v != 0);
		let left = self.fd.left_nz[plane][ly..(ly + n).min(16)].iter().any(|&v| v != 0);
		usize::from(above) + usize::from(left)
	}

	/// Intra modes for `plane` (0: luma, 1: both chroma planes) ranked by
	/// SATD plus mode bits, and the best SATD. Inside the block, the
	/// predictors see source samples where reconstructed ones would be.
	fn rank_intra_modes(
		&mut self,
		p: &BlockPos,
		bsize: u8,
		plane: usize,
		y_mode: u8,
	) -> (Vec<u8>, u32) {
		let planes: &[usize] = if plane == 0 { &[0] } else { &[1, 2] };
		// Put the source into the block.
		for &pl in planes {
			self.copy_source(p, pl);
		}
		let mut scores: Vec<(f64, u8, u32)> = Vec::with_capacity(10);
		let mut mi = self.base_mi(bsize);
		mi.mode = y_mode;
		for mode in 0..10u8 {
			let mut total = 0u32;
			for &pl in planes {
				let (sx, sy) = self.fd.plane_ss(pl);
				let (x0, y0) = ((p.col * 8) >> sx, (p.row * 8) >> sy);
				self.for_each_tx(p, &mi, pl, |enc, col, row, tx_size, _, _| {
					let (x, y) = (x0 + col * 4, y0 + row * 4);
					let n = 4usize << tx_size;
					let e = enc.edges(p, pl, col, row, tx_size);
					let plane_buf = &mut enc.fd.pic.planes[pl];
					let stride = plane_buf.stride;
					intra::predict(&mut plane_buf.data, stride, x, y, tx_size, mode, &e, 8);
					let ss = enc.src.stride[pl];
					let s = &enc.src.planes[pl][y * ss + x..];
					total += satd(s, ss, &plane_buf.data[y * stride + x..], stride, n, n);
					for r in 0..n {
						let at = (y + r) * stride + x;
						plane_buf.data[at..at + n]
							.copy_from_slice(&enc.src.planes[pl][(y + r) * ss + x..(y + r) * ss + x + n]);
					}
				});
			}
			let bits =
				if plane == 0 { self.y_mode_cost(p, bsize, mode) } else { self.uv_mode_cost(y_mode, mode) };
			scores.push((f64::from(total) + self.sad_per_bit * f64::from(bits) / 256.0, mode, total));
		}
		scores.sort_by(|a, b| a.0.total_cmp(&b.0));
		(scores.iter().map(|s| s.1).collect(), scores[0].2)
	}

	fn copy_source(&mut self, p: &BlockPos, plane: usize) {
		let (sx, sy) = self.fd.plane_ss(plane);
		let (x, y) = ((p.col * 8) >> sx, (p.row * 8) >> sy);
		let n = (p.bw * 8) >> sx;
		let ss = self.src.stride[plane];
		let pl = &mut self.fd.pic.planes[plane];
		for r in 0..n {
			let at = (y + r) * pl.stride + x;
			pl.data[at..at + n]
				.copy_from_slice(&self.src.planes[plane][(y + r) * ss + x..(y + r) * ss + x + n]);
		}
	}

	// --- Inter prediction and motion search ---

	/// Motion-compensated prediction of the block, every plane
	/// (`dec_build_inter_predictors_sb`, unscaled).
	fn predict_inter(&mut self, p: &BlockPos, mi: &ModeInfo) {
		let Some(reference) = self.fd.refs[0].clone() else { return };
		for plane in 0..3 {
			let (sx, sy) = self.fd.plane_ss(plane);
			let src = &reference.planes[plane];
			let src =
				RefPlane { data: &src.data, stride: src.stride, width: src.width, height: src.height };
			let dst = &mut self.fd.pic.planes[plane];
			let stride = dst.stride;
			let (x0, y0) = ((p.col * 8) >> sx, (p.row * 8) >> sy);
			let (w, h) = ((p.bw * 8) >> sx, (p.bh * 8) >> sy);
			let mv = mi.mv[0];
			let (mr, mc) = (i32::from(mv.row) * (2 - sy as i32), i32::from(mv.col) * (2 - sx as i32));
			inter::predict(
				&mut self.fd.scratch,
				&src,
				&mut dst.data,
				stride,
				x0,
				y0,
				w,
				h,
				mr,
				mc,
				mi.interp_filter,
				false,
				255,
			);
		}
	}

	/// SATD of the luma prediction from `mv`.
	fn luma_satd(&mut self, reference: &Arc<Picture<u8>>, p: &BlockPos, mv: Mv, filter: u8) -> u32 {
		let src = &reference.planes[0];
		let src =
			RefPlane { data: &src.data, stride: src.stride, width: src.width, height: src.height };
		let (x0, y0) = (p.col * 8, p.row * 8);
		let n = p.bw * 8;
		// `predict` reads the reference at the destination position: the
		// block's position goes into the vector (whole pixels keep the phase).
		let (mr, mc) =
			(i32::from(mv.row) * 2 + (y0 as i32) * 16, i32::from(mv.col) * 2 + (x0 as i32) * 16);
		inter::predict(
			&mut self.fd.scratch,
			&src,
			&mut self.mc,
			64,
			0,
			0,
			n,
			n,
			mr,
			mc,
			filter,
			false,
			255,
		);
		let ss = self.src.stride[0];
		satd(&self.src.planes[0][y0 * ss + x0..], ss, &self.mc, 64, n, n)
	}

	fn mv_cost(&mut self, mv: Mv, reference: Mv) -> u32 {
		let mut c = Cost(0);
		self.write_mv(&mut c, mv, reference);
		c.0
	}

	/// The inter candidates of a block as (SATD + bits, mode, vector).
	fn inter_candidates(&mut self, p: &BlockPos, bsize: u8) -> Vec<(u32, u8, Mv)> {
		let Some(reference) = self.fd.refs[0].clone() else { return Vec::new() };
		let filter = self.fd.h.interp_filter;
		let nearest = self.ref_mv(p, bsize, pm::NEARESTMV);
		let near = self.ref_mv(p, bsize, pm::NEARMV);
		let best_ref = self.ref_mv(p, bsize, pm::NEWMV);
		let mode_bits = |enc: &mut Self, mode: u8| -> u32 {
			let ctx = enc.fd.mode_context(p, bsize);
			tree_cost(&super::super::INTER_MODE_TREE, &enc.fd.fc.inter_mode[ctx], mode - pm::NEARESTMV)
		};
		let mut out = Vec::with_capacity(4);
		let mut seen: Vec<Mv> = Vec::with_capacity(4);
		for (mode, mv) in [(pm::ZEROMV, Mv::ZERO), (pm::NEARESTMV, nearest), (pm::NEARMV, near)] {
			if seen.contains(&mv) {
				continue;
			}
			seen.push(mv);
			let s = self.luma_satd(&reference, p, mv, filter);
			let bits = mode_bits(self, mode);
			out.push(((f64::from(s) + self.sad_per_bit * f64::from(bits) / 256.0) as u32, mode, mv));
		}
		let new = self.motion_search(&reference, p, best_ref, &[nearest, near, Mv::ZERO]);
		if !seen.contains(&new) {
			let s = self.luma_satd(&reference, p, new, filter);
			let bits = mode_bits(self, pm::NEWMV) + self.mv_cost(new, best_ref);
			out.push((
				(f64::from(s) + self.sad_per_bit * f64::from(bits) / 256.0) as u32,
				pm::NEWMV,
				new,
			));
		}
		out
	}

	/// Diamond full-pel search from the best of `starts`, then half- and
	/// quarter-pel refinement; returns a vector in 1/8 pixel (even values).
	fn motion_search(
		&mut self,
		reference: &Arc<Picture<u8>>,
		p: &BlockPos,
		pred: Mv,
		starts: &[Mv],
	) -> Mv {
		let Some(ref_luma) = self.ref_luma.take() else { return pred };
		let (x0, y0) = ((p.col * 8) as i32, (p.row * 8) as i32);
		let n = p.bw * 8;
		let ss = self.src.stride[0];
		let src_at = y0 as usize * ss + x0 as usize;
		let fw = (self.fd.mi_cols * 8) as i32;
		let fh = (self.fd.mi_rows * 8) as i32;
		let range = self.search.range;
		let (min_x, max_x) = (
			(-x0 - BORDER + MARGIN).max(-range * 2),
			(fw - x0 + BORDER - n as i32 - MARGIN).min(range * 2),
		);
		let (min_y, max_y) = (
			(-y0 - BORDER + MARGIN).max(-range * 2),
			(fh - y0 + BORDER - n as i32 - MARGIN).min(range * 2),
		);
		let cost_at = |enc: &mut Self, dx: i32, dy: i32| -> f64 {
			let at = (y0 + dy + BORDER) as usize * ref_luma.stride + (x0 + dx + BORDER) as usize;
			let s = sad(&enc.src.planes[0][src_at..], ss, &ref_luma.data[at..], ref_luma.stride, n, n);
			let mv = Mv::new(dy * 8, dx * 8);
			f64::from(s) + enc.sad_per_bit * f64::from(enc.mv_cost(mv, pred)) / 256.0
		};
		let full = |mv: Mv| ((i32::from(mv.col) + 4) >> 3, (i32::from(mv.row) + 4) >> 3);
		let mut best = (0, 0);
		let mut best_cost = f64::MAX;
		for &m in std::iter::once(&pred).chain(starts) {
			let (dx, dy) = full(m);
			let (dx, dy) = (dx.clamp(min_x, max_x), dy.clamp(min_y, max_y));
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
		self.ref_luma = Some(ref_luma);
		// Sub-pel refinement with the real filter.
		let filter = self.fd.h.interp_filter;
		let mut mv = Mv::new(best.1 * 8, best.0 * 8);
		let mut cost = f64::from(self.luma_satd(reference, p, mv, filter))
			+ self.sad_per_bit * f64::from(self.mv_cost(mv, pred)) / 256.0;
		for d in [4i32, 2] {
			let center = mv;
			for (oy, ox) in [(-d, -d), (-d, 0), (-d, d), (0, -d), (0, d), (d, -d), (d, 0), (d, d)] {
				let m = Mv::new(i32::from(center.row) + oy, i32::from(center.col) + ox);
				let c = f64::from(self.luma_satd(reference, p, m, filter))
					+ self.sad_per_bit * f64::from(self.mv_cost(m, pred)) / 256.0;
				if c < cost {
					cost = c;
					mv = m;
				}
			}
		}
		mv
	}
}
