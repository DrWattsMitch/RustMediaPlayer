//! Symbol writing: partitions, mode info, motion vectors and coefficient
//! tokens, the decoder's parsing (`decoder.rs`, `mode.rs`, `residual.rs`)
//! inverted line by line. Contexts are derived by the decoder's own
//! functions from the same state, and symbol counts are kept exactly where
//! the decoder keeps them, so backward adaptation stays in step.
//!
//! The same functions price candidates during the search (with a
//! [`Cost`](super::writer::Cost) writer and counting off) and write the
//! tiles (with a [`BoolEncoder`](super::writer::BoolEncoder)).

use super::super::decoder::BlockPos;
use super::super::header::TX_MODE_SELECT;
use super::super::mode::{ModeInfo, MvRef, lower_mv_precision, use_mv_hp};
use super::super::probs::{MvComponentProbs, mv_class};
use super::super::residual::{self, Scan};
use super::super::tables::{self, KF_PARTITION_PROBS, PARETO8};
use super::super::{INTER_MODE_TREE, INTRA_MODE_TREE, MV_CLASS_TREE, MV_FP_TREE, MV_JOINT_TREE};
use super::super::{LAST_FRAME, Mv, NUM_8X8_WIDE, PARTITION_CONTEXT, PARTITION_TREE, SIZE_GROUP};
use super::super::{MAX_TX_SIZE, SUBSIZE, bs, pm, uv_tx_size};
use super::frame::{Decision, FrameEncoder};
use super::writer::Bits;

const CAT1: [u8; 1] = [159];
const CAT2: [u8; 2] = [165, 145];
const CAT3: [u8; 3] = [173, 148, 140];
const CAT4: [u8; 4] = [176, 155, 140, 135];
const CAT5: [u8; 5] = [180, 157, 141, 134, 130];

fn extra<B: Bits>(w: &mut B, v: u32, probs: &[u8]) {
	let n = probs.len();
	for (i, &p) in probs.iter().enumerate() {
		w.put((v >> (n - 1 - i)) & 1 == 1, p);
	}
}

/// Largest coefficient level an 8-bit stream can code (cat6, 14 bits).
pub const MAX_LEVEL: i32 = 67 + (1 << 14) - 1;

/// The token cache value of a level (`vp9_pt_energy_class`).
#[inline]
pub fn energy(v: u32) -> u8 {
	match v {
		0 => 0,
		1 => 1,
		2 => 2,
		3 | 4 => 3,
		5..=10 => 4,
		_ => 5,
	}
}

/// Where symbol counts go while writing (`None`: not counting).
pub struct TokenCounts<'a> {
	pub coef: &'a mut [[[u32; 4]; 6]; 6],
	pub eob_branch: &'a mut [[u32; 6]; 6],
}

/// Writes the tokens of one transform block: `levels` in scan order up to
/// the end of block (the inverse of `residual::decode_coefs`).
#[allow(clippy::too_many_arguments)]
pub fn write_coefs<B: Bits>(
	w: &mut B,
	probs: &[[[u8; 3]; 6]; 6],
	mut counts: Option<TokenCounts>,
	levels: &[i16],
	tx_size: u8,
	mut ctx: usize,
	sc: &Scan,
	cat6: &[u8],
) {
	let max_eob = 16usize << (tx_size << 1);
	let eob = levels.len();
	let band_of: &[u8] = if tx_size == 0 { &tables::COEFBAND_4X4 } else { &tables::COEFBAND_8X8PLUS };
	let mut token_cache = [0u8; 1024];
	let context = |cache: &[u8; 1024], c: usize| {
		let n = sc.neighbors[c];
		(1 + usize::from(cache[usize::from(n[0])]) + usize::from(cache[usize::from(n[1])])) >> 1
	};
	let mut c = 0usize;
	while c < max_eob {
		let mut band = usize::from(band_of[c]);
		let mut p = &probs[band][ctx];
		if let Some(k) = counts.as_mut() {
			k.eob_branch[band][ctx] += 1;
		}
		if c >= eob {
			w.put(false, p[0]);
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][3] += 1;
			}
			return;
		}
		w.put(true, p[0]);
		while levels[c] == 0 {
			w.put(false, p[1]);
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][0] += 1;
			}
			token_cache[usize::from(sc.scan[c])] = 0;
			c += 1;
			ctx = context(&token_cache, c);
			band = usize::from(band_of[c]);
			p = &probs[band][ctx];
		}
		w.put(true, p[1]);
		let v = u32::from(levels[c].unsigned_abs());
		if v == 1 {
			w.put(false, p[2]);
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][1] += 1;
			}
		} else {
			w.put(true, p[2]);
			if let Some(k) = counts.as_mut() {
				k.coef[band][ctx][2] += 1;
			}
			let pareto = &PARETO8[usize::from(p[2]).max(1) - 1];
			if v <= 4 {
				w.put(false, pareto[0]);
				if v == 2 {
					w.put(false, pareto[1]);
				} else {
					w.put(true, pareto[1]);
					w.put(v == 4, pareto[2]);
				}
			} else {
				w.put(true, pareto[0]);
				if v <= 10 {
					w.put(false, pareto[3]);
					if v <= 6 {
						w.put(false, pareto[4]);
						extra(w, v - 5, &CAT1);
					} else {
						w.put(true, pareto[4]);
						extra(w, v - 7, &CAT2);
					}
				} else {
					w.put(true, pareto[3]);
					if v <= 34 {
						w.put(false, pareto[5]);
						if v <= 18 {
							w.put(false, pareto[6]);
							extra(w, v - 11, &CAT3);
						} else {
							w.put(true, pareto[6]);
							extra(w, v - 19, &CAT4);
						}
					} else {
						w.put(true, pareto[5]);
						if v <= 66 {
							w.put(false, pareto[7]);
							extra(w, v - 35, &CAT5);
						} else {
							w.put(true, pareto[7]);
							extra(w, v - 67, cat6);
						}
					}
				}
			}
		}
		token_cache[usize::from(sc.scan[c])] = energy(v);
		w.put(levels[c] < 0, 128);
		c += 1;
		if c < max_eob {
			ctx = context(&token_cache, c);
		}
	}
}

fn write_mv_component<B: Bits>(w: &mut B, c: &MvComponentProbs, v: i32, use_hp: bool) {
	w.put(v < 0, c.sign);
	let z = v.unsigned_abs() as usize - 1;
	let class = mv_class(z);
	let offset = z - if class == 0 { 0 } else { 2 << (class + 2) };
	let (d, fr, hp) = (offset >> 3, (offset >> 1) & 3, offset & 1);
	w.tree(&MV_CLASS_TREE, &c.classes, class as u8);
	if class == 0 {
		w.put(d == 1, c.class0[0]);
	} else {
		for i in 0..class {
			w.put((d >> i) & 1 == 1, c.bits[i]);
		}
	}
	w.tree(&MV_FP_TREE, if class == 0 { &c.class0_fp[d] } else { &c.fp }, fr as u8);
	if use_hp {
		w.put(hp == 1, if class == 0 { c.class0_hp } else { c.hp });
	}
}

/// The (unclamped) position of a block, as `decode_block` computes it.
pub fn block_pos(
	mi_row: usize,
	mi_col: usize,
	bsize: u8,
	mi_rows: usize,
	mi_cols: usize,
	tile_col_start: usize,
) -> BlockPos {
	let bw = usize::from(NUM_8X8_WIDE[usize::from(bsize)]);
	let bh = bw;
	BlockPos {
		row: mi_row,
		col: mi_col,
		bw,
		bh,
		x_mis: bw.min(mi_cols - mi_col),
		y_mis: bh.min(mi_rows - mi_row),
		to_left: -((mi_col * 8 * 8) as i32),
		to_right: ((mi_cols as i32 - bw as i32 - mi_col as i32) * 8) * 8,
		to_top: -((mi_row * 8 * 8) as i32),
		to_bottom: ((mi_rows as i32 - bh as i32 - mi_row as i32) * 8) * 8,
		have_above: mi_row > 0,
		have_left: mi_col > tile_col_start,
	}
}

impl FrameEncoder {
	pub fn pos(&self, mi_row: usize, mi_col: usize, bsize: u8) -> BlockPos {
		block_pos(mi_row, mi_col, bsize, self.fd.mi_rows, self.fd.mi_cols, self.fd.tile.mi_col_start)
	}

	/// Partition context of a block of `bsl` (log2 of its 8x8 width).
	pub fn partition_ctx(&self, mi_row: usize, mi_col: usize, bsl: usize) -> usize {
		let above = usize::from((self.fd.above_partition[mi_col] >> bsl) & 1);
		let left = usize::from((self.fd.left_partition[mi_row & 7] >> bsl) & 1);
		left * 2 + above + bsl * 4
	}

	fn partition_probs(&self, ctx: usize) -> [u8; 3] {
		if self.fd.h.intra_only_frame() { KF_PARTITION_PROBS[ctx] } else { self.fd.fc.partition[ctx] }
	}

	/// Writes the partition of a `bsize` block (of `1 << bsl` 8x8 units) at
	/// (`mi_row`, `mi_col`); only NONE (0) and SPLIT (3) are used.
	pub fn write_partition<B: Bits>(
		&mut self,
		w: &mut B,
		mi_row: usize,
		mi_col: usize,
		bsl: usize,
		partition: u8,
	) {
		let hbs = (1usize << bsl) >> 1;
		let has_rows = mi_row + hbs < self.fd.mi_rows;
		let has_cols = mi_col + hbs < self.fd.mi_cols;
		let ctx = self.partition_ctx(mi_row, mi_col, bsl);
		let probs = self.partition_probs(ctx);
		if has_rows && has_cols {
			w.tree(&PARTITION_TREE, &probs, partition);
		} else if !has_rows && has_cols {
			w.put(partition == 3, probs[1]);
		} else if has_rows && !has_cols {
			w.put(partition == 3, probs[2]);
		}
		if self.fd.counting {
			self.fd.counts.partition[ctx][usize::from(partition)] += 1;
		}
	}

	/// The partition context a coded (unsplit) block leaves behind.
	pub fn set_partition_ctx(&mut self, mi_row: usize, mi_col: usize, bsize: u8) {
		let num_8x8 = usize::from(NUM_8X8_WIDE[usize::from(bsize)]);
		let (a, l) = PARTITION_CONTEXT[usize::from(bsize)];
		self.fd.above_partition[mi_col..mi_col + num_8x8].fill(a);
		let lr = mi_row & 7;
		self.fd.left_partition[lr..(lr + num_8x8).min(8)].fill(l);
	}

	/// Stores a block's mode info for its neighbours (and, in inter
	/// frames, its motion for the next frame's candidate search).
	pub fn commit_mi(&mut self, p: &BlockPos, mi: &ModeInfo) {
		let fd = &mut self.fd;
		for y in 0..p.y_mis {
			let at = (p.row + y) * fd.mi_cols + p.col;
			fd.mi[at..at + p.x_mis].fill(*mi);
		}
		if !fd.h.intra_only_frame() {
			let entry = MvRef { ref_frame: mi.ref_frame, mv: mi.mv };
			for y in 0..p.y_mis {
				let at = (p.row + y) * fd.mi_cols + p.col;
				fd.cur_mvs[at..at + p.x_mis].fill(entry);
			}
		}
	}

	/// Clears the non-zero contexts over a skipped block.
	pub fn clear_nz(&mut self, p: &BlockPos) {
		for plane in 0..3 {
			let (sx, sy) = self.fd.plane_ss(plane);
			let (x, y) = ((p.col * 2) >> sx, ((p.row & 7) * 2) >> sy);
			let (n4w, n4h) = ((p.bw * 2) >> sx, (p.bh * 2) >> sy);
			let a = &mut self.fd.above_nz[plane];
			let end = (x + n4w).min(a.len());
			a[x..end].fill(0);
			let l = &mut self.fd.left_nz[plane];
			l[y..(y + n4h).min(16)].fill(0);
		}
	}

	/// The motion vector a NEARESTMV / NEARMV / NEWMV block starts from
	/// (`find_mv_refs`, then the precision of the frame).
	pub fn ref_mv(&self, p: &BlockPos, bsize: u8, mode: u8) -> Mv {
		let (list, count) = self.fd.find_mv_refs(p, bsize, mode, LAST_FRAME, None);
		let mut mv = list[count - 1];
		lower_mv_precision(&mut mv, self.fd.h.allow_hp);
		mv
	}

	/// Writes a new motion vector relative to `reference` (`read_mv`).
	pub fn write_mv<B: Bits>(&mut self, w: &mut B, mv: Mv, reference: Mv) {
		let row = i32::from(mv.row) - i32::from(reference.row);
		let col = i32::from(mv.col) - i32::from(reference.col);
		let joint = usize::from(row != 0) * 2 + usize::from(col != 0);
		w.tree(&MV_JOINT_TREE, &self.fd.fc.mv_joints, joint as u8);
		let use_hp = self.fd.h.allow_hp && use_mv_hp(reference);
		if row != 0 {
			write_mv_component(w, &self.fd.fc.mv[0], row, use_hp);
		}
		if col != 0 {
			write_mv_component(w, &self.fd.fc.mv[1], col, use_hp);
		}
		if self.fd.counting {
			self.fd.counts.add_mv(row, col);
		}
	}

	/// Writes a block's mode info (`read_mode_info` inverted). Blocks are
	/// 8x8 or larger, single reference (LAST), no segmentation, the frame's
	/// transform mode fixes the transform size.
	pub fn write_mode_info<B: Bits>(&mut self, w: &mut B, p: &BlockPos, mi: &ModeInfo) {
		let above = self.fd.above_mi(p);
		let left = self.fd.left_mi(p);
		let skip_ctx =
			usize::from(above.is_some_and(|m| m.skip)) + usize::from(left.is_some_and(|m| m.skip));
		let counting = self.fd.counting;
		if self.fd.h.intra_only_frame() {
			w.put(mi.skip, self.fd.fc.skip[skip_ctx]);
			if counting {
				self.fd.counts.skip[skip_ctx][usize::from(mi.skip)] += 1;
			}
			self.write_tx_size(w, mi, true, above, left);
			let a = above.filter(|m| !m.is_inter()).map_or(pm::DC, |m| m.y_mode(2));
			let l = left.filter(|m| !m.is_inter()).map_or(pm::DC, |m| m.y_mode(1));
			w.tree(&INTRA_MODE_TREE, &tables::KF_Y_MODE_PROBS[usize::from(a)][usize::from(l)], mi.mode);
			w.tree(&INTRA_MODE_TREE, &tables::KF_UV_MODE_PROBS[usize::from(mi.mode)], mi.uv_mode);
			return;
		}
		w.put(mi.skip, self.fd.fc.skip[skip_ctx]);
		if counting {
			self.fd.counts.skip[skip_ctx][usize::from(mi.skip)] += 1;
		}
		let inter_ctx = match (above, left) {
			(Some(a), Some(l)) => {
				let (ai, li) = (!a.is_inter(), !l.is_inter());
				if ai && li { 3 } else { usize::from(ai || li) }
			}
			(Some(m), None) | (None, Some(m)) => 2 * usize::from(!m.is_inter()),
			(None, None) => 0,
		};
		let inter = mi.is_inter();
		w.put(inter, self.fd.fc.intra_inter[inter_ctx]);
		if counting {
			self.fd.counts.intra_inter[inter_ctx][usize::from(inter)] += 1;
		}
		self.write_tx_size(w, mi, !mi.skip || !inter, above, left);
		if !inter {
			let group = usize::from(SIZE_GROUP[usize::from(mi.sb_type)]);
			w.tree(&INTRA_MODE_TREE, &self.fd.fc.y_mode[group], mi.mode);
			w.tree(&INTRA_MODE_TREE, &self.fd.fc.uv_mode[usize::from(mi.mode)], mi.uv_mode);
			if counting {
				self.fd.counts.y_mode[group][usize::from(mi.mode)] += 1;
				self.fd.counts.uv_mode[usize::from(mi.mode)][usize::from(mi.uv_mode)] += 1;
			}
			return;
		}
		// Single reference: LAST is the first bit of the single-ref tree.
		let ctx0 = super::super::decoder::FrameDecoder::<u8>::single_ref_p1_context(above, left);
		w.put(false, self.fd.fc.single_ref[ctx0][0]);
		if counting {
			self.fd.counts.single_ref[ctx0][0][0] += 1;
		}
		let ctx = self.fd.mode_context(p, mi.sb_type);
		let m = mi.mode - pm::NEARESTMV;
		w.tree(&INTER_MODE_TREE, &self.fd.fc.inter_mode[ctx], m);
		if counting {
			self.fd.counts.inter_mode[ctx][usize::from(m)] += 1;
		}
		if mi.mode == pm::NEWMV {
			let best = self.ref_mv(p, mi.sb_type, pm::NEWMV);
			self.write_mv(w, mi.mv[0], best);
		}
	}

	/// Writes the transform size when the frame selects it per block
	/// (`read_tx_size`).
	fn write_tx_size<B: Bits>(
		&mut self,
		w: &mut B,
		mi: &ModeInfo,
		allow_select: bool,
		above: Option<ModeInfo>,
		left: Option<ModeInfo>,
	) {
		let max = MAX_TX_SIZE[usize::from(mi.sb_type)];
		if !(allow_select && self.fd.ch.tx_mode == TX_MODE_SELECT && mi.sb_type >= bs::B8X8) {
			debug_assert_eq!(mi.tx_size, max.min(self.fd.ch.tx_mode.min(3)));
			return;
		}
		let side = |m: Option<ModeInfo>| m.map(|m| if m.skip { max } else { m.tx_size });
		let (mut a, mut l) = (side(above), side(left));
		if l.is_none() {
			l = a;
		}
		if a.is_none() {
			a = l;
		}
		let ctx = usize::from(a.unwrap_or(max) + l.unwrap_or(max) > max);
		let fc = &self.fd.fc;
		let probs: &[u8] = match max {
			1 => &fc.tx8[ctx],
			2 => &fc.tx16[ctx],
			_ => &fc.tx32[ctx],
		};
		let tx = mi.tx_size;
		w.put(tx >= 1, probs[0]);
		if tx >= 1 && max >= 2 {
			w.put(tx >= 2, probs[1]);
			if tx >= 2 && max >= 3 {
				w.put(tx == 3, probs[2]);
			}
		}
		if self.fd.counting {
			let c = &mut self.fd.counts;
			match max {
				1 => c.tx8[ctx][usize::from(tx)] += 1,
				2 => c.tx16[ctx][usize::from(tx)] += 1,
				_ => c.tx32[ctx][usize::from(tx)] += 1,
			}
		}
	}

	/// Writes one transform block's tokens and updates the non-zero
	/// contexts as `decode_tokens` does.
	#[allow(clippy::too_many_arguments)]
	pub fn write_tokens<B: Bits>(
		&mut self,
		w: &mut B,
		p: &BlockPos,
		mi: &ModeInfo,
		plane: usize,
		col: usize,
		row: usize,
		tx_size: u8,
		tx_type: u8,
		clip: (usize, usize),
		levels: &[i16],
	) {
		let (sx, sy) = self.fd.plane_ss(plane);
		let ax = ((p.col * 2) >> sx) + col;
		let ly = (((p.row & 7) * 2) >> sy) + row;
		let n = 1usize << tx_size;
		let fd = &mut self.fd;
		let above_ctx =
			fd.above_nz[plane][ax..(ax + n).min(fd.above_nz[plane].len())].iter().any(|&v| v != 0);
		let left_ctx = fd.left_nz[plane][ly..(ly + n).min(16)].iter().any(|&v| v != 0);
		let ctx = usize::from(above_ctx) + usize::from(left_ctx);
		let plane_type = usize::from(plane > 0);
		let reference = usize::from(mi.is_inter());
		let sc = residual::scan(tx_size, tx_type);
		let (t, pt) = (usize::from(tx_size), plane_type);
		let probs = &fd.fc.coef[t][pt][reference];
		let counts = fd.counting.then(|| TokenCounts {
			coef: &mut fd.counts.coef[t][pt][reference],
			eob_branch: &mut fd.counts.eob_branch[t][pt][reference],
		});
		write_coefs(w, probs, counts, levels, tx_size, ctx, &sc, residual::cat6(8));
		let nz = u8::from(!levels.is_empty());
		let (clip_w, clip_h) = clip;
		for k in 0..n {
			let v = if tx_size > 0 && clip_w != 0 && col + k >= clip_w { 0 } else { nz };
			if let Some(a) = fd.above_nz[plane].get_mut(ax + k) {
				*a = v;
			}
			let v = if tx_size > 0 && clip_h != 0 && row + k >= clip_h { 0 } else { nz };
			if let Some(l) = fd.left_nz[plane].get_mut(ly + k) {
				*l = v;
			}
		}
	}

	// --- Packing a frame from the stored decisions ---

	fn next_decision(&mut self) -> Decision {
		let d = self.decisions[self.cursor.0];
		self.cursor.0 += 1;
		d
	}

	/// Writes one superblock row of the current tile.
	pub fn pack_sb_row<B: Bits>(&mut self, w: &mut B, mi_row: usize) {
		self.fd.left_nz = [[0; 16]; 3];
		self.fd.left_partition = [0; 8];
		let mut mi_col = self.fd.tile.mi_col_start;
		while mi_col < self.fd.tile.mi_col_end {
			self.pack_partition(w, mi_row, mi_col, bs::B64X64, 3);
			mi_col += 8;
		}
	}

	fn pack_partition<B: Bits>(
		&mut self,
		w: &mut B,
		mi_row: usize,
		mi_col: usize,
		bsize: u8,
		bsl: usize,
	) {
		if mi_row >= self.fd.mi_rows || mi_col >= self.fd.mi_cols {
			return;
		}
		let Decision::Partition(partition) = self.next_decision() else {
			unreachable!("vp9 encoder: decisions out of order");
		};
		self.write_partition(w, mi_row, mi_col, bsl, partition);
		let subsize = SUBSIZE[usize::from(partition)][usize::from(bsize)];
		if partition == 0 {
			self.pack_block(w, mi_row, mi_col, subsize);
			self.set_partition_ctx(mi_row, mi_col, bsize);
		} else {
			let hbs = (1usize << bsl) >> 1;
			self.pack_partition(w, mi_row, mi_col, subsize, bsl - 1);
			self.pack_partition(w, mi_row, mi_col + hbs, subsize, bsl - 1);
			self.pack_partition(w, mi_row + hbs, mi_col, subsize, bsl - 1);
			self.pack_partition(w, mi_row + hbs, mi_col + hbs, subsize, bsl - 1);
		}
	}

	fn pack_block<B: Bits>(&mut self, w: &mut B, mi_row: usize, mi_col: usize, bsize: u8) {
		let Decision::Block(mi) = self.next_decision() else {
			unreachable!("vp9 encoder: decisions out of order");
		};
		let p = self.pos(mi_row, mi_col, bsize);
		self.write_mode_info(w, &p, &mi);
		if mi.skip {
			self.clear_nz(&p);
		} else {
			for plane in 0..3 {
				self.for_each_tx(&p, &mi, plane, |enc, col, row, tx_size, tx_type, clip| {
					let at = enc.cursor.1;
					let eob = enc.tokens[at] as usize;
					enc.cursor.1 = at + 1 + eob;
					let levels = std::mem::take(&mut enc.tokens);
					enc.write_tokens(
						w,
						&p,
						&mi,
						plane,
						col,
						row,
						tx_size,
						tx_type,
						clip,
						&levels[at + 1..at + 1 + eob],
					);
					enc.tokens = levels;
				});
			}
		}
		self.commit_mi(&p, &mi);
	}

	/// Visits the visible transform blocks of one plane of a block in
	/// coding order (as `reconstruct_plane`): `f(enc, col, row, tx_size,
	/// tx_type, (clip_w, clip_h))`, positions in 4x4 units.
	pub fn for_each_tx(
		&mut self,
		p: &BlockPos,
		mi: &ModeInfo,
		plane: usize,
		mut f: impl FnMut(&mut Self, usize, usize, u8, u8, (usize, usize)),
	) {
		let (sx, sy) = self.fd.plane_ss(plane);
		let tx_size = if plane == 0 { mi.tx_size } else { uv_tx_size(mi.sb_type, mi.tx_size, sx, sy) };
		let n4w = (p.bw * 2) >> sx;
		let n4h = (p.bh * 2) >> sy;
		let step = 1usize << tx_size;
		let max_w = if p.to_right >= 0 { n4w as i32 } else { n4w as i32 + (p.to_right >> (5 + sx)) };
		let max_h = if p.to_bottom >= 0 { n4h as i32 } else { n4h as i32 + (p.to_bottom >> (5 + sy)) };
		let clip_w = if p.to_right >= 0 { 0 } else { max_w.max(0) as usize };
		let clip_h = if p.to_bottom >= 0 { 0 } else { max_h.max(0) as usize };
		let tx_type = if plane == 0 && !mi.is_inter() && !self.fd.h.lossless {
			super::super::intra::MODE_TO_TX_TYPE[usize::from(mi.mode)]
		} else {
			residual::DCT_DCT
		};
		let mut row = 0;
		while (row as i32) < max_h {
			let mut col = 0;
			while (col as i32) < max_w {
				f(self, col, row, tx_size, tx_type, (clip_w, clip_h));
				col += step;
			}
			row += step;
		}
	}
}
