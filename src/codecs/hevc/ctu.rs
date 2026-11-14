//! Slice segment data: coding tree units, coding/prediction/transform
//! units, residual coding and reconstruction (ITU-T H.265 §7.3.8, §8.4,
//! §8.6; FFmpeg `hevcdec.c`, `hevc/cabac.c`).

use std::sync::Arc;

use super::cabac::{Cabac, Contexts};
use super::intra;
use super::mc;
use super::picture::{FrameRef, MvField, PF_BI, PF_INTRA, PF_L0, PF_L1, Plane, RefInfo};
use super::ps::{Layout, Pps, Sps, diag_scan};
use super::slice::{SliceHeader, SliceType};
use super::tables::*;
use super::transform;

#[path = "mvs.rs"]
mod mvs;
use mvs::Pb;

/// A scaling matrix (8x8, upsampled for larger blocks) and its DC value.
type ScaleMatrix<'a> = (&'a [u8; 64], u8);

/// SAO parameters of one CTB.
#[derive(Clone, Copy, Default, Debug)]
pub struct Sao {
	/// 0 off, 1 band, 2 edge; per component.
	pub type_idx: [u8; 3],
	pub band_position: [u8; 3],
	pub eo_class: [u8; 3],
	pub offset: [[i16; 5]; 3],
}

/// Per-slice-segment information kept for in-loop filtering.
#[derive(Clone, Debug)]
pub struct SliceInfo {
	pub beta_offset: i32,
	pub tc_offset: i32,
	pub lf_across_slices: bool,
	pub refs: RefInfo,
}

/// The picture under construction and all its block-level side data.
pub struct PicState {
	pub sps: Arc<Sps>,
	pub pps: Arc<Pps>,
	pub layout: Arc<Layout>,
	pub planes: [Plane; 3],
	pub poc: i32,
	pub w4: usize,
	pub h4: usize,
	pub mvf: Vec<MvField>,
	pub ipm: Vec<u8>,
	pub cbf_luma: Vec<u8>,
	/// 1 where deblocking and SAO must leave samples alone (PCM with
	/// pcm_loop_filter_disabled, or transquant bypass).
	pub no_filter: Vec<u8>,
	pub bs_v: Vec<u8>,
	pub bs_h: Vec<u8>,
	pub skip: Vec<u8>,
	pub ct_depth: Vec<u8>,
	pub qp_y: Vec<i8>,
	/// SliceAddrRs of each CTB (u32::MAX: not decoded).
	pub ctb_addr: Vec<u32>,
	pub ctb_slice: Vec<u16>,
	pub slices: Vec<SliceInfo>,
	pub sao: Vec<Sao>,
	/// Contexts at the end of the previous slice segment (dependent slices).
	pub ds_contexts: Option<Box<Contexts>>,
	/// Contexts saved after the second CTB of a row (WPP).
	pub wpp_contexts: Option<Box<Contexts>>,
	/// Last QpY of the previous slice segment (dependent slices continue it).
	pub ds_qp_y: i32,
	pub ctbs_decoded: usize,
}

impl PicState {
	pub fn new(sps: Arc<Sps>, pps: Arc<Pps>, layout: Arc<Layout>, poc: i32) -> Self {
		let (w, h) = (sps.width as usize, sps.height as usize);
		let mid = 1u16 << (sps.bit_depth - 1);
		let planes = [
			Plane::new(w, h, super::picture::PAD_LUMA, mid),
			Plane::new(w / 2, h / 2, super::picture::PAD_CHROMA, mid),
			Plane::new(w / 2, h / 2, super::picture::PAD_CHROMA, mid),
		];
		let (w4, h4) = (w.div_ceil(4), h.div_ceil(4));
		let mincb = (sps.min_cb_width * sps.min_cb_height) as usize;
		let nctb = (sps.ctb_width * sps.ctb_height) as usize;
		Self {
			planes,
			poc,
			w4,
			h4,
			mvf: vec![MvField::default(); w4 * h4],
			ipm: vec![intra::MODE_DC as u8; w4 * h4],
			cbf_luma: vec![0; w4 * h4],
			no_filter: vec![0; w4 * h4],
			bs_v: vec![0; w4 * h4],
			bs_h: vec![0; w4 * h4],
			skip: vec![0; mincb],
			ct_depth: vec![0; mincb],
			qp_y: vec![0; mincb],
			ctb_addr: vec![u32::MAX; nctb],
			ctb_slice: vec![0; nctb],
			slices: Vec::new(),
			sao: vec![Sao::default(); nctb],
			ds_contexts: None,
			wpp_contexts: None,
			ds_qp_y: 0,
			ctbs_decoded: 0,
			sps,
			pps,
			layout,
		}
	}
}

#[derive(Debug)]
pub struct DecodeError(pub &'static str);

/// Reference pictures of the current slice.
pub struct SliceRefs {
	pub list: [Vec<Option<FrameRef>>; 2],
	pub info: RefInfo,
	pub collocated: Option<FrameRef>,
	/// All references precede the current picture in output order.
	pub no_backward_pred: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PartMode {
	P2Nx2N,
	P2NxN,
	PNx2N,
	PNxN,
	P2NxnU,
	P2NxnD,
	PnLx2N,
	PnRx2N,
}

/// Coding unit state.
struct Cu {
	x: i32,
	y: i32,
	log2_size: u32,
	intra: bool,
	part: PartMode,
	bypass: bool,
	intra_split: bool,
	max_trafo_depth: u32,
	intra_mode: [u32; 4],
	intra_mode_c: u32,
	merge_flag: bool,
}

const SCAN_DIAG: usize = 0;
const SCAN_HORIZ: usize = 1;
const SCAN_VERT: usize = 2;

struct Scans {
	/// [scan][log2 size 2..5] positions as (x, y) for 4x4 sub-blocks and
	/// within a sub-block.
	sub4: [Vec<(u8, u8)>; 3],
	cg: [[Vec<(u8, u8)>; 4]; 3],
}

fn scans() -> &'static Scans {
	static S: std::sync::OnceLock<Scans> = std::sync::OnceLock::new();
	S.get_or_init(|| {
		let horiz = |n: usize| (0..n * n).map(|i| ((i % n) as u8, (i / n) as u8)).collect::<Vec<_>>();
		let vert = |n: usize| (0..n * n).map(|i| ((i / n) as u8, (i % n) as u8)).collect::<Vec<_>>();
		let make = |f: &dyn Fn(usize) -> Vec<(u8, u8)>| [f(1), f(2), f(4), f(8)];
		Scans {
			sub4: [diag_scan(4), horiz(4), vert(4)],
			cg: [make(&diag_scan), make(&horiz), make(&vert)],
		}
	})
}

const CTX_IDX_MAP_4X4: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];
const QP_C_TABLE: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// Chroma QP mapping for 4:2:0 (Table 8-10).
pub fn chroma_qp(qp_i: i32) -> i32 {
	if qp_i < 30 {
		qp_i
	} else if qp_i > 43 {
		qp_i - 6
	} else {
		QP_C_TABLE[(qp_i - 30) as usize]
	}
}

pub struct SliceDecoder<'a, 'b> {
	pub pic: &'a mut PicState,
	pub sh: &'a SliceHeader,
	pub refs: &'a SliceRefs,
	pub slice_idx: u16,
	cabac: Cabac<'b>,
	sps: Arc<Sps>,
	pps: Arc<Pps>,
	layout: Arc<Layout>,
	init_type: usize,
	// QP state.
	qp_y: i32,
	qp_prev: i32,
	is_cu_qp_delta_coded: bool,
	cu_qp_delta: i32,
	qg_x: i32,
	qg_y: i32,
	ct_depth: u32,
	ctb_x: i32,
	ctb_y: i32,
	coeffs: Box<[i32; 32 * 32]>,
	pred: Box<[[i16; mc::MAX_PB * mc::MAX_PB]; 2]>,
	mc_tmp: Box<[i16; mc::TMP_LEN]>,
	tr_tmp: Box<[i32; 32 * 32]>,
	tr_tmp2: Box<[i32; 32 * 32]>,
	pub(super) cur_slice_addr: u32,
}

impl<'a, 'b> SliceDecoder<'a, 'b> {
	pub fn new(
		pic: &'a mut PicState,
		sh: &'a SliceHeader,
		refs: &'a SliceRefs,
		data: &'b [u8],
	) -> Self {
		let sps = pic.sps.clone();
		let pps = pic.pps.clone();
		let layout = pic.layout.clone();
		let mut init_type = match sh.slice_type {
			SliceType::I => 0,
			SliceType::P => 1,
			SliceType::B => 2,
		};
		if sh.cabac_init && sh.slice_type != SliceType::I {
			init_type = 3 - init_type;
		}
		let ctx = Contexts::new(init_type, sh.slice_qp);
		let cabac = Cabac::new(data, sh.data_offset, ctx);
		pic.slices.push(SliceInfo {
			beta_offset: sh.beta_offset,
			tc_offset: sh.tc_offset,
			lf_across_slices: sh.loop_filter_across_slices,
			refs: refs.info.clone(),
		});
		let slice_idx = (pic.slices.len() - 1) as u16;
		Self {
			pic,
			sh,
			refs,
			slice_idx,
			cabac,
			sps,
			pps,
			layout,
			init_type,
			qp_y: sh.slice_qp,
			qp_prev: sh.slice_qp,
			is_cu_qp_delta_coded: false,
			cu_qp_delta: 0,
			qg_x: 0,
			qg_y: 0,
			ct_depth: 0,
			ctb_x: 0,
			ctb_y: 0,
			coeffs: Box::new([0; 32 * 32]),
			pred: Box::new([[0; mc::MAX_PB * mc::MAX_PB]; 2]),
			mc_tmp: Box::new([0; mc::TMP_LEN]),
			tr_tmp: Box::new([0; 32 * 32]),
			tr_tmp2: Box::new([0; 32 * 32]),
			cur_slice_addr: sh.slice_addr_rs,
		}
	}

	fn fresh_contexts(&self) -> Contexts {
		Contexts::new(self.init_type, self.sh.slice_qp)
	}

	/// Decodes every CTB of the slice segment.
	pub fn decode(&mut self) -> Result<(), DecodeError> {
		let sps = self.sps.clone();
		let pps = self.pps.clone();
		let layout = self.layout.clone();
		let ctb_w = sps.ctb_width as usize;
		let nctb = (sps.ctb_width * sps.ctb_height) as usize;
		let mut ts = layout.ctb_addr_rs_to_ts[self.sh.segment_address as usize] as usize;
		let first_ts = ts;
		if self.sh.dependent {
			if let Some(c) = self.pic.ds_contexts.take() {
				self.cabac.ctx = *c;
			}
			self.qp_y = self.pic.ds_qp_y;
		}
		loop {
			if ts >= nctb {
				return Err(DecodeError("hevc: slice runs past the picture"));
			}
			let rs = layout.ctb_addr_ts_to_rs[ts] as usize;
			let (cx, cy) = (rs % ctb_w, rs / ctb_w);
			let tile_x0 = tile_col_start(&layout, cx);
			let first_in_tile =
				pps.tiles_enabled && ts > 0 && layout.tile_id[ts] != layout.tile_id[ts - 1];
			let row_start = pps.entropy_coding_sync && cx == tile_x0;
			// The CTB above-right of a tile row's first CTB, for WPP sync.
			let tr_available = row_start && cy > 0 && cx + 1 < ctb_w && {
				let tr = rs + 1 - ctb_w;
				self.pic.ctb_addr[tr] == self.cur_slice_addr
					&& layout.tile_id[layout.ctb_addr_rs_to_ts[tr] as usize] == layout.tile_id[ts]
			};
			if ts == first_ts {
				// Independent segments start from fresh contexts (set up in
				// `new`); dependent ones from the previous segment's.
				if first_in_tile {
					self.cabac.ctx = self.fresh_contexts();
				} else if row_start && !self.sh.first_slice_in_pic {
					if tr_available {
						if let Some(c) = &self.pic.wpp_contexts {
							self.cabac.ctx = (**c).clone();
						}
					} else {
						self.cabac.ctx = self.fresh_contexts();
					}
				}
				if !self.sh.dependent || first_in_tile || row_start {
					self.qp_y = self.sh.slice_qp;
				}
			} else if first_in_tile {
				self.cabac.ctx = self.fresh_contexts();
				self.qp_y = self.sh.slice_qp;
			} else if row_start {
				match (&self.pic.wpp_contexts, tr_available) {
					(Some(c), true) => self.cabac.ctx = (**c).clone(),
					_ => self.cabac.ctx = self.fresh_contexts(),
				}
				self.qp_y = self.sh.slice_qp;
			}
			self.pic.ctb_addr[rs] = self.cur_slice_addr;
			self.pic.ctb_slice[rs] = self.slice_idx;
			let log2 = sps.log2_ctb_size;
			self.ctb_x = (cx as i32) << log2;
			self.ctb_y = (cy as i32) << log2;
			self.sao_syntax(cx, cy, rs);
			self.coding_quadtree(self.ctb_x, self.ctb_y, log2, 0)?;
			self.pic.ctbs_decoded += 1;
			let end_of_slice = self.cabac.terminate();
			if pps.entropy_coding_sync && cx - tile_x0 <= 1 {
				// Stored after a tile row's second CTB (the first's copy is
				// only ever read when the tile is one CTB wide, never).
				self.pic.wpp_contexts = Some(Box::new(self.cabac.ctx.clone()));
			}
			if self.cabac.overrun() {
				return Err(DecodeError("hevc: slice data truncated"));
			}
			ts += 1;
			if end_of_slice {
				break;
			}
			if ts >= nctb {
				return Err(DecodeError("hevc: missing end of slice"));
			}
			let next_new_tile = pps.tiles_enabled && layout.tile_id[ts] != layout.tile_id[ts - 1];
			let next_cx = layout.ctb_addr_ts_to_rs[ts] as usize % ctb_w;
			let next_row = pps.entropy_coding_sync && next_cx == tile_col_start(&layout, next_cx);
			if next_new_tile || next_row {
				// end_of_subset_one_bit, then byte_alignment().
				self.cabac.terminate();
				self.cabac.reinit_after_terminate();
			}
		}
		self.pic.ds_contexts = Some(Box::new(self.cabac.ctx.clone()));
		self.pic.ds_qp_y = self.qp_y;
		Ok(())
	}

	// -----------------------------------------------------------------
	// Availability (§6.4)
	// -----------------------------------------------------------------

	/// z-scan order block availability (6.4.1), luma coordinates.
	#[inline]
	fn available(&self, xc: i32, yc: i32, xn: i32, yn: i32) -> bool {
		let sps = &*self.sps;
		if xn < 0 || yn < 0 || xn >= sps.width as i32 || yn >= sps.height as i32 {
			return false;
		}
		let s = sps.log2_min_tb_size;
		let l = &*self.layout;
		if l.zs(xn >> s, yn >> s) > l.zs(xc >> s, yc >> s) {
			return false;
		}
		let log2 = sps.log2_ctb_size;
		let cw = sps.ctb_width as i32;
		let ctb_n = ((yn >> log2) * cw + (xn >> log2)) as usize;
		let ctb_c = ((yc >> log2) * cw + (xc >> log2)) as usize;
		if self.pic.ctb_addr[ctb_n] != self.cur_slice_addr {
			return false;
		}
		ctb_n == ctb_c
			|| l.tile_id[l.ctb_addr_rs_to_ts[ctb_n] as usize]
				== l.tile_id[l.ctb_addr_rs_to_ts[ctb_c] as usize]
	}

	#[inline]
	fn i4(&self, x: i32, y: i32) -> usize {
		(y as usize >> 2) * self.pic.w4 + (x as usize >> 2)
	}

	#[inline]
	fn min_cb(&self, x: i32, y: i32) -> usize {
		let s = self.sps.log2_min_cb_size;
		(y as usize >> s) * self.sps.min_cb_width as usize + (x as usize >> s)
	}

	// -----------------------------------------------------------------
	// SAO syntax (§7.3.8.3)
	// -----------------------------------------------------------------

	fn sao_syntax(&mut self, cx: usize, cy: usize, rs: usize) {
		let mut sao = Sao::default();
		if !(self.sh.sao_luma || self.sh.sao_chroma) {
			self.pic.sao[rs] = sao;
			return;
		}
		let ctb_w = self.sps.ctb_width as usize;
		let (x, y) = (self.ctb_x, self.ctb_y);
		let size = 1 << self.sps.log2_ctb_size;
		let mut merge_left = false;
		let mut merge_up = false;
		if cx > 0 && self.available(x, y, x - size, y) {
			merge_left = self.cabac.decode(SAO_MERGE_FLAG);
		}
		if cy > 0 && !merge_left && self.available(x, y, x, y - size) {
			merge_up = self.cabac.decode(SAO_MERGE_FLAG);
		}
		if merge_left || merge_up {
			let src = if merge_left { rs - 1 } else { rs - ctb_w };
			sao = self.pic.sao[src];
			if !self.sh.sao_luma {
				sao.type_idx[0] = 0;
			}
			if !self.sh.sao_chroma {
				sao.type_idx[1] = 0;
				sao.type_idx[2] = 0;
			}
			self.pic.sao[rs] = sao;
			return;
		}
		let bit_depth = self.sps.bit_depth;
		let max_abs = (1 << (bit_depth.min(10) - 5)) - 1;
		for c in 0..3 {
			let enabled = if c == 0 { self.sh.sao_luma } else { self.sh.sao_chroma };
			if !enabled {
				continue;
			}
			if c == 2 {
				sao.type_idx[2] = sao.type_idx[1];
				sao.eo_class[2] = sao.eo_class[1];
			} else {
				sao.type_idx[c] = if !self.cabac.decode(SAO_TYPE_IDX) {
					0
				} else if !self.cabac.bypass() {
					1
				} else {
					2
				};
			}
			if sao.type_idx[c] == 0 {
				continue;
			}
			let mut abs = [0i16; 4];
			for a in &mut abs {
				let mut v = 0;
				while v < max_abs && self.cabac.bypass() {
					v += 1;
				}
				*a = v as i16;
			}
			if sao.type_idx[c] == 1 {
				for a in &mut abs {
					if *a != 0 && self.cabac.bypass() {
						*a = -*a;
					}
				}
				sao.band_position[c] = self.cabac.bypass_bits(5) as u8;
			} else {
				abs[2] = -abs[2];
				abs[3] = -abs[3];
				if c != 2 {
					sao.eo_class[c] = self.cabac.bypass_bits(2) as u8;
				}
			}
			// log2_sao_offset_scale (range extensions) is 0.
			sao.offset[c] = [0, abs[0], abs[1], abs[2], abs[3]];
		}
		self.pic.sao[rs] = sao;
	}

	// -----------------------------------------------------------------
	// Coding quadtree / coding unit
	// -----------------------------------------------------------------

	fn coding_quadtree(
		&mut self,
		x0: i32,
		y0: i32,
		log2: u32,
		depth: u32,
	) -> Result<(), DecodeError> {
		let sps = self.sps.clone();
		let size = 1i32 << log2;
		let split = if x0 + size <= sps.width as i32
			&& y0 + size <= sps.height as i32
			&& log2 > sps.log2_min_cb_size
		{
			let mut inc = 0;
			if self.available(x0, y0, x0 - 1, y0)
				&& u32::from(self.pic.ct_depth[self.min_cb(x0 - 1, y0)]) > depth
			{
				inc += 1;
			}
			if self.available(x0, y0, x0, y0 - 1)
				&& u32::from(self.pic.ct_depth[self.min_cb(x0, y0 - 1)]) > depth
			{
				inc += 1;
			}
			self.cabac.decode(SPLIT_CODING_UNIT_FLAG + inc)
		} else {
			log2 > sps.log2_min_cb_size
		};
		if self.pps.cu_qp_delta_enabled && log2 >= sps.log2_ctb_size - self.pps.diff_cu_qp_delta_depth {
			self.is_cu_qp_delta_coded = false;
			self.cu_qp_delta = 0;
			self.qg_x = x0;
			self.qg_y = y0;
			self.qp_prev = self.qp_y;
		}
		if split {
			let half = size >> 1;
			let (x1, y1) = (x0 + half, y0 + half);
			self.coding_quadtree(x0, y0, log2 - 1, depth + 1)?;
			if x1 < sps.width as i32 {
				self.coding_quadtree(x1, y0, log2 - 1, depth + 1)?;
			}
			if y1 < sps.height as i32 {
				self.coding_quadtree(x0, y1, log2 - 1, depth + 1)?;
			}
			if x1 < sps.width as i32 && y1 < sps.height as i32 {
				self.coding_quadtree(x1, y1, log2 - 1, depth + 1)?;
			}
		} else {
			self.ct_depth = depth;
			self.coding_unit(x0, y0, log2)?;
		}
		Ok(())
	}

	/// QpY prediction and update (8.6.1).
	fn set_qp_y(&mut self) {
		let sps = &*self.sps;
		let mask = (1i32 << sps.log2_ctb_size) - 1;
		let (xq, yq) = (self.qg_x, self.qg_y);
		let pred = self.qp_prev;
		let qa = if xq & mask != 0 { i32::from(self.pic.qp_y[self.min_cb(xq - 1, yq)]) } else { pred };
		let qb = if yq & mask != 0 { i32::from(self.pic.qp_y[self.min_cb(xq, yq - 1)]) } else { pred };
		let p = (qa + qb + 1) >> 1;
		let off = sps.qp_bd_offset;
		self.qp_y = if self.cu_qp_delta != 0 {
			(p + self.cu_qp_delta + 52 + 2 * off).rem_euclid(52 + off) - off
		} else {
			p
		};
	}

	fn coding_unit(&mut self, x0: i32, y0: i32, log2: u32) -> Result<(), DecodeError> {
		let sps = self.sps.clone();
		let pps = self.pps.clone();
		let size = 1i32 << log2;
		let mut cu = Cu {
			x: x0,
			y: y0,
			log2_size: log2,
			intra: true,
			part: PartMode::P2Nx2N,
			bypass: false,
			intra_split: false,
			max_trafo_depth: 0,
			intra_mode: [intra::MODE_DC; 4],
			intra_mode_c: intra::MODE_DC,
			merge_flag: false,
		};
		if !self.pps.cu_qp_delta_enabled {
			// No QGs: every CU predicts from the slice QP.
		}
		if pps.transquant_bypass_enabled {
			cu.bypass = self.cabac.decode(CU_TRANSQUANT_BYPASS_FLAG);
		}
		let mut skip = false;
		if self.sh.slice_type != SliceType::I {
			let mut inc = 0;
			if self.available(x0, y0, x0 - 1, y0) && self.pic.skip[self.min_cb(x0 - 1, y0)] != 0 {
				inc += 1;
			}
			if self.available(x0, y0, x0, y0 - 1) && self.pic.skip[self.min_cb(x0, y0 - 1)] != 0 {
				inc += 1;
			}
			skip = self.cabac.decode(SKIP_FLAG + inc);
			cu.intra = false;
		}
		self.fill_min_cb(x0, y0, log2, |p, i| p.skip[i] = u8::from(skip));
		let mut pcm = false;
		if skip {
			self.prediction_unit(&mut cu, x0, y0, size, size, 0)?;
			self.set_intra_default(x0, y0, size, false);
			self.mark_cu_edges_and_bs(x0, y0, log2, &cu);
		} else {
			if self.sh.slice_type != SliceType::I {
				cu.intra = self.cabac.decode(PRED_MODE_FLAG);
			}
			if !cu.intra || log2 == sps.log2_min_cb_size {
				cu.part = self.part_mode(&cu, log2);
				cu.intra_split = cu.part == PartMode::PNxN && cu.intra;
			}
			if cu.intra {
				if cu.part == PartMode::P2Nx2N
					&& sps.pcm_enabled
					&& log2 >= sps.log2_min_pcm_cb_size
					&& log2 <= sps.log2_max_pcm_cb_size
				{
					pcm = self.cabac.terminate();
				}
				if pcm {
					self.set_intra_default(x0, y0, size, true);
					self.pcm_sample(x0, y0, log2);
					if sps.pcm_loop_filter_disabled {
						self.fill_4x4(x0, y0, size, size, |p, i| p.no_filter[i] = 1);
					}
					self.mark_cu_edges_and_bs(x0, y0, log2, &cu);
				} else {
					self.intra_prediction_unit(&mut cu, x0, y0, log2);
				}
			} else {
				self.set_intra_default(x0, y0, size, false);
				let (h, q) = (size / 2, size / 4);
				let parts: &[(i32, i32, i32, i32)] = match cu.part {
					PartMode::P2Nx2N => &[(0, 0, size, size)],
					PartMode::P2NxN => &[(0, 0, size, h), (0, h, size, h)],
					PartMode::PNx2N => &[(0, 0, h, size), (h, 0, h, size)],
					PartMode::P2NxnU => &[(0, 0, size, q), (0, q, size, size - q)],
					PartMode::P2NxnD => &[(0, 0, size, size - q), (0, size - q, size, q)],
					PartMode::PnLx2N => &[(0, 0, q, size), (q, 0, size - q, size)],
					PartMode::PnRx2N => &[(0, 0, size - q, size), (size - q, 0, q, size)],
					PartMode::PNxN => &[(0, 0, h, h), (h, 0, h, h), (0, h, h, h), (h, h, h, h)],
				};
				let parts = parts.to_vec();
				for (idx, &(dx, dy, w, hh)) in parts.iter().enumerate() {
					self.prediction_unit(&mut cu, x0 + dx, y0 + dy, w, hh, idx)?;
				}
			}
			if !pcm {
				let mut rqt_root_cbf = true;
				if !cu.intra && !(cu.part == PartMode::P2Nx2N && cu.merge_flag) {
					rqt_root_cbf = self.cabac.decode(NO_RESIDUAL_DATA_FLAG);
				}
				if rqt_root_cbf {
					cu.max_trafo_depth = if cu.intra {
						sps.max_transform_hierarchy_depth_intra + u32::from(cu.intra_split)
					} else {
						sps.max_transform_hierarchy_depth_inter
					};
					self.transform_tree(&cu, x0, y0, x0, y0, log2, 0, 0, [false, false])?;
				} else {
					self.mark_cu_edges_and_bs(x0, y0, log2, &cu);
				}
			}
		}
		if self.pps.cu_qp_delta_enabled && !self.is_cu_qp_delta_coded {
			self.set_qp_y();
		} else if !self.pps.cu_qp_delta_enabled {
			self.qp_y = self.sh.slice_qp;
		}
		let qp = self.qp_y as i8;
		let depth = self.ct_depth as u8;
		self.fill_min_cb(x0, y0, log2, |p, i| {
			p.qp_y[i] = qp;
			p.ct_depth[i] = depth;
		});
		if cu.bypass {
			self.fill_4x4(x0, y0, size, size, |p, i| p.no_filter[i] = 1);
		}
		Ok(())
	}

	fn fill_min_cb(&mut self, x0: i32, y0: i32, log2: u32, mut f: impl FnMut(&mut PicState, usize)) {
		let s = self.sps.log2_min_cb_size;
		let n = 1usize << (log2 - s);
		let w = self.sps.min_cb_width as usize;
		let (xb, yb) = ((x0 >> s) as usize, (y0 >> s) as usize);
		for y in yb..(yb + n).min(self.sps.min_cb_height as usize) {
			for x in xb..(xb + n).min(w) {
				f(self.pic, y * w + x);
			}
		}
	}

	fn fill_4x4(
		&mut self,
		x0: i32,
		y0: i32,
		w: i32,
		h: i32,
		mut f: impl FnMut(&mut PicState, usize),
	) {
		let (w4, h4) = (self.pic.w4, self.pic.h4);
		let (xb, yb) = ((x0 >> 2) as usize, (y0 >> 2) as usize);
		for y in yb..(yb + (h as usize).div_ceil(4)).min(h4) {
			for x in xb..(xb + (w as usize).div_ceil(4)).min(w4) {
				f(self.pic, y * w4 + x);
			}
		}
	}

	fn set_intra_default(&mut self, x0: i32, y0: i32, size: i32, intra: bool) {
		self.fill_4x4(x0, y0, size, size, |p, i| {
			p.ipm[i] = intra::MODE_DC as u8;
			if intra {
				p.mvf[i] = MvField { pred: PF_INTRA, ..Default::default() };
			}
		});
	}

	fn part_mode(&mut self, cu: &Cu, log2: u32) -> PartMode {
		let sps = &*self.sps;
		if self.cabac.decode(PART_MODE) {
			return PartMode::P2Nx2N;
		}
		if log2 == sps.log2_min_cb_size {
			if cu.intra {
				return PartMode::PNxN;
			}
			if self.cabac.decode(PART_MODE + 1) {
				return PartMode::P2NxN;
			}
			if log2 == 3 {
				return PartMode::PNx2N;
			}
			if self.cabac.decode(PART_MODE + 2) {
				return PartMode::PNx2N;
			}
			return PartMode::PNxN;
		}
		if !sps.amp_enabled {
			return if self.cabac.decode(PART_MODE + 1) { PartMode::P2NxN } else { PartMode::PNx2N };
		}
		if self.cabac.decode(PART_MODE + 1) {
			if self.cabac.decode(PART_MODE + 3) {
				return PartMode::P2NxN;
			}
			return if self.cabac.bypass() { PartMode::P2NxnD } else { PartMode::P2NxnU };
		}
		if self.cabac.decode(PART_MODE + 3) {
			return PartMode::PNx2N;
		}
		if self.cabac.bypass() { PartMode::PnRx2N } else { PartMode::PnLx2N }
	}

	fn pcm_sample(&mut self, x0: i32, y0: i32, log2: u32) {
		let sps = self.sps.clone();
		let size = 1usize << log2;
		let bits = size * size * sps.pcm_bit_depth as usize
			+ 2 * (size / 2) * (size / 2) * sps.pcm_bit_depth_chroma as usize;
		let raw = self.cabac.take_raw(bits.div_ceil(8));
		let mut br = crate::codecs::bitstream::BitReader::new(raw);
		let bd = sps.bit_depth;
		for c in 0..3 {
			let (n, pbd) =
				if c == 0 { (size, sps.pcm_bit_depth) } else { (size / 2, sps.pcm_bit_depth_chroma) };
			let (px, py) = if c == 0 { (x0, y0) } else { (x0 / 2, y0 / 2) };
			let plane = &mut self.pic.planes[c];
			for y in 0..n {
				let at = plane.idx(px as isize, py as isize + y as isize);
				for x in 0..n {
					plane.data[at + x] = (br.bits(pbd) << (bd - pbd)) as u16;
				}
			}
		}
	}

	// -----------------------------------------------------------------
	// Intra mode signalling (§8.4.2)
	// -----------------------------------------------------------------

	fn intra_prediction_unit(&mut self, cu: &mut Cu, x0: i32, y0: i32, log2: u32) {
		let split = cu.part == PartMode::PNxN;
		let pb = (1i32 << log2) >> u32::from(split);
		let side = if split { 2 } else { 1 };
		let mut prev = [false; 4];
		for p in prev.iter_mut().take(side * side) {
			*p = self.cabac.decode(PREV_INTRA_LUMA_PRED_FLAG);
		}
		for i in 0..side {
			for j in 0..side {
				let k = 2 * i + j;
				let (mpm_idx, rem) = if prev[k] {
					let mut m = 0;
					while m < 2 && self.cabac.bypass() {
						m += 1;
					}
					(m, 0)
				} else {
					(0, self.cabac.bypass_bits(5))
				};
				let (px, py) = (x0 + pb * j as i32, y0 + pb * i as i32);
				cu.intra_mode[k] = self.luma_intra_mode(px, py, pb, prev[k], mpm_idx, rem);
			}
		}
		if !split {
			cu.intra_mode = [cu.intra_mode[0]; 4];
		}
		const CHROMA_TABLE: [u32; 4] = [0, 26, 10, 1];
		let chroma_mode =
			if !self.cabac.decode(INTRA_CHROMA_PRED_MODE) { 4 } else { self.cabac.bypass_bits(2) };
		cu.intra_mode_c = if chroma_mode == 4 {
			cu.intra_mode[0]
		} else if cu.intra_mode[0] == CHROMA_TABLE[chroma_mode as usize] {
			34
		} else {
			CHROMA_TABLE[chroma_mode as usize]
		};
	}

	fn luma_intra_mode(
		&mut self,
		x0: i32,
		y0: i32,
		pb: i32,
		prev_flag: bool,
		mpm_idx: usize,
		rem: u32,
	) -> u32 {
		let log2_ctb = self.sps.log2_ctb_size;
		let cand_left =
			if self.available(x0, y0, x0 - 1, y0) && self.pic.mvf[self.i4(x0 - 1, y0)].pred == PF_INTRA {
				u32::from(self.pic.ipm[self.i4(x0 - 1, y0)])
			} else {
				intra::MODE_DC
			};
		let cand_up = if self.available(x0, y0, x0, y0 - 1)
			&& self.pic.mvf[self.i4(x0, y0 - 1)].pred == PF_INTRA
			&& ((y0 - 1) >> log2_ctb) == (y0 >> log2_ctb)
		{
			u32::from(self.pic.ipm[self.i4(x0, y0 - 1)])
		} else {
			intra::MODE_DC
		};
		let mut c = if cand_left == cand_up {
			if cand_left < 2 {
				[0, 1, 26]
			} else {
				[cand_left, 2 + ((cand_left + 29) % 32), 2 + ((cand_left - 2 + 1) % 32)]
			}
		} else {
			let third = if cand_left != 0 && cand_up != 0 {
				0
			} else if cand_left != 1 && cand_up != 1 {
				1
			} else {
				26
			};
			[cand_left, cand_up, third]
		};
		let mode = if prev_flag {
			c[mpm_idx]
		} else {
			c.sort_unstable();
			let mut m = rem;
			for &v in &c {
				if m >= v {
					m += 1;
				}
			}
			m
		};
		let ipm = mode as u8;
		self.fill_4x4(x0, y0, pb, pb, |p, i| {
			p.ipm[i] = ipm;
			p.mvf[i] = MvField { pred: PF_INTRA, ..Default::default() };
		});
		mode
	}

	// -----------------------------------------------------------------
	// Transform tree / unit (§7.3.8.8-10)
	// -----------------------------------------------------------------

	#[allow(clippy::too_many_arguments)]
	fn transform_tree(
		&mut self,
		cu: &Cu,
		x0: i32,
		y0: i32,
		x_base: i32,
		y_base: i32,
		log2: u32,
		depth: u32,
		blk_idx: usize,
		parent_cbf: [bool; 2],
	) -> Result<(), DecodeError> {
		let sps = self.sps.clone();
		let split = if log2 <= sps.log2_max_tb_size
			&& log2 > sps.log2_min_tb_size
			&& depth < cu.max_trafo_depth
			&& !(cu.intra_split && depth == 0)
		{
			self.cabac.decode(SPLIT_TRANSFORM_FLAG + 5 - log2 as usize)
		} else {
			let inter_split = sps.max_transform_hierarchy_depth_inter == 0
				&& !cu.intra
				&& cu.part != PartMode::P2Nx2N
				&& depth == 0;
			log2 > sps.log2_max_tb_size || (cu.intra_split && depth == 0) || inter_split
		};
		let mut cbf = parent_cbf;
		if log2 > 2 {
			for c in 0..2 {
				if depth == 0 || parent_cbf[c] {
					cbf[c] = self.cabac.decode(CBF_CB_CR + depth as usize);
				} else {
					cbf[c] = false;
				}
			}
		}
		if split {
			let h = 1 << (log2 - 1);
			self.transform_tree(cu, x0, y0, x0, y0, log2 - 1, depth + 1, 0, cbf)?;
			self.transform_tree(cu, x0 + h, y0, x0, y0, log2 - 1, depth + 1, 1, cbf)?;
			self.transform_tree(cu, x0, y0 + h, x0, y0, log2 - 1, depth + 1, 2, cbf)?;
			self.transform_tree(cu, x0 + h, y0 + h, x0, y0, log2 - 1, depth + 1, 3, cbf)?;
		} else {
			let cbf_luma = if cu.intra || depth != 0 || cbf[0] || cbf[1] {
				self.cabac.decode(CBF_LUMA + usize::from(depth == 0))
			} else {
				true
			};
			let luma_mode = if cu.intra_split && depth >= 1 {
				cu.intra_mode[if depth == 1 { blk_idx } else { self.split_blk(cu, x0, y0) }]
			} else {
				cu.intra_mode[0]
			};
			self.transform_unit(cu, x0, y0, x_base, y_base, log2, blk_idx, cbf_luma, cbf, luma_mode)?;
			if cbf_luma {
				let n = 1 << log2;
				self.fill_4x4(x0, y0, n, n, |p, i| p.cbf_luma[i] = 1);
			}
			self.boundary_strengths(x0, y0, log2, true);
		}
		Ok(())
	}

	/// Which NxN prediction block contains (x0, y0) of an intra-split CU.
	fn split_blk(&self, cu: &Cu, x0: i32, y0: i32) -> usize {
		let h = 1 << (cu.log2_size - 1);
		usize::from(x0 >= cu.x + h) + 2 * usize::from(y0 >= cu.y + h)
	}

	#[allow(clippy::too_many_arguments)]
	fn transform_unit(
		&mut self,
		cu: &Cu,
		x0: i32,
		y0: i32,
		x_base: i32,
		y_base: i32,
		log2: u32,
		blk_idx: usize,
		cbf_luma: bool,
		cbf: [bool; 2],
		luma_mode: u32,
	) -> Result<(), DecodeError> {
		let n = 1i32 << log2;
		if cu.intra {
			self.intra_pred(x0, y0, log2, 0, luma_mode);
		}
		let chroma_here = log2 > 2;
		let chroma_mode = cu.intra_mode_c;
		if cbf_luma || cbf[0] || cbf[1] {
			if self.pps.cu_qp_delta_enabled && !self.is_cu_qp_delta_coded {
				let mut v = 0i32;
				let mut inc = 0;
				while v < 5 && self.cabac.decode(CU_QP_DELTA + inc) {
					v += 1;
					inc = 1;
				}
				if v >= 5 {
					let mut k = 0;
					let mut suffix = 0;
					while k < 7 && self.cabac.bypass() {
						suffix += 1 << k;
						k += 1;
					}
					if k == 7 {
						return Err(DecodeError("hevc: invalid cu_qp_delta"));
					}
					suffix += self.cabac.bypass_bits(k) as i32;
					v += suffix;
				}
				if v != 0 && self.cabac.bypass() {
					v = -v;
				}
				let off = self.sps.qp_bd_offset;
				if v < -(26 + off / 2) || v > 25 + off / 2 {
					return Err(DecodeError("hevc: cu_qp_delta out of range"));
				}
				self.cu_qp_delta = v;
				self.is_cu_qp_delta_coded = true;
				self.set_qp_y();
			}
			let scan = |mode: u32, size_log2: u32| -> usize {
				if cu.intra && size_log2 == 2 {
					if (6..=14).contains(&mode) {
						SCAN_VERT
					} else if (22..=30).contains(&mode) {
						SCAN_HORIZ
					} else {
						SCAN_DIAG
					}
				} else {
					SCAN_DIAG
				}
			};
			if cbf_luma {
				self.residual(
					cu,
					x0,
					y0,
					log2,
					0,
					scan(luma_mode, if log2 <= 3 { 2 } else { log2 }),
					luma_mode,
				);
			}
			if chroma_here {
				let (xc, yc, l2c) = (x0 / 2, y0 / 2, log2 - 1);
				if cu.intra {
					self.intra_pred(xc, yc, l2c, 1, chroma_mode);
				}
				if cbf[0] {
					self.residual(cu, xc, yc, l2c, 1, scan(chroma_mode, l2c), chroma_mode);
				}
				if cu.intra {
					self.intra_pred(xc, yc, l2c, 2, chroma_mode);
				}
				if cbf[1] {
					self.residual(cu, xc, yc, l2c, 2, scan(chroma_mode, l2c), chroma_mode);
				}
			} else if blk_idx == 3 {
				let (xc, yc) = (x_base / 2, y_base / 2);
				if cu.intra {
					self.intra_pred(xc, yc, 2, 1, chroma_mode);
				}
				if cbf[0] {
					self.residual(cu, xc, yc, 2, 1, scan(chroma_mode, 2), chroma_mode);
				}
				if cu.intra {
					self.intra_pred(xc, yc, 2, 2, chroma_mode);
				}
				if cbf[1] {
					self.residual(cu, xc, yc, 2, 2, scan(chroma_mode, 2), chroma_mode);
				}
			}
		} else if cu.intra {
			if chroma_here {
				self.intra_pred(x0 / 2, y0 / 2, log2 - 1, 1, chroma_mode);
				self.intra_pred(x0 / 2, y0 / 2, log2 - 1, 2, chroma_mode);
			} else if blk_idx == 3 {
				self.intra_pred(x_base / 2, y_base / 2, 2, 1, chroma_mode);
				self.intra_pred(x_base / 2, y_base / 2, 2, 2, chroma_mode);
			}
		}
		let _ = n;
		Ok(())
	}

	/// Intra prediction of one transform block (component coordinates).
	fn intra_pred(&mut self, x: i32, y: i32, log2: u32, c: usize, mode: u32) {
		let n = 1usize << log2;
		let shift = u32::from(c > 0);
		let unit = if c == 0 { 4 } else { 2 };
		let bit_depth = self.sps.bit_depth;
		let (xl, yl) = (x << shift, y << shift);
		let constrained = self.pps.constrained_intra_pred;
		let refs = {
			let this = &*self;
			let plane = &this.pic.planes[c];
			intra::build_refs(plane, x as usize, y as usize, n, unit, bit_depth, |xn, yn| {
				let (xn, yn) = ((xn as i32) << shift, (yn as i32) << shift);
				this.available(xl, yl, xn, yn)
					&& (!constrained || this.pic.mvf[this.i4(xn, yn)].pred == PF_INTRA)
			})
		};
		let mut refs = refs;
		if c == 0 {
			intra::filter_refs(&mut refs, n, mode, self.sps.strong_intra_smoothing, bit_depth);
		}
		let plane = &mut self.pic.planes[c];
		let at = plane.idx(x as isize, y as isize);
		intra::predict(plane, at, n, mode, &refs, c == 0, bit_depth);
	}

	// -----------------------------------------------------------------
	// Residual coding (§7.3.8.11, §9.3.4.2.5-7) and reconstruction
	// -----------------------------------------------------------------

	#[allow(clippy::too_many_arguments)]
	fn residual(
		&mut self,
		cu: &Cu,
		x0: i32,
		y0: i32,
		log2: u32,
		c: usize,
		scan_idx: usize,
		pred_mode: u32,
	) {
		let sps = self.sps.clone();
		let pps = self.pps.clone();
		let n = 1usize << log2;
		let mut transform_skip = false;
		if pps.transform_skip_enabled && !cu.bypass && log2 == 2 {
			transform_skip = self.cabac.decode(TRANSFORM_SKIP_FLAG + usize::from(c > 0));
		}
		// Last significant coefficient position.
		let (ctx_off, ctx_shift) =
			if c == 0 { (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2) } else { (15, log2 - 2) };
		let max = (log2 << 1) - 1;
		let prefix = |cab: &mut Cabac, base: usize| {
			let mut i = 0;
			while i < max && cab.decode(base + ((i >> ctx_shift) + ctx_off) as usize) {
				i += 1;
			}
			i
		};
		let px = prefix(&mut self.cabac, LAST_SIGNIFICANT_COEFF_X_PREFIX);
		let py = prefix(&mut self.cabac, LAST_SIGNIFICANT_COEFF_Y_PREFIX);
		let suffix = |cab: &mut Cabac, p: u32| -> u32 {
			if p > 3 {
				let len = (p >> 1) - 1;
				let s = cab.bypass_bits(len);
				(1 << ((p >> 1) - 1)) * (2 + (p & 1)) + s
			} else {
				p
			}
		};
		let mut last_x = suffix(&mut self.cabac, px) as usize;
		let mut last_y = suffix(&mut self.cabac, py) as usize;
		if scan_idx == SCAN_VERT {
			std::mem::swap(&mut last_x, &mut last_y);
		}
		let scans = scans();
		let sub = &scans.sub4[scan_idx];
		let cg_scan = &scans.cg[scan_idx][(log2 - 2) as usize];
		let cgs = n >> 2;
		// Index of the last sub-block and position.
		let last_cg = cg_scan
			.iter()
			.position(|&(x, y)| usize::from(x) == last_x >> 2 && usize::from(y) == last_y >> 2)
			.unwrap_or(0);
		let last_pos = sub
			.iter()
			.position(|&(x, y)| usize::from(x) == last_x & 3 && usize::from(y) == last_y & 3)
			.unwrap_or(0);

		let coeffs = &mut self.coeffs;
		coeffs[..n * n].fill(0);
		let mut csbf = [0u8; 64];
		let mut greater1_ctx_prev_zero = false; // greater1Ctx == 0 at the end of the previous sub-block
		let mut first_subblock = true;
		let mut max_x = 0usize;
		let mut max_y = 0usize;

		// Dequantisation parameters.
		let bit_depth = sps.bit_depth;
		let (scale, shift, add, matrix): (i64, u32, i64, Option<ScaleMatrix>) = if cu.bypass {
			(0, 0, 0, None)
		} else {
			let qp = if c == 0 {
				self.qp_y + sps.qp_bd_offset
			} else {
				let off = if c == 1 {
					pps.cb_qp_offset + self.sh.cb_qp_offset
				} else {
					pps.cr_qp_offset + self.sh.cr_qp_offset
				};
				let qp_i = (self.qp_y + off).clamp(-sps.qp_bd_offset, 57);
				chroma_qp(qp_i) + sps.qp_bd_offset
			};
			let shift = bit_depth + log2 - 5;
			let scale = i64::from(LEVEL_SCALE[(qp % 6) as usize] << (qp / 6));
			let matrix = sps.scaling_list.as_ref().filter(|_| !(transform_skip && log2 > 2)).map(|sl| {
				let sl = pps.scaling_list.as_ref().unwrap_or(sl);
				let m = 3 * usize::from(!cu.intra) + c;
				let dc = if log2 >= 4 { sl.dc[(log2 - 4) as usize][m] } else { 16 };
				(&sl.sl[(log2 - 2) as usize][m], dc)
			});
			(scale, shift, 1i64 << (shift - 1), matrix)
		};

		let sign_hiding_allowed = pps.sign_data_hiding && !cu.bypass;
		let cabac = &mut self.cabac;
		for i in (0..=last_cg).rev() {
			let (xs, ys) = (usize::from(cg_scan[i].0), usize::from(cg_scan[i].1));
			// Coded sub-block flag.
			let mut infer_dc = false;
			let coded = if i < last_cg && i > 0 {
				let mut csbf_ctx = 0;
				if xs + 1 < cgs {
					csbf_ctx += csbf[ys * 8 + xs + 1];
				}
				if ys + 1 < cgs {
					csbf_ctx += csbf[(ys + 1) * 8 + xs];
				}
				let inc = usize::from(csbf_ctx.min(1)) + if c > 0 { 2 } else { 0 };
				let f = cabac.decode(SIGNIFICANT_COEFF_GROUP_FLAG + inc);
				infer_dc = true;
				f
			} else {
				true
			};
			csbf[ys * 8 + xs] = u8::from(coded);
			let prev_csbf = (if xs + 1 < cgs { csbf[ys * 8 + xs + 1] } else { 0 })
				| (if ys + 1 < cgs { csbf[(ys + 1) * 8 + xs] } else { 0 }) << 1;
			// Significant coefficients, in reverse scan order.
			let mut sig_pos = [0u8; 16];
			let mut nsig = 0usize;
			let start = if i == last_cg {
				sig_pos[0] = last_pos as u8;
				nsig = 1;
				last_pos as isize - 1
			} else {
				15
			};
			if coded {
				let mut n_pos = start;
				while n_pos >= 0 {
					let p = n_pos as usize;
					let (xp, yp) = (usize::from(sub[p].0), usize::from(sub[p].1));
					if p == 0 && infer_dc && nsig == 0 {
						sig_pos[nsig] = 0;
						nsig += 1;
						break;
					}
					let sig_ctx = if log2 == 2 {
						usize::from(CTX_IDX_MAP_4X4[(yp << 2) + xp])
					} else if xs == 0 && ys == 0 && p == 0 {
						0
					} else {
						let s = match prev_csbf {
							0 => {
								if xp + yp == 0 {
									2
								} else if xp + yp < 3 {
									1
								} else {
									0
								}
							}
							1 => {
								if yp == 0 {
									2
								} else if yp == 1 {
									1
								} else {
									0
								}
							}
							2 => {
								if xp == 0 {
									2
								} else if xp == 1 {
									1
								} else {
									0
								}
							}
							_ => 2,
						};
						if c == 0 {
							s + if xs > 0 || ys > 0 { 3 } else { 0 }
								+ if log2 == 3 { if scan_idx == 0 { 9 } else { 15 } } else { 21 }
						} else {
							s + if log2 == 3 { 9 } else { 12 }
						}
					};
					let inc = if c == 0 { sig_ctx } else { 27 + sig_ctx };
					if cabac.decode(SIGNIFICANT_COEFF_FLAG + inc) {
						sig_pos[nsig] = p as u8;
						nsig += 1;
					}
					n_pos -= 1;
				}
			}
			if nsig == 0 {
				continue;
			}
			// Levels.
			let mut ctx_set = if i == 0 || c > 0 { 0 } else { 2 };
			if !first_subblock && greater1_ctx_prev_zero {
				ctx_set += 1;
			}
			first_subblock = false;
			let mut greater1_ctx = 1usize;
			let mut gt1 = [false; 16];
			let mut first_g1: Option<usize> = None;
			for m in 0..nsig.min(8) {
				let inc = ctx_set * 4 + greater1_ctx + if c > 0 { 16 } else { 0 };
				let f = cabac.decode(COEFF_ABS_LEVEL_GREATER1_FLAG + inc);
				gt1[m] = f;
				if f {
					greater1_ctx = 0;
					if first_g1.is_none() {
						first_g1 = Some(m);
					}
				} else if greater1_ctx > 0 && greater1_ctx < 3 {
					greater1_ctx += 1;
				}
			}
			greater1_ctx_prev_zero = greater1_ctx == 0;
			let mut gt2 = false;
			if first_g1.is_some() {
				gt2 = cabac.decode(COEFF_ABS_LEVEL_GREATER2_FLAG + ctx_set + if c > 0 { 4 } else { 0 });
			}
			let first_nz = sig_pos[nsig - 1];
			let last_nz = sig_pos[0];
			let sign_hidden = sign_hiding_allowed && (last_nz as i32 - first_nz as i32) >= 4;
			let nsigns = if sign_hidden { nsig - 1 } else { nsig };
			let signs =
				cabac.bypass_bits(nsigns as u32) << (32 - nsigns as u32).min(31) << u32::from(nsigns == 0);
			let mut signs = if nsigns == 0 { 0 } else { signs };
			let mut rice = 0u32;
			let mut sum_abs = 0i64;
			for m in 0..nsig {
				let mut level: i64 = if m < 8 {
					let base = 1 + i64::from(gt1[m]) + i64::from(Some(m) == first_g1 && gt2);
					let threshold = if m < 8 { if Some(m) == first_g1 { 3 } else { 2 } } else { 1 };
					if base == threshold {
						let rem = coeff_abs_level_remaining(cabac, rice);
						let l = base + i64::from(rem);
						if l > 3 * (1 << rice) {
							rice = (rice + 1).min(4);
						}
						l
					} else {
						base
					}
				} else {
					let rem = coeff_abs_level_remaining(cabac, rice);
					let l = 1 + i64::from(rem);
					if l > 3 * (1 << rice) {
						rice = (rice + 1).min(4);
					}
					l
				};
				let p = usize::from(sig_pos[m]);
				if sign_hidden {
					sum_abs += level;
					if p == usize::from(first_nz) {
						if sum_abs & 1 != 0 {
							level = -level;
						}
					} else if signs & 0x8000_0000 != 0 {
						level = -level;
						signs <<= 1;
					} else {
						signs <<= 1;
					}
				} else {
					if signs & 0x8000_0000 != 0 {
						level = -level;
					}
					signs <<= 1;
				}
				let (xc, yc) = ((xs << 2) + usize::from(sub[p].0), (ys << 2) + usize::from(sub[p].1));
				let v = if cu.bypass {
					level
				} else {
					let m_scale = match matrix {
						Some((mat, dc)) => {
							if xc == 0 && yc == 0 && log2 >= 4 {
								i64::from(dc)
							} else {
								let pos = match log2 {
									2 => (yc << 2) + xc,
									3 => (yc << 3) + xc,
									4 => ((yc >> 1) << 3) + (xc >> 1),
									_ => ((yc >> 2) << 3) + (xc >> 2),
								};
								i64::from(mat[pos])
							}
						}
						None => 16,
					};
					((level * scale * m_scale + add) >> shift).clamp(-32768, 32767)
				};
				coeffs[yc * n + xc] = v as i32;
				max_x = max_x.max(xc);
				max_y = max_y.max(yc);
			}
		}

		// Inverse transform.
		if cu.bypass {
			// Residual is the coefficients.
		} else if transform_skip {
			transform::transform_skip(&mut coeffs[..], n, bit_depth);
		} else if cu.intra && c == 0 && log2 == 2 {
			transform::idst4(&mut coeffs[..], bit_depth);
		} else {
			transform::idct(
				&mut coeffs[..],
				n,
				max_x,
				max_y,
				bit_depth,
				&mut self.tr_tmp[..],
				&mut self.tr_tmp2[..],
			);
		}
		let _ = pred_mode;
		// Add to the prediction.
		let max_v = (1i32 << bit_depth) - 1;
		let plane = &mut self.pic.planes[c];
		let at = plane.idx(x0 as isize, y0 as isize);
		let s = plane.stride;
		for y in 0..n {
			let row = &mut plane.data[at + y * s..at + y * s + n];
			for (d, &r) in row.iter_mut().zip(&coeffs[y * n..y * n + n]) {
				*d = (i32::from(*d) + r).clamp(0, max_v) as u16;
			}
		}
	}

	// -----------------------------------------------------------------
	// Deblocking boundary strengths (§8.7.2.3-4)
	// -----------------------------------------------------------------

	fn mark_cu_edges_and_bs(&mut self, x0: i32, y0: i32, log2: u32, _cu: &Cu) {
		let n = 1 << log2;
		self.edge_bs(x0, y0, n, n, true, true);
	}

	/// bS of the transform block edges (left and top) of a TU, or of a CU
	/// coded without residual.
	fn boundary_strengths(&mut self, x0: i32, y0: i32, log2: u32, _tu: bool) {
		let n = 1 << log2;
		self.edge_bs(x0, y0, n, n, true, true);
	}

	/// bS of the left (`left`) and/or top edge of a `w`x`h` block at
	/// (`x0`, `y0`) (§8.7.2.3-4). `transform`: the edge is a transform block
	/// edge (non-zero coefficients count); otherwise a prediction block edge
	/// inside a coding unit (motion only).
	fn edge_bs(&mut self, x0: i32, y0: i32, w: i32, h: i32, transform: bool, left: bool) {
		if self.sh.deblocking_disabled {
			return;
		}
		let (pw, ph) = (self.sps.width as i32, self.sps.height as i32);
		if y0 > 0 && y0 & 7 == 0 && (left || !transform) && self.edge_ok(x0, y0, x0, y0 - 1) {
			for x in (x0..(x0 + w).min(pw)).step_by(4) {
				let bs = self.bs_between(x, y0, x, y0 - 1, transform);
				let idx = self.i4(x, y0);
				self.pic.bs_h[idx] = bs;
			}
		}
		if x0 > 0 && x0 & 7 == 0 && (left || !transform) && self.edge_ok(x0, y0, x0 - 1, y0) {
			for y in (y0..(y0 + h).min(ph)).step_by(4) {
				let bs = self.bs_between(x0, y, x0 - 1, y, transform);
				let idx = self.i4(x0, y);
				self.pic.bs_v[idx] = bs;
			}
		}
	}

	/// Prediction block edges inside a coding unit (motion-based bS).
	fn pu_edge_bs(&mut self, cu_x: i32, cu_y: i32, x0: i32, y0: i32, w: i32, h: i32) {
		if self.sh.deblocking_disabled {
			return;
		}
		let (pw, ph) = (self.sps.width as i32, self.sps.height as i32);
		if y0 > cu_y && y0 & 7 == 0 {
			for x in (x0..(x0 + w).min(pw)).step_by(4) {
				let bs = self.bs_between(x, y0, x, y0 - 1, false);
				let idx = self.i4(x, y0);
				self.pic.bs_h[idx] = bs;
			}
		}
		if x0 > cu_x && x0 & 7 == 0 {
			for y in (y0..(y0 + h).min(ph)).step_by(4) {
				let bs = self.bs_between(x0, y, x0 - 1, y, false);
				let idx = self.i4(x0, y);
				self.pic.bs_v[idx] = bs;
			}
		}
	}

	/// Slice and tile rules for filtering across the edge between the
	/// block at (`x0`, `y0`) and its neighbour sample (`xn`, `yn`).
	fn edge_ok(&self, x0: i32, y0: i32, xn: i32, yn: i32) -> bool {
		let ctb_c = self.ctb_index(x0, y0);
		let ctb_n = self.ctb_index(xn, yn);
		if ctb_n == ctb_c {
			return true;
		}
		let l = &*self.layout;
		if !self.pps.loop_filter_across_tiles
			&& l.tile_id[l.ctb_addr_rs_to_ts[ctb_n] as usize]
				!= l.tile_id[l.ctb_addr_rs_to_ts[ctb_c] as usize]
		{
			return false;
		}
		!(self.pic.ctb_addr[ctb_n] != self.pic.ctb_addr[ctb_c] && !self.sh.loop_filter_across_slices)
	}

	fn bs_between(&self, xq: i32, yq: i32, xp: i32, yp: i32, transform_edge: bool) -> u8 {
		let q = self.pic.mvf[self.i4(xq, yq)];
		let p = self.pic.mvf[self.i4(xp, yp)];
		if q.pred == PF_INTRA || p.pred == PF_INTRA {
			return 2;
		}
		if transform_edge
			&& (self.pic.cbf_luma[self.i4(xq, yq)] != 0 || self.pic.cbf_luma[self.i4(xp, yp)] != 0)
		{
			return 1;
		}
		let qr = &self.refs.info;
		let pr = &self.pic.slices[self.pic.ctb_slice[self.ctb_index(xp, yp)] as usize].refs;
		u8::from(motion_differs(&q, qr, &p, pr))
	}

	fn ctb_index(&self, x: i32, y: i32) -> usize {
		let l = self.sps.log2_ctb_size;
		((y >> l) * self.sps.ctb_width as i32 + (x >> l)) as usize
	}

	// -----------------------------------------------------------------
	// Inter prediction units
	// -----------------------------------------------------------------

	fn prediction_unit(
		&mut self,
		cu: &mut Cu,
		x0: i32,
		y0: i32,
		w: i32,
		h: i32,
		part_idx: usize,
	) -> Result<(), DecodeError> {
		let skip = self.pic.skip[self.min_cb(x0, y0)] != 0;
		let pb = Pb {
			xcb: cu.x,
			ycb: cu.y,
			ncbs: 1 << cu.log2_size,
			x: x0,
			y: y0,
			w,
			h,
			part_idx,
			part: cu.part,
		};
		let mut mv;
		if !skip {
			cu.merge_flag = self.cabac.decode(MERGE_FLAG);
		} else {
			cu.merge_flag = true;
		}
		if skip || cu.merge_flag {
			let merge_idx = if self.sh.max_num_merge_cand > 1 {
				let mut i = usize::from(self.cabac.decode(MERGE_IDX));
				if i != 0 {
					while i < self.sh.max_num_merge_cand as usize - 1 && self.cabac.bypass() {
						i += 1;
					}
				}
				i
			} else {
				0
			};
			mv = self.merge_mode(pb, merge_idx);
		} else {
			let mut pred_idc = 0; // 0 L0, 1 L1, 2 BI
			if self.sh.slice_type == SliceType::B {
				pred_idc = if w + h == 12 {
					usize::from(self.cabac.decode(INTER_PRED_IDC + 4))
				} else if self.cabac.decode(INTER_PRED_IDC + self.ct_depth as usize) {
					2
				} else {
					usize::from(self.cabac.decode(INTER_PRED_IDC + 4))
				};
			}
			mv = MvField::default();
			let mut mvd = [[0i32; 2]; 2];
			let mut mvp_flag = [false; 2];
			if pred_idc != 1 {
				mv.ref_idx[0] = self.ref_idx(0) as i8;
				mvd[0] = self.mvd();
				mvp_flag[0] = self.cabac.decode(MVP_LX_FLAG);
				mv.pred |= PF_L0;
			}
			if pred_idc != 0 {
				mv.ref_idx[1] = self.ref_idx(1) as i8;
				if self.sh.mvd_l1_zero && pred_idc == 2 {
					mvd[1] = [0, 0];
				} else {
					mvd[1] = self.mvd();
				}
				mvp_flag[1] = self.cabac.decode(MVP_LX_FLAG);
				mv.pred |= PF_L1;
			}
			for l in 0..2 {
				if mv.pred & (1 << l) != 0 {
					let p = self.mvp(pb, l, mv.ref_idx[l] as usize);
					let mp = p[usize::from(mvp_flag[l])];
					mv.mv[l] = [(i32::from(mp[0]) + mvd[l][0]) as i16, (i32::from(mp[1]) + mvd[l][1]) as i16];
				}
			}
		}
		// Canonical form: unused lists carry no motion (candidates compare
		// whole fields).
		for l in 0..2 {
			if mv.pred & (1 << l) == 0 {
				mv.ref_idx[l] = -1;
				mv.mv[l] = [0, 0];
			}
		}
		self.fill_4x4(x0, y0, w, h, |p, i| p.mvf[i] = mv);
		self.pu_edge_bs(cu.x, cu.y, x0, y0, w, h);
		self.motion_compensate(x0, y0, w, h, &mv);
		Ok(())
	}

	fn ref_idx(&mut self, list: usize) -> usize {
		let n = self.sh.num_ref_idx[list] as usize;
		if n <= 1 {
			return 0;
		}
		let max = n - 1;
		let max_ctx = max.min(2);
		let mut i = 0;
		while i < max_ctx && self.cabac.decode(REF_IDX_L0 + i) {
			i += 1;
		}
		if i == 2 {
			while i < max && self.cabac.bypass() {
				i += 1;
			}
		}
		i
	}

	fn mvd(&mut self) -> [i32; 2] {
		let gx = self.cabac.decode(ABS_MVD_GREATER0_FLAG);
		let gy = self.cabac.decode(ABS_MVD_GREATER0_FLAG);
		let g1x = gx && self.cabac.decode(ABS_MVD_GREATER1_FLAG + 1);
		let g1y = gy && self.cabac.decode(ABS_MVD_GREATER1_FLAG + 1);
		let mut comp = |g0: bool, g1: bool| -> i32 {
			if !g0 {
				return 0;
			}
			let mut v = 1i32;
			if g1 {
				// abs_mvd_minus2: EG1.
				let mut k = 1u32;
				let mut val = 0i32;
				while k < 31 && self.cabac.bypass() {
					val += 1 << k;
					k += 1;
				}
				val += self.cabac.bypass_bits(k) as i32;
				v = 2 + val;
			}
			if self.cabac.bypass() { -v } else { v }
		};
		let x = comp(gx, g1x);
		let y = comp(gy, g1y);
		[x, y]
	}

	fn motion_compensate(&mut self, x0: i32, y0: i32, w: i32, h: i32, mv: &MvField) {
		let bit_depth = self.sps.bit_depth;
		let weights = self.sh.weights.as_ref();
		let (w, h) = (w as usize, h as usize);
		let lists: Vec<usize> = (0..2).filter(|&l| mv.pred & (1 << l) != 0).collect();
		let mut frames = [None, None];
		for &l in &lists {
			let f = self.refs.list[l].get(mv.ref_idx[l] as usize).and_then(|f| f.clone());
			match f {
				Some(f) => frames[l] = Some(f),
				None => return,
			}
		}
		for c in 0..3 {
			let (cx, cy, cw, ch) = if c == 0 { (x0, y0, w, h) } else { (x0 / 2, y0 / 2, w / 2, h / 2) };
			for (k, &l) in lists.iter().enumerate() {
				let f = frames[l].as_ref().expect("checked above");
				let m = mv.mv[l];
				let (ix, iy, fx, fy) = if c == 0 {
					(
						cx as isize + (m[0] >> 2) as isize,
						cy as isize + (m[1] >> 2) as isize,
						(m[0] & 3) as usize,
						(m[1] & 3) as usize,
					)
				} else {
					(
						cx as isize + (m[0] >> 3) as isize,
						cy as isize + (m[1] >> 3) as isize,
						(m[0] & 7) as usize,
						(m[1] & 7) as usize,
					)
				};
				mc::interpolate(
					&f.planes[c],
					ix,
					iy,
					fx,
					fy,
					cw,
					ch,
					c > 0,
					bit_depth,
					&mut self.pred[k][..cw * ch],
					&mut self.mc_tmp[..],
				);
			}
			let wp_of = |l: usize| -> mc::Wp {
				let wt = &weights.expect("weights").l[l][mv.ref_idx[l] as usize];
				if c == 0 {
					mc::Wp { weight: wt.luma_weight, offset: wt.luma_offset }
				} else {
					mc::Wp { weight: wt.chroma_weight[c - 1], offset: wt.chroma_offset[c - 1] }
				}
			};
			let denom =
				weights.map_or(0, |wt| if c == 0 { wt.luma_log2_denom } else { wt.chroma_log2_denom });
			let plane = &mut self.pic.planes[c];
			let at = plane.idx(cx as isize, cy as isize);
			let (p0, p1) = self.pred.split_at(1);
			if lists.len() == 2 {
				let wp = weights.map(|_| (wp_of(0), wp_of(1), denom));
				mc::put_bi(plane, at, &p0[0][..], &p1[0][..], cw, ch, bit_depth, wp);
			} else {
				let wp = weights.map(|_| (wp_of(lists[0]), denom));
				mc::put_uni(plane, at, &p0[0][..], cw, ch, bit_depth, wp);
			}
		}
	}
}

/// First CTB column of the tile column containing CTB column `cx`.
fn tile_col_start(layout: &Layout, cx: usize) -> usize {
	layout.col_bd.iter().rev().map(|&b| b as usize).find(|&b| b <= cx).unwrap_or(0)
}

/// `coeff_abs_level_remaining` binarisation (9.3.3.11).
fn coeff_abs_level_remaining(c: &mut Cabac, rice: u32) -> u32 {
	let mut prefix = 0u32;
	while prefix < 32 && c.bypass() {
		prefix += 1;
	}
	if prefix < 3 {
		(prefix << rice) + c.bypass_bits(rice)
	} else {
		let k = (prefix - 3 + rice).min(31);
		let suffix = if k > 16 {
			(c.bypass_bits(16) << (k - 16)) | c.bypass_bits(k - 16)
		} else {
			c.bypass_bits(k)
		};
		(((1u32 << (prefix - 3).min(31)) + 3 - 1) << rice).wrapping_add(suffix)
	}
}

/// Motion-based bS: 1 when the blocks use different reference pictures or
/// a different number of MVs, or MVs differ by one integer sample or more.
pub fn motion_differs(q: &MvField, qr: &RefInfo, p: &MvField, pr: &RefInfo) -> bool {
	let poc = |m: &MvField, r: &RefInfo, l: usize| r.poc[l][(m.ref_idx[l].max(0) as usize).min(15)];
	let far = |a: [i16; 2], b: [i16; 2]| {
		(i32::from(a[0]) - i32::from(b[0])).abs() >= 4 || (i32::from(a[1]) - i32::from(b[1])).abs() >= 4
	};
	if q.pred == PF_BI && p.pred == PF_BI {
		let (q0, q1, p0, p1) = (poc(q, qr, 0), poc(q, qr, 1), poc(p, pr, 0), poc(p, pr, 1));
		if p0 == q0 && q0 == q1 && p0 == p1 {
			(far(p.mv[0], q.mv[0]) || far(p.mv[1], q.mv[1]))
				&& (far(p.mv[1], q.mv[0]) || far(p.mv[0], q.mv[1]))
		} else if p0 == q0 && p1 == q1 {
			far(p.mv[0], q.mv[0]) || far(p.mv[1], q.mv[1])
		} else if p1 == q0 && p0 == q1 {
			far(p.mv[1], q.mv[0]) || far(p.mv[0], q.mv[1])
		} else {
			true
		}
	} else if q.pred != PF_BI && p.pred != PF_BI {
		let lq = if q.pred & PF_L0 != 0 { 0 } else { 1 };
		let lp = if p.pred & PF_L0 != 0 { 0 } else { 1 };
		if poc(q, qr, lq) == poc(p, pr, lp) { far(q.mv[lq], p.mv[lp]) } else { true }
	} else {
		true
	}
}
