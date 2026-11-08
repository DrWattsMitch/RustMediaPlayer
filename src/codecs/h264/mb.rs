//! Macroblock layer parsing (ISO/IEC 14496-10 §7.3.5) for both entropy
//! coders, with the neighbour-dependent context derivations of §9.2.1
//! (CAVLC nC) and §9.3.3.1.1 (CABAC ctxIdxInc). FFmpeg: `h264_cavlc.c`,
//! `h264_cabac.c`, `h264_mvpred.h` (`fill_decode_caches`).

use super::cabac::{BlockCat, Cabac, ctx};
use super::cavlc;
use super::consts::{
	BLK_XY, GOLOMB_TO_INTER_CBP, GOLOMB_TO_INTRA_CBP, RASTER_TO_BLK, ZIGZAG_4X4, ZIGZAG_8X8,
};
use super::decode::SliceDecoder;
use super::slice::SliceType;
use crate::codecs::bitstream::BitReader;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MbKind {
	#[default]
	I4x4,
	I8x8,
	I16x16,
	Pcm,
	PSkip,
	BSkip,
	/// Any non-skip inter macroblock (including B_Direct_16x16).
	Inter,
}

/// Per-macroblock state of the picture being decoded: everything later
/// macroblocks, the deblocking filter and direct prediction need.
#[derive(Debug, Clone, Copy, Default)]
pub struct MbInfo {
	/// Slice number + 1; 0 = not decoded (yet).
	pub slice: u32,
	pub kind: MbKind,
	/// B_Skip or B_Direct_16x16.
	pub direct16: bool,
	/// Bit n (0-3): luma 8x8 block n has coefficients; bits 4-5: chroma.
	pub cbp: u8,
	pub chroma_pred: u8,
	pub t8x8: bool,
	/// QP'Y for deblocking (0 for I_PCM) and QPc per chroma component.
	pub qp: u8,
	pub qpc: [u8; 2],
	/// Non-zero coefficient counts: luma raster 4x4 (0-15), Cb (16-19),
	/// Cr (20-23) raster 2x2.
	pub nnz: [u8; 24],
	/// coded_block_flag of the DC blocks: bit 0 luma, bit 1 Cb, bit 2 Cr.
	pub dc_cbf: u8,
	/// Intra 4x4/8x8 prediction modes per raster 4x4 block (2 otherwise).
	pub ipred: [u8; 16],
	pub mv: [[[i16; 2]; 16]; 2],
	pub ref_idx: [[i8; 4]; 2],
	pub ref_id: [[u32; 4]; 2],
	/// |mvd| per list, raster 4x4 and component, saturated (CABAC context).
	pub mvd: [[[u8; 2]; 16]; 2],
	/// Bit n: 8x8 block n was predicted in direct mode.
	pub direct8: u8,
}

impl MbInfo {
	pub fn is_intra(&self) -> bool {
		matches!(self.kind, MbKind::I4x4 | MbKind::I8x8 | MbKind::I16x16 | MbKind::Pcm)
	}

	pub fn is_skip(&self) -> bool {
		matches!(self.kind, MbKind::PSkip | MbKind::BSkip)
	}
}

/// Coefficient levels of the current macroblock, in raster order per block.
#[derive(Debug, Clone)]
pub struct Residual {
	pub luma: [[i32; 16]; 16],
	pub luma8: [[i32; 64]; 4],
	pub dc: [i32; 16],
	pub chroma_dc: [[i32; 4]; 2],
	pub chroma_ac: [[[i32; 16]; 4]; 2],
	pub pcm: [u8; 384],
}

impl Default for Residual {
	fn default() -> Self {
		Self {
			luma: [[0; 16]; 16],
			luma8: [[0; 64]; 4],
			dc: [0; 16],
			chroma_dc: [[0; 4]; 2],
			chroma_ac: [[[0; 16]; 4]; 2],
			pcm: [0; 384],
		}
	}
}

/// Syntax decoder for the slice data.
pub enum Entropy<'a> {
	Cabac(Box<Cabac<'a>>),
	Cavlc(BitReader<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BadMb(pub &'static str);

pub type MbResult<T> = Result<T, BadMb>;

/// Inter partition prediction direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pred {
	L0,
	L1,
	Bi,
	Direct,
}

impl Pred {
	pub fn uses(self, list: usize) -> bool {
		matches!((self, list), (Pred::L0, 0) | (Pred::L1, 1) | (Pred::Bi, _))
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
	S16x16,
	S16x8,
	S8x16,
	S8x8,
}

/// Parsed macroblock type.
#[derive(Debug, Clone, Copy)]
pub enum MbType {
	I4x4,
	I16x16 {
		pred: u8,
		cbp: u8,
	},
	Pcm,
	/// Inter macroblock with up to two partitions (or 8x8 sub-partitions).
	Inter {
		shape: Shape,
		pred: [Pred; 2],
		ref0: bool,
	},
	/// B_Direct_16x16.
	Direct,
}

/// `(shape, pred0, pred1)` for B mb_type 1..=21 (Table 7-14).
const B_TYPES: [(Shape, Pred, Pred); 21] = {
	use Pred::*;
	use Shape::*;
	[
		(S16x16, L0, L0),
		(S16x16, L1, L1),
		(S16x16, Bi, Bi),
		(S16x8, L0, L0),
		(S8x16, L0, L0),
		(S16x8, L1, L1),
		(S8x16, L1, L1),
		(S16x8, L0, L1),
		(S8x16, L0, L1),
		(S16x8, L1, L0),
		(S8x16, L1, L0),
		(S16x8, L0, Bi),
		(S8x16, L0, Bi),
		(S16x8, L1, Bi),
		(S8x16, L1, Bi),
		(S16x8, Bi, L0),
		(S8x16, Bi, L0),
		(S16x8, Bi, L1),
		(S8x16, Bi, L1),
		(S16x8, Bi, Bi),
		(S8x16, Bi, Bi),
	]
};

/// Sub-macroblock partitioning: number of partitions and size in 4x4 units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubType {
	pub pred: Pred,
	pub parts: usize,
	pub w: usize,
	pub h: usize,
}

pub fn p_sub_type(t: u32) -> Option<SubType> {
	let (parts, w, h) = match t {
		0 => (1, 2, 2),
		1 => (2, 2, 1),
		2 => (2, 1, 2),
		3 => (4, 1, 1),
		_ => return None,
	};
	Some(SubType { pred: Pred::L0, parts, w, h })
}

pub fn b_sub_type(t: u32) -> Option<SubType> {
	use Pred::*;
	let (pred, parts, w, h) = match t {
		0 => (Direct, 4, 1, 1),
		1 => (L0, 1, 2, 2),
		2 => (L1, 1, 2, 2),
		3 => (Bi, 1, 2, 2),
		4 => (L0, 2, 2, 1),
		5 => (L0, 2, 1, 2),
		6 => (L1, 2, 2, 1),
		7 => (L1, 2, 1, 2),
		8 => (Bi, 2, 2, 1),
		9 => (Bi, 2, 1, 2),
		10 => (L0, 4, 1, 1),
		11 => (L1, 4, 1, 1),
		12 => (Bi, 4, 1, 1),
		_ => return None,
	};
	Some(SubType { pred, parts, w, h })
}

fn intra_type(t: u32) -> Option<MbType> {
	match t {
		0 => Some(MbType::I4x4),
		1..=24 => {
			let pred = ((t - 1) % 4) as u8;
			let chroma = ((t - 1) / 4 % 3) as u8;
			let luma = if t >= 13 { 15 } else { 0 };
			Some(MbType::I16x16 { pred, cbp: luma | (chroma << 4) })
		}
		25 => Some(MbType::Pcm),
		_ => None,
	}
}

impl<'a> SliceDecoder<'a> {
	// -----------------------------------------------------------------------
	// Neighbour access

	/// Neighbouring macroblock at offset (dx, dy) if it is available
	/// (inside the picture and in the current slice).
	pub fn nb_mb(&self, dx: i32, dy: i32) -> Option<&MbInfo> {
		let x = self.mbx as i32 + dx;
		let y = self.mby as i32 + dy;
		if x < 0 || y < 0 || x >= self.f.mb_w as i32 || y >= self.f.mb_h as i32 {
			return None;
		}
		let mb = &self.f.mbs[y as usize * self.f.mb_w + x as usize];
		(mb.slice == self.slice_num).then_some(mb)
	}

	/// The macroblock and raster 4x4 index covering 4x4 position (x4, y4)
	/// relative to the current macroblock (-1..=4). Positions inside the
	/// current macroblock resolve to `cur`.
	pub fn nb4<'b>(&'b self, cur: &'b MbInfo, x4: i32, y4: i32) -> Option<(&'b MbInfo, usize)> {
		let (dx, dy) = (x4.div_euclid(4), y4.div_euclid(4));
		let idx = (y4.rem_euclid(4) * 4 + x4.rem_euclid(4)) as usize;
		if dx == 0 && dy == 0 {
			return Some((cur, idx));
		}
		self.nb_mb(dx, dy).map(|m| (m, idx))
	}

	/// Chroma 4x4 neighbour (2x2 grid per component): returns the macroblock
	/// and the index into `nnz` (16 + 4 * comp + raster).
	fn nb_chroma<'b>(
		&'b self,
		cur: &'b MbInfo,
		comp: usize,
		x2: i32,
		y2: i32,
	) -> Option<(&'b MbInfo, usize)> {
		let (dx, dy) = (x2.div_euclid(2), y2.div_euclid(2));
		let idx = 16 + 4 * comp + (y2.rem_euclid(2) * 2 + x2.rem_euclid(2)) as usize;
		if dx == 0 && dy == 0 {
			return Some((cur, idx));
		}
		self.nb_mb(dx, dy).map(|m| (m, idx))
	}

	// -----------------------------------------------------------------------
	// Entropy-coder dispatch for macroblock-level syntax elements

	fn cabac(&mut self) -> Option<&mut Cabac<'a>> {
		match &mut self.entropy {
			Entropy::Cabac(c) => Some(c),
			Entropy::Cavlc(_) => None,
		}
	}

	/// CABAC mb_skip_flag (9.3.3.1.1.1).
	pub fn cabac_skip_flag(&mut self) -> bool {
		let cond = |m: Option<&MbInfo>| m.is_some_and(|m| !m.is_skip()) as usize;
		let inc = cond(self.nb_mb(-1, 0)) + cond(self.nb_mb(0, -1));
		let base = if self.h.slice_type == SliceType::B { ctx::MB_SKIP_B } else { ctx::MB_SKIP_P };
		self.cabac().is_some_and(|c| c.decision(base + inc))
	}

	fn read_mb_type(&mut self) -> MbResult<MbType> {
		let st = self.h.slice_type;
		let raw = match &mut self.entropy {
			Entropy::Cavlc(bs) => bs.ue(),
			Entropy::Cabac(_) => return self.cabac_mb_type(),
		};
		let bad = BadMb("invalid mb_type");
		match st {
			SliceType::I => intra_type(raw).ok_or(bad),
			SliceType::P => match raw {
				0 => Ok(MbType::Inter { shape: Shape::S16x16, pred: [Pred::L0; 2], ref0: false }),
				1 => Ok(MbType::Inter { shape: Shape::S16x8, pred: [Pred::L0; 2], ref0: false }),
				2 => Ok(MbType::Inter { shape: Shape::S8x16, pred: [Pred::L0; 2], ref0: false }),
				3 => Ok(MbType::Inter { shape: Shape::S8x8, pred: [Pred::L0; 2], ref0: false }),
				4 => Ok(MbType::Inter { shape: Shape::S8x8, pred: [Pred::L0; 2], ref0: true }),
				_ => intra_type(raw - 5).ok_or(bad),
			},
			SliceType::B => match raw {
				0 => Ok(MbType::Direct),
				1..=21 => {
					let (shape, a, b) = B_TYPES[raw as usize - 1];
					Ok(MbType::Inter { shape, pred: [a, b], ref0: false })
				}
				22 => Ok(MbType::Inter { shape: Shape::S8x8, pred: [Pred::Bi; 2], ref0: false }),
				_ => intra_type(raw - 23).ok_or(bad),
			},
		}
	}

	/// CABAC mb_type for I/P/B slices (9.3.2.5, 9.3.3.1.1.3).
	fn cabac_mb_type(&mut self) -> MbResult<MbType> {
		let st = self.h.slice_type;
		match st {
			SliceType::I => {
				let cond = |m: Option<&MbInfo>| {
					m.is_some_and(|m| !matches!(m.kind, MbKind::I4x4 | MbKind::I8x8)) as usize
				};
				let inc = cond(self.nb_mb(-1, 0)) + cond(self.nb_mb(0, -1));
				self.cabac_intra_type(ctx::MB_TYPE_I, inc, true)
			}
			SliceType::P => {
				let c = self.cabac().ok_or(BadMb("cabac state"))?;
				if c.decision(ctx::MB_TYPE_P) {
					return self.cabac_intra_type(ctx::MB_TYPE_P_INTRA, 0, false);
				}
				let t = if !c.decision(ctx::MB_TYPE_P + 1) {
					if c.decision(ctx::MB_TYPE_P + 2) { 3 } else { 0 }
				} else if c.decision(ctx::MB_TYPE_P + 3) {
					1
				} else {
					2
				};
				let shape = [Shape::S16x16, Shape::S16x8, Shape::S8x16, Shape::S8x8][t];
				Ok(MbType::Inter { shape, pred: [Pred::L0; 2], ref0: false })
			}
			SliceType::B => {
				let cond = |m: Option<&MbInfo>| m.is_some_and(|m| !m.direct16) as usize;
				let inc = cond(self.nb_mb(-1, 0)) + cond(self.nb_mb(0, -1));
				let c = self.cabac().ok_or(BadMb("cabac state"))?;
				let b = ctx::MB_TYPE_B;
				if !c.decision(b + inc) {
					return Ok(MbType::Direct);
				}
				let t = if !c.decision(b + 3) {
					1 + u32::from(c.decision(b + 5))
				} else {
					let mut bits = u32::from(c.decision(b + 4)) << 3;
					bits |= u32::from(c.decision(b + 5)) << 2;
					bits |= u32::from(c.decision(b + 5)) << 1;
					bits |= u32::from(c.decision(b + 5));
					match bits {
						0..=7 => bits + 3,
						13 => return self.cabac_intra_type(ctx::MB_TYPE_B_INTRA, 0, false),
						14 => 11,
						15 => 22,
						_ => ((bits << 1) | u32::from(c.decision(b + 5))) - 4,
					}
				};
				if t == 22 {
					return Ok(MbType::Inter { shape: Shape::S8x8, pred: [Pred::Bi; 2], ref0: false });
				}
				let (shape, a, bb) = B_TYPES[t as usize - 1];
				Ok(MbType::Inter { shape, pred: [a, bb], ref0: false })
			}
		}
	}

	/// Intra mb_type bins (prefix context `base`, Table 9-36).
	fn cabac_intra_type(&mut self, base: usize, inc: usize, intra_slice: bool) -> MbResult<MbType> {
		let c = self.cabac().ok_or(BadMb("cabac state"))?;
		if !c.decision(base + inc) {
			return Ok(MbType::I4x4);
		}
		if c.terminate() {
			return Ok(MbType::Pcm);
		}
		let s = if intra_slice { base + 2 } else { base };
		let i = usize::from(intra_slice);
		let luma = c.decision(s + 1);
		let mut chroma = 0u8;
		if c.decision(s + 2) {
			chroma = 1 + u8::from(c.decision(s + 2 + i));
		}
		let pred = (u8::from(c.decision(s + 3 + i)) << 1) | u8::from(c.decision(s + 3 + 2 * i));
		Ok(MbType::I16x16 { pred, cbp: (if luma { 15 } else { 0 }) | (chroma << 4) })
	}

	fn read_transform_8x8(&mut self) -> bool {
		match &mut self.entropy {
			Entropy::Cavlc(bs) => bs.bit(),
			Entropy::Cabac(_) => {
				let cond = |m: Option<&MbInfo>| m.is_some_and(|m| m.t8x8) as usize;
				let inc = cond(self.nb_mb(-1, 0)) + cond(self.nb_mb(0, -1));
				self.cabac().is_some_and(|c| c.decision(ctx::TRANSFORM_8X8 + inc))
			}
		}
	}

	/// prev_intra_pred_mode_flag / rem_intra_pred_mode; returns None when
	/// the predicted mode is used.
	fn read_intra_mode(&mut self) -> Option<u8> {
		match &mut self.entropy {
			Entropy::Cavlc(bs) => {
				if bs.bit() {
					None
				} else {
					Some(bs.bits(3) as u8)
				}
			}
			Entropy::Cabac(c) => {
				if c.decision(ctx::PREV_INTRA) {
					None
				} else {
					let mut m = 0u8;
					for i in 0..3 {
						m |= u8::from(c.decision(ctx::REM_INTRA)) << i;
					}
					Some(m)
				}
			}
		}
	}

	fn read_chroma_pred(&mut self) -> MbResult<u8> {
		match &mut self.entropy {
			Entropy::Cavlc(bs) => {
				let v = bs.ue();
				if v > 3 {
					return Err(BadMb("invalid intra_chroma_pred_mode"));
				}
				Ok(v as u8)
			}
			Entropy::Cabac(_) => {
				let cond = |m: Option<&MbInfo>| {
					m.is_some_and(|m| m.is_intra() && m.kind != MbKind::Pcm && m.chroma_pred != 0) as usize
				};
				let inc = cond(self.nb_mb(-1, 0)) + cond(self.nb_mb(0, -1));
				let c = self.cabac().ok_or(BadMb("cabac state"))?;
				Ok(c.unary(ctx::CHROMA_PRED + inc, ctx::CHROMA_PRED + 3, 3) as u8)
			}
		}
	}

	fn read_sub_type(&mut self) -> MbResult<SubType> {
		let st = self.h.slice_type;
		let raw = match &mut self.entropy {
			Entropy::Cavlc(bs) => bs.ue(),
			Entropy::Cabac(c) => {
				if st == SliceType::P {
					if c.decision(ctx::SUB_MB_P) {
						0
					} else if !c.decision(ctx::SUB_MB_P + 1) {
						1
					} else if c.decision(ctx::SUB_MB_P + 2) {
						2
					} else {
						3
					}
				} else {
					let b = ctx::SUB_MB_B;
					if !c.decision(b) {
						0
					} else if !c.decision(b + 1) {
						1 + u32::from(c.decision(b + 3))
					} else {
						let mut t = 3;
						if c.decision(b + 2) {
							if c.decision(b + 3) {
								return b_sub_type(11 + u32::from(c.decision(b + 3)))
									.ok_or(BadMb("invalid sub_mb_type"));
							}
							t += 4;
						}
						t += 2 * u32::from(c.decision(b + 3));
						t += u32::from(c.decision(b + 3));
						t
					}
				}
			}
		};
		let t = if st == SliceType::P { p_sub_type(raw) } else { b_sub_type(raw) };
		t.ok_or(BadMb("invalid sub_mb_type"))
	}

	/// ref_idx_lX for the partition whose top-left 4x4 block is (x4, y4).
	fn read_ref_idx(&mut self, cur: &MbInfo, list: usize, x4: usize, y4: usize) -> MbResult<i8> {
		let count = self.h.num_ref_idx[list];
		if count <= 1 {
			return Ok(0);
		}
		let v = match &mut self.entropy {
			Entropy::Cavlc(bs) => {
				if count == 2 {
					u32::from(!bs.bit())
				} else {
					bs.ue()
				}
			}
			Entropy::Cabac(_) => {
				let cond = |n: Option<(&MbInfo, usize)>| -> usize {
					let Some((m, idx)) = n else { return 0 };
					let b8 = (idx / 8) * 2 + (idx % 4) / 2;
					if m.is_intra() || m.direct8 & (1 << b8) != 0 {
						return 0;
					}
					usize::from(m.ref_idx[list][b8] > 0)
				};
				let inc = cond(self.nb4(cur, x4 as i32 - 1, y4 as i32))
					+ 2 * cond(self.nb4(cur, x4 as i32, y4 as i32 - 1));
				let c = self.cabac().ok_or(BadMb("cabac state"))?;
				c.unary3(ctx::REF_IDX + inc, ctx::REF_IDX + 4, ctx::REF_IDX + 5, 32)
			}
		};
		if v as usize >= count {
			return Err(BadMb("ref_idx out of range"));
		}
		Ok(v as i8)
	}

	/// One mvd_lX component for the partition at (x4, y4).
	fn read_mvd(
		&mut self,
		cur: &MbInfo,
		list: usize,
		x4: usize,
		y4: usize,
		comp: usize,
	) -> MbResult<i32> {
		match &mut self.entropy {
			Entropy::Cavlc(bs) => Ok(bs.se()),
			Entropy::Cabac(_) => {
				let get =
					|n: Option<(&MbInfo, usize)>| n.map_or(0u32, |(m, i)| u32::from(m.mvd[list][i][comp]));
				let sum = get(self.nb4(cur, x4 as i32 - 1, y4 as i32))
					+ get(self.nb4(cur, x4 as i32, y4 as i32 - 1));
				let inc = if sum < 3 {
					0
				} else if sum > 32 {
					2
				} else {
					1
				};
				let base = if comp == 0 { ctx::MVD_X } else { ctx::MVD_Y };
				let c = self.cabac().ok_or(BadMb("cabac state"))?;
				if !c.decision(base + inc) {
					return Ok(0);
				}
				let mut v = 1u32;
				let incs = [3usize, 4, 5, 6, 6, 6, 6, 6];
				while v < 9 && c.decision(base + incs[(v - 1) as usize]) {
					v += 1;
				}
				if v >= 9 {
					v = v.saturating_add(c.exp_golomb_bypass(3));
				}
				let v = v.min(1 << 20) as i32;
				Ok(if c.bypass() { -v } else { v })
			}
		}
	}

	fn read_cbp(&mut self, cur: &MbInfo, intra: bool) -> MbResult<u8> {
		match &mut self.entropy {
			Entropy::Cavlc(bs) => {
				let code = bs.ue() as usize;
				let table = if intra { &GOLOMB_TO_INTRA_CBP } else { &GOLOMB_TO_INTER_CBP };
				table.get(code).copied().ok_or(BadMb("invalid coded_block_pattern"))
			}
			Entropy::Cabac(_) => {
				let mut cbp = 0u8;
				for b8 in 0..4usize {
					let (x4, y4) = ((b8 % 2) as i32 * 2, (b8 / 2) as i32 * 2);
					let cond = |n: Option<(&MbInfo, usize)>, cbp_cur: u8, same: bool| -> usize {
						let Some((m, idx)) = n else { return 0 };
						if m.kind == MbKind::Pcm {
							return 0;
						}
						let nb8 = (idx / 8) * 2 + (idx % 4) / 2;
						let bits = if same { cbp_cur } else { m.cbp };
						if !m.is_skip() && bits & (1 << nb8) != 0 { 0 } else { 1 }
					};
					let a = self.nb4(cur, x4 - 1, y4);
					let a_same = x4 > 0;
					let b = self.nb4(cur, x4, y4 - 1);
					let b_same = y4 > 0;
					let inc = cond(a, cbp, a_same) + 2 * cond(b, cbp, b_same);
					let c = self.cabac().ok_or(BadMb("cabac state"))?;
					if c.decision(ctx::CBP_LUMA + inc) {
						cbp |= 1 << b8;
					}
				}
				let cond = |m: Option<&MbInfo>, need: u8| -> usize {
					m.is_some_and(|m| m.kind == MbKind::Pcm || (!m.is_skip() && (m.cbp >> 4) >= need))
						as usize
				};
				let (a, b) = (self.nb_mb(-1, 0), self.nb_mb(0, -1));
				let inc0 = cond(a, 1) + 2 * cond(b, 1);
				let inc1 = cond(a, 2) + 2 * cond(b, 2) + 4;
				let c = self.cabac().ok_or(BadMb("cabac state"))?;
				if c.decision(ctx::CBP_CHROMA + inc0) {
					cbp |= if c.decision(ctx::CBP_CHROMA + inc1) { 0x20 } else { 0x10 };
				}
				Ok(cbp)
			}
		}
	}

	fn read_qp_delta(&mut self) -> MbResult<i32> {
		let d = match &mut self.entropy {
			Entropy::Cavlc(bs) => bs.se(),
			Entropy::Cabac(c) => {
				let inc = usize::from(self.prev_qp_delta_nonzero);
				let k = c.unary3(ctx::QP_DELTA + inc, ctx::QP_DELTA + 2, ctx::QP_DELTA + 3, 104) as i32;
				let mag = (k + 1) / 2;
				if k % 2 == 1 { mag } else { -mag }
			}
		};
		if !(-26..=25).contains(&d) {
			return Err(BadMb("mb_qp_delta out of range"));
		}
		Ok(d)
	}

	// -----------------------------------------------------------------------
	// Macroblock layer

	/// Parses one non-skipped macroblock into `cur` and `self.res`.
	pub fn parse_macroblock(&mut self, cur: &mut MbInfo) -> MbResult<MbType> {
		let mb_type = self.read_mb_type()?;
		let pps_8x8 = self.pps.transform_8x8_mode;
		match mb_type {
			MbType::Pcm => {
				cur.kind = MbKind::Pcm;
				self.read_pcm()?;
				cur.cbp = 0x2F;
				cur.nnz = [16; 24];
				cur.dc_cbf = 7;
				cur.qp = 0;
				cur.qpc = [
					super::consts::chroma_qp(0, self.pps.chroma_qp_offset[0]),
					super::consts::chroma_qp(0, self.pps.chroma_qp_offset[1]),
				];
				cur.ipred = [2; 16];
				cur.ref_idx = [[-1; 4]; 2];
				self.prev_qp_delta_nonzero = false;
				return Ok(mb_type);
			}
			MbType::I4x4 => {
				cur.kind = MbKind::I4x4;
				if pps_8x8 && self.read_transform_8x8() {
					cur.t8x8 = true;
					cur.kind = MbKind::I8x8;
				}
				self.parse_intra_modes(cur)?;
				cur.chroma_pred = self.read_chroma_pred()?;
			}
			MbType::I16x16 { cbp, .. } => {
				cur.kind = MbKind::I16x16;
				cur.cbp = cbp;
				cur.ipred = [2; 16];
				cur.chroma_pred = self.read_chroma_pred()?;
			}
			MbType::Direct | MbType::Inter { .. } => {
				cur.kind = MbKind::Inter;
				cur.ipred = [2; 16];
				self.parse_inter(cur, mb_type)?;
			}
		}
		if cur.is_intra() {
			cur.ref_idx = [[-1; 4]; 2];
		}

		if !matches!(mb_type, MbType::I16x16 { .. }) {
			cur.cbp = self.read_cbp(cur, cur.is_intra())?;
			if cur.cbp & 15 != 0 && pps_8x8 && !cur.is_intra() && self.inter_allows_8x8(cur, mb_type) {
				cur.t8x8 = self.read_transform_8x8();
			}
		}
		let has_residual = cur.cbp != 0 || cur.kind == MbKind::I16x16;
		if has_residual {
			let d = self.read_qp_delta()?;
			self.prev_qp_delta_nonzero = d != 0;
			self.qp = (self.qp + d + 52) % 52;
		} else {
			self.prev_qp_delta_nonzero = false;
		}
		cur.qp = self.qp as u8;
		cur.qpc = [
			super::consts::chroma_qp(self.qp, self.pps.chroma_qp_offset[0]),
			super::consts::chroma_qp(self.qp, self.pps.chroma_qp_offset[1]),
		];
		if has_residual {
			self.parse_residual(cur)?;
		}
		Ok(mb_type)
	}

	/// transform_size_8x8_flag is only present for inter macroblocks whose
	/// partitions are all at least 8x8 (7.3.5).
	fn inter_allows_8x8(&self, cur: &MbInfo, mb_type: MbType) -> bool {
		match mb_type {
			MbType::Direct => self.sps.direct_8x8_inference,
			MbType::Inter { shape: Shape::S8x8, .. } => {
				self.sub_types.iter().all(|s| {
					if s.pred == Pred::Direct { self.sps.direct_8x8_inference } else { s.parts == 1 }
				}) && cur.kind == MbKind::Inter
			}
			_ => true,
		}
	}

	fn read_pcm(&mut self) -> MbResult<()> {
		let res = self.res.get_or_insert_with(Box::default);
		match &mut self.entropy {
			Entropy::Cavlc(bs) => {
				bs.align();
				for s in res.pcm.iter_mut() {
					*s = bs.bits(8) as u8;
				}
				if bs.overrun() {
					return Err(BadMb("truncated pcm samples"));
				}
			}
			Entropy::Cabac(c) => {
				let bs = c.reader();
				bs.align();
				for s in res.pcm.iter_mut() {
					*s = bs.bits(8) as u8;
				}
				if bs.overrun() {
					return Err(BadMb("truncated pcm samples"));
				}
				c.restart();
			}
		}
		Ok(())
	}

	/// Intra 4x4 / 8x8 prediction modes (8.3.1.1, 8.3.2.1).
	fn parse_intra_modes(&mut self, cur: &mut MbInfo) -> MbResult<()> {
		let constrained = self.pps.constrained_intra_pred;
		let step = if cur.t8x8 { 4 } else { 1 };
		for blk in (0..16).step_by(step) {
			let (x4, y4) = (i32::from(BLK_XY[blk].0), i32::from(BLK_XY[blk].1));
			let mode_of = |n: Option<(&MbInfo, usize)>| -> Option<u8> {
				let (m, idx) = n?;
				if m.is_intra() {
					Some(if matches!(m.kind, MbKind::I4x4 | MbKind::I8x8) { m.ipred[idx] } else { 2 })
				} else if constrained {
					None
				} else {
					Some(2)
				}
			};
			let a = mode_of(self.nb4(cur, x4 - 1, y4));
			let b = mode_of(self.nb4(cur, x4, y4 - 1));
			let pred = match (a, b) {
				(Some(a), Some(b)) => a.min(b),
				_ => 2,
			};
			let mode = match self.read_intra_mode() {
				None => pred,
				Some(rem) if rem < pred => rem,
				Some(rem) => rem + 1,
			};
			if cur.t8x8 {
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					cur.ipred[((y4 + dy) * 4 + x4 + dx) as usize] = mode;
				}
			} else {
				cur.ipred[(y4 * 4 + x4) as usize] = mode;
			}
		}
		Ok(())
	}

	// -----------------------------------------------------------------------
	// Residual

	/// CABAC coded_block_flag ctxIdxInc for a luma 4x4 block (cat 1/2) at
	/// raster position (x4, y4).
	fn cbf_luma_inc(&self, cur: &MbInfo, x4: i32, y4: i32) -> usize {
		let intra = cur.is_intra();
		let cond = |n: Option<(&MbInfo, usize)>| -> usize {
			match n {
				None => usize::from(intra),
				Some((m, idx)) => {
					if m.kind == MbKind::Pcm {
						1
					} else if m.is_skip() {
						0
					} else {
						let b8 = (idx / 8) * 2 + (idx % 4) / 2;
						usize::from(m.cbp & (1 << b8) != 0 && m.nnz[idx] != 0)
					}
				}
			}
		};
		cond(self.nb4(cur, x4 - 1, y4)) + 2 * cond(self.nb4(cur, x4, y4 - 1))
	}

	fn cbf_dc_inc(&self, cur: &MbInfo, comp: usize) -> usize {
		let intra = cur.is_intra();
		let cond = |m: Option<&MbInfo>| -> usize {
			match m {
				None => usize::from(intra),
				Some(m) if m.kind == MbKind::Pcm => 1,
				Some(m) if m.is_skip() => 0,
				Some(m) => {
					if comp == 0 {
						usize::from(m.kind == MbKind::I16x16 && m.dc_cbf & 1 != 0)
					} else {
						usize::from(m.cbp >> 4 != 0 && m.dc_cbf & (1 << comp) != 0)
					}
				}
			}
		};
		cond(self.nb_mb(-1, 0)) + 2 * cond(self.nb_mb(0, -1))
	}

	fn cbf_chroma_ac_inc(&self, cur: &MbInfo, comp: usize, x2: i32, y2: i32) -> usize {
		let intra = cur.is_intra();
		let cond = |n: Option<(&MbInfo, usize)>| -> usize {
			match n {
				None => usize::from(intra),
				Some((m, _)) if m.kind == MbKind::Pcm => 1,
				Some((m, _)) if m.is_skip() => 0,
				Some((m, idx)) => usize::from(m.cbp >> 4 == 2 && m.nnz[idx] != 0),
			}
		};
		cond(self.nb_chroma(cur, comp, x2 - 1, y2)) + 2 * cond(self.nb_chroma(cur, comp, x2, y2 - 1))
	}

	/// CAVLC nC for a luma block (9.2.1).
	fn nc_luma(&self, cur: &MbInfo, x4: i32, y4: i32) -> i32 {
		let get = |n: Option<(&MbInfo, usize)>| n.map(|(m, i)| i32::from(m.nnz[i]));
		combine_nc(get(self.nb4(cur, x4 - 1, y4)), get(self.nb4(cur, x4, y4 - 1)))
	}

	fn nc_chroma(&self, cur: &MbInfo, comp: usize, x2: i32, y2: i32) -> i32 {
		let get = |n: Option<(&MbInfo, usize)>| n.map(|(m, i)| i32::from(m.nnz[i]));
		combine_nc(
			get(self.nb_chroma(cur, comp, x2 - 1, y2)),
			get(self.nb_chroma(cur, comp, x2, y2 - 1)),
		)
	}

	/// Decodes one residual block with whichever entropy coder is active.
	#[allow(clippy::too_many_arguments)]
	fn block(
		&mut self,
		cat: BlockCat,
		cbf_inc: Option<usize>,
		nc: i32,
		scan: &[u8],
		start: usize,
		max: usize,
		out: &mut [i32],
	) -> MbResult<u8> {
		match &mut self.entropy {
			Entropy::Cabac(c) => Ok(c.residual(cat, cbf_inc, scan, start, max, out)),
			Entropy::Cavlc(bs) => {
				cavlc::residual(bs, nc, scan, start, max, out).map_err(|_| BadMb("invalid cavlc residual"))
			}
		}
	}

	fn parse_residual(&mut self, cur: &mut MbInfo) -> MbResult<()> {
		let cabac = matches!(self.entropy, Entropy::Cabac(_));
		let mut res = self.res.take().unwrap_or_default();
		let result = self.parse_residual_into(cur, &mut res, cabac);
		self.res = Some(res);
		result
	}

	fn parse_residual_into(
		&mut self,
		cur: &mut MbInfo,
		res: &mut Residual,
		cabac: bool,
	) -> MbResult<()> {
		let i16 = cur.kind == MbKind::I16x16;
		if i16 {
			let inc = self.cbf_dc_inc(cur, 0);
			let nc = self.nc_luma(cur, 0, 0);
			let n = self.block(BlockCat::LumaDc, Some(inc), nc, &ZIGZAG_4X4, 0, 16, &mut res.dc)?;
			if n != 0 {
				cur.dc_cbf |= 1;
			}
		}
		for b8 in 0..4usize {
			let coded = cur.cbp & (1 << b8) != 0;
			if cabac && cur.t8x8 {
				let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
				let n = if coded {
					self.block(BlockCat::Luma8x8, None, 0, &ZIGZAG_8X8, 0, 64, &mut res.luma8[b8])?
				} else {
					0
				};
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					cur.nnz[(y4 + dy) * 4 + x4 + dx] = n;
				}
				continue;
			}
			for i in 0..4 {
				let blk = b8 * 4 + i;
				let (x4, y4) = (usize::from(BLK_XY[blk].0), usize::from(BLK_XY[blk].1));
				let raster = y4 * 4 + x4;
				if !coded {
					cur.nnz[raster] = 0;
					continue;
				}
				let nc = if cabac { 0 } else { self.nc_luma(cur, x4 as i32, y4 as i32) };
				let inc = if cabac { Some(self.cbf_luma_inc(cur, x4 as i32, y4 as i32)) } else { None };
				let n = if i16 {
					self.block(BlockCat::LumaAc, inc, nc, &ZIGZAG_4X4, 1, 15, &mut res.luma[raster])?
				} else if cur.t8x8 {
					// CAVLC 8x8: four interleaved 4x4 scans (7.3.5.3.2).
					let mut tmp = [0i32; 16];
					let identity: [u8; 16] = std::array::from_fn(|k| k as u8);
					let n = self.block(BlockCat::Luma4x4, None, nc, &identity, 0, 16, &mut tmp)?;
					for (k, &v) in tmp.iter().enumerate() {
						if v != 0 {
							res.luma8[b8][usize::from(ZIGZAG_8X8[4 * k + i])] = v;
						}
					}
					n
				} else {
					self.block(BlockCat::Luma4x4, inc, nc, &ZIGZAG_4X4, 0, 16, &mut res.luma[raster])?
				};
				cur.nnz[raster] = n;
			}
		}
		if cur.cbp >> 4 != 0 {
			for comp in 0..2 {
				let inc = self.cbf_dc_inc(cur, comp + 1);
				let scan: [u8; 4] = [0, 1, 2, 3];
				let n =
					self.block(BlockCat::ChromaDc, Some(inc), -1, &scan, 0, 4, &mut res.chroma_dc[comp])?;
				if n != 0 {
					cur.dc_cbf |= 2 << comp;
				}
			}
		}
		for comp in 0..2 {
			for blk in 0..4usize {
				let idx = 16 + 4 * comp + blk;
				if cur.cbp >> 4 != 2 {
					cur.nnz[idx] = 0;
					continue;
				}
				let (x2, y2) = ((blk % 2) as i32, (blk / 2) as i32);
				let nc = if cabac { 0 } else { self.nc_chroma(cur, comp, x2, y2) };
				let inc = if cabac { Some(self.cbf_chroma_ac_inc(cur, comp, x2, y2)) } else { None };
				let n = self.block(
					BlockCat::ChromaAc,
					inc,
					nc,
					&ZIGZAG_4X4,
					1,
					15,
					&mut res.chroma_ac[comp][blk],
				)?;
				cur.nnz[idx] = n;
			}
		}
		Ok(())
	}

	// -----------------------------------------------------------------------
	// Inter prediction syntax

	fn parse_inter(&mut self, cur: &mut MbInfo, mb_type: MbType) -> MbResult<()> {
		cur.ref_idx = [[-1; 4]; 2];
		match mb_type {
			MbType::Direct => {
				cur.direct16 = true;
				cur.direct8 = 0xF;
				self.predict_direct(cur, 0xF)?;
				Ok(())
			}
			MbType::Inter { shape: Shape::S8x8, ref0, .. } => self.parse_sub_mb(cur, ref0),
			MbType::Inter { shape, pred, .. } => {
				let parts: &[(usize, usize, usize, usize)] = match shape {
					Shape::S16x16 => &[(0, 0, 4, 4)],
					Shape::S16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
					_ => &[(0, 0, 2, 4), (2, 0, 2, 4)],
				};
				for list in 0..2 {
					for (p, &(x4, y4, w4, h4)) in parts.iter().enumerate() {
						if pred[p].uses(list) {
							let r = self.read_ref_idx(cur, list, x4, y4)?;
							set_ref(cur, list, x4, y4, w4, h4, r);
						}
					}
				}
				for list in 0..2 {
					for (p, &(x4, y4, w4, h4)) in parts.iter().enumerate() {
						if !pred[p].uses(list) {
							continue;
						}
						let mx = self.read_mvd(cur, list, x4, y4, 0)?;
						let my = self.read_mvd(cur, list, x4, y4, 1)?;
						let r = cur.ref_idx[list][(y4 / 2) * 2 + x4 / 2];
						let pmv = self.predict_mv(cur, list, x4, y4, w4, h4, r, shape, p);
						let mv = [pmv[0].wrapping_add(mx as i16), pmv[1].wrapping_add(my as i16)];
						set_mv(cur, list, x4, y4, w4, h4, mv, [mx, my]);
					}
				}
				Ok(())
			}
			_ => Ok(()),
		}
	}

	fn parse_sub_mb(&mut self, cur: &mut MbInfo, ref0: bool) -> MbResult<()> {
		let mut subs = [SubType { pred: Pred::L0, parts: 1, w: 2, h: 2 }; 4];
		for s in &mut subs {
			*s = self.read_sub_type()?;
		}
		self.sub_types = subs;
		let direct_mask =
			subs.iter().enumerate().fold(0u8, |m, (i, s)| m | (u8::from(s.pred == Pred::Direct) << i));
		cur.direct8 = direct_mask;
		if direct_mask != 0 {
			self.predict_direct(cur, direct_mask)?;
		}
		for list in 0..2 {
			for (b8, s) in subs.iter().enumerate() {
				if s.pred == Pred::Direct || !s.pred.uses(list) {
					continue;
				}
				let (x4, y4) = ((b8 % 2) * 2, (b8 / 2) * 2);
				let r = if ref0 { 0 } else { self.read_ref_idx(cur, list, x4, y4)? };
				set_ref(cur, list, x4, y4, 2, 2, r);
			}
		}
		for list in 0..2 {
			for (b8, s) in subs.iter().enumerate() {
				if s.pred == Pred::Direct || !s.pred.uses(list) {
					continue;
				}
				let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
				let r = cur.ref_idx[list][b8];
				for part in 0..s.parts {
					let (x4, y4) = match (s.w, s.h) {
						(2, 2) => (bx, by),
						(2, 1) => (bx, by + part),
						(1, 2) => (bx + part, by),
						_ => (bx + part % 2, by + part / 2),
					};
					let mx = self.read_mvd(cur, list, x4, y4, 0)?;
					let my = self.read_mvd(cur, list, x4, y4, 1)?;
					let pmv = self.predict_mv(cur, list, x4, y4, s.w, s.h, r, Shape::S8x8, 0);
					let mv = [pmv[0].wrapping_add(mx as i16), pmv[1].wrapping_add(my as i16)];
					set_mv(cur, list, x4, y4, s.w, s.h, mv, [mx, my]);
				}
			}
		}
		Ok(())
	}
}

fn combine_nc(a: Option<i32>, b: Option<i32>) -> i32 {
	match (a, b) {
		(Some(a), Some(b)) => (a + b + 1) >> 1,
		(Some(a), None) => a,
		(None, Some(b)) => b,
		(None, None) => 0,
	}
}

/// Sets the reference index of the 8x8 blocks covered by a partition.
pub fn set_ref(cur: &mut MbInfo, list: usize, x4: usize, y4: usize, w4: usize, h4: usize, r: i8) {
	for y in (y4..y4 + h4).step_by(2) {
		for x in (x4..x4 + w4).step_by(2) {
			cur.ref_idx[list][(y / 2) * 2 + x / 2] = r;
		}
	}
}

/// Stores a partition's motion vector and |mvd| in its 4x4 blocks.
#[allow(clippy::too_many_arguments)]
pub fn set_mv(
	cur: &mut MbInfo,
	list: usize,
	x4: usize,
	y4: usize,
	w4: usize,
	h4: usize,
	mv: [i16; 2],
	mvd: [i32; 2],
) {
	let sat = |v: i32| v.unsigned_abs().min(255) as u8;
	for y in y4..y4 + h4 {
		for x in x4..x4 + w4 {
			cur.mv[list][y * 4 + x] = mv;
			cur.mvd[list][y * 4 + x] = [sat(mvd[0]), sat(mvd[1])];
		}
	}
}

/// Decoding-order index of the raster 4x4 block (x4, y4).
pub fn blk_index(x4: usize, y4: usize) -> usize {
	usize::from(RASTER_TO_BLK[y4 * 4 + x4])
}
