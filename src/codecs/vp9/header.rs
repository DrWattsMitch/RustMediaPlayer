//! VP9 frame headers (spec §6.2, §6.3; libvpx `read_uncompressed_header`,
//! `read_compressed_header`) and superframe indexes (spec Annex B).

use super::probs::{self, FrameContext};
use super::range::BoolDecoder;
use super::{ALTREF_FRAME, GOLDEN_FRAME, LAST_FRAME};
use crate::codecs::bitstream::BitReader;

/// Largest frame accepted: 16384 pixels per side, 8K-class area. Bigger
/// headers are treated as damage rather than allocated.
pub const MAX_DIMENSION: u32 = 16_384;
pub const MAX_AREA: u64 = 8192 * 4352;

/// Segment features (`SEG_LVL_*`).
pub const SEG_ALT_Q: usize = 0;
pub const SEG_ALT_LF: usize = 1;
pub const SEG_REF_FRAME: usize = 2;
pub const SEG_SKIP: usize = 3;

/// Header state that persists from frame to frame.
#[derive(Debug, Clone)]
pub struct Persistent {
	pub ref_deltas: [i8; 4],
	pub mode_deltas: [i8; 2],
	pub seg: Segmentation,
	pub contexts: [FrameContext; 4],
}

impl Default for Persistent {
	fn default() -> Self {
		Self {
			ref_deltas: [1, 0, -1, -1],
			mode_deltas: [0, 0],
			seg: Segmentation::default(),
			contexts: std::array::from_fn(|_| FrameContext::default()),
		}
	}
}

#[derive(Debug, Clone, Default)]
pub struct Segmentation {
	pub enabled: bool,
	pub update_map: bool,
	pub temporal_update: bool,
	pub abs_delta: bool,
	pub tree_probs: [u8; 7],
	pub pred_probs: [u8; 3],
	pub feature_enabled: [[bool; 4]; 8],
	pub feature_data: [[i16; 4]; 8],
}

impl Segmentation {
	pub fn active(&self, segment: u8, feature: usize) -> bool {
		self.enabled && self.feature_enabled[usize::from(segment)][feature]
	}

	pub fn data(&self, segment: u8, feature: usize) -> i16 {
		self.feature_data[usize::from(segment)][feature]
	}

	fn clear_features(&mut self) {
		self.feature_enabled = [[false; 4]; 8];
		self.feature_data = [[0; 4]; 8];
	}
}

/// Colour configuration (spec §6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorConfig {
	pub bit_depth: u8,
	pub color_space: u8,
	pub full_range: bool,
	pub ss_x: bool,
	pub ss_y: bool,
}

/// What the decoder needs to know about the reference slots while parsing.
pub trait RefInfo {
	/// Size of the frame in slot `slot` (0..8), if it holds one.
	fn ref_size(&self, slot: usize) -> Option<(u32, u32)>;
}

/// The uncompressed header (plus derived values).
#[derive(Debug, Clone, Default)]
pub struct FrameHeader {
	pub profile: u8,
	/// `show_existing_frame`: the slot to show; nothing else is set.
	pub show_existing: Option<usize>,
	pub key_frame: bool,
	pub show_frame: bool,
	pub error_resilient: bool,
	pub intra_only: bool,
	pub reset_context: u8,
	pub color: ColorConfig,
	pub refresh: u8,
	/// Slot of LAST, GOLDEN and ALTREF.
	pub ref_slot: [usize; 3],
	/// Sign bias of each reference frame (index 1..=3).
	pub sign_bias: [bool; 4],
	pub width: u32,
	pub height: u32,
	pub render_size: (u32, u32),
	pub allow_hp: bool,
	/// 0..=3 (`INTERP_FILTER`) or 4 = switchable.
	pub interp_filter: u8,
	pub refresh_context: bool,
	pub parallel_mode: bool,
	pub context_idx: usize,
	pub filter_level: u8,
	pub sharpness: u8,
	pub lf_delta_enabled: bool,
	pub base_q: u8,
	pub y_dc_delta: i8,
	pub uv_dc_delta: i8,
	pub uv_ac_delta: i8,
	pub lossless: bool,
	pub log2_tile_cols: u32,
	pub log2_tile_rows: u32,
	pub compressed_size: usize,
	/// Bytes of uncompressed header.
	pub header_size: usize,
}

impl FrameHeader {
	pub fn intra_only_frame(&self) -> bool {
		self.key_frame || self.intra_only
	}
}

fn invalid(what: &'static str) -> crate::message::Message {
	crate::message::Message::Codec(what)
}

/// `vpx_rb_read_signed_literal`: magnitude, then sign.
fn signed(b: &mut BitReader, n: u32) -> i32 {
	let v = b.bits(n) as i32;
	if b.bit() { -v } else { v }
}

fn color_config(b: &mut BitReader, profile: u8) -> crate::Result<ColorConfig> {
	let bit_depth = if profile >= 2 { if b.bit() { 12 } else { 10 } } else { 8 };
	let color_space = b.bits(3) as u8;
	let mut c = ColorConfig { bit_depth, color_space, ..Default::default() };
	if color_space != 7 {
		c.full_range = b.bit();
		if profile == 1 || profile == 3 {
			c.ss_x = b.bit();
			c.ss_y = b.bit();
			if c.ss_x && c.ss_y {
				return Err(invalid("vp9: 4:2:0 is not allowed in profile 1 or 3"));
			}
			if b.bit() {
				return Err(invalid("vp9: reserved bit set"));
			}
		} else {
			c.ss_x = true;
			c.ss_y = true;
		}
	} else {
		// sRGB: 4:4:4, profiles 1 and 3 only.
		c.full_range = true;
		if profile == 1 || profile == 3 {
			if b.bit() {
				return Err(invalid("vp9: reserved bit set"));
			}
		} else {
			return Err(invalid("vp9: 4:4:4 is not allowed in profile 0 or 2"));
		}
	}
	Ok(c)
}

fn frame_size(b: &mut BitReader) -> (u32, u32) {
	let w = b.bits(16) + 1;
	let h = b.bits(16) + 1;
	(w, h)
}

fn sync_code(b: &mut BitReader) -> crate::Result<()> {
	if b.bits(24) != 0x49_8342 {
		return Err(invalid("vp9: invalid frame sync code"));
	}
	Ok(())
}

/// Superblock columns for a width (64-pixel units).
pub fn sb64_cols(mi_cols: u32) -> u32 {
	mi_cols.div_ceil(8)
}

/// Parses the uncompressed header, updating the persistent state the way
/// libvpx does (including `setup_past_independence`). `last_color` is the
/// colour configuration of the previous frame, for inter frames.
pub fn parse_uncompressed(
	data: &[u8],
	state: &mut Persistent,
	refs: &dyn RefInfo,
	last_color: ColorConfig,
) -> crate::Result<FrameHeader> {
	let mut b = BitReader::new(data);
	let mut h = FrameHeader::default();
	if b.bits(2) != 2 {
		return Err(invalid("vp9: invalid frame marker"));
	}
	let mut profile = b.bits(1) as u8;
	profile |= (b.bits(1) as u8) << 1;
	if profile > 2 {
		profile += b.bits(1) as u8;
	}
	if profile > 3 {
		return Err(invalid("vp9: unsupported profile"));
	}
	h.profile = profile;
	if b.bit() {
		h.show_existing = Some(b.bits(3) as usize);
		h.show_frame = true;
		h.header_size = b.pos().div_ceil(8);
		return Ok(h);
	}
	h.key_frame = !b.bit();
	h.show_frame = b.bit();
	h.error_resilient = b.bit();

	if h.key_frame {
		sync_code(&mut b)?;
		h.color = color_config(&mut b, profile)?;
		h.refresh = 0xFF;
		(h.width, h.height) = frame_size(&mut b);
		h.render_size = if b.bit() { frame_size(&mut b) } else { (h.width, h.height) };
	} else {
		h.intra_only = if h.show_frame { false } else { b.bit() };
		h.reset_context = if h.error_resilient { 0 } else { b.bits(2) as u8 };
		if h.intra_only {
			sync_code(&mut b)?;
			h.color = if profile > 0 {
				color_config(&mut b, profile)?
			} else {
				ColorConfig { bit_depth: 8, color_space: 1, full_range: false, ss_x: true, ss_y: true }
			};
			h.refresh = b.bits(8) as u8;
			(h.width, h.height) = frame_size(&mut b);
			h.render_size = if b.bit() { frame_size(&mut b) } else { (h.width, h.height) };
		} else {
			h.color = last_color;
			h.refresh = b.bits(8) as u8;
			for i in 0..3 {
				h.ref_slot[i] = b.bits(3) as usize;
				h.sign_bias[i + 1] = b.bit();
				if refs.ref_size(h.ref_slot[i]).is_none() {
					return Err(invalid("vp9: missing reference frame"));
				}
			}
			let mut size = None;
			for i in 0..3 {
				if b.bit() {
					size = refs.ref_size(h.ref_slot[i]);
					break;
				}
			}
			(h.width, h.height) = match size {
				Some(s) => s,
				None => frame_size(&mut b),
			};
			h.render_size = if b.bit() { frame_size(&mut b) } else { (h.width, h.height) };
			h.allow_hp = b.bit();
			h.interp_filter =
				if b.bit() { super::FILTER_SWITCHABLE } else { [1, 0, 2, 3][b.bits(2) as usize] };
		}
	}
	if h.width > MAX_DIMENSION
		|| h.height > MAX_DIMENSION
		|| u64::from(h.width) * u64::from(h.height) > MAX_AREA
	{
		return Err(invalid("vp9: frame size too large"));
	}

	if !h.error_resilient {
		h.refresh_context = b.bit();
		h.parallel_mode = b.bit();
	} else {
		h.parallel_mode = true;
	}
	h.context_idx = b.bits(2) as usize;

	if h.intra_only_frame() || h.error_resilient {
		// setup_past_independence.
		state.seg.clear_features();
		state.seg.abs_delta = false;
		state.ref_deltas = [1, 0, -1, -1];
		state.mode_deltas = [0, 0];
		if h.key_frame || h.error_resilient || h.reset_context == 3 {
			state.contexts = std::array::from_fn(|_| FrameContext::default());
		} else if h.reset_context == 2 {
			state.contexts[h.context_idx] = FrameContext::default();
		}
		h.sign_bias = [false; 4];
		h.context_idx = 0;
	}

	// Loop filter (§6.2.8).
	h.filter_level = b.bits(6) as u8;
	h.sharpness = b.bits(3) as u8;
	h.lf_delta_enabled = b.bit();
	if h.lf_delta_enabled && b.bit() {
		for d in state.ref_deltas.iter_mut() {
			if b.bit() {
				*d = signed(&mut b, 6) as i8;
			}
		}
		for d in state.mode_deltas.iter_mut() {
			if b.bit() {
				*d = signed(&mut b, 6) as i8;
			}
		}
	}

	// Quantization (§6.2.9).
	h.base_q = b.bits(8) as u8;
	let mut delta = || if b.bit() { signed(&mut b, 4) as i8 } else { 0 };
	h.y_dc_delta = delta();
	h.uv_dc_delta = delta();
	h.uv_ac_delta = delta();
	h.lossless = h.base_q == 0 && h.y_dc_delta == 0 && h.uv_dc_delta == 0 && h.uv_ac_delta == 0;

	// Segmentation (§6.2.11).
	let seg = &mut state.seg;
	seg.update_map = false;
	seg.enabled = b.bit();
	if seg.enabled {
		seg.update_map = b.bit();
		if seg.update_map {
			for p in seg.tree_probs.iter_mut() {
				*p = if b.bit() { b.bits(8) as u8 } else { 255 };
			}
			seg.temporal_update = b.bit();
			for p in seg.pred_probs.iter_mut() {
				*p = if seg.temporal_update && b.bit() { b.bits(8) as u8 } else { 255 };
			}
		}
		if b.bit() {
			seg.abs_delta = b.bit();
			seg.clear_features();
			const BITS: [u32; 4] = [8, 6, 2, 0];
			const MAX: [i16; 4] = [255, 63, 3, 0];
			for s in 0..8 {
				for f in 0..4 {
					if b.bit() {
						seg.feature_enabled[s][f] = true;
						let mut v = (b.bits(BITS[f]) as i16).min(MAX[f]);
						if f < 2 && b.bit() {
							v = -v;
						}
						seg.feature_data[s][f] = v;
					}
				}
			}
		}
	}

	// Tiles (§6.2.12).
	let mi_cols = h.width.div_ceil(8);
	let sb_cols = sb64_cols(mi_cols);
	let mut min_log2 = 0;
	while (64 << min_log2) < sb_cols {
		min_log2 += 1;
	}
	let mut max_log2 = 1;
	while (sb_cols >> max_log2) >= 4 {
		max_log2 += 1;
	}
	max_log2 -= 1;
	h.log2_tile_cols = min_log2;
	while h.log2_tile_cols < max_log2 && b.bit() {
		h.log2_tile_cols += 1;
	}
	h.log2_tile_rows = u32::from(b.bit());
	if h.log2_tile_rows == 1 {
		h.log2_tile_rows += u32::from(b.bit());
	}

	h.compressed_size = b.bits(16) as usize;
	if b.overrun() {
		return Err(invalid("vp9: truncated frame header"));
	}
	if h.compressed_size == 0 {
		return Err(invalid("vp9: invalid compressed header size"));
	}
	h.header_size = b.pos().div_ceil(8);
	if h.color.bit_depth != 8 {
		return Err(invalid("vp9: only 8-bit profiles (0 and 1) are supported"));
	}
	Ok(h)
}

/// Transform modes (`TX_MODE`).
pub const ONLY_4X4: u8 = 0;
pub const TX_MODE_SELECT: u8 = 4;

/// Reference modes (`REFERENCE_MODE`).
pub const SINGLE_REFERENCE: u8 = 0;
pub const COMPOUND_REFERENCE: u8 = 1;
pub const REFERENCE_MODE_SELECT: u8 = 2;

/// The compressed header: probability updates and frame-level modes.
#[derive(Debug, Clone, Default)]
pub struct CompressedHeader {
	pub tx_mode: u8,
	pub reference_mode: u8,
	pub comp_fixed_ref: i8,
	pub comp_var_ref: [i8; 2],
}

/// Largest transform allowed by a transform mode.
pub fn tx_mode_max(tx_mode: u8) -> usize {
	usize::from(tx_mode.min(3))
}

/// Parses the compressed header into `fc` (a copy of the frame's saved
/// context).
pub fn parse_compressed(
	data: &[u8],
	h: &FrameHeader,
	fc: &mut FrameContext,
) -> crate::Result<CompressedHeader> {
	let mut r = BoolDecoder::new(data).ok_or(invalid("vp9: compressed header marker bit set"))?;
	let tx_mode = if h.lossless {
		ONLY_4X4
	} else {
		let m = r.literal(2) as u8;
		if m == 3 { m + u8::from(r.bit()) } else { m }
	};
	let mut c = CompressedHeader { tx_mode, ..Default::default() };
	if c.tx_mode == TX_MODE_SELECT {
		for p in fc.tx8.iter_mut() {
			probs::diff_update(&mut r, &mut p[0]);
		}
		for row in fc.tx16.iter_mut() {
			for p in row.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
		for row in fc.tx32.iter_mut() {
			for p in row.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
	}
	probs::read_coef_updates(&mut r, fc, tx_mode_max(c.tx_mode));
	for p in fc.skip.iter_mut() {
		probs::diff_update(&mut r, p);
	}
	if !h.intra_only_frame() {
		for row in fc.inter_mode.iter_mut() {
			for p in row.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
		if h.interp_filter == super::FILTER_SWITCHABLE {
			for row in fc.interp_filter.iter_mut() {
				for p in row.iter_mut() {
					probs::diff_update(&mut r, p);
				}
			}
		}
		for p in fc.intra_inter.iter_mut() {
			probs::diff_update(&mut r, p);
		}
		// Compound prediction needs references on both sides in time.
		let bias = h.sign_bias;
		let compound_allowed = bias[2] != bias[1] || bias[3] != bias[1];
		c.reference_mode = if compound_allowed {
			if r.bit() {
				if r.bit() { REFERENCE_MODE_SELECT } else { COMPOUND_REFERENCE }
			} else {
				SINGLE_REFERENCE
			}
		} else {
			SINGLE_REFERENCE
		};
		if c.reference_mode != SINGLE_REFERENCE {
			if bias[LAST_FRAME as usize] == bias[GOLDEN_FRAME as usize] {
				c.comp_fixed_ref = ALTREF_FRAME;
				c.comp_var_ref = [LAST_FRAME, GOLDEN_FRAME];
			} else if bias[LAST_FRAME as usize] == bias[ALTREF_FRAME as usize] {
				c.comp_fixed_ref = GOLDEN_FRAME;
				c.comp_var_ref = [LAST_FRAME, ALTREF_FRAME];
			} else {
				c.comp_fixed_ref = LAST_FRAME;
				c.comp_var_ref = [GOLDEN_FRAME, ALTREF_FRAME];
			}
		}
		if c.reference_mode == REFERENCE_MODE_SELECT {
			for p in fc.comp_inter.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
		if c.reference_mode != COMPOUND_REFERENCE {
			for row in fc.single_ref.iter_mut() {
				probs::diff_update(&mut r, &mut row[0]);
				probs::diff_update(&mut r, &mut row[1]);
			}
		}
		if c.reference_mode != SINGLE_REFERENCE {
			for p in fc.comp_ref.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
		for row in fc.y_mode.iter_mut() {
			for p in row.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
		for row in fc.partition.iter_mut() {
			for p in row.iter_mut() {
				probs::diff_update(&mut r, p);
			}
		}
		probs::read_mv_updates(&mut r, fc, h.allow_hp);
	}
	if r.overrun() {
		return Err(invalid("vp9: corrupt compressed header"));
	}
	Ok(c)
}

/// Splits a packet into frames using the superframe index (Annex B); a
/// packet without an index is one frame.
pub fn split_superframe(data: &[u8]) -> Vec<&[u8]> {
	if let Some(&last) = data.last() {
		if last & 0xE0 == 0xC0 {
			let frames = usize::from(last & 7) + 1;
			let size_bytes = usize::from((last >> 3) & 3) + 1;
			let index_len = 2 + size_bytes * frames;
			if data.len() >= index_len && data[data.len() - index_len] == last {
				let index = &data[data.len() - index_len + 1..data.len() - 1];
				let mut out = Vec::with_capacity(frames);
				let mut at = 0usize;
				for f in 0..frames {
					let mut size = 0usize;
					for k in 0..size_bytes {
						size |= usize::from(index[f * size_bytes + k]) << (8 * k);
					}
					let end = at.saturating_add(size);
					if end > data.len() - index_len {
						break;
					}
					if size > 0 {
						out.push(&data[at..end]);
					}
					at = end;
				}
				return out;
			}
		}
	}
	vec![data]
}

#[cfg(test)]
mod tests {
	use super::split_superframe;

	#[test]
	fn superframe_index() {
		// Two frames of 3 and 2 bytes, one size byte each: marker 0b11000_001.
		let data = [1, 2, 3, 4, 5, 0xC1, 3, 2, 0xC1];
		assert_eq!(split_superframe(&data), vec![&[1, 2, 3][..], &[4, 5][..]]);
		assert_eq!(split_superframe(&[9, 9]), vec![&[9, 9][..]]);
	}
}
