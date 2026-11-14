//! Decoded pictures and their per-block side information.

use std::sync::Arc;

/// Padding around every plane so motion compensation can read outside the
/// picture without bounds checks (edges are replicated into it).
pub const PAD_LUMA: usize = 80;
pub const PAD_CHROMA: usize = 40;

/// One plane of samples (8-bit streams use the low 8 bits).
#[derive(Clone)]
pub struct Plane {
	pub data: Vec<u16>,
	pub stride: usize,
	pub width: usize,
	pub height: usize,
	pub pad: usize,
}

impl Plane {
	pub fn new(width: usize, height: usize, pad: usize, fill: u16) -> Self {
		let stride = width + 2 * pad;
		Self { data: vec![fill; stride * (height + 2 * pad)], stride, width, height, pad }
	}

	/// Index of sample (x, y) (may be negative within the padding).
	#[inline]
	pub fn idx(&self, x: isize, y: isize) -> usize {
		((y + self.pad as isize) as usize) * self.stride + (x + self.pad as isize) as usize
	}

	#[cfg(test)]
	pub fn get(&self, x: usize, y: usize) -> u16 {
		self.data[(y + self.pad) * self.stride + x + self.pad]
	}

	/// Replicates the outermost samples into the padding.
	pub fn extend_edges(&mut self) {
		let (w, h, p, s) = (self.width, self.height, self.pad, self.stride);
		for y in 0..h {
			let row = (y + p) * s;
			let left = self.data[row + p];
			let right = self.data[row + p + w - 1];
			self.data[row..row + p].fill(left);
			self.data[row + p + w..row + s].fill(right);
		}
		let first = p * s;
		let last = (p + h - 1) * s;
		for y in 0..p {
			self.data.copy_within(first..first + s, y * s);
			self.data.copy_within(last..last + s, (p + h + y) * s);
		}
	}
}

/// Prediction flags of a motion field entry.
pub const PF_INTRA: u8 = 0;
pub const PF_L0: u8 = 1;
pub const PF_L1: u8 = 2;
pub const PF_BI: u8 = 3;

/// Motion of one 4x4 luma block.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MvField {
	pub mv: [[i16; 2]; 2],
	pub ref_idx: [i8; 2],
	pub pred: u8,
}

/// The reference lists a slice used, by picture identity (POC) and
/// long-term marking, kept with the picture for temporal MV prediction and
/// deblocking.
#[derive(Clone, Debug, Default)]
pub struct RefInfo {
	pub poc: [[i32; 16]; 2],
	pub long_term: [[bool; 16]; 2],
}

/// A fully decoded picture as kept in the DPB.
pub struct Frame {
	pub planes: [Plane; 3],
	pub poc: i32,
	/// Motion field on the 4x4 grid (`w4` columns).
	pub mvf: Vec<MvField>,
	pub w4: usize,
	/// Slice index of each CTB, into `slice_refs`.
	pub ctb_slice: Vec<u16>,
	pub slice_refs: Vec<RefInfo>,
	pub log2_ctb_size: u32,
	pub ctb_width: usize,
	/// Bit depth of the samples.
	pub bit_depth: u32,
}

pub type FrameRef = Arc<Frame>;

impl Frame {
	/// Motion at luma sample (x, y).
	#[inline]
	pub fn mv_at(&self, x: usize, y: usize) -> &MvField {
		&self.mvf[(y >> 2) * self.w4 + (x >> 2)]
	}

	/// Reference lists of the slice covering luma sample (x, y).
	pub fn refs_at(&self, x: usize, y: usize) -> &RefInfo {
		let ctb = (y >> self.log2_ctb_size) * self.ctb_width + (x >> self.log2_ctb_size);
		let s = self.ctb_slice.get(ctb).copied().unwrap_or(0) as usize;
		&self.slice_refs[s.min(self.slice_refs.len().saturating_sub(1))]
	}
}
