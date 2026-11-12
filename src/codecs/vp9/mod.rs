//! VP9 video decoder (VP9 Bitstream & Decoding Process Specification v0.6;
//! libvpx `vp9/decoder`, `vp9/common`, `vpx_dsp`).
//!
//! Profiles 0 and 1 (8-bit; 4:2:0, 4:2:2, 4:4:0 and 4:4:4), every frame
//! type (key, inter, intra-only, hidden frames and `show_existing_frame`),
//! tiles, segmentation, loop-filter deltas, compound prediction, switchable
//! interpolation filters, lossless (WHT) frames, backward probability
//! adaptation and superframes. Output is bit-exact with libvpx / FFmpeg.
//! Not supported (decoding fails with an error): 10/12-bit profiles 2 and
//! 3, and inter prediction from a reference of a different size.
//!
//! Layout:
//! - [`range`]: the boolean (arithmetic) decoder;
//! - [`header`]: uncompressed and compressed frame headers;
//! - [`probs`]: probability contexts, counts and adaptation;
//! - [`mode`]: per-block mode info (segments, skip, modes, references,
//!   motion vectors and their candidate search);
//! - [`residual`]: coefficient tokens;
//! - [`transform`]: inverse DCT / ADST / WHT;
//! - [`intra`], [`inter`]: prediction;
//! - [`loopfilter`]: the deblocking filter;
//! - [`decoder`]: frames, tiles, partitions, references and the
//!   [`crate::core::Decoder`] implementation.
//!
//! See `docs/VORBIS_VP9.md`.

pub mod decoder;
pub mod header;
mod inter;
mod intra;
mod loopfilter;
mod mode;
mod probs;
pub mod range;
mod residual;
mod simd;
mod tables;
mod transform;

pub use decoder::{Vp9Decoder, Vp9DecoderState};

/// Block sizes (`BLOCK_SIZE`), in libvpx order.
pub(crate) mod bs {
	pub const B4X4: u8 = 0;
	pub const B4X8: u8 = 1;
	pub const B8X4: u8 = 2;
	pub const B8X8: u8 = 3;
	pub const B64X64: u8 = 12;
	pub const INVALID: u8 = 255;
}

/// Width of each block size in 4-pixel units, log2.
pub(crate) const B_WIDTH_LOG2: [u8; 13] = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4];
/// Height of each block size in 4-pixel units, log2.
pub(crate) const B_HEIGHT_LOG2: [u8; 13] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
/// Width in 8x8 units (at least 1).
pub(crate) const NUM_8X8_WIDE: [u8; 13] = [1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8];
/// Height in 8x8 units (at least 1).
pub(crate) const NUM_8X8_HIGH: [u8; 13] = [1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8];
/// Width in 4x4 units.
pub(crate) const NUM_4X4_WIDE: [u8; 13] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16];
/// Height in 4x4 units.
pub(crate) const NUM_4X4_HIGH: [u8; 13] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16];
/// Luma mode probability group (inter frames).
pub(crate) const SIZE_GROUP: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
/// Largest transform of each block size.
pub(crate) const MAX_TX_SIZE: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3];
/// Block size after a partition `[partition][size]`.
pub(crate) const SUBSIZE: [[u8; 13]; 4] = {
	const X: u8 = bs::INVALID;
	[
		[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
		[X, X, X, 2, X, X, 5, X, X, 8, X, X, 11],
		[X, X, X, 1, X, X, 4, X, X, 7, X, X, 10],
		[X, X, X, 0, X, X, 3, X, X, 6, X, X, 9],
	]
};
/// Chroma block size `[size][ss_x][ss_y]` (`ss_size_lookup`).
pub(crate) const SS_SIZE: [[[u8; 2]; 2]; 13] = {
	const X: u8 = bs::INVALID;
	[
		[[0, X], [X, X]],
		[[1, 0], [X, X]],
		[[2, X], [0, X]],
		[[3, 2], [1, 0]],
		[[4, 3], [X, 1]],
		[[5, X], [3, 2]],
		[[6, 5], [4, 3]],
		[[7, 6], [X, 4]],
		[[8, X], [6, 5]],
		[[9, 8], [7, 6]],
		[[10, 9], [X, 7]],
		[[11, X], [9, 8]],
		[[12, 11], [10, 9]],
	]
};
/// Partition context bits written above and left of each block size.
pub(crate) const PARTITION_CONTEXT: [(u8, u8); 13] = [
	(15, 15),
	(15, 14),
	(14, 15),
	(14, 14),
	(14, 12),
	(12, 14),
	(12, 12),
	(12, 8),
	(8, 12),
	(8, 8),
	(8, 0),
	(0, 8),
	(0, 0),
];

/// Transform size of a chroma plane (`get_uv_tx_size`): the luma transform
/// limited by the chroma block size.
pub(crate) fn uv_tx_size(block: u8, tx: u8, ss_x: usize, ss_y: usize) -> u8 {
	if block < bs::B8X8 {
		return 0;
	}
	let uv = SS_SIZE[usize::from(block)][ss_x][ss_y];
	if uv == bs::INVALID {
		return 0;
	}
	let largest = B_WIDTH_LOG2[usize::from(uv)].min(B_HEIGHT_LOG2[usize::from(uv)]);
	tx.min(largest)
}

/// Prediction modes (`PREDICTION_MODE`).
pub(crate) mod pm {
	pub const DC: u8 = 0;
	pub const V: u8 = 1;
	pub const H: u8 = 2;
	pub const D45: u8 = 3;
	pub const D135: u8 = 4;
	pub const D117: u8 = 5;
	pub const D153: u8 = 6;
	pub const D207: u8 = 7;
	pub const D63: u8 = 8;
	pub const TM: u8 = 9;
	pub const NEARESTMV: u8 = 10;
	pub const NEARMV: u8 = 11;
	pub const ZEROMV: u8 = 12;
	pub const NEWMV: u8 = 13;
}

/// Reference frames (`MV_REFERENCE_FRAME`).
pub(crate) const INTRA_FRAME: i8 = 0;
pub(crate) const LAST_FRAME: i8 = 1;
pub(crate) const GOLDEN_FRAME: i8 = 2;
pub(crate) const ALTREF_FRAME: i8 = 3;
pub(crate) const NO_REF: i8 = -1;

/// A motion vector in 1/8 pixel units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mv {
	pub row: i16,
	pub col: i16,
}

impl Mv {
	pub const ZERO: Mv = Mv { row: 0, col: 0 };

	pub fn new(row: i32, col: i32) -> Self {
		Self { row: row as i16, col: col as i16 }
	}

	pub fn negated(self) -> Self {
		Self { row: self.row.wrapping_neg(), col: self.col.wrapping_neg() }
	}
}

/// Tree-coded symbols: a tree is pairs of entries, `<= 0` is a leaf
/// (`-symbol`), `> 0` the index of the next pair (libvpx `vpx_tree_index`).
pub(crate) type Tree = [i8];

pub(crate) const INTRA_MODE_TREE: [i8; 18] = [
	-(pm::DC as i8),
	2,
	-(pm::TM as i8),
	4,
	-(pm::V as i8),
	6,
	8,
	12,
	-(pm::H as i8),
	10,
	-(pm::D135 as i8),
	-(pm::D117 as i8),
	-(pm::D45 as i8),
	14,
	-(pm::D63 as i8),
	16,
	-(pm::D153 as i8),
	-(pm::D207 as i8),
];
/// Inter modes relative to NEARESTMV: ZERO (2), NEAREST (0), NEAR (1), NEW (3).
pub(crate) const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
pub(crate) const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
/// EIGHTTAP (0), EIGHTTAP_SMOOTH (1), EIGHTTAP_SHARP (2).
pub(crate) const INTERP_FILTER_TREE: [i8; 4] = [0, 2, -1, -2];
pub(crate) const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
pub(crate) const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub(crate) const MV_CLASS_TREE: [i8; 20] =
	[0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10];
pub(crate) const MV_CLASS0_TREE: [i8; 2] = [0, -1];
pub(crate) const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

/// Interpolation filters (`INTERP_FILTER`).
pub(crate) const FILTER_SWITCHABLE: u8 = 4;
