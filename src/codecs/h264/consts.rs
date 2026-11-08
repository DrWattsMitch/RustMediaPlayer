//! Small normative constants of ISO/IEC 14496-10 (scan orders, CBP
//! mappings, deblocking tables). Larger tables are generated into
//! `tables.rs`.

/// 4x4 frame zig-zag scan: scan position → raster index (Table 8-13).
pub const ZIGZAG_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// 8x8 frame zig-zag scan: scan position → raster index (Table 8-14).
pub const ZIGZAG_8X8: [u8; 64] = [
	0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
	13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
	45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Luma 4x4 block index (decoding order, Figure 6-10) → (x, y) in 4x4 units.
pub const BLK_XY: [(u8, u8); 16] = [
	(0, 0),
	(1, 0),
	(0, 1),
	(1, 1),
	(2, 0),
	(3, 0),
	(2, 1),
	(3, 1),
	(0, 2),
	(1, 2),
	(0, 3),
	(1, 3),
	(2, 2),
	(3, 2),
	(2, 3),
	(3, 3),
];

/// Raster 4x4 position (y * 4 + x) → decoding-order block index.
pub const RASTER_TO_BLK: [u8; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// `coded_block_pattern` for Intra_4x4/8x8 macroblocks by `me(v)` codeNum
/// (Table 9-4, ChromaArrayType 1 or 2).
pub const GOLOMB_TO_INTRA_CBP: [u8; 48] = [
	47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
	35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

/// `coded_block_pattern` for inter macroblocks by `me(v)` codeNum.
pub const GOLOMB_TO_INTER_CBP: [u8; 48] = [
	0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34, 36,
	40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// QPc as a function of qPI for qPI >= 30 (Table 8-15); below 30 QPc = qPI.
pub const CHROMA_QP_HIGH: [u8; 22] =
	[29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];

pub fn chroma_qp(qp_y: i32, offset: i32) -> u8 {
	let qpi = (qp_y + offset).clamp(0, 51);
	if qpi < 30 { qpi as u8 } else { CHROMA_QP_HIGH[(qpi - 30) as usize] }
}

/// normAdjust4x4 (8-315) values `v[m][0..3]`.
pub const NORM_ADJUST_4X4: [[u8; 3]; 6] =
	[[10, 16, 13], [11, 18, 14], [13, 20, 16], [14, 23, 18], [16, 25, 20], [18, 29, 23]];

/// normAdjust8x8 (8-318) values `v[m][0..6]`.
pub const NORM_ADJUST_8X8: [[u8; 6]; 6] = [
	[20, 18, 32, 19, 25, 24],
	[22, 19, 35, 21, 28, 26],
	[26, 23, 42, 24, 33, 31],
	[28, 25, 45, 26, 35, 33],
	[32, 28, 51, 30, 40, 38],
	[36, 32, 58, 34, 46, 43],
];

/// Deblocking α' by indexA (Table 8-16).
pub const ALPHA: [u8; 52] = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22,
	25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255,
];

/// Deblocking β' by indexB (Table 8-16).
pub const BETA: [u8; 52] = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
	9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Deblocking tC0 by indexA and bS 1..3 (Table 8-17).
pub const TC0: [[u8; 3]; 52] = [
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 1],
	[0, 0, 1],
	[0, 0, 1],
	[0, 0, 1],
	[0, 1, 1],
	[0, 1, 1],
	[1, 1, 1],
	[1, 1, 1],
	[1, 1, 1],
	[1, 1, 1],
	[1, 1, 2],
	[1, 1, 2],
	[1, 1, 2],
	[1, 1, 2],
	[1, 2, 3],
	[1, 2, 3],
	[2, 2, 3],
	[2, 2, 4],
	[2, 3, 4],
	[2, 3, 4],
	[3, 3, 5],
	[3, 4, 6],
	[3, 4, 6],
	[4, 5, 7],
	[4, 5, 8],
	[4, 6, 9],
	[5, 7, 10],
	[6, 8, 11],
	[6, 8, 13],
	[7, 10, 14],
	[8, 11, 16],
	[9, 12, 18],
	[10, 13, 20],
	[11, 15, 23],
	[13, 17, 25],
];

/// CABAC ctxIdxInc of significant_coeff_flag for frame-coded 8x8 blocks
/// (Table 9-43).
pub const SIG_8X8_CTX: [u8; 63] = [
	0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7,
	7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13,
	11, 14, 10, 12,
];

/// CABAC ctxIdxInc of last_significant_coeff_flag for 8x8 blocks (Table 9-43).
pub const LAST_8X8_CTX: [u8; 63] = [
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];

/// Maximum decoded picture buffer size in macroblocks per level_idc
/// (Table A-1).
pub fn max_dpb_mbs(level_idc: u8) -> u32 {
	match level_idc {
		0..=10 => 396,
		11 => 900,
		12 | 13 | 20 => 2376,
		21 => 4752,
		22 | 30 => 8100,
		31 => 18000,
		32 => 20480,
		40 | 41 => 32768,
		42 => 34816,
		50 => 110400,
		51 | 52 => 184320,
		_ => 696320,
	}
}
