//! NAL unit framing: `avcC` configuration records (MP4), length-prefixed
//! access units, Annex B start codes and emulation-prevention removal
//! (FFmpeg: `h264_parse.c`, `h2645_parse.c`).

/// NAL unit types used by the decoder (Table 7-1).
pub mod kind {
	pub const SLICE: u8 = 1;
	pub const IDR: u8 = 5;
	pub const SEI: u8 = 6;
	pub const SPS: u8 = 7;
	pub const PPS: u8 = 8;
	pub const AUD: u8 = 9;
	pub const END_SEQUENCE: u8 = 10;
}

/// A NAL unit with emulation-prevention bytes removed.
#[derive(Debug, Clone)]
pub struct Nal {
	pub ref_idc: u8,
	pub kind: u8,
	/// RBSP payload (after the one-byte header).
	pub rbsp: Vec<u8>,
}

impl Nal {
	pub fn parse(data: &[u8]) -> Option<Self> {
		let (&header, payload) = data.split_first()?;
		if header & 0x80 != 0 {
			return None; // forbidden_zero_bit
		}
		Some(Self { ref_idc: (header >> 5) & 3, kind: header & 0x1F, rbsp: unescape(payload) })
	}
}

/// Removes `emulation_prevention_three_byte`s (00 00 03 → 00 00).
pub fn unescape(data: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(data.len());
	let mut zeros = 0usize;
	for &b in data {
		if zeros >= 2 && b == 3 {
			zeros = 0;
			continue;
		}
		zeros = if b == 0 { zeros + 1 } else { 0 };
		out.push(b);
	}
	out
}

/// Parsed `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §5.3.3.1).
#[derive(Debug, Clone, Default)]
pub struct AvcConfig {
	pub nal_length_size: usize,
	pub sps: Vec<Vec<u8>>,
	pub pps: Vec<Vec<u8>>,
}

impl AvcConfig {
	pub fn parse(data: &[u8]) -> Option<Self> {
		if data.len() < 7 || data[0] != 1 {
			return None;
		}
		let nal_length_size = usize::from(data[4] & 3) + 1;
		let mut at = 5;
		let take_list = |count: usize, at: &mut usize| -> Option<Vec<Vec<u8>>> {
			let mut list = Vec::with_capacity(count);
			for _ in 0..count {
				let len = usize::from(u16::from_be_bytes([*data.get(*at)?, *data.get(*at + 1)?]));
				let nal = data.get(*at + 2..*at + 2 + len)?;
				list.push(nal.to_vec());
				*at += 2 + len;
			}
			Some(list)
		};
		let n_sps = usize::from(*data.get(at)? & 0x1F);
		at += 1;
		let sps = take_list(n_sps, &mut at)?;
		let n_pps = usize::from(*data.get(at)?);
		at += 1;
		let pps = take_list(n_pps, &mut at)?;
		Some(Self { nal_length_size, sps, pps })
	}
}

/// Splits a length-prefixed access unit into NAL units. Stops at the first
/// length that runs past the end of the data.
pub fn split_length_prefixed(data: &[u8], length_size: usize) -> Vec<&[u8]> {
	let mut out = Vec::new();
	let mut at = 0usize;
	while at + length_size <= data.len() {
		let len = data[at..at + length_size].iter().fold(0usize, |acc, &b| (acc << 8) | usize::from(b));
		at += length_size;
		match data.get(at..at.saturating_add(len)) {
			Some(nal) if len > 0 => out.push(nal),
			Some(_) => {}
			None => break,
		}
		at += len;
	}
	out
}

/// Splits an Annex B byte stream on `00 00 01` start codes.
pub fn split_annex_b(data: &[u8]) -> Vec<&[u8]> {
	let mut starts = Vec::new();
	let mut i = 0;
	while i + 3 <= data.len() {
		if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
			starts.push(i + 3);
			i += 3;
		} else {
			i += 1;
		}
	}
	let mut out = Vec::with_capacity(starts.len());
	for (n, &s) in starts.iter().enumerate() {
		let mut end = starts.get(n + 1).map_or(data.len(), |&next| next - 3);
		// Trailing zero bytes belong to the next start code (or are padding).
		while end > s && data[end - 1] == 0 {
			end -= 1;
		}
		if end > s {
			out.push(&data[s..end]);
		}
	}
	out
}

/// `true` when `data` starts with an Annex B start code.
pub fn is_annex_b(data: &[u8]) -> bool {
	data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

/// `true` if there is RBSP data before the trailing stop bit at bit `pos`
/// (`more_rbsp_data()`).
pub fn more_rbsp_data(rbsp: &[u8], pos: usize) -> bool {
	let Some(last) = rbsp.iter().rposition(|&b| b != 0) else {
		return false;
	};
	let stop_bit = last * 8 + 7 - rbsp[last].trailing_zeros() as usize;
	pos < stop_bit
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn removes_emulation_prevention() {
		assert_eq!(unescape(&[0, 0, 3, 1, 0, 0, 3, 0, 3]), vec![0, 0, 1, 0, 0, 0, 3]);
	}

	#[test]
	fn splits_streams() {
		let au = [0, 0, 0, 2, 0x65, 1, 0, 0, 0, 1, 0x41];
		assert_eq!(split_length_prefixed(&au, 4), vec![&[0x65, 1][..], &[0x41][..]]);
		let annexb = [0, 0, 0, 1, 0x67, 5, 0, 0, 1, 0x68, 0, 0];
		assert_eq!(split_annex_b(&annexb), vec![&[0x67, 5][..], &[0x68][..]]);
	}

	#[test]
	fn detects_trailing_bits() {
		assert!(more_rbsp_data(&[0b1010_0000], 1));
		assert!(!more_rbsp_data(&[0b1010_0000], 2));
	}
}

/// What containers need from an SPS: picture size, chroma format, bit
/// depth and the profile bytes. Unlike [`super::params::parse_sps`] this
/// accepts every profile (10-bit, 4:2:2, ...), since it only reads the
/// header fields (H.264 §7.3.2.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpsSummary {
	pub profile_idc: u8,
	pub constraint_flags: u8,
	pub level_idc: u8,
	pub chroma_format_idc: u32,
	pub bit_depth_luma: u32,
	pub bit_depth_chroma: u32,
	pub width: u32,
	pub height: u32,
}

/// Parses an SPS RBSP (after the NAL header byte). `None` if it is cut
/// short or out of range.
pub fn sps_summary(rbsp: &[u8]) -> Option<SpsSummary> {
	use crate::codecs::bitstream::BitReader;
	fn ue(bs: &mut BitReader) -> Option<u32> {
		let mut zeros = 0;
		while !bs.bit() {
			zeros += 1;
			if zeros > 31 || bs.overrun() {
				return None;
			}
		}
		Some(((1u64 << zeros) - 1 + u64::from(bs.bits(zeros))) as u32)
	}
	fn se(bs: &mut BitReader) -> Option<i32> {
		let k = ue(bs)?;
		Some(if k & 1 == 1 { (k / 2 + 1) as i32 } else { -((k / 2) as i32) })
	}
	let mut bs = BitReader::new(rbsp);
	let profile_idc = bs.bits(8) as u8;
	let constraint_flags = bs.bits(8) as u8;
	let level_idc = bs.bits(8) as u8;
	ue(&mut bs)?; // seq_parameter_set_id
	let (mut chroma_format_idc, mut depth_luma, mut depth_chroma) = (1, 8, 8);
	if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135)
	{
		chroma_format_idc = ue(&mut bs)?;
		if chroma_format_idc == 3 {
			bs.skip(1); // separate_colour_plane_flag
		}
		depth_luma = ue(&mut bs)? + 8;
		depth_chroma = ue(&mut bs)? + 8;
		bs.skip(1); // qpprime_y_zero_transform_bypass_flag
		if bs.bit() {
			let lists = if chroma_format_idc == 3 { 12 } else { 8 };
			for i in 0..lists {
				if !bs.bit() {
					continue;
				}
				let size = if i < 6 { 16 } else { 64 };
				let (mut last, mut next) = (8i32, 8i32);
				for _ in 0..size {
					if next != 0 {
						next = (last + se(&mut bs)? + 256) % 256;
					}
					last = if next == 0 { last } else { next };
				}
			}
		}
	}
	ue(&mut bs)?; // log2_max_frame_num_minus4
	match ue(&mut bs)? {
		0 => {
			ue(&mut bs)?;
		}
		1 => {
			bs.skip(1);
			se(&mut bs)?;
			se(&mut bs)?;
			for _ in 0..ue(&mut bs)?.min(255) {
				se(&mut bs)?;
			}
		}
		_ => {}
	}
	ue(&mut bs)?; // max_num_ref_frames
	bs.skip(1); // gaps_in_frame_num_value_allowed_flag
	let mb_width = ue(&mut bs)?.checked_add(1)?;
	let map_height = ue(&mut bs)?.checked_add(1)?;
	let frame_mbs_only = bs.bit();
	if !frame_mbs_only {
		bs.skip(1); // mb_adaptive_frame_field_flag
	}
	bs.skip(1); // direct_8x8_inference_flag
	let field_factor = if frame_mbs_only { 1 } else { 2 };
	let mut width = mb_width.checked_mul(16)?;
	let mut height = map_height.checked_mul(16 * field_factor)?;
	if bs.bit() {
		let (unit_x, unit_y) = match chroma_format_idc {
			1 => (2, 2 * field_factor),
			2 => (2, field_factor),
			_ => (1, field_factor),
		};
		let (l, r, t, b) = (ue(&mut bs)?, ue(&mut bs)?, ue(&mut bs)?, ue(&mut bs)?);
		width = width.checked_sub(l.checked_add(r)?.checked_mul(unit_x)?)?;
		height = height.checked_sub(t.checked_add(b)?.checked_mul(unit_y)?)?;
	}
	if bs.overrun() || width == 0 || height == 0 || chroma_format_idc > 3 {
		return None;
	}
	Some(SpsSummary {
		profile_idc,
		constraint_flags,
		level_idc,
		chroma_format_idc,
		bit_depth_luma: depth_luma,
		bit_depth_chroma: depth_chroma,
		width,
		height,
	})
}
