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
