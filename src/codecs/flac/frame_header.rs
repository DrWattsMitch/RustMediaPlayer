//! FLAC frame header (FLAC format specification §9.1, RFC 9639 §9.1).
//!
//! FFmpeg: `libavcodec/flac.c` (`ff_flac_decode_frame_header`).

use crate::codecs::bitstream::BitReader;

/// Sample rates selected by codes 1-11 of the frame header.
const RATES: [u32; 12] =
	[0, 88_200, 176_400, 192_000, 8000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000];
/// Bit depths selected by codes 1-7 (0 = from STREAMINFO, 3 reserved).
const DEPTHS: [u8; 8] = [0, 8, 12, 0, 16, 20, 24, 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAssignment {
	/// `n` independent channels.
	Independent(u8),
	LeftSide,
	RightSide,
	MidSide,
}

impl ChannelAssignment {
	pub fn channels(self) -> u8 {
		match self {
			ChannelAssignment::Independent(n) => n,
			_ => 2,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
	pub variable_block_size: bool,
	pub block_size: u32,
	/// 0 = take the rate from STREAMINFO.
	pub sample_rate: u32,
	pub channels: ChannelAssignment,
	/// 0 = take the depth from STREAMINFO.
	pub bits_per_sample: u8,
	/// Frame number (fixed blocking) or first sample number (variable).
	pub number: u64,
	/// Header length in bytes, including the CRC-8.
	pub len: usize,
}

/// CRC-8, polynomial 0x07 (the frame header checksum).
pub fn crc8(data: &[u8]) -> u8 {
	data.iter().fold(0u8, |crc, &b| {
		let mut c = crc ^ b;
		for _ in 0..8 {
			c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
		}
		c
	})
}

/// UTF-8-style coded number (up to 36 bits, 7 bytes).
fn coded_number(bs: &mut BitReader) -> Option<u64> {
	let first = bs.bits(8);
	let ones = (first as u8).leading_ones();
	if ones == 1 || ones > 7 {
		return None;
	}
	if ones == 0 {
		return Some(u64::from(first));
	}
	let mut value = u64::from(first & (0x7F >> ones));
	for _ in 1..ones {
		let byte = bs.bits(8);
		if byte & 0xC0 != 0x80 {
			return None;
		}
		value = (value << 6) | u64::from(byte & 0x3F);
	}
	Some(value)
}

impl FrameHeader {
	/// Parses and CRC-checks a frame header at the start of `data`.
	pub fn parse(data: &[u8]) -> Option<Self> {
		let mut bs = BitReader::new(data);
		if bs.bits(15) != 0x7FFC {
			return None;
		}
		let variable_block_size = bs.bit();
		let block_code = bs.bits(4);
		let rate_code = bs.bits(4);
		let channel_code = bs.bits(4) as u8;
		let depth_code = bs.bits(3) as usize;
		if bs.bit() {
			return None; // reserved
		}
		let channels = match channel_code {
			0..=7 => ChannelAssignment::Independent(channel_code + 1),
			8 => ChannelAssignment::LeftSide,
			9 => ChannelAssignment::RightSide,
			10 => ChannelAssignment::MidSide,
			_ => return None,
		};
		if depth_code == 3 {
			return None;
		}
		let number = coded_number(&mut bs)?;
		let block_size = match block_code {
			0 => return None,
			1 => 192,
			2..=5 => 576 << (block_code - 2),
			6 => bs.bits(8) + 1,
			7 => bs.bits(16) + 1,
			_ => 256 << (block_code - 8),
		};
		let sample_rate = match rate_code {
			0..=11 => RATES[rate_code as usize],
			12 => bs.bits(8) * 1000,
			13 => bs.bits(16),
			14 => bs.bits(16) * 10,
			_ => return None,
		};
		if bs.overrun() || !bs.is_aligned() {
			return None;
		}
		let len = bs.pos() / 8 + 1;
		let crc = *data.get(len - 1)?;
		if crc8(&data[..len - 1]) != crc {
			return None;
		}
		Some(Self {
			variable_block_size,
			block_size,
			sample_rate,
			channels,
			bits_per_sample: DEPTHS[depth_code],
			number,
			len,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_a_header() {
		// Fixed blocking, 4096 samples (code 12), 44.1 kHz (9), stereo
		// independent, 16-bit, frame 0.
		let mut h = vec![0xFF, 0xF8, 0xC9, 0x18, 0x00];
		h.push(crc8(&h));
		let f = FrameHeader::parse(&h).unwrap();
		assert_eq!((f.block_size, f.sample_rate, f.bits_per_sample), (4096, 44_100, 16));
		assert_eq!(f.channels, ChannelAssignment::Independent(2));
		assert_eq!(f.len, 6);
		h[5] ^= 1;
		assert!(FrameHeader::parse(&h).is_none());
	}
}
