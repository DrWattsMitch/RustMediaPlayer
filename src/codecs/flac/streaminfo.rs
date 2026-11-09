//! The FLAC `STREAMINFO` metadata block and the extradata forms it travels
//! in.
//!
//! RMP's FLAC extradata is FFmpeg's: the bare 34-byte STREAMINFO body. Other
//! containers wrap it differently, and [`StreamInfo::from_extradata`] accepts
//! all of them:
//!
//! - native FLAC and Matroska `CodecPrivate`: `"fLaC"` + metadata blocks;
//! - MP4 `dfLa`: a full-box header (version, flags) + metadata blocks;
//! - a bare metadata block (4-byte header + STREAMINFO) or the bare body.
//!
//! FFmpeg: `libavcodec/flac.c` (`ff_flac_parse_streaminfo`,
//! `ff_flac_is_extradata_valid`).

use crate::codecs::bitstream::BitReader;

/// Size of a STREAMINFO body.
pub const STREAMINFO_SIZE: usize = 34;
/// Metadata block type of STREAMINFO.
pub const BLOCK_STREAMINFO: u8 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamInfo {
	pub min_block_size: u16,
	pub max_block_size: u16,
	pub min_frame_size: u32,
	pub max_frame_size: u32,
	pub sample_rate: u32,
	pub channels: u8,
	pub bits_per_sample: u8,
	/// Total samples per channel, 0 = unknown.
	pub total_samples: u64,
	pub md5: [u8; 16],
}

impl StreamInfo {
	/// Parses a 34-byte STREAMINFO body, validating the fields a decoder
	/// relies on.
	pub fn parse(body: &[u8]) -> Option<Self> {
		let body = body.get(..STREAMINFO_SIZE)?;
		let mut bs = BitReader::new(body);
		let min_block_size = bs.bits(16) as u16;
		let max_block_size = bs.bits(16) as u16;
		let min_frame_size = bs.bits(24);
		let max_frame_size = bs.bits(24);
		let sample_rate = bs.bits(20);
		let channels = bs.bits(3) as u8 + 1;
		let bits_per_sample = bs.bits(5) as u8 + 1;
		let total_samples = (u64::from(bs.bits(4)) << 32) | u64::from(bs.bits(32));
		let mut md5 = [0u8; 16];
		md5.copy_from_slice(&body[18..34]);
		if sample_rate == 0 || max_block_size < 16 || bits_per_sample < 4 {
			return None;
		}
		Some(Self {
			min_block_size,
			max_block_size,
			min_frame_size,
			max_frame_size,
			sample_rate,
			channels,
			bits_per_sample,
			total_samples,
			md5,
		})
	}

	/// Finds STREAMINFO in any of the extradata forms listed in the module docs.
	pub fn from_extradata(data: &[u8]) -> Option<Self> {
		if data.len() == STREAMINFO_SIZE {
			return Self::parse(data);
		}
		let blocks = if let Some(rest) = data.strip_prefix(b"fLaC") {
			rest
		} else if data.len() >= 4 + 4 + STREAMINFO_SIZE && data[..4] == [0, 0, 0, 0] {
			// dfLa: version 0 and zero flags, then the STREAMINFO block. (A
			// bare STREAMINFO block header is never all zero: its length is 34.)
			&data[4..]
		} else {
			data
		};
		let header = blocks.get(..4)?;
		let len = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
		if header[0] & 0x7F != BLOCK_STREAMINFO || len < STREAMINFO_SIZE {
			return None;
		}
		Self::parse(blocks.get(4..)?)
	}

	/// Serialises the 34-byte STREAMINFO body.
	pub fn to_bytes(&self) -> [u8; STREAMINFO_SIZE] {
		let mut out = [0u8; STREAMINFO_SIZE];
		out[0..2].copy_from_slice(&self.min_block_size.to_be_bytes());
		out[2..4].copy_from_slice(&self.max_block_size.to_be_bytes());
		out[4..7].copy_from_slice(&self.min_frame_size.to_be_bytes()[1..]);
		out[7..10].copy_from_slice(&self.max_frame_size.to_be_bytes()[1..]);
		let packed: u64 = (u64::from(self.sample_rate & 0xF_FFFF) << 44)
			| (u64::from(self.channels.saturating_sub(1) & 7) << 41)
			| (u64::from(self.bits_per_sample.saturating_sub(1) & 31) << 36)
			| (self.total_samples & 0xF_FFFF_FFFF);
		out[10..18].copy_from_slice(&packed.to_be_bytes());
		out[18..34].copy_from_slice(&self.md5);
		out
	}

	/// A metadata block (header + body), with the "last block" flag as given.
	pub fn to_block(&self, last: bool) -> Vec<u8> {
		let mut out = vec![if last { 0x80 } else { 0 }, 0, 0, STREAMINFO_SIZE as u8];
		out.extend_from_slice(&self.to_bytes());
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn sample() -> StreamInfo {
		StreamInfo {
			min_block_size: 4096,
			max_block_size: 4096,
			min_frame_size: 14,
			max_frame_size: 9876,
			sample_rate: 44_100,
			channels: 2,
			bits_per_sample: 16,
			total_samples: 123_456_789,
			md5: [7; 16],
		}
	}

	#[test]
	fn round_trips_every_wrapping() {
		let info = sample();
		let body = info.to_bytes();
		assert_eq!(StreamInfo::parse(&body), Some(info));
		let mut native = b"fLaC".to_vec();
		native.extend(info.to_block(true));
		assert_eq!(StreamInfo::from_extradata(&native), Some(info));
		let mut dfla = vec![0, 0, 0, 0];
		dfla.extend(info.to_block(true));
		assert_eq!(StreamInfo::from_extradata(&dfla), Some(info));
		assert_eq!(StreamInfo::from_extradata(&info.to_block(false)), Some(info));
		assert_eq!(StreamInfo::from_extradata(&body), Some(info));
		assert_eq!(StreamInfo::from_extradata(b"fLaC"), None);
	}
}
