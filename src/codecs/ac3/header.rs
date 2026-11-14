//! AC-3 `syncinfo()` / `bsi()` and the E-AC-3 equivalent (ATSC A/52
//! §5.3.1-5.3.2 and Annex E §E.1.2.1-E.1.2.2). FFmpeg: `ac3_parser.c`.
//!
//! [`Header::parse`] leaves the bit reader at the first bit after the
//! bitstream information, where `audfrm()` (E-AC-3) or the first
//! `audblk()` (AC-3) starts.

use super::tables::FRAME_SIZE_WORDS;
use crate::codecs::bitstream::BitReader;

pub const SYNC_WORD: u16 = 0x0B77;
/// Bytes needed to read the fields that give a frame's size.
pub const MIN_HEADER_BYTES: usize = 7;
pub const SAMPLE_RATES: [u32; 3] = [48_000, 44_100, 32_000];
const BITRATES: [u32; 19] =
	[32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640];
/// Full-bandwidth channels by `acmod` (Table 5.8).
pub const ACMOD_CHANNELS: [usize; 8] = [2, 1, 2, 3, 3, 4, 4, 5];
const EAC3_BLOCKS: [usize; 4] = [1, 2, 3, 6];
const CENTER_LEVELS: [u8; 4] = [4, 5, 6, 5];
const SURROUND_LEVELS: [u8; 4] = [4, 6, 7, 6];

/// `acmod` values.
pub const MODE_DUAL_MONO: u8 = 0;
pub const MODE_MONO: u8 = 1;
pub const MODE_STEREO: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
	/// E-AC-3 independent substream.
	Independent,
	/// E-AC-3 dependent substream (extra or replacement channels).
	Dependent,
	/// Plain AC-3, or E-AC-3 converted from AC-3.
	Ac3Convert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderError {
	Sync,
	BitstreamId,
	SampleRate,
	FrameSize,
	/// Reserved frame type or a substream other than 0.
	FrameType,
	ChannelMap,
}

#[derive(Debug, Clone, Default)]
pub struct Header {
	pub bitstream_id: u8,
	pub frame_type: Option<FrameType>,
	pub substream_id: u8,
	/// Frame size in bytes.
	pub frame_size: usize,
	pub sr_code: u8,
	/// Right shift of the bit allocation tables (AC-3 half/quarter rates and
	/// E-AC-3 reduced rates).
	pub sr_shift: u8,
	pub sample_rate: u32,
	pub bit_rate: u32,
	pub num_blocks: usize,
	pub channel_mode: u8,
	pub lfe_on: bool,
	/// Coded channels including the LFE.
	pub channels: usize,
	pub bitstream_mode: u8,
	pub center_mix_level: u8,
	pub surround_mix_level: u8,
	pub dialog_normalization: [i8; 2],
	pub compression_exists: [bool; 2],
	pub heavy_dynamic_range: [u8; 2],
	/// Dependent substream `chanmap` (0 when absent).
	pub channel_map: u16,
	/// Atmos (JOC) signalling in the additional bitstream info.
	pub extension_type_a: bool,
}

impl Header {
	pub fn is_eac3(&self) -> bool {
		self.bitstream_id > 10
	}

	/// Cheap frame-size probe on the first bytes of a frame (for splitting
	/// raw elementary streams). Returns `(frame size, is E-AC-3)`.
	pub fn peek_size(data: &[u8]) -> Option<(usize, bool)> {
		if data.len() < MIN_HEADER_BYTES || u16::from_be_bytes([data[0], data[1]]) != SYNC_WORD {
			return None;
		}
		let bsid = data[5] >> 3;
		if bsid <= 10 {
			let sr = usize::from(data[4] >> 6);
			let code = usize::from(data[4] & 0x3F);
			if sr == 3 || code > 37 {
				return None;
			}
			Some((usize::from(FRAME_SIZE_WORDS[code][sr]) * 2, false))
		} else if bsid <= 16 {
			let words = (usize::from(data[2] & 0x07) << 8 | usize::from(data[3])) + 1;
			Some((words * 2, true))
		} else {
			None
		}
	}

	/// Parses syncinfo + bsi.
	pub fn parse(bs: &mut BitReader) -> Result<Header, HeaderError> {
		if bs.bits(16) as u16 != SYNC_WORD {
			return Err(HeaderError::Sync);
		}
		let bitstream_id = (bs.peek(29) & 0x1F) as u8;
		if bitstream_id > 16 {
			return Err(HeaderError::BitstreamId);
		}
		let mut h = Header {
			bitstream_id,
			num_blocks: 6,
			center_mix_level: 5,
			surround_mix_level: 6,
			..Header::default()
		};
		if bitstream_id <= 10 {
			bs.skip(16); // crc1
			h.sr_code = bs.bits(2) as u8;
			if h.sr_code == 3 {
				return Err(HeaderError::SampleRate);
			}
			let frame_size_code = bs.bits(6) as usize;
			if frame_size_code > 37 {
				return Err(HeaderError::FrameSize);
			}
			bs.skip(5); // bsid
			h.bitstream_mode = bs.bits(3) as u8;
			h.channel_mode = bs.bits(3) as u8;
			if h.channel_mode == MODE_STEREO {
				bs.skip(2); // dsurmod
			} else {
				if h.channel_mode & 1 != 0 && h.channel_mode != MODE_MONO {
					h.center_mix_level = CENTER_LEVELS[bs.bits(2) as usize];
				}
				if h.channel_mode & 4 != 0 {
					h.surround_mix_level = SURROUND_LEVELS[bs.bits(2) as usize];
				}
			}
			h.lfe_on = bs.bit();
			h.sr_shift = bitstream_id.max(8) - 8;
			h.sample_rate = SAMPLE_RATES[usize::from(h.sr_code)] >> h.sr_shift;
			h.bit_rate = (BITRATES[frame_size_code >> 1] * 1000) >> h.sr_shift;
			h.channels = ACMOD_CHANNELS[usize::from(h.channel_mode)] + usize::from(h.lfe_on);
			h.frame_size = usize::from(FRAME_SIZE_WORDS[frame_size_code][usize::from(h.sr_code)]) * 2;
			h.frame_type = Some(FrameType::Ac3Convert);
			parse_ac3_bsi(bs, &mut h);
		} else {
			h.frame_type = match bs.bits(2) {
				0 => Some(FrameType::Independent),
				1 => Some(FrameType::Dependent),
				2 => Some(FrameType::Ac3Convert),
				_ => return Err(HeaderError::FrameType),
			};
			h.substream_id = bs.bits(3) as u8;
			h.frame_size = (bs.bits(11) as usize + 1) << 1;
			if h.frame_size < MIN_HEADER_BYTES {
				return Err(HeaderError::FrameSize);
			}
			h.sr_code = bs.bits(2) as u8;
			if h.sr_code == 3 {
				let sr_code2 = bs.bits(2) as usize;
				if sr_code2 == 3 {
					return Err(HeaderError::SampleRate);
				}
				h.sample_rate = SAMPLE_RATES[sr_code2] / 2;
				h.sr_shift = 1;
			} else {
				h.num_blocks = EAC3_BLOCKS[bs.bits(2) as usize];
				h.sample_rate = SAMPLE_RATES[usize::from(h.sr_code)];
			}
			h.channel_mode = bs.bits(3) as u8;
			h.lfe_on = bs.bit();
			h.bit_rate =
				(8 * h.frame_size as u64 * u64::from(h.sample_rate) / (h.num_blocks as u64 * 256)) as u32;
			h.channels = ACMOD_CHANNELS[usize::from(h.channel_mode)] + usize::from(h.lfe_on);
			if h.substream_id != 0 {
				return Err(HeaderError::FrameType);
			}
			parse_eac3_bsi(bs, &mut h)?;
		}
		Ok(h)
	}
}

fn parse_ac3_bsi(bs: &mut BitReader, h: &mut Header) {
	for i in 0..if h.channel_mode == 0 { 2 } else { 1 } {
		h.dialog_normalization[i] = -(bs.bits(5) as i8);
		h.compression_exists[i] = bs.bit();
		if h.compression_exists[i] {
			h.heavy_dynamic_range[i] = bs.bits(8) as u8;
		}
		if bs.bit() {
			bs.skip(8); // langcod
		}
		if bs.bit() {
			bs.skip(7); // mixlevel, roomtyp
		}
	}
	bs.skip(2); // copyrightb, origbs
	if h.bitstream_id != 6 {
		if bs.bit() {
			bs.skip(14); // timecod1
		}
		if bs.bit() {
			bs.skip(14); // timecod2
		}
	} else {
		// Alternate bit stream syntax (Annex D).
		if bs.bit() {
			bs.skip(2 + 3 + 3);
			h.center_mix_level = bs.bits(3) as u8;
			h.surround_mix_level = (bs.bits(3) as u8).clamp(3, 7);
		}
		if bs.bit() {
			bs.skip(2 + 2 + 10);
		}
	}
	if bs.bit() {
		let n = bs.bits(6) as usize + 1;
		bs.skip(8 * n); // addbsi
	}
}

fn parse_eac3_bsi(bs: &mut BitReader, h: &mut Header) -> Result<(), HeaderError> {
	bs.skip(5); // bsid
	let dual = h.channel_mode == 0;
	for i in 0..if dual { 2 } else { 1 } {
		h.dialog_normalization[i] = -(bs.bits(5) as i8);
		h.compression_exists[i] = bs.bit();
		if h.compression_exists[i] {
			h.heavy_dynamic_range[i] = bs.bits(8) as u8;
		}
	}
	if h.frame_type == Some(FrameType::Dependent) && bs.bit() {
		let map = bs.bits(16) as u16;
		let mut count = 0;
		for (i, loc) in super::decoder::CUSTOM_LOCATIONS.iter().enumerate() {
			if map & (1 << (15 - i)) != 0 {
				count += loc.1.count_ones();
			}
		}
		if count > 16 {
			return Err(HeaderError::ChannelMap);
		}
		h.channel_map = map;
	}
	// Mixing metadata.
	if bs.bit() {
		if h.channel_mode > MODE_STEREO {
			bs.skip(2); // dmixmod
			if h.channel_mode & 1 != 0 {
				bs.skip(3); // ltrtcmixlev
				h.center_mix_level = bs.bits(3) as u8;
			}
			if h.channel_mode & 4 != 0 {
				bs.skip(3); // ltrtsurmixlev
				h.surround_mix_level = (bs.bits(3) as u8).clamp(3, 7);
			}
		}
		if h.lfe_on && bs.bit() {
			bs.skip(5); // lfemixlevcod
		}
		if h.frame_type == Some(FrameType::Independent) {
			for _ in 0..if dual { 2 } else { 1 } {
				if bs.bit() {
					bs.skip(6); // pgmscl
				}
			}
			if bs.bit() {
				bs.skip(6); // extpgmscl
			}
			match bs.bits(2) {
				1 => bs.skip(5),
				2 => bs.skip(12),
				3 => {
					let size = (bs.bits(5) as usize + 2) << 3;
					bs.skip(size);
				}
				_ => {}
			}
			if h.channel_mode < MODE_STEREO {
				for _ in 0..if dual { 2 } else { 1 } {
					if bs.bit() {
						bs.skip(8 + 6); // panmean, paninfo
					}
				}
			}
			if bs.bit() {
				for _ in 0..h.num_blocks {
					if h.num_blocks == 1 || bs.bit() {
						bs.skip(5); // blkmixcfginfo
					}
				}
			}
		}
	}
	// Informational metadata.
	if bs.bit() {
		h.bitstream_mode = bs.bits(3) as u8;
		bs.skip(2);
		if h.channel_mode == MODE_STEREO {
			bs.skip(4); // dsurmod, dheadphonmod
		}
		if h.channel_mode >= 6 {
			bs.skip(2); // dsurexmod
		}
		for _ in 0..if dual { 2 } else { 1 } {
			if bs.bit() {
				bs.skip(8); // mixlevel, roomtyp, adconvtyp
			}
		}
		if h.sr_code != 3 {
			bs.skip(1); // sourcefscod
		}
	}
	if h.frame_type == Some(FrameType::Independent) && h.num_blocks != 6 {
		bs.skip(1); // convsync
	}
	if h.frame_type == Some(FrameType::Ac3Convert) && (h.num_blocks == 6 || bs.bit()) {
		bs.skip(6); // frmsizecod
	}
	if bs.bit() {
		let len = bs.bits(6) as usize + 1;
		let mut i = 0;
		while i < len {
			if i == 0 {
				bs.skip(7);
				h.extension_type_a = bs.bit();
				if h.extension_type_a {
					bs.skip(8); // complexity_index_type_a
					i += 1;
				}
			} else {
				bs.skip(8);
			}
			i += 1;
		}
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rejects_bad_sync() {
		let data = [0u8; 16];
		assert_eq!(Header::parse(&mut BitReader::new(&data)).unwrap_err(), HeaderError::Sync);
		assert!(Header::peek_size(&data).is_none());
	}

	#[test]
	fn ac3_frame_size_from_table() {
		// 48 kHz, frmsizecod 28 (384 kbps): 1536 bytes.
		let data = [0x0B, 0x77, 0, 0, 28, 8 << 3, 0, 0];
		assert_eq!(Header::peek_size(&data), Some((1536, false)));
	}
}
