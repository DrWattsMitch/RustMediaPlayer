//! MPEG audio frame header (ISO/IEC 11172-3 §2.4.2.3, 13818-3, and the
//! unofficial MPEG-2.5 extension).
//!
//! Shared by the MP3 demuxer/muxer (framing, duration, seeking) and the
//! Layer III decoder. Mirrors FFmpeg's `mpegaudiodecheader.c`.

/// Size of the fixed frame header in bytes.
pub const HEADER_SIZE: usize = 4;

/// Largest possible frame (Layer II, 384 kbit/s, 32 kHz, padded) is 1729
/// bytes; Layer I peaks at 1536. Anything above this is corrupt input.
pub const MAX_FRAME_SIZE: usize = 2881;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpegVersion {
	Mpeg1,
	Mpeg2,
	/// Unofficial low-sample-rate extension (8/11.025/12 kHz).
	Mpeg25,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
	Stereo,
	JointStereo,
	DualChannel,
	Mono,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
	/// The four raw header bytes, kept so the decoder can test bit flags
	/// exactly as the specification describes them.
	pub raw: [u8; 4],
	pub version: MpegVersion,
	/// 1, 2 or 3.
	pub layer: u8,
	pub has_crc: bool,
	pub bitrate_kbps: u32,
	pub sample_rate: u32,
	pub padding: bool,
	pub channel_mode: ChannelMode,
	/// Joint-stereo mode extension bits (intensity / mid-side for Layer III).
	pub mode_extension: u8,
	/// Total frame length in bytes, including this header and padding.
	pub frame_size: usize,
	/// PCM samples per channel this frame decodes to (384, 576 or 1152).
	pub samples: u32,
}

const BITRATES_KBPS: [[[u16; 15]; 3]; 2] = [
	// MPEG-2 / 2.5: Layer I, II, III
	[
		[0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
		[0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
		[0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
	],
	// MPEG-1: Layer I, II, III
	[
		[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
		[0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
		[0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
	],
];

const SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];

impl FrameHeader {
	/// Parses four bytes as a frame header.
	///
	/// Returns `None` for anything that is not a usable header: missing sync,
	/// reserved version/layer/bitrate/sample-rate values, the reserved
	/// emphasis value, and free-format bitrate (index 0), which needs
	/// out-of-band frame-size detection and is not supported.
	pub fn parse(bytes: &[u8]) -> Option<Self> {
		let raw: [u8; 4] = bytes.get(..HEADER_SIZE)?.try_into().ok()?;
		// 11-bit sync word.
		if raw[0] != 0xFF || raw[1] & 0xE0 != 0xE0 {
			return None;
		}
		let version = match (raw[1] >> 3) & 0b11 {
			0b00 => MpegVersion::Mpeg25,
			0b10 => MpegVersion::Mpeg2,
			0b11 => MpegVersion::Mpeg1,
			_ => return None,
		};
		let layer = match (raw[1] >> 1) & 0b11 {
			0b01 => 3,
			0b10 => 2,
			0b11 => 1,
			_ => return None,
		};
		let bitrate_index = usize::from(raw[2] >> 4);
		if bitrate_index == 0 || bitrate_index == 15 {
			return None;
		}
		let sample_rate_index = usize::from((raw[2] >> 2) & 0b11);
		if sample_rate_index == 3 || raw[3] & 0b11 == 0b10 {
			return None;
		}

		let mpeg1 = version == MpegVersion::Mpeg1;
		let bitrate_kbps =
			u32::from(BITRATES_KBPS[usize::from(mpeg1)][usize::from(layer - 1)][bitrate_index]);
		let sample_rate = match version {
			MpegVersion::Mpeg1 => SAMPLE_RATES[sample_rate_index],
			MpegVersion::Mpeg2 => SAMPLE_RATES[sample_rate_index] / 2,
			MpegVersion::Mpeg25 => SAMPLE_RATES[sample_rate_index] / 4,
		};
		let samples: u32 = match (layer, mpeg1) {
			(1, _) => 384,
			(2, _) | (3, true) => 1152,
			_ => 576,
		};
		let padding = raw[2] & 0b10 != 0;
		// bytes = samples/8 * bitrate / sample_rate (+ padding slot).
		let frame_size = if layer == 1 {
			(12 * bitrate_kbps * 1000 / sample_rate + u32::from(padding)) * 4
		} else {
			samples * bitrate_kbps * 125 / sample_rate + u32::from(padding)
		} as usize;

		let channel_mode = match raw[3] >> 6 {
			0 => ChannelMode::Stereo,
			1 => ChannelMode::JointStereo,
			2 => ChannelMode::DualChannel,
			_ => ChannelMode::Mono,
		};

		Some(Self {
			raw,
			version,
			layer,
			has_crc: raw[1] & 1 == 0,
			bitrate_kbps,
			sample_rate,
			padding,
			channel_mode,
			mode_extension: (raw[3] >> 4) & 0b11,
			frame_size,
			samples,
		})
	}

	pub fn channels(&self) -> u8 {
		if self.channel_mode == ChannelMode::Mono { 1 } else { 2 }
	}

	pub fn is_mpeg1(&self) -> bool {
		self.version == MpegVersion::Mpeg1
	}

	/// Size of the Layer III side information that follows the header (and
	/// optional CRC).
	pub fn side_info_size(&self) -> usize {
		match (self.is_mpeg1(), self.channels()) {
			(true, 1) => 17,
			(true, _) => 32,
			(false, 1) => 9,
			(false, _) => 17,
		}
	}

	/// FFmpeg-style codec name for this layer.
	pub fn codec_name(&self) -> &'static str {
		match self.layer {
			1 => "mp1",
			2 => "mp2",
			_ => "mp3",
		}
	}

	/// `true` when `other` can belong to the same elementary stream: same
	/// version, layer and sample rate (the fields that may never change
	/// mid-stream). Used to reject false sync words when scanning.
	pub fn is_compatible(&self, other: &FrameHeader) -> bool {
		self.version == other.version
			&& self.layer == other.layer
			&& self.sample_rate == other.sample_rate
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_common_mpeg1_layer3_header() {
		// 128 kbit/s, 44.1 kHz, joint stereo, no CRC, no padding.
		let h = FrameHeader::parse(&[0xFF, 0xFB, 0x90, 0x44]).unwrap();
		assert_eq!(h.version, MpegVersion::Mpeg1);
		assert_eq!(h.layer, 3);
		assert_eq!(h.bitrate_kbps, 128);
		assert_eq!(h.sample_rate, 44_100);
		assert_eq!(h.frame_size, 417);
		assert_eq!(h.samples, 1152);
		assert_eq!(h.channel_mode, ChannelMode::JointStereo);
	}

	#[test]
	fn rejects_reserved_and_free_format() {
		assert!(FrameHeader::parse(&[0xFF, 0xFB, 0x00, 0x44]).is_none()); // free format
		assert!(FrameHeader::parse(&[0xFF, 0xFB, 0xF0, 0x44]).is_none()); // bad bitrate
		assert!(FrameHeader::parse(&[0xFF, 0xFB, 0x9C, 0x44]).is_none()); // bad sample rate
		assert!(FrameHeader::parse(&[0xFF, 0xE9, 0x90, 0x44]).is_none()); // reserved layer
		assert!(FrameHeader::parse(&[0xFF, 0xFB]).is_none()); // too short
	}
}
