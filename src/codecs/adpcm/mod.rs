//! ADPCM decoders for WAV (FFmpeg: `libavcodec/adpcm.c`, `adpcm_data.c`).
//!
//! | Codec ID | WAV tag | Format |
//! | --- | --- | --- |
//! | `adpcm_ima_wav` | `0x0011` | IMA / DVI ADPCM, 4 bits per sample |
//! | `adpcm_ms` | `0x0002` | Microsoft ADPCM, 4 bits per sample, 1-2 channels |
//!
//! Both formats code audio in self-contained blocks (`nBlockAlign` bytes)
//! that start with a per-channel header holding the predictor state, so the
//! decoder needs no state between packets and seeking is exact at block
//! boundaries. The WAV demuxer returns one block per packet. A short final
//! block decodes to however many whole samples it holds, as in FFmpeg.
//!
//! Output is interleaved S16.

pub mod decoder;
pub mod ima;
pub mod ms;

pub use decoder::AdpcmDecoder;

/// The ADPCM variants RMP decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdpcmCodec {
	ImaWav,
	Ms,
}

impl AdpcmCodec {
	pub const fn id(self) -> &'static str {
		match self {
			AdpcmCodec::ImaWav => "adpcm_ima_wav",
			AdpcmCodec::Ms => "adpcm_ms",
		}
	}

	pub fn from_id(id: &str) -> Option<Self> {
		match id {
			"adpcm_ima_wav" => Some(AdpcmCodec::ImaWav),
			"adpcm_ms" => Some(AdpcmCodec::Ms),
			_ => None,
		}
	}

	/// WAVE format tag (`wFormatTag`).
	pub const fn wav_tag(self) -> u16 {
		match self {
			AdpcmCodec::ImaWav => 0x0011,
			AdpcmCodec::Ms => 0x0002,
		}
	}

	/// Smallest block that holds every channel's header.
	pub const fn header_bytes(self, channels: usize) -> usize {
		match self {
			AdpcmCodec::ImaWav => ima::HEADER_BYTES * channels,
			AdpcmCodec::Ms => ms::HEADER_BYTES * channels,
		}
	}

	/// Whether the codec supports this channel count.
	pub const fn supports_channels(self, channels: usize) -> bool {
		match self {
			AdpcmCodec::ImaWav => channels >= 1,
			AdpcmCodec::Ms => channels == 1 || channels == 2,
		}
	}

	/// Samples per channel decoded from a block of `bytes` bytes, or `None`
	/// if the block is too short to hold its headers.
	pub fn samples_in_block(self, bytes: usize, channels: usize) -> Option<usize> {
		if channels == 0 || bytes < self.header_bytes(channels) {
			return None;
		}
		let payload = bytes - self.header_bytes(channels);
		Some(match self {
			// One sample in the header, then 8 per 4-byte group per channel.
			AdpcmCodec::ImaWav => 1 + payload / (ima::GROUP_BYTES * channels) * ima::GROUP_SAMPLES,
			// Two samples in the header, then two nibbles per byte.
			AdpcmCodec::Ms => 2 + payload * 2 / channels,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn block_sample_counts_match_ffmpeg_encoders() {
		// ffmpeg -c:a adpcm_ima_wav: 1024 / 2048-byte blocks (mono / stereo).
		assert_eq!(AdpcmCodec::ImaWav.samples_in_block(1024, 1), Some(2041));
		assert_eq!(AdpcmCodec::ImaWav.samples_in_block(2048, 2), Some(2041));
		// ffmpeg -c:a adpcm_ms: 1024 / 2048-byte blocks.
		assert_eq!(AdpcmCodec::Ms.samples_in_block(1024, 1), Some(2036));
		assert_eq!(AdpcmCodec::Ms.samples_in_block(2048, 2), Some(2036));
		assert_eq!(AdpcmCodec::Ms.samples_in_block(13, 2), None);
		assert_eq!(AdpcmCodec::ImaWav.samples_in_block(4, 0), None);
	}
}
