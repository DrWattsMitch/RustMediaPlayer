//! PCM codecs (FFmpeg: `libavcodec/pcm.c`).
//!
//! PCM is "uncompressed", but the stored layout still differs from what
//! downstream code wants: 24-bit samples are packed into 3 bytes, some
//! streams are big-endian, 8-bit WAV is unsigned, and A-law / µ-law
//! (ITU-T G.711) are companded. The decoder turns each layout into a native
//! [`SampleFormat`], exactly as FFmpeg does:
//!
//! | Codec ID | Decodes to |
//! | --- | --- |
//! | `pcm_u8`, `pcm_s8` | `U8` |
//! | `pcm_s16le`, `pcm_s16be`, `pcm_alaw`, `pcm_mulaw` | `S16` |
//! | `pcm_s24le`, `pcm_s24be`, `pcm_s32le`, `pcm_s32be` | `S32` (24-bit values in the top bits) |
//! | `pcm_s64le`, `pcm_s64be` | `S64` |
//! | `pcm_f32le`, `pcm_f32be` | `F32` |
//! | `pcm_f64le`, `pcm_f64be` | `F64` |
//!
//! The encoder goes the other way, and accepts frames in any sample format
//! (see [`convert`]).

pub mod convert;
pub mod decoder;
pub mod encoder;
pub mod g711;

pub use decoder::PcmDecoder;
pub use encoder::PcmEncoder;

use crate::core::SampleFormat;

/// One stored PCM layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmCodec {
	U8,
	S8,
	S16Le,
	S16Be,
	S24Le,
	S24Be,
	S32Le,
	S32Be,
	S64Le,
	S64Be,
	F32Le,
	F32Be,
	F64Le,
	F64Be,
	ALaw,
	MuLaw,
}

impl PcmCodec {
	pub const ALL: [PcmCodec; 16] = [
		PcmCodec::U8,
		PcmCodec::S8,
		PcmCodec::S16Le,
		PcmCodec::S16Be,
		PcmCodec::S24Le,
		PcmCodec::S24Be,
		PcmCodec::S32Le,
		PcmCodec::S32Be,
		PcmCodec::S64Le,
		PcmCodec::S64Be,
		PcmCodec::F32Le,
		PcmCodec::F32Be,
		PcmCodec::F64Le,
		PcmCodec::F64Be,
		PcmCodec::ALaw,
		PcmCodec::MuLaw,
	];

	/// FFmpeg codec name.
	pub const fn id(self) -> &'static str {
		match self {
			PcmCodec::U8 => "pcm_u8",
			PcmCodec::S8 => "pcm_s8",
			PcmCodec::S16Le => "pcm_s16le",
			PcmCodec::S16Be => "pcm_s16be",
			PcmCodec::S24Le => "pcm_s24le",
			PcmCodec::S24Be => "pcm_s24be",
			PcmCodec::S32Le => "pcm_s32le",
			PcmCodec::S32Be => "pcm_s32be",
			PcmCodec::S64Le => "pcm_s64le",
			PcmCodec::S64Be => "pcm_s64be",
			PcmCodec::F32Le => "pcm_f32le",
			PcmCodec::F32Be => "pcm_f32be",
			PcmCodec::F64Le => "pcm_f64le",
			PcmCodec::F64Be => "pcm_f64be",
			PcmCodec::ALaw => "pcm_alaw",
			PcmCodec::MuLaw => "pcm_mulaw",
		}
	}

	/// Parses a codec ID. The WAV muxer's short aliases (`pcm_s16`, ...) are
	/// accepted as little-endian.
	pub fn from_id(id: &str) -> Option<Self> {
		let id = match id {
			"pcm_s16" => "pcm_s16le",
			"pcm_s24" => "pcm_s24le",
			"pcm_s32" => "pcm_s32le",
			"pcm_s64" => "pcm_s64le",
			"pcm_f32" => "pcm_f32le",
			"pcm_f64" => "pcm_f64le",
			other => other,
		};
		Self::ALL.into_iter().find(|c| c.id() == id)
	}

	/// Bytes per stored sample.
	pub const fn coded_bytes(self) -> usize {
		match self {
			PcmCodec::U8 | PcmCodec::S8 | PcmCodec::ALaw | PcmCodec::MuLaw => 1,
			PcmCodec::S16Le | PcmCodec::S16Be => 2,
			PcmCodec::S24Le | PcmCodec::S24Be => 3,
			PcmCodec::S32Le | PcmCodec::S32Be | PcmCodec::F32Le | PcmCodec::F32Be => 4,
			PcmCodec::S64Le | PcmCodec::S64Be | PcmCodec::F64Le | PcmCodec::F64Be => 8,
		}
	}

	/// Sample format the decoder produces.
	pub const fn sample_format(self) -> SampleFormat {
		match self {
			PcmCodec::U8 | PcmCodec::S8 => SampleFormat::U8,
			PcmCodec::S16Le | PcmCodec::S16Be | PcmCodec::ALaw | PcmCodec::MuLaw => SampleFormat::S16,
			PcmCodec::S24Le | PcmCodec::S24Be | PcmCodec::S32Le | PcmCodec::S32Be => SampleFormat::S32,
			PcmCodec::S64Le | PcmCodec::S64Be => SampleFormat::S64,
			PcmCodec::F32Le | PcmCodec::F32Be => SampleFormat::F32,
			PcmCodec::F64Le | PcmCodec::F64Be => SampleFormat::F64,
		}
	}
}
