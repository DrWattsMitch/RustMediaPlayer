//! Exact STREAMINFO for a FLAC stream that is being muxed.
//!
//! Encoders (RMP's and FFmpeg's) only know the final block and frame size
//! bounds, the sample count and the MD5 of the audio once the last frame is
//! out, so the STREAMINFO in the extradata a muxer starts with is
//! provisional. Every container that carries FLAC (native FLAC, Matroska,
//! Ogg) feeds the frames it writes through [`FlacStats`] and rewrites
//! STREAMINFO with [`FlacStats::finish`] when the output can seek. The MD5
//! is computed by decoding each frame, so stream copies and fresh encodes
//! both get an exact header.

use crate::codecs::flac::decoder::decode_frame;
use crate::codecs::flac::md5::Md5;
use crate::codecs::flac::{FrameHeader, StreamInfo};

#[derive(Debug, Clone)]
pub struct FlacStats {
	info: StreamInfo,
	md5: Md5,
	md5_valid: bool,
	total_samples: u64,
	min_frame: u32,
	max_frame: u32,
	min_block: u16,
	max_block: u16,
	last_block: u16,
	frames: u64,
}

impl FlacStats {
	pub fn new(info: StreamInfo) -> Self {
		Self {
			info,
			md5: Md5::new(),
			md5_valid: true,
			total_samples: 0,
			min_frame: u32::MAX,
			max_frame: 0,
			min_block: u16::MAX,
			max_block: 0,
			last_block: 0,
			frames: 0,
		}
	}

	/// The STREAMINFO the stream started with.
	pub fn initial(&self) -> StreamInfo {
		self.info
	}

	/// Accounts for one frame. Returns `false` when `data` is not a FLAC
	/// frame (the frame is not counted).
	pub fn add(&mut self, data: &[u8]) -> bool {
		let Some(header) = FrameHeader::parse(data) else { return false };
		let size = u32::try_from(data.len()).unwrap_or(u32::MAX);
		let block = u16::try_from(header.block_size).unwrap_or(u16::MAX);
		self.hash(data);
		if self.frames > 0 {
			// STREAMINFO's minimum block size excludes the last block.
			self.min_block = self.min_block.min(self.last_block);
		}
		self.max_block = self.max_block.max(block);
		self.last_block = block;
		self.min_frame = self.min_frame.min(size);
		self.max_frame = self.max_frame.max(size);
		self.total_samples = self.total_samples.saturating_add(u64::from(header.block_size));
		self.frames += 1;
		true
	}

	/// Appends the decoded samples of a frame to the MD5, in the layout the
	/// FLAC specification defines (interleaved, little-endian, signed,
	/// `ceil(bits / 8)` bytes per sample).
	fn hash(&mut self, data: &[u8]) {
		if !self.md5_valid {
			return;
		}
		let Ok(frame) = decode_frame(data, Some(&self.info)) else {
			self.md5_valid = false;
			return;
		};
		let bytes = usize::from(frame.bits).div_ceil(8);
		let block = frame.channels.first().map_or(0, Vec::len);
		let mut out = Vec::with_capacity(block * frame.channels.len() * bytes);
		for i in 0..block {
			for ch in &frame.channels {
				out.extend_from_slice(&ch[i].to_le_bytes()[..bytes]);
			}
		}
		self.md5.update(&out);
	}

	/// The STREAMINFO describing every frame added so far. Without frames,
	/// the initial one.
	pub fn finish(&mut self) -> StreamInfo {
		let mut info = self.info;
		if self.frames > 0 {
			info.min_block_size = if self.frames == 1 { self.last_block } else { self.min_block };
			info.max_block_size = self.max_block;
			info.min_frame_size = self.min_frame.min(0xFF_FFFF);
			info.max_frame_size = self.max_frame.min(0xFF_FFFF);
			info.total_samples = self.total_samples.min(0xF_FFFF_FFFF);
			info.md5 = if self.md5_valid { std::mem::take(&mut self.md5).finish() } else { [0; 16] };
		}
		info
	}
}
