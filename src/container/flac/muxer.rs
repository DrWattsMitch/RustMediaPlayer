//! Native FLAC muxer (`.flac`).
//!
//! Writes `"fLaC"`, STREAMINFO, a Vorbis comment block, PICTURE blocks for
//! attached images and a PADDING block, then the frames as they come. On
//! `finish`, STREAMINFO is rewritten with what was actually written: block
//! and frame size bounds, the total sample count and the MD5 of the decoded
//! audio, which the muxer computes by decoding each frame. That keeps the
//! header exact for stream copies and for RMP's own encoder alike (FFmpeg
//! instead has the encoder send updated extradata at the end).
//!
//! FFmpeg: `libavformat/flacenc.c`.

use crate::codecs::flac::decoder::decode_frame;
use crate::codecs::flac::md5::Md5;
use crate::codecs::flac::{FrameHeader, StreamInfo};
use crate::container::xiph;
use crate::core::{Metadata, Muxer, Packet, Stream};
use crate::io::Io;
use crate::message::{Message, Result};

/// Padding left after the metadata so tags can be edited in place
/// (FFmpeg's default, 8192 bytes).
const PADDING: usize = 8192;
const VENDOR: &str = "RMP";

pub struct FlacMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	info: Option<StreamInfo>,
	header_written: bool,
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

impl FlacMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self {
			io,
			metadata: Metadata::default(),
			info: None,
			header_written: false,
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

	fn write_header(&mut self) -> Result<()> {
		let info = self.info.ok_or(Message::Container("flac muxer has no stream"))?;
		let mut out = b"fLaC".to_vec();
		out.extend(info.to_block(false));
		let comments = xiph::write_comments(&self.metadata, VENDOR);
		let push_block = |out: &mut Vec<u8>, kind: u8, body: &[u8]| -> Result<()> {
			let len = u32::try_from(body.len()).ok().filter(|&l| l < 1 << 24);
			let len = len.ok_or(Message::Container("flac metadata block too large"))?;
			out.push(kind);
			out.extend_from_slice(&len.to_be_bytes()[1..]);
			out.extend_from_slice(body);
			Ok(())
		};
		push_block(&mut out, 4, &comments)?;
		for image in &self.metadata.images {
			push_block(&mut out, 6, &xiph::write_picture(image))?;
		}
		push_block(&mut out, 0x80 | 1, &[0u8; PADDING])?;
		self.io.write_all(&out)?;
		self.header_written = true;
		Ok(())
	}

	/// Appends the decoded samples of a frame to the MD5, in the layout the
	/// FLAC specification defines (interleaved, little-endian, signed,
	/// `ceil(bits / 8)` bytes per sample).
	fn hash(&mut self, data: &[u8], info: &StreamInfo) {
		let Ok(frame) = decode_frame(data, Some(info)) else {
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
}

impl Muxer for FlacMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.info.is_some() {
			return Err(Message::Container("flac files hold a single stream"));
		}
		if stream.parameters.codec.id != "flac" {
			return Err(Message::Container("flac muxer accepts only flac audio"));
		}
		let info = StreamInfo::from_extradata(&stream.parameters.extradata)
			.ok_or(Message::Container("flac muxer: stream has no STREAMINFO extradata"))?;
		self.info = Some(info);
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if !self.header_written {
			self.write_header()?;
		}
		let info = self.info.ok_or(Message::Container("flac muxer has no stream"))?;
		let header =
			FrameHeader::parse(&packet.data).ok_or(Message::Container("packet is not a flac frame"))?;
		let size = u32::try_from(packet.data.len()).unwrap_or(u32::MAX);
		let block = u16::try_from(header.block_size).unwrap_or(u16::MAX);
		self.hash(&packet.data, &info);
		self.io.write_all(&packet.data)?;
		if self.frames > 0 {
			// STREAMINFO's minimum block size excludes the last block.
			self.min_block = self.min_block.min(self.last_block);
		}
		self.max_block = self.max_block.max(block);
		self.last_block = block;
		self.min_frame = self.min_frame.min(size);
		self.max_frame = self.max_frame.max(size);
		self.total_samples += u64::from(header.block_size);
		self.frames += 1;
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		if !self.header_written {
			self.write_header()?;
		}
		let Some(mut info) = self.info else { return Ok(()) };
		if self.frames > 0 {
			info.min_block_size = if self.frames == 1 { self.last_block } else { self.min_block };
			info.max_block_size = self.max_block;
			info.min_frame_size = self.min_frame.min(0xFF_FFFF);
			info.max_frame_size = self.max_frame.min(0xFF_FFFF);
			info.total_samples = self.total_samples.min(0xF_FFFF_FFFF);
			info.md5 = if self.md5_valid { std::mem::take(&mut self.md5).finish() } else { [0; 16] };
		}
		// STREAMINFO body starts after "fLaC" and its block header. Outputs
		// that can't seek keep the provisional header.
		if self.io.seek(8).is_ok() {
			self.io.write_all(&info.to_bytes())?;
		}
		Ok(())
	}
}
