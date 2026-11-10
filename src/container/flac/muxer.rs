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

use super::stats::FlacStats;
use crate::codecs::flac::StreamInfo;
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
	stats: Option<FlacStats>,
	header_written: bool,
}

impl FlacMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self { io, metadata: Metadata::default(), stats: None, header_written: false }
	}

	fn write_header(&mut self) -> Result<()> {
		let info: StreamInfo =
			self.stats.as_ref().ok_or(Message::Container("flac muxer has no stream"))?.initial();
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
		if self.stats.is_some() {
			return Err(Message::Container("flac files hold a single stream"));
		}
		if stream.parameters.codec.id != "flac" {
			return Err(Message::Container("flac muxer accepts only flac audio"));
		}
		let info = StreamInfo::from_extradata(&stream.parameters.extradata)
			.ok_or(Message::Container("flac muxer: stream has no STREAMINFO extradata"))?;
		self.stats = Some(FlacStats::new(info));
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if !self.header_written {
			self.write_header()?;
		}
		let stats = self.stats.as_mut().ok_or(Message::Container("flac muxer has no stream"))?;
		if !stats.add(&packet.data) {
			return Err(Message::Container("packet is not a flac frame"));
		}
		self.io.write_all(&packet.data)?;
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		if !self.header_written {
			self.write_header()?;
		}
		let Some(stats) = self.stats.as_mut() else { return Ok(()) };
		let info = stats.finish();
		// STREAMINFO body starts after "fLaC" and its block header. Outputs
		// that can't seek keep the provisional header.
		if self.io.seek(8).is_ok() {
			self.io.write_all(&info.to_bytes())?;
		}
		Ok(())
	}
}
