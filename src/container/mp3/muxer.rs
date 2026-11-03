//! MP3 muxer: `[ID3v2.4 tag] [Xing/Info frame] frames...`
//!
//! Packets must already contain MPEG audio frames (RMP has no MP3 encoder).
//! As in FFmpeg's `mp3enc.c`, a placeholder info frame is written up front
//! and filled in by [`Muxer::finish`] with the frame count, byte count and a
//! seek table, which gives other players exact duration and VBR seeking.

use super::demuxer::LAME_TAG_KEY;
use super::id3;
use super::xing::{self, InfoKind, LAME_TAG_SIZE, TOC_SIZE};
use crate::codecs::mp3::FrameHeader;
use crate::codecs::mp3::header::HEADER_SIZE;
use crate::core::{Metadata, Muxer, Packet, RawValue, Stream, StreamKind};
use crate::io::Io;
use crate::message::{Message, Result};

pub struct Mp3Muxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	stream_added: bool,
	started: bool,
	/// Header of the first frame: later frames must be compatible.
	first: Option<FrameHeader>,
	/// Info frame placeholder: (file position, header).
	info: Option<(u64, FrameHeader)>,
	/// Byte offset of every frame, relative to the first audio frame.
	frame_offsets: Vec<u64>,
	audio_bytes: u64,
	constant_bitrate: bool,
	/// LAME extension from the source stream (gapless info), if any.
	lame_tag: Option<[u8; LAME_TAG_SIZE]>,
}

impl Mp3Muxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self {
			io,
			metadata: Metadata::default(),
			stream_added: false,
			started: false,
			first: None,
			info: None,
			frame_offsets: Vec::new(),
			audio_bytes: 0,
			constant_bitrate: true,
			lame_tag: None,
		}
	}

	fn start(&mut self, first: &FrameHeader) -> Result<()> {
		let mut position = 0u64;
		if let Some(tag) = id3::build_v2(&self.metadata)? {
			self.io.write_all(&tag)?;
			position += tag.len() as u64;
		}
		if let Some(header) = xing::info_frame_header(first, self.lame_tag.is_some()) {
			// Written for real in `finish`; zeros decode as silence.
			self.io.write_all(&vec![0u8; header.frame_size])?;
			self.info = Some((position, header));
		}
		self.first = Some(*first);
		self.started = true;
		Ok(())
	}

	fn build_toc(&self, info_size: u64) -> [u8; TOC_SIZE] {
		let mut toc = [0u8; TOC_SIZE];
		let frames = self.frame_offsets.len();
		let total = (info_size + self.audio_bytes).max(1);
		if frames == 0 {
			return toc;
		}
		for (i, slot) in toc.iter_mut().enumerate() {
			let frame = (i * frames / TOC_SIZE).min(frames - 1);
			let offset = info_size + self.frame_offsets[frame];
			*slot = (u128::from(offset) * 256 / u128::from(total)).min(255) as u8;
		}
		toc
	}
}

impl Muxer for Mp3Muxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		if self.started {
			return Err(Message::Container("mp3 metadata must be set before writing"));
		}
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.stream_added {
			return Err(Message::Container("mp3 supports only one stream"));
		}
		if stream.kind != StreamKind::Audio
			|| !matches!(stream.parameters.codec.id.as_str(), "mp3" | "mp2" | "mp1")
		{
			return Err(Message::Container("mp3 muxer only accepts mpeg audio streams"));
		}
		if let Some(RawValue::Bytes(tag)) = stream.metadata.raw.get(LAME_TAG_KEY) {
			self.lame_tag = <[u8; LAME_TAG_SIZE]>::try_from(tag.as_slice()).ok();
		}
		self.stream_added = true;
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		// Validate every frame in the packet before writing anything.
		let mut at = 0usize;
		let mut headers = Vec::new();
		while at < packet.data.len() {
			let header = FrameHeader::parse(&packet.data[at..])
				.filter(|h| at + h.frame_size <= packet.data.len())
				.ok_or(Message::Container("mp3 muxer: packet is not a sequence of mpeg audio frames"))?;
			if let Some(first) = self.first.as_ref().or(headers.first()) {
				if !first.is_compatible(&header) {
					return Err(Message::Container("mp3 muxer: incompatible frame in stream"));
				}
			}
			headers.push(header);
			at += header.frame_size;
		}
		let Some(&first_in_packet) = headers.first() else { return Ok(()) };

		if !self.started {
			self.start(&first_in_packet)?;
		}
		let reference_bitrate = self.first.map(|f| f.bitrate_kbps);
		let mut offset = self.audio_bytes;
		for header in &headers {
			self.frame_offsets.push(offset);
			offset += header.frame_size as u64;
			if Some(header.bitrate_kbps) != reference_bitrate {
				self.constant_bitrate = false;
			}
		}
		self.io.write_all(&packet.data)?;
		self.audio_bytes = offset;
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		let Some((position, header)) = self.info else { return Ok(()) };
		let info_size = header.frame_size as u64;
		let frames = u32::try_from(self.frame_offsets.len()).unwrap_or(u32::MAX);
		let bytes = u32::try_from(info_size + self.audio_bytes).unwrap_or(u32::MAX);
		let kind = if self.constant_bitrate { InfoKind::Info } else { InfoKind::Xing };
		let toc = self.build_toc(info_size);
		let frame = xing::build_info_frame(&header, kind, frames, bytes, &toc, self.lame_tag.as_ref());
		debug_assert!(frame.len() >= HEADER_SIZE);
		self.io.seek(position)?;
		self.io.write_all(&frame)?;
		Ok(())
	}
}
