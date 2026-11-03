//! Xing / Info / VBRI header frames.
//!
//! Encoders store stream-level info in the first MPEG frame (whose audio is
//! silent): total frame and byte counts (for exact duration), a 100-entry
//! seek table (TOC) and, in the LAME extension, encoder delay/padding for
//! gapless playback. Mirrors FFmpeg's `mp3_parse_info_tag` /
//! `mp3_parse_vbri_tag`, and `mp3enc.c` for writing.

use crate::codecs::mp3::{FrameHeader, header::HEADER_SIZE};
use crate::io::ByteCursor;

pub const TOC_SIZE: usize = 100;
const XING_FLAG_FRAMES: u32 = 0x1;
const XING_FLAG_BYTES: u32 = 0x2;
const XING_FLAG_TOC: u32 = 0x4;
const XING_FLAG_QUALITY: u32 = 0x8;
/// VBRI always sits 32 bytes after the frame header.
const VBRI_OFFSET: usize = HEADER_SIZE + 32;
/// Size of the LAME extension that may follow the Xing fields.
pub const LAME_TAG_SIZE: usize = 36;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfoKind {
	/// "Xing": variable bitrate.
	Xing,
	/// "Info": constant bitrate (same layout as Xing).
	Info,
	/// Fraunhofer "VBRI".
	Vbri,
}

#[derive(Debug, Clone)]
pub struct InfoTag {
	pub kind: InfoKind,
	/// Audio frames in the stream, excluding this info frame.
	pub frames: Option<u32>,
	/// Stream size in bytes (Xing: including the info frame).
	pub bytes: Option<u32>,
	/// Seek table: `toc[i]` = byte position of `i`% of the duration, in 1/256
	/// units of `bytes`.
	pub toc: Option<[u8; TOC_SIZE]>,
	/// LAME gapless info, in samples.
	pub encoder_delay: Option<u32>,
	pub encoder_padding: Option<u32>,
	/// The raw 36-byte LAME extension, kept so a remux can reproduce it.
	pub lame_tag: Option<[u8; LAME_TAG_SIZE]>,
}

/// Looks for an info tag inside `frame` (a complete first frame).
pub fn parse(frame: &[u8], header: &FrameHeader) -> Option<InfoTag> {
	parse_xing(frame, header).or_else(|| parse_vbri(frame))
}

fn parse_xing(frame: &[u8], header: &FrameHeader) -> Option<InfoTag> {
	let offset = HEADER_SIZE + header.side_info_size();
	let mut c = ByteCursor::new(frame.get(offset..)?);
	let kind = match &c.read_fourcc().ok()? {
		b"Xing" => InfoKind::Xing,
		b"Info" => InfoKind::Info,
		_ => return None,
	};
	let flags = c.read_u32().ok()?;
	let frames = if flags & XING_FLAG_FRAMES != 0 { Some(c.read_u32().ok()?) } else { None };
	let bytes = if flags & XING_FLAG_BYTES != 0 { Some(c.read_u32().ok()?) } else { None };
	let toc = if flags & XING_FLAG_TOC != 0 { Some(c.read_array::<TOC_SIZE>().ok()?) } else { None };
	if flags & XING_FLAG_QUALITY != 0 {
		c.skip(4).ok()?;
	}

	// Optional LAME extension: 9-byte encoder string, then fixed fields; the
	// 24-bit delay/padding pair sits 21 bytes into the extension.
	let (mut encoder_delay, mut encoder_padding, mut lame_tag) = (None, None, None);
	let lame = c.rest();
	if lame.len() >= LAME_TAG_SIZE && [&b"LAME"[..], b"Lavf", b"Lavc"].contains(&&lame[..4]) {
		let d = &lame[21..24];
		encoder_delay = Some((u32::from(d[0]) << 4) | (u32::from(d[1]) >> 4));
		encoder_padding = Some((u32::from(d[1] & 0x0F) << 8) | u32::from(d[2]));
		let mut tag = [0u8; LAME_TAG_SIZE];
		tag.copy_from_slice(&lame[..LAME_TAG_SIZE]);
		lame_tag = Some(tag);
	}

	Some(InfoTag {
		kind,
		frames: frames.filter(|&f| f > 0),
		bytes: bytes.filter(|&b| b > 0),
		toc,
		encoder_delay,
		encoder_padding,
		lame_tag,
	})
}

fn parse_vbri(frame: &[u8]) -> Option<InfoTag> {
	let mut c = ByteCursor::new(frame.get(VBRI_OFFSET..)?);
	if &c.read_fourcc().ok()? != b"VBRI" {
		return None;
	}
	let _version = c.read_u16().ok()?;
	let delay = c.read_u16().ok()?;
	let _quality = c.read_u16().ok()?;
	let bytes = c.read_u32().ok()?;
	let frames = c.read_u32().ok()?;
	Some(InfoTag {
		kind: InfoKind::Vbri,
		frames: Some(frames).filter(|&f| f > 0),
		bytes: Some(bytes).filter(|&b| b > 0),
		toc: None,
		encoder_delay: Some(u32::from(delay)),
		encoder_padding: None,
		lame_tag: None,
	})
}

/// Builds the header of an info frame compatible with `first` (same version,
/// sample rate and channel mode), using the smallest bitrate whose frame can
/// hold a Xing tag with TOC (plus a LAME extension when `with_lame`).
/// Returns `None` for non-Layer III streams.
pub fn info_frame_header(first: &FrameHeader, with_lame: bool) -> Option<FrameHeader> {
	if first.layer != 3 {
		return None;
	}
	let lame = if with_lame { LAME_TAG_SIZE } else { 0 };
	let needed = HEADER_SIZE + first.side_info_size() + 4 + 4 + 4 + 4 + TOC_SIZE + lame;
	let mut raw = first.raw;
	raw[1] |= 0x01; // no CRC
	raw[3] &= 0xC0; // keep channel mode, clear extension/copyright/emphasis
	(1u8..=14).find_map(|bitrate_index| {
		raw[2] = (bitrate_index << 4) | (first.raw[2] & 0x0C); // keep rate, no padding
		FrameHeader::parse(&raw).filter(|h| h.frame_size >= needed)
	})
}

/// CRC-16/ARC (poly 0x8005 reflected, init 0), as used by the LAME tag.
fn crc16(data: &[u8]) -> u16 {
	let mut crc = 0u16;
	for &byte in data {
		crc ^= u16::from(byte);
		for _ in 0..8 {
			crc = if crc & 1 != 0 { (crc >> 1) ^ 0xA001 } else { crc >> 1 };
		}
	}
	crc
}

/// Serialises a complete Xing/Info frame, optionally followed by a LAME
/// extension (whose trailing tag CRC is recomputed for the new frame).
pub fn build_info_frame(
	header: &FrameHeader,
	kind: InfoKind,
	frames: u32,
	bytes: u32,
	toc: &[u8; TOC_SIZE],
	lame_tag: Option<&[u8; LAME_TAG_SIZE]>,
) -> Vec<u8> {
	let mut frame = vec![0u8; header.frame_size];
	frame[..HEADER_SIZE].copy_from_slice(&header.raw);
	let mut tag = Vec::with_capacity(16 + TOC_SIZE);
	tag.extend_from_slice(if kind == InfoKind::Info { b"Info" } else { b"Xing" });
	tag.extend_from_slice(&(XING_FLAG_FRAMES | XING_FLAG_BYTES | XING_FLAG_TOC).to_be_bytes());
	tag.extend_from_slice(&frames.to_be_bytes());
	tag.extend_from_slice(&bytes.to_be_bytes());
	tag.extend_from_slice(toc);
	if let Some(lame) = lame_tag {
		tag.extend_from_slice(lame);
	}
	let offset = HEADER_SIZE + header.side_info_size();
	let end = (offset + tag.len()).min(frame.len());
	frame[offset..end].copy_from_slice(&tag[..end - offset]);
	if lame_tag.is_some() && end == offset + tag.len() {
		// The tag CRC covers every byte of the frame before it.
		let crc = crc16(&frame[..end - 2]);
		frame[end - 2..end].copy_from_slice(&crc.to_be_bytes());
	}
	frame
}

/// Byte offset for `percent` (0.0..=100.0) using a Xing TOC, with linear
/// interpolation between entries (FFmpeg's `mp3_seek` behaviour).
pub fn toc_offset(toc: &[u8; TOC_SIZE], bytes: u64, percent: f64) -> u64 {
	let p = percent.clamp(0.0, 99.999);
	let index = p as usize;
	let fa = f64::from(toc[index]);
	let fb = if index + 1 < TOC_SIZE { f64::from(toc[index + 1]) } else { 256.0 };
	let fx = fa + (fb - fa) * (p - index as f64);
	(fx / 256.0 * bytes as f64) as u64
}
