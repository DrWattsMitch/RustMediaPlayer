//! EBML (RFC 8794) primitives: variable-length integers, element headers and
//! typed element values.
//!
//! Matroska is a tree of EBML elements. Each element is an ID (1-4 bytes,
//! length marker kept, as the Matroska specification writes them), a size
//! (1-8 bytes, marker stripped; all value bits set means "unknown") and a
//! body. Reading here works on in-memory slices, so every access is
//! bounds-checked; the demuxer uses [`parse_header`] on its read-ahead window
//! for the streamed part of the file. FFmpeg: `libavformat/matroskadec.c`
//! (`ebml_read_num`, `ebml_read_uint`, `ebml_read_float`, ...).

use crate::io::ByteCursor;
use crate::message::{Message, Result};

const INVALID: Message = Message::Container("invalid ebml element");

/// Longest element header: a 4-byte ID plus an 8-byte size.
pub const MAX_HEADER_LEN: usize = 12;

/// Element IDs used by the demuxer (Matroska specification, `ebml_matroska.xml`).
pub mod id {
	pub const EBML: u32 = 0x1A45_DFA3;
	pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
	pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
	pub const DOC_TYPE: u32 = 0x4282;
	pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;
	pub const VOID: u32 = 0xEC;
	pub const CRC32: u32 = 0xBF;

	pub const SEGMENT: u32 = 0x1853_8067;
	pub const SEEK_HEAD: u32 = 0x114D_9B74;
	pub const SEEK: u32 = 0x4DBB;
	pub const SEEK_ID: u32 = 0x53AB;
	pub const SEEK_POSITION: u32 = 0x53AC;
	pub const INFO: u32 = 0x1549_A966;
	pub const TIMESTAMP_SCALE: u32 = 0x2AD7B1;
	pub const DURATION: u32 = 0x4489;
	pub const TITLE: u32 = 0x7BA9;
	pub const MUXING_APP: u32 = 0x4D80;
	pub const WRITING_APP: u32 = 0x5741;
	pub const DATE_UTC: u32 = 0x4461;
	pub const CLUSTER: u32 = 0x1F43_B675;
	pub const TRACKS: u32 = 0x1654_AE6B;
	pub const CUES: u32 = 0x1C53_BB6B;
	pub const TAGS: u32 = 0x1254_C367;
	pub const CHAPTERS: u32 = 0x1043_A770;
	pub const ATTACHMENTS: u32 = 0x1941_A469;

	// Cluster children.
	pub const TIMESTAMP: u32 = 0xE7;
	pub const SIMPLE_BLOCK: u32 = 0xA3;
	pub const BLOCK_GROUP: u32 = 0xA0;
	pub const BLOCK: u32 = 0xA1;
	pub const BLOCK_DURATION: u32 = 0x9B;
	pub const REFERENCE_BLOCK: u32 = 0xFB;
	pub const DISCARD_PADDING: u32 = 0x75A2;
	pub const POSITION: u32 = 0xA7;
	pub const PREV_SIZE: u32 = 0xAB;
	pub const SILENT_TRACKS: u32 = 0x5854;
	pub const ENCRYPTED_BLOCK: u32 = 0xAF;
	pub const BLOCK_ADDITIONS: u32 = 0x75A1;

	// Tracks.
	pub const TRACK_ENTRY: u32 = 0xAE;
	pub const TRACK_NUMBER: u32 = 0xD7;
	pub const TRACK_UID: u32 = 0x73C5;
	pub const TRACK_TYPE: u32 = 0x83;
	pub const FLAG_ENABLED: u32 = 0xB9;
	pub const FLAG_DEFAULT: u32 = 0x88;
	pub const FLAG_FORCED: u32 = 0x55AA;
	pub const DEFAULT_DURATION: u32 = 0x23E383;
	pub const NAME: u32 = 0x536E;
	pub const LANGUAGE: u32 = 0x22B59C;
	pub const LANGUAGE_BCP47: u32 = 0x22B59D;
	pub const CODEC_ID: u32 = 0x86;
	pub const CODEC_PRIVATE: u32 = 0x63A2;
	pub const CODEC_DELAY: u32 = 0x56AA;
	pub const SEEK_PRE_ROLL: u32 = 0x56BB;
	pub const VIDEO: u32 = 0xE0;
	pub const PIXEL_WIDTH: u32 = 0xB0;
	pub const PIXEL_HEIGHT: u32 = 0xBA;
	pub const DISPLAY_WIDTH: u32 = 0x54B0;
	pub const DISPLAY_HEIGHT: u32 = 0x54BA;
	pub const AUDIO: u32 = 0xE1;
	pub const SAMPLING_FREQUENCY: u32 = 0xB5;
	pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
	pub const CHANNELS: u32 = 0x9F;
	pub const BIT_DEPTH: u32 = 0x6264;
	pub const CONTENT_ENCODINGS: u32 = 0x6D80;
	pub const CONTENT_ENCODING: u32 = 0x6240;
	pub const CONTENT_ENCODING_SCOPE: u32 = 0x5032;
	pub const CONTENT_ENCODING_TYPE: u32 = 0x5033;
	pub const CONTENT_COMPRESSION: u32 = 0x5034;
	pub const CONTENT_COMP_ALGO: u32 = 0x4254;
	pub const CONTENT_COMP_SETTINGS: u32 = 0x4255;

	// Cues.
	pub const CUE_POINT: u32 = 0xBB;
	pub const CUE_TIME: u32 = 0xB3;
	pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
	pub const CUE_TRACK: u32 = 0xF7;
	pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
	pub const CUE_RELATIVE_POSITION: u32 = 0xF0;

	// Tags.
	pub const TAG: u32 = 0x7373;
	pub const TARGETS: u32 = 0x63C0;
	pub const TARGET_TYPE_VALUE: u32 = 0x68CA;
	pub const TAG_TRACK_UID: u32 = 0x63C5;
	pub const SIMPLE_TAG: u32 = 0x67C8;
	pub const TAG_NAME: u32 = 0x45A3;
	pub const TAG_STRING: u32 = 0x4487;

	// Attachments.
	pub const ATTACHED_FILE: u32 = 0x61A7;
	pub const FILE_DESCRIPTION: u32 = 0x467E;
	pub const FILE_NAME: u32 = 0x466E;
	pub const FILE_MEDIA_TYPE: u32 = 0x4660;
	pub const FILE_DATA: u32 = 0x465C;
}

/// IDs that may appear directly under `Segment`. Seeing one while inside an
/// unknown-size cluster ends that cluster.
pub fn is_top_level(id: u32) -> bool {
	matches!(
		id,
		id::SEEK_HEAD
			| id::INFO
			| id::CLUSTER
			| id::TRACKS
			| id::CUES
			| id::TAGS
			| id::CHAPTERS
			| id::ATTACHMENTS
			| id::SEGMENT
			| id::EBML
	)
}

/// An element header: ID, body size (`None` = unknown) and header length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
	pub id: u32,
	pub size: Option<u64>,
	pub len: usize,
}

/// Length of a vint from its first byte (1-8), or `None` for `0x00`.
fn vint_len(first: u8) -> Option<usize> {
	(first != 0).then(|| first.leading_zeros() as usize + 1)
}

/// Reads an element ID (marker bits kept). IDs are at most 4 bytes long.
pub fn read_id(c: &mut ByteCursor) -> Result<u32> {
	let first = c.read_u8()?;
	let len = vint_len(first).filter(|&l| l <= 4).ok_or(INVALID)?;
	let mut value = u32::from(first);
	for _ in 1..len {
		value = (value << 8) | u32::from(c.read_u8()?);
	}
	Ok(value)
}

/// Reads an unsigned vint (marker stripped). Returns the value and whether
/// every value bit was set (the "unknown size" encoding).
pub fn read_vint(c: &mut ByteCursor) -> Result<(u64, bool)> {
	let first = c.read_u8()?;
	let len = vint_len(first).ok_or(INVALID)?;
	let mut value = u64::from(first) & (0xFF >> len);
	for _ in 1..len {
		value = (value << 8) | u64::from(c.read_u8()?);
	}
	let all_ones = value == (1u64 << (7 * len)) - 1;
	Ok((value, all_ones))
}

/// Reads a signed vint as used by EBML lacing (value minus half the range).
pub fn read_signed_vint(c: &mut ByteCursor) -> Result<i64> {
	let start = c.position();
	let (value, _) = read_vint(c)?;
	let len = c.position() - start;
	let bias = (1i64 << (7 * len - 1)) - 1;
	Ok(value as i64 - bias)
}

/// Parses an element header from the start of `data`.
pub fn parse_header(data: &[u8]) -> Result<Header> {
	let mut c = ByteCursor::new(data);
	let id = read_id(&mut c)?;
	let (size, unknown) = read_vint(&mut c)?;
	Ok(Header { id, size: (!unknown).then_some(size), len: c.position() })
}

/// Iterates over the child elements of an in-memory master element.
///
/// A child whose size runs past the end of the parent (or is unknown) is
/// clamped to the remaining bytes, so a truncated file still yields what it
/// holds; a malformed header ends the iteration.
pub struct Children<'a> {
	cursor: ByteCursor<'a>,
}

impl<'a> Children<'a> {
	pub fn new(data: &'a [u8]) -> Self {
		Self { cursor: ByteCursor::new(data) }
	}
}

impl<'a> Iterator for Children<'a> {
	type Item = (u32, &'a [u8]);

	fn next(&mut self) -> Option<Self::Item> {
		if self.cursor.is_empty() {
			return None;
		}
		let header = parse_header(self.cursor.rest()).ok()?;
		self.cursor.skip(header.len).ok()?;
		let remaining = self.cursor.remaining();
		let len =
			header.size.map_or(remaining, |s| usize::try_from(s).unwrap_or(usize::MAX).min(remaining));
		let body = self.cursor.read_bytes(len).ok()?;
		Some((header.id, body))
	}
}

/// First child with the given ID.
pub fn find(data: &[u8], wanted: u32) -> Option<&[u8]> {
	Children::new(data).find(|(id, _)| *id == wanted).map(|(_, body)| body)
}

/// Big-endian unsigned integer element (0-8 bytes; empty means 0).
pub fn uint(body: &[u8]) -> Result<u64> {
	if body.len() > 8 {
		return Err(INVALID);
	}
	Ok(body.iter().fold(0u64, |acc, &b| (acc << 8) | u64::from(b)))
}

/// Big-endian two's complement signed integer element.
pub fn sint(body: &[u8]) -> Result<i64> {
	if body.is_empty() {
		return Ok(0);
	}
	let raw = uint(body)?;
	let shift = 64 - 8 * body.len() as u32;
	Ok(((raw << shift) as i64) >> shift)
}

/// IEEE float element (0, 4 or 8 bytes).
pub fn float(body: &[u8]) -> Result<f64> {
	match body.len() {
		0 => Ok(0.0),
		4 => Ok(f64::from(f32::from_bits(uint(body)? as u32))),
		8 => Ok(f64::from_bits(uint(body)?)),
		_ => Err(INVALID),
	}
}

/// String / UTF-8 element. Trailing NUL padding is removed; invalid UTF-8
/// is replaced rather than rejected (tags written by old muxers).
pub fn string(body: &[u8]) -> String {
	let end = body.iter().position(|&b| b == 0).unwrap_or(body.len());
	String::from_utf8_lossy(&body[..end]).into_owned()
}

/// Child unsigned integer with a default.
pub fn child_uint(data: &[u8], wanted: u32, default: u64) -> u64 {
	find(data, wanted).and_then(|b| uint(b).ok()).unwrap_or(default)
}

/// Child float, when present and valid.
pub fn child_float(data: &[u8], wanted: u32) -> Option<f64> {
	find(data, wanted).and_then(|b| float(b).ok())
}

/// Child string, when present.
pub fn child_string(data: &[u8], wanted: u32) -> Option<String> {
	find(data, wanted).map(string)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn vints_and_headers() {
		let h = parse_header(&[0x1A, 0x45, 0xDF, 0xA3, 0x9F]).unwrap();
		assert_eq!(h, Header { id: id::EBML, size: Some(0x1F), len: 5 });
		let h = parse_header(&[0x18, 0x53, 0x80, 0x67, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
			.unwrap();
		assert_eq!(h.size, None);
		assert_eq!(h.len, 12);
		let h = parse_header(&[0xA3, 0x40, 0x02]).unwrap();
		assert_eq!((h.id, h.size), (0xA3, Some(2)));
		assert!(parse_header(&[0x00, 0x81]).is_err());
		assert!(parse_header(&[0x08, 0, 0, 0, 0, 0x81]).is_err()); // 5-byte id
	}

	#[test]
	fn signed_vints() {
		assert_eq!(read_signed_vint(&mut ByteCursor::new(&[0xBF])).unwrap(), 0);
		assert_eq!(read_signed_vint(&mut ByteCursor::new(&[0xBE])).unwrap(), -1);
		assert_eq!(read_signed_vint(&mut ByteCursor::new(&[0x60, 0x00])).unwrap(), 0x2000 - 0x1FFF);
	}

	#[test]
	fn typed_values() {
		assert_eq!(uint(&[]).unwrap(), 0);
		assert_eq!(uint(&[0x0F, 0x42, 0x40]).unwrap(), 1_000_000);
		assert_eq!(sint(&[0xFF, 0xFE]).unwrap(), -2);
		assert_eq!(float(&0.5f32.to_bits().to_be_bytes()).unwrap(), 0.5);
		assert_eq!(float(&2.25f64.to_bits().to_be_bytes()).unwrap(), 2.25);
		assert!(float(&[1, 2, 3]).is_err());
		assert_eq!(string(b"und\0\0"), "und");
	}
}
