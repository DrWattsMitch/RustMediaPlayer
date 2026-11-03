//! ISO Base Media File Format (ISO/IEC 14496-12) box primitives.
//!
//! Reading works on in-memory slices (the `moov` box is loaded whole, like
//! FFmpeg's `mov_read_header` walk), so every access is a bounds-checked
//! slice operation. Writing appends to a `Vec<u8>` and back-patches sizes.

use crate::io::ByteCursor;
use crate::message::{Message, Result};

pub type FourCC = [u8; 4];

/// Iterates over the boxes contained in a byte slice.
///
/// A truncated final box is clamped to the available bytes (FFmpeg also
/// tolerates this, which keeps partially-downloaded files playable); a box
/// whose size is smaller than its own header is an error.
pub struct BoxIter<'a> {
	cursor: ByteCursor<'a>,
	failed: bool,
}

impl<'a> BoxIter<'a> {
	pub fn new(data: &'a [u8]) -> Self {
		Self { cursor: ByteCursor::new(data), failed: false }
	}
}

impl<'a> Iterator for BoxIter<'a> {
	type Item = Result<(FourCC, &'a [u8])>;

	fn next(&mut self) -> Option<Self::Item> {
		if self.failed || self.cursor.remaining() < 8 {
			return None;
		}
		let result = (|| {
			let size = u64::from(self.cursor.read_u32()?);
			let kind = self.cursor.read_fourcc()?;
			let (size, header) = match size {
				0 => (8 + self.cursor.remaining() as u64, 8),
				1 => (self.cursor.read_u64()?, 16),
				_ => (size, 8),
			};
			if size < header {
				return Err(Message::Container("invalid mp4 box size"));
			}
			let body_len = usize::try_from(size - header).unwrap_or(usize::MAX);
			let body = self.cursor.read_bytes(body_len.min(self.cursor.remaining()))?;
			Ok((kind, body))
		})();
		if result.is_err() {
			self.failed = true;
		}
		Some(result)
	}
}

/// First direct child of `kind`, ignoring malformed trailing data.
pub fn find<'a>(data: &'a [u8], kind: &FourCC) -> Option<&'a [u8]> {
	BoxIter::new(data).map_while(|r| r.ok()).find(|(k, _)| k == kind).map(|(_, body)| body)
}

/// Follows a path of nested boxes, e.g. `[b"mdia", b"minf", b"stbl"]`.
pub fn find_path<'a>(data: &'a [u8], path: &[&FourCC]) -> Option<&'a [u8]> {
	path.iter().try_fold(data, |current, kind| find(current, kind))
}

/// Splits a "full box" body into (version, flags, payload).
pub fn full_box(body: &[u8]) -> Result<(u8, u32, &[u8])> {
	let header = body.get(..4).ok_or(Message::Container("truncated mp4 full box"))?;
	let flags = u32::from_be_bytes([0, header[1], header[2], header[3]]);
	Ok((header[0], flags, &body[4..]))
}

/// Human-readable four-character code (0xA9 → '©', as in iTunes tags).
pub fn fourcc_string(kind: &FourCC) -> String {
	kind.iter().map(|&b| char::from(b)).collect::<String>().trim_end().to_string()
}

/// Big-endian append helpers for building boxes.
pub trait Put {
	fn put_u8(&mut self, v: u8);
	fn put_u16(&mut self, v: u16);
	fn put_u24(&mut self, v: u32);
	fn put_u32(&mut self, v: u32);
	fn put_i32(&mut self, v: i32);
	fn put_u64(&mut self, v: u64);
	fn put(&mut self, bytes: &[u8]);
	fn put_zeros(&mut self, n: usize);
}

impl Put for Vec<u8> {
	fn put_u8(&mut self, v: u8) {
		self.push(v);
	}
	fn put_u16(&mut self, v: u16) {
		self.extend_from_slice(&v.to_be_bytes());
	}
	fn put_u24(&mut self, v: u32) {
		self.extend_from_slice(&v.to_be_bytes()[1..]);
	}
	fn put_u32(&mut self, v: u32) {
		self.extend_from_slice(&v.to_be_bytes());
	}
	fn put_i32(&mut self, v: i32) {
		self.extend_from_slice(&v.to_be_bytes());
	}
	fn put_u64(&mut self, v: u64) {
		self.extend_from_slice(&v.to_be_bytes());
	}
	fn put(&mut self, bytes: &[u8]) {
		self.extend_from_slice(bytes);
	}
	fn put_zeros(&mut self, n: usize) {
		self.resize(self.len() + n, 0);
	}
}

/// Starts a box; returns the offset to pass to [`end_box`].
pub fn begin_box(out: &mut Vec<u8>, kind: &FourCC) -> usize {
	let at = out.len();
	out.put_u32(0);
	out.put(kind);
	at
}

/// Starts a full box (version + flags).
pub fn begin_full_box(out: &mut Vec<u8>, kind: &FourCC, version: u8, flags: u32) -> usize {
	let at = begin_box(out, kind);
	out.put_u8(version);
	out.put_u24(flags);
	at
}

/// Back-patches the size of the box started at `at`.
pub fn end_box(out: &mut [u8], at: usize) -> Result<()> {
	let size = u32::try_from(out.len() - at).map_err(|_| Message::Container("mp4 box too large"))?;
	out[at..at + 4].copy_from_slice(&size.to_be_bytes());
	Ok(())
}

/// Writes a complete box whose body is produced by `body`.
pub fn write_box(
	out: &mut Vec<u8>,
	kind: &FourCC,
	body: impl FnOnce(&mut Vec<u8>) -> Result<()>,
) -> Result<()> {
	let at = begin_box(out, kind);
	body(out)?;
	end_box(out, at)
}

/// Writes a complete full box whose body is produced by `body`.
pub fn write_full_box(
	out: &mut Vec<u8>,
	kind: &FourCC,
	version: u8,
	flags: u32,
	body: impl FnOnce(&mut Vec<u8>) -> Result<()>,
) -> Result<()> {
	let at = begin_full_box(out, kind, version, flags);
	body(out)?;
	end_box(out, at)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn iterates_and_round_trips() {
		let mut out = Vec::new();
		write_box(&mut out, b"moov", |o| {
			write_full_box(o, b"mvhd", 0, 0, |o| {
				o.put_u32(7);
				Ok(())
			})?;
			write_box(o, b"trak", |_| Ok(()))
		})
		.unwrap();
		let moov = find(&out, b"moov").unwrap();
		let kinds: Vec<_> = BoxIter::new(moov).map(|r| r.unwrap().0).collect();
		assert_eq!(kinds, vec![*b"mvhd", *b"trak"]);
		let (version, _, payload) = full_box(find(moov, b"mvhd").unwrap()).unwrap();
		assert_eq!((version, payload), (0, &[0, 0, 0, 7][..]));
	}

	#[test]
	fn rejects_undersized_box_and_clamps_truncated_box() {
		let bad = [0, 0, 0, 4, b'f', b'r', b'e', b'e'];
		assert!(BoxIter::new(&bad).next().unwrap().is_err());
		let truncated = [0, 0, 0, 100, b'm', b'd', b'a', b't', 1, 2];
		let (kind, body) = BoxIter::new(&truncated).next().unwrap().unwrap();
		assert_eq!((&kind, body), (b"mdat", &[1, 2][..]));
	}
}
