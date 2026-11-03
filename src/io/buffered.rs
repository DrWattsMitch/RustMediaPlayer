//! Buffered, position-tracking reader on top of any [`Io`] backend.
//!
//! Demuxers for frame-based formats (MP3) and box-based formats (MP4) need to
//! peek ahead, skip and seek without issuing a tiny `read` call per field. This
//! plays the role of FFmpeg's `AVIOContext` read buffer, in safe Rust.

use crate::io::Io;
use crate::message::{Message, Result};

/// Default buffer capacity. Large enough to hold several maximum-size MPEG
/// audio frames, small enough to be cheap per demuxer.
const DEFAULT_CAPACITY: usize = 64 * 1024;

/// Upper bound for a single `fill` request; protects against callers asking
/// for absurd peek windows derived from corrupt input.
const MAX_PEEK: usize = 4 * 1024 * 1024;

pub struct BufferedReader {
	io: Box<dyn Io>,
	buf: Vec<u8>,
	/// First unread byte in `buf`.
	start: usize,
	/// One past the last valid byte in `buf`.
	end: usize,
	/// Absolute stream position of `buf[0]`.
	offset: u64,
	eof: bool,
}

impl BufferedReader {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self::with_capacity(io, DEFAULT_CAPACITY)
	}

	pub fn with_capacity(io: Box<dyn Io>, capacity: usize) -> Self {
		Self { io, buf: vec![0; capacity.max(16)], start: 0, end: 0, offset: 0, eof: false }
	}

	/// Absolute position of the next byte that will be returned.
	pub fn position(&self) -> u64 {
		self.offset + self.start as u64
	}

	/// Total size of the underlying stream, when the backend knows it.
	pub fn size(&self) -> Option<u64> {
		self.io.size().ok()
	}

	/// Bytes currently buffered and not yet consumed.
	pub fn buffered(&self) -> &[u8] {
		&self.buf[self.start..self.end]
	}

	/// Ensures at least `want` bytes are buffered (fewer only at end of
	/// stream) and returns the buffered window.
	pub fn fill(&mut self, want: usize) -> Result<&[u8]> {
		if want > MAX_PEEK {
			return Err(Message::Other("read window too large"));
		}
		if self.end - self.start >= want || self.eof {
			return Ok(self.buffered());
		}
		if want > self.buf.len() {
			self.buf.resize(want, 0);
		}
		if self.start > 0 {
			self.buf.copy_within(self.start..self.end, 0);
			self.offset += self.start as u64;
			self.end -= self.start;
			self.start = 0;
		}
		while self.end < want {
			let n = self.io.read(&mut self.buf[self.end..])?;
			if n == 0 {
				self.eof = true;
				break;
			}
			self.end += n;
		}
		Ok(self.buffered())
	}

	/// Marks `n` buffered bytes as read. `n` is clamped to what is buffered.
	pub fn consume(&mut self, n: usize) {
		self.start += n.min(self.end - self.start);
	}

	/// Reads exactly `out.len()` bytes or fails with "unexpected end of file".
	pub fn read_exact(&mut self, out: &mut [u8]) -> Result<()> {
		let from_buf = out.len().min(self.end - self.start);
		out[..from_buf].copy_from_slice(&self.buf[self.start..self.start + from_buf]);
		self.start += from_buf;

		let mut filled = from_buf;
		if filled < out.len() {
			// Buffer exhausted: drop the window so its origin tracks the stream.
			self.offset = self.position();
			self.start = 0;
			self.end = 0;
		}
		while filled < out.len() {
			// Large reads bypass the buffer to avoid a needless copy.
			let n = self.io.read(&mut out[filled..])?;
			if n == 0 {
				self.eof = true;
				return Err(Message::Container("unexpected end of file"));
			}
			filled += n;
			// Keep `position()` correct: the buffer is empty, so move its origin.
			self.offset += n as u64;
		}
		Ok(())
	}

	/// Reads exactly `len` bytes into a new vector.
	pub fn read_vec(&mut self, len: usize) -> Result<Vec<u8>> {
		let mut out = vec![0u8; len];
		self.read_exact(&mut out)?;
		Ok(out)
	}

	/// Moves to an absolute position. Seeks inside the buffered window are free.
	pub fn seek(&mut self, position: u64) -> Result<()> {
		if position >= self.offset && position <= self.offset + self.end as u64 {
			// Within (or at the end of) the window: no I/O needed.
			self.start = (position - self.offset) as usize;
			return Ok(());
		}
		self.io.seek(position)?;
		self.offset = position;
		self.start = 0;
		self.end = 0;
		self.eof = false;
		Ok(())
	}

	/// Skips `n` bytes forward.
	pub fn skip(&mut self, n: u64) -> Result<()> {
		let target = self.position().checked_add(n).ok_or(Message::Container("offset overflow"))?;
		self.seek(target)
	}
}
