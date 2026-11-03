//! Bounds-checked big-endian reader over an in-memory byte slice.
//!
//! Every read returns `Result`, so parsing untrusted headers (ID3 frames,
//! MP4 boxes) can never index out of bounds or panic.

use crate::message::{Message, Result};

const EOD: Message = Message::Container("unexpected end of data");

#[derive(Debug, Clone)]
pub struct ByteCursor<'a> {
	data: &'a [u8],
	pos: usize,
}

impl<'a> ByteCursor<'a> {
	pub fn new(data: &'a [u8]) -> Self {
		Self { data, pos: 0 }
	}

	pub fn position(&self) -> usize {
		self.pos
	}

	pub fn remaining(&self) -> usize {
		self.data.len() - self.pos
	}

	pub fn is_empty(&self) -> bool {
		self.remaining() == 0
	}

	/// The unread remainder of the slice.
	pub fn rest(&self) -> &'a [u8] {
		&self.data[self.pos..]
	}

	pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
		let end = self.pos.checked_add(n).ok_or(EOD)?;
		let bytes = self.data.get(self.pos..end).ok_or(EOD)?;
		self.pos = end;
		Ok(bytes)
	}

	pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
		let bytes = self.read_bytes(N)?;
		let mut out = [0u8; N];
		out.copy_from_slice(bytes);
		Ok(out)
	}

	pub fn skip(&mut self, n: usize) -> Result<()> {
		self.read_bytes(n).map(|_| ())
	}

	pub fn read_u8(&mut self) -> Result<u8> {
		Ok(self.read_array::<1>()?[0])
	}

	pub fn read_u16(&mut self) -> Result<u16> {
		Ok(u16::from_be_bytes(self.read_array()?))
	}

	pub fn read_u24(&mut self) -> Result<u32> {
		let [a, b, c] = self.read_array()?;
		Ok(u32::from_be_bytes([0, a, b, c]))
	}

	pub fn read_u32(&mut self) -> Result<u32> {
		Ok(u32::from_be_bytes(self.read_array()?))
	}

	pub fn read_i32(&mut self) -> Result<i32> {
		Ok(i32::from_be_bytes(self.read_array()?))
	}

	pub fn read_u64(&mut self) -> Result<u64> {
		Ok(u64::from_be_bytes(self.read_array()?))
	}

	pub fn read_fourcc(&mut self) -> Result<[u8; 4]> {
		self.read_array()
	}
}
