//! In-memory [`Io`] backend.
//!
//! Useful for tests and for muxing into a buffer. Clones share the same
//! underlying bytes (but keep independent cursors), so a caller can hand one
//! clone to a muxer and read the result back through another.

use std::cell::RefCell;
use std::rc::Rc;

use crate::io::Io;
use crate::message::{Message, Result};

#[derive(Debug, Clone, Default)]
pub struct MemoryIo {
	data: Rc<RefCell<Vec<u8>>>,
	pos: usize,
}

impl MemoryIo {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn from_vec(data: Vec<u8>) -> Self {
		Self { data: Rc::new(RefCell::new(data)), pos: 0 }
	}

	/// Copy of the current contents.
	pub fn contents(&self) -> Vec<u8> {
		self.data.borrow().clone()
	}
}

impl Io for MemoryIo {
	fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
		let data = self.data.borrow();
		let available = data.get(self.pos..).unwrap_or(&[]);
		let n = available.len().min(buf.len());
		buf[..n].copy_from_slice(&available[..n]);
		self.pos += n;
		Ok(n)
	}

	fn write(&mut self, buf: &[u8]) -> Result<usize> {
		let mut data = self.data.borrow_mut();
		let end = self.pos.checked_add(buf.len()).ok_or(Message::Other("write overflow"))?;
		if end > data.len() {
			data.resize(end, 0);
		}
		data[self.pos..end].copy_from_slice(buf);
		self.pos = end;
		Ok(buf.len())
	}

	fn seek(&mut self, pos: u64) -> Result<()> {
		self.pos = usize::try_from(pos).map_err(|_| Message::Other("seek overflow"))?;
		Ok(())
	}

	fn size(&self) -> Result<u64> {
		Ok(self.data.borrow().len() as u64)
	}
}
