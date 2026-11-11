use std::path::Path;

use crate::message::Result;

pub trait Io {
	fn read(&mut self, buf: &mut [u8]) -> Result<usize>;
	fn write(&mut self, buf: &[u8]) -> Result<usize>;
	// fn write_all(&mut self, buf: &[u8]) -> Result<()>;
	fn write_all(&mut self, buf: &[u8]) -> Result<()> {
		let mut slice = buf;
		while !slice.is_empty() {
			let n = self.write(slice)?;
			if n == 0 {
				return Err("short write".into());
			}
			slice = &slice[n..];
		}
		Ok(())
	}

	// fn seek(&mut self, pos: u64) -> Result<()>;
	#[allow(unused_variables)]
	fn seek(&mut self, pos: u64) -> Result<()> {
		Err("seek not supported on this backend".into())
	}

	fn size(&self) -> Result<u64> {
		Err("size not supported on this backend".into())
	}
}

pub struct File {
	file: std::fs::File,
}

impl File {
	pub fn open(path: impl AsRef<Path>) -> Result<Self> {
		let file = std::fs::File::open(path.as_ref())?;
		Ok(Self { file })
	}

	/// Creates (or truncates) a file for writing. It is also opened for
	/// reading when the system allows, so muxers can move data they already
	/// wrote (MP4 `+faststart`); otherwise it is write-only, as before.
	pub fn create(path: impl AsRef<Path>) -> Result<Self> {
		let path = path.as_ref();
		let read_write =
			std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path);
		let file = match read_write {
			Ok(file) => file,
			Err(_) => std::fs::File::create(path)?,
		};
		Ok(Self { file })
	}
}

impl Io for File {
	fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
		use std::io::Read;
		Ok(self.file.read(buf)?)
	}

	fn seek(&mut self, position: u64) -> Result<()> {
		use std::io::{Seek, SeekFrom};
		self.file.seek(SeekFrom::Start(position))?;
		Ok(())
	}

	fn size(&self) -> Result<u64> {
		Ok(self.file.metadata()?.len())
	}

	fn write(&mut self, buf: &[u8]) -> Result<usize> {
		use std::io::Write;
		Ok(self.file.write(buf)?)
	}

	fn write_all(&mut self, buf: &[u8]) -> Result<()> {
		use std::io::Write;
		self.file.write_all(buf)?;
		Ok(())
	}
}
