use crate::Message;
use crate::container;
use crate::core::{Hint, Metadata, Packet, Stream};
use crate::io::File;
use crate::message::Result;
use crate::{core::Muxer, io::Io};
use std::path::PathBuf;

pub struct Output {
	path: PathBuf,
	io: Option<Box<dyn Io>>,
	called: bool,
	metadata: Metadata,
	muxer: Option<Box<dyn Muxer>>,
	/// Container forced by name (`mkv`, `ogg`, ...) instead of the extension.
	format: Option<String>,
	/// Container options set before the muxer exists, applied when it is
	/// created.
	options: Vec<(String, String)>,
}

impl Output {
	fn ensure_file(&mut self) -> Result<()> {
		if self.io.is_none() {
			let io = File::create(&self.path)?;
			self.io = Some(Box::new(io));
		}
		Ok(())
	}

	fn ensure_muxer(&mut self, _stream: Option<&Stream>) -> Result<()> {
		if self.muxer.is_some() {
			return Ok(());
		}
		self.ensure_file()?;

		// auto infer based of codec?
		let extension = self.extension();
		let hint = Hint { extension, ..Default::default() };
		// todo: take io?
		let muxer = container::select_muxer(hint, self.io.take().unwrap())?;
		let muxer = self.muxer.insert(muxer);
		muxer.set_metadata(self.metadata.clone())?;
		for (key, value) in &self.options {
			muxer.set_option(key, value)?;
		}
		Ok(())
	}

	fn muxer(&mut self, stream: Option<&Stream>) -> Result<&mut dyn Muxer> {
		self.ensure_muxer(stream)?;

		if let Some(muxer) = self.muxer.as_deref_mut() {
			return Ok(muxer);
		}

		Err(Message::Other("muxer not ready"))
	}

	fn extension(&self) -> Option<String> {
		if let Some(format) = &self.format {
			return Some(format.clone());
		}
		self.path.extension().and_then(|e| e.to_str()).map(|s| s.to_string())
	}

	pub fn create(path: impl AsRef<std::path::Path>) -> Result<Self> {
		let path = path.as_ref().to_path_buf();
		Ok(Self {
			path,
			io: None,
			called: false,
			metadata: Metadata::default(),
			muxer: None,
			format: None,
			options: Vec::new(),
		})
	}

	/// Like [`Output::create`], with the container named explicitly (an
	/// extension such as `mkv`, `webm`, `ogg`, `flac`), as FFmpeg's `-f`.
	pub fn create_as(path: impl AsRef<std::path::Path>, format: &str) -> Result<Self> {
		let mut output = Self::create(path)?;
		output.format = Some(format.to_ascii_lowercase());
		Ok(output)
	}
}

impl Muxer for Output {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata.clone();
		if let Some(muxer) = self.muxer.as_deref_mut() {
			muxer.set_metadata(metadata)?;
		}
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	/// Kept until the muxer is created (on the first `add`), then passed on.
	fn set_option(&mut self, key: &str, value: &str) -> Result<()> {
		match self.muxer.as_deref_mut() {
			Some(muxer) => muxer.set_option(key, value),
			None => {
				self.options.push((key.to_string(), value.to_string()));
				Ok(())
			}
		}
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		self.muxer(Some(stream))?.add(stream)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		self.called = true;
		self.muxer(None)?.write(packet)
	}

	fn finish(&mut self) -> Result<()> {
		// todo: make sense?
		if !self.called {
			return Ok(());
		}
		self.muxer(None)?.finish()
	}
}
