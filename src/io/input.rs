use crate::container::segment::StreamingOptions;
use crate::container::{dash, hls};
use crate::core::{Metadata, Packet, StreamSet, Time};
use crate::{container, message::*};
use crate::{core::Demuxer, io::*};

pub struct Input {
	demuxer: Box<dyn Demuxer>,
}

impl Input {
	/// Opens a file or URL. HLS playlists (`.m3u8`) and DASH manifests
	/// (`.mpd`) are recognised by their content and played with the default
	/// [`StreamingOptions`].
	pub fn open(input: impl AsRef<str>) -> Result<Self> {
		Self::open_with(input, &StreamingOptions::default())
	}

	/// [`open`](Self::open), with the rendition choice for streaming
	/// manifests.
	pub fn open_with(input: impl AsRef<str>, options: &StreamingOptions) -> Result<Self> {
		let location = input.as_ref();
		let io: Box<dyn Io> = match source::parse_source(location)? {
			source::Source::File(path) => Box::new(File::open(path)?),
			source::Source::Url(url) => Box::new(HttpIo::open(&url)?),
		};
		Self::from_io(io, location, options)
	}

	fn from_io(io: Box<dyn Io>, location: &str, options: &StreamingOptions) -> Result<Self> {
		let mut io = io;
		let mut buffer = vec![0u8; 4096];
		let mut size = 0;
		// Short reads (HTTP chunks) are retried so detection sees 4 KiB.
		while size < buffer.len() {
			let n = io.read(&mut buffer[size..])?;
			if n == 0 {
				break;
			}
			size += n;
		}
		buffer.truncate(size);

		if hls::looks_like_m3u8(&buffer) {
			return Ok(Self { demuxer: Box::new(hls::open(location, options)?) });
		}
		if dash::looks_like_mpd(&buffer) {
			return Ok(Self { demuxer: Box::new(dash::open(location, options)?) });
		}

		io.seek(0)?;

		let demuxer = container::detect(&buffer[..], io)?;
		Ok(Self { demuxer })
	}
}

impl Demuxer for Input {
	#[inline(always)]
	fn read(&mut self) -> Result<Option<Packet>> {
		self.demuxer.read()
	}

	#[inline(always)]
	fn streams(&self) -> &StreamSet {
		self.demuxer.streams()
	}

	#[inline(always)]
	fn metadata(&self) -> &Metadata {
		self.demuxer.metadata()
	}

	#[inline(always)]
	fn seek(&mut self, time: f64) -> Result<()> {
		self.demuxer.seek(time)
	}

	#[inline(always)]
	fn duration(&self) -> Time {
		self.demuxer.duration()
	}
}
