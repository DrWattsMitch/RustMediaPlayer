//! YUV4MPEG2 muxer (FFmpeg: `libavformat/yuv4mpegenc.c`).
//!
//! Accepts one raw video stream. Each packet must be exactly one planar
//! picture in the stream's pixel format; it is written as `FRAME\n` plus the
//! bytes. The header is written by [`Muxer::add`], so streaming outputs that
//! cannot seek work too.

use crate::Message;
use crate::container::y4m::header::{self, ChromaLocation, Interlace, Y4mHeader};
use crate::core::{Metadata, Muxer, Packet, Pixel, RawValue, Stream, StreamKind};
use crate::io::Io;
use crate::message::Result;

pub struct Y4mMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	frame_size: Option<usize>,
}

impl Y4mMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self { io, metadata: Metadata::default(), frame_size: None }
	}
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
	while b != 0 {
		(a, b) = (b, a % b);
	}
	a
}

fn raw_string<'a>(metadata: &'a Metadata, key: &str) -> Option<&'a str> {
	match metadata.raw.get(key) {
		Some(RawValue::String(value)) => Some(value.as_str()),
		_ => None,
	}
}

/// Builds the Y4M header for a raw video stream.
pub fn header_for_stream(stream: &Stream) -> Result<Y4mHeader> {
	if stream.kind != StreamKind::Video {
		return Err(Message::Container("y4m carries only video"));
	}
	let p = &stream.parameters;
	let codec = p.codec.id.as_str();
	// The raw codec's default id is the pixel format name itself.
	let pixel = match p.pixel_format.as_deref() {
		Some(name) => Pixel::from_name(name),
		None if codec == "rawvideo" => Pixel::from_name("yuv420p"),
		None => Pixel::from_name(codec),
	};
	if codec != "rawvideo" && Pixel::from_name(codec).is_none() {
		return Err(Message::Container("y4m requires rawvideo; decode the stream first"));
	}
	let pixel = pixel.ok_or(Message::Container("unsupported y4m pixel format"))?;
	let width = p.width.filter(|w| *w > 0).ok_or(Message::Container("missing video width"))?;
	let height = p.height.filter(|h| *h > 0).ok_or(Message::Container("missing video height"))?;

	// Frame rate is the inverse of the time base, reduced like FFmpeg's av_reduce.
	let (num, den) = (stream.time_base.den, stream.time_base.num);
	if num == 0 || den == 0 {
		return Err(Message::Container("invalid video time base"));
	}
	let g = gcd(num, den);
	let rate = (num / g, den / g);

	let metadata = &stream.metadata;
	let aspect = raw_string(metadata, "sample_aspect_ratio")
		.and_then(|v| v.split_once(':'))
		.and_then(|(n, d)| Some((n.parse().ok()?, d.parse().ok()?)))
		.unwrap_or((0, 0));
	let header = Y4mHeader {
		width,
		height,
		pixel,
		rate,
		aspect,
		interlace: raw_string(metadata, "field_order").and_then(Interlace::from_name),
		chroma_location: raw_string(metadata, "chroma_location").and_then(ChromaLocation::from_name),
		full_range: match raw_string(metadata, "color_range") {
			Some("pc") => Some(true),
			Some("tv") => Some(false),
			_ => None,
		},
	};
	match header.frame_size() {
		Some(size) if size <= header::MAX_FRAME_BYTES => Ok(header),
		_ => Err(Message::Container("y4m picture too large")),
	}
}

impl Muxer for Y4mMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		// Y4M has no container-level tags; kept only so `metadata()` echoes it.
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.frame_size.is_some() {
			return Err(Message::Container("y4m supports only one stream"));
		}
		let header = header_for_stream(stream)?;
		let line = header::write_header(&header)?;
		let frame_size = header.frame_size().ok_or(Message::Container("y4m picture too large"))?;
		self.io.write_all(line.as_bytes())?;
		self.frame_size = Some(frame_size);
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		let frame_size = self.frame_size.ok_or(Message::Container("y4m stream not added"))?;
		if packet.data.len() != frame_size {
			return Err(Message::Container("y4m packet is not one whole picture"));
		}
		self.io.write_all(b"FRAME\n")?;
		self.io.write_all(&packet.data)
	}

	fn finish(&mut self) -> Result<()> {
		if self.frame_size.is_none() {
			return Err(Message::Container("y4m stream not added"));
		}
		Ok(())
	}
}
