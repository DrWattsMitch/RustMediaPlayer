//! YUV4MPEG2 demuxer (FFmpeg: `libavformat/yuv4mpegdec.c`).
//!
//! One packet per frame: the planar picture bytes, without the `FRAME`
//! header, with `pts = dts = frame index` in a `1/fps` time base. The stream
//! codec is `rawvideo` and `Parameters::pixel_format` names the layout.

use crate::Message;
use crate::container::y4m::header::{
	self, ChromaLocation, FRAME_MAGIC, MAX_FRAME_HEADER_LEN, MAX_HEADER_LEN, PLAIN_FRAME_HEADER_LEN,
	Y4mHeader,
};
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, RawValue, Stream, StreamId, StreamSet, Time, TimeBase,
};
use crate::io::{BufferedReader, Io};
use crate::message::Result;

pub struct Y4mDemuxer {
	reader: BufferedReader,
	header: Y4mHeader,
	frame_size: usize,
	/// Offset of the first `FRAME` header.
	data_start: u64,
	/// Frame count when the stream size is known.
	frames: Option<u64>,
	next_frame: u64,
	time_base: TimeBase,
	streams: StreamSet,
	metadata: Metadata,
}

/// Finds the `\n` ending a header line in the first `max` buffered bytes.
fn read_line(reader: &mut BufferedReader, max: usize) -> Result<Option<Vec<u8>>> {
	let window = reader.fill(max + 1)?;
	if window.is_empty() {
		return Ok(None);
	}
	let limit = window.len().min(max + 1);
	let Some(end) = window[..limit].iter().position(|b| *b == b'\n') else {
		return Err(Message::Container(if window.len() > max {
			"y4m header line too long"
		} else {
			"truncated y4m header"
		}));
	};
	let line = window[..end].to_vec();
	reader.consume(end + 1);
	Ok(Some(line))
}

impl Y4mDemuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let line =
			read_line(&mut reader, MAX_HEADER_LEN)?.ok_or(Message::Container("empty y4m stream"))?;
		let header = header::parse_header(&line)?;
		let frame_size = header.frame_size().ok_or(Message::Container("y4m picture too large"))?;
		let data_start = reader.position();

		let stride = frame_size as u64 + PLAIN_FRAME_HEADER_LEN;
		let frames = reader.size().map(|size| size.saturating_sub(data_start) / stride);

		// Seconds per frame: the inverse of the frame rate.
		let time_base = TimeBase::new(header.rate.1, header.rate.0);
		let mut stream = Stream::video(
			StreamId(0),
			header.width,
			header.height,
			&header.pixel.name(),
			CodecId::new("rawvideo"),
		);
		stream.time_base = time_base;
		stream.duration = frames.map(|n| Time::new(n, time_base));
		stream.parameters.bitrate = u32::try_from(
			(frame_size as u64 * 8).saturating_mul(u64::from(header.rate.0)) / u64::from(header.rate.1),
		)
		.ok();
		describe(&header, &mut stream.metadata);

		Ok(Self {
			reader,
			header,
			frame_size,
			data_start,
			frames,
			next_frame: 0,
			time_base,
			streams: StreamSet::new(vec![stream]),
			metadata: Metadata::default(),
		})
	}

	pub fn header(&self) -> &Y4mHeader {
		&self.header
	}
}

/// Records header fields that have no `Parameters` slot as stream metadata,
/// using FFmpeg's names. The muxer reads the same keys back.
fn describe(header: &Y4mHeader, metadata: &mut Metadata) {
	let mut put = |key: &str, value: String| {
		metadata.raw.insert(key.to_string(), RawValue::String(value));
	};
	if let Some(interlace) = header.interlace {
		put("field_order", interlace.name().to_string());
	}
	if header.aspect.0 > 0 && header.aspect.1 > 0 {
		put("sample_aspect_ratio", format!("{}:{}", header.aspect.0, header.aspect.1));
	}
	if let Some(location) = header.chroma_location.map(ChromaLocation::name) {
		put("chroma_location", location.to_string());
	}
	if let Some(full) = header.full_range {
		put("color_range", if full { "pc" } else { "tv" }.to_string());
	}
}

impl Demuxer for Y4mDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		let Some(line) = read_line(&mut self.reader, MAX_FRAME_HEADER_LEN)? else {
			return Ok(None);
		};
		if !line.starts_with(FRAME_MAGIC) {
			return Err(Message::Container("invalid y4m frame header"));
		}
		// A frame cut short by the end of the file ends the stream: a
		// truncated download stays playable up to its last whole frame.
		let size = self.frame_size as u64;
		if let Some(total) = self.reader.size() {
			if self.reader.position().saturating_add(size) > total {
				self.reader.seek(total)?;
				return Ok(None);
			}
		}
		let data = match self.reader.read_vec(self.frame_size) {
			Ok(data) => data,
			// Unknown-length source (HTTP) that ends mid-frame.
			Err(Message::Container("unexpected end of file")) => return Ok(None),
			Err(e) => return Err(e),
		};

		let pts = Time::new(self.next_frame, self.time_base);
		self.next_frame += 1;
		let mut packet = Packet::new(StreamId(0), data);
		packet.pts = Some(pts);
		packet.dts = Some(pts);
		packet.duration = Some(Time::new(1, self.time_base));
		Ok(Some(packet))
	}

	/// Seeks to the frame containing `time`. Like FFmpeg, this assumes plain
	/// `FRAME\n` headers; a file with per-frame fields fails on the next read.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Container("invalid seek time"));
		}
		let (num, den) = self.header.rate;
		let mut frame = (time * f64::from(num) / f64::from(den)).floor() as u64;
		if let Some(frames) = self.frames {
			frame = frame.min(frames);
		}
		let stride = self.frame_size as u64 + PLAIN_FRAME_HEADER_LEN;
		let offset = frame
			.checked_mul(stride)
			.and_then(|o| o.checked_add(self.data_start))
			.ok_or(Message::Container("seek offset overflow"))?;
		self.reader.seek(offset)?;
		self.next_frame = frame;
		Ok(())
	}

	fn duration(&self) -> Time {
		Time::new(self.frames.unwrap_or(0), self.time_base)
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn streams(&self) -> &StreamSet {
		&self.streams
	}
}
