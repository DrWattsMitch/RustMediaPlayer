//! MP3 (MPEG audio elementary stream) demuxer.
//!
//! Layout handled, in file order:
//!
//! ```text
//! [ID3v2 tag(s)] [Xing/Info/VBRI frame] frame frame frame ... [ID3v1 "TAG"]
//! ```
//!
//! Each packet is exactly one MPEG audio frame with `pts`/`dts`/`duration`
//! in samples (time base `1/sample_rate`). Like FFmpeg's `mp3dec.c`, frame
//! sync is only accepted when the *next* header also matches, and junk
//! between frames is skipped by resynchronising.

use super::id3;
use super::xing::{self, InfoTag};
use crate::codecs::mp3::FrameHeader;
use crate::codecs::mp3::header::{HEADER_SIZE, MAX_FRAME_SIZE};
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, RawValue, Stream, StreamId, StreamSet, Time, TimeBase,
};
use crate::io::{BufferedReader, Io};
use crate::message::{Message, Result};

/// Stream metadata key carrying the raw LAME extension bytes.
pub const LAME_TAG_KEY: &str = "lame_tag";
/// Container-neutral gapless keys (shared with the MP4 demuxer/muxer).
pub use crate::container::mp4::demuxer::{ENCODER_DELAY_KEY, ENCODER_PADDING_KEY};
/// Delay of the MPEG audio decoder's filterbank, in samples (528 + 1).
const DECODER_DELAY: u32 = 529;

/// How far to search for the first frame after the tags.
const MAX_INITIAL_SYNC_SEARCH: u64 = 1024 * 1024;

pub struct Mp3Demuxer {
	reader: BufferedReader,
	metadata: Metadata,
	streams: StreamSet,
	/// Reference header: later frames must be compatible with it.
	first: FrameHeader,
	/// Position of the first audio frame (after any info frame).
	data_start: u64,
	/// End of audio data (start of an ID3v1 tag, or file size), if known.
	data_end: Option<u64>,
	info: Option<InfoTag>,
	/// Position of the info frame; Xing TOC offsets are relative to it.
	info_pos: u64,
	/// PTS of the next packet, in samples.
	next_sample: u64,
	/// Total duration in samples (exact with an info tag, else estimated).
	duration_samples: u64,
	time_base: TimeBase,
}

impl Mp3Demuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let mut metadata = Metadata::default();
		let size = reader.size();

		// 1. Leading ID3v2 tags (files occasionally carry more than one).
		loop {
			let head = reader.fill(id3::ID3V2_HEADER_SIZE)?;
			let Some(tag_size) = id3::v2_tag_size(head) else { break };
			if tag_size <= id3::MAX_PARSED_TAG_SIZE {
				let tag = reader.read_vec(tag_size)?;
				// A damaged tag must not make the audio unplayable.
				let _ = id3::parse_v2(&tag, &mut metadata);
			} else {
				reader.skip(tag_size as u64)?;
			}
		}
		let audio_search_start = reader.position();

		// 2. Trailing ID3v1 tag, when the stream size is known.
		let mut data_end = size;
		if let Some(size) = size {
			if size >= audio_search_start + id3::ID3V1_SIZE as u64 {
				let tag_pos = size - id3::ID3V1_SIZE as u64;
				reader.seek(tag_pos)?;
				let tag = reader.read_vec(id3::ID3V1_SIZE)?;
				if id3::parse_v1(&tag, &mut metadata) {
					data_end = Some(tag_pos);
				}
				reader.seek(audio_search_start)?;
			}
		}

		// 3. First frame, confirmed by its successor.
		let first_pos = find_first_frame(&mut reader, data_end)?;
		reader.seek(first_pos)?;
		let head = reader.fill(MAX_FRAME_SIZE)?;
		let first = FrameHeader::parse(head).ok_or(Message::Container("mp3 frame sync lost"))?;
		let first_frame = head.get(..first.frame_size).unwrap_or(head);

		// 4. Optional Xing/Info/VBRI frame.
		let info = xing::parse(first_frame, &first);
		let data_start = if info.is_some() { first_pos + first.frame_size as u64 } else { first_pos };
		reader.seek(data_start)?;

		let duration_samples =
			estimate_duration(&first, info.as_ref(), data_start, data_end.unwrap_or(data_start));
		let time_base = TimeBase::new(1, first.sample_rate);

		let mut stream = Stream::audio(
			StreamId(0),
			first.sample_rate,
			first.channels(),
			CodecId::new(first.codec_name()),
		);
		stream.duration = Some(Time::new(duration_samples, time_base));
		stream.parameters.bitrate = Some(average_bitrate(&first, info.as_ref(), duration_samples));
		if let Some(info) = &info {
			// Gapless information, in the container-neutral meaning used by
			// every RMP demuxer: samples to drop from the start / end of the
			// decoded output. LAME stores the encoder's own delay; the
			// decoder adds 529 more (FFmpeg's `start_pad`/`end_pad`).
			let raw = &mut stream.metadata.raw;
			if let Some(delay) = info.encoder_delay {
				let skip = delay + DECODER_DELAY;
				raw.insert(ENCODER_DELAY_KEY.into(), RawValue::String(skip.to_string()));
			}
			if let Some(padding) = info.encoder_padding.filter(|&p| p >= DECODER_DELAY) {
				let trim = padding - DECODER_DELAY;
				raw.insert(ENCODER_PADDING_KEY.into(), RawValue::String(trim.to_string()));
			}
			if let Some(tag) = info.lame_tag {
				// Lets the MP3 muxer write the same gapless info back.
				raw.insert(LAME_TAG_KEY.into(), RawValue::Bytes(tag.to_vec()));
			}
		}

		let mut streams = StreamSet::default();
		streams.add(stream);

		Ok(Self {
			reader,
			metadata,
			streams,
			first,
			data_start,
			data_end,
			info,
			info_pos: first_pos,
			next_sample: 0,
			duration_samples,
			time_base,
		})
	}

	fn at_end(&self) -> bool {
		self.data_end.is_some_and(|end| self.reader.position() >= end)
	}

	/// Remaining bytes before `data_end` that may be inspected.
	fn limit(&self, want: usize) -> usize {
		match self.data_end {
			Some(end) => want.min(end.saturating_sub(self.reader.position()) as usize),
			None => want,
		}
	}

	/// Advances to the next position holding a frame compatible with the
	/// stream (confirmed by the following header when it is in range).
	/// Returns `false` at end of data.
	fn resync(&mut self, skip_current: bool) -> Result<bool> {
		if skip_current {
			self.reader.consume(1);
		}
		loop {
			if self.at_end() {
				return Ok(false);
			}
			let wanted = 2 * MAX_FRAME_SIZE + HEADER_SIZE;
			let window = self.limit(wanted);
			let buf = self.reader.fill(window)?;
			let buf = &buf[..buf.len().min(window)];
			if buf.len() < HEADER_SIZE {
				return Ok(false);
			}
			let at_end = buf.len() < wanted;
			match find_confirmed_frame(buf, Some(&self.first), at_end) {
				Scan::Found(offset) => {
					self.reader.consume(offset);
					return Ok(true);
				}
				Scan::Advance(0) if at_end => return Ok(false),
				Scan::Advance(n) => self.reader.consume(n.max(1)),
			}
		}
	}
}

/// Outcome of scanning a window for a confirmed frame start.
enum Scan {
	/// A frame starts at this offset.
	Found(usize),
	/// Nothing usable before this offset; refill from here and retry.
	Advance(usize),
}

/// Searches `buf` for a frame header whose successor is also a valid,
/// compatible header. A candidate whose successor lies beyond `buf` is
/// accepted only when `at_end` (the window reaches end of data); otherwise
/// the scan stops there so the caller can refill and confirm it.
fn find_confirmed_frame(buf: &[u8], reference: Option<&FrameHeader>, at_end: bool) -> Scan {
	let Some(end) = buf.len().checked_sub(HEADER_SIZE - 1) else { return Scan::Advance(0) };
	for i in 0..end {
		if buf[i] != 0xFF {
			continue;
		}
		let Some(h) = FrameHeader::parse(&buf[i..]) else { continue };
		if reference.is_some_and(|r| !r.is_compatible(&h)) {
			continue;
		}
		let next = i + h.frame_size;
		match buf.get(next..next + HEADER_SIZE) {
			Some(n) => match FrameHeader::parse(n) {
				Some(n) if n.is_compatible(&h) => return Scan::Found(i),
				_ => continue,
			},
			None if at_end => {
				// Last frame of the stream: accept if it fits exactly.
				if next <= buf.len() {
					return Scan::Found(i);
				}
			}
			None => return Scan::Advance(i),
		}
	}
	Scan::Advance(end)
}

fn find_first_frame(reader: &mut BufferedReader, data_end: Option<u64>) -> Result<u64> {
	let start = reader.position();
	loop {
		let pos = reader.position();
		if pos - start > MAX_INITIAL_SYNC_SEARCH || data_end.is_some_and(|e| pos >= e) {
			return Err(Message::Container("no mpeg audio frame found"));
		}
		let mut window = 2 * MAX_FRAME_SIZE + HEADER_SIZE;
		if let Some(end) = data_end {
			window = window.min((end - pos) as usize);
		}
		let buf = reader.fill(window)?;
		let buf = &buf[..buf.len().min(window)];
		if buf.len() < HEADER_SIZE {
			return Err(Message::Container("no mpeg audio frame found"));
		}
		let at_end = buf.len() < 2 * MAX_FRAME_SIZE + HEADER_SIZE;
		match find_confirmed_frame(buf, None, at_end) {
			Scan::Found(offset) => return Ok(pos + offset as u64),
			Scan::Advance(0) if at_end => return Err(Message::Container("no mpeg audio frame found")),
			Scan::Advance(n) => reader.consume(n.max(1)),
		}
	}
}

fn estimate_duration(first: &FrameHeader, info: Option<&InfoTag>, start: u64, end: u64) -> u64 {
	if let Some(frames) = info.and_then(|i| i.frames) {
		return u64::from(frames) * u64::from(first.samples);
	}
	// Constant-bitrate estimate from the data size, as FFmpeg does.
	let bytes = u128::from(end.saturating_sub(start));
	let samples = bytes * 8 * u128::from(first.sample_rate) / (u128::from(first.bitrate_kbps) * 1000);
	u64::try_from(samples).unwrap_or(u64::MAX)
}

fn average_bitrate(first: &FrameHeader, info: Option<&InfoTag>, duration_samples: u64) -> u32 {
	if let (Some(bytes), true) = (info.and_then(|i| i.bytes), duration_samples > 0) {
		let bps = u128::from(bytes) * 8 * u128::from(first.sample_rate) / u128::from(duration_samples);
		if let Ok(bps) = u32::try_from(bps) {
			return bps;
		}
	}
	first.bitrate_kbps * 1000
}

impl Demuxer for Mp3Demuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		loop {
			if self.at_end() {
				return Ok(None);
			}
			let head = self.reader.fill(HEADER_SIZE)?;
			if head.len() < HEADER_SIZE {
				return Ok(None);
			}
			let header = FrameHeader::parse(head).filter(|h| self.first.is_compatible(h));
			let Some(header) = header else {
				if !self.resync(true)? {
					return Ok(None);
				}
				continue;
			};

			let size = header.frame_size;
			if self.limit(size) < size {
				return Ok(None); // truncated final frame before the ID3v1 tag
			}
			let buf = self.reader.fill(size)?;
			if buf.len() < size {
				return Ok(None); // truncated final frame at end of file
			}
			let data = buf[..size].to_vec();
			self.reader.consume(size);

			let pts = Time::new(self.next_sample, self.time_base);
			self.next_sample += u64::from(header.samples);
			return Ok(Some(Packet {
				stream_id: StreamId(0),
				data,
				pts: Some(pts),
				dts: Some(pts),
				duration: Some(Time::new(u64::from(header.samples), self.time_base)),
				keyframe: true,
			}));
		}
	}

	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let samples_per_frame = u64::from(self.first.samples);
		let target = ((time * f64::from(self.first.sample_rate)) as u64).min(self.duration_samples);
		let frame_index = target / samples_per_frame;
		let data_end = self.data_end.or_else(|| self.reader.size()).unwrap_or(self.data_start);

		let toc = self.info.as_ref().and_then(|i| Some((i.toc?, i.bytes?, i.frames?)));
		let mut position = match toc {
			Some((toc, bytes, frames)) => {
				let percent = frame_index as f64 * 100.0 / f64::from(frames);
				self.info_pos + xing::toc_offset(&toc, u64::from(bytes), percent)
			}
			None => {
				// Constant bitrate: bytes are proportional to time.
				let offset = u128::from(frame_index * samples_per_frame)
					* u128::from(self.first.bitrate_kbps)
					* 125 / u128::from(self.first.sample_rate);
				self.data_start.saturating_add(u64::try_from(offset).unwrap_or(u64::MAX))
			}
		};
		position = position.clamp(self.data_start, data_end.max(self.data_start));

		self.reader.seek(position)?;
		// Land on a real frame boundary (the estimate can point mid-frame).
		self.resync(false)?;
		self.next_sample = frame_index * samples_per_frame;
		Ok(())
	}

	fn duration(&self) -> Time {
		Time::new(self.duration_samples, self.time_base)
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn streams(&self) -> &StreamSet {
		&self.streams
	}
}
