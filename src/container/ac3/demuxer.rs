//! Raw AC-3 / E-AC-3 demuxer.

use crate::codecs::ac3::{Header, HeaderError, layout_of};
use crate::codecs::bitstream::BitReader;
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, Stream, StreamId, StreamSet, Time, TimeBase,
};
use crate::io::{BufferedReader, Io};
use crate::message::{Message, Result};

/// Largest syncframe: 2048 16-bit words.
const MAX_FRAME: usize = 4096;
/// How far to look for the first syncframe.
const MAX_SYNC_SEARCH: u64 = 1 << 20;

pub struct Ac3Demuxer {
	reader: BufferedReader,
	metadata: Metadata,
	streams: StreamSet,
	data_start: u64,
	next_sample: u64,
	samples_per_frame: u64,
	/// Bytes per access unit of the first frame (seek estimate).
	unit_bytes: u64,
	duration_samples: u64,
	time_base: TimeBase,
}

fn parse(data: &[u8]) -> std::result::Result<Header, HeaderError> {
	Header::parse(&mut BitReader::new(data))
}

impl Ac3Demuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let start = reader.position();
		let first = loop {
			let pos = reader.position();
			if pos - start > MAX_SYNC_SEARCH {
				return Err(Message::Container("no ac-3 syncframe found"));
			}
			let buf = reader.fill(3 * MAX_FRAME)?;
			if buf.len() < 8 {
				return Err(Message::Container("no ac-3 syncframe found"));
			}
			let hit = (0..buf.len() - 1).find(|&i| {
				buf[i] == 0x0B
					&& buf[i + 1] == 0x77
					&& super::probe::frame_chain(buf, i) >= 2.min(buf.len() / MAX_FRAME + 1)
			});
			match hit {
				Some(i) => {
					reader.consume(i);
					let h = parse(reader.fill(MAX_FRAME)?)
						.map_err(|_| Message::Container("invalid ac-3 header"))?;
					break h;
				}
				None => {
					let n = buf.len() - 1;
					reader.consume(n);
				}
			}
		};
		let data_start = reader.position();
		// Size of one access unit (independent frame + dependent frames).
		let buf = reader.fill(2 * MAX_FRAME)?;
		let mut unit = first.frame_size;
		while let Some(Ok(h)) = buf.get(unit..).map(parse) {
			if h.frame_type != Some(crate::codecs::ac3::header::FrameType::Dependent) {
				break;
			}
			unit += h.frame_size;
		}
		let (channels, _) = layout_of(first.channel_mode, first.lfe_on);
		let codec = if first.is_eac3() { "eac3" } else { "ac3" };
		let samples_per_frame = first.num_blocks as u64 * 256;
		let time_base = TimeBase::new(1, first.sample_rate);
		let duration_samples = reader
			.size()
			.map_or(0, |s| s.saturating_sub(data_start) / unit.max(1) as u64 * samples_per_frame);
		let mut stream =
			Stream::audio(StreamId(0), first.sample_rate, channels as u8, CodecId::new(codec));
		stream.parameters.bitrate = Some(first.bit_rate);
		stream.duration = Some(Time::new(duration_samples, time_base));
		let mut streams = StreamSet::default();
		streams.add(stream);
		Ok(Self {
			reader,
			metadata: Metadata::default(),
			streams,
			data_start,
			next_sample: 0,
			samples_per_frame,
			unit_bytes: unit as u64,
			duration_samples,
			time_base,
		})
	}

	/// Skips to the next syncword. Returns `false` at end of data.
	fn resync(&mut self) -> Result<bool> {
		loop {
			let buf = self.reader.fill(MAX_FRAME)?;
			if buf.len() < 8 {
				return Ok(false);
			}
			match (1..buf.len() - 1).find(|&i| buf[i] == 0x0B && buf[i + 1] == 0x77) {
				Some(i) => {
					self.reader.consume(i);
					return Ok(true);
				}
				None => {
					let n = buf.len() - 1;
					self.reader.consume(n);
				}
			}
		}
	}
}

impl Demuxer for Ac3Demuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		loop {
			let buf = self.reader.fill(3 * MAX_FRAME)?;
			if buf.len() < 8 {
				return Ok(None);
			}
			let Ok(first) = parse(buf) else {
				if !self.resync()? {
					return Ok(None);
				}
				continue;
			};
			if first.frame_type == Some(crate::codecs::ac3::header::FrameType::Dependent)
				|| first.frame_size > buf.len()
			{
				if first.frame_size > buf.len() {
					return Ok(None); // truncated final frame
				}
				self.reader.consume(first.frame_size);
				continue;
			}
			let mut size = first.frame_size;
			while let Some(next) = buf.get(size..).and_then(|b| Header::peek_size(b).map(|_| b)) {
				match parse(next) {
					Ok(h) if h.frame_type == Some(crate::codecs::ac3::header::FrameType::Dependent) => {
						if size + h.frame_size > buf.len() {
							break;
						}
						size += h.frame_size;
					}
					// Other substreams (ids 1-7) ride along with the unit.
					Err(HeaderError::FrameType) => match Header::peek_size(next) {
						Some((s, _)) if size + s <= buf.len() => size += s,
						_ => break,
					},
					_ => break,
				}
			}
			let data = buf[..size].to_vec();
			self.reader.consume(size);
			let samples = first.num_blocks as u64 * 256;
			let pts = Time::new(self.next_sample, self.time_base);
			self.next_sample += samples;
			return Ok(Some(Packet {
				stream_id: StreamId(0),
				data,
				pts: Some(pts),
				dts: Some(pts),
				duration: Some(Time::new(samples, self.time_base)),
				keyframe: true,
			}));
		}
	}

	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let rate = f64::from(self.time_base.den);
		let target = ((time * rate) as u64).min(self.duration_samples);
		let unit_index = target / self.samples_per_frame.max(1);
		self.reader.seek(self.data_start + unit_index * self.unit_bytes)?;
		self.resync_at_unit()?;
		self.next_sample = unit_index * self.samples_per_frame;
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

impl Ac3Demuxer {
	/// After a byte-position seek: land on an independent frame.
	fn resync_at_unit(&mut self) -> Result<()> {
		loop {
			let buf = self.reader.fill(MAX_FRAME)?;
			if buf.len() < 8 {
				return Ok(());
			}
			if let Ok(h) = parse(buf) {
				if h.frame_type != Some(crate::codecs::ac3::header::FrameType::Dependent) {
					return Ok(());
				}
			}
			self.reader.consume(1);
			if !self.resync()? {
				return Ok(());
			}
		}
	}
}
