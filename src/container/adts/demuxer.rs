//! ADTS demuxer.

use crate::codecs::aac::AdtsHeader;
use crate::codecs::aac::config::SAMPLE_RATES;
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, Stream, StreamId, StreamSet, Time, TimeBase,
};
use crate::io::{BufferedReader, Io};
use crate::message::{Message, Result};

/// ADTS frames are at most 8191 bytes.
const MAX_FRAME: usize = 8192;
const MAX_SYNC_SEARCH: u64 = 1 << 20;
const SAMPLES_PER_BLOCK: u64 = 1024;

pub struct AdtsDemuxer {
	reader: BufferedReader,
	metadata: Metadata,
	streams: StreamSet,
	data_start: u64,
	next_sample: u64,
	/// Average bytes per 1024 samples (for seeking and duration).
	bytes_per_block: f64,
	duration_samples: u64,
	time_base: TimeBase,
}

impl AdtsDemuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let head = reader.fill(16)?;
		let tag = super::probe::id3_size(head);
		reader.skip(tag as u64)?;
		let start = reader.position();
		let first = loop {
			if reader.position() - start > MAX_SYNC_SEARCH {
				return Err(Message::Container("no adts frame found"));
			}
			let buf = reader.fill(3 * MAX_FRAME)?;
			if buf.len() < 7 {
				return Err(Message::Container("no adts frame found"));
			}
			match (0..buf.len() - 1).find(|&i| buf[i] == 0xFF && super::probe::frame_chain(buf, i) >= 2) {
				Some(i) => {
					reader.consume(i);
					let h =
						AdtsHeader::parse(reader.fill(16)?).ok_or(Message::Container("invalid adts header"))?;
					break h;
				}
				None => {
					let n = buf.len() - 1;
					reader.consume(n);
				}
			}
		};
		let data_start = reader.position();
		// Average frame size over the first frames.
		let buf = reader.fill(16 * MAX_FRAME)?;
		let (mut at, mut blocks) = (0usize, 0u64);
		while let Some(h) = buf.get(at..).and_then(AdtsHeader::parse) {
			at += h.frame_length;
			blocks += h.raw_blocks as u64;
			if blocks >= 64 {
				break;
			}
		}
		let bytes_per_block =
			if blocks > 0 { at as f64 / blocks as f64 } else { first.frame_length as f64 };
		let rate = SAMPLE_RATES[first.sample_rate_index];
		let time_base = TimeBase::new(1, rate);
		let duration_samples = reader.size().map_or(0, |s| {
			(s.saturating_sub(data_start) as f64 / bytes_per_block) as u64 * SAMPLES_PER_BLOCK
		});
		let channels = match first.channel_config {
			7 => 8,
			0 => 2,
			c => c,
		};
		let mut stream = Stream::audio(StreamId(0), rate, channels, CodecId::new("aac"));
		stream.duration = Some(Time::new(duration_samples, time_base));
		stream.parameters.bitrate = Some((bytes_per_block * 8.0 * f64::from(rate) / 1024.0) as u32);
		let mut streams = StreamSet::default();
		streams.add(stream);
		Ok(Self {
			reader,
			metadata: Metadata::default(),
			streams,
			data_start,
			next_sample: 0,
			bytes_per_block,
			duration_samples,
			time_base,
		})
	}

	/// Moves to the next confirmed ADTS header. `false` at end of data.
	fn resync(&mut self) -> Result<bool> {
		loop {
			let buf = self.reader.fill(2 * MAX_FRAME)?;
			if buf.len() < 7 {
				return Ok(false);
			}
			let at_end = buf.len() < 2 * MAX_FRAME;
			let hit = (1..buf.len() - 1).find(|&i| {
				buf[i] == 0xFF && {
					let n = super::probe::frame_chain(buf, i);
					n >= 2 || (n == 1 && at_end)
				}
			});
			match hit {
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

impl Demuxer for AdtsDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		loop {
			let buf = self.reader.fill(MAX_FRAME)?;
			if buf.len() < 7 {
				return Ok(None);
			}
			let Some(h) = AdtsHeader::parse(buf) else {
				if !self.resync()? {
					return Ok(None);
				}
				continue;
			};
			if h.frame_length > buf.len() {
				return Ok(None); // truncated final frame
			}
			let data = buf[..h.frame_length].to_vec();
			self.reader.consume(h.frame_length);
			let samples = h.raw_blocks as u64 * SAMPLES_PER_BLOCK;
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
		let target = ((time * f64::from(self.time_base.den)) as u64).min(self.duration_samples);
		let block = target / SAMPLES_PER_BLOCK;
		let pos = self.data_start + (block as f64 * self.bytes_per_block) as u64;
		self.reader.seek(pos)?;
		if pos != self.data_start {
			self.resync()?;
		}
		self.next_sample = block * SAMPLES_PER_BLOCK;
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
