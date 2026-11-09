//! Native FLAC demuxer (`.flac`).
//!
//! A FLAC file is `"fLaC"`, metadata blocks (STREAMINFO first), then frames
//! with no length prefix. Frames are delimited the way FFmpeg's
//! `flac_parser.c` does it: a candidate next frame must have a valid header
//! (CRC-8), the expected frame/sample number, and the bytes before it must
//! pass the current frame's CRC-16. That rejects false sync codes inside
//! audio data. An optional ID3v2 tag before `"fLaC"` is skipped.
//!
//! FFmpeg: `libavformat/flacdec.c`, `libavcodec/flac_parser.c`.

use crate::codecs::flac::decoder::crc16;
use crate::codecs::flac::{FrameHeader, StreamInfo};
use crate::container::xiph;
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, Parameters, Stream, StreamId, StreamKind, StreamSet, Time,
	TimeBase,
};
use crate::io::{BufferedReader, Io};
use crate::message::{Message, Result};

const BLOCK_STREAMINFO: u8 = 0;
const BLOCK_SEEKTABLE: u8 = 3;
const BLOCK_VORBIS_COMMENT: u8 = 4;
const BLOCK_PICTURE: u8 = 6;
/// Seek table placeholder points have this sample number.
const PLACEHOLDER: u64 = u64::MAX;
/// Smallest possible frame: header (6) + one 1-byte subframe + CRC-16.
const MIN_FRAME: usize = 9;
/// Largest frame searched for (8 channels × 65535 samples × 32 bits is
/// about 2 MiB; anything bigger is damage).
const MAX_FRAME: usize = 4 * 1024 * 1024;
/// Largest number of probes of the bisection seek.
const MAX_SEEK_PROBES: usize = 48;

pub struct FlacDemuxer {
	reader: BufferedReader,
	streams: StreamSet,
	metadata: Metadata,
	info: StreamInfo,
	duration: Time,
	first_frame: u64,
	data_end: u64,
	/// `(sample, absolute offset)` from SEEKTABLE.
	seek_table: Vec<(u64, u64)>,
}

/// Skips an ID3v2 tag at the reader position, if there is one.
fn skip_id3(reader: &mut BufferedReader) -> Result<()> {
	let head = reader.fill(10)?;
	if head.len() == 10 && &head[..3] == b"ID3" {
		let size = head[6..10].iter().fold(0u64, |acc, &b| (acc << 7) | u64::from(b & 0x7F));
		let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
		reader.skip(10 + size + footer)?;
	}
	Ok(())
}

/// `true` when `h` could follow a frame of this stream: same depth, rate
/// and channel count as STREAMINFO whenever the header states them.
fn consistent(h: &FrameHeader, info: &StreamInfo) -> bool {
	(h.bits_per_sample == 0 || h.bits_per_sample == info.bits_per_sample)
		&& (h.sample_rate == 0 || h.sample_rate == info.sample_rate)
		&& h.channels.channels() == info.channels
}

impl FlacDemuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		skip_id3(&mut reader)?;
		let mut magic = [0u8; 4];
		reader.read_exact(&mut magic).map_err(|_| Message::Container("not a flac file"))?;
		if &magic != b"fLaC" {
			return Err(Message::Container("not a flac file"));
		}

		let mut info = None;
		let mut metadata = Metadata::default();
		let mut seek_points = Vec::new();
		for _ in 0..1024 {
			let mut header = [0u8; 4];
			reader.read_exact(&mut header)?;
			let last = header[0] & 0x80 != 0;
			let kind = header[0] & 0x7F;
			let len = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
			match kind {
				BLOCK_STREAMINFO | BLOCK_SEEKTABLE | BLOCK_VORBIS_COMMENT | BLOCK_PICTURE => {
					let body = reader.read_vec(len)?;
					match kind {
						BLOCK_STREAMINFO => info = info.or(StreamInfo::parse(&body)),
						BLOCK_SEEKTABLE => {
							for point in body.chunks_exact(18) {
								let sample = u64::from_be_bytes(point[..8].try_into().unwrap_or([0xFF; 8]));
								let offset = u64::from_be_bytes(point[8..16].try_into().unwrap_or([0; 8]));
								if sample != PLACEHOLDER {
									seek_points.push((sample, offset));
								}
							}
						}
						BLOCK_VORBIS_COMMENT => {
							xiph::parse_comments(&body, &mut metadata);
						}
						_ => {
							if let Some(image) = xiph::parse_picture(&body) {
								metadata.images.push(image);
							}
						}
					}
				}
				127 => return Err(Message::Container("invalid flac metadata block")),
				_ => reader.skip(len as u64)?,
			}
			if last {
				break;
			}
		}
		let info = info.ok_or(Message::Container("flac STREAMINFO missing or invalid"))?;
		let first_frame = reader.position();
		let data_end = reader.size().unwrap_or(u64::MAX);
		let mut seek_table: Vec<(u64, u64)> = seek_points
			.into_iter()
			.filter_map(|(s, o): (u64, u64)| {
				first_frame.checked_add(o).filter(|&p| p < data_end).map(|p| (s, p))
			})
			.collect();
		seek_table.sort_unstable();

		let time_base = TimeBase::new(1, info.sample_rate);
		let stream = Stream {
			id: StreamId(0),
			kind: StreamKind::Audio,
			time_base,
			duration: (info.total_samples > 0).then(|| Time::new(info.total_samples, time_base)),
			parameters: Parameters {
				codec: CodecId::new("flac"),
				profile: None,
				bitrate: None,
				sample_rate: Some(info.sample_rate),
				channels: Some(info.channels),
				width: None,
				height: None,
				pixel_format: None,
				extradata: info.to_bytes().to_vec(),
				bits_per_sample: Some(info.bits_per_sample),
			},
			metadata: Metadata::default(),
		};
		Ok(Self {
			reader,
			streams: StreamSet::new(vec![stream]),
			metadata,
			info,
			duration: Time::new(info.total_samples, time_base),
			first_frame,
			data_end,
			seek_table,
		})
	}

	/// Sample number of the first sample of a frame.
	fn first_sample(&self, h: &FrameHeader) -> u64 {
		if h.variable_block_size {
			h.number
		} else {
			let block = if self.info.min_block_size == self.info.max_block_size {
				u64::from(self.info.max_block_size)
			} else {
				u64::from(h.block_size)
			};
			h.number.saturating_mul(block)
		}
	}

	/// Moves forward to the next byte that starts a plausible frame header.
	/// Returns the header, or `None` at the end of the data.
	fn sync(&mut self) -> Result<Option<FrameHeader>> {
		loop {
			if self.reader.position() >= self.data_end {
				return Ok(None);
			}
			let window = self.reader.fill(64 * 1024)?;
			if window.len() < 6 {
				return Ok(None);
			}
			let found = (0..window.len() - 1).find_map(|i| {
				if window[i] != 0xFF || window[i + 1] & 0xFE != 0xF8 {
					return None;
				}
				FrameHeader::parse(&window[i..]).filter(|h| consistent(h, &self.info)).map(|h| (i, h))
			});
			match found {
				Some((i, h)) => {
					self.reader.consume(i);
					return Ok(Some(h));
				}
				None => {
					let advance = window.len().saturating_sub(16).max(1);
					self.reader.consume(advance);
				}
			}
		}
	}

	/// Length of the frame starting at the reader position, whose header is
	/// `h`: up to the next confirmed frame header, or to the end of the data.
	fn frame_len(&mut self, h: &FrameHeader) -> Result<usize> {
		let expected = if h.variable_block_size {
			h.number.saturating_add(u64::from(h.block_size))
		} else {
			h.number.saturating_add(1)
		};
		let mut want = (self.info.max_frame_size as usize).clamp(16 * 1024, MAX_FRAME) + 32;
		loop {
			let remaining = self.data_end.saturating_sub(self.reader.position());
			let window = self.reader.fill(want.min(usize::try_from(remaining).unwrap_or(usize::MAX)))?;
			let len = window.len();
			for i in MIN_FRAME.min(len)..len.saturating_sub(1) {
				if window[i] != 0xFF || window[i + 1] & 0xFE != 0xF8 {
					continue;
				}
				let Some(next) = FrameHeader::parse(&window[i..]) else { continue };
				if next.variable_block_size == h.variable_block_size
					&& next.number == expected
					&& consistent(&next, &self.info)
					&& crc16(&window[..i]) == 0
				{
					return Ok(i);
				}
			}
			// No following frame inside the window: either this is the last
			// frame, or the window is too small.
			if (len as u64) >= remaining || len < want || want >= MAX_FRAME {
				return Ok(len);
			}
			want = (want * 2).min(MAX_FRAME);
		}
	}

	/// Next frame as `(header, bytes)`.
	fn next_frame(&mut self) -> Result<Option<(FrameHeader, Vec<u8>)>> {
		let Some(h) = self.sync()? else { return Ok(None) };
		let len = self.frame_len(&h)?;
		let data = self.reader.read_vec(len)?;
		Ok(Some((h, data)))
	}

	/// First frame at or after `position`, as `(offset, first sample)`.
	fn probe(&mut self, position: u64) -> Result<Option<(u64, u64)>> {
		self.reader.seek(position)?;
		let Some(h) = self.sync()? else { return Ok(None) };
		Ok(Some((self.reader.position(), self.first_sample(&h))))
	}
}

impl Demuxer for FlacDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		let Some((h, data)) = self.next_frame()? else { return Ok(None) };
		let time_base = TimeBase::new(1, self.info.sample_rate);
		let pts = Time::new(self.first_sample(&h), time_base);
		let mut packet = Packet::new(StreamId(0), data);
		packet.pts = Some(pts);
		packet.dts = Some(pts);
		packet.duration = Some(Time::new(u64::from(h.block_size), time_base));
		Ok(Some(packet))
	}

	/// Seeks to the frame containing `time`: the seek table narrows the
	/// range, then a bisection over byte offsets (each probe resynchronises
	/// to a frame, whose header states its own sample number) finds the
	/// frame, as FFmpeg's `flac_seek` / generic binary search does.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let target = (time * f64::from(self.info.sample_rate)) as u64;
		let mut best_offset = self.first_frame;
		if let Some(&(_, o)) = self.seek_table.iter().rfind(|(s, _)| *s <= target) {
			best_offset = o;
		}
		let mut hi =
			self.seek_table.iter().find(|(s, _)| *s > target).map_or(self.data_end, |&(_, o)| o);
		if hi == u64::MAX {
			hi = self.reader.size().unwrap_or(best_offset);
		}
		let mut lo = best_offset;
		for _ in 0..MAX_SEEK_PROBES {
			if hi <= lo + 1 {
				break;
			}
			let mid = lo + (hi - lo) / 2;
			match self.probe(mid)? {
				Some((offset, sample)) if sample <= target && offset < hi => {
					best_offset = offset;
					lo = offset.max(mid);
				}
				_ => hi = mid,
			}
		}
		// Step forward frame by frame to the one containing the target.
		self.reader.seek(best_offset)?;
		loop {
			let start = self.reader.position();
			let Some(h) = self.sync()? else {
				self.reader.seek(start)?;
				break;
			};
			let at = self.reader.position();
			if self.first_sample(&h) + u64::from(h.block_size) > target {
				self.reader.seek(at)?;
				break;
			}
			let len = self.frame_len(&h)?;
			self.reader.skip(len as u64)?;
		}
		Ok(())
	}

	fn duration(&self) -> Time {
		self.duration
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn streams(&self) -> &StreamSet {
		&self.streams
	}
}
