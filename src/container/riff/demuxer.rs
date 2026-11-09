use crate::Message;
use crate::codecs::adpcm::AdpcmCodec;
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, Stream, StreamId, StreamSet, Time, TimeBase,
};
use crate::io::Io;
use crate::message::Result;

/// Largest `fmt ` chunk accepted; real ones are 16-40 bytes (plus 32 bytes
/// of coefficients for MS ADPCM).
const MAX_FMT_SIZE: u64 = 64 * 1024;
/// Largest `LIST` chunk parsed for tags; bigger ones are skipped unread.
const MAX_LIST_SIZE: u64 = 1024 * 1024;
/// Target payload of one PCM packet (FFmpeg's wav demuxer uses 4096 too).
const PCM_PACKET_BYTES: usize = 4096;

/// How `data` is cut into packets and timed.
#[derive(Debug, Clone, Copy)]
enum Layout {
	/// Packets of whole sample frames; `block_align` bytes per sample frame.
	Pcm,
	/// One self-contained block per packet, `samples_per_block` samples each.
	Adpcm { codec: AdpcmCodec, samples_per_block: u64 },
}

pub struct RiffDemuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	streams: StreamSet,
	data_start: u64,
	size: u64,
	remaining: u64,
	byte_rate: u32,
	block_align: u16,
	layout: Layout,
	time_base: TimeBase,
	/// Total samples from the `fact` chunk, if any (ADPCM duration).
	fact_samples: Option<u64>,
}

/// The parsed `fmt ` chunk (WAVEFORMATEX).
struct Fmt {
	format_tag: u16,
	channels: u16,
	sample_rate: u32,
	byte_rate: u32,
	block_align: u16,
	bits_per_sample: u16,
	/// Bytes after `cbSize` (FFmpeg's `extradata` for WAV codecs).
	extension: Vec<u8>,
}

fn le16(bytes: &[u8], at: usize) -> Option<u16> {
	Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn le32(bytes: &[u8], at: usize) -> Option<u32> {
	Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

fn parse_fmt(fmt: &[u8]) -> Result<Fmt> {
	let bad = || Message::Container("invalid wav fmt chunk");
	let mut parsed = Fmt {
		format_tag: le16(fmt, 0).ok_or_else(bad)?,
		channels: le16(fmt, 2).ok_or_else(bad)?,
		sample_rate: le32(fmt, 4).ok_or_else(bad)?,
		byte_rate: le32(fmt, 8).ok_or_else(bad)?,
		block_align: le16(fmt, 12).ok_or_else(bad)?,
		bits_per_sample: le16(fmt, 14).ok_or_else(bad)?,
		extension: Vec::new(),
	};
	if let Some(cb_size) = le16(fmt, 16) {
		// A cbSize past the chunk end is clamped, as FFmpeg does.
		let end = (18 + usize::from(cb_size)).min(fmt.len());
		parsed.extension = fmt.get(18..end).unwrap_or_default().to_vec();
	}
	Ok(parsed)
}

impl RiffDemuxer {
	pub fn new(mut io: Box<dyn Io>) -> Result<Self> {
		let mut riff_header = [0u8; 12];
		read_exact(&mut *io, &mut riff_header)?;
		if &riff_header[0..4] != b"RIFF" || &riff_header[8..12] != b"WAVE" {
			return Err(Message::Container("invalid wav header"));
		}

		let mut offset = 12u64;
		let mut metadata = Metadata::default();
		let mut fmt = None;
		let mut fact_samples = None;
		let mut data = None;

		loop {
			let mut chunk_header = [0u8; 8];
			if read_exact_or_eof(&mut *io, &mut chunk_header)? == 0 {
				break;
			}
			offset += 8;

			let chunk_id = &chunk_header[0..4];
			let chunk_size = u64::from(le32(&chunk_header, 4).unwrap_or(0));
			let padded_size = chunk_size + (chunk_size & 1);

			if chunk_id == b"data" {
				data = Some((offset, chunk_size));
				break;
			}

			let wanted = match chunk_id {
				b"fmt " => {
					if !(16..=MAX_FMT_SIZE).contains(&chunk_size) {
						return Err(Message::Container("invalid wav fmt chunk"));
					}
					true
				}
				b"fact" => (4..=64).contains(&chunk_size),
				b"LIST" => chunk_size <= MAX_LIST_SIZE,
				_ => false,
			};
			if !wanted {
				skip_exact(&mut *io, padded_size)?;
				offset += padded_size;
				continue;
			}

			// Sizes are bounded above, so this allocation is small.
			let mut body = vec![0u8; chunk_size as usize];
			read_exact(&mut *io, &mut body)?;
			match chunk_id {
				b"fmt " => fmt = Some(parse_fmt(&body)?),
				b"fact" => fact_samples = le32(&body, 0).map(u64::from),
				_ => parse_list_chunk(&body, &mut metadata),
			}
			if padded_size > chunk_size {
				skip_exact(&mut *io, padded_size - chunk_size)?;
			}
			offset += padded_size;
		}

		let fmt = fmt.ok_or(Message::Container("wav fmt chunk not found"))?;
		let (data_start, mut size) = data.ok_or(Message::Container("wav data chunk not found"))?;
		// Streamed or truncated files declare more data than they hold.
		if let Ok(file_size) = io.size() {
			size = size.min(file_size.saturating_sub(data_start));
		}

		if fmt.channels == 0 || fmt.channels > u16::from(u8::MAX) {
			return Err(Message::Container("invalid wav channel count"));
		}
		if fmt.sample_rate == 0 {
			return Err(Message::Container("invalid wav sample rate"));
		}
		if fmt.block_align == 0 {
			return Err(Message::Container("invalid wav block align"));
		}
		if fmt.byte_rate == 0 {
			return Err(Message::Container("invalid wav byte rate"));
		}
		if fmt.bits_per_sample == 0 {
			return Err(Message::Container("invalid wav bits per sample"));
		}

		let (codec, layout) = resolve_codec(&fmt)?;
		let channels = fmt.channels as u8;

		let mut stream = Stream::audio(StreamId(0), fmt.sample_rate, channels, codec);
		stream.parameters.bitrate = fmt.byte_rate.checked_mul(8);
		if matches!(layout, Layout::Adpcm { .. }) {
			stream.parameters.extradata = fmt.extension.clone();
		}
		let time_base = stream.time_base;

		let mut demuxer = Self {
			io,
			metadata,
			streams: StreamSet::default(),
			data_start,
			size,
			remaining: size,
			byte_rate: fmt.byte_rate,
			block_align: fmt.block_align,
			layout,
			time_base,
			fact_samples,
		};
		stream.duration = Some(demuxer.duration());
		demuxer.streams.add(stream);
		Ok(demuxer)
	}

	/// Samples per channel before `offset` bytes of `data`.
	fn samples_before(&self, offset: u64) -> u64 {
		let align = u64::from(self.block_align);
		match self.layout {
			Layout::Pcm => offset / align,
			Layout::Adpcm { samples_per_block, .. } => offset / align * samples_per_block,
		}
	}

	fn total_samples(&self) -> u64 {
		match self.layout {
			Layout::Pcm => self.samples_before(self.size),
			Layout::Adpcm { codec, .. } => {
				let whole = self.samples_before(self.size);
				let tail = (self.size % u64::from(self.block_align)) as usize;
				let channels = self.streams.streams.first().and_then(|s| s.parameters.channels);
				let partial =
					channels.and_then(|ch| codec.samples_in_block(tail, usize::from(ch))).unwrap_or(0) as u64;
				let decoded = whole + partial;
				// `fact` holds the exact length; the last block is padded.
				self.fact_samples.filter(|n| *n > 0 && *n <= decoded).unwrap_or(decoded)
			}
		}
	}
}

/// Maps the fmt chunk to a codec ID and packet layout, validating the fields
/// that layout depends on.
fn resolve_codec(fmt: &Fmt) -> Result<(CodecId, Layout)> {
	let tag =
		if fmt.format_tag == 0xFFFE { resolve_extensible_subformat(fmt)? } else { fmt.format_tag };
	let channels = usize::from(fmt.channels);

	if let Some(codec) = [AdpcmCodec::ImaWav, AdpcmCodec::Ms].into_iter().find(|c| c.wav_tag() == tag)
	{
		if fmt.bits_per_sample != 4 {
			return Err(Message::Container("unsupported adpcm bits per sample"));
		}
		if !codec.supports_channels(channels) {
			return Err(Message::Container("unsupported adpcm channel count"));
		}
		let samples_per_block = codec
			.samples_in_block(usize::from(fmt.block_align), channels)
			.ok_or(Message::Container("invalid adpcm block align"))?;
		return Ok((
			CodecId::new(codec.id()),
			Layout::Adpcm { codec, samples_per_block: samples_per_block as u64 },
		));
	}

	let id = match (tag, fmt.bits_per_sample) {
		(1, 8) => "pcm_u8",
		(1, 16) => "pcm_s16le",
		(1, 24) => "pcm_s24le",
		(1, 32) => "pcm_s32le",
		(1, 64) => "pcm_s64le",
		(1, _) => return Err(Message::Container("unsupported wav pcm bit depth")),
		(3, 32) => "pcm_f32le",
		(3, 64) => "pcm_f64le",
		(3, _) => return Err(Message::Container("unsupported wav float bit depth")),
		(6, 8) => "pcm_alaw",
		(7, 8) => "pcm_mulaw",
		(6 | 7, _) => return Err(Message::Container("unsupported g711 bit depth")),
		_ => return Err(Message::Container("unsupported wav format")),
	};

	let bytes_per_sample = u32::from(fmt.bits_per_sample.div_ceil(8));
	let expected_block_align = u32::from(fmt.channels) * bytes_per_sample;
	if u32::from(fmt.block_align) != expected_block_align {
		return Err(Message::Container("inconsistent wav block align"));
	}
	if u64::from(fmt.byte_rate) != u64::from(fmt.sample_rate) * u64::from(expected_block_align) {
		return Err(Message::Container("inconsistent wav byte rate"));
	}
	Ok((CodecId::new(id), Layout::Pcm))
}

fn parse_list_chunk(list: &[u8], metadata: &mut Metadata) {
	if list.len() < 4 || &list[..4] != b"INFO" {
		return;
	}

	let mut offset = 4usize;
	while offset + 8 <= list.len() {
		let id = &list[offset..offset + 4];
		let Some(size) = le32(list, offset + 4).map(|n| n as usize) else {
			break;
		};
		offset += 8;

		if size > list.len() - offset {
			break;
		}

		let value = decode_info_string(&list[offset..offset + size]);
		apply_info_tag(metadata, id, value);

		offset += size + (size & 1);
	}
}

fn decode_info_string(bytes: &[u8]) -> String {
	let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
	String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

fn apply_info_tag(metadata: &mut Metadata, id: &[u8], value: String) {
	if value.is_empty() {
		return;
	}

	match id {
		b"INAM" => metadata.title = Some(value),
		b"IART" => metadata.artist = Some(value),
		b"IPRD" => metadata.album = Some(value),
		b"ICMT" => metadata.comment = Some(value),
		b"IGNR" => metadata.genre = Some(value),
		b"ICRD" => metadata.date = Some(value),
		b"ITRK" => {
			metadata.track_number = value.parse().ok();
			metadata.raw.insert("ITRK".into(), crate::core::RawValue::String(value));
		}
		_ => {
			let key = String::from_utf8_lossy(id).to_string();
			metadata.raw.insert(key, crate::core::RawValue::String(value));
		}
	}
}

impl Demuxer for RiffDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		if self.remaining == 0 {
			return Ok(None);
		}

		let align = usize::from(self.block_align.max(1));
		let remaining = usize::try_from(self.remaining).unwrap_or(usize::MAX);
		let size = match self.layout {
			// Whole sample frames, about PCM_PACKET_BYTES per packet.
			Layout::Pcm => {
				let target = (PCM_PACKET_BYTES / align).max(1) * align;
				target.min(remaining)
			}
			Layout::Adpcm { .. } => align.min(remaining),
		};

		let offset = self.size - self.remaining;
		let mut data = vec![0u8; size];
		let mut filled = 0;
		while filled < size {
			let n = self.io.read(&mut data[filled..])?;
			if n == 0 {
				break;
			}
			filled += n;
		}
		if filled == 0 {
			self.remaining = 0;
			return Ok(None);
		}
		data.truncate(filled);
		self.remaining -= filled as u64;

		let pts = Time::new(self.samples_before(offset), self.time_base);
		let mut packet = Packet::new(StreamId(0), data);
		packet.pts = Some(pts);
		packet.dts = Some(pts);
		Ok(Some(packet))
	}

	fn streams(&self) -> &StreamSet {
		&self.streams
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	/// Seeks to the sample frame (PCM) or block (ADPCM) containing `time`.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err("invalid seek time".into());
		}

		let align = u64::from(self.block_align.max(1));
		let rate = f64::from(self.time_base.den) / f64::from(self.time_base.num.max(1));
		let raw_offset = match self.layout {
			Layout::Pcm => (time * f64::from(self.byte_rate)).floor() as u64,
			Layout::Adpcm { samples_per_block, .. } => {
				let block = (time * rate).floor() as u64 / samples_per_block.max(1);
				block.saturating_mul(align)
			}
		};
		let aligned_offset = (raw_offset / align) * align;
		let clamped_offset = aligned_offset.min(self.size);

		self.io.seek(self.data_start + clamped_offset)?;
		self.remaining = self.size.saturating_sub(clamped_offset);
		Ok(())
	}

	fn duration(&self) -> Time {
		Time::new(self.total_samples(), self.time_base)
	}
}

fn read_exact(io: &mut dyn Io, mut buf: &mut [u8]) -> Result<()> {
	while !buf.is_empty() {
		let n = io.read(buf)?;
		if n == 0 {
			return Err(Message::Container("unexpected end of file"));
		}
		let (_, rest) = buf.split_at_mut(n);
		buf = rest;
	}
	Ok(())
}

fn read_exact_or_eof(io: &mut dyn Io, mut buf: &mut [u8]) -> Result<usize> {
	let mut total = 0;
	while !buf.is_empty() {
		let n = io.read(buf)?;
		if n == 0 {
			if total == 0 {
				return Ok(0);
			}
			return Err(Message::Container("unexpected end of file"));
		}
		total += n;
		let (_, rest) = buf.split_at_mut(n);
		buf = rest;
	}
	Ok(total)
}

fn skip_exact(io: &mut dyn Io, mut size: u64) -> Result<()> {
	let mut buf = [0u8; 1024];
	while size > 0 {
		let chunk = (buf.len() as u64).min(size) as usize;
		let n = io.read(&mut buf[..chunk])?;
		if n == 0 {
			return Err(Message::Container("unexpected end of file"));
		}
		size -= n as u64;
	}
	Ok(())
}

fn resolve_extensible_subformat(fmt: &Fmt) -> Result<u16> {
	// WAVEFORMATEXTENSIBLE: 22 extension bytes; the sub-format GUID starts at
	// byte 6 of the extension and its first two bytes are the format tag.
	if fmt.extension.len() < 22 {
		return Err(Message::Container("invalid wav extensible chunk"));
	}
	le16(&fmt.extension, 6).ok_or(Message::Container("invalid wav extensible chunk"))
}
