//! Ogg muxer (`.ogg`, `.oga`, `.opus`) for Opus, FLAC and Vorbis.
//!
//! Each stream gets its own serial number. Pages are written in the order
//! the mappings require:
//!
//! 1. the BOS page of every stream, each holding only the identification
//!    header (`OpusHead`, the `0x7F "FLAC"` packet, the Vorbis
//!    identification header);
//! 2. the remaining header packets of every stream (`OpusTags`, FLAC
//!    metadata blocks, Vorbis comment and setup), flushed so that audio
//!    starts on a fresh page;
//! 3. audio pages, closed after one second of audio or 255 segments
//!    (FFmpeg's `page_duration`), the last one with the EOS flag.
//!
//! A page's granule position is the sample position (48 kHz for Opus, the
//! sample rate otherwise) at the end of the last packet completed on it.
//! RMP's timestamps already count the Opus pre-skip, so a packet's end is
//! `pts + duration` in those units. Packets without a known duration
//! (Vorbis, which needs a parser RMP doesn't have) end where the next
//! packet starts. On the last Opus packet, `encoder_padding` is applied as
//! end trimming unless the packet already arrived shortened.
//!
//! The comment headers are rebuilt from the stream's metadata merged with
//! the file's (cover art becomes `METADATA_BLOCK_PICTURE`, FLAC pictures
//! become PICTURE blocks). For FLAC, the first page is rewritten on
//! `finish` with the exact STREAMINFO when the output can seek.
//!
//! FFmpeg: `libavformat/oggenc.c`.

use super::demuxer::crc32;
use crate::codecs::duration::packet_samples;
use crate::codecs::flac::StreamInfo;
use crate::codecs::opus::header::OpusHead;
use crate::codecs::vorbis::VorbisParser;
use crate::container::flac::stats::FlacStats;
use crate::container::xiph;
use crate::core::{
	Metadata, Muxer, Packet, RawValue, Stream, StreamId, StreamKind, Time, TimeBase,
};
use crate::io::Io;
use crate::message::{Message, Result};

const FLAG_CONTINUED: u8 = 0x01;
const FLAG_BOS: u8 = 0x02;
const FLAG_EOS: u8 = 0x04;
const MAX_SEGMENTS: usize = 255;
const VENDOR: &str = "RMP";
/// Offset of the STREAMINFO body inside the FLAC mapping's first packet:
/// `0x7F "FLAC"`, version (2), header count (2), `"fLaC"`, block header (4).
const FLAC_STREAMINFO_AT: usize = 17;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mapping {
	Opus,
	Flac,
	Vorbis,
}

/// The page being filled for one stream.
#[derive(Default)]
struct PageBuf {
	lacing: Vec<u8>,
	body: Vec<u8>,
	/// Granule at the end of the last packet completed on this page.
	granule: Option<u64>,
	/// The page starts with the rest of a packet from the previous page.
	continued: bool,
	/// Granule at the start of the page's first packet (the previous
	/// page's granule).
	start: Option<u64>,
}

/// A packet waiting for its end position (the next packet's start, or the
/// stream end).
struct Pending {
	data: Vec<u8>,
	start: u64,
	end: Option<u64>,
	/// Arrived shorter than its coded duration (already end-trimmed).
	trimmed: bool,
}

struct OggStream {
	id: StreamId,
	serial: u32,
	sequence: u32,
	mapping: Mapping,
	/// Granule units per second.
	rate: u32,
	parameters: crate::core::Parameters,
	/// Header packets; the first goes alone on the BOS page.
	headers: Vec<Vec<u8>>,
	page: PageBuf,
	/// A closed page not yet written, held back so that the stream's last
	/// page can still get the EOS flag.
	ready: Option<PageBuf>,
	pending: Option<Pending>,
	/// Granule at the end of the last packet passed to the page.
	granule: u64,
	padding: u64,
	flac: Option<FlacStats>,
	vorbis: Option<VorbisParser>,
	/// No audio packet has been written yet.
	first: bool,
	/// `(offset, bytes)` of the FLAC BOS page, rewritten on finish.
	flac_page: Option<(u64, Vec<u8>)>,
}

pub struct OggMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	streams: Vec<OggStream>,
	opus_only: bool,
	header_written: bool,
	position: u64,
}

/// Serial numbers: fixed per stream index (FFmpeg draws random ones), so
/// RMP's output is reproducible.
fn serial_for(index: usize) -> u32 {
	0x524D_5000 ^ (index as u32).wrapping_mul(0x9E37_79B9)
}

fn string_value(metadata: &Metadata, key: &str) -> Option<String> {
	match metadata.raw.get(key) {
		Some(RawValue::String(v)) if !v.is_empty() => Some(v.clone()),
		_ => None,
	}
}

/// The stream's tags with the file's filling the gaps (FFmpeg copies the
/// file metadata into each stream with `AV_DICT_DONT_OVERWRITE`).
fn merged_metadata(stream: &Metadata, file: &Metadata) -> Metadata {
	let mut m = stream.clone();
	macro_rules! fill {
		($($field:ident),*) => { $( if m.$field.is_none() { m.$field = file.$field.clone(); } )* };
	}
	fill!(
		title,
		description,
		artist,
		album,
		album_artist,
		track_number,
		tracks_total,
		disc_number,
		discs_total,
		genre,
		date,
		lyrics,
		comment
	);
	if m.images.is_empty() {
		m.images = file.images.clone();
	}
	for (k, v) in &file.raw {
		m.raw.entry(k.clone()).or_insert_with(|| v.clone());
	}
	m
}

/// Header packets for a stream, identification header first.
fn build_headers(mapping: Mapping, stream: &Stream, metadata: &Metadata) -> Result<Vec<Vec<u8>>> {
	let p = &stream.parameters;
	match mapping {
		Mapping::Opus => {
			let head = if OpusHead::parse(&p.extradata).is_some() {
				p.extradata.clone()
			} else {
				let channels = p.channels.ok_or(Message::Container("opus stream has no channel count"))?;
				if channels > 2 {
					return Err(Message::Container("multichannel opus needs its OpusHead extradata"));
				}
				let delay = string_value(&stream.metadata, "encoder_delay")
					.and_then(|v| v.parse::<u16>().ok())
					.unwrap_or(0);
				OpusHead::new(channels, delay, 48_000).to_bytes()
			};
			let mut tags = b"OpusTags".to_vec();
			tags.extend(xiph::write_comments_with(metadata, VENDOR, true));
			Ok(vec![head, tags])
		}
		Mapping::Flac => {
			let info = StreamInfo::from_extradata(&p.extradata)
				.ok_or(Message::Container("flac stream has no STREAMINFO extradata"))?;
			let mut blocks: Vec<(u8, Vec<u8>)> =
				vec![(4, xiph::write_comments_with(metadata, VENDOR, false))];
			for image in &metadata.images {
				blocks.push((6, xiph::write_picture(image)));
			}
			let mut first = vec![0x7F];
			first.extend_from_slice(b"FLAC");
			first.extend_from_slice(&[1, 0]);
			first.extend_from_slice(&(blocks.len() as u16).to_be_bytes());
			first.extend_from_slice(b"fLaC");
			first.extend(info.to_block(false));
			let mut headers = vec![first];
			let count = blocks.len();
			for (i, (kind, body)) in blocks.into_iter().enumerate() {
				let len = u32::try_from(body.len()).ok().filter(|&l| l < 1 << 24);
				let len = len.ok_or(Message::Container("flac metadata block too large"))?;
				let last = if i + 1 == count { 0x80 } else { 0 };
				let mut packet = vec![kind | last];
				packet.extend_from_slice(&len.to_be_bytes()[1..]);
				packet.extend(body);
				headers.push(packet);
			}
			Ok(headers)
		}
		Mapping::Vorbis => {
			let mut headers = xiph::vorbis_headers(&p.extradata)
				.ok_or(Message::Container("vorbis stream has no header extradata"))?;
			if headers[0].get(..7) != Some(b"\x01vorbis") || headers[2].get(..7) != Some(b"\x05vorbis") {
				return Err(Message::Container("invalid vorbis headers"));
			}
			let mut comment = b"\x03vorbis".to_vec();
			comment.extend(xiph::write_comments_with(metadata, VENDOR, true));
			comment.push(1); // framing bit
			headers[1] = comment;
			Ok(headers)
		}
	}
}

/// Lacing values of one packet: 255s, then the remainder (0 when the
/// length is a multiple of 255, so the packet still terminates).
fn laces(len: usize) -> impl Iterator<Item = u8> {
	std::iter::repeat_n(255u8, len / 255).chain(std::iter::once((len % 255) as u8))
}

impl OggStream {
	fn granule_base(&self) -> TimeBase {
		TimeBase::new(1, self.rate)
	}

	fn to_granule(&self, t: Option<Time>) -> Option<u64> {
		t.and_then(|t| t.rescale(self.granule_base())).map(|t| t.ticks())
	}
}

impl OggMuxer {
	pub fn new(io: Box<dyn Io>, opus_only: bool) -> Self {
		Self {
			io,
			metadata: Metadata::default(),
			streams: Vec::new(),
			opus_only,
			header_written: false,
			position: 0,
		}
	}

	/// Writes one page. Returns its offset and bytes.
	fn write_page(
		&mut self,
		index: usize,
		flags: u8,
		granule: Option<u64>,
		lacing: &[u8],
		body: &[u8],
	) -> Result<(u64, Vec<u8>)> {
		let s = &mut self.streams[index];
		let mut page = Vec::with_capacity(27 + lacing.len() + body.len());
		page.extend_from_slice(b"OggS");
		page.push(0);
		page.push(flags);
		let granule = granule.map_or(-1i64, |g| i64::try_from(g).unwrap_or(i64::MAX));
		page.extend_from_slice(&granule.to_le_bytes());
		page.extend_from_slice(&s.serial.to_le_bytes());
		page.extend_from_slice(&s.sequence.to_le_bytes());
		page.extend_from_slice(&[0; 4]);
		page.push(lacing.len() as u8);
		page.extend_from_slice(lacing);
		page.extend_from_slice(body);
		let crc = crc32(&page);
		page[22..26].copy_from_slice(&crc.to_le_bytes());
		s.sequence = s.sequence.wrapping_add(1);
		let offset = self.position;
		self.io.write_all(&page)?;
		self.position += page.len() as u64;
		Ok((offset, page))
	}

	/// Writes a page of stream `index`.
	fn emit_page(&mut self, index: usize, page: PageBuf, extra_flags: u8) -> Result<()> {
		let mut flags = extra_flags;
		if page.continued {
			flags |= FLAG_CONTINUED;
		}
		let granule =
			if page.lacing.is_empty() { Some(self.streams[index].granule) } else { page.granule };
		self.write_page(index, flags, granule, &page.lacing, &page.body)?;
		Ok(())
	}

	/// Closes the page being filled: the previously closed page is written
	/// and this one is held back.
	fn close_page(&mut self, index: usize) -> Result<()> {
		let page = std::mem::take(&mut self.streams[index].page);
		if page.lacing.is_empty() {
			return Ok(());
		}
		if let Some(ready) = self.streams[index].ready.replace(page) {
			self.emit_page(index, ready, 0)?;
		}
		Ok(())
	}

	/// Writes the held-back page, if any (`eos`: as the stream's last page;
	/// a stream with no page left gets an empty EOS page).
	fn drain(&mut self, index: usize, eos: bool) -> Result<()> {
		let flags = if eos { FLAG_EOS } else { 0 };
		match self.streams[index].ready.take() {
			Some(ready) => self.emit_page(index, ready, flags),
			None if eos => self.emit_page(index, PageBuf::default(), flags),
			None => Ok(()),
		}
	}

	/// Appends a packet spanning `start..end` (granules) to the stream's
	/// page, splitting it across pages when it runs past 255 segments.
	fn push_packet(&mut self, index: usize, data: &[u8], start: u64, end: u64) -> Result<()> {
		// A packet that would not fit is started on a new page rather than
		// split, unless it needs more than a page anyway (as FFmpeg does).
		let segments = data.len() / 255 + 1;
		let used = self.streams[index].page.lacing.len();
		if used > 0 && used + segments > MAX_SEGMENTS && segments <= MAX_SEGMENTS {
			self.close_page(index)?;
		}
		let mut offset = 0usize;
		let mut first_lace = true;
		for lace in laces(data.len()) {
			if self.streams[index].page.lacing.len() >= MAX_SEGMENTS {
				self.close_page(index)?;
				// The packet continues on the next page.
				self.streams[index].page.continued = !first_lace;
			}
			let page = &mut self.streams[index].page;
			page.start.get_or_insert(start);
			page.lacing.push(lace);
			page.body.extend_from_slice(&data[offset..offset + usize::from(lace)]);
			offset += usize::from(lace);
			first_lace = false;
		}
		let s = &mut self.streams[index];
		let end = end.max(s.granule);
		s.granule = end;
		s.page.granule = Some(end);
		let start = s.page.start.unwrap_or(start);
		// FFmpeg's `page_duration`: close the page once it spans a second.
		if end.saturating_sub(start) >= u64::from(s.rate) || s.page.lacing.len() >= MAX_SEGMENTS {
			self.close_page(index)?;
		}
		Ok(())
	}

	fn write_headers(&mut self) -> Result<()> {
		if self.streams.is_empty() {
			return Err(Message::Container("ogg muxer has no stream"));
		}
		for i in 0..self.streams.len() {
			let first = self.streams[i].headers[0].clone();
			let lacing: Vec<u8> = laces(first.len()).collect();
			if lacing.len() > MAX_SEGMENTS {
				return Err(Message::Container("ogg identification header too large"));
			}
			let (offset, page) = self.write_page(i, FLAG_BOS, Some(0), &lacing, &first)?;
			if self.streams[i].mapping == Mapping::Flac {
				self.streams[i].flac_page = Some((offset, page));
			}
		}
		for i in 0..self.streams.len() {
			let headers: Vec<Vec<u8>> = self.streams[i].headers[1..].to_vec();
			for header in &headers {
				self.push_packet(i, header, 0, 0)?;
			}
			// Audio must start on a new page.
			self.close_page(i)?;
			self.drain(i, false)?;
		}
		self.header_written = true;
		Ok(())
	}

	/// Passes a pending packet on to the page, once its end is known.
	fn release(&mut self, index: usize, pending: Pending, last: bool) -> Result<()> {
		let s = &self.streams[index];
		let mut end = pending.end.unwrap_or(pending.start);
		if last && s.mapping == Mapping::Opus && !pending.trimmed && s.padding > 0 {
			end = end.saturating_sub(s.padding).max(pending.start);
		}
		self.push_packet(index, &pending.data, pending.start, end)
	}

	/// Rewrites the FLAC BOS pages with the final STREAMINFO.
	fn patch_flac(&mut self) -> Result<()> {
		let end = self.position;
		let mut seeked = false;
		for s in &mut self.streams {
			let (Some(stats), Some((offset, page))) = (s.flac.as_mut(), s.flac_page.as_mut()) else {
				continue;
			};
			let info = stats.finish().to_bytes();
			let body_at = 27 + usize::from(page[26]);
			let at = body_at + FLAC_STREAMINFO_AT;
			let Some(slot) = page.get_mut(at..at + info.len()) else { continue };
			slot.copy_from_slice(&info);
			page[22..26].fill(0);
			let crc = crc32(page);
			page[22..26].copy_from_slice(&crc.to_le_bytes());
			if self.io.seek(*offset).is_err() {
				return Ok(());
			}
			seeked = true;
			self.io.write_all(page)?;
		}
		if seeked {
			self.io.seek(end)?;
		}
		Ok(())
	}
}

impl Muxer for OggMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.header_written {
			return Err(Message::Container("ogg streams must be added before the first packet"));
		}
		if self.streams.iter().any(|s| s.id == stream.id) {
			return Err(Message::Container("ogg muxer: duplicate stream id"));
		}
		if stream.kind != StreamKind::Audio {
			return Err(Message::Container("ogg muxer accepts only audio (opus, flac, vorbis)"));
		}
		let p = &stream.parameters;
		let mapping = match p.codec.id.as_str() {
			"opus" => Mapping::Opus,
			"flac" if !self.opus_only => Mapping::Flac,
			"vorbis" if !self.opus_only => Mapping::Vorbis,
			_ if self.opus_only => return Err(Message::Container(".opus files hold only opus audio")),
			_ => return Err(Message::Container("ogg muxer accepts opus, flac and vorbis")),
		};
		let rate = match mapping {
			Mapping::Opus => 48_000,
			_ => p
				.sample_rate
				.filter(|&r| r > 0)
				.ok_or(Message::Container("ogg stream has no sample rate"))?,
		};
		let metadata = merged_metadata(&stream.metadata, &self.metadata);
		let headers = build_headers(mapping, stream, &metadata)?;
		let flac = (mapping == Mapping::Flac)
			.then(|| StreamInfo::from_extradata(&p.extradata))
			.flatten()
			.map(FlacStats::new);
		let vorbis =
			(mapping == Mapping::Vorbis).then(|| VorbisParser::new(&headers[0], &headers[2])).flatten();
		let padding = string_value(&stream.metadata, "encoder_padding")
			.and_then(|v| v.parse::<u64>().ok())
			.unwrap_or(0);
		let index = self.streams.len();
		self.streams.push(OggStream {
			id: stream.id,
			serial: serial_for(index),
			sequence: 0,
			mapping,
			rate,
			parameters: p.clone(),
			headers,
			page: PageBuf::default(),
			ready: None,
			pending: None,
			granule: 0,
			padding,
			flac,
			vorbis,
			first: true,
			flac_page: None,
		});
		Ok(index)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if !self.header_written {
			self.write_headers()?;
		}
		let index = self
			.streams
			.iter()
			.position(|s| s.id == packet.stream_id)
			.ok_or(Message::Container("packet for a stream that was not added to the ogg muxer"))?;
		if packet.data.is_empty() {
			return Ok(());
		}
		let s = &mut self.streams[index];
		if let Some(stats) = s.flac.as_mut() {
			stats.add(&packet.data);
		}
		let coded = match s.vorbis.as_mut() {
			Some(parser) => parser.packet_samples(&packet.data).map(u64::from),
			None => packet_samples(&s.parameters, &packet.data).and_then(|(n, rate)| {
				crate::core::rescale_ticks(n, TimeBase::new(1, rate), s.granule_base())
			}),
		};
		let mut start = s.to_granule(packet.pts.or(packet.dts));
		if s.first && start.is_none() && s.mapping == Mapping::Vorbis {
			// The first Vorbis packet only primes the decoder: it ends at
			// granule 0 (FFmpeg gives it a negative pts).
			start = Some(0);
		}
		let first_vorbis = s.first && s.mapping == Mapping::Vorbis && packet.pts.is_none();
		s.first = false;
		let duration = s.to_granule(packet.duration).or(coded);
		let trimmed = matches!((s.to_granule(packet.duration), coded), (Some(d), Some(c)) if d < c);

		// The previous packet ends where this one starts, if it had no
		// duration of its own.
		let previous_end = s.pending.as_ref().map(|p| p.end);
		// Where the previous packet ends, from exact sample counts.
		let running = s.pending.as_ref().and_then(|p| p.end);
		// A timestamp that only differs from the running position by its own
		// rounding (Matroska stores 1 ms ticks: up to 48 Opus samples) is
		// the same instant; a larger difference is a real gap.
		let tolerance = packet
			.pts
			.or(packet.dts)
			.and_then(|t| crate::core::rescale_ticks(1, t.base(), s.granule_base()))
			.unwrap_or(0);
		let start = match (start, running) {
			(Some(t), Some(r)) if t.abs_diff(r) <= tolerance => r,
			(Some(t), _) => t,
			(None, Some(r)) => r,
			(None, None) => s.pending.as_ref().map_or(s.granule, |p| p.start),
		};
		if let Some(mut previous) = s.pending.take() {
			if previous_end.flatten().is_none() {
				previous.end = Some(start.max(previous.start));
			}
			self.release(index, previous, false)?;
		}
		let end = if first_vorbis { Some(start) } else { duration.map(|d| start.saturating_add(d)) };
		self.streams[index].pending = Some(Pending { data: packet.data, start, end, trimmed });
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		if self.streams.is_empty() {
			return Ok(());
		}
		if !self.header_written {
			self.write_headers()?;
		}
		for i in 0..self.streams.len() {
			if let Some(pending) = self.streams[i].pending.take() {
				self.release(i, pending, true)?;
			}
			self.close_page(i)?;
			self.drain(i, true)?;
		}
		self.patch_flac()
	}
}
