//! Ogg demuxer (`.ogg`, `.opus`, `.oga`).
//!
//! An Ogg file is a sequence of pages (`"OggS"`, a 27-byte header, a lacing
//! table and the payload, protected by a CRC-32). Each logical stream has
//! its own serial number; packets are split into 255-byte segments across
//! pages. A page's granule position is the codec's sample position at the
//! end of the last packet that completes on it, so packet timestamps are
//! found by walking back from the granule with each packet's duration,
//! as FFmpeg's `ogg_calc_pts` does.
//!
//! Supported mappings: Opus (RFC 7845), FLAC (the `0x7F "FLAC"` mapping)
//! and Vorbis (demux only; RMP has no Vorbis decoder). Other logical
//! streams are skipped.
//!
//! Timestamps follow RMP's gapless convention: they start at 0 and the Opus
//! pre-skip is reported as `encoder_delay` metadata (FFmpeg instead starts
//! the first packet at `-pre_skip`); end trimming shows up as a shorter
//! last packet and `encoder_padding`.
//!
//! FFmpeg: `libavformat/oggdec.c`, `oggparseopus.c`, `oggparseflac.c`,
//! `oggparsevorbis.c`.

use std::collections::VecDeque;

use crate::codecs::duration::packet_samples;
use crate::codecs::flac::StreamInfo;
use crate::codecs::opus::header::OpusHead;
use crate::container::xiph;
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, Parameters, RawValue, Stream, StreamId, StreamKind,
	StreamSet, Time, TimeBase,
};
use crate::io::{BufferedReader, Io};
use crate::message::{Message, Result};

const HEADER_LEN: usize = 27;
const FLAG_CONTINUED: u8 = 0x01;
const FLAG_BOS: u8 = 0x02;
/// Largest packet kept (a 16 MiB cover-art comment header is the realistic
/// worst case); bigger ones are dropped as damage.
const MAX_PACKET: usize = 16 * 1024 * 1024;
/// Most logical streams accepted.
const MAX_STREAMS: usize = 32;
/// Pages read while looking for the codec headers.
const MAX_HEADER_PAGES: usize = 4096;
/// How far back from the end the last granule position is searched.
const TAIL_SEARCH: u64 = 256 * 1024;
const MAX_SEEK_PROBES: usize = 48;
/// Opus decoders need 80 ms to converge after a seek (RFC 7845 §4.6).
const OPUS_PREROLL: u64 = 3840;

const fn crc_table() -> [u32; 256] {
	let mut table = [0u32; 256];
	let mut i = 0;
	while i < 256 {
		let mut r = (i as u32) << 24;
		let mut k = 0;
		while k < 8 {
			r = if r & 0x8000_0000 != 0 { (r << 1) ^ 0x04C1_1DB7 } else { r << 1 };
			k += 1;
		}
		table[i] = r;
		i += 1;
	}
	table
}

static CRC_TABLE: [u32; 256] = crc_table();

/// Ogg's CRC-32 (polynomial 0x04C11DB7, no reflection, initial value 0).
pub fn crc32(data: &[u8]) -> u32 {
	data.iter().fold(0u32, |crc, &b| (crc << 8) ^ CRC_TABLE[((crc >> 24) as u8 ^ b) as usize])
}

/// One page, parsed and CRC-checked.
#[derive(Debug, Clone)]
pub struct Page {
	pub flags: u8,
	/// `None` when no packet completes on the page (granule `-1`).
	pub granule: Option<u64>,
	pub serial: u32,
	pub lacing: Vec<u8>,
	pub body: Vec<u8>,
}

/// Parses a page from the start of `data`; returns it and its total length.
pub fn parse_page(data: &[u8]) -> Option<(Page, usize)> {
	if data.len() < HEADER_LEN || &data[..4] != b"OggS" || data[4] != 0 {
		return None;
	}
	let segments = usize::from(data[26]);
	let lacing = data.get(HEADER_LEN..HEADER_LEN + segments)?;
	let body_len: usize = lacing.iter().map(|&l| usize::from(l)).sum();
	let total = HEADER_LEN + segments + body_len;
	let page = data.get(..total)?;
	let stored = u32::from_le_bytes([page[22], page[23], page[24], page[25]]);
	let mut copy = page.to_vec();
	copy[22..26].fill(0);
	if crc32(&copy) != stored {
		return None;
	}
	let granule = i64::from_le_bytes(page[6..14].try_into().ok()?);
	Some((
		Page {
			flags: page[5],
			granule: u64::try_from(granule).ok(),
			serial: u32::from_le_bytes([page[14], page[15], page[16], page[17]]),
			lacing: lacing.to_vec(),
			body: page[HEADER_LEN + segments..].to_vec(),
		},
		total,
	))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mapping {
	Opus,
	Flac,
	Vorbis,
}

/// Per logical stream state.
struct Logical {
	serial: u32,
	mapping: Mapping,
	index: usize,
	parameters: Parameters,
	metadata: Metadata,
	/// Header packets still expected (`None`: until the first audio packet,
	/// for FLAC files that do not state the count).
	headers_left: Option<usize>,
	/// Vorbis headers, kept to build FFmpeg's xiph-laced extradata.
	vorbis_headers: Vec<Vec<u8>>,
	pre_skip: u64,
	partial: Vec<u8>,
	/// The partial packet's start was lost (after a seek or a damaged page).
	partial_broken: bool,
	/// Ticks of the end of the last packet emitted, for pages whose packets
	/// have unknown durations.
	next_pts: Option<u64>,
}

impl Logical {
	fn rate(&self) -> u32 {
		self.parameters.sample_rate.filter(|&r| r > 0).unwrap_or(48_000)
	}
}

pub struct OggDemuxer {
	reader: BufferedReader,
	streams: StreamSet,
	metadata: Metadata,
	logical: Vec<Logical>,
	queue: VecDeque<Packet>,
	duration: Time,
	data_start: u64,
	data_end: u64,
}

fn new_logical(first: &[u8], serial: u32, index: usize) -> Option<Logical> {
	let mut parameters = Parameters {
		codec: CodecId::new("none"),
		profile: None,
		bitrate: None,
		sample_rate: None,
		channels: None,
		width: None,
		height: None,
		pixel_format: None,
		extradata: Vec::new(),
		bits_per_sample: None,
	};
	let mut metadata = Metadata::default();
	let (mapping, headers_left, pre_skip, vorbis_headers) = if first.starts_with(b"OpusHead") {
		let head = OpusHead::parse(first)?;
		parameters.codec = CodecId::new("opus");
		parameters.sample_rate = Some(48_000);
		parameters.channels = Some(head.channels);
		parameters.extradata = first.to_vec();
		let pre_skip = u64::from(head.pre_skip);
		if pre_skip > 0 {
			metadata.raw.insert("encoder_delay".into(), RawValue::String(pre_skip.to_string()));
		}
		(Mapping::Opus, Some(1), pre_skip, Vec::new())
	} else if first.len() >= 51
		&& first[0] == 0x7F
		&& &first[1..5] == b"FLAC"
		&& &first[9..13] == b"fLaC"
	{
		let info = StreamInfo::parse(&first[17..51])?;
		parameters.codec = CodecId::new("flac");
		parameters.sample_rate = Some(info.sample_rate);
		parameters.channels = Some(info.channels);
		parameters.bits_per_sample = Some(info.bits_per_sample);
		parameters.extradata = info.to_bytes().to_vec();
		let count = usize::from(u16::from_be_bytes([first[7], first[8]]));
		(Mapping::Flac, (count > 0).then_some(count), 0, Vec::new())
	} else if first.len() >= 30 && first[0] == 1 && &first[1..7] == b"vorbis" {
		parameters.codec = CodecId::new("vorbis");
		parameters.channels = Some(first[11]);
		let rate = u32::from_le_bytes([first[12], first[13], first[14], first[15]]);
		parameters.sample_rate = Some(rate).filter(|&r| r > 0);
		let nominal = i32::from_le_bytes([first[20], first[21], first[22], first[23]]);
		parameters.bitrate = u32::try_from(nominal).ok().filter(|&b| b > 0);
		(Mapping::Vorbis, Some(2), 0, vec![first.to_vec()])
	} else {
		return None;
	};
	Some(Logical {
		serial,
		mapping,
		index,
		parameters,
		metadata,
		headers_left,
		vorbis_headers,
		pre_skip,
		partial: Vec::new(),
		partial_broken: false,
		next_pts: None,
	})
}

/// FFmpeg's Vorbis extradata: `2`, the first two header sizes in Xiph
/// lacing, then the three headers.
fn xiph_extradata(headers: &[Vec<u8>]) -> Vec<u8> {
	let mut out = vec![2u8];
	for h in &headers[..2] {
		let mut n = h.len();
		while n >= 255 {
			out.push(255);
			n -= 255;
		}
		out.push(n as u8);
	}
	for h in headers {
		out.extend_from_slice(h);
	}
	out
}

impl OggDemuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let data_end = reader.size().unwrap_or(u64::MAX);
		let mut logical: Vec<Logical> = Vec::new();
		let mut queue = VecDeque::new();
		let mut container_metadata = Metadata::default();
		let mut data_start = None;

		for _ in 0..MAX_HEADER_PAGES {
			if !logical.is_empty() && logical.iter().all(|l| l.headers_left == Some(0)) {
				break;
			}
			let position = reader.position();
			let Some(page) = next_page(&mut reader)? else { break };
			if page.flags & FLAG_BOS == 0 {
				data_start = data_start.or(Some(position));
			}
			match logical.iter().position(|l| l.serial == page.serial) {
				Some(at) => handle_page(&mut logical[at], &page, &mut queue, Some(&mut container_metadata)),
				None if page.flags & FLAG_BOS != 0 && logical.len() < MAX_STREAMS => {
					if let Some(l) =
						first_packet(&page).and_then(|f| new_logical(&f, page.serial, logical.len()))
					{
						logical.push(l);
					}
				}
				// Pages of streams RMP does not handle.
				None => {}
			}
		}
		if logical.is_empty() {
			return Err(Message::Container("ogg: no supported stream"));
		}
		for l in logical.iter_mut() {
			if l.mapping == Mapping::Vorbis {
				if l.vorbis_headers.len() < 3 {
					return Err(Message::Container("ogg: vorbis headers missing"));
				}
				l.parameters.extradata = xiph_extradata(&l.vorbis_headers);
			}
		}
		let data_start = data_start.unwrap_or_else(|| reader.position());

		// Duration: the last granule position of the first stream.
		let primary = logical[0].serial;
		let mut last_granule = None;
		if data_end != u64::MAX {
			let resume = reader.position();
			last_granule = find_last_granule(&mut reader, primary, data_end)?;
			reader.seek(resume)?;
		}

		let streams: Vec<Stream> = logical
			.iter()
			.map(|l| {
				let tb = TimeBase::new(1, l.rate());
				let metadata = l.metadata.clone();
				Stream {
					id: StreamId(l.index),
					kind: StreamKind::Audio,
					time_base: tb,
					duration: if l.serial == primary { last_granule.map(|g| Time::new(g, tb)) } else { None },
					parameters: l.parameters.clone(),
					metadata,
				}
			})
			.collect();
		let tb = TimeBase::new(1, logical[0].rate());
		let duration = Time::new(last_granule.unwrap_or(0), tb);
		Ok(Self {
			reader,
			streams: StreamSet::new(streams),
			metadata: container_metadata,
			logical,
			queue,
			duration,
			data_start,
			data_end,
		})
	}

	/// First page at or after `position` of the primary stream that has a
	/// granule position, as `(page offset, granule)`.
	fn probe(&mut self, position: u64) -> Result<Option<(u64, u64)>> {
		self.reader.seek(position)?;
		let serial = self.logical[0].serial;
		for _ in 0..1024 {
			if !sync(&mut self.reader)? {
				return Ok(None);
			}
			let offset = self.reader.position();
			let Some(page) = next_page(&mut self.reader)? else { return Ok(None) };
			if page.serial == serial {
				if let Some(g) = page.granule {
					return Ok(Some((offset, g)));
				}
			}
		}
		Ok(None)
	}
}

/// Advances to the next `"OggS"` capture pattern; `false` at end of data.
fn sync(reader: &mut BufferedReader) -> Result<bool> {
	loop {
		let window = reader.fill(64 * 1024)?;
		if window.len() < HEADER_LEN {
			return Ok(false);
		}
		match window.windows(4).position(|w| w == b"OggS") {
			Some(i) => {
				reader.consume(i);
				return Ok(true);
			}
			None => {
				let advance = window.len() - 3;
				reader.consume(advance);
			}
		}
	}
}

/// Reads the next valid page, resynchronising past damage.
fn next_page(reader: &mut BufferedReader) -> Result<Option<Page>> {
	loop {
		if !sync(reader)? {
			return Ok(None);
		}
		let head = reader.fill(HEADER_LEN + 255)?;
		if head.len() < HEADER_LEN {
			return Ok(None);
		}
		let segments = usize::from(head[26]);
		let want = HEADER_LEN + segments + 255 * segments;
		let window = reader.fill(want)?;
		match parse_page(window) {
			Some((page, len)) => {
				reader.consume(len);
				return Ok(Some(page));
			}
			None => reader.consume(1),
		}
	}
}

/// The packet that starts the page (for BOS pages, which hold exactly the
/// identification header).
fn first_packet(page: &Page) -> Option<Vec<u8>> {
	if page.flags & FLAG_CONTINUED != 0 {
		return None;
	}
	let mut len = 0usize;
	for &l in &page.lacing {
		len += usize::from(l);
		if l < 255 {
			return page.body.get(..len).map(<[u8]>::to_vec);
		}
	}
	None
}

/// Splits a page into packets for its stream: header packets go to the
/// stream's parameters / metadata, audio packets to `queue` with
/// timestamps.
fn handle_page(
	l: &mut Logical,
	page: &Page,
	queue: &mut VecDeque<Packet>,
	mut container: Option<&mut Metadata>,
) {
	if page.flags & FLAG_CONTINUED == 0 {
		// A packet left open by the previous page lost its end.
		l.partial.clear();
		l.partial_broken = false;
	} else if l.partial.is_empty() {
		// Continues a packet whose start was not seen.
		l.partial_broken = true;
	}
	let mut complete: Vec<Vec<u8>> = Vec::new();
	let mut offset = 0usize;
	for &lace in &page.lacing {
		let seg = &page.body[offset..offset + usize::from(lace)];
		offset += usize::from(lace);
		if !l.partial_broken {
			if l.partial.len() + seg.len() > MAX_PACKET {
				l.partial_broken = true;
				l.partial = Vec::new();
			} else {
				l.partial.extend_from_slice(seg);
			}
		}
		if lace < 255 {
			if l.partial_broken {
				l.partial.clear();
			} else {
				complete.push(std::mem::take(&mut l.partial));
			}
			l.partial_broken = false;
		}
	}

	let mut audio: Vec<Vec<u8>> = Vec::new();
	for packet in complete {
		if is_header(l, &packet) {
			// Headers seen again after a seek back to the start are skipped.
			if l.headers_left != Some(0) {
				take_header(l, &packet, container.as_deref_mut());
			}
			continue;
		}
		if l.headers_left.is_none() {
			l.headers_left = Some(0);
		}
		if l.headers_left == Some(0) && !packet.is_empty() {
			audio.push(packet);
		}
	}
	if audio.is_empty() {
		return;
	}

	let tb = TimeBase::new(1, l.rate());
	let durations: Vec<Option<u64>> =
		audio.iter().map(|p| packet_samples(&l.parameters, p).map(|(n, _)| n)).collect();
	let n = audio.len();
	let mut pts: Vec<Option<u64>> = vec![None; n];
	let mut trimmed_last: Option<u64> = None;
	if let Some(g) = page.granule {
		// Walk back from the granule while durations are known.
		let mut end = g;
		let mut known = true;
		let total: Option<u64> = durations.iter().copied().sum();
		// When the page is the stream's first data page and the durations
		// add up to more than the granule, the stream starts trimmed; for
		// any other page a granule smaller than the sum marks end trimming
		// of the last packet (RFC 7845 §4.4).
		if let (Some(total), Some(start)) = (total, l.next_pts) {
			if start + total > g && g >= start {
				// End trimming: packets start where the previous ended.
				let mut t = start;
				for i in 0..n {
					pts[i] = Some(t);
					t += durations[i].unwrap_or(0);
				}
				trimmed_last = Some(g - pts[n - 1].unwrap_or(g).min(g));
				known = false;
			}
		}
		if known {
			for i in (0..n).rev() {
				let Some(d) = durations[i] else { break };
				end = end.saturating_sub(d);
				pts[i] = Some(end);
			}
		}
	}
	// Packets still without a timestamp follow the previous one (with
	// unknown durations, only the page's first packet gets one).
	let mut t = l.next_pts;
	for i in 0..n {
		if pts[i].is_none() {
			pts[i] = t;
		}
		t = match (pts[i], durations[i]) {
			(Some(p), Some(d)) => Some(p + d),
			_ => None,
		};
	}
	l.next_pts = page.granule.or(t);
	for (i, data) in audio.into_iter().enumerate() {
		let mut packet = Packet::new(StreamId(l.index), data);
		packet.pts = pts[i].map(|p| Time::new(p, tb));
		packet.dts = packet.pts;
		let duration = if i == n - 1 { trimmed_last.or(durations[i]) } else { durations[i] };
		packet.duration = duration.map(|d| Time::new(d, tb));
		queue.push_back(packet);
	}
}

fn is_header(l: &Logical, packet: &[u8]) -> bool {
	match l.mapping {
		Mapping::Opus => packet.starts_with(b"OpusTags"),
		// FLAC audio frames start with the 0xFFF8 sync code; metadata
		// blocks never start with 0xFF.
		Mapping::Flac => packet.first().is_some_and(|&b| b != 0xFF),
		Mapping::Vorbis => {
			packet.first().is_some_and(|&b| b & 1 == 1) && packet.get(1..7) == Some(b"vorbis")
		}
	}
}

fn take_header(l: &mut Logical, packet: &[u8], container: Option<&mut Metadata>) {
	let metadata = &mut l.metadata;
	match l.mapping {
		Mapping::Opus => {
			xiph::parse_comments(&packet[8..], metadata);
		}
		Mapping::Flac => {
			if packet.len() >= 4 && packet[0] & 0x7F == 4 {
				xiph::parse_comments(&packet[4..], metadata);
			} else if packet.len() >= 4 && packet[0] & 0x7F == 6 {
				if let Some(image) = xiph::parse_picture(&packet[4..]) {
					metadata.images.push(image);
				}
			}
		}
		Mapping::Vorbis => {
			if packet[0] == 3 {
				xiph::parse_comments(&packet[7..], metadata);
			}
			if l.vorbis_headers.len() < 3 {
				l.vorbis_headers.push(packet.to_vec());
			}
		}
	}
	if let Some(container) = container {
		if l.index == 0 {
			// The first stream's tags are also the file's tags, as FFmpeg
			// reports them.
			*container = metadata.clone();
			container.raw.remove("encoder_delay");
		}
	}
	l.headers_left = Some(l.headers_left.map_or(0, |n| n.saturating_sub(1)));
	if l.mapping == Mapping::Flac && packet[0] & 0x80 != 0 {
		l.headers_left = Some(0);
	}
}

/// Last granule position of stream `serial`, searched backward from the end.
fn find_last_granule(reader: &mut BufferedReader, serial: u32, end: u64) -> Result<Option<u64>> {
	let mut start = end.saturating_sub(64 * 1024);
	loop {
		reader.seek(start)?;
		let mut last = None;
		let mut guard = 0;
		while guard < 4096 && sync(reader)? {
			guard += 1;
			match next_page(reader)? {
				Some(page) => {
					if page.serial == serial {
						if let Some(g) = page.granule {
							last = Some(g);
						}
					}
				}
				None => break,
			}
		}
		if last.is_some() || start == 0 || end - start >= TAIL_SEARCH {
			return Ok(last);
		}
		start = start.saturating_sub(64 * 1024);
	}
}

impl Demuxer for OggDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		loop {
			if let Some(packet) = self.queue.pop_front() {
				return Ok(Some(packet));
			}
			if self.reader.position() >= self.data_end {
				return Ok(None);
			}
			let Some(page) = next_page(&mut self.reader)? else { return Ok(None) };
			if let Some(l) = self.logical.iter_mut().find(|l| l.serial == page.serial) {
				handle_page(l, &page, &mut self.queue, None);
			}
		}
	}

	/// Bisects over byte offsets for the last page of the first stream
	/// whose granule is before the target less the Opus pre-roll, and
	/// resumes right after it, so a decoder gets 80 ms to converge before
	/// the target (FFmpeg: `ogg_read_seek` + `seek_preroll`).
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let first = &self.logical[0];
		let mut target = (time * f64::from(first.rate())) as u64;
		if first.mapping == Mapping::Opus {
			target = target.saturating_add(first.pre_skip).saturating_sub(OPUS_PREROLL);
		}
		let mut best = None;
		let mut lo = self.data_start;
		let mut hi =
			if self.data_end == u64::MAX { self.reader.size().unwrap_or(lo) } else { self.data_end };
		for _ in 0..MAX_SEEK_PROBES {
			if hi <= lo + 1 {
				break;
			}
			let mid = lo + (hi - lo) / 2;
			match self.probe(mid)? {
				Some((offset, g)) if g < target && offset < hi => {
					best = Some((offset, g));
					lo = offset.max(mid) + 1;
				}
				_ => hi = mid,
			}
		}
		self.queue.clear();
		for l in self.logical.iter_mut() {
			l.partial.clear();
			l.partial_broken = false;
			l.next_pts = None;
		}
		match best {
			// Resume after the last page that ends before the target: its
			// granule is where the next page's packets start.
			Some((offset, granule)) => {
				self.reader.seek(offset)?;
				let _ = next_page(&mut self.reader)?;
				self.logical[0].next_pts = Some(granule);
			}
			None => self.reader.seek(self.data_start)?,
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn crc_matches_reference() {
		assert_eq!(crc32(b""), 0);
		// One set bit in the top byte gives the polynomial itself.
		assert_eq!(crc32(&[0x01]), 0x04C1_1DB7);
	}

	#[test]
	fn xiph_lacing() {
		let headers = vec![vec![1u8; 300], vec![3u8; 10], vec![5u8; 2]];
		let out = xiph_extradata(&headers);
		assert_eq!(&out[..4], &[2, 255, 45, 10]);
		assert_eq!(out.len(), 4 + 312);
	}
}
