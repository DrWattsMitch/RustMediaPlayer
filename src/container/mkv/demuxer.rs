//! Matroska / WebM demuxer.
//!
//! Opening reads the EBML header, then the `Segment`'s level-1 elements up
//! to the first `Cluster` (`Info`, `Tracks`, `Cues`, `Tags`,
//! `Attachments`, `SeekHead`). Elements that the `SeekHead` places after the
//! clusters (usually `Cues` and `Tags`) are then loaded by seeking, when the
//! input size is known. Clusters are never loaded whole: reading walks their
//! children one element at a time, so unknown-size (live) clusters, huge
//! clusters and truncated files all work the same way.
//!
//! FFmpeg: `libavformat/matroskadec.c` (`matroska_read_header`,
//! `matroska_parse_cluster`, `matroska_parse_block`,
//! `matroska_parse_laces`, `matroska_read_seek`).

use std::collections::VecDeque;

use super::ebml::{self, Children, Header, MAX_HEADER_LEN, id};
use super::tracks::{self, Track};
use crate::codecs::duration::packet_samples;
use crate::core::{
	AttachedFile, AttachedImage, Demuxer, Metadata, Packet, RawValue, StreamKind, StreamSet, Time,
	TimeBase,
};
use crate::io::{BufferedReader, ByteCursor, Io};
use crate::message::{Message, Result};

/// Default `TimestampScale`: 1 ms per tick.
const DEFAULT_SCALE_NS: u64 = 1_000_000;
/// Largest level-1 metadata element loaded into memory (`Tracks`, `Cues`,
/// `Tags`, `Attachments`). Larger ones are skipped.
const MAX_LEVEL1_SIZE: u64 = 64 * 1024 * 1024;
/// Largest `EBML` header accepted.
const MAX_EBML_HEADER: u64 = 4096;
/// Largest block (one `SimpleBlock` / `BlockGroup`) accepted.
const MAX_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
/// Most clusters indexed when a file has no `Cues`.
const MAX_INDEXED_CLUSTERS: usize = 1 << 20;
/// Read-ahead used when scanning for the next `Cluster` after damage.
const RESYNC_WINDOW: usize = 64 * 1024;

#[derive(Debug, Clone, Copy)]
struct CuePoint {
	/// In segment ticks.
	time: u64,
	track: u64,
	/// Absolute file offset of the cluster.
	position: u64,
}

/// What to drop after a seek so playback starts cleanly.
#[derive(Debug, Clone)]
struct SeekFilter {
	/// Packets with an earlier timestamp are dropped (except video, below).
	start: u64,
	/// Video streams still waiting for their first keyframe.
	need_keyframe: Vec<usize>,
}

pub struct MkvDemuxer {
	reader: BufferedReader,
	streams: StreamSet,
	tracks: Vec<Track>,
	metadata: Metadata,
	scale_ns: u64,
	time_base: TimeBase,
	duration: Time,
	segment_start: u64,
	segment_end: u64,
	first_cluster: u64,
	cues: Vec<CuePoint>,
	/// `(timestamp, offset)` of each cluster, built on the first seek of a
	/// file without cues.
	cluster_index: Option<Vec<(u64, u64)>>,
	cluster_ts: u64,
	queue: VecDeque<Packet>,
	seek_filter: Option<SeekFilter>,
}

/// Parsed level-1 bodies found while opening.
#[derive(Default)]
struct Level1 {
	info: Option<Vec<u8>>,
	tracks: Option<Vec<u8>>,
	cues: Option<Vec<u8>>,
	tags: Vec<Vec<u8>>,
	attachments: Option<Vec<u8>>,
	/// `(element id, absolute position)` from every `SeekHead`.
	seek_entries: Vec<(u32, u64)>,
	seek_heads: usize,
}

fn read_header(reader: &mut BufferedReader) -> Result<Option<Header>> {
	let head = reader.fill(MAX_HEADER_LEN)?;
	if head.is_empty() {
		return Ok(None);
	}
	match ebml::parse_header(head) {
		Ok(h) => {
			reader.consume(h.len);
			Ok(Some(h))
		}
		Err(_) if head.len() < MAX_HEADER_LEN => Ok(None),
		Err(e) => Err(e),
	}
}

fn read_body(reader: &mut BufferedReader, header: &Header, max: u64) -> Result<Option<Vec<u8>>> {
	let Some(size) = header.size else { return Ok(None) };
	if size > max {
		reader.skip(size)?;
		return Ok(None);
	}
	// A truncated element keeps what the file holds.
	let available =
		reader.size().map_or(size, |total| total.saturating_sub(reader.position()).min(size));
	reader.read_vec(available as usize).map(Some)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
	while b != 0 {
		(a, b) = (b, a % b);
	}
	a
}

/// `TimestampScale` (ns per tick) as a time base in seconds.
fn time_base_for(scale_ns: u64) -> Result<TimeBase> {
	let g = gcd(scale_ns, 1_000_000_000);
	let num = u32::try_from(scale_ns / g)
		.map_err(|_| Message::Container("unsupported mkv timestamp scale"))?;
	Ok(TimeBase::new(num, (1_000_000_000 / g) as u32))
}

impl MkvDemuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let file_size = reader.size();

		let header = read_header(&mut reader)?.ok_or(Message::Container("empty matroska file"))?;
		if header.id != id::EBML {
			return Err(Message::Container("not a matroska file"));
		}
		let ebml_header = read_body(&mut reader, &header, MAX_EBML_HEADER)?
			.ok_or(Message::Container("invalid ebml header"))?;
		let doc_type =
			ebml::child_string(&ebml_header, id::DOC_TYPE).unwrap_or_else(|| "matroska".into());
		if doc_type != "matroska" && doc_type != "webm" {
			return Err(Message::Container("unsupported ebml document type"));
		}
		if ebml::child_uint(&ebml_header, id::EBML_MAX_ID_LENGTH, 4) > 4
			|| ebml::child_uint(&ebml_header, id::EBML_MAX_SIZE_LENGTH, 8) > 8
		{
			return Err(Message::Container("unsupported ebml header limits"));
		}

		// Skip anything (Void, CRC) between the EBML header and the Segment.
		let segment = loop {
			let h = read_header(&mut reader)?.ok_or(Message::Container("matroska segment not found"))?;
			if h.id == id::SEGMENT {
				break h;
			}
			reader.skip(h.size.ok_or(Message::Container("matroska segment not found"))?)?;
		};
		let segment_start = reader.position();
		let mut segment_end =
			segment.size.and_then(|s| segment_start.checked_add(s)).unwrap_or(u64::MAX);
		if let Some(total) = file_size {
			segment_end = segment_end.min(total);
		}

		let mut level1 = Level1::default();
		let mut visited: Vec<u64> = Vec::new();
		let mut first_cluster = segment_end;
		loop {
			let position = reader.position();
			if position >= segment_end {
				break;
			}
			let Some(h) = read_header(&mut reader)? else { break };
			if h.id == id::CLUSTER {
				first_cluster = position;
				break;
			}
			visited.push(position);
			if !Self::load_level1(&mut reader, &h, segment_start, &mut level1)? {
				break;
			}
		}

		// Elements the SeekHead places after the clusters.
		if file_size.is_some() {
			let mut i = 0;
			while i < level1.seek_entries.len() && i < 64 {
				let (eid, position) = level1.seek_entries[i];
				i += 1;
				let wanted = match eid {
					id::INFO => level1.info.is_none(),
					id::TRACKS => level1.tracks.is_none(),
					id::CUES => level1.cues.is_none(),
					id::TAGS => level1.tags.is_empty(),
					id::ATTACHMENTS => level1.attachments.is_none(),
					id::SEEK_HEAD => level1.seek_heads < 4,
					_ => false,
				};
				if !wanted || position >= segment_end || visited.contains(&position) {
					continue;
				}
				visited.push(position);
				if reader.seek(position).is_err() {
					continue;
				}
				match read_header(&mut reader) {
					Ok(Some(h)) if h.id == eid => {
						let _ = Self::load_level1(&mut reader, &h, segment_start, &mut level1);
					}
					_ => continue,
				}
			}
		}

		let info = level1.info.as_deref().unwrap_or(&[]);
		let scale_ns = ebml::child_uint(info, id::TIMESTAMP_SCALE, DEFAULT_SCALE_NS);
		let scale_ns = if scale_ns == 0 { DEFAULT_SCALE_NS } else { scale_ns };
		let time_base = time_base_for(scale_ns)?;
		let duration_ticks = ebml::child_float(info, id::DURATION)
			.filter(|d| d.is_finite() && *d >= 0.0 && *d < 1e18)
			.map_or(0, |d| d.round() as u64);

		let mut metadata = Metadata::default();
		if let Some(title) = ebml::child_string(info, id::TITLE).filter(|t| !t.is_empty()) {
			metadata.title = Some(title);
		}
		if let Some(app) = ebml::child_string(info, id::WRITING_APP) {
			metadata.raw.insert("encoder".into(), RawValue::String(app));
		}

		let mut streams = StreamSet::default();
		let mut track_list: Vec<Track> = Vec::new();
		for (eid, entry) in Children::new(level1.tracks.as_deref().unwrap_or(&[])) {
			if eid != id::TRACK_ENTRY {
				continue;
			}
			let Ok(track) = tracks::parse_entry(entry, track_list.len(), time_base) else { continue };
			if track_list.iter().any(|t| t.number == track.number) {
				continue;
			}
			streams.add(track.stream.clone());
			track_list.push(track);
		}
		if track_list.is_empty() {
			return Err(Message::Container("matroska file has no playable tracks"));
		}

		for tags in &level1.tags {
			apply_tags(tags, &mut metadata, &mut streams, &track_list);
		}
		if let Some(attachments) = &level1.attachments {
			apply_attachments(attachments, &mut metadata);
		}
		let mut cues = level1.cues.as_deref().map(|c| parse_cues(c, segment_start)).unwrap_or_default();
		cues.retain(|c| c.position < segment_end);
		cues.sort_by_key(|c| (c.time, c.position));

		reader.seek(first_cluster)?;
		Ok(Self {
			reader,
			streams,
			tracks: track_list,
			metadata,
			scale_ns,
			time_base,
			duration: Time::new(duration_ticks, time_base),
			segment_start,
			segment_end,
			first_cluster,
			cues,
			cluster_index: None,
			cluster_ts: 0,
			queue: VecDeque::new(),
			seek_filter: None,
		})
	}

	/// Loads one level-1 element during open. Returns `false` when the scan
	/// can't continue (an unknown-size element other than a cluster).
	fn load_level1(
		reader: &mut BufferedReader,
		h: &Header,
		segment_start: u64,
		level1: &mut Level1,
	) -> Result<bool> {
		let slot: Option<&mut Option<Vec<u8>>> = match h.id {
			id::INFO => Some(&mut level1.info),
			id::TRACKS => Some(&mut level1.tracks),
			id::CUES => Some(&mut level1.cues),
			id::ATTACHMENTS => Some(&mut level1.attachments),
			_ => None,
		};
		if h.size.is_none() {
			return Ok(false);
		}
		match (h.id, slot) {
			(_, Some(slot)) => {
				let body = read_body(reader, h, MAX_LEVEL1_SIZE)?;
				if slot.is_none() {
					*slot = body;
				}
			}
			(id::TAGS, None) => {
				if let Some(body) = read_body(reader, h, MAX_LEVEL1_SIZE)? {
					level1.tags.push(body);
				}
			}
			(id::SEEK_HEAD, None) => {
				level1.seek_heads += 1;
				if let Some(body) = read_body(reader, h, MAX_LEVEL1_SIZE)? {
					for (eid, seek) in Children::new(&body) {
						if eid != id::SEEK {
							continue;
						}
						let target = ebml::find(seek, id::SEEK_ID).and_then(|b| ebml::uint(b).ok());
						let position = ebml::find(seek, id::SEEK_POSITION).and_then(|b| ebml::uint(b).ok());
						if let (Some(target), Some(position)) = (target, position) {
							if let (Ok(target), Some(abs)) =
								(u32::try_from(target), segment_start.checked_add(position))
							{
								level1.seek_entries.push((target, abs));
							}
						}
					}
				}
			}
			_ => reader.skip(h.size.unwrap_or(0))?,
		}
		Ok(true)
	}

	fn track_index(&self, number: u64) -> Option<usize> {
		self.tracks.iter().position(|t| t.number == number)
	}

	fn ns_to_ticks(&self, ns: u64) -> u64 {
		(ns + self.scale_ns / 2) / self.scale_ns
	}

	/// Reads the next element of the cluster stream. Returns `false` at the
	/// end of the segment (or of the file).
	fn next_element(&mut self) -> Result<bool> {
		let position = self.reader.position();
		if position >= self.segment_end {
			return Ok(false);
		}
		let head = self.reader.fill(MAX_HEADER_LEN)?;
		if head.is_empty() {
			return Ok(false);
		}
		let header = match ebml::parse_header(head) {
			Ok(h) => h,
			Err(_) if head.len() < MAX_HEADER_LEN => return Ok(false),
			Err(_) => return self.resync(),
		};
		match (header.id, header.size) {
			(id::CLUSTER, _) => {
				self.reader.consume(header.len);
				self.cluster_ts = 0;
			}
			(id::SEGMENT | id::EBML, _) => return Ok(false), // chained segments are not followed
			(_, None) => return self.resync(),
			(id::TIMESTAMP, Some(size)) if size <= 8 => {
				self.reader.consume(header.len);
				let body = self.reader.read_vec(size as usize).unwrap_or_default();
				self.cluster_ts = ebml::uint(&body).unwrap_or(0);
			}
			(id::SIMPLE_BLOCK | id::BLOCK_GROUP, Some(size)) if size <= MAX_BLOCK_SIZE => {
				self.reader.consume(header.len);
				if self
					.reader
					.size()
					.is_some_and(|total| self.reader.position().saturating_add(size) > total)
				{
					return Ok(false); // truncated file: end of stream
				}
				let body = match self.reader.read_vec(size as usize) {
					Ok(body) => body,
					Err(_) => return Ok(false),
				};
				if header.id == id::SIMPLE_BLOCK {
					self.parse_block(&body, None);
				} else {
					self.parse_block_group(&body);
				}
			}
			(eid, Some(size)) if is_skippable(eid) => {
				self.reader.consume(header.len);
				self.reader.skip(size)?;
			}
			_ => return self.resync(),
		}
		Ok(true)
	}

	/// Scans forward for the next `Cluster` ID after damage.
	fn resync(&mut self) -> Result<bool> {
		const MAGIC: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];
		self.reader.consume(1);
		loop {
			if self.reader.position() >= self.segment_end {
				return Ok(false);
			}
			let window = self.reader.fill(RESYNC_WINDOW)?;
			if window.len() < MAGIC.len() {
				return Ok(false);
			}
			if let Some(at) = window.windows(4).position(|w| w == MAGIC) {
				self.reader.consume(at);
				return Ok(true);
			}
			let keep = MAGIC.len() - 1;
			let advance = window.len() - keep;
			self.reader.consume(advance);
		}
	}

	fn parse_block_group(&mut self, body: &[u8]) {
		let mut block = None;
		let mut duration = None;
		let mut referenced = false;
		for (eid, child) in Children::new(body) {
			match eid {
				id::BLOCK => block = Some(child),
				id::BLOCK_DURATION => duration = ebml::uint(child).ok(),
				id::REFERENCE_BLOCK => referenced = true,
				_ => {}
			}
		}
		if let Some(block) = block {
			self.parse_block(block, Some((duration, !referenced)));
		}
	}

	/// Parses a `SimpleBlock` (`group` = `None`) or the `Block` of a
	/// `BlockGroup` (`group` = `(BlockDuration, keyframe)`) and queues its
	/// frames. Malformed blocks are dropped.
	fn parse_block(&mut self, data: &[u8], group: Option<(Option<u64>, bool)>) {
		let mut c = ByteCursor::new(data);
		let Ok((number, _)) = ebml::read_vint(&mut c) else { return };
		let (Ok(relative), Ok(flags)) = (c.read_u16(), c.read_u8()) else { return };
		let Some(index) = self.track_index(number) else { return };
		let Some(sizes) = lace_sizes(&mut c, (flags >> 1) & 3) else { return };

		let track = &self.tracks[index];
		let timestamp =
			(self.cluster_ts as i64).saturating_add(i64::from(relative as i16)).max(0) as u64;
		let keyframe = match group {
			None => flags & 0x80 != 0,
			Some((_, key)) => key,
		};
		let default_ticks =
			(track.default_duration_ns > 0).then(|| self.ns_to_ticks(track.default_duration_ns));
		let block_duration = group.and_then(|(d, _)| d);
		let count = sizes.len() as u64;
		let lace_duration = match block_duration {
			Some(d) => Some(d / count),
			None => default_ticks,
		};
		let stream_id = track.stream.id;
		let is_video = track.stream.kind == StreamKind::Video;
		let is_audio = track.stream.kind == StreamKind::Audio;
		let webvtt = track.stream.parameters.codec.id == "webvtt";
		let mut pts = Some(timestamp);
		for size in sizes {
			let Ok(frame) = c.read_bytes(size) else { return };
			let mut payload = Vec::with_capacity(track.strip_prefix.len() + frame.len());
			payload.extend_from_slice(&track.strip_prefix);
			payload.extend_from_slice(frame);
			if webvtt {
				payload = strip_webvtt_header(&payload);
			}
			// Without a container duration, audio packets are timed from
			// their own headers, as FFmpeg's parsers do.
			let duration = lace_duration.or_else(|| {
				let (samples, rate) =
					is_audio.then(|| packet_samples(&track.stream.parameters, &payload)).flatten()?;
				// Rounded down, like FFmpeg's `compute_pkt_fields`.
				let num = u128::from(samples) * u128::from(self.time_base.den);
				let den = u128::from(rate) * u128::from(self.time_base.num);
				u64::try_from(num / den.max(1)).ok()
			});
			let mut packet = Packet::new(stream_id, payload);
			packet.pts = pts.map(|t| Time::new(t, self.time_base));
			packet.dts = if is_video { None } else { packet.pts };
			packet.duration = duration.map(|d| Time::new(d, self.time_base));
			packet.keyframe = keyframe;
			self.queue.push_back(packet);
			pts = match (pts, duration) {
				(Some(t), Some(d)) => t.checked_add(d),
				_ => None,
			};
		}
	}

	fn accept(&mut self, packet: &Packet) -> bool {
		let Some(filter) = self.seek_filter.as_mut() else { return true };
		let stream = packet.stream_id.0;
		if let Some(i) = filter.need_keyframe.iter().position(|&s| s == stream) {
			if !packet.keyframe {
				return false;
			}
			filter.need_keyframe.swap_remove(i);
			return true;
		}
		let is_video = self.streams.get(packet.stream_id).is_some_and(|s| s.kind == StreamKind::Video);
		is_video || packet.pts.is_none_or(|t| t.ticks() >= filter.start)
	}

	/// Cluster offsets and timestamps, scanned from the first cluster by
	/// header (cluster bodies are skipped, not read).
	fn build_cluster_index(&mut self) -> Result<Vec<(u64, u64)>> {
		let mut index = Vec::new();
		let mut position = self.first_cluster;
		while position < self.segment_end && index.len() < MAX_INDEXED_CLUSTERS {
			self.reader.seek(position)?;
			let Some(h) = read_header(&mut self.reader)? else { break };
			let Some(size) = h.size else {
				if h.id == id::CLUSTER {
					// Unknown size: the timestamp is still usable, but the
					// next cluster can't be found without reading this one.
					if let Some(ts) = self.read_cluster_timestamp()? {
						index.push((ts, position));
					}
				}
				break;
			};
			if h.id == id::CLUSTER {
				if let Some(ts) = self.read_cluster_timestamp()? {
					index.push((ts, position));
				}
			} else if !is_skippable(h.id) {
				break;
			}
			position = match position.checked_add(h.len as u64 + size) {
				Some(next) => next,
				None => break,
			};
		}
		Ok(index)
	}

	/// The `Timestamp` element at the start of the cluster being read.
	fn read_cluster_timestamp(&mut self) -> Result<Option<u64>> {
		for _ in 0..4 {
			let Some(h) = read_header(&mut self.reader)? else { return Ok(None) };
			let Some(size) = h.size else { return Ok(None) };
			if h.id == id::TIMESTAMP && size <= 8 {
				let body = self.reader.read_vec(size as usize)?;
				return Ok(ebml::uint(&body).ok());
			}
			if !matches!(h.id, id::VOID | id::CRC32 | id::POSITION | id::PREV_SIZE) {
				return Ok(None);
			}
			self.reader.skip(size)?;
		}
		Ok(None)
	}

	/// Cluster to start reading from for `target` ticks, and the time
	/// playback starts at.
	fn locate(&mut self, target: u64) -> Result<(u64, u64)> {
		if !self.cues.is_empty() {
			let video = self
				.tracks
				.iter()
				.find(|t| {
					t.stream.kind == StreamKind::Video && self.cues.iter().any(|c| c.track == t.number)
				})
				.map(|t| t.number);
			let track = video.unwrap_or(self.cues[0].track);
			let best = self.cues.iter().rfind(|c| c.track == track && c.time <= target);
			return Ok(best.map_or((self.first_cluster, 0), |c| (c.position, c.time)));
		}
		if self.cluster_index.is_none() {
			let index = self.build_cluster_index()?;
			self.cluster_index = Some(index);
		}
		let index = self.cluster_index.as_deref().unwrap_or(&[]);
		let at = index.partition_point(|&(ts, _)| ts <= target);
		Ok(at.checked_sub(1).map_or((self.first_cluster, 0), |i| (index[i].1, index[i].0)))
	}
}

/// WebM WebVTT blocks start with the cue identifier and settings lines
/// (WebM "WebVTT in WebM" §4); the packet is the cue text alone, as FFmpeg's
/// `matroska_parse_webvtt` returns it.
fn strip_webvtt_header(block: &[u8]) -> Vec<u8> {
	let mut rest = block;
	for _ in 0..2 {
		match rest.iter().position(|&b| b == b'\n') {
			Some(at) => rest = &rest[at + 1..],
			None => return Vec::new(),
		}
	}
	rest.to_vec()
}

/// IDs skipped while walking clusters: everything legitimately found at the
/// segment or cluster level that isn't a block.
fn is_skippable(eid: u32) -> bool {
	ebml::is_top_level(eid)
		|| matches!(
			eid,
			id::VOID
				| id::CRC32
				| id::POSITION
				| id::PREV_SIZE
				| id::SILENT_TRACKS
				| id::ENCRYPTED_BLOCK
				| id::TIMESTAMP
		)
}

/// Frame sizes of a block's lace (`lacing`: 0 none, 1 Xiph, 2 fixed, 3 EBML).
/// Every size is checked against the bytes that remain.
fn lace_sizes(c: &mut ByteCursor, lacing: u8) -> Option<Vec<usize>> {
	if lacing == 0 {
		return Some(vec![c.remaining()]);
	}
	let count = usize::from(c.read_u8().ok()?) + 1;
	let mut sizes = Vec::with_capacity(count);
	match lacing {
		1 => {
			for _ in 0..count - 1 {
				let mut size = 0usize;
				loop {
					let byte = c.read_u8().ok()?;
					size += usize::from(byte);
					if byte != 255 {
						break;
					}
				}
				sizes.push(size);
			}
		}
		3 => {
			let (first, _) = ebml::read_vint(c).ok()?;
			let mut size = i64::try_from(first).ok()?;
			sizes.push(usize::try_from(size).ok()?);
			for _ in 1..count - 1 {
				size = size.checked_add(ebml::read_signed_vint(c).ok()?)?;
				sizes.push(usize::try_from(size).ok()?);
			}
		}
		_ => {
			let each = c.remaining() / count;
			if each * count != c.remaining() {
				return None;
			}
			return Some(vec![each; count]);
		}
	}
	let used = sizes.iter().try_fold(0usize, |acc, &s| acc.checked_add(s))?;
	let last = c.remaining().checked_sub(used)?;
	sizes.push(last);
	Some(sizes)
}

fn parse_cues(cues: &[u8], segment_start: u64) -> Vec<CuePoint> {
	let mut out = Vec::new();
	for (eid, point) in Children::new(cues) {
		if eid != id::CUE_POINT {
			continue;
		}
		let time = ebml::child_uint(point, id::CUE_TIME, 0);
		for (pid, positions) in Children::new(point) {
			if pid != id::CUE_TRACK_POSITIONS {
				continue;
			}
			let track = ebml::child_uint(positions, id::CUE_TRACK, 0);
			let Some(cluster) =
				ebml::find(positions, id::CUE_CLUSTER_POSITION).and_then(|b| ebml::uint(b).ok())
			else {
				continue;
			};
			if let Some(position) = segment_start.checked_add(cluster) {
				out.push(CuePoint { time, track, position });
			}
		}
	}
	out
}

/// Applies `SimpleTag`s: global ones to the file metadata, track-targeted
/// ones to that stream's metadata (FFmpeg's `matroska_convert_tags`).
fn apply_tags(tags: &[u8], metadata: &mut Metadata, streams: &mut StreamSet, tracks: &[Track]) {
	for (eid, tag) in Children::new(tags) {
		if eid != id::TAG {
			continue;
		}
		let targets = ebml::find(tag, id::TARGETS).unwrap_or(&[]);
		let track_uid = ebml::child_uint(targets, id::TAG_TRACK_UID, 0);
		let target_type = ebml::child_uint(targets, id::TARGET_TYPE_VALUE, 50);
		let stream = (track_uid != 0)
			.then(|| tracks.iter().find(|t| t.uid == track_uid).map(|t| t.stream.id))
			.flatten();
		if track_uid != 0 && stream.is_none() {
			continue;
		}
		for (sid, simple) in Children::new(tag) {
			if sid != id::SIMPLE_TAG {
				continue;
			}
			let Some(name) = ebml::child_string(simple, id::TAG_NAME) else { continue };
			let Some(value) = ebml::child_string(simple, id::TAG_STRING) else { continue };
			let key = name.to_ascii_uppercase();
			match stream {
				Some(sid) => {
					if let Some(s) = streams.streams.iter_mut().find(|s| s.id == sid) {
						if key == "TITLE" {
							s.metadata.title = Some(value.clone());
						}
						s.metadata.raw.insert(name.to_ascii_lowercase(), RawValue::String(value));
					}
				}
				None => apply_global_tag(metadata, &key, value, target_type),
			}
		}
	}
}

fn apply_global_tag(m: &mut Metadata, key: &str, value: String, target_type: u64) {
	let number = |v: &str| v.split('/').next().and_then(|n| n.trim().parse::<u32>().ok());
	match key {
		// An album-level (50+) TITLE is the album when a track title exists.
		"TITLE" if target_type > 50 => m.album = Some(value),
		"TITLE" => m.title = Some(value),
		"ARTIST" | "LEAD_PERFORMER" => m.artist = Some(value),
		"ALBUM" => m.album = Some(value),
		"ALBUM_ARTIST" | "ALBUMARTIST" => m.album_artist = Some(value),
		"GENRE" => m.genre = Some(value),
		"DATE" | "DATE_RELEASED" | "DATE_RECORDED" => m.date = Some(value),
		"COMMENT" => m.comment = Some(value),
		"DESCRIPTION" | "SYNOPSIS" => m.description = Some(value),
		"LYRICS" => m.lyrics = Some(value),
		"PART_NUMBER" | "TRACK" | "TRACKNUMBER" => {
			m.track_number = number(&value);
			if let Some(total) = value.split('/').nth(1).and_then(|t| t.trim().parse().ok()) {
				m.tracks_total = Some(total);
			}
		}
		"TOTAL_PARTS" => m.tracks_total = number(&value),
		"DISC" | "DISCNUMBER" => m.disc_number = number(&value),
		_ => {
			m.raw.insert(key.to_ascii_lowercase(), RawValue::String(value));
		}
	}
}

/// Attached images become cover art; other attachments (fonts, ...) are
/// kept in `raw` under their file name.
fn apply_attachments(attachments: &[u8], metadata: &mut Metadata) {
	for (eid, file) in Children::new(attachments) {
		if eid != id::ATTACHED_FILE {
			continue;
		}
		let Some(data) = ebml::find(file, id::FILE_DATA) else { continue };
		let mime = ebml::child_string(file, id::FILE_MEDIA_TYPE).unwrap_or_default();
		let name = ebml::child_string(file, id::FILE_NAME).unwrap_or_default();
		let description = ebml::child_string(file, id::FILE_DESCRIPTION).or_else(|| Some(name.clone()));
		if mime.starts_with("image/") {
			metadata.images.push(AttachedImage { data: data.to_vec(), mime_type: mime, description });
		} else if !name.is_empty() {
			metadata
				.raw
				.insert(name.clone(), RawValue::File(AttachedFile { data: data.to_vec(), filename: name }));
		}
	}
}

impl Demuxer for MkvDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		loop {
			if let Some(packet) = self.queue.pop_front() {
				if self.accept(&packet) {
					return Ok(Some(packet));
				}
				continue;
			}
			if !self.next_element()? {
				return Ok(None);
			}
		}
	}

	/// Seeks to the last cue point (or, without cues, cluster) at or before
	/// `time`. Video then starts at its next keyframe and other streams at
	/// the cue time, as with FFmpeg's `matroska_read_seek`.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let target = (time * 1e9 / self.scale_ns as f64).min(u64::MAX as f64) as u64;
		let (position, start) = self.locate(target)?;
		self.reader.seek(position)?;
		self.queue.clear();
		self.cluster_ts = 0;
		let need_keyframe =
			self.streams.iter().filter(|s| s.kind == StreamKind::Video).map(|s| s.id.0).collect();
		self.seek_filter = Some(SeekFilter { start, need_keyframe });
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

impl MkvDemuxer {
	/// Offset of the segment's data (cue positions are relative to it).
	pub fn segment_start(&self) -> u64 {
		self.segment_start
	}

	/// Number of cue points loaded.
	pub fn cue_count(&self) -> usize {
		self.cues.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn lacing_variants() {
		// Xiph: 3 frames, sizes 2 and 256 (255+1), last = remainder (1).
		let mut data = vec![2, 2, 255, 1];
		data.extend(std::iter::repeat_n(0u8, 2 + 256 + 1));
		let mut c = ByteCursor::new(&data);
		assert_eq!(lace_sizes(&mut c, 1), Some(vec![2, 256, 1]));
		// Fixed: 2 frames over 6 bytes.
		let data = [1, 0, 0, 0, 0, 0, 0];
		assert_eq!(lace_sizes(&mut ByteCursor::new(&data), 2), Some(vec![3, 3]));
		assert_eq!(lace_sizes(&mut ByteCursor::new(&data[..6]), 2), None);
		// EBML: 3 frames, first 4, then +1 (0xC0 = signed vint +1), rest.
		let mut data = vec![2, 0x84, 0xC0];
		data.extend([0u8; 4 + 5 + 2]);
		assert_eq!(lace_sizes(&mut ByteCursor::new(&data), 3), Some(vec![4, 5, 2]));
		// Sizes past the end are rejected.
		assert_eq!(lace_sizes(&mut ByteCursor::new(&[1, 200, 0]), 1), None);
	}

	#[test]
	fn time_bases() {
		assert_eq!(time_base_for(1_000_000).unwrap(), TimeBase::new(1, 1000));
		assert_eq!(time_base_for(1).unwrap(), TimeBase::new(1, 1_000_000_000));
		assert_eq!(time_base_for(22_675).unwrap(), TimeBase::new(907, 40_000_000));
	}
}
