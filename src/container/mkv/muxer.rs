//! Matroska / WebM muxer (`.mkv`, `.mka`, `.mks`, `.webm`).
//!
//! Layout, as FFmpeg's `matroskaenc.c` writes it:
//!
//! ```text
//! EBML header
//! Segment (size patched on finish)
//! ├── Void (reserved, replaced by the SeekHead on finish)
//! ├── Info (TimestampScale 1 ms, Duration patched on finish, Title)
//! ├── Tracks
//! ├── Tags          (file tags and per-track tags, when there are any)
//! ├── Attachments   (cover art and other files; not in WebM)
//! ├── Cluster …     (Timestamp, SimpleBlock / BlockGroup)
//! └── Cues          (video keyframes, or each cluster's first audio block)
//! ```
//!
//! Each cluster is assembled in memory and written with its exact size, so
//! only the `Segment` size, the `SeekHead`, the `Info` duration and (for
//! FLAC) the STREAMINFO in `CodecPrivate` need a seek at the end. An output
//! that can't seek keeps the unknown segment size and an empty `Void`, which
//! is still a valid (live-style) Matroska file.
//!
//! The header is written together with the first cluster, so a video
//! track's `DefaultDuration` can be taken from the packets of that cluster.
//!
//! Timestamps follow RMP's gapless convention: a packet's `pts` is the
//! block timestamp, and an Opus stream's `encoder_delay` becomes
//! `CodecDelay` (FFmpeg shifts the timestamps by it instead, so the stored
//! values are the same).

use super::ebml::{self, id};
use super::tracks::{self, MuxCodec};
use crate::codecs::duration::packet_samples;
use crate::container::flac::stats::FlacStats;
use crate::container::xiph;
use crate::core::{
	AttachedFile, AttachedImage, Metadata, Muxer, Packet, RawValue, Stream, StreamId, StreamKind,
	TimeBase,
};
use crate::io::Io;
use crate::message::{Message, Result};

/// `TimestampScale`: 1 ms per tick (FFmpeg's default).
const TIMESTAMP_SCALE_NS: u64 = 1_000_000;
const MS: TimeBase = TimeBase::new(1, 1000);
const NS: TimeBase = TimeBase::new(1, 1_000_000_000);
/// A cluster is closed after this long or this many bytes (FFmpeg's
/// `cluster_time_limit` and `cluster_size_limit` defaults).
const CLUSTER_TIME_LIMIT_MS: u64 = 5000;
const CLUSTER_SIZE_LIMIT: usize = 5 * 1024 * 1024;
/// Bytes reserved for the SeekHead: room for five entries.
const SEEK_HEAD_RESERVE: usize = 128;
/// Opus decoders need 80 ms of pre-roll after a seek (RFC 7845 §4.6).
const OPUS_SEEK_PRE_ROLL_NS: u64 = 80_000_000;
/// Video packets looked at to find a constant frame duration.
const DURATION_SAMPLES: usize = 32;
const APP: &str = "RMP";

/// Matroska `TrackType` values.
const TYPE_VIDEO: u64 = 1;
const TYPE_AUDIO: u64 = 2;
const TYPE_SUBTITLE: u64 = 17;

struct Track {
	stream: Stream,
	number: u64,
	uid: u64,
	codec: MuxCodec,
	codec_delay_ns: u64,
	/// Native-time-base durations of the first video packets.
	durations: Vec<Option<u64>>,
	flac: Option<FlacStats>,
	/// Absolute offset of the `CodecPrivate` body, for the FLAC patch.
	private_offset: Option<u64>,
	/// End of the last block, in ms (timestamp for packets without one).
	next_ms: u64,
	/// The current cluster already has a cue for this track.
	cued: bool,
}

impl Track {
	fn kind(&self) -> StreamKind {
		self.stream.kind
	}
}

struct Cluster {
	timestamp: u64,
	body: Vec<u8>,
	blocks: usize,
	/// `(track number, time, offset of the block in the cluster body)`.
	cues: Vec<(u64, u64, u64)>,
}

#[derive(Debug, Clone, Copy)]
struct CuePoint {
	time: u64,
	track: u64,
	/// Cluster offset from the segment data start.
	cluster: u64,
	relative: u64,
}

pub struct MkvMuxer {
	io: Box<dyn Io>,
	webm: bool,
	metadata: Metadata,
	tracks: Vec<Track>,
	header_written: bool,
	/// Bytes written so far (the current output offset).
	position: u64,
	segment_size_offset: u64,
	segment_start: u64,
	duration_offset: u64,
	/// `(element id, absolute offset)` of the level-1 elements for the SeekHead.
	level1: Vec<(u32, u64)>,
	cluster: Option<Cluster>,
	cues: Vec<CuePoint>,
	end_ms: u64,
	has_video: bool,
}

/// SplitMix64: stable, well-spread track UIDs (FFmpeg draws random ones;
/// fixed values keep RMP's output reproducible).
fn track_uid(index: u64) -> u64 {
	let mut z = index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
	z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	(z ^ (z >> 31)).max(1)
}

fn string_value(metadata: &Metadata, key: &str) -> Option<String> {
	match metadata.raw.get(key) {
		Some(RawValue::String(v)) if !v.is_empty() => Some(v.clone()),
		_ => None,
	}
}

/// File tags as `(name, value)`, using the Matroska tag names FFmpeg's
/// `ff_mkv_metadata_conv` produces and the demuxer reads back.
fn global_tags(m: &Metadata) -> Vec<(String, String)> {
	let mut tags: Vec<(String, String)> = Vec::new();
	let mut push = |name: &str, value: &Option<String>| {
		if let Some(v) = value.as_ref().filter(|v| !v.is_empty()) {
			tags.push((name.to_string(), v.clone()));
		}
	};
	push("ARTIST", &m.artist);
	push("ALBUM", &m.album);
	push("ALBUM_ARTIST", &m.album_artist);
	push("GENRE", &m.genre);
	push("DATE", &m.date);
	push("COMMENT", &m.comment);
	push("DESCRIPTION", &m.description);
	push("LYRICS", &m.lyrics);
	if let Some(n) = m.track_number {
		tags.push(("PART_NUMBER".into(), n.to_string()));
	}
	if let Some(n) = m.tracks_total {
		tags.push(("TOTAL_PARTS".into(), n.to_string()));
	}
	if let Some(n) = m.disc_number {
		tags.push(("DISC".into(), n.to_string()));
	}
	tags.extend(raw_tags(m));
	tags
}

/// Free-form string tags, sorted for reproducible output.
fn raw_tags(m: &Metadata) -> Vec<(String, String)> {
	let mut tags: Vec<(String, String)> = m
		.raw
		.iter()
		.filter_map(|(k, v)| match v {
			RawValue::String(s) if !s.is_empty() && !k.is_empty() && !xiph::is_internal_key(k) => {
				Some((k.to_ascii_uppercase(), s.clone()))
			}
			_ => None,
		})
		.collect();
	tags.sort();
	tags
}

fn put_tag(out: &mut Vec<u8>, track_uid: Option<u64>, tags: &[(String, String)]) {
	ebml::put_master(out, id::TAG, |tag| {
		ebml::put_master(tag, id::TARGETS, |targets| {
			if let Some(uid) = track_uid {
				ebml::put_uint(targets, id::TAG_TRACK_UID, uid);
			}
		});
		for (name, value) in tags {
			ebml::put_master(tag, id::SIMPLE_TAG, |simple| {
				ebml::put_string(simple, id::TAG_NAME, name);
				ebml::put_string(simple, id::TAG_STRING, value);
			});
		}
	});
}

fn image_extension(mime: &str) -> &'static str {
	match mime {
		"image/png" => "png",
		"image/gif" => "gif",
		"image/bmp" => "bmp",
		"image/webp" => "webp",
		_ => "jpg",
	}
}

fn file_mime(name: &str) -> &'static str {
	let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
	match ext.as_str() {
		"ttf" => "font/ttf",
		"otf" => "font/otf",
		"woff" => "font/woff",
		"woff2" => "font/woff2",
		"txt" => "text/plain",
		"jpg" | "jpeg" => "image/jpeg",
		"png" => "image/png",
		_ => "application/octet-stream",
	}
}

fn put_attachment(
	out: &mut Vec<u8>,
	name: &str,
	mime: &str,
	description: Option<&str>,
	data: &[u8],
	uid: u64,
) {
	ebml::put_master(out, id::ATTACHED_FILE, |file| {
		if let Some(d) = description.filter(|d| !d.is_empty() && *d != name) {
			ebml::put_string(file, id::FILE_DESCRIPTION, d);
		}
		ebml::put_string(file, id::FILE_NAME, name);
		ebml::put_string(file, id::FILE_MEDIA_TYPE, mime);
		ebml::put_element(file, id::FILE_DATA, data);
		ebml::put_uint(file, id::FILE_UID, uid);
	});
}

/// Header length of an element whose body is `body_len` bytes.
fn header_len(element_id: u32, body_len: usize) -> u64 {
	let mut out = Vec::new();
	ebml::put_id(&mut out, element_id);
	(out.len() + ebml::size_len(body_len as u64)) as u64
}

impl MkvMuxer {
	pub fn new(io: Box<dyn Io>, webm: bool) -> Self {
		Self {
			io,
			webm,
			metadata: Metadata::default(),
			tracks: Vec::new(),
			header_written: false,
			position: 0,
			segment_size_offset: 0,
			segment_start: 0,
			duration_offset: 0,
			level1: Vec::new(),
			cluster: None,
			cues: Vec::new(),
			end_ms: 0,
			has_video: false,
		}
	}

	fn emit(&mut self, bytes: &[u8]) -> Result<()> {
		self.io.write_all(bytes)?;
		self.position += bytes.len() as u64;
		Ok(())
	}

	/// `DefaultDuration` of a video track: the first packets' duration when
	/// they all agree.
	fn default_duration_ns(track: &Track) -> Option<u64> {
		let first = (*track.durations.first()?)?;
		if first == 0 || track.durations.iter().any(|d| *d != Some(first)) {
			return None;
		}
		crate::core::Time::new(first, track.stream.time_base).rescale(NS).map(|t| t.ticks())
	}

	fn track_entry(&self, track: &Track) -> (Vec<u8>, Option<u64>) {
		let s = &track.stream;
		let p = &s.parameters;
		let mut e = Vec::new();
		ebml::put_uint(&mut e, id::TRACK_NUMBER, track.number);
		ebml::put_uint(&mut e, id::TRACK_UID, track.uid);
		let kind = match s.kind {
			StreamKind::Video => TYPE_VIDEO,
			StreamKind::Audio => TYPE_AUDIO,
			_ => TYPE_SUBTITLE,
		};
		ebml::put_uint(&mut e, id::TRACK_TYPE, kind);
		ebml::put_uint(&mut e, id::FLAG_LACING, 0);
		if string_value(&s.metadata, "default").as_deref() == Some("0") {
			ebml::put_uint(&mut e, id::FLAG_DEFAULT, 0);
		}
		// The demuxer (like FFmpeg's) reads a missing Language as "eng".
		let language = string_value(&s.metadata, "language").unwrap_or_else(|| "und".into());
		ebml::put_string(&mut e, id::LANGUAGE, &language);
		if let Some(name) = s.metadata.title.as_deref().filter(|n| !n.is_empty()) {
			ebml::put_string(&mut e, id::NAME, name);
		}
		ebml::put_string(&mut e, id::CODEC_ID, track.codec.codec_id);
		let mut private_at = None;
		if !track.codec.private.is_empty() {
			let element_header = header_len(id::CODEC_PRIVATE, track.codec.private.len());
			private_at = Some(e.len() as u64 + element_header);
			ebml::put_element(&mut e, id::CODEC_PRIVATE, &track.codec.private);
		}
		if s.kind == StreamKind::Video {
			if let Some(ns) = Self::default_duration_ns(track) {
				ebml::put_uint(&mut e, id::DEFAULT_DURATION, ns);
			}
		}
		if track.codec_delay_ns > 0 {
			ebml::put_uint(&mut e, id::CODEC_DELAY, track.codec_delay_ns);
		}
		if p.codec.id == "opus" {
			ebml::put_uint(&mut e, id::SEEK_PRE_ROLL, OPUS_SEEK_PRE_ROLL_NS);
		}
		match s.kind {
			StreamKind::Video => ebml::put_master(&mut e, id::VIDEO, |v| {
				ebml::put_uint(v, id::PIXEL_WIDTH, u64::from(p.width.unwrap_or(0)));
				ebml::put_uint(v, id::PIXEL_HEIGHT, u64::from(p.height.unwrap_or(0)));
				if let Some(fourcc) = track.codec.colour_space {
					ebml::put_element(v, id::COLOUR_SPACE, &fourcc);
				}
			}),
			StreamKind::Audio => ebml::put_master(&mut e, id::AUDIO, |a| {
				let rate = if p.codec.id == "opus" { 48_000 } else { p.sample_rate.unwrap_or(0) };
				ebml::put_float(a, id::SAMPLING_FREQUENCY, f64::from(rate));
				ebml::put_uint(a, id::CHANNELS, u64::from(p.channels.unwrap_or(1)));
				let depth = track.codec.bit_depth.or(p.bits_per_sample.filter(|&b| b > 0));
				if let Some(depth) = depth {
					ebml::put_uint(a, id::BIT_DEPTH, u64::from(depth));
				}
			}),
			_ => {}
		}
		(e, private_at)
	}

	fn write_header(&mut self) -> Result<()> {
		let mut out = Vec::new();
		let (doc_type, version) = if self.webm { ("webm", 4) } else { ("matroska", 4) };
		ebml::put_master(&mut out, id::EBML, |h| {
			ebml::put_uint(h, id::EBML_VERSION, 1);
			ebml::put_uint(h, id::EBML_READ_VERSION, 1);
			ebml::put_uint(h, id::EBML_MAX_ID_LENGTH, 4);
			ebml::put_uint(h, id::EBML_MAX_SIZE_LENGTH, 8);
			ebml::put_string(h, id::DOC_TYPE, doc_type);
			ebml::put_uint(h, id::DOC_TYPE_VERSION, version);
			ebml::put_uint(h, id::DOC_TYPE_READ_VERSION, 2);
		});
		ebml::put_id(&mut out, id::SEGMENT);
		self.segment_size_offset = self.position + out.len() as u64;
		out.extend_from_slice(&ebml::UNKNOWN_SIZE);
		self.segment_start = self.position + out.len() as u64;
		ebml::put_void(&mut out, SEEK_HEAD_RESERVE);

		// Info, with the Duration value's offset noted for the patch.
		let mut info = Vec::new();
		ebml::put_uint(&mut info, id::TIMESTAMP_SCALE, TIMESTAMP_SCALE_NS);
		ebml::put_string(&mut info, id::MUXING_APP, APP);
		ebml::put_string(&mut info, id::WRITING_APP, APP);
		let duration_at = info.len() as u64 + header_len(id::DURATION, 8);
		ebml::put_float(&mut info, id::DURATION, 0.0);
		if let Some(title) = self.metadata.title.as_deref().filter(|t| !t.is_empty()) {
			ebml::put_string(&mut info, id::TITLE, title);
		}
		let info_at = self.position + out.len() as u64;
		self.level1.push((id::INFO, info_at));
		self.duration_offset = info_at + header_len(id::INFO, info.len()) + duration_at;
		ebml::put_element(&mut out, id::INFO, &info);

		// Tracks, noting where each FLAC CodecPrivate lands.
		let mut body = Vec::new();
		let mut private_offsets = Vec::new();
		for track in &self.tracks {
			let (entry, private_at) = self.track_entry(track);
			let entry_at = body.len() as u64 + header_len(id::TRACK_ENTRY, entry.len());
			private_offsets.push(private_at.map(|p| entry_at + p));
			ebml::put_element(&mut body, id::TRACK_ENTRY, &entry);
		}
		let tracks_at = self.position + out.len() as u64;
		let tracks_body_at = tracks_at + header_len(id::TRACKS, body.len());
		for (track, offset) in self.tracks.iter_mut().zip(private_offsets) {
			track.private_offset = offset.map(|o| tracks_body_at + o);
		}
		self.level1.push((id::TRACKS, tracks_at));
		ebml::put_element(&mut out, id::TRACKS, &body);

		let mut tags = Vec::new();
		let file_tags = global_tags(&self.metadata);
		if !file_tags.is_empty() {
			put_tag(&mut tags, None, &file_tags);
		}
		for track in &self.tracks {
			let track_tags = raw_tags(&track.stream.metadata);
			if !track_tags.is_empty() {
				put_tag(&mut tags, Some(track.uid), &track_tags);
			}
		}
		if !tags.is_empty() {
			self.level1.push((id::TAGS, self.position + out.len() as u64));
			ebml::put_element(&mut out, id::TAGS, &tags);
		}

		if !self.webm {
			let mut files = Vec::new();
			let mut uid = 1u64;
			for (i, AttachedImage { data, mime_type, description }) in
				self.metadata.images.iter().enumerate()
			{
				let stem = if i == 0 { "cover".to_string() } else { format!("cover{}", i + 1) };
				let name = format!("{stem}.{}", image_extension(mime_type));
				put_attachment(
					&mut files,
					&name,
					mime_type,
					description.as_deref(),
					data,
					track_uid(1000 + uid),
				);
				uid += 1;
			}
			let mut raw_files: Vec<&AttachedFile> = self
				.metadata
				.raw
				.values()
				.filter_map(|v| if let RawValue::File(f) = v { Some(f) } else { None })
				.collect();
			raw_files.sort_by(|a, b| a.filename.cmp(&b.filename));
			for f in raw_files {
				put_attachment(
					&mut files,
					&f.filename,
					file_mime(&f.filename),
					None,
					&f.data,
					track_uid(1000 + uid),
				);
				uid += 1;
			}
			if !files.is_empty() {
				self.level1.push((id::ATTACHMENTS, self.position + out.len() as u64));
				ebml::put_element(&mut out, id::ATTACHMENTS, &files);
			}
		}

		self.emit(&out)?;
		self.header_written = true;
		Ok(())
	}

	fn flush_cluster(&mut self) -> Result<()> {
		let Some(cluster) = self.cluster.take() else { return Ok(()) };
		if !self.header_written {
			self.write_header()?;
		}
		let at = self.position - self.segment_start;
		let mut out = Vec::with_capacity(cluster.body.len() + 12);
		ebml::put_element(&mut out, id::CLUSTER, &cluster.body);
		self.emit(&out)?;
		for (track, time, relative) in cluster.cues {
			self.cues.push(CuePoint { time, track, cluster: at, relative });
		}
		for track in &mut self.tracks {
			track.cued = false;
		}
		Ok(())
	}

	fn write_cues(&mut self) -> Result<()> {
		if self.cues.is_empty() {
			return Ok(());
		}
		let mut body = Vec::new();
		let mut i = 0;
		while i < self.cues.len() {
			// One CuePoint per time, with a position for each track cued there.
			let time = self.cues[i].time;
			let mut j = i;
			while j < self.cues.len() && self.cues[j].time == time {
				j += 1;
			}
			let group = &self.cues[i..j];
			ebml::put_master(&mut body, id::CUE_POINT, |point| {
				ebml::put_uint(point, id::CUE_TIME, time);
				for cue in group {
					ebml::put_master(point, id::CUE_TRACK_POSITIONS, |positions| {
						ebml::put_uint(positions, id::CUE_TRACK, cue.track);
						ebml::put_uint(positions, id::CUE_CLUSTER_POSITION, cue.cluster);
						ebml::put_uint(positions, id::CUE_RELATIVE_POSITION, cue.relative);
					});
				}
			});
			i = j;
		}
		self.level1.push((id::CUES, self.position));
		let mut out = Vec::new();
		ebml::put_element(&mut out, id::CUES, &body);
		self.emit(&out)
	}

	/// Back-patches the segment size, SeekHead, duration and FLAC
	/// STREAMINFO. Outputs that can't seek keep the provisional values.
	fn patch(&mut self) -> Result<()> {
		let end = self.position;
		if self.io.seek(self.segment_size_offset).is_err() {
			return Ok(());
		}
		let mut size = Vec::new();
		ebml::put_size_len(&mut size, end - self.segment_start, 8);
		self.io.write_all(&size)?;

		let mut seeks = Vec::new();
		for &(element, offset) in &self.level1 {
			ebml::put_master(&mut seeks, id::SEEK, |seek| {
				let mut element_id = Vec::new();
				ebml::put_id(&mut element_id, element);
				ebml::put_element(seek, id::SEEK_ID, &element_id);
				ebml::put_uint(seek, id::SEEK_POSITION, offset - self.segment_start);
			});
		}
		let mut head = Vec::new();
		ebml::put_element(&mut head, id::SEEK_HEAD, &seeks);
		if head.len() + 2 <= SEEK_HEAD_RESERVE {
			let free = SEEK_HEAD_RESERVE - head.len();
			ebml::put_void(&mut head, free);
			self.io.seek(self.segment_start)?;
			self.io.write_all(&head)?;
		}

		self.io.seek(self.duration_offset)?;
		self.io.write_all(&(self.end_ms as f64).to_bits().to_be_bytes())?;

		for i in 0..self.tracks.len() {
			let (Some(offset), Some(stats)) =
				(self.tracks[i].private_offset, self.tracks[i].flac.as_mut())
			else {
				continue;
			};
			let info = stats.finish();
			// "fLaC" and the STREAMINFO block header precede the body.
			self.io.seek(offset + 8)?;
			self.io.write_all(&info.to_bytes())?;
		}
		self.io.seek(end)?;
		Ok(())
	}

	fn track_index(&self, stream: StreamId) -> Result<usize> {
		self
			.tracks
			.iter()
			.position(|t| t.stream.id == stream)
			.ok_or(Message::Container("packet for a stream that was not added to the matroska muxer"))
	}
}

impl Muxer for MkvMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.header_written || self.cluster.is_some() {
			return Err(Message::Container("matroska streams must be added before the first packet"));
		}
		if self.tracks.iter().any(|t| t.stream.id == stream.id) {
			return Err(Message::Container("matroska muxer: duplicate stream id"));
		}
		if self.tracks.len() >= 126 {
			return Err(Message::Container("matroska muxer: too many tracks"));
		}
		if stream.kind == StreamKind::Other {
			return Err(Message::Container("matroska muxer: unsupported stream type"));
		}
		let codec = tracks::mux_codec(&stream.parameters, self.webm).map_err(Message::Container)?;
		let p = &stream.parameters;
		let delay = string_value(&stream.metadata, "encoder_delay").and_then(|v| v.parse::<u64>().ok());
		let codec_delay_ns = match (p.codec.id.as_str(), delay) {
			("opus", d) => {
				let d = d.or_else(|| crate::codecs::opus::header::pre_skip(&p.extradata).map(u64::from));
				crate::core::rescale_ticks(d.unwrap_or(0), TimeBase::new(1, 48_000), NS).unwrap_or(0)
			}
			(_, Some(d)) if stream.kind == StreamKind::Audio => match p.sample_rate {
				Some(rate) if rate > 0 => {
					crate::core::rescale_ticks(d, TimeBase::new(1, rate), NS).unwrap_or(0)
				}
				_ => 0,
			},
			_ => 0,
		};
		let flac = (p.codec.id == "flac")
			.then(|| crate::codecs::flac::StreamInfo::from_extradata(&p.extradata))
			.flatten()
			.map(FlacStats::new);
		let index = self.tracks.len();
		self.has_video |= stream.kind == StreamKind::Video;
		self.tracks.push(Track {
			stream: stream.clone(),
			number: index as u64 + 1,
			uid: track_uid(index as u64),
			codec,
			codec_delay_ns,
			durations: Vec::new(),
			flac,
			private_offset: None,
			next_ms: 0,
			cued: false,
		});
		Ok(index)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		let index = self.track_index(packet.stream_id)?;
		let to_ms = |t: Option<crate::core::Time>| t.and_then(|t| t.rescale(MS)).map(|t| t.ticks());
		let track = &mut self.tracks[index];
		if let Some(stats) = track.flac.as_mut() {
			stats.add(&packet.data);
		}
		let timestamp = to_ms(packet.pts.or(packet.dts)).unwrap_or(track.next_ms);
		let duration = to_ms(packet.duration);
		if track.kind() == StreamKind::Video
			&& track.durations.len() < DURATION_SAMPLES
			&& !self.header_written
		{
			track.durations.push(packet.duration.map(|d| d.ticks()));
		}
		let end = timestamp.saturating_add(duration.unwrap_or(0));
		track.next_ms = end;
		self.end_ms = self.end_ms.max(end);
		let kind = track.kind();
		let number = track.number;
		let keyframe = kind != StreamKind::Video || packet.keyframe;
		let webvtt = track.stream.parameters.codec.id == "webvtt";
		// An Opus packet shorter than its coded duration is end-trimmed:
		// Matroska says so with DiscardPadding (in ns) on a BlockGroup.
		let discard_ns = if track.stream.parameters.codec.id == "opus" {
			let coded = packet_samples(&track.stream.parameters, &packet.data).map(|(n, _)| n);
			let kept =
				packet.duration.and_then(|d| d.rescale(TimeBase::new(1, 48_000))).map(|d| d.ticks());
			match (coded, kept) {
				(Some(c), Some(k)) if k < c => {
					crate::core::rescale_ticks(c - k, TimeBase::new(1, 48_000), NS).unwrap_or(0)
				}
				_ => 0,
			}
		} else {
			0
		};

		let start_new = match &self.cluster {
			None => true,
			Some(c) => {
				let relative = i128::from(timestamp) - i128::from(c.timestamp);
				c.blocks > 0
					&& ((kind == StreamKind::Video && keyframe)
						|| relative < i128::from(i16::MIN)
						|| relative > i128::from(i16::MAX)
						|| timestamp.saturating_sub(c.timestamp) >= CLUSTER_TIME_LIMIT_MS
						|| c.body.len() >= CLUSTER_SIZE_LIMIT)
			}
		};
		if start_new {
			self.flush_cluster()?;
			let mut body = Vec::new();
			ebml::put_uint(&mut body, id::TIMESTAMP, timestamp);
			self.cluster = Some(Cluster { timestamp, body, blocks: 0, cues: Vec::new() });
		}
		let has_video = self.has_video;
		let track = &mut self.tracks[index];
		let Some(cluster) = self.cluster.as_mut() else {
			return Err(Message::Container("matroska muxer: no open cluster"));
		};
		let relative = (i128::from(timestamp) - i128::from(cluster.timestamp)) as i16;

		let mut block = Vec::with_capacity(packet.data.len() + 8);
		ebml::put_size(&mut block, number);
		block.extend_from_slice(&relative.to_be_bytes());
		let offset = cluster.body.len() as u64;
		let payload: &[u8] = &packet.data;
		if kind == StreamKind::Subtitle {
			// Subtitles need their duration, so they go in a BlockGroup.
			block.push(0);
			if webvtt {
				// WebM WebVTT blocks: cue identifier and settings lines first
				// (both empty here), then the cue text.
				block.extend_from_slice(b"\n\n");
			}
			block.extend_from_slice(payload);
			ebml::put_master(&mut cluster.body, id::BLOCK_GROUP, |group| {
				ebml::put_element(group, id::BLOCK, &block);
				if let Some(d) = duration {
					ebml::put_uint(group, id::BLOCK_DURATION, d);
				}
			});
		} else if discard_ns > 0 {
			block.push(0);
			block.extend_from_slice(payload);
			ebml::put_master(&mut cluster.body, id::BLOCK_GROUP, |group| {
				ebml::put_element(group, id::BLOCK, &block);
				let mut padding = Vec::new();
				ebml::put_sint(
					&mut padding,
					id::DISCARD_PADDING,
					i64::try_from(discard_ns).unwrap_or(i64::MAX),
				);
				group.extend_from_slice(&padding);
			});
		} else {
			block.push(if keyframe { 0x80 } else { 0 });
			block.extend_from_slice(payload);
			ebml::put_element(&mut cluster.body, id::SIMPLE_BLOCK, &block);
		}
		cluster.blocks += 1;
		let cue = match kind {
			StreamKind::Video => keyframe,
			StreamKind::Audio => !has_video && !track.cued,
			_ => false,
		};
		if cue {
			cluster.cues.push((number, timestamp, offset));
			track.cued = true;
		}
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		self.flush_cluster()?;
		if !self.header_written {
			if self.tracks.is_empty() {
				return Ok(());
			}
			self.write_header()?;
		}
		self.write_cues()?;
		self.patch()
	}
}
