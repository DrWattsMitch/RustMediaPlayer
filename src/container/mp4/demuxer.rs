//! MP4 / M4A / MOV demuxer, plain and fragmented ISO BMFF.
//!
//! Opening a file walks the top-level boxes until `moov`, loads it into
//! memory, and builds a flat sample index per track
//! ([`super::sample_table`]). When `moov` has an `mvex` (fragmented MP4),
//! the rest of the file is scanned for `moof` boxes, skipping `mdat`
//! payloads, and their samples are appended to the same index
//! ([`super::fragment::parse_moof`]); a file cut short mid-fragment keeps
//! every complete sample. Reading then returns samples from all tracks
//! interleaved by decode time, seeking to each sample's offset in `mdat`.
//! This follows FFmpeg's `mov.c` (`mov_read_header` + `mov_read_packet`).

use super::boxes::{BoxIter, find, find_path, full_box};
use super::codec::{self, CodecInfo};
use super::fragment::{self, FragmentedTrack, TrackDefaults};
use super::metadata;
use super::sample_table::{self, Sample};
use crate::core::{
	CodecId, Demuxer, Metadata, Packet, Parameters, RawValue, Stream, StreamId, StreamKind,
	StreamSet, Time, TimeBase,
};
use crate::io::{BufferedReader, ByteCursor, Io};
use crate::message::{Message, Result};

/// Largest `moov` box we are willing to load (sample tables of very long
/// files are a few tens of MB; anything far larger is corrupt).
const MAX_MOOV_SIZE: u64 = 256 * 1024 * 1024;
/// Largest `moof` we load (FFmpeg-written ones are a few KB).
const MAX_MOOF_SIZE: u64 = 64 * 1024 * 1024;

/// Stream metadata keys for audio priming / trailing padding (in samples),
/// derived from the edit list. Same keys as the MP3 demuxer uses, so audio
/// gapless information survives MP3 → MP4 and MP4 → MP4 remuxing.
pub const ENCODER_DELAY_KEY: &str = "encoder_delay";
pub const ENCODER_PADDING_KEY: &str = "encoder_padding";

/// First real edit of an edit list.
#[derive(Debug, Clone, Copy, Default)]
struct Edit {
	/// Presentation shift in media ticks (empty-edit delay − media time).
	shift: i64,
	media_time: i64,
	/// Duration of the edit in media ticks (0 = unknown / whole track).
	segment: u64,
}

struct Track {
	timescale: u32,
	samples: Vec<Sample>,
	/// Index of the next sample to return.
	next: usize,
	/// Presentation shift from the edit list, in track ticks.
	pts_shift: i64,
	has_sync_table: bool,
}

pub struct Mp4Demuxer {
	reader: BufferedReader,
	metadata: Metadata,
	streams: StreamSet,
	tracks: Vec<Track>,
	duration: Time,
}

/// Locates and loads the `moov` box, scanning top-level boxes.
fn load_moov(reader: &mut BufferedReader) -> Result<Vec<u8>> {
	let file_size = reader.size();
	loop {
		let position = reader.position();
		let head = reader.fill(16)?;
		if head.len() < 8 {
			return Err(Message::Container("mp4 moov box not found"));
		}
		let mut c = ByteCursor::new(head);
		let size32 = c.read_u32()?;
		let kind = c.read_fourcc()?;
		let (size, header) = match size32 {
			0 => match file_size {
				Some(total) => (total.saturating_sub(position), 8),
				None => return Err(Message::Container("mp4 moov box not found")),
			},
			1 => (c.read_u64()?, 16),
			n => (u64::from(n), 8),
		};
		if size < header {
			return Err(Message::Container("invalid mp4 box size"));
		}
		match &kind {
			b"moov" => {
				let body = size - header;
				if body > MAX_MOOV_SIZE {
					return Err(Message::Container("mp4 moov box too large"));
				}
				reader.skip(header)?;
				return reader.read_vec(body as usize);
			}
			_ => {
				let next = position.checked_add(size).ok_or(Message::Container("invalid mp4 box size"))?;
				if file_size.is_some_and(|total| next >= total) {
					return Err(Message::Container("mp4 moov box not found"));
				}
				reader.seek(next)?;
			}
		}
	}
}

/// Parses `mvhd` → (timescale, duration).
fn parse_mvhd(body: &[u8]) -> Result<(u32, u64)> {
	let (version, _, payload) = full_box(body)?;
	let mut c = ByteCursor::new(payload);
	if version == 1 {
		c.skip(16)?;
		Ok((c.read_u32()?, c.read_u64()?))
	} else {
		c.skip(8)?;
		Ok((c.read_u32()?, u64::from(c.read_u32()?)))
	}
}

/// A top-level box header: file position, type, header size, total size.
type BoxHeader = (u64, [u8; 4], u64, u64);

/// Reads the header of the top-level box at the reader's position. `None`
/// at the end of input.
fn next_box(reader: &mut BufferedReader, file_size: Option<u64>) -> Result<Option<BoxHeader>> {
	let position = reader.position();
	let head = reader.fill(16)?;
	if head.len() < 8 {
		return Ok(None);
	}
	let mut c = ByteCursor::new(head);
	let size32 = c.read_u32()?;
	let kind = c.read_fourcc()?;
	let (size, header) = match size32 {
		0 => match file_size {
			Some(total) => (total.saturating_sub(position), 8),
			None => return Ok(None),
		},
		1 => (c.read_u64()?, 16),
		n => (u64::from(n), 8),
	};
	if size < header {
		return Err(Message::Container("invalid mp4 box size"));
	}
	Ok(Some((position, kind, header, size)))
}

/// `tkhd` → track id.
fn parse_tkhd_id(body: &[u8]) -> Result<u32> {
	let (version, _, payload) = full_box(body)?;
	let mut c = ByteCursor::new(payload);
	c.skip(if version == 1 { 16 } else { 8 })?;
	c.read_u32()
}

/// `mvex/mehd` → fragmented movie duration (movie timescale), if set.
fn parse_mehd(mvex: &[u8]) -> Option<u64> {
	let (version, _, payload) = full_box(find(mvex, b"mehd")?).ok()?;
	let mut c = ByteCursor::new(payload);
	let duration = if version == 1 { c.read_u64().ok()? } else { u64::from(c.read_u32().ok()?) };
	Some(duration).filter(|&d| d > 0)
}

/// Decode-time span of a sample index, in track ticks.
fn track_span(samples: &[Sample]) -> u64 {
	match (samples.first(), samples.last()) {
		(Some(first), Some(last)) => {
			last.dts.saturating_add(u64::from(last.duration)).saturating_sub(first.dts)
		}
		_ => 0,
	}
}

/// Parses `mdhd` → (timescale, duration, ISO-639-2 language).
fn parse_mdhd(body: &[u8]) -> Result<(u32, u64, Option<String>)> {
	let (version, _, payload) = full_box(body)?;
	let mut c = ByteCursor::new(payload);
	let (timescale, duration) = if version == 1 {
		c.skip(16)?;
		(c.read_u32()?, c.read_u64()?)
	} else {
		c.skip(8)?;
		(c.read_u32()?, u64::from(c.read_u32()?))
	};
	let packed = c.read_u16().unwrap_or(0);
	let language: String = [10u16, 5, 0]
		.iter()
		.map(|shift| char::from((((packed >> shift) & 0x1F) as u8) + 0x60))
		.collect();
	let language = Some(language).filter(|l| l != "und" && l.chars().all(|c| c.is_ascii_lowercase()));
	Ok((timescale, duration, language))
}

fn parse_handler(hdlr: &[u8]) -> StreamKind {
	match hdlr.get(8..12) {
		Some(b"vide") => StreamKind::Video,
		Some(b"soun") => StreamKind::Audio,
		Some(b"sbtl" | b"subt" | b"text") => StreamKind::Subtitle,
		_ => StreamKind::Other,
	}
}

/// Interprets the edit list as a single presentation shift (in media
/// ticks): leading empty edits delay the track, the first real edit's
/// media time skips into it.
fn parse_edit_list(trak: &[u8], movie_timescale: u32, media_timescale: u32) -> Edit {
	let none = Edit::default();
	let Some(elst) = find_path(trak, &[b"edts", b"elst"]) else { return none };
	let Ok((version, _, payload)) = full_box(elst) else { return none };
	let mut c = ByteCursor::new(payload);
	let Ok(count) = c.read_u32() else { return none };
	let to_media = |ticks: u64| {
		if movie_timescale == 0 {
			return 0;
		}
		crate::core::rescale_ticks(
			ticks,
			TimeBase::new(1, movie_timescale),
			TimeBase::new(1, media_timescale),
		)
		.unwrap_or(0)
	};
	let mut empty_ticks: u64 = 0;
	for _ in 0..count.min(1024) {
		let entry = if version == 1 {
			c.read_u64().and_then(|d| c.read_u64().map(|m| (d, m as i64)))
		} else {
			c.read_u32().and_then(|d| c.read_i32().map(|m| (u64::from(d), i64::from(m))))
		};
		let Ok((segment_duration, media_time)) = entry else { break };
		if c.skip(4).is_err() {
			break; // media rate
		}
		if media_time == -1 {
			empty_ticks = empty_ticks.saturating_add(segment_duration);
			continue;
		}
		let delay = i64::try_from(to_media(empty_ticks)).unwrap_or(0);
		return Edit {
			shift: delay.saturating_sub(media_time),
			media_time,
			segment: to_media(segment_duration),
		};
	}
	none
}

fn build_stream(index: usize, kind: StreamKind, timescale: u32, info: CodecInfo) -> Stream {
	let parameters = Parameters {
		codec: CodecId::new(&info.codec),
		profile: None,
		bitrate: info.bitrate,
		sample_rate: info.sample_rate,
		channels: info.channels,
		width: info.width,
		height: info.height,
		pixel_format: None,
		extradata: info.extradata,
		bits_per_sample: None,
	};
	Stream {
		id: StreamId(index),
		kind,
		time_base: TimeBase::new(1, timescale),
		duration: None,
		parameters,
		metadata: Metadata::default(),
	}
}

impl Mp4Demuxer {
	pub fn new(io: Box<dyn Io>) -> Result<Self> {
		let mut reader = BufferedReader::new(io);
		let file_size = reader.size();
		let moov = load_moov(&mut reader)?;

		let mvhd = find(&moov, b"mvhd").ok_or(Message::Container("mp4 mvhd box not found"))?;
		let (movie_timescale, movie_duration) = parse_mvhd(mvhd)?;

		let mvex = find(&moov, b"mvex");
		let mut parsed = Vec::new();
		for (kind, trak) in BoxIter::new(&moov).map_while(|r| r.ok()) {
			if &kind != b"trak" {
				continue;
			}
			// A malformed track is skipped rather than failing the file,
			// so the remaining tracks stay playable.
			if let Ok(Some(track)) = Self::parse_track(trak, movie_timescale, file_size, mvex.is_some()) {
				parsed.push(track);
			}
		}
		if let Some(mvex) = mvex {
			Self::read_fragments(&mut reader, mvex, file_size, &mut parsed);
		}

		let mut streams = StreamSet::default();
		let mut tracks = Vec::new();
		for (_, mut stream, track) in parsed.into_iter().filter(|(_, _, t)| !t.samples.is_empty()) {
			stream.id = StreamId(tracks.len());
			if stream.duration.is_none_or(|d| d.ticks() == 0) {
				let span = track_span(&track.samples);
				stream.duration = Some(Time::new(span, stream.time_base));
			}
			streams.add(stream);
			tracks.push(track);
		}
		if tracks.is_empty() {
			return Err(Message::Container("mp4 file has no playable tracks"));
		}
		// Fragmented files leave mvhd's duration at 0; mehd or the
		// fragments themselves give it.
		let movie_duration = match mvex.and_then(parse_mehd) {
			Some(d) if movie_duration == 0 => d,
			_ => movie_duration,
		};

		let mut metadata = Metadata::default();
		metadata::parse_moov_metadata(&moov, &mut metadata);

		let duration = if movie_timescale > 0 && movie_duration > 0 {
			Time::new(movie_duration, TimeBase::new(1, movie_timescale))
		} else {
			streams
				.iter()
				.filter_map(|s| s.duration)
				.max_by(|a, b| a.as_seconds().total_cmp(&b.as_seconds()))
				.unwrap_or(Time::zero(TimeBase::new(1, 1000)))
		};

		Ok(Self { reader, metadata, streams, tracks, duration })
	}

	/// Parses one `trak` → (track id, stream, track). A fragmented file's
	/// tracks may have no samples yet.
	fn parse_track(
		trak: &[u8],
		movie_timescale: u32,
		file_size: Option<u64>,
		fragmented: bool,
	) -> Result<Option<(u32, Stream, Track)>> {
		let track_id = find(trak, b"tkhd").map_or(Ok(0), parse_tkhd_id)?;
		let Some(mdia) = find(trak, b"mdia") else { return Ok(None) };
		let mdhd = find(mdia, b"mdhd").ok_or(Message::Container("mp4 mdhd box not found"))?;
		let (timescale, media_duration, language) = parse_mdhd(mdhd)?;
		if timescale == 0 {
			return Err(Message::Container("mp4 track timescale is zero"));
		}
		let kind = find(mdia, b"hdlr").map_or(StreamKind::Other, parse_handler);
		let stbl = find_path(mdia, &[b"minf", b"stbl"])
			.ok_or(Message::Container("mp4 sample table not found"))?;
		let info = match find(stbl, b"stsd") {
			Some(stsd) => codec::parse_stsd(stsd, kind)?,
			None => return Ok(None),
		};
		let samples = match sample_table::build_index(stbl, file_size) {
			// Fragmented files often omit or empty the sample tables.
			Err(_) if fragmented => Vec::new(),
			other => other?,
		};
		if samples.is_empty() && !fragmented {
			return Ok(None);
		}

		let edit = parse_edit_list(trak, movie_timescale, timescale);
		let mut stream = build_stream(0, kind, timescale, info);
		stream.duration = Some(Time::new(media_duration, stream.time_base));
		if let Some(language) = language {
			stream.metadata.raw.insert("language".into(), RawValue::String(language));
		}

		// Timestamps are unsigned, so audio priming (a negative shift that
		// would push the first packets before zero) is exposed as metadata
		// instead, mirroring how the MP3 demuxer reports encoder delay.
		let pts_shift = if kind == StreamKind::Audio && edit.shift < 0 {
			let raw = &mut stream.metadata.raw;
			raw.insert(ENCODER_DELAY_KEY.into(), RawValue::String(edit.media_time.to_string()));
			let used = (edit.media_time as u64).saturating_add(edit.segment);
			if edit.segment > 0 && used < media_duration {
				let padding = media_duration - used;
				raw.insert(ENCODER_PADDING_KEY.into(), RawValue::String(padding.to_string()));
			}
			0
		} else {
			edit.shift
		};

		let has_sync_table = find(stbl, b"stss").is_some();
		Ok(Some((track_id, stream, Track { timescale, samples, next: 0, pts_shift, has_sync_table })))
	}

	/// Scans the top-level boxes after `moov` for `moof`s and appends their
	/// samples. Stops quietly at the first box that can't be read, so a
	/// partly written file plays up to its last complete fragment.
	fn read_fragments(
		reader: &mut BufferedReader,
		mvex: &[u8],
		file_size: Option<u64>,
		parsed: &mut [(u32, Stream, Track)],
	) {
		let trex = fragment::parse_trex(mvex);
		let mut tracks: Vec<FragmentedTrack> = parsed
			.iter()
			.map(|(id, _, track)| FragmentedTrack {
				track_id: *id,
				defaults: trex.iter().find(|(t, _)| t == id).map_or(TrackDefaults::default(), |(_, d)| *d),
				next_dts: track.samples.last().map_or(0, |s| s.dts.saturating_add(u64::from(s.duration))),
				samples: Vec::new(),
			})
			.collect();
		let max_samples =
			|i: usize| sample_table::MAX_SAMPLES.saturating_sub(parsed[i].2.samples.len());
		let cap = (0..parsed.len()).map(max_samples).min().unwrap_or(0);
		while let Ok(Some((position, kind, header, size))) = next_box(reader, file_size) {
			if &kind == b"moof" {
				if size - header > MAX_MOOF_SIZE || reader.skip(header).is_err() {
					break;
				}
				let Ok(body) = reader.read_vec((size - header) as usize) else { break };
				if fragment::parse_moof(&body, position, file_size, cap, &mut tracks).is_err() {
					break;
				}
			}
			let Some(next) = position.checked_add(size) else { break };
			if file_size.is_some_and(|total| next >= total) || reader.seek(next).is_err() {
				break;
			}
		}
		for ((_, _, track), fragmented) in parsed.iter_mut().zip(tracks) {
			track.samples.extend(fragmented.samples);
			track.has_sync_table |= track.samples.iter().any(|s| !s.keyframe);
		}
	}

	/// Track whose next sample has the smallest decode time (ties broken by
	/// file offset, which minimises seeking).
	fn next_track(&self) -> Option<usize> {
		let mut best: Option<(usize, &Sample, u32)> = None;
		for (index, track) in self.tracks.iter().enumerate() {
			let Some(sample) = track.samples.get(track.next) else { continue };
			let better = match best {
				None => true,
				Some((_, other, other_scale)) => {
					// Compare dts_a / scale_a with dts_b / scale_b exactly.
					let a = u128::from(sample.dts) * u128::from(other_scale);
					let b = u128::from(other.dts) * u128::from(track.timescale);
					a < b || (a == b && sample.offset < other.offset)
				}
			};
			if better {
				best = Some((index, sample, track.timescale));
			}
		}
		best.map(|(index, _, _)| index)
	}
}

fn shifted(ticks: u64, shift: i64) -> u64 {
	if shift >= 0 {
		ticks.saturating_add(shift as u64)
	} else {
		ticks.saturating_sub(shift.unsigned_abs())
	}
}

impl Demuxer for Mp4Demuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		let Some(index) = self.next_track() else { return Ok(None) };
		let track = &mut self.tracks[index];
		let sample = track.samples[track.next];
		track.next += 1;
		let time_base = TimeBase::new(1, track.timescale);

		let dts = if track.pts_shift > 0 { shifted(sample.dts, track.pts_shift) } else { sample.dts };
		let pts = shifted(sample.dts, i64::from(sample.cts).saturating_add(track.pts_shift));

		if self.reader.position() != sample.offset {
			self.reader.seek(sample.offset)?;
		}
		let data = self.reader.read_vec(sample.size as usize)?;

		Ok(Some(Packet {
			stream_id: StreamId(index),
			data,
			pts: Some(Time::new(pts, time_base)),
			dts: Some(Time::new(dts, time_base)),
			duration: Some(Time::new(u64::from(sample.duration), time_base)),
			keyframe: sample.keyframe,
		}))
	}

	/// Seeks every track so that playback can start at `time`: tracks with
	/// sync-sample tables (video) snap back to the preceding keyframe, and
	/// all other tracks are aligned to that keyframe's time, as FFmpeg does.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let locate = |track: &Track, seconds: f64| -> usize {
			let target = (seconds * f64::from(track.timescale)) as u64;
			track.samples.partition_point(|s| s.dts <= target).saturating_sub(1)
		};

		let mut start = time;
		for track in self.tracks.iter().filter(|t| t.has_sync_table) {
			let mut i = locate(track, time);
			while i > 0 && !track.samples[i].keyframe {
				i -= 1;
			}
			let seconds = track.samples[i].dts as f64 / f64::from(track.timescale);
			start = start.min(seconds);
		}
		for track in self.tracks.iter_mut() {
			let mut i = locate(track, start);
			if track.has_sync_table {
				while i > 0 && !track.samples[i].keyframe {
					i -= 1;
				}
			}
			track.next = i;
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
