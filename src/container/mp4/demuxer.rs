//! MP4 / M4A / MOV demuxer (non-fragmented ISO BMFF).
//!
//! Opening a file walks the top-level boxes until `moov`, loads it into
//! memory, and builds a flat sample index per track
//! ([`super::sample_table`]). Reading then returns samples from all tracks
//! interleaved by decode time, seeking to each sample's offset in `mdat`.
//! This follows FFmpeg's `mov.c` (`mov_read_header` + `mov_read_packet`).

use super::boxes::{BoxIter, find, find_path, full_box};
use super::codec::{self, CodecInfo};
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

		let mut streams = StreamSet::default();
		let mut tracks = Vec::new();
		for (kind, trak) in BoxIter::new(&moov).map_while(|r| r.ok()) {
			if &kind != b"trak" {
				continue;
			}
			// A malformed track is skipped rather than failing the file,
			// so the remaining tracks stay playable.
			let Ok(Some((stream, track))) =
				Self::parse_track(trak, tracks.len(), movie_timescale, file_size)
			else {
				continue;
			};
			streams.add(stream);
			tracks.push(track);
		}
		if tracks.is_empty() {
			return Err(if find(&moov, b"mvex").is_some() {
				Message::Container("fragmented mp4 is not supported")
			} else {
				Message::Container("mp4 file has no playable tracks")
			});
		}

		let mut metadata = Metadata::default();
		metadata::parse_moov_metadata(&moov, &mut metadata);

		let duration = if movie_timescale > 0 {
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

	fn parse_track(
		trak: &[u8],
		index: usize,
		movie_timescale: u32,
		file_size: Option<u64>,
	) -> Result<Option<(Stream, Track)>> {
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
		let samples = sample_table::build_index(stbl, file_size)?;
		if samples.is_empty() {
			return Ok(None);
		}

		let edit = parse_edit_list(trak, movie_timescale, timescale);
		let mut stream = build_stream(index, kind, timescale, info);
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
		Ok(Some((stream, Track { timescale, samples, next: 0, pts_shift, has_sync_table })))
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
