//! Demuxing a presentation delivered as segments (HLS, DASH).
//!
//! Each *track* is one rendition: an ordered list of media segments, each
//! with an optional initialization section and key. A segment is fetched
//! whole, decrypted, prefixed with its initialization section, and demuxed
//! on its own by the demuxer its bytes call for (MPEG-TS, fragmented MP4,
//! WebM, packed ADTS/AC-3/MP3), so memory stays at about one segment per
//! track and seeking only needs the segment list. Tracks are interleaved
//! by decode time. Live tracks re-read their manifest when they run out of
//! segments.
//!
//! FFmpeg demuxes HLS the same way in spirit (`hls.c`: a nested demuxer
//! per playlist reading through a custom AVIOContext); DASH in `dashdec.c`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::container::hls::sample_aes::{self, SampleCipher, SampleCodec};
use crate::container::mpegts::{TsDemuxer, packet::detect_layout};
use crate::core::{
	Demuxer, Metadata, Packet, Stream, StreamId, StreamKind, StreamSet, Time, TimeBase,
};
use crate::io::{ByteRange, Fetcher, MemoryIo};
use crate::message::{Message, Result};

/// Encryption of a whole segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentKey {
	None,
	/// AES-128 CBC with PKCS#7 padding; the key is fetched from `uri`.
	Aes128 {
		uri: String,
		iv: [u8; 16],
	},
	/// SAMPLE-AES: the segment demuxes in the clear; each packet's samples
	/// are decrypted ([`crate::container::hls::sample_aes`]).
	SampleAes {
		uri: String,
		iv: [u8; 16],
	},
}

/// One media segment of a track.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentRef {
	pub uri: String,
	pub range: Option<ByteRange>,
	/// Initialization section prepended to the segment's bytes.
	pub init: Option<(String, Option<ByteRange>)>,
	/// Nominal duration in seconds (from the manifest).
	pub duration: f64,
	pub key: SegmentKey,
	/// Increases with each segment: how a live refresh finds new ones.
	pub sequence: u64,
	/// Seconds added to the segment's timestamps (DASH period boundaries).
	pub time_offset: f64,
}

/// Re-reads a live manifest: the track's segments now listed, and whether
/// the presentation has ended.
pub type Refresh = Box<dyn FnMut(&mut Fetcher) -> Result<(Vec<SegmentRef>, bool)>>;

/// A rendition to demux, as a manifest parser describes it.
pub struct TrackSpec {
	pub segments: Vec<SegmentRef>,
	/// First segment to play (live streams start near the live edge).
	pub start: usize,
	pub ended: bool,
	pub refresh: Option<Refresh>,
	/// Time between live manifest reloads.
	pub poll: Duration,
	/// Metadata added to each of the track's streams (`variant_bitrate`,
	/// `language`, ...), without replacing what the media says.
	pub metadata: Vec<(String, String)>,
}

/// How to pick renditions from a manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingOptions {
	/// A variant (HLS) or video representation (DASH) by manifest order,
	/// overriding the bandwidth choice.
	pub variant: Option<usize>,
	/// Highest bandwidth to choose (bits/s). The default is the highest
	/// available, as players pick on a fast link.
	pub max_bandwidth: Option<u64>,
	/// Tallest picture to choose.
	pub max_height: Option<u32>,
	/// Preferred audio language (`en`, `fra`, ...): matched as a prefix,
	/// ignoring case.
	pub audio_language: Option<String>,
	/// Live streams start this many segments before the live edge
	/// (FFmpeg's `live_start_index` of -3).
	pub live_start: usize,
}

impl Default for StreamingOptions {
	fn default() -> Self {
		Self {
			variant: None,
			max_bandwidth: None,
			max_height: None,
			audio_language: None,
			live_start: 3,
		}
	}
}

impl StreamingOptions {
	/// Picks among `(bandwidth, height)` candidates: the explicit index, else
	/// the highest bandwidth within the limits, else the lowest overall.
	pub fn choose(&self, candidates: &[(u64, Option<u32>)]) -> Option<usize> {
		if let Some(i) = self.variant {
			return (i < candidates.len()).then_some(i);
		}
		let fits = |&(bw, h): &(u64, Option<u32>)| {
			self.max_bandwidth.is_none_or(|m| bw <= m)
				&& self.max_height.is_none_or(|m| h.is_none_or(|h| h <= m))
		};
		let best = candidates.iter().enumerate().filter(|(_, c)| fits(c)).max_by_key(|(_, c)| c.0);
		best.or_else(|| candidates.iter().enumerate().min_by_key(|(_, c)| c.0)).map(|(i, _)| i)
	}

	pub fn language_matches(&self, language: Option<&str>) -> bool {
		match (&self.audio_language, language) {
			(Some(want), Some(have)) => {
				let (w, h) = (want.to_ascii_lowercase(), have.to_ascii_lowercase());
				h.starts_with(&w) || w.starts_with(&h)
			}
			_ => false,
		}
	}
}

/// Live polls without a new segment before a stream is taken as ended.
const MAX_IDLE_POLLS: u32 = 30;

struct Track {
	spec: TrackSpec,
	next: usize,
	demuxer: Option<Box<dyn Demuxer>>,
	/// Global stream ids of this track, with their kinds.
	streams: Vec<(StreamId, StreamKind)>,
	/// Inner stream index → global stream, for the open segment.
	map: Vec<Option<StreamId>>,
	/// Seconds added to the open segment's timestamps (packed-audio ID3
	/// timestamps, DASH period offsets).
	offset: f64,
	peeked: Option<Packet>,
	done: bool,
	/// Index and first decode time (s) of the first segment opened: the
	/// anchor that maps segment start times to packet times.
	origin: Option<(usize, f64)>,
	last_refresh: Instant,
	idle_polls: u32,
	/// SAMPLE-AES key of the open segment, and how each of its streams is
	/// encrypted (`None`: in the clear).
	cipher: Option<SampleCipher>,
	encrypted: Vec<Option<SampleCodec>>,
}

/// What [`SegmentedDemuxer::load`] opened.
struct Loaded {
	demuxer: Box<dyn Demuxer>,
	/// Seconds to add to the segment's timestamps.
	offset: f64,
	cipher: Option<SampleCipher>,
}

impl Track {
	fn attach(&mut self, loaded: Loaded) {
		self.map = map_streams(&self.streams, loaded.demuxer.streams());
		self.encrypted = match loaded.cipher {
			Some(_) => {
				loaded.demuxer.streams().iter().map(|s| SampleCodec::of(&s.parameters.codec.id)).collect()
			}
			None => Vec::new(),
		};
		self.cipher = loaded.cipher;
		self.offset = loaded.offset;
		self.demuxer = Some(loaded.demuxer);
	}
}

pub struct SegmentedDemuxer {
	tracks: Vec<Track>,
	streams: StreamSet,
	metadata: Metadata,
	duration: Time,
	fetcher: Fetcher,
	keys: HashMap<String, [u8; 16]>,
	/// Initialization sections by location; a few at most per track.
	inits: HashMap<(String, Option<ByteRange>), Vec<u8>>,
}

/// Opens the demuxer for one segment's bytes. MPEG-TS segments skip the
/// end-of-file duration scan, which a segment never needs.
pub fn open_segment_demuxer(data: Vec<u8>) -> Result<Box<dyn Demuxer>> {
	let head = &data[..data.len().min(4096)];
	if detect_layout(head).is_some() {
		return Ok(Box::new(TsDemuxer::new_segment(Box::new(MemoryIo::from_vec(data)))?));
	}
	if crate::container::webvtt::looks_like_webvtt(head) {
		return Ok(Box::new(crate::container::webvtt::WebVttDemuxer::new(Box::new(
			MemoryIo::from_vec(data),
		))?));
	}
	let head = head.to_vec();
	crate::container::detect(&head, Box::new(MemoryIo::from_vec(data)))
}

/// `com.apple.streaming.transportStreamTimestamp` from the ID3 tag that
/// starts a packed-audio segment (RFC 8216 §3.4), in 90 kHz ticks.
fn packed_audio_timestamp(data: &[u8]) -> Option<u64> {
	if data.len() < 10 || &data[..3] != b"ID3" {
		return None;
	}
	let syncsafe = |b: &[u8]| b.iter().fold(0usize, |acc, &x| (acc << 7) | usize::from(x & 0x7F));
	let version = data[3];
	let size = syncsafe(&data[6..10]);
	let tag = data.get(10..10 + size)?;
	let mut at = 0;
	while at + 10 <= tag.len() {
		let id = &tag[at..at + 4];
		let len = if version >= 4 {
			syncsafe(&tag[at + 4..at + 8])
		} else {
			u32::from_be_bytes([tag[at + 4], tag[at + 5], tag[at + 6], tag[at + 7]]) as usize
		};
		let body = tag.get(at + 10..at + 10 + len)?;
		const OWNER: &[u8] = b"com.apple.streaming.transportStreamTimestamp\0";
		if id == b"PRIV" && body.len() >= OWNER.len() + 8 && body.starts_with(OWNER) {
			let ts = &body[OWNER.len()..OWNER.len() + 8];
			return Some(u64::from_be_bytes(ts.try_into().ok()?) & ((1 << 33) - 1));
		}
		if id[0] == 0 {
			break;
		}
		at += 10 + len;
	}
	None
}

/// Moves a timestamp by `seconds` (clamped at zero: RMP times are
/// unsigned).
fn shift(t: Option<Time>, seconds: f64) -> Option<Time> {
	let t = t?;
	let base = t.base();
	let delta = (seconds * f64::from(base.den) / f64::from(base.num.max(1))).round();
	let ticks = if delta >= 0.0 {
		t.ticks().saturating_add(delta as u64)
	} else {
		t.ticks().saturating_sub((-delta) as u64)
	};
	Some(Time::new(ticks, base))
}

fn seconds(p: &Packet) -> f64 {
	p.dts.or(p.pts).map_or(f64::MAX, |t| t.as_seconds())
}

impl SegmentedDemuxer {
	/// Opens every track's first segment to learn its streams.
	pub fn open(specs: Vec<TrackSpec>, metadata: Metadata, fetcher: Fetcher) -> Result<Self> {
		let mut demuxer = Self {
			tracks: Vec::new(),
			streams: StreamSet::default(),
			metadata,
			duration: Time::zero(TimeBase::new(1, 1000)),
			fetcher,
			keys: HashMap::new(),
			inits: HashMap::new(),
		};
		let mut total = 0.0f64;
		for spec in specs {
			let start = spec.start.min(spec.segments.len());
			if spec.ended {
				total = total.max(spec.segments.iter().map(|s| s.duration).sum());
			}
			let mut track = Track {
				spec,
				next: start,
				demuxer: None,
				streams: Vec::new(),
				map: Vec::new(),
				offset: 0.0,
				peeked: None,
				done: false,
				origin: None,
				last_refresh: Instant::now(),
				idle_polls: 0,
				cipher: None,
				encrypted: Vec::new(),
			};
			// The first segment that opens defines the track's streams.
			let inner = loop {
				if track.next >= track.spec.segments.len() {
					return Err(Message::Container("streaming: no playable segment in a track"));
				}
				let index = track.next;
				track.next += 1;
				match demuxer.load(&track.spec.segments[index]) {
					Ok(opened) => break (index, opened),
					Err(e) if index == start => return Err(e),
					Err(_) => continue,
				}
			};
			let (index, loaded) = inner;
			for s in loaded.demuxer.streams().iter() {
				let mut stream: Stream = s.clone();
				stream.id = StreamId(demuxer.streams.streams.len());
				for (k, v) in &track.spec.metadata {
					if k == "title" {
						if stream.metadata.title.is_none() && !v.is_empty() {
							stream.metadata.title = Some(v.clone());
						}
						continue;
					}
					stream.metadata.raw.entry(k.clone()).or_insert(crate::core::RawValue::String(v.clone()));
				}
				if track.spec.ended {
					let secs: f64 = track.spec.segments.iter().map(|s| s.duration).sum();
					stream.duration = Some(Time::from_seconds(secs, stream.time_base));
				} else {
					stream.duration = None;
				}
				track.streams.push((stream.id, stream.kind));
				demuxer.streams.add(stream);
			}
			track.attach(loaded);
			track.origin = Some((index, f64::NAN));
			demuxer.tracks.push(track);
		}
		if demuxer.streams.streams.is_empty() {
			return Err(Message::Container("streaming: no streams found"));
		}
		demuxer.duration = Time::from_seconds(total, TimeBase::new(1, 1000));
		Ok(demuxer)
	}

	/// A 16-byte key, fetched once per location.
	fn key(&mut self, uri: &str) -> Result<[u8; 16]> {
		if let Some(k) = self.keys.get(uri) {
			return Ok(*k);
		}
		let bytes = self.fetcher.fetch(uri, None)?;
		let key: [u8; 16] =
			bytes.try_into().map_err(|_| Message::Container("hls: AES-128 key is not 16 bytes"))?;
		self.keys.insert(uri.to_string(), key);
		Ok(key)
	}

	/// Fetches, decrypts and opens one segment.
	fn load(&mut self, segment: &SegmentRef) -> Result<Loaded> {
		let mut data = self.fetcher.fetch(&segment.uri, segment.range)?;
		let mut cipher = None;
		match &segment.key {
			SegmentKey::None => {}
			SegmentKey::Aes128 { uri, iv } => {
				let key = self.key(uri)?;
				data = crate::container::hls::aes::decrypt_cbc(&key, iv, &data)
					.ok_or(Message::Container("hls: segment decryption failed"))?;
			}
			SegmentKey::SampleAes { uri, iv } => {
				// In fragmented MP4, SAMPLE-AES is CENC `cbcs`, whose sample
				// encryption tables RMP's MP4 demuxer doesn't read.
				if segment.init.is_some() {
					return Err(Message::Container(
						"hls: SAMPLE-AES in fragmented MP4 (cbcs) is not supported, only in MPEG-TS and packed audio",
					));
				}
				cipher = Some(SampleCipher::new(&self.key(uri)?, iv));
			}
		}
		if let Some((uri, range)) = &segment.init {
			let id = (uri.clone(), *range);
			if !self.inits.contains_key(&id) {
				let init = self.fetcher.fetch(uri, *range)?;
				// Keyed by location; a long DASH presentation has one per
				// period at most, so the cache stays small.
				self.inits.insert(id.clone(), init);
			}
			let init = self.inits.get(&id).map(Vec::as_slice).unwrap_or(&[]);
			let mut joined = Vec::with_capacity(init.len() + data.len());
			joined.extend_from_slice(init);
			joined.extend_from_slice(&data);
			data = joined;
		}
		let id3 = packed_audio_timestamp(&data).map_or(0.0, |t| t as f64 / 90_000.0);
		let mut offset = id3 + segment.time_offset;
		// WebVTT cues are mapped onto the media clock by their own header.
		if let Some(map) = crate::container::webvtt::timestamp_map(&data) {
			offset += map;
		}
		Ok(Loaded { demuxer: open_segment_demuxer(data)?, offset, cipher })
	}

	/// Next packet of track `t`, opening segments and refreshing the
	/// manifest as needed. `None` when the track has ended.
	fn track_packet(&mut self, t: usize) -> Result<Option<Packet>> {
		loop {
			let track = &mut self.tracks[t];
			if let Some(inner) = track.demuxer.as_mut() {
				match inner.read() {
					Ok(Some(mut p)) => {
						let inner_index = p.stream_id.0;
						let Some(Some(id)) = track.map.get(inner_index).copied() else { continue };
						p.stream_id = id;
						if let (Some(cipher), Some(Some(codec))) =
							(track.cipher.as_ref(), track.encrypted.get(inner_index))
						{
							p.data = sample_aes::decrypt_packet(cipher, *codec, &p.data);
						}
						if track.offset != 0.0 {
							p.pts = shift(p.pts, track.offset);
							p.dts = shift(p.dts, track.offset);
						}
						if let Some((i, start)) = track.origin
							&& start.is_nan()
						{
							track.origin = Some((i, seconds(&p)));
						}
						return Ok(Some(p));
					}
					// A damaged segment ends early; the next one continues.
					Ok(None) | Err(_) => track.demuxer = None,
				}
				continue;
			}
			if track.next < track.spec.segments.len() {
				let segment = track.spec.segments[track.next].clone();
				track.next += 1;
				// A segment that can't be fetched or opened is skipped (after
				// one retry), as players and FFmpeg do.
				let opened = self.load(&segment).or_else(|_| self.load(&segment));
				if let Ok(loaded) = opened {
					self.tracks[t].attach(loaded);
				}
				continue;
			}
			if track.spec.ended || track.spec.refresh.is_none() || track.idle_polls >= MAX_IDLE_POLLS {
				return Ok(None);
			}
			self.refresh(t)?;
		}
	}

	/// Waits for the poll interval, re-reads the manifest and appends the
	/// segments that are new.
	fn refresh(&mut self, t: usize) -> Result<()> {
		let wait = self.tracks[t].spec.poll.saturating_sub(self.tracks[t].last_refresh.elapsed());
		std::thread::sleep(wait);
		let track = &mut self.tracks[t];
		let Some(refresh) = track.spec.refresh.as_mut() else { return Ok(()) };
		let result = refresh(&mut self.fetcher);
		track.last_refresh = Instant::now();
		let Ok((segments, ended)) = result else {
			track.idle_polls += 1;
			return Ok(());
		};
		let last = track.spec.segments.last().map(|s| s.sequence);
		let fresh: Vec<SegmentRef> =
			segments.into_iter().filter(|s| last.is_none_or(|l| s.sequence > l)).collect();
		track.idle_polls = if fresh.is_empty() { track.idle_polls + 1 } else { 0 };
		track.spec.segments.extend(fresh);
		track.spec.ended = ended;
		Ok(())
	}
}

/// Maps a segment's streams onto the track's: the n-th stream of a kind
/// is the track's n-th stream of that kind.
fn map_streams(track: &[(StreamId, StreamKind)], inner: &StreamSet) -> Vec<Option<StreamId>> {
	let mut seen: Vec<StreamKind> = Vec::new();
	inner
		.iter()
		.map(|s| {
			let n = seen.iter().filter(|&&k| k == s.kind).count();
			seen.push(s.kind);
			track.iter().filter(|(_, k)| *k == s.kind).nth(n).map(|(id, _)| *id)
		})
		.collect()
}

impl Demuxer for SegmentedDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		for t in 0..self.tracks.len() {
			if self.tracks[t].peeked.is_none() && !self.tracks[t].done {
				match self.track_packet(t)? {
					Some(p) => self.tracks[t].peeked = Some(p),
					None => self.tracks[t].done = true,
				}
			}
		}
		let next = self
			.tracks
			.iter()
			.enumerate()
			.filter_map(|(i, t)| t.peeked.as_ref().map(|p| (i, seconds(p))))
			.min_by(|a, b| a.1.total_cmp(&b.1))
			.map(|(i, _)| i);
		Ok(next.and_then(|i| self.tracks[i].peeked.take()))
	}

	/// `time` is on the packet clock, like the other demuxers. Every track
	/// restarts at the segment containing it; segments start with a
	/// keyframe, so playback can begin there.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		for track in &mut self.tracks {
			let Some((anchor, start)) = track.origin else { continue };
			let start = if start.is_nan() { 0.0 } else { start };
			let segs = &track.spec.segments;
			// Segment start times on the packet clock.
			let mut at = start - segs.iter().take(anchor).map(|s| s.duration).sum::<f64>();
			let mut target = 0;
			for (i, s) in segs.iter().enumerate() {
				if at <= time + 1e-6 {
					target = i;
				}
				at += s.duration;
			}
			track.next = target;
			track.demuxer = None;
			track.peeked = None;
			track.done = false;
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
	fn id3_transport_timestamp() {
		let owner = b"com.apple.streaming.transportStreamTimestamp\0";
		let mut frame = b"PRIV".to_vec();
		let body_len = owner.len() + 8;
		frame.extend_from_slice(&[0, 0, 0, body_len as u8, 0, 0]);
		frame.extend_from_slice(owner);
		frame.extend_from_slice(&126_000u64.to_be_bytes());
		let mut tag = b"ID3\x04\x00\x00".to_vec();
		tag.extend_from_slice(&[0, 0, 0, frame.len() as u8]);
		tag.extend_from_slice(&frame);
		assert_eq!(packed_audio_timestamp(&tag), Some(126_000));
		assert_eq!(packed_audio_timestamp(b"\xFF\xF1"), None);
	}

	#[test]
	fn streams_map_by_kind() {
		let track = [(StreamId(3), StreamKind::Video), (StreamId(4), StreamKind::Audio)];
		let mut inner = StreamSet::default();
		inner.add(Stream::audio(StreamId(0), 48_000, 2, crate::core::CodecId::new("aac")));
		inner.add(Stream::audio(StreamId(1), 48_000, 2, crate::core::CodecId::new("aac")));
		assert_eq!(map_streams(&track, &inner), vec![Some(StreamId(4)), None]);
	}
}
