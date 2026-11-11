//! MP4 muxer (ISO BMFF), after FFmpeg's `movenc.c`. Three layouts:
//!
//! ```text
//! default       ftyp | mdat (64-bit size, patched in finish) | moov
//! +faststart    ftyp | moov | mdat           (moov moved to the front in finish)
//! fragmented    ftyp | moov (empty) | moof mdat | moof mdat | ... | mfra
//! ```
//!
//! **Progressive** (default and `+faststart`): packets are appended to
//! `mdat` as they arrive; consecutive packets of the same track form a
//! chunk. Per-sample bookkeeping (size, dts, composition offset, keyframe
//! flag) is kept in memory and serialised into the sample tables when
//! [`Muxer::finish`] writes `moov`. With `+faststart`, `finish` then slides
//! `mdat` forward by the size of `moov` (in place, back to front, through a
//! fixed 1 MiB buffer) and writes `moov` in the gap, with every chunk offset
//! raised by that size — FFmpeg's `shift_data`. Players can then start
//! playback after downloading the first few kilobytes.
//!
//! **Fragmented** (`frag_keyframe`, `frag_duration`, ...; see
//! [`super::options`]): samples are buffered per fragment and written as
//! `moof` + `mdat` when the next fragment starts ([`super::fragment`]). The
//! `moov` holds no samples and is written together with the first fragment
//! (FFmpeg's `delay_moov`), which lets it carry the edit lists for B-frame
//! delay and AAC priming. Nothing is ever seeked back to, so fragmented
//! output also works on pipes and sockets; `mehd` (total duration) is filled
//! in at the end only when the output can seek. Memory use is bounded by one
//! fragment.
//!
//! Video packets that carry a `pts` but no `dts` (Matroska stores only
//! presentation times) are accepted: decode times are derived from the
//! sorted presentation times — once the track is complete in progressive
//! files (shifted so no sample is presented before it is decoded, with the
//! shift compensated by the edit list), per fragment in fragmented files
//! (with negative composition offsets in a version 1 `trun`).

use super::boxes::{Put, write_box, write_full_box};
use super::codec::{self, EntryParams};
use super::demuxer::{ENCODER_DELAY_KEY, ENCODER_PADDING_KEY};
use super::fragment::{self, FragmentSample, RandomAccess, TrackFragment};
use super::metadata;
use super::options::Mp4Options;
use crate::codecs::mp3::FrameHeader;
use crate::core::{
	Metadata, Muxer, Packet, RawValue, Stream, StreamId, StreamKind, Time, TimeBase, rescale_ticks,
};
use crate::io::Io;
use crate::message::{Message, Result};

/// Timescale of `mvhd`/`tkhd`/`elst` durations.
const MOVIE_TIMESCALE: u32 = 1000;
/// Size of the `mdat` header we write (32-bit size = 1, then 64-bit size).
const MDAT_HEADER_SIZE: u64 = 16;
const ENCODER_NAME: &str = "RMP";
/// Buffer for the `+faststart` move.
const SHIFT_BUFFER: usize = 1 << 20;
/// A fragment is cut when its buffered samples reach this size, whatever
/// the rules say, so memory stays bounded (a 100 Mbit/s stream reaches it
/// after about 5 s).
const MAX_FRAGMENT_BYTES: u64 = 64 * 1024 * 1024;
/// Fragment length when no rule would ever cut one (no video track and
/// no `-frag_duration`).
const DEFAULT_FRAGMENT_US: u64 = 1_000_000;

const IDENTITY_MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

struct MuxSample {
	size: u32,
	dts: u64,
	/// pts - dts, in track ticks.
	cts: i64,
	keyframe: bool,
}

struct Chunk {
	offset: u64,
	samples: u32,
}

struct MuxTrack {
	stream_id: StreamId,
	kind: StreamKind,
	codec: String,
	track_id: u32,
	timescale: u32,
	width: u32,
	height: u32,
	sample_rate: u32,
	channels: u8,
	extradata: Vec<u8>,
	/// Audio priming / trailing padding to signal through an edit list,
	/// in track ticks.
	encoder_delay: Option<u64>,
	encoder_padding: u64,
	samples: Vec<MuxSample>,
	chunks: Vec<Chunk>,
	/// Duration of the last packet, when the packet said so.
	last_duration: Option<u64>,
	/// dts to assign to a packet that carries no timestamps.
	next_dts: u64,
	bytes: u64,
	max_sample_size: u32,
	/// Decode times are derived from presentation times in `finish` (per
	/// fragment when fragmenting); until then `dts` holds the pts and `cts`
	/// is 0.
	derive_dts: bool,
	/// `Stream::parameters.bitrate`, for `esds`/`btrt` when the samples
	/// aren't all known yet (fragmented output).
	bitrate: Option<u32>,
	/// dts of the latest packet, across fragments.
	last_dts: Option<u64>,
	/// Fragmented: sample data of the current fragment.
	pending: Vec<u8>,
	/// Fragmented: decode time of the track's first sample (fragment
	/// decode times are relative to it, as sample tables are) and of the
	/// end of the last written fragment.
	origin: Option<u64>,
	end_dts: u64,
	/// Fragmented: end of the latest presented sample, relative to
	/// `origin` (for `mehd`).
	end_pts: u64,
	/// Fragmented: edit list media time written in the init segment, and
	/// the delay added to every composition offset so they stay
	/// non-negative (chosen from the first fragment, like the B-frame delay
	/// of a progressive file; FFmpeg's demuxer misplaces negative `trun`
	/// offsets by a frame).
	edit: Option<u64>,
	cts_shift: i64,
	/// Fragmented: fragments starting with a sync sample, for `tfra`.
	random_access: Vec<RandomAccess>,
}

impl MuxTrack {
	/// Replaces placeholder decode times with the sorted presentation times.
	/// With `negative_cts` false, every composition offset is kept
	/// non-negative by delaying presentation (compensated by the edit list).
	fn derive_decode_times(&mut self, negative_cts: bool) {
		if !self.derive_dts || self.samples.is_empty() {
			return;
		}
		let pts: Vec<u64> = self.samples.iter().map(|s| s.dts).collect();
		let mut sorted = pts.clone();
		sorted.sort_unstable();
		// Largest amount by which a sorted time runs ahead of the
		// presentation time at the same position.
		let shift = if negative_cts {
			0
		} else {
			sorted.iter().zip(&pts).map(|(d, p)| d.saturating_sub(*p)).max().unwrap_or(0)
		};
		for (sample, (&d, &p)) in self.samples.iter_mut().zip(sorted.iter().zip(&pts)) {
			sample.dts = d;
			sample.cts = (i128::from(p) + i128::from(shift) - i128::from(d)) as i64;
		}
	}

	fn duration_of(&self, index: usize) -> u64 {
		match self.samples.get(index + 1) {
			Some(next) => next.dts.saturating_sub(self.samples[index].dts),
			None => self.last_duration.unwrap_or_else(|| {
				// Repeat the previous delta for the final sample.
				index
					.checked_sub(1)
					.map_or(0, |prev| self.samples[index].dts.saturating_sub(self.samples[prev].dts))
			}),
		}
	}

	/// Total media duration in track ticks.
	fn media_duration(&self) -> u64 {
		match (self.samples.first(), self.samples.len().checked_sub(1)) {
			(Some(first), Some(last)) => {
				(self.samples[last].dts + self.duration_of(last)).saturating_sub(first.dts)
			}
			_ => 0,
		}
	}

	/// Media time where presentation starts (edit list), if any.
	fn edit_media_time(&self) -> Option<u64> {
		match self.kind {
			StreamKind::Audio => self.encoder_delay.filter(|&d| d > 0 && d < self.media_duration()),
			_ => {
				// Sample times in the file start at the first decode time, so
				// the edit is relative to it.
				let first_dts = self.samples.first().map_or(0, |s| i128::from(s.dts));
				let first_pts =
					self.samples.iter().map(|s| i128::from(s.dts) + i128::from(s.cts)).min().unwrap_or(0);
				u64::try_from(first_pts - first_dts).ok().filter(|&p| p > 0)
			}
		}
	}

	/// Presented duration in track ticks (media duration minus edits).
	fn presentation_duration(&self) -> u64 {
		let media = self.media_duration();
		match self.edit_media_time() {
			Some(start) if self.kind == StreamKind::Audio => {
				media.saturating_sub(start).saturating_sub(self.encoder_padding)
			}
			Some(_) => {
				// Video: from the first to the end of the last presented
				// picture (with B-frames that ends after the last decode time).
				let pts = |i: usize| i128::from(self.samples[i].dts) + i128::from(self.samples[i].cts);
				let first = (0..self.samples.len()).map(pts).min().unwrap_or(0);
				let end = (0..self.samples.len())
					.map(|i| pts(i) + i128::from(self.duration_of(i)))
					.max()
					.unwrap_or(0);
				u64::try_from(end - first).unwrap_or(media)
			}
			None => media,
		}
	}

	/// Edit list media time for a fragmented file, from the first
	/// fragment: audio priming, or the B-frame delay of the first picture.
	fn fragment_edit_media_time(&self) -> Option<u64> {
		match self.kind {
			StreamKind::Audio => self.encoder_delay.filter(|&d| d > 0),
			_ => {
				let first_dts = self.samples.first().map_or(0, |s| i128::from(s.dts));
				let first_pts = self
					.samples
					.iter()
					.map(|s| i128::from(s.dts) + i128::from(s.cts) + i128::from(self.cts_shift))
					.min()
					.unwrap_or(0);
				u64::try_from(first_pts - first_dts).ok().filter(|&p| p > 0)
			}
		}
	}

	/// Smallest delay that makes every composition offset of the current
	/// samples non-negative.
	fn needed_cts_shift(&self) -> i64 {
		self.samples.iter().map(|s| -s.cts).max().unwrap_or(0).max(0)
	}

	/// Average bit rate: measured over the whole track when it is complete,
	/// otherwise (fragmented init segment) the stream's declared rate, or
	/// the first fragment's.
	fn average_bitrate(&self, complete: bool) -> u32 {
		if !complete && let Some(declared) = self.bitrate.filter(|&b| b > 0) {
			return declared;
		}
		let seconds = self.media_duration() as f64 / f64::from(self.timescale);
		if seconds > 0.0 { (self.bytes as f64 * 8.0 / seconds) as u32 } else { 0 }
	}

	/// The current fragment's samples, as `trun` entries.
	fn fragment_samples(&self) -> Vec<FragmentSample> {
		let clamp_i32 = |v: i64| v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
		self
			.samples
			.iter()
			.enumerate()
			.map(|(i, s)| FragmentSample {
				duration: u32::try_from(self.duration_of(i)).unwrap_or(u32::MAX),
				size: s.size,
				cts: clamp_i32(s.cts.saturating_add(self.cts_shift)),
				keyframe: s.keyframe,
			})
			.collect()
	}

	fn to_movie(&self, ticks: u64) -> u64 {
		rescale_ticks(ticks, TimeBase::new(1, self.timescale), TimeBase::new(1, MOVIE_TIMESCALE))
			.unwrap_or(0)
	}

	/// Fallback packet duration for packets without any timing.
	fn implicit_duration(&self, data: &[u8]) -> Option<u64> {
		let samples = match self.codec.as_str() {
			"aac" => 1024,
			"mp3" => FrameHeader::parse(data)?.samples,
			_ => return None,
		};
		rescale_ticks(
			u64::from(samples),
			TimeBase::new(1, self.sample_rate.max(1)),
			TimeBase::new(1, self.timescale),
		)
	}
}

/// How `moov` describes the media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
	/// Sample tables for everything written; chunk offsets raised by
	/// `offset_shift` (the size of a `moov` placed in front of `mdat`).
	Progressive { offset_shift: u64 },
	/// Empty sample tables + `mvex`; samples live in fragments.
	Fragmented,
}

pub struct Mp4Muxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	options: Mp4Options,
	tracks: Vec<MuxTrack>,
	started: bool,
	/// The output accepted a seek before anything was written.
	seekable: bool,
	mdat_start: u64,
	position: u64,
	last_track: Option<usize>,
	/// Fragmented: init segment written, `moof` sequence number, file
	/// position of the `mehd` duration, and the current fragment's first
	/// sample (dts, timescale), sample count and bytes.
	moov_written: bool,
	sequence: u32,
	mehd_at: Option<u64>,
	fragment_start: Option<(u64, u32)>,
	pending_samples: usize,
	pending_bytes: u64,
}

impl Mp4Muxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self::with_options(io, Mp4Options::default())
	}

	/// A muxer with `-movflags` / fragment options already set.
	pub fn with_options(io: Box<dyn Io>, options: Mp4Options) -> Self {
		Self {
			io,
			metadata: Metadata::default(),
			options,
			tracks: Vec::new(),
			started: false,
			seekable: false,
			mdat_start: 0,
			position: 0,
			last_track: None,
			moov_written: false,
			sequence: 0,
			mehd_at: None,
			fragment_start: None,
			pending_samples: 0,
			pending_bytes: 0,
		}
	}

	pub fn options(&self) -> &Mp4Options {
		&self.options
	}

	fn fragmented(&self) -> bool {
		self.options.fragmented()
	}

	fn start(&mut self) -> Result<()> {
		// Probing with a seek to where we already are changes nothing.
		self.seekable = self.io.seek(0).is_ok();
		let mut header = Vec::with_capacity(64);
		let fragmented = self.fragmented();
		write_box(&mut header, b"ftyp", |out| {
			if fragmented {
				// Brands FFmpeg writes for default-base-is-moof fragments.
				out.put(b"iso5");
				out.put_u32(0x200);
				out.put(b"iso5");
				out.put(b"iso6");
				out.put(b"mp41");
				return Ok(());
			}
			out.put(b"isom");
			out.put_u32(0x200);
			out.put(b"isom");
			out.put(b"iso2");
			if self.tracks.iter().any(|t| t.codec == "h264") {
				out.put(b"avc1");
			}
			out.put(b"mp41");
			Ok(())
		})?;
		if !fragmented {
			self.mdat_start = header.len() as u64;
			header.put_u32(1); // 64-bit size follows
			header.put(b"mdat");
			header.put_u64(0); // patched in finish
		}
		self.io.write_all(&header)?;
		self.position = header.len() as u64;
		self.started = true;
		Ok(())
	}

	fn timescale_for(stream: &Stream) -> Result<u32> {
		if stream.kind == StreamKind::Audio {
			let rate = stream.parameters.sample_rate.unwrap_or(0);
			if rate > 0 {
				return Ok(rate);
			}
		}
		let den = stream.time_base.den;
		if den == 0 || stream.time_base.num == 0 {
			return Err(Message::Container("mp4 muxer: invalid stream time base"));
		}
		// Like FFmpeg, keep enough precision for video timestamps.
		let mut timescale = den;
		while timescale < 10_000 {
			timescale = timescale.saturating_mul(2);
		}
		Ok(timescale)
	}

	fn to_track_ticks(time: Time, timescale: u32) -> Result<u64> {
		time
			.rescale(TimeBase::new(1, timescale))
			.map(|t| t.ticks())
			.ok_or(Message::Container("mp4 muxer: invalid packet timestamp"))
	}

	/// Serialises `moov`. Returns it and, for [`Layout::Fragmented`], the
	/// position of the `mehd` duration within it.
	fn build_moov(&self, layout: Layout) -> Result<(Vec<u8>, Option<usize>)> {
		let mut moov = Vec::new();
		let movie_duration = match layout {
			Layout::Progressive { .. } => {
				self.tracks.iter().map(|t| t.to_movie(t.presentation_duration())).max().unwrap_or(0)
			}
			Layout::Fragmented => 0,
		};
		let mut mehd_at = None;
		write_box(&mut moov, b"moov", |out| {
			write_mvhd(out, movie_duration, self.tracks.len() as u32 + 1)?;
			for track in &self.tracks {
				write_trak(out, track, layout)?;
			}
			if layout == Layout::Fragmented {
				let ids: Vec<u32> = self.tracks.iter().map(|t| t.track_id).collect();
				mehd_at = Some(fragment::write_mvex(out, &ids)?);
			}
			metadata::write_udta(out, &self.metadata, ENCODER_NAME)
		})?;
		Ok((moov, mehd_at))
	}

	/// `moov` for a `+faststart` file: its chunk offsets must account for
	/// its own size, which can grow when offsets pass 4 GiB (`stco` →
	/// `co64`), so iterate to a fixed point like FFmpeg's
	/// `compute_moov_size`.
	fn build_faststart_moov(&self) -> Result<Vec<u8>> {
		let mut size = self.build_moov(Layout::Progressive { offset_shift: 0 })?.0.len();
		for _ in 0..4 {
			let (moov, _) = self.build_moov(Layout::Progressive { offset_shift: size as u64 })?;
			if moov.len() == size {
				return Ok(moov);
			}
			size = moov.len();
		}
		Err(Message::Container("mp4 muxer: faststart moov size does not converge"))
	}

	/// Moves `mdat` behind a `moov` of `shift` bytes. Bytes are copied back
	/// to front, so each block is read before anything overwrites it.
	fn shift_mdat(&mut self, shift: u64) -> Result<()> {
		let mut buffer = vec![0u8; SHIFT_BUFFER];
		let mut at = self.position;
		while at > self.mdat_start {
			let n = (at - self.mdat_start).min(SHIFT_BUFFER as u64) as usize;
			at -= n as u64;
			self.io.seek(at)?;
			read_exact(self.io.as_mut(), &mut buffer[..n])?;
			self.io.seek(at + shift)?;
			self.io.write_all(&buffer[..n])?;
		}
		Ok(())
	}

	/// Whether the output can be read back (needed by `+faststart`).
	fn readable(&mut self) -> bool {
		let mut byte = [0u8; 1];
		self.io.seek(self.mdat_start).is_ok() && matches!(self.io.read(&mut byte), Ok(1))
	}

	fn finish_progressive(&mut self) -> Result<()> {
		for track in &mut self.tracks {
			track.derive_decode_times(false);
		}
		let mdat_size = self.position - self.mdat_start;
		debug_assert!(mdat_size >= MDAT_HEADER_SIZE);
		self.io.seek(self.mdat_start + 8)?;
		self.io.write_all(&mdat_size.to_be_bytes())?;

		if self.options.flags.faststart {
			if self.readable() {
				let moov = self.build_faststart_moov()?;
				self.shift_mdat(moov.len() as u64)?;
				self.io.seek(self.mdat_start)?;
				self.io.write_all(&moov)?;
				self.position += moov.len() as u64;
				return Ok(());
			}
			// Leave a playable file behind, but say the flag wasn't honoured.
			let (moov, _) = self.build_moov(Layout::Progressive { offset_shift: 0 })?;
			self.io.seek(self.position)?;
			self.io.write_all(&moov)?;
			return Err(Message::Container(
				"+faststart needs an output that can be read back; moov was written at the end",
			));
		}
		let (moov, _) = self.build_moov(Layout::Progressive { offset_shift: 0 })?;
		self.io.seek(self.position)?;
		self.io.write_all(&moov)?;
		Ok(())
	}

	/// Whether a packet of track `index` at `dts` (track ticks) starts a
	/// new fragment.
	fn starts_fragment(&self, index: usize, dts: u64, keyframe: bool) -> bool {
		let Some((start, start_scale)) = self.fragment_start else { return false };
		if self.pending_samples == 0 {
			return false;
		}
		if self.pending_bytes >= MAX_FRAGMENT_BYTES {
			return true;
		}
		let track = &self.tracks[index];
		let elapsed = elapsed_us(start, start_scale, dts, track.timescale);
		if elapsed < self.options.min_frag_duration_us {
			return false;
		}
		// Decode times derived per fragment are only consistent when a
		// fragment holds whole GOPs: cut only at that track's keyframes.
		if self.tracks.iter().any(|t| t.derive_dts) && !(track.derive_dts && keyframe) {
			return false;
		}
		let flags = &self.options.flags;
		if flags.frag_every_frame {
			return true;
		}
		let has_video = self.tracks.iter().any(|t| t.kind == StreamKind::Video);
		let by_keyframe = flags.frag_keyframe || self.options.frag_duration_us.is_none();
		if by_keyframe && track.kind == StreamKind::Video && keyframe && !track.samples.is_empty() {
			return true;
		}
		let by_duration = self.options.frag_duration_us.or((!has_video).then_some(DEFAULT_FRAGMENT_US));
		by_duration.is_some_and(|limit| elapsed >= limit)
	}

	/// Writes the init segment (`moov` without samples) before the first
	/// fragment, so its edit lists can describe the first samples.
	fn write_init(&mut self) -> Result<()> {
		for track in &mut self.tracks {
			track.cts_shift = track.needed_cts_shift();
			track.edit = track.fragment_edit_media_time();
		}
		let (moov, mehd_at) = self.build_moov(Layout::Fragmented)?;
		self.mehd_at = mehd_at.map(|at| self.position + at as u64);
		self.io.write_all(&moov)?;
		self.position += moov.len() as u64;
		self.moov_written = true;
		Ok(())
	}

	/// Writes the buffered fragment as `moof` + `mdat` (one pair per track
	/// with `separate_moof`).
	fn flush_fragment(&mut self) -> Result<()> {
		if self.pending_samples == 0 {
			return Ok(());
		}
		for track in &mut self.tracks {
			if track.derive_dts && !track.samples.is_empty() {
				let previous = track.last_dts;
				track.derive_decode_times(true);
				if previous.is_some_and(|p| track.samples[0].dts <= p) {
					return Err(Message::Container(
						"mp4 muxer: frames are reordered across a fragment boundary; packets without dts can't be fragmented this way",
					));
				}
			}
		}
		if !self.moov_written {
			self.write_init()?;
		}
		let with_samples: Vec<usize> =
			(0..self.tracks.len()).filter(|&i| !self.tracks[i].samples.is_empty()).collect();
		let groups: Vec<Vec<usize>> = if self.options.flags.separate_moof {
			with_samples.iter().map(|&i| vec![i]).collect()
		} else {
			vec![with_samples]
		};
		for group in groups {
			self.write_fragment(&group)?;
		}
		for track in &mut self.tracks {
			let Some(last) = track.samples.len().checked_sub(1) else { continue };
			let origin = *track.origin.get_or_insert(track.samples[0].dts);
			track.end_dts =
				track.samples[last].dts.saturating_add(track.duration_of(last)).saturating_sub(origin);
			let presented_end = (0..track.samples.len())
				.map(|i| {
					let s = &track.samples[i];
					i128::from(s.dts)
						+ i128::from(s.cts)
						+ i128::from(track.cts_shift)
						+ i128::from(track.duration_of(i))
						- i128::from(origin)
				})
				.max()
				.unwrap_or(0);
			track.end_pts = track.end_pts.max(u64::try_from(presented_end).unwrap_or(0));
			track.last_dts = Some(track.samples[last].dts);
			track.samples.clear();
			track.pending.clear();
		}
		self.pending_samples = 0;
		self.pending_bytes = 0;
		self.fragment_start = None;
		Ok(())
	}

	fn write_fragment(&mut self, group: &[usize]) -> Result<()> {
		self.sequence =
			self.sequence.checked_add(1).ok_or(Message::Container("mp4 muxer: too many fragments"))?;
		let moof_offset = self.position;
		let samples: Vec<Vec<FragmentSample>> =
			group.iter().map(|&i| self.tracks[i].fragment_samples()).collect();
		let parts: Vec<TrackFragment> = group
			.iter()
			.zip(&samples)
			.map(|(&i, samples)| {
				let t = &self.tracks[i];
				let origin = t.origin.unwrap_or(t.samples[0].dts);
				TrackFragment {
					track_id: t.track_id,
					kind: t.kind,
					base_dts: t.samples[0].dts.saturating_sub(origin),
					samples,
				}
			})
			.collect();
		let sizes: Vec<u64> = group.iter().map(|&i| self.tracks[i].pending.len() as u64).collect();
		let header = fragment::fragment_header(self.sequence, &parts, &sizes)?;
		self.io.write_all(&header)?;
		self.position += header.len() as u64;
		for (traf_number, &i) in (1u8..).zip(group) {
			let track = &mut self.tracks[i];
			self.io.write_all(&track.pending)?;
			self.position += track.pending.len() as u64;
			if track.samples[0].keyframe {
				let origin = track.origin.unwrap_or(track.samples[0].dts);
				let first = &track.samples[0];
				let pts = (i128::from(first.dts) + i128::from(first.cts) + i128::from(track.cts_shift)
					- i128::from(origin))
				.max(0);
				track.random_access.push(RandomAccess {
					time: u64::try_from(pts).unwrap_or(0),
					moof_offset,
					traf_number,
				});
			}
		}
		Ok(())
	}

	fn finish_fragmented(&mut self) -> Result<()> {
		self.flush_fragment()?;
		if !self.moov_written {
			self.write_init()?;
		}
		if !self.options.flags.skip_trailer {
			let index: Vec<(u32, &[RandomAccess])> =
				self.tracks.iter().map(|t| (t.track_id, t.random_access.as_slice())).collect();
			if index.iter().any(|(_, e)| !e.is_empty()) {
				let mfra = fragment::write_mfra(&index)?;
				self.io.write_all(&mfra)?;
				self.position += mfra.len() as u64;
			}
		}
		// mehd: the longest track, after its edit.
		if let Some(at) = self.mehd_at.filter(|_| self.seekable) {
			let duration = self
				.tracks
				.iter()
				.map(|t| t.to_movie(t.end_pts.saturating_sub(t.edit.unwrap_or(0))))
				.max()
				.unwrap_or(0);
			self.io.seek(at)?;
			self.io.write_all(&duration.to_be_bytes())?;
			self.io.seek(self.position)?;
		}
		Ok(())
	}
}

/// Microseconds from `(start, start_scale)` to `(now, scale)`, 0 if `now`
/// is earlier.
fn elapsed_us(start: u64, start_scale: u32, now: u64, scale: u32) -> u64 {
	let a = u128::from(now) * u128::from(start_scale);
	let b = u128::from(start) * u128::from(scale);
	let denominator = u128::from(start_scale) * u128::from(scale);
	if denominator == 0 {
		return 0;
	}
	u64::try_from(a.saturating_sub(b) * 1_000_000 / denominator).unwrap_or(u64::MAX)
}

fn read_exact(io: &mut dyn Io, mut buf: &mut [u8]) -> Result<()> {
	while !buf.is_empty() {
		let n = io.read(buf)?;
		if n == 0 {
			return Err(Message::Container("mp4 muxer: output ended while moving mdat"));
		}
		buf = &mut buf[n..];
	}
	Ok(())
}

/// Writes version 0 when every value fits 32 bits, version 1 otherwise.
fn version_for(values: &[u64]) -> u8 {
	u8::from(values.iter().any(|&v| v > u64::from(u32::MAX)))
}

fn put_time(out: &mut Vec<u8>, version: u8, value: u64) {
	if version == 1 { out.put_u64(value) } else { out.put_u32(value as u32) }
}

fn write_mvhd(out: &mut Vec<u8>, duration: u64, next_track_id: u32) -> Result<()> {
	let version = version_for(&[duration]);
	write_full_box(out, b"mvhd", version, 0, |out| {
		put_time(out, version, 0); // creation time
		put_time(out, version, 0); // modification time
		out.put_u32(MOVIE_TIMESCALE);
		put_time(out, version, duration);
		out.put_u32(0x0001_0000); // rate 1.0
		out.put_u16(0x0100); // volume 1.0
		out.put_zeros(10);
		IDENTITY_MATRIX.iter().for_each(|&v| out.put_u32(v));
		out.put_zeros(24); // pre_defined
		out.put_u32(next_track_id);
		Ok(())
	})
}

fn write_trak(out: &mut Vec<u8>, track: &MuxTrack, layout: Layout) -> Result<()> {
	// Edit list: start presentation at the first composition time (B-frame
	// delay) or after the audio priming samples. Fragmented files don't
	// know their durations up front: 0 there means "the whole track", as
	// in FFmpeg's `delay_moov` output.
	let (media_duration, movie_duration, media_time) = match layout {
		Layout::Progressive { .. } => (
			track.media_duration(),
			track.to_movie(track.presentation_duration()),
			track.edit_media_time(),
		),
		Layout::Fragmented => (0, 0, track.edit),
	};

	write_box(out, b"trak", |out| {
		let version = version_for(&[movie_duration]);
		write_full_box(out, b"tkhd", version, 0x3, |out| {
			put_time(out, version, 0);
			put_time(out, version, 0);
			out.put_u32(track.track_id);
			out.put_u32(0);
			put_time(out, version, movie_duration);
			out.put_zeros(8);
			out.put_u16(0); // layer
			out.put_u16(0); // alternate group
			out.put_u16(if track.kind == StreamKind::Audio { 0x0100 } else { 0 });
			out.put_u16(0);
			IDENTITY_MATRIX.iter().for_each(|&v| out.put_u32(v));
			out.put_u32(track.width << 16);
			out.put_u32(track.height << 16);
			Ok(())
		})?;

		if let Some(media_time) = media_time {
			let segment = movie_duration;
			let version = version_for(&[segment, media_time]);
			write_box(out, b"edts", |out| {
				write_full_box(out, b"elst", version, 0, |out| {
					out.put_u32(1);
					put_time(out, version, segment);
					put_time(out, version, media_time);
					out.put_u32(0x0001_0000); // rate 1.0
					Ok(())
				})
			})?;
		}

		write_box(out, b"mdia", |out| {
			let version = version_for(&[media_duration]);
			write_full_box(out, b"mdhd", version, 0, |out| {
				put_time(out, version, 0);
				put_time(out, version, 0);
				out.put_u32(track.timescale);
				put_time(out, version, media_duration);
				out.put_u16(0x55C4); // "und"
				out.put_u16(0);
				Ok(())
			})?;
			let (handler, name): (&[u8; 4], &[u8]) = match track.kind {
				StreamKind::Video => (b"vide", b"VideoHandler\0"),
				_ => (b"soun", b"SoundHandler\0"),
			};
			write_full_box(out, b"hdlr", 0, 0, |out| {
				out.put_u32(0);
				out.put(handler);
				out.put_zeros(12);
				out.put(name);
				Ok(())
			})?;
			write_box(out, b"minf", |out| {
				if track.kind == StreamKind::Video {
					write_full_box(out, b"vmhd", 0, 1, |out| {
						out.put_zeros(8);
						Ok(())
					})?;
				} else {
					write_full_box(out, b"smhd", 0, 0, |out| {
						out.put_zeros(4);
						Ok(())
					})?;
				}
				write_box(out, b"dinf", |out| {
					write_full_box(out, b"dref", 0, 0, |out| {
						out.put_u32(1);
						write_full_box(out, b"url ", 0, 1, |_| Ok(()))
					})
				})?;
				write_stbl(out, track, layout)
			})
		})
	})
}

fn write_stbl(out: &mut Vec<u8>, track: &MuxTrack, layout: Layout) -> Result<()> {
	let (samples, chunks, shift): (&[MuxSample], &[Chunk], u64) = match layout {
		Layout::Progressive { offset_shift } => (&track.samples, &track.chunks, offset_shift),
		Layout::Fragmented => (&[], &[], 0),
	};
	let avg_bitrate = track.average_bitrate(matches!(layout, Layout::Progressive { .. }));
	let params = EntryParams {
		codec: &track.codec,
		kind: track.kind,
		width: track.width,
		height: track.height,
		sample_rate: track.sample_rate,
		channels: track.channels,
		extradata: &track.extradata,
		track_id: track.track_id,
		avg_bitrate,
		max_bitrate: avg_bitrate,
		buffer_size: track.max_sample_size,
	};

	write_box(out, b"stbl", |out| {
		codec::write_stsd(out, &params)?;

		// stts: run-length decode-time deltas.
		let mut stts: Vec<(u32, u64)> = Vec::new();
		for i in 0..samples.len() {
			let delta = track.duration_of(i);
			match stts.last_mut() {
				Some((count, d)) if *d == delta && *count < u32::MAX => *count += 1,
				_ => stts.push((1, delta)),
			}
		}
		write_full_box(out, b"stts", 0, 0, |out| {
			out.put_u32(stts.len() as u32);
			for &(count, delta) in &stts {
				out.put_u32(count);
				out.put_u32(u32::try_from(delta).unwrap_or(u32::MAX));
			}
			Ok(())
		})?;

		// ctts: only when some sample is displayed out of decode order.
		if samples.iter().any(|s| s.cts != 0) {
			let mut ctts: Vec<(u32, i64)> = Vec::new();
			for sample in samples {
				match ctts.last_mut() {
					Some((count, offset)) if *offset == sample.cts && *count < u32::MAX => *count += 1,
					_ => ctts.push((1, sample.cts)),
				}
			}
			let version = u8::from(ctts.iter().any(|&(_, o)| o < 0));
			write_full_box(out, b"ctts", version, 0, |out| {
				out.put_u32(ctts.len() as u32);
				for &(count, offset) in &ctts {
					out.put_u32(count);
					out.put_i32(offset.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32);
				}
				Ok(())
			})?;
		}

		// stss: only for video with non-key samples.
		if track.kind == StreamKind::Video && samples.iter().any(|s| !s.keyframe) {
			let sync: Vec<u32> =
				(1u32..).zip(samples).filter(|(_, s)| s.keyframe).map(|(n, _)| n).collect();
			write_full_box(out, b"stss", 0, 0, |out| {
				out.put_u32(sync.len() as u32);
				sync.iter().for_each(|&n| out.put_u32(n));
				Ok(())
			})?;
		}

		// stsc: run-length samples-per-chunk.
		let mut stsc: Vec<(u32, u32)> = Vec::new();
		for (chunk, number) in chunks.iter().zip(1u32..) {
			if stsc.last().is_none_or(|&(_, per_chunk)| per_chunk != chunk.samples) {
				stsc.push((number, chunk.samples));
			}
		}
		write_full_box(out, b"stsc", 0, 0, |out| {
			out.put_u32(stsc.len() as u32);
			for &(first_chunk, per_chunk) in &stsc {
				out.put_u32(first_chunk);
				out.put_u32(per_chunk);
				out.put_u32(1); // sample description index
			}
			Ok(())
		})?;

		// stsz: constant size when possible.
		let constant =
			track.samples.first().map(|s| s.size).filter(|&size| samples.iter().all(|s| s.size == size));
		write_full_box(out, b"stsz", 0, 0, |out| {
			out.put_u32(constant.unwrap_or(0));
			out.put_u32(samples.len() as u32);
			if constant.is_none() {
				samples.iter().for_each(|s| out.put_u32(s.size));
			}
			Ok(())
		})?;

		// stco, or co64 once offsets pass 4 GiB.
		if chunks.iter().any(|c| c.offset + shift > u64::from(u32::MAX)) {
			write_full_box(out, b"co64", 0, 0, |out| {
				out.put_u32(chunks.len() as u32);
				chunks.iter().for_each(|c| out.put_u64(c.offset + shift));
				Ok(())
			})
		} else {
			write_full_box(out, b"stco", 0, 0, |out| {
				out.put_u32(chunks.len() as u32);
				chunks.iter().for_each(|c| out.put_u32((c.offset + shift) as u32));
				Ok(())
			})
		}
	})
}

impl Muxer for Mp4Muxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	/// `movflags`, `frag_duration` and `min_frag_duration`, with FFmpeg's
	/// meanings (see [`super::options`]). They decide the file layout, so
	/// they must be set before the first packet.
	fn set_option(&mut self, key: &str, value: &str) -> Result<()> {
		if self.started {
			return Err(Message::Container("mp4 muxer: options must be set before writing"));
		}
		if self.options.set(key, value)? {
			Ok(())
		} else {
			Err(Message::Container("mp4 muxer: unknown option"))
		}
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.started {
			return Err(Message::Container("mp4 muxer: streams must be added before writing"));
		}
		if self.tracks.iter().any(|t| t.stream_id == stream.id) {
			return Err(Message::Container("mp4 muxer: duplicate stream id"));
		}
		if !matches!(stream.kind, StreamKind::Audio | StreamKind::Video) {
			return Err(Message::Container("mp4 muxer supports only audio and video streams"));
		}
		let p = &stream.parameters;
		if !codec::muxer_supports(&p.codec.id) {
			return Err(Message::Container("mp4 muxer: unsupported codec"));
		}
		let timescale = Self::timescale_for(stream)?;
		let sample_rate = p.sample_rate.unwrap_or(0);
		let samples_key = |key: &str| match stream.metadata.raw.get(key) {
			Some(RawValue::String(v)) => v.parse::<u64>().ok().and_then(|d| {
				rescale_ticks(d, TimeBase::new(1, sample_rate.max(1)), TimeBase::new(1, timescale))
			}),
			_ => None,
		};
		let encoder_delay = samples_key(ENCODER_DELAY_KEY);
		let encoder_padding = samples_key(ENCODER_PADDING_KEY).unwrap_or(0);
		self.tracks.push(MuxTrack {
			stream_id: stream.id,
			kind: stream.kind,
			codec: p.codec.id.clone(),
			track_id: self.tracks.len() as u32 + 1,
			timescale,
			width: p.width.unwrap_or(0),
			height: p.height.unwrap_or(0),
			sample_rate,
			channels: p.channels.unwrap_or(0),
			extradata: p.extradata.clone(),
			encoder_delay,
			encoder_padding,
			samples: Vec::new(),
			chunks: Vec::new(),
			last_duration: None,
			next_dts: 0,
			bytes: 0,
			max_sample_size: 0,
			derive_dts: false,
			bitrate: p.bitrate,
			last_dts: None,
			pending: Vec::new(),
			origin: None,
			end_dts: 0,
			end_pts: 0,
			edit: None,
			cts_shift: 0,
			random_access: Vec::new(),
		});
		Ok(self.tracks.len() - 1)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		let index = self
			.tracks
			.iter()
			.position(|t| t.stream_id == packet.stream_id)
			.ok_or(Message::Container("mp4 muxer: packet for unknown stream"))?;
		if !self.started {
			self.start()?;
		}
		let size = u32::try_from(packet.data.len())
			.map_err(|_| Message::Container("mp4 muxer: packet too large"))?;

		// 1. Timing, in track ticks.
		let track = &mut self.tracks[index];
		let timescale = track.timescale;
		let duration = match packet.duration {
			Some(d) => Some(Self::to_track_ticks(d, timescale)?),
			None => None,
		};
		let first_packet = track.last_dts.is_none() && track.samples.is_empty();
		if track.kind == StreamKind::Video && packet.dts.is_none() && packet.pts.is_some() {
			if first_packet {
				track.derive_dts = true;
			}
		} else if track.derive_dts {
			return Err(Message::Container("mp4 muxer: mixed packets with and without dts"));
		}
		let dts = match packet.dts.or(packet.pts) {
			Some(t) => Self::to_track_ticks(t, timescale)?,
			None if !first_packet || duration.is_some() => track.next_dts,
			None => match track.implicit_duration(&packet.data) {
				Some(_) => track.next_dts,
				None => return Err(Message::Container("mp4 muxer: packet has no timestamps")),
			},
		};
		let pts = match packet.pts {
			Some(t) => Self::to_track_ticks(t, timescale)?,
			None => dts,
		};
		if !track.derive_dts {
			let previous = track.samples.last().map(|s| s.dts).or(track.last_dts);
			if previous.is_some_and(|last| dts < last) {
				return Err(Message::Container("mp4 muxer: non-monotonic dts"));
			}
			track.last_dts = Some(dts);
		}
		let step = duration.or_else(|| track.implicit_duration(&packet.data));

		// 2. Fragmented: this packet may open the next fragment.
		if self.fragmented() {
			if self.starts_fragment(index, dts, packet.keyframe) {
				self.flush_fragment()?;
			}
			if self.fragment_start.is_none() {
				self.fragment_start = Some((dts, timescale));
			}
			self.pending_samples += 1;
			self.pending_bytes += u64::from(size);
		}

		// 3. Bookkeeping, then the data: straight to mdat, or into the
		// fragment buffer.
		let track = &mut self.tracks[index];
		track.last_duration = step;
		track.next_dts = dts.saturating_add(step.unwrap_or(0));
		track.samples.push(MuxSample {
			size,
			dts,
			cts: (i128::from(pts) - i128::from(dts)).clamp(i128::from(i64::MIN), i128::from(i64::MAX))
				as i64,
			keyframe: packet.keyframe,
		});
		track.bytes += u64::from(size);
		track.max_sample_size = track.max_sample_size.max(size);

		if self.options.fragmented() {
			track.pending.extend_from_slice(&packet.data);
			return Ok(());
		}
		let continues_chunk = self.last_track == Some(index);
		match track.chunks.last_mut() {
			Some(chunk) if continues_chunk => chunk.samples += 1,
			_ => track.chunks.push(Chunk { offset: self.position, samples: 1 }),
		}
		self.last_track = Some(index);

		self.io.write_all(&packet.data)?;
		self.position += u64::from(size);
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		if !self.started {
			self.start()?;
		}
		if self.fragmented() { self.finish_fragmented() } else { self.finish_progressive() }
	}
}
