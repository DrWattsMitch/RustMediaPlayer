//! Movie fragments (ISO/IEC 14496-12 §8.8): the boxes a fragmented MP4 is
//! made of, as FFmpeg's `movenc.c` writes them with
//! `-movflags frag_keyframe+empty_moov+default_base_moof`.
//!
//! ```text
//! ftyp (iso5) | moov (no samples, mvex/trex) | moof mdat | moof mdat | ... | mfra
//!                                               │
//!                                               ├ mfhd  sequence number
//!                                               └ traf  per track
//!                                                  ├ tfhd  track id, defaults, default-base-is-moof
//!                                                  ├ tfdt  decode time of the first sample
//!                                                  └ trun  per-sample duration/size/flags/cts
//! ```
//!
//! Writing: [`super::muxer`] decides where fragments start and buffers
//! their samples; this module serialises them. Reading: [`parse_moof`]
//! turns one `moof` into flat [`Sample`]s for [`super::demuxer`]. Both are
//! pure functions of their inputs, so they are tested without any I/O.

use super::boxes::{BoxIter, Put, begin_box, end_box, find, full_box, write_box, write_full_box};
use super::sample_table::{MAX_SAMPLE_SIZE, Sample};
use crate::core::StreamKind;
use crate::io::ByteCursor;
use crate::message::{Message, Result};

/// `tfhd` flags.
const TFHD_DEFAULT_DURATION: u32 = 0x00_0008;
const TFHD_DEFAULT_SIZE: u32 = 0x00_0010;
const TFHD_DEFAULT_FLAGS: u32 = 0x00_0020;
const TFHD_DEFAULT_BASE_IS_MOOF: u32 = 0x02_0000;

/// `trun` flags.
const TRUN_DATA_OFFSET: u32 = 0x00_0001;
const TRUN_FIRST_SAMPLE_FLAGS: u32 = 0x00_0004;
const TRUN_DURATION: u32 = 0x00_0100;
const TRUN_SIZE: u32 = 0x00_0200;
const TRUN_FLAGS: u32 = 0x00_0400;
const TRUN_CTS: u32 = 0x00_0800;

/// Sample flags (§8.8.3.1): `sample_depends_on = 2` (an I-frame / any
/// audio sample) and `sample_depends_on = 1 | sample_is_non_sync_sample`.
pub const SYNC_SAMPLE_FLAGS: u32 = 0x0200_0000;
pub const NON_SYNC_SAMPLE_FLAGS: u32 = 0x0101_0000;

/// `true` when sample flags mark a sync sample (keyframe).
pub fn is_sync(flags: u32) -> bool {
	flags & 0x0001_0000 == 0
}

/// One sample of a track fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentSample {
	pub duration: u32,
	pub size: u32,
	/// pts − dts in track ticks (negative needs a version 1 `trun`).
	pub cts: i32,
	pub keyframe: bool,
}

/// The part of one track that goes into a fragment.
#[derive(Debug, Clone, Copy)]
pub struct TrackFragment<'a> {
	pub track_id: u32,
	pub kind: StreamKind,
	/// `tfdt`: decode time of the first sample, in track ticks.
	pub base_dts: u64,
	pub samples: &'a [FragmentSample],
}

fn sample_flags(kind: StreamKind, keyframe: bool) -> u32 {
	if kind == StreamKind::Video && !keyframe { NON_SYNC_SAMPLE_FLAGS } else { SYNC_SAMPLE_FLAGS }
}

/// The value every item shares, if any (`None` for no items).
fn uniform(mut values: impl Iterator<Item = u32>) -> Option<u32> {
	let first = values.next()?;
	values.all(|v| v == first).then_some(first)
}

/// Writes one `traf`; returns the position of its `trun` data offset.
fn write_traf(out: &mut Vec<u8>, t: &TrackFragment) -> Result<usize> {
	let samples = t.samples;
	if samples.is_empty() {
		return Err(Message::Container("mp4 fragment: empty track fragment"));
	}
	let flags: Vec<u32> = samples.iter().map(|s| sample_flags(t.kind, s.keyframe)).collect();

	// Values every sample shares go into tfhd once (FFmpeg's
	// `mov_write_tfhd_tag` does the same); usually only the first sample of
	// a video fragment (the keyframe) differs, which trun's
	// first_sample_flags covers.
	let default_duration = uniform(samples.iter().map(|s| s.duration));
	let default_size = uniform(samples.iter().map(|s| s.size));
	let rest_flags = uniform(flags.iter().skip(1).copied());
	let (default_flags, first_flags, per_sample_flags) = match (samples.len(), rest_flags) {
		(1, _) => (flags[0], None, false),
		(_, Some(rest)) => (rest, (flags[0] != rest).then_some(flags[0]), false),
		(_, None) => (flags[0], None, true),
	};
	let any_cts = samples.iter().any(|s| s.cts != 0);
	let version = u8::from(samples.iter().any(|s| s.cts < 0));

	let mut tfhd_flags = TFHD_DEFAULT_BASE_IS_MOOF | TFHD_DEFAULT_FLAGS;
	tfhd_flags |= if default_duration.is_some() { TFHD_DEFAULT_DURATION } else { 0 };
	tfhd_flags |= if default_size.is_some() { TFHD_DEFAULT_SIZE } else { 0 };

	let mut trun_flags = TRUN_DATA_OFFSET;
	trun_flags |= if first_flags.is_some() { TRUN_FIRST_SAMPLE_FLAGS } else { 0 };
	trun_flags |= if default_duration.is_none() { TRUN_DURATION } else { 0 };
	trun_flags |= if default_size.is_none() { TRUN_SIZE } else { 0 };
	trun_flags |= if per_sample_flags { TRUN_FLAGS } else { 0 };
	trun_flags |= if any_cts { TRUN_CTS } else { 0 };

	let count =
		u32::try_from(samples.len()).map_err(|_| Message::Container("mp4 fragment too long"))?;
	let mut data_offset_at = 0;
	write_box(out, b"traf", |out| {
		write_full_box(out, b"tfhd", 0, tfhd_flags, |out| {
			out.put_u32(t.track_id);
			default_duration.iter().for_each(|&d| out.put_u32(d));
			default_size.iter().for_each(|&s| out.put_u32(s));
			out.put_u32(default_flags);
			Ok(())
		})?;
		write_full_box(out, b"tfdt", 1, 0, |out| {
			out.put_u64(t.base_dts);
			Ok(())
		})?;
		write_full_box(out, b"trun", version, trun_flags, |out| {
			out.put_u32(count);
			data_offset_at = out.len();
			out.put_i32(0); // patched once the moof size is known
			first_flags.iter().for_each(|&f| out.put_u32(f));
			for (sample, &f) in samples.iter().zip(&flags) {
				if default_duration.is_none() {
					out.put_u32(sample.duration);
				}
				if default_size.is_none() {
					out.put_u32(sample.size);
				}
				if per_sample_flags {
					out.put_u32(f);
				}
				if any_cts {
					out.put_i32(sample.cts);
				}
			}
			Ok(())
		})
	})?;
	Ok(data_offset_at)
}

/// Builds `moof` + the `mdat` header for one fragment. The caller writes
/// each track's sample data right after, in the order of `tracks`, so
/// `data_sizes[i]` must be the byte length of track `i`'s data.
pub fn fragment_header(
	sequence: u32,
	tracks: &[TrackFragment],
	data_sizes: &[u64],
) -> Result<Vec<u8>> {
	if tracks.is_empty() || tracks.len() != data_sizes.len() {
		return Err(Message::Container("mp4 fragment: track data mismatch"));
	}
	let mut out =
		Vec::with_capacity(256 + tracks.iter().map(|t| t.samples.len() * 16).sum::<usize>());
	let mut offsets = Vec::with_capacity(tracks.len());
	write_box(&mut out, b"moof", |out| {
		write_full_box(out, b"mfhd", 0, 0, |out| {
			out.put_u32(sequence);
			Ok(())
		})?;
		for track in tracks {
			offsets.push(write_traf(out, track)?);
		}
		Ok(())
	})?;

	let payload: u64 = data_sizes.iter().sum();
	let large = payload > u64::from(u32::MAX) - 8;
	let mdat_header: u64 = if large { 16 } else { 8 };
	// default-base-is-moof: offsets count from the first byte of moof.
	let mut data_offset = out.len() as u64 + mdat_header;
	for (&at, &size) in offsets.iter().zip(data_sizes) {
		let value = i32::try_from(data_offset)
			.map_err(|_| Message::Container("mp4 fragment too large (lower -frag_duration)"))?;
		out[at..at + 4].copy_from_slice(&value.to_be_bytes());
		data_offset += size;
	}
	if large {
		out.put_u32(1);
		out.put(b"mdat");
		out.put_u64(payload + 16);
	} else {
		out.put_u32((payload + 8) as u32);
		out.put(b"mdat");
	}
	Ok(out)
}

/// Writes `mvex`: `mehd` (overall duration, version 1 so it can be patched
/// in place) and one `trex` per track. Returns the position of the `mehd`
/// duration field within `out`.
pub fn write_mvex(out: &mut Vec<u8>, track_ids: &[u32]) -> Result<usize> {
	let mut mehd_at = 0;
	write_box(out, b"mvex", |out| {
		write_full_box(out, b"mehd", 1, 0, |out| {
			mehd_at = out.len();
			out.put_u64(0); // patched by the muxer when the output can seek
			Ok(())
		})?;
		for &id in track_ids {
			write_full_box(out, b"trex", 0, 0, |out| {
				out.put_u32(id);
				out.put_u32(1); // sample description index
				out.put_u32(0); // default duration / size / flags: per tfhd
				out.put_u32(0);
				out.put_u32(0);
				Ok(())
			})?;
		}
		Ok(())
	})?;
	Ok(mehd_at)
}

/// A random access point: a fragment whose track data starts with a sync
/// sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomAccess {
	/// Presentation time in track ticks.
	pub time: u64,
	/// File offset of the `moof`.
	pub moof_offset: u64,
	/// 1-based position of the `traf` within its `moof`.
	pub traf_number: u8,
}

/// Writes `mfra` (`tfra` per track + `mfro`), the index players use to
/// seek in a fragmented file without scanning every `moof`.
pub fn write_mfra(tracks: &[(u32, &[RandomAccess])]) -> Result<Vec<u8>> {
	let mut out = Vec::new();
	let at = begin_box(&mut out, b"mfra");
	for &(track_id, entries) in tracks.iter().filter(|(_, e)| !e.is_empty()) {
		let count =
			u32::try_from(entries.len()).map_err(|_| Message::Container("mp4 mfra too large"))?;
		write_full_box(&mut out, b"tfra", 1, 0, |out| {
			out.put_u32(track_id);
			// Reserved + traf/trun/sample number sizes: 1 byte each.
			out.put_u32(0);
			out.put_u32(count);
			for e in entries {
				out.put_u64(e.time);
				out.put_u64(e.moof_offset);
				out.put_u8(e.traf_number);
				out.put_u8(1); // trun number
				out.put_u8(1); // sample number
			}
			Ok(())
		})?;
	}
	write_full_box(&mut out, b"mfro", 0, 0, |out| {
		out.put_u32((out.len() - at + 4) as u32);
		Ok(())
	})?;
	end_box(&mut out, at)?;
	Ok(out)
}

// ---------------------------------------------------------------------------
// Reading

/// Per-track defaults from `trex` (overridden by `tfhd`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackDefaults {
	pub duration: u32,
	pub size: u32,
	pub flags: u32,
}

/// Parses `mvex/trex` → (track id, defaults).
pub fn parse_trex(mvex: &[u8]) -> Vec<(u32, TrackDefaults)> {
	BoxIter::new(mvex)
		.map_while(|r| r.ok())
		.filter(|(kind, _)| kind == b"trex")
		.filter_map(|(_, body)| {
			let (_, _, payload) = full_box(body).ok()?;
			let mut c = ByteCursor::new(payload);
			let id = c.read_u32().ok()?;
			c.skip(4).ok()?; // sample description index
			let duration = c.read_u32().ok()?;
			let size = c.read_u32().ok()?;
			let flags = c.read_u32().ok()?;
			Some((id, TrackDefaults { duration, size, flags }))
		})
		.collect()
}

/// What the demuxer knows about a track while reading fragments.
#[derive(Debug, Clone)]
pub struct FragmentedTrack {
	pub track_id: u32,
	pub defaults: TrackDefaults,
	/// Decode time where the next fragment continues when it has no `tfdt`.
	pub next_dts: u64,
	pub samples: Vec<Sample>,
}

const TRUNCATED: Message = Message::Container("truncated mp4 track fragment");

/// Appends the samples of one `moof` (body at file offset `moof_offset`) to
/// `tracks`. Samples that lie beyond `file_size` (an unfinished download
/// or recording) and samples past `max_samples` per track are dropped, so a
/// hostile file can't make the index grow without bound.
pub fn parse_moof(
	moof: &[u8],
	moof_offset: u64,
	file_size: Option<u64>,
	max_samples: usize,
	tracks: &mut [FragmentedTrack],
) -> Result<()> {
	// Without base-data-offset or default-base-is-moof, a traf's data
	// continues where the previous traf's ended (§8.8.7.1).
	let mut previous_end = moof_offset;
	for (kind, traf) in BoxIter::new(moof).map_while(|r| r.ok()) {
		if &kind != b"traf" {
			continue;
		}
		let tfhd = find(traf, b"tfhd").ok_or(Message::Container("mp4 traf without tfhd"))?;
		let (_, flags, payload) = full_box(tfhd)?;
		let mut c = ByteCursor::new(payload);
		let id = c.read_u32()?;
		let Some(track) = tracks.iter_mut().find(|t| t.track_id == id) else { continue };
		let base = if flags & 0x1 != 0 {
			c.read_u64()?
		} else if flags & TFHD_DEFAULT_BASE_IS_MOOF != 0 {
			moof_offset
		} else {
			previous_end
		};
		if flags & 0x2 != 0 {
			c.skip(4)?; // sample description index
		}
		let mut defaults = track.defaults;
		if flags & TFHD_DEFAULT_DURATION != 0 {
			defaults.duration = c.read_u32()?;
		}
		if flags & TFHD_DEFAULT_SIZE != 0 {
			defaults.size = c.read_u32()?;
		}
		if flags & TFHD_DEFAULT_FLAGS != 0 {
			defaults.flags = c.read_u32()?;
		}

		if let Some(tfdt) = find(traf, b"tfdt") {
			let (version, _, payload) = full_box(tfdt)?;
			let mut c = ByteCursor::new(payload);
			track.next_dts = if version == 1 { c.read_u64()? } else { u64::from(c.read_u32()?) };
		}

		let mut data = base;
		for (kind, trun) in BoxIter::new(traf).map_while(|r| r.ok()) {
			if &kind != b"trun" {
				continue;
			}
			data = parse_trun(trun, base, data, defaults, file_size, max_samples, track)?;
		}
		previous_end = data;
	}
	Ok(())
}

/// One `trun`; returns the file offset after its data.
fn parse_trun(
	trun: &[u8],
	base: u64,
	continues_at: u64,
	defaults: TrackDefaults,
	file_size: Option<u64>,
	max_samples: usize,
	track: &mut FragmentedTrack,
) -> Result<u64> {
	let (_, flags, payload) = full_box(trun)?;
	let mut c = ByteCursor::new(payload);
	let count = c.read_u32()? as usize;
	let mut offset = if flags & TRUN_DATA_OFFSET != 0 {
		let relative = i64::from(c.read_i32()?);
		base.checked_add_signed(relative).ok_or(TRUNCATED)?
	} else {
		continues_at
	};
	let first_flags = if flags & TRUN_FIRST_SAMPLE_FLAGS != 0 { Some(c.read_u32()?) } else { None };
	let per_sample =
		[TRUN_DURATION, TRUN_SIZE, TRUN_FLAGS, TRUN_CTS].iter().filter(|&&f| flags & f != 0).count()
			* 4;
	if count.checked_mul(per_sample).is_none_or(|bytes| bytes > c.remaining()) {
		return Err(TRUNCATED);
	}
	for i in 0..count {
		let duration = if flags & TRUN_DURATION != 0 { c.read_u32()? } else { defaults.duration };
		let size = if flags & TRUN_SIZE != 0 { c.read_u32()? } else { defaults.size };
		let sample_flags = if flags & TRUN_FLAGS != 0 { c.read_u32()? } else { defaults.flags };
		let sample_flags = if i == 0 { first_flags.unwrap_or(sample_flags) } else { sample_flags };
		// Version 0 offsets are unsigned, but FFmpeg reads both as signed.
		let cts = if flags & TRUN_CTS != 0 { c.read_u32()? as i32 } else { 0 };
		let end = offset.checked_add(u64::from(size));
		let outside = end.is_none_or(|e| file_size.is_some_and(|s| e > s));
		if size > MAX_SAMPLE_SIZE || outside || track.samples.len() >= max_samples {
			// Keep what is complete; the rest of this run is unusable.
			return Ok(end.unwrap_or(u64::MAX));
		}
		track.samples.push(Sample {
			offset,
			dts: track.next_dts,
			size,
			cts,
			duration,
			keyframe: is_sync(sample_flags),
		});
		track.next_dts = track.next_dts.saturating_add(u64::from(duration));
		offset = end.unwrap_or(u64::MAX);
	}
	Ok(offset)
}

#[cfg(test)]
mod tests {
	use super::super::boxes::{BoxIter, find, full_box};
	use super::*;

	fn sample(duration: u32, size: u32, cts: i32, keyframe: bool) -> FragmentSample {
		FragmentSample { duration, size, cts, keyframe }
	}

	#[test]
	fn video_fragment_uses_tfhd_defaults_like_ffmpeg() {
		let samples =
			[sample(512, 100, 1024, true), sample(512, 40, 0, false), sample(512, 30, -512, false)];
		let tf =
			TrackFragment { track_id: 1, kind: StreamKind::Video, base_dts: 12800, samples: &samples };
		let head = fragment_header(7, &[tf], &[170]).unwrap();
		let moof = find(&head, b"moof").unwrap();
		let traf = find(moof, b"traf").unwrap();
		let (_, flags, tfhd) = full_box(find(traf, b"tfhd").unwrap()).unwrap();
		assert_eq!(flags, 0x02_0028); // base-is-moof, default duration + flags
		assert_eq!(&tfhd[..12], &[0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0]);
		let (version, flags, trun) = full_box(find(traf, b"trun").unwrap()).unwrap();
		assert_eq!((version, flags), (1, 0xA05));
		let data_offset = i32::from_be_bytes(trun[4..8].try_into().unwrap());
		assert_eq!(data_offset as usize, head.len()); // first byte after mdat header
		assert_eq!(&trun[8..12], &SYNC_SAMPLE_FLAGS.to_be_bytes());
		let (_, _, tfdt) = full_box(find(traf, b"tfdt").unwrap()).unwrap();
		assert_eq!(tfdt, &12800u64.to_be_bytes());
		let mdat: Vec<_> = BoxIter::new(&head).map(|r| r.unwrap().0).collect();
		assert_eq!(mdat, vec![*b"moof", *b"mdat"]);
	}

	#[test]
	fn second_track_offset_follows_first_track_data() {
		let a = [sample(1024, 10, 0, true); 3];
		let v = [sample(512, 50, 0, true)];
		let tracks = [
			TrackFragment { track_id: 1, kind: StreamKind::Video, base_dts: 0, samples: &v },
			TrackFragment { track_id: 2, kind: StreamKind::Audio, base_dts: 0, samples: &a },
		];
		let head = fragment_header(1, &tracks, &[50, 30]).unwrap();
		let moof = find(&head, b"moof").unwrap();
		let offsets: Vec<i32> = BoxIter::new(moof)
			.map(|r| r.unwrap())
			.filter(|(k, _)| k == b"traf")
			.map(|(_, traf)| {
				let (_, _, trun) = full_box(find(traf, b"trun").unwrap()).unwrap();
				i32::from_be_bytes(trun[4..8].try_into().unwrap())
			})
			.collect();
		assert_eq!(offsets, vec![head.len() as i32, head.len() as i32 + 50]);
		assert!(fragment_header(1, &tracks, &[50]).is_err());
	}

	#[test]
	fn mfra_ends_with_its_own_size() {
		let entries = [RandomAccess { time: 0, moof_offset: 1000, traf_number: 1 }];
		let mfra = write_mfra(&[(1, &entries), (2, &[])]).unwrap();
		let size = u32::from_be_bytes(mfra[mfra.len() - 4..].try_into().unwrap());
		assert_eq!(size as usize, mfra.len());
		let body = find(&mfra, b"mfra").unwrap();
		let kinds: Vec<_> = BoxIter::new(body).map(|r| r.unwrap().0).collect();
		assert_eq!(kinds, vec![*b"tfra", *b"mfro"]);
	}

	#[test]
	fn written_fragments_parse_back() {
		let v = [sample(512, 100, 1024, true), sample(512, 40, 0, false)];
		let a = [sample(1024, 7, 0, true); 2];
		let parts = [
			TrackFragment { track_id: 1, kind: StreamKind::Video, base_dts: 5120, samples: &v },
			TrackFragment { track_id: 2, kind: StreamKind::Audio, base_dts: 9000, samples: &a },
		];
		let head = fragment_header(3, &parts, &[140, 14]).unwrap();
		let moof_offset = 1000u64;
		let moof = find(&head, b"moof").unwrap();
		let mut tracks: Vec<FragmentedTrack> = [1, 2]
			.iter()
			.map(|&id| FragmentedTrack {
				track_id: id,
				defaults: TrackDefaults::default(),
				next_dts: 0,
				samples: Vec::new(),
			})
			.collect();
		parse_moof(moof, moof_offset, None, 100, &mut tracks).unwrap();
		let data = moof_offset + head.len() as u64;
		let video = &tracks[0].samples;
		assert_eq!(video.len(), 2);
		assert_eq!(
			(video[0].offset, video[0].dts, video[0].cts, video[0].keyframe),
			(data, 5120, 1024, true)
		);
		assert_eq!(
			(video[1].offset, video[1].dts, video[1].size, video[1].keyframe),
			(data + 100, 5632, 40, false)
		);
		let audio = &tracks[1].samples;
		assert_eq!((audio[0].offset, audio[1].dts, audio[1].duration), (data + 140, 10024, 1024));
		assert_eq!(tracks[1].next_dts, 11048);

		// A file cut short keeps only the complete samples, and the sample
		// cap is honoured.
		let mut short = tracks.clone();
		short.iter_mut().for_each(|t| t.samples.clear());
		parse_moof(moof, moof_offset, Some(data + 120), 100, &mut short).unwrap();
		assert_eq!((short[0].samples.len(), short[1].samples.len()), (1, 0));
		let mut capped = tracks.clone();
		capped.iter_mut().for_each(|t| t.samples.clear());
		parse_moof(moof, moof_offset, None, 1, &mut capped).unwrap();
		assert_eq!(capped[0].samples.len(), 1);
	}
}
