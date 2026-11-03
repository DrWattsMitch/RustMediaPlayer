//! MP4 muxer (ISO BMFF, `moov` at the end, like FFmpeg's `movenc.c` without
//! `+faststart`).
//!
//! ```text
//! ftyp | mdat (64-bit size, patched in finish) | moov
//! ```
//!
//! Packets are appended to `mdat` as they arrive; consecutive packets of the
//! same track form a chunk. Per-sample bookkeeping (size, dts, composition
//! offset, keyframe flag) is kept in memory and serialised into the sample
//! tables when [`Muxer::finish`] writes `moov`.

use super::boxes::{Put, write_box, write_full_box};
use super::codec::{self, EntryParams};
use super::demuxer::{ENCODER_DELAY_KEY, ENCODER_PADDING_KEY};
use super::metadata;
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
}

impl MuxTrack {
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
				let first_pts =
					self.samples.iter().map(|s| i128::from(s.dts) + i128::from(s.cts)).min().unwrap_or(0);
				u64::try_from(first_pts).ok().filter(|&p| p > 0)
			}
		}
	}

	/// Presented duration in track ticks (media duration minus edits).
	fn presentation_duration(&self) -> u64 {
		let media = self.media_duration();
		match self.edit_media_time() {
			Some(start) => {
				let padding = if self.kind == StreamKind::Audio { self.encoder_padding } else { 0 };
				media.saturating_sub(start).saturating_sub(padding)
			}
			None => media,
		}
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

pub struct Mp4Muxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	tracks: Vec<MuxTrack>,
	started: bool,
	mdat_start: u64,
	position: u64,
	last_track: Option<usize>,
}

impl Mp4Muxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self {
			io,
			metadata: Metadata::default(),
			tracks: Vec::new(),
			started: false,
			mdat_start: 0,
			position: 0,
			last_track: None,
		}
	}

	fn start(&mut self) -> Result<()> {
		let mut header = Vec::with_capacity(64);
		write_box(&mut header, b"ftyp", |out| {
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
		self.mdat_start = header.len() as u64;
		header.put_u32(1); // 64-bit size follows
		header.put(b"mdat");
		header.put_u64(0); // patched in finish
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

	fn build_moov(&self) -> Result<Vec<u8>> {
		let mut moov = Vec::new();
		let movie_duration =
			self.tracks.iter().map(|t| t.to_movie(t.presentation_duration())).max().unwrap_or(0);
		write_box(&mut moov, b"moov", |out| {
			write_mvhd(out, movie_duration, self.tracks.len() as u32 + 1)?;
			for track in &self.tracks {
				write_trak(out, track)?;
			}
			metadata::write_udta(out, &self.metadata, ENCODER_NAME)
		})?;
		Ok(moov)
	}
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

fn write_trak(out: &mut Vec<u8>, track: &MuxTrack) -> Result<()> {
	let media_duration = track.media_duration();
	let movie_duration = track.to_movie(track.presentation_duration());
	// Edit list: start presentation at the first composition time (B-frame
	// delay) or after the audio priming samples.
	let media_time = track.edit_media_time();

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
				write_stbl(out, track)
			})
		})
	})
}

fn write_stbl(out: &mut Vec<u8>, track: &MuxTrack) -> Result<()> {
	let seconds = track.media_duration() as f64 / f64::from(track.timescale);
	let avg_bitrate = if seconds > 0.0 { (track.bytes as f64 * 8.0 / seconds) as u32 } else { 0 };
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
		for i in 0..track.samples.len() {
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
		if track.samples.iter().any(|s| s.cts != 0) {
			let mut ctts: Vec<(u32, i64)> = Vec::new();
			for sample in &track.samples {
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
		if track.kind == StreamKind::Video && track.samples.iter().any(|s| !s.keyframe) {
			let sync: Vec<u32> =
				(1u32..).zip(&track.samples).filter(|(_, s)| s.keyframe).map(|(n, _)| n).collect();
			write_full_box(out, b"stss", 0, 0, |out| {
				out.put_u32(sync.len() as u32);
				sync.iter().for_each(|&n| out.put_u32(n));
				Ok(())
			})?;
		}

		// stsc: run-length samples-per-chunk.
		let mut stsc: Vec<(u32, u32)> = Vec::new();
		for (chunk, number) in track.chunks.iter().zip(1u32..) {
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
		let constant = track
			.samples
			.first()
			.map(|s| s.size)
			.filter(|&size| track.samples.iter().all(|s| s.size == size));
		write_full_box(out, b"stsz", 0, 0, |out| {
			out.put_u32(constant.unwrap_or(0));
			out.put_u32(track.samples.len() as u32);
			if constant.is_none() {
				track.samples.iter().for_each(|s| out.put_u32(s.size));
			}
			Ok(())
		})?;

		// stco, or co64 once offsets pass 4 GiB.
		if track.chunks.iter().any(|c| c.offset > u64::from(u32::MAX)) {
			write_full_box(out, b"co64", 0, 0, |out| {
				out.put_u32(track.chunks.len() as u32);
				track.chunks.iter().for_each(|c| out.put_u64(c.offset));
				Ok(())
			})
		} else {
			write_full_box(out, b"stco", 0, 0, |out| {
				out.put_u32(track.chunks.len() as u32);
				track.chunks.iter().for_each(|c| out.put_u32(c.offset as u32));
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

		let track = &mut self.tracks[index];
		let timescale = track.timescale;
		let duration = match packet.duration {
			Some(d) => Some(Self::to_track_ticks(d, timescale)?),
			None => None,
		};
		let dts = match packet.dts.or(packet.pts) {
			Some(t) => Self::to_track_ticks(t, timescale)?,
			None if !track.samples.is_empty() || duration.is_some() => track.next_dts,
			None => match track.implicit_duration(&packet.data) {
				Some(_) => track.next_dts,
				None => return Err(Message::Container("mp4 muxer: packet has no timestamps")),
			},
		};
		let pts = match packet.pts {
			Some(t) => Self::to_track_ticks(t, timescale)?,
			None => dts,
		};
		if track.samples.last().is_some_and(|last| dts < last.dts) {
			return Err(Message::Container("mp4 muxer: non-monotonic dts"));
		}
		let step = duration.or_else(|| track.implicit_duration(&packet.data));
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
		let moov = self.build_moov()?;
		let mdat_size = self.position - self.mdat_start;
		self.io.seek(self.mdat_start + 8)?;
		self.io.write_all(&mdat_size.to_be_bytes())?;
		self.io.seek(self.position)?;
		self.io.write_all(&moov)?;
		debug_assert!(mdat_size >= MDAT_HEADER_SIZE);
		Ok(())
	}
}
