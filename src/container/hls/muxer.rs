//! HLS output (RFC 8216), after FFmpeg's `hlsenc.c`: a media playlist and
//! its segments, MPEG-TS (default) or fragmented MP4, with optional
//! AES-128 or SAMPLE-AES encryption, WebVTT subtitle renditions and a
//! multivariant ("master") playlist.
//!
//! ```text
//! out.m3u8            media playlist (the output path)
//! out0.ts out1.ts ... segments (-hls_segment_filename, %d = sequence number)
//! init.mp4            fMP4 initialization section (-hls_segment_type fmp4)
//! out_vtt.m3u8        subtitle playlist, segments out0.vtt ... (FFmpeg's names)
//! out_master.m3u8     multivariant playlist (-master_pl_name, or when there are subtitles)
//! ```
//!
//! Segments are cut at the first keyframe of the reference stream (the
//! first video stream, else the first stream) at or after each multiple of
//! `-hls_time` (2 s) from the start, as FFmpeg cuts them; durations are the
//! times between cuts. Playlists are rewritten (through a temporary file
//! and a rename) after every segment, so they can be served while being
//! written.
//!
//! Options (FFmpeg's names, see [`HlsOptions::set`]): `hls_time`,
//! `hls_list_size`, `hls_segment_filename`, `hls_segment_type`,
//! `hls_fmp4_init_filename`, `hls_playlist_type`, `hls_flags`,
//! `hls_base_url`, `start_number`, `hls_key_info_file`, `hls_enc`,
//! `hls_enc_key`, `hls_enc_key_url`, `hls_enc_iv`, `master_pl_name`,
//! `hls_subtitle_path`; RMP's own `hls_enc_method` (`aes-128`,
//! `sample-aes`); and the MPEG-TS muxer's options for TS segments.
//!
//! One difference from FFmpeg: `hls_list_size` defaults to 0 (every
//! segment stays listed), as a file written to be played back needs; set
//! it for a sliding live window.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::aes::encrypt_cbc;
use super::sample_aes::SampleCipher;
use crate::container::mp4::muxer::Mp4Muxer;
use crate::container::mp4::options::Mp4Options;
use crate::container::mpegts::{TsMuxer, muxer::TsOptions};
use crate::container::webvtt;
use crate::core::{Metadata, Muxer, Packet, RawValue, Stream, StreamId, StreamKind};
use crate::io::Io;
use crate::message::{Message, Result};

/// An [`Io`] whose bytes the segmenter takes out after each segment.
#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Sink {
	fn take(&self) -> Vec<u8> {
		std::mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()))
	}
}

impl Io for Sink {
	fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
		Err(Message::Other("hls: segment buffer is write-only"))
	}

	fn write(&mut self, buf: &[u8]) -> Result<usize> {
		self.0.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(buf);
		Ok(buf.len())
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
	MpegTs,
	Fmp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionMethod {
	Aes128,
	SampleAes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistType {
	Event,
	Vod,
}

/// `-hls_flags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HlsFlags {
	/// Remove segment files that fall out of the `hls_list_size` window.
	pub delete_segments: bool,
	/// `#EXT-X-INDEPENDENT-SEGMENTS`: every segment starts with a keyframe.
	pub independent_segments: bool,
	/// No `#EXT-X-ENDLIST` at the end.
	pub omit_endlist: bool,
	/// Cut at the reference stream's packets even when they aren't
	/// keyframes.
	pub split_by_time: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HlsOptions {
	pub time: f64,
	pub list_size: usize,
	pub segment_filename: Option<String>,
	pub segment_type: SegmentType,
	pub init_filename: String,
	pub playlist_type: Option<PlaylistType>,
	pub flags: HlsFlags,
	pub base_url: String,
	pub start_number: u64,
	pub key_info_file: Option<PathBuf>,
	pub enc: bool,
	pub enc_key: Option<[u8; 16]>,
	pub enc_key_url: Option<String>,
	pub enc_iv: Option<[u8; 16]>,
	pub enc_method: EncryptionMethod,
	pub master_name: Option<String>,
	pub subtitle_path: Option<String>,
	/// Options for the MPEG-TS segments.
	pub ts: TsOptions,
}

impl Default for HlsOptions {
	fn default() -> Self {
		Self {
			time: 2.0,
			list_size: 0,
			segment_filename: None,
			segment_type: SegmentType::MpegTs,
			init_filename: "init.mp4".into(),
			playlist_type: None,
			flags: HlsFlags::default(),
			base_url: String::new(),
			start_number: 0,
			key_info_file: None,
			enc: false,
			enc_key: None,
			enc_key_url: None,
			enc_iv: None,
			enc_method: EncryptionMethod::Aes128,
			master_name: None,
			subtitle_path: None,
			ts: TsOptions::default(),
		}
	}
}

/// 32 hex digits (an optional `0x` first) → 16 bytes.
pub fn parse_hex16(value: &str) -> Option<[u8; 16]> {
	let v = value.trim();
	let v = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")).unwrap_or(v);
	if v.len() != 32 {
		return None;
	}
	let mut out = [0u8; 16];
	for (i, b) in out.iter_mut().enumerate() {
		*b = u8::from_str_radix(v.get(2 * i..2 * i + 2)?, 16).ok()?;
	}
	Some(out)
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every option name [`HlsOptions::set`] knows, for the command line.
pub const HLS_OPTIONS: [&str; 17] = [
	"hls_time",
	"hls_list_size",
	"hls_segment_filename",
	"hls_segment_type",
	"hls_fmp4_init_filename",
	"hls_playlist_type",
	"hls_flags",
	"hls_base_url",
	"start_number",
	"hls_key_info_file",
	"hls_enc",
	"hls_enc_key",
	"hls_enc_key_url",
	"hls_enc_iv",
	"hls_enc_method",
	"master_pl_name",
	"hls_subtitle_path",
];

impl HlsOptions {
	/// Sets an option by its FFmpeg name; `Ok(false)` for unknown names.
	pub fn set(&mut self, key: &str, value: &str) -> Result<bool> {
		let v = value.trim();
		let bad = |what: &'static str| Message::Container(what);
		match key {
			"hls_time" => {
				self.time = v
					.parse::<f64>()
					.ok()
					.filter(|t| t.is_finite() && *t > 0.0 && *t <= 86_400.0)
					.ok_or(bad("-hls_time must be a positive number of seconds"))?;
			}
			"hls_list_size" => {
				self.list_size = v.parse().map_err(|_| bad("-hls_list_size must be a whole number"))?
			}
			"hls_segment_filename" => self.segment_filename = Some(v.to_string()),
			"hls_segment_type" => {
				self.segment_type = match v {
					"mpegts" => SegmentType::MpegTs,
					"fmp4" => SegmentType::Fmp4,
					_ => return Err(bad("-hls_segment_type is mpegts or fmp4")),
				}
			}
			"hls_fmp4_init_filename" => self.init_filename = v.to_string(),
			"hls_playlist_type" => {
				self.playlist_type = Some(match v {
					"event" => PlaylistType::Event,
					"vod" => PlaylistType::Vod,
					_ => return Err(bad("-hls_playlist_type is event or vod")),
				})
			}
			"hls_flags" => {
				// FFmpeg's flag syntax: `a+b` replaces, `+a` / `-a` edit.
				if !v.starts_with(['+', '-']) {
					self.flags = HlsFlags::default();
				}
				let mut rest = v;
				while !rest.is_empty() {
					let (on, body) = match rest.as_bytes()[0] {
						b'+' => (true, &rest[1..]),
						b'-' => (false, &rest[1..]),
						_ => (true, rest),
					};
					let end = body.find(['+', '-']).unwrap_or(body.len());
					let flag = match &body[..end] {
						"delete_segments" => &mut self.flags.delete_segments,
						"independent_segments" => &mut self.flags.independent_segments,
						"omit_endlist" => &mut self.flags.omit_endlist,
						"split_by_time" => &mut self.flags.split_by_time,
						// RMP always writes playlists through a temporary file.
						"temp_file" => {
							rest = &body[end..];
							continue;
						}
						_ => {
							return Err(bad(
								"unknown -hls_flags flag (delete_segments, independent_segments, omit_endlist, split_by_time, temp_file)",
							));
						}
					};
					*flag = on;
					rest = &body[end..];
				}
			}
			"hls_base_url" => self.base_url = v.to_string(),
			"start_number" => {
				self.start_number = v.parse().map_err(|_| bad("-start_number must be a whole number"))?
			}
			"hls_key_info_file" => self.key_info_file = Some(PathBuf::from(v)),
			"hls_enc" => self.enc = matches!(v, "1" | "true" | "on"),
			"hls_enc_key" => {
				self.enc_key = Some(parse_hex16(v).ok_or(bad("-hls_enc_key must be 32 hex digits"))?)
			}
			"hls_enc_key_url" => self.enc_key_url = Some(v.to_string()),
			"hls_enc_iv" => {
				self.enc_iv = Some(parse_hex16(v).ok_or(bad("-hls_enc_iv must be 32 hex digits"))?)
			}
			"hls_enc_method" => {
				self.enc_method = match v.to_ascii_lowercase().as_str() {
					"aes-128" | "aes128" => EncryptionMethod::Aes128,
					"sample-aes" | "sample_aes" => EncryptionMethod::SampleAes,
					_ => return Err(bad("-hls_enc_method is aes-128 or sample-aes")),
				}
			}
			"master_pl_name" => self.master_name = Some(v.to_string()),
			"hls_subtitle_path" => self.subtitle_path = Some(v.to_string()),
			_ => return self.ts.set(key, value),
		}
		Ok(true)
	}

	fn encrypted(&self) -> bool {
		self.enc || self.key_info_file.is_some()
	}
}

/// The key in force for a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Key {
	uri: String,
	key: [u8; 16],
	/// Written in the playlist; `None`: the IV is the sequence number.
	iv: Option<[u8; 16]>,
}

impl Key {
	fn iv_for(&self, sequence: u64) -> [u8; 16] {
		self.iv.unwrap_or_else(|| u128::from(sequence).to_be_bytes())
	}
}

/// 16 random bytes: the operating system's generator where there is one,
/// else the randomly seeded hashers of the standard library.
fn random_key() -> [u8; 16] {
	use std::io::Read;
	let mut key = [0u8; 16];
	if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut key)).is_ok() {
		return key;
	}
	use std::hash::{BuildHasher, Hasher};
	for (i, chunk) in key.chunks_mut(8).enumerate() {
		let mut h = std::collections::hash_map::RandomState::new().build_hasher();
		h.write_usize(i);
		h.write_u128(
			std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.map_or(0, |d| d.as_nanos()),
		);
		chunk.copy_from_slice(&h.finish().to_le_bytes());
	}
	key
}

#[derive(Debug, Clone)]
struct SegmentEntry {
	uri: String,
	path: PathBuf,
	duration: f64,
	sequence: u64,
	bytes: u64,
	key: Option<Key>,
}

/// One WebVTT rendition.
struct SubtitleTrack {
	id: StreamId,
	language: Option<String>,
	name: String,
	default: bool,
	playlist: String,
	pattern: String,
	/// Cues of the open segment: start and end (s, input clock), text.
	cues: Vec<(f64, f64, String)>,
	segments: Vec<SegmentEntry>,
}

enum Inner {
	Ts(Box<TsMuxer>),
	Fmp4(Box<Mp4Muxer>),
}

impl Inner {
	fn muxer(&mut self) -> &mut dyn Muxer {
		match self {
			Inner::Ts(m) => m.as_mut(),
			Inner::Fmp4(m) => m.as_mut(),
		}
	}
}

pub struct HlsMuxer {
	dir: PathBuf,
	playlist_name: String,
	stem: String,
	options: HlsOptions,
	metadata: Metadata,
	streams: Vec<Stream>,
	subtitles: Vec<SubtitleTrack>,
	sink: Sink,
	inner: Option<Inner>,
	started: bool,
	reference: Option<StreamId>,
	/// Input time (s) mapped to output time zero, and the start of the open
	/// segment on the reference stream's clock.
	origin: Option<f64>,
	segment_start: Option<f64>,
	/// First presentation time of the reference stream.
	first_start: Option<f64>,
	/// Segments cut so far (the next cut is at `cuts + 1` × `hls_time`).
	cuts: u64,
	/// Latest end time (pts + duration) of the reference stream.
	end: f64,
	sequence: u64,
	segments: Vec<SegmentEntry>,
	/// Segments dropped from the window (media sequence of the first one).
	media_sequence: u64,
	init_uri: Option<String>,
	key: Option<Key>,
	/// Packets written into the open segment.
	pending: bool,
}

impl HlsMuxer {
	/// A muxer writing the media playlist at `path` and everything else
	/// next to it.
	pub fn create(path: impl AsRef<Path>) -> Result<Self> {
		let path = path.as_ref();
		let playlist_name = path
			.file_name()
			.and_then(|n| n.to_str())
			.ok_or(Message::Container("hls: output needs a file name"))?
			.to_string();
		let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("stream").to_string();
		let dir = path.parent().map_or_else(PathBuf::new, Path::to_path_buf);
		Ok(Self {
			dir,
			playlist_name,
			stem,
			options: HlsOptions::default(),
			metadata: Metadata::default(),
			streams: Vec::new(),
			subtitles: Vec::new(),
			sink: Sink::default(),
			inner: None,
			started: false,
			reference: None,
			origin: None,
			segment_start: None,
			first_start: None,
			cuts: 0,
			end: 0.0,
			sequence: 0,
			segments: Vec::new(),
			media_sequence: 0,
			init_uri: None,
			key: None,
			pending: false,
		})
	}

	pub fn options(&self) -> &HlsOptions {
		&self.options
	}

	fn segment_extension(&self) -> &'static str {
		match self.options.segment_type {
			SegmentType::MpegTs => "ts",
			SegmentType::Fmp4 => "m4s",
		}
	}

	/// The path for a file name or pattern relative to the playlist, and
	/// the URI the playlist lists it by.
	fn place(&self, name: &str) -> (PathBuf, String) {
		let path = Path::new(name);
		if path.is_absolute() {
			let uri = path
				.strip_prefix(&self.dir)
				.ok()
				.and_then(|p| p.to_str())
				.map(str::to_string)
				.or_else(|| path.file_name().and_then(|n| n.to_str()).map(str::to_string))
				.unwrap_or_default();
			(path.to_path_buf(), uri.replace('\\', "/"))
		} else {
			(self.dir.join(path), name.replace('\\', "/"))
		}
	}

	fn start(&mut self) -> Result<()> {
		self.started = true;
		self.sequence = self.options.start_number;
		self.media_sequence = self.sequence;
		if self.options.playlist_type == Some(PlaylistType::Vod) {
			self.options.list_size = 0;
		}
		let sample_aes =
			self.options.encrypted() && self.options.enc_method == EncryptionMethod::SampleAes;
		if sample_aes && self.options.segment_type == SegmentType::Fmp4 {
			return Err(Message::Container(
				"hls: SAMPLE-AES needs MPEG-TS segments (fMP4 would need CENC cbcs)",
			));
		}
		let mut inner = match self.options.segment_type {
			SegmentType::MpegTs => {
				let mut ts = TsMuxer::with_options(Box::new(self.sink.clone()), self.options.ts.clone());
				ts.set_sample_aes(sample_aes)?;
				Inner::Ts(Box::new(ts))
			}
			SegmentType::Fmp4 => {
				let mut o = Mp4Options::default();
				// One fragment per segment, cut by this muxer; no mfra.
				o.set("movflags", "empty_moov+default_base_moof+skip_trailer")?;
				o.frag_duration_us = Some(u64::MAX);
				Inner::Fmp4(Box::new(Mp4Muxer::with_options(Box::new(self.sink.clone()), o)))
			}
		};
		inner.muxer().set_metadata(self.metadata.clone())?;
		for s in &self.streams {
			inner.muxer().add(s)?;
		}
		self.inner = Some(inner);
		self.reference = self
			.streams
			.iter()
			.find(|s| s.kind == StreamKind::Video)
			.or_else(|| self.streams.first())
			.map(|s| s.id);
		let (single, many) = (self.subtitles.len() == 1, self.subtitles.len() > 1);
		for (i, track) in self.subtitles.iter_mut().enumerate() {
			let suffix = if many { format!("{i}") } else { String::new() };
			track.playlist = match (&self.options.subtitle_path, single || i == 0) {
				(Some(p), true) => p.clone(),
				_ => format!("{}_vtt{suffix}.m3u8", self.stem),
			};
			// FFmpeg's names for one rendition: out_vtt.m3u8, out0.vtt, ...
			track.pattern = if many {
				format!("{}_vtt{suffix}_%d.vtt", self.stem)
			} else {
				format!("{}%d.vtt", self.stem)
			};
		}
		self.key = self.load_key()?;
		self.begin_segment();
		Ok(())
	}

	/// The key for the next segment: re-read from `hls_key_info_file` each
	/// time (so keys can rotate), else the one `hls_enc` set up.
	fn load_key(&mut self) -> Result<Option<Key>> {
		if let Some(info) = &self.options.key_info_file {
			// FFmpeg's key info file: key URI, key file path, optional IV.
			let text = std::fs::read_to_string(info)
				.map_err(|_| Message::Container("hls: cannot read -hls_key_info_file"))?;
			let mut lines = text.lines().map(str::trim);
			let uri = lines.next().filter(|l| !l.is_empty());
			let file = lines.next().filter(|l| !l.is_empty());
			let (Some(uri), Some(file)) = (uri, file) else {
				return Err(Message::Container("hls: key info file needs a key URI and a key file"));
			};
			let key_path = info.parent().map_or_else(|| PathBuf::from(file), |d| d.join(file));
			let bytes = std::fs::read(&key_path)
				.map_err(|_| Message::Container("hls: cannot read the key file"))?;
			let key: [u8; 16] =
				bytes.try_into().map_err(|_| Message::Container("hls: the key file must hold 16 bytes"))?;
			let iv = match lines.next().filter(|l| !l.is_empty()) {
				Some(l) => {
					Some(parse_hex16(l).ok_or(Message::Container("hls: invalid IV in the key info file"))?)
				}
				None => self.options.enc_iv,
			};
			return Ok(Some(Key { uri: uri.to_string(), key, iv }));
		}
		if !self.options.enc {
			return Ok(None);
		}
		if let Some(k) = &self.key {
			return Ok(Some(k.clone()));
		}
		let key = self.options.enc_key.unwrap_or_else(random_key);
		let name = format!("{}.key", self.stem);
		std::fs::write(self.dir.join(&name), key)?;
		let uri = format!("{}{name}", self.options.enc_key_url.as_deref().unwrap_or(""));
		Ok(Some(Key { uri, key, iv: self.options.enc_iv }))
	}

	fn begin_segment(&mut self) {
		let sequence = self.sequence;
		if let Some(Inner::Ts(ts)) = self.inner.as_mut() {
			ts.resend_headers();
			let cipher = match (&self.key, self.options.enc_method) {
				(Some(k), EncryptionMethod::SampleAes) => {
					Some(SampleCipher::new(&k.key, &k.iv_for(sequence)))
				}
				_ => None,
			};
			ts.set_cipher(cipher);
		}
	}

	fn segment_name(pattern: &str, sequence: u64) -> String {
		// `%d` or `%0Nd`: the sequence number.
		if let Some(at) = pattern.find('%') {
			let rest = &pattern[at + 1..];
			let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
			if rest[digits.len()..].starts_with('d') {
				let width = digits.parse::<usize>().unwrap_or(0);
				return format!(
					"{}{:0width$}{}",
					&pattern[..at],
					sequence,
					&rest[digits.len() + 1..],
					width = width
				);
			}
		}
		format!("{pattern}{sequence}")
	}

	/// Ends the open segment at `at` seconds (input clock): writes its
	/// media and subtitle files and the playlists.
	fn cut(&mut self, at: f64) -> Result<()> {
		let start = self.segment_start.unwrap_or(at);
		let duration = (at - start).max(0.0);
		match self.inner.as_mut() {
			Some(Inner::Ts(ts)) => ts.flush()?,
			Some(Inner::Fmp4(mp4)) => mp4.cut_fragment()?,
			None => {}
		}
		let mut data = self.sink.take();
		if self.options.segment_type == SegmentType::Fmp4 && self.init_uri.is_none() {
			let split = first_box(&data, b"moof").unwrap_or(data.len());
			let init: Vec<u8> = data.drain(..split).collect();
			let (path, uri) = self.place(&self.options.init_filename.clone());
			std::fs::write(&path, init)?;
			self.init_uri = Some(format!("{}{uri}", self.options.base_url));
		}
		let sequence = self.sequence;
		let pattern = self
			.options
			.segment_filename
			.clone()
			.unwrap_or_else(|| format!("{}%d.{}", self.stem, self.segment_extension()));
		let (path, uri) = self.place(&Self::segment_name(&pattern, sequence));
		let key = self.key.clone();
		if let Some(k) = &key
			&& self.options.enc_method == EncryptionMethod::Aes128
		{
			data = encrypt_cbc(&k.key, &k.iv_for(sequence), &data);
		}
		std::fs::write(&path, &data)?;
		let entry = SegmentEntry {
			uri: format!("{}{uri}", self.options.base_url),
			path,
			duration,
			sequence,
			bytes: data.len() as u64,
			key,
		};
		self.segments.push(entry);

		// Subtitle segments mirror the media segments.
		// Cue time zero is the media's: the TS muxer's origin at its start
		// offset, or (fMP4) the first packet at time zero.
		let (ts_offset, origin) = match self.inner.as_ref() {
			Some(Inner::Ts(ts)) => (ts.start_offset(), ts.origin().or(self.origin).unwrap_or(0.0)),
			_ => (0, self.origin.unwrap_or(0.0)),
		};
		let ms = |s: f64| ((s - origin).max(0.0) * 1000.0).round() as u64;
		for i in 0..self.subtitles.len() {
			let name = Self::segment_name(&self.subtitles[i].pattern, sequence);
			let (path, uri) = self.place(&name);
			let track = &mut self.subtitles[i];
			let mut text = webvtt::header(Some(ts_offset));
			for (start, end, cue) in std::mem::take(&mut track.cues) {
				webvtt::write_cue(&mut text, ms(start), ms(end), &cue);
			}
			std::fs::write(&path, text.as_bytes())?;
			track.segments.push(SegmentEntry {
				uri: format!("{}{uri}", self.options.base_url),
				path,
				duration,
				sequence,
				bytes: text.len() as u64,
				key: None,
			});
		}
		self.slide_window()?;
		self.sequence += 1;
		self.cuts += 1;
		self.segment_start = Some(at);
		self.pending = false;
		self.write_playlists(false)?;
		self.key = self.load_key()?;
		self.begin_segment();
		Ok(())
	}

	/// Keeps the last `hls_list_size` segments listed.
	fn slide_window(&mut self) -> Result<()> {
		let size = self.options.list_size;
		if size == 0 {
			return Ok(());
		}
		let delete = self.options.flags.delete_segments;
		let drop = |list: &mut Vec<SegmentEntry>| -> Result<()> {
			while list.len() > size {
				let old = list.remove(0);
				if delete {
					let _ = std::fs::remove_file(&old.path);
				}
			}
			Ok(())
		};
		drop(&mut self.segments)?;
		for t in &mut self.subtitles {
			drop(&mut t.segments)?;
		}
		self.media_sequence = self.segments.first().map_or(self.sequence, |s| s.sequence);
		Ok(())
	}

	fn version(&self) -> u32 {
		let mut v = 3;
		if self.key.is_some() && self.options.enc_method == EncryptionMethod::SampleAes {
			v = 5;
		}
		if self.options.segment_type == SegmentType::Fmp4 {
			v = 7;
		}
		v
	}

	fn media_playlist(&self, segments: &[SegmentEntry], init: Option<&str>, ended: bool) -> String {
		let target = segments.iter().map(|s| ceil_duration(s.duration)).max().unwrap_or(1).max(1);
		let mut out = format!("#EXTM3U\n#EXT-X-VERSION:{}\n", self.version());
		out += &format!("#EXT-X-TARGETDURATION:{target}\n");
		out += &format!(
			"#EXT-X-MEDIA-SEQUENCE:{}\n",
			segments.first().map_or(self.media_sequence, |s| s.sequence)
		);
		match self.options.playlist_type {
			Some(PlaylistType::Vod) => out += "#EXT-X-PLAYLIST-TYPE:VOD\n",
			Some(PlaylistType::Event) => out += "#EXT-X-PLAYLIST-TYPE:EVENT\n",
			None => {}
		}
		if self.options.flags.independent_segments {
			out += "#EXT-X-INDEPENDENT-SEGMENTS\n";
		}
		if let Some(init) = init {
			out += &format!("#EXT-X-MAP:URI=\"{init}\"\n");
		}
		let method = match self.options.enc_method {
			EncryptionMethod::Aes128 => "AES-128",
			EncryptionMethod::SampleAes => "SAMPLE-AES",
		};
		let mut key: Option<&Key> = None;
		for s in segments {
			if s.key.as_ref() != key {
				match &s.key {
					Some(k) => {
						out += &format!("#EXT-X-KEY:METHOD={method},URI=\"{}\"", k.uri);
						if let Some(iv) = k.iv {
							out += &format!(",IV=0x{}", hex(&iv));
						}
						out += "\n";
					}
					None => out += "#EXT-X-KEY:METHOD=NONE\n",
				}
				key = s.key.as_ref();
			}
			out += &format!("#EXTINF:{:.6},\n{}\n", s.duration, s.uri);
		}
		if ended && !self.options.flags.omit_endlist {
			out += "#EXT-X-ENDLIST\n";
		}
		out
	}

	fn write_playlists(&mut self, ended: bool) -> Result<()> {
		let media = self.media_playlist(&self.segments, self.init_uri.as_deref(), ended);
		write_atomically(&self.dir.join(&self.playlist_name), &media)?;
		for track in &self.subtitles {
			let mut text = self.media_playlist(&track.segments, None, ended);
			// Subtitle segments are never encrypted.
			text =
				text.lines().filter(|l| !l.starts_with("#EXT-X-KEY")).collect::<Vec<_>>().join("\n") + "\n";
			text = text.replace(&format!("#EXT-X-VERSION:{}", self.version()), "#EXT-X-VERSION:3");
			write_atomically(&self.dir.join(&track.playlist), &text)?;
		}
		let master = self
			.options
			.master_name
			.clone()
			.or_else(|| (!self.subtitles.is_empty()).then(|| format!("{}_master.m3u8", self.stem)));
		if let Some(name) = master {
			let text = self.master_playlist();
			write_atomically(&self.dir.join(name), &text)?;
		}
		Ok(())
	}

	fn master_playlist(&self) -> String {
		let mut out = format!("#EXTM3U\n#EXT-X-VERSION:{}\n", self.version());
		if self.options.flags.independent_segments {
			out += "#EXT-X-INDEPENDENT-SEGMENTS\n";
		}
		for t in &self.subtitles {
			out += &format!("#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs\",NAME=\"{}\"", quote(&t.name));
			out += &format!(",DEFAULT={},AUTOSELECT=YES", if t.default { "YES" } else { "NO" });
			if let Some(lang) = &t.language {
				out += &format!(",LANGUAGE=\"{}\"", quote(lang));
			}
			out += &format!(",URI=\"{}{}\"\n", self.options.base_url, t.playlist);
		}
		let (total_bytes, total_time) =
			self.segments.iter().fold((0u64, 0f64), |(b, t), s| (b + s.bytes, t + s.duration));
		let peak = self
			.segments
			.iter()
			.filter(|s| s.duration > 0.0)
			.map(|s| (s.bytes as f64 * 8.0 / s.duration) as u64)
			.max()
			.unwrap_or(0);
		let average = if total_time > 0.0 { (total_bytes as f64 * 8.0 / total_time) as u64 } else { 0 };
		out += &format!("#EXT-X-STREAM-INF:BANDWIDTH={}", peak.max(average).max(1));
		if average > 0 {
			out += &format!(",AVERAGE-BANDWIDTH={average}");
		}
		if let Some(v) = self.streams.iter().find(|s| s.kind == StreamKind::Video)
			&& let (Some(w), Some(h)) = (v.parameters.width, v.parameters.height)
		{
			out += &format!(",RESOLUTION={w}x{h}");
		}
		let codecs: Option<Vec<String>> = self.streams.iter().map(codec_string).collect();
		if let Some(codecs) = codecs.filter(|c| !c.is_empty()) {
			out += &format!(",CODECS=\"{}\"", codecs.join(","));
		}
		if !self.subtitles.is_empty() {
			out += ",SUBTITLES=\"subs\"";
		}
		out += &format!("\n{}{}\n", self.options.base_url, self.playlist_name);
		out
	}

	fn seconds(time: Option<crate::core::Time>) -> Option<f64> {
		time.map(|t| t.as_seconds())
	}
}

/// Quoted-string attribute values can't hold `"` or line breaks.
fn quote(s: &str) -> String {
	s.chars().filter(|c| !matches!(c, '"' | '\n' | '\r')).collect()
}

/// `EXT-X-TARGETDURATION` rounds up, ignoring the last millisecond as
/// FFmpeg's `get_int_from_double` does.
fn ceil_duration(d: f64) -> u64 {
	let whole = d.trunc();
	if d - whole >= 0.001 { whole as u64 + 1 } else { whole as u64 }
}

/// Offset of the first top-level box of `kind` in `data`.
fn first_box(data: &[u8], kind: &[u8; 4]) -> Option<usize> {
	let mut at = 0usize;
	while at + 8 <= data.len() {
		let size = u32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
		if &data[at + 4..at + 8] == kind {
			return Some(at);
		}
		if size < 8 {
			return None;
		}
		at = at.checked_add(size)?;
	}
	None
}

/// Writes through a temporary file and a rename, so a reader never sees
/// half a playlist.
fn write_atomically(path: &Path, text: &str) -> Result<()> {
	let mut tmp = path.as_os_str().to_owned();
	tmp.push(".tmp");
	let tmp = PathBuf::from(tmp);
	std::fs::write(&tmp, text)?;
	std::fs::rename(&tmp, path)?;
	Ok(())
}

/// RFC 6381 codec string for the `CODECS` attribute, when RMP can tell.
pub fn codec_string(stream: &Stream) -> Option<String> {
	let p = &stream.parameters;
	match p.codec.id.as_str() {
		"h264" => {
			let sps: Vec<u8> = match crate::codecs::h264::nal::AvcConfig::parse(&p.extradata) {
				Some(c) => c.sps.first()?.clone(),
				None => crate::codecs::h264::nal::split_annex_b(&p.extradata)
					.into_iter()
					.find(|n| n[0] & 0x1F == 7)?
					.to_vec(),
			};
			Some(format!("avc1.{}", hex(sps.get(1..4)?)))
		}
		"hevc" => {
			let hvcc = if p.extradata.first() == Some(&1) {
				p.extradata.clone()
			} else {
				crate::container::bsf::hvcc_from_annex_b(&p.extradata).ok()?
			};
			let g = hvcc.get(1..13)?;
			let space = ["", "A", "B", "C"][usize::from(g[0] >> 6)];
			let tier = if g[0] & 0x20 != 0 { 'H' } else { 'L' };
			let profile = g[0] & 0x1F;
			let compat = u32::from_be_bytes([g[1], g[2], g[3], g[4]]).reverse_bits();
			let mut constraints: Vec<u8> = g[5..11].to_vec();
			while constraints.last() == Some(&0) {
				constraints.pop();
			}
			let mut s = format!("hvc1.{space}{profile}.{compat:X}.{tier}{}", g[11]);
			for c in constraints {
				s += &format!(".{c:X}");
			}
			Some(s)
		}
		"aac" => {
			let aot = crate::codecs::aac::config::parse_audio_specific_config(&p.extradata)
				.map(|c| if c.sbr == 1 { if c.ps == 1 { 29 } else { 5 } } else { c.object_type })
				.unwrap_or(2);
			Some(format!("mp4a.40.{aot}"))
		}
		"mp3" => Some("mp4a.40.34".into()),
		"mp2" => Some("mp4a.40.33".into()),
		"ac3" => Some("ac-3".into()),
		"eac3" => Some("ec-3".into()),
		"opus" => Some("Opus".into()),
		_ => None,
	}
}

impl Muxer for HlsMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn set_option(&mut self, key: &str, value: &str) -> Result<()> {
		if self.started {
			return Err(Message::Container("hls muxer: options must be set before writing"));
		}
		if !self.options.set(key, value)? {
			return Err(Message::Container("hls muxer: unknown option"));
		}
		Ok(())
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.started {
			return Err(Message::Container("hls muxer: streams must be added before writing"));
		}
		if self.streams.iter().any(|s| s.id == stream.id)
			|| self.subtitles.iter().any(|s| s.id == stream.id)
		{
			return Err(Message::Container("hls muxer: duplicate stream id"));
		}
		match stream.kind {
			StreamKind::Audio | StreamKind::Video => self.streams.push(stream.clone()),
			StreamKind::Subtitle if webvtt::holds(&stream.parameters.codec.id) => {
				let language = match stream.metadata.raw.get("language") {
					Some(RawValue::String(l)) if !l.is_empty() && l != "und" => Some(l.clone()),
					_ => None,
				};
				let n = self.subtitles.len();
				let name = stream
					.metadata
					.title
					.clone()
					.filter(|t| !t.is_empty())
					.or_else(|| language.clone())
					.unwrap_or_else(|| format!("subtitles {}", n + 1));
				self.subtitles.push(SubtitleTrack {
					id: stream.id,
					language,
					name,
					default: n == 0,
					playlist: String::new(),
					pattern: String::new(),
					cues: Vec::new(),
					segments: Vec::new(),
				});
			}
			_ => return Err(Message::Container("hls muxer: subtitles must be webvtt or subrip")),
		}
		Ok(self.streams.len() + self.subtitles.len() - 1)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if !self.started {
			self.start()?;
		}
		let time = Self::seconds(packet.dts.or(packet.pts));
		if self.origin.is_none()
			&& let Some(t) = time
		{
			self.origin = Some(t);
		}
		if let Some(i) = self.subtitles.iter().position(|t| t.id == packet.stream_id) {
			let track = &mut self.subtitles[i];
			let Some(start) = Self::seconds(packet.pts.or(packet.dts)) else { return Ok(()) };
			let length = packet.duration.map_or(0.0, |d| d.as_seconds());
			track.cues.push((start, start + length, String::from_utf8_lossy(&packet.data).into_owned()));
			return Ok(());
		}
		if !self.streams.iter().any(|s| s.id == packet.stream_id) {
			return Err(Message::Container("hls muxer: packet for unknown stream"));
		}
		let pts = Self::seconds(packet.pts.or(packet.dts));
		if Some(packet.stream_id) == self.reference
			&& let Some(pts) = pts
		{
			self.segment_start.get_or_insert(pts);
			let first = *self.first_start.get_or_insert(pts);
			let splits = packet.keyframe || self.options.flags.split_by_time;
			// FFmpeg's rule: the n-th cut at the first keyframe at least n ×
			// hls_time after the start (so durations don't drift).
			let due = pts - first >= self.options.time * (self.cuts + 1) as f64 - 1e-9;
			if splits && due && self.pending {
				self.cut(pts)?;
			}
		}
		// The last segment ends with the reference stream, as in FFmpeg.
		if let Some(pts) = pts
			&& (Some(packet.stream_id) == self.reference || self.reference.is_none())
		{
			let end = pts + packet.duration.map_or(0.0, |d| d.as_seconds());
			self.end = self.end.max(end);
		}
		self.pending = true;
		match self.inner.as_mut() {
			Some(inner) => inner.muxer().write(packet),
			None => Err(Message::Container("hls muxer: not started")),
		}
	}

	fn finish(&mut self) -> Result<()> {
		if !self.started {
			self.start()?;
		}
		if self.pending || self.segments.is_empty() {
			match self.inner.as_mut() {
				Some(Inner::Ts(ts)) => ts.finish()?,
				Some(Inner::Fmp4(mp4)) => mp4.cut_fragment()?,
				None => {}
			}
			let end = self.end.max(self.segment_start.unwrap_or(0.0));
			self.cut(end)?;
		}
		self.write_playlists(true)
	}
}
