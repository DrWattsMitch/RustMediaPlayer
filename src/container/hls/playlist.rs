//! M3U8 playlists (RFC 8216): multivariant ("master") and media playlists.
//!
//! FFmpeg: `libavformat/hls.c` (`parse_playlist`, `new_variant`,
//! `new_rendition`).

use crate::io::{ByteRange, resolve};
use crate::message::{Message, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaType {
	Audio,
	Video,
	Subtitles,
	ClosedCaptions,
}

/// `EXT-X-MEDIA`: an alternative rendition.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendition {
	pub kind: MediaType,
	pub group: String,
	pub name: String,
	pub language: Option<String>,
	pub default: bool,
	pub autoselect: bool,
	/// `None`: the rendition is muxed into the variant's own segments.
	pub uri: Option<String>,
	pub channels: Option<String>,
}

/// `EXT-X-STREAM-INF`: one variant stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
	pub uri: String,
	pub bandwidth: u64,
	pub average_bandwidth: Option<u64>,
	pub codecs: Option<String>,
	pub resolution: Option<(u32, u32)>,
	pub frame_rate: Option<f64>,
	pub audio: Option<String>,
	pub video: Option<String>,
	pub subtitles: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MultivariantPlaylist {
	pub variants: Vec<Variant>,
	pub renditions: Vec<Rendition>,
}

/// `EXT-X-KEY` in force for a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
	None,
	/// AES-128 CBC over the whole segment; `iv` defaults to the media
	/// sequence number.
	Aes128 {
		uri: String,
		iv: Option<[u8; 16]>,
	},
	/// SAMPLE-AES (identity key format): samples encrypted inside MPEG-TS or
	/// packed-audio segments.
	SampleAes {
		uri: String,
		iv: Option<[u8; 16]>,
	},
	/// DRM key formats and methods RMP can't decrypt.
	Unsupported(String),
}

/// `EXT-X-MAP`: the initialization section (fMP4 `moov`, or TS PAT/PMT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Map {
	pub uri: String,
	pub range: Option<ByteRange>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
	pub uri: String,
	pub duration: f64,
	pub range: Option<ByteRange>,
	pub sequence: u64,
	pub discontinuity: bool,
	pub map: Option<Map>,
	pub key: Key,
	/// `EXT-X-GAP`: no media; skipped.
	pub gap: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MediaPlaylist {
	pub target_duration: f64,
	pub media_sequence: u64,
	pub segments: Vec<Segment>,
	/// `EXT-X-ENDLIST`, or `EXT-X-PLAYLIST-TYPE:VOD`: no more segments will
	/// be added.
	pub ended: bool,
}

impl MediaPlaylist {
	pub fn duration(&self) -> f64 {
		self.segments.iter().map(|s| s.duration).sum()
	}
}

#[derive(Debug, Clone, PartialEq)]
pub enum Playlist {
	Multivariant(MultivariantPlaylist),
	Media(MediaPlaylist),
}

/// `true` when `data` begins like an M3U8 playlist.
pub fn looks_like_m3u8(data: &[u8]) -> bool {
	let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(data);
	let start = data.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(data.len());
	data[start..].starts_with(b"#EXTM3U")
}

/// Splits an attribute list (`KEY=value,KEY="quoted, value"`).
pub fn attributes(list: &str) -> Vec<(&str, &str)> {
	let mut out = Vec::new();
	let mut rest = list.trim();
	while !rest.is_empty() {
		let Some(eq) = rest.find('=') else { break };
		let key = rest[..eq].trim();
		let after = &rest[eq + 1..];
		let (value, next) = if let Some(quoted) = after.strip_prefix('"') {
			match quoted.find('"') {
				Some(end) => (&quoted[..end], &quoted[end + 1..]),
				None => (quoted, ""),
			}
		} else {
			match after.find(',') {
				Some(end) => (&after[..end], &after[end..]),
				None => (after, ""),
			}
		};
		out.push((key, value.trim()));
		rest = next.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
	}
	out
}

fn attr<'a>(list: &[(&str, &'a str)], key: &str) -> Option<&'a str> {
	list.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// `n[@o]` from `EXT-X-BYTERANGE` or a `BYTERANGE` attribute. Without an
/// offset the range continues the previous one.
fn byte_range(value: &str, previous_end: Option<u64>) -> Option<ByteRange> {
	let (len, offset) = match value.split_once('@') {
		Some((l, o)) => (l, Some(o.trim().parse().ok()?)),
		None => (value, None),
	};
	let length: u64 = len.trim().parse().ok()?;
	Some(ByteRange { start: offset.or(previous_end).unwrap_or(0), length: Some(length) })
}

fn parse_iv(value: &str) -> Option<[u8; 16]> {
	let hex = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X"))?;
	if hex.len() > 32 || hex.is_empty() {
		return None;
	}
	let padded = format!("{hex:0>32}");
	let mut iv = [0u8; 16];
	for (i, b) in iv.iter_mut().enumerate() {
		*b = u8::from_str_radix(&padded[2 * i..2 * i + 2], 16).ok()?;
	}
	Some(iv)
}

/// Parses a playlist; URIs are resolved against `base` (the playlist's own
/// location).
pub fn parse(text: &str, base: &str) -> Result<Playlist> {
	let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
	let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
	if lines.next() != Some("#EXTM3U") {
		return Err(Message::Container("hls: playlist does not start with #EXTM3U"));
	}
	let mut master = MultivariantPlaylist::default();
	let mut media = MediaPlaylist::default();
	let mut is_media = false;

	let mut pending_variant: Option<Variant> = None;
	let mut duration: Option<f64> = None;
	// The range, and whether it gave its own offset.
	let mut range: Option<(ByteRange, bool)> = None;
	let mut discontinuity = false;
	let mut gap = false;
	let mut map: Option<Map> = None;
	let mut key = Key::None;
	// Where the last byte range ended, per URI (for ranges without offset).
	let mut last_range: Option<(String, u64)> = None;
	let mut sequence_set = false;

	for line in lines {
		if let Some(tag) = line.strip_prefix('#') {
			let (name, value) = tag.split_once(':').unwrap_or((tag, ""));
			match name {
				"EXT-X-STREAM-INF" => {
					let a = attributes(value);
					let resolution = attr(&a, "RESOLUTION").and_then(|r| {
						let (w, h) = r.split_once(['x', 'X'])?;
						Some((w.parse().ok()?, h.parse().ok()?))
					});
					pending_variant = Some(Variant {
						uri: String::new(),
						bandwidth: attr(&a, "BANDWIDTH").and_then(|v| v.parse().ok()).unwrap_or(0),
						average_bandwidth: attr(&a, "AVERAGE-BANDWIDTH").and_then(|v| v.parse().ok()),
						codecs: attr(&a, "CODECS").map(str::to_string),
						resolution,
						frame_rate: attr(&a, "FRAME-RATE").and_then(|v| v.parse().ok()),
						audio: attr(&a, "AUDIO").map(str::to_string),
						video: attr(&a, "VIDEO").map(str::to_string),
						subtitles: attr(&a, "SUBTITLES").map(str::to_string),
					});
				}
				"EXT-X-MEDIA" => {
					let a = attributes(value);
					let kind = match attr(&a, "TYPE") {
						Some("AUDIO") => MediaType::Audio,
						Some("VIDEO") => MediaType::Video,
						Some("SUBTITLES") => MediaType::Subtitles,
						Some("CLOSED-CAPTIONS") => MediaType::ClosedCaptions,
						_ => continue,
					};
					master.renditions.push(Rendition {
						kind,
						group: attr(&a, "GROUP-ID").unwrap_or_default().to_string(),
						name: attr(&a, "NAME").unwrap_or_default().to_string(),
						language: attr(&a, "LANGUAGE").map(str::to_string),
						default: attr(&a, "DEFAULT") == Some("YES"),
						autoselect: attr(&a, "AUTOSELECT") == Some("YES"),
						uri: attr(&a, "URI").map(|u| resolve(base, u)),
						channels: attr(&a, "CHANNELS").map(str::to_string),
					});
				}
				"EXT-X-TARGETDURATION" => {
					is_media = true;
					media.target_duration = value.trim().parse().unwrap_or(0.0);
				}
				"EXT-X-MEDIA-SEQUENCE" => {
					is_media = true;
					if !sequence_set {
						media.media_sequence = value.trim().parse().unwrap_or(0);
						sequence_set = true;
					}
				}
				"EXT-X-ENDLIST" => media.ended = true,
				"EXT-X-PLAYLIST-TYPE" if value.trim() == "VOD" => media.ended = true,
				"EXTINF" => {
					is_media = true;
					let d = value.split(',').next().unwrap_or("").trim();
					duration =
						Some(d.parse().map_err(|_| Message::Container("hls: invalid #EXTINF duration"))?);
				}
				"EXT-X-BYTERANGE" => {
					// Resolved against the URI on the next line.
					range = byte_range(value, None).map(|r| (r, value.contains('@')));
				}
				"EXT-X-DISCONTINUITY" => discontinuity = true,
				"EXT-X-GAP" => gap = true,
				"EXT-X-MAP" => {
					let a = attributes(value);
					let Some(uri) = attr(&a, "URI") else { continue };
					let uri = resolve(base, uri);
					let range = attr(&a, "BYTERANGE").and_then(|r| byte_range(r, Some(0)));
					map = Some(Map { uri, range });
				}
				"EXT-X-KEY" => {
					let a = attributes(value);
					key = match attr(&a, "METHOD") {
						Some("NONE") | None => Key::None,
						Some("AES-128") => match attr(&a, "URI") {
							Some(uri) => {
								Key::Aes128 { uri: resolve(base, uri), iv: attr(&a, "IV").and_then(parse_iv) }
							}
							None => return Err(Message::Container("hls: AES-128 key without URI")),
						},
						// Other key formats are DRM systems (FairPlay, ...).
						Some("SAMPLE-AES") if attr(&a, "KEYFORMAT").is_none_or(|f| f == "identity") => {
							match attr(&a, "URI") {
								Some(uri) => {
									Key::SampleAes { uri: resolve(base, uri), iv: attr(&a, "IV").and_then(parse_iv) }
								}
								None => return Err(Message::Container("hls: SAMPLE-AES key without URI")),
							}
						}
						Some(other) => Key::Unsupported(other.to_string()),
					};
				}
				_ => {}
			}
			continue;
		}
		// A URI line.
		let uri = resolve(base, line);
		if let Some(mut variant) = pending_variant.take() {
			variant.uri = uri;
			master.variants.push(variant);
			continue;
		}
		let Some(d) = duration.take() else { continue };
		is_media = true;
		let range = range.take().map(|(r, explicit_offset)| {
			// `n` alone continues the previous range of the same resource.
			match &last_range {
				Some((u, end)) if !explicit_offset && *u == uri => ByteRange { start: *end, ..r },
				_ => r,
			}
		});
		if let Some(r) = range {
			last_range = Some((uri.clone(), r.start + r.length.unwrap_or(0)));
		}
		let sequence = media.media_sequence + media.segments.len() as u64;
		media.segments.push(Segment {
			uri,
			duration: d,
			range,
			sequence,
			discontinuity: std::mem::take(&mut discontinuity),
			map: map.clone(),
			key: key.clone(),
			gap: std::mem::take(&mut gap),
		});
	}
	if !master.variants.is_empty() {
		return Ok(Playlist::Multivariant(master));
	}
	if is_media {
		return Ok(Playlist::Media(media));
	}
	Err(Message::Container("hls: playlist has neither variants nor segments"))
}

#[cfg(test)]
mod tests {
	use super::*;

	const MASTER: &str = "#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aud\",NAME=\"English\",LANGUAGE=\"en\",DEFAULT=YES,AUTOSELECT=YES,URI=\"audio/en.m3u8\"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aud\",NAME=\"Français\",LANGUAGE=\"fr\",URI=\"audio/fr.m3u8\"
#EXT-X-STREAM-INF:BANDWIDTH=1280000,AVERAGE-BANDWIDTH=1000000,CODECS=\"avc1.4d401f,mp4a.40.2\",RESOLUTION=640x360,AUDIO=\"aud\"
low/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2560000,RESOLUTION=1280x720,FRAME-RATE=29.970,AUDIO=\"aud\"
https://cdn.example.com/hi/index.m3u8
";

	#[test]
	fn multivariant() {
		let Playlist::Multivariant(m) = parse(MASTER, "http://example.com/live/master.m3u8").unwrap()
		else {
			panic!()
		};
		assert_eq!(m.variants.len(), 2);
		assert_eq!(m.variants[0].uri, "http://example.com/live/low/index.m3u8");
		assert_eq!(m.variants[0].codecs.as_deref(), Some("avc1.4d401f,mp4a.40.2"));
		assert_eq!(m.variants[1].resolution, Some((1280, 720)));
		assert_eq!(m.variants[1].uri, "https://cdn.example.com/hi/index.m3u8");
		assert_eq!(m.renditions.len(), 2);
		assert!(m.renditions[0].default);
		assert_eq!(m.renditions[1].name, "Français");
		assert_eq!(m.renditions[1].uri.as_deref(), Some("http://example.com/live/audio/fr.m3u8"));
	}

	#[test]
	fn media_with_ranges_keys_and_maps() {
		let text = "#EXTM3U
#EXT-X-VERSION:7
#EXT-X-TARGETDURATION:4
#EXT-X-MEDIA-SEQUENCE:10
#EXT-X-MAP:URI=\"init.mp4\",BYTERANGE=\"800@0\"
#EXTINF:4.0,
#EXT-X-BYTERANGE:1000@800
all.mp4
#EXTINF:3.5,title
#EXT-X-BYTERANGE:500
all.mp4
#EXT-X-KEY:METHOD=AES-128,URI=\"k.bin\",IV=0x1
#EXT-X-DISCONTINUITY
#EXTINF:2,
seg.ts
#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"k2\"
#EXTINF:2,
seg2.ts
#EXT-X-ENDLIST
";
		let Playlist::Media(p) = parse(text, "/v/index.m3u8").unwrap() else { panic!() };
		assert!(p.ended);
		assert_eq!(p.segments.len(), 4);
		assert_eq!(p.segments[0].range, Some(ByteRange { start: 800, length: Some(1000) }));
		assert_eq!(p.segments[1].range, Some(ByteRange { start: 1800, length: Some(500) }));
		assert_eq!(p.segments[1].sequence, 11);
		assert_eq!(p.segments[0].map.as_ref().unwrap().uri, "/v/init.mp4");
		assert!(p.segments[2].discontinuity);
		let mut iv = [0u8; 16];
		iv[15] = 1;
		assert_eq!(p.segments[2].key, Key::Aes128 { uri: "/v/k.bin".into(), iv: Some(iv) });
		assert_eq!(p.segments[3].key, Key::SampleAes { uri: "/v/k2".into(), iv: None });
		assert_eq!(p.duration(), 11.5);
	}

	#[test]
	fn rejects_non_playlists() {
		assert!(parse("hello", "x").is_err());
		assert!(looks_like_m3u8(b"\xEF\xBB\xBF#EXTM3U\n"));
		assert!(!looks_like_m3u8(b"<MPD"));
	}
}
