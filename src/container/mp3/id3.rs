//! ID3 tag support for MP3 files (FFmpeg's `id3v1.c` / `id3v2.c` /
//! `id3v2enc.c`, reduced to the fields RMP's [`Metadata`] models).
//!
//! * ID3v2.2 / 2.3 / 2.4 reading, including unsynchronisation, extended
//!   headers, per-frame flags and all four text encodings.
//! * ID3v1 / v1.1 reading (only fills fields ID3v2 did not provide).
//! * ID3v2.4 writing with UTF-8 text.

use crate::core::{AttachedImage, Metadata, RawValue};
use crate::io::ByteCursor;
use crate::message::{Message, Result};

/// Size of the ID3v2 header (and of the optional footer).
pub const ID3V2_HEADER_SIZE: usize = 10;
/// Size of an ID3v1 tag at the end of the file.
pub const ID3V1_SIZE: usize = 128;
/// Tags larger than this are skipped instead of parsed (protects memory
/// against corrupt size fields; real tags with cover art are far smaller).
pub const MAX_PARSED_TAG_SIZE: usize = 64 * 1024 * 1024;
/// Largest value a 28-bit syncsafe integer can hold.
const MAX_SYNCSAFE: usize = (1 << 28) - 1;

/// Standard ID3v1 genres 0..=79 (the Winamp extensions are left numeric).
pub const ID3V1_GENRES: [&str; 80] = [
	"Blues",
	"Classic Rock",
	"Country",
	"Dance",
	"Disco",
	"Funk",
	"Grunge",
	"Hip-Hop",
	"Jazz",
	"Metal",
	"New Age",
	"Oldies",
	"Other",
	"Pop",
	"R&B",
	"Rap",
	"Reggae",
	"Rock",
	"Techno",
	"Industrial",
	"Alternative",
	"Ska",
	"Death Metal",
	"Pranks",
	"Soundtrack",
	"Euro-Techno",
	"Ambient",
	"Trip-Hop",
	"Vocal",
	"Jazz+Funk",
	"Fusion",
	"Trance",
	"Classical",
	"Instrumental",
	"Acid",
	"House",
	"Game",
	"Sound Clip",
	"Gospel",
	"Noise",
	"AlternRock",
	"Bass",
	"Soul",
	"Punk",
	"Space",
	"Meditative",
	"Instrumental Pop",
	"Instrumental Rock",
	"Ethnic",
	"Gothic",
	"Darkwave",
	"Techno-Industrial",
	"Electronic",
	"Pop-Folk",
	"Eurodance",
	"Dream",
	"Southern Rock",
	"Comedy",
	"Cult",
	"Gangsta",
	"Top 40",
	"Christian Rap",
	"Pop/Funk",
	"Jungle",
	"Native American",
	"Cabaret",
	"New Wave",
	"Psychadelic",
	"Rave",
	"Showtunes",
	"Trailer",
	"Lo-Fi",
	"Tribal",
	"Acid Punk",
	"Acid Jazz",
	"Polka",
	"Retro",
	"Musical",
	"Rock & Roll",
	"Hard Rock",
];

fn syncsafe(bytes: &[u8]) -> Option<usize> {
	let mut value = 0usize;
	for &b in bytes.get(..4)? {
		if b & 0x80 != 0 {
			return None;
		}
		value = (value << 7) | usize::from(b);
	}
	Some(value)
}

fn write_syncsafe(out: &mut Vec<u8>, value: usize) {
	out.extend_from_slice(&[
		((value >> 21) & 0x7F) as u8,
		((value >> 14) & 0x7F) as u8,
		((value >> 7) & 0x7F) as u8,
		(value & 0x7F) as u8,
	]);
}

/// If `data` starts with a valid ID3v2 header, returns the total tag size
/// (header + body + optional footer).
pub fn v2_tag_size(data: &[u8]) -> Option<usize> {
	let header = data.get(..ID3V2_HEADER_SIZE)?;
	if &header[..3] != b"ID3" || header[3] == 0xFF || header[4] == 0xFF {
		return None;
	}
	let body = syncsafe(&header[6..10])?;
	let footer = if header[3] >= 4 && header[5] & 0x10 != 0 { ID3V2_HEADER_SIZE } else { 0 };
	Some(ID3V2_HEADER_SIZE + body + footer)
}

/// Reverses ID3 unsynchronisation (`FF 00` → `FF`).
fn unsynchronise(data: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(data.len());
	let mut prev_ff = false;
	for &b in data {
		if !(prev_ff && b == 0) {
			out.push(b);
		}
		prev_ff = b == 0xFF;
	}
	out
}

/// Parses a complete ID3v2 tag (as delimited by [`v2_tag_size`]).
/// Malformed frames are skipped; a malformed header is reported.
pub fn parse_v2(tag: &[u8], metadata: &mut Metadata) -> Result<()> {
	let total = v2_tag_size(tag).ok_or(Message::Container("invalid id3v2 header"))?;
	let header = &tag[..ID3V2_HEADER_SIZE];
	let version = header[3];
	let flags = header[5];
	if !(2..=4).contains(&version) {
		return Ok(()); // unknown major version: skip silently, like FFmpeg
	}
	let end = total.min(tag.len());
	let mut body: Vec<u8> = tag[ID3V2_HEADER_SIZE..end].to_vec();
	// v2.2/2.3 apply unsynchronisation to the whole tag; v2.4 per frame.
	if flags & 0x80 != 0 && version < 4 {
		body = unsynchronise(&body);
	}

	let mut cursor = ByteCursor::new(&body);
	if flags & 0x40 != 0 && version >= 3 {
		// Extended header: v2.3 size excludes itself, v2.4 is syncsafe and
		// includes itself.
		let size = if version == 3 {
			cursor.read_u32()? as usize
		} else {
			let raw = cursor.read_array::<4>()?;
			syncsafe(&raw).ok_or(Message::Container("invalid id3v2 extended header"))?.saturating_sub(4)
		};
		cursor.skip(size)?;
	}

	let id_len = if version == 2 { 3 } else { 4 };
	let header_len = if version == 2 { 6 } else { 10 };
	while cursor.remaining() >= header_len {
		let id = cursor.read_bytes(id_len)?;
		if id[0] == 0 {
			break; // padding
		}
		let size = match version {
			2 => cursor.read_u24()? as usize,
			3 => cursor.read_u32()? as usize,
			_ => syncsafe(cursor.read_bytes(4)?).ok_or(Message::Container("invalid id3 frame size"))?,
		};
		let frame_flags = if version == 2 { 0 } else { cursor.read_u16()? };
		let Ok(data) = cursor.read_bytes(size) else { break };

		let mut data: &[u8] = data;
		let owned;
		if version == 3 {
			if frame_flags & 0x00C0 != 0 {
				continue; // compressed or encrypted
			}
			if frame_flags & 0x0020 != 0 {
				data = data.get(1..).unwrap_or(&[]); // group id
			}
		} else if version == 4 {
			if frame_flags & 0x000C != 0 {
				continue; // compressed or encrypted
			}
			if frame_flags & 0x0040 != 0 {
				data = data.get(1..).unwrap_or(&[]);
			}
			if frame_flags & 0x0001 != 0 {
				data = data.get(4..).unwrap_or(&[]); // data length indicator
			}
			if frame_flags & 0x0002 != 0 || flags & 0x80 != 0 {
				owned = unsynchronise(data);
				data = &owned;
			}
		}
		apply_frame(&normalise_id(id), data, metadata);
	}
	Ok(())
}

/// Maps ID3v2.2 three-character IDs onto their v2.3+ equivalents.
fn normalise_id(id: &[u8]) -> [u8; 4] {
	if id.len() == 4 {
		return [id[0], id[1], id[2], id[3]];
	}
	let mapped: &[u8; 4] = match id {
		b"TT2" => b"TIT2",
		b"TP1" => b"TPE1",
		b"TP2" => b"TPE2",
		b"TAL" => b"TALB",
		b"TRK" => b"TRCK",
		b"TPA" => b"TPOS",
		b"TCO" => b"TCON",
		b"TYE" => b"TYER",
		b"TT3" => b"TIT3",
		b"TXX" => b"TXXX",
		b"COM" => b"COMM",
		b"ULT" => b"USLT",
		b"PIC" => b"PIC ",
		_ => {
			let mut out = [b' '; 4];
			for (slot, &b) in out.iter_mut().zip(id) {
				*slot = b;
			}
			return out;
		}
	};
	*mapped
}

/// Splits `data` at the first string terminator for `encoding`, returning
/// (string bytes, remainder after the terminator).
fn split_terminated(data: &[u8], encoding: u8) -> (&[u8], &[u8]) {
	if encoding == 1 || encoding == 2 {
		let mut i = 0;
		while i + 1 < data.len() {
			if data[i] == 0 && data[i + 1] == 0 {
				return (&data[..i], &data[i + 2..]);
			}
			i += 2;
		}
		(data, &[])
	} else {
		match data.iter().position(|&b| b == 0) {
			Some(i) => (&data[..i], &data[i + 1..]),
			None => (data, &[]),
		}
	}
}

fn decode_text(data: &[u8], encoding: u8) -> String {
	let text = match encoding {
		0 => data.iter().map(|&b| char::from(b)).collect(),
		1 | 2 => {
			let (big_endian, body) = match data {
				[0xFF, 0xFE, rest @ ..] => (false, rest),
				[0xFE, 0xFF, rest @ ..] => (true, rest),
				_ => (true, data),
			};
			let units = body.chunks_exact(2).map(|c| {
				if big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) }
			});
			char::decode_utf16(units).map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER)).collect()
		}
		_ => String::from_utf8_lossy(data).into_owned(),
	};
	text.trim_end_matches('\0').trim().to_string()
}

/// Decodes a text frame body; v2.4 multi-value lists become `"a; b"`.
fn text_frame(data: &[u8]) -> Option<String> {
	let (&encoding, rest) = data.split_first()?;
	let mut values = Vec::new();
	let mut remaining = rest;
	while !remaining.is_empty() {
		let (value, next) = split_terminated(remaining, encoding);
		let value = decode_text(value, encoding);
		if !value.is_empty() {
			values.push(value);
		}
		remaining = next;
	}
	if values.is_empty() { None } else { Some(values.join("; ")) }
}

/// Resolves `"(17)"`, `"(17)Rock"` and `"17"` genre references.
fn resolve_genre(value: &str) -> String {
	let inner = value.strip_prefix('(').and_then(|v| v.split_once(')'));
	if let Some((number, rest)) = inner {
		if !rest.is_empty() {
			return rest.to_string();
		}
		if let Some(name) = number.parse::<usize>().ok().and_then(|n| ID3V1_GENRES.get(n)) {
			return (*name).to_string();
		}
	}
	if let Some(name) = value.parse::<usize>().ok().and_then(|n| ID3V1_GENRES.get(n)) {
		return (*name).to_string();
	}
	value.to_string()
}

/// Parses "n" or "n/total".
fn number_pair(value: &str) -> (Option<u32>, Option<u32>) {
	let mut parts = value.splitn(2, '/');
	let first = parts.next().and_then(|v| v.trim().parse().ok());
	let second = parts.next().and_then(|v| v.trim().parse().ok());
	(first, second)
}

fn apply_frame(id: &[u8; 4], data: &[u8], metadata: &mut Metadata) {
	match id {
		b"COMM" | b"USLT" => {
			// encoding, 3-byte language, short description, text
			let Some((&encoding, rest)) = data.split_first() else { return };
			let Some(rest) = rest.get(3..) else { return };
			let (_description, text) = split_terminated(rest, encoding);
			let text = decode_text(text, encoding);
			if text.is_empty() {
				return;
			}
			let slot = if id == b"COMM" { &mut metadata.comment } else { &mut metadata.lyrics };
			if slot.is_none() {
				*slot = Some(text);
			}
		}
		b"APIC" => {
			if let Some(image) = parse_picture(data, false) {
				metadata.images.push(image);
			}
		}
		b"PIC " => {
			if let Some(image) = parse_picture(data, true) {
				metadata.images.push(image);
			}
		}
		b"TXXX" => {
			let Some((&encoding, rest)) = data.split_first() else { return };
			let (description, value) = split_terminated(rest, encoding);
			let key = decode_text(description, encoding);
			let value = decode_text(value, encoding);
			if key.is_empty() || value.is_empty() {
				return;
			}
			// FFmpeg stores keys without a dedicated frame as TXXX.
			match key.to_ascii_lowercase().as_str() {
				"comment" if metadata.comment.is_none() => metadata.comment = Some(value),
				"description" => metadata.description = Some(value),
				"lyrics" if metadata.lyrics.is_none() => metadata.lyrics = Some(value),
				_ => {
					metadata.raw.insert(key, RawValue::String(value));
				}
			}
		}
		[b'T', ..] => {
			let Some(value) = text_frame(data) else { return };
			match id {
				b"TIT2" => metadata.title = Some(value),
				b"TPE1" => metadata.artist = Some(value),
				b"TALB" => metadata.album = Some(value),
				b"TPE2" => metadata.album_artist = Some(value),
				b"TCON" => metadata.genre = Some(resolve_genre(&value)),
				b"TDRC" => metadata.date = Some(value),
				b"TYER" => {
					if metadata.date.is_none() {
						metadata.date = Some(value);
					}
				}
				b"TRCK" => {
					let (n, total) = number_pair(&value);
					metadata.track_number = n;
					metadata.tracks_total = total;
				}
				b"TPOS" => {
					let (n, total) = number_pair(&value);
					metadata.disc_number = n;
					metadata.discs_total = total;
				}
				_ => {
					let key = String::from_utf8_lossy(id).trim().to_string();
					metadata.raw.insert(key, RawValue::String(value));
				}
			}
		}
		_ => {}
	}
}

fn parse_picture(data: &[u8], v22: bool) -> Option<AttachedImage> {
	let (&encoding, rest) = data.split_first()?;
	let (mime_type, rest) = if v22 {
		let format = rest.get(..3)?;
		let mime = match format.to_ascii_uppercase().as_slice() {
			b"PNG" => "image/png",
			_ => "image/jpeg",
		};
		(mime.to_string(), rest.get(3..)?)
	} else {
		let (mime, rest) = split_terminated(rest, 0);
		let mime = decode_text(mime, 0);
		let mime = if mime.contains('/') { mime } else { format!("image/{}", mime.to_lowercase()) };
		(mime, rest)
	};
	let (_picture_type, rest) = rest.split_first()?;
	let (description, image) = split_terminated(rest, encoding);
	if image.is_empty() {
		return None;
	}
	let description = decode_text(description, encoding);
	Some(AttachedImage {
		data: image.to_vec(),
		mime_type,
		description: if description.is_empty() { None } else { Some(description) },
	})
}

/// Parses a 128-byte ID3v1/v1.1 tag, filling only fields that are unset.
/// Returns `false` if `tag` is not an ID3v1 tag.
pub fn parse_v1(tag: &[u8], metadata: &mut Metadata) -> bool {
	let Some(tag) = tag.get(..ID3V1_SIZE) else { return false };
	if &tag[..3] != b"TAG" {
		return false;
	}
	let field = |range: std::ops::Range<usize>| {
		let text = decode_text(&tag[range], 0);
		if text.is_empty() { None } else { Some(text) }
	};
	metadata.title = metadata.title.take().or_else(|| field(3..33));
	metadata.artist = metadata.artist.take().or_else(|| field(33..63));
	metadata.album = metadata.album.take().or_else(|| field(63..93));
	metadata.date = metadata.date.take().or_else(|| field(93..97));
	// ID3v1.1: a zero byte before the last comment byte marks a track number.
	if tag[125] == 0 && tag[126] != 0 {
		metadata.comment = metadata.comment.take().or_else(|| field(97..125));
		metadata.track_number = metadata.track_number.or(Some(u32::from(tag[126])));
	} else {
		metadata.comment = metadata.comment.take().or_else(|| field(97..127));
	}
	if metadata.genre.is_none() {
		metadata.genre = ID3V1_GENRES.get(usize::from(tag[127])).map(|g| (*g).to_string());
	}
	true
}

fn push_frame(out: &mut Vec<u8>, id: &[u8; 4], body: &[u8]) -> bool {
	if body.len() > MAX_SYNCSAFE {
		return false;
	}
	out.extend_from_slice(id);
	write_syncsafe(out, body.len());
	out.extend_from_slice(&[0, 0]);
	out.extend_from_slice(body);
	true
}

fn push_text(out: &mut Vec<u8>, id: &[u8; 4], value: Option<&str>) {
	if let Some(value) = value.filter(|v| !v.is_empty()) {
		let mut body = Vec::with_capacity(value.len() + 1);
		body.push(3); // UTF-8
		body.extend_from_slice(value.as_bytes());
		push_frame(out, id, &body);
	}
}

fn pair_text(n: Option<u32>, total: Option<u32>) -> Option<String> {
	match (n, total) {
		(Some(n), Some(t)) => Some(format!("{n}/{t}")),
		(Some(n), None) => Some(n.to_string()),
		_ => None,
	}
}

fn is_text_frame_id(key: &str) -> bool {
	let b = key.as_bytes();
	b.len() == 4
		&& b[0] == b'T'
		&& b.iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
		&& key != "TXXX"
}

/// Builds an ID3v2.4 tag for `metadata`, or `None` when there is nothing to
/// write. Raw string values whose key is a v2.4 text frame ID (e.g. `TCOM`)
/// are written as that frame, so ID3 → ID3 remuxing round-trips them.
pub fn build_v2(metadata: &Metadata) -> Result<Option<Vec<u8>>> {
	let mut frames = Vec::new();
	push_text(&mut frames, b"TIT2", metadata.title.as_deref());
	push_text(&mut frames, b"TPE1", metadata.artist.as_deref());
	push_text(&mut frames, b"TALB", metadata.album.as_deref());
	push_text(&mut frames, b"TPE2", metadata.album_artist.as_deref());
	push_text(&mut frames, b"TCON", metadata.genre.as_deref());
	push_text(&mut frames, b"TDRC", metadata.date.as_deref());
	push_text(
		&mut frames,
		b"TRCK",
		pair_text(metadata.track_number, metadata.tracks_total).as_deref(),
	);
	push_text(&mut frames, b"TPOS", pair_text(metadata.disc_number, metadata.discs_total).as_deref());

	let standard = ["TIT2", "TPE1", "TALB", "TPE2", "TCON", "TDRC", "TRCK", "TPOS"];
	let mut raw_keys: Vec<&String> = metadata.raw.keys().collect();
	raw_keys.sort(); // deterministic output
	for key in raw_keys {
		if let Some(RawValue::String(value)) = metadata.raw.get(key) {
			if is_text_frame_id(key) && !standard.contains(&key.as_str()) {
				let id = key.as_bytes();
				push_text(&mut frames, &[id[0], id[1], id[2], id[3]], Some(value));
			}
		}
	}

	for (id, value) in [(b"COMM", &metadata.comment), (b"USLT", &metadata.lyrics)] {
		if let Some(text) = value.as_deref().filter(|v| !v.is_empty()) {
			let mut body = vec![3];
			body.extend_from_slice(b"eng");
			body.push(0); // empty description
			body.extend_from_slice(text.as_bytes());
			push_frame(&mut frames, id, &body);
		}
	}
	for image in &metadata.images {
		let mut body = vec![3];
		body.extend_from_slice(image.mime_type.as_bytes());
		body.push(0);
		body.push(3); // front cover
		if let Some(description) = &image.description {
			body.extend_from_slice(description.as_bytes());
		}
		body.push(0);
		body.extend_from_slice(&image.data);
		// Images too large for a syncsafe size are skipped, not truncated.
		push_frame(&mut frames, b"APIC", &body);
	}

	if frames.is_empty() {
		return Ok(None);
	}
	if frames.len() > MAX_SYNCSAFE {
		return Err(Message::Container("id3v2 tag too large"));
	}
	let mut tag = Vec::with_capacity(ID3V2_HEADER_SIZE + frames.len());
	tag.extend_from_slice(b"ID3");
	tag.extend_from_slice(&[4, 0, 0]);
	write_syncsafe(&mut tag, frames.len());
	tag.extend_from_slice(&frames);
	Ok(Some(tag))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn v2_round_trip() {
		let metadata = Metadata {
			title: Some("Tïtle".into()),
			artist: Some("Artist".into()),
			track_number: Some(3),
			tracks_total: Some(12),
			genre: Some("Rock".into()),
			comment: Some("hello".into()),
			images: vec![AttachedImage {
				data: vec![1, 2, 3],
				mime_type: "image/png".into(),
				description: None,
			}],
			..Default::default()
		};
		let tag = build_v2(&metadata).unwrap().unwrap();
		assert_eq!(v2_tag_size(&tag), Some(tag.len()));
		let mut parsed = Metadata::default();
		parse_v2(&tag, &mut parsed).unwrap();
		assert_eq!(parsed.title.as_deref(), Some("Tïtle"));
		assert_eq!(parsed.track_number, Some(3));
		assert_eq!(parsed.tracks_total, Some(12));
		assert_eq!(parsed.comment.as_deref(), Some("hello"));
		assert_eq!(parsed.images.len(), 1);
		assert_eq!(parsed.images[0].data, vec![1, 2, 3]);
	}

	#[test]
	fn genre_references_resolve() {
		assert_eq!(resolve_genre("(17)"), "Rock");
		assert_eq!(resolve_genre("(17)Rock 'n Roll"), "Rock 'n Roll");
		assert_eq!(resolve_genre("13"), "Pop");
		assert_eq!(resolve_genre("Shoegaze"), "Shoegaze");
	}

	#[test]
	fn utf16_text_with_bom() {
		let data = [1u8, 0xFF, 0xFE, b'H', 0, b'i', 0, 0, 0];
		assert_eq!(text_frame(&data).as_deref(), Some("Hi"));
	}

	#[test]
	fn v1_fills_missing_fields() {
		let mut tag = [0u8; 128];
		tag[..3].copy_from_slice(b"TAG");
		tag[3..8].copy_from_slice(b"Title");
		tag[126] = 5;
		tag[127] = 17;
		let mut metadata = Metadata { artist: Some("Kept".into()), ..Default::default() };
		assert!(parse_v1(&tag, &mut metadata));
		assert_eq!(metadata.title.as_deref(), Some("Title"));
		assert_eq!(metadata.artist.as_deref(), Some("Kept"));
		assert_eq!(metadata.track_number, Some(5));
		assert_eq!(metadata.genre.as_deref(), Some("Rock"));
	}
}
