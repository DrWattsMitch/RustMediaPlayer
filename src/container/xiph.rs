//! Metadata formats shared by FLAC and Ogg: Vorbis comments and the FLAC
//! `PICTURE` block (also used, base64-encoded, as `METADATA_BLOCK_PICTURE`
//! in Ogg comments).
//!
//! FFmpeg: `libavformat/oggparsevorbis.c` (`ff_vorbis_comment`),
//! `vorbiscomment.c`, `flac_picture.c`.

use crate::core::{AttachedImage, Metadata, RawValue};
use crate::io::ByteCursor;

/// Largest comment list / picture accepted.
const MAX_FIELD: usize = 64 * 1024 * 1024;
/// FLAC picture type "front cover".
const FRONT_COVER: u32 = 3;

fn read_le_u32(c: &mut ByteCursor) -> Option<u32> {
	c.read_array::<4>().ok().map(u32::from_le_bytes)
}

/// Applies one `KEY=value` comment.
fn apply(metadata: &mut Metadata, key: &str, value: &str) {
	let upper = key.to_ascii_uppercase();
	let number = || value.split('/').next().and_then(|n| n.trim().parse::<u32>().ok());
	let total = || value.split('/').nth(1).and_then(|n| n.trim().parse::<u32>().ok());
	match upper.as_str() {
		"TITLE" => metadata.title = Some(value.into()),
		"ARTIST" => metadata.artist = Some(value.into()),
		"ALBUM" => metadata.album = Some(value.into()),
		"ALBUMARTIST" | "ALBUM_ARTIST" | "ALBUM ARTIST" => metadata.album_artist = Some(value.into()),
		"GENRE" => metadata.genre = Some(value.into()),
		"DATE" | "YEAR" => metadata.date = Some(value.into()),
		"COMMENT" => metadata.comment = Some(value.into()),
		"DESCRIPTION" => metadata.description = Some(value.into()),
		"LYRICS" | "UNSYNCEDLYRICS" => metadata.lyrics = Some(value.into()),
		"TRACKNUMBER" | "TRACK" => {
			metadata.track_number = number();
			metadata.tracks_total = total().or(metadata.tracks_total);
		}
		"TRACKTOTAL" | "TOTALTRACKS" => metadata.tracks_total = number(),
		"DISCNUMBER" | "DISC" => {
			metadata.disc_number = number();
			metadata.discs_total = total().or(metadata.discs_total);
		}
		"DISCTOTAL" | "TOTALDISCS" => metadata.discs_total = number(),
		"METADATA_BLOCK_PICTURE" => {
			if let Some(image) = base64_decode(value).as_deref().and_then(parse_picture) {
				metadata.images.push(image);
			}
		}
		_ => {
			metadata.raw.insert(key.to_ascii_lowercase(), RawValue::String(value.into()));
		}
	}
}

/// Parses a Vorbis comment list (vendor string + `KEY=value` entries,
/// little-endian lengths) into `metadata`. Returns the vendor string.
pub fn parse_comments(data: &[u8], metadata: &mut Metadata) -> Option<String> {
	let mut c = ByteCursor::new(data);
	let vendor_len = read_le_u32(&mut c)? as usize;
	let vendor = String::from_utf8_lossy(c.read_bytes(vendor_len).ok()?).into_owned();
	let count = read_le_u32(&mut c)?;
	for _ in 0..count {
		let Some(len) = read_le_u32(&mut c) else { break };
		let Ok(entry) = c.read_bytes(len as usize) else { break };
		let entry = String::from_utf8_lossy(entry);
		if let Some((key, value)) = entry.split_once('=') {
			if !key.is_empty() {
				apply(metadata, key, value);
			}
		}
	}
	Some(vendor)
}

/// Serialises `metadata` as a Vorbis comment list.
pub fn write_comments(metadata: &Metadata, vendor: &str) -> Vec<u8> {
	let mut entries: Vec<String> = Vec::new();
	let mut push = |key: &str, value: &Option<String>| {
		if let Some(v) = value {
			entries.push(format!("{key}={v}"));
		}
	};
	push("TITLE", &metadata.title);
	push("ARTIST", &metadata.artist);
	push("ALBUM", &metadata.album);
	push("ALBUMARTIST", &metadata.album_artist);
	push("GENRE", &metadata.genre);
	push("DATE", &metadata.date);
	push("COMMENT", &metadata.comment);
	push("DESCRIPTION", &metadata.description);
	push("LYRICS", &metadata.lyrics);
	let mut numbered = |key: &str, n: Option<u32>| {
		if let Some(n) = n {
			entries.push(format!("{key}={n}"));
		}
	};
	numbered("TRACKNUMBER", metadata.track_number);
	numbered("TRACKTOTAL", metadata.tracks_total);
	numbered("DISCNUMBER", metadata.disc_number);
	numbered("DISCTOTAL", metadata.discs_total);
	let mut raw: Vec<_> = metadata
		.raw
		.iter()
		.filter_map(|(k, v)| match v {
			RawValue::String(s) if !k.contains('=') && k != "encoder" => {
				Some((k.to_ascii_uppercase(), s))
			}
			_ => None,
		})
		.collect();
	raw.sort();
	for (key, value) in raw {
		entries.push(format!("{key}={value}"));
	}

	let mut out = Vec::new();
	out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
	out.extend_from_slice(vendor.as_bytes());
	out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
	for entry in entries {
		out.extend_from_slice(&(entry.len() as u32).to_le_bytes());
		out.extend_from_slice(entry.as_bytes());
	}
	out
}

/// Parses a FLAC `PICTURE` block body (big-endian lengths).
pub fn parse_picture(data: &[u8]) -> Option<AttachedImage> {
	let mut c = ByteCursor::new(data);
	let _picture_type = c.read_u32().ok()?;
	let mime_len = c.read_u32().ok()? as usize;
	let mime = String::from_utf8_lossy(c.read_bytes(mime_len).ok()?).into_owned();
	let desc_len = c.read_u32().ok()? as usize;
	let description = String::from_utf8_lossy(c.read_bytes(desc_len).ok()?).into_owned();
	c.skip(16).ok()?; // width, height, depth, colours
	let len = c.read_u32().ok()? as usize;
	if len > MAX_FIELD {
		return None;
	}
	let data = c.read_bytes(len).ok()?.to_vec();
	let mime_type = if mime.is_empty() { "image/jpeg".to_string() } else { mime };
	Some(AttachedImage { data, mime_type, description: Some(description).filter(|d| !d.is_empty()) })
}

/// Serialises a FLAC `PICTURE` block body (front cover, size fields 0).
pub fn write_picture(image: &AttachedImage) -> Vec<u8> {
	let mut out = Vec::new();
	let description = image.description.as_deref().unwrap_or("");
	out.extend_from_slice(&FRONT_COVER.to_be_bytes());
	out.extend_from_slice(&(image.mime_type.len() as u32).to_be_bytes());
	out.extend_from_slice(image.mime_type.as_bytes());
	out.extend_from_slice(&(description.len() as u32).to_be_bytes());
	out.extend_from_slice(description.as_bytes());
	out.extend_from_slice(&[0u8; 16]);
	out.extend_from_slice(&(image.data.len() as u32).to_be_bytes());
	out.extend_from_slice(&image.data);
	out
}

/// Standard base64 (RFC 4648) decoding; whitespace is ignored and padding
/// is optional. `None` on any other character.
pub fn base64_decode(text: &str) -> Option<Vec<u8>> {
	let mut out = Vec::with_capacity(text.len() * 3 / 4);
	let (mut acc, mut bits) = (0u32, 0u32);
	for &b in text.as_bytes() {
		let v = match b {
			b'A'..=b'Z' => b - b'A',
			b'a'..=b'z' => b - b'a' + 26,
			b'0'..=b'9' => b - b'0' + 52,
			b'+' => 62,
			b'/' => 63,
			b'=' | b'\r' | b'\n' | b' ' => continue,
			_ => return None,
		};
		acc = (acc << 6) | u32::from(v);
		bits += 6;
		if bits >= 8 {
			bits -= 8;
			out.push((acc >> bits) as u8);
		}
	}
	Some(out)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn comments_round_trip() {
		let mut m = Metadata {
			title: Some("Song".into()),
			artist: Some("Band".into()),
			track_number: Some(3),
			tracks_total: Some(9),
			..Default::default()
		};
		m.raw.insert("isrc".into(), RawValue::String("XX123".into()));
		let bytes = write_comments(&m, "RMP");
		let mut back = Metadata::default();
		assert_eq!(parse_comments(&bytes, &mut back).as_deref(), Some("RMP"));
		assert_eq!(back.title.as_deref(), Some("Song"));
		assert_eq!((back.track_number, back.tracks_total), (Some(3), Some(9)));
		assert!(matches!(back.raw.get("isrc"), Some(RawValue::String(v)) if v == "XX123"));
	}

	#[test]
	fn pictures_and_base64() {
		let image =
			AttachedImage { data: vec![1, 2, 3], mime_type: "image/png".into(), description: None };
		let parsed = parse_picture(&write_picture(&image)).unwrap();
		assert_eq!((parsed.data, parsed.mime_type), (vec![1, 2, 3], "image/png".to_string()));
		assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
		assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
		assert!(base64_decode("a*b").is_none());
	}
}
