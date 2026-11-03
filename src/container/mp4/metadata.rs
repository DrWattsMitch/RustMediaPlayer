//! iTunes-style metadata (`moov/udta/meta/ilst`), as read by FFmpeg's
//! `mov_read_udta_string` and written by `mov_write_ilst_tag`.

use super::boxes::{
	BoxIter, FourCC, Put, begin_box, end_box, find, fourcc_string, write_box, write_full_box,
};
use crate::container::mp3::id3::ID3V1_GENRES;
use crate::core::{AttachedImage, Metadata, RawValue};
use crate::io::ByteCursor;
use crate::message::Result;

/// `data` box well-known types.
const TYPE_UTF8: u32 = 1;
const TYPE_JPEG: u32 = 13;
const TYPE_PNG: u32 = 14;
const TYPE_BMP: u32 = 27;
const TYPE_INTEGER: u32 = 21;

/// Reads `moov/udta` (and a bare `moov/meta`) into `metadata`.
pub fn parse_moov_metadata(moov: &[u8], metadata: &mut Metadata) {
	if let Some(meta) = find(moov, b"udta").and_then(|udta| find(udta, b"meta")) {
		parse_meta(meta, metadata);
	}
	if let Some(meta) = find(moov, b"meta") {
		parse_meta(meta, metadata);
	}
}

fn parse_meta(meta: &[u8], metadata: &mut Metadata) {
	// ISO `meta` is a full box; QuickTime's is not. Tell them apart by
	// checking whether a box header starts right away.
	let body = if meta.get(4..8) == Some(b"hdlr") { meta } else { meta.get(4..).unwrap_or(&[]) };
	let Some(ilst) = find(body, b"ilst") else { return };
	for (kind, item) in BoxIter::new(ilst).map_while(|r| r.ok()) {
		parse_item(&kind, item, metadata);
	}
}

/// Each item holds one or more `data` boxes: type(4) locale(4) value.
fn data_values(item: &[u8]) -> impl Iterator<Item = (u32, &[u8])> {
	BoxIter::new(item).map_while(|r| r.ok()).filter(|(k, _)| k == b"data").filter_map(|(_, body)| {
		let type_code = u32::from_be_bytes(body.get(..4)?.try_into().ok()?) & 0x00FF_FFFF;
		Some((type_code, body.get(8..)?))
	})
}

fn number_pair(value: &[u8]) -> (Option<u32>, Option<u32>) {
	let mut c = ByteCursor::new(value);
	let _ = c.skip(2);
	let n = c.read_u16().ok().map(u32::from).filter(|&v| v > 0);
	let total = c.read_u16().ok().map(u32::from).filter(|&v| v > 0);
	(n, total)
}

fn integer(value: &[u8]) -> Option<i64> {
	match value.len() {
		1 => Some(i64::from(value[0] as i8)),
		2 => Some(i64::from(i16::from_be_bytes([value[0], value[1]]))),
		4 => Some(i64::from(i32::from_be_bytes([value[0], value[1], value[2], value[3]]))),
		8 => Some(i64::from_be_bytes(value.try_into().ok()?)),
		_ => None,
	}
}

fn parse_item(kind: &FourCC, item: &[u8], metadata: &mut Metadata) {
	if kind == b"----" {
		// Freeform: mean / name / data.
		let name = find(item, b"name").and_then(|n| n.get(4..)).map(|n| String::from_utf8_lossy(n));
		let value = data_values(item).next();
		if let (Some(name), Some((_, value))) = (name, value) {
			let value = String::from_utf8_lossy(value).into_owned();
			metadata.raw.insert(name.into_owned(), RawValue::String(value));
		}
		return;
	}
	for (type_code, value) in data_values(item) {
		let text = || String::from_utf8_lossy(value).trim_end_matches('\0').to_string();
		match kind {
			b"\xa9nam" => metadata.title = Some(text()),
			b"\xa9ART" => metadata.artist = Some(text()),
			b"aART" => metadata.album_artist = Some(text()),
			b"\xa9alb" => metadata.album = Some(text()),
			b"\xa9gen" => metadata.genre = Some(text()),
			b"\xa9day" => metadata.date = Some(text()),
			b"\xa9cmt" => metadata.comment = Some(text()),
			b"\xa9lyr" => metadata.lyrics = Some(text()),
			b"desc" => metadata.description = Some(text()),
			b"trkn" => (metadata.track_number, metadata.tracks_total) = number_pair(value),
			b"disk" => (metadata.disc_number, metadata.discs_total) = number_pair(value),
			b"gnre" => {
				let index = integer(value).and_then(|g| usize::try_from(g - 1).ok());
				if let Some(name) = index.and_then(|i| ID3V1_GENRES.get(i)) {
					metadata.genre = Some((*name).to_string());
				}
			}
			b"covr" => {
				let mime_type = match type_code {
					TYPE_PNG => "image/png",
					TYPE_BMP => "image/bmp",
					_ => "image/jpeg",
				};
				metadata.images.push(AttachedImage {
					data: value.to_vec(),
					mime_type: mime_type.into(),
					description: None,
				});
			}
			_ => {
				let value = if type_code == TYPE_INTEGER || (type_code == 0 && value.len() <= 8) {
					match integer(value) {
						Some(v) => v.to_string(),
						None => text(),
					}
				} else {
					text()
				};
				metadata.raw.insert(fourcc_string(kind), RawValue::String(value));
			}
		}
	}
}

fn write_data(out: &mut Vec<u8>, type_code: u32, value: &[u8]) -> Result<()> {
	write_box(out, b"data", |out| {
		out.put_u32(type_code);
		out.put_u32(0); // locale
		out.put(value);
		Ok(())
	})
}

fn write_text_item(out: &mut Vec<u8>, kind: &FourCC, value: Option<&str>) -> Result<()> {
	match value.filter(|v| !v.is_empty()) {
		Some(value) => write_box(out, kind, |out| write_data(out, TYPE_UTF8, value.as_bytes())),
		None => Ok(()),
	}
}

fn write_pair_item(
	out: &mut Vec<u8>,
	kind: &FourCC,
	n: Option<u32>,
	total: Option<u32>,
) -> Result<()> {
	let Some(n) = n else { return Ok(()) };
	let clamp = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
	let mut value = Vec::with_capacity(8);
	value.put_u16(0);
	value.put_u16(clamp(n));
	value.put_u16(total.map_or(0, clamp));
	if kind == b"trkn" {
		value.put_u16(0);
	}
	write_box(out, kind, |out| write_data(out, 0, &value))
}

/// Writes `udta/meta/ilst` for `metadata`; writes nothing when empty.
pub fn write_udta(out: &mut Vec<u8>, metadata: &Metadata, encoder: &str) -> Result<()> {
	let mut ilst = Vec::new();
	write_text_item(&mut ilst, b"\xa9nam", metadata.title.as_deref())?;
	write_text_item(&mut ilst, b"\xa9ART", metadata.artist.as_deref())?;
	write_text_item(&mut ilst, b"aART", metadata.album_artist.as_deref())?;
	write_text_item(&mut ilst, b"\xa9alb", metadata.album.as_deref())?;
	write_text_item(&mut ilst, b"\xa9gen", metadata.genre.as_deref())?;
	write_text_item(&mut ilst, b"\xa9day", metadata.date.as_deref())?;
	write_text_item(&mut ilst, b"\xa9cmt", metadata.comment.as_deref())?;
	write_text_item(&mut ilst, b"\xa9lyr", metadata.lyrics.as_deref())?;
	write_text_item(&mut ilst, b"desc", metadata.description.as_deref())?;
	write_text_item(&mut ilst, b"\xa9too", Some(encoder))?;
	write_pair_item(&mut ilst, b"trkn", metadata.track_number, metadata.tracks_total)?;
	write_pair_item(&mut ilst, b"disk", metadata.disc_number, metadata.discs_total)?;
	if !metadata.images.is_empty() {
		write_box(&mut ilst, b"covr", |out| {
			for image in &metadata.images {
				let type_code = match image.mime_type.as_str() {
					"image/png" => TYPE_PNG,
					"image/bmp" => TYPE_BMP,
					_ => TYPE_JPEG,
				};
				write_data(out, type_code, &image.data)?;
			}
			Ok(())
		})?;
	}

	write_box(out, b"udta", |out| {
		write_full_box(out, b"meta", 0, 0, |out| {
			write_full_box(out, b"hdlr", 0, 0, |out| {
				out.put_u32(0);
				out.put(b"mdir");
				out.put(b"appl");
				out.put_zeros(9);
				Ok(())
			})?;
			let at = begin_box(out, b"ilst");
			out.put(&ilst);
			end_box(out, at)
		})
	})
}
