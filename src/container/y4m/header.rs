//! YUV4MPEG2 stream header: parsing and writing.
//!
//! The header is one text line, `YUV4MPEG2` followed by space-separated
//! tagged fields (`W640 H480 F30000:1001 Ip A1:1 C420jpeg XYSCSS=420JPEG`).
//! Field handling follows FFmpeg's `libavformat/yuv4mpegdec.c` (reading) and
//! `yuv4mpegenc.c` (writing): the same colorspace tokens, the same defaults
//! (25 fps, `420jpeg`) and the same `X` extensions.

use crate::Message;
use crate::core::Pixel;
use crate::message::Result;

pub const MAGIC: &[u8] = b"YUV4MPEG2";
pub const FRAME_MAGIC: &[u8] = b"FRAME";
/// `FRAME\n`: the frame header FFmpeg writes, and the one seeking assumes.
pub const PLAIN_FRAME_HEADER_LEN: u64 = 6;
/// Longest stream header accepted. FFmpeg's limit is lower; comments (`X`
/// fields) make longer headers legitimate, and the line is bounded either way.
pub const MAX_HEADER_LEN: usize = 1024;
/// Longest frame header (`FRAME` plus optional per-frame fields).
pub const MAX_FRAME_HEADER_LEN: usize = 256;
/// Largest picture accepted, in bytes. Bounds the per-frame allocation the
/// header can request (16384x16384 4:4:4 at 8 bits is 768 MiB, so the limit
/// is on the product, not the dimensions).
pub const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

/// Chroma sample position for 4:2:0 (FFmpeg's `AVChromaLocation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaLocation {
	/// `420jpeg`: chroma centered between luma samples (MPEG-1, JPEG).
	Center,
	/// `420mpeg2`: chroma co-sited horizontally with the left luma sample.
	Left,
	/// `420paldv`: chroma co-sited with the top-left luma sample (PAL DV).
	TopLeft,
}

impl ChromaLocation {
	pub const fn name(self) -> &'static str {
		match self {
			ChromaLocation::Center => "center",
			ChromaLocation::Left => "left",
			ChromaLocation::TopLeft => "topleft",
		}
	}

	pub fn from_name(name: &str) -> Option<Self> {
		match name {
			"center" => Some(Self::Center),
			"left" => Some(Self::Left),
			"topleft" => Some(Self::TopLeft),
			_ => None,
		}
	}
}

/// Field order from the `I` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interlace {
	Progressive,
	TopFirst,
	BottomFirst,
	/// `Im`: each frame header says. Frames are still returned as stored.
	Mixed,
}

impl Interlace {
	/// FFmpeg's `field_order` names.
	pub const fn name(self) -> &'static str {
		match self {
			Interlace::Progressive => "progressive",
			Interlace::TopFirst => "tt",
			Interlace::BottomFirst => "bb",
			Interlace::Mixed => "mixed",
		}
	}

	pub fn from_name(name: &str) -> Option<Self> {
		match name {
			"progressive" => Some(Self::Progressive),
			"tt" => Some(Self::TopFirst),
			"bb" => Some(Self::BottomFirst),
			"mixed" => Some(Self::Mixed),
			_ => None,
		}
	}

	const fn code(self) -> char {
		match self {
			Interlace::Progressive => 'p',
			Interlace::TopFirst => 't',
			Interlace::BottomFirst => 'b',
			Interlace::Mixed => 'm',
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Y4mHeader {
	pub width: u32,
	pub height: u32,
	pub pixel: Pixel,
	/// Frames per second as `(num, den)`; never zero.
	pub rate: (u32, u32),
	/// Sample (pixel) aspect ratio; `(0, 0)` means unknown.
	pub aspect: (u32, u32),
	pub interlace: Option<Interlace>,
	pub chroma_location: Option<ChromaLocation>,
	/// `Some(true)` for full range, `Some(false)` for limited.
	pub full_range: Option<bool>,
}

impl Y4mHeader {
	/// Bytes of picture data per frame (already checked against
	/// [`MAX_FRAME_BYTES`] by [`parse_header`]).
	pub fn frame_size(&self) -> Option<usize> {
		self.pixel.frame_size(self.width, self.height)
	}
}

/// Maps a `C` token to its pixel format and implied chroma location.
fn colorspace(token: &str) -> Option<(&'static str, Option<ChromaLocation>)> {
	Some(match token {
		"420jpeg" | "420" => ("yuv420p", Some(ChromaLocation::Center)),
		"420mpeg2" => ("yuv420p", Some(ChromaLocation::Left)),
		"420paldv" => ("yuv420p", Some(ChromaLocation::TopLeft)),
		"411" => ("yuv411p", None),
		"422" => ("yuv422p", None),
		"444" => ("yuv444p", None),
		"444alpha" => ("yuva444p", None),
		"mono" => ("gray", None),
		"mono9" => ("gray9le", None),
		"mono10" => ("gray10le", None),
		"mono12" => ("gray12le", None),
		"mono16" => ("gray16le", None),
		"420p9" => ("yuv420p9le", None),
		"420p10" => ("yuv420p10le", None),
		"420p12" => ("yuv420p12le", None),
		"420p14" => ("yuv420p14le", None),
		"420p16" => ("yuv420p16le", None),
		"422p9" => ("yuv422p9le", None),
		"422p10" => ("yuv422p10le", None),
		"422p12" => ("yuv422p12le", None),
		"422p14" => ("yuv422p14le", None),
		"422p16" => ("yuv422p16le", None),
		"444p9" => ("yuv444p9le", None),
		"444p10" => ("yuv444p10le", None),
		"444p12" => ("yuv444p12le", None),
		"444p14" => ("yuv444p14le", None),
		"444p16" => ("yuv444p16le", None),
		_ => return None,
	})
}

/// The `C...` token (and FFmpeg's `XYSCSS` echo) for a pixel format, or
/// `None` if Y4M cannot carry it.
fn colorspace_token(pixel: &Pixel, location: Option<ChromaLocation>) -> Option<String> {
	let name = pixel.name();
	let token = match name.as_str() {
		"yuv420p" => {
			return Some(
				match location {
					Some(ChromaLocation::TopLeft) => " C420paldv XYSCSS=420PALDV",
					Some(ChromaLocation::Left) => " C420mpeg2 XYSCSS=420MPEG2",
					_ => " C420jpeg XYSCSS=420JPEG",
				}
				.to_string(),
			);
		}
		"yuv411p" => " C411 XYSCSS=411",
		"yuv422p" => " C422 XYSCSS=422",
		"yuv444p" => " C444 XYSCSS=444",
		"yuva444p" => " C444alpha XYSCSS=444",
		"gray" => " Cmono",
		"gray9le" => " Cmono9",
		"gray10le" => " Cmono10",
		"gray12le" => " Cmono12",
		"gray16le" => " Cmono16",
		_ => {
			// yuv4NNpDDle -> " C4NNpDD XYSCSS=4NNPDD"
			let rest = name.strip_prefix("yuv")?.strip_suffix("le")?;
			let (sub, depth) = rest.split_once('p')?;
			if !matches!(sub, "420" | "422" | "444") || depth.is_empty() {
				return None;
			}
			return Some(format!(" C{sub}p{depth} XYSCSS={sub}P{depth}"));
		}
	};
	Some(token.to_string())
}

fn parse_ratio(value: &str) -> Option<(u32, u32)> {
	let (num, den) = value.split_once(':')?;
	Some((num.parse().ok()?, den.parse().ok()?))
}

/// Parses the stream header line, without its trailing `\n`.
pub fn parse_header(line: &[u8]) -> Result<Y4mHeader> {
	let invalid = || Message::Container("invalid y4m header");
	let rest = line.strip_prefix(MAGIC).ok_or_else(invalid)?;
	let text = std::str::from_utf8(rest).map_err(|_| invalid())?;

	let mut width = None;
	let mut height = None;
	let mut colorspace_name = None;
	let mut alt_colorspace = None;
	let mut rate = None;
	let mut aspect = (0, 0);
	let mut interlace = None;
	let mut full_range = None;

	for token in text.split(' ').filter(|t| !t.is_empty()) {
		// Split on a char boundary: the tag may be any (non-ASCII) character.
		let mut chars = token.chars();
		let tag = chars.next();
		let value = chars.as_str();
		match tag {
			Some('W') => width = Some(value.parse::<u32>().map_err(|_| invalid())?),
			Some('H') => height = Some(value.parse::<u32>().map_err(|_| invalid())?),
			Some('C') => colorspace_name = Some(value),
			// Like FFmpeg, a malformed rate or aspect falls back to the default.
			Some('F') => rate = parse_ratio(value),
			Some('A') => aspect = parse_ratio(value).unwrap_or((0, 0)),
			Some('I') => {
				interlace = match value {
					"p" => Some(Interlace::Progressive),
					"t" => Some(Interlace::TopFirst),
					"b" => Some(Interlace::BottomFirst),
					"m" => Some(Interlace::Mixed),
					_ => None,
				}
			}
			Some('X') => {
				if let Some(v) = value.strip_prefix("YSCSS=") {
					alt_colorspace = Some(v.to_ascii_lowercase());
				} else if let Some(v) = value.strip_prefix("COLORRANGE=") {
					full_range = match v {
						"FULL" => Some(true),
						"LIMITED" => Some(false),
						_ => None,
					};
				}
			}
			// Unknown tags are ignored, as the format requires.
			_ => {}
		}
	}

	let width = width.filter(|w| *w > 0).ok_or_else(invalid)?;
	let height = height.filter(|h| *h > 0).ok_or_else(invalid)?;

	let (pixel_name, chroma_location) = match colorspace_name {
		Some(name) => colorspace(name).ok_or(Message::Container("unsupported y4m colorspace"))?,
		// Without `C`, FFmpeg honours the `XYSCSS` extension, then defaults to 4:2:0.
		None => alt_colorspace
			.as_deref()
			.and_then(colorspace)
			.unwrap_or(("yuv420p", Some(ChromaLocation::Center))),
	};
	let pixel = Pixel::from_name(pixel_name).ok_or_else(invalid)?;

	let rate = match rate {
		Some((num, den)) if num > 0 && den > 0 => (num, den),
		_ => (25, 1),
	};

	let header =
		Y4mHeader { width, height, pixel, rate, aspect, interlace, chroma_location, full_range };
	match header.frame_size() {
		Some(size) if size <= MAX_FRAME_BYTES => Ok(header),
		_ => Err(Message::Container("y4m picture too large")),
	}
}

/// Formats the stream header line, including its trailing `\n`.
pub fn write_header(header: &Y4mHeader) -> Result<String> {
	let colorspace = colorspace_token(&header.pixel, header.chroma_location)
		.ok_or(Message::Container("pixel format not representable in y4m"))?;
	let interlace = header.interlace.unwrap_or(Interlace::Progressive).code();
	let range = match header.full_range {
		Some(true) => " XCOLORRANGE=FULL",
		Some(false) => " XCOLORRANGE=LIMITED",
		None => "",
	};
	Ok(format!(
		"YUV4MPEG2 W{} H{} F{}:{} I{} A{}:{}{}{}\n",
		header.width,
		header.height,
		header.rate.0,
		header.rate.1,
		interlace,
		header.aspect.0,
		header.aspect.1,
		colorspace,
		range
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_ffmpeg_style_header() {
		let h = parse_header(
			b"YUV4MPEG2 W320 H240 F30000:1001 It A1:1 C420mpeg2 XYSCSS=420MPEG2 XCOLORRANGE=FULL",
		)
		.unwrap();
		assert_eq!((h.width, h.height, h.rate, h.aspect), (320, 240, (30000, 1001), (1, 1)));
		assert_eq!(h.pixel.name(), "yuv420p");
		assert_eq!(h.chroma_location, Some(ChromaLocation::Left));
		assert_eq!(h.interlace, Some(Interlace::TopFirst));
		assert_eq!(h.full_range, Some(true));
		assert_eq!(h.frame_size(), Some(320 * 240 * 3 / 2));
	}

	#[test]
	fn applies_defaults() {
		let h = parse_header(b"YUV4MPEG2 W2 H2").unwrap();
		assert_eq!(h.rate, (25, 1));
		assert_eq!(h.pixel.name(), "yuv420p");
		let h = parse_header(b"YUV4MPEG2 W2 H2 F0:0 XYSCSS=444").unwrap();
		assert_eq!((h.rate, h.pixel.name()), ((25, 1), "yuv444p".to_string()));
	}

	#[test]
	fn rejects_bad_headers() {
		for line in [
			&b"YUV4MPEG2 H2"[..],
			b"YUV4MPEG2 W0 H2",
			b"YUV4MPEG2 Wx H2",
			b"YUV4MPEG2 W2 H2 Crgb",
			b"YUV4MPEG W2 H2",
			b"YUV4MPEG2 W100000 H100000",
		] {
			assert!(parse_header(line).is_err(), "{}", String::from_utf8_lossy(line));
		}
	}

	#[test]
	fn header_round_trips() {
		for pix in [
			"yuv420p",
			"yuv422p",
			"yuv444p",
			"yuv411p",
			"yuva444p",
			"gray",
			"gray16le",
			"yuv420p10le",
			"yuv444p12le",
		] {
			let header = Y4mHeader {
				width: 8,
				height: 6,
				pixel: Pixel::from_name(pix).unwrap(),
				rate: (24, 1),
				aspect: (0, 0),
				interlace: Some(Interlace::Progressive),
				chroma_location: None,
				full_range: None,
			};
			let line = write_header(&header).unwrap();
			let parsed = parse_header(line.trim_end().as_bytes()).unwrap();
			assert_eq!(parsed.pixel, header.pixel, "{line}");
		}
		let gray14 = Y4mHeader {
			pixel: Pixel::from_name("gray14le").unwrap(),
			..parse_header(b"YUV4MPEG2 W2 H2").unwrap()
		};
		assert!(write_header(&gray14).is_err());
	}
}
