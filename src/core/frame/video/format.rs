//! Raw (uncompressed) picture layouts.
//!
//! A [`Pixel`] is a layout plus a bit depth. It maps one-to-one onto
//! FFmpeg's `AV_PIX_FMT_*` names (`yuv420p`, `yuv422p10le`, `gray`, `rgb24`,
//! `nv12`, ...), which is how streams describe it in
//! `Parameters::pixel_format`. Planar samples deeper than 8 bits take two
//! bytes, little-endian, as in FFmpeg's `*le` formats. Packed RGB and NV12
//! are 8-bit only.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
	YUV420,
	YUV422,
	YUV444,
	/// 4:1:1, chroma subsampled 4x horizontally.
	YUV411,
	/// 4:4:4 with a full-resolution alpha plane.
	YUVA444,
	/// Luma only.
	Gray,
	/// Packed 8-bit R, G, B.
	RGB24,
	/// Packed 8-bit B, G, R.
	BGR24,
	/// Packed 8-bit R, G, B, A.
	RGBA,
	/// Packed 8-bit B, G, R, A.
	BGRA,
	/// 4:2:0 with a luma plane and one interleaved U/V plane.
	NV12,
}

impl PixelFormat {
	/// `(horizontal, vertical)` chroma subsampling as right shifts
	/// (FFmpeg's `log2_chroma_w` / `log2_chroma_h`).
	pub const fn chroma_shift(self) -> (u32, u32) {
		match self {
			PixelFormat::YUV420 | PixelFormat::NV12 => (1, 1),
			PixelFormat::YUV422 => (1, 0),
			PixelFormat::YUV411 => (2, 0),
			_ => (0, 0),
		}
	}

	/// Number of stored planes (1 for packed RGB, 2 for NV12).
	pub const fn planes(self) -> usize {
		match self {
			PixelFormat::Gray
			| PixelFormat::RGB24
			| PixelFormat::BGR24
			| PixelFormat::RGBA
			| PixelFormat::BGRA => 1,
			PixelFormat::NV12 => 2,
			PixelFormat::YUVA444 => 4,
			_ => 3,
		}
	}

	/// `true` for the RGB family (packed formats).
	pub const fn is_rgb(self) -> bool {
		matches!(self, PixelFormat::RGB24 | PixelFormat::BGR24 | PixelFormat::RGBA | PixelFormat::BGRA)
	}

	/// `true` when the layout carries an alpha channel.
	pub const fn has_alpha(self) -> bool {
		matches!(self, PixelFormat::YUVA444 | PixelFormat::RGBA | PixelFormat::BGRA)
	}

	/// Bytes per pixel of a packed RGB format (0 otherwise).
	pub const fn packed_bytes(self) -> usize {
		match self {
			PixelFormat::RGB24 | PixelFormat::BGR24 => 3,
			PixelFormat::RGBA | PixelFormat::BGRA => 4,
			_ => 0,
		}
	}

	const fn base_name(self) -> &'static str {
		match self {
			PixelFormat::YUV420 => "yuv420p",
			PixelFormat::YUV422 => "yuv422p",
			PixelFormat::YUV444 => "yuv444p",
			PixelFormat::YUV411 => "yuv411p",
			PixelFormat::YUVA444 => "yuva444p",
			PixelFormat::Gray => "gray",
			PixelFormat::RGB24 => "rgb24",
			PixelFormat::BGR24 => "bgr24",
			PixelFormat::RGBA => "rgba",
			PixelFormat::BGRA => "bgra",
			PixelFormat::NV12 => "nv12",
		}
	}
}

/// Bit depths each layout supports, mirroring the FFmpeg pixel formats that
/// exist (there is no `yuv411p10le`, for example).
fn supported_depth(format: PixelFormat, depth: u8) -> bool {
	match format {
		PixelFormat::YUV420 | PixelFormat::YUV422 | PixelFormat::YUV444 => {
			matches!(depth, 8 | 9 | 10 | 12 | 14 | 16)
		}
		PixelFormat::Gray => matches!(depth, 8 | 9 | 10 | 12 | 14 | 16),
		_ => depth == 8,
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pixel {
	pub depth: u8,
	pub format: PixelFormat,
}

impl Default for Pixel {
	fn default() -> Self {
		Self { depth: 8, format: PixelFormat::YUV420 }
	}
}

impl Pixel {
	pub fn new(format: PixelFormat, depth: u8) -> Option<Self> {
		supported_depth(format, depth).then_some(Self { depth, format })
	}

	/// Parses an FFmpeg pixel format name such as `yuv420p` or `yuv444p10le`.
	pub fn from_name(name: &str) -> Option<Self> {
		const FORMATS: [PixelFormat; 11] = [
			PixelFormat::YUVA444,
			PixelFormat::YUV420,
			PixelFormat::YUV422,
			PixelFormat::YUV444,
			PixelFormat::YUV411,
			PixelFormat::Gray,
			PixelFormat::RGB24,
			PixelFormat::BGR24,
			PixelFormat::RGBA,
			PixelFormat::BGRA,
			PixelFormat::NV12,
		];
		for format in FORMATS {
			let Some(rest) = name.strip_prefix(format.base_name()) else {
				continue;
			};
			if rest.is_empty() {
				return Self::new(format, 8);
			}
			let digits = rest.strip_suffix("le")?;
			let depth: u8 = digits.parse().ok()?;
			if depth == 8 {
				return None;
			}
			return Self::new(format, depth);
		}
		None
	}

	/// The FFmpeg pixel format name (`yuv420p`, `gray10le`, ...).
	pub fn name(&self) -> String {
		if self.depth == 8 {
			self.format.base_name().to_string()
		} else {
			format!("{}{}le", self.format.base_name(), self.depth)
		}
	}

	pub const fn bytes_per_sample(&self) -> usize {
		if self.depth > 8 { 2 } else { 1 }
	}

	/// Byte sizes of each plane for a `width` x `height` picture, or `None` if
	/// the size does not fit in `usize`. Chroma dimensions round up, as with
	/// FFmpeg's `AV_CEIL_RSHIFT`.
	pub fn plane_sizes(&self, width: u32, height: u32) -> Option<Vec<usize>> {
		let bps = self.bytes_per_sample();
		let (sx, sy) = self.format.chroma_shift();
		let pixels = (width as usize).checked_mul(height as usize)?;
		if self.format.is_rgb() {
			return Some(vec![pixels.checked_mul(self.format.packed_bytes())?]);
		}
		let luma = pixels.checked_mul(bps)?;
		let cw = (width as usize).div_ceil(1 << sx);
		let ch = (height as usize).div_ceil(1 << sy);
		let chroma = cw.checked_mul(ch)?.checked_mul(bps)?;
		Some(match self.format {
			PixelFormat::Gray => vec![luma],
			PixelFormat::NV12 => vec![luma, chroma.checked_mul(2)?],
			PixelFormat::YUVA444 => vec![luma, chroma, chroma, luma],
			_ => vec![luma, chroma, chroma],
		})
	}

	/// Total bytes of one packed-planar picture.
	pub fn frame_size(&self, width: u32, height: u32) -> Option<usize> {
		self.plane_sizes(width, height)?.into_iter().try_fold(0usize, |acc, n| acc.checked_add(n))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn names_round_trip() {
		for name in [
			"yuv420p",
			"yuv422p10le",
			"yuv444p16le",
			"yuv411p",
			"yuva444p",
			"gray",
			"gray12le",
			"rgb24",
			"bgra",
			"nv12",
		] {
			let pixel = Pixel::from_name(name).unwrap_or_else(|| panic!("{name}"));
			assert_eq!(pixel.name(), name);
		}
		for name in
			["yuv420p8le", "yuv411p10le", "rgb48", "rgb2410le", "yuv420p10be", "yuv420p11le", ""]
		{
			assert!(Pixel::from_name(name).is_none(), "{name}");
		}
	}

	#[test]
	fn frame_sizes_round_chroma_up() {
		let p = Pixel::from_name("yuv420p").unwrap();
		assert_eq!(p.frame_size(5, 3), Some(15 + 2 * 3 * 2));
		let p = Pixel::from_name("yuv422p10le").unwrap();
		assert_eq!(p.frame_size(4, 2), Some(2 * (8 + 2 * 4)));
		let p = Pixel::from_name("yuv411p").unwrap();
		assert_eq!(p.frame_size(9, 1), Some(9 + 2 * 3));
		assert_eq!(Pixel::from_name("rgba").unwrap().frame_size(3, 2), Some(24));
		assert_eq!(Pixel::from_name("nv12").unwrap().plane_sizes(5, 3), Some(vec![15, 12]));
	}
}
