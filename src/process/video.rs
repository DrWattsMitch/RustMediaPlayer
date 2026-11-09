//! Video conversion: scaling and pixel-format conversion, the job of
//! FFmpeg's `libswscale` (`-s`, `-pix_fmt`, the `scale` filter).
//!
//! A frame is unpacked into floating-point planes in 8-bit-equivalent units
//! (a 10-bit limited-range black of 64 reads as 16.0), so bit depth never
//! changes the arithmetic. Then:
//!
//! - **YUV → YUV**: luma and chroma planes are each scaled straight to their
//!   output sizes (subsampling changes fold into the chroma scale).
//! - **YUV → RGB**: luma and chroma are scaled to the output picture size
//!   and the colour matrix is applied per pixel.
//! - **RGB → YUV**: R, G, B are scaled to the output size, converted per
//!   pixel, and the chroma planes are then filtered down to their
//!   subsampled size.
//! - **RGB → RGB**: planes are scaled and repacked.
//!
//! Scaling is separable (horizontal then vertical). The kernels are
//! swscale's: point, bilinear, bicubic with `B = 0, C = 0.6` (the default,
//! `SWS_BICUBIC`) and Lanczos-3; when shrinking, the kernel is widened by the
//! scale factor so it also low-passes. Samples are centre-sited
//! (`x_src = (x_dst + 0.5)·src/dst − 0.5`) and edges clamp.
//!
//! Colour conversion defaults to swscale's: BT.601 coefficients, limited
//! ("TV") range YUV, full-range RGB. Output samples are rounded to nearest
//! and clamped.

use crate::core::{Pixel, PixelFormat, VideoFrame};
use crate::message::{Message, Result};

/// Largest picture accepted (as the H.264 decoder's cap, in both axes).
const MAX_DIMENSION: u32 = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoSpec {
	pub width: u32,
	pub height: u32,
	pub pixel: Pixel,
}

impl VideoSpec {
	pub fn new(width: u32, height: u32, pixel: Pixel) -> Self {
		Self { width, height, pixel }
	}

	pub fn of(frame: &VideoFrame) -> Self {
		Self::new(frame.width, frame.height, frame.pixel.clone())
	}
}

/// Resampling kernel for scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleFilter {
	/// Nearest neighbour (`SWS_POINT`).
	Point,
	/// Triangle (`SWS_BILINEAR`).
	Bilinear,
	/// Cubic with B = 0, C = 0.6 (`SWS_BICUBIC`, FFmpeg's default).
	#[default]
	Bicubic,
	/// Lanczos, 3 lobes (`SWS_LANCZOS`).
	Lanczos,
}

impl ScaleFilter {
	pub fn from_name(name: &str) -> Option<Self> {
		Some(match name {
			"point" | "neighbor" | "nearest" => Self::Point,
			"bilinear" | "fast_bilinear" => Self::Bilinear,
			"bicubic" => Self::Bicubic,
			"lanczos" => Self::Lanczos,
			_ => return None,
		})
	}

	fn radius(self) -> f64 {
		match self {
			Self::Point => 0.5,
			Self::Bilinear => 1.0,
			Self::Bicubic => 2.0,
			Self::Lanczos => 3.0,
		}
	}

	fn weight(self, x: f64) -> f64 {
		let ax = x.abs();
		match self {
			Self::Point => {
				if ax < 0.5 || x == -0.5 {
					1.0
				} else {
					0.0
				}
			}
			Self::Bilinear => (1.0 - ax).max(0.0),
			Self::Bicubic => {
				// Mitchell-Netravali with B = 0, C = 0.6.
				let (b, c) = (0.0, 0.6);
				if ax < 1.0 {
					((12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax
						+ (-18.0 + 12.0 * b + 6.0 * c) * ax * ax
						+ (6.0 - 2.0 * b))
						/ 6.0
				} else if ax < 2.0 {
					((-b - 6.0 * c) * ax * ax * ax
						+ (6.0 * b + 30.0 * c) * ax * ax
						+ (-12.0 * b - 48.0 * c) * ax
						+ (8.0 * b + 24.0 * c))
						/ 6.0
				} else {
					0.0
				}
			}
			Self::Lanczos => {
				if ax < 1e-9 {
					1.0
				} else if ax < 3.0 {
					let px = std::f64::consts::PI * x;
					3.0 * px.sin() * (px / 3.0).sin() / (px * px)
				} else {
					0.0
				}
			}
		}
	}
}

/// YUV ↔ RGB coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMatrix {
	#[default]
	Bt601,
	Bt709,
	Bt2020,
}

impl ColorMatrix {
	/// `(Kr, Kb)`.
	fn k(self) -> (f32, f32) {
		match self {
			Self::Bt601 => (0.299, 0.114),
			Self::Bt709 => (0.2126, 0.0722),
			Self::Bt2020 => (0.2627, 0.0593),
		}
	}
}

/// YUV sample range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Range {
	/// Y in 16-235, chroma in 16-240 (8-bit units): "TV", MPEG range.
	#[default]
	Limited,
	/// 0-255: "PC", JPEG range.
	Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoOptions {
	pub filter: ScaleFilter,
	pub matrix: ColorMatrix,
	pub input_range: Range,
	pub output_range: Range,
}

/// One plane of float samples in 8-bit units.
#[derive(Debug, Clone)]
struct Plane {
	width: usize,
	height: usize,
	data: Vec<f32>,
}

impl Plane {
	fn filled(width: usize, height: usize, value: f32) -> Self {
		Self { width, height, data: vec![value; width * height] }
	}
}

/// Precomputed taps for one axis: per output index, the first source index
/// and normalised weights (edges clamped into range).
#[derive(Debug, Clone)]
struct Taps {
	taps: usize,
	start: Vec<usize>,
	weights: Vec<f32>,
	identity: bool,
}

impl Taps {
	fn new(src: usize, dst: usize, filter: ScaleFilter) -> Self {
		if src == dst {
			return Self { taps: 1, start: (0..dst).collect(), weights: vec![1.0; dst], identity: true };
		}
		let scale = src as f64 / dst as f64;
		let stretch = scale.max(1.0);
		let support = filter.radius() * stretch;
		let taps = ((support * 2.0).ceil() as usize + 1).min(src.max(1));
		let mut start = Vec::with_capacity(dst);
		let mut weights = Vec::with_capacity(dst * taps);
		for x in 0..dst {
			let center = (x as f64 + 0.5) * scale - 0.5;
			let first = (center - support).ceil() as i64;
			let mut w: Vec<f64> = (0..taps)
				.map(|k| filter.weight(((first + k as i64) as f64 - center) / stretch) / stretch)
				.collect();
			let sum: f64 = w.iter().sum();
			if sum.abs() > 1e-12 {
				w.iter_mut().for_each(|v| *v /= sum);
			}
			// Fold taps outside the image onto the edge samples.
			let max = src as i64 - 1;
			let lo = first.clamp(0, (max - taps as i64 + 1).max(0));
			let mut folded = vec![0f64; taps];
			for (k, &wk) in w.iter().enumerate() {
				let idx = (first + k as i64).clamp(0, max);
				let slot = (idx - lo).clamp(0, taps as i64 - 1) as usize;
				folded[slot] += wk;
			}
			start.push(lo as usize);
			weights.extend(folded.iter().map(|&v| v as f32));
		}
		Self { taps, start, weights, identity: false }
	}
}

fn scale_plane(p: &Plane, width: usize, height: usize, filter: ScaleFilter) -> Plane {
	if p.width == width && p.height == height {
		return p.clone();
	}
	let h = Taps::new(p.width, width, filter);
	let v = Taps::new(p.height, height, filter);
	// Horizontal pass: p.height rows of `width`.
	let mut tmp = vec![0f32; width * p.height];
	for y in 0..p.height {
		let row = &p.data[y * p.width..(y + 1) * p.width];
		let out = &mut tmp[y * width..(y + 1) * width];
		if h.identity {
			out.copy_from_slice(row);
			continue;
		}
		for (x, o) in out.iter_mut().enumerate() {
			let s = h.start[x];
			let w = &h.weights[x * h.taps..(x + 1) * h.taps];
			*o = w.iter().zip(&row[s..s + h.taps]).map(|(a, b)| a * b).sum();
		}
	}
	if v.identity {
		return Plane { width, height, data: tmp };
	}
	let mut out = vec![0f32; width * height];
	for y in 0..height {
		let s = v.start[y];
		let w = &v.weights[y * v.taps..(y + 1) * v.taps];
		let dst = &mut out[y * width..(y + 1) * width];
		for (k, &wk) in w.iter().enumerate() {
			let src = &tmp[(s + k) * width..(s + k + 1) * width];
			for (d, &sv) in dst.iter_mut().zip(src) {
				*d += wk * sv;
			}
		}
	}
	Plane { width, height, data: out }
}

/// Unpacked picture: YUV(A) or RGB(A) planes.
struct Unpacked {
	rgb: bool,
	/// Y, U, V or R, G, B.
	planes: Vec<Plane>,
	alpha: Option<Plane>,
}

fn chroma_dims(width: u32, height: u32, format: PixelFormat) -> (usize, usize) {
	let (sx, sy) = format.chroma_shift();
	((width as usize).div_ceil(1 << sx), (height as usize).div_ceil(1 << sy))
}

fn read_plane(data: &[u8], width: usize, height: usize, depth: u8) -> Plane {
	let n = width * height;
	let values: Vec<f32> = if depth > 8 {
		let scale = 1.0 / f32::from(1u16 << (depth - 8));
		data
			.chunks_exact(2)
			.take(n)
			.map(|c| f32::from(u16::from_le_bytes([c[0], c[1]])) * scale)
			.collect()
	} else {
		data.iter().take(n).map(|&b| f32::from(b)).collect()
	};
	Plane { width, height, data: values }
}

fn unpack(frame: &VideoFrame) -> Result<Unpacked> {
	let pixel = &frame.pixel;
	let (w, h) = (frame.width as usize, frame.height as usize);
	let sizes =
		pixel.plane_sizes(frame.width, frame.height).ok_or(Message::Other("picture too large"))?;
	let total: usize = sizes.iter().sum();
	if frame.data.len() < total {
		return Err(Message::Other("video frame is smaller than its format requires"));
	}
	let format = pixel.format;
	if format.is_rgb() {
		let bpp = format.packed_bytes();
		let (ri, gi, bi) = match format {
			PixelFormat::BGR24 | PixelFormat::BGRA => (2, 1, 0),
			_ => (0, 1, 2),
		};
		let mut planes =
			vec![Plane::filled(w, h, 0.0), Plane::filled(w, h, 0.0), Plane::filled(w, h, 0.0)];
		let mut alpha = format.has_alpha().then(|| Plane::filled(w, h, 0.0));
		for (i, px) in frame.data[..w * h * bpp].chunks_exact(bpp).enumerate() {
			planes[0].data[i] = f32::from(px[ri]);
			planes[1].data[i] = f32::from(px[gi]);
			planes[2].data[i] = f32::from(px[bi]);
			if let Some(a) = alpha.as_mut() {
				a.data[i] = f32::from(px[3]);
			}
		}
		return Ok(Unpacked { rgb: true, planes, alpha });
	}
	let (cw, ch) = chroma_dims(frame.width, frame.height, format);
	let mut at = 0usize;
	let mut take = |len: usize| {
		let slice = &frame.data[at..at + len];
		at += len;
		slice
	};
	let y = read_plane(take(sizes[0]), w, h, pixel.depth);
	let (u, v) = match format {
		PixelFormat::Gray => (Plane::filled(w, h, 128.0), Plane::filled(w, h, 128.0)),
		PixelFormat::NV12 => {
			let uv = take(sizes[1]);
			let mut u = Plane::filled(cw, ch, 0.0);
			let mut v = Plane::filled(cw, ch, 0.0);
			for (i, pair) in uv.chunks_exact(2).take(cw * ch).enumerate() {
				u.data[i] = f32::from(pair[0]);
				v.data[i] = f32::from(pair[1]);
			}
			(u, v)
		}
		_ => (
			read_plane(take(sizes[1]), cw, ch, pixel.depth),
			read_plane(take(sizes[2]), cw, ch, pixel.depth),
		),
	};
	let alpha = (format == PixelFormat::YUVA444).then(|| {
		let mut a = read_plane(take(sizes[3]), w, h, pixel.depth);
		// Alpha is full range: rescale codes to 0-255.
		let max = f32::from(((1u32 << pixel.depth) - 1) as u16) / f32::from(1u16 << (pixel.depth - 8));
		a.data.iter_mut().for_each(|v| *v *= 255.0 / max);
		a
	});
	Ok(Unpacked { rgb: false, planes: vec![y, u, v], alpha })
}

/// Normalised luma (0-1) and chroma (-0.5-0.5) from 8-bit units.
fn yuv_norm(range: Range) -> (f32, f32, f32) {
	match range {
		Range::Limited => (16.0, 219.0, 224.0),
		Range::Full => (0.0, 255.0, 255.0),
	}
}

fn yuv_to_rgb(planes: &mut [Plane], matrix: ColorMatrix, range: Range) {
	let (kr, kb) = matrix.k();
	let kg = 1.0 - kr - kb;
	let (y0, ys, cs) = yuv_norm(range);
	let (ry, gy, by) = (2.0 * (1.0 - kr), 2.0 * (1.0 - kb), 1.0 / kg);
	let n = planes[0].data.len();
	for i in 0..n {
		let y = (planes[0].data[i] - y0) / ys;
		let pb = (planes[1].data[i] - 128.0) / cs;
		let pr = (planes[2].data[i] - 128.0) / cs;
		let r = y + ry * pr;
		let b = y + gy * pb;
		let g = (y - kr * r - kb * b) * by;
		planes[0].data[i] = r * 255.0;
		planes[1].data[i] = g * 255.0;
		planes[2].data[i] = b * 255.0;
	}
}

fn rgb_to_yuv(planes: &mut [Plane], matrix: ColorMatrix, range: Range) {
	let (kr, kb) = matrix.k();
	let kg = 1.0 - kr - kb;
	let (y0, ys, cs) = yuv_norm(range);
	for i in 0..planes[0].data.len() {
		let (r, g, b) =
			(planes[0].data[i] / 255.0, planes[1].data[i] / 255.0, planes[2].data[i] / 255.0);
		let y = kr * r + kg * g + kb * b;
		let pb = (b - y) / (2.0 * (1.0 - kb));
		let pr = (r - y) / (2.0 * (1.0 - kr));
		planes[0].data[i] = y0 + y * ys;
		planes[1].data[i] = 128.0 + pb * cs;
		planes[2].data[i] = 128.0 + pr * cs;
	}
}

/// Re-expresses YUV planes from one range in another.
fn convert_range(planes: &mut [Plane], from: Range, to: Range) {
	if from == to {
		return;
	}
	let (fy0, fys, fcs) = yuv_norm(from);
	let (ty0, tys, tcs) = yuv_norm(to);
	for v in &mut planes[0].data {
		*v = ty0 + (*v - fy0) / fys * tys;
	}
	for plane in &mut planes[1..] {
		for v in &mut plane.data {
			*v = 128.0 + (*v - 128.0) / fcs * tcs;
		}
	}
}

fn write_plane(out: &mut Vec<u8>, p: &Plane, depth: u8) {
	if depth > 8 {
		let scale = f32::from(1u16 << (depth - 8));
		let max = f32::from(((1u32 << depth) - 1) as u16);
		for &v in &p.data {
			out.extend_from_slice(&((v * scale).round().clamp(0.0, max) as u16).to_le_bytes());
		}
	} else {
		out.extend(p.data.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8));
	}
}

/// Scales and converts pictures between two [`VideoSpec`]s.
#[derive(Debug, Clone)]
pub struct VideoConverter {
	input: VideoSpec,
	output: VideoSpec,
	options: VideoOptions,
}

impl VideoConverter {
	pub fn new(input: VideoSpec, output: VideoSpec, options: VideoOptions) -> Result<Self> {
		for spec in [&input, &output] {
			if spec.width == 0
				|| spec.height == 0
				|| spec.width > MAX_DIMENSION
				|| spec.height > MAX_DIMENSION
			{
				return Err(Message::Other("video converter: invalid picture size"));
			}
			if spec.pixel.frame_size(spec.width, spec.height).is_none() {
				return Err(Message::Other("video converter: picture too large"));
			}
		}
		Ok(Self { input, output, options })
	}

	pub fn input(&self) -> &VideoSpec {
		&self.input
	}

	pub fn output(&self) -> &VideoSpec {
		&self.output
	}

	pub fn is_identity(&self) -> bool {
		self.input == self.output && self.options.input_range == self.options.output_range
	}

	pub fn convert(&self, frame: &VideoFrame) -> Result<VideoFrame> {
		if VideoSpec::of(frame) != self.input {
			return Err(Message::Other("video converter: frame does not match the input spec"));
		}
		if self.is_identity() {
			return Ok(frame.clone());
		}
		let o = &self.options;
		let (ow, oh) = (self.output.width as usize, self.output.height as usize);
		let out_format = self.output.pixel.format;
		let src = unpack(frame)?;
		let scale = |p: &Plane, w: usize, h: usize| scale_plane(p, w, h, o.filter);

		let mut planes: Vec<Plane>;
		if out_format.is_rgb() {
			planes = src.planes.iter().map(|p| scale(p, ow, oh)).collect();
			if !src.rgb {
				yuv_to_rgb(&mut planes, o.matrix, o.input_range);
			}
		} else {
			let (cw, ch) = chroma_dims(self.output.width, self.output.height, out_format);
			if src.rgb {
				planes = src.planes.iter().map(|p| scale(p, ow, oh)).collect();
				rgb_to_yuv(&mut planes, o.matrix, o.output_range);
				planes[1] = scale(&planes[1], cw, ch);
				planes[2] = scale(&planes[2], cw, ch);
			} else {
				planes = vec![
					scale(&src.planes[0], ow, oh),
					scale(&src.planes[1], cw, ch),
					scale(&src.planes[2], cw, ch),
				];
				convert_range(&mut planes, o.input_range, o.output_range);
			}
		}
		let alpha = out_format.has_alpha().then(|| match &src.alpha {
			Some(a) => scale(a, ow, oh),
			None => Plane::filled(ow, oh, 255.0),
		});

		let size = self.output.pixel.frame_size(self.output.width, self.output.height).unwrap_or(0);
		let mut data = Vec::with_capacity(size);
		let depth = self.output.pixel.depth;
		match out_format {
			f if f.is_rgb() => {
				let order: [usize; 3] =
					if matches!(f, PixelFormat::BGR24 | PixelFormat::BGRA) { [2, 1, 0] } else { [0, 1, 2] };
				for i in 0..ow * oh {
					for &c in &order {
						data.push(planes[c].data[i].round().clamp(0.0, 255.0) as u8);
					}
					if let Some(a) = &alpha {
						data.push(a.data[i].round().clamp(0.0, 255.0) as u8);
					}
				}
			}
			PixelFormat::Gray => write_plane(&mut data, &planes[0], depth),
			PixelFormat::NV12 => {
				write_plane(&mut data, &planes[0], 8);
				for (u, v) in planes[1].data.iter().zip(&planes[2].data) {
					data.push(u.round().clamp(0.0, 255.0) as u8);
					data.push(v.round().clamp(0.0, 255.0) as u8);
				}
			}
			_ => {
				for p in &planes {
					write_plane(&mut data, p, depth);
				}
				if let Some(mut a) = alpha {
					let max = f32::from(((1u32 << depth) - 1) as u16) / f32::from(1u16 << (depth - 8));
					a.data.iter_mut().for_each(|v| *v *= max / 255.0);
					write_plane(&mut data, &a, depth);
				}
			}
		}
		Ok(VideoFrame {
			data,
			width: self.output.width,
			height: self.output.height,
			pixel: self.output.pixel.clone(),
			keyframe: frame.keyframe,
			pts: frame.pts,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::core::Keyframe;

	fn frame(w: u32, h: u32, pixel: &str, data: Vec<u8>) -> VideoFrame {
		VideoFrame {
			data,
			width: w,
			height: h,
			pixel: Pixel::from_name(pixel).unwrap(),
			keyframe: Keyframe::Key,
			pts: None,
		}
	}

	fn convert(f: &VideoFrame, w: u32, h: u32, pixel: &str) -> VideoFrame {
		let out = VideoSpec::new(w, h, Pixel::from_name(pixel).unwrap());
		VideoConverter::new(VideoSpec::of(f), out, VideoOptions::default()).unwrap().convert(f).unwrap()
	}

	#[test]
	fn primaries_round_trip_through_yuv() {
		// Pure red, green, blue, white, black in BT.601 limited range.
		let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 128, 128, 128];
		let f = frame(6, 1, "rgb24", rgb.clone());
		let yuv = convert(&f, 6, 1, "yuv444p");
		assert_eq!(&yuv.data[..6], &[81, 145, 41, 235, 16, 126]);
		let back = convert(&yuv, 6, 1, "rgb24");
		for (a, b) in back.data.iter().zip(&rgb) {
			assert!((i16::from(*a) - i16::from(*b)).abs() <= 2, "{back:?}");
		}
	}

	#[test]
	fn depth_and_layout_changes() {
		let f = frame(2, 2, "yuv420p", vec![16, 235, 128, 64, 100, 200]);
		let ten = convert(&f, 2, 2, "yuv420p10le");
		let words: Vec<u16> = ten.data.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
		assert_eq!(words, vec![64, 940, 512, 256, 400, 800]);
		assert_eq!(convert(&ten, 2, 2, "yuv420p").data, f.data);
		let nv12 = convert(&f, 2, 2, "nv12");
		assert_eq!(nv12.data, vec![16, 235, 128, 64, 100, 200]);
		let gray = convert(&f, 2, 2, "gray");
		assert_eq!(gray.data, vec![16, 235, 128, 64]);
		let yuv444 = convert(&f, 2, 2, "yuv444p");
		assert_eq!(&yuv444.data[4..], &[100, 100, 100, 100, 200, 200, 200, 200]);
	}

	#[test]
	fn scaling_preserves_flat_areas_and_gradients() {
		let w = 16;
		let data: Vec<u8> = (0..w * w).map(|i| (i % w * 10) as u8).collect();
		let mut g = frame(w as u32, w as u32, "gray", data);
		g.pts = Some(crate::core::Time::new(7, crate::core::TimeBase::new(1, 25)));
		for filter in
			[ScaleFilter::Point, ScaleFilter::Bilinear, ScaleFilter::Bicubic, ScaleFilter::Lanczos]
		{
			let options = VideoOptions { filter, ..Default::default() };
			let up =
				VideoConverter::new(VideoSpec::of(&g), VideoSpec::new(32, 32, g.pixel.clone()), options)
					.unwrap();
			let big = up.convert(&g).unwrap();
			assert_eq!(big.pts, g.pts);
			// Rows are identical (no vertical variation in the source).
			assert_eq!(&big.data[..32], &big.data[32 * 10..32 * 11]);
			let down =
				VideoConverter::new(VideoSpec::of(&big), VideoSpec::new(16, 16, g.pixel.clone()), options)
					.unwrap();
			let small = down.convert(&big).unwrap();
			let err: i32 = small
				.data
				.iter()
				.zip(&g.data)
				.map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
				.max()
				.unwrap();
			assert!(err <= 6, "{filter:?}: {err}");
		}
	}

	#[test]
	fn rejects_mismatched_frames() {
		let f = frame(2, 2, "gray", vec![0; 4]);
		let c = VideoConverter::new(
			VideoSpec::new(4, 4, Pixel::from_name("gray").unwrap()),
			VideoSpec::new(2, 2, Pixel::from_name("gray").unwrap()),
			VideoOptions::default(),
		)
		.unwrap();
		assert!(c.convert(&f).is_err());
		assert!(
			VideoConverter::new(
				VideoSpec::new(0, 2, f.pixel.clone()),
				VideoSpec::of(&f),
				VideoOptions::default()
			)
			.is_err()
		);
	}
}
