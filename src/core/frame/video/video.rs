//! Decoded pictures: reference-counted planes with strides (FFmpeg's
//! `AVFrame` with `buf[]`, `data[]` and `linesize[]`).
//!
//! A [`VideoFrame`] does not own a `Vec`. Each plane is a [`PlaneRef`]: a
//! [`FrameBuffer`] plus an offset, a stride and the visible row size. That
//! lets decoders return their own (padded, strided) reference pictures
//! without copying or cropping them, makes `clone` an atomic increment, and
//! lets a player hand each plane to the GPU as it is
//! (`bytes()` + `stride()` is exactly what `glTexSubImage2D` with
//! `GL_UNPACK_ROW_LENGTH`, `wgpu::Queue::write_texture`, Vulkan
//! `vkCmdCopyBufferToImage` or Metal `replaceRegion:bytesPerRow:` take).
//!
//! Code that wants the classic tightly packed planar layout (raw `.yuv`,
//! Y4M) uses [`VideoFrame::data`], [`VideoFrame::into_packed`] or
//! [`VideoFrame::write_packed`]; a frame that was *built* from packed bytes
//! (raw video, the scaler) gives them back without copying.

use std::borrow::Cow;
use std::io;

use super::format::Pixel;
use crate::core::frame::buffer::FrameBuffer;
use crate::core::time::Time;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyframe {
	Key,
	NonKey,
}

/// Most planes a pixel format has (YUVA).
pub const MAX_PLANES: usize = 4;

/// One plane of a picture: `rows` rows of `row_bytes` bytes, `stride` bytes
/// apart, starting `offset` bytes into `buffer`.
#[derive(Debug, Clone)]
pub struct PlaneRef {
	buffer: FrameBuffer,
	offset: usize,
	stride: usize,
	row_bytes: usize,
	rows: usize,
}

impl PlaneRef {
	/// Returns `None` if the rows don't fit in the buffer or overlap
	/// (`row_bytes > stride`).
	pub fn new(
		buffer: FrameBuffer,
		offset: usize,
		stride: usize,
		row_bytes: usize,
		rows: usize,
	) -> Option<Self> {
		if rows > 1 && row_bytes > stride {
			return None;
		}
		let span = Self::span_of(stride, row_bytes, rows)?;
		if offset.checked_add(span)? > buffer.len() {
			return None;
		}
		Some(Self { buffer, offset, stride, row_bytes, rows })
	}

	fn span_of(stride: usize, row_bytes: usize, rows: usize) -> Option<usize> {
		match rows {
			0 => Some(0),
			n => stride.checked_mul(n - 1)?.checked_add(row_bytes),
		}
	}

	/// The buffer this plane lives in. Clone it to keep the plane alive
	/// (while a GPU upload is in flight, say).
	pub fn buffer(&self) -> &FrameBuffer {
		&self.buffer
	}

	/// Byte offset of the first row in [`Self::buffer`].
	pub fn offset(&self) -> usize {
		self.offset
	}

	/// Distance between rows in bytes (`linesize`). At least
	/// [`Self::row_bytes`]; decoders usually pad it to 16 or 64.
	pub fn stride(&self) -> usize {
		self.stride
	}

	/// Visible bytes per row.
	pub fn row_bytes(&self) -> usize {
		self.row_bytes
	}

	pub fn rows(&self) -> usize {
		self.rows
	}

	/// The plane from its first visible byte to its last, padding between
	/// rows included: pass this with [`Self::stride`] to a GPU upload.
	pub fn bytes(&self) -> &[u8] {
		let span = Self::span_of(self.stride, self.row_bytes, self.rows).unwrap_or(0);
		&self.buffer[self.offset..self.offset + span]
	}

	/// Visible bytes of row `y`.
	#[inline]
	pub fn row(&self, y: usize) -> &[u8] {
		let at = self.offset + y * self.stride;
		&self.buffer[at..at + self.row_bytes]
	}

	/// The visible rows, top to bottom.
	pub fn rows_iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + '_ {
		let data = self.bytes();
		(0..self.rows).map(move |y| &data[y * self.stride..y * self.stride + self.row_bytes])
	}

	/// `true` if rows are stored back to back (no padding).
	pub fn is_contiguous(&self) -> bool {
		self.rows <= 1 || self.stride == self.row_bytes
	}

	/// `true` if the first byte and the stride are multiples of `align`
	/// (some GPU copy paths need 4, 64 or 256).
	pub fn is_aligned(&self, align: usize) -> bool {
		let align = align.max(1);
		(self.bytes().as_ptr() as usize).is_multiple_of(align) && self.stride.is_multiple_of(align)
	}
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
	pub width: u32,
	pub height: u32,
	pub pixel: Pixel,
	pub keyframe: Keyframe,
	pub pts: Option<Time>,
	planes: [Option<PlaneRef>; MAX_PLANES],
	/// The packed bytes the frame was built from, if it was.
	packed: Option<FrameBuffer>,
}

impl VideoFrame {
	/// A frame over tightly packed planar bytes (Y, then U, then V...;
	/// FFmpeg's `rawvideo` layout). Never copies. If `data` is shorter than
	/// the format needs, the planes cover what is there and
	/// [`Self::is_complete`] is `false`.
	pub fn from_packed(
		data: impl Into<FrameBuffer>,
		width: u32,
		height: u32,
		pixel: Pixel,
		keyframe: Keyframe,
		pts: Option<Time>,
	) -> Self {
		let data = data.into();
		let mut planes: [Option<PlaneRef>; MAX_PLANES] = Default::default();
		let mut at = 0usize;
		for (slot, (row_bytes, rows)) in
			planes.iter_mut().zip(pixel.plane_dims(width, height).unwrap_or_default())
		{
			let avail = data.len().saturating_sub(at);
			let rows = avail.checked_div(row_bytes).map_or(rows, |fit| rows.min(fit));
			*slot = PlaneRef::new(data.clone(), at, row_bytes, row_bytes, rows);
			at = at.saturating_add(row_bytes * rows);
		}
		Self { width, height, pixel, keyframe, pts, planes, packed: Some(data) }
	}

	/// A frame over existing planes (a decoder's reference picture, a GPU
	/// readback...). Never copies. Extra planes beyond [`MAX_PLANES`] are
	/// ignored.
	pub fn from_planes(
		planes: impl IntoIterator<Item = PlaneRef>,
		width: u32,
		height: u32,
		pixel: Pixel,
		keyframe: Keyframe,
		pts: Option<Time>,
	) -> Self {
		let mut slots: [Option<PlaneRef>; MAX_PLANES] = Default::default();
		for (slot, plane) in slots.iter_mut().zip(planes) {
			*slot = Some(plane);
		}
		Self { width, height, pixel, keyframe, pts, planes: slots, packed: None }
	}

	pub fn plane_count(&self) -> usize {
		self.planes.iter().take_while(|p| p.is_some()).count()
	}

	pub fn plane(&self, index: usize) -> Option<&PlaneRef> {
		self.planes.get(index)?.as_ref()
	}

	pub fn planes(&self) -> impl Iterator<Item = &PlaneRef> {
		self.planes.iter().map_while(Option::as_ref)
	}

	/// `true` if every plane the pixel format needs is present at full size.
	pub fn is_complete(&self) -> bool {
		let Some(dims) = self.pixel.plane_dims(self.width, self.height) else {
			return false;
		};
		dims.len() == self.plane_count()
			&& dims.iter().zip(self.planes()).all(|(&(w, h), p)| p.row_bytes >= w && p.rows >= h)
	}

	/// Size of the packed planar layout of the visible picture.
	pub fn packed_len(&self) -> usize {
		match &self.packed {
			Some(b) => b.len(),
			None => self.planes().map(|p| p.row_bytes * p.rows).sum(),
		}
	}

	/// The picture as packed planar bytes: borrowed if the frame was built
	/// from packed bytes, otherwise copied row by row.
	pub fn data(&self) -> Cow<'_, [u8]> {
		match &self.packed {
			Some(b) => Cow::Borrowed(b.as_slice()),
			None => {
				let mut out = Vec::with_capacity(self.packed_len());
				for p in self.planes() {
					for row in p.rows_iter() {
						out.extend_from_slice(row);
					}
				}
				Cow::Owned(out)
			}
		}
	}

	/// Like [`Self::data`], as an owned `Vec`: free when this frame is the
	/// only reference to packed bytes.
	pub fn into_packed(self) -> Vec<u8> {
		match self.packed {
			Some(b) if self.planes.iter().flatten().all(|p| p.buffer.ptr_eq(&b)) => {
				// Drop the planes' references first so `b` may be unique.
				drop(self.planes);
				b.into_vec()
			}
			_ => self.data().into_owned(),
		}
	}

	/// Writes the packed planar bytes without building them in memory.
	pub fn write_packed(&self, out: &mut impl io::Write) -> io::Result<()> {
		if let Some(b) = &self.packed {
			return out.write_all(b);
		}
		for p in self.planes() {
			if p.is_contiguous() {
				out.write_all(&p.bytes()[..p.row_bytes * p.rows])?;
			} else {
				for row in p.rows_iter() {
					out.write_all(row)?;
				}
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::core::{BufferPool, PixelFormat};

	fn yuv420() -> Pixel {
		Pixel { depth: 8, format: PixelFormat::YUV420 }
	}

	#[test]
	fn packed_frames_round_trip_without_copies() {
		let data: Vec<u8> = (0..24).collect(); // 4x4 4:2:0
		let ptr = data.as_ptr();
		let f = VideoFrame::from_packed(data, 4, 4, yuv420(), Keyframe::Key, None);
		assert!(f.is_complete());
		assert_eq!(f.plane_count(), 3);
		assert_eq!(f.plane(1).unwrap().row(1), &[18, 19]);
		assert!(matches!(f.data(), Cow::Borrowed(_)));
		let copy = f.clone();
		assert!(copy.plane(0).unwrap().buffer().ptr_eq(f.plane(0).unwrap().buffer()));
		drop(copy);
		let back = f.into_packed();
		assert_eq!(back.as_ptr(), ptr);
	}

	#[test]
	fn short_packed_data_is_incomplete() {
		let f = VideoFrame::from_packed(vec![0u8; 20], 4, 4, yuv420(), Keyframe::Key, None);
		assert!(!f.is_complete());
		assert_eq!(f.data().len(), 20);
		assert_eq!(f.plane(2).unwrap().rows(), 0);
	}

	#[test]
	fn strided_planes_pack_their_visible_rows() {
		// A 2x2 luma plane inside a 4-byte stride, offset by one row + one byte.
		let luma = FrameBuffer::from_vec((0..16).collect());
		let chroma = FrameBuffer::from_vec(vec![100, 200]);
		let planes = [
			PlaneRef::new(luma, 5, 4, 2, 2).unwrap(),
			PlaneRef::new(chroma.clone(), 0, 1, 1, 1).unwrap(),
			PlaneRef::new(chroma, 1, 1, 1, 1).unwrap(),
		];
		let f = VideoFrame::from_planes(planes, 2, 2, yuv420(), Keyframe::Key, None);
		assert!(f.is_complete());
		assert_eq!(f.plane(0).unwrap().bytes(), &[5, 6, 7, 8, 9, 10]);
		assert_eq!(&f.data()[..], &[5, 6, 9, 10, 100, 200]);
		let mut out = Vec::new();
		f.write_packed(&mut out).unwrap();
		assert_eq!(out, vec![5, 6, 9, 10, 100, 200]);
		assert!(PlaneRef::new(FrameBuffer::from_vec(vec![0; 7]), 0, 4, 4, 2).is_none());
	}

	#[test]
	fn pooled_planes_return_when_the_frame_goes() {
		let pool = BufferPool::new(4);
		let buf = pool.take_filled(24, 1).freeze();
		let f = VideoFrame::from_packed(buf, 4, 4, yuv420(), Keyframe::Key, None);
		let shown = f.clone();
		drop(f);
		assert_eq!(pool.stats().idle, 0);
		drop(shown);
		assert_eq!(pool.stats().idle, 1);
	}
}
