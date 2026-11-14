//! The only `unsafe` in the AV1 decoder: a minimal safe wrapper over
//! rav1d's dav1d-compatible C API (`rav1d` is the Rust port of dav1d; its
//! public interface is still the `extern "C"` dav1d functions).
//!
//! Invariants upheld here:
//! * a [`Context`] owns exactly one `Dav1dContext` from `dav1d_open`,
//!   released once in `Drop` via `dav1d_close`;
//! * every `Dav1dData` passed to `dav1d_send_data` comes from
//!   `dav1d_data_create` (dav1d owns the buffer; we only fill it);
//! * every picture from `dav1d_get_picture` is copied into owned memory
//!   and released with `dav1d_picture_unref` before returning, so no
//!   pointer into dav1d memory outlives a call.
#![allow(unsafe_code)]

use std::mem::MaybeUninit;
use std::ptr::NonNull;

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::{
	DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422,
	DAV1D_PIXEL_LAYOUT_I444,
};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
	dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_flush,
	dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

const EAGAIN: i32 = -libc::EAGAIN;

/// Chroma layout of a decoded picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
	I400,
	I420,
	I422,
	I444,
}

/// One decoded picture, copied out of dav1d. Samples deeper than 8 bits
/// are stored as little-endian `u16`.
#[derive(Debug, Clone)]
pub struct Picture {
	pub width: usize,
	pub height: usize,
	pub bit_depth: u8,
	pub layout: Layout,
	/// Y, U, V tightly packed one after the other.
	pub data: Vec<u8>,
	pub keyframe: bool,
	/// Presentation timestamp handed in with the data, if any.
	pub timestamp: i64,
}

pub struct Context {
	ctx: Dav1dContext,
}

// SAFETY: `Rav1dContext` is `Send + Sync` (rav1d marks it so; it is used
// from its own worker threads). `Context` hands the raw handle to no one.
unsafe impl Send for Context {}

impl Context {
	/// Opens a decoder with `threads` worker threads (0 = one per core).
	pub fn open(threads: usize) -> Option<Self> {
		let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
		// SAFETY: `dav1d_default_settings` fully initialises the struct.
		let mut settings = unsafe {
			dav1d_default_settings(NonNull::new_unchecked(settings.as_mut_ptr()));
			settings.assume_init()
		};
		settings.n_threads = threads.min(256) as i32;
		let mut out: Option<Dav1dContext> = None;
		// SAFETY: both pointers come from live locals.
		let r =
			unsafe { dav1d_open(Some(NonNull::from(&mut out)), Some(NonNull::from(&mut settings))) };
		if r.0 != 0 {
			return None;
		}
		out.map(|ctx| Self { ctx })
	}

	/// Queues one temporal unit (and its timestamp) and returns every
	/// picture that became available.
	pub fn decode(&mut self, bytes: &[u8], timestamp: i64) -> Result<Vec<Picture>, i32> {
		let mut pictures = Vec::new();
		if bytes.is_empty() {
			return Ok(pictures);
		}
		let mut data = Dav1dData::default();
		// SAFETY: `data` is a live local; on success the returned pointer is
		// a dav1d-owned buffer of exactly `bytes.len()` bytes.
		let buf = unsafe { dav1d_data_create(Some(NonNull::from(&mut data)), bytes.len()) };
		if buf.is_null() {
			return Err(-libc::ENOMEM);
		}
		// SAFETY: `buf` points to `bytes.len()` writable bytes (see above).
		unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len()) };
		data.m.timestamp = timestamp;
		loop {
			// SAFETY: `self.ctx` is open; `data` came from `dav1d_data_create`.
			let r = unsafe { dav1d_send_data(Some(self.ctx), Some(NonNull::from(&mut data))) };
			if r.0 != 0 && r.0 != EAGAIN {
				// SAFETY: releases the data reference we still hold.
				unsafe { dav1d_data_unref(Some(NonNull::from(&mut data))) };
				return Err(r.0);
			}
			self.drain(&mut pictures)?;
			if data.sz == 0 {
				break;
			}
		}
		Ok(pictures)
	}

	/// Returns the pictures still held back for reordering/threading.
	pub fn flush_out(&mut self) -> Result<Vec<Picture>, i32> {
		let mut pictures = Vec::new();
		// Without new data, dav1d drains its frame-thread queue.
		for _ in 0..64 {
			let before = pictures.len();
			self.drain(&mut pictures)?;
			if pictures.len() == before {
				break;
			}
		}
		Ok(pictures)
	}

	/// Discards all state (after a seek).
	pub fn flush(&mut self) {
		// SAFETY: `self.ctx` is open.
		unsafe { dav1d_flush(self.ctx) };
	}

	fn drain(&mut self, out: &mut Vec<Picture>) -> Result<(), i32> {
		loop {
			let mut pic = MaybeUninit::<Dav1dPicture>::zeroed();
			// SAFETY: `self.ctx` is open; `pic` is writable.
			let r = unsafe {
				dav1d_get_picture(Some(self.ctx), Some(NonNull::new_unchecked(pic.as_mut_ptr())))
			};
			if r.0 == EAGAIN {
				return Ok(());
			}
			if r.0 != 0 {
				return Err(r.0);
			}
			// SAFETY: success means dav1d wrote a valid picture.
			let mut pic = unsafe { pic.assume_init() };
			let copied = copy_picture(&pic);
			// SAFETY: `pic` holds a reference from `dav1d_get_picture`.
			unsafe { dav1d_picture_unref(Some(NonNull::from(&mut pic))) };
			out.push(copied);
		}
	}
}

fn copy_picture(pic: &Dav1dPicture) -> Picture {
	let width = pic.p.w.max(0) as usize;
	let height = pic.p.h.max(0) as usize;
	let bit_depth = pic.p.bpc as u8;
	let bytes = if bit_depth > 8 { 2 } else { 1 };
	let layout = match pic.p.layout {
		DAV1D_PIXEL_LAYOUT_I400 => Layout::I400,
		DAV1D_PIXEL_LAYOUT_I420 => Layout::I420,
		DAV1D_PIXEL_LAYOUT_I422 => Layout::I422,
		DAV1D_PIXEL_LAYOUT_I444 => Layout::I444,
		_ => Layout::I420,
	};
	let (cw, ch) = match layout {
		Layout::I400 => (0, 0),
		Layout::I420 => (width.div_ceil(2), height.div_ceil(2)),
		Layout::I422 => (width.div_ceil(2), height),
		Layout::I444 => (width, height),
	};
	let mut data = Vec::with_capacity((width * height + 2 * cw * ch) * bytes);
	let planes = [(0usize, width, height), (1, cw, ch), (2, cw, ch)];
	for (p, w, h) in planes {
		if w == 0 || h == 0 {
			continue;
		}
		let stride = pic.stride[usize::from(p > 0)];
		let Some(base) = pic.data[p] else { continue };
		let row_bytes = w * bytes;
		for y in 0..h {
			// SAFETY: dav1d guarantees each plane holds `h` rows of at least
			// `w` samples at `stride` bytes apart (stride may be negative).
			let row = unsafe {
				let start = base.as_ptr().cast::<u8>().offset(y as isize * stride);
				std::slice::from_raw_parts(start, row_bytes)
			};
			data.extend_from_slice(row);
		}
	}
	let keyframe = pic.frame_hdr.is_some_and(|h| {
		// SAFETY: the frame header lives as long as the picture reference.
		let hdr = unsafe { h.as_ref() };
		hdr.frame_type == rav1d::include::dav1d::headers::DAV1D_FRAME_TYPE_KEY
	});
	Picture { width, height, bit_depth, layout, data, keyframe, timestamp: pic.m.timestamp }
}

impl Drop for Context {
	fn drop(&mut self) {
		let mut ctx = Some(self.ctx);
		// SAFETY: `self.ctx` came from `dav1d_open` and is closed only here.
		unsafe { dav1d_close(Some(NonNull::from(&mut ctx))) };
	}
}
