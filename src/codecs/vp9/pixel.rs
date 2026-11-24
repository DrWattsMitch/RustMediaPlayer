//! Sample types. 8-bit streams (profiles 0 and 1) decode into `u8` planes,
//! 10- and 12-bit streams (profiles 2 and 3) into `u16` planes. Everything
//! that touches samples (prediction, reconstruction, the loop filter, tile
//! threading) is generic over [`Px`] and monomorphised for both, so the
//! 8-bit path keeps its vector kernels and pays nothing for the wide one.
//!
//! The high-bit-depth arithmetic is libvpx's `highbd` code: the same
//! filters and transforms, with clipping to `(1 << bit_depth) - 1`, intra
//! edge defaults of `128 << (bit_depth - 8)` (± 1), and loop-filter
//! thresholds shifted left by `bit_depth - 8`.

use std::fmt::Debug;
use std::sync::Arc;

use super::decoder::{DecodedPicture, Picture, Plane, TilePool, Vp9DecoderState};
use super::inter::{self, RefPlane, Scratch};
use super::loopfilter::{self, Thresholds};
use crate::core::VecPool;

/// A sample type: `u8` (8-bit video) or `u16` (10 and 12 bits).
pub trait Px: Copy + Default + Send + Sync + Debug + PartialEq + Into<i32> + 'static {
	/// `true` for `u16`.
	const WIDE: bool;
	/// `v`, known to be in range.
	fn of(v: i32) -> Self;
	/// `v` clipped to `0..=max` (`max` is ignored for `u8`: always 255).
	fn clip(v: i32, max: i32) -> Self;
	/// The same samples as bytes, if this is the 8-bit type (lets generic
	/// code use the 8-bit vector routines).
	fn narrow_mut(s: &mut [Self]) -> Option<&mut [u8]>;

	/// Inter prediction ([`inter::predict_scalar`]; vector code for `u8`).
	#[allow(clippy::too_many_arguments)]
	fn predict(
		scratch: &mut Scratch<Self>,
		src: &RefPlane<Self>,
		dst: &mut [Self],
		dst_stride: usize,
		x: usize,
		y: usize,
		w: usize,
		h: usize,
		mv_row: i32,
		mv_col: i32,
		filter: u8,
		avg: bool,
		max: i32,
	);
	/// One 8-pixel stretch of a vertical / horizontal loop-filter edge
	/// (vector code for `u8`). `shift` is `bit_depth - 8`.
	fn lf_vertical(plane: &mut Plane<Self>, x: usize, y: usize, t: &Thresholds, taps: u8, shift: u32);
	fn lf_horizontal(
		plane: &mut Plane<Self>,
		x: usize,
		y: usize,
		t: &Thresholds,
		taps: u8,
		shift: u32,
	);

	/// The picture in the bit-depth-erased form kept in reference slots.
	fn wrap(p: Arc<Picture<Self>>) -> DecodedPicture;
	/// The picture, if it has this sample type.
	fn unwrap(p: &DecodedPicture) -> Option<&Arc<Picture<Self>>>;
	/// The decoder's sample pools of this type: picture planes and tile
	/// workspaces.
	fn pools(s: &mut Vp9DecoderState) -> (&VecPool<Self>, &mut TilePool<Self>);
}

impl Px for u8 {
	const WIDE: bool = false;
	#[inline(always)]
	fn of(v: i32) -> u8 {
		v as u8
	}
	#[inline(always)]
	fn clip(v: i32, _max: i32) -> u8 {
		v.clamp(0, 255) as u8
	}
	#[inline(always)]
	fn narrow_mut(s: &mut [u8]) -> Option<&mut [u8]> {
		Some(s)
	}

	#[inline]
	fn predict(
		scratch: &mut Scratch<u8>,
		src: &RefPlane<u8>,
		dst: &mut [u8],
		dst_stride: usize,
		x: usize,
		y: usize,
		w: usize,
		h: usize,
		mv_row: i32,
		mv_col: i32,
		filter: u8,
		avg: bool,
		max: i32,
	) {
		if crate::simd::enabled() && (w == 4 || w.is_multiple_of(8)) {
			super::simd::predict(scratch, src, dst, dst_stride, x, y, w, h, mv_row, mv_col, filter, avg);
		} else {
			inter::predict_scalar(
				scratch, src, dst, dst_stride, x, y, w, h, mv_row, mv_col, filter, avg, max,
			);
		}
	}

	#[inline]
	fn lf_vertical(plane: &mut Plane<u8>, x: usize, y: usize, t: &Thresholds, taps: u8, shift: u32) {
		if crate::simd::enabled() {
			super::simd::vertical(plane, x, y, t, taps);
		} else {
			loopfilter::vertical_scalar(plane, x, y, t, taps, shift);
		}
	}

	#[inline]
	fn lf_horizontal(
		plane: &mut Plane<u8>,
		x: usize,
		y: usize,
		t: &Thresholds,
		taps: u8,
		shift: u32,
	) {
		if crate::simd::enabled() {
			super::simd::horizontal(plane, x, y, t, taps);
		} else {
			loopfilter::horizontal_scalar(plane, x, y, t, taps, shift);
		}
	}

	fn wrap(p: Arc<Picture<u8>>) -> DecodedPicture {
		DecodedPicture::Narrow(p)
	}
	fn unwrap(p: &DecodedPicture) -> Option<&Arc<Picture<u8>>> {
		match p {
			DecodedPicture::Narrow(p) => Some(p),
			DecodedPicture::Wide(_) => None,
		}
	}
	fn pools(s: &mut Vp9DecoderState) -> (&VecPool<u8>, &mut TilePool<u8>) {
		(&s.planes, &mut s.tile_pool)
	}
}

impl Px for u16 {
	const WIDE: bool = true;
	#[inline(always)]
	fn of(v: i32) -> u16 {
		v as u16
	}
	#[inline(always)]
	fn clip(v: i32, max: i32) -> u16 {
		v.clamp(0, max) as u16
	}
	#[inline(always)]
	fn narrow_mut(_: &mut [u16]) -> Option<&mut [u8]> {
		None
	}

	#[inline]
	fn predict(
		scratch: &mut Scratch<u16>,
		src: &RefPlane<u16>,
		dst: &mut [u16],
		dst_stride: usize,
		x: usize,
		y: usize,
		w: usize,
		h: usize,
		mv_row: i32,
		mv_col: i32,
		filter: u8,
		avg: bool,
		max: i32,
	) {
		inter::predict_scalar(
			scratch, src, dst, dst_stride, x, y, w, h, mv_row, mv_col, filter, avg, max,
		);
	}

	#[inline]
	fn lf_vertical(plane: &mut Plane<u16>, x: usize, y: usize, t: &Thresholds, taps: u8, shift: u32) {
		loopfilter::vertical_scalar(plane, x, y, t, taps, shift);
	}

	#[inline]
	fn lf_horizontal(
		plane: &mut Plane<u16>,
		x: usize,
		y: usize,
		t: &Thresholds,
		taps: u8,
		shift: u32,
	) {
		loopfilter::horizontal_scalar(plane, x, y, t, taps, shift);
	}

	fn wrap(p: Arc<Picture<u16>>) -> DecodedPicture {
		DecodedPicture::Wide(p)
	}
	fn unwrap(p: &DecodedPicture) -> Option<&Arc<Picture<u16>>> {
		match p {
			DecodedPicture::Wide(p) => Some(p),
			DecodedPicture::Narrow(_) => None,
		}
	}
	fn pools(s: &mut Vp9DecoderState) -> (&VecPool<u16>, &mut TilePool<u16>) {
		(&s.planes_wide, &mut s.tile_pool_wide)
	}
}
