//! Sample-format conversion for interleaved and planar audio.
//!
//! The rules match FFmpeg's `libswresample/audioconvert.c`, so a frame
//! converted here equals one converted by `swr_convert` without dithering:
//!
//! - integer to wider integer shifts left (`s16 -> s32` is `x << 16`);
//! - integer to narrower integer truncates (`s32 -> s16` is `x >> 16`);
//! - integer to float divides by `2^(bits-1)`;
//! - float to integer multiplies by `2^(bits-1)`, rounds half to even
//!   (`lrint`) and saturates;
//! - `u8` is offset binary, silence at 128.

use crate::Message;
use crate::core::{AudioFrame, SampleFormat};
use crate::message::Result;

/// One sample, in the widest form of its kind: integers left-aligned in 64
/// bits, floats as `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sample {
	Int(i64),
	Float(f64),
}

impl Sample {
	/// Reads the sample at the start of `bytes` (native little-endian layout,
	/// `bytes.len() >= format.bytes()`).
	fn read(format: SampleFormat, bytes: &[u8]) -> Option<Sample> {
		Some(match format {
			SampleFormat::U8 => Sample::Int(i64::from(*bytes.first()? as i8 ^ i8::MIN) << 56),
			SampleFormat::S16 => Sample::Int(i64::from(i16::from_le_bytes(array(bytes)?)) << 48),
			SampleFormat::S32 => Sample::Int(i64::from(i32::from_le_bytes(array(bytes)?)) << 32),
			SampleFormat::S64 => Sample::Int(i64::from_le_bytes(array(bytes)?)),
			SampleFormat::F32 => Sample::Float(f64::from(f32::from_le_bytes(array(bytes)?))),
			SampleFormat::F64 => Sample::Float(f64::from_le_bytes(array(bytes)?)),
		})
	}

	/// The sample as a signed integer of `bits` bits (8, 16, 32 or 64).
	pub fn to_int(self, bits: u32) -> i64 {
		match self {
			Sample::Int(v) => v >> (64 - bits),
			Sample::Float(x) => {
				let scale = (1u64 << (bits - 1)) as f64;
				let max = if bits == 64 { i64::MAX } else { (1i64 << (bits - 1)) - 1 };
				let min = if bits == 64 { i64::MIN } else { -(1i64 << (bits - 1)) };
				let v = (x * scale).round_ties_even();
				// NaN saturates to zero, like a cast; infinities clip.
				if v.is_nan() {
					0
				} else if v >= max as f64 {
					max
				} else if v <= min as f64 {
					min
				} else {
					v as i64
				}
			}
		}
	}

	pub fn to_float(self) -> f64 {
		match self {
			Sample::Int(v) => v as f64 / 9_223_372_036_854_775_808.0,
			Sample::Float(x) => x,
		}
	}

	pub fn to_i16(self) -> i16 {
		self.to_int(16) as i16
	}

	pub fn to_i32(self) -> i32 {
		self.to_int(32) as i32
	}
}

fn array<const N: usize>(bytes: &[u8]) -> Option<[u8; N]> {
	bytes.get(..N)?.try_into().ok()
}

/// Calls `visit` once per sample of `frame`, in interleaved order
/// (sample 0 of every channel, then sample 1, ...), whatever its layout.
pub fn for_each_sample(frame: &AudioFrame, mut visit: impl FnMut(Sample)) -> Result<()> {
	let channels = usize::from(frame.channels.count());
	let size = frame.format.bytes();
	if channels == 0 {
		return Err(Message::Codec("audio frame has no channels"));
	}
	let bad = || Message::Codec("audio frame size does not match its format");
	if frame.layout.is_planar() {
		// Planes are stored back to back, each `len / channels` bytes.
		let plane = frame.data.len() / channels;
		let samples = plane / size;
		for i in 0..samples {
			for ch in 0..channels {
				let at = ch * plane + i * size;
				visit(Sample::read(frame.format, frame.data.get(at..).ok_or_else(bad)?).ok_or_else(bad)?);
			}
		}
	} else {
		let whole = frame.data.len() / (size * channels) * size * channels;
		for chunk in frame.data[..whole].chunks_exact(size) {
			visit(Sample::read(frame.format, chunk).ok_or_else(bad)?);
		}
	}
	Ok(())
}

/// Appends one sample in `format` (native little-endian layout).
#[inline]
pub fn write_sample(format: SampleFormat, s: Sample, out: &mut Vec<u8>) {
	match format {
		SampleFormat::U8 => out.push((s.to_int(8) as i8 as u8) ^ 0x80),
		SampleFormat::S16 => out.extend_from_slice(&s.to_i16().to_le_bytes()),
		SampleFormat::S32 => out.extend_from_slice(&s.to_i32().to_le_bytes()),
		SampleFormat::S64 => out.extend_from_slice(&s.to_int(64).to_le_bytes()),
		SampleFormat::F32 => out.extend_from_slice(&(s.to_float() as f32).to_le_bytes()),
		SampleFormat::F64 => out.extend_from_slice(&s.to_float().to_le_bytes()),
	}
}

/// Converts `frame` to interleaved `format`, returning the new sample bytes.
pub fn convert(frame: &AudioFrame, format: SampleFormat) -> Result<Vec<u8>> {
	let mut out = Vec::with_capacity(frame.data.len() / frame.format.bytes() * format.bytes());
	for_each_sample(frame, |s| write_sample(format, s, &mut out))?;
	Ok(out)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn integer_widths_shift() {
		let s = Sample::Int(i64::from(-12345i16) << 48);
		assert_eq!(s.to_i32(), -12345 << 16);
		assert_eq!(s.to_int(8), -49); // truncation, like swresample
		let s = Sample::Int(i64::from(0x1234_5678i32) << 32);
		assert_eq!(s.to_i16(), 0x1234);
	}

	#[test]
	fn float_conversion_rounds_and_saturates() {
		assert_eq!(Sample::Float(0.5).to_i16(), 16384);
		assert_eq!(Sample::Float(1.0).to_i16(), i16::MAX);
		assert_eq!(Sample::Float(-1.0).to_i16(), i16::MIN);
		assert_eq!(Sample::Float(-2.0).to_i32(), i32::MIN);
		assert_eq!(Sample::Float(f64::NAN).to_i16(), 0);
		assert_eq!(Sample::Float(2.5 / 32768.0).to_i16(), 2); // half to even
		assert_eq!(Sample::Float(1.0).to_int(64), i64::MAX);
		assert_eq!(Sample::Int(i64::MIN).to_float(), -1.0);
	}

	#[test]
	fn u8_is_offset_binary() {
		assert_eq!(Sample::read(SampleFormat::U8, &[128]), Some(Sample::Int(0)));
		assert_eq!(Sample::read(SampleFormat::U8, &[0]).map(|s| s.to_i16()), Some(i16::MIN));
		assert_eq!(Sample::read(SampleFormat::U8, &[255]).map(|s| s.to_i16()), Some(127 << 8));
	}
}
