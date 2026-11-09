//! Microsoft ADPCM (`adpcm_ms`).
//!
//! Block layout (Microsoft Multimedia Standards Update, 1994):
//!
//! ```text
//! predictor index per channel (u8)
//! idelta per channel (i16 LE)
//! sample1 per channel (i16 LE)   -- the second output sample
//! sample2 per channel (i16 LE)   -- the first output sample
//! then nibbles, high first; in stereo the high nibble is left, the low right
//! ```
//!
//! Each sample is predicted from the previous two with one of seven fixed
//! coefficient pairs. Like FFmpeg, the decoder uses the standard pairs and
//! ignores the copy that encoders store in the `fmt ` chunk.

use crate::Message;
use crate::message::Result;

pub const HEADER_BYTES: usize = 7;

/// The seven standard predictor coefficient pairs, in units of 1/256.
const COEFF1: [i32; 7] = [256, 512, 0, 192, 240, 460, 392];
const COEFF2: [i32; 7] = [0, -256, 0, 64, 0, -208, -232];

/// Scale factor for the next step, per 4-bit code, in units of 1/256.
const ADAPTATION: [i32; 16] =
	[230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230];

const MIN_IDELTA: i32 = 16;
/// FFmpeg's clamp, keeping `ADAPTATION[n] * idelta` inside `i32`.
const MAX_IDELTA: i32 = i32::MAX / 768;

#[derive(Debug, Clone, Copy, Default)]
struct Channel {
	coeff1: i32,
	coeff2: i32,
	idelta: i32,
	sample1: i32,
	sample2: i32,
}

impl Channel {
	fn expand(&mut self, code: u8) -> i16 {
		// Coefficients are in units of 1/256. C division (truncation toward
		// zero), as in the reference decoder; FFmpeg stores the coefficients
		// pre-divided by 4 and divides by 64, which truncates identically.
		let mut predictor = (self.sample1 * self.coeff1 + self.sample2 * self.coeff2) / 256;
		let signed = if code & 8 != 0 { i32::from(code) - 16 } else { i32::from(code) };
		predictor = predictor.saturating_add(signed * self.idelta);
		self.sample2 = self.sample1;
		self.sample1 = predictor.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
		let next = (i64::from(ADAPTATION[usize::from(code)]) * i64::from(self.idelta)) >> 8;
		self.idelta = next.clamp(i64::from(MIN_IDELTA), i64::from(MAX_IDELTA)) as i32;
		self.sample1 as i16
	}
}

fn i16_at(bytes: &[u8], at: usize) -> i32 {
	i32::from(i16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Decodes one block into `out` (interleaved), returning samples per channel.
pub fn decode_block(block: &[u8], channels: usize, out: &mut Vec<i16>) -> Result<usize> {
	if !super::AdpcmCodec::Ms.supports_channels(channels) {
		return Err(Message::Codec("ms adpcm supports 1 or 2 channels"));
	}
	let samples = super::AdpcmCodec::Ms
		.samples_in_block(block.len(), channels)
		.ok_or(Message::Codec("ms adpcm block too short"))?;

	let mut state = [Channel::default(); 2];
	for (ch, s) in state.iter_mut().enumerate().take(channels) {
		let predictor = usize::from(block[ch]);
		if predictor >= COEFF1.len() {
			return Err(Message::Codec("ms adpcm predictor index out of range"));
		}
		s.coeff1 = COEFF1[predictor];
		s.coeff2 = COEFF2[predictor];
		s.idelta = i16_at(block, channels + 2 * ch);
		s.sample1 = i16_at(block, 3 * channels + 2 * ch);
		s.sample2 = i16_at(block, 5 * channels + 2 * ch);
	}

	let start = out.len();
	out.reserve(samples * channels);
	for s in state.iter().take(channels) {
		out.push(s.sample2 as i16);
	}
	for s in state.iter().take(channels) {
		out.push(s.sample1 as i16);
	}
	let body = &block[HEADER_BYTES * channels..];
	// Mono packs two consecutive samples per byte; stereo one per channel.
	let bytes = (samples - 2) * channels / 2;
	let second = channels - 1;
	for byte in &body[..bytes] {
		let hi = state[0].expand(byte >> 4);
		let lo = state[second].expand(byte & 0x0F);
		out.push(hi);
		out.push(lo);
	}
	debug_assert_eq!(out.len() - start, samples * channels);
	Ok(samples)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn header_samples_come_first_in_order() {
		// Mono, predictor 0, idelta 16, sample1 = 200, sample2 = 100, one byte.
		let block = [0, 16, 0, 200, 0, 100, 0, 0x10];
		let mut out = Vec::new();
		assert_eq!(decode_block(&block, 1, &mut out).unwrap(), 4);
		assert_eq!(&out[..2], &[100, 200]);
		// Predictor pair 0 is (256, 0), i.e. "repeat sample1": 200, plus
		// code +1 times idelta 16.
		assert_eq!(out[2], 216);
		// Code 0 adds nothing.
		assert_eq!(out[3], 216);
	}

	#[test]
	fn rejects_invalid_blocks() {
		let mut out = Vec::new();
		assert!(decode_block(&[7, 16, 0, 0, 0, 0, 0], 1, &mut out).is_err());
		assert!(decode_block(&[0; 6], 1, &mut out).is_err());
		assert!(decode_block(&[0; 21], 3, &mut out).is_err());
		// Extreme idelta and codes saturate instead of overflowing.
		let block = [1, 0xFF, 0x7F, 0xFF, 0x7F, 0x00, 0x80, 0x77, 0x77, 0x88, 0x88];
		assert_eq!(decode_block(&block, 1, &mut out).unwrap(), 10);
	}
}
