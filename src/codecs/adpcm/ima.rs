//! IMA ADPCM in WAV blocks (`adpcm_ima_wav`).
//!
//! Block layout (IMA Digital Audio Focus and Technical Working Groups,
//! "Recommended Practices for Enhancing Digital Audio Compatibility in
//! Multimedia Systems", 1992):
//!
//! ```text
//! per channel: predictor (i16 LE) | step index (u8) | reserved (u8)
//! then repeatedly, per channel: 4 bytes = 8 samples, low nibble first
//! ```
//!
//! The header predictor is the block's first output sample.

use crate::Message;
use crate::message::Result;

pub const HEADER_BYTES: usize = 4;
pub const GROUP_BYTES: usize = 4;
pub const GROUP_SAMPLES: usize = 8;
const MAX_STEP_INDEX: usize = 88;

/// Step-index adjustment per 4-bit code (sign bit ignored).
const INDEX_TABLE: [i8; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Quantiser step sizes (89 entries, roughly 1.1x apart).
const STEP_TABLE: [i32; 89] = [
	7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73,
	80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
	544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499,
	2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
	12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

#[derive(Debug, Clone, Copy)]
struct Channel {
	predictor: i32,
	step_index: usize,
}

impl Channel {
	/// Decodes one 4-bit code. The difference is `(2·|code| + 1)·step / 8`,
	/// computed in one multiply as FFmpeg's `adpcm_ima_expand_nibble` does,
	/// rather than the shift-and-add form of the reference algorithm.
	fn expand(&mut self, code: u8) -> i16 {
		let step = STEP_TABLE[self.step_index];
		let delta = i32::from(code & 7);
		let diff = ((2 * delta + 1) * step) >> 3;
		let predictor = if code & 8 != 0 { self.predictor - diff } else { self.predictor + diff };
		self.predictor = predictor.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
		let index = self.step_index as i32 + i32::from(INDEX_TABLE[usize::from(code & 0x0F)]);
		self.step_index = index.clamp(0, MAX_STEP_INDEX as i32) as usize;
		self.predictor as i16
	}
}

/// Decodes one block into `out` (interleaved), returning samples per channel.
pub fn decode_block(block: &[u8], channels: usize, out: &mut Vec<i16>) -> Result<usize> {
	let samples = super::AdpcmCodec::ImaWav
		.samples_in_block(block.len(), channels)
		.ok_or(Message::Codec("ima adpcm block too short"))?;
	let (headers, body) = block.split_at(HEADER_BYTES * channels);

	let mut state = Vec::with_capacity(channels);
	for h in headers.chunks_exact(HEADER_BYTES) {
		let predictor = i32::from(i16::from_le_bytes([h[0], h[1]]));
		// FFmpeg reads the step index as a signed 16-bit value (index and
		// reserved byte together) and rejects anything outside 0..=88.
		let step_index = i16::from_le_bytes([h[2], h[3]]);
		if !(0..=MAX_STEP_INDEX as i16).contains(&step_index) {
			return Err(Message::Codec("ima adpcm step index out of range"));
		}
		state.push(Channel { predictor, step_index: step_index as usize });
	}

	let start = out.len();
	out.resize(start + samples * channels, 0);
	let frame = &mut out[start..];
	for (ch, s) in state.iter().enumerate() {
		frame[ch] = s.predictor as i16;
	}
	// Each group holds 8 consecutive samples of one channel; groups rotate
	// through the channels.
	let groups = (samples - 1) / GROUP_SAMPLES;
	let mut bytes = body.chunks_exact(GROUP_BYTES);
	for group in 0..groups {
		for (ch, s) in state.iter_mut().enumerate() {
			let Some(chunk) = bytes.next() else {
				return Err(Message::Codec("ima adpcm block too short"));
			};
			let first = 1 + group * GROUP_SAMPLES;
			for (i, byte) in chunk.iter().enumerate() {
				frame[(first + 2 * i) * channels + ch] = s.expand(byte & 0x0F);
				frame[(first + 2 * i + 1) * channels + ch] = s.expand(byte >> 4);
			}
		}
	}
	Ok(samples)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn header_sample_comes_first_and_codes_adapt() {
		// Mono: predictor 1000, step index 0, then 0x77 (two +7 codes) x4.
		let block = [0xE8, 0x03, 0, 0, 0x77, 0x77, 0x77, 0x77];
		let mut out = Vec::new();
		assert_eq!(decode_block(&block, 1, &mut out).unwrap(), 9);
		assert_eq!(out[0], 1000);
		// First +7 at step 7: diff = (15 * 7) >> 3 = 13; index -> 8 (step 16).
		assert_eq!(out[1], 1013);
		assert_eq!(out[2], 1013 + ((15 * 16) >> 3));
		assert!(out.windows(2).all(|w| w[1] > w[0]));
	}

	#[test]
	fn rejects_bad_step_index_and_short_blocks() {
		let mut out = Vec::new();
		assert!(decode_block(&[0, 0, 89, 0], 1, &mut out).is_err());
		assert!(decode_block(&[0, 0, 0, 1], 1, &mut out).is_err());
		assert!(decode_block(&[0, 0, 0], 1, &mut out).is_err());
		assert_eq!(decode_block(&[0, 0, 0, 0, 0x11], 1, &mut out).unwrap(), 1);
	}
}
