//! SAMPLE-AES: the HLS encryption that leaves the stream structure in the
//! clear and encrypts only parts of each sample ("MPEG-2 Stream Encryption
//! Format for HTTP Live Streaming", Apple; RFC 8216 §4.3.2.4).
//!
//! A transport stream (or packed audio) stays demuxable without the key:
//! PES headers, NAL unit headers, parameter sets and audio frame headers
//! are not encrypted, so RMP's TS demuxer finds streams, timestamps and
//! keyframes as usual, and the segment engine decrypts each packet's
//! payload ([`decrypt_packet`]). The HLS muxer does the inverse
//! ([`encrypt_packet`]) on what the TS muxer is about to write.
//!
//! What is encrypted, always AES-128-CBC with the IV reset at the start of
//! every NAL unit or audio frame:
//!
//! | Codec | Encrypted |
//! | --- | --- |
//! | H.264 | NAL units of type 1 and 5 longer than 48 bytes: after 32 clear bytes, one 16-byte block of every 160 (then 144 clear), as long as more than 16 bytes remain. Emulation prevention is then applied to the whole NAL unit again. |
//! | AAC (ADTS) | after the ADTS header and 16 clear bytes, every whole 16-byte block |
//! | AC-3, E-AC-3 | after 16 clear bytes, every whole 16-byte block |
//!
//! FFmpeg: `libavformat/hls_sample_encryption.c` (decryption only).

use super::aes::Aes128;
use crate::container::mpegts::es::{Framing, audio_frame};

/// The codecs SAMPLE-AES defines, by how their samples are encrypted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCodec {
	H264,
	Aac,
	/// AC-3 and E-AC-3.
	Ac3,
}

impl SampleCodec {
	pub fn of(codec: &str) -> Option<Self> {
		match codec {
			"h264" => Some(Self::H264),
			"aac" => Some(Self::Aac),
			"ac3" | "eac3" => Some(Self::Ac3),
			_ => None,
		}
	}
}

/// Clear bytes at the start of an encrypted NAL unit.
const NAL_LEADER: usize = 32;
/// Only NAL units longer than this are encrypted.
const NAL_MIN: usize = 48;
/// Clear bytes after each encrypted block of a NAL unit.
const NAL_SKIP: usize = 144;
/// Clear bytes after an audio frame header.
const AUDIO_LEADER: usize = 16;

/// An AES-128 key and IV for one segment.
pub struct SampleCipher {
	aes: Aes128,
	iv: [u8; 16],
}

impl SampleCipher {
	pub fn new(key: &[u8; 16], iv: &[u8; 16]) -> Self {
		Self { aes: Aes128::new(key), iv: *iv }
	}
}

/// Encrypts one packet of an Annex B H.264, ADTS or AC-3 stream.
pub fn encrypt_packet(cipher: &SampleCipher, codec: SampleCodec, data: &[u8]) -> Vec<u8> {
	match codec {
		SampleCodec::H264 => map_nal_units(data, |nal| encrypt_nal(cipher, nal)),
		SampleCodec::Aac | SampleCodec::Ac3 => {
			let mut out = data.to_vec();
			for_each_frame(codec, &mut out, |frame| cipher.aes.encrypt_cbc_in_place(&cipher.iv, frame));
			out
		}
	}
}

/// Decrypts one packet; the inverse of [`encrypt_packet`]. Data that isn't
/// what the codec says (no start codes, no frame headers) is returned as it
/// is: undecryptable is better reported by the decoder than guessed at.
pub fn decrypt_packet(cipher: &SampleCipher, codec: SampleCodec, data: &[u8]) -> Vec<u8> {
	match codec {
		SampleCodec::H264 => map_nal_units(data, |nal| decrypt_nal(cipher, nal)),
		SampleCodec::Aac | SampleCodec::Ac3 => {
			let mut out = data.to_vec();
			for_each_frame(codec, &mut out, |frame| cipher.aes.decrypt_cbc_in_place(&cipher.iv, frame));
			out
		}
	}
}

fn encrypted_nal(nal: &[u8]) -> bool {
	nal.len() > NAL_MIN && matches!(nal[0] & 0x1F, 1 | 5)
}

/// The 1:9 block pattern after the leader: every encrypted block, chained
/// in CBC from the IV, is followed by up to 144 clear bytes.
fn pattern(cipher: &SampleCipher, nal: &mut [u8], encrypt: bool) {
	let mut chain = cipher.iv;
	let mut at = NAL_LEADER;
	while nal.len() - at > 16 {
		let block: &mut [u8; 16] = match (&mut nal[at..at + 16]).try_into() {
			Ok(b) => b,
			Err(_) => return,
		};
		if encrypt {
			for (b, c) in block.iter_mut().zip(&chain) {
				*b ^= c;
			}
			cipher.aes.encrypt_block(block);
			chain = *block;
		} else {
			let next = *block;
			cipher.aes.decrypt_block(block);
			for (b, c) in block.iter_mut().zip(&chain) {
				*b ^= c;
			}
			chain = next;
		}
		at += 16;
		at += (nal.len() - at).min(NAL_SKIP);
	}
}

fn encrypt_nal(cipher: &SampleCipher, nal: &[u8]) -> Vec<u8> {
	if !encrypted_nal(nal) {
		return nal.to_vec();
	}
	let mut out = nal.to_vec();
	pattern(cipher, &mut out, true);
	escape(&out)
}

fn decrypt_nal(cipher: &SampleCipher, nal: &[u8]) -> Vec<u8> {
	if !encrypted_nal(nal) {
		return nal.to_vec();
	}
	let mut out = unescape(nal);
	if out.len() <= NAL_LEADER {
		return nal.to_vec();
	}
	pattern(cipher, &mut out, false);
	out
}

/// Inserts emulation prevention bytes: `00 00 0x` (x ≤ 3) → `00 00 03 0x`.
fn escape(data: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(data.len() + data.len() / 64);
	let mut zeros = 0;
	for &b in data {
		if zeros >= 2 && b <= 3 {
			out.push(3);
			zeros = 0;
		}
		out.push(b);
		zeros = if b == 0 { zeros + 1 } else { 0 };
	}
	out
}

/// Removes emulation prevention bytes: `00 00 03` → `00 00`.
fn unescape(data: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(data.len());
	let mut zeros = 0;
	for &b in data {
		if zeros >= 2 && b == 3 {
			zeros = 0;
			continue;
		}
		out.push(b);
		zeros = if b == 0 { zeros + 1 } else { 0 };
	}
	out
}

/// Rebuilds an Annex B access unit with each NAL unit passed through `f`.
/// Start codes and the bytes between NAL units are kept as they are.
fn map_nal_units(data: &[u8], mut f: impl FnMut(&[u8]) -> Vec<u8>) -> Vec<u8> {
	// Payload start of every NAL unit (after its 00 00 01).
	let mut starts = Vec::new();
	let mut i = 0;
	while i + 3 <= data.len() {
		if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
			starts.push(i + 3);
			i += 3;
		} else {
			i += 1;
		}
	}
	if starts.is_empty() {
		return data.to_vec();
	}
	let mut out = Vec::with_capacity(data.len() + 64);
	out.extend_from_slice(&data[..starts[0]]);
	for (n, &s) in starts.iter().enumerate() {
		let mut end = starts.get(n + 1).map_or(data.len(), |&next| next - 3);
		// Zero bytes before the next start code belong to it (or are
		// trailing padding); an encrypted NAL unit never ends in zero.
		while end > s && data[end - 1] == 0 {
			end -= 1;
		}
		out.extend_from_slice(&f(&data[s..end]));
		let next = starts.get(n + 1).copied().unwrap_or(data.len());
		out.extend_from_slice(&data[end..next]);
	}
	out
}

/// Calls `f` on the encrypted part of every audio frame in `data`: its
/// whole 16-byte blocks after the header and the 16-byte leader. Bytes
/// that don't parse as a frame end the walk.
fn for_each_frame(codec: SampleCodec, data: &mut [u8], mut f: impl FnMut(&mut [u8])) {
	let framing = if codec == SampleCodec::Aac { Framing::Adts } else { Framing::Ac3 };
	let mut at = 0;
	while at < data.len() {
		let Some(frame) = audio_frame(framing, &data[at..]) else { break };
		let len = frame.len.min(data.len() - at);
		if len == 0 {
			break;
		}
		let header = match codec {
			// protection_absent selects a 7- or 9-byte header.
			SampleCodec::Aac => {
				if data[at + 1] & 1 == 1 {
					7
				} else {
					9
				}
			}
			_ => 0,
		};
		let lead = header + AUDIO_LEADER;
		if len > lead {
			let body = &mut data[at + lead..at + len];
			let whole = body.len() - body.len() % 16;
			f(&mut body[..whole]);
		}
		at += len;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn cipher() -> SampleCipher {
		SampleCipher::new(&[7; 16], &[9; 16])
	}

	#[test]
	fn escaping_round_trips() {
		let data = [0, 0, 0, 0, 1, 0, 0, 3, 5, 0, 0, 2, 0];
		let escaped = escape(&data);
		assert_eq!(escaped, [0, 0, 3, 0, 0, 3, 1, 0, 0, 3, 3, 5, 0, 0, 3, 2, 0]);
		assert_eq!(unescape(&escaped), data);
	}

	#[test]
	fn h264_pattern_and_round_trip() {
		// An access unit: SPS (clear), a short slice (clear), an IDR slice.
		let mut idr = vec![0x65];
		idr.extend((0..400u32).map(|i| (i * 7 % 251) as u8 | 1));
		let mut au = vec![0, 0, 0, 1, 0x67, 0x42, 0, 0x1E, 0, 0, 1, 0x41];
		au.extend_from_slice(&[0x11; 40]);
		au.extend_from_slice(&[0, 0, 1]);
		au.extend_from_slice(&idr);
		let enc = encrypt_packet(&cipher(), SampleCodec::H264, &au);
		assert_ne!(enc, au);
		// The headers and the first 32 bytes of the IDR slice are clear.
		let at = enc.windows(4).position(|w| w == [0, 0, 1, 0x65]).unwrap() + 3;
		assert_eq!(&enc[..at], &au[..at]);
		assert_eq!(&enc[at..at + 32], &idr[..32]);
		// No start code emulation in the ciphertext.
		assert_eq!(enc.windows(3).filter(|w| *w == [0, 0, 1]).count(), 3);
		assert_eq!(decrypt_packet(&cipher(), SampleCodec::H264, &enc), au);
	}

	#[test]
	fn adts_frames_round_trip() {
		// Two ADTS frames of 100 bytes (AAC-LC, 48 kHz, stereo).
		let mut frame = vec![0xFF, 0xF1, 0x4C, 0x80, 0, 0, 0xFC];
		let len = 100usize;
		frame[3] |= ((len >> 11) & 3) as u8;
		frame[4] = ((len >> 3) & 0xFF) as u8;
		frame[5] = (((len & 7) << 5) as u8) | 0x1F;
		frame.extend((0..len - 7).map(|i| i as u8));
		let data = [frame.clone(), frame.clone()].concat();
		let enc = encrypt_packet(&cipher(), SampleCodec::Aac, &data);
		// Header + 16 bytes clear, 64 bytes encrypted, 13 bytes clear.
		assert_eq!(&enc[..23], &data[..23]);
		assert_ne!(&enc[23..87], &data[23..87]);
		assert_eq!(&enc[87..100], &data[87..100]);
		// The IV restarts with each frame.
		assert_eq!(&enc[..100], &enc[100..]);
		assert_eq!(decrypt_packet(&cipher(), SampleCodec::Aac, &enc), data);
	}
}
