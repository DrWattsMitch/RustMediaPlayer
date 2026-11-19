//! AES-128 CBC for `EXT-X-KEY:METHOD=AES-128` segments (RFC 8216
//! §4.3.2.4; FIPS-197), and the in-place block chains SAMPLE-AES uses.
//! Byte-oriented and table-free apart from the S-boxes: a segment is a few
//! megabytes, so clarity wins over a T-table implementation.
//!
//! FFmpeg: `libavutil/aes.c`, `libavformat/hls.c` (via the `crypto:`
//! protocol).

/// The AES S-box, generated from the multiplicative inverse in GF(2^8)
/// and the affine transform (FIPS-197 §5.1.1).
const SBOX: [u8; 256] = {
	let mut sbox = [0u8; 256];
	let mut i = 0;
	while i < 256 {
		// Inverse by exponentiation: x^254.
		let x = i as u8;
		let mut inv = 1u8;
		let mut base = x;
		let mut e = 254u32;
		while e > 0 {
			if e & 1 == 1 {
				inv = gmul(inv, base);
			}
			base = gmul(base, base);
			e >>= 1;
		}
		if x == 0 {
			inv = 0;
		}
		let mut s = inv;
		let mut r = 0;
		while r < 4 {
			inv = inv.rotate_left(1);
			s ^= inv;
			r += 1;
		}
		sbox[i] = s ^ 0x63;
		i += 1;
	}
	sbox
};

const INV_SBOX: [u8; 256] = {
	let mut inv = [0u8; 256];
	let mut i = 0;
	while i < 256 {
		inv[SBOX[i] as usize] = i as u8;
		i += 1;
	}
	inv
};

const fn gmul(mut a: u8, mut b: u8) -> u8 {
	let mut p = 0u8;
	while b != 0 {
		if b & 1 != 0 {
			p ^= a;
		}
		let hi = a & 0x80;
		a <<= 1;
		if hi != 0 {
			a ^= 0x1B;
		}
		b >>= 1;
	}
	p
}

/// An expanded AES-128 key (11 round keys).
pub struct Aes128 {
	round_keys: [[u8; 16]; 11],
}

impl Aes128 {
	pub fn new(key: &[u8; 16]) -> Self {
		let mut w = [[0u8; 4]; 44];
		for (i, word) in w.iter_mut().take(4).enumerate() {
			word.copy_from_slice(&key[4 * i..4 * i + 4]);
		}
		let mut rcon = 1u8;
		for i in 4..44 {
			let mut t = w[i - 1];
			if i % 4 == 0 {
				t = [
					SBOX[t[1] as usize] ^ rcon,
					SBOX[t[2] as usize],
					SBOX[t[3] as usize],
					SBOX[t[0] as usize],
				];
				rcon = gmul(rcon, 2);
			}
			for k in 0..4 {
				w[i][k] = w[i - 4][k] ^ t[k];
			}
		}
		let mut round_keys = [[0u8; 16]; 11];
		for (r, rk) in round_keys.iter_mut().enumerate() {
			for c in 0..4 {
				rk[4 * c..4 * c + 4].copy_from_slice(&w[4 * r + c]);
			}
		}
		Self { round_keys }
	}

	/// Encrypts one 16-byte block in place (FIPS-197 §5.1).
	pub fn encrypt_block(&self, s: &mut [u8; 16]) {
		xor(s, &self.round_keys[0]);
		for round in 1..11 {
			for b in s.iter_mut() {
				*b = SBOX[*b as usize];
			}
			// ShiftRows (state is column-major: s[row + 4 * col]).
			for row in 1..4 {
				let r = [s[row], s[row + 4], s[row + 8], s[row + 12]];
				for col in 0..4 {
					s[row + 4 * col] = r[(col + row) % 4];
				}
			}
			if round < 10 {
				for col in 0..4 {
					let c = [s[4 * col], s[4 * col + 1], s[4 * col + 2], s[4 * col + 3]];
					for row in 0..4 {
						s[4 * col + row] =
							gmul(c[row], 2) ^ gmul(c[(row + 1) % 4], 3) ^ c[(row + 2) % 4] ^ c[(row + 3) % 4];
					}
				}
			}
			xor(s, &self.round_keys[round]);
		}
	}

	/// CBC-encrypts the whole 16-byte blocks of `data` in place, chained
	/// from `iv`; a trailing partial block is left as it is (SAMPLE-AES).
	pub fn encrypt_cbc_in_place(&self, iv: &[u8; 16], data: &mut [u8]) {
		let mut previous = *iv;
		for chunk in data.chunks_exact_mut(16) {
			let mut block = [0u8; 16];
			block.copy_from_slice(chunk);
			xor(&mut block, &previous);
			self.encrypt_block(&mut block);
			chunk.copy_from_slice(&block);
			previous = block;
		}
	}

	/// CBC-decrypts the whole 16-byte blocks of `data` in place.
	pub fn decrypt_cbc_in_place(&self, iv: &[u8; 16], data: &mut [u8]) {
		let mut previous = *iv;
		for chunk in data.chunks_exact_mut(16) {
			let mut block = [0u8; 16];
			block.copy_from_slice(chunk);
			let cipher = block;
			self.decrypt_block(&mut block);
			xor(&mut block, &previous);
			chunk.copy_from_slice(&block);
			previous = cipher;
		}
	}

	/// Decrypts one 16-byte block in place (FIPS-197 §5.3).
	pub fn decrypt_block(&self, s: &mut [u8; 16]) {
		xor(s, &self.round_keys[10]);
		for round in (0..10).rev() {
			// InvShiftRows (state is column-major: s[row + 4 * col]).
			for row in 1..4 {
				let r = [s[row], s[row + 4], s[row + 8], s[row + 12]];
				for col in 0..4 {
					s[row + 4 * ((col + row) % 4)] = r[col];
				}
			}
			for b in s.iter_mut() {
				*b = INV_SBOX[*b as usize];
			}
			xor(s, &self.round_keys[round]);
			if round > 0 {
				for col in 0..4 {
					let c = [s[4 * col], s[4 * col + 1], s[4 * col + 2], s[4 * col + 3]];
					for row in 0..4 {
						s[4 * col + row] = gmul(c[row], 14)
							^ gmul(c[(row + 1) % 4], 11)
							^ gmul(c[(row + 2) % 4], 13)
							^ gmul(c[(row + 3) % 4], 9);
					}
				}
			}
		}
	}
}

fn xor(a: &mut [u8; 16], b: &[u8; 16]) {
	for (x, y) in a.iter_mut().zip(b) {
		*x ^= y;
	}
}

/// Decrypts AES-128-CBC data and removes the PKCS#7 padding. `None` when
/// the length isn't a whole number of blocks or the padding is invalid
/// (wrong key or IV).
pub fn decrypt_cbc(key: &[u8; 16], iv: &[u8; 16], data: &[u8]) -> Option<Vec<u8>> {
	if data.is_empty() || !data.len().is_multiple_of(16) {
		return None;
	}
	let aes = Aes128::new(key);
	let mut out = Vec::with_capacity(data.len());
	let mut previous = *iv;
	for chunk in data.chunks_exact(16) {
		let mut block = [0u8; 16];
		block.copy_from_slice(chunk);
		let cipher = block;
		aes.decrypt_block(&mut block);
		xor(&mut block, &previous);
		out.extend_from_slice(&block);
		previous = cipher;
	}
	let pad = usize::from(*out.last()?);
	if pad == 0 || pad > 16 || out[out.len() - pad..].iter().any(|&b| usize::from(b) != pad) {
		return None;
	}
	out.truncate(out.len() - pad);
	Some(out)
}

/// Encrypts with AES-128-CBC after PKCS#7 padding (always 1-16 bytes),
/// as `EXT-X-KEY:METHOD=AES-128` segments are.
pub fn encrypt_cbc(key: &[u8; 16], iv: &[u8; 16], data: &[u8]) -> Vec<u8> {
	let pad = 16 - data.len() % 16;
	let mut out = Vec::with_capacity(data.len() + pad);
	out.extend_from_slice(data);
	out.resize(data.len() + pad, pad as u8);
	Aes128::new(key).encrypt_cbc_in_place(iv, &mut out);
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	fn hex(s: &str) -> Vec<u8> {
		(0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
	}

	#[test]
	fn sbox_values() {
		assert_eq!((SBOX[0], SBOX[1], SBOX[0x53]), (0x63, 0x7C, 0xED));
	}

	#[test]
	fn fips197_appendix_c1() {
		let key: [u8; 16] = hex("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
		let mut block: [u8; 16] = hex("69c4e0d86a7b0430d8cdb78070b4c55a").try_into().unwrap();
		Aes128::new(&key).decrypt_block(&mut block);
		assert_eq!(block.to_vec(), hex("00112233445566778899aabbccddeeff"));
		Aes128::new(&key).encrypt_block(&mut block);
		assert_eq!(block.to_vec(), hex("69c4e0d86a7b0430d8cdb78070b4c55a"));
	}

	#[test]
	fn sp800_38a_cbc_with_padding() {
		// NIST SP 800-38A F.2.2 first block, followed by a full padding block
		// encrypted under the same chain (computed with OpenSSL).
		let key: [u8; 16] = hex("2b7e151628aed2a6abf7158809cf4f3c").try_into().unwrap();
		let iv: [u8; 16] = hex("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
		let data = hex("7649abac8119b246cee98e9b12e9197d8964e0b149c10b7b682e6e39aaeb731c");
		let plain = decrypt_cbc(&key, &iv, &data).unwrap();
		assert_eq!(plain, hex("6bc1bee22e409f96e93d7e117393172a"));
		// A wrong key leaves invalid padding (a wrong IV would only garble
		// the first block).
		assert!(decrypt_cbc(&[0; 16], &iv, &data).is_none());
		assert!(decrypt_cbc(&key, &iv, &data[..20]).is_none());
		// Encryption produces the same chain, and round-trips any length.
		assert_eq!(encrypt_cbc(&key, &iv, &plain), data);
		for len in [0usize, 1, 15, 16, 17, 100] {
			let text: Vec<u8> = (0..len as u8).collect();
			assert_eq!(decrypt_cbc(&key, &iv, &encrypt_cbc(&key, &iv, &text)).unwrap(), text);
		}
	}
}
