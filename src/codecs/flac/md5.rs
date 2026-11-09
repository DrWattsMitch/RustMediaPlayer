//! MD5 (RFC 1321), for the audio signature in FLAC's STREAMINFO.
//!
//! FFmpeg: `libavutil/md5.c`. Incremental: [`Md5::update`] any number of
//! times, then [`Md5::finish`].

const S: [u32; 64] = [
	7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14,
	20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6,
	10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

fn k_table() -> &'static [u32; 64] {
	static K: std::sync::OnceLock<[u32; 64]> = std::sync::OnceLock::new();
	K.get_or_init(|| {
		let mut k = [0u32; 64];
		for (i, slot) in k.iter_mut().enumerate() {
			*slot = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
		}
		k
	})
}

#[derive(Debug, Clone)]
pub struct Md5 {
	state: [u32; 4],
	buffer: [u8; 64],
	buffered: usize,
	length: u64,
}

impl Default for Md5 {
	fn default() -> Self {
		Self::new()
	}
}

impl Md5 {
	pub fn new() -> Self {
		Self {
			state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
			buffer: [0; 64],
			buffered: 0,
			length: 0,
		}
	}

	fn block(&mut self, block: &[u8; 64]) {
		let k = k_table();
		let mut m = [0u32; 16];
		for (i, word) in m.iter_mut().enumerate() {
			*word =
				u32::from_le_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
		}
		let [mut a, mut b, mut c, mut d] = self.state;
		for i in 0..64 {
			let (f, g) = match i / 16 {
				0 => ((b & c) | (!b & d), i),
				1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
				2 => (b ^ c ^ d, (3 * i + 5) % 16),
				_ => (c ^ (b | !d), (7 * i) % 16),
			};
			let rotated = a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g]).rotate_left(S[i]);
			a = d;
			d = c;
			c = b;
			b = b.wrapping_add(rotated);
		}
		for (s, v) in self.state.iter_mut().zip([a, b, c, d]) {
			*s = s.wrapping_add(v);
		}
	}

	pub fn update(&mut self, mut data: &[u8]) {
		self.length = self.length.wrapping_add(data.len() as u64);
		if self.buffered > 0 {
			let take = (64 - self.buffered).min(data.len());
			self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
			self.buffered += take;
			data = &data[take..];
			if self.buffered < 64 {
				return;
			}
			let block = self.buffer;
			self.block(&block);
			self.buffered = 0;
		}
		let mut chunks = data.chunks_exact(64);
		for chunk in &mut chunks {
			let mut block = [0u8; 64];
			block.copy_from_slice(chunk);
			self.block(&block);
		}
		let rest = chunks.remainder();
		self.buffer[..rest.len()].copy_from_slice(rest);
		self.buffered = rest.len();
	}

	pub fn finish(mut self) -> [u8; 16] {
		let bits = self.length.wrapping_mul(8);
		let mut pad = vec![0x80u8];
		let used = (self.buffered + 1) % 64;
		pad.resize(1 + if used <= 56 { 56 - used } else { 120 - used }, 0);
		let saved = self.length;
		self.update(&pad);
		self.update(&bits.to_le_bytes());
		self.length = saved;
		let mut out = [0u8; 16];
		for (i, s) in self.state.iter().enumerate() {
			out[4 * i..4 * i + 4].copy_from_slice(&s.to_le_bytes());
		}
		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn hex(d: [u8; 16]) -> String {
		d.iter().map(|b| format!("{b:02x}")).collect()
	}

	#[test]
	fn rfc1321_vectors() {
		let digest = |s: &[u8]| {
			let mut m = Md5::new();
			m.update(s);
			hex(m.finish())
		};
		assert_eq!(digest(b""), "d41d8cd98f00b204e9800998ecf8427e");
		assert_eq!(digest(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
		assert_eq!(
			digest(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
			"57edf4a22be3c955ac49da2e2107b67a"
		);
		// Split updates give the same digest.
		let mut m = Md5::new();
		for chunk in b"message digest".chunks(3) {
			m.update(chunk);
		}
		assert_eq!(hex(m.finish()), "f96b697d7cb7938d525a2f31aaf161d0");
	}
}
