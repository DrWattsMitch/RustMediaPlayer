//! Opus range decoder (RFC 6716 §4.1; libopus `celt/entdec.c`,
//! `celt/entcode.c`, `celt/laplace.c`).
//!
//! Symbols are read MSB-first from the front of the packet; raw bits
//! (`ec_dec_bits`) are read LSB-first from the back. Reads past either end
//! yield zeros, as the reference decoder does, so a truncated frame decodes
//! to *something* without any out-of-bounds access; [`RangeDecoder::error`]
//! records invalid `ec_dec_uint` values.

const SYM_BITS: u32 = 8;
const CODE_BITS: u32 = 32;
const SYM_MAX: u32 = (1 << SYM_BITS) - 1;
const CODE_TOP: u32 = 1 << (CODE_BITS - 1);
const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;
const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;
const WINDOW_SIZE: u32 = 32;
const UINT_BITS: u32 = 8;
/// Fractional bits of [`RangeDecoder::tell_frac`].
pub const BITRES: u32 = 3;

/// Number of bits needed to represent `x` (0 for 0): libopus `EC_ILOG`.
#[inline]
pub fn ilog(x: u32) -> u32 {
	32 - x.leading_zeros()
}

#[derive(Debug, Clone)]
pub struct RangeDecoder<'a> {
	buf: &'a [u8],
	/// Bytes used by the range coder (front) and raw bits (back) so far.
	offs: usize,
	end_offs: usize,
	end_window: u32,
	nend_bits: u32,
	nbits_total: i32,
	rng: u32,
	val: u32,
	ext: u32,
	rem: u32,
	pub error: bool,
}

impl<'a> RangeDecoder<'a> {
	pub fn new(buf: &'a [u8]) -> Self {
		let mut d = Self {
			buf,
			offs: 0,
			end_offs: 0,
			end_window: 0,
			nend_bits: 0,
			nbits_total: (CODE_BITS + 1 - ((CODE_BITS - CODE_EXTRA) / SYM_BITS) * SYM_BITS) as i32,
			rng: 1 << CODE_EXTRA,
			val: 0,
			ext: 0,
			rem: 0,
			error: false,
		};
		d.rem = d.read_byte();
		d.val = d.rng - 1 - (d.rem >> (SYM_BITS - CODE_EXTRA));
		d.normalize();
		d
	}

	/// Final range, used by the encoder/decoder conformance check
	/// (`OPUS_GET_FINAL_RANGE`).
	pub fn range(&self) -> u32 {
		self.rng
	}

	/// Counts `bits` more as consumed without reading them (CELT silence).
	pub fn add_bits(&mut self, bits: i32) {
		self.nbits_total += bits;
	}

	/// Removes `bytes` from the end of the buffer (the hybrid redundancy
	/// frame lives there): raw bits are then read from the new end.
	pub fn shrink(&mut self, bytes: usize) {
		let len = self.buf.len().saturating_sub(bytes);
		self.buf = &self.buf[..len];
		self.offs = self.offs.min(len);
	}

	/// Total bytes of the buffer this decoder works on.
	pub fn storage(&self) -> usize {
		self.buf.len()
	}

	#[inline]
	fn read_byte(&mut self) -> u32 {
		match self.buf.get(self.offs) {
			Some(&b) => {
				self.offs += 1;
				u32::from(b)
			}
			None => 0,
		}
	}

	#[inline]
	fn read_byte_from_end(&mut self) -> u32 {
		if self.end_offs < self.buf.len() {
			self.end_offs += 1;
			u32::from(self.buf[self.buf.len() - self.end_offs])
		} else {
			0
		}
	}

	#[inline]
	fn normalize(&mut self) {
		while self.rng <= CODE_BOT {
			self.nbits_total += SYM_BITS as i32;
			self.rng <<= SYM_BITS;
			let mut sym = self.rem;
			self.rem = self.read_byte();
			sym = ((sym << SYM_BITS) | self.rem) >> (SYM_BITS - CODE_EXTRA);
			self.val = ((self.val << SYM_BITS).wrapping_add(SYM_MAX & !sym)) & (CODE_TOP - 1);
		}
	}

	/// Cumulative frequency of the next symbol, for a total of `ft`.
	#[inline]
	pub fn decode(&mut self, ft: u32) -> u32 {
		self.ext = self.rng / ft;
		let s = self.val / self.ext;
		ft - (s + 1).min(ft)
	}

	/// [`RangeDecoder::decode`] with `ft = 1 << bits`.
	#[inline]
	pub fn decode_bin(&mut self, bits: u32) -> u32 {
		self.ext = self.rng >> bits;
		let s = self.val / self.ext;
		(1 << bits) - (s + 1).min(1 << bits)
	}

	/// Consumes the symbol occupying `[fl, fh)` of `ft`.
	#[inline]
	pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
		let s = self.ext.wrapping_mul(ft - fh);
		self.val = self.val.wrapping_sub(s);
		self.rng = if fl > 0 { self.ext.wrapping_mul(fh - fl) } else { self.rng.wrapping_sub(s) };
		self.normalize();
	}

	/// A bit whose probability of being 1 is `1 / 2^logp`.
	#[inline]
	pub fn bit_logp(&mut self, logp: u32) -> bool {
		let r = self.rng;
		let d = self.val;
		let s = r >> logp;
		let ret = d < s;
		if !ret {
			self.val = d - s;
		}
		self.rng = if ret { s } else { r - s };
		self.normalize();
		ret
	}

	/// A symbol from an "inverse CDF" table with `1 << ftb` total.
	#[inline]
	pub fn icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
		let mut s = self.rng;
		let d = self.val;
		let r = s >> ftb;
		let mut ret = 0usize;
		let mut t;
		loop {
			t = s;
			s = r.wrapping_mul(u32::from(icdf.get(ret).copied().unwrap_or(0)));
			if d >= s {
				break;
			}
			ret += 1;
		}
		self.val = d.wrapping_sub(s);
		self.rng = t.wrapping_sub(s);
		self.normalize();
		ret
	}

	/// A uniform integer in `[0, ft)` (`ft > 1`).
	pub fn uint(&mut self, ft: u32) -> u32 {
		if ft <= 1 {
			return 0;
		}
		let ft = ft - 1;
		let mut ftb = ilog(ft);
		if ftb > UINT_BITS {
			ftb -= UINT_BITS;
			let top = (ft >> ftb) + 1;
			let s = self.decode(top);
			self.update(s, s + 1, top);
			let t = (s << ftb) | self.bits(ftb);
			if t <= ft {
				return t;
			}
			self.error = true;
			ft
		} else {
			let total = ft + 1;
			let s = self.decode(total);
			self.update(s, s + 1, total);
			s
		}
	}

	/// `bits` (0-25) raw bits from the end of the buffer.
	pub fn bits(&mut self, bits: u32) -> u32 {
		let mut window = self.end_window;
		let mut available = self.nend_bits;
		if available < bits {
			loop {
				window |= self.read_byte_from_end() << available;
				available += SYM_BITS;
				if available > WINDOW_SIZE - SYM_BITS {
					break;
				}
			}
		}
		let ret = if bits >= 32 { window } else { window & ((1u32 << bits) - 1) };
		window = if bits >= 32 { 0 } else { window >> bits };
		available -= bits;
		self.end_window = window;
		self.nend_bits = available;
		self.nbits_total += bits as i32;
		ret
	}

	/// Bits used so far (rounded up).
	#[inline]
	pub fn tell(&self) -> i32 {
		self.nbits_total - ilog(self.rng) as i32
	}

	/// Bits used so far, in 1/8 bit.
	pub fn tell_frac(&self) -> u32 {
		let nbits = (self.nbits_total as u32) << BITRES;
		let mut l = ilog(self.rng);
		let mut r = self.rng >> (l - 16);
		for _ in 0..BITRES {
			r = (r * r) >> 15;
			let b = r >> 16;
			l = (l << 1) | b;
			r >>= b;
		}
		nbits.wrapping_sub(l)
	}

	/// Laplace-distributed integer (CELT coarse energy; `celt/laplace.c`).
	pub fn laplace(&mut self, mut fs: u32, decay: i32) -> i32 {
		const MINP: u32 = 1;
		const NMIN: u32 = 16;
		let mut val = 0i32;
		let fm = self.decode_bin(15);
		let mut fl = 0u32;
		if fm >= fs {
			val += 1;
			fl = fs;
			let ft = 32768 - MINP * (2 * NMIN) - fs;
			fs = ((ft as i64 * i64::from(16384 - decay)) >> 15) as u32 + MINP;
			while fs > MINP && fm >= fl + 2 * fs {
				fs *= 2;
				fl += fs;
				fs = (((fs - 2 * MINP) as i64 * i64::from(decay)) >> 15) as u32;
				fs += MINP;
				val += 1;
			}
			if fs <= MINP {
				let di = (fm - fl) >> 1;
				val += di as i32;
				fl += 2 * di * MINP;
			}
			if fm < fl + fs {
				val = -val;
			} else {
				fl += fs;
			}
		}
		let fh = (fl + fs).min(32768);
		let fl = fl.min(fh.saturating_sub(1));
		self.update(fl, fh, 32768);
		val
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tell_starts_at_one_bit_and_reads_zeros_past_the_end() {
		let d = RangeDecoder::new(&[]);
		assert_eq!(d.tell(), 1);
		assert_eq!(d.tell_frac(), 8);
		let mut d = RangeDecoder::new(&[0xFF, 0x00, 0x12]);
		assert_eq!(d.bits(8), 0x12);
		assert_eq!(d.bits(4), 0x0);
		for _ in 0..100 {
			d.bit_logp(1);
		}
		assert!(d.tell() > 0);
	}
}
