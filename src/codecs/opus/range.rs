//! Opus range decoder and encoder (RFC 6716 §4.1, §5.1; libopus
//! `celt/entdec.c`, `celt/entenc.c`, `celt/entcode.c`, `celt/laplace.c`).
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
const CODE_SHIFT: u32 = CODE_BITS - SYM_BITS - 1;
/// Laplace model constants (`laplace.c`).
const LAPLACE_MINP: u32 = 1;
const LAPLACE_NMIN: u32 = 16;
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

/// Range encoder (`ec_enc`): symbols go to the front of a fixed-size
/// buffer, raw bits to its back, as the decoder reads them.
#[derive(Debug, Clone)]
pub struct RangeEncoder {
	buf: Vec<u8>,
	offs: usize,
	end_offs: usize,
	end_window: u32,
	nend_bits: u32,
	nbits_total: i32,
	rng: u32,
	val: u32,
	ext: u32,
	rem: i32,
	pub error: bool,
}

impl RangeEncoder {
	/// An encoder for a packet of exactly `size` bytes.
	pub fn new(size: usize) -> Self {
		Self {
			buf: vec![0; size],
			offs: 0,
			end_offs: 0,
			end_window: 0,
			nend_bits: 0,
			nbits_total: (CODE_BITS + 1) as i32,
			rng: CODE_TOP,
			val: 0,
			ext: 0,
			rem: -1,
			error: false,
		}
	}

	/// Final range (`OPUS_GET_FINAL_RANGE`): equals the decoder's after the
	/// same packet.
	pub fn range(&self) -> u32 {
		self.rng
	}

	pub fn storage(&self) -> usize {
		self.buf.len()
	}

	fn write_byte(&mut self, v: u32) {
		if self.offs + self.end_offs >= self.buf.len() {
			self.error = true;
			return;
		}
		self.buf[self.offs] = v as u8;
		self.offs += 1;
	}

	fn write_byte_at_end(&mut self, v: u32) {
		if self.offs + self.end_offs >= self.buf.len() {
			self.error = true;
			return;
		}
		self.end_offs += 1;
		let at = self.buf.len() - self.end_offs;
		self.buf[at] = v as u8;
	}

	fn carry_out(&mut self, c: u32) {
		if c != SYM_MAX {
			let carry = c >> SYM_BITS;
			if self.rem >= 0 {
				self.write_byte(self.rem as u32 + carry);
			}
			if self.ext > 0 {
				let sym = (SYM_MAX + carry) & SYM_MAX;
				while self.ext > 0 {
					self.write_byte(sym);
					self.ext -= 1;
				}
			}
			self.rem = (c & SYM_MAX) as i32;
		} else {
			self.ext += 1;
		}
	}

	#[inline]
	fn normalize(&mut self) {
		while self.rng <= CODE_BOT {
			self.carry_out(self.val >> CODE_SHIFT);
			self.val = (self.val << SYM_BITS) & (CODE_TOP - 1);
			self.rng <<= SYM_BITS;
			self.nbits_total += SYM_BITS as i32;
		}
	}

	/// Codes the symbol occupying `[fl, fh)` of `ft` (`ec_encode`).
	pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
		let r = self.rng / ft;
		if fl > 0 {
			self.val = self.val.wrapping_add(self.rng - r * (ft - fl));
			self.rng = r * (fh - fl);
		} else {
			self.rng -= r * (ft - fh);
		}
		self.normalize();
	}

	/// [`Self::encode`] with `ft = 1 << bits`.
	pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
		let r = self.rng >> bits;
		if fl > 0 {
			self.val = self.val.wrapping_add(self.rng - r * ((1 << bits) - fl));
			self.rng = r * (fh - fl);
		} else {
			self.rng -= r * ((1 << bits) - fh);
		}
		self.normalize();
	}

	pub fn bit_logp(&mut self, val: bool, logp: u32) {
		let r = self.rng;
		let s = r >> logp;
		if val {
			self.val = self.val.wrapping_add(r - s);
			self.rng = s;
		} else {
			self.rng = r - s;
		}
		self.normalize();
	}

	pub fn icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
		let r = self.rng >> ftb;
		if s > 0 {
			self.val = self.val.wrapping_add(self.rng - r * u32::from(icdf[s - 1]));
			self.rng = r * u32::from(icdf[s - 1] - icdf[s]);
		} else {
			self.rng -= r * u32::from(icdf[s]);
		}
		self.normalize();
	}

	pub fn uint(&mut self, fl: u32, ft: u32) {
		if ft <= 1 {
			return;
		}
		let ft = ft - 1;
		let mut ftb = ilog(ft);
		if ftb > UINT_BITS {
			ftb -= UINT_BITS;
			let top = (ft >> ftb) + 1;
			let f = fl >> ftb;
			self.encode(f, f + 1, top);
			self.bits(fl & ((1 << ftb) - 1), ftb);
		} else {
			self.encode(fl, fl + 1, ft + 1);
		}
	}

	pub fn bits(&mut self, fl: u32, bits: u32) {
		let mut window = self.end_window;
		let mut used = self.nend_bits;
		if used + bits > WINDOW_SIZE {
			loop {
				self.write_byte_at_end(window & SYM_MAX);
				window >>= SYM_BITS;
				used -= SYM_BITS;
				if used < SYM_BITS {
					break;
				}
			}
		}
		window |= fl << used;
		used += bits;
		self.end_window = window;
		self.nend_bits = used;
		self.nbits_total += bits as i32;
	}

	/// Laplace-distributed integer (`ec_laplace_encode`). Values beyond
	/// the model's range are clamped; returns the value coded.
	pub fn laplace(&mut self, value: i32, mut fs: u32, decay: i32) -> i32 {
		let mut val = value;
		let mut fl = 0u32;
		if val != 0 {
			let s = -i32::from(val < 0);
			val = (val + s) ^ s;
			fl = fs;
			let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs;
			fs = ((ft as i64 * i64::from(16384 - decay)) >> 15) as u32;
			let mut i = 1;
			while fs > 0 && i < val {
				fs *= 2;
				fl += fs + 2 * LAPLACE_MINP;
				fs = ((fs as i64 * i64::from(decay)) >> 15) as u32;
				i += 1;
			}
			if fs == 0 {
				let ndi_max = (32768 - fl).div_ceil(LAPLACE_MINP) as i32;
				let ndi_max = (ndi_max - s) >> 1;
				let di = (val - i).min(ndi_max - 1);
				fl = (fl as i32 + (2 * di + 1 + s) * LAPLACE_MINP as i32) as u32;
				fs = LAPLACE_MINP.min(32768 - fl);
				val = (i + di + s) ^ s;
			} else {
				fs += LAPLACE_MINP;
				if s == 0 {
					fl += fs;
				}
				val = value;
			}
		}
		self.encode_bin(fl, fl + fs, 15);
		val
	}

	pub fn tell(&self) -> i32 {
		self.nbits_total - ilog(self.rng) as i32
	}

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

	/// Flushes (`ec_enc_done`) and returns the packet bytes.
	pub fn finish(mut self) -> Vec<u8> {
		let mut l = CODE_BITS as i32 - ilog(self.rng) as i32;
		let mut msk = (CODE_TOP - 1) >> l;
		let mut end = self.val.wrapping_add(msk) & !msk;
		if (end | msk) >= self.val.wrapping_add(self.rng) {
			l += 1;
			msk >>= 1;
			end = self.val.wrapping_add(msk) & !msk;
		}
		while l > 0 {
			self.carry_out(end >> CODE_SHIFT);
			end = (end << SYM_BITS) & (CODE_TOP - 1);
			l -= SYM_BITS as i32;
		}
		if self.rem >= 0 || self.ext > 0 {
			self.carry_out(0);
		}
		let mut window = self.end_window;
		let mut used = self.nend_bits;
		while used >= SYM_BITS {
			self.write_byte_at_end(window & SYM_MAX);
			window >>= SYM_BITS;
			used -= SYM_BITS;
		}
		if !self.error {
			let (offs, end_offs, len) = (self.offs, self.end_offs, self.buf.len());
			self.buf[offs..len - end_offs].fill(0);
			if used > 0 {
				if end_offs >= len {
					self.error = true;
				} else {
					let l = -l;
					if offs + end_offs >= len && (l as u32) < used {
						window &= (1 << l) - 1;
						self.error = true;
					}
					self.buf[len - end_offs - 1] |= window as u8;
				}
			}
		}
		self.buf
	}
}

/// What the CELT band code needs from an entropy coder: the decoder reads
/// each symbol (ignoring the value passed in), the encoder writes the value
/// passed in. Every method returns the symbol coded. This is libopus's
/// shared `quant_*` code with its `encode` flag as a type.
pub trait Coder {
	const ENCODE: bool;
	fn ec_tell(&self) -> i32;
	fn ec_tell_frac(&self) -> u32;
	/// Total packet bytes.
	fn ec_storage(&self) -> usize;
	fn code_bit_logp(&mut self, val: bool, logp: u32) -> bool;
	fn code_uint(&mut self, val: u32, ft: u32) -> u32;
	fn code_bits(&mut self, val: u32, bits: u32) -> u32;
	fn code_icdf(&mut self, val: usize, icdf: &[u8], ftb: u32) -> usize;
	fn code_laplace(&mut self, val: i32, fs: u32, decay: i32) -> i32;
	/// A symbol of a custom distribution: `pick` maps a decoded cumulative
	/// frequency (decoder) or nothing (encoder, `None`) to the symbol and
	/// its `[fl, fh)`.
	fn code_symbol(&mut self, ft: u32, pick: impl FnOnce(Option<u32>) -> (i32, u32, u32)) -> i32;
}

impl Coder for RangeDecoder<'_> {
	const ENCODE: bool = false;
	fn ec_tell(&self) -> i32 {
		self.tell()
	}
	fn ec_tell_frac(&self) -> u32 {
		self.tell_frac()
	}
	fn ec_storage(&self) -> usize {
		self.storage()
	}
	fn code_bit_logp(&mut self, _val: bool, logp: u32) -> bool {
		self.bit_logp(logp)
	}
	fn code_uint(&mut self, _val: u32, ft: u32) -> u32 {
		self.uint(ft)
	}
	fn code_bits(&mut self, _val: u32, bits: u32) -> u32 {
		self.bits(bits)
	}
	fn code_icdf(&mut self, _val: usize, icdf: &[u8], ftb: u32) -> usize {
		self.icdf(icdf, ftb)
	}
	fn code_laplace(&mut self, _val: i32, fs: u32, decay: i32) -> i32 {
		self.laplace(fs, decay)
	}
	fn code_symbol(&mut self, ft: u32, pick: impl FnOnce(Option<u32>) -> (i32, u32, u32)) -> i32 {
		let fs = self.decode(ft);
		let (x, fl, fh) = pick(Some(fs));
		self.update(fl, fh, ft);
		x
	}
}

impl Coder for RangeEncoder {
	const ENCODE: bool = true;
	fn ec_tell(&self) -> i32 {
		self.tell()
	}
	fn ec_tell_frac(&self) -> u32 {
		self.tell_frac()
	}
	fn ec_storage(&self) -> usize {
		self.storage()
	}
	fn code_bit_logp(&mut self, val: bool, logp: u32) -> bool {
		self.bit_logp(val, logp);
		val
	}
	fn code_uint(&mut self, val: u32, ft: u32) -> u32 {
		self.uint(val, ft);
		val
	}
	fn code_bits(&mut self, val: u32, bits: u32) -> u32 {
		self.bits(val, bits);
		val
	}
	fn code_icdf(&mut self, val: usize, icdf: &[u8], ftb: u32) -> usize {
		self.icdf(val, icdf, ftb);
		val
	}
	fn code_laplace(&mut self, val: i32, fs: u32, decay: i32) -> i32 {
		self.laplace(val, fs, decay)
	}
	fn code_symbol(&mut self, ft: u32, pick: impl FnOnce(Option<u32>) -> (i32, u32, u32)) -> i32 {
		let (x, fl, fh) = pick(None);
		self.encode(fl, fh, ft);
		x
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

	#[test]
	fn encoder_round_trips_through_the_decoder() {
		let mut seed = 99u32;
		let mut rnd = |n: u32| {
			seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			(seed >> 8) % n
		};
		const ICDF: [u8; 4] = [25, 23, 2, 0];
		for size in [8usize, 40, 300] {
			let mut ops = Vec::new();
			let mut enc = RangeEncoder::new(size);
			loop {
				if enc.tell() + 40 > (size * 8) as i32 {
					break;
				}
				let op = match rnd(6) {
					0 => {
						let v = rnd(2) == 1;
						let logp = 1 + rnd(14);
						enc.bit_logp(v, logp);
						(0, v as i32, logp as i32)
					}
					1 => {
						let ft = 2 + rnd(5000);
						let v = rnd(ft);
						enc.uint(v, ft);
						(1, v as i32, ft as i32)
					}
					2 => {
						let n = 1 + rnd(16);
						let v = rnd(1 << n);
						enc.bits(v, n);
						(2, v as i32, n as i32)
					}
					3 => {
						let v = rnd(3) as usize;
						enc.icdf(v, &ICDF, 5);
						(3, v as i32, 0)
					}
					4 => {
						let v = rnd(41) as i32 - 20;
						let got = enc.laplace(v, 72 << 7, 127 << 6);
						(4, got, 0)
					}
					_ => {
						let ft = 2 + rnd(300);
						let v = rnd(ft);
						enc.encode(v, v + 1, ft);
						(5, v as i32, ft as i32)
					}
				};
				ops.push(op);
			}
			let rng = enc.range();
			let tell = enc.tell();
			let data = enc.finish();
			assert_eq!(data.len(), size);
			let mut dec = RangeDecoder::new(&data);
			for (n, &(kind, v, p)) in ops.iter().enumerate() {
				let got = match kind {
					0 => dec.bit_logp(p as u32) as i32,
					1 => dec.uint(p as u32) as i32,
					2 => dec.bits(p as u32) as i32,
					3 => dec.icdf(&ICDF, 5) as i32,
					4 => dec.laplace(72 << 7, 127 << 6),
					_ => {
						let s = dec.decode(p as u32);
						dec.update(s, s + 1, p as u32);
						s as i32
					}
				};
				assert_eq!(got, v, "size {size}, op {n} (kind {kind})");
			}
			assert_eq!(dec.range(), rng, "final range, size {size}");
			assert_eq!(dec.tell(), tell);
		}
	}
}
