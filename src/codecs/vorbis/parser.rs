//! Vorbis packet durations, without decoding (FFmpeg's
//! `libavcodec/vorbis_parser.c`).
//!
//! A Vorbis audio packet decodes to `(previous_blocksize + blocksize) / 4`
//! samples, where each block size is one of the two in the identification
//! header, chosen by the packet's mode. The mode → block flag table sits at
//! the end of the setup header, after the codebooks, floors, residues and
//! mappings, which can only be walked by a full decoder. FFmpeg's parser
//! instead reads the setup header **backwards** from the framing bit: each
//! mode is 41 bits (block flag, two 16-bit zero fields, an 8-bit mapping
//! number), preceded by a 6-bit mode count, so the table can be found
//! without parsing what comes before it. RMP does the same.
//!
//! Containers use this to time Vorbis packets (Ogg granule positions).

/// Durations of the packets of one Vorbis stream.
#[derive(Debug, Clone)]
pub struct VorbisParser {
	blocksize: [u32; 2],
	mode_blockflag: Vec<bool>,
	mode_mask: u8,
	/// Block size of the previous packet; `None` after [`Self::reset`].
	previous: Option<u32>,
}

/// MSB-first reader over the reversed setup header, which visits the
/// original (LSB-first packed) bitstream from its end.
struct Reverse<'a> {
	data: &'a [u8],
	pos: usize,
}

impl Reverse<'_> {
	fn left(&self) -> usize {
		(self.data.len() * 8).saturating_sub(self.pos)
	}

	fn bits(&mut self, n: usize) -> u32 {
		let mut value = 0u32;
		for _ in 0..n {
			let byte = self.data.get(self.pos / 8).copied().unwrap_or(0);
			let bit = (byte >> (7 - self.pos % 8)) & 1;
			value = (value << 1) | u32::from(bit);
			self.pos += 1;
		}
		value
	}
}

impl VorbisParser {
	/// Builds the parser from the identification and setup headers.
	pub fn new(identification: &[u8], setup: &[u8]) -> Option<Self> {
		if identification.len() < 30 || identification.get(..7) != Some(b"\x01vorbis") {
			return None;
		}
		let sizes = identification[28];
		let blocksize = [1u32 << (sizes & 0x0F), 1u32 << (sizes >> 4)];
		if !(64..=8192).contains(&blocksize[0]) || blocksize[1] < blocksize[0] || blocksize[1] > 8192 {
			return None;
		}
		if setup.len() < 8 || setup.get(..7) != Some(b"\x05vorbis") {
			return None;
		}
		let reversed: Vec<u8> = setup.iter().rev().copied().collect();

		// The framing bit is the last bit set.
		let mut r = Reverse { data: &reversed, pos: 0 };
		let mut framing = None;
		while r.left() > 97 {
			if r.bits(1) == 1 {
				framing = Some(r.pos);
				break;
			}
		}
		let framing = framing?;

		// Walk modes backwards while they look like modes; the count that
		// matches the 6-bit field before a candidate is the mode count.
		let mut mode_count = 0usize;
		let mut found = None;
		while r.left() >= 97 {
			if r.bits(8) > 63 || r.bits(16) != 0 || r.bits(16) != 0 {
				break;
			}
			r.bits(1);
			mode_count += 1;
			if mode_count > 64 {
				break;
			}
			let mut ahead = Reverse { data: &reversed, pos: r.pos };
			if ahead.bits(6) as usize + 1 == mode_count {
				found = Some(mode_count);
			}
		}
		let mode_count = found?;

		let mut r = Reverse { data: &reversed, pos: framing };
		let mut mode_blockflag = vec![false; mode_count];
		for flag in mode_blockflag.iter_mut().rev() {
			r.bits(40);
			*flag = r.bits(1) == 1;
		}
		let bits = usize::BITS - (mode_count - 1).leading_zeros();
		let mode_mask = (((1u32 << bits) - 1) << 1) as u8;
		Some(Self { blocksize, mode_blockflag, mode_mask, previous: Some(blocksize[0]) })
	}

	/// From FFmpeg-style extradata (see [`crate::container::xiph::vorbis_headers`]).
	pub fn from_extradata(extradata: &[u8]) -> Option<Self> {
		let headers = crate::container::xiph::vorbis_headers(extradata)?;
		Self::new(&headers[0], &headers[2])
	}

	/// Forgets the previous block (after a seek): the next packet's
	/// duration is unknown, the ones after it are exact again.
	pub fn reset(&mut self) {
		self.previous = None;
	}

	/// Samples the packet adds, or `None` for a header or invalid packet
	/// (or the first packet after [`Self::reset`]). The first audio packet
	/// of a stream counts `blocksize / 4` samples that a decoder discards,
	/// as FFmpeg reports it (its pts is negative).
	pub fn packet_samples(&mut self, packet: &[u8]) -> Option<u32> {
		let first = *packet.first()?;
		if first & 1 != 0 {
			return None;
		}
		let mode =
			if self.mode_blockflag.len() == 1 { 0 } else { usize::from((first & self.mode_mask) >> 1) };
		let long = *self.mode_blockflag.get(mode)?;
		let current = self.blocksize[usize::from(long)];
		let samples = self.previous.map(|previous| (previous + current) >> 2);
		self.previous = Some(current);
		samples
	}
}
