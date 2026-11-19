//! Program-specific information: section reassembly, CRC, PAT, PMT and the
//! DVB SDT (ISO/IEC 13818-1 §2.4.4; ETSI EN 300 468 §5.2.3).
//!
//! FFmpeg: `libavformat/mpegts.c` (`write_section_data`, `pat_cb`,
//! `pmt_cb`, `sdt_cb`).

/// Largest PSI section (`section_length` is 12 bits, header 3 bytes).
const MAX_SECTION: usize = 4096 + 3;

/// CRC-32/MPEG-2: polynomial 0x04C11DB7, initial value all ones, no
/// reflection or final XOR. A section with its CRC appended checks to 0.
const CRC_TABLE: [u32; 256] = {
	let mut table = [0u32; 256];
	let mut i = 0;
	while i < 256 {
		let mut c = (i as u32) << 24;
		let mut k = 0;
		while k < 8 {
			c = if c & 0x8000_0000 != 0 { (c << 1) ^ 0x04C1_1DB7 } else { c << 1 };
			k += 1;
		}
		table[i] = c;
		i += 1;
	}
	table
};

pub fn crc32(data: &[u8]) -> u32 {
	data.iter().fold(0xFFFF_FFFF, |crc, &b| (crc << 8) ^ CRC_TABLE[((crc >> 24) as u8 ^ b) as usize])
}

/// Reassembles the sections carried on one PID.
#[derive(Debug, Default)]
pub struct SectionBuffer {
	data: Vec<u8>,
	/// `false` until a payload unit start: bytes before it can't be placed.
	synced: bool,
}

impl SectionBuffer {
	/// Adds one TS packet payload; returns every section completed by it,
	/// CRC already checked (sections with a bad CRC are dropped).
	pub fn push(&mut self, payload: &[u8], unit_start: bool) -> Vec<Vec<u8>> {
		let mut done = Vec::new();
		if unit_start {
			let Some((&pointer, rest)) = payload.split_first() else { return done };
			let pointer = usize::from(pointer);
			if self.synced {
				// The bytes before the pointer finish the previous section.
				self.data.extend_from_slice(rest.get(..pointer).unwrap_or(rest));
				self.drain(&mut done);
			}
			self.data.clear();
			self.synced = true;
			self.data.extend_from_slice(rest.get(pointer..).unwrap_or(&[]));
		} else if self.synced {
			self.data.extend_from_slice(payload);
		}
		self.drain(&mut done);
		if self.data.len() > MAX_SECTION {
			self.data.clear();
			self.synced = false;
		}
		done
	}

	/// Takes complete sections off the front of the buffer.
	fn drain(&mut self, done: &mut Vec<Vec<u8>>) {
		loop {
			// 0xFF in place of a table id is stuffing up to the packet end.
			if self.data.first().is_none_or(|&t| t == 0xFF) {
				if !self.data.is_empty() {
					self.data.clear();
					self.synced = false;
				}
				return;
			}
			if self.data.len() < 3 {
				return;
			}
			let len = 3 + (usize::from(self.data[1] & 0x0F) << 8 | usize::from(self.data[2]));
			if self.data.len() < len {
				return;
			}
			let section: Vec<u8> = self.data.drain(..len).collect();
			// Long-form sections (all the tables read here) end with a CRC.
			if section[1] & 0x80 == 0 || (len >= 8 && crc32(&section) == 0) {
				done.push(section);
			}
		}
	}
}

/// Fields common to long-form sections, and the table body.
pub struct Section<'a> {
	pub table_id: u8,
	/// `transport_stream_id`, `program_number`, ... (table-dependent).
	pub id: u16,
	pub version: u8,
	pub current: bool,
	/// Bytes between the 8-byte header and the CRC.
	pub body: &'a [u8],
}

impl<'a> Section<'a> {
	pub fn parse(data: &'a [u8]) -> Option<Self> {
		if data.len() < 12 || data[1] & 0x80 == 0 {
			return None;
		}
		Some(Self {
			table_id: data[0],
			id: u16::from_be_bytes([data[3], data[4]]),
			version: (data[5] >> 1) & 0x1F,
			current: data[5] & 1 != 0,
			body: &data[8..data.len() - 4],
		})
	}
}

/// PAT: program number → PMT PID (the NIT entry, program 0, is skipped).
pub fn parse_pat(section: &Section) -> Vec<(u16, u16)> {
	section
		.body
		.chunks_exact(4)
		.map(|e| (u16::from_be_bytes([e[0], e[1]]), u16::from_be_bytes([e[2], e[3]]) & 0x1FFF))
		.filter(|&(program, _)| program != 0)
		.collect()
}

/// One elementary stream listed in a PMT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EsInfo {
	pub stream_type: u8,
	pub pid: u16,
	pub descriptors: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct Pmt {
	pub program: u16,
	pub pcr_pid: u16,
	/// Program-level descriptors (a registration descriptor here applies to
	/// every stream, as in FFmpeg).
	pub descriptors: Vec<u8>,
	pub streams: Vec<EsInfo>,
}

pub fn parse_pmt(section: &Section) -> Option<Pmt> {
	let body = section.body;
	if body.len() < 4 {
		return None;
	}
	let pcr_pid = u16::from_be_bytes([body[0], body[1]]) & 0x1FFF;
	let info_len = usize::from(u16::from_be_bytes([body[2], body[3]]) & 0x0FFF);
	let descriptors = body.get(4..4 + info_len)?.to_vec();
	let mut at = 4 + info_len;
	let mut streams = Vec::new();
	while at + 5 <= body.len() {
		let stream_type = body[at];
		let pid = u16::from_be_bytes([body[at + 1], body[at + 2]]) & 0x1FFF;
		let len = usize::from(u16::from_be_bytes([body[at + 3], body[at + 4]]) & 0x0FFF);
		let Some(desc) = body.get(at + 5..at + 5 + len) else { break };
		streams.push(EsInfo { stream_type, pid, descriptors: desc.to_vec() });
		at += 5 + len;
	}
	Some(Pmt { program: section.id, pcr_pid, descriptors, streams })
}

/// Iterates `(tag, payload)` over a descriptor loop.
pub fn descriptors(data: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
	let mut at = 0usize;
	std::iter::from_fn(move || {
		let tag = *data.get(at)?;
		let len = usize::from(*data.get(at + 1)?);
		let body = data.get(at + 2..at + 2 + len)?;
		at += 2 + len;
		Some((tag, body))
	})
}

/// A service from the SDT: the names FFmpeg reports as program tags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Service {
	pub id: u16,
	pub provider: String,
	pub name: String,
}

pub fn parse_sdt(section: &Section) -> Vec<Service> {
	let mut out = Vec::new();
	// original_network_id (2) + reserved (1).
	let Some(mut rest) = section.body.get(3..) else { return out };
	while rest.len() >= 5 {
		let id = u16::from_be_bytes([rest[0], rest[1]]);
		let len = usize::from(u16::from_be_bytes([rest[3], rest[4]]) & 0x0FFF);
		let Some(loop_data) = rest.get(5..5 + len) else { break };
		for (tag, d) in descriptors(loop_data) {
			// service_descriptor: type, provider, name.
			if tag != 0x48 || d.len() < 2 {
				continue;
			}
			let plen = usize::from(d[1]);
			let Some(provider) = d.get(2..2 + plen) else { continue };
			let Some(&nlen) = d.get(2 + plen) else { continue };
			let Some(name) = d.get(3 + plen..3 + plen + usize::from(nlen)) else { continue };
			out.push(Service { id, provider: dvb_string(provider), name: dvb_string(name) });
		}
		rest = &rest[5 + len..];
	}
	out
}

/// Decodes a DVB string (EN 300 468 Annex A). A leading byte below 0x20
/// selects the character table: UTF-8 (0x15) is decoded as such, the
/// default table and the ISO 8859 tables as Latin-1, which is exact for
/// ASCII names and close enough for the rest.
fn dvb_string(data: &[u8]) -> String {
	let (table, text) = match data.first() {
		Some(&b) if b < 0x20 => {
			let skip = match b {
				0x10 => 3,
				0x1F => 2,
				_ => 1,
			};
			(Some(b), data.get(skip..).unwrap_or(&[]))
		}
		_ => (None, data),
	};
	if table == Some(0x15) {
		return String::from_utf8_lossy(text).into_owned();
	}
	text.iter().filter(|&&b| !(0x80..0xA0).contains(&b)).map(|&b| char::from(b)).collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	fn long_section(table_id: u8, id: u16, body: &[u8]) -> Vec<u8> {
		let len = 5 + body.len() + 4;
		let mut s = vec![table_id, 0xB0 | (len >> 8) as u8, len as u8];
		s.extend_from_slice(&id.to_be_bytes());
		s.extend_from_slice(&[0xC1, 0, 0]);
		s.extend_from_slice(body);
		let crc = crc32(&s);
		s.extend_from_slice(&crc.to_be_bytes());
		s
	}

	#[test]
	fn crc_check_value() {
		assert_eq!(crc32(b"123456789"), 0x0376_E6E7);
	}

	#[test]
	fn pat_across_packets() {
		let pat = long_section(0, 1, &[0, 0, 0xE0, 0x10, 0, 1, 0xF0, 0]);
		let mut buf = SectionBuffer::default();
		let mut first = vec![0u8];
		first.extend_from_slice(&pat[..6]);
		assert!(buf.push(&first, true).is_empty());
		let done = buf.push(&pat[6..], false);
		assert_eq!(done.len(), 1);
		let section = Section::parse(&done[0]).unwrap();
		assert_eq!(parse_pat(&section), vec![(1, 0x1000)]);
	}

	#[test]
	fn bad_crc_is_dropped() {
		let mut pat = long_section(0, 1, &[0, 1, 0xF0, 0]);
		let last = pat.len() - 1;
		pat[last] ^= 1;
		let mut payload = vec![0u8];
		payload.extend_from_slice(&pat);
		assert!(SectionBuffer::default().push(&payload, true).is_empty());
	}

	#[test]
	fn pmt_streams_and_descriptors() {
		let body = [
			0xE1, 0x00, 0xF0, 0x00, // PCR PID 0x100, no program descriptors
			0x1B, 0xE1, 0x00, 0xF0, 0x00, // H.264 on 0x100
			0x0F, 0xE1, 0x01, 0xF0, 0x06, 0x0A, 0x04, b'e', b'n', b'g', 0, // AAC, language
		];
		let s = long_section(2, 1, &body);
		let pmt = parse_pmt(&Section::parse(&s).unwrap()).unwrap();
		assert_eq!(pmt.pcr_pid, 0x100);
		assert_eq!(pmt.streams.len(), 2);
		assert_eq!(pmt.streams[1].stream_type, 0x0F);
		let (tag, lang) = descriptors(&pmt.streams[1].descriptors).next().unwrap();
		assert_eq!((tag, &lang[..3]), (0x0A, &b"eng"[..]));
	}

	#[test]
	fn dvb_strings() {
		assert_eq!(dvb_string(b"\x15caf\xC3\xA9"), "café");
		assert_eq!(dvb_string(b"FFmpeg"), "FFmpeg");
	}
}
