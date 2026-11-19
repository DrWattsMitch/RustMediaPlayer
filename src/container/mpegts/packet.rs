//! Transport packets and PES headers (ISO/IEC 13818-1 §2.4.3, §2.4.3.6).

/// Size of the transport packet proper; M2TS (192) and DVB-ASI (204)
/// framings add a prefix or suffix around it.
pub const TS_SIZE: usize = 188;
pub const SYNC: u8 = 0x47;
pub const PID_PAT: u16 = 0x0000;
pub const PID_SDT: u16 = 0x0011;
pub const PID_NULL: u16 = 0x1FFF;

/// How packets are laid out in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
	/// Distance between packet starts: 188, 192 (M2TS / Blu-ray) or 204.
	pub stride: usize,
	/// Bytes before the sync byte inside one stride (4 for M2TS).
	pub prefix: usize,
}

/// Finds the packet size and the first packet's offset, as FFmpeg's
/// `get_packet_size` does: the stride whose grid holds the most sync bytes.
/// At least four packets must be seen and 80% of the grid must sync, which
/// chance doesn't produce but a few damaged packets don't prevent.
pub fn detect_layout(data: &[u8]) -> Option<(Layout, usize)> {
	let mut best: Option<(usize, Layout, usize)> = None;
	for (stride, prefix) in [(188, 0), (192, 4), (204, 0)] {
		for start in 0..stride.min(data.len()) {
			if data[start] != SYNC {
				continue;
			}
			let hits = (start..data.len()).step_by(stride).filter(|&i| data[i] == SYNC).count();
			let possible = (data.len() - start).div_ceil(stride);
			if hits >= 4 && hits * 5 >= possible * 4 && best.is_none_or(|(b, _, _)| hits > b) {
				let first = start.checked_sub(prefix).unwrap_or(start + stride - prefix);
				best = Some((hits, Layout { stride, prefix }, first));
			}
		}
	}
	best.map(|(_, layout, first)| (layout, first))
}

/// A transport packet header.
#[derive(Debug, Clone, Copy)]
pub struct Header {
	pub pid: u16,
	pub unit_start: bool,
	pub continuity: u8,
	pub has_payload: bool,
	/// Adaptation field flags.
	pub discontinuity: bool,
	pub random_access: bool,
	/// Offset of the payload in the 188-byte packet.
	pub payload_at: usize,
}

/// Parses a 188-byte packet. `None` for a lost sync, a transport error, or
/// an adaptation field that overruns the packet.
pub fn parse(p: &[u8]) -> Option<Header> {
	if p.len() < TS_SIZE || p[0] != SYNC || p[1] & 0x80 != 0 {
		return None;
	}
	let pid = (u16::from(p[1] & 0x1F) << 8) | u16::from(p[2]);
	let control = (p[3] >> 4) & 3;
	let mut header = Header {
		pid,
		unit_start: p[1] & 0x40 != 0,
		continuity: p[3] & 0x0F,
		has_payload: control & 1 != 0,
		discontinuity: false,
		random_access: false,
		payload_at: 4,
	};
	if control & 2 != 0 {
		let len = usize::from(p[4]);
		if 5 + len > TS_SIZE {
			return None;
		}
		if len > 0 {
			header.discontinuity = p[5] & 0x80 != 0;
			header.random_access = p[5] & 0x40 != 0;
		}
		header.payload_at = 5 + len;
	}
	Some(header)
}

/// A 33-bit PES timestamp (`marker`-separated 3 + 15 + 15 bits).
fn timestamp(b: &[u8]) -> Option<u64> {
	if b.len() < 5 || b[0] & 1 == 0 || b[2] & 1 == 0 || b[4] & 1 == 0 {
		return None;
	}
	Some(
		(u64::from(b[0] >> 1) & 7) << 30
			| u64::from(b[1]) << 22
			| u64::from(b[2] >> 1) << 15
			| u64::from(b[3]) << 7
			| u64::from(b[4] >> 1),
	)
}

/// A parsed PES packet header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pes {
	pub stream_id: u8,
	pub pts: Option<u64>,
	pub dts: Option<u64>,
	/// Offset of the elementary stream data.
	pub payload_at: usize,
	/// Total PES size from `PES_packet_length` (`None`: unbounded, video).
	pub total_len: Option<usize>,
}

/// Parses the header at the start of a PES packet. `None` when the start
/// code is missing or the header is cut short.
pub fn parse_pes(data: &[u8]) -> Option<Pes> {
	if data.len() < 6 || data[..3] != [0, 0, 1] {
		return None;
	}
	let stream_id = data[3];
	let len = usize::from(u16::from_be_bytes([data[4], data[5]]));
	let total_len = (len != 0).then_some(6 + len);
	// Stream ids without the optional header: program stream map, padding,
	// private stream 2, ECM, EMM, DSMCC, H.222.1 type E, directory.
	if matches!(stream_id, 0xBC | 0xBE | 0xBF | 0xF0 | 0xF1 | 0xF2 | 0xF8 | 0xFF) {
		return Some(Pes { stream_id, pts: None, dts: None, payload_at: 6, total_len });
	}
	if data.len() < 9 || data[6] & 0xC0 != 0x80 {
		return None;
	}
	let flags = data[7] >> 6;
	let header_len = usize::from(data[8]);
	let fields = data.get(9..9 + header_len)?;
	let (pts, dts) = match flags {
		2 => (timestamp(fields), None),
		3 => (timestamp(fields), fields.get(5..).and_then(timestamp)),
		_ => (None, None),
	};
	Some(Pes { stream_id, pts, dts, payload_at: 9 + header_len, total_len })
}

/// Undoes the 33-bit wrap of PES timestamps: the result is the value
/// congruent to `raw` modulo 2^33 that is closest to the previous one.
#[derive(Debug, Default, Clone, Copy)]
pub struct Unwrap {
	last: Option<u64>,
}

const WRAP: u64 = 1 << 33;

impl Unwrap {
	pub fn apply(&mut self, raw: u64) -> u64 {
		let raw = raw & (WRAP - 1);
		let value = match self.last {
			None => raw,
			Some(last) => {
				let base = last & !(WRAP - 1);
				[base.checked_sub(WRAP), Some(base), base.checked_add(WRAP)]
					.into_iter()
					.flatten()
					.map(|b| b + raw)
					.min_by_key(|&v| v.abs_diff(last))
					.unwrap_or(raw)
			}
		};
		self.last = Some(value);
		value
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn layouts() {
		let mut ts = vec![0u8; 188 * 6];
		for i in 0..6 {
			ts[i * 188] = SYNC;
		}
		assert_eq!(detect_layout(&ts), Some((Layout { stride: 188, prefix: 0 }, 0)));
		let mut m2ts = vec![0u8; 192 * 6];
		for i in 0..6 {
			m2ts[i * 192 + 4] = SYNC;
		}
		assert_eq!(detect_layout(&m2ts), Some((Layout { stride: 192, prefix: 4 }, 0)));
		assert_eq!(detect_layout(&ts[..188 * 2]), None);
		let mut noisy = vec![0u8; 188 * 10];
		for i in [0, 1, 2, 4, 5, 6, 7, 8, 9] {
			noisy[i * 188] = SYNC;
		}
		assert!(detect_layout(&noisy).is_some()); // one damaged packet
		for i in [1, 2, 4] {
			noisy[i * 188] = 0;
		}
		assert_eq!(detect_layout(&noisy), None);
	}

	#[test]
	fn pes_timestamps() {
		// PTS 133200, DTS 126000 (FFmpeg's first H.264 frame).
		let enc = |prefix: u8, t: u64| {
			[
				prefix << 4 | ((t >> 29) & 0x0E) as u8 | 1,
				(t >> 22) as u8,
				((t >> 14) & 0xFE) as u8 | 1,
				(t >> 7) as u8,
				((t << 1) & 0xFE) as u8 | 1,
			]
		};
		let mut pes = vec![0, 0, 1, 0xE0, 0, 0, 0x80, 0xC0, 10];
		pes.extend(enc(3, 133_200));
		pes.extend(enc(1, 126_000));
		pes.push(0xAA);
		let h = parse_pes(&pes).unwrap();
		assert_eq!((h.pts, h.dts, h.payload_at, h.total_len), (Some(133_200), Some(126_000), 19, None));
	}

	#[test]
	fn unwrap_across_33_bits() {
		let mut u = Unwrap::default();
		assert_eq!(u.apply(WRAP - 90_000), WRAP - 90_000);
		assert_eq!(u.apply(3_000), WRAP + 3_000);
		assert_eq!(u.apply(WRAP - 1_000), WRAP - 1_000); // B-frame before the wrap
		assert_eq!(u.apply(6_000), WRAP + 6_000);
	}
}
