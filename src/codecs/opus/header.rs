//! The Opus identification header (`OpusHead`, RFC 7845 §5.1) and its MP4
//! form (`dOps`, "Encapsulation of Opus in ISO Base Media File Format" §4.3).
//!
//! RMP's Opus extradata is FFmpeg's: the `OpusHead` packet as stored in Ogg
//! and in Matroska `CodecPrivate` (little-endian, with the magic). MP4 stores
//! the same fields big-endian without the magic, so the MP4 demuxer and
//! muxer convert with [`OpusHead::from_dops`] / [`OpusHead::to_dops`].

pub const MAGIC: &[u8; 8] = b"OpusHead";

/// Channel mapping of an Opus stream (RFC 7845 §5.1.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMapping {
	pub family: u8,
	/// Number of Opus streams in each packet.
	pub streams: u8,
	/// How many of them are coupled (stereo) streams.
	pub coupled: u8,
	/// Output channel → decoded channel index (255 = silence).
	pub mapping: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
	pub version: u8,
	pub channels: u8,
	pub pre_skip: u16,
	pub input_sample_rate: u32,
	/// Output gain in Q7.8 dB.
	pub output_gain: i16,
	pub mapping: ChannelMapping,
}

impl OpusHead {
	/// Default header for `channels` with mapping family 0 (mono/stereo).
	pub fn new(channels: u8, pre_skip: u16, input_sample_rate: u32) -> Self {
		Self {
			version: 1,
			channels,
			pre_skip,
			input_sample_rate,
			output_gain: 0,
			mapping: ChannelMapping {
				family: 0,
				streams: 1,
				coupled: u8::from(channels > 1),
				mapping: if channels > 1 { vec![0, 1] } else { vec![0] },
			},
		}
	}

	fn validate(self) -> Option<Self> {
		let m = &self.mapping;
		let ok = self.channels > 0
			&& m.streams > 0
			&& m.coupled <= m.streams
			&& usize::from(m.streams) + usize::from(m.coupled) <= 255
			&& m.mapping.len() == usize::from(self.channels)
			&& m
				.mapping
				.iter()
				.all(|&i| i == 255 || u16::from(i) < u16::from(m.streams) + u16::from(m.coupled));
		ok.then_some(self)
	}

	fn parse_fields(
		version: u8,
		channels: u8,
		pre_skip: u16,
		rate: u32,
		gain: i16,
		family: u8,
		table: &[u8],
	) -> Option<Self> {
		let mapping = if family == 0 {
			if channels > 2 {
				return None;
			}
			ChannelMapping {
				family,
				streams: 1,
				coupled: u8::from(channels > 1),
				mapping: (0..channels).collect(),
			}
		} else {
			let streams = *table.first()?;
			let coupled = *table.get(1)?;
			let mapping = table.get(2..2 + usize::from(channels))?.to_vec();
			ChannelMapping { family, streams, coupled, mapping }
		};
		Self { version, channels, pre_skip, input_sample_rate: rate, output_gain: gain, mapping }
			.validate()
	}

	/// Parses an `OpusHead` packet.
	pub fn parse(data: &[u8]) -> Option<Self> {
		let rest = data.strip_prefix(MAGIC)?;
		if rest.len() < 11 || rest[0] >> 4 != 0 {
			return None; // only major version 0 is defined
		}
		Self::parse_fields(
			rest[0],
			rest[1],
			u16::from_le_bytes([rest[2], rest[3]]),
			u32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]),
			i16::from_le_bytes([rest[8], rest[9]]),
			rest[10],
			&rest[11..],
		)
	}

	/// Parses an MP4 `dOps` box body.
	pub fn from_dops(data: &[u8]) -> Option<Self> {
		if data.len() < 11 || data[0] != 0 {
			return None;
		}
		Self::parse_fields(
			1,
			data[1],
			u16::from_be_bytes([data[2], data[3]]),
			u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
			i16::from_be_bytes([data[8], data[9]]),
			data[10],
			&data[11..],
		)
	}

	/// Serialises an `OpusHead` packet.
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut out = MAGIC.to_vec();
		out.push(self.version);
		out.push(self.channels);
		out.extend_from_slice(&self.pre_skip.to_le_bytes());
		out.extend_from_slice(&self.input_sample_rate.to_le_bytes());
		out.extend_from_slice(&self.output_gain.to_le_bytes());
		out.push(self.mapping.family);
		if self.mapping.family != 0 {
			out.push(self.mapping.streams);
			out.push(self.mapping.coupled);
			out.extend_from_slice(&self.mapping.mapping);
		}
		out
	}

	/// Serialises a `dOps` box body.
	pub fn to_dops(&self) -> Vec<u8> {
		let mut out = vec![0, self.channels];
		out.extend_from_slice(&self.pre_skip.to_be_bytes());
		out.extend_from_slice(&self.input_sample_rate.to_be_bytes());
		out.extend_from_slice(&self.output_gain.to_be_bytes());
		out.push(self.mapping.family);
		if self.mapping.family != 0 {
			out.push(self.mapping.streams);
			out.push(self.mapping.coupled);
			out.extend_from_slice(&self.mapping.mapping);
		}
		out
	}
}

/// Samples per frame (at 48 kHz) of the configuration in a TOC byte
/// (RFC 6716 §3.1, Table 2).
pub fn frame_samples(toc: u8) -> u32 {
	let config = u32::from(toc >> 3);
	match config {
		// SILK-only: 10, 20, 40, 60 ms.
		0..=11 => [480, 960, 1920, 2880][(config & 3) as usize],
		// Hybrid: 10, 20 ms.
		12..=15 => [480, 960][(config & 1) as usize],
		// CELT-only: 2.5, 5, 10, 20 ms.
		_ => [120, 240, 480, 960][(config & 3) as usize],
	}
}

/// Number of 48 kHz samples in an Opus packet (RFC 6716 §3.2), or `None`
/// for a malformed packet. Like libopus, more than 120 ms is invalid.
pub fn packet_samples(packet: &[u8]) -> Option<u32> {
	let toc = *packet.first()?;
	let frames = match toc & 3 {
		0 => 1,
		1 | 2 => 2,
		_ => u32::from(*packet.get(1)? & 0x3F),
	};
	let total = frames * frame_samples(toc);
	(frames > 0 && total <= 5760).then_some(total)
}

/// Pre-skip of an `OpusHead` extradata, in 48 kHz samples.
pub fn pre_skip(extradata: &[u8]) -> Option<u16> {
	OpusHead::parse(extradata).map(|h| h.pre_skip)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn head_and_dops_round_trip() {
		let head = OpusHead::new(2, 312, 44_100);
		let bytes = head.to_bytes();
		assert_eq!(bytes.len(), 19);
		assert_eq!(OpusHead::parse(&bytes), Some(head.clone()));
		assert_eq!(OpusHead::from_dops(&head.to_dops()), Some(head.clone()));
		let mut surround = head;
		surround.channels = 6;
		surround.mapping =
			ChannelMapping { family: 1, streams: 4, coupled: 2, mapping: vec![0, 4, 1, 2, 3, 5] };
		assert_eq!(OpusHead::parse(&surround.to_bytes()), Some(surround.clone()));
		assert_eq!(OpusHead::from_dops(&surround.to_dops()), Some(surround));
		assert!(OpusHead::parse(b"OpusHead\x01\x03").is_none());
	}

	#[test]
	fn packet_durations() {
		assert_eq!(packet_samples(&[0xFC]), Some(960)); // CELT FB 20 ms, 1 frame
		assert_eq!(packet_samples(&[0x19]), Some(5760)); // SILK 60 ms, 2 frames (code 1)
		assert_eq!(packet_samples(&[0x0B, 0]), None); // code 3 with zero frames
		assert_eq!(packet_samples(&[0x03 | (16 << 3), 0x05]), Some(600)); // 5 x 2.5 ms
		assert_eq!(packet_samples(&[0x1B, 0x03]), None); // 3 x 60 ms > 120 ms
		assert_eq!(packet_samples(&[]), None);
	}
}
