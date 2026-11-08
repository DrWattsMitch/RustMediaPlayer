//! AAC stream configuration: AudioSpecificConfig (MP4 `esds` extradata),
//! ADTS headers (raw `.aac` streams) and program config elements.
//!
//! FFmpeg: `libavcodec/mpeg4audio.c`, `adts_header.c`, `aac/aacdec.c`
//! (`decode_pce`, `decode_ga_specific_config`).

use crate::codecs::bitstream::BitReader;

/// Sampling rates by `sampling_frequency_index` (ISO/IEC 14496-3 Table 1.18).
pub const SAMPLE_RATES: [u32; 13] =
	[96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

pub const AOT_AAC_MAIN: u8 = 1;
pub const AOT_AAC_LC: u8 = 2;
pub const AOT_SBR: u8 = 5;
pub const AOT_PS: u8 = 29;

/// Syntactic element ids (`id_syn_ele`, Table 4.85).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
	Sce,
	Cpe,
	Cce,
	Lfe,
	Dse,
	Pce,
	Fil,
	End,
}

impl ElementKind {
	pub fn from_id(id: u32) -> Self {
		match id {
			0 => Self::Sce,
			1 => Self::Cpe,
			2 => Self::Cce,
			3 => Self::Lfe,
			4 => Self::Dse,
			5 => Self::Pce,
			6 => Self::Fil,
			_ => Self::End,
		}
	}
}

/// One audio element of the channel layout and the output channel(s) it
/// feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementSlot {
	pub kind: ElementKind,
	pub tag: u8,
	/// Output channel of the first (or only) channel of the element.
	pub out: [u8; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AacConfig {
	pub object_type: u8,
	pub sample_rate_index: usize,
	pub sample_rate: u32,
	/// Channel layout in bitstream element order.
	pub layout: Vec<ElementSlot>,
	pub channels: u8,
	/// `true` when the stream signals SBR/PS (HE-AAC). Only the AAC-LC core
	/// is decoded, at the core sample rate.
	pub sbr: bool,
}

impl AacConfig {
	/// Layout for `channelConfiguration` 1-7, ordered like FFmpeg's native
	/// channel order (FL FR FC LFE BL BR …).
	pub fn from_channel_config(
		object_type: u8,
		sample_rate_index: usize,
		channel_config: u8,
	) -> Option<Self> {
		use ElementKind::*;
		let slot = |kind, tag, a: u8, b: u8| ElementSlot { kind, tag, out: [a, b] };
		let layout = match channel_config {
			1 => vec![slot(Sce, 0, 0, 0)],
			2 => vec![slot(Cpe, 0, 0, 1)],
			3 => vec![slot(Sce, 0, 2, 2), slot(Cpe, 0, 0, 1)],
			4 => vec![slot(Sce, 0, 2, 2), slot(Cpe, 0, 0, 1), slot(Sce, 1, 3, 3)],
			5 => vec![slot(Sce, 0, 2, 2), slot(Cpe, 0, 0, 1), slot(Cpe, 1, 3, 4)],
			6 => vec![slot(Sce, 0, 2, 2), slot(Cpe, 0, 0, 1), slot(Cpe, 1, 4, 5), slot(Lfe, 0, 3, 3)],
			7 => vec![
				slot(Sce, 0, 2, 2),
				slot(Cpe, 0, 6, 7),
				slot(Cpe, 1, 0, 1),
				slot(Cpe, 2, 4, 5),
				slot(Lfe, 0, 3, 3),
			],
			_ => return None,
		};
		Self::with_layout(object_type, sample_rate_index, layout)
	}

	fn with_layout(
		object_type: u8,
		sample_rate_index: usize,
		layout: Vec<ElementSlot>,
	) -> Option<Self> {
		let sample_rate = *SAMPLE_RATES.get(sample_rate_index)?;
		let channels =
			layout.iter().map(|s| if s.kind == ElementKind::Cpe { 2 } else { 1 }).sum::<u8>();
		if channels == 0 || channels > 8 {
			return None;
		}
		Some(Self { object_type, sample_rate_index, sample_rate, layout, channels, sbr: false })
	}

	/// Finds the layout slot for the `nth` element of `kind` with `tag`.
	/// Streams frequently number instance tags loosely, so the tag is used
	/// only to break ties; the element's position among elements of its kind
	/// decides.
	pub fn slot(&self, kind: ElementKind, nth: usize) -> Option<ElementSlot> {
		self.layout.iter().filter(|s| s.kind == kind).nth(nth).copied()
	}
}

fn object_type(bs: &mut BitReader) -> u8 {
	let aot = bs.bits(5) as u8;
	if aot == 31 { 32 + bs.bits(6) as u8 } else { aot }
}

fn sample_rate_index(bs: &mut BitReader) -> Option<usize> {
	let index = bs.bits(4) as usize;
	if index == 15 {
		// Explicit rate: map to the nearest table entry, as FFmpeg does for
		// band tables.
		let rate = bs.bits(24);
		return Some(nearest_rate_index(rate));
	}
	(index < SAMPLE_RATES.len()).then_some(index)
}

/// Index of the table rate used for band layout at an arbitrary rate
/// (ISO/IEC 14496-3 Table 4.82).
pub fn nearest_rate_index(rate: u32) -> usize {
	const LIMITS: [u32; 12] =
		[92017, 75132, 55426, 46009, 37566, 27713, 23004, 18783, 13856, 11502, 9391, 0];
	LIMITS.iter().position(|&l| rate >= l).unwrap_or(11)
}

/// Parses an AudioSpecificConfig (ISO/IEC 14496-3 §1.6.2.1).
pub fn parse_audio_specific_config(data: &[u8]) -> Option<AacConfig> {
	let mut bs = BitReader::new(data);
	let mut aot = object_type(&mut bs);
	let mut rate_index = sample_rate_index(&mut bs)?;
	let channel_config = bs.bits(4) as u8;
	let mut sbr = false;
	if aot == AOT_SBR || aot == AOT_PS {
		sbr = true;
		// Extension (output) rate, then the core object type.
		sample_rate_index(&mut bs)?;
		aot = object_type(&mut bs);
	}
	if !matches!(aot, AOT_AAC_MAIN | AOT_AAC_LC | 3 | 4) {
		return None;
	}
	// GASpecificConfig.
	if bs.bit() {
		// frameLengthFlag: 960-sample frames are not supported.
		return None;
	}
	if bs.bit() {
		bs.skip(14); // coreCoderDelay
	}
	let _extension_flag = bs.bit();
	let mut config = if channel_config == 0 {
		let (pce_rate, layout) = parse_program_config(&mut bs)?;
		if pce_rate != rate_index {
			rate_index = pce_rate;
		}
		AacConfig::with_layout(aot, rate_index, layout)?
	} else {
		AacConfig::from_channel_config(aot, rate_index, channel_config)?
	};
	if bs.overrun() {
		return None;
	}
	config.sbr = sbr;
	Some(config)
}

/// Parses a program_config_element (Table 4.2) into an element layout.
/// Returns the PCE's sampling-frequency index and the layout, with output
/// channels assigned in front, side, back, LFE order.
pub fn parse_program_config(bs: &mut BitReader) -> Option<(usize, Vec<ElementSlot>)> {
	bs.skip(4); // element_instance_tag
	bs.skip(2); // object_type
	let rate_index = bs.bits(4) as usize;
	let front = bs.bits(4);
	let side = bs.bits(4);
	let back = bs.bits(4);
	let lfe = bs.bits(2);
	let assoc = bs.bits(3);
	let cc = bs.bits(4);
	if bs.bit() {
		bs.skip(4); // mono mixdown
	}
	if bs.bit() {
		bs.skip(4);
	}
	if bs.bit() {
		bs.skip(3);
	}
	let mut layout = Vec::new();
	let mut next = 0u8;
	for _ in 0..front + side + back {
		let is_cpe = bs.bit();
		let tag = bs.bits(4) as u8;
		let (kind, out) = if is_cpe {
			(ElementKind::Cpe, [next, next.saturating_add(1)])
		} else {
			(ElementKind::Sce, [next, next])
		};
		next = next.saturating_add(if is_cpe { 2 } else { 1 });
		layout.push(ElementSlot { kind, tag, out });
	}
	for _ in 0..lfe {
		let tag = bs.bits(4) as u8;
		layout.push(ElementSlot { kind: ElementKind::Lfe, tag, out: [next, next] });
		next = next.saturating_add(1);
	}
	bs.skip(4 * assoc as usize);
	bs.skip(5 * cc as usize);
	bs.align();
	let comment = bs.bits(8) as usize;
	bs.skip(comment * 8);
	if bs.overrun() || rate_index >= SAMPLE_RATES.len() {
		return None;
	}
	Some((rate_index, layout))
}

/// A parsed ADTS frame header (ISO/IEC 13818-7 §6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsHeader {
	pub object_type: u8,
	pub sample_rate_index: usize,
	pub channel_config: u8,
	/// Whole frame size including the header.
	pub frame_length: usize,
	/// Header size (7, or 9 with CRC).
	pub header_length: usize,
	pub raw_blocks: usize,
}

pub const ADTS_HEADER_SIZE: usize = 7;

impl AdtsHeader {
	pub fn parse(data: &[u8]) -> Option<Self> {
		if data.len() < ADTS_HEADER_SIZE || data[0] != 0xFF || data[1] & 0xF6 != 0xF0 {
			return None;
		}
		let mut bs = BitReader::new(data);
		bs.skip(12 + 1 + 2); // syncword, ID, layer
		let crc_absent = bs.bit();
		let object_type = bs.bits(2) as u8 + 1;
		let sample_rate_index = bs.bits(4) as usize;
		bs.skip(1);
		let channel_config = bs.bits(3) as u8;
		bs.skip(4);
		let frame_length = bs.bits(13) as usize;
		bs.skip(11);
		let raw_blocks = bs.bits(2) as usize + 1;
		let header_length = if crc_absent { 7 } else { 9 };
		if sample_rate_index >= SAMPLE_RATES.len() || frame_length < header_length {
			return None;
		}
		Some(Self {
			object_type,
			sample_rate_index,
			channel_config,
			frame_length,
			header_length,
			raw_blocks,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_lc_stereo_asc() {
		// AAC-LC, 44.1 kHz, stereo: 00010 0100 0010 000
		let c = parse_audio_specific_config(&[0x12, 0x10]).unwrap();
		assert_eq!((c.object_type, c.sample_rate, c.channels), (2, 44100, 2));
		assert!(!c.sbr);
	}

	#[test]
	fn parses_he_aac_asc() {
		// AOT 5, 24 kHz core, stereo, 48 kHz extension, AOT 2.
		let c = parse_audio_specific_config(&[0x2B, 0x11, 0x88, 0x00]).unwrap();
		assert!(c.sbr);
		assert_eq!((c.sample_rate, c.channels), (24000, 2));
	}

	#[test]
	fn parses_adts_header() {
		let h = AdtsHeader::parse(&[0xFF, 0xF1, 0x50, 0x80, 0x02, 0x1F, 0xFC]).unwrap();
		assert_eq!(h.object_type, 2);
		assert_eq!(h.sample_rate_index, 4);
		assert_eq!(h.channel_config, 2);
		assert_eq!(h.frame_length, 16);
		assert_eq!(h.header_length, 7);
	}

	#[test]
	fn maps_51_to_native_order() {
		let c = AacConfig::from_channel_config(2, 3, 6).unwrap();
		assert_eq!(c.channels, 6);
		assert_eq!(c.slot(ElementKind::Lfe, 0).unwrap().out[0], 3);
		assert_eq!(c.slot(ElementKind::Cpe, 1).unwrap().out, [4, 5]);
	}
}
