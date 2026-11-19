//! Bitstream conversions for stream copy between containers that frame a
//! codec differently.
//!
//! MPEG-TS (and raw `.h264`/`.aac` streams) carry H.264/HEVC as an Annex B
//! byte stream with in-band parameter sets, and AAC as ADTS frames. MP4 and
//! Matroska want length-prefixed NAL units with an `avcC`/`hvcC` record,
//! and raw AAC frames with an AudioSpecificConfig. FFmpeg's `mov` and
//! `matroska` muxers convert automatically (`ff_isom_write_avcc`,
//! `ff_isom_write_hvcc`, `ff_avc_parse_nal_units`, the `aac_adtstoasc`
//! bitstream filter they insert); [`Converter`] does the same for RMP's
//! muxers.

use crate::codecs::aac::AdtsHeader;
use crate::codecs::bitstream::BitReader;
use crate::codecs::h264::nal::{is_annex_b, split_annex_b, sps_summary, unescape};
use crate::core::{Packet, Parameters};
use crate::message::{Message, Result};

/// What a stream's packets need before a length-prefixed container can
/// store them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Converter {
	#[default]
	None,
	/// Annex B → 4-byte NAL lengths.
	AnnexB,
	/// ADTS → raw AAC; the AudioSpecificConfig comes from the first header
	/// when the stream had none.
	Adts,
}

impl Converter {
	/// Decides the conversion for a stream and converts its extradata in
	/// place (Annex B parameter sets → `avcC`/`hvcC`).
	pub fn for_iso(parameters: &mut Parameters) -> Result<Self> {
		match parameters.codec.id.as_str() {
			"h264" | "hevc" if is_annex_b(&parameters.extradata) => {
				parameters.extradata = if parameters.codec.id == "h264" {
					avcc_from_annex_b(&parameters.extradata)?
				} else {
					hvcc_from_annex_b(&parameters.extradata)?
				};
				Ok(Self::AnnexB)
			}
			"aac" => Ok(Self::Adts),
			_ => Ok(Self::None),
		}
	}

	/// Converts one packet. ADTS is recognised per packet, as FFmpeg's
	/// `mov` muxer does: a whole ADTS frame (sync word and a frame length
	/// equal to the packet's) is converted. Returns `true` when it filled in missing
	/// extradata (the first ADTS frame of a stream without a config).
	pub fn apply(self, packet: &mut Packet, parameters: &mut Parameters) -> Result<bool> {
		match self {
			Self::None => Ok(false),
			Self::AnnexB => {
				// Only streams whose extradata was Annex B get here; their
				// packets are too (an `avcC` stream's never are converted, as
				// a 256-511 byte NAL length also begins 00 00 01).
				if is_annex_b(&packet.data) {
					packet.data = annex_b_to_length_prefixed(&packet.data);
				}
				Ok(false)
			}
			Self::Adts => {
				let Some(h) = AdtsHeader::parse(&packet.data) else { return Ok(false) };
				if h.frame_length != packet.data.len() || h.header_length > packet.data.len() {
					return Ok(false);
				}
				if h.raw_blocks > 1 {
					return Err(Message::Container(
						"adts frames with several raw data blocks can't be stream-copied",
					));
				}
				let filled = parameters.extradata.is_empty();
				if filled {
					if h.channel_config == 0 {
						return Err(Message::Container(
							"adts with an in-band channel configuration can't be stream-copied",
						));
					}
					parameters.extradata = audio_specific_config(&h);
				}
				packet.data.drain(..h.header_length);
				Ok(filled)
			}
		}
	}
}

/// Two-byte AudioSpecificConfig of an ADTS header (ISO/IEC 14496-3
/// §1.6.2.1).
pub fn audio_specific_config(h: &AdtsHeader) -> Vec<u8> {
	let v = (u16::from(h.object_type) << 11)
		| ((h.sample_rate_index as u16 & 0x0F) << 7)
		| (u16::from(h.channel_config & 0x0F) << 3);
	v.to_be_bytes().to_vec()
}

/// Replaces start codes with 4-byte big-endian NAL lengths.
pub fn annex_b_to_length_prefixed(data: &[u8]) -> Vec<u8> {
	let nals = split_annex_b(data);
	let mut out = Vec::with_capacity(data.len() + nals.len());
	for nal in nals {
		out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
		out.extend_from_slice(nal);
	}
	out
}

fn put_nal_list(out: &mut Vec<u8>, nals: &[&[u8]]) -> Result<()> {
	for nal in nals {
		let len =
			u16::try_from(nal.len()).map_err(|_| Message::Container("parameter set too large"))?;
		out.extend_from_slice(&len.to_be_bytes());
		out.extend_from_slice(nal);
	}
	Ok(())
}

/// `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §5.3.3.1) from Annex B
/// SPS/PPS, as FFmpeg's `ff_isom_write_avcc` writes it, including the
/// chroma/bit-depth extension of the high profiles.
pub fn avcc_from_annex_b(data: &[u8]) -> Result<Vec<u8>> {
	let nals = split_annex_b(data);
	let sps: Vec<&[u8]> = nals.iter().copied().filter(|n| n[0] & 0x1F == 7 && n.len() >= 4).collect();
	let pps: Vec<&[u8]> = nals.iter().copied().filter(|n| n[0] & 0x1F == 8).collect();
	let first = sps.first().ok_or(Message::Container("h264 extradata has no sps"))?;
	if pps.is_empty() || sps.len() > 31 || pps.len() > 255 {
		return Err(Message::Container("h264 extradata needs 1-31 sps and 1-255 pps"));
	}
	let mut out = vec![1, first[1], first[2], first[3], 0xFF, 0xE0 | sps.len() as u8];
	put_nal_list(&mut out, &sps)?;
	out.push(pps.len() as u8);
	put_nal_list(&mut out, &pps)?;
	if matches!(first[1], 100 | 110 | 122 | 144) {
		let s = sps_summary(&unescape(&first[1..]))
			.ok_or(Message::Container("h264 sps could not be parsed"))?;
		out.push(0xFC | s.chroma_format_idc as u8);
		out.push(0xF8 | (s.bit_depth_luma.saturating_sub(8) as u8 & 7));
		out.push(0xF8 | (s.bit_depth_chroma.saturating_sub(8) as u8 & 7));
		out.push(0); // numOfSequenceParameterSetExt
	}
	Ok(out)
}

/// The fields of an HEVC SPS that `hvcC` and stream parameters need
/// (H.265 §7.3.2.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcSpsSummary {
	/// `general_profile_space` … `general_level_idc`: hvcC bytes 1-12.
	pub general: [u8; 12],
	pub max_sub_layers: u8,
	pub temporal_id_nesting: bool,
	pub chroma_format_idc: u32,
	pub bit_depth_luma: u32,
	pub bit_depth_chroma: u32,
	pub width: u32,
	pub height: u32,
}

/// Parses an unescaped SPS NAL unit, header included.
pub fn hevc_sps_summary(nal: &[u8]) -> Option<HevcSpsSummary> {
	fn ue(bs: &mut BitReader) -> Option<u32> {
		let mut zeros = 0;
		while !bs.bit() {
			zeros += 1;
			if zeros > 31 || bs.overrun() {
				return None;
			}
		}
		Some(((1u64 << zeros) - 1 + u64::from(bs.bits(zeros))) as u32)
	}
	let mut bs = BitReader::new(nal);
	bs.skip(16 + 4); // NAL header, sps_video_parameter_set_id
	let max_sub_layers_minus1 = bs.bits(3) as usize;
	let temporal_id_nesting = bs.bit();
	let mut general = [0u8; 12];
	for b in &mut general {
		*b = bs.bits(8) as u8;
	}
	// sub_layer_{profile,level}_present_flag, alignment, sub-layer info.
	let mut present = [(false, false); 8];
	for p in present.iter_mut().take(max_sub_layers_minus1) {
		*p = (bs.bit(), bs.bit());
	}
	if max_sub_layers_minus1 > 0 {
		bs.skip(2 * (8 - max_sub_layers_minus1));
	}
	for &(profile, level) in present.iter().take(max_sub_layers_minus1) {
		bs.skip(if profile { 88 } else { 0 } + if level { 8 } else { 0 });
	}
	ue(&mut bs)?; // sps_seq_parameter_set_id
	let chroma_format_idc = ue(&mut bs)?;
	if chroma_format_idc == 3 {
		bs.skip(1);
	}
	let mut width = ue(&mut bs)?;
	let mut height = ue(&mut bs)?;
	if bs.bit() {
		let (unit_x, unit_y) = match chroma_format_idc {
			1 => (2, 2),
			2 => (2, 1),
			_ => (1, 1),
		};
		let (l, r, t, b) = (ue(&mut bs)?, ue(&mut bs)?, ue(&mut bs)?, ue(&mut bs)?);
		width = width.checked_sub(l.checked_add(r)?.checked_mul(unit_x)?)?;
		height = height.checked_sub(t.checked_add(b)?.checked_mul(unit_y)?)?;
	}
	let bit_depth_luma = ue(&mut bs)? + 8;
	let bit_depth_chroma = ue(&mut bs)? + 8;
	if bs.overrun() || width == 0 || height == 0 || chroma_format_idc > 3 {
		return None;
	}
	Some(HevcSpsSummary {
		general,
		max_sub_layers: max_sub_layers_minus1 as u8 + 1,
		temporal_id_nesting,
		chroma_format_idc,
		bit_depth_luma,
		bit_depth_chroma,
		width,
		height,
	})
}

/// `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §8.3.3.1) from Annex
/// B VPS/SPS/PPS (and prefix SEI), as FFmpeg's `ff_isom_write_hvcc` lays it
/// out. The profile fields are the SPS's; segmentation, parallelism and
/// frame rate are written as unknown, which the record allows.
pub fn hvcc_from_annex_b(data: &[u8]) -> Result<Vec<u8>> {
	let nals = split_annex_b(data);
	let of_type = |t: u8| -> Vec<&[u8]> {
		nals.iter().copied().filter(|n| n.len() >= 2 && (n[0] >> 1) & 0x3F == t).collect()
	};
	let (vps, sps, pps, sei) = (of_type(32), of_type(33), of_type(34), of_type(39));
	let first = sps.first().ok_or(Message::Container("hevc extradata has no sps"))?;
	if vps.is_empty() || pps.is_empty() {
		return Err(Message::Container("hevc extradata needs a vps, sps and pps"));
	}
	let s =
		hevc_sps_summary(&unescape(first)).ok_or(Message::Container("hevc sps could not be parsed"))?;
	let mut out = vec![1];
	out.extend_from_slice(&s.general);
	out.extend_from_slice(&[0xF0, 0x00]); // min_spatial_segmentation_idc = 0
	out.push(0xFC); // parallelismType = 0 (unknown)
	out.push(0xFC | s.chroma_format_idc as u8);
	out.push(0xF8 | (s.bit_depth_luma.saturating_sub(8) as u8 & 7));
	out.push(0xF8 | (s.bit_depth_chroma.saturating_sub(8) as u8 & 7));
	out.extend_from_slice(&[0, 0]); // avgFrameRate
	// constantFrameRate 0, numTemporalLayers, temporalIdNested, 4-byte lengths.
	out.push(((s.max_sub_layers & 7) << 3) | (u8::from(s.temporal_id_nesting) << 2) | 3);
	let arrays: Vec<(u8, &Vec<&[u8]>)> = [(32, &vps), (33, &sps), (34, &pps), (39, &sei)]
		.into_iter()
		.filter(|(_, l)| !l.is_empty())
		.collect();
	out.push(arrays.len() as u8);
	for (kind, list) in arrays {
		// array_completeness: parameter sets are all here (as `hvc1` needs).
		let complete = if kind == 39 { 0 } else { 0x80 };
		out.push(complete | kind);
		let count =
			u16::try_from(list.len()).map_err(|_| Message::Container("too many hevc nal units"))?;
		out.extend_from_slice(&count.to_be_bytes());
		put_nal_list(&mut out, list)?;
	}
	Ok(out)
}

/// A length-prefixed stream's NAL length size and its parameter sets as
/// Annex B, from an `avcC` or `hvcC` record: what a byte-stream muxer
/// (MPEG-TS) needs to undo the MP4 layout, as FFmpeg's `h264_mp4toannexb`
/// and `hevc_mp4toannexb` filters do. `None` when the extradata isn't such
/// a record (Annex B or empty extradata: the packets are Annex B already).
pub fn annex_b_from_config(codec: &str, extradata: &[u8]) -> Option<(usize, Vec<u8>)> {
	let mut out = Vec::new();
	let mut put = |nal: &[u8]| {
		out.extend_from_slice(&[0, 0, 0, 1]);
		out.extend_from_slice(nal);
	};
	match codec {
		"h264" => {
			let cfg = crate::codecs::h264::nal::AvcConfig::parse(extradata)?;
			cfg.sps.iter().chain(&cfg.pps).for_each(|n| put(n));
			Some((cfg.nal_length_size, out))
		}
		"hevc" => {
			if extradata.len() < 23 || extradata[0] != 1 {
				return None;
			}
			let length_size = usize::from(extradata[21] & 3) + 1;
			let mut at = 23;
			for _ in 0..extradata[22] {
				let count = u16::from_be_bytes([*extradata.get(at + 1)?, *extradata.get(at + 2)?]);
				at += 3;
				for _ in 0..count {
					let len = usize::from(u16::from_be_bytes([*extradata.get(at)?, *extradata.get(at + 1)?]));
					put(extradata.get(at + 2..at + 2 + len)?);
					at += 2 + len;
				}
			}
			Some((length_size, out))
		}
		_ => None,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::h264::nal::AvcConfig;
	use crate::core::{CodecId, Stream, StreamId};

	// x264's SPS/PPS for 320x240 High profile.
	const SPS: [u8; 25] = [
		0x67, 0x64, 0x00, 0x0D, 0xAC, 0xD9, 0x41, 0x41, 0xFB, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00, 0x10,
		0x00, 0x00, 0x03, 0x03, 0x20, 0xF1, 0x42, 0x99, 0x60,
	];
	const PPS: [u8; 4] = [0x68, 0xEF, 0x8F, 0xCB];

	fn annex_b(nals: &[&[u8]]) -> Vec<u8> {
		nals.iter().flat_map(|n| [&[0u8, 0, 0, 1][..], n].concat()).collect()
	}

	#[test]
	fn sps_summary_320x240() {
		let s = sps_summary(&unescape(&SPS[1..])).unwrap();
		assert_eq!((s.width, s.height, s.profile_idc, s.chroma_format_idc), (320, 240, 100, 1));
	}

	#[test]
	fn avcc_round_trip() {
		let avcc = avcc_from_annex_b(&annex_b(&[&SPS, &PPS])).unwrap();
		let cfg = AvcConfig::parse(&avcc).unwrap();
		assert_eq!(cfg.nal_length_size, 4);
		assert_eq!(annex_b_from_config("h264", &avcc), Some((4, annex_b(&[&SPS, &PPS]))));
		assert_eq!(annex_b_from_config("h264", &annex_b(&[&SPS, &PPS])), None);
		assert_eq!(cfg.sps, vec![SPS.to_vec()]);
		assert_eq!(cfg.pps, vec![PPS.to_vec()]);
		assert_eq!(&avcc[avcc.len() - 4..], &[0xFD, 0xF8, 0xF8, 0]); // high-profile extension
	}

	#[test]
	fn packets_and_adts() {
		let mut h264 = Stream::video(StreamId(0), 320, 240, "yuv420p", CodecId::new("h264"));
		h264.parameters.extradata = annex_b(&[&SPS, &PPS]);
		let conv = Converter::for_iso(&mut h264.parameters).unwrap();
		assert_eq!(conv, Converter::AnnexB);
		let mut p = Packet::new(StreamId(0), annex_b(&[&[0x09, 0xF0], &[0x65, 1, 2, 3]]));
		conv.apply(&mut p, &mut h264.parameters).unwrap();
		assert_eq!(p.data, [0, 0, 0, 2, 0x09, 0xF0, 0, 0, 0, 4, 0x65, 1, 2, 3]);

		// avcC streams are left alone, even when a NAL length looks like a
		// start code (a 0x1F0-byte NAL).
		let mut mp4 = Stream::video(StreamId(0), 320, 240, "yuv420p", CodecId::new("h264"));
		mp4.parameters.extradata = h264.parameters.extradata.clone();
		let conv = Converter::for_iso(&mut mp4.parameters).unwrap();
		assert_eq!(conv, Converter::None);
		let mut data = vec![0, 0, 1, 0xF0];
		data.extend(std::iter::repeat_n(0x41, 0x1F0));
		let mut p = Packet::new(StreamId(0), data.clone());
		conv.apply(&mut p, &mut mp4.parameters).unwrap();
		assert_eq!(p.data, data);

		let mut aac = Stream::audio(StreamId(1), 48_000, 2, CodecId::new("aac"));
		let conv = Converter::for_iso(&mut aac.parameters).unwrap();
		// ADTS: LC, 48 kHz, stereo, 10-byte frame.
		let mut p = Packet::new(StreamId(1), vec![0xFF, 0xF1, 0x4C, 0x80, 0x01, 0x5F, 0xFC, 7, 8, 9]);
		assert!(conv.apply(&mut p, &mut aac.parameters).unwrap());
		assert_eq!(aac.parameters.extradata, [0x11, 0x90]);
		assert_eq!(p.data, [7, 8, 9]);
	}
}
