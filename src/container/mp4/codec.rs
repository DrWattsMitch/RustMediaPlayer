//! Sample descriptions (`stsd` entries): mapping between MP4 sample entries
//! and RMP codec ids + parameters, in both directions.
//!
//! Codec ids follow FFmpeg's names (`h264`, `hevc`, `aac`, `mp3`, …) and the
//! codec configuration record (`avcC`, `hvcC`, the AAC AudioSpecificConfig in
//! `esds`, …) is carried as [`Parameters::extradata`], exactly like
//! `AVCodecParameters.extradata`.

use super::boxes::{BoxIter, FourCC, Put, find, full_box, write_box, write_full_box};
use crate::codecs::flac::StreamInfo;
use crate::codecs::opus::OpusHead;
use crate::core::StreamKind;
use crate::io::ByteCursor;
use crate::message::{Message, Result};

/// Decoded sample description of a track.
#[derive(Debug, Clone, Default)]
pub struct CodecInfo {
	pub codec: String,
	pub width: Option<u32>,
	pub height: Option<u32>,
	pub sample_rate: Option<u32>,
	pub channels: Option<u8>,
	pub extradata: Vec<u8>,
	pub bitrate: Option<u32>,
}

/// MPEG-4 Systems object type indications (ISO/IEC 14496-1, Table 5).
mod oti {
	pub const MPEG4_VISUAL: u8 = 0x20;
	pub const AAC: u8 = 0x40;
	pub const MPEG2_AAC_MAIN: u8 = 0x66;
	pub const MPEG2_AAC_LC: u8 = 0x67;
	pub const MPEG2_AAC_SSR: u8 = 0x68;
	pub const MPEG2_AUDIO: u8 = 0x69;
	pub const MPEG1_AUDIO: u8 = 0x6B;
}

/// Bytes of fixed fields in a VisualSampleEntry after the box header.
const VISUAL_ENTRY_FIELDS: usize = 78;
/// Bytes of fixed fields in a (version 0) AudioSampleEntry.
const AUDIO_ENTRY_FIELDS: usize = 28;

/// Parses the first entry of an `stsd` box.
pub fn parse_stsd(stsd: &[u8], kind: StreamKind) -> Result<CodecInfo> {
	let (_, _, payload) = full_box(stsd)?;
	let mut c = ByteCursor::new(payload);
	let count = c.read_u32()?;
	if count == 0 {
		return Err(Message::Container("mp4 stsd has no sample entry"));
	}
	let (format, entry) =
		BoxIter::new(c.rest()).next().ok_or(Message::Container("mp4 stsd has no sample entry"))??;

	match kind {
		StreamKind::Video => parse_visual_entry(&format, entry),
		StreamKind::Audio => parse_audio_entry(&format, entry),
		_ => Ok(CodecInfo { codec: codec_from_fourcc(&format), ..Default::default() }),
	}
}

fn parse_visual_entry(format: &FourCC, entry: &[u8]) -> Result<CodecInfo> {
	let mut c = ByteCursor::new(entry);
	c.skip(8 + 16)?; // reserved, data_reference_index, pre_defined/reserved
	let width = u32::from(c.read_u16()?);
	let height = u32::from(c.read_u16()?);
	let children = entry.get(VISUAL_ENTRY_FIELDS..).unwrap_or(&[]);

	let mut info = CodecInfo {
		codec: codec_from_fourcc(format),
		width: Some(width),
		height: Some(height),
		..Default::default()
	};
	let config: Option<&FourCC> = match format {
		b"avc1" | b"avc3" => Some(b"avcC"),
		b"hvc1" | b"hev1" => Some(b"hvcC"),
		b"av01" => Some(b"av1C"),
		b"vp09" => Some(b"vpcC"),
		_ => None,
	};
	if let Some(config) = config {
		info.extradata = find(children, config).unwrap_or(&[]).to_vec();
	}
	if let Some(esds) = find(children, b"esds") {
		if let Some(es) = parse_esds(esds) {
			info.codec = codec_from_oti(es.object_type, StreamKind::Video).to_string();
			info.extradata = es.decoder_specific_info;
			info.bitrate = Some(es.avg_bitrate).filter(|&b| b > 0);
		}
	}
	if let Some(btrt) = find(children, b"btrt") {
		info.bitrate = info.bitrate.or(parse_btrt(btrt));
	}
	Ok(info)
}

fn parse_audio_entry(format: &FourCC, entry: &[u8]) -> Result<CodecInfo> {
	let mut c = ByteCursor::new(entry);
	c.skip(8)?; // reserved, data_reference_index
	let version = c.read_u16()?; // QuickTime sound description version
	c.skip(2 + 4)?; // revision, vendor
	let mut channels = u32::from(c.read_u16()?);
	let sample_size = c.read_u16()?;
	c.skip(4)?; // compression id, packet size
	let mut sample_rate = c.read_u32()? >> 16;
	let children_at = match version {
		1 => AUDIO_ENTRY_FIELDS + 16,
		2 => {
			c.skip(4)?; // sizeOfStructOnly
			let rate = f64::from_bits(c.read_u64()?);
			if rate.is_finite() && rate > 0.0 && rate < f64::from(u32::MAX) {
				sample_rate = rate as u32;
			}
			channels = c.read_u32()?;
			AUDIO_ENTRY_FIELDS + 36
		}
		_ => AUDIO_ENTRY_FIELDS,
	};
	let mut children = entry.get(children_at..).unwrap_or(&[]);
	// QuickTime wraps codec boxes in a `wave` box.
	if let Some(wave) = find(children, b"wave") {
		children = wave;
	}

	let mut info = CodecInfo {
		codec: codec_from_fourcc(format),
		sample_rate: Some(sample_rate).filter(|&r| r > 0),
		channels: u8::try_from(channels).ok().filter(|&c| c > 0),
		..Default::default()
	};
	if (format == b"twos" || format == b"sowt") && sample_size == 8 {
		info.codec = "pcm_s8".into();
	}

	if let Some(es) = find(children, b"esds").and_then(parse_esds) {
		info.codec = codec_from_oti(es.object_type, StreamKind::Audio).to_string();
		info.bitrate = Some(es.avg_bitrate).filter(|&b| b > 0);
		if info.codec == "aac" {
			if let Some((rate, ch)) = parse_audio_specific_config(&es.decoder_specific_info) {
				info.sample_rate = Some(rate);
				if ch > 0 {
					info.channels = Some(ch);
				}
			}
		}
		info.extradata = es.decoder_specific_info;
	} else {
		let config: Option<&FourCC> = match format {
			b"Opus" => Some(b"dOps"),
			b"fLaC" => Some(b"dfLa"),
			b"alac" => Some(b"alac"),
			_ => None,
		};
		if let Some(config) = config {
			info.extradata = find(children, config).unwrap_or(&[]).to_vec();
		}
		// FFmpeg's extradata layouts: the bare STREAMINFO for FLAC, the
		// `OpusHead` packet for Opus (mov.c `mov_read_dfla` / `mov_read_dops`).
		match format {
			b"fLaC" => {
				if let Some(si) = StreamInfo::from_extradata(&info.extradata) {
					info.extradata = si.to_bytes().to_vec();
					info.sample_rate = Some(si.sample_rate);
					info.channels = Some(si.channels);
				}
			}
			b"Opus" => {
				if let Some(head) = OpusHead::from_dops(&info.extradata) {
					info.channels = Some(head.channels);
					info.sample_rate = Some(48_000);
					info.extradata = head.to_bytes();
				}
			}
			_ => {}
		}
	}
	if let Some(btrt) = find(children, b"btrt") {
		info.bitrate = info.bitrate.or(parse_btrt(btrt));
	}
	Ok(info)
}

fn parse_btrt(body: &[u8]) -> Option<u32> {
	let mut c = ByteCursor::new(body);
	c.skip(8).ok()?; // bufferSizeDB, maxBitrate
	c.read_u32().ok().filter(|&b| b > 0)
}

fn codec_from_fourcc(format: &FourCC) -> String {
	let name = match format {
		b"avc1" | b"avc3" => "h264",
		b"hvc1" | b"hev1" => "hevc",
		b"av01" => "av1",
		b"vp09" => "vp9",
		b"mp4v" => "mpeg4",
		b"jpeg" | b"mjpa" => "mjpeg",
		b"mp4a" => "aac",
		b".mp3" | b"ms\0U" => "mp3",
		b"Opus" => "opus",
		b"fLaC" => "flac",
		b"alac" => "alac",
		b"ac-3" => "ac3",
		b"ec-3" => "eac3",
		b"sowt" => "pcm_s16le",
		b"twos" => "pcm_s16be",
		b"tx3g" => "mov_text",
		b"wvtt" => "webvtt",
		_ => return super::boxes::fourcc_string(format).to_lowercase(),
	};
	name.to_string()
}

fn codec_from_oti(object_type: u8, kind: StreamKind) -> &'static str {
	match object_type {
		oti::MPEG4_VISUAL => "mpeg4",
		0x60..=0x65 => "mpeg2video",
		0x6A => "mpeg1video",
		0x6C => "mjpeg",
		oti::AAC | oti::MPEG2_AAC_MAIN | oti::MPEG2_AAC_LC | oti::MPEG2_AAC_SSR => "aac",
		oti::MPEG2_AUDIO | oti::MPEG1_AUDIO => "mp3",
		0xA5 => "ac3",
		0xA6 => "eac3",
		0xA9..=0xAC => "dts",
		0xAD => "opus",
		_ if kind == StreamKind::Video => "mpeg4",
		_ => "aac",
	}
}

/// Fields of an `esds` ES_Descriptor that matter for decoding.
#[derive(Debug, Clone, Default)]
pub struct EsDescriptor {
	pub object_type: u8,
	pub avg_bitrate: u32,
	pub decoder_specific_info: Vec<u8>,
}

/// Reads an MPEG-4 descriptor header: tag and "expandable" length.
fn read_descriptor(c: &mut ByteCursor) -> Option<(u8, usize)> {
	let tag = c.read_u8().ok()?;
	let mut len = 0usize;
	for _ in 0..4 {
		let b = c.read_u8().ok()?;
		len = (len << 7) | usize::from(b & 0x7F);
		if b & 0x80 == 0 {
			break;
		}
	}
	Some((tag, len))
}

/// Parses an `esds` box body (ISO/IEC 14496-1 §7.2.6.5).
pub fn parse_esds(body: &[u8]) -> Option<EsDescriptor> {
	let (_, _, payload) = full_box(body).ok()?;
	let mut c = ByteCursor::new(payload);
	let (tag, _) = read_descriptor(&mut c)?;
	if tag != 0x03 {
		return None;
	}
	c.skip(2).ok()?; // ES_ID
	let flags = c.read_u8().ok()?;
	if flags & 0x80 != 0 {
		c.skip(2).ok()?; // dependsOn_ES_ID
	}
	if flags & 0x40 != 0 {
		let url_len = usize::from(c.read_u8().ok()?);
		c.skip(url_len).ok()?;
	}
	if flags & 0x20 != 0 {
		c.skip(2).ok()?; // OCR_ES_Id
	}

	let (tag, _) = read_descriptor(&mut c)?;
	if tag != 0x04 {
		return None;
	}
	let mut es = EsDescriptor { object_type: c.read_u8().ok()?, ..Default::default() };
	c.skip(1 + 3 + 4).ok()?; // stream type, buffer size, max bitrate
	es.avg_bitrate = c.read_u32().ok()?;
	if let Some((0x05, len)) = read_descriptor(&mut c) {
		es.decoder_specific_info = c.read_bytes(len.min(c.remaining())).ok()?.to_vec();
	}
	Some(es)
}

const AAC_SAMPLE_RATES: [u32; 13] =
	[96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

/// Extracts (sample rate, channel configuration) from an AAC
/// AudioSpecificConfig (ISO/IEC 14496-3 §1.6.2.1).
pub fn parse_audio_specific_config(asc: &[u8]) -> Option<(u32, u8)> {
	let mut bits = 0u64;
	for (i, &b) in asc.iter().take(8).enumerate() {
		bits |= u64::from(b) << (56 - 8 * i);
	}
	let available = asc.len().min(8) * 8;
	let mut pos = 0usize;
	let mut take = |n: usize| -> Option<u32> {
		if pos + n > available {
			return None;
		}
		let v = ((bits << pos) >> (64 - n)) as u32;
		pos += n;
		Some(v)
	};
	let mut object_type = take(5)?;
	if object_type == 31 {
		object_type = 32 + take(6)?;
	}
	let _ = object_type;
	let index = take(4)? as usize;
	let rate = if index == 15 { take(24)? } else { *AAC_SAMPLE_RATES.get(index)? };
	let channels = take(4)? as u8;
	Some((rate, channels))
}

/// Parameters needed to write a sample entry.
pub struct EntryParams<'a> {
	pub codec: &'a str,
	pub kind: StreamKind,
	pub width: u32,
	pub height: u32,
	pub sample_rate: u32,
	pub channels: u8,
	pub extradata: &'a [u8],
	pub track_id: u32,
	pub avg_bitrate: u32,
	pub max_bitrate: u32,
	pub buffer_size: u32,
}

/// Codecs the MP4 muxer can describe.
pub fn muxer_supports(codec: &str) -> bool {
	matches!(codec, "h264" | "hevc" | "av1" | "mpeg4" | "aac" | "mp3" | "alac" | "opus" | "flac")
}

/// Writes a complete `stsd` box with one sample entry.
pub fn write_stsd(out: &mut Vec<u8>, p: &EntryParams) -> Result<()> {
	write_full_box(out, b"stsd", 0, 0, |out| {
		out.put_u32(1);
		match p.kind {
			StreamKind::Video => write_visual_entry(out, p),
			StreamKind::Audio => write_audio_entry(out, p),
			_ => Err(Message::Container("mp4 muxer supports only audio and video streams")),
		}
	})
}

fn require_extradata<'a>(p: &EntryParams<'a>) -> Result<&'a [u8]> {
	if p.extradata.is_empty() {
		return Err(Message::Container("mp4 muxer: codec requires extradata (decoder configuration)"));
	}
	Ok(p.extradata)
}

fn write_visual_entry(out: &mut Vec<u8>, p: &EntryParams) -> Result<()> {
	let (format, config): (&FourCC, Option<&FourCC>) = match p.codec {
		"h264" => (b"avc1", Some(b"avcC")),
		"hevc" => (b"hvc1", Some(b"hvcC")),
		"av1" => (b"av01", Some(b"av1C")),
		"mpeg4" => (b"mp4v", None),
		_ => return Err(Message::Container("mp4 muxer: unsupported video codec")),
	};
	let width =
		u16::try_from(p.width).map_err(|_| Message::Container("mp4 video width too large"))?;
	let height =
		u16::try_from(p.height).map_err(|_| Message::Container("mp4 video height too large"))?;
	write_box(out, format, |out| {
		out.put_zeros(6);
		out.put_u16(1); // data_reference_index
		out.put_zeros(16); // pre_defined / reserved
		out.put_u16(width);
		out.put_u16(height);
		out.put_u32(0x0048_0000); // 72 dpi
		out.put_u32(0x0048_0000);
		out.put_u32(0);
		out.put_u16(1); // frame_count
		out.put_zeros(32); // compressorname
		out.put_u16(0x0018); // depth
		out.put_u16(0xFFFF); // pre_defined = -1
		match config {
			Some(config) => {
				let extradata = require_extradata(p)?;
				write_box(out, config, |out| {
					out.put(extradata);
					Ok(())
				})
			}
			None => write_esds(out, p, oti::MPEG4_VISUAL, 0x04),
		}
	})
}

fn write_audio_entry(out: &mut Vec<u8>, p: &EntryParams) -> Result<()> {
	let format: &FourCC = match p.codec {
		"aac" | "mp3" => b"mp4a",
		"alac" => b"alac",
		"opus" => b"Opus",
		"flac" => b"fLaC",
		_ => return Err(Message::Container("mp4 muxer: unsupported audio codec")),
	};
	write_box(out, format, |out| {
		out.put_zeros(6);
		out.put_u16(1); // data_reference_index
		out.put_zeros(8); // version 0 sound description
		out.put_u16(u16::from(p.channels));
		out.put_u16(16); // sample size
		out.put_u32(0); // pre_defined, reserved
		// 16.16 fixed point; rates above 65535 Hz cannot be represented and
		// are signalled by the codec configuration instead.
		out.put_u32(if p.sample_rate <= 0xFFFF { p.sample_rate << 16 } else { 0 });
		match p.codec {
			"aac" => {
				require_extradata(p)?;
				write_esds(out, p, oti::AAC, 0x05)
			}
			"mp3" => {
				let object_type = if p.sample_rate >= 32_000 { oti::MPEG1_AUDIO } else { oti::MPEG2_AUDIO };
				write_esds(out, p, object_type, 0x05)
			}
			"alac" | "opus" | "flac" => {
				let config: &FourCC = match p.codec {
					"alac" => b"alac",
					"opus" => b"dOps",
					_ => b"dfLa",
				};
				let extradata = require_extradata(p)?;
				let body = match p.codec {
					"flac" => {
						let si = StreamInfo::from_extradata(extradata)
							.ok_or(Message::Container("mp4 muxer: invalid flac extradata"))?;
						let mut body = vec![0, 0, 0, 0]; // dfLa full box: version 0, flags 0
						body.extend(si.to_block(true));
						body
					}
					"opus" => OpusHead::parse(extradata)
						.or_else(|| OpusHead::from_dops(extradata))
						.ok_or(Message::Container("mp4 muxer: invalid opus extradata"))?
						.to_dops(),
					_ => extradata.to_vec(),
				};
				write_box(out, config, |out| {
					out.put(&body);
					Ok(())
				})
			}
			_ => Err(Message::Container("mp4 muxer: unsupported audio codec")),
		}
	})
}

/// Writes a descriptor header with a fixed 4-byte length (as FFmpeg does).
fn put_descriptor(out: &mut Vec<u8>, tag: u8, len: usize) -> Result<()> {
	if len >= 1 << 28 {
		return Err(Message::Container("mp4 descriptor too large"));
	}
	out.put_u8(tag);
	for shift in [21, 14, 7] {
		out.put_u8(((len >> shift) & 0x7F) as u8 | 0x80);
	}
	out.put_u8((len & 0x7F) as u8);
	Ok(())
}

fn write_esds(out: &mut Vec<u8>, p: &EntryParams, object_type: u8, stream_type: u8) -> Result<()> {
	write_full_box(out, b"esds", 0, 0, |out| {
		let dsi_len = if p.extradata.is_empty() { 0 } else { 5 + p.extradata.len() };
		let decoder_config_len = 13 + dsi_len;
		let es_len = 3 + (5 + decoder_config_len) + (5 + 1);
		put_descriptor(out, 0x03, es_len)?;
		out.put_u16(u16::try_from(p.track_id).unwrap_or(u16::MAX));
		out.put_u8(0); // flags
		put_descriptor(out, 0x04, decoder_config_len)?;
		out.put_u8(object_type);
		out.put_u8((stream_type << 2) | 1);
		out.put_u24(p.buffer_size.min(0xFF_FFFF));
		out.put_u32(p.max_bitrate);
		out.put_u32(p.avg_bitrate);
		if !p.extradata.is_empty() {
			put_descriptor(out, 0x05, p.extradata.len())?;
			out.put(p.extradata);
		}
		put_descriptor(out, 0x06, 1)?; // SLConfigDescriptor
		out.put_u8(0x02); // predefined: MP4 file
		Ok(())
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn asc_lc_stereo_44k() {
		assert_eq!(parse_audio_specific_config(&[0x12, 0x10]), Some((44_100, 2)));
	}

	#[test]
	fn esds_round_trip() {
		let params = EntryParams {
			codec: "aac",
			kind: StreamKind::Audio,
			width: 0,
			height: 0,
			sample_rate: 44_100,
			channels: 2,
			extradata: &[0x12, 0x10],
			track_id: 1,
			avg_bitrate: 128_000,
			max_bitrate: 130_000,
			buffer_size: 1024,
		};
		let mut out = Vec::new();
		write_stsd(&mut out, &params).unwrap();
		let stsd = super::super::boxes::find(&out, b"stsd").unwrap();
		let info = parse_stsd(stsd, StreamKind::Audio).unwrap();
		assert_eq!(info.codec, "aac");
		assert_eq!(info.sample_rate, Some(44_100));
		assert_eq!(info.channels, Some(2));
		assert_eq!(info.extradata, vec![0x12, 0x10]);
		assert_eq!(info.bitrate, Some(128_000));
	}
}
