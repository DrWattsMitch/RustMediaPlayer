//! Elementary streams: which codec a PMT entry carries, how its PES
//! payloads become packets, and the codec parameters found in them.
//!
//! FFmpeg demuxes MPEG-TS with its parsers enabled (`AVSTREAM_PARSE_FULL`):
//! audio PES payloads are split into one packet per frame with
//! interpolated timestamps, and the video parsers mark keyframes. This
//! module does the same, so packets match `ffprobe -show_packets`.
//!
//! FFmpeg: `mpegts.c` (`ISO_types`, `HDMV_types`, `REGD_types`,
//! `DESC_types`, `ff_parse_mpeg2_descriptor`), `aac_parser.c`,
//! `mpegaudio_parser.c`, `ac3_parser.c`, `h264_parser.c`, `hevc_parser.c`.

use super::psi::{self, EsInfo};
use crate::codecs::aac::AdtsHeader;
use crate::codecs::aac::config::SAMPLE_RATES;
use crate::codecs::ac3::{self, Header as Ac3Header};
use crate::codecs::bitstream::BitReader;
use crate::codecs::h264::nal::{split_annex_b, sps_summary, unescape};
use crate::codecs::mp3::FrameHeader;
use crate::core::{CodecId, Parameters, Stream, StreamId, StreamKind, TimeBase};

/// MPEG-TS timestamps tick at 90 kHz; every stream uses this time base.
pub const TIME_BASE: TimeBase = TimeBase::new(1, 90_000);

/// How a stream's PES payloads become packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
	H264,
	Hevc,
	Mpeg2Video,
	/// Other video: one packet per PES, keyframes unknown.
	Video,
	Adts,
	MpegAudio,
	Ac3,
	/// Opus access units, each behind a control header (ETSI TS 102 366
	/// style framing from the "Opus in MPEG-TS" specification).
	Opus,
	/// One packet per PES (subtitles, timed metadata, LATM).
	Whole,
	/// Private data with nothing in the PMT naming it: the first PES is
	/// probed for an audio format, as FFmpeg's `request_probe` does.
	Unknown,
}

/// Stream types of SAMPLE-AES encrypted streams (Apple's "MPEG-2 Stream
/// Encryption Format for HTTP Live Streaming" §2.1): the same framing as
/// their clear counterparts, with parts of each sample encrypted.
pub fn is_sample_aes(stream_type: u8) -> bool {
	matches!(stream_type, 0xDB | 0xCF | 0xC1 | 0xC2)
}

/// The codec a PMT entry carries, or `None` for streams RMP skips
/// (unknown types, SCTE-35).
pub fn classify(
	es: &EsInfo,
	program_descriptors: &[u8],
) -> Option<(&'static str, StreamKind, Framing)> {
	use Framing::*;
	use StreamKind::{Audio, Other, Subtitle, Video as V};
	let registration = |data: &[u8]| {
		psi::descriptors(data)
			.find(|(t, d)| *t == 0x05 && d.len() >= 4)
			.map(|(_, d)| [d[0], d[1], d[2], d[3]])
	};
	let reg = registration(&es.descriptors).or_else(|| registration(program_descriptors));
	let hdmv = reg == Some(*b"HDMV");
	let by_type = match es.stream_type {
		0x01 | 0x02 => Some(("mpeg2video", V, Mpeg2Video)),
		0x03 | 0x04 => Some(("mp3", Audio, MpegAudio)),
		0x0F => Some(("aac", Audio, Adts)),
		0x10 => Some(("mpeg4", V, Video)),
		0x11 => Some(("aac_latm", Audio, Whole)),
		0x15 => Some(("timed_id3", Other, Whole)),
		0x1B => Some(("h264", V, H264)),
		0x24 => Some(("hevc", V, Hevc)),
		0x81 => Some(("ac3", Audio, Ac3)),
		0x87 => Some(("eac3", Audio, Ac3)),
		// SAMPLE-AES: H.264, ADTS AAC, AC-3, E-AC-3.
		0xDB => Some(("h264", V, H264)),
		0xCF => Some(("aac", Audio, Adts)),
		0xC1 => Some(("ac3", Audio, Ac3)),
		0xC2 => Some(("eac3", Audio, Ac3)),
		// Blu-ray (HDMV) stream types.
		0x84 | 0xA1 if hdmv => Some(("eac3", Audio, Ac3)),
		0x90 if hdmv => Some(("hdmv_pgs_subtitle", Subtitle, Whole)),
		_ => None,
	};
	if by_type.is_some() {
		return by_type;
	}
	// Registration descriptors name the format of private stream types.
	let by_reg = match reg.as_ref() {
		Some(b"AC-3") => Some(("ac3", Audio, Ac3)),
		Some(b"EAC3") => Some(("eac3", Audio, Ac3)),
		Some(b"HEVC") => Some(("hevc", V, Hevc)),
		Some(b"ID3 ") => Some(("timed_id3", Other, Whole)),
		Some(b"Opus") => Some(("opus", Audio, Opus)),
		_ => None,
	};
	if by_reg.is_some() || es.stream_type != 0x06 {
		return by_reg;
	}
	// PES private data (0x06): the DVB descriptors say what it holds.
	psi::descriptors(&es.descriptors)
		.find_map(|(tag, _)| match tag {
			0x6A => Some(("ac3", Audio, Ac3)),
			0x7A => Some(("eac3", Audio, Ac3)),
			0x59 => Some(("dvb_subtitle", Subtitle, Whole)),
			0x56 => Some(("dvb_teletext", Subtitle, Whole)),
			_ => None,
		})
		.or(Some(("", Audio, Unknown)))
}

/// Identifies an [`Framing::Unknown`] stream from its first PES payload:
/// two consecutive frames of one audio format (or one that fills the PES).
pub fn probe_audio(data: &[u8]) -> Option<(&'static str, Framing)> {
	[Framing::Adts, Framing::Ac3, Framing::MpegAudio].into_iter().find_map(|framing| {
		let first = audio_frame(framing, data)?;
		let confirmed = first.len == data.len()
			|| data.get(first.len..).and_then(|rest| audio_frame(framing, rest)).is_some();
		confirmed.then_some((first.codec, framing))
	})
}

/// `channel_config_code` of an Opus stream: the DVB extension descriptor
/// (0x7F) with the provisional Opus tag 0x80.
pub fn opus_channel_config(es: &EsInfo) -> Option<u8> {
	psi::descriptors(&es.descriptors)
		.find(|(t, d)| *t == 0x7F && d.len() >= 2 && d[0] == 0x80)
		.map(|(_, d)| d[1])
}

/// Coupled streams and channel order of the Vorbis channel layouts
/// (RFC 7845 §5.1.1.2), by channel count - 1.
const OPUS_COUPLED: [u8; 8] = [0, 1, 1, 2, 2, 2, 3, 3];
const OPUS_VORBIS_MAP: [&[u8]; 8] = [
	&[0],
	&[0, 1],
	&[0, 2, 1],
	&[0, 1, 2, 3],
	&[0, 4, 1, 2, 3],
	&[0, 4, 1, 2, 3, 5],
	&[0, 4, 1, 2, 3, 5, 6],
	&[0, 6, 1, 2, 3, 4, 5, 7],
];

/// The `OpusHead` a channel configuration code stands for: 1-2 channels
/// with mapping family 0, 3-8 with the Vorbis layouts (family 1), `0x80 |
/// n` for n uncoupled streams (family 255), 0 for dual mono. Without a
/// descriptor, stereo.
pub fn opus_head(code: Option<u8>, pre_skip: u16) -> Option<crate::codecs::opus::OpusHead> {
	use crate::codecs::opus::{OpusHead, header::ChannelMapping};
	let mut head = OpusHead::new(2, pre_skip, 48_000);
	match code.unwrap_or(2) {
		c @ 1..=2 => head = OpusHead::new(c, pre_skip, 48_000),
		c @ 3..=8 => {
			let i = usize::from(c - 1);
			head.channels = c;
			head.mapping = ChannelMapping {
				family: 1,
				streams: c - OPUS_COUPLED[i],
				coupled: OPUS_COUPLED[i],
				mapping: OPUS_VORBIS_MAP[i].to_vec(),
			};
		}
		0 | 0x82..=0x88 => {
			let c = if code == Some(0) { 2 } else { code.unwrap_or(2) & 0x7F };
			head.channels = c;
			head.mapping =
				ChannelMapping { family: 255, streams: c, coupled: 0, mapping: (0..c).collect() };
		}
		_ => return None,
	}
	Some(head)
}

/// The `channel_config_code` for an `OpusHead`: the inverse of
/// [`opus_head`], or `None` for layouts it can't express.
pub fn opus_channel_code(head: &crate::codecs::opus::OpusHead) -> Option<u8> {
	let m = &head.mapping;
	let c = head.channels;
	match m.family {
		0 if (1..=2).contains(&c) => Some(c),
		1 if (3..=8).contains(&c)
			&& m.coupled == OPUS_COUPLED[usize::from(c - 1)]
			&& m.streams == c - m.coupled
			&& m.mapping == OPUS_VORBIS_MAP[usize::from(c - 1)] =>
		{
			Some(c)
		}
		1 | 255
			if (2..=8).contains(&c)
				&& m.streams == c
				&& m.coupled == 0
				&& m.mapping.iter().copied().eq(0..c) =>
		{
			Some(if c == 2 && m.family == 255 { 0 } else { 0x80 | c })
		}
		_ => None,
	}
}

/// One Opus access unit behind its control header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusUnit {
	/// Bytes of the control header.
	pub header: usize,
	/// Bytes of the Opus packet that follows it.
	pub size: usize,
	pub start_trim: u16,
	pub end_trim: u16,
}

/// Parses the control header at the start of `data`: the 11-bit prefix
/// 0x3FF, trim and extension flags, the AU size as a run of bytes summed
/// until one is below 255, then the trims and the extension. `None` when
/// `data` doesn't start with one or is too short to tell.
pub fn opus_control_header(data: &[u8]) -> Option<OpusUnit> {
	if data.len() < 3 || data[0] != 0x7F || data[1] & 0xE0 != 0xE0 {
		return None;
	}
	let flags = data[1];
	let mut at = 2;
	let mut size = 0usize;
	loop {
		let b = *data.get(at)?;
		at += 1;
		size += usize::from(b);
		if b != 0xFF {
			break;
		}
	}
	let mut field = || -> Option<u16> {
		let v = u16::from_be_bytes([*data.get(at)?, *data.get(at + 1)?]) & 0x1FFF;
		at += 2;
		Some(v)
	};
	let start_trim = if flags & 0x10 != 0 { field()? } else { 0 };
	let end_trim = if flags & 0x08 != 0 { field()? } else { 0 };
	if flags & 0x04 != 0 {
		let len = usize::from(*data.get(at)?);
		at += 1 + len;
	}
	Some(OpusUnit { header: at, size, start_trim, end_trim })
}

/// Writes the control header for an Opus packet of `size` bytes.
pub fn write_opus_control_header(out: &mut Vec<u8>, size: usize, start_trim: u16, end_trim: u16) {
	let mut flags = 0xE0u8;
	if start_trim > 0 {
		flags |= 0x10;
	}
	if end_trim > 0 {
		flags |= 0x08;
	}
	out.extend_from_slice(&[0x7F, flags]);
	// FFmpeg's encoding: size / 255 bytes of 255, then the remainder (so a
	// multiple of 255 ends with a 0 byte).
	out.extend(std::iter::repeat_n(0xFF, size / 255));
	out.push((size % 255) as u8);
	if start_trim > 0 {
		out.extend_from_slice(&(start_trim & 0x1FFF).to_be_bytes());
	}
	if end_trim > 0 {
		out.extend_from_slice(&(end_trim & 0x1FFF).to_be_bytes());
	}
}

/// ISO 639 language from the ES descriptors (`und` is left out).
pub fn language(es: &EsInfo) -> Option<String> {
	let (_, d) = psi::descriptors(&es.descriptors).find(|(t, d)| *t == 0x0A && d.len() >= 3)?;
	let lang: String = d[..3].iter().map(|&b| char::from(b)).collect();
	(lang.chars().all(|c| c.is_ascii_alphabetic()) && lang != "und").then_some(lang)
}

/// One audio frame found in a byte stream.
pub struct AudioFrame {
	pub len: usize,
	pub samples: u32,
	pub sample_rate: u32,
	pub channels: u8,
	/// Codec name when the header refines the stream type (`mp2` vs `mp3`,
	/// `eac3` vs `ac3`).
	pub codec: &'static str,
	pub bitrate: Option<u32>,
}

/// Parses the frame header at the start of `data` (`None`: not a header,
/// or too few bytes to tell).
pub fn audio_frame(framing: Framing, data: &[u8]) -> Option<AudioFrame> {
	match framing {
		Framing::Adts => {
			let h = AdtsHeader::parse(data)?;
			let rate = *SAMPLE_RATES.get(h.sample_rate_index)?;
			if h.frame_length < h.header_length {
				return None;
			}
			let channels = match h.channel_config {
				7 => 8,
				0 => 2,
				c => c,
			};
			Some(AudioFrame {
				len: h.frame_length,
				samples: 1024 * h.raw_blocks as u32,
				sample_rate: rate,
				channels,
				codec: "aac",
				bitrate: None,
			})
		}
		Framing::MpegAudio => {
			let h = FrameHeader::parse(data.get(..4)?)?;
			Some(AudioFrame {
				len: h.frame_size,
				samples: h.samples,
				sample_rate: h.sample_rate,
				channels: h.channels(),
				codec: h.codec_name(),
				bitrate: Some(h.bitrate_kbps * 1000),
			})
		}
		Framing::Ac3 => {
			let h = Ac3Header::parse(&mut BitReader::new(data.get(..data.len().min(64))?)).ok()?;
			if h.frame_size == 0 {
				return None;
			}
			let (channels, _) = ac3::layout_of(h.channel_mode, h.lfe_on);
			Some(AudioFrame {
				len: h.frame_size,
				samples: h.num_blocks as u32 * 256,
				sample_rate: h.sample_rate,
				channels: channels as u8,
				codec: if h.is_eac3() { "eac3" } else { "ac3" },
				bitrate: Some(h.bit_rate),
			})
		}
		_ => None,
	}
}

/// What a video access unit tells about itself.
#[derive(Debug, Default)]
pub struct VideoInfo {
	pub keyframe: bool,
	/// Annex B parameter sets (with start codes), FFmpeg's extradata layout.
	pub parameter_sets: Vec<u8>,
	pub width: u32,
	pub height: u32,
	pub pixel_format: Option<&'static str>,
}

fn push_annex_b(out: &mut Vec<u8>, nal: &[u8]) {
	out.extend_from_slice(&[0, 0, 0, 1]);
	out.extend_from_slice(nal);
}

/// Scans an access unit: keyframe (H.264 IDR or recovery-point SEI, HEVC
/// IRAP, MPEG-2 I picture) and, when it carries them, the parameter sets
/// and picture size.
pub fn inspect_video(framing: Framing, data: &[u8]) -> VideoInfo {
	let mut info = VideoInfo::default();
	match framing {
		Framing::H264 => {
			let (mut sps, mut pps) = (Vec::new(), Vec::new());
			for nal in split_annex_b(data) {
				match nal[0] & 0x1F {
					5 => info.keyframe = true,
					// SEI with a recovery point (payload type 6) first.
					6 if nal.get(1) == Some(&6) => info.keyframe = true,
					7 => {
						if let Some(s) = sps_summary(&unescape(&nal[1..])) {
							info.width = s.width;
							info.height = s.height;
							info.pixel_format = chroma_format(s.chroma_format_idc, s.bit_depth_luma);
						}
						push_annex_b(&mut sps, nal);
					}
					8 => push_annex_b(&mut pps, nal),
					_ => {}
				}
			}
			if !sps.is_empty() && !pps.is_empty() {
				sps.extend_from_slice(&pps);
				info.parameter_sets = sps;
			}
		}
		Framing::Hevc => {
			let mut sets = [Vec::new(), Vec::new(), Vec::new()];
			for nal in split_annex_b(data) {
				let kind = (nal[0] >> 1) & 0x3F;
				match kind {
					16..=23 => info.keyframe = true,
					32..=34 => {
						if kind == 33 {
							if let Some(s) = crate::container::bsf::hevc_sps_summary(&unescape(nal)) {
								info.width = s.width;
								info.height = s.height;
								info.pixel_format = chroma_format(s.chroma_format_idc, s.bit_depth_luma);
							}
						}
						push_annex_b(&mut sets[usize::from(kind - 32)], nal);
					}
					_ => {}
				}
			}
			if sets.iter().all(|s| !s.is_empty()) {
				info.parameter_sets = sets.concat();
			}
		}
		Framing::Mpeg2Video => {
			let mut i = 0;
			while i + 6 <= data.len() {
				if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
					match data[i + 3] {
						// sequence_header: 12-bit width and height.
						0xB3 => {
							info.width = (u32::from(data[i + 4]) << 4) | u32::from(data[i + 5] >> 4);
							info.height =
								data.get(i + 6).map_or(0, |&b| (u32::from(data[i + 5] & 0x0F) << 8) | u32::from(b));
							info.pixel_format = Some("yuv420p");
						}
						// picture_header: picture_coding_type 1 = I.
						0x00 => {
							info.keyframe = (data[i + 5] >> 3) & 7 == 1;
							break;
						}
						_ => {}
					}
					i += 4;
				} else {
					i += 1;
				}
			}
		}
		_ => {}
	}
	info
}

fn chroma_format(idc: u32, bit_depth: u32) -> Option<&'static str> {
	Some(match (idc, bit_depth) {
		(0, 8) => "gray",
		(1, 8) => "yuv420p",
		(2, 8) => "yuv422p",
		(3, 8) => "yuv444p",
		(1, 10) => "yuv420p10le",
		(2, 10) => "yuv422p10le",
		(3, 10) => "yuv444p10le",
		_ => return None,
	})
}

/// Builds the stream for a classified entry once its first data was seen.
pub fn make_stream(
	index: usize,
	codec: &str,
	kind: StreamKind,
	audio: Option<&AudioFrame>,
	video: Option<&VideoInfo>,
) -> Stream {
	let id = StreamId(index);
	let mut stream = match (kind, audio, video) {
		(StreamKind::Audio, Some(a), _) => {
			let mut s = Stream::audio(id, a.sample_rate, a.channels, CodecId::new(a.codec));
			s.parameters.bitrate = a.bitrate;
			s
		}
		(StreamKind::Video, _, Some(v)) => {
			let mut s = Stream::video(
				id,
				v.width,
				v.height,
				v.pixel_format.unwrap_or("yuv420p"),
				CodecId::new(codec),
			);
			s.parameters.extradata = v.parameter_sets.clone();
			s
		}
		_ => Stream {
			id,
			kind,
			time_base: TIME_BASE,
			duration: None,
			parameters: Parameters {
				codec: CodecId::new(codec),
				profile: None,
				bitrate: None,
				sample_rate: None,
				channels: None,
				width: None,
				height: None,
				pixel_format: None,
				extradata: Vec::new(),
				bits_per_sample: None,
			},
			metadata: Default::default(),
		},
	};
	stream.time_base = TIME_BASE;
	stream
}

#[cfg(test)]
mod tests {
	use super::*;

	fn es(stream_type: u8, descriptors: &[u8]) -> EsInfo {
		EsInfo { stream_type, pid: 0x100, descriptors: descriptors.to_vec() }
	}

	#[test]
	fn stream_types() {
		assert_eq!(classify(&es(0x1B, &[]), &[]).unwrap().0, "h264");
		assert_eq!(classify(&es(0x0F, &[]), &[]).unwrap().2, Framing::Adts);
		assert_eq!(classify(&es(0x06, &[0x6A, 1, 0]), &[]).unwrap().0, "ac3");
		assert_eq!(classify(&es(0x06, &[0x05, 4, b'E', b'A', b'C', b'3']), &[]).unwrap().0, "eac3");
		assert_eq!(classify(&es(0xC1, &[0x05, 4, b'A', b'C', b'-', b'3']), &[]).unwrap().0, "ac3");
		assert!(classify(&es(0x86, &[]), &[]).is_none()); // SCTE-35
		let hdmv = [0x05, 4, b'H', b'D', b'M', b'V'];
		assert_eq!(classify(&es(0x90, &[]), &hdmv).unwrap().0, "hdmv_pgs_subtitle");
		assert_eq!(classify(&es(0x06, &[]), &hdmv).unwrap().2, Framing::Unknown);
		assert_eq!(classify(&es(0xDB, &[]), &[]).unwrap().0, "h264"); // SAMPLE-AES H.264
		assert!(is_sample_aes(0xCF) && !is_sample_aes(0x0F));
		let opus = [0x05, 4, b'O', b'p', b'u', b's', 0x7F, 2, 0x80, 6];
		assert_eq!(classify(&es(0x06, &opus), &[]).unwrap().2, Framing::Opus);
		assert_eq!(opus_channel_config(&es(0x06, &opus)), Some(6));
		assert_eq!(language(&es(0x0F, &[0x0A, 4, b'f', b'r', b'a', 0])).as_deref(), Some("fra"));
	}

	#[test]
	fn opus_framing() {
		let mut out = Vec::new();
		write_opus_control_header(&mut out, 510, 312, 0);
		assert_eq!(out, [0x7F, 0xF0, 0xFF, 0xFF, 0x00, 0x01, 0x38]);
		let unit = opus_control_header(&out).unwrap();
		assert_eq!(unit, OpusUnit { header: 7, size: 510, start_trim: 312, end_trim: 0 });
		assert!(opus_control_header(&[0x7F, 0xE0]).is_none());
		assert!(opus_control_header(&[0xFF, 0xF1, 0]).is_none());
		for code in [1u8, 2, 3, 6, 8, 0, 0x84] {
			let head = opus_head(Some(code), 0).unwrap();
			assert_eq!(opus_channel_code(&head), Some(code), "code {code:#x}");
		}
		assert!(opus_head(Some(9), 0).is_none());
	}

	#[test]
	fn mpeg2_picture_types() {
		let i_pic = [0, 0, 1, 0, 0, 0x08, 0, 0];
		let p_pic = [0, 0, 1, 0, 0, 0x10, 0, 0];
		assert!(inspect_video(Framing::Mpeg2Video, &i_pic).keyframe);
		assert!(!inspect_video(Framing::Mpeg2Video, &p_pic).keyframe);
	}
}
