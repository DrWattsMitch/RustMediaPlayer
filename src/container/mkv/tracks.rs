//! `Tracks` parsing: Matroska codec IDs → RMP codec IDs (FFmpeg names) and
//! `CodecPrivate` → [`Parameters::extradata`] in FFmpeg's layout.
//!
//! FFmpeg: `matroska_parse_tracks` and the `ff_mkv_codec_tags` table in
//! `libavformat/matroska.c`.

use super::ebml::{self, Children, id};
use crate::codecs::flac;
use crate::core::{
	CodecId, Metadata, Parameters, RawValue, Stream, StreamId, StreamKind, TimeBase,
};

/// Matroska `TrackType` values.
const TYPE_VIDEO: u64 = 1;
const TYPE_AUDIO: u64 = 2;
const TYPE_SUBTITLE: u64 = 17;

/// Largest `CodecPrivate` accepted (real ones are at most a few KB; Vorbis
/// headers with big comment blocks can reach a few hundred KB).
const MAX_CODEC_PRIVATE: usize = 16 * 1024 * 1024;

/// One demuxable track.
#[derive(Debug, Clone)]
pub struct Track {
	pub number: u64,
	pub uid: u64,
	pub stream: Stream,
	/// `DefaultDuration`, in nanoseconds (0 = unknown).
	pub default_duration_ns: u64,
	/// Bytes removed from the front of every frame by "header stripping"
	/// content compression (`ContentCompAlgo` 3), restored on read.
	pub strip_prefix: Vec<u8>,
}

/// Why a track can't be demuxed. The track is skipped; the file stays playable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
	/// Not audio, video or subtitles (buttons, logos, complex tracks).
	Kind,
	/// Encrypted, or compressed with zlib/bzlib/lzo.
	Encoding,
	Invalid,
}

/// Maps a Matroska codec ID to the FFmpeg codec name.
pub fn codec_name(codec_id: &str, bit_depth: u64) -> String {
	let name = match codec_id {
		"V_MPEG4/ISO/AVC" => "h264",
		"V_MPEGH/ISO/HEVC" => "hevc",
		"V_AV1" => "av1",
		"V_VP8" => "vp8",
		"V_VP9" => "vp9",
		"V_MPEG4/ISO/ASP" | "V_MPEG4/ISO/SP" | "V_MPEG4/ISO/AP" => "mpeg4",
		"V_MPEG1" => "mpeg1video",
		"V_MPEG2" => "mpeg2video",
		"V_MJPEG" => "mjpeg",
		"V_THEORA" => "theora",
		"V_PRORES" => "prores",
		"V_FFV1" => "ffv1",
		"A_OPUS" => "opus",
		"A_VORBIS" => "vorbis",
		"A_FLAC" => "flac",
		"A_ALAC" => "alac",
		"A_MPEG/L3" => "mp3",
		"A_MPEG/L2" => "mp2",
		"A_MPEG/L1" => "mp1",
		"A_AC3" => "ac3",
		"A_EAC3" => "eac3",
		"A_DTS" => "dts",
		"A_TRUEHD" => "truehd",
		"A_PCM/INT/LIT" => match bit_depth {
			8 => "pcm_u8",
			24 => "pcm_s24le",
			32 => "pcm_s32le",
			_ => "pcm_s16le",
		},
		"A_PCM/INT/BIG" => match bit_depth {
			8 => "pcm_s8",
			24 => "pcm_s24be",
			32 => "pcm_s32be",
			_ => "pcm_s16be",
		},
		"A_PCM/FLOAT/IEEE" => match bit_depth {
			64 => "pcm_f64le",
			_ => "pcm_f32le",
		},
		"S_TEXT/UTF8" => "subrip",
		"S_TEXT/ASS" | "S_TEXT/SSA" | "S_ASS" | "S_SSA" => "ass",
		"S_TEXT/WEBVTT" | "D_WEBVTT/SUBTITLES" => "webvtt",
		"S_VOBSUB" => "dvd_subtitle",
		"S_HDMV/PGS" => "hdmv_pgs_subtitle",
		id if id.starts_with("A_AAC") => "aac",
		id if id.starts_with("A_REAL") || id.starts_with("V_REAL") => "real",
		_ => return codec_id.to_ascii_lowercase(),
	};
	name.to_string()
}

/// AAC sampling frequency table (ISO/IEC 14496-3 Table 1.18).
const AAC_RATES: [u32; 13] =
	[96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

fn aac_rate_index(rate: u32) -> u32 {
	AAC_RATES.iter().position(|&r| r == rate).map_or(15, |i| i as u32)
}

/// Builds an AudioSpecificConfig for legacy `A_AAC/MPEG{2,4}/<profile>`
/// tracks that carry no `CodecPrivate`, as FFmpeg's matroskadec does.
pub fn synthesize_aac_config(codec_id: &str, rate: u32, channels: u8, output_rate: u32) -> Vec<u8> {
	let profile = codec_id.rsplit('/').next().unwrap_or("LC");
	let object_type: u32 = match profile {
		"MAIN" => 1,
		"SSR" => 3,
		"LTP" => 4,
		_ => 2,
	};
	let sbr = profile.contains("SBR");
	let mut bits: Vec<(u32, u32)> = vec![(object_type, 5)];
	let index = aac_rate_index(rate);
	bits.push((index, 4));
	if index == 15 {
		bits.push((rate, 24));
	}
	bits.push((u32::from(channels.min(7)), 4));
	bits.push((0, 3)); // GASpecificConfig: frameLength, dependsOnCore, extension
	if sbr {
		let ext_rate = if output_rate > 0 { output_rate } else { rate * 2 };
		let ext_index = aac_rate_index(ext_rate);
		bits.extend([(0x2B7, 11), (5, 5), (1, 1), (ext_index, 4)]);
		if ext_index == 15 {
			bits.push((ext_rate, 24));
		}
	}
	let mut out = Vec::new();
	let (mut acc, mut used) = (0u64, 0u32);
	for (value, width) in bits {
		acc = (acc << width) | (u64::from(value) & ((1u64 << width) - 1));
		used += width;
		while used >= 8 {
			out.push((acc >> (used - 8)) as u8);
			used -= 8;
		}
	}
	if used > 0 {
		out.push((acc << (8 - used)) as u8);
	}
	out
}

/// Header-stripping prefix of a track, or `Err` when the track uses an
/// encoding RMP can't undo.
fn content_encoding(entry: &[u8]) -> Result<Vec<u8>, Skip> {
	let Some(encodings) = ebml::find(entry, id::CONTENT_ENCODINGS) else { return Ok(Vec::new()) };
	let mut prefix = Vec::new();
	for (eid, encoding) in Children::new(encodings) {
		if eid != id::CONTENT_ENCODING {
			continue;
		}
		let scope = ebml::child_uint(encoding, id::CONTENT_ENCODING_SCOPE, 1);
		if ebml::child_uint(encoding, id::CONTENT_ENCODING_TYPE, 0) != 0 {
			return Err(Skip::Encoding); // encryption
		}
		let Some(compression) = ebml::find(encoding, id::CONTENT_COMPRESSION) else { continue };
		let algo = ebml::child_uint(compression, id::CONTENT_COMP_ALGO, 0);
		if algo != 3 {
			return Err(Skip::Encoding);
		}
		if scope & 1 != 0 {
			prefix = ebml::find(compression, id::CONTENT_COMP_SETTINGS).unwrap_or(&[]).to_vec();
		}
	}
	Ok(prefix)
}

/// Parses one `TrackEntry`. `index` becomes the stream ID.
pub fn parse_entry(entry: &[u8], index: usize, time_base: TimeBase) -> Result<Track, Skip> {
	let number = ebml::child_uint(entry, id::TRACK_NUMBER, 0);
	if number == 0 {
		return Err(Skip::Invalid);
	}
	let kind = match ebml::child_uint(entry, id::TRACK_TYPE, 0) {
		TYPE_VIDEO => StreamKind::Video,
		TYPE_AUDIO => StreamKind::Audio,
		TYPE_SUBTITLE => StreamKind::Subtitle,
		_ => return Err(Skip::Kind),
	};
	let strip_prefix = content_encoding(entry)?;
	let codec_id = ebml::child_string(entry, id::CODEC_ID).unwrap_or_default();
	let private = ebml::find(entry, id::CODEC_PRIVATE).unwrap_or(&[]);
	if private.len() > MAX_CODEC_PRIVATE {
		return Err(Skip::Invalid);
	}

	let mut parameters = Parameters {
		codec: CodecId::new(""),
		profile: None,
		bitrate: None,
		sample_rate: None,
		channels: None,
		width: None,
		height: None,
		pixel_format: None,
		extradata: private.to_vec(),
		bits_per_sample: None,
	};
	let mut bit_depth = 0;
	let mut output_rate = 0;
	if let Some(video) = ebml::find(entry, id::VIDEO) {
		let width = ebml::child_uint(video, id::PIXEL_WIDTH, 0);
		let height = ebml::child_uint(video, id::PIXEL_HEIGHT, 0);
		parameters.width = u32::try_from(width).ok().filter(|&w| w > 0);
		parameters.height = u32::try_from(height).ok().filter(|&h| h > 0);
	}
	if let Some(audio) = ebml::find(entry, id::AUDIO) {
		let rate = ebml::child_float(audio, id::SAMPLING_FREQUENCY).unwrap_or(8000.0);
		let out = ebml::child_float(audio, id::OUTPUT_SAMPLING_FREQUENCY).unwrap_or(0.0);
		let to_rate =
			|r: f64| if r.is_finite() && (1.0..1e7).contains(&r) { r.round() as u32 } else { 0 };
		parameters.sample_rate = Some(to_rate(rate)).filter(|&r| r > 0);
		output_rate = to_rate(out);
		let channels = ebml::child_uint(audio, id::CHANNELS, 1);
		parameters.channels = u8::try_from(channels).ok().filter(|&c| c > 0);
		bit_depth = ebml::child_uint(audio, id::BIT_DEPTH, 0);
	}
	let codec = codec_name(&codec_id, bit_depth);

	match codec.as_str() {
		"aac" if parameters.extradata.is_empty() => {
			let rate = parameters.sample_rate.unwrap_or(44_100);
			parameters.extradata =
				synthesize_aac_config(&codec_id, rate, parameters.channels.unwrap_or(2), output_rate);
		}
		"aac" => {
			if let Some(config) = crate::codecs::aac::config::parse_audio_specific_config(private) {
				parameters.sample_rate = Some(config.sample_rate);
				if config.channels > 0 {
					parameters.channels = Some(config.channels);
				}
			}
		}
		"flac" => {
			// Matroska stores "fLaC" + metadata blocks; FFmpeg's FLAC
			// extradata is the bare 34-byte STREAMINFO.
			if let Some(info) = flac::StreamInfo::from_extradata(private) {
				parameters.extradata = info.to_bytes().to_vec();
				parameters.sample_rate = Some(info.sample_rate);
				parameters.channels = Some(info.channels);
				parameters.bits_per_sample = Some(info.bits_per_sample);
			}
		}
		_ => {}
	}
	if output_rate > 0 && codec != "aac" {
		parameters.sample_rate = Some(output_rate);
	}
	if codec == "opus" {
		// Opus always decodes at 48 kHz; the track's rate is only the
		// encoder's input rate (FFmpeg reports 48 kHz as well).
		parameters.sample_rate = Some(48_000);
	}
	parameters.codec = CodecId::new(&codec);

	let mut metadata = Metadata::default();
	let language = ebml::child_string(entry, id::LANGUAGE_BCP47)
		.or_else(|| ebml::child_string(entry, id::LANGUAGE))
		.unwrap_or_else(|| "eng".into());
	if language != "und" && !language.is_empty() {
		metadata.raw.insert("language".into(), RawValue::String(language));
	}
	if let Some(name) = ebml::child_string(entry, id::NAME).filter(|n| !n.is_empty()) {
		metadata.title = Some(name.clone());
		metadata.raw.insert("title".into(), RawValue::String(name));
	}
	// CodecDelay (ns) is the Opus pre-skip; expose it with the gapless key the
	// MP3/MP4 demuxers use, in samples at the output rate.
	let codec_delay = ebml::child_uint(entry, id::CODEC_DELAY, 0);
	let delay_samples = match (codec.as_str(), codec_delay) {
		("opus", 0) => crate::codecs::opus::pre_skip(private).map(u64::from).unwrap_or(0),
		(_, 0) => 0,
		(_, ns) => {
			let rate = if codec == "opus" { 48_000 } else { parameters.sample_rate.unwrap_or(0) };
			((u128::from(ns) * u128::from(rate) + 500_000_000) / 1_000_000_000) as u64
		}
	};
	if delay_samples > 0 {
		metadata.raw.insert("encoder_delay".into(), RawValue::String(delay_samples.to_string()));
	}
	if ebml::child_uint(entry, id::FLAG_DEFAULT, 1) == 0 {
		metadata.raw.insert("default".into(), RawValue::String("0".into()));
	}
	let uid = ebml::child_uint(entry, id::TRACK_UID, 0);
	metadata.raw.insert("matroska_codec_id".into(), RawValue::String(codec_id));

	let stream =
		Stream { id: StreamId(index), kind, time_base, duration: None, parameters, metadata };
	Ok(Track {
		number,
		uid,
		stream,
		default_duration_ns: ebml::child_uint(entry, id::DEFAULT_DURATION, 0),
		strip_prefix,
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn aac_configs() {
		assert_eq!(synthesize_aac_config("A_AAC/MPEG4/LC", 44_100, 2, 0), vec![0x12, 0x10]);
		assert_eq!(synthesize_aac_config("A_AAC/MPEG2/MAIN", 48_000, 1, 0), vec![0x09, 0x88]);
		let sbr = synthesize_aac_config("A_AAC/MPEG4/LC/SBR", 22_050, 2, 44_100);
		let parsed = crate::codecs::aac::config::parse_audio_specific_config(&sbr).unwrap();
		assert_eq!(parsed.sample_rate, 22_050);
	}

	#[test]
	fn pcm_ids_use_bit_depth() {
		assert_eq!(codec_name("A_PCM/INT/LIT", 24), "pcm_s24le");
		assert_eq!(codec_name("A_PCM/FLOAT/IEEE", 64), "pcm_f64le");
		assert_eq!(codec_name("V_MPEG4/ISO/AVC", 0), "h264");
		assert_eq!(codec_name("A_AAC/MPEG4/LC/SBR", 0), "aac");
	}
}
