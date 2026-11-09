//! Duration of one compressed audio packet, from its own header bytes.
//!
//! Containers without per-packet durations (Matroska without
//! `DefaultDuration`, Ogg) use this the way FFmpeg uses its parsers
//! (`av_parser_parse2` → `duration`).

use crate::codecs::{flac, mp3, opus, pcm};
use crate::core::Parameters;

/// `(samples, sample_rate)` of one packet, when the codec allows it.
pub fn packet_samples(parameters: &Parameters, data: &[u8]) -> Option<(u64, u32)> {
	let rate = parameters.sample_rate.filter(|&r| r > 0);
	match parameters.codec.id.as_str() {
		"aac" => Some((1024, rate?)),
		"mp3" | "mp2" | "mp1" => {
			let header = mp3::FrameHeader::parse(data)?;
			Some((u64::from(header.samples), header.sample_rate))
		}
		"opus" => Some((u64::from(opus::packet_samples(data)?), 48_000)),
		"flac" => {
			let header = flac::FrameHeader::parse(data)?;
			let rate = if header.sample_rate > 0 { header.sample_rate } else { rate? };
			Some((u64::from(header.block_size), rate))
		}
		id => {
			let codec = pcm::PcmCodec::from_id(id)?;
			let frame = codec.coded_bytes() * usize::from(parameters.channels.filter(|&c| c > 0)?);
			Some(((data.len() / frame) as u64, rate?))
		}
	}
}
