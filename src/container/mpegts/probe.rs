use super::packet::detect_layout;
use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

pub struct TsProbe;

impl Probe for TsProbe {
	/// A run of sync bytes at a 188, 192 or 204-byte stride, like FFmpeg's
	/// `mpegts_probe`.
	fn demuxer_matches(data: &[u8]) -> bool {
		detect_layout(data).is_some()
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if hint.mime_type.as_deref().map(str::trim) == Some("video/mp2t") {
			return true;
		}
		hint.extension.as_deref().is_some_and(|e| {
			["ts", "m2t", "mts", "m2ts", "mpegts", "trp"].iter().any(|x| e.eq_ignore_ascii_case(x))
		})
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::TsDemuxer::new(io)?))
	}

	/// `.m2ts` is written with plain 188-byte packets, which players
	/// accept; Blu-ray's 192-byte layout isn't produced.
	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::TsMuxer::new(io)))
	}
}
