use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

pub struct OggProbe;

impl Probe for OggProbe {
	/// `"OggS"`, stream structure version 0, with the BOS flag set.
	fn demuxer_matches(data: &[u8]) -> bool {
		data.len() >= 6 && data.starts_with(b"OggS") && data[4] == 0 && data[5] & 0x02 != 0
	}

	/// `.ogg`, `.oga`, `.ogx` or an Ogg MIME type.
	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if matches!(mime, "audio/ogg" | "application/ogg" | "video/ogg") {
				return true;
			}
		}
		hint.extension.as_deref().is_some_and(|e| {
			let e = e.to_ascii_lowercase();
			matches!(e.as_str(), "ogg" | "oga" | "ogx")
		})
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::OggDemuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::OggMuxer::new(io, false)))
	}
}

/// `.opus`: Ogg restricted to Opus streams (FFmpeg's `opus` muxer).
/// Demuxing goes through [`OggProbe`].
pub struct OpusProbe;

impl Probe for OpusProbe {
	fn demuxer_matches(data: &[u8]) -> bool {
		OggProbe::demuxer_matches(data)
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if mime == "audio/opus" {
				return true;
			}
		}
		hint.extension.as_deref().is_some_and(|e| e.eq_ignore_ascii_case("opus"))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		OggProbe::create_demuxer(io)
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::OggMuxer::new(io, true)))
	}
}
