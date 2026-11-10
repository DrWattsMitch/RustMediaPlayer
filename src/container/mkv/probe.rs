use super::ebml;
use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

pub struct MkvProbe;

impl Probe for MkvProbe {
	/// EBML magic, then a `matroska` or `webm` DocType inside the header
	/// (FFmpeg's `matroska_probe`).
	fn demuxer_matches(data: &[u8]) -> bool {
		let Ok(header) = ebml::parse_header(data) else { return false };
		if header.id != ebml::id::EBML {
			return false;
		}
		let end =
			header.size.map_or(data.len(), |s| (header.len as u64 + s).min(data.len() as u64) as usize);
		let body = &data[header.len..end];
		body.windows(8).any(|w| w == b"matroska") || body.windows(4).any(|w| w == b"webm")
	}

	/// `.mkv`, `.mka`, `.mks` or a Matroska MIME type.
	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if matches!(
				mime,
				"video/x-matroska" | "audio/x-matroska" | "video/matroska" | "audio/matroska"
			) {
				return true;
			}
		}
		hint.extension.as_deref().is_some_and(|e| {
			let e = e.to_ascii_lowercase();
			matches!(e.as_str(), "mkv" | "mka" | "mks" | "mk3d")
		})
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::MkvDemuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::MkvMuxer::new(io, false)))
	}
}

/// WebM: the Matroska subset with the `webm` DocType. Demuxing goes through
/// [`MkvProbe`]; this probe only selects the restricted muxer.
pub struct WebmProbe;

impl Probe for WebmProbe {
	fn demuxer_matches(data: &[u8]) -> bool {
		MkvProbe::demuxer_matches(data)
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if matches!(mime, "video/webm" | "audio/webm") {
				return true;
			}
		}
		hint.extension.as_deref().is_some_and(|e| e.eq_ignore_ascii_case("webm"))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		MkvProbe::create_demuxer(io)
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::MkvMuxer::new(io, true)))
	}
}
