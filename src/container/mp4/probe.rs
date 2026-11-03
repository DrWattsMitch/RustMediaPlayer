use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

/// Top-level box types that can start an ISO BMFF / QuickTime file.
const LEADING_BOXES: [&[u8; 4]; 8] =
	[b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"styp"];

pub struct Mp4Probe;

impl Probe for Mp4Probe {
	/// Checks that the file starts with a plausible top-level box, like the
	/// first stage of FFmpeg's `mov_probe`.
	fn demuxer_matches(data: &[u8]) -> bool {
		let Some(header) = data.get(..8) else { return false };
		let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
		let kind = &header[4..8];
		// size 0 = "to end of file", 1 = 64-bit size follows, else >= 8.
		let plausible_size = size == 0 || size == 1 || size >= 8;
		plausible_size && LEADING_BOXES.iter().any(|b| &b[..] == kind)
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if matches!(mime, "video/mp4" | "audio/mp4" | "audio/x-m4a" | "video/x-m4v") {
				return true;
			}
		}
		hint
			.extension
			.as_deref()
			.is_some_and(|e| ["mp4", "m4a", "m4v", "m4b"].iter().any(|x| e.eq_ignore_ascii_case(x)))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::Mp4Demuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::Mp4Muxer::new(io)))
	}
}
