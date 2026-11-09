use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

pub struct FlacProbe;

impl Probe for FlacProbe {
	/// `"fLaC"` at the start, or right after an ID3v2 tag.
	fn demuxer_matches(data: &[u8]) -> bool {
		if data.starts_with(b"fLaC") {
			return true;
		}
		if data.len() >= 10 && &data[..3] == b"ID3" {
			let size = data[6..10].iter().fold(0usize, |acc, &b| (acc << 7) | usize::from(b & 0x7F));
			let at = 10 + size + if data[5] & 0x10 != 0 { 10 } else { 0 };
			return data.get(at..at + 4) == Some(b"fLaC");
		}
		false
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if matches!(mime, "audio/flac" | "audio/x-flac") {
				return true;
			}
		}
		hint.extension.as_deref().is_some_and(|e| e.eq_ignore_ascii_case("flac"))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::FlacDemuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::FlacMuxer::new(io)))
	}
}
