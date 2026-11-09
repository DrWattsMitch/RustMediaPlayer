use super::ebml;
use crate::Message;
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

	fn muxer_matches(_hint: &Hint) -> bool {
		false
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::MkvDemuxer::new(io)?))
	}

	fn create_muxer(_io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Err(Message::Container("matroska muxing is not supported"))
	}
}
