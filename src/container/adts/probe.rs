use crate::codecs::aac::AdtsHeader;
use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::{Message, Result};

pub struct AdtsProbe;

/// Number of back-to-back ADTS frames starting at `start`.
pub(crate) fn frame_chain(data: &[u8], start: usize) -> usize {
	let mut count = 0;
	let mut at = start;
	while let Some(h) = data.get(at..).and_then(AdtsHeader::parse) {
		count += 1;
		at += h.frame_length;
	}
	count
}

/// Size of a leading ID3v2 tag, which some `.aac` files carry.
pub(crate) fn id3_size(data: &[u8]) -> usize {
	crate::container::mp3::id3::v2_tag_size(data).unwrap_or(0)
}

impl Probe for AdtsProbe {
	/// Like FFmpeg's `adts_aac_probe`: a run of consistent ADTS headers.
	fn demuxer_matches(data: &[u8]) -> bool {
		let start = id3_size(data);
		let end = data.len().min(start + 4096);
		(start..end).any(|i| data[i] == 0xFF && frame_chain(data, i) >= 3)
	}

	fn muxer_matches(_hint: &Hint) -> bool {
		false
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::AdtsDemuxer::new(io)?))
	}

	fn create_muxer(_io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Err(Message::Container("raw adts output is not supported"))
	}
}
