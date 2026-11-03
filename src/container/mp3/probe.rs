use super::id3;
use crate::codecs::mp3::FrameHeader;
use crate::codecs::mp3::header::HEADER_SIZE;
use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

/// Consecutive matching frames required to claim a stream without ID3 tag.
const REQUIRED_FRAMES: usize = 3;

pub struct Mp3Probe;

/// Length of the chain of compatible frames starting at `start`, and whether
/// the chain ran off the end of `data` (i.e. it might continue).
fn frame_chain(data: &[u8], start: usize) -> (usize, bool) {
	let Some(first) = FrameHeader::parse(&data[start..]) else { return (0, false) };
	let mut count = 0;
	let mut at = start;
	loop {
		match data.get(at..).and_then(FrameHeader::parse) {
			Some(h) if h.is_compatible(&first) => {
				count += 1;
				at += h.frame_size;
			}
			_ => return (count, at + HEADER_SIZE > data.len()),
		}
	}
}

impl Probe for Mp3Probe {
	/// Like FFmpeg's `mp3_read_probe`: skip ID3v2, then require a run of
	/// consistent frame headers. Since MPEG audio has only an 11-bit sync
	/// word, this is checked after the formats with real magic numbers.
	fn demuxer_matches(data: &[u8]) -> bool {
		let mut start = 0;
		while let Some(size) = data.get(start..).and_then(id3::v2_tag_size) {
			start += size;
			if start >= data.len() {
				// Tag extends past the probe window; ID3v2 is an MP3 tag format.
				return true;
			}
		}
		let tagged = start > 0;
		let end = data.len().saturating_sub(HEADER_SIZE - 1);
		for i in start..end {
			if data[i] != 0xFF {
				continue;
			}
			let (count, open_ended) = frame_chain(data, i);
			if count >= REQUIRED_FRAMES || (count >= 1 && open_ended && (tagged || count >= 2)) {
				return true;
			}
			// Without a tag, the sync must be near the start of the data.
			if !tagged && i > 2048 {
				return false;
			}
		}
		false
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_deref().map(str::trim) {
			if mime == "audio/mpeg" || mime == "audio/mp3" {
				return true;
			}
		}
		hint.extension.as_deref().is_some_and(|e| e.eq_ignore_ascii_case("mp3"))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::Mp3Demuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::Mp3Muxer::new(io)))
	}
}
