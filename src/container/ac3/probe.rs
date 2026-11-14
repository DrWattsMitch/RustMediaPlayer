use crate::codecs::ac3::Header;
use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::{Message, Result};

/// Consecutive syncframes required to claim a stream.
const REQUIRED_FRAMES: usize = 3;

pub struct Ac3Probe;

/// Number of back-to-back syncframes starting at `start`.
pub(crate) fn frame_chain(data: &[u8], start: usize) -> usize {
	let mut count = 0;
	let mut at = start;
	while let Some((size, _)) = data.get(at..).and_then(Header::peek_size) {
		if size == 0 {
			break;
		}
		count += 1;
		at += size;
	}
	count
}

impl Probe for Ac3Probe {
	/// Like FFmpeg's `ac3_eac3_probe`: a run of valid syncframes near the
	/// start of the data (the last frame may run past the probe window).
	fn demuxer_matches(data: &[u8]) -> bool {
		let end = data.len().min(4096);
		(0..end).any(|i| {
			data[i] == 0x0B && data.get(i + 1) == Some(&0x77) && {
				let n = frame_chain(data, i);
				n >= REQUIRED_FRAMES
					|| (n >= 1 && i == 0 && Header::peek_size(data).is_some_and(|(s, _)| s > data.len() / 2))
			}
		})
	}

	fn muxer_matches(_hint: &Hint) -> bool {
		false
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::Ac3Demuxer::new(io)?))
	}

	fn create_muxer(_io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Err(Message::Container("raw ac-3 output is not supported"))
	}
}
