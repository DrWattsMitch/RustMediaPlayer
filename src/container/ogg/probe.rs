use crate::core::{Demuxer, Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::{Message, Result};

pub struct OggProbe;

impl Probe for OggProbe {
	/// `"OggS"`, stream structure version 0, with the BOS flag set.
	fn demuxer_matches(data: &[u8]) -> bool {
		data.len() >= 6 && data.starts_with(b"OggS") && data[4] == 0 && data[5] & 0x02 != 0
	}

	/// RMP has no Ogg muxer yet.
	fn muxer_matches(_hint: &Hint) -> bool {
		false
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::OggDemuxer::new(io)?))
	}

	fn create_muxer(_io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Err(Message::Container("ogg muxing is not supported"))
	}
}
