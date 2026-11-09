use crate::core::{Demuxer, Hint, Muxer};
use crate::message::Result;
use crate::{core::Probe, io::Io};

use super::header::MAGIC as Y4M_MAGIC;

pub struct Y4mProbe;

impl Probe for Y4mProbe {
	fn demuxer_matches(data: &[u8]) -> bool {
		data.starts_with(Y4M_MAGIC)
	}

	fn muxer_matches(hint: &Hint) -> bool {
		if let Some(mime) = hint.mime_type.as_ref().map(|x| x.trim()) {
			if mime == "video/x-yuv4mpegpipe" {
				return true;
			}
		}

		if hint.extension.as_ref().is_some_and(|e| e.eq_ignore_ascii_case("y4m")) {
			return true;
		}
		false
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(super::demuxer::Y4mDemuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(super::muxer::Y4mMuxer::new(io)))
	}
}
