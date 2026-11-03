use crate::Message;
use crate::container::mp3;
use crate::container::mp4;
use crate::container::riff;
use crate::container::y4m;
use crate::core::Demuxer;
use crate::core::Probe;
use crate::io::Io;
use crate::message::Result;

pub fn detect(data: &[u8], io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
	if riff::RiffProbe::demuxer_matches(data) {
		return riff::RiffProbe::create_demuxer(io);
	}

	if y4m::Y4mProbe::demuxer_matches(data) {
		return y4m::Y4mProbe::create_demuxer(io);
	}

	if mp4::Mp4Probe::demuxer_matches(data) {
		return mp4::Mp4Probe::create_demuxer(io);
	}

	// MPEG audio has no magic number (only an 11-bit sync word), so it is
	// probed last, after every format with a real signature.
	if mp3::Mp3Probe::demuxer_matches(data) {
		return mp3::Mp3Probe::create_demuxer(io);
	}

	Err(Message::Container("unsupported container format"))
}
