use crate::Message;
use crate::container::{flac, mkv, mp3, mp4, ogg, riff, y4m};
use crate::core::{Hint, Muxer, Probe};
use crate::io::Io;
use crate::message::Result;

pub fn select_muxer(hint: Hint, io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
	if riff::RiffProbe::muxer_matches(&hint) {
		return riff::RiffProbe::create_muxer(io);
	}

	if y4m::Y4mProbe::muxer_matches(&hint) {
		return y4m::Y4mProbe::create_muxer(io);
	}

	if mp4::Mp4Probe::muxer_matches(&hint) {
		return mp4::Mp4Probe::create_muxer(io);
	}

	if flac::FlacProbe::muxer_matches(&hint) {
		return flac::FlacProbe::create_muxer(io);
	}

	if mkv::MkvProbe::muxer_matches(&hint) {
		return mkv::MkvProbe::create_muxer(io);
	}

	if mkv::WebmProbe::muxer_matches(&hint) {
		return mkv::WebmProbe::create_muxer(io);
	}

	if ogg::OpusProbe::muxer_matches(&hint) {
		return ogg::OpusProbe::create_muxer(io);
	}

	if ogg::OggProbe::muxer_matches(&hint) {
		return ogg::OggProbe::create_muxer(io);
	}

	if mp3::Mp3Probe::muxer_matches(&hint) {
		return mp3::Mp3Probe::create_muxer(io);
	}

	Err(Message::Container("unsupported container format"))
}
