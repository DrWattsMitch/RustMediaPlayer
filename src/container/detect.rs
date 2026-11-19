use crate::Message;
use crate::container::ac3;
use crate::container::adts;
use crate::container::flac;
use crate::container::mkv;
use crate::container::mp3;
use crate::container::mp4;
use crate::container::mpegts;
use crate::container::ogg;
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

	// Before MP3: an ID3v2-tagged FLAC file starts like an MP3.
	if flac::FlacProbe::demuxer_matches(data) {
		return flac::FlacProbe::create_demuxer(io);
	}

	if mkv::MkvProbe::demuxer_matches(data) {
		return mkv::MkvProbe::create_demuxer(io);
	}

	if ogg::OggProbe::demuxer_matches(data) {
		return ogg::OggProbe::create_demuxer(io);
	}

	if mp4::Mp4Probe::demuxer_matches(data) {
		return mp4::Mp4Probe::create_demuxer(io);
	}

	// A transport stream has no magic number, but sync bytes every 188
	// (192, 204) bytes are distinctive; checked before the audio formats
	// whose sync words a TS payload could contain.
	if mpegts::TsProbe::demuxer_matches(data) {
		return mpegts::TsProbe::create_demuxer(io);
	}

	if crate::container::webvtt::looks_like_webvtt(data) {
		return Ok(Box::new(crate::container::webvtt::WebVttDemuxer::new(io)?));
	}

	// Text subtitles: a `[Script Info]` section, or SubRip's cue number and
	// `-->` timing line (no binary format starts with either).
	if crate::container::ass::AssProbe::demuxer_matches(data) {
		return crate::container::ass::AssProbe::create_demuxer(io);
	}

	if crate::container::srt::SrtProbe::demuxer_matches(data) {
		return crate::container::srt::SrtProbe::create_demuxer(io);
	}

	// AC-3's 16-bit sync word is stronger than MPEG audio's, and the probe
	// demands a run of consistent syncframes.
	if ac3::Ac3Probe::demuxer_matches(data) {
		return ac3::Ac3Probe::create_demuxer(io);
	}

	// ADTS shares MPEG audio's 0xFFF sync; its probe needs three
	// consistent headers, and MP3 layer bits never match ADTS's layer 0.
	if adts::AdtsProbe::demuxer_matches(data) {
		return adts::AdtsProbe::create_demuxer(io);
	}

	// MPEG audio has no magic number (only an 11-bit sync word), so it is
	// probed last, after every format with a real signature.
	if mp3::Mp3Probe::demuxer_matches(data) {
		return mp3::Mp3Probe::create_demuxer(io);
	}

	Err(Message::Container("unsupported container format"))
}
