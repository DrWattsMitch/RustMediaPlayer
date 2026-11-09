use crate::codecs::{aac, adpcm, h264, mp3, pcm, raw};
use crate::core::{CodecId, Decoder, Demuxer, Encoder, Muxer, Stream};

#[derive(Debug, Clone, Default)]
pub struct Hint {
	pub extension: Option<String>,
	pub mime_type: Option<String>,
	pub codec: Option<CodecId>,
}

pub trait CodecResolver {
	fn decoder_for(&self, stream: &Stream) -> Option<Box<dyn Decoder>>;
	fn encoder_for(&self, stream: &Stream) -> Option<Box<dyn Encoder>>;
}

pub trait ContainerResolver {
	fn demuxer_for(&self, hint: &Hint) -> Option<Box<dyn Demuxer>>;
	fn muxer_for(&self, hint: &Hint) -> Option<Box<dyn Muxer>>;
}

#[derive(Debug, Clone, Default)]
pub struct Resolver;

impl Resolver {
	pub fn new() -> Self {
		Self
	}

	/// Picks a decoder by codec id. Codecs without a dedicated decoder fall
	/// back to the raw pass-through decoder.
	pub fn decoder_for(&self, stream: &Stream) -> Option<Box<dyn Decoder>> {
		let id = stream.parameters.codec.id.as_str();
		if let Some(decoder) = pcm::PcmDecoder::for_id(id) {
			return Some(Box::new(decoder));
		}
		if let Some(decoder) = adpcm::AdpcmDecoder::for_id(id) {
			return Some(Box::new(decoder));
		}
		match id {
			"mp3" => Some(Box::new(mp3::Mp3Decoder::new())),
			"aac" => Some(Box::new(aac::AacDecoder::new())),
			"h264" => Some(Box::new(h264::H264Decoder::new())),
			_ => Some(Box::new(raw::decoder::Decoder::new(stream.kind))),
		}
	}

	/// Picks an encoder producing the stream's codec. PCM codecs get a real
	/// encoder (which converts from any sample format); everything else falls
	/// back to raw pass-through.
	pub fn encoder_for(&self, stream: &Stream) -> Option<Box<dyn Encoder>> {
		if let Some(encoder) = pcm::PcmEncoder::for_id(stream.id, &stream.parameters.codec.id) {
			return Some(Box::new(encoder));
		}
		Some(Box::new(raw::encoder::Encoder::new(stream.id, stream.kind)))
	}

	pub fn demuxer_for(&self, _hint: &Hint) -> Option<Box<dyn Demuxer>> {
		None
	}

	pub fn muxer_for(&self, _hint: &Hint) -> Option<Box<dyn Muxer>> {
		None
	}
}
