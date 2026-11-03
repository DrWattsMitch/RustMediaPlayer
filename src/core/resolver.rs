use crate::codecs::{mp3, raw};
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
		match stream.parameters.codec.id.as_str() {
			"mp3" => Some(Box::new(mp3::Mp3Decoder::new())),
			_ => Some(Box::new(raw::decoder::Decoder::new(stream.kind))),
		}
	}

	pub fn encoder_for(&self, stream: &Stream) -> Option<Box<dyn Encoder>> {
		Some(Box::new(raw::encoder::Encoder::new(stream.id, stream.kind)))
	}

	pub fn demuxer_for(&self, _hint: &Hint) -> Option<Box<dyn Demuxer>> {
		None
	}

	pub fn muxer_for(&self, _hint: &Hint) -> Option<Box<dyn Muxer>> {
		None
	}
}
