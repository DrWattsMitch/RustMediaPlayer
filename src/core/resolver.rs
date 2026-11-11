use crate::codecs::{aac, adpcm, flac, h264, mp3, opus, pcm, raw};
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

	/// `true` when a real decoder exists for `codec` (as opposed to the raw
	/// pass-through that [`Resolver::decoder_for`] falls back to). Raw video
	/// codec IDs (`rawvideo` or a pixel format name) count as decodable.
	pub fn has_decoder(&self, codec: &str) -> bool {
		pcm::PcmCodec::from_id(codec).is_some()
			|| adpcm::AdpcmDecoder::for_id(codec).is_some()
			|| matches!(codec, "mp3" | "aac" | "h264" | "flac" | "opus" | "rawvideo")
			|| crate::core::Pixel::from_name(codec).is_some()
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
			"flac" => Some(Box::new(flac::FlacDecoder::new())),
			"opus" => Some(Box::new(opus::OpusDecoder::new())),
			_ => Some(Box::new(raw::decoder::Decoder::new(stream.kind))),
		}
	}

	/// Picks an encoder producing the stream's codec. PCM codecs, FLAC, AAC
	/// and H.264 (AAC and H.264 at `parameters.bitrate`, if set) get a real
	/// encoder; everything else falls back to raw pass-through.
	pub fn encoder_for(&self, stream: &Stream) -> Option<Box<dyn Encoder>> {
		if let Some(encoder) = pcm::PcmEncoder::for_id(stream.id, &stream.parameters.codec.id) {
			return Some(Box::new(encoder));
		}
		match stream.parameters.codec.id.as_str() {
			"flac" => return Some(Box::new(flac::FlacEncoder::new(stream.id))),
			"aac" => {
				let bitrate = stream.parameters.bitrate.filter(|&b| b > 0);
				return Some(Box::new(aac::AacEncoder::new(stream.id).with_bitrate(bitrate)));
			}
			"h264" => {
				let bitrate = stream.parameters.bitrate.filter(|&b| b > 0);
				let encoder = h264::H264Encoder::new(stream.id)
					.with_rate(None, bitrate)
					.with_time_base(stream.time_base);
				return Some(Box::new(encoder));
			}
			_ => {}
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
