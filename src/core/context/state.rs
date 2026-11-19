use crate::core::{Channels, SampleFormat, SampleLayout, Time, VideoFrame, video};

#[derive(Debug, Clone)]
pub struct CodecState {
	pub extradata: Vec<u8>,
	pub pts: Option<Time>,
	pub dts: Option<Time>,
	pub last_pts: Option<Time>,
}

#[derive(Debug, Clone)]
pub struct VideoState {
	pub width: u32,
	pub height: u32,
	pub pixel: video::Pixel,
	pub reorder: Vec<VideoFrame>,
	pub has_keyframe: bool,

	pub codec_state: CodecState,
	pub private: CodecPrivate,
}

/// Codec-private decoder/encoder state (FFmpeg's `priv_data`).
///
/// Stateless codecs (raw, PCM) use [`CodecPrivate::None`]. Codecs that must
/// carry history between packets (e.g. the MP3 bit reservoir and filterbank
/// overlap) get a typed variant here instead of hiding bytes in `buffer`, so
/// the state stays type-checked and `Clone`.
#[derive(Debug, Clone, Default)]
pub enum CodecPrivate {
	#[default]
	None,
	Mp3(Box<crate::codecs::mp3::Mp3DecoderState>),
	Aac(Box<crate::codecs::aac::AacDecoderState>),
	H264(Box<crate::codecs::h264::H264DecoderState>),
	Flac(Box<crate::codecs::flac::FlacDecoderState>),
	Alac(Box<crate::codecs::alac::AlacDecoderState>),
	FlacEncoder(Box<crate::codecs::flac::encoder::FlacEncoderState>),
	AacEncoder(Box<crate::codecs::aac::AacEncoderState>),
	H264Encoder(Box<crate::codecs::h264::H264EncoderState>),
	Opus(Box<crate::codecs::opus::OpusDecoderState>),
	Vorbis(Box<crate::codecs::vorbis::VorbisDecoderState>),
	Vp8(Box<crate::codecs::vp8::Vp8DecoderState>),
	Vp9(Box<crate::codecs::vp9::Vp9DecoderState>),
	Ac3(Box<crate::codecs::ac3::Ac3DecoderState>),
	Hevc(Box<crate::codecs::hevc::HevcDecoderState>),
	#[cfg(feature = "av1")]
	Av1(Box<crate::codecs::av1::Av1DecoderState>),
}

#[derive(Debug, Clone)]
pub struct AudioState {
	pub sample_rate: u32,
	pub channels: Channels,
	pub format: SampleFormat,
	pub layout: SampleLayout,
	pub frame_size: usize,
	pub codec_state: CodecState,

	pub samples_consumed: u64,
	pub samples_produced: u64,

	pub buffer: Vec<u8>,
	pub private: CodecPrivate,
}

/// State of a text subtitle decoder: the codec header (ASS `[Script Info]`
/// and styles, WebVTT header) and the last cue's timing.
#[derive(Debug, Clone)]
pub struct SubtitleState {
	pub codec_state: CodecState,
}

#[derive(Debug, Clone)]
pub enum State {
	Audio(AudioState),
	Video(VideoState),
	Subtitle(SubtitleState),
}
