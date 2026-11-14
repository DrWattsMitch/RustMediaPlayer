use crate::core;
use crate::{core::*, message::Result};

#[derive(Debug, Clone)]
pub struct Decoder {
	kind: StreamKind,
}

impl Decoder {
	pub fn new(kind: StreamKind) -> Self {
		Self { kind }
	}
}

impl core::Decoder for Decoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		match self.kind {
			StreamKind::Audio => super::audio_state(parameters),
			_ => super::video_state(parameters),
		}
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let frame = match self.kind {
			StreamKind::Audio => {
				let (sample_rate, channels, format, layout) = match state {
					State::Audio(audio) => (audio.sample_rate, audio.channels, audio.format, audio.layout),
					State::Video(_) => {
						(48_000, Channels::Stereo, SampleFormat::S16, SampleLayout::Interleaved)
					}
				};
				let mut frame = AudioFrame::new(packet.data, sample_rate, channels, format, layout);
				frame.pts = packet.pts;
				Frame::Audio(frame)
			}
			_ => {
				let (width, height, pixel) = match state {
					State::Video(video) => (video.width, video.height, video.pixel.clone()),
					State::Audio(_) => (0, 0, video::Pixel { depth: 8, format: video::PixelFormat::YUV420 }),
				};
				// The packet's bytes become the frame's planes: no copy.
				Frame::Video(VideoFrame::from_packed(
					packet.data,
					width,
					height,
					pixel,
					Keyframe::Key,
					packet.pts,
				))
			}
		};

		Ok(DecodeOut { value: vec![frame] })
	}

	fn flush(&self, _state: &mut State) -> Result<DecodeOut> {
		Ok(DecodeOut { value: Vec::new() })
	}
}
