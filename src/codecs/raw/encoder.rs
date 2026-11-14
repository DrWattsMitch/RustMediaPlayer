use crate::core;
use crate::{core::*, message::Result};
#[derive(Debug, Clone)]
pub struct Encoder {
	stream_id: StreamId,
	kind: StreamKind,
}

impl Encoder {
	pub fn new(stream_id: StreamId, kind: StreamKind) -> Self {
		Self { stream_id, kind }
	}
}

impl core::Encoder for Encoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		match self.kind {
			StreamKind::Audio => super::audio_state(parameters),
			_ => super::video_state(parameters),
		}
	}

	fn encode(&self, _state: &mut State, frame: Frame) -> Result<EncodeOut> {
		let packet = match frame {
			Frame::Audio(frame) => Packet {
				stream_id: self.stream_id,
				data: frame.data,
				pts: frame.pts,
				dts: frame.pts,
				duration: None,
				keyframe: true,
			},
			Frame::Video(frame) => Packet {
				stream_id: self.stream_id,
				pts: frame.pts,
				dts: frame.pts,
				duration: None,
				keyframe: frame.keyframe == Keyframe::Key,
				// Free for frames built from packed bytes; otherwise the
				// visible rows are packed once (`av_image_copy_to_buffer`).
				data: frame.into_packed(),
			},
		};

		Ok(EncodeOut { value: vec![packet] })
	}

	fn flush(&self, _state: &mut State) -> Result<EncodeOut> {
		Ok(EncodeOut { value: Vec::new() })
	}
}
