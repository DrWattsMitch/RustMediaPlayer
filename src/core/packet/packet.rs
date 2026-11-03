use crate::core::stream::StreamId;
use crate::core::time::Time;

#[derive(Debug, Clone)]
pub struct Packet {
	pub stream_id: StreamId,
	pub data: Vec<u8>,
	pub pts: Option<Time>,
	pub dts: Option<Time>,
	pub duration: Option<Time>,
	/// `true` when the packet can be decoded without any previous packet
	/// (FFmpeg's `AV_PKT_FLAG_KEY`). Audio packets are almost always keyframes.
	/// Containers with sync-sample tables (MP4 `stss`) use this when muxing.
	pub keyframe: bool,
}

impl Packet {
	pub fn new(stream_id: StreamId, data: Vec<u8>) -> Self {
		Self { stream_id, data, pts: None, dts: None, duration: None, keyframe: true }
	}
}
