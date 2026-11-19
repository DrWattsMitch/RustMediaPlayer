pub mod audio;
pub mod buffer;
pub mod subtitle;
pub mod video;

pub use audio::{AudioFrame, AudioParams, Channels, SampleFormat, SampleLayout};
pub use buffer::{BufferPool, BufferSource, FrameBuffer, PoolStats, Pooled, VecPool};
pub use subtitle::{AssEvent, SubtitleFormat, SubtitleFrame};
pub use video::{Keyframe, MAX_PLANES, Pixel, PixelFormat, PlaneRef, VideoFrame};

// A video frame is a few plane references (~250 bytes), not its samples:
// boxing it would cost an allocation per frame to save a short memcpy.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Frame {
	Audio(AudioFrame),
	Video(VideoFrame),
	/// A timed text cue (see [`subtitle`]).
	Subtitle(SubtitleFrame),
}
