//! MP3 container (MPEG audio elementary stream with ID3 tags).
//!
//! See `docs/MP3_MP4.md` for the format walkthrough.

pub mod demuxer;
pub mod id3;
pub mod muxer;
pub mod probe;
pub mod xing;
pub use probe::*;
