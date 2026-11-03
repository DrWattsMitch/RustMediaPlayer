//! MPEG audio (MP3) codec support.
//!
//! * [`header`] — frame header parsing, shared with the MP3 container.
//! * [`decoder`] — Layer III decoder producing interleaved S16 PCM.
//!
//! Encoding is not implemented; MP3 output is supported by remuxing
//! already-encoded frames (see `container::mp3`).

mod bits;
pub mod decoder;
pub mod header;
mod layer3;
mod synth;
mod tables;

pub use decoder::{DecodedFrame, Mp3Decoder, Mp3DecoderState};
pub use header::{ChannelMode, FrameHeader, MpegVersion};
