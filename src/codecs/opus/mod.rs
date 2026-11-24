//! Opus (RFC 6716, RFC 7845).
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md` (decoder) and
//! `docs/H264_HIGH_OPUS_ENCODING.md` (the CELT encoder).

pub mod celt;
pub mod decoder;
pub mod encoder;
pub mod header;
pub mod mdct;
mod pitch;
pub mod range;
pub mod silk;
mod tables;
pub use decoder::{OpusDecoder, OpusDecoderState};
pub use encoder::{OpusEncoder, OpusEncoderState};
pub use header::{OpusHead, packet_samples, pre_skip};
