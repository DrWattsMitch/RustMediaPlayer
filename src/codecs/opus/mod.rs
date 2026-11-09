//! Opus (RFC 6716, RFC 7845).
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md`.

pub mod celt;
pub mod decoder;
pub mod header;
pub mod mdct;
pub mod range;
pub mod silk;
mod tables;
pub use decoder::{OpusDecoder, OpusDecoderState};
pub use header::{OpusHead, packet_samples, pre_skip};
