//! FLAC (Free Lossless Audio Codec).
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md`.

pub mod decoder;
pub mod encoder;
pub mod frame_header;
pub mod lpc;
pub mod md5;
pub mod streaminfo;
pub use decoder::{FlacDecoder, FlacDecoderState};
pub use encoder::FlacEncoder;
pub use frame_header::FrameHeader;
pub use streaminfo::StreamInfo;
