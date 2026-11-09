//! Ogg container (`.ogg`, `.opus`, `.oga`): demuxing of Opus, FLAC and
//! Vorbis streams.
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md`.

pub mod demuxer;
pub mod probe;
pub use probe::*;
