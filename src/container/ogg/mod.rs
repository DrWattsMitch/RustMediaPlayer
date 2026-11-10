//! Ogg container (`.ogg`, `.opus`, `.oga`): demuxing of Opus, FLAC and
//! Vorbis streams, and muxing of the same three mappings.
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md` (demuxing) and
//! `docs/MKV_OGG_MUXING_CLI.md` (muxing).

pub mod demuxer;
pub mod muxer;
pub mod probe;
pub use probe::*;
