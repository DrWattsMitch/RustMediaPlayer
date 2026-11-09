//! Native FLAC container (`.flac`).
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md`.

pub mod demuxer;
pub mod muxer;
pub mod probe;
pub use probe::*;
