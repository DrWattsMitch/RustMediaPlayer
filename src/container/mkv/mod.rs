//! Matroska (`.mkv`, `.mka`, `.mks`) and WebM (`.webm`) demuxer.
//!
//! * [`ebml`] — EBML variable-length integers, element headers and values.
//! * [`tracks`] — `TrackEntry` parsing, codec ID mapping, extradata.
//! * [`demuxer`] — level-1 loading, cluster/block streaming, lacing, seeking.
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md` for the walkthrough.

pub mod demuxer;
pub mod ebml;
pub mod probe;
pub mod tracks;
pub use probe::*;
