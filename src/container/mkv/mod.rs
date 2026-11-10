//! Matroska (`.mkv`, `.mka`, `.mks`) and WebM (`.webm`) demuxer and muxer.
//!
//! * [`ebml`] — EBML variable-length integers, element headers and values.
//! * [`tracks`] — `TrackEntry` parsing, codec ID mapping, extradata.
//! * [`demuxer`] — level-1 loading, cluster/block streaming, lacing, seeking.
//! * [`muxer`] — header, clusters, cues, tags, attachments, back-patching.
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md` (demuxing) and
//! `docs/MKV_OGG_MUXING_CLI.md` (muxing) for the walkthroughs.

pub mod demuxer;
pub mod ebml;
pub mod muxer;
pub mod probe;
pub mod tracks;
pub use probe::*;
