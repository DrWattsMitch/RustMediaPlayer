//! MP4 / M4A container (ISO/IEC 14496-12 and -14), plus the QuickTime
//! variations commonly found in `.mov`/`.m4a` files.
//!
//! See `docs/MP3_MP4.md` for the format walkthrough and
//! `docs/FRAGMENTED_MP4_DITHER.md` for fragmented output and `+faststart`.

pub mod boxes;
pub mod codec;
pub mod demuxer;
pub mod fragment;
pub mod metadata;
pub mod muxer;
pub mod options;
pub mod probe;
pub mod sample_table;
pub use probe::*;
