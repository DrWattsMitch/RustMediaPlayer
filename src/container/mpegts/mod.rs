//! MPEG transport stream (`.ts`, `.m2ts`, `.mts`): the container of
//! broadcast and of most HLS streaming (ISO/IEC 13818-1).
//!
//! * [`packet`] — transport packets, packet-size detection, PES headers.
//! * [`psi`] — PAT, PMT and SDT sections.
//! * [`es`] — stream types, audio framing, video keyframes and parameters.
//! * [`demuxer`] — [`TsDemuxer`].

pub mod demuxer;
pub mod es;
pub mod muxer;
pub mod packet;
mod probe;
pub mod psi;

pub use demuxer::TsDemuxer;
pub use muxer::TsMuxer;
pub use probe::*;
