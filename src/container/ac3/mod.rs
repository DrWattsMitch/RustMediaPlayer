//! Raw AC-3 / E-AC-3 elementary streams (`.ac3`, `.eac3`, `.ec3`; FFmpeg
//! `ac3dec.c` in libavformat). Each packet is one access unit: an
//! independent syncframe plus any dependent substream frames after it.

pub mod demuxer;
pub mod probe;
pub use probe::*;
