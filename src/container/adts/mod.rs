//! Raw AAC in ADTS framing (`.aac`; FFmpeg `aacdec.c` in libavformat).
//! Each packet is one ADTS frame, header included; the AAC decoder reads
//! the configuration from the headers.

pub mod demuxer;
pub mod probe;
pub use probe::*;
