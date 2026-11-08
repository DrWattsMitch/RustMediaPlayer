//! Bitstream primitives shared by the AAC and H.264 decoders.

mod reader;
mod vlc;

pub use reader::BitReader;
pub use vlc::{Code, Vlc};
