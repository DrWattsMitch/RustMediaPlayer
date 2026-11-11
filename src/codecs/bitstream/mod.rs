//! Bitstream primitives shared by the AAC and H.264 decoders and encoders.

mod reader;
mod vlc;
mod writer;

pub use reader::BitReader;
pub use vlc::{Code, Vlc};
pub use writer::{BitWriter, se_bits, ue_bits};
