//! Vorbis I audio (Xiph.Org's "Vorbis I specification"; FFmpeg's
//! `libavcodec/vorbisdec.c`, libvorbis' `lib/`).
//!
//! - [`parser`]: packet durations without decoding, for containers (Ogg
//!   granule positions, Matroska blocks).
//! - [`setup`]: the three header packets (identification, comment, setup:
//!   codebooks, floors, residues, mappings, modes).
//! - [`decoder`]: audio packets → planar `F32` PCM.
//!
//! See `docs/VORBIS_VP9.md`.

mod bits;
pub mod codebook;
pub mod decoder;
mod floor;
pub mod parser;
mod residue;
pub mod setup;

pub use decoder::{VorbisDecoder, VorbisDecoderState};
pub use parser::VorbisParser;

/// Vorbis → FFmpeg channel order for 1 to 8 channels
/// (`ff_vorbis_channel_layout_offsets`): output channel `i` is coded
/// channel `CHANNEL_ORDER[n - 1][i]`. Opus mapping family 1 uses the same
/// order.
pub(crate) const CHANNEL_ORDER: [&[usize]; 8] = [
	&[0],
	&[0, 1],
	&[0, 2, 1],
	&[0, 1, 2, 3],
	&[0, 2, 1, 3, 4],
	&[0, 2, 1, 5, 3, 4],
	&[0, 2, 1, 6, 5, 3, 4],
	&[0, 2, 1, 7, 5, 6, 3, 4],
];

/// `ilog` of the specification: the position of the highest set bit
/// (`ilog(0) = 0`, `ilog(1) = 1`, `ilog(7) = 3`).
pub(crate) fn ilog(v: u32) -> u32 {
	u32::BITS - v.leading_zeros()
}
