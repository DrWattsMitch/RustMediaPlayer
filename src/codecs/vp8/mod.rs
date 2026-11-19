//! VP8 video decoder (RFC 6386; libvpx `vp8/decoder`, `vp8/common`).
//!
//! Every feature of the format: key and inter frames, golden and alt-ref
//! references (with sign bias, copies and hidden alt-ref frames), 16x16,
//! 4x4 (B_PRED) and chroma intra prediction, split motion vectors, the
//! six-tap and bilinear (versions 1-3, full-pixel chroma for version 3)
//! interpolation filters, multiple token partitions, segmentation, the
//! normal and simple loop filters with reference/mode deltas, and
//! probability persistence. Output is bit-exact with libvpx / FFmpeg.
//! Spatial resampling (the key frame's scaling bits) is left to the player,
//! as in FFmpeg.
//!
//! Layout:
//! - [`range`]: the boolean entropy decoder;
//! - [`tables`]: default and update probabilities, quantizer tables;
//! - [`dsp`]: inverse transforms, intra predictors, interpolation and the
//!   loop filters;
//! - [`decoder`]: headers, modes, tokens, reconstruction, references and
//!   the [`crate::core::Decoder`] implementation.

pub mod decoder;
mod dsp;
pub mod range;
mod tables;

pub use decoder::{Picture, Vp8Decoder, Vp8DecoderState};
