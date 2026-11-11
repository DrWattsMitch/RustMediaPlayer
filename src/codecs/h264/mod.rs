//! H.264 / AVC (ISO/IEC 14496-10) video decoder and encoder.
//!
//! * [`nal`] — `avcC`, length-prefixed and Annex B framing.
//! * [`params`] — SPS/PPS, scaling matrices, dequantisation tables.
//! * [`slice`] — slice headers.
//! * `cabac` / `cavlc` — entropy decoding.
//! * `mb` — macroblock layer parsing and context derivation.
//! * `inter` — motion vector prediction, direct modes, motion compensation.
//! * `recon` — transforms, intra prediction, interpolation, weighting.
//! * `deblock` — loop filter.
//! * [`dpb`] — POC, reference lists and marking, output order.
//! * [`decoder`] — picture lifecycle and the [`crate::core::Decoder`] impl.
//! * [`encoder`] — Constrained Baseline encoder (CAVLC, I/P, quarter-sample
//!   motion), reconstructing with the decoder's own `recon`/`deblock`.
//!
//! Supported: Constrained Baseline, Main and High profiles at 8-bit 4:2:0,
//! progressive frames — CAVLC and CABAC, I/P/B slices, 8x8 transform,
//! scaling matrices, multiple slices, multiple and long-term references,
//! reference list modification, MMCO, POC types 0-2, explicit and implicit
//! weighted prediction, spatial and temporal direct, I_PCM.
//!
//! Not supported (reported as errors): interlaced coding (field pictures,
//! MBAFF), 4:2:2/4:4:4, bit depths above 8, lossless, FMO/ASO, SP/SI
//! slices, data partitioning, SVC/MVC extensions.

// Transform, prediction and table code indexes several parallel arrays by
// the same position, mirroring the equations of the specification.
#![allow(clippy::needless_range_loop)]

mod cabac;
mod cavlc;
mod consts;
mod deblock;
pub mod decode;
pub mod decoder;
pub mod dpb;
pub mod encoder;
mod inter;
mod mb;
pub mod nal;
pub mod params;
mod recon;
pub mod slice;
mod tables;

pub use decoder::{DecodedPicture, H264Decoder, H264DecoderState};
pub use encoder::{H264Encoder, H264EncoderState};
