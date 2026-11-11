//! AAC (MPEG-4 Audio, ISO/IEC 14496-3) decoder and AAC-LC encoder.
//!
//! * [`config`] — AudioSpecificConfig, ADTS headers, program config.
//! * [`ics`] — individual channel stream: sections, scalefactors, Huffman
//!   spectral data, pulses, TNS, inverse quantisation, noise substitution.
//! * [`decoder`] — element loop, M/S and intensity stereo, IMDCT filterbank.
//! * [`encoder`] — AAC-LC encoder: block switching, MDCT, psychoacoustic
//!   thresholds, M/S, rate loop, Huffman sectioning.
//!
//! Supported: AAC-LC (and the LC core of HE-AAC v1/v2 streams), mono to 7.1,
//! sine and KBD windows, all window sequences, PNS, M/S, intensity stereo,
//! TNS, pulses, ADTS framing. Output is interleaved S16.
//!
//! Not supported: SBR/PS reconstruction (HE-AAC decodes at the core rate),
//! AAC Main prediction, LTP, SSR gain control, coupling channels, 960-sample
//! frames.

// Transform, prediction and table code indexes several parallel arrays by
// the same position, mirroring the equations of the specification.
#![allow(clippy::needless_range_loop)]

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod ics;
mod tables;

pub use config::{AacConfig, AdtsHeader};
pub use decoder::{AacDecoder, AacDecoderState, DecodedFrame};
pub use encoder::{AacEncoder, AacEncoderState};
