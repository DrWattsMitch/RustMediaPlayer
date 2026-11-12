//! Signal-processing building blocks shared by the audio decoders and the
//! AAC encoder.

// Transform, prediction and table code indexes several parallel arrays by
// the same position, mirroring the equations of the specification.
#![allow(clippy::needless_range_loop)]

pub mod fft;
pub mod mdct;
mod simd;

pub use fft::{Complex, Fft};
pub use mdct::{Imdct, Mdct};
