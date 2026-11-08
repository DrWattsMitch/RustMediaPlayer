//! Signal-processing building blocks shared by audio decoders.

// Transform, prediction and table code indexes several parallel arrays by
// the same position, mirroring the equations of the specification.
#![allow(clippy::needless_range_loop)]

pub mod fft;
pub mod mdct;

pub use fft::{Complex, Fft};
pub use mdct::Imdct;
