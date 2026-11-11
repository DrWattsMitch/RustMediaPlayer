//! The processing layer between decoders and encoders: what FFmpeg's
//! `libswresample` and `libswscale` do.
//!
//! * [`audio`] — sample format, layout, channel remix, sample rate.
//! * [`dither`] — dither and noise shaping when audio is rounded to integers.
//! * [`video`] — pixel format, colour matrix and range, scaling.
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md`.

pub mod audio;
pub mod dither;
pub mod video;

pub use audio::{AudioConverter, AudioSpec};
pub use dither::{Dither, DitherMethod};
pub use video::{ColorMatrix, Range, ScaleFilter, VideoConverter, VideoOptions, VideoSpec};
