//! The processing layer between decoders and encoders: what FFmpeg's
//! `libswresample` and `libswscale` do.
//!
//! * [`audio`] — sample format, layout, channel remix, sample rate.
//! * [`video`] — pixel format, colour matrix and range, scaling.
//!
//! See `docs/MKV_FLAC_OPUS_PROCESSING.md`.

pub mod audio;
pub mod video;

pub use audio::{AudioConverter, AudioSpec};
pub use video::{ColorMatrix, Range, ScaleFilter, VideoConverter, VideoOptions, VideoSpec};
