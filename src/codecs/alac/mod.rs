//! ALAC (Apple Lossless Audio Codec) decoder.
//!
//! A port of FFmpeg's `libavcodec/alac.c` / `alacdsp.c` (itself following
//! Apple's open-source reference decoder): adaptive Golomb-Rice residuals,
//! the sign-adaptive LPC predictor, stereo decorrelation and the "extra
//! bits" (low bits stored verbatim for 20/24/32-bit audio). Output is
//! bit-exact with FFmpeg: interleaved S16 for 16-bit streams, S32 (samples
//! in the high bits) for 20, 24 and 32-bit streams, with channels in
//! FFmpeg's order (ALAC codes the centre first).
//!
//! The configuration (`ALACSpecificConfig`) comes from the stream's
//! extradata in any of its usual wrappings: FFmpeg's 36-byte `alac` atom
//! (MP4), the bare 24-byte config (Matroska `A_ALAC`) or the config after a
//! 4-byte version/flags field (the `alac` box body).

pub mod decoder;

pub use decoder::{AlacConfig, AlacDecoder, AlacDecoderState};
