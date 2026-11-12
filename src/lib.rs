#![allow(clippy::module_inception, clippy::collapsible_if)]
// `unsafe` lives only in the audited `simd` module (see its docs).
#![deny(unsafe_code)]
pub mod codecs;
pub mod container;
pub mod core;
pub mod io;
mod message;
pub mod process;
#[allow(unsafe_code)]
pub mod simd;
pub mod transcode;
pub use message::*;
