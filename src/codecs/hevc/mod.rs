//! HEVC / H.265 (ITU-T H.265) video decoder, written natively in safe Rust
//! after FFmpeg's `hevcdec`.
//!
//! * [`ps`] — VPS/SPS/PPS, scaling lists, tile and z-scan layout.
//! * [`slice`] — slice segment headers, RPS, weighted prediction tables.
//! * `cabac` — arithmetic decoding engine and context initialisation.
//! * `ctu` — coding tree, residual coding, reconstruction, QP derivation,
//!   deblocking boundary strengths; `mvs` — merge / AMVP / temporal MVs.
//! * `intra`, `mc`, `transform` — prediction and inverse transforms.
//! * `filter` — deblocking and SAO, CTB row by CTB row.
//! * [`decoder`] — POC, reference picture sets, lists, DPB and output,
//!   and the picture jobs that frame threads run.
//! * `picture` — planes, motion, and `RefPic`, through which pictures still
//!   being decoded on another thread are read (row progress).
//! * `wpp` — WPP slice segments decoded row by row from their entry
//!   points, on several threads.
//!
//! Supported: Main, Main 10 and the format range extensions profiles up to
//! 12 bits — 4:0:0, 4:2:0, 4:2:2 and 4:4:4 (Main 12, Main 4:2:2 10/12,
//! Main 4:4:4 8/10/12, Monochrome 8/12 and their Intra variants) with the
//! range extension tools (cross-component prediction, implicit and explicit
//! RDPCM, extended transform skip with rotation and its own contexts,
//! persistent Rice adaptation, CU chroma QP offsets, intra smoothing
//! disabled, high-precision weighted prediction offsets, SAO offset
//! scaling) — plus tiles, WPP, dependent slices, AMP, SAO, scaling lists,
//! PCM, lossless (transquant bypass), weighted prediction and long-term
//! references.
//! Not supported: 16-bit profiles (extended precision, CABAC bypass
//! alignment), separate colour planes, SCC and multi-layer extensions.

#![allow(clippy::needless_range_loop)]

mod cabac;
mod ctu;
pub mod decoder;
mod filter;
mod intra;
mod mc;
mod picture;
pub mod ps;
mod simd;
pub mod slice;
mod tables;
mod transform;
mod wpp;

pub use decoder::{HevcDecoderState, OutputPicture};

use crate::codecs::threading;
use crate::core::{
	self, CodecPrivate, CodecState, DecodeOut, Frame, Keyframe, Packet, Parameters, State,
	VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

/// [`core::Decoder`] for `hevc`: one access unit per packet (`hvcC`
/// length-prefixed or Annex B), pictures out in display order as planar
/// YUV 4:2:0, 4:2:2, 4:4:4 or gray, 8-bit or 16-bit little-endian samples.
#[derive(Debug, Clone, Default)]
pub struct HevcDecoder;

impl HevcDecoder {
	pub fn new() -> Self {
		Self
	}
}

fn to_frames(video: &mut VideoState, pictures: Vec<OutputPicture>) -> Vec<Frame> {
	pictures
		.into_iter()
		.map(|p| {
			let format = match p.chroma_format {
				0 => video::PixelFormat::Gray,
				2 => video::PixelFormat::YUV422,
				3 => video::PixelFormat::YUV444,
				_ => video::PixelFormat::YUV420,
			};
			let pixel = video::Pixel { depth: p.bit_depth as u8, format };
			video.width = p.width;
			video.height = p.height;
			video.pixel = pixel.clone();
			video.has_keyframe |= p.keyframe;
			let key = if p.keyframe { Keyframe::Key } else { Keyframe::NonKey };
			Frame::Video(VideoFrame::from_packed(p.data, p.width, p.height, pixel, key, p.pts))
		})
		.collect()
}

impl core::Decoder for HevcDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		State::Video(VideoState {
			width: parameters.width.unwrap_or(0),
			height: parameters.height.unwrap_or(0),
			pixel: video::Pixel { depth: 8, format: video::PixelFormat::YUV420 },
			reorder: Vec::new(),
			has_keyframe: false,
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			private: CodecPrivate::Hevc(Box::new(HevcDecoderState::with_threads(
				&parameters.extradata,
				threading::default_threads(),
			))),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Video(video) = state else {
			return Err(Message::Codec("hevc decoder requires a video state"));
		};
		if !matches!(video.private, CodecPrivate::Hevc(_)) {
			video.private = CodecPrivate::Hevc(Box::new(HevcDecoderState::with_threads(
				&video.codec_state.extradata,
				threading::default_threads(),
			)));
		}
		let CodecPrivate::Hevc(dec) = &mut video.private else {
			return Err(Message::Codec("hevc decoder state missing"));
		};
		let pictures = dec.decode_packet(&packet.data, packet.pts);
		Ok(DecodeOut { value: to_frames(video, pictures) })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut> {
		let State::Video(video) = state else { return Ok(DecodeOut { value: Vec::new() }) };
		let CodecPrivate::Hevc(dec) = &mut video.private else {
			return Ok(DecodeOut { value: Vec::new() });
		};
		let pictures = dec.drain();
		Ok(DecodeOut { value: to_frames(video, pictures) })
	}
}
