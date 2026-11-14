//! AV1 decoding through `rav1d`, the Rust port of dav1d (BSD-2-Clause).
//!
//! The decoder proper (OBU parsing, prediction, transforms, loop filters,
//! CDEF, loop restoration, film grain, frame and tile threading, SIMD) is
//! rav1d's. RMP adds the [`core::Decoder`] glue and keeps every `unsafe`
//! call in [`dav1d`], a small wrapper that copies pictures into owned
//! frames. Enabled by the default `av1` feature.

mod dav1d;

pub use dav1d::{Layout, Picture};

use crate::codecs::threading;
use crate::core::Keyframe;
use crate::core::{
	self, CodecPrivate, CodecState, DecodeOut, Frame, Packet, Parameters, State, Time, TimeBase,
	VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

/// Decoder state: an open rav1d context (re-opened lazily, so cloning a
/// state yields a fresh decoder rather than sharing one).
pub struct Av1DecoderState {
	ctx: Option<dav1d::Context>,
	threads: usize,
	/// `configOBUs` from the `av1C` box, sent ahead of the first packet.
	config_obus: Vec<u8>,
	time_base: Option<TimeBase>,
}

impl std::fmt::Debug for Av1DecoderState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Av1DecoderState")
			.field("open", &self.ctx.is_some())
			.field("threads", &self.threads)
			.finish()
	}
}

impl Clone for Av1DecoderState {
	fn clone(&self) -> Self {
		Self {
			ctx: None,
			threads: self.threads,
			config_obus: self.config_obus.clone(),
			time_base: self.time_base,
		}
	}
}

impl Av1DecoderState {
	/// `threads` worker threads (0: one per core, as dav1d does).
	pub fn new(extradata: &[u8], threads: usize) -> Self {
		// av1C: 4 header bytes (marker/version, profile/level, tier/depth...,
		// presentation delay) followed by configOBUs.
		let config_obus = if extradata.len() > 4 && extradata[0] & 0x80 != 0 {
			extradata[4..].to_vec()
		} else {
			Vec::new()
		};
		Self { ctx: None, threads, config_obus, time_base: None }
	}

	fn context(&mut self) -> Result<&mut dav1d::Context> {
		if self.ctx.is_none() {
			let mut ctx =
				dav1d::Context::open(self.threads).ok_or(Message::Codec("av1: cannot open decoder"))?;
			if !self.config_obus.is_empty() {
				// Sequence header ahead of the first temporal unit; no picture.
				let _ = ctx.decode(&self.config_obus, i64::MIN);
			}
			self.ctx = Some(ctx);
		}
		Ok(self.ctx.as_mut().expect("opened above"))
	}

	/// Decodes one temporal unit; returns the pictures now available.
	pub fn decode_packet(&mut self, data: &[u8], timestamp: i64) -> Result<Vec<Picture>> {
		self.context()?.decode(data, timestamp).map_err(|_| Message::Codec("av1: invalid bitstream"))
	}

	/// Returns the pictures still queued in the decoder.
	pub fn drain(&mut self) -> Result<Vec<Picture>> {
		match self.ctx.as_mut() {
			Some(c) => c.flush_out().map_err(|_| Message::Codec("av1: invalid bitstream")),
			None => Ok(Vec::new()),
		}
	}

	pub fn reset(&mut self) {
		if let Some(c) = self.ctx.as_mut() {
			c.flush();
		}
	}
}

fn pixel_of(p: &Picture) -> video::Pixel {
	let format = match p.layout {
		Layout::I400 => video::PixelFormat::Gray,
		Layout::I420 => video::PixelFormat::YUV420,
		Layout::I422 => video::PixelFormat::YUV422,
		Layout::I444 => video::PixelFormat::YUV444,
	};
	video::Pixel { depth: p.bit_depth, format }
}

/// [`core::Decoder`] for `av1`: one temporal unit per packet in, decoded
/// pictures out in presentation order (planar YUV, 8 or 10/12-bit).
#[derive(Debug, Clone, Default)]
pub struct Av1Decoder {
	threads: usize,
}

impl Av1Decoder {
	pub fn new() -> Self {
		Self { threads: 0 }
	}

	/// Worker threads for rav1d (0: [`threading::default_threads`], which
	/// honours `RMP_THREADS`).
	pub fn with_threads(threads: usize) -> Self {
		Self { threads }
	}

	fn thread_count(&self) -> usize {
		if self.threads == 0 { threading::default_threads() } else { self.threads }
	}
}

fn to_frames(video: &mut VideoState, pictures: Vec<Picture>, base: Option<TimeBase>) -> Vec<Frame> {
	pictures
		.into_iter()
		.map(|p| {
			let pixel = pixel_of(&p);
			video.width = p.width as u32;
			video.height = p.height as u32;
			video.pixel = pixel.clone();
			video.has_keyframe |= p.keyframe;
			let pts = base.filter(|_| p.timestamp >= 0).map(|b| Time::new(p.timestamp as u64, b));
			let key = if p.keyframe { Keyframe::Key } else { Keyframe::NonKey };
			Frame::Video(VideoFrame::from_packed(
				p.data,
				p.width as u32,
				p.height as u32,
				pixel,
				key,
				pts,
			))
		})
		.collect()
}

impl core::Decoder for Av1Decoder {
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
			private: CodecPrivate::Av1(Box::new(Av1DecoderState::new(
				&parameters.extradata,
				self.thread_count(),
			))),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Video(video) = state else {
			return Err(Message::Codec("av1 decoder requires a video state"));
		};
		if !matches!(video.private, CodecPrivate::Av1(_)) {
			let dec = Av1DecoderState::new(&video.codec_state.extradata, self.thread_count());
			video.private = CodecPrivate::Av1(Box::new(dec));
		}
		let CodecPrivate::Av1(dec) = &mut video.private else {
			return Err(Message::Codec("av1 decoder state missing"));
		};
		if let Some(t) = packet.pts {
			dec.time_base = Some(t.base());
		}
		let ts = packet.pts.map_or(-1, |t| t.ticks() as i64);
		let pictures = dec.decode_packet(&packet.data, ts)?;
		let base = dec.time_base;
		Ok(DecodeOut { value: to_frames(video, pictures, base) })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut> {
		let State::Video(video) = state else { return Ok(DecodeOut { value: Vec::new() }) };
		let CodecPrivate::Av1(dec) = &mut video.private else {
			return Ok(DecodeOut { value: Vec::new() });
		};
		let pictures = dec.drain()?;
		let base = dec.time_base;
		Ok(DecodeOut { value: to_frames(video, pictures, base) })
	}
}
