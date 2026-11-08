//! H.264 decoder front end: NAL dispatch, picture lifecycle, error
//! concealment and display-order output (FFmpeg `h264dec.c`,
//! `h264_slice.c`, `h264_picture.c`).

use std::sync::Arc;

use super::deblock::deblock_frame;
use super::decode::{Frame, SliceDecoder, SliceFilter};
use super::dpb::{Dpb, MbMotion, Picture, PocState};
use super::nal::{self, AvcConfig, Nal, kind};
use super::params::{self, MAX_PPS, MAX_SPS, Pps, Sps};
use super::slice::{self, Mmco, SliceHeader, SliceType};
use crate::core::{
	self, CodecPrivate, CodecState, DecodeOut, Frame as OutFrame, Keyframe, Packet, Parameters,
	State, Time, VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

/// The picture currently being assembled from slices.
#[derive(Debug, Clone)]
struct Current {
	frame: Frame,
	sps: Arc<Sps>,
	header: SliceHeader,
	poc: i32,
	/// (top, bottom) field order counts, for MMCO 5.
	poc_fields: (i32, i32),
	pts: Option<Time>,
	all_intra: bool,
}

/// Persistent decoder state (FFmpeg's `H264Context`).
#[derive(Debug, Clone)]
pub struct H264DecoderState {
	sps: Vec<Option<Arc<Sps>>>,
	pps: Vec<Option<Arc<Pps>>>,
	/// NAL length prefix size from `avcC`; `None` = Annex B byte stream.
	nal_length_size: Option<usize>,
	active_sps: Option<Arc<Sps>>,
	dpb: Dpb,
	poc: PocState,
	prev_ref_frame_num: u32,
	prev_had_mmco5: bool,
	next_id: u32,
	current: Option<Box<Current>>,
	/// Slices that failed to decode (their area is concealed).
	pub damaged_slices: u64,
	/// Slices or parameter sets that use unsupported features.
	pub unsupported: Option<&'static str>,
}

impl Default for H264DecoderState {
	fn default() -> Self {
		Self {
			sps: vec![None; MAX_SPS],
			pps: vec![None; MAX_PPS],
			nal_length_size: None,
			active_sps: None,
			dpb: Dpb::default(),
			poc: PocState::default(),
			prev_ref_frame_num: 0,
			prev_had_mmco5: false,
			next_id: 0,
			current: None,
			damaged_slices: 0,
			unsupported: None,
		}
	}
}

/// A decoded picture in display order.
#[derive(Debug, Clone)]
pub struct DecodedPicture {
	pub width: u32,
	pub height: u32,
	/// Planar YUV 4:2:0 (I420), cropped.
	pub data: Vec<u8>,
	pub pts: Option<Time>,
	pub keyframe: bool,
}

impl H264DecoderState {
	/// Creates a decoder from an `avcC` record (MP4) or Annex B extradata
	/// (SPS/PPS with start codes), or with no extradata at all.
	pub fn new(extradata: &[u8]) -> Self {
		let mut s = Self::default();
		if let Some(cfg) = AvcConfig::parse(extradata) {
			s.nal_length_size = Some(cfg.nal_length_size);
			for nal in cfg.sps.iter().chain(cfg.pps.iter()) {
				let _ = s.handle_param_set(nal);
			}
		} else if nal::is_annex_b(extradata) {
			for nal in nal::split_annex_b(extradata) {
				let _ = s.handle_param_set(nal);
			}
		}
		s
	}

	fn handle_param_set(&mut self, data: &[u8]) -> Option<()> {
		let nal = Nal::parse(data)?;
		match nal.kind {
			kind::SPS => match params::parse_sps(&nal.rbsp) {
				Ok(sps) => {
					let id = sps.id;
					self.sps[id] = Some(Arc::new(sps));
				}
				Err(e) => self.unsupported = Some(e.0),
			},
			kind::PPS => {
				let lookup = |id: usize| self.sps.get(id).and_then(|s| s.as_deref().cloned());
				match params::parse_pps(&nal.rbsp, lookup) {
					Ok(pps) => {
						let id = pps.id;
						self.pps[id] = Some(Arc::new(pps));
					}
					Err(e) => self.unsupported = Some(e.0),
				}
			}
			_ => return None,
		}
		Some(())
	}

	/// Clears all pictures and history (after a seek).
	pub fn reset(&mut self) {
		self.dpb.clear();
		self.poc = PocState::default();
		self.current = None;
		self.prev_ref_frame_num = 0;
		self.prev_had_mmco5 = false;
	}

	/// Decodes one access unit (packet) and returns the pictures that became
	/// ready for display.
	pub fn decode_packet(&mut self, data: &[u8], pts: Option<Time>) -> Result<Vec<DecodedPicture>> {
		// With an avcC record the stream is length-prefixed; a prefix such as
		// 00 00 01 xx must not be mistaken for a start code. Annex B is the
		// fallback only when the length framing does not fit at all.
		let nals = match self.nal_length_size {
			Some(n) => {
				let v = nal::split_length_prefixed(data, n);
				if v.is_empty() && nal::is_annex_b(data) { nal::split_annex_b(data) } else { v }
			}
			None => nal::split_annex_b(data),
		};
		let mut decoded_slice = false;
		for raw in nals {
			let Some(nal) = Nal::parse(raw) else { continue };
			match nal.kind {
				kind::SPS | kind::PPS => {
					self.handle_param_set(raw);
				}
				kind::SLICE | kind::IDR => {
					if self.handle_slice(&nal, pts).is_ok() {
						decoded_slice = true;
					}
				}
				kind::AUD | kind::END_SEQUENCE => self.finish_picture(),
				_ => {}
			}
		}
		self.finish_picture();
		if !decoded_slice {
			if let Some(reason) = self.unsupported {
				if self.dpb.output.is_empty() {
					return Err(Message::Codec(reason));
				}
			}
		}
		Ok(self.take_output())
	}

	/// Outputs every remaining picture (end of stream).
	pub fn flush(&mut self) -> Vec<DecodedPicture> {
		self.finish_picture();
		self.dpb.flush_output();
		self.take_output()
	}

	fn take_output(&mut self) -> Vec<DecodedPicture> {
		let Some(sps) = self.active_sps.clone() else {
			self.dpb.output.clear();
			return Vec::new();
		};
		self.dpb.output.drain(..).filter(|p| !p.non_existing).map(|p| crop(&p, &sps)).collect()
	}

	fn handle_slice(&mut self, nal: &Nal, pts: Option<Time>) -> std::result::Result<(), ()> {
		let pps_id = slice::peek_pps_id(&nal.rbsp).ok_or(())?;
		let pps = self.pps.get(pps_id).cloned().flatten().ok_or(())?;
		let sps = self.sps.get(pps.sps_id).cloned().flatten().ok_or(())?;
		let h = match slice::parse(&nal.rbsp, nal.kind, nal.ref_idc, &sps, &pps) {
			Ok(h) => h,
			Err(e) => {
				self.unsupported = Some(e.0);
				return Err(());
			}
		};
		if h.redundant_pic_cnt > 0 {
			return Ok(());
		}
		let new_picture = match &self.current {
			None => true,
			Some(c) => {
				h.first_mb == 0
					|| c.header.frame_num != h.frame_num
					|| c.header.pps_id != h.pps_id
					|| c.header.idr != h.idr
					|| (c.header.nal_ref_idc == 0) != (h.nal_ref_idc == 0)
					|| c.header.poc_lsb != h.poc_lsb
			}
		};
		if new_picture {
			self.finish_picture();
			self.start_picture(&sps, &h, pts);
		}
		let Some(mut cur) = self.current.take() else {
			return Err(());
		};
		let result = self.decode_slice(&mut cur, &sps, &pps, &h, &nal.rbsp);
		self.current = Some(cur);
		result
	}

	fn start_picture(&mut self, sps: &Arc<Sps>, h: &SliceHeader, pts: Option<Time>) {
		let changed = self.active_sps.as_ref().is_none_or(|a| {
			a.mb_width != sps.mb_width || a.mb_height != sps.mb_height || a.crop != sps.crop
		});
		if changed {
			// New resolution: everything buffered belongs to the old one.
			self.dpb.flush_output();
			let pending = std::mem::take(&mut self.dpb.output);
			self.dpb.clear();
			self.dpb.output = pending;
		}
		self.active_sps = Some(sps.clone());
		if h.idr {
			self.prev_ref_frame_num = 0;
		} else {
			let max = sps.max_frame_num();
			if h.frame_num != self.prev_ref_frame_num
				&& h.frame_num != (self.prev_ref_frame_num + 1) % max
			{
				let prev = self.prev_ref_frame_num;
				self.dpb.fill_frame_num_gap(sps, prev, h.frame_num, &mut self.next_id);
				self.prev_ref_frame_num = (h.frame_num + max - 1) % max;
				self.poc.set_prev_frame_num(self.prev_ref_frame_num);
			}
		}
		let fields = self.poc.compute(sps, h, self.prev_had_mmco5);
		let poc = fields.0.min(fields.1);
		let frame = Frame::new(sps.mb_width as usize, sps.mb_height as usize);
		self.current = Some(Box::new(Current {
			frame,
			sps: sps.clone(),
			header: h.clone(),
			poc,
			poc_fields: fields,
			pts,
			all_intra: true,
		}));
	}

	fn decode_slice(
		&mut self,
		cur: &mut Current,
		sps: &Sps,
		pps: &Pps,
		h: &SliceHeader,
		rbsp: &[u8],
	) -> std::result::Result<(), ()> {
		if h.slice_type != SliceType::I {
			cur.all_intra = false;
		}
		cur.frame.slices.push(SliceFilter {
			disable: h.disable_deblocking,
			alpha: h.alpha_offset,
			beta: h.beta_offset,
		});
		let slice_num = cur.frame.slices.len() as u32;
		let lists = match self.dpb.ref_lists(h, sps, cur.poc) {
			Ok(l) => l,
			Err(_) => {
				self.damaged_slices += 1;
				return Err(());
			}
		};
		let mut dec = SliceDecoder::new(&mut cur.frame, sps, pps, h, &lists, rbsp, slice_num, cur.poc);
		match dec.decode(rbsp) {
			Ok(()) => Ok(()),
			Err(_) => {
				self.damaged_slices += 1;
				Err(())
			}
		}
	}

	/// Conceals, deblocks and stores the current picture.
	fn finish_picture(&mut self) {
		let Some(cur) = self.current.take() else {
			return;
		};
		let Current { mut frame, sps, header, poc, poc_fields, pts, all_intra } = *cur;
		self.conceal(&mut frame);
		deblock_frame(&mut frame);

		let has_mmco5 = header.mmco.as_ref().is_some_and(|ops| ops.contains(&Mmco::Reset));
		let poc = if has_mmco5 { poc - poc_fields.0.min(poc_fields.1) } else { poc };
		let motion = frame
			.mbs
			.iter()
			.map(|m| MbMotion {
				mv: m.mv,
				ref_idx: m.ref_idx,
				ref_id: m.ref_id,
				intra: m.is_intra() || m.slice == 0,
			})
			.collect();
		self.next_id = self.next_id.wrapping_add(1).max(1);
		let pic = Picture {
			id: self.next_id,
			y: frame.y,
			cb: frame.cb,
			cr: frame.cr,
			poc,
			motion,
			pts,
			keyframe: header.idr || all_intra,
			non_existing: false,
		};
		let mmco5 = self.dpb.store(Arc::new(pic), &header, &sps, sps.reorder_depth());
		if header.nal_ref_idc != 0 {
			self.prev_ref_frame_num = if mmco5 { 0 } else { header.frame_num };
		}
		if mmco5 {
			let top_after = poc_fields.0 - poc_fields.0.min(poc_fields.1);
			self.poc.after_mmco5(top_after);
		}
		self.prev_had_mmco5 = mmco5;
	}

	/// Fills macroblocks that no slice decoded with the co-located area of
	/// the most recent reference picture (or mid-grey).
	fn conceal(&self, frame: &mut Frame) {
		let missing = frame.mbs.iter().any(|m| m.slice == 0);
		if !missing {
			return;
		}
		let reference = self.dpb.latest_ref();
		let w = frame.mb_w;
		for (addr, m) in frame.mbs.iter().enumerate() {
			if m.slice != 0 {
				continue;
			}
			let (mx, my) = (addr % w, addr / w);
			let copy = |dst: &mut super::recon::Plane, src: Option<&super::recon::Plane>, size: usize| {
				for y in 0..size {
					let row = (my * size + y) * dst.width + mx * size;
					match src {
						Some(s) if s.width == dst.width && s.height == dst.height => {
							dst.data[row..row + size].copy_from_slice(&s.data[row..row + size]);
						}
						_ => dst.data[row..row + size].fill(128),
					}
				}
			};
			copy(&mut frame.y, reference.as_ref().map(|r| &r.y), 16);
			copy(&mut frame.cb, reference.as_ref().map(|r| &r.cb), 8);
			copy(&mut frame.cr, reference.as_ref().map(|r| &r.cr), 8);
		}
	}
}

/// Copies the visible (cropped) area of a picture into an I420 buffer.
fn crop(p: &Picture, sps: &Sps) -> DecodedPicture {
	let (w, h) = (sps.width() as usize, sps.height() as usize);
	let (cl, ct) = (sps.crop[0] as usize, sps.crop[2] as usize);
	let mut data = Vec::with_capacity(w * h * 3 / 2);
	let mut plane = |src: &super::recon::Plane, x0: usize, y0: usize, pw: usize, ph: usize| {
		for y in 0..ph {
			let start = ((y0 + y) * src.width + x0).min(src.data.len());
			let end = (start + pw).min(src.data.len());
			data.extend_from_slice(&src.data[start..end]);
		}
	};
	plane(&p.y, cl, ct, w, h);
	plane(&p.cb, cl / 2, ct / 2, w / 2, h / 2);
	plane(&p.cr, cl / 2, ct / 2, w / 2, h / 2);
	DecodedPicture { width: w as u32, height: h as u32, data, pts: p.pts, keyframe: p.keyframe }
}

/// [`core::Decoder`] implementation: one access unit per packet in,
/// zero or more [`VideoFrame`]s (planar 8-bit 4:2:0, display order) out.
#[derive(Debug, Clone, Default)]
pub struct H264Decoder;

impl H264Decoder {
	pub fn new() -> Self {
		Self
	}
}

fn to_frames(video: &mut VideoState, pics: Vec<DecodedPicture>) -> Vec<OutFrame> {
	pics
		.into_iter()
		.map(|p| {
			video.width = p.width;
			video.height = p.height;
			if p.keyframe {
				video.has_keyframe = true;
			}
			OutFrame::Video(VideoFrame {
				data: p.data,
				width: p.width,
				height: p.height,
				pixel: video::Pixel { depth: 8, format: video::PixelFormat::YUV420 },
				keyframe: if p.keyframe { Keyframe::Key } else { Keyframe::NonKey },
				pts: p.pts,
			})
		})
		.collect()
}

impl core::Decoder for H264Decoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec = H264DecoderState::new(&parameters.extradata);
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
			private: CodecPrivate::H264(Box::new(dec)),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let frames = {
			let State::Video(video) = state else {
				return Err(Message::Codec("h264 decoder requires a video state"));
			};
			if !matches!(video.private, CodecPrivate::H264(_)) {
				video.private =
					CodecPrivate::H264(Box::new(H264DecoderState::new(&video.codec_state.extradata)));
			}
			let CodecPrivate::H264(dec) = &mut video.private else {
				return Err(Message::Codec("h264 decoder state missing"));
			};
			let pics = dec.decode_packet(&packet.data, packet.pts)?;
			to_frames(video, pics)
		};
		Ok(DecodeOut { value: frames, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		let frames = match state {
			State::Video(video) => {
				let pics = match &mut video.private {
					CodecPrivate::H264(dec) => dec.flush(),
					_ => Vec::new(),
				};
				to_frames(video, pics)
			}
			State::Audio(_) => Vec::new(),
		};
		Ok(DecodeOut { value: frames, state: state.clone() })
	}
}
