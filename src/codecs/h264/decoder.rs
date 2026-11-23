//! H.264 decoder front end: NAL dispatch, picture lifecycle, error
//! concealment and display-order output (FFmpeg `h264dec.c`,
//! `h264_slice.c`, `h264_picture.c`).
//!
//! The picture machinery ([`Engine`]) is generic over the sample type:
//! 8-bit streams decode into `u8` planes, 9- to 14-bit streams into `u16`
//! ones. The decoder switches engine when a sequence of another bit depth
//! starts.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::deblock::{deblock_field, deblock_mbaff, deblock_rows};
use super::decode::{DecodePools, Frame, SliceDecoder, SliceFilter};
use super::dpb::{
	Dpb, MbMotion, PicData, PicPlanes, Picture, PlanesRef, PocState, RefPic, next_picture_id,
};
use super::nal::{self, AvcConfig, Nal, kind};
use super::params::{self, MAX_PPS, MAX_SPS, Pps, Sps};
use super::recon::{Pixel, Plane};
use super::slice::{self, SliceHeader, SliceType};
use crate::codecs::threading::{self, WorkerPool};
use crate::core::{
	self, CodecPrivate, CodecState, DecodeOut, Frame as OutFrame, FrameBuffer, Keyframe, Packet,
	Parameters, PlaneRef, State, Time, VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

/// One slice, parsed and with its reference lists built, waiting to have
/// its macroblocks decoded.
#[derive(Debug, Clone)]
struct SliceJob<T: Pixel> {
	pps: Arc<Pps>,
	header: SliceHeader,
	/// `None` if the reference lists could not be built (the slice is
	/// counted as damaged and its area concealed).
	lists: Option<[Vec<Option<RefPic<T>>>; 2]>,
	rbsp: Vec<u8>,
}

/// The picture (frame or field) currently being assembled from slices.
#[derive(Debug, Clone)]
struct Current<T: Pixel> {
	sps: Arc<Sps>,
	header: SliceHeader,
	/// POC of the frame or field.
	poc: i32,
	/// (top, bottom) field order counts (a field's own count twice).
	poc_fields: (i32, i32),
	pts: Option<Time>,
	all_intra: bool,
	slices: Vec<SliceJob<T>>,
	/// The second field of the pending first field.
	second: bool,
}

/// A first field, stored in the DPB, whose decoding waits for its second
/// field so that the pair is decoded by one job into one frame.
#[derive(Debug, Clone)]
struct Pending<T: Pixel> {
	job: PictureJob<T>,
	pic: Arc<Picture<T>>,
	parity: usize,
	frame_num: u32,
	sps: Arc<Sps>,
}

/// State shared by both sample types: parameter sets, POC and frame_num
/// history, threads and error counters.
#[derive(Debug, Clone)]
struct Common {
	sps: Vec<Option<Arc<Sps>>>,
	pps: Vec<Option<Arc<Pps>>>,
	/// NAL length prefix size from `avcC`; `None` = Annex B byte stream.
	nal_length_size: Option<usize>,
	active_sps: Option<Arc<Sps>>,
	poc: PocState,
	prev_ref_frame_num: u32,
	prev_had_mmco5: bool,
	next_id: u32,
	damaged_slices: u64,
	unsupported: Option<&'static str>,
	/// Frame threads (FFmpeg `-threads` with `thread_type=frame`); 1 decodes
	/// every picture on the calling thread.
	threads: usize,
	pool: Option<Arc<WorkerPool>>,
}

/// The pictures of one sample type (FFmpeg's DPB plus its frame-thread
/// bookkeeping).
#[derive(Debug, Clone)]
struct Engine<T: Pixel> {
	dpb: Dpb<T>,
	current: Option<Box<Current<T>>>,
	/// Pictures handed to the pool and not yet known to be finished, in
	/// decode order.
	in_flight: VecDeque<Arc<PicData<T>>>,
	pending: Option<Box<Pending<T>>>,
	/// Pictures in output order, waiting for their decode to finish. With
	/// `n` threads up to `n - 1` are held back (FFmpeg's frame-threading
	/// delay), so the caller doesn't wait on the picture just started.
	pending_out: VecDeque<(Arc<Picture<T>>, Arc<Sps>)>,
	/// Where every picture's buffers come from and return to.
	pools: DecodePools<T>,
}

impl<T: Pixel> Default for Engine<T> {
	fn default() -> Self {
		Self {
			dpb: Dpb::default(),
			current: None,
			in_flight: VecDeque::new(),
			pending: None,
			pending_out: VecDeque::new(),
			pools: DecodePools::default(),
		}
	}
}

#[derive(Debug, Clone)]
enum Engines {
	Narrow(Engine<u8>),
	Wide(Engine<u16>),
}

/// Persistent decoder state (FFmpeg's `H264Context`).
#[derive(Debug, Clone)]
pub struct H264DecoderState {
	common: Common,
	engine: Engines,
	/// Slices that failed to decode (their area is concealed). With frame
	/// threads, a picture's slices are counted once it has been decoded.
	pub damaged_slices: u64,
	/// Slices or parameter sets that use unsupported features.
	pub unsupported: Option<&'static str>,
}

impl Default for H264DecoderState {
	fn default() -> Self {
		Self {
			common: Common {
				sps: vec![None; MAX_SPS],
				pps: vec![None; MAX_PPS],
				nal_length_size: None,
				active_sps: None,
				poc: PocState::default(),
				prev_ref_frame_num: 0,
				prev_had_mmco5: false,
				next_id: 0,
				damaged_slices: 0,
				unsupported: None,
				threads: 1,
				pool: None,
			},
			engine: Engines::Narrow(Engine::default()),
			damaged_slices: 0,
			unsupported: None,
		}
	}
}

/// A decoded picture in display order.
///
/// For 8-bit streams its planes are views (cropping offset + the decoder's
/// stride) into the decoder's own reference picture, not copies: the
/// picture stays alive, and out of the buffer pool, while any frame
/// references it. Deeper samples are copied out as 16-bit little-endian.
#[derive(Debug, Clone)]
pub struct DecodedPicture {
	pub width: u32,
	pub height: u32,
	/// 8 to 14.
	pub bit_depth: u32,
	/// 4:2:2 rather than 4:2:0.
	pub chroma_422: bool,
	/// Y, Cb, Cr, cropped.
	pub planes: [PlaneRef; 3],
	pub pts: Option<Time>,
	pub keyframe: bool,
}

impl DecodedPicture {
	/// The picture as packed planar bytes (a copy): I420/I422, or 16-bit
	/// little-endian samples above 8 bits.
	pub fn data(&self) -> Vec<u8> {
		let mut out = Vec::with_capacity(self.planes.iter().map(|p| p.row_bytes() * p.rows()).sum());
		for p in &self.planes {
			for row in p.rows_iter() {
				out.extend_from_slice(row);
			}
		}
		out
	}

	pub fn pixel(&self) -> video::Pixel {
		let format =
			if self.chroma_422 { video::PixelFormat::YUV422 } else { video::PixelFormat::YUV420 };
		video::Pixel { depth: self.bit_depth as u8, format }
	}

	pub fn into_frame(self) -> VideoFrame {
		let pixel = self.pixel();
		VideoFrame::from_planes(
			self.planes,
			self.width,
			self.height,
			pixel,
			if self.keyframe { Keyframe::Key } else { Keyframe::NonKey },
			self.pts,
		)
	}
}

impl H264DecoderState {
	/// Creates a decoder from an `avcC` record (MP4) or Annex B extradata
	/// (SPS/PPS with start codes), or with no extradata at all.
	///
	/// Decodes on the calling thread; see [`Self::with_threads`].
	pub fn new(extradata: &[u8]) -> Self {
		Self::with_threads(extradata, 1)
	}

	/// Like [`Self::new`], decoding up to `threads` pictures at once (frame
	/// threading). Output is identical for every thread count; with more
	/// than one thread, pictures come out up to `threads - 1` packets later.
	pub fn with_threads(extradata: &[u8], threads: usize) -> Self {
		let mut s = Self::default();
		let threads = threads.clamp(1, threading::MAX_THREADS);
		if threads > 1 {
			let pool = WorkerPool::new(threads, "rmp-h264");
			s.common.threads = pool.size().max(1);
			s.common.pool = (pool.size() > 0).then_some(pool);
		}
		if let Some(cfg) = AvcConfig::parse(extradata) {
			s.common.nal_length_size = Some(cfg.nal_length_size);
			for nal in cfg.sps.iter().chain(cfg.pps.iter()) {
				let _ = s.common.handle_param_set(nal);
			}
		} else if nal::is_annex_b(extradata) {
			for nal in nal::split_annex_b(extradata) {
				let _ = s.common.handle_param_set(nal);
			}
		}
		s.sync();
		s
	}

	/// Copies the shared counters into the public fields.
	fn sync(&mut self) {
		self.damaged_slices = self.common.damaged_slices;
		self.unsupported = self.common.unsupported;
	}

	/// Number of frame threads.
	pub fn threads(&self) -> usize {
		self.common.threads
	}

	/// Allocation counters of the picture-sample pool. Once the DPB and
	/// the caller's frames reach a steady state, `allocated` stops growing:
	/// every new picture reuses the planes of one nobody holds any more.
	pub fn pool_stats(&self) -> crate::core::PoolStats {
		match &self.engine {
			Engines::Narrow(e) => e.pools.planes.stats(),
			Engines::Wide(e) => e.pools.planes.stats(),
		}
	}

	/// Clears all pictures and history (after a seek).
	pub fn reset(&mut self) {
		let c = &mut self.common;
		match &mut self.engine {
			Engines::Narrow(e) => e.reset(c),
			Engines::Wide(e) => e.reset(c),
		}
		c.poc = PocState::default();
		c.prev_ref_frame_num = 0;
		c.prev_had_mmco5 = false;
		self.sync();
	}

	/// Decodes one access unit (packet) and returns the pictures that became
	/// ready for display.
	pub fn decode_packet(&mut self, data: &[u8], pts: Option<Time>) -> Result<Vec<DecodedPicture>> {
		// With an avcC record the stream is length-prefixed; a prefix such as
		// 00 00 01 xx must not be mistaken for a start code. Annex B is the
		// fallback only when the length framing does not fit at all.
		let nals = match self.common.nal_length_size {
			Some(n) => {
				let v = nal::split_length_prefixed(data, n);
				if v.is_empty() && nal::is_annex_b(data) { nal::split_annex_b(data) } else { v }
			}
			None => nal::split_annex_b(data),
		};
		let mut decoded_slice = false;
		let mut out = Vec::new();
		for raw in nals {
			let Some(nal) = Nal::parse(raw) else { continue };
			match nal.kind {
				kind::SPS | kind::PPS => {
					self.common.handle_param_set(raw);
				}
				kind::SLICE | kind::IDR => {
					if self.handle_slice(nal, pts, &mut out).is_ok() {
						decoded_slice = true;
					}
				}
				kind::AUD | kind::END_SEQUENCE => self.finish_picture(),
				_ => {}
			}
		}
		self.finish_picture();
		let c = &mut self.common;
		let has_output = !out.is_empty()
			|| match &self.engine {
				Engines::Narrow(e) => !e.dpb.output.is_empty(),
				Engines::Wide(e) => !e.dpb.output.is_empty(),
			};
		if !decoded_slice && !has_output {
			if let Some(reason) = c.unsupported {
				self.sync();
				return Err(Message::Codec(reason));
			}
		}
		match &mut self.engine {
			Engines::Narrow(e) => {
				e.retire(c, false);
				out.extend(e.take_output(c, false));
			}
			Engines::Wide(e) => {
				e.retire(c, false);
				out.extend(e.take_output(c, false));
			}
		}
		self.sync();
		Ok(out)
	}

	/// Outputs every remaining picture (end of stream).
	pub fn flush(&mut self) -> Vec<DecodedPicture> {
		self.finish_picture();
		let c = &mut self.common;
		let out = match &mut self.engine {
			Engines::Narrow(e) => e.flush(c),
			Engines::Wide(e) => e.flush(c),
		};
		self.sync();
		out
	}

	fn finish_picture(&mut self) {
		let c = &mut self.common;
		match &mut self.engine {
			Engines::Narrow(e) => e.finish_picture(c),
			Engines::Wide(e) => e.finish_picture(c),
		}
	}

	fn handle_slice(
		&mut self,
		nal: Nal,
		pts: Option<Time>,
		out: &mut Vec<DecodedPicture>,
	) -> std::result::Result<(), ()> {
		let c = &mut self.common;
		let pps_id = slice::peek_pps_id(&nal.rbsp).ok_or(())?;
		let pps = c.pps.get(pps_id).cloned().flatten().ok_or(())?;
		let sps = c.sps.get(pps.sps_id).cloned().flatten().ok_or(())?;
		// A sequence of the other sample width: drain the current engine,
		// then switch.
		let wide = sps.bit_depth > 8;
		let is_wide = matches!(self.engine, Engines::Wide(_));
		if wide != is_wide {
			match &mut self.engine {
				Engines::Narrow(e) => out.extend(e.flush(c)),
				Engines::Wide(e) => out.extend(e.flush(c)),
			}
			self.engine =
				if wide { Engines::Wide(Engine::default()) } else { Engines::Narrow(Engine::default()) };
		}
		match &mut self.engine {
			Engines::Narrow(e) => e.handle_slice(c, nal, pps, sps, pts),
			Engines::Wide(e) => e.handle_slice(c, nal, pps, sps, pts),
		}
	}
}

impl Common {
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
}

impl<T: Pixel> Engine<T> {
	fn reset(&mut self, c: &mut Common) {
		self.retire(c, true);
		self.pending_out.clear();
		self.dpb.clear();
		self.current = None;
		self.pending = None;
	}

	fn flush(&mut self, c: &mut Common) -> Vec<DecodedPicture> {
		self.finish_picture(c);
		self.flush_pending(c);
		self.dpb.flush_output();
		let out = self.take_output(c, true);
		self.retire(c, true);
		out
	}

	/// Moves the DPB's output into the delay queue and returns the pictures
	/// that may leave it (all of them when `all`).
	fn take_output(&mut self, c: &Common, all: bool) -> Vec<DecodedPicture> {
		match c.active_sps.clone() {
			Some(sps) => {
				for p in self.dpb.output.drain(..).filter(|p| !p.non_existing) {
					self.pending_out.push_back((p, sps.clone()));
				}
			}
			None => self.dpb.output.clear(),
		}
		// A fixed delay (not "whenever it's done"), so which call returns a
		// picture depends only on the thread count, never on timing.
		let delay = if all { 0 } else { c.threads - 1 };
		let mut out = Vec::new();
		while self.pending_out.len() > delay {
			if let Some((p, sps)) = self.pending_out.pop_front() {
				out.push(crop(&p, &sps));
			}
		}
		out
	}

	/// Collects the damage counts of finished pictures; with `wait`, waits
	/// for every picture in flight.
	fn retire(&mut self, c: &mut Common, wait: bool) {
		while let Some(front) = self.in_flight.front() {
			if !wait && !front.is_done() {
				break;
			}
			front.wait_done();
			c.damaged_slices += u64::from(front.damaged.load(Ordering::Acquire));
			self.in_flight.pop_front();
		}
	}

	fn retire_one(&mut self, c: &mut Common) {
		if let Some(front) = self.in_flight.pop_front() {
			front.wait_done();
			c.damaged_slices += u64::from(front.damaged.load(Ordering::Acquire));
		}
	}

	fn handle_slice(
		&mut self,
		c: &mut Common,
		nal: Nal,
		pps: Arc<Pps>,
		sps: Arc<Sps>,
		pts: Option<Time>,
	) -> std::result::Result<(), ()> {
		let h = match slice::parse(&nal.rbsp, nal.kind, nal.ref_idc, &sps, &pps) {
			Ok(h) => h,
			Err(e) => {
				c.unsupported = Some(e.0);
				return Err(());
			}
		};
		if h.redundant_pic_cnt > 0 {
			return Ok(());
		}
		let new_picture = match &self.current {
			None => true,
			Some(cur) => {
				h.first_mb == 0
					|| cur.header.frame_num != h.frame_num
					|| cur.header.pps_id != h.pps_id
					|| cur.header.idr != h.idr
					|| (cur.header.nal_ref_idc == 0) != (h.nal_ref_idc == 0)
					|| cur.header.poc_lsb != h.poc_lsb
					|| cur.header.field_pic != h.field_pic
					|| cur.header.bottom_field != h.bottom_field
			}
		};
		if new_picture {
			self.finish_picture(c);
			self.start_picture(c, &sps, &h, pts);
		}
		let Some(mut cur) = self.current.take() else {
			return Err(());
		};
		if h.slice_type != SliceType::I {
			cur.all_intra = false;
		}
		// Reference lists depend on the DPB as it is now, so they are built
		// here, in decode order; the macroblocks are decoded later, possibly
		// on another thread.
		let lists = self.dpb.ref_lists(&h, &sps, cur.poc).ok();
		let ok = lists.is_some();
		cur.slices.push(SliceJob { pps, header: h, lists, rbsp: nal.rbsp });
		self.current = Some(cur);
		if ok { Ok(()) } else { Err(()) }
	}

	fn start_picture(&mut self, c: &mut Common, sps: &Arc<Sps>, h: &SliceHeader, pts: Option<Time>) {
		// The second field of a pair: opposite parity, same frame_num, right
		// after its first field (FFmpeg pairs regardless of nal_ref_idc).
		let second = self.pending.as_ref().is_some_and(|p| {
			h.field_pic
				&& h.parity() != Some(p.parity)
				&& h.frame_num == p.frame_num
				&& !h.idr
				&& Arc::ptr_eq(&p.sps, sps)
		});
		if !second {
			self.flush_pending(c);
		}
		let changed = c.active_sps.as_ref().is_none_or(|a| {
			a.mb_width != sps.mb_width
				|| a.mb_height != sps.mb_height
				|| a.crop != sps.crop
				|| a.chroma_format_idc != sps.chroma_format_idc
				|| a.bit_depth != sps.bit_depth
		});
		if changed {
			// New format: everything buffered belongs to the old one, and is
			// output with the parameters it was decoded with.
			self.dpb.flush_output();
			if let Some(old) = c.active_sps.clone() {
				for p in self.dpb.output.drain(..).filter(|p| !p.non_existing) {
					self.pending_out.push_back((p, old.clone()));
				}
			}
			self.dpb.clear();
		}
		c.active_sps = Some(sps.clone());
		if h.idr {
			c.prev_ref_frame_num = 0;
		} else if !second {
			let max = sps.max_frame_num();
			if h.frame_num != c.prev_ref_frame_num && h.frame_num != (c.prev_ref_frame_num + 1) % max {
				let prev = c.prev_ref_frame_num;
				self.dpb.fill_frame_num_gap(sps, prev, h.frame_num, &mut c.next_id);
				c.prev_ref_frame_num = (h.frame_num + max - 1) % max;
				c.poc.set_prev_frame_num(c.prev_ref_frame_num);
			}
		}
		let fields = c.poc.compute(sps, h, c.prev_had_mmco5);
		let poc = fields.0.min(fields.1);
		self.current = Some(Box::new(Current {
			sps: sps.clone(),
			header: h.clone(),
			poc,
			poc_fields: fields,
			pts,
			all_intra: true,
			slices: Vec::new(),
			second,
		}));
	}

	/// Decodes a first field whose pair did not come on its own.
	fn flush_pending(&mut self, c: &mut Common) {
		if let Some(p) = self.pending.take() {
			self.dpb.complete_unpaired(p.pic.id, p.sps.reorder_depth(), p.sps.dpb_frames());
			self.submit(c, p.job);
		}
	}

	/// Runs a picture job (on a worker, or here with one thread).
	fn submit(&mut self, c: &mut Common, job: PictureJob<T>) {
		let data = job.data.clone();
		match c.pool.clone() {
			Some(pool) => {
				// At most `threads` pictures in flight (FFmpeg keeps one per
				// frame thread): wait for the oldest before starting another.
				while self.in_flight.len() >= c.threads {
					self.retire_one(c);
				}
				self.in_flight.push_back(data);
				pool.submit(move || job.run());
			}
			None => {
				job.run();
				c.damaged_slices += u64::from(data.damaged.load(Ordering::Acquire));
			}
		}
	}

	/// Stores the current frame or field in the DPB and starts decoding it.
	/// A first field waits for its second field; the pair is then decoded
	/// as one job.
	fn finish_picture(&mut self, c: &mut Common) {
		let Some(cur) = self.current.take() else {
			return;
		};
		let Current { sps, header, poc, poc_fields, pts, all_intra, slices, second } = *cur;
		let parity = header.parity();
		let part =
			Part { parity, poc, field_poc: [poc_fields.0, poc_fields.1], mbaff: header.mbaff, slices };
		let pending = if second { self.pending.take() } else { None };
		let pic = match &pending {
			Some(p) => p.pic.clone(),
			None => {
				let (mb_w, mb_h) = (sps.mb_width as usize, sps.mb_height as usize);
				let c422 = sps.chroma_format_idc == 2;
				let mb_ch = if c422 { 16 } else { 8 };
				// Field pairs are published whole.
				let whole = header.field_pic;
				let publish = c.pool.is_some() && !whole;
				let live = if publish {
					// Readers only touch rows after they have been published into
					// `live`, so its old contents are never seen.
					let pools = &self.pools;
					PicPlanes {
						y: Plane::recycled(&pools.planes, mb_w * 16, mb_h * 16),
						cb: Plane::recycled(&pools.planes, mb_w * 8, mb_h * mb_ch),
						cr: Plane::recycled(&pools.planes, mb_w * 8, mb_h * mb_ch),
						motion: pools.motion.take_recycled(mb_w * mb_h),
					}
				} else {
					PicPlanes::default()
				};
				let data = Arc::new(PicData::pending(mb_h as u32, mb_ch as u32, whole, header.mbaff, live));
				c.next_id = next_picture_id(c.next_id);
				Arc::new(Picture {
					id: c.next_id,
					data,
					pts,
					keyframe: header.idr || all_intra,
					non_existing: false,
					field_coded: header.field_pic,
				})
			}
		};
		// Concealment uses the most recent reference before this picture.
		let conceal_ref = self.dpb.latest_ref().filter(|r| r.id != pic.id);
		let mmco5 = self.dpb.store(pic.clone(), &header, &sps, poc_fields, sps.reorder_depth());
		if header.nal_ref_idc != 0 {
			c.prev_ref_frame_num = if mmco5 { 0 } else { header.frame_num };
		}
		if mmco5 {
			let top_after = match parity {
				None => poc_fields.0 - poc_fields.0.min(poc_fields.1),
				Some(_) => 0,
			};
			c.poc.after_mmco5(top_after);
		}
		c.prev_had_mmco5 = mmco5;

		match pending {
			Some(mut p) => {
				p.job.parts.push(part);
				self.submit(c, p.job);
			}
			None => {
				let job = PictureJob {
					sps: sps.clone(),
					parts: vec![part],
					data: pic.data.clone(),
					id: pic.id,
					conceal_ref,
					publish: c.pool.is_some() && !header.field_pic,
					pools: self.pools.clone(),
				};
				match parity {
					Some(p) => {
						self.pending =
							Some(Box::new(Pending { job, pic, parity: p, frame_num: header.frame_num, sps }))
					}
					None => self.submit(c, job),
				}
			}
		}
	}
}

/// The slices of one frame or field.
#[derive(Debug, Clone)]
struct Part<T: Pixel> {
	parity: Option<usize>,
	poc: i32,
	field_poc: [i32; 2],
	/// An MBAFF frame (deblocked by pair rows).
	mbaff: bool,
	slices: Vec<SliceJob<T>>,
}

/// Everything needed to decode one picture's macroblocks, independent of
/// the decoder state (so it can run on a worker thread): a frame, or the
/// fields of a pair, decoded one after the other into the same frame.
#[derive(Debug, Clone)]
struct PictureJob<T: Pixel> {
	sps: Arc<Sps>,
	parts: Vec<Part<T>>,
	data: Arc<PicData<T>>,
	id: u32,
	conceal_ref: Option<Arc<Picture<T>>>,
	/// Copy finished rows into the shared picture as they complete (other
	/// threads may be predicting from it).
	publish: bool,
	pools: DecodePools<T>,
}

/// Poisons the picture if its job unwinds, so that its readers fail
/// instead of waiting forever.
struct PoisonGuard<T: Pixel>(Option<Arc<PicData<T>>>);

impl<T: Pixel> Drop for PoisonGuard<T> {
	fn drop(&mut self) {
		if let Some(data) = self.0.take() {
			data.poison();
		}
	}
}

/// Deblocks and publishes macroblock rows as soon as they can no longer
/// change (FFmpeg's `ff_thread_report_progress` after each row).
struct RowProgress<T: Pixel> {
	/// Macroblocks decoded so far, as a contiguous prefix in decoding order
	/// (raster, or pairs for MBAFF).
	prefix: usize,
	/// MBAFF frame: rows advance by pairs.
	mbaff: bool,
	/// Rows deblocked (and published) so far.
	deblocked: usize,
	motion: crate::core::Pooled<MbMotion>,
	data: Arc<PicData<T>>,
	publish: bool,
}

impl<T: Pixel> RowProgress<T> {
	/// Row `r` is deblocked once row `r + 1` is decoded: intra prediction in
	/// row `r + 1` needs row `r`'s samples before filtering.
	fn advance(&mut self, f: &mut Frame<T>) {
		let (w, count) = (f.mb_w, f.mb_w * f.mb_h);
		if self.mbaff {
			// Pair p's top macroblock is stored at row 2 (p / w), its bottom
			// one a row below.
			let store = |a: usize| (2 * (a / 2 / w) + a % 2) * w + (a / 2) % w;
			while self.prefix < count && f.mbs[store(self.prefix)].slice != 0 {
				self.prefix += 1;
			}
			let pairs = (self.prefix / (2 * w)).saturating_sub(1);
			self.deblock_to(f, 2 * pairs);
			return;
		}
		while self.prefix < count && f.mbs[self.prefix].slice != 0 {
			self.prefix += 1;
		}
		let target = (self.prefix / w).saturating_sub(1);
		self.deblock_to(f, target);
	}

	fn deblock_to(&mut self, f: &mut Frame<T>, target: usize) {
		if target <= self.deblocked {
			return;
		}
		let from = self.deblocked;
		if self.mbaff {
			deblock_mbaff(f, from / 2..target / 2);
		} else {
			deblock_rows(f, from..target);
		}
		for addr in from * f.mb_w..target * f.mb_w {
			self.motion[addr] = f.mbs[addr].motion();
		}
		self.deblocked = target;
		if self.publish {
			let view = PlanesRef { y: &f.y, cb: &f.cb, cr: &f.cr, motion: &self.motion };
			self.data.publish(view, from as u32, target as u32);
		}
	}
}

impl<T: Pixel> PictureJob<T> {
	fn run(self) {
		let PictureJob { sps, parts, data, id, conceal_ref, publish, pools } = self;
		let mut guard = PoisonGuard(Some(data.clone()));
		let (mb_w, mb_h) = (sps.mb_width as usize, sps.mb_height as usize);
		let mut frame = Frame::pooled(&pools, mb_w, mb_h, sps.bit_depth, sps.chroma_format_idc == 2);
		let mut rows = RowProgress {
			prefix: 0,
			mbaff: parts.iter().any(|p| p.mbaff),
			deblocked: 0,
			// Every entry is written as its row is deblocked.
			motion: pools.motion.take_recycled(mb_w * mb_h),
			data: data.clone(),
			publish,
		};
		let mut damaged = 0u32;
		let mut slice_num = 0u32;
		let fields = parts.iter().any(|p| p.parity.is_some());
		for part in &parts {
			for s in &part.slices {
				let h = &s.header;
				slice_num += 1;
				frame.slices.push(SliceFilter {
					disable: h.disable_deblocking,
					alpha: h.alpha_offset,
					beta: h.beta_offset,
				});
				let Some(lists) = &s.lists else {
					damaged += 1;
					continue;
				};
				if part.parity.is_none() {
					// A slice may not rewrite rows that were already filtered
					// (only out-of-order slices, which RMP doesn't support,
					// would). MBAFF rows go by pairs.
					let next = if part.mbaff { rows.deblocked + 2 } else { rows.deblocked + 1 };
					if rows.deblocked > 0 && (h.first_mb as usize) < next * mb_w {
						damaged += 1;
						continue;
					}
					let mut hook = |f: &mut Frame<T>| rows.advance(f);
					let mut dec =
						SliceDecoder::new(&mut frame, &sps, &s.pps, h, lists, &s.rbsp, slice_num, part.poc)
							.with_picture_id(id)
							.with_field_poc(part.field_poc)
							.with_row_hook(&mut hook);
					if dec.decode(&s.rbsp).is_err() {
						damaged += 1;
					}
					drop(dec);
					rows.advance(&mut frame);
				} else {
					let mut dec =
						SliceDecoder::new(&mut frame, &sps, &s.pps, h, lists, &s.rbsp, slice_num, part.poc)
							.with_picture_id(id)
							.with_field_poc(part.field_poc);
					if dec.decode(&s.rbsp).is_err() {
						damaged += 1;
					}
				}
			}
			if let Some(p) = part.parity {
				// The field is complete before the second field of the pair
				// (which may predict from it) starts.
				conceal_field(&mut frame, p, conceal_ref.as_deref());
				deblock_field(&mut frame, p);
			}
		}
		let motion = if fields {
			// A lone field: the missing one copies the reference (or grey).
			let present: Vec<usize> = parts.iter().filter_map(|p| p.parity).collect();
			for p in 0..2 {
				if !present.contains(&p) {
					conceal_field(&mut frame, p, conceal_ref.as_deref());
				}
			}
			let mut motion = rows.motion;
			for (m, info) in motion.iter_mut().zip(frame.mbs.iter()) {
				*m = info.motion();
			}
			motion
		} else {
			conceal(&mut frame, conceal_ref.as_deref());
			rows.deblock_to(&mut frame, mb_h);
			// Concealed macroblocks count as intra for later direct prediction.
			let mut motion = rows.motion;
			for (m, info) in motion.iter_mut().zip(frame.mbs.iter()) {
				if info.slice == 0 {
					*m = info.motion();
				}
			}
			motion
		};
		data.damaged.store(damaged, Ordering::Release);
		data.finish(PicPlanes { y: frame.y, cb: frame.cb, cr: frame.cr, motion });
		guard.0 = None;
	}
}

/// Fills the macroblocks of field `parity` that no slice decoded with the
/// same field of `reference` (or mid-grey).
fn conceal_field<T: Pixel>(frame: &mut Frame<T>, parity: usize, reference: Option<&Picture<T>>) {
	let w = frame.mb_w;
	let rows = frame.mb_h / 2;
	let missing = (0..rows).any(|r| {
		frame.mbs[(2 * r + parity) * w..(2 * r + parity + 1) * w].iter().any(|m| m.slice == 0)
	});
	if !missing {
		return;
	}
	let reference = reference.map(|r| r.data.wait_done());
	let grey = T::of(1 << (frame.bit_depth - 1));
	let max = frame.max();
	let ch = frame.mb_chroma_h();
	for r in 0..rows {
		for x in 0..w {
			if frame.mbs[(2 * r + parity) * w + x].slice != 0 {
				continue;
			}
			let copy = |dst: &mut Plane<T>, src: Option<&Plane<T>>, bw: usize, bh: usize| {
				let ok = src.filter(|s| s.width == dst.width && s.height == dst.height);
				let d = dst.dst(Some(parity), max);
				for y in 0..bh {
					let at = d.idx(x * bw, r * bh + y);
					match ok {
						Some(s) => d.data[at..at + bw].copy_from_slice(&s.data[at..at + bw]),
						None => d.data[at..at + bw].fill(grey),
					}
				}
			};
			copy(&mut frame.y, reference.map(|r| &r.y), 16, 16);
			copy(&mut frame.cb, reference.map(|r| &r.cb), 8, ch);
			copy(&mut frame.cr, reference.map(|r| &r.cr), 8, ch);
		}
	}
}

/// Fills macroblocks that no slice decoded with the co-located area of
/// the most recent reference picture (or mid-grey).
fn conceal<T: Pixel>(frame: &mut Frame<T>, reference: Option<&Picture<T>>) {
	let missing = frame.mbs.iter().any(|m| m.slice == 0);
	if !missing {
		return;
	}
	let reference = reference.map(|r| r.data.wait_done());
	let w = frame.mb_w;
	let grey = T::of(1 << (frame.bit_depth - 1));
	let ch = frame.mb_chroma_h();
	for (addr, m) in frame.mbs.iter().enumerate() {
		if m.slice != 0 {
			continue;
		}
		let (mx, my) = (addr % w, addr / w);
		let copy = |dst: &mut Plane<T>, src: Option<&Plane<T>>, bw: usize, bh: usize| {
			for y in 0..bh {
				let row = (my * bh + y) * dst.width + mx * bw;
				match src {
					Some(s) if s.width == dst.width && s.height == dst.height => {
						dst.data[row..row + bw].copy_from_slice(&s.data[row..row + bw]);
					}
					_ => dst.data[row..row + bw].fill(grey),
				}
			}
		};
		copy(&mut frame.y, reference.map(|r| &r.y), 16, 16);
		copy(&mut frame.cb, reference.map(|r| &r.cb), 8, ch);
		copy(&mut frame.cr, reference.map(|r| &r.cr), 8, ch);
	}
}

/// One plane of a finished 8-bit picture.
type Projection = fn(&PicData<u8>) -> &[u8];

fn luma_of(d: &PicData<u8>) -> &[u8] {
	d.done().map_or(&[], |p| &p.y.data)
}

fn cb_of(d: &PicData<u8>) -> &[u8] {
	d.done().map_or(&[], |p| &p.cb.data)
}

fn cr_of(d: &PicData<u8>) -> &[u8] {
	d.done().map_or(&[], |p| &p.cr.data)
}

/// Cropping: the visible area of a finished picture. 8-bit pictures are
/// shared as plane views (no copy); deeper ones are copied out as 16-bit
/// little-endian samples.
fn crop<T: Pixel>(pic: &Picture<T>, sps: &Sps) -> DecodedPicture {
	let p = pic.data.wait_done();
	let (w, h) = (sps.width() as usize, sps.height() as usize);
	let (cl, ct) = (sps.crop[0] as usize, sps.crop[2] as usize);
	let c422 = sps.chroma_format_idc == 2;
	let (cw, chh, cy0) = (w / 2, if c422 { h } else { h / 2 }, if c422 { ct } else { ct / 2 });
	// The SPS validation keeps the crop inside the coded picture; clamp
	// anyway, so a bad crop shows less rather than panicking.
	let fit = |src: &Plane<T>, x0: usize, y0: usize, pw: usize, ph: usize| {
		let x0 = x0.min(src.width);
		let y0 = y0.min(src.height);
		(x0, y0, pw.min(src.width - x0), ph.min(src.height - y0))
	};
	let rects =
		[fit(&p.y, cl, ct, w, h), fit(&p.cb, cl / 2, cy0, cw, chh), fit(&p.cr, cl / 2, cy0, cw, chh)];
	let planes: [PlaneRef; 3] = match narrow_data(&pic.data) {
		Some(narrow) => {
			// 8-bit: views into the shared picture.
			let projections: [Projection; 3] = [luma_of, cb_of, cr_of];
			let strides = [p.y.width, p.cb.width, p.cr.width];
			std::array::from_fn(|i| {
				let (x0, y0, pw, ph) = rects[i];
				let buffer = FrameBuffer::project(narrow.clone(), projections[i]);
				let stride = strides[i];
				PlaneRef::new(buffer.clone(), y0 * stride + x0, stride, pw, ph)
					.or_else(|| PlaneRef::new(buffer, 0, stride, 0, 0))
					.unwrap_or_else(|| unreachable!("an empty plane always fits"))
			})
		}
		None => {
			let src = [&p.y, &p.cb, &p.cr];
			std::array::from_fn(|i| {
				let (x0, y0, pw, ph) = rects[i];
				let mut bytes = Vec::with_capacity(pw * ph * 2);
				for y in y0..y0 + ph {
					let row = &src[i].data[y * src[i].width + x0..y * src[i].width + x0 + pw];
					for &v in row {
						let v: i32 = v.into();
						bytes.extend_from_slice(&(v as u16).to_le_bytes());
					}
				}
				let buffer = FrameBuffer::from_vec(bytes);
				PlaneRef::new(buffer.clone(), 0, pw * 2, pw * 2, ph)
					.or_else(|| PlaneRef::new(buffer, 0, 0, 0, 0))
					.unwrap_or_else(|| unreachable!("an empty plane always fits"))
			})
		}
	};
	DecodedPicture {
		width: w as u32,
		height: h as u32,
		bit_depth: sps.bit_depth,
		chroma_422: c422,
		planes,
		pts: pic.pts,
		keyframe: pic.keyframe,
	}
}

/// The picture data as its 8-bit type, if it is one.
fn narrow_data<T: Pixel>(d: &Arc<PicData<T>>) -> Option<Arc<PicData<u8>>> {
	let any: Arc<dyn std::any::Any + Send + Sync> = d.clone();
	any.downcast::<PicData<u8>>().ok()
}

/// [`core::Decoder`] implementation: one access unit per packet in,
/// zero or more [`VideoFrame`]s (planar 4:2:0 or 4:2:2, 8 to 14 bits,
/// display order) out.
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
			video.pixel = p.pixel();
			if p.keyframe {
				video.has_keyframe = true;
			}
			OutFrame::Video(p.into_frame())
		})
		.collect()
}

impl core::Decoder for H264Decoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec = H264DecoderState::with_threads(&parameters.extradata, threading::default_threads());
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

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let frames = {
			let State::Video(video) = state else {
				return Err(Message::Codec("h264 decoder requires a video state"));
			};
			if !matches!(video.private, CodecPrivate::H264(_)) {
				video.private = CodecPrivate::H264(Box::new(H264DecoderState::with_threads(
					&video.codec_state.extradata,
					threading::default_threads(),
				)));
			}
			let CodecPrivate::H264(dec) = &mut video.private else {
				return Err(Message::Codec("h264 decoder state missing"));
			};
			let pics = dec.decode_packet(&packet.data, packet.pts)?;
			to_frames(video, pics)
		};
		Ok(DecodeOut { value: frames })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut> {
		let frames = match state {
			State::Video(video) => {
				let pics = match &mut video.private {
					CodecPrivate::H264(dec) => dec.flush(),
					_ => Vec::new(),
				};
				to_frames(video, pics)
			}
			State::Audio(_) | State::Subtitle(_) => Vec::new(),
		};
		Ok(DecodeOut { value: frames })
	}
}
