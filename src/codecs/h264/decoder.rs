//! H.264 decoder front end: NAL dispatch, picture lifecycle, error
//! concealment and display-order output (FFmpeg `h264dec.c`,
//! `h264_slice.c`, `h264_picture.c`).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::deblock::deblock_rows;
use super::decode::{DecodePools, Frame, SliceDecoder, SliceFilter};
use super::dpb::{Dpb, MbMotion, PicData, PicPlanes, Picture, PlanesRef, PocState, RefPic};
use super::mb::MbInfo;
use super::nal::{self, AvcConfig, Nal, kind};
use super::params::{self, MAX_PPS, MAX_SPS, Pps, Sps};
use super::recon::Plane;
use super::slice::{self, Mmco, SliceHeader, SliceType};
use crate::codecs::threading::{self, WorkerPool};
use crate::core::{
	self, CodecPrivate, CodecState, DecodeOut, Frame as OutFrame, FrameBuffer, Keyframe, Packet,
	Parameters, PlaneRef, State, Time, VideoFrame, VideoState, video,
};
use crate::message::{Message, Result};

/// One slice, parsed and with its reference lists built, waiting to have
/// its macroblocks decoded.
#[derive(Debug, Clone)]
struct SliceJob {
	pps: Arc<Pps>,
	header: SliceHeader,
	/// `None` if the reference lists could not be built (the slice is
	/// counted as damaged and its area concealed).
	lists: Option<[Vec<Option<RefPic>>; 2]>,
	rbsp: Vec<u8>,
}

/// The picture currently being assembled from slices.
#[derive(Debug, Clone)]
struct Current {
	sps: Arc<Sps>,
	header: SliceHeader,
	poc: i32,
	/// (top, bottom) field order counts, for MMCO 5.
	poc_fields: (i32, i32),
	pts: Option<Time>,
	all_intra: bool,
	slices: Vec<SliceJob>,
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
	/// Slices that failed to decode (their area is concealed). With frame
	/// threads, a picture's slices are counted once it has been decoded.
	pub damaged_slices: u64,
	/// Slices or parameter sets that use unsupported features.
	pub unsupported: Option<&'static str>,
	/// Frame threads (FFmpeg `-threads` with `thread_type=frame`); 1 decodes
	/// every picture on the calling thread.
	threads: usize,
	pool: Option<Arc<WorkerPool>>,
	/// Pictures handed to the pool and not yet known to be finished, in
	/// decode order.
	in_flight: VecDeque<Arc<PicData>>,
	/// Pictures in output order, waiting for their decode to finish. With
	/// `n` threads up to `n - 1` are held back (FFmpeg's frame-threading
	/// delay), so the caller doesn't wait on the picture just started.
	pending_out: VecDeque<(Arc<Picture>, Arc<Sps>)>,
	/// Where every picture's buffers come from and return to.
	pools: DecodePools,
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
			threads: 1,
			pool: None,
			in_flight: VecDeque::new(),
			pending_out: VecDeque::new(),
			pools: DecodePools::default(),
		}
	}
}

/// A decoded picture in display order.
///
/// Its planes are views (cropping offset + the decoder's stride) into the
/// decoder's own reference picture, not copies: the picture stays alive,
/// and out of the buffer pool, while any frame references it.
#[derive(Debug, Clone)]
pub struct DecodedPicture {
	pub width: u32,
	pub height: u32,
	/// Y, Cb, Cr (4:2:0), cropped.
	pub planes: [PlaneRef; 3],
	pub pts: Option<Time>,
	pub keyframe: bool,
}

impl DecodedPicture {
	/// The picture as packed planar I420 bytes (a copy).
	pub fn data(&self) -> Vec<u8> {
		let mut out = Vec::with_capacity(self.planes.iter().map(|p| p.row_bytes() * p.rows()).sum());
		for p in &self.planes {
			for row in p.rows_iter() {
				out.extend_from_slice(row);
			}
		}
		out
	}

	pub fn into_frame(self) -> VideoFrame {
		VideoFrame::from_planes(
			self.planes,
			self.width,
			self.height,
			video::Pixel { depth: 8, format: video::PixelFormat::YUV420 },
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
			s.threads = pool.size().max(1);
			s.pool = (pool.size() > 0).then_some(pool);
		}
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

	/// Number of frame threads.
	pub fn threads(&self) -> usize {
		self.threads
	}

	/// Allocation counters of the picture-sample pool. Once the DPB and
	/// the caller's frames reach a steady state, `allocated` stops growing:
	/// every new picture reuses the planes of one nobody holds any more.
	pub fn pool_stats(&self) -> crate::core::PoolStats {
		self.pools.planes.stats()
	}

	/// Clears all pictures and history (after a seek).
	pub fn reset(&mut self) {
		self.retire(true);
		self.pending_out.clear();
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
					if self.handle_slice(nal, pts).is_ok() {
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
		self.retire(false);
		Ok(self.take_output(false))
	}

	/// Outputs every remaining picture (end of stream).
	pub fn flush(&mut self) -> Vec<DecodedPicture> {
		self.finish_picture();
		self.dpb.flush_output();
		let out = self.take_output(true);
		self.retire(true);
		out
	}

	/// Moves the DPB's output into the delay queue and returns the pictures
	/// that may leave it (all of them when `all`).
	fn take_output(&mut self, all: bool) -> Vec<DecodedPicture> {
		match self.active_sps.clone() {
			Some(sps) => {
				for p in self.dpb.output.drain(..).filter(|p| !p.non_existing) {
					self.pending_out.push_back((p, sps.clone()));
				}
			}
			None => self.dpb.output.clear(),
		}
		// A fixed delay (not "whenever it's done"), so which call returns a
		// picture depends only on the thread count, never on timing.
		let delay = if all { 0 } else { self.threads - 1 };
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
	fn retire(&mut self, wait: bool) {
		while let Some(front) = self.in_flight.front() {
			if !wait && !front.is_done() {
				break;
			}
			front.wait_done();
			self.damaged_slices += u64::from(front.damaged.load(Ordering::Acquire));
			self.in_flight.pop_front();
		}
	}

	fn handle_slice(&mut self, nal: Nal, pts: Option<Time>) -> std::result::Result<(), ()> {
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
		self.current = Some(Box::new(Current {
			sps: sps.clone(),
			header: h.clone(),
			poc,
			poc_fields: fields,
			pts,
			all_intra: true,
			slices: Vec::new(),
		}));
	}

	/// Stores the current picture in the DPB and starts decoding it
	/// (on a worker, or here with one thread).
	fn finish_picture(&mut self) {
		let Some(cur) = self.current.take() else {
			return;
		};
		let Current { sps, header, poc, poc_fields, pts, all_intra, slices } = *cur;
		let (mb_w, mb_h) = (sps.mb_width as usize, sps.mb_height as usize);
		// Concealment uses the most recent reference before this picture.
		let conceal_ref = self.dpb.latest_ref();
		let publish = self.pool.is_some();
		let live = if publish {
			// Readers only touch rows after they have been published into
			// `live`, so its old contents are never seen.
			let pools = &self.pools;
			PicPlanes {
				y: Plane::recycled(&pools.planes, mb_w * 16, mb_h * 16),
				cb: Plane::recycled(&pools.planes, mb_w * 8, mb_h * 8),
				cr: Plane::recycled(&pools.planes, mb_w * 8, mb_h * 8),
				motion: pools.motion.take_recycled(mb_w * mb_h),
			}
		} else {
			PicPlanes::default()
		};
		let data = Arc::new(PicData::pending(mb_h as u32, live));

		let has_mmco5 = header.mmco.as_ref().is_some_and(|ops| ops.contains(&Mmco::Reset));
		let poc_out = if has_mmco5 { poc - poc_fields.0.min(poc_fields.1) } else { poc };
		self.next_id = self.next_id.wrapping_add(1).max(1);
		let pic = Picture {
			id: self.next_id,
			data: data.clone(),
			poc: poc_out,
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

		let pools = self.pools.clone();
		let job = PictureJob { sps, poc, slices, data: data.clone(), conceal_ref, publish, pools };
		match self.pool.clone() {
			Some(pool) => {
				// At most `threads` pictures in flight (FFmpeg keeps one per
				// frame thread): wait for the oldest before starting another.
				while self.in_flight.len() >= self.threads {
					self.retire_one();
				}
				self.in_flight.push_back(data);
				pool.submit(move || job.run());
			}
			None => {
				job.run();
				self.damaged_slices += u64::from(data.damaged.load(Ordering::Acquire));
			}
		}
	}

	fn retire_one(&mut self) {
		if let Some(front) = self.in_flight.pop_front() {
			front.wait_done();
			self.damaged_slices += u64::from(front.damaged.load(Ordering::Acquire));
		}
	}
}

/// Everything needed to decode one picture's macroblocks, independent of
/// the decoder state (so it can run on a worker thread).
struct PictureJob {
	sps: Arc<Sps>,
	poc: i32,
	slices: Vec<SliceJob>,
	data: Arc<PicData>,
	conceal_ref: Option<Arc<Picture>>,
	/// Copy finished rows into the shared picture as they complete (other
	/// threads may be predicting from it).
	publish: bool,
	pools: DecodePools,
}

/// Poisons the picture if its job unwinds, so that its readers fail
/// instead of waiting forever.
struct PoisonGuard(Option<Arc<PicData>>);

impl Drop for PoisonGuard {
	fn drop(&mut self) {
		if let Some(data) = self.0.take() {
			data.poison();
		}
	}
}

fn motion_of(m: &MbInfo) -> MbMotion {
	MbMotion { mv: m.mv, ref_idx: m.ref_idx, ref_id: m.ref_id, intra: m.is_intra() || m.slice == 0 }
}

/// Deblocks and publishes macroblock rows as soon as they can no longer
/// change (FFmpeg's `ff_thread_report_progress` after each row).
struct RowProgress {
	/// Macroblocks decoded so far, as a contiguous prefix in raster order.
	prefix: usize,
	/// Rows deblocked (and published) so far.
	deblocked: usize,
	motion: crate::core::Pooled<MbMotion>,
	data: Arc<PicData>,
	publish: bool,
}

impl RowProgress {
	/// Row `r` is deblocked once row `r + 1` is decoded: intra prediction in
	/// row `r + 1` needs row `r`'s samples before filtering.
	fn advance(&mut self, f: &mut Frame) {
		let count = f.mb_w * f.mb_h;
		while self.prefix < count && f.mbs[self.prefix].slice != 0 {
			self.prefix += 1;
		}
		let target = (self.prefix / f.mb_w).saturating_sub(1);
		self.deblock_to(f, target);
	}

	fn deblock_to(&mut self, f: &mut Frame, target: usize) {
		if target <= self.deblocked {
			return;
		}
		let from = self.deblocked;
		deblock_rows(f, from..target);
		for addr in from * f.mb_w..target * f.mb_w {
			self.motion[addr] = motion_of(&f.mbs[addr]);
		}
		self.deblocked = target;
		if self.publish {
			let view = PlanesRef { y: &f.y, cb: &f.cb, cr: &f.cr, motion: &self.motion };
			self.data.publish(view, from as u32, target as u32);
		}
	}
}

impl PictureJob {
	fn run(self) {
		let PictureJob { sps, poc, slices, data, conceal_ref, publish, pools } = self;
		let mut guard = PoisonGuard(Some(data.clone()));
		let (mb_w, mb_h) = (sps.mb_width as usize, sps.mb_height as usize);
		let mut frame = Frame::pooled(&pools, mb_w, mb_h);
		let mut rows = RowProgress {
			prefix: 0,
			deblocked: 0,
			// Every entry is written as its row is deblocked.
			motion: pools.motion.take_recycled(mb_w * mb_h),
			data: data.clone(),
			publish,
		};
		let mut damaged = 0u32;
		for (i, s) in slices.iter().enumerate() {
			let h = &s.header;
			frame.slices.push(SliceFilter {
				disable: h.disable_deblocking,
				alpha: h.alpha_offset,
				beta: h.beta_offset,
			});
			let Some(lists) = &s.lists else {
				damaged += 1;
				continue;
			};
			// A slice may not rewrite rows that were already filtered (only
			// out-of-order slices, which RMP doesn't support, would).
			if rows.deblocked > 0 && (h.first_mb as usize) < (rows.deblocked + 1) * mb_w {
				damaged += 1;
				continue;
			}
			let slice_num = i as u32 + 1;
			{
				let mut hook = |f: &mut Frame| rows.advance(f);
				let mut dec =
					SliceDecoder::new(&mut frame, &sps, &s.pps, h, lists, &s.rbsp, slice_num, poc)
						.with_row_hook(&mut hook);
				if dec.decode(&s.rbsp).is_err() {
					damaged += 1;
				}
			}
			rows.advance(&mut frame);
		}
		conceal(&mut frame, conceal_ref.as_deref());
		rows.deblock_to(&mut frame, mb_h);
		// Concealed macroblocks count as intra for later direct prediction.
		let mut motion = rows.motion;
		for (m, info) in motion.iter_mut().zip(frame.mbs.iter()) {
			if info.slice == 0 {
				*m = motion_of(info);
			}
		}
		data.damaged.store(damaged, Ordering::Release);
		data.finish(PicPlanes { y: frame.y, cb: frame.cb, cr: frame.cr, motion });
		guard.0 = None;
	}
}

/// Fills macroblocks that no slice decoded with the co-located area of
/// the most recent reference picture (or mid-grey).
fn conceal(frame: &mut Frame, reference: Option<&Picture>) {
	let missing = frame.mbs.iter().any(|m| m.slice == 0);
	if !missing {
		return;
	}
	let reference = reference.map(|r| r.data.wait_done());
	let w = frame.mb_w;
	for (addr, m) in frame.mbs.iter().enumerate() {
		if m.slice != 0 {
			continue;
		}
		let (mx, my) = (addr % w, addr / w);
		let copy = |dst: &mut Plane, src: Option<&Plane>, size: usize| {
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
		copy(&mut frame.y, reference.map(|r| &r.y), 16);
		copy(&mut frame.cb, reference.map(|r| &r.cb), 8);
		copy(&mut frame.cr, reference.map(|r| &r.cr), 8);
	}
}

fn luma_of(d: &PicData) -> &[u8] {
	d.done().map_or(&[], |p| &p.y.data)
}

fn cb_of(d: &PicData) -> &[u8] {
	d.done().map_or(&[], |p| &p.cb.data)
}

fn cr_of(d: &PicData) -> &[u8] {
	d.done().map_or(&[], |p| &p.cr.data)
}

/// Cropping as plane views: the visible area of a finished picture,
/// sharing its samples (no copy).
fn crop(pic: &Picture, sps: &Sps) -> DecodedPicture {
	let p = pic.data.wait_done();
	let (w, h) = (sps.width() as usize, sps.height() as usize);
	let (cl, ct) = (sps.crop[0] as usize, sps.crop[2] as usize);
	let view =
		|src: &Plane, project: fn(&PicData) -> &[u8], x0: usize, y0: usize, pw: usize, ph: usize| {
			let buffer = FrameBuffer::project(pic.data.clone(), project);
			// The SPS validation keeps the crop inside the coded picture; clamp
			// anyway, so a bad crop shows less rather than panicking.
			let x0 = x0.min(src.width);
			let y0 = y0.min(src.height);
			let pw = pw.min(src.width - x0);
			let ph = ph.min(src.height - y0);
			PlaneRef::new(buffer.clone(), y0 * src.width + x0, src.width, pw, ph)
				.or_else(|| PlaneRef::new(buffer, 0, src.width, 0, 0))
				.unwrap_or_else(|| unreachable!("an empty plane always fits"))
		};
	let planes = [
		view(&p.y, luma_of, cl, ct, w, h),
		view(&p.cb, cb_of, cl / 2, ct / 2, w / 2, h / 2),
		view(&p.cr, cr_of, cl / 2, ct / 2, w / 2, h / 2),
	];
	DecodedPicture { width: w as u32, height: h as u32, planes, pts: pic.pts, keyframe: pic.keyframe }
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
