//! HEVC decoder front end: NAL dispatch, POC, reference picture sets,
//! reference lists, the DPB and output order (ITU-T H.265 §8.1-8.3,
//! Annex C; FFmpeg `hevcdec.c`, `hevc/refs.c`).

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use super::ctu::{PicState, SliceDecoder, SliceRefs};
use super::picture::{Frame, FrameRef, MvField, RefInfo, RefPic, picture_planes};
use super::ps::{Layout, MAX_PPS, MAX_SPS, Pps, Sps};
use super::slice::{self, SliceHeader, SliceType, is_idr, is_irap, nal};
use super::wpp::{self, Workspaces};
use crate::codecs::h264::nal::{split_annex_b, split_length_prefixed};
use crate::codecs::threading::WorkerPool;
use crate::core::Time;

/// A picture in display order, cropped.
#[derive(Debug, Clone)]
pub struct OutputPicture {
	pub width: u32,
	pub height: u32,
	pub bit_depth: u32,
	/// chroma_format_idc: 0 monochrome (luma plane only), 1 4:2:0, 2 4:2:2,
	/// 3 4:4:4.
	pub chroma_format: u32,
	/// Packed planar: one byte per sample at 8-bit, little-endian u16
	/// above.
	pub data: Vec<u8>,
	pub pts: Option<Time>,
	pub keyframe: bool,
}

/// A picture bumped for output in packet `at`.
struct PendingOutput {
	at: u64,
	frame: FrameRef,
	crop: [u32; 4],
	pts: Option<Time>,
	keyframe: bool,
}

/// Recycled buffers for pictures published while they decode.
type LivePool = Arc<Mutex<Vec<Frame>>>;

struct DpbEntry {
	frame: Option<FrameRef>,
	poc: i32,
	short_ref: bool,
	long_ref: bool,
	output: bool,
	/// Bumping-process counter (pictures decoded since it entered the DPB).
	sequence: u32,
	pts: Option<Time>,
	keyframe: bool,
	crop: [u32; 4],
}

/// One slice segment of the current picture, kept until the picture is
/// complete and decoded by its picture job.
pub(super) struct SliceJob {
	pub header: SliceHeader,
	pub rbsp: Vec<u8>,
	/// Positions (in the escaped NAL unit) of the emulation prevention
	/// bytes removed from `rbsp`, to locate the entry points.
	pub escapes: Vec<u32>,
	pub refs: Arc<SliceRefs>,
}

struct Current {
	sps: Arc<Sps>,
	pps: Arc<Pps>,
	layout: Arc<Layout>,
	poc: i32,
	header: SliceHeader,
	dpb_index: usize,
	handle: FrameRef,
	slices: Vec<SliceJob>,
	refs_of_slice: Option<Arc<SliceRefs>>,
	/// Pictures of the RPS current lists (with long-term marking).
	st_before: Vec<Option<FrameRef>>,
	st_after: Vec<Option<FrameRef>>,
	lt_curr: Vec<Option<FrameRef>>,
	error: bool,
}

/// Persistent decoder state.
pub struct HevcDecoderState {
	sps: Vec<Option<Arc<Sps>>>,
	pps: Vec<Option<Arc<Pps>>>,
	layouts: Vec<Option<Arc<Layout>>>,
	nal_length_size: Option<usize>,
	dpb: Vec<DpbEntry>,
	cur: Option<Box<Current>>,
	prev_tid0_poc: i32,
	/// The next picture starts a new coded video sequence.
	first_picture: bool,
	/// NoRaslOutputFlag of the last IRAP picture.
	no_rasl_output: bool,
	active_sps: Option<Arc<Sps>>,
	/// Last independent slice header of the current picture.
	prev_header: Option<SliceHeader>,
	/// Bumped pictures, waiting for their picture job and the output delay.
	pending_out: VecDeque<PendingOutput>,
	out: Vec<OutputPicture>,
	/// Packets (access units) decoded so far.
	packets: u64,
	threads: usize,
	/// Frame threads (none with one thread: pictures decode on the caller).
	pool: Option<Arc<WorkerPool>>,
	in_flight: VecDeque<FrameRef>,
	/// Recycled buffers for pictures published while they decode.
	live_pool: LivePool,
	/// Threads per picture for the rows of WPP slice segments.
	wpp_threads: usize,
	workspaces: Arc<Workspaces>,
	pub damaged_slices: u64,
	pub unsupported: Option<&'static str>,
}

impl std::fmt::Debug for HevcDecoderState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("HevcDecoderState").field("dpb", &self.dpb.len()).finish()
	}
}

impl Clone for HevcDecoderState {
	/// Cloning yields a fresh decoder with the same configuration.
	fn clone(&self) -> Self {
		let mut s = Self {
			sps: self.sps.clone(),
			pps: self.pps.clone(),
			layouts: self.layouts.clone(),
			nal_length_size: self.nal_length_size,
			..Self::default()
		};
		s.set_threads(self.threads);
		s.wpp_threads = self.wpp_threads;
		s
	}
}

impl Default for HevcDecoderState {
	fn default() -> Self {
		Self {
			sps: vec![None; MAX_SPS],
			pps: vec![None; MAX_PPS],
			layouts: vec![None; MAX_PPS],
			nal_length_size: None,
			dpb: Vec::new(),
			cur: None,
			prev_tid0_poc: 0,
			first_picture: true,
			no_rasl_output: true,
			active_sps: None,
			prev_header: None,
			pending_out: VecDeque::new(),
			out: Vec::new(),
			packets: 0,
			threads: 1,
			pool: None,
			in_flight: VecDeque::new(),
			live_pool: Arc::new(Mutex::new(Vec::new())),
			wpp_threads: 1,
			workspaces: Arc::new(Workspaces::default()),
			damaged_slices: 0,
			unsupported: None,
		}
	}
}

#[derive(Debug)]
pub struct HevcError(pub &'static str);

impl HevcDecoderState {
	/// Creates a decoder that decodes on the calling thread; `extradata` may
	/// be an `hvcC` record or Annex B parameter sets. See
	/// [`Self::with_threads`].
	pub fn new(extradata: &[u8]) -> Self {
		Self::with_threads(extradata, 1)
	}

	/// Creates a decoder that decodes up to `threads` pictures at once
	/// (frame threading, FFmpeg's `-threads`). Pictures are returned
	/// `threads - 1` packets later than with one thread; the output is
	/// identical.
	pub fn with_threads(extradata: &[u8], threads: usize) -> Self {
		let (frames, rows) = split_threads(threads);
		Self::with_threading(extradata, frames, rows)
	}

	/// Creates a decoder with `frame_threads` pictures in flight and up to
	/// `row_threads` threads per picture for the rows of WPP slice segments
	/// (both 1: everything on the calling thread).
	pub fn with_threading(extradata: &[u8], frame_threads: usize, row_threads: usize) -> Self {
		let mut s = Self::default();
		s.set_threads(frame_threads);
		s.wpp_threads = row_threads.clamp(1, crate::codecs::threading::MAX_THREADS);
		if extradata.len() > 22 && extradata[0] == 1 {
			s.nal_length_size = Some(usize::from(extradata[21] & 3) + 1);
			let narrays = extradata[22] as usize;
			let mut at = 23;
			'outer: for _ in 0..narrays {
				if at + 3 > extradata.len() {
					break;
				}
				let count = usize::from(u16::from_be_bytes([extradata[at + 1], extradata[at + 2]]));
				at += 3;
				for _ in 0..count {
					if at + 2 > extradata.len() {
						break 'outer;
					}
					let len = usize::from(u16::from_be_bytes([extradata[at], extradata[at + 1]]));
					at += 2;
					let Some(n) = extradata.get(at..at + len) else { break 'outer };
					let _ = s.nal_unit(n, None);
					at += len;
				}
			}
		} else if !extradata.is_empty() {
			for n in split_annex_b(extradata) {
				let _ = s.nal_unit(n, None);
			}
		}
		s
	}

	fn set_threads(&mut self, threads: usize) {
		self.threads = threads.clamp(1, crate::codecs::threading::MAX_THREADS);
		self.pool = (self.threads > 1).then(|| WorkerPool::new(self.threads, "hevc"));
	}

	/// Decodes one access unit (or a run of NAL units); returns pictures
	/// ready for display.
	pub fn decode_packet(&mut self, data: &[u8], pts: Option<Time>) -> Vec<OutputPicture> {
		let nals = match self.nal_length_size {
			Some(n) if tiles_exactly(data, n) => split_length_prefixed(data, n),
			_ => split_annex_b(data),
		};
		for n in nals {
			if let Err(e) = self.nal_unit(n, pts) {
				if std::env::var_os("RMP_HEVC_DEBUG").is_some() {
					eprintln!("hevc: {}", e.0);
				}
				self.damaged_slices += 1;
				if self.unsupported.is_none() && e.0.contains("unsupported") {
					self.unsupported = Some(e.0);
				}
			}
		}
		// Packets carry whole access units: finish the picture now.
		self.finish_picture();
		self.packets += 1;
		self.emit(false);
		std::mem::take(&mut self.out)
	}

	/// Outputs every picture still waiting in the DPB.
	pub fn drain(&mut self) -> Vec<OutputPicture> {
		self.finish_picture();
		while self.bump() {}
		self.dpb.clear();
		self.emit(true);
		while !self.in_flight.is_empty() {
			self.retire_one();
		}
		std::mem::take(&mut self.out)
	}

	pub fn reset(&mut self) {
		self.cur = None;
		self.dpb.clear();
		self.out.clear();
		self.pending_out.clear();
		while !self.in_flight.is_empty() {
			self.retire_one();
		}
		self.first_picture = true;
	}

	/// Crops the bumped pictures whose output delay has passed (all of
	/// them with `all`), waiting for their picture jobs. The delay is fixed
	/// (`threads - 1` packets), so which call returns a picture never
	/// depends on timing.
	fn emit(&mut self, all: bool) {
		let delay = self.threads as u64 - 1;
		while let Some(p) = self.pending_out.front() {
			// Bumped while decoding packet `at`: returned by the call for
			// packet `at + delay` (`packets` counts the packets done).
			if !all && p.at + delay >= self.packets {
				break;
			}
			let PendingOutput { frame: f, crop, pts, keyframe, .. } =
				self.pending_out.pop_front().expect("front");
			let frame = f.wait_done();
			let ready = f.output.lock().unwrap_or_else(|e| e.into_inner()).take();
			self.out.push(match ready {
				Some(data) => output_picture(frame, crop, data, pts, keyframe),
				None => crop_output(frame, crop, pts, keyframe),
			});
		}
		// Count the damage of finished jobs.
		while self.in_flight.front().is_some_and(|f| f.done().is_some()) {
			self.retire_one();
		}
	}

	fn retire_one(&mut self) {
		if let Some(f) = self.in_flight.pop_front() {
			f.wait_done();
			self.damaged_slices += f.damaged.load(Ordering::Acquire);
		}
	}

	fn nal_unit(&mut self, data: &[u8], pts: Option<Time>) -> Result<(), HevcError> {
		if data.len() < 2 || data[0] & 0x80 != 0 {
			return Ok(());
		}
		let t = (data[0] >> 1) & 0x3f;
		let layer = ((data[0] & 1) << 5) | (data[1] >> 3);
		let tid = (data[1] & 7).saturating_sub(1);
		if layer != 0 {
			return Ok(());
		}
		match t {
			nal::SPS => {
				let rbsp = unescape_with_positions(data).0;
				let sps = Sps::parse(&rbsp).map_err(|e| HevcError(e.0))?;
				let id = sps.id;
				self.sps[id] = Some(Arc::new(sps));
			}
			nal::PPS => {
				let rbsp = unescape_with_positions(data).0;
				let pps = Pps::parse(&rbsp, &self.sps).map_err(|e| HevcError(e.0))?;
				let sps = self.sps[pps.sps_id].clone().ok_or(HevcError("hevc: pps without sps"))?;
				let layout = pps.layout(&sps).map_err(|e| HevcError(e.0))?;
				let id = pps.id;
				self.layouts[id] = Some(Arc::new(layout));
				self.pps[id] = Some(Arc::new(pps));
			}
			nal::EOS | nal::EOB => {
				self.finish_picture();
				self.first_picture = true;
			}
			0..=9 | 16..=21 => {
				let (rbsp, escapes) = unescape_with_positions(data);
				self.slice(rbsp, escapes, t, tid, pts)?;
			}
			_ => {}
		}
		Ok(())
	}

	fn slice(
		&mut self,
		rbsp: Vec<u8>,
		escapes: Vec<u32>,
		t: u8,
		tid: u8,
		pts: Option<Time>,
	) -> Result<(), HevcError> {
		// Peek first_slice_segment_in_pic_flag.
		let first = rbsp.get(2).is_some_and(|b| b & 0x80 != 0);
		if first {
			self.finish_picture();
		}
		// RASL pictures of an IRAP that starts the sequence are skipped.
		if (t == nal::RASL_N || t == nal::RASL_R) && self.no_rasl_output {
			return Ok(());
		}
		if !first && self.cur.is_none() {
			return Err(HevcError("hevc: slice without picture start"));
		}
		let prev = if first { None } else { self.prev_header.as_ref() };
		let sh = slice::parse(&rbsp, t, &self.sps, &self.pps, prev).map_err(|e| HevcError(e.0))?;
		if !sh.dependent {
			self.prev_header = Some(sh.clone());
		}
		if first {
			self.start_picture(&sh, t, tid, pts)?;
		}
		let Some(cur) = self.cur.as_mut() else { return Ok(()) };
		if cur.error {
			return Ok(());
		}
		if sh.pps_id != cur.header.pps_id {
			return Err(HevcError("hevc: pps change within a picture"));
		}
		// Reference lists.
		let refs = if sh.dependent {
			cur.refs_of_slice.clone().ok_or(HevcError("hevc: dependent slice without lists"))?
		} else {
			let r = Arc::new(build_lists(cur, &sh)?);
			cur.refs_of_slice = Some(r.clone());
			r
		};
		cur.slices.push(SliceJob { header: sh, rbsp, escapes, refs });
		Ok(())
	}

	fn start_picture(
		&mut self,
		sh: &SliceHeader,
		t: u8,
		tid: u8,
		pts: Option<Time>,
	) -> Result<(), HevcError> {
		let pps = self.pps[sh.pps_id].clone().ok_or(HevcError("hevc: missing pps"))?;
		let sps = self.sps[pps.sps_id].clone().ok_or(HevcError("hevc: missing sps"))?;
		let layout = self.layouts[sh.pps_id].clone().ok_or(HevcError("hevc: missing pps layout"))?;
		if let Some(a) = &self.active_sps {
			if a.width != sps.width
				|| a.height != sps.height
				|| a.bit_depth != sps.bit_depth
				|| a.chroma_format_idc != sps.chroma_format_idc
			{
				// New sequence geometry: output everything first.
				while self.bump() {}
				self.dpb.clear();
			}
		}
		self.active_sps = Some(sps.clone());
		if is_irap(t) {
			self.no_rasl_output =
				is_idr(t) || (nal::BLA_W_LP..=nal::BLA_N_LP).contains(&t) || self.first_picture;
		}
		// POC (8.3.1).
		let max_lsb = 1i32 << sps.log2_max_poc_lsb;
		let lsb = sh.poc_lsb as i32;
		let msb = if is_irap(t) && self.no_rasl_output {
			0
		} else {
			let prev_lsb = self.prev_tid0_poc & (max_lsb - 1);
			let prev_msb = self.prev_tid0_poc - prev_lsb;
			if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
				prev_msb + max_lsb
			} else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
				prev_msb - max_lsb
			} else {
				prev_msb
			}
		};
		let poc = msb + lsb;
		let rasl_radl = (6..=9).contains(&t);
		if tid == 0 && !rasl_radl && !slice::is_sub_layer_non_ref(t) {
			self.prev_tid0_poc = poc;
		}
		self.first_picture = false;

		// Reference picture set (8.3.2).
		if is_irap(t) && self.no_rasl_output {
			for e in &mut self.dpb {
				e.short_ref = false;
				e.long_ref = false;
			}
			if sh.no_output_of_prior_pics && t != nal::CRA_NUT {
				self.dpb.retain(|e| !e.output || e.frame.is_none());
				for e in &mut self.dpb {
					e.output = false;
				}
			} else {
				while self.bump() {}
			}
			self.dpb.retain(|e| e.output || e.short_ref || e.long_ref);
		}
		let mut keep = vec![(false, false); self.dpb.len()];
		let find = |dpb: &[DpbEntry], p: i32, mask: i32| {
			dpb.iter().position(|e| (e.poc & mask) == p && e.frame.is_some())
		};
		let mut st_before = Vec::new();
		let mut st_after = Vec::new();
		let mut lt_curr = Vec::new();
		let mut missing: Vec<(i32, u8)> = Vec::new(); // (poc, list: 0 before, 1 after, 2 lt)
		if let Some(st) = &sh.short_term_rps {
			for i in 0..st.num_delta_pocs {
				let p = poc + st.delta_poc[i];
				match find(&self.dpb, p, !0) {
					Some(k) => {
						keep[k].0 = true;
						if st.used[i] {
							let f = self.dpb[k].frame.clone();
							if i < st.num_negative { st_before.push(f) } else { st_after.push(f) }
						}
					}
					None if st.used[i] => missing.push((p, if i < st.num_negative { 0 } else { 1 })),
					None => {}
				}
			}
		}
		let lt = &sh.long_term;
		let mut lt_entries = Vec::new();
		for i in 0..lt.count {
			let (p, mask) = if lt.has_msb[i] {
				(lt.poc_lsb[i] + poc - lt.msb_cycle[i] * max_lsb - lsb, !0)
			} else {
				(lt.poc_lsb[i], max_lsb - 1)
			};
			match find(&self.dpb, p, mask) {
				Some(k) => {
					keep[k].1 = true;
					if lt.used[i] {
						lt_entries.push(k);
						lt_curr.push(self.dpb[k].frame.clone());
					}
				}
				None if lt.used[i] => missing.push((p, 2)),
				None => {}
			}
		}
		for (e, &(s, l)) in self.dpb.iter_mut().zip(&keep) {
			e.short_ref = s && !l;
			e.long_ref = l;
		}
		// Generate missing references (gray, intra) as FFmpeg does.
		for (p, which) in missing {
			let f = Some(missing_frame(&sps, p));
			match which {
				0 => st_before.push(f),
				1 => st_after.push(f),
				_ => lt_curr.push(f),
			}
		}
		self.dpb.retain(|e| e.output || e.short_ref || e.long_ref);

		// The current picture enters the DPB.
		let output = sh.pic_output && !((t == nal::RASL_N || t == nal::RASL_R) && self.no_rasl_output);
		for e in &mut self.dpb {
			e.sequence += 1;
		}
		self.dpb.push(DpbEntry {
			frame: None,
			poc,
			short_ref: true,
			long_ref: false,
			output,
			sequence: 0,
			pts,
			keyframe: is_irap(t),
			crop: sps.crop,
		});
		let dpb_index = self.dpb.len() - 1;
		// C.5.2.2 bumping before decoding the current picture.
		loop {
			let waiting = self.dpb.iter().filter(|e| e.output && e.frame.is_some()).count();
			let full = self.dpb.len() > sps.max_dec_pic_buffering.max(1) as usize;
			if waiting > sps.max_num_reorder as usize || (full && waiting > 0) {
				if !self.bump() {
					break;
				}
			} else {
				break;
			}
		}
		let dpb_index = self.dpb.iter().position(|e| e.frame.is_none()).unwrap_or(dpb_index);
		// Other pictures can read this one while it decodes unless it is a
		// sub-layer non-reference picture of the highest sub-layer, which no
		// picture references.
		let referenced = !(slice::is_sub_layer_non_ref(t) && u32::from(tid) + 1 >= sps.max_sub_layers);
		let live = (self.pool.is_some() && referenced).then(|| self.live_frame(&sps, poc));
		let handle = Arc::new(RefPic::pending(
			poc,
			sps.height as usize,
			sps.ctb_height as usize,
			sps.log2_ctb_size,
			live,
		));
		self.cur = Some(Box::new(Current {
			sps,
			pps,
			layout,
			poc,
			handle,
			slices: Vec::new(),
			header: sh.clone(),
			dpb_index,
			refs_of_slice: None,
			st_before,
			st_after,
			lt_curr,
			error: false,
		}));
		Ok(())
	}

	/// A buffer for publishing a picture's rows while it decodes, recycled
	/// from earlier pictures when the geometry matches. Readers only see
	/// rows after they are published, so its old contents never matter.
	fn live_frame(&self, sps: &Sps, poc: i32) -> Box<Frame> {
		let reuse = {
			let mut pool = self.live_pool.lock().unwrap_or_else(|e| e.into_inner());
			let k = pool.iter().position(|f| {
				f.planes[0].width == sps.width as usize
					&& f.planes[0].height == sps.height as usize
					&& f.log2_ctb_size == sps.log2_ctb_size
					&& f.chroma_format == sps.chroma_format_idc
			});
			k.map(|k| Box::new(pool.swap_remove(k)))
		};
		let mut f = reuse.unwrap_or_else(|| Box::new(blank_frame(sps, 0, 0)));
		f.poc = poc;
		f.bit_depth = sps.bit_depth;
		f.slice_refs.clear();
		f
	}

	/// Stores the current picture in the DPB and starts its picture job
	/// (on a frame thread, or here with one thread).
	fn finish_picture(&mut self) {
		let Some(cur) = self.cur.take() else { return };
		let sps = cur.sps.clone();
		let handle = cur.handle.clone();
		let mut crop = None;
		if let Some(e) = self.dpb.get_mut(cur.dpb_index) {
			e.frame = Some(handle.clone());
			// Frame threads also prepare the output copy.
			if e.output && self.pool.is_some() {
				crop = Some(e.crop);
			}
		}
		let job = PictureJob {
			sps: cur.sps,
			pps: cur.pps,
			layout: cur.layout,
			poc: cur.poc,
			slices: cur.slices,
			out: handle.clone(),
			live_pool: self.live_pool.clone(),
			wpp_threads: self.wpp_threads,
			workspaces: self.workspaces.clone(),
			crop,
		};
		match self.pool.clone() {
			Some(pool) => {
				// At most `threads` pictures in flight (FFmpeg keeps one per
				// frame thread): wait for the oldest before starting another.
				// Jobs only wait on earlier jobs and the pool is FIFO, so this
				// can't deadlock.
				while self.in_flight.len() >= self.threads {
					self.retire_one();
				}
				self.in_flight.push_back(handle);
				pool.submit(move || job.run());
			}
			None => {
				job.run();
				self.damaged_slices += handle.damaged.load(Ordering::Acquire);
			}
		}
		// Output now if allowed (C.5.2.3).
		loop {
			let waiting = self.dpb.iter().filter(|e| e.output && e.frame.is_some()).count();
			if waiting > sps.max_num_reorder as usize {
				if !self.bump() {
					break;
				}
			} else {
				break;
			}
		}
	}

	/// Outputs the waiting picture with the smallest POC. Returns false if
	/// none is waiting.
	fn bump(&mut self) -> bool {
		let Some(k) = self
			.dpb
			.iter()
			.enumerate()
			.filter(|(_, e)| e.output && e.frame.is_some())
			.min_by_key(|(_, e)| e.poc)
			.map(|(k, _)| k)
		else {
			return false;
		};
		let e = &mut self.dpb[k];
		e.output = false;
		let f = e.frame.clone().expect("filtered");
		self.pending_out.push_back(PendingOutput {
			at: self.packets,
			frame: f,
			crop: e.crop,
			pts: e.pts,
			keyframe: e.keyframe,
		});
		if !e.short_ref && !e.long_ref {
			self.dpb.remove(k);
		}
		true
	}
}

/// Row threads per picture that `threads` threads get by default: frame
/// threads alone below 8 (on 2 vCPUs, adding rows was slower), plus 2 WPP
/// row threads from 8 up (7–15% faster than frame threads alone on a
/// 14-core Apple-silicon Mac, `docs/THREADING.md`). The rows only run on
/// WPP streams and give each in-flight picture a second thread when frame
/// threads wait on references.
const fn default_rows(threads: usize) -> usize {
	if threads >= 8 { 2 } else { 1 }
}

/// How `threads` are used: (pictures in flight, row threads per picture).
/// `threads` frame threads, as FFmpeg, plus [`default_rows`].
/// `RMP_HEVC_ROWS` sets the row threads per picture; `RMP_HEVC_FRAMES`
/// overrides the pictures in flight.
fn split_threads(threads: usize) -> (usize, usize) {
	let env = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok());
	let frames = env("RMP_HEVC_FRAMES").filter(|&n| n > 0).unwrap_or(threads);
	let rows = env("RMP_HEVC_ROWS").filter(|&n| n > 0).unwrap_or(default_rows(threads));
	(frames, rows)
}

/// Whether `data` is a sequence of `n`-byte length-prefixed NAL units
/// (a length prefix such as 00 00 01 xx can look like an Annex B start code).
fn tiles_exactly(data: &[u8], n: usize) -> bool {
	let mut at = 0;
	while at + n <= data.len() {
		let len = data[at..at + n].iter().fold(0usize, |a, &b| (a << 8) | usize::from(b));
		at += n + len;
	}
	at == data.len()
}

/// An all-`fill` picture with no motion (intra everywhere).
fn blank_frame(sps: &Sps, poc: i32, fill: u16) -> Frame {
	let (w, h) = (sps.width as usize, sps.height as usize);
	let w4 = w.div_ceil(4);
	Frame {
		planes: picture_planes(sps, fill),
		poc,
		mvf: vec![MvField::default(); w4 * h.div_ceil(4)],
		w4,
		ctb_slice: vec![0; (sps.ctb_width * sps.ctb_height) as usize],
		slice_refs: vec![RefInfo::default()],
		log2_ctb_size: sps.log2_ctb_size,
		ctb_width: sps.ctb_width as usize,
		bit_depth: sps.bit_depth,
		chroma_format: sps.chroma_format_idc,
	}
}

fn missing_frame(sps: &Sps, poc: i32) -> FrameRef {
	RefPic::complete(blank_frame(sps, poc, 1u16 << (sps.bit_depth - 1)))
}

/// Everything needed to decode one picture's slices, independent of the
/// decoder state, so it can run on a frame thread.
struct PictureJob {
	sps: Arc<Sps>,
	pps: Arc<Pps>,
	layout: Arc<Layout>,
	poc: i32,
	slices: Vec<SliceJob>,
	out: FrameRef,
	live_pool: LivePool,
	/// Threads for the rows of a WPP slice segment.
	wpp_threads: usize,
	workspaces: Arc<Workspaces>,
	/// Crop the output copy (the picture will be output).
	crop: Option<[u32; 4]>,
}

/// Poisons the picture if its job unwinds, so that its readers fail
/// instead of waiting forever.
struct PoisonGuard(Option<FrameRef>);

impl Drop for PoisonGuard {
	fn drop(&mut self) {
		if let Some(f) = self.0.take() {
			f.poison();
		}
	}
}

impl PictureJob {
	fn run(self) {
		let mut guard = PoisonGuard(Some(self.out.clone()));
		let mut pic = PicState::new(self.sps, self.pps, self.layout, self.poc);
		if self.out.publishes() {
			pic.out = Some(self.out.clone());
		}
		for s in &self.slices {
			if let Err(e) = decode_slice(&mut pic, s, self.wpp_threads, &self.workspaces) {
				if std::env::var_os("RMP_HEVC_DEBUG").is_some() {
					eprintln!("hevc: {}", e.0);
				}
				self.out.damaged.fetch_add(1, Ordering::AcqRel);
			}
		}
		pic.advance(true);
		pic.out = None;
		let frame = pic.into_frame();
		if let Some(crop) = self.crop {
			*self.out.output.lock().unwrap_or_else(|e| e.into_inner()) = Some(crop_samples(&frame, crop));
		}
		if let Some(live) = self.out.finish(frame) {
			self.live_pool.lock().unwrap_or_else(|e| e.into_inner()).push(*live);
		}
		guard.0 = None;
	}
}

/// Decodes one slice segment into the picture.
fn decode_slice(
	pic: &mut PicState,
	s: &SliceJob,
	wpp_threads: usize,
	workspaces: &Workspaces,
) -> Result<(), HevcError> {
	// A segment starting on an already decoded CTB (a duplicate or
	// overlapping slice) is dropped, as FFmpeg does, rather than
	// overwriting good samples.
	if pic.ctb_addr.get(s.header.segment_address as usize) != Some(&u32::MAX) {
		return Err(HevcError("hevc: overlapping slice segment"));
	}
	if let Some(rows) = wpp::rows_of(pic, s) {
		return wpp::decode_rows(pic, s, rows, wpp_threads, workspaces).map_err(|e| HevcError(e.0));
	}
	let mut sd = SliceDecoder::new(pic, &s.header, &s.refs, &s.rbsp);
	sd.decode().map_err(|e| HevcError(e.0))
}

/// Removes emulation prevention bytes (00 00 03 -> 00 00) and returns the
/// RBSP with the escaped positions of the removed bytes.
fn unescape_with_positions(data: &[u8]) -> (Vec<u8>, Vec<u32>) {
	let mut out = Vec::with_capacity(data.len());
	let mut esc = Vec::new();
	let mut zeros = 0;
	for (i, &b) in data.iter().enumerate() {
		if zeros >= 2 && b == 3 {
			esc.push(i as u32);
			zeros = 0;
			continue;
		}
		zeros = if b == 0 { zeros + 1 } else { 0 };
		out.push(b);
	}
	(out, esc)
}

/// Reference picture lists of a slice (8.3.4).
fn build_lists(cur: &Current, sh: &SliceHeader) -> Result<SliceRefs, HevcError> {
	let mut refs = SliceRefs {
		list: [Vec::new(), Vec::new()],
		info: RefInfo::default(),
		collocated: None,
		no_backward_pred: true,
	};
	if sh.slice_type == SliceType::I {
		return Ok(refs);
	}
	let total = cur.st_before.len() + cur.st_after.len() + cur.lt_curr.len();
	if total == 0 {
		return Err(HevcError("hevc: inter slice without references"));
	}
	let lists = if sh.slice_type == SliceType::B { 2 } else { 1 };
	for l in 0..lists {
		let order: [(&Vec<Option<FrameRef>>, bool); 3] = if l == 0 {
			[(&cur.st_before, false), (&cur.st_after, false), (&cur.lt_curr, true)]
		} else {
			[(&cur.st_after, false), (&cur.st_before, false), (&cur.lt_curr, true)]
		};
		let n = sh.num_ref_idx[l] as usize;
		let len = n.max(total);
		let mut temp: Vec<(Option<FrameRef>, bool)> = Vec::with_capacity(len);
		while temp.len() < len {
			for (v, lt) in order {
				for f in v {
					if temp.len() < len {
						temp.push((f.clone(), lt));
					}
				}
			}
		}
		for i in 0..n {
			let idx = match &sh.list_entry[l] {
				Some(e) => e[i] as usize,
				None => i,
			};
			let (f, lt) = temp.get(idx).cloned().ok_or(HevcError("hevc: invalid list entry"))?;
			if let Some(fr) = &f {
				refs.info.poc[l][i] = fr.poc;
				if fr.poc > cur.poc {
					refs.no_backward_pred = false;
				}
			}
			refs.info.long_term[l][i] = lt;
			refs.list[l].push(f);
		}
	}
	if sh.temporal_mvp_enabled {
		let l = if sh.slice_type == SliceType::B && !sh.collocated_from_l0 { 1 } else { 0 };
		refs.collocated = refs.list[l].get(sh.collocated_ref_idx as usize).cloned().flatten();
	}
	Ok(refs)
}

/// Crops a frame into packed planar bytes.
fn crop_output(f: &Frame, crop: [u32; 4], pts: Option<Time>, keyframe: bool) -> OutputPicture {
	output_picture(f, crop, crop_samples(f, crop), pts, keyframe)
}

fn output_picture(
	f: &Frame,
	crop: [u32; 4],
	data: Vec<u8>,
	pts: Option<Time>,
	keyframe: bool,
) -> OutputPicture {
	let [l, r, t, b] = crop.map(|v| v as usize);
	let w = f.planes[0].width - l - r;
	let h = f.planes[0].height - t - b;
	OutputPicture {
		width: w as u32,
		height: h as u32,
		bit_depth: f.bit_depth,
		chroma_format: f.chroma_format,
		data,
		pts,
		keyframe,
	}
}

/// The cropped samples of a frame, packed planar.
fn crop_samples(f: &Frame, crop: [u32; 4]) -> Vec<u8> {
	// crop: left, right, top, bottom in luma samples.
	let [l, r, t, b] = crop.map(|v| v as usize);
	let w = f.planes[0].width - l - r;
	let h = f.planes[0].height - t - b;
	let wide = f.bit_depth > 8;
	// Chroma subsampling shifts (the crop is in whole chroma samples).
	let (hs, vs) = match f.chroma_format {
		1 => (1, 1),
		2 => (1, 0),
		_ => (0, 0),
	};
	let ncomp = if f.chroma_format == 0 { 1 } else { 3 };
	let samples = w * h + (ncomp - 1) * (w >> hs) * (h >> vs);
	let mut data = Vec::with_capacity(samples * if wide { 2 } else { 1 });
	for (c, p) in f.planes.iter().enumerate().take(ncomp) {
		let (x0, y0, pw, ph) = if c == 0 { (l, t, w, h) } else { (l >> hs, t >> vs, w >> hs, h >> vs) };
		for y in 0..ph {
			let at = p.idx(x0 as isize, (y0 + y) as isize);
			let row = &p.data[at..at + pw];
			if wide {
				for &v in row {
					data.extend_from_slice(&v.to_le_bytes());
				}
			} else {
				data.extend(row.iter().map(|&v| v as u8));
			}
		}
	}
	data
}

#[cfg(test)]
mod tests {
	use super::default_rows;

	#[test]
	fn row_threads_start_at_eight_threads() {
		assert_eq!([1, 2, 4, 7].map(default_rows), [1; 4]);
		assert_eq!([8, 14, 16].map(default_rows), [2; 3]);
	}
}
