//! Decoded picture buffer: picture order count (§8.2.1), reference list
//! construction and modification (§8.2.4), reference marking (§8.2.5) and
//! output ordering (C.4). FFmpeg: `h264_picture.c`, `h264_refs.c`,
//! `h2645_parse`/`h264_poc.c`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use super::params::Sps;
use super::recon::{Pixel, Plane};
use super::slice::{Mmco, RefModification, SliceHeader, SliceType};
use crate::core::{Pooled, Time};

/// Motion of one macroblock, kept with its picture for direct prediction
/// in later B pictures and for deblocking.
#[derive(Debug, Clone, Copy, Default)]
pub struct MbMotion {
	/// Motion vectors per list and raster 4x4 block, quarter-sample units
	/// (of a field for field macroblocks).
	pub mv: [[[i16; 2]; 16]; 2],
	/// Reference index per list and 8x8 block (-1 = list unused).
	pub ref_idx: [[i8; 4]; 2],
	/// [`RefPic::key`] of the referenced frame or field per list and 8x8
	/// block (0 = none).
	pub ref_id: [[u32; 4]; 2],
	pub intra: bool,
	/// Decoded as a field macroblock (field picture or MBAFF field pair).
	pub field: bool,
}

/// A decoded frame or field pair (its bookkeeping; the samples and motion
/// live in [`PicData`], which may still be being decoded by another
/// thread). Picture order counts live in the DPB and in [`RefPic`].
#[derive(Debug, Clone)]
pub struct Picture<T: Pixel = u8> {
	/// Unique, non-zero identity (decode counter, below 2^30).
	pub id: u32,
	pub data: Arc<PicData<T>>,
	pub pts: Option<Time>,
	pub keyframe: bool,
	/// Frames inserted for a frame_num gap; never output.
	pub non_existing: bool,
	/// Coded as field pictures (its macroblock motion is stored per field,
	/// interleaved by macroblock row).
	pub field_coded: bool,
}

/// Borrowed view of a picture being decoded, for [`PicData::publish`].
#[derive(Clone, Copy)]
pub struct PlanesRef<'a, T> {
	pub y: &'a Plane<T>,
	pub cb: &'a Plane<T>,
	pub cr: &'a Plane<T>,
	pub motion: &'a [MbMotion],
}

/// The samples and motion of one picture.
#[derive(Debug, Clone, Default)]
pub struct PicPlanes<T = u8> {
	pub y: Plane<T>,
	pub cb: Plane<T>,
	pub cr: Plane<T>,
	pub motion: Pooled<MbMotion>,
}

/// A picture's content, shared between the thread decoding it and the
/// threads predicting from it (FFmpeg's `ThreadFrame`).
///
/// Progress counts *deblocked macroblock rows* `d`: once it is `d`, luma
/// rows below `16d - 3` and chroma rows below `8d - 2` are final (deblocking
/// row `d` may still change the 3 luma / 1 chroma rows above its top edge;
/// MBAFF frames advance by pairs, with wider margins), and so is the
/// motion of macroblock rows below `d`. `mb_h + 1` means complete. Until then, finished rows are copied into `live` as they are
/// reported; afterwards everything is read lock-free from `done`.
#[derive(Debug)]
pub struct PicData<T = u8> {
	mb_h: u32,
	/// Rows are only published at completion (field pairs, whose rows are
	/// not finished top to bottom).
	whole: bool,
	/// Sample rows above a deblocked macroblock row that deblocking the
	/// next one may still change (luma, chroma): 3 and 2 for frames, 6 and
	/// 3 for MBAFF frames (field macroblock edges reach three field rows
	/// into the pair above).
	margin: (u32, u32),
	/// Chroma rows per macroblock row (8 or 16).
	mb_ch: u32,
	progress: crate::codecs::threading::Progress,
	live: RwLock<PicPlanes<T>>,
	done: OnceLock<PicPlanes<T>>,
	/// Slices of this picture that failed to decode.
	pub damaged: AtomicU32,
	/// Set if the decoding thread died: readers then fail loudly instead of
	/// waiting forever.
	poisoned: AtomicBool,
}

impl<T: Pixel> PicData<T> {
	/// A picture that is already complete.
	pub fn complete(planes: PicPlanes<T>) -> Arc<Self> {
		let mb_h = (planes.y.height / 16) as u32;
		let mb_ch = (planes.cb.height as u32 * 16).checked_div(planes.y.height as u32).unwrap_or(8);
		let data = Self::pending(mb_h, mb_ch, false, false, PicPlanes::default());
		let _ = data.done.set(planes);
		data.progress.report(mb_h + 1);
		Arc::new(data)
	}

	/// A picture to be decoded. `live` holds the buffers rows are published
	/// into (empty when nobody reads the picture before it is complete).
	pub fn pending(mb_h: u32, mb_ch: u32, whole: bool, mbaff: bool, live: PicPlanes<T>) -> Self {
		Self {
			mb_h,
			whole,
			margin: if mbaff { (6, 3) } else { (3, 2) },
			mb_ch,
			progress: crate::codecs::threading::Progress::new(0),
			live: RwLock::new(live),
			done: OnceLock::new(),
			damaged: AtomicU32::new(0),
			poisoned: AtomicBool::new(false),
		}
	}

	pub fn is_done(&self) -> bool {
		self.done.get().is_some()
	}

	/// The finished picture, without waiting (`None` while it is decoded).
	pub fn done(&self) -> Option<&PicPlanes<T>> {
		self.done.get()
	}

	/// Progress needed before luma row `y` can be read.
	pub fn need_luma(&self, y: usize) -> u32 {
		if self.whole {
			return self.mb_h + 1;
		}
		((y as u32 + self.margin.0) / 16 + 1).min(self.mb_h + 1)
	}

	/// Progress needed before chroma row `y` can be read.
	pub fn need_chroma(&self, y: usize) -> u32 {
		if self.whole {
			return self.mb_h + 1;
		}
		((y as u32 + self.margin.1) / self.mb_ch + 1).min(self.mb_h + 1)
	}

	/// Progress needed before the motion of macroblock row `row` can be read.
	pub fn need_motion(&self, row: usize) -> u32 {
		if self.whole {
			return self.mb_h + 1;
		}
		(row as u32 + 1).min(self.mb_h + 1)
	}

	pub fn progress(&self) -> u32 {
		self.progress.get()
	}

	/// Runs `f` on the picture once progress has reached `need`.
	#[inline]
	pub fn with<R>(&self, need: u32, f: impl FnOnce(&PicPlanes<T>) -> R) -> R {
		if let Some(done) = self.done.get() {
			return f(done);
		}
		self.progress.wait_for(need);
		if let Some(done) = self.done.get() {
			return f(done);
		}
		self.check_poison();
		let live = self.live.read().unwrap_or_else(|e| e.into_inner());
		if live.y.data.is_empty() {
			// Completed (and `live` released) between the two checks.
			drop(live);
			return f(self.wait_done());
		}
		f(&live)
	}

	/// The complete picture, waiting for it if necessary.
	pub fn wait_done(&self) -> &PicPlanes<T> {
		self.progress.wait_for(self.mb_h + 1);
		self.check_poison();
		match self.done.get() {
			Some(d) => d,
			None => panic!("h264: picture reported complete without data"),
		}
	}

	fn check_poison(&self) {
		if self.poisoned.load(Ordering::Acquire) {
			panic!("h264: the thread decoding a reference picture panicked");
		}
	}

	/// Copies rows that became final into `live` and raises progress to
	/// `rows` deblocked macroblock rows.
	pub fn publish(&self, src: PlanesRef<'_, T>, from: u32, rows: u32) {
		if rows <= from {
			return;
		}
		{
			let mut live = self.live.write().unwrap_or_else(|e| e.into_inner());
			let (lm, cm) = (self.margin.0 as usize, self.margin.1 as usize);
			let luma = |d: u32| (16 * d as usize).saturating_sub(lm);
			let ch = self.mb_ch as usize;
			let chroma = |d: u32| (ch * d as usize).saturating_sub(cm);
			let copy = |dst: &mut Plane<T>, s: &Plane<T>, a: usize, b: usize| {
				let (a, b) = (a * s.width, (b * s.width).min(s.data.len()));
				if a < b && dst.data.len() >= b {
					dst.data[a..b].copy_from_slice(&s.data[a..b]);
				}
			};
			copy(&mut live.y, src.y, luma(from), luma(rows));
			copy(&mut live.cb, src.cb, chroma(from), chroma(rows));
			copy(&mut live.cr, src.cr, chroma(from), chroma(rows));
			let mb_w = src.y.width / 16;
			let (a, b) = (from as usize * mb_w, (rows as usize * mb_w).min(src.motion.len()));
			if a < b && live.motion.len() >= b {
				live.motion[a..b].copy_from_slice(&src.motion[a..b]);
			}
		}
		self.progress.report(rows);
	}

	/// Stores the final picture and wakes every reader.
	pub fn finish(&self, planes: PicPlanes<T>) {
		let _ = self.done.set(planes);
		// `live` is no longer needed: readers go to `done` from now on.
		*self.live.write().unwrap_or_else(|e| e.into_inner()) = PicPlanes::default();
		self.progress.report(self.mb_h + 1);
	}

	/// Marks the picture as failed (its decoder panicked) and releases the
	/// waiters, which then panic too rather than hang.
	pub fn poison(&self) {
		self.poisoned.store(true, Ordering::Release);
		self.progress.report(self.mb_h + 1);
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marking {
	Unused,
	Short,
	Long(u32),
}

/// One frame (or field pair, or lone field) in the DPB.
#[derive(Debug, Clone)]
struct Entry<T: Pixel> {
	pic: Arc<Picture<T>>,
	/// Marking of the top and bottom field; a frame marks both alike.
	marking: [Marking; 2],
	frame_num: u32,
	/// Top and bottom field order counts; `i32::MAX` for a field not
	/// decoded (yet).
	field_poc: [i32; 2],
	needed_for_output: bool,
	/// Both fields (or the frame) are decoded. A first field waits here
	/// for its pair before it can be output.
	complete: bool,
}

impl<T: Pixel> Entry<T> {
	/// PicOrderCnt of the frame or pair: the smaller field count (FFmpeg's
	/// `poc`, also for a pair of which only one field is decoded).
	fn poc(&self) -> i32 {
		self.field_poc[0].min(self.field_poc[1])
	}

	fn is_ref(&self) -> bool {
		self.marking.iter().any(|&m| m != Marking::Unused)
	}

	fn any_short(&self) -> bool {
		self.marking.contains(&Marking::Short)
	}

	/// The long-term frame index of either field.
	fn long_idx(&self) -> Option<u32> {
		self.marking.iter().find_map(|m| if let Marking::Long(i) = *m { Some(i) } else { None })
	}

	fn as_frame(&self, long: bool) -> RefPic<T> {
		RefPic { pic: self.pic.clone(), long, parity: None, poc: self.poc(), field_poc: self.field_poc }
	}

	fn as_field(&self, parity: usize, long: bool) -> RefPic<T> {
		RefPic {
			pic: self.pic.clone(),
			long,
			parity: Some(parity),
			poc: self.field_poc[parity],
			field_poc: self.field_poc,
		}
	}
}

/// A reference frame or field as seen by one slice.
#[derive(Debug, Clone)]
pub struct RefPic<T: Pixel = u8> {
	pub pic: Arc<Picture<T>>,
	pub long: bool,
	/// `None` for a frame, `Some(parity)` for one of its fields.
	pub parity: Option<usize>,
	/// PicOrderCnt of the frame (smaller field count) or of the field.
	pub poc: i32,
	/// Both field order counts of the frame.
	pub field_poc: [i32; 2],
}

impl<T: Pixel> RefPic<T> {
	/// Identity of a reference frame or field, for comparing references
	/// (deblocking, direct prediction): the picture id and the structure,
	/// as in FFmpeg (`4 * id + reference`): 1 top, 2 bottom, 3 frame.
	#[inline]
	pub fn key(&self) -> u32 {
		(self.pic.id << 2) | self.parity.map_or(3, |p| 1 + p as u32)
	}

	/// One field of a reference frame (MBAFF field macroblocks).
	pub fn field(&self, parity: usize) -> RefPic<T> {
		RefPic {
			pic: self.pic.clone(),
			long: self.long,
			parity: Some(parity),
			poc: self.field_poc[parity],
			field_poc: self.field_poc,
		}
	}
}

/// A reference picture list (entries may be missing in damaged streams).
pub type RefList<T> = Vec<Option<RefPic<T>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BadRefs(pub &'static str);

/// Picture-order-count state carried between pictures (8.2.1).
#[derive(Debug, Clone, Default)]
pub struct PocState {
	prev_msb: i32,
	prev_lsb: i32,
	prev_frame_num_offset: i32,
	prev_frame_num: u32,
}

impl PocState {
	/// Computes the POC of the current picture and returns `(top, bottom)`
	/// field order counts. For a field picture only its own field's count
	/// is derived; it is returned in both places.
	pub fn compute(&mut self, sps: &Sps, h: &SliceHeader, prev_had_mmco5: bool) -> (i32, i32) {
		let max_frame_num = sps.max_frame_num() as i32;
		let frame_num = h.frame_num as i32;
		let parity = h.parity();
		let one = |v: i32| (v, v);
		match sps.poc_type {
			0 => {
				if h.idr {
					self.prev_msb = 0;
					self.prev_lsb = 0;
				}
				let max_lsb = 1i32 << sps.log2_max_poc_lsb;
				let lsb = h.poc_lsb as i32;
				let msb = if lsb < self.prev_lsb && self.prev_lsb - lsb >= max_lsb / 2 {
					self.prev_msb + max_lsb
				} else if lsb > self.prev_lsb && lsb - self.prev_lsb > max_lsb / 2 {
					self.prev_msb - max_lsb
				} else {
					self.prev_msb
				};
				if h.nal_ref_idc != 0 {
					self.prev_msb = msb;
					self.prev_lsb = lsb;
				}
				let top = msb.wrapping_add(lsb);
				match parity {
					Some(_) => one(top),
					None => (top, top.wrapping_add(h.delta_poc_bottom)),
				}
			}
			_ => {
				let prev_offset = if prev_had_mmco5 { 0 } else { self.prev_frame_num_offset };
				let offset = if h.idr {
					0
				} else if self.prev_frame_num > h.frame_num {
					prev_offset.wrapping_add(max_frame_num)
				} else {
					prev_offset
				};
				self.prev_frame_num_offset = offset;
				self.prev_frame_num = h.frame_num;
				if sps.poc_type == 2 {
					let t = if h.idr {
						0
					} else {
						let t = offset.wrapping_add(frame_num).wrapping_mul(2);
						if h.nal_ref_idc == 0 { t.wrapping_sub(1) } else { t }
					};
					return (t, t);
				}
				let cycle = sps.offset_for_ref_frame.len() as i32;
				let mut abs = if cycle != 0 { offset.wrapping_add(frame_num) } else { 0 };
				if h.nal_ref_idc == 0 && abs > 0 {
					abs -= 1;
				}
				let mut expected = 0i32;
				if abs > 0 {
					let per_cycle: i32 =
						sps.offset_for_ref_frame.iter().fold(0i32, |a, &b| a.wrapping_add(b));
					let cycle_cnt = (abs - 1) / cycle;
					let in_cycle = ((abs - 1) % cycle) as usize;
					expected = cycle_cnt.wrapping_mul(per_cycle);
					for &o in &sps.offset_for_ref_frame[..=in_cycle] {
						expected = expected.wrapping_add(o);
					}
				}
				if h.nal_ref_idc == 0 {
					expected = expected.wrapping_add(sps.offset_for_non_ref_pic);
				}
				match parity {
					Some(0) => one(expected.wrapping_add(h.delta_poc[0])),
					Some(_) => one(
						expected.wrapping_add(sps.offset_for_top_to_bottom_field).wrapping_add(h.delta_poc[0]),
					),
					None => {
						let top = expected.wrapping_add(h.delta_poc[0]);
						let bottom =
							top.wrapping_add(sps.offset_for_top_to_bottom_field).wrapping_add(h.delta_poc[1]);
						(top, bottom)
					}
				}
			}
		}
	}

	/// Applies the effect of memory_management_control_operation 5 on the
	/// state used by the next picture.
	pub fn after_mmco5(&mut self, top_after_reset: i32) {
		self.prev_msb = 0;
		self.prev_lsb = top_after_reset;
		self.prev_frame_num_offset = 0;
		self.prev_frame_num = 0;
	}

	pub fn set_prev_frame_num(&mut self, frame_num: u32) {
		self.prev_frame_num = frame_num;
	}
}

#[derive(Debug, Clone)]
pub struct Dpb<T: Pixel = u8> {
	entries: Vec<Entry<T>>,
	max_long_term_idx: Option<u32>,
	/// Pictures ready for output, in output order.
	pub output: Vec<Arc<Picture<T>>>,
}

impl<T: Pixel> Default for Dpb<T> {
	fn default() -> Self {
		Self { entries: Vec::new(), max_long_term_idx: None, output: Vec::new() }
	}
}

/// The field a field PicNum (or LongTermPicNum) names, seen from a field
/// of parity `cur`: `(FrameNumWrap or LongTermFrameIdx, parity)`. Odd
/// numbers are the same parity (8-28, 8-29).
fn split_pic_num(n: i64, cur: usize) -> (i64, usize) {
	(n.div_euclid(2), if n.rem_euclid(2) == 1 { cur } else { 1 - cur })
}

impl<T: Pixel> Dpb<T> {
	pub fn clear(&mut self) {
		self.entries.clear();
		self.output.clear();
		self.max_long_term_idx = None;
	}

	/// FrameNumWrap (8-27).
	fn frame_num_wrap(e: &Entry<T>, cur_frame_num: u32, max_frame_num: u32) -> i32 {
		if e.frame_num > cur_frame_num {
			e.frame_num as i32 - max_frame_num as i32
		} else {
			e.frame_num as i32
		}
	}

	fn refs(&self) -> impl Iterator<Item = &Entry<T>> {
		self.entries.iter().filter(|e| e.is_ref())
	}

	/// The most recent reference picture (used to conceal missing frames).
	pub fn latest_ref(&self) -> Option<Arc<Picture<T>>> {
		self.entries.iter().rev().find(|e| e.is_ref()).map(|e| e.pic.clone())
	}

	/// Fields of `frames`, alternating between the parity of the current
	/// field and the other one, each list in its own order, as long as
	/// fields with `marked` remain (8.2.4.2.5).
	fn alternate(
		frames: &[&Entry<T>],
		parity: usize,
		long: bool,
		marked: impl Fn(Marking) -> bool,
		out: &mut Vec<RefPic<T>>,
	) {
		let n = frames.len();
		let (mut same, mut other) = (0, 0);
		loop {
			while same < n && !marked(frames[same].marking[parity]) {
				same += 1;
			}
			while other < n && !marked(frames[other].marking[1 - parity]) {
				other += 1;
			}
			if same >= n && other >= n {
				break;
			}
			if same < n {
				out.push(frames[same].as_field(parity, long));
				same += 1;
			}
			if other < n {
				out.push(frames[other].as_field(1 - parity, long));
				other += 1;
			}
		}
	}

	/// Builds RefPicList0/1 for a slice (8.2.4.2-3). `cur_poc` is the POC
	/// of the current frame or field.
	pub fn ref_lists(
		&self,
		h: &SliceHeader,
		sps: &Sps,
		cur_poc: i32,
	) -> Result<[Vec<Option<RefPic<T>>>; 2], BadRefs> {
		let max_frame_num = sps.max_frame_num();
		let parity = h.parity();
		let both = |e: &Entry<T>, m: Marking| e.marking == [m, m];
		// Frames for frame decoding: both fields marked alike. Fields: any
		// field marked (including the first field of the current pair).
		let short: Vec<&Entry<T>> = self
			.refs()
			.filter(|e| if parity.is_none() { both(e, Marking::Short) } else { e.any_short() })
			.collect();
		let mut long: Vec<(&Entry<T>, u32)> = self
			.refs()
			.filter_map(|e| {
				let idx = e.long_idx()?;
				(parity.is_some() || both(e, Marking::Long(idx))).then_some((e, idx))
			})
			.collect();
		long.sort_by_key(|&(_, i)| i);
		let long: Vec<&Entry<T>> = long.into_iter().map(|(e, _)| e).collect();
		let ordered: [Vec<&Entry<T>>; 2] = match h.slice_type {
			SliceType::I => return Ok([Vec::new(), Vec::new()]),
			SliceType::P => {
				let mut s = short;
				s.sort_by_key(|e| std::cmp::Reverse(Self::frame_num_wrap(e, h.frame_num, max_frame_num)));
				[s, Vec::new()]
			}
			SliceType::B => {
				// Equal counts go first in list 0 (FFmpeg; the first field of
				// the current pair, seen from its second field).
				let mut before: Vec<&Entry<T>> =
					short.iter().copied().filter(|e| e.poc() <= cur_poc).collect();
				let mut after: Vec<&Entry<T>> =
					short.iter().copied().filter(|e| e.poc() > cur_poc).collect();
				before.sort_by_key(|e| std::cmp::Reverse(e.poc()));
				after.sort_by_key(|e| e.poc());
				let l0 = before.iter().chain(after.iter()).copied().collect();
				let l1 = after.iter().chain(before.iter()).copied().collect();
				[l0, l1]
			}
		};
		let lists_used = if h.slice_type == SliceType::B { 2 } else { 1 };
		let mut init: [Vec<RefPic<T>>; 2] = [Vec::new(), Vec::new()];
		for (list, frames) in ordered.iter().enumerate().take(lists_used) {
			let out = &mut init[list];
			match parity {
				None => {
					out.extend(frames.iter().map(|e| e.as_frame(false)));
					out.extend(long.iter().map(|e| e.as_frame(true)));
				}
				Some(p) => {
					Self::alternate(frames, p, false, |m| m == Marking::Short, out);
					Self::alternate(&long, p, true, |m| matches!(m, Marking::Long(_)), out);
				}
			}
		}
		if h.slice_type == SliceType::B
			&& init[1].len() > 1
			&& init[0].len() == init[1].len()
			&& init[0].iter().zip(&init[1]).all(|(a, b)| a.pic.id == b.pic.id && a.parity == b.parity)
		{
			init[1].swap(0, 1);
		}
		let mut lists: [Vec<Option<RefPic<T>>>; 2] = [Vec::new(), Vec::new()];
		for l in 0..2 {
			let n = h.num_ref_idx[l];
			let mut list: Vec<Option<RefPic<T>>> = init[l].iter().take(n).cloned().map(Some).collect();
			list.resize(n, None);
			self.modify(&mut list, &h.ref_mods[l], h, max_frame_num)?;
			lists[l] = list;
		}
		Ok(lists)
	}

	/// Reference picture list modification (8.2.4.3).
	fn modify(
		&self,
		list: &mut Vec<Option<RefPic<T>>>,
		mods: &[RefModification],
		h: &SliceHeader,
		max_frame_num: u32,
	) -> Result<(), BadRefs> {
		let n = list.len();
		let parity = h.parity();
		let curr = h.curr_pic_num();
		let max = i64::from(max_frame_num) * if parity.is_some() { 2 } else { 1 };
		let fnw = |e: &Entry<T>| i64::from(Self::frame_num_wrap(e, h.frame_num, max_frame_num));
		let mut pred = curr;
		for (ref_idx, m) in mods.iter().enumerate() {
			if ref_idx >= n {
				return Err(BadRefs("too many reference list modifications"));
			}
			let found = match *m {
				RefModification::ShortTerm { add, abs_diff } => {
					let diff = i64::from(abs_diff) + 1;
					if diff > max {
						return Err(BadRefs("invalid abs_diff_pic_num"));
					}
					let mut no_wrap = if add { pred + diff } else { pred - diff };
					if no_wrap < 0 {
						no_wrap += max;
					} else if no_wrap >= max {
						no_wrap -= max;
					}
					pred = no_wrap;
					let pic_num = if no_wrap > curr { no_wrap - max } else { no_wrap };
					match parity {
						None => self
							.refs()
							.find(|e| e.marking == [Marking::Short; 2] && fnw(e) == pic_num)
							.map(|e| e.as_frame(false)),
						Some(p) => {
							let (wrap, par) = split_pic_num(pic_num, p);
							self
								.refs()
								.find(|e| e.marking[par] == Marking::Short && fnw(e) == wrap)
								.map(|e| e.as_field(par, false))
						}
					}
					.ok_or(BadRefs("reference list modification names a missing picture"))?
				}
				RefModification::LongTerm(num) => match parity {
					None => {
						self.refs().find(|e| e.marking == [Marking::Long(num); 2]).map(|e| e.as_frame(true))
					}
					Some(p) => {
						let (idx, par) = split_pic_num(i64::from(num), p);
						self
							.refs()
							.find(|e| e.marking[par] == Marking::Long(idx as u32))
							.map(|e| e.as_field(par, true))
					}
				}
				.ok_or(BadRefs("reference list modification names a missing long-term picture"))?,
			};
			let (id, par, is_long) = (found.pic.id, found.parity, found.long);
			list.insert(ref_idx, Some(found));
			// Remove the later duplicate of the same frame or field.
			let mut k = ref_idx + 1;
			while k < list.len() {
				if list[k].as_ref().is_some_and(|r| r.pic.id == id && r.parity == par && r.long == is_long)
				{
					list.remove(k);
				} else {
					k += 1;
				}
			}
			list.truncate(n);
			list.resize(n, None);
		}
		Ok(())
	}

	/// Inserts "non-existing" frames for a gap in frame_num (8.2.5.2).
	pub fn fill_frame_num_gap(
		&mut self,
		sps: &Sps,
		prev_ref_frame_num: u32,
		frame_num: u32,
		next_id: &mut u32,
	) {
		let max = sps.max_frame_num();
		let Some(template) = self.entries.iter().rev().find(|e| e.is_ref()).cloned() else {
			return;
		};
		let mut unused = (prev_ref_frame_num + 1) % max;
		let mut guard = 0;
		while unused != frame_num && guard < 16 {
			self.sliding_window(sps, unused, None);
			let mut pic = (*template.pic).clone();
			*next_id = next_picture_id(*next_id);
			pic.id = *next_id;
			pic.non_existing = true;
			self.entries.push(Entry {
				pic: Arc::new(pic),
				marking: [Marking::Short; 2],
				frame_num: unused,
				field_poc: template.field_poc,
				needed_for_output: false,
				complete: true,
			});
			unused = (unused + 1) % max;
			guard += 1;
		}
	}

	/// Sliding window marking (8.2.5.3), never evicting picture `keep`.
	fn sliding_window(&mut self, sps: &Sps, cur_frame_num: u32, keep: Option<u32>) {
		let max_refs = sps.max_num_ref_frames.max(1) as usize;
		let max_frame_num = sps.max_frame_num();
		while self.refs().filter(|e| Some(e.pic.id) != keep).count() >= max_refs {
			let victim = self
				.entries
				.iter_mut()
				.filter(|e| e.any_short() && Some(e.pic.id) != keep)
				.min_by_key(|e| Self::frame_num_wrap(e, cur_frame_num, max_frame_num));
			match victim {
				Some(e) => {
					for m in &mut e.marking {
						if *m == Marking::Short {
							*m = Marking::Unused;
						}
					}
				}
				None => break,
			}
		}
	}

	/// Frees LongTermFrameIdx `idx` for picture `owner` (mmco 3 and 6): a
	/// long-term frame or pair holding it is unmarked, a long-term field
	/// only if it belongs to another frame.
	fn free_long_idx(&mut self, idx: u32, owner: u32, field: bool) {
		for e in &mut self.entries {
			if e.long_idx() == Some(idx) && (!field || e.pic.id != owner) {
				for m in &mut e.marking {
					if matches!(m, Marking::Long(_)) {
						*m = Marking::Unused;
					}
				}
			}
		}
	}

	/// The entry and fields a short-term picNumX names (mmco 1 and 3).
	fn short_target(
		&self,
		h: &SliceHeader,
		pic_num: i64,
		max_frame_num: u32,
	) -> Option<(usize, [bool; 2])> {
		let fnw = |e: &Entry<T>| i64::from(Self::frame_num_wrap(e, h.frame_num, max_frame_num));
		match h.parity() {
			None => self
				.entries
				.iter()
				.position(|e| e.marking == [Marking::Short; 2] && fnw(e) == pic_num)
				.map(|i| (i, [true, true])),
			Some(p) => {
				let (wrap, par) = split_pic_num(pic_num, p);
				self
					.entries
					.iter()
					.position(|e| e.marking[par] == Marking::Short && fnw(e) == wrap)
					.map(|i| (i, [par == 0, par == 1]))
			}
		}
	}

	/// Applies one memory_management_control_operation (8.2.5.4) for the
	/// current picture `cur_id`. Returns `true` for mmco 5.
	fn apply_mmco(
		&mut self,
		op: Mmco,
		h: &SliceHeader,
		sps: &Sps,
		cur_id: u32,
		cur: &mut Marking,
	) -> bool {
		let max_frame_num = sps.max_frame_num();
		let parity = h.parity();
		let field = parity.is_some();
		let curr = h.curr_pic_num();
		match op {
			Mmco::ForgetShort { diff } => {
				if let Some((i, fields)) = self.short_target(h, curr - (i64::from(diff) + 1), max_frame_num)
				{
					for (m, hit) in self.entries[i].marking.iter_mut().zip(fields) {
						if hit {
							*m = Marking::Unused;
						}
					}
				}
			}
			Mmco::ForgetLong { long_term_pic_num } => {
				let (idx, fields) = match parity {
					None => (long_term_pic_num, [true, true]),
					Some(p) => {
						let (idx, par) = split_pic_num(i64::from(long_term_pic_num), p);
						(idx as u32, [par == 0, par == 1])
					}
				};
				for e in &mut self.entries {
					if !field && e.marking != [Marking::Long(idx); 2] {
						continue;
					}
					for (m, hit) in e.marking.iter_mut().zip(fields) {
						if hit && *m == Marking::Long(idx) {
							*m = Marking::Unused;
						}
					}
				}
			}
			Mmco::ShortToLong { diff, long_term_frame_idx } => {
				let target = self.short_target(h, curr - (i64::from(diff) + 1), max_frame_num);
				let owner = target.map_or(0, |(i, _)| self.entries[i].pic.id);
				self.free_long_idx(long_term_frame_idx, owner, field);
				if let Some((i, fields)) = target {
					for (m, hit) in self.entries[i].marking.iter_mut().zip(fields) {
						if hit {
							*m = Marking::Long(long_term_frame_idx);
						}
					}
				}
			}
			Mmco::MaxLongIdx { max_plus1 } => {
				self.max_long_term_idx = max_plus1.checked_sub(1);
				for e in &mut self.entries {
					for m in &mut e.marking {
						if let Marking::Long(i) = *m {
							if self.max_long_term_idx.is_none_or(|mx| i > mx) {
								*m = Marking::Unused;
							}
						}
					}
				}
			}
			Mmco::Reset => {
				for e in &mut self.entries {
					e.marking = [Marking::Unused; 2];
				}
				self.max_long_term_idx = None;
				return true;
			}
			Mmco::CurrentToLong { long_term_frame_idx } => {
				self.free_long_idx(long_term_frame_idx, cur_id, field);
				*cur = Marking::Long(long_term_frame_idx);
			}
		}
		false
	}

	/// Marks references after decoding a frame or field (8.2.5) and stores
	/// it; the second field of a pair joins its first field's entry.
	/// `poc` is the (top, bottom) field order count pair (for a field, its
	/// own count twice). Returns `true` if the picture carried
	/// memory_management_control_operation 5.
	pub fn store(
		&mut self,
		pic: Arc<Picture<T>>,
		h: &SliceHeader,
		sps: &Sps,
		poc: (i32, i32),
		reorder_depth: usize,
	) -> bool {
		let parity = h.parity();
		let id = pic.id;
		let second = parity.is_some() && self.entries.iter().any(|e| e.pic.id == id);
		let mut mmco5 = false;
		let mut marking = if h.nal_ref_idc != 0 { Marking::Short } else { Marking::Unused };
		if h.idr {
			for e in &mut self.entries {
				e.marking = [Marking::Unused; 2];
			}
			self.flush_output();
			if h.long_term_reference {
				marking = Marking::Long(0);
				self.max_long_term_idx = Some(0);
			} else {
				self.max_long_term_idx = None;
			}
		} else if h.nal_ref_idc != 0 {
			if let Some(ops) = &h.mmco {
				for &op in ops {
					mmco5 |= self.apply_mmco(op, h, sps, id, &mut marking);
				}
			} else {
				// The second field of a reference pair joins its first field
				// without a sliding window (8.2.5.3).
				let first_is_ref = second && self.entries.iter().any(|e| e.pic.id == id && e.is_ref());
				if !first_is_ref {
					self.sliding_window(sps, h.frame_num, Some(id));
				}
			}
			// Never exceed the reference limit, even with a broken stream.
			let max_refs = sps.max_num_ref_frames.max(1) as usize;
			if marking == Marking::Short
				&& !second
				&& self.refs().filter(|e| e.pic.id != id).count() >= max_refs
			{
				self.sliding_window(sps, h.frame_num, Some(id));
			}
		}
		if mmco5 {
			self.flush_output();
		}
		// After mmco 5 the picture's counts restart from 0 (8.2.1).
		let (mut top, mut bottom) = poc;
		if mmco5 {
			let temp = match parity {
				None => top.min(bottom),
				Some(0) => top,
				Some(_) => bottom,
			};
			top -= temp;
			bottom -= temp;
		}
		let frame_num = if mmco5 { 0 } else { h.frame_num };
		if second {
			let p = parity.unwrap_or(0);
			if let Some(e) = self.entries.iter_mut().find(|e| e.pic.id == id) {
				e.marking[p] = marking;
				e.field_poc[p] = if p == 0 { top } else { bottom };
				e.complete = true;
			}
		} else {
			self.entries.retain(|e| e.is_ref() || e.needed_for_output);
			let (marks, field_poc) = match parity {
				None => ([marking; 2], [top, bottom]),
				Some(0) => ([marking, Marking::Unused], [top, i32::MAX]),
				Some(_) => ([Marking::Unused, marking], [i32::MAX, bottom]),
			};
			let needed_for_output = !pic.non_existing;
			self.entries.push(Entry {
				pic,
				marking: marks,
				frame_num,
				field_poc,
				needed_for_output,
				complete: parity.is_none(),
			});
		}
		if parity.is_none() || second {
			self.bump(reorder_depth, sps.dpb_frames());
		}
		mmco5
	}

	/// Lets a first field whose pair never came be output on its own.
	pub fn complete_unpaired(&mut self, id: u32, reorder_depth: usize, capacity: usize) {
		if let Some(e) = self.entries.iter_mut().find(|e| e.pic.id == id && !e.complete) {
			e.complete = true;
			self.bump(reorder_depth, capacity);
		}
	}

	/// The field order counts of picture `id` (the current pair's first
	/// field, say).
	pub fn field_poc(&self, id: u32) -> Option<[i32; 2]> {
		self.entries.iter().find(|e| e.pic.id == id).map(|e| e.field_poc)
	}

	/// Moves pictures to `output` in POC order while more than
	/// `reorder_depth` are waiting or the buffer is over capacity.
	fn bump(&mut self, reorder_depth: usize, capacity: usize) {
		loop {
			let waiting = self.entries.iter().filter(|e| e.needed_for_output && e.complete).count();
			if waiting == 0 || (waiting <= reorder_depth && self.entries.len() <= capacity) {
				break;
			}
			self.output_one(false);
			self.entries.retain(|e| e.is_ref() || e.needed_for_output);
		}
	}

	fn output_one(&mut self, any: bool) {
		if let Some(e) = self
			.entries
			.iter_mut()
			.filter(|e| e.needed_for_output && (any || e.complete))
			.min_by_key(|e| e.poc())
		{
			e.needed_for_output = false;
			self.output.push(e.pic.clone());
		}
	}

	/// Outputs every waiting picture (end of stream, IDR, MMCO 5).
	pub fn flush_output(&mut self) {
		while self.entries.iter().any(|e| e.needed_for_output) {
			self.output_one(true);
		}
		self.entries.retain(|e| e.is_ref());
	}
}

/// The decode counter after `id`: non-zero and below 2^30, so that
/// [`RefPic::key`] fits.
pub fn next_picture_id(id: u32) -> u32 {
	let n = id.wrapping_add(1) & ((1 << 30) - 1);
	n.max(1)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::h264::slice::SliceType;

	fn sps() -> Sps {
		Sps {
			profile_idc: 100,
			level_idc: 40,
			id: 0,
			chroma_format_idc: 1,
			bit_depth: 8,
			scaling: None,
			log2_max_frame_num: 4,
			poc_type: 0,
			log2_max_poc_lsb: 6,
			delta_pic_order_always_zero: false,
			offset_for_non_ref_pic: 0,
			offset_for_top_to_bottom_field: 0,
			offset_for_ref_frame: Vec::new(),
			max_num_ref_frames: 4,
			gaps_in_frame_num_allowed: false,
			mb_width: 2,
			mb_height: 2,
			frame_mbs_only: false,
			mbaff: false,
			direct_8x8_inference: true,
			crop: [0; 4],
			max_num_reorder_frames: Some(0),
			max_dec_frame_buffering: Some(4),
		}
	}

	fn header(
		slice_type: SliceType,
		frame_num: u32,
		parity: Option<usize>,
		poc_lsb: u32,
	) -> SliceHeader {
		SliceHeader {
			first_mb: 0,
			field_pic: parity.is_some(),
			bottom_field: parity == Some(1),
			mbaff: false,
			slice_type,
			pps_id: 0,
			frame_num,
			idr: frame_num == 0 && parity != Some(1),
			nal_ref_idc: 1,
			idr_pic_id: 0,
			poc_lsb,
			delta_poc_bottom: 0,
			delta_poc: [0; 2],
			redundant_pic_cnt: 0,
			direct_spatial: false,
			num_ref_idx: [8, 8],
			ref_mods: [Vec::new(), Vec::new()],
			weights: None,
			no_output_of_prior_pics: false,
			long_term_reference: false,
			mmco: None,
			cabac_init_idc: 0,
			qp: 26,
			disable_deblocking: 0,
			alpha_offset: 0,
			beta_offset: 0,
			data_bit_offset: 0,
		}
	}

	fn picture(id: u32) -> Arc<Picture> {
		Arc::new(Picture {
			id,
			data: Arc::new(PicData::pending(2, 8, true, false, PicPlanes::default())),
			pts: None,
			keyframe: false,
			non_existing: false,
			field_coded: true,
		})
	}

	/// (picture id, parity) of every list entry.
	fn ids(list: &[Option<RefPic>]) -> Vec<(u32, Option<usize>)> {
		list.iter().flatten().map(|r| (r.pic.id, r.parity)).collect()
	}

	/// Two field pairs (ids 1, 2), then the first (top) field of a third.
	fn dpb_with_pairs() -> Dpb {
		let sps = sps();
		let mut dpb = Dpb::default();
		for (id, frame_num) in [(1u32, 0u32), (2, 1)] {
			let pic = picture(id);
			for parity in 0..2 {
				let h = header(SliceType::P, frame_num, Some(parity), 4 * frame_num + parity as u32);
				let poc = (4 * frame_num + parity as u32) as i32;
				dpb.store(pic.clone(), &h, &sps, (poc, poc), 0);
			}
		}
		let h = header(SliceType::P, 2, Some(0), 8);
		dpb.store(picture(3), &h, &sps, (8, 8), 0);
		dpb
	}

	#[test]
	fn field_lists_alternate_parities_starting_with_the_current_one() {
		let dpb = dpb_with_pairs();
		// The bottom field of frame 3 (8.2.4.2.5): bottom fields and top
		// fields alternate, frames by descending FrameNumWrap; frame 3 has
		// only its top field yet.
		let h = header(SliceType::P, 2, Some(1), 9);
		let lists = dpb.ref_lists(&h, &sps(), 9).unwrap();
		assert_eq!(
			ids(&lists[0]),
			[(2, Some(1)), (3, Some(0)), (1, Some(1)), (2, Some(0)), (1, Some(0))]
		);
	}

	#[test]
	fn frame_lists_use_complete_reference_frames_only() {
		let mut dpb = dpb_with_pairs();
		// Pair 3 completes; a frame then sees the three frames, newest first.
		let h = header(SliceType::P, 2, Some(1), 9);
		dpb.store(picture(3), &h, &sps(), (9, 9), 0);
		let h = header(SliceType::P, 3, None, 12);
		let lists = dpb.ref_lists(&h, &sps(), 12).unwrap();
		assert_eq!(ids(&lists[0]), [(3, None), (2, None), (1, None)]);
		// The pair's counts were joined: top 8, bottom 9.
		assert_eq!(dpb.field_poc(3), Some([8, 9]));
	}

	#[test]
	fn field_pic_nums_select_parity_in_list_modification() {
		let dpb = dpb_with_pairs();
		// Bottom field of frame_num 2: CurrPicNum = 5. abs_diff 3 + 1 = 4
		// back is PicNum 1 = 2 * FrameNumWrap(0) + 1: the same-parity
		// (bottom) field of frame_num 0.
		let mut h = header(SliceType::P, 2, Some(1), 9);
		h.ref_mods[0] = vec![RefModification::ShortTerm { add: false, abs_diff: 3 }];
		let lists = dpb.ref_lists(&h, &sps(), 9).unwrap();
		assert_eq!(ids(&lists[0])[0], (1, Some(1)));
		// Then PicNum 0: the top field of frame_num 0 (opposite parity).
		h.ref_mods[0] = vec![RefModification::ShortTerm { add: false, abs_diff: 4 }];
		let lists = dpb.ref_lists(&h, &sps(), 9).unwrap();
		assert_eq!(ids(&lists[0])[0], (1, Some(0)));
	}

	#[test]
	fn field_poc_type0_gives_each_field_its_own_count() {
		let sps = sps();
		let mut poc = PocState::default();
		let top = header(SliceType::I, 0, Some(0), 0);
		assert_eq!(poc.compute(&sps, &top, false), (0, 0));
		let bottom = header(SliceType::P, 0, Some(1), 1);
		assert_eq!(poc.compute(&sps, &bottom, false), (1, 1));
		// A frame: bottom = top + delta_pic_order_cnt_bottom.
		let mut frame = header(SliceType::P, 1, None, 4);
		frame.delta_poc_bottom = 1;
		assert_eq!(poc.compute(&sps, &frame, false), (4, 5));
	}
}
