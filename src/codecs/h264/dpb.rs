//! Decoded picture buffer: picture order count (§8.2.1), reference list
//! construction and modification (§8.2.4), reference marking (§8.2.5) and
//! output ordering (C.4). FFmpeg: `h264_picture.c`, `h264_refs.c`,
//! `h2645_parse`/`h264_poc.c`.

use std::sync::Arc;

use super::params::Sps;
use super::recon::Plane;
use super::slice::{Mmco, RefModification, SliceHeader, SliceType};
use crate::core::Time;

/// Motion of one macroblock, kept with its picture for direct prediction
/// in later B pictures and for deblocking.
#[derive(Debug, Clone, Copy, Default)]
pub struct MbMotion {
	/// Motion vectors per list and raster 4x4 block, quarter-sample units.
	pub mv: [[[i16; 2]; 16]; 2],
	/// Reference index per list and 8x8 block (-1 = list unused).
	pub ref_idx: [[i8; 4]; 2],
	/// Identity of the referenced picture per list and 8x8 block (0 = none).
	pub ref_id: [[u32; 4]; 2],
	pub intra: bool,
}

/// A decoded frame.
#[derive(Debug, Clone)]
pub struct Picture {
	/// Unique, non-zero identity (decode counter).
	pub id: u32,
	pub y: Plane,
	pub cb: Plane,
	pub cr: Plane,
	pub poc: i32,
	pub motion: Vec<MbMotion>,
	pub pts: Option<Time>,
	pub keyframe: bool,
	/// Frames inserted for a frame_num gap; never output.
	pub non_existing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marking {
	Unused,
	Short,
	Long(u32),
}

#[derive(Debug, Clone)]
struct Entry {
	pic: Arc<Picture>,
	marking: Marking,
	frame_num: u32,
	needed_for_output: bool,
}

/// A reference picture as seen by one slice.
#[derive(Debug, Clone)]
pub struct RefPic {
	pub pic: Arc<Picture>,
	pub long: bool,
}

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
	/// Computes the POC of the current frame and returns `(top, bottom)`
	/// field order counts.
	pub fn compute(&mut self, sps: &Sps, h: &SliceHeader, prev_had_mmco5: bool) -> (i32, i32) {
		let max_frame_num = sps.max_frame_num() as i32;
		let frame_num = h.frame_num as i32;
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
				let top = msb.wrapping_add(lsb);
				let bottom = top.wrapping_add(h.delta_poc_bottom);
				if h.nal_ref_idc != 0 {
					self.prev_msb = msb;
					self.prev_lsb = lsb;
				}
				(top, bottom)
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
				let top = expected.wrapping_add(h.delta_poc[0]);
				let bottom =
					top.wrapping_add(sps.offset_for_top_to_bottom_field).wrapping_add(h.delta_poc[1]);
				(top, bottom)
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

#[derive(Debug, Clone, Default)]
pub struct Dpb {
	entries: Vec<Entry>,
	max_long_term_idx: Option<u32>,
	/// Pictures ready for output, in output order.
	pub output: Vec<Arc<Picture>>,
}

impl Dpb {
	pub fn clear(&mut self) {
		self.entries.clear();
		self.output.clear();
		self.max_long_term_idx = None;
	}

	fn pic_num(e: &Entry, cur_frame_num: u32, max_frame_num: u32) -> i32 {
		if e.frame_num > cur_frame_num {
			e.frame_num as i32 - max_frame_num as i32
		} else {
			e.frame_num as i32
		}
	}

	fn refs(&self) -> impl Iterator<Item = &Entry> {
		self.entries.iter().filter(|e| e.marking != Marking::Unused)
	}

	/// The most recent reference picture (used to conceal missing frames).
	pub fn latest_ref(&self) -> Option<Arc<Picture>> {
		self.entries.iter().rev().find(|e| e.marking != Marking::Unused).map(|e| e.pic.clone())
	}

	/// Builds RefPicList0/1 for a slice (8.2.4.2-3).
	pub fn ref_lists(
		&self,
		h: &SliceHeader,
		sps: &Sps,
		cur_poc: i32,
	) -> Result<[Vec<Option<RefPic>>; 2], BadRefs> {
		let max_frame_num = sps.max_frame_num();
		let mut short: Vec<&Entry> = self.refs().filter(|e| e.marking == Marking::Short).collect();
		let mut long: Vec<(&Entry, u32)> = self
			.refs()
			.filter_map(|e| if let Marking::Long(i) = e.marking { Some((e, i)) } else { None })
			.collect();
		long.sort_by_key(|&(_, i)| i);
		let as_ref = |e: &Entry, long: bool| RefPic { pic: e.pic.clone(), long };
		let mut init: [Vec<RefPic>; 2] = [Vec::new(), Vec::new()];
		match h.slice_type {
			SliceType::I => return Ok([Vec::new(), Vec::new()]),
			SliceType::P => {
				short.sort_by_key(|e| std::cmp::Reverse(Self::pic_num(e, h.frame_num, max_frame_num)));
				init[0] = short.iter().map(|e| as_ref(e, false)).collect();
				init[0].extend(long.iter().map(|(e, _)| as_ref(e, true)));
			}
			SliceType::B => {
				let mut before: Vec<&Entry> =
					short.iter().copied().filter(|e| e.pic.poc < cur_poc).collect();
				let mut after: Vec<&Entry> =
					short.iter().copied().filter(|e| e.pic.poc > cur_poc).collect();
				before.sort_by_key(|e| std::cmp::Reverse(e.pic.poc));
				after.sort_by_key(|e| e.pic.poc);
				for (list, (first, second)) in
					[(&before, &after), (&after, &before)].into_iter().enumerate()
				{
					init[list] = first.iter().chain(second.iter()).map(|e| as_ref(e, false)).collect();
					init[list].extend(long.iter().map(|(e, _)| as_ref(e, true)));
				}
				if init[1].len() > 1
					&& init[0].len() == init[1].len()
					&& init[0].iter().zip(&init[1]).all(|(a, b)| a.pic.id == b.pic.id)
				{
					init[1].swap(0, 1);
				}
			}
		}
		let mut lists: [Vec<Option<RefPic>>; 2] = [Vec::new(), Vec::new()];
		for l in 0..2 {
			let n = h.num_ref_idx[l];
			let mut list: Vec<Option<RefPic>> = init[l].iter().take(n).cloned().map(Some).collect();
			list.resize(n, None);
			self.modify(&mut list, &h.ref_mods[l], h.frame_num, max_frame_num)?;
			lists[l] = list;
		}
		Ok(lists)
	}

	/// Reference picture list modification (8.2.4.3).
	fn modify(
		&self,
		list: &mut Vec<Option<RefPic>>,
		mods: &[RefModification],
		frame_num: u32,
		max_frame_num: u32,
	) -> Result<(), BadRefs> {
		let n = list.len();
		let curr = frame_num as i64;
		let max = i64::from(max_frame_num);
		let mut pred = curr;
		for (ref_idx, m) in mods.iter().enumerate() {
			if ref_idx >= n {
				return Err(BadRefs("too many reference list modifications"));
			}
			let (pic, is_long) = match *m {
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
					let e = self
						.refs()
						.find(|e| {
							e.marking == Marking::Short
								&& i64::from(Self::pic_num(e, frame_num, max_frame_num)) == pic_num
						})
						.ok_or(BadRefs("reference list modification names a missing picture"))?;
					(e.pic.clone(), false)
				}
				RefModification::LongTerm(num) => {
					let e = self
						.refs()
						.find(|e| e.marking == Marking::Long(num))
						.ok_or(BadRefs("reference list modification names a missing long-term picture"))?;
					(e.pic.clone(), true)
				}
			};
			let id = pic.id;
			list.insert(ref_idx, Some(RefPic { pic, long: is_long }));
			// Remove the later duplicate of the same picture.
			let mut k = ref_idx + 1;
			while k < list.len() {
				if list[k].as_ref().is_some_and(|r| r.pic.id == id && r.long == is_long) {
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
		let Some(template) = self.latest_ref() else {
			return;
		};
		let mut unused = (prev_ref_frame_num + 1) % max;
		let mut guard = 0;
		while unused != frame_num && guard < 16 {
			self.sliding_window(sps, unused);
			let mut pic = (*template).clone();
			*next_id = next_id.wrapping_add(1).max(1);
			pic.id = *next_id;
			pic.non_existing = true;
			self.entries.push(Entry {
				pic: Arc::new(pic),
				marking: Marking::Short,
				frame_num: unused,
				needed_for_output: false,
			});
			unused = (unused + 1) % max;
			guard += 1;
		}
	}

	fn sliding_window(&mut self, sps: &Sps, cur_frame_num: u32) {
		let max_refs = sps.max_num_ref_frames.max(1) as usize;
		let max_frame_num = sps.max_frame_num();
		while self.refs().count() >= max_refs {
			let victim = self
				.entries
				.iter_mut()
				.filter(|e| e.marking == Marking::Short)
				.min_by_key(|e| Self::pic_num(e, cur_frame_num, max_frame_num));
			match victim {
				Some(e) => e.marking = Marking::Unused,
				None => break,
			}
		}
	}

	/// Marks references after decoding a picture (8.2.5) and stores it.
	/// Returns `true` if the picture carried memory_management_control_operation 5.
	pub fn store(
		&mut self,
		pic: Arc<Picture>,
		h: &SliceHeader,
		sps: &Sps,
		reorder_depth: usize,
	) -> bool {
		let mut mmco5 = false;
		let mut marking = if h.nal_ref_idc != 0 { Marking::Short } else { Marking::Unused };
		let max_frame_num = sps.max_frame_num();
		if h.idr {
			for e in &mut self.entries {
				e.marking = Marking::Unused;
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
				for op in ops {
					match *op {
						Mmco::ForgetShort { diff } => {
							let pn = h.frame_num as i64 - (i64::from(diff) + 1);
							for e in &mut self.entries {
								if e.marking == Marking::Short
									&& i64::from(Self::pic_num(e, h.frame_num, max_frame_num)) == pn
								{
									e.marking = Marking::Unused;
								}
							}
						}
						Mmco::ForgetLong { long_term_pic_num } => {
							for e in &mut self.entries {
								if e.marking == Marking::Long(long_term_pic_num) {
									e.marking = Marking::Unused;
								}
							}
						}
						Mmco::ShortToLong { diff, long_term_frame_idx } => {
							let pn = h.frame_num as i64 - (i64::from(diff) + 1);
							for e in &mut self.entries {
								if e.marking == Marking::Long(long_term_frame_idx) {
									e.marking = Marking::Unused;
								}
							}
							for e in &mut self.entries {
								if e.marking == Marking::Short
									&& i64::from(Self::pic_num(e, h.frame_num, max_frame_num)) == pn
								{
									e.marking = Marking::Long(long_term_frame_idx);
								}
							}
						}
						Mmco::MaxLongIdx { max_plus1 } => {
							self.max_long_term_idx = max_plus1.checked_sub(1);
							for e in &mut self.entries {
								if let Marking::Long(i) = e.marking {
									if self.max_long_term_idx.is_none_or(|m| i > m) {
										e.marking = Marking::Unused;
									}
								}
							}
						}
						Mmco::Reset => {
							for e in &mut self.entries {
								e.marking = Marking::Unused;
							}
							self.max_long_term_idx = None;
							mmco5 = true;
						}
						Mmco::CurrentToLong { long_term_frame_idx } => {
							for e in &mut self.entries {
								if e.marking == Marking::Long(long_term_frame_idx) {
									e.marking = Marking::Unused;
								}
							}
							marking = Marking::Long(long_term_frame_idx);
						}
					}
				}
			} else {
				self.sliding_window(sps, h.frame_num);
			}
			// Never exceed the reference limit, even with a broken stream.
			let max_refs = sps.max_num_ref_frames.max(1) as usize;
			if marking == Marking::Short && self.refs().count() >= max_refs {
				self.sliding_window(sps, h.frame_num);
			}
		}
		if mmco5 {
			self.flush_output();
		}
		let frame_num = if mmco5 { 0 } else { h.frame_num };
		self.entries.retain(|e| e.marking != Marking::Unused || e.needed_for_output);
		let needed_for_output = !pic.non_existing;
		self.entries.push(Entry { pic, marking, frame_num, needed_for_output });
		self.bump(reorder_depth, sps.dpb_frames());
		mmco5
	}

	/// Moves pictures to `output` in POC order while more than
	/// `reorder_depth` are waiting or the buffer is over capacity.
	fn bump(&mut self, reorder_depth: usize, capacity: usize) {
		loop {
			let waiting = self.entries.iter().filter(|e| e.needed_for_output).count();
			if waiting == 0 || (waiting <= reorder_depth && self.entries.len() <= capacity) {
				break;
			}
			self.output_one();
			self.entries.retain(|e| e.marking != Marking::Unused || e.needed_for_output);
		}
	}

	fn output_one(&mut self) {
		if let Some(e) =
			self.entries.iter_mut().filter(|e| e.needed_for_output).min_by_key(|e| e.pic.poc)
		{
			e.needed_for_output = false;
			self.output.push(e.pic.clone());
		}
	}

	/// Outputs every waiting picture (end of stream, IDR, MMCO 5).
	pub fn flush_output(&mut self) {
		while self.entries.iter().any(|e| e.needed_for_output) {
			self.output_one();
		}
		self.entries.retain(|e| e.marking != Marking::Unused);
	}

	/// `true` if `pic` is currently marked as a short-term reference.
	pub fn is_short_term(&self, id: u32) -> bool {
		self.entries.iter().any(|e| e.pic.id == id && e.marking == Marking::Short)
	}
}
