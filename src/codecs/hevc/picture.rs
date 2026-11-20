//! Decoded pictures and their per-block side information, and
//! [`RefPic`], the shared handle through which pictures that are still
//! being decoded on another thread are read (frame threading).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::codecs::threading::Progress;

/// Padding around every plane so motion compensation can read outside the
/// picture without bounds checks (edges are replicated into it).
pub const PAD_LUMA: usize = 80;
pub const PAD_CHROMA: usize = 40;

/// One plane of samples (8-bit streams use the low 8 bits).
#[derive(Clone)]
pub struct Plane {
	pub data: Vec<u16>,
	pub stride: usize,
	pub width: usize,
	pub height: usize,
	pub pad: usize,
}

impl Plane {
	pub fn new(width: usize, height: usize, pad: usize, fill: u16) -> Self {
		let stride = width + 2 * pad;
		Self { data: vec![fill; stride * (height + 2 * pad)], stride, width, height, pad }
	}

	/// Index of sample (x, y) (may be negative within the padding).
	#[inline]
	pub fn idx(&self, x: isize, y: isize) -> usize {
		((y + self.pad as isize) as usize) * self.stride + (x + self.pad as isize) as usize
	}

	#[cfg(test)]
	pub fn get(&self, x: usize, y: usize) -> u16 {
		self.data[(y + self.pad) * self.stride + x + self.pad]
	}

	/// Replicates the outermost samples of rows `y0..y1` into the left and
	/// right padding.
	pub fn extend_rows(&mut self, y0: usize, y1: usize) {
		let (w, p, s) = (self.width, self.pad, self.stride);
		for y in y0..y1.min(self.height) {
			let row = (y + p) * s;
			let left = self.data[row + p];
			let right = self.data[row + p + w - 1];
			self.data[row..row + p].fill(left);
			self.data[row + p + w..row + s].fill(right);
		}
	}

	/// Replicates the (already extended) first row into the top padding.
	pub fn extend_top(&mut self) {
		let (p, s) = (self.pad, self.stride);
		let first = p * s;
		for y in 0..p {
			self.data.copy_within(first..first + s, y * s);
		}
	}

	/// Replicates the (already extended) last row into the bottom padding.
	pub fn extend_bottom(&mut self) {
		let (h, p, s) = (self.height, self.pad, self.stride);
		let last = (p + h - 1) * s;
		for y in 0..p {
			self.data.copy_within(last..last + s, (p + h + y) * s);
		}
	}

	/// Index range of rows `y0..y1` (whole stride rows; `y0` may be
	/// negative to include the top padding, `y1` up to `height + pad`).
	pub fn rows_range(&self, y0: isize, y1: isize) -> std::ops::Range<usize> {
		let p = self.pad as isize;
		((y0 + p) as usize) * self.stride..((y1 + p) as usize) * self.stride
	}

	/// Replicates the outermost samples into the padding.
	#[allow(dead_code)]
	pub fn extend_edges(&mut self) {
		let (w, h, p, s) = (self.width, self.height, self.pad, self.stride);
		for y in 0..h {
			let row = (y + p) * s;
			let left = self.data[row + p];
			let right = self.data[row + p + w - 1];
			self.data[row..row + p].fill(left);
			self.data[row + p + w..row + s].fill(right);
		}
		let first = p * s;
		let last = (p + h - 1) * s;
		for y in 0..p {
			self.data.copy_within(first..first + s, y * s);
			self.data.copy_within(last..last + s, (p + h + y) * s);
		}
	}
}

/// Prediction flags of a motion field entry.
pub const PF_INTRA: u8 = 0;
pub const PF_L0: u8 = 1;
pub const PF_L1: u8 = 2;
pub const PF_BI: u8 = 3;

/// Motion of one 4x4 luma block.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MvField {
	pub mv: [[i16; 2]; 2],
	pub ref_idx: [i8; 2],
	pub pred: u8,
}

/// The reference lists a slice used, by picture identity (POC) and
/// long-term marking, kept with the picture for temporal MV prediction and
/// deblocking.
#[derive(Clone, Debug, Default)]
pub struct RefInfo {
	pub poc: [[i32; 16]; 2],
	pub long_term: [[bool; 16]; 2],
}

/// A fully decoded picture as kept in the DPB.
pub struct Frame {
	pub planes: [Plane; 3],
	pub poc: i32,
	/// Motion field on the 4x4 grid (`w4` columns).
	pub mvf: Vec<MvField>,
	pub w4: usize,
	/// Slice index of each CTB, into `slice_refs`.
	pub ctb_slice: Vec<u16>,
	pub slice_refs: Vec<RefInfo>,
	pub log2_ctb_size: u32,
	pub ctb_width: usize,
	/// Bit depth of the samples.
	pub bit_depth: u32,
}

/// A picture in the DPB, possibly still being decoded by a picture job on
/// another thread (FFmpeg's `ThreadFrame`).
///
/// The job decodes into its own buffers and, when other pictures may read
/// this one while it is in flight, copies every finished band of rows into
/// `live` before raising `rows` (final luma sample rows) and `motion`
/// (CTB rows whose motion field is final). Readers wait for what they
/// need, then read `live` under a shared lock. When the picture is
/// complete the job's own buffers move into `done`, which is read without
/// locking.
pub struct RefPic {
	pub poc: i32,
	height: u32,
	ctb_rows: u32,
	log2_ctb: u32,
	rows: Progress,
	motion: Progress,
	live: RwLock<Option<Box<Frame>>>,
	/// Whether `live` is published as rows finish (otherwise readers wait
	/// for the whole picture).
	publishes: bool,
	done: OnceLock<Frame>,
	poisoned: AtomicBool,
	/// Slice segments of this picture that failed to decode.
	pub damaged: AtomicU64,
	/// The cropped output samples, prepared by the picture job when the
	/// picture is to be output (so the copy runs on the frame thread).
	pub output: std::sync::Mutex<Option<Vec<u8>>>,
}

pub type FrameRef = Arc<RefPic>;

/// `rows` value of a complete picture (padding included).
const COMPLETE: u32 = u32::MAX;

impl RefPic {
	/// A picture that is already complete.
	pub fn complete(frame: Frame) -> FrameRef {
		let pic = Self::pending(frame.poc, frame.planes[0].height, 1, frame.log2_ctb_size, None);
		pic.finish(frame);
		Arc::new(pic)
	}

	/// A picture about to be decoded. `live` receives the finished rows
	/// (`None` when nobody reads the picture before it is complete).
	pub fn pending(
		poc: i32,
		height: usize,
		ctb_rows: usize,
		log2_ctb: u32,
		live: Option<Box<Frame>>,
	) -> Self {
		Self {
			poc,
			height: height as u32,
			ctb_rows: ctb_rows as u32,
			log2_ctb,
			rows: Progress::new(0),
			motion: Progress::new(0),
			publishes: live.is_some(),
			live: RwLock::new(live),
			done: OnceLock::new(),
			poisoned: AtomicBool::new(false),
			damaged: AtomicU64::new(0),
			output: std::sync::Mutex::new(None),
		}
	}

	pub fn publishes(&self) -> bool {
		self.publishes
	}

	/// The finished picture, if it is complete.
	pub fn done(&self) -> Option<&Frame> {
		self.done.get()
	}

	/// Blocks until the picture is complete and returns it.
	pub fn wait_done(&self) -> &Frame {
		self.rows.wait_for(COMPLETE);
		self.check_poison();
		self.done.get().expect("complete picture")
	}

	fn check_poison(&self) {
		if self.poisoned.load(Ordering::Acquire) {
			panic!("hevc: reference picture decoding failed");
		}
	}

	/// Copies finished rows into `live` and raises the progress counters:
	/// luma rows `< rows` (and the matching chroma rows) are final, and so
	/// is the motion of CTB rows `< motion`. `copy` gets the live picture.
	pub fn publish(&self, rows: u32, motion: u32, copy: impl FnOnce(&mut Frame)) {
		if self.publishes {
			let mut live = self.live.write().unwrap_or_else(|e| e.into_inner());
			if let Some(f) = live.as_mut() {
				copy(f);
			}
		}
		self.motion.report(motion);
		self.rows.report(rows);
	}

	/// Stores the complete picture and wakes every reader. Returns the
	/// `live` copy (for reuse).
	pub fn finish(&self, frame: Frame) -> Option<Box<Frame>> {
		let _ = self.done.set(frame);
		self.motion.report(COMPLETE);
		self.rows.report(COMPLETE);
		self.live.write().unwrap_or_else(|e| e.into_inner()).take()
	}

	/// Marks the picture as failed: readers panic instead of waiting.
	pub fn poison(&self) {
		self.poisoned.store(true, Ordering::Release);
		self.motion.report(COMPLETE);
		self.rows.report(COMPLETE);
	}

	/// Runs `f` on the picture once luma rows `lo..=hi` can be read (`lo`
	/// below 0 or `hi` past the bottom reach into the replicated padding).
	#[inline]
	pub fn with_rows<R>(&self, lo: isize, hi: isize, f: impl FnOnce(&Frame) -> R) -> R {
		if let Some(d) = self.done.get() {
			return f(d);
		}
		let need = if !self.publishes || hi >= self.height as isize - 1 {
			COMPLETE
		} else {
			// The top padding is written with the first band.
			(hi + 1).max(if lo < 0 { 1 } else { 0 }) as u32
		};
		self.read(&self.rows, need, f)
	}

	/// Runs `f` on the picture once the motion at luma row `y` is final.
	#[inline]
	pub fn with_motion<R>(&self, y: usize, f: impl FnOnce(&Frame) -> R) -> R {
		if let Some(d) = self.done.get() {
			return f(d);
		}
		let row = (y >> self.log2_ctb) as u32 + 1;
		let need = if !self.publishes || row >= self.ctb_rows { COMPLETE } else { row };
		self.read(&self.motion, need, f)
	}

	fn read<R>(&self, p: &Progress, need: u32, f: impl FnOnce(&Frame) -> R) -> R {
		p.wait_for(need);
		self.check_poison();
		if let Some(d) = self.done.get() {
			return f(d);
		}
		let live = self.live.read().unwrap_or_else(|e| e.into_inner());
		match live.as_deref() {
			Some(fr) => f(fr),
			None => {
				drop(live);
				f(self.wait_done())
			}
		}
	}
}

impl std::fmt::Debug for RefPic {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("RefPic").field("poc", &self.poc).finish()
	}
}

impl Frame {
	/// Motion at luma sample (x, y).
	#[inline]
	pub fn mv_at(&self, x: usize, y: usize) -> &MvField {
		&self.mvf[(y >> 2) * self.w4 + (x >> 2)]
	}

	/// Reference lists of the slice covering luma sample (x, y).
	pub fn refs_at(&self, x: usize, y: usize) -> &RefInfo {
		let ctb = (y >> self.log2_ctb_size) * self.ctb_width + (x >> self.log2_ctb_size);
		let s = self.ctb_slice.get(ctb).copied().unwrap_or(0) as usize;
		&self.slice_refs[s.min(self.slice_refs.len().saturating_sub(1))]
	}
}
