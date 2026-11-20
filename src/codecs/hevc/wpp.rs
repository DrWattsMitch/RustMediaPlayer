//! Wavefront parallel processing: the CTB rows of a WPP slice segment are
//! decoded from their own substreams (the slice header's entry points), on
//! several threads at once, each row two CTBs behind the row above
//! (ITU-T H.265 §9.3.1, §9.3.2.4; FFmpeg `hls_decode_entry_wpp`).
//!
//! Everything here is safe Rust. A row thread decodes into its own
//! full-size [`PicState`] workspace, so the CTU code keeps its
//! whole-picture indexing unchanged. A row only ever reads its own CTBs and
//! the last line of the row above (samples, motion, CBFs, split depths,
//! skip flags, SAO parameters, slice addresses), so after every CTB the
//! row thread copies that CTB's bottom line into a small shared [`Edge`]
//! and raises the row's progress. The thread below waits for CTB `x + 1`
//! of the row above before decoding CTB `x` and copies the edge into its
//! own workspace. Finished rows are merged into the picture, which runs
//! the in-loop filters as rows complete.
//!
//! The result is the same for every thread count, damaged streams
//! included: with one thread the same row decoder runs directly on the
//! picture, rows stop at the same CTBs (a row decodes CTB `x` only if the
//! row above decoded CTB `x + 1`), and workspaces reset each CTB's area
//! before decoding it, so a partly decoded CTB merges exactly as it would
//! have been written in place.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use super::ctu::{
	Contexts, DecodeError, PicState, RowHooks, Sao, SliceDecoder, SliceInfo, init_type_of,
};
use super::decoder::SliceJob;
use super::intra;
use super::picture::MvField;
use crate::codecs::threading::Progress;

/// Reusable row-thread workspaces (allocating and clearing a full-size
/// picture per segment would cost more than the copies it saves).
#[derive(Default)]
pub struct Workspaces(Mutex<Vec<PicState>>);

impl Workspaces {
	fn take(&self, pic: &PicState) -> PicState {
		let found = {
			let mut v = self.0.lock().unwrap_or_else(|e| e.into_inner());
			let k = v.iter().position(|w| {
				let (a, b) = (&*w.sps, &*pic.sps);
				a.width == b.width
					&& a.height == b.height
					&& a.log2_ctb_size == b.log2_ctb_size
					&& a.log2_min_cb_size == b.log2_min_cb_size
			});
			k.map(|k| v.swap_remove(k))
		};
		let mut w = found.unwrap_or_else(|| {
			PicState::new(pic.sps.clone(), pic.pps.clone(), pic.layout.clone(), pic.poc)
		});
		w.sps = pic.sps.clone();
		w.pps = pic.pps.clone();
		w.layout = pic.layout.clone();
		w.poc = pic.poc;
		w.slices = pic.slices.clone();
		w.workspace = true;
		w
	}

	fn put(&self, w: PicState) {
		let mut v = self.0.lock().unwrap_or_else(|e| e.into_inner());
		// One per row thread is enough; keep a few for alternating sizes.
		if v.len() < crate::codecs::threading::MAX_THREADS {
			v.push(w);
		}
	}
}

/// Whether slice segment `s` is decoded row by row from its entry points:
/// WPP without tiles, starting at the start of a CTB row, with one entry
/// point per further row. Returns the number of rows. The choice depends
/// only on the stream, never on the thread count.
pub(super) fn rows_of(pic: &PicState, s: &SliceJob) -> Option<usize> {
	let (sps, pps) = (&*pic.sps, &*pic.pps);
	let cw = sps.ctb_width as usize;
	let addr = s.header.segment_address as usize;
	let rows = s.header.entry_points.len() + 1;
	(std::env::var_os("RMP_NO_ROWS").is_none()
		&& pps.entropy_coding_sync
		&& !pps.tiles_enabled
		&& rows > 1
		&& addr.is_multiple_of(cw)
		&& addr / cw + rows <= sps.ctb_height as usize)
		.then_some(rows)
}

/// Start of each row's substream in the RBSP. Entry point offsets count
/// bytes of the NAL unit as sent, emulation prevention bytes included
/// (7.4.7.1), so they are mapped through the removed positions.
fn substream_starts(s: &SliceJob) -> Vec<usize> {
	let esc = &s.escapes;
	// Escaped position of RBSP byte `i`: every removed byte before it
	// shifts it by one.
	let mut j = s.header.data_offset;
	for &e in esc {
		if (e as usize) <= j {
			j += 1;
		} else {
			break;
		}
	}
	let mut starts = vec![s.header.data_offset];
	let mut at = j;
	for &off in &s.header.entry_points {
		at = at.saturating_add(off as usize);
		let removed = esc.partition_point(|&e| (e as usize) < at);
		starts.push(at - removed);
	}
	starts
}

/// The bottom line of a CTB row, as the row below reads it.
struct Edge {
	planes: [Vec<u16>; 3],
	mvf: Vec<MvField>,
	cbf: Vec<u8>,
	skip: Vec<u8>,
	ct_depth: Vec<u8>,
	ctb_addr: Vec<u32>,
	ctb_slice: Vec<u16>,
	sao: Vec<Sao>,
	/// Contexts after the row's second CTB (its first, while only one is
	/// done).
	ctx: Option<Contexts>,
}

/// Geometry of the bottom line of CTB row `cy` and of CTB column `cx`.
struct Line {
	/// Luma row of the line and the 4x4 / min-CB grid rows.
	y: usize,
	y4: usize,
	ycb: usize,
}

fn line_of(pic: &PicState, cy: usize) -> Line {
	let sps = &*pic.sps;
	let y = (((cy + 1) << sps.log2_ctb_size).min(sps.height as usize)) - 1;
	Line { y, y4: y >> 2, ycb: y >> sps.log2_min_cb_size }
}

/// Luma x range of CTB columns `c0..c1`.
fn cols(pic: &PicState, c0: usize, c1: usize) -> (usize, usize) {
	let sps = &*pic.sps;
	let w = sps.width as usize;
	((c0 << sps.log2_ctb_size).min(w), (c1 << sps.log2_ctb_size).min(w))
}

impl Edge {
	fn new(pic: &PicState) -> Self {
		let sps = &*pic.sps;
		let (w, cw) = (sps.width as usize, sps.ctb_width as usize);
		Self {
			planes: [vec![0; w], vec![0; w / 2], vec![0; w / 2]],
			mvf: vec![MvField::default(); pic.w4],
			cbf: vec![0; pic.w4],
			skip: vec![0; sps.min_cb_width as usize],
			ct_depth: vec![0; sps.min_cb_width as usize],
			ctb_addr: vec![u32::MAX; cw],
			ctb_slice: vec![0; cw],
			sao: vec![Sao::default(); cw],
			ctx: None,
		}
	}

	/// Copies CTB columns `c0..c1` of row `cy`'s bottom line between the
	/// edge and `pic` (`export`: picture to edge).
	fn transfer(&mut self, pic: &mut PicState, cy: usize, c0: usize, c1: usize, export: bool) {
		let l = line_of(pic, cy);
		let (x0, x1) = cols(pic, c0, c1);
		if x0 >= x1 {
			return;
		}
		let cw = pic.sps.ctb_width as usize;
		let lcb = pic.sps.log2_min_cb_size;
		let mcw = pic.sps.min_cb_width as usize;
		fn mv<T: Copy>(edge: &mut [T], grid: &mut [T], at: usize, a: usize, b: usize, export: bool) {
			if export {
				edge[a..b].copy_from_slice(&grid[at + a..at + b]);
			} else {
				grid[at + a..at + b].copy_from_slice(&edge[a..b]);
			}
		}
		for c in 0..3 {
			let sh = usize::from(c > 0);
			let p = &mut pic.planes[c];
			let y = if c == 0 { l.y } else { ((l.y + 1) >> 1) - 1 };
			let at = p.idx(0, y as isize);
			mv(&mut self.planes[c], &mut p.data, at, x0 >> sh, x1 >> sh, export);
		}
		let (a4, b4) = (x0 >> 2, x1.div_ceil(4).min(pic.w4));
		let at4 = l.y4 * pic.w4;
		mv(&mut self.mvf, &mut pic.mvf, at4, a4, b4, export);
		mv(&mut self.cbf, &mut pic.cbf_luma, at4, a4, b4, export);
		let (acb, bcb) = (x0 >> lcb, (x1.div_ceil(1 << lcb)).min(mcw));
		let atcb = l.ycb * mcw;
		mv(&mut self.skip, &mut pic.skip, atcb, acb, bcb, export);
		mv(&mut self.ct_depth, &mut pic.ct_depth, atcb, acb, bcb, export);
		let atc = cy * cw;
		mv(&mut self.ctb_addr, &mut pic.ctb_addr, atc, c0, c1, export);
		mv(&mut self.ctb_slice, &mut pic.ctb_slice, atc, c0, c1, export);
		mv(&mut self.sao, &mut pic.sao, atc, c0, c1, export);
	}
}

/// What a row shares with the row below.
struct RowShared {
	/// CTBs decoded; `Progress` holds the same count, or `u32::MAX` once the
	/// row has ended (decoded to its end, failed, or reached the end of
	/// the segment).
	count: AtomicU32,
	progress: Progress,
	edge: Mutex<Edge>,
}

struct Hooks<'s> {
	k: usize,
	cy: usize,
	cw: usize,
	rows: &'s [RowShared],
	/// Running in a workspace: the edge of the row above must be copied in.
	workspace: bool,
	/// Columns of the row above already copied in.
	imported: usize,
	fresh: &'s Contexts,
}

impl RowHooks for Hooks<'_> {
	fn before(&mut self, pic: &mut PicState, cx: usize) -> bool {
		if self.workspace {
			reset_ctb(pic, cx, self.cy);
		}
		if self.k == 0 {
			return true;
		}
		let above = &self.rows[self.k - 1];
		let need = (cx + 2).min(self.cw);
		above.progress.wait_for(need as u32);
		if (above.count.load(Ordering::Acquire) as usize) < need {
			return false;
		}
		if self.workspace && self.imported < need {
			let mut e = above.edge.lock().unwrap_or_else(|e| e.into_inner());
			e.transfer(pic, self.cy - 1, self.imported, need, false);
			self.imported = need;
		}
		true
	}

	fn start_contexts(&mut self) -> Option<Contexts> {
		if self.k == 0 {
			return None;
		}
		// The row above has decoded its second CTB (`before` waited for it),
		// which is in this slice: synchronise from it (9.3.1). A one-CTB
		// wide picture has no CTB above-right: fresh contexts.
		if self.cw == 1 {
			return Some(self.fresh.clone());
		}
		let e = self.rows[self.k - 1].edge.lock().unwrap_or_else(|e| e.into_inner());
		Some(e.ctx.clone().unwrap_or_else(|| self.fresh.clone()))
	}

	fn after(&mut self, pic: &PicState, cx: usize, ctx: Option<&Contexts>) {
		let me = &self.rows[self.k];
		if self.workspace || ctx.is_some() {
			let mut e = me.edge.lock().unwrap_or_else(|e| e.into_inner());
			if self.workspace {
				// `transfer` needs `&mut` for import; exporting only reads.
				export_column(&mut e, pic, self.cy, cx);
			}
			if let Some(c) = ctx {
				e.ctx = Some(c.clone());
			}
		}
		me.count.store(cx as u32 + 1, Ordering::Release);
		if self.workspace {
			// Rows on one thread run in order: nobody waits.
			me.progress.report(cx as u32 + 1);
		}
	}
}

/// [`Edge::transfer`] in the export direction, from a shared picture.
fn export_column(e: &mut Edge, pic: &PicState, cy: usize, cx: usize) {
	let l = line_of(pic, cy);
	let (x0, x1) = cols(pic, cx, cx + 1);
	let lcb = pic.sps.log2_min_cb_size;
	let mcw = pic.sps.min_cb_width as usize;
	let cw = pic.sps.ctb_width as usize;
	for c in 0..3 {
		let sh = usize::from(c > 0);
		let p = &pic.planes[c];
		let y = if c == 0 { l.y } else { ((l.y + 1) >> 1) - 1 };
		let at = p.idx(0, y as isize);
		let (a, b) = (x0 >> sh, x1 >> sh);
		e.planes[c][a..b].copy_from_slice(&p.data[at + a..at + b]);
	}
	let (a4, b4) = (x0 >> 2, x1.div_ceil(4).min(pic.w4));
	let at4 = l.y4 * pic.w4;
	e.mvf[a4..b4].copy_from_slice(&pic.mvf[at4 + a4..at4 + b4]);
	e.cbf[a4..b4].copy_from_slice(&pic.cbf_luma[at4 + a4..at4 + b4]);
	let (acb, bcb) = (x0 >> lcb, (x1.div_ceil(1 << lcb)).min(mcw));
	let atcb = l.ycb * mcw;
	e.skip[acb..bcb].copy_from_slice(&pic.skip[atcb + acb..atcb + bcb]);
	e.ct_depth[acb..bcb].copy_from_slice(&pic.ct_depth[atcb + acb..atcb + bcb]);
	let at = cy * cw + cx;
	e.ctb_addr[cx] = pic.ctb_addr[at];
	e.ctb_slice[cx] = pic.ctb_slice[at];
	e.sao[cx] = pic.sao[at];
}

/// Rectangle of CTB `(cx, cy)` in each grid, as (first row, rows end,
/// first column, columns end).
struct CtbRect {
	luma: [usize; 4],
	g4: [usize; 4],
	gcb: [usize; 4],
}

/// The area of CTBs `c0..c1` of row `cy`, clipped to the picture.
fn rect(pic: &PicState, cy: usize, c0: usize, c1: usize) -> CtbRect {
	let sps = &*pic.sps;
	let l = sps.log2_ctb_size;
	let h = sps.height as usize;
	let (x0, x1) = cols(pic, c0, c1);
	let (y0, y1) = ((cy << l).min(h), ((cy + 1) << l).min(h));
	let lcb = sps.log2_min_cb_size;
	let (mcw, mch) = (sps.min_cb_width as usize, sps.min_cb_height as usize);
	CtbRect {
		luma: [y0, y1, x0, x1],
		g4: [y0 >> 2, y1.div_ceil(4).min(pic.h4), x0 >> 2, x1.div_ceil(4).min(pic.w4)],
		gcb: [y0 >> lcb, (y1.div_ceil(1 << lcb)).min(mch), x0 >> lcb, (x1.div_ceil(1 << lcb)).min(mcw)],
	}
}

fn fill_rect<T: Copy>(grid: &mut [T], stride: usize, r: [usize; 4], v: T) {
	for y in r[0]..r[1] {
		grid[y * stride + r[2]..y * stride + r[3]].fill(v);
	}
}

fn copy_rect<T: Copy>(dst: &mut [T], src: &[T], stride: usize, r: [usize; 4]) {
	for y in r[0]..r[1] {
		let (a, b) = (y * stride + r[2], y * stride + r[3]);
		dst[a..b].copy_from_slice(&src[a..b]);
	}
}

/// Returns CTB `(cx, cy)` of a workspace to the state of a new picture, so
/// that whatever the decoder leaves there (even a partly decoded CTB)
/// merges exactly as if it had been decoded in place.
fn reset_ctb(pic: &mut PicState, cx: usize, cy: usize) {
	let r = rect(pic, cy, cx, cx + 1);
	let mid = 1u16 << (pic.sps.bit_depth - 1);
	for c in 0..3 {
		let sh = usize::from(c > 0);
		let p = &mut pic.planes[c];
		for y in r.luma[0] >> sh..r.luma[1] >> sh {
			let at = p.idx(0, y as isize);
			p.data[at + (r.luma[2] >> sh)..at + (r.luma[3] >> sh)].fill(mid);
		}
	}
	let w4 = pic.w4;
	fill_rect(&mut pic.mvf, w4, r.g4, MvField::default());
	fill_rect(&mut pic.ipm, w4, r.g4, intra::MODE_DC as u8);
	fill_rect(&mut pic.cbf_luma, w4, r.g4, 0);
	fill_rect(&mut pic.no_filter, w4, r.g4, 0);
	fill_rect(&mut pic.bs_v, w4, r.g4, 0);
	fill_rect(&mut pic.bs_h, w4, r.g4, 0);
	let mcw = pic.sps.min_cb_width as usize;
	fill_rect(&mut pic.skip, mcw, r.gcb, 0);
	fill_rect(&mut pic.ct_depth, mcw, r.gcb, 0);
	fill_rect(&mut pic.qp_y, mcw, r.gcb, 0);
	let at = cy * pic.sps.ctb_width as usize + cx;
	pic.ctb_addr[at] = u32::MAX;
	pic.ctb_slice[at] = 0;
	pic.sao[at] = Sao::default();
}

/// Copies CTBs `0..n` of row `cy` from a workspace into the picture.
/// `decoded` of them completed (the next one may be partial).
fn merge_row(pic: &mut PicState, ws: &PicState, cy: usize, n: usize, decoded: usize) {
	let r = rect(pic, cy, 0, n);
	for c in 0..3 {
		let sh = usize::from(c > 0);
		let (d, s) = (&mut pic.planes[c], &ws.planes[c]);
		for y in r.luma[0] >> sh..r.luma[1] >> sh {
			let at = d.idx(0, y as isize);
			let (a, b) = (at + (r.luma[2] >> sh), at + (r.luma[3] >> sh));
			d.data[a..b].copy_from_slice(&s.data[a..b]);
		}
	}
	let w4 = pic.w4;
	copy_rect(&mut pic.mvf, &ws.mvf, w4, r.g4);
	copy_rect(&mut pic.ipm, &ws.ipm, w4, r.g4);
	copy_rect(&mut pic.cbf_luma, &ws.cbf_luma, w4, r.g4);
	copy_rect(&mut pic.no_filter, &ws.no_filter, w4, r.g4);
	copy_rect(&mut pic.bs_v, &ws.bs_v, w4, r.g4);
	copy_rect(&mut pic.bs_h, &ws.bs_h, w4, r.g4);
	let mcw = pic.sps.min_cb_width as usize;
	copy_rect(&mut pic.skip, &ws.skip, mcw, r.gcb);
	copy_rect(&mut pic.ct_depth, &ws.ct_depth, mcw, r.gcb);
	copy_rect(&mut pic.qp_y, &ws.qp_y, mcw, r.gcb);
	let cw = pic.sps.ctb_width as usize;
	let a = cy * cw;
	pic.ctb_addr[a..a + n].copy_from_slice(&ws.ctb_addr[a..a + n]);
	pic.ctb_slice[a..a + n].copy_from_slice(&ws.ctb_slice[a..a + n]);
	pic.sao[a..a + n].copy_from_slice(&ws.sao[a..a + n]);
	pic.row_ctbs[cy] += decoded as u32;
	pic.ctbs_decoded += decoded;
	pic.advance(false);
}

/// How a row ended: CTBs decoded, and the decoder's result.
type RowResult = (usize, usize, Result<bool, DecodeError>);

/// Decodes a WPP slice segment of `rows` rows (see [`rows_of`]) with up
/// to `threads` threads.
pub(super) fn decode_rows(
	pic: &mut PicState,
	s: &SliceJob,
	rows: usize,
	threads: usize,
	workspaces: &Workspaces,
) -> Result<(), DecodeError> {
	let sh = &s.header;
	let cw = pic.sps.ctb_width as usize;
	let r0 = sh.segment_address as usize / cw;
	// Rows already (partly) decoded mean a duplicate or overlapping
	// segment: drop it rather than rewrite filtered rows.
	if pic.ctb_addr[r0 * cw..(r0 + rows) * cw].iter().any(|&a| a != u32::MAX) {
		return Err(DecodeError("hevc: slice overlaps decoded CTBs"));
	}
	let starts = substream_starts(s);
	pic.slices.push(SliceInfo::of(sh, &s.refs));
	let slice_idx = (pic.slices.len() - 1) as u16;
	let fresh = Contexts::new(init_type_of(sh), sh.slice_qp);
	// Contexts of the first row (the `ts == first_ts` rules of the serial
	// decoder, for a segment starting a row).
	let mut ctx0 = if sh.dependent {
		pic.ds_contexts.as_deref().cloned().unwrap_or_else(|| fresh.clone())
	} else {
		fresh.clone()
	};
	if !sh.first_slice_in_pic {
		let tr_available = r0 > 0 && cw > 1 && pic.ctb_addr[r0 * cw + 1 - cw] == sh.slice_addr_rs;
		if !tr_available {
			ctx0 = fresh.clone();
		} else if let Some(c) = &pic.wpp_contexts {
			ctx0 = (**c).clone();
		}
	}
	let shared: Vec<RowShared> = (0..rows)
		.map(|_| RowShared {
			count: AtomicU32::new(0),
			progress: Progress::new(0),
			edge: Mutex::new(if threads > 1 { Edge::new(pic) } else { Edge::empty() }),
		})
		.collect();
	let ctx_for = |k: usize| if k == 0 { ctx0.clone() } else { fresh.clone() };
	let mut results: Vec<Option<RowResult>> = (0..rows).map(|_| None).collect();
	let mut last_state = None;

	if threads <= 1 {
		let mut sd =
			SliceDecoder::with_slice(pic, sh, &s.refs, &s.rbsp, starts[0], slice_idx, ctx0.clone());
		for k in 0..rows {
			let mut hooks =
				Hooks { k, cy: r0 + k, cw, rows: &shared, workspace: false, imported: 0, fresh: &fresh };
			if k > 0 {
				sd.restart(starts[k], fresh.clone());
			}
			let res = sd.decode_row(r0 + k, &mut hooks);
			if k + 1 == rows && matches!(res.2, Ok(true)) {
				last_state = Some(sd.final_state());
			}
			end_row(&shared[k], res.0);
			results[k] = Some(res);
		}
	} else {
		let next = AtomicUsize::new(0);
		let results_m = Mutex::new(&mut results);
		let last_m = Mutex::new(&mut last_state);
		let pic_m = Mutex::new(&mut *pic);
		let work = || {
			let mut ws = {
				let p = pic_m.lock().unwrap_or_else(|e| e.into_inner());
				workspaces.take(&p)
			};
			let mut sd = SliceDecoder::with_slice(
				&mut ws,
				sh,
				&s.refs,
				&s.rbsp,
				starts[0],
				slice_idx,
				fresh.clone(),
			);
			loop {
				let k = next.fetch_add(1, Ordering::AcqRel);
				if k >= rows {
					break;
				}
				let cy = r0 + k;
				if k == 0 && cy > 0 {
					// The row above the segment is complete in the picture (and
					// not yet deblocked: that waits for this row).
					let mut e = Edge::new(sd.pic);
					let mut p = pic_m.lock().unwrap_or_else(|e| e.into_inner());
					e.transfer(&mut p, cy - 1, 0, cw, true);
					drop(p);
					e.transfer(sd.pic, cy - 1, 0, cw, false);
				}
				let mut hooks =
					Hooks { k, cy, cw, rows: &shared, workspace: true, imported: 0, fresh: &fresh };
				sd.restart(starts[k], ctx_for(k));
				let res = sd.decode_row(cy, &mut hooks);
				if k + 1 == rows && matches!(res.2, Ok(true)) {
					**last_m.lock().unwrap_or_else(|e| e.into_inner()) = Some(sd.final_state());
				}
				end_row(&shared[k], res.0);
				// A CTB that failed half way is merged too: in place it would
				// have been partly written.
				merge_row(&mut pic_m.lock().unwrap_or_else(|e| e.into_inner()), sd.pic, cy, res.1, res.0);
				results_m.lock().unwrap_or_else(|e| e.into_inner())[k] = Some(res);
			}
			drop(sd);
			workspaces.put(ws);
		};
		std::thread::scope(|scope| {
			for _ in 1..threads.min(rows) {
				scope.spawn(work);
			}
			work();
		});
	}

	if let Some((ctx, qp)) = last_state {
		pic.ds_contexts = Some(Box::new(ctx));
		pic.ds_qp_y = qp;
	}
	if let Some(c) = shared[rows - 1].edge.lock().unwrap_or_else(|e| e.into_inner()).ctx.take() {
		pic.wpp_contexts = Some(Box::new(c));
	}
	if std::env::var_os("RMP_HEVC_DEBUG").is_some() {
		for (n, _, res) in results.iter().flatten() {
			eprintln!("hevc: wpp row: {n} ctbs, {res:?}");
		}
	}
	for (k, r) in results.into_iter().enumerate() {
		match r {
			Some((.., Err(e))) => return Err(e),
			Some((.., Ok(true))) if k + 1 < rows => {
				return Err(DecodeError("hevc: slice ends before its last entry point"));
			}
			Some((.., Ok(false))) if k + 1 == rows => {
				return Err(DecodeError("hevc: missing end of slice"));
			}
			None => return Err(DecodeError("hevc: row not decoded")),
			_ => {}
		}
	}
	Ok(())
}

/// Lets the row below see that this row has ended after `n` CTBs.
fn end_row(row: &RowShared, n: usize) {
	row.count.store(n as u32, Ordering::Release);
	row.progress.report(u32::MAX);
}

impl Edge {
	/// An edge that only carries contexts (one thread: rows decode in place).
	fn empty() -> Self {
		Self {
			planes: [Vec::new(), Vec::new(), Vec::new()],
			mvf: Vec::new(),
			cbf: Vec::new(),
			skip: Vec::new(),
			ct_depth: Vec::new(),
			ctb_addr: Vec::new(),
			ctb_slice: Vec::new(),
			sao: Vec::new(),
			ctx: None,
		}
	}
}
