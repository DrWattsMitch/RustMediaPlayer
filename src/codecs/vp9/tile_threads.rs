//! Tile and row threading (FFmpeg `vp9.c`: `decode_tiles_mt` with
//! `loopfilter_proc`; libvpx `vp9_dthread.c`).
//!
//! VP9 tile *columns* share no entropy or prediction state: blocks only
//! look left inside their tile, above-right never leaves the block, and
//! the above context arrays are indexed by column. Each column is decoded
//! by a worker into its own [`FrameDecoder`] (a full-size workspace of
//! which it only touches its own columns). After every superblock row the
//! worker sends that row's rectangle to the calling thread, which copies it
//! into the frame and loop-filters each row as soon as every column has
//! delivered it, in the same order as the single-threaded filter.
//!
//! Decoding never reads filtered pixels (the serial decoder filters only
//! after all tiles), and the workspaces keep their own unfiltered copy, so
//! the filter can run concurrently with decoding without changing a bit.
//! With one tile column this is still a two-stage pipeline (decode | loop
//! filter), FFmpeg's and libvpx's row-based loop filter threading.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Condvar, Mutex};

use super::loopfilter::{self, LfCell};
use super::{
	BoolDecoder, FrameDecoder, ModeInfo, MvRef, Picture, Plane, Px, Result, corrupt, inter,
};
use crate::core::{Pooled, VecPool};

/// Reusable buffers of one tile column (the big arrays of a
/// [`FrameDecoder`]).
pub(super) struct TileArrays<P> {
	mi: Vec<ModeInfo>,
	lf_cells: Vec<LfCell>,
	cur_mvs: Vec<MvRef>,
	seg_map: Vec<u8>,
	planes: [Vec<P>; 3],
	above_nz: [Vec<u8>; 3],
	above_partition: Vec<u8>,
	coefs: Vec<i32>,
	scratch: Option<inter::Scratch<P>>,
}

impl<P> Default for TileArrays<P> {
	fn default() -> Self {
		Self {
			mi: Vec::new(),
			lf_cells: Vec::new(),
			cur_mvs: Vec::new(),
			seg_map: Vec::new(),
			planes: Default::default(),
			above_nz: Default::default(),
			above_partition: Vec::new(),
			coefs: Vec::new(),
			scratch: None,
		}
	}
}

/// Tile workspaces kept between frames. Cloning a decoder state gives the
/// clone an empty pool (the buffers are only scratch space).
pub struct TilePool<P>(Vec<TileArrays<P>>, ChunkPools<P>);

impl<P> Default for TilePool<P> {
	fn default() -> Self {
		Self(Vec::new(), ChunkPools::default())
	}
}

/// Buffers of the row hand-over ([`Chunk`]s), recycled once merged.
struct ChunkPools<P> {
	samples: VecPool<P>,
	lf_cells: VecPool<LfCell>,
}

impl<P> Default for ChunkPools<P> {
	fn default() -> Self {
		Self { samples: VecPool::default(), lf_cells: VecPool::default() }
	}
}

impl<P> Clone for ChunkPools<P> {
	fn clone(&self) -> Self {
		Self { samples: self.samples.clone(), lf_cells: self.lf_cells.clone() }
	}
}

impl<P> Clone for TilePool<P> {
	fn clone(&self) -> Self {
		Self::default()
	}
}

impl<P> std::fmt::Debug for TilePool<P> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "TilePool({} workspaces)", self.0.len())
	}
}

/// One finished superblock row of one tile column.
struct Chunk<P> {
	tile_col: usize,
	mi_row: usize,
	planes: [Pooled<P>; 3],
	lf_cells: Pooled<LfCell>,
}

/// Resizes a reused buffer to `len` (new elements are `fill`).
fn sized<T: Clone>(mut v: Vec<T>, len: usize, fill: T) -> Vec<T> {
	v.resize(len, fill);
	v
}

impl<P: Px> FrameDecoder<P> {
	/// Sample columns of tile column `tile_col` in `plane`. The last column
	/// also owns the padding up to the stride.
	fn tile_x_range(&self, plane: usize, tile_col: usize) -> (usize, usize) {
		let (sx, _) = self.plane_ss(plane);
		let (c0, c1) = self.tile_mi_cols(tile_col);
		let stride = self.pic.planes[plane].stride;
		let last = tile_col + 1 == 1usize << self.h.log2_tile_cols;
		((c0 * 8) >> sx, if last { stride } else { ((c1 * 8) >> sx).min(stride) })
	}

	/// Sample rows of the superblock row starting at `mi_row`. The last one
	/// also owns the padding at the bottom of the buffer.
	fn sb_y_range(&self, plane: usize, mi_row: usize) -> (usize, usize) {
		let (_, sy) = self.plane_ss(plane);
		let p = &self.pic.planes[plane];
		let rows = p.data.len() / p.stride.max(1);
		let last = mi_row + 8 >= self.mi_rows;
		((mi_row * 8) >> sy, if last { rows } else { (((mi_row + 8) * 8) >> sy).min(rows) })
	}

	/// A workspace for tile column `tile_col`: shares the frame's headers,
	/// contexts and references, and starts with that column of every array
	/// exactly as this (not yet decoded) frame has it.
	fn fork(&self, ws: TileArrays<P>, tile_col: usize) -> FrameDecoder<P> {
		let cells = self.mi_rows * self.mi_cols;
		let (c0, c1) = self.tile_mi_cols(tile_col);
		// Every block writes its mode info and filter inputs, and blocks tile
		// the whole grid, so those cells are never read before this frame
		// writes them. Motion is not written on intra-only frames, and the
		// segment map only when segmentation is on: start those from the
		// frame's values.
		let mi = sized(ws.mi, cells, ModeInfo::default());
		let lf_cells = sized(ws.lf_cells, cells, LfCell::default());
		let mut cur_mvs = sized(ws.cur_mvs, cells, MvRef::default());
		let mut seg_map = sized(ws.seg_map, cells, 0);
		let intra_only = self.h.intra_only_frame();
		for row in 0..self.mi_rows {
			let (a, b) = (row * self.mi_cols + c0, row * self.mi_cols + c1);
			if intra_only {
				cur_mvs[a..b].copy_from_slice(&self.cur_mvs[a..b]);
			}
			if self.seg.enabled {
				seg_map[a..b].copy_from_slice(&self.seg_map[a..b]);
			}
		}
		let [d0, d1, d2] = ws.planes;
		let planes = [d0, d1, d2].into_iter().enumerate().map(|(i, data)| {
			let src = &self.pic.planes[i];
			let mut data = sized(data, src.data.len(), P::default());
			let (x0, x1) = self.tile_x_range(i, tile_col);
			// A new picture starts zeroed (`Picture::new`).
			for row in data.chunks_exact_mut(src.stride.max(1)) {
				row[x0..x1].fill(P::default());
			}
			Plane {
				data: data.into(),
				stride: src.stride,
				width: src.width,
				height: src.height,
				aligned_w: src.aligned_w,
				aligned_h: src.aligned_h,
			}
		});
		let planes: Vec<Plane<P>> = planes.collect();
		let Ok(planes) = <[Plane<P>; 3]>::try_from(planes) else {
			unreachable!("three planes");
		};
		let [a0, a1, a2] = ws.above_nz;
		let above_nz = [a0, a1, a2]
			.into_iter()
			.enumerate()
			.map(|(i, v)| {
				let mut v = sized(v, self.above_nz[i].len(), 0);
				v.fill(0);
				v
			})
			.collect::<Vec<_>>();
		let Ok(above_nz) = <[Vec<u8>; 3]>::try_from(above_nz) else {
			unreachable!("three planes");
		};
		let mut above_partition = sized(ws.above_partition, self.above_partition.len(), 0);
		above_partition.fill(0);
		let mut coefs = sized(ws.coefs, 32 * 32, 0);
		coefs.fill(0);
		FrameDecoder {
			h: self.h.clone(),
			ch: self.ch.clone(),
			fc: self.fc.clone(),
			seg: self.seg.clone(),
			mi_rows: self.mi_rows,
			mi_cols: self.mi_cols,
			mi,
			lf_cells,
			cur_mvs,
			prev_mvs: self.prev_mvs.clone(),
			seg_map,
			last_seg_map: self.last_seg_map.clone(),
			counts: Box::default(),
			counting: self.counting,
			tile: Default::default(),
			above_nz,
			left_nz: [[0; 16]; 3],
			above_partition,
			left_partition: [0; 8],
			pic: Picture {
				planes,
				width: self.pic.width,
				height: self.pic.height,
				ss_x: self.pic.ss_x,
				ss_y: self.pic.ss_y,
				bit_depth: self.pic.bit_depth,
				key_frame: self.pic.key_frame,
				rgb: self.pic.rgb,
			},
			refs: self.refs.clone(),
			ref_scale: self.ref_scale,
			bit_depth: self.bit_depth,
			max: self.max,
			dequant: self.dequant,
			lf_table: self.lf_table,
			coefs,
			scratch: ws.scratch.unwrap_or_default(),
		}
	}

	/// Gives the workspace's buffers back for the next frame.
	fn into_arrays(self) -> TileArrays<P> {
		let [p0, p1, p2] = self.pic.planes;
		TileArrays {
			mi: self.mi,
			lf_cells: self.lf_cells,
			cur_mvs: self.cur_mvs,
			seg_map: self.seg_map,
			planes: [p0.data.into_vec(), p1.data.into_vec(), p2.data.into_vec()],
			above_nz: self.above_nz,
			above_partition: self.above_partition,
			coefs: self.coefs,
			scratch: Some(self.scratch),
		}
	}

	/// Copies out the superblock row at `mi_row` of tile column `tile_col`.
	fn take_chunk(&self, tile_col: usize, mi_row: usize, pools: &ChunkPools<P>) -> Chunk<P> {
		let (c0, c1) = self.tile_mi_cols(tile_col);
		let r1 = (mi_row + 8).min(self.mi_rows);
		let mut lf_cells = pools.lf_cells.take_empty((r1 - mi_row) * (c1 - c0));
		for row in mi_row..r1 {
			let (a, b) = (row * self.mi_cols + c0, row * self.mi_cols + c1);
			lf_cells.vec_mut().extend_from_slice(&self.lf_cells[a..b]);
		}
		let planes = std::array::from_fn(|i| {
			let p = &self.pic.planes[i];
			let (x0, x1) = self.tile_x_range(i, tile_col);
			let (y0, y1) = self.sb_y_range(i, mi_row);
			let mut out = pools.samples.take_empty((x1 - x0) * (y1 - y0));
			for y in y0..y1 {
				out.vec_mut().extend_from_slice(&p.data[y * p.stride + x0..y * p.stride + x1]);
			}
			out
		});
		Chunk { tile_col, mi_row, planes, lf_cells }
	}

	/// Copies tile column `tile_col`'s motion and segment map (not needed
	/// until the frame is done) from its workspace.
	fn gather_column(&mut self, ws: &FrameDecoder<P>, tile_col: usize) {
		let (c0, c1) = self.tile_mi_cols(tile_col);
		for row in 0..self.mi_rows {
			let (a, b) = (row * self.mi_cols + c0, row * self.mi_cols + c1);
			self.cur_mvs[a..b].copy_from_slice(&ws.cur_mvs[a..b]);
			if self.seg.enabled {
				self.seg_map[a..b].copy_from_slice(&ws.seg_map[a..b]);
			}
		}
	}

	/// Copies a chunk into the frame.
	fn put_chunk(&mut self, c: &Chunk<P>) {
		let (c0, c1) = self.tile_mi_cols(c.tile_col);
		let w = c1 - c0;
		let r1 = (c.mi_row + 8).min(self.mi_rows);
		for (k, row) in (c.mi_row..r1).enumerate() {
			let (a, b) = (row * self.mi_cols + c0, row * self.mi_cols + c1);
			let s = k * w..(k + 1) * w;
			self.lf_cells[a..b].copy_from_slice(&c.lf_cells[s]);
		}
		for i in 0..3 {
			let (x0, x1) = self.tile_x_range(i, c.tile_col);
			let (y0, y1) = self.sb_y_range(i, c.mi_row);
			let p = &mut self.pic.planes[i];
			let w = x1 - x0;
			for (k, y) in (y0..y1).enumerate() {
				p.data[y * p.stride + x0..y * p.stride + x1]
					.copy_from_slice(&c.planes[i][k * w..(k + 1) * w]);
			}
		}
	}

	/// [`FrameDecoder::decode_tiles`] followed by the loop filter, with the
	/// tile columns on up to `threads` workers and the filter on this
	/// thread, one superblock row behind.
	pub(super) fn decode_tiles_threaded(
		&mut self,
		data: &[u8],
		threads: usize,
		pool: &mut TilePool<P>,
	) -> Result<()> {
		let tile_cols = 1usize << self.h.log2_tile_cols;
		let tile_rows = 1usize << self.h.log2_tile_rows;
		let buffers = self.split_tiles(data)?;
		// The calling thread is one of the `threads`: it decodes too, and
		// merges and filters whenever a row is ready.
		let helpers = threads.min(tile_cols + 1).max(2) - 1;
		// Every column starts as a task at its first superblock row.
		let mut tasks = VecDeque::with_capacity(tile_cols);
		for tile_col in 0..tile_cols {
			let ws = pool.0.pop().unwrap_or_default();
			tasks.push_back(Column {
				tile_col,
				fd: self.fork(ws, tile_col),
				reader: None,
				tile_row: 0,
				mi_row: 0,
			});
		}
		let sched = Sched {
			state: Mutex::new(SchedState {
				tasks,
				remaining: tile_cols,
				failed: None,
				finished: Vec::new(),
			}),
			wake: Condvar::new(),
			buffers: &buffers,
			chunks: pool.1.clone(),
			tile_rows,
			tile_cols,
		};
		let mut merger = Merger {
			delivered: vec![0; self.mi_rows.div_ceil(8)],
			next: 0,
			tile_cols,
			filter: self.h.filter_level != 0,
			thr: loopfilter::thresholds(self.h.sharpness),
		};
		let main_decodes = tile_cols > helpers;
		let (tx, rx) = mpsc::channel::<Chunk<P>>();
		std::thread::scope(|scope| {
			for _ in 0..helpers {
				let tx = tx.clone();
				let sched = &sched;
				scope.spawn(move || {
					let mut last = usize::MAX;
					while let Some(mut col) = sched.take(true, last) {
						last = col.tile_col;
						let r = sched.step(&mut col, &mut |c| {
							// The receiver only goes away if the frame is abandoned.
							let _ = tx.send(c);
						});
						sched.put_back(col, r);
					}
				});
			}
			drop(tx);
			loop {
				// Rows first, so the filter stays right behind the decoders.
				while let Ok(chunk) = rx.try_recv() {
					merger.accept(self, &chunk);
				}
				// With a helper per column, this thread only merges and
				// filters (decode | filter pipeline, also for one column).
				let col = if main_decodes { sched.take(false, usize::MAX) } else { None };
				if let Some(mut col) = col {
					let r = sched.step(&mut col, &mut |c| merger.accept(self, &c));
					sched.put_back(col, r);
					continue;
				}
				// Nothing to decode here: wait for the helpers' rows. The
				// channel closes once every helper has stopped.
				match rx.recv() {
					Ok(chunk) => merger.accept(self, &chunk),
					Err(_) => break,
				}
			}
		});
		let state = sched.state.into_inner().unwrap_or_else(|e| e.into_inner());
		let mut finished = state.finished;
		// Columns left after an error still return their buffers.
		finished.extend(state.tasks.into_iter().map(|c| (c.tile_col, c.fd)));
		finished.sort_by_key(|(c, _)| *c);
		for (tile_col, fd) in finished {
			if state.failed.is_none() {
				self.gather_column(&fd, tile_col);
				self.counts.add(&fd.counts);
			}
			pool.0.push(fd.into_arrays());
		}
		if let Some(e) = state.failed {
			return Err(e);
		}
		if merger.next != merger.delivered.len() {
			return Err(corrupt("vp9: tile threads stopped early"));
		}
		Ok(())
	}
}

/// Merges rows into the frame and loop-filters each one once every tile
/// column has delivered it, top to bottom.
struct Merger {
	delivered: Vec<usize>,
	next: usize,
	tile_cols: usize,
	filter: bool,
	thr: [loopfilter::Thresholds; 64],
}

impl Merger {
	fn accept<P: Px>(&mut self, fd: &mut FrameDecoder<P>, chunk: &Chunk<P>) {
		fd.put_chunk(chunk);
		self.delivered[chunk.mi_row / 8] += 1;
		while self.next < self.delivered.len() && self.delivered[self.next] == self.tile_cols {
			if self.filter {
				fd.filter_sb_row(self.next * 8, &self.thr);
			}
			self.next += 1;
		}
	}
}

/// A tile column in progress: its workspace, its position and the bool
/// decoder of its current tile.
struct Column<'d, P> {
	tile_col: usize,
	fd: FrameDecoder<P>,
	reader: Option<BoolDecoder<'d>>,
	tile_row: usize,
	mi_row: usize,
}

struct SchedState<'d, P> {
	/// Columns waiting for a thread, longest-waiting first.
	tasks: VecDeque<Column<'d, P>>,
	/// Columns not yet complete (queued or being decoded).
	remaining: usize,
	failed: Option<crate::Message>,
	finished: Vec<(usize, FrameDecoder<P>)>,
}

/// Hands out tile columns one superblock row at a time. The least
/// advanced column goes first, so columns advance together (the filter can
/// follow) and any free thread picks up whichever column is behind.
struct Sched<'s, 'd, P> {
	state: Mutex<SchedState<'d, P>>,
	wake: Condvar,
	buffers: &'s [&'d [u8]],
	chunks: ChunkPools<P>,
	tile_rows: usize,
	tile_cols: usize,
}

impl<'d, P: Px> Sched<'_, 'd, P> {
	/// The next column to work on; `None` when all are done (or decoding
	/// failed). With `wait`, blocks while every column is busy elsewhere.
	/// Picks the least advanced waiting column, and among equals the one
	/// this thread had last (`last`), which is still in its cache.
	fn take(&self, wait: bool, last: usize) -> Option<Column<'d, P>> {
		let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
		loop {
			if s.failed.is_some() || s.remaining == 0 {
				return None;
			}
			let best = s
				.tasks
				.iter()
				.enumerate()
				.min_by_key(|(_, c)| (c.tile_row, c.mi_row, c.tile_col != last))
				.map(|(i, _)| i);
			if let Some(c) = best.and_then(|i| s.tasks.remove(i)) {
				return Some(c);
			}
			if !wait {
				return None;
			}
			s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
		}
	}

	/// Returns a column after [`Sched::step`].
	fn put_back(&self, col: Column<'d, P>, result: Result<bool>) {
		let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
		match result {
			Ok(true) => s.tasks.push_back(col),
			Ok(false) => {
				s.remaining -= 1;
				s.finished.push((col.tile_col, col.fd));
			}
			Err(e) => {
				s.failed.get_or_insert(e);
				s.tasks.push_back(col);
			}
		}
		drop(s);
		self.wake.notify_all();
	}

	/// Decodes one superblock row of `col` and emits it; `Ok(false)` once
	/// the column is complete.
	fn step(&self, col: &mut Column<'d, P>, emit: &mut dyn FnMut(Chunk<P>)) -> Result<bool> {
		loop {
			if col.tile_row >= self.tile_rows {
				return Ok(false);
			}
			let rows = col.fd.tile_mi_rows(col.tile_row);
			if col.reader.is_none() {
				let buf = self.buffers[col.tile_row * self.tile_cols + col.tile_col];
				col.reader = Some(BoolDecoder::new(buf).ok_or(corrupt("vp9: tile marker bit set"))?);
				col.mi_row = rows.0;
			}
			if col.mi_row >= rows.1 {
				col.tile_row += 1;
				col.reader = None;
				continue;
			}
			let Some(r) = col.reader.as_mut() else { continue };
			col.fd.decode_sb_row(r, rows, col.tile_col, col.mi_row)?;
			emit(col.fd.take_chunk(col.tile_col, col.mi_row, &self.chunks));
			col.mi_row += 8;
			return Ok(true);
		}
	}
}
