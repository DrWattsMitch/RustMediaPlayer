//! HEVC decoder front end: NAL dispatch, POC, reference picture sets,
//! reference lists, the DPB and output order (ITU-T H.265 §8.1-8.3,
//! Annex C; FFmpeg `hevcdec.c`, `hevc/refs.c`).

use std::sync::Arc;

use super::ctu::{PicState, SliceDecoder, SliceRefs};
use super::picture::{Frame, FrameRef, MvField, PAD_CHROMA, PAD_LUMA, Plane, RefInfo};
use super::ps::{Layout, MAX_PPS, MAX_SPS, Pps, Sps};
use super::slice::{self, SliceHeader, SliceType, is_idr, is_irap, nal};
use crate::codecs::h264::nal::{split_annex_b, split_length_prefixed, unescape};
use crate::core::Time;

/// A picture in display order, cropped.
#[derive(Debug, Clone)]
pub struct OutputPicture {
	pub width: u32,
	pub height: u32,
	pub bit_depth: u32,
	/// Packed planar 4:2:0: one byte per sample at 8-bit, little-endian
	/// u16 above.
	pub data: Vec<u8>,
	pub pts: Option<Time>,
	pub keyframe: bool,
}

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

struct Current {
	pic: PicState,
	header: SliceHeader,
	dpb_index: usize,
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
	out: Vec<OutputPicture>,
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
		Self {
			sps: self.sps.clone(),
			pps: self.pps.clone(),
			layouts: self.layouts.clone(),
			nal_length_size: self.nal_length_size,
			..Self::default()
		}
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
			out: Vec::new(),
			damaged_slices: 0,
			unsupported: None,
		}
	}
}

#[derive(Debug)]
pub struct HevcError(pub &'static str);

impl HevcDecoderState {
	/// Creates a decoder; `extradata` may be an `hvcC` record or Annex B
	/// parameter sets.
	pub fn new(extradata: &[u8]) -> Self {
		let mut s = Self::default();
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
		std::mem::take(&mut self.out)
	}

	/// Outputs every picture still waiting in the DPB.
	pub fn drain(&mut self) -> Vec<OutputPicture> {
		self.finish_picture();
		while self.bump() {}
		self.dpb.clear();
		std::mem::take(&mut self.out)
	}

	pub fn reset(&mut self) {
		self.cur = None;
		self.dpb.clear();
		self.out.clear();
		self.first_picture = true;
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
				let rbsp = unescape(data);
				let sps = Sps::parse(&rbsp).map_err(|e| HevcError(e.0))?;
				let id = sps.id;
				self.sps[id] = Some(Arc::new(sps));
			}
			nal::PPS => {
				let rbsp = unescape(data);
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
				let rbsp = unescape(data);
				self.slice(&rbsp, t, tid, pts)?;
			}
			_ => {}
		}
		Ok(())
	}

	fn slice(&mut self, rbsp: &[u8], t: u8, tid: u8, pts: Option<Time>) -> Result<(), HevcError> {
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
		let sh = slice::parse(rbsp, t, &self.sps, &self.pps, prev).map_err(|e| HevcError(e.0))?;
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
		// A segment starting on an already decoded CTB (a duplicate or
		// overlapping slice) is dropped, as FFmpeg does, rather than
		// overwriting good samples.
		if cur.pic.ctb_addr.get(sh.segment_address as usize) != Some(&u32::MAX) {
			return Err(HevcError("hevc: overlapping slice segment"));
		}
		// Reference lists.
		let refs = if sh.dependent {
			cur.refs_of_slice.clone().ok_or(HevcError("hevc: dependent slice without lists"))?
		} else {
			let r = Arc::new(build_lists(cur, &sh)?);
			cur.refs_of_slice = Some(r.clone());
			r
		};
		let mut sd = SliceDecoder::new(&mut cur.pic, &sh, &refs, rbsp);
		sd.decode().map_err(|e| HevcError(e.0))?;
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
			if a.width != sps.width || a.height != sps.height || a.bit_depth != sps.bit_depth {
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
		let pic = PicState::new(sps, pps, layout, poc);
		self.cur = Some(Box::new(Current {
			pic,
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

	/// Runs the in-loop filters on the current picture and stores it.
	fn finish_picture(&mut self) {
		let Some(cur) = self.cur.take() else { return };
		let mut pic = cur.pic;
		pic.deblock();
		pic.sao();
		for p in &mut pic.planes {
			p.extend_edges();
		}
		let sps = pic.sps.clone();
		let frame = Frame {
			planes: pic.planes,
			poc: pic.poc,
			mvf: pic.mvf,
			w4: pic.w4,
			ctb_slice: pic.ctb_slice,
			slice_refs: pic.slices.into_iter().map(|s| s.refs).collect(),
			log2_ctb_size: sps.log2_ctb_size,
			ctb_width: sps.ctb_width as usize,
			bit_depth: sps.bit_depth,
		};
		if let Some(e) = self.dpb.get_mut(cur.dpb_index) {
			e.frame = Some(Arc::new(frame));
		}
		let _ = cur.header;
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
		self.out.push(crop_output(&f, e.crop, e.pts, e.keyframe));
		if !e.short_ref && !e.long_ref {
			self.dpb.remove(k);
		}
		true
	}
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

fn missing_frame(sps: &Sps, poc: i32) -> FrameRef {
	let mid = 1u16 << (sps.bit_depth - 1);
	let (w, h) = (sps.width as usize, sps.height as usize);
	let w4 = w.div_ceil(4);
	Arc::new(Frame {
		planes: [
			Plane::new(w, h, PAD_LUMA, mid),
			Plane::new(w / 2, h / 2, PAD_CHROMA, mid),
			Plane::new(w / 2, h / 2, PAD_CHROMA, mid),
		],
		poc,
		mvf: vec![MvField::default(); w4 * h.div_ceil(4)],
		w4,
		ctb_slice: vec![0; (sps.ctb_width * sps.ctb_height) as usize],
		slice_refs: vec![RefInfo::default()],
		log2_ctb_size: sps.log2_ctb_size,
		ctb_width: sps.ctb_width as usize,
		bit_depth: sps.bit_depth,
	})
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
				if fr.poc > cur.pic.poc {
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
	// crop: left, right, top, bottom in luma samples.
	let [l, r, t, b] = crop.map(|v| v as usize);
	let w = f.planes[0].width - l - r;
	let h = f.planes[0].height - t - b;
	let wide = f.bit_depth > 8;
	let mut data = Vec::with_capacity(w * h * 3 / 2 * if wide { 2 } else { 1 });
	for (c, p) in f.planes.iter().enumerate() {
		let (x0, y0, pw, ph) = if c == 0 { (l, t, w, h) } else { (l / 2, t / 2, w / 2, h / 2) };
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
	OutputPicture { width: w as u32, height: h as u32, bit_depth: f.bit_depth, data, pts, keyframe }
}
