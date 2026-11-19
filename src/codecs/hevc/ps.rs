//! Video, sequence and picture parameter sets (ITU-T H.265 §7.3.2; FFmpeg
//! `hevc/ps.c`), plus the derived CTB address tables (§6.5).

use crate::codecs::bitstream::BitReader;

pub const MAX_SPS: usize = 16;
pub const MAX_PPS: usize = 64;
pub const MAX_SHORT_TERM_RPS: usize = 65;
pub const MAX_REFS: usize = 16;

#[derive(Debug)]
pub struct PsError(pub &'static str);
type PsResult<T> = Result<T, PsError>;

fn check(cond: bool, what: &'static str) -> PsResult<()> {
	if cond { Ok(()) } else { Err(PsError(what)) }
}

/// A short-term reference picture set (§7.4.8): deltas relative to the
/// current POC, negative ones first (closest first), then positive ones.
#[derive(Debug, Clone, Default)]
pub struct ShortTermRps {
	pub num_negative: usize,
	pub num_delta_pocs: usize,
	pub delta_poc: [i32; 32],
	pub used: [bool; 32],
}

/// Scaling factors per size (4x4..32x32) and matrixId (intra Y/Cb/Cr,
/// inter Y/Cb/Cr), each an 8x8 (or 4x4) coefficient grid.
#[derive(Debug, Clone)]
pub struct ScalingList {
	pub sl: [[[u8; 64]; 6]; 4],
	pub dc: [[u8; 6]; 2],
}

const DEFAULT_INTRA_8X8: [u8; 64] = [
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
	21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
	29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];
const DEFAULT_INTER_8X8: [u8; 64] = [
	16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
	20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
	28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];

/// Up-right diagonal scan of an `n`x`n` block: `(x, y)` per scan position.
pub fn diag_scan(n: usize) -> Vec<(u8, u8)> {
	let mut out = Vec::with_capacity(n * n);
	let (mut x, mut y) = (0i32, 0i32);
	while out.len() < n * n {
		while y >= 0 {
			if x < n as i32 && y < n as i32 {
				out.push((x as u8, y as u8));
			}
			y -= 1;
			x += 1;
		}
		y = x;
		x = 0;
	}
	out
}

impl ScalingList {
	/// The default lists (Tables 7-5 and 7-6), stored in raster order.
	pub fn default_lists() -> Self {
		let mut s = Self { sl: [[[16; 64]; 6]; 4], dc: [[16; 6]; 2] };
		let scan8 = diag_scan(8);
		for size in 1..4 {
			for m in 0..6 {
				let src = if m < 3 { &DEFAULT_INTRA_8X8 } else { &DEFAULT_INTER_8X8 };
				for (i, &(x, y)) in scan8.iter().enumerate() {
					s.sl[size][m][usize::from(y) * 8 + usize::from(x)] = src[i];
				}
			}
		}
		s
	}

	/// `scaling_list_data()` (§7.3.4).
	fn parse(bs: &mut BitReader) -> PsResult<Self> {
		let mut s = Self::default_lists();
		let scan4 = diag_scan(4);
		let scan8 = diag_scan(8);
		for size in 0..4 {
			let step = if size == 3 { 3 } else { 1 };
			let mut m = 0;
			while m < 6 {
				if !bs.bit() {
					// scaling_list_pred_mode_flag = 0: copy or default.
					let delta = bs.ue() as usize * step;
					if delta != 0 {
						check(delta <= m, "hevc: invalid scaling list delta")?;
						let r = m - delta;
						s.sl[size][m] = s.sl[size][r];
						if size >= 2 {
							s.dc[size - 2][m] = s.dc[size - 2][r];
						}
					} else {
						let def = Self::default_lists();
						s.sl[size][m] = def.sl[size][m];
						if size >= 2 {
							s.dc[size - 2][m] = 16;
						}
					}
				} else {
					let mut next: i32 = 8;
					let coefs = (1usize << (4 + (size << 1))).min(64);
					if size >= 2 {
						let dc = bs.se().saturating_add(8);
						check((1..=255).contains(&dc), "hevc: invalid scaling list dc")?;
						next = dc;
						s.dc[size - 2][m] = dc as u8;
					}
					for i in 0..coefs {
						let d = bs.se();
						check((-128..=127).contains(&d), "hevc: invalid scaling list delta coef")?;
						next = (next + d + 256) % 256;
						let (x, y) = if size == 0 { scan4[i] } else { scan8[i] };
						let w = if size == 0 { 4 } else { 8 };
						s.sl[size][m][usize::from(y) * w + usize::from(x)] = next as u8;
					}
				}
				m += step;
			}
		}
		// 32x32 chroma lists (matrixId 1, 2, 4, 5) copy the 16x16 ones.
		for m in [1, 2, 4, 5] {
			s.sl[3][m] = s.sl[2][m];
			s.dc[1][m] = s.dc[0][m];
		}
		Ok(s)
	}
}

/// `profile_tier_level()`: everything is skipped except the profile.
fn profile_tier_level(bs: &mut BitReader, max_sub_layers_minus1: usize) -> u8 {
	bs.skip(2 + 1);
	let profile_idc = bs.bits(5) as u8;
	bs.skip(32 + 4 + 43 + 1 + 8);
	let mut sub_profile = [false; 8];
	let mut sub_level = [false; 8];
	for i in 0..max_sub_layers_minus1 {
		sub_profile[i] = bs.bit();
		sub_level[i] = bs.bit();
	}
	if max_sub_layers_minus1 > 0 {
		for _ in max_sub_layers_minus1..8 {
			bs.skip(2);
		}
	}
	for i in 0..max_sub_layers_minus1 {
		if sub_profile[i] {
			bs.skip(88);
		}
		if sub_level[i] {
			bs.skip(8);
		}
	}
	profile_idc
}

/// `st_ref_pic_set(idx)` (§7.3.7).
pub fn parse_short_term_rps(
	bs: &mut BitReader,
	idx: usize,
	num_sets: usize,
	sets: &[ShortTermRps],
	in_slice_header: bool,
) -> PsResult<ShortTermRps> {
	let mut rps = ShortTermRps::default();
	let predict = idx != 0 && bs.bit();
	if predict {
		let delta_idx = if in_slice_header { bs.ue() as usize + 1 } else { 1 };
		check(delta_idx <= idx, "hevc: invalid delta_idx in rps")?;
		let r = sets.get(idx - delta_idx).ok_or(PsError("hevc: invalid reference rps"))?;
		let sign = bs.bit();
		let abs = bs.ue() as i32 + 1;
		check(abs <= 1 << 15, "hevc: invalid delta_rps")?;
		let delta_rps = if sign { -abs } else { abs };
		let mut used_by_curr = [false; 33];
		let mut use_delta = [true; 33];
		for j in 0..=r.num_delta_pocs {
			used_by_curr[j] = bs.bit();
			if !used_by_curr[j] {
				use_delta[j] = bs.bit();
			}
		}
		// (7-61, 7-62): rebuild sorted negative and positive lists.
		let mut neg: Vec<(i32, bool)> = Vec::new();
		let mut pos: Vec<(i32, bool)> = Vec::new();
		let r_neg = r.num_negative;
		let r_pos = r.num_delta_pocs - r.num_negative;
		// Negative pictures.
		for j in (0..r_pos).rev() {
			let d = r.delta_poc[r_neg + j] + delta_rps;
			if d < 0 && use_delta[r_neg + j] {
				neg.push((d, used_by_curr[r_neg + j]));
			}
		}
		if delta_rps < 0 && use_delta[r.num_delta_pocs] {
			neg.push((delta_rps, used_by_curr[r.num_delta_pocs]));
		}
		for j in 0..r_neg {
			let d = r.delta_poc[j] + delta_rps;
			if d < 0 && use_delta[j] {
				neg.push((d, used_by_curr[j]));
			}
		}
		// Positive pictures.
		for j in (0..r_neg).rev() {
			let d = r.delta_poc[j] + delta_rps;
			if d > 0 && use_delta[j] {
				pos.push((d, used_by_curr[j]));
			}
		}
		if delta_rps > 0 && use_delta[r.num_delta_pocs] {
			pos.push((delta_rps, used_by_curr[r.num_delta_pocs]));
		}
		for j in 0..r_pos {
			let d = r.delta_poc[r_neg + j] + delta_rps;
			if d > 0 && use_delta[r_neg + j] {
				pos.push((d, used_by_curr[r_neg + j]));
			}
		}
		check(neg.len() + pos.len() <= 32, "hevc: too many rps pictures")?;
		rps.num_negative = neg.len();
		rps.num_delta_pocs = neg.len() + pos.len();
		for (i, (d, u)) in neg.into_iter().chain(pos).enumerate() {
			rps.delta_poc[i] = d;
			rps.used[i] = u;
		}
	} else {
		let num_negative = bs.ue() as usize;
		let num_positive = bs.ue() as usize;
		check(num_negative <= 16 && num_positive <= 16, "hevc: too many rps pictures")?;
		rps.num_negative = num_negative;
		rps.num_delta_pocs = num_negative + num_positive;
		let mut poc = 0i32;
		for i in 0..num_negative {
			poc -= bs.ue() as i32 + 1;
			rps.delta_poc[i] = poc;
			rps.used[i] = bs.bit();
		}
		poc = 0;
		for i in 0..num_positive {
			poc += bs.ue() as i32 + 1;
			rps.delta_poc[num_negative + i] = poc;
			rps.used[num_negative + i] = bs.bit();
		}
	}
	let _ = num_sets;
	Ok(rps)
}

#[derive(Debug, Clone)]
pub struct Sps {
	pub id: usize,
	pub chroma_format_idc: u32,
	pub width: u32,
	pub height: u32,
	/// Conformance window in luma samples: left, right, top, bottom.
	pub crop: [u32; 4],
	pub bit_depth: u32,
	pub log2_max_poc_lsb: u32,
	pub max_dec_pic_buffering: u32,
	pub max_num_reorder: u32,
	pub max_latency_increase: u32,
	pub log2_min_cb_size: u32,
	pub log2_ctb_size: u32,
	pub log2_min_tb_size: u32,
	pub log2_max_tb_size: u32,
	pub max_transform_hierarchy_depth_inter: u32,
	pub max_transform_hierarchy_depth_intra: u32,
	pub scaling_list: Option<ScalingList>,
	pub amp_enabled: bool,
	pub sao_enabled: bool,
	pub pcm_enabled: bool,
	pub pcm_bit_depth: u32,
	pub pcm_bit_depth_chroma: u32,
	pub log2_min_pcm_cb_size: u32,
	pub log2_max_pcm_cb_size: u32,
	pub pcm_loop_filter_disabled: bool,
	pub short_term_rps: Vec<ShortTermRps>,
	pub long_term_refs_present: bool,
	pub lt_ref_pic_poc_lsb: Vec<u32>,
	pub used_by_curr_pic_lt: Vec<bool>,
	pub temporal_mvp_enabled: bool,
	pub strong_intra_smoothing: bool,
	// Derived.
	pub ctb_size: u32,
	pub ctb_width: u32,
	pub ctb_height: u32,
	pub min_cb_width: u32,
	pub min_cb_height: u32,
	pub min_tb_width: u32,
	pub min_tb_height: u32,
	pub qp_bd_offset: i32,
}

impl Sps {
	pub fn parse(rbsp: &[u8]) -> PsResult<Self> {
		let mut bs = BitReader::new(rbsp);
		bs.skip(16); // NAL header
		bs.skip(4); // sps_video_parameter_set_id
		let max_sub_layers_minus1 = bs.bits(3) as usize;
		check(max_sub_layers_minus1 < 7, "hevc: invalid sub layer count")?;
		bs.skip(1);
		let profile = profile_tier_level(&mut bs, max_sub_layers_minus1);
		let id = bs.ue() as usize;
		check(id < MAX_SPS, "hevc: invalid sps id")?;
		let chroma_format_idc = bs.ue();
		check(chroma_format_idc == 1, "hevc: only 4:2:0 is supported")?;
		let width = bs.ue();
		let height = bs.ue();
		check(
			(8..=16888).contains(&width) && (8..=16888).contains(&height),
			"hevc: invalid picture size",
		)?;
		let mut crop = [0u32; 4];
		if bs.bit() {
			for c in &mut crop {
				*c = bs.ue().saturating_mul(2);
			}
			// As FFmpeg does: a conformance window that leaves no picture is
			// invalid, so ignore it rather than reject the stream. Sums in u64
			// because each offset can be near u32::MAX.
			let [l, r, t, b] = crop.map(u64::from);
			if l + r >= u64::from(width) || t + b >= u64::from(height) {
				crop = [0; 4];
			}
		}
		let bit_depth = bs.ue().saturating_add(8);
		let bit_depth_chroma = bs.ue().saturating_add(8);
		check(bit_depth <= 12 && bit_depth == bit_depth_chroma, "hevc: unsupported bit depth")?;
		let _ = profile;
		let log2_max_poc_lsb = bs.ue().saturating_add(4);
		check(log2_max_poc_lsb <= 16, "hevc: invalid poc lsb size")?;
		let sub_layer_ordering_info = bs.bit();
		let first = if sub_layer_ordering_info { 0 } else { max_sub_layers_minus1 };
		let (mut dpb, mut reorder, mut latency) = (0, 0, 0);
		for _ in first..=max_sub_layers_minus1 {
			dpb = bs.ue() + 1;
			reorder = bs.ue();
			latency = bs.ue();
		}
		check(dpb <= 16 && reorder < dpb, "hevc: invalid dpb size")?;
		let log2_min_cb_size = bs.ue().saturating_add(3);
		let log2_ctb_size = log2_min_cb_size.saturating_add(bs.ue());
		let log2_min_tb_size = bs.ue().saturating_add(2);
		let log2_max_tb_size = log2_min_tb_size.saturating_add(bs.ue());
		check(
			(4..=6).contains(&log2_ctb_size) && log2_min_cb_size <= log2_ctb_size,
			"hevc: invalid ctb size",
		)?;
		check(
			log2_min_tb_size < log2_min_cb_size
				&& log2_max_tb_size <= 5
				&& log2_max_tb_size <= log2_ctb_size,
			"hevc: invalid transform sizes",
		)?;
		check(
			width.is_multiple_of(1 << log2_min_cb_size) && height.is_multiple_of(1 << log2_min_cb_size),
			"hevc: misaligned size",
		)?;
		let max_transform_hierarchy_depth_inter = bs.ue();
		let max_transform_hierarchy_depth_intra = bs.ue();
		let scaling_list = if bs.bit() {
			Some(if bs.bit() { ScalingList::parse(&mut bs)? } else { ScalingList::default_lists() })
		} else {
			None
		};
		let amp_enabled = bs.bit();
		let sao_enabled = bs.bit();
		let pcm_enabled = bs.bit();
		let (mut pcm_bd, mut pcm_bdc, mut pcm_min, mut pcm_max, mut pcm_lf) = (0, 0, 0, 0, false);
		if pcm_enabled {
			pcm_bd = bs.bits(4) + 1;
			pcm_bdc = bs.bits(4) + 1;
			pcm_min = bs.ue().saturating_add(3);
			pcm_max = pcm_min.saturating_add(bs.ue());
			pcm_lf = bs.bit();
			check(
				pcm_bd <= bit_depth && pcm_max <= 5.min(log2_ctb_size),
				"hevc: invalid pcm parameters",
			)?;
		}
		let num_sets = bs.ue() as usize;
		check(num_sets < MAX_SHORT_TERM_RPS, "hevc: too many rps")?;
		let mut short_term_rps = Vec::with_capacity(num_sets);
		for i in 0..num_sets {
			let rps = parse_short_term_rps(&mut bs, i, num_sets, &short_term_rps, false)?;
			short_term_rps.push(rps);
		}
		let long_term_refs_present = bs.bit();
		let mut lt_ref_pic_poc_lsb = Vec::new();
		let mut used_by_curr_pic_lt = Vec::new();
		if long_term_refs_present {
			let n = bs.ue() as usize;
			check(n <= 32, "hevc: too many long-term refs")?;
			for _ in 0..n {
				lt_ref_pic_poc_lsb.push(bs.bits(log2_max_poc_lsb));
				used_by_curr_pic_lt.push(bs.bit());
			}
		}
		let temporal_mvp_enabled = bs.bit();
		let strong_intra_smoothing = bs.bit();
		// VUI and extensions follow; nothing in them affects Main/Main 10
		// decoding.
		check(!bs.overrun(), "hevc: truncated sps")?;
		let ctb_size = 1 << log2_ctb_size;
		Ok(Self {
			id,
			chroma_format_idc,
			width,
			height,
			crop,
			bit_depth,
			log2_max_poc_lsb,
			max_dec_pic_buffering: dpb,
			max_num_reorder: reorder,
			max_latency_increase: latency,
			log2_min_cb_size,
			log2_ctb_size,
			log2_min_tb_size,
			log2_max_tb_size,
			max_transform_hierarchy_depth_inter,
			max_transform_hierarchy_depth_intra,
			scaling_list,
			amp_enabled,
			sao_enabled,
			pcm_enabled,
			pcm_bit_depth: pcm_bd,
			pcm_bit_depth_chroma: pcm_bdc,
			log2_min_pcm_cb_size: pcm_min,
			log2_max_pcm_cb_size: pcm_max,
			pcm_loop_filter_disabled: pcm_lf,
			short_term_rps,
			long_term_refs_present,
			lt_ref_pic_poc_lsb,
			used_by_curr_pic_lt,
			temporal_mvp_enabled,
			strong_intra_smoothing,
			ctb_size,
			ctb_width: width.div_ceil(ctb_size),
			ctb_height: height.div_ceil(ctb_size),
			min_cb_width: width >> log2_min_cb_size,
			min_cb_height: height >> log2_min_cb_size,
			min_tb_width: width >> log2_min_tb_size,
			min_tb_height: height >> log2_min_tb_size,
			qp_bd_offset: 6 * (bit_depth as i32 - 8),
		})
	}
}

#[derive(Debug, Clone)]
pub struct Pps {
	pub id: usize,
	pub sps_id: usize,
	pub dependent_slice_segments_enabled: bool,
	pub output_flag_present: bool,
	pub num_extra_slice_header_bits: u32,
	pub sign_data_hiding: bool,
	pub cabac_init_present: bool,
	pub num_ref_idx_l0_default: u32,
	pub num_ref_idx_l1_default: u32,
	pub init_qp: i32,
	pub constrained_intra_pred: bool,
	pub transform_skip_enabled: bool,
	pub cu_qp_delta_enabled: bool,
	pub diff_cu_qp_delta_depth: u32,
	pub cb_qp_offset: i32,
	pub cr_qp_offset: i32,
	pub slice_chroma_qp_offsets_present: bool,
	pub weighted_pred: bool,
	pub weighted_bipred: bool,
	pub transquant_bypass_enabled: bool,
	pub tiles_enabled: bool,
	pub entropy_coding_sync: bool,
	pub uniform_spacing: bool,
	pub column_widths: Vec<u32>,
	pub row_heights: Vec<u32>,
	pub loop_filter_across_tiles: bool,
	pub loop_filter_across_slices: bool,
	pub deblocking_override_enabled: bool,
	pub deblocking_disabled: bool,
	pub beta_offset: i32,
	pub tc_offset: i32,
	pub scaling_list: Option<ScalingList>,
	pub lists_modification_present: bool,
	pub log2_parallel_merge_level: u32,
	pub slice_header_extension_present: bool,
	// Explicit tile layout (in CTBs) as signalled; resolved against an SPS
	// by [`Pps::layout`].
	num_tile_columns: u32,
	num_tile_rows: u32,
}

/// CTB address tables of a PPS/SPS pair (§6.5.1-6.5.2).
#[derive(Debug, Clone)]
pub struct Layout {
	pub col_bd: Vec<u32>,
	pub row_bd: Vec<u32>,
	pub ctb_addr_rs_to_ts: Vec<u32>,
	pub ctb_addr_ts_to_rs: Vec<u32>,
	pub tile_id: Vec<u32>,
	/// z-scan order address of each min TB (row-major over min TBs), with
	/// one extra row/column of `-1` padding on the top/left.
	pub min_tb_addr_zs: Vec<i32>,
	pub min_tb_stride: usize,
}

impl Pps {
	pub fn parse(rbsp: &[u8], sps_list: &[Option<std::sync::Arc<Sps>>]) -> PsResult<Self> {
		let mut bs = BitReader::new(rbsp);
		bs.skip(16);
		let id = bs.ue() as usize;
		check(id < MAX_PPS, "hevc: invalid pps id")?;
		let sps_id = bs.ue() as usize;
		check(sps_id < MAX_SPS, "hevc: invalid sps id in pps")?;
		let sps = sps_list
			.get(sps_id)
			.and_then(|s| s.as_ref())
			.ok_or(PsError("hevc: pps refers to missing sps"))?;
		let dependent_slice_segments_enabled = bs.bit();
		let output_flag_present = bs.bit();
		let num_extra_slice_header_bits = bs.bits(3);
		let sign_data_hiding = bs.bit();
		let cabac_init_present = bs.bit();
		let num_ref_idx_l0_default = bs.ue() + 1;
		let num_ref_idx_l1_default = bs.ue() + 1;
		check(num_ref_idx_l0_default <= 15 && num_ref_idx_l1_default <= 15, "hevc: invalid ref count")?;
		let init_qp = bs.se().saturating_add(26);
		check(init_qp >= -sps.qp_bd_offset && init_qp <= 51, "hevc: invalid init qp")?;
		let constrained_intra_pred = bs.bit();
		let transform_skip_enabled = bs.bit();
		let cu_qp_delta_enabled = bs.bit();
		let diff_cu_qp_delta_depth = if cu_qp_delta_enabled { bs.ue() } else { 0 };
		check(
			diff_cu_qp_delta_depth <= sps.log2_ctb_size - sps.log2_min_cb_size,
			"hevc: invalid qp delta depth",
		)?;
		let cb_qp_offset = bs.se();
		let cr_qp_offset = bs.se();
		check(
			(-12..=12).contains(&cb_qp_offset) && (-12..=12).contains(&cr_qp_offset),
			"hevc: invalid chroma qp",
		)?;
		let slice_chroma_qp_offsets_present = bs.bit();
		let weighted_pred = bs.bit();
		let weighted_bipred = bs.bit();
		let transquant_bypass_enabled = bs.bit();
		let tiles_enabled = bs.bit();
		let entropy_coding_sync = bs.bit();
		let (mut num_tile_columns, mut num_tile_rows, mut uniform_spacing) = (1, 1, true);
		let mut column_widths = Vec::new();
		let mut row_heights = Vec::new();
		let mut loop_filter_across_tiles = true;
		if tiles_enabled {
			num_tile_columns = bs.ue() + 1;
			num_tile_rows = bs.ue() + 1;
			check(
				num_tile_columns <= sps.ctb_width && num_tile_rows <= sps.ctb_height,
				"hevc: invalid tile count",
			)?;
			uniform_spacing = bs.bit();
			if !uniform_spacing {
				for _ in 0..num_tile_columns - 1 {
					column_widths.push(bs.ue() + 1);
				}
				for _ in 0..num_tile_rows - 1 {
					row_heights.push(bs.ue() + 1);
				}
			}
			loop_filter_across_tiles = bs.bit();
		}
		let loop_filter_across_slices = bs.bit();
		let (mut deblocking_override_enabled, mut deblocking_disabled, mut beta_offset, mut tc_offset) =
			(false, false, 0, 0);
		if bs.bit() {
			deblocking_override_enabled = bs.bit();
			deblocking_disabled = bs.bit();
			if !deblocking_disabled {
				beta_offset = bs.se().saturating_mul(2);
				tc_offset = bs.se().saturating_mul(2);
				check(
					(-12..=12).contains(&beta_offset) && (-12..=12).contains(&tc_offset),
					"hevc: invalid offsets",
				)?;
			}
		}
		let scaling_list = if bs.bit() { Some(ScalingList::parse(&mut bs)?) } else { None };
		let lists_modification_present = bs.bit();
		let log2_parallel_merge_level = bs.ue().saturating_add(2);
		check(log2_parallel_merge_level <= sps.log2_ctb_size, "hevc: invalid merge level")?;
		let slice_header_extension_present = bs.bit();
		if bs.bit() {
			// pps_range_extension etc.: only relevant to RExt profiles.
			let range = bs.bit();
			let _multilayer = bs.bit();
			let _3d = bs.bit();
			let _scc = bs.bit();
			let _other = bs.bits(4);
			if range {
				let log2_max_ts = if transform_skip_enabled { bs.ue().saturating_add(2) } else { 2 };
				let cross_component = bs.bit();
				let chroma_qp_offset_list = bs.bit();
				check(
					log2_max_ts == 2 && !cross_component && !chroma_qp_offset_list,
					"hevc: range extension tools are not supported",
				)?;
			}
		}
		Ok(Self {
			id,
			sps_id,
			dependent_slice_segments_enabled,
			output_flag_present,
			num_extra_slice_header_bits,
			sign_data_hiding,
			cabac_init_present,
			num_ref_idx_l0_default,
			num_ref_idx_l1_default,
			init_qp,
			constrained_intra_pred,
			transform_skip_enabled,
			cu_qp_delta_enabled,
			diff_cu_qp_delta_depth,
			cb_qp_offset,
			cr_qp_offset,
			slice_chroma_qp_offsets_present,
			weighted_pred,
			weighted_bipred,
			transquant_bypass_enabled,
			tiles_enabled,
			entropy_coding_sync,
			uniform_spacing,
			column_widths,
			row_heights,
			loop_filter_across_tiles,
			loop_filter_across_slices,
			deblocking_override_enabled,
			deblocking_disabled,
			beta_offset,
			tc_offset,
			scaling_list,
			lists_modification_present,
			log2_parallel_merge_level,
			slice_header_extension_present,
			num_tile_columns,
			num_tile_rows,
		})
	}

	/// Builds the tile and scan conversion tables for `sps`.
	pub fn layout(&self, sps: &Sps) -> PsResult<Layout> {
		let (cw, ch) = (sps.ctb_width, sps.ctb_height);
		let ncol = self.num_tile_columns as usize;
		let nrow = self.num_tile_rows as usize;
		let mut col_w = vec![0u32; ncol];
		let mut row_h = vec![0u32; nrow];
		if self.uniform_spacing {
			for i in 0..ncol {
				col_w[i] = ((i as u32 + 1) * cw) / ncol as u32 - (i as u32 * cw) / ncol as u32;
			}
			for i in 0..nrow {
				row_h[i] = ((i as u32 + 1) * ch) / nrow as u32 - (i as u32 * ch) / nrow as u32;
			}
		} else {
			let mut sum = 0;
			for i in 0..ncol - 1 {
				col_w[i] = self.column_widths[i];
				sum += col_w[i];
			}
			check(sum < cw, "hevc: invalid tile columns")?;
			col_w[ncol - 1] = cw - sum;
			let mut sum = 0;
			for i in 0..nrow - 1 {
				row_h[i] = self.row_heights[i];
				sum += row_h[i];
			}
			check(sum < ch, "hevc: invalid tile rows")?;
			row_h[nrow - 1] = ch - sum;
		}
		let mut col_bd = vec![0u32; ncol + 1];
		for i in 0..ncol {
			col_bd[i + 1] = col_bd[i] + col_w[i];
		}
		let mut row_bd = vec![0u32; nrow + 1];
		for i in 0..nrow {
			row_bd[i + 1] = row_bd[i] + row_h[i];
		}
		let n = (cw * ch) as usize;
		let mut rs_to_ts = vec![0u32; n];
		for rs in 0..n as u32 {
			let (tbx, tby) = (rs % cw, rs / cw);
			let tile_x = (0..ncol).rev().find(|&i| tbx >= col_bd[i]).unwrap_or(0);
			let tile_y = (0..nrow).rev().find(|&j| tby >= row_bd[j]).unwrap_or(0);
			let mut v = 0;
			for i in 0..tile_x {
				v += row_h[tile_y] * col_w[i];
			}
			for j in 0..tile_y {
				v += cw * row_h[j];
			}
			v += (tby - row_bd[tile_y]) * col_w[tile_x] + tbx - col_bd[tile_x];
			rs_to_ts[rs as usize] = v;
		}
		let mut ts_to_rs = vec![0u32; n];
		for (rs, &ts) in rs_to_ts.iter().enumerate() {
			ts_to_rs[ts as usize] = rs as u32;
		}
		let mut tile_id = vec![0u32; n];
		let mut tid = 0;
		for j in 0..nrow {
			for i in 0..ncol {
				for y in row_bd[j]..row_bd[j + 1] {
					for x in col_bd[i]..col_bd[i + 1] {
						tile_id[rs_to_ts[(y * cw + x) as usize] as usize] = tid;
					}
				}
				tid += 1;
			}
		}
		// MinTbAddrZs (6-10), with a -1 border row/column.
		let log2_diff = sps.log2_ctb_size - sps.log2_min_tb_size;
		let tbw = (cw << log2_diff) as usize;
		let tbh = (ch << log2_diff) as usize;
		let stride = tbw + 1;
		let mut zs = vec![-1i32; stride * (tbh + 1)];
		for y in 0..tbh {
			for x in 0..tbw {
				let tb_x = (x << sps.log2_min_tb_size) >> sps.log2_ctb_size;
				let tb_y = (y << sps.log2_min_tb_size) >> sps.log2_ctb_size;
				let rs = cw as usize * tb_y + tb_x;
				let mut v = (rs_to_ts[rs] as i32) << (log2_diff * 2);
				for i in 0..log2_diff {
					let m = 1 << i;
					v += if m & x != 0 { m * m } else { 0 } as i32
						+ if m & y != 0 { 2 * m * m } else { 0 } as i32;
				}
				zs[(y + 1) * stride + x + 1] = v;
			}
		}
		Ok(Layout {
			col_bd,
			row_bd,
			ctb_addr_rs_to_ts: rs_to_ts,
			ctb_addr_ts_to_rs: ts_to_rs,
			tile_id,
			min_tb_addr_zs: zs,
			min_tb_stride: stride,
		})
	}
}

impl Layout {
	/// MinTbAddrZs at min-TB coordinates (may be -1 outside).
	#[inline]
	pub fn zs(&self, x_tb: i32, y_tb: i32) -> i32 {
		let (x, y) = ((x_tb + 1) as usize, (y_tb + 1) as usize);
		self.min_tb_addr_zs.get(y * self.min_tb_stride + x).copied().unwrap_or(-1)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn diagonal_scan_order() {
		let s = diag_scan(4);
		assert_eq!(&s[..6], &[(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]);
		assert_eq!(s.len(), 16);
	}
}
