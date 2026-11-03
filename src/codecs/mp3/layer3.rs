//! Layer III bitstream decoding up to (and including) the hybrid filterbank:
//!
//! side info → scalefactors → Huffman + requantisation → stereo processing →
//! short-block reordering → alias reduction → IMDCT with overlap-add.
//!
//! These are the stages of ISO/IEC 11172-3 Annex A, in the same order FFmpeg's
//! `mpegaudiodec_template.c` runs them. The arithmetic follows the compact,
//! well-tested formulation of minimp3 (public domain). All buffers are
//! fixed-size arrays indexed with bounds that are either proven by the tables
//! or checked explicitly, so corrupt input degrades into silence, never a
//! panic or out-of-bounds access.

#![allow(clippy::excessive_precision)]

use super::bits::BitReader;
use super::header::{ChannelMode, FrameHeader};
use super::tables::*;

/// Coefficients per granule per channel.
pub const GRANULE_SIZE: usize = 576;

const SHORT_BLOCK_TYPE: u8 = 2;
const STOP_BLOCK_TYPE: u8 = 3;

/// `BITS_DEQUANTIZER_OUT` in minimp3: output is scaled so the synthesis
/// filter produces 16-bit range samples directly.
const BITS_DEQUANTIZER_OUT: i32 = -1;
const MAX_SCF: i32 = 255 + BITS_DEQUANTIZER_OUT * 4 - 210;
const MAX_SCFI: i32 = (MAX_SCF + 3) & !3;

/// Per-granule, per-channel side information.
#[derive(Debug, Clone, Copy, Default)]
pub struct GranuleInfo {
	/// Scalefactor band widths for this block type; zero-terminated.
	pub sfb: &'static [u8],
	pub part_23_length: u16,
	pub big_values: u16,
	pub scalefac_compress: u16,
	pub global_gain: u8,
	pub block_type: u8,
	pub mixed_block_flag: bool,
	pub n_long_sfb: u8,
	pub n_short_sfb: u8,
	pub table_select: [u8; 3],
	pub region_count: [u8; 3],
	pub subblock_gain: [u8; 3],
	pub preflag: bool,
	pub scalefac_scale: u8,
	pub count1_table: bool,
	pub scfsi: u8,
}

/// Header properties the Layer III stages test repeatedly.
#[derive(Debug, Clone, Copy)]
pub struct StreamFlags {
	pub mpeg1: bool,
	pub channels: usize,
	pub ms_stereo: bool,
	pub i_stereo: bool,
	/// Index into the 9 supported sample rates (0 = 11.025 kHz MPEG-2.5 …
	/// 8 = 32 kHz MPEG-1), minimp3's `HDR_GET_MY_SAMPLE_RATE`.
	pub rate_index: usize,
}

impl StreamFlags {
	pub fn from_header(h: &FrameHeader) -> Self {
		let joint = h.channel_mode == ChannelMode::JointStereo;
		let raw = h.raw;
		let rate_index =
			usize::from((raw[2] >> 2) & 3) + usize::from(((raw[1] >> 3) & 1) + ((raw[1] >> 4) & 1)) * 3;
		Self {
			mpeg1: h.is_mpeg1(),
			channels: usize::from(h.channels()),
			// Stereo tools are only defined for joint-stereo frames.
			ms_stereo: joint && h.mode_extension & 0b10 != 0,
			i_stereo: joint && h.mode_extension & 0b01 != 0,
			rate_index,
		}
	}
}

/// Reads the side information. Returns `main_data_begin`, or `None` if the
/// side info is self-inconsistent (FFmpeg rejects the same conditions).
pub fn read_side_info(
	bs: &mut BitReader,
	granules: &mut [GranuleInfo; 4],
	flags: &StreamFlags,
) -> Option<usize> {
	let sr_idx = flags.rate_index.saturating_sub(1).min(7);
	let mono = flags.channels == 1;
	let mut gr_count: u32 = if mono { 1 } else { 2 };
	let main_data_begin;
	let mut scfsi: u32 = 0;

	if flags.mpeg1 {
		gr_count *= 2;
		main_data_begin = bs.get_bits(9);
		scfsi = bs.get_bits(7 + gr_count);
	} else {
		main_data_begin = bs.get_bits(8 + gr_count) >> gr_count;
	}

	let mut part_23_sum = 0usize;
	for gr in granules.iter_mut().take(gr_count as usize) {
		if mono {
			scfsi <<= 4;
		}
		gr.part_23_length = bs.get_bits(12) as u16;
		part_23_sum += usize::from(gr.part_23_length);
		gr.big_values = bs.get_bits(9) as u16;
		if gr.big_values > 288 {
			return None;
		}
		gr.global_gain = bs.get_bits(8) as u8;
		gr.scalefac_compress = bs.get_bits(if flags.mpeg1 { 4 } else { 9 }) as u16;
		gr.sfb = &SCF_LONG[sr_idx];
		gr.n_long_sfb = 22;
		gr.n_short_sfb = 0;

		let tables;
		if bs.get_bits(1) != 0 {
			gr.block_type = bs.get_bits(2) as u8;
			if gr.block_type == 0 {
				return None;
			}
			gr.mixed_block_flag = bs.get_bits(1) != 0;
			gr.region_count = [7, 255, 0];
			if gr.block_type == SHORT_BLOCK_TYPE {
				scfsi &= 0x0F0F;
				if !gr.mixed_block_flag {
					gr.region_count[0] = 8;
					gr.sfb = &SCF_SHORT[sr_idx];
					gr.n_long_sfb = 0;
					gr.n_short_sfb = 39;
				} else {
					gr.sfb = &SCF_MIXED[sr_idx];
					gr.n_long_sfb = if flags.mpeg1 { 8 } else { 6 };
					gr.n_short_sfb = 30;
				}
			}
			tables = bs.get_bits(10) << 5;
			for gain in gr.subblock_gain.iter_mut() {
				*gain = bs.get_bits(3) as u8;
			}
		} else {
			gr.block_type = 0;
			gr.mixed_block_flag = false;
			tables = bs.get_bits(15);
			gr.region_count[0] = bs.get_bits(4) as u8;
			gr.region_count[1] = bs.get_bits(3) as u8;
			gr.region_count[2] = 255;
		}
		gr.table_select = [(tables >> 10) as u8, ((tables >> 5) & 31) as u8, (tables & 31) as u8];
		gr.preflag = if flags.mpeg1 { bs.get_bits(1) != 0 } else { gr.scalefac_compress >= 500 };
		gr.scalefac_scale = bs.get_bits(1) as u8;
		gr.count1_table = bs.get_bits(1) != 0;
		gr.scfsi = ((scfsi >> 12) & 15) as u8;
		scfsi <<= 4;
	}

	// Scalefactor selection information only applies to granule 1; granule 0
	// must never reuse scalefactors (the bits that land there are the
	// "private bits" of the header). The reference decoder leaves them in.
	if flags.mpeg1 {
		for gr in granules.iter_mut().take(flags.channels) {
			gr.scfsi = 0;
		}
	}

	let main_data_begin = main_data_begin as usize;
	if part_23_sum + bs.pos() > bs.limit() + main_data_begin * 8 {
		return None;
	}
	Some(main_data_begin)
}

/// `y * 2^(-exp_q2 / 4)`, computed in steps that stay inside f32 range.
fn ldexp_q2(mut y: f32, exp_q2: i32) -> f32 {
	let mut exp_q2 = exp_q2.max(0);
	loop {
		let e = exp_q2.min(30 * 4);
		y *= EXPFRAC[(e & 3) as usize] * ((1i32 << 30) >> (e >> 2)) as f32;
		exp_q2 -= e;
		if exp_q2 <= 0 {
			return y;
		}
	}
}

fn read_scalefactors(
	scf: &mut [u8; 40],
	ist_pos: &mut [u8; 40],
	scf_size: &[u8; 4],
	scf_count: &[u8],
	bs: &mut BitReader,
	mut scfsi: i32,
) {
	let mut at = 0usize;
	for (i, &count) in scf_count.iter().take(4).enumerate() {
		if count == 0 {
			break;
		}
		let cnt = usize::from(count);
		let end = (at + cnt).min(scf.len());
		if scfsi & 8 != 0 {
			scf[at..end].copy_from_slice(&ist_pos[at..end]);
		} else {
			let bits = u32::from(scf_size[i]);
			if bits == 0 {
				scf[at..end].fill(0);
				ist_pos[at..end].fill(0);
			} else {
				let max_scf: i32 = if scfsi < 0 { (1 << bits) - 1 } else { -1 };
				for k in at..end {
					let s = bs.get_bits(bits) as i32;
					ist_pos[k] = if s == max_scf { 0xFF } else { s as u8 };
					scf[k] = s as u8;
				}
			}
		}
		at = end;
		scfsi = scfsi.wrapping_mul(2);
	}
	for slot in scf.iter_mut().skip(at).take(3) {
		*slot = 0;
	}
}

/// Decodes scalefactors and converts them (with global gain, subblock gain,
/// pre-emphasis and the M/S normalisation) into linear band multipliers.
pub fn decode_scalefactors(
	flags: &StreamFlags,
	ist_pos: &mut [u8; 40],
	bs: &mut BitReader,
	gr: &GranuleInfo,
	scf: &mut [f32; 40],
	ch: usize,
) {
	let partition_row = usize::from(gr.n_short_sfb != 0) + usize::from(gr.n_long_sfb == 0);
	let mut partition_offset = 0usize;
	let mut scf_size = [0u8; 4];
	let scf_shift = i32::from(gr.scalefac_scale) + 1;
	let mut scfsi = i32::from(gr.scfsi);

	if flags.mpeg1 {
		let part = SCFC_DECODE[usize::from(gr.scalefac_compress) & 15];
		scf_size[0] = part >> 2;
		scf_size[1] = part >> 2;
		scf_size[2] = part & 3;
		scf_size[3] = part & 3;
	} else {
		let ist = usize::from(flags.i_stereo && ch != 0);
		let mut sfc = i32::from(gr.scalefac_compress >> ist);
		let mut k = ist * 3 * 4;
		while sfc >= 0 && k + 4 <= SCF_MOD.len() {
			let mut modprod = 1i32;
			for i in (0..4).rev() {
				let m = i32::from(SCF_MOD[k + i]);
				scf_size[i] = (sfc / modprod % m) as u8;
				modprod *= m;
			}
			sfc -= modprod;
			k += 4;
		}
		partition_offset = k;
		scfsi = -16;
	}
	let partition = SCF_PARTITIONS[partition_row].get(partition_offset..).unwrap_or(&[]);

	let mut iscf = [0u8; 40];
	read_scalefactors(&mut iscf, ist_pos, &scf_size, partition, bs, scfsi);

	let n_long = usize::from(gr.n_long_sfb);
	let n_short = usize::from(gr.n_short_sfb);
	if n_short != 0 {
		let sh = 3 - scf_shift;
		for i in (0..n_short).step_by(3) {
			for (w, &gain) in gr.subblock_gain.iter().enumerate() {
				if let Some(v) = iscf.get_mut(n_long + i + w) {
					*v = v.wrapping_add(gain << sh);
				}
			}
		}
	} else if gr.preflag {
		for (i, &p) in PREAMP.iter().enumerate() {
			iscf[11 + i] = iscf[11 + i].wrapping_add(p);
		}
	}

	let ms_stereo = flags.ms_stereo;
	let gain_exp =
		i32::from(gr.global_gain) + BITS_DEQUANTIZER_OUT * 4 - 210 - if ms_stereo { 2 } else { 0 };
	let gain = ldexp_q2((1 << (MAX_SCFI / 4)) as f32, MAX_SCFI - gain_exp);
	for i in 0..(n_long + n_short).min(40) {
		scf[i] = ldexp_q2(gain, i32::from(iscf[i]) << scf_shift);
	}
}

/// `|x|^(4/3)` for x in 0..=8206 (15 + 13 linbits), via table + polynomial.
fn pow_43(x: i32) -> f32 {
	if x < 129 {
		return POW43[(16 + x) as usize];
	}
	let mut x = x;
	let mut mult = 256.0f32;
	if x < 1024 {
		mult = 16.0;
		x <<= 3;
	}
	let sign = (2 * x) & 64;
	let frac = ((x & 63) - sign) as f32 / ((x & !63) + sign) as f32;
	let index = (16 + ((x + sign) >> 6)) as usize;
	POW43[index.min(POW43.len() - 1)] * (1.0 + frac * ((4.0 / 3.0) + frac * (2.0 / 9.0))) * mult
}

/// Huffman decoding of the big-values and count1 regions with inline
/// requantisation (`sign * |v|^(4/3) * scale`). Writes into `dst`, which the
/// caller zeroed. Always leaves the reader at `limit` (the end of
/// `part2_3_length`), like the reference decoder.
pub fn huffman(
	dst: &mut [f32],
	bs: &mut BitReader,
	gr: &GranuleInfo,
	scf: &[f32; 40],
	limit: usize,
) {
	huffman_inner(dst, bs, gr, scf, limit);
	bs.set_pos(limit);
}

fn huffman_inner(
	dst: &mut [f32],
	bs: &mut BitReader,
	gr: &GranuleInfo,
	scf: &[f32; 40],
	limit: usize,
) {
	let sfb = gr.sfb;
	let mut sfb_at = 0usize;
	let mut scf_at = 0usize;
	let mut di = 0usize;
	let mut one = 0.0f32;
	let mut big_val_cnt = i32::from(gr.big_values);
	let mut region = 0usize;

	while big_val_cnt > 0 {
		if region >= 3 {
			return;
		}
		let tab_num = usize::from(gr.table_select[region] & 31);
		let mut sfb_cnt = i32::from(gr.region_count[region]);
		region += 1;
		let base = HUFF_TAB_INDEX[tab_num] as isize;
		let linbits = u32::from(LINBITS[tab_num]);

		loop {
			let np = i32::from(sfb.get(sfb_at).copied().unwrap_or(0) / 2);
			sfb_at += 1;
			if np == 0 {
				return; // ran off the band table: corrupt big_values/region data
			}
			let mut pairs = big_val_cnt.min(np);
			one = scf.get(scf_at).copied().unwrap_or(0.0);
			scf_at += 1;

			while pairs > 0 {
				let mut w = 5u32;
				let Some(mut leaf) = huff_entry(base, bs.peek(w) as isize) else { return };
				while leaf < 0 {
					bs.skip(w);
					w = (leaf & 7) as u32;
					if w == 0 {
						return;
					}
					let Some(next) = huff_entry(base, bs.peek(w) as isize - (leaf >> 3) as isize) else {
						return;
					};
					leaf = next;
				}
				bs.skip((leaf >> 8) as u32);

				if di + 2 > dst.len() {
					return;
				}
				for _ in 0..2 {
					let mut lsb = i32::from(leaf & 0x0F);
					if lsb == 15 && linbits != 0 {
						lsb += bs.peek(linbits) as i32;
						bs.skip(linbits);
						let negative = bs.peek(1) != 0;
						let magnitude = one * pow_43(lsb);
						dst[di] = if negative { -magnitude } else { magnitude };
					} else {
						let sign = bs.peek(1) as i32;
						dst[di] = POW43[(16 + lsb - 16 * sign) as usize] * one;
					}
					if lsb != 0 {
						bs.skip(1);
					}
					di += 1;
					leaf >>= 4;
				}
				pairs -= 1;
			}

			big_val_cnt -= np;
			sfb_cnt -= 1;
			if big_val_cnt <= 0 || sfb_cnt < 0 {
				break;
			}
		}
	}

	// count1 region: quadruples of values in {-1, 0, 1}.
	let codebook: &[u8] = if gr.count1_table { &HUFF_COUNT1_B } else { &HUFF_COUNT1_A };
	let mut np = 1 - big_val_cnt;
	loop {
		let Some(&first) = codebook.get(bs.peek(4) as usize) else { return };
		let mut leaf = u32::from(first);
		if leaf & 8 == 0 {
			let n = leaf & 3;
			let extra = bs.peek(4 + n) & ((1u32 << n) - 1);
			let Some(&second) = codebook.get(((leaf >> 3) + extra) as usize) else { return };
			leaf = u32::from(second);
		}
		bs.skip(leaf & 7);
		if bs.pos() > limit {
			return;
		}
		if di + 4 > dst.len() {
			return;
		}
		for half in 0..2 {
			np -= 1;
			if np == 0 {
				np = i32::from(sfb.get(sfb_at).copied().unwrap_or(0) / 2);
				sfb_at += 1;
				if np == 0 {
					return;
				}
				one = scf.get(scf_at).copied().unwrap_or(0.0);
				scf_at += 1;
			}
			for s in [half * 2, half * 2 + 1] {
				if leaf & (128 >> s) != 0 {
					dst[di + s] = if bs.peek(1) != 0 { -one } else { one };
					bs.skip(1);
				}
			}
		}
		di += 4;
	}
}

#[inline]
fn huff_entry(base: isize, offset: isize) -> Option<i16> {
	let index = usize::try_from(base + offset).ok()?;
	HUFF_TABS.get(index).copied()
}

fn midside_stereo(buf: &mut [f32; 2 * GRANULE_SIZE], start: usize, n: usize) {
	let end = (start + n).min(GRANULE_SIZE);
	for i in start..end {
		let a = buf[i];
		let b = buf[i + GRANULE_SIZE];
		buf[i] = a + b;
		buf[i + GRANULE_SIZE] = a - b;
	}
}

fn intensity_stereo_band(
	buf: &mut [f32; 2 * GRANULE_SIZE],
	start: usize,
	n: usize,
	kl: f32,
	kr: f32,
) {
	let end = (start + n).min(GRANULE_SIZE);
	for i in start..end {
		buf[i + GRANULE_SIZE] = buf[i] * kr;
		buf[i] *= kl;
	}
}

/// Finds, per short window (or for the long block), the highest band whose
/// right channel still carries data; intensity stereo applies above it.
fn stereo_top_band(right: &[f32], sfb: &[u8], nbands: usize, max_band: &mut [i32; 3]) {
	*max_band = [-1; 3];
	let mut at = 0usize;
	for (i, &width) in sfb.iter().take(nbands).enumerate() {
		let width = usize::from(width);
		let band = right.get(at..at + width).unwrap_or(&[]);
		if band.iter().any(|&v| v != 0.0) {
			max_band[i % 3] = i as i32;
		}
		at += width;
	}
}

fn stereo_process(
	buf: &mut [f32; 2 * GRANULE_SIZE],
	ist_pos: &[u8; 40],
	sfb: &[u8],
	flags: &StreamFlags,
	max_band: &[i32; 3],
	mpeg2_sh: u16,
) {
	let max_pos: u32 = if flags.mpeg1 { 7 } else { 64 };
	let mut at = 0usize;
	for (i, &width) in sfb.iter().enumerate().take(ist_pos.len()) {
		if width == 0 {
			break;
		}
		let width = usize::from(width);
		let ipos = u32::from(ist_pos[i]);
		if i as i32 > max_band[i % 3] && ipos < max_pos {
			let s = if flags.ms_stereo { std::f32::consts::SQRT_2 } else { 1.0 };
			let (kl, kr) = if flags.mpeg1 {
				(PAN[2 * ipos as usize], PAN[2 * ipos as usize + 1])
			} else {
				let k = ldexp_q2(1.0, (((ipos + 1) >> 1) << mpeg2_sh) as i32);
				if ipos & 1 != 0 { (k, 1.0) } else { (1.0, k) }
			};
			intensity_stereo_band(buf, at, width, kl * s, kr * s);
		} else if flags.ms_stereo {
			midside_stereo(buf, at, width);
		}
		at += width;
	}
}

/// Intensity stereo (optionally combined with M/S) for one granule.
/// `granules` holds the left and right channel side info.
pub fn intensity_stereo(
	buf: &mut [f32; 2 * GRANULE_SIZE],
	ist_pos: &mut [u8; 40],
	granules: &[GranuleInfo],
	flags: &StreamFlags,
) {
	let (Some(left), Some(right)) = (granules.first(), granules.get(1)) else { return };
	let n_sfb = usize::from(left.n_long_sfb) + usize::from(left.n_short_sfb);
	let max_blocks = if left.n_short_sfb != 0 { 3 } else { 1 };
	let mut max_band = [-1i32; 3];

	stereo_top_band(&buf[GRANULE_SIZE..], left.sfb, n_sfb, &mut max_band);
	if left.n_long_sfb != 0 {
		let m = max_band[0].max(max_band[1]).max(max_band[2]);
		max_band = [m; 3];
	}
	let default_pos = if flags.mpeg1 { 3 } else { 0 };
	for (i, &top_band) in max_band.iter().enumerate().take(max_blocks) {
		let Some(itop) = (n_sfb + i).checked_sub(max_blocks) else { continue };
		let Some(prev) = itop.checked_sub(max_blocks) else { continue };
		if itop < ist_pos.len() {
			ist_pos[itop] = if top_band >= prev as i32 { default_pos } else { ist_pos[prev] };
		}
	}
	stereo_process(buf, ist_pos, left.sfb, flags, &max_band, right.scalefac_compress & 1);
}

/// Plain mid/side stereo over the whole granule.
pub fn midside_all(buf: &mut [f32; 2 * GRANULE_SIZE]) {
	midside_stereo(buf, 0, GRANULE_SIZE);
}

/// Reorders short-block coefficients from (window, band, freq) scan order
/// into (band, freq, window) order expected by the short IMDCT.
pub fn reorder(grbuf: &mut [f32], scratch: &mut [f32], sfb: &[u8]) {
	let mut src = 0usize;
	let mut dst = 0usize;
	for &len in sfb.iter().step_by(3) {
		let len = usize::from(len);
		if len == 0 || src + 3 * len > grbuf.len() || dst + 3 * len > scratch.len() {
			break;
		}
		for i in 0..len {
			scratch[dst] = grbuf[src + i];
			scratch[dst + 1] = grbuf[src + i + len];
			scratch[dst + 2] = grbuf[src + i + 2 * len];
			dst += 3;
		}
		src += 3 * len;
	}
	grbuf[..dst].copy_from_slice(&scratch[..dst]);
}

/// Alias-reduction butterflies between adjacent long-block subbands.
pub fn antialias(grbuf: &mut [f32], nbands: usize) {
	for band in 0..nbands.min(31) {
		let base = band * 18;
		for i in 0..8 {
			let u = grbuf[base + 18 + i];
			let d = grbuf[base + 17 - i];
			grbuf[base + 18 + i] = u * ANTIALIAS[0][i] - d * ANTIALIAS[1][i];
			grbuf[base + 17 - i] = u * ANTIALIAS[1][i] + d * ANTIALIAS[0][i];
		}
	}
}

/// 9-point DCT-III used by the 36-point IMDCT.
fn dct3_9(y: &mut [f32; 9]) {
	let mut s0 = y[0];
	let mut s2 = y[2];
	let mut s4 = y[4];
	let mut s6 = y[6];
	let mut s8 = y[8];
	let mut t0 = s0 + s6 * 0.5;
	s0 -= s6;
	let mut t4 = (s4 + s2) * 0.93969262;
	let mut t2 = (s8 + s2) * 0.76604444;
	s6 = (s4 - s8) * 0.17364818;
	s4 += s8 - s2;

	s2 = s0 - s4 * 0.5;
	y[4] = s4 + s0;
	s8 = t0 - t2 + s6;
	s0 = t0 - t4 + t2;
	s4 = t0 + t4 - s6;

	let mut s1 = y[1];
	let mut s3 = y[3];
	let mut s5 = y[5];
	let mut s7 = y[7];

	s3 *= 0.86602540;
	t0 = (s5 + s1) * 0.98480775;
	t4 = (s5 - s7) * 0.34202014;
	t2 = (s1 + s7) * 0.64278761;
	s1 = (s1 - s5 - s7) * 0.86602540;

	s5 = t0 - s3 - t2;
	s7 = t4 - s3 - t0;
	s3 = t4 + s3 - t2;

	y[0] = s4 - s7;
	y[1] = s2 + s1;
	y[2] = s0 - s3;
	y[3] = s8 + s5;
	y[5] = s8 - s5;
	y[6] = s0 + s3;
	y[7] = s2 - s1;
	y[8] = s4 + s7;
}

fn imdct36(grbuf: &mut [f32], overlap: &mut [f32], window: &[f32; 18], nbands: usize) {
	for j in 0..nbands {
		let g = &mut grbuf[j * 18..j * 18 + 18];
		let ovl = &mut overlap[j * 9..j * 9 + 9];
		let mut co = [0.0f32; 9];
		let mut si = [0.0f32; 9];
		co[0] = -g[0];
		si[0] = g[17];
		for i in 0..4 {
			si[8 - 2 * i] = g[4 * i + 1] - g[4 * i + 2];
			co[1 + 2 * i] = g[4 * i + 1] + g[4 * i + 2];
			si[7 - 2 * i] = g[4 * i + 4] - g[4 * i + 3];
			co[2 + 2 * i] = -(g[4 * i + 3] + g[4 * i + 4]);
		}
		dct3_9(&mut co);
		dct3_9(&mut si);
		si[1] = -si[1];
		si[3] = -si[3];
		si[5] = -si[5];
		si[7] = -si[7];

		for i in 0..9 {
			let prev = ovl[i];
			let sum = co[i] * TWID9[9 + i] + si[i] * TWID9[i];
			ovl[i] = co[i] * TWID9[i] - si[i] * TWID9[9 + i];
			g[i] = prev * window[i] - sum * window[9 + i];
			g[17 - i] = prev * window[9 + i] + sum * window[i];
		}
	}
}

fn idct3(x0: f32, x1: f32, x2: f32) -> [f32; 3] {
	let m1 = x1 * 0.86602540;
	let a1 = x0 - x2 * 0.5;
	[a1 + m1, x0 + x2, a1 - m1]
}

/// 12-point IMDCT of one short window. `x` is read with stride 3.
fn imdct12(x: &[f32], dst: &mut [f32], overlap: &mut [f32]) {
	let co = idct3(-x[0], x[6] + x[3], x[12] + x[9]);
	let mut si = idct3(x[15], x[12] - x[9], x[6] - x[3]);
	si[1] = -si[1];
	for i in 0..3 {
		let prev = overlap[i];
		let sum = co[i] * TWID3[3 + i] + si[i] * TWID3[i];
		overlap[i] = co[i] * TWID3[i] - si[i] * TWID3[3 + i];
		dst[i] = prev * TWID3[2 - i] - sum * TWID3[5 - i];
		dst[5 - i] = prev * TWID3[5 - i] + sum * TWID3[2 - i];
	}
}

fn imdct_short(grbuf: &mut [f32], overlap: &mut [f32], nbands: usize) {
	for band in 0..nbands {
		let g = &mut grbuf[band * 18..band * 18 + 18];
		let ovl = &mut overlap[band * 9..band * 9 + 9];
		let mut tmp = [0.0f32; 18];
		tmp.copy_from_slice(g);
		g[..6].copy_from_slice(&ovl[..6]);
		imdct12(&tmp, &mut g[6..12], &mut ovl[6..9]);
		imdct12(&tmp[1..], &mut g[12..18], &mut ovl[6..9]);
		let (head, tail) = ovl.split_at_mut(6);
		imdct12(&tmp[2..], head, tail);
	}
}

/// Hybrid filterbank: IMDCT for long/short blocks with overlap-add.
pub fn imdct_granule(grbuf: &mut [f32], overlap: &mut [f32], block_type: u8, n_long_bands: usize) {
	let n_long_bands = n_long_bands.min(32);
	if n_long_bands > 0 {
		imdct36(grbuf, overlap, &MDCT_WINDOW[0], n_long_bands);
	}
	let g = &mut grbuf[18 * n_long_bands..];
	let o = &mut overlap[9 * n_long_bands..];
	if block_type == SHORT_BLOCK_TYPE {
		imdct_short(g, o, 32 - n_long_bands);
	} else {
		let window = &MDCT_WINDOW[usize::from(block_type == STOP_BLOCK_TYPE)];
		imdct36(g, o, window, 32 - n_long_bands);
	}
}

/// Frequency inversion: negate odd samples of odd subbands.
pub fn change_sign(grbuf: &mut [f32]) {
	for band in (1..32).step_by(2) {
		for i in (1..18).step_by(2) {
			grbuf[band * 18 + i] = -grbuf[band * 18 + i];
		}
	}
}
