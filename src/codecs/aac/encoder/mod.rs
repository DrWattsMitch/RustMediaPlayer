//! AAC-LC encoder (ISO/IEC 14496-3 §4, FFmpeg `libavcodec/aacenc.c`).
//!
//! Per 1024-sample frame and channel element:
//!
//! 1. **Block switching** ([`psy::has_attack`] on the *next* frame, one
//!    frame of lookahead): ONLY_LONG → LONG_START → EIGHT_SHORT … →
//!    LONG_STOP, so a transient is always covered by short windows.
//! 2. **Filterbank**: sine window and forward MDCT (2048 → 1024, or 8 ×
//!    256 → 128), the exact inverse of the decoder's synthesis.
//! 3. **Psychoacoustics** ([`psy`]): masking threshold per scalefactor band;
//!    short windows are grouped by energy.
//! 4. **Mid/side**: per band, M = (L+R)/2 and S = (L−R)/2 when that lowers
//!    the perceptual entropy (both channels of a CPE always share a window
//!    so `common_window` and M/S are available).
//! 5. **Rate loop**: a single noise offset (in dB, over every band of the
//!    frame) is searched so the frame fits its bit budget. Per band the
//!    coarsest scalefactor meeting `threshold · offset` is picked
//!    ([`quant::choose_sf`]), scalefactor differences are kept within ±60,
//!    and codebooks are chosen by a sectioning trellis. Every count is
//!    exact, so the frame never exceeds its budget.
//! 6. **Bit reservoir**: the budget follows perceptual entropy (more for
//!    transients, less for easy frames), limited by the bits saved earlier
//!    and by the 6144 bits per channel a decoder buffers.
//!
//! The bandwidth follows FFmpeg's `AAC_CUTOFF_FROM_BITRATE`. Output is one
//! raw access unit per packet with a 2-byte AudioSpecificConfig as
//! extradata. As in FFmpeg the first frame overlaps 1024 samples of
//! silence ([`ENCODER_DELAY`]); the last packet's duration counts only real
//! samples, so an MP4 edit list (or a decoder honouring `encoder_delay`)
//! gives back exactly the input length.
//!
//! Not used (all optional encoder tools): TNS, PNS, intensity stereo,
//! pulses, KBD windows.

mod psy;
mod quant;

use super::config::{AOT_AAC_LC, AacConfig, ElementKind, SAMPLE_RATES};
use super::ics::WindowSequence;
use crate::codecs::bitstream::BitWriter;
use crate::codecs::dsp::{Complex, Mdct};
use crate::codecs::pcm::convert::for_each_sample;
use crate::core::{
	self, AudioState, Channels, CodecPrivate, CodecState, EncodeOut, Frame, Packet, Parameters,
	SampleFormat, SampleLayout, State, StreamId, Time, TimeBase,
};
use crate::message::{Message, Result};
use psy::BandModel;
use quant::BandStats;
use std::sync::OnceLock;

const FRAME: usize = 1024;
const SHORT: usize = 128;
/// Samples of silence the first frame overlaps (FFmpeg's `initial_padding`).
pub const ENCODER_DELAY: u32 = 1024;
/// Samples buffered before a frame can be coded: its 2048-sample window
/// plus the next frame's new samples (block-switching lookahead).
const NEEDED: usize = 3 * FRAME;
/// Bits a decoder buffers per channel (§4.5.3.2).
const MAX_CHANNEL_BITS: usize = 6144;

/// FFmpeg's default bit rate: 64 kb/s per channel (128 kb/s stereo).
pub fn default_bitrate(channels: u8) -> u32 {
	64_000 * u32::from(channels.max(1))
}

/// FFmpeg's `AAC_CUTOFF_FROM_BITRATE`, limited to Nyquist.
fn cutoff_hz(bitrate: u32, channels: u8, sample_rate: u32) -> u32 {
	let per = bitrate / u32::from(channels.max(1));
	let a = (per / 5).max((per * 15 / 32).saturating_sub(5500));
	a.min(3000 + per / 4).min(12000 + per / 16).min(sample_rate / 2)
}

struct Filterbank {
	long: Mdct,
	short: Mdct,
	/// Rising halves of the sine windows.
	sine_long: Vec<f32>,
	sine_short: Vec<f32>,
}

fn filterbank() -> &'static Filterbank {
	static FB: OnceLock<Filterbank> = OnceLock::new();
	FB.get_or_init(|| {
		let sine = |n: usize| {
			(0..n / 2)
				.map(|i| (std::f64::consts::PI / n as f64 * (i as f64 + 0.5)).sin() as f32)
				.collect()
		};
		// The specification's forward MDCT has a factor 2 (§4.6.11 inverts
		// it with 2/N).
		Filterbank {
			long: Mdct::new(2 * FRAME, 2.0),
			short: Mdct::new(2 * SHORT, 2.0),
			sine_long: sine(2 * FRAME),
			sine_short: sine(2 * SHORT),
		}
	})
}

/// One channel element of the output layout.
#[derive(Debug, Clone)]
struct Element {
	kind: ElementKind,
	tag: u8,
	/// Input channels (FFmpeg order) of the element's channel(s).
	channels: [usize; 2],
	count: usize,
	prev_seq: WindowSequence,
}

/// Encoder state: buffered input, block-switching history and the bit
/// reservoir.
#[derive(Debug, Clone)]
pub struct AacEncoderState {
	pub sample_rate: u32,
	pub channels: u8,
	pub bitrate: u32,
	rate_index: usize,
	channel_config: u8,
	elements: Vec<Element>,
	/// Highest coded MDCT line + 1 for long windows (LFE: its own limit).
	cutoff_long: usize,
	cutoff_lfe: usize,
	/// Per channel, input samples from the current frame's window start.
	input: Vec<Vec<f32>>,
	samples_in: u64,
	frames_out: u64,
	/// Bits saved in the reservoir.
	reservoir: f32,
	pe_average: f32,
	/// Noise offset (dB) of the last frame: the next search starts there.
	offset_db: f32,
	long: BandModel,
	short: BandModel,
}

/// Analysis and quantisation results for one channel of the current frame.
#[derive(Debug, Clone)]
struct ChannelFrame {
	seq: WindowSequence,
	grouping: u8,
	groups: Vec<usize>,
	/// Bands per group considered (bandwidth limit).
	max_sfb: usize,
	/// Bands per group written (trailing silent bands trimmed).
	coded_sfb: usize,
	/// Coefficients in bitstream order (group, band, window).
	coef: Vec<f32>,
	x34: Vec<f32>,
	bands: Vec<BandStats>,
	/// M/S per band (stored on the first channel of a CPE).
	ms: Vec<bool>,
	sf: Vec<i32>,
	coded: Vec<bool>,
	q: Vec<i32>,
	cbs: Vec<u8>,
}

impl ChannelFrame {
	fn band(&self, idx: usize) -> std::ops::Range<usize> {
		self.bands[idx].start..self.bands[idx].end
	}

	fn short(&self) -> bool {
		self.seq == WindowSequence::EightShort
	}

	fn len_bits(&self) -> u32 {
		if self.short() { 3 } else { 5 }
	}
}

impl AacEncoderState {
	pub fn new(sample_rate: u32, channels: u8, bitrate: u32) -> Option<Self> {
		let rate_index = SAMPLE_RATES.iter().position(|&r| r == sample_rate)?;
		let channel_config = match channels {
			1..=6 => channels,
			8 => 7,
			_ => return None,
		};
		let config = AacConfig::from_channel_config(AOT_AAC_LC, rate_index, channel_config)?;
		let elements = config
			.layout
			.iter()
			.map(|s| Element {
				kind: s.kind,
				tag: s.tag,
				channels: [usize::from(s.out[0]), usize::from(s.out[1])],
				count: if s.kind == ElementKind::Cpe { 2 } else { 1 },
				prev_seq: WindowSequence::OnlyLong,
			})
			.collect();
		// At most 6144 bits per channel and frame.
		let max_rate = (MAX_CHANNEL_BITS as u64 * u64::from(sample_rate) / FRAME as u64) as u32;
		let bitrate = bitrate.clamp(8000 * u32::from(channels), max_rate * u32::from(channels));
		let line = |hz: u32| ((u64::from(hz) * 2 * FRAME as u64) / u64::from(sample_rate)) as usize;
		Some(Self {
			sample_rate,
			channels,
			bitrate,
			rate_index,
			channel_config,
			elements,
			cutoff_long: line(cutoff_hz(bitrate, channels, sample_rate)).clamp(1, FRAME),
			cutoff_lfe: line(250).clamp(1, FRAME),
			// The first frame's window starts 1024 samples before the input.
			input: vec![vec![0.0; FRAME]; usize::from(channels)],
			samples_in: 0,
			frames_out: 0,
			reservoir: 0.0,
			pe_average: 0.0,
			offset_db: 0.0,
			long: BandModel::new(rate_index, sample_rate, false),
			short: BandModel::new(rate_index, sample_rate, true),
		})
	}

	/// AudioSpecificConfig: AAC-LC, the rate index and channel
	/// configuration, 1024-sample frames.
	pub fn audio_specific_config(&self) -> Vec<u8> {
		let mut w = BitWriter::new();
		w.put(u32::from(AOT_AAC_LC), 5);
		w.put(self.rate_index as u32, 4);
		w.put(u32::from(self.channel_config), 4);
		w.put(0, 3); // frameLengthFlag, dependsOnCoreCoder, extensionFlag
		w.finish()
	}

	fn average_bits(&self) -> f32 {
		self.bitrate as f32 * FRAME as f32 / self.sample_rate as f32
	}

	/// Total frames for the samples seen so far: frame j outputs samples
	/// [(j-1)·1024, j·1024), so the last real sample needs
	/// `ceil(n/1024) + 1` frames.
	fn frames_needed(&self) -> u64 {
		if self.samples_in == 0 { 0 } else { self.samples_in.div_ceil(FRAME as u64) + 1 }
	}

	/// Codes one frame from the head of `input`, then drops 1024 samples.
	fn encode_frame(&mut self) -> Vec<u8> {
		let fb = filterbank();
		let mut chans: Vec<Option<ChannelFrame>> = vec![None; usize::from(self.channels)];

		for e in 0..self.elements.len() {
			let element = self.elements[e].clone();
			// Block switching from the next frame's short-window region.
			let attack_next = element.kind != ElementKind::Lfe
				&& element.channels[..element.count]
					.iter()
					.any(|&c| psy::has_attack(&self.input[c][1472 - 384..2624], 384));
			let seq = if element.kind == ElementKind::Lfe {
				WindowSequence::OnlyLong
			} else {
				next_sequence(element.prev_seq, attack_next)
			};
			self.elements[e].prev_seq = seq;

			let mut spectra: Vec<[f32; FRAME]> = Vec::with_capacity(2);
			for &c in &element.channels[..element.count] {
				spectra.push(transform(fb, &self.input[c][..2 * FRAME], seq));
			}
			// One grouping for the element (CPEs share ics_info).
			let grouping = if seq == WindowSequence::EightShort {
				let mut energy = [0.0f32; 8];
				for s in &spectra {
					for (w, e) in energy.iter_mut().enumerate() {
						*e += s[w * SHORT..(w + 1) * SHORT].iter().map(|v| v * v).sum::<f32>();
					}
				}
				psy::grouping(&energy)
			} else {
				0
			};
			let cutoff =
				if element.kind == ElementKind::Lfe { self.cutoff_lfe } else { self.cutoff_long };
			let mut frames: Vec<ChannelFrame> =
				spectra.iter().map(|s| self.analyse(s, seq, grouping, cutoff)).collect();
			if let [l, r] = frames.as_mut_slice() {
				mid_side(l, r);
			}
			for (i, f) in frames.into_iter().enumerate() {
				chans[element.channels[i]] = Some(f);
			}
		}
		let mut chans: Vec<ChannelFrame> = chans.into_iter().flatten().collect();
		for ch in &mut chans {
			abs_and_stats(ch);
		}

		// Bit budget from perceptual entropy and the reservoir.
		let avg = self.average_bits();
		let capacity = (MAX_CHANNEL_BITS * usize::from(self.channels)) as f32 - avg;
		let pe: f32 = chans
			.iter()
			.flat_map(|c| c.bands.iter())
			.map(|b| psy::perceptual_entropy(b.energy, b.threshold, b.end - b.start))
			.sum();
		self.pe_average =
			if self.frames_out == 0 { pe.max(1.0) } else { 0.9 * self.pe_average + 0.1 * pe.max(1.0) };
		let short = chans.iter().any(ChannelFrame::short);
		let mut ratio = (pe.max(1.0) / self.pe_average).clamp(0.6, 2.0);
		if short {
			ratio = ratio.max(1.5);
		}
		let fixed = self.fixed_bits() as f32;
		let upper = (avg + self.reservoir).min(capacity + avg) - fixed;
		let lower = (avg + self.reservoir - capacity).max(0.0) - fixed;
		let target = (avg * ratio - fixed).clamp(lower.max(0.0), upper.max(0.0));

		let offset = self.search_offset(&mut chans, target as usize);
		self.offset_db = offset;

		let data = self.write_frame(&chans);
		self.reservoir = (self.reservoir + avg - (data.len() * 8) as f32).clamp(0.0, capacity);
		for buf in &mut self.input {
			buf.drain(..FRAME);
		}
		self.frames_out += 1;
		data
	}

	/// Band energies and thresholds of one channel, regrouped into
	/// bitstream order.
	fn analyse(
		&self,
		spec: &[f32; FRAME],
		seq: WindowSequence,
		grouping: u8,
		cutoff: usize,
	) -> ChannelFrame {
		let short = seq == WindowSequence::EightShort;
		let model = if short { &self.short } else { &self.long };
		let offsets = model.offsets;
		let (groups, windows, win_len, line_cut) = if short {
			(psy::group_lengths(grouping), 8, SHORT, cutoff.div_ceil(8))
		} else {
			(vec![1], 1, FRAME, cutoff)
		};
		let max_sfb = (0..model.bands()).filter(|&b| usize::from(offsets[b]) < line_cut).count().max(1);
		let mut coef = Vec::with_capacity(FRAME);
		let mut bands = Vec::with_capacity(groups.len() * max_sfb);
		let mut energy = vec![0.0f32; model.bands()];
		let mut thr = vec![0.0f32; model.bands()];
		let mut window = 0;
		for &glen in &groups {
			let mut genergy = vec![0.0f32; max_sfb];
			let mut gthr = vec![0.0f32; max_sfb];
			for w in window..window + glen {
				let spec_w = &spec[w * win_len..(w + 1) * win_len];
				for b in 0..model.bands() {
					let (lo, hi) = (usize::from(offsets[b]), usize::from(offsets[b + 1]));
					energy[b] = spec_w[lo..hi].iter().map(|v| v * v).sum();
				}
				model.thresholds(&energy, &mut thr);
				for b in 0..max_sfb {
					genergy[b] += energy[b];
					gthr[b] += thr[b];
				}
			}
			for b in 0..max_sfb {
				let (lo, hi) = (usize::from(offsets[b]), usize::from(offsets[b + 1]));
				let start = coef.len();
				for w in window..window + glen {
					coef.extend_from_slice(&spec[w * win_len + lo..w * win_len + hi]);
				}
				bands.push(BandStats {
					start,
					end: coef.len(),
					energy: genergy[b],
					threshold: gthr[b],
					..Default::default()
				});
			}
			window += glen;
		}
		debug_assert_eq!(window, windows);
		let n = bands.len();
		let len = coef.len();
		ChannelFrame {
			seq,
			grouping,
			groups,
			max_sfb,
			coded_sfb: max_sfb,
			coef,
			x34: vec![0.0; len],
			bands,
			ms: vec![false; n],
			sf: vec![0; n],
			coded: vec![false; n],
			q: vec![0; len],
			cbs: vec![0; n],
		}
	}

	/// Element headers, `ics_info`, M/S masks and the END element: the
	/// bits that don't depend on quantisation (the per-channel parts are
	/// counted by [`ics_bits`]).
	fn fixed_bits(&self) -> usize {
		let mut bits = 3; // END
		for e in &self.elements {
			bits += 3 + 4;
			if e.kind == ElementKind::Cpe {
				bits += 1 + 2; // common_window, ms_mask_present
			}
		}
		bits + 7 // worst-case byte alignment
	}

	/// Finds the smallest noise offset whose frame fits `target` bits, and
	/// leaves `chans` quantised with it.
	fn search_offset(&self, chans: &mut [ChannelFrame], target: usize) -> f32 {
		const LO: f32 = -24.0;
		const HI: f32 = 80.0;
		let eval = |chans: &mut [ChannelFrame], g: f32| self.quantise_all(chans, g);
		let start = self.offset_db.clamp(LO, HI);
		let mut fits = eval(chans, start) <= target;
		let (mut good, mut bad) = if fits { (start, f32::NAN) } else { (f32::NAN, start) };
		// Bracket: step towards the boundary, doubling.
		let mut step = 2.0f32;
		while good.is_nan() || bad.is_nan() {
			let probe = if fits { good - step } else { bad + step };
			if probe < LO {
				if fits {
					good = LO;
					eval(chans, good);
					return good;
				}
				break;
			}
			if probe > HI {
				// Even the coarsest offset doesn't fit: drop everything above
				// the coarsest result's bands.
				eval(chans, 1000.0);
				return HI;
			}
			fits = eval(chans, probe) <= target;
			if fits {
				good = probe;
			} else {
				bad = probe;
			}
			step *= 2.0;
		}
		// A larger offset means coarser quantisation: `good > bad`.
		while good - bad > 0.25 {
			let mid = 0.5 * (good + bad);
			if eval(chans, mid) <= target {
				good = mid;
			} else {
				bad = mid;
			}
		}
		eval(chans, good);
		good
	}

	/// Quantises every channel at noise offset `g` dB and returns the
	/// variable bits of the frame.
	fn quantise_all(&self, chans: &mut [ChannelFrame], g: f32) -> usize {
		let allowance = Allowance::new(chans, g);
		for ch in chans.iter_mut() {
			for b in 0..ch.bands.len() {
				let stats = ch.bands[b];
				let allowed = allowance.of(stats.threshold);
				match quant::choose_sf(&ch.coef, &ch.x34, &stats, allowed, &mut ch.q) {
					Some(sf) => {
						ch.sf[b] = sf;
						ch.coded[b] = true;
					}
					None => {
						ch.coded[b] = false;
						let range = ch.band(b);
						ch.q[range].fill(0);
					}
				}
			}
			constrain_scalefactors(ch);
		}
		// Elements: CPEs share max_sfb.
		let mut bits = 0;
		for e in &self.elements {
			let idx = &e.channels[..e.count];
			let coded_sfb = idx.iter().map(|&c| last_coded_band(&chans[c])).max().unwrap_or(0);
			for &c in idx {
				chans[c].coded_sfb = coded_sfb;
				choose_codebooks(&mut chans[c]);
				bits += ics_bits(&chans[c], e.kind == ElementKind::Cpe);
			}
			if e.kind == ElementKind::Cpe {
				bits += ics_info_bits(&chans[idx[0]]) + ms_bits(&chans[idx[0]]);
			}
		}
		bits
	}

	fn write_frame(&self, chans: &[ChannelFrame]) -> Vec<u8> {
		let mut w = BitWriter::with_capacity(MAX_CHANNEL_BITS / 8 * chans.len());
		for e in &self.elements {
			match e.kind {
				ElementKind::Cpe => {
					let (l, r) = (&chans[e.channels[0]], &chans[e.channels[1]]);
					w.put(1, 3);
					w.put(u32::from(e.tag), 4);
					w.bit(true); // common_window
					write_ics_info(&mut w, l);
					write_ms(&mut w, l);
					write_ics(&mut w, l, true);
					write_ics(&mut w, r, true);
				}
				kind => {
					w.put(if kind == ElementKind::Lfe { 3 } else { 0 }, 3);
					w.put(u32::from(e.tag), 4);
					write_ics(&mut w, &chans[e.channels[0]], false);
				}
			}
		}
		w.put(7, 3); // END
		w.finish()
	}
}

/// Allowed noise per band at noise offset `g` (3GPP TS 26.403 §5.6.1.1,
/// FFmpeg `aacpsy.c` threshold reduction).
///
/// Below 0 dB (more bits than the masking thresholds need) every threshold
/// is lowered by the same factor. Above it, thresholds are raised in the
/// fourth-root domain, `(T^¼ + r)^4`: quiet bands gain noise faster than
/// loud ones, so the noise tends towards white as the bit rate drops,
/// instead of all of it piling into the loud bands. `r` is scaled by the
/// frame's mean `T^¼`, which makes both branches agree for a flat spectrum
/// and keeps the offset meaningful in dB.
struct Allowance {
	g: f32,
	r: f32,
}

impl Allowance {
	fn new(chans: &[ChannelFrame], g: f32) -> Self {
		if g <= 0.0 {
			return Self { g, r: 0.0 };
		}
		let (mut sum, mut count) = (0.0f32, 0usize);
		for b in chans.iter().flat_map(|c| c.bands.iter()) {
			if b.energy > b.threshold {
				sum += b.threshold.sqrt().sqrt();
				count += 1;
			}
		}
		let mean = if count == 0 { 1.0 } else { sum / count as f32 };
		Self { g, r: mean * (10f32.powf(g / 40.0) - 1.0) }
	}

	#[inline]
	fn of(&self, threshold: f32) -> f32 {
		if self.g <= 0.0 {
			threshold * 10f32.powf(self.g / 10.0)
		} else {
			(threshold.sqrt().sqrt() + self.r).powi(4)
		}
	}
}

/// Window sequence for a frame given the previous one and whether the next
/// frame has an attack (Table 4.4 transitions).
fn next_sequence(prev: WindowSequence, attack_next: bool) -> WindowSequence {
	use WindowSequence::*;
	match prev {
		// LONG_START was chosen because this frame has the attack.
		LongStart => EightShort,
		// Stay short while attacks continue, then stop.
		EightShort => {
			if attack_next {
				EightShort
			} else {
				LongStop
			}
		}
		OnlyLong | LongStop => {
			if attack_next {
				LongStart
			} else {
				OnlyLong
			}
		}
	}
}

/// Windows `x` (2048 samples) for `seq` and runs the MDCT. Short windows
/// are stored window after window (`w * 128 + k`).
fn transform(fb: &Filterbank, x: &[f32], seq: WindowSequence) -> [f32; FRAME] {
	let (sl, ss) = (&fb.sine_long, &fb.sine_short);
	let mut out = [0.0f32; FRAME];
	let mut scratch = [0.0f32; FRAME];
	let mut buf = [Complex::default(); FRAME / 2];
	if seq == WindowSequence::EightShort {
		let mut z = [0.0f32; 2 * SHORT];
		for w in 0..8 {
			let base = 448 + w * SHORT;
			for n in 0..SHORT {
				z[n] = x[base + n] * ss[n];
				z[2 * SHORT - 1 - n] = x[base + 2 * SHORT - 1 - n] * ss[n];
			}
			fb.short.forward(&z, &mut out[w * SHORT..(w + 1) * SHORT], &mut scratch, &mut buf);
		}
		return out;
	}
	let mut z = vec![0.0f32; 2 * FRAME];
	match seq {
		WindowSequence::LongStop => {
			for n in 448..576 {
				z[n] = x[n] * ss[n - 448];
			}
			z[576..FRAME].copy_from_slice(&x[576..FRAME]);
		}
		_ => {
			for n in 0..FRAME {
				z[n] = x[n] * sl[n];
			}
		}
	}
	match seq {
		WindowSequence::LongStart => {
			z[FRAME..1472].copy_from_slice(&x[FRAME..1472]);
			for n in 1472..1600 {
				z[n] = x[n] * ss[1599 - n];
			}
		}
		_ => {
			for n in 0..FRAME {
				z[FRAME + n] = x[FRAME + n] * sl[FRAME - 1 - n];
			}
		}
	}
	fb.long.forward(&z, &mut out, &mut scratch, &mut buf);
	out
}

/// Per-band M/S decision for a CPE (both channels share the layout): M/S
/// is used where it lowers the perceptual entropy, with both thresholds set
/// to half the lower L/R threshold (noise in M and S adds up in L and R).
fn mid_side(l: &mut ChannelFrame, r: &mut ChannelFrame) {
	for b in 0..l.bands.len() {
		let range = l.band(b);
		let (mut em, mut es) = (0.0f32, 0.0f32);
		for i in range.clone() {
			let (m, s) = (0.5 * (l.coef[i] + r.coef[i]), 0.5 * (l.coef[i] - r.coef[i]));
			em += m * m;
			es += s * s;
		}
		let (bl, br) = (l.bands[b], r.bands[b]);
		let width = range.len();
		let t = 0.5 * bl.threshold.min(br.threshold);
		let pe_lr = psy::perceptual_entropy(bl.energy, bl.threshold, width)
			+ psy::perceptual_entropy(br.energy, br.threshold, width);
		let pe_ms = psy::perceptual_entropy(em, t, width) + psy::perceptual_entropy(es, t, width);
		if pe_ms < pe_lr {
			l.ms[b] = true;
			for i in range {
				let (a, c) = (l.coef[i], r.coef[i]);
				l.coef[i] = 0.5 * (a + c);
				r.coef[i] = 0.5 * (a - c);
			}
			l.bands[b].energy = em;
			r.bands[b].energy = es;
			l.bands[b].threshold = t;
			r.bands[b].threshold = t;
		}
	}
}

fn abs_and_stats(ch: &mut ChannelFrame) {
	quant::abs34(&ch.coef, &mut ch.x34);
	for b in 0..ch.bands.len() {
		let s = ch.bands[b];
		ch.bands[b] = BandStats::new(&ch.coef, &ch.x34, s.start, s.end, s.threshold);
	}
}

/// Keeps consecutive coded scalefactors within ±60 (the scalefactor
/// codebook's range). Each band may only get coarser than its choice, or
/// finer down to the smallest step that still avoids overflow; with those
/// lower bounds made 60-Lipschitz first, a forward clamp always succeeds.
fn constrain_scalefactors(ch: &mut ChannelFrame) {
	let n = ch.bands.len();
	let coded: Vec<usize> = (0..n).filter(|&b| ch.coded[b]).collect();
	if coded.is_empty() {
		return;
	}
	let mut floor: Vec<i32> = coded.iter().map(|&b| ch.bands[b].min_sf()).collect();
	for i in 1..floor.len() {
		floor[i] = floor[i].max(floor[i - 1] - quant::MAX_SF_DIFF);
	}
	for i in (0..floor.len().saturating_sub(1)).rev() {
		floor[i] = floor[i].max(floor[i + 1] - quant::MAX_SF_DIFF);
	}
	let mut prev: Option<i32> = None;
	for (i, &b) in coded.iter().enumerate() {
		let want = ch.sf[b];
		let mut sf = match prev {
			Some(p) => want.clamp(p - quant::MAX_SF_DIFF, p + quant::MAX_SF_DIFF),
			None => want,
		};
		sf = sf.max(floor[i]).min(255);
		if sf != want {
			let range = ch.band(b);
			quant::quantize(&ch.coef[range.clone()], &ch.x34[range.clone()], sf, &mut ch.q[range]);
		}
		ch.sf[b] = sf;
		prev = Some(sf);
	}
}

/// Bands per group up to the last coded one.
fn last_coded_band(ch: &ChannelFrame) -> usize {
	let mut last = 0;
	for g in 0..ch.groups.len() {
		for sfb in 0..ch.max_sfb {
			if ch.coded[g * ch.max_sfb + sfb] {
				last = last.max(sfb + 1);
			}
		}
	}
	last
}

fn choose_codebooks(ch: &mut ChannelFrame) {
	let len_bits = ch.len_bits();
	for g in 0..ch.groups.len() {
		let base = g * ch.max_sfb;
		let costs: Vec<[u32; 12]> = (0..ch.coded_sfb)
			.map(|sfb| quant::codebook_costs(&ch.q[ch.band(base + sfb)], ch.coded[base + sfb]))
			.collect();
		let cbs = quant::section(&costs, len_bits);
		ch.cbs[base..base + ch.coded_sfb].copy_from_slice(&cbs);
		for sfb in ch.coded_sfb..ch.max_sfb {
			ch.cbs[base + sfb] = 0;
		}
	}
}

fn global_gain(ch: &ChannelFrame) -> i32 {
	(0..ch.groups.len())
		.flat_map(|g| (0..ch.coded_sfb).map(move |sfb| g * ch.max_sfb + sfb))
		.find(|&i| ch.cbs[i] != 0)
		.map_or(quant::SF_OFFSET, |i| ch.sf[i])
}

fn ics_info_bits(ch: &ChannelFrame) -> usize {
	1 + 2 + 1 + if ch.short() { 4 + 7 } else { 6 + 1 }
}

fn ms_bits(l: &ChannelFrame) -> usize {
	match ms_mode(l) {
		1 => l.groups.len() * l.coded_sfb,
		_ => 0,
	}
}

/// `ms_mask_present`: 0 none, 1 per band, 2 all bands.
fn ms_mode(l: &ChannelFrame) -> u32 {
	let used = |g: usize, sfb: usize| l.ms[g * l.max_sfb + sfb];
	let bands = || (0..l.groups.len()).flat_map(|g| (0..l.coded_sfb).map(move |s| (g, s)));
	if l.coded_sfb == 0 || bands().all(|(g, s)| !used(g, s)) {
		0
	} else if bands().all(|(g, s)| used(g, s)) {
		2
	} else {
		1
	}
}

/// Bits of `individual_channel_stream` (without `ics_info` when the window
/// is common).
fn ics_bits(ch: &ChannelFrame, common_window: bool) -> usize {
	let mut bits = 8 + 3; // global_gain; pulse, tns, gain-control flags
	if !common_window {
		bits += ics_info_bits(ch);
	}
	let mut last = global_gain(ch);
	for g in 0..ch.groups.len() {
		let base = g * ch.max_sfb;
		bits += quant::section_bits(&ch.cbs[base..base + ch.coded_sfb], ch.len_bits()) as usize;
		for sfb in 0..ch.coded_sfb {
			let i = base + sfb;
			let cb = ch.cbs[i];
			if cb != 0 {
				bits += quant::sf_diff_bits(ch.sf[i] - last) as usize;
				last = ch.sf[i];
				bits += quant::band_bits(cb, &ch.q[ch.band(i)]) as usize;
			}
		}
	}
	bits
}

fn write_ics_info(w: &mut BitWriter, ch: &ChannelFrame) {
	w.bit(false); // ics_reserved_bit
	let seq = match ch.seq {
		WindowSequence::OnlyLong => 0,
		WindowSequence::LongStart => 1,
		WindowSequence::EightShort => 2,
		WindowSequence::LongStop => 3,
	};
	w.put(seq, 2);
	w.bit(false); // sine window
	if ch.short() {
		w.put(ch.coded_sfb as u32, 4);
		w.put(u32::from(ch.grouping), 7);
	} else {
		w.put(ch.coded_sfb as u32, 6);
		w.bit(false); // predictor_data_present
	}
}

fn write_ms(w: &mut BitWriter, l: &ChannelFrame) {
	let mode = ms_mode(l);
	w.put(mode, 2);
	if mode == 1 {
		for g in 0..l.groups.len() {
			for sfb in 0..l.coded_sfb {
				w.bit(l.ms[g * l.max_sfb + sfb]);
			}
		}
	}
}

fn write_ics(w: &mut BitWriter, ch: &ChannelFrame, common_window: bool) {
	let gain = global_gain(ch);
	w.put(gain as u32, 8);
	if !common_window {
		write_ics_info(w, ch);
	}
	for g in 0..ch.groups.len() {
		let base = g * ch.max_sfb;
		quant::write_sections(w, &ch.cbs[base..base + ch.coded_sfb], ch.len_bits());
	}
	let mut last = gain;
	for g in 0..ch.groups.len() {
		for sfb in 0..ch.coded_sfb {
			let i = g * ch.max_sfb + sfb;
			if ch.cbs[i] != 0 {
				quant::write_sf_diff(w, ch.sf[i] - last);
				last = ch.sf[i];
			}
		}
	}
	w.put(0, 3); // pulse_data_present, tns_data_present, gain_control_data_present
	for g in 0..ch.groups.len() {
		for sfb in 0..ch.coded_sfb {
			let i = g * ch.max_sfb + sfb;
			if ch.cbs[i] != 0 {
				quant::write_band(w, ch.cbs[i], &ch.q[ch.band(i)]);
			}
		}
	}
}

/// AAC-LC encoder producing raw access units.
#[derive(Debug, Clone, Copy)]
pub struct AacEncoder {
	stream_id: StreamId,
	bitrate: Option<u32>,
}

impl AacEncoder {
	pub fn new(stream_id: StreamId) -> Self {
		Self { stream_id, bitrate: None }
	}

	/// Total bit rate in bits per second (`-b:a`); default
	/// [`default_bitrate`].
	pub fn with_bitrate(mut self, bitrate: Option<u32>) -> Self {
		self.bitrate = bitrate.filter(|&b| b > 0);
		self
	}

	fn new_state(&self, parameters: &Parameters) -> Result<AacEncoderState> {
		let rate = parameters.sample_rate.unwrap_or(0);
		let channels = parameters.channels.unwrap_or(0);
		if !SAMPLE_RATES.contains(&rate) {
			return Err(Message::Codec(
				"aac encoder: unsupported sample rate (use 8000-96000 Hz standard rates, e.g. -ar 48000)",
			));
		}
		let bitrate = self.bitrate.unwrap_or_else(|| default_bitrate(channels));
		AacEncoderState::new(rate, channels, bitrate)
			.ok_or(Message::Codec("aac encoder: 1-6 or 8 channels are supported"))
	}

	/// Bit rate the encoder will use for `parameters` (after clamping to
	/// what AAC can carry).
	pub fn bitrate_for(&self, parameters: &Parameters) -> u32 {
		self.new_state(parameters).map_or(0, |s| s.bitrate)
	}

	/// Codec extradata (AudioSpecificConfig) for `parameters`.
	pub fn extradata(&self, parameters: &Parameters) -> Result<Vec<u8>> {
		Ok(self.new_state(parameters)?.audio_specific_config())
	}

	fn state<'a>(&self, state: &'a mut State) -> Result<&'a mut AacEncoderState> {
		let State::Audio(audio) = state else {
			return Err(Message::Codec("aac encoder requires an audio state"));
		};
		match &mut audio.private {
			CodecPrivate::AacEncoder(enc) => Ok(enc),
			_ => Err(Message::Codec(
				"aac encoder: unsupported parameters (standard sample rate, 1-6 or 8 channels)",
			)),
		}
	}

	fn packet(&self, data: Vec<u8>, frame: u64, enc: &AacEncoderState) -> Packet {
		let tb = TimeBase::new(1, enc.sample_rate);
		let total = enc.samples_in + u64::from(ENCODER_DELAY);
		let start = frame * FRAME as u64;
		let duration = total.saturating_sub(start).min(FRAME as u64);
		let mut packet = Packet::new(self.stream_id, data);
		packet.pts = Some(Time::new(start, tb));
		packet.dts = packet.pts;
		packet.duration = Some(Time::new(duration, tb));
		packet.keyframe = true;
		packet
	}
}

impl core::Encoder for AacEncoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let enc = self.new_state(parameters).ok();
		State::Audio(AudioState {
			sample_rate: parameters.sample_rate.unwrap_or(48_000),
			channels: Channels::from_count(parameters.channels.unwrap_or(2)),
			format: SampleFormat::F32,
			layout: SampleLayout::Planar,
			frame_size: FRAME,
			codec_state: CodecState {
				extradata: enc.as_ref().map(AacEncoderState::audio_specific_config).unwrap_or_default(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: enc.map_or(CodecPrivate::None, |e| CodecPrivate::AacEncoder(Box::new(e))),
		})
	}

	fn encode(&self, state: &mut State, frame: Frame) -> Result<EncodeOut<State>> {
		let Frame::Audio(frame) = frame else {
			return Err(Message::Codec("aac encoder requires audio frames"));
		};
		let enc = self.state(state)?;
		let channels = usize::from(enc.channels);
		if usize::from(frame.channels.count()) != channels {
			return Err(Message::Codec("aac encoder: frame channel count does not match the stream"));
		}
		if frame.sample_rate != enc.sample_rate {
			return Err(Message::Codec(
				"aac encoder: frame sample rate does not match the stream (resample first)",
			));
		}
		let mut ch = 0usize;
		let mut count = 0u64;
		for_each_sample(&frame, |s| {
			enc.input[ch].push((s.to_float() * 32768.0) as f32);
			ch += 1;
			if ch == channels {
				ch = 0;
				count += 1;
			}
		})?;
		enc.samples_in += count;
		let mut packets = Vec::new();
		while enc.input[0].len() >= NEEDED {
			let index = enc.frames_out;
			let data = enc.encode_frame();
			packets.push(self.packet(data, index, enc));
		}
		if let State::Audio(audio) = state {
			audio.samples_consumed += count;
		}
		Ok(EncodeOut { value: packets, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<EncodeOut<State>> {
		let enc = self.state(state)?;
		let mut packets = Vec::new();
		while enc.frames_out < enc.frames_needed() {
			for buf in &mut enc.input {
				buf.resize(buf.len().max(NEEDED), 0.0);
			}
			let index = enc.frames_out;
			let data = enc.encode_frame();
			packets.push(self.packet(data, index, enc));
		}
		Ok(EncodeOut { value: packets, state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::codecs::aac::AacDecoderState;
	use crate::codecs::bitstream::BitReader;

	fn encode_all(
		rate: u32,
		channels: u8,
		bitrate: u32,
		input: &[Vec<f32>],
	) -> (AacEncoderState, Vec<Vec<u8>>) {
		let mut enc = AacEncoderState::new(rate, channels, bitrate).expect("config");
		let n = input[0].len();
		for c in 0..usize::from(channels) {
			enc.input[c].extend_from_slice(&input[c]);
		}
		enc.samples_in = n as u64;
		let mut frames = Vec::new();
		while enc.frames_out < enc.frames_needed() {
			for buf in &mut enc.input {
				buf.resize(buf.len().max(NEEDED), 0.0);
			}
			frames.push(enc.encode_frame());
		}
		(enc, frames)
	}

	fn decode_all(asc: &[u8], frames: &[Vec<u8>], channels: usize) -> Vec<Vec<f32>> {
		let mut dec = AacDecoderState::from_extradata(asc).expect("asc");
		let mut out = vec![Vec::new(); channels];
		for f in frames {
			let mut bs = BitReader::new(f);
			let frame = dec.decode_block(&mut bs).expect("decodes");
			assert_eq!(dec.concealed_frames, 0);
			for (i, &s) in frame.pcm.iter().enumerate() {
				out[i % channels].push(f32::from(s));
			}
		}
		out
	}

	fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
		let (mut sig, mut err) = (0.0f64, 0.0f64);
		for (a, b) in reference.iter().zip(decoded) {
			sig += f64::from(*a) * f64::from(*a);
			err += f64::from(a - b) * f64::from(a - b);
		}
		10.0 * (sig / err.max(1e-9)).log10()
	}

	#[test]
	fn round_trips_through_the_decoder() {
		let n: usize = 48_000;
		let left: Vec<f32> = (0..n).map(|i| 9000.0 * (i as f32 * 0.031).sin()).collect();
		let right: Vec<f32> = (0..n)
			.map(|i| 7000.0 * (i as f32 * 0.047).sin() + 2000.0 * (i as f32 * 0.21).sin())
			.collect();
		let (enc, frames) = encode_all(48_000, 2, 192_000, &[left.clone(), right.clone()]);
		assert_eq!(frames.len(), n.div_ceil(FRAME) + 1);
		let out = decode_all(&enc.audio_specific_config(), &frames, 2);
		let d = ENCODER_DELAY as usize;
		assert!(snr_db(&left, &out[0][d..d + n]) > 30.0, "{}", snr_db(&left, &out[0][d..d + n]));
		assert!(snr_db(&right, &out[1][d..d + n]) > 30.0);
		let bits: usize = frames.iter().map(|f| f.len() * 8).sum();
		let rate = bits as f64 * 48_000.0 / (frames.len() * FRAME) as f64;
		assert!(rate < 192_000.0 * 1.05, "{rate}");
	}

	#[test]
	fn transients_use_short_windows_and_decode() {
		let n = 20_000;
		let mut x = vec![0.0f32; n];
		for start in [5000usize, 12_000] {
			for i in start..start + 600 {
				let t = i as f32;
				let tone = (t * 0.05).sin() + 0.6 * (t * 0.17).sin() + 0.4 * (t * 0.31).sin();
				x[i] = 12000.0 * tone * (-((i - start) as f32) / 150.0).exp();
			}
		}
		let mut enc = AacEncoderState::new(44_100, 1, 96_000).expect("config");
		enc.input[0].extend_from_slice(&x);
		enc.samples_in = n as u64;
		let mut seqs = Vec::new();
		let mut frames = Vec::new();
		while enc.frames_out < enc.frames_needed() {
			let len = enc.input[0].len().max(NEEDED);
			enc.input[0].resize(len, 0.0);
			frames.push(enc.encode_frame());
			seqs.push(enc.elements[0].prev_seq);
		}
		assert!(seqs.contains(&WindowSequence::EightShort), "{seqs:?}");
		for pair in seqs.windows(2) {
			if pair[0] == WindowSequence::LongStart {
				assert_eq!(pair[1], WindowSequence::EightShort);
			}
		}
		let out = decode_all(&enc.audio_specific_config(), &frames, 1);
		let snr = snr_db(&x, &out[0][1024..1024 + n]);
		assert!(snr > 20.0, "{snr}");
		// No pre-echo: the silence before the first burst stays quiet.
		let pre: f32 = out[0][1024 + 4000..1024 + 4990].iter().map(|v| v.abs()).fold(0.0, f32::max);
		assert!(pre < 200.0, "pre-echo {pre}");
	}

	#[test]
	fn surround_layout_decodes() {
		let n = 8192;
		let input: Vec<Vec<f32>> = (0..6)
			.map(|c| (0..n).map(|i| 3000.0 * (i as f32 * (0.01 + 0.007 * c as f32)).sin()).collect())
			.collect();
		let (enc, frames) = encode_all(48_000, 6, 384_000, &input);
		let out = decode_all(&enc.audio_specific_config(), &frames, 6);
		for c in 0..6 {
			// The LFE channel is band-limited to 250 Hz.
			let want = if c == 3 { 0.0 } else { 20.0 };
			let snr = snr_db(&input[c], &out[c][1024..1024 + n]);
			assert!(c == 3 || snr > want, "channel {c}: {snr}");
		}
	}
}
