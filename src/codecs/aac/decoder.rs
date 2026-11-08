//! AAC-LC frame decoder (FFmpeg: `aac/aacdec.c`, `aacdec_dsp_template.c`,
//! `aacdec_float.c`).
//!
//! One `raw_data_block()` per call:
//!
//! 1. walk the syntactic elements (SCE/CPE/LFE decode, DSE/FIL/PCE skip),
//! 2. per channel: Huffman decode + inverse quantisation + PNS ([`super::ics`]),
//! 3. per channel pair: mid/side, then intensity stereo,
//! 4. TNS, then the IMDCT filterbank with window switching and overlap-add,
//! 5. interleave to signed 16-bit PCM in FFmpeg's native channel order.

use std::sync::OnceLock;

use super::config::{self, AacConfig, AdtsHeader, ElementKind};
use super::ics::{
	self, Channel, FRAME_LEN, INTENSITY_BT, INTENSITY_BT2, IcsResult, Invalid, NOISE_BT, SHORT_LEN,
	WindowSequence,
};
use crate::codecs::bitstream::BitReader;
use crate::codecs::dsp::{Complex, Imdct};
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State,
};
use crate::message::{Message, Result};

const LONG_WINDOW: usize = 2 * FRAME_LEN;
const SHORT_WINDOW: usize = 2 * SHORT_LEN;
/// Short windows start this far into the long-window frame (§4.6.11.3.2).
const SHORT_START: usize = (FRAME_LEN - SHORT_LEN) / 2;
/// FFmpeg seeds its PNS generator with this value.
const RANDOM_SEED: u32 = 0x1f2e_3d4c;
/// Kaiser window alpha for long and short KBD windows (§4.6.11.3.2).
const KBD_ALPHA_LONG: f64 = 4.0;
const KBD_ALPHA_SHORT: f64 = 6.0;

/// Window halves and transforms shared by every decoder instance.
struct Filterbank {
	sine_long: Vec<f32>,
	sine_short: Vec<f32>,
	kbd_long: Vec<f32>,
	kbd_short: Vec<f32>,
	imdct_long: Imdct,
	imdct_short: Imdct,
}

impl Filterbank {
	fn get() -> &'static Filterbank {
		static FB: OnceLock<Filterbank> = OnceLock::new();
		FB.get_or_init(|| Filterbank {
			sine_long: sine_window(LONG_WINDOW),
			sine_short: sine_window(SHORT_WINDOW),
			kbd_long: kbd_window(LONG_WINDOW, KBD_ALPHA_LONG),
			kbd_short: kbd_window(SHORT_WINDOW, KBD_ALPHA_SHORT),
			// The specification's IMDCT carries a 2/N factor.
			imdct_long: Imdct::new(LONG_WINDOW, 2.0 / LONG_WINDOW as f32),
			imdct_short: Imdct::new(SHORT_WINDOW, 2.0 / SHORT_WINDOW as f32),
		})
	}

	fn long(&self, kbd: bool) -> &[f32] {
		if kbd { &self.kbd_long } else { &self.sine_long }
	}

	fn short(&self, kbd: bool) -> &[f32] {
		if kbd { &self.kbd_short } else { &self.sine_short }
	}
}

/// Rising half (`n/2` values) of a sine window of length `n`.
fn sine_window(n: usize) -> Vec<f32> {
	(0..n / 2).map(|i| ((std::f64::consts::PI / n as f64) * (i as f64 + 0.5)).sin() as f32).collect()
}

/// Rising half of a Kaiser-Bessel-derived window of length `n`.
fn kbd_window(n: usize, alpha: f64) -> Vec<f32> {
	let half = n / 2;
	let kernel: Vec<f64> = (0..=half)
		.map(|j| {
			let x = (j as f64 - half as f64 / 2.0) / (half as f64 / 2.0);
			bessel_i0(std::f64::consts::PI * alpha * (1.0 - x * x).max(0.0).sqrt())
		})
		.collect();
	let total: f64 = kernel.iter().sum();
	let mut acc = 0.0;
	(0..half)
		.map(|i| {
			acc += kernel[i];
			(acc / total).sqrt() as f32
		})
		.collect()
}

/// Zeroth-order modified Bessel function of the first kind (power series).
fn bessel_i0(x: f64) -> f64 {
	let mut sum = 1.0;
	let mut term = 1.0;
	let q = x * x / 4.0;
	for k in 1..64 {
		term *= q / (k as f64 * k as f64);
		sum += term;
		if term < sum * 1e-17 {
			break;
		}
	}
	sum
}

/// Per-output-channel history between frames.
#[derive(Debug, Clone)]
struct History {
	overlap: Vec<f32>,
	prev_kbd: bool,
}

impl Default for History {
	fn default() -> Self {
		Self { overlap: vec![0.0; FRAME_LEN], prev_kbd: false }
	}
}

/// Persistent decoder state (FFmpeg's `AACDecContext`).
#[derive(Debug, Clone)]
pub struct AacDecoderState {
	config: Option<AacConfig>,
	/// `true` when the configuration came from extradata (ADTS headers are
	/// then only skipped, not trusted for the layout).
	from_extradata: bool,
	history: Vec<History>,
	random: u32,
	work: Box<[Channel; 2]>,
	/// Frames that failed to decode and were replaced by silence.
	pub concealed_frames: u64,
}

impl Default for AacDecoderState {
	fn default() -> Self {
		Self::new(None)
	}
}

/// Result of decoding one `raw_data_block()`.
pub struct DecodedFrame {
	pub sample_rate: u32,
	pub channels: u8,
	/// Interleaved samples: `1024 * channels` values.
	pub pcm: Vec<i16>,
	/// `false` when the frame was concealed with silence.
	pub complete: bool,
}

impl AacDecoderState {
	pub fn new(config: Option<AacConfig>) -> Self {
		let from_extradata = config.is_some();
		let channels = config.as_ref().map_or(0, |c| usize::from(c.channels));
		Self {
			config,
			from_extradata,
			history: vec![History::default(); channels],
			random: RANDOM_SEED,
			work: Box::default(),
			concealed_frames: 0,
		}
	}

	/// Creates a state from an AudioSpecificConfig (MP4 extradata).
	pub fn from_extradata(asc: &[u8]) -> Option<Self> {
		config::parse_audio_specific_config(asc).map(|c| Self::new(Some(c)))
	}

	pub fn config(&self) -> Option<&AacConfig> {
		self.config.as_ref()
	}

	/// Clears overlap history (after a seek).
	pub fn reset(&mut self) {
		for h in &mut self.history {
			*h = History::default();
		}
		self.random = RANDOM_SEED;
	}

	fn set_config(&mut self, config: AacConfig) {
		if self.config.as_ref() != Some(&config) {
			self.history = vec![History::default(); usize::from(config.channels)];
			self.config = Some(config);
		}
	}

	/// Decodes one packet: either one raw_data_block (MP4) or one or more
	/// ADTS frames.
	pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<DecodedFrame>> {
		if AdtsHeader::parse(data).is_some() {
			let mut out = Vec::new();
			let mut at = 0;
			while let Some(h) = data.get(at..).and_then(AdtsHeader::parse) {
				let frame =
					data.get(at..at + h.frame_length).ok_or(Message::Codec("truncated adts frame"))?;
				if !self.from_extradata {
					let c =
						AacConfig::from_channel_config(h.object_type, h.sample_rate_index, h.channel_config)
							.ok_or(Message::Codec("unsupported adts channel configuration"))?;
					self.set_config(c);
				}
				let payload = &frame[h.header_length..];
				let mut bs = BitReader::new(payload);
				for _ in 0..h.raw_blocks {
					out.push(self.decode_block(&mut bs)?);
					bs.align();
				}
				at += h.frame_length;
			}
			return Ok(out);
		}
		let mut bs = BitReader::new(data);
		Ok(vec![self.decode_block(&mut bs)?])
	}

	/// Decodes one raw_data_block, concealing damaged content with silence.
	pub fn decode_block(&mut self, bs: &mut BitReader) -> Result<DecodedFrame> {
		let config = self.config.clone().ok_or(Message::Codec("aac decoder has no configuration"))?;
		let channels = usize::from(config.channels);
		let mut planar = vec![0.0f32; channels * FRAME_LEN];
		let complete = match self.decode_elements(bs, &config, &mut planar) {
			Ok(()) => true,
			Err(_) => {
				planar.fill(0.0);
				self.reset();
				self.concealed_frames += 1;
				false
			}
		};
		let mut pcm = vec![0i16; channels * FRAME_LEN];
		for (c, plane) in planar.chunks_exact(FRAME_LEN).enumerate() {
			for (i, &x) in plane.iter().enumerate() {
				pcm[i * channels + c] = x.round().clamp(-32768.0, 32767.0) as i16;
			}
		}
		Ok(DecodedFrame { sample_rate: config.sample_rate, channels: config.channels, pcm, complete })
	}

	fn decode_elements(
		&mut self,
		bs: &mut BitReader,
		config: &AacConfig,
		out: &mut [f32],
	) -> IcsResult<()> {
		let mut seen = [0usize; 4];
		let rate = config.sample_rate_index;
		let aot = config.object_type;
		loop {
			if bs.overrun() {
				return Err(Invalid("aac truncated frame"));
			}
			let kind = ElementKind::from_id(bs.bits(3));
			match kind {
				ElementKind::End => return Ok(()),
				ElementKind::Sce | ElementKind::Lfe => {
					bs.skip(4);
					let nth = &mut seen[if kind == ElementKind::Sce { 0 } else { 2 }];
					let slot = config.slot(kind, *nth);
					*nth += 1;
					let ch = &mut self.work[0];
					ics::decode_ics(bs, ch, rate, aot, false, &mut self.random)?;
					ics::apply_tns(ch);
					if let Some(s) = slot {
						let out_ch = usize::from(s.out[0]);
						if let (Some(h), Some(dst)) =
							(self.history.get_mut(out_ch), out.chunks_exact_mut(FRAME_LEN).nth(out_ch))
						{
							synthesize(ch, h, dst);
						}
					}
				}
				ElementKind::Cpe => {
					bs.skip(4);
					let slot = config.slot(kind, seen[1]);
					seen[1] += 1;
					self.decode_cpe(bs, rate, aot)?;
					if let Some(s) = slot {
						for (i, ch) in self.work.iter().enumerate() {
							let out_ch = usize::from(s.out[i]);
							if let (Some(h), Some(dst)) =
								(self.history.get_mut(out_ch), out.chunks_exact_mut(FRAME_LEN).nth(out_ch))
							{
								synthesize(ch, h, dst);
							}
						}
					}
				}
				ElementKind::Cce => return Err(Invalid("aac coupling channel elements are not supported")),
				ElementKind::Dse => {
					bs.skip(4);
					let align = bs.bit();
					let mut count = bs.bits(8) as usize;
					if count == 255 {
						count += bs.bits(8) as usize;
					}
					if align {
						bs.align();
					}
					bs.skip(8 * count);
				}
				ElementKind::Pce => {
					config::parse_program_config(bs).ok_or(Invalid("aac invalid program config"))?;
				}
				ElementKind::Fil => {
					let mut count = bs.bits(4) as usize;
					if count == 15 {
						count += bs.bits(8) as usize;
						count -= 1;
					}
					// Extension payloads (SBR, DRC) are skipped.
					bs.skip(8 * count);
				}
			}
		}
	}

	/// `channel_pair_element()` (Table 4.5) plus M/S and intensity stereo.
	fn decode_cpe(&mut self, bs: &mut BitReader, rate: usize, aot: u8) -> IcsResult<()> {
		let common_window = bs.bit();
		let mut ms_mask = [false; 128];
		let mut ms_present = 0;
		let [left, right] = &mut *self.work;
		if common_window {
			ics::decode_ics_info(bs, rate, &mut left.ics)?;
			right.ics = left.ics.clone();
			ms_present = bs.bits(2);
			let bands = left.ics.num_window_groups * left.ics.max_sfb;
			match ms_present {
				1 => {
					for m in &mut ms_mask[..bands] {
						*m = bs.bit();
					}
				}
				2 => ms_mask[..bands].fill(true),
				3 => return Err(Invalid("aac reserved ms_mask_present")),
				_ => {}
			}
		}
		ics::decode_ics(bs, left, rate, aot, common_window, &mut self.random)?;
		ics::decode_ics(bs, right, rate, aot, common_window, &mut self.random)?;
		if common_window {
			if ms_present != 0 {
				apply_mid_side(left, right, &ms_mask);
			}
			apply_intensity(left, right, &ms_mask, ms_present != 0);
		}
		ics::apply_tns(left);
		ics::apply_tns(right);
		Ok(())
	}
}

/// Iterates `(band index, window range, coefficient range)` over a channel's
/// groups and bands.
fn for_each_band(ch: &Channel, mut f: impl FnMut(usize, usize, usize, usize)) {
	let ics = &ch.ics;
	let mut window = 0;
	for g in 0..ics.num_window_groups {
		let glen = usize::from(ics.group_len[g]);
		for sfb in 0..ics.max_sfb {
			let lo = usize::from(ics.swb_offset[sfb]);
			let hi = usize::from(ics.swb_offset[sfb + 1]);
			for w in window..window + glen {
				f(g * ics.max_sfb + sfb, w * SHORT_LEN + lo, w * SHORT_LEN + hi, w);
			}
		}
		window += glen;
	}
}

/// Mid/side stereo (§4.6.8.1). Like FFmpeg, noise and intensity bands are
/// left untouched.
fn apply_mid_side(left: &mut Channel, right: &mut Channel, mask: &[bool; 128]) {
	let (lt, rt) = (left.band_type, right.band_type);
	let mut ranges = Vec::new();
	for_each_band(left, |idx, lo, hi, _| {
		if mask[idx] && lt[idx] < NOISE_BT && rt[idx] < NOISE_BT {
			ranges.push((lo, hi));
		}
	});
	for (lo, hi) in ranges {
		for k in lo..hi.min(FRAME_LEN) {
			let (m, s) = (left.coef[k], right.coef[k]);
			left.coef[k] = m + s;
			right.coef[k] = m - s;
		}
	}
}

/// Intensity stereo (§4.6.8.2): the right channel is a scaled copy of the
/// left in bands coded with codebook 14/15.
fn apply_intensity(left: &Channel, right: &mut Channel, mask: &[bool; 128], ms_present: bool) {
	let rt = right.band_type;
	let sf = right.sf;
	let mut ranges = Vec::new();
	for_each_band(right, |idx, lo, hi, _| {
		let bt = rt[idx];
		if bt == INTENSITY_BT || bt == INTENSITY_BT2 {
			let mut sign = if bt == INTENSITY_BT { 1.0 } else { -1.0 };
			if ms_present && mask[idx] {
				sign = -sign;
			}
			ranges.push((lo, hi, sign * ics::pow2_quarter(-sf[idx])));
		}
	});
	for (lo, hi, scale) in ranges {
		for k in lo..hi.min(FRAME_LEN) {
			right.coef[k] = left.coef[k] * scale;
		}
	}
}

/// IMDCT, windowing and overlap-add (§4.6.11; FFmpeg `imdct_and_windowing`).
fn synthesize(ch: &Channel, hist: &mut History, out: &mut [f32]) {
	let fb = Filterbank::get();
	let ics = &ch.ics;
	let mut buf = [0.0f32; LONG_WINDOW];
	let mut scratch = [0.0f32; FRAME_LEN];
	let mut cbuf = [Complex::default(); FRAME_LEN / 2];
	let cur_long = fb.long(ics.kbd_window);
	let cur_short = fb.short(ics.kbd_window);
	let prev_long = fb.long(hist.prev_kbd);
	let prev_short = fb.short(hist.prev_kbd);

	if ics.window_sequence == WindowSequence::EightShort {
		let mut tmp = [0.0f32; SHORT_WINDOW];
		for w in 0..8 {
			fb.imdct_short.inverse(
				&ch.coef[w * SHORT_LEN..(w + 1) * SHORT_LEN],
				&mut tmp,
				&mut scratch,
				&mut cbuf,
			);
			let rise = if w == 0 { prev_short } else { cur_short };
			let base = SHORT_START + w * SHORT_LEN;
			for i in 0..SHORT_LEN {
				buf[base + i] += tmp[i] * rise[i];
				buf[base + SHORT_LEN + i] += tmp[SHORT_LEN + i] * cur_short[SHORT_LEN - 1 - i];
			}
		}
	} else {
		fb.imdct_long.inverse(&ch.coef, &mut buf, &mut scratch, &mut cbuf);
		// Left half: shaped by the previous frame's window.
		if ics.window_sequence == WindowSequence::LongStop {
			buf[..SHORT_START].fill(0.0);
			for i in 0..SHORT_LEN {
				buf[SHORT_START + i] *= prev_short[i];
			}
		} else {
			for i in 0..FRAME_LEN {
				buf[i] *= prev_long[i];
			}
		}
		// Right half: shaped by this frame's window.
		if ics.window_sequence == WindowSequence::LongStart {
			let base = FRAME_LEN + SHORT_START;
			for i in 0..SHORT_LEN {
				buf[base + i] *= cur_short[SHORT_LEN - 1 - i];
			}
			buf[base + SHORT_LEN..].fill(0.0);
		} else {
			for i in 0..FRAME_LEN {
				buf[FRAME_LEN + i] *= cur_long[FRAME_LEN - 1 - i];
			}
		}
	}

	for i in 0..FRAME_LEN {
		out[i] = buf[i] + hist.overlap[i];
	}
	hist.overlap.copy_from_slice(&buf[FRAME_LEN..]);
	hist.prev_kbd = ics.kbd_window;
}

/// [`core::Decoder`] implementation: one AAC access unit (or ADTS frame)
/// per packet in, one interleaved S16 [`AudioFrame`] of 1024 samples out.
#[derive(Debug, Clone, Default)]
pub struct AacDecoder;

impl AacDecoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for AacDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		let dec = AacDecoderState::from_extradata(&parameters.extradata).unwrap_or_default();
		let (rate, channels) = dec
			.config()
			.map_or((parameters.sample_rate.unwrap_or(44_100), parameters.channels.unwrap_or(2)), |c| {
				(c.sample_rate, c.channels)
			});
		State::Audio(AudioState {
			sample_rate: rate,
			channels: Channels::from_count(channels),
			format: SampleFormat::S16,
			layout: SampleLayout::Interleaved,
			frame_size: FRAME_LEN,
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::Aac(Box::new(dec)),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let frames = {
			let State::Audio(audio) = state else {
				return Err(Message::Codec("aac decoder requires an audio state"));
			};
			if !matches!(audio.private, CodecPrivate::Aac(_)) {
				let dec = AacDecoderState::from_extradata(&audio.codec_state.extradata).unwrap_or_default();
				audio.private = CodecPrivate::Aac(Box::new(dec));
			}
			let CodecPrivate::Aac(dec) = &mut audio.private else {
				return Err(Message::Codec("aac decoder state missing"));
			};
			let decoded = dec.decode_packet(&packet.data)?;
			let mut frames = Vec::with_capacity(decoded.len());
			let mut pts = packet.pts;
			for frame in decoded {
				let mut bytes = Vec::with_capacity(frame.pcm.len() * 2);
				for s in &frame.pcm {
					bytes.extend_from_slice(&s.to_le_bytes());
				}
				let mut out = AudioFrame::new(
					bytes,
					frame.sample_rate,
					Channels::from_count(frame.channels),
					SampleFormat::S16,
					SampleLayout::Interleaved,
				);
				out.pts = pts;
				pts = pts.and_then(|t| {
					let step = core::Time::new(FRAME_LEN as u64, core::TimeBase::new(1, frame.sample_rate));
					step.rescale(t.base()).and_then(|s| t.checked_add(s))
				});
				audio.sample_rate = frame.sample_rate;
				audio.channels = Channels::from_count(frame.channels);
				audio.samples_produced += FRAME_LEN as u64;
				frames.push(Frame::Audio(out));
			}
			audio.samples_consumed += packet.data.len() as u64;
			frames
		};
		Ok(DecodeOut { value: frames, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		if let State::Audio(audio) = state {
			if let CodecPrivate::Aac(dec) = &mut audio.private {
				dec.reset();
			}
		}
		Ok(DecodeOut { value: Vec::new(), state: state.clone() })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn windows_satisfy_princen_bradley() {
		for w in [sine_window(2048), kbd_window(2048, 4.0), kbd_window(256, 6.0)] {
			let n = w.len();
			for i in 0..n {
				let s = w[i] * w[i] + w[n - 1 - i] * w[n - 1 - i];
				assert!((s - 1.0).abs() < 1e-5, "{s}");
			}
		}
	}

	#[test]
	fn missing_config_is_an_error() {
		assert!(AacDecoderState::default().decode_packet(&[0x00, 0x01]).is_err());
	}
}
