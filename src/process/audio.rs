//! Audio conversion: sample format, layout (interleaved/planar), channel
//! count and sample rate — what FFmpeg's `libswresample` (`aresample`,
//! `-ac`, `-ar`, `-sample_fmt`) does between a decoder and an encoder.
//!
//! [`AudioConverter`] is built once for an input and output [`AudioSpec`]
//! and then fed frames:
//!
//! - Only format/layout differ: samples are converted exactly with the PCM
//!   rules (`codecs::pcm::convert`), so the result equals `swr_convert`
//!   without dithering, bit for bit.
//! - Channels or rate differ: samples go through `f64`, the channel matrix
//!   ([`remix_matrix`], FFmpeg's default downmix/upmix coefficients) and one
//!   [`Resampler`] per output channel, then are rounded to the output format.
//! - With [`AudioConverter::dithered`], every conversion that loses
//!   precision (float → integer, wider → narrower integer, or any remix /
//!   resample into an integer format) is dithered instead of rounded or
//!   truncated ([`super::dither`]). Lossless conversions (integer → wider
//!   integer) stay exact.
//!
//! Output timestamps continue from the first input frame's `pts` in a
//! `1 / output rate` time base, counting the samples actually produced, so
//! they stay exact across resampling.

use super::dither::{Dither, Quantizer};
use crate::codecs::pcm::convert::{Sample, for_each_sample, write_sample};
use crate::core::resampler::Resampler;
use crate::core::{
	AudioFrame, Channels, SampleFormat, SampleLayout, Time, TimeBase, rescale_ticks,
};
use crate::message::{Message, Result};

const SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// What a stage of the audio pipeline carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpec {
	pub sample_rate: u32,
	pub channels: u8,
	pub format: SampleFormat,
	pub layout: SampleLayout,
}

impl AudioSpec {
	pub fn new(sample_rate: u32, channels: u8, format: SampleFormat, layout: SampleLayout) -> Self {
		Self { sample_rate, channels, format, layout }
	}

	/// The spec of an existing frame.
	pub fn of(frame: &AudioFrame) -> Self {
		Self::new(frame.sample_rate, frame.channels.count(), frame.format, frame.layout)
	}
}

/// Speaker positions used by the default layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
	FL,
	FR,
	FC,
	LFE,
	BL,
	BR,
	BC,
	SL,
	SR,
}

/// FFmpeg's default layout for a channel count (`av_channel_layout_default`):
/// mono, stereo, 2.1, 4.0, 5.0, 5.1, 6.1, 7.1.
pub fn default_layout(channels: u8) -> Option<&'static [Speaker]> {
	use Speaker::*;
	Some(match channels {
		1 => &[FC],
		2 => &[FL, FR],
		3 => &[FL, FR, LFE],
		4 => &[FL, FR, FC, BC],
		5 => &[FL, FR, FC, BL, BR],
		6 => &[FL, FR, FC, LFE, BL, BR],
		7 => &[FL, FR, FC, LFE, BC, SL, SR],
		8 => &[FL, FR, FC, LFE, BL, BR, SL, SR],
		_ => return None,
	})
}

/// Mixing matrix (`out × in`, row-major) between two default layouts, with
/// FFmpeg's default levels: centre and surround at -3 dB (`1/√2`), LFE
/// dropped. Channels present on both sides map one to one. As in
/// `swr_build_matrix`, rows are normalised to avoid clipping only when the
/// output is an integer format.
pub fn remix_matrix(input: u8, output: u8, integer_output: bool) -> Result<Vec<f64>> {
	use Speaker::*;
	let unsupported = Message::Other("no default channel layout for this channel count");
	let ins = default_layout(input).ok_or(unsupported)?;
	let outs = default_layout(output)
		.ok_or(Message::Other("no default channel layout for this channel count"))?;
	let (ni, no) = (ins.len(), outs.len());
	let mut m = vec![0.0f64; ni * no];
	let at = |s: Speaker| outs.iter().position(|&o| o == s);
	let has_in = |s: Speaker| ins.contains(&s);
	for (i, &speaker) in ins.iter().enumerate() {
		let mut add = |s: Speaker, gain: f64| -> bool {
			match at(s) {
				Some(o) => {
					m[o * ni + i] += gain;
					true
				}
				None => false,
			}
		};
		if add(speaker, 1.0) {
			continue;
		}
		// Fold an unmatched speaker into the nearest ones the output has.
		let to_front = |add: &mut dyn FnMut(Speaker, f64) -> bool, left: bool, gain: f64| {
			let side = if left { FL } else { FR };
			if !add(side, gain) {
				add(FC, gain * SQRT1_2);
			}
		};
		match speaker {
			FC => {
				if at(FL).is_some() && at(FR).is_some() {
					add(FL, SQRT1_2);
					add(FR, SQRT1_2);
				}
			}
			FL | FR => {
				add(FC, SQRT1_2);
			}
			LFE => {}
			BL | BR => {
				let left = speaker == BL;
				let side = if left { SL } else { SR };
				let gain = if has_in(side) { SQRT1_2 } else { 1.0 };
				if !add(side, gain) {
					to_front(&mut add, left, SQRT1_2);
				}
			}
			SL | SR => {
				let left = speaker == SL;
				let back = if left { BL } else { BR };
				let gain = if has_in(back) { SQRT1_2 } else { 1.0 };
				if !add(back, gain) {
					to_front(&mut add, left, SQRT1_2);
				}
			}
			BC => {
				if at(BL).is_some() && at(BR).is_some() {
					add(BL, SQRT1_2);
					add(BR, SQRT1_2);
				} else if at(SL).is_some() && at(SR).is_some() {
					add(SL, SQRT1_2);
					add(SR, SQRT1_2);
				} else if at(FL).is_some() && at(FR).is_some() {
					add(FL, 0.5);
					add(FR, 0.5);
				} else {
					add(FC, SQRT1_2);
				}
			}
		}
	}
	if integer_output {
		let max = (0..no)
			.map(|o| m[o * ni..(o + 1) * ni].iter().map(|v| v.abs()).sum::<f64>())
			.fold(0.0, f64::max);
		if max > 1.0 {
			for v in &mut m {
				*v /= max;
			}
		}
	}
	Ok(m)
}

/// Converts audio between two [`AudioSpec`]s, frame by frame.
#[derive(Debug, Clone)]
pub struct AudioConverter {
	input: AudioSpec,
	output: AudioSpec,
	/// `None` when the channel layout is unchanged.
	matrix: Option<Vec<f64>>,
	/// One per output channel; empty when the rate is unchanged.
	resamplers: Vec<Resampler>,
	/// pts of the first output sample, in output-rate ticks.
	start: Option<u64>,
	produced: u64,
	/// Dithered quantisation to `bits` bits; `None`: round / truncate.
	quantizer: Option<(Quantizer, u32)>,
}

/// Bits of an integer sample format.
fn format_bits(format: SampleFormat) -> u32 {
	(format.bytes() * 8) as u32
}

/// Appends a `bits`-bit signed integer, left-aligned, in integer `format`.
fn write_quantized(format: SampleFormat, bits: u32, value: i64, out: &mut Vec<u8>) {
	let aligned = value << (format_bits(format) - bits);
	match format {
		SampleFormat::U8 => out.push((aligned as i8 as u8) ^ 0x80),
		SampleFormat::S16 => out.extend_from_slice(&(aligned as i16).to_le_bytes()),
		SampleFormat::S32 => out.extend_from_slice(&(aligned as i32).to_le_bytes()),
		// Floats and S64 are never dithered (see `dithered`).
		_ => write_sample(format, Sample::Int(aligned << (64 - format_bits(format))), out),
	}
}

fn float_output(format: SampleFormat) -> bool {
	format.is_float()
}

/// Writes per-channel `f64` samples in `format` / `layout`, through the
/// dithering quantiser when there is one.
fn store(
	planes: &[Vec<f64>],
	format: SampleFormat,
	layout: SampleLayout,
	mut quantizer: Option<&mut (Quantizer, u32)>,
) -> Vec<u8> {
	let n = planes.first().map_or(0, Vec::len);
	let mut out = Vec::with_capacity(n * planes.len() * format.bytes());
	let mut put = |ch: usize, v: f64, out: &mut Vec<u8>| match quantizer.as_deref_mut() {
		Some((q, bits)) => write_quantized(format, *bits, q.quantize(ch, v), out),
		None => write_sample(format, Sample::Float(v), out),
	};
	match layout {
		SampleLayout::Interleaved => {
			for i in 0..n {
				for (ch, plane) in planes.iter().enumerate() {
					put(ch, plane[i], &mut out);
				}
			}
		}
		SampleLayout::Planar => {
			for (ch, plane) in planes.iter().enumerate() {
				for &v in plane {
					put(ch, v, &mut out);
				}
			}
		}
	}
	out
}

impl AudioConverter {
	pub fn new(input: AudioSpec, output: AudioSpec) -> Result<Self> {
		if input.channels == 0
			|| output.channels == 0
			|| input.sample_rate == 0
			|| output.sample_rate == 0
		{
			return Err(Message::Other("audio converter: invalid channel count or sample rate"));
		}
		let matrix = (input.channels != output.channels)
			.then(|| remix_matrix(input.channels, output.channels, !float_output(output.format)))
			.transpose()?;
		let resamplers = if input.sample_rate == output.sample_rate {
			Vec::new()
		} else {
			let first = Resampler::new(input.sample_rate, output.sample_rate)
				.ok_or(Message::Other("audio converter: invalid sample rate"))?;
			let mut all = vec![first.clone()];
			all.extend((1..output.channels).map(|_| first.new_like()));
			all
		};
		Ok(Self { input, output, matrix, resamplers, start: None, produced: 0, quantizer: None })
	}

	/// Dithers every conversion that loses precision, quantising to `bits`
	/// significant bits (at most the output format's; a 24-bit target held
	/// in `S32` is `bits = 24`, the low byte then stays zero). Conversions
	/// that lose nothing, float outputs and `S64` are left as they are.
	pub fn dithered(mut self, dither: Dither, bits: u32) -> Result<Self> {
		self.quantizer = None;
		let format = self.output.format;
		if dither.is_none() || format.is_float() || format == SampleFormat::S64 {
			return Ok(self);
		}
		let bits = bits.clamp(2, format_bits(format));
		let lossy =
			!self.exact_path() || self.input.format.is_float() || format_bits(self.input.format) > bits;
		if lossy {
			let quantizer =
				Quantizer::new(dither, bits, usize::from(self.output.channels), self.output.sample_rate)?;
			self.quantizer = Some((quantizer, bits));
		}
		Ok(self)
	}

	/// The conversion is dithered.
	pub fn is_dithered(&self) -> bool {
		self.quantizer.is_some()
	}

	pub fn input(&self) -> AudioSpec {
		self.input
	}

	pub fn output(&self) -> AudioSpec {
		self.output
	}

	/// `true` when frames pass through unchanged.
	pub fn is_identity(&self) -> bool {
		self.input == self.output && self.quantizer.is_none()
	}

	fn exact_path(&self) -> bool {
		self.matrix.is_none() && self.resamplers.is_empty()
	}

	fn frame(&mut self, data: Vec<u8>) -> AudioFrame {
		let out = self.output;
		let mut frame = AudioFrame::new(
			data,
			out.sample_rate,
			Channels::from_count(out.channels),
			out.format,
			out.layout,
		);
		let tb = TimeBase::new(1, out.sample_rate);
		frame.pts = self.start.map(|s| Time::new(s + self.produced, tb));
		self.produced += frame.nb_samples as u64;
		frame
	}

	/// Converts one frame. Returns `None` while the resampler is still
	/// filling up (no output yet).
	pub fn convert(&mut self, frame: &AudioFrame) -> Result<Option<AudioFrame>> {
		if frame.channels.count() != self.input.channels || frame.sample_rate != self.input.sample_rate
		{
			return Err(Message::Other("audio converter: frame does not match the input spec"));
		}
		if self.start.is_none() {
			let tb = TimeBase::new(1, self.output.sample_rate);
			self.start = frame.pts.map(|t| rescale_ticks(t.ticks(), t.base(), tb).unwrap_or(0));
		}
		if self.is_identity() {
			let mut out = frame.clone();
			out.pts =
				self.start.map(|s| Time::new(s + self.produced, TimeBase::new(1, self.output.sample_rate)));
			self.produced += out.nb_samples as u64;
			return Ok(Some(out));
		}
		if self.exact_path()
			&& let Some(quantizer) = self.quantizer.as_mut()
		{
			let channels = usize::from(self.output.channels);
			let mut planes: Vec<Vec<f64>> = vec![Vec::with_capacity(frame.nb_samples); channels];
			let mut ch = 0;
			for_each_sample(frame, |s| {
				planes[ch].push(s.to_float());
				ch = (ch + 1) % channels;
			})?;
			let data = store(&planes, self.output.format, self.output.layout, Some(quantizer));
			return Ok(Some(self.frame(data)));
		}
		if self.exact_path() {
			let format = self.output.format;
			let data = match self.output.layout {
				SampleLayout::Interleaved => crate::codecs::pcm::convert::convert(frame, format)?,
				SampleLayout::Planar => {
					let channels = usize::from(self.output.channels);
					let mut planes: Vec<Vec<u8>> = vec![Vec::new(); channels];
					let mut ch = 0;
					for_each_sample(frame, |s| {
						write_sample(format, s, &mut planes[ch]);
						ch = (ch + 1) % channels;
					})?;
					planes.concat()
				}
			};
			return Ok(Some(self.frame(data)));
		}

		// General path: f64 planes → matrix → resample → output format.
		let ni = usize::from(self.input.channels);
		let mut planes: Vec<Vec<f64>> = vec![Vec::with_capacity(frame.nb_samples); ni];
		let mut ch = 0;
		for_each_sample(frame, |s| {
			planes[ch].push(s.to_float());
			ch = (ch + 1) % ni;
		})?;
		let planes = self.mix(planes);
		let planes = if self.resamplers.is_empty() {
			planes
		} else {
			planes.iter().zip(self.resamplers.iter_mut()).map(|(p, r)| r.process(p)).collect()
		};
		if planes.first().is_none_or(Vec::is_empty) {
			return Ok(None);
		}
		let data = store(&planes, self.output.format, self.output.layout, self.quantizer.as_mut());
		Ok(Some(self.frame(data)))
	}

	fn mix(&self, planes: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
		let Some(m) = &self.matrix else { return planes };
		let ni = planes.len();
		let n = planes.first().map_or(0, Vec::len);
		(0..usize::from(self.output.channels))
			.map(|o| {
				let row = &m[o * ni..(o + 1) * ni];
				(0..n).map(|i| row.iter().zip(&planes).map(|(g, p)| g * p[i]).sum()).collect()
			})
			.collect()
	}

	/// Ends the stream: returns the resampler's remaining output, if any.
	pub fn flush(&mut self) -> Result<Option<AudioFrame>> {
		if self.resamplers.is_empty() {
			return Ok(None);
		}
		let planes: Vec<Vec<f64>> = self.resamplers.iter_mut().map(Resampler::flush).collect();
		if planes.first().is_none_or(Vec::is_empty) {
			return Ok(None);
		}
		let data = store(&planes, self.output.format, self.output.layout, self.quantizer.as_mut());
		Ok(Some(self.frame(data)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn row(m: &[f64], ni: usize, o: usize) -> Vec<f64> {
		m[o * ni..(o + 1) * ni].iter().map(|v| (v * 1e4).round() / 1e4).collect()
	}

	/// Coefficients measured from FFmpeg 6.1 (`-ac N` on constant inputs).
	#[test]
	#[allow(clippy::approx_constant)]
	fn matrices_match_ffmpeg() {
		let m = remix_matrix(6, 2, false).unwrap();
		assert_eq!(row(&m, 6, 0), vec![1.0, 0.0, 0.7071, 0.0, 0.7071, 0.0]);
		let m = remix_matrix(6, 2, true).unwrap();
		assert!((m[0] - 1.0 / (1.0 + 2.0 * SQRT1_2)).abs() < 1e-12);
		let m = remix_matrix(2, 1, false).unwrap();
		assert_eq!(row(&m, 2, 0), vec![0.7071, 0.7071]);
		assert_eq!(row(&remix_matrix(2, 1, true).unwrap(), 2, 0), vec![0.5, 0.5]);
		let m = remix_matrix(1, 2, true).unwrap();
		assert_eq!((row(&m, 1, 0), row(&m, 1, 1)), (vec![0.7071], vec![0.7071]));
		let m = remix_matrix(6, 1, false).unwrap();
		assert_eq!(row(&m, 6, 0), vec![0.7071, 0.7071, 1.0, 0.0, 0.5, 0.5]);
		let m = remix_matrix(8, 6, false).unwrap();
		assert_eq!(row(&m, 8, 4), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7071, 0.0]);
		let m = remix_matrix(8, 2, false).unwrap();
		assert_eq!(row(&m, 8, 0), vec![1.0, 0.0, 0.7071, 0.0, 0.7071, 0.0, 0.7071, 0.0]);
		assert!(remix_matrix(9, 2, false).is_err());
	}

	#[test]
	fn format_only_conversion_is_exact() {
		let data: Vec<u8> =
			[1000i16, -2000, 32767, -32768].iter().flat_map(|v| v.to_le_bytes()).collect();
		let frame =
			AudioFrame::new(data, 8000, Channels::Stereo, SampleFormat::S16, SampleLayout::Interleaved);
		let mut c = AudioConverter::new(
			AudioSpec::of(&frame),
			AudioSpec::new(8000, 2, SampleFormat::S32, SampleLayout::Planar),
		)
		.unwrap();
		let out = c.convert(&frame).unwrap().unwrap();
		let v: Vec<i32> =
			out.data.chunks(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
		assert_eq!(v, vec![1000 << 16, 32767 << 16, -2000 << 16, -32768 << 16]);
		assert_eq!(out.nb_samples, 2);
	}

	fn f32_frame(samples: &[f32], rate: u32, channels: Channels) -> AudioFrame {
		let data = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
		AudioFrame::new(data, rate, channels, SampleFormat::F32, SampleLayout::Interleaved)
	}

	fn s16(data: &[u8]) -> Vec<i16> {
		data.chunks(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
	}

	/// |DFT|² of `x` at `freq` Hz.
	fn tone_power(x: &[f64], rate: f64, freq: f64) -> f64 {
		let w = 2.0 * std::f64::consts::PI * freq / rate;
		let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (i, &v)| {
			(re + v * (w * i as f64).cos(), im + v * (w * i as f64).sin())
		});
		re * re + im * im
	}

	#[test]
	fn dither_removes_quantisation_distortion() {
		// A 1 kHz tone 1.5 LSB high: plain rounding turns it into a
		// staircase with strong odd harmonics; TPDF dither leaves noise only.
		let rate = 48_000;
		let n = 48_000;
		let tone: Vec<f32> = (0..n)
			.map(|i| {
				((i as f64 * 2.0 * std::f64::consts::PI * 1000.0 / rate as f64).sin() * 1.5 / 32768.0)
					as f32
			})
			.collect();
		let frame = f32_frame(&tone, rate, Channels::Mono);
		let spec = AudioSpec::new(rate, 1, SampleFormat::S16, SampleLayout::Interleaved);
		let third = |dither: Dither| {
			let mut c =
				AudioConverter::new(AudioSpec::of(&frame), spec).unwrap().dithered(dither, 16).unwrap();
			let out: Vec<f64> =
				s16(&c.convert(&frame).unwrap().unwrap().data).iter().map(|&v| f64::from(v)).collect();
			(tone_power(&out, f64::from(rate), 1000.0), tone_power(&out, f64::from(rate), 3000.0))
		};
		let (_, harmonic_plain) = third(Dither::NONE);
		let (fundamental, harmonic) = third(Dither::TRIANGULAR);
		// The tone keeps its true level (within 1 dB) and the third
		// harmonic drops by more than 20 dB.
		let exact = (1.5 * n as f64 / 2.0).powi(2);
		assert!((fundamental / exact - 1.0).abs() < 0.26, "{fundamental} vs {exact}");
		assert!(harmonic < harmonic_plain / 100.0, "{harmonic} vs {harmonic_plain}");
	}

	#[test]
	fn dither_applies_only_where_precision_is_lost() {
		let s16_spec = AudioSpec::new(8000, 2, SampleFormat::S16, SampleLayout::Interleaved);
		let s32_spec = AudioSpec::new(8000, 2, SampleFormat::S32, SampleLayout::Interleaved);
		let f32_spec = AudioSpec::new(8000, 2, SampleFormat::F32, SampleLayout::Planar);
		let dithered = |a, b, bits| {
			AudioConverter::new(a, b).unwrap().dithered(Dither::TRIANGULAR, bits).unwrap().is_dithered()
		};
		assert!(!dithered(s16_spec, s32_spec, 32), "widening is exact");
		assert!(!dithered(s16_spec, s16_spec, 16));
		assert!(!dithered(s16_spec, f32_spec, 32), "float output");
		assert!(dithered(s32_spec, s16_spec, 16));
		assert!(dithered(f32_spec, s16_spec, 16));
		assert!(dithered(s32_spec, s32_spec, 24), "24-bit target held in S32");
		let resample = AudioSpec::new(16_000, 2, SampleFormat::S16, SampleLayout::Interleaved);
		assert!(dithered(s16_spec, resample, 16), "resampling recomputes every sample");
		assert!(
			!AudioConverter::new(f32_spec, s16_spec)
				.unwrap()
				.dithered(Dither::NONE, 16)
				.unwrap()
				.is_dithered()
		);
	}

	#[test]
	fn dithered_24_bit_output_keeps_the_low_byte_clear() {
		let samples: Vec<f32> = (0..2000).map(|i| ((i as f32) * 0.01).sin() * 0.3).collect();
		let frame = f32_frame(&samples, 44_100, Channels::Stereo);
		let mut c = AudioConverter::new(
			AudioSpec::of(&frame),
			AudioSpec::new(44_100, 2, SampleFormat::S32, SampleLayout::Planar),
		)
		.unwrap()
		.dithered(Dither::new(crate::process::DitherMethod::Lipshitz), 24)
		.unwrap();
		let out = c.convert(&frame).unwrap().unwrap();
		let values: Vec<i32> =
			out.data.chunks(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
		assert!(values.iter().all(|v| v & 0xFF == 0));
		// Planar: channel 0 first. Each value is within a few 24-bit LSB of
		// the exact one.
		for (i, v) in values[..1000].iter().enumerate() {
			let exact = f64::from(samples[2 * i]) * f64::from(1 << 23);
			assert!((f64::from(v >> 8) - exact).abs() < 8.0);
		}
	}

	#[test]
	fn resampling_keeps_count_and_timestamps() {
		let rate_in = 44_100;
		let n = 44_100usize;
		let data: Vec<u8> =
			(0..n).flat_map(|i| (((i as f64 * 0.05).sin() * 0.5) as f32).to_le_bytes()).collect();
		let input = AudioSpec::new(rate_in, 1, SampleFormat::F32, SampleLayout::Interleaved);
		let mut c = AudioConverter::new(
			input,
			AudioSpec::new(48_000, 2, SampleFormat::S16, SampleLayout::Interleaved),
		)
		.unwrap();
		let mut total = 0;
		let mut first_pts = None;
		for (i, chunk) in data.chunks(4 * 1024).enumerate() {
			let mut f = AudioFrame::new(
				chunk.to_vec(),
				rate_in,
				Channels::Mono,
				SampleFormat::F32,
				SampleLayout::Interleaved,
			);
			f.pts = Some(Time::new(1000 + (i * 1024) as u64, TimeBase::new(1, rate_in)));
			if let Some(out) = c.convert(&f).unwrap() {
				first_pts = first_pts.or(out.pts);
				total += out.nb_samples;
			}
		}
		total += c.flush().unwrap().map_or(0, |f| f.nb_samples);
		assert_eq!(total, 48_000);
		// pts 1000 at 44.1 kHz = 1088 at 48 kHz.
		assert_eq!(first_pts.map(|t| t.ticks()), Some(1088));
	}
}

/// Drops the gapless priming (`encoder_delay`) and padding
/// (`encoder_padding`) samples a demuxer reports, as FFmpeg's
/// `skip_samples` / end trimming do, so a decode → encode keeps the
/// source's exact length and timing.
///
/// Frames pass through [`Gapless::push`]; the last `padding` samples are
/// held back until [`Gapless::finish`] drops them.
#[derive(Debug, Clone, Default)]
pub struct Gapless {
	skip: u64,
	padding: u64,
	held: std::collections::VecDeque<AudioFrame>,
	held_samples: u64,
}

impl Gapless {
	pub fn new(delay: u64, padding: u64) -> Self {
		Self { skip: delay, padding, held: Default::default(), held_samples: 0 }
	}

	/// Accepts a decoded frame; returns the frames now safe to emit.
	pub fn push(&mut self, frame: AudioFrame) -> Vec<AudioFrame> {
		let mut frame = frame;
		if self.skip > 0 {
			let n = frame.nb_samples as u64;
			let drop = self.skip.min(n);
			self.skip -= drop;
			if drop == n {
				return Vec::new();
			}
			frame = slice_frame(&frame, drop as usize, (n - drop) as usize);
		}
		if self.padding == 0 {
			return vec![frame];
		}
		self.held_samples += frame.nb_samples as u64;
		self.held.push_back(frame);
		let mut out = Vec::new();
		while let Some(front) = self.held.front() {
			let n = front.nb_samples as u64;
			if self.held_samples - n < self.padding {
				break;
			}
			self.held_samples -= n;
			out.extend(self.held.pop_front());
		}
		out
	}

	/// End of stream: the held frames minus the trailing padding.
	pub fn finish(&mut self) -> Vec<AudioFrame> {
		let mut keep = self.held_samples.saturating_sub(self.padding);
		let mut out = Vec::new();
		while let Some(frame) = self.held.pop_front() {
			let n = frame.nb_samples as u64;
			if keep == 0 {
				break;
			}
			if n <= keep {
				keep -= n;
				out.push(frame);
			} else {
				out.push(slice_frame(&frame, 0, keep as usize));
				keep = 0;
			}
		}
		self.held.clear();
		self.held_samples = 0;
		out
	}
}

/// `len` samples of `frame` from sample `start` (any layout), with the
/// timestamp moved accordingly.
pub fn slice_frame(frame: &AudioFrame, start: usize, len: usize) -> AudioFrame {
	let channels = usize::from(frame.channels.count()).max(1);
	let size = frame.format.bytes();
	let start = start.min(frame.nb_samples);
	let len = len.min(frame.nb_samples - start);
	let data = if frame.layout.is_planar() {
		let plane = frame.data.len() / channels;
		let mut data = Vec::with_capacity(len * size * channels);
		for c in 0..channels {
			let base = c * plane;
			data.extend_from_slice(
				frame.data.get(base + start * size..base + (start + len) * size).unwrap_or(&[]),
			);
		}
		data
	} else {
		let stride = size * channels;
		frame.data.get(start * stride..(start + len) * stride).unwrap_or(&[]).to_vec()
	};
	let pts = frame.pts.and_then(|pts| {
		let offset = Time::new(start as u64, TimeBase::new(1, frame.sample_rate.max(1)));
		offset.rescale(pts.base()).and_then(|o| pts.checked_add(o))
	});
	AudioFrame { data, nb_samples: len, pts, ..frame.clone() }
}
