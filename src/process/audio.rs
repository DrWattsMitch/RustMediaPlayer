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
//!
//! Output timestamps continue from the first input frame's `pts` in a
//! `1 / output rate` time base, counting the samples actually produced, so
//! they stay exact across resampling.

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
}

fn float_output(format: SampleFormat) -> bool {
	format.is_float()
}

/// Writes per-channel `f64` samples in `format` / `layout`.
fn store(planes: &[Vec<f64>], format: SampleFormat, layout: SampleLayout) -> Vec<u8> {
	let n = planes.first().map_or(0, Vec::len);
	let mut out = Vec::with_capacity(n * planes.len() * format.bytes());
	match layout {
		SampleLayout::Interleaved => {
			for i in 0..n {
				for plane in planes {
					write_sample(format, Sample::Float(plane[i]), &mut out);
				}
			}
		}
		SampleLayout::Planar => {
			for plane in planes {
				for &v in plane {
					write_sample(format, Sample::Float(v), &mut out);
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
		Ok(Self { input, output, matrix, resamplers, start: None, produced: 0 })
	}

	pub fn input(&self) -> AudioSpec {
		self.input
	}

	pub fn output(&self) -> AudioSpec {
		self.output
	}

	/// `true` when frames pass through unchanged.
	pub fn is_identity(&self) -> bool {
		self.input == self.output
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
		let data = store(&planes, self.output.format, self.output.layout);
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
		let data = store(&planes, self.output.format, self.output.layout);
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
