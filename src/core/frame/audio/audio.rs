use crate::core::time::Time;

use super::format::{Channels, SampleFormat, SampleLayout};

#[derive(Debug, Clone)]
pub struct AudioFrame {
	pub data: Vec<u8>,
	pub sample_rate: u32,
	pub channels: Channels,
	pub format: SampleFormat,
	pub layout: SampleLayout,
	pub nb_samples: usize,
	pub pts: Option<Time>,
}

impl AudioFrame {
	pub fn new(
		data: Vec<u8>,
		sample_rate: u32,
		channels: Channels,
		format: SampleFormat,
		layout: SampleLayout,
	) -> Self {
		// Samples per channel. Planar data holds one plane per channel back
		// to back, so both layouts divide by channels × bytes per sample.
		let nb_samples = data.len() / (usize::from(channels.count()).max(1) * format.bytes());
		Self { data, sample_rate, channels, format, layout, nb_samples, pts: None }
	}

	/// Bytes of sample data (all channels, either layout).
	pub fn frame_size(&self) -> usize {
		self.nb_samples * self.channels.count() as usize * self.format.bytes()
	}

	pub fn is_planar(&self) -> bool {
		self.layout.is_planar()
	}
}
