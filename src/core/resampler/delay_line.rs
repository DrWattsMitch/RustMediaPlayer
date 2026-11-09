//! Input history of one channel for the resampler.

#[derive(Debug, Clone, Default)]
pub struct DelayLine {
	buffer: Vec<f64>,
	/// Absolute input index of `buffer[0]` (negative while the leading
	/// zeros that precede the first sample are still buffered).
	position: i64,
}

impl DelayLine {
	/// Starts with `zeros` silent samples before input index 0.
	pub fn new(zeros: usize) -> Self {
		Self { buffer: vec![0.0; zeros], position: -(zeros as i64) }
	}

	pub fn push(&mut self, samples: &[f64]) {
		self.buffer.extend_from_slice(samples);
	}

	/// One past the last buffered absolute index.
	pub fn end(&self) -> i64 {
		self.position + self.buffer.len() as i64
	}

	/// `len` samples starting at absolute index `from` (must be buffered).
	pub fn window(&self, from: i64, len: usize) -> Option<&[f64]> {
		let start = usize::try_from(from - self.position).ok()?;
		self.buffer.get(start..start + len)
	}

	/// Drops samples before absolute index `before`.
	pub fn discard(&mut self, before: i64) {
		let n = usize::try_from(before - self.position).unwrap_or(0).min(self.buffer.len());
		if n > 0 {
			self.buffer.drain(..n);
			self.position += n as i64;
		}
	}
}
