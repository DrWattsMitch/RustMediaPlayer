//! Sample table (`stbl`) parsing: turns the run-length coded tables into a
//! flat per-sample index, like FFmpeg's `mov_build_index`.
//!
//! ```text
//! stsz/stz2  size of each sample
//! stco/co64  file offset of each chunk
//! stsc       how many samples each chunk holds (run-length by chunk)
//! stts       decode-time delta of each sample (run-length)
//! ctts       composition offset (pts - dts) of each sample (run-length)
//! stss       which samples are sync samples (keyframes); absent = all
//! ```
//!
//! Every count read from the file is validated against the size of the box
//! that holds it (and against the file size where that bounds it), so a
//! corrupt or hostile file cannot make the index allocate unbounded memory.

use super::boxes::{find, full_box};
use crate::io::ByteCursor;
use crate::message::{Message, Result};

/// Hard cap on samples per track (≈ 9 hours of 60 fps video plus margin).
pub const MAX_SAMPLES: usize = 1 << 22;
/// Hard cap on a single sample's size.
pub const MAX_SAMPLE_SIZE: u32 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
	pub offset: u64,
	/// Decode timestamp in track timescale units.
	pub dts: u64,
	pub size: u32,
	/// Composition offset: pts = dts + cts.
	pub cts: i32,
	/// Decode duration (the `stts` delta) in track timescale units.
	pub duration: u32,
	pub keyframe: bool,
}

const TRUNCATED: Message = Message::Container("truncated mp4 sample table");

/// Reads a full box with an entry count followed by fixed-size entries,
/// checking the count against the payload size.
fn entries(body: &[u8], entry_size: usize) -> Result<(u8, ByteCursor<'_>, usize)> {
	let (version, _, payload) = full_box(body)?;
	let mut c = ByteCursor::new(payload);
	let count = c.read_u32()? as usize;
	if count.checked_mul(entry_size).is_none_or(|bytes| bytes > c.remaining()) {
		return Err(TRUNCATED);
	}
	Ok((version, c, count))
}

/// Sizes from `stsz` (constant or per-sample) or `stz2` (compact).
enum Sizes<'a> {
	Constant(u32, usize),
	Table { data: &'a [u8], field_bits: u8, count: usize },
}

impl Sizes<'_> {
	fn count(&self) -> usize {
		match self {
			Sizes::Constant(_, count) | Sizes::Table { count, .. } => *count,
		}
	}

	fn get(&self, i: usize) -> Option<u32> {
		match self {
			Sizes::Constant(size, _) => Some(*size),
			Sizes::Table { data, field_bits, .. } => match field_bits {
				4 => {
					data.get(i / 2).map(|&b| u32::from(if i.is_multiple_of(2) { b >> 4 } else { b & 0x0F }))
				}
				8 => data.get(i).map(|&b| u32::from(b)),
				16 => data.get(2 * i..2 * i + 2).map(|b| u32::from(u16::from_be_bytes([b[0], b[1]]))),
				_ => data.get(4 * i..4 * i + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])),
			},
		}
	}
}

fn parse_sizes(stbl: &[u8], file_size: Option<u64>) -> Result<Sizes<'_>> {
	if let Some(stsz) = find(stbl, b"stsz") {
		let (_, _, payload) = full_box(stsz)?;
		let mut c = ByteCursor::new(payload);
		let sample_size = c.read_u32()?;
		let count = c.read_u32()? as usize;
		if sample_size != 0 {
			// No per-sample table bounds the count, so the file size must:
			// every sample occupies `sample_size` bytes of it.
			let max = file_size.map_or(MAX_SAMPLES as u64, |s| s / u64::from(sample_size));
			if count as u64 > max {
				return Err(Message::Container("mp4 sample count exceeds file size"));
			}
			return Ok(Sizes::Constant(sample_size, count));
		}
		if count.checked_mul(4).is_none_or(|bytes| bytes > c.remaining()) {
			return Err(TRUNCATED);
		}
		return Ok(Sizes::Table { data: c.rest(), field_bits: 32, count });
	}
	if let Some(stz2) = find(stbl, b"stz2") {
		let (_, _, payload) = full_box(stz2)?;
		let mut c = ByteCursor::new(payload);
		c.skip(3)?;
		let field_bits = c.read_u8()?;
		if !matches!(field_bits, 4 | 8 | 16) {
			return Err(Message::Container("invalid mp4 stz2 field size"));
		}
		let count = c.read_u32()? as usize;
		if count.saturating_mul(usize::from(field_bits)).div_ceil(8) > c.remaining() {
			return Err(TRUNCATED);
		}
		return Ok(Sizes::Table { data: c.rest(), field_bits, count });
	}
	Err(Message::Container("mp4 track has no sample size table"))
}

fn parse_chunk_offsets(stbl: &[u8]) -> Result<Vec<u64>> {
	if let Some(stco) = find(stbl, b"stco") {
		let (_, mut c, count) = entries(stco, 4)?;
		return (0..count).map(|_| c.read_u32().map(u64::from)).collect();
	}
	if let Some(co64) = find(stbl, b"co64") {
		let (_, mut c, count) = entries(co64, 8)?;
		return (0..count).map(|_| c.read_u64()).collect();
	}
	Err(Message::Container("mp4 track has no chunk offset table"))
}

/// Run-length table: (repeat count, value).
fn run_table(body: Option<&[u8]>) -> Result<Vec<(u32, i64)>> {
	let Some(body) = body else { return Ok(Vec::new()) };
	let (version, mut c, count) = entries(body, 8)?;
	(0..count)
		.map(|_| {
			let n = c.read_u32()?;
			let raw = c.read_u32()?;
			// ctts v1 offsets are signed; v0 ones are too in practice
			// (FFmpeg treats both as signed), deltas are unsigned.
			let value = if version == 1 { i64::from(raw as i32) } else { i64::from(raw) };
			Ok((n, value))
		})
		.collect()
}

/// Builds the per-sample index for one track.
pub fn build_index(stbl: &[u8], file_size: Option<u64>) -> Result<Vec<Sample>> {
	let sizes = parse_sizes(stbl, file_size)?;
	let total = sizes.count();
	if total > MAX_SAMPLES {
		return Err(Message::Container("mp4 track has too many samples"));
	}
	let chunk_offsets = parse_chunk_offsets(stbl)?;

	let stsc: Vec<(u32, u32)> = match find(stbl, b"stsc") {
		Some(body) => {
			let (_, mut c, count) = entries(body, 12)?;
			(0..count)
				.map(|_| {
					let first_chunk = c.read_u32()?;
					let per_chunk = c.read_u32()?;
					c.skip(4)?; // sample description index
					Ok((first_chunk, per_chunk))
				})
				.collect::<Result<_>>()?
		}
		None => Vec::new(),
	};

	// 1. Offsets and sizes, chunk by chunk.
	let mut samples = Vec::with_capacity(total.min(1 << 16));
	let mut run = 0usize;
	'chunks: for (chunk_index, &chunk_offset) in chunk_offsets.iter().enumerate() {
		let chunk_number = chunk_index as u64 + 1;
		while run + 1 < stsc.len() && u64::from(stsc[run + 1].0) <= chunk_number {
			run += 1;
		}
		let Some(&(_, per_chunk)) = stsc.get(run) else { break };
		let mut offset = chunk_offset;
		for _ in 0..per_chunk {
			if samples.len() == total {
				break 'chunks;
			}
			let size = sizes.get(samples.len()).ok_or(TRUNCATED)?;
			let end = offset.checked_add(u64::from(size));
			// Stop at samples that lie outside the file (truncated download)
			// or are implausibly large (corruption).
			if size > MAX_SAMPLE_SIZE || end.is_none_or(|e| file_size.is_some_and(|s| e > s)) {
				break 'chunks;
			}
			samples.push(Sample { offset, dts: 0, size, cts: 0, duration: 0, keyframe: true });
			offset = end.unwrap_or(u64::MAX);
		}
	}

	// 2. Decode timestamps.
	let stts = run_table(find(stbl, b"stts"))?;
	let mut dts = 0u64;
	let mut last_delta = 0u64;
	let mut deltas = stts.iter().flat_map(|&(n, d)| std::iter::repeat_n(d.max(0) as u64, n as usize));
	for sample in samples.iter_mut() {
		sample.dts = dts;
		let delta = deltas.next().unwrap_or(last_delta);
		last_delta = delta;
		sample.duration = u32::try_from(delta).unwrap_or(u32::MAX);
		dts = dts.saturating_add(delta);
	}

	// 3. Composition offsets.
	let ctts = run_table(find(stbl, b"ctts"))?;
	let mut offsets = ctts.iter().flat_map(|&(n, o)| std::iter::repeat_n(o, n as usize));
	for sample in samples.iter_mut() {
		let Some(offset) = offsets.next() else { break };
		sample.cts = offset.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
	}

	// 4. Sync samples.
	if let Some(stss) = find(stbl, b"stss") {
		let (_, mut c, count) = entries(stss, 4)?;
		samples.iter_mut().for_each(|s| s.keyframe = false);
		for _ in 0..count {
			let number = c.read_u32()? as usize;
			if let Some(sample) = number.checked_sub(1).and_then(|i| samples.get_mut(i)) {
				sample.keyframe = true;
			}
		}
	}

	Ok(samples)
}
