//! Residues (Vorbis I §8): the fine spectral detail, coded per partition
//! with a classification that selects up to eight cascaded VQ books.
//!
//! Type 0 interleaves each VQ vector across the partition, type 1 lays
//! vectors end to end, type 2 codes all channels of a submap as one
//! interleaved vector (each value cycling through the channels) with the
//! type 1 layout. libvorbis: `lib/res0.c`.

use super::bits::Bits;
use super::codebook::Codebook;

#[derive(Debug, Clone)]
pub struct Residue {
	kind: u16,
	begin: usize,
	end: usize,
	partition_size: usize,
	classifications: usize,
	classbook: usize,
	/// Per classification, the book of each of the eight passes.
	books: Vec<[Option<usize>; 8]>,
	/// `classifications ^ classbook.dimensions`: larger class words are
	/// invalid.
	class_words: usize,
}

impl Residue {
	/// Reads a residue configuration (§8.6.1).
	pub(super) fn read(b: &mut Bits, codebooks: &[Codebook]) -> Option<Self> {
		let kind = b.read(16)? as u16;
		if kind > 2 {
			return None;
		}
		let begin = b.read(24)? as usize;
		let end = b.read(24)? as usize;
		let partition_size = b.read(24)? as usize + 1;
		let classifications = b.read(6)? as usize + 1;
		let classbook = b.read(8)? as usize;
		let class_dims = codebooks.get(classbook)?.dimensions;
		let mut cascade = Vec::with_capacity(classifications);
		for _ in 0..classifications {
			let low = b.read(3)?;
			let high = if b.flag()? { b.read(5)? } else { 0 };
			cascade.push(high << 3 | low);
		}
		let mut books = Vec::with_capacity(classifications);
		for &bits in &cascade {
			let mut row = [None; 8];
			for (pass, slot) in row.iter_mut().enumerate() {
				if bits & (1 << pass) != 0 {
					let book = b.read(8)? as usize;
					// A residue book must map entries to vectors.
					if !codebooks.get(book)?.has_vq() {
						return None;
					}
					*slot = Some(book);
				}
			}
			books.push(row);
		}
		let class_words = (0..class_dims).try_fold(1usize, |acc, _| acc.checked_mul(classifications));
		Some(Self {
			kind,
			begin,
			end,
			partition_size,
			classifications,
			classbook,
			books,
			class_words: class_words.unwrap_or(usize::MAX),
		})
	}

	/// Decodes the residue of a submap's channels into `vectors` (each
	/// `n` values, zeroed by the caller). `decode[j]` is false for the
	/// channels whose floor is unused. An early end of packet leaves the
	/// remaining values at zero (§8.6.2).
	pub(super) fn decode(
		&self,
		b: &mut Bits,
		codebooks: &[Codebook],
		vectors: &mut [Vec<f32>],
		decode: &[bool],
		n: usize,
	) {
		if self.kind == 2 {
			if !decode.iter().any(|&d| d) {
				return;
			}
			let ch = vectors.len();
			let mut flat = vec![0f32; n * ch];
			self.decode_partitions(b, codebooks, std::slice::from_mut(&mut flat), &[true], n * ch);
			for (i, chunk) in flat.chunks_exact(ch).enumerate() {
				for (v, &x) in vectors.iter_mut().zip(chunk) {
					v[i] = x;
				}
			}
		} else {
			self.decode_partitions(b, codebooks, vectors, decode, n);
		}
	}

	fn decode_partitions(
		&self,
		b: &mut Bits,
		codebooks: &[Codebook],
		vectors: &mut [Vec<f32>],
		decode: &[bool],
		n: usize,
	) {
		let Some(classbook) = codebooks.get(self.classbook) else { return };
		let begin = self.begin.min(n);
		let end = self.end.min(n);
		let partitions = end.saturating_sub(begin) / self.partition_size;
		if partitions == 0 {
			return;
		}
		let per_word = classbook.dimensions;
		let ch = vectors.len();
		// Classification of each partition, per channel (rounded up to
		// whole class words).
		let words = partitions.div_ceil(per_word);
		let mut classes = vec![0u8; ch * words * per_word];
		for pass in 0..8 {
			let mut partition = 0;
			while partition < partitions {
				if pass == 0 {
					for j in (0..ch).filter(|&j| decode[j]) {
						let Some(word) = classbook.scalar(b) else { return };
						let mut word = word as usize;
						if word >= self.class_words {
							return;
						}
						for i in (0..per_word).rev() {
							classes[j * words * per_word + partition + i] = (word % self.classifications) as u8;
							word /= self.classifications;
						}
					}
				}
				for _ in 0..per_word {
					if partition >= partitions {
						break;
					}
					let offset = begin + partition * self.partition_size;
					for j in (0..ch).filter(|&j| decode[j]) {
						let class = usize::from(classes[j * words * per_word + partition]);
						let Some(book) = self.books[class][pass] else { continue };
						let Some(book) = codebooks.get(book) else { return };
						let target = &mut vectors[j][offset..offset + self.partition_size];
						let ok = if self.kind == 0 {
							decode_interleaved(b, book, target)
						} else {
							decode_sequential(b, book, target)
						};
						if !ok {
							return;
						}
					}
					partition += 1;
				}
			}
		}
	}
}

/// Type 0: vector `i` of the partition fills positions `i, i + step, ...`
/// (libvorbis `vorbis_book_decodevs_add`).
fn decode_interleaved(b: &mut Bits, book: &Codebook, out: &mut [f32]) -> bool {
	let dims = book.dimensions;
	let step = out.len() / dims;
	for i in 0..step {
		let Some(v) = book.vector(b) else { return false };
		for (j, &x) in v.iter().enumerate() {
			if let Some(o) = out.get_mut(i + j * step) {
				*o += x;
			}
		}
	}
	true
}

/// Types 1 and 2: vectors end to end (`vorbis_book_decodev_add`).
fn decode_sequential(b: &mut Bits, book: &Codebook, out: &mut [f32]) -> bool {
	let mut i = 0;
	while i < out.len() {
		let Some(v) = book.vector(b) else { return false };
		for &x in v {
			if i >= out.len() {
				break;
			}
			out[i] += x;
			i += 1;
		}
	}
	true
}
