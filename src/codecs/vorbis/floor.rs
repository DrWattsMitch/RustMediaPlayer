//! Floors (Vorbis I §6, §7): the spectral envelope each channel's residue
//! is multiplied by.
//!
//! - Floor 1 (every encoder since 2002): a piecewise-linear curve in the
//!   dB domain through up to 65 points whose amplitudes are coded as
//!   corrections to the line through their neighbours.
//! - Floor 0 (pre-1.0 streams): a line spectral pair (LSP) curve on a
//!   Bark-scale frequency map.
//!
//! libvorbis: `lib/floor1.c`, `lib/floor0.c`.

use super::bits::Bits;
use super::codebook::Codebook;
use super::ilog;

/// §10.1 `floor1_inverse_dB_table` (libvorbis `FLOOR1_fromdB_LOOKUP`).
#[rustfmt::skip]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static INVERSE_DB: [f32; 256] = [
	1.0649863e-07, 1.1341951e-07, 1.2079015e-07, 1.2863978e-07, 1.3699951e-07, 1.4590251e-07,
	1.5538408e-07, 1.6548181e-07, 1.7623575e-07, 1.8768855e-07, 1.9988561e-07, 2.128753e-07,
	2.2670913e-07, 2.4144197e-07, 2.5713223e-07, 2.7384213e-07, 2.9163793e-07, 3.1059021e-07,
	3.3077411e-07, 3.5226968e-07, 3.7516214e-07, 3.9954229e-07, 4.2550680e-07, 4.5315863e-07,
	4.8260743e-07, 5.1396998e-07, 5.4737065e-07, 5.8294187e-07, 6.2082472e-07, 6.6116941e-07,
	7.0413592e-07, 7.4989464e-07, 7.9862701e-07, 8.5052630e-07, 9.0579828e-07, 9.6466216e-07,
	1.0273513e-06, 1.0941144e-06, 1.1652161e-06, 1.2409384e-06, 1.3215816e-06, 1.4074654e-06,
	1.4989305e-06, 1.5963394e-06, 1.7000785e-06, 1.8105592e-06, 1.9282195e-06, 2.0535261e-06,
	2.1869758e-06, 2.3290978e-06, 2.4804557e-06, 2.6416497e-06, 2.8133190e-06, 2.9961443e-06,
	3.1908506e-06, 3.3982101e-06, 3.6190449e-06, 3.8542308e-06, 4.1047004e-06, 4.3714470e-06,
	4.6555282e-06, 4.9580707e-06, 5.2802740e-06, 5.6234160e-06, 5.9888572e-06, 6.3780469e-06,
	6.7925283e-06, 7.2339451e-06, 7.7040476e-06, 8.2047000e-06, 8.7378876e-06, 9.3057248e-06,
	9.9104632e-06, 1.0554501e-05, 1.1240392e-05, 1.1970856e-05, 1.2748789e-05, 1.3577278e-05,
	1.4459606e-05, 1.5399272e-05, 1.6400004e-05, 1.7465768e-05, 1.8600792e-05, 1.9809576e-05,
	2.1096914e-05, 2.2467911e-05, 2.3928002e-05, 2.5482978e-05, 2.7139006e-05, 2.8902651e-05,
	3.0780908e-05, 3.2781225e-05, 3.4911534e-05, 3.7180282e-05, 3.9596466e-05, 4.2169667e-05,
	4.4910090e-05, 4.7828601e-05, 5.0936773e-05, 5.4246931e-05, 5.7772202e-05, 6.1526565e-05,
	6.5524908e-05, 6.9783085e-05, 7.4317983e-05, 7.9147585e-05, 8.4291040e-05, 8.9768747e-05,
	9.5602426e-05, 0.00010181521, 0.00010843174, 0.00011547824, 0.00012298267, 0.00013097477,
	0.00013948625, 0.00014855085, 0.00015820453, 0.00016848555, 0.00017943469, 0.00019109536,
	0.00020351382, 0.00021673929, 0.00023082423, 0.00024582449, 0.00026179955, 0.00027881276,
	0.00029693158, 0.00031622787, 0.00033677814, 0.00035866388, 0.00038197188, 0.00040679456,
	0.00043323036, 0.00046138411, 0.00049136745, 0.00052329927, 0.00055730621, 0.00059352311,
	0.00063209358, 0.00067317058, 0.00071691700, 0.00076350630, 0.00081312324, 0.00086596457,
	0.00092223983, 0.00098217216, 0.0010459992, 0.0011139742, 0.0011863665, 0.0012634633,
	0.0013455702, 0.0014330129, 0.0015261382, 0.0016253153, 0.0017309374, 0.0018434235, 0.0019632195,
	0.0020908006, 0.0022266726, 0.0023713743, 0.0025254795, 0.0026895994, 0.0028643847, 0.0030505286,
	0.0032487691, 0.0034598925, 0.0036847358, 0.0039241906, 0.0041792066, 0.0044507950, 0.0047400328,
	0.0050480668, 0.0053761186, 0.0057254891, 0.0060975636, 0.0064938176, 0.0069158225, 0.0073652516,
	0.0078438871, 0.0083536271, 0.0088964928, 0.009474637, 0.010090352, 0.010746080, 0.011444421,
	0.012188144, 0.012980198, 0.013823725, 0.014722068, 0.015678791, 0.016697687, 0.017782797,
	0.018938423, 0.020169149, 0.021479854, 0.022875735, 0.024362330, 0.025945531, 0.027631618,
	0.029427276, 0.031339626, 0.033376252, 0.035545228, 0.037855157, 0.040315199, 0.042935108,
	0.045725273, 0.048696758, 0.051861348, 0.055231591, 0.058820850, 0.062643361, 0.066714279,
	0.071049749, 0.075666962, 0.080584227, 0.085821044, 0.091398179, 0.097337747, 0.10366330,
	0.11039993, 0.11757434, 0.12521498, 0.13335215, 0.14201813, 0.15124727, 0.16107617, 0.17154380,
	0.18269168, 0.19456402, 0.20720788, 0.22067342, 0.23501402, 0.25028656, 0.26655159, 0.28387361,
	0.30232132, 0.32196786, 0.34289114, 0.36517414, 0.38890521, 0.41417847, 0.44109412, 0.46975890,
	0.50028648, 0.53279791, 0.56742212, 0.60429640, 0.64356699, 0.68538959, 0.72993007, 0.77736504,
	0.82788260, 0.88168307, 0.9389798, 1.0,];

/// Most points of a floor 1 (2 + 31 partitions × at most 8, capped by
/// the specification at 65).
const MAX_POSTS: usize = 65;

#[derive(Debug, Clone)]
pub enum Floor {
	Zero(Floor0),
	One(Floor1),
}

#[derive(Debug, Clone)]
pub struct Floor0 {
	order: usize,
	bark_map_size: u32,
	amplitude_bits: u32,
	amplitude_offset: u32,
	books: Vec<usize>,
	/// Bark map for the short and long block (`n / 2 + 1` entries).
	map: [Vec<i32>; 2],
}

#[derive(Debug, Clone)]
pub struct Floor1 {
	partition_class: Vec<usize>,
	class_dimensions: Vec<usize>,
	class_subclasses: Vec<u32>,
	class_masterbook: Vec<usize>,
	/// Per class, `1 << subclasses` books (`None`: value 0).
	subclass_books: Vec<Vec<Option<usize>>>,
	multiplier: i32,
	/// X positions, in coding order (`[0, 1 << rangebits, ...]`).
	x: Vec<i32>,
	/// Low and high neighbour (§9.2.4, §9.2.5) of each point from 2 on.
	low: Vec<usize>,
	high: Vec<usize>,
	/// Point indices sorted by X.
	sorted: Vec<usize>,
	/// Amplitude range (256, 128, 86 or 64).
	range: i32,
}

/// One channel's decoded floor, kept until the residue is ready.
#[derive(Debug, Clone)]
pub enum FloorData {
	/// Floor 0 amplitude and LSP coefficients.
	Zero { amplitude: u32, coefficients: Vec<f32> },
	/// Floor 1 final Y values, and whether each point is used (step 2).
	One { y: Vec<i32>, used: Vec<bool> },
}

fn book(books: &[Codebook], index: usize) -> Option<&Codebook> {
	books.get(index)
}

impl Floor {
	/// Reads a floor configuration (§6.2.1, §7.2.2). `blocksize` is the
	/// short and long block size, for the floor 0 Bark maps.
	pub(super) fn read(b: &mut Bits, books: &[Codebook], blocksize: [usize; 2]) -> Option<Self> {
		match b.read(16)? {
			0 => {
				let order = b.read(8)? as usize;
				let rate = b.read(16)?;
				let bark_map_size = b.read(16)?;
				let amplitude_bits = b.read(6)?;
				let amplitude_offset = b.read(8)?;
				let count = b.read(4)? as usize + 1;
				let floor_books: Vec<usize> =
					(0..count).map(|_| b.read(8).map(|v| v as usize)).collect::<Option<_>>()?;
				if order == 0 || rate == 0 || bark_map_size == 0 || amplitude_bits == 0 {
					return None;
				}
				if floor_books.iter().any(|&i| books.get(i).is_none_or(|bk| !bk.has_vq())) {
					return None;
				}
				let map = blocksize.map(|size| bark_map(size / 2, rate, bark_map_size));
				Some(Floor::Zero(Floor0 {
					order,
					bark_map_size,
					amplitude_bits,
					amplitude_offset,
					books: floor_books,
					map,
				}))
			}
			1 => Floor1::read(b, books).map(Floor::One),
			_ => None,
		}
	}

	/// Decodes a channel's floor (§6.2.2, §7.2.3); `None` when the channel
	/// is unused in this packet (or the packet ended).
	pub(super) fn decode(&self, b: &mut Bits, books: &[Codebook]) -> Option<FloorData> {
		match self {
			Floor::Zero(f) => f.decode(b, books),
			Floor::One(f) => f.decode(b, books),
		}
	}

	/// Multiplies `out` (`n` = half the block size values) by the curve.
	pub(super) fn apply(&self, data: &FloorData, long: bool, out: &mut [f32]) {
		match (self, data) {
			(Floor::Zero(f), FloorData::Zero { amplitude, coefficients }) => {
				f.apply(*amplitude, coefficients, long, out)
			}
			(Floor::One(f), FloorData::One { y, used }) => f.apply(y, used, out),
			_ => out.fill(0.0),
		}
	}
}

fn bark(x: f64) -> f64 {
	13.1 * (0.00074 * x).atan() + 2.24 * (0.000_000_018_5 * x * x).atan() + 0.0001 * x
}

/// §6.2.3 Bark map for `n` output values, terminated by `-1`.
fn bark_map(n: usize, rate: u32, size: u32) -> Vec<i32> {
	let rate = f64::from(rate);
	let scale = f64::from(size) / bark(0.5 * rate);
	let mut map: Vec<i32> = (0..n)
		.map(|i| {
			let v = (bark(rate * i as f64 / (2.0 * n as f64)) * scale).floor();
			(v as i32).min(size as i32 - 1)
		})
		.collect();
	map.push(-1);
	map
}

impl Floor0 {
	fn decode(&self, b: &mut Bits, books: &[Codebook]) -> Option<FloorData> {
		let amplitude = b.read(self.amplitude_bits)?;
		if amplitude == 0 {
			return None;
		}
		let index = b.read(ilog(self.books.len() as u32))? as usize;
		let codebook = book(books, *self.books.get(index)?)?;
		let mut coefficients = Vec::with_capacity(self.order + codebook.dimensions);
		let mut last = 0f32;
		while coefficients.len() < self.order {
			let v = codebook.vector(b)?;
			coefficients.extend(v.iter().map(|&x| x + last));
			last = *coefficients.last()?;
		}
		coefficients.truncate(self.order);
		Some(FloorData::Zero { amplitude, coefficients })
	}

	fn apply(&self, amplitude: u32, coefficients: &[f32], long: bool, out: &mut [f32]) {
		let map = &self.map[usize::from(long)];
		let n = out.len().min(map.len() - 1);
		let cos: Vec<f64> = coefficients.iter().map(|&c| f64::from(c).cos()).collect();
		let max_amplitude = f64::from((1u32 << self.amplitude_bits) - 1);
		let offset = f64::from(self.amplitude_offset);
		let mut i = 0;
		while i < n {
			let w = std::f64::consts::PI * f64::from(map[i]) / f64::from(self.bark_map_size);
			let cw = w.cos();
			let product = |start: usize| {
				cos.iter().skip(start).step_by(2).fold(1.0, |acc, &c| acc * 4.0 * (c - cw) * (c - cw))
			};
			let (p, q) = if self.order % 2 == 1 {
				((1.0 - cw * cw) * product(1), 0.25 * product(0))
			} else {
				((1.0 - cw) / 2.0 * product(1), (1.0 + cw) / 2.0 * product(0))
			};
			let db = f64::from(amplitude) * offset / (max_amplitude * (p + q).sqrt()) - offset;
			let value = (0.115_129_25 * db).exp() as f32;
			let bucket = map[i];
			while i < n && map[i] == bucket {
				out[i] *= value;
				i += 1;
			}
		}
		out[n..].fill(0.0);
	}
}

impl Floor1 {
	fn read(b: &mut Bits, books: &[Codebook]) -> Option<Self> {
		let partitions = b.read(5)? as usize;
		let partition_class: Vec<usize> =
			(0..partitions).map(|_| b.read(4).map(|v| v as usize)).collect::<Option<_>>()?;
		let classes = partition_class.iter().max().map_or(0, |&m| m + 1);
		let (mut class_dimensions, mut class_subclasses, mut class_masterbook, mut subclass_books) =
			(Vec::new(), Vec::new(), Vec::new(), Vec::new());
		for _ in 0..classes {
			class_dimensions.push(b.read(3)? as usize + 1);
			let subclasses = b.read(2)?;
			class_subclasses.push(subclasses);
			let master = if subclasses > 0 { b.read(8)? as usize } else { 0 };
			if subclasses > 0 && master >= books.len() {
				return None;
			}
			class_masterbook.push(master);
			let mut subs = Vec::new();
			for _ in 0..1 << subclasses {
				let v = b.read(8)?;
				let book = v.checked_sub(1).map(|v| v as usize);
				if book.is_some_and(|i| i >= books.len()) {
					return None;
				}
				subs.push(book);
			}
			subclass_books.push(subs);
		}
		let multiplier = b.read(2)? as i32 + 1;
		let range_bits = b.read(4)?;
		let mut x = vec![0i32, 1 << range_bits];
		for &class in &partition_class {
			for _ in 0..class_dimensions[class] {
				if x.len() >= MAX_POSTS {
					return None;
				}
				x.push(b.read(range_bits)? as i32);
			}
		}
		let mut sorted: Vec<usize> = (0..x.len()).collect();
		sorted.sort_by_key(|&i| x[i]);
		if sorted.windows(2).any(|w| x[w[0]] == x[w[1]]) {
			return None;
		}
		let (mut low, mut high) = (vec![0; x.len()], vec![1; x.len()]);
		for i in 2..x.len() {
			let (mut lx, mut hx) = (-1, i32::MAX);
			for j in 0..i {
				if x[j] < x[i] && x[j] > lx {
					lx = x[j];
					low[i] = j;
				}
				if x[j] > x[i] && x[j] < hx {
					hx = x[j];
					high[i] = j;
				}
			}
		}
		let range = [256, 128, 86, 64][(multiplier - 1) as usize];
		Some(Self {
			partition_class,
			class_dimensions,
			class_subclasses,
			class_masterbook,
			subclass_books,
			multiplier,
			x,
			low,
			high,
			sorted,
			range,
		})
	}

	fn decode(&self, b: &mut Bits, books: &[Codebook]) -> Option<FloorData> {
		if !b.flag()? {
			return None;
		}
		let bits = ilog(self.range as u32 - 1);
		let mut y = Vec::with_capacity(self.x.len());
		y.push(b.read(bits)? as i32);
		y.push(b.read(bits)? as i32);
		for &class in &self.partition_class {
			let cdim = self.class_dimensions[class];
			let cbits = self.class_subclasses[class];
			let csub = (1u32 << cbits) - 1;
			let mut cval =
				if cbits > 0 { book(books, self.class_masterbook[class])?.scalar(b)? } else { 0 };
			for _ in 0..cdim {
				let sub = self.subclass_books[class][(cval & csub) as usize];
				cval >>= cbits;
				y.push(match sub {
					Some(i) => book(books, i)?.scalar(b)? as i32,
					None => 0,
				});
			}
		}

		// Step 1: amplitude synthesis (§7.2.4).
		let n = self.x.len();
		let mut used = vec![false; n];
		used[0] = true;
		used[1] = true;
		let mut fin = vec![0i32; n];
		fin[0] = y[0];
		fin[1] = y[1];
		for i in 2..n {
			let (lo, hi) = (self.low[i], self.high[i]);
			let predicted = render_point(self.x[lo], fin[lo], self.x[hi], fin[hi], self.x[i]);
			let val = y[i];
			let high_room = self.range - predicted;
			let low_room = predicted;
			let room = high_room.min(low_room) * 2;
			if val != 0 {
				used[lo] = true;
				used[hi] = true;
				used[i] = true;
				fin[i] = if val >= room {
					if high_room > low_room {
						val - low_room + predicted
					} else {
						predicted - val + high_room - 1
					}
				} else if val & 1 == 1 {
					predicted - (val + 1) / 2
				} else {
					predicted + val / 2
				};
			} else {
				fin[i] = predicted;
			}
		}
		Some(FloorData::One { y: fin, used })
	}

	/// Step 2: curve synthesis (§7.2.4), multiplied into `out`.
	fn apply(&self, y: &[i32], used: &[bool], out: &mut [f32]) {
		let n = out.len() as i32;
		let mut lx = 0i32;
		let mut ly = (y[0] * self.multiplier).clamp(0, 255);
		for &i in self.sorted.iter().skip(1) {
			if !used[i] {
				continue;
			}
			let hx = self.x[i];
			let hy = (y[i] * self.multiplier).clamp(0, 255);
			render_line(lx, ly, hx, hy, out);
			lx = hx;
			ly = hy;
		}
		if lx < n {
			let v = INVERSE_DB[ly as usize];
			out[lx as usize..].iter_mut().for_each(|o| *o *= v);
		}
	}
}

/// §9.2.6 `render_point`.
fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
	let dy = y1 - y0;
	let adx = x1 - x0;
	let off = dy.abs() * (x - x0) / adx;
	if dy < 0 { y0 - off } else { y0 + off }
}

/// §9.2.7 `render_line`, multiplying `out[x0..min(x1, n)]` by the dB
/// curve (libvorbis multiplies in place the same way).
fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, out: &mut [f32]) {
	let dy = y1 - y0;
	let adx = x1 - x0;
	if adx <= 0 {
		return;
	}
	let base = dy / adx;
	let sy = if dy < 0 { base - 1 } else { base + 1 };
	let ady = dy.abs() - base.abs() * adx;
	let end = x1.min(out.len() as i32);
	let (mut x, mut y, mut err) = (x0, y0, 0);
	if x < end {
		out[x as usize] *= INVERSE_DB[y.clamp(0, 255) as usize];
	}
	x += 1;
	while x < end {
		err += ady;
		if err >= adx {
			err -= adx;
			y += sy;
		} else {
			y += base;
		}
		out[x as usize] *= INVERSE_DB[y.clamp(0, 255) as usize];
		x += 1;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn render_line_matches_bresenham() {
		let mut out = vec![1.0f32; 8];
		render_line(0, 255, 4, 251, &mut out);
		let expect: Vec<f32> = [255, 254, 253, 252].iter().map(|&i| INVERSE_DB[i]).collect();
		assert_eq!(&out[..4], &expect[..]);
		assert_eq!(&out[4..], &[1.0; 4]);
		assert_eq!(render_point(0, 10, 10, 20, 5), 15);
		assert_eq!(render_point(0, 20, 10, 10, 3), 17);
	}
}
