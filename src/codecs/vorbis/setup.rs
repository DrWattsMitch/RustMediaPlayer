//! Vorbis header packets (Vorbis I §4.2): identification (`0x01`),
//! comment (`0x03`, parsed by the containers) and setup (`0x05`).

use super::bits::Bits;
use super::codebook::Codebook;
use super::floor::Floor;
use super::ilog;
use super::residue::Residue;

/// The identification header (§4.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identification {
	pub channels: u8,
	pub sample_rate: u32,
	pub bitrate_nominal: i32,
	/// Short and long block sizes.
	pub blocksize: [usize; 2],
}

impl Identification {
	pub fn parse(packet: &[u8]) -> Option<Self> {
		if packet.len() < 30 || packet.get(..7) != Some(b"\x01vorbis") {
			return None;
		}
		let version = u32::from_le_bytes(packet[7..11].try_into().ok()?);
		let channels = packet[11];
		let sample_rate = u32::from_le_bytes(packet[12..16].try_into().ok()?);
		let bitrate_nominal = i32::from_le_bytes(packet[20..24].try_into().ok()?);
		let sizes = packet[28];
		let blocksize = [1usize << (sizes & 0x0F), 1usize << (sizes >> 4)];
		let framing = packet[29] & 1 == 1;
		let valid_size = |s: usize| (64..=8192).contains(&s);
		if version != 0
			|| channels == 0
			|| sample_rate == 0
			|| !framing
			|| !valid_size(blocksize[0])
			|| !valid_size(blocksize[1])
			|| blocksize[0] > blocksize[1]
		{
			return None;
		}
		Some(Self { channels, sample_rate, bitrate_nominal, blocksize })
	}
}

/// A mapping (§4.2.4 step 5): channel coupling and which floor / residue
/// each submap uses.
#[derive(Debug, Clone)]
pub struct Mapping {
	/// `(magnitude, angle)` channel pairs, applied in reverse order.
	pub coupling: Vec<(usize, usize)>,
	/// Submap of each channel.
	pub mux: Vec<usize>,
	pub submap_floor: Vec<usize>,
	pub submap_residue: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct Mode {
	pub long: bool,
	pub mapping: usize,
}

/// The decoded setup header.
#[derive(Debug, Clone)]
pub struct Setup {
	pub codebooks: Vec<Codebook>,
	pub floors: Vec<Floor>,
	pub residues: Vec<Residue>,
	pub mappings: Vec<Mapping>,
	pub modes: Vec<Mode>,
}

impl Setup {
	pub fn parse(packet: &[u8], id: &Identification) -> Option<Self> {
		if packet.get(..7) != Some(b"\x05vorbis") {
			return None;
		}
		let mut b = Bits::new(&packet[7..]);
		let channels = usize::from(id.channels);

		let count = b.read(8)? as usize + 1;
		let codebooks: Vec<Codebook> =
			(0..count).map(|_| Codebook::read(&mut b)).collect::<Option<_>>()?;

		// Time domain transforms: placeholders, all zero.
		let count = b.read(6)? + 1;
		for _ in 0..count {
			if b.read(16)? != 0 {
				return None;
			}
		}

		let count = b.read(6)? + 1;
		let floors: Vec<Floor> =
			(0..count).map(|_| Floor::read(&mut b, &codebooks, id.blocksize)).collect::<Option<_>>()?;

		let count = b.read(6)? + 1;
		let residues: Vec<Residue> =
			(0..count).map(|_| Residue::read(&mut b, &codebooks)).collect::<Option<_>>()?;

		let count = b.read(6)? + 1;
		let mut mappings = Vec::with_capacity(count as usize);
		for _ in 0..count {
			mappings.push(read_mapping(&mut b, channels, floors.len(), residues.len())?);
		}

		let count = b.read(6)? as usize + 1;
		let mut modes = Vec::with_capacity(count);
		for _ in 0..count {
			let long = b.flag()?;
			let window = b.read(16)?;
			let transform = b.read(16)?;
			let mapping = b.read(8)? as usize;
			if window != 0 || transform != 0 || mapping >= mappings.len() {
				return None;
			}
			modes.push(Mode { long, mapping });
		}
		if !b.flag()? {
			return None;
		}
		Some(Self { codebooks, floors, residues, mappings, modes })
	}
}

fn read_mapping(b: &mut Bits, channels: usize, floors: usize, residues: usize) -> Option<Mapping> {
	if b.read(16)? != 0 {
		return None;
	}
	let submaps = if b.flag()? { b.read(4)? as usize + 1 } else { 1 };
	let mut coupling = Vec::new();
	if b.flag()? {
		let steps = b.read(8)? as usize + 1;
		let bits = ilog(channels as u32 - 1);
		for _ in 0..steps {
			let magnitude = b.read(bits)? as usize;
			let angle = b.read(bits)? as usize;
			if magnitude == angle || magnitude >= channels || angle >= channels {
				return None;
			}
			coupling.push((magnitude, angle));
		}
	}
	if b.read(2)? != 0 {
		return None;
	}
	let mut mux = vec![0usize; channels];
	if submaps > 1 {
		for m in mux.iter_mut() {
			*m = b.read(4)? as usize;
			if *m >= submaps {
				return None;
			}
		}
	}
	let (mut submap_floor, mut submap_residue) = (Vec::new(), Vec::new());
	for _ in 0..submaps {
		b.read(8)?;
		let floor = b.read(8)? as usize;
		let residue = b.read(8)? as usize;
		if floor >= floors || residue >= residues {
			return None;
		}
		submap_floor.push(floor);
		submap_residue.push(residue);
	}
	Some(Mapping { coupling, mux, submap_floor, submap_residue })
}
