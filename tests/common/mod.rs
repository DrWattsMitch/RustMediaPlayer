//! Helpers shared by integration tests.
#![allow(dead_code)]

use rmp::core::{Demuxer, Packet};
use rmp::io::MemoryIo;

pub fn fixture(path: &str) -> Vec<u8> {
	let full = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), path);
	std::fs::read(&full).unwrap_or_else(|e| panic!("missing fixture {full}: {e}"))
}

pub fn memory(data: Vec<u8>) -> Box<MemoryIo> {
	Box::new(MemoryIo::from_vec(data))
}

pub fn read_all(demuxer: &mut dyn Demuxer) -> Vec<Packet> {
	let mut packets = Vec::new();
	while let Some(packet) = demuxer.read().expect("read packet") {
		packets.push(packet);
	}
	packets
}

/// Small deterministic PRNG (xorshift) so robustness tests are reproducible
/// without extra dependencies.
pub struct Rng(u64);

impl Rng {
	pub fn new(seed: u64) -> Self {
		Self(seed.max(1))
	}

	pub fn next(&mut self) -> u64 {
		let mut x = self.0;
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		self.0 = x;
		x
	}

	pub fn below(&mut self, n: usize) -> usize {
		(self.next() % n.max(1) as u64) as usize
	}
}
