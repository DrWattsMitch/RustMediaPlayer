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

/// Payload of a WAV file's `data` chunk.
pub fn wav_data(wav: &[u8]) -> &[u8] {
	let mut at = 12;
	while at + 8 <= wav.len() {
		let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().unwrap()) as usize;
		if &wav[at..at + 4] == b"data" {
			return &wav[at + 8..(at + 8 + size).min(wav.len())];
		}
		at += 8 + size + (size & 1);
	}
	panic!("no data chunk")
}

/// Decodes every packet of the first stream with the resolver's decoder and
/// returns the concatenated frame bytes.
pub fn decode_all(demuxer: &mut dyn Demuxer) -> Vec<u8> {
	use rmp::core::{Frame, Resolver};
	let stream = demuxer.streams().streams[0].clone();
	let decoder = Resolver::new().decoder_for(&stream).expect("decoder");
	let mut state = decoder.create_state(&stream.parameters);
	let mut out = Vec::new();
	while let Some(packet) = demuxer.read().expect("read packet") {
		for frame in decoder.decode(&mut state, packet).expect("decode").value {
			match frame {
				Frame::Audio(a) => out.extend_from_slice(&a.data),
				Frame::Video(v) => out.extend_from_slice(&v.data),
			}
		}
	}
	out
}
