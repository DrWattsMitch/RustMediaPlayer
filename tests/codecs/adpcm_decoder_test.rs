//! IMA and MS ADPCM against FFmpeg reference decodes (bit-exact).

use crate::common::{decode_all, fixture, memory};
use rmp::codecs::adpcm::{AdpcmCodec, decoder::decode_block};
use rmp::container::riff::demuxer::RiffDemuxer;
use rmp::core::Demuxer;

const CASES: [(&str, &str, u8, u32); 3] = [
	("wav/adpcm/ima_stereo_22k", "adpcm_ima_wav", 2, 22_050),
	("wav/adpcm/ms_mono_8k", "adpcm_ms", 1, 8_000),
	("wav/adpcm/ms_stereo_11k", "adpcm_ms", 2, 11_025),
];

#[test]
fn decodes_bit_exact_with_ffmpeg() {
	for (path, codec, channels, rate) in CASES {
		let mut demuxer = RiffDemuxer::new(memory(fixture(&format!("{path}.wav")))).unwrap();
		let stream = demuxer.streams().streams[0].clone();
		assert_eq!(stream.parameters.codec.id, codec, "{path}");
		assert_eq!(stream.parameters.channels, Some(channels));
		assert_eq!(stream.parameters.sample_rate, Some(rate));
		assert!(!stream.parameters.extradata.is_empty(), "{path}: fmt extension kept");
		assert_eq!(decode_all(&mut demuxer), fixture(&format!("{path}.ffmpeg.s16le")), "{path}");
	}
}

#[test]
fn one_block_per_packet_with_sample_timestamps() {
	for (path, codec, channels, _) in CASES {
		let wav = fixture(&format!("{path}.wav"));
		let mut demuxer = RiffDemuxer::new(memory(wav)).unwrap();
		let codec = AdpcmCodec::from_id(codec).unwrap();
		let mut first = None;
		let mut next = 0u64;
		while let Some(packet) = demuxer.read().unwrap() {
			let block = *first.get_or_insert(packet.data.len());
			assert!(packet.data.len() <= block);
			assert_eq!(packet.pts.map(|t| t.ticks()), Some(next), "{path}");
			next += codec.samples_in_block(packet.data.len(), usize::from(channels)).unwrap() as u64;
		}
		let decoded = fixture(&format!("{path}.ffmpeg.s16le")).len() as u64 / 2 / u64::from(channels);
		assert_eq!(next, decoded, "{path}");
		// `fact` gives the exact length; the final block is padded.
		assert!(demuxer.duration().ticks() <= decoded && demuxer.duration().ticks() > decoded - 2048);
	}
}

#[test]
fn seek_lands_on_a_block_boundary() {
	let (path, _, channels, rate) = CASES[0];
	let mut demuxer = RiffDemuxer::new(memory(fixture(&format!("{path}.wav")))).unwrap();
	let reference = fixture(&format!("{path}.ffmpeg.s16le"));
	demuxer.seek(0.15).unwrap();
	let packet = demuxer.read().unwrap().unwrap();
	let pts = packet.pts.unwrap().ticks();
	assert!(pts as f64 / f64::from(rate) <= 0.15 && pts > 0);
	let pcm = decode_block(AdpcmCodec::ImaWav, &packet.data, usize::from(channels)).unwrap();
	let at = pts as usize * usize::from(channels) * 2;
	let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
	assert_eq!(bytes, reference[at..at + bytes.len()]);
}

#[test]
fn truncated_final_block_still_decodes() {
	let (path, _, _, _) = CASES[2];
	let mut wav = fixture(&format!("{path}.wav"));
	let reference = fixture(&format!("{path}.ffmpeg.s16le"));
	wav.truncate(wav.len() - 100);
	let mut demuxer = RiffDemuxer::new(memory(wav)).unwrap();
	let decoded = decode_all(&mut demuxer);
	assert!(decoded.len() < reference.len());
	assert_eq!(decoded, reference[..decoded.len()]);
}

#[test]
fn rejects_unsupported_layouts() {
	// 3-channel MS ADPCM, 3-bit IMA ADPCM, and a block too small for headers.
	for (tag, channels, bits, block) in [(2u16, 3u16, 4u16, 256u16), (0x11, 1, 3, 256), (2, 2, 4, 13)]
	{
		let mut wav = b"RIFF\0\0\0\0WAVEfmt \x14\0\0\0".to_vec();
		for v in [tag, channels] {
			wav.extend_from_slice(&v.to_le_bytes());
		}
		wav.extend_from_slice(&8000u32.to_le_bytes());
		wav.extend_from_slice(&4000u32.to_le_bytes());
		for v in [block, bits, 2, 0] {
			wav.extend_from_slice(&v.to_le_bytes());
		}
		wav.extend_from_slice(b"data\x04\0\0\0\0\0\0\0");
		assert!(RiffDemuxer::new(memory(wav)).is_err(), "tag {tag:#x} ch {channels} bits {bits}");
	}
}
