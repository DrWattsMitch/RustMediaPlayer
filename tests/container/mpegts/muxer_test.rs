//! MPEG-TS muxer tests: TS → TS remuxes come back packet for packet, MP4 /
//! Matroska / WebM input becomes what TS carries (Annex B, ADTS, Opus
//! control headers) and decodes to the same frames, the tables and packet
//! layout follow ISO/IEC 13818-1, and SAMPLE-AES streams decrypt back.

use super::mpegts_demuxer_test::{Reference, as_reference, per_stream};
use crate::common::{fixture, memory, read_all};
use rmp::container::hls::sample_aes::{SampleCipher, SampleCodec, decrypt_packet};
use rmp::container::mkv::demuxer::MkvDemuxer;
use rmp::container::mp4::demuxer::Mp4Demuxer;
use rmp::container::mpegts::{TsDemuxer, TsMuxer};
use rmp::core::{Demuxer, Frame, Muxer, Packet, RawValue, Resolver, StreamId};
use rmp::io::MemoryIo;

/// Opens a demuxer over a file's bytes.
type Open = fn(Vec<u8>) -> Box<dyn Demuxer>;

/// Muxes everything `source` holds into a TS, set up by `setup`.
fn mux_with(source: &mut dyn Demuxer, setup: impl FnOnce(&mut TsMuxer)) -> Vec<u8> {
	let out = MemoryIo::new();
	let mut muxer = TsMuxer::new(Box::new(out.clone()));
	setup(&mut muxer);
	muxer.set_metadata(source.metadata().clone()).unwrap();
	for s in source.streams().iter() {
		muxer.add(s).unwrap();
	}
	while let Some(p) = source.read().unwrap() {
		muxer.write(p).unwrap();
	}
	muxer.finish().unwrap();
	out.contents()
}

fn mux(source: &mut dyn Demuxer) -> Vec<u8> {
	mux_with(source, |_| {})
}

fn ts(data: Vec<u8>) -> TsDemuxer {
	TsDemuxer::new(memory(data)).unwrap()
}

fn language(d: &dyn Demuxer, i: usize) -> Option<String> {
	match d.streams().streams[i].metadata.raw.get("language") {
		Some(RawValue::String(l)) => Some(l.clone()),
		_ => None,
	}
}

/// Every stream decoded with RMP's decoders, frame bytes per stream.
fn decode_streams(demuxer: &mut dyn Demuxer) -> Vec<Vec<u8>> {
	let streams = demuxer.streams().streams.clone();
	let resolver = Resolver::new();
	let mut decoders: Vec<_> = streams
		.iter()
		.map(|s| {
			let d = resolver.decoder_for(s).expect("decoder");
			let state = d.create_state(&s.parameters);
			(d, state)
		})
		.collect();
	let mut out = vec![Vec::new(); streams.len()];
	let mut push = |i: usize, frames: Vec<Frame>| {
		for f in frames {
			match f {
				Frame::Audio(a) => out[i].extend_from_slice(&a.data),
				Frame::Video(v) => out[i].extend_from_slice(&v.data()),
				Frame::Subtitle(_) => {}
			}
		}
	};
	while let Some(p) = demuxer.read().unwrap() {
		let i = p.stream_id.0;
		let (d, state) = &mut decoders[i];
		let frames = d.decode(state, p).unwrap().value;
		push(i, frames);
	}
	for (i, (d, state)) in decoders.iter_mut().enumerate() {
		let frames = d.flush(state).unwrap().value;
		push(i, frames);
	}
	out
}

#[test]
fn ts_remux_comes_back_packet_for_packet() {
	for name in ["h264_aac_2audio.ts", "hevc_ac3.ts", "mpeg2_mp2.ts", "h264_mp3.m2ts", "opus_51.ts"] {
		let data = fixture(&format!("mpegts/{name}"));
		let mut source = ts(data.clone());
		let out = mux(&mut source);
		assert_eq!(out.len() % 188, 0, "{name}");
		let mut copy = ts(out);
		let mut source = ts(data);
		let want = per_stream(read_all(&mut source).iter().map(as_reference));
		let got = per_stream(read_all(&mut copy).iter().map(as_reference));
		assert_eq!(got.len(), want.len(), "{name}: streams");
		// One shift for the whole file: the output starts at 1.4 s.
		let shift = got[0][0].dts.unwrap() - want[0][0].dts.unwrap();
		for (s, (g, w)) in got.iter().zip(&want).enumerate() {
			assert_eq!(g.len(), w.len(), "{name} stream {s}");
			for (i, (g, w)) in g.iter().zip(w).enumerate() {
				let moved =
					Reference { pts: w.pts.map(|t| t + shift), dts: w.dts.map(|t| t + shift), ..w.clone() };
				assert_eq!(*g, moved, "{name} stream {s} packet {i}");
			}
		}
		for i in 0..source.streams().streams.len() {
			assert_eq!(language(&copy, i), language(&source, i), "{name} stream {i}");
			let (a, b) = (&copy.streams().streams[i].parameters, &source.streams().streams[i].parameters);
			assert_eq!(
				(&a.codec, a.sample_rate, a.channels, a.width),
				(&b.codec, b.sample_rate, b.channels, b.width)
			);
		}
		assert!(copy.metadata().raw.contains_key("service_name"), "{name}");
	}
	// The first file starts at FFmpeg's 1.4 s.
	let mut copy = ts(mux(&mut ts(fixture("mpegts/h264_aac_2audio.ts"))));
	let first = read_all(&mut copy).iter().filter_map(|p| p.dts).map(|t| t.ticks()).min();
	assert_eq!(first, Some(126_000));
}

#[test]
fn mp4_and_matroska_input_decodes_the_same() {
	// MP4: length-prefixed H.264 with B-frames and raw AAC. Matroska: the
	// same, with presentation times only (decode times are derived).
	let sources: [(&str, Open); 2] = [
		("mp4/h264_bframes_aac.mp4", |d| Box::new(Mp4Demuxer::new(memory(d)).unwrap())),
		("mkv/h264_aac.mkv", |d| Box::new(MkvDemuxer::new(memory(d)).unwrap())),
	];
	for (name, open) in sources {
		let out = mux(open(fixture(name)).as_mut());
		let mut copy = ts(out.clone());
		let video = copy.streams().streams[0].clone();
		assert!(
			video.parameters.extradata.starts_with(&[0, 0, 0, 1, 0x67]),
			"{name}: Annex B parameters"
		);
		let packets = read_all(&mut copy);
		let mut last_dts = 0;
		for p in packets.iter().filter(|p| p.stream_id == StreamId(0)) {
			// An access unit delimiter first; decode times in order and never
			// after the presentation time.
			assert!(p.data.starts_with(&[0, 0, 0, 1, 0x09, 0xF0]), "{name}");
			let (pts, dts) = (p.pts.unwrap().ticks(), p.dts.unwrap().ticks());
			assert!(dts <= pts && dts >= last_dts, "{name}: pts {pts} dts {dts}");
			last_dts = dts;
		}
		assert!(
			packets.iter().filter(|p| p.stream_id == StreamId(1)).all(|p| p.data[0] == 0xFF),
			"{name}: ADTS"
		);
		let want = decode_streams(open(fixture(name)).as_mut());
		let got = decode_streams(&mut ts(out));
		assert_eq!(got.len(), want.len());
		for (s, (g, w)) in got.iter().zip(&want).enumerate() {
			assert!(!w.is_empty());
			assert!(g == w, "{name}: stream {s} decodes differently");
		}
	}
}

#[test]
fn opus_from_webm_and_ogg_keeps_layout_and_pre_skip() {
	use rmp::container::ogg::demuxer::OggDemuxer;
	let inputs: [(&str, Open); 3] = [
		("opus/celt_51.webm", |d| Box::new(MkvDemuxer::new(memory(d)).unwrap())),
		("opus/hybrid_st.webm", |d| Box::new(MkvDemuxer::new(memory(d)).unwrap())),
		("ogg/mono.opus", |d| Box::new(OggDemuxer::new(memory(d)).unwrap())),
	];
	for (name, open) in inputs {
		let mut source = open(fixture(name));
		let head =
			rmp::codecs::opus::OpusHead::parse(&source.streams().streams[0].parameters.extradata)
				.unwrap();
		let mut copy = ts(mux(source.as_mut()));
		let s = &copy.streams().streams[0];
		assert_eq!(s.parameters.codec.id, "opus", "{name}");
		let got = rmp::codecs::opus::OpusHead::parse(&s.parameters.extradata).unwrap();
		assert_eq!(
			(got.channels, got.pre_skip, &got.mapping),
			(head.channels, head.pre_skip, &head.mapping),
			"{name}"
		);
		let mut source = open(fixture(name));
		let want: Vec<Vec<u8>> = read_all(source.as_mut()).into_iter().map(|p| p.data).collect();
		let got: Vec<Vec<u8>> = read_all(&mut copy).into_iter().map(|p| p.data).collect();
		assert_eq!(got, want, "{name}: Opus packets");
	}
}

#[test]
fn tables_pids_and_continuity() {
	let out = mux(&mut ts(fixture("mpegts/h264_aac_2audio.ts")));
	let mut counters: std::collections::HashMap<u16, u8> = Default::default();
	let mut pcr_pids = std::collections::HashSet::new();
	for (i, p) in out.chunks(188).enumerate() {
		assert_eq!(p[0], 0x47, "packet {i}: sync");
		let pid = (u16::from(p[1] & 0x1F) << 8) | u16::from(p[2]);
		let has_payload = p[3] & 0x10 != 0;
		let has_af = p[3] & 0x20 != 0;
		if has_af && p[4] > 0 && p[5] & 0x10 != 0 {
			pcr_pids.insert(pid);
		}
		if has_payload {
			let cc = p[3] & 0x0F;
			if let Some(prev) = counters.insert(pid, cc) {
				assert_eq!(cc, (prev + 1) & 0x0F, "packet {i} pid {pid:#x}: continuity");
			}
		}
		if i == 0 {
			assert_eq!(pid, 0x11, "SDT first");
		}
	}
	let mut pids: Vec<u16> = counters.keys().copied().collect();
	pids.sort_unstable();
	assert_eq!(pids, [0x0000, 0x0011, 0x0100, 0x0101, 0x0102, 0x1000]);
	assert_eq!(pcr_pids.into_iter().collect::<Vec<_>>(), [0x100], "PCR on the video PID");

	// FFmpeg's option names move the PIDs and the service.
	let out = mux_with(&mut ts(fixture("mpegts/hevc_ac3.ts")), |m| {
		m.set_option("mpegts_start_pid", "0x200").unwrap();
		m.set_option("mpegts_pmt_start_pid", "0x500").unwrap();
		m.set_option("mpegts_service_id", "7").unwrap();
		assert!(m.set_option("mpegts_start_pid", "1").is_err());
		assert!(m.set_option("nonsense", "1").is_err());
	});
	let pids: std::collections::BTreeSet<u16> =
		out.chunks(188).map(|p| (u16::from(p[1] & 0x1F) << 8) | u16::from(p[2])).collect();
	assert_eq!(pids.into_iter().collect::<Vec<_>>(), [0x0000, 0x0011, 0x0200, 0x0201, 0x0500]);
	assert_eq!(read_all(&mut ts(out)).len(), read_all(&mut ts(fixture("mpegts/hevc_ac3.ts"))).len());
}

#[test]
fn sample_aes_streams_decrypt_to_the_clear_ones() {
	let key = [0x3C; 16];
	let iv = [0x11; 16];
	let clear = mux(&mut ts(fixture("mpegts/h264_aac_2audio.ts")));
	let encrypted = mux_with(&mut ts(fixture("mpegts/h264_aac_2audio.ts")), |m| {
		m.set_sample_aes(true).unwrap();
		m.set_cipher(Some(SampleCipher::new(&key, &iv)));
	});
	let mut clear = ts(clear);
	let mut enc = ts(encrypted);
	// Stream types 0xDB / 0xCF: still H.264 and AAC, flagged as encrypted.
	for s in enc.streams().iter() {
		assert!(
			matches!(s.metadata.raw.get("encryption"), Some(RawValue::String(m)) if m == "SAMPLE-AES")
		);
	}
	let codecs: Vec<SampleCodec> =
		enc.streams().iter().map(|s| SampleCodec::of(&s.parameters.codec.id).unwrap()).collect();
	let cipher = SampleCipher::new(&key, &iv);
	let want: Vec<Packet> = read_all(&mut clear);
	let got: Vec<Packet> = read_all(&mut enc);
	assert_eq!(got.len(), want.len());
	let mut changed = 0;
	for (g, w) in got.iter().zip(&want) {
		assert_eq!((g.stream_id, g.pts, g.dts, g.keyframe), (w.stream_id, w.pts, w.dts, w.keyframe));
		changed += usize::from(g.data != w.data);
		assert_eq!(decrypt_packet(&cipher, codecs[g.stream_id.0], &g.data), w.data);
	}
	assert!(changed > got.len() / 2, "most samples are encrypted");
}
