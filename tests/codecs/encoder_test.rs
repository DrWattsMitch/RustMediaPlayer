//! AAC and H.264 encoders end to end: transcode a fixture, read the result
//! back with RMP's demuxers and decoders, and compare with the source.

use rmp::codecs::aac::AacEncoder;
use rmp::codecs::h264::H264Encoder;
use rmp::core::{
	AudioFrame, Channels, Demuxer, Encoder, Frame, RawValue, Resolver, SampleFormat, SampleLayout,
	StreamId, StreamKind, VideoFrame,
};
use rmp::io::Input;
use rmp::transcode::{Options, transcode};

fn fixture(path: &str) -> String {
	format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), path)
}

/// Decodes the first stream of `kind`: interleaved S16 audio or I420
/// frames (concatenated), plus the packets' keyframe flags.
fn decode(path: &str, kind: StreamKind) -> (Vec<u8>, Vec<bool>, rmp::core::Stream) {
	let mut input = Input::open(path).unwrap();
	let stream = input.streams().iter().find(|s| s.kind == kind).unwrap().clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let (mut out, mut keys) = (Vec::new(), Vec::new());
	let take = |frames: Vec<Frame>, out: &mut Vec<u8>| {
		for f in frames {
			match f {
				Frame::Audio(a) => out.extend_from_slice(&a.data),
				Frame::Video(v) => out.extend_from_slice(&v.data),
			}
		}
	};
	while let Some(p) = input.read().unwrap() {
		if p.stream_id == stream.id {
			keys.push(p.keyframe);
			take(decoder.decode(&mut state, p).unwrap().value, &mut out);
		}
	}
	take(decoder.flush(&mut state).unwrap().value, &mut out);
	(out, keys, stream)
}

fn s16(bytes: &[u8]) -> Vec<f64> {
	bytes.chunks_exact(2).map(|c| f64::from(i16::from_le_bytes([c[0], c[1]]))).collect()
}

fn snr(reference: &[f64], decoded: &[f64]) -> f64 {
	let sig: f64 = reference.iter().map(|v| v * v).sum();
	let err: f64 = reference.iter().zip(decoded).map(|(a, b)| (a - b) * (a - b)).sum();
	10.0 * (sig / err.max(1e-9)).log10()
}

#[test]
fn aac_transcode_round_trips() {
	let dir = tempfile::tempdir().unwrap();
	let out = dir.path().join("out.m4a");
	let src = fixture("wav/wav_stereo_44k_mix_sample.wav");
	let options = Options { audio_bitrate: Some(160_000), ..Options::default() };
	let summary = transcode(&src, out.to_str().unwrap(), &options).unwrap();
	assert!(summary.streams[0].contains("-> aac"), "{:?}", summary.streams);

	let (reference, _, _) = decode(&src, StreamKind::Audio);
	let (decoded, keys, stream) = decode(out.to_str().unwrap(), StreamKind::Audio);
	assert_eq!(stream.parameters.codec.id, "aac");
	assert!(keys.iter().all(|&k| k));
	// The edit list carries the encoder's 1024 priming samples.
	let delay = match stream.metadata.raw.get("encoder_delay") {
		Some(RawValue::String(v)) => v.parse::<usize>().unwrap(),
		other => panic!("encoder_delay: {other:?}"),
	};
	assert_eq!(delay, 1024);
	let (reference, decoded) = (s16(&reference), s16(&decoded));
	let skip = delay * 2;
	assert!(decoded.len() >= skip + reference.len());
	let quality = snr(&reference, &decoded[skip..skip + reference.len()]);
	assert!(quality > 15.0, "SNR {quality:.1} dB");
	// Presented duration equals the source.
	let duration = Input::open(out.to_str().unwrap()).unwrap().duration().as_seconds();
	assert!((duration - 3.0).abs() < 0.002, "{duration}");
	// Bit rate within 10% of the request.
	let bytes = std::fs::metadata(&out).unwrap().len() as f64;
	assert!(bytes * 8.0 / 3.0 < 160_000.0 * 1.1, "{bytes}");
}

#[test]
fn h264_transcode_round_trips() {
	let dir = tempfile::tempdir().unwrap();
	let out = dir.path().join("out.mp4");
	let src = fixture("y4m/yuv420p_34x18.y4m");
	let options =
		Options { video_codec: Some("h264".into()), qp: Some(18), gop: Some(2), ..Options::default() };
	transcode(&src, out.to_str().unwrap(), &options).unwrap();

	let (reference, _, _) = decode(&src, StreamKind::Video);
	let (decoded, keys, stream) = decode(out.to_str().unwrap(), StreamKind::Video);
	assert_eq!(stream.parameters.codec.id, "h264");
	assert_eq!((stream.parameters.width, stream.parameters.height), (Some(34), Some(18)));
	// avcC: Constrained Baseline.
	assert_eq!(&stream.parameters.extradata[1..3], &[66, 0xC0]);
	assert_eq!(decoded.len(), reference.len());
	assert_eq!(keys, (0..keys.len()).map(|i| i % 2 == 0).collect::<Vec<_>>());
	let mse: f64 = reference
		.iter()
		.zip(&decoded)
		.map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
		.sum::<f64>()
		/ reference.len() as f64;
	let psnr = 10.0 * (255.0f64.powi(2) / mse.max(1e-9)).log10();
	assert!(psnr > 38.0, "PSNR {psnr:.1} dB");
}

#[test]
fn encoders_reject_bad_input_without_panicking() {
	use rmp::core::{CodecId, Pixel, Stream};

	// AAC: unsupported rate, mismatched frames.
	let enc = AacEncoder::new(StreamId(0));
	let bad = Stream::audio(StreamId(0), 44_000, 2, CodecId::new("aac"));
	assert!(enc.extradata(&bad.parameters).is_err());
	let mut state = enc.create_state(&bad.parameters);
	let frame = AudioFrame::new(
		vec![0; 64],
		44_000,
		Channels::from_count(2),
		SampleFormat::S16,
		SampleLayout::Interleaved,
	);
	assert!(enc.encode(&mut state, Frame::Audio(frame)).is_err());
	let good = Stream::audio(StreamId(0), 48_000, 2, CodecId::new("aac"));
	let mut state = enc.create_state(&good.parameters);
	let mono = AudioFrame::new(
		vec![0; 64],
		48_000,
		Channels::from_count(1),
		SampleFormat::S16,
		SampleLayout::Interleaved,
	);
	assert!(enc.encode(&mut state, Frame::Audio(mono)).is_err());
	assert!(enc.flush(&mut state).unwrap().value.is_empty());

	// H.264: odd size, wrong pixel format, short data.
	let enc = H264Encoder::new(StreamId(0));
	let odd = Stream::video(StreamId(0), 33, 18, "yuv420p", CodecId::new("h264"));
	assert!(enc.extradata(&odd.parameters).is_err());
	let ok = Stream::video(StreamId(0), 32, 16, "yuv420p", CodecId::new("h264"));
	let mut state = enc.create_state(&ok.parameters);
	let yuv444 = Pixel::from_name("yuv444p").unwrap();
	let frame = |pixel: Pixel, len: usize| VideoFrame {
		data: vec![128; len],
		width: 32,
		height: 16,
		pixel,
		keyframe: rmp::core::Keyframe::Key,
		pts: None,
	};
	assert!(enc.encode(&mut state, Frame::Video(frame(yuv444, 32 * 16 * 3))).is_err());
	let yuv420 = Pixel::from_name("yuv420p").unwrap();
	assert!(enc.encode(&mut state, Frame::Video(frame(yuv420.clone(), 100))).is_err());
	let packets = enc.encode(&mut state, Frame::Video(frame(yuv420, 32 * 16 * 3 / 2))).unwrap().value;
	assert_eq!(packets.len(), 1);
	assert!(packets[0].keyframe);
}
