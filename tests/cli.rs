//! End-to-end tests of the `rmp` binary and the `rmp::transcode` pipeline
//! it is built on.

use std::path::{Path, PathBuf};
use std::process::Command;

use rmp::core::{Demuxer, StreamKind};
use rmp::io::Input;
use rmp::transcode::{Options, transcode};

fn fixture(path: &str) -> String {
	format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), path)
}

fn rmp(args: &[&str]) -> (bool, String, String) {
	let out = Command::new(env!("CARGO_BIN_EXE_rmp")).args(args).output().expect("run rmp");
	(
		out.status.success(),
		String::from_utf8_lossy(&out.stdout).into_owned(),
		String::from_utf8_lossy(&out.stderr).into_owned(),
	)
}

fn tmp(dir: &tempfile::TempDir, name: &str) -> PathBuf {
	dir.path().join(name)
}

fn kinds(path: &Path) -> Vec<(StreamKind, String)> {
	let input = Input::open(path.to_str().unwrap()).unwrap();
	input.streams().iter().map(|s| (s.kind, s.parameters.codec.id.clone())).collect()
}

#[test]
fn inspect_lists_streams_and_packets() {
	let (ok, out, _) = rmp(&["-i", &fixture("mkv/h264_flac_srt.mkv")]);
	assert!(ok);
	assert!(out.contains("Stream #0: video h264, 64x48"), "{out}");
	assert!(out.contains("Stream #1: audio flac, 44100 Hz, 1 ch, 16-bit"), "{out}");
	assert!(out.contains("Stream #2: subtitle subrip"), "{out}");
	assert!(out.contains("Title: MKV Song"), "{out}");
	let (ok, out, _) = rmp(&["-i", &fixture("ogg/mono.opus"), "--show"]);
	assert!(ok);
	assert_eq!(out.lines().filter(|l| l.starts_with("  0 ")).count(), 101, "{out}");
}

#[test]
fn copies_into_every_new_container() {
	let dir = tempfile::tempdir().unwrap();
	for (input, output, expected) in [
		("mp4/h264_bframes_aac.mp4", "a.mkv", vec!["h264", "aac"]),
		("mkv/h264_flac_srt.mkv", "b.mka", vec!["flac", "subrip"]),
		("mkv/vp9_opus_vtt.webm", "c.webm", vec!["vp9", "opus", "webvtt"]),
		("opus/hybrid_st.webm", "d.opus", vec!["opus"]),
		("ogg/vorbis.ogg", "e.webm", vec!["vorbis"]),
		("flac/noise_22k.flac", "f.oga", vec!["flac"]),
	] {
		let out = tmp(&dir, output);
		let (ok, _, err) = rmp(&["-i", &fixture(input), out.to_str().unwrap()]);
		assert!(ok, "{input} -> {output}: {err}");
		let codecs: Vec<String> = kinds(&out).into_iter().map(|(_, c)| c).collect();
		assert_eq!(codecs, expected, "{input} -> {output}");
	}
}

#[test]
fn transcodes_with_ffmpeg_style_options() {
	let dir = tempfile::tempdir().unwrap();
	let out = tmp(&dir, "song.mka");
	let (ok, stdout, err) = rmp(&[
		"-i",
		&fixture("mp3/cbr_44k_stereo_tagged.mp3"),
		"-ar",
		"48000",
		"-ac",
		"1",
		"-metadata",
		"title=Renamed",
		out.to_str().unwrap(),
	]);
	assert!(ok, "{err}");
	assert!(stdout.contains("audio mp3 -> flac (48000 Hz, 1 ch)"), "{stdout}");
	let input = Input::open(out.to_str().unwrap()).unwrap();
	let s = &input.streams().streams[0];
	assert_eq!(
		(s.parameters.codec.id.as_str(), s.parameters.sample_rate, s.parameters.channels),
		("flac", Some(48000), Some(1))
	);
	assert_eq!(input.metadata().title.as_deref(), Some("Renamed"));

	// Scaling raw video into Matroska.
	let out = tmp(&dir, "big.mkv");
	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("y4m/yuv420p_34x18.y4m"),
		"-s",
		"68x36",
		"-pix_fmt",
		"yuv444p",
		out.to_str().unwrap(),
	]);
	assert!(ok, "{err}");
	let input = Input::open(out.to_str().unwrap()).unwrap();
	let p = &input.streams().streams[0].parameters;
	assert_eq!((p.width, p.height, p.pixel_format.as_deref()), (Some(68), Some(36), Some("yuv444p")));
}

#[test]
fn refuses_to_overwrite_and_cleans_up_failures() {
	let dir = tempfile::tempdir().unwrap();
	let out = tmp(&dir, "x.mkv");
	let input = fixture("mkv/h264_aac.mkv");
	assert!(rmp(&["-i", &input, out.to_str().unwrap()]).0);
	let (ok, _, err) = rmp(&["-i", &input, out.to_str().unwrap()]);
	assert!(!ok && err.contains("already exists"), "{err}");
	assert!(rmp(&["-y", "-i", &input, out.to_str().unwrap()]).0);

	let bad = tmp(&dir, "bad.webm");
	let (ok, _, err) = rmp(&["-i", &input, bad.to_str().unwrap()]);
	assert!(!ok && err.contains("webm only holds"), "{err}");
	assert!(!bad.exists(), "failed output removed");

	let (ok, _, err) = rmp(&["-i", &input, "-frobnicate", "1", out.to_str().unwrap()]);
	assert!(!ok && err.contains("unknown option -frobnicate"), "{err}");
}

#[test]
fn batch_converts_a_glob() {
	let dir = tempfile::tempdir().unwrap();
	let src = dir.path().join("in");
	std::fs::create_dir(&src).unwrap();
	for name in ["wav_mono_8khz_tone_sample.wav", "wav_voice_band_16khz_sample.wav"] {
		std::fs::copy(fixture(&format!("wav/{name}")), src.join(name)).unwrap();
	}
	std::fs::write(src.join("notes.txt"), b"not audio").unwrap();
	let out = dir.path().join("out");
	let pattern = format!("{}/*.wav", src.display());
	let (ok, stdout, err) = rmp(&["-i", &pattern, "-f", "flac", "-j", "2", out.to_str().unwrap()]);
	assert!(ok, "{err}");
	assert_eq!(stdout.lines().count(), 2, "{stdout}");
	for name in ["wav_mono_8khz_tone_sample.flac", "wav_voice_band_16khz_sample.flac"] {
		assert_eq!(kinds(&out.join(name)), vec![(StreamKind::Audio, "flac".to_string())]);
	}
}

#[test]
fn library_options_parse_like_ffmpeg() {
	let args: Vec<String> =
		["-vn", "-c:a", "flac", "-compression_level", "8", "-f", "ogg", "-sample_fmt", "s32"]
			.iter()
			.map(|s| s.to_string())
			.collect();
	let o = Options::parse(&args).unwrap();
	assert!(o.no_video && !o.no_audio);
	assert_eq!(o.audio_codec.as_deref(), Some("flac"));
	assert_eq!((o.level, o.format.as_deref()), (Some(8), Some("ogg")));
	assert!(Options::parse(&["-compression_level".into(), "13".into()]).is_err());
	assert!(Options::parse(&["-c".into(), "flac".into()]).is_err());

	let dir = tempfile::tempdir().unwrap();
	let out = tmp(&dir, "forced.bin");
	let summary = transcode(&fixture("ogg/flac.oga"), out.to_str().unwrap(), &o).unwrap();
	assert!(summary.packets > 0);
	let input = Input::open(out.to_str().unwrap()).unwrap();
	assert_eq!(input.streams().streams[0].parameters.codec.id, "flac");
}

#[test]
fn encodes_aac_and_h264() {
	let dir = tempfile::tempdir().unwrap();
	// MP4-family outputs re-encode audio to AAC by default.
	let out = tmp(&dir, "a.m4a");
	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("wav/wav_stereo_44k_mix_sample.wav"),
		"-b:a",
		"96k",
		out.to_str().unwrap(),
	]);
	assert!(ok, "{err}");
	assert_eq!(kinds(&out), vec![(StreamKind::Audio, "aac".to_string())]);

	let out = tmp(&dir, "v.mp4");
	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("y4m/yuv420p_34x18.y4m"),
		"-c:v",
		"h264",
		"-b:v",
		"200k",
		"-g",
		"12",
		out.to_str().unwrap(),
	]);
	assert!(ok, "{err}");
	assert_eq!(kinds(&out), vec![(StreamKind::Video, "h264".to_string())]);

	// Scaling into MP4 now encodes H.264 instead of failing on rawvideo.
	let out = tmp(&dir, "s.mp4");
	let (ok, _, err) =
		rmp(&["-i", &fixture("mkv/h264_aac.mkv"), "-s", "32x24", out.to_str().unwrap()]);
	assert!(ok, "{err}");
	assert_eq!(
		kinds(&out),
		vec![(StreamKind::Video, "h264".to_string()), (StreamKind::Audio, "aac".to_string())]
	);

	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("y4m/yuv420p_34x18.y4m"),
		"-qp",
		"60",
		tmp(&dir, "x.mp4").to_str().unwrap(),
	]);
	assert!(!ok && err.contains("-qp"), "{err}");
	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("wav/wav_stereo_44k_mix_sample.wav"),
		"-b:a",
		"fast",
		tmp(&dir, "x.m4a").to_str().unwrap(),
	]);
	assert!(!ok && err.contains("bit rate"), "{err}");
}

#[test]
fn movflags_select_the_mp4_layout() {
	let dir = tempfile::tempdir().unwrap();
	let source = fixture("mp4/h264_bframes_aac.mp4");
	let top = |path: &Path| {
		let data = std::fs::read(path).unwrap();
		let mut kinds = Vec::new();
		let mut at = 0;
		while at + 8 <= data.len() {
			let mut size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
			if size == 1 {
				size = u64::from_be_bytes(data[at + 8..at + 16].try_into().unwrap()) as usize;
			}
			kinds.push(String::from_utf8_lossy(&data[at + 4..at + 8]).into_owned());
			at += size.max(8);
		}
		kinds
	};

	let fast = tmp(&dir, "fast.mp4");
	let (ok, _, err) =
		rmp(&["-i", &source, "-c", "copy", "-movflags", "+faststart", fast.to_str().unwrap()]);
	assert!(ok, "{err}");
	assert_eq!(top(&fast), vec!["ftyp", "moov", "mdat"]);

	let frag = tmp(&dir, "frag.mp4");
	let args =
		["-i", &source, "-c", "copy", "-movflags", "frag_keyframe+empty_moov+default_base_moof"];
	let (ok, _, err) =
		rmp(&[&args[..], &["-frag_duration", "500000", frag.to_str().unwrap()]].concat());
	assert!(ok, "{err}");
	let kinds = top(&frag);
	assert_eq!((&kinds[0][..], &kinds[1][..], &kinds[2][..]), ("ftyp", "moov", "moof"));
	assert_eq!(kinds.last().map(String::as_str), Some("mfra"));

	// Typos and containers without these options are refused up front.
	let bad = tmp(&dir, "bad.mp4");
	let (ok, _, err) = rmp(&["-i", &source, "-movflags", "+fast_start", bad.to_str().unwrap()]);
	assert!(!ok && err.contains("faststart"), "{err}");
	let mkv = tmp(&dir, "out.mkv");
	let (ok, _, err) =
		rmp(&["-i", &source, "-c", "copy", "-movflags", "+faststart", mkv.to_str().unwrap()]);
	assert!(!ok && err.contains("MP4"), "{err}");
	assert!(!mkv.exists());
}

#[test]
fn dithers_when_rounding_to_integer_samples() {
	let dir = tempfile::tempdir().unwrap();
	// Opus decodes to float, so writing 16-bit PCM loses precision.
	let source = fixture("opus/hybrid_st.webm");
	let pcm = |name: &str, extra: &[&str]| {
		let out = tmp(&dir, name);
		let (ok, _, err) = rmp(&[&["-i", &source][..], extra, &[out.to_str().unwrap()]].concat());
		assert!(ok, "{err}");
		let wav = std::fs::read(&out).unwrap();
		let data = &wav[wav.windows(4).position(|w| w == b"data").unwrap() + 8..];
		data.chunks(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect::<Vec<_>>()
	};
	let plain = pcm("plain.wav", &["-dither_method", "none"]);
	let default = pcm("default.wav", &[]);
	let tpdf = pcm("tpdf.wav", &["-dither_method", "triangular"]);
	let shaped = pcm("shaped.wav", &["-dither_method", "lipshitz", "-dither_scale", "0.5"]);
	assert!(default == tpdf, "triangular is the default");
	assert!(default == pcm("again.wav", &[]), "deterministic");
	assert_eq!(plain.len(), default.len());
	assert!(plain != default && default != shaped);
	let diff: Vec<i32> =
		plain.iter().zip(&default).map(|(a, b)| i32::from(*b) - i32::from(*a)).collect();
	assert!(diff.iter().all(|d| d.abs() <= 2));
	let mean = diff.iter().map(|&d| f64::from(d)).sum::<f64>() / diff.len() as f64;
	assert!(mean.abs() < 0.05, "unbiased: {mean}");

	let bad = tmp(&dir, "bad.wav");
	let (ok, _, err) = rmp(&["-i", &source, "-dither_method", "shibata", bad.to_str().unwrap()]);
	assert!(!ok && err.contains("triangular"), "{err}");
	let (ok, _, err) =
		rmp(&["-i", &source, "-ar", "96000", "-dither_method", "lipshitz", bad.to_str().unwrap()]);
	assert!(!ok && err.contains("44100"), "{err}");
}

#[test]
fn transport_streams_and_manifests_as_input() {
	let dir = tempfile::tempdir().unwrap();
	let (ok, out, _) = rmp(&["-i", &fixture("mpegts/h264_aac_2audio.ts")]);
	assert!(ok);
	assert!(out.contains("Stream #0: video h264, 96x64"), "{out}");
	assert!(out.contains("service_name: RMPTest"), "{out}");
	// TS → MP4 / MKV stream copy (Annex B → avcC, ADTS → raw AAC), and an
	// HLS master playlist and a DASH MPD straight into a file.
	for (input, name) in [
		("mpegts/h264_aac_2audio.ts", "ts.mp4"),
		("mpegts/hevc_ac3.ts", "ts.mkv"),
		("hls/master/master.m3u8", "hls.mp4"),
		("dash/template/manifest.mpd", "dash.mkv"),
	] {
		let out = tmp(&dir, name);
		let (ok, _, err) = rmp(&["-i", &fixture(input), "-c", "copy", out.to_str().unwrap()]);
		assert!(ok, "{input}: {err}");
		let kinds = kinds(&out);
		assert_eq!(kinds[0].0, StreamKind::Video, "{input}");
		assert!(kinds.iter().any(|(k, _)| *k == StreamKind::Audio), "{input}");
	}
}

#[test]
fn transport_stream_and_hls_output() {
	let dir = tempfile::tempdir().unwrap();
	// MP4 → TS: Annex B and ADTS; reads back with both streams.
	let ts = tmp(&dir, "out.ts");
	let (ok, _, err) = rmp(&["-i", &fixture("mp4/h264_bframes_aac.mp4"), ts.to_str().unwrap()]);
	assert!(ok, "{err}");
	assert_eq!(kinds(&ts), [(StreamKind::Video, "h264".into()), (StreamKind::Audio, "aac".into())]);
	// Matroska with FLAC and SRT → HLS: FLAC becomes AAC, SRT a WebVTT
	// rendition behind a master playlist.
	let hls = tmp(&dir, "live.m3u8");
	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("mkv/h264_flac_srt.mkv"),
		"-hls_time",
		"1",
		"-hls_playlist_type",
		"vod",
		hls.to_str().unwrap(),
	]);
	assert!(ok, "{err}");
	assert!(std::fs::read_to_string(&hls).unwrap().contains("#EXT-X-ENDLIST"));
	let master = tmp(&dir, "live_master.m3u8");
	assert_eq!(
		kinds(&master),
		[
			(StreamKind::Video, "h264".into()),
			(StreamKind::Audio, "aac".into()),
			(StreamKind::Subtitle, "webvtt".into())
		]
	);
	// Options go where they belong.
	let (ok, _, err) = rmp(&[
		"-i",
		&fixture("mp4/h264_bframes_aac.mp4"),
		"-hls_time",
		"2",
		tmp(&dir, "x.mp4").to_str().unwrap(),
	]);
	assert!(!ok && err.contains("HLS"), "{err}");
	let (ok, _, err) =
		rmp(&["-i", &fixture("mp4/h264_bframes_aac.mp4"), "-hls_time", "-3", hls.to_str().unwrap()]);
	assert!(!ok && err.contains("hls_time"), "{err}");
}

/// Decoded samples of the first audio stream, as one byte vector.
fn decoded_audio(path: &str) -> Vec<u8> {
	use rmp::core::{Frame, Resolver};
	let mut input = Input::open(path).unwrap();
	let stream = input.streams().iter().find(|s| s.kind == StreamKind::Audio).unwrap().clone();
	let decoder = Resolver::new().decoder_for(&stream).unwrap();
	let mut state = decoder.create_state(&stream.parameters);
	let mut out = Vec::new();
	while let Some(p) = input.read().unwrap() {
		if p.stream_id == stream.id {
			for f in decoder.decode(&mut state, p).unwrap().value {
				if let Frame::Audio(a) = f {
					out.extend(a.data);
				}
			}
		}
	}
	out
}

#[test]
fn vp8_alac_and_text_subtitles_convert() {
	let dir = tempfile::tempdir().unwrap();
	// VP8 decodes for re-encoding.
	let mp4 = tmp(&dir, "vp8.mp4");
	let (ok, _, err) = rmp(&["-i", &fixture("vp8/basic.webm"), mp4.to_str().unwrap()]);
	assert!(ok, "{err}");
	assert_eq!(kinds(&mp4), [(StreamKind::Video, "h264".to_string())]);
	// 24-bit and 16-bit ALAC to FLAC stay lossless (no dither), and ALAC
	// copies between MP4 and Matroska.
	for (input, bits) in [("alac/s24_51.mkv", 24), ("alac/s16_stereo.m4a", 16)] {
		let flac = tmp(&dir, &format!("{bits}.flac"));
		let (ok, _, err) = rmp(&["-i", &fixture(input), flac.to_str().unwrap()]);
		assert!(ok, "{err}");
		let info = Input::open(flac.to_str().unwrap()).unwrap();
		assert_eq!(info.streams().streams[0].parameters.bits_per_sample, Some(bits));
		assert!(decoded_audio(flac.to_str().unwrap()) == decoded_audio(&fixture(input)), "{input}");
	}
	let mkv = tmp(&dir, "alac.mkv");
	let m4a = tmp(&dir, "alac.m4a");
	assert!(rmp(&["-i", &fixture("alac/s24_stereo.m4a"), mkv.to_str().unwrap()]).0);
	assert!(rmp(&["-i", mkv.to_str().unwrap(), m4a.to_str().unwrap()]).0);
	assert!(decoded_audio(m4a.to_str().unwrap()) == decoded_audio(&fixture("alac/s24_stereo.m4a")));
	// Text subtitles move between SRT, WebVTT and ASS files.
	let vtt = tmp(&dir, "s.vtt");
	let srt = tmp(&dir, "s.srt");
	let ass = tmp(&dir, "s.ass");
	assert!(rmp(&["-i", &fixture("subtitles/sample.srt"), vtt.to_str().unwrap()]).0);
	assert!(rmp(&["-i", vtt.to_str().unwrap(), srt.to_str().unwrap()]).0);
	let text =
		|p: &str| std::fs::read_to_string(p).unwrap().replace("\r\n", "\n").trim_end().to_string();
	assert_eq!(text(srt.to_str().unwrap()), text(&fixture("subtitles/sample.srt")));
	assert!(rmp(&["-i", &fixture("subtitles/sample.ass"), ass.to_str().unwrap()]).0);
	assert_eq!(kinds(&ass), [(StreamKind::Subtitle, "ass".to_string())]);
}
