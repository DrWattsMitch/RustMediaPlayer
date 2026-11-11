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
