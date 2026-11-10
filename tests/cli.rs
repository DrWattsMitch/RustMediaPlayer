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
