//! `rmp`: the RMP command-line tool, built on [`rmp::transcode`].
//!
//! ```text
//! rmp -i <input>                         show streams and tags
//! rmp -i <input> --show                  ... and every packet
//! rmp [-y|-n] -i <input> [options] <output>
//! rmp -i "dir/*.wav" -f flac <out-dir>/  batch, one job per file
//! ```
//!
//! Options use FFmpeg's spellings; `rmp --help` lists them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rmp::Message;
use rmp::core::traits::*;
use rmp::core::{RawValue, StreamKind, Time};
use rmp::io::Input;
use rmp::transcode::{Options, container_of, transcode};

const USAGE: &str = "\
usage: rmp [-y|-n] -i <input> [options] <output>
       rmp -i <input> [--show]
       rmp -i \"<dir>/*.<ext>\" [-f <format>] [-j <jobs>] [options] <out-dir>/

Without an output, rmp prints the input's streams and tags (--show also
lists every packet). With one, it copies or transcodes into the container
named by the output extension (or -f).

Global options:
  -i <input>            input file, URL or glob pattern (*, ? in the file name)
  -o <output>           output file or directory (or give it last)
  -y | -n               overwrite / never overwrite an existing output
  -j <jobs>             files converted at once in batch mode (default: CPU count)
  --show                list packets when inspecting
  -h, --help            this help;  --version  the version

Stream options:
  -an | -vn | -sn       drop audio / video / subtitles
  -c:a <codec|copy>     audio codec: aac, flac, pcm_s16le, pcm_s24le, pcm_f32le, ...
  -c:v <codec|copy>     video codec: h264, rawvideo or copy
  -c:s copy             copy subtitles (Matroska, WebM)
  -c copy               copy every stream
  -ar <rate>            output sample rate (resampled)
  -ac <channels>        output channels (FFmpeg's default remix)
  -sample_fmt <fmt>     u8, s16, s32, s64, flt, dbl (add p for planar)
  -dither_method <m>    rounding to integer samples: triangular (default when
                        precision is lost), triangular_hp, rectangular,
                        lipshitz (noise shaped, 44.1/48 kHz), none (FFmpeg's)
  -dither_scale <x>     dither noise level, 0-8 (default 1)
  -compression_level n  FLAC level 0-12 (default 5)
  -b:a <rate>           AAC bit rate, e.g. 128k (default 64k per channel)
  -b:v <rate>           H.264 average bit rate, e.g. 2M
  -qp <0-51>            H.264 constant quantiser (default 23)
  -g <frames>           H.264 keyframe interval (default 250)
  -s <W>x<H>            output picture size
  -pix_fmt <name>       yuv420p, yuv422p, yuv444p, yuv420p10le, gray, rgb24, nv12, ...
  -sws_flags <filter>   point, bilinear, bicubic (default), lanczos
  -f <format>           output container: wav flac mp3 mp4 m4a mov mkv mka webm ogg oga opus y4m
  -movflags <flags>     MP4 layout: +faststart (moov first, for progressive web
                        playback) or frag_keyframe+empty_moov+default_base_moof
                        (fragmented MP4 for MSE / HLS / DASH); also
                        frag_every_frame, separate_moof, skip_trailer
  -frag_duration <us>   fragmented MP4: cut a fragment every <us> microseconds
  -min_frag_duration <us>  fragmented MP4: never cut fragments shorter than this
  -metadata key=value   set a tag (title, artist, album, date, track, ...)
  -map_metadata -1      don't copy the input's tags

Containers read: WAV, Y4M, MP3, MP4/M4A/MOV, Matroska/WebM, FLAC, Ogg (Opus/FLAC/Vorbis).
Containers written: all of the above.
Decoders: PCM, ADPCM, MP3, AAC-LC, H.264, VP9, FLAC, Opus, Vorbis. Encoders: PCM, FLAC, AAC-LC, H.264 (Constrained Baseline), raw video.";

/// Parsed command line.
#[derive(Default)]
struct Cli {
	input: Option<String>,
	output: Option<String>,
	overwrite: Option<bool>,
	show: bool,
	jobs: Option<usize>,
	options: Options,
}

fn describe(error: &Message) -> String {
	match error {
		Message::Io(e) => e.to_string(),
		Message::Other(s) | Message::Container(s) | Message::Codec(s) => (*s).to_string(),
	}
}

fn parse(args: &[String]) -> Result<Cli, String> {
	let mut cli = Cli::default();
	let mut i = 0;
	while i < args.len() {
		let arg = args[i].as_str();
		let value = || args.get(i + 1).cloned().ok_or_else(|| format!("option {arg} needs a value"));
		let mut step = 2;
		match arg {
			"-i" => {
				if cli.input.is_some() {
					return Err("only one -i input is supported".into());
				}
				cli.input = Some(value()?);
			}
			"-o" => cli.output = Some(value()?),
			"-y" => {
				cli.overwrite = Some(true);
				step = 1;
			}
			"-n" => {
				cli.overwrite = Some(false);
				step = 1;
			}
			"--show" | "-show" => {
				cli.show = true;
				step = 1;
			}
			"-j" | "-threads" => {
				let v = value()?;
				let jobs = v.parse::<usize>().ok().filter(|&j| j > 0);
				cli.jobs = Some(jobs.ok_or_else(|| format!("invalid job count: {v}"))?);
			}
			"-hide_banner" | "-nostdin" => step = 1,
			_ if Options::is_flag(arg) => {
				cli.options.apply(arg, None)?;
				step = 1;
			}
			_ if Options::is_option(arg) => cli.options.apply(arg, Some(&value()?))?,
			_ if arg.starts_with('-') && arg.len() > 1 => return Err(format!("unknown option {arg}")),
			_ => {
				if cli.output.is_some() {
					return Err(format!("more than one output given ({arg})"));
				}
				cli.output = Some(arg.to_string());
				step = 1;
			}
		}
		i += step;
	}
	Ok(cli)
}

fn seconds(t: Option<Time>) -> String {
	t.map_or_else(|| "-".to_string(), |t| format!("{:.6}", t.as_seconds()))
}

/// `writeln!` that stops quietly when stdout is closed (`rmp ... | head`).
macro_rules! out {
	($w:expr, $($arg:tt)*) => {
		if writeln!($w, $($arg)*).is_err() {
			return Ok(());
		}
	};
}

/// `rmp -i <input>`: streams, tags and (with `--show`) packets.
fn inspect(path: &str, show: bool) -> rmp::Result<()> {
	let mut input = Input::open(path)?;
	let stdout = std::io::stdout();
	let mut w = stdout.lock();
	let duration = input.duration();
	out!(w, "Input: {path}");
	out!(w, "  Duration: {duration} ({:.3} s)", duration.as_seconds());
	let tags = input.metadata().to_string();
	if !tags.trim().is_empty() {
		out!(w, "  Metadata:");
		for line in tags.lines().filter(|l| !l.starts_with("Raw:")) {
			out!(w, "    {line}");
		}
		for (key, value) in &input.metadata().raw {
			if let RawValue::String(v) = value {
				out!(w, "    {key}: {v}");
			}
		}
	}
	for stream in input.streams().iter() {
		let p = &stream.parameters;
		let detail = match stream.kind {
			StreamKind::Audio => format!(
				"{} Hz, {} ch{}",
				p.sample_rate.unwrap_or(0),
				p.channels.unwrap_or(0),
				p.bits_per_sample.map_or(String::new(), |b| format!(", {b}-bit"))
			),
			StreamKind::Video => format!(
				"{}x{}{}",
				p.width.unwrap_or(0),
				p.height.unwrap_or(0),
				p.pixel_format.as_deref().map_or(String::new(), |f| format!(", {f}"))
			),
			_ => String::new(),
		};
		let kind = format!("{:?}", stream.kind).to_ascii_lowercase();
		out!(
			w,
			"  Stream #{}: {kind} {}{}{detail}",
			stream.id.0,
			p.codec.id,
			if detail.is_empty() { "" } else { ", " }
		);
		let mut raw: Vec<_> = stream.metadata.raw.iter().collect();
		raw.sort_by(|a, b| a.0.cmp(b.0));
		for (key, value) in raw {
			if let RawValue::String(v) = value {
				out!(w, "    {key}: {v}");
			}
		}
	}
	if show {
		out!(w, "  Packets: stream, pts (s), dts (s), duration (s), size, flags");
		while let Some(packet) = input.read()? {
			out!(
				w,
				"  {} {} {} {} {} {}",
				packet.stream_id.0,
				seconds(packet.pts),
				seconds(packet.dts),
				seconds(packet.duration),
				packet.data.len(),
				if packet.keyframe { "K" } else { "_" }
			);
		}
	}
	Ok(())
}

fn check_output(input: &str, output: &Path, overwrite: Option<bool>) -> Result<(), String> {
	if Path::new(input) == output {
		return Err(format!("output {} is the input file", output.display()));
	}
	if output.exists() && overwrite != Some(true) {
		return Err(format!("{} already exists (use -y to overwrite)", output.display()));
	}
	Ok(())
}

fn convert_one(input: &str, output: &str, cli: &Cli) -> Result<rmp::transcode::Summary, String> {
	let path = Path::new(output);
	check_output(input, path, cli.overwrite)?;
	let existed = path.exists();
	transcode(input, output, &cli.options).map_err(|e| {
		// Don't leave a half-written file behind (one that was there
		// before, and -y allowed replacing, is gone either way).
		if !existed {
			let _ = std::fs::remove_file(path);
		}
		describe(&e)
	})
}

/// `*` and `?` matching on one path component.
fn wildcard(pattern: &[u8], name: &[u8]) -> bool {
	let (mut p, mut n) = (0, 0);
	let (mut star, mut mark) = (None, 0);
	while n < name.len() {
		if p < pattern.len() && (pattern[p] == b'?' || pattern[p] == name[n]) {
			p += 1;
			n += 1;
		} else if p < pattern.len() && pattern[p] == b'*' {
			star = Some(p);
			mark = n;
			p += 1;
		} else if let Some(s) = star {
			p = s + 1;
			mark += 1;
			n = mark;
		} else {
			return false;
		}
	}
	pattern[p..].iter().all(|&c| c == b'*')
}

fn is_glob(path: &str) -> bool {
	path.contains('*') || path.contains('?')
}

fn expand(pattern: &str) -> Result<Vec<PathBuf>, String> {
	let path = Path::new(pattern);
	let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
	if is_glob(&dir.to_string_lossy()) {
		return Err("wildcards are only supported in the file name".into());
	}
	let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
	let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
	let mut files: Vec<PathBuf> = entries
		.filter_map(|e| e.ok())
		.filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
		.filter(|e| wildcard(name.as_bytes(), e.file_name().to_string_lossy().as_bytes()))
		.map(|e| e.path())
		.collect();
	files.sort();
	Ok(files)
}

/// Batch mode: every file matching the pattern goes to `out_dir`, with the
/// `-f` format's extension (default: the input's own).
fn batch(pattern: &str, out_dir: &str, cli: &Cli) -> Result<(), String> {
	let files = expand(pattern)?;
	if files.is_empty() {
		return Err(format!("no file matches {pattern}"));
	}
	let out_dir = Path::new(out_dir);
	if out_dir.exists() && !out_dir.is_dir() {
		return Err(format!("{} is not a directory (batch output must be one)", out_dir.display()));
	}
	std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;

	let jobs = cli
		.jobs
		.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
		.min(files.len());
	let next = AtomicUsize::new(0);
	let failures = AtomicUsize::new(0);
	let console = Mutex::new(());
	std::thread::scope(|scope| {
		for _ in 0..jobs {
			scope.spawn(|| {
				loop {
					let i = next.fetch_add(1, Ordering::Relaxed);
					let Some(file) = files.get(i) else { break };
					let input = file.to_string_lossy().into_owned();
					let ext = cli.options.format.clone().unwrap_or_else(|| {
						file.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()
					});
					let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
					let output = out_dir.join(format!("{stem}.{ext}"));
					let result = convert_one(&input, &output.to_string_lossy(), cli);
					let _guard = console.lock();
					match result {
						Ok(summary) => {
							println!("{input} -> {} ({} packets)", output.display(), summary.packets)
						}
						Err(e) => {
							failures.fetch_add(1, Ordering::Relaxed);
							eprintln!("{input}: error: {e}");
						}
					}
				}
			});
		}
	});
	match failures.into_inner() {
		0 => Ok(()),
		n => Err(format!("{n} of {} files failed", files.len())),
	}
}

fn run(args: &[String]) -> Result<(), String> {
	let cli = parse(args)?;
	let input = cli.input.clone().ok_or("no input given (-i <input>); see rmp --help")?;
	let Some(output) = cli.output.clone() else {
		if is_glob(&input) {
			for file in expand(&input)? {
				inspect(&file.to_string_lossy(), cli.show).map_err(|e| describe(&e))?;
			}
			return Ok(());
		}
		return inspect(&input, cli.show).map_err(|e| format!("{input}: {}", describe(&e)));
	};
	if is_glob(&input) {
		return batch(&input, &output, &cli);
	}
	if container_of(&output, &cli.options).is_empty() {
		return Err(format!("can't tell the output format of {output} (add an extension or -f)"));
	}
	let summary = convert_one(&input, &output, &cli)?;
	for line in &summary.streams {
		println!("{line}");
	}
	println!("{input} -> {output}: {} packets, {} frames converted", summary.packets, summary.frames);
	Ok(())
}

fn main() -> ExitCode {
	let args: Vec<String> = std::env::args().skip(1).collect();
	if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help" || a == "-help") {
		println!("{USAGE}");
		return if args.is_empty() { ExitCode::from(2) } else { ExitCode::SUCCESS };
	}
	if args.iter().any(|a| a == "--version" || a == "-version") {
		println!("rmp {}", env!("CARGO_PKG_VERSION"));
		return ExitCode::SUCCESS;
	}
	match run(&args) {
		Ok(()) => ExitCode::SUCCESS,
		Err(message) => {
			eprintln!("rmp: {message}");
			ExitCode::FAILURE
		}
	}
}
