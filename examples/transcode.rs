//! Decode → convert → encode → mux, like `ffmpeg -i in [options] out`.
//! A thin wrapper over [`rmp::transcode`], the pipeline behind the `rmp`
//! command-line tool (`rmp -i <input> [options] <output>`).
//!
//! cargo run --release --example transcode -- <input> <output> [options]
//!
//! Options (FFmpeg spellings): `-an -vn -sn`, `-c:a -c:v -c:s <codec|copy>`,
//! `-ar`, `-ac`, `-sample_fmt`, `-compression_level`, `-s WxH`, `-pix_fmt`,
//! `-sws_flags`, `-f <format>`, `-metadata key=value`, `-map_metadata -1`.
//! See `rmp::transcode` for the per-container defaults.
//!
//! ```text
//!   transcode song.mp3 song.flac                      # MP3 → FLAC
//!   transcode in.mp4 out.mkv                          # stream copy into Matroska
//!   transcode in.mkv out.mp4 -c:a flac                # copy H.264, AAC → FLAC
//!   transcode clip.webm clip.opus                     # WebM Opus → Ogg Opus
//!   transcode in.flac out.wav -ar 48000 -ac 1 -sample_fmt s32
//!   transcode in.mp4 out.y4m -s 320x180 -pix_fmt yuv444p
//! ```
use rmp::Result;
use rmp::transcode::{Options, transcode};

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
		return Err("usage: transcode <input> <output> [options] (see the file header)".into());
	};
	let options = match Options::parse(&args[3..]) {
		Ok(options) => options,
		Err(message) => {
			eprintln!("{message}");
			std::process::exit(2);
		}
	};
	let summary = transcode(input, output, &options)?;
	for line in &summary.streams {
		println!("{line}");
	}
	println!("frames converted: {}", summary.frames);
	Ok(())
}
