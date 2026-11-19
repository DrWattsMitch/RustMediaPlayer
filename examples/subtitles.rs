//! Prints the text cues of every subtitle stream (SRT, WebVTT, ASS/SSA,
//! or subtitles inside Matroska, MP4 and HLS) with their timing.
//!
//! cargo run --release --example subtitles -- input.mkv [seconds]
//!
//! With `seconds`, prints only the cues on screen at that time, as a
//! player would show them.
use rmp::Result;
use rmp::core::traits::*;
use rmp::core::{Frame, Resolver, StreamKind};
use rmp::io::Input;

fn clock(s: f64) -> String {
	if !s.is_finite() {
		return "--:--:--.---".into();
	}
	let ms = (s * 1000.0).round() as u64;
	format!("{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let input = args.get(1).ok_or("usage: subtitles <input> [seconds]")?;
	let at: Option<f64> = args.get(2).and_then(|s| s.parse().ok());

	let mut demuxer = Input::open(input)?;
	let resolver = Resolver::new();
	let mut decoders: Vec<_> = demuxer
		.streams()
		.by_kind(StreamKind::Subtitle)
		.into_iter()
		.filter_map(|s| {
			let d = resolver.decoder_for(s)?;
			let st = d.create_state(&s.parameters);
			println!("stream {}: {}", s.id.0, s.parameters.codec.id);
			Some((s.id, d, st))
		})
		.collect();
	if decoders.is_empty() {
		return Err("no subtitle stream".into());
	}
	while let Some(packet) = demuxer.read()? {
		let Some((id, dec, st)) = decoders.iter_mut().find(|(id, _, _)| *id == packet.stream_id) else {
			continue;
		};
		for frame in dec.decode(st, packet)?.value {
			let Frame::Subtitle(cue) = frame else { continue };
			if at.is_some_and(|t| !cue.is_showing(t)) {
				continue;
			}
			let style = cue.ass.as_ref().map(|e| format!(" [{}]", e.style)).unwrap_or_default();
			println!(
				"#{} {} --> {}{style}\n{}\n",
				id.0,
				clock(cue.start_seconds()),
				clock(cue.end_seconds()),
				cue.text
			);
		}
	}
	Ok(())
}
