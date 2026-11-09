//! Decodes the first audio stream to raw interleaved S16LE PCM and the first
//! video stream to raw planar YUV (like `ffmpeg -f s16le` / `-f rawvideo`).
//!
//! cargo run --release --example decode -- input.mp4 [audio.pcm] [video.yuv]
//!
//! Pass `-` to skip an output. Prints a summary of what was decoded.
//!
//! Audio honours the stream's `encoder_delay` / `encoder_padding` metadata
//! (from the MP4 edit list or the MP3 LAME tag), dropping priming and padding
//! samples exactly as FFmpeg's `skip_samples` does, so the PCM output lines
//! up sample-for-sample with `ffmpeg -f s16le`.
use std::fs::File;
use std::io::{BufWriter, Write};

use rmp::Result;
use rmp::codecs::pcm::convert::convert;
use rmp::core::traits::*;
use rmp::core::{Frame, RawValue, Resolver, SampleFormat, SampleLayout, Stream, StreamKind};
use rmp::io::Input;

fn open(path: Option<&String>) -> Result<Option<BufWriter<File>>> {
	match path.map(String::as_str) {
		None | Some("-") => Ok(None),
		Some(p) => Ok(Some(BufWriter::new(File::create(p)?))),
	}
}

/// Reads a sample-count metadata key such as `encoder_delay`.
fn samples_key(stream: Option<&Stream>, key: &str) -> u64 {
	match stream.and_then(|s| s.metadata.raw.get(key)) {
		Some(RawValue::String(v)) => v.trim().parse().unwrap_or(0),
		_ => 0,
	}
}

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let input = args.get(1).ok_or("usage: decode <input> [audio.pcm|-] [video.yuv|-]")?;
	let mut audio_out = open(args.get(2))?;
	let mut video_out = open(args.get(3))?;

	let mut demuxer = Input::open(input)?;
	let resolver = Resolver::new();
	let pick = |kind| demuxer.streams().iter().find(|s| s.kind == kind).cloned();
	let audio = pick(StreamKind::Audio);
	let video = pick(StreamKind::Video);

	let mut audio_dec = audio.as_ref().and_then(|s| {
		resolver.decoder_for(s).map(|d| {
			let st = d.create_state(&s.parameters);
			(s.id, d, st)
		})
	});
	let mut video_dec = video.as_ref().and_then(|s| {
		resolver.decoder_for(s).map(|d| {
			let st = d.create_state(&s.parameters);
			(s.id, d, st)
		})
	});

	// Gapless trimming, in samples per channel.
	let mut skip = samples_key(audio.as_ref(), "encoder_delay");
	let padding = samples_key(audio.as_ref(), "encoder_padding");
	// Trailing padding is only known to be padding once the stream ends, so the
	// most recent `padding` samples are held back until more audio arrives.
	let mut held: Vec<u8> = Vec::new();
	let mut bytes_per_sample = 0usize;

	let (mut samples, mut pictures) = (0usize, 0usize);
	let mut sink = |frames: Vec<Frame>, samples: &mut usize, pictures: &mut usize| -> Result<()> {
		for frame in frames {
			match frame {
				Frame::Audio(mut a) => {
					if a.nb_samples == 0 {
						continue;
					}
					// High-resolution PCM decodes to S32/F32/...; convert so the
					// output is always S16LE, as `ffmpeg -f s16le` would write.
					if a.format != SampleFormat::S16 || a.is_planar() {
						a.data = convert(&a, SampleFormat::S16)?;
						a.format = SampleFormat::S16;
						a.layout = SampleLayout::Interleaved;
					}
					bytes_per_sample = a.data.len() / a.nb_samples;
					let mut data: &[u8] = &a.data;
					let drop = skip.min(a.nb_samples as u64) as usize;
					skip -= drop as u64;
					data = data.get(drop * bytes_per_sample..).unwrap_or(&[]);
					held.extend_from_slice(data);
					let keep = (padding as usize).saturating_mul(bytes_per_sample);
					if held.len() > keep {
						let ready: Vec<u8> = held.drain(..held.len() - keep).collect();
						*samples += ready.len() / bytes_per_sample.max(1);
						if let Some(w) = audio_out.as_mut() {
							w.write_all(&ready)?;
						}
					}
				}
				Frame::Video(v) => {
					*pictures += 1;
					if let Some(w) = video_out.as_mut() {
						w.write_all(&v.data)?;
					}
				}
			}
		}
		Ok(())
	};

	while let Some(packet) = demuxer.read()? {
		let target = [audio_dec.as_mut(), video_dec.as_mut()]
			.into_iter()
			.flatten()
			.find(|(id, _, _)| *id == packet.stream_id);
		if let Some((_, dec, st)) = target {
			let frames = dec.decode(st, packet)?.value;
			sink(frames, &mut samples, &mut pictures)?;
		}
	}
	if let Some((_, dec, st)) = audio_dec.as_mut() {
		let frames = dec.flush(st)?.value;
		sink(frames, &mut samples, &mut pictures)?;
	}
	if let Some((_, dec, st)) = video_dec.as_mut() {
		let frames = dec.flush(st)?.value;
		sink(frames, &mut samples, &mut pictures)?;
	}
	if let Some(mut w) = audio_out {
		w.flush()?;
	}
	if let Some(mut w) = video_out {
		w.flush()?;
	}
	println!("audio samples per channel: {samples}");
	println!("video frames: {pictures}");
	Ok(())
}
