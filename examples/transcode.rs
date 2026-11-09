//! Decode → convert → encode → mux, like `ffmpeg -i in [options] out`.
//!
//! cargo run --release --example transcode -- <input> <output> [options]
//!
//! Options (FFmpeg spellings):
//!
//! ```text
//!   -an | -vn                 drop audio / video
//!   -c:a <codec|copy>         audio codec: flac, pcm_s16le, pcm_s24le, pcm_f32le, ...
//!   -c:v <codec|copy>         video codec: rawvideo or copy
//!   -ar <rate>                output sample rate
//!   -ac <channels>            output channel count (FFmpeg's default remix)
//!   -sample_fmt <fmt>         u8, s16, s32, s64, flt, dbl (adds "p" for planar)
//!   -compression_level <n>    FLAC level 0-12 (default 5)
//!   -s <W>x<H>                output picture size
//!   -pix_fmt <name>           yuv420p, yuv444p10le, rgb24, nv12, gray, ...
//!   -sws_flags <filter>       point, bilinear, bicubic (default), lanczos
//! ```
//!
//! The container comes from the output extension: `.wav`, `.flac`, `.y4m`,
//! `.mp4`/`.m4a`, `.mp3`. Defaults per container: WAV → `pcm_s16le`, FLAC →
//! `flac`, Y4M → `rawvideo`, MP4 → stream copy when MP4 can hold the codec,
//! else FLAC audio.
//!
//! Examples:
//!
//! ```text
//!   transcode song.mp3 song.flac                      # MP3 → FLAC
//!   transcode in.mkv out.mp4 -c:a flac                # copy H.264, AAC → FLAC
//!   transcode in.flac out.wav -ar 48000 -ac 1 -sample_fmt s32
//!   transcode in.mp4 out.y4m -s 320x180 -pix_fmt yuv444p
//! ```
use std::collections::HashMap;

use rmp::Result;
use rmp::codecs::flac::FlacEncoder;
use rmp::codecs::pcm::PcmCodec;
use rmp::container::mp4::codec::muxer_supports;
use rmp::core::traits::*;
use rmp::core::{
	CodecId, Frame, Packet, Pixel, Resolver, SampleFormat, SampleLayout, State, Stream, StreamId,
	StreamKind, TimeBase,
};
use rmp::io::{Input, Output};
use rmp::process::{
	AudioConverter, AudioSpec, ScaleFilter, VideoConverter, VideoOptions, VideoSpec,
};

#[derive(Default)]
struct Options {
	no_audio: bool,
	no_video: bool,
	audio_codec: Option<String>,
	video_codec: Option<String>,
	sample_rate: Option<u32>,
	channels: Option<u8>,
	sample_format: Option<(SampleFormat, SampleLayout)>,
	level: Option<u8>,
	size: Option<(u32, u32)>,
	pixel: Option<Pixel>,
	filter: ScaleFilter,
}

fn parse_sample_format(name: &str) -> Option<(SampleFormat, SampleLayout)> {
	let (base, layout) = match name.strip_suffix('p') {
		Some(b) if b != "s" => (b, SampleLayout::Planar),
		_ => (name, SampleLayout::Interleaved),
	};
	let format = match base {
		"u8" => SampleFormat::U8,
		"s16" => SampleFormat::S16,
		"s32" => SampleFormat::S32,
		"s64" => SampleFormat::S64,
		"flt" => SampleFormat::F32,
		"dbl" => SampleFormat::F64,
		_ => return None,
	};
	Some((format, layout))
}

fn parse(args: &[String]) -> Result<Options> {
	let mut o = Options::default();
	let mut i = 0;
	let value =
		|i: usize| args.get(i + 1).cloned().ok_or(rmp::Message::Other("option needs a value"));
	while i < args.len() {
		match args[i].as_str() {
			"-an" => o.no_audio = true,
			"-vn" => o.no_video = true,
			"-c:a" | "-acodec" => o.audio_codec = Some(value(i)?),
			"-c:v" | "-vcodec" => o.video_codec = Some(value(i)?),
			"-ar" => o.sample_rate = Some(value(i)?.parse().map_err(|_| "invalid -ar")?),
			"-ac" => o.channels = Some(value(i)?.parse().map_err(|_| "invalid -ac")?),
			"-sample_fmt" => {
				o.sample_format = Some(parse_sample_format(&value(i)?).ok_or("invalid -sample_fmt")?)
			}
			"-compression_level" => {
				o.level = Some(value(i)?.parse().map_err(|_| "invalid -compression_level")?)
			}
			"-s" => {
				let v = value(i)?;
				let (w, h) = v.split_once('x').ok_or("invalid -s, expected WxH")?;
				o.size =
					Some((w.parse().map_err(|_| "invalid width")?, h.parse().map_err(|_| "invalid height")?));
			}
			"-pix_fmt" => o.pixel = Some(Pixel::from_name(&value(i)?).ok_or("unsupported -pix_fmt")?),
			"-sws_flags" => {
				o.filter = ScaleFilter::from_name(&value(i)?).ok_or("unsupported -sws_flags")?
			}
			other => {
				return Err(rmp::Message::Other(Box::leak(
					format!("unknown option {other}").into_boxed_str(),
				)));
			}
		}
		i += if matches!(args[i].as_str(), "-an" | "-vn") { 1 } else { 2 };
	}
	Ok(o)
}

/// What happens to one input stream.
enum Route {
	Copy { out: StreamId },
	Audio(Box<AudioRoute>),
	Video(Box<VideoRoute>),
}

struct AudioRoute {
	out: StreamId,
	decoder: Box<dyn Decoder>,
	dec_state: State,
	encoder: Box<dyn Encoder>,
	enc_state: State,
	target: AudioSpec,
	converter: Option<AudioConverter>,
}

struct VideoRoute {
	out: StreamId,
	decoder: Box<dyn Decoder>,
	dec_state: State,
	encoder: Box<dyn Encoder>,
	enc_state: State,
	target: (Option<(u32, u32)>, Option<Pixel>),
	filter: ScaleFilter,
	converter: Option<VideoConverter>,
}

fn container(path: &str) -> String {
	path.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

/// Bit depth a decoded stream will most likely have.
fn source_bits(stream: &Stream) -> u8 {
	let id = stream.parameters.codec.id.as_str();
	if let Some(bits) = stream.parameters.bits_per_sample {
		return bits;
	}
	match PcmCodec::from_id(id) {
		Some(codec) => match codec.sample_format() {
			SampleFormat::U8 | SampleFormat::S16 => 16,
			_ => 24,
		},
		None => 16,
	}
}

fn audio_target(stream: &Stream, codec: &str, o: &Options) -> Result<(Stream, AudioSpec)> {
	let p = &stream.parameters;
	let rate = o.sample_rate.or(p.sample_rate).ok_or("input audio has no sample rate")?;
	let channels = o.channels.or(p.channels).ok_or("input audio has no channel count")?;
	let mut out = Stream::audio(StreamId(0), rate, channels, CodecId::new(codec));
	out.metadata = stream.metadata.clone();
	let (format, layout) = if codec == "flac" {
		let bits = match o.sample_format {
			Some((SampleFormat::U8 | SampleFormat::S16, _)) => 16,
			Some(_) => 24,
			None => source_bits(stream).clamp(16, 24),
		};
		out.parameters.bits_per_sample = Some(if bits > 16 { 24 } else { 16 });
		let encoder = FlacEncoder::new(StreamId(0)).with_level(o.level.unwrap_or(5));
		out.parameters.extradata = encoder.extradata(&out.parameters)?;
		(if bits > 16 { SampleFormat::S32 } else { SampleFormat::S16 }, SampleLayout::Interleaved)
	} else {
		let pcm = PcmCodec::from_id(codec).ok_or("unsupported audio codec (use flac or pcm_*)")?;
		(pcm.sample_format(), SampleLayout::Interleaved)
	};
	// Gapless metadata describes the source timing; it doesn't survive
	// decoding (the decoded audio still contains the priming samples).
	out.metadata.raw.remove("encoder_delay");
	out.metadata.raw.remove("encoder_padding");
	let format = o.sample_format.map_or((format, layout), |f| f);
	Ok((out, AudioSpec::new(rate, channels, format.0, format.1)))
}

/// Frame rate of a video stream from its first packet duration.
fn frame_time_base(stream: &Stream, first_duration: Option<u64>) -> TimeBase {
	let tb = stream.time_base;
	match first_duration.filter(|&d| d > 0 && d < u64::from(u32::MAX)) {
		Some(d) => TimeBase::new(tb.num.saturating_mul(d as u32), tb.den),
		None => TimeBase::new(1, 25),
	}
}

pub fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().collect();
	let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
		return Err("usage: transcode <input> <output> [options] (see the file header)".into());
	};
	let options = parse(&args[3..])?;
	let ext = container(output);
	let resolver = Resolver::new();

	let mut demuxer = Input::open(input)?;
	let mut muxer = Output::create(output)?;
	let mut metadata = demuxer.metadata().clone();
	if ext == "y4m" {
		metadata = Default::default();
	}
	muxer.set_metadata(metadata)?;

	// The first video packet's duration gives the frame rate for raw output.
	let streams: Vec<Stream> = demuxer.streams().iter().cloned().collect();
	let mut routes: HashMap<StreamId, Route> = HashMap::new();
	let mut pending: Vec<Packet> = Vec::new();
	let mut next_out = 0usize;

	for stream in &streams {
		let out_id = StreamId(next_out);
		match stream.kind {
			StreamKind::Audio if !options.no_audio && ext != "y4m" => {
				let default = match ext.as_str() {
					"wav" => "pcm_s16le",
					"flac" => "flac",
					"mp4" | "m4a" | "mov" if muxer_supports(&stream.parameters.codec.id) => "copy",
					"mp3" => "copy",
					_ => "flac",
				};
				let codec = options.audio_codec.clone().unwrap_or_else(|| default.to_string());
				if codec == "copy" {
					let mut s = stream.clone();
					s.id = out_id;
					muxer.add(&s)?;
					routes.insert(stream.id, Route::Copy { out: out_id });
				} else {
					if !resolver.has_decoder(&stream.parameters.codec.id) {
						return Err(rmp::Message::Codec(
							"no decoder for the input audio codec (try -c:a copy or -an)",
						));
					}
					let (mut out, target) = audio_target(stream, &codec, &options)?;
					out.id = out_id;
					muxer.add(&out)?;
					let decoder = resolver.decoder_for(stream).ok_or("no decoder")?;
					let dec_state = decoder.create_state(&stream.parameters);
					let encoder: Box<dyn Encoder> = if codec == "flac" {
						Box::new(FlacEncoder::new(out_id).with_level(options.level.unwrap_or(5)))
					} else {
						resolver.encoder_for(&out).ok_or("no encoder")?
					};
					let enc_state = encoder.create_state(&out.parameters);
					println!(
						"stream {}: audio {} -> {} ({} Hz, {} ch)",
						stream.id.0, stream.parameters.codec.id, codec, target.sample_rate, target.channels
					);
					routes.insert(
						stream.id,
						Route::Audio(Box::new(AudioRoute {
							out: out_id,
							decoder,
							dec_state,
							encoder,
							enc_state,
							target,
							converter: None,
						})),
					);
				}
				next_out += 1;
			}
			StreamKind::Video
				if !options.no_video && !matches!(ext.as_str(), "wav" | "flac" | "mp3" | "m4a") =>
			{
				let default = if ext == "y4m" { "rawvideo" } else { "copy" };
				let codec = options.video_codec.clone().unwrap_or_else(|| default.to_string());
				if codec == "copy" {
					let mut s = stream.clone();
					s.id = out_id;
					muxer.add(&s)?;
					routes.insert(stream.id, Route::Copy { out: out_id });
				} else {
					if codec != "rawvideo" {
						return Err("only rawvideo (or copy) video output is supported".into());
					}
					if !resolver.has_decoder(&stream.parameters.codec.id) {
						return Err(rmp::Message::Codec(
							"no decoder for the input video codec (try -c:v copy or -vn)",
						));
					}
					// Find the frame rate from this stream's first packet.
					let mut first = None;
					while first.is_none() {
						let Some(p) = demuxer.read()? else { break };
						if p.stream_id == stream.id {
							first = p.duration.map(|d| d.ticks());
						}
						pending.push(p);
					}
					let p = &stream.parameters;
					let (w, h) = options.size.unwrap_or((p.width.unwrap_or(0), p.height.unwrap_or(0)));
					let pixel = options
						.pixel
						.clone()
						.or_else(|| p.pixel_format.as_deref().and_then(Pixel::from_name))
						.unwrap_or_default();
					let mut out = Stream::video(out_id, w, h, &pixel.name(), CodecId::new("rawvideo"));
					out.time_base = frame_time_base(stream, first);
					muxer.add(&out)?;
					let decoder = resolver.decoder_for(stream).ok_or("no decoder")?;
					let dec_state = decoder.create_state(&stream.parameters);
					let encoder = resolver.encoder_for(&out).ok_or("no encoder")?;
					let enc_state = encoder.create_state(&out.parameters);
					println!(
						"stream {}: video {} -> rawvideo {}x{} {}",
						stream.id.0,
						p.codec.id,
						w,
						h,
						pixel.name()
					);
					routes.insert(
						stream.id,
						Route::Video(Box::new(VideoRoute {
							out: out_id,
							decoder,
							dec_state,
							encoder,
							enc_state,
							target: (options.size, Some(pixel)),
							filter: options.filter,
							converter: None,
						})),
					);
				}
				next_out += 1;
			}
			_ => {}
		}
	}
	if routes.is_empty() {
		return Err("no stream to write".into());
	}

	let mut frames_out = 0u64;
	let mut handle =
		|packet: Packet, routes: &mut HashMap<StreamId, Route>, muxer: &mut Output| -> Result<()> {
			let Some(route) = routes.get_mut(&packet.stream_id) else { return Ok(()) };
			match route {
				Route::Copy { out } => {
					let mut p = packet;
					p.stream_id = *out;
					muxer.write(p)
				}
				Route::Audio(r) => {
					let frames = r.decoder.decode(&mut r.dec_state, packet)?.value;
					audio_frames(r, frames, muxer, &mut frames_out)
				}
				Route::Video(r) => {
					let frames = r.decoder.decode(&mut r.dec_state, packet)?.value;
					video_frames(r, frames, muxer, &mut frames_out)
				}
			}
		};
	for packet in std::mem::take(&mut pending) {
		handle(packet, &mut routes, &mut muxer)?;
	}
	while let Some(packet) = demuxer.read()? {
		handle(packet, &mut routes, &mut muxer)?;
	}
	// Drain decoders, converters and encoders.
	for route in routes.values_mut() {
		match route {
			Route::Copy { .. } => {}
			Route::Audio(r) => {
				let frames = r.decoder.flush(&mut r.dec_state)?.value;
				audio_frames(r, frames, &mut muxer, &mut frames_out)?;
				if let Some(frame) = r.converter.as_mut().map(AudioConverter::flush).transpose()?.flatten()
				{
					for p in r.encoder.encode(&mut r.enc_state, Frame::Audio(frame))?.value {
						muxer.write(p)?;
					}
				}
				for p in r.encoder.flush(&mut r.enc_state)?.value {
					muxer.write(p)?;
				}
			}
			Route::Video(r) => {
				let frames = r.decoder.flush(&mut r.dec_state)?.value;
				video_frames(r, frames, &mut muxer, &mut frames_out)?;
			}
		}
	}
	muxer.finish()?;
	println!("frames converted: {frames_out}");
	Ok(())
}

fn audio_frames(
	r: &mut AudioRoute,
	frames: Vec<Frame>,
	muxer: &mut Output,
	count: &mut u64,
) -> Result<()> {
	for frame in frames {
		let Frame::Audio(a) = frame else { continue };
		if a.nb_samples == 0 {
			continue;
		}
		let converter = match &mut r.converter {
			Some(c) => c,
			None => r.converter.insert(AudioConverter::new(AudioSpec::of(&a), r.target)?),
		};
		let Some(converted) = converter.convert(&a)? else { continue };
		*count += 1;
		for mut p in r.encoder.encode(&mut r.enc_state, Frame::Audio(converted))?.value {
			p.stream_id = r.out;
			muxer.write(p)?;
		}
	}
	Ok(())
}

fn video_frames(
	r: &mut VideoRoute,
	frames: Vec<Frame>,
	muxer: &mut Output,
	count: &mut u64,
) -> Result<()> {
	for frame in frames {
		let Frame::Video(v) = frame else { continue };
		let converter = match &mut r.converter {
			Some(c) => c,
			None => {
				let input = VideoSpec::of(&v);
				let (w, h) = r.target.0.unwrap_or((v.width, v.height));
				let pixel = r.target.1.clone().unwrap_or_else(|| v.pixel.clone());
				let options = VideoOptions { filter: r.filter, ..Default::default() };
				r.converter.insert(VideoConverter::new(input, VideoSpec::new(w, h, pixel), options)?)
			}
		};
		let converted = converter.convert(&v)?;
		*count += 1;
		for mut p in r.encoder.encode(&mut r.enc_state, Frame::Video(converted))?.value {
			p.stream_id = r.out;
			muxer.write(p)?;
		}
	}
	Ok(())
}
