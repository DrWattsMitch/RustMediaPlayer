//! Decode → convert → encode → mux, like `ffmpeg -i in [options] out`.
//!
//! This is the pipeline behind the `rmp` command-line tool and the
//! `transcode` example. Each input stream is either copied (packets go
//! straight to the muxer) or decoded, converted with
//! [`AudioConverter`]/[`VideoConverter`] and re-encoded.
//!
//! The output container comes from the output extension (or
//! [`Options::format`]). When no codec is given, each container gets a
//! sensible default:
//!
//! | Output | Audio | Video | Subtitles |
//! | --- | --- | --- | --- |
//! | `.wav` | `pcm_s16le` | dropped | dropped |
//! | `.flac` | `flac` | dropped | dropped |
//! | `.mp3` | copy | dropped | dropped |
//! | `.y4m` | dropped | `rawvideo` | dropped |
//! | `.mp4`, `.mov` | copy when MP4 holds the codec, else `aac` | copy (`h264` when converting) | dropped |
//! | `.m4a` | as `.mp4` | dropped | dropped |
//! | `.mkv` | copy when Matroska holds the codec, else `flac` | copy (`h264` when converting) | copy |
//! | `.mka` | as `.mkv` | dropped | copy |
//! | `.webm` | copy (Opus/Vorbis only) | copy (VP8/VP9/AV1 only) | copy (WebVTT only) |
//! | `.ogg`, `.oga` | copy for Opus/FLAC/Vorbis, else `flac` | dropped | dropped |
//! | `.opus` | copy (Opus only) | dropped | dropped |
//!
//! RMP's encoders are AAC-LC, FLAC, PCM, H.264 (Constrained Baseline) and
//! raw video, so a transcode can produce those codecs; anything else is a
//! stream copy. Video is "converted" by `-s`, `-pix_fmt`, `-b:v` or `-qp`,
//! or by naming a codec with `-c:v`.

use std::collections::HashMap;

use crate::Message;
use crate::codecs::aac::{self, AacEncoder};
use crate::codecs::flac::FlacEncoder;
use crate::codecs::h264::H264Encoder;
use crate::codecs::pcm::PcmCodec;
use crate::container::mkv::tracks::mux_codec;
use crate::container::mp4::codec::muxer_supports;
use crate::core::traits::*;
use crate::core::{
	CodecId, Frame, Metadata, Packet, Pixel, RawValue, Resolver, SampleFormat, SampleLayout, State,
	Stream, StreamId, StreamKind, TimeBase,
};
use crate::io::{Input, Output};
use crate::message::Result;
use crate::process::{
	AudioConverter, AudioSpec, ScaleFilter, VideoConverter, VideoOptions, VideoSpec, audio::Gapless,
};

/// What to do with the input, in FFmpeg's option vocabulary.
#[derive(Debug, Clone, Default)]
pub struct Options {
	/// `-an`, `-vn`, `-sn`: drop audio, video or subtitle streams.
	pub no_audio: bool,
	pub no_video: bool,
	pub no_subtitles: bool,
	/// `-c:a`, `-c:v`, `-c:s`: codec name or `copy`.
	pub audio_codec: Option<String>,
	pub video_codec: Option<String>,
	pub subtitle_codec: Option<String>,
	/// `-ar`, `-ac`, `-sample_fmt`.
	pub sample_rate: Option<u32>,
	pub channels: Option<u8>,
	pub sample_format: Option<(SampleFormat, SampleLayout)>,
	/// `-compression_level` (FLAC 0-12, default 5).
	pub level: Option<u8>,
	/// `-b:a`, `-b:v`: target bit rates in bits per second (`128k`, `2M`).
	pub audio_bitrate: Option<u32>,
	pub video_bitrate: Option<u32>,
	/// `-qp`: constant H.264 quantiser (0-51).
	pub qp: Option<u8>,
	/// `-g`: H.264 keyframe interval in frames.
	pub gop: Option<u32>,
	/// `-s WxH`, `-pix_fmt`, `-sws_flags`.
	pub size: Option<(u32, u32)>,
	pub pixel: Option<Pixel>,
	pub filter: ScaleFilter,
	/// `-f`: output container by name instead of by extension.
	pub format: Option<String>,
	/// `-metadata key=value`, applied over the input's tags.
	pub metadata: Vec<(String, String)>,
	/// `-map_metadata -1`: don't copy the input's tags.
	pub drop_metadata: bool,
}

/// What a finished transcode did.
#[derive(Debug, Clone, Default)]
pub struct Summary {
	/// One line per output stream, e.g. `stream 1: audio aac -> flac (48000 Hz, 2 ch)`.
	pub streams: Vec<String>,
	/// Frames decoded and re-encoded.
	pub frames: u64,
	/// Packets written to the output.
	pub packets: u64,
}

/// Parses `-sample_fmt` names: `u8`, `s16`, `s32`, `s64`, `flt`, `dbl`,
/// with a `p` suffix for planar.
pub fn parse_sample_format(name: &str) -> Option<(SampleFormat, SampleLayout)> {
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

impl Options {
	/// Options that take no value.
	pub fn is_flag(option: &str) -> bool {
		matches!(option, "-an" | "-vn" | "-sn")
	}

	/// `true` for every option [`Options::apply`] understands.
	pub fn is_option(option: &str) -> bool {
		Self::is_flag(option)
			|| matches!(
				option,
				"-c:a"
					| "-acodec"
					| "-c:v"
					| "-vcodec"
					| "-c:s"
					| "-scodec"
					| "-codec"
					| "-c"
					| "-ar"
					| "-ac"
					| "-sample_fmt"
					| "-compression_level"
					| "-b:a"
					| "-ab"
					| "-b:v"
					| "-vb"
					| "-qp"
					| "-g"
					| "-s"
					| "-pix_fmt"
					| "-sws_flags"
					| "-f"
					| "-metadata"
					| "-map_metadata"
			)
	}

	/// Applies one option (`value` is `None` for flags).
	pub fn apply(&mut self, option: &str, value: Option<&str>) -> std::result::Result<(), String> {
		let need = || value.ok_or_else(|| format!("option {option} needs a value"));
		let number = |v: &str| v.parse::<u32>().map_err(|_| format!("invalid value for {option}: {v}"));
		match option {
			"-an" => self.no_audio = true,
			"-vn" => self.no_video = true,
			"-sn" => self.no_subtitles = true,
			"-c:a" | "-acodec" => self.audio_codec = Some(need()?.to_string()),
			"-c:v" | "-vcodec" => self.video_codec = Some(need()?.to_string()),
			"-c:s" | "-scodec" => self.subtitle_codec = Some(need()?.to_string()),
			"-c" | "-codec" => {
				let v = need()?;
				if v != "copy" {
					return Err(format!("{option} only accepts copy; use -c:a / -c:v for codecs"));
				}
				self.audio_codec = Some(v.into());
				self.video_codec = Some(v.into());
				self.subtitle_codec = Some(v.into());
			}
			"-ar" => self.sample_rate = Some(number(need()?)?),
			"-ac" => {
				let v = need()?;
				let n = number(v)?;
				self.channels = Some(
					u8::try_from(n)
						.ok()
						.filter(|&c| (1..=8).contains(&c))
						.ok_or_else(|| format!("invalid channel count: {v}"))?,
				);
			}
			"-sample_fmt" => {
				let v = need()?;
				self.sample_format =
					Some(parse_sample_format(v).ok_or_else(|| format!("unknown sample format: {v}"))?);
			}
			"-compression_level" => {
				let v = need()?;
				let n = number(v)?;
				self.level = Some(
					u8::try_from(n)
						.ok()
						.filter(|&l| l <= 12)
						.ok_or_else(|| format!("compression level must be 0-12: {v}"))?,
				);
			}
			"-b:a" | "-ab" => self.audio_bitrate = Some(parse_bitrate(need()?)?),
			"-b:v" | "-vb" => self.video_bitrate = Some(parse_bitrate(need()?)?),
			"-qp" => {
				let v = need()?;
				let n = number(v)?;
				self.qp = Some(
					u8::try_from(n)
						.ok()
						.filter(|&q| q <= 51)
						.ok_or_else(|| format!("-qp must be 0-51: {v}"))?,
				);
			}
			"-g" => {
				let v = need()?;
				let n = number(v)?;
				if n == 0 {
					return Err(format!("-g must be at least 1: {v}"));
				}
				self.gop = Some(n);
			}
			"-s" => {
				let v = need()?;
				let (w, h) = v.split_once('x').ok_or_else(|| format!("invalid size {v}, expected WxH"))?;
				let (w, h) = (number(w)?, number(h)?);
				if w == 0 || h == 0 {
					return Err(format!("invalid size {v}"));
				}
				self.size = Some((w, h));
			}
			"-pix_fmt" => {
				let v = need()?;
				self.pixel =
					Some(Pixel::from_name(v).ok_or_else(|| format!("unsupported pixel format: {v}"))?);
			}
			"-sws_flags" => {
				let v = need()?;
				self.filter = ScaleFilter::from_name(v).ok_or_else(|| format!("unknown scaler: {v}"))?;
			}
			"-f" => self.format = Some(need()?.to_ascii_lowercase()),
			"-metadata" => {
				let v = need()?;
				let (key, value) =
					v.split_once('=').ok_or_else(|| format!("-metadata expects key=value: {v}"))?;
				self.metadata.push((key.to_string(), value.to_string()));
			}
			"-map_metadata" => {
				if need()? != "-1" {
					return Err("-map_metadata only supports -1 (drop the input's tags)".into());
				}
				self.drop_metadata = true;
			}
			other => return Err(format!("unknown option {other}")),
		}
		Ok(())
	}

	/// `-ar`, `-ac` or `-sample_fmt` was given.
	pub fn converts_audio(&self) -> bool {
		self.sample_rate.is_some() || self.channels.is_some() || self.sample_format.is_some()
	}

	/// `-s` or `-pix_fmt` was given.
	pub fn converts_video(&self) -> bool {
		self.size.is_some() || self.pixel.is_some()
	}

	/// Parses a list of options such as `["-c:a", "flac", "-ar", "48000"]`.
	pub fn parse(args: &[String]) -> std::result::Result<Self, String> {
		let mut options = Self::default();
		let mut i = 0;
		while i < args.len() {
			let option = args[i].as_str();
			if Self::is_flag(option) {
				options.apply(option, None)?;
				i += 1;
			} else {
				options.apply(option, args.get(i + 1).map(String::as_str))?;
				i += 2;
			}
		}
		Ok(options)
	}
}

/// Parses a bit rate such as `128000`, `128k` or `2.5M` (FFmpeg's SI
/// suffixes).
pub fn parse_bitrate(value: &str) -> std::result::Result<u32, String> {
	let bad = || format!("invalid bit rate: {value}");
	let (number, scale) = match value.chars().last() {
		Some('k' | 'K') => (&value[..value.len() - 1], 1e3),
		Some('M' | 'm') => (&value[..value.len() - 1], 1e6),
		_ => (value, 1.0),
	};
	let v = number.parse::<f64>().map_err(|_| bad())? * scale;
	if !(1000.0..=4.0e9).contains(&v) {
		return Err(bad());
	}
	Ok(v.round() as u32)
}

/// Applies `-metadata` pairs, using the field names FFmpeg uses.
pub fn apply_metadata(metadata: &mut Metadata, pairs: &[(String, String)]) {
	for (key, value) in pairs {
		let text = (!value.is_empty()).then(|| value.clone());
		let number = |v: &str| v.split('/').next().and_then(|n| n.trim().parse::<u32>().ok());
		match key.to_ascii_lowercase().as_str() {
			"title" => metadata.title = text,
			"artist" => metadata.artist = text,
			"album" => metadata.album = text,
			"album_artist" => metadata.album_artist = text,
			"genre" => metadata.genre = text,
			"date" | "year" => metadata.date = text,
			"comment" => metadata.comment = text,
			"description" => metadata.description = text,
			"lyrics" => metadata.lyrics = text,
			"track" => {
				metadata.track_number = number(value);
				metadata.tracks_total = value.split('/').nth(1).and_then(|t| t.trim().parse().ok());
			}
			"disc" => {
				metadata.disc_number = number(value);
				metadata.discs_total = value.split('/').nth(1).and_then(|t| t.trim().parse().ok());
			}
			other => {
				if value.is_empty() {
					metadata.raw.remove(other);
				} else {
					metadata.raw.insert(other.to_string(), RawValue::String(value.clone()));
				}
			}
		}
	}
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
	/// Source priming / padding to drop from the decoded audio.
	gapless: Gapless,
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

/// Output container name: `-f`, else the output extension.
pub fn container_of(output: &str, options: &Options) -> String {
	options
		.format
		.clone()
		.unwrap_or_else(|| output.rsplit('.').next().unwrap_or("").to_ascii_lowercase())
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
	} else if codec == "aac" {
		let encoder = AacEncoder::new(StreamId(0)).with_bitrate(o.audio_bitrate);
		out.parameters.extradata = encoder.extradata(&out.parameters)?;
		out.parameters.bitrate = Some(encoder.bitrate_for(&out.parameters));
		(SampleFormat::F32, SampleLayout::Planar)
	} else {
		let pcm = PcmCodec::from_id(codec).ok_or(Message::Codec(
			"unsupported audio codec: RMP encodes aac, flac and pcm_* (or use -c:a copy)",
		))?;
		(pcm.sample_format(), SampleLayout::Interleaved)
	};
	// Gapless metadata describes the source's timing: the transcode drops
	// those samples (see `gapless_of`), and the encoder states its own.
	out.metadata.raw.remove("encoder_delay");
	out.metadata.raw.remove("encoder_padding");
	if codec == "aac" {
		// The encoder's priming; the MP4 muxer turns it into an edit list.
		out
			.metadata
			.raw
			.insert("encoder_delay".into(), RawValue::String(aac::encoder::ENCODER_DELAY.to_string()));
	}
	let format = o.sample_format.map_or((format, layout), |f| f);
	Ok((out, AudioSpec::new(rate, channels, format.0, format.1)))
}

/// Priming and padding samples to drop from a decoded stream, from the
/// demuxer's gapless metadata.
fn gapless_of(stream: &Stream) -> Gapless {
	let samples = |key: &str| match stream.metadata.raw.get(key) {
		Some(RawValue::String(v)) => v.parse::<u64>().unwrap_or(0),
		_ => 0,
	};
	Gapless::new(samples("encoder_delay"), samples("encoder_padding"))
}

/// Frame rate of a video stream from its first packet duration.
fn frame_time_base(stream: &Stream, first_duration: Option<u64>) -> TimeBase {
	let tb = stream.time_base;
	match first_duration.filter(|&d| d > 0 && d < u64::from(u32::MAX)) {
		Some(d) => TimeBase::new(tb.num.saturating_mul(d as u32), tb.den),
		None => TimeBase::new(1, 25),
	}
}

fn matroska_holds(stream: &Stream, webm: bool) -> bool {
	mux_codec(&stream.parameters, webm).is_ok()
}

/// Default audio codec for a container (`None`: drop audio).
fn default_audio(ext: &str, stream: &Stream) -> Option<&'static str> {
	let codec = stream.parameters.codec.id.as_str();
	Some(match ext {
		"y4m" => return None,
		"wav" => "pcm_s16le",
		"flac" => "flac",
		"mp3" => "copy",
		"mp4" | "m4a" | "mov" if muxer_supports(codec) => "copy",
		"mp4" | "m4a" | "mov" => "aac",
		"mkv" | "mka" | "mks" if matroska_holds(stream, false) => "copy",
		// WebM and .opus only hold codecs RMP can't encode: copy, and let
		// the muxer explain when the codec doesn't fit.
		"webm" | "opus" => "copy",
		"ogg" | "oga" if matches!(codec, "opus" | "flac" | "vorbis") => "copy",
		_ => "flac",
	})
}

/// Whether a container takes video / subtitle streams by default.
fn keeps_video(ext: &str) -> bool {
	matches!(ext, "y4m" | "mp4" | "mov" | "mkv" | "webm")
}

fn keeps_subtitles(ext: &str, stream: &Stream) -> bool {
	match ext {
		"mkv" | "mka" | "mks" => matroska_holds(stream, false),
		"webm" => stream.parameters.codec.id == "webvtt",
		_ => false,
	}
}

/// Runs a transcode from `input` to `output`.
pub fn transcode(input: &str, output: &str, options: &Options) -> Result<Summary> {
	let ext = container_of(output, options);
	let resolver = Resolver::new();
	let mut summary = Summary::default();

	let mut demuxer = Input::open(input)?;
	let mut muxer = match &options.format {
		Some(format) => Output::create_as(output, format)?,
		None => Output::create(output)?,
	};
	let mut metadata = if options.drop_metadata || ext == "y4m" {
		Metadata::default()
	} else {
		demuxer.metadata().clone()
	};
	apply_metadata(&mut metadata, &options.metadata);
	muxer.set_metadata(metadata)?;

	let streams: Vec<Stream> = demuxer.streams().iter().cloned().collect();
	let mut routes: HashMap<StreamId, Route> = HashMap::new();
	let mut pending: Vec<Packet> = Vec::new();
	let mut next_out = 0usize;

	for stream in &streams {
		let out_id = StreamId(next_out);
		let codec_in = stream.parameters.codec.id.clone();
		match stream.kind {
			StreamKind::Audio if !options.no_audio => {
				let Some(mut default) = default_audio(&ext, stream) else { continue };
				if default == "copy" && options.converts_audio() {
					// Resampling or remixing means re-encoding (as in FFmpeg).
					default = match ext.as_str() {
						"webm" | "opus" | "mp3" => {
							return Err(Message::Codec(
								"-ar/-ac/-sample_fmt need re-encoding, and RMP has no encoder for this container's audio",
							));
						}
						"mp4" | "m4a" | "mov" => "aac",
						_ => "flac",
					};
				}
				let codec = options.audio_codec.clone().unwrap_or_else(|| default.to_string());
				if codec == "copy" {
					let mut s = stream.clone();
					s.id = out_id;
					muxer.add(&s)?;
					routes.insert(stream.id, Route::Copy { out: out_id });
					summary.streams.push(format!("stream {}: audio {codec_in} (copy)", stream.id.0));
				} else {
					if !resolver.has_decoder(&codec_in) {
						return Err(Message::Codec(
							"no decoder for the input audio codec (try -c:a copy or -an)",
						));
					}
					let (mut out, target) = audio_target(stream, &codec, options)?;
					out.id = out_id;
					muxer.add(&out)?;
					let decoder = resolver.decoder_for(stream).ok_or("no decoder")?;
					let dec_state = decoder.create_state(&stream.parameters);
					let encoder: Box<dyn Encoder> = if codec == "flac" {
						Box::new(FlacEncoder::new(out_id).with_level(options.level.unwrap_or(5)))
					} else if codec == "aac" {
						Box::new(AacEncoder::new(out_id).with_bitrate(options.audio_bitrate))
					} else {
						resolver.encoder_for(&out).ok_or("no encoder")?
					};
					let enc_state = encoder.create_state(&out.parameters);
					summary.streams.push(format!(
						"stream {}: audio {codec_in} -> {codec} ({} Hz, {} ch)",
						stream.id.0, target.sample_rate, target.channels
					));
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
							gapless: gapless_of(stream),
						})),
					);
				}
				next_out += 1;
			}
			StreamKind::Video if !options.no_video && keeps_video(&ext) => {
				// Scaling or a pixel format change means re-encoding: raw for
				// Y4M, H.264 for the containers that hold it.
				// The H.264 encoder is 4:2:0; other pixel formats stay raw.
				let yuv420 = options.pixel.as_ref().is_none_or(|p| p.name() == "yuv420p");
				let default = match ext.as_str() {
					"y4m" => "rawvideo",
					_ if options.video_bitrate.is_some() || options.qp.is_some() => "h264",
					_ if options.converts_video() => {
						if yuv420 {
							"h264"
						} else {
							"rawvideo"
						}
					}
					_ => "copy",
				};
				let codec = options.video_codec.clone().unwrap_or_else(|| default.to_string());
				if codec == "copy" {
					let mut s = stream.clone();
					s.id = out_id;
					muxer.add(&s)?;
					routes.insert(stream.id, Route::Copy { out: out_id });
					summary.streams.push(format!("stream {}: video {codec_in} (copy)", stream.id.0));
				} else {
					if codec != "rawvideo" && codec != "h264" {
						return Err(Message::Codec("RMP encodes h264 and rawvideo (or use -c:v copy)"));
					}
					if !resolver.has_decoder(&codec_in) {
						return Err(Message::Codec(
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
					let pixel = if codec == "h264" {
						// The encoder takes 8-bit 4:2:0.
						let yuv420p = Pixel::from_name("yuv420p").unwrap_or_default();
						if options.pixel.as_ref().is_some_and(|p| *p != yuv420p) {
							return Err(Message::Codec("the h264 encoder supports only -pix_fmt yuv420p"));
						}
						yuv420p
					} else {
						options
							.pixel
							.clone()
							.or_else(|| p.pixel_format.as_deref().and_then(Pixel::from_name))
							.unwrap_or_default()
					};
					let mut out = Stream::video(out_id, w, h, &pixel.name(), CodecId::new(&codec));
					out.time_base = frame_time_base(stream, first);
					let decoder = resolver.decoder_for(stream).ok_or("no decoder")?;
					let dec_state = decoder.create_state(&stream.parameters);
					let encoder: Box<dyn Encoder> = if codec == "h264" {
						let h264 = H264Encoder::new(out_id)
							.with_rate(options.qp, options.video_bitrate)
							.with_gop(options.gop)
							.with_time_base(out.time_base);
						out.parameters.extradata = h264.extradata(&out.parameters)?;
						out.parameters.bitrate = options.video_bitrate;
						Box::new(h264)
					} else {
						resolver.encoder_for(&out).ok_or("no encoder")?
					};
					muxer.add(&out)?;
					let enc_state = encoder.create_state(&out.parameters);
					summary.streams.push(format!(
						"stream {}: video {codec_in} -> {codec} {w}x{h} {}",
						stream.id.0,
						pixel.name()
					));
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
			StreamKind::Subtitle if !options.no_subtitles && keeps_subtitles(&ext, stream) => {
				if options.subtitle_codec.as_deref().is_some_and(|c| c != "copy") {
					return Err(Message::Codec("subtitles can only be copied (-c:s copy)"));
				}
				let mut s = stream.clone();
				s.id = out_id;
				muxer.add(&s)?;
				routes.insert(stream.id, Route::Copy { out: out_id });
				summary.streams.push(format!("stream {}: subtitle {codec_in} (copy)", stream.id.0));
				next_out += 1;
			}
			_ => {}
		}
	}
	if routes.is_empty() {
		return Err("no stream to write (check -an/-vn/-sn and the output format)".into());
	}

	let mut frames_out = 0u64;
	let mut packets_out = 0u64;
	let mut handle =
		|packet: Packet, routes: &mut HashMap<StreamId, Route>, muxer: &mut Output| -> Result<()> {
			let Some(route) = routes.get_mut(&packet.stream_id) else { return Ok(()) };
			match route {
				Route::Copy { out } => {
					let mut p = packet;
					p.stream_id = *out;
					packets_out += 1;
					muxer.write(p)
				}
				Route::Audio(r) => {
					let frames = r.decoder.decode(&mut r.dec_state, packet)?.value;
					packets_out += audio_frames(r, frames, muxer, &mut frames_out)?;
					Ok(())
				}
				Route::Video(r) => {
					let frames = r.decoder.decode(&mut r.dec_state, packet)?.value;
					packets_out += video_frames(r, frames, muxer, &mut frames_out)?;
					Ok(())
				}
			}
		};
	for packet in std::mem::take(&mut pending) {
		handle(packet, &mut routes, &mut muxer)?;
	}
	while let Some(packet) = demuxer.read()? {
		handle(packet, &mut routes, &mut muxer)?;
	}
	// Drain decoders, converters and encoders, in output stream order so
	// the result doesn't depend on hash order.
	let mut ids: Vec<StreamId> = routes.keys().copied().collect();
	ids.sort_by_key(|id| id.0);
	for id in ids {
		let Some(route) = routes.get_mut(&id) else { continue };
		match route {
			Route::Copy { .. } => {}
			Route::Audio(r) => {
				let frames = r.decoder.flush(&mut r.dec_state)?.value;
				packets_out += audio_frames(r, frames, &mut muxer, &mut frames_out)?;
				for frame in r.gapless.finish() {
					packets_out += encode_audio(r, frame, &mut muxer, &mut frames_out)?;
				}
				if let Some(frame) = r.converter.as_mut().map(AudioConverter::flush).transpose()?.flatten()
				{
					for mut p in r.encoder.encode(&mut r.enc_state, Frame::Audio(frame))?.value {
						p.stream_id = r.out;
						packets_out += 1;
						muxer.write(p)?;
					}
				}
				for mut p in r.encoder.flush(&mut r.enc_state)?.value {
					p.stream_id = r.out;
					packets_out += 1;
					muxer.write(p)?;
				}
			}
			Route::Video(r) => {
				let frames = r.decoder.flush(&mut r.dec_state)?.value;
				packets_out += video_frames(r, frames, &mut muxer, &mut frames_out)?;
			}
		}
	}
	muxer.finish()?;
	summary.frames = frames_out;
	summary.packets = packets_out;
	Ok(summary)
}

fn audio_frames(
	r: &mut AudioRoute,
	frames: Vec<Frame>,
	muxer: &mut Output,
	count: &mut u64,
) -> Result<u64> {
	let mut written = 0;
	for frame in frames {
		let Frame::Audio(a) = frame else { continue };
		if a.nb_samples == 0 {
			continue;
		}
		for a in r.gapless.push(a) {
			written += encode_audio(r, a, muxer, count)?;
		}
	}
	Ok(written)
}

/// Converts one decoded frame to the target format and encodes it.
fn encode_audio(
	r: &mut AudioRoute,
	a: crate::core::AudioFrame,
	muxer: &mut Output,
	count: &mut u64,
) -> Result<u64> {
	let converter = match &mut r.converter {
		Some(c) => c,
		None => r.converter.insert(AudioConverter::new(AudioSpec::of(&a), r.target)?),
	};
	let Some(converted) = converter.convert(&a)? else { return Ok(0) };
	*count += 1;
	let mut written = 0;
	for mut p in r.encoder.encode(&mut r.enc_state, Frame::Audio(converted))?.value {
		p.stream_id = r.out;
		written += 1;
		muxer.write(p)?;
	}
	Ok(written)
}

fn video_frames(
	r: &mut VideoRoute,
	frames: Vec<Frame>,
	muxer: &mut Output,
	count: &mut u64,
) -> Result<u64> {
	let mut written = 0;
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
			written += 1;
			muxer.write(p)?;
		}
	}
	Ok(written)
}
