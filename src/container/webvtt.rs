//! WebVTT (`.vtt`): the subtitle format of HLS and of the web
//! (W3C WebVTT; RFC 8216 §3.5 for HLS segments).
//!
//! The demuxer returns one packet per cue: the cue text (FFmpeg's
//! `webvtt` packet payload, also what Matroska/WebM stores), with the cue
//! start as pts, its length as duration, in milliseconds. Cue identifiers,
//! settings, `NOTE`, `STYLE` and `REGION` blocks are dropped.
//!
//! HLS subtitle segments carry an `X-TIMESTAMP-MAP=MPEGTS:<90 kHz>,LOCAL:<cue
//! time>` header line that ties cue times to the media timeline;
//! [`timestamp_map`] reads it and the segment engine moves the cues by it.
//! The muxer writes the same line when asked to ([`WebVttMuxer::with_timestamp_map`]).
//!
//! FFmpeg: `libavformat/webvttdec.c`, `webvttenc.c`, `hlsenc.c`.

use crate::core::{
	CodecId, Demuxer, Metadata, Muxer, Packet, Parameters, Stream, StreamId, StreamKind, StreamSet,
	Time, TimeBase,
};
use crate::io::Io;
use crate::message::{Message, Result};

/// Cue times are in milliseconds.
pub const TIME_BASE: TimeBase = TimeBase::new(1, 1000);
/// A subtitle file is read whole; bigger ones are refused.
const MAX_FILE: usize = 64 << 20;

/// `true` when `data` starts like a WebVTT file: `WEBVTT` after an
/// optional byte order mark, followed by the end of the line, a space or a
/// tab.
pub fn looks_like_webvtt(data: &[u8]) -> bool {
	let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(data);
	data.starts_with(b"WEBVTT")
		&& data.get(6).is_none_or(|b| matches!(b, b'\n' | b'\r' | b' ' | b'\t'))
}

/// Seconds to add to a segment's cue times from its `X-TIMESTAMP-MAP`
/// (`MPEGTS / 90000 - LOCAL`), or `None` when `data` isn't WebVTT or has no
/// map.
pub fn timestamp_map(data: &[u8]) -> Option<f64> {
	if !looks_like_webvtt(data) {
		return None;
	}
	let text = String::from_utf8_lossy(&data[..data.len().min(4096)]);
	for line in text.lines().skip(1) {
		let line = line.trim();
		if line.is_empty() {
			break;
		}
		let Some(value) = line.strip_prefix("X-TIMESTAMP-MAP=") else { continue };
		let (mut mpegts, mut local) = (None, None);
		for part in value.split(',') {
			let (k, v) = part.split_once(':')?;
			match k.trim() {
				"MPEGTS" => mpegts = v.trim().parse::<u64>().ok(),
				"LOCAL" => local = parse_time(v.trim()),
				_ => {}
			}
		}
		return Some(mpegts? as f64 / 90_000.0 - local.unwrap_or(0.0));
	}
	None
}

/// `[hh:]mm:ss.ttt` → seconds.
pub fn parse_time(s: &str) -> Option<f64> {
	let (clock, frac) = s.split_once('.')?;
	if frac.len() != 3 || !frac.bytes().all(|b| b.is_ascii_digit()) {
		return None;
	}
	let parts: Vec<&str> = clock.split(':').collect();
	let field = |p: &str| {
		(!p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
			.then(|| p.parse::<u64>().ok())
			.flatten()
	};
	let (h, m, sec) = match parts.as_slice() {
		[m, s] => (0, field(m)?, field(s)?),
		[h, m, s] => (field(h)?, field(m)?, field(s)?),
		_ => return None,
	};
	if m > 59 || sec > 59 {
		return None;
	}
	Some((h * 3600 + m * 60 + sec) as f64 + frac.parse::<u64>().ok()? as f64 / 1000.0)
}

/// `hh:mm:ss.ttt`, as FFmpeg's `webvtt` muxer writes it.
pub fn format_time(ms: u64) -> String {
	format!("{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

/// One cue: start and end in milliseconds, and its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
	pub start: u64,
	pub end: u64,
	pub text: String,
}

/// Parses a WebVTT document into its cues.
pub fn parse(text: &str) -> Result<Vec<Cue>> {
	let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
	if !looks_like_webvtt(text.as_bytes()) {
		return Err(Message::Container("webvtt: missing WEBVTT header"));
	}
	let text = text.replace("\r\n", "\n").replace('\r', "\n");
	let mut blocks = text.split("\n\n").map(|b| b.trim_matches('\n')).filter(|b| !b.is_empty());
	blocks.next(); // the header block
	let mut cues = Vec::new();
	for block in blocks {
		let mut lines = block.lines();
		let Some(mut first) = lines.next() else { continue };
		if first.starts_with("NOTE") || first == "STYLE" || first == "REGION" {
			continue;
		}
		if !first.contains("-->") {
			// A cue identifier.
			let Some(next) = lines.next() else { continue };
			first = next;
		}
		let Some((start, rest)) = first.split_once("-->") else { continue };
		let end = rest.split_whitespace().next().unwrap_or("");
		let (Some(start), Some(end)) = (parse_time(start.trim()), parse_time(end)) else { continue };
		let ms = |s: f64| (s * 1000.0).round() as u64;
		let payload: Vec<&str> = lines.collect();
		cues.push(Cue { start: ms(start), end: ms(end).max(ms(start)), text: payload.join("\n") });
	}
	// Cues are in start order in a valid file; be lenient with the others.
	cues.sort_by_key(|c| c.start);
	Ok(cues)
}

fn subtitle_stream() -> Stream {
	Stream {
		id: StreamId(0),
		kind: StreamKind::Subtitle,
		time_base: TIME_BASE,
		duration: None,
		parameters: Parameters {
			codec: CodecId::new("webvtt"),
			profile: None,
			bitrate: None,
			sample_rate: None,
			channels: None,
			width: None,
			height: None,
			pixel_format: None,
			extradata: Vec::new(),
			bits_per_sample: None,
		},
		metadata: Metadata::default(),
	}
}

pub struct WebVttDemuxer {
	cues: Vec<Cue>,
	next: usize,
	streams: StreamSet,
	metadata: Metadata,
	duration: Time,
}

impl WebVttDemuxer {
	pub fn new(mut io: Box<dyn Io>) -> Result<Self> {
		let mut data = Vec::new();
		let mut buf = [0u8; 64 << 10];
		loop {
			let n = io.read(&mut buf)?;
			if n == 0 {
				break;
			}
			if data.len() + n > MAX_FILE {
				return Err(Message::Container("webvtt: file too large"));
			}
			data.extend_from_slice(&buf[..n]);
		}
		let cues = parse(&String::from_utf8_lossy(&data))?;
		let end = cues.iter().map(|c| c.end).max().unwrap_or(0);
		let mut stream = subtitle_stream();
		stream.duration = Some(Time::new(end, TIME_BASE));
		Ok(Self {
			cues,
			next: 0,
			streams: StreamSet::new(vec![stream]),
			metadata: Metadata::default(),
			duration: Time::new(end, TIME_BASE),
		})
	}
}

impl Demuxer for WebVttDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		let Some(cue) = self.cues.get(self.next) else { return Ok(None) };
		self.next += 1;
		let mut p = Packet::new(StreamId(0), cue.text.clone().into_bytes());
		p.pts = Some(Time::new(cue.start, TIME_BASE));
		p.dts = p.pts;
		p.duration = Some(Time::new(cue.end - cue.start, TIME_BASE));
		Ok(Some(p))
	}

	/// Continues at the first cue still showing at `time`.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let t = (time * 1000.0) as u64;
		self.next = self.cues.iter().position(|c| c.end > t).unwrap_or(self.cues.len());
		Ok(())
	}

	fn duration(&self) -> Time {
		self.duration
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn streams(&self) -> &StreamSet {
		&self.streams
	}
}

/// Text that can't end a cue: a blank line would, and `-->` would start a
/// new timing line.
fn sanitize(text: &str) -> String {
	let text = text.replace("\r\n", "\n").replace('\r', "\n");
	text
		.lines()
		.filter(|l| !l.trim().is_empty())
		.map(|l| l.replace("-->", "--&gt;"))
		.collect::<Vec<_>>()
		.join("\n")
}

/// Appends one cue in WebVTT syntax.
pub fn write_cue(out: &mut String, start_ms: u64, end_ms: u64, text: &str) {
	out.push_str(&format_time(start_ms));
	out.push_str(" --> ");
	out.push_str(&format_time(end_ms.max(start_ms)));
	out.push('\n');
	out.push_str(&sanitize(text));
	out.push_str("\n\n");
}

/// The header of a WebVTT document, with an HLS timestamp map when given
/// (`MPEGTS` time, in 90 kHz ticks, of cue time zero).
pub fn header(timestamp_map: Option<u64>) -> String {
	match timestamp_map {
		Some(ts) => format!("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:{ts},LOCAL:00:00:00.000\n\n"),
		None => "WEBVTT\n\n".to_string(),
	}
}

/// Milliseconds of a packet time; `None` without one.
pub fn packet_ms(t: Option<Time>) -> Option<u64> {
	t.and_then(|t| t.rescale(TIME_BASE)).map(|t| t.ticks())
}

/// Writes subtitle packets (`webvtt` cue text, or `subrip` text, which
/// WebVTT accepts as it is apart from the timing syntax) as a `.vtt` file.
pub struct WebVttMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	stream: Option<(StreamId, TimeBase)>,
	started: bool,
	timestamp_map: Option<u64>,
}

impl WebVttMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self { io, metadata: Metadata::default(), stream: None, started: false, timestamp_map: None }
	}

	/// Writes `X-TIMESTAMP-MAP=MPEGTS:<ts>,LOCAL:00:00:00.000` in the header.
	pub fn with_timestamp_map(mut self, mpegts: u64) -> Self {
		self.timestamp_map = Some(mpegts);
		self
	}

	fn start(&mut self) -> Result<()> {
		if !self.started {
			self.io.write_all(header(self.timestamp_map).as_bytes())?;
			self.started = true;
		}
		Ok(())
	}
}

/// Subtitle codecs whose packets are plain cue text.
pub fn holds(codec: &str) -> bool {
	matches!(codec, "webvtt" | "subrip" | "text")
}

impl Muxer for WebVttMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.stream.is_some() {
			return Err(Message::Container("webvtt muxer: only one subtitle stream"));
		}
		if stream.kind != StreamKind::Subtitle || !holds(&stream.parameters.codec.id) {
			return Err(Message::Container("webvtt muxer: needs a webvtt or subrip subtitle stream"));
		}
		self.stream = Some((stream.id, stream.time_base));
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		self.start()?;
		if self.stream.is_none_or(|(id, _)| id != packet.stream_id) {
			return Err(Message::Container("webvtt muxer: packet for unknown stream"));
		}
		let Some(start) = packet_ms(packet.pts.or(packet.dts)) else { return Ok(()) };
		let end = start + packet_ms(packet.duration).unwrap_or(0);
		let mut cue = String::new();
		write_cue(&mut cue, start, end, &String::from_utf8_lossy(&packet.data));
		self.io.write_all(cue.as_bytes())
	}

	fn finish(&mut self) -> Result<()> {
		self.start()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::io::MemoryIo;

	const SEGMENT: &str = "WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000\n\nNOTE a comment\n\n1\n00:00:01.000 --> 00:00:02.500 align:start\nHello\nworld\n\n00:03.000 --> 00:04.000\n<i>Bye</i>\n";

	#[test]
	fn parses_cues_and_map() {
		let cues = parse(SEGMENT).unwrap();
		assert_eq!(
			cues,
			[
				Cue { start: 1000, end: 2500, text: "Hello\nworld".into() },
				Cue { start: 3000, end: 4000, text: "<i>Bye</i>".into() }
			]
		);
		assert_eq!(timestamp_map(SEGMENT.as_bytes()), Some(10.0));
		assert_eq!(timestamp_map(b"WEBVTT\n\n00:01.000 --> 00:02.000\nx\n"), None);
		assert!(parse("1\n00:00:01.000 --> 00:00:02.000\nx").is_err());
		assert_eq!(parse_time("1:02:03.004"), Some(3723.004));
		assert_eq!(parse_time("00:61.000"), None);
		assert_eq!(format_time(3_723_004), "01:02:03.004");
	}

	#[test]
	fn round_trip() {
		let out = MemoryIo::new();
		let mut mux = WebVttMuxer::new(Box::new(out.clone())).with_timestamp_map(126_000);
		let mut s = subtitle_stream();
		s.time_base = TimeBase::new(1, 90_000);
		mux.add(&s).unwrap();
		let mut p = Packet::new(StreamId(0), b"Line one\n\nline --> two".to_vec());
		p.pts = Some(Time::new(90_000, s.time_base));
		p.duration = Some(Time::new(45_000, s.time_base));
		mux.write(p).unwrap();
		mux.finish().unwrap();
		let text = String::from_utf8(out.contents()).unwrap();
		assert!(text.starts_with("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000\n\n"));
		let mut d = WebVttDemuxer::new(Box::new(MemoryIo::from_vec(text.into_bytes()))).unwrap();
		let p = d.read().unwrap().unwrap();
		assert_eq!((p.pts.unwrap().ticks(), p.duration.unwrap().ticks()), (1000, 500));
		assert_eq!(p.data, b"Line one\nline --&gt; two");
		assert!(d.read().unwrap().is_none());
	}
}
