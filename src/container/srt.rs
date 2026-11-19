//! SubRip (`.srt`): numbered cues with `HH:MM:SS,mmm --> HH:MM:SS,mmm`
//! timing lines.
//!
//! The demuxer returns one `subrip` packet per cue: its text (with the
//! HTML-like markup left in), pts at the cue start and the cue length as
//! duration, in milliseconds. Cue numbers and the optional `X1:.. Y2:..`
//! box after the timing are dropped. Text lines may contain blank lines:
//! a cue only ends where the next timing line (and its number) starts,
//! as in FFmpeg. The muxer writes `subrip`, `webvtt` or `text` packets.
//!
//! FFmpeg: `libavformat/srtdec.c`, `srtenc.c`.

use crate::core::{
	CodecId, Demuxer, Hint, Metadata, Muxer, Packet, Parameters, Probe, Stream, StreamId, StreamKind,
	StreamSet, Time, TimeBase,
};
use crate::io::Io;
use crate::message::{Message, Result};

/// Cue times are in milliseconds.
pub const TIME_BASE: TimeBase = TimeBase::new(1, 1000);
/// A subtitle file is read whole; bigger ones are refused.
const MAX_FILE: usize = 64 << 20;

/// Reads a whole text file (subtitle files are small).
pub(crate) fn read_all(io: &mut dyn Io, what: &'static str) -> Result<Vec<u8>> {
	let mut data = Vec::new();
	let mut buf = [0u8; 64 << 10];
	loop {
		let n = io.read(&mut buf)?;
		if n == 0 {
			break;
		}
		if data.len() + n > MAX_FILE {
			return Err(Message::Container(what));
		}
		data.extend_from_slice(&buf[..n]);
	}
	Ok(data)
}

/// Decodes subtitle file bytes: UTF-8 (with or without BOM), UTF-16 with a
/// BOM, and otherwise Latin-1 for bytes that aren't valid UTF-8 (FFmpeg
/// leaves the guess to `-sub_charenc`; most legacy `.srt` files are
/// Windows-1252, whose letters match Latin-1).
pub(crate) fn decode_text(data: &[u8]) -> String {
	if let Some(rest) = data.strip_prefix(b"\xEF\xBB\xBF") {
		return String::from_utf8_lossy(rest).into_owned();
	}
	let utf16 = |rest: &[u8], be: bool| {
		let units: Vec<u16> = rest
			.chunks_exact(2)
			.map(|c| if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
			.collect();
		String::from_utf16_lossy(&units)
	};
	if let Some(rest) = data.strip_prefix(b"\xFF\xFE") {
		return utf16(rest, false);
	}
	if let Some(rest) = data.strip_prefix(b"\xFE\xFF") {
		return utf16(rest, true);
	}
	match std::str::from_utf8(data) {
		Ok(s) => s.to_string(),
		Err(_) => data.iter().map(|&b| b as char).collect(),
	}
}

/// `[-]H+:MM:SS[,.]m{1,3}` → milliseconds (signed: a few files start
/// before zero).
fn parse_time(s: &str) -> Option<i64> {
	let s = s.trim();
	let (neg, s) = match s.strip_prefix('-') {
		Some(rest) => (true, rest),
		None => (false, s),
	};
	let (clock, frac) = s.split_once([',', '.']).unwrap_or((s, ""));
	let parts: Vec<&str> = clock.split(':').collect();
	let digits = |p: &str| !p.is_empty() && p.len() <= 9 && p.bytes().all(|b| b.is_ascii_digit());
	let (h, m, sec) = match parts.as_slice() {
		[h, m, s] if digits(h) && digits(m) && digits(s) => (*h, *m, *s),
		[m, s] if digits(m) && digits(s) => ("0", *m, *s),
		_ => return None,
	};
	if !(frac.is_empty() || (frac.len() <= 3 && frac.bytes().all(|b| b.is_ascii_digit()))) {
		return None;
	}
	let ms = match frac.len() {
		0 => 0,
		n => frac.parse::<i64>().ok()? * 10i64.pow(3 - n as u32),
	};
	let (h, m, sec): (i64, i64, i64) = (h.parse().ok()?, m.parse().ok()?, sec.parse().ok()?);
	let t = ((h * 60 + m) * 60 + sec) * 1000 + ms;
	Some(if neg { -t } else { t })
}

/// A timing line `start --> end [X1:...]` → (start, end) in milliseconds.
pub fn parse_timing(line: &str) -> Option<(i64, i64)> {
	let (start, rest) = line.split_once("-->")?;
	let end = rest.split_whitespace().next()?;
	Some((parse_time(start)?, parse_time(end)?))
}

/// `HH:MM:SS,mmm`.
pub fn format_time(ms: u64) -> String {
	format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn is_number(line: &str) -> bool {
	let l = line.trim();
	!l.is_empty() && l.bytes().all(|b| b.is_ascii_digit())
}

/// `true` when `data` starts like a SubRip file: after a BOM and blank
/// lines, a cue number followed by a timing line, or a timing line.
pub fn looks_like_srt(data: &[u8]) -> bool {
	let head = &data[..data.len().min(4096)];
	let text = decode_text(head);
	let mut lines = text.lines().map(str::trim).skip_while(|l| l.is_empty());
	let Some(first) = lines.next() else { return false };
	if parse_timing(first).is_some() {
		return true;
	}
	is_number(first) && lines.next().is_some_and(|l| parse_timing(l).is_some())
}

/// One cue, in milliseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
	pub start: u64,
	pub end: u64,
	pub text: String,
}

/// Parses a SubRip document. Cues are returned in start order; negative
/// times are clamped to zero.
pub fn parse(text: &str) -> Vec<Cue> {
	let text =
		text.strip_prefix('\u{FEFF}').unwrap_or(text).replace("\r\n", "\n").replace('\r', "\n");
	let lines: Vec<&str> = text.split('\n').collect();
	let mut cues = Vec::new();
	let mut current: Option<(i64, i64, usize)> = None;
	let close = |cues: &mut Vec<Cue>, (start, end, from): (i64, i64, usize), to: usize| {
		let mut body = &lines[from..to];
		while body.last().is_some_and(|l| l.trim().is_empty()) {
			body = &body[..body.len() - 1];
		}
		let text = body.join("\n");
		let start = start.max(0) as u64;
		cues.push(Cue { start, end: (end.max(0) as u64).max(start), text });
	};
	let mut i = 0;
	while i < lines.len() {
		let line = lines[i].trim();
		if let Some((start, end)) = parse_timing(line) {
			// The cue number line belongs to this cue, not the previous one.
			let mut text_end = i;
			if i > 0 && is_number(lines[i - 1]) && (i == 1 || lines[i - 2].trim().is_empty()) {
				text_end = i - 1;
			}
			if let Some(c) = current.take() {
				close(&mut cues, c, text_end.max(c.2));
			}
			current = Some((start, end, i + 1));
		}
		i += 1;
	}
	if let Some(c) = current {
		close(&mut cues, c, lines.len());
	}
	cues.sort_by_key(|c| c.start);
	cues
}

pub(crate) fn text_stream(codec: &str, time_base: TimeBase, extradata: Vec<u8>) -> Stream {
	Stream {
		id: StreamId(0),
		kind: StreamKind::Subtitle,
		time_base,
		duration: None,
		parameters: Parameters {
			codec: CodecId::new(codec),
			profile: None,
			bitrate: None,
			sample_rate: None,
			channels: None,
			width: None,
			height: None,
			pixel_format: None,
			extradata,
			bits_per_sample: None,
		},
		metadata: Metadata::default(),
	}
}

pub struct SrtDemuxer {
	cues: Vec<Cue>,
	next: usize,
	streams: StreamSet,
	metadata: Metadata,
	duration: Time,
}

impl SrtDemuxer {
	pub fn new(mut io: Box<dyn Io>) -> Result<Self> {
		let data = read_all(io.as_mut(), "srt: file too large")?;
		let cues = parse(&decode_text(&data));
		if cues.is_empty() && !looks_like_srt(&data) {
			return Err(Message::Container("srt: no cues"));
		}
		let end = cues.iter().map(|c| c.end).max().unwrap_or(0);
		let mut stream = text_stream("subrip", TIME_BASE, Vec::new());
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

impl Demuxer for SrtDemuxer {
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

/// Writes text subtitle packets as a `.srt` file, numbering the cues.
pub struct SrtMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	stream: Option<StreamId>,
	count: u64,
}

impl SrtMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self { io, metadata: Metadata::default(), stream: None, count: 0 }
	}
}

/// Text that can't end a cue early: blank lines are dropped.
fn sanitize(text: &str) -> String {
	text
		.replace("\r\n", "\n")
		.replace('\r', "\n")
		.lines()
		.filter(|l| !l.trim().is_empty())
		.collect::<Vec<_>>()
		.join("\n")
}

impl Muxer for SrtMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.stream.is_some() {
			return Err(Message::Container("srt muxer: only one subtitle stream"));
		}
		if stream.kind != StreamKind::Subtitle
			|| !crate::container::webvtt::holds(&stream.parameters.codec.id)
		{
			return Err(Message::Container("srt muxer: needs a subrip or webvtt subtitle stream"));
		}
		self.stream = Some(stream.id);
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if self.stream != Some(packet.stream_id) {
			return Err(Message::Container("srt muxer: packet for unknown stream"));
		}
		let ms = |t: Option<Time>| t.and_then(|t| t.rescale(TIME_BASE)).map(|t| t.ticks());
		let Some(start) = ms(packet.pts.or(packet.dts)) else { return Ok(()) };
		let end = start + ms(packet.duration).unwrap_or(0);
		self.count += 1;
		let cue = format!(
			"{}\n{} --> {}\n{}\n\n",
			self.count,
			format_time(start),
			format_time(end),
			sanitize(&String::from_utf8_lossy(&packet.data))
		);
		self.io.write_all(cue.as_bytes())
	}

	fn finish(&mut self) -> Result<()> {
		Ok(())
	}
}

pub struct SrtProbe;

impl Probe for SrtProbe {
	fn demuxer_matches(data: &[u8]) -> bool {
		looks_like_srt(data)
	}

	fn muxer_matches(hint: &Hint) -> bool {
		hint.mime_type.as_deref().is_some_and(|m| m.trim() == "application/x-subrip")
			|| hint.extension.as_deref().is_some_and(|e| e.eq_ignore_ascii_case("srt"))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(SrtDemuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(SrtMuxer::new(io)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::io::MemoryIo;

	const FILE: &str = "\u{FEFF}1\r\n00:00:01,000 --> 00:00:02,500 X1:10 X2:20 Y1:0 Y2:5\r\n<i>Hello</i>\r\nworld\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nwith\r\n\r\na gap\r\n\r\n3\r\n00:01:05.5 --> 00:01:06,250\r\n7\r\n";

	#[test]
	fn parses() {
		assert!(looks_like_srt(FILE.as_bytes()));
		assert!(!looks_like_srt(b"WEBVTT\n\n00:01.000 --> 00:02.000\n"));
		assert_eq!(
			parse(FILE),
			[
				Cue { start: 1000, end: 2500, text: "<i>Hello</i>\nworld".into() },
				Cue { start: 3000, end: 4000, text: "with\n\na gap".into() },
				Cue { start: 65_500, end: 66_250, text: "7".into() },
			]
		);
		assert_eq!(parse_timing("-00:00:00,500 --> 00:00:01,000"), Some((-500, 1000)));
		assert_eq!(format_time(3_723_004), "01:02:03,004");
		assert_eq!(decode_text(b"caf\xe9"), "café");
	}

	#[test]
	fn round_trip() {
		let out = MemoryIo::new();
		let mut mux = SrtMuxer::new(Box::new(out.clone()));
		let s = text_stream("subrip", TimeBase::new(1, 90_000), Vec::new());
		mux.add(&s).unwrap();
		let mut p = Packet::new(StreamId(0), b"One\n\ntwo".to_vec());
		p.pts = Some(Time::new(90_000, s.time_base));
		p.duration = Some(Time::new(45_000, s.time_base));
		mux.write(p).unwrap();
		mux.finish().unwrap();
		let text = String::from_utf8(out.contents()).unwrap();
		assert_eq!(text, "1\n00:00:01,000 --> 00:00:01,500\nOne\ntwo\n\n");
		let mut d = SrtDemuxer::new(Box::new(MemoryIo::from_vec(text.into_bytes()))).unwrap();
		let p = d.read().unwrap().unwrap();
		assert_eq!((p.pts.unwrap().ticks(), p.duration.unwrap().ticks()), (1000, 500));
		assert!(d.read().unwrap().is_none());
	}
}
