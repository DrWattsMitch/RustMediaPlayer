//! ASS / SSA scripts (`.ass`, `.ssa`).
//!
//! The demuxer follows FFmpeg's `assdec.c`: the stream's extradata is the
//! script header up to and including the `[Events]` `Format:` line, and
//! each `Dialogue:` event becomes one `ass` packet in the Matroska layout
//! `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`, with
//! pts and duration in centiseconds. Packets come in start order; the read
//! order keeps the script's own order for renderers that need it.
//! `Comment:` events and the `[Fonts]` / `[Graphics]` sections after
//! `[Events]` are dropped.
//!
//! The muxer writes the extradata header (or a minimal one) and turns the
//! packets back into `Dialogue:` lines, in read order.

use crate::codecs::subtitle::ass::{self as codec, DEFAULT_EVENT_FORMAT};
use crate::container::srt::{decode_text, read_all, text_stream};
use crate::core::{
	Demuxer, Hint, Metadata, Muxer, Packet, Probe, Stream, StreamId, StreamKind, StreamSet, Time,
	TimeBase,
};
use crate::io::Io;
use crate::message::{Message, Result};

/// Event times are in centiseconds.
pub const TIME_BASE: TimeBase = TimeBase::new(1, 100);

/// `true` when `data` starts with a `[Script Info]` section.
pub fn looks_like_ass(data: &[u8]) -> bool {
	let text = decode_text(&data[..data.len().min(1024)]);
	text
		.lines()
		.map(str::trim)
		.find(|l| !l.is_empty() && !l.starts_with(';'))
		.is_some_and(|l| l.eq_ignore_ascii_case("[Script Info]"))
}

pub struct AssDemuxer {
	packets: Vec<Packet>,
	next: usize,
	streams: StreamSet,
	metadata: Metadata,
	duration: Time,
}

impl AssDemuxer {
	pub fn new(mut io: Box<dyn Io>) -> Result<Self> {
		let data = read_all(io.as_mut(), "ass: file too large")?;
		let text = decode_text(&data);
		let text = text.strip_prefix('\u{FEFF}').unwrap_or(&text);
		if !looks_like_ass(text.as_bytes()) {
			return Err(Message::Container("ass: missing [Script Info]"));
		}
		let mut header = String::new();
		let mut in_events = false;
		let mut header_done = false;
		let mut format: Vec<String> = DEFAULT_EVENT_FORMAT.iter().map(|s| s.to_string()).collect();
		let mut events = Vec::new();
		for line in text.lines() {
			let trimmed = line.trim();
			if trimmed.starts_with('[') && trimmed.ends_with(']') {
				in_events = trimmed.eq_ignore_ascii_case("[Events]");
				if !header_done {
					header.push_str(trimmed);
					header.push('\n');
				}
				continue;
			}
			if !in_events {
				if !header_done {
					header.push_str(line.trim_end());
					header.push('\n');
				}
				continue;
			}
			if let Some(value) = trimmed.strip_prefix("Format:") {
				format = value.split(',').map(|f| f.trim().to_ascii_lowercase()).collect();
				if !header_done {
					header.push_str(trimmed);
					header.push('\n');
					header_done = true;
				}
			} else if let Some(value) = trimmed.strip_prefix("Dialogue:") {
				if let Some(d) = codec::parse_dialogue(value.trim_start(), &format) {
					events.push(d);
				}
			}
		}
		if !header_done {
			header.push_str(
				"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
			);
		}
		let mut packets: Vec<Packet> = events
			.iter()
			.enumerate()
			.map(|(order, d)| {
				let payload = codec::packet_payload(order as i64, &d.event, &d.text);
				let mut p = Packet::new(StreamId(0), payload.into_bytes());
				p.pts = Some(Time::new(d.start, TIME_BASE));
				p.dts = p.pts;
				p.duration = Some(Time::new(d.end - d.start, TIME_BASE));
				p
			})
			.collect();
		// Stable: events starting together keep their script order.
		packets.sort_by_key(|p| p.pts.map(|t| t.ticks()));
		let end = events.iter().map(|d| d.end).max().unwrap_or(0);
		let mut stream = text_stream("ass", TIME_BASE, header.into_bytes());
		stream.duration = Some(Time::new(end, TIME_BASE));
		Ok(Self {
			packets,
			next: 0,
			streams: StreamSet::new(vec![stream]),
			metadata: Metadata::default(),
			duration: Time::new(end, TIME_BASE),
		})
	}
}

impl Demuxer for AssDemuxer {
	fn read(&mut self) -> Result<Option<Packet>> {
		let Some(p) = self.packets.get(self.next) else { return Ok(None) };
		self.next += 1;
		Ok(Some(p.clone()))
	}

	/// Continues at the first event still showing at `time`.
	fn seek(&mut self, time: f64) -> Result<()> {
		if !time.is_finite() || time < 0.0 {
			return Err(Message::Other("invalid seek time"));
		}
		let t = (time * 100.0) as u64;
		let end = |p: &Packet| p.pts.map_or(0, |t| t.ticks()) + p.duration.map_or(0, |d| d.ticks());
		self.next = self.packets.iter().position(|p| end(p) > t).unwrap_or(self.packets.len());
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

const MINIMAL_HEADER: &str = "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

/// Writes `ass` packets as an `.ass` script. Events are buffered and
/// written in read order at the end, as FFmpeg's `assenc.c` does.
pub struct AssMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	stream: Option<(StreamId, Vec<u8>)>,
	events: Vec<(i64, usize, String)>,
}

impl AssMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self { io, metadata: Metadata::default(), stream: None, events: Vec::new() }
	}
}

impl Muxer for AssMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.stream.is_some() {
			return Err(Message::Container("ass muxer: only one subtitle stream"));
		}
		if stream.kind != StreamKind::Subtitle
			|| !matches!(stream.parameters.codec.id.as_str(), "ass" | "ssa")
		{
			return Err(Message::Container("ass muxer: needs an ass subtitle stream"));
		}
		self.stream = Some((stream.id, stream.parameters.extradata.clone()));
		Ok(0)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if self.stream.as_ref().is_none_or(|(id, _)| *id != packet.stream_id) {
			return Err(Message::Container("ass muxer: packet for unknown stream"));
		}
		let cs = |t: Option<Time>| t.and_then(|t| t.rescale(TIME_BASE)).map(|t| t.ticks());
		let Some(start) = cs(packet.pts.or(packet.dts)) else { return Ok(()) };
		let mut end = start + cs(packet.duration).unwrap_or(0);
		let data = String::from_utf8_lossy(&packet.data);
		let Some((event, text, times)) = codec::parse_packet(&data) else {
			return Err(Message::Container("ass muxer: malformed event packet"));
		};
		let start = match times {
			Some((s, e)) if packet.pts.is_none() => {
				end = e;
				s
			}
			_ => start,
		};
		let line = format!(
			"Dialogue: {},{},{},{},{},{},{},{},{},{}\n",
			event.layer,
			codec::format_time(start),
			codec::format_time(end.max(start)),
			event.style,
			event.name,
			event.margin_l,
			event.margin_r,
			event.margin_v,
			event.effect,
			text
		);
		let order = self.events.len();
		self.events.push((event.read_order, order, line));
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		let header = match &self.stream {
			Some((_, h)) if !h.is_empty() => {
				let mut h = String::from_utf8_lossy(h).into_owned();
				if !h.ends_with('\n') {
					h.push('\n');
				}
				h
			}
			_ => MINIMAL_HEADER.to_string(),
		};
		self.io.write_all(header.as_bytes())?;
		self.events.sort_by_key(|(order, i, _)| (*order, *i));
		for (_, _, line) in self.events.drain(..) {
			self.io.write_all(line.as_bytes())?;
		}
		Ok(())
	}
}

pub struct AssProbe;

impl Probe for AssProbe {
	fn demuxer_matches(data: &[u8]) -> bool {
		looks_like_ass(data)
	}

	fn muxer_matches(hint: &Hint) -> bool {
		hint
			.extension
			.as_deref()
			.is_some_and(|e| e.eq_ignore_ascii_case("ass") || e.eq_ignore_ascii_case("ssa"))
	}

	fn create_demuxer(io: Box<dyn Io>) -> Result<Box<dyn Demuxer>> {
		Ok(Box::new(AssDemuxer::new(io)?))
	}

	fn create_muxer(io: Box<dyn Io>) -> Result<Box<dyn Muxer>> {
		Ok(Box::new(AssMuxer::new(io)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::io::MemoryIo;

	const SCRIPT: &str = "[Script Info]\n; comment\nScriptType: v4.00+\nPlayResX: 640\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,Second\nComment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,ignored\nDialogue: 1,0:00:01.00,0:00:02.50,Default,Bob,0,0,0,,{\\i1}First{\\i0}, with comma\n";

	#[test]
	fn demux_and_round_trip() {
		assert!(looks_like_ass(SCRIPT.as_bytes()));
		let mut d = AssDemuxer::new(Box::new(MemoryIo::from_vec(SCRIPT.as_bytes().to_vec()))).unwrap();
		let header = String::from_utf8(d.streams().streams[0].parameters.extradata.clone()).unwrap();
		assert!(header.starts_with("[Script Info]") && header.ends_with("Effect, Text\n"));
		assert!(!header.contains("Dialogue"));
		let a = d.read().unwrap().unwrap();
		assert_eq!(a.data, b"1,1,Default,Bob,0,0,0,,{\\i1}First{\\i0}, with comma");
		assert_eq!((a.pts.unwrap().ticks(), a.duration.unwrap().ticks()), (100, 150));
		let b = d.read().unwrap().unwrap();
		assert_eq!(b.data, b"0,0,Default,,0,0,0,,Second");
		assert!(d.read().unwrap().is_none());

		let out = MemoryIo::new();
		let mut mux = AssMuxer::new(Box::new(out.clone()));
		mux.add(&d.streams().streams[0].clone()).unwrap();
		mux.write(a).unwrap();
		mux.write(b).unwrap();
		mux.finish().unwrap();
		let text = String::from_utf8(out.contents()).unwrap();
		assert!(text.ends_with("Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,Second\nDialogue: 1,0:00:01.00,0:00:02.50,Default,Bob,0,0,0,,{\\i1}First{\\i0}, with comma\n"));
	}
}
