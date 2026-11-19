//! Text subtitle decoding: SubRip, WebVTT, ASS/SSA and MP4 timed text
//! (`mov_text`) packets become [`SubtitleFrame`] cues with their timing,
//! their original markup and a plain-text rendering. Drawing them is left
//! to the player; [`ass::AssScript`] parses the styles an ASS renderer
//! needs from the stream's extradata.
//!
//! FFmpeg: `libavcodec/srtdec.c`, `webvttdec.c`, `assdec.c`, `movtextdec.c`
//! (which all convert to ASS; RMP keeps each format's own markup instead).

pub mod ass;
pub mod markup;

use crate::core::{
	CodecState, DecodeOut, Decoder, Frame, Packet, Parameters, State, SubtitleFormat, SubtitleFrame,
	SubtitleState, Time,
};
use crate::message::{Message, Result};

/// The text subtitle codecs RMP decodes, by FFmpeg codec name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleCodec {
	SubRip,
	WebVtt,
	Ass,
	MovText,
	Text,
}

impl SubtitleCodec {
	pub fn from_id(id: &str) -> Option<Self> {
		Some(match id {
			"subrip" | "srt" => Self::SubRip,
			"webvtt" => Self::WebVtt,
			"ass" | "ssa" => Self::Ass,
			"mov_text" => Self::MovText,
			"text" => Self::Text,
			_ => return None,
		})
	}
}

#[derive(Debug, Clone)]
pub struct SubtitleDecoder {
	codec: SubtitleCodec,
}

impl SubtitleDecoder {
	pub fn new(codec: SubtitleCodec) -> Self {
		Self { codec }
	}

	pub fn for_id(id: &str) -> Option<Self> {
		SubtitleCodec::from_id(id).map(Self::new)
	}
}

/// The text of an MP4 timed-text sample: a 16-bit length, then UTF-8 (or
/// UTF-16 with a BOM); style boxes after the text are ignored.
fn mov_text(data: &[u8]) -> Option<String> {
	let len = u16::from_be_bytes([*data.first()?, *data.get(1)?]) as usize;
	let text = data.get(2..2 + len)?;
	if let Some(utf16) = text.strip_prefix(&[0xFE, 0xFF]) {
		let units: Vec<u16> = utf16.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
		return Some(String::from_utf16_lossy(&units));
	}
	Some(String::from_utf8_lossy(text).into_owned())
}

/// The cue text of an MP4 WebVTT (`wvtt`, ISO 14496-30) sample: `vttc`
/// boxes hold a `payl` box with the text; `vtte` is an empty gap. `None`
/// when `data` isn't boxed.
fn boxes(data: &[u8]) -> Option<Vec<(&[u8], &[u8])>> {
	let mut out = Vec::new();
	let mut at = 0;
	while at + 8 <= data.len() {
		let size = u32::from_be_bytes(data[at..at + 4].try_into().ok()?) as usize;
		if size < 8 || at + size > data.len() {
			return None;
		}
		out.push((&data[at + 4..at + 8], &data[at + 8..at + size]));
		at += size;
	}
	(at == data.len()).then_some(out)
}

fn boxed_webvtt(data: &[u8]) -> Option<Vec<String>> {
	let top = boxes(data)?;
	if top.is_empty() || !top.iter().all(|(t, _)| matches!(*t, b"vttc" | b"vtte" | b"vtta")) {
		return None;
	}
	let mut cues = Vec::new();
	for (kind, body) in top {
		if kind == b"vttc" {
			for (k, b) in boxes(body)? {
				if k == b"payl" {
					cues.push(String::from_utf8_lossy(b).into_owned());
				}
			}
		}
	}
	Some(cues)
}

fn cue(
	packet: &Packet,
	markup: String,
	format: SubtitleFormat,
	soft_breaks: bool,
) -> SubtitleFrame {
	let text = match format {
		SubtitleFormat::SubRip => markup::subrip_to_text(&markup),
		SubtitleFormat::WebVtt => markup::webvtt_to_text(&markup),
		SubtitleFormat::Ass => markup::ass_to_text(&markup, soft_breaks),
		SubtitleFormat::Plain => markup.replace("\r\n", "\n").trim_end().to_string(),
	};
	SubtitleFrame { pts: packet.pts, duration: packet.duration, text, markup, format, ass: None }
}

impl Decoder for SubtitleDecoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		State::Subtitle(SubtitleState {
			codec_state: CodecState {
				extradata: parameters.extradata.clone(),
				pts: None,
				dts: None,
				last_pts: None,
			},
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut> {
		let State::Subtitle(st) = state else {
			return Err(Message::Codec("subtitle decoder requires a subtitle state"));
		};
		st.codec_state.pts = packet.pts;
		let mut frames = Vec::new();
		match self.codec {
			SubtitleCodec::SubRip | SubtitleCodec::Text => {
				let markup = String::from_utf8_lossy(&packet.data).into_owned();
				let format = if self.codec == SubtitleCodec::SubRip {
					SubtitleFormat::SubRip
				} else {
					SubtitleFormat::Plain
				};
				if !markup.trim().is_empty() {
					frames.push(cue(&packet, markup, format, false));
				}
			}
			SubtitleCodec::WebVtt => {
				let texts = boxed_webvtt(&packet.data)
					.unwrap_or_else(|| vec![String::from_utf8_lossy(&packet.data).into_owned()]);
				for markup in texts.into_iter().filter(|t| !t.trim().is_empty()) {
					frames.push(cue(&packet, markup, SubtitleFormat::WebVtt, false));
				}
			}
			SubtitleCodec::MovText => {
				if let Some(text) = mov_text(&packet.data).filter(|t| !t.is_empty()) {
					frames.push(cue(&packet, text, SubtitleFormat::Plain, false));
				}
			}
			SubtitleCodec::Ass => {
				let data = String::from_utf8_lossy(&packet.data);
				let Some((event, text, times)) = ass::parse_packet(&data) else {
					return Err(Message::Codec("ass: malformed event packet"));
				};
				let header = String::from_utf8_lossy(&st.codec_state.extradata);
				let soft_breaks = wrap_style(&header) == Some(2);
				let mut frame = cue(&packet, text, SubtitleFormat::Ass, soft_breaks);
				if let (Some((start, end)), None) = (times, packet.pts) {
					let base = crate::core::TimeBase::new(1, 100);
					frame.pts = Some(Time::new(start, base));
					frame.duration = Some(Time::new(end - start, base));
				}
				frame.ass = Some(event);
				frames.push(frame);
			}
		}
		st.codec_state.last_pts = packet.pts;
		Ok(DecodeOut { value: frames.into_iter().map(Frame::Subtitle).collect() })
	}

	fn flush(&self, _state: &mut State) -> Result<DecodeOut> {
		Ok(DecodeOut { value: Vec::new() })
	}
}

/// `WrapStyle` from a script header, without parsing its styles.
fn wrap_style(header: &str) -> Option<u8> {
	header
		.lines()
		.find_map(|l| l.trim().strip_prefix("WrapStyle:"))
		.and_then(|v| v.trim().parse().ok())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::core::{StreamId, TimeBase};

	fn decode(codec: &str, extradata: &[u8], data: &[u8]) -> Vec<SubtitleFrame> {
		let dec = SubtitleDecoder::for_id(codec).unwrap();
		let params = Parameters {
			codec: crate::core::CodecId::new(codec),
			profile: None,
			bitrate: None,
			sample_rate: None,
			channels: None,
			width: None,
			height: None,
			pixel_format: None,
			extradata: extradata.to_vec(),
			bits_per_sample: None,
		};
		let mut st = dec.create_state(&params);
		let mut p = Packet::new(StreamId(0), data.to_vec());
		p.pts = Some(Time::new(1500, TimeBase::new(1, 1000)));
		p.duration = Some(Time::new(1000, TimeBase::new(1, 1000)));
		dec
			.decode(&mut st, p)
			.unwrap()
			.value
			.into_iter()
			.map(|f| match f {
				Frame::Subtitle(s) => s,
				_ => panic!("subtitle expected"),
			})
			.collect()
	}

	#[test]
	fn codecs() {
		let c = decode("subrip", &[], b"<i>Hi</i>\nthere");
		assert_eq!((c[0].text.as_str(), c[0].markup.as_str()), ("Hi\nthere", "<i>Hi</i>\nthere"));
		assert_eq!((c[0].start_seconds(), c[0].end_seconds()), (1.5, 2.5));
		assert!(c[0].is_showing(2.0) && !c[0].is_showing(2.5));
		assert_eq!(c[0].end(), Some(Time::new(2500, TimeBase::new(1, 1000))));

		let c = decode("ass", b"[Script Info]\nWrapStyle: 2\n", b"0,0,Default,,0,0,0,,a\\nb");
		assert_eq!(c[0].text, "a\nb");
		assert_eq!(c[0].ass.as_ref().unwrap().style, "Default");

		assert_eq!(decode("mov_text", &[], b"\x00\x05Hello\x00\x00\x00\x08styl")[0].text, "Hello");
		assert!(decode("mov_text", &[], b"\x00\x00").is_empty());

		let mut wvtt = Vec::new();
		let payl = b"<b>Boxed</b>";
		wvtt.extend((8 + 8 + payl.len() as u32).to_be_bytes());
		wvtt.extend(b"vttc");
		wvtt.extend((8 + payl.len() as u32).to_be_bytes());
		wvtt.extend(b"payl");
		wvtt.extend(payl);
		assert_eq!(decode("webvtt", &[], &wvtt)[0].text, "Boxed");
		assert!(decode("webvtt", &[], b"\x00\x00\x00\x08vtte").is_empty());
		assert_eq!(decode("webvtt", &[], b"Plain &amp; simple")[0].text, "Plain & simple");
	}
}
