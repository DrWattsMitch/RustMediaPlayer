//! ASS / SSA (Advanced SubStation Alpha) scripts: the header (`[Script
//! Info]`, styles, the `[Events]` format line) and dialogue events.
//!
//! A stream's extradata is the script header up to and including the
//! `[Events]` `Format:` line; each packet is one event in the Matroska /
//! FFmpeg layout `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`
//! (times live in the packet). FFmpeg: `libavcodec/ass_split.c`,
//! `libavformat/assdec.c`.

use crate::core::AssEvent;

/// A style from `[V4+ Styles]` (ASS) or `[V4 Styles]` (SSA).
#[derive(Debug, Clone, PartialEq)]
pub struct AssStyle {
	pub name: String,
	pub font_name: String,
	pub font_size: f32,
	/// Colours as `0xAABBGGRR` (ASS `&HAABBGGRR`; alpha 0 is opaque).
	pub primary_colour: u32,
	pub secondary_colour: u32,
	pub outline_colour: u32,
	pub back_colour: u32,
	pub bold: bool,
	pub italic: bool,
	pub underline: bool,
	pub strike_out: bool,
	pub scale_x: f32,
	pub scale_y: f32,
	pub spacing: f32,
	pub angle: f32,
	pub border_style: i32,
	pub outline: f32,
	pub shadow: f32,
	/// Numpad alignment (1–9: 1 bottom-left, 5 centre, 9 top-right); SSA's
	/// legacy numbering is converted.
	pub alignment: u8,
	pub margin_l: i32,
	pub margin_r: i32,
	pub margin_v: i32,
	pub encoding: i32,
}

impl Default for AssStyle {
	/// libass's defaults for a style missing from the script.
	fn default() -> Self {
		Self {
			name: "Default".into(),
			font_name: "Arial".into(),
			font_size: 18.0,
			primary_colour: 0x00FF_FFFF,
			secondary_colour: 0x00FF_FFFF,
			outline_colour: 0x0000_0000,
			back_colour: 0x0000_0000,
			bold: false,
			italic: false,
			underline: false,
			strike_out: false,
			scale_x: 100.0,
			scale_y: 100.0,
			spacing: 0.0,
			angle: 0.0,
			border_style: 1,
			outline: 2.0,
			shadow: 2.0,
			alignment: 2,
			margin_l: 20,
			margin_r: 20,
			margin_v: 20,
			encoding: 1,
		}
	}
}

/// What a renderer needs from a script's header.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AssScript {
	/// `ScriptType: v4.00` (SSA) rather than `v4.00+` (ASS).
	pub ssa: bool,
	/// The coordinate space of positions and margins (`PlayResX/Y`).
	pub play_res_x: Option<u32>,
	pub play_res_y: Option<u32>,
	/// 0: smart wrap, 1: end-of-line wrap, 2: no wrap (`\n` breaks), 3:
	/// smart wrap, lower line wider.
	pub wrap_style: u8,
	pub scaled_border_and_shadow: bool,
	pub styles: Vec<AssStyle>,
	/// The `[Events]` `Format:` fields, lower-cased.
	pub event_format: Vec<String>,
}

pub const DEFAULT_EVENT_FORMAT: [&str; 10] =
	["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"];

/// `&HAABBGGRR`, `&HBBGGRR&`, or a decimal number → `0xAABBGGRR`.
pub fn parse_colour(s: &str) -> u32 {
	let s = s.trim().trim_end_matches('&');
	if let Some(hex) = s.strip_prefix("&H").or_else(|| s.strip_prefix("&h")) {
		return u32::from_str_radix(hex, 16).unwrap_or(0);
	}
	s.parse::<i64>().map(|v| v as u32).unwrap_or(0)
}

/// SSA alignment (1–3 bottom, 5–7 top, 9–11 middle) → numpad.
fn ssa_alignment(a: i32) -> u8 {
	match a {
		1..=3 => a as u8,
		5..=7 => (a + 2) as u8,
		9..=11 => (a - 5) as u8,
		_ => 2,
	}
}

fn flag(s: &str) -> bool {
	s.trim().parse::<i32>().is_ok_and(|v| v != 0)
}

fn num<T: std::str::FromStr + Default>(s: &str) -> T {
	s.trim().parse().unwrap_or_default()
}

impl AssScript {
	/// Parses a script header (or a whole script; events are ignored).
	pub fn parse(text: &str) -> Self {
		let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
		let mut script = AssScript {
			event_format: DEFAULT_EVENT_FORMAT.iter().map(|s| s.to_string()).collect(),
			..Default::default()
		};
		let mut section = String::new();
		let mut style_format: Vec<String> = Vec::new();
		for line in text.lines() {
			let line = line.trim();
			if line.starts_with('[') && line.ends_with(']') {
				section = line[1..line.len() - 1].to_ascii_lowercase();
				continue;
			}
			let Some((key, value)) = line.split_once(':') else { continue };
			let (key, value) = (key.trim(), value.trim());
			match section.as_str() {
				"script info" => match key.to_ascii_lowercase().as_str() {
					"scripttype" => script.ssa = !value.contains('+'),
					"playresx" => script.play_res_x = value.parse().ok(),
					"playresy" => script.play_res_y = value.parse().ok(),
					"wrapstyle" => script.wrap_style = value.parse().unwrap_or(0),
					"scaledborderandshadow" => {
						script.scaled_border_and_shadow = value.eq_ignore_ascii_case("yes")
					}
					_ => {}
				},
				"v4+ styles" | "v4 styles" | "v4 styles+" => {
					if section == "v4 styles" {
						script.ssa = true;
					}
					match key {
						"Format" => style_format = fields(value),
						"Style" => {
							let format = if style_format.is_empty() {
								default_style_format(script.ssa)
							} else {
								style_format.clone()
							};
							script.styles.push(parse_style(value, &format, script.ssa));
						}
						_ => {}
					}
				}
				"events" if key == "Format" => script.event_format = fields(value),
				_ => {}
			}
		}
		script
	}

	/// The style called `name` (ASS matches names case-sensitively, with
	/// a leading `*` ignored), falling back to `Default` and then to
	/// libass's defaults.
	pub fn style(&self, name: &str) -> AssStyle {
		let name = name.trim().trim_start_matches('*');
		self
			.styles
			.iter()
			.find(|s| s.name == name)
			.or_else(|| self.styles.iter().find(|s| s.name == "Default"))
			.cloned()
			.unwrap_or_default()
	}
}

fn fields(value: &str) -> Vec<String> {
	value.split(',').map(|f| f.trim().to_ascii_lowercase()).collect()
}

fn default_style_format(ssa: bool) -> Vec<String> {
	let ass = "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
	           Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, \
	           Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding";
	let ssa_fmt = "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, \
	               BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, \
	               MarginR, MarginV, AlphaLevel, Encoding";
	fields(if ssa { ssa_fmt } else { ass })
}

fn parse_style(value: &str, format: &[String], ssa: bool) -> AssStyle {
	let mut style = AssStyle::default();
	// The name may not contain commas; the last field takes the rest.
	let parts: Vec<&str> = value.splitn(format.len().max(1), ',').collect();
	for (key, v) in format.iter().zip(parts) {
		match key.as_str() {
			"name" => style.name = v.trim().to_string(),
			"fontname" => style.font_name = v.trim().to_string(),
			"fontsize" => style.font_size = num(v),
			"primarycolour" => style.primary_colour = parse_colour(v),
			"secondarycolour" => style.secondary_colour = parse_colour(v),
			"outlinecolour" | "tertiarycolour" => style.outline_colour = parse_colour(v),
			"backcolour" => style.back_colour = parse_colour(v),
			"bold" => style.bold = flag(v),
			"italic" => style.italic = flag(v),
			"underline" => style.underline = flag(v),
			"strikeout" => style.strike_out = flag(v),
			"scalex" => style.scale_x = num(v),
			"scaley" => style.scale_y = num(v),
			"spacing" => style.spacing = num(v),
			"angle" => style.angle = num(v),
			"borderstyle" => style.border_style = num(v),
			"outline" => style.outline = num(v),
			"shadow" => style.shadow = num(v),
			"alignment" => {
				let a: i32 = num(v);
				style.alignment = if ssa { ssa_alignment(a) } else { a.clamp(1, 9) as u8 };
			}
			"marginl" => style.margin_l = num(v),
			"marginr" => style.margin_r = num(v),
			"marginv" => style.margin_v = num(v),
			"encoding" => style.encoding = num(v),
			_ => {}
		}
	}
	style
}

/// `H:MM:SS.CC` → centiseconds.
pub fn parse_time(s: &str) -> Option<u64> {
	let s = s.trim();
	let (clock, frac) = s.split_once('.').unwrap_or((s, "0"));
	let parts: Vec<&str> = clock.split(':').collect();
	let [h, m, sec] = parts.as_slice() else { return None };
	let (h, m, sec): (u64, u64, u64) = (h.parse().ok()?, m.parse().ok()?, sec.parse().ok()?);
	// Two digits of hundredths; more digits are truncated, one is tenths.
	let digits: String = frac.chars().take(2).collect();
	let cs = match digits.len() {
		0 => 0,
		1 => digits.parse::<u64>().ok()? * 10,
		_ => digits.parse::<u64>().ok()?,
	};
	Some(((h * 60 + m) * 60 + sec) * 100 + cs)
}

/// `H:MM:SS.CC` from centiseconds, as ASS writes it.
pub fn format_time(cs: u64) -> String {
	format!("{}:{:02}:{:02}.{:02}", cs / 360_000, cs / 6000 % 60, cs / 100 % 60, cs % 100)
}

/// One `Dialogue:` line of a script, split by its `[Events]` format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dialogue {
	/// Centiseconds.
	pub start: u64,
	pub end: u64,
	pub event: AssEvent,
	pub text: String,
}

/// Parses the value of a `Dialogue:` line (after the colon) with the
/// script's event format.
pub fn parse_dialogue(value: &str, format: &[String]) -> Option<Dialogue> {
	let parts: Vec<&str> = value.splitn(format.len().max(1), ',').collect();
	if parts.len() < format.len() {
		return None;
	}
	let (mut start, mut end) = (None, None);
	let mut event = AssEvent::default();
	let mut text = String::new();
	for (key, v) in format.iter().zip(parts) {
		match key.as_str() {
			"layer" => event.layer = num(v),
			"start" => start = parse_time(v),
			"end" => end = parse_time(v),
			"style" => event.style = v.trim().to_string(),
			"name" | "actor" => event.name = v.trim().to_string(),
			"marginl" => event.margin_l = num(v),
			"marginr" => event.margin_r = num(v),
			"marginv" => event.margin_v = num(v),
			"effect" => event.effect = v.trim().to_string(),
			"text" => text = v.to_string(),
			_ => {}
		}
	}
	let start = start?;
	Some(Dialogue { start, end: end.unwrap_or(start).max(start), event, text })
}

/// An event packet's fields, its text, and the centisecond times of a
/// whole `Dialogue:` line.
pub type EventPacket = (AssEvent, String, Option<(u64, u64)>);

/// The packet layout of an event (`ReadOrder,Layer,Style,Name,MarginL,
/// MarginR,MarginV,Effect,Text`).
pub fn packet_payload(read_order: i64, event: &AssEvent, text: &str) -> String {
	format!(
		"{read_order},{},{},{},{},{},{},{},{text}",
		event.layer,
		event.style,
		event.name,
		event.margin_l,
		event.margin_r,
		event.margin_v,
		event.effect
	)
}

/// Splits an event packet into its fields and text. Also accepts a whole
/// `Dialogue: Layer,Start,End,...` line (FFmpeg's layout before 2015, still
/// written by some muxers), whose times are then returned in centiseconds.
pub fn parse_packet(data: &str) -> Option<EventPacket> {
	let data = data.trim_end_matches(['\r', '\n', '\0']);
	if let Some(line) = data.strip_prefix("Dialogue:") {
		let format: Vec<String> = DEFAULT_EVENT_FORMAT.iter().map(|s| s.to_string()).collect();
		let d = parse_dialogue(line.trim_start(), &format)?;
		return Some((d.event, d.text, Some((d.start, d.end))));
	}
	let parts: Vec<&str> = data.splitn(9, ',').collect();
	let [read_order, layer, style, name, ml, mr, mv, effect, text] = parts.as_slice() else {
		return None;
	};
	let event = AssEvent {
		read_order: num(read_order),
		layer: num(layer),
		style: style.trim().to_string(),
		name: name.trim().to_string(),
		margin_l: num(ml),
		margin_r: num(mr),
		margin_v: num(mv),
		effect: effect.trim().to_string(),
	};
	Some((event, text.to_string(), None))
}

#[cfg(test)]
mod tests {
	use super::*;

	const HEADER: &str = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\nWrapStyle: 2\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\nStyle: Sign,Times,36,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,1,0,0,100,100,0,0,1,1,0,8,0,0,15,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";

	#[test]
	fn header() {
		let s = AssScript::parse(HEADER);
		assert!(!s.ssa);
		assert_eq!((s.play_res_x, s.play_res_y, s.wrap_style), (Some(1280), Some(720), 2));
		assert_eq!(s.styles.len(), 2);
		let d = s.style("Default");
		assert!(d.bold && !d.italic);
		assert_eq!((d.font_size, d.back_colour, d.alignment, d.margin_v), (48.0, 0x8000_0000, 2, 30));
		assert_eq!(s.style("*Sign").alignment, 8);
		assert_eq!(s.style("Missing").name, "Default");
		assert_eq!(s.event_format.len(), 10);
	}

	#[test]
	fn ssa_styles() {
		let s = AssScript::parse(
			"[Script Info]\nScriptType: v4.00\n[V4 Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding\nStyle: Default,Arial,20,16777215,65535,65535,0,0,0,1,2,0,6,10,10,10,0,0\n",
		);
		assert!(s.ssa);
		let d = s.style("Default");
		assert_eq!((d.primary_colour, d.alignment), (0xFFFFFF, 8));
	}

	#[test]
	fn events() {
		assert_eq!(parse_time("1:02:03.45"), Some(372_345));
		assert_eq!(parse_time("0:00:01.5"), Some(150));
		assert_eq!(format_time(372_345), "1:02:03.45");
		let (e, text, times) = parse_packet("3,1,Sign,Bob,0,0,15,,{\\an8}Hi, there").unwrap();
		assert_eq!((e.read_order, e.layer, e.style.as_str(), e.name.as_str()), (3, 1, "Sign", "Bob"));
		assert_eq!((text.as_str(), times), ("{\\an8}Hi, there", None));
		let (e, text, times) =
			parse_packet("Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Old, style").unwrap();
		assert_eq!(
			(e.style.as_str(), text.as_str(), times),
			("Default", "Old, style", Some((100, 250)))
		);
		assert_eq!(packet_payload(3, &e, "x"), "3,0,Default,,0,0,0,,x");
	}
}
