//! Markup → plain text for the text subtitle formats.
//!
//! FFmpeg: `libavcodec/htmlsubtitles.c` (SubRip), `webvttdec.c`, and
//! libass's override-tag syntax for ASS.

/// Resolves an HTML character reference at the start of `s` (after the
/// `&`): returns the character and the bytes consumed, `;` included.
fn entity(s: &str) -> Option<(char, usize)> {
	let end = s.find(';').filter(|&e| e > 0 && e <= 10)?;
	let name = &s[..end];
	let c = match name {
		"amp" => '&',
		"lt" => '<',
		"gt" => '>',
		"quot" => '"',
		"apos" => '\'',
		"nbsp" => '\u{a0}',
		"lrm" => '\u{200e}',
		"rlm" => '\u{200f}',
		_ => {
			let n = name.strip_prefix('#')?;
			let code = match n.strip_prefix(['x', 'X']) {
				Some(hex) => u32::from_str_radix(hex, 16).ok()?,
				None => n.parse::<u32>().ok()?,
			};
			char::from_u32(code)?
		}
	};
	Some((c, end + 1))
}

/// Whether `s` (just after a `<`) opens an HTML-like tag: an optional `/`,
/// a letter, and a `>` before the next `<` or line end. Returns the tag's
/// length after the `<`, `>` included.
fn tag_len(s: &str, any_start: bool) -> Option<usize> {
	let body = s.strip_prefix('/').unwrap_or(s);
	let first = body.chars().next()?;
	if !any_start && !first.is_ascii_alphabetic() {
		return None;
	}
	for (i, c) in s.char_indices() {
		match c {
			'>' => return Some(i + 1),
			'<' | '\n' => return None,
			_ => {}
		}
	}
	None
}

/// SubRip: drops `<b>`/`<i>`/`<u>`/`<s>`/`<font ...>` (and any other
/// HTML-like tag), `{\an8}`-style override blocks, and resolves character
/// references. A `<` that doesn't open a tag is kept.
pub fn subrip_to_text(markup: &str) -> String {
	let markup = markup.replace("\r\n", "\n");
	let mut out = String::with_capacity(markup.len());
	let mut rest = markup.as_str();
	while let Some(c) = rest.chars().next() {
		let after = &rest[c.len_utf8()..];
		match c {
			'<' => {
				if let Some(n) = tag_len(after, false) {
					rest = &after[n..];
					continue;
				}
			}
			'{' if after.starts_with('\\') => {
				if let Some(end) = after.find('}') {
					rest = &after[end + 1..];
					continue;
				}
			}
			'&' => {
				if let Some((ch, n)) = entity(after) {
					out.push(ch);
					rest = &after[n..];
					continue;
				}
			}
			_ => {}
		}
		out.push(c);
		rest = after;
	}
	tidy(&out)
}

/// WebVTT cue text: every `<...>` is a tag (a literal `<` must be written
/// `&lt;`): class, voice, language, ruby and timestamp tags are dropped
/// along with `b`/`i`/`u`; character references are resolved.
pub fn webvtt_to_text(markup: &str) -> String {
	let markup = markup.replace("\r\n", "\n");
	let mut out = String::with_capacity(markup.len());
	let mut rest = markup.as_str();
	while let Some(c) = rest.chars().next() {
		let after = &rest[c.len_utf8()..];
		match c {
			'<' => {
				if let Some(n) = tag_len(after, true) {
					rest = &after[n..];
					continue;
				}
			}
			'&' => {
				if let Some((ch, n)) = entity(after) {
					out.push(ch);
					rest = &after[n..];
					continue;
				}
			}
			_ => {}
		}
		out.push(c);
		rest = after;
	}
	tidy(&out)
}

/// ASS dialogue text: drops `{...}` override blocks and the text drawn in
/// vector-drawing mode (`{\p1}` ... `{\p0}`); `\N` is a line break, `\n` a
/// line break with `soft_breaks` (`WrapStyle: 2`) and a space otherwise,
/// `\h` a non-breaking space.
pub fn ass_to_text(markup: &str, soft_breaks: bool) -> String {
	let mut out = String::with_capacity(markup.len());
	let mut drawing = false;
	let mut rest = markup;
	while let Some(c) = rest.chars().next() {
		let after = &rest[c.len_utf8()..];
		match c {
			'{' => {
				if let Some(end) = after.find('}') {
					if let Some(level) = drawing_level(&after[..end]) {
						drawing = level > 0;
					}
					rest = &after[end + 1..];
					continue;
				}
			}
			'\\' => {
				let esc = match after.chars().next() {
					Some('N') => Some('\n'),
					Some('n') => Some(if soft_breaks { '\n' } else { ' ' }),
					Some('h') => Some('\u{a0}'),
					_ => None,
				};
				if let Some(e) = esc {
					if !drawing {
						out.push(e);
					}
					rest = &after[1..];
					continue;
				}
			}
			_ => {}
		}
		if !drawing {
			out.push(c);
		}
		rest = after;
	}
	tidy(&out)
}

/// The last `\pN` (drawing mode) tag in an override block, if any.
fn drawing_level(block: &str) -> Option<u32> {
	let mut level = None;
	for tag in block.split('\\').skip(1) {
		if let Some(n) = tag.strip_prefix('p') {
			let digits: String = n.chars().take_while(char::is_ascii_digit).collect();
			if !digits.is_empty() && digits.len() == n.trim_end().len() {
				level = digits.parse().ok();
			}
		}
	}
	level
}

/// Trims trailing whitespace from each line, and blank lines from both ends.
fn tidy(s: &str) -> String {
	let lines: Vec<&str> = s.split('\n').map(|l| l.trim_end_matches([' ', '\t', '\r'])).collect();
	let first = lines.iter().position(|l| !l.is_empty()).unwrap_or(lines.len());
	let last = lines.iter().rposition(|l| !l.is_empty()).map_or(first, |i| i + 1);
	lines[first..last].join("\n")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn subrip() {
		assert_eq!(
			subrip_to_text("<i>Hello</i>\n<font color=\"#ff0000\">world</font>"),
			"Hello\nworld"
		);
		assert_eq!(subrip_to_text("{\\an8}Top &amp; a < b"), "Top & a < b");
		assert_eq!(subrip_to_text("x <3 y\r\n"), "x <3 y");
	}

	#[test]
	fn webvtt() {
		assert_eq!(
			webvtt_to_text("<v Roger Bingham>We&apos;re <c.loud>in</c> <00:00:01.500>space&nbsp;now"),
			"We're in space\u{a0}now"
		);
		assert_eq!(webvtt_to_text("1 &lt; 2 &#x41;&#66;"), "1 < 2 AB");
	}

	#[test]
	fn ass() {
		assert_eq!(ass_to_text("{\\an8\\b1}Hello\\Nworld", false), "Hello\nworld");
		assert_eq!(ass_to_text("soft\\nbreak\\hx", false), "soft break\u{a0}x");
		assert_eq!(ass_to_text("soft\\nbreak", true), "soft\nbreak");
		assert_eq!(ass_to_text("a{\\p1}m 0 0 l 10 0{\\p0}b", false), "ab");
		assert_eq!(ass_to_text("{\\pos(10,20)}keep", false), "keep");
	}
}
