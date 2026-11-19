//! A small XML reader for DASH manifests: elements, attributes, text,
//! comments, CDATA, processing instructions and the predefined and
//! numeric entities. No DTDs or external entities (they are skipped, so
//! nothing is fetched or expanded). Namespace prefixes are dropped from
//! names, since MPD elements are matched by local name.

use crate::message::{Message, Result};

/// Deepest nesting accepted (an MPD is 6 levels deep).
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Element {
	pub name: String,
	pub attributes: Vec<(String, String)>,
	pub children: Vec<Element>,
	pub text: String,
}

impl Element {
	pub fn attr(&self, name: &str) -> Option<&str> {
		self.attributes.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
	}

	pub fn child(&self, name: &str) -> Option<&Element> {
		self.children.iter().find(|c| c.name == name)
	}

	pub fn children<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Element> + 'a {
		self.children.iter().filter(move |c| c.name == name)
	}
}

fn local(name: &str) -> &str {
	name.rsplit(':').next().unwrap_or(name)
}

fn decode_entities(raw: &str) -> String {
	if !raw.contains('&') {
		return raw.to_string();
	}
	let mut out = String::with_capacity(raw.len());
	let mut rest = raw;
	while let Some(at) = rest.find('&') {
		out.push_str(&rest[..at]);
		rest = &rest[at..];
		let Some(end) = rest.find(';').filter(|&e| e <= 12) else {
			out.push('&');
			rest = &rest[1..];
			continue;
		};
		let entity = &rest[1..end];
		let decoded = match entity {
			"lt" => Some('<'),
			"gt" => Some('>'),
			"amp" => Some('&'),
			"quot" => Some('"'),
			"apos" => Some('\''),
			_ => entity
				.strip_prefix("#x")
				.map(|h| u32::from_str_radix(h, 16))
				.or_else(|| entity.strip_prefix('#').map(str::parse))
				.and_then(|n| n.ok())
				.and_then(char::from_u32),
		};
		match decoded {
			Some(c) => {
				out.push(c);
				rest = &rest[end + 1..];
			}
			None => {
				out.push('&');
				rest = &rest[1..];
			}
		}
	}
	out.push_str(rest);
	out
}

struct Parser<'a> {
	s: &'a str,
	at: usize,
}

impl<'a> Parser<'a> {
	fn rest(&self) -> &'a str {
		&self.s[self.at..]
	}

	fn skip_until(&mut self, end: &str) -> Result<()> {
		let i = self.rest().find(end).ok_or(Message::Container("xml: unterminated markup"))?;
		self.at += i + end.len();
		Ok(())
	}

	fn skip_ws(&mut self) {
		let r = self.rest();
		self.at += r.len() - r.trim_start().len();
	}

	/// Skips declarations, comments and processing instructions.
	fn skip_misc(&mut self) -> Result<()> {
		loop {
			self.skip_ws();
			let r = self.rest();
			if r.starts_with("<?") {
				self.skip_until("?>")?;
			} else if r.starts_with("<!--") {
				self.skip_until("-->")?;
			} else if r.starts_with("<!DOCTYPE") || r.starts_with("<!doctype") {
				// An internal subset ends with "]>"; skipped unread.
				let bracket = r.find('[');
				let close = r.find('>');
				match (bracket, close) {
					(Some(b), Some(c)) if b < c => self.skip_until("]>")?,
					_ => self.skip_until(">")?,
				}
			} else {
				return Ok(());
			}
		}
	}

	fn name(&mut self) -> Result<&'a str> {
		let r = self.rest();
		let len =
			r.find(|c: char| c.is_whitespace() || matches!(c, '>' | '/' | '=')).unwrap_or(r.len());
		if len == 0 {
			return Err(Message::Container("xml: expected a name"));
		}
		self.at += len;
		Ok(&r[..len])
	}

	fn element(&mut self, depth: usize) -> Result<Element> {
		if depth > MAX_DEPTH {
			return Err(Message::Container("xml: nesting too deep"));
		}
		if !self.rest().starts_with('<') {
			return Err(Message::Container("xml: expected an element"));
		}
		self.at += 1;
		let raw_name = self.name()?;
		let mut el = Element { name: local(raw_name).to_string(), ..Default::default() };
		loop {
			self.skip_ws();
			let r = self.rest();
			if let Some(after) = r.strip_prefix("/>") {
				self.at = self.s.len() - after.len();
				return Ok(el);
			}
			if r.starts_with('>') {
				self.at += 1;
				break;
			}
			let key = self.name()?;
			self.skip_ws();
			if !self.rest().starts_with('=') {
				return Err(Message::Container("xml: attribute without a value"));
			}
			self.at += 1;
			self.skip_ws();
			let quote = self.rest().chars().next().filter(|&c| c == '"' || c == '\'');
			let Some(quote) = quote else { return Err(Message::Container("xml: unquoted attribute")) };
			self.at += 1;
			let end = self.rest().find(quote).ok_or(Message::Container("xml: unterminated attribute"))?;
			let value = decode_entities(&self.rest()[..end]);
			self.at += end + 1;
			// xmlns declarations aren't attributes of the document.
			if key != "xmlns" && !key.starts_with("xmlns:") {
				el.attributes.push((local(key).to_string(), value));
			}
		}
		// Content up to the matching end tag.
		loop {
			let r = self.rest();
			let lt = r.find('<').ok_or(Message::Container("xml: unterminated element"))?;
			el.text.push_str(&decode_entities(&r[..lt]));
			self.at += lt;
			let r = self.rest();
			if r.starts_with("</") {
				self.at += 2;
				let name = self.name()?;
				if local(name) != el.name {
					return Err(Message::Container("xml: mismatched end tag"));
				}
				self.skip_until(">")?;
				el.text = el.text.trim().to_string();
				return Ok(el);
			} else if r.starts_with("<!--") {
				self.skip_until("-->")?;
			} else if let Some(cdata) = r.strip_prefix("<![CDATA[") {
				let end = cdata.find("]]>").ok_or(Message::Container("xml: unterminated CDATA"))?;
				el.text.push_str(&cdata[..end]);
				self.at += 9 + end + 3;
			} else if r.starts_with("<?") {
				self.skip_until("?>")?;
			} else {
				el.children.push(self.element(depth + 1)?);
			}
		}
	}
}

/// Parses a document and returns its root element.
pub fn parse(text: &str) -> Result<Element> {
	let text = text.strip_prefix('\u{FEFF}').unwrap_or(text);
	let mut p = Parser { s: text, at: 0 };
	p.skip_misc()?;
	p.element(0)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_documents() {
		let doc = r#"<?xml version="1.0" encoding="utf-8"?>
<!-- generated -->
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" xmlns:cenc="urn:mpeg:cenc:2013" type='static'>
  <BaseURL>http://a/b?x=1&amp;y=2</BaseURL>
  <Period><AdaptationSet mimeType="video/mp4"><cenc:pssh>AAAA</cenc:pssh><Representation id="v&#49;" bandwidth="100"/></AdaptationSet></Period>
  <Note><![CDATA[a <b> c]]></Note>
</MPD>"#;
		let root = parse(doc).unwrap();
		assert_eq!(root.name, "MPD");
		assert_eq!(root.attr("type"), Some("static"));
		assert!(root.attr("xmlns").is_none());
		assert_eq!(root.child("BaseURL").unwrap().text, "http://a/b?x=1&y=2");
		let set = root.child("Period").unwrap().child("AdaptationSet").unwrap();
		assert_eq!(set.child("pssh").unwrap().text, "AAAA");
		assert_eq!(set.child("Representation").unwrap().attr("id"), Some("v1"));
		assert_eq!(root.child("Note").unwrap().text, "a <b> c");
	}

	#[test]
	fn rejects_broken_documents() {
		assert!(parse("<a><b></a>").is_err());
		assert!(parse("<a x=1/>").is_err());
		assert!(parse("<a>").is_err());
		let deep = "<a>".repeat(100) + &"</a>".repeat(100);
		assert!(parse(&deep).is_err());
	}
}
