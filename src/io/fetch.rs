//! Whole-resource reads for streaming manifests and their segments: local
//! files or HTTP(S), optionally a byte range, and relative-reference
//! resolution (RFC 3986 §5.2).

use std::io::{Read, Seek, SeekFrom};

use crate::message::{Message, Result};

/// Largest resource read into memory (a manifest or one segment).
pub const MAX_RESOURCE: u64 = 1 << 30;

/// `length` bytes from `start` (`None`: to the end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteRange {
	pub start: u64,
	pub length: Option<u64>,
}

fn is_url(location: &str) -> bool {
	location.starts_with("http://") || location.starts_with("https://")
}

/// Reads resources, reusing one HTTP connection pool for all of them.
#[derive(Default)]
pub struct Fetcher {
	client: Option<reqwest::blocking::Client>,
}

fn http_error(e: reqwest::Error) -> Message {
	Message::Io(std::io::Error::other(e.to_string()))
}

impl Fetcher {
	pub fn new() -> Self {
		Self::default()
	}

	/// Reads a file or URL (or the given range of it) into memory.
	pub fn fetch(&mut self, location: &str, range: Option<ByteRange>) -> Result<Vec<u8>> {
		if is_url(location) {
			return self.fetch_http(location, range);
		}
		if location.contains("://") {
			return Err(Message::Other("unsupported url scheme"));
		}
		let mut file = std::fs::File::open(location)?;
		let (start, length) = range.map_or((0, None), |r| (r.start, r.length));
		file.seek(SeekFrom::Start(start))?;
		let limit = length.unwrap_or(MAX_RESOURCE + 1).min(MAX_RESOURCE + 1);
		let mut out = Vec::new();
		file.take(limit).read_to_end(&mut out)?;
		if out.len() as u64 > MAX_RESOURCE {
			return Err(Message::Other("resource too large"));
		}
		Ok(out)
	}

	fn fetch_http(&mut self, url: &str, range: Option<ByteRange>) -> Result<Vec<u8>> {
		if self.client.is_none() {
			self.client = Some(reqwest::blocking::Client::builder().build().map_err(http_error)?);
		}
		let client = self.client.as_ref().ok_or(Message::Other("http client unavailable"))?;
		let mut request = client.get(url);
		if let Some(r) = range {
			let value = match r.length {
				Some(0) => return Ok(Vec::new()),
				Some(n) => format!("bytes={}-{}", r.start, r.start + n - 1),
				None => format!("bytes={}-", r.start),
			};
			request = request.header(reqwest::header::RANGE, value);
		}
		let response = request.send().map_err(http_error)?;
		let status = response.status();
		if !status.is_success() {
			return Err(Message::Other("http request failed"));
		}
		let mut out = Vec::new();
		response.take(MAX_RESOURCE + 1).read_to_end(&mut out)?;
		if out.len() as u64 > MAX_RESOURCE {
			return Err(Message::Other("resource too large"));
		}
		// A server that ignores Range answers 200 with the whole resource.
		if let Some(r) = range
			&& status == reqwest::StatusCode::OK
		{
			let start = usize::try_from(r.start).unwrap_or(usize::MAX).min(out.len());
			out.drain(..start);
			if let Some(n) = r.length {
				out.truncate(usize::try_from(n).unwrap_or(usize::MAX));
			}
		}
		Ok(out)
	}
}

/// Resolves `reference` against `base` (a URL or a file path), as a
/// browser resolves a link: absolute references stay as they are.
pub fn resolve(base: &str, reference: &str) -> String {
	if reference.contains("://") {
		return reference.to_string();
	}
	if !is_url(base) {
		if reference.starts_with('/') {
			return reference.to_string();
		}
		let dir = base.rfind(['/', '\\']).map_or("", |i| &base[..=i]);
		return format!("{dir}{reference}");
	}
	// scheme://authority/path?query
	let scheme_end = base.find("://").map_or(0, |i| i + 3);
	let authority_end =
		base[scheme_end..].find(['/', '?', '#']).map_or(base.len(), |i| scheme_end + i);
	let origin = &base[..authority_end];
	if let Some(rest) = reference.strip_prefix("//") {
		return format!("{}{rest}", &base[..scheme_end]);
	}
	let path_end = base[authority_end..].find(['?', '#']).map_or(base.len(), |i| authority_end + i);
	let base_path = &base[authority_end..path_end];
	if reference.is_empty() {
		return base.to_string();
	}
	if reference.starts_with('?') {
		return format!("{origin}{base_path}{reference}");
	}
	let (ref_path, suffix) = match reference.find(['?', '#']) {
		Some(i) => (&reference[..i], &reference[i..]),
		None => (reference, ""),
	};
	let merged = if ref_path.starts_with('/') {
		ref_path.to_string()
	} else {
		let dir = base_path.rfind('/').map_or("/", |i| &base_path[..=i]);
		format!("{dir}{ref_path}")
	};
	format!("{origin}{}{suffix}", remove_dot_segments(&merged))
}

fn remove_dot_segments(path: &str) -> String {
	let mut out: Vec<&str> = Vec::new();
	let segments: Vec<&str> = path.split('/').collect();
	for (i, seg) in segments.iter().enumerate() {
		let last = i + 1 == segments.len();
		match *seg {
			"." => {
				if last {
					out.push("");
				}
			}
			".." => {
				if out.len() > 1 {
					out.pop();
				}
				if last {
					out.push("");
				}
			}
			s => out.push(s),
		}
	}
	let joined = out.join("/");
	if joined.starts_with('/') { joined } else { format!("/{joined}") }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn resolves_like_rfc3986() {
		let base = "http://a/b/c/d;p?q";
		for (r, want) in [
			("g", "http://a/b/c/g"),
			("./g", "http://a/b/c/g"),
			("g/", "http://a/b/c/g/"),
			("/g", "http://a/g"),
			("//g", "http://g"),
			("?y", "http://a/b/c/d;p?y"),
			("g?y", "http://a/b/c/g?y"),
			("..", "http://a/b/"),
			("../g", "http://a/b/g"),
			("../../g", "http://a/g"),
			("../../../g", "http://a/g"),
			("https://cdn/x.ts", "https://cdn/x.ts"),
		] {
			assert_eq!(resolve(base, r), want, "{r}");
		}
		assert_eq!(resolve("media/live/index.m3u8", "seg1.ts"), "media/live/seg1.ts");
		assert_eq!(resolve("index.m3u8", "seg1.ts"), "seg1.ts");
		assert_eq!(resolve("/srv/x/index.m3u8", "/abs/seg.ts"), "/abs/seg.ts");
	}
}
