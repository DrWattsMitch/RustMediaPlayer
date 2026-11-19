//! MPEG-DASH media presentation descriptions (ISO/IEC 23009-1): the MPD
//! model RMP needs and the expansion of a representation into its segment
//! list (`SegmentTemplate` with `$Number$` or `SegmentTimeline`,
//! `SegmentList`, `SegmentBase` with a `sidx` index).
//!
//! FFmpeg: `libavformat/dashdec.c`.

use super::xml::Element;
use crate::container::segment::{SegmentKey, SegmentRef};
use crate::io::{ByteRange, Fetcher, resolve};
use crate::message::{Message, Result};

/// `PnYnMnDTnHnMnS` (xs:duration) in seconds. Years and months are taken
/// as 365 and 30 days, which MPDs never rely on.
pub fn parse_duration(value: &str) -> Option<f64> {
	let v = value.trim();
	let (negative, v) = match v.strip_prefix('-') {
		Some(rest) => (true, rest),
		None => (false, v),
	};
	let v = v.strip_prefix('P')?;
	let (date, time) = v.split_once('T').unwrap_or((v, ""));
	let mut total = 0.0;
	let mut take = |part: &str, units: &[(char, f64)]| -> Option<()> {
		let mut num = String::new();
		for c in part.chars() {
			if c.is_ascii_digit() || c == '.' {
				num.push(c);
			} else {
				let (_, scale) = units.iter().find(|(u, _)| *u == c)?;
				total += num.parse::<f64>().ok()? * scale;
				num.clear();
			}
		}
		num.is_empty().then_some(())
	};
	take(
		date,
		&[('Y', 365.0 * 86400.0), ('M', 30.0 * 86400.0), ('W', 7.0 * 86400.0), ('D', 86400.0)],
	)?;
	take(time, &[('H', 3600.0), ('M', 60.0), ('S', 1.0)])?;
	Some(if negative { -total } else { total })
}

/// `YYYY-MM-DDThh:mm:ss[.f][Z|±hh:mm]` (xs:dateTime) as seconds since the
/// Unix epoch. A time without a zone is taken as UTC.
pub fn parse_datetime(value: &str) -> Option<f64> {
	let v = value.trim();
	let (date, time) = v.split_once('T')?;
	let mut d = date.splitn(3, '-');
	let year: i64 = d.next()?.parse().ok()?;
	let month: i64 = d.next()?.parse().ok()?;
	let day: i64 = d.next()?.parse().ok()?;
	let (clock, zone) = match time.find(['Z', '+', '-']) {
		Some(i) => (&time[..i], &time[i..]),
		None => (time, ""),
	};
	let mut c = clock.splitn(3, ':');
	let hour: f64 = c.next()?.parse().ok()?;
	let minute: f64 = c.next()?.parse().ok()?;
	let second: f64 = c.next().unwrap_or("0").parse().ok()?;
	let offset = match zone.chars().next() {
		None | Some('Z') => 0.0,
		Some(sign) => {
			let (h, m) = zone[1..].split_once(':').unwrap_or((&zone[1..], "0"));
			let off = h.parse::<f64>().ok()? * 3600.0 + m.parse::<f64>().ok()? * 60.0;
			if sign == '-' { -off } else { off }
		}
	};
	if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
		return None;
	}
	// Days from civil (Howard Hinnant's algorithm).
	let y = if month <= 2 { year - 1 } else { year };
	let era = y.div_euclid(400);
	let yoe = y - era * 400;
	let mp = (month + 9) % 12;
	let doy = (153 * mp + 2) / 5 + day - 1;
	let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
	let days = era * 146_097 + doe - 719_468;
	Some(days as f64 * 86400.0 + hour * 3600.0 + minute * 60.0 + second - offset)
}

/// `a-b` (inclusive) as a [`ByteRange`].
fn parse_range(value: &str) -> Option<ByteRange> {
	let (a, b) = value.trim().split_once('-')?;
	let start: u64 = a.parse().ok()?;
	let end: u64 = b.parse().ok()?;
	(end >= start).then(|| ByteRange { start, length: Some(end - start + 1) })
}

/// Expands `$RepresentationID$`, `$Number$`, `$Bandwidth$`, `$Time$`
/// (each optionally with a `%0Nd` width) and `$$`.
pub fn expand_template(template: &str, id: &str, number: u64, bandwidth: u64, time: u64) -> String {
	let mut out = String::with_capacity(template.len() + 16);
	let mut rest = template;
	while let Some(start) = rest.find('$') {
		out.push_str(&rest[..start]);
		let after = &rest[start + 1..];
		let Some(end) = after.find('$') else {
			out.push_str(&rest[start..]);
			return out;
		};
		let token = &after[..end];
		let (name, format) = token.split_once('%').unwrap_or((token, ""));
		let width = format
			.strip_prefix('0')
			.unwrap_or(format)
			.trim_end_matches(['d', 'i', 'u'])
			.parse()
			.unwrap_or(0);
		match name {
			"" => out.push('$'),
			"RepresentationID" => out.push_str(id),
			"Number" => out.push_str(&format!("{number:0width$}")),
			"Bandwidth" => out.push_str(&format!("{bandwidth:0width$}")),
			"Time" => out.push_str(&format!("{time:0width$}")),
			_ => {
				out.push('$');
				out.push_str(token);
				out.push('$');
			}
		}
		rest = &after[end + 1..];
	}
	out.push_str(rest);
	out
}

/// Segment information merged across Period, AdaptationSet and
/// Representation (lower levels override attributes).
#[derive(Debug, Clone, Default)]
struct SegmentInfo {
	kind: Option<&'static str>,
	media: Option<String>,
	initialization: Option<String>,
	init_range: Option<ByteRange>,
	index_range: Option<ByteRange>,
	start_number: Option<u64>,
	timescale: Option<u64>,
	duration: Option<u64>,
	pto: Option<u64>,
	end_number: Option<u64>,
	/// `S` entries: `t`, `d`, `r`.
	timeline: Option<Vec<(Option<u64>, u64, i64)>>,
	/// `SegmentURL`s: media, mediaRange.
	urls: Vec<(Option<String>, Option<ByteRange>)>,
}

fn merge_level(info: &mut SegmentInfo, parent: &Element) {
	for (name, kind) in
		[("SegmentTemplate", "template"), ("SegmentList", "list"), ("SegmentBase", "base")]
	{
		let Some(el) = parent.child(name) else { continue };
		info.kind = Some(kind);
		let num = |k: &str| el.attr(k).and_then(|v| v.trim().parse::<u64>().ok());
		if let Some(v) = el.attr("media") {
			info.media = Some(v.to_string());
		}
		if let Some(v) = el.attr("initialization") {
			info.initialization = Some(v.to_string());
		}
		if let Some(init) = el.child("Initialization") {
			if let Some(v) = init.attr("sourceURL") {
				info.initialization = Some(v.to_string());
			}
			if let Some(r) = init.attr("range").and_then(parse_range) {
				info.init_range = Some(r);
			}
		}
		if let Some(r) = el.attr("indexRange").and_then(parse_range) {
			info.index_range = Some(r);
		}
		info.start_number = num("startNumber").or(info.start_number);
		info.timescale = num("timescale").or(info.timescale);
		info.duration = num("duration").or(info.duration);
		info.pto = num("presentationTimeOffset").or(info.pto);
		info.end_number = num("endNumber").or(info.end_number);
		if let Some(tl) = el.child("SegmentTimeline") {
			info.timeline = Some(
				tl.children("S")
					.map(|s| {
						let n = |k: &str| s.attr(k).and_then(|v| v.trim().parse::<u64>().ok());
						let r = s.attr("r").and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(0);
						(n("t"), n("d").unwrap_or(0), r)
					})
					.collect(),
			);
		}
		let urls: Vec<_> = el
			.children("SegmentURL")
			.map(|u| (u.attr("media").map(str::to_string), u.attr("mediaRange").and_then(parse_range)))
			.collect();
		if !urls.is_empty() {
			info.urls = urls;
		}
	}
}

/// The context a representation's segments are generated in.
pub struct Context<'a> {
	pub period_start: f64,
	/// `None` for an open-ended (live) period.
	pub period_duration: Option<f64>,
	/// Seconds since the epoch at which segment 0 of the period became
	/// available (`availabilityStartTime` + period start), for live.
	pub live_origin: Option<f64>,
	/// Wall clock, seconds since the epoch (live only).
	pub now: f64,
	pub time_shift_buffer: Option<f64>,
	pub base_url: &'a str,
	/// Numbers segments that have no number of their own (static
	/// presentations renumber across periods).
	pub sequence_base: u64,
}

/// Largest number of segments generated for one representation.
const MAX_SEGMENTS: u64 = 1_000_000;

/// The segments of one representation, with the seconds its media
/// timestamps lead the presentation clock by (`presentationTimeOffset`).
pub fn segments(
	levels: [&Element; 3],
	representation: &Element,
	ctx: &Context,
	fetcher: &mut Fetcher,
) -> Result<(Vec<SegmentRef>, f64)> {
	let mut info = SegmentInfo::default();
	for level in levels {
		merge_level(&mut info, level);
	}
	let id = representation.attr("id").unwrap_or("");
	let bandwidth = representation.attr("bandwidth").and_then(|b| b.parse().ok()).unwrap_or(0);
	let timescale = info.timescale.unwrap_or(1).max(1);
	let pto_seconds = info.pto.unwrap_or(0) as f64 / timescale as f64;
	let base = ctx.base_url;
	// An Initialization with a range but no URL is a range of the media
	// file itself (single-file representations).
	let init = match (&info.initialization, info.init_range) {
		(Some(i), range) => Some((resolve(base, &expand_template(i, id, 0, bandwidth, 0)), range)),
		(None, Some(range)) if info.kind != Some("base") => Some((base.to_string(), Some(range))),
		_ => None,
	};
	let make = |uri: String, range: Option<ByteRange>, duration: f64, sequence: u64| SegmentRef {
		uri,
		range,
		init: init.clone(),
		duration,
		key: SegmentKey::None,
		sequence,
		time_offset: 0.0,
	};
	let period_end_ticks = ctx.period_duration.map(|d| (d * timescale as f64).round() as u64);
	let mut out = Vec::new();
	match info.kind {
		Some("template") | Some("list") if info.timeline.is_some() => {
			let start_number = info.start_number.unwrap_or(1);
			let live_edge = ctx
				.live_origin
				.map(|o| ((ctx.now - o).max(0.0) * timescale as f64) as u64 + info.pto.unwrap_or(0));
			let mut t = 0u64;
			let mut number = start_number;
			let entries = info.timeline.as_deref().unwrap_or(&[]);
			for (i, &(start, d, r)) in entries.iter().enumerate() {
				if let Some(s) = start {
					t = s;
				}
				if d == 0 {
					continue;
				}
				// r = -1: repeat until the next S, the period end or (live) now.
				let until = entries
					.get(i + 1)
					.and_then(|e| e.0)
					.or(period_end_ticks.map(|e| e + info.pto.unwrap_or(0)))
					.or(live_edge);
				let count = if r >= 0 {
					r as u64 + 1
				} else {
					until.map_or(1, |end| end.saturating_sub(t).div_ceil(d))
				};
				for _ in 0..count.min(MAX_SEGMENTS) {
					if live_edge.is_some_and(|edge| t + d > edge) {
						break;
					}
					let uri = match info.kind {
						Some("list") => {
							let (media, range) =
								info.urls.get((number - start_number) as usize).cloned().unwrap_or((None, None));
							let uri = media.map_or_else(|| base.to_string(), |m| resolve(base, &m));
							out.push(make(uri, range, d as f64 / timescale as f64, t));
							t += d;
							number += 1;
							continue;
						}
						_ => resolve(
							base,
							&expand_template(info.media.as_deref().unwrap_or(""), id, number, bandwidth, t),
						),
					};
					// Live timelines are identified by time, which stays
					// stable when the window slides.
					out.push(make(uri, None, d as f64 / timescale as f64, t));
					t += d;
					number += 1;
				}
			}
		}
		Some("template") => {
			let media =
				info.media.as_deref().ok_or(Message::Container("dash: SegmentTemplate without media"))?;
			let d = info
				.duration
				.ok_or(Message::Container("dash: SegmentTemplate without duration or timeline"))?;
			if d == 0 {
				return Err(Message::Container("dash: zero segment duration"));
			}
			let start_number = info.start_number.unwrap_or(1);
			let seconds = d as f64 / timescale as f64;
			let (first, last) = match (ctx.live_origin, ctx.period_duration) {
				(Some(origin), _) => {
					// Segment k is available once it has been fully produced.
					let elapsed = (ctx.now - origin).max(0.0);
					let available = (elapsed / seconds).floor() as u64;
					let window = ctx.time_shift_buffer.map_or(available, |b| (b / seconds).ceil() as u64);
					(available.saturating_sub(window), available)
				}
				(None, Some(duration)) => (0, (duration / seconds).ceil() as u64),
				(None, None) => return Err(Message::Container("dash: presentation duration unknown")),
			};
			let last = match info.end_number {
				Some(end) => last.min(end.saturating_sub(start_number) + 1),
				None => last,
			};
			if last - first > MAX_SEGMENTS {
				return Err(Message::Container("dash: too many segments"));
			}
			for k in first..last {
				let number = start_number + k;
				let time = info.pto.unwrap_or(0) + k * d;
				let uri = resolve(base, &expand_template(media, id, number, bandwidth, time));
				out.push(make(uri, None, seconds, number));
			}
		}
		Some("list") => {
			let seconds =
				info.duration.map_or(ctx.period_duration.unwrap_or(0.0), |d| d as f64 / timescale as f64);
			let start_number = info.start_number.unwrap_or(1);
			for (i, (media, range)) in info.urls.iter().enumerate() {
				let uri = media.as_ref().map_or_else(|| base.to_string(), |m| resolve(base, m));
				out.push(make(uri, *range, seconds, ctx.sequence_base + start_number + i as u64));
			}
		}
		Some("base") if info.index_range.is_some() => {
			// The sidx lists the subsegments of the single file.
			let index = info.index_range.ok_or(Message::Container("dash: missing indexRange"))?;
			let sidx_bytes = fetcher.fetch(base, Some(index))?;
			let sidx = parse_sidx(&sidx_bytes).ok_or(Message::Container("dash: invalid sidx"))?;
			let init_range = info.init_range.unwrap_or(ByteRange { start: 0, length: Some(index.start) });
			let mut offset = index.start + index.length.unwrap_or(0) + sidx.first_offset;
			for (i, (size, duration)) in sidx.references.iter().enumerate() {
				out.push(SegmentRef {
					uri: base.to_string(),
					range: Some(ByteRange { start: offset, length: Some(u64::from(*size)) }),
					init: Some((base.to_string(), Some(init_range))),
					duration: f64::from(*duration) / f64::from(sidx.timescale.max(1)),
					key: SegmentKey::None,
					sequence: ctx.sequence_base + i as u64,
					time_offset: 0.0,
				});
				offset += u64::from(*size);
			}
		}
		// SegmentBase without an index, or BaseURL alone: one segment.
		_ => {
			out.push(make(base.to_string(), None, ctx.period_duration.unwrap_or(0.0), ctx.sequence_base))
		}
	}
	// Nominal durations (`@duration`) overstate a shorter last segment;
	// the period's length is the reference.
	if let Some(limit) = ctx.period_duration {
		let mut elapsed = 0.0;
		for s in &mut out {
			s.duration = s.duration.min((limit - elapsed).max(0.0));
			elapsed += s.duration;
		}
	}
	Ok((out, pto_seconds))
}

/// A `sidx` box (ISO/IEC 14496-12 §8.16.3): media references only.
pub struct Sidx {
	pub timescale: u32,
	pub first_offset: u64,
	/// `(referenced_size, subsegment_duration)`.
	pub references: Vec<(u32, u32)>,
}

pub fn parse_sidx(data: &[u8]) -> Option<Sidx> {
	// The index range may start with other boxes (`styp`); find the sidx.
	let mut at = 0usize;
	let body = loop {
		let size = u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?) as usize;
		let kind = data.get(at + 4..at + 8)?;
		if size < 8 {
			return None;
		}
		if kind == b"sidx" {
			break data.get(at + 8..at + size)?;
		}
		at += size;
	};
	let version = *body.first()?;
	let u32_at = |i: usize| body.get(i..i + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
	let u64_at =
		|i: usize| body.get(i..i + 8).map(|b| u64::from_be_bytes(b.try_into().unwrap_or([0; 8])));
	let timescale = u32_at(8)?;
	let (first_offset, mut at) =
		if version == 0 { (u64::from(u32_at(16)?), 20) } else { (u64_at(20)?, 28) };
	let count = u16::from_be_bytes([*body.get(at + 2)?, *body.get(at + 3)?]);
	at += 4;
	let mut references = Vec::with_capacity(usize::from(count));
	for _ in 0..count {
		let reference = u32_at(at)?;
		let duration = u32_at(at + 4)?;
		// reference_type 1 points at another sidx (hierarchical index); RMP
		// reads single-level indexes, which is what packagers write.
		if reference & 0x8000_0000 != 0 {
			return None;
		}
		references.push((reference & 0x7FFF_FFFF, duration));
		at += 12;
	}
	Some(Sidx { timescale, first_offset, references })
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn durations_and_dates() {
		assert_eq!(parse_duration("PT1H2M3.5S"), Some(3723.5));
		assert_eq!(parse_duration("P1DT0.5S"), Some(86400.5));
		assert_eq!(parse_duration("PT0S"), Some(0.0));
		assert_eq!(parse_duration("1H"), None);
		assert_eq!(parse_datetime("1970-01-01T00:00:00Z"), Some(0.0));
		assert_eq!(parse_datetime("2026-11-18T10:57:00-07:00"), Some(1_795_024_620.0));
		assert_eq!(parse_datetime("2000-03-01T00:00:01.5Z"), Some(951_868_801.5));
	}

	#[test]
	fn templates() {
		assert_eq!(
			expand_template("$RepresentationID$/seg-$Number%05d$.m4s", "v1", 42, 0, 0),
			"v1/seg-00042.m4s"
		);
		assert_eq!(
			expand_template("t$Time$_$Bandwidth$$$.mp4", "a", 1, 128000, 9000),
			"t9000_128000$.mp4"
		);
		assert_eq!(expand_template("x$Unknown$y", "a", 1, 0, 0), "x$Unknown$y");
	}

	#[test]
	fn ranges_and_sidx() {
		assert_eq!(parse_range("100-199"), Some(ByteRange { start: 100, length: Some(100) }));
		assert_eq!(parse_range("5-4"), None);
		let mut sidx = vec![0, 0, 0, 0, b's', b'i', b'd', b'x', 0, 0, 0, 0];
		sidx.extend_from_slice(&1u32.to_be_bytes()); // reference_ID
		sidx.extend_from_slice(&1000u32.to_be_bytes()); // timescale
		sidx.extend_from_slice(&0u32.to_be_bytes()); // earliest_presentation_time
		sidx.extend_from_slice(&16u32.to_be_bytes()); // first_offset
		sidx.extend_from_slice(&[0, 0, 0, 2]);
		for (size, dur) in [(5000u32, 2000u32), (4000, 1500)] {
			sidx.extend_from_slice(&size.to_be_bytes());
			sidx.extend_from_slice(&dur.to_be_bytes());
			sidx.extend_from_slice(&0x9000_0000u32.to_be_bytes());
		}
		let n = sidx.len() as u32;
		sidx[..4].copy_from_slice(&n.to_be_bytes());
		let s = parse_sidx(&sidx).unwrap();
		assert_eq!((s.timescale, s.first_offset), (1000, 16));
		assert_eq!(s.references, vec![(5000, 2000), (4000, 1500)]);
	}
}
