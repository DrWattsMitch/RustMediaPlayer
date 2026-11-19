//! MPEG-DASH (ISO/IEC 23009-1): `.mpd` manifests over fragmented MP4,
//! WebM or MPEG-TS segments.
//!
//! [`open`] reads the MPD, picks one video representation (the highest
//! bandwidth by default, see [`StreamingOptions`]) and one audio
//! representation (preferred language, else the `main` role, else the
//! first set; the highest bandwidth in it), and returns a
//! [`SegmentedDemuxer`] over their segments. Static presentations with
//! several periods play them in order, with timestamps continued across
//! period boundaries. Dynamic (live) presentations start near the live
//! edge and are re-read as they advance.
//!
//! FFmpeg's `dash` demuxer exposes every representation as a stream; RMP
//! plays one per kind, as a player does.

pub mod mpd;
pub mod xml;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use self::xml::Element;
use crate::container::segment::{Refresh, SegmentedDemuxer, StreamingOptions, TrackSpec};
use crate::core::Metadata;
use crate::io::{Fetcher, resolve};
use crate::message::{Message, Result};

/// `true` when `data` looks like the start of an MPD document.
pub fn looks_like_mpd(data: &[u8]) -> bool {
	let head = String::from_utf8_lossy(&data[..data.len().min(4096)]);
	let head = head.trim_start_matches('\u{FEFF}').trim_start();
	head.starts_with('<') && (head.contains("<MPD") || head.contains(":MPD"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
	Video,
	Audio,
}

fn now() -> f64 {
	SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn content_kind(set: &Element) -> Option<Kind> {
	let rep = set.child("Representation");
	let attr = |k: &str| set.attr(k).or_else(|| rep.and_then(|r| r.attr(k)));
	let by_name = |s: &str| {
		if s.starts_with("video") {
			Some(Kind::Video)
		} else if s.starts_with("audio") {
			Some(Kind::Audio)
		} else {
			None
		}
	};
	if let Some(k) =
		attr("contentType").and_then(by_name).or_else(|| attr("mimeType").and_then(by_name))
	{
		return Some(k);
	}
	let codecs = attr("codecs")?;
	let video = ["avc", "hvc", "hev", "av01", "vp0", "vp8", "vp9", "dvh"];
	let audio = ["mp4a", "opus", "ac-3", "ec-3", "flac", "vorbis", "mp3"];
	if video.iter().any(|p| codecs.starts_with(p)) {
		Some(Kind::Video)
	} else if audio.iter().any(|p| codecs.starts_with(p)) {
		Some(Kind::Audio)
	} else {
		None
	}
}

/// Trick-play sets (I-frame tracks for fast scrubbing) are not for playback.
fn is_trick_mode(set: &Element) -> bool {
	set
		.children("EssentialProperty")
		.any(|p| p.attr("schemeIdUri").is_some_and(|s| s.contains("trickmode")))
}

fn protected(el: &Element) -> bool {
	el.child("ContentProtection").is_some()
}

fn bandwidth(rep: &Element) -> u64 {
	rep.attr("bandwidth").and_then(|b| b.parse().ok()).unwrap_or(0)
}

fn first_base_url(el: &Element) -> Option<&str> {
	el.child("BaseURL").map(|b| b.text.as_str()).filter(|t| !t.is_empty())
}

fn join_base(base: &str, el: &Element) -> String {
	first_base_url(el).map_or_else(|| base.to_string(), |b| resolve(base, b))
}

/// Picks `(adaptation set, representation)` of a kind in one period.
fn choose<'a>(
	period: &'a Element,
	kind: Kind,
	options: &StreamingOptions,
) -> Result<Option<(&'a Element, &'a Element)>> {
	let sets: Vec<&Element> = period
		.children("AdaptationSet")
		.filter(|s| content_kind(s) == Some(kind) && !is_trick_mode(s))
		.collect();
	let set = match kind {
		Kind::Video => sets.first().copied(),
		Kind::Audio => sets
			.iter()
			.find(|s| options.language_matches(s.attr("lang")))
			.or_else(|| sets.iter().find(|s| s.children("Role").any(|r| r.attr("value") == Some("main"))))
			.or(sets.first())
			.copied(),
	};
	let Some(set) = set else { return Ok(None) };
	let reps: Vec<&Element> = set.children("Representation").collect();
	let index = match kind {
		Kind::Video => {
			let candidates: Vec<(u64, Option<u32>)> = reps
				.iter()
				.map(|r| {
					(bandwidth(r), r.attr("height").or(set.attr("height")).and_then(|h| h.parse().ok()))
				})
				.collect();
			options.choose(&candidates)
		}
		Kind::Audio => reps.iter().enumerate().max_by_key(|(_, r)| bandwidth(r)).map(|(i, _)| i),
	};
	let Some(rep) = index.map(|i| reps[i]) else { return Ok(None) };
	if protected(set) || protected(rep) {
		return Err(Message::Container("dash: encrypted (DRM) content is not supported"));
	}
	Ok(Some((set, rep)))
}

/// Builds one track per kind from an MPD document.
fn build_tracks(
	root: &Element,
	location: &str,
	options: &StreamingOptions,
	fetcher: &mut Fetcher,
	wall_clock: f64,
) -> Result<Vec<(Kind, TrackSpec)>> {
	let dynamic = root.attr("type") == Some("dynamic");
	let total = root.attr("mediaPresentationDuration").and_then(mpd::parse_duration);
	let ast = root.attr("availabilityStartTime").and_then(mpd::parse_datetime);
	let shift_buffer = root.attr("timeShiftBufferDepth").and_then(mpd::parse_duration);
	let mpd_base = join_base(location, root);
	let periods: Vec<&Element> = root.children("Period").collect();
	if periods.is_empty() {
		return Err(Message::Container("dash: mpd has no period"));
	}
	// Period starts: explicit, else the end of the previous one.
	let mut starts = Vec::with_capacity(periods.len());
	let mut next_start = 0.0;
	for p in &periods {
		let start = p.attr("start").and_then(mpd::parse_duration).unwrap_or(next_start);
		next_start = start + p.attr("duration").and_then(mpd::parse_duration).unwrap_or(0.0);
		starts.push(start);
	}
	// Live: only the current (last) period is played.
	let first = if dynamic { periods.len() - 1 } else { 0 };
	let mut tracks: Vec<(Kind, TrackSpec)> = Vec::new();
	let mut origin_shift: Option<f64> = None;
	for (i, period) in periods.iter().enumerate().skip(first) {
		let duration = period
			.attr("duration")
			.and_then(mpd::parse_duration)
			.or_else(|| starts.get(i + 1).map(|next| next - starts[i]))
			.or_else(|| total.map(|t| t - starts[i]));
		let period_base = join_base(&mpd_base, period);
		for kind in [Kind::Video, Kind::Audio] {
			let Some((set, rep)) = choose(period, kind, options)? else { continue };
			let base = join_base(&join_base(&period_base, set), rep);
			let sequence_base = tracks
				.iter()
				.find(|(k, _)| *k == kind)
				.map_or(0, |(_, t)| t.segments.last().map_or(0, |s| s.sequence + 1));
			let ctx = mpd::Context {
				period_start: starts[i],
				period_duration: if dynamic { None } else { duration },
				live_origin: if dynamic { ast.map(|a| a + starts[i]) } else { None },
				now: wall_clock,
				time_shift_buffer: shift_buffer,
				base_url: &base,
				sequence_base,
			};
			let (mut segments, pto) = mpd::segments([period, set, rep], rep, &ctx, fetcher)?;
			// Later periods continue the first one's timeline: their media
			// time (minus presentationTimeOffset) is moved to the period
			// start, measured from where the first period's media began.
			let lead = starts[i] - pto;
			let shift = lead - *origin_shift.get_or_insert(lead);
			for s in &mut segments {
				s.time_offset = shift;
			}
			let mut metadata = vec![("variant_bitrate".to_string(), bandwidth(rep).to_string())];
			if let Some(lang) = set.attr("lang") {
				metadata.push(("language".to_string(), lang.to_string()));
			}
			match tracks.iter_mut().find(|(k, _)| *k == kind) {
				Some((_, track)) => track.segments.extend(segments),
				None => tracks.push((
					kind,
					TrackSpec {
						segments,
						start: 0,
						ended: !dynamic,
						refresh: None,
						poll: Duration::from_secs(2),
						metadata,
					},
				)),
			}
		}
	}
	if tracks.is_empty() {
		return Err(Message::Container("dash: no audio or video representation"));
	}
	Ok(tracks)
}

/// Opens a DASH presentation from the location of its MPD.
pub fn open(location: &str, options: &StreamingOptions) -> Result<SegmentedDemuxer> {
	let mut fetcher = Fetcher::new();
	let text = String::from_utf8_lossy(&fetcher.fetch(location, None)?).into_owned();
	let root = xml::parse(&text)?;
	if root.name != "MPD" {
		return Err(Message::Container("dash: document is not an MPD"));
	}
	let dynamic = root.attr("type") == Some("dynamic");
	let update = root.attr("minimumUpdatePeriod").and_then(mpd::parse_duration).unwrap_or(2.0);
	let mut tracks = build_tracks(&root, location, options, &mut fetcher, now())?;
	for (kind, spec) in &mut tracks {
		if !dynamic {
			continue;
		}
		spec.start = spec.segments.len().saturating_sub(options.live_start);
		spec.poll = Duration::from_secs_f64(update.clamp(0.5, 30.0));
		// A refresh re-reads the MPD and regenerates this kind's segments.
		let (location, options, kind) = (location.to_string(), options.clone(), *kind);
		let refresh: Refresh = Box::new(move |fetcher: &mut Fetcher| {
			let text = String::from_utf8_lossy(&fetcher.fetch(&location, None)?).into_owned();
			let root = xml::parse(&text)?;
			let ended = root.attr("type") != Some("dynamic");
			let tracks = build_tracks(&root, &location, &options, fetcher, now())?;
			let segments =
				tracks.into_iter().find(|(k, _)| *k == kind).map(|(_, t)| t.segments).unwrap_or_default();
			Ok((segments, ended))
		});
		spec.refresh = Some(refresh);
	}
	let mut metadata = Metadata::default();
	if let Some(title) = root.child("ProgramInformation").and_then(|p| p.child("Title")) {
		metadata.title = Some(title.text.clone());
	}
	SegmentedDemuxer::open(tracks.into_iter().map(|(_, t)| t).collect(), metadata, fetcher)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn detects_mpd() {
		assert!(looks_like_mpd(
			b"<?xml version=\"1.0\"?>\n<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\">"
		));
		assert!(!looks_like_mpd(b"#EXTM3U"));
	}

	#[test]
	fn static_template_tracks() {
		let doc = r#"<MPD type="static" mediaPresentationDuration="PT10S">
<Period><AdaptationSet contentType="video"><SegmentTemplate timescale="1000" duration="4000" startNumber="1" media="v_$RepresentationID$_$Number$.m4s" initialization="v_$RepresentationID$_init.mp4"/>
<Representation id="lo" bandwidth="100000" height="240"/><Representation id="hi" bandwidth="900000" height="720"/></AdaptationSet>
<AdaptationSet mimeType="audio/mp4" lang="fr"><SegmentTemplate timescale="48000" media="a_$Time$.m4s" initialization="a_init.mp4"><SegmentTimeline><S t="0" d="96000" r="2"/><S d="48000"/></SegmentTimeline></SegmentTemplate><Representation id="a" bandwidth="64000"/></AdaptationSet>
<AdaptationSet mimeType="audio/mp4" lang="en"><SegmentTemplate media="x" duration="1"/><Representation id="e" bandwidth="64000"/></AdaptationSet>
</Period></MPD>"#;
		let root = xml::parse(doc).unwrap();
		let mut f = Fetcher::new();
		let options = StreamingOptions { audio_language: Some("fr".into()), ..Default::default() };
		let tracks = build_tracks(&root, "http://h/p/manifest.mpd", &options, &mut f, 0.0).unwrap();
		let video = &tracks[0].1;
		assert_eq!(video.segments.len(), 3);
		assert_eq!(video.segments[0].uri, "http://h/p/v_hi_1.m4s");
		assert_eq!(video.segments[2].uri, "http://h/p/v_hi_3.m4s");
		assert_eq!(video.segments[0].init.as_ref().unwrap().0, "http://h/p/v_hi_init.mp4");
		let audio = &tracks[1].1;
		let uris: Vec<_> = audio.segments.iter().map(|s| s.uri.as_str()).collect();
		assert_eq!(
			uris,
			[
				"http://h/p/a_0.m4s",
				"http://h/p/a_96000.m4s",
				"http://h/p/a_192000.m4s",
				"http://h/p/a_288000.m4s"
			]
		);
		let limited = StreamingOptions { max_height: Some(480), ..Default::default() };
		let tracks = build_tracks(&root, "m.mpd", &limited, &mut f, 0.0).unwrap();
		assert_eq!(tracks[0].1.segments[0].uri, "v_lo_1.m4s");
	}

	#[test]
	fn live_number_template() {
		let doc = r#"<MPD type="dynamic" availabilityStartTime="1970-01-01T00:00:00Z" timeShiftBufferDepth="PT20S">
<Period start="PT0S"><AdaptationSet mimeType="video/mp4"><SegmentTemplate duration="2" media="s$Number$.m4s" startNumber="0"/><Representation id="v" bandwidth="1"/></AdaptationSet></Period></MPD>"#;
		let root = xml::parse(doc).unwrap();
		let tracks =
			build_tracks(&root, "m.mpd", &StreamingOptions::default(), &mut Fetcher::new(), 101.0)
				.unwrap();
		let segs = &tracks[0].1.segments;
		// 50 segments complete by t=101; a 20 s window holds the last 10.
		assert_eq!(segs.len(), 10);
		assert_eq!(segs.last().unwrap().uri, "s49.m4s");
	}

	#[test]
	fn drm_is_refused() {
		let doc = r#"<MPD type="static" mediaPresentationDuration="PT4S"><Period><AdaptationSet mimeType="video/mp4"><ContentProtection schemeIdUri="urn:mpeg:dash:mp4protection:2011"/><Representation id="v" bandwidth="1"><BaseURL>v.mp4</BaseURL></Representation></AdaptationSet></Period></MPD>"#;
		let root = xml::parse(doc).unwrap();
		assert!(
			build_tracks(&root, "m.mpd", &StreamingOptions::default(), &mut Fetcher::new(), 0.0).is_err()
		);
	}
}
