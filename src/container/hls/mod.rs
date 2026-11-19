//! HTTP Live Streaming (RFC 8216): `.m3u8` playlists over MPEG-TS,
//! fragmented MP4 or packed-audio segments.
//!
//! [`open`] reads a multivariant or media playlist, picks a variant (the
//! highest bandwidth by default, see [`StreamingOptions`]) and its audio
//! rendition, and its WebVTT subtitle renditions, and returns a
//! [`SegmentedDemuxer`] over their segments. AES-128 segments are
//! decrypted, SAMPLE-AES samples too; live playlists are reloaded as they
//! advance.
//!
//! FFmpeg's `hls` demuxer opens every variant at once and exposes all of
//! their streams; RMP plays one variant, as a player does.

pub mod aes;
pub mod muxer;
pub mod playlist;
pub mod sample_aes;

use std::time::Duration;

use self::playlist::{Key, MediaPlaylist, MediaType, Playlist};
use crate::container::segment::{
	SegmentKey, SegmentRef, SegmentedDemuxer, StreamingOptions, TrackSpec,
};
use crate::core::{Metadata, RawValue};
use crate::io::Fetcher;
use crate::message::{Message, Result};

pub use playlist::looks_like_m3u8;

fn fetch_playlist(fetcher: &mut Fetcher, uri: &str) -> Result<Playlist> {
	let bytes = fetcher.fetch(uri, None)?;
	playlist::parse(&String::from_utf8_lossy(&bytes), uri)
}

fn fetch_media(fetcher: &mut Fetcher, uri: &str) -> Result<MediaPlaylist> {
	match fetch_playlist(fetcher, uri)? {
		Playlist::Media(m) => Ok(m),
		Playlist::Multivariant(_) => Err(Message::Container("hls: expected a media playlist")),
	}
}

/// The segments of a media playlist, in the form the segment demuxer
/// takes. Gaps are dropped; keys RMP can't use are an error.
fn segment_refs(playlist: &MediaPlaylist) -> Result<Vec<SegmentRef>> {
	let mut out = Vec::with_capacity(playlist.segments.len());
	for s in playlist.segments.iter().filter(|s| !s.gap) {
		let key = match &s.key {
			Key::None => SegmentKey::None,
			Key::Aes128 { uri, iv } => SegmentKey::Aes128 {
				uri: uri.clone(),
				// Without an IV attribute, the media sequence number is the IV.
				iv: iv.unwrap_or_else(|| u128::from(s.sequence).to_be_bytes()),
			},
			Key::SampleAes { uri, iv } => SegmentKey::SampleAes {
				uri: uri.clone(),
				iv: iv.unwrap_or_else(|| u128::from(s.sequence).to_be_bytes()),
			},
			Key::Unsupported(_) => {
				return Err(Message::Container(
					"hls: only AES-128 and SAMPLE-AES (identity key) encryption are supported, not DRM",
				));
			}
		};
		out.push(SegmentRef {
			uri: s.uri.clone(),
			range: s.range,
			init: s.map.as_ref().map(|m| (m.uri.clone(), m.range)),
			duration: s.duration,
			key,
			sequence: s.sequence,
			time_offset: 0.0,
		});
	}
	Ok(out)
}

fn track(
	playlist: MediaPlaylist,
	uri: String,
	options: &StreamingOptions,
	metadata: Vec<(String, String)>,
) -> Result<TrackSpec> {
	let segments = segment_refs(&playlist)?;
	let start = if playlist.ended { 0 } else { segments.len().saturating_sub(options.live_start) };
	// RFC 8216 §6.3.4: reload no sooner than half the target duration.
	let poll = Duration::from_secs_f64((playlist.target_duration / 2.0).clamp(0.5, 30.0));
	let refresh: crate::container::segment::Refresh = Box::new(move |fetcher: &mut Fetcher| {
		let p = fetch_media(fetcher, &uri)?;
		Ok((segment_refs(&p)?, p.ended))
	});
	Ok(TrackSpec { segments, start, ended: playlist.ended, refresh: Some(refresh), poll, metadata })
}

/// Opens an HLS presentation from the location of its playlist.
pub fn open(location: &str, options: &StreamingOptions) -> Result<SegmentedDemuxer> {
	let mut fetcher = Fetcher::new();
	let mut metadata = Metadata::default();
	let specs = match fetch_playlist(&mut fetcher, location)? {
		Playlist::Media(media) => vec![track(media, location.to_string(), options, Vec::new())?],
		Playlist::Multivariant(master) => {
			let candidates: Vec<(u64, Option<u32>)> =
				master.variants.iter().map(|v| (v.bandwidth, v.resolution.map(|r| r.1))).collect();
			let chosen =
				options.choose(&candidates).ok_or(Message::Container("hls: no variant to play"))?;
			let variant = &master.variants[chosen];
			metadata
				.raw
				.insert("variant_bitrate".into(), RawValue::String(variant.bandwidth.to_string()));
			metadata
				.raw
				.insert("variant_count".into(), RawValue::String(master.variants.len().to_string()));
			let media = fetch_media(&mut fetcher, &variant.uri)?;
			let tags = vec![("variant_bitrate".to_string(), variant.bandwidth.to_string())];
			let mut specs = vec![track(media, variant.uri.clone(), options, tags)?];

			// The audio rendition: the preferred language, else the default,
			// else the first of the group. One without a URI is muxed in.
			if let Some(group) = &variant.audio {
				let group: Vec<_> = master
					.renditions
					.iter()
					.filter(|r| r.kind == MediaType::Audio && &r.group == group)
					.collect();
				let audio = group
					.iter()
					.find(|r| options.language_matches(r.language.as_deref()))
					.or_else(|| group.iter().find(|r| r.default))
					.or_else(|| group.first());
				if let Some(r) = audio
					&& let Some(uri) = &r.uri
					&& uri != &variant.uri
				{
					let media = fetch_media(&mut fetcher, uri)?;
					let mut tags = vec![("title".to_string(), r.name.clone())];
					if let Some(lang) = &r.language {
						tags.push(("language".to_string(), lang.clone()));
					}
					specs.push(track(media, uri.clone(), options, tags)?);
				}
			}

			// Every subtitle rendition of the variant's group (WebVTT
			// segments), each as one stream. One that fails to load is left
			// out rather than failing the presentation.
			if let Some(group) = &variant.subtitles {
				for r in
					master.renditions.iter().filter(|r| r.kind == MediaType::Subtitles && &r.group == group)
				{
					let Some(uri) = &r.uri else { continue };
					let Ok(media) = fetch_media(&mut fetcher, uri) else { continue };
					let mut tags = vec![("title".to_string(), r.name.clone())];
					if let Some(lang) = &r.language {
						tags.push(("language".to_string(), lang.clone()));
					}
					if !r.default {
						tags.push(("default".to_string(), "0".to_string()));
					}
					if let Ok(spec) = track(media, uri.clone(), options, tags) {
						specs.push(spec);
					}
				}
			}
			specs
		}
	};
	SegmentedDemuxer::open(specs, metadata, fetcher)
}
