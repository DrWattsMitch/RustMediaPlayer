//! Text subtitle decoding: SubRip, ASS and WebVTT files, the same cues in
//! Matroska (muxed by FFmpeg) and MP4 timed text, all decoded to timed
//! cues. Timings are FFmpeg's (`ffprobe -show_packets`).

use crate::common::{fixture, memory};
use rmp::codecs::subtitle::ass::AssScript;
use rmp::container::detect;
use rmp::core::{Frame, Resolver, StreamKind, SubtitleFormat, SubtitleFrame};

/// Every cue of each subtitle stream, by stream.
fn cues(name: &str) -> Vec<(String, Vec<SubtitleFrame>)> {
	let data = fixture(&format!("subtitles/{name}"));
	let mut demuxer = detect(&data, memory(data.clone())).unwrap();
	let streams = demuxer.streams().by_kind(StreamKind::Subtitle);
	let resolver = Resolver::new();
	let mut decoders: Vec<_> = streams
		.iter()
		.map(|s| {
			assert!(resolver.has_decoder(&s.parameters.codec.id), "{}", s.parameters.codec.id);
			let d = resolver.decoder_for(s).unwrap();
			let st = d.create_state(&s.parameters);
			(s.id, s.parameters.codec.id.clone(), d, st, Vec::new())
		})
		.collect();
	while let Some(packet) = demuxer.read().unwrap() {
		let Some((_, _, d, st, out)) = decoders.iter_mut().find(|e| e.0 == packet.stream_id) else {
			continue;
		};
		for frame in d.decode(st, packet).unwrap().value {
			let Frame::Subtitle(s) = frame else { panic!("subtitle expected") };
			out.push(s);
		}
	}
	decoders.into_iter().map(|(_, codec, _, _, out)| (codec, out)).collect()
}

fn timed(cues: &[SubtitleFrame]) -> Vec<(u64, u64, &str)> {
	let ms = |s: f64| (s * 1000.0).round() as u64;
	cues.iter().map(|c| (ms(c.start_seconds()), ms(c.end_seconds()), c.text.as_str())).collect()
}

const SRT_CUES: [(u64, u64, &str); 3] = [
	(500, 2000, "Hello, world!"),
	(2500, 4250, "Second line & bold\nthird line"),
	(5000, 6000, "Top"),
];

const ASS_CUES: [(u64, u64, &str); 3] = [
	(500, 2000, "Hello, world!"),
	(1000, 3000, "Sign text\nsecond row"),
	(4000, 5500, "Soft break\u{a0}here"),
];

#[test]
fn subrip_file() {
	let streams = cues("sample.srt");
	let [(codec, cues)] = streams.as_slice() else { panic!("one stream") };
	assert_eq!(codec, "subrip");
	assert_eq!(timed(cues), SRT_CUES);
	assert_eq!(cues[0].markup, "<i>Hello</i>, world!");
	assert_eq!(cues[0].format, SubtitleFormat::SubRip);
}

#[test]
fn webvtt_file() {
	let streams = cues("sample.vtt");
	let [(codec, cues)] = streams.as_slice() else { panic!("one stream") };
	assert_eq!(codec, "webvtt");
	assert_eq!(timed(cues), SRT_CUES);
	assert_eq!(cues[1].format, SubtitleFormat::WebVtt);
}

#[test]
fn ass_file() {
	let data = fixture("subtitles/sample.ass");
	let demuxer = detect(&data, memory(data.clone())).unwrap();
	let script =
		AssScript::parse(&String::from_utf8_lossy(&demuxer.streams().streams[0].parameters.extradata));
	assert_eq!((script.play_res_x, script.play_res_y), (Some(640), Some(360)));
	assert_eq!(script.style("Top").alignment, 8);

	let streams = cues("sample.ass");
	let [(codec, cues)] = streams.as_slice() else { panic!("one stream") };
	assert_eq!(codec, "ass");
	assert_eq!(timed(cues), ASS_CUES);
	let sign = cues[1].ass.as_ref().unwrap();
	assert_eq!(
		(sign.layer, sign.style.as_str(), sign.name.as_str(), sign.read_order),
		(1, "Top", "Sign", 1)
	);
	assert_eq!(cues[1].markup, "{\\pos(320,40)}Sign text\\Nsecond row");
}

#[test]
fn matroska_srt_and_ass() {
	let streams = cues("sample.mkv");
	let [(srt, srt_cues), (ass, ass_cues)] = streams.as_slice() else { panic!("two streams") };
	assert_eq!((srt.as_str(), ass.as_str()), ("subrip", "ass"));
	assert_eq!(timed(srt_cues), SRT_CUES);
	assert_eq!(timed(ass_cues), ASS_CUES);
	assert_eq!(ass_cues[1].ass.as_ref().unwrap().style, "Top");
}

#[test]
fn mp4_timed_text() {
	let streams = cues("sample_tx3g.mp4");
	let [(codec, cues)] = streams.as_slice() else { panic!("one stream") };
	assert_eq!(codec, "mov_text");
	// FFmpeg's mov_text encoder turns the SRT tags into style boxes and
	// keeps `&amp;` as it is (timed text has no character references);
	// the empty samples it writes between cues decode to nothing.
	let mut expected = SRT_CUES;
	expected[1].2 = "Second line &amp; bold\nthird line";
	assert_eq!(timed(cues), expected);
	assert_eq!(cues[0].format, SubtitleFormat::Plain);
}
