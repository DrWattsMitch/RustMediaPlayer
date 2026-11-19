# MPEG-TS, HLS and DASH

This document covers the sixth batch of work on RMP: an MPEG transport
stream demuxer, HTTP Live Streaming (HLS) and MPEG-DASH input, the segment
engine they share, and the bitstream conversions that make a TS stream copy
into MP4 or Matroska. It covers the design, how each part maps to FFmpeg,
how the work was verified, and how to extend it. Read it before changing
anything under `src/container/mpegts/`, `src/container/hls/`,
`src/container/dash/`, `src/container/segment.rs`, `src/container/bsf.rs`
or `src/io/fetch.rs`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **MPEG-TS demuxer** | `.ts`, `.m2ts`, `.mts`: 188/192/204-byte packets (detected), PAT/PMT/SDT with CRC checks, PES reassembly, 33-bit timestamp unwrapping. H.264, HEVC, MPEG-2 video, MPEG-4 part 2, AAC (ADTS), MP1/2/3, AC-3, E-AC-3, LATM, DVB subtitles/teletext, Blu-ray PGS, timed ID3. Unlabelled private streams are identified by content. One packet per audio frame and per video access unit, keyframes from the bitstream, durations, languages, service names. Duration from the end of the file; keyframe seeking by bisection. | `container::mpegts::TsDemuxer` |
| **HLS** | Multivariant and media playlists; variant choice by bandwidth/height/index; separate audio renditions by language/default; MPEG-TS, fragmented MP4 (`EXT-X-MAP`) and packed-audio segments (ID3 timestamps); `EXT-X-BYTERANGE`; AES-128 decryption; `EXT-X-GAP`; live playlists reloaded until `EXT-X-ENDLIST`. | `container::hls::open` |
| **DASH** | Static and dynamic MPDs; `SegmentTemplate` with `$Number$` or `SegmentTimeline`, `SegmentList` (with media ranges), `SegmentBase` with a `sidx` index; BaseURL inheritance; representation choice; multi-period presentations with continuous timestamps; fragmented MP4, WebM and TS segments; live edge from `availabilityStartTime`. | `container::dash::open` |
| **Segment engine** | Per-segment demuxing with init sections, decryption, interleaving of renditions by decode time, segment-level seeking, live refresh. | `container::segment::SegmentedDemuxer` |
| **Bitstream conversion** | Annex B → `avcC`/`hvcC` + length-prefixed NALs, ADTS → raw AAC + AudioSpecificConfig, applied automatically by the MP4 and Matroska muxers. | `container::bsf::Converter` |
| **Fetching** | Files and HTTP(S), byte ranges, one connection pool, RFC 3986 reference resolution. | `io::Fetcher`, `io::resolve` |
| **Small parsers** | M3U8 attribute lists; an XML reader (no DTDs, no external entities); `xs:duration` and `xs:dateTime`; `sidx`; an SPS summary that accepts every H.264 profile and an HEVC one. | `hls::playlist`, `dash::xml`, `dash::mpd`, `codecs::h264::nal::sps_summary` |
| **Wiring** | `Input::open` recognises playlists and MPDs by content; `Input::open_with(location, &StreamingOptions)`; TS detection in `container::detect`. | `io::Input` |

```bash
rmp -i broadcast.ts                                 # streams, languages, service name
rmp -i broadcast.ts -c copy out.mp4                 # H.264 + AAC: Annex B → avcC, ADTS → raw
rmp -i https://example.com/live/master.m3u8 -c copy rec.mkv
rmp -i manifest.mpd -c copy out.mp4
```

```rust
use rmp::container::segment::StreamingOptions;
let options = StreamingOptions { max_height: Some(720), audio_language: Some("fr".into()), ..Default::default() };
let mut input = rmp::io::Input::open_with("https://example.com/master.m3u8", &options)?;
```

---

## 2. How the project values were applied

**Safety.** No `unsafe`, and no `unwrap`/`expect`/`panic!` in non-test
code. Every length read from a file is checked before it is used: PSI
`section_length` (capped at 4 KiB), the adaptation field length, PES header
lengths, ADTS/MPEG audio/AC-3 frame lengths, ID3 sizes, M3U8 byte ranges,
`sidx` reference counts, NAL list lengths in `avcC`/`hvcC` (which must fit
`u16`). A PES is capped at 16 MiB and a fetched resource at 1 GiB, so a
broken stream can't grow memory without bound; a DASH representation is
capped at a million segments. The XML reader limits nesting to 64 levels,
never expands DTDs or external entities, and accepts only the five
predefined entities and numeric ones. Integer arithmetic on file values is
checked or saturating; one bug of exactly this kind was caught by the tests
(`then_some` evaluates eagerly, so `parse_range("5-4")` underflowed before
its guard) and fixed with `then`. The robustness suite fuzzes the TS
demuxer, both manifest parsers and whole HLS/DASH presentations with one
damaged file each. A debug-build run of 1,500 rounds each was clean.

**Efficiency.** The TS demuxer reads 64 KiB at a time, copies each
transport packet once (into the PES buffer), and parses video access units
only as far as their NAL headers (and the SPS when present). It demuxes a
117 MB, 2-minute 720p H.264 + AAC file in 0.16 s, printing every packet;
`ffmpeg -c copy -f null` takes 0.36 s on the same file. Streaming keeps
one segment per rendition in memory; init sections and keys are fetched
once; HTTP connections are reused. Segment TS demuxers skip the end-of-file
duration scan, which only whole files need.

**Readability.** The TS demuxer is split by layer: `packet.rs` (transport
packets, PES headers, unwrapping), `psi.rs` (tables), `es.rs` (what a stream
is and how its payload becomes packets), `demuxer.rs` (the state machine).
HLS and DASH only translate manifests into `TrackSpec`s; everything about
fetching, decrypting, demuxing and interleaving is in `segment.rs`, once.

**Licensing.** Written from ISO/IEC 13818-1, ETSI EN 300 468, RFC 8216,
ISO/IEC 23009-1, ISO/IEC 14496-12/-15, FIPS-197 and RFC 3986. FFmpeg's
output was the reference for packetisation and timestamp choices; no FFmpeg
code was ported.

---

## 3. Architecture map

```
src/
├── io/
│   ├── fetch.rs               Fetcher (file / HTTP, ranges, pool), resolve (RFC 3986)
│   └── input.rs               + manifest detection, Input::open_with
├── codecs/h264/nal.rs         + sps_summary (any profile)
└── container/
    ├── bsf.rs                 Converter, avcC/hvcC builders, hevc_sps_summary
    ├── mpegts/
    │   ├── packet.rs          layout detection, TS header, PES header, Unwrap
    │   ├── psi.rs             CRC-32/MPEG-2, SectionBuffer, PAT, PMT, SDT
    │   ├── es.rs              stream types → codec + Framing, audio frames, video info
    │   ├── demuxer.rs         TsDemuxer
    │   └── probe.rs           TsProbe
    ├── segment.rs             SegmentRef, TrackSpec, StreamingOptions, SegmentedDemuxer
    ├── hls/
    │   ├── playlist.rs        M3U8 model and parser
    │   ├── aes.rs             AES-128 CBC decryption
    │   └── mod.rs             open: variant + rendition choice → TrackSpecs
    └── dash/
        ├── xml.rs             XML reader
        ├── mpd.rs             durations, dates, templates, segment expansion, sidx
        └── mod.rs             open: representation choice, periods, live refresh
```

Data flow:

```
            ┌ .ts ─────────────────────────────────────────────► TsDemuxer ─┐
Input::open ┤ .m3u8 ─► hls::open ─┐                                        ├─► packets
            └ .mpd ──► dash::open ┴► TrackSpec per rendition ─► SegmentedDemuxer
                                       │  fetch ► decrypt ► init + segment ► detect ► demux
                                       └─ interleave by dts, refresh live manifests

MP4 / MKV muxer ─► bsf::Converter: Annex B → avcC + lengths, ADTS → ASC + raw
```

---

## 4. The MPEG-TS demuxer

### 4.1 Packets and tables (`packet.rs`, `psi.rs`)

**Layout detection** looks for the stride (188, 192 with a 4-byte M2TS
prefix, or 204 with Reed-Solomon bytes) and the offset whose grid holds the
most sync bytes, as FFmpeg's `get_packet_size` does. At least four packets
must be seen and 80% of the grid must sync: chance doesn't do that, and a
few damaged packets don't prevent it. Reading resynchronises on lost sync
by finding the next byte where two consecutive packets line up.

**Sections** are reassembled per PID with the pointer field, including
several sections per packet and stuffing. Long-form sections must pass the
CRC-32/MPEG-2 check. The demuxer reads:

- the **PAT** (program → PMT PID);
- every **PMT** (PCR PID, program and stream descriptors);
- the **SDT** (actual transport stream only), whose first service gives the
  `service_name` and `service_provider` tags, as FFmpeg reports them.
  DVB strings are decoded as UTF-8 when they say so, otherwise as Latin-1.

### 4.2 Stream types (`es.rs`)

| PMT says | Codec | Framing |
| --- | --- | --- |
| 0x01, 0x02 | `mpeg2video` | access units, I-picture keyframes, size from the sequence header |
| 0x03, 0x04 | `mp1`/`mp2`/`mp3` (from the header) | one packet per frame |
| 0x0F | `aac` (ADTS) | one packet per frame |
| 0x10 | `mpeg4` | one packet per PES |
| 0x11 | `aac_latm` | one packet per PES |
| 0x15, or registration `ID3 ` | `timed_id3` (data stream) | one packet per PES |
| 0x1B | `h264` | access units |
| 0x24, or registration `HEVC` | `hevc` | access units |
| 0x81, registration `AC-3`, DVB descriptor 0x6A | `ac3` | one packet per frame |
| 0x87, registration `EAC3`, descriptor 0x7A, HDMV 0x84/0xA1 | `eac3` | one packet per frame |
| 0x06 + descriptor 0x59 / 0x56 | `dvb_subtitle` / `dvb_teletext` | one packet per PES |
| HDMV 0x90 | `hdmv_pgs_subtitle` | one packet per PES |
| 0x06 with nothing else | probed: two consecutive ADTS, AC-3 or MPEG audio frames | as found |
| registration `Opus` (0x06) | `opus` | one packet per access unit, control headers stripped (see [TS_HLS_OUTPUT.md](TS_HLS_OUTPUT.md)) |
| SAMPLE-AES 0xDB / 0xCF / 0xC1 / 0xC2 | `h264` / `aac` / `ac3` / `eac3`, tagged `encryption=SAMPLE-AES` | as the clear types |
| SCTE-35 (0x86), anything else | skipped | — |

The ISO 639 descriptor becomes the stream's `language` (unless `und`).

### 4.3 From PES to packets (`demuxer.rs`)

A PES is complete at the next payload-unit start on its PID, or as soon as
its `PES_packet_length` is reached (audio), or at the end of the input.
Its PTS and DTS are unwrapped (`Unwrap` picks the value congruent modulo
2^33 that is closest to the previous one).

FFmpeg demuxes TS with its parsers on (`AVSTREAM_PARSE_FULL`), and RMP
matches what that produces:

- **Audio** payloads are split into frames. A frame carries a timestamp
  when it is the first frame starting in a PES that had one (H.222.0
  §2.4.3.7); frames may span PES boundaries. The frames after it are
  placed as FFmpeg places them, which differs by codec:
  - **AAC**: FFmpeg's AAC parser leaves the duration unset, so the next
    timestamp comes from `av_add_stable`: the time is snapped to the grid
    of whole frames (in 1024/rate units), moved one frame, and its offset
    from the grid kept. At 44.1 kHz this gives steps of 2090, 2090, 2090,
    2089, ... ticks. `add_stable` reproduces it.
  - **MPEG audio and AC-3**: those parsers set the duration, so the
    whole-tick duration (floor of samples × 90000 / rate) is added.
  - Packet durations are that floor in both cases, as in FFmpeg.
- **Video**: one packet per PES (one access unit, as every muxer writes
  them). Keyframes: H.264 IDR or a recovery-point SEI, HEVC IRAP
  (types 16–23), MPEG-2 I pictures; other video uses the adaptation
  field's random access indicator. Each packet is held until the next
  one, whose decode time gives its duration (the last packet repeats the
  previous duration): FFmpeg's values come from the frame rate, which is
  the same for constant-rate streams.
- **HEVC start codes**: FFmpeg's HEVC parser splits access units at the
  3-byte start code, so when a PES starts with `00 00 00 01`, the zero byte
  ends the *previous* access unit. RMP does the same so packets match byte
  for byte; both are valid Annex B.
- **Parameters**: H.264/HEVC streams become ready when an access unit
  carries the SPS/PPS (and VPS); the Annex B parameter sets are the
  extradata, as in FFmpeg. Sizes and pixel formats come from
  `sps_summary` / `hevc_sps_summary`, which read only header fields, so a
  10-bit or 4:2:2 stream is identified even though RMP's H.264 decoder is
  8-bit only. Audio streams become ready at their first frame.

### 4.4 Probing, duration, seeking

Opening reads until every PMT has been seen and every stream is ready, or
5 MB (FFmpeg's `probesize`). Packets read meanwhile are queued, so nothing
is lost. Streams that never showed parameters are dropped, as FFmpeg drops
what `avformat_find_stream_info` can't identify. Stream ids follow PMT
order.

The **duration** comes from demuxing the last 2 MiB with a copy of the
stream table (`Core::fork`): each stream's end is its last packet's
timestamp plus duration. That includes the last audio frames, which
FFmpeg's estimate (from PES timestamps only) leaves out, so RMP's durations
can be a frame or two longer than ffprobe's.

**Seeking** (`time` is on the packet clock, like every RMP demuxer):

1. Bisect the file in 256 KiB windows on the first PES timestamp of the
   reference stream (the first video stream, else the first stream).
2. Walk back from there, window by window, to the last keyframe PES (random
   access indicator, or a keyframe in its first packet) at or before the
   target.
3. Clear all PES state and resume there. PES packets that started before
   the new position are skipped because their start isn't seen.

---

## 5. Streaming (`segment.rs`)

### 5.1 Why per segment

FFmpeg's `hls` demuxer feeds every playlist's segments into a nested
demuxer through a custom I/O context. RMP instead opens a fresh demuxer for
each segment, over its bytes in memory (`init section + segment`):

- RMP's MP4 demuxer indexes every `moof` when it opens. Over a
  concatenated stream that would download the whole presentation first,
  and never finish on a live stream.
- Memory stays at one segment per rendition.
- Seeking is just choosing a segment; discontinuities and codec changes
  start a clean demuxer.

`open_segment_demuxer` routes TS segments to `TsDemuxer::new_segment`
(no duration scan) and everything else through `container::detect`, so
fragmented MP4, WebM and packed ADTS/AC-3/MP3 segments work without
special cases.

### 5.2 Tracks and streams

A `TrackSpec` is one rendition: its `SegmentRef`s (URI, byte range, init
section, duration, key, sequence number, time offset), where to start, and
an optional `Refresh` closure for live manifests. The first segment that
opens defines the track's streams. Later segments are mapped onto them by
kind and order ("the second audio stream"), so a segment missing a stream
doesn't shift the others.

`read` keeps one packet peeked per track and returns the one with the
smallest decode time. Every stream keeps its own time base.

Timestamps are the media's own: TS segments keep their 90 kHz PTS, fMP4
segments their `tfdt`-based times (through RMP's MP4 demuxer, which keeps
them non-negative), packed audio gets the ID3
`com.apple.streaming.transportStreamTimestamp` added. DASH periods after
the first are moved by `time_offset` so time continues across them.

### 5.3 Seeking and live

`seek(t)` places every track at the segment containing `t`, from the
manifest durations anchored at the first packet seen. Segments begin with
a keyframe (both specifications require it for switching), so playback can
start there; packets before `t` within the segment are returned, as with
the TS demuxer's keyframe seek.

A live track that runs out of segments waits for its poll interval (HLS:
half the target duration; DASH: `minimumUpdatePeriod`), calls its
`Refresh`, and appends the segments whose sequence number is new. After 30
polls without one, or when the manifest says the presentation ended, the
track ends. A segment that can't be fetched or opened is retried once and
then skipped, as players and FFmpeg do; the first one failing is an error.

---

## 6. HLS (`hls/`)

**Playlists** follow RFC 8216: `EXT-X-STREAM-INF` variants (bandwidth,
average bandwidth, codecs, resolution, frame rate, audio/video/subtitle
groups), `EXT-X-MEDIA` renditions, `EXTINF`, `EXT-X-MEDIA-SEQUENCE`,
`EXT-X-BYTERANGE` (an offset-less range continues the previous range of
the same URI), `EXT-X-MAP`, `EXT-X-KEY`, `EXT-X-DISCONTINUITY`,
`EXT-X-GAP`, `EXT-X-ENDLIST` and `EXT-X-PLAYLIST-TYPE:VOD`. Quoted
attribute values may contain commas. URIs are resolved against the
playlist's own location.

**Choice.** FFmpeg opens every variant and exposes all of their streams.
RMP plays one variant, as a player does: `StreamingOptions::variant`, else
the highest bandwidth within `max_bandwidth` / `max_height`, else the
lowest. Its audio group gives the audio rendition: the
`audio_language` match, else `DEFAULT=YES`, else the first. A rendition
without a URI is muxed into the variant. Streams get `variant_bitrate`
(FFmpeg's tag), and the audio rendition's `language` and `NAME` (as
`title`).

**AES-128** segments are decrypted with the key from `EXT-X-KEY` (fetched
once) and its IV, or the media sequence number when there is none, and
PKCS#7 padding is checked: a wrong key is an error, not garbage packets.
`aes.rs` is a plain byte-oriented FIPS-197 inverse cipher; a few-megabyte
segment decrypts in milliseconds, so a T-table version wasn't worth its
size. SAMPLE-AES (TS and packed audio) is decrypted per sample and WebVTT
subtitle renditions are read since the seventh batch
([TS_HLS_OUTPUT.md](TS_HLS_OUTPUT.md)); DRM key formats are refused with
an error.

**Live**: starts `live_start` (default 3, FFmpeg's `live_start_index -3`)
segments before the end of the playlist.

---

## 7. DASH (`dash/`)

**XML** (`xml.rs`) is a small reader: elements, attributes (namespace
prefixes dropped), text, comments, CDATA, processing instructions, the
predefined and numeric entities. DOCTYPEs are skipped unread.

**Segments** (`mpd.rs`). Segment information is merged from Period,
AdaptationSet and Representation (lower levels override attributes).
BaseURLs are resolved level by level from the MPD's location.

| Addressing | Segments |
| --- | --- |
| `SegmentTemplate` + `SegmentTimeline` | each `S` (`t`, `d`, `r`; `r=-1` repeats to the next `S`, the period end, or the live edge), `$Number$` from `startNumber`, `$Time$` from `t` |
| `SegmentTemplate` + `@duration` | static: ⌈period duration / duration⌉; dynamic: those complete since `availabilityStartTime` + period start, within `timeShiftBufferDepth` |
| `SegmentList` | `SegmentURL@media` / `@mediaRange`, with an optional timeline |
| `SegmentBase` + `indexRange` | the `sidx` (fetched by range) gives each subsegment's byte range and duration; the init section is `Initialization@range`, or the bytes before the index |
| `BaseURL` alone | one segment: the whole file |

Template identifiers `$RepresentationID$`, `$Number$`, `$Bandwidth$`,
`$Time$` (with `%0Nd` widths) and `$$` are expanded. An `Initialization`
with a range and no URL is a range of the media file (single-file
representations). Nominal durations are trimmed to the period's length.

**Choice**: video from the first non-trick-mode video set by
`StreamingOptions` (bandwidth, height, index); audio from the set matching
`audio_language`, else the one with `Role value="main"`, else the first,
at its highest bandwidth. A set or representation with `ContentProtection`
is refused (CENC media would only demux to encrypted samples).

**Periods**: static presentations play every period in order; their
segments join one track per kind, and each period's
`start - presentationTimeOffset` is aligned to the first period's, so
time keeps increasing. Dynamic presentations play the last period, starting
`live_start` segments before the live edge, and re-read the MPD every
`minimumUpdatePeriod`. Timeline segments are identified by their start
time, so the window can slide without renumbering.

---

## 8. Stream copy into MP4 and Matroska (`bsf.rs`)

TS (and raw `.h264`, `.aac`) carry H.264/HEVC as Annex B with in-band
parameter sets, and AAC as ADTS. MP4 and Matroska need length-prefixed NAL
units with an `avcC`/`hvcC` record, and raw AAC with an
AudioSpecificConfig. FFmpeg's `mov` and `matroska` muxers convert
automatically; `Converter` does it for RMP's:

- **`add`**: a stream whose extradata is Annex B gets an `avcC` (FFmpeg's
  `ff_isom_write_avcc` layout, with the chroma/bit-depth extension for the
  high profiles) or an `hvcC` (profile/tier/level from the SPS,
  `lengthSizeMinusOne` 3, VPS/SPS/PPS/SEI arrays). Only such streams have
  their packets converted: an `avcC` stream's packet can legitimately begin
  `00 00 01 xx` (a NAL length of 256–511), so packets are never sniffed.
- **`write`**: Annex B packets become 4-byte-length-prefixed; an AAC packet
  that is exactly one ADTS frame loses its header, and the first one fills
  in a missing AudioSpecificConfig before the header is written (MP4 writes
  `moov` at the end or at the first fragment; Matroska writes its header
  with the first cluster). ADTS with several raw data blocks or an in-band
  channel configuration is refused, as in FFmpeg's `aac_adtstoasc`.

---

## 9. Verification

All references are FFmpeg 6.1. Packets were compared with `ffprobe
-show_packets -show_data_hash adler32` on stream, pts, dts, duration, size,
flags and payload hash, stream by stream (the interleaving of streams may
differ).

**MPEG-TS**, packet for packet:

| Source | Result |
| --- | --- |
| H.264 + AAC 48 kHz | identical |
| H.264 + AAC 48 kHz + AAC 44.1 kHz (two languages) | identical (`av_add_stable` interpolation) |
| HEVC + AC-3 | identical (start-code split) |
| H.264 + E-AC-3 | identical |
| MPEG-2 video + MP2 44.1 kHz | identical |
| M2TS (192-byte), H.264 + unlabelled private-stream MP3 | identical (content probe) |
| Timestamps crossing the 33-bit wrap | identical + 2^33 (RMP keeps increasing where FFmpeg goes negative before the wrap) |
| 117 MB 720p H.264 + AAC | seeks to 1.4, 10, 31.7, 60.05 and 121 s land on the last keyframe before each; 0.16 s to demux (FFmpeg `-c copy -f null`: 0.36 s) |

**Remux**: TS → MP4 and TS → MKV stream copies (H.264 + AAC, HEVC + AC-3,
H.264 + E-AC-3, two audio tracks) decode to identical video frame MD5s and
audio samples as the TS (AC-3 and E-AC-3 into MP4 are re-encoded to AAC,
as the MP4 muxer has always done, since it doesn't carry them).

**HLS** (FFmpeg's `hls` muxer):

| Presentation | Result |
| --- | --- |
| TS segments | identical to ffprobe of the playlist |
| fMP4 segments (`EXT-X-MAP`) | identical, except the last AAC packet's duration, which RMP's MP4 demuxer trims to the edit list; equal to RMP's MP4 demuxer on the joined file |
| Single file with `EXT-X-BYTERANGE` | equal to the separate-file segments |
| AES-128 | identical (ffprobe with `-allowed_extensions ALL`) |
| Multivariant + separate audio rendition | highest variant chosen, audio from its rendition playlist, interleaved by time; `max_bandwidth` / `variant` choose the other |
| Live, over HTTP (`-hls_list_size 4 -hls_flags delete_segments`, served while being written) | starts 3 segments from the edge, reloads, ends at `EXT-X-ENDLIST`, in real time |

**DASH** (FFmpeg's `dash` muxer, and a hand-written `SegmentBase` MPD):

| Presentation | Result |
| --- | --- |
| `SegmentTemplate` + `SegmentTimeline` | each representation equals RMP's MP4 demuxer on its joined segments; payloads, flags and order equal ffprobe's, timestamps one constant apart (FFmpeg's go negative). FFmpeg 6.1's `dash` demuxer drops the last timeline segment; RMP reads it. |
| `SegmentTemplate` + `$Number$` | the same packets |
| `SegmentList` with separate files / with `mediaRange`s of one file | the same packets as the template |
| `SegmentBase` + global `sidx` | payloads equal ffprobe of the file (ffprobe hangs on FFmpeg's own single-file MPD) |
| WebM (VP9 + Opus) | payloads, flags and order equal ffprobe's |

**Tests**:

- `tests/container/mpegts/demuxer_test.rs`: the four FFmpeg comparisons,
  stream parameters and tags, the 33-bit wrap, keyframe seeking, detection
  and a file cut mid-packet, and TS → MP4/MKV remux (avcC, ASC, payloads).
- `tests/container/streaming/streaming_test.rs`: HLS TS, fMP4, byte
  ranges, AES-128, variant and rendition choice, seeking; DASH template,
  list, base; `Input::open` detection; a local HTTP server for byte-range
  HLS and a live playlist that grows while it is read; broken manifests.
- Unit tests in each module: CRC, sections, PMT, DVB strings, layouts, PES
  timestamps, unwrapping, stream types, MPEG-2 pictures, AES (FIPS-197
  C.1, SP 800-38A with padding), playlists, RFC 3986 resolution, XML,
  durations and dates, templates, `sidx`, live template windows, DRM
  refusal, `avcC` round trip, packet conversion, ID3 timestamps, stream
  mapping.
- `robustness_test.rs`: `mpegts_demuxer_survives_corruption`,
  `playlist_and_mpd_parsers_survive_corruption`,
  `hls_and_dash_survive_corrupted_segments`.
- `tests/cli.rs`: `transport_streams_and_manifests_as_input`.
- `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D
  warnings` pass.

```bash
cargo test --test container mpegts streaming
RMP_FUZZ_ROUNDS=1500 cargo test --test container -- mpegts_demuxer_survives playlist_and_mpd hls_and_dash
```

To repeat the FFmpeg comparison for any file or manifest:

```bash
p() { ffprobe -v error -show_packets -show_data_hash adler32 -of csv=p=0 \
	-show_entries packet=stream_index,pts,dts,duration,size,flags,data_hash "$1"; }
```

and compare with the same fields printed from RMP's packets, stream by
stream.

---

## 10. Known limitations (with pointers)

| Limitation | Where to start |
| --- | --- |
| ~~No MPEG-TS muxer~~, ~~SAMPLE-AES~~, ~~Opus in TS~~, ~~HLS WebVTT renditions~~: done in the seventh batch, see [TS_HLS_OUTPUT.md](TS_HLS_OUTPUT.md). No DASH *output* yet. | `hls/muxer.rs`, `dash/mpd.rs` |
| CENC (DASH `ContentProtection`, fMP4 SAMPLE-AES `cbcs`) and DRM aren't supported; they are refused. | `dash::choose`, `mp4/fragment.rs` |
| DTS / TrueHD in TS aren't mapped. | `es::classify` |
| HLS: I-frame playlists and LL-HLS parts (`EXT-X-PART`) are ignored; `EXT-X-DEFINE` variables aren't substituted. | `hls/playlist.rs`, `hls::open` |
| DASH: hierarchical `sidx`, `xlink` periods, `SegmentTemplate@index`, text adaptation sets, and live multi-period transitions aren't handled. | `mpd::parse_sidx`, `dash::build_tracks`. |
| Seeking in HLS/DASH lands on a segment start; packets before the target in that segment are returned. | `SegmentedDemuxer::seek`; dropping them needs per-stream keyframe tracking. |
| `HttpIo` (plain HTTP input) caches from byte 0, so the TS duration scan of a large remote `.ts` downloads it. Streaming manifests don't use it. | `io/http.rs`: range requests on seek. |
| The CLI has no options for variant choice; the library has `StreamingOptions`. | `transcode::Options`, then `Input::open_with`. |
| Encrypted keys are fetched without authentication headers or cookies. | `io::Fetcher`. |

---

## 11. How to build on this

### Add a TS stream type

1. Map it in `es::classify` (stream type, registration descriptor or DVB
   descriptor) to a codec name, kind and `Framing`.
2. For a framed audio codec, add its header to `es::audio_frame` and its
   sync byte to the resync in `Core::audio_pes`; for video, its keyframe
   and parameter rules to `es::inspect_video`.
3. Add a fixture to `tools/make_fixtures.sh` with its `ts_packets`
   reference and an `assert_matches_ffmpeg` test.

### Add a manifest feature

Manifests only produce `TrackSpec`s. A new addressing mode is a new branch
in `mpd::segments` (or a tag in `hls/playlist.rs`) that yields
`SegmentRef`s; nothing in `segment.rs` should need to change. Anything that
changes how segments are fetched or demuxed belongs in `segment.rs`, so
both formats get it.

### Rules for contributors touching these modules

- Never trust a length from the stream: check it against what's buffered,
  and cap anything that grows (PES, resources, segment counts).
- Use `bool::then` (lazy), not `then_some`, when the value does arithmetic
  on unchecked input.
- Keep packetisation and timestamps identical to FFmpeg's, and re-run the
  comparisons in §9 after any change to `demuxer.rs` or `es.rs`.
- Don't sniff packet contents to decide a conversion; decide from the
  stream's extradata (§8).

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `container/mpegts/` | `libavformat/mpegts.c` |
| `mpegts/es.rs` (framing) | `libavcodec/aac_parser.c`, `aac_ac3_parser.c`, `mpegaudio_parser.c`, `h264_parser.c`, `hevc_parser.c`, `mpegvideo_parser.c` |
| `mpegts/demuxer.rs` (`add_stable`) | `libavutil/mathematics.c` (`av_add_stable`), `libavformat/demux.c` (`compute_pkt_fields`) |
| `container/hls/` | `libavformat/hls.c` |
| `hls/aes.rs` | `libavutil/aes.c` |
| `container/dash/` | `libavformat/dashdec.c` |
| `container/bsf.rs` | `libavformat/avc.c` (`ff_isom_write_avcc`), `hevc.c` (`ff_isom_write_hvcc`), `libavcodec/bsf/aac_adtstoasc.c` |
| `io/fetch.rs` | `libavformat/http.c`, `libavformat/url.c` (`ff_make_absolute_url`) |
