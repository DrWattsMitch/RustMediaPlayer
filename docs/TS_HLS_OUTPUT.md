# MPEG-TS and HLS output, Opus in TS, SAMPLE-AES, WebVTT, MP4 languages

This document covers the seventh batch of work on RMP, which closes the
gaps listed at the end of [MPEGTS_HLS_DASH.md](MPEGTS_HLS_DASH.md): an
MPEG transport stream muxer, HLS output on top of it, Opus in TS (both
directions), SAMPLE-AES (both directions), WebVTT subtitle renditions (both
directions), and stream languages in the MP4 muxer. It covers the design,
how each part maps to FFmpeg, how it was verified, and what is left. Read
it before changing `src/container/mpegts/muxer.rs`,
`src/container/hls/muxer.rs`, `src/container/hls/sample_aes.rs` or
`src/container/webvtt.rs`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **MPEG-TS muxer** | `.ts`, `.m2ts`, `.mts`. H.264, HEVC, MPEG-2, MPEG-4 part 2, AAC, MP1/2/3, AC-3, E-AC-3, Opus. FFmpeg's layout: PIDs 0x100+, PMT 0x1000, PAT/PMT every 100 ms and before video keyframes, SDT every 500 ms, PCR on the first video stream, 1.4 s start offset. Length-prefixed H.264/HEVC → Annex B + AUD + parameter sets; raw AAC → ADTS; ISO 639 descriptors; decode times derived for pts-only (Matroska) video. FFmpeg's `mpegts_*`, `pes_payload_size`, `pat_period`, `sdt_period`, `muxdelay` options. | `container::mpegts::TsMuxer` |
| **HLS muxer** | `.m3u8`: media playlist + segments, cut at keyframes every `hls_time`; MPEG-TS or fMP4 (`init.mp4` + `.m4s`) segments; AES-128 (`hls_enc`, `hls_key_info_file`, key rotation) or SAMPLE-AES; WebVTT renditions with `X-TIMESTAMP-MAP`; master playlist with `BANDWIDTH`, `AVERAGE-BANDWIDTH`, `RESOLUTION`, `CODECS`, `SUBTITLES`; sliding window with `delete_segments`; VOD/EVENT types. | `container::hls::muxer::HlsMuxer` |
| **Opus in TS** | Demux: registration `Opus` + extension descriptor 0x80 (channel configuration code), control headers stripped, start trim → pre-skip and `encoder_delay`, end trim → shorter last duration. Mux: the same descriptors, control headers with FFmpeg's size coding, pre-skip as start trim, end trim on the last packet. | `mpegts::es::{opus_control_header, opus_head}` |
| **SAMPLE-AES** | Apple's "MPEG-2 Stream Encryption Format for HTTP Live Streaming": H.264 (1:9 pattern after 32 clear bytes, emulation prevention re-applied), ADTS AAC and AC-3/E-AC-3 (whole blocks after a 16-byte leader). Demux: stream types 0xDB/0xCF/0xC1/0xC2, `EXT-X-KEY:METHOD=SAMPLE-AES`, per-packet decryption in the segment engine. Mux: H.264 and AAC encrypted, with `zavc`/`aacd` and `apad` descriptors. | `container::hls::sample_aes` |
| **WebVTT** | Demuxer (cue text per packet, ms time base), muxer, `X-TIMESTAMP-MAP`. HLS input: every rendition of the variant's `SUBTITLES` group becomes a stream. `.vtt` is an input and output format of its own. | `container::webvtt` |
| **MP4 language** | `mdhd` language from the stream's `language` tag (ISO 639-2, lower-cased; anything else is `und`, as FFmpeg). Re-encoded video keeps `language`, `default` and the title. | `mp4::muxer` (`pack_language`) |
| **AES** | Forward cipher and in-place CBC chains next to the existing decryption. | `hls::aes::{encrypt_cbc, Aes128}` |
| **CLI** | `.ts`, `.m3u8`, `.vtt` outputs (`-f ts`, `-f hls`, `-f vtt`), FFmpeg's `-hls_*` and `-mpegts_*` options, per-container defaults. | `transcode`, `io::Output` |

```bash
rmp -i movie.mkv out.ts
rmp -i movie.mkv -hls_time 4 -hls_playlist_type vod hls/out.m3u8
rmp -i movie.mp4 -hls_segment_type fmp4 -hls_flags independent_segments hls/out.m3u8
rmp -i movie.mp4 -hls_enc 1 -hls_enc_method sample-aes hls/out.m3u8
rmp -i live.m3u8 -c copy rec.mkv        # now with the WebVTT renditions
```

```rust
use rmp::container::hls::muxer::HlsMuxer;
use rmp::core::Muxer;
let mut hls = HlsMuxer::create("out/stream.m3u8")?;
hls.set_option("hls_time", "4")?;
hls.set_option("hls_segment_type", "fmp4")?;
// add streams, write packets, finish
```

---

## 2. How the project values were applied

**Safety.** No `unsafe`, no `unwrap`/`expect`/`panic!` outside tests. Every
index into packet data is checked against its length first (NAL scans stop
3 bytes before the end, ADTS/AC-3 frames are parsed before their bodies are
touched, the SAMPLE-AES pattern loop can't step past the NAL unit, WebVTT
files are capped at 64 MiB, the PMT at 1000 bytes, Opus control headers are
parsed byte by byte with `get`). Options are validated before they are
stored (a test caught `mpegts_start_pid 1` being kept after its error).
The fuzz test `sample_aes_webvtt_and_ts_muxer_survive_corruption` runs
damaged samples through decryption and encryption, damaged WebVTT through
its parser and damaged packets through the TS muxer; the TS demuxer fuzz
now includes the Opus fixture. 3,000 rounds with two seeds in a debug
build (overflow checks on) were clean.

**Efficiency.** The TS muxer copies each payload once into its PES and the
PES once into transport packets, which go to the output 64 KiB at a time.
Video is held only at the start (17 frames per video stream, to decide the
decode time shift and the origin), then passes straight through. HLS
segments are built in memory and written with one call each; playlists are
written through a temporary file and a rename. On a 93 MB, 2-minute 720p
H.264 + AAC MP4, `rmp -i big.mp4 big.ts` takes 0.17–0.19 s and `ffmpeg -i
big.mp4 -c copy big.ts` 0.27–0.37 s; to HLS with 4 s segments, RMP
0.24–0.38 s and FFmpeg 0.37–0.94 s (two-core machine, warm cache).

**Readability.** One file per job: `mpegts/muxer.rs` (tables, PES,
timing, payload conversion), `hls/muxer.rs` (segmentation, encryption,
playlists), `hls/sample_aes.rs` (the encryption scheme only),
`webvtt.rs` (the text format). The HLS muxer drives the TS or MP4 muxer
through the ordinary `Muxer` trait plus three hooks (`flush`,
`resend_headers`, `set_cipher` on TS; `cut_fragment` on MP4).

**Licensing.** Written from ISO/IEC 13818-1, ETSI EN 300 468, RFC 8216,
Apple's SAMPLE-AES specification, the "Opus in MPEG-TS" specification
(ETSI TS 102 366 style framing), W3C WebVTT and ISO/IEC 14496-12. FFmpeg's
output was the reference for layout and timing choices; no FFmpeg code was
ported.

---

## 3. Architecture map

```
src/container/
├── mpegts/
│   ├── muxer.rs      TsMuxer, TsOptions; PAT/PMT/SDT, PES, PCR, Annex B/ADTS/Opus payloads
│   ├── es.rs         + Opus control headers and channel codes, SAMPLE-AES stream types
│   └── demuxer.rs    + Opus PES framing, SAMPLE-AES flag
├── hls/
│   ├── muxer.rs      HlsMuxer, HlsOptions; segments, keys, playlists, WebVTT renditions
│   ├── sample_aes.rs encrypt_packet / decrypt_packet
│   ├── aes.rs        + encrypt_block, CBC in place, encrypt_cbc
│   ├── playlist.rs   + Key::SampleAes
│   └── mod.rs        + SUBTITLES renditions as tracks
├── webvtt.rs         WebVttDemuxer, WebVttMuxer, timestamp_map
├── segment.rs        + SegmentKey::SampleAes (per-packet decryption), WebVTT segments
├── bsf.rs            + annex_b_from_config (avcC/hvcC → Annex B)
└── mp4/muxer.rs      + mdhd language, cut_fragment
```

```
packets ─► HlsMuxer ─┬─ audio/video ─► TsMuxer ──► Sink ─► segment file (AES-128: whole file)
                     │                  │ SAMPLE-AES: per sample, IV per segment
                     │                  └ or Mp4Muxer (fragmented) ─► init.mp4 + .m4s
                     └─ subtitles ─► cues ─► .vtt per segment (X-TIMESTAMP-MAP)
                cut at reference keyframe ≥ n × hls_time ─► playlists rewritten
```

---

## 4. The MPEG-TS muxer

### 4.1 Layout

Everything follows `ffmpeg -f mpegts` defaults, so files look alike and
`ffprobe` shows the same programs, PIDs and timestamps:

| Item | Value |
| --- | --- |
| Program | service 1, transport stream 1, network 0xFF01, PMT PID 0x1000 |
| Streams | PIDs 0x100, 0x101, ... in `add` order |
| PAT + PMT | before the first packet, every 100 ms and before every video keyframe PES |
| SDT | every 500 ms; provider `RMP`, name `Service01`, or the `service_provider` / `service_name` tags (DVB UTF-8 when not ASCII) |
| Timestamps | the earliest decode time becomes 1.4 s (2 × `muxdelay`) |
| PCR | first video stream, else first audio stream; in the first TS packet of each of its PES; 0.7 s before the decode time |
| PES | one per video access unit (length 0); audio gathered up to 2930 bytes or 0.7 s (100 ms on an audio-only PCR stream, which needs a PCR that often) |
| Adaptation field | random access flag on keyframes, PCR, stuffing in the last packet of a PES |

| Codec | Stream type | PES id | Descriptors |
| --- | --- | --- | --- |
| H.264 | 0x1B (SAMPLE-AES 0xDB) | 0xE0 | (`zavc`) |
| HEVC | 0x24 | 0xE0 | |
| MPEG-2 / MPEG-4 video | 0x02 / 0x10 | 0xE0 | |
| AAC | 0x0F (SAMPLE-AES 0xCF) | 0xC0 | language; (`aacd`, `apad` with the AudioSpecificConfig) |
| MP1/2/3 | 0x03 (0x04 below 32 kHz) | 0xC0 | language |
| AC-3 / E-AC-3 | 0x81 / 0x87 | 0xBD | language |
| Opus | 0x06 | 0xBD | registration `Opus`, extension 0x7F/0x80 with the channel code, language |

### 4.2 Payloads

- **H.264 / HEVC from MP4 or Matroska** (an `avcC`/`hvcC` record): NAL
  lengths become start codes, the parameter sets from the record go right
  before the first IDR (HEVC: IRAP) slice of a keyframe that carries none,
  and an access unit delimiter is put first. Start codes are 4 bytes for
  the first unit and parameter sets and 3 for the others (H.264), 4 for
  all (HEVC), exactly as FFmpeg's `h264_mp4toannexb` / `hevc_mp4toannexb`
  write them.
- **Annex B input** (a TS source) is kept byte for byte; only a missing
  delimiter is added (with the parameter sets, on a keyframe without
  them), as FFmpeg's `mpegtsenc` does.
- **AAC**: raw frames get a 7-byte ADTS header from the
  AudioSpecificConfig (profile, rate index, channel configuration; HE-AAC
  signals its core). A config with a program config element is refused.
  ADTS input passes.
- **Opus**: each packet gets a control header: `0x7F 0xE0|flags`, the size
  as `size / 255` bytes of 255 and the remainder (FFmpeg's coding), the
  start trim (the pre-skip, spread over the first packets) and, on the last
  packet, the end trim (the `encoder_padding` tag or a last packet shorter
  than its samples). One packet is held back so the last one is known.

### 4.3 Timing

TS needs `dts ≤ pts` and increasing decode times. Two kinds of input
don't give that directly:

- **MP4 with B-frames**: RMP's MP4 demuxer applies the edit list to
  presentation times only, so the first B-frames have `dts > pts`. The
  muxer looks at the first 17 frames of each video stream, takes the
  largest `dts - pts` as a shift, and moves that stream's decode times back
  by it.
- **Matroska** stores presentation times only. Decode times are derived:
  the k-th is the (k − d)-th smallest presentation time, d being the
  deepest reordering seen in the first 17 frames (the first d are spaced
  back by one frame step). They are kept `d` steps higher internally (RMP
  times are unsigned) and the same shift takes them back off.

The origin (what becomes 1.4 s) is the earliest decode time after these
shifts, over all streams, so the output starts at 1.4 s like FFmpeg's.
Start-up packets are queued in arrival order until the shifts are known,
so interleaving is kept.

---

## 5. HLS output

**Segmenting** (FFmpeg's rule): the reference stream is the first video
stream, else the first stream. The n-th cut happens at the first keyframe
of the reference stream whose pts is at least n × `hls_time` after the
first one (`split_by_time` allows non-keyframes). A segment's duration is
the time between its cut and the next; the last one ends at the reference
stream's last pts + duration. On a 120 s file with `-hls_time 4`, the
playlist is identical to FFmpeg's.

Before each cut the TS muxer is flushed (pending audio PES written) and
told to resend its tables, so every segment starts with PAT/PMT and a
keyframe; fMP4 output cuts one fragment per segment
(`Mp4Muxer::cut_fragment`), and the bytes before the first `moof` are
written once as the initialization section.

**Files and names** (FFmpeg's): `out.m3u8`, `out%d.ts` (or `.m4s`,
`-hls_segment_filename` with `%d`/`%0Nd`), `init.mp4`, `out.key`,
`out_vtt.m3u8` with `out%d.vtt`. Several subtitle streams use
`out_vtt<i>.m3u8` / `out_vtt<i>_%d.vtt`. The master playlist is written
with `-master_pl_name`, and as `out_master.m3u8` whenever there are
subtitles (a subtitle rendition needs one).

**Playlist**: `#EXT-X-VERSION` 3 (5 with SAMPLE-AES, 7 with fMP4),
`TARGETDURATION` rounded up as FFmpeg's `get_int_from_double`, `EXTINF`
with six decimals, `EXT-X-MAP` before any key (the init section is never
encrypted), a new `EXT-X-KEY` when the key changes, `ENDLIST` unless
`omit_endlist`. `hls_list_size` keeps a sliding window (default 0 = all, a
deliberate difference from FFmpeg's 5: an output file meant for playback
needs every segment); `delete_segments` removes what falls out.

**Keys**:

| Option | Key | URI | IV |
| --- | --- | --- | --- |
| `-hls_enc 1` | `-hls_enc_key` or 16 random bytes (OS generator), written to `out.key` | `-hls_enc_key_url` + `out.key` | `-hls_enc_iv`, else the sequence number (not written) |
| `-hls_key_info_file f` | the file named on line 2, re-read before every segment (key rotation) | line 1 | line 3, else the sequence number |

`-hls_enc_method aes-128` (default) encrypts whole segments with PKCS#7
padding; `sample-aes` hands the cipher to the TS muxer, which encrypts
H.264 and AAC samples after converting them (AC-3, E-AC-3 and HEVC stay in
the clear, with their usual stream types, which the specification allows).
SAMPLE-AES with fMP4 segments would be CENC `cbcs` and is refused.

**Subtitles**: WebVTT or SubRip packets are collected per segment and
written as `.vtt` files whose header maps cue time 0 to the TS muxer's
start offset (`X-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000`; 0 for
fMP4), with cue times measured from the TS muxer's origin, so the cues
land on the video's clock in every player that honours the map.

---

## 6. Reading: SAMPLE-AES and WebVTT in the segment engine

- `EXT-X-KEY:METHOD=SAMPLE-AES` (key format `identity`) becomes
  `SegmentKey::SampleAes`. The segment is demuxed as it is; each packet of
  an H.264, AAC, AC-3 or E-AC-3 stream is decrypted with the segment's key
  and IV. TS and packed-audio segments are supported; fMP4 SAMPLE-AES
  (`cbcs`) is an error, and other key formats (FairPlay and the like)
  still are.
- The TS demuxer maps 0xDB/0xCF/0xC1/0xC2 to their codecs and tags such
  streams `encryption=SAMPLE-AES`, so a file opened without its playlist
  says why its samples don't decode.
- WebVTT segments are recognised by their `WEBVTT` header; the
  `X-TIMESTAMP-MAP` offset (`MPEGTS / 90000 − LOCAL`) is added to their
  cues like packed audio's ID3 timestamp. Every rendition of the variant's
  `SUBTITLES` group becomes one stream with its `language`, `NAME` (as the
  title) and `default=0` for non-default renditions; one whose playlist
  fails to load is left out.

---

## 7. Verification

References are FFmpeg 6.1, packets compared with `ffprobe -show_packets
-show_data_hash adler32` (stream, pts, dts, duration, size, flags, hash).

**TS muxer against `ffmpeg -c copy -f mpegts`:**

| Source | Result |
| --- | --- |
| TS: H.264 + 2 × AAC, HEVC + AC-3, MPEG-2 + MP2 | every packet identical (timestamps and payloads); HEVC + AC-3 and MPEG-2 + MP2 files are the same size |
| MP4: H.264 B-frames + AAC; H.264 High 4:4:4 | every payload identical; timestamps follow RMP's MP4 timeline (A/V start together, `dts ≤ pts`) |
| Matroska: H.264 + AAC; HEVC Main 10 | every payload identical; HEVC timestamps identical too |
| Opus (WebM stereo, Ogg 5.1) | same file sizes, every packet identical; decodes to the same samples (the bytes differ only in the SDT provider name) |
| 93 MB 720p MP4 | all 9,226 payloads identical; 0.17–0.19 s (FFmpeg 0.27–0.37 s) |

Every output decodes with FFmpeg without a warning, to the same frame
hashes as the source.

**TS demuxer, Opus**: an FFmpeg-written 5.1 Opus TS demuxes packet for
packet like `ffprobe`, except the last packet's duration, which RMP
shortens by the end trim (as for Ogg) where FFmpeg reports skip side data.

**HLS output** (read by FFmpeg and by RMP):

| Presentation | Result |
| --- | --- |
| TS segments | playlist identical to FFmpeg's `hls` muxer (120 s, `-hls_time 4`); decodes to the source's frames; RMP reads back every packet |
| fMP4 segments | FFmpeg decodes to the source's frames; RMP reads back every payload |
| AES-128 (`hls_enc`, key info file) | FFmpeg decodes identically; RMP reads back the clear packets |
| SAMPLE-AES | RMP reads back the clear packets exactly. FFmpeg decrypts every AAC frame and every H.264 frame except the last one of each segment: its TS demuxer releases a segment's last video PES only after reading into the next segment, so it uses the next segment's IV (with one fixed IV for all segments, only the stream's very last frame differs). This is FFmpeg's limitation; Apple's specification sets the IV per segment |
| WebVTT rendition | FFmpeg (`-strict experimental -allowed_extensions ALL`) reads the cues; it ignores `X-TIMESTAMP-MAP`, RMP applies it and puts the cues on the video clock within 1 ms of the source |

**MP4 language**: `ffprobe` shows the `language` tags written by the
muxer, progressive and fragmented.

**Tests**:

- `tests/container/mpegts/muxer_test.rs`: TS → TS packet for packet (five
  fixtures, languages, service name, 1.4 s start), MP4/Matroska → TS
  decodes to the same frames with RMP's decoders, Opus layouts and
  pre-skip, PIDs/continuity/PCR and the PID options, SAMPLE-AES round trip.
- `tests/container/streaming/hls_output_test.rs`: TS and fMP4 segments read
  back, AES-128 / SAMPLE-AES / key info file, WebVTT rendition with cue
  timing, sliding window with deletion, option parsing.
- `tests/container/mpegts/demuxer_test.rs`: `opus_matches_ffmpeg`.
- `tests/container/mp4/muxer_test.rs`: `stream_languages_are_written_to_mdhd`.
- Unit tests: AES encryption (FIPS-197, SP 800-38A), SAMPLE-AES patterns
  and escaping, Opus control headers and channel codes, WebVTT parsing and
  writing, ADTS headers, Annex B conversion, `annex_b_from_config`, PES
  timestamps, playlist SAMPLE-AES keys.
- `robustness_test.rs`: `sample_aes_webvtt_and_ts_muxer_survive_corruption`,
  Opus added to `mpegts_demuxer_survives_corruption`.
- `tests/cli.rs`: `transport_stream_and_hls_output`.
- `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D
  warnings` pass. (`hevc_decoder_test::damaged_hevc_packets_do_not_panic`
  fails, before and after this work: a panic at `codecs/hevc/slice.rs:289`,
  unrelated to these changes.)

```bash
cargo test --test container mpegts hls_output stream_languages
RMP_FUZZ_ROUNDS=3000 cargo test --test container -- sample_aes_webvtt mpegts_demuxer_survives
```

---

## 8. Known limitations

| Limitation | Where to start |
| --- | --- |
| No DASH *output* (the fMP4 segmenter is there; an MPD writer with `SegmentTemplate` + `SegmentTimeline` per representation is what's missing). | `hls/muxer.rs` segmentation, `dash/mpd.rs` for the model |
| SAMPLE-AES in fMP4 (CENC `cbcs`) can't be read or written: RMP's MP4 demuxer/muxer don't handle `tenc`/`senc`/`saiz`/`saio`. | `mp4/fragment.rs` |
| SAMPLE-AES output encrypts H.264 and AAC only; AC-3/E-AC-3 would need the `dac3`/`dec3` setup data in the `apad` descriptor before the first frame is seen. | `mpegts::muxer::stream_setup` |
| One variant per HLS output: no `var_stream_map`, no separate audio renditions or bitrate ladders. | `HlsMuxer` (one inner muxer per variant) |
| No `.m2ts` 192-byte output, no constant bit rate (`muxrate`), no DVB subtitles / teletext / timed ID3 in the TS muxer. | `mpegts/muxer.rs` |
| The key info file's key is re-read per segment, but `EXT-X-KEY` rotation can't be driven from the library except through that file. | `HlsMuxer::load_key` |
| Opus channel mappings beyond the codes 0–8 and 0x82–0x88 (extended descriptors) are refused. | `es::opus_channel_code` |
| HLS input: I-frame playlists, LL-HLS parts and `EXT-X-DEFINE` are still ignored. | `hls/playlist.rs` |

---

## 9. How to build on this

### Add a codec to the TS muxer

1. In `TsMuxer::stream_setup`, give it a stream type, PES id and
   descriptors, and a `Payload` if its packets need converting.
2. Add it to `muxer::supports` (the CLI's copy-or-encode choice).
3. Make sure the TS demuxer maps it back (`es::classify`) and add it to
   `ts_remux_comes_back_packet_for_packet`.

### Add a segment format to HLS

Wrap its muxer in `hls::muxer::Inner`, give it a way to end a segment
(like `cut_fragment`), and decide in `cut` what goes into the
initialization section.

### Rules for contributors touching these modules

- Keep payload bytes identical to FFmpeg's and re-run the §7 comparisons
  after any change to `video_annex_b`, `adts` or the Opus framing.
- Never sniff packets to decide a conversion: the stream's extradata says
  whether H.264/HEVC is length-prefixed (§4.2), as in `bsf.rs`.
- The SAMPLE-AES IV is per segment and restarts with every NAL unit and
  audio frame; change one and the other side breaks silently, so the round
  trip tests must stay.
- Validate options before storing them.

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `mpegts/muxer.rs` | `libavformat/mpegtsenc.c`, `libavcodec/bsf/h264_mp4toannexb.c`, `hevc_mp4toannexb.c`, `libavcodec/adts_header.c` |
| `hls/muxer.rs` | `libavformat/hlsenc.c`, `hlsplaylist.c` |
| `hls/sample_aes.rs` | `libavformat/hls_sample_encryption.c` |
| `mpegts/es.rs` (Opus) | `libavformat/mpegts.c` (`opus_*` tables), `mpegtsenc.c` (Opus control header) |
| `webvtt.rs` | `libavformat/webvttdec.c`, `webvttenc.c` |
| `mp4/muxer.rs` (`pack_language`) | `libavformat/isom.c` (`ff_mov_iso639_to_lang`) |
