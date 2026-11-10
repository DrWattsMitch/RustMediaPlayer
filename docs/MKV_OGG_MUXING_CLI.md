# Matroska/WebM and Ogg muxing, and the `rmp` CLI

This document covers the fifth batch of work on RMP: a Matroska / WebM
muxer, an Ogg muxer (Opus, FLAC, Vorbis), a Vorbis packet-duration parser,
and the `rmp` command-line tool, which is built on the transcode pipeline
that used to live in `examples/transcode.rs`. It covers the design, how each
part maps to FFmpeg, how the work was verified, and how to extend it. Read
it before changing anything under `src/container/mkv/muxer.rs`,
`src/container/ogg/muxer.rs`, `src/codecs/vorbis.rs`, `src/transcode.rs` or
`src/main.rs`.

It builds on [MKV_FLAC_OPUS_PROCESSING.md](MKV_FLAC_OPUS_PROCESSING.md),
which describes the demuxers these muxers mirror.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Matroska / WebM muxer** | `.mkv`, `.mka`, `.mks`, `.webm`. Every codec the demuxer maps (H.264, HEVC, AV1, VP8/9, MPEG-4, raw video, AAC, MP3, Opus, Vorbis, FLAC, ALAC, AC-3, PCM, SubRip, ASS, WebVTT, ...). SeekHead, Info with exact duration, Tracks, Tags (file and per track), Attachments (cover art, fonts), clusters with exact sizes, Cues. Opus `CodecDelay`/`SeekPreRoll`, `DiscardPadding` end trimming, `CodecDelay` for AAC/MP3 priming. FLAC STREAMINFO back-patched. WebM codec restrictions. | `container::mkv::muxer::MkvMuxer` |
| **Ogg muxer** | `.ogg`, `.oga`, `.opus`: Opus, FLAC and Vorbis mappings. BOS/header/audio page order, exact granule positions, FFmpeg's page layout, EOS, Vorbis comments with cover art, FLAC header page back-patched, Opus end trimming. `.opus` accepts only Opus. | `container::ogg::muxer::OggMuxer` |
| **Vorbis parser** | Packet durations from the mode table at the end of the setup header, without decoding. | `codecs::vorbis::VorbisParser` |
| **FLAC stats** | Shared STREAMINFO tracker (block/frame bounds, sample count, MD5 by decoding) used by the FLAC, Matroska and Ogg muxers. | `container::flac::stats::FlacStats` |
| **Matroska demuxer** | `V_UNCOMPRESSED` ↔ `rawvideo` with `ColourSpace` FourCC; `DiscardPadding` → shortened last Opus packet. | `container::mkv::{tracks, demuxer}` |
| **Transcode library** | The decode → convert → encode → mux pipeline as a library: `Options` (FFmpeg-style parsing), per-container defaults, subtitle copy, `-metadata`, `-f`. | `transcode::{Options, transcode}` |
| **`rmp` CLI** | `src/main.rs`: inspect (`rmp -i in`, `--show`), convert (`rmp -i in [opts] out`), batch (`rmp -i "dir/*.wav" -f flac out/`, parallel), overwrite protection, cleanup of failed outputs. | `rmp` binary |
| **Wiring** | Muxer selection by extension or MIME for `mkv`/`mka`/`mks`/`webm`/`ogg`/`oga`/`ogx`/`opus`; `Output::create_as` (FFmpeg's `-f`). | `container::select_muxer`, `io::Output` |

```bash
rmp -i input.mp4 output.mkv                       # H.264 + AAC stream copy
rmp -i clip.webm clip.opus                        # WebM Opus → Ogg Opus, sample exact
rmp -i song.mp3 -ar 48000 -ac 1 song.mka          # MP3 → FLAC in Matroska
rmp -i "music/*.wav" -f flac -compression_level 8 flac/
```

---

## 2. How the project values were applied

**Safety.** There is no `unsafe`, and no `unwrap`/`expect`/`panic!` in
non-test code. The muxers never index with a value they didn't compute:
EBML sizes are produced by `ebml::size_len`, track numbers are capped at 126
(one-byte vint), relative block timestamps are checked against the `i16`
range before the cast, and Ogg lacing is generated, not parsed. A packet for
a stream that wasn't added, a codec a container can't hold, or a stream
added after the first packet is a `Message::Container` error. Muxed output
is fuzzed too: `robustness_test.rs` remuxes corrupted Matroska, Ogg and MP4
files (with damaged timestamps) into all three muxers and reads every result
back. A deep debug-build run of 3 seeds × 1,500 rounds was clean.

**Efficiency.** A cluster is assembled in memory and written once with its
exact size, so the muxer needs no seeks while writing, and the only
back-patches are four small fixed-size fields. The Ogg muxer computes each
page's CRC once, over a buffer it already holds. Remuxing a 114 MB,
2-minute 720p H.264 + AAC MP4 to MKV takes 0.11 s (FFmpeg `-c copy`:
0.34 s); 10 minutes of Opus WebM → Ogg takes 0.05 s (FFmpeg: 0.29 s).

**Readability.** The Matroska muxer is one file whose order follows the
file it writes (header, track entry, tags, attachments, clusters, cues,
patch). The Ogg muxer separates page writing, page filling and packet
timing. The CLI holds no media logic: everything it does goes through
`rmp::transcode`.

**Licensing.** Written from the Matroska specification (RFC 9559, RFC
8794), RFC 3533 / 7845 (Ogg, Ogg Opus), the Ogg FLAC mapping and the Vorbis
I specification. FFmpeg's output was the reference for layout choices; no
FFmpeg code was ported.

---

## 3. Architecture map

```
src/
├── main.rs                    rmp CLI: argument parsing, inspect, convert, batch
├── transcode.rs               Options, per-container defaults, routes, the pipeline
├── codecs/vorbis.rs           VorbisParser (mode table → packet durations)
├── container/
│   ├── flac/stats.rs          FlacStats (exact STREAMINFO for any container)
│   ├── mkv/
│   │   ├── ebml.rs            + writers: put_id/size/uint/sint/float/string/void
│   │   ├── tracks.rs          + mux_codec: codec ID, CodecPrivate, BitDepth, ColourSpace
│   │   ├── muxer.rs           MkvMuxer
│   │   └── probe.rs           MkvProbe (.mkv/.mka/.mks), WebmProbe (.webm)
│   ├── ogg/
│   │   ├── muxer.rs           OggMuxer
│   │   └── probe.rs           OggProbe (.ogg/.oga/.ogx), OpusProbe (.opus)
│   └── xiph.rs                + vorbis_headers, lace_headers, base64_encode,
│                                write_comments_with (pictures), is_internal_key
└── io/output.rs               + Output::create_as
```

Data flow for the CLI:

```
rmp args ─► main::parse ─► transcode::Options
                              │
Input::open ─► Demuxer ─► Route per stream ─┬─ Copy ─────────────────────────► Output (muxer by ext / -f)
                                            ├─ Audio: Decoder ► AudioConverter ► Encoder ─┘
                                            └─ Video: Decoder ► VideoConverter ► Encoder ─┘
```

---

## 4. Matroska / WebM muxer

### 4.1 Layout (`muxer.rs`)

```
EBML header        DocType matroska|webm, DocTypeVersion 4, ReadVersion 2
Segment            8-byte size, unknown until finish
├── Void (128 B)   replaced by the SeekHead on finish
├── Info           TimestampScale 1 ms, MuxingApp/WritingApp "RMP",
│                  Duration (8-byte float, patched), Title
├── Tracks
├── Tags           file tags, then one Tag per track (TagTrackUID)
├── Attachments    images as cover.jpg/png..., other RawValue::File entries (not WebM)
├── Cluster …      Timestamp, then SimpleBlocks / BlockGroups
└── Cues           one CuePoint per time, CueTrack/CueClusterPosition/CueRelativePosition
```

This is FFmpeg's `matroskaenc` layout. It also writes `CRC-32` elements
inside level-1 elements; RMP doesn't, as they're optional.

The header is written together with the first cluster, not at the first
`write`. That lets a video track's `DefaultDuration` come from its first
packets: it is written when up to 32 of them all have the same duration
(MP4 and Y4M sources always provide durations; Matroska sources only when
their `DefaultDuration` was set).

**Back-patching** (`patch`). When the output can seek, `finish` writes:

- the `Segment` size;
- the SeekHead (Info, Tracks, Tags, Attachments, Cues);
- the duration (end of the last block);
- each FLAC track's STREAMINFO inside `CodecPrivate`.

An output that can't seek keeps the unknown segment size and the `Void`.
That is still a valid live-style file, which RMP's and FFmpeg's demuxers
read (tested).

### 4.2 Tracks (`tracks::mux_codec`)

`mux_codec` is the inverse of `codec_name`, and a unit test checks that
round trip:

- FFmpeg codec names map to Matroska codec IDs. PCM gets `BitDepth` and
  `A_PCM/INT/LIT`, `A_PCM/INT/BIG` or `A_PCM/FLOAT/IEEE`.
- Extradata becomes `CodecPrivate` in Matroska's layout:

| Codec | `CodecPrivate` |
| --- | --- |
| H.264 / HEVC | the `avcC` / `hvcC` record (Annex B extradata is refused) |
| FLAC | `"fLaC"` + STREAMINFO block (last-block flag set) |
| Opus | `OpusHead`; synthesised for mono/stereo when missing |
| Vorbis | the three headers in Xiph lacing (`xiph::lace_headers`) |
| ALAC | the ALACSpecificConfig without FFmpeg's 12-byte atom header |
| raw video | none; `V_UNCOMPRESSED` with `ColourSpace` = FourCC (`I420`, `Y42B`, `444P`, `Y41B`, `Y800`, `NV12`, RGB/BGR 24, `RGBA`) |
| others | extradata as is |

A `TrackEntry` also carries:

- `TrackNumber`, a stable `TrackUID` (SplitMix64 of the index, so output is
  reproducible), `TrackType` and `FlagLacing 0`;
- `Language`, always written: a missing one reads as `eng`, so RMP writes
  `und`;
- `Name` (the stream title) and `FlagDefault 0` when the source said so;
- `Video{PixelWidth, PixelHeight}` or `Audio{SamplingFrequency, Channels,
  BitDepth}`.

**WebM** accepts only VP8, VP9, AV1, Opus, Vorbis and WebVTT, as FFmpeg's
`webm` muxer does. It never writes attachments.

### 4.3 Blocks and clusters

A packet's timestamp is its `pts` (or `dts`), rescaled to 1 ms. Without
one, it is the end of the track's previous block.

- **SimpleBlock** is used for audio and video, with the keyframe flag for
  video.
- **BlockGroup** is used where a block needs more than a SimpleBlock holds:
  - Subtitles: a `Block` and its `BlockDuration`. WebVTT payloads are
    prefixed with the empty cue-identifier and settings lines (`"\n\n"`),
    which is the WebM WebVTT layout FFmpeg 6.1 writes for both flavours
    (`D_WEBVTT/SUBTITLES`).
  - An end-trimmed Opus packet (see §4.4): a `Block` and
    `DiscardPadding`.

A new cluster starts:

- at every video keyframe (so every cue points at a cluster start);
- when a relative timestamp would leave the `i16` range;
- after 5 s or 5 MiB (FFmpeg's `cluster_time_limit` / `cluster_size_limit`).

Cues are written for video keyframes. Files without video get one cue per
audio track per cluster.

### 4.4 Gapless audio

RMP's convention (MP3_MP4.md §5.2) is `encoder_delay` / `encoder_padding`
metadata in samples, with timestamps that include the priming samples.

- **Opus**: `encoder_delay` (or the `OpusHead` pre-skip) becomes
  `CodecDelay` in ns, and `SeekPreRoll` is 80 ms. Block timestamps are RMP's
  pts unchanged. FFmpeg instead adds the delay to its negative pts, so the
  stored values match.
- **Opus end trimming**: a packet whose duration is shorter than its coded
  length gets `DiscardPadding`. The demuxer now reads `DiscardPadding` back
  the way the Ogg demuxer reports end trimming: the packet's duration
  shrinks to the samples kept, in 1/48000 s so no precision is lost to the
  1 ms ticks. FFmpeg shows it as `Skip Samples` side data; remuxed files
  show the same side data as the source.
- **AAC, MP3 and other audio with `encoder_delay`**: the delay becomes
  `CodecDelay` too. RMP reads it back, so MP4 → MKV → MP4 keeps the edit
  list. FFmpeg reads it as a timestamp shift (audio starts at −21 ms, so it
  stays aligned with video as the MP4 edit list had it), but FFmpeg's
  AAC/MP3 decoders don't drop the priming samples, and FFmpeg's own muxer
  doesn't write `CodecDelay` for them. `encoder_padding` has no Matroska
  equivalent outside Opus and is dropped.

### 4.5 Tags and attachments

| `Metadata` field | Tag |
| --- | --- |
| `title` | `Info/Title` |
| `artist` | `ARTIST` |
| `album` | `ALBUM` |
| `album_artist` | `ALBUM_ARTIST` |
| `genre` | `GENRE` |
| `date` | `DATE` |
| `comment` | `COMMENT` |
| `description` | `DESCRIPTION` |
| `lyrics` | `LYRICS` |
| `track_number` | `PART_NUMBER` |
| `tracks_total` | `TOTAL_PARTS` |
| `disc_number` | `DISC` |
| other string `raw` keys | upper-cased tag names |

These are the names FFmpeg's `ff_mkv_metadata_conv` uses and the demuxer
reads back. Keys that only describe how RMP demuxed something are never
written as tags: `encoder_delay`, `language`, `matroska_codec_id`, `title`
and the rest of `xiph::is_internal_key`. The Ogg comment writer shares the
same filter.

Images become `AttachedFile`s named `cover.<ext>`, with their MIME type and
description. `RawValue::File` entries (fonts from other MKVs) are written
back under their own names.

---

## 5. Ogg muxer

### 5.1 Pages (`muxer.rs`)

Each stream has a fixed serial number (FFmpeg draws random ones), a page
sequence counter and one page being filled.

1. **BOS pages.** Every stream's identification header gets its own BOS
   page, with granule 0. These come first in the file, as RFC 3533 requires
   for multiplexed streams.
2. **Header pages.** The remaining headers follow, flushed so that audio
   starts on a fresh page, which the Opus and Vorbis mappings require.
3. **Audio pages.** These follow FFmpeg's `oggenc` rules:
   - A packet that would not fit in the current page's remaining segments
     starts a new page instead of being split (unless it needs more than
     one page anyway).
   - A page is closed once it spans a second, measured from the start of
     its first packet (`page_duration`), or when it reaches 255 segments.
   - One closed page is always held back, so the stream's real last page
     gets the EOS flag (a stream with no audio gets an empty EOS page).

The result: remuxing FFmpeg-made Opus, FLAC and Vorbis files gives the same
audio pages FFmpeg writes: flags, granule positions, segment counts and
packets per page (tested).

### 5.2 Granule positions

A packet's end, in granule units, is `start + duration`:

- **Units**: 48 kHz for Opus, the sample rate for FLAC and Vorbis.
- **Start**: the packet's pts. A pts that differs from the end of the
  previous packet only by its own rounding (Matroska's 1 ms ticks are up to
  48 Opus samples) is taken as that end, so WebM → Ogg keeps sample-exact
  positions. FFmpeg's own `-c copy` of the 5.1 test file comes out 648
  samples (its `DiscardPadding`) too long; RMP's is exact.
- **Duration**: the packet's duration, or what the payload says
  (`codecs::duration::packet_samples` for Opus and FLAC, `VorbisParser`
  for Vorbis).
- **Opus pre-skip** is already part of RMP's timestamps, so granules count
  it as RFC 7845 requires.
- **Opus end trimming**: a shortened last packet, or `encoder_padding` when
  the last packet arrived full length, lowers the final granule.

Packets are held back one at a time, so a packet with no duration ends
where the next one starts.

### 5.3 Headers per mapping

| Mapping | First packet (BOS) | Other header packets |
| --- | --- | --- |
| Opus | `OpusHead` (extradata, or synthesised for ≤ 2 channels) | `OpusTags` |
| FLAC | `0x7F "FLAC"`, version 1.0, header count, `"fLaC"`, STREAMINFO block | Vorbis comment block, then PICTURE blocks (last-block flag on the final one) |
| Vorbis | identification header | comment header (rebuilt), setup header |

The comments are the stream's metadata, with the file's metadata filling
the gaps, as FFmpeg does with `AV_DICT_DONT_OVERWRITE`. For Opus and
Vorbis, cover art becomes base64 `METADATA_BLOCK_PICTURE` comments.

For FLAC, the BOS page is rewritten on `finish` with the exact STREAMINFO
from `FlacStats`, and its CRC is recomputed. FFmpeg's Ogg FLAC keeps the
encoder's provisional header (no sample count, no MD5); RMP's has both.

### 5.4 The Vorbis parser (`codecs/vorbis.rs`)

A Vorbis packet decodes to `(previous_blocksize + blocksize) / 4` samples.
Which block size a packet uses depends on its mode, and the mode → block
flag table sits at the end of the setup header, after structures only a
full decoder can walk.

Like FFmpeg's `vorbis_parser.c`, RMP reads the setup header backwards from
the framing bit. Each mode is 41 bits, preceded by a 6-bit mode count, so
the table can be found by scanning backwards for the longest run of
plausible modes whose count matches.

The first audio packet only primes the decoder. It counts `blocksize / 4`
samples, which FFmpeg shows as a negative pts, and the Ogg muxer ends it at
granule 0. A test checks that the remaining durations add up to the
fixture's last granule.

---

## 6. The transcode library and the `rmp` CLI

`rmp::transcode::transcode(input, output, &Options) -> Summary` is the old
example's pipeline, made a library function:

- `Options` parses FFmpeg option spellings. Errors are plain strings, so
  they can name the bad option.
- `Summary` reports what each stream became and how many packets and frames
  were written.

What changed from the example:

- **New container defaults.** Stream copy whenever the container holds the
  codec, FLAC otherwise (see the table in the module docs). WebM and
  `.opus` only copy, because RMP has no Opus, Vorbis or VP9 encoder; the
  muxer's error explains when a codec doesn't fit.
- **Conversion options force re-encoding**, as in FFmpeg. `-ar`, `-ac` or
  `-sample_fmt` with a copy default re-encode to FLAC; `-s` or `-pix_fmt`
  decode the video to `rawvideo`.
- **Subtitles are copied** into Matroska, and WebVTT into WebM. `-sn` drops
  them.
- **New options**: `-f`, `-metadata key=value`, `-map_metadata -1`,
  `-c copy`, `-sn`.
- **Encoder flush packets get the output stream id**, and the drain runs in
  stream order, so output is deterministic.

`src/main.rs` adds what a command-line tool needs:

| Command | Does |
| --- | --- |
| `rmp -i in` | Duration, tags, streams, stream metadata |
| `rmp -i in --show` | ...plus one line per packet (stream, pts/dts/duration in seconds, size, keyframe) |
| `rmp [-y\|-n] -i in [options] out` | Converts, printing what each stream became |
| `rmp -i "dir/*.ext" [-f fmt] [-j N] out-dir/` | Batch: each match becomes `out-dir/<stem>.<fmt>`, `N` at a time (default: CPU count) |

Behaviour:

- An existing output is refused unless `-y` is given.
- A failed conversion removes the file it started.
- Globs support `*` and `?` in the file name.
- A closed stdout (`| head`) ends the listing quietly.
- Exit status is 0 on success, 1 on error, and 2 for usage.

The `transcode` example is now a short wrapper over the same function.

---

## 7. Verification

All references are FFmpeg 6.1. The output of every check was read with
`ffprobe` and decoded with `ffmpeg`.

**Remuxes of the fixtures, packet for packet**: `ffprobe -show_packets
-show_data_hash adler32` of the source equals that of RMP's remux. The
compared fields are stream, pts, dts, duration, size, flags, payload hash
and `Skip Samples` side data.

| Source | Output | Result |
| --- | --- | --- |
| `h264_aac.mkv`, `h264_flac_srt.mkv` | `.mkv` | identical |
| `vp9_opus_vtt.webm`, `live_vp9_opus.webm` | `.webm` | identical, including the `DiscardPadding` of the last Opus packet |
| `celt_51.webm`, `mode_switch.webm` | `.webm` | identical |
| `mono.opus`, `st_cover.opus`, `51.opus` | `.opus` | identical; audio pages equal FFmpeg's |
| `flac.oga`, `vorbis.ogg` | `.oga`, `.ogg` | identical; audio pages equal FFmpeg's |

**Cross-container**: the decoded audio (s32le) and video (frame MD5s,
`-fps_mode passthrough`) of the output equal the source's.

| Source | Output | Result |
| --- | --- | --- |
| MP4 H.264 + AAC (B-frames) | `.mkv` | Video identical. Audio identical to FFmpeg's own `-c copy` MKV (both keep the AAC priming, §4.4). |
| MKV (FFmpeg-made) | `.mp4` | identical to the original MP4 |
| M4A with cover art | `.mka` | audio identical; cover art identical as an attachment |
| `vp9_opus_vtt.webm` | `.mkv` | identical |
| WebM Opus (stereo hybrid, 5.1 CELT) | `.opus` | identical, sample-exact length (FFmpeg's own copy of the 5.1 file is 648 samples too long) |
| Ogg Opus with cover | `.mka`, `.webm` | identical (WebM drops the cover, as it has no attachments) |
| Ogg Vorbis | `.webm`, `.mka` | identical |
| Ogg FLAC, 24-bit 5.1 FLAC | `.mka`, `.oga`, `.flac` | identical; STREAMINFO MD5 and sample count match the decoded audio |
| Y4M | `.mkv` (`V_UNCOMPRESSED`) | identical |
| H.264 → `-c:v rawvideo` | `.mkv` | bit-exact with FFmpeg's decode |
| WAV | `.mka` | identical |

**Large files**, compared with FFmpeg's `-c copy` of the same input:

- 2-minute 720p H.264 + AAC, MP4 → MKV: decoded video identical, audio
  identical, 0.11 s vs 0.34 s.
- 10-minute Opus, Ogg → WebM → Ogg: packets identical to the source,
  decoded audio identical.

Two timestamp differences remain, and both are by design:

- **MP4 → MKV audio.** It starts at −21 ms (`CodecDelay`, §4.4). FFmpeg
  instead shifts the video to +21 ms. The relative timing is the same.
- **Opus in WebM.** Some block timestamps differ from FFmpeg's by 1 ms.
  FFmpeg rounds twice (the pts, then the codec-delay offset); RMP rounds the
  exact sample time once. Decoded output is identical either way.

**Tests**:

- **`tests/container/mkv/muxer_test.rs`** covers:
  - round trips of all four Matroska fixtures (packets, durations, trimming,
    extradata, language, `encoder_delay`, tags, duration);
  - the layout (SeekHead in place, segment size, cues);
  - seeking a muxed file;
  - M4A cover art → attachment, with AAC priming kept;
  - FLAC STREAMINFO patching from a provisional header;
  - Ogg Opus → WebM;
  - WebM codec restrictions and API misuse;
  - an unseekable output;
  - raw video through `V_UNCOMPRESSED`.
- **`tests/container/ogg/muxer_test.rs`** covers:
  - round trips of all five Ogg fixtures, with the audio pages compared to
    FFmpeg's;
  - Vorbis parser durations against the granules;
  - WebM → sample-exact Ogg Opus;
  - the FLAC header page rewrite;
  - `.opus` restrictions;
  - the page CRC.
- **`tests/cli.rs`** runs the `rmp` binary: inspect and `--show`, copies
  into every new container, FFmpeg-style options, overwrite protection,
  failure cleanup, unknown options, batch globbing, and the library
  `Options` API with `-f`.
- **`robustness_test.rs`** adds `mkv_and_ogg_muxers_survive_corrupted_input`
  (§2).
- **Tooling**: `cargo fmt --check` and `cargo clippy --all-targets -- -D
  warnings` pass.

```bash
cargo test --test container muxer_test
cargo test --test cli
RMP_FUZZ_ROUNDS=1500 RMP_FUZZ_SEED=7 cargo test --test container mkv_and_ogg_muxers
```

To repeat the FFmpeg comparison for any file:

```bash
p() { ffprobe -v error -show_packets -show_data_hash adler32 -of csv=p=0 \
	-show_entries packet=stream_index,pts,dts,duration,size,flags,data_hash "$1"; }
rmp -y -i in.webm out.webm && diff <(p in.webm) <(p out.webm)
```

---

## 8. Known limitations (with pointers)

| Limitation | Where to start |
| --- | --- |
| No Matroska chapters, `BlockAdditions`, `ContentEncoding` or per-track `DURATION` tags on output. | `MkvMuxer::write_header` (chapters and DURATION tags would go before or after the clusters, with a `Void` reserved for the latter as FFmpeg does). |
| Matroska clusters are buffered in memory (≤ 5 s / 5 MiB, or one GOP). A GOP longer than that is split across clusters, which is valid but makes cues coarser. | `CLUSTER_*_LIMIT` in `muxer.rs`. |
| AAC/MP3 `encoder_padding` is dropped in Matroska, and FFmpeg doesn't trim AAC/MP3 priming from `CodecDelay` (§4.4). | Write `DiscardPadding` on the last block: the cluster is still in memory at `finish`. |
| Ogg: no Theora/Speex mappings, no chained streams. Pages of several streams are written as each closes, not strictly interleaved by time. | `muxer.rs::add` (mapping), `close_page` (interleaving queue). |
| No Opus, Vorbis or VP8/9 **encoders**, so WebM and `.opus` output are stream copies only. | `transcode.rs::default_audio`; encoders would register in `Resolver::encoder_for`. |
| The CLI has one input, no `-map`, `-ss`/`-t`, or filter graph. | `main.rs::parse`, `transcode.rs` routes. `Demuxer::seek` is there for `-ss`. |
| The readme's `rmp::transform` (gain/normalize) API was never implemented; the readme no longer documents it. | A `Filter` implementation in `process/` and an `-af` option. |

---

## 9. How to build on this

### Add a codec to the Matroska muxer

1. Map the FFmpeg name to the Matroska codec ID in `tracks::mux_codec_id`,
   and add it to the demuxer's `codec_name`. The round-trip unit test checks
   that the two agree.
2. If FFmpeg's extradata layout differs from `CodecPrivate`, convert it in
   `tracks::mux_codec`, with the inverse in `parse_entry`.
3. If the codec belongs in WebM, add it to `webm_allows`.
4. Add a fixture and a `round_trip_mkv` test.

### Add an Ogg mapping

1. Add a `Mapping` variant, and map it from the codec in `OggMuxer::add`.
2. Build its header packets in `build_headers`, identification header
   first.
3. Make sure packet durations are known: through `packet_samples`, a parser
   like `VorbisParser`, or packet durations from the source.
4. Mirror it in the demuxer (`new_logical`, `is_header`, `take_header`), and
   add a `round_trip` test that compares pages with an FFmpeg-made fixture.

### Add a CLI option

1. Add the field to `transcode::Options`.
2. Parse it in `Options::apply`, and list it in `Options::is_option` (or
   `is_flag`).
3. Use it in `transcode`.
4. Document it in `USAGE` in `main.rs` and in the readme's CLI reference,
   and add a case to `tests/cli.rs`.

### Rules for contributors touching these muxers

- Compute sizes; never cast a value to a narrower type without checking it
  first (`i16` block timestamps, `u8` segment counts, 24-bit FLAC block
  lengths).
- Anything written before its final value is known must be fixed-size, so
  `finish` can patch it in place: the 8-byte segment size, the 8-byte float
  duration, the 34-byte STREAMINFO.
- Keep both muxers reproducible: no random serials or UIDs, and sort
  anything that comes from a `HashMap`.
- Re-run the FFmpeg comparison in §7 for any change to block, page or
  granule logic, and the muxer fuzz test with a high `RMP_FUZZ_ROUNDS`.

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `container/mkv/muxer.rs` | `libavformat/matroskaenc.c` |
| `container/mkv/ebml.rs` (writers) | `matroskaenc.c` (`put_ebml_*`), `libavformat/ebml.h` |
| `container/mkv/tracks.rs` (`mux_codec`) | `matroskaenc.c` (`mkv_write_codecprivate`), `matroska.c` (`ff_mkv_codec_tags`) |
| `container/ogg/muxer.rs` | `libavformat/oggenc.c` |
| `codecs/vorbis.rs` | `libavcodec/vorbis_parser.c` |
| `container/xiph.rs` (comments, pictures) | `libavformat/vorbiscomment.c`, `flac_picture.c` |
| `container/flac/stats.rs` | `libavcodec/flacenc.c` (STREAMINFO update), `libavformat/flacenc.c` |
| `transcode.rs`, `main.rs` | `fftools/ffmpeg*.c` (in spirit; far smaller) |
