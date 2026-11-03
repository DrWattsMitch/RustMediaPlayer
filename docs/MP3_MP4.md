# MP3 and MP4 support in RMP

This document explains how MP3 and MP4 support was added to RMP. It covers the
design, how each part maps to FFmpeg, how the work was verified, and how to
extend it. Read it before changing anything under `src/container/mp3`,
`src/container/mp4` or `src/codecs/mp3`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **MP3 container** | Demux MPEG-1/2/2.5 Layer I/II/III streams. Reads ID3v2.2/2.3/2.4 and ID3v1 tags and Xing/Info/VBRI headers (exact duration, VBR seek table, LAME gapless info). Seeks and resyncs after damaged data. | `container::mp3::demuxer::Mp3Demuxer` |
| | Mux MPEG audio frames with an ID3v2.4 tag and a Xing/Info frame (frame count, byte count, seek table, original LAME tag). | `container::mp3::muxer::Mp3Muxer` |
| **MP3 codec** | Layer III decoder (MPEG-1, MPEG-2 LSF, MPEG-2.5; mono/stereo/joint/intensity stereo; long/short/mixed blocks). Output is interleaved S16. | `codecs::mp3::Mp3Decoder` |
| **MP4 container** | Demux non-fragmented MP4/M4A/M4V/MOV: all tracks, sample index, B-frame timing, edit lists, keyframe-aware seeking, iTunes metadata and cover art. | `container::mp4::demuxer::Mp4Demuxer` |
| | Mux H.264, HEVC, AV1, MPEG-4 Part 2, AAC, MP3, ALAC, Opus and FLAC into MP4 with full sample tables, edit lists, 64-bit offsets and iTunes metadata. | `container::mp4::muxer::Mp4Muxer` |
| **Wiring** | Format detection by content, muxer selection by extension/MIME, and decoder selection by codec ID. | `container::detect`, `container::select_muxer`, `core::Resolver` |
| **Core** | `Packet::keyframe`, `Parameters::extradata`, `CodecPrivate` codec state, `Time::rescale`/`ticks`/`base`, `BufferedReader`, `ByteCursor`, `MemoryIo`. | `core`, `io` |
| **Examples** | `probe`, `remux`, `mp3_to_wav`, `seek_check`. | `examples/` |

In FFmpeg terms, `rmp -i in.mp4 -c copy out.mp4`-style remuxing works for both
formats, and so does MP3 → PCM decoding. RMP still has no AAC or H.264 decoder,
so MP4 *decoding* needs a new codec (see §8).

---

## 2. How the project values were applied

**1. Safety**

- There is no `unsafe` code. There is also no `unwrap`, `expect`, `panic!` or
  `todo!` in any new non-test code. Release builds use `panic = "abort"`, so a
  panic would crash the host application. Malformed input therefore always
  becomes a `Result::Err`, end-of-stream, or concealed silence.
- Every parse of untrusted bytes goes through bounds-checked readers:
  - `io::ByteCursor` handles in-memory structures (ID3 frames, MP4 boxes,
    descriptors).
  - `io::BufferedReader` handles streams.
  - `codecs::mp3::bits::BitReader` handles bitstreams. It returns zero bits
    past the end of the data instead of reading out of bounds, which is the
    same contract the reference decoder relies on.
- Every count and size read from a file is checked against the bytes that
  could actually hold it before anything is allocated:
  - Sample-table entry counts are checked against the box size.
  - A constant-size sample count is checked against the file size.
  - The `moov` box is limited to 256 MiB, an ID3 tag to 64 MiB, a single sample
    to 256 MiB, and a track to 2²² samples.
- A truncated download stays playable. Samples or frames past the end of the
  file end the stream instead of failing the open.
- Arithmetic that a hostile file can influence uses `checked_*`/`saturating_*`
  or 128-bit intermediates (`core::rescale_ticks`).
- Robustness tests mutate and truncate real files thousands of times and run
  the demuxers and decoder over the result (§7). Debug builds also turn
  integer overflow into a test failure.

**2. Efficiency**

- The MP3 decoder uses the standard fast factorisations (9-point DCT-III
  inside the 36-point IMDCT, 32-point DCT-II for the polyphase filterbank,
  table-driven Huffman decoding) on fixed-size stack arrays. It allocates
  nothing per granule. It decodes about 470× real time, within about 5% of
  FFmpeg's single-threaded `mp3float` decoder.
- The MP4 demuxer loads `moov` once, then builds a flat 32-byte-per-sample
  index. Each `read` is one seek (skipped when already in place) and one read.
  Choosing which track to read next compares timestamps exactly in `u128`.
  Remuxing a 76 MB, 5-minute file takes 0.33 s (FFmpeg: 0.27 s).
- `BufferedReader` provides a 64 KiB window and free seeks inside that window,
  and large reads bypass the buffer.

**3. Readability**

- Each file covers one concept: frame header, ID3, Xing, framing, boxes,
  sample tables, sample descriptions, metadata, demuxer and muxer.
- Module docs name the FFmpeg source file each part corresponds to, and name
  the ISO clause where it helps.
- Constants are named, not magic, for example `MAX_RESERVOIR_BYTES`,
  `VISUAL_ENTRY_FIELDS` and `MDAT_HEADER_SIZE`.

---

## 3. Architecture map

```
src/
├── codecs/mp3/                  MPEG audio codec
│   ├── header.rs                FrameHeader::parse — shared by container + decoder
│   ├── bits.rs                  BitReader (MSB-first, zero-padded)
│   ├── layer3.rs                side info → scalefactors → Huffman → stereo → IMDCT
│   ├── synth.rs                 polyphase synthesis filterbank → PCM
│   ├── decoder.rs               Mp3DecoderState (reservoir, overlap), Decoder impl
│   └── tables.rs                generated constants (tools/gen_mp3_tables.py)
├── container/mp3/
│   ├── probe.rs                 detection (ID3 or ≥3 consistent frames)
│   ├── id3.rs                   ID3v1/v2 read, ID3v2.4 write
│   ├── xing.rs                  Xing/Info/VBRI/LAME read + write, TOC math
│   ├── demuxer.rs               framing, resync, duration, seek
│   └── muxer.rs                 tag + info frame + frames; finalises info frame
├── container/mp4/
│   ├── probe.rs                 detection by first box type
│   ├── boxes.rs                 BoxIter, find/find_path, box writers, Put trait
│   ├── sample_table.rs          stsz/stz2/stco/co64/stsc/stts/ctts/stss → Vec<Sample>
│   ├── codec.rs                 stsd entries ↔ CodecInfo, esds, AudioSpecificConfig
│   ├── metadata.rs              udta/meta/ilst read + write
│   ├── demuxer.rs               moov loading, tracks, interleaving, edit lists, seek
│   └── muxer.rs                 ftyp/mdat/moov writing, chunking, sample tables
├── io/
│   ├── buffered.rs              BufferedReader over any Io
│   ├── cursor.rs                ByteCursor over &[u8]
│   └── memory.rs                MemoryIo (tests, in-memory muxing)
└── core/
    ├── packet/packet.rs         + keyframe
    ├── stream/mod.rs            + Parameters::extradata
    ├── context/state.rs         + CodecPrivate (typed per-codec state)
    ├── time.rs                  + ticks(), base(), rescale(), rescale_ticks()
    └── resolver.rs              "mp3" → Mp3Decoder
```

The data flow is unchanged from RMP's existing design:

```
Input::open ─► container::detect ─► Demuxer::read ─► Packet ─► Decoder ─► Frame
                (riff, y4m, mp4, mp3)                 │
                                                      └─► Muxer::write (remux)
Output::create ─► container::select_muxer (by extension / MIME)
```

Detection order matters. RIFF, Y4M and MP4 have real magic numbers. MPEG
audio has only an 11-bit sync word, so it is probed **last**.

---

## 4. MP3 in detail

### 4.1 Frame header (`codecs/mp3/header.rs`)

`FrameHeader::parse` decodes the 32-bit header into version, layer, bitrate,
sample rate, channel mode, frame size and samples per frame. It rejects
reserved values and **free-format** streams (bitrate index 0), which need
out-of-band size detection and are almost never used. Frame size is
`samples/8 × bitrate / rate + padding`, or `(12 × bitrate / rate + padding) × 4`
for Layer I. `is_compatible` checks the fields that may never change within a
stream. Both the demuxer and the probe use it to reject false sync words, as
FFmpeg's `mp3_read_probe` does.

### 4.2 Demuxer (`container/mp3/demuxer.rs`)

Opening a file:

1. Read any number of leading ID3v2 tags (`id3::parse_v2`). A damaged tag is
   ignored, never fatal.
2. If the stream size is known, read the last 128 bytes for an ID3v1 tag and
   exclude it from the audio range (`data_end`).
3. Find the first frame **confirmed by its successor**
   (`find_confirmed_frame`). The scan returns `Scan::Advance(n)` when a
   candidate can't be confirmed yet, so the window is refilled without
   skipping a real frame that sits across the buffer edge.
4. Parse a Xing/Info/VBRI tag from that frame. If one is present, the frame
   carries no audio and packets start after it.
5. Duration is `frames × samples_per_frame` from the info tag. Without one,
   it is a constant-bitrate estimate from the byte count, as in FFmpeg.

Reading returns exactly one frame per packet, with `pts = dts = sample index`
and time base `1/sample_rate`. If a header is invalid or incompatible, the
demuxer resyncs.

Seeking uses the Xing TOC with linear interpolation (`xing::toc_offset`) when
there is one. Otherwise it uses a bitrate-proportional offset. It then resyncs
to a real frame and sets `pts` to the frame-aligned target. After a seek the
decoder lacks the previous frame's bit reservoir, so one or two frames are
concealed as silence, which FFmpeg also does.

### 4.3 Muxer (`container/mp3/muxer.rs`)

The muxer writes an ID3v2.4 tag (UTF-8 text frames, `COMM`, `USLT`, `APIC`;
raw `T***` keys round-trip), then a zeroed info frame, then the frames.
`finish` seeks back and fills in the info frame (`Info` if every frame shares
one bitrate, otherwise `Xing`): frame count, byte count, a 100-entry TOC and,
when the source had one, the original 36-byte LAME tag with its CRC
recomputed. Every packet is validated as a run of whole, compatible frames
before anything is written.

### 4.4 Layer III decoder (`codecs/mp3/*`)

These are the stages of ISO/IEC 11172-3 Annex A, in the same order FFmpeg's
`mpegaudiodec_template.c` runs them:

| Stage | Function |
| --- | --- |
| Side information (granule/channel parameters) | `layer3::read_side_info` |
| Main data = bit reservoir tail + frame payload | `Mp3DecoderState::decode_frame` |
| Scalefactors (+ subblock gain, preflag, M/S normalisation) | `layer3::decode_scalefactors` |
| Huffman decoding with inline requantisation `sign·|v|^(4/3)·scale` | `layer3::huffman` |
| Intensity and mid/side stereo | `layer3::intensity_stereo`, `midside_all` |
| Short-block reordering | `layer3::reorder` |
| Alias reduction | `layer3::antialias` |
| IMDCT (36/12-point) with overlap-add | `layer3::imdct_granule` |
| Frequency inversion | `layer3::change_sign` |
| Polyphase synthesis (DCT-II + windowing) | `synth::synth_granule` |

The arithmetic and the compact table layout come from **minimp3**, which is
public domain (CC0). That makes it licence-compatible with Apache-2.0, whereas
porting FFmpeg's LGPL code would not be. The tables in `tables.rs` are
generated by `tools/gen_mp3_tables.py` and never edited by hand.

Two changes from the reference decoder:

- **Granule 0 never reuses scalefactors.** The reference copies `scfsi` bits
  that are really the header's private bits into granule 0. RMP clears them,
  as the specification requires.
- **Stereo tools require joint-stereo mode.** Intensity and M/S stereo are
  applied only when the channel mode is joint stereo, matching FFmpeg.

State between frames lives in `Mp3DecoderState`: IMDCT overlap, synthesis
history and up to 511 reservoir bytes. It is stored in the new typed
`CodecPrivate::Mp3` slot of `AudioState`, so the existing `Decoder` trait
didn't change. Damaged frames, and frames whose reservoir data is missing,
still produce correctly-sized silent output, so timestamps stay continuous.
`concealed_frames` counts them.

---

## 5. MP4 in detail

### 5.1 Opening (`demuxer.rs` → `load_moov`)

The demuxer walks top-level boxes until `moov` and loads it into memory. It
reads only box headers, so `mdat` is skipped with a seek. Both "moov first"
(faststart) and "moov last" files work. Everything below `moov` is parsed
from that slice with `boxes::BoxIter`, so no further I/O happens until
samples are read.

For each `trak`:

- **`mdhd`**: timescale, media duration, language (stored as stream metadata
  `language`).
- **`hdlr`**: `vide` → video, `soun` → audio, `sbtl/subt/text` → subtitle.
- **`stsd`** (`codec.rs`): the first sample entry gives the codec ID, using
  FFmpeg's names, plus dimensions, rate, channels, bitrate and **extradata**.
  Extradata is the payload of `avcC`/`hvcC`/`av1C`/`vpcC`/`dOps`/`dfLa`/`alac`,
  or the DecoderSpecificInfo inside `esds`. QuickTime sound description
  versions 1 and 2 and the `wave` wrapper are handled. For AAC, the sample
  rate and channels come from the AudioSpecificConfig, as in FFmpeg.
- **`stbl`** (`sample_table.rs`): the run-length tables are expanded into
  `Vec<Sample { offset, dts, size, cts, duration, keyframe }>`. This is the
  same idea as FFmpeg's `mov_build_index`.
- **`edts/elst`**: see §5.2.

A malformed track is skipped and the rest of the file stays playable. A file
with only `mvex`/`moof` fragments is rejected with a clear message.

### 5.2 Timestamps and edit lists

`core::Time` is **unsigned**, but MP4 can describe negative decode times. Two
rules handle this:

- **Video (B-frames).** The first real edit's `media_time` is subtracted from
  **pts only**, so presentation starts at 0, and dts stays as stored (≥ 0).
  FFmpeg instead shifts dts below zero. The presentation timeline, payloads,
  keyframes and decoded frames are identical (verified in §7). Only the dts
  numbering differs.
- **Audio priming (AAC).** The edit becomes container-neutral stream metadata,
  `encoder_delay` and `encoder_padding`, instead of a negative pts.

**The gapless convention.** RMP uses these two metadata keys in every format.
Both are measured in samples to drop from the start or end of the *decoded
output*, which is FFmpeg's `skip_samples` meaning:

- The MP3 demuxer reports `LAME delay + 529` and `LAME padding − 529`.
- The MP4 demuxer reports the edit list.
- The MP4 muxer turns the keys back into an edit list.
- The MP3 muxer re-emits the original LAME tag (stored under `lame_tag`).

As a result, MP3→MP3, MP4→MP4 and MP3→MP4 remuxes all keep gapless playback.

### 5.3 Reading and seeking

`read` returns the sample with the smallest decode time across all tracks.
The comparison is exact (`dts_a × scale_b` vs `dts_b × scale_a`), and ties go
to the lower file offset. `seek(t)` snaps tracks that have a sync-sample table
(`stss`, meaning video) back to the preceding keyframe. It then aligns every
other track to that keyframe's time, as FFmpeg's `mov_read_seek` does.

### 5.4 Metadata (`metadata.rs`)

Reading handles these `ilst` items:

- `©nam`, `©ART`, `aART`, `©alb`, `©gen`, `©day`, `©cmt`, `©lyr` and `desc`
  map to the matching `Metadata` fields.
- `trkn` and `disk` give the number and total.
- `gnre` is an ID3v1 genre index.
- `covr` holds JPEG, PNG or BMP images.
- `----` freeform items go into `raw`, keyed by their name.
- Other atoms also go into `raw`.

Both ISO full-box and QuickTime-style `meta` boxes are accepted. The muxer
writes the same fields and adds `©too = RMP`.

### 5.5 Muxer (`muxer.rs`)

```
ftyp(isom, iso2, [avc1], mp41) | mdat(size=1 + 64-bit size) | moov
```

- Streams must be added before the first `write`.
- Packet timestamps are rescaled to the track timescale. Audio uses the sample
  rate. Video uses the stream time base, doubled until it is at least 10 000,
  as FFmpeg does.
- A dts that goes backwards is rejected.
- If a packet has no timestamps, AAC and MP3 durations are derived from the
  frame. Any other codec without timestamps is an error.
- Consecutive packets of one track form a chunk.
- `finish` patches the `mdat` size and writes `moov`, containing:
  - `mvhd`/`tkhd`/`mdhd`, switching to version 1 when values pass 32 bits
  - an edit list for B-frame delay or audio priming and padding
  - `stsd` (`codec::write_stsd`), `stts`, `ctts` (version 1 when offsets are
    negative), `stss` (video only, when some frames aren't keyframes), `stsc`,
    `stsz` (constant size when possible), and `stco`, or `co64` once an
    offset passes 4 GiB
  - `udta`

---

## 6. Core changes and why

| Change | Reason |
| --- | --- |
| `Packet::keyframe` (default `true` in `Packet::new`) | MP4 needs it for `stss` and for seeking. It is FFmpeg's `AV_PKT_FLAG_KEY`. The raw encoder sets it from `VideoFrame::keyframe`. |
| `Parameters::extradata` | Remuxing H.264/HEVC/AAC needs the codec configuration record. It is FFmpeg's `codecpar->extradata`. |
| `CodecPrivate` in `AudioState` | Holds typed per-codec decoder state without changing the `Decoder` trait or hiding floats in `buffer: Vec<u8>`. |
| `Time::ticks/base/rescale`, `rescale_ticks` | Muxers must convert between time bases without overflow (`av_rescale_q`). |
| `io::BufferedReader`, `ByteCursor`, `MemoryIo` | Reusable safe I/O primitives for any future demuxer or muxer, and for tests. |

The RIFF demuxer was adjusted only for the new `Packet` field
(`Packet::new`).

---

## 7. Verification

The following was run while implementing. The automated parts live in
`tests/`.

- **Decoder accuracy against FFmpeg**, across 9 encodings: MPEG-1 at 32, 44.1
  and 48 kHz; MPEG-2 at 22.05 kHz; MPEG-2.5 at 8 and 11.025 kHz; mono and
  stereo; joint and L/R stereo; CBR (64–320 kbit/s) and VBR; LAME and shine
  encoders. Every case was within **±1 LSB, at 97–104 dB SNR**.
  `tests/codecs/mp3_decoder_test.rs` locks this in against an FFmpeg
  reference decode.
- **MP4 remux identity.** For AAC M4A, MP3-in-MP4 and HEVC, `ffmpeg -f
  framemd5` of the original and the RMP remux is identical. That covers pts,
  dts, duration, size and MD5 of every packet. For H.264 with B-frames, plus
  faststart and `.mov` variants, the pts, sizes, keyframes and decoded video
  and audio frames are identical. Only dts numbering differs (§5.2).
- **MP3 remux.** FFmpeg's decode of the remuxed file is byte-identical to its
  decode of the source, so gapless information is preserved.
- **Seeking** lands on frame boundaries (MP3) and keyframes (MP4).
- **Robustness.** `tests/container/robustness_test.rs` truncates files,
  flips random bytes and smears runs of `0xFF`/`0x00` over all fixtures. It
  then runs demux, seek and decode on the result. The default is 150 rounds;
  deep runs of 3 × 6000 rounds were clean.
- **Tooling.** `cargo fmt --check` and `cargo clippy -- -D warnings` pass, as
  in the CI workflow.

Run the tests:

```bash
cargo test                                   # everything
cargo test --test container mp4              # one area
RMP_FUZZ_ROUNDS=20000 RMP_FUZZ_SEED=7 cargo test --test container robustness
```

To regenerate fixtures, run `tools/make_fixtures.sh` (requires FFmpeg with
libmp3lame and libx264). If you change a command there, update the asserted
packet counts.

---

## 8. Known limitations (deliberate, with pointers)

| Limitation | Where to start |
| --- | --- |
| No AAC/H.264/HEVC **decoders**, so MP4 is remux-only. | Add `codecs/aac` and register it in `Resolver::decoder_for`. Extradata is already present. |
| No MP3 **encoder**. MP3 output requires already-encoded frames. | A Layer III encoder (psychoacoustic model + bit allocation) is a large, separate project. |
| Fragmented MP4 (`moof`/`mvex`, DASH/CMAF) is rejected. | Extend `load_moov` to collect `moof`/`traf`/`trun` and append to each `Track::samples`. |
| No `+faststart` (`moov` before `mdat`) when muxing. | In `finish`, build `moov` first, add its size to every chunk offset, then move `mdat`. This needs a second pass or a temporary file. |
| The MP4 muxer writes only the first sample description, so mid-stream codec changes aren't supported. | `codec::write_stsd` |
| Raw PCM tracks in MOV (`sowt`/`twos`), one sample per audio frame, can hit the 2²² sample cap on long files. | Group PCM samples per chunk in `sample_table::build_index`, as FFmpeg does. |
| MP3 free-format bitrate is unsupported. | `FrameHeader::parse` plus minimp3's `mp3d_find_frame` logic. |
| MP3 Layer I/II are demuxed and muxed, but not decoded. | minimp3's `L12_*` functions port the same way `layer3.rs` did. |
| The `Decoder` trait returns a clone of `State` on every call. For MP3 that copies about 8 KB per frame. | A future API change: `DecodeOut` could drop `state`, since it is already `&mut`. |
| `HttpIo` buffers everything it reads, so reading the ID3v1 tag (or a moov-at-end MP4) over HTTP downloads the whole file. | Range requests in `io/http.rs`. |

---

## 9. How to build on this

### Add a codec to the MP4 muxer

1. `codec.rs::muxer_supports`: add the codec ID.
2. `write_visual_entry` or `write_audio_entry`: map it to its sample-entry
   fourcc and configuration box.
3. If it can come from a demuxer, make sure `parse_*_entry` produces the same
   ID and extradata. Round-tripping is then automatic.
4. Add a fixture and a test to `tests/container/mp4/muxer_test.rs`.

### Add a decoder (e.g. AAC)

1. Create `src/codecs/aac/` with a type implementing `core::Decoder`.
2. If it needs state between packets, add a `CodecPrivate::Aac(Box<…>)`
   variant.
3. Register it in `Resolver::decoder_for` by codec ID.
4. Validate it against FFmpeg the way §7 does: decode to raw PCM and compare
   per-sample error and SNR.

### Add a container

Follow the `mp3`/`mp4` layout. Then:

1. Implement `core::Probe` in `probe.rs`.
2. Register the demuxer in `container/detect.rs`. Put formats with strong
   magic numbers before weak ones.
3. Register the muxer in `container/select.rs`.
4. Parse with `ByteCursor` and `BufferedReader`, and add the format to the
   robustness test.

### Add a metadata field

1. Add it to `core::Metadata`.
2. Map it in `id3::apply_frame` and `id3::build_v2` (MP3), and in
   `metadata::parse_item` and `metadata::write_udta` (MP4).
3. Extend the round-trip tests.

### Rules for contributors touching parsers

- Never index a slice with a value read from a file without a bound check.
  Prefer `get`, `ByteCursor` or `checked_sub`.
- Never allocate a size read from a file until it is checked against the
  bytes that could actually contain it.
- Return `Err` for "this is not a valid file". Prefer end-of-stream or
  concealment for "this part is damaged".
- Keep generated tables generated. Change `tools/gen_mp3_tables.py`, not
  `tables.rs`.
- Run the robustness test with a high `RMP_FUZZ_ROUNDS` before merging parser
  changes.

### Useful examples

```bash
cargo run --release --example probe      -- song.m4a        # ffprobe-like summary
cargo run --release --example remux      -- in.mp4 out.mp4  # stream copy (mp3/mp4/m4a)
cargo run --release --example mp3_to_wav -- in.mp3 out.wav  # decode
cargo run --release --example seek_check -- in.mp4          # seek behaviour
```

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `codecs/mp3/header.rs` | `libavcodec/mpegaudiodecheader.c` |
| `codecs/mp3/{layer3,synth,decoder}.rs` | `libavcodec/mpegaudiodec_template.c`, `mpegaudiodsp*.c` |
| `container/mp3/demuxer.rs`, `xing.rs` | `libavformat/mp3dec.c` |
| `container/mp3/muxer.rs` | `libavformat/mp3enc.c` |
| `container/mp3/id3.rs` | `libavformat/id3v1.c`, `id3v2.c`, `id3v2enc.c` |
| `container/mp4/{demuxer,sample_table,codec,metadata}.rs` | `libavformat/mov.c`, `isom.c` |
| `container/mp4/muxer.rs` | `libavformat/movenc.c` |
