# WAV, Y4M, PCM and ADPCM in RMP

This document covers the formats RMP inherited from FFmpReg, finished to the
same standard as the MP3/MP4 work: the Y4M container, the PCM codecs and the
WAV ADPCM decoders. It covers the design, how each part maps to FFmpeg, how
the work was verified, and how to extend it. Read it before changing anything
under `src/container/riff`, `src/container/y4m`, `src/codecs/pcm` or
`src/codecs/adpcm`.

Before this work, the Y4M demuxer and muxer were `todo!()` (opening a `.y4m`
file aborted the process), the `pcm` and `adpcm` modules were not compiled,
and the WAV demuxer rejected every ADPCM file.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Y4M container** | Demux and mux YUV4MPEG2: 4:2:0 (jpeg/mpeg2/paldv siting), 4:2:2, 4:4:4, 4:1:1, 4:4:4+alpha and gray, 8 to 16 bits. Frame rate, interlacing, aspect ratio, chroma siting and colour range round-trip. Frame-accurate seeking. | `container::y4m::demuxer::Y4mDemuxer`, `container::y4m::muxer::Y4mMuxer` |
| **PCM codec** | Decoder and encoder for `pcm_u8`, `pcm_s8`, `pcm_s16le/be`, `pcm_s24le/be`, `pcm_s32le/be`, `pcm_s64le/be`, `pcm_f32le/be`, `pcm_f64le/be`, `pcm_alaw`, `pcm_mulaw`. The encoder accepts frames in any sample format, interleaved or planar. | `codecs::pcm::{PcmDecoder, PcmEncoder}` |
| **Sample conversion** | swresample-compatible conversion between U8, S16, S32, S64, F32 and F64. | `codecs::pcm::convert` |
| **ADPCM decoders** | IMA ADPCM (`adpcm_ima_wav`, tag `0x0011`) and Microsoft ADPCM (`adpcm_ms`, tag `0x0002`), 4-bit, to interleaved S16. | `codecs::adpcm::AdpcmDecoder` |
| **WAV demuxer** | ADPCM (one block per packet, `fact` chunk duration, `fmt ` extension as extradata), packet timestamps, A-law/µ-law, bounded chunk allocation, data size clamped to the file. | `container::riff::demuxer::RiffDemuxer` |
| **WAV muxer** | A-law, µ-law and 64-bit PCM; correct RIFF size with LIST tags; pad byte for odd-sized data. | `container::riff::muxer::RiffMuxer` |
| **Wiring** | PCM and ADPCM codec IDs registered in `Resolver::decoder_for`; PCM IDs in `Resolver::encoder_for`. | `core::Resolver` |
| **Core** | `SampleFormat::U8` and `S64`; `PixelFormat::YUV411`, `YUVA444`, `Gray`; `Pixel::from_name`, `name`, `plane_sizes`, `frame_size`. The raw codec now takes its pixel layout from `Parameters::pixel_format`. | `core::frame` |
| **Examples** | `pcm_convert` (re-encode to any WAV PCM codec). `decode` now always writes S16LE, converting high-resolution PCM. | `examples/` |

---

## 2. How the project values were applied

**1. Safety**

- There is no `unsafe`, `unwrap`, `expect`, `panic!` or `todo!` in the new
  non-test code. The old WAV demuxer's `try_into().unwrap()` calls are gone.
- Every size read from a file is bounded before allocation:
  - WAV `fmt ` chunks are limited to 64 KiB and `fact` to 64 bytes. `LIST`
    chunks over 1 MiB are skipped unread. Previously a corrupt `LIST` size
    allocated up to 4 GiB.
  - A WAV `data` size is clamped to the bytes the file actually holds.
  - Y4M header lines are limited to 1 KiB, frame headers to 256 bytes, and a
    picture to 256 MiB. Frame size arithmetic is checked.
- Header text is split on character boundaries, so non-ASCII Y4M tags can't
  panic.
- ADPCM state arithmetic saturates. The MS `idelta` update runs in 64-bit and
  clamps to FFmpeg's range.
- Damage is local:
  - A truncated Y4M file ends at its last whole frame. A bad `FRAME` header
    is an error for that read.
  - An ADPCM block with an invalid predictor or step index fails only that
    packet, since every block starts from its own header.
  - A short final ADPCM block decodes the samples it holds.

**2. Efficiency**

- PCM layouts that are already native (`u8`, `s16le`, `s32le`, `s64le`,
  `f32le`, `f64le`) decode with one copy.
- A-law/µ-law compression uses a 16 KiB table built once (`OnceLock`).
  Expansion is a closed-form formula.
- ADPCM decodes straight into the output buffer with no per-sample
  allocation.
- Y4M reads each frame with one `read_vec`. Frames larger than the buffered
  reader's peek window are read directly.

**3. Readability**

- One concept per file:
  - Y4M: `header.rs` (text format), `demuxer.rs`, `muxer.rs`.
  - PCM: `mod.rs` (layouts), `g711.rs`, `convert.rs`, `decoder.rs`,
    `encoder.rs`.
  - ADPCM: `ima.rs`, `ms.rs`, `decoder.rs`.
- Module docs name the FFmpeg file each part corresponds to and the
  specification it follows.

---

## 3. Architecture map

```
src/
├── codecs/pcm/
│   ├── mod.rs        PcmCodec: id <-> layout, coded size, native SampleFormat
│   ├── decoder.rs    stored bytes -> native samples (decode_samples)
│   ├── encoder.rs    any frame -> stored bytes (via convert)
│   ├── convert.rs    Sample, for_each_sample, convert (swresample rules)
│   └── g711.rs       A-law / µ-law expand + table-driven compress
├── codecs/adpcm/
│   ├── mod.rs        AdpcmCodec: id, WAV tag, samples_in_block
│   ├── ima.rs        IMA block decoder
│   ├── ms.rs         MS block decoder
│   └── decoder.rs    core::Decoder impl
├── container/y4m/
│   ├── header.rs     Y4mHeader, parse_header, write_header, colorspace table
│   ├── demuxer.rs    Y4mDemuxer
│   ├── muxer.rs      Y4mMuxer, header_for_stream
│   └── probe.rs      detection by magic, muxer by .y4m / MIME
└── container/riff/
    ├── demuxer.rs    fmt/fact/LIST/data, Layout::{Pcm, Adpcm}, pts, seek
    └── muxer.rs      fmt + LIST + data, finalise sizes
```

---

## 4. WAV in detail

### 4.1 Demuxer (`riff/demuxer.rs`)

The demuxer walks chunks until `data`, reading only `fmt `, `fact` and
`LIST` and skipping every other chunk. `resolve_codec` maps the format tag to
a codec ID and a packet `Layout`. `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`) is
resolved through its sub-format GUID.

| Tag | Codec | Checks |
| --- | --- | --- |
| 1 | `pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_s64le` by bit depth | `block_align = channels × bytes`, `byte_rate = rate × block_align` |
| 3 | `pcm_f32le`, `pcm_f64le` | as above |
| 6 / 7 | `pcm_alaw` / `pcm_mulaw` | 8 bits, as above |
| 2 | `adpcm_ms` | 4 bits, 1-2 channels, `block_align ≥ 7 × channels` |
| 0x11 | `adpcm_ima_wav` | 4 bits, `block_align ≥ 4 × channels` |

Packets:

- **PCM.** Whole sample frames, about 4096 bytes each (FFmpeg's size), with
  `pts` equal to the sample index.
- **ADPCM.** Exactly one block each, so the decoder sees FFmpeg's
  packetisation, with `pts = block index × samples per block`.

**Duration.** For PCM it is `data size / block_align`. For ADPCM it is the
`fact` sample count when it is plausible (non-zero and no more than the blocks
can hold), otherwise the decodable sample count. `fact` excludes the final
block's padding. For a truncated file the decodable count is smaller, and it
wins: RMP reports 0.669 s for a file cut short whose `fact` claims 1.7 s,
where ffprobe reports 1.7 s.

**Seeking** snaps to the containing sample frame (PCM) or block (ADPCM). ADPCM
blocks are self-contained, so seeking is exact at block granularity.

### 4.2 Muxer (`riff/muxer.rs`)

The muxer writes a 16-byte `fmt `, an optional `LIST/INFO`, then `data`.
`finish` adds a pad byte when the data size is odd and patches both sizes. The
RIFF size now includes the `LIST` chunk; previously it was always
`36 + data`. A file over 4 GiB is an error instead of a silently wrapped size.
ADPCM muxing (stream copy) is not supported yet (§10).

---

## 5. PCM in detail

### 5.1 Decoding (`decoder.rs`)

Decoded formats follow FFmpeg's `pcm.c`:

- 24-bit samples become S32 with the value in the top bits (`x << 8`).
- `s8` becomes U8 (`x ^ 0x80`).
- Big-endian layouts are byte-swapped.
- A-law and µ-law expand to S16 (ITU-T G.711).

Bytes that do not form a whole sample frame are dropped.

### 5.2 Conversion (`convert.rs`)

`Sample` holds one value, either as an integer left-aligned in 64 bits or as
an `f64`. The conversions are FFmpeg's `libswresample/audioconvert.c` rules,
without dithering:

| From → to | Rule |
| --- | --- |
| int → wider int | shift left |
| int → narrower int | truncate (`s32 → s16` is `x >> 16`) |
| int → float | `x / 2^(bits-1)` |
| float → int | `lrint(x · 2^(bits-1))`, saturating; NaN becomes 0 |
| u8 | offset binary, silence at 128 |

### 5.3 Encoding (`encoder.rs`)

The encoder converts each sample to the codec's native format first, then
packs it, as FFmpeg's encoder input would be. This matters for
`float → pcm_s24le`: FFmpeg computes `lrint(x · 2³¹) >> 8`. Rounding straight
to 24 bits differs by one LSB, and an early version of this encoder did
exactly that; the verification matrix (§9) caught it.

### 5.4 G.711 (`g711.rs`)

Expansion is the closed form from the recommendation. Compression uses a
16384-entry table indexed by `(sample + 32768) >> 2`. Each entry is the code
whose reconstruction level is nearest, with thresholds halfway between
adjacent levels on that 14-bit grid. This is the same quantiser
`pcm_tablegen.h` builds, so encoded bytes equal FFmpeg's. A unit test checks
that `compress(expand(code)) == code` for every code. The one exception is
µ-law's second zero, `0x7F`.

---

## 6. ADPCM in detail

Both decoders are written from the format specifications. FFmpeg's choices
are followed wherever they affect output.

**IMA (`ima.rs`).**

- **Header.** The block header holds the predictor (also the first output
  sample) and the step index. Like FFmpeg, the step index and the reserved
  byte are read as one signed 16-bit value, and anything outside 0-88 is
  rejected.
- **Data.** 4-byte groups of 8 samples rotate through the channels, low
  nibble first.
- **Difference.** It is `((2·|code| + 1)·step) >> 3`, as in FFmpeg's
  `adpcm_ima_expand_nibble`. The reference algorithm's shift-and-add form
  rounds differently.

**MS (`ms.rs`).**

- **Header.** The block header holds a predictor index (0-6), `idelta`, and
  the first two samples, stored second-sample first.
- **Prediction.** `(s1·c1 + s2·c2) / 256`, with the standard coefficients in
  units of 1/256 and C truncation. FFmpeg stores the coefficients divided by
  4 and divides by 64, which truncates identically. The first version used
  /64 with the full-size coefficients, and the FFmpeg comparison caught it on
  the first coded sample.
- **Adaptation.** `idelta` adapts through the 16-entry table, clamped to
  `[16, i32::MAX/768]`.
- **Coefficients.** Like FFmpeg, the decoder ignores the coefficient table
  in the `fmt ` extension and uses the standard seven pairs.

---

## 7. Y4M in detail

### 7.1 Header (`header.rs`)

The header is one line: `YUV4MPEG2` followed by tagged fields. Parsing
follows FFmpeg's `yuv4mpegdec.c`:

- `W` and `H` are required and must be non-zero.
- `F` gives the frame rate. If it is missing or zero, the rate is 25 fps.
- `A` gives the aspect ratio. `0:0` means unknown.
- `I` gives interlacing: `p`, `t`, `b` or `m`.
- `C` gives the colorspace (table below).
- `X` fields are FFmpeg's extensions. `XYSCSS=` is the colorspace when `C`
  is absent, and `XCOLORRANGE=FULL|LIMITED` is the colour range.
- Unknown tags are ignored.

| `C` token | Pixel format |
| --- | --- |
| `420jpeg`, `420`, `420mpeg2`, `420paldv` | `yuv420p` (chroma siting `center`, `center`, `left`, `topleft`) |
| `422`, `444`, `411`, `444alpha` | `yuv422p`, `yuv444p`, `yuv411p`, `yuva444p` |
| `mono`, `mono9/10/12/16` | `gray`, `gray9le`… |
| `420p9…16`, `422p9…16`, `444p9…16` | `yuv420p10le`… |

`write_header` produces FFmpeg's exact text, including the `XYSCSS` echo, so
remuxed files are byte-identical.

### 7.2 Demuxer and muxer

- **Packets.** The stream is `rawvideo`, and `Parameters::pixel_format` holds
  the FFmpeg pixel format name. Each packet is one picture with
  `pts = frame index` in a `1/fps` time base.
- **Metadata.** Header fields without a `Parameters` slot go into stream
  metadata, under FFmpeg's names: `field_order` (`progressive`/`tt`/`bb`/
  `mixed`), `sample_aspect_ratio`, `chroma_location` and `color_range`
  (`pc`/`tv`). The muxer reads the same keys back.
- **Seeking** uses fixed-stride offsets, which assume plain `FRAME\n`
  headers, as FFmpeg's `yuv4_read_seek` does.
- **Muxer input.** The muxer accepts `rawvideo`, or a codec ID that is
  itself a pixel format name (the raw codec's default). Each packet must be
  exactly one picture, and a compressed stream is refused, so decode first.

---

## 8. Core changes and why

| Change | Reason |
| --- | --- |
| `SampleFormat::U8`, `S64` (+ `is_float`) | FFmpeg decodes `pcm_u8`/`pcm_s8` to U8 and 64-bit PCM to S64. Without them, decoding would be lossy or format-dependent. |
| `PixelFormat::YUV411`, `YUVA444`, `Gray`; `Pixel` helpers | Y4M carries these. `Pixel::from_name`/`name` map to FFmpeg's pixel format names, and `plane_sizes`/`frame_size` centralise the overflow-checked size calculation. |
| Raw codec reads `pixel_format` | Video frames from Y4M report their real layout and depth instead of a hard-coded 8-bit 4:2:0. |
| Resolver PCM/ADPCM entries | Previously every PCM stream went through the raw pass-through, so an `s24` frame was reported as S16. |

---

## 9. Verification

All references are FFmpeg 6.1.

- **PCM decoding.** All 9 WAV layouts (u8, s16, s24, s32, s64, f32, f64,
  A-law, µ-law), mono and stereo, decode **bit-exact** with `ffmpeg -f s16le`.
- **PCM encoding.** The full source × target matrix is **bit-exact** against
  FFmpeg's encoders.
  - Sources: s16, s24, f32, f64, u8, A-law, MS ADPCM and IMA ADPCM.
  - Targets: u8, s16, s24, s32, s64, f32, f64, A-law and µ-law.
  - The check uses the `pcm_convert` example and compares `data` chunks.
- **ADPCM.** IMA and MS decode **bit-exact** in all of these cases:
  - mono and stereo
  - 8, 11.025, 22.05 and 44.1 kHz
  - FFmpeg's default and custom block sizes (256, 512)
  - files truncated mid-block
- **Y4M.** FFmpeg-written files in 8 pixel formats remux **byte-identical**:
  `yuv420p`, `yuv422p`, `yuv444p`, `yuv411p`, `gray`, `yuv420p10le`,
  `yuv444p16le` and `gray16le`. The tests use odd dimensions (34×18), which
  exercise chroma rounding. Packet payloads equal `ffmpeg -f rawvideo`.
- **Regression tests.**
  - `tests/codecs/pcm_codec_test.rs` covers decode and encode against
    FFmpeg for every layout.
  - `tests/codecs/adpcm_decoder_test.rs` covers bit-exactness, timestamps,
    seeking, truncation and invalid layouts.
  - `tests/container/y4m/*` covers headers, seeking, truncation, the remux
    identity and muxer validation.
- **Robustness.** `robustness_test.rs` now corrupts the WAV (PCM and ADPCM)
  and Y4M fixtures and runs demux, decode and seek on the result. It also
  feeds raw corrupted bytes to the ADPCM block decoders with 1-3 channels.
  Deep run in a debug build: 4 seeds × 12 000 rounds per format, clean.
- **Tooling.** `cargo fmt --check` and `cargo clippy --all-targets -- -D
  warnings` pass. CI now lints tests and examples too.

```bash
bash tools/make_fixtures.sh   # WAV/Y4M section needs only FFmpeg's built-in codecs
cargo test
RMP_FUZZ_ROUNDS=6000 RMP_FUZZ_SEED=7 cargo test --test container robustness
```

---

## 10. Known limitations (with pointers)

| Limitation | Where to start |
| --- | --- |
| No ADPCM **encoders**, and ADPCM WAV can't be stream-copied, because the muxer writes only a 16-byte `fmt `. | Write `cbSize` + extension from `Parameters::extradata` in `RiffMuxer::write_header` and use the stream's block align. Encoders would go in `codecs/adpcm`. |
| IMA ADPCM at 2, 3 or 5 bits per sample is rejected. | `ima.rs`: FFmpeg's `adpcm_ima_wav_expand_nibble` path; generalise `GROUP_*` by bit depth. |
| MS ADPCM with more than 2 channels is rejected. FFmpeg ≥ 6 accepts it. | `ms.rs`: per-channel nibble order for `channels > 2`. |
| Other WAV codecs (GSM 6.10, G.726, `0x0055` MP3-in-WAV, WAVE_FORMAT_EXTENSIBLE channel masks) | `resolve_codec`. MP3-in-WAV can reuse the MP3 decoder. |
| The WAV muxer doesn't write WAVE_FORMAT_EXTENSIBLE for more than 2 channels or more than 16 bits. FFmpeg and most readers accept the plain form. | `RiffMuxer::write_header`. |
| No RF64/BW64 for files over 4 GiB. Muxing fails cleanly. | New chunk handling in both RIFF files. |
| Sample-format conversion has no dithering, and there is no channel remix or resampling in the pipeline. | Done in later batches: `process::audio` ([MKV_FLAC_OPUS_PROCESSING.md](MKV_FLAC_OPUS_PROCESSING.md)) and `process::dither` ([FRAGMENTED_MP4_DITHER.md](FRAGMENTED_MP4_DITHER.md)). |
| Y4M per-frame parameters (`FRAME Ixyz`) are skipped, not applied, and seeking assumes plain `FRAME\n` headers. | `Y4mDemuxer::read` / `seek`. |

---

## 11. How to build on this

### Add a PCM layout

1. Add a `PcmCodec` variant, with its `id`, `coded_bytes` and
   `sample_format`.
2. Add arms to `decode_samples` and `encoder::store`.
3. If WAV can carry it, map the tag in `resolve_codec` and in
   `wav_format_from_codec`.
4. Add it to the `make_fixtures.sh` loop. The PCM tests pick it up from
   `CODECS`.

### Add an ADPCM variant

1. Add an `AdpcmCodec` variant: ID, WAV tag, `header_bytes`,
   `samples_in_block`.
2. Write the block decoder next to `ima.rs` and `ms.rs`, and dispatch to it
   in `decoder::decode_block`.
3. Add a fixture with its FFmpeg reference to `make_fixtures.sh` and a row to
   `CASES` in `adpcm_decoder_test.rs`.

### Rules for contributors

- WAV chunks: never allocate a chunk size read from the file without a
  bound. Skip unknown chunks.
- Keep PCM and ADPCM output bit-exact with FFmpeg. Run the fixture tests,
  and for conversion changes rerun the matrix in §9.
- Run the robustness test with a high `RMP_FUZZ_ROUNDS` before merging
  parser changes.

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `container/riff/demuxer.rs` | `libavformat/wavdec.c`, `riffdec.c` |
| `container/riff/muxer.rs` | `libavformat/wavenc.c`, `riffenc.c` |
| `container/y4m/{header,demuxer}.rs` | `libavformat/yuv4mpegdec.c` |
| `container/y4m/muxer.rs` | `libavformat/yuv4mpegenc.c` |
| `codecs/pcm/{decoder,encoder}.rs` | `libavcodec/pcm.c` |
| `codecs/pcm/g711.rs` | `libavcodec/pcm_tablegen.h` |
| `codecs/pcm/convert.rs` | `libswresample/audioconvert.c` |
| `codecs/adpcm/{ima,ms}.rs` | `libavcodec/adpcm.c`, `adpcm_data.c` |
