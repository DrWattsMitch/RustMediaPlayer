# Matroska, FLAC, Ogg, Opus and the processing layer in RMP

This document explains the fourth batch of work on RMP: a Matroska/WebM
demuxer, a FLAC decoder, encoder and container, an Ogg demuxer, an Opus
decoder, and the processing layer (sample-format, channel and rate
conversion, pixel-format and scaling conversion) that turns RMP from a
remuxer and decoder into a transcoder. It covers the design, how each part
maps to FFmpeg (and libopus), how it was verified and how to extend it. Read
it before changing anything under `src/container/{mkv,flac,ogg}`,
`src/codecs/{flac,opus}`, `src/core/resampler` or `src/process`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Matroska / WebM** | Demux `.mkv`, `.mka`, `.webm`: every track type, Xiph/EBML/fixed lacing, header-stripping compression, unknown-size (live) segments and clusters, `SeekHead`-directed loading of `Cues`/`Tags` after the clusters, cue-based seeking (or a lazily built cluster index without cues), resync after damage, tags, attachments (cover art), `CodecDelay` → `encoder_delay`, WebVTT/SubRip/ASS subtitles. | `container::mkv::demuxer::MkvDemuxer` |
| **FLAC codec** | Decoder for every FLAC feature (fixed/LPC predictors, Rice and escaped partitions, wasted bits, all stereo decorrelations, 4–32 bits, 1–8 channels), CRC-checked. Encoder with FFmpeg's compression levels 0–12. | `codecs::flac::{FlacDecoder, FlacEncoder}` |
| **FLAC container** | Demux native `.flac` (ID3v2 skip, Vorbis comments, pictures, seek table, bisection seeking, CRC-confirmed frame delimiting). Mux with exact STREAMINFO (sizes, sample count, MD5) rewritten on `finish`. | `container::flac::{demuxer::FlacDemuxer, muxer::FlacMuxer}` |
| **Ogg** | Demux `.ogg`, `.opus`, `.oga`: Opus, FLAC and Vorbis mappings, CRC-checked pages, granule → pts, Opus pre-skip and end trimming, Vorbis comments and `METADATA_BLOCK_PICTURE`, bisection seeking with the Opus 80 ms pre-roll. | `container::ogg::demuxer::OggDemuxer` |
| **Opus codec** | Decoder for SILK, hybrid and CELT packets of 2.5–120 ms, mono, stereo and multistream (mapping families 0, 1, 255), with mode transitions, redundancy frames and output gain. 48 kHz `F32` output in FFmpeg's channel order. | `codecs::opus::OpusDecoder` |
| **Processing** | `AudioConverter`: sample format and layout (exact), channel remix with FFmpeg's default matrices, resampling. `VideoConverter`: scaling (point, bilinear, bicubic, Lanczos) and YUV/RGB/gray/NV12 conversion with BT.601/709/2020 and limited/full range. | `process::{audio, video}` |
| **Resampler** | The `core/resampler` stub is now a polyphase windowed-sinc resampler. | `core::resampler::Resampler` |
| **MP4** | `dfLa` ↔ STREAMINFO and `dOps` ↔ `OpusHead` on demux and mux, so FLAC and Opus round-trip between MKV/Ogg/FLAC and MP4; dts derived for video packets that only carry pts (Matroska B-frames). | `container::mp4::{codec, muxer}` |
| **Examples** | `transcode` (FFmpeg-style options: `-an -vn -c:a -c:v -ar -ac -sample_fmt -compression_level -s -pix_fmt -sws_flags`), `framecrc` (FFmpeg's `-f framecrc`). | `examples/` |

In FFmpeg terms, these now work:

```bash
transcode in.mkv out.mp4                       # H.264 + AAC stream copy
transcode in.mkv out.mp4 -c:a flac             # copy video, AAC → FLAC
transcode song.opus song.flac                  # Opus → FLAC (48 kHz)
transcode in.flac out.wav -ar 44100 -ac 1 -sample_fmt s32
transcode clip.y4m small.y4m -s 320x180 -pix_fmt yuv444p -sws_flags lanczos
```

---

## 2. How the project values were applied

**Safety.** There is no `unsafe`, and no `unwrap`/`expect`/`panic!` in
non-test code. Every parse goes through bounds-checked readers
(`ByteCursor`, `BufferedReader`, the FLAC/Opus bit readers that return zeros
past the end like the reference decoders do). Sizes read from files are
checked against what could hold them before allocating: a Matroska level-1
element is capped at 64 MiB, a block at 256 MiB and `CodecPrivate` at
16 MiB, a FLAC frame search at 4 MiB, an Ogg packet at 16 MiB and 32
logical streams, an Opus packet at 120 ms (5760 samples). The SILK port uses explicit
`wrapping_*`/`saturating_*` wherever libopus relies on two's complement, so
debug builds (overflow checks on) and release builds behave identically.
Damage becomes an error, end of stream, a dropped page/frame, or silence.

**Efficiency.** Matroska clusters are streamed element by element, never
loaded whole. FLAC residual coding computes exact Rice costs over partition
sums once per order. The resampler uses precomputed polyphase filter banks
(exact phases when the reduced ratio has ≤ 1024 of them, else 1024
interpolated phases). The scaler precomputes per-row/column filter taps.
CELT uses a mixed-radix FFT (radix 2/3/4/5) like libopus' KISS FFT.

**Readability.** One concept per file: `ebml`, `tracks`, `demuxer` for
Matroska; `frame_header`, `streaminfo`, `decoder`, `encoder`, `lpc`, `md5`
for FLAC; `range`, `celt`, `silk`, `mdct`, `header`, `decoder` for Opus.
Module docs name the FFmpeg or libopus source each part ports.

**Licensing.** No FFmpeg (LGPL) code was ported. The algorithms follow the
specifications (RFC 8794/Matroska, RFC 9639 FLAC, RFC 3533/7845 Ogg/Opus,
RFC 6716 Opus). The Opus decoder is a port of **libopus 1.4**, which is
BSD-3-Clause (compatible with RMP's Apache-2.0); the attribution is in the
module docs, and the normative tables are generated by
`tools/gen_opus_tables.py` from a libopus checkout rather than typed in.

---

## 3. Architecture map

```
src/container/mkv/      ebml.rs (vints, headers, values) · tracks.rs (codec IDs, extradata)
                        demuxer.rs (level-1 load, clusters, lacing, cues, seek) · probe.rs
src/container/flac/     demuxer.rs · muxer.rs · probe.rs
src/container/ogg/      demuxer.rs (pages, CRC, packets, granules, seek) · probe.rs
src/container/xiph.rs   Vorbis comments, FLAC PICTURE blocks, base64 (shared by FLAC, Ogg, MKV)
src/codecs/duration.rs  packet duration from the payload (AAC, MP3, Opus, FLAC, PCM)
src/codecs/flac/        frame_header.rs · streaminfo.rs · decoder.rs · lpc.rs · encoder.rs · md5.rs
src/codecs/opus/        header.rs (OpusHead/dOps, TOC durations) · range.rs (entropy decoder)
                        celt.rs · mdct.rs · silk.rs · decoder.rs (modes, multistream) · tables.rs
src/core/resampler/     resampler.rs · filter_bank.rs · delay_line.rs
src/process/            audio.rs (AudioConverter) · video.rs (VideoConverter)
```

Detection order (`container/detect.rs`): RIFF, Y4M, FLAC (before MP3, since
an ID3-tagged FLAC starts like MP3), Matroska, Ogg, MP4, MP3 last.

---

## 4. Matroska / WebM

**Opening.** The EBML header must declare `matroska` or `webm`. The
`Segment`'s level-1 elements are read until the first `Cluster`; elements
the `SeekHead` places later (usually `Cues`, `Tags`) are loaded by seeking
when the input size is known. Unknown-size segments and clusters (live
WebM) are handled by treating a level-1 ID as the end of the open element,
as FFmpeg does.

**Tracks** (`tracks.rs`) map Matroska codec IDs to FFmpeg codec names and
convert `CodecPrivate` to FFmpeg's extradata layout (e.g. `A_FLAC` → bare
STREAMINFO, `A_AAC/MPEG4/LC` without private data → a synthesised
AudioSpecificConfig, `V_MPEG4/ISO/AVC` → avcC as is). `CodecDelay` and the
Opus pre-skip become `encoder_delay`, RMP's gapless key (FFmpeg instead
shifts pts negative).

**Blocks.** `SimpleBlock` and `BlockGroup` are parsed with all three lacing
schemes; header-stripping `ContentCompression` prefixes are restored.
Audio packet durations are derived from the payload (`codecs/duration.rs`),
rounded down like FFmpeg's `AV_ROUND_DOWN`, so `ffprobe` and RMP agree.
WebVTT payloads drop the cue header like FFmpeg's `webvtt` packets.

**Seeking** uses `Cues` when present; otherwise a cluster index is built
lazily by scanning cluster headers. Reading resumes at the cluster that
starts at or before the target; a `SeekFilter` then drops video packets
until each video stream's first keyframe, and other packets before the
target.

## 5. FLAC

**Decoder.** `FrameHeader::parse` validates the sync code, reserved bits,
the UTF-8 coded number and CRC-8. Each subframe (constant, verbatim, fixed
order 0–4, LPC order 1–32) is decoded with 64-bit accumulation where
needed (33-bit side channels), then the stereo decorrelation and the CRC-16
footer. Output is S16 for ≤ 16 bits, else S32 left-aligned, exactly what
`ffmpeg -f s16le/s32le` writes.

**Encoder** (`encoder.rs`, `lpc.rs`). Per frame: choose the cheapest of
independent/left-side/right-side/mid-side; per channel remove wasted bits,
then compare constant, verbatim, fixed orders and LPC (Tukey(0.5) window,
autocorrelation, Levinson-Durbin, coefficient quantisation with precision 15
capped for ≤ 16-bit input); Rice-code residuals with the exact best
partition order and parameters. Levels 0–12 follow FFmpeg's presets; the
block size is 4608 at 44.1/48 kHz (the 105 ms rule) and 1152 at levels 0–2.

**Container.** The demuxer delimits frames like FFmpeg's `flac_parser`: a
candidate next header must parse, carry the expected frame/sample number,
and the bytes before it must pass the CRC-16. Seeking bisects over byte
offsets (each probe resyncs and reads the frame's own sample number),
narrowed by the seek table. The muxer writes STREAMINFO, Vorbis comments
(vendor `RMP`), pictures and 8 KiB of padding, and on `finish` rewrites
STREAMINFO with the real block/frame sizes, sample count and the MD5 of the
decoded audio (it decodes each frame it writes), so stream copies and RMP's
own encodes both get exact headers.

## 6. Ogg

Pages are CRC-checked (polynomial `0x04C11DB7`, unreflected); a damaged
page is dropped and the reader resyncs on the next `OggS`. Logical streams
are identified from their BOS page: `OpusHead`, `0x7F "FLAC"`, or
`\x01vorbis`; other streams are skipped. Header packets (`OpusTags`, FLAC
metadata blocks, Vorbis comment/setup) fill parameters and metadata; Vorbis
extradata is built in FFmpeg's Xiph-laced layout.

**Timestamps.** A page's granule is the sample position at the end of the
last packet completed on it, so pts are found by walking back from the
granule with each packet's duration (`ogg_calc_pts` in FFmpeg). When a page
ends a stream with fewer samples than its packets hold, the last packet is
shortened (Opus end trimming). RMP's timestamps start at 0 with the Opus
pre-skip in `encoder_delay`; FFmpeg starts at `-pre_skip`, so RMP's pts
equal FFmpeg's plus the pre-skip. Duration is the last granule position (as
FFmpeg reports it). Vorbis packets get a pts only where it is known (RMP has
no Vorbis parser for per-packet block sizes).

**Seeking** bisects over byte offsets for the last page of the first stream
whose granule is before `target − 80 ms` and resumes after it, so a decoder
has the RFC 7845 pre-roll to converge.

## 7. Opus

**Packets** (`decoder.rs`). TOC parsing covers codes 0–3 (CBR/VBR, padding),
self-delimited framing for multistream, and the 120 ms limit. Each stream
has a `StreamDecoder` holding SILK and CELT state; transitions between modes
follow `opus_decode_frame`: SILK reset on entering from CELT, the 5 ms CELT
"lost" decode and cross-fade on CELT ↔ SILK switches, redundancy frames with
their fades, and the hybrid band limit. Multistream output is mixed by the
`OpusHead` mapping and family-1 surround is reordered from Vorbis to
FFmpeg's channel order; the header's output gain is applied.

**CELT** (`celt.rs`, `mdct.rs`) is a port of libopus' float build: energy
decoding (Laplace coarse, fine, final), TF/spread/dynalloc/trim, the exact
bit allocator, PVQ band decoding with splitting, stereo and folding,
anti-collapse, denormalisation, inverse MDCT via the N/4 complex FFT, the
pitch post-filter (an in-place IIR comb with the cross-fade), and
de-emphasis. Lost frames (only used internally for transitions) use noise
concealment.

**SILK** (`silk.rs`) is a bit-exact port of libopus' fixed-point decoder:
indices, shell-coded pulses and signs, gain dequantisation, NLSF decoding
and stabilisation, NLSF → LPC with `LPC_fit` and the inverse-prediction-gain
check, pitch lags and LTP, the excitation/LTP/LPC synthesis of
`decode_core`, mid/side stereo with predictor interpolation, LBRR (FEC) data
skipped, and the decoder's own resampler (`up2_HQ` plus the 12-tap
fractional FIR) to 48 kHz. Packet-loss concealment and comfort noise are not
ported: they only change the output after a lost packet, which RMP never
signals (see §11).

**Output** is 48 kHz interleaved `F32` with every decoded sample; dropping
the pre-skip (`encoder_delay`) is the caller's job, as for AAC and MP3.

## 8. Processing layer

**Audio** (`process/audio.rs`). When only format or layout change, samples
go through the PCM conversion rules and the result equals `swr_convert`
without dithering, bit for bit. When channels or rate change, samples go
through `f64`, the remix matrix and one resampler per output channel.
`remix_matrix` reproduces swresample's default `-ac` coefficients
(`swr_build_matrix`: centre/surround at −3 dB, LFE dropped, etc.),
normalised only for integer output as swresample does. Output pts continue
from the first input frame in `1/output_rate`, counting produced samples.

**Resampler** (`core/resampler`). Kaiser(β = 9) windowed sinc, 32 taps per
phase, cut-off 0.97 of the lower Nyquist, polyphase banks as above. Same
rate is a pass-through. `flush` pads the tail so the output length is
`ceil(n · out / in)`, matching swresample's sample count.

**Video** (`process/video.rs`). Planes are unpacked to floats in
8-bit-equivalent units, so bit depth never changes the arithmetic; scaling
is separable with swscale's kernels (bicubic B = 0, C = 0.6 by default,
widened by the scale factor when shrinking) and centre-sited sampling;
colour conversion defaults to swscale's (BT.601, limited-range YUV,
full-range RGB). New pixel formats: `rgb24`, `bgr24`, `rgba`, `bgra`,
`nv12`.

---

## 9. Core changes and why

| Change | Why |
| --- | --- |
| `Parameters::bits_per_sample` | FLAC and PCM-in-MKV need the coded depth (FFmpeg's `bits_per_raw_sample`). |
| `CodecPrivate::{Flac, FlacEncoder, Opus}` | Typed per-codec state without changing the `Decoder`/`Encoder` traits. |
| `Resolver::has_decoder` | `transcode` refuses to "decode" a codec through the raw pass-through. |
| `AudioFrame` planar `nb_samples` fix | Planar frames reported samples × channels. |
| `PixelFormat::{Rgb24, Bgr24, Rgba, Bgra, Nv12}` | Targets for the scaler. |
| `codecs::pcm::convert::write_sample` | Shared by the PCM encoder and the audio converter. |
| MP4 muxer: relative edit-list media time, derived dts, presentation duration to the last picture | MKV → MP4 with B-frames now decodes with identical frames and pts. |

---

## 10. Verification

- **Matroska.** For four fixtures (H.264 + AAC, H.264 + FLAC + SubRip,
  VP9 + Opus + WebVTT, live WebM with unknown sizes) every packet's stream,
  pts (FFmpeg's plus `encoder_delay`), duration, size, keyframe flag and
  Adler-32 equals `ffprobe -show_packets -show_data_hash adler32`.
  MKV → MP4 and MP4 → MP4 remuxes decode to identical frames and pts.
- **FLAC.** Six fixtures (16/24-bit, 1–6 channels, 8–96 kHz, levels 0/5/8/12,
  wasted bits, mid/side) decode bit-exactly against FFmpeg. Every encoder
  level round-trips bit-exactly, its STREAMINFO MD5 equals FFmpeg's, and at
  level 5 files are within 3% of FFmpeg's size.
- **Ogg.** Opus (mono, stereo with cover, 5.1), FLAC and Vorbis fixtures
  match `ffprobe` packet for packet (payload hashes; pts and durations too
  for Opus and FLAC, including the trimmed last packet). Seeks land within
  the pre-roll before the target and continue with the original packets.
- **Opus**, against libopus 1.4 (`ffmpeg -flags2 +skip_manual -c:a libopus`):
  SILK-only streams (NB/MB/WB, mono/stereo, 10/20/40/60 ms, FEC on) are
  **bit-exact**; hybrid, CELT (2.5–60 ms, CBR/VBR, up to 510 kbit/s, 5.1) and
  streams that switch between all three modes are within float rounding,
  **~138 dB SNR**, at most one 16-bit step apart.
- **Processing.** Bicubic scaling is 59–64 dB PSNR from swscale (max
  difference 1–5 code values); YUV ↔ RGB matches swscale's accurate path
  (> 35/40 dB); resampling 44.1 → 48 kHz and → 8 kHz mono is ~97 dB SNR
  from swresample away from the stream edges; format-only conversions are
  exact.
- **Robustness.** The corruption tests now cover Matroska, FLAC, Ogg and the
  Opus decoder (SILK, hybrid, CELT and multistream packets plus random
  packets). Deep runs (debug builds, overflow checks on) of 3 × 1500–3000
  rounds were clean.
- **Tooling.** `cargo fmt --check` and `cargo clippy --all-targets -- -D
  warnings` pass.

```bash
cargo test --test container mkv            # or flac, ogg
cargo test --test codecs opus
RMP_FUZZ_ROUNDS=3000 cargo test --test container robustness
```

Fixtures and references are regenerated by `tools/make_fixtures.sh`.

---

## 11. Known limitations

| Limitation | Where to start |
| --- | --- |
| No Matroska, Ogg or WebM **muxer**. | `container/mkv/ebml.rs` already has the element model; a muxer writes `EBML`, `Segment` (`Info`, `Tracks`, `Clusters`, `Cues`) and back-patches sizes. |
| Matroska chapters, `ContentEncoding` zlib/encryption and `BlockAdditions` are ignored. | `demuxer.rs::load_level1`, `tracks.rs::content_encoding`. |
| Ogg: chained streams after the first EOS, Theora/Speex mappings. | `ogg/demuxer.rs::new_logical`, page loop in `read`. |
| No Opus/SILK **packet-loss concealment** or comfort noise; a damaged packet is silence. | libopus `silk/PLC.c`, `CNG.c`, CELT `celt_decode_lost` pitch path. |
| No Opus or Vorbis **encoder**, no Vorbis decoder. | The CELT/SILK decoders' tables and range coder are the starting point for an encoder. |
| FLAC encoder: fixed block sizes only, no search beyond FFmpeg's presets, single-threaded. | `encoder.rs::encode_frame`, `analyse`. |
| Resampler: no dithering, no noise shaping, no `soxr`-style quality presets. | `process/audio.rs` (TPDF dither on integer output), `core/resampler`. |
| Scaler: no interlaced, Bayer or palettised formats; chroma siting is centre only. | `process/video.rs::unpack`/`write_plane`. |

---

## 12. How to build on this

**Add a Matroska codec mapping.** Add the codec ID to `tracks.rs::codec_name`
and, if FFmpeg's extradata differs from `CodecPrivate`, convert it in
`tracks.rs::parse_entry`. If its packet durations can be computed from the
payload, add it to `codecs/duration.rs`.

**Add an Ogg mapping.** Recognise the BOS packet in `new_logical`, decide
which packets are headers in `is_header`/`take_header`, and make sure
`packet_samples` knows its durations so granule back-tracking works.

**Add an encoder.** Implement `core::Encoder`, keep state in a
`CodecPrivate` variant, register it in `Resolver::encoder_for`, and wire it
into `examples/transcode.rs`. Use `AudioConverter`/`VideoConverter` to feed
it the format it wants.

**Regenerate Opus tables.** `python3 tools/gen_opus_tables.py
path/to/libopus > src/codecs/opus/tables.rs` (libopus 1.4). Never edit
`tables.rs` by hand.

### FFmpeg / libopus cross-reference

| RMP | Reference |
| --- | --- |
| `container/mkv/*` | `libavformat/matroskadec.c`, `matroska.c` |
| `container/flac/demuxer.rs` | `libavformat/flacdec.c`, `libavcodec/flac_parser.c` |
| `container/flac/muxer.rs` | `libavformat/flacenc.c` |
| `container/ogg/demuxer.rs` | `libavformat/oggdec.c`, `oggparseopus.c`, `oggparseflac.c`, `oggparsevorbis.c` |
| `container/xiph.rs` | `libavformat/oggparsevorbis.c` (`ff_vorbis_comment`), `flac_picture.c` |
| `codecs/flac/decoder.rs` | `libavcodec/flacdec.c`, `flacdsp_template.c` |
| `codecs/flac/encoder.rs`, `lpc.rs` | `libavcodec/flacenc.c`, `lpc.c` |
| `codecs/opus/decoder.rs` | libopus `src/opus_decoder.c`, `opus_multistream_decoder.c`; FFmpeg `libavcodec/opus_parse.c` (channel order) |
| `codecs/opus/celt.rs`, `mdct.rs` | libopus `celt/*.c` (float build) |
| `codecs/opus/silk.rs` | libopus `silk/*.c` (decoder, fixed point) |
| `core/resampler`, `process/audio.rs` | `libswresample/resample.c`, `rematrix.c`, `audioconvert.c` |
| `process/video.rs` | `libswscale/utils.c` (filter init), `swscale.c`, `yuv2rgb.c` |
