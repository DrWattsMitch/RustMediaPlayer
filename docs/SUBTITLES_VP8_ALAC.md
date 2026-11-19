# Text subtitles, VP8 and ALAC decoding in RMP

This batch adds three decoders that fill gaps in formats RMP could already
demux: text subtitle cues (SubRip, WebVTT, ASS/SSA, MP4 timed text) for the
player to draw, VP8 video (older WebM), and ALAC audio (MP4/M4A and
Matroska already muxed it). Read it before changing `src/codecs/subtitle/`,
`src/codecs/vp8/`, `src/codecs/alac/`, `src/container/srt.rs` or
`src/container/ass.rs`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Subtitle frames** | `Frame::Subtitle(SubtitleFrame)`: one cue with pts, duration, the original markup, a plain-text rendering, its `SubtitleFormat` and (ASS) the event fields. `State::Subtitle` for the decoder state. | `core::{SubtitleFrame, SubtitleFormat, AssEvent}` |
| **Subtitle decoders** | `subrip`/`srt`, `webvtt` (plain cue text and MP4 `wvtt` boxes), `ass`/`ssa` (Matroska/FFmpeg event packets and legacy `Dialogue:` lines), `mov_text` (`tx3g`), `text`. | `codecs::subtitle::{SubtitleDecoder, SubtitleCodec}` |
| **ASS styles** | Parses `[Script Info]` (PlayResX/Y, WrapStyle, ScaledBorderAndShadow) and `[V4+ Styles]`/`[V4 Styles]` with the script's own `Format:` order; SSA alignment converted to numpad. | `codecs::subtitle::ass::AssScript` |
| **SRT / ASS files** | Demuxers and muxers for `.srt`, `.ass`, `.ssa`; auto-detected; UTF-8, UTF-16 (BOM) and Latin-1 input. | `container::{srt, ass}` |
| **VP8 decoder** | Every VP8 feature, bit-exact with libvpx and FFmpeg. | `codecs::vp8::{Vp8Decoder, Vp8DecoderState}` |
| **ALAC decoder** | 16/20/24/32-bit, 1-8 channels, verbatim and compressed frames, bit-exact with FFmpeg. | `codecs::alac::{AlacDecoder, AlacConfig}` |
| **Transcoding** | `rmp -i in.webm out.mp4` decodes VP8 for H.264; ALAC converts losslessly to FLAC/WAV (no dither for 24-bit sources, which also fixed 24-bit FLAC → FLAC); `-c:s copy` between `.srt`, `.vtt`, `.ass`, Matroska. | `transcode.rs` |

All decoders are registered in `core::Resolver`, so `decoder_for`,
`has_decoder`, the CLI and the examples pick them up. The `subtitles`
example prints the cues of a file, or only those on screen at a time.

## 2. Subtitles

### Model

A subtitle "frame" is one cue. Rendering stays in the player, so the frame
carries what a renderer needs rather than a bitmap:

- `pts` / `duration` straight from the packet (container time base);
  `end()`, `start_seconds()`, `end_seconds()` and `is_showing(t)` do the
  arithmetic. A cue without a duration lasts until the next one.
- `markup`: the cue text as coded, in `format`'s syntax (SubRip HTML-like
  tags, WebVTT cue text, the ASS `Text` field).
- `text`: the markup removed, character references resolved, `\N` / line
  breaks as `\n`, blank lines trimmed. Good enough for a simple renderer,
  accessibility, search, or logging.
- `ass`: for ASS cues, the event fields (`ReadOrder`, `Layer`, `Style`,
  `Name`, margins, `Effect`). With the stream's extradata parsed by
  `AssScript::parse`, `script.style(&event.style)` gives font, colours,
  alignment and margins (falling back to `Default`, then libass's defaults).

FFmpeg converts every text codec to ASS in its decoders; RMP keeps each
format's own markup instead, because the player may want WebVTT or SubRip
semantics, and converting is lossy (e.g. WebVTT voice and class spans).

### Markup → text (`markup.rs`)

- **SubRip**: drops anything shaped like a tag (`<`, optional `/`, a
  letter, up to `>` on the same line) and `{\...}` override blocks (the
  `{\an8}` position extension); a `<` that doesn't open a tag (`a < b`,
  `<3`) stays. Character references are resolved (`&amp;` and friends,
  numeric). FFmpeg leaves references in SubRip text; real files written by
  web tools often contain them, so RMP resolves them.
- **WebVTT**: every `<...>` is a tag (a literal `<` must be `&lt;`), so
  class, voice, language, ruby and timestamp tags all go.
- **ASS**: `{...}` blocks go, `\N` breaks, `\n` breaks only with
  `WrapStyle: 2` (a space otherwise, as libass does), `\h` is U+00A0, and
  text inside vector drawing mode (`{\p1}` ... `{\p0}`) is dropped.

### Containers

- `srt.rs` reads the whole file (64 MiB cap), finds timing lines
  (`H:MM:SS,mmm --> H:MM:SS,mmm`, lenient about `.` vs `,`, short
  fractions, negative starts, the `X1:.. Y2:..` box), and ends a cue only
  where the next timing line (with its number) begins — so blank lines
  inside a cue survive, as in FFmpeg's `srtdec.c`. Packets are `subrip`
  in milliseconds. The muxer accepts `subrip`, `webvtt` and `text`.
- `ass.rs` follows FFmpeg's `assdec.c`: extradata is the header up to and
  including the `[Events]` `Format:` line; each `Dialogue:` becomes a
  packet `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`
  with pts/duration in centiseconds, sorted by start (stable, so read order
  breaks ties). Event fields are mapped by the script's `Format:` line, so
  reordered or SSA (`Marked`) formats work. The muxer writes the header
  back and events in read order (FFmpeg's `assenc.c`).
- Detection: `[Script Info]` as the first non-comment line for ASS; a cue
  number then a timing line (or a timing line) for SRT. Both run after
  WebVTT and before the audio sync-word probes.
- MP4 `wvtt` samples are ISO 14496-30 boxes; the WebVTT decoder unwraps
  `vttc`/`payl` and treats `vtte` as an empty gap, so MP4 WebVTT decodes
  like `.vtt` cues. `mov_text` samples are a 16-bit length and UTF-8 (or
  UTF-16 with BOM); empty samples (the gaps FFmpeg writes) give no cue.

### Not done

- WebVTT cue settings (position, line, align) are dropped by the demuxer,
  as before; Matroska stores them in BlockAdditions, which RMP doesn't
  read. Adding a settings field to `SubtitleFrame` is the natural place if
  a renderer needs them.
- Bitmap subtitles (PGS, VobSub, DVB) are out of scope.
- `-c:s` other than `copy` still isn't supported in transcode (no text
  subtitle *encoder* converting between markups).

## 3. VP8 (`src/codecs/vp8/`)

A port of libvpx's decoder (`vp8/decoder/decodeframe.c`, `decodemv.c`,
`detokenize.c`, `onyxd_if.c`, `vp8/common/*`), checked frame by frame
against both FFmpeg's native decoder and libvpx.

- `range.rs`: the boolean decoder with libvpx's 64-bit window, so bits
  read past a partition's end are the same zeros libvpx reads; `sign()` is
  `GetSigned` (an even split normalised by exactly one bit).
- `tables.rs`: generated from libvpx's headers (coefficient defaults and
  update probabilities, key-frame sub-block mode probabilities, MV
  defaults/updates, mode contexts, quantizer lookups).
- `dsp.rs`: IDCT/WHT with libvpx's `short` truncation, the 16x16/8x8 and
  4x4 intra predictors (VP8's smoothed `B_VE`/`B_HE` and the `d45e`/`d63e`
  variants), six-tap and bilinear interpolation (always the two-pass C
  path; identity passes are exact), and the normal/simple loop filters.
- `decoder.rs`: frame header, segmentation and loop-filter deltas,
  probability updates and their save/restore (`refresh_entropy_probs`),
  per-macroblock modes and the near-MV search (`read_mb_modes_mv`, kept in
  the decoder's own order, which differs slightly from
  `vp8_find_near_mvs`), split MVs, tokens with above/left contexts,
  reconstruction, the frame loop filter, and reference updates.

Points that matter for bit-exactness:

1. **Intra prediction uses unfiltered pixels.** libvpx filters row *r-1*
   only after decoding row *r*, and filtering never touches the row
   below, so decoding the whole frame and then filtering it in macroblock
   order is equivalent. Edges follow libvpx's frame setup: 127 above the
   frame (corner and above-right included), 129 to the left, the corner
   129 below the first row; the above-right of the last column repeats the
   row's last pixel (`vp8_extend_mb_row`); B_PRED's right column takes the
   macroblock's above-right for every sub-block row
   (`intra_prediction_down_copy`).
2. **References are read with unbounded edge replication.** libvpx keeps a
   32-pixel border and clamps motion vectors that reach further
   (`clamp_mv_to_umv_border`); a clamped vector only moves a block that is
   entirely outside the picture, where every pixel equals the edge, so
   replication without bounds gives the same pixels. The clamps are still
   applied because they change the chroma vectors derived from luma.
3. **Planes are whole macroblocks.** Prediction, filtering and edge
   replication use the 16-aligned size, as libvpx and FFmpeg do; the
   output frame is a cropped view (no copy, like VP9).
4. **Golden/alt-ref copies** happen before refreshes, and a golden copy
   from the alt-ref sees an alt-ref just copied this frame (libvpx's
   `swap_frame_buffers` order).
5. Versions 1-3 use bilinear prediction and version 3 full-pixel chroma;
   the loop filter type comes from the header bit, not the version.

Hidden alt-ref frames return no picture. Packets shorter than 3 bytes,
streams that don't start with a key frame, and truncated partitions are
errors; corrupt token data decodes (as zeros past the end) without
panicking.

Not done: multithreading (VP8 decodes at several hundred 640x360 frames
per second single-threaded; partitions or rows could be threaded like
VP9's tiles), SIMD, error concealment, and the key frame's upscaling bits
(ignored, as in FFmpeg).

## 4. ALAC (`src/codecs/alac/`)

A port of FFmpeg's `alac.c`/`alacdsp.c`: adaptive Golomb-Rice residuals
(`rice_decompress` with the zero-run escape), the sign-adaptive FIR
predictor (prediction type 15's extra first-order pass included), stereo
decorrelation, and the verbatim low "extra bits" of 20/24/32-bit audio
(appended after decorrelation, FFmpeg's default). All arithmetic wraps at
32 bits like the C code. Output is interleaved `S16` for 16-bit streams and
`S32` (samples in the high bits) otherwise, in FFmpeg's channel order
(ALAC codes C, L, R, ...; `CHANNEL_OFFSETS` maps elements to L, R, C, LFE,
...).

Extradata: FFmpeg's layout, the 36-byte `alac` atom, is now RMP's canonical
form. The MP4 demuxer wraps the box body into it and the Matroska demuxer
the bare `ALACSpecificConfig`; both set `sample_rate`, `channels` and
`bits_per_sample` from the config. The MP4 muxer writes the box body and
the Matroska muxer the bare config from any of the three wrappings, so
MP4 ↔ Matroska remuxes round-trip. `AlacConfig::from_extradata` accepts all
three.

`bits_per_sample` also lets the transcoder see that a 24-bit stream decoded
to `S32` has 24 significant bits: `AudioConverter::dithered_from` skips
dither when the target keeps at least the source's bits. That makes ALAC →
FLAC lossless, and fixed the same issue for 24-bit FLAC → FLAC.

## 5. Tests

- `tests/codecs/subtitle_decoder_test.rs`: SRT, WebVTT, ASS files and the
  same cues muxed by FFmpeg into Matroska (SRT + ASS) and MP4 (`tx3g`);
  timings are FFmpeg's. Unit tests cover markup stripping, ASS styles
  (ASS and SSA), event packets, SRT edge cases and muxer round trips.
- `tests/codecs/vp8_decoder_test.rs`: framecrc (adler32 per frame) of
  libvpx encodes: basic, 2-pass hidden alt-refs with sign bias, an odd
  99x67 size, 4 token partitions with B_PRED/split MVs, profiles 1 and 3,
  q = 0, segmentation with error resilience and frame-local probabilities,
  temporal layers, a fast pan with MVs past the edge; plus damage, a
  missing key frame, and size changes at key frames. FFmpeg's native
  decoder and libvpx agree on every fixture. 8-partition 640x360 streams
  were checked too (not committed, for size).
- `tests/codecs/alac_decoder_test.rs`: MD5 of FFmpeg's PCM for 16-bit
  mono/stereo, 24-bit stereo and 5.1 (channel order, extra bits), and
  verbatim frames, from both MP4 and Matroska; damaged packets.
- `tests/cli.rs` `vp8_alac_and_text_subtitles_convert`: VP8 → H.264 MP4,
  lossless ALAC → FLAC (16 and 24-bit), ALAC MP4 → Matroska → MP4, and
  SRT → VTT → SRT / ASS → ASS file conversions.

Regenerating fixtures: the VP8 streams are `ffmpeg -f lavfi -i testsrc2/
mandelbrot ... -c:v libvpx` with the options named in the test docs (the
odd-size stream is a 98x66 encode with its key frame and IVF header
patched to 99x67, which decodes identically and crops differently); the
references are `ffmpeg -i X -f framecrc`. ALAC fixtures are `-c:a alac`
with `aformat=sample_fmts=s16p|s32p`; references `-c:a pcm_s16le|pcm_s32le
-f md5`.
