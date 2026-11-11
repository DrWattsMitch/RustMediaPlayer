# Vorbis and VP9 decoding in RMP

This batch closes the two gaps left by the Matroska/Ogg work: Ogg and WebM
Vorbis audio could be demuxed and remuxed but not decoded, and WebM video
(VP9) could be demuxed but not decoded. It adds a Vorbis I decoder, exact
Vorbis packet timestamps in the Ogg demuxer, and a VP9 decoder that is
bit-exact with libvpx and FFmpeg. Read it before changing
`src/codecs/vorbis/`, `src/codecs/vp9/` or the Vorbis paths of
`src/container/ogg/demuxer.rs`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Vorbis decoder** | Every Vorbis I stream: codebooks (all lookup types), floor 0 and floor 1, residue types 0/1/2, square-polar coupling, submaps, short/long blocks with window switching. Planar `F32` in FFmpeg's channel order, within one 16-bit step of FFmpeg's float decoder. | `codecs::vorbis::{VorbisDecoder, VorbisDecoderState}` |
| **Ogg Vorbis timestamps** | Per-packet durations from the block sizes, pts walked back from the granule positions, end trimming from the last granule, correct timing after a seek. Matches `ffprobe -show_packets` packet for packet. | `container::ogg::demuxer` + `codecs::vorbis::VorbisParser` |
| **VP9 decoder** | Profiles 0 and 1 (8-bit 4:2:0, 4:2:2, 4:4:4, sRGB): key, inter, intra-only and hidden frames, superframes, `show_existing_frame`, tiles, segmentation (incl. temporal prediction), loop-filter deltas, compound prediction, switchable filters, lossless (WHT), error-resilient and frame-parallel modes, backward adaptation. Bit-exact with libvpx/FFmpeg. | `codecs::vp9::{Vp9Decoder, Vp9DecoderState}` |
| **Transcoding** | `rmp -i in.webm out.y4m` decodes VP9; `.mp4`/`.mov` output re-encodes VP9 to H.264 by default (MP4 can't hold VP9 here); Vorbis decodes to any audio output. | `transcode.rs` default codec table |

Both decoders are registered in `core::Resolver`, so `decoder_for`,
`has_decoder`, the `rmp` CLI and the examples pick them up.

## 2. Vorbis

### Decoding

`setup.rs` parses the identification and setup headers into an immutable
`Setup` (shared through an `Arc`, so cloning the decoder state per packet,
as the `Decoder` trait does, is cheap). Each audio packet then follows the
specification's §4.3:

1. **Mode and windows.** The mode picks the block size; long blocks carry
   the previous/next window flags that shape the window slopes.
2. **Floors.** Floor 1 decodes the Y list, unwraps each value against the
   line through its neighbours and renders Bresenham segments through
   `floor1_inverse_dB_table` (`floor.rs`, following libvorbis' `floor1.c`,
   including its clamping of the curve to `[0, 255]`). Floor 0 (pre-1.0
   streams) evaluates the LSP curve on the Bark map in `f64`.
3. **Residues.** Classification words, up to eight cascaded passes, the
   three vector layouts (`residue.rs`). An early end of packet leaves the
   rest at zero, as the specification requires.
4. **Coupling, floor × residue, IMDCT.** The existing `dsp::mdct::Imdct`
   (scale 1.0) gives the full-length output; windows are applied with the
   left/right slopes, and the right half is kept for the next packet.
5. **Overlap-add.** The output spans the centres of the previous and the
   current block: `previous / 4 + current / 4` samples.

Codebooks build their codewords with libvorbis' `_make_words` (lowest free
node first, over- and under-full trees rejected except the single-entry
book) and decode with a 10-bit lookup table that falls back to a tree walk
for longer codewords (`codebook.rs`).

### Priming, trimming and timestamps

The first packet after the headers (or after a reset) only fills the
overlap buffer: like FFmpeg (which decodes it and then skips it through
`skip_samples`), RMP returns no samples for it. This is independent of the
container, so Matroska and Ogg streams decode to the same length as FFmpeg.

In Ogg, FFmpeg gives that packet a negative pts (`-128` for a short first
block). RMP's timestamps are unsigned, so the priming packet keeps
`pts: None` with its duration; every later packet has FFmpeg's exact pts and
duration. The Ogg muxer already ends a pts-less first Vorbis packet at
granule 0, so Ogg → Ogg remuxes keep their granules. When the last
granule cuts the stream short, the last packet's duration is shortened and
the decoder trims its output to it (a packet duration in sample units,
shorter than the block, means end trimming).

`VorbisParser` (formerly `codecs/vorbis.rs`, now `vorbis/parser.rs`) gained
`reset()`: after a seek the first packet's duration is unknown (it depends
on the previous block), its pts comes from the previous page's granule,
and the following packets are exact again.

## 3. VP9

### Data flow

```
packet ──split_superframe──► frame ──parse_uncompressed──► FrameHeader
                                    └─parse_compressed (BoolDecoder)──► FrameContext + tx/reference modes
tiles (4-byte sizes) ──► per SB row, per tile column ──► decode_partition (recursive)
    decode_block: read_mode_info ─► intra: predict + tokens + inverse transform per transform block
                                 └► inter: motion compensation (1–2 refs) ─► tokens + inverse transform
after all tiles: loop filter (per superblock: Y, U, V; vertical then horizontal edges)
               ─► backward adaptation (coefficients, modes, MVs) ─► save context ─► refresh slots
```

`Vp9DecoderState` keeps what crosses frames: the persistent header state
(loop-filter deltas, segmentation features, four probability contexts), the
eight reference slots (`Arc<Picture>`), the previous frame's motion vectors
and the two segmentation maps. Everything large is behind an `Arc`, so the
per-packet state clone is cheap. `FrameDecoder` owns one frame's working
data (mode-info grid, entropy contexts, the picture being built).

### How bit-exactness was reached

VP9 decoding is fully specified in integers, so the goal is equality, not
closeness. The code follows libvpx's reference decoder rather than
FFmpeg's optimised one:

- **Tables** (`tables.rs`, ~75 KB) are generated by
  `tools/gen_vp9_tables.py` from libvpx sources (BSD-3, compatible with
  Apache-2.0), with C initializer semantics (zero padding) reproduced.
- **Inverse transforms**: the 1-D DCT/ADST butterflies were translated
  mechanically from `vpx_dsp/inv_txfm.c` and reviewed; they run in 64-bit
  with libvpx's 32-bit `WRAPLOW`. libvpx's partial transforms (`_1`, `_10`,
  `_34`, …) are exact shortcuts of the full ones, so RMP only skips zero
  rows.
- **libvpx behaviours that affect the output**, reproduced deliberately:
  - the MV high-precision bit is counted for adaptation even when it is
    not coded (it then reads as 1);
  - an inter block of 8x8 or larger with no coefficients becomes `skip`
    after reconstruction, which changes later skip/tx contexts and the
    loop filter;
  - sub-8x8 chroma motion vectors for 4:2:2 average blocks `(i, i + 1)`;
  - intra edges stop at the 8-aligned picture width, and above-right
    pixels are only real for 4x4 transforms inside the block; the 4x4
    directional predictors are libvpx's special cases;
  - the previous frame's motion vectors are used only if it was shown, the
    same size, not intra-only and not a key frame (`use_prev_frame_mvs`);
  - intra-only frames decode with probability context 0 after the reset
    rules of `setup_past_independence`;
  - the segmentation map is double-buffered and swapped only when
    segmentation is on;
  - the loop filter runs superblock by superblock (libvpx's
    `vp9_filter_block_plane_non420` mask rules for every plane), vertical
    edges before horizontal ones.

### Output

One frame per shown frame, with the packet's pts (a superframe usually
holds a hidden alt-ref plus one shown frame). Planar `yuv420p`, `yuv422p`
or `yuv444p`; sRGB streams (stored as G, B, R planes, FFmpeg's `gbrp`) are
emitted as packed `rgb24`, since RMP has no planar RGB layout. A stream must
start with a key frame (an inter frame first is an error, not garbage).

### Performance

Pure safe Rust, single-threaded, no SIMD. On a 2-core cloud VM, release
build: 1280x720 at ~125 frames/s (FFmpeg's assembly decoder: ~290 with one
thread), small sizes on par with FFmpeg. The loop filter works on fixed-size
pixel windows (it was 80% of the time before that) and motion compensation
skips filter passes whose phase is 0 (a unit tap, so the result is exact).
Vorbis decodes at ~400x real time.

## 4. Limitations

| Not supported | Where to start |
| --- | --- |
| VP9 profiles 2 and 3 (10/12-bit). Headers parse; decoding stops with an error. | Make `Plane` hold `u16`, use the `[1]`/`[2]` rows of `DC_QLOOKUP`/`AC_QLOOKUP`, the 18-bit cat6 token, and `clip_pixel_highbd`. |
| Inter prediction from a reference of another size (resolution switching inside a stream). Fails with "scaled reference frames are not supported". | libvpx `dec_build_inter_predictors` (`is_scaled` path) and `vpx_scaled_2d`; FFmpeg's `mc_luma_scaled` documents the libvpx rounding it must match. |
| 4:4:0 output (decodes, but RMP has no 4:4:0 pixel format). | `PixelFormat` + the scaler. |
| Frame or tile threading. | Tiles of a row are independent apart from the counts; the loop filter can trail decoding by a superblock row as libvpx does. |
| Error concealment: a corrupt frame is an error and does not refresh any reference. | libvpx keeps decoding with `corrupted` flags. |
| AV1. | The VP9 structure (bool decoder → multi-symbol CDF coder, same partition/mode/residual/filter split) is the template; dav1d (BSD-2) is the reference to follow. |
| Vorbis encoding. | — |

## 5. Tests

- `tests/codecs/vorbis_decoder_test.rs`: low-quality mono (floor 1 with
  long codewords), 5.1 (channel order), high-quality stereo in WebM, and
  the Ogg fixture's exact length; references are FFmpeg's decodes.
- `tests/container/ogg/demuxer_test.rs`: Vorbis pts/durations against
  `ffprobe`, end trimming and timestamps after a seek.
- `tests/codecs/vp9_decoder_test.rs`: seven libvpx streams compared frame
  by frame with FFmpeg `framecrc` dumps (alt-refs/superframes, odd-size sRGB
  4:4:4, 4:2:2, lossless, 2 tile columns × 4 tile rows, segmentation + error resilience,
  frame-parallel), plus the start-without-key-frame error.
- `tests/container/robustness_test.rs`: damaged Vorbis setup headers and
  packets, VP9 streams with bit flips, truncation and garbage in some
  frames, random VP9 packets and corrupted WebM files; never a panic.
- Unit tests next to the code: codeword assignment, window
  power-complementarity, the boolean decoder against an encoder, the
  probability merge rules, transform DC paths, loop-filter taps, motion
  compensation shortcuts.

Fixture commands are in `tools/make_fixtures.sh` (sections "Vorbis
decoder references" and "VP9 decoder references").

During development the decoder was also checked bit-exact against FFmpeg on
larger libvpx encodes not kept as fixtures: CIF, 720p with four tile
columns, a 300-frame GOP with alt-refs, `cpu-used 0`, realtime with
`aq-mode 3`, and a noisy high-rate clip.

Not yet covered by a test stream (implemented from libvpx, but FFmpeg's
libvpx/libvorbis encoders don't produce them): VP9 intra-only frames and
`show_existing_frame`, and Vorbis floor 0. A conformance stream for each
(the libvpx `vp90-2-*` test vectors, an old Vorbis 1.0 beta file) would
close that gap.

## 6. File map

| File | Contents |
| --- | --- |
| `src/codecs/vorbis/mod.rs` | Module, `CHANNEL_ORDER` (shared with Opus), `ilog`. |
| `vorbis/bits.rs` | LSB-first bit reader with end-of-packet as `None`. |
| `vorbis/codebook.rs` | Codeword assignment, VQ lookup, fast decode table. |
| `vorbis/floor.rs` | Floor 0 and floor 1 (decode and curve). |
| `vorbis/residue.rs` | Residue types 0/1/2. |
| `vorbis/setup.rs` | Identification and setup headers, mappings, modes. |
| `vorbis/decoder.rs` | Packet decode, windows, overlap, `Decoder` impl. |
| `vorbis/parser.rs` | Packet durations without decoding (containers). |
| `src/codecs/vp9/mod.rs` | Block-size tables, trees, prediction modes, `Mv`. |
| `vp9/range.rs` | Boolean decoder. |
| `vp9/header.rs` | Uncompressed/compressed headers, superframe index. |
| `vp9/probs.rs` | Frame contexts, counts, differential updates, adaptation. |
| `vp9/mode.rs` | Mode info: segments, skip, tx size, modes, references, MV search. |
| `vp9/residual.rs` | Coefficient tokens and scans. |
| `vp9/transform.rs` | Inverse DCT/ADST/WHT. |
| `vp9/intra.rs`, `vp9/inter.rs` | Prediction. |
| `vp9/loopfilter.rs` | Deblocking. |
| `vp9/decoder.rs` | Frames, tiles, partitions, references, `Decoder` impl. |
| `vp9/tables.rs` | Generated by `tools/gen_vp9_tables.py`. |
