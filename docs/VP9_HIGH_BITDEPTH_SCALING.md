# VP9 profiles 2 and 3, and resolution changes

This batch closes the two VP9 gaps listed in `VORBIS_VP9.md`: 10- and
12-bit streams (profiles 2 and 3) used to stop at the header, and a frame
whose size differed from a reference's failed with "scaled reference
frames are not supported". Both now decode bit-exactly with libvpx and
FFmpeg. Read this before changing `src/codecs/vp9/`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **High bit depth** | Profiles 2 (10/12-bit 4:2:0) and 3 (10/12-bit 4:2:2, 4:4:0, 4:4:4), with every tool the 8-bit decoder has: tiles and tile threading, lossless, segmentation, compound prediction, hidden frames. Output is `yuv420p10le`, `yuv422p12le`, ... (`Pixel { depth: 10 or 12 }`, two little-endian bytes per sample). | `Vp9DecoderState`, `DecodedPicture::Wide` |
| **Scaled references** | Inter frames whose size differs from their references (resolution switching without key frames, as libvpx's dynamic resize, WebRTC and SVC produce): libvpx's fixed-point scale factors, motion-vector clamping, per-pixel stepping 8-tap filter. Any mix of scaled and unscaled references, compound included, at every bit depth and chroma format. | `decoder::RefScale`, `inter::predict_scaled` |
| **API** | `decode_packet` returns `DecodedPicture` (`Narrow(Arc<Picture<u8>>)` or `Wide(Arc<Picture<u16>>)`), with `to_frame`, `to_bytes`, `pixel`, `width`, `height`, `bit_depth`. 8-bit frames are still zero-copy views of the reference picture. | `codecs::vp9::DecodedPicture` |

The `rmp` CLI, `Resolver::decoder_for` and transcoding pick this up with no
change: a 10-bit WebM now decodes to `yuv420p10le` (Y4M `C420p10`), and a
stream that changes size produces frames of each size in turn.

## 2. High bit depth

### One decoder, two sample types

Everything that touches samples is generic over `pixel::Px`, implemented
by `u8` and `u16`, the same split as the H.264 decoder: `Plane<P>`,
`Picture<P>`, `FrameDecoder<P>`, the tile workspaces, intra and inter
prediction, the inverse-transform add and the loop filter. Each frame picks
its type from the header's bit depth (`decode_frame` → `decode_samples::<P>`),
so both are monomorphised and the 8-bit path costs the same as before (the
8-bit 720p benchmark is unchanged, output identical).

`Px` also carries the kernels that have vector code: for `u8`,
`Px::predict` and `Px::lf_vertical`/`lf_horizontal` go to `simd.rs`; for
`u16` they are the generic scalar versions. Reference slots hold the
bit-depth-erased `DecodedPicture`, and a frame whose references are of
another depth fails ("reference frame has another bit depth"), like
libvpx's "incompatible color format".

### What changes with the bit depth

All of it is libvpx's `highbd` code (`CONFIG_VP9_HIGHBITDEPTH`), which
FFmpeg's templated decoder matches:

| Step | 8-bit | 10/12-bit |
| --- | --- | --- |
| Dequantisation | `DC_QLOOKUP[0]`, `AC_QLOOKUP[0]` | rows `[1]` (10-bit) and `[2]` (12-bit) |
| Cat6 token | 14 extra bits | 16 / 18 (`vp9_cat6_prob_high12`, leading 255s) |
| Inverse transforms | same butterflies, 64-bit with 32-bit `WRAPLOW` | same; a 1-D input of 2^25 or more gives zeros (`detect_invalid_highbd_input`; only damaged streams) |
| Clipping (reconstruction, MC, TM) | 0..255 | 0..(1 << bd) - 1 |
| Intra edge defaults | 127 / 129 / 128 | `base - 1` / `base + 1` / `base`, `base = 128 << (bd - 8)` |
| Loop filter | `limit`, `blimit`, `thresh`, flat threshold 1, signed range -128..127 | each threshold `<< (bd - 8)`, signed range `-128 << s ..= (128 << s) - 1` (`signed_char_clamp_high`) |

The 4x4 directional intra predictors keep their 8-bit special cases at
high bit depth (libvpx has `vpx_highbd_d45_predictor_4x4` and friends, and
FFmpeg shares one template).

### Output

10/12-bit pictures become frames by writing their samples out as
little-endian bytes (RMP's layout for deep samples), one pass over the
picture; 8-bit pictures stay zero-copy. 10/12-bit sRGB streams decode, and
`to_bytes` gives FFmpeg's planar `gbrp10le`/`gbrp12le`, but RMP has no
deep RGB pixel format, so the `Decoder` interface reports them as
unsupported (as it does 4:4:0).

## 3. Scaled references

A reference of another size is predicted through libvpx's scale factors
(`vp9_scale.c`, `vp9_decodeframe.c: dec_build_inter_predictors`):

1. **Scale factors** per reference (`RefScale::new`):
   `x_fp = (ref_w << 14) / w`, `y_fp` likewise, step `x_step = 16 * x_fp >> 14`.
   A reference is usable when the frame is at most 2x smaller and 16x
   larger; at least one of the three must be, and a block using an
   unusable one fails ("reference frame has invalid dimensions").
2. **Clamp the vector** (`clamp_mv_to_umv_border_sb`) to at most 4 pixels
   plus the block size past the picture, in 1/16 of the plane.
3. **Position** (`scaled_position`): the block's integer position is
   scaled (`x0 = sx(x_start + x)`), the vector is scaled in 1/16, and the
   phase of the scaled block position is added to it. For chroma that
   phase comes from the *luma* position plus the chroma offset
   (`vp9_scale_mv(&mv_q4, mi_x + x, mi_y + y, sf)`); this is a libvpx quirk
   that FFmpeg reproduces too. The 4:2:0 fixtures fail if it is "fixed".
4. **Filter** (`inter::predict_scaled`, libvpx `vpx_scaled_2d` /
   `vpx_convolve8_c`): a horizontal pass over
   `((h - 1) * y_step + phase_y >> 4) + 8` rows, the source column advancing
   by `x_step` sixteenths per output pixel and each pixel using the kernel of
   its own phase, rounded and clipped into a 64 x 135 intermediate; then the
   vertical pass the same way. A phase-0 kernel is the unit tap, so libvpx's
   one-direction variants (`vpx_scaled_horiz`, `vpx_scaled_vert`) give the
   same result and one function covers them.
5. **Edges**: reads outside the reference repeat its edge. libvpx either
   reads the frame directly (when its conservative bounding box is inside)
   or builds a border-extended copy (`extend_and_predict`); both give the
   edge-clamped samples, and the bounding box always covers the reads.

Unscaled references keep the existing path (and its vector code).
Everything else was already size-aware: segmentation maps reset on a size
change, and the previous frame's motion vectors are only used at the same
size.

## 4. Performance

On a 2-core cloud VM, one thread, release build, 1280x720 with 50 frames:

| Stream | RMP | FFmpeg (one thread) |
| --- | --- | --- |
| 8-bit (4 Mb/s) | ~225 frames/s, unchanged by this batch | ~255 frames/s |
| 10-bit (4 Mb/s) | ~90 frames/s | ~225 frames/s |

The 10-bit path is scalar: the VP9 vector kernels (`simd.rs`) are 8-bit
only. Scaled prediction is scalar at every depth (one kernel per pixel).

## 5. Limitations

| Not supported | Where to start |
| --- | --- |
| Vector code for 10/12-bit (loop filter, 8-tap convolution). | The `simd.rs` kernels already compute in 16-bit lanes; the high-bit-depth ones need 32-bit sums for 12-bit and `min`/`max` instead of byte packing for the clip. Hook them into `impl Px for u16`. |
| Vector code for scaled prediction. | libvpx `vpx_scaled_2d_ssse3` special-cases the 2:1 and 4:3 steps. |
| 10/12-bit sRGB and 4:4:0 output through the `Decoder` interface (both decode). | A deep RGB / 4:4:0 `PixelFormat`. |
| Frame threading, error concealment, AV1 (see `VORBIS_VP9.md`). | — |

## 6. Tests

`tests/codecs/vp9_decoder_test.rs` compares every frame with FFmpeg
`framecrc` dumps (`-autoscale 0`, so frames keep their sizes). FFmpeg 4.4
and 6.1 produce the same references:

- `p2_420_10bit_176x144` (alt-refs, superframes), `p2_420_12bit_99x61` (odd
  size), `p3_444_10bit_96x64`, `p3_422_12bit_96x64` (noisy, large
  coefficients), `lossless_10bit_64x48`, `tiles_10bit_512x128`;
- `resize_420_176x144`: one key frame, then 176x144 → 132x108 → 88x72 →
  176x144 (the golden reference is unscaled again) → 131x77;
- `resize_444_128x96` (no chroma subsampling) and `resize_10bit_160x96`
  (scaling at 10 bits).

The threading test runs the new streams on 2, 3 and 8 threads with and
without the row pipeline; the robustness test damages 10/12-bit and resize
streams. Unit tests: cat6 widths, high-bit-depth clipping and the invalid
input rule of the transforms. A 720p 10-bit libvpx encode also matched
FFmpeg exactly during development.

### Making the fixtures

FFmpeg's libvpx wrapper cannot change the size of a running encode, so
`tools/vp9_resize_enc.py` drives libvpx's encoder API through `ctypes`: a
new size through `vpx_codec_enc_config_set` makes libvpx keep coding inter
frames from the old references (one pass, no lag). Any `libvpx.so` works
(PyAV's wheel bundles an 8-bit one; distribution builds usually have high
bit depth). The commands are in `tools/make_fixtures.sh`, section "VP9
profiles 2/3 and scaled references".

## 7. File map

| File | Change |
| --- | --- |
| `vp9/pixel.rs` | New: the `Px` trait (`u8`, `u16`), kernel dispatch, slot and pool access. |
| `vp9/decoder.rs` | `Plane<P>`, `Picture<P>`, `DecodedPicture`, `FrameDecoder<P>`, `RefScale`, `scaled_position`, per-depth dispatch, high-bit-depth output. |
| `vp9/inter.rs` | Generic 8-tap prediction, `predict_scaled`. |
| `vp9/intra.rs`, `vp9/transform.rs`, `vp9/loopfilter.rs` | Generic over `Px`, with the bit depth's edges, clip and thresholds. |
| `vp9/residual.rs` | `cat6(bit_depth)`. |
| `vp9/tile_threads.rs`, `vp9/mode.rs` | Generic over `Px`. |
| `vp9/header.rs` | No longer rejects profiles 2 and 3. |
| `tools/vp9_resize_enc.py` | Resize fixture encoder. |
