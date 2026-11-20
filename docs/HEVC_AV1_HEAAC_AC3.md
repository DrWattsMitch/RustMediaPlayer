# HEVC, AV1, HE-AAC and AC-3 / E-AC-3

This batch adds four decoders that between them cover most streaming and
broadcast content: AC-3 / E-AC-3, HE-AAC v1/v2 (SBR + PS), AV1 and HEVC.
All are reached through `Resolver` like every other codec. All output is
checked against FFmpeg, and the video output is bit-exact.

---

## 1. What was added

| Codec | Where | How |
|---|---|---|
| AC-3, E-AC-3 | `src/codecs/ac3/`, raw `.ac3`/`.eac3` demuxer in `src/container/ac3/` | Native port of FFmpeg `ac3dec` (float path) |
| HE-AAC v1 / v2 | `src/codecs/aac/sbr.rs`, `ps.rs`, `sbr_tables.rs`; ADTS demuxer in `src/container/adts/` | Native port of FFmpeg `aacsbr` / `aacps`, plugged into the existing AAC-LC decoder |
| AV1 | `src/codecs/av1/` | `rav1d` 1.1 (the Rust port of dav1d, BSD-2) behind a small safe wrapper |
| HEVC / H.265 | `src/codecs/hevc/` | Native, written after FFmpeg `hevcdec` |

### AC-3 / E-AC-3

- Integer-exact parametric bit allocation and FFmpeg's MD5-seeded dither
  generator, so zero-bit mantissas are dithered exactly as FFmpeg does.
- Coupling, rematrixing, dynamic range, downmix flags, block switching,
  E-AC-3 AHT, SPX and dependent substreams (7.1 layouts).
- Enhanced coupling is not supported, as in FFmpeg.
- Output is F32 planar.

### HE-AAC

- SBR: QMF analysis and synthesis (on the shared `dsp::Imdct`), HF
  generation and adjustment, and noise and sinusoid insertion.
- PS: hybrid analysis, decorrelation and the 20/34-band stereo mixing.
- Explicit and implicit signalling follow FFmpeg's -1/0/1 rules, including
  the "locked" behaviour. Mono HE-AAC with implicit PS decodes as stereo,
  like FFmpeg.
- The tables are generated from FFmpeg's sources by
  `tools/gen_sbr_tables.py`.

### AV1

- `codecs/av1/dav1d.rs` holds the only `unsafe` code, under its own
  `#![allow(unsafe_code)]`; the crate root keeps `#![deny(unsafe_code)]`.
- Each `Picture` is copied out of rav1d into an owned frame.
- The `av1` cargo feature is on by default.
- The thread count follows `RMP_THREADS` / `threading::default_threads()`.
- Note that rav1d itself contains `unsafe` code and assembly.

### HEVC

The decoder supports Main, Main 10 and Main 12 at 4:2:0, including:

- I/P/B slices with B-pyramids and open GOP (CRA with RASL skipping at a
  sequence start)
- tiles, WPP (wavefront parallel processing), and dependent slice segments
- AMP, SAO, scaling lists (SPS and PPS), transform skip, and transquant
  bypass / lossless
- PCM, sign hiding, and explicit weighted prediction
- long-term references and generated missing references
- `hvcC` and Annex B input, and cropping

Not supported: 4:0:0, 4:2:2 and 4:4:4, and the range, SCC and multi-layer
extensions. These are rejected with an error.

| File | Contents |
|---|---|
| `cabac.rs` | FFmpeg-style CABAC engine; recovers the byte position after `terminate` (PCM, tile/WPP substreams) |
| `ps.rs`, `slice.rs` | parameter sets, tile layout and z-scan tables; slice headers and RPS |
| `ctu.rs` | coding tree, residual coding, QP derivation, intra/inter reconstruction, deblocking bS |
| `mvs.rs` | merge, AMVP, temporal MV prediction |
| `intra.rs`, `mc.rs`, `transform.rs` | prediction and inverse transforms; scalar reference kernels |
| `filter.rs` | deblocking and SAO, run in CTB-row bands as rows complete (same result as whole-picture filtering) |
| `simd.rs` | vector kernels, see below |
| `decoder.rs` | NAL dispatch, POC, RPS marking, reference lists, DPB bumping, output; picture jobs (frame threading) |
| `picture.rs` | planes, motion field, and `RefPic`: a DPB picture other threads read while it decodes |
| `wpp.rs` | WPP rows decoded from the entry points, on several threads |

Two design points:

- Samples are stored as `u16` at every bit depth.
- Planes are padded (80 luma / 40 chroma) with replicated edges. Motion
  compensation therefore only clamps block positions and never checks
  bounds per sample.

Behaviour on broken streams:

- A slice segment that starts on an already-decoded CTB is dropped, as
  FFmpeg does.
- Missing references are generated as grey, intra-only pictures.
- Damaged input never panics. A test flips random bits in every fixture,
  and about 9,000 corrupted decodes were run while developing.

**SIMD.** All kernels are written against `crate::simd` and are
bit-exact with their scalar originals, which differential tests check:

- luma 8-tap and chroma 4-tap interpolation (h, v, hv), using `madd` on
  interleaved sample pairs
- uni- and bi-prediction stores
- the inverse DCT: 8x8 to 32x32 in 16-bit lanes (even/odd halves with
  `madd` on interleaved coefficient rows, `transpose8x8` between the
  passes, exact 32-bit sums); 4x4 in 32-bit lanes
- the residual add (16-bit saturating add, then clip)
- planar and angular intra prediction (`madd` on reference pairs;
  horizontal modes are transposed in registers)
- SAO edge and band offsets
- luma deblocking, 8 lines per call through `transpose8x8` (bit depth ≤ 10)

Explicit weighted prediction whose weight is `1 << denom` with offset 0
(the tables most P slices carry) takes the default, vectorised path. The
rounding terms cancel, so the result is the same.

`I16x8::load_u16` / `store_u16` were added to all three SIMD backends,
with tests.

## 2. Verification

**AC-3 / E-AC-3.** Output is within ±1 LSB of FFmpeg's float decoder
(`tests/codecs/ac3_decoder_test.rs`).

**HE-AAC.** Measured against FFmpeg:

| Stream | SNR vs FFmpeg |
|---|---|
| Test fixtures | 84–92 dB |
| 2-minute fdk-aac v1 benchmark | 106 dB |
| 2-minute fdk-aac v2 benchmark | 96 dB |

**AV1.** Bit-exact with libdav1d (`framecrc`).

**HEVC.** Bit-exact with FFmpeg on all of the following, at every
thread count (frame and WPP row threads) and SIMD level:

- x265 streams:
  - 8, 10 and 12-bit
  - B-pyramid and open/closed GOP
  - AMP and rectangular partitions
  - weighted prediction (with fades)
  - slices and WPP
  - transform skip, scaling lists, lossless and `cu-lossless`
  - 16 and 32-pixel CTBs, QP groups, chroma QP offsets, deblocking offsets
  - SAO and deblocking off
  - TMVP off, 5 references, constrained intra
  - odd sizes, and 1080p (8- and 10-bit, 150 frames)
  - WPP over many CTB rows, with three slices per picture and 16x16 CTBs
- Kvazaar streams: tiles (uniform and as slices), dependent slices
  (`--slices wpp`), scaling lists, lossless, transform skip, AMP/SMP.

FFmpeg is not a usable reference for **tiles combined with WPP**: its
output for such streams is about 6 dB PSNR. RMP matches Kvazaar's own
reconstruction exactly, so that fixture uses the encoder's reconstruction
as its reference.

The fixtures in `tests/fixtures/hevc/` cover main, Main 10 with slices in
MKV, coding tools, tiles + WPP, multi-row WPP (`x265_wpp.mp4`,
`x265_wpp_ctu16_slices.mkv`), and damaged input. The threading tests
decode every fixture, clean and damaged, with 1 to 8 frame threads and 1
to 8 row threads, and require identical output and damage counts. `tools/make_fixtures.sh`
lists the commands that made them.

## 3. Performance vs FFmpeg

Setup:

- Decode to `/dev/null`, best of 9 runs, wall seconds.
- Machine: 2-vCPU Xeon @ 2.1 GHz (AVX2) in a shared VM, so expect ±10% noise.
- FFmpeg 6.1: `ffmpeg [-threads 1] -i X -f null -`, with libdav1d for AV1.
- RMP: `examples/decode`, with `RMP_THREADS=1` for the single-thread column.

Clips:

- Video: 5 s of 1080p30 `testsrc2` with temporal noise.
  - HEVC: x265 medium, CRF 26 (~4.7 Mb/s).
  - AV1: SVT-AV1 preset 8, CRF 35.
- Audio: 120 s, 48 kHz.
  - HE-AAC v1 at 64 kb/s and HE-AAC v2 at 32 kb/s (fdk-aac).
  - AC-3 5.1 at 448 kb/s and E-AC-3 5.1 at 640 kb/s.

| Clip | RMP, 1 thread | FFmpeg, 1 thread | ratio | RMP, auto | FFmpeg, auto | ratio |
|---|---|---|---|---|---|---|
| HEVC 1080p 8-bit | 1.94 | 0.90 | 2.16x | **1.17** | 0.68 | **1.72x** |
| HEVC 1080p 10-bit | 2.03 | 1.33 | 1.53x | **1.30** | 0.81 | **1.60x** |
| AV1 1080p 8-bit | 1.20 | 1.20 | 1.00x | 0.77 | 0.82 | 0.94x |
| AV1 1080p 10-bit | 1.47 | 1.37 | 1.07x | 0.95 | 0.91 | 1.04x |
| HE-AAC v1 stereo | 0.33 | 0.33 | 0.98x | 0.32 | 0.34 | 0.94x |
| HE-AAC v2 (PS) | 0.30 | 0.38 | 0.79x | 0.31 | 0.35 | 0.89x |
| AC-3 5.1 | 0.48 | 0.58 | 0.83x | 0.48 | 0.56 | 0.86x |
| E-AC-3 5.1 | 0.44 | 0.52 | 0.84x | 0.51 | 0.53 | 0.96x |

Ratios above 1 mean RMP is slower.

- **AV1:** rav1d runs at libdav1d speed, as expected.
- **Audio:** the decoders match FFmpeg or beat it. These figures include
  process start-up, demuxing and the S16 conversion on both sides.
- **HEVC, single thread:** RMP is 1.5–2.2x behind FFmpeg (FFmpeg uses
  hand-written assembly), against 2.3x for RMP's H.264 decoder.
- **HEVC, auto:** frame threading took the gap from 2.75x to about 1.7x;
  RMP now scales like FFmpeg. The HEVC rows were re-measured for that
  change on a noisier VM, so they don't compare one to one with the other
  rows. See `docs/THREADING.md` for the full HEVC threading table.

HEVC speed during this work, on a 1080p clip, single thread:

| Stage | Time |
|---|---|
| Before optimisation | 2.8 s (2.9x FFmpeg) |
| After removing per-block scratch zeroing and recomputed edges | 2.5 s |
| After adding MC SIMD | 2.0 s |
| After adding SAO, IDCT and deblocking SIMD | about 1.5 s (1.7–1.8x FFmpeg) |
| After the 16-bit IDCT, residual-add and intra SIMD, MC without bounds checks, and default-weight WP on the vector path | 1.7–2.2x FFmpeg on the clips above. On the 12.6 Mb/s clip: 5.27 → 4.20 s (1.9x → 1.5x FFmpeg). On the 4.7 Mb/s clip, instructions went from 4.73 G to 4.33 G (−8%). |

### What's left (HEVC)

1. **Threading** is done (`docs/THREADING.md`). Frame threads are the
   default and WPP row threads are optional. Neither was measured on more
   than two vCPUs for this change.
2. **Kernels.** Profile of the 4.7 Mb/s clip, single thread, inclusive:
   - inter prediction (MV prediction + MC): about 38%
   - in-loop filters: about 22%, including the SAO band copy
   - residual parsing and reconstruction: about 19%, of which CABAC
     decisions are about 7%
   - bS computation: about 4.5%
   - chroma deblocking and 12-bit luma deblocking are still scalar
3. **8-bit storage.** Storing 8-bit pictures as `u8` would halve the memory
   traffic, at the cost of a generic sample type. Two hyperthreads share
   one core's bandwidth, so this also limits threaded scaling.
4. **Allocation.** Each picture still allocates and fills a new
   `PicState` (about 2% of instructions). Recycling the planes of pictures
   that left the DPB would remove that.

## 4. Known limitations

- E-AC-3 SPX, AHT and dependent substreams have no test samples: FFmpeg's
  encoder cannot produce them. They are ported line by line from FFmpeg
  but are unverified.
- HEVC PCM blocks are implemented but untested: neither x265 nor Kvazaar
  emits PCM.
- On Matroska, RMP drops the 256 priming samples of FFmpeg-encoded AC-3,
  as the container's codec delay says. The `ffmpeg` CLI keeps them. The
  decoded samples are identical after aligning for that offset.
