# AAC and H.264 decoding in RMP

This document explains how AAC and H.264 decoding was added to RMP, so MP4
files can be decoded and not just remuxed. It covers the design, how each
part maps to FFmpeg and the ISO specifications, how the work was verified, and
how to extend it. Read it before changing anything under `src/codecs/aac`,
`src/codecs/h264`, `src/codecs/bitstream` or `src/codecs/dsp`.

It builds on [MP3_MP4.md](MP3_MP4.md): the MP4 demuxer already produces
packets and the `avcC` / AudioSpecificConfig extradata these decoders consume.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **AAC decoder** | AAC-LC (and the LC core of HE-AAC v1/v2), mono to 7.1, ADTS or raw access units. Sine and KBD windows, all four window sequences, M/S and intensity stereo, PNS, TNS, pulses. Output is interleaved S16 in FFmpeg's channel order. | `codecs::aac::AacDecoder` |
| **H.264 decoder** | Constrained Baseline, Main and High profiles, 8-bit 4:2:0 progressive. CAVLC and CABAC, I/P/B slices, 8x8 transform, scaling matrices, multiple slices, multiple and long-term references, list modification, MMCO, POC types 0-2, explicit and implicit weighted prediction, spatial and temporal direct, I_PCM, deblocking, frame_num gap filling. Output is planar I420 in display order. | `codecs::h264::H264Decoder` |
| **Shared bitstream** | MSB-first `BitReader` with Exp-Golomb codes; table-driven multi-level `Vlc`. | `codecs::bitstream` |
| **Shared DSP** | Radix-2 complex FFT; IMDCT via a half-length DCT-IV. | `codecs::dsp` |
| **Wiring** | `"aac"` and `"h264"` registered in `Resolver::decoder_for`; typed `CodecPrivate::Aac` / `CodecPrivate::H264` state. `VideoState` gained a `private` slot like `AudioState`. | `core::Resolver`, `core::context::state` |
| **Example** | `decode`: first audio stream to raw S16LE, first video stream to raw I420. | `examples/decode.rs` |

```bash
cargo run --release --example decode -- in.mp4 audio.pcm video.yuv
ffplay -f rawvideo -pixel_format yuv420p -video_size 1280x720 video.yuv
ffplay -f s16le -ar 48000 -ch_layout stereo audio.pcm
```

Pass `-` for an output you don't want.

The example applies the stream's `encoder_delay` / `encoder_padding` metadata
(§5.2 of MP3_MP4.md), so AAC priming and padding samples are dropped and the
PCM output matches `ffmpeg -f s16le` sample for sample. The decoder itself
always returns every decoded sample; trimming is the caller's job, as with
FFmpeg's `skip_samples`.

---

## 2. How the project values were applied

**1. Safety**

- There is no `unsafe` code and no `unwrap`, `expect`, `panic!`, `todo!` or
  `unreachable!` in non-test code.
- Every bitstream read goes through `BitReader`. It returns zero bits past
  the end and sets a sticky overrun flag. Callers check it at syntax-element
  boundaries, which keeps the hot loops branch-light.
- Every value read from the bitstream that later indexes something is
  range-checked first:
  - AAC: `max_sfb`, section lengths, scalefactors, TNS order, pulse
    positions and escape lengths.
  - H.264: parameter-set ids, `num_ref_idx`, `ref_idx`, `mb_type`,
    `sub_mb_type`, `coded_block_pattern`, `mb_qp_delta`, intra modes,
    `cabac_init_idc` and weights.
- Intra modes that would read samples outside the picture are rejected.
  For example, vertical prediction on the top row would underflow.
- Picture size is capped at 8192x4352, which bounds every allocation.
- Coefficient levels are clamped to the range a conforming 8-bit stream can
  produce (§8.5.12.1), and dequantisation runs in 64-bit. Damaged streams
  therefore can't overflow the integer transforms. This doesn't affect
  conforming streams (§7).
- Damage is concealed, not fatal:
  - **AAC.** A bad frame becomes correctly-sized silence, so timestamps stay
    continuous. This matches the MP3 decoder. `concealed_frames` counts these.
  - **H.264.** A bad slice stops at the failing macroblock. Undecoded
    macroblocks are copied from the latest reference picture. The picture is
    still deblocked, stored and output. `damaged_slices` counts these.
- Unsupported features return a clear `Message::Codec` error. These are
  interlace, 4:2:2/4:4:4, more than 8 bits, lossless, FMO, SP/SI slices and AAC
  Main/LTP/SSR/CCE.

**2. Efficiency**

- **AAC.** Huffman decoding is a one- or two-lookup `Vlc`. `|q|^(4/3)` is a
  8192-entry table. The IMDCT is O(N log N): a 512-point FFT for long windows,
  64-point for short. Windows and transforms are built once in a `OnceLock`.
  A 60 s stereo file decodes in under 0.1 s (about 700× real time), on par
  with FFmpeg.
- **H.264:**
  - Luma interpolation is separable. It fetches one (w+5)×(h+5) window
    (edge-clamped only when the block crosses the border), then computes only
    the half-sample planes the fraction needs.
  - Prediction weighting is chosen once per block, not per pixel.
  - The deblocker borrows macroblock data instead of copying it.
  - Pictures in the DPB are `Arc`-shared, and parameter sets are `Arc`-shared.
    The `Decoder` trait's per-call `state.clone()` therefore copies only
    pointers.

  A 1280×720, 150-frame High-profile clip decodes at about 105 fps on one
  core. FFmpeg's plain-C build is about 2.3× faster (`-cpuflags 0`), and its
  SIMD build about 4× faster. §8 lists the next optimisations.

**3. Readability**

- Each file covers one stage: NAL framing, parameter sets, slice header,
  CABAC, CAVLC, macroblock syntax, inter prediction, reconstruction,
  deblocking and DPB. AAC is split the same way: configuration, channel
  stream and filterbank.
- Module and function docs cite the clause of ISO/IEC 14496-3 / 14496-10 and
  the FFmpeg file that does the same job.
- Normative tables are generated, not hand-typed (§5).

---

## 3. Architecture map

```
src/codecs/
├── bitstream/
│   ├── reader.rs        BitReader: bits, signed, ue/se, align, overrun flag
│   └── vlc.rs           Vlc: multi-level lookup tables from (code, len, symbol)
├── dsp/
│   ├── fft.rs           in-place radix-2 FFT
│   └── mdct.rs          Imdct (DCT-IV with N/4-point FFT)
├── aac/
│   ├── config.rs        AudioSpecificConfig, ADTS, PCE, channel layouts
│   ├── ics.rs           ics_info, sections, scalefactors, Huffman spectra,
│   │                    pulses, TNS, dequantisation, PNS
│   ├── decoder.rs       element loop, M/S, intensity, IMDCT + windows, Decoder impl
│   └── tables.rs        generated (tools/gen_aac_tables.py)
└── h264/
    ├── nal.rs           avcC, length-prefixed / Annex B split, unescaping
    ├── params.rs        SPS (+VUI reorder info), PPS, scaling lists, LevelScale
    ├── slice.rs         slice header, list modification, weights, MMCO
    ├── cabac.rs         arithmetic decoder, context init, residual blocks
    ├── cavlc.rs         coeff_token / levels / total_zeros / run_before
    ├── mb.rs            MbInfo, neighbour access, macroblock_layer for both coders
    ├── inter.rs         MV prediction, P_Skip, direct spatial/temporal, MC + weights
    ├── recon.rs         IDCT 4x4/8x8, DC transforms, intra 4x4/8x8/16x16/chroma,
    │                    6-tap luma and bilinear chroma interpolation
    ├── deblock.rs       bS derivation and edge filters
    ├── decode.rs        slice_data loop, reconstruction per macroblock
    ├── dpb.rs           POC, reference lists, marking, output bumping
    ├── decoder.rs       picture lifecycle, concealment, cropping, Decoder impl
    ├── consts.rs        small normative constants (scans, CBP maps, α/β/tC0)
    └── tables.rs        generated (tools/gen_h264_tables.py)
```

The data flow is the existing RMP pipeline:

```
Mp4Demuxer::read ─► Packet ─► Resolver::decoder_for(stream)
                                 ├─ "aac"  ─► AacDecoder  ─► Frame::Audio (S16, 1024 samples)
                                 └─ "h264" ─► H264Decoder ─► Frame::Video (I420, display order)
```

---

## 4. AAC in detail

### 4.1 Configuration (`config.rs`)

`parse_audio_specific_config` reads the object type and sample rate. An
explicit 24-bit rate is mapped to the nearest band table (Table 4.82). It then
reads the channel configuration and GASpecificConfig. A configuration of 0
means the layout comes from a program_config_element.

For SBR/PS (object types 5 and 29), the extension rate is skipped, the core
object type is read, and `sbr` is set.

Each layout is a list of `ElementSlot { kind, tag, out }` in bitstream order.
`out` maps an element's channels to FFmpeg's native order, for example
FL FR FC LFE BL BR for 5.1. A raw_data_block's n-th SCE/CPE/LFE goes to the
n-th slot of that kind.

ADTS headers are recognised by their sync word. Without extradata, each
header provides the configuration. With extradata, headers are only stripped.

### 4.2 Channel stream (`ics.rs`)

`decode_ics` runs the stages of ISO/IEC 14496-3 §4.6 in order:

| Stage | Function | Notes |
| --- | --- | --- |
| ics_info | `decode_ics_info` | Window grouping, band tables per rate, TNS limits. Main prediction / LTP is rejected. |
| Section data | `decode_band_types` | Escape-coded lengths, bounded by `max_sfb`. |
| Scalefactors | `decode_scalefactors` | Three running offsets (global gain, noise − 90, intensity), clamped as FFmpeg does. |
| Pulses | `decode_pulse`, `apply_pulses` | Applied to the integers before requantisation, as the specification orders it. |
| TNS | `decode_tns` | Reflection coefficients `sin(c / iq)`, with separate scales for positive and negative `c`. |
| Spectral data | `decode_spectrum`, `decode_band` | Codebooks 1-11. Quads/pairs, sign bits, then escapes. Grouped short windows are written straight to their window. |
| Dequantise + PNS | `dequantise` | `sign·|q|^(4/3)·2^((sf−100)/4)`. Noise bands use FFmpeg's LCG (`1664525·x + 1013904223`, seed `0x1f2e3d4c`), so noise matches FFmpeg sample for sample. |

### 4.3 Stereo and filterbank (`decoder.rs`)

- **M/S** is applied per band (`L = M + S`, `R = M − S`). As in FFmpeg, it
  skips noise and intensity bands.
- **Intensity stereo** copies the left band scaled by `±2^(−is_pos/4)`. The
  sign is inverted where M/S is signalled.
- **TNS** runs the all-pole filter over the band range, walking up or down
  per filter.
- **Filterbank.** `synthesize` performs the IMDCT. It uses N = 2048, or 8 ×
  256 for short windows, with the specification's 2/N factor. Windowing
  follows §4.6.11: the left half uses the previous frame's window shape and
  the right half the current one. LONG_START and LONG_STOP use the short
  window slopes and zero regions. Overlap-add then uses 1024 saved samples
  per output channel.

KBD windows are computed from a Bessel-I0 series (α = 4 long, 6 short). The
Princen-Bradley condition `w[n]² + w[N−1−n]² = 1` is unit-tested for all
windows.

---

## 5. H.264 in detail

### 5.1 Framing and parameter sets

With an `avcC` record, packets are always split by their length prefix. Annex
B start codes are only a fallback when the length framing doesn't fit.

This matters because a 4-byte prefix such as `00 00 01 B0` (a 432-byte NAL)
looks exactly like a start code. Treating it as one silently dropped those
frames. That was the one bug the B-frame tests caught.

Without `avcC`, packets are parsed as Annex B and SPS/PPS may arrive in-band.

The SPS parser keeps `max_num_reorder_frames` / `max_dec_frame_buffering`
from the VUI. Without them, the DPB size comes from the level's MaxDpbMbs
(Table A-1).

Scaling lists follow fall-back rules A and B (Table 7-2). They are turned into
`LevelScale` tables (normAdjust × weight) once per PPS.

### 5.2 Entropy decoding

- **CABAC** (`cabac.rs`) is the bit-exact engine of §9.3.3.2 (9-bit range and
  offset). Context initialisation uses Tables 9-12 to 9-33. `residual`
  handles all block categories, using the 8x8 significance/last context maps
  of Table 9-43.
- **CAVLC** (`cavlc.rs`) builds `Vlc`s for coeff_token (5 tables, including
  chroma DC), total_zeros and run_before. Level decoding has the adaptive
  suffix length and the escape forms of §9.2.2.1.
- **Macroblock syntax** (`mb.rs`) walks `macroblock_layer` once for both
  coders. Each element dispatches to CABAC or CAVLC. The neighbour-dependent
  contexts are derived from `MbInfo` of the current and neighbouring
  macroblocks: the CABAC ctxIdxInc rules of §9.3.3.1.1 and the CAVLC nC of
  §9.2.1. `MbInfo` holds type, cbp, nnz, DC flags, intra modes, mv / ref_idx /
  ref picture id, |mvd| and direct flags.

### 5.3 Prediction

- **MV prediction** (`inter.rs`) implements §8.4.1.3 directly:
  - Neighbours A, B and C, with D replacing C when C is unavailable.
  - Inside the current macroblock, a 4x4 block is available only if its
    decoding-order index is lower than the partition's first block. This one
    rule reproduces every partition and sub-partition case.
  - The 16x8 and 8x16 directional rules, then the median.
- **P_Skip** zeroes the motion vector at picture edges or with zero-motion
  neighbours (§8.4.1.1).
- **Direct prediction:**
  - Spatial: MinPositive reference selection, then colZeroFlag on the
    co-located block of RefPicList1[0], with `direct_8x8_inference` corner
    blocks.
  - Temporal: the co-located reference is mapped into RefPicList0 by picture
    identity, and DistScaleFactor scaling is applied.
  - The co-located motion is stored with every picture (`MbMotion`).
- **Weighted prediction** supports explicit tables and implicit POC-distance
  weights (§8.4.2.3).
- **Intra prediction** (`recon.rs`) covers all nine 4x4 and 8x8 modes. The
  8x8 modes use the reference-sample filter of §8.3.2.2.1. It also covers
  16x16 and chroma DC/H/V/plane, and constrained_intra_pred availability.

### 5.4 Residual, deblocking, pictures

- **Residual.** 4x4 and 8x8 inverse transforms, the Intra16x16 luma DC
  Hadamard and the 2x2 chroma DC, each with its own dequantisation rule.
- **Deblocking** (`deblock.rs`) runs over the whole picture after all its
  slices. Intra prediction must see unfiltered samples, so filtering waits
  until then.
  - bS 4/3/2 come from the intra and coefficient rules.
  - bS 1 compares reference *pictures*, not indices, and handles the
    bi-predicted pairings.
  - Luma uses the normal and strong filters. Chroma edges reuse the luma bS
    of edges 0 and 2.
  - Per-slice α/β offsets and `disable_deblocking_filter_idc` 2 (no
    filtering across slice edges) are honoured.
- **DPB** (`dpb.rs`):
  - POC types 0, 1 and 2.
  - Initial lists for P and B (with the identical-list swap), then
    modification.
  - Sliding window and all six MMCOs.
  - Non-existing frames for frame_num gaps.
  - Output in POC order once more than the reorder depth is waiting. IDR and
    MMCO 5 flush the queue first.
- **Timestamps.** Every output picture keeps the `pts` of the packet that
  carried it, so B-frame reordering yields increasing presentation times.

### 5.5 Tables and licences

The algorithms are written from the ISO specifications. Normative numeric
tables come from permissively licensed sources through generators, the same
way `tools/gen_mp3_tables.py` uses minimp3:

| Table | Source | Generator |
| --- | --- | --- |
| AAC Huffman codebooks 1-11 and scalefactor codebook, swb offsets | vo-aacenc (Apache-2.0) | `tools/gen_aac_tables.py` (checks prefix-freeness and Kraft sums) |
| CABAC init (460 contexts × 4), rangeTabLPS, transIdx, CAVLC codes, default scaling lists | OpenH264 (BSD-2-Clause) | `tools/gen_h264_tables.py` |

The small constants in `h264/consts.rs` were typed in and cross-checked
against FFmpeg's copies: scans, CBP maps, the chroma QP table, normAdjust,
α/β/tC0 and the 8x8 CABAC context maps. No FFmpeg (LGPL) code was ported.

---

## 6. Core changes and why

| Change | Reason |
| --- | --- |
| `CodecPrivate::Aac`, `CodecPrivate::H264` | Typed decoder state (overlap buffers, DPB) without changing the `Decoder` trait. |
| `VideoState::private` | Video decoders need persistent state just like audio. Raw video uses `CodecPrivate::None`. |
| `Resolver::decoder_for`: `"aac"`, `"h264"` | Codec IDs are FFmpeg's names, as produced by the MP4 demuxer. |

---

## 7. Verification

All references are FFmpeg 6.1 decodes. Fixtures were encoded with FFmpeg's
native AAC encoder and libx264.

**AAC: ±1 LSB, 102-109 dB SNR, sample counts identical.** Tested cases:

- Mono 8 kHz
- Stereo 22.05/32/44.1/48/96 kHz
- 5.1
- `-aac_pns 1 -aac_is 1 -aac_tns 1 -aac_ms 1`, and the `fast` coder
- Transient content that forces LONG_START/EIGHT_SHORT/LONG_STOP sequences
- ADTS input

Instrumented runs confirmed that short windows, PNS, intensity (codebooks 14
and 15) and TNS were all exercised.

**H.264: bit-exact, every frame and every plane.** Tested cases:

| Area | Configurations |
| --- | --- |
| Profiles and entropy coding | Baseline CAVLC (I-only and I/P); Main CABAC I/P/B; Main CAVLC with B-frames; High with 8x8 transform. |
| Prediction and references | B-pyramid; spatial and temporal direct; 6 references with `b-adapt=2`; explicit weighted P (fades, `weightp=2`); implicit weighted B (`weighted_bipred_idc` 2). |
| Slices, quantisation, filtering | 5 slices per picture; JVT scaling matrices; `constrained-intra`; QP 1 and QP 51; deblock offsets −3/+3; deblocking off. |
| Stream structure | Intra refresh; open GOP; frame cropping (100x58); a 1280×720, 150-frame clip. |

**Tests in `tests/codecs/`:**

- `aac_decoder_test.rs` checks mono M4A, 5.1 channel order, and ADTS with
  PNS/IS/TNS against FFmpeg references. It also covers concealment and
  errors.
- `h264_decoder_test.rs` checks bit-exactness against FFmpeg for the CAVLC
  and High-profile fixtures, display order, Annex B input without extradata,
  reset after seek, and garbage input.

**Robustness.** `tests/container/robustness_test.rs` decodes the fixtures
after corrupting random packets and their extradata (bit flips, truncation,
0x00/0xFF runs). It also feeds random payloads behind valid slice NAL
headers.

The deep runs were done in debug builds, where integer overflow panics:
9 seeds × 12-16 thousand corrupted streams, all clean. The only finding was
a dequantisation overflow on absurd levels, which led to the clamp in §2.

`cargo fmt --check` and `cargo clippy -- -D warnings` pass.

```bash
bash tools/make_fixtures.sh          # needs FFmpeg with libx264 (and libmp3lame)
cargo test
RMP_FUZZ_ROUNDS=4000 RMP_FUZZ_SEED=11 cargo test --test container robustness
```

---

## 8. Known limitations (with pointers)

| Limitation | Where to start |
| --- | --- |
| Interlaced H.264 (field pictures, MBAFF) is rejected. | `params::parse_sps` rejects `frame_mbs_only_flag = 0`. Needs field POC, field reference lists, MBAFF neighbour derivation in `mb.rs`, and field deblocking. |
| H.264 4:2:2, 4:4:4, >8-bit, lossless; FMO/ASO; SP/SI; data partitioning; SVC/MVC. | `params.rs` checks. 4:4:4 needs CABAC contexts 460-1023 (extend `gen_h264_tables.py`). |
| HE-AAC: SBR and PS are not reconstructed. The AAC-LC core is decoded at half rate and bandwidth, and reported with that rate. | Add `aac/sbr.rs` (QMF analysis/synthesis, HF generation, envelope adjustment) driven by the FIL element's extension payload. |
| AAC Main prediction, LTP, SSR, coupling channels (CCE), 960-sample frames. | `ics::decode_ics_info` and the `Cce` arm in `decoder.rs` report these. |
| H.264 speed is about 2.3× FFmpeg's C path. | In profile order: `mc_rect` and interpolation (SIMD-friendly fixed-size kernels per block size); deblocking (filter whole 16-sample edges with early-out on α/β); CABAC (FFmpeg's scaled `low` with byte-wise refills); then frame-level threading. |
| Error concealment copies co-located blocks; it doesn't interpolate motion. | `H264DecoderState::conceal`. |
| The `Decoder` trait still clones state per call. This is cheap now (Arc-shared DPB and parameter sets) but not free. | Same API note as in MP3_MP4.md §8. |

---

## 9. How to build on this

### Add an AAC tool (e.g. SBR)

1. Parse the extension payload in the `Fil` arm of
   `AacDecoderState::decode_elements` instead of skipping it.
2. Keep per-channel SBR state next to `History`.
3. Run SBR after `synthesize` (it works on the time-domain output), and
   report the doubled rate in `DecodedFrame`.
4. Validate it against FFmpeg the way §7 does.

### Add an H.264 feature

- **Syntax:**
  - Slice-level syntax goes in `slice.rs`.
  - Macroblock-level syntax goes in `mb.rs`. Add both the CABAC and the
    CAVLC branch, and keep neighbour context in `MbInfo`.
  - Anything a later picture needs (motion for direct prediction) belongs in
    `dpb::MbMotion`.
- **Comparing against FFmpeg.** `ffmpeg -i x.mp4 -f rawvideo -pix_fmt yuv420p
  ref.yuv` versus `cargo run --release --example decode -- x.mp4 - ours.yuv`.
  Find the first differing macroblock in raster order: the error pattern
  usually identifies the stage.
  - A constant error per 4x4 block points at DC transforms.
  - Errors only in the bottom rows of a macroblock point at deblocking of the
    next row.
  - Missing frames point at framing or output.

### Rules for contributors touching these decoders

- **Bitstream values.** Check every value read from the bitstream before it
  indexes, allocates or loops. Prefer `get`, and return `BadMb` / `Invalid`.
- **Arithmetic.** Arithmetic on values derived from the stream (POC, motion
  vectors, coefficients) must not overflow. Use wrapping/i64 arithmetic or the
  spec's range clamp.
- **Generated tables.** Keep them generated: change the generator scripts,
  not `tables.rs`.
- **Bit-exactness.** Any change to H.264 reconstruction must stay bit-exact
  on the fixtures. Any change to AAC must stay within ±1 LSB.
- **Fuzzing.** Run the robustness tests with a high `RMP_FUZZ_ROUNDS` before
  merging.

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `aac/config.rs` | `libavcodec/mpeg4audio.c`, `adts_header.c`, `aac/aacdec.c` (`decode_pce`, GA config) |
| `aac/ics.rs` | `aac/aacdec.c` (`decode_ics`, `decode_scalefactors`, `decode_tns`), `aacdec_proc_template.c` |
| `aac/decoder.rs` | `aac/aacdec.c` (`decode_cpe`, element loop), `aacdec_dsp_template.c`, `aacdec_float.c` |
| `h264/nal.rs` | `h2645_parse.c`, `h264_parse.c` |
| `h264/params.rs` | `h264_ps.c` |
| `h264/slice.rs`, `h264/decoder.rs` | `h264_slice.c`, `h264dec.c` |
| `h264/cabac.rs`, `h264/mb.rs` | `cabac.c`, `h264_cabac.c`, `h264_cavlc.c`, `h264_mvpred.h` |
| `h264/cavlc.rs` | `h264_cavlc.c` |
| `h264/inter.rs` | `h264_mvpred.h`, `h264_direct.c`, `h264_mb.c` |
| `h264/recon.rs` | `h264idct_template.c`, `h264pred_template.c`, `h264qpel_template.c`, `h264chroma_template.c` |
| `h264/deblock.rs` | `h264_loopfilter.c`, `h264dsp_template.c` |
| `h264/dpb.rs` | `h264_refs.c`, `h264_picture.c`, `h2645_sei`/`h264_parse.c` (POC) |
