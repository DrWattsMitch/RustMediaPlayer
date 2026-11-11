# AAC and H.264 encoding in RMP

This document explains how the AAC-LC and H.264 encoders were added, so a
transcode can produce MP4 files that play everywhere, not only copy streams
or fall back to FLAC. It covers the design, how each stage maps to FFmpeg,
x264 and the ISO specifications, how the work was verified, and how to
extend it. Read it before changing anything under `src/codecs/aac/encoder`
or `src/codecs/h264/encoder`.

It builds on [AAC_H264.md](AAC_H264.md): both encoders reuse the decoders'
normative pieces (tables, transforms, prediction, deblocking), and the
decoders are how the encoders are tested.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **AAC-LC encoder** | Mono to 7.1 (channel configurations 1-7), every standard rate 8-96 kHz. Block switching with one frame of lookahead, sine-window MDCT, psychoacoustic thresholds, per-band M/S, scalefactor rate loop, optimal Huffman sectioning, bit reservoir. Raw access units plus a 2-byte AudioSpecificConfig. | `codecs::aac::AacEncoder` |
| **H.264 encoder** | Constrained Baseline: CAVLC, IDR + P pictures, one reference, Intra16x16 / Intra4x4, P_Skip and P_L0 16x16 / 16x8 / 8x16 / 8x8, quarter-sample motion, deblocking, frame cropping. Constant QP or average bit rate. `avcC` extradata, 4-byte length-prefixed NAL units. | `codecs::h264::H264Encoder` |
| **Shared bitstream** | MSB-first `BitWriter` with Exp-Golomb codes and exact bit counts. | `codecs::bitstream::BitWriter` |
| **Shared DSP** | Forward MDCT through the existing half-length DCT-IV. | `codecs::dsp::Mdct` |
| **Gapless trimming** | Transcodes drop the source's `encoder_delay` / `encoder_padding` samples after decoding, as FFmpeg does with `skip_samples`. | `process::audio::Gapless` |
| **Wiring** | `-c:a aac`, `-c:v h264`, `-b:a`, `-b:v`, `-qp`, `-g`. MP4/M4A/MOV re-encode audio to AAC by default. Scaling, `-b:v` or `-qp` into MP4/MOV/Matroska encode H.264. `Resolver::encoder_for` knows both codecs. | `transcode`, `core::Resolver` |

```bash
rmp -i song.flac song.m4a -b:a 192k                    # AAC-LC
rmp -i frames.y4m video.mp4 -c:v h264                  # QP 23
rmp -i input.mkv out.mp4 -s 1280x720 -b:v 2M -b:a 128k # both, scaled
```

---

## 2. How the project values were applied

**1. Safety**

- There is no `unsafe` code and no `unwrap`, `expect`, `panic!`, `todo!` or
  `unreachable!` in non-test code.
- Every input the encoders take is checked before use: sample rate and
  channel count (AAC), picture size, pixel format and frame data length
  (H.264). Bad input returns a `Message::Codec` error naming the fix
  (`-ar 48000`, `-pix_fmt yuv420p`, even dimensions).
- Every value the encoders write is in range by construction:
  - AAC: quantised magnitudes ≤ 8191 (each band's scalefactor floor is
    computed from its peak), scalefactors in 0-255, consecutive differences
    ≤ 60 (§3.4), at most 6144 bits per channel per frame.
  - H.264: motion vectors within the level's vertical range (Table A-1)
    and ±2048 horizontally, levels coded with the full escape range of
    §9.2.2.1, QP 0-51.
- The H.264 encoder reconstructs pictures with the decoder's own
  `recon` and `deblock` code, so its references can't drift from what a
  decoder sees. The tests check this bit for bit.
- The tests also pass in debug builds, where integer overflow panics.

**2. Efficiency**

- **AAC.** The MDCT is O(N log N) through the existing FFT. Windows and
  transforms are built once in a `OnceLock`. Bit counts come from table
  lookups and are exact, so the rate loop never re-encodes to measure.
  A 10 s stereo file encodes in about 0.11 s (about 90× real time).
- **H.264:**
  - Integer motion search runs on an edge-padded copy of the reference
    (branch-free SAD).
  - Sub-sample search reads three precomputed half-sample planes per
    reference (x264's `hpel` planes): each candidate costs one or two
    lookups per sample instead of a 6-tap filter. This made the encoder
    1.45× faster without changing a single output bit.
  - P_Skip is tried first and taken without further search when its
    residual quantises away.
  - Pictures and parameter sets are `Arc`-shared, so the `Encoder` trait's
    per-call `state.clone()` copies only pointers.

  720p encodes at about 25 fps and 640x360 at about 40 fps, on one core.
  x264 is much faster (SIMD, threads); §7 lists the next optimisations.

**3. Readability**

- Each file covers one stage. AAC: psychoacoustics, quantisation and
  sectioning, frame loop. H.264: transform and quantisation, CAVLC
  writer, motion-estimation helpers, macroblock coding, picture loop.
- Module and function docs cite the clause of ISO/IEC 14496-3 / 14496-10
  and the FFmpeg or x264 file that does the same job.
- Normative tables are not duplicated: the encoders read the decoders'
  generated tables (Huffman codebooks, band offsets, CAVLC codes).

---

## 3. AAC in detail

### 3.1 Frame structure (`encoder/mod.rs`)

```
input (any sample format) ─► per-channel f32 buffers (16-bit scale)
  └─ every 1024 samples, with 2048 + 1024 buffered:
       block switching ─► window + MDCT ─► psy thresholds ─► M/S
       ─► rate loop (quantise, sectioning) ─► bitstream ─► packet
```

- **Delay.** The first frame overlaps 1024 samples of silence
  (`ENCODER_DELAY`, FFmpeg's `initial_padding`). Packet `pts` count from
  that silence. The transcode stores `encoder_delay = 1024` on the output
  stream, so the MP4 muxer writes an edit list.
- **End.** `flush` pads with zeros and emits frames until every input
  sample has been output: `ceil(n/1024) + 1` frames for `n` samples. The
  last packet's duration counts only real samples, so the presented
  duration equals the input exactly (checked with `ffprobe`).
- **Layout.** Elements come from `AacConfig::from_channel_config`, so input
  channels in FFmpeg order map to SCE/CPE/LFE exactly as the decoder maps
  them back. Seven channels have no channel configuration and are refused.

### 3.2 Block switching (`psy.rs`)

`has_attack` high-passes the *next* frame's short-window region
(samples 448-1600 of its window), splits it into 128-sample blocks, and
reports an attack when a block's energy is more than 10× the running
average and above an absolute floor. `next_sequence` applies Table 4.4's
transitions:

| Previous | Attack next | This frame |
| --- | --- | --- |
| ONLY_LONG, LONG_STOP | no / yes | ONLY_LONG / LONG_START |
| LONG_START | (any) | EIGHT_SHORT |
| EIGHT_SHORT | no / yes | LONG_STOP / EIGHT_SHORT |

Both channels of a CPE always share a window sequence and grouping, so
`common_window` and M/S are always available. The LFE is always long.
Short windows are grouped by energy (`grouping`): a window joins the
current group unless its energy is more than 3× above or below the group's
average.

### 3.3 Psychoacoustics (`psy.rs`)

This is a compact version of 3GPP TS 26.403 §5.4, which FFmpeg's default
`aacpsy.c` also follows:

1. Band energies per scalefactor band (per window for short blocks).
2. Spreading on the Bark scale, applied to band *energies* (not
   densities): −15 dB/Bark towards higher bands, −27 dB/Bark towards lower
   ones.
3. A fixed 29 dB SNR below the spread energy.
4. The absolute threshold of hearing (Terhardt, 0 dBFS = 96 dB SPL),
   capped at 20 dB SPL.

Two details matter more than they look. Both were found by measuring
against FFmpeg (§6.1):

- Spreading densities instead of energies let one loud narrow low band
  raise a wide high band's threshold by the ratio of their widths.
- The uncapped ATH curve rises to over 100 dB SPL at 18 kHz, which threw
  away everything above about 13 kHz even at 256 kb/s.

### 3.4 Quantisation and the rate loop (`quant.rs`, `mod.rs`)

- **Quantiser.** `q = int(|x|^¾ · 2^(-3(sf-100)/16) + 0.4054)`, inverted by
  the decoder's `|q|^(4/3) · 2^((sf-100)/4)`. `|x|^¾` is computed once per
  frame.
- **Per band.** `choose_sf` starts from the analytic noise estimate of a
  uniform quantiser in the `|x|^¾` domain,
  `noise ≈ (4/27) · Δ² · Σ|x|^½`. It then refines against the real error
  to the coarsest scalefactor whose noise stays within the band's
  allowance. Bands whose energy is below the allowance aren't coded.
- **Allowance.** One noise offset `g` (in dB) applies to the whole frame,
  in the 3GPP form (§5.6.1.1):
  - `g ≤ 0`: `T · 10^(g/10)`. All thresholds drop together.
  - `g > 0`: `(T^¼ + r)^4`. Quiet bands gain noise faster than loud ones,
    so noise tends towards white as the bit rate drops instead of piling
    into loud bands. `r` is scaled by the frame's mean `T^¼`, so both
    branches agree on a flat spectrum.
- **Search.** `search_offset` brackets from the previous frame's offset,
  then bisects to 0.25 dB for the finest offset that fits the budget.
- **Scalefactor chain.** `constrain_scalefactors` keeps consecutive coded
  scalefactors within ±60. The overflow floors are first made 60-Lipschitz
  in both directions, so a single forward clamp always succeeds.
- **Sectioning.** `section` is FFmpeg's `encode_window_bands_info` trellis
  over the 12 codebooks, with a section header costing 4 + length bits.
  Bit counts are the normative codeword lengths, so the count used by the
  rate loop equals the length written.
- **Budget.** The budget follows perceptual entropy (`Σ width ·
  log2(E/T)`): a frame's share is `PE / running-average PE`, clamped to
  0.6-2.0 (at least 1.5 for short-window frames). It is limited by the
  bits saved in the reservoir and by the decoder buffer (6144 bits per
  channel).
- **Bandwidth.** The cutoff follows FFmpeg's `AAC_CUTOFF_FROM_BITRATE`, for
  example 16 kHz at 64 kb/s per channel and 20 kHz at 128 kb/s per channel.

### 3.5 Mid/side

Per band, M = (L+R)/2 and S = (L−R)/2 are used when the M/S perceptual
entropy is lower than L/R's. Both thresholds are set to half the lower L/R
threshold, because noise in M and S adds up in each output channel.
`ms_mask_present` is 0, 1 or 2, whichever is exact.

---

## 4. H.264 in detail

### 4.1 Stream structure (`encoder/mod.rs`)

- **SPS.** `profile_idc` 66 with `constraint_set0/1` (Constrained
  Baseline), the smallest level that fits the size, macroblock rate and
  bit rate (Table A-1), `pic_order_cnt_type` 2, one reference frame, and
  cropping for sizes that aren't multiples of 16. VUI carries timing and
  `bitstream_restriction` (`max_num_reorder_frames = 0`,
  `max_dec_frame_buffering = 1`), so decoders output each picture at once.
- **PPS.** CAVLC, deblocking control present, QP 26 base.
- **Slice header.** One slice per picture, `slice_type` 7 (I) or 5 (P),
  `slice_qp_delta`, deblocking on with zero offsets.
- **GOP.** An IDR every `-g` pictures (default 250, as x264). `frame_num`
  restarts at each IDR and `idr_pic_id` alternates.
- **Packets.** One access unit per packet, 4-byte length prefixes,
  emulation prevention per §7.4.1, `pts = dts` passed through.

### 4.2 Macroblock decisions (`macroblock.rs`)

Every candidate is coded for real: prediction, transform, quantisation,
reconstruction and CAVLC into a scratch writer. The type is then chosen by
`SSD + λ²·bits`, where `λ² = 0.85 · 2^((QP−12)/3)` (x264's RD λ). The bit
count is therefore exact.

- **I pictures.**
  - Intra16x16: the mode with the lowest SATD.
  - Intra4x4: each block's mode by `SATD + λ · mode bits`, block by block
    on the reconstructed neighbours.
  - Chroma: the mode with the lowest SATD over both planes.
  - The better of the two by RD.
- **P pictures**, in order:
  1. P_Skip. Its prediction is taken outright when the residual would
     quantise (and decimate) to nothing.
  2. 16x16 motion search, seeded with the skip vector and the top and left
     neighbours' vectors.
  3. 8x8, 16x8 and 8x16, only when the 16x16 match is poor. Each partition
     is predicted from the ones before it, as the decoder will.
  4. Intra, when its SATD estimate competes with the best inter cost.
  5. RD among skip, the best inter partitioning and the best intra.
- **Motion search.** Diamond search at steps 4, 2, 1 on SAD +
  λ·mv bits. Then half-sample and quarter-sample square refinement on
  SATD, read from the half-sample planes (`me::HalfPel`, Table 8-12). The
  `half_pel_planes_match_mc_luma` test shows these are exactly the
  decoder's interpolation.
- **Quantisation.** Deadzone rounding: 1/3 of a step for intra, 1/6 for
  inter (JM / x264). Luma DC Hadamard for Intra16x16 and the 2x2 chroma DC.
  x264's decimation drops inter 8x8s, or the whole macroblock's luma, or a
  chroma component's AC, when only a few isolated ±1 levels survive.

### 4.3 Exactness

The neighbour derivations are written to mirror the decoder clause for
clause. They cover:

- macroblock and 4x4 availability, and intra mode prediction (8.3.1.1);
- the CAVLC nC of each block (9.2.1);
- motion vector prediction, including the 16x8/8x16 directional rules and
  the decoding-order availability inside the macroblock (8.4.1.3);
- P_Skip motion (8.4.1.1).

The residual writer `vlc::write_block` is the exact inverse of
`cavlc::residual`. A test round-trips coefficient sets through both,
including escapes up to ±20000 and every nC class.

Reconstruction calls `recon::pred4x4`, `pred_block`, `dequant4`,
`luma_dc_dequant`, `chroma_dc_dequant`, `idct4_add`, `mc_luma`,
`mc_chroma`, `store_prediction` and finally `deblock_frame`, in the same
order as the decoder. The encoder's `MbInfo` (kind, cbp, nnz, QP, motion,
`ref_id`) is what the deblocking filter reads, so bS derivation matches too.

### 4.4 Rate control

- **`-qp N`.** P pictures use N and IDR pictures N − 3 (x264's default
  `ipratio` of 1.4).
- **`-b:v R`.**
  - Each picture type keeps a complexity estimate `K = bits · Qstep`,
    smoothed 50/50.
  - The next P picture gets `QP = 6 · log2(K / (0.625 · target))`.
  - `target` is the per-picture budget, corrected by the over- or
    under-spend spread over about a second.
  - QP moves by at most ±4 between P pictures. IDR pictures use P − 3.
  - The first picture starts from a bits-per-pixel guess.
- **Default.** QP 23, the same number as x264's default CRF, though
  constant QP spends more bits on complex content than CRF does.

---

## 5. Core changes and why

| Change | Reason |
| --- | --- |
| `CodecPrivate::AacEncoder`, `CodecPrivate::H264Encoder` | Typed encoder state without changing the `Encoder` trait. |
| `BitWriter` in `codecs::bitstream` | One writer with exact bit counts for both encoders. `append` splices a candidate's bits into the slice. |
| `Mdct` in `codecs::dsp` | The forward transform folds its input and reuses `Imdct`'s DCT-IV and FFT (same module, nothing made public). |
| `process::audio::Gapless`, `slice_frame` | Decoded AAC/MP3 still contains the source's priming. Re-encoding it would add 1024 more samples of delay per generation and shift A/V sync. |
| `transcode`: options, defaults, H.264 route | `-b:a`, `-b:v`, `-qp`, `-g`. AAC is the MP4 default, H.264 when video must be re-encoded. The H.264 route forces `yuv420p` and passes the frame rate for rate control and the level. |
| `transcode`: H.264 only for 4:2:0 | `-pix_fmt yuv444p` into Matroska still gives raw video, as before. |

---

## 6. Verification

All references are FFmpeg 6.1 (native `aac` encoder, libx264).

### 6.1 AAC

Test signals mix a vibrato chord, a pink-ish noise bed, 20 drum hits and a
sweep. "SNR" is waveform SNR after FFmpeg decodes each file. That isn't a
perceptual measure, but it catches bandwidth, rate-loop and stereo
mistakes, and it is how §3.3's two fixes were found.

| Rate / channels | Bit rate (RMP / FFmpeg) | SNR dB (RMP / FFmpeg) |
| --- | --- | --- |
| 8 kHz mono, 16k | 16 065 / 16 101 | 19.0 / 17.9 |
| 22.05 kHz stereo, 48k | 48 098 / 47 837 | 14.2 / 12.9 |
| 44.1 kHz stereo, 64k | 64 011 / 64 352 | 11.5 / 10.1 |
| 44.1 kHz stereo, 128k | 127 909 / 133 154 | 15.5 / 14.3 |
| 44.1 kHz stereo, 256k | 255 630 / 266 850 | 23.9 / 21.4 |
| 48 kHz mono, 96k | 95 649 / 96 922 | 13.9 / 12.7 |
| 48 kHz 5.1, 384k | 382 956 / 388 460 | 13.9 / 11.7 |
| 48 kHz 7.1, 512k | 510 346 / 519 437 | 10.0 / 11.8 |
| 96 kHz stereo, 256k | 254 750 / 262 330 | 19.4 / 20.8 |

Also checked:

- FFmpeg decodes every file with `-v error` silent.
- The presented duration equals the input to the sample. Inputs of 1 and
  100 samples behave exactly like FFmpeg's encoder.
- Silence, full-scale noise and square waves, DC and a lone click all
  encode and decode cleanly.
- A second AAC generation lines up with the first at zero lag.

### 6.2 H.264

**Bit-exact everywhere.** Two checks, which together show the encoder's
references are what any conforming decoder reconstructs:

- **RMP's decoder vs the encoder.** The unit tests compare RMP's decoder
  output with the encoder's own reconstruction, for every picture. They
  cover sizes 32x32, 48x32, 64x48, 98x70 (cropped), 160x96 and 176x144;
  QP 0, 12, 26, 30, 45 and 51; GOPs of 2-250; and bit-rate mode.
- **FFmpeg vs RMP's decoder.** FFmpeg's decode of each encoded file is
  byte-identical to RMP's decoder. Files checked:
  - sizes 2x2, 16x16, 18x34, 352x288, 640x360, 1280x720 and 1920x1080
    (1088 coded, cropped);
  - QP 18, 23, 30 and 40;
  - 800 kb/s and 2 Mb/s bit-rate mode;
  - a 4-picture GOP.

FFmpeg reports `profile=Constrained Baseline` with the expected level
(1.0 for tiny sizes, 3.1 for 720p, 4.0 for 1080p25).

**Against libx264** (`-profile:v baseline -preset medium`, same `-b:v`):

| Clip (3 s, 30 fps) | Target | RMP rate / PSNR | x264 rate / PSNR |
| --- | --- | --- | --- |
| `testsrc2` 1280x720 | 800k | 802 602 / 35.88 dB | 775 549 / 36.78 dB |
| `testsrc2` 1280x720 | 2M | 2 011 656 / 39.66 dB | 1 987 789 / 40.98 dB |
| `mandelbrot` 640x360 | 800k | 805 352 / 34.22 dB | 879 597 / 35.85 dB |
| `mandelbrot` 640x360 | 2M | 2 018 256 / 39.10 dB | 2 348 125 / 42.20 dB |

RMP's bit rate is within 1% of the target, while x264 overshoots by up to
17% on the Mandelbrot clip. At matched rates RMP is roughly 1 dB below
x264's Baseline `medium`. §7 lists what x264 does that RMP doesn't yet.

**Tests:**

- In-crate: `aac::encoder` round trips through RMP's decoder (stereo SNR,
  bit rate, transients forcing short windows with no pre-echo, 5.1);
  quantiser, sectioning and escape bit counts against the writer; the
  MDCT against the direct formula.
- `h264::encoder::tests`:
  - bit-exact with the decoder on every size and GOP pattern above;
  - QP 0 and 51;
  - bit-rate convergence;
  - SPS/PPS parsed by the decoder's parser;
  - odd sizes rejected.
- Other H.264 unit tests: CAVLC round trip, CBP tables, transform plus
  quantise through the decoder's dequantiser, half-sample planes against
  `mc_luma`.
- `tests/codecs/encoder_test.rs`: WAV → M4A and Y4M → MP4 through
  `transcode`, read back with RMP's demuxers and decoders. Checks the
  edit-list delay, exact duration, bit rate, avcC profile, keyframe
  pattern and PSNR, and that bad input is rejected without panicking.
- `tests/cli.rs::encodes_aac_and_h264`: CLI defaults and options.

```bash
cargo test                      # 263 tests, also clean in debug builds
cargo clippy --all-targets -- -D warnings && cargo fmt --check
```

---

## 7. Known limitations (with pointers)

| Limitation | Where to start |
| --- | --- |
| AAC: no TNS, PNS, intensity stereo or KBD windows. FFmpeg's encoder uses TNS and PNS by default; RMP's SNR is still higher, but PNS and TNS mainly help perceptually at low rates. | Signal in `write_ics` (the flags are written as 0 today). The decoder's `ics.rs` already parses all of them, so its tests can validate each tool. |
| AAC: no HE-AAC (SBR/PS), no 960-sample frames, no 6.1 (7 channels needs a PCE). | SBR: a QMF bank plus envelope estimation, signalled in a FIL element. |
| H.264: no B-frames, CABAC, 8x8 transform or multiple references (Main/High profile tools). They are worth roughly 15-30% bit rate. | CABAC: invert `h264::cabac` (the context tables are shared). B-frames: a reorder queue in `H264EncoderState` and direct prediction mirrored from `inter.rs`. |
| H.264: no trellis quantisation, psy-RD, adaptive quantisation or lookahead (mb-tree). This is most of the gap to x264 in §6.2. | `transform::Quant` (trellis), `PicEncoder::new` (per-macroblock QP offsets need `mb_qp_delta`). |
| H.264 speed (about 25 fps at 720p on one core). | In profile order: SATD/SAD kernels on 8 or 16 lanes; the Intra4x4 search (9 predictions per block); slices or row threading; early termination in partition search. |
| H.264: 8-bit 4:2:0 only, progressive only, one slice per picture. | `transcode` converts other formats to `yuv420p`. |

---

## 8. How to build on this

### Add an AAC tool (e.g. TNS)

1. Run the analysis per channel after the MDCT, before `analyse` (LPC over
   the spectrum, then reflection coefficients quantised as in `decode_tns`).
2. Filter the spectrum, store the filter in `ChannelFrame`, and add its
   bits in `ics_bits`.
3. Write `tns_data_present` and the data in `write_ics`.
4. Check that RMP's and FFmpeg's decoders both decode it cleanly and that
   SNR on transients doesn't drop.

### Add an H.264 feature

- **Syntax.** Add the writer next to the reader it inverts, and test it
  by round-tripping through the decoder's parser (as `vlc.rs` does).
- **Prediction or reconstruction.** Use the decoder's `recon` functions
  rather than new code, so exactness holds by construction.
- **New macroblock state.** Fill the `MbInfo` fields the decoder fills
  (deblocking reads `kind`, `cbp`, `nnz`, `qp`, `mv`, `ref_id`).
- **Exactness.** Keep `bit_exact_with_the_decoder` passing. When it fails
  it reports the first differing pixel. The macroblock containing it, and
  whether the difference is in prediction, residual or the bottom rows
  (deblocking), usually identifies the stage.

### Rules for contributors touching these encoders

- **Bit counts.** Keep them exact: decisions compare real bit counts, and
  the AAC rate loop relies on them to never exceed a frame's budget.
- **Reconstruction.** H.264 reconstruction must go through the decoder's
  functions in the decoder's order.
- **Checks.** Validate every caller-provided parameter in `new_state` /
  `encode` and return an error. Never panic.
- **Comparisons.** Compare with FFmpeg (and x264 for H.264) as §6 does, at
  matched bit rates, before and after a change.

### FFmpeg / x264 cross-reference

| RMP | FFmpeg / x264 |
| --- | --- |
| `aac/encoder/mod.rs` | `libavcodec/aacenc.c` (`aac_encode_frame`, `apply_window_and_mdct`, element writing) |
| `aac/encoder/psy.rs` | `libavcodec/aacpsy.c` (3GPP model, `psy_lame_window`), 3GPP TS 26.403 |
| `aac/encoder/quant.rs` | `libavcodec/aaccoder.c`, `aaccoder_twoloop.h`, `aacenc_quantization.h` |
| `dsp/mdct.rs` (`Mdct`) | `libavutil/tx` (`AV_TX_FLOAT_MDCT`) |
| `h264/encoder/mod.rs` | x264 `encoder/encoder.c`, `set.c`, `ratecontrol.c`; FFmpeg `libx264.c` |
| `h264/encoder/macroblock.rs` | x264 `encoder/analyse.c`, `macroblock.c` |
| `h264/encoder/me.rs` | x264 `encoder/me.c`, `common/mc.c` (hpel planes), `common/pixel.c` |
| `h264/encoder/transform.rs` | x264 `common/dct.c`, `common/quant.c` |
| `h264/encoder/vlc.rs` | x264 `encoder/cavlc.c` |
