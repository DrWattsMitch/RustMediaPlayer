# H.264 Main/High and Opus (CELT) encoding in RMP

This document covers two encoder additions: the Main and High profile
tools of the H.264 encoder, and a CELT-mode Opus encoder. It explains the
design, how each stage maps to x264, libopus and the specifications, how
the work was verified, and what is left. Read it before changing anything
under `src/codecs/h264/encoder` or `src/codecs/opus/encoder*`.

It builds on [AAC_H264_ENCODING.md](AAC_H264_ENCODING.md) (the Constrained
Baseline encoder this extends) and [OPUS_PLC.md](OPUS_PLC.md) /
[MKV_FLAC_OPUS_PROCESSING.md](MKV_FLAC_OPUS_PROCESSING.md) (the Opus
decoder the encoder shares its band code with).

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **H.264 profiles** | `-profile:v high` (default), `main`, `baseline`. High and Main use CABAC; High adds the 8x8 transform and Intra8x8. Baseline is the previous Constrained Baseline encoder. | `h264::encoder::{Profile, Settings}` |
| **CABAC** | Bit-exact arithmetic encoder (§9.3.4) and a context-state bit-cost model with the same interface, so the bits priced in mode decisions are the bits written. | `h264/encoder/cabac.rs` |
| **Macroblock syntax** | One writer for both entropy coders, mirroring the decoder's context derivations. | `h264/encoder/syntax.rs` |
| **B pictures** | Up to `-bf` (default 3) non-reference B pictures, placed by the lookahead. B_Skip, B_Direct_16x16 (spatial direct), B 16x16 / 16x8 / 8x16 with L0, L1 or bi-prediction per partition. | `macroblock.rs` (`decide_b`) |
| **Multiple references** | `-refs` (default 3) reference pictures for P; a reference per partition (16x16 down to 8x8). | `macroblock.rs` (`search_part`) |
| **8x8 transform** | Forward 8x8 transform and quantiser, Intra8x8 (9 modes with the filtered references), inter macroblocks choose 4x4 or 8x8 by RD. | `transform.rs`, `macroblock.rs` |
| **Trellis quantisation** | RD-optimal levels per block with the CABAC rate model (x264 `--trellis 2`). | `transform.rs` (`trellis_cabac`) |
| **Lookahead** | Half-resolution analysis: x264's fast adaptive B-frame decision and scene-cut detection (new IDR). | `h264/encoder/lookahead.rs` |
| **Opus encoder** | CELT-only, fullband, mono or stereo, 2.5-20 ms frames, CBR 6-510 kb/s. Transient detection (short blocks), spreading decision, dynamic allocation, allocation trim, mid/side and intensity stereo. | `opus::OpusEncoder`, `opus/encoder.rs` |
| **Range encoder** | `ec_enc` (symbols front, raw bits back), and a `Coder` trait over encoder and decoder. | `opus/range.rs` |
| **Forward MDCT** | `clt_mdct_forward`, the inverse of the existing backward MDCT. | `opus/mdct.rs` |
| **Wiring** | `-c:a opus` (`libopus`), `-c:v libx264` alias, `-profile:v`, `-bf`, `-refs`. `.opus` and `.webm` encode Opus when they can't copy. Video encoders are flushed at the end of a transcode (the H.264 lookahead holds pictures back). | `transcode`, `core::Resolver` |

```bash
rmp -i input.mkv out.mp4 -c:v h264                     # High, CABAC, B pictures
rmp -i input.mkv out.mp4 -c:v h264 -profile:v baseline # as before
rmp -i input.mp4 out.mp4 -c:v h264 -bf 0 -refs 1       # High without B / extra refs
rmp -i song.flac song.opus -b:a 128k                   # Opus (CELT)
rmp -i talk.wav talk.webm                              # Opus at 64 kb/s per channel
```

---

## 2. H.264 in detail

### 2.1 Profiles and stream structure (`encoder/mod.rs`)

| | Baseline | Main | High |
|---|---|---|---|
| `profile_idc` / constraints | 66 / set0+set1 | 77 / set1 | 100 |
| entropy coding | CAVLC | CABAC | CABAC |
| B pictures (adaptive, up to) | — | 3 | 3 |
| reference pictures | 1 | 3 | 3 |
| 8x8 transform, Intra8x8 | — | — | yes |
| trellis | — | yes | yes |
| POC type | 2 | 0 (2 with `-bf 0`) | 0 (2 with `-bf 0`) |

`Settings::sanitised` keeps every combination legal (no B or CABAC in
Baseline, at least two DPB frames with B pictures). The SPS carries
`max_num_reorder_frames` 1 with B pictures, so decoders output after one
picture of delay. The level choice now also checks the DPB size
(`MaxDpbMbs`).

**GOP.** Pictures queue in the lookahead (`bframes + 1` of them). For the
first queued picture: an IDR at the GOP interval, at a scene cut, or at
the start. Otherwise the candidates are cut short before any forced IDR or
scene cut, `lookahead::decide` returns how many B pictures precede the
next P, and the P is coded before them. B pictures are never references
(`nal_ref_idc` 0), so reordering is one picture deep and frame numbering is
simple (`frame_num` = last reference + 1).

**Reference lists** follow 8.2.4.2 exactly (`ref_lists`): P by descending
`FrameNumWrap` (most recently coded first), B by POC around the current
picture, the L1 swap when both lists match, then truncation to the active
counts written with `num_ref_idx_active_override_flag`. The DPB is a
sliding window of `max_num_ref_frames` pictures.

**Timestamps.** With B pictures, packets carry presentation times only;
the MP4, Matroska and MPEG-TS muxers already derive decode times for such
streams (as for Matroska input). Without B pictures `dts = pts`, as before.

### 2.2 CABAC (`cabac.rs`)

`CabacWriter` is §9.3.4.2 literally: `codILow` / `codIRange`,
`bitsOutstanding`, `PutBit`, `EncodeBypass`, `EncodeTerminate` with
`EncodeFlush` (whose last bit is the `rbsp_stop_one_bit`). Contexts are
initialised exactly as the decoder does and stored in the same packed form.

`CabacCost` runs the same state machine but adds `-log2 p` per bin (1/256
bit, from `RANGE_LPS` averaged over the range quarters), like x264's
`cabac_size_decision`. Both implement `Bins`, and `syntax.rs` writes every
element generically, so mode decision, trellis and the bitstream share
one code path. Residual blocks (`residual`) invert `Cabac::residual`:
coded_block_flag, significance / last flags (8x8 context tables),
`coeff_abs_level_minus1` with its 14-bin prefix and UEG0 suffix, signs.

### 2.3 Macroblock syntax (`syntax.rs`)

`MbWriter` serialises an `MbInfo` (the decoder's per-macroblock record)
plus an `MbSyn` (mb_type, intra mode remainders, ref_idx and mvd per
partition) in either coder. Every ctxIdxInc is the decoder's (`h264::mb`)
recomputed from the same neighbours: mb_skip_flag, mb_type (I/P/B
binarisations including intra in P/B), transform_size_8x8_flag,
intra_chroma_pred_mode, ref_idx, mvd (|mvd| sums), coded_block_pattern
(with the partially built cbp of the current macroblock), coded_block_flag
for every block category, `mb_qp_delta` (always 0). The CAVLC side
includes the 8x8 transform's four interleaved 4x4 scans.

Writing happens before the macroblock is stored, with its own `MbInfo`
already final: a partition's left and upper neighbours always precede it,
so what the writer sees is what a decoder has parsed at that point.

### 2.4 Mode decision (`macroblock.rs`)

Every candidate is fully coded and priced; the choice is
`SSD + λ²·bits` (λ² = 0.85·2^((QP−12)/3)):

- **Intra:** I16x16 (mode by SATD), I4x4, and with the 8x8 transform I8x8
  (9 modes per block by 8x8 SATD + mode bits, `pred8x8l`). The predicted
  mode counts Intra8x8 neighbours as the decoder does.
- **P:** P_Skip when its residual vanishes after decimation; otherwise
  16x16 searched in every reference (cost includes `te()` ref_idx bits),
  and when that match is poor 16x8 / 8x16 / 8x8 searched in the 16x16
  winner's reference and reference 0, each partition picking its own.
- **B:** B_Skip / B_Direct_16x16 from spatial direct prediction (8.4.1.2.2
  with `direct_8x8_inference`, the colocated motion kept with each
  reference); per partition the best of L0 (every reference), L1
  (reference 0) and their average, by SATD of the actual bi-prediction;
  16x8 / 8x16 when 16x16 is poor. The B mb_type comes from the decoder's
  Table 7-14.
- **Transform size:** inter candidates with luma coefficients are coded
  with both transforms; the 8x8 one uses x264's 8x8 decimation.
- **Intra in inter pictures** when its SATD competes, as before.

Motion compensation goes through the decoder's `mc_luma` / `mc_chroma` /
`store_prediction` (default bi-prediction average), so predictions are
bit-exact. `ref_id` is a per-picture identity, which is all the
deblocking filter compares.

### 2.5 Transforms and trellis (`transform.rs`)

`fdct8` is x264's `sub8x8_dct8`; the quantiser uses `quant8_scale` per
normAdjust8x8 class with `qbits = 16 + QP/6` (tested to round-trip through
the decoder's `dequant8` + `idct8_add`).

`trellis_cabac` chooses each block's levels to minimise
`Σ(|z| − level)² + λ·bits`, where `z` is the unrounded level (the
quantiser step is uniform in the orthonormal transform domain, so this is
proportional to pixel SSD) and bits come from the current CABAC context
states: significance and last flags, level bins and signs, and the
coded_block_flag when the block has one. Each coefficient may keep its
rounded level, drop by one or become zero; the dynamic programme runs over
the eight level-context states of §9.3.3.1.3 plus "nothing coded yet", in
reverse scan order, as CABAC codes levels. λ is the RD λ² over the
squared quantiser step. Trellis applies to every luma and chroma AC block
(4x4, 8x8, Intra16x16 AC); DC blocks keep the deadzone quantiser.

### 2.6 Lookahead (`lookahead.rs`)

Each queued picture gets a half-resolution luma copy; an 8x8 lowres block
stands for a macroblock. `frame_cost(p0, p1, b)` is x264's
`slicetype_frame_cost`: per block, the cheapest of an intra estimate
(DC/H/V SATD), an integer diamond search in each reference (±16 lowres
samples, predicted from the neighbours), and the bi-predicted average.
`decide` is `--b-adapt 1`: compare P-P against B-P for the first pair,
then extend the B run while a P further out stays under x264's
`INTER_THRESH` and few blocks go intra. `is_scenecut` is
`scenecut_internal` with threshold 40 and the keyint-dependent bias.

### 2.7 Rate control

Constant QP: I = QP − 3, P = QP, B = QP + 2. Bit-rate mode keeps the
per-type complexity model and now solves for the P quantiser that makes
the recent mix of P and B pictures (B at +2) average to the per-picture
target, so B pictures don't make the encoder undershoot.

---

## 3. Opus (CELT) in detail

### 3.1 Sharing the band code with the decoder

libopus has one `quant_all_bands` / `quant_band` / `compute_theta` /
`compute_allocation` for both directions, with an `encode` flag and a
`resynth` flag. RMP's decoder had decode-only ports. They are now generic
over `range::Coder`, implemented by `RangeDecoder` (reads a symbol,
ignores the value passed) and `RangeEncoder` (writes the value passed);
`C::ENCODE` gates the encoder-only steps:

| Function | Encoder-only work |
| --- | --- |
| `compute_allocation` | band-skip decisions (`depth_threshold`), intensity band, dual-stereo flag (`AllocChoices`) |
| `compute_theta` | `stereo_itheta`, rounding to `qn` with `avoid_split_noise`, `stereo_split` / `intensity_stereo` |
| `quant_band_n1`, N = 2 stereo | sign from the signal |
| `quant_partition` | `alg_quant`: rotation, `op_pvq_search` (float build), `icwrs` |
| `quant_band` | Haar recombination / time division / Hadamard de-interleave of the signal |

The encoder always resynthesises (it runs the decoder's reconstruction),
so its state is exactly a decoder's. Coarse, fine and final energy have
separate encoder functions that mirror the decoder's arithmetic line by
line (`quant_coarse_energy` with libopus' decay bound and low-budget
clamps).

### 3.2 One frame (`encoder.rs`, `CeltEncoder::encode`)

1. Pre-emphasis (`x[n] − 0.85·x[n−1]`, ×32768) into `[history | frame]`.
2. `transient_analysis` (float, high-pass, forward/backward masking,
   threshold 200) → short blocks when there is room for the flag.
3. Forward MDCT per block (`Mdct::forward`), band amplitudes, log energies
   relative to `E_MEANS`, normalisation.
4. Flags: silence 0, no post-filter, transient, intra (first frame).
5. Coarse energy, TF (no change), `spreading_decision`,
   `dynalloc_analysis` (follower / median envelope boosts, ≤ 2/3 of the
   bits in CBR), `alloc_trim_analysis` (tilt, stereo correlation),
   intensity band by bit rate with hysteresis.
6. Allocation, fine energy, bands, anti-collapse flag, final energy bits.

Packets are `TOC + CELT frame` with configs 28-31 (CELT-only fullband,
2.5-20 ms), code 0. The delay is the MDCT overlap: `pre_skip = 120`. The
`OpusHead` extradata, packet `pts` (in 48 kHz units, counting the
pre-skip) and the shortened last `duration` let the Ogg, Matroska, WebM,
MP4 and MPEG-TS muxers write correct gapless timing.

---

## 4. Verification

### 4.1 H.264 (`h264/encoder/tests.rs`)

- **Bit exactness, every profile.** `run_with` encodes, decodes every
  access unit in decoding order with RMP's decoder (flushing at the end),
  and compares each output picture (display order) with the encoder's
  reconstruction byte for byte: Baseline, Main, High, High without
  trellis; sizes 32x32 to 200x120 including non-multiples of 16; QP 0 to
  51; GOPs 3 to 250; busy synthetic content with fades and several moving
  objects (`busy_content_stays_exact`, refs 1-4, `-bf` 2-3).
- **Tools are exercised** (`high_profile_uses_its_tools`): B pictures,
  Intra8x8, inter 8x8 transform, list 1 / bi-prediction and direct
  macroblocks all occur, and B pictures are coded after their anchors.
- **Tools pay off:** three references beat one on alternating content,
  CABAC beats CAVLC, trellis lowers the rate at equal quality.
- **GOP structure:** IDRs at the interval with B pictures, at scene cuts
  for Baseline and High.
- **Engine and parts:** CABAC bins and residual blocks round-trip through
  the decoder's engine; the 8x8 transform/quantiser round-trips through
  `dequant8` + `idct8_add`; trellis never prices worse than rounding.
- **End to end** (`tests/codecs/encoder_test.rs`): MP4 transcodes for all
  three profiles, and High with B pictures through MP4 keeping every
  frame.

Outside the test suite, streams from every profile were decoded with
FFmpeg's H.264 decoder: identical to RMP's decoder, byte for byte. On
FFmpeg's `testsrc2` (640x360, 60 frames, QP 26) High gives 266 kB at
46.3 dB PSNR against x264 High's 257 kB at 45.2 dB (x264 trades PSNR for
psycho-visual tuning) and RMP Baseline's 295 kB at 45.9 dB.

### 4.2 Opus (`opus/encoder/tests.rs`, `opus/range.rs`, `opus/mdct.rs`)

- **Final range.** Every packet's final range-coder state equals RMP's
  decoder's after it (`OPUS_GET_FINAL_RANGE`, the conformance check of
  RFC 6716 §6): mono and stereo; tones, chords, noise, clicks, chirps,
  silence; 12-510 kb/s; 2.5, 5, 10 and 20 ms frames.
- **Quality:** SNR thresholds against the delayed input, and more bits
  give at least 6 dB more.
- **Transients:** clicks select short blocks; a steady tone only at its
  onset and cut-off.
- **Encoder trait:** packetisation from arbitrary input chunks, pts,
  the shortened last packet, OpusHead, clean errors for non-48 kHz input.
- **Range encoder** round-trips every symbol type through the decoder;
  **forward MDCT** plus the decoder's backward MDCT reconstructs the
  signal (TDAC) for long and short blocks.

Outside the test suite, `.opus`, `.webm` and `.mkv` files decode with
FFmpeg's libopus to the exact input length, within 1 LSB of RMP's
decoder. Against libopus' own CBR encoder at the same rate, SNR on a
tone-plus-noise mix is on par (20.3 vs 20.2 dB at 128 kb/s); on a pure
tone libopus is ahead (43.6 vs 35.0 dB at 64 kb/s), mostly from its pitch
pre-filter.

---

## 5. Known limitations

- **H.264:** no B-pyramid or B pictures as references, no B_8x8 /
  sub-8x8 partitions, temporal direct and weighted prediction are not
  chosen (the decoder supports them), one slice per picture, constant QP
  within a picture (no adaptive quantisation), no VBV/HRD, no
  interlaced coding.
- **Opus:** CELT only (no SILK or hybrid, so speech at very low rates
  is worse than libopus), CBR only, no pitch pre-filter, no TF analysis,
  no tonality/masking analysis, no dual stereo, single-pass intra energy
  decision (intra on the first frame only).

## 6. How to build on this

**H.264.** A new mode means: a `SynType` (if its syntax is new) written in
both `MbWriter::cavlc` and `MbWriter::cabac` exactly as `h264::mb` parses
it; an `MbInfo` filled exactly as the decoder fills it (kind, cbp, nnz,
dc_cbf, ipred, ref_idx/ref_id, mv, mvd, direct flags, t8x8); prediction
through the decoder's functions; a candidate in `decide_*`. Then run the
bit-exact tests for every profile.

**Opus.** Encoder decisions go where libopus has `if (encode)`; anything
that changes what is coded must go through the `Coder` (never read or
write the range coder directly in shared code). A new analysis only
changes values passed to existing symbols; the final-range test tells you
whether the bitstream still means what the encoder thinks.

**Cross-reference.** x264: `encoder/encoder.c`, `slicetype.c`,
`analyse.c`, `macroblock.c`, `rdo.c`, `cabac.c`, `common/cabac.c`,
`common/dct.c`, `common/quant.c`. libopus: `celt/celt_encoder.c`,
`bands.c`, `vq.c`, `cwrs.c`, `rate.c`, `quant_bands.c`, `entenc.c`,
`laplace.c`, `mdct.c`.
