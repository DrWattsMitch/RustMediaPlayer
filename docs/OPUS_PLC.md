# Opus packet-loss recovery in RMP

This document covers the batch that made RMP's Opus decoder survive damaged
and incomplete streams: SILK packet-loss concealment (PLC) and comfort noise
(CNG), CELT's pitch-based concealment, SILK in-band FEC (LBRR) decoding, and
the loss detection that drives them from the `Decoder` trait. Before it, a
packet that failed to decode became 20 ms of digital silence and a decoder
reset, and a missing packet was not noticed at all. Read it before changing
`src/codecs/opus/{silk,celt,pitch,decoder}.rs`.

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Opus layer** | `opus_decode_frame` / `opus_decode_native` ported in full for missing data: concealment in the last mode, 2.5–120 ms requests split like libopus, PLC-driven mode-transition cross-fades, FEC recovery of the last lost frame. | `StreamDecoder::decode_native` |
| **Multistream API** | `conceal(samples)`, `decode_fec(next_packet, samples)`, `last_packet_duration()`. | `OpusDecoderState` |
| **Loss detection** | Through `core::Decoder`: a damaged or empty packet is concealed for its duration; a gap in packet timestamps is concealed (with FEC from the next packet) before it is decoded, and the output timeline stays continuous. | `OpusDecoderState::decode_with_recovery` |
| **SILK** | `PLC.c` (update, conceal, glue frames), `CNG.c`, the LBRR decode path of `dec_API.c`, and the two loss-dependent bits of normal decoding (LPC bandwidth expansion after a loss, the voiced→unvoiced LTP bridge in `decode_core`). | `silk::{plc, cng, plc_glue_frames}`, `SilkMode` |
| **CELT** | `celt_decode_lost` complete: pitch search, LPC excitation extrapolation with decay detection, energy-explosion guard, pre-filtered TDAC overlap; noise PLC/CNG unchanged; `skip_plc` and `last_pitch_index` state. | `CeltDecoder::decode_lost` |
| **Pitch / LPC** | Float `pitch_downsample`, `pitch_search`, `_celt_autocorr`, `_celt_lpc`, `celt_fir`, `celt_iir`. | `codecs::opus::pitch` (crate-private) |

## 2. How concealment behaves

What a listener hears for a loss, following libopus exactly:

- **SILK (speech).** The first lost frame repeats the last pitch period
  through the last LPC filter (voiced) or plays the last frame's quieter
  excitation as noise (unvoiced), with gains that decay each subframe
  (harmonics ×0.99 then ×0.95, noise ×0.95/0.99 then ×0.8/0.9) while the
  pitch period slowly lengthens. Comfort noise, shaped by NLSFs and gain
  smoothed over inactive frames, is added on top, so a long loss settles
  at the stream's own background level. The first good frame is faded in
  when it is louder than the concealment, and decoded with bandwidth-
  expanded LPC filters.
- **CELT (music, high-rate speech).** After two good frames in a row, the
  last pitch period (100–720 samples, searched at 1/4 then 1/2 rate) of the
  LPC excitation is repeated, scaled by the decay the signal already had
  and ×0.8 per further loss, re-synthesised, and muted if it comes out
  louder than the original. Its overlap is pre-filtered and windowed so the
  next decoded frame blends in by TDAC. After 100 ms of loss, right after a
  loss or reset, and in hybrid mode, band energies drop 1.5 dB then 0.5 dB
  per frame towards the background estimate over random spectra: comfort
  noise.
- **Hybrid.** SILK concealment below 8 kHz plus CELT noise above.
- **FEC.** When the packet after a loss carries LBRR data (the encoder ran
  with `-fec 1` and a non-zero `-packet_loss`), the last lost SILK frame is
  decoded from it instead of being concealed; earlier frames of a longer
  gap are concealed. CELT-only packets carry no FEC.

## 3. Detecting losses

The `Decoder` implementation cannot be told about a loss, so it infers one
in `decode_with_recovery`:

1. **Damaged packets.** If a packet fails to parse or decode (including an
   empty one), its nominal duration is concealed: from the TOC when it is
   readable, else the container's duration, else the last packet's. The
   decoder is no longer reset, so concealment continues from the real
   history. `concealed_frames` counts these.
2. **Timestamp gaps.** The decoder remembers where the next packet should
   start (`pts + samples`, in 1/48000). A later pts is a gap when it is at
   least half the last packet's duration (smaller differences are muxer
   jitter: Matroska stores milliseconds) and at most `max_gap` (2 s by
   default; longer jumps are seeks or splices and are passed through). The
   gap is rounded to 2.5 ms and recovered with `decode_fec` (or `conceal`
   when `use_fec` is off), then the packet is decoded; the frame's pts is
   the start of the gap. `concealed_samples` counts all concealed audio.

This catches what the demuxers already do with damage: Ogg drops pages that
fail their CRC and resyncs, Matroska resyncs to the next cluster, and both
keep timestamps that reveal the hole. Set `conceal_gaps = false` on the
state to get the old behaviour (FFmpeg's, which never fills gaps).

Callers that know about losses themselves (RTP, a jitter buffer) use the
API directly:

```rust
let mut dec = OpusDecoderState::from_extradata(&extradata, 2);
let pcm = dec.decode_packet(&packet)?;             // normal decode
let pcm = dec.conceal(960)?;                       // 20 ms lost, no FEC
let pcm = dec.decode_fec(&next_packet, 960)?;      // 20 ms lost, FEC from next
let pcm = dec.decode_packet(&next_packet)?;        // then decode it as usual
```

Durations are samples per channel at 48 kHz, multiples of 120 (2.5 ms),
any length (split into ≤ 120 ms calls as libopus requires).

## 4. Project values

**Safety.** No `unsafe`, no panics on any input: concealment only indexes
buffers whose bounds follow from the frame sizes, and the two places
libopus merely asserts (the SILK PLC lag fitting the LTP history, a CELT
pitch index that was never searched) are clamped. The SILK port keeps
libopus' fixed-point wrapping/saturation explicitly (`wrapping_*`,
`saturating_*`, the `SAT16`/`LSHIFT_SAT32` helpers), so debug and release
builds agree. The CELT NaN guard (`!(S1 > 0.2·S2)`) is kept as a
`partial_cmp` so a NaN still mutes.

**Efficiency.** Nothing runs for good packets beyond what libopus does
(PLC/CNG state updates are a few hundred operations per SILK frame). The
CELT pitch search runs once per loss, not per concealed frame.

**Readability.** libopus' monolithic `opus_decode_frame` keeps its shape,
so the port can be diffed against it; concealment lives in named functions
(`plc_update`, `plc_conceal`, `plc_glue_frames`, `cng`,
`conceal_with_pitch`, `conceal_with_noise`) with libopus' names in their
docs.

**Licensing.** All of it is a port of libopus 1.4 (BSD-3-Clause), as the
rest of the Opus decoder.

## 5. Verification

References are libopus 1.4 itself, through `tools/opus_plc_ref.c` (a
multistream decode that drops packets and conceals each gap the way RMP
does) and `tools/opus_packets.py` (dumps a stream's packets via ffprobe).
The test drops 15 packets per stream: singles, a pair, three in a row and a
160 ms burst.

| Stream | Result vs libopus |
| --- | --- |
| SILK WB voiced/silence, PLC + CNG (`plc_silk_fec`) | **bit-exact** |
| SILK WB, FEC (`plc_silk_fec`) | **bit-exact**; 1.7 dB closer to the loss-free decode than PLC alone |
| SILK WB stereo 40 ms (`silk_wb_st_40ms`) | **bit-exact** |
| Hybrid stereo, PLC and FEC (`hybrid_st`, `plc_hybrid_fec_st`) | ≤ 1 step (100–109 dB) |
| CELT 5.1 sine, mode switches (`celt_51`, `mode_switch`) | ≤ 1 step (≥ 108 dB) |
| CELT mono 64 kbit/s, pitch PLC (`plc_celt`) | ≤ 10 steps, 74.5 dB |

The pitch-PLC figure is float rounding amplified by the 24th-order IIR
over the extrapolation: libopus' own SIMD and scalar builds are 13 steps /
74.2 dB apart on the same stream. Further tests check that timestamp gaps
through the `Decoder` trait produce exactly the explicit-API output with a
continuous timeline, that damaged and empty packets are concealed (not
silent, right duration), that a 1 s loss fades to comfort noise at about
the background level and decoding resumes, and the edge cases (nothing
decoded yet → silence, non-2.5 ms requests rejected, long jumps passed
through). The corruption fuzzer now also mixes random losses, FEC calls,
truncated packets and timestamp jumps into real streams.

```bash
cargo test --test codecs opus            # decoder + packet-loss tests
RMP_FUZZ_ROUNDS=3000 cargo test --test container ogg_demuxer_and_opus
LIBOPUS=path/to/opus-1.4 tools/make_fixtures.sh   # regenerate references
```

## 6. Known limitations

| Limitation | Where to start |
| --- | --- |
| No deep PLC / DRED (libopus 1.5's neural concealment and redundancy). | libopus 1.5 `dnn/`, `opus_decoder.c` `OPUS_SET_DRED_DURATION`. |
| Loss is inferred only from undecodable packets and timestamp gaps; a stream with broken timestamps (e.g. muxer jitter of more than half a packet) can hide a loss or show a false one. | `decode_with_recovery`; set `conceal_gaps = false` for such sources. |
| Multistream packets that fail in one stream after earlier streams decoded leave those streams one frame ahead before concealment. | `OpusDecoderState::run`. |
| No soft clipping of concealed float output (libopus applies it only to 16-bit output). | `opus_pcm_soft_clip`. |
