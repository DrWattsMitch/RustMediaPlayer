# Fragmented MP4, `+faststart` and dithering in RMP

This document covers three additions: **fragmented MP4** output and input
(the format MSE players, HLS and DASH use), **`+faststart`** (moov in front
of mdat, so a progressive download starts playing at once), and
**dithering** when audio is rounded to integer samples. It explains the
design, how each part maps to FFmpeg, how it was verified and how to extend
it. Read it before changing `src/container/mp4/{muxer,fragment,options}.rs`,
the fragment code in `demuxer.rs`, or `src/process/{audio,dither}.rs`.

It builds on [MP3_MP4.md](MP3_MP4.md) (the MP4 container) and
[MKV_FLAC_OPUS_PROCESSING.md](MKV_FLAC_OPUS_PROCESSING.md) (the audio
converter).

---

## 1. What was added

| Area | Capability | Entry point |
| --- | --- | --- |
| **Fragmented MP4 muxing** | `ftyp(iso5) · moov(empty, mvex) · (moof mdat)* · mfra`, as FFmpeg writes with `-movflags frag_keyframe+empty_moov+default_base_moof`. Cuts at video keyframes, every N µs, every packet, or with a minimum length; one `moof` per fragment or per track. Edit lists for B-frame delay and AAC priming. Works on non-seekable outputs (pipes). | `container::mp4::muxer::Mp4Muxer`, `container::mp4::fragment` |
| **Fragmented MP4 demuxing** | Reads `moof/traf/tfhd/tfdt/trun` (every flag combination, `trex` defaults, explicit or moof-relative base offsets) into the same flat sample index as `stbl`, so reading, seeking and duration work unchanged. A file cut off mid-fragment plays up to the cut. | `container::mp4::demuxer::Mp4Demuxer`, `fragment::parse_moof` |
| **`+faststart`** | `moov` moved in front of `mdat` when the file is finished: an in-place, back-to-front move through a 1 MiB buffer, with every chunk offset raised by the size of `moov` (iterated until `stco`/`co64` settles). | `Mp4Muxer::finish` |
| **Muxer options** | `Muxer::set_option(key, value)`, with FFmpeg's names: `movflags`, `frag_duration`, `min_frag_duration`. `Output` holds them until it creates the muxer. | `core::traits::Muxer`, `container::mp4::options` |
| **Dithering** | `rectangular`, `triangular` (TPDF), `triangular_hp`, and `lipshitz` noise shaping, at any precision from 2 to 32 bits (24-bit FLAC / `pcm_s24le` dither at the 24th bit). Applied only when precision is lost; deterministic. | `process::dither`, `AudioConverter::dithered` |
| **CLI** | `-movflags`, `-frag_duration`, `-min_frag_duration`, `-dither_method`, `-dither_scale`. The transcode pipeline dithers with TPDF by default. | `transcode::Options`, `rmp --help` |

```bash
rmp -i in.mkv -c copy -movflags +faststart web.mp4                      # progressive download
rmp -i in.mkv -c copy -movflags frag_keyframe+empty_moov+default_base_moof live.mp4   # MSE
rmp -i in.mp4 -c copy -frag_duration 2000000 segments.mp4               # 2 s fragments
rmp -i song.opus song.wav                                               # TPDF-dithered 16-bit
rmp -i song.opus -dither_method lipshitz song.wav                       # noise-shaped
rmp -i song.opus -dither_method none song.wav                           # FFmpeg's plain rounding
```

---

## 2. How the project values were applied

**1. Safety**

- No `unsafe`, and no `unwrap`, `expect` or `panic!` outside tests. Every
  index into a sample list is guarded by an emptiness check, and timestamp
  arithmetic on values read from files saturates (it passes in debug
  builds, where overflow panics).
- **Reading fragments:** every `trun` sample count is checked against the
  bytes of the box before anything is allocated. `moof` boxes over 64 MiB
  are refused. The per-track sample cap (`MAX_SAMPLES`) covers `stbl` and
  fragment samples together. Samples that would lie past the end of the
  file are dropped. A malformed box stops the scan quietly, and what was
  read so far stays playable.
- **Writing fragments:** a fragment is cut when it buffers 64 MiB, whatever
  the rules say, so memory stays bounded even if a stream has no keyframes.
  `trun` data offsets are checked to fit `i32`, and an `mdat` over 4 GiB
  switches to a 64-bit size.
- **`+faststart`** first checks that the output can be read back. If it
  can't, the muxer writes a normal file (moov at the end) and returns an
  error saying so, rather than leaving a half-moved `mdat`. `File::create`
  now opens files read+write (write-only if the system refuses), which is
  what makes the move possible.
- **Options** are validated when they are given (`rmp` fails before
  touching the input on a typo), and refused once the first packet is
  written, because they decide the file layout.
- **Dither** can't make the noise-shaping loop unstable: the error fed back
  is measured before clipping, so it stays within ±1.5 LSB whatever the
  signal does. NaN input becomes 0.

**2. Efficiency**

- Fragmented output never seeks. It holds one fragment in memory and
  writes each sample once. `mehd` (total duration) is patched at the end
  only when the output can seek.
- The `+faststart` move is O(file size) with a fixed 1 MiB buffer, the same
  approach as FFmpeg's `shift_data`. Remuxing a 112 MB, 60 s 720p file
  (release build, warm page cache) takes 0.07 s plain, 0.12 s with
  `+faststart` and 0.12 s fragmented.
- Values every sample in a track fragment shares go into `tfhd` once
  (duration, size, flags); usually only the first sample's flags differ,
  and `trun`'s `first_sample_flags` covers that. An audio fragment then
  costs 4 bytes per sample, as in FFmpeg's output.
- Reading fragments loads only `moof` boxes and seeks over `mdat`.
- Dither costs a few multiply-adds per sample. Lossless conversions
  (integer → wider integer) skip it entirely.

**3. Readability**

- `fragment.rs` is pure serialisation and parsing (no I/O), so its unit
  tests check box layouts byte for byte. `muxer.rs` decides *when* to cut
  and keeps the per-track state. `options.rs` is just the option grammar.
- The three layouts share the timing code in `Mp4Muxer::write` and the
  `moov` writer, which takes a `Layout`.
- `dither.rs` documents each method's noise in a table.

---

## 3. Fragmented MP4

### 3.1 Layout

```text
ftyp       major iso5, compatible iso5 iso6 mp41   (FFmpeg's brands for this mode)
moov       mvhd/tkhd/mdhd durations 0, empty stts/stsc/stsz/stco
  trak     + edts/elst (segment duration 0 = "whole track") when needed
  mvex     mehd (v1, patched at the end) + trex per track
  udta     metadata
moof       mfhd sequence number
  traf     tfhd: default-base-is-moof + shared duration/size/flags
           tfdt: v1 base media decode time
           trun: data offset, first-sample flags, per-sample fields
mdat       track 1's samples, then track 2's, ...
...
mfra       tfra per track (fragments that start with a sync sample) + mfro
```

`separate_moof` writes one `moof`/`mdat` pair per track per fragment.

### 3.2 When a fragment is cut

`Mp4Muxer::starts_fragment` runs before a packet is added. A new fragment
starts when the current one holds at least one sample and lasts at least
`-min_frag_duration`, and:

1. `frag_every_frame`: always; or
2. `frag_keyframe` (also the default when no rule is given): the packet is a
   video keyframe and its track already has samples in this fragment (as
   FFmpeg checks `trk->entry`); or
3. `-frag_duration`: the fragment has reached that length. With no video
   track and no `-frag_duration`, fragments are 1 s long, so audio-only
   output is still fragmented (FFmpeg would write a single fragment).

The 64 MiB cap overrides all of these.

### 3.3 The init segment is written with the first fragment

RMP writes `moov` when the first fragment is flushed, not before the first
packet. This is FFmpeg's `delay_moov`, and it is the default here because it
lets `moov` describe the first samples:

- **Composition offsets.** If the first fragment has B-frames presented
  before they are decoded, the track gets a constant delay added to every
  composition offset (`cts_shift`) so `trun` keeps them non-negative, and an
  edit list removes the delay from presentation. Without this, FFmpeg's
  demuxer places pictures a frame late when it reads negative `trun`
  offsets (it reports `start_time = 0.04` where the source has `0`). Later
  fragments that need more delay than the first fall back to version 1
  `trun` with negative offsets.
- **AAC priming.** The encoder delay becomes `elst` media time, as in
  progressive files.

Edit list segment durations are 0, which means "to the end of the track",
the same as FFmpeg writes with `delay_moov`. ffprobe reads both RMP's and
FFmpeg's files with `start_time = 0` and the same stream durations.

### 3.4 Timestamps

Fragment decode times (`tfdt`) are relative to each track's first decode
time, which is what the sample tables of a progressive file also assume. The
last sample of a fragment gets its packet's duration (or the previous
delta); the next fragment's `tfdt` carries the exact time, so an estimate
never accumulates.

**Packets without dts** (Matroska): decode times are the sorted presentation
times *within each fragment*. That is only right when fragments hold whole
GOPs, so with such a track fragments are cut only at its keyframes, and a
frame reordered across a cut is an error rather than wrong timing.

### 3.5 Reading fragments (`demuxer.rs`)

When `moov` has `mvex`, `Mp4Demuxer::new` keeps tracks with empty sample
tables, then scans the top-level boxes after `moov`, reads each `moof`, and
`fragment::parse_moof` appends samples:

- base offset: `tfhd` base-data-offset, else the `moof` start
  (default-base-is-moof), else the end of the previous `traf`'s data;
- data start: the `trun` data offset, else where the previous `trun` ended;
- duration / size / flags: `trun`, else `tfhd`, else `trex`;
  `first_sample_flags` overrides the first sample's; sync = flag bit 16 clear;
- decode time: `tfdt`, else continuing from the previous fragment.

The result is the same `Vec<Sample>` the progressive path builds, so `read`
and `seek` didn't change. Tracks get `has_sync_table` when any sample is
non-sync, so seeking snaps to keyframes. Duration comes from `mehd`, else
from the samples. `mfra` isn't needed (every `moof` is indexed).

---

## 4. `+faststart`

`finish` first writes the `mdat` size as usual. Then, if the output can be
read back:

1. `build_faststart_moov` serialises `moov` with every chunk offset raised
   by its own size. If that pushes an offset past 4 GiB, `stco` becomes
   `co64` and `moov` grows, so it repeats until the size is stable (two
   passes in practice, at most four).
2. `shift_mdat` copies `[mdat_start, end)` forward by that size, one 1 MiB
   block at a time **from the end backwards**. Each block is read before any
   write can reach it, so no temporary file is needed.
3. `moov` is written where `mdat` used to start.

The file has exactly the same size as without the flag. A fragmented file
already has `moov` first, so `+faststart` changes nothing there (FFmpeg
ignores it in that case too).

---

## 5. Dithering (`process/dither.rs`)

### 5.1 When it applies

`AudioConverter::dithered(dither, bits)` turns on a `Quantizer` only when the
conversion loses precision:

| Conversion | Dithered? |
| --- | --- |
| float → integer | yes |
| integer → narrower integer, or a 24-bit target in `S32` | yes |
| any remix or resample into an integer format | yes |
| integer → same or wider integer | no (exact) |
| anything → float, anything → `S64` | no |

`bits` is the precision the encoder keeps: 24 for `pcm_s24le`/`be` and
24-bit FLAC (both are carried in `S32` and shifted right by 8), otherwise
the format's width. The quantiser rounds to that many bits and leaves the
low bits zero, so the encoder's shift is exact and the dither sits at the
real LSB.

`AudioConverter::new` alone never dithers, so the library default stays
bit-exact with `swr_convert` (the process tests compare against FFmpeg).
The **transcode pipeline** (`rmp`, `transcode::Options`) uses triangular by
default; `-dither_method none` gives FFmpeg's output.

### 5.2 Methods

Values are in output LSBs. `u` is FFmpeg's LCG (`seed·1664525 + 1013904223`)
mapped to [-0.5, 0.5), seeded differently per channel so channels are
uncorrelated and every run gives the same bytes.

| Method | Added noise | Spectrum |
| --- | --- | --- |
| `rectangular` | `u` | flat |
| `triangular` | `u₁ + u₂` (TPDF) | flat; error power and signal are statistically independent |
| `triangular_hp` | `uₙ − uₙ₋₁` | TPDF, rising 6 dB/oct |
| `lipshitz` | TPDF, plus error feedback `v = x − Σ cₖ eₙ₋ₖ` | `1 − Σ cₖ z⁻ᵏ` |

`-dither_scale` multiplies the noise (0 to 8; 0 means plain rounding).

`lipshitz` uses the five-tap E-weighted filter from Lipshitz, Vanderkooy and
Wannamaker (1991), `c = [2.033, −2.165, 1.959, −1.590, 0.6149]`, designed for
44.1 kHz. Its noise transfer function, measured in the RMP implementation on
a 0.3 FS sine at 44.1 kHz (relative to flat TPDF):

| Band | Noise vs TPDF |
| --- | --- |
| 0-1 kHz | −17 dB |
| 2-5 kHz | −19 dB (−24 dB around 4 kHz) |
| 16-22 kHz | +17 dB |

It is accepted at 44.1 and 48 kHz (at 48 kHz the curve moves 9% higher in
frequency). Other rates are refused, as FFmpeg refuses noise shaping at
rates its tables don't cover.

FFmpeg's other shapers (`shibata`, `low_shibata`, `high_shibata`,
`f_weighted`, `e_weighted`, `modified_e_weighted`) need FFmpeg's coefficient
tables and are not included. They are refused with the list of supported
names.

### 5.3 What dithering buys

`dither_removes_quantisation_distortion` encodes a 1 kHz tone 1.5 LSB high.
Plain rounding turns it into a staircase with a strong 3 kHz harmonic. TPDF
keeps the tone at its true level and lowers that harmonic by more than
20 dB. `tpdf_makes_the_error_independent_of_the_signal` shows a 0.4 LSB sine
that rounding deletes entirely, but that survives (gain ≈ 1) with dither.

---

## 6. Verification

| Check | How |
| --- | --- |
| Fragmented and `+faststart` remuxes decode **frame for frame identically** to the source in FFmpeg | `ffmpeg -f framemd5` on source, RMP progressive, RMP fragmented, RMP faststart: same hashes and timestamps. |
| Fragment structure matches FFmpeg | Box-by-box comparison with `ffmpeg -movflags frag_keyframe+empty_moov+default_base_moof(+delay_moov)`: same brands, `tfhd` flags (`0x020028`/`0x020038`), `trun` flags (`0xa05`/`0x301`), `tfdt` v1, `tfra` v1 and `mfro`. |
| ffprobe timing | RMP's fragmented output: `start_time 0`, video 3.000 s, audio 3.021 s; the same as FFmpeg's `delay_moov` output. |
| Packets round-trip | `tests/container/mp4/layout_test.rs`: data, keyframes and pts identical to the progressive layout for faststart, keyframe, every-frame, duration-based and separate-moof fragments. |
| FFmpeg's fragmented files | `h264_aac.ffmpeg_frag.mp4` (explicit base data offsets, no edit lists) reads with the same packets as its source. |
| Streaming | Fragmented output to a write-only, non-seekable `Io` is byte-identical apart from `mehd`. Progressive output to it fails cleanly. |
| Partial files | A fragmented file cut in its last `mdat` plays every complete sample. |
| Matroska input | `h264_aac.mkv` (pts only) → fragmented MP4: same packets and relative pts. |
| Robustness | `mp4_demuxer_survives_corruption` now also mutates a fragmented file. |
| Dither | `process/dither.rs` and `process/audio.rs` unit tests (bounds, bias, signal independence, harmonic removal, noise-shaping spectrum, stability under clipping, 24-bit alignment, determinism); `tests/cli.rs` checks the CLI default, determinism, `none`, and errors. |
| CLI | `movflags_select_the_mp4_layout`: box order for `+faststart` and fragmented output; typos and non-MP4 outputs refused. |

---

## 7. Known limitations

| Limitation | Where to start |
| --- | --- |
| No `sidx` / `global_sidx`, `dash`, `cmaf`, `isml` or `frag_custom` modes; no segment files (HLS/DASH packagers split the single fragmented file). | `fragment.rs` (a `sidx` before each `moof`), `muxer.rs::write_fragment`. |
| Without `empty_moov`, FFmpeg puts the first fragment's samples in `moov`; RMP always writes an empty `moov`. | `Mp4Muxer::write_init`. |
| `-frag_size` (cut by bytes) isn't an option; only the fixed 64 MiB cap. | `starts_fragment`. |
| Fragments are only read after `moov` (the normal layout); `moof` boxes before it are skipped. | `Mp4Demuxer::read_fragments`. |
| The AAC and MP3 decoders produce `S16` themselves (to match FFmpeg's `s16` references), so `aac → wav` isn't dithered unless it is resampled or remixed. Opus, Vorbis, float PCM, 24/32-bit sources and every remix/resample are. | Have those decoders output `F32` and let the converter round. |
| FFmpeg's table-based noise shapers (`shibata`, `f_weighted`, ...) aren't included. | Add coefficient sets to `dither.rs` next to `LIPSHITZ`, keyed by rate. |
| `+faststart` needs an output that can be read back; on a pipe it fails (with a playable file left behind on seekable outputs). | Write to a temporary file and copy, as FFmpeg can't do better on a pipe either. |

---

## 8. How to build on this

**Add a fragment rule.** Extend `MovFlags` (or add an option to
`Mp4Options::set`), then decide in `Mp4Muxer::starts_fragment`. Keep the
`pending_samples == 0` and 64 MiB checks first.

**Write `sidx` for DASH.** After a fragment is serialised its size is known
(`fragment_header` + data sizes), so a `sidx` per fragment can be written
before the `moof`, or a global one patched at the end like `mehd`.

**Add a dither method.** Add a `DitherMethod` variant, its name in
`from_name`/`name`/`DITHER_METHODS`, and its noise in
`Quantizer::quantize`. A noise shaper only needs a coefficient array:
`errors` already keeps the last five errors, newest first.

### FFmpeg cross-reference

| RMP | FFmpeg |
| --- | --- |
| `mp4/options.rs` | `libavformat/movenc.c` `options[]` (`movflags`, `frag_duration`, `min_frag_duration`) |
| `mp4/fragment.rs` (writing) | `mov_write_moof_tag`, `mov_write_traf_tag`, `mov_write_tfhd_tag`, `mov_write_tfdt_tag`, `mov_write_trun_tag`, `mov_write_mfra_tag`, `mov_write_mvex_tag` |
| `Mp4Muxer::starts_fragment` | `mov_write_single_packet` / `mov_flush_fragment` |
| `Mp4Muxer::write_init` | `delay_moov` handling in `mov_flush_fragment` |
| `Mp4Muxer::shift_mdat`, `build_faststart_moov` | `shift_data`, `compute_moov_size` |
| `mp4/fragment.rs::parse_moof` | `libavformat/mov.c` `mov_read_moof`, `mov_read_tfhd`, `mov_read_tfdt`, `mov_read_trun`, `mov_read_trex` |
| `process/dither.rs` | `libswresample/dither.c` (`swri_get_dither`, noise-shaping loop), `-dither_method`, `-dither_scale` |
