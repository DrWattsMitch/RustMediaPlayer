# H.264: interlaced (PAFF/MBAFF), High 10 and High 4:2:2

The H.264 decoder now accepts three kinds of stream that `params.rs` used to
reject:

- **interlaced** streams (`frame_mbs_only_flag = 0`), with field pictures
  (PAFF), MBAFF frames, or both mixed in one stream;
- **High 10**, at any bit depth from 9 to 14 (the luma and chroma depths
  must be equal);
- **High 4:2:2** chroma (`chroma_format_idc = 2`).

All three combine freely, so 4:2:2 10-bit MBAFF works. Every fixture matches
FFmpeg bit for bit with 1, 3 and 8 threads, and damaged streams conceal the
same way at every thread count. Everything is safe Rust; no `unsafe` was
added.

Still rejected: 4:4:4, monochrome, luma and chroma bit depths that differ,
lossless (`qpprime_y_zero_transform_bypass`), FMO/ASO, SP/SI slices and data
partitioning.

## Output

`DecodedPicture` gains `bit_depth` and `chroma_422`. Above 8 bits, the planes
hold little-endian `u16` samples (FFmpeg's `yuv420p10le` / `yuv422p10le`
layout), and `pixel()` reads one sample at either depth. 4:2:2 chroma planes
are full height. Interlaced content comes out as whole interleaved frames,
like FFmpeg without a deinterlacer. A field with no partner is output with
the missing field concealed from the field that is present.

## Sample type: `Pixel`

Everything that touches samples is now generic over `T: Pixel` (`u8` or
`u16`). This covers prediction, transforms, MC, deblocking and the frame
pools in `recon.rs`, `deblock.rs` and `decode.rs`.

- `decoder.rs` holds an `Engines` enum: `Narrow(Engine<u8>)` or
  `Wide(Engine<u16>)`. It switches when an SPS changes the bit depth.
  The state that does not depend on the sample type lives in `Common`.
- The 8-bit SIMD kernels stay. `Pixel` has narrowing hooks (`narrow_dst`,
  `narrow_src`, …). They return `Some` only for `u8`, so the vector paths
  are chosen at compile time and cost nothing. High bit depth uses the
  scalar paths for now.
- `Src`/`Dst` are strided views with a `max` sample value. A field is the
  same frame buffer seen with a doubled stride and a one-row offset. This
  lets PAFF fields and MBAFF field macroblocks reuse every kernel without
  copying.

8-bit progressive speed is unchanged: 720p takes 4.9 s here, against 4.9 s
before (noise between runs is ±0.2 s).

## High bit depth

The parts of the spec that change with bit depth are all implemented as
written: QpBdOffset, the chroma QP table, weighted-prediction offsets,
deblocking thresholds, PCM samples, and the coefficient and level clamps.

| Item | Rule | Code |
| --- | --- | --- |
| QP | QP<sub>Y</sub> wraps in −QpBdOffset..51; dequant uses QP′ = QP + QpBdOffset | `decode.rs`, `slice.rs`, `params.rs` (PPS `init_qp`) |
| Chroma QP | table index clamped to −QpBdOffset..51 | `consts::chroma_qp` |
| Weighted prediction | offsets ≪ (bd − 8) | `inter.rs` |
| Deblocking | α, β, t<sub>C0</sub> scaled by 2<sup>bd−8</sup> | `deblock.rs` |
| I_PCM | `bd` bits per sample | `decode.rs` |
| Clamps | coefficients ±2<sup>7+bd</sup>; CAVLC/CABAC levels 2<sup>22</sup> | `recon.rs`, `cavlc.rs`, `cabac.rs` |

## 4:2:2

- Chroma DC is a 2×4 block: scan `[0,2,1,4,6,3,5,7]`, a 2×4 Hadamard, and
  dequantisation with QP′<sub>C</sub> + 3 (`recon::chroma422_dc_dequant`).
- CAVLC uses the nC = −2 coeff_token table and the 4:2:2 DC total_zeros
  tables (`tables.rs`). CABAC DC significance uses ctxIdxInc = min(i/2, 2).
- Each MB has eight 4×4 chroma blocks per plane. `MbInfo::nnz` is `[u8; 32]`:
  16 luma, then 8 Cb, then 8 Cr. `nb_chroma` addresses a 2×4 grid.
- 8×16 chroma intra prediction, using the 4:2:2 plane-mode coefficients.
- Vertical chroma MVs are in quarter-sample units, not eighth-sample units.
- Deblocking filters four horizontal chroma edges per MB. The inner edges
  are filtered even when the 8×8 transform is used.

## Interlaced: PAFF

- **Slice header.** `field_pic`, `bottom_field`, `parity()`, and
  CurrPicNum = 2·frame_num + 1.
- **POC.** `PocState::compute` gives each field its own count, for all
  three POC types.
- **DPB** (`dpb.rs`). An `Entry` carries:
  - per-field marking `[Marking; 2]`;
  - per-field POC (`i32::MAX` for a field that is absent);
  - a `complete` flag.
- **Field reference lists** (8.2.4.2.5). Frames are ordered as for frame
  lists. The list then alternates parity, starting with the current
  field's parity, and skips fields that are missing. P fields order by
  FrameNumWrap; B fields order by POC, with entry POC ≤ current going to
  list 0, as FFmpeg does. List modification works on picture numbers
  split by `split_pic_num`. MMCOs mark single fields. The second field of
  a reference pair skips the sliding window.
- **Pairing** (`decoder.rs`). Two fields form one picture when they have
  opposite parity, the same frame_num and the same SPS, and the second
  field is not an IDR. Both fields decode into one `PictureJob`: the first
  field is `Pending`, then each field adds a `Part`. The second field can
  reference the first, which is still being written; it reads that field
  through `cur_id`.
- **Field decoding.** Field MB row r is stored at frame row 2r + parity.
  Fields use the field scans and the CABAC field contexts (277, 338, 436,
  451 and the field 8×8 significance map). In 4:2:0, chroma MVs are offset
  by 2·(current parity − reference parity). Deblocking runs per field,
  with a vertical MV limit of 2.
- **Direct prediction.** `colocated()` implements the three cases of
  Table 8-6:
  - frame to frame, or field to field;
  - a field reading a frame (FrmToFld);
  - a frame reading a field pair (FldToFrm). `col_parity` picks the field
    whose POC is closer.

  Temporal direct maps the colocated reference to the current list. It
  halves or doubles the vertical MV when frame and field coding differ.

## Interlaced: MBAFF

- **Pair addressing.** Macroblocks come in vertical pairs: first_mb counts
  pairs. The field flag is either read or inferred: from the left pair,
  else from the pair above, else frame. When the top MB is skipped, the
  flag is read early. In CABAC this also reads the bottom MB's skip flag
  ahead of time, as FFmpeg does.
- **Neighbours.** `mb::locate` implements Table 6-4 as a mapping from a
  neighbouring sample to the physical MB and 4×4 row. `nb_scale` adjusts
  a neighbour's MV and ref_idx when its coding (frame or field) differs
  from the current MB's: ref × 2 with MV ÷ 2, or the reverse. This also
  applies to the CABAC mvd and ref_idx contexts. The D neighbour is
  located at sample row y − 1 (`nb4_row`, `mv_nb_row`).
- **References.** Each slice derives per-parity field lists: entry 2i is the
  same-parity field of frame entry i, and entry 2i + 1 is the opposite
  field. Explicit weights are indexed by refIdx ≫ 1. `cur_poc()` returns
  the current field's POC for implicit weights and temporal direct.
- **Deblocking** (`deblock_mbaff`):
  - **Mixed left edge** (frame MB next to field MB): bS and QP are worked
    out per row.
  - **Chroma rows** use the matching luma row's bS: c in 4:2:2, 2c in a
    field MB, 4(c/2) + c mod 2 in a frame MB.
  - **Frame MB under a field pair**: its top edge is filtered twice, once
    against each field.
  - **Horizontal edges between frame and field MBs** get bS ≥ 1. Intra
    MB edges in field MBs use bS 3.
- **Threading.** MBAFF frames publish progress in pair rows, with wider
  margins (6 rows for luma, 3 for chroma). Frame threading therefore still
  overlaps pictures: 1080i decodes in 2.6 s with 2 threads, against 2.7 s
  for the same content as 1080p.

## Reference identity

The colocated and deblocking comparisons need to know whether two partitions
use the same picture. A reference is identified by the key
`(id << 2) | (1 + parity, or 3 for a frame)`, so the two fields of one frame
compare as different pictures. `next_picture_id` keeps picture ids below
2<sup>30</sup>, so the key cannot overflow.

## Tests and fixtures

`tests/fixtures/h264/` has 7 new streams (about 83 KB together).
`tests/codecs/h264_decoder_test.rs` compares an FNV-1a hash of each decoded
stream with FFmpeg's:

| Fixture | Covers |
| --- | --- |
| `h264_high10.264` | High 10, CABAC, B-pyramid, weighted P |
| `h264_high422_10.264` | 4:2:2 10-bit CABAC |
| `h264_high422_cavlc.264` | 4:2:2 8-bit CAVLC |
| `h264_mbaff.264` | MBAFF, CABAC, B-pyramid, weighted P |
| `h264_mbaff_422_10_cavlc.264` | MBAFF 4:2:2 10-bit, CAVLC, temporal direct |
| `h264_paff_bff.264` | field pictures, bottom field first, temporal direct |
| `h264_paff_mbaff_mix.264` | field pairs and MBAFF frames mixed in one stream |

The test file also checks:
- 1, 3 and 8 threads give identical output, damaged streams included;
- the `Decoder` trait reports the bit depth and chroma format.

`dpb.rs` has unit tests for:
- field list alternation;
- frame lists over completed field pairs;
- field PicNum list modification;
- field POC.

`simd.rs` checks that the vector MC, IDCT and deblocking kernels match the
scalar ones on field views.

During development I also checked a larger set of streams against FFmpeg,
with deblocking on and off, all bit-exact:
- 9 high-bit-depth/4:2:2 streams;
- 9 PAFF streams;
- 18 MBAFF streams;
- 4 mixed PAFF/MBAFF streams (spatial and temporal direct);
- a 1080i MBAFF clip.

320 fuzzed streams ran with no panics or hangs.

`tools/make_h264_fixtures.sh` regenerates the fixtures and prints FFmpeg's
hashes. x264 cannot write field pictures, so `tools/mkpaff.py` and
`tools/mkmix.py` splice them together from separately encoded top-field and
bottom-field streams. These splices are CAVLC only. CABAC field contexts
are still tested, because MBAFF field macroblocks use the same contexts.
`examples/h264_yuv.rs` decodes an Annex B stream to raw YUV
(`<in> <out> [threads]`) for comparison with `ffmpeg -f rawvideo`.

## Gaps

- **I_PCM has no test stream at high bit depth or in field MBs.** x264
  never writes PCM. The code follows the spec; a JM-encoded stream would
  cover it.
- **There is no CABAC field-picture (PAFF) fixture.** See above.
- **High bit depth runs on scalar code.** The `u16` kernels are the next
  step for 10-bit performance.
