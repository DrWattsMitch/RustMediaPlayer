# HEVC range extensions: 4:0:0, 4:2:2, 4:4:4 and the RExt tools

The HEVC decoder (`src/codecs/hevc/`) used to accept Main and Main 10 at
4:2:0 only. It now decodes the format range extensions (RExt) profiles up
to 12 bits:

- Monochrome, Monochrome 12
- Main 12
- Main 4:2:2 10 and 12
- Main 4:4:4, 4:4:4 10 and 4:4:4 12
- the Intra and Still Picture variants of these

That covers camera and mezzanine footage (4:2:2 10-bit), screen and
graphics content (4:4:4) and monochrome streams. Frames come out as
`Gray`, `YUV420`, `YUV422` or `YUV444`, 8-bit or 16-bit little-endian.

## 1. What was added

### Chroma formats

| Area | 4:0:0 | 4:2:2 | 4:4:4 |
|---|---|---|---|
| Parameter sets | no chroma syntax (SAO, weights, PCM) | crop in chroma units | separate colour planes rejected |
| Transform tree | luma only | two square chroma blocks per TU, a `cbf_cb`/`cbf_cr` for each; chroma of 4x4 luma TUs carried by the 4th | chroma TB = luma TB, down to 4x4 |
| Intra | – | chroma mode mapped by Table 8-3; 2nd block predicted from the 1st | one chroma mode per NxN block; reference smoothing on chroma |
| Chroma QP | – | `Min(qPi, 51)` instead of Table 8-10 | same |
| Inter | – | MVs in 1/8 horizontally, 1/4 vertically | 1/4 both ways (fed to the 4-tap filter in eighths) |
| Deblocking | – | 8x8 chroma grid: every 16 luma columns, 8 rows | every 8 luma columns and rows |
| SAO | – | CTB 1/2 wide, full height | full CTB |

Chroma planes of 4:2:2 and 4:4:4 are padded like luma (80 samples), so the
"motion compensation only clamps block positions" design still holds for
64-sample-wide chroma blocks.

### Range extension tools

| Tool | Where |
|---|---|
| `sps_range_extension`, `pps_range_extension`, and the VUI (parsed, with FFmpeg's recovery for old-draft VUIs, to reach the extension) | `ps.rs` |
| Transform skip up to 32x32 (`log2_max_transform_skip_block_size`), tsShift = 5 + log2 size | `transform::transform_skip` |
| Transform skip rotation (4x4 intra), transform skip contexts | `ctu.rs` residual |
| Implicit (intra) and explicit (inter) RDPCM, including no sign hiding | `transform::rdpcm`, `ctu.rs` |
| Persistent Rice adaptation (StatCoeff, carried with the CABAC contexts through WPP and dependent slices) | `cabac::Contexts`, `ctu.rs` |
| Cross-component prediction (4:4:4) | `ctu.rs` `transform_unit` |
| CU chroma QP offsets (`cu_chroma_qp_offset_flag` / `_idx`, chroma QG reset) | `ctu.rs` |
| `intra_smoothing_disabled`, no boundary filters for implicit RDPCM + lossless | `ctu.rs` `intra_pred` |
| High-precision weighted prediction offsets | `slice.rs` |
| SAO offset scaling (`log2_sao_offset_scale_luma/chroma`) | `ctu.rs` `sao_syntax` |

Not supported (rejected with an "unsupported" error): the 16-bit
profiles' `extended_precision_processing_flag` and
`cabac_bypass_alignment_enabled_flag` (FFmpeg does not implement them
either), `separate_colour_plane_flag`, SCC and multi-layer.

Safety: everything stays in safe Rust under the crate's
`#![deny(unsafe_code)]`. Damaged input never panics: Rice parameters are
bounded and StatCoeff saturates.

**Performance.** Main/Main 10 streams take the same paths as before. The
significance-flag contexts now come from a per-sub-block table plus an
offset, as in FFmpeg. On 12 frames of the 1080p 8-bit benchmark clip,
that cut the decoder's instruction count by 0.8% from before this change
(2.808 G → 2.785 G with Cachegrind, one thread), with identical output.

## 2. Where RMP follows the standard and HM, not FFmpeg 6.1

These differences only show on streams that use the tool combinations
listed here. HM's code is in `TComTrQuant.cpp` / `TDecSbac.cpp`.

1. **Lossless chroma after SAO.** FFmpeg's `restore_tqb_pixels` mixes luma
   positions with chroma sizes, so in 4:2:0 and 4:2:2 it only restores the
   top-left part of each CTB's lossless (or PCM) chroma samples after SAO.
   RMP restores all of them. This also affects Main 4:2:0 streams that
   use `cu_transquant_bypass` with SAO.
2. **4:2:2 SAO at CTB edges.** On an x265 4:2:2 stream with 16x16 CTBs,
   FFmpeg's output differs from x265's reconstruction by one sample. RMP
   matches x265 (`hevc_422_small_ctbs_match_the_encoder`).
3. **Rotation of lossless blocks.** `transform_skip_rotation` applies to
   lossless 4x4 intra blocks too (HM `isNonTransformedResidualRotated`).
   FFmpeg rotates transform-skip blocks only.
4. **Explicit RDPCM direction in lossless inter CUs** when implicit RDPCM
   is also enabled: RMP uses `explicit_rdpcm_dir_flag`. FFmpeg uses the
   (absent) intra mode.
5. **disableIntraBoundaryFilter** (implicit RDPCM + lossless): RMP skips
   the DC/angular edge filters, as HM does. FFmpeg does not.
6. **`cu_chroma_qp_offset_idx`** is truncated at `chroma_qp_offset_list_len_minus1`.
   FFmpeg reads up to 5 bins whatever the list length, so it loses sync
   when lists are shorter than 6.
7. **High-precision offsets.** These are implemented. FFmpeg only logs
   "not yet implemented".

On damaged data (values outside what a conforming stream can produce),
RMP keeps FFmpeg's behaviour: lossless levels and RDPCM sums wrap to 16
bits, as FFmpeg's `int16_t` coefficients do.

## 3. Verification

**x265 streams, against x265's own reconstruction** (8-bit) **or FFmpeg**
(10/12-bit, where the x265 CLI's reconstruction is 8-bit only).
4:0:0, 4:2:0, 4:2:2 and 4:4:4 at 8, 10 and 12 bits, each with these
option sets:

- intra NxN
- B-pyramid + AMP + rd 6
- transform skip
- lossless and CU-lossless
- WPP + slices
- weighted prediction
- scaling lists with chroma QP offsets
- no SAO / no deblocking / constrained intra
- sign hiding
- QP 4 with 64x64 CTBs
- AQ with 8x8 QGs
- 16x16 CTBs with 4x4 TUs
- max-merge 1

Odd sizes (131x71, 130x71) are also covered.

Result: 168 + 18 streams, all bit-exact at 1, 3 and 4 threads. There are
two exceptions, both involving another decoder:

- 4:2:2 with 16x16 CTBs: RMP = x265 ≠ FFmpeg (point 2 above).
- 4:0:0 with weighted prediction: RMP = FFmpeg ≠ x265. x265 writes the
  syntax correctly, so this looks like an x265 reconstruction issue.

**Range extension tools, against FFmpeg.** x265 never uses these tools,
and the JCT-VC RExt conformance streams were not reachable from the build
machine. Instead, `tools/hevc_rext_streams.py` writes streams with valid
VPS/SPS/PPS/slice headers that enable the tools, followed by random CABAC
slice data. Every byte string is some CABAC bitstream, so two decoders
that implement the standard the same way produce the same pictures.

How the streams are built:

- Bytes are drawn with few one bits, to keep levels realistic.
- Each picture is a single CTB, so a random end-of-slice bin cannot leave
  half a picture undecoded.
- The first 9 bits are kept below 510 (FFmpeg rejects those slices).

The campaign covered 4 chroma formats × 3 bit depths × 3 densities ×
lossless on/off × cu_qp_delta on/off, with random scaling lists, sign
hiding, smoothing and SAO scale. It ran 576 streams (3,456 pictures,
I + weighted P) at 1 to 3 threads. 573 were bit-exact. 3 were skipped
because FFmpeg reported the random data invalid (`cu_qp_delta` out of
range). None differed. Combinations from §2 were left out of that
comparison.

**Tests** (`tests/codecs/hevc_decoder_test.rs`):

- `hevc_monochrome_12_is_bit_exact`
- `hevc_422_10_is_bit_exact`
- `hevc_444_12_is_bit_exact`
- `hevc_422_small_ctbs_match_the_encoder`
- `hevc_444_odd_size_matches_the_encoder`
- `hevc_range_extension_tools_are_bit_exact`: 5 synthetic streams
- `hevc_16_bit_tools_are_rejected`
- the new fixtures joined the thread-count, WPP-row and damaged-input
  tests

There are also unit tests for transform skip (against FFmpeg's `dequant`
formula) and RDPCM.

Still unverified against an independent decoder: the choices in §2
(points 3 to 7), where FFmpeg cannot be the reference and HM could not be
built here. The JCT-VC RExt conformance suite (`ADJUST_IPRED_ANGLE`,
`CCP_*`, `ExplicitRdpcm_*`, `PERSIST_RPARAM_*`, `IPCM_*_444`, `Main_422_10_*`,
`WAVETILES_RExt`, ...) is the next check: decode each stream with
`examples/decode` and compare against the MD5s that ship with it.
