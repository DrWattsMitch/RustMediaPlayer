# SIMD in RMP

This document covers the batch that vectorised RMP's hot loops: a small
audited module of portable vector types (`src/simd/`), runtime CPU
dispatch, and SIMD versions of the H.264, VP9 and audio transform kernels.
Read it before writing a new kernel or touching `src/simd/`.

---

## 1. Design

### One audited module, safe kernels everywhere else

`src/simd/` is the **only** place in the crate that contains `unsafe`.
`lib.rs` sets `#![deny(unsafe_code)]` and allows it just for `mod simd`, so
the compiler enforces the boundary: a kernel cannot reach for an intrinsic
directly.

The module exposes four `Copy` vector types whose every method is a safe
function:

| type    | lanes      | x86_64 (SSE2) | aarch64 (NEON) | other targets |
|---------|------------|---------------|----------------|---------------|
| `U8x16` | 16 x `u8`  | `__m128i`     | `uint8x16_t`   | `[u8; 16]`    |
| `I16x8` | 8 x `i16`  | `__m128i`     | `int16x8_t`    | `[i16; 8]`    |
| `I32x4` | 4 x `i32`  | `__m128i`     | `int32x4_t`    | `[i32; 4]`    |
| `F32x4` | 4 x `f32`  | `__m128`      | `float32x4_t`  | `[f32; 4]`    |

plus `transpose8x8` (16-bit) and `transpose4x4` (32-bit). Kernels are
written once against these types and compile to SSE2, NEON or plain loops.

Semantics are fixed and identical on every backend: integer ops wrap unless
named `saturating_*`, shifts take a run-time count, comparisons return
all-ones masks, `select` blends by mask, and float ops are plain IEEE-754
per lane **with no fused multiply-add**. `scalar.rs` is the executable
specification: `src/simd/tests.rs` checks every native method lane for lane
against it on random and edge-case inputs (`i16::MIN`, `i32::MAX`, `-0.0`,
denormals...).

### Why the `unsafe` is sound

There are exactly three kinds of `unsafe` in `src/simd/`:

1. **Value intrinsics** (`_mm_add_epi16`, `vaddq_s16`, ...). They touch no
   memory and are `unsafe` only because they require a CPU feature. Each
   backend file is compiled only under `#[cfg(target_feature = "sse2")]` /
   `#[cfg(target_feature = "neon")]`, which are part of the x86_64 and
   aarch64 baselines, so the requirement is proven at compile time. These
   calls go through one macro per backend (`sse!` / `neon!`), documented
   once.
2. **Loads and stores.** Each re-slices its argument to the exact length it
   touches (`let s = &s[..16];`) before taking the pointer, so the bounds
   check runs in safe code: a short slice panics, it never reads out of
   bounds (tested). All are the unaligned variants.
3. **The AVX2 call in `multiversion!`**, made only when `level()` reports
   `Avx2`, which it does only after `is_x86_feature_detected!("avx2")`;
   `set_level` can lower the level but never raise it above the detected
   one.

### Dispatch

`simd::level()` is detected once and cached in an atomic:

| level     | meaning                                                     |
|-----------|-------------------------------------------------------------|
| `Scalar`  | the original scalar code                                    |
| `Simd128` | the vector kernels (SSE2 on x86_64, NEON on aarch64)        |
| `Avx2`    | the same kernels recompiled with AVX2 (x86_64, run-time)    |

The `RMP_SIMD` environment variable caps it (`scalar`, `sse2`/`neon`,
`avx2`), and `simd::set_level` does the same from code. That is how the
scalar fallback is exercised on SIMD hardware.

Every public kernel entry point is a thin dispatcher in front of the
untouched scalar function, which was renamed `*_scalar`:

```rust
pub fn mc_luma(r: &Plane, /* ... */) {
	if crate::simd::enabled() && matches!(bw, 4 | 8 | 16) {
		super::simd::mc_luma(r, /* ... */)
	} else {
		mc_luma_scalar(r, /* ... */)
	}
}
```

`multiversion!` wraps a kernel so its body is compiled twice on x86_64,
once for SSE2 and once inside `#[target_feature(enable = "avx2")]`, and
picks one per call. Under AVX2 LLVM uses VEX three-operand encodings and
folds the SSE2 emulations (absolute value, 32-bit multiply, min/max) into
native instructions. For that to work the whole kernel has to inline into
the dispatched body. A closure or helper that LLVM keeps out of line is
compiled *without* AVX2. That is why the edge-emulation windows use a macro
rather than a closure.

## 2. Kernels

| Codec | Kernel | File | Notes |
|---|---|---|---|
| H.264 | luma 6-tap quarter-pel MC | `h264/simd.rs` | sliding row windows; centre tap in 32 bits via `madd`; reads straight from the reference plane, copying an edge-emulated window only when the block crosses the picture edge |
| H.264 | chroma bilinear MC | `h264/simd.rs` | 4- and 8-wide (2-wide stays scalar) |
| H.264 | prediction store: copy, average, explicit/implicit weighting | `h264/simd.rs` | bipred weights via `madd` in 32 bits; widths specialised at compile time |
| H.264 | deblocking edge filter (bS 1-4, luma and chroma) | `h264/simd.rs` | 8 lines per vector, per-lane bS/tc0; vertical edges via 8x8 transpose |
| H.264 | 4x4 / 8x8 inverse transform + add | `h264/simd.rs` | 32-bit lanes (exact for any coefficients) |
| VP9 | loop filter: 4-, 8- and 16-wide, both directions | `vp9/simd.rs` | mask/flat/flat2 per lane; 15-tap running sum |
| VP9 | 8-tap sub-pixel convolution (+ compound average) | `vp9/simd.rs` | tap pairs through `madd`; 16- and 8-wide paths |
| VP9 | DC-only inverse transform (`vpx_idctNxN_1_add`) | `vp9/transform.rs`, `vp9/simd.rs` | `eob == 1` shortcut, proven equal to the full transform |
| AAC, Vorbis | radix-2 FFT stages; IMDCT pre/post twiddle | `dsp/simd.rs` | shared `dsp::Imdct` (also the AAC encoder's MDCT) |
| Opus | CELT mixed-radix FFT combining loop | `opus/mdct.rs` | vectorised over `k`; twiddle table hoisted out of the loop |
| FLAC | LPC restore | `flac/decoder.rs` | scalar, monomorphised per order (see below) |
| HEVC | inverse DCT 8x8–32x32 | `hevc/simd.rs` (`idct16`) | 16-bit lanes; even/odd halves via `madd` on interleaved coefficient rows; `transpose8x8` between passes; the saturating pack is the spec's 16-bit clip |
| HEVC | residual add | `hevc/simd.rs` | 16-bit saturating add + clip, exact for every bit depth |
| HEVC | planar / angular intra prediction | `hevc/simd.rs` | `madd` on `(ref[i], ref[i+1])` pairs; horizontal modes transposed in registers; differential test over all 35 modes, 4 sizes, 3 bit depths |

Non-SIMD fixes made along the way, all exact:

- VP9 `inverse_transform_add` zeroed a 4 KB 32x32 scratch buffer for every
  block, including 4x4 ones. That was the decoder's largest `memset`. The
  scratch is now sized to the transform.
- After a DC-only block the VP9 decoder clears one coefficient instead of
  the whole block.
- Opus recomputed `(r·q·m) % len` twiddle indices for every butterfly term.
  They are now a per-call table.

### Deliberately not vectorised (yet)

- **VP9 inverse DCT/ADST.** RMP computes them in 64-bit with libvpx's
  32-bit wrap, so every conformant stream (whose intermediates may legally
  reach 24 bits) decodes exactly. A 32-bit-lane version would differ on
  such streams. A sound version needs either 64-bit lanes or a proven
  per-block range check. The DC-only shortcut already covers a large share
  of blocks.
- **FLAC LPC.** The recursion is serial (each sample needs the previous
  one), and an exact sum needs 64-bit accumulation for 16-bit audio with
  15-bit coefficients. SSE2 has no 64-bit multiply, so the gain is small.
  The order-specialised scalar loop is about 10% faster.
- **Intra prediction** in H.264, and the H.264 and HEVC boundary-strength
  computation, are scalar. (HEVC planar/angular intra is vectorised; DC
  stays scalar since it is a fill.)

## 3. Correctness

Every SIMD kernel must be **bit-exact** with the scalar code it replaces,
integer and float alike. That is enforced at three levels:

1. **Backend conformance** (`src/simd/tests.rs`): native vs `scalar.rs`
   model, every method, about 2000 random rounds each.
2. **Kernel differential tests**: each `*/simd.rs` (and the dsp/opus/flac
   modules) compares the vector kernel against its `*_scalar` original:
   every H.264 quarter-pel fraction x block size x position, including
   blocks hanging off every picture edge; all weightings and lists; deblock
   at random qp/alpha/beta/bS on blocky content (asserting that filtering
   actually happened); VP9 4/8/16-wide filters (with a test that the
   15-tap path really runs), all filter types/phases/sizes/compound; IDCTs
   with coefficients up to 2^20; FFT/DCT-IV/Opus FFT compared bit-for-bit
   on the `f32` bit patterns; FLAC orders 0-32 including wrapping inputs.
3. **End to end**: the fixture suite (decoded output vs FFmpeg references)
   at every level, plus 720p H.264 High (B-frames, CABAC, 8x8) and VP9
   clips whose output is byte-identical at `scalar`, `sse2` and `avx2`, to
   the pre-SIMD build, and to FFmpeg.

The same tests were run on **aarch64/NEON** (cross-compiled, under qemu):
all lib, codec, container, io and process tests pass, and every benchmark
clip (video and all four audio codecs) decodes byte-identically at `neon`
and `scalar` to the x86 output. (`tests/cli.rs` spawns the `rmp` binary
directly, so it can't run under user-mode qemu in a container without
binfmt; the binary itself was checked through qemu by hand.)

To check a change yourself:

```sh
cargo test                                   # includes all differential tests
RMP_SIMD=scalar cargo run --release --example decode -- in.mp4 - a.yuv
cargo run --release --example decode -- in.mp4 - b.yuv
cmp a.yuv b.yuv                              # must be identical
```

## 4. Performance

Single-threaded decode to `/dev/null`, best of 5, 2-core Xeon @ 2.1 GHz
(AVX2), full-LTO release build. Clips: 720p H.264 High (crf 23, 3 B-frames,
6 s) and 720p VP9 (3 Mb/s, 4 s) of fractal content; 60 s of 48 kHz stereo
for each audio codec.

| clip | before | scalar now | SSE2 | AVX2 | FFmpeg 1 thread |
|---|---|---|---|---|---|
| H.264 720p | 3.53 s | 3.58 s | 2.10 s | **2.04 s** | 0.88 s |
| VP9 720p | 2.47 s | 2.42 s | 1.27 s | **1.26 s** | 0.69 s |
| Opus | 0.41 s | 0.34 s | 0.30 s | **0.31 s** | 0.35 s |
| FLAC | 0.17 s | 0.15 s | 0.15 s | 0.15 s | 0.16 s |
| AAC | 0.14 s | 0.14 s | 0.14 s | 0.14 s | 0.26 s |
| Vorbis | 0.15 s | 0.16 s | 0.15 s | 0.15 s | 0.26 s |

The video gap to FFmpeg went from about 4.0x to 2.3x (H.264) and from
3.6x to 1.8x (VP9). Instruction counts (callgrind, 1 s of each clip) fell
from 6.2 G to 3.2 G for H.264 and from 8.9 G to under 3.3 G for VP9. The
audio times are mostly process start-up and I/O. The IMDCT used to be
30-55% of the audio decoders' instructions.

The AVX2 recompile is worth about 3-12% on H.264 depending on the run
(it is what keeps the luma MC loop in registers) and nothing measurable on
VP9 yet. Real 256-bit kernels (16- and 32-wide VP9 MC and loop filter
rows) are where it would pay.

### What's left in the profiles

- **H.264:** CABAC decoding and macroblock parsing (about 15%), the scalar
  deblocking boundary-strength computation (about 10%), and per-call
  overhead in `mc_rect` (zeroed prediction buffers, three planes x two
  lists).
- **VP9:** MC per-call overhead and bounds checks on small blocks, the
  64-bit transforms, and token decoding.
- Both: no frame/slice threading. FFmpeg's single-thread numbers above
  are the fair comparison.

## 5. Writing a new kernel

1. Keep the scalar function as `*_scalar` and make the public name a
   dispatcher on `crate::simd::enabled()` (plus any shape it can't take).
2. Write the vector version in the codec's `simd.rs` with the
   `crate::simd` types, inside `multiversion!` if it's hot. Keep the
   scalar operation order, and keep closures out of the hot loop (or make
   sure they inline).
3. If a type needs an operation it doesn't have, add it to **all three**
   backends and to `src/simd/tests.rs`.
4. Add a differential test against `*_scalar`, including the edges, and
   assert that the interesting path actually ran.
5. Check end to end with `RMP_SIMD=scalar` vs default, and build for
   aarch64 (`cargo clippy --target aarch64-unknown-linux-gnu`).
