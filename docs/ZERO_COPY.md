# Zero-copy frames: reference-counted buffers and frame pools

Decoded pictures used to leave RMP as a `Vec<u8>` in packed planar layout.
Each output frame was a fresh allocation and a full copy of the decoder's
picture (H.264 cropped it, VP9 de-strided it). Every `frame.clone()` copied
it again. Now a frame is a set of **reference-counted plane views** into
memory the decoder already owns. Picture memory comes from **per-decoder
pools**, so steady-state decoding allocates almost nothing per frame. This
is the job FFmpeg's `AVBufferRef`, `AVBufferPool` and `AVFrame.buf[]` /
`linesize[]` do. Everything is safe Rust: `unsafe` stays confined to
`src/simd`.

## Pieces

| RMP | FFmpeg | Where |
| --- | --- | --- |
| `FrameBuffer`: immutable, `Arc`-counted bytes; `clone` = one atomic increment | `AVBufferRef`, `av_buffer_ref` | `core/frame/buffer.rs` |
| `VecPool<T>` / `BufferPool`, `Pooled<T>` (unique, mutable, returns to its pool on drop) | `AVBufferPool`, `av_buffer_pool_get` | `core/frame/buffer.rs` |
| `Pooled::freeze` (unique → shared); `FrameBuffer::into_mut` (copy-on-write) | `av_buffer_make_writable` | `core/frame/buffer.rs` |
| `FrameBuffer::project(Arc<owner>, fn)`: shares one region of an owner (a decoder's reference picture) and keeps the owner alive | frames that reference the DPB picture | `core/frame/buffer.rs` |
| `PlaneRef`: buffer + offset + stride + visible row bytes + rows | `data[i]`, `linesize[i]` | `core/frame/video/video.rs` |
| `VideoFrame::from_planes` / `from_packed`; `planes()`, `plane(i)` | `AVFrame` | `core/frame/video/video.rs` |
| `VideoFrame::data()` (borrowed if built from packed bytes), `into_packed()` (free if unique), `write_packed()` (streams rows) | `av_image_copy_to_buffer` | `core/frame/video/video.rs` |

### How the decoders use it

* **H.264.** `DecodePools` holds pools for samples, `MbInfo` and stored
  motion. `PictureJob` takes its planes, macroblock array and motion from
  them (`Frame::pooled`). With frame threads, the `live` publish planes come
  from them too. Output cropping is now *views*: three `PlaneRef`s with the
  crop offset and the macroblock-aligned stride into the finished
  `PicData`. The DPB and every frame share the picture. When both have let
  go, `PicData` drops and its planes go back to the pool.
* **VP9.** Picture planes come from a `BufferPool`. Shown frames are views
  into the `Arc<Picture>` that the reference slots hold, at the superblock
  stride (`Picture::to_frame`). sRGB streams still pack G/B/R into RGB24
  (one pass), because RMP has no `gbrp` format yet. Per-frame arrays
  (`ModeInfo`, loop-filter cells, motion vectors) are reused between frames.
  The tile-thread row hand-over (`Chunk`) uses pooled buffers.
* **Raw video / Y4M.** The packet's `Vec` *becomes* the frame
  (`from_packed`, no `clone()`). The raw encoder gets it back with
  `into_packed()`, which is free when the frame is the only reference.
* **Scaler.** An identity conversion returns a new reference instead of a
  deep copy. Real conversions write into a pooled buffer.
* **H.264 encoder.** `encode_frame(&VideoFrame)` reads planes at any
  stride, so H.264 → H.264 no longer packs the decoded picture first.

Pool rules. `take_filled` gives a buffer filled with a known value. It is
used where decoding reads before writing, or where error concealment could
leave samples untouched. Determinism across thread counts depends on that.
`take_recycled` keeps old contents. It is used only where every element is
written before it is read: H.264 `live` planes (readers wait for the
published rows) and stored motion. Pools are bounded (32 idle buffers by
default) and thread-safe. A buffer that outlives its pool is simply freed.

### Two non-frame fixes found along the way

* `Decoder::decode` / `Encoder::encode` used to return `state.clone()` with
  every call. That was a deep copy of the codec state per packet, and no
  caller used it. `DecodeOut` / `EncodeOut` now hold only the output.
* H.264 `pred_block` (Intra16x16 and chroma prediction) allocated a `Vec`
  per block. It now uses a stack array. This alone removed ~9,700
  allocations per 1080p frame.

## For players: GPU hand-off

```rust
for plane in frame.planes() {
    // wgpu:
    queue.write_texture(tex(plane), plane.bytes(),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(plane.stride() as u32), rows_per_image: None },
        size(plane.row_bytes(), plane.rows()));
    // OpenGL: glPixelStorei(GL_UNPACK_ROW_LENGTH, stride); glTexSubImage2D(..., plane.bytes().as_ptr())
}
```

`VideoFrame` is `Send + Sync`. A render thread can hold frames while the
decoder moves on, and they stay valid even after the decoder is dropped
(tested). To keep only one plane alive during an async upload, clone
`plane.buffer()`. Strides: H.264 is a multiple of 16 (macroblock width);
VP9 luma is a multiple of 64. `PlaneRef::is_aligned(n)` checks the base
pointer and stride for paths with stricter rules (wgpu's
`copy_buffer_to_texture` wants 256). `examples/zero_copy.rs` decodes on one
thread and "uploads" on another through a 4-frame queue.

## API changes (breaking)

* `VideoFrame.data: Vec<u8>` is gone. Use `data()` (a `Cow`),
  `into_packed()`, `write_packed(&mut w)`, `planes()` / `plane(i)`. Build
  frames with `VideoFrame::from_packed(bytes, w, h, pixel, keyframe, pts)` or
  `from_planes(...)`.
* `Decoder::decode/flush -> Result<DecodeOut>` and
  `Encoder::encode/flush -> Result<EncodeOut>` no longer take a state type
  parameter, and they carry no `state` field.
* `H264DecoderState::decode_packet` returns `DecodedPicture { planes, .. }`
  with `data()` and `into_frame()`. Both decoders have a `pool_stats()`
  method.
* `AudioFrame` is unchanged (still a `Vec`). Audio frames are small (a few
  KB), so they were left for later.

## Measurements

Same streams as `THREADING.md`: 1080p30, 120 frames of `testsrc2` + noise
(H.264 High with B-frames and CABAC, H.264 Baseline, VP9 with 4 tile
columns and with 1). x86-64 cloud VM, 2 vCPUs, FFmpeg 6.1.1. Wall-clock
medians of interleaved runs (3 for H.264, 5 for VP9). "Before" is the
tree as it was right before this change (with the threading work). Output
is byte-identical before and after, and bit-identical to FFmpeg
(`-f rawvideo` MD5) at 1 and 2 threads on all four streams.

### Allocations per decoded frame

Counted with a counting global allocator (a bench-only tool, not shipped),
from the first packet to the end of `flush`. The frames are dropped as
they arrive, as a player would.

| Stream | threads | allocations/frame, before → after | bytes/frame, before → after |
| --- | --- | --- | --- |
| H.264 High | 1 | 9,720 → **21.9** | 12.8 MB → **0.49 MB** |
| H.264 High | 2 | 9,727 → **23.1** | 17.4 MB → **0.66 MB** |
| H.264 Baseline | 1 | 6,741 → **17.3** | 12.9 MB → **0.39 MB** |
| H.264 Baseline | 2 | 6,748 → **18.5** | 17.4 MB → **0.53 MB** |
| VP9, 4 tiles | 1 | 37.1 → **21.2** | 13.3 MB → **0.51 MB** |
| VP9, 4 tiles | 2 | 332 → **45.7** | 15.1 MB → **0.78 MB** |
| VP9, 1 tile | 1 | 37.1 → **21.2** | 13.3 MB → **0.51 MB** |

What remains is mostly input-sized: H.264's escaped-byte removal (RBSP)
copies each slice, plus slice jobs, reference lists, VP9's per-frame
`Counts`, segment map and above-context arrays, and the output `Vec<Frame>`.
A 1080p picture is 3.1 MB, so less than a sixth of one picture is allocated
per frame. Before, it was four to five pictures' worth.

`VideoFrame::clone()` at 1080p went from **399 µs** (H.264) / **322 µs**
(VP9), a full copy, to **0.2-0.5 µs**.

Picture copies between the decoder and the consumer:

| Path | Before | After | FFmpeg |
| --- | --- | --- | --- |
| decode → player / `-f null` | 1 (crop / de-stride) | **0** | 0 |
| decode → Y4M / rawvideo | 2 (crop + identity "conversion") | **1** (packing into the packet) | 1 (`av_image_copy_to_buffer`) |
| hand a frame to another thread / queue it | full copy per `clone` | refcount | refcount (`av_frame_ref`) |

### Speed

Decode to null (RMP `examples/decode in - -`, FFmpeg `-f null -`):

| Stream | T | before | after | change | FFmpeg | RMP/FFmpeg, before → after |
| --- | --- | --- | --- | --- | --- | --- |
| H.264 High | 1 | 8.09 s | **7.67 s** | −5.2% | 4.39 s | 1.84x → 1.75x |
| H.264 High | 2 | 4.18 s | 4.23 s | +1% (noise) | 2.33 s | 1.79x → 1.81x |
| H.264 Baseline | 1 | 4.75 s | 4.81 s | +1% (noise) | 1.98 s | 2.40x → 2.43x |
| H.264 Baseline | 2 | 2.94 s | **2.66 s** | −9.6% (CPU −8.5%) | 1.07 s | 2.76x → 2.49x |
| VP9, 4 tiles | 1 | 1.31 s | **1.22 s** | −6.8% | 0.89 s | 1.47x → 1.37x |
| VP9, 4 tiles | 2 | 0.82 s | **0.77 s** | −5.7% | 0.65 s | 1.26x → 1.19x |
| VP9, 1 tile | 1 | 1.28 s | 1.27 s | −1.2% | 0.86 s | 1.49x → 1.47x |
| VP9, 1 tile | 2 | 1.26 s | **1.19 s** | −6.2% | 0.64 s | 1.98x → 1.86x |

Transcode to Y4M (`rmp -threads 2 -i X -an out.y4m` vs `ffmpeg -threads 2`),
writing 373 MB per run:

| Stream | before | after | FFmpeg |
| --- | --- | --- | --- |
| H.264 High | 4.69 s | 4.67 s | 2.87 s |
| H.264 Baseline | 3.20 s | **2.99 s** | 1.77 s |
| VP9, 4 tiles | 1.06 s | 1.06 s | 1.15 s |
| VP9, 1 tile | 1.49 s | 1.56 s | 1.06 s |

### Reading the numbers

* **Allocation is solved; raw decode speed barely moves.** Decoding is
  compute-bound (entropy decoding, prediction, transforms, filters). glibc
  was already recycling the freed 3 MB blocks, so page faults hardly
  changed. The removed work is about one or two 3 MB memsets/copies per
  frame (≈0.3-0.6 ms out of 10-65 ms), which gives the 0-10% above. On a
  2-vCPU VM, ±3% is within noise.
* **The gap to FFmpeg is now almost all codec work, not plumbing.** On the
  frame path RMP now does what FFmpeg does: no output copy, refcounted
  hand-off, pooled pictures. The remaining 1.2-2.5x is in the decoding
  kernels and in threading overhead (below).
* **Where it matters most is in a player.** Before, each displayed 1080p
  frame cost a 3 MB allocation plus a copy at output, and another copy for
  every queue or thread hop. Now it costs nothing and can be uploaded from
  the decoder's own memory. Memory traffic per frame falls by 3-6 MB
  (≈100-200 MB/s at 30 fps; more at 4K or 60 fps), which helps battery
  life and cache behavior more than these throughput numbers show.
* **Transcode to Y4M is bound by output bandwidth** (373 MB written per
  run), so dropping one in-memory copy is mostly hidden. VP9 1-tile is
  within noise.

## Still different from FFmpeg

* **H.264 frame threads still copy finished rows into `live`.** FFmpeg's
  threads write and read one shared picture, guarded by progress counters.
  Doing that in safe Rust needs the picture split into independently owned
  row bands. Deblocking writes 3 rows across each band edge, so this is a
  real redesign. It is the biggest remaining cost of H.264 threading (see
  `THREADING.md`).
* **Alignment is the allocator's.** FFmpeg's `av_malloc` gives 64-byte
  aligned planes with padded linesizes. RMP's bases are what the system
  allocator gives (16 bytes on 64-bit glibc and macOS), and strides are the
  codec's natural ones (16 / 64). Over-allocating pool buffers to align the
  base (`align_offset`, safe) is a small follow-up if an upload path needs it.
* **No `get_buffer2` hook.** FFmpeg lets the application supply frame
  memory (decode straight into GPU-mapped buffers). `BufferSource` is the
  read side of that. A decoder-side "allocate from the caller" hook would
  be next.
* **Packets are still `Vec<u8>`.** FFmpeg packets are refcounted too, and
  H.264 RBSP extraction copies each slice. Making `Packet` hold a
  `FrameBuffer` and reusing an RBSP buffer would remove most of the
  remaining per-frame allocations.
* **Audio frames are still `Vec`-backed.**

## Tests

* `core::frame::buffer`: recycling after the last reference, bounded idle
  count, a pool dropped before its buffers, `into_vec` / `into_mut`
  without copies, projected buffers keeping their owner alive.
* `core::frame::video`: packed frames round-trip without copying, short
  data is reported incomplete, strided planes pack correctly, pooled planes
  return when the last frame clone goes.
* `h264_decoder_test::output_frames_share_and_recycle_decoder_buffers`
  (1 and 3 threads, the clip decoded 3 times): clones share planes, strides
  are macroblock-aligned, and the pool's allocations stay bounded while
  reuse dominates. `frames_outlive_the_decoder`: frames match FFmpeg after
  the decoder is dropped.
* `vp9_decoder_test::output_frames_share_and_recycle_decoder_buffers`:
  views match the framecrc reference, strides are superblock-aligned, and
  at most 27 planes are ever allocated (8 slots + 1, × 3).
* All 376 tests pass (the existing ones unchanged in what they check), including
  bit-exactness against FFmpeg for every H.264/VP9 fixture at 1, 2, 3 and
  8 threads.
