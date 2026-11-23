# Threading: H.264 and HEVC frames, HEVC WPP rows, VP9 tiles/rows, transcode pipeline

RMP decodes on several threads in four places. Every one of them is
safe Rust (`std::thread`, `std::thread::scope`, `mpsc` channels, `Arc`,
`Mutex`/`RwLock`/`Condvar`, `OnceLock`). There is no new `unsafe`. Output is
**bit-identical** for every thread count: the tests decode every H.264,
HEVC and VP9 fixture with 1, 2, 3 and 8 threads and compare, damaged
streams included, and the pipelined transcode is checked byte-for-byte
against the serial one.

| Where | Model (FFmpeg equivalent) | Code |
| --- | --- | --- |
| H.264 decoder | frame threading: picture N+1 waits only for the reference rows it reads (`pthread_frame.c`, `ff_thread_await_progress`) | `codecs/h264/decoder.rs`, `dpb.rs` (`PicData`) |
| HEVC decoder | frame threading with per-CTB-row progress, plus optional WPP row threading from the entry points (`hevc_refs.c` / `pthread_frame.c`; `hls_decode_entry_wpp`) | `codecs/hevc/decoder.rs`, `picture.rs` (`RefPic`), `filter.rs` (`advance`), `wpp.rs` |
| VP9 decoder | tile-column threading, with the loop filter running one superblock row behind (`vp9.c: decode_tiles_mt` + `loopfilter_proc`) | `codecs/vp9/tile_threads.rs` |
| `transcode` / CLI | demux, decode and encode+mux on separate threads (FFmpeg 7's threaded `ffmpeg`) | `transcode.rs` (`run_pipelined`) |
| shared | `Progress`, `WorkerPool`, thread-count defaults | `codecs/threading.rs` |

## Thread count

`-threads <n|auto>` on the CLI (`Options::threads`), `RMP_THREADS` in the
environment, or `threading::set_default_threads`. The default is the CPU
count (capped at 16), like FFmpeg. `-threads 1` makes everything
single-threaded. `-j` is still the number of files a batch converts at once.
The library constructors `H264DecoderState::new()`,
`HevcDecoderState::new()` and `Vp9DecoderState::new()` stay
single-threaded. Use `with_threads(n)` to choose a count
(`HevcDecoderState::with_threading(extradata, frames, rows)` sets the two
HEVC levels separately). The `Decoder` trait implementations (which the resolver and
`transcode` use) take `default_threads()`.

## H.264 frame threading

* **Setup stays serial.** NAL parsing, slice headers, POC, reference-list
  construction and DPB marking run on the calling thread in decode order,
  as before. Only macroblock decoding, concealment and deblocking (about
  95% of the time) go to a `WorkerPool` job (`PictureJob`). The DPB stores
  the `Picture` before its pixels exist. A picture's samples and motion
  live in `Arc<PicData>`.
* **Progress = deblocked macroblock rows.** The decoder now deblocks row
  `r` as soon as row `r+1` is decoded (intra prediction in `r+1` needs `r`
  unfiltered). This is the same raster order as the old whole-frame
  deblock, so the result is identical. After row `d` is deblocked, luma
  rows `< 16d-3`, chroma rows `< 8d-2` and the motion of MB rows `< d` are
  final. Those rows are copied into `PicData::live` (an `RwLock`) and
  `Progress` is raised.
* **Readers wait for what they read.** Motion compensation computes the
  bottom row its 6-tap/bilinear window touches (clamped to the picture,
  like edge extension) and calls `PicData::with(need, …)`. Temporal and
  spatial direct do the same for the co-located motion. Finished pictures
  are read lock-free from a `OnceLock`.
* **No deadlock by construction.** Jobs only wait on earlier jobs, the
  pool is FIFO, and at most `threads` pictures are in flight. If a job
  panics, its picture is poisoned and its waiters panic instead of hanging.
* **Output delay.** Pictures leave the decoder `threads - 1` packets later
  (FFmpeg has the same delay). The delay is fixed, so *which* call returns a
  picture depends only on the thread count, never on timing.
* Out-of-order slices that would rewrite rows that were already filtered
  are dropped as damaged. Only ASO streams, which RMP rejects anyway, have
  them.

## HEVC frame threading and WPP rows

**Frame threads (the default).** As for H.264, the calling thread parses
NAL units, slice headers, POC, the RPS, reference lists and DPB marking in
decode order. It no longer decodes slices: it keeps each slice segment's
RBSP, header and lists, and when the picture is complete it hands them to
a `PictureJob` on the `WorkerPool`. The DPB holds an `Arc<RefPic>` for the
picture from that moment.

* **In-loop filters run in CTB-row bands** (`PicState::advance`). Row `r`
  is deblocked once row `r + 1` is decoded, because intra prediction in
  row `r + 1` reads row `r`'s last line before filtering. SAO of row `r`
  runs once row `r + 1` is deblocked, because it reads one deblocked row
  below. The result equals filtering the whole picture at once. A
  horizontal edge at `y` reads rows `y - 4..y + 4`, which this band and the
  one above have already filtered vertically. It writes only rows
  `y - 3..y + 3`, which no later vertical edge reads. SAO keeps a
  deblocked copy of the band plus one row on each side; the row above
  comes from a copy saved before that row had its own SAO.
* **Progress = final rows.** After SAO of a band, its rows get their
  left/right padding (the first band also the top padding), are copied
  into `RefPic::live` (an `RwLock`), and `rows` is raised. The motion
  field, CTB slice map and slice reference POCs of fully decoded CTB rows
  are published with them, under a second counter (`motion`).
* **Readers wait for what they read.** Motion compensation computes the
  reference rows the 8-tap luma or 4-tap chroma window touches (anything
  in the bottom padding needs the whole picture) and calls
  `RefPic::with_rows`. TMVP calls `with_motion` for the collocated CTB
  row. Finished pictures are read without locking from a `OnceLock`.
* **No copy when nobody can read.** Sub-layer non-reference pictures of
  the highest sub-layer (x265's non-reference B frames) never publish:
  their readers, if any, wait for the whole picture. With one thread
  nothing is published and the job runs on the calling thread.
* The cropped output copy is also made by the job, so the calling thread
  does only parsing.
* **Output delay**: pictures are returned `threads - 1` packets after the
  packet that bumped them. The delay is fixed, so which call returns a
  picture never depends on timing.
* At most `threads` pictures are in flight, jobs wait only on earlier
  jobs, and the pool is FIFO, so there is no deadlock. A panicking job
  poisons its picture, and readers then panic instead of hanging.

**WPP rows (`wpp.rs`).** A WPP slice segment (entropy coding sync, no
tiles) that starts a CTB row and carries entry points is decoded row by
row, each row from its own substream. Entry points count bytes of the
escaped NAL unit, so the unescaper records where it removed emulation
prevention bytes and the offsets are mapped through them. With
`row_threads > 1` the rows run on scoped threads, each row two CTBs behind
the row above:

* Each row thread decodes into its own full-size `PicState` workspace
  (recycled across pictures), so the CTU code keeps whole-picture
  indexing. A row reads only its own CTBs and the last line of the row
  above: samples, motion, CBFs, split depths, skip flags, SAO parameters
  and slice addresses. After every CTB the thread copies that CTB's bottom
  line into a shared `Edge` and raises the row's `Progress`. The thread
  below waits for CTB `x + 1` above before decoding CTB `x`, copies the
  edge in, and starts its row from the contexts saved after CTB 1 above
  (9.3.1).
* Finished rows are merged into the picture, which runs the filters and
  publishes progress as rows complete. That is how frame and row threads
  combine.
* **Same output with any thread count, damaged streams included.** The
  one-thread path runs the same row decoder in place. A row decodes CTB
  `x` only if the row above decoded CTB `x + 1`, whatever the timing.
  Workspaces reset each CTB's area before decoding it, so a half-decoded
  CTB merges exactly as it would have been written in place.
* Whether a segment goes row by row depends only on the stream. Valid
  streams decode identically either way. A segment that overlaps decoded
  CTBs is dropped.

**Defaults.** `-threads n` / `RMP_THREADS=n` give HEVC `n` frame
threads, as FFmpeg does. From 8 threads up, each picture also gets 2 WPP
row threads. On a 14-core Mac that was 7–15% faster than frame threads
alone, on every clip. On 2 vCPUs the extra row threads oversubscribed the
cores and were slower, so below 8 the default stays frame threads only.
Rows alone, with one picture in flight, lost on both machines: frame
threads are the main mechanism, and rows fill in when they wait on
references. `RMP_HEVC_ROWS=r` sets the row threads per picture (1 turns
them off) and `RMP_HEVC_FRAMES=f` overrides the pictures in flight.
`tools/bench_hevc.sh` measures the combinations.

## VP9 tile and row threading

* Tile columns share no state: blocks look left only inside their tile,
  above-right never leaves the block, and the above-context arrays are
  per column. Each column decodes into its own reusable `FrameDecoder`
  workspace (`TilePool`). It touches only its own columns, so the arrays
  keep full-frame indexing and the decode code is unchanged.
* A scheduler hands out (column, superblock row) steps, least-advanced
  column first, preferring the column a thread had last (cache). The
  calling thread is one of the `threads`. After each step the row's
  rectangle is merged into the frame, and the loop filter runs each
  superblock row once every column has delivered it, top to bottom. That
  is the serial filter's order.
* Decoding never reads filtered pixels (the serial decoder filters only
  after all tiles), and workspaces keep their own unfiltered copy, so the
  filter overlaps decoding without changing a bit.
* Symbol counts are summed per tile (`Counts::add`) before backward
  adaptation. The order doesn't matter.
* The loop filter receives a 4-byte `LfCell` per 8x8 cell instead of the
  whole `ModeInfo`, which shrinks the per-row hand-over.
* **Single-tile-column frames stay serial by default.** The decode | filter
  pipeline works (`Vp9DecoderState::with_row_pipeline(true)`, covered by
  the tests). But the loop filter is only ~20% of the work, and the row
  hand-over cost about as much, so it measured break-even. FFmpeg's speed-up
  on such streams comes from VP9 *frame* threading. RMP doesn't have that
  yet, and it's the natural next step.

## Transcode pipeline

`DecodeStage` (decoders) and `EncodeStage` (converters, encoders, muxer)
are joined to the demuxer by bounded channels: 64 packets, then 8 decoded
items. Everything flows in input order, and errors travel down the line in
order too. So with the codecs' thread count held fixed, `run_pipelined`
writes the same bytes as `run_serial` and reports the same first error.
This needed `Send` on `Decoder`, `Encoder`, `Muxer` and `Io`. (`MemoryIo`
moved from `Rc<RefCell>` to `Arc<Mutex>`.)

One effect of frame threading: in outputs with several streams, the
*interleaving* of video and audio packets can shift by the decoder delay
between `-threads` values. Every stream's packets are identical, but RMP's
muxers write in arrival order. FFmpeg's `av_interleaved_write_frame` sorts
by DTS. An interleaving muxer queue would make files byte-identical across
thread counts.

## Efficiency vs FFmpeg

1080p30, 120 frames of `testsrc2` + noise. Decode to null (RMP:
`examples/decode in - -`; FFmpeg: `-f null -`). Times are medians of 3–5
interleaved runs, wall-clock; "cpu" is the total CPU time.

**Apple-silicon Linux VM, 4 cores, aarch64/NEON. FFmpeg 4.4.2.**

| Stream | RMP 1T | RMP 2T | RMP 4T | RMP speed-up | FFmpeg 1T | FFmpeg 2T | FFmpeg 4T | FFmpeg speed-up | RMP/FFmpeg at 4T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H.264 High (CABAC, B-frames) | 0.956 s | 0.585 s | **0.351 s** | **2.72x** | 0.524 s | 0.326 s | 0.220 s | 2.38x | 1.60x slower |
| H.264 Baseline (CAVLC, P only) | 0.550 s | 0.398 s | **0.223 s** | **2.47x** | 0.245 s | 0.152 s | 0.105 s | 2.33x | 2.12x slower |
| VP9, 4 tile columns | 0.708 s | 0.444 s | **0.306 s** | **2.31x** | 0.399 s | 0.283 s | 0.248 s | 1.61x | 1.23x slower |
| VP9, 1 tile column | 0.685 s | 0.681 s | 0.675 s | 1.0x | 0.391 s | 0.277 s | 0.242 s | 1.62x (frame threads) | 2.79x slower |

Transcodes on the same machine:

| Job | 1T | 4T | Notes |
| --- | --- | --- | --- |
| RMP H.264 → H.264 (RMP encoder, `-qp 26`) | 5.58 s | 4.64 s | encoder-bound: the pipeline hides decoding behind the single-threaded encoder. Same MD5 at 1, 2 and 4 threads. |
| RMP VP9 → Y4M | 0.84 s | 0.39 s | tile threads + pipeline (2.1x). |
| FFmpeg VP9 → Y4M | 0.53 s | 0.33 s | 1.6x |

**x86-64 cloud VM, 2 vCPUs (noisy neighbours), SSE2/AVX2. FFmpeg 6.1.1.**

| Stream | RMP 1T → 2T | speed-up | FFmpeg 1T → 2T | speed-up |
| --- | --- | --- | --- | --- |
| H.264 High | 3.28 → 2.25 s | 1.46x | 1.27 → 0.83 s | 1.53x |
| H.264 Baseline | 2.40 → 1.75 s | 1.37x | 0.89 → 0.64 s | 1.39x |
| VP9, 4 tiles | 1.99 → 1.28 s | 1.56x | 1.13 → 0.85 s | 1.32x |
| VP9, 1 tile | 1.80 → 1.85 s | 1.0x | 1.11 → 0.80 s | 1.40x |

**HEVC, x86-64 cloud VM, 2 vCPUs, AVX2, FFmpeg 6.1.1.** 1080p30 `testsrc2`
+ noise, x265 medium CRF 26 (B-pyramid, WPP), decode to null, best of 5.
The VM is noisy (±10%).

| Clip | RMP 1T | RMP 2T (2 frames) | RMP 2 rows | RMP speed-up | FFmpeg 1T → 2T | FFmpeg speed-up | RMP/FFmpeg at 2T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8-bit, 4.7 Mb/s, 150 frames | 1.94 s | **1.17 s** | 1.52 s | 1.66x | 0.90 → 0.68 s | 1.32x | 1.72x slower |
| 10-bit, 4.7 Mb/s | 2.03 s | **1.30 s** | 1.72 s | 1.56x | 1.33 → 0.81 s | 1.64x | 1.60x slower |
| 8-bit, 12.6 Mb/s | 4.20 s | **2.57 s** | | 1.63x | 2.78 → 1.54 s | 1.81x | 1.67x slower |
| 8-bit intra only, 60 frames | 1.13 s | **0.69 s** | 0.90 s | 1.64x | 0.66 → 0.46 s | 1.43x | 1.50x slower |

**HEVC, Apple-silicon Mac (macOS, arm64/NEON), 14 cores, FFmpeg 9.0.2.**
The same clips, `tools/bench_hevc.sh`, best of 5. "14T + rows" is 14 frame
threads plus 2 row threads per picture (the default from 8 threads);
"rows only" is one picture in flight with 14 row threads.

| Clip | RMP 1T | RMP 2T | RMP 4T | RMP 14T | RMP 14T + rows | RMP rows only | FFmpeg 1T | FFmpeg 2T | FFmpeg 4T | FFmpeg 14T | RMP/FFmpeg 1T | RMP/FFmpeg 14T (+ rows) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8-bit, 4.7 Mb/s | 0.78 s | 0.45 s | 0.34 s | 0.30 s | **0.28 s** | 0.43 s | 0.47 s | 0.29 s | 0.23 s | 0.19 s | 1.66x | 1.47x |
| 10-bit, 4.7 Mb/s | 0.89 s | 0.50 s | 0.37 s | 0.31 s | **0.28 s** | 0.54 s | 0.89 s | 0.50 s | 0.35 s | 0.25 s | 1.00x | 1.12x |
| 8-bit, 12.6 Mb/s | 2.27 s | 1.23 s | 0.91 s | 0.55 s | **0.47 s** | 0.69 s | 1.39 s | 0.81 s | 0.62 s | 0.36 s | 1.63x | 1.31x |
| 8-bit intra only, 60 frames | 0.55 s | 0.32 s | 0.21 s | 0.12 s | **0.11 s** | 0.31 s | 0.34 s | 0.21 s | 0.15 s | 0.11 s | 1.62x | 1.00x |

From 1 to 14 threads RMP speeds up 2.6–4.6x, against 2.5–3.9x for FFmpeg.
These clips are short: at 14 threads a run is 0.1–0.5 s, so start-up and
the stream's ramp-up and drain weigh a lot, and the 14 cores mix
performance and efficiency cores. The script now also measures 4 and 8
threads with rows, and a 600-frame clip (`long8.mp4`) for steadier
many-core numbers.

Before this work HEVC had no threads at all: 2.03 s, 2.04 s, 5.32 s and
1.11 s on two threads, so the gap at 2T was 2.5–3.5x. Total CPU rises by
about 6% with two frame threads (1.81 → 1.93 s on the first clip), and
by 15% (inter) to 50% (intra only) with two row threads.

### Reading the numbers

* **On 4 cores, scaling matches or beats FFmpeg's** for H.264 frame
  threading and VP9 tile threading. On the 2-vCPU box, H.264 scaling is
  within a few percent of FFmpeg's. Partly that's because RMP's single-threaded baseline is
  slower, so there is more work to spread. The absolute gap shrinks with
  threads: on H.264 High it goes from 1.82x at 1 thread to 1.60x at 4, and
  on 4-tile VP9 from 1.77x to 1.23x.
* **Threading overhead is higher than FFmpeg's.** Total CPU rises by 27% at
  4 threads for H.264 High (0.95 → 1.22 s), against 14% for FFmpeg. There
  are three costs. FFmpeg's frame threads share one buffer and need no
  `live` copy; RMP copies each finished row once. RMP takes an `RwLock`
  read for each motion-compensated block while the reference is still
  being decoded. And the `Condvar` wake-ups. The next steps are batching
  the waits per macroblock (FFmpeg's `await_references` per MB) and
  skipping the copy for pictures nobody references (non-reference B
  frames).
* For VP9, total CPU rises by 28% at 4 threads (0.71 → 0.91 s), against 15%
  for FFmpeg. The cost is the row hand-over: a workspace → frame copy of
  each superblock row, plus zeroing each workspace per frame.
* **Single-thread speed is still the main gap** (1.8–2.2x for H.264, ~1.8x
  for VP9). Threading doesn't change that. It's SIMD and entropy-decoding
  work (see `docs/SIMD.md`).
* **VP9 without tiles needs frame threading** to scale. That's the
  remaining gap against FFmpeg in this table.
* **HEVC scales like FFmpeg** on 2 vCPUs and on 14 Apple-silicon cores.
  What is left is the single-thread gap (1.6–2.2x for 8-bit; 10-bit is
  level with FFmpeg 9 on the Mac), see `docs/HEVC_AV1_HEAAC_AC3.md`.
