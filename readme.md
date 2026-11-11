# RMP (Rust Media Player)

RMP is a Rust-native multimedia toolkit that decodes, transforms, and
encodes audio and video without requiring FFmpeg. It provides both a
command-line interface and a library API, allowing you to process media files
directly from the terminal or integrate media processing into Rust applications.
The project aims to deliver a safe, deterministic, and modular media processing
engine with a focus on explicit pipelines and developer control.

RMP is a fork of [FFmpReg](https://github.com/yazaldefilimone/ffmpreg) by
Yazalde Filimone. It builds on FFmpReg's foundation and expands the set of
supported container formats and codecs, following the algorithms and
performance standards of FFmpeg, redesigned for Rust and written in safe Rust.

[Fork Notice](#fork-notice) | [License](/LICENSE) | [Upstream FFmpReg](https://github.com/yazaldefilimone/ffmpreg)

## Installation

Build and install the `rmp` command-line tool from this repository:

```bash
cargo install --git https://github.com/DrWattsMitch/RustMediaPlayer
# or, from a checkout:
cargo install --path .
```

This compiles `rmp` and places it in your Cargo bin directory. (The `rmp`
crate on crates.io is an unrelated MessagePack library, so
`cargo install rmp` doesn't install this tool.)

To use RMP as a library, add it as a Git dependency:

```toml
[dependencies]
rmp = { git = "https://github.com/DrWattsMitch/RustMediaPlayer" }
```

## Getting Started

`rmp` uses FFmpeg's command-line conventions: `-i` names the input, the
last argument is the output, and the output's extension picks the
container. Without an output, `rmp` describes the input:

```bash
rmp -i movie.mkv                   # duration, tags, streams
rmp -i movie.mkv --show            # ... and one line per packet
```

```
Input: movie.mkv
  Duration: 00:00:02 (2.000 s)
  Metadata:
    Title: MKV Song
  Stream #0: video h264, 64x48
  Stream #1: audio flac, 44100 Hz, 1 ch, 16-bit
    language: fra
  Stream #2: subtitle subrip
```

### Changing containers (stream copy)

When the output container can hold a stream's codec, `rmp` copies it
without re-encoding, like `ffmpeg -c copy`:

```bash
rmp -i input.mp4 output.mkv        # H.264 + AAC into Matroska
rmp -i input.mkv output.mp4        # and back (subtitles are dropped: MP4 can't hold them)
rmp -i input.mkv audio.mka         # audio and subtitles only
rmp -i clip.webm clip.opus         # WebM Opus → Ogg Opus, sample-exact
rmp -i voice.opus voice.webm       # Ogg Opus → WebM
rmp -i music.ogg music.mka         # Ogg Vorbis → Matroska audio
rmp -i song.flac song.oga          # FLAC → Ogg FLAC
rmp -i frames.y4m frames.mkv       # raw video into Matroska (V_UNCOMPRESSED)
rmp -i input.mkv -c copy out.webm  # WebM takes only VP8/VP9/AV1, Opus/Vorbis, WebVTT
```

### Decoding and re-encoding

RMP decodes PCM, ADPCM, MP3, AAC, H.264, VP9, FLAC, Opus and Vorbis, and encodes AAC-LC,
H.264, FLAC, PCM and raw video. Without `-c:a`/`-c:v`, each container gets a
sensible default (WAV → `pcm_s16le`, MP4/M4A/MOV → AAC and FLAC/Ogg/Matroska
→ FLAC when the source can't be copied, Y4M → `rawvideo`):

```bash
rmp -i song.mp3 song.flac                          # MP3 → FLAC
rmp -i song.m4a song.wav                           # AAC → 16-bit PCM WAV
rmp -i voice.opus voice.wav -c:a pcm_f32le         # Opus → 32-bit float WAV
rmp -i music.ogg music.flac                        # Vorbis → FLAC
rmp -i clip.webm frames.y4m                        # VP9 → raw YUV
rmp -i clip.webm clip.mp4                          # VP9 + Opus → H.264 + Opus in MP4
rmp -i song.mp3 song.ogg                           # MP3 → FLAC in Ogg
rmp -i input.mp4 output.mkv -c:a flac              # copy H.264, AAC → FLAC
rmp -i input.wav output.flac -compression_level 8  # FLAC level 0-12
rmp -i song.flac song.m4a -b:a 192k                # FLAC → AAC-LC
rmp -i input.mkv output.mp4 -c:a aac               # copy H.264, re-encode audio
```

H.264 encoding (Constrained Baseline: plays everywhere) takes a constant
quantiser or a bit rate:

```bash
rmp -i frames.y4m video.mp4 -c:v h264              # QP 23
rmp -i input.mp4 small.mp4 -s 640x360 -b:v 1M      # scale, re-encode at 1 Mb/s
rmp -i input.mkv out.mp4 -c:v h264 -qp 20 -g 60 -c:a aac -b:a 128k
```

Resampling, remixing and sample-format changes re-encode (as in FFmpeg):

```bash
rmp -i input.flac output.wav -ar 48000 -ac 1 -sample_fmt s32
rmp -i input.mp3 output.mka -ar 48000              # MP3 → 48 kHz FLAC in Matroska
```

Video is decoded to raw pictures, and can be scaled and converted:

```bash
rmp -i input.mp4 frames.y4m                        # H.264 → YUV4MPEG2
rmp -i input.mp4 small.y4m -s 320x180 -pix_fmt yuv444p -sws_flags lanczos
rmp -i input.mp4 -c:v rawvideo -an raw.mkv         # decoded video in Matroska
rmp -i frames.y4m big.mkv -s 1280x720              # scale raw video
```

### Choosing streams, tags and formats

```bash
rmp -i movie.mkv -vn soundtrack.flac               # drop video (-an: audio, -sn: subtitles)
rmp -i song.flac -metadata title="New Title" -metadata artist=Me out.oga
rmp -i song.flac -map_metadata -1 untagged.flac    # drop the input's tags
rmp -i input.wav -f flac output.audio              # container by name, not extension
rmp -y -i input.wav output.flac                    # overwrite (rmp refuses by default)
```

### Streaming and web playback (MP4)

```bash
rmp -i movie.mkv -c copy -movflags +faststart web.mp4      # moov first: plays while downloading
rmp -i movie.mkv -c copy -movflags frag_keyframe+empty_moov+default_base_moof live.mp4
                                                           # fragmented MP4 for MSE / HLS / DASH
rmp -i movie.mp4 -c copy -frag_duration 2000000 frag.mp4   # a fragment every 2 s
```

Fragmented output never seeks, so it also works when writing to a pipe.
Fragmented MP4 files (RMP's, FFmpeg's, DASH/CMAF segments joined into one
file) are read like any other MP4.

### Bit depth and dithering

Audio rounded to fewer bits (float or 24-bit sources to 16-bit WAV/FLAC,
any resample or remix to integers) is dithered with triangular (TPDF) noise
by default, so quiet passages and fades don't turn into distortion:

```bash
rmp -i song.opus song.wav                                  # TPDF dither (default)
rmp -i song.opus -dither_method lipshitz song.wav          # noise-shaped (44.1/48 kHz)
rmp -i song.opus -dither_method none song.wav              # plain rounding, like FFmpeg
```

### Batch processing

A glob (`*`, `?` in the file name) converts every matching file into an
output directory, several at a time:

```bash
rmp -i "recordings/*.wav" -f flac flac/            # recordings/a.wav → flac/a.flac
rmp -i "videos/*.mp4" -f mkv -j 4 mkv/             # 4 files at a time (default: CPU count)
rmp -i "music/*.flac"                              # describe each file
```

## Library

The CLI is a thin layer over the library. The whole pipeline is one call:

```rust
use rmp::transcode::{Options, transcode};

fn main() -> rmp::Result<()> {
    let mut options = Options::default();
    options.audio_codec = Some("flac".into());
    options.sample_rate = Some(48_000);
    let summary = transcode("input.mp4", "output.mka", &options)?;
    for line in &summary.streams {
        println!("{line}"); // e.g. "stream 1: audio aac -> flac (48000 Hz, 2 ch)"
    }
    Ok(())
}
```

`Options::parse` accepts the same FFmpeg-style arguments as the CLI.

The building blocks are the same ones the pipeline uses. A demuxer reads
packets from a container, a decoder turns packets into frames, `rmp::process`
converts frames, an encoder turns frames back into packets, and a muxer
writes them. Remuxing (stream copy) is a demuxer feeding a muxer:

```rust
use rmp::core::traits::*;
use rmp::io::{Input, Output};

fn main() -> rmp::Result<()> {
    let mut input = Input::open("input.mp4")?;
    let mut output = Output::create("output.mkv")?; // container from the extension
    output.set_metadata(input.metadata().clone())?;
    for stream in input.streams().iter() {
        output.add(stream)?;
    }
    while let Some(packet) = input.read()? {
        output.write(packet)?;
    }
    output.finish()
}
```

Decoding goes through the `Resolver`, which picks a decoder for a stream's
codec:

```rust
use rmp::core::traits::*;
use rmp::core::{Frame, Resolver};
use rmp::io::Input;

fn main() -> rmp::Result<()> {
    let mut input = Input::open("song.flac")?;
    let stream = input.streams().streams[0].clone();
    let decoder = Resolver::new().decoder_for(&stream).ok_or("no decoder")?;
    let mut state = decoder.create_state(&stream.parameters);
    let mut samples = 0;
    while let Some(packet) = input.read()? {
        for frame in decoder.decode(&mut state, packet)?.value {
            if let Frame::Audio(audio) = frame {
                samples += audio.nb_samples;
            }
        }
    }
    println!("{samples} samples per channel");
    Ok(())
}
```

## Core Concepts

Media processing revolves around two main data types: `Packet` and `Frame`.
A `Packet` is encoded data from a container, and a `Frame` is decoded audio
or video ready for processing or playback. Data flows predictably: demuxers
split containers into packets, decoders turn packets into frames, converters
in `rmp::process` change sample format, rate, layout, size or pixel format,
encoders turn frames back into packets, and muxers write packets into output
files.

```rust
pub struct Packet {
    pub stream_id: StreamId,
    pub data: Vec<u8>,
    pub pts: Option<Time>,      // each Time carries its own time base
    pub dts: Option<Time>,
    pub duration: Option<Time>,
    pub keyframe: bool,
}
```

A `Stream` describes one track: its kind (audio, video, subtitle), time
base, codec `Parameters` (codec ID using FFmpeg's names, sample rate,
channels, picture size, pixel format, `extradata`) and metadata. Gapless
audio is described by `encoder_delay` / `encoder_padding` stream metadata,
in samples, which every container that can store it reads and writes.

## Formats

Containers:

- **WAV**: PCM (8/16/24/32/64-bit integer, 32/64-bit float, A-law, µ-law)
  and IMA / Microsoft ADPCM. Demux and mux (PCM), with LIST/INFO tags.
- **Y4M (YUV4MPEG2)**: raw video in 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:4+alpha
  and grayscale, 8 to 16 bits. Demux and mux, byte-identical to FFmpeg.
- **MP3**: MPEG-1/2/2.5 audio streams with ID3v1/ID3v2 tags, Xing/Info/VBRI
  headers (exact duration, VBR seeking) and LAME gapless info. Demux and mux.
- **MP4 / M4A / MOV**: ISO BMFF with any number of tracks, B-frame timing,
  edit lists, keyframe seeking and iTunes metadata/cover art. Demux
  (including fragmented MP4), and mux for H.264, HEVC, AV1, MPEG-4, AAC,
  MP3, ALAC, Opus and FLAC, with `+faststart` or as fragmented MP4.
- **Matroska / WebM** (`.mkv`, `.mka`, `.webm`): demux of every track type,
  laced blocks, live streams with unknown sizes, cues or index-based seeking,
  tags, attachments (cover art) and subtitle tracks. Mux with SeekHead,
  cues, tags, attachments, Opus `CodecDelay`/`DiscardPadding`, raw video
  (`V_UNCOMPRESSED`), and WebM's codec rules; remuxes are packet-identical
  to the source in `ffprobe`.
- **FLAC** (`.flac`): demux (with seek table and bisection seeking) and mux
  (STREAMINFO with MD5, Vorbis comments, pictures).
- **Ogg** (`.ogg`, `.opus`, `.oga`): demux and mux of Opus, FLAC and Vorbis
  streams, with Vorbis comments and `METADATA_BLOCK_PICTURE` cover art. The
  muxer writes exact granule positions and FFmpeg's page layout.

Codecs:

- **PCM**: decoder and encoder for every layout above plus big-endian and
  signed 8-bit; the encoder converts from any sample format. Bit-exact with
  FFmpeg.
- **ADPCM**: IMA (`adpcm_ima_wav`) and Microsoft (`adpcm_ms`) decoders,
  bit-exact with FFmpeg.
- **Raw video**: passes planar YUV/gray frames through unchanged.
- **MP3 (MPEG audio Layer III)**: decoder to 16-bit PCM, within ±1 LSB of
  FFmpeg.
- **AAC-LC** and **H.264** (Baseline/Main/High, 8-bit 4:2:0) decoders.
- **AAC-LC encoder**: block switching, psychoacoustic model, M/S stereo,
  bit reservoir; mono to 7.1 at 8-96 kHz; bit rate within ~0.5% of the
  target and SNR on par with FFmpeg's native `aac` encoder.
- **H.264 encoder**: Constrained Baseline (CAVLC, I/P, Intra16x16/4x4,
  P_Skip and 16x16-8x8 partitions, quarter-sample motion, deblocking),
  constant QP or bit rate; FFmpeg decodes its output bit-exactly to the
  encoder's own reconstruction.
- **FLAC**: decoder (bit-exact with FFmpeg) and encoder (compression levels
  0–12, FFmpeg's defaults, files within ~3% of FFmpeg's size).
- **Opus**: decoder for SILK, hybrid and CELT packets, mono, stereo and
  multichannel (SILK bit-exact with libopus, CELT within float rounding).
- **Vorbis**: decoder for every Vorbis I stream (floor 0 and 1, residue
  types 0–2, channel coupling, up to 255 channels in FFmpeg's channel
  order), within one 16-bit step of FFmpeg's decoder. Ogg Vorbis packets
  get exact timestamps and durations from the granule positions.
- **VP9**: decoder for profiles 0 and 1 (8-bit 4:2:0, 4:2:2, 4:4:4 and
  sRGB): every frame type including hidden alt-refs, superframes and
  `show_existing_frame`, tiles, segmentation, compound prediction,
  lossless frames and backward probability adaptation. Bit-exact with
  libvpx / FFmpeg.

Processing (`rmp::process`), so files can be transcoded and not only
remuxed: sample-format conversion, resampling (polyphase windowed sinc),
channel remixing with FFmpeg's default matrices, and picture scaling (point,
bilinear, bicubic, Lanczos) with YUV ↔ RGB conversion (BT.601/709/2020,
limited or full range).

See [docs/WAV_Y4M.md](docs/WAV_Y4M.md), [docs/MP3_MP4.md](docs/MP3_MP4.md),
[docs/AAC_H264.md](docs/AAC_H264.md),
[docs/AAC_H264_ENCODING.md](docs/AAC_H264_ENCODING.md),
[docs/MKV_FLAC_OPUS_PROCESSING.md](docs/MKV_FLAC_OPUS_PROCESSING.md) and
[docs/MKV_OGG_MUXING_CLI.md](docs/MKV_OGG_MUXING_CLI.md),
[docs/OPUS_PLC.md](docs/OPUS_PLC.md) and
[docs/VORBIS_VP9.md](docs/VORBIS_VP9.md) for how each part
works and how to extend it. The examples show single features in a few
lines each:

```bash
cargo run --release --example probe       -- input.mp4
cargo run --release --example remux       -- input.y4m output.y4m
cargo run --release --example decode      -- input.wav audio.s16le -
cargo run --release --example pcm_convert -- input.wav output.wav pcm_s24le
cargo run --release --example mp3_to_wav  -- input.mp3 output.wav
cargo run --release --example transcode   -- input.mkv output.flac -ar 48000
cargo run --release --example transcode   -- input.opus output.wav -c:a pcm_f32le
cargo run --release --example framecrc    -- input.mkv   # like ffmpeg -f framecrc
```

Raw video is the `rawvideo` codec with its layout in `pixel_format`, and
each packet is one planar picture:

```rust
use rmp::core::traits::*;
use rmp::io::{Input, Output};

let mut input = Input::open("input.y4m")?;
let stream = input.streams().streams[0].clone();
println!("{}x{} {:?}", stream.parameters.width.unwrap_or(0),
	stream.parameters.height.unwrap_or(0), stream.parameters.pixel_format);
println!("frame rate: {}/{}", stream.time_base.den, stream.time_base.num);

let mut output = Output::create("output.y4m")?;
output.add(&stream)?;
while let Some(packet) = input.read()? {
	output.write(packet)?;
}
output.finish()?;
```

## CLI Reference

```text
rmp -i <input> [--show]                           describe the input
rmp [-y|-n] -i <input> [options] <output>         convert
rmp -i "<dir>/*.<ext>" [-f fmt] [-j N] [options] <out-dir>/   batch
```

| Option | Meaning |
| --- | --- |
| `-i <input>` | Input file, URL, or glob pattern (`*`, `?` in the file name) |
| `-o <output>` | Output file or directory (or give it as the last argument) |
| `-y` / `-n` | Overwrite / never overwrite an existing output (the default refuses) |
| `-j <jobs>` | Files converted at once in batch mode (default: CPU count) |
| `--show` | When describing an input, also list every packet |
| `-an`, `-vn`, `-sn` | Drop audio, video or subtitle streams |
| `-c:a <codec\|copy>` | Audio codec: `flac`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le`, `pcm_f32le`, `pcm_u8`, `pcm_alaw`, `pcm_mulaw`, ..., or `copy` |
| `-c:v <codec\|copy>` | Video codec: `rawvideo` or `copy` |
| `-c:s copy` | Copy subtitles (Matroska; WebVTT into WebM) |
| `-c copy` | Copy every stream |
| `-ar <rate>` | Output sample rate (polyphase resampler) |
| `-ac <channels>` | Output channel count (FFmpeg's default remix matrices) |
| `-sample_fmt <fmt>` | `u8`, `s16`, `s32`, `s64`, `flt`, `dbl` (add `p` for planar) |
| `-dither_method <m>` | Rounding to integer samples: `triangular` (default when precision is lost), `triangular_hp`, `rectangular`, `lipshitz` (noise shaped, 44.1/48 kHz), `none` |
| `-dither_scale <x>` | Dither noise level, 0-8 (default 1) |
| `-compression_level <n>` | FLAC level 0-12 (default 5) |
| `-s <W>x<H>` | Output picture size |
| `-pix_fmt <name>` | `yuv420p`, `yuv422p`, `yuv444p`, `yuv420p10le`, `gray`, `rgb24`, `nv12`, ... |
| `-sws_flags <filter>` | Scaler: `point`, `bilinear`, `bicubic` (default), `lanczos` |
| `-f <format>` | Output container by name: `wav`, `flac`, `mp3`, `mp4`, `m4a`, `mov`, `mkv`, `mka`, `webm`, `ogg`, `oga`, `opus`, `y4m` |
| `-movflags <flags>` | MP4 layout: `+faststart`; `frag_keyframe`, `frag_every_frame`, `empty_moov`, `default_base_moof`, `separate_moof`, `skip_trailer` (fragmented MP4) |
| `-frag_duration <µs>` | Fragmented MP4: cut a fragment every N microseconds |
| `-min_frag_duration <µs>` | Fragmented MP4: never cut a fragment shorter than this |
| `-metadata key=value` | Set a tag (`title`, `artist`, `album`, `album_artist`, `genre`, `date`, `comment`, `track`, `disc`, or any other key) |
| `-map_metadata -1` | Don't copy the input's tags |
| `-h`, `--help`, `--version` | Help and version |

Default codecs per output container (when `-c:a`/`-c:v` aren't given):

| Output | Audio | Video | Subtitles |
| --- | --- | --- | --- |
| `.wav` | `pcm_s16le` | dropped | dropped |
| `.flac` | `flac` | dropped | dropped |
| `.mp3` | copy (MP3 only) | dropped | dropped |
| `.y4m` | dropped | `rawvideo` | dropped |
| `.mp4`, `.mov` | copy if MP4 holds it, else `aac` | copy if MP4 holds it, else `h264` (VP9) | dropped |
| `.m4a` | copy if MP4 holds it, else `aac` | dropped | dropped |
| `.mkv` | copy if Matroska holds it, else `flac` | copy | copy |
| `.mka` | copy if Matroska holds it, else `flac` | dropped | copy |
| `.webm` | copy (Opus, Vorbis) | copy (VP8, VP9, AV1) | copy (WebVTT) |
| `.ogg`, `.oga` | copy (Opus, FLAC, Vorbis), else `flac` | dropped | dropped |
| `.opus` | copy (Opus only) | dropped | dropped |

`-ar`, `-ac` and `-sample_fmt` switch a copy default to a re-encode (AAC in
MP4/M4A/MOV, FLAC elsewhere); `-s` and `-pix_fmt` switch it to `h264` (to
`rawvideo` for Y4M, or when the pixel format isn't `yuv420p`), and so do
`-b:v` and `-qp`. RMP has no Opus, Vorbis, MP3 or VP9 encoder, so WebM,
`.opus` and `.mp3` output are stream copies; their decoders let those
streams go to any other format.

## Fork Notice

RMP is a fork of [FFmpReg](https://github.com/yazaldefilimone/ffmpreg),
created by Yazalde Filimone and released under the
[Apache 2.0](/LICENSE) License. The original copyright and license are
preserved, and this fork is distributed under the same terms.

FFmpReg set out to support WAV and Y4M containers and the PCM, ADPCM, and raw
video codecs. RMP completes those and extends them with additional file types
and codecs, starting with MP3, MP4, AAC, H.264, Matroska, FLAC, Ogg, Opus, Vorbis and VP9. If you find FFmpReg useful, please consider supporting the original
author on [Ko-fi](https://ko-fi.com/yazaldefilimone).

## Versioning

RMP uses `MAJOR.MINOR.PATCH`.

- **MAJOR**: incompatible API changes or major new features
- **MINOR**: new features compatible with existing APIs
- **PATCH**: bug fixes or small adjustments

Pre-1.0 releases are functional but APIs may change.
1.0 marks API stability and includes minimum beta features.
Release tags reflect tested versions; dev branches evolve faster.

## Goals

RMP aims to be a safe, reliable, and maintainable Rust-native alternative
to FFmpeg. The project emphasizes deterministic and explicit pipelines, clear
developer control, and minimal runtime surprises. All development prioritizes
safety by avoiding `unsafe` code wherever possible, minimizing `unwrap`
usage, reducing panics, and keeping external dependencies to a minimum. The
goal is to provide a high-performance media processing engine that is easy to
understand, integrate, and extend while maintaining strong Rust safety
guarantees.
