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

To install the command-line tool, run:

```bash
cargo install rmp
```

This downloads and compiles `rmp`, placing the binary in your Cargo bin
directory.

To use `rmp` as a library in your Rust project, add it to your `Cargo.toml`
dependencies:

```toml
[dependencies]
rmp = "0.2"
```

## Getting Started

The simplest operation is transcoding a file from one format to another. For
example, reading a WAV file and writing it back:

```bash
rmp -i input.wav -o output.wav
```

You can apply transforms to modify the media during processing. Transforms
execute in the order they appear and can perform operations such as gain
adjustment or normalization:

```bash
rmp -i input.wav -o output.wav --apply gain=2.0 --apply normalize
```

To inspect a media file without transcoding, use the `--show` flag. This prints
frame-level metadata, similar to `ffprobe`, including timing, sample count,
channel configuration, and sample rate:

```bash
rmp -i input.wav --show
```

Output example:

```
Frame 0: pts=0, samples=1024, channels=2, rate=44100
Frame 1: pts=1024, samples=1024, channels=2, rate=44100
```

Batch processing is supported using glob patterns. Each file is processed
independently, enabling parallel execution:

```bash
rmp -i "folder/*.wav" -o out/
```

## Library

The library exposes the same primitives used internally by the CLI. A pipeline
reads packets from a container, decodes them into frames, optionally applies
transforms, encodes the frames, and writes them to an output container.

Example WAV transcoding pipeline:

```rust
use rmp::container::{WavReader, WavWriter};
use rmp::codecs::{PcmDecoder, PcmEncoder};
use rmp::core::{Decoder, Encoder, Demuxer, Muxer, Timebase};
use std::fs::File;

fn main() -> std::io::Result<()> {
    let input = File::open("input.wav")?;
    let mut reader = WavReader::new(input)?;
    let format = reader.format();

    let output = File::create("output.wav")?;
    let mut writer = WavWriter::new(output, format)?;

    let mut decoder = PcmDecoder::new(format);
    let mut encoder = PcmEncoder::new(Timebase::new(1, format.sample_rate));

    while let Some(packet) = reader.read_packet()? {
        if let Some(frame) = decoder.decode(packet)? {
            if let Some(out_packet) = encoder.encode(frame)? {
                writer.write_packet(out_packet)?;
            }
        }
    }

    writer.finalize()?;
    Ok(())
}
```

## Core Concepts

Media processing revolves around two main data types: `Packet` and `Frame`. A
`Packet` represents encoded data in a container, while a `Frame` represents
decoded data ready for processing or playback. Data flows predictably: readers
demux containers into packets, decoders convert packets into frames, transforms
modify frames, encoders convert frames back into packets, and writers mux
packets into output files.

Example `Frame` structure:

```rust
pub struct Frame {
    pub data: Vec<u8>,
    pub pts: i64,
    pub timebase: Timebase,
    pub sample_rate: u32,
    pub channels: u8,
    pub nb_samples: usize,
}
```

Example `Packet` structure:

```rust
pub struct Packet {
    pub data: Vec<u8>,
    pub pts: i64,
    pub dts: i64,
    pub timebase: Timebase,
    pub stream_index: usize,
}
```

## Transforms

Transforms modify frames in the pipeline. Built-in transforms include `Gain` for
multiplying sample amplitudes and `Normalize` for scaling samples to a target
peak.

Example of applying transforms:

```rust
use rmp::transform::{Gain, Normalize};
use rmp::core::Transform;

let mut gain = Gain::new(2.0);
let mut normalize = Normalize::new(0.95);

let frame = decoder.decode(packet)?.unwrap();
let frame = gain.apply(frame)?;
let frame = normalize.apply(frame)?;
```

Custom transforms can be implemented by the `Transform` trait:

```rust
pub struct Invert;

impl Transform for Invert {
    fn apply(&mut self, mut frame: Frame) -> Result<Frame> {
        for i in (0..frame.data.len()).step_by(2) {
            let sample = i16::from_le_bytes([frame.data[i], frame.data[i + 1]]);
            let inverted = (-sample).to_le_bytes();
            frame.data[i] = inverted[0];
            frame.data[i + 1] = inverted[1];
        }
        Ok(frame)
    }

    fn name(&self) -> &'static str {
        "invert"
    }
}
```

CLI transforms are specified as `name=value` pairs:

```bash
rmp -i input.wav -o output.wav --apply gain=1.5 --apply normalize=0.9
```

## Formats

Containers:

- **WAV**: uncompressed audio, suitable for lossless pipelines.
- **Y4M**: raw video frames with a text-based header.
- **MP3**: MPEG-1/2/2.5 audio streams with ID3v1/ID3v2 tags, Xing/Info/VBRI
  headers (exact duration, VBR seeking) and LAME gapless info. Demux and mux.
- **MP4 / M4A / MOV**: ISO BMFF with any number of tracks, B-frame timing,
  edit lists, keyframe seeking and iTunes metadata/cover art. Demux, and mux
  for H.264, HEVC, AV1, MPEG-4, AAC, MP3, ALAC, Opus and FLAC.

Codecs:

- **PCM**: uncompressed 16-bit audio, multi-channel support.
- **ADPCM**: adaptive differential audio compression.
- **Raw video**: passes through YUV frame data unchanged.
- **MP3 (MPEG audio Layer III)**: decoder to 16-bit PCM, within ±1 LSB of
  FFmpeg.

See [docs/MP3_MP4.md](docs/MP3_MP4.md) for how MP3/MP4 support works and how to
extend it. Try it with the examples:

```bash
cargo run --release --example probe      -- input.mp4
cargo run --release --example remux      -- input.mp4 output.m4a
cargo run --release --example mp3_to_wav -- input.mp3 output.wav
```

Video processing follows the same pipeline as audio:

```rust
use rmp::container::{Y4mReader, Y4mWriter};
use rmp::codecs::{RawVideoDecoder, RawVideoEncoder};

let input = File::open("input.y4m")?;
let mut reader = Y4mReader::new(input)?;
let output = File::create("output.y4m")?;
let mut writer = Y4mWriter::new(output, reader.format().clone())?;
```

Format metadata is accessible via the reader:

```rust
let format = reader.format();
println!("Resolution: {}x{}", format.width, format.height);
println!("Frame rate: {}/{}", format.framerate_num, format.framerate_den);
```

## CLI Reference

- `-i`: input file or glob pattern.
- `-o`: output file or directory.
- `--show`: inspection mode, prints frame metadata.
- `--apply`: add transform to pipeline (multiple allowed).
- `--codec`: select output codec (default matches input).

## Fork Notice

RMP is a fork of [FFmpReg](https://github.com/yazaldefilimone/ffmpreg),
created by Yazalde Filimone and released under the
[Apache 2.0](/LICENSE) License. The original copyright and license are
preserved, and this fork is distributed under the same terms.

FFmpReg supports WAV and Y4M containers and the PCM, ADPCM, and raw video
codecs. RMP extends that with additional file types and codecs, starting with
MP3 and MP4. If you find FFmpReg useful, please consider supporting the original
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
