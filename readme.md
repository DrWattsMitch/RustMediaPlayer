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

- **WAV**: PCM (8/16/24/32/64-bit integer, 32/64-bit float, A-law, µ-law)
  and IMA / Microsoft ADPCM. Demux and mux (PCM), with LIST/INFO tags.
- **Y4M (YUV4MPEG2)**: raw video in 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:4+alpha
  and grayscale, 8 to 16 bits. Demux and mux, byte-identical to FFmpeg.
- **MP3**: MPEG-1/2/2.5 audio streams with ID3v1/ID3v2 tags, Xing/Info/VBRI
  headers (exact duration, VBR seeking) and LAME gapless info. Demux and mux.
- **MP4 / M4A / MOV**: ISO BMFF with any number of tracks, B-frame timing,
  edit lists, keyframe seeking and iTunes metadata/cover art. Demux, and mux
  for H.264, HEVC, AV1, MPEG-4, AAC, MP3, ALAC, Opus and FLAC.

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

See [docs/WAV_Y4M.md](docs/WAV_Y4M.md), [docs/MP3_MP4.md](docs/MP3_MP4.md)
and [docs/AAC_H264.md](docs/AAC_H264.md) for how each part works and how to
extend it. Try it with the examples:

```bash
cargo run --release --example probe       -- input.mp4
cargo run --release --example remux       -- input.y4m output.y4m
cargo run --release --example decode      -- input.wav audio.s16le -
cargo run --release --example pcm_convert -- input.wav output.wav pcm_s24le
cargo run --release --example mp3_to_wav  -- input.mp3 output.wav
```

Video uses the same pipeline as audio. A Y4M stream is `rawvideo` with its
layout in `pixel_format`, and each packet is one planar picture:

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

FFmpReg set out to support WAV and Y4M containers and the PCM, ADPCM, and raw
video codecs. RMP completes those and extends them with additional file types
and codecs, starting with MP3, MP4, AAC and H.264. If you find FFmpReg useful, please consider supporting the original
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
