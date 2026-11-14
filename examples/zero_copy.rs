//! Zero-copy frame hand-off, the way a player would feed a GPU.
//!
//! cargo run --release --example zero_copy -- input.mp4
//!
//! The decoder thread sends each frame to a "render" thread over a channel.
//! Sending moves a handful of plane references, not pixels. The render
//! thread reads every plane in place, at the decoder's own stride, which is
//! the shape every GPU upload API takes:
//!
//! ```text
//! // wgpu
//! queue.write_texture(texture_for(i), plane.bytes(),
//!     wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(plane.stride() as u32),
//!                                   rows_per_image: None }, extent(plane));
//! // OpenGL
//! glPixelStorei(GL_UNPACK_ROW_LENGTH, plane.stride());
//! glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, plane.bytes().as_ptr());
//! ```
//!
//! The planes stay valid for as long as the frame (or a cloned
//! `plane.buffer()`) is held, even after the decoder has moved on or been
//! dropped. When the last reference goes, the decoder's buffer pool gets
//! the memory back.
use std::sync::mpsc;

use rmp::Result;
use rmp::core::traits::*;
use rmp::core::{Frame, Resolver, StreamKind, VideoFrame};
use rmp::io::Input;

/// Stands in for a texture upload: touches every visible row in place.
fn upload(frame: &VideoFrame) -> u64 {
	let mut sum = 0u64;
	for plane in frame.planes() {
		// `plane.bytes()` + `plane.stride()` is what goes to the GPU.
		for row in plane.rows_iter() {
			sum = sum.wrapping_add(row.iter().map(|&b| u64::from(b)).sum::<u64>());
		}
	}
	sum
}

pub fn main() -> Result<()> {
	let path = std::env::args().nth(1).ok_or("usage: zero_copy <input>")?;
	let mut demuxer = Input::open(&path)?;
	let stream = demuxer
		.streams()
		.iter()
		.find(|s| s.kind == StreamKind::Video)
		.cloned()
		.ok_or("no video stream")?;
	let decoder = Resolver::new().decoder_for(&stream).ok_or("no decoder for the video stream")?;
	let mut state = decoder.create_state(&stream.parameters);

	// A bounded queue like a player's present queue: the decoder runs at
	// most 4 frames ahead of the renderer.
	let (tx, rx) = mpsc::sync_channel::<VideoFrame>(4);
	let render = std::thread::spawn(move || {
		let (mut frames, mut checksum) = (0u64, 0u64);
		for frame in rx {
			if frames == 0 {
				for (i, p) in frame.planes().enumerate() {
					println!(
						"plane {i}: {}x{} bytes, stride {}, 16-byte aligned: {}",
						p.row_bytes(),
						p.rows(),
						p.stride(),
						p.is_aligned(16)
					);
				}
			}
			checksum = checksum.wrapping_add(upload(&frame));
			frames += 1;
		}
		(frames, checksum)
	});

	let send = |frames: Vec<Frame>| {
		for f in frames {
			if let Frame::Video(v) = f {
				// Moves plane references only; the samples stay where the
				// decoder wrote them.
				let _ = tx.send(v);
			}
		}
	};
	while let Some(packet) = demuxer.read()? {
		if packet.stream_id == stream.id {
			send(decoder.decode(&mut state, packet)?.value);
		}
	}
	send(decoder.flush(&mut state)?.value);
	drop(tx);
	let (frames, checksum) = render.join().map_err(|_| "render thread panicked")?;
	println!("rendered {frames} frames (checksum {checksum:#x})");
	Ok(())
}
