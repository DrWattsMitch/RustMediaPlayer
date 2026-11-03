//! MPEG-1/2/2.5 Audio Layer III decoder.
//!
//! Frame flow (one call to [`Mp3DecoderState::decode_frame`]):
//!
//! 1. parse header and side info,
//! 2. assemble main data from the bit reservoir + this frame's payload,
//! 3. per granule: scalefactors, Huffman/requantise, stereo, reorder,
//!    antialias, IMDCT ([`super::layer3`]), then synthesis ([`super::synth`]),
//! 4. keep up to 511 bytes of unused main data for the next frame.
//!
//! Output is interleaved signed 16-bit PCM.

use super::bits::BitReader;
use super::header::{FrameHeader, HEADER_SIZE};
use super::layer3::{self, GRANULE_SIZE, GranuleInfo, StreamFlags};
use super::synth::{self, QMF_STATE_LEN, SYNTH_SCRATCH_LEN};
use crate::core::{
	self, AudioFrame, AudioState, Channels, CodecPrivate, CodecState, DecodeOut, Frame, Packet,
	Parameters, SampleFormat, SampleLayout, State,
};
use crate::message::{Message, Result};

/// The reservoir can reference at most 511 bytes back (9-bit main_data_begin).
const MAX_RESERVOIR_BYTES: usize = 511;

/// Persistent decoder state (FFmpeg's `MPADecodeContext`).
#[derive(Debug, Clone)]
pub struct Mp3DecoderState {
	/// IMDCT overlap-add history, per channel (32 subbands × 9).
	mdct_overlap: [[f32; 9 * 32]; 2],
	/// Synthesis window history.
	qmf_state: [f32; QMF_STATE_LEN],
	/// Unused main data carried into the next frame.
	reservoir: Vec<u8>,
	/// Stream parameters of the last decoded frame; a change resets state.
	last_format: Option<(u32, u8)>,
	/// Frames that could not be decoded and were replaced by silence.
	pub concealed_frames: u64,
}

impl Default for Mp3DecoderState {
	fn default() -> Self {
		Self::new()
	}
}

/// Result of decoding one frame.
pub struct DecodedFrame {
	pub header: FrameHeader,
	/// Interleaved samples: `header.samples * channels` values.
	pub pcm: Vec<i16>,
	/// `false` when the frame was concealed with silence (missing reservoir
	/// data after a seek, or corrupt side info).
	pub complete: bool,
}

impl Mp3DecoderState {
	pub fn new() -> Self {
		Self {
			mdct_overlap: [[0.0; 9 * 32]; 2],
			qmf_state: [0.0; QMF_STATE_LEN],
			reservoir: Vec::with_capacity(MAX_RESERVOIR_BYTES),
			last_format: None,
			concealed_frames: 0,
		}
	}

	/// Clears all history, e.g. after a seek or a format change.
	pub fn reset(&mut self) {
		self.mdct_overlap = [[0.0; 9 * 32]; 2];
		self.qmf_state = [0.0; QMF_STATE_LEN];
		self.reservoir.clear();
		self.last_format = None;
	}

	/// Decodes the single frame at the start of `data`.
	///
	/// Errors only when `data` does not start with a complete Layer III
	/// frame. Damaged frame *contents* are concealed (silence) so playback
	/// timing stays intact, the same policy FFmpeg applies.
	pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedFrame> {
		let header =
			FrameHeader::parse(data).ok_or(Message::Codec("invalid mpeg audio frame header"))?;
		if header.layer != 3 {
			return Err(Message::Codec("only mpeg audio layer iii is supported"));
		}
		let frame = data.get(..header.frame_size).ok_or(Message::Codec("truncated mp3 frame"))?;

		let format = (header.sample_rate, header.channels());
		if self.last_format != Some(format) {
			self.reset();
			self.last_format = Some(format);
		}

		let flags = StreamFlags::from_header(&header);
		let nch = flags.channels;
		let granules = if flags.mpeg1 { 2 } else { 1 };
		let mut pcm = vec![0i16; header.samples as usize * nch];

		let mut bs = BitReader::new(&frame[HEADER_SIZE..]);
		if header.has_crc {
			bs.get_bits(16);
		}
		let mut side = [GranuleInfo::default(); 4];
		let Some(main_data_begin) =
			layer3::read_side_info(&mut bs, &mut side, &flags).filter(|_| bs.pos() <= bs.limit())
		else {
			self.reset();
			self.last_format = Some(format);
			self.concealed_frames += 1;
			return Ok(DecodedFrame { header, pcm, complete: false });
		};

		// Main data = tail of the reservoir + this frame's payload.
		let payload = frame.get(HEADER_SIZE + bs.pos() / 8..).unwrap_or(&[]);
		let from_reservoir = self.reservoir.len().min(main_data_begin);
		let mut main_data = Vec::with_capacity(from_reservoir + payload.len());
		main_data.extend_from_slice(&self.reservoir[self.reservoir.len() - from_reservoir..]);
		main_data.extend_from_slice(payload);
		let complete = self.reservoir.len() >= main_data_begin;

		let mut main = BitReader::new(&main_data);
		if complete {
			let mut grbuf = [0.0f32; 2 * GRANULE_SIZE];
			let mut scratch = [0.0f32; SYNTH_SCRATCH_LEN];
			// Scalefactor/intensity positions are shared by both granules
			// (granule 1 may reuse granule 0's scalefactors via scfsi).
			let mut ist_pos = [[0u8; 40]; 2];
			for gr in 0..granules {
				grbuf.fill(0.0);
				let info = &side[gr * nch..gr * nch + nch];
				self.decode_granule(&mut main, info, &flags, &mut grbuf, &mut ist_pos, &mut scratch);
				let out = &mut pcm[gr * GRANULE_SIZE * nch..(gr + 1) * GRANULE_SIZE * nch];
				synth::synth_granule(&mut self.qmf_state, &mut grbuf, nch, out, &mut scratch);
			}
		} else {
			self.concealed_frames += 1;
		}

		self.save_reservoir(&main_data, main.pos());
		Ok(DecodedFrame { header, pcm, complete })
	}

	fn decode_granule(
		&mut self,
		bs: &mut BitReader,
		info: &[GranuleInfo],
		flags: &StreamFlags,
		grbuf: &mut [f32; 2 * GRANULE_SIZE],
		ist_pos: &mut [[u8; 40]; 2],
		scratch: &mut [f32; SYNTH_SCRATCH_LEN],
	) {
		let mut scf = [0.0f32; 40];
		for (ch, gr) in info.iter().enumerate() {
			let limit = bs.pos() + usize::from(gr.part_23_length);
			layer3::decode_scalefactors(flags, &mut ist_pos[ch], bs, gr, &mut scf, ch);
			let dst = &mut grbuf[ch * GRANULE_SIZE..(ch + 1) * GRANULE_SIZE];
			layer3::huffman(dst, bs, gr, &scf, limit);
		}

		if flags.i_stereo {
			layer3::intensity_stereo(grbuf, &mut ist_pos[1], info, flags);
		} else if flags.ms_stereo {
			layer3::midside_all(grbuf);
		}

		for (ch, gr) in info.iter().enumerate() {
			let n_long_bands: usize =
				(if gr.mixed_block_flag { 2 } else { 0 }) << usize::from(flags.rate_index == 2);
			let channel = &mut grbuf[ch * GRANULE_SIZE..(ch + 1) * GRANULE_SIZE];
			let mut aa_bands = 31;
			if gr.n_short_sfb != 0 {
				aa_bands = n_long_bands.saturating_sub(1);
				let sfb = gr.sfb.get(usize::from(gr.n_long_sfb)..).unwrap_or(&[]);
				layer3::reorder(&mut channel[n_long_bands * 18..], &mut scratch[..], sfb);
			}
			layer3::antialias(channel, aa_bands);
			layer3::imdct_granule(channel, &mut self.mdct_overlap[ch], gr.block_type, n_long_bands);
			layer3::change_sign(channel);
		}
	}

	fn save_reservoir(&mut self, main_data: &[u8], bit_pos: usize) {
		let mut pos = bit_pos.div_ceil(8).min(main_data.len());
		let mut remains = main_data.len() - pos;
		if remains > MAX_RESERVOIR_BYTES {
			pos += remains - MAX_RESERVOIR_BYTES;
			remains = MAX_RESERVOIR_BYTES;
		}
		self.reservoir.clear();
		self.reservoir.extend_from_slice(&main_data[pos..pos + remains]);
	}
}

/// [`core::Decoder`] implementation: one or more MPEG audio frames per
/// packet in, one interleaved S16 [`AudioFrame`] per MPEG frame out.
#[derive(Debug, Clone, Default)]
pub struct Mp3Decoder;

impl Mp3Decoder {
	pub fn new() -> Self {
		Self
	}
}

impl core::Decoder for Mp3Decoder {
	fn create_state(&self, parameters: &Parameters) -> State {
		State::Audio(AudioState {
			sample_rate: parameters.sample_rate.unwrap_or(44_100),
			channels: Channels::from_count(parameters.channels.unwrap_or(2)),
			format: SampleFormat::S16,
			layout: SampleLayout::Interleaved,
			frame_size: 1152,
			codec_state: CodecState { extradata: Vec::new(), pts: None, dts: None, last_pts: None },
			samples_consumed: 0,
			samples_produced: 0,
			buffer: Vec::new(),
			private: CodecPrivate::Mp3(Box::default()),
		})
	}

	fn decode(&self, state: &mut State, packet: Packet) -> Result<DecodeOut<State>> {
		let frames = {
			let State::Audio(audio) = state else {
				return Err(Message::Codec("mp3 decoder requires an audio state"));
			};
			if !matches!(audio.private, CodecPrivate::Mp3(_)) {
				audio.private = CodecPrivate::Mp3(Box::default());
			}
			let CodecPrivate::Mp3(dec) = &mut audio.private else {
				return Err(Message::Codec("mp3 decoder state missing"));
			};
			let decoded = decode_packet(dec, &packet.data)?;

			let mut frames = Vec::with_capacity(decoded.len());
			let mut pts = packet.pts;
			for frame in decoded {
				let channels = frame.header.channels();
				let samples = u64::from(frame.header.samples);
				let mut bytes = Vec::with_capacity(frame.pcm.len() * 2);
				for sample in &frame.pcm {
					bytes.extend_from_slice(&sample.to_le_bytes());
				}
				let mut out = AudioFrame::new(
					bytes,
					frame.header.sample_rate,
					Channels::from_count(channels),
					SampleFormat::S16,
					SampleLayout::Interleaved,
				);
				out.pts = pts;
				// Later frames in a multi-frame packet follow contiguously.
				pts = pts.and_then(|t| {
					let step = core::Time::new(samples, core::TimeBase::new(1, frame.header.sample_rate));
					step.rescale(t.base()).and_then(|step| t.checked_add(step))
				});
				audio.sample_rate = frame.header.sample_rate;
				audio.channels = Channels::from_count(channels);
				audio.samples_produced += samples;
				frames.push(Frame::Audio(out));
			}
			audio.samples_consumed += packet.data.len() as u64;
			frames
		};
		Ok(DecodeOut { value: frames, state: state.clone() })
	}

	fn flush(&self, state: &mut State) -> Result<DecodeOut<State>> {
		if let State::Audio(audio) = state {
			if let CodecPrivate::Mp3(dec) = &mut audio.private {
				dec.reset();
			}
		}
		Ok(DecodeOut { value: Vec::new(), state: state.clone() })
	}
}

/// Decodes every frame in `data`, skipping leading or trailing junk that
/// does not form a valid frame (e.g. a stray ID3 tag in a raw stream).
pub fn decode_packet(dec: &mut Mp3DecoderState, data: &[u8]) -> Result<Vec<DecodedFrame>> {
	let mut out = Vec::new();
	let mut at = 0usize;
	while at + HEADER_SIZE <= data.len() {
		let rest = &data[at..];
		match FrameHeader::parse(rest) {
			Some(h) if h.layer == 3 && h.frame_size <= rest.len() => {
				out.push(dec.decode_frame(rest)?);
				at += h.frame_size;
			}
			_ => at += 1,
		}
	}
	if out.is_empty() && !data.is_empty() {
		return Err(Message::Codec("no mp3 frame found in packet"));
	}
	Ok(out)
}
