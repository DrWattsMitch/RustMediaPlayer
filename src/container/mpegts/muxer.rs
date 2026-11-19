//! MPEG transport stream muxer (ISO/IEC 13818-1), after FFmpeg's
//! `mpegtsenc.c`, whose defaults it follows so its files look like
//! `ffmpeg -f mpegts` output:
//!
//! * one program (service 1, PMT on PID 0x1000), streams on PIDs 0x100,
//!   0x101, ... in the order they are added;
//! * PAT and PMT every 100 ms and before every video keyframe, SDT every
//!   500 ms (service "Service01" from provider "RMP", or the
//!   `service_name` / `service_provider` tags);
//! * timestamps start at 1.4 s (twice FFmpeg's default 0.7 s `muxdelay`)
//!   and the PCR, carried by the first video stream (else the first audio
//!   stream) in the first packet of each of its PES, runs 0.7 s ahead of
//!   decode times, as in FFmpeg's variable-rate mode;
//! * one PES per video access unit (with `PES_packet_length` 0), audio
//!   frames gathered into PES of up to 2930 bytes or 0.7 s.
//!
//! Packets are converted to what TS carries ([`Payload`]): length-prefixed
//! H.264 / HEVC (from MP4 or Matroska) becomes Annex B with the parameter
//! sets before each keyframe that lacks them and an access unit
//! delimiter in front (FFmpeg's `h264_mp4toannexb` + `mpegtsenc`); raw
//! AAC gets ADTS headers; Opus gets its control headers. MPEG audio,
//! AC-3, E-AC-3, MPEG-2 and MPEG-4 video pass as they are.
//!
//! [`TsMuxer::set_sample_aes`] and [`TsMuxer::set_cipher`] encrypt H.264
//! and AAC samples for HLS SAMPLE-AES; [`TsMuxer::flush`] and
//! [`TsMuxer::resend_headers`] let the HLS muxer cut segments.

use super::es;
use super::packet::{PID_PAT, PID_SDT, SYNC, TS_SIZE};
use super::psi::crc32;
use crate::codecs::aac::config::parse_audio_specific_config;
use crate::codecs::h264::nal::{split_annex_b, split_length_prefixed};
use crate::codecs::opus::{OpusHead, packet_samples};
use crate::container::bsf::annex_b_from_config;
use crate::container::hls::sample_aes::{SampleCipher, SampleCodec, encrypt_packet};
use crate::core::{
	Metadata, Muxer, Packet, RawValue, Stream, StreamId, StreamKind, Time, TimeBase,
};
use crate::io::Io;
use crate::message::{Message, Result};

const TS_TIME_BASE: TimeBase = TimeBase::new(1, 90_000);
/// Transport packets are written to the output in blocks of this size
/// (348 packets), not one system call each.
const WRITE_BLOCK: usize = 348 * TS_SIZE;
/// Video packets without decode times are held this long to derive them.
const REORDER_WINDOW: usize = 16;

/// What the muxer can be told, by FFmpeg's option names.
#[derive(Debug, Clone, PartialEq)]
pub struct TsOptions {
	/// `mpegts_transport_stream_id`.
	pub transport_stream_id: u16,
	/// `mpegts_original_network_id`.
	pub original_network_id: u16,
	/// `mpegts_service_id`.
	pub service_id: u16,
	/// `mpegts_pmt_start_pid`.
	pub pmt_pid: u16,
	/// `mpegts_start_pid`: the first stream's PID.
	pub start_pid: u16,
	/// `pes_payload_size`: audio PES are cut near this size (rounded as
	/// FFmpeg rounds it, to fill whole TS packets).
	pub pes_payload_size: usize,
	/// `pat_period`, `sdt_period` in seconds.
	pub pat_period: f64,
	pub sdt_period: f64,
	/// `muxdelay` in seconds: the PCR lead, and half the start offset.
	pub mux_delay: f64,
}

impl Default for TsOptions {
	fn default() -> Self {
		Self {
			transport_stream_id: 1,
			original_network_id: 0xFF01,
			service_id: 1,
			pmt_pid: 0x1000,
			start_pid: 0x100,
			pes_payload_size: 2930,
			pat_period: 0.1,
			sdt_period: 0.5,
			mux_delay: 0.7,
		}
	}
}

/// Every option name [`TsOptions::set`] knows.
pub const TS_OPTIONS: [&str; 9] = [
	"mpegts_transport_stream_id",
	"mpegts_original_network_id",
	"mpegts_service_id",
	"mpegts_pmt_start_pid",
	"mpegts_start_pid",
	"pes_payload_size",
	"pat_period",
	"sdt_period",
	"muxdelay",
];

impl TsOptions {
	/// Sets an option by its FFmpeg name; `Ok(false)` for an unknown name.
	pub fn set(&mut self, key: &str, value: &str) -> Result<bool> {
		let int = |v: &str, max: u64| {
			let v = v.trim();
			let n = match v.strip_prefix("0x") {
				Some(hex) => u64::from_str_radix(hex, 16).ok(),
				None => v.parse::<u64>().ok(),
			};
			n.filter(|&n| n <= max).ok_or(Message::Container("invalid mpegts option value"))
		};
		let seconds = |v: &str| {
			v.trim()
				.parse::<f64>()
				.ok()
				.filter(|s| s.is_finite() && *s >= 0.0 && *s <= 100.0)
				.ok_or(Message::Container("invalid mpegts period"))
		};
		match key {
			"mpegts_transport_stream_id" => self.transport_stream_id = int(value, 0xFFFF)? as u16,
			"mpegts_original_network_id" => self.original_network_id = int(value, 0xFFFF)? as u16,
			"mpegts_service_id" => self.service_id = int(value, 0xFFFF)? as u16,
			"mpegts_pmt_start_pid" => {
				let pid = int(value, 0x1FFE)? as u16;
				if pid < 0x10 {
					return Err(Message::Container("mpegts_pmt_start_pid must be 0x0010-0x1ffe"));
				}
				self.pmt_pid = pid;
			}
			"mpegts_start_pid" => {
				let pid = int(value, 0x1FFE)? as u16;
				if pid < 0x20 {
					return Err(Message::Container("mpegts_start_pid must be 0x0020-0x1ffe"));
				}
				self.start_pid = pid;
			}
			"pes_payload_size" => self.pes_payload_size = int(value, 1 << 20)? as usize,
			"pat_period" => self.pat_period = seconds(value)?,
			"sdt_period" => self.sdt_period = seconds(value)?,
			"muxdelay" => self.mux_delay = seconds(value)?,
			_ => return Ok(false),
		}
		Ok(true)
	}
}

/// How a stream's packets become TS payloads.
#[derive(Debug, Clone)]
enum Payload {
	/// As they are.
	Copy,
	/// H.264 / HEVC. `length_size` is set for length-prefixed input;
	/// `parameter_sets` (Annex B) go before keyframes that don't carry them.
	Video { hevc: bool, length_size: Option<usize>, parameter_sets: Vec<u8> },
	/// Raw AAC: an ADTS header is added (`[profile, rate index, channel
	/// configuration]` from the AudioSpecificConfig).
	Adts([u8; 3]),
	/// Opus with a control header; the pre-skip still to signal as start
	/// trim, and the end trim for the last packet.
	Opus { pending_start_trim: u32, end_trim: u32 },
}

type HeldOpus = (Vec<u8>, Option<i64>, Option<i64>, Option<u64>);

struct TsStream {
	id: StreamId,
	pid: u16,
	kind: StreamKind,
	stream_type: u8,
	/// PES `stream_id`.
	pes_id: u8,
	descriptors: Vec<u8>,
	time_base: TimeBase,
	continuity: u8,
	payload: Payload,
	/// SAMPLE-AES: how this stream's samples are encrypted.
	encrypted: Option<SampleCodec>,
	/// Audio gathered for the next PES, its first times and the Opus
	/// samples in it.
	pending: Vec<u8>,
	pending_pts: Option<i64>,
	pending_dts: Option<i64>,
	pending_samples: u32,
	/// The latest Opus packet (data, pts, dts, duration in 48 kHz samples),
	/// held so the last one can carry the end trim.
	opus_held: Option<HeldOpus>,
	/// Video packets without dts, waiting for their decode times.
	reorder: Vec<Packet>,
	reorder_delay: Option<usize>,
	reordered: u64,
	/// Presentation times not yet used as decode times, sorted (stream
	/// ticks).
	pts_window: Vec<i64>,
	last_reordered: Option<i64>,
	/// Video: decode times are moved back by this much (stream ticks) so
	/// none is after its presentation time, as TS requires (RMP's MP4
	/// demuxer applies the edit list to presentation times only). Decided
	/// from the first frames.
	dts_shift: Option<i64>,
	last_dts: Option<i64>,
}

pub struct TsMuxer {
	io: Box<dyn Io>,
	metadata: Metadata,
	options: TsOptions,
	streams: Vec<TsStream>,
	started: bool,
	pat_cc: u8,
	pmt_cc: u8,
	sdt_cc: u8,
	/// Index of the stream carrying the PCR.
	pcr: Option<usize>,
	last_pat: Option<i64>,
	last_sdt: Option<i64>,
	force_tables: bool,
	/// Decode time (90 kHz) mapped to the start offset.
	origin: Option<i64>,
	sample_aes: bool,
	cipher: Option<SampleCipher>,
	/// Bytes written so far.
	written: u64,
	/// Packets received before every video stream's decode time shift (and
	/// so the origin) is known, in arrival order.
	startup: Option<Vec<(usize, Packet)>>,
	buffer: Vec<u8>,
}

impl TsMuxer {
	pub fn new(io: Box<dyn Io>) -> Self {
		Self::with_options(io, TsOptions::default())
	}

	pub fn with_options(io: Box<dyn Io>, options: TsOptions) -> Self {
		Self {
			io,
			metadata: Metadata::default(),
			options,
			streams: Vec::new(),
			started: false,
			pat_cc: 0,
			pmt_cc: 0,
			sdt_cc: 0,
			pcr: None,
			last_pat: None,
			last_sdt: None,
			force_tables: true,
			origin: None,
			sample_aes: false,
			cipher: None,
			written: 0,
			startup: Some(Vec::new()),
			buffer: Vec::with_capacity(WRITE_BLOCK),
		}
	}

	/// The start offset in 90 kHz ticks: what time 0 of the input becomes.
	pub fn start_offset(&self) -> u64 {
		(self.options.mux_delay * 2.0 * 90_000.0).round() as u64
	}

	/// The input time (s) written at the start offset, once known: the
	/// HLS muxer places its subtitle cues on the same clock.
	pub fn origin(&self) -> Option<f64> {
		self.origin.map(|t| t as f64 / 90_000.0)
	}

	fn delay(&self) -> i64 {
		(self.options.mux_delay * 90_000.0).round() as i64
	}

	/// Marks H.264 and AAC streams as SAMPLE-AES encrypted (their stream
	/// types become 0xDB / 0xCF, with Apple's descriptors). Must be called
	/// before the first packet; packets are then encrypted with the cipher
	/// given to [`Self::set_cipher`].
	pub fn set_sample_aes(&mut self, on: bool) -> Result<()> {
		if self.started {
			return Err(Message::Container("mpegts muxer: SAMPLE-AES must be set before writing"));
		}
		self.sample_aes = on;
		Ok(())
	}

	/// The key and IV for the packets written from now on (HLS changes the
	/// IV with each segment).
	pub fn set_cipher(&mut self, cipher: Option<SampleCipher>) {
		self.cipher = cipher;
	}

	/// Writes PAT, PMT and SDT before the next packet (each HLS segment
	/// starts with them).
	pub fn resend_headers(&mut self) {
		self.force_tables = true;
	}

	/// Writes the audio gathered so far (and a held Opus packet), so the
	/// output ends on a whole PES: called at the end of each HLS segment.
	pub fn flush(&mut self) -> Result<()> {
		self.release_startup()?;
		for i in 0..self.streams.len() {
			self.release_reorder(i, true)?;
			self.release_opus(i, false)?;
			self.flush_pending(i)?;
		}
		self.drain()
	}

	/// Bytes written so far.
	pub fn bytes_written(&self) -> u64 {
		self.written
	}

	fn stream_setup(&self, stream: &Stream, index: usize) -> Result<TsStream> {
		let p = &stream.parameters;
		let codec = p.codec.id.clone();
		let mut descriptors = Vec::new();
		let encrypted = if self.sample_aes { SampleCodec::of(&codec) } else { None };
		// Apple's SAMPLE-AES covers H.264 and AAC here; other streams stay
		// in the clear (their own stream types say so).
		let encrypted = encrypted.filter(|c| *c != SampleCodec::Ac3);
		let mut payload = Payload::Copy;
		let (stream_type, pes_id) = match codec.as_str() {
			"h264" | "hevc" => {
				let hevc = codec == "hevc";
				let (length_size, parameter_sets) = match annex_b_from_config(&codec, &p.extradata) {
					Some((size, sets)) => (Some(size), sets),
					None => (None, p.extradata.clone()),
				};
				payload = Payload::Video { hevc, length_size, parameter_sets };
				match (hevc, encrypted.is_some()) {
					(false, false) => (0x1B, 0xE0),
					(false, true) => {
						descriptors.extend_from_slice(&[0x0F, 4]);
						descriptors.extend_from_slice(b"zavc");
						(0xDB, 0xE0)
					}
					(true, _) => (0x24, 0xE0),
				}
			}
			"mpeg2video" => (0x02, 0xE0),
			"mpeg4" => (0x10, 0xE0),
			"aac" => {
				if !p.extradata.is_empty() {
					let cfg = parse_audio_specific_config(&p.extradata)
						.ok_or(Message::Container("mpegts muxer: invalid AAC AudioSpecificConfig"))?;
					let channel_config = match cfg.channels {
						c @ 1..=6 => c,
						8 => 7,
						_ => {
							return Err(Message::Container(
								"mpegts muxer: AAC channel layout has no ADTS channel configuration",
							));
						}
					};
					if !(1..=4).contains(&cfg.object_type) {
						return Err(Message::Container("mpegts muxer: AAC object type has no ADTS profile"));
					}
					payload =
						Payload::Adts([cfg.object_type - 1, cfg.sample_rate_index as u8, channel_config]);
				}
				if encrypted.is_some() {
					descriptors.extend_from_slice(&[0x0F, 4]);
					descriptors.extend_from_slice(b"aacd");
					// registration 'apad' + audio_setup_information: type,
					// priming, version, the AudioSpecificConfig.
					let asc = &p.extradata;
					let len = (4 + 4 + 2 + 1 + 1 + asc.len()).min(255);
					descriptors.extend_from_slice(&[0x05, len as u8]);
					descriptors.extend_from_slice(b"apad");
					descriptors.extend_from_slice(b"zaac");
					descriptors.extend_from_slice(&[0, 0, 1, asc.len().min(243) as u8]);
					descriptors.extend_from_slice(&asc[..asc.len().min(243)]);
					(0xCF, 0xC0)
				} else {
					(0x0F, 0xC0)
				}
			}
			"mp2" | "mp3" | "mp1" => (
				if p.sample_rate.unwrap_or(0) >= 32_000 || p.sample_rate.is_none() { 0x03 } else { 0x04 },
				0xC0,
			),
			"ac3" => (0x81, 0xBD),
			"eac3" => (0x87, 0xBD),
			"opus" => {
				let head = if p.extradata.is_empty() {
					Some(OpusHead::new(p.channels.unwrap_or(2), 0, 48_000))
				} else {
					OpusHead::parse(&p.extradata)
				};
				let head = head.ok_or(Message::Container("mpegts muxer: invalid Opus extradata"))?;
				let code = es::opus_channel_code(&head).ok_or(Message::Container(
					"mpegts muxer: this Opus channel mapping can't be signalled in MPEG-TS",
				))?;
				descriptors.extend_from_slice(&[0x05, 4]);
				descriptors.extend_from_slice(b"Opus");
				descriptors.extend_from_slice(&[0x7F, 2, 0x80, code]);
				let end_trim = match stream.metadata.raw.get("encoder_padding") {
					Some(RawValue::String(v)) => v.parse::<u32>().unwrap_or(0),
					_ => 0,
				};
				payload = Payload::Opus { pending_start_trim: u32::from(head.pre_skip), end_trim };
				(0x06, 0xBD)
			}
			_ => return Err(Message::Container("mpegts muxer: unsupported codec")),
		};
		if stream.kind == StreamKind::Audio
			&& let Some(RawValue::String(lang)) = stream.metadata.raw.get("language")
		{
			let lang = lang.trim().to_ascii_lowercase();
			if lang.len() == 3 && lang.bytes().all(|b| b.is_ascii_lowercase()) {
				descriptors.extend_from_slice(&[0x0A, 4]);
				descriptors.extend_from_slice(lang.as_bytes());
				descriptors.push(0); // audio_type: undefined
			}
		}
		let pid = self
			.options
			.start_pid
			.checked_add(index as u16)
			.filter(|&pid| pid < 0x1FFF && pid != self.options.pmt_pid)
			.ok_or(Message::Container("mpegts muxer: out of PIDs"))?;
		Ok(TsStream {
			id: stream.id,
			pid,
			kind: stream.kind,
			stream_type,
			pes_id,
			descriptors,
			time_base: stream.time_base,
			continuity: 0,
			payload,
			encrypted,
			pending: Vec::new(),
			pending_pts: None,
			pending_dts: None,
			pending_samples: 0,
			opus_held: None,
			reorder: Vec::new(),
			reorder_delay: None,
			reordered: 0,
			pts_window: Vec::new(),
			last_reordered: None,
			dts_shift: if stream.kind == StreamKind::Video { None } else { Some(0) },
			last_dts: None,
		})
	}

	// --- Tables -----------------------------------------------------------

	/// Writes a PSI section on `pid`, splitting it over as many packets as
	/// it takes (pointer field 0, 0xFF stuffing).
	fn write_section(&mut self, pid: u16, section: &[u8], cc: CounterOf) -> Result<()> {
		let mut rest = section;
		let mut first = true;
		while first || !rest.is_empty() {
			let mut p = [0xFFu8; TS_SIZE];
			let counter = self.counter(cc);
			p[0] = SYNC;
			p[1] = (u8::from(first) << 6) | (pid >> 8) as u8;
			p[2] = pid as u8;
			p[3] = 0x10 | counter;
			let mut at = 4;
			if first {
				p[4] = 0; // pointer_field
				at = 5;
			}
			let n = rest.len().min(TS_SIZE - at);
			p[at..at + n].copy_from_slice(&rest[..n]);
			rest = &rest[n..];
			first = false;
			self.put(&p)?;
		}
		Ok(())
	}

	fn counter(&mut self, of: CounterOf) -> u8 {
		let c = match of {
			CounterOf::Pat => &mut self.pat_cc,
			CounterOf::Pmt => &mut self.pmt_cc,
			CounterOf::Sdt => &mut self.sdt_cc,
		};
		let value = *c;
		*c = (*c + 1) & 0x0F;
		value
	}

	/// Queues one transport packet; the output gets them in blocks.
	fn put(&mut self, packet: &[u8; TS_SIZE]) -> Result<()> {
		self.buffer.extend_from_slice(packet);
		self.written += TS_SIZE as u64;
		if self.buffer.len() >= WRITE_BLOCK {
			self.drain()?;
		}
		Ok(())
	}

	fn drain(&mut self) -> Result<()> {
		if !self.buffer.is_empty() {
			self.io.write_all(&self.buffer)?;
			self.buffer.clear();
		}
		Ok(())
	}

	fn write_pat_pmt(&mut self) -> Result<()> {
		let o = &self.options;
		let mut body = Vec::new();
		body.extend_from_slice(&o.service_id.to_be_bytes());
		body.extend_from_slice(&(0xE000 | o.pmt_pid).to_be_bytes());
		let pat = section(0x00, o.transport_stream_id, &body);
		self.write_section(PID_PAT, &pat, CounterOf::Pat)?;

		let pcr_pid = self.pcr.map_or(0x1FFF, |i| self.streams[i].pid);
		let mut body = Vec::new();
		body.extend_from_slice(&(0xE000 | pcr_pid).to_be_bytes());
		body.extend_from_slice(&0xF000u16.to_be_bytes()); // no program descriptors
		for s in &self.streams {
			body.push(s.stream_type);
			body.extend_from_slice(&(0xE000 | s.pid).to_be_bytes());
			body.extend_from_slice(&(0xF000 | s.descriptors.len() as u16).to_be_bytes());
			body.extend_from_slice(&s.descriptors);
		}
		if body.len() > 1000 {
			return Err(Message::Container("mpegts muxer: PMT too large"));
		}
		let pmt = section(0x02, self.options.service_id, &body);
		let pid = self.options.pmt_pid;
		self.write_section(pid, &pmt, CounterOf::Pmt)
	}

	fn write_sdt(&mut self) -> Result<()> {
		let tag = |key: &str, default: &str| match self.metadata.raw.get(key) {
			Some(RawValue::String(v)) if !v.is_empty() => v.clone(),
			_ => default.to_string(),
		};
		let provider = dvb_string(&tag("service_provider", "RMP"));
		let name = dvb_string(&tag("service_name", "Service01"));
		let o = &self.options;
		let mut descriptor = vec![0x48, 0, 0x01]; // service_descriptor, digital television
		descriptor.push(provider.len() as u8);
		descriptor.extend_from_slice(&provider);
		descriptor.push(name.len() as u8);
		descriptor.extend_from_slice(&name);
		descriptor[1] = (descriptor.len() - 2) as u8;
		let mut body = Vec::new();
		body.extend_from_slice(&o.original_network_id.to_be_bytes());
		body.push(0xFF);
		body.extend_from_slice(&o.service_id.to_be_bytes());
		body.push(0xFC); // no EIT
		// running_status 4 (running), free_CA_mode 0, descriptors length.
		body.extend_from_slice(&(0x8000 | descriptor.len() as u16).to_be_bytes());
		body.extend_from_slice(&descriptor);
		let sdt = section(0x42, o.transport_stream_id, &body);
		self.write_section(PID_SDT, &sdt, CounterOf::Sdt)
	}

	/// PAT/PMT and SDT when they are due (`dts` in output ticks), or forced.
	fn tables(&mut self, dts: i64, video_key: bool) -> Result<()> {
		let pat_due = self.force_tables
			|| video_key
			|| self.last_pat.is_none_or(|t| dts - t >= (self.options.pat_period * 90_000.0) as i64);
		let sdt_due = self.force_tables
			|| self.last_sdt.is_none_or(|t| dts - t >= (self.options.sdt_period * 90_000.0) as i64);
		if sdt_due {
			self.write_sdt()?;
			self.last_sdt = Some(dts);
		}
		if pat_due {
			self.write_pat_pmt()?;
			self.last_pat = Some(dts);
		}
		self.force_tables = false;
		Ok(())
	}

	// --- PES --------------------------------------------------------------

	/// Writes one PES as transport packets.
	fn write_pes(
		&mut self,
		index: usize,
		data: &[u8],
		pts: Option<i64>,
		dts: Option<i64>,
		key: bool,
	) -> Result<()> {
		let reference = dts.or(pts).unwrap_or(0);
		let video = self.streams[index].kind == StreamKind::Video;
		self.tables(reference, video && key)?;

		let s = &self.streams[index];
		let mut header = vec![0, 0, 1, s.pes_id, 0, 0];
		debug_assert!(!data.is_empty() || pts.is_some());
		let both = matches!((pts, dts), (Some(p), Some(d)) if p != d);
		let flags2 = match (pts.is_some(), both) {
			(true, true) => 0xC0,
			(true, false) => 0x80,
			_ => 0,
		};
		header.push(0x80); // '10', not scrambled, no alignment flag
		header.push(flags2);
		header.push(if both {
			10
		} else if pts.is_some() {
			5
		} else {
			0
		});
		if let Some(p) = pts {
			put_timestamp(&mut header, if both { 3 } else { 2 }, p);
		}
		if both && let Some(d) = dts {
			put_timestamp(&mut header, 1, d);
		}
		let length = header.len() - 6 + data.len();
		// Video PES have length 0 (unbounded), as FFmpeg's
		// `omit_video_pes_length`; so do those too long for 16 bits.
		let length = if video || length > 0xFFFF { 0 } else { length };
		header[4..6].copy_from_slice(&(length as u16).to_be_bytes());

		let pcr = (Some(index) == self.pcr).then(|| (reference - self.delay()).max(0) as u64 * 300);
		let pid = s.pid;
		let mut payload = header;
		payload.extend_from_slice(data);
		let mut rest = payload.as_slice();
		let mut first = true;
		while !rest.is_empty() {
			let mut p = [0xFFu8; TS_SIZE];
			let s = &mut self.streams[index];
			let cc = s.continuity;
			s.continuity = (s.continuity + 1) & 0x0F;
			p[0] = SYNC;
			p[1] = (u8::from(first) << 6) | (pid >> 8) as u8;
			p[2] = pid as u8;
			// Adaptation field: PCR and the random access flag on the first
			// packet, stuffing on the last.
			let mut af: Vec<u8> = Vec::new();
			let mut af_flags = 0u8;
			if first && key {
				af_flags |= 0x40;
			}
			let pcr_here = if first { pcr } else { None };
			if pcr_here.is_some() {
				af_flags |= 0x10;
			}
			if af_flags != 0 {
				af.push(af_flags);
				if let Some(pcr) = pcr_here {
					let base = (pcr / 300) & ((1 << 33) - 1);
					let ext = pcr % 300;
					af.extend_from_slice(&[
						(base >> 25) as u8,
						(base >> 17) as u8,
						(base >> 9) as u8,
						(base >> 1) as u8,
						(((base & 1) << 7) as u8) | 0x7E | ((ext >> 8) as u8 & 1),
						ext as u8,
					]);
				}
			}
			// The adaptation field takes what the payload leaves: none when
			// a full 184 bytes of payload remain and no flag is needed.
			let room = TS_SIZE - 4 - if af.is_empty() { 0 } else { 1 + af.len() };
			let n = rest.len().min(room);
			let mut at = 4;
			if n == TS_SIZE - 4 {
				p[3] = 0x10 | cc;
			} else {
				let total = TS_SIZE - 4 - n; // adaptation field bytes, length byte included
				p[3] = 0x30 | cc;
				p[4] = (total - 1) as u8;
				if total > 1 {
					if af.is_empty() {
						af.push(0); // no flags
					}
					af.resize(total - 1, 0xFF);
					p[5..5 + af.len()].copy_from_slice(&af);
				}
				at = 4 + total;
			}
			p[at..at + n].copy_from_slice(&rest[..n]);
			debug_assert_eq!(at + n, TS_SIZE);
			rest = &rest[n..];
			first = false;
			self.put(&p)?;
		}
		Ok(())
	}

	fn flush_pending(&mut self, index: usize) -> Result<()> {
		let s = &mut self.streams[index];
		if s.pending.is_empty() {
			return Ok(());
		}
		let data = std::mem::take(&mut s.pending);
		let (pts, dts) = (s.pending_pts.take(), s.pending_dts.take());
		s.pending_samples = 0;
		self.write_pes(index, &data, pts, dts, true)
	}

	/// Queues or writes one payload in output ticks.
	fn emit(
		&mut self,
		index: usize,
		data: Vec<u8>,
		pts: Option<i64>,
		dts: Option<i64>,
		key: bool,
		samples: u32,
	) -> Result<()> {
		let s = &self.streams[index];
		if s.kind != StreamKind::Audio {
			return self.write_pes(index, &data, pts, dts, key);
		}
		let limit = pes_payload_limit(self.options.pes_payload_size);
		// An audio-only PCR stream needs a PCR at least every 100 ms.
		let max_delay = if Some(index) == self.pcr { 9000 } else { self.delay() };
		let t = dts.or(pts);
		let due = !s.pending.is_empty()
			&& (s.pending.len() + data.len() > limit
				|| matches!((t, s.pending_dts), (Some(a), Some(b)) if a - b >= max_delay)
				|| s.pending_samples + samples > 5760);
		if due {
			self.flush_pending(index)?;
		}
		if data.len() > limit {
			return self.write_pes(index, &data, pts, dts, key);
		}
		let s = &mut self.streams[index];
		if s.pending.is_empty() {
			s.pending_pts = pts;
			s.pending_dts = dts.or(pts);
		}
		s.pending.extend_from_slice(&data);
		s.pending_samples += samples;
		Ok(())
	}

	/// Output ticks of a packet time: rescaled to 90 kHz, moved so the
	/// first packet starts at the start offset.
	fn ticks(&mut self, time: Option<Time>) -> Result<Option<i64>> {
		let Some(t) = time else { return Ok(None) };
		let t = t
			.rescale(TS_TIME_BASE)
			.ok_or(Message::Container("mpegts muxer: invalid packet timestamp"))?
			.ticks() as i64;
		let origin = *self.origin.get_or_insert(t);
		Ok(Some((t - origin + self.start_offset() as i64).max(0)))
	}

	fn write_converted(&mut self, index: usize, packet: Packet) -> Result<()> {
		// The origin is a decode time: the first packet's dts.
		self.ticks(packet.dts.or(packet.pts))?;
		let pts = self.ticks(packet.pts)?;
		let shift = self.shift_90k(&self.streams[index]);
		let dts = self.ticks(packet.dts)?.map(|d| {
			let d = d - shift;
			let d = pts.map_or(d, |p| d.min(p));
			self.streams[index].last_dts.map_or(d, |l| d.max(l))
		});
		let dts = dts.or(pts);
		let key = packet.keyframe;
		let s = &mut self.streams[index];
		if let (Some(d), Some(last)) = (dts, s.last_dts)
			&& d < last
		{
			return Err(Message::Container("mpegts muxer: decode times go backwards"));
		}
		s.last_dts = dts.or(s.last_dts);
		let data = match &mut s.payload {
			Payload::Copy => packet.data,
			Payload::Video { hevc, length_size, parameter_sets } => {
				video_annex_b(*hevc, *length_size, parameter_sets, &packet.data, key)
			}
			Payload::Adts(cfg) => {
				let looks_adts =
					packet.data.len() >= 7 && packet.data[0] == 0xFF && packet.data[1] & 0xF6 == 0xF0;
				if looks_adts { packet.data } else { adts(*cfg, &packet.data)? }
			}
			Payload::Opus { .. } => {
				// Opus: hold one packet, so the last can carry the end trim.
				let samples = packet
					.duration
					.and_then(|d| d.rescale(crate::core::TimeBase::new(1, 48_000)))
					.map(|d| d.ticks());
				let previous = s.opus_held.replace((packet.data, pts, dts, samples));
				if let Some(previous) = previous {
					self.write_opus(index, previous, false)?;
				}
				return Ok(());
			}
		};
		let data = self.encrypt(index, data);
		self.emit(index, data, pts, dts, key, 0)
	}

	fn encrypt(&self, index: usize, data: Vec<u8>) -> Vec<u8> {
		match (&self.cipher, self.streams[index].encrypted) {
			(Some(cipher), Some(codec)) => encrypt_packet(cipher, codec, &data),
			_ => data,
		}
	}

	fn write_opus(
		&mut self,
		index: usize,
		(data, pts, dts, duration): HeldOpus,
		last: bool,
	) -> Result<()> {
		let samples = packet_samples(&data).unwrap_or(960);
		let s = &mut self.streams[index];
		let Payload::Opus { pending_start_trim, end_trim } = &mut s.payload else { return Ok(()) };
		// Already framed (from a TS source): as it is.
		if es::opus_control_header(&data).is_some_and(|u| u.header + u.size == data.len()) {
			return self.emit(index, data, pts, dts, true, samples);
		}
		let start = (*pending_start_trim).min(samples);
		*pending_start_trim -= start;
		// The end trim: the `encoder_padding` tag, or a last packet that
		// lasts less than its samples (how RMP's demuxers report it).
		let short = duration.map_or(0, |d| u64::from(samples).saturating_sub(d) as u32);
		let end = if last { (*end_trim).max(short).min(samples - start) } else { 0 };
		let mut out = Vec::with_capacity(data.len() + 8);
		es::write_opus_control_header(&mut out, data.len(), start as u16, end as u16);
		out.extend_from_slice(&data);
		self.emit(index, out, pts, dts, true, samples)
	}

	fn release_opus(&mut self, index: usize, last: bool) -> Result<()> {
		if let Some(held) = self.streams[index].opus_held.take() {
			self.write_opus(index, held, last)?;
		}
		Ok(())
	}

	/// Video packets without decode times (Matroska stores only
	/// presentation times): the k-th decode time is the (k - d)-th smallest
	/// presentation time, where d is the deepest reordering seen in the
	/// first window of [`REORDER_WINDOW`] frames, so no frame is decoded
	/// after it is shown and decode times keep increasing. The first d
	/// frames are spaced back from the first time by one frame step.
	fn release_reorder(&mut self, index: usize, all: bool) -> Result<()> {
		loop {
			let s = &mut self.streams[index];
			if s.reorder.is_empty() || (!all && s.reorder.len() <= REORDER_WINDOW) {
				return Ok(());
			}
			let pts_of = |p: &Packet| p.pts.map_or(0, |t| t.ticks() as i64);
			let delay = *s.reorder_delay.get_or_insert_with(|| {
				let pts: Vec<i64> = s.reorder.iter().map(pts_of).collect();
				reorder_depth(&pts).0
			});
			let mut packet = s.reorder.remove(0);
			let pts = pts_of(&packet);
			let k = s.reordered as usize;
			s.reordered += 1;
			let dts = if k >= delay && !s.pts_window.is_empty() {
				s.pts_window.remove(0)
			} else {
				let first = s.pts_window.first().copied().unwrap_or(pts);
				let step = match s.pts_window.as_slice() {
					[a, b, ..] => b - a,
					_ => 0,
				};
				first - step * (delay - k.min(delay)) as i64
			};
			// Stored `bias` (the stream's shift) higher, so never negative;
			// still increasing and not after the presentation time.
			let bias = s.dts_shift.unwrap_or(0);
			let floor = s.last_reordered.map_or(i64::MIN, |l| l + 1);
			let dts = (dts.min(pts) + bias).max(floor).max(0);
			s.last_reordered = Some(dts);
			packet.dts = Some(Time::new(dts as u64, s.time_base));
			self.write_converted(index, packet)?;
		}
	}

	/// Whether every video stream's decode time shift can be decided from
	/// the start-up packets: [`REORDER_WINDOW`] frames each, or a frame
	/// without dts (those streams get derived decode times instead).
	fn startup_decided(&self) -> bool {
		let Some(queue) = &self.startup else { return true };
		self
			.streams
			.iter()
			.enumerate()
			.filter(|(_, s)| s.kind == StreamKind::Video)
			.all(|(i, _)| queue.iter().filter(|(j, _)| *j == i).count() > REORDER_WINDOW)
	}

	/// Decides the shifts and the origin (the earliest shifted decode time,
	/// so the output starts at the start offset as FFmpeg's does), then
	/// writes the start-up packets.
	fn release_startup(&mut self) -> Result<()> {
		let Some(queue) = self.startup.take() else { return Ok(()) };
		for (i, s) in self.streams.iter_mut().enumerate() {
			if s.kind != StreamKind::Video {
				s.dts_shift = Some(0);
				continue;
			}
			let mine: Vec<&Packet> = queue.iter().filter(|(j, _)| *j == i).map(|(_, p)| p).collect();
			let shift = if mine.iter().any(|p| p.dts.is_none()) {
				// Derived decode times can start below zero: they are kept
				// `bias` higher (see `release_reorder`) and the shift takes
				// it back off.
				let pts: Vec<i64> = mine.iter().filter_map(|p| p.pts.map(|t| t.ticks() as i64)).collect();
				let (delay, first) = reorder_depth(&pts);
				s.reorder_delay = Some(delay);
				let start = pts.iter().copied().min().unwrap_or(0);
				start - first
			} else {
				mine
					.iter()
					.filter_map(|p| Some(p.dts?.ticks() as i64 - p.pts?.ticks() as i64))
					.max()
					.unwrap_or(0)
					.max(0)
			};
			s.dts_shift = Some(shift);
		}
		let origin = queue
			.iter()
			.filter_map(|(i, p)| {
				let s = &self.streams[*i];
				// A pts-only stream's first (biased) decode time is its
				// smallest presentation time.
				let t = match p.dts {
					Some(d) => d,
					None => queue.iter().filter(|(j, _)| j == i).filter_map(|(_, q)| q.pts).min()?,
				};
				Some(t.rescale(TS_TIME_BASE)?.ticks() as i64 - self.shift_90k(s))
			})
			.min();
		if self.origin.is_none() {
			self.origin = origin;
		}
		for (index, packet) in queue {
			self.route(index, packet)?;
		}
		Ok(())
	}

	fn shift_90k(&self, s: &TsStream) -> i64 {
		let ticks = s.dts_shift.unwrap_or(0).max(0) as u64;
		Time::new(ticks, s.time_base).rescale(TS_TIME_BASE).map_or(0, |t| t.ticks() as i64)
	}

	/// After start-up: pts-only video through the decode time derivation,
	/// everything else straight on.
	fn route(&mut self, index: usize, packet: Packet) -> Result<()> {
		let s = &mut self.streams[index];
		if (s.kind == StreamKind::Video && packet.dts.is_none() && packet.pts.is_some())
			|| !s.reorder.is_empty()
		{
			if let Some(t) = packet.pts.and_then(|t| t.rescale(s.time_base)) {
				let t = t.ticks() as i64;
				let at = s.pts_window.partition_point(|x| *x < t);
				s.pts_window.insert(at, t);
			}
			s.reorder.push(packet);
			return self.release_reorder(index, false);
		}
		self.write_converted(index, packet)
	}

	fn start(&mut self) -> Result<()> {
		self.pcr = self
			.streams
			.iter()
			.position(|s| s.kind == StreamKind::Video)
			.or_else(|| self.streams.iter().position(|s| s.kind == StreamKind::Audio))
			.or(if self.streams.is_empty() { None } else { Some(0) });
		self.started = true;
		Ok(())
	}
}

/// Codecs the muxer writes.
pub fn supports(codec: &str) -> bool {
	matches!(
		codec,
		"h264"
			| "hevc"
			| "mpeg2video"
			| "mpeg4"
			| "aac"
			| "mp1"
			| "mp2"
			| "mp3"
			| "ac3"
			| "eac3"
			| "opus"
	)
}

#[derive(Clone, Copy)]
enum CounterOf {
	Pat,
	Pmt,
	Sdt,
}

/// How many frames deep presentation times are reordered in `pts` (decode
/// order), and the first decode time derived from them: the first
/// presentation time moved back by that many frame steps.
fn reorder_depth(pts: &[i64]) -> (usize, i64) {
	let mut sorted = pts.to_vec();
	sorted.sort_unstable();
	let delay = pts
		.iter()
		.enumerate()
		.map(|(k, p)| k.saturating_sub(sorted.partition_point(|x| x < p)))
		.max()
		.unwrap_or(0);
	let step = match sorted.as_slice() {
		[a, b, ..] => b - a,
		_ => 0,
	};
	(delay, sorted.first().copied().unwrap_or(0) - step * delay as i64)
}

/// FFmpeg rounds `pes_payload_size` so a PES fills whole TS packets.
fn pes_payload_limit(size: usize) -> usize {
	((size + 14).div_ceil(184) * 184).saturating_sub(14).max(1)
}

/// A long-form PSI section with its CRC.
fn section(table_id: u8, id: u16, body: &[u8]) -> Vec<u8> {
	let length = 5 + body.len() + 4;
	let mut s = vec![table_id, 0xB0 | (length >> 8) as u8, length as u8];
	s.extend_from_slice(&id.to_be_bytes());
	s.push(0xC1); // version 0, current
	s.extend_from_slice(&[0, 0]); // section 0 of 0
	s.extend_from_slice(body);
	let crc = crc32(&s);
	s.extend_from_slice(&crc.to_be_bytes());
	s
}

/// DVB text: Latin-1-compatible ASCII as it is, anything else as UTF-8
/// behind the 0x15 selector (EN 300 468 Annex A), at most 255 bytes.
fn dvb_string(s: &str) -> Vec<u8> {
	let mut out =
		if s.is_ascii() { s.as_bytes().to_vec() } else { [&[0x15][..], s.as_bytes()].concat() };
	out.truncate(255);
	out
}

/// A 33-bit PTS/DTS with its 4-bit prefix and marker bits.
fn put_timestamp(out: &mut Vec<u8>, prefix: u8, t: i64) {
	let t = (t as u64) & ((1 << 33) - 1);
	out.push((prefix << 4) | (((t >> 30) as u8 & 7) << 1) | 1);
	out.extend_from_slice(&((((t >> 15) & 0x7FFF) << 1 | 1) as u16).to_be_bytes());
	out.extend_from_slice(&((((t & 0x7FFF) << 1) | 1) as u16).to_be_bytes());
}

/// An ADTS header (no CRC) in front of a raw AAC frame.
fn adts(cfg: [u8; 3], frame: &[u8]) -> Result<Vec<u8>> {
	let len = frame.len() + 7;
	if len >= 1 << 13 {
		return Err(Message::Container("mpegts muxer: AAC frame too large for ADTS"));
	}
	let [profile, rate, channels] = cfg;
	let mut out = Vec::with_capacity(len);
	out.extend_from_slice(&[
		0xFF,
		0xF1,
		(profile << 6) | (rate << 2) | (channels >> 2),
		((channels & 3) << 6) | (len >> 11) as u8,
		(len >> 3) as u8,
		(((len & 7) << 5) as u8) | 0x1F,
		0xFC,
	]);
	out.extend_from_slice(frame);
	Ok(out)
}

/// An H.264 / HEVC access unit as TS carries it: Annex B with an access
/// unit delimiter first and, on a keyframe without them, the parameter
/// sets right after it. Annex B input is otherwise kept byte for byte (as
/// FFmpeg's `mpegtsenc` does); length-prefixed input gets 4-byte start
/// codes on its first NAL unit and parameter sets and 3-byte ones on the
/// others (FFmpeg's `h264_mp4toannexb`).
fn video_annex_b(
	hevc: bool,
	length_size: Option<usize>,
	parameter_sets: &[u8],
	data: &[u8],
	key: bool,
) -> Vec<u8> {
	let nals: Vec<&[u8]> = match length_size {
		Some(size) => split_length_prefixed(data, size),
		None => split_annex_b(data),
	};
	let nal_type = |n: &[u8]| if hevc { (n[0] >> 1) & 0x3F } else { n[0] & 0x1F };
	let (aud_type, sps_type) = if hevc { (35, 33) } else { (9, 7) };
	let has_aud = nals.first().is_some_and(|n| nal_type(n) == aud_type);
	let needs_sets =
		key && !parameter_sets.is_empty() && !nals.iter().any(|n| nal_type(n) == sps_type);
	let mut out = Vec::with_capacity(data.len() + parameter_sets.len() + 16);
	if !has_aud {
		out.extend_from_slice(if hevc {
			&[0, 0, 0, 1, 0x46, 0x01, 0x50]
		} else {
			&[0, 0, 0, 1, 0x09, 0xF0]
		});
	}
	match length_size {
		// FFmpeg's `mpegtsenc` adds the parameter sets only together with
		// a delimiter it inserts.
		None => {
			if needs_sets && !has_aud {
				out.extend_from_slice(parameter_sets);
			}
			out.extend_from_slice(data);
		}
		Some(_) => {
			// The sets go right before the first IDR (HEVC: IRAP) slice, or
			// the first slice of a keyframe that has none.
			let is_target = |t: u8| if hevc { (16..=23).contains(&t) } else { t == 5 };
			let is_slice = |t: u8| if hevc { t < 32 } else { (1..=5).contains(&t) };
			let target = nals
				.iter()
				.position(|n| is_target(nal_type(n)))
				.or_else(|| nals.iter().position(|n| is_slice(nal_type(n))));
			let mut written = false;
			for (i, nal) in nals.iter().enumerate() {
				if needs_sets && Some(i) == target {
					out.extend_from_slice(parameter_sets);
					written = true;
				}
				let t = nal_type(nal);
				// h264_mp4toannexb: 4 bytes for the first unit it writes and
				// for parameter sets, 3 for the others; HEVC: always 4.
				let long = hevc || !written || t == 7 || t == 8;
				out.extend_from_slice(if long { &[0, 0, 0, 1] } else { &[0, 0, 1] });
				out.extend_from_slice(nal);
				written = true;
			}
		}
	}
	out
}

impl Muxer for TsMuxer {
	fn set_metadata(&mut self, metadata: Metadata) -> Result<()> {
		self.metadata = metadata;
		Ok(())
	}

	fn metadata(&self) -> &Metadata {
		&self.metadata
	}

	fn set_option(&mut self, key: &str, value: &str) -> Result<()> {
		if self.started {
			return Err(Message::Container("mpegts muxer: options must be set before writing"));
		}
		match self.options.set(key, value)? {
			true => Ok(()),
			false => Err(Message::Container("mpegts muxer: unknown option")),
		}
	}

	fn add(&mut self, stream: &Stream) -> Result<usize> {
		if self.started {
			return Err(Message::Container("mpegts muxer: streams must be added before writing"));
		}
		if self.streams.iter().any(|s| s.id == stream.id) {
			return Err(Message::Container("mpegts muxer: duplicate stream id"));
		}
		if !matches!(stream.kind, StreamKind::Audio | StreamKind::Video) {
			return Err(Message::Container("mpegts muxer supports only audio and video streams"));
		}
		let s = self.stream_setup(stream, self.streams.len())?;
		self.streams.push(s);
		Ok(self.streams.len() - 1)
	}

	fn write(&mut self, packet: Packet) -> Result<()> {
		if !self.started {
			self.start()?;
		}
		let index = self
			.streams
			.iter()
			.position(|s| s.id == packet.stream_id)
			.ok_or(Message::Container("mpegts muxer: packet for unknown stream"))?;
		if let Some(queue) = self.startup.as_mut() {
			queue.push((index, packet));
			if self.startup_decided() {
				self.release_startup()?;
			}
			return Ok(());
		}
		self.route(index, packet)
	}

	fn finish(&mut self) -> Result<()> {
		if !self.started {
			self.start()?;
		}
		self.release_startup()?;
		for i in 0..self.streams.len() {
			self.release_reorder(i, true)?;
			self.release_opus(i, true)?;
			self.flush_pending(i)?;
		}
		if self.written == 0 {
			// No packet at all: still a valid (empty) program.
			self.tables(0, false)?;
		}
		self.drain()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn timestamps_and_payload_limit() {
		let mut out = Vec::new();
		put_timestamp(&mut out, 2, 126_000);
		let parsed = super::super::packet::parse_pes(
			&[[0, 0, 1, 0xE0, 0, 0, 0x80, 0x80, 5].as_slice(), &out].concat(),
		)
		.unwrap();
		assert_eq!(parsed.pts, Some(126_000));
		assert_eq!(pes_payload_limit(2930), 2930);
	}

	#[test]
	fn adts_header_round_trips() {
		let frame = adts([1, 3, 2], &[0u8; 100]).unwrap();
		let h = crate::codecs::aac::AdtsHeader::parse(&frame).unwrap();
		assert_eq!(
			(h.object_type, h.sample_rate_index, h.channel_config, h.frame_length),
			(2, 3, 2, 107)
		);
	}

	#[test]
	fn video_gets_delimiter_and_parameter_sets() {
		let sets = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3];
		let au = [0, 0, 0, 3, 0x65, 9, 9];
		let out = video_annex_b(false, Some(4), &sets, &au, true);
		assert_eq!(out, [&[0, 0, 0, 1, 0x09, 0xF0][..], &sets, &[0, 0, 1, 0x65, 9, 9]].concat());
		// Annex B input with its own delimiter is kept as it is.
		let annex_b = [0, 0, 0, 1, 0x09, 0xF0, 0, 0, 1, 0x41, 1];
		assert_eq!(video_annex_b(false, None, &sets, &annex_b, false), annex_b);
		let p = video_annex_b(false, Some(4), &sets, &[0, 0, 0, 2, 0x41, 1], false);
		assert_eq!(p, [0, 0, 0, 1, 0x09, 0xF0, 0, 0, 0, 1, 0x41, 1]);
	}
}
